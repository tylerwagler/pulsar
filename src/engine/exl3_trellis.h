/**
 * @file exl3_trellis.h
 * @brief EXL3 trellis tile arithmetic: the tile bit layout, the mul1 codebook
 *        and the 128-block Hadamard basis (L245).
 *
 * Ported from exllamav3 (MIT, turboderp-org/exllamav3 @ 6b84a21):
 * `quant/pack.cu` (the packer, which fixes the bit order), `quant/exl3_dq.cuh`
 * (the state windows, integer and half-integer K), `quant/codebook.cuh`
 * (`decode_3inst<2>`, the mul1 codebook) and `exl3_lib/quantize.py`
 * (`tensor_core_perm`, the position map).  This header is the ONE authority
 * for those facts in pulsar -- and for the container's per-expert byte model
 * (exl3_expert_layout): the loader, the expert-address table, the lane that
 * transcodes an EXL3 checkpoint and the dequant gate all read it, and the
 * device kernels are graded against it.
 *
 * The format, in one paragraph.  A quantized linear W (in, out) is stored as
 * 16x16 tiles, tile (kt, nt) covering W[16kt..16kt+16, 16nt..16nt+16], each
 * tile an independent bit stream of 256 * K bits (K = bits per weight; the
 * half-integer rates 1.5/2.5/3.5 spend 2K+1 bits per PAIR of positions).  The
 * stream is stored as 16 * K uint16 words (16 * K + 8 for a half-integer K)
 * with the two halves of every uint32 swapped, so that stream bit s lives in
 * uint32 s/32 at bit 31 - (s % 32).  Position p (0..255, tensor-core order)
 * holds a 16-bit STATE = the 16 stream bits ending at end(p), wrapping mod the
 * stream length -- a tail-biting ring -- and the weight is the mul1 codebook's
 * value of that state, one fp16.  Two per-channel fp16 scale vectors, suh (in)
 * and svh (out), and a 128-block Hadamard on both sides map W_hat back to the
 * original basis: W = diag(suh) H128 W_hat H128 diag(svh) (1/sqrt(128) per
 * side).
 */
#ifndef PULSAR_EXL3_TRELLIS_H
#define PULSAR_EXL3_TRELLIS_H

#include <stdint.h>
#include <string.h>

#include "pulsar_gpu.h" /* the PULSAR_TENSOR_* ids */

/* The bit-layout functions are callable from device code so the GEMV reads
 * the SAME authority for "where does position p's state end" as the host
 * dequant and the transcoder; nothing about the layout is restated in a
 * kernel. */
#ifdef __CUDACC__
/** `__host__ __device__` under nvcc, nothing in a host TU: marks the layout helpers callable from both. */
#define EXL3_HD __host__ __device__
#else
/** Host-only build: EXL3_HD expands to nothing. */
#define EXL3_HD
#endif

/* The header is included by host TUs and by CUDA TUs (the expert-table gate,
 * the kernels' launchers), so the fp16 conversions are software and exact --
 * no _Float16, no NEON, no engine header.  The dequant gate holds them to
 * exllamav3's own codebook values, all 65536 of them. */

/** IEEE binary16 bits -> float, exact (subnormals, Inf, NaN included). */
/// @param h binary16 bit pattern
/// @return the same value as a float
static inline float exl3_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu, bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/** float -> IEEE binary16 bits, round to nearest even (the rounding every
 *  fp16 instruction applies), overflow to Inf, subnormals exact. */
/// @param f value to convert
/// @return its binary16 bit pattern (NaN stays a quiet NaN, sign kept)
static inline uint16_t exl3_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t e = (x >> 23) & 0xffu;
    uint32_t mant = x & 0x7fffffu;
    if (e == 0xffu) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    const int exp = (int)e - 127 + 15;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u), mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1u))) half++;
        return (uint16_t)(sign | half);
    }
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half++; /* a carry rolls into the exponent correctly */
    return (uint16_t)half;
}

/** Weights per tile (16 x 16). */
#define EXL3_TILE_WEIGHTS 256
/** Hadamard block: both dims of an EXL3 tensor are multiples of this. */
#define EXL3_HAD_BLOCK 128

/** The mul1 codebook multiplier; the checkpoint stores it as the `.mul1`
 *  marker's int32 value (2212286765) but never reads it back. */
#define EXL3_MUL1_MULTIPLIER 0x83DCD12Du

/**
 * Bit rate in half-bit units: k2 = 2K for an integer K in 1..8, 2K+1 for the
 * half-integer rates 1.5, 2.5, 3.5.  Anything else is not an EXL3 rate.
 */
/// @param k2 candidate rate in half-bit units
/// @return true for k2 in {2,4,..,16} or {3,5,7}
EXL3_HD static inline bool exl3_k2_valid(int k2) {
    const int bits = k2 >> 1;
    if (k2 & 1) return bits >= 1 && bits <= 3;
    return bits >= 1 && bits <= 8;
}

/** uint16 words per 256-weight tile: 16K, or 16K + 8 for a half-integer K. */
/// @param k2 rate in half-bit units (exl3_k2_valid)
/// @return uint16 words in one tile's stream
EXL3_HD static inline int exl3_words_per_tile(int k2) {
    return 16 * (k2 >> 1) + ((k2 & 1) ? 8 : 0);
}

/**
 * The rate a tile width encodes, or 0 when no EXL3 rate stores that many
 * words.  K is DERIVED from the trellis' last dimension; nothing else in the
 * checkpoint records it.  (16K + 8 for K=2 is 40 = 16 * 2.5: the half-integer
 * widths collide with nothing because 16K + 8 is never a multiple of 16.)
 */
/// @param words the trellis' last dimension: uint16 words per tile
/// @return the rate in half-bit units, or 0 if no valid rate stores `words`
EXL3_HD static inline int exl3_k2_from_words(int words) {
    if (words <= 0 || words % 8) return 0;
    const int k2 = (words % 16 == 0) ? 2 * (words / 16) : 2 * (words / 16) + 1;
    return exl3_k2_valid(k2) ? k2 : 0;
}

/** Stream bits per tile. */
/// @param k2 rate in half-bit units
/// @return bits in one tile's stream (16 * exl3_words_per_tile)
EXL3_HD static inline int exl3_stream_bits(int k2) {
    return 16 * exl3_words_per_tile(k2);
}

/**
 * Stream bit index (exclusive) at which position p's 16-bit state ends.
 * Integer K: (p+1)K.  Half-integer K: a pair (2j, 2j+1) spans 2K+1 bits, the
 * even position ends K bits in, the odd one at the pair's end
 * (`dq8_half`: w6 = w7 >> (K+1), w5 = w6 >> K).
 */
/// @param k2 rate in half-bit units
/// @param p tile position, 0..255 (tensor-core order)
/// @return the exclusive end bit, not yet wrapped mod exl3_stream_bits
EXL3_HD static constexpr inline int exl3_state_end_bit(int k2, int p) {
    const int bits = k2 >> 1;
    if (!(k2 & 1)) return (p + 1) * bits;
    const int bits2 = 2 * bits + 1;
    return (p >> 1) * bits2 + ((p & 1) ? bits2 : bits);
}

/**
 * The 16-bit state of position p in one tile.  `tile` points at the tile's
 * words as stored (uint16, little-endian, halves of each uint32 swapped by the
 * packer -- reading the pair as one little-endian uint32 undoes the swap).
 */
/// @param tile the tile's exl3_words_per_tile(k2) words
/// @param k2 rate in half-bit units
/// @param p tile position, 0..255
/// @return the state, in the low 16 bits
static inline uint32_t exl3_tile_state(const uint16_t *tile, int k2, int p) {
    const int total = exl3_stream_bits(k2);
    const int end = exl3_state_end_bit(k2, p);
    uint32_t v = 0;
    for (int b = 0; b < 16; b++) {
        int s = end - 16 + b;
        s %= total;
        if (s < 0) s += total;
        const uint32_t w = (uint32_t)tile[2 * (s >> 5)] | ((uint32_t)tile[2 * (s >> 5) + 1] << 16);
        v = (v << 1) | ((w >> (31 - (s & 31))) & 1u);
    }
    return v;
}

/**
 * The mul1 codebook: the fp16 value (as bits) of a 16-bit state.
 * y = x * 0x83DCD12D mod 2^32; s = 1024 + bytesum(y), read as an fp16 bit
 * pattern (0x6400 + n is exactly 1024 + n for n < 1024); v = fma(s, 1/147.7,
 * -10.39) in fp16 with ONE rounding.  The product s * 0x1eee has 22
 * significant bits and the sum with 0xc931 at most 22, so both are exact in
 * fp32 and the single rounding is the final conversion.  Range +-3.45,
 * quantum 0.00677.
 */
/// @param state trellis state (only the low 16 bits are read)
/// @return the weight's fp16 bit pattern
static inline uint16_t exl3_mul1_decode(uint32_t state) {
    const uint32_t y = (state & 0xffffu) * EXL3_MUL1_MULTIPLIER;
    const uint32_t s = 0x6400u + (y & 0xffu) + ((y >> 8) & 0xffu) + ((y >> 16) & 0xffu) + (y >> 24);
    const float h = exl3_f16_to_f32((uint16_t)s);
    const float v = h * exl3_f16_to_f32(0x1eee) + exl3_f16_to_f32(0xc931);
    return exl3_f32_to_f16(v);
}

/**
 * Tensor-core position map: position p of a tile is weight (row, col) of the
 * 16x16 tile, row along `in`, col along `out`.  Lane t = p >> 3, i = p & 7:
 * row = (t % 4) * 2 + {0, 1, 8, 9}[i & 3], col = t / 4 + 8 * (i >> 2).
 */
/// @param p tile position, 0..255
/// @param[out] row 0..15, along `in`
/// @param[out] col 0..15, along `out`
EXL3_HD static inline void exl3_tile_position(int p, int *row, int *col) {
    const int t = p >> 3, i = p & 7;
    *row = (t % 4) * 2 + (i & 1) + 8 * ((i >> 1) & 1); /* {0, 1, 8, 9}[i & 3] */
    *col = t / 4 + 8 * (i >> 2);
}

/**
 * Dequantize one tile to its 256 fp16 weights (bits), row-major 16x16 in the
 * rotated basis (W_hat).  Byte-exact with exllamav3's `reconstruct`.
 */
/// @param tile the tile's words as stored
/// @param k2 rate in half-bit units
/// @param[out] out 256 fp16 bit patterns, out[row * 16 + col]
static inline void exl3_tile_dequant(const uint16_t *tile, int k2, uint16_t out[EXL3_TILE_WEIGHTS]) {
    for (int p = 0; p < EXL3_TILE_WEIGHTS; p++) {
        int r, c;
        exl3_tile_position(p, &r, &c);
        out[r * 16 + c] = exl3_mul1_decode(exl3_tile_state(tile, k2, p));
    }
}

/**
 * The rate a pulsar tensor type carries, in half-bit units, or 0 when the type
 * is not an EXL3 layout.  One id per rate because the rate is not recoverable
 * from a stack's dims; `m` in the names pins the mul1 codebook.
 */
/// @param type a PULSAR_TENSOR_* id
/// @return the rate in half-bit units, or 0 for a non-EXL3 type
EXL3_HD static inline int exl3_type_k2(uint32_t type) {
    switch (type) {
    case PULSAR_TENSOR_EXL3M_K2:  return 4;
    case PULSAR_TENSOR_EXL3M_K2H: return 5;
    case PULSAR_TENSOR_EXL3M_K3:  return 6;
    case PULSAR_TENSOR_EXL3M_K4:  return 8;
    case PULSAR_TENSOR_EXL3M_K5:  return 10;
    case PULSAR_TENSOR_EXL3M_K6:  return 12;
    case PULSAR_TENSOR_EXL3M_K8:  return 16;
    default:                      return 0;
    }
}

/**
 * The EXL3 arms and the rates each reads -- the ONE table, used by the CUDA
 * arms' dispatch (mmq/exl3_moe_prefill.cu, whose tensor-core arms instantiate them) and
 * by the binders that admit a container's tensors (a rate no arm reads is
 * refused at load, not at first use):
 *   DOWN   a routed down on the fold's pre-rotated mid: 2, 2.5, 3 (DeepSeek /
 *          V4.1) and 4, 5, 6 (Qwen3.8-Flash-Next)
 *   PAIR   split gate / up stacks, each rotating its own input: 2, 2.5, 3 (DeepSeek; Qwen's MTP layer
 *          K3), 4, 6 (turboderp's Qwen trunk, which ships gate and up apart with their own suh) and 5
 *          (Inkling's pair build, L295 F7: the tensor-core arms instantiate every routed rate, so a rate is
 *          this table's line)
 *   GATE_UP_FUSED   one [in -> 2 mid] gate | up slice (our Qwen quant): 4, 5
 *   DENSE  the dense-Linear arm (mmq/ds4_exl3_dense.cuh): 2, 3, 4, 5, 6, 8
 *   MOE_PREFILL     the routed-expert tensor-core arms (prompt and decode), every family
 *          (mmq/exl3_moe_prefill.cu): every rate DOWN / PAIR / GATE_UP_FUSED admit -- 2, 2.5, 3, 4, 5, 6
 * Qwen's rates are turboderp's packs (L266): 4.05 bpw = K4 experts, K6 dense and
 * shared; 6.05 = K6 experts, K8 dense and shared; the MTP layer's experts K3.
 */
enum { EXL3_ARM_DOWN = 0, EXL3_ARM_PAIR = 1, EXL3_ARM_GATE_UP_FUSED = 2, EXL3_ARM_DENSE = 3,
       EXL3_ARM_MOE_PREFILL = 4 };
/** Whether EXL3 arm `arm` reads rate `k2` (the table above).
 * @param arm one of the EXL3_ARM_* ids
 * @param k2 rate in half-bit units
 * @return false for an unknown arm or a rate the arm does not instantiate */
EXL3_HD static constexpr inline bool exl3_arm_has_rate(int arm, int k2) {
    switch (arm) {
    case EXL3_ARM_DOWN:          return k2 == 4 || k2 == 5 || k2 == 6 || k2 == 8 || k2 == 10 || k2 == 12;
    case EXL3_ARM_PAIR:          return k2 == 4 || k2 == 5 || k2 == 6 || k2 == 8 || k2 == 10 || k2 == 12;
    case EXL3_ARM_GATE_UP_FUSED: return k2 == 8 || k2 == 10;
    case EXL3_ARM_DENSE:         return k2 == 4 || k2 == 6 || k2 == 8 || k2 == 10 || k2 == 12 || k2 == 16;
    case EXL3_ARM_MOE_PREFILL:   return k2 == 4 || k2 == 5 || k2 == 6 || k2 == 8 || k2 == 10 || k2 == 12;
    default:                     return false;
    }
}

/**
 * Bytes of one expert-projection (in = k, out = n) at rate k2 in pulsar's
 * container: the trellis plane, (k/16)(n/16) tiles x words x 2 B in
 * exllamav3's (kt, nt, word) order, verbatim; then the scales plane,
 * suh[k] | svh[n] fp16.  Each expert is one self-contained
 * [trellis | scales] slice, so the stride to the next expert is their sum
 * and the scales plane of expert e starts `trellis_bytes` into its slice.
 * Both dims must be multiples of EXL3_HAD_BLOCK -- the quantizer's own
 * precondition, and what keeps every plane 128-byte aligned; anything else is
 * refused (returns false), never padded.
 */
/// @param k input width (multiple of EXL3_HAD_BLOCK)
/// @param n output width (multiple of EXL3_HAD_BLOCK)
/// @param k2 rate in half-bit units
/// @param[out] trellis_bytes bytes of the trellis plane
/// @param[out] scale_bytes bytes of the suh | svh plane, (k + n) * 2
/// @param[out] stride bytes of one expert's whole slice
/// @return false (outputs untouched) on an invalid rate or a dim off the block
EXL3_HD static inline bool exl3_expert_layout(uint64_t k, uint64_t n, int k2,
                                      uint64_t *trellis_bytes, uint64_t *scale_bytes,
                                      uint64_t *stride) {
    if (!exl3_k2_valid(k2) || k == 0 || n == 0 || k % EXL3_HAD_BLOCK || n % EXL3_HAD_BLOCK) return false;
    *trellis_bytes = (k / 16) * (n / 16) * (uint64_t)exl3_words_per_tile(k2) * 2u;
    *scale_bytes = (k + n) * 2u;
    *stride = *trellis_bytes + *scale_bytes;
    return true;
}

/**
 * The output width a dense EXL3 Linear's planes STORE for a logical width n: n
 * rounded up to the Hadamard block.  exllamav3 pads every Linear it quantizes
 * that way (`Linear(pad_to = 128)`: the weight zero-padded before the
 * quantizer, svh 0 on the pad columns), so a head of 201024 outputs ships as a
 * 201088-wide trellis; the logical width is the model's and the pad never
 * leaves the arm (L295 F7).  The input width carries no pad: a Linear's input
 * is the previous layer's output, a multiple of 128 by construction.
 */
/// @param n logical output width
/// @return the stored width, n rounded up to a multiple of EXL3_HAD_BLOCK
EXL3_HD static inline uint64_t exl3_out_stored(uint64_t n) {
    return (n + EXL3_HAD_BLOCK - 1) / EXL3_HAD_BLOCK * EXL3_HAD_BLOCK;
}

/**
 * Bytes of one dense Linear (in = k, logical out = n) at rate k2: the
 * exl3_expert_layout of (k, exl3_out_stored(n)) -- [trellis | suh[k] |
 * svh[stored]] -- for every n >= 1.  A routed expert-projection keeps
 * exl3_expert_layout's 128-multiples (its slices are cut and addressed by the
 * MoE arms at the stored widths); this is the 2-D tensor's model.
 */
/// @param k input width (multiple of EXL3_HAD_BLOCK)
/// @param n logical output width (>= 1)
/// @param k2 rate in half-bit units
/// @param[out] trellis_bytes bytes of the trellis plane
/// @param[out] scale_bytes bytes of suh[k] | svh[stored]
/// @param[out] stride bytes of the whole tensor
/// @return false on n == 0 or whatever exl3_expert_layout refuses
EXL3_HD static inline bool exl3_linear_layout(uint64_t k, uint64_t n, int k2, uint64_t *trellis_bytes,
                                              uint64_t *scale_bytes, uint64_t *stride) {
    return n != 0 && exl3_expert_layout(k, exl3_out_stored(n), k2, trellis_bytes, scale_bytes, stride);
}

/** One strided byte copy: `height` runs of `width` bytes, run i from src + i * src_pitch to dst + i * dst_pitch
 *  (cudaMemcpy2D's shape; a single run when height == 1). */
typedef struct {
    /// byte offset of the first run in the source
    uint64_t src, src_pitch, dst, dst_pitch, width, height;
             ///< runs (1 for a single contiguous copy)
} exl3_copy2d;
/** @var exl3_copy2d::src_pitch
 *  bytes between consecutive runs in the source (0 when height == 1) */
/** @var exl3_copy2d::dst
 *  byte offset of the first run in the destination */
/** @var exl3_copy2d::dst_pitch
 *  bytes between consecutive runs in the destination (0 when height == 1) */
/** @var exl3_copy2d::width
 *  bytes per run */

/** A tensor-parallel cut of one expert-projection (exl3_expert_cut): the source and destination layouts and the
 *  three copies that move the bytes -- the trellis, suh, svh. */
typedef struct {
    /// source expert slice bytes (stride); src_trellis is its trellis-plane bytes
    uint64_t src_stride, src_trellis;   ///< exl3_expert_layout of the whole (k, n)
    /// destination expert slice bytes (stride); dst_trellis is its trellis-plane bytes
    uint64_t dst_stride, dst_trellis;   ///< exl3_expert_layout of the destination shape
    exl3_copy2d copy[3];                ///< trellis, suh, svh
} exl3_cut;

/**
 * The bytes of a range [lo, hi) of one expert-projection (in = k, out = n, rate k2) along one axis -- OUT: output
 * columns (a gate / up's intermediate), IN: input rows (a down's intermediate) -- landing at [at, at + hi - lo) of
 * a destination of extent `ext` along that axis (the other axis whole).  The destination is exl3_expert_layout of
 * its own shape, so a rank's half of every expert (at = 0, ext = hi - lo: L269 W1, the tensor-parallel expert
 * half) and the rows reader's gathered slices (several ranges into one destination: tp_slice.cpp) are the same
 * cut, and the arms read either unchanged.
 *   OUT: per k-tile row the n-tiles [lo/16, hi/16) (one run per k-tile row), suh whole, svh[lo, hi);
 *   IN:  the k-tile rows [lo/16, hi/16) (one run), suh[lo, hi), svh whole.
 * Every bound is 128-aligned, so the Hadamard blocks on both sides stay whole and the cut is bytes copied: the
 * destination's matmul is exactly the whole's columns / its rows' partial.  Returns false (refuses) on an invalid
 * rate, a dim off the 128 block, or a range unaligned, empty or out of bounds.
 */
/// @param k source input width
/// @param n source output width
/// @param k2 rate in half-bit units
/// @param in_axis true: cut input rows (IN); false: output columns (OUT)
/// @param lo first index of the range (128-aligned)
/// @param hi one past the last index (128-aligned, <= the axis extent)
/// @param at where the range lands in the destination along the axis (128-aligned)
/// @param ext destination extent along the axis (128-aligned, >= at + hi - lo)
/// @param[out] c the layouts and copies; partially written on refusal
/// @return true when the cut is valid
static inline bool exl3_expert_cut(uint64_t k, uint64_t n, int k2, bool in_axis, uint64_t lo, uint64_t hi,
                                   uint64_t at, uint64_t ext, exl3_cut *c) {
    uint64_t ss = 0, ds = 0;
    const uint64_t full = in_axis ? k : n, w = hi - lo;
    if (lo >= hi || hi > full || at + w > ext || lo % EXL3_HAD_BLOCK || hi % EXL3_HAD_BLOCK ||
        at % EXL3_HAD_BLOCK || ext % EXL3_HAD_BLOCK)
        return false;
    const uint64_t dk = in_axis ? ext : k, dn = in_axis ? n : ext;
    if (!exl3_expert_layout(k, n, k2, &c->src_trellis, &ss, &c->src_stride) ||
        !exl3_expert_layout(dk, dn, k2, &c->dst_trellis, &ds, &c->dst_stride))
        return false;
    const uint64_t tile = (uint64_t)exl3_words_per_tile(k2) * 2u, st = c->src_trellis, dt = c->dst_trellis;
    if (in_axis) {
        c->copy[0] = {lo / 16u * (n / 16u) * tile, 0, at / 16u * (n / 16u) * tile, 0, w / 16u * (n / 16u) * tile, 1};
        c->copy[1] = {st + lo * 2u, 0, dt + at * 2u, 0, w * 2u, 1};
        c->copy[2] = {st + k * 2u, 0, dt + dk * 2u, 0, n * 2u, 1};
    } else {
        c->copy[0] = {lo / 16u * tile, n / 16u * tile, at / 16u * tile, dn / 16u * tile, w / 16u * tile, k / 16u};
        c->copy[1] = {st, 0, dt, 0, k * 2u, 1};
        c->copy[2] = {st + k * 2u + lo * 2u, 0, dt + k * 2u + at * 2u, 0, w * 2u, 1};
    }
    return true;
}

/** Run a cut on the host: one expert's bytes from `src` (the whole's layout) into `dst` (the destination's). */
/// @param c a cut built by exl3_expert_cut
/// @param src one whole expert slice (c->src_stride bytes)
/// @param[out] dst the destination expert slice (c->dst_stride bytes); only the cut bytes are written
static inline void exl3_expert_cut_apply(const exl3_cut *c, const uint8_t *src, uint8_t *dst) {
    for (int i = 0; i < 3; i++) {
        const exl3_copy2d *p = &c->copy[i];
        for (uint64_t r = 0; r < p->height; r++)
            memcpy(dst + p->dst + r * p->dst_pitch, src + p->src + r * p->src_pitch, (size_t)p->width);
    }
}

/**
 * Natural-order Sylvester Hadamard of one 128-vector in place, scaled by
 * 1/sqrt(128): the basis exllamav3 quantizes in (`hadamard_inner.cuh`, and the
 * fp32 reference in its `tests/test_reconstruct_had.py`).  Double precision:
 * this is the reference the device rotations are graded against.
 */
/// @param[in,out] v the 128-vector, rotated in place
static inline void exl3_had128(double v[EXL3_HAD_BLOCK]) {
    for (int h = 1; h < EXL3_HAD_BLOCK; h <<= 1) {
        for (int i = 0; i < EXL3_HAD_BLOCK; i += 2 * h) {
            for (int j = i; j < i + h; j++) {
                const double a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
    const double inv = 1.0 / 11.313708498984761; /* 1/sqrt(128) */
    for (int i = 0; i < EXL3_HAD_BLOCK; i++) v[i] *= inv;
}

#endif /* PULSAR_EXL3_TRELLIS_H */

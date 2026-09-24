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
#define EXL3_HD __host__ __device__
#else
#define EXL3_HD
#endif

/* The header is included by host TUs and by CUDA TUs (the expert-table gate,
 * the kernels' launchers), so the fp16 conversions are software and exact --
 * no _Float16, no NEON, no engine header.  The dequant gate holds them to
 * exllamav3's own codebook values, all 65536 of them. */

/** IEEE binary16 bits -> float, exact (subnormals, Inf, NaN included). */
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
EXL3_HD static inline bool exl3_k2_valid(int k2) {
    const int bits = k2 >> 1;
    if (k2 & 1) return bits >= 1 && bits <= 3;
    return bits >= 1 && bits <= 8;
}

/** uint16 words per 256-weight tile: 16K, or 16K + 8 for a half-integer K. */
EXL3_HD static inline int exl3_words_per_tile(int k2) {
    return 16 * (k2 >> 1) + ((k2 & 1) ? 8 : 0);
}

/**
 * The rate a tile width encodes, or 0 when no EXL3 rate stores that many
 * words.  K is DERIVED from the trellis' last dimension; nothing else in the
 * checkpoint records it.  (16K + 8 for K=2 is 40 = 16 * 2.5: the half-integer
 * widths collide with nothing because 16K + 8 is never a multiple of 16.)
 */
EXL3_HD static inline int exl3_k2_from_words(int words) {
    if (words <= 0 || words % 8) return 0;
    const int k2 = (words % 16 == 0) ? 2 * (words / 16) : 2 * (words / 16) + 1;
    return exl3_k2_valid(k2) ? k2 : 0;
}

/** Stream bits per tile. */
EXL3_HD static inline int exl3_stream_bits(int k2) {
    return 16 * exl3_words_per_tile(k2);
}

/**
 * Stream bit index (exclusive) at which position p's 16-bit state ends.
 * Integer K: (p+1)K.  Half-integer K: a pair (2j, 2j+1) spans 2K+1 bits, the
 * even position ends K bits in, the odd one at the pair's end
 * (`dq8_half`: w6 = w7 >> (K+1), w5 = w6 >> K).
 */
EXL3_HD static inline int exl3_state_end_bit(int k2, int p) {
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
EXL3_HD static inline void exl3_tile_position(int p, int *row, int *col) {
    const int t = p >> 3, i = p & 7;
    *row = (t % 4) * 2 + (i & 1) + 8 * ((i >> 1) & 1); /* {0, 1, 8, 9}[i & 3] */
    *col = t / 4 + 8 * (i >> 2);
}

/**
 * Dequantize one tile to its 256 fp16 weights (bits), row-major 16x16 in the
 * rotated basis (W_hat).  Byte-exact with exllamav3's `reconstruct`.
 */
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
EXL3_HD static inline int exl3_type_k2(uint32_t type) {
    switch (type) {
    case PULSAR_TENSOR_EXL3M_K2:  return 4;
    case PULSAR_TENSOR_EXL3M_K2H: return 5;
    case PULSAR_TENSOR_EXL3M_K3:  return 6;
    default:                      return 0;
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
 * Natural-order Sylvester Hadamard of one 128-vector in place, scaled by
 * 1/sqrt(128): the basis exllamav3 quantizes in (`hadamard_inner.cuh`, and the
 * fp32 reference in its `tests/test_reconstruct_had.py`).  Double precision:
 * this is the reference the device rotations are graded against.
 */
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

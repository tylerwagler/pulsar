/**
 * @file exl3_trellis.h
 * @brief EXL3 trellis tile arithmetic: the tile bit layout, the mul1 codebook
 *        and the 128-block Hadamard basis (L245).
 *
 * Ported from exllamav3 (MIT, turboderp-org/exllamav3 @ 6b84a21):
 * `quant/pack.cu` (the packer, which fixes the bit order), `quant/exl3_dq.cuh`
 * (the state windows, integer and half-integer K), `quant/codebook.cuh`
 * (`decode_3inst<2>`, the mul1 codebook) and `exl3_lib/quantize.py`
 * (`tensor_core_perm`, the position map).  This header is the ONE host
 * authority for those facts in pulsar; the transcoder that turns an EXL3
 * checkpoint into our container and the dequant gate both read it, and the
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

#include "pulsar_engine_internal.h" /* f16_to_f32 */

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
static inline bool exl3_k2_valid(int k2) {
    const int bits = k2 >> 1;
    if (k2 & 1) return bits >= 1 && bits <= 3;
    return bits >= 1 && bits <= 8;
}

/** uint16 words per 256-weight tile: 16K, or 16K + 8 for a half-integer K. */
static inline int exl3_words_per_tile(int k2) {
    return 16 * (k2 >> 1) + ((k2 & 1) ? 8 : 0);
}

/**
 * The rate a tile width encodes, or 0 when no EXL3 rate stores that many
 * words.  K is DERIVED from the trellis' last dimension; nothing else in the
 * checkpoint records it.  (16K + 8 for K=2 is 40 = 16 * 2.5: the half-integer
 * widths collide with nothing because 16K + 8 is never a multiple of 16.)
 */
static inline int exl3_k2_from_words(int words) {
    if (words <= 0 || words % 8) return 0;
    const int k2 = (words % 16 == 0) ? 2 * (words / 16) : 2 * (words / 16) + 1;
    return exl3_k2_valid(k2) ? k2 : 0;
}

/** Stream bits per tile. */
static inline int exl3_stream_bits(int k2) {
    return 16 * exl3_words_per_tile(k2);
}

/**
 * Stream bit index (exclusive) at which position p's 16-bit state ends.
 * Integer K: (p+1)K.  Half-integer K: a pair (2j, 2j+1) spans 2K+1 bits, the
 * even position ends K bits in, the odd one at the pair's end
 * (`dq8_half`: w6 = w7 >> (K+1), w5 = w6 >> K).
 */
static inline int exl3_state_end_bit(int k2, int p) {
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
    const float h = f16_to_f32((uint16_t)s);
    const float v = h * f16_to_f32(0x1eee) + f16_to_f32(0xc931);
    const _Float16 r = (_Float16)v;
    uint16_t bits;
    __builtin_memcpy(&bits, &r, sizeof bits);
    return bits;
}

/**
 * Tensor-core position map: position p of a tile is weight (row, col) of the
 * 16x16 tile, row along `in`, col along `out`.  Lane t = p >> 3, i = p & 7:
 * row = (t % 4) * 2 + {0, 1, 8, 9}[i & 3], col = t / 4 + 8 * (i >> 2).
 */
static inline void exl3_tile_position(int p, int *row, int *col) {
    static const int row_of[4] = {0, 1, 8, 9};
    const int t = p >> 3, i = p & 7;
    *row = (t % 4) * 2 + row_of[i & 3];
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

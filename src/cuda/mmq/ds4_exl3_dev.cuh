// SPDX-License-Identifier: MIT
/** @file
 *  EXL3 device arithmetic shared by every EXL3 kernel: tile geometry, run window, codebook decode, H128. */
// The EXL3 device arithmetic shared by every EXL3 kernel (L245 routed experts,
// L251 dense Linears): the per-rate tile geometry, the four-state run window,
// the mul1 codebook decode, and the 128-block Hadamard over one warp.
//
// src/engine/exl3_trellis.h is the format authority; nothing here restates the
// bit layout -- the run window asks the header's exl3_state_end_bit where each
// state ends.  The codebook decode is the same three instructions as
// exllamav3's decode_3inst<2>, whose values the header's host decode is held
// to (tests/exl3_dequant_gate.cpp), and every kernel built on these helpers is
// graded against that host decode (tests/exl3_moe_prefill_gate.cu,
// tests/exl3_dense_gate.cu).

#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "engine/exl3_trellis.h"

namespace exl3dev {

/// 1 / sqrt(128): the Hadamard's normalisation.
constexpr float kInvSqrt128 = 0.08838834764831845f;

/** The fewest bits a run's 64-bit window has to spare at rate K2: over every
 *  run start p0 (p0 % 4 == 0), the window is the two words ending at the word
 *  holding the fourth state's last bit, and the first state's 16 bits must
 *  begin inside it.  Runs start 4-aligned, so the shift is never arbitrary and
 *  the run's span (3K + 16) bounds nothing: K6 spares 6 bits and K8 24, while
 *  K6.5 (-2), K7 (-1) and K7.5 (-5) do not fit (L266; TensorFold's
 *  experts_grouped.cuh window table agrees). */
/// @param K2 the rate in half-bit units (2 x bits per weight).
/// @return the least spare bits over every run start (< 0: some run does not fit).
__host__ __device__ constexpr int run4_slack(int K2) {
    int least = 64;
    for (int p0 = 0; p0 < 256; p0 += 4) {
        const int e3 = exl3_state_end_bit(K2, p0 + 3);
        const int spare = exl3_state_end_bit(K2, p0) - 16 - (((e3 - 1) >> 5) - 1) * 32;
        least = spare < least ? spare : least;
    }
    return least;
}

/** Per-rate constants.  A lane decodes its 16 weights of a tile as four runs
 *  of four consecutive positions, each run from ONE 64-bit window of two
 *  consecutive uint32 words (run4_window); run4_slack checks the run fits. */
template <int K2> struct Rate {
    /// uint16 words per 16 x 16 tile: 256 weights x K2 / 2 bits / 16.
    static constexpr int words16 = 16 * (K2 >> 1) + ((K2 & 1) ? 8 : 0);
    /// uint32 words per tile.
    static constexpr int words32 = words16 / 2;
    static_assert(words16 % 2 == 0, "a tile is whole uint32 words");
    static_assert(run4_slack(K2) >= 0, "a run of this rate does not fit the 64-bit run window");
};

/** exllamav3's mul1 codebook, the device form: x * 0x83DCD12D, a dp4a byte
 *  sum onto 0x6400 (= fp16 1024 + bytesum, exact), one hfma.  Bit-identical to
 *  the host exl3_mul1_decode (both graded against the extension's table). */
/// @param x a 16-bit trellis state (in the low bits).
/// @return the codebook value, f32.
__device__ __forceinline__ float mul1(uint32_t x) {
    x *= EXL3_MUL1_MULTIPLIER;
    const uint32_t s = __dp4a(x, 0x01010101u, 0x6400u);
    const __half h = __ushort_as_half((unsigned short)s);
    return __half2float(__hfma(h, __ushort_as_half(0x1eee), __ushort_as_half(0xc931)));
}

/** Two codebook values as one fp16x2 (x0 low, x1 high): the same byte sums
 *  and the same fp16 fma as mul1, done as ONE hfma2 -- each half is
 *  bit-identical to mul1's fp16 before its f32 conversion (exllamav3's
 *  decode_3inst<2> pairs them the same way).  The pair is an mma.m16n8k16 B
 *  register as it stands. */
/// @param x0 the first state.
/// @param x1 the second state.
/// @return the two values as fp16x2 (x0 low).
__device__ __forceinline__ uint32_t mul1_pair(uint32_t x0, uint32_t x1) {
    const uint32_t s0 = __dp4a(x0 * EXL3_MUL1_MULTIPLIER, 0x01010101u, 0x6400u);
    const uint32_t s1 = __dp4a(x1 * EXL3_MUL1_MULTIPLIER, 0x01010101u, 0x6400u);
    const uint32_t h = __byte_perm(s0, s1, 0x5410);                     /* s0 low 16 | s1 low 16 */
    const __half2 v = __hfma2(*reinterpret_cast<const __half2 *>(&h),
                              __halves2half2(__ushort_as_half(0x1eee), __ushort_as_half(0x1eee)),
                              __halves2half2(__ushort_as_half(0xc931), __ushort_as_half(0xc931)));
    return *reinterpret_cast<const uint32_t *>(&v);
}

/** Where the run starting at position p0 (p0 % 4 == 0) of a tile lives: the
 *  two word indices of its 64-bit window (lo wraps to the tile's last word when
 *  the window starts before the tile -- the tail-biting ring) and the right
 *  shift that puts the fourth state's end at bit 0.  Depends on p0 only, so a
 *  kernel computes it once per lane and reuses it for every tile. */
/// @tparam K2 the rate in half-bit units (2 x bits per weight).
/// @param p0 the run's first position in the tile (a multiple of 4; of 8 for run8).
/// @param lo out: the window's first word index.
/// @param hi out: the window's second word index.
/// @param sh out: the right shift.
template <int K2>
__device__ __forceinline__ void run4_window(int p0, int &lo, int &hi, int &sh) {
    constexpr int nw = Rate<K2>::words32;
    const int e3 = exl3_state_end_bit(K2, p0 + 3);
    hi = (e3 - 1) >> 5;
    lo = (hi + nw - 1) % nw;
    sh = ((hi + 1) << 5) - e3;
}

/** The four states of the run at p0 from its window value `v` (the two words
 *  joined and shifted, EXL3_RUN4_JOIN: bit 0 is the fourth state's last stream bit;
 *  stream bit s is in word s/32 at bit 31 - s%32, the layout the header
 *  documents). */
/// @tparam K2 the rate in half-bit units (2 x bits per weight).
/// @param v the window value (EXL3_RUN4_JOIN).
/// @param p0 the run's first position in the tile (a multiple of 4; of 8 for run8).
/// @param st out: the run's states.
template <int K2>
__device__ __forceinline__ void run4_split(uint64_t v, int p0, uint32_t st[4]) {
    const int e3 = exl3_state_end_bit(K2, p0 + 3);
    st[3] = (uint32_t)v & 0xffffu;
    st[2] = (uint32_t)(v >> (e3 - exl3_state_end_bit(K2, p0 + 2))) & 0xffffu;
    st[1] = (uint32_t)(v >> (e3 - exl3_state_end_bit(K2, p0 + 1))) & 0xffffu;
    st[0] = (uint32_t)(v >> (e3 - exl3_state_end_bit(K2, p0))) & 0xffffu;
}

/** A run's window value from its two words and run4_window's shift.  A macro,
 *  not a function: the expert arm's kernels were tuned with this expression
 *  inline at the load site, and routing the two loads through a call's
 *  arguments reschedules them (SASS-compared, 2026-09-26); the expression
 *  itself is written once, here. */
#define EXL3_RUN4_JOIN(wlo, whi, sh) ((((uint64_t)(wlo) << 32) | (uint64_t)(whi)) >> (sh))

/** run4_slack for a run of EIGHT positions starting at p0 % 8 == 0 -- a lane's whole share of an mma
 *  B fragment (positions 8t..8t+7) -- from ONE 64-bit window: >= 0 when every such run fits (K2 <= 6 and
 *  K2 == 8; TensorFold's experts_grouped.cuh window table agrees), < 0 when it does not (the run then takes
 *  two run4 windows). */
/// @param K2 the rate in half-bit units (2 x bits per weight).
/// @return the least spare bits over every eight-position run start.
__host__ __device__ constexpr int run8_slack(int K2) {
    int least = 64;
    for (int p0 = 0; p0 < 256; p0 += 8) {
        const int e7 = exl3_state_end_bit(K2, p0 + 7);
        const int spare = exl3_state_end_bit(K2, p0) - 16 - (((e7 - 1) >> 5) - 1) * 32;
        least = spare < least ? spare : least;
    }
    return least;
}

/** run4_window for the eight-position run at p0 (p0 % 8 == 0; run8_slack(K2) >= 0). */
/// @tparam K2 the rate in half-bit units (2 x bits per weight).
/// @param p0 the run's first position in the tile (a multiple of 4; of 8 for run8).
/// @param lo out: the window's first word index.
/// @param hi out: the window's second word index.
/// @param sh out: the right shift.
template <int K2>
__device__ __forceinline__ void run8_window(int p0, int &lo, int &hi, int &sh) {
    static_assert(run8_slack(K2) >= 0, "an eight-position run of this rate does not fit one 64-bit window");
    constexpr int nw = Rate<K2>::words32;
    const int e7 = exl3_state_end_bit(K2, p0 + 7);
    hi = (e7 - 1) >> 5;
    lo = (hi + nw - 1) % nw;
    sh = ((hi + 1) << 5) - e7;
}

/** The eight states of the run at p0 from its window value (EXL3_RUN4_JOIN with run8_window's shift). */
/// @tparam K2 the rate in half-bit units (2 x bits per weight).
/// @param v the window value (EXL3_RUN4_JOIN).
/// @param p0 the run's first position in the tile (a multiple of 4; of 8 for run8).
/// @param st out: the run's states.
template <int K2>
__device__ __forceinline__ void run8_split(uint64_t v, int p0, uint32_t st[8]) {
    const int e7 = exl3_state_end_bit(K2, p0 + 7);
#pragma unroll
    for (int j = 0; j < 8; ++j) st[j] = (uint32_t)(v >> (e7 - exl3_state_end_bit(K2, p0 + j))) & 0xffffu;
}

/** The four states of positions p0..p0+3 of one tile, `w` = the tile's words. */
/// @tparam K2 the rate in half-bit units (2 x bits per weight).
/// @param w the tile's uint32 words.
/// @param p0 the run's first position in the tile (a multiple of 4; of 8 for run8).
/// @param st out: the run's states.
template <int K2>
__device__ __forceinline__ void run4(const uint32_t *__restrict__ w, int p0, uint32_t st[4]) {
    int lo, hi, sh;
    run4_window<K2>(p0, lo, hi, sh);
    run4_split<K2>(EXL3_RUN4_JOIN(w[lo], w[hi], sh), p0, st);
}

/** In-place natural-order Sylvester H128 / sqrt(128) over one 128-block held
 *  as four consecutive values per lane (lane l holds elements 4l..4l+3): two
 *  in-lane stages, then five xor-shuffle stages (the lane with the bit set
 *  takes a - b, the other a + b).  Full warp required. */
/// @param v this lane's four values, transformed in place.
__device__ __forceinline__ void had128(float v[4]) {
    float a, b;
    a = v[0]; b = v[1]; v[0] = a + b; v[1] = a - b;
    a = v[2]; b = v[3]; v[2] = a + b; v[3] = a - b;
    a = v[0]; b = v[2]; v[0] = a + b; v[2] = a - b;
    a = v[1]; b = v[3]; v[1] = a + b; v[3] = a - b;
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int m = 1; m <= 16; m <<= 1) {
        const bool upper = (lane & m) != 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float p = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = upper ? (p - v[j]) : (v[j] + p);
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= kInvSqrt128;
}

/// @param bits an E4M3 byte.
/// @return its value, f32 (exact).
__device__ __forceinline__ float e4m3_to_f32(uint8_t bits) {
    return (float)(*reinterpret_cast<const __nv_fp8_e4m3 *>(&bits));
}

} // namespace exl3dev

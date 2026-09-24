// SPDX-License-Identifier: MIT
// The EXL3 routed-expert arm (L245).  See ds4_exl3_gemv.cuh for the contracts
// and src/engine/exl3_trellis.h for the format: this file restates NOTHING
// about the bit layout -- it calls the header's functions for where a state
// ends and what a tile is, and its codebook decode is the same three
// instructions as exllamav3's decode_3inst<2> (whose values the header's host
// decode is held to by tests/exl3_dequant_gate.cpp).
//
// The dataflow is L210's IQ2 decode GEMV (ds4_mmq_d2r.cu) generalised to R
// rows: one CTA per (32 output rows, a block of R consecutive expert-sorted
// assignments), 8 warps split K, the activations staged as f32 in shared
// memory 256 k at a time, f32 fma, a fixed-order cross-warp reduction.  The
// R rows of a CTA that share an expert are computed as one run: the expert's
// weights stream from DRAM ONCE for the run and every row's accumulator takes
// the same k-tile, run and warp order as a row computed alone -- so R is a
// PERFORMANCE choice (occupancy at decode widths, weight reuse at prefill
// widths) and every R is bit-identical.  Decode and prefill are one
// arithmetic here, literally.  What is EXL3-specific:
//   * the weights: a lane owning output column n of tile column c reads its
//     16 weights of a k-tile as FOUR runs of four consecutive positions
//     (positions 32(c%8) + 8a + 4(c/8) + b, rows 2a + {0,1,8,9}); one run is
//     two uint32 loads and one 64-bit funnel window (the four states span at
//     most 3K+16 bits, <= 26 for the rates here) -- exllamav3's dq4 pattern.
//     A warp's two n-tiles are 192 contiguous bytes per k-tile at K=3.
//   * the input rotation for gate/up: suh is per expert-projection and sits
//     INSIDE the Hadamard, so it cannot be hoisted before routing; the kernel
//     stages x * suh_g and x * suh_u and rotates each 128-block in place
//     (warp-local Sylvester butterflies, 1/sqrt(128)).  Down's input arrives
//     pre-rotated from the fold (the producer rule holds there).
//   * the outputs are UNROTATED: svh and the output Hadamard mix 128
//     consecutive outputs and belong to the epilogues below, which already
//     touch those elements.

#include "ds4_exl3_gemv.cuh"
#include "ds4_act_block.cuh"
#include "cuda/pulsar_cuda_mx.cuh"
#include "engine/exl3_trellis.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdio>

namespace {

constexpr int kRows   = 32;    ///< output rows per CTA = one warp's lanes = two n-tiles
constexpr int kWarps  = 8;     ///< K split
constexpr int kMaxK   = 5120;  ///< the activation row (V4.1 n_embd)
constexpr int kChunk  = 256;   ///< k staged per pass: two Hadamard blocks, 16 k-tiles = 2 per warp
constexpr int kMaxR   = 16;    ///< the widest row block
constexpr float kInvSqrt128 = 0.08838834764831845f;

template <int K2> struct Exl3Rate {
    static constexpr int words16 = 16 * (K2 >> 1) + ((K2 & 1) ? 8 : 0);
    static constexpr int words32 = words16 / 2;
    static_assert(words16 % 2 == 0, "a tile is whole uint32 words");
    /* the four states of a run lie in one 64-bit window: their span is
     * 3K + 16 (integer K) or 2(2K+1) + 16 (half-integer K) bits, <= 32 */
    static_assert(3 * (K2 >> 1) + 16 + 2 <= 32, "rate too wide for the 64-bit run window");
};

/** exllamav3's mul1 codebook, the device form: x * 0x83DCD12D, a dp4a byte
 *  sum onto 0x6400 (= fp16 1024 + bytesum, exact), one hfma.  Bit-identical to
 *  the host exl3_mul1_decode (both graded against the extension's table). */
__device__ __forceinline__ float exl3_dev_mul1(uint32_t x) {
    x *= EXL3_MUL1_MULTIPLIER;
    const uint32_t s = __dp4a(x, 0x01010101u, 0x6400u);
    const __half h = __ushort_as_half((unsigned short)s);
    return __half2float(__hfma(h, __ushort_as_half(0x1eee), __ushort_as_half(0xc931)));
}

/** The four states of positions p0..p0+3 (p0 % 4 == 0) of one tile.  `w` is
 *  the tile's words as uint32 (stream bit s in w[s/32] at bit 31 - s%32, the
 *  layout the header documents); the window is the 64 bits ending at
 *  end(p0+3), wrapping to the last word when it starts before the tile. */
template <int K2>
__device__ __forceinline__ void exl3_dev_run4(const uint32_t *__restrict__ w, int p0, uint32_t st[4]) {
    constexpr int nw = Exl3Rate<K2>::words32;
    const int e3 = exl3_state_end_bit(K2, p0 + 3);
    const int hi = (e3 - 1) >> 5;
    const int lo = (hi + nw - 1) % nw;
    const int s  = ((hi + 1) << 5) - e3;
    const uint64_t v = (((uint64_t)w[lo] << 32) | (uint64_t)w[hi]) >> s;
    st[3] = (uint32_t)v & 0xffffu;
    st[2] = (uint32_t)(v >> (e3 - exl3_state_end_bit(K2, p0 + 2))) & 0xffffu;
    st[1] = (uint32_t)(v >> (e3 - exl3_state_end_bit(K2, p0 + 1))) & 0xffffu;
    st[0] = (uint32_t)(v >> (e3 - exl3_state_end_bit(K2, p0))) & 0xffffu;
}

/** In-place natural-order Sylvester H128 / sqrt(128) over one 128-block held
 *  as four consecutive values per lane (lane l holds elements 4l..4l+3): two
 *  in-lane stages, then five xor-shuffle stages (the lane with the bit set
 *  takes a - b, the other a + b).  Full warp required. */
__device__ __forceinline__ void exl3_dev_had128(float v[4]) {
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

__device__ __forceinline__ float exl3_dev_e4m3_to_f32(uint8_t bits) {
    return (float)(*reinterpret_cast<const __nv_fp8_e4m3 *>(&bits));
}

/* ---------------------------------------------------------------------- */
/* the GEMV                                                                */

/* The single-row instance (R = 1): the step-3 body, whole-vector staging and
 * one Hadamard pass, no per-chunk barriers -- 50 us for one assignment where
 * the chunked multi-row body pays 68.  Bit-identical to the multi-row form
 * (the gate proves R = 1 / 4 / 16 equal), kept because a decode step is one
 * assignment per CTA and the barriers are its whole difference. */
template <bool PAIR, int K2>
__global__ void __launch_bounds__(kRows * kWarps)
exl3_moe_gemv_kernel_r1(const void *__restrict__ gate_table,
                     const void *__restrict__ up_table,
                     const block_mx_act_mmq *__restrict__ act,
                     const int32_t *__restrict__ ids_dst,
                     const int32_t *__restrict__ expert_bounds,
                     float *__restrict__ out_gate,
                     float *__restrict__ out_up,
                     int M, int K, int n_assign, int E) {
    constexpr int W32 = Exl3Rate<K2>::words32;
    __shared__ float s_x[PAIR ? 2 : 1][kMaxK];
    __shared__ float s_red[kWarps][kRows][2];   /* 42 KB with s_x: the whole-vector staging */
    __shared__ int   s_expert;

    const int col  = blockIdx.y;                 /* assignment (expert-sorted) */
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * kRows + lane;
    const int row  = blockIdx.x * kRows + lane;  /* output column n of W (in, out) */
    if (col >= n_assign) return;

    if (tid == 0) {
        int lo = 0, hi = E - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (expert_bounds[mid] <= col) lo = mid; else hi = mid - 1;
        }
        s_expert = lo;
    }
    __syncthreads();
    const int expert = s_expert;

    const void *const *pg = reinterpret_cast<const void *const *>(gate_table);
    const void *const *pu = reinterpret_cast<const void *const *>(up_table);
    const uint32_t *tg = reinterpret_cast<const uint32_t *>(pg[2 * (size_t)expert]);
    const uint32_t *tu = PAIR ? reinterpret_cast<const uint32_t *>(pu[2 * (size_t)expert]) : nullptr;
    const __half   *sg = reinterpret_cast<const __half *>(pg[2 * (size_t)expert + 1]);   /* suh_g[K] | svh_g[M] */
    const __half   *su = PAIR ? reinterpret_cast<const __half *>(pu[2 * (size_t)expert + 1]) : nullptr;

    /* Stage the assignment's activation row as f32 -- for gate/up already
     * multiplied by each projection's suh, so the rotation below is per
     * projection.  block i holds k in [128 i, 128 i + 128) as 4 groups of 32
     * E4M3 under one ue8m0 byte each. */
    const int n_k128 = K >> 7;
    for (int i = tid; i < n_k128 * 4; i += kRows * kWarps) {
        const int blk = i >> 2, grp = i & 3;
        const block_mx_act_mmq &b = act[(uint64_t)blk * (uint64_t)n_assign + (uint64_t)col];
        const float sc = exp2f(b.d4[grp] - 127.0f);
        const int8_t *q = b.qs + grp * 32;
        const int k0 = blk * 128 + grp * 32;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const float v = exl3_dev_e4m3_to_f32((uint8_t)q[j]) * sc;
            if constexpr (PAIR) {
                s_x[0][k0 + j] = v * __half2float(sg[k0 + j]);
                s_x[1][k0 + j] = v * __half2float(su[k0 + j]);
            } else {
                s_x[0][k0 + j] = v;
            }
        }
    }
    __syncthreads();
    if constexpr (PAIR) {
        /* H128 per 128-block, per projection: one warp per (block, projection). */
        for (int t = warp; t < 2 * n_k128; t += kWarps) {
            float *v4 = s_x[t & 1] + (t >> 1) * 128 + lane * 4;
            float v[4] = {v4[0], v4[1], v4[2], v4[3]};
            exl3_dev_had128(v);
            v4[0] = v[0]; v4[1] = v[1]; v4[2] = v[2]; v4[3] = v[3];
        }
        __syncthreads();
    }

    /* This lane's tile column and the run geometry (see the file comment). */
    const int ntn = M >> 4;                              /* n-tiles per k-tile row */
    const int nt  = (blockIdx.x << 1) + (lane >> 4);
    const int c   = lane & 15;
    const bool row_ok = row < M;
    const int ktn = K >> 4;

    float acc_g = 0.0f, acc_u = 0.0f;
    for (int kt = warp; kt < ktn; kt += kWarps) {
        if (!row_ok) break;
        const uint32_t *wg = tg + ((size_t)kt * (size_t)ntn + (size_t)nt) * W32;
        const uint32_t *wu = PAIR ? tu + ((size_t)kt * (size_t)ntn + (size_t)nt) * W32 : nullptr;
        const float *xg = s_x[0] + kt * 16;
        const float *xu = PAIR ? s_x[1] + kt * 16 : nullptr;
        uint32_t stg[4][4], stu[4][4];
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            const int p0 = 32 * (c & 7) + 8 * a + 4 * (c >> 3);
            exl3_dev_run4<K2>(wg, p0, stg[a]);
            if constexpr (PAIR) exl3_dev_run4<K2>(wu, p0, stu[a]);
        }
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            /* run a: positions p0..p0+3 are rows 2a, 2a+1, 2a+8, 2a+9 */
            const int r[4] = {2 * a, 2 * a + 1, 2 * a + 8, 2 * a + 9};
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                acc_g = fmaf(exl3_dev_mul1(stg[a][b]), xg[r[b]], acc_g);
                if constexpr (PAIR) acc_u = fmaf(exl3_dev_mul1(stu[a][b]), xu[r[b]], acc_u);
            }
        }
    }
    s_red[warp][lane][0] = acc_g;
    if constexpr (PAIR) s_red[warp][lane][1] = acc_u;
    __syncthreads();
    if (warp == 0 && row_ok) {
        float g = 0.0f, u = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            g += s_red[w][lane][0];
            if constexpr (PAIR) u += s_red[w][lane][1];
        }
        const uint64_t o = (uint64_t)ids_dst[col] * (uint64_t)M + (uint64_t)row;
        out_gate[o] = g;
        if constexpr (PAIR) out_up[o] = u;
    }
}


/* Shared memory is ONE buffer: the staged activation chunks during the k loop
 * (PAIR ? 2 : 1) x R x kChunk f32, then the cross-warp reduction after it
 * (kWarps x kRows x R x 2 f32).  The larger of the two sizes it. */
template <bool PAIR, int R>
struct GemvSmem {
    static constexpr int stage_floats = (PAIR ? 2 : 1) * R * kChunk;
    static constexpr int red_floats   = kWarps * kRows * R * (PAIR ? 2 : 1);
    static constexpr int floats = stage_floats > red_floats ? stage_floats : red_floats;
    static_assert(floats * 4 <= 40 * 1024, "the GEMV's shared buffer must stay under the static limit");
};

template <bool PAIR, int K2, int R>
__global__ void __launch_bounds__(kRows * kWarps)
exl3_moe_gemv_kernel(const void *__restrict__ gate_table,
                     const void *__restrict__ up_table,
                     const block_mx_act_mmq *__restrict__ act,
                     const int32_t *__restrict__ ids_dst,
                     const int32_t *__restrict__ expert_bounds,
                     float *__restrict__ out_gate,
                     float *__restrict__ out_up,
                     int M, int K, int n_assign, int E) {
    constexpr int W32 = Exl3Rate<K2>::words32;
    constexpr int NV = PAIR ? 2 : 1;
    __shared__ __align__(16) float s_buf[GemvSmem<PAIR, R>::floats];
    __shared__ int s_expert[R];

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * kRows + lane;
    const int row  = blockIdx.x * kRows + lane;      /* output column n of W (in, out) */
    const int col0 = blockIdx.y * R;                 /* first assignment of this block */
    const int nrow = min(R, n_assign - col0);
    if (nrow <= 0) return;

    /* which expert owns each assignment: expert_bounds is E+1 ascending offsets */
    if (tid < nrow) {
        const int col = col0 + tid;
        int lo = 0, hi = E - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (expert_bounds[mid] <= col) lo = mid; else hi = mid - 1;
        }
        s_expert[tid] = lo;
    }
    __syncthreads();

    /* the reduction view of the shared buffer: [warp][lane][r][v] */
    float *s_red = s_buf;
    const int n_chunk = K / kChunk;
    const int ntn = M >> 4;                              /* n-tiles per k-tile row */
    const int nt  = (blockIdx.x << 1) + (lane >> 4);
    const int c   = lane & 15;
    const bool row_ok = row < M;

    /* runs of consecutive rows sharing one expert (the assignments are sorted) */
    for (int r0 = 0; r0 < nrow;) {
        const int expert = s_expert[r0];
        int r1 = r0 + 1;
        while (r1 < nrow && s_expert[r1] == expert) ++r1;
        const int nr = r1 - r0;

        const void *const *pg = reinterpret_cast<const void *const *>(gate_table);
        const void *const *pu = reinterpret_cast<const void *const *>(up_table);
        const uint32_t *tg = reinterpret_cast<const uint32_t *>(pg[2 * (size_t)expert]);
        const uint32_t *tu = PAIR ? reinterpret_cast<const uint32_t *>(pu[2 * (size_t)expert]) : nullptr;
        const __half   *sg = reinterpret_cast<const __half *>(pg[2 * (size_t)expert + 1]);   /* suh_g[K] | svh_g[M] */
        const __half   *su = PAIR ? reinterpret_cast<const __half *>(pu[2 * (size_t)expert + 1]) : nullptr;

        float acc[R][NV];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int v = 0; v < NV; ++v) acc[r][v] = 0.0f;

        for (int kc = 0; kc < n_chunk; ++kc) {
            const int k0 = kc * kChunk;
            /* stage this chunk of each row's activation as f32 -- for gate/up
             * already multiplied by each projection's suh.  block i of the row
             * holds k in [128 i, 128 i + 128) as 4 groups of 32 E4M3 under one
             * ue8m0 byte each; a chunk is 2 blocks = 8 groups per row. */
            __syncthreads();                                   /* the previous pass's readers are done */
            for (int i = tid; i < nr * 8; i += kRows * kWarps) {
                const int r = i >> 3, g8 = i & 7;
                const int blk = (k0 >> 7) + (g8 >> 2), grp = g8 & 3;
                const block_mx_act_mmq &b = act[(uint64_t)blk * (uint64_t)n_assign + (uint64_t)(col0 + r0 + r)];
                const float sc = exp2f(b.d4[grp] - 127.0f);
                const int8_t *q = b.qs + grp * 32;
                const int kk = g8 * 32;                        /* offset inside the chunk */
                float *xg = s_buf + (0 * R + r) * kChunk + kk;
                float *xu = PAIR ? s_buf + (1 * R + r) * kChunk + kk : nullptr;
#pragma unroll
                for (int j = 0; j < 32; ++j) {
                    const float v = exl3_dev_e4m3_to_f32((uint8_t)q[j]) * sc;
                    if constexpr (PAIR) {
                        xg[j] = v * __half2float(sg[k0 + kk + j]);
                        xu[j] = v * __half2float(su[k0 + kk + j]);
                    } else {
                        xg[j] = v;
                    }
                }
            }
            __syncthreads();
            if constexpr (PAIR) {
                /* H128 per 128-block, per row, per projection: one warp per task */
                for (int t = warp; t < nr * 2 * 2; t += kWarps) {
                    const int v = t & 1, blk = (t >> 1) & 1, r = t >> 2;
                    float *v4 = s_buf + (v * R + r) * kChunk + blk * 128 + lane * 4;
                    float x[4] = {v4[0], v4[1], v4[2], v4[3]};
                    exl3_dev_had128(x);
                    v4[0] = x[0]; v4[1] = x[1]; v4[2] = x[2]; v4[3] = x[3];
                }
                __syncthreads();
            }
            if (!row_ok) continue;
            /* this warp's two k-tiles of the chunk: the same k order a row alone takes */
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int kt = (k0 >> 4) + half * kWarps + warp;
                const int kk = (half * kWarps + warp) * 16;    /* the tile's k inside the chunk */
                const uint32_t *wg = tg + ((size_t)kt * (size_t)ntn + (size_t)nt) * W32;
                const uint32_t *wu = PAIR ? tu + ((size_t)kt * (size_t)ntn + (size_t)nt) * W32 : nullptr;
                uint32_t stg[4][4], stu[4][4];
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    const int p0 = 32 * (c & 7) + 8 * a + 4 * (c >> 3);
                    exl3_dev_run4<K2>(wg, p0, stg[a]);
                    if constexpr (PAIR) exl3_dev_run4<K2>(wu, p0, stu[a]);
                }
                /* decode once, apply to every row of the run */
                float wgv[16], wuv[16];
#pragma unroll
                for (int a = 0; a < 4; ++a)
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        wgv[a * 4 + b] = exl3_dev_mul1(stg[a][b]);
                        if constexpr (PAIR) wuv[a * 4 + b] = exl3_dev_mul1(stu[a][b]);
                    }
                for (int r = 0; r < nr; ++r) {
                    const float *xg = s_buf + (0 * R + r) * kChunk + kk;
                    const float *xu = PAIR ? s_buf + (1 * R + r) * kChunk + kk : nullptr;
#pragma unroll
                    for (int a = 0; a < 4; ++a) {
                        /* run a: positions p0..p0+3 are rows 2a, 2a+1, 2a+8, 2a+9 */
                        const int rw[4] = {2 * a, 2 * a + 1, 2 * a + 8, 2 * a + 9};
#pragma unroll
                        for (int b = 0; b < 4; ++b) {
                            acc[r][0] = fmaf(wgv[a * 4 + b], xg[rw[b]], acc[r][0]);
                            if constexpr (PAIR) acc[r][1] = fmaf(wuv[a * 4 + b], xu[rw[b]], acc[r][1]);
                        }
                    }
                }
            }
        }
        /* the run's partial sums into the reduction view (the staging reads are
         * done), reduced in warp order and written out BEFORE the next run
         * stages over the same buffer */
        __syncthreads();
        for (int r = 0; r < nr; ++r) {
#pragma unroll
            for (int v = 0; v < NV; ++v) s_red[((warp * kRows + lane) * R + r) * NV + v] = acc[r][v];
        }
        __syncthreads();
        if (warp == 0 && row_ok) {
            for (int r = 0; r < nr; ++r) {
                float g = 0.0f, u = 0.0f;
#pragma unroll
                for (int w = 0; w < kWarps; ++w) {
                    g += s_red[((w * kRows + lane) * R + r) * NV + 0];
                    if constexpr (PAIR) u += s_red[((w * kRows + lane) * R + r) * NV + 1];
                }
                const uint64_t o = (uint64_t)ids_dst[col0 + r0 + r] * (uint64_t)M + (uint64_t)row;
                out_gate[o] = g;
                if constexpr (PAIR) out_up[o] = u;
            }
        }
        r0 = r1;
    }
}

/* ---------------------------------------------------------------------- */
/* the epilogues                                                            */

/* One warp per (pair, 128-block of mid); lane l holds mid columns 4l..4l+3
 * of the block.  The four values per lane make one 32-element MX group an
 * EIGHT-lane group, as in moe_mmq_swiglu_fold_v4_kernel. */
__global__ void __launch_bounds__(256)
exl3_moe_fold_kernel(const float *__restrict__ gate_z,
                     const float *__restrict__ up_z,
                     const int32_t *__restrict__ selected,
                     const float *__restrict__ weights,
                     const void *__restrict__ gate_table,
                     const void *__restrict__ up_table,
                     const void *__restrict__ down_table,
                     int in_dim, int mid_dim, int64_t pairs, float clamp,
                     __nv_fp8_e4m3 *__restrict__ mid_q,
                     unsigned char *__restrict__ mid_sf,
                     int mid_kbp) {
    const int64_t task = ((int64_t)blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int n_blk = mid_dim >> 7;
    const int64_t n_tasks = pairs * n_blk;
    if (task >= n_tasks) return;                      /* whole warps: task is per warp */
    const int64_t pair = task / n_blk;
    const int blk = (int)(task - pair * n_blk);
    const int lane = threadIdx.x & 31;
    const int e = selected[pair];
    const void *const *pg = reinterpret_cast<const void *const *>(gate_table);
    const void *const *pu = reinterpret_cast<const void *const *>(up_table);
    const void *const *pd = reinterpret_cast<const void *const *>(down_table);
    /* gate/up scales: suh[in_dim] | svh[mid_dim]; down scales: suh[mid_dim] | svh[out] */
    const __half *svh_g = reinterpret_cast<const __half *>(pg[2 * (size_t)e + 1]) + in_dim;
    const __half *svh_u = reinterpret_cast<const __half *>(pu[2 * (size_t)e + 1]) + in_dim;
    const __half *suh_d = reinterpret_cast<const __half *>(pd[2 * (size_t)e + 1]);
    const int col0 = blk * 128 + lane * 4;
    const float wv = weights[pair];
    const float4 g4 = *reinterpret_cast<const float4 *>(gate_z + pair * mid_dim + col0);
    const float4 u4 = *reinterpret_cast<const float4 *>(up_z + pair * mid_dim + col0);
    float g[4] = {g4.x, g4.y, g4.z, g4.w}, u[4] = {u4.x, u4.y, u4.z, u4.w};
    exl3_dev_had128(g);
    exl3_dev_had128(u);
    float t[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float yg = g[j] * __half2float(svh_g[col0 + j]);
        const float yu = u[j] * __half2float(svh_u[col0 + j]);
        t[j] = pulsar_swiglu_elem(yg, yu, wv, clamp) * __half2float(suh_d[col0 + j]);
    }
    exl3_dev_had128(t);
    float a = fmaxf(fmaxf(fabsf(t[0]), fabsf(t[1])), fmaxf(fabsf(t[2]), fabsf(t[3])));
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    const int se = pulsar_mx_shared_exp(a);
    __nv_fp8_e4m3 *dst = mid_q + (size_t)pair * mid_dim + col0;
    dst[0] = pulsar_mx_encode(t[0], se);
    dst[1] = pulsar_mx_encode(t[1], se);
    dst[2] = pulsar_mx_encode(t[2], se);
    dst[3] = pulsar_mx_encode(t[3], se);
    if ((lane & 7) == 0)
        mid_sf[pulsar_mx_sfoff((int)pair, col0 >> 5, mid_kbp)] = pulsar_mx_scale_byte(se);
}

/* One warp per (token, 128-block of out); slots summed in order. */
__global__ void __launch_bounds__(256)
exl3_moe_sum_kernel(float *__restrict__ out,
                    const float *__restrict__ down_z,
                    const int32_t *__restrict__ selected,
                    const void *__restrict__ down_table,
                    int mid_dim, int out_dim, int n_expert, int n_tokens,
                    uint32_t *__restrict__ nf_flag, uint32_t nf_code) {
    const int64_t task = ((int64_t)blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int n_blk = out_dim >> 7;
    if (task >= (int64_t)n_tokens * n_blk) return;
    const int tok = (int)(task / n_blk);
    const int blk = (int)(task - (int64_t)tok * n_blk);
    const int lane = threadIdx.x & 31;
    const int col0 = blk * 128 + lane * 4;
    const void *const *pd = reinterpret_cast<const void *const *>(down_table);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int s = 0; s < n_expert; ++s) {
        const int64_t pair = (int64_t)tok * n_expert + s;
        const int e = selected[pair];
        const __half *svh_d = reinterpret_cast<const __half *>(pd[2 * (size_t)e + 1]) + mid_dim;
        const float4 z4 = *reinterpret_cast<const float4 *>(down_z + pair * out_dim + col0);
        float z[4] = {z4.x, z4.y, z4.z, z4.w};
        exl3_dev_had128(z);
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] += z[j] * __half2float(svh_d[col0 + j]);
    }
    bool bad = false;
#pragma unroll
    for (int j = 0; j < 4; ++j) bad |= !isfinite(acc[j]);
    if (bad) atomicCAS(nf_flag, 0u, nf_code);
    *reinterpret_cast<float4 *>(out + (size_t)tok * out_dim + col0) = make_float4(acc[0], acc[1], acc[2], acc[3]);
}

/* The row block: a PERFORMANCE choice (every R is bit-identical).  Few
 * assignments (a decode step's handful) want one assignment per CTA so the
 * grid fills the SMs; a prefill chunk's thousands want the widest block so
 * an expert's weights stream once per 16 rows. */
static int exl3_rows_per_block(int64_t n_assign) {
    if (n_assign <= 96) return 1;
    if (n_assign <= 768) return 4;
    return kMaxR;
}

template <bool PAIR, int K2, int R>
static void exl3_gemv_kernel_launch(const dim3 &grid, const dim3 &block, cudaStream_t stream,
                                    const void *gt, const void *ut, const block_mx_act_mmq *a,
                                    const int32_t *ids_dst, const int32_t *expert_bounds,
                                    float *og, float *ou, int M, int K, int n_assign, int E) {
    exl3_moe_gemv_kernel<PAIR, K2, R><<<grid, block, 0, stream>>>(gt, ut, a, ids_dst, expert_bounds, og, ou, M, K, n_assign, E);
}

template <bool PAIR, int K2>
static void exl3_gemv_dispatch_r(int R, const dim3 &grid, const dim3 &block, cudaStream_t stream,
                                 const void *gt, const void *ut, const block_mx_act_mmq *a,
                                 const int32_t *ids_dst, const int32_t *expert_bounds,
                                 float *og, float *ou, int M, int K, int n_assign, int E) {
    switch (R) {
    case 1:  exl3_moe_gemv_kernel_r1<PAIR, K2><<<grid, block, 0, stream>>>(gt, ut, a, ids_dst, expert_bounds, og, ou, M, K, n_assign, E); break;
    case 4:  exl3_gemv_kernel_launch<PAIR, K2, 4>(grid, block, stream, gt, ut, a, ids_dst, expert_bounds, og, ou, M, K, n_assign, E); break;
    default: exl3_gemv_kernel_launch<PAIR, K2, kMaxR>(grid, block, stream, gt, ut, a, ids_dst, expert_bounds, og, ou, M, K, n_assign, E); break;
    }
}

template <bool PAIR>
int exl3_gemv_launch(const void *gt, const void *ut, int k2, const void *act,
                     const int32_t *ids_dst, const int32_t *expert_bounds,
                     float *og, float *ou, int M, int K, int64_t n_assign, int E,
                     int rows_per_block, cudaStream_t stream) {
    const char *tag = PAIR ? "ds4_exl3_moe_gemv_pair_launch" : "ds4_exl3_moe_gemv_single_launch";
    if (!gt || (PAIR && !ut) || !act || !ids_dst || !expert_bounds || !og || (PAIR && !ou) ||
        M <= 0 || K <= 0 || n_assign <= 0 || E <= 0) {
        fprintf(stderr, "%s: null pointer or bad shape\n", tag);
        return -1;
    }
    if (M % kRows || K % kChunk || K > kMaxK || n_assign > INT32_MAX) {
        fprintf(stderr, "%s: shape M=%d K=%d n_assign=%lld outside the arm's contract "
                        "(M %% 32, K %% 256, K <= %d) -- refusing\n",
                tag, M, K, (long long)n_assign, kMaxK);
        return -1;
    }
    const int R = rows_per_block > 0 ? rows_per_block : exl3_rows_per_block(n_assign);
    if (R != 1 && R != 4 && R != kMaxR) {
        fprintf(stderr, "%s: rows per block %d has no instance (1, 4, %d) -- refusing\n", tag, R, kMaxR);
        return -1;
    }
    const dim3 grid((unsigned)(M / kRows), (unsigned)((n_assign + R - 1) / R), 1);
    const dim3 block(kRows, kWarps, 1);
    const block_mx_act_mmq *a = (const block_mx_act_mmq *)act;
    switch (k2) {
    case 4: exl3_gemv_dispatch_r<PAIR, 4>(R, grid, block, stream, gt, ut, a, ids_dst, expert_bounds, og, ou, M, K, (int)n_assign, E); break;
    case 5: exl3_gemv_dispatch_r<PAIR, 5>(R, grid, block, stream, gt, ut, a, ids_dst, expert_bounds, og, ou, M, K, (int)n_assign, E); break;
    case 6: exl3_gemv_dispatch_r<PAIR, 6>(R, grid, block, stream, gt, ut, a, ids_dst, expert_bounds, og, ou, M, K, (int)n_assign, E); break;
    default:
        fprintf(stderr, "%s: rate k2=%d has no instance (2, 2.5, 3) -- refusing\n", tag, k2);
        return -1;
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

} // namespace

bool ds4_exl3_gemv_rate_supported(int k2) { return k2 == 4 || k2 == 5 || k2 == 6; }

int ds4_exl3_moe_gemv_pair_launch(const void *gate_table, const void *up_table, int k2, const void *act,
                                  const int32_t *ids_dst, const int32_t *expert_bounds,
                                  float *out_gate, float *out_up, int M, int K, int64_t n_assign,
                                  int n_experts, cudaStream_t stream) {
    return exl3_gemv_launch<true>(gate_table, up_table, k2, act, ids_dst, expert_bounds,
                                  out_gate, out_up, M, K, n_assign, n_experts, 0, stream);
}

int ds4_exl3_moe_gemv_single_launch(const void *table, int k2, const void *act,
                                    const int32_t *ids_dst, const int32_t *expert_bounds,
                                    float *out, int M, int K, int64_t n_assign,
                                    int n_experts, cudaStream_t stream) {
    return exl3_gemv_launch<false>(table, nullptr, k2, act, ids_dst, expert_bounds,
                                   out, nullptr, M, K, n_assign, n_experts, 0, stream);
}

int ds4_exl3_moe_gemv_pair_launch_rows(const void *gate_table, const void *up_table, int k2, const void *act,
                                       const int32_t *ids_dst, const int32_t *expert_bounds,
                                       float *out_gate, float *out_up, int M, int K, int64_t n_assign,
                                       int n_experts, int rows_per_block, cudaStream_t stream) {
    return exl3_gemv_launch<true>(gate_table, up_table, k2, act, ids_dst, expert_bounds,
                                  out_gate, out_up, M, K, n_assign, n_experts, rows_per_block, stream);
}

int ds4_exl3_moe_fold_launch(const float *gate_z, const float *up_z, const int32_t *selected,
                             const float *weights, const void *gate_table, const void *up_table,
                             const void *down_table, int in_dim, int mid_dim, int64_t pairs,
                             float clamp, void *mid_q, void *mid_sf, int mid_kbp, cudaStream_t stream) {
    if (!gate_z || !up_z || !selected || !weights || !gate_table || !up_table || !down_table ||
        !mid_q || !mid_sf || pairs <= 0 || in_dim <= 0 || mid_dim <= 0 || mid_dim % EXL3_HAD_BLOCK) {
        fprintf(stderr, "ds4_exl3_moe_fold_launch: null pointer or bad shape (mid_dim %% 128)\n");
        return -1;
    }
    const int64_t tasks = pairs * (mid_dim >> 7);
    const unsigned blocks = (unsigned)((tasks + 7) / 8);
    exl3_moe_fold_kernel<<<blocks, 256, 0, stream>>>(
        gate_z, up_z, selected, weights, gate_table, up_table, down_table, in_dim, mid_dim, pairs,
        clamp, (__nv_fp8_e4m3 *)mid_q, (unsigned char *)mid_sf, mid_kbp);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_exl3_moe_fold_launch: launch failed: %s\n", cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

int ds4_exl3_moe_sum_launch(float *out, const float *down_z, const int32_t *selected,
                            const void *down_table, int mid_dim, int out_dim, int n_expert,
                            int n_tokens, uint32_t *nf_flag, uint32_t nf_code, cudaStream_t stream) {
    if (!out || !down_z || !selected || !down_table || !nf_flag || mid_dim <= 0 || out_dim <= 0 ||
        out_dim % EXL3_HAD_BLOCK || n_expert <= 0 || n_tokens <= 0) {
        fprintf(stderr, "ds4_exl3_moe_sum_launch: null pointer or bad shape (out_dim %% 128)\n");
        return -1;
    }
    const int64_t tasks = (int64_t)n_tokens * (out_dim >> 7);
    const unsigned blocks = (unsigned)((tasks + 7) / 8);
    exl3_moe_sum_kernel<<<blocks, 256, 0, stream>>>(
        out, down_z, selected, down_table, mid_dim, out_dim, n_expert, n_tokens, nf_flag, nf_code);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_exl3_moe_sum_launch: launch failed: %s\n", cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

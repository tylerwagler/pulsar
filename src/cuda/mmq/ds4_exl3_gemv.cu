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
// activation is read in the format its producer emitted (exl3_moe_act, rule
// 3): a family's bf16 rows, or the E4M3 slot with its ue8m0 plane, each
// assignment's row addressed through ids_src -- nothing gathers or re-stages
// it first.  Assignments outside the expert-sorted schedule [expert_bounds[0],
// expert_bounds[E]) are not computed: a pick < 0 (another rank's expert, the
// router's NaN path) sorts below expert 0 and has no row (before task #44 such
// a CTA ran expert 0 on row 0 and scattered into pair 0's output).  The
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
//     most 3K+16 bits, <= 31 for the rates here) -- exllamav3's dq4 pattern.
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
#include "ds4_exl3_dev.cuh"
#include "cuda/pulsar_cuda_mx.cuh"
#include "engine/exl3_trellis.h"

#include <cstdio>

namespace {

constexpr int kRows   = 32;    ///< output rows per CTA = one warp's lanes = two n-tiles
constexpr int kWarps  = 8;     ///< K split
constexpr int kMaxK   = 5120;  ///< the activation row (V4.1 n_embd)
constexpr int kChunk  = 256;   ///< k staged per pass: two Hadamard blocks, 16 k-tiles = 2 per warp;
                               ///< the last pass of a K % 256 == 128 row (Qwen's expert down, 640)
                               ///< is one block, 8 k-tiles = 1 per warp -- the same k order
constexpr int kMaxR   = 16;    ///< the widest row block

/* What an instance computes (DS4_EXL3_* in the header): DOWN = one projection
 * of an input the fold already rotated; PAIR = gate and up from two slices,
 * each rotating the input by its own suh (DeepSeek's split stacks); FUSED = one
 * projection that rotates the input by its suh (Qwen's fused gate_up, whose
 * output rows are gate | up). */
constexpr int kDown = EXL3_ARM_DOWN, kPair = EXL3_ARM_PAIR, kFused = EXL3_ARM_GATE_UP_FUSED;
/* ---------------------------------------------------------------------- */
/* the GEMV                                                                */

/* The single-row instance (R = 1): the step-3 body, whole-vector staging and
 * one Hadamard pass, no per-chunk barriers -- 50 us for one assignment where
 * the chunked multi-row body pays 68.  Bit-identical to the multi-row form
 * (the gate proves R = 1 / 4 / 16 equal), kept because a decode step is one
 * assignment per CTA and the barriers are its whole difference. */
template <int MODE, int K2, int FMT>
__global__ void __launch_bounds__(kRows * kWarps)
exl3_moe_gemv_kernel_r1(const void *__restrict__ gate_table,
                     const void *__restrict__ up_table,
                     const void *__restrict__ act,          /* FMT: bf16 rows or the E4M3 slot [rows][K] */
                     const uint8_t *__restrict__ act_sf,    /* the E4M3 slot's ue8m0 plane (pulsar_mx_sfoff) */
                     int act_kbp,
                     const int32_t *__restrict__ ids_dst,
                     const int32_t *__restrict__ ids_src,
                     const int32_t *__restrict__ expert_bounds,
                     float *__restrict__ out_gate,
                     float *__restrict__ out_up,
                     int M, int K, int n_assign, int E) {
    constexpr bool PAIR = MODE == kPair, ROT = MODE != kDown;
    constexpr bool A8 = FMT == EXL3_MOE_ACT_E4M3_SLOT;
    constexpr int W32 = exl3dev::Rate<K2>::words32;
    /* ids_src[col] is the SOURCE row (the token for gate / up, the pair for a down); ids_dst[col] is the
     * flat (token * n_slots + slot) OUTPUT row the scatter below writes to.  Confusing the two reads the
     * wrong token for every slot past the first and yields plausible garbage. */
    const __nv_bfloat16 *__restrict__ xb = reinterpret_cast<const __nv_bfloat16 *>(act);
    const uint8_t *__restrict__ xq = reinterpret_cast<const uint8_t *>(act);
    /* the whole-vector staging, sized by the launch to (PAIR ? 2 : 1) x K floats (L251: a static kMaxK
     * buffer held the kernel to 4 CTAs per SM at Qwen's K = 2560 / 640) */
    extern __shared__ __align__(16) float s_dyn[];
    float *s_x[2] = {s_dyn, s_dyn + (PAIR ? K : 0)};
    __shared__ float s_red[kWarps][kRows][2];
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
        /* outside the schedule [expert_bounds[0], expert_bounds[E]) -- a pick < 0 (another rank's expert, the
         * router's NaN path) sorts below expert 0 -- there is nothing to compute */
        s_expert = col >= expert_bounds[0] && col < expert_bounds[E] ? lo : -1;
    }
    __syncthreads();
    const int expert = s_expert;
    if (expert < 0) return;
    const size_t src = (size_t)ids_src[col];

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
    if constexpr (!A8) {
        /* bf16 (the Qwen family): every thread stages runs of 8 with 16-byte loads of x and of the suh
         * rows -- the per-group form below had a third of the threads doing 32 serial 2-byte loads each,
         * with stores 32 floats apart (one bank per warp).  Each element is the same v * suh product. */
        const __nv_bfloat16 *xr = xb + src * (size_t)K;
        for (int i = tid; i < (K >> 3); i += kRows * kWarps) {
            const int k0 = i * 8;
            const uint4 xw = *reinterpret_cast<const uint4 *>(xr + k0);
            const __nv_bfloat16 *xv = reinterpret_cast<const __nv_bfloat16 *>(&xw);
            if constexpr (ROT) {
                const uint4 gw = *reinterpret_cast<const uint4 *>(sg + k0);
                const __half *gv = reinterpret_cast<const __half *>(&gw);
                uint4 uw = gw;
                if constexpr (PAIR) uw = *reinterpret_cast<const uint4 *>(su + k0);
                const __half *uv = reinterpret_cast<const __half *>(&uw);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float v = __bfloat162float(xv[j]);
                    s_x[0][k0 + j] = v * __half2float(gv[j]);
                    if constexpr (PAIR) s_x[1][k0 + j] = v * __half2float(uv[j]);
                }
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) s_x[0][k0 + j] = __bfloat162float(xv[j]);
            }
        }
    }
    for (int i = tid; A8 && i < n_k128 * 4; i += kRows * kWarps) {
        const int blk = i >> 2, grp = i & 3;
        const int k0 = blk * 128 + grp * 32;
        /* the E4M3 slot (the DeepSeek family): the row's per-32 ue8m0 byte folds in here as 2^(byte - 127) */
        const float sc = exp2f((float)act_sf[pulsar_mx_sfoff((int)src, k0 >> 5, act_kbp)] - 127.0f);
        const uint8_t *q = xq + src * (size_t)K + (size_t)k0;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const float v = exl3dev::e4m3_to_f32(q[j]) * sc;
            if constexpr (PAIR) {
                s_x[0][k0 + j] = v * __half2float(sg[k0 + j]);
                s_x[1][k0 + j] = v * __half2float(su[k0 + j]);
            } else if constexpr (ROT) {
                s_x[0][k0 + j] = v * __half2float(sg[k0 + j]);
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
            exl3dev::had128(v);
            v4[0] = v[0]; v4[1] = v[1]; v4[2] = v[2]; v4[3] = v[3];
        }
        __syncthreads();
    } else if constexpr (ROT) {
        /* H128 per 128-block of the one projection */
        for (int t = warp; t < n_k128; t += kWarps) {
            float *v4 = s_x[0] + t * 128 + lane * 4;
            float v[4] = {v4[0], v4[1], v4[2], v4[3]};
            exl3dev::had128(v);
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
            exl3dev::run4<K2>(wg, p0, stg[a]);
            if constexpr (PAIR) exl3dev::run4<K2>(wu, p0, stu[a]);
        }
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            /* run a: positions p0..p0+3 are rows 2a, 2a+1, 2a+8, 2a+9 */
            const int r[4] = {2 * a, 2 * a + 1, 2 * a + 8, 2 * a + 9};
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                acc_g = fmaf(exl3dev::mul1(stg[a][b]), xg[r[b]], acc_g);
                if constexpr (PAIR) acc_u = fmaf(exl3dev::mul1(stu[a][b]), xu[r[b]], acc_u);
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

template <int MODE, int K2, int R, int FMT>
__global__ void __launch_bounds__(kRows * kWarps)
exl3_moe_gemv_kernel(const void *__restrict__ gate_table,
                     const void *__restrict__ up_table,
                     const void *__restrict__ act,          /* as in kernel r1 */
                     const uint8_t *__restrict__ act_sf,
                     int act_kbp,
                     const int32_t *__restrict__ ids_dst,
                     const int32_t *__restrict__ ids_src,
                     const int32_t *__restrict__ expert_bounds,
                     float *__restrict__ out_gate,
                     float *__restrict__ out_up,
                     int M, int K, int n_assign, int E) {
    constexpr bool PAIR = MODE == kPair, ROT = MODE != kDown;
    constexpr bool A8 = FMT == EXL3_MOE_ACT_E4M3_SLOT;
    constexpr int W32 = exl3dev::Rate<K2>::words32;
    const __nv_bfloat16 *__restrict__ xb = reinterpret_cast<const __nv_bfloat16 *>(act);
    const uint8_t *__restrict__ xq = reinterpret_cast<const uint8_t *>(act);
    constexpr int NV = PAIR ? 2 : 1;
    __shared__ __align__(16) float s_buf[GemvSmem<PAIR, R>::floats];
    __shared__ int s_expert[R];

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * kRows + lane;
    const int row  = blockIdx.x * kRows + lane;      /* output column n of W (in, out) */
    /* this block's assignments, clipped to the schedule [expert_bounds[0], expert_bounds[E]): a pick < 0
     * (another rank's expert, the router's NaN path) sorts below expert 0 and has no row to compute */
    const int col0 = max((int)blockIdx.y * R, expert_bounds[0]);
    const int nrow = min((int)blockIdx.y * R + R, min(n_assign, expert_bounds[E])) - col0;
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
    const int n_chunk = (K + kChunk - 1) / kChunk;
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
            const int nblk = min(2, (K - k0) >> 7);           /* Hadamard blocks in this pass: 2, or 1 at a 128 tail */
            /* stage this chunk of each row's activation as f32 -- for gate/up
             * already multiplied by each projection's suh.  block i of the row
             * holds k in [128 i, 128 i + 128) as 4 groups of 32 E4M3 under one
             * ue8m0 byte each; a chunk is 2 blocks = 8 groups per row. */
            __syncthreads();                                   /* the previous pass's readers are done */
            for (int i = tid; i < nr * 8; i += kRows * kWarps) {
                const int r = i >> 3, g8 = i & 7;
                if ((g8 >> 2) >= nblk) continue;               /* past a 128 tail: nothing to stage */
                const int blk = (k0 >> 7) + (g8 >> 2), grp = g8 & 3;
                const int arow = col0 + r0 + r;
                const size_t src = (size_t)ids_src[arow];
                const float sc = A8 ? exp2f((float)act_sf[pulsar_mx_sfoff((int)src, (blk * 128 + grp * 32) >> 5, act_kbp)] - 127.0f)
                                    : 0.0f;
                const uint8_t *q = A8 ? xq + src * (size_t)K + (size_t)(blk * 128 + grp * 32) : nullptr;
                const int kk = g8 * 32;                        /* offset inside the chunk */
                float *xg = s_buf + (0 * R + r) * kChunk + kk;
                float *xu = PAIR ? s_buf + (1 * R + r) * kChunk + kk : nullptr;
#pragma unroll
                for (int j = 0; j < 32; ++j) {
                    const float v = A8
                        ? (exl3dev::e4m3_to_f32(q[j]) * sc)
                        : __bfloat162float(xb[src * (size_t)K + (size_t)k0 + (size_t)kk + (size_t)j]);
                    if constexpr (PAIR) {
                        xg[j] = v * __half2float(sg[k0 + kk + j]);
                        xu[j] = v * __half2float(su[k0 + kk + j]);
                    } else if constexpr (ROT) {
                        xg[j] = v * __half2float(sg[k0 + kk + j]);
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
                    if (blk >= nblk) continue;
                    float *v4 = s_buf + (v * R + r) * kChunk + blk * 128 + lane * 4;
                    float x[4] = {v4[0], v4[1], v4[2], v4[3]};
                    exl3dev::had128(x);
                    v4[0] = x[0]; v4[1] = x[1]; v4[2] = x[2]; v4[3] = x[3];
                }
                __syncthreads();
            } else if constexpr (ROT) {
                /* H128 per 128-block, per row, of the one projection */
                for (int t = warp; t < nr * 2; t += kWarps) {
                    const int blk = t & 1, r = t >> 1;
                    if (blk >= nblk) continue;
                    float *v4 = s_buf + r * kChunk + blk * 128 + lane * 4;
                    float x[4] = {v4[0], v4[1], v4[2], v4[3]};
                    exl3dev::had128(x);
                    v4[0] = x[0]; v4[1] = x[1]; v4[2] = x[2]; v4[3] = x[3];
                }
                __syncthreads();
            }
            if (!row_ok) continue;
            /* this warp's two k-tiles of the chunk: the same k order a row alone takes */
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                if (half >= nblk) break;                       /* a 128 tail is one k-tile per warp */
                const int kt = (k0 >> 4) + half * kWarps + warp;
                const int kk = (half * kWarps + warp) * 16;    /* the tile's k inside the chunk */
                const uint32_t *wg = tg + ((size_t)kt * (size_t)ntn + (size_t)nt) * W32;
                const uint32_t *wu = PAIR ? tu + ((size_t)kt * (size_t)ntn + (size_t)nt) * W32 : nullptr;
                uint32_t stg[4][4], stu[4][4];
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    const int p0 = 32 * (c & 7) + 8 * a + 4 * (c >> 3);
                    exl3dev::run4<K2>(wg, p0, stg[a]);
                    if constexpr (PAIR) exl3dev::run4<K2>(wu, p0, stu[a]);
                }
                /* decode once, apply to every row of the run */
                float wgv[16], wuv[16];
#pragma unroll
                for (int a = 0; a < 4; ++a)
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        wgv[a * 4 + b] = exl3dev::mul1(stg[a][b]);
                        if constexpr (PAIR) wuv[a * 4 + b] = exl3dev::mul1(stu[a][b]);
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

/* the riding shared expert's SwiGLU: warps of 128 elements after the routed fold's */
static int64_t shared_swiglu_tasks(const exl3_moe_shared_swiglu *sh) {
    return sh ? ((int64_t)sh->rows * sh->mid + 127) / 128 : 0;
}
static bool shared_swiglu_ok(const exl3_moe_shared_swiglu *sh) {
    if (sh && (!sh->g || !sh->u || !sh->h || sh->rows <= 0 || sh->mid <= 0)) {
        fprintf(stderr, "ds4_exl3_moe_fold: a shared-expert SwiGLU with a null plane or no rows -- refusing\n");
        return false;
    }
    return true;
}

/* One warp per (pair, 128-block of mid); lane l holds mid columns 4l..4l+3
 * of the block.  The four values per lane make one 32-element MX group an
 * EIGHT-lane group, as in moe_mmq_swiglu_fold_v4_kernel. */
__global__ void __launch_bounds__(256)
exl3_moe_fold_kernel(const float *__restrict__ gate_z,
                     const float *__restrict__ up_z,
                     int z_stride,
                     const int32_t *__restrict__ selected,
                     const float *__restrict__ weights,
                     const void *__restrict__ gate_table,
                     const void *__restrict__ up_table,
                     int up_svh_off,
                     const void *__restrict__ down_table,
                     int in_dim, int mid_dim, int64_t pairs, float clamp,
                     __nv_fp8_e4m3 *__restrict__ mid_q,
                     unsigned char *__restrict__ mid_sf,
                     __nv_bfloat16 *__restrict__ mid_bf16,   /* the format the down arm reads */
                     int mid_kbp,
                     const exl3_moe_shared_swiglu sh) {
    const int64_t task = ((int64_t)blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int n_blk = mid_dim >> 7;
    const int64_t n_tasks = pairs * n_blk;
    if (task >= n_tasks) {
        /* a riding shared expert's SwiGLU, emitted bf16 for its down (the producer rule): 128 elements of the
         * [rows][mid] planes per warp, the plain silu(g) * u of the family's shared expert */
        const int64_t i0 = (task - n_tasks) * 128 + (int64_t)(threadIdx.x & 31) * 4;
        const int64_t n = (int64_t)sh.rows * sh.mid;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (i0 + j < n) sh.h[i0 + j] = __float2bfloat16(pulsar_swiglu_elem(sh.g[i0 + j], sh.u[i0 + j], 1.0f, 0.0f));
        return;
    }
    const int64_t pair = task / n_blk;
    const int blk = (int)(task - pair * n_blk);
    const int lane = threadIdx.x & 31;
    const int e = selected[pair];
    if (e < 0) return;                                /* another rank's expert (L266 EP) or the router's NaN path */
    const void *const *pg = reinterpret_cast<const void *const *>(gate_table);
    const void *const *pu = reinterpret_cast<const void *const *>(up_table);
    const void *const *pd = reinterpret_cast<const void *const *>(down_table);
    /* gate/up scales: suh[in_dim] | svh[mid_dim]; down scales: suh[mid_dim] | svh[out] */
    const __half *svh_g = reinterpret_cast<const __half *>(pg[2 * (size_t)e + 1]) + in_dim;
    const __half *svh_u = reinterpret_cast<const __half *>(pu[2 * (size_t)e + 1]) + up_svh_off;
    const __half *suh_d = reinterpret_cast<const __half *>(pd[2 * (size_t)e + 1]);
    const int col0 = blk * 128 + lane * 4;
    const float wv = weights[pair];
    const float4 g4 = *reinterpret_cast<const float4 *>(gate_z + pair * z_stride + col0);
    const float4 u4 = *reinterpret_cast<const float4 *>(up_z + pair * z_stride + col0);
    float g[4] = {g4.x, g4.y, g4.z, g4.w}, u[4] = {u4.x, u4.y, u4.z, u4.w};
    exl3dev::had128(g);
    exl3dev::had128(u);
    float t[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float yg = g[j] * __half2float(svh_g[col0 + j]);
        const float yu = u[j] * __half2float(svh_u[col0 + j]);
        t[j] = pulsar_swiglu_elem(yg, yu, wv, clamp) * __half2float(suh_d[col0 + j]);
    }
    exl3dev::had128(t);
    float a = fmaxf(fmaxf(fabsf(t[0]), fabsf(t[1])), fmaxf(fabsf(t[2]), fabsf(t[3])));
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    const int se = pulsar_mx_shared_exp(a);
    if (mid_q) {
        __nv_fp8_e4m3 *dst = mid_q + (size_t)pair * mid_dim + col0;
        dst[0] = pulsar_mx_encode(t[0], se);
        dst[1] = pulsar_mx_encode(t[1], se);
        dst[2] = pulsar_mx_encode(t[2], se);
        dst[3] = pulsar_mx_encode(t[3], se);
        if ((lane & 7) == 0)
            mid_sf[pulsar_mx_sfoff((int)pair, col0 >> 5, mid_kbp)] = pulsar_mx_scale_byte(se);
    }
    /* L251 / ac69748f: bf16 carries its own exponent, so there is no per-32 block and no scale byte.
     * The fold's output is an op-internal Linear activation, so it follows the family's rule. */
    if (mid_bf16) {
        __nv_bfloat16 *d16 = mid_bf16 + (size_t)pair * mid_dim + col0;
        d16[0] = __float2bfloat16(t[0]);
        d16[1] = __float2bfloat16(t[1]);
        d16[2] = __float2bfloat16(t[2]);
        d16[3] = __float2bfloat16(t[3]);
    }
}

/* Slots whose loads the sum issues together (L284 #7): a decode step's sum is a
 * few warps, latency-bound on its chain of selected -> table -> partial loads
 * when they wait one slot at a time. */
constexpr int kSumBatch = 8;

/* One warp per (token, 128-block of out); slots summed in order, their loads
 * issued kSumBatch slots at a time. */
__global__ void __launch_bounds__(256)
exl3_moe_sum_kernel(float *__restrict__ out,
                    const float *__restrict__ down_z,
                    const int32_t *__restrict__ selected,
                    const void *__restrict__ down_table,
                    int mid_dim, int out_dim, int n_expert, int n_tokens,
                    uint32_t *__restrict__ nf_flag, uint32_t nf_code,
                    const float *__restrict__ shared_y, const float *__restrict__ shared_scale) {
    const int64_t task = ((int64_t)blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int n_blk = out_dim >> 7;
    if (task >= (int64_t)n_tokens * n_blk) return;
    const int tok = (int)(task / n_blk);
    const int blk = (int)(task - (int64_t)tok * n_blk);
    const int lane = threadIdx.x & 31;
    const int col0 = blk * 128 + lane * 4;
    const void *const *pd = reinterpret_cast<const void *const *>(down_table);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int s0 = 0; s0 < n_expert; s0 += kSumBatch) {
        int ex[kSumBatch];
#pragma unroll
        for (int b = 0; b < kSumBatch; ++b)
            ex[b] = s0 + b < n_expert ? selected[(int64_t)tok * n_expert + s0 + b] : -1;
        float4 z4[kSumBatch];
        __half sv[kSumBatch][4];
#pragma unroll
        for (int b = 0; b < kSumBatch; ++b) {
            if (ex[b] < 0) continue;                  /* past the slots, or another rank's expert (L266 EP): its partial is that rank's */
            const __half *svh_d = reinterpret_cast<const __half *>(pd[2 * (size_t)ex[b] + 1]) + mid_dim;
            z4[b] = *reinterpret_cast<const float4 *>(down_z + ((int64_t)tok * n_expert + s0 + b) * out_dim + col0);
#pragma unroll
            for (int j = 0; j < 4; ++j) sv[b][j] = svh_d[col0 + j];
        }
#pragma unroll
        for (int b = 0; b < kSumBatch; ++b) {
            if (ex[b] < 0) continue;
            float z[4] = {z4[b].x, z4[b].y, z4[b].z, z4[b].w};
            exl3dev::had128(z);
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[j] += z[j] * __half2float(sv[b][j]);
        }
    }
    bool bad = false;
#pragma unroll
    for (int j = 0; j < 4; ++j) bad |= !isfinite(acc[j]);
    if (bad) atomicCAS(nf_flag, 0u, nf_code);
    if (shared_y) {
        /* a riding shared expert's output, added after the routed sum: out + scale * y (scale 1 when absent) */
        const float4 y4 = *reinterpret_cast<const float4 *>(shared_y + (size_t)tok * out_dim + col0);
        const float y[4] = {y4.x, y4.y, y4.z, y4.w};
        const float sc = shared_scale ? shared_scale[tok] : 1.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float s = sc * y[j];
            acc[j] = acc[j] + s;
        }
    }
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

/* One instance's launch: the argument list every instance shares. */
struct gemv_args {
    const void *gt, *ut, *x;
    const uint8_t *sf;
    int kbp;
    const int32_t *ids_dst, *ids_src, *bounds;
    float *og, *ou;
    int M, K, n_assign, E;
};

template <int MODE, int K2, int FMT>
static void exl3_gemv_dispatch_r(int R, const dim3 &grid, const dim3 &block, cudaStream_t stream, const gemv_args &a) {
    switch (R) {
    case 1: {
        const size_t smem = (size_t)(MODE == kPair ? 2 : 1) * (size_t)a.K * sizeof(float);
        static bool attr = false;                    /* per instance; idempotent */
        if (!attr) {
            cudaFuncSetAttribute(exl3_moe_gemv_kernel_r1<MODE, K2, FMT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)((MODE == kPair ? 2 : 1) * kMaxK * sizeof(float)));
            attr = true;
        }
        exl3_moe_gemv_kernel_r1<MODE, K2, FMT><<<grid, block, smem, stream>>>(
            a.gt, a.ut, a.x, a.sf, a.kbp, a.ids_dst, a.ids_src, a.bounds, a.og, a.ou, a.M, a.K, a.n_assign, a.E);
        break;
    }
    case 4:
        exl3_moe_gemv_kernel<MODE, K2, 4, FMT><<<grid, block, 0, stream>>>(
            a.gt, a.ut, a.x, a.sf, a.kbp, a.ids_dst, a.ids_src, a.bounds, a.og, a.ou, a.M, a.K, a.n_assign, a.E);
        break;
    default:
        exl3_moe_gemv_kernel<MODE, K2, kMaxR, FMT><<<grid, block, 0, stream>>>(
            a.gt, a.ut, a.x, a.sf, a.kbp, a.ids_dst, a.ids_src, a.bounds, a.og, a.ou, a.M, a.K, a.n_assign, a.E);
        break;
    }
}

/* The rates each kind instantiates: exl3_arm_has_rate (exl3_trellis.h), the
 * one table the binders read too. */
constexpr bool kind_has_rate(int mode, int k2) { return exl3_arm_has_rate(mode, k2); }

template <int MODE, int FMT>
static bool exl3_gemv_dispatch_k2(int k2, int R, const dim3 &grid, const dim3 &block, cudaStream_t stream,
                                  const gemv_args &a) {
    if constexpr (kind_has_rate(MODE, 4)) if (k2 == 4) { exl3_gemv_dispatch_r<MODE, 4, FMT>(R, grid, block, stream, a); return true; }
    if constexpr (kind_has_rate(MODE, 5)) if (k2 == 5) { exl3_gemv_dispatch_r<MODE, 5, FMT>(R, grid, block, stream, a); return true; }
    if constexpr (kind_has_rate(MODE, 6)) if (k2 == 6) { exl3_gemv_dispatch_r<MODE, 6, FMT>(R, grid, block, stream, a); return true; }
    if constexpr (kind_has_rate(MODE, 8)) if (k2 == 8) { exl3_gemv_dispatch_r<MODE, 8, FMT>(R, grid, block, stream, a); return true; }
    if constexpr (kind_has_rate(MODE, 10)) if (k2 == 10) { exl3_gemv_dispatch_r<MODE, 10, FMT>(R, grid, block, stream, a); return true; }
    if constexpr (kind_has_rate(MODE, 12)) if (k2 == 12) { exl3_gemv_dispatch_r<MODE, 12, FMT>(R, grid, block, stream, a); return true; }
    return false;
}

template <int MODE>
static bool exl3_gemv_dispatch_fmt(int fmt, int k2, int R, const dim3 &grid, const dim3 &block, cudaStream_t stream,
                                   const gemv_args &a) {
    return fmt == EXL3_MOE_ACT_E4M3_SLOT
        ? exl3_gemv_dispatch_k2<MODE, EXL3_MOE_ACT_E4M3_SLOT>(k2, R, grid, block, stream, a)
        : exl3_gemv_dispatch_k2<MODE, EXL3_MOE_ACT_BF16_ROWS>(k2, R, grid, block, stream, a);
}

} // namespace

bool ds4_exl3_gemv_rate_supported(int kind, int k2) {
    return (kind == kDown || kind == kPair || kind == kFused) && exl3_arm_has_rate(kind, k2);
}

int ds4_exl3_moe_gemv_launch(int kind, const void *table, const void *up_table, int k2, const exl3_moe_act &act,
                             const int32_t *ids_dst, const int32_t *ids_src, const int32_t *expert_bounds,
                             float *out, float *out_up, int M, int K, int64_t n_assign, int n_experts,
                             int rows_per_block, cudaStream_t stream) {
    const char *tag = "ds4_exl3_moe_gemv_launch";
    const bool pair = kind == kPair;
    const bool e4m3 = act.format == EXL3_MOE_ACT_E4M3_SLOT;
    if ((kind != kDown && kind != kPair && kind != kFused) || !table || (pair && !up_table) || !act.x ||
        (e4m3 && (!act.sf || act.kbp <= 0)) || (!e4m3 && act.format != EXL3_MOE_ACT_BF16_ROWS) ||
        !ids_dst || !ids_src || !expert_bounds || !out || (pair && !out_up) ||
        M <= 0 || K <= 0 || n_assign <= 0 || n_experts <= 0) {
        fprintf(stderr, "%s: kind %d, act format %d: a null pointer or a bad shape -- refusing\n", tag, kind, act.format);
        return -1;
    }
    if (M % kRows || K % EXL3_HAD_BLOCK || K > kMaxK || n_assign > INT32_MAX) {
        fprintf(stderr, "%s: shape M=%d K=%d n_assign=%lld outside the arm's contract "
                        "(M %% 32, K %% 128, K <= %d) -- refusing\n",
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
    const gemv_args a{table, up_table, act.x, static_cast<const uint8_t *>(act.sf), act.kbp, ids_dst, ids_src,
                      expert_bounds, out, out_up, M, K, (int)n_assign, n_experts};
    const bool ok = kind == kPair  ? exl3_gemv_dispatch_fmt<kPair>(act.format, k2, R, grid, block, stream, a)
                  : kind == kFused ? exl3_gemv_dispatch_fmt<kFused>(act.format, k2, R, grid, block, stream, a)
                                   : exl3_gemv_dispatch_fmt<kDown>(act.format, k2, R, grid, block, stream, a);
    if (!ok) {
        fprintf(stderr, "%s: rate k2=%d has no instance for kind %d (exl3_arm_has_rate) -- refusing\n", tag, k2, kind);
        return -1;
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

int ds4_exl3_moe_fold_launch(const float *gate_z, const float *up_z, const int32_t *selected,
                             const float *weights, const void *gate_table, const void *up_table,
                             const void *down_table, int in_dim, int mid_dim, int64_t pairs,
                             float clamp, void *mid_q, void *mid_sf, int mid_kbp, cudaStream_t stream, void *mid_bf16,
                             const exl3_moe_shared_swiglu *shared) {
    if (!shared_swiglu_ok(shared)) return -1;
    if (!gate_z || !up_z || !selected || !weights || !gate_table || !up_table || !down_table ||
        (!mid_q || !mid_sf) && !mid_bf16 || pairs <= 0 || in_dim <= 0 || mid_dim <= 0 || mid_dim % EXL3_HAD_BLOCK) {
        fprintf(stderr, "ds4_exl3_moe_fold_launch: null pointer or bad shape (mid_dim %% 128)\n");
        return -1;
    }
    const int64_t tasks = pairs * (mid_dim >> 7) + shared_swiglu_tasks(shared);
    const unsigned blocks = (unsigned)((tasks + 7) / 8);
    exl3_moe_fold_kernel<<<blocks, 256, 0, stream>>>(
        gate_z, up_z, mid_dim, selected, weights, gate_table, up_table, in_dim, down_table, in_dim, mid_dim, pairs,
        clamp, (__nv_fp8_e4m3 *)mid_q, (unsigned char *)mid_sf, (__nv_bfloat16 *)mid_bf16, mid_kbp,
        shared ? *shared : exl3_moe_shared_swiglu{});
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_exl3_moe_fold_launch: launch failed: %s\n", cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

int ds4_exl3_moe_fold_fused_launch(const float *gate_up_z, const int32_t *selected, const float *weights,
                                   const void *gate_up_table, const void *down_table, int in_dim, int mid_dim,
                                   int64_t pairs, float clamp, void *mid_q, void *mid_sf, int mid_kbp,
                                   cudaStream_t stream, void *mid_bf16, const exl3_moe_shared_swiglu *shared) {
    if (!shared_swiglu_ok(shared)) return -1;
    if (!gate_up_z || !selected || !weights || !gate_up_table || !down_table || (!mid_q || !mid_sf) && !mid_bf16 || pairs <= 0 ||
        in_dim <= 0 || mid_dim <= 0 || mid_dim % EXL3_HAD_BLOCK) {
        fprintf(stderr, "ds4_exl3_moe_fold_fused_launch: null pointer or bad shape (mid_dim %% 128)\n");
        return -1;
    }
    /* gate = rows 0..mid-1 of the fused output, up = rows mid..2 mid-1; the scales
     * plane is suh [in_dim] | svh [2 mid], so up's svh starts at in_dim + mid */
    const int64_t tasks = pairs * (mid_dim >> 7) + shared_swiglu_tasks(shared);
    const unsigned blocks = (unsigned)((tasks + 7) / 8);
    exl3_moe_fold_kernel<<<blocks, 256, 0, stream>>>(
        gate_up_z, gate_up_z + mid_dim, 2 * mid_dim, selected, weights, gate_up_table, gate_up_table, in_dim + mid_dim,
        down_table, in_dim, mid_dim, pairs, clamp, (__nv_fp8_e4m3 *)mid_q, (unsigned char *)mid_sf, (__nv_bfloat16 *)mid_bf16, mid_kbp,
        shared ? *shared : exl3_moe_shared_swiglu{});
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_exl3_moe_fold_fused_launch: launch failed: %s\n", cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

int ds4_exl3_moe_sum_launch(float *out, const float *down_z, const int32_t *selected,
                            const void *down_table, int mid_dim, int out_dim, int n_expert,
                            int n_tokens, uint32_t *nf_flag, uint32_t nf_code, cudaStream_t stream,
                            const float *shared_y, const float *shared_scale) {
    if (!out || !down_z || !selected || !down_table || !nf_flag || mid_dim <= 0 || out_dim <= 0 ||
        out_dim % EXL3_HAD_BLOCK || n_expert <= 0 || n_tokens <= 0) {
        fprintf(stderr, "ds4_exl3_moe_sum_launch: null pointer or bad shape (out_dim %% 128)\n");
        return -1;
    }
    const int64_t tasks = (int64_t)n_tokens * (out_dim >> 7);
    const unsigned blocks = (unsigned)((tasks + 7) / 8);
    exl3_moe_sum_kernel<<<blocks, 256, 0, stream>>>(
        out, down_z, selected, down_table, mid_dim, out_dim, n_expert, n_tokens, nf_flag, nf_code, shared_y, shared_scale);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_exl3_moe_sum_launch: launch failed: %s\n", cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

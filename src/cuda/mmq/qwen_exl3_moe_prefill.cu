// SPDX-License-Identifier: MIT
// The Qwen routed-expert EXL3 arm for PROMPT CHUNKS (L251).
//
// The decode arm (ds4_exl3_gemv.cu) is a GEMV: f32 CUDA-core FMAs, at most 16
// same-expert assignments sharing one weight decode.  Run over a 4096-row
// prompt it was 47% of the prefill (6.6 s of 13.9 s, nsys on sparky) -- a
// decode kernel doing a GEMM's work.  This is the GEMM: a grouped tensor-core
// kernel over the same expert-sorted schedule (mm_ids_helper's ids_src1 /
// ids_dst / expert_bounds) that writes the same UNROTATED z the fold and the
// sum already consume, so nothing downstream changes.
//
// Work: one CTA per (64-output tile, block of up to 64 consecutive
// assignments of ONE expert); qwen_moe_plan_kernel turns expert_bounds into
// that list on the device.  Per 128-k Hadamard block the CTA:
//   A  stages its assignments' activation rows -- bf16, times the expert's
//      suh and rotated by H128 when the projection rotates its input (the
//      fused gate_up; the down's input arrives rotated from the fold) -- with
//      a power-of-two prescale per (row, block) and a split into fp16 hi + lo
//      planes (~22 significant bits), the dense arm's operand;
//   B  decodes the block's 8 k-tiles x 4 n16-tiles of the expert's trellis
//      ONCE into shared memory as mma.m16n8k16 B registers (the tile order is
//      the B-fragment order; mul1_pair is the GEMV's codebook, bit for bit);
//   MMA 8 warps (4 m16 x 2 n32): per k-tile two MMAs (hi, lo) into a fresh
//      f32 fragment, added into the accumulator scaled by the row's inverse
//      prescale -- a fresh fragment because the tensor core's own accumulate
//      is not IEEE f32.
// A prefilled assignment and a decoded one agree to rounding, not to the bit:
// the two arms sum in different orders (Tyler 2026-09-29, "Don't use a decode
// kernel for prefill").  Within this arm a row's arithmetic depends only on
// its own activation and its expert's weights.

#include "ds4_exl3_gemv.cuh"
#include "ds4_exl3_dev.cuh"
#include "engine/exl3_trellis.h"

#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kBM      = 64;          ///< assignments per CTA: four m16 tiles
constexpr int kBN      = 64;          ///< outputs per CTA: four n16 tiles
constexpr int kThreads = 256;         ///< 8 warps
constexpr int kApad    = 128 + 8;     ///< halves per staged A row: 16-byte rows, conflict-free ldmatrix
constexpr int kHiExp   = 14;          ///< the prescale puts a block's largest |value| just under 2^14
constexpr int kPlanCap = 1 << 15;     ///< work items the device plan holds (4096 rows x 10 slots / 64 + 512 << this)
constexpr int kMaxE    = 1024;        ///< experts the one-block plan kernel scans

struct Smem {
    __half a_hi[kBM][kApad];
    __half a_lo[kBM][kApad];
    uint2  b[8][4][2][32];            ///< [k-tile][n16 tile][n8 half][lane] = the lane's {b0, b1}
    float  inv[kBM];
    int    src[kBM];
};

__device__ __forceinline__ float pow2f(int e) { return __int_as_float((127 + e) << 23); }   /* e in [-126, 127] */

__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&a)[4], const __half *p) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(s));
}

/* The work list: expert e's assignments [bounds[e], bounds[e+1]) cut into
 * blocks of kBM, in expert order.  One block of E threads, an inclusive
 * Hillis-Steele scan over the per-expert block counts. */
__global__ void qwen_moe_plan_kernel(const int32_t *__restrict__ bounds, int E, int2 *__restrict__ work,
                                     int *__restrict__ n_work) {
    __shared__ int s[kMaxE];
    const int e = threadIdx.x;
    int mb = 0;
    if (e < E) mb = (bounds[e + 1] - bounds[e] + kBM - 1) / kBM;
    s[e] = mb;
    __syncthreads();
    for (int off = 1; off < (int)blockDim.x; off <<= 1) {
        const int v = e >= off ? s[e - off] : 0;
        __syncthreads();
        s[e] += v;
        __syncthreads();
    }
    const int start = s[e] - mb;
    for (int j = 0; j < mb; ++j)
        if (start + j < kPlanCap) work[start + j] = make_int2(e, bounds[e] + j * kBM);
    if (e == (int)blockDim.x - 1) *n_work = s[e] < kPlanCap ? s[e] : kPlanCap;
}

template <int K2, bool ROT>
__global__ void __launch_bounds__(kThreads)
qwen_moe_prefill_kernel(const void *__restrict__ table, const __nv_bfloat16 *__restrict__ x,
                        const int32_t *__restrict__ ids_dst, const int32_t *__restrict__ ids_src,
                        const int32_t *__restrict__ bounds, const int2 *__restrict__ work,
                        const int *__restrict__ n_work, float *__restrict__ out, int M, int K) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem &S = *reinterpret_cast<Smem *>(smem_raw);
    if ((int)blockIdx.y >= *n_work) return;
    const int2 wk = work[blockIdx.y];
    const int e = wk.x, a0 = wk.y;
    const int mr = min(kBM, bounds[e + 1] - a0);
    const int n0 = blockIdx.x * kBN;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr int W32 = exl3dev::Rate<K2>::words32;
    const void *const *pt = reinterpret_cast<const void *const *>(table);
    const uint32_t *tr = reinterpret_cast<const uint32_t *>(pt[2 * (size_t)e]);
    const __half *suh = reinterpret_cast<const __half *>(pt[2 * (size_t)e + 1]);
    const int ntn = M >> 4;

    if (tid < kBM) S.src[tid] = tid < mr ? ids_src[a0 + tid] : 0;
    int lo[2], hi[2], sh[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) exl3dev::run4_window<K2>(8 * lane + 4 * h, lo[h], hi[h], sh[h]);
    const int wm = warp & 3, wn = warp >> 2;          /* rows 16 wm .., n16 tiles 2 wn, 2 wn + 1 */
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    const int arow = lane & 15, ahalf = lane >> 4;
    float acc[4][4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[q][i] = 0.0f;
    __syncthreads();

    for (int blk = 0; blk < (K >> 7); ++blk) {
        /* A: warp w stages rows w, w + 8, ...; lane l owns k = 4l..4l+3 of the block.  A whole
         * warp takes one row, so had128's shuffles see a full warp even on rows past mr (zeros). */
        for (int r = warp; r < kBM; r += kThreads / 32) {
            float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const int k = blk * 128 + lane * 4;
            if (r < mr) {
                const uint2 xw = *reinterpret_cast<const uint2 *>(x + (size_t)S.src[r] * (size_t)K + (size_t)k);
                const float2 b01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&xw.x));
                const float2 b23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&xw.y));
                v[0] = b01.x; v[1] = b01.y; v[2] = b23.x; v[3] = b23.y;
            }
            if constexpr (ROT) {
                const uint2 sw = *reinterpret_cast<const uint2 *>(suh + k);
                const __half2 s01 = *reinterpret_cast<const __half2 *>(&sw.x);
                const __half2 s23 = *reinterpret_cast<const __half2 *>(&sw.y);
                v[0] *= __low2float(s01); v[1] *= __high2float(s01);
                v[2] *= __low2float(s23); v[3] *= __high2float(s23);
                exl3dev::had128(v);
            }
            float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            int p = amax > 0.0f ? kHiExp - (((__float_as_int(amax) >> 23) & 0xff) - 126) : 0;
            p = max(-100, min(100, p));
            const float up = pow2f(p);
            uint32_t hw[2], lw[2];
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const float x0 = v[2 * q] * up, x1 = v[2 * q + 1] * up;
                const __half2 h2 = __floats2half2_rn(x0, x1);
                const __half2 l2 = __floats2half2_rn(x0 - __low2float(h2), x1 - __high2float(h2));
                hw[q] = *reinterpret_cast<const uint32_t *>(&h2);
                lw[q] = *reinterpret_cast<const uint32_t *>(&l2);
            }
            *reinterpret_cast<uint2 *>(&S.a_hi[r][lane * 4]) = make_uint2(hw[0], hw[1]);
            *reinterpret_cast<uint2 *>(&S.a_lo[r][lane * 4]) = make_uint2(lw[0], lw[1]);
            if (lane == 0) S.inv[r] = pow2f(-p);
        }
        /* B: warp w decodes k-tile w of the block for the CTA's four n16 tiles, once */
        {
            const int kt = blk * 8 + warp;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t *t = tr + ((size_t)kt * (size_t)ntn + (size_t)((n0 >> 4) + j)) * W32;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t st[4];
                    exl3dev::run4_split<K2>(EXL3_RUN4_JOIN(t[lo[h]], t[hi[h]], sh[h]), 8 * lane + 4 * h, st);
                    S.b[warp][j][h][lane] = make_uint2(exl3dev::mul1_pair(st[0], st[1]),
                                                       exl3dev::mul1_pair(st[2], st[3]));
                }
            }
        }
        __syncthreads();
        const float ia = S.inv[wm * 16 + g], ib = S.inv[wm * 16 + g + 8];
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            uint32_t ahi[4], alo[4];
            ldmatrix_x4(ahi, &S.a_hi[wm * 16 + arow][kt * 16 + ahalf * 8]);
            ldmatrix_x4(alo, &S.a_lo[wm * 16 + arow][kt * 16 + ahalf * 8]);
#pragma unroll
            for (int jj = 0; jj < 2; ++jj)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint2 bb = S.b[kt][2 * wn + jj][h][lane];
                    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    mma_16816(d, ahi, bb.x, bb.y);
                    mma_16816(d, alo, bb.x, bb.y);
                    float *ac = acc[jj * 2 + h];
                    ac[0] = fmaf(d[0], ia, ac[0]);
                    ac[1] = fmaf(d[1], ia, ac[1]);
                    ac[2] = fmaf(d[2], ib, ac[2]);
                    ac[3] = fmaf(d[3], ib, ac[3]);
                }
        }
        __syncthreads();
    }
    /* acc[jj*2+h][i]: row 16 wm + g + 8 (i >> 1), column n0 + 16 (2 wn + jj) + 8 h + c2 + (i & 1) */
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = wm * 16 + g + 8 * (i >> 1);
            if (r < mr) {
                const int col = n0 + 16 * (2 * wn + (q >> 1)) + 8 * (q & 1) + c2 + (i & 1);
                out[(size_t)ids_dst[a0 + r] * (size_t)M + (size_t)col] = acc[q][i];
            }
        }
}

template <int K2, bool ROT>
bool launch_k2(const dim3 &grid, cudaStream_t stream, const void *table, const __nv_bfloat16 *x,
               const int32_t *ids_dst, const int32_t *ids_src, const int32_t *bounds, const int2 *work,
               const int *n_work, float *out, int M, int K) {
    static bool attr = false;
    if (!attr) {
        if (cudaFuncSetAttribute(qwen_moe_prefill_kernel<K2, ROT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)sizeof(Smem)) != cudaSuccess) return false;
        attr = true;
    }
    qwen_moe_prefill_kernel<K2, ROT><<<grid, kThreads, sizeof(Smem), stream>>>(table, x, ids_dst, ids_src, bounds, work,
                                                                               n_work, out, M, K);
    return true;
}

} // namespace

/* The plan buffer: one small device allocation for the process, made on first use. */
static int2 *g_plan_work = nullptr;
static int  *g_plan_n    = nullptr;

int qwen_exl3_moe_prefill_launch(const void *table, int k2, bool rotate_input, const void *x_bf16,
                                 const int32_t *ids_dst, const int32_t *ids_src, const int32_t *expert_bounds,
                                 float *out, int M, int K, int64_t n_assign, int n_experts, cudaStream_t stream) {
    const char *tag = "qwen_exl3_moe_prefill_launch";
    if (!table || !x_bf16 || !ids_dst || !ids_src || !expert_bounds || !out || n_assign <= 0 ||
        n_experts <= 0 || n_experts > kMaxE || M % kBN || K % EXL3_HAD_BLOCK) {
        fprintf(stderr, "%s: bad arguments (M=%d K=%d n_assign=%lld E=%d; M %% %d, K %% 128, E <= %d) -- refusing\n",
                tag, M, K, (long long)n_assign, n_experts, kBN, kMaxE);
        return -1;
    }
    const int64_t cap = n_assign / kBM + n_experts + 1;
    if (cap > kPlanCap) {
        fprintf(stderr, "%s: %lld assignments over %d experts need %lld work items > %d -- refusing\n",
                tag, (long long)n_assign, n_experts, (long long)cap, kPlanCap);
        return -1;
    }
    if (!g_plan_work) {
        if (cudaMalloc(&g_plan_work, (size_t)kPlanCap * sizeof(int2)) != cudaSuccess ||
            cudaMalloc(&g_plan_n, sizeof(int)) != cudaSuccess) {
            fprintf(stderr, "%s: plan buffer allocation failed -- refusing\n", tag);
            return -1;
        }
    }
    int threads = 32;
    while (threads < n_experts) threads <<= 1;
    qwen_moe_plan_kernel<<<1, threads, 0, stream>>>(expert_bounds, n_experts, g_plan_work, g_plan_n);
    const dim3 grid((unsigned)(M / kBN), (unsigned)cap, 1);
    const __nv_bfloat16 *x = static_cast<const __nv_bfloat16 *>(x_bf16);
    bool ok;
    if (rotate_input) {
        if (k2 == 8)       ok = launch_k2<8, true>(grid, stream, table, x, ids_dst, ids_src, expert_bounds, g_plan_work, g_plan_n, out, M, K);
        else if (k2 == 10) ok = launch_k2<10, true>(grid, stream, table, x, ids_dst, ids_src, expert_bounds, g_plan_work, g_plan_n, out, M, K);
        else { fprintf(stderr, "%s: fused gate_up rate k2=%d has no prefill instance (8, 10) -- refusing\n", tag, k2); return -1; }
    } else {
        if (k2 == 8)       ok = launch_k2<8, false>(grid, stream, table, x, ids_dst, ids_src, expert_bounds, g_plan_work, g_plan_n, out, M, K);
        else if (k2 == 10) ok = launch_k2<10, false>(grid, stream, table, x, ids_dst, ids_src, expert_bounds, g_plan_work, g_plan_n, out, M, K);
        else { fprintf(stderr, "%s: down rate k2=%d has no prefill instance (8, 10) -- refusing\n", tag, k2); return -1; }
    }
    const cudaError_t err = cudaGetLastError();
    if (!ok || err != cudaSuccess) {
        fprintf(stderr, "%s: launch failed: %s\n", tag, ok ? cudaGetErrorString(err) : "shared-memory attribute");
        return -3;
    }
    return 0;
}

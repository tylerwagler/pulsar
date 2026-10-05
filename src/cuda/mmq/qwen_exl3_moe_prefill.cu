// SPDX-License-Identifier: MIT
// The Qwen routed-expert EXL3 arm for PROMPT CHUNKS (L251).
//
// The decode arm (ds4_exl3_gemv.cu) is a GEMV: f32 CUDA-core FMAs, at most 16 same-expert
// assignments per weight decode -- on a 4096-row prompt it was 47% of the prefill.  This is the
// GEMM over the same expert-sorted schedule (mm_ids_helper's ids_src1 / ids_dst / expert_bounds),
// writing the same UNROTATED z the fold and the sum consume, so nothing downstream changes.
//
//   PREP  once per (assignment, 128-block): the token's bf16 row, times the expert's suh and
//         rotated by H128 when the projection rotates its input (the fused gate_up; the down's
//         input arrives rotated from the fold), a power-of-two prescale per block, rounded to ONE
//         fp16 plane: 11 significant bits relative to the block's largest value -- 8x finer than
//         the bf16 row it came from carries (exllamav3's and TensorFold's operand is the same one
//         fp16 plane; a second "lo" plane doubled the MMAs for precision the input does not have).
//   GEMM  one CTA per (128-output tile, <= 64 consecutive assignments of ONE expert; the plan
//         kernel turns expert_bounds into that list): per 128-k block the A tiles go to shared
//         memory by cp.async while each of the 8 warps decodes its n16 tile's 8 k-tiles of the
//         expert's trellis ONCE into mma B registers (mul1_pair, the GEMV's codebook bit for bit),
//         then applies them to the item's live m16 tiles only;
//         then one MMA per n8 tile into a fresh f32 fragment, added into the accumulator
//         scaled by the (row, block) inverse prescale -- a fresh fragment because the tensor
//         core's own accumulate is not IEEE f32.
//
// The first version staged A INSIDE the GEMM, so every assignment's row was gathered, rotated and
// split once per 64-output tile (20x for the fused gate_up) with no load pipelining: 11 TFLOP/s.
// A prefilled assignment and a decoded one agree to rounding, not to the bit (Tyler 2026-09-29,
// "Don't use a decode kernel for prefill"); graded by qwen_s4_gate D and the NLL suite.

#include "ds4_exl3_gemv.cuh"
#include "ds4_exl3_dev.cuh"
#include "engine/exl3_trellis.h"

#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kBM      = 64;          ///< assignments per CTA: four m16 tiles
constexpr int kBN      = 128;         ///< outputs per CTA: eight n16 tiles
constexpr int kThreads = 256;         ///< 8 warps: one n16 tile each
constexpr int kApad    = 128 + 8;     ///< halves per staged A row: 16-byte rows, conflict-free ldmatrix
constexpr int kHiExp   = 14;          ///< the prescale puts a block's largest |value| just under 2^14
constexpr int kPlanCap = 1 << 15;     ///< work items the device plan holds (4096 rows x 10 slots / 64 + 512 << this)
constexpr int kMaxE    = 1024;        ///< experts the one-block plan kernel scans

struct Smem {
    __half a_hi[2][kBM][kApad];       ///< A of block b in buffer b & 1: block b + 1 loads while b computes
    float  inv[2][kBM];
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

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem, int src_bytes) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(src_bytes));
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


/* PREP: one warp per (assignment, 128-block); lane l owns k = 4l..4l+3 of the block. */
template <bool ROT>
__global__ void __launch_bounds__(256)
qwen_moe_prep_kernel(const void *__restrict__ table, const __nv_bfloat16 *__restrict__ x,
                     const int32_t *__restrict__ ids_src, const int32_t *__restrict__ bounds, int E,
                     int n_assign, int K, __half *__restrict__ phi,
                     float *__restrict__ pinv) {
    const int64_t task = (int64_t)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int nblk = K >> 7;
    if (task >= (int64_t)n_assign * nblk) return;
    const int a = (int)(task / nblk), blk = (int)(task % nblk), lane = threadIdx.x & 31;
    const int k = blk * 128 + lane * 4;
    const uint2 xw = *reinterpret_cast<const uint2 *>(x + (size_t)ids_src[a] * (size_t)K + (size_t)k);
    const float2 b01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&xw.x));
    const float2 b23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&xw.y));
    float v[4] = {b01.x, b01.y, b23.x, b23.y};
    if constexpr (ROT) {
        int lo = 0, hi = E - 1;                     /* the assignment's expert: bounds is E+1 ascending */
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (bounds[mid] <= a) lo = mid; else hi = mid - 1;
        }
        const void *const *pt = reinterpret_cast<const void *const *>(table);
        const __half *suh = reinterpret_cast<const __half *>(pt[2 * (size_t)lo + 1]);
        const uint2 sw = *reinterpret_cast<const uint2 *>(suh + k);
        const __half2 s01 = *reinterpret_cast<const __half2 *>(&sw.x), s23 = *reinterpret_cast<const __half2 *>(&sw.y);
        v[0] *= __low2float(s01); v[1] *= __high2float(s01); v[2] *= __low2float(s23); v[3] *= __high2float(s23);
        exl3dev::had128(v);
    }
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    int p = amax > 0.0f ? kHiExp - (((__float_as_int(amax) >> 23) & 0xff) - 126) : 0;
    p = max(-100, min(100, p));
    const float up = pow2f(p);
    uint32_t hw[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const __half2 h2 = __floats2half2_rn(v[2 * q] * up, v[2 * q + 1] * up);
        hw[q] = *reinterpret_cast<const uint32_t *>(&h2);
    }
    const size_t off = (size_t)a * K + k;
    *reinterpret_cast<uint2 *>(phi + off) = make_uint2(hw[0], hw[1]);
    if (lane == 0) pinv[(size_t)a * nblk + blk] = pow2f(-p);
}

/* GEMM: CTA = (128 outputs, one work item); warp w = n16 tile w (outputs n0 + 16 w ..), all of the item's
 * LIVE m16 tiles.  Each warp decodes its own 8 k-tiles of the block into registers (no shared B: every
 * trellis tile is still decoded once per CTA) and applies each B fragment to every live m16 tile, whose
 * A comes from the staged planes by ldmatrix.  An item of mr rows runs ceil(mr / 16) m16 tiles, not 4:
 * at 4096 rows an expert averages ~80 assignments, so fixed 64-row tiles ran ~1.6x the MMAs.  The
 * per-(row, column) arithmetic is the 4 x 2 warp layout's exactly (a fresh fragment per k-tile,
 * scaled into the accumulator, blocks and k-tiles in order). */
template <int K2>
__global__ void __launch_bounds__(kThreads, 2)
qwen_moe_gemm_kernel(const void *__restrict__ table, const __half *__restrict__ phi,
                     const float *__restrict__ pinv,
                     const int32_t *__restrict__ ids_dst, const int32_t *__restrict__ bounds,
                     const int2 *__restrict__ work, const int *__restrict__ n_work, float *__restrict__ out,
                     int M, int K) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem &S = *reinterpret_cast<Smem *>(smem_raw);
    if ((int)blockIdx.y >= *n_work) return;
    const int2 wk = work[blockIdx.y];
    const int e = wk.x, a0 = wk.y;
    const int mr = min(kBM, bounds[e + 1] - a0);
    const int mt_n = (mr + 15) >> 4;                   /* live m16 tiles */
    const int n0 = blockIdx.x * kBN, nblk = K >> 7;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr int W32 = exl3dev::Rate<K2>::words32;
    const uint32_t *tr = reinterpret_cast<const uint32_t *>(reinterpret_cast<const void *const *>(table)[2 * (size_t)e]);
    const int ntn = M >> 4, nt = (n0 >> 4) + warp;
    int lo[2], hi[2], sh[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) exl3dev::run4_window<K2>(8 * lane + 4 * h, lo[h], hi[h], sh[h]);
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    const int arow = lane & 15, ahalf = lane >> 4;
    float acc[4][2][4];
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[m][h][i] = 0.0f;

    /* The pipeline: block b + 1's A goes to shared memory by cp.async and its raw trellis words to
     * registers while block b's MMAs run; a k-tile's words are decoded just before its MMAs.  Every
     * accumulator still takes its k-tiles 0..7 of each block in order, so the arithmetic is the
     * unpipelined kernel's (ncu: 66% of cycles had no eligible warp -- the loads were serialised
     * with the MMAs). */
    const int arows = mt_n * 16;
    auto issue_a = [&](int blk, int buf) {
        for (int c = tid; c < arows * 16; c += kThreads) {
            const int r = c >> 4, ch = c & 15;
            const bool real = r < mr;
            const __half *src = phi + (size_t)(a0 + (real ? r : 0)) * K + (size_t)blk * 128 + ch * 8;
            cp_async16(&S.a_hi[buf][r][ch * 8], src, real ? 16 : 0);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        if (tid < kBM) S.inv[buf][tid] = tid < mr ? pinv[(size_t)(a0 + tid) * nblk + blk] : 0.0f;
    };
    auto load_words = [&](int blk, uint32_t (&wd)[8][4]) {
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            const uint32_t *t = tr + ((size_t)(blk * 8 + kt) * (size_t)ntn + (size_t)nt) * W32;
            wd[kt][0] = t[lo[0]]; wd[kt][1] = t[hi[0]];
            wd[kt][2] = t[lo[1]]; wd[kt][3] = t[hi[1]];
        }
    };
    uint32_t wcur[8][4], wnxt[8][4];
    issue_a(0, 0);
    load_words(0, wcur);
    for (int blk = 0; blk < nblk; ++blk) {
        const int buf = blk & 1;
        const bool more = blk + 1 < nblk;
        if (more) {
            issue_a(blk + 1, buf ^ 1);
            load_words(blk + 1, wnxt);
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        }
        __syncthreads();
        float ia[4], ib[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) { ia[m] = S.inv[buf][m * 16 + g]; ib[m] = S.inv[buf][m * 16 + g + 8]; }
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            uint2 bw[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint32_t st[4];
                exl3dev::run4_split<K2>(EXL3_RUN4_JOIN(wcur[kt][2 * h], wcur[kt][2 * h + 1], sh[h]), 8 * lane + 4 * h, st);
                bw[h] = make_uint2(exl3dev::mul1_pair(st[0], st[1]), exl3dev::mul1_pair(st[2], st[3]));
            }
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (m < mt_n) {                        /* warp-uniform */
                    uint32_t ahi[4];
                    ldmatrix_x4(ahi, &S.a_hi[buf][m * 16 + arow][kt * 16 + ahalf * 8]);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        mma_16816(d, ahi, bw[h].x, bw[h].y);
                        float *ac = acc[m][h];
                        ac[0] = fmaf(d[0], ia[m], ac[0]);
                        ac[1] = fmaf(d[1], ia[m], ac[1]);
                        ac[2] = fmaf(d[2], ib[m], ac[2]);
                        ac[3] = fmaf(d[3], ib[m], ac[3]);
                    }
                }
            }
        }
        __syncthreads();                               /* buf is refilled two blocks on */
        if (more) {
#pragma unroll
            for (int kt = 0; kt < 8; ++kt)
#pragma unroll
                for (int i = 0; i < 4; ++i) wcur[kt][i] = wnxt[kt][i];
        }
    }
    /* acc[m][h][i]: row 16 m + g + 8 (i >> 1), column n0 + 16 warp + 8 h + c2 + (i & 1) */
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = m * 16 + g + 8 * (i >> 1);
                if (r < mr) {
                    const int col = n0 + 16 * warp + 8 * h + c2 + (i & 1);
                    out[(size_t)ids_dst[a0 + r] * (size_t)M + (size_t)col] = acc[m][h][i];
                }
            }
}

template <int K2>
bool launch_gemm(const dim3 &grid, cudaStream_t stream, const void *table, const __half *phi,
                 const float *pinv, const int32_t *ids_dst, const int32_t *bounds, const int2 *work,
                 const int *n_work, float *out, int M, int K) {
    static bool attr = false;
    if (!attr) {
        if (cudaFuncSetAttribute(qwen_moe_gemm_kernel<K2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)sizeof(Smem)) != cudaSuccess) return false;
        attr = true;
    }
    qwen_moe_gemm_kernel<K2><<<grid, kThreads, sizeof(Smem), stream>>>(table, phi, pinv, ids_dst, bounds, work,
                                                                       n_work, out, M, K);
    return true;
}

} // namespace

/* The plan list and the prepared planes: device buffers for the process, grown to the largest call. */
static int2 *g_plan_work = nullptr;
static int  *g_plan_n    = nullptr;
static uint8_t *g_planes = nullptr;
static size_t g_planes_bytes = 0;

static size_t up256(size_t b) { return (b + 255u) & ~(size_t)255u; }

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
    if (!exl3_arm_has_rate(EXL3_ARM_MOE_PREFILL, k2)) {
        fprintf(stderr, "%s: rate k2=%d has no prefill instance (K = 3, 4, 5, 6) -- refusing\n", tag, k2);
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
    const int nblk = K >> 7;
    const size_t plane_b = up256((size_t)n_assign * K * sizeof(__half));
    const size_t need = plane_b + up256((size_t)n_assign * nblk * sizeof(float));
    if (need > g_planes_bytes) {
        if (g_planes) cudaFree(g_planes);
        g_planes = nullptr;
        g_planes_bytes = 0;
        if (cudaMalloc(&g_planes, need) != cudaSuccess) {
            fprintf(stderr, "%s: operand planes of %zu bytes refused -- refusing\n", tag, need);
            return -1;
        }
        g_planes_bytes = need;
    }
    __half *phi = reinterpret_cast<__half *>(g_planes);
    float *pinv = reinterpret_cast<float *>(g_planes + plane_b);
    int threads = 32;
    while (threads < n_experts) threads <<= 1;
    qwen_moe_plan_kernel<<<1, threads, 0, stream>>>(expert_bounds, n_experts, g_plan_work, g_plan_n);
    const __nv_bfloat16 *x = static_cast<const __nv_bfloat16 *>(x_bf16);
    const int64_t tasks = n_assign * nblk;
    const unsigned pgrid = (unsigned)((tasks + 7) / 8);
    if (rotate_input)
        qwen_moe_prep_kernel<true><<<pgrid, 256, 0, stream>>>(table, x, ids_src, expert_bounds, n_experts, (int)n_assign, K,
                                                              phi, pinv);
    else
        qwen_moe_prep_kernel<false><<<pgrid, 256, 0, stream>>>(table, x, ids_src, expert_bounds, n_experts, (int)n_assign, K,
                                                               phi, pinv);
    const dim3 grid((unsigned)(M / kBN), (unsigned)cap, 1);
    /* k2 = 6: the MTP layer's experts (turboderp's K = 3); 8 / 10 / 12: the trunk's (4.05 / 3.05 / 6.05 bpw) */
    const bool ok = k2 == 6 ? launch_gemm<6>(grid, stream, table, phi, pinv, ids_dst, expert_bounds, g_plan_work,
                                             g_plan_n, out, M, K)
                  : k2 == 8 ? launch_gemm<8>(grid, stream, table, phi, pinv, ids_dst, expert_bounds, g_plan_work,
                                             g_plan_n, out, M, K)
                  : k2 == 10 ? launch_gemm<10>(grid, stream, table, phi, pinv, ids_dst, expert_bounds, g_plan_work,
                                               g_plan_n, out, M, K)
                             : launch_gemm<12>(grid, stream, table, phi, pinv, ids_dst, expert_bounds, g_plan_work,
                                               g_plan_n, out, M, K);
    const cudaError_t err = cudaGetLastError();
    if (!ok || err != cudaSuccess) {
        fprintf(stderr, "%s: launch failed: %s\n", tag, ok ? cudaGetErrorString(err) : "shared-memory attribute");
        return -3;
    }
    return 0;
}

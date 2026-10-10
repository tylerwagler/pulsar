// SPDX-License-Identifier: MIT
// The EXL3 routed-expert arm for PROMPT CHUNKS, one arm for every family (L251 Qwen; L287 DeepSeek).
//
// The decode arm (ds4_exl3_gemv.cu) is a GEMV: f32 CUDA-core FMAs, at most 16 same-expert
// assignments per weight decode -- on a 4096-row prompt it was 47% of Qwen's prefill, and DeepSeek
// ran every prompt row through it.  This is the GEMM over the same expert-sorted schedule
// (mm_ids_helper's ids_src1 / ids_dst / expert_bounds), writing the same UNROTATED z the fold and the
// sum consume, so nothing downstream changes.  Shapes are the call's (M, K, experts): nothing here is
// a family's.
//
//   PREP  once per (assignment, 128-block): the row in the format its PRODUCER emitted (rule 3) --
//         bf16 rows (Qwen) or the E4M3 slot with its per-32 ue8m0 scales (DeepSeek; e4m3 x 2^e is
//         exact in f32) -- times the expert's suh and rotated by H128 when the projection rotates its
//         input (gate / up, fused or split; the down's input arrives rotated from the fold), a
//         power-of-two prescale per block, rounded to ONE fp16 plane: 11 significant bits relative
//         to the block's largest value -- 8x finer than a bf16 row, and EXACT for an unrotated E4M3
//         value (4 significant bits), so DeepSeek's down reads its slot's values bit for bit.
//         (exllamav3's and TensorFold's operand is the same one fp16 plane; a second "lo" plane
//         doubled the MMAs for precision the input does not have.)
//   GEMM  one CTA per (128-output tile, <= 64 consecutive assignments of ONE expert; the plan
//         kernel turns expert_bounds into that list): per 128-k block the A tiles go to shared
//         memory by cp.async while each of the 8 warps decodes its n16 tile's 8 k-tiles of the
//         expert's trellis ONCE into mma B registers (mul1_pair, the GEMV's codebook bit for bit),
//         then applies them to the item's live m16 tiles only;
//         then one MMA per n8 tile into a fresh f32 fragment, added into the accumulator
//         scaled by the (row, block) inverse prescale -- a fresh fragment because the tensor
//         core's own accumulate is not IEEE f32.
//   The trellis is decoded straight into the mma B fragments (TensorFold's experts design, "D2R"),
//   never to memory: decoding each routed expert ONCE to fp16 and running a cuBLASLt GEMM per expert
//   (the dense prefill arm's shape) writes and re-reads 2 bytes per weight against the trellis's 3/8,
//   and measured 4.0x slower on V4's gate projection at a 4096-token chunk (96 rows per expert) and
//   5.6x on V4.1's (64 rows per expert) -- tests/exl3_moe_prefill_bench, L287.
//
// A row's arithmetic is a function of its own operand row and its expert's weights alone (its own
// prescales, the fixed block / k-tile order, a fresh fragment per k-tile), so its bits never depend
// on its chunk or its batchmates (exl3_moe_prefill_gate proves it).  The first version staged A
// INSIDE the GEMM, so every assignment's row was gathered, rotated and split once per 64-output tile
// (20x for the fused gate_up) with no load pipelining: 11 TFLOP/s.  A prefilled assignment and a
// decoded one agree to rounding, not to the bit (Tyler 2026-09-29, "Don't use a decode kernel for
// prefill"); graded by exl3_moe_prefill_gate, qwen_s4_gate D and the reference gates.

#include "exl3_moe_prefill.cuh"
#include "ds4_exl3_dev.cuh"
#include "cuda/pulsar_cuda_mx.cuh"
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
constexpr int kPlanCap = 1 << 15;     ///< work items one call may plan (8192 rows x 10 slots / 64 + 1024 << this)
constexpr int kMaxE    = 1024;        ///< experts the one-block plan kernel scans

/* The prefill arm reads exactly the rates the routed GEMV arms read (exl3_arm_has_rate is the one table):
 * a prompt chunk can never meet a rate only decode serves, and the arm instantiates nothing no arm reads. */
constexpr bool covers_gemv_rates(int k2 = 0) {
    return k2 > 16 || (exl3_arm_has_rate(EXL3_ARM_MOE_PREFILL, k2) ==
                           (exl3_arm_has_rate(EXL3_ARM_DOWN, k2) || exl3_arm_has_rate(EXL3_ARM_PAIR, k2) ||
                            exl3_arm_has_rate(EXL3_ARM_GATE_UP_FUSED, k2)) &&
                       covers_gemv_rates(k2 + 1));
}
static_assert(covers_gemv_rates(), "EXL3_ARM_MOE_PREFILL's rates must be the routed GEMV arms' rates");

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
 * Hillis-Steele scan over the per-expert block counts.  `cap` is the list's
 * length (the launch sizes it to n_assign / kBM + E + 1, which no routing exceeds). */
__global__ void exl3_moe_plan_kernel(const int32_t *__restrict__ bounds, int E, int2 *__restrict__ work,
                                     int *__restrict__ n_work, int cap) {
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
        if (start + j < cap) work[start + j] = make_int2(e, bounds[e] + j * kBM);
    if (e == (int)blockDim.x - 1) *n_work = s[e] < cap ? s[e] : cap;
}


/* PREP: one warp per (assignment, 128-block); lane l owns k = 4l..4l+3 of the block.  FMT is the
 * producer's activation format (exl3_moe_act_format): the row is read as given, never re-encoded. */
template <int FMT, bool ROT>
__global__ void __launch_bounds__(256)
exl3_moe_prep_kernel(const void *__restrict__ table, const void *__restrict__ xv, const uint8_t *__restrict__ xsf,
                     int kbp, const int32_t *__restrict__ ids_src, const int32_t *__restrict__ bounds, int E,
                     int n_assign, int K, __half *__restrict__ phi, float *__restrict__ pinv) {
    const int64_t task = (int64_t)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int nblk = K >> 7;
    if (task >= (int64_t)n_assign * nblk) return;
    const int a = (int)(task / nblk), blk = (int)(task % nblk), lane = threadIdx.x & 31;
    const int k = blk * 128 + lane * 4;
    const int row = ids_src[a];
    float v[4];
    if constexpr (FMT == EXL3_MOE_ACT_BF16_ROWS) {
        const __nv_bfloat16 *x = static_cast<const __nv_bfloat16 *>(xv);
        const uint2 xw = *reinterpret_cast<const uint2 *>(x + (size_t)row * (size_t)K + (size_t)k);
        const float2 b01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&xw.x));
        const float2 b23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&xw.y));
        v[0] = b01.x; v[1] = b01.y; v[2] = b23.x; v[3] = b23.y;
    } else {
        /* the slot: [row][K] E4M3 and one ue8m0 byte per (row, 32-group); x 2^(byte - 127) by ldexpf,
         * exact (byte 0, an all-zero group's, included) */
        const uchar4 q = *reinterpret_cast<const uchar4 *>(static_cast<const uint8_t *>(xv) + (size_t)row * (size_t)K +
                                                           (size_t)k);
        const int e = (int)xsf[pulsar_mx_sfoff(row, k >> 5, kbp)] - 127;
        v[0] = ldexpf(exl3dev::e4m3_to_f32(q.x), e);
        v[1] = ldexpf(exl3dev::e4m3_to_f32(q.y), e);
        v[2] = ldexpf(exl3dev::e4m3_to_f32(q.z), e);
        v[3] = ldexpf(exl3dev::e4m3_to_f32(q.w), e);
    }
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
exl3_moe_gemm_kernel(const void *__restrict__ table, const __half *__restrict__ phi,
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
        if (cudaFuncSetAttribute(exl3_moe_gemm_kernel<K2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)sizeof(Smem)) != cudaSuccess) return false;
        attr = true;
    }
    exl3_moe_gemm_kernel<K2><<<grid, kThreads, sizeof(Smem), stream>>>(table, phi, pinv, ids_dst, bounds, work,
                                                                       n_work, out, M, K);
    return true;
}

template <int FMT>
void launch_prep(bool rotate, unsigned grid, cudaStream_t stream, const void *table, const exl3_moe_act &act,
                 const int32_t *ids_src, const int32_t *bounds, int E, int n_assign, int K, __half *phi,
                 float *pinv) {
    const uint8_t *sf = static_cast<const uint8_t *>(act.sf);
    if (rotate)
        exl3_moe_prep_kernel<FMT, true><<<grid, 256, 0, stream>>>(table, act.x, sf, act.kbp, ids_src, bounds, E,
                                                                  n_assign, K, phi, pinv);
    else
        exl3_moe_prep_kernel<FMT, false><<<grid, 256, 0, stream>>>(table, act.x, sf, act.kbp, ids_src, bounds, E,
                                                                   n_assign, K, phi, pinv);
}

size_t up256(size_t b) { return (b + 255u) & ~(size_t)255u; }

/* The workspace's three slices, in order: the fp16 plane, the inverse prescales, the work list + its count. */
struct ws_layout {
    size_t plane, inv, work, total;
    int64_t cap;
};
ws_layout layout_for(int64_t n_assign, int K, int n_experts) {
    ws_layout l;
    l.cap = n_assign / kBM + n_experts + 1;
    l.plane = up256((size_t)n_assign * (size_t)K * sizeof(__half));
    l.inv = up256((size_t)n_assign * (size_t)(K >> 7) * sizeof(float));
    l.work = up256((size_t)l.cap * sizeof(int2) + sizeof(int));
    l.total = l.plane + l.inv + l.work;
    return l;
}

} // namespace

size_t exl3_moe_prefill_ws_bytes(int64_t n_assign, int K, int n_experts) {
    if (n_assign <= 0 || K <= 0 || n_experts <= 0) return 0;
    return layout_for(n_assign, K, n_experts).total;
}

int exl3_moe_prefill_launch(const void *table, int k2, bool rotate_input, const exl3_moe_act &act,
                            const int32_t *ids_dst, const int32_t *ids_src, const int32_t *expert_bounds,
                            float *out, int M, int K, int64_t n_assign, int n_experts,
                            void *ws, size_t ws_bytes, cudaStream_t stream) {
    const char *tag = "exl3_moe_prefill_launch";
    const bool e4m3 = act.format == EXL3_MOE_ACT_E4M3_SLOT;
    if (!table || !act.x || !ids_dst || !ids_src || !expert_bounds || !out || !ws || n_assign <= 0 ||
        n_experts <= 0 || n_experts > kMaxE || M <= 0 || M % kBN || K <= 0 || K % EXL3_HAD_BLOCK ||
        (act.format != EXL3_MOE_ACT_BF16_ROWS && !e4m3) || (e4m3 && (!act.sf || act.kbp < K / 32)) ||
        (!e4m3 && act.sf)) {
        fprintf(stderr, "%s: bad arguments (M=%d K=%d n_assign=%lld E=%d act format %d sf=%p kbp=%d; M %% %d, "
                        "K %% 128, E <= %d) -- refusing\n", tag, M, K, (long long)n_assign, n_experts, act.format,
                act.sf, act.kbp, kBN, kMaxE);
        return -1;
    }
    if (!exl3_arm_has_rate(EXL3_ARM_MOE_PREFILL, k2)) {
        fprintf(stderr, "%s: rate k2=%d has no prefill instance -- refusing\n", tag, k2);
        return -1;
    }
    const ws_layout l = layout_for(n_assign, K, n_experts);
    if (l.cap > kPlanCap) {
        fprintf(stderr, "%s: %lld assignments over %d experts need %lld work items > %d -- refusing\n",
                tag, (long long)n_assign, n_experts, (long long)l.cap, kPlanCap);
        return -1;
    }
    if (ws_bytes < l.total || ((uintptr_t)ws & 255u)) {
        fprintf(stderr, "%s: workspace %zu B at %p < %zu B (256-aligned) -- refusing\n", tag, ws_bytes, ws, l.total);
        return -1;
    }
    static bool announced[2] = {false, false};
    if (!announced[e4m3]) {
        announced[e4m3] = true;
        fprintf(stderr, "pulsar: EXL3 MoE prefill arm = fp16 tensor-core GEMM over %s rows (first call: K=%g, "
                        "%d x %d, %lld assignments over %d experts)\n", e4m3 ? "E4M3-slot" : "bf16",
                k2 / 2.0, K, M, (long long)n_assign, n_experts);
    }
    uint8_t *base = static_cast<uint8_t *>(ws);
    __half *phi = reinterpret_cast<__half *>(base);
    float *pinv = reinterpret_cast<float *>(base + l.plane);
    int2 *work = reinterpret_cast<int2 *>(base + l.plane + l.inv);
    int *n_work = reinterpret_cast<int *>(work + l.cap);
    int threads = 32;
    while (threads < n_experts) threads <<= 1;
    exl3_moe_plan_kernel<<<1, threads, 0, stream>>>(expert_bounds, n_experts, work, n_work, (int)l.cap);
    const int64_t tasks = n_assign * (K >> 7);
    const unsigned pgrid = (unsigned)((tasks + 7) / 8);
    if (e4m3)
        launch_prep<EXL3_MOE_ACT_E4M3_SLOT>(rotate_input, pgrid, stream, table, act, ids_src, expert_bounds,
                                            n_experts, (int)n_assign, K, phi, pinv);
    else
        launch_prep<EXL3_MOE_ACT_BF16_ROWS>(rotate_input, pgrid, stream, table, act, ids_src, expert_bounds,
                                            n_experts, (int)n_assign, K, phi, pinv);
    const dim3 grid((unsigned)(M / kBN), (unsigned)l.cap, 1);
    /* the rates exl3_arm_has_rate(EXL3_ARM_MOE_PREFILL) names: every rate a routed GEMV arm reads, so a
     * prompt chunk never meets a rate only decode serves -- k2 = 4 / 5 / 6 DeepSeek's (K2, 2.5, 3) and Qwen's
     * MTP experts; 8 / 10 / 12 Qwen's trunk (4.05 / 3.05 / 6.05 bpw) */
    bool ok = false;
    switch (k2) {
    case 4:  ok = launch_gemm<4>(grid, stream, table, phi, pinv, ids_dst, expert_bounds, work, n_work, out, M, K); break;
    case 5:  ok = launch_gemm<5>(grid, stream, table, phi, pinv, ids_dst, expert_bounds, work, n_work, out, M, K); break;
    case 6:  ok = launch_gemm<6>(grid, stream, table, phi, pinv, ids_dst, expert_bounds, work, n_work, out, M, K); break;
    case 8:  ok = launch_gemm<8>(grid, stream, table, phi, pinv, ids_dst, expert_bounds, work, n_work, out, M, K); break;
    case 10: ok = launch_gemm<10>(grid, stream, table, phi, pinv, ids_dst, expert_bounds, work, n_work, out, M, K); break;
    case 12: ok = launch_gemm<12>(grid, stream, table, phi, pinv, ids_dst, expert_bounds, work, n_work, out, M, K); break;
    default: break;
    }
    const cudaError_t err = cudaGetLastError();
    if (!ok || err != cudaSuccess) {
        fprintf(stderr, "%s: launch failed: %s\n", tag, ok ? cudaGetErrorString(err) : "shared-memory attribute / rate");
        return -3;
    }
    return 0;
}

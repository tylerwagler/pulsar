/* Attention on the tensor cores, fp16 operands, f32 accumulate.
 *
 * WHY fp16 AND NOT SOMETHING SMALLER.  Measured, not assumed
 * (tests/attn_precision_fidelity.cc, real activations, f64 reference):
 *   format  rate          top-1 attention position preserved
 *   f32     14.5 TMAC/s   100%      <- what the FMA-pipe kernel does today
 *   fp16    62.9 (4.3x)   ~100%     <- this kernel
 *   bf16    62.9 (4.3x)   99.87%    <- same speed, 3-8x the error: dominated
 *   fp8     125.0 (8.6x)  96.8%     <- worst single head off by 100%
 * bf16 is strictly dominated by fp16 -- identical throughput for more error --
 * so the only real question was fp16 vs fp8, and fp8 moves where 2-3% of
 * head-token pairs attend.  Nothing saturates in any format (max|q| 17.6,
 * max|kv| 5.65), so this is a resolution choice, not a range one.
 *
 * WHAT THIS REPLACES.  attention_static_mixed_heads8_online_kernel runs at
 * pipe_tensor 0%, pipe_fma 40%, pipe_lsu 56%: one warp per head, one KV row at
 * a time, and a 32-lane shuffle reduction to produce ONE score out of 16 FMAs
 * of useful work -- while all 8 warps in the block re-read the same 2 KB latent
 * row out of shared memory.  Both costs are structural, and both disappear when
 * the reduction becomes the MMA's k dimension and the operands are reused
 * across the MMA's m dimension.
 *
 * MLA, so K AND V ARE THE SAME TENSOR.  The launcher passes raw_kv as both, and
 * there is one latent 512-wide vector per position.  The tile is therefore
 * staged ONCE and used in both orientations: as the B operand transposed for
 * the scores, and as the B operand directly for the value sum.
 *
 * DECOMPOSITION.  Block = 16 heads (the MMA's M) x 8 warps.
 *   phase 1  S[16][16] = Q[16][512] . KV[16][512]^T
 *            2 n-tiles x 32 k-steps = 64 MMAs.  With 8 warps that is 2 n-tiles
 *            x a 4-way split of k, so each warp owns 8 k-steps and the four
 *            partials are summed through smem.  Splitting k (not n) is what
 *            lets every warp hold its Q fragments in REGISTERS for the whole
 *            kernel -- 8 k-steps x 4 regs = 32 -- since Q never changes.
 *   phase 2  online softmax over the tile, f32, per head.
 *   phase 3  O[16][512] += P[16][16] . KV[16][512]
 *            warp w owns dims 64w..64w+63: 8 n-tiles x 1 k-step = 8 MMAs, and
 *            its accumulator is 16 heads x 64 dims = 32 regs/lane.
 *
 * The output accumulator is what sets the shape.  O is heads x 512 in f32, so
 * splitting the 512 across the 8 warps is the only way it fits in registers at
 * all: 16x512/(8x32) = 32 per lane.  Everything else follows from that.
 *
 * Fragment layout is the one verified in tests/attn_mma_probe.cu, not the one
 * remembered from the ISA doc -- the block-scaled work turned up two layout
 * details that were silent-wrong-answer traps, and a layout bug here produces
 * plausible attention, not an error.
 */
#include "pulsar_cuda_internal.h"

#include <cstdio>
#include <cuda_fp16.h>

#define AF16_HEADS     16u        /* heads per block = the MMA's M */
#define AF16_DIM      512u        /* head_dim this kernel is specialised for */
#define AF16_ROWS      16u        /* KV rows per tile = the MMA's K in phase 3 */
#define AF16_WARPS      8u
#define AF16_THREADS  (AF16_WARPS * 32u)
#define AF16_DPW      (AF16_DIM / AF16_WARPS)      /* 64 output dims per warp */
#define AF16_KSTEPS   (AF16_DIM / 16u)             /* 32 k-steps for the scores */
#define AF16_KPW      (AF16_KSTEPS / 4u)           /* 8 k-steps per warp (4-way) */
/* +8 halves of padding: the phase-1 B fragment walks DOWN a column of the tile
 * (fixed dim, varying row), so an unpadded 512-wide row stride would put every
 * lane in the same bank. */
#define AF16_KVSTRIDE (AF16_DIM + 8u)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define PULSAR_ATTN_F16_MMA 1
#else
#define PULSAR_ATTN_F16_MMA 0
#endif

#if PULSAR_ATTN_F16_MMA
__device__ __forceinline__ static void af16_mma(
        float &d0, float &d1, float &d2, float &d3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

__device__ __forceinline__ static uint32_t af16_pack(float lo, float hi) {
    return (uint32_t)__half_as_ushort(__float2half(lo)) |
           ((uint32_t)__half_as_ushort(__float2half(hi)) << 16);
}

/* raw_kv may be stored f16 or f32; comp_kv is always f32. */
__device__ __forceinline__ static float af16_kv(const float *raw, int raw_f16,
                                                uint64_t base, uint32_t d) {
    if (raw_f16) return __half2float(((const __half *)raw)[base + d]);
    return raw[base + d];
}

__global__ __launch_bounds__(AF16_THREADS, 2)
static void attn_f16_kernel(
        float *__restrict__ heads,            /* [n_tokens][n_head][512] */
        const float *__restrict__ sinks,      /* [n_head] */
        const float *__restrict__ q,          /* [n_tokens][n_head][512] */
        const float *__restrict__ raw_kv,     /* [n_tokens][512], f16 or f32 */
        const float *__restrict__ comp_kv,    /* [n_comp][512] f32, may alias raw */
        uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio,
        uint32_t n_head, int raw_f16) {
#if !PULSAR_ATTN_F16_MMA
    (void)heads; (void)sinks; (void)q; (void)raw_kv; (void)comp_kv;
    (void)n_tokens; (void)n_comp; (void)window; (void)ratio; (void)n_head; (void)raw_f16;
#else
    const uint32_t t = blockIdx.x;
    const uint32_t hbase = blockIdx.y * AF16_HEADS;
    if (t >= n_tokens || hbase >= n_head) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t g = lane >> 2u;          /* fragment groupID 0..7 */
    const uint32_t tg = lane & 3u;          /* thread-in-group 0..3 */

    const uint32_t raw_count = (window != 0u && t + 1u > window) ? window : t + 1u;
    const uint32_t raw_start = t + 1u - raw_count;
    uint32_t comp_count = 0u;
    if (n_comp != 0u && ratio != 0u) {
        comp_count = (t + 1u) / ratio;
        if (comp_count > n_comp) comp_count = n_comp;
    }
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)AF16_DIM);

    __shared__ __half sKV[AF16_ROWS * AF16_KVSTRIDE];
    __shared__ float  sPart[4][AF16_HEADS][AF16_ROWS];   /* phase-1 k-split partials */
    __shared__ float  sS[AF16_HEADS][AF16_ROWS];
    __shared__ __half sP[AF16_HEADS][AF16_ROWS];
    __shared__ float  sCorr[AF16_HEADS];
    __shared__ float  sM[AF16_HEADS], sL[AF16_HEADS];

    /* ---- Q fragments, once, into registers -----------------------------
     * Warp w owns k-steps [ (w>>1)*KPW, +KPW ) of head-block M=16.  Q is
     * constant for the whole kernel, so this is loaded once and never re-read;
     * that is the entire reason phase 1 splits k rather than n. */
    const uint32_t kgrp = warp >> 1u;                   /* 0..3 */
    const uint32_t ntile = warp & 1u;                   /* which 8 rows */
    uint32_t qf[AF16_KPW][4];
    {
        const uint64_t qbase = ((uint64_t)t * n_head + hbase) * AF16_DIM;
        #pragma unroll
        for (uint32_t s = 0; s < AF16_KPW; s++) {
            const uint32_t k0 = (kgrp * AF16_KPW + s) * 16u;
            const uint32_t hA = g, hB = g + 8u;         /* the fragment's two rows */
            const uint32_t c0 = k0 + tg * 2u, c1 = k0 + tg * 2u + 8u;
            const float *qa = q + qbase + (uint64_t)hA * AF16_DIM;
            const float *qb = q + qbase + (uint64_t)hB * AF16_DIM;
            qf[s][0] = af16_pack(qa[c0], qa[c0 + 1u]);
            qf[s][1] = af16_pack(qb[c0], qb[c0 + 1u]);
            qf[s][2] = af16_pack(qa[c1], qa[c1 + 1u]);
            qf[s][3] = af16_pack(qb[c1], qb[c1 + 1u]);
        }
    }

    /* ---- running softmax state + output accumulator --------------------- */
    if (tid < AF16_HEADS) { sM[tid] = -INFINITY; sL[tid] = 0.0f; }
    float acc[AF16_DPW / 8u][4];
    #pragma unroll
    for (uint32_t n = 0; n < AF16_DPW / 8u; n++)
        acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    __syncthreads();

    for (uint32_t row0 = 0; row0 < n_score; row0 += AF16_ROWS) {
        const uint32_t nr = min(AF16_ROWS, n_score - row0);

        /* ---- stage the tile, converting to fp16 ------------------------- */
        for (uint32_t i = tid; i < AF16_ROWS * AF16_DIM / 2u; i += AF16_THREADS) {
            const uint32_t r = i / (AF16_DIM / 2u);
            const uint32_t d2 = (i % (AF16_DIM / 2u)) * 2u;
            __half2 v = make_half2(__float2half(0.f), __float2half(0.f));
            if (r < nr) {
                const uint32_t sr = row0 + r;
                if (sr < raw_count) {
                    const uint64_t b = (uint64_t)(raw_start + sr) * AF16_DIM;
                    v = make_half2(__float2half(af16_kv(raw_kv, raw_f16, b, d2)),
                                   __float2half(af16_kv(raw_kv, raw_f16, b, d2 + 1u)));
                } else {
                    const float *cr = comp_kv + (uint64_t)(sr - raw_count) * AF16_DIM;
                    v = make_half2(__float2half(cr[d2]), __float2half(cr[d2 + 1u]));
                }
            }
            *(__half2 *)&sKV[r * AF16_KVSTRIDE + d2] = v;
        }
        __syncthreads();

        /* ---- phase 1: S = Q . KV^T ------------------------------------- */
        {
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            const uint32_t rbase = ntile * 8u;
            #pragma unroll
            for (uint32_t s = 0; s < AF16_KPW; s++) {
                const uint32_t k0 = (kgrp * AF16_KPW + s) * 16u;
                /* B[k=dim][n=row] = KV[row][dim]: fixed row per lane group,
                 * two dim pairs.  This is the column walk the padding is for. */
                const __half *kr = &sKV[(rbase + g) * AF16_KVSTRIDE + k0];
                const uint32_t b0 = *(const uint32_t *)&kr[tg * 2u];
                const uint32_t b1 = *(const uint32_t *)&kr[tg * 2u + 8u];
                af16_mma(s0, s1, s2, s3, qf[s][0], qf[s][1], qf[s][2], qf[s][3], b0, b1);
            }
            sPart[kgrp][g][rbase + tg * 2u]      = s0;
            sPart[kgrp][g][rbase + tg * 2u + 1u] = s1;
            sPart[kgrp][g + 8u][rbase + tg * 2u]      = s2;
            sPart[kgrp][g + 8u][rbase + tg * 2u + 1u] = s3;
        }
        __syncthreads();

        /* ---- phase 2: sum the k-split, then online softmax --------------- */
        for (uint32_t i = tid; i < AF16_HEADS * AF16_ROWS; i += AF16_THREADS) {
            const uint32_t h = i / AF16_ROWS, r = i % AF16_ROWS;
            const float v = sPart[0][h][r] + sPart[1][h][r] + sPart[2][h][r] + sPart[3][h][r];
            sS[h][r] = (r < nr) ? v * scale : -INFINITY;
        }
        __syncthreads();

        if (tid < AF16_HEADS) {
            const uint32_t h = tid;
            float mx = sM[h];
            for (uint32_t r = 0; r < nr; r++) mx = fmaxf(mx, sS[h][r]);
            const float corr = (sM[h] == -INFINITY) ? 0.0f : __expf(sM[h] - mx);
            float l = sL[h] * corr;
            for (uint32_t r = 0; r < nr; r++) {
                const float p = __expf(sS[h][r] - mx);
                sP[h][r] = __float2half(p);
                l += p;
            }
            for (uint32_t r = nr; r < AF16_ROWS; r++) sP[h][r] = __float2half(0.f);
            sM[h] = mx; sL[h] = l; sCorr[h] = corr;
        }
        __syncthreads();

        /* ---- phase 3: O = O*corr + P . KV ------------------------------- */
        {
            /* The accumulator's two head rows for this lane are g and g+8. */
            const float ca = sCorr[g], cb = sCorr[g + 8u];
            const uint32_t pa0 = *(const uint32_t *)&sP[g][tg * 2u];
            const uint32_t pa1 = *(const uint32_t *)&sP[g + 8u][tg * 2u];
            const uint32_t pa2 = *(const uint32_t *)&sP[g][tg * 2u + 8u];
            const uint32_t pa3 = *(const uint32_t *)&sP[g + 8u][tg * 2u + 8u];
            #pragma unroll
            for (uint32_t n = 0; n < AF16_DPW / 8u; n++) {
                acc[n][0] *= ca; acc[n][1] *= ca;
                acc[n][2] *= cb; acc[n][3] *= cb;
                /* B[k=row][n=dim]: this lane's column is n_base+g, and its four
                 * k values are rows 2t, 2t+1, 2t+8, 2t+9.  Already fp16 in
                 * smem, so repack the bits rather than round-tripping f32. */
                const __half *kc = &sKV[warp * AF16_DPW + n * 8u + g];
                const uint32_t b0 =
                    (uint32_t)__half_as_ushort(kc[(tg * 2u)      * AF16_KVSTRIDE]) |
                    ((uint32_t)__half_as_ushort(kc[(tg * 2u + 1u) * AF16_KVSTRIDE]) << 16);
                const uint32_t b1 =
                    (uint32_t)__half_as_ushort(kc[(tg * 2u + 8u) * AF16_KVSTRIDE]) |
                    ((uint32_t)__half_as_ushort(kc[(tg * 2u + 9u) * AF16_KVSTRIDE]) << 16);
                af16_mma(acc[n][0], acc[n][1], acc[n][2], acc[n][3],
                         pa0, pa1, pa2, pa3, b0, b1);
            }
        }
        __syncthreads();
    }

    /* ---- epilogue: fold the sink, normalise, store ---------------------- */
    if (tid < AF16_HEADS) {
        const uint32_t h = hbase + tid;
        const float sink = (h < n_head) ? sinks[h] : -INFINITY;
        const float nm = fmaxf(sM[tid], sink);
        const float os = (sM[tid] == -INFINITY) ? 0.0f : __expf(sM[tid] - nm);
        sCorr[tid] = os;
        sL[tid] = sL[tid] * os + __expf(sink - nm);
    }
    __syncthreads();

    {
        /* D layout is d0=D[g][2t], d1=D[g][2t+1], d2=D[g+8][2t], d3=D[g+8][2t+1]
         * with M = head and N = dim.  So this lane holds heads g and g+8 (which
         * is why the corrections above are sCorr[g]/sCorr[g+8]) at dims
         * n_base+2t and n_base+2t+1.  Note N is the DIM axis here and the ROW
         * axis in phase 1 -- the two phases transpose, and mixing them up is
         * the one bug this kernel cannot detect at runtime. */
        const float ia = sCorr[g]      / (sL[g]      == 0.0f ? 1.0f : sL[g]);
        const float ib = sCorr[g + 8u] / (sL[g + 8u] == 0.0f ? 1.0f : sL[g + 8u]);
        const uint32_t ha = hbase + g, hb = hbase + g + 8u;
        float *oa = heads + ((uint64_t)t * n_head + ha) * AF16_DIM;
        float *ob = heads + ((uint64_t)t * n_head + hb) * AF16_DIM;
        #pragma unroll
        for (uint32_t n = 0; n < AF16_DPW / 8u; n++) {
            const uint32_t nb = warp * AF16_DPW + n * 8u;
            oa[nb + tg * 2u]      = acc[n][0] * ia;
            oa[nb + tg * 2u + 1u] = acc[n][1] * ia;
            ob[nb + tg * 2u]      = acc[n][2] * ib;
            ob[nb + tg * 2u + 1u] = acc[n][3] * ib;
        }
    }
#endif
}

/* ---- launcher ----------------------------------------------------------- */

/* PULSAR_ATTN_F16_MMA is a DEVICE-side guard -- __CUDA_ARCH__ is undefined in
 * the host pass, so testing it here would compile the launcher down to
 * "return 0" and the kernel would look like a permanent shape-gate refusal.
 * The host question is a runtime one: does this device actually have the
 * m16n8k16 tensor cores?  Ask the driver, once. */
static int af16_device_supported(void) {
    static int ok = -1;
    if (ok < 0) {
        int major = 0, minor = 0, dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) { ok = 0; return ok; }
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            ok = 0; return ok;
        }
        ok = (major >= 8) ? 1 : 0;      /* mma.m16n8k16.f16 is sm_80+ */
    }
    return ok;
}

int pulsar_gpu_attention_f16_prefill(
        float *heads, const float *sinks, const float *q,
        const float *raw_kv, const float *comp_kv,
        uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim, int raw_f16) {
    if (!heads || !sinks || !q || !raw_kv) return 0;
    if (head_dim != AF16_DIM) return 0;
    if (n_head == 0u || (n_head % AF16_HEADS) != 0u) return 0;
    if (n_tokens == 0u) return 0;
    if (n_comp != 0u && !comp_kv) return 0;
    if (!af16_device_supported()) return 0;
    {
    dim3 grid(n_tokens, n_head / AF16_HEADS, 1);
    attn_f16_kernel<<<grid, AF16_THREADS>>>(heads, sinks, q, raw_kv,
                                            comp_kv ? comp_kv : raw_kv,
                                            n_tokens, n_comp, window, ratio,
                                            n_head, raw_f16);
    return cuda_ok(cudaGetLastError(), "attention f16 mma launch");
    }
}

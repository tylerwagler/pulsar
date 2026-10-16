/* The MoE router, one parameterized implementation for every family (L295 F2): contracts in pulsar_cuda_router.h.
 *
 * It replaces the two per-family routers: Qwen's softmax + shared-gate kernels (pulsar_cuda_qwen_moe.cu) and
 * DeepSeek's sqrt-softplus top-k / hash kernels (pulsar_cuda_hc_router.cu).  Each family's arithmetic is kept
 * exactly -- the same expressions in the same order -- so moving a family onto this file moves no byte; the
 * families now differ only in the cfg they declare. */
#include "pulsar_cuda_router.h"

#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>

namespace {

constexpr int kLogitsTB = 8;     ///< tokens per logits CTA row
constexpr int kSelectWarps = 4;  ///< rows (warps) per select CTA

__device__ __forceinline__ float bf16_round(float v) { return __bfloat162float(__float2bfloat16(v)); }

__device__ __forceinline__ void bf16x8(const uint4 &u, float f[8]) {
    const __nv_bfloat162 *p = reinterpret_cast<const __nv_bfloat162 *>(&u);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 v = __bfloat1622float2(p[j]);
        f[2 * j] = v.x;
        f[2 * j + 1] = v.y;
    }
}

/* One warp per logit row (rows n_w.. come from w_tail), kLogitsTB tokens per CTA row.  Lane l owns the uint4
 * chunks l, l + 32, ... of the row and accumulates each token's partial in chunk order; the xor tree then sums the
 * lanes.  A token's logit is the same arithmetic at any T. */
__global__ void __launch_bounds__(256)
router_logits_kernel(const __nv_bfloat16 *__restrict__ x, int H, const __nv_bfloat16 *__restrict__ w, int n_w,
                     const __nv_bfloat16 *__restrict__ w_tail, int n_rows, int T, float *__restrict__ logits) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * 8 + warp;
    if (e >= n_rows) return;                      /* whole warps */
    const int t0 = blockIdx.y * kLogitsTB;
    const int nt = min(kLogitsTB, T - t0);
    const uint4 *w4 = reinterpret_cast<const uint4 *>(e < n_w ? w + (size_t)e * H : w_tail + (size_t)(e - n_w) * H);
    float acc[kLogitsTB];
#pragma unroll
    for (int tt = 0; tt < kLogitsTB; ++tt) acc[tt] = 0.0f;
    for (int i = lane; i < H / 8; i += 32) {
        float wf[8];
        bf16x8(w4[i], wf);
#pragma unroll
        for (int tt = 0; tt < kLogitsTB; ++tt) {
            if (tt < nt) {
                float xf[8];
                bf16x8(reinterpret_cast<const uint4 *>(x + (size_t)(t0 + tt) * H)[i], xf);
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[tt] = fmaf(wf[j], xf[j], acc[tt]);
            }
        }
    }
#pragma unroll
    for (int tt = 0; tt < kLogitsTB; ++tt) {
        float v = acc[tt];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0 && tt < nt) logits[(size_t)(t0 + tt) * n_rows + e] = v;
    }
}

__device__ __forceinline__ float softplus_dev(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

/* (a, ai) ranks before (b, bi): the larger value, ties to the lower expert id. */
__device__ __forceinline__ bool ranks_before(float av, int ai, float bv, int bi) {
    return av > bv || (av == bv && ai < bi);
}

/* The device copy of the cfg and the layer, by value. */
struct router_args {
    pulsar_router_cfg c;
    pulsar_router_layer l;
    int n_rows;               ///< logit row width
};

/* One warp per row: each lane holds the NE / 32 experts lane, lane + 32, ...; the TOPK picks are a TOPK-round warp
 * argmax over the selection score carrying the weight's source value (the unbiased score, or for the sink
 * renormalisation the raw logit); lane 0 renormalises and writes. */
template <int NE, int TOPK>
__global__ void __launch_bounds__(kSelectWarps * 32)
router_select_kernel(router_args a, const float *__restrict__ logits, const int32_t *__restrict__ tokens, int T,
                     int32_t *__restrict__ selected, float *__restrict__ weights, float *__restrict__ gammas,
                     float *__restrict__ probs) {
    static_assert(NE % 32 == 0, "experts per lane must be whole");
    constexpr int PER = NE / 32;
    const pulsar_router_cfg &c = a.c;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int t = blockIdx.x * kSelectWarps + warp;
    if (t >= T) return;
    const float *lg = logits + (size_t)t * a.n_rows;
    int32_t *sel = selected + (size_t)t * TOPK;
    float *w = weights + (size_t)t * TOPK;
    const int S = c.shared_gate == PULSAR_ROUTER_SHARED_NONE ? 0 : c.n_shared;

    float raw[PER], p[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        raw[i] = lg[lane + 32 * i];
        if (c.logits_bf16) raw[i] = bf16_round(raw[i]);
    }
    if (c.score == PULSAR_ROUTER_SCORE_SOFTMAX) {
        float m = -INFINITY;
#pragma unroll
        for (int i = 0; i < PER; ++i) { p[i] = raw[i]; m = fmaxf(m, p[i]); }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float z = 0.0f;
#pragma unroll
        for (int i = 0; i < PER; ++i) { p[i] = expf(p[i] - m); z += p[i]; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
#pragma unroll
        for (int i = 0; i < PER; ++i) p[i] = p[i] / z;
    } else if (c.score == PULSAR_ROUTER_SCORE_SQRT_SOFTPLUS) {
#pragma unroll
        for (int i = 0; i < PER; ++i) p[i] = sqrtf(softplus_dev(raw[i]));
    } else {
#pragma unroll
        for (int i = 0; i < PER; ++i) p[i] = 1.0f / (1.0f + expf(-raw[i]));
    }
    if (probs) {
#pragma unroll
        for (int i = 0; i < PER; ++i) probs[(size_t)t * NE + lane + 32 * i] = p[i];
    }
    /* A non-finite score row has no routing (rule 9): experts 0..k-1 with NaN weights, so the NaN reaches the block
     * output and the sum's non-finite flag names the layer -- never a plausible routing. */
    bool bad = false;
#pragma unroll
    for (int i = 0; i < PER; ++i) bad |= !isfinite(p[i]);
    if (__any_sync(0xffffffffu, bad)) {
        const float qnan = __int_as_float(0x7fc00000);
        if (lane < TOPK) {
            sel[lane] = lane;
            w[lane] = qnan;
        }
        if (lane < S) gammas[(size_t)lane * T + t] = qnan;
        return;
    }

    const int32_t tok = tokens ? tokens[t] : 0;
    const bool alt = a.l.alt_bias && tokens && tok >= a.l.alt_token_min;
    const bool sinks = c.renorm == PULSAR_ROUTER_RENORM_LOGSIGMOID_SINKS;
    float vals[TOPK + PULSAR_ROUTER_SHARED_MAX];   /* the weights' source values, picks first then the sinks */
    int ids[TOPK];
    if (a.l.table && !alt) {
        /* the token's table row names the experts; each weight's source is that expert's unbiased score, read
         * from the lane that holds it */
        const int32_t row_id = tok < 0 || tok >= a.l.table_rows ? 0 : tok;
        const int32_t *row = a.l.table + (size_t)row_id * TOPK;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            const int32_t e = row[k];
            float v = 0.0f;
            if (e >= 0 && e < NE) {
                float mine = 0.0f;
#pragma unroll
                for (int i = 0; i < PER; ++i)
                    if (i == (e >> 5)) mine = p[i];
                v = __shfl_sync(0xffffffffu, mine, e & 31);
            }
            ids[k] = e;
            vals[k] = v;
        }
    } else {
        const float *bias = alt ? a.l.alt_bias : a.l.bias;
        float score[PER], carry[PER];
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            score[i] = bias ? p[i] + bias[lane + 32 * i] : p[i];
            carry[i] = sinks ? raw[i] : p[i];
        }
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            float bs = -INFINITY, bv = 0.0f;
            int be = 0x7fffffff;
#pragma unroll
            for (int i = 0; i < PER; ++i) {
                const int e = lane + 32 * i;
                if (ranks_before(score[i], e, bs, be)) { bs = score[i]; bv = carry[i]; be = e; }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float os = __shfl_xor_sync(0xffffffffu, bs, o);
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oe = __shfl_xor_sync(0xffffffffu, be, o);
                if (ranks_before(os, oe, bs, be)) { bs = os; bv = ov; be = oe; }
            }
#pragma unroll
            for (int i = 0; i < PER; ++i)
                if (lane + 32 * i == be) score[i] = -INFINITY;
            ids[k] = be;
            vals[k] = bv;
        }
    }
    if (lane != 0) return;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) sel[k] = ids[k];
    if (sinks) {
        /* log-sigmoid over the k picks' raw logits and the S sink logits, then a softmax over those, scaled */
        const int A = TOPK + S;
        constexpr int AMAX = TOPK + PULSAR_ROUTER_SHARED_MAX;
#pragma unroll
        for (int j = 0; j < PULSAR_ROUTER_SHARED_MAX; ++j) {
            if (j < S) {
                float g = lg[c.n_expert + j];
                if (c.logits_bf16) g = bf16_round(g);
                vals[TOPK + j] = g;
            }
        }
        float m = -INFINITY;
#pragma unroll
        for (int i = 0; i < AMAX; ++i) {
            if (i < A) {
                const float x = vals[i];
                vals[i] = fminf(x, 0.0f) - log1pf(expf(-fabsf(x)));
                m = fmaxf(m, vals[i]);
            }
        }
        float z = 0.0f;
#pragma unroll
        for (int i = 0; i < AMAX; ++i) {
            if (i < A) {
                vals[i] = expf(vals[i] - m);
                z += vals[i];
            }
        }
        const float scale = a.l.global_scale ? c.route_scale * a.l.global_scale[0] : c.route_scale;
#pragma unroll
        for (int i = 0; i < AMAX; ++i) {
            if (i < A) {
                float v = vals[i] / z * scale;
                if (c.weights_bf16) v = bf16_round(v);
                if (i < TOPK) w[i] = v;
                else gammas[(size_t)(i - TOPK) * T + t] = v;
            }
        }
        return;
    }
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) sum += vals[k];
    sum = fmaxf(sum, c.sum_floor);
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        float v = vals[k] / sum * c.route_scale;
        if (c.weights_bf16) v = bf16_round(v);
        w[k] = v;
    }
    for (int j = 0; j < S; ++j) {   /* PULSAR_ROUTER_SHARED_SIGMOID: each gate row's own sigmoid */
        float g = lg[c.n_expert + j];
        if (c.logits_bf16) g = bf16_round(g);
        float v = 1.0f / (1.0f + expf(-g));
        if (c.weights_bf16) v = bf16_round(v);
        gammas[(size_t)j * T + t] = v;
    }
}

bool launch_ok(const char *what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: router %s: launch failed: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

} // namespace

extern "C" int pulsar_router_logit_rows(const pulsar_router_cfg *c) {
    return c ? c->n_expert + (c->shared_gate == PULSAR_ROUTER_SHARED_NONE ? 0 : c->n_shared) : 0;
}

extern "C" int pulsar_router_logits_launch(const uint16_t *x_bf16, int T, int hidden, const uint16_t *w, int n_w,
                                           const uint16_t *w_tail, int n_tail, float *logits, cudaStream_t stream) {
    if (!x_bf16 || !w || !logits || T <= 0 || hidden <= 0 || hidden % 8 || n_w <= 0 || n_tail < 0 ||
        (n_tail > 0) != (w_tail != nullptr)) {
        fprintf(stderr, "pulsar: router logits: %d rows of %d (a multiple of 8), %d + %d logit rows, or a null pointer "
                        "-- refusing\n", T, hidden, n_w, n_tail);
        return -1;
    }
    if (((uintptr_t)x_bf16 | (uintptr_t)w | (uintptr_t)w_tail) & 15u) {
        fprintf(stderr, "pulsar: router logits: bf16 rows must be 16-byte aligned -- refusing\n");
        return -1;
    }
    const int n_rows = n_w + n_tail;
    const dim3 grid((unsigned)((n_rows + 7) / 8), (unsigned)((T + kLogitsTB - 1) / kLogitsTB));
    router_logits_kernel<<<grid, 256, 0, stream>>>((const __nv_bfloat16 *)x_bf16, hidden, (const __nv_bfloat16 *)w, n_w,
                                                   (const __nv_bfloat16 *)w_tail, n_rows, T, logits);
    return launch_ok("logits") ? 0 : -3;
}

extern "C" int pulsar_router_select_launch(const pulsar_router_cfg *c, const pulsar_router_layer *l,
                                           const float *logits, const int32_t *tokens, int T, int32_t *selected,
                                           float *weights, float *gammas, float *probs, cudaStream_t stream) {
    if (!c || !l || !logits || !selected || !weights || T <= 0) {
        fprintf(stderr, "pulsar: router select: a null input or no rows -- refusing\n");
        return -1;
    }
    const int S = c->shared_gate == PULSAR_ROUTER_SHARED_NONE ? 0 : c->n_shared;
    const bool sinks = c->renorm == PULSAR_ROUTER_RENORM_LOGSIGMOID_SINKS;
    const bool score_ok = c->score == PULSAR_ROUTER_SCORE_SOFTMAX || c->score == PULSAR_ROUTER_SCORE_SQRT_SOFTPLUS ||
                          c->score == PULSAR_ROUTER_SCORE_SIGMOID;
    const bool gate_ok = c->shared_gate == PULSAR_ROUTER_SHARED_NONE || c->shared_gate == PULSAR_ROUTER_SHARED_SIGMOID ||
                         c->shared_gate == PULSAR_ROUTER_SHARED_SINK;
    if (!score_ok || !gate_ok || (!sinks && c->renorm != PULSAR_ROUTER_RENORM_SUM) ||
        sinks != (c->shared_gate == PULSAR_ROUTER_SHARED_SINK) || c->n_shared < 0 ||
        c->n_shared > PULSAR_ROUTER_SHARED_MAX || (S > 0 && !gammas) || (sinks && S == 0) ||
        (l->table && (sinks || !tokens || l->table_rows <= 0))) {
        fprintf(stderr, "pulsar: router select: score %d, renorm %d, %d shared experts gated %d, a %s table -- not a "
                        "router this file has (a sink gate needs the sink renormalisation and gammas, a table needs "
                        "token ids and the sum renormalisation) -- refusing\n",
                (int)c->score, (int)c->renorm, c->n_shared, (int)c->shared_gate, l->table ? "token-id" : "no");
        return -1;
    }
    const router_args a{*c, *l, pulsar_router_logit_rows(c)};
    const unsigned grid = (unsigned)((T + kSelectWarps - 1) / kSelectWarps);
    const int E = c->n_expert, K = c->top_k;
#define PULSAR_ROUTER_ARM(NE, TOPK)                                                                                   \
    if (E == NE && K == TOPK) {                                                                                       \
        router_select_kernel<NE, TOPK><<<grid, kSelectWarps * 32, 0, stream>>>(a, logits, tokens, T, selected,        \
                                                                              weights, gammas, probs);               \
        return launch_ok("select") ? 0 : -3;                                                                          \
    }
    /* the routers the engine serves: Qwen3.8-Flash-Next 512 / top-10; Nemotron 3.5 Super's 512 / top-22 (L296 N4);
     * DeepSeek V4.1's 384 / top-6 target, 0731's 256 / top-6 target (and Inkling-Small's 256 / top-6) and V4.1's
     * 128 / top-3 DSpark drafter */
    PULSAR_ROUTER_ARM(512, 10)
    PULSAR_ROUTER_ARM(512, 22)
    PULSAR_ROUTER_ARM(384, 6)
    PULSAR_ROUTER_ARM(256, 6)
    PULSAR_ROUTER_ARM(128, 3)
#undef PULSAR_ROUTER_ARM
    fprintf(stderr, "pulsar: router select: no instance for %d experts / top-%d -- refusing\n", E, K);
    return -1;
}

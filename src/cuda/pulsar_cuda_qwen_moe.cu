/* Qwen3.8-Flash-Next MoE block (L251 S4): the router, the EXL3 routed
 * experts, the sigmoid-gated EXL3 shared expert, and the fixed-order sum.
 * Contracts in pulsar_cuda_qwen.h; the source is transformers'
 * Qwen4ExpTextSparseMoeBlock / Qwen4ExpTextTopKRouter.  Since L272 P4c the
 * block is three launches the engine composes (family_qwen_s4.cpp
 * pulsar_qwen_s4_moe) over one carved workspace (pulsar_qwen_moe_carve):
 *
 *   router     pulsar_qwen_router_launch: logits = W_r x in f32 (bf16 x bf16,
 *              fixed order), rounded to bf16; softmax; top-10 by probability;
 *              renormalised; bf16; and the shared expert's sigmoid gate
 *   routed     pulsar_rows_moe_routed_launch -- the core front door's arm over
 *              bf16 rows (moe.cpp), family-neutral, shapes from the model:
 *              Tessera, or the EXL3 routed front door (mmq/exl3_moe.cuh, the
 *              one every family's EXL3 experts take) with a FUSED gate_up or a
 *              gate + up PAIR, no clamp, the fold's mid emitted bf16; expert
 *              parallelism localises the picks first
 *   shared     pulsar_qwen_moe_shared_launch: the dense arm (gate, up) on the
 *              same bf16 rows, SwiGLU emitted bf16 by its producer, the down
 *   out        routed + sigmoid(w_sg . x) * shared -- in that order
 *
 * L251 / ac69748f: there is no E4M3 activation slot in this family. */
#include "pulsar_cuda_qwen.h"
#include "pulsar_cuda_mx.cuh"
#include "mmq/ds4_exl3_dense.cuh"
#include "mmq/exl3_moe.cuh"
#ifdef PULSAR_HAVE_MMQ
#include "mmq/pulsar_tessera.h"
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <stdlib.h>

namespace {

constexpr int kH = PULSAR_QWEN_HIDDEN;
constexpr int kE = PULSAR_QWEN_N_EXPERT;
constexpr int kTopK = PULSAR_QWEN_TOPK;
constexpr int kRouterTB = 8;                      ///< tokens per router CTA row
static_assert(kH % 256 == 0, "the router's uint4 walk covers the row in whole warps");
static_assert(kE % 32 == 0, "the top-k warp holds E / 32 logits per lane");

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

/* One warp per expert row (row kE = the shared-expert gate), kRouterTB tokens
 * per CTA row.  Lane l owns the uint4 chunks l, l + 32, ... of the row and
 * accumulates each token's partial in chunk order; the xor tree then sums the
 * lanes.  A token's logit is the same arithmetic at any T. */
__global__ void __launch_bounds__(256)
qwen_router_logits_kernel(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ wr,
                          const __nv_bfloat16 *__restrict__ wsg, int T, float *__restrict__ logits) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * 8 + warp;
    if (e > kE) return;                           /* whole warps */
    const int t0 = blockIdx.y * kRouterTB;
    const int nt = min(kRouterTB, T - t0);
    const uint4 *w4 = reinterpret_cast<const uint4 *>(e < kE ? wr + (size_t)e * kH : wsg);
    float acc[kRouterTB];
#pragma unroll
    for (int tt = 0; tt < kRouterTB; ++tt) acc[tt] = 0.0f;
    for (int i = lane; i < kH / 8; i += 32) {
        float wf[8];
        bf16x8(w4[i], wf);
#pragma unroll
        for (int tt = 0; tt < kRouterTB; ++tt) {
            if (tt < nt) {
                float xf[8];
                bf16x8(reinterpret_cast<const uint4 *>(x + (size_t)(t0 + tt) * kH)[i], xf);
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[tt] = fmaf(wf[j], xf[j], acc[tt]);
            }
        }
    }
#pragma unroll
    for (int tt = 0; tt < kRouterTB; ++tt) {
        float v = acc[tt];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0 && tt < nt) logits[(size_t)(t0 + tt) * (kE + 1) + e] = v;
    }
}

/* One warp per token: bf16 logits, softmax (f32), top-k by probability with
 * ties to the lower expert id, renormalise, bf16 weights; the shared gate. */
__global__ void __launch_bounds__(128)
qwen_router_topk_kernel(const float *__restrict__ logits, int T, int32_t *__restrict__ sel,
                        float *__restrict__ wts, float *__restrict__ sgate) {
    constexpr int NV = kE / 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int t = blockIdx.x * (blockDim.x >> 5) + warp;
    if (t >= T) return;
    const float *lg = logits + (size_t)t * (kE + 1);
    float p[NV];
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < NV; ++i) { p[i] = bf16_round(lg[lane + 32 * i]); m = fmaxf(m, p[i]); }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float z = 0.0f;
#pragma unroll
    for (int i = 0; i < NV; ++i) { p[i] = expf(p[i] - m); z += p[i]; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
#pragma unroll
    for (int i = 0; i < NV; ++i) p[i] = p[i] / z;
    /* A non-finite logit row has no routing.  Rule 9: do not pick a plausible
     * one -- route to experts 0..k-1 with NaN weights, so the NaN reaches the
     * block output and the sum's non-finite flag names the layer. */
    bool bad = false;
#pragma unroll
    for (int i = 0; i < NV; ++i) bad |= !isfinite(p[i]);
    if (__any_sync(0xffffffffu, bad)) {
        if (lane < kTopK) {
            sel[(size_t)t * kTopK + lane] = lane;
            wts[(size_t)t * kTopK + lane] = __int_as_float(0x7fc00000);
        }
        if (lane == 0) sgate[t] = __int_as_float(0x7fc00000);
        return;
    }

    uint32_t taken = 0;                          /* bit i: this lane's p[i] is selected */
    float top_p[kTopK];
    int top_e[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        float bp = -1.0f;
        int be = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < NV; ++i) {
            const int e = lane + 32 * i;
            if (!(taken >> i & 1u) && (p[i] > bp || (p[i] == bp && e < be))) { bp = p[i]; be = e; }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float op = __shfl_xor_sync(0xffffffffu, bp, o);
            const int oe = __shfl_xor_sync(0xffffffffu, be, o);
            if (op > bp || (op == bp && oe < be)) { bp = op; be = oe; }
        }
        top_p[k] = bp;
        top_e[k] = be;
        if ((be & 31) == lane) taken |= 1u << (be >> 5);
    }
    if (lane == 0) {
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < kTopK; ++k) s += top_p[k];
#pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            sel[(size_t)t * kTopK + k] = top_e[k];
            wts[(size_t)t * kTopK + k] = bf16_round(top_p[k] / s);
        }
        const float g = bf16_round(lg[kE]);
        sgate[t] = bf16_round(1.0f / (1.0f + expf(-g)));
    }
}

/* The shared expert's SwiGLU, emitted in the format its consumer reads (rule 3).  L251 / ac69748f:
 * the shared down projection reads a bf16 row, so the producer emits bf16 -- and bf16 carries its own
 * exponent, so there is no per-32 block and no scale slab any more. */
__global__ void __launch_bounds__(256)
qwen_swiglu_emit_kernel(const float *__restrict__ g, const float *__restrict__ u, int D,
                        __nv_bfloat16 *__restrict__ xb) {
    const int t = blockIdx.x;
    for (int c0 = 0; c0 < D; c0 += 256) {
        const int c = c0 + (int)threadIdx.x;
        if (c < D) {
            xb[(size_t)t * D + c] = __float2bfloat16(pulsar_swiglu_elem(
                g[(size_t)t * D + c], u[(size_t)t * D + c], 1.0f, 0.0f));
        }
    }
}

/* out = routed + sgate * shared, in that order. */
/* L266 EP: a pick of expert e becomes this rank's local index e - base, or -1 when another rank owns it
 * (the routed arms drop -1, mm_ids_helper by range and the fold / sum explicitly). */
__global__ void qwen_ep_localize_kernel(int32_t *__restrict__ sel, int n, int base, int n_local) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int e = sel[i] - base;
    sel[i] = e >= 0 && e < n_local ? e : -1;
}

__global__ void qwen_shared_add_kernel(float *__restrict__ out, const float *__restrict__ ys,
                                       const float *__restrict__ sgate, int T) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * kH) return;
    const float s = sgate[i / kH] * ys[i];
    out[i] = out[i] + s;
}

/* A bump allocator over the caller's workspace (256-byte aligned slices). */
struct ws_bump {
    uint8_t *base;
    size_t cap, used;
    bool failed;
    void *take(size_t bytes) {
        const size_t off = (used + 255) & ~(size_t)255;
        if (failed || off + bytes > cap) { failed = true; return nullptr; }
        used = off + bytes;
        return base ? base + off : nullptr;
    }
};

/* L272 P4c: the routed part's workspace (pulsar_rows_moe_routed_launch): the gate / up outputs, the fold's bf16
 * rows, the down outputs -- a function of the rows and the routing shape alone (the model's, not a family's). */
struct routed_ws {
    float *gu_z, *down_z;
    uint8_t *mid_x;                 /* the routed fold's bf16 rows (ac69748f) */
};
static size_t routed_ws_layout(int T, int top_k, int mid, int out_dim, void *base, size_t cap, routed_ws *o) {
    ws_bump b{(uint8_t *)base, base ? cap : (size_t)-1, 0, false};
    const size_t pairs = (size_t)T * (size_t)top_k;
    routed_ws m{};
    m.gu_z   = (float *)b.take(pairs * 2 * (size_t)mid * 4);
    m.mid_x  = (uint8_t *)b.take(pairs * (size_t)mid * 2);   /* bf16: no per-32 block, no scale slab */
    m.down_z = (float *)b.take(pairs * (size_t)out_dim * 4);
    if (o) *o = m;
    return b.failed ? 0 : b.used;
}

/* L255: the Tessera routed launch's bf16 [T][H] output -- the weighted, fixed-order sum over the top-k, exactly
 * Tessera's token_sum -- is the block's routed term in f32, exactly; a non-finite value records nf_code (first
 * writer wins), as the EXL3 sum does. */
__global__ static void qwen_tessera_routed_kernel(float *__restrict__ out, const uint16_t *__restrict__ routed, size_t n,
                                                  uint32_t *__restrict__ nf_flag, uint32_t nf_code) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __uint_as_float((uint32_t)routed[i] << 16);
    out[i] = v;
    if (!isfinite(v)) atomicCAS(nf_flag, 0u, nf_code);
}

extern "C" const uint16_t *pulsar_rows_moe_routed_mid(void *ws, size_t ws_bytes, int T, int top_k, int mid, int out_dim) {
    routed_ws m;
    return ws && routed_ws_layout(T, top_k, mid, out_dim, ws, ws_bytes, &m) != 0 ? (const uint16_t *)m.mid_x : nullptr;
}

/* The one layout of the Qwen MoE block's workspace (router, routed region, shared expert): sizing and carving
 * are the same walk. */
static size_t moe_ws_layout(int T, void *base, size_t cap, pulsar_qwen_moe_parts *o) {
    ws_bump b{(uint8_t *)base, base ? cap : (size_t)-1, 0, false};
    const size_t pairs = (size_t)T * kTopK;
    const int smid = PULSAR_QWEN_SHARED_MID;
    pulsar_qwen_moe_parts m{};
    m.logits = (float *)b.take((size_t)T * (kE + 1) * 4);
    m.sel    = (int32_t *)b.take(pairs * 4);
    m.wts    = (float *)b.take(pairs * 4);
    m.sgate  = (float *)b.take((size_t)T * 4);
    m.routed_bytes = routed_ws_layout(T, kTopK, PULSAR_QWEN_EXPERT_MID, kH, nullptr, 0, nullptr);
    m.routed = b.take(m.routed_bytes);
    m.yg     = (float *)b.take((size_t)T * smid * 4);
    m.yu     = (float *)b.take((size_t)T * smid * 4);
    m.h_x    = (uint16_t *)b.take((size_t)T * smid * 2);   /* bf16: no per-32 block, no scale slab */
    m.ys     = (float *)b.take((size_t)T * kH * 4);
    /* the dense arm's workspace is a function of (rows, in, out) alone: the riding shared expert's (gate and up in
     * one launch), which covers the separate launches' */
    const size_t lb = exl3_moe_shared_ws_bytes(T, kH, smid, kH);
    m.lin = b.take(lb);
    m.lin_bytes = lb;
    if (o) *o = m;
    return b.failed ? 0 : b.used;
}

static bool launch_ok(const char *what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: qwen %s: launch failed: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

} // namespace

extern "C" size_t pulsar_rows_linear_workspace_bytes(const pulsar_rows_linear *l, int rows) {
    if (!l || rows <= 0) return 0;
    return ds4_exl3_dense_workspace_bytes(rows, l->in, l->out);
}

extern "C" int pulsar_rows_linear_launch(const pulsar_rows_linear *l, const uint16_t *x_bf16, int rows, float *y,
                                         void *ws, size_t ws_bytes, cudaStream_t stream) {
    /* L251 / ac69748f: the reader takes the block input's bf16 row.  There is no E4M3 activation slot in
     * this family, so a missing activation is an error -- never a reason to reach for another format. */
    if (!l || !l->w || !x_bf16 || rows <= 0) {
        fprintf(stderr, "pulsar: qwen linear: no weight or no bf16 activation of width %d -- refusing\n", l ? l->in : -1);
        return -1;
    }
    if (l->k2 == 0) {                       /* the recipe's mxfp8_lt dense tier */
        if (!l->sf) {
            fprintf(stderr, "pulsar: qwen linear: a %d->%d tensor is neither EXL3 nor mxfp8_lt -- refusing\n",
                    l->in, l->out);
            return -1;
        }
        const pulsar_qwen_lowrank lr{l->w, l->sf, l->out, l->in, l->prompt};
        /* L288 C1: decode rows past the decode arm's width (a wide verify, its head) in blocks of that width -- each
         * row the bytes a one-row decode gives at any width (the EXL3 tier blocks inside its own launch) */
        const int blk = l->prompt ? rows : PULSAR_QWEN_DECODE_ROWS_MAX;
        for (int r0 = 0; r0 < rows; r0 += blk) {
            const int m = rows - r0 < blk ? rows - r0 : blk;
            const int rc = pulsar_qwen_mxfp8_linear_launch(&lr, x_bf16 + (size_t)r0 * l->in, m, y + (size_t)r0 * l->out,
                                                           ws, ws_bytes, stream);
            if (rc) return rc;
        }
        return 0;
    }
    return ds4_exl3_dense_launch(l->w, l->k2, x_bf16, y, rows, l->in, l->out, ws, ws_bytes, stream, l->prompt);
}

extern "C" size_t pulsar_rows_linear_multi_workspace_bytes(const pulsar_rows_linear *const *ls, int n, int rows) {
    if (!ls || n < 1 || n > DS4_EXL3_DENSE_MAX_PROJ || rows <= 0) return 0;
    ds4_exl3_dense_proj p[DS4_EXL3_DENSE_MAX_PROJ] = {};
    int ne = 0;
    for (int j = 0; j < n; ++j) {
        if (!ls[j] || ls[j]->in != ls[0]->in) return 0;
        if (ls[j]->out % 128 == 0) p[ne++].N = ls[j]->out;   /* an EXL3 shape (mxfp8_lt-only widths need none) */
    }
    return ne ? ds4_exl3_dense_multi_workspace_bytes(p, ne, rows, ls[0]->in) : 0;
}

extern "C" int pulsar_rows_linear_multi_launch(const pulsar_rows_linear *const *ls, int n, const uint16_t *x_bf16,
                                               int rows, float *const *y, void *ws, size_t ws_bytes, cudaStream_t stream) {
    if (!ls || n < 1 || n > DS4_EXL3_DENSE_MAX_PROJ || !y || !x_bf16 || rows <= 0) {
        fprintf(stderr, "pulsar: qwen linears: %d Linears (1..%d), no input or no rows -- refusing\n", n,
                DS4_EXL3_DENSE_MAX_PROJ);
        return -1;
    }
    ds4_exl3_dense_proj p[DS4_EXL3_DENSE_MAX_PROJ];
    int ne = 0;
    for (int j = 0; j < n; ++j) {
        const pulsar_rows_linear *l = ls[j];
        if (!l || !l->w || !y[j] || l->in != ls[0]->in || l->prompt != ls[0]->prompt) {
            fprintf(stderr, "pulsar: qwen linears: Linear %d has no weight / output, or not the first one's input width "
                            "and row kind -- refusing\n", j);
            return -1;
        }
        if (l->k2) p[ne++] = ds4_exl3_dense_proj{l->w, l->k2, l->out, y[j]};
    }
    if (ne && ds4_exl3_dense_multi_launch(p, ne, x_bf16, rows, ls[0]->in, ws, ws_bytes, stream, ls[0]->prompt) != 0)
        return -1;
    for (int j = 0; j < n; ++j)   /* the mxfp8_lt tier, one by one (its own arm) */
        if (!ls[j]->k2 && pulsar_rows_linear_launch(ls[j], x_bf16, rows, y[j], ws, ws_bytes, stream) != 0) return -1;
    return 0;
}

/* The MX slab geometry, for the engine TUs that cannot include pulsar_cuda_mx.cuh
 * (g++ does not know __host__/__device__).  One authority: these delegate. */
int pulsar_gpu_mx_kbp(int in_dim) { return pulsar_mx_kbp(in_dim); }
size_t pulsar_gpu_mx_sf_slab_bytes(int rows, int kbp) { return pulsar_mx_sf_slab_bytes(rows, kbp); }

extern "C" int pulsar_qwen_router_launch(const uint16_t *x_bf16, const uint16_t *router_w, const uint16_t *shared_gate_w,
                                         int T, int hidden, int n_expert, int top_k,
                                         float *logits, int32_t *selected, float *weights, float *sgate,
                                         cudaStream_t stream) {
    if (hidden != kH || n_expert != kE || top_k != kTopK || T <= 0 || !x_bf16 || !router_w || !shared_gate_w ||
        !logits || !selected || !weights || !sgate) {
        fprintf(stderr, "pulsar: qwen router: shape %d/%d/top-%d (built for %d/%d/top-%d) or a null pointer -- refusing\n",
                hidden, n_expert, top_k, kH, kE, kTopK);
        return -1;
    }
    if (((uintptr_t)x_bf16 | (uintptr_t)router_w | (uintptr_t)shared_gate_w) & 15u) {
        fprintf(stderr, "pulsar: qwen router: bf16 rows must be 16-byte aligned -- refusing\n");
        return -1;
    }
    const dim3 lg((kE + 1 + 7) / 8, (T + kRouterTB - 1) / kRouterTB);
    qwen_router_logits_kernel<<<lg, 256, 0, stream>>>((const __nv_bfloat16 *)x_bf16, (const __nv_bfloat16 *)router_w,
                                                      (const __nv_bfloat16 *)shared_gate_w, T, logits);
    qwen_router_topk_kernel<<<(T + 3) / 4, 128, 0, stream>>>(logits, T, selected, weights, sgate);
    return launch_ok("router") ? 0 : -3;
}

extern "C" size_t pulsar_qwen_moe_workspace_bytes(int T) {
    return T > 0 ? moe_ws_layout(T, nullptr, 0, nullptr) : 0;
}

extern "C" int pulsar_qwen_moe_carve(int T, void *ws, size_t ws_bytes, pulsar_qwen_moe_parts *parts) {
    if (T <= 0 || !ws || !parts || moe_ws_layout(T, ws, ws_bytes, parts) == 0) {
        fprintf(stderr, "pulsar: qwen MoE: workspace %zu B < %zu B for %d rows -- refusing\n", ws_bytes,
                pulsar_qwen_moe_workspace_bytes(T), T);
        return -1;
    }
    return 0;
}

extern "C" size_t pulsar_rows_moe_shared_ws_bytes(int T, int in, int mid, int out) {
    return T > 0 && in > 0 && mid > 0 && out > 0 ? exl3_moe_shared_ws_bytes(T, in, mid, out) : 0;
}

extern "C" size_t pulsar_rows_moe_routed_workspace_bytes(int T, int top_k, int mid, int out_dim) {
    return T > 0 && top_k > 0 && mid > 0 && out_dim > 0 ? routed_ws_layout(T, top_k, mid, out_dim, nullptr, 0, nullptr) : 0;
}

/* A riding shared expert's descriptor in the front door's terms: three EXL3 slices hidden -> mid -> hidden. */
static bool rows_moe_shared(const pulsar_rows_moe_shared *q, int H, exl3_moe_shared *sh) {
    const int smid = q->gate.out;
    if (!q->gate.w || !q->up.w || !q->down.w || !q->gate.k2 || !q->up.k2 || !q->down.k2 || q->gate.in != H ||
        q->up.in != H || q->up.out != smid || q->down.in != smid || q->down.out != H) {
        fprintf(stderr, "pulsar: routed MoE (bf16 rows): a riding shared expert is three EXL3 slices %d -> %d -> %d "
                        "(got %d->%d / %d->%d / %d->%d) -- refusing\n", H, smid, H, q->gate.in, q->gate.out,
                q->up.in, q->up.out, q->down.in, q->down.out);
        return false;
    }
    *sh = exl3_moe_shared{q->gate.w, q->up.w, q->down.w, q->gate.k2, q->up.k2, q->down.k2, smid, q->scale, q->yg, q->yu,
                          q->ys, reinterpret_cast<__nv_bfloat16 *>(q->h), q->ws, q->ws_bytes};
    return true;
}

extern "C" int pulsar_rows_moe_shared_fork(const pulsar_rows_moe_shared *q, const uint16_t *x_bf16, int T,
                                           cudaStream_t stream) {
    exl3_moe_shared sh{};
    if (!q || !x_bf16 || T <= 0 || !rows_moe_shared(q, q->gate.in, &sh)) return -1;
#ifndef PULSAR_HAVE_MMQ
    (void)stream;
    fprintf(stderr, "pulsar: shared expert fork: built without the MMQ drivers the front door rides -- refusing\n");
    return -1;
#else
    return exl3_moe_shared_fork(sh, exl3_moe_act{EXL3_MOE_ACT_BF16_ROWS, x_bf16, nullptr, 0}, T, q->gate.in, q->down.out,
                                q->gate.prompt, stream);
#endif
}

extern "C" int pulsar_rows_moe_routed_launch(const pulsar_rows_moe *w, int32_t *sel, const float *wts,
                                             const uint16_t *x_bf16, int T, float *out, void *ws, size_t ws_bytes,
                                             uint32_t *nf_flag, uint32_t nf_code, cudaStream_t stream) {
    if (!w || !sel || !wts || !x_bf16 || !out || !nf_flag || T <= 0) {
        fprintf(stderr, "pulsar: routed MoE (bf16 rows): a null input -- refusing\n");
        return -1;
    }
    /* the routed experts are ONE of: EXL3 fused gate_up + down, EXL3 gate + up pair + down, Tessera (L255) */
    const bool tessera = w->tessera_gate != nullptr;
    const bool split = w->gate_table != nullptr;
    const bool exl3 = split || w->gate_up_table != nullptr || w->down_table != nullptr;
    if (tessera ? (!w->tessera_up || !w->tessera_down || exl3)
                : (!w->down_table || split != (w->up_table != nullptr) || split == (w->gate_up_table != nullptr))) {
        fprintf(stderr, "pulsar: routed MoE (bf16 rows): the experts must be ONE of an EXL3 fused gate_up table, an "
                        "EXL3 gate + up pair, or a Tessera gate / up / down triple -- refusing\n");
        return -1;
    }
    const int H = w->out_dim, mid = w->mid_dim, top_k = w->top_k, E = w->n_expert;
    if (w->in_dim != H || mid <= 0 || top_k <= 0 || top_k > E || w->n_local <= 0 || w->ex_lo < 0 ||
        w->ex_lo + w->n_local > E) {
        fprintf(stderr, "pulsar: routed MoE (bf16 rows): %d -> %d -> %d, top-%d of %d, experts [%d, +%d) -- refusing\n",
                w->in_dim, mid, H, top_k, E, w->ex_lo, w->n_local);
        return -1;
    }
    routed_ws m;
    if (!ws || routed_ws_layout(T, top_k, mid, H, ws, ws_bytes, &m) == 0) {
        fprintf(stderr, "pulsar: routed MoE (bf16 rows): workspace %zu B < %zu B for %d rows -- refusing\n",
                ws_bytes, pulsar_rows_moe_routed_workspace_bytes(T, top_k, mid, H), T);
        return -1;
    }
    const int64_t pairs = (int64_t)T * top_k;
    const int nE = w->n_local;   /* the experts this rank's tables hold */
#ifndef PULSAR_HAVE_MMQ
    (void)nf_code; (void)stream; (void)pairs; (void)nE; (void)tessera;
    fprintf(stderr, "pulsar: routed MoE (bf16 rows): built without the MMQ drivers the routed arms ride -- refusing\n");
    return -1;
#else
    if (tessera && w->shared) {
        fprintf(stderr, "pulsar: routed MoE (bf16 rows): a shared expert rides the EXL3 front door only, not the Tessera "
                        "arm -- refusing\n");
        return -1;
    }
    if (tessera) {
        /* L255: Tessera's fused window kernel on the router's own selection -- routing prep, gate/up + SwiGLU,
         * weighted down, fixed-order token sum -- then the block's f32 routed term.  Its workspace and its bf16
         * output live in the EXL3 arm's buffers (down_z, gu_z): exactly one routed arm runs.  No expert range:
         * the stacks must hold every expert (expert parallelism is the EXL3 arm's alone today). */
        static int announced_tessera = 0;
        if (!announced_tessera) {
            announced_tessera = 1;
            fprintf(stderr, "pulsar: routed MoE (bf16 rows) = Tessera routed fused window kernel (value family, gate/up "
                            "tile_words %d, down %d), top-%d of %d\n",
                    w->tessera_gate->tile_words, w->tessera_down->tile_words, top_k, E);
        }
        const size_t tws = pulsar_tessera_moe_workspace_bytes(w->tessera_gate, w->tessera_down, T, top_k);
        const size_t have_ws = (size_t)pairs * H * 4, have_out = (size_t)pairs * 2 * mid * 4;
        if (nE != E || w->ex_lo != 0 || w->tessera_gate->E != E || w->tessera_gate->K != H ||
            w->tessera_gate->N != mid || tws > have_ws || (size_t)T * H * 2 > have_out) {
            fprintf(stderr, "pulsar: routed MoE (bf16 rows): the Tessera stacks are E %d, %d -> %d (the model's %d, %d -> %d) "
                            "over experts [%d, +%d), or their workspace %zu B exceeds the routed arm's %zu B -- refusing\n",
                    w->tessera_gate->E, w->tessera_gate->K, w->tessera_gate->N, E, H, mid, w->ex_lo, nE, tws, have_ws);
            return -1;
        }
        uint16_t *routed = (uint16_t *)m.gu_z;
        const int rc = pulsar_tessera_moe_launch(w->tessera_gate, w->tessera_up, w->tessera_down, x_bf16, T, top_k, sel,
                                                 wts, routed, m.down_z, have_ws, stream);
        if (rc) { fprintf(stderr, "pulsar: routed MoE (bf16 rows): the Tessera routed launch refused (rc=%d) -- no fallback\n", rc); return -1; }
        const size_t n = (size_t)T * H;
        qwen_tessera_routed_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(out, routed, n, nf_flag, nf_code);
        return launch_ok("tessera routed term") ? 0 : -3;
    }
    static int announced[2] = {0, 0};
    if (!announced[split]) {
        announced[split] = 1;
        fprintf(stderr, "pulsar: routed MoE (bf16 rows) = EXL3 %s K=%g, down K=%g, top-%d of %d (experts [%d, +%d)) "
                        "through the EXL3 routed front door (exl3_moe.cuh)\n",
                split ? "gate + up pair" : "fused gate_up", w->k2_gate_up / 2.0, w->k2_down / 2.0, top_k, E, w->ex_lo,
                w->n_local);
    }
    if (nE != E) {
        qwen_ep_localize_kernel<<<(unsigned)((pairs + 255) / 256), 256, 0, stream>>>(sel, (int)pairs, w->ex_lo, nE);
        if (!launch_ok("expert localize")) return -3;
    }
    /* L251 / ac69748f: the block input's bf16 row and the fold's bf16 output -- no E4M3 slot in this lane */
    const exl3_moe_stacks st{w->gate_up_table, w->gate_table, w->up_table, w->down_table, w->k2_gate_up, w->k2_down,
                             nE, H, mid, H};
    const exl3_moe_bufs bufs{m.gu_z, split ? m.gu_z + (size_t)pairs * mid : nullptr, m.down_z,
                             exl3_moe_act{EXL3_MOE_ACT_BF16_ROWS, m.mid_x, nullptr, 0}};
    exl3_moe_shared sh{};
    if (w->shared && !rows_moe_shared(w->shared, H, &sh)) return -1;
    if (exl3_moe_routed_launch(st, sel, wts, T, top_k, 0.0f, exl3_moe_act{EXL3_MOE_ACT_BF16_ROWS, x_bf16, nullptr, 0},
                               bufs, out, nf_flag, nf_code, w->prompt, stream, w->shared ? &sh : nullptr) != 0)
        return -1;
    /* DIAGNOSTIC (PULSAR_MOE_SPILL_MID=<prefix>): the fold's bf16 output -- the down's input -- so the xcheck
     * can compare the DEVICE's own values against its emulation's per element.  Armed only by the env var;
     * nothing runs and nothing allocates when it is unset. */
    { const char *sp = getenv("PULSAR_MOE_SPILL_MID");
      if (sp && sp[0] && T > 1) {   /* the batch call; the M=1 loop would overwrite it */
          const size_t nb = (size_t)pairs * (size_t)mid * 2u;
          uint8_t *hb = (uint8_t *)malloc(nb);
          cudaStreamSynchronize(stream);
          if (hb && cudaMemcpy(hb, m.mid_x, nb, cudaMemcpyDeviceToHost) == cudaSuccess) {
              char fp[1024];
              snprintf(fp, sizeof(fp), "%s.mid_x.bin", sp);
              FILE *f = fopen(fp, "wb"); if (f) { fwrite(hb, 1, nb, f); fclose(f); }
          }
          free(hb);
      }
    }
    return 0;
#endif
}

extern "C" int pulsar_qwen_moe_shared_launch(const pulsar_rows_linear *gate, const pulsar_rows_linear *up,
                                             const pulsar_rows_linear *down, const uint16_t *x_bf16, int T,
                                             float *out, const pulsar_qwen_moe_parts *p, cudaStream_t stream) {
    const int smid = gate ? gate->out : 0;
    if (!gate || !up || !down || !x_bf16 || !out || !p || T <= 0 || gate->in != kH || up->in != kH || up->out != smid ||
        down->in != smid || down->out != kH || smid != PULSAR_QWEN_SHARED_MID) {
        fprintf(stderr, "pulsar: qwen MoE: shared expert %d->%d / %d->%d / %d->%d, built for %d->%d->%d -- refusing\n",
                gate ? gate->in : -1, smid, up ? up->in : -1, up ? up->out : -1, down ? down->in : -1,
                down ? down->out : -1, kH, PULSAR_QWEN_SHARED_MID, kH);
        return -1;
    }
    /* gate + up on the same row, the SwiGLU producer, down, then out += sigmoid(w_sg . x) * shared */
    int rc = pulsar_rows_linear_launch(gate, x_bf16, T, p->yg, p->lin, p->lin_bytes, stream);
    if (!rc) rc = pulsar_rows_linear_launch(up, x_bf16, T, p->yu, p->lin, p->lin_bytes, stream);
    if (rc) return rc;
    qwen_swiglu_emit_kernel<<<T, 256, 0, stream>>>(p->yg, p->yu, smid, (__nv_bfloat16 *)p->h_x);
    if (!launch_ok("shared swiglu")) return -3;
    rc = pulsar_rows_linear_launch(down, p->h_x, T, p->ys, p->lin, p->lin_bytes, stream);
    if (rc) return rc;
    const size_t n = (size_t)T * kH;
    qwen_shared_add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(out, p->ys, p->sgate, T);
    return launch_ok("shared add") ? 0 : -3;
}

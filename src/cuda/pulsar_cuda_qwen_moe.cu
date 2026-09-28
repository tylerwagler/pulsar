/* Qwen3.8-Flash-Next MoE block (L251 S4): the router, the EXL3 routed
 * experts, the sigmoid-gated EXL3 shared expert, and the fixed-order sum.
 * Contracts in pulsar_cuda_qwen.h; the source is transformers'
 * Qwen4ExpTextSparseMoeBlock / Qwen4ExpTextTopKRouter.
 *
 *   router     logits = W_r x in f32 (bf16 x bf16, fixed order), rounded to
 *              bf16; softmax; top-10 by probability; renormalised; bf16
 *   routed     the L245 EXL3 arm with Qwen's FUSED gate_up (one [2560 -> 1280]
 *              slice per expert, gate rows then up rows, one suh -- the layout
 *              the graded quant produced and the container carries):
 *              ds4_exl3_moe_fused (one GEMV, the input rotated in-kernel), the
 *              fused fold (no clamp; the router weight folded into the mid
 *              before its E4M3 encode, as on DeepSeek), the down on the fold's
 *              pre-rotated mid (640 = 5 x 128, the arm's K % 128 tail), the
 *              slot-ordered sum with the output rotation
 *   shared     the EXL3 dense arm on the same E4M3 slot (gate, up), SwiGLU
 *              encoded by its producer, the dense down
 *   out        routed + sigmoid(w_sg . x) * shared -- in that order
 */
#include "pulsar_cuda_qwen.h"
#include "pulsar_cuda_mx.cuh"
#include "mmq/ds4_exl3_dense.cuh"
#ifdef PULSAR_HAVE_MMQ
#include "mmq/ds4_mmq.h"
#include "mmq/ds4_exl3_gemv.cuh"
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

/* The shared expert's SwiGLU, encoded where it is produced (rule 3). */
__global__ void __launch_bounds__(256)
qwen_swiglu_emit_kernel(const float *__restrict__ g, const float *__restrict__ u, int D,
                        __nv_fp8_e4m3 *__restrict__ q, unsigned char *__restrict__ sf, int kbp) {
    const int t = blockIdx.x;
    for (int c0 = 0; c0 < D; c0 += 256) {
        const int c = c0 + (int)threadIdx.x;
        const float v = c < D ? pulsar_swiglu_elem(g[(size_t)t * D + c], u[(size_t)t * D + c], 1.0f, 0.0f) : 0.0f;
        pulsar_mx_emit_block(v, (uint32_t)c, (uint32_t)t, (uint32_t)D, kbp, q, sf);
    }
}

/* out = routed + sgate * shared, in that order. */
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

struct moe_ws {
    float *logits, *wts, *sgate, *gu_z, *down_z, *yg, *yu, *ys;
    int32_t *sel;
    uint8_t *mid_q, *mid_sf, *h_q, *h_sf;
    void *lin;
    size_t lin_bytes;
};

/* The one layout of the MoE workspace: sizing and carving are the same walk. */
static size_t moe_ws_layout(int T, void *base, size_t cap, moe_ws *o) {
    ws_bump b{(uint8_t *)base, base ? cap : (size_t)-1, 0, false};
    const size_t pairs = (size_t)T * kTopK;
    const int mid = PULSAR_QWEN_EXPERT_MID, smid = PULSAR_QWEN_SHARED_MID;
    moe_ws m{};
    m.logits = (float *)b.take((size_t)T * (kE + 1) * 4);
    m.sel    = (int32_t *)b.take(pairs * 4);
    m.wts    = (float *)b.take(pairs * 4);
    m.sgate  = (float *)b.take((size_t)T * 4);
    m.gu_z   = (float *)b.take(pairs * 2 * mid * 4);
    m.mid_q  = (uint8_t *)b.take(pairs * mid);
    m.mid_sf = (uint8_t *)b.take(pulsar_mx_sf_slab_bytes((int)pairs, pulsar_mx_kbp(mid)));
    m.down_z = (float *)b.take(pairs * kH * 4);
    m.yg     = (float *)b.take((size_t)T * smid * 4);
    m.yu     = (float *)b.take((size_t)T * smid * 4);
    m.h_q    = (uint8_t *)b.take((size_t)T * smid);
    m.h_sf   = (uint8_t *)b.take(pulsar_mx_sf_slab_bytes(T, pulsar_mx_kbp(smid)));
    m.ys     = (float *)b.take((size_t)T * kH * 4);
    /* the dense arm's workspace is a function of (rows, in, out) alone */
    size_t lb = ds4_exl3_dense_workspace_bytes(T, kH, smid);
    const size_t ld = ds4_exl3_dense_workspace_bytes(T, smid, kH);
    lb = lb > ld ? lb : ld;
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

extern "C" size_t pulsar_qwen_linear_workspace_bytes(const pulsar_qwen_linear *l, int rows) {
    if (!l || rows <= 0) return 0;
    return ds4_exl3_dense_workspace_bytes(rows, l->in, l->out);
}

extern "C" int pulsar_qwen_linear_launch(const pulsar_qwen_linear *l, const pulsar_qwen_slot *x, int rows, float *y,
                                         void *ws, size_t ws_bytes, cudaStream_t stream) {
    if (!l || !l->w || !x || !x->q || !x->sf || x->kbp != pulsar_mx_kbp(l->in)) {
        fprintf(stderr, "pulsar: qwen linear: no weight or no E4M3 slot of width %d -- refusing\n", l ? l->in : -1);
        return -1;
    }
    if (l->k2 == 0) {                       /* the recipe's mxfp8_lt dense tier */
        if (!l->sf) {
            fprintf(stderr, "pulsar: qwen linear: a %d->%d tensor is neither EXL3 nor mxfp8_lt -- refusing\n",
                    l->in, l->out);
            return -1;
        }
        const pulsar_qwen_lowrank lr{l->w, l->sf, l->out, l->in};
        return pulsar_qwen_mxfp8_linear_launch(&lr, x, rows, y, ws, ws_bytes, stream);
    }
    return ds4_exl3_dense_launch(l->w, l->k2, x->q, x->sf, y, rows, l->in, l->out, ws, ws_bytes, stream);
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

extern "C" int pulsar_qwen_moe_launch(const pulsar_qwen_moe_dev *w, const uint16_t *x_bf16, const pulsar_qwen_slot *x,
                                      int T, float *out, void *ws, size_t ws_bytes,
                                      uint32_t *nf_flag, uint32_t nf_code, cudaStream_t stream) {
    if (!w || !x_bf16 || !x || !x->q || !x->sf || !out || !nf_flag || T <= 0 ||
        !w->gate_up_table || !w->down_table || x->kbp != pulsar_mx_kbp(kH)) {
        fprintf(stderr, "pulsar: qwen MoE: a null input or a slot that is not %d wide -- refusing\n", kH);
        return -1;
    }
    const int smid = w->shared_gate.out;
    if (w->shared_gate.in != kH || w->shared_up.in != kH || w->shared_up.out != smid ||
        w->shared_down.in != smid || w->shared_down.out != kH || smid != PULSAR_QWEN_SHARED_MID) {
        fprintf(stderr, "pulsar: qwen MoE: shared expert %d->%d / %d->%d / %d->%d, built for %d->%d->%d -- refusing\n",
                w->shared_gate.in, w->shared_gate.out, w->shared_up.in, w->shared_up.out,
                w->shared_down.in, w->shared_down.out, kH, PULSAR_QWEN_SHARED_MID, kH);
        return -1;
    }
#ifndef PULSAR_HAVE_MMQ
    (void)ws; (void)ws_bytes; (void)nf_code; (void)stream;
    fprintf(stderr, "pulsar: qwen MoE: built without the MMQ drivers the EXL3 routed arm rides -- refusing\n");
    return -1;
#else
    moe_ws m;
    if (!ws || moe_ws_layout(T, ws, ws_bytes, &m) == 0) {
        fprintf(stderr, "pulsar: qwen MoE: workspace %zu B < %zu B for %d rows -- refusing\n",
                ws_bytes, pulsar_qwen_moe_workspace_bytes(T), T);
        return -1;
    }
    static int mmq_ready = -1;
    if (mmq_ready < 0) {
        int dev = 0;
        (void)cudaGetDevice(&dev);
        mmq_ready = ds4_mmq_init(dev) == 0 ? 1 : 0;
    }
    if (!mmq_ready) {
        fprintf(stderr, "pulsar: qwen MoE: the MMQ drivers are unavailable on this device -- refusing\n");
        return -1;
    }
    static int announced = 0;
    if (!announced) {
        announced = 1;
        fprintf(stderr, "pulsar: L251 qwen MoE = bf16 router softmax top-%d of %d + EXL3 routed trellis GEMV "
                        "(fused gate_up K=%g, down K=%g) + EXL3 dense shared expert (K=%g/%g/%g), sigmoid-gated\n",
                kTopK, kE, w->k2_gate_up / 2.0, w->k2_down / 2.0, w->shared_gate.k2 / 2.0,
                w->shared_up.k2 / 2.0, w->shared_down.k2 / 2.0);
    }
    const int mid = PULSAR_QWEN_EXPERT_MID;
    const int64_t pairs = (int64_t)T * kTopK;
    int rc = pulsar_qwen_router_launch(x_bf16, w->router_w, w->shared_gate_w, T, kH, kE, kTopK,
                                       m.logits, m.sel, m.wts, m.sgate, stream);
    if (rc) return rc;

    /* routed: the L245 arm's four launches */
    const int mid_kbp = pulsar_mx_kbp(mid);
    cudaMemsetAsync(m.mid_sf, 0, pulsar_mx_sf_slab_bytes((int)pairs, mid_kbp), stream);
    rc = ds4_exl3_moe_fused(w->gate_up_table, w->k2_gate_up, m.sel, m.gu_z, 2 * mid, kH, T, kE, kTopK, stream,
                            x->q, x->sf, x->kbp);
    if (rc) { fprintf(stderr, "pulsar: qwen MoE gate_up declined (rc=%d) -- no fallback\n", rc); return -1; }
    rc = ds4_exl3_moe_fold_fused_launch(m.gu_z, m.sel, m.wts, w->gate_up_table, w->down_table,
                                        kH, mid, pairs, 0.0f, m.mid_q, m.mid_sf, mid_kbp, stream);
    if (rc) return -1;
    /* DIAGNOSTIC (PULSAR_MOE_SPILL_MID=<prefix>): the fold's A8 output -- the down GEMV's input --
     * so the xcheck can compare the DEVICE's own codes against its emulation's per element.  The
     * one place those may legitimately differ is an E4M3 rounding TIE: the fold's terms are ~0.03
     * and its input ~5.9e-05, so the fold's fp32-vs-fp64 difference (~1e-8 absolute) is ~1.7e-4
     * RELATIVE, and scaled by 2^19 that is ~3e-3 -- enough to choose the other code wherever a
     * value lands within ~1e-5 of a midpoint, which an E4M3 code comparison can see and an
     * absolute epsilon on the output cannot.  Armed only by the env var; nothing runs and nothing
     * allocates when it is unset. */
    { const char *sp = getenv("PULSAR_MOE_SPILL_MID");
      if (sp && sp[0] && T > 1) {   /* the batch call; the M=1 loop would overwrite it */
          const size_t nq = (size_t)pairs * (size_t)mid;
          const size_t ns = pulsar_mx_sf_slab_bytes((int)pairs, mid_kbp);
          uint8_t *hq = (uint8_t *)malloc(nq), *hs = (uint8_t *)malloc(ns);
          cudaStreamSynchronize(stream);
          if (hq && hs && cudaMemcpy(hq, m.mid_q, nq, cudaMemcpyDeviceToHost) == cudaSuccess &&
                          cudaMemcpy(hs, m.mid_sf, ns, cudaMemcpyDeviceToHost) == cudaSuccess) {
              char fp[1024]; FILE *f;
              snprintf(fp, sizeof(fp), "%s.mid_q.bin", sp);  f = fopen(fp, "wb"); if (f) { fwrite(hq, 1, nq, f); fclose(f); }
              snprintf(fp, sizeof(fp), "%s.mid_sf.bin", sp); f = fopen(fp, "wb"); if (f) { fwrite(hs, 1, ns, f); fclose(f); }
          }
          free(hq); free(hs);
      }
    }
    rc = ds4_exl3_moe_single(w->down_table, w->k2_down, m.sel, m.down_z, kH, mid, (int)pairs, kE, 1,
                             stream, m.mid_q, m.mid_sf, mid_kbp);
    if (rc) { fprintf(stderr, "pulsar: qwen MoE down declined (rc=%d) -- no fallback\n", rc); return -1; }
    rc = ds4_exl3_moe_sum_launch(out, m.down_z, m.sel, w->down_table, mid, kH, kTopK, T, nf_flag, nf_code, stream);
    if (rc) return -1;

    /* shared: gate + up on the same slot, the SwiGLU producer, down */
    rc = pulsar_qwen_linear_launch(&w->shared_gate, x, T, m.yg, m.lin, m.lin_bytes, stream);
    if (!rc) rc = pulsar_qwen_linear_launch(&w->shared_up, x, T, m.yu, m.lin, m.lin_bytes, stream);
    if (rc) return rc;
    const int h_kbp = pulsar_mx_kbp(smid);
    cudaMemsetAsync(m.h_sf, 0, pulsar_mx_sf_slab_bytes(T, h_kbp), stream);
    qwen_swiglu_emit_kernel<<<T, 256, 0, stream>>>(m.yg, m.yu, smid, (__nv_fp8_e4m3 *)m.h_q, m.h_sf, h_kbp);
    if (!launch_ok("shared swiglu")) return -3;
    const pulsar_qwen_slot h = {m.h_q, m.h_sf, h_kbp};
    rc = pulsar_qwen_linear_launch(&w->shared_down, &h, T, m.ys, m.lin, m.lin_bytes, stream);
    if (rc) return rc;
    const size_t n = (size_t)T * kH;
    qwen_shared_add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(out, m.ys, m.sgate, T);
    return launch_ok("shared add") ? 0 : -3;
#endif
}

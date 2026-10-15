/* The rows MoE block's pieces over bf16 rows (L251 S4, family-neutral since L272 P4c / L295 F2): the routed experts
 * through the EXL3 front door (or Tessera), N shared experts, and the block's one carved workspace.  The engine
 * composes the block (Qwen: family_qwen_s4.cpp pulsar_qwen_s4_moe) as
 *
 *   router     pulsar_router_logits_launch + pulsar_router_select_launch (pulsar_cuda_router.cu) with the
 *              family's declared cfg: the picks, their weights and one gamma plane per gated shared expert
 *   routed     pulsar_rows_moe_routed_launch -- the core front door's arm over bf16 rows (moe.cpp),
 *              family-neutral, shapes from the model: Tessera, or the EXL3 routed front door (mmq/exl3_moe.cuh,
 *              the one every family's EXL3 experts take) with a FUSED gate_up or a gate + up PAIR, no clamp, the
 *              fold's mid emitted bf16; expert parallelism localises the picks first
 *   shared     each shared expert either RIDES the EXL3 routed launches (pulsar_rows_moe_shared_fork, joined at
 *              the routed sum) or runs after it (pulsar_rows_moe_shared_launch): gate, up, the SwiGLU emitted
 *              bf16 by its producer, the down
 *   out        routed + gamma_0 * shared_0 + gamma_1 * shared_1 + ... -- in that order
 *
 * L251 / ac69748f: there is no E4M3 activation slot on this lane. */
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


/* The shared expert's SwiGLU, emitted in the format its consumer reads (rule 3).  L251 / ac69748f:
 * the shared down projection reads a bf16 row, so the producer emits bf16 -- and bf16 carries its own
 * exponent, so there is no per-32 block and no scale slab any more. */
__global__ void __launch_bounds__(256)
rows_swiglu_emit_kernel(const float *__restrict__ g, const float *__restrict__ u, int D,
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

/* L266 EP: a pick of expert e becomes this rank's local index e - base, or -1 when another rank owns it
 * (the routed arms drop -1, mm_ids_helper by range and the fold / sum explicitly). */
__global__ void qwen_ep_localize_kernel(int32_t *__restrict__ sel, int n, int base, int n_local) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int e = sel[i] - base;
    sel[i] = e >= 0 && e < n_local ? e : -1;
}

/* out = out + scale * shared (scale 1 when the expert has no gate), added after the routed sum. */
__global__ void rows_shared_add_kernel(float *__restrict__ out, const float *__restrict__ ys,
                                       const float *__restrict__ scale, int T, int H) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * H) return;
    const float s = (scale ? scale[i / H] : 1.0f) * ys[i];
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

/* The one layout of a rows MoE block's workspace (the router's outputs, the routed region, the shared experts):
 * sizing and carving are the same walk.  Each shared expert has its own rows (a pending fork's buffers); the
 * dense arm's workspace is one, shared. */
static size_t moe_ws_layout(const pulsar_moe_block_shape *s, int T, void *base, size_t cap,
                            pulsar_moe_block_parts *o) {
    ws_bump b{(uint8_t *)base, base ? cap : (size_t)-1, 0, false};
    const size_t pairs = (size_t)T * s->top_k;
    const size_t S = (size_t)s->n_shared, smid = (size_t)s->shared_mid, H = (size_t)s->hidden;
    pulsar_moe_block_parts m{};
    m.logits = (float *)b.take((size_t)T * s->n_logit_rows * 4);
    m.sel    = (int32_t *)b.take(pairs * 4);
    m.wts    = (float *)b.take(pairs * 4);
    m.gammas = (float *)b.take(S * T * 4);
    m.routed_bytes = routed_ws_layout(T, s->top_k, s->expert_mid, s->hidden, nullptr, 0, nullptr);
    m.routed = b.take(m.routed_bytes);
    if (S) {
        m.yg = (float *)b.take(S * T * smid * 4);
        m.yu = (float *)b.take(S * T * smid * 4);
        m.h  = (uint16_t *)b.take(S * T * smid * 2);   /* bf16: no per-32 block, no scale slab */
        m.ys = (float *)b.take(S * T * H * 4);
        /* the dense arm's workspace is a function of (rows, in, out) alone: a riding shared expert's (gate and up
         * in one launch), which covers the separate launches' */
        m.lin_bytes = exl3_moe_shared_ws_bytes(T, s->hidden, s->shared_mid, s->hidden);
        m.lin = b.take(m.lin_bytes);
    }
    if (o) *o = m;
    return b.failed ? 0 : b.used;
}

static bool block_shape_ok(const pulsar_moe_block_shape *s) {
    return s && s->n_expert > 0 && s->top_k > 0 && s->n_logit_rows >= s->n_expert && s->hidden > 0 &&
           s->expert_mid > 0 && s->n_shared >= 0 && s->n_shared <= PULSAR_ROUTER_SHARED_MAX &&
           (s->n_shared == 0 || s->shared_mid > 0);
}

static bool launch_ok(const char *what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: rows MoE %s: launch failed: %s\n", what, cudaGetErrorString(e));
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
        return pulsar_qwen_mxfp8_linear_launch(&lr, x_bf16, rows, y, ws, ws_bytes, stream);
    }
    return ds4_exl3_dense_launch(l->w, l->k2, x_bf16, y, rows, l->in, l->out, ws, ws_bytes, stream, l->prompt);
}

extern "C" size_t pulsar_rows_linear_multi_workspace_bytes(const pulsar_rows_linear *const *ls, int n, int rows) {
    if (!ls || n < 1 || n > DS4_EXL3_DENSE_MAX_PROJ || rows <= 0) return 0;
    ds4_exl3_dense_proj p[DS4_EXL3_DENSE_MAX_PROJ] = {};
    int ne = 0;
    for (int j = 0; j < n; ++j) {
        if (!ls[j] || ls[j]->in != ls[0]->in) return 0;
        /* an EXL3 Linear (any logical width, L295 F7), or a width an EXL3 one could have (mxfp8_lt needs none) */
        if (ls[j]->k2 || ls[j]->out % 128 == 0) p[ne++].N = ls[j]->out;
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

extern "C" size_t pulsar_moe_block_workspace_bytes(const pulsar_moe_block_shape *s, int T) {
    return T > 0 && block_shape_ok(s) ? moe_ws_layout(s, T, nullptr, 0, nullptr) : 0;
}

extern "C" int pulsar_moe_block_carve(const pulsar_moe_block_shape *s, int T, void *ws, size_t ws_bytes,
                                      pulsar_moe_block_parts *parts) {
    if (T <= 0 || !ws || !parts || !block_shape_ok(s) || moe_ws_layout(s, T, ws, ws_bytes, parts) == 0) {
        fprintf(stderr, "pulsar: MoE block: workspace %zu B < %zu B for %d rows, or not a block shape -- refusing\n",
                ws_bytes, pulsar_moe_block_workspace_bytes(s, T), T);
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
    if (w->n_shared < 0 || w->n_shared > PULSAR_ROUTER_SHARED_MAX || (w->n_shared > 0) != (w->shared != nullptr) ||
        (tessera && w->n_shared)) {
        fprintf(stderr, "pulsar: routed MoE (bf16 rows): %d riding shared experts (0..%d, with their descriptors; they "
                        "ride the EXL3 front door only, not the Tessera arm) -- refusing\n", w->n_shared,
                PULSAR_ROUTER_SHARED_MAX);
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
    exl3_moe_shared sh[PULSAR_ROUTER_SHARED_MAX]{};
    for (int j = 0; j < w->n_shared; ++j)
        if (!rows_moe_shared(&w->shared[j], H, &sh[j])) return -1;
    if (exl3_moe_routed_launch(st, sel, wts, T, top_k, 0.0f, exl3_moe_act{EXL3_MOE_ACT_BF16_ROWS, x_bf16, nullptr, 0},
                               bufs, out, nf_flag, nf_code, w->prompt, stream, w->n_shared ? sh : nullptr,
                               w->n_shared) != 0)
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

extern "C" int pulsar_rows_moe_shared_launch(const pulsar_rows_moe_shared *q, const uint16_t *x_bf16, int T,
                                             float *out, cudaStream_t stream) {
    const int smid = q ? q->gate.out : 0, H = q ? q->gate.in : 0;
    if (!q || !x_bf16 || !out || T <= 0 || !q->yg || !q->yu || !q->ys || !q->h || q->up.in != H ||
        q->up.out != smid || q->down.in != smid || q->down.out != H) {
        fprintf(stderr, "pulsar: MoE shared expert: %d->%d / %d->%d / %d->%d (in -> mid -> in), its buffers, rows and an "
                        "output -- refusing\n", H, smid, q ? q->up.in : -1, q ? q->up.out : -1, q ? q->down.in : -1,
                q ? q->down.out : -1);
        return -1;
    }
    /* gate + up on the same row, the SwiGLU producer, down, then out += scale * shared */
    int rc = pulsar_rows_linear_launch(&q->gate, x_bf16, T, q->yg, q->ws, q->ws_bytes, stream);
    if (!rc) rc = pulsar_rows_linear_launch(&q->up, x_bf16, T, q->yu, q->ws, q->ws_bytes, stream);
    if (rc) return rc;
    rows_swiglu_emit_kernel<<<T, 256, 0, stream>>>(q->yg, q->yu, smid, (__nv_bfloat16 *)q->h);
    if (!launch_ok("shared swiglu")) return -3;
    rc = pulsar_rows_linear_launch(&q->down, q->h, T, q->ys, q->ws, q->ws_bytes, stream);
    if (rc) return rc;
    const size_t n = (size_t)T * H;
    rows_shared_add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(out, q->ys, q->scale, T, H);
    return launch_ok("shared add") ? 0 : -3;
}

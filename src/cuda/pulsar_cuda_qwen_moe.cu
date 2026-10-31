/** @file
 *  The rows MoE block over bf16 rows: routed experts through the format front doors, shared experts, workspace. */
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
 *              bf16 by its producer, the down -- or (L296 N4, pulsar_ffn_act RELU2) up, relu^2, the down
 *   latent     (L296 N4) a latent block's routed experts run on the latent width: the block's workspace carries the
 *              latent rows and the projections' workspace (moe.cpp composes fc1 -> routed -> fc2)
 *   out        routed + gamma_0 * shared_0 + gamma_1 * shared_1 + ... -- in that order
 *
 * L251 / ac69748f: there is no E4M3 activation slot on this lane. */
#include "pulsar_cuda_qwen.h"
#include "pulsar_cuda_mx.cuh"
#include "mmq/ds4_exl3_dense.cuh"
#include "mmq/exl3_moe.cuh"
#include "pulsar_cuda_bf16_rows.cuh"
#include "pulsar_cuda_nvfp4.h"
#include "pulsar_cuda_f16_rows.h"
#include "pulsar_cuda_rowops.h"
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
/* L296 N4: the same producer for the non-gated activation (pulsar_ffn_act RELU2): relu(u)^2, g unread. */
/** The shared expert's activation, emitted bf16: SwiGLU silu(g) * u, or (act RELU2) relu(u)^2 with g unread.
 *  Grid = T blocks (one per row), block = 256 threads striding the D channels.
 *  @param g   f32 [T][D] gate projection (unread under RELU2)
 *  @param u   f32 [T][D] up projection
 *  @param D   the expert's intermediate width
 *  @param act a pulsar_ffn_act value
 *  @param xb  out, bf16 [T][D] the down projection's input */
__global__ void __launch_bounds__(256)
rows_act_emit_kernel(const float *__restrict__ g, const float *__restrict__ u, int D, int act,
                     __nv_bfloat16 *__restrict__ xb) {
    const int t = blockIdx.x;
    for (int c0 = 0; c0 < D; c0 += 256) {
        const int c = c0 + (int)threadIdx.x;
        if (c < D) {
            const size_t i = (size_t)t * D + c;
            xb[i] = __float2bfloat16(act == PULSAR_FFN_ACT_RELU2 ? pulsar_relu2_elem(u[i], 1.0f)
                                                                 : pulsar_swiglu_elem(g[i], u[i], 1.0f, 0.0f));
        }
    }
}

/* L266 EP: a pick of expert e becomes this rank's local index e - base, or -1 when another rank owns it
 * (the routed arms drop -1, mm_ids_helper by range and the fold / sum explicitly). */
/** Expert parallelism: map each pick to this rank's local index in place.  One thread per pick, any 1-D grid
 *  covering n.
 *  @param sel     int32 [n] picks (global expert ids in, local ids or -1 out)
 *  @param n       picks (T * top_k)
 *  @param base    first expert this rank holds (ex_lo)
 *  @param n_local experts this rank holds */
__global__ void qwen_ep_localize_kernel(int32_t *__restrict__ sel, int n, int base, int n_local) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int e = sel[i] - base;
    sel[i] = e >= 0 && e < n_local ? e : -1;
}

/* out = out + scale * shared (scale 1 when the expert has no gate), added after the routed sum. */
/** out += scale[t] * ys elementwise.  One thread per element, grid = ceil(T * H / 256), block = 256.
 *  @param out   f32 [T][H] the routed sum, updated in place
 *  @param ys    f32 [T][H] the shared expert's down output
 *  @param scale f32 [T] the shared expert's gate (router gamma), or NULL for 1
 *  @param T     rows
 *  @param H     hidden width */
__global__ void rows_shared_add_kernel(float *__restrict__ out, const float *__restrict__ ys,
                                       const float *__restrict__ scale, int T, int H) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * H) return;
    const float s = (scale ? scale[i / H] : 1.0f) * ys[i];
    out[i] = out[i] + s;
}

/* A bump allocator over the caller's workspace (256-byte aligned slices). */
/// A bump allocator over the caller's workspace: 256-byte aligned slices; base NULL only sizes.
struct ws_bump {
    /// Workspace start, or NULL to size only.
    uint8_t *base;
    /** @var cap
     *  Workspace bytes ((size_t)-1 when sizing).
     *  @var used
     *  Bytes handed out so far (end of the last slice). */
    size_t cap, used;
    /// A take() overran cap; every later take() fails too.
    bool failed;
    /** Take the next 256-B aligned slice.
     *  @param bytes slice size
     *  @return the slice, or nullptr when sizing or on overflow (then `failed` is set) */
    void *take(size_t bytes) {
        const size_t off = (used + 255) & ~(size_t)255;
        if (failed || off + bytes > cap) { failed = true; return nullptr; }
        used = off + bytes;
        return base ? base + off : nullptr;
    }
};

/* L272 P4c: the routed part's workspace (pulsar_rows_moe_routed_launch): the gate / up outputs, the fold's bf16
 * rows, the down outputs -- a function of the rows and the routing shape alone (the model's, not a family's). */
/// The routed part's workspace pieces, carved by routed_ws_layout.
struct routed_ws {
    /** @var gu_z
     *  f32 [T * top_k][2 * mid] the gate / up outputs.
     *  @var down_z
     *  f32 [T * top_k][out_dim] the down outputs. */
    float *gu_z, *down_z;
    /// bf16 [T * top_k][mid] the routed fold's rows (the down projection's input).
    uint8_t *mid_x;                 /* the routed fold's bf16 rows (ac69748f) */
};
/** Lay the routed part's workspace out (sizing and carving are the same walk).
 *  @param T       rows
 *  @param top_k   picks per row
 *  @param mid     expert intermediate width
 *  @param out_dim expert output width
 *  @param base    workspace start, or nullptr to size only
 *  @param cap     workspace bytes (checked when base is non-null)
 *  @param o       out: the pieces, or nullptr
 *  @return bytes used, 0 when they exceed cap */
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
/** Widen Tessera's bf16 routed sum to the block's f32 term and flag non-finite values.  One thread per element,
 *  grid = ceil(n / 256), block = 256.
 *  @param out     f32 [n] the routed term
 *  @param routed  bf16 [n] Tessera's token_sum output ([T][H])
 *  @param n       elements (T * H)
 *  @param nf_flag device flag: nf_code is CAS'd in on the first non-finite value
 *  @param nf_code the code to record */
__global__ static void qwen_tessera_routed_kernel(float *__restrict__ out, const uint16_t *__restrict__ routed, size_t n,
                                                  uint32_t *__restrict__ nf_flag, uint32_t nf_code) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __uint_as_float((uint32_t)routed[i] << 16);
    out[i] = v;
    if (!isfinite(v)) atomicCAS(nf_flag, 0u, nf_code);
}

/** @param ws       the routed workspace the last pulsar_rows_moe_routed_launch used
 *  @param ws_bytes its bytes
 *  @param T        rows of that launch
 *  @param top_k    picks per row
 *  @param mid      expert intermediate width
 *  @param out_dim  expert output width
 *  @return the bf16 [T * top_k][mid] fold rows, or nullptr when ws is NULL or too small */
extern "C" const uint16_t *pulsar_rows_moe_routed_mid(void *ws, size_t ws_bytes, int T, int top_k, int mid, int out_dim) {
    routed_ws m;
    return ws && routed_ws_layout(T, top_k, mid, out_dim, ws, ws_bytes, &m) != 0 ? (const uint16_t *)m.mid_x : nullptr;
}

/* The one layout of a rows MoE block's workspace (the router's outputs, the routed region, the shared experts):
 * sizing and carving are the same walk.  Each shared expert has its own rows (a pending fork's buffers); the
 * dense arm's workspace is one, shared. */
/** @param s    the block's shape
 *  @param T    rows
 *  @param base workspace start, or nullptr to size only
 *  @param cap  workspace bytes (checked when base is non-null)
 *  @param o    out: the carved parts, or nullptr
 *  @return bytes used, 0 when they exceed cap */
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
    /* the routed experts' width: the latent's on a latent block (L296 N4), else the hidden's */
    const int rw = s->latent > 0 ? s->latent : s->hidden;
    m.routed_bytes = routed_ws_layout(T, s->top_k, s->expert_mid, rw, nullptr, 0, nullptr);
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
    if (s->latent > 0) {
        const size_t L = (size_t)s->latent;
        m.lat_f = (float *)b.take((size_t)T * L * 4);
        m.lat_x = (uint16_t *)b.take((size_t)T * L * 2);
        m.lat_y = (uint16_t *)b.take((size_t)T * L * 2);
        /* the two projections' dense-arm workspace (hidden -> latent, latent -> hidden), one, shared */
        const size_t w1 = ds4_exl3_dense_workspace_bytes(T, s->hidden, s->latent);
        const size_t w2 = ds4_exl3_dense_workspace_bytes(T, s->latent, s->hidden);
        m.lat_lin_bytes = w1 > w2 ? w1 : w2;
        m.lat_lin = b.take(m.lat_lin_bytes);
    }
    if (o) *o = m;
    return b.failed ? 0 : b.used;
}

/** Validate a block shape (counts positive, n_shared in 0..PULSAR_ROUTER_SHARED_MAX, logit rows cover the experts).
 *  @param s the shape, may be NULL
 *  @return true when the shape is usable */
static bool block_shape_ok(const pulsar_moe_block_shape *s) {
    return s && s->n_expert > 0 && s->top_k > 0 && s->n_logit_rows >= s->n_expert && s->hidden > 0 &&
           s->expert_mid > 0 && s->n_shared >= 0 && s->n_shared <= PULSAR_ROUTER_SHARED_MAX &&
           (s->n_shared == 0 || s->shared_mid > 0) && s->latent >= 0;
}

/** Check the last launch; prints the failure with `what` and returns false on error.
 *  @param what step name for the message
 *  @return true when no launch error is pending */
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
    return l->f16 ? pulsar_f16_rows_workspace_bytes(rows, l->in) : ds4_exl3_dense_workspace_bytes(rows, l->in, l->out);
}

extern "C" int pulsar_rows_linear_launch(const pulsar_rows_linear *l, const uint16_t *x_bf16, int rows, float *y,
                                         void *ws, size_t ws_bytes, cudaStream_t stream) {
    /* L251 / ac69748f: the reader takes the block input's bf16 row.  There is no E4M3 activation slot in
     * this family, so a missing activation is an error -- never a reason to reach for another format. */
    if (!l || !l->w || !x_bf16 || rows <= 0) {
        fprintf(stderr, "pulsar: qwen linear: no weight or no bf16 activation of width %d -- refusing\n", l ? l->in : -1);
        return -1;
    }
    if (l->bf16 || l->nvfp4 || l->f16) {    /* L295 F7 / L298 / L321: a BF16 / nvfp4 / f16 weight */
        if (l->k2 || l->sf || (int)l->bf16 + (int)l->nvfp4 + (int)l->f16 != 1) {
            fprintf(stderr, "pulsar: rows linear: a %d->%d BF16 / nvfp4 / f16 weight with an EXL3 rate, an E8M0 plane or "
                            "two flags -- refusing\n", l->in, l->out);
            return -1;
        }
        if (l->nvfp4) return pulsar_nvfp4_rows_linear_launch(l->w, x_bf16, rows, l->in, l->out, y, nullptr, stream);
        if (l->f16) return pulsar_f16_rows_linear_launch(l->w, x_bf16, rows, l->in, l->out, y, l->prompt, ws, ws_bytes,
                                                         stream);
        return pulsar_bf16_rows_linear_launch(l->w, x_bf16, rows, l->in, l->out, y, l->prompt, stream);
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

extern "C" int pulsar_rows_linear_f16_launch(const pulsar_rows_linear *l, const uint16_t *x_f16, int rows, float *y,
                                             cudaStream_t stream) {
    if (!l || !l->w || !l->f16 || l->bf16 || l->nvfp4 || l->k2 || l->sf) {
        fprintf(stderr, "pulsar: rows linear: f16 operand rows are read by the f16 arm only (a %d->%d weight that is not "
                        "f16) -- refusing\n", l ? l->in : -1, l ? l->out : -1);
        return -1;
    }
    return pulsar_f16_rows_linear_h_launch(l->w, x_f16, rows, l->in, l->out, y, l->prompt, stream);
}

extern "C" size_t pulsar_rows_linear_multi_workspace_bytes(const pulsar_rows_linear *const *ls, int n, int rows) {
    if (!ls || n < 1 || n > DS4_EXL3_DENSE_MAX_PROJ || rows <= 0) return 0;
    ds4_exl3_dense_proj p[DS4_EXL3_DENSE_MAX_PROJ] = {};
    int ne = 0;
    size_t f16 = 0;   /* L321: an f16 Linear runs alone after the EXL3 launch, on the same workspace */
    for (int j = 0; j < n; ++j) {
        if (!ls[j] || ls[j]->in != ls[0]->in) return 0;
        if (ls[j]->f16) {
            f16 = pulsar_f16_rows_workspace_bytes(rows, ls[0]->in);
            continue;
        }
        /* an EXL3 Linear (any logical width, L295 F7), or a width an EXL3 one could have (mxfp8_lt needs none) */
        if (ls[j]->k2 || ls[j]->out % 128 == 0) p[ne++].N = ls[j]->out;
    }
    const size_t exl3 = ne ? ds4_exl3_dense_multi_workspace_bytes(p, ne, rows, ls[0]->in) : 0;
    return exl3 > f16 ? exl3 : f16;
}

/* The multi launch, f32 outputs y (yb NULL) or bf16 rows yb with y[j] the f32 staging of the arms that have no bf16
 * epilogue (L296 dec2): the EXL3 decode GEMV emits yb itself; every other arm writes y[j] and the front door emits. */
/** @param ls       [n] the Linears
 *  @param n        Linears, 1 .. DS4_EXL3_DENSE_MAX_PROJ
 *  @param x_bf16   device bf16 [rows][in] shared input
 *  @param rows     activation rows
 *  @param y        [n] device f32 outputs, or (with yb) the f32 staging
 *  @param yb       [n] device bf16 outputs, or NULL
 *  @param ws       device workspace
 *  @param ws_bytes bytes at ws
 *  @param stream   stream to launch on
 *  @return 0 on success, negative on a refusal (said) or a launch failure */
static int rows_multi(const pulsar_rows_linear *const *ls, int n, const uint16_t *x_bf16, int rows, float *const *y,
                      uint16_t *const *yb, void *ws, size_t ws_bytes, cudaStream_t stream) {
    if (!ls || n < 1 || n > DS4_EXL3_DENSE_MAX_PROJ || !y || !x_bf16 || rows <= 0) {
        fprintf(stderr, "pulsar: qwen linears: %d Linears (1..%d), no input or no rows -- refusing\n", n,
                DS4_EXL3_DENSE_MAX_PROJ);
        return -1;
    }
    ds4_exl3_dense_proj p[DS4_EXL3_DENSE_MAX_PROJ];
    int ne = 0;
    for (int j = 0; j < n; ++j) {
        const pulsar_rows_linear *l = ls[j];
        if (!l || !l->w || !y[j] || (yb && !yb[j]) || l->in != ls[0]->in || l->prompt != ls[0]->prompt) {
            fprintf(stderr, "pulsar: qwen linears: Linear %d has no weight / output, or not the first one's input width "
                            "and row kind -- refusing\n", j);
            return -1;
        }
        const bool own_bf16 = yb && !l->prompt;   /* the EXL3 decode epilogue emits the bf16 rows */
        if (l->k2) p[ne++] = ds4_exl3_dense_proj{l->w, l->k2, l->out, own_bf16 ? nullptr : y[j], own_bf16 ? yb[j] : nullptr};
    }
    if (ne && ds4_exl3_dense_multi_launch(p, ne, x_bf16, rows, ls[0]->in, ws, ws_bytes, stream, ls[0]->prompt) != 0)
        return -1;
    /* L298: the nvfp4 Linears, those of one output width in one launch (a gate and an up), each bit for bit its
     * launch alone; with bf16 outputs the kernel's epilogue emits them (every row kind) */
    bool done[DS4_EXL3_DENSE_MAX_PROJ] = {};
    for (int j = 0; j < n; ++j) {
        if (!ls[j]->nvfp4 || done[j]) continue;
        const void *w[PULSAR_NVFP4_MULTI_MAX];
        float *yf[PULSAR_NVFP4_MULTI_MAX];
        uint16_t *yh[PULSAR_NVFP4_MULTI_MAX];
        int m = 0;
        for (int i = j; i < n && m < PULSAR_NVFP4_MULTI_MAX; ++i)
            if (ls[i]->nvfp4 && !done[i] && ls[i]->out == ls[j]->out) {
                w[m] = ls[i]->w;
                yf[m] = yb ? nullptr : y[i];
                yh[m] = yb ? yb[i] : nullptr;
                done[i] = true;
                ++m;
            }
        if (pulsar_nvfp4_rows_multi_launch(w, m, x_bf16, rows, ls[j]->in, ls[j]->out, yf, yh, stream) != 0) return -1;
    }
    for (int j = 0; j < n; ++j) {   /* the mxfp8_lt / BF16 tiers, one by one (their own arms) */
        const pulsar_rows_linear *l = ls[j];
        if (l->nvfp4) continue;
        if (!l->k2 && pulsar_rows_linear_launch(l, x_bf16, rows, y[j], ws, ws_bytes, stream) != 0) return -1;
        if (yb && (!l->k2 || l->prompt) &&
            pulsar_rows_emit_bf16_launch(y[j], (uint64_t)l->out, (uint32_t)rows, (uint32_t)l->out, NULL,
                                         PULSAR_ROWS_SCALE_NONE, yb[j], (uint64_t)l->out, stream) != 0)
            return -1;
    }
    return 0;
}

extern "C" int pulsar_rows_linear_multi_launch(const pulsar_rows_linear *const *ls, int n, const uint16_t *x_bf16,
                                               int rows, float *const *y, void *ws, size_t ws_bytes, cudaStream_t stream) {
    return rows_multi(ls, n, x_bf16, rows, y, nullptr, ws, ws_bytes, stream);
}

extern "C" int pulsar_rows_linear_multi_bf16_launch(const pulsar_rows_linear *const *ls, int n, const uint16_t *x_bf16,
                                                    int rows, float *const *stage, uint16_t *const *yb, void *ws,
                                                    size_t ws_bytes, cudaStream_t stream) {
    if (!yb) {
        fprintf(stderr, "pulsar: rows linears (bf16 outputs): no outputs -- refusing\n");
        return -1;
    }
    return rows_multi(ls, n, x_bf16, rows, stage, yb, ws, ws_bytes, stream);
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
/// @param q  the shared expert
/// @param H  the block's hidden width
/// @param sh out: the EXL3 front door's descriptor
/// @return true when q can ride (SwiGLU, three EXL3 slices H -> mid -> H); false said
static bool rows_moe_shared(const pulsar_rows_moe_shared *q, int H, exl3_moe_shared *sh) {
    const int smid = q->gate.out;
    if (q->act != PULSAR_FFN_ACT_SWIGLU) {
        fprintf(stderr, "pulsar: routed MoE (bf16 rows): a riding shared expert is SwiGLU (the EXL3 front door's fold); "
                        "activation %d runs pulsar_rows_moe_shared_launch -- refusing\n", (int)q->act);
        return false;
    }
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
    /* the routed experts are ONE of: EXL3 fused gate_up + down, EXL3 gate + up pair + down, Tessera (L255), BF16
     * gate / up / down stacks (L295 F7) */
    const bool tessera = w->tessera_gate != nullptr;
    const bool bf16 = w->bf16_gate != nullptr || w->bf16_up != nullptr;
    const bool split = w->gate_table != nullptr;
    const bool exl3 = split || w->gate_up_table != nullptr || w->down_table != nullptr;
    /* L296: EXL3 up-only (an up table alone: ungated experts) */
    const bool exl3_up = exl3 && !split && w->gate_up_table == nullptr && w->up_table != nullptr;
    if (tessera && bf16) {
        fprintf(stderr, "pulsar: routed MoE (bf16 rows): both Tessera and BF16 stacks -- refusing\n");
        return -1;
    }
    /* L295 F7 / L296: a non-gated activation (relu^2) runs on up-only stacks alone -- BF16 (no gate stack) or EXL3
     * (an up table alone); an EXL3 up table alone is relu^2 only */
    if ((w->act != PULSAR_FFN_ACT_SWIGLU &&
         (w->act != PULSAR_FFN_ACT_RELU2 || (bf16 ? w->bf16_gate != nullptr : !exl3_up))) ||
        (exl3_up && w->act != PULSAR_FFN_ACT_RELU2)) {
        fprintf(stderr, "pulsar: routed MoE (bf16 rows): activation %d runs on up-only stacks (BF16 without a gate, or an "
                        "EXL3 up table alone) and only there; the gated arms are SwiGLU -- refusing\n", (int)w->act);
        return -1;
    }
    if (tessera || bf16 ? (tessera ? (!w->tessera_up || !w->tessera_down)
                                   : (!w->bf16_up || !w->bf16_down ||
                                      (w->act == PULSAR_FFN_ACT_SWIGLU) != (w->bf16_gate != nullptr))) || exl3
                        : (!w->down_table || (!exl3_up && (split != (w->up_table != nullptr) ||
                                                           split == (w->gate_up_table != nullptr))))) {
        fprintf(stderr, "pulsar: routed MoE (bf16 rows): the experts must be ONE of an EXL3 fused gate_up table, an "
                        "EXL3 gate + up pair, an EXL3 up table alone, a Tessera gate / up / down triple or a BF16 one -- "
                        "refusing\n");
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
    (void)nf_code; (void)stream; (void)pairs; (void)nE; (void)tessera; (void)bf16;
    fprintf(stderr, "pulsar: routed MoE (bf16 rows): built without the MMQ drivers the routed arms ride -- refusing\n");
    return -1;
#else
    if (w->n_shared < 0 || w->n_shared > PULSAR_ROUTER_SHARED_MAX || (w->n_shared > 0) != (w->shared != nullptr) ||
        ((tessera || bf16) && w->n_shared)) {
        fprintf(stderr, "pulsar: routed MoE (bf16 rows): %d riding shared experts (0..%d, with their descriptors; they "
                        "ride the EXL3 front door only, not the Tessera or BF16 arms) -- refusing\n", w->n_shared,
                PULSAR_ROUTER_SHARED_MAX);
        return -1;
    }
    if (bf16) {
        /* L295 F7: the BF16 routed arm over the router's selection, its buffers the EXL3 arm's (exactly one routed
         * arm runs): gate z = gu_z, up z = its second half, the fold's bf16 rows = mid_x, the down's = down_z */
        if (nE != E) {
            qwen_ep_localize_kernel<<<(unsigned)((pairs + 255) / 256), 256, 0, stream>>>(sel, (int)pairs, w->ex_lo, nE);
            if (!launch_ok("expert localize")) return -3;
        }
        const pulsar_bf16_moe bw{w->bf16_gate, w->bf16_up, w->bf16_down, nE, H, mid, H, w->act};
        return pulsar_bf16_moe_routed_launch(bw, sel, wts, T, top_k, 0.0f, x_bf16, m.gu_z, m.gu_z + (size_t)pairs * mid,
                                             reinterpret_cast<uint16_t *>(m.mid_x), m.down_z, out, nf_flag, nf_code,
                                             stream) == 0 ? 0 : -1;
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
    static int announced[3] = {0, 0, 0};
    const int form = exl3_up ? 2 : split ? 1 : 0;
    if (!announced[form]) {
        announced[form] = 1;
        fprintf(stderr, "pulsar: routed MoE (bf16 rows) = EXL3 %s K=%g, down K=%g, top-%d of %d (experts [%d, +%d)) "
                        "through the EXL3 routed front door (exl3_moe.cuh)\n",
                exl3_up ? "up-only relu^2 up" : split ? "gate + up pair" : "fused gate_up", w->k2_gate_up / 2.0,
                w->k2_down / 2.0, top_k, E, w->ex_lo, w->n_local);
    }
    if (nE != E) {
        qwen_ep_localize_kernel<<<(unsigned)((pairs + 255) / 256), 256, 0, stream>>>(sel, (int)pairs, w->ex_lo, nE);
        if (!launch_ok("expert localize")) return -3;
    }
    /* L251 / ac69748f: the block input's bf16 row and the fold's bf16 output -- no E4M3 slot in this lane */
    const exl3_moe_stacks st{w->gate_up_table, w->gate_table, w->up_table, w->down_table, w->k2_gate_up, w->k2_down,
                             nE, H, mid, H, (int)w->act};
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

/* A non-riding shared expert's descriptor, checked (the launch, the fork and the join take the same one). */
/// @param q    the shared expert
/// @param what the caller's name for the message
/// @return true when the descriptor and its buffers are consistent; false said
static bool shared_ok(const pulsar_rows_moe_shared *q, const char *what) {
    const bool gated = q && q->act == PULSAR_FFN_ACT_SWIGLU;
    const int smid = q ? q->up.out : 0, H = q ? q->up.in : 0;
    if (!q || (q->act != PULSAR_FFN_ACT_SWIGLU && q->act != PULSAR_FFN_ACT_RELU2) ||
        (gated && (!q->yg || q->gate.in != H || q->gate.out != smid)) || !q->yu || !q->ys || !q->h ||
        q->down.in != smid || q->down.out != H) {
        fprintf(stderr, "pulsar: MoE shared expert %s: activation %d, %s%d->%d / %d->%d (in -> mid -> in), its buffers -- "
                        "refusing\n", what, q ? (int)q->act : -1, gated ? "a gate of the up's shape, " : "", H, smid,
                q ? q->down.in : -1, q ? q->down.out : -1);
        return false;
    }
    return true;
}

/* [gate +] up on the same row, the activation's producer, down: ys = down(act(gate x, up x)) */
/// @param q      the shared expert
/// @param x_bf16 device bf16 [T][in]
/// @param T      rows
/// @param stream stream to launch on
/// @return 0 on success, negative on a refusal or a launch failure
static int shared_ys(const pulsar_rows_moe_shared *q, const uint16_t *x_bf16, int T, cudaStream_t stream) {
    int rc = q->act == PULSAR_FFN_ACT_SWIGLU ? pulsar_rows_linear_launch(&q->gate, x_bf16, T, q->yg, q->ws, q->ws_bytes,
                                                                         stream) : 0;
    if (!rc) rc = pulsar_rows_linear_launch(&q->up, x_bf16, T, q->yu, q->ws, q->ws_bytes, stream);
    if (rc) return rc;
    rows_act_emit_kernel<<<T, 256, 0, stream>>>(q->yg, q->yu, q->up.out, (int)q->act, (__nv_bfloat16 *)q->h);
    if (!launch_ok("shared activation")) return -3;
    return pulsar_rows_linear_launch(&q->down, q->h, T, q->ys, q->ws, q->ws_bytes, stream);
}

/* out += scale * ys */
/// @param q      the shared expert (its ys and scale)
/// @param T      rows
/// @param out    device f32 [T][H] accumulated into
/// @param stream stream to launch on
/// @return 0, or -3 on a launch failure
static int shared_add(const pulsar_rows_moe_shared *q, int T, float *out, cudaStream_t stream) {
    const int H = q->down.out;
    const size_t n = (size_t)T * H;
    rows_shared_add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(out, q->ys, q->scale, T, H);
    return launch_ok("shared add") ? 0 : -3;
}

extern "C" int pulsar_rows_moe_shared_launch(const pulsar_rows_moe_shared *q, const uint16_t *x_bf16, int T,
                                             float *out, cudaStream_t stream) {
    if (!shared_ok(q, "launch") || !x_bf16 || !out || T <= 0) return -1;
    const int rc = shared_ys(q, x_bf16, T, stream);
    return rc ? rc : shared_add(q, T, out, stream);
}

extern "C" int pulsar_rows_moe_shared_ys_launch(const pulsar_rows_moe_shared *q, const uint16_t *x_bf16, int T,
                                                cudaStream_t stream) {
    if (!shared_ok(q, "ys") || !x_bf16 || T <= 0) return -1;
    return shared_ys(q, x_bf16, T, stream);
}

extern "C" int pulsar_rows_moe_shared_add_launch(const pulsar_rows_moe_shared *q, int T, float *out, cudaStream_t stream) {
    if (!shared_ok(q, "add") || !out || T <= 0) return -1;
    return shared_add(q, T, out, stream);
}

namespace {

/* L296 dec2: the routed chain's stream per (thread, device, caller stream) -- the device's GREATEST priority, so its
 * latency-bound launches take the SMs ahead of the bandwidth-bound shared expert left on the caller (in a captured
 * step the nodes keep it: kernel nodes copy their stream's priority, and the segment cache instantiates with
 * cudaGraphInstantiateFlagUseNodePriority) -- with the fork and join events.  One fork pending at a time. */
/// L296 dec2: one (thread, device, caller stream)'s priority stream for the routed chain, with its fork / join events.
struct routed_lane {
    /// CUDA device the lane belongs to.
    int dev;
    /** @var caller
     *  The caller stream the lane is keyed on.
     *  @var s
     *  The lane's own non-blocking stream at the device's greatest priority. */
    cudaStream_t caller, s;
    /** @var fork
     *  Recorded on caller at the fork; s waits on it.
     *  @var done
     *  Recorded on s at the join; caller waits on it. */
    cudaEvent_t fork, done;
    /// A fork is outstanding (one at a time).
    bool pending;
};

/** The calling thread's lane for (current device, caller), created on first use (at most 8 per thread).
 *  @param caller the caller stream
 *  @return the lane, or nullptr when the device query or stream / event creation fails or the 8 slots are used */
routed_lane *routed_lane_for(cudaStream_t caller) {
    thread_local routed_lane t_lanes[8];
    thread_local int t_n = 0;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return nullptr;
    for (int i = 0; i < t_n; ++i)
        if (t_lanes[i].dev == dev && t_lanes[i].caller == caller) return &t_lanes[i];
    if (t_n == 8) return nullptr;
    routed_lane l{};
    l.dev = dev;
    l.caller = caller;
    int least = 0, greatest = 0;
    if (cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess ||
        cudaStreamCreateWithPriority(&l.s, cudaStreamNonBlocking, greatest) != cudaSuccess ||
        cudaEventCreateWithFlags(&l.fork, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&l.done, cudaEventDisableTiming) != cudaSuccess) {
        /* L335: a failure part way releases what was already created (it leaked the stream and the fork event) */
        if (l.done) (void)cudaEventDestroy(l.done);
        if (l.fork) (void)cudaEventDestroy(l.fork);
        if (l.s) (void)cudaStreamDestroy(l.s);
        (void)cudaGetLastError();
        fprintf(stderr, "pulsar: MoE routed lane: its stream / events could not be created -- refusing\n");
        return nullptr;
    }
    t_lanes[t_n] = l;
    static bool announced = false;
    if (!__atomic_exchange_n(&announced, true, __ATOMIC_RELAXED))
        fprintf(stderr, "pulsar: MoE decode = the router + routed experts on a priority-%d stream, the shared expert "
                        "alongside on the caller's (priority %d), joined before the shared add\n", greatest, least);
    return &t_lanes[t_n++];
}

} // namespace

extern "C" pulsar_rows_stream pulsar_rows_moe_routed_fork(cudaStream_t caller) {
    routed_lane *lane = routed_lane_for(caller);
    if (!lane || lane->pending) {
        fprintf(stderr, "pulsar: MoE routed fork: no priority stream, or a fork already pending on it -- refusing\n");
        return nullptr;
    }
    if (cudaEventRecord(lane->fork, caller) != cudaSuccess || cudaStreamWaitEvent(lane->s, lane->fork, 0) != cudaSuccess) {
        fprintf(stderr, "pulsar: MoE routed fork: the fork event failed -- refusing\n");
        return nullptr;
    }
    lane->pending = true;
    return lane->s;
}

extern "C" int pulsar_rows_moe_routed_join(cudaStream_t caller) {
    routed_lane *lane = routed_lane_for(caller);
    if (!lane || !lane->pending) {
        fprintf(stderr, "pulsar: MoE routed join: no fork pending on this stream -- refusing\n");
        return -1;
    }
    lane->pending = false;
    return cudaEventRecord(lane->done, lane->s) == cudaSuccess && cudaStreamWaitEvent(caller, lane->done, 0) == cudaSuccess
               ? 0 : -3;
}

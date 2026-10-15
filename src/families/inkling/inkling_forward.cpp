/* inkling_forward.cpp -- the Inkling forward on the core row-step driver (L294 / L295; step_driver.h).
 *
 * The family declares; the core runs.  What is Inkling's here is only the composition: which shared blocks a layer
 * runs, in which order, on which state, with which declaration.  Every kernel is a shared block's:
 *
 *   embed      E5   pulsar_head_embed_launch: the gather + the embed RMSNorm -> the residual h (bf16)
 *   per layer  (pulsar_layer_desc: a window or a full attention, a dense or a MoE FFN)
 *       xn = rmsnorm(h) * attn_norm                                             row ops (pulsar_cuda_rowops.h)
 *       q | k | v | r = xn W  (one multi launch, f32) -> bf16 at the producer   rows front door (linear.cpp)
 *       k, v = SConv_k(k), SConv_v(v)  (E3, one launch, the layer's record)
 *       a = GQA(q, k, v; per-head q/k norm, bias r.P, tau on a global layer)    F1 (ring / pool: F3's state)
 *       h += SConv_attn(bf16(a W_o));  xn = rmsnorm(h) * mlp_norm               E3, row ops
 *       f = dense SwiGLU MLP x global_scale  |  F2 router (sigmoid + bias top-6, log-sigmoid over 6 + 2 sinks,
 *           x route_scale x global_scale) -> routed EXL3 experts + 2 sink-gated shared experts
 *       h += SConv_ffn(bf16(f))                                                 E3, row ops
 *   head       rmsnorm(h) * output_norm -> the head linear (rows front door) -> E5 finish (x 1/16, pad -> -inf)
 *
 * vLLM's InklingDecoderLayer (models/inkling/nvidia/model.py) is the order: the K/V convs after the projection and
 * before the k-norm; the attention / FFN deltas through their convs before the residual add; the residual stream
 * bf16, each add one rounding, each norm one rounding (vLLM's add_rmsnorm).  Where pulsar rounds differently, it is
 * said at the site and graded (rule 7): the f32 linears' one bf16 rounding (= a bf16 GEMM's), the MoE's gammas at the
 * expert output and its f32 sum (F2), the dense MLP's global scale in f32 before the one rounding.
 *
 * ROW KINDS (L284 #2): every arm-sensitive launch (the EXL3 linears, the routed MoE) runs rows [0, n_dec) on the
 * decode arms and [n_dec, n_rows) on the prompt arms; the router, the row ops, SConv and F1 are one arithmetic at any
 * row count.  So a prompt cut anywhere is the cold prefill's bytes (the plan is split-invariant), and a decode row is
 * the same bytes alone or batched.
 *
 * STATE (state_kinds.h; inkling.h PULSAR_INKLING_STATE_*): the SConv record (FIXED, a verify's SNAPSHOT per row), a
 * window layer's F1 ring (RING), a global layer's K|V pool (POOL).  The core allocates, prices, pages, walks and
 * commits it; nothing here copies state. */
#include "inkling.h"
#include "cuda/pulsar_cuda_attn_gqa.h"
#include "cuda/pulsar_cuda_head.h"
#include "cuda/pulsar_cuda_qwen.h"     /* the rows front door's launcher + the rows MoE block (family-neutral) */
#include "cuda/pulsar_cuda_router.h"
#include "cuda/pulsar_cuda_rowops.h"
#include "cuda/pulsar_cuda_sconv.h"

namespace {

const char *const ARCH = PULSAR_INKLING_ARCH;

/* The resume grid (kv_state.h): the global pools are one token a row and every prompt chunk takes the prompt arms
 * at any row count, so a cut anywhere is the cold bytes; 128 keeps the slots on the server's cadence. */
constexpr uint32_t kResumeGrid = 128u;
/* A checkpoint walks the rings whole (35 x 543 slots (W - 1 + the 32 verify rows) x 4 KiB = 76 MB on Inkling-Small) and the SConv records */
constexpr uint32_t kCkptSlots = 4u;

/* The op arenas (pulsar_driver_ops::scratch_bytes) */
enum ink_arena { ARENA_ROWS = 0, ARENA_ATTN, ARENA_FFN, ARENA_HEAD, ARENA_COUNT };

uint64_t a256(uint64_t x) { return (x + 255u) & ~(uint64_t)255u; }
void *dptr(pulsar_gpu_tensor *t) { return pulsar_gpu_tensor_device_ptr(t); }

bool fail(const char *what) {
    fprintf(stderr, "pulsar: %s: %s -- refusing the step\n", ARCH, what);
    return false;
}
bool failf(const char *what, uint32_t il) {
    fprintf(stderr, "pulsar: %s: %s (layer %u) -- refusing the step\n", ARCH, what, il);
    return false;
}

/* ---- the geometry every op reads ----------------------------------------------------------------------------------
 * The scratch is laid out from the shape before any session exists (pulsar_driver_ops::scratch_bytes has no engine),
 * and the geometry IS the profile's: inkling_shape.cpp pins every geometry field to PULSAR_INKLING_SHAPE_SMALL. */
const pulsar_inkling_shape &G() { return PULSAR_INKLING_SHAPE_SMALL; }

/* a layer's attention widths: q heads x head_dim (== the o_proj input), kv heads x head_dim, r */
struct attn_dims { uint32_t nq, nkv, hd, extent, q, kv, r; };
attn_dims attn_of(const pulsar_inkling_shape &s, bool window) {
    attn_dims a;
    a.nq = window ? s.swa_n_head : s.n_head;
    a.nkv = window ? s.swa_n_head_kv : s.n_head_kv;
    a.hd = window ? s.swa_head_dim : s.head_dim;
    a.extent = window ? s.window : s.rel_extent;
    a.q = a.nq * a.hd;
    a.kv = a.nkv * a.hd;
    a.r = a.nq * s.d_rel;
    return a;
}

/* F1's declaration of a layer (weights NULL when only the geometry is wanted: the workspace and row sizes) */
pulsar_gqa_layer gqa_decl(const pulsar_inkling_shape &s, bool window, const uint16_t *qn, const uint16_t *kn,
                          const uint16_t *P) {
    const attn_dims a = attn_of(s, window);
    pulsar_gqa_layer L{};
    L.n_q = a.nq;
    L.n_kv = a.nkv;
    L.head_dim = a.hd;
    L.kv_fmt = PULSAR_GQA_KV_BF16;
    L.span = window ? PULSAR_GQA_WINDOW : PULSAR_GQA_FULL;
    L.window = window ? s.window : 0u;
    L.scale = 1.0f / (float)a.hd;   /* q and k are unit-RMS per head: Inkling scales by 1 / head_dim */
    L.qk_norm = 1;
    L.norm_eps = s.rms_eps;
    L.q_norm_w = qn;
    L.k_norm_w = kn;
    L.rel_extent = a.extent;
    L.rel_dim = s.d_rel;
    L.rel_proj = P;
    /* the log-length temperature on the GLOBAL layers only (vLLM passes log_scaling to non-local layers) */
    L.temp_floor = window ? 0.0f : (float)s.log_scale_floor;
    L.temp_alpha = window ? 0.0f : s.log_scale_alpha;
    L.out_gate = 0;
    return L;
}

/* F2: Inkling's router as the shared router's declaration -- sigmoid scores, the selection bias picks the top-k (the
 * weights stay unbiased), then a softmax of log-sigmoid over the k picks' RAW logits and the S sink logits (the
 * router weight's last S rows), x route_scale x the layer's global scale; the S sink weights gate the S shared
 * experts.  f32 logits throughout (vLLM's gate GEMM has an f32 out), f32 weights. */
pulsar_router_cfg router_cfg(const pulsar_inkling_shape &s) {
    pulsar_router_cfg c{};
    c.n_expert = (int)s.n_expert;
    c.top_k = (int)s.n_expert_used;
    c.n_shared = (int)s.n_shared;
    c.shared_gate = PULSAR_ROUTER_SHARED_SINK;
    c.score = PULSAR_ROUTER_SCORE_SIGMOID;
    c.renorm = PULSAR_ROUTER_RENORM_LOGSIGMOID_SINKS;
    c.logits_bf16 = false;
    c.weights_bf16 = false;
    c.sum_floor = 0.0f;
    c.route_scale = s.route_scale;
    return c;
}

pulsar_moe_block_shape moe_shape(const pulsar_inkling_shape &s) {
    const pulsar_router_cfg rc = router_cfg(s);
    pulsar_moe_block_shape b{};
    b.n_expert = rc.n_expert;
    b.top_k = rc.top_k;
    b.n_logit_rows = pulsar_router_logit_rows(&rc);
    b.hidden = (int)s.n_embd;
    b.expert_mid = (int)s.n_ff_exp;
    b.n_shared = rc.n_shared;
    b.shared_mid = (int)s.n_ff_exp;
    return b;
}

/* The dense-arm workspace of n linears of one input (a multi launch) or of one linear in -> out */
uint64_t multi_ws(int in, int n, const int *out, uint32_t rows) {
    pulsar_rows_linear l[4] = {};
    const pulsar_rows_linear *lp[4];
    for (int j = 0; j < n; j++) {
        l[j].in = in;
        l[j].out = out[j];
        lp[j] = &l[j];
    }
    return a256(pulsar_rows_linear_multi_workspace_bytes(lp, n, (int)rows));
}
uint64_t lin_ws(int in, int out, uint32_t rows) {
    pulsar_rows_linear l{};
    l.in = in;
    l.out = out;
    return a256(pulsar_rows_linear_workspace_bytes(&l, (int)rows));
}

/* ---- scratch layouts: one function each, for sizing AND carving -------------------------------------------------- */

/* ROWS: the step's tables -- the token ids, the SConv calls' sequence tables (the verify rows' captured call, then
 * the rest), the MoE non-finite flag */
struct rows_scratch { uint64_t ids, first_a, slot_a, first_b, slot_b, nf, total; };
rows_scratch rows_layout(uint32_t R) {
    rows_scratch r{};
    uint64_t o = 0;
    r.ids = o;     o += a256((uint64_t)R * 4u);
    r.first_a = o; o += a256((uint64_t)(R + 1u) * 4u);
    r.slot_a = o;  o += a256((uint64_t)R * 4u);
    r.first_b = o; o += a256((uint64_t)(R + 1u) * 4u);
    r.slot_b = o;  o += a256((uint64_t)R * 4u);
    r.nf = o;      o += 256u;
    r.total = o;
    return r;
}

/* ATTN: the projections' f32 outputs and their bf16 rows, the convolved K / V, the attention's output, the o_proj's
 * f32 and bf16 rows, the conv's output, the dense-arm and F1 workspaces */
struct attn_scratch { uint64_t qf, kf, vf, rf, qb, kb, vb, rb, kc, vc, ab, of, db, yb, lin, lin_bytes, gqa, gqa_bytes, total; };
attn_scratch attn_layout(uint32_t R, uint32_t ctx) {
    const pulsar_inkling_shape &s = G();
    const attn_dims a = attn_of(s, false);   /* the profile pins window and global widths equal (inkling_shape.cpp) */
    const uint64_t E = s.n_embd;
    attn_scratch t{};
    uint64_t o = 0;
    t.qf = o; o += a256((uint64_t)R * a.q * 4u);
    t.kf = o; o += a256((uint64_t)R * a.kv * 4u);
    t.vf = o; o += a256((uint64_t)R * a.kv * 4u);
    t.rf = o; o += a256((uint64_t)R * a.r * 4u);
    t.qb = o; o += a256((uint64_t)R * a.q * 2u);
    t.kb = o; o += a256((uint64_t)R * a.kv * 2u);
    t.vb = o; o += a256((uint64_t)R * a.kv * 2u);
    t.rb = o; o += a256((uint64_t)R * a.r * 2u);
    t.kc = o; o += a256((uint64_t)R * a.kv * 2u);
    t.vc = o; o += a256((uint64_t)R * a.kv * 2u);
    t.ab = o; o += a256((uint64_t)R * a.q * 2u);
    t.of = o; o += a256((uint64_t)R * E * 4u);
    t.db = o; o += a256((uint64_t)R * E * 2u);
    t.yb = o; o += a256((uint64_t)R * E * 2u);
    const int outs[4] = {(int)a.q, (int)a.kv, (int)a.kv, (int)a.r};
    const uint64_t m = multi_ws((int)E, 4, outs, R), w = lin_ws((int)a.q, (int)E, R);
    t.lin = o; t.lin_bytes = m > w ? m : w; o += t.lin_bytes;
    const pulsar_gqa_layer gw = gqa_decl(s, true, NULL, NULL, NULL), gg = gqa_decl(s, false, NULL, NULL, NULL);
    const uint64_t bw = pulsar_gqa_workspace_bytes(&gw, R, ctx ? ctx : 1u);
    const uint64_t bg = pulsar_gqa_workspace_bytes(&gg, R, ctx ? ctx : 1u);
    t.gqa = o; t.gqa_bytes = a256(bw > bg ? bw : bg); o += t.gqa_bytes;
    t.total = o;
    return t;
}

/* FFN: the block's f32 output, its bf16 row, the conv's output, then ONE region a layer uses as either the dense
 * MLP's buffers or the MoE block's workspace (a layer runs one of them) */
struct ffn_scratch { uint64_t fo, db, yb, u, yg, yu, ys, hh, lin, lin_bytes, moe_bytes, total; };
ffn_scratch ffn_layout(uint32_t R) {
    const pulsar_inkling_shape &s = G();
    const uint64_t E = s.n_embd, F = s.n_ff_dense;
    ffn_scratch f{};
    uint64_t o = 0;
    f.fo = o; o += a256((uint64_t)R * E * 4u);
    f.db = o; o += a256((uint64_t)R * E * 2u);
    f.yb = o; o += a256((uint64_t)R * E * 2u);
    f.u = o;
    /* the dense MLP's view of the region */
    uint64_t d = o;
    f.yg = d; d += a256((uint64_t)R * F * 4u);
    f.yu = d; d += a256((uint64_t)R * F * 4u);
    f.ys = d; d += a256((uint64_t)R * E * 4u);
    f.hh = d; d += a256((uint64_t)R * F * 2u);
    const uint64_t w1 = lin_ws((int)E, (int)F, R), w2 = lin_ws((int)F, (int)E, R);
    f.lin = d; f.lin_bytes = w1 > w2 ? w1 : w2; d += f.lin_bytes;
    /* the MoE block's view: its one carved workspace */
    const pulsar_moe_block_shape bs = moe_shape(s);
    f.moe_bytes = a256(pulsar_moe_block_workspace_bytes(&bs, (int)R));
    const uint64_t m = o + f.moe_bytes;
    f.total = d > m ? d : m;
    return f;
}

/* HEAD: the head rows' final-norm bf16 rows and the head linear's workspace (at most PULSAR_DRIVER_HEADS_MAX rows) */
struct head_scratch { uint64_t xb, lin, lin_bytes, total; };
head_scratch head_layout() {
    const pulsar_inkling_shape &s = G();
    head_scratch h{};
    h.xb = 0;
    h.lin = a256((uint64_t)PULSAR_DRIVER_HEADS_MAX * s.n_embd * 2u);
    h.lin_bytes = lin_ws((int)s.n_embd, (int)s.n_vocab, PULSAR_DRIVER_HEADS_MAX);
    h.total = h.lin + h.lin_bytes;
    return h;
}

uint64_t scratch_bytes(uint32_t arena, uint32_t max_rows, uint32_t ctx) {
    switch (arena) {
    case ARENA_ROWS: return rows_layout(max_rows).total;
    case ARENA_ATTN: return attn_layout(max_rows, ctx).total;
    case ARENA_FFN:  return ffn_layout(max_rows).total;
    case ARENA_HEAD: return head_layout().total;
    }
    return 0;
}

/* ---- the session's state and the step ------------------------------------------------------------------------- */

struct ink_step;
/* The family's session state (the driver's buffers are its base): the residual stream and the normed rows the
 * blocks read, the trunk's step object, and the step's host tables (rebuilt by every embed). */
struct ink_state : pulsar_driver_state {
    pulsar_gpu_tensor *h;     /* [max_rows][n_embd] bf16: the residual stream */
    pulsar_gpu_tensor *xn;    /* [max_rows][n_embd] bf16: the RMSNorm of h a block reads */
    ink_step *trunk;
    /* the step's sequences: one a decode row, one a prefill run (a verify run, then the prompt runs) */
    uint32_t n_seq;           /* sequences of the step */
    uint32_t n_seq_cap;       /* [0, n_seq_cap) are the verify's runs (their SConv state captured), rows [0, rows_cap) */
    uint32_t rows_cap;
    uint32_t *seq_first;      /* host [max_rows + 1] */
    int32_t *seq_bank;        /* host [max_rows] */
    uint32_t *row_seq;        /* host [max_rows]: F1's */
    uint32_t *row_pos;        /* host [max_rows]: F1's */
    int32_t *tab_a, *tab_b;   /* host [2 (max_rows + 1)]: the SConv calls' seq_first then seq_slot, as uploaded */
    uint32_t max_run_a, max_run_b;
    pulsar_gqa_seq *seqs;     /* host [max_rows]: F1's sequences for the layer being run */
};

struct ink_step : pulsar_driver_step {
    const pulsar_inkling_model *w;
    ink_state *st;
};

const ink_step *S(const pulsar_driver_step *st) { return static_cast<const ink_step *>(st); }
ink_state *ST(pulsar_session *s) { return static_cast<ink_state *>(s->driver); }
const pulsar_inkling_model *W(const pulsar_engine *e) { return (const pulsar_inkling_model *)e->family_state; }

void state_free(ink_state *st) {
    if (!st) return;
    pulsar_gpu_tensor_free(st->h);
    pulsar_gpu_tensor_free(st->xn);
    free(st->seq_first);
    free(st->seq_bank);
    free(st->row_seq);
    free(st->row_pos);
    free(st->tab_a);
    free(st->tab_b);
    free(st->seqs);
    free(st->trunk);
    free(st);
}

pulsar_driver_state *create(pulsar_session *s, uint32_t n_banks) {
    const pulsar_engine *e = s->engine;
    const pulsar_inkling_shape &sh = W(e)->shape;
    ink_state *st = (ink_state *)xcalloc(1, sizeof(*st));
    const uint64_t R = s->prefill_cap;
    st->h = pulsar_gpu_tensor_alloc(R * sh.n_embd * 2u);
    st->xn = pulsar_gpu_tensor_alloc(R * sh.n_embd * 2u);
    st->seq_first = (uint32_t *)xmalloc((R + 1u) * sizeof(uint32_t));
    st->seq_bank = (int32_t *)xmalloc(R * sizeof(int32_t));
    st->row_seq = (uint32_t *)xmalloc(R * sizeof(uint32_t));
    st->row_pos = (uint32_t *)xmalloc(R * sizeof(uint32_t));
    st->tab_a = (int32_t *)xmalloc(2u * (R + 1u) * sizeof(int32_t));
    st->tab_b = (int32_t *)xmalloc(2u * (R + 1u) * sizeof(int32_t));
    st->seqs = (pulsar_gqa_seq *)xcalloc(R, sizeof(pulsar_gqa_seq));
    st->trunk = (ink_step *)xcalloc(1, sizeof(ink_step));
    st->trunk->w = W(e);
    st->trunk->st = st;
    if (!st->h || !st->xn) {
        fprintf(stderr, "pulsar: %s: session state allocation failed (%u banks x %d tokens)\n", ARCH, n_banks,
                s->ctx_size);
        state_free(st);
        return NULL;
    }
    return st;
}

void destroy(pulsar_session *s) { state_free(ST(s)); }

/* the declared state's verify capture exists with a drafter lane -- F6's chained MTP; none yet */
bool verifies(const pulsar_session *) { return false; }

pulsar_driver_step *trunk(pulsar_session *s) { return ST(s)->trunk; }

uint64_t artifact(const pulsar_engine *e) { return W(e)->artifact_digest; }

const void *wp(const ink_step *st, const pulsar_tensor *t, const char *what) {
    return pulsar_weight_device_ptr(st->model, t, what);
}

/* ---- row kinds ----------------------------------------------------------------------------------------------------
 * L284 #2: rows [lo, hi) of an arm-sensitive launch, cut at n_dec -- the decode rows first (prompt = false), then the
 * prompt rows; an empty segment is skipped.  `launch(prompt, r0, m)` offsets its row pointers by r0. */
template <class Launch>
bool arm_segments(const ink_step *st, uint32_t lo, uint32_t hi, Launch &&launch) {
    const uint32_t cut = st->n_dec < lo ? lo : st->n_dec > hi ? hi : st->n_dec;
    return (cut == lo || launch(false, lo, cut - lo)) && (cut == hi || launch(true, cut, hi - cut));
}

/* n linears (1..4) of the same bf16 input x [n_rows][in] -> y[j] f32 [n_rows][out[j]], each row-kind segment through
 * its arm (one multi launch: the EXL3 ones share one prep) */
bool step_linears(const ink_step *st, int n, const pulsar_tensor *const *t, int in, const int *out,
                  const char *const *what, const uint16_t *x, float *const *y, void *ws, uint64_t ws_bytes) {
    return arm_segments(st, 0, st->n_rows, [&](bool prompt, uint32_t r0, uint32_t m) {
        pulsar_rows_linear l[4];
        const pulsar_rows_linear *lp[4];
        float *yr[4];
        for (int j = 0; j < n; j++) {
            if (!pulsar_linear_rows_ref(st->model, t[j], in, out[j], prompt, what[j], &l[j])) return false;
            lp[j] = &l[j];
            yr[j] = y[j] + (size_t)r0 * (size_t)out[j];
        }
        return pulsar_rows_linear_multi_launch(lp, n, x + (size_t)r0 * in, (int)m, yr, ws, ws_bytes, 0) == 0;
    });
}

/* ---- the step's tables (embed) ---------------------------------------------------------------------------------- */

/* The step's sequences: a DECODE row is one; a PREFILL run is one (a verify's runs first, rows [0, n_dec)).  SConv
 * runs two calls when the step verifies -- the verify rows with their per-row capture, then the prompt rows without
 * (the capture holds PULSAR_STATE_VERIFY_ROWS rows) -- so each call's table is uploaded on its own. */
bool build_tables(const ink_step *st) {
    ink_state *q = st->st;
    const uint32_t n = st->n_rows;
    uint32_t ns = 0;
    if (st->mode == PULSAR_STEP_DECODE) {
        for (uint32_t r = 0; r < n; r++) {
            q->seq_first[ns] = r;
            q->seq_bank[ns++] = st->bank[r];
        }
    } else {
        for (uint32_t k = 0; k < st->n_runs; k++) {
            q->seq_first[ns] = st->run_first[k];
            q->seq_bank[ns++] = st->bank[st->run_first[k]];
        }
    }
    q->seq_first[ns] = n;
    q->n_seq = ns;
    for (uint32_t k = 0; k < ns; k++)
        for (uint32_t r = q->seq_first[k]; r < q->seq_first[k + 1]; r++) {
            q->row_seq[r] = k;
            q->row_pos[r] = (uint32_t)st->pos[r];
        }
    q->n_seq_cap = st->verify ? pulsar_step_verify_runs(st) : 0u;
    q->rows_cap = q->seq_first[q->n_seq_cap];
    /* call A: the captured sequences [0, n_seq_cap); call B: the rest, rows from rows_cap */
    const uint32_t na = q->n_seq_cap, nb = ns - na;
    q->max_run_a = q->max_run_b = 0;
    for (uint32_t k = 0; k <= na; k++) q->tab_a[k] = (int32_t)q->seq_first[k];
    for (uint32_t k = 0; k < na; k++) {
        q->tab_a[na + 1 + k] = q->seq_bank[k];
        const uint32_t m = q->seq_first[k + 1] - q->seq_first[k];
        if (m > q->max_run_a) q->max_run_a = m;
    }
    for (uint32_t k = 0; k <= nb; k++) q->tab_b[k] = (int32_t)(q->seq_first[na + k] - q->rows_cap);
    for (uint32_t k = 0; k < nb; k++) {
        q->tab_b[nb + 1 + k] = q->seq_bank[na + k];
        const uint32_t m = q->seq_first[na + k + 1] - q->seq_first[na + k];
        if (m > q->max_run_b) q->max_run_b = m;
    }
    pulsar_gpu_tensor *rt = q->scratch[ARENA_ROWS];
    const rows_scratch L = rows_layout(q->max_rows);
    static const uint32_t zero = 0;   /* L289: an enqueued write reads it later */
    bool ok = rt != NULL;
    ok = ok && pulsar_gpu_tensor_write_async(rt, L.ids, st->tokens, (uint64_t)n * 4u) != 0;
    ok = ok && pulsar_gpu_tensor_write_async(rt, L.nf, &zero, sizeof(zero)) != 0;
    if (ok && na) {
        ok = pulsar_gpu_tensor_write_async(rt, L.first_a, q->tab_a, (uint64_t)(na + 1u) * 4u) != 0 &&
             pulsar_gpu_tensor_write_async(rt, L.slot_a, q->tab_a + na + 1, (uint64_t)na * 4u) != 0;
    }
    if (ok && nb) {
        ok = pulsar_gpu_tensor_write_async(rt, L.first_b, q->tab_b, (uint64_t)(nb + 1u) * 4u) != 0 &&
             pulsar_gpu_tensor_write_async(rt, L.slot_b, q->tab_b + nb + 1, (uint64_t)nb * 4u) != 0;
    }
    return ok || fail("could not upload the step's tables");
}

bool embed(const pulsar_driver_step *st0) {
    const ink_step *st = S(st0);
    const pulsar_inkling_shape &s = st->w->shape;
    for (uint32_t r = 0; r < st->n_rows; r++)
        if (st->tokens[r] < 0 || (uint32_t)st->tokens[r] >= s.n_vocab) return fail("a token id outside the vocabulary");
    if (!build_tables(st)) return false;
    pulsar_head_decl hd{};
    hd.n_embd = s.n_embd;
    hd.embed_norm_w = (const uint16_t *)wp(st, st->w->embed_norm, "inkling embed_norm");
    hd.embed_norm_eps = s.rms_eps;
    hd.n_logits = s.n_vocab;
    hd.n_real = s.n_vocab_real;
    hd.logit_scale = 1.0f / s.logit_div;
    const uint16_t *table = (const uint16_t *)wp(st, st->w->embed, "inkling embed");
    const rows_scratch L = rows_layout(st->st->max_rows);
    if (!table || !hd.embed_norm_w) return false;
    return pulsar_head_embed_launch(&hd, table, s.n_vocab,
                                    (const int32_t *)((uint8_t *)dptr(st->st->scratch[ARENA_ROWS]) + L.ids), st->n_rows,
                                    (uint16_t *)dptr(st->st->h), s.n_embd, 0) == 0 ||
           fail("the embedding refused");
}

/* ---- SConv over the step's rows --------------------------------------------------------------------------------- */

/* One stream's rows: x and y are the step's row 0 (bf16, ld apart). */
struct conv_io {
    pulsar_inkling_sconv_stream k;
    const uint16_t *x;
    uint64_t ld_x;
    uint16_t *y;
    uint64_t ld_y;
};

/* n streams of layer il over the step's rows: the verify rows' call with the capture (state after each row: the
 * SNAPSHOT a commit copies back), then the rest */
bool sconv_run(const ink_step *st, uint32_t il, const conv_io *io, uint32_t n, const pulsar_sconv_decl *decl) {
    ink_state *q = st->st;
    const pulsar_inkling_shape &s = st->w->shape;
    const uint64_t ld = pulsar_inkling_sconv_ld(&s), rec = (uint64_t)(s.conv_k - 1u) * ld;
    pulsar_gpu_tensor *t = pulsar_state_tensor(st->states, il, PULSAR_INKLING_STATE_SCONV);
    if (!t) return failf("no SConv state", il);
    uint16_t *state = (uint16_t *)dptr(t);
    const rows_scratch L = rows_layout(q->max_rows);
    uint8_t *rt = (uint8_t *)dptr(q->scratch[ARENA_ROWS]);
    for (int call = 0; call < 2; call++) {
        const bool cap = call == 0;
        const uint32_t r0 = cap ? 0u : q->rows_cap, r1 = cap ? q->rows_cap : st->n_rows;
        if (r1 == r0) continue;
        uint16_t *capture = NULL;
        if (cap) {
            capture = (uint16_t *)pulsar_state_capture_rows(st->states, il, PULSAR_INKLING_STATE_SCONV);
            if (!capture) return failf("a verify step with no SConv capture", il);
        }
        pulsar_sconv_stream ss[PULSAR_SCONV_STREAMS_MAX];
        for (uint32_t j = 0; j < n; j++) {
            const uint64_t off = pulsar_inkling_sconv_offset(&s, io[j].k);
            ss[j] = pulsar_sconv_stream{};
            ss[j].decl = &decl[j];
            ss[j].x = io[j].x + (size_t)r0 * io[j].ld_x;
            ss[j].ld_x = io[j].ld_x;
            ss[j].y = io[j].y + (size_t)r0 * io[j].ld_y;
            ss[j].ld_y = io[j].ld_y;
            ss[j].state = state + off;
            ss[j].state_ld = ld;
            ss[j].state_stride = rec;
            ss[j].capture = capture ? capture + off : NULL;
            ss[j].capture_ld = capture ? ld : 0;
            ss[j].capture_stride = capture ? rec : 0;
        }
        pulsar_sconv_rows rows{};
        rows.n_seqs = cap ? q->n_seq_cap : q->n_seq - q->n_seq_cap;
        rows.n_rows = r1 - r0;
        rows.max_seq_rows = cap ? q->max_run_a : q->max_run_b;
        rows.seq_first = (const int32_t *)(rt + (cap ? L.first_a : L.first_b));
        rows.seq_slot = (const int32_t *)(rt + (cap ? L.slot_a : L.slot_b));
        if (pulsar_sconv_forward(ss, n, &rows, 0) != 0) return failf("the short convolution refused", il);
    }
    return true;
}

pulsar_sconv_decl conv_decl(const ink_step *st, const pulsar_tensor *w, uint32_t channels, const char *what) {
    pulsar_sconv_decl d{};
    d.channels = channels;
    d.kernel = st->w->shape.conv_k;
    d.residual = 1;   /* y = x + conv(x): vLLM's fused_sconv(use_residual=True), no bias, no activation */
    d.weight = (const uint16_t *)wp(st, w, what);
    return d;
}

/* ---- the layer --------------------------------------------------------------------------------------------------- */

/* attention: xn (the attn_norm of h, written by the caller) -> h += SConv_attn(o_proj(GQA(...))); xn = mlp_norm(h) */
bool attention(const ink_step *st, uint32_t il, bool window) {
    ink_state *q = st->st;
    const pulsar_inkling_shape &s = st->w->shape;
    const pulsar_inkling_block &L = st->w->layer[il];
    const attn_dims a = attn_of(s, window);
    const uint32_t n = st->n_rows, E = s.n_embd;
    const attn_scratch A = attn_layout(q->max_rows, st->states->ctx);
    uint8_t *base = (uint8_t *)dptr(q->scratch[ARENA_ATTN]);
    float *qf = (float *)(base + A.qf), *kf = (float *)(base + A.kf), *vf = (float *)(base + A.vf),
          *rf = (float *)(base + A.rf), *of = (float *)(base + A.of);
    uint16_t *qb = (uint16_t *)(base + A.qb), *kb = (uint16_t *)(base + A.kb), *vb = (uint16_t *)(base + A.vb),
             *rb = (uint16_t *)(base + A.rb), *kc = (uint16_t *)(base + A.kc), *vc = (uint16_t *)(base + A.vc),
             *ab = (uint16_t *)(base + A.ab), *db = (uint16_t *)(base + A.db), *yb = (uint16_t *)(base + A.yb);
    const uint16_t *xn = (const uint16_t *)dptr(q->xn);
    /* q | k | v | r: one multi launch per row kind (vLLM's merged qkvr GEMM), each output's one bf16 rounding at the
     * producer stage (a bf16 GEMM's output) */
    {
        const pulsar_tensor *t[4] = {L.attn_q, L.attn_k, L.attn_v, L.attn_rel_proj};
        const int outs[4] = {(int)a.q, (int)a.kv, (int)a.kv, (int)a.r};
        const char *what[4] = {"inkling wq_du", "inkling wk_dv", "inkling wv_dv", "inkling wr_du"};
        float *y[4] = {qf, kf, vf, rf};
        if (!step_linears(st, 4, t, (int)E, outs, what, xn, y, base + A.lin, A.lin_bytes)) return failf("qkvr", il);
        const float *src[4] = {qf, kf, vf, rf};
        uint16_t *dst[4] = {qb, kb, vb, rb};
        for (int j = 0; j < 4; j++)
            if (pulsar_rows_emit_bf16_launch(src[j], (uint64_t)outs[j], n, (uint32_t)outs[j], NULL,
                                             PULSAR_ROWS_SCALE_NONE, dst[j], (uint64_t)outs[j], 0) != 0)
                return failf("the qkvr bf16 emit refused", il);
    }
    /* K and V through their convolutions (one launch), before the k-norm */
    {
        const pulsar_sconv_decl d[2] = {conv_decl(st, L.conv_k, a.kv, "inkling k_sconv"),
                                        conv_decl(st, L.conv_v, a.kv, "inkling v_sconv")};
        if (!d[0].weight || !d[1].weight) return false;
        const conv_io io[2] = {{PULSAR_INKLING_SCONV_K, kb, a.kv, kc, a.kv}, {PULSAR_INKLING_SCONV_V, vb, a.kv, vc, a.kv}};
        if (!sconv_run(st, il, io, 2, d)) return false;
    }
    /* F1: the layer's sequences on its ring (window) or pool (global) */
    {
        const pulsar_gqa_layer G1 = gqa_decl(s, window, (const uint16_t *)wp(st, L.attn_q_norm, "inkling q_norm"),
                                             (const uint16_t *)wp(st, L.attn_k_norm, "inkling k_norm"),
                                             (const uint16_t *)wp(st, L.attn_rel_basis, "inkling rel_logits_proj"));
        if (!G1.q_norm_w || !G1.k_norm_w || !G1.rel_proj) return false;
        const uint64_t tb = pulsar_gqa_kv_token_bytes(&G1);
        for (uint32_t k = 0; k < q->n_seq; k++) {
            const uint32_t bank = (uint32_t)q->seq_bank[k];
            pulsar_gqa_seq &sq = q->seqs[k];
            if (window) {
                pulsar_gpu_tensor *ring = pulsar_state_tensor(st->states, il, PULSAR_INKLING_STATE_KV_WIN);
                const uint64_t bb = pulsar_state_decl_bank_bytes(&st->states->plan->decl[PULSAR_INKLING_STATE_KV_WIN]);
                if (!ring) return failf("no window ring", il);
                sq.kv = (uint8_t *)dptr(ring) + (uint64_t)bank * bb;
                sq.bytes = bb;
                sq.cap = (uint32_t)(bb / tb);
            } else {
                pulsar_gpu_tensor *pool = pulsar_state_pool(st->states, il, PULSAR_INKLING_STATE_KV, bank);
                if (!pool) return failf("a global layer's pool is evicted", il);
                sq.kv = dptr(pool);
                sq.bytes = pulsar_gpu_tensor_bytes(pool);
                sq.cap = (uint32_t)(sq.bytes / tb);
            }
        }
        pulsar_gqa_io io{};
        io.q = {qb, a.q, a.hd};
        io.k = {kc, a.kv, a.hd};
        io.v = {vc, a.kv, a.hd};
        io.r = {rb, a.r, s.d_rel};
        io.out_bf16 = ab;
        io.ld_out = a.q;
        io.workspace = base + A.gqa;
        io.workspace_bytes = A.gqa_bytes;
        if (pulsar_gqa_forward(&G1, q->seqs, q->n_seq, q->row_seq, q->row_pos, n, &io, 0) != 0)
            return failf("the attention refused", il);
    }
    /* the o_proj, its bf16 row, the attention delta's convolution, the residual add + the MLP norm */
    {
        const pulsar_tensor *t[1] = {L.attn_o};
        const int outs[1] = {(int)E};
        const char *what[1] = {"inkling wo_ud"};
        float *y[1] = {of};
        if (!step_linears(st, 1, t, (int)a.q, outs, what, ab, y, base + A.lin, A.lin_bytes)) return failf("wo_ud", il);
        if (pulsar_rows_emit_bf16_launch(of, E, n, E, NULL, PULSAR_ROWS_SCALE_NONE, db, E, 0) != 0)
            return failf("the o_proj bf16 emit refused", il);
        const pulsar_sconv_decl d[1] = {conv_decl(st, L.conv_attn_out, E, "inkling attn_sconv")};
        if (!d[0].weight) return false;
        const conv_io io[1] = {{PULSAR_INKLING_SCONV_ATTN, db, E, yb, E}};
        if (!sconv_run(st, il, io, 1, d)) return false;
        const uint16_t *mn = (const uint16_t *)wp(st, L.ffn_norm, "inkling mlp_norm");
        if (!mn || pulsar_rows_add_launch((uint16_t *)dptr(q->h), E, yb, E, n, E, mn, s.rms_eps,
                                          (uint16_t *)dptr(q->xn), E, 0) != 0)
            return failf("the attention residual add refused", il);
    }
    return true;
}

/* the dense SwiGLU MLP of xn into fo (f32), x its global scale at the emit: vLLM's down then mul_(global_scale) */
bool dense_mlp(const ink_step *st, uint32_t il, float *fo, uint8_t *base, const ffn_scratch &F) {
    const pulsar_inkling_shape &s = st->w->shape;
    const pulsar_inkling_block &L = st->w->layer[il];
    const int E = (int)s.n_embd, M = (int)s.n_ff_dense;
    const uint16_t *xn = (const uint16_t *)dptr(st->st->xn);
    /* the shared launcher adds into its output (out += down(silu(gate x) * up x)): the block's output starts at 0 */
    pulsar_gpu_tensor *v = pulsar_gpu_tensor_view(st->st->scratch[ARENA_FFN], F.fo, (uint64_t)st->n_rows * E * 4u);
    const bool z = v && pulsar_gpu_tensor_fill_f32(v, 0.0f, (uint64_t)st->n_rows * E) != 0;
    pulsar_gpu_tensor_free(v);
    if (!z) return failf("could not clear the MLP output", il);
    const bool ok = arm_segments(st, 0, st->n_rows, [&](bool prompt, uint32_t r0, uint32_t m) {
        pulsar_rows_moe_shared q{};
        q.scale = NULL;   /* an ungated MLP */
        q.yg = (float *)(base + F.yg);
        q.yu = (float *)(base + F.yu);
        q.ys = (float *)(base + F.ys);
        q.h = (uint16_t *)(base + F.hh);
        q.ws = base + F.lin;
        q.ws_bytes = F.lin_bytes;
        return pulsar_linear_rows_ref(st->model, L.ffn_gate, E, M, prompt, "inkling dense gate", &q.gate) &&
               pulsar_linear_rows_ref(st->model, L.ffn_up, E, M, prompt, "inkling dense up", &q.up) &&
               pulsar_linear_rows_ref(st->model, L.ffn_down, M, E, prompt, "inkling dense down", &q.down) &&
               pulsar_rows_moe_shared_launch(&q, xn + (size_t)r0 * E, (int)m, fo + (size_t)r0 * E, 0) == 0;
    });
    return ok || failf("the dense MLP refused", il);
}

/* the MoE block of xn into fo (f32): F2's router, the routed experts and the two sink-gated shared experts */
bool moe(const ink_step *st, uint32_t il, float *fo, uint8_t *base, const ffn_scratch &F) {
    const pulsar_inkling_shape &s = st->w->shape;
    const pulsar_inkling_block &L = st->w->layer[il];
    const uint32_t n = st->n_rows, NS = s.n_shared;
    const int E = (int)s.n_embd, MID = (int)s.n_ff_exp;
    const uint16_t *xn = (const uint16_t *)dptr(st->st->xn);
    const pulsar_moe_block_shape bs = moe_shape(s);
    pulsar_moe_block_parts parts;
    if (pulsar_moe_block_carve(&bs, (int)n, base + F.u, F.moe_bytes, &parts) != 0) return failf("the MoE workspace", il);
    /* the shared experts: three EXL3 dense slices ride the routed launches (forked here, ahead of the router, per
     * row-kind segment; each segment's routed launch joins its forks); another format runs after the routed sum */
    pulsar_rows_linear g0[PULSAR_INKLING_SHARED_MAX], u0[PULSAR_INKLING_SHARED_MAX], d0[PULSAR_INKLING_SHARED_MAX];
    bool ride = true;
    for (uint32_t j = 0; j < NS; j++) {
        if (!(pulsar_linear_rows_ref(st->model, L.sh_gate[j], E, MID, false, "inkling shared gate", &g0[j]) &&
              pulsar_linear_rows_ref(st->model, L.sh_up[j], E, MID, false, "inkling shared up", &u0[j]) &&
              pulsar_linear_rows_ref(st->model, L.sh_down[j], MID, E, false, "inkling shared down", &d0[j])))
            return failf("a shared expert", il);
        ride = ride && g0[j].k2 && u0[j].k2 && d0[j].k2;
    }
    pulsar_rows_moe_shared shs[2][PULSAR_INKLING_SHARED_MAX];   /* at most a decode and a prompt segment */
    int n_seg = 0;
    if (ride && !arm_segments(st, 0, n, [&](bool prompt, uint32_t r0, uint32_t rows) {
            for (uint32_t j = 0; j < NS; j++) {
                pulsar_rows_moe_shared &sh = shs[n_seg][j];
                sh = pulsar_rows_moe_shared{};
                sh.gate = g0[j]; sh.up = u0[j]; sh.down = d0[j];
                sh.gate.prompt = sh.up.prompt = sh.down.prompt = prompt;
                sh.scale = parts.gammas + (size_t)j * n + r0;   /* the router writes it, on this stream, first */
                sh.yg = parts.yg + ((size_t)j * n + r0) * MID;
                sh.yu = parts.yu + ((size_t)j * n + r0) * MID;
                sh.ys = parts.ys + ((size_t)j * n + r0) * E;
                sh.h = parts.h + ((size_t)j * n + r0) * MID;
                sh.ws = parts.lin; sh.ws_bytes = parts.lin_bytes;
                if (pulsar_rows_moe_shared_fork(&sh, xn + (size_t)r0 * E, (int)rows, 0) != 0) return false;
            }
            n_seg++;
            return true;
        }))
        return failf("a shared expert's fork was refused", il);
    /* the router: f32 logits over the routed rows and the sink rows (the weight's last S rows), then the select */
    const pulsar_router_cfg rc = router_cfg(s);
    pulsar_router_layer rl{};
    rl.bias = (const float *)wp(st, L.router_select_bias, "inkling gate bias");
    rl.global_scale = (const float *)wp(st, L.router_out_scale, "inkling gate global_scale");
    const uint16_t *rw = (const uint16_t *)wp(st, L.router, "inkling gate");
    if (!rw || !rl.bias || !rl.global_scale ||
        pulsar_router_logits_launch(xn, (int)n, E, rw, bs.n_logit_rows, NULL, 0, parts.logits, 0) != 0 ||
        pulsar_router_select_launch(&rc, &rl, parts.logits, NULL, (int)n, parts.sel, parts.wts, parts.gammas, NULL,
                                    0) != 0)
        return failf("the router refused", il);
    /* the routed experts through the core's front door (moe.cpp): the gate + up pair and the down */
    const rows_scratch R = rows_layout(st->st->max_rows);
    pulsar_moe_rows_call c{};
    c.m = st->model;
    c.gate = L.exp_gate;
    c.up = L.exp_up;
    c.down = L.exp_down;
    c.top_k = s.n_expert_used;
    c.ws = parts.routed;
    c.ws_bytes = parts.routed_bytes;
    c.nf_flag = (uint32_t *)((uint8_t *)dptr(st->st->scratch[ARENA_ROWS]) + R.nf);
    c.nf_code = 0x49000000u | il;
    int seg = 0;
    if (!arm_segments(st, 0, n, [&](bool prompt, uint32_t r0, uint32_t rows) {
            c.shared = ride ? shs[seg++] : NULL;
            c.n_shared = ride ? (int)NS : 0;
            c.selected = parts.sel + (size_t)r0 * c.top_k;
            c.weights = parts.wts + (size_t)r0 * c.top_k;
            c.x_bf16 = xn + (size_t)r0 * E;
            c.n_rows = (int)rows;
            c.out = fo + (size_t)r0 * E;
            c.prompt = prompt;
            return pulsar_moe_routed_rows(&c);
        }))
        return failf("the routed experts refused", il);
    if (ride) return true;
    const bool ok = arm_segments(st, 0, n, [&](bool prompt, uint32_t r0, uint32_t rows) {
        for (uint32_t j = 0; j < NS; j++) {
            pulsar_rows_moe_shared q{};   /* the gamma row-indexed; the expert's buffers reused from 0 */
            q.scale = parts.gammas + (size_t)j * n + r0;
            q.yg = parts.yg;
            q.yu = parts.yu;
            q.ys = parts.ys;
            q.h = parts.h;
            q.ws = parts.lin;
            q.ws_bytes = parts.lin_bytes;
            if (!(pulsar_linear_rows_ref(st->model, L.sh_gate[j], E, MID, prompt, "inkling shared gate", &q.gate) &&
                  pulsar_linear_rows_ref(st->model, L.sh_up[j], E, MID, prompt, "inkling shared up", &q.up) &&
                  pulsar_linear_rows_ref(st->model, L.sh_down[j], MID, E, prompt, "inkling shared down", &q.down) &&
                  pulsar_rows_moe_shared_launch(&q, xn + (size_t)r0 * E, (int)rows, fo + (size_t)r0 * E, 0) == 0))
                return false;
        }
        return true;
    });
    return ok || failf("a shared expert refused", il);
}

/* the FFN: xn (the mlp_norm of h) -> h += SConv_ffn(bf16(FFN(xn))) */
bool ffn(const ink_step *st, uint32_t il, bool is_moe) {
    ink_state *q = st->st;
    const pulsar_inkling_shape &s = st->w->shape;
    const pulsar_inkling_block &L = st->w->layer[il];
    const uint32_t n = st->n_rows, E = s.n_embd;
    const ffn_scratch F = ffn_layout(q->max_rows);
    uint8_t *base = (uint8_t *)dptr(q->scratch[ARENA_FFN]);
    float *fo = (float *)(base + F.fo);
    uint16_t *db = (uint16_t *)(base + F.db), *yb = (uint16_t *)(base + F.yb);
    if (!(is_moe ? moe(st, il, fo, base, F) : dense_mlp(st, il, fo, base, F))) return false;
    /* the FFN delta's bf16 row (the dense MLP's global scale applied here, in f32, before the one rounding) */
    const void *gs = is_moe ? NULL : wp(st, L.ffn_out_scale, "inkling mlp global_scale");
    if ((!is_moe && !gs) ||
        pulsar_rows_emit_bf16_launch(fo, E, n, E, gs, is_moe ? PULSAR_ROWS_SCALE_NONE : PULSAR_ROWS_SCALE_BF16, db, E,
                                     0) != 0)
        return failf("the FFN bf16 emit refused", il);
    const pulsar_sconv_decl d[1] = {conv_decl(st, L.conv_ffn_out, E, "inkling mlp_sconv")};
    if (!d[0].weight) return false;
    const conv_io io[1] = {{PULSAR_INKLING_SCONV_FFN, db, E, yb, E}};
    if (!sconv_run(st, il, io, 1, d)) return false;
    return pulsar_rows_add_launch((uint16_t *)dptr(q->h), E, yb, E, n, E, NULL, 0.0f, NULL, 0, 0) == 0 ||
           failf("the FFN residual add refused", il);
}

/* block[PULSAR_LAYER_BLOCKS]: one layer as its description says */
bool block(const pulsar_driver_step *st0, uint32_t il) {
    const ink_step *st = S(st0);
    const pulsar_layer_desc &d = st->plan->desc[il];
    const pulsar_inkling_shape &s = st->w->shape;
    if ((d.mixer != PULSAR_MIXER_ATTN_WINDOW && d.mixer != PULSAR_MIXER_ATTN_FULL) ||
        (d.ffn != PULSAR_FFN_DENSE && d.ffn != PULSAR_FFN_MOE))
        return failf("a layer description with no attention mixer or no FFN", il);
    const uint16_t *an = (const uint16_t *)wp(st, st->w->layer[il].attn_norm, "inkling attn_norm");
    if (!an || pulsar_rows_rmsnorm_launch((const uint16_t *)dptr(st->st->h), s.n_embd, st->n_rows, s.n_embd, an,
                                          s.rms_eps, (uint16_t *)dptr(st->st->xn), s.n_embd, 0) != 0)
        return failf("the attention norm refused", il);
    return attention(st, il, d.mixer == PULSAR_MIXER_ATTN_WINDOW) && ffn(st, il, d.ffn == PULSAR_FFN_MOE);
}

/* step rows [row0, row0 + n) -> logits rows [out0, out0 + n): the final norm, the head linear (the decode arm at every
 * call: a row's logits never depend on the rows beside it), the finish (x 1/16, the padding -> -inf) */
bool head(const pulsar_driver_step *st0, uint32_t row0, uint32_t n, uint32_t out0) {
    const ink_step *st = S(st0);
    const pulsar_inkling_shape &s = st->w->shape;
    if (n == 0 || out0 + n > PULSAR_DRIVER_HEADS_MAX || row0 + n > st->n_rows) return fail("head rows out of range");
    const head_scratch H = head_layout();
    uint8_t *base = (uint8_t *)dptr(st->st->scratch[ARENA_HEAD]);
    uint16_t *xb = (uint16_t *)(base + H.xb);
    const uint16_t *on = (const uint16_t *)wp(st, st->w->output_norm, "inkling norm");
    if (!on || pulsar_rows_rmsnorm_launch((const uint16_t *)dptr(st->st->h) + (size_t)row0 * s.n_embd, s.n_embd, n,
                                          s.n_embd, on, s.rms_eps, xb, s.n_embd, 0) != 0)
        return fail("the final norm refused");
    pulsar_rows_linear l{};
    float *logits = (float *)dptr(st->buf->logits) + (size_t)out0 * s.n_vocab;
    if (!pulsar_linear_rows_ref(st->model, st->w->head, (int)s.n_embd, (int)s.n_vocab, false, "inkling unembed", &l) ||
        pulsar_rows_linear_launch(&l, xb, (int)n, logits, base + H.lin, H.lin_bytes, 0) != 0)
        return fail("the head linear refused");
    pulsar_head_decl hd{};
    hd.n_embd = s.n_embd;
    hd.n_logits = s.n_vocab;
    hd.n_real = s.n_vocab_real;
    hd.logit_scale = 1.0f / s.logit_div;
    return pulsar_head_logits_finish_launch(&hd, logits, n, s.n_vocab, 0) == 0 || fail("the logits finish refused");
}

/* the step's own checks (rule 9): the routed experts' non-finite flag */
bool step_end(const pulsar_driver_step *st0, bool ok) {
    if (!ok) return false;
    const ink_step *st = S(st0);
    uint32_t nf = 0;
    if (!pulsar_gpu_tensor_read(st->st->scratch[ARENA_ROWS], rows_layout(st->st->max_rows).nf, &nf, sizeof(nf)))
        return fail("could not read the MoE non-finite flag");
    if (nf) {
        fprintf(stderr, "pulsar: %s: the MoE output of layer %u went non-finite (code 0x%08x) -- refusing the step\n",
                ARCH, nf & 0xffffffu, nf);
        return false;
    }
    return true;
}

/* the diagnostic tap: the residual stream after layer il, the whole step's rows (bf16 widened to f32), named
 * "inkling_h" -- what the layer grade reads against the reference's hidden_states[il + 1] */
void dump(const pulsar_driver_step *st0, uint32_t il) {
    const ink_step *st = S(st0);
    gpu_graph_debug_dump_hc_tensor("inkling_h", st->st->h, (uint64_t)st->n_rows * st->w->shape.n_embd, il,
                                   (uint32_t)st->pos[st->n_rows - 1]);
}

}  // namespace

/* ---- the state declaration ----------------------------------------------------------------------------------- */

uint32_t pulsar_inkling_sconv_channels(const pulsar_inkling_shape *s, pulsar_inkling_sconv_stream k) {
    /* K / V are a layer's kv heads x head_dim: the profile pins the window and global widths equal, so one record
     * layout serves every layer */
    return k == PULSAR_INKLING_SCONV_K || k == PULSAR_INKLING_SCONV_V ? s->n_head_kv * s->head_dim : s->n_embd;
}
uint32_t pulsar_inkling_sconv_offset(const pulsar_inkling_shape *s, pulsar_inkling_sconv_stream k) {
    uint32_t o = 0;
    for (uint32_t j = 0; j < (uint32_t)k; j++) o += pulsar_inkling_sconv_channels(s, (pulsar_inkling_sconv_stream)j);
    return o;
}
uint32_t pulsar_inkling_sconv_ld(const pulsar_inkling_shape *s) {
    return pulsar_inkling_sconv_offset(s, PULSAR_INKLING_SCONV_STREAMS);
}

void pulsar_inkling_state_plan(pulsar_engine *e) {
    const pulsar_inkling_shape *s = &W(e)->shape;
    const pulsar_layer_plan *lp = &e->plan;
    pulsar_state_plan *p = &e->state_plan;
    *p = pulsar_state_plan{};
    p->name = "inkling";
    p->resume_grid = kResumeGrid;
    p->split_invariant = true;
    p->ckpt_slots = kCkptSlots;
    const pulsar_gqa_layer gw = gqa_decl(*s, true, NULL, NULL, NULL), gg = gqa_decl(*s, false, NULL, NULL, NULL);
    pulsar_state_decl d[PULSAR_INKLING_STATE_COUNT] = {};
    d[PULSAR_INKLING_STATE_SCONV] = {"sconv", PULSAR_STATE_FIXED, PULSAR_STATE_CAPTURE_SNAPSHOT, false, false, 0,
                                     (uint64_t)(s->conv_k - 1u) * pulsar_inkling_sconv_ld(s) * sizeof(uint16_t), 0, 0,
                                     NULL, NULL, NULL};
    d[PULSAR_INKLING_STATE_KV_WIN] = {"kv_win", PULSAR_STATE_RING, PULSAR_STATE_CAPTURE_NONE, false, false, s->window,
                                      pulsar_gqa_kv_token_bytes(&gw), 0, 0, NULL, NULL, NULL};
    d[PULSAR_INKLING_STATE_KV] = {"kv", PULSAR_STATE_POOL, PULSAR_STATE_CAPTURE_NONE, false, false, 1u,
                                  pulsar_gqa_kv_token_bytes(&gg), 0, 0, NULL, NULL, NULL};
    for (uint32_t id = 0; id < PULSAR_INKLING_STATE_COUNT; id++) (void)pulsar_state_plan_add(p, d[id]);
    for (uint32_t il = 0; il < lp->n_layer; il++)
        p->slot[il] = 1u << PULSAR_INKLING_STATE_SCONV |
                      1u << (lp->desc[il].mixer == PULSAR_MIXER_ATTN_WINDOW ? PULSAR_INKLING_STATE_KV_WIN
                                                                            : PULSAR_INKLING_STATE_KV);
    p->n_trunk = lp->n_layer;
    p->n_slots = lp->n_layer;   /* no drafter slot until F6 */
    p->session = 0;
}

const pulsar_driver_ops PULSAR_INKLING_DRIVER = {
    /* .create        = */ create,
    /* .destroy       = */ destroy,
    /* .verifies      = */ verifies,
    /* .trunk         = */ trunk,
    /* .scratch_bytes = */ scratch_bytes,
    /* .n_scratch     = */ ARENA_COUNT,
    /* .embed         = */ embed,
    /* .block         = */ {NULL, NULL, NULL, NULL, block},   /* by pulsar_layer_kind: BLOCKS */
    /* .head          = */ head,
    /* .step_end      = */ step_end,
    /* .absorb        = */ NULL,   /* no drafter lane until F6 */
    /* .rope          = */ NULL,   /* no RoPE: text positions are the relative bias's */
    /* .image_rows    = */ NULL,   /* images: E9 */
    /* .dump          = */ dump,
    /* .artifact      = */ artifact,
};
static_assert(PULSAR_LAYER_BLOCKS == 4 && PULSAR_LAYER_KIND_COUNT == 5, "the block table's slot is the kind's value");

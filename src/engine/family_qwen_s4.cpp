/** @file family_qwen_s4.cpp
 *  @brief The Qwen4-exp ops stream S4: embed, PLE, gated residual, MoE block and head, and their load-time admission. */
/* family_qwen_s4.cpp -- the Qwen4-exp ops stream S4 owns (L251): embed, the PLE
 * injection, the gated-residual read / write, the MoE block and the head
 * (mixer + lm_head); their scratch; and at load, the admission of every tensor
 * those ops read and the PLE row file.
 *
 * Each op resolves its weights to device pointers (the engine's model-range
 * cache) and calls the S4 launchers in src/cuda/pulsar_cuda_qwen.h -- the same
 * launchers tests/qwen_s4_gate.cu grades against the double references.  Formats (family_qwen.h
 * section 4): streams bf16, x bf16 + its armed E4M3 slot, y f32.
 *
 * The PLE's n-gram rows are hashed on the host from the step's tokens (the ids
 * are known before the forward) and read from the row file by the Engram pread
 * pool: embed hashes and ISSUES the gather, the PLE op at layer 1 waits for it,
 * so the reads overlap layer 0. */
#include "pulsar_engine_internal.h"
#include "exl3_trellis.h"
#include "qwen_ngram.h"
#include "tp/pulsar_tp.h"
#include "cuda/pulsar_cuda_qwen.h"
#include "cuda/pulsar_cuda_gdn.h"
#include "cuda/mmq/pulsar_tessera.h"
#include "pulsar_nvtx.h"

#include <sys/stat.h>

/** The Qwen PLE row file's host side (pulsar_qwen_weights::ple_io): the opened row table, its pread pool, the
 *  hash layout, and the pinned buffers one step's gather lands in.  Opened by pulsar_qwen_s4_load, freed at unload. */
struct pulsar_qwen_ple_io {
    /// the opened PLE row file (pulsar.ple_rows.*)
    pulsar_engram_table table;
    /// the pread pool the gathers and read-aheads run on (owned)
    pulsar_engram_io *io;
    /// the n-gram hash parameters read from the row file's header
    pulsar_qwen_ngram_layout layout;
    unsigned char *rows;            ///< pinned host [cap][16][320]: the gathered rows, head order (L289: uploaded async)
    int32_t *map;                   ///< pinned host [cap][5]: the PLE op's row / sequence maps (uploaded async)
    uint64_t *ids;                  ///< host [cap][16]
    uint32_t cap;                   ///< tokens the host buffers hold
    pulsar_engram_gather *pending;  ///< the gather embed issued, waited on by the PLE op
    uint32_t pending_n;             ///< its tokens
    bool ahead;                     ///< L329 (b2): the prompt read-ahead runs (PULSAR_PLE_READ_AHEAD, at load)
};

namespace {

/// bytes of one row's GR inject output: PULSAR_QWEN_HC f32
constexpr uint64_t kGrInjStride = (uint64_t)PULSAR_QWEN_HC * sizeof(float);   /* inj per row */

/// @param st the step
/// @param il layer index (n_layer = the MTP layer)
/// @return layer il's tensors
const pulsar_qwen_layer_weights &layer_w(const pulsar_qwen_step *st, uint32_t il) { return st->w->layer[il]; }

/* the core resolves a weight (the rank's TP slice, else the mapped range: linear.cpp) */
/// @param st the step (its model)
/// @param t the weight
/// @param what name for the refusal message
/// @return t's device pointer, or NULL (reported) when it does not resolve
const void *wptr(const pulsar_qwen_step *st, const pulsar_tensor *t, const char *what) {
    return pulsar_weight_device_ptr(st->model, t, what);
}

/// @param t a device tensor
/// @return its device pointer
void *dptr(pulsar_gpu_tensor *t) { return pulsar_gpu_tensor_device_ptr(t); }

} // namespace

bool pulsar_qwen_tp_slices(pulsar_engine *e, pulsar_tp_plan *plan) {
    const pulsar_qwen_shape *s = &g_qwen_shape;
    pulsar_model *m = &e->model;
    const pulsar_qwen_weights *w = e->qwen_weights;
    const uint32_t nr = pulsar_qwen_tp(s);
    const int rank = (int)s->tp_rank;
    /* L272 P0: every range from the ONE range rule (pulsar_tp_owned_range, as DeepSeek's slices and
     * the vocab gather).  The shape holds the rank's widths (pulsar_qwen_tp_load divided the heads),
     * so the full tensor is nr times each; a range the rule hands out that is not that width refuses. */
    struct span { uint32_t lo, hi; };
    auto owned = [&](uint64_t rank_width, span *o) {
        return pulsar_tp_owned_range(rank, nr, (uint32_t)(rank_width * nr), &o->lo, &o->hi) &&
               (uint64_t)(o->hi - o->lo) == rank_width;
    };
    span qk, vt, qin, kvin, od, ex;
    if (!owned((uint64_t)s->gdn_n_k_head * s->gdn_k_dim, &qk) || !owned((uint64_t)s->gdn_n_v_head * s->gdn_v_dim, &vt) ||
        !owned(pulsar_qwen_qsa_q_in(s), &qin) || !owned(pulsar_qwen_qsa_kv_in(s), &kvin) ||
        !owned(pulsar_qwen_qsa_out_dim(s), &od) || !pulsar_tp_owned_range(rank, nr, s->n_expert, &ex.lo, &ex.hi)) {
        fprintf(stderr, "pulsar: %s: TP rank %d of %u owns no whole range of the heads or the experts -- refusing\n",
                PULSAR_QWEN_ARCH, rank, nr);
        return false;
    }
    const uint64_t qkF = (uint64_t)s->gdn_n_k_head * s->gdn_k_dim * nr;
    /* the rank's q, k, v channels of the GDN conv / in_proj_qkv, in the model's order */
    const uint64_t qkv_lo[3] = {qk.lo, qkF + qk.lo, 2u * qkF + vt.lo};
    const uint64_t qkv_hi[3] = {qk.hi, qkF + qk.hi, 2u * qkF + vt.hi};
    bool ok = true;
    auto out = [&](const pulsar_tensor *t, uint64_t lo, uint64_t hi) {
        ok = ok && pulsar_tp_plan_add1(plan, m, t, PULSAR_TP_AXIS_OUT, lo, hi);
    };
    auto in = [&](const pulsar_tensor *t, uint64_t lo, uint64_t hi) {
        ok = ok && pulsar_tp_plan_add1(plan, m, t, PULSAR_TP_AXIS_IN, lo, hi);
    };
    const uint32_t n_state = e->plan.n_layer + (w->mtp.present ? 1u : 0u);
    for (uint32_t il = 0; ok && il < n_state; il++) {
        const pulsar_qwen_layer_weights &L = w->layer[il];
        if (L.gdn_in_qkv) {
            ok = pulsar_tp_plan_add(plan, m, L.gdn_in_qkv, PULSAR_TP_AXIS_OUT, 3, qkv_lo, qkv_hi);
            out(L.gdn_in_z, vt.lo, vt.hi);
            in(L.gdn_out, vt.lo, vt.hi);
            ok = ok && pulsar_tp_plan_add(plan, m, L.gdn_conv, PULSAR_TP_AXIS_OUT, 3, qkv_lo, qkv_hi);
        }
        if (L.attn_q) {
            out(L.attn_q, qin.lo, qin.hi);
            out(L.attn_k, kvin.lo, kvin.hi);
            out(L.attn_v, kvin.lo, kvin.hi);
            in(L.attn_o, od.lo, od.hi);
        }
        /* the rank's experts [ex.lo, ex.hi) of each stack, staged at open rather than at the first step */
        for (const pulsar_tensor *t : {L.moe_gate_up, L.moe_gate, L.moe_up, L.moe_down})
            if (t) ok = ok && pulsar_tp_plan_add1(plan, m, t, PULSAR_TP_AXIS_EXPERTS, ex.lo, ex.hi);
    }
    if (!ok) fprintf(stderr, "pulsar: %s: TP rank %u: the slices do not declare -- refusing\n", PULSAR_QWEN_ARCH, s->tp_rank);
    return ok;
}

namespace {

/* ---- scratch layouts: one function each, used for sizing AND carving -------- */

/// @param x a byte count
/// @return x rounded up to a multiple of 256
uint64_t a256(uint64_t x) { return (x + 255u) & ~(uint64_t)255u; }

/* key / value: the op's f32 projections the launcher reads (L284 #2: run through the row-kind segments);
 * lin_ws the dense-arm workspace the two share (they serialize on the stream) */
/** The PLE op's scratch layout (ple_layout): byte offsets into scratch[PULSAR_QWEN_OP_PLE]; `total` is its size. */
struct ple_scratch { uint64_t emb, key, value, lin_ws, lin_ws_bytes, row_seq, row_j, seq_first, seq_rows, seq_bank, ws,
                     ws_bytes, total; };
/** @var ple_scratch::emb
 *  byte offset in the op's scratch tensor: the gathered PLE rows, [rows][hidden] bf16 */
/** @var ple_scratch::key
 *  byte offset in the op's scratch tensor: the key projection, [rows][hc_hidden] f32 */
/** @var ple_scratch::value
 *  byte offset in the op's scratch tensor: the value projection, [rows][hidden] f32 */
/** @var ple_scratch::lin_ws
 *  byte offset in the op's scratch tensor: the dense-arm workspace key and value share */
/** @var ple_scratch::lin_ws_bytes
 *  bytes of lin_ws (the larger of the two projections' needs) */
/** @var ple_scratch::row_seq
 *  byte offset in the op's scratch tensor: [rows] i32, each row's sequence index */
/** @var ple_scratch::row_j
 *  byte offset in the op's scratch tensor: [rows] i32, each row's index within its sequence */
/** @var ple_scratch::seq_first
 *  byte offset in the op's scratch tensor: [n_seq] i32, each sequence's first row */
/** @var ple_scratch::seq_rows
 *  byte offset in the op's scratch tensor: [n_seq] i32, each sequence's row count */
/** @var ple_scratch::seq_bank
 *  byte offset in the op's scratch tensor: [n_seq] i32, each sequence's bank */
/** @var ple_scratch::ws
 *  byte offset in the op's scratch tensor: pulsar_qwen_ple's own workspace */
/** @var ple_scratch::ws_bytes
 *  bytes of ws (pulsar_qwen_ple_workspace_bytes) */
/** @var ple_scratch::total
 *  bytes of the whole scratch */
/// @param rows the most rows a step carries
/// @return the layout
ple_scratch ple_layout(uint32_t rows) {
    ple_scratch p{};
    uint64_t o = 0;
    p.emb = o;       o += a256((uint64_t)rows * PULSAR_QWEN_HIDDEN * 2u);
    p.key = o;       o += a256((uint64_t)rows * PULSAR_QWEN_HC_HIDDEN * sizeof(float));
    p.value = o;     o += a256((uint64_t)rows * PULSAR_QWEN_HIDDEN * sizeof(float));
    for (const int out : {PULSAR_QWEN_HC_HIDDEN, PULSAR_QWEN_HIDDEN}) {
        pulsar_rows_linear l{};
        l.in = PULSAR_QWEN_HIDDEN; l.out = out;
        const uint64_t b = a256(pulsar_rows_linear_workspace_bytes(&l, (int)rows));
        if (b > p.lin_ws_bytes) p.lin_ws_bytes = b;
    }
    p.lin_ws = o;    o += p.lin_ws_bytes;
    p.row_seq = o;   o += a256((uint64_t)rows * 4u);
    p.row_j = o;     o += a256((uint64_t)rows * 4u);
    p.seq_first = o; o += a256((uint64_t)rows * 4u);
    p.seq_rows = o;  o += a256((uint64_t)rows * 4u);
    p.seq_bank = o;  o += a256((uint64_t)rows * 4u);
    p.ws = o;
    p.ws_bytes = pulsar_qwen_ple_workspace_bytes((int)rows);
    p.total = o + p.ws_bytes;
    return p;
}
/** The GR read op's scratch layout (gr_layout): byte offsets into scratch[PULSAR_QWEN_OP_GR_READ]. */
struct gr_scratch { uint64_t ws, ws_bytes, inj[2], total; };
/** @var gr_scratch::ws
 *  byte offset in the op's scratch tensor: the GR read's workspace (0) */
/** @var gr_scratch::ws_bytes
 *  bytes of ws */
/** @var gr_scratch::inj
 *  byte offset in the op's scratch tensor: per side (pulsar_qwen_gr_side), the read's inject output the write consumes, [rows][PULSAR_QWEN_HC] f32 */
/** @var gr_scratch::total
 *  bytes of the whole scratch */
/// @param rows the most rows a step carries
/// @return the layout
gr_scratch gr_layout(uint32_t rows) {
    gr_scratch g{};
    g.ws = 0;
    g.ws_bytes = a256(pulsar_qwen_gr_workspace_bytes((int)rows));
    g.inj[0] = g.ws_bytes;
    g.inj[1] = g.inj[0] + a256((uint64_t)rows * kGrInjStride);
    g.total = g.inj[1] + a256((uint64_t)rows * kGrInjStride);
    return g;
}
/* L295 F2: the MoE block as the shared router and the rows MoE front door see it -- the router this family declares
 * (pulsar_qwen_router_cfg), the routed experts, one shared expert gated by its own logit row */
/// @param s the loaded shape
/// @return the MoE block shape (router, routed and shared experts)
pulsar_moe_block_shape moe_block_shape(const pulsar_qwen_shape *s) {
    const pulsar_router_cfg rc = pulsar_qwen_router_cfg((int)s->n_expert, (int)s->n_expert_used);
    pulsar_moe_block_shape b{};
    b.n_expert = rc.n_expert;
    b.top_k = rc.top_k;
    b.n_logit_rows = pulsar_router_logit_rows(&rc);
    b.hidden = (int)s->n_embd;
    b.expert_mid = (int)s->n_ff_exp;
    b.n_shared = rc.n_shared;
    b.shared_mid = (int)s->n_ff_shexp;
    return b;
}
/** The MoE op's scratch layout (moe_layout): byte offsets into scratch[PULSAR_QWEN_OP_MOE]. */
struct moe_scratch { uint64_t nf, ws, ws_bytes, total; };
/** @var moe_scratch::nf
 *  byte offset in the op's scratch tensor: the u32 non-finite flag (cleared at embed, read at step end; 0 = finite, else 0x51000000 | layer) */
/** @var moe_scratch::ws
 *  byte offset in the op's scratch tensor: the MoE block workspace (256, after the flag) */
/** @var moe_scratch::ws_bytes
 *  bytes of ws (pulsar_moe_block_workspace_bytes) */
/** @var moe_scratch::total
 *  bytes of the whole scratch */
/// @param s the loaded shape
/// @param rows the most rows a step carries
/// @return the layout
moe_scratch moe_layout(const pulsar_qwen_shape *s, uint32_t rows) {
    moe_scratch m{};
    m.nf = 0;
    m.ws = 256;
    const pulsar_moe_block_shape b = moe_block_shape(s);
    m.ws_bytes = pulsar_moe_block_workspace_bytes(&b, (int)rows);
    m.total = m.ws + m.ws_bytes;
    return m;
}
/* L288 qwen-exl3-head: lin_ws is the head Linear's dense-arm workspace at the head's widest call -- sized for an
 * n_embd -> n_vocab head whatever format the container stores (the draft head's rows are fewer; an mxfp8_lt head
 * takes none), because the scratch is laid out from the shape, before any head is admitted. */
/** The head op's scratch layout (head_layout): byte offsets into scratch[PULSAR_QWEN_OP_HEAD]. */
struct head_scratch { uint64_t ws, ws_bytes, xkey, lin_ws, lin_ws_bytes, total; };
/** @var head_scratch::ws
 *  byte offset in the op's scratch tensor: the mixer's GR workspace (0) */
/** @var head_scratch::ws_bytes
 *  bytes of ws */
/** @var head_scratch::xkey
 *  byte offset in the op's scratch tensor: the key of the head's bf16 activation slot, PULSAR_QWEN_HEAD_ROWS_MAX x hidden f32-sized */
/** @var head_scratch::lin_ws
 *  byte offset in the op's scratch tensor: the head Linear's dense-arm workspace */
/** @var head_scratch::lin_ws_bytes
 *  bytes of lin_ws, sized for an n_embd -> n_vocab head */
/** @var head_scratch::total
 *  bytes of the whole scratch */
/// @param s the loaded shape
/// @return the layout (independent of the step's rows: the head runs at most PULSAR_QWEN_HEAD_ROWS_MAX)
head_scratch head_layout(const pulsar_qwen_shape *s) {
    head_scratch h{};
    h.ws = 0;
    h.ws_bytes = a256(pulsar_qwen_gr_workspace_bytes((int)PULSAR_QWEN_HEAD_ROWS_MAX));
    h.xkey = h.ws_bytes;   /* the key of the head's bf16 activation slot: [16][2560] f32-sized */
    h.lin_ws = h.xkey + a256((uint64_t)PULSAR_QWEN_HEAD_ROWS_MAX * PULSAR_QWEN_HIDDEN * sizeof(float));
    pulsar_rows_linear head{};
    head.in = (int)s->n_embd;
    head.out = (int)s->n_vocab;
    h.lin_ws_bytes = a256(pulsar_rows_linear_workspace_bytes(&head, (int)PULSAR_QWEN_HEAD_ROWS_MAX));
    h.total = h.lin_ws + h.lin_ws_bytes;
    return h;
}

/* A view of an op's scratch at [off, off + bytes) -- a tensor key for the
 * activation-cache calls and a write target; freed by the caller. */
/// @param st the step
/// @param op whose scratch
/// @param off byte offset
/// @param bytes view length
/// @return the view (caller frees), or NULL when the op has no scratch
pulsar_gpu_tensor *scratch_view(const pulsar_qwen_step *st, pulsar_qwen_op_id op, uint64_t off, uint64_t bytes) {
    pulsar_gpu_tensor *s = st->st->scratch[op];
    return s ? pulsar_gpu_tensor_view(s, off, bytes) : NULL;
}

/** Print a step refusal once.
 * @param what the reason
 * @return false, always */
bool fail(const char *what) {
    fprintf(stderr, "pulsar: %s: %s -- refusing the step\n", PULSAR_QWEN_ARCH, what);
    return false;
}

/* L284 #2: the row-kind boundary (pulsar_qwen_step::n_dec).  Rows [lo, hi) of an arm-sensitive launch, cut at
 * n_dec: `launch(prompt, r0, m)` runs [lo, min(hi, n_dec)) with the decode arms (prompt = false), then
 * [max(lo, n_dec), hi) with the prompt arms; an empty segment is skipped.  The launch builds its descriptor
 * with `prompt` and offsets its row pointers by r0 -- every kernel behind one is row-independent (a row's
 * bytes depend on its own input rows alone), so a segment's rows are the bytes the whole launch gives them.
 * The ONE place an op learns which arm a row takes. */
/// @param st the step (its n_dec)
/// @param lo first row
/// @param hi one past the last row
/// @param launch `bool(bool prompt, uint32_t r0, uint32_t m)`, run per non-empty segment
/// @return false at the first launch that fails
template <class Launch>
bool arm_segments(const pulsar_qwen_step *st, uint32_t lo, uint32_t hi, Launch &&launch) {
    const uint32_t cut = st->n_dec < lo ? lo : st->n_dec > hi ? hi : st->n_dec;
    return (cut == lo || launch(false, lo, cut - lo)) && (cut == hi || launch(true, cut, hi - cut));
}

/* L284 #2: the calls of a launch whose bytes depend on the call's row count (the QSA attention): the decode rows
 * [0, n_dec) as one call, then each prompt run [run_first[k], run_first[k + 1]) as its own -- a classic chunk's one
 * call.  A step without runs (DECODE, the MTP layer's) is one call over its rows of one kind. */
/// @param st the step (its n_dec, runs)
/// @param launch `bool(uint32_t r0, uint32_t m)`, run per call
/// @return false at the first launch that fails
template <class Launch>
bool attention_calls(const pulsar_qwen_step *st, Launch &&launch) {
    const uint32_t n = st->n_rows;
    if (st->n_dec > 0 && !launch(0u, st->n_dec)) return false;
    if (st->n_dec >= n) return true;
    if (!st->n_runs) return launch(st->n_dec, n - st->n_dec);
    for (uint32_t k = pulsar_step_verify_runs(st); k < st->n_runs; k++)
        if (!launch(st->run_first[k], st->run_first[k + 1] - st->run_first[k])) return false;
    return true;
}

/* The GR weights of one site, as the launcher takes them, at one arm. */
/// @param st the step
/// @param g the site's GR tensors
/// @param prompt true for the prompt arm, false for decode
/// @param[out] d the launcher's weights
/// @return false when a weight does not resolve
bool gr_dev(const pulsar_qwen_step *st, const pulsar_qwen_gr_weights &g, bool prompt, pulsar_qwen_gr_dev *d) {
    const uint64_t R = st->shape->n_hc_lowrank, HC = pulsar_qwen_hc_dim(st->shape);
    const uint8_t *down = (const uint8_t *)wptr(st, g.mix_down, "qwen GR W_down");
    const uint8_t *up = (const uint8_t *)wptr(st, g.mix_up, "qwen GR W_up");
    d->norm_w = (const uint16_t *)wptr(st, g.hc_norm, "qwen GR hc_norm");
    d->inject = g.inject ? (const uint16_t *)wptr(st, g.inject, "qwen GR inject") : NULL;
    if (!down || !up || !d->norm_w || (g.inject && !d->inject)) return false;
    /* mxfp8_lt: E4M3 [out][in] then the swizzled E8M0 plane; bf16: [out][in] (the
     * mixer in the graded recipe).  Admission made both the same format. */
    const bool w8 = g.mix_down->type == PULSAR_TENSOR_MXFP8_LT;
    d->prompt = prompt;
    d->down = {down, w8 ? down + R * HC : NULL, (int)R, (int)HC, prompt};
    d->up = {up, w8 ? up + HC * R : NULL, (int)HC, (int)R, prompt};
    return true;
}

/* L272 P4c: a dense linear's reference through the core's front door (the arm by t's format at bf16 rows) */
/// @param st the step (its model)
/// @param t the weight
/// @param in input width
/// @param out output width
/// @param prompt true for the prompt arm
/// @param what name for the refusal message
/// @param[out] l the launch descriptor
/// @return false when t has no arm or does not resolve
bool linear_dev(const pulsar_qwen_step *st, const pulsar_tensor *t, int in, int out, bool prompt, const char *what,
                pulsar_rows_linear *l) {
    return pulsar_linear_rows_ref(st->model, t, in, out, prompt, what, l);
}

/* The step's dense linear over its rows: y [n_rows][out] f32 = t x, x bf16 [n_rows][in] -- each row-kind segment
 * through its arm (arm_segments).  `ws` is the dense-arm workspace, sized for the step's rows. */
/// @param st the step
/// @param t the weight
/// @param in input width
/// @param out output width
/// @param what name for the refusal message
/// @param x device input [n_rows][in] bf16
/// @param[out] y device output [n_rows][out] f32
/// @param ws device workspace
/// @param ws_bytes its size
/// @return false on a refused weight or a failed launch
bool step_linear(const pulsar_qwen_step *st, const pulsar_tensor *t, int in, int out, const char *what,
                 const uint16_t *x, float *y, void *ws, uint64_t ws_bytes) {
    if (st->tap) pulsar_imatrix_note_dense(st->tap, t, x, st->n_rows, (uint32_t)in);
    return arm_segments(st, 0, st->n_rows, [&](bool prompt, uint32_t r0, uint32_t m) {
        pulsar_rows_linear l;
        return linear_dev(st, t, in, out, prompt, what, &l) &&
               pulsar_rows_linear_launch(&l, x + (size_t)r0 * in, (int)m, y + (size_t)r0 * out, ws, ws_bytes, 0) == 0;
    });
}

/* L288 exl3-dense: n (1..4) dense linears of the SAME input x [n_rows][in] -- y[j] [n_rows][out[j]] f32 = t[j] x --
 * in one call per row-kind segment (pulsar_rows_linear_multi_launch: the EXL3 ones share one prep and one GEMV
 * launch per rate; every output the bytes step_linear gives it alone).  `ws` is sized by multi_lin_ws. */
/// @param st the step
/// @param n linears, 1..4
/// @param t [n] the weights
/// @param in the shared input width
/// @param out [n] output widths
/// @param what [n] names for the refusal message
/// @param x device input [n_rows][in] bf16
/// @param[out] y [n] device outputs, y[j] [n_rows][out[j]] f32
/// @param ws device workspace (multi_lin_ws)
/// @param ws_bytes its size
/// @return false on a refused weight or a failed launch
bool step_linear_multi(const pulsar_qwen_step *st, int n, const pulsar_tensor *const *t, int in, const int *out,
                       const char *const *what, const uint16_t *x, float *const *y, void *ws, uint64_t ws_bytes) {
    if (st->tap)
        for (int j = 0; j < n; j++) pulsar_imatrix_note_dense(st->tap, t[j], x, st->n_rows, (uint32_t)in);
    return arm_segments(st, 0, st->n_rows, [&](bool prompt, uint32_t r0, uint32_t m) {
        pulsar_rows_linear l[4];
        const pulsar_rows_linear *lp[4];
        float *yr[4];
        for (int j = 0; j < n; j++) {
            if (!linear_dev(st, t[j], in, out[j], prompt, what[j], &l[j])) return false;
            lp[j] = &l[j];
            yr[j] = y[j] + (size_t)r0 * (size_t)out[j];
        }
        return pulsar_rows_linear_multi_launch(lp, n, x + (size_t)r0 * in, (int)m, yr, ws, ws_bytes, 0) == 0;
    });
}

/* The dense-arm workspace of a block whose n input projections (in -> out[j]) run as one multi launch and whose
 * output projection (out_in -> out_out) runs alone after them: the larger of the two. */
/// @param in the input projections' shared input width
/// @param n input projections, 1..4
/// @param out [n] their output widths
/// @param out_in the output projection's input width
/// @param out_out its output width
/// @param rows the most rows a step carries
/// @return workspace bytes (256-aligned)
uint64_t multi_lin_ws(int in, int n, const int *out, int out_in, int out_out, uint32_t rows) {
    pulsar_rows_linear l[4] = {};
    const pulsar_rows_linear *lp[4];
    for (int j = 0; j < n; j++) {
        l[j].in = in;
        l[j].out = out[j];
        lp[j] = &l[j];
    }
    pulsar_rows_linear o{};
    o.in = out_in;
    o.out = out_out;
    const uint64_t a = a256(pulsar_rows_linear_multi_workspace_bytes(lp, n, (int)rows));
    const uint64_t b = a256(pulsar_rows_linear_workspace_bytes(&o, (int)rows));
    return a > b ? a : b;
}

/* L251 S2: the Gated DeltaNet op's scratch.  qkv/z/a/b hold the four f32
 * projections; `ws` is pulsar_gdn_forward's own scratch; obf16 is the bf16
 * output the kernel emits and out_proj then reads (ac69748f; the A8 slot it
 * replaced is gone, L335).  `lin_ws` is ONE dense-arm workspace shared by the
 * five launches (they serialize on the stream). */
/** The GDN op's scratch layout (gdn_layout): byte offsets into scratch[PULSAR_QWEN_OP_GDN]. */
struct gdn_scratch { uint64_t qkv, z, a, b, ws, ws_bytes, obf16, lin_ws, lin_ws_bytes, total; };

/** @var gdn_scratch::qkv
 *  byte offset in the op's scratch tensor: in_proj_qkv output, [rows][conv_dim] f32 */
/** @var gdn_scratch::z
 *  byte offset in the op's scratch tensor: in_proj_z output, [rows][v_total] f32 */
/** @var gdn_scratch::a
 *  byte offset in the op's scratch tensor: in_proj_a output, [rows][n_v_head (whole)] f32 */
/** @var gdn_scratch::b
 *  byte offset in the op's scratch tensor: in_proj_b output, [rows][n_v_head (whole)] f32 */
/** @var gdn_scratch::ws
 *  byte offset in the op's scratch tensor: pulsar_gdn_forward's scratch */
/** @var gdn_scratch::ws_bytes
 *  bytes of ws */
/** @var gdn_scratch::obf16
 *  byte offset in the op's scratch tensor: the GDN output, [rows][v_total] bf16 -- what out_proj reads */
/** @var gdn_scratch::lin_ws
 *  byte offset in the op's scratch tensor: the dense-arm workspace the five linears share */
/** @var gdn_scratch::lin_ws_bytes
 *  bytes of lin_ws (gdn_lin_ws) */
/** @var gdn_scratch::total
 *  bytes of the whole scratch */
/// @param s the loaded shape
/// @param rows the most rows a step carries
/// @return the GDN linears' dense-arm workspace bytes
uint64_t gdn_lin_ws(const pulsar_qwen_shape *s, uint32_t rows) {
    const int H = (int)s->n_embd, CD = (int)pulsar_qwen_gdn_conv_dim(s);
    const int VT = (int)pulsar_qwen_gdn_v_total(s), NV = (int)(s->gdn_n_v_head * pulsar_qwen_tp(s));   /* a / b whole */
    const int outs[4] = {CD, VT, NV, NV};   /* in_proj_qkv / _z / _a / _b as one multi launch, then out_proj */
    return multi_lin_ws(H, 4, outs, VT, H, rows);
}

/// @param s the loaded shape
/// @param rows the most rows a step carries
/// @return the layout
gdn_scratch gdn_layout(const pulsar_qwen_shape *s, uint32_t rows) {
    const uint64_t CD = pulsar_qwen_gdn_conv_dim(s), VT = pulsar_qwen_gdn_v_total(s), NV = (uint64_t)s->gdn_n_v_head * pulsar_qwen_tp(s);   /* a / b whole */
    gdn_scratch g{};
    uint64_t o = 0;
    g.qkv = o; o += a256((uint64_t)rows * CD * sizeof(float));
    g.z   = o; o += a256((uint64_t)rows * VT * sizeof(float));
    g.a   = o; o += a256((uint64_t)rows * NV * sizeof(float));
    g.b   = o; o += a256((uint64_t)rows * NV * sizeof(float));
    g.ws  = o; g.ws_bytes = a256((uint64_t)pulsar_gdn_scratch_bytes((int)rows)); o += g.ws_bytes;
    g.lin_ws = o; g.lin_ws_bytes = gdn_lin_ws(s, rows); o += g.lin_ws_bytes;
    /* L251 / ac69748f: the GDN output's bf16 row -- what the out_proj reads */
    g.obf16 = o; o += a256((uint64_t)rows * VT * sizeof(uint16_t));
g.total = o;
    return g;
}

/* L251 S3: the QSA op's scratch.  qg/k/v/idx hold the four f32 projections (the
 * op hands the kernel VIEWS of them, because pulsar_qsa_io wants tensors and the
 * scratch IS one); obf16 is the bf16 output the o_proj reads (ac69748f) and a8_sf
 * the scale slab the kernel is handed beside it; ws is pulsar_gpu_qsa_forward's
 * workspace; lin_ws is the shared dense workspace. */
/** The QSA op's scratch layout (qsa_layout): byte offsets into scratch[PULSAR_QWEN_OP_QSA]. */
struct qsa_scratch { uint64_t qg, k, v, idx, a8_sf, obf16, ws, ws_bytes, lin_ws, lin_ws_bytes, total; };

/** @var qsa_scratch::qg
 *  byte offset in the op's scratch tensor: q_proj output (query | gate), [rows][q_in] f32 */
/** @var qsa_scratch::k
 *  byte offset in the op's scratch tensor: k_proj output, [rows][kv_in] f32 */
/** @var qsa_scratch::v
 *  byte offset in the op's scratch tensor: v_proj output, [rows][kv_in] f32 */
/** @var qsa_scratch::idx
 *  byte offset in the op's scratch tensor: indexer index_qk output, [rows][PULSAR_QSA_IDX_IN] f32 */
/** @var qsa_scratch::a8_sf
 *  byte offset in the op's scratch tensor: the scale-factor slab passed as pulsar_qsa_io::out_scale (with out_e4m3 NULL) */
/** @var qsa_scratch::obf16
 *  byte offset in the op's scratch tensor: the attention output, [rows][out_dim] bf16 -- what o_proj reads */
/** @var qsa_scratch::ws
 *  byte offset in the op's scratch tensor: pulsar_gpu_qsa_forward's workspace */
/** @var qsa_scratch::ws_bytes
 *  bytes of ws (at the session's context) */
/** @var qsa_scratch::lin_ws
 *  byte offset in the op's scratch tensor: the shared dense-arm workspace */
/** @var qsa_scratch::lin_ws_bytes
 *  bytes of lin_ws (qsa_lin_ws) */
/** @var qsa_scratch::total
 *  bytes of the whole scratch */
/// @param s the loaded shape
/// @param rows the most rows a step carries
/// @return the QSA linears' dense-arm workspace bytes
uint64_t qsa_lin_ws(const pulsar_qwen_shape *s, uint32_t rows) {
    const int H = (int)s->n_embd;
    const int outs[4] = {(int)pulsar_qwen_qsa_q_in(s), (int)pulsar_qwen_qsa_kv_in(s), (int)pulsar_qwen_qsa_kv_in(s),
                         PULSAR_QSA_IDX_IN};   /* q / k / v / index_qk as one multi launch, then o_proj */
    return multi_lin_ws(H, 4, outs, (int)pulsar_qwen_qsa_out_dim(s), H, rows);
}

/// @param s the loaded shape
/// @param rows the most rows a step carries
/// @param ctx the session's context, in tokens (sizes the attention workspace)
/// @return the layout
qsa_scratch qsa_layout(const pulsar_qwen_shape *s, uint32_t rows, uint32_t ctx) {
    qsa_scratch g{};
    uint64_t o = 0;
    g.qg  = o; o += a256((uint64_t)rows * pulsar_qwen_qsa_q_in(s) * sizeof(float));
    g.k   = o; o += a256((uint64_t)rows * pulsar_qwen_qsa_kv_in(s) * sizeof(float));
    g.v   = o; o += a256((uint64_t)rows * pulsar_qwen_qsa_kv_in(s) * sizeof(float));
    g.idx = o; o += a256((uint64_t)rows * PULSAR_QSA_IDX_IN * sizeof(float));
    g.a8_sf = o; o += a256(pulsar_gpu_mx_sf_slab_bytes((int)rows, pulsar_gpu_mx_kbp(pulsar_qwen_qsa_out_dim(s))));
    g.ws = o; g.ws_bytes = a256((uint64_t)pulsar_gpu_qsa_workspace_bytes(rows, ctx)); o += g.ws_bytes;
    g.lin_ws = o; g.lin_ws_bytes = qsa_lin_ws(s, rows); o += g.lin_ws_bytes;
    /* L251 / ac69748f: the o_proj's bf16 row */
    g.obf16 = o; o += a256((uint64_t)rows * pulsar_qwen_qsa_out_dim(s) * sizeof(uint16_t));
g.total = o;
    (void)s;
    return g;
}

} // namespace

/* ======================================================================== */
/* load                                                                      */

/// Admit or refuse one tensor by a format test, through the core's admission.
/// @param t the tensor
/// @param ok the format test's result
/// @param want the format named in the refusal
/// @return ok (false printed)
static bool admit(const pulsar_tensor *t, bool ok, const char *want) {
    return pulsar_tensor_admit(t, PULSAR_QWEN_ARCH, ok, want);   /* L272 P4a: the core's mechanics */
}
/// @param t the tensor
/// @return true when bf16 (else refused, printed)
static bool admit_bf16(const pulsar_tensor *t) { return admit(t, t->type == PULSAR_TENSOR_BF16, "bf16"); }
/// @param t the tensor
/// @return true when mxfp8_lt (else refused, printed)
static bool admit_mx8(const pulsar_tensor *t) { return admit(t, t->type == PULSAR_TENSOR_MXFP8_LT, "mxfp8_lt"); }
/* L272 P4: a tensor's role at this family's bf16-row activations -- the format registry's answer
 * (weight_format.cpp), so which formats serve is the core's, not this family's */
/// @param t the tensor
/// @param role the weight role it serves
/// @return true when the registry serves its format in that role at bf16 rows
static bool admit_role(const pulsar_tensor *t, pulsar_weight_role role) {
    return pulsar_tensor_admit_role(t, PULSAR_QWEN_ARCH, role, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16));
}
/* a dense Linear the shared launcher (linear_dev) reads */
/// @param t the tensor
/// @return admit_role(t, PULSAR_ROLE_DENSE)
static bool admit_linear(const pulsar_tensor *t) { return admit_role(t, PULSAR_ROLE_DENSE); }
/* L288 qwen-exl3-head: the heads' formats are the container's.  lm_head: bf16 (encoded to MXFP8 on the device at the
 * first head, L251) or a format the dense bf16-rows arm reads (EXL3 at a dense-arm rate) -- under tensor parallelism
 * only bf16, whose rank slices the MXFP8 encode cuts.  The MTP draft head: the artifact's mtp.draft_head in a format
 * the dense arm reads, or -- absent -- the rows gathered from a bf16 lm_head; an lm_head in another format leaves no
 * rows to gather, so such an artifact must carry its own.  One path per artifact; nothing is converted to fit. */
/// @param e the engine (its TP ranks)
/// @param w the weights (output, mtp)
/// @return false (printed) on a head format this artifact cannot serve
static bool admit_head(const pulsar_engine *e, const pulsar_qwen_weights *w) {
    const pulsar_tensor *out = w->output;
    const bool bf16 = out->type == PULSAR_TENSOR_BF16;
    if (!bf16 && !admit_linear(out)) return false;
    if (!bf16 && e->model.tp_n_ranks > 1) {
        fprintf(stderr, "pulsar: %s: lm_head.weight is %s; under tensor parallelism the head is a bf16 tensor the ranks "
                        "encode their vocab slices from -- refusing\n", PULSAR_QWEN_ARCH, tensor_type_name(out->type));
        return false;
    }
    if (!w->mtp.present) return true;
    if (w->mtp.draft_head) return admit_linear(w->mtp.draft_head);
    if (!bf16) {
        fprintf(stderr, "pulsar: %s: lm_head.weight is %s, which leaves no bf16 rows to gather the MTP draft head from, "
                        "and the artifact carries no mtp.draft_head -- refusing\n", PULSAR_QWEN_ARCH,
                tensor_type_name(out->type));
        return false;
    }
    return true;
}

/* the low-rank pair: MXFP8 (the per-layer sites) or BF16 (the mixer, as the
 * graded recipe stores it) -- both arms of the GR read, one format per pair */
/// @param g one gated-residual unit
/// @return false (printed) on a tensor in another format
static bool admit_gr(const pulsar_qwen_gr_weights &g) {
    const bool pair_bf16 = g.mix_down->type == PULSAR_TENSOR_BF16 && g.mix_up->type == PULSAR_TENSOR_BF16;
    bool ok = admit_bf16(g.hc_norm);
    if (!pair_bf16) ok &= admit_mx8(g.mix_down) & admit_mx8(g.mix_up);
    if (g.inject) ok &= admit_bf16(g.inject);
    return ok;
}

/* L255: a Tessera stack's planes in the storage the launcher reads (family_qwen.h PULSAR_QWEN_TESS_*) -- native
 * I32 / BF16 / F32 tensors, so the format registry has no row for them: admitted plane by plane here */
/// @param P the stack's planes, by PULSAR_QWEN_TESS_* id
/// @return false (printed) on a plane in another storage type
static bool admit_tessera(pulsar_tensor *const P[PULSAR_QWEN_TESS_PLANES]) {
    static const uint32_t want[PULSAR_QWEN_TESS_PLANES] = {
        PULSAR_TENSOR_I32, PULSAR_TENSOR_BF16, PULSAR_TENSOR_I32, PULSAR_TENSOR_I32,
        PULSAR_TENSOR_F32, PULSAR_TENSOR_I32,  PULSAR_TENSOR_I32, PULSAR_TENSOR_I32};
    bool ok = true;
    for (int i = 0; i < PULSAR_QWEN_TESS_PLANES; ++i)
        ok &= admit(P[i], P[i]->type == want[i], tensor_type_name(want[i]));
    return ok;
}

/* a layer's MoE: the router, the routed experts (fused gate_up, the MTP layer's gate + up pair, or L255's
 * Tessera planes), the shared expert */
/// @param L the layer
/// @return false (printed) on any refused tensor or expert combination
static bool admit_moe(const pulsar_qwen_layer_weights &L) {
    bool ok = admit_bf16(L.moe_router) & admit_bf16(L.sh_gate_scalar);
    if (L.moe_tess[PULSAR_QWEN_TESS_GATE][PULSAR_QWEN_TESS_WORDS]) {
        for (int p = 0; p < PULSAR_QWEN_TESS_PROJS; ++p) ok &= admit_tessera(L.moe_tess[p]);
    } else {
        if (L.moe_gate_up)
            ok &= admit_role(L.moe_gate_up, PULSAR_ROLE_EXPERT_GATE_UP_FUSED);
        else
            ok &= admit_role(L.moe_gate, PULSAR_ROLE_EXPERT_GATE_UP) & admit_role(L.moe_up, PULSAR_ROLE_EXPERT_GATE_UP);
        ok &= admit_role(L.moe_down, PULSAR_ROLE_EXPERT_DOWN);
        if (ok) ok = pulsar_format_moe_combo(L.moe_gate_up ? L.moe_gate_up : L.moe_gate, L.moe_gate_up ? NULL : L.moe_up,
                                             L.moe_down, PULSAR_ACT_ROWS_BF16, PULSAR_QWEN_ARCH);
    }
    ok &= admit_role(L.sh_gate, PULSAR_ROLE_SHARED_EXPERT) & admit_role(L.sh_up, PULSAR_ROLE_SHARED_EXPERT) &
          admit_role(L.sh_down, PULSAR_ROLE_SHARED_EXPERT);
    return ok;
}

bool pulsar_qwen_s4_load(pulsar_engine *e, const pulsar_engine_options *opt) {
    pulsar_qwen_weights *w = e->qwen_weights;
    const pulsar_qwen_shape *s = &g_qwen_shape;
    bool ok = admit_bf16(w->token_embd) & admit_head(e, w) & admit_gr(w->mixer);
    uint32_t tessera_layers = 0;
    for (uint32_t il = 0; il < e->plan.n_layer; il++) {
        const pulsar_qwen_layer_weights &L = w->layer[il];
        ok &= admit_gr(L.gr_attn) & admit_gr(L.gr_mlp);
        ok &= admit_moe(L);
        tessera_layers += L.moe_tess[PULSAR_QWEN_TESS_GATE][PULSAR_QWEN_TESS_WORDS] != NULL;
        if (L.ple_key) {
            /* EXL3 (our quant) or mxfp8_lt (turboderp's packs leave them F16; the builder encodes the
             * HF bf16): linear_dev launches either */
            ok &= admit_linear(L.ple_key) & admit_linear(L.ple_value);
            ok &= admit_bf16(L.ple_norm_key) & admit_bf16(L.ple_norm_query) & admit_bf16(L.ple_norm_conv) &
                  admit_bf16(L.ple_conv);
        }
    }
    if (w->mtp.present) {
        /* L251 MTP: the layer's GR sites and MoE as the trunk's (split experts); the head side's
         * norms bf16 and its two fc Linears mxfp8_lt (the sidecar recipe); the mixer's pair as the
         * trunk mixer's rule allows */
        const pulsar_qwen_layer_weights &L = w->layer[e->plan.n_layer];
        ok &= admit_gr(L.gr_attn) & admit_gr(L.gr_mlp) & admit_moe(L) & admit_gr(w->mtp.mixer);
        ok &= admit_bf16(w->mtp.norm_embd) & admit_bf16(w->mtp.norm_hidden) &
              admit_mx8(w->mtp.fc_embd) & admit_mx8(w->mtp.fc_hidden);
    }
    if (!ok) return false;

    /* the PLE row file, beside the container */
    pulsar_str file = {NULL, 0};
    uint64_t n_rows = 0;
    if (!model_get_string(&e->model, "pulsar.ple_rows.file", &file) ||
        !model_get_u64_compat(&e->model, "pulsar.ple_rows.n_rows", &n_rows)) {
        fprintf(stderr, "pulsar: %s: the container names no PLE row file (pulsar.ple_rows.file / .n_rows) -- "
                        "refusing\n", PULSAR_QWEN_ARCH);
        return false;
    }
    char path[4096];
    struct stat mst;
    const char *mp = opt->model_path ? opt->model_path : "";
    if (stat(mp, &mst) == 0 && S_ISDIR(mst.st_mode)) {
        snprintf(path, sizeof(path), "%s/%.*s", mp, (int)file.len, file.ptr);
    } else {
        const char *slash = strrchr(mp, '/');
        snprintf(path, sizeof(path), "%.*s%.*s", slash ? (int)(slash - mp + 1) : 0, mp, (int)file.len, file.ptr);
    }
    pulsar_qwen_ple_io *io = (pulsar_qwen_ple_io *)xcalloc(1, sizeof(*io));
    pulsar_qwen_ngram_layout &Ly = io->layout;
    for (uint32_t i = 0; i < PULSAR_QWEN_NGRAM_ORDER; i++) Ly.mult[i] = (int64_t)w->ple_multipliers[i];
    for (uint32_t h = 0; h < PULSAR_QWEN_NGRAM_COLS; h++) {
        Ly.prime[h] = w->ple_head_vocab[h];
        Ly.offset[h] = w->ple_head_offset[h];
    }
    Ly.eos = (int32_t)s->eos_id;
    Ly.vocab = s->n_vocab;
    Ly.n_rows = n_rows;
    if (s->ngram_size != PULSAR_QWEN_NGRAM_ORDER || (s->ngram_size - 1u) * s->ngram_heads != PULSAR_QWEN_NGRAM_COLS ||
        !pulsar_qwen_ngram_layout_check(&Ly) || !pulsar_qwen_ngram_table_open(&io->table, path, s->ple_layer, &Ly)) {
        fprintf(stderr, "pulsar: %s: the PLE row file %s does not open against this model -- refusing\n",
                PULSAR_QWEN_ARCH, path);
        free(io);
        return false;
    }
    /* L329 (b2): the next prompt chunk's rows read ahead (prefill_loop.cpp); PULSAR_PLE_READ_AHEAD=0 is the A/B's off
     * arm -- a diagnostic switch, no byte changes either way -- and any other value but 1 refuses the load */
    const char *ra = getenv("PULSAR_PLE_READ_AHEAD");
    if (ra && strcmp(ra, "0") != 0 && strcmp(ra, "1") != 0) {
        fprintf(stderr, "pulsar: %s: PULSAR_PLE_READ_AHEAD=%s -- 0 (off) or 1 (on, the default); refusing\n",
                PULSAR_QWEN_ARCH, ra);
        pulsar_engram_table_close(&io->table);
        free(io);
        return false;
    }
    io->ahead = !ra || strcmp(ra, "1") == 0;
    io->io = pulsar_engram_io_create(PULSAR_ENGRAM_IO_THREADS);
    if (!io->io) {
        pulsar_engram_table_close(&io->table);
        free(io);
        return false;
    }
    w->ple_io = io;
    if (io->ahead)
        fprintf(stderr, "pulsar: L329 (b2) qwen PLE read-ahead: on -- a prefill reads the next chunk's rows on %u "
                        "background readers, below every gather (PULSAR_PLE_READ_AHEAD=0 turns it off)\n",
                PULSAR_ENGRAM_AHEAD_READERS);
    else
        fprintf(stderr, "pulsar: L329 (b2) qwen PLE read-ahead: OFF (PULSAR_PLE_READ_AHEAD=0)\n");
    fprintf(stderr, "pulsar: L251 qwen PLE rows: %s (%llu rows x %u B, %u per token, %u pread threads); S4 tensors "
                    "admitted (routed experts: %u layers fused gate_up + down EXL3, %u layers Tessera value family "
                    "[L255]; dense EXL3, GR low-rank MXFP8)\n",
            path, (unsigned long long)n_rows, PULSAR_QWEN_NGRAM_ROW_BYTES, PULSAR_QWEN_NGRAM_COLS,
            PULSAR_ENGRAM_IO_THREADS, e->plan.n_layer - tessera_layers, tessera_layers);
    /* L288 qwen-exl3-head: the path each head takes, by the format the container stores it in */
    const bool out_bf16 = w->output->type == PULSAR_TENSOR_BF16;
    fprintf(stderr, "pulsar: %s: lm_head %s -> %s", PULSAR_QWEN_ARCH, tensor_type_name(w->output->type),
            out_bf16 ? "MXFP8 on the device, the mxfp8_lt rows arm" : "the dense bf16-rows arm as stored");
    if (w->mtp.present)
        fprintf(stderr, "; MTP draft head %s", w->mtp.draft_head ? tensor_type_name(w->mtp.draft_head->type)
                                                                 : "gathered from lm_head into MXFP8");
    fprintf(stderr, "\n");
    return true;
}

void pulsar_qwen_s4_unload(pulsar_qwen_weights *w) {
    if (w && w->head_mx) {
        pulsar_gpu_tensor_free(w->head_mx);
        w->head_mx = NULL;
    }
    if (w) {
        pulsar_gpu_tensor_free(w->draft_head_mx);
        pulsar_gpu_tensor_free(w->draft_ids_dev);
        w->draft_head_mx = NULL;
        w->draft_ids_dev = NULL;
    }
    if (w) {
        free(w->draft_ids);
        w->draft_ids = NULL;
        w->n_draft = 0;
    }
    pulsar_qwen_ple_io *io = w ? w->ple_io : NULL;
    if (!io) return;
    if (io->pending) (void)pulsar_engram_gather_wait(io->pending);
    pulsar_engram_io_destroy(io->io);
    pulsar_engram_table_close(&io->table);
    pulsar_gpu_host_free(io->rows);
    pulsar_gpu_host_free(io->map);
    free(io->ids);
    free(io);
    w->ple_io = NULL;
}

uint64_t pulsar_qwen_s4_scratch_bytes(pulsar_qwen_op_id op, const pulsar_qwen_shape *s, uint32_t max_rows, uint32_t ctx) {
    switch (op) {
    case PULSAR_QWEN_OP_EMBED:   return (uint64_t)max_rows * sizeof(int32_t);
    case PULSAR_QWEN_OP_PLE:     return ple_layout(max_rows).total;
    case PULSAR_QWEN_OP_GR_READ: return gr_layout(max_rows).total;
    case PULSAR_QWEN_OP_MOE:     return moe_layout(s, max_rows).total;
    case PULSAR_QWEN_OP_HEAD:    return head_layout(s).total;
    case PULSAR_QWEN_OP_GDN:     return gdn_layout(s, max_rows).total;
    case PULSAR_QWEN_OP_QSA:     return qsa_layout(s, max_rows, ctx).total;
    default:                     return 0;   /* GR write reads GR read's inj; GDN / QSA are S2's / S3's */
    }
}

/* ======================================================================== */
/* the ops                                                                   */

bool pulsar_qwen_s4_embed(const pulsar_qwen_step *st) {
    const pulsar_qwen_shape *s = st->shape;
    const uint32_t n = st->n_rows;
    for (uint32_t r = 0; r < n; r++)
        if (st->tokens[r] < 0 || (uint32_t)st->tokens[r] >= s->n_vocab) return fail("a token id outside the vocabulary");
    /* the MoE non-finite flag starts every step clear (read at pulsar_qwen_s4_step_end) */
    pulsar_gpu_tensor *moe_sc = st->st->scratch[PULSAR_QWEN_OP_MOE];
    static const uint32_t zero = 0;   /* L289: an enqueued write reads it later */
    if (!moe_sc || !pulsar_gpu_tensor_write_async(moe_sc, moe_layout(st->shape, st->st->max_rows).nf, &zero, sizeof(zero)))
        return fail("could not clear the MoE non-finite flag");
    pulsar_gpu_tensor *tok = st->st->scratch[PULSAR_QWEN_OP_EMBED];
    const pulsar_tensor *te = st->w->token_embd;
    const uint16_t *table = (const uint16_t *)wptr(st, te, "qwen embed_tokens");
    if (!tok || !table || !pulsar_gpu_tensor_write_async(tok, 0, st->tokens, (uint64_t)n * sizeof(int32_t))) return false;
    if (pulsar_qwen_embed_launch(table, (const int32_t *)dptr(tok), (int)n, (int)s->n_vocab,
                                 (uint16_t *)dptr(st->streams), 0))
        return false;

    /* the PLE rows of these tokens: hash on the host, issue the gather now */
    pulsar_qwen_ple_io *io = st->w->ple_io;
    if (!io) return fail("no PLE row file is open");
    if (io->pending) return fail("a previous step's PLE gather was never consumed");
    if (io->cap < n) {
        /* the last step's uploads from these buffers finished at its drain */
        pulsar_gpu_host_free(io->rows);
        pulsar_gpu_host_free(io->map);
        free(io->ids);
        io->rows = (unsigned char *)pulsar_gpu_host_alloc((uint64_t)n * PULSAR_QWEN_NGRAM_COLS * PULSAR_QWEN_NGRAM_ROW_BYTES);
        io->map = (int32_t *)pulsar_gpu_host_alloc((uint64_t)n * 5u * sizeof(int32_t));
        io->ids = (uint64_t *)xmalloc((size_t)n * PULSAR_QWEN_NGRAM_COLS * sizeof(uint64_t));
        io->cap = io->rows && io->map ? n : 0;
        if (!io->cap) return fail("could not allocate the pinned PLE row buffers");
    }
    for (uint32_t r = 0; r < n; r++) {
        int32_t *ctx_b = (int32_t *)pulsar_state_host(st->states, PULSAR_QWEN_STATE_NGRAM, (uint32_t)st->bank[r]);   /* oldest first */
        pulsar_qwen_ngram_ctx ctx = {{ctx_b[0], ctx_b[1]}};
        pulsar_qwen_ngram_rows(&io->layout, &ctx, &st->tokens[r], 1, io->ids + (size_t)r * PULSAR_QWEN_NGRAM_COLS);
        ctx_b[0] = ctx.prev[0];
        ctx_b[1] = ctx.prev[1];
    }
    io->pending = pulsar_engram_gather_start(io->io, &io->table, io->ids, n * PULSAR_QWEN_NGRAM_COLS, io->rows);
    io->pending_n = n;
    return io->pending != NULL || fail("the PLE gather did not start");
}

void pulsar_qwen_s4_ple_prefetch(const pulsar_qwen_weights *w, int32_t ctx[2], const int32_t *tokens, uint32_t n) {
    pulsar_qwen_ple_io *io = w->ple_io;
    if (!io || n == 0 || n > PULSAR_QWEN_SPEC_ROWS) return;
    for (uint32_t r = 0; r < n; r++)
        if (tokens[r] < 0 || (uint32_t)tokens[r] >= io->layout.vocab) return;   /* the embed refuses it by name */
    uint64_t ids[PULSAR_QWEN_SPEC_ROWS * PULSAR_QWEN_NGRAM_COLS];
    pulsar_qwen_ngram_ctx c = {{ctx[0], ctx[1]}};
    pulsar_qwen_ngram_rows(&io->layout, &c, tokens, (int)n, ids);
    ctx[0] = c.prev[0];
    ctx[1] = c.prev[1];
    pulsar_engram_prefetch(io->io, &io->table, ids, n * PULSAR_QWEN_NGRAM_COLS);
}

uint32_t pulsar_qwen_s4_ple_read_ahead(const pulsar_qwen_weights *w, uint64_t tag, const int32_t *prompt, uint32_t from,
                                       uint32_t to) {
    pulsar_qwen_ple_io *io = w ? w->ple_io : NULL;
    if (!io || !io->ahead || !prompt || to <= from) return 0;
    /* the n-gram reaches two tokens back: the context at `from` is the prompt's own (EOS before its start) */
    for (uint32_t i = from >= 2u ? from - 2u : 0u; i < to; i++)
        if (prompt[i] < 0 || (uint32_t)prompt[i] >= io->layout.vocab) return 0;   /* the embed refuses it by name */
    pulsar_qwen_ngram_ctx c;
    pulsar_qwen_ngram_ctx_init(&io->layout, &c);
    if (from >= 2u) c.prev[0] = prompt[from - 2u];
    if (from >= 1u) c.prev[1] = prompt[from - 1u];
    const uint32_t n = to - from;
    std::vector<uint64_t> ids((size_t)n * PULSAR_QWEN_NGRAM_COLS);
    pulsar_qwen_ngram_rows(&io->layout, &c, prompt + from, n, ids.data());
    return pulsar_engram_read_ahead(io->io, &io->table, ids.data(), n * PULSAR_QWEN_NGRAM_COLS, tag);
}

pulsar_engram_io *pulsar_qwen_s4_ple_pool(const pulsar_qwen_weights *w) {
    return w && w->ple_io ? w->ple_io->io : NULL;
}

bool pulsar_qwen_s4_step_end(const pulsar_qwen_step *st, bool ok) {
    pulsar_qwen_ple_io *io = st->w->ple_io;
    if (io && io->pending) {                 /* a step that failed before layer 1 */
        (void)pulsar_engram_gather_wait(io->pending);
        io->pending = NULL;
    }
    if (!ok) return false;
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_MOE];
    uint32_t nf = 0;
    if (!sc || !pulsar_gpu_tensor_read(sc, moe_layout(st->shape, st->st->max_rows).nf, &nf, sizeof(nf)))
        return fail("could not read the MoE non-finite flag");
    if (nf) {
        fprintf(stderr, "pulsar: %s: the MoE output of layer %u went non-finite (code 0x%08x) -- refusing the step\n",
                PULSAR_QWEN_ARCH, nf & 0xffffffu, nf);
        return false;
    }
    return true;
}

bool pulsar_qwen_s4_ple(const pulsar_qwen_step *st, uint32_t il) {
    const uint32_t n = st->n_rows;
    pulsar_qwen_ple_io *io = st->w->ple_io;
    if (!io || !io->pending || io->pending_n != n) return fail("the PLE op found no gather for this step");
    int got;
    {
        PULSAR_NVTX("qwen: PLE gather wait");
        got = pulsar_engram_gather_wait(io->pending);
    }
    io->pending = NULL;
    if (!got) return fail("the PLE row gather failed (message above)");

    const ple_scratch p = ple_layout(st->st->max_rows);
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_PLE];
    if (!sc) return fail("no PLE scratch");
    /* the rows, and where each row sits: DECODE = one row per bank, PREFILL = its runs (one bank's chunk, or
     * L272 P1 S4: one run per bank of a verify) */
    const bool decode = st->mode == PULSAR_STEP_DECODE;
    const uint32_t runs = st->n_runs ? st->n_runs : 1u;
    const uint32_t n_seq = decode ? n : runs;
    int32_t *map = io->map;   /* pinned: the uploads below read it when the stream reaches them */
    int32_t *row_seq = map, *row_j = map + n, *seq_first = map + 2 * n, *seq_rows = map + 3 * n, *seq_bank = map + 4 * n;
    for (uint32_t q = 0; q < n_seq; q++) {
        const uint32_t f = decode ? q : (st->n_runs ? st->run_first[q] : 0u);
        const uint32_t e = decode ? q + 1u : (st->n_runs ? st->run_first[q + 1] : n);
        seq_first[q] = (int32_t)f;
        seq_rows[q] = (int32_t)(e - f);
        seq_bank[q] = st->bank[f];
        for (uint32_t r = f; r < e; r++) { row_seq[r] = (int32_t)q; row_j[r] = (int32_t)(r - f); }
    }
    /* L289: enqueued behind layer 0, not waited for -- the host goes on queueing the step */
    bool ok = pulsar_gpu_tensor_write_async(sc, p.emb, io->rows, (uint64_t)n * PULSAR_QWEN_HIDDEN * 2u) &&
              pulsar_gpu_tensor_write_async(sc, p.row_seq, row_seq, (uint64_t)n * 4u) &&
              pulsar_gpu_tensor_write_async(sc, p.row_j, row_j, (uint64_t)n * 4u) &&
              pulsar_gpu_tensor_write_async(sc, p.seq_first, seq_first, (uint64_t)n_seq * 4u) &&
              pulsar_gpu_tensor_write_async(sc, p.seq_rows, seq_rows, (uint64_t)n_seq * 4u) &&
              pulsar_gpu_tensor_write_async(sc, p.seq_bank, seq_bank, (uint64_t)n_seq * 4u);
    if (!ok) return fail("could not stage the PLE rows");

    const pulsar_qwen_layer_weights &L = layer_w(st, il);
    const int H = (int)st->shape->n_embd, HC = (int)pulsar_qwen_hc_dim(st->shape);
    pulsar_qwen_ple_dev w;
    w.norm_key = (const uint16_t *)wptr(st, L.ple_norm_key, "qwen PLE norm_key");
    w.norm_query = (const uint16_t *)wptr(st, L.ple_norm_query, "qwen PLE norm_query");
    w.norm_conv = (const uint16_t *)wptr(st, L.ple_norm_conv, "qwen PLE norm_conv");
    w.conv_w = (const uint16_t *)wptr(st, L.ple_conv, "qwen PLE conv1d");
    if (!w.norm_key || !w.norm_query || !w.norm_conv || !w.conv_w) return false;
    uint8_t *base = (uint8_t *)dptr(sc);
    /* key / value: row-independent Linears of the gathered rows, so they take the row-kind segments; the gate
     * and the dilated conv that follow read across a sequence's rows and stay one launch */
    const uint16_t *emb = (const uint16_t *)(base + p.emb);
    float *key = (float *)(base + p.key), *value = (float *)(base + p.value);
    if (!step_linear(st, L.ple_key, H, HC, "qwen PLE key_proj", emb, key, base + p.lin_ws, p.lin_ws_bytes) ||
        !step_linear(st, L.ple_value, H, H, "qwen PLE value_proj", emb, value, base + p.lin_ws, p.lin_ws_bytes))
        return fail("a PLE projection launch failed");
    /* every kernel of the launch reads a row's own sequence alone, so the prompt run behind a fused step's verify
     * shares it; the capture covers the verify rows [0, n_dec) only (L284 #2) */
    const pulsar_qwen_rows rows = {(const int32_t *)(base + p.row_seq), (const int32_t *)(base + p.row_j),
                                   (const int32_t *)(base + p.seq_first), (const int32_t *)(base + p.seq_rows),
                                   (const int32_t *)(base + p.seq_bank), (int)n_seq,
                                   st->verify ? (float *)pulsar_state_capture_rows(st->states, il, PULSAR_QWEN_STATE_PLE_CONV) : NULL,
                                   st->verify ? (int)st->n_dec : 0};
    return pulsar_qwen_ple_launch(&w, key, value, (uint16_t *)dptr(st->streams), (int)n,
                                  &rows, (float *)dptr(pulsar_state_tensor(st->states, il, PULSAR_QWEN_STATE_PLE_CONV)),
                                  base + p.ws, p.ws_bytes, 0) == 0;
}

bool pulsar_qwen_s4_gr_read(const pulsar_qwen_step *st, uint32_t il, pulsar_qwen_gr_side side) {
    const pulsar_qwen_layer_weights &L = layer_w(st, il);
    const pulsar_qwen_gr_weights &gw = side == PULSAR_QWEN_GR_ATTN ? L.gr_attn : L.gr_mlp;
    const gr_scratch g = gr_layout(st->st->max_rows);
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_GR_READ];
    if (!sc) return fail("no GR scratch");
    uint8_t *base = (uint8_t *)dptr(sc);
    /* L251 / ac69748f: the block input is bf16 and there is no E4M3 activation slot in this family, so the
     * read emits the bf16 row and nothing else -- no slot, no arming, no notes.  `nullptr` is the read's
     * documented "no slot" form, so the mxfp8 W_down weights are still read by the same arm.
     * L284 #2: every kernel of the read indexes its row t alone (the workspace's per-row planes from 0), so
     * a row-kind segment is the launch at its rows' offsets. */
    const uint16_t *streams = (const uint16_t *)dptr(st->streams);
    uint16_t *x = (uint16_t *)dptr(st->st->x);
    float *inj = (float *)(base + g.inj[side]);
    const uint64_t HC = pulsar_qwen_hc_dim(st->shape), H = st->shape->n_embd;
    return arm_segments(st, 0, st->n_rows, [&](bool prompt, uint32_t r0, uint32_t m) {
        pulsar_qwen_gr_dev w;
        return gr_dev(st, gw, prompt, &w) &&
               pulsar_qwen_gr_read_launch(&w, streams + r0 * HC, (int)m, x + r0 * H, inj + (size_t)r0 * PULSAR_QWEN_HC,
                                          base + g.ws, g.ws_bytes, 0) == 0;
    });
}

bool pulsar_qwen_s4_gr_write(const pulsar_qwen_step *st, uint32_t, pulsar_qwen_gr_side side) {
    const gr_scratch g = gr_layout(st->st->max_rows);
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_GR_READ];
    if (!sc) return fail("no GR scratch");
    return pulsar_qwen_gr_write_launch((uint16_t *)dptr(st->streams), (const float *)dptr(st->st->y),
                                       (const float *)((uint8_t *)dptr(sc) + g.inj[side]), (int)st->n_rows, 0) == 0;
}

/* L255: one Tessera stack's launch descriptor from its bound planes -- device pointers, the stack geometry
 * (in -> out, n_expert) and the two host integers of its geom plane. */
/// @param st the step
/// @param P the stack's planes
/// @param in input width
/// @param out output width
/// @param[out] o the descriptor
/// @return false when a plane does not resolve
static bool tessera_proj(const pulsar_qwen_step *st, pulsar_tensor *const P[PULSAR_QWEN_TESS_PLANES], int in,
                         int out, pulsar_tessera_proj *o) {
    const int32_t *geom = (const int32_t *)tensor_data(st->model, P[PULSAR_QWEN_TESS_GEOM]);
    o->E = (int)st->shape->n_expert;
    o->K = in;
    o->N = out;
    o->words = (const int32_t *)wptr(st, P[PULSAR_QWEN_TESS_WORDS], "qwen tessera words");
    o->words_stride = (long)P[PULSAR_QWEN_TESS_WORDS]->dim[0];
    o->table = (const uint16_t *)wptr(st, P[PULSAR_QWEN_TESS_TABLE], "qwen tessera table");
    o->init = (const int32_t *)wptr(st, P[PULSAR_QWEN_TESS_INIT], "qwen tessera init");
    o->has_init = (const int32_t *)wptr(st, P[PULSAR_QWEN_TESS_HAS_INIT], "qwen tessera has_init");
    o->wscale = (const float *)wptr(st, P[PULSAR_QWEN_TESS_WSCALE], "qwen tessera wscale");
    o->runs = (const int32_t *)wptr(st, P[PULSAR_QWEN_TESS_RUNS], "qwen tessera runs");
    o->bdesc = (const int32_t *)wptr(st, P[PULSAR_QWEN_TESS_BDESC], "qwen tessera bdesc");
    o->tile_words = geom ? geom[0] : 0;
    o->slot_words = geom ? geom[1] : 0;
    if (!geom) return fail("no host view of a Tessera geom plane");
    return o->words && o->table && o->init && o->has_init && o->wscale && o->runs && o->bdesc;
}

bool pulsar_qwen_s4_moe(const pulsar_qwen_step *st, uint32_t il) {
    const uint32_t n = st->n_rows;
    const pulsar_qwen_shape *s = st->shape;
    const int H = (int)s->n_embd, SMID = (int)s->n_ff_shexp;
    const pulsar_qwen_layer_weights &L = layer_w(st, il);
    /* L251 / ac69748f: there is no E4M3 activation slot in this family.  The block reads its input's bf16 row
     * directly -- the router, the routed arm (by ids_src1) and the shared expert as a plain row. */
    const uint16_t *x = (const uint16_t *)dptr(st->st->x);
    float *y = (float *)dptr(st->st->y);
    const moe_scratch m = moe_layout(st->shape, st->st->max_rows);
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_MOE];
    if (!sc) return fail("no MoE scratch");
    uint8_t *base = (uint8_t *)dptr(sc);
    const pulsar_moe_block_shape bs = moe_block_shape(s);
    pulsar_moe_block_parts parts;
    if (pulsar_moe_block_carve(&bs, (int)n, base + m.ws, m.ws_bytes, &parts) != 0) return false;
    /* the shared expert, sigmoid-gated, added after the routed sum -- under expert parallelism rank 0's alone.
     * Keyed on its FORMAT (task #44): three EXL3 dense slices ride the EXL3 routed launches (mmq/exl3_moe.cuh);
     * any other format, or a Tessera routed arm, runs its own launches after the routed sum.  L288 exl3-dense: a
     * riding one is FORKED here, ahead of the router, per row-kind segment over that segment's own rows of its
     * buffers (the dense workspace is its stream's alone), and each segment's routed launch joins its fork. */
    const bool shared_here = !(pulsar_qwen_tp(s) > 1 && s->tp_rank != 0);
    pulsar_rows_linear sg0{}, su0{}, sd0{};
    if (shared_here && !(linear_dev(st, L.sh_gate, H, SMID, false, "qwen shared gate_proj", &sg0) &&
                         linear_dev(st, L.sh_up, H, SMID, false, "qwen shared up_proj", &su0) &&
                         linear_dev(st, L.sh_down, SMID, H, false, "qwen shared down_proj", &sd0)))
        return false;
    const bool tess_layer = L.moe_tess[PULSAR_QWEN_TESS_GATE][PULSAR_QWEN_TESS_WORDS] != NULL;
    const bool ride = shared_here && !tess_layer && sg0.k2 && su0.k2 && sd0.k2;
    pulsar_rows_moe_shared shs[2];   /* arm_segments: at most a decode and a prompt segment */
    int n_sh = 0;
    if (ride && !arm_segments(st, 0, n, [&](bool prompt, uint32_t r0, uint32_t rows) {
            pulsar_rows_moe_shared &sh = shs[n_sh++];
            sh = pulsar_rows_moe_shared{};
            sh.gate = sg0; sh.up = su0; sh.down = sd0;
            sh.gate.prompt = sh.up.prompt = sh.down.prompt = prompt;
            sh.scale = parts.gammas + r0;   /* the router writes it, on this stream, before the sum reads it */
            sh.yg = parts.yg + (size_t)r0 * SMID;
            sh.yu = parts.yu + (size_t)r0 * SMID;
            sh.ys = parts.ys + (size_t)r0 * H;
            sh.h = parts.h + (size_t)r0 * SMID;
            sh.ws = parts.lin; sh.ws_bytes = parts.lin_bytes;
            return pulsar_rows_moe_shared_fork(&sh, x + (size_t)r0 * H, (int)rows, 0) == 0;
        }))
        return fail("the shared expert's fork was refused");
    /* the router (L295 F2: the shared router, this family's cfg): softmax top-k over the experts and the shared
     * expert's sigmoid gate, whose weight row (shared_expert_gate) follows the router's */
    const uint16_t *router_w = (const uint16_t *)wptr(st, L.moe_router, "qwen router");
    const uint16_t *shared_gate_w = (const uint16_t *)wptr(st, L.sh_gate_scalar, "qwen shared_expert_gate");
    const pulsar_router_cfg rcfg = pulsar_qwen_router_cfg((int)s->n_expert, (int)s->n_expert_used);
    const pulsar_router_layer rlayer{};   /* no selection bias, no table, no global scale */
    if (!router_w || !shared_gate_w ||
        pulsar_router_logits_launch(x, (int)n, H, router_w, rcfg.n_expert, shared_gate_w, rcfg.n_shared, parts.logits,
                                    0) != 0 ||
        pulsar_router_select_launch(&rcfg, &rlayer, parts.logits, nullptr, (int)n, parts.sel, parts.wts, parts.gammas,
                                    nullptr, 0) != 0)
        return false;
    /* the routed experts through the core's front door (moe.cpp): the trunk's gate_up is ONE fused stack, the MTP
     * layer's gate and up are two; under TP the rank reads its plan's range of whole experts.  L284 #2: the router
     * above has no arm (one kernel at every row count, each row's logits / picks its own) and stays whole; the
     * routed part and the shared expert take the row-kind segments -- a row's output reads its own x and picks
     * alone (the grouped GEMMs, the fold and the sum index each (row, pick)), the workspace is reused from 0 */
    const uint32_t K = s->n_expert_used;
    pulsar_moe_rows_call rc{};
    rc.m = st->model;
    rc.gate = L.moe_gate_up ? L.moe_gate_up : L.moe_gate;
    rc.up = L.moe_gate_up ? NULL : L.moe_up;
    rc.down = L.moe_down;
    rc.top_k = K;
    rc.ws = parts.routed;
    rc.ws_bytes = parts.routed_bytes;
    rc.nf_flag = (uint32_t *)(base + m.nf);
    rc.nf_code = 0x51000000u | il;
    pulsar_tessera_proj tp[PULSAR_QWEN_TESS_PROJS];
    if (L.moe_tess[PULSAR_QWEN_TESS_GATE][PULSAR_QWEN_TESS_WORDS]) {
        /* L255: the layer's experts are Tessera planes; the front door reads them through tp, built once here and
         * read by every segment below.  The arm holds every expert on one GPU -- no expert range yet, so not under
         * tensor parallelism -- and the imatrix taps read the EXL3 stacks (rc.gate / rc.down), so a collection run
         * is refused on a Tessera layer rather than tapping a NULL stack. */
        const int MID = (int)s->n_ff_exp;
        if (pulsar_qwen_tp(s) > 1) return fail("Tessera routed experts under tensor parallelism (the arm has no expert range)");
        if (st->tap) return fail("imatrix collection over Tessera routed experts (the taps read EXL3 stacks)");
        if (!tessera_proj(st, L.moe_tess[PULSAR_QWEN_TESS_GATE], H, MID, &tp[PULSAR_QWEN_TESS_GATE]) ||
            !tessera_proj(st, L.moe_tess[PULSAR_QWEN_TESS_UP], H, MID, &tp[PULSAR_QWEN_TESS_UP]) ||
            !tessera_proj(st, L.moe_tess[PULSAR_QWEN_TESS_DOWN], MID, H, &tp[PULSAR_QWEN_TESS_DOWN]))
            return false;
        rc.tessera_gate = &tp[PULSAR_QWEN_TESS_GATE];
        rc.tessera_up = &tp[PULSAR_QWEN_TESS_UP];
        rc.tessera_down = &tp[PULSAR_QWEN_TESS_DOWN];
    }
    int seg = 0;
    if (!arm_segments(st, 0, n, [&](bool prompt, uint32_t r0, uint32_t rows) {
            rc.shared = ride ? &shs[seg++] : NULL;   /* the fork above, joined before this segment's sum */
            rc.n_shared = ride ? 1 : 0;
            rc.selected = parts.sel + (size_t)r0 * K;
            rc.weights = parts.wts + (size_t)r0 * K;
            rc.x_bf16 = x + (size_t)r0 * H;
            rc.n_rows = (int)rows;
            rc.out = y + (size_t)r0 * H;
            rc.prompt = prompt;
            if (st->tap) pulsar_imatrix_note_routed_in(st->tap, rc.gate, rc.up, rc.x_bf16, rc.selected, rows, K, s->n_expert, H);
            if (!pulsar_moe_routed_rows(&rc)) return false;
            if (st->tap)
                pulsar_imatrix_note_routed_mid(st->tap, rc.down, pulsar_rows_moe_routed_mid(rc.ws, rc.ws_bytes, (int)rows, (int)K,
                                                                                      (int)s->n_ff_exp, H),
                                               rc.selected, rows, K, s->n_expert, (uint32_t)rc.down->dim[0]);
            if (ride && st->tap) {
                /* the shared expert's down read the SwiGLU rows its fork left in its rows of parts.h (bf16) */
                pulsar_imatrix_note_dense(st->tap, L.sh_gate, rc.x_bf16, rows, (uint32_t)H);
                pulsar_imatrix_note_dense(st->tap, L.sh_up, rc.x_bf16, rows, (uint32_t)H);
                pulsar_imatrix_note_dense(st->tap, L.sh_down, parts.h + (size_t)r0 * SMID, rows, (uint32_t)SMID);
            }
            return true;
        }))
        return false;
    if (!shared_here || ride) return true;
    return arm_segments(st, 0, n, [&](bool prompt, uint32_t r0, uint32_t rows) {
        pulsar_rows_moe_shared q{};   /* the gate row-indexed, the shared expert's buffers reused from 0 */
        q.scale = parts.gammas + r0;
        q.yg = parts.yg;
        q.yu = parts.yu;
        q.ys = parts.ys;
        q.h = parts.h;
        q.ws = parts.lin;
        q.ws_bytes = parts.lin_bytes;
        if (!(linear_dev(st, L.sh_gate, H, SMID, prompt, "qwen shared gate_proj", &q.gate) &&
              linear_dev(st, L.sh_up, H, SMID, prompt, "qwen shared up_proj", &q.up) &&
              linear_dev(st, L.sh_down, SMID, H, prompt, "qwen shared down_proj", &q.down) &&
              pulsar_rows_moe_shared_launch(&q, x + (size_t)r0 * H, (int)rows, y + (size_t)r0 * H, 0) == 0))
            return false;
        if (st->tap) {
            /* the shared expert's down reads the SwiGLU rows the launch left in q.h (bf16) */
            pulsar_imatrix_note_dense(st->tap, L.sh_gate, x + (size_t)r0 * H, rows, (uint32_t)H);
            pulsar_imatrix_note_dense(st->tap, L.sh_up, x + (size_t)r0 * H, rows, (uint32_t)H);
            pulsar_imatrix_note_dense(st->tap, L.sh_down, q.h, rows, (uint32_t)SMID);
        }
        return true;
    });
}

/* L251 MTP: the draft head (pulsar_qwen_weights::draft_head_mx), gathered once from the bf16 head over the ids the
 * load chose (L288: qwen_draft_vocab, family_qwen.cpp). */
/// @param st the step (its model and bf16 lm_head)
/// @param[in,out] wm draft_head_mx and draft_ids_dev are made
/// @return false (printed) without draft ids or on a failed build
static bool build_draft_head(const pulsar_qwen_step *st, pulsar_qwen_weights *wm) {
    const pulsar_qwen_shape *s = st->shape;
    const int H = (int)s->n_embd;
    const uint32_t nd = wm->n_draft;
    if (!wm->draft_ids || nd == 0 || nd >= s->n_vocab) return fail("the MTP draft vocabulary was not chosen at load");
    const uint16_t *hb = (const uint16_t *)wptr(st, st->w->output, "qwen lm_head");
    const uint64_t bytes = pulsar_qwen_mxfp8_bytes((int)nd, H);
    /* the device id list lives as long as the head (the gather kernel may still be reading it) */
    wm->draft_ids_dev = pulsar_gpu_tensor_alloc((uint64_t)nd * sizeof(int32_t));
    wm->draft_head_mx = hb && bytes ? pulsar_gpu_tensor_alloc(bytes) : NULL;
    const bool ok = wm->draft_ids_dev && wm->draft_head_mx &&
                    pulsar_gpu_tensor_write(wm->draft_ids_dev, 0, wm->draft_ids, (uint64_t)nd * sizeof(int32_t)) &&
                    pulsar_qwen_bf16_to_mxfp8(hb, (const int32_t *)dptr(wm->draft_ids_dev), (int)nd, H,
                                              dptr(wm->draft_head_mx), 0) == 0;
    if (!ok) {
        pulsar_gpu_tensor_free(wm->draft_head_mx);
        pulsar_gpu_tensor_free(wm->draft_ids_dev);
        wm->draft_head_mx = NULL;
        wm->draft_ids_dev = NULL;
        return fail("the MTP draft head could not be built");
    }
    fprintf(stderr, "pulsar: %s: MTP draft head: %u rows (%.0f MB MXFP8)\n", PULSAR_QWEN_ARCH, nd, (double)bytes / 1e6);
    return true;
}

bool pulsar_qwen_s4_head(const pulsar_qwen_step *st, uint32_t row0, uint32_t n, uint32_t out0) {
    const pulsar_qwen_shape *s = st->shape;
    const int H = (int)s->n_embd;
    if (n == 0 || out0 + n > PULSAR_QWEN_HEAD_ROWS_MAX || row0 + n > st->n_rows) return fail("head rows out of range");
    const head_scratch h = head_layout(s);
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_HEAD];
    if (!sc) return fail("no head scratch");
    uint8_t *base = (uint8_t *)dptr(sc);
    /* the mixer's bf16 row IS the head GEMM's bf16 activation plane: written
     * straight into the slot of a key tensor, noted, read by the bf16 GEMM */
    pulsar_gpu_tensor *xkey = scratch_view(st, PULSAR_QWEN_OP_HEAD, h.xkey, (uint64_t)n * H * sizeof(float));
    void *xb = NULL;
    bool ok = xkey && pulsar_gpu_bf16_act_slot(xkey, n, (uint64_t)H, &xb);
    /* the mixer's read over rows [row0, row0 + n), each row-kind segment through its arm, into xb's row r - row0 */
    const uint16_t *streams = (const uint16_t *)dptr(st->streams);
    const uint64_t HC = pulsar_qwen_hc_dim(s);
    ok = ok && arm_segments(st, row0, row0 + n, [&](bool prompt, uint32_t r0, uint32_t m) {
        pulsar_qwen_gr_dev w;
        return gr_dev(st, *st->mixer, prompt, &w) &&
               pulsar_qwen_gr_read_launch(&w, streams + r0 * HC, (int)m, (uint16_t *)xb + (size_t)(r0 - row0) * H, NULL,
                                          base + h.ws, h.ws_bytes, 0) == 0;
    });
    if (ok) {
        /* The head is a dense Linear of the mixer's bf16 rows, through the core's bf16-rows front door
         * (pulsar_rows_linear_launch: the arm by the weight's format, the decode arm at every call -- a row's logits
         * never depend on the rows beside it, a prompt's last row included; L288 C1: a wide verify's span in 16-row
         * blocks).  L288 qwen-exl3-head: the weight is the
         * container's -- an lm_head / mtp.draft_head stored EXL3 runs the EXL3 dense arm as stored; a bf16 lm_head
         * is L251's: MXFP8 made once on the device (the producers' own encoder), multiplied by the mxfp8_lt rows
         * arm on the tensor cores, the weight read once for all the call's rows. */
        pulsar_qwen_weights *wm = const_cast<pulsar_qwen_weights *>(st->w);
        const bool dh = st->draft_head;
        const pulsar_tensor *stored = dh ? wm->mtp.draft_head : wm->output->type != PULSAR_TENSOR_BF16 ? wm->output : NULL;
        /* L266 step 7: under TP a rank's head is its vocab range [rank * V / tp, +V / tp) -- the rows it
         * computes, the gather below fills the rest; the draft head stays whole on every rank */
        const uint32_t tp = pulsar_qwen_tp(s);
        uint32_t v_lo = 0, v_hi = s->n_vocab;
        if (tp > 1 && !pulsar_tp_owned_range((int)s->tp_rank, tp, s->n_vocab, &v_lo, &v_hi)) ok = false;
        const uint32_t Vl = v_hi - v_lo;
        if (ok && !dh && !stored && !wm->head_mx) {
            const uint16_t *hb = (const uint16_t *)wptr(st, st->w->output, "qwen lm_head");
            if (hb) hb += (uint64_t)v_lo * H;
            const uint64_t bytes = pulsar_qwen_mxfp8_bytes((int)Vl, H);
            wm->head_mx = hb && bytes ? pulsar_gpu_tensor_alloc(bytes) : NULL;
            ok = wm->head_mx && pulsar_qwen_bf16_to_mxfp8(hb, NULL, (int)Vl, H, dptr(wm->head_mx), 0) == 0;
            if (ok) fprintf(stderr, "pulsar: %s: lm_head as MXFP8 (%.2f GB, from bf16 %.2f GB)\n", PULSAR_QWEN_ARCH,
                            (double)bytes / 1e9, (double)s->n_vocab * H * 2 / 1e9);
            else if (wm->head_mx) { pulsar_gpu_tensor_free(wm->head_mx); wm->head_mx = NULL; }
        }
        if (ok && dh && !stored && !wm->draft_head_mx) ok = build_draft_head(st, wm);
        /* logits rows [out0, out0 + n) of the slab (the draft head's rows are n_draft wide) */
        const uint64_t row_bytes = (uint64_t)(dh ? wm->n_draft : s->n_vocab) * sizeof(float);
        pulsar_gpu_tensor *dst = ok ? pulsar_gpu_tensor_view(st->st->logits, out0 * row_bytes, n * row_bytes) : NULL;
        ok = ok && dst;
        if (ok) {
            const uint32_t rows = dh ? wm->n_draft : Vl;
            pulsar_rows_linear l{};
            if (stored) {
                ok = linear_dev(st, stored, H, (int)rows, false, dh ? "qwen MTP draft head" : "qwen lm_head", &l);
            } else {
                const uint8_t *hq = (const uint8_t *)dptr(dh ? wm->draft_head_mx : wm->head_mx);
                l = pulsar_rows_linear{hq, hq + (uint64_t)rows * H, 0, H, (int)rows, false};
            }
            /* this rank's rows of the head into `dst` -- the whole head on one GPU, the vocab range the engine's
             * gather asks for under TP (the rank's own; the gather assembles the rest) */
            auto head = [&](uint32_t lo, uint32_t width, pulsar_gpu_tensor *dst) {
                if (!dh && tp > 1 && (lo != v_lo || width != Vl)) return false;
                return pulsar_rows_linear_launch(&l, (const uint16_t *)xb, (int)n, (float *)dptr(dst), base + h.lin_ws,
                                                 h.lin_ws_bytes, 0) == 0;
            };
            if (ok && (tp == 1 || dh)) {
                ok = head(0, rows, dst);
            } else if (ok) {
                pulsar_qwen_state *q = st->st;
                const pulsar_tp_vocab x = {q->tp, q->tp_slab_dev, q->tp_ticket, q->tp_vocab_own, &q->tp_vocab_seq,
                                           s->n_vocab, PULSAR_QWEN_HEAD_ROWS_MAX};
                ok = pulsar_tp_vocab_gather(&x, n, head, dst);
            }
        }
        pulsar_gpu_tensor_free(dst);
    }
    if (xkey) {
        pulsar_gpu_act_slot_drop(xkey);
        pulsar_gpu_tensor_free(xkey);
    }
    return ok || fail("the head (mixer + lm_head) failed");
}

/* L251 MTP: the input combine -- the MTP layer's starting streams from the trunk stacks `h` (device,
 * one per row) and the rows' NEXT tokens (st->tokens).  It plays embed's part for the MTP step: the
 * token check, the MoE non-finite flag cleared, the ids on the device. */
uint64_t pulsar_qwen_s4_mtp_combine_ws_bytes(void) { return pulsar_qwen_mtp_combine_workspace_bytes(); }

bool pulsar_qwen_s4_mtp_combine(const pulsar_qwen_step *st, const void *h) {
    const pulsar_qwen_shape *s = st->shape;
    const pulsar_qwen_mtp_weights &M = st->w->mtp;
    const uint32_t n = st->n_rows;
    if (!M.present || !st->st->mtp) return fail("the MTP combine without the MTP layer");
    for (uint32_t r = 0; r < n; r++)
        if (st->tokens[r] < 0 || (uint32_t)st->tokens[r] >= s->n_vocab) return fail("a token id outside the vocabulary");
    pulsar_gpu_tensor *moe_sc = st->st->scratch[PULSAR_QWEN_OP_MOE];
    static const uint32_t zero = 0;   /* L289: an enqueued write reads it later */
    if (!moe_sc || !pulsar_gpu_tensor_write_async(moe_sc, moe_layout(st->shape, st->st->max_rows).nf, &zero, sizeof(zero)))
        return fail("could not clear the MoE non-finite flag");
    if (!pulsar_gpu_tensor_write_async(st->st->mtp_tok, 0, st->tokens, (uint64_t)n * sizeof(int32_t)))
        return fail("could not stage the MTP rows' tokens");
    const int H = (int)s->n_embd;
    pulsar_qwen_mtp_dev d{};
    d.embd = (const uint16_t *)wptr(st, st->w->token_embd, "qwen embed_tokens");
    d.n_vocab = (int)s->n_vocab;
    d.norm_embd = (const uint16_t *)wptr(st, M.norm_embd, "qwen MTP pre_fc_norm_embedding");
    d.norm_hidden = (const uint16_t *)wptr(st, M.norm_hidden, "qwen MTP pre_fc_norm_hidden");
    const uint8_t *fe = (const uint8_t *)wptr(st, M.fc_embd, "qwen MTP fc_embedding");
    const uint8_t *fh = (const uint8_t *)wptr(st, M.fc_hidden, "qwen MTP fc_hidden");
    if (!d.embd || !d.norm_embd || !d.norm_hidden || !fe || !fh) return false;
    d.fc_embd = {fe, fe + (uint64_t)H * H, H, H};      /* mxfp8_lt: E4M3 [out][in] then the E8M0 plane */
    d.fc_hidden = {fh, fh + (uint64_t)H * H, H, H};
    return pulsar_qwen_mtp_combine_launch(&d, (const uint16_t *)h, (const int32_t *)dptr(st->st->mtp_tok), (int)n,
                                          (uint16_t *)dptr(st->streams), dptr(st->st->mtp_ws),
                                          pulsar_gpu_tensor_bytes(st->st->mtp_ws), 0) == 0 ||
           fail("the MTP combine failed");
}

uint64_t pulsar_qwen_s4_draft_top_ws_bytes(void) { return pulsar_qwen_draft_top_ws_bytes(); }

static_assert(PULSAR_QWEN_HEAD_ROWS_MAX <= PULSAR_QWEN_DRAFT_TOP_ROWS_MAX, "every headed MTP row can be picked");

bool pulsar_qwen_s4_draft_top(const pulsar_qwen_step *st, uint32_t n) {
    if (!st->draft_head || !st->st->mtp_top || !st->w->n_draft) return fail("the draft pick outside a drafting MTP step");
    return pulsar_qwen_draft_top_launch((const float *)dptr(st->st->logits), (int)st->w->n_draft, (int)n,
                                        dptr(st->st->mtp_top), 0) == 0 ||
           fail("the draft pick launch failed");
}

bool pulsar_qwen_s4_draft_top_read(const pulsar_qwen_state *q, const pulsar_qwen_weights *w, uint32_t n, int32_t *id,
                                   float *conf) {
    pulsar_qwen_draft_top rec[PULSAR_QWEN_HEAD_ROWS_MAX];
    if (n == 0 || n > PULSAR_QWEN_HEAD_ROWS_MAX) return fail("a draft pick of no rows or past the head's rows");
    if (!q->mtp_top || !pulsar_gpu_tensor_read(q->mtp_top, 0, rec, n * sizeof rec[0]))
        return fail("could not read the draft pick");
    for (uint32_t r = 0; r < n; r++) {
        if (rec[r].nan) return fail("the draft head's logits hold a NaN");
        if (rec[r].index < 0 || (uint32_t)rec[r].index >= w->n_draft) return fail("the draft pick is outside the draft head");
        id[r] = w->draft_ids[rec[r].index];
        conf[r] = rec[r].conf;
    }
    return true;
}

/* L295 F3: the n-gram context's verify commit (the declared state's REPLAY, family_qwen.cpp): a bank's context before
 * the step, advanced over the kept tokens as the embed advanced it over every row. */
bool pulsar_qwen_s4_ngram_replay(const pulsar_qwen_weights *w, int32_t ctx[2], const int32_t *tokens, uint32_t keep) {
    pulsar_qwen_ple_io *io = w->ple_io;
    if (!io) return fail("no PLE row file is open");
    if (keep > PULSAR_QWEN_SPEC_ROWS) return fail("an n-gram replay past the verify width");
    pulsar_qwen_ngram_ctx c = {{ctx[0], ctx[1]}};
    uint64_t ids[PULSAR_QWEN_SPEC_ROWS * PULSAR_QWEN_NGRAM_COLS];
    pulsar_qwen_ngram_rows(&io->layout, &c, tokens, (int)keep, ids);
    ctx[0] = c.prev[0];
    ctx[1] = c.prev[1];
    return true;
}

/* ======================================================================== */
/* L251 S2's op: the Gated DeltaNet block.  On the integration branch it lives
 * in this TU because it needs the weight / scratch helpers above (wptr,
 * step_linear, dptr, fail, gdn_layout); split into family_qwen_s2.cpp when S2
 * rebases on the family interface.  The kernel is src/cuda/pulsar_cuda_gdn.cu.
 *
 * x (bf16 + its armed E4M3 slot) -> in_proj_qkv / _z / _a / _b through the EXL3
 * dense arm -> pulsar_gdn_forward (the recurrence; it EMITS the A8 slot) ->
 * out_proj -> y f32.  The four kernel weights are bf16, the container's
 * storage; the kernel widens them (rule 3). */
bool pulsar_qwen_s2_gdn(const pulsar_qwen_step *st, uint32_t il) {
    const pulsar_qwen_shape *s = st->shape;
    const uint32_t n = st->n_rows;
    const int H = (int)s->n_embd, CD = (int)pulsar_qwen_gdn_conv_dim(s);
    /* NV: this rank's V heads; NVG: the model's (in_proj_a / _b stay whole -- 48 rows of MXFP8, not a 128-row
     * slice -- and the rank reads its own NV columns of them, its A_log and dt_bias by the same offset) */
    const int VT = (int)pulsar_qwen_gdn_v_total(s), NV = (int)s->gdn_n_v_head;
    const int NVG = NV * (int)pulsar_qwen_tp(s), hv0 = (int)s->tp_rank * NV;
    const pulsar_qwen_layer_weights &L = layer_w(st, il);
    const pulsar_tensor *gw4[4] = {L.gdn_conv, L.gdn_a_log, L.gdn_dt_bias, L.gdn_norm};
    for (int i = 0; i < 4; i++)
        if (!admit(gw4[i], gw4[i]->type == PULSAR_TENSOR_BF16, "bf16 (the GDN kernel's weight dtype)")) return false;
    /* L251 / ac69748f: the block input is bf16 and there is no E4M3 activation slot in this family, so
     * these projections read the bf16 row its producer emitted (rule 3). */
    const uint16_t *xin = (const uint16_t *)dptr(st->st->x);
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_GDN];
    if (!sc) return fail("no GDN scratch");
    uint8_t *base = (uint8_t *)dptr(sc);
    const gdn_scratch g = gdn_layout(s, st->st->max_rows);
    void *linws = base + g.lin_ws;
    {
        const pulsar_tensor *const t[4] = {L.gdn_in_qkv, L.gdn_in_z, L.gdn_in_a, L.gdn_in_b};
        const int outs[4] = {CD, VT, NVG, NVG};
        const char *const what[4] = {"qwen GDN in_proj_qkv", "qwen GDN in_proj_z", "qwen GDN in_proj_a", "qwen GDN in_proj_b"};
        float *const y[4] = {(float *)(base + g.qkv), (float *)(base + g.z), (float *)(base + g.a), (float *)(base + g.b)};
        if (!step_linear_multi(st, 4, t, H, outs, what, xin, y, linws, g.lin_ws_bytes))
            return fail("a GDN projection launch failed");
    }
    pulsar_gdn_weights gw;
    gw.conv_w  = (const uint16_t *)wptr(st, L.gdn_conv,    "qwen GDN conv1d");
    gw.A_log   = (const uint16_t *)wptr(st, L.gdn_a_log,   "qwen GDN A_log");
    gw.dt_bias = (const uint16_t *)wptr(st, L.gdn_dt_bias, "qwen GDN dt_bias");
    if (gw.A_log) gw.A_log += hv0;
    if (gw.dt_bias) gw.dt_bias += hv0;
    gw.norm_w  = (const uint16_t *)wptr(st, L.gdn_norm,    "qwen GDN norm");
    if (!gw.conv_w || !gw.A_log || !gw.dt_bias || !gw.norm_w) return false;
    /* the out_proj over every row of the step (its row-kind segments) */
    auto gdn_out = [&]() {
        return step_linear(st, L.gdn_out, VT, H, "qwen GDN out_proj", (const uint16_t *)(base + g.obf16),
                           (float *)dptr(st->st->y), linws, g.lin_ws_bytes) ||
               fail("the GDN out_proj launch failed");
    };
    pulsar_gdn_call c{};
    c.row_slot = (const int32_t *)dptr(st->st->row_bank);
    c.conv_state = (float *)dptr(pulsar_state_tensor(st->states, il, PULSAR_QWEN_STATE_GDN_CONV));
    c.rec_state  = (float *)dptr(pulsar_state_tensor(st->states, il, PULSAR_QWEN_STATE_GDN_REC));
    c.qkv = (const float *)(base + g.qkv); c.ld_qkv = CD;
    c.z   = (const float *)(base + g.z);   c.ld_z   = VT;
    c.a   = (const float *)(base + g.a) + hv0;   c.ld_a = NVG;
    c.b   = (const float *)(base + g.b) + hv0;   c.ld_b = NVG;
    c.tp_ranks = (int)pulsar_qwen_tp(s);
    c.scratch = base + g.ws; c.scratch_bytes = g.ws_bytes;
    c.out_f32 = NULL;                               /* the bf16 row is the consumer's input, not f32 */
    c.out_bf16 = base + g.obf16;                    /* L251 / ac69748f: no E4M3 slot in this family */
    if (st->mode == PULSAR_STEP_DECODE) {      /* one row per bank */
        c.n_seq = (int)n;
        c.seq_rows = 1;
        return (pulsar_gdn_forward(&gw, &c, 0) == 0 || fail("pulsar_gdn_forward failed")) && gdn_out();
    }
    /* PREFILL: the verify's rows [0, n_dec) -- one sequence, or (L272 P1 S4) one ragged run a bank, each row's
     * state captured for the rollback -- then (L284 #2) the prompt runs after them, uncaptured */
    const uint32_t nvr = pulsar_step_verify_runs(st);
    if (st->n_dec > 0) {
        pulsar_gdn_call v = c;
        if (nvr > 1) {
            uint32_t longest = 0;
            for (uint32_t k = 0; k < nvr; k++)
                longest = st->run_first[k + 1] - st->run_first[k] > longest ? st->run_first[k + 1] - st->run_first[k] : longest;
            v.n_seq = (int)nvr;
            v.seq_rows = (int)longest;
            v.seq_first = (const int32_t *)dptr(st->st->run_first_dev);
            v.n_rows = (int)st->n_dec;
        } else {
            v.n_seq = 1;
            v.seq_rows = (int)st->n_dec;
        }
        if (st->verify) {   /* L251 MTP: the per-row conv states a rollback copies back; L288: the recurrent
                             * state is left for the commit, which applies the kept rows' updates */
            /* row r of the step at row r of the layer's capture (state_kinds.h) */
            v.conv_rows = (float *)pulsar_state_capture_rows(st->states, il, PULSAR_QWEN_STATE_GDN_CONV);
            v.rec_upd   = (float *)pulsar_state_capture_rows(st->states, il, PULSAR_QWEN_STATE_GDN_REC);
        }
        if (pulsar_gdn_forward(&gw, &v, 0) != 0) return fail("pulsar_gdn_forward failed (the verify rows)");
    }
    /* each prompt run its own n_seq = 1 call at its row offset: the call a classic chunk of its rows makes (a
     * ragged call over several runs would chunk them by the longest, not as each alone) */
    for (uint32_t k = nvr; k < st->n_runs; k++) {
        const uint32_t r0 = st->run_first[k], m = st->run_first[k + 1] - r0;
        pulsar_gdn_call p = c;
        p.n_seq = 1;
        p.seq_rows = (int)m;
        p.row_slot += r0;
        p.qkv += (size_t)r0 * CD;
        p.z += (size_t)r0 * VT;
        p.a += (size_t)r0 * NVG;
        p.b += (size_t)r0 * NVG;
        p.out_bf16 = (uint16_t *)p.out_bf16 + (size_t)r0 * VT;
        if (pulsar_gdn_forward(&gw, &p, 0) != 0) return fail("pulsar_gdn_forward failed (a prompt run)");
    }
    return gdn_out();
}

/* ======================================================================== */
/* L251 S3's op: the Qwen full-attention + QSA layer.  On the integration branch
 * it lives in this TU for the same reason the gdn op does; split into
 * family_qwen_s3.cpp when S3 rebases.  The kernel is src/cuda/pulsar_cuda_qsa.cu.
 *
 * x (bf16 + its armed E4M3 slot) -> q_proj / k_proj / v_proj / index_qk_proj
 * through the EXL3 dense arm -> pulsar_gpu_qsa_forward (appends K/V + the pooled
 * block keys to the bank's caches and EMITS the A8 slot) -> o_proj -> y f32.
 * The four norms are bf16 (the recipe's format; the kernel widens). */
bool pulsar_qwen_s3_qsa(const pulsar_qwen_step *st, uint32_t il) {
    const pulsar_qwen_shape *s = st->shape;
    const uint32_t n = st->n_rows;
    const int H = (int)s->n_embd;
    const pulsar_qwen_layer_weights &L = layer_w(st, il);
    const pulsar_tensor *nrm[4] = {L.attn_q_norm, L.attn_k_norm, L.idx_q_norm, L.idx_k_norm};
    for (int i = 0; i < 4; i++)
        if (!admit(nrm[i], nrm[i]->type == PULSAR_TENSOR_BF16, "bf16 (the QSA kernel's norm dtype)")) return false;
    /* L251 / ac69748f: bf16, as above -- the QSA projections read the block input directly. */
    const uint16_t *xin = (const uint16_t *)dptr(st->st->x);
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_QSA];
    if (!sc) return fail("no QSA scratch");
    const qsa_scratch g = qsa_layout(s, st->st->max_rows, st->states->ctx);
    void *linws = (uint8_t *)dptr(sc) + g.lin_ws;

    /* pulsar_qsa_io wants tensors for the four projections; the scratch IS a
     * tensor, so hand it views (and free them on every exit). */
    pulsar_gpu_tensor *vq = scratch_view(st, PULSAR_QWEN_OP_QSA, g.qg, (uint64_t)n * pulsar_qwen_qsa_q_in(s) * 4);
    pulsar_gpu_tensor *vk = scratch_view(st, PULSAR_QWEN_OP_QSA, g.k, (uint64_t)n * pulsar_qwen_qsa_kv_in(s) * 4);
    pulsar_gpu_tensor *vv = scratch_view(st, PULSAR_QWEN_OP_QSA, g.v, (uint64_t)n * pulsar_qwen_qsa_kv_in(s) * 4);
    pulsar_gpu_tensor *vi = scratch_view(st, PULSAR_QWEN_OP_QSA, g.idx, (uint64_t)n * PULSAR_QSA_IDX_IN * 4);
    pulsar_gpu_tensor *views[64 + 4];
    int nv = 0;
    if (vq) views[nv++] = vq;
    if (vk) views[nv++] = vk;
    if (vv) views[nv++] = vv;
    if (vi) views[nv++] = vi;
    auto drop = [&]() { for (int i = 0; i < nv; i++) pulsar_gpu_tensor_free(views[i]); };

    const uint32_t nb = st->states->n_banks;
    if (nv != 4 || nb == 0 || nb > 64) { drop(); return fail("the QSA op needs 4 projection views and 1..64 banks"); }
    const int QI = (int)pulsar_qwen_qsa_q_in(s), KVI = (int)pulsar_qwen_qsa_kv_in(s);
    const pulsar_tensor *const qt[4] = {L.attn_q, L.attn_k, L.attn_v, L.idx_qk};
    const int qouts[4] = {QI, KVI, KVI, PULSAR_QSA_IDX_IN};
    const char *const qwhat[4] = {"qwen QSA q_proj", "qwen QSA k_proj", "qwen QSA v_proj", "qwen QSA index_qk_proj"};
    float *const qy[4] = {(float *)dptr(vq), (float *)dptr(vk), (float *)dptr(vv), (float *)dptr(vi)};
    bool ok = step_linear_multi(st, 4, qt, H, qouts, qwhat, xin, qy, linws, g.lin_ws_bytes);
    if (!ok) { drop(); return fail("a QSA projection launch failed"); }

    /* the per-bank caches: each bank's own KV and pooled-key tensors (L284 #3), its slice of the bank-major
     * index stage.  An evicted bank's are absent; the kernel refuses a row of it ("a sequence's cache is
     * missing"), and the server restores a bank before stepping it. */
    const uint64_t itb = pulsar_qwen_index_tail_bytes(s);
    pulsar_qsa_seq seqs[64];
    for (uint32_t b = 0; ok && b < nb; b++) {
        seqs[b].kv    = pulsar_state_pool(st->states, il, PULSAR_QWEN_STATE_KV, b);
        seqs[b].bkey  = pulsar_state_pool(st->states, il, PULSAR_QWEN_STATE_IDX_KEYS, b);
        seqs[b].stage = pulsar_gpu_tensor_view(pulsar_state_tensor(st->states, il, PULSAR_QWEN_STATE_IDX_TAIL), b * itb, itb);
        seqs[b].cap   = st->states->ctx;
        views[nv++] = seqs[b].stage;
        ok = seqs[b].stage != NULL;
    }
    if (!ok) { drop(); return fail("a QSA bank cache view failed"); }
    if (st->verify) {
        /* L251 MTP: the verify rows' index projections (the raw key rides in each), each at its row of the layer's
         * capture (L284 #2: a fused step's prompt run behind them is not captured); the core kept each run's stage
         * before the step (pulsar_state_verify_capture) and rebuilds it at the commit (pulsar_state_commit) */
        ok = pulsar_gpu_tensor_copy_async(pulsar_state_capture_tensor(st->states, PULSAR_QWEN_STATE_IDX_TAIL),
                                          pulsar_state_capture_offset(st->states, il, PULSAR_QWEN_STATE_IDX_TAIL), vi, 0,
                                          (uint64_t)st->n_dec * PULSAR_QSA_IDX_IN * 4u) != 0;
        if (!ok) { drop(); return fail("the verify capture of a QSA layer failed"); }
    }

    const pulsar_qsa_layer layer = {(const uint16_t *)wptr(st, L.attn_q_norm,  "qwen QSA q_norm"),
                                    (const uint16_t *)wptr(st, L.attn_k_norm,  "qwen QSA k_norm"),
                                    (const uint16_t *)wptr(st, L.idx_q_norm,   "qwen QSA idx_q_norm"),
                                    (const uint16_t *)wptr(st, L.idx_k_norm,   "qwen QSA idx_k_norm")};
    /* L284 #2: the attention picks its schedule by the call's row count (the CTA fold from QSA_FOLD_ROWS rows, the
     * split + combine below it -- not the same bytes), so the verify rows are one call and each prompt run its own
     * (attention_calls): a verify row attends as the verify alone does, a prompt row as its classic chunk does.
     * Every other kernel of the call is per row, and a bank's rows lie in one call. */
    const uint64_t OD = pulsar_qwen_qsa_out_dim(s);
    pulsar_gpu_tensor *ws_view = scratch_view(st, PULSAR_QWEN_OP_QSA, g.ws, g.ws_bytes);
    ok = ok && ws_view && attention_calls(st, [&](uint32_t r0, uint32_t m) {
        pulsar_gpu_tensor *sq = pulsar_gpu_tensor_view(vq, (uint64_t)r0 * QI * 4u, (uint64_t)m * QI * 4u);
        pulsar_gpu_tensor *sk = pulsar_gpu_tensor_view(vk, (uint64_t)r0 * KVI * 4u, (uint64_t)m * KVI * 4u);
        pulsar_gpu_tensor *sv = pulsar_gpu_tensor_view(vv, (uint64_t)r0 * KVI * 4u, (uint64_t)m * KVI * 4u);
        pulsar_gpu_tensor *si = pulsar_gpu_tensor_view(vi, (uint64_t)r0 * PULSAR_QSA_IDX_IN * 4u,
                                                       (uint64_t)m * PULSAR_QSA_IDX_IN * 4u);
        pulsar_qsa_io io{};
        io.qg = sq; io.k = sk; io.v = sv; io.idx = si;
        io.out_bf16 = (uint8_t *)dptr(sc) + g.obf16 + (uint64_t)r0 * OD * 2u;   /* L251 / ac69748f: bf16, not A8 */
        io.out_e4m3 = NULL;
        io.out_scale = (uint8_t *)dptr(sc) + g.a8_sf;
        io.out_sf_pitch = pulsar_gpu_mx_kbp(pulsar_qwen_qsa_out_dim(s));
        io.tap_out_f32 = NULL; io.tap_sel = NULL;              /* the lane, not a gate */
        io.tp_ranks = (int)pulsar_qwen_tp(s);
        io.row_rope = st->rope ? st->rope + (size_t)r0 * 6u : NULL;   /* L268: multi-axis rope, NULL for text */
        io.rope = &s->qsa_rope;                                       /* L280: the load's table (native or YaRN) */
        /* row_seq / row_pos are HOST arrays -- pulsar_gpu_qsa_forward walks them on
         * the host to build its row list.  The step already carries them as `bank` /
         * `pos`.  (The GDN call's row_slot is the opposite: a DEVICE pointer.  Passing
         * the device row_bank here segfaulted the host walk, 18:56.) */
        const bool done = sq && sk && sv && si &&
                          pulsar_gpu_qsa_forward(&layer, seqs, nb, (const uint32_t *)st->bank + r0,
                                                 (const uint32_t *)st->pos + r0, m, &io, ws_view) == 1;
        for (pulsar_gpu_tensor *t : {sq, sk, sv, si}) pulsar_gpu_tensor_free(t);
        return done;
    });
    pulsar_gpu_tensor_free(ws_view);
    drop();
    if (!ok) return fail("pulsar_gpu_qsa_forward refused the step");
    return step_linear(st, L.attn_o, (int)pulsar_qwen_qsa_out_dim(s), H, "qwen QSA o_proj",
                       (const uint16_t *)((uint8_t *)dptr(sc) + g.obf16), (float *)dptr(st->st->y), linws,
                       g.lin_ws_bytes) ||
           fail("the QSA o_proj launch failed");
}

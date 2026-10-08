/* spec_qwen.cpp -- Qwen4-exp's verify hooks and its MTP drafter behind the speculation round API
 * (L272 P1 S3).  What qwen_session_generate_speculative did in one private loop (L251 / L270 / L271)
 * is now the core's round (session_spec.cpp) with these hooks under it:
 *
 *   verify   the trunk runs [base, d_1 .. d_k] at p .. p + k as ONE PREFILL step of one bank with every
 *            row headed and the per-row state capture armed (qwen_forward verify = true; the server's
 *            lane reaches the same step through decode_mixed with heads on every row).  k + 1 <= 7, so
 *            every kernel takes its decode-width arm and each row's logits are the bytes one-token
 *            decode would give.
 *   commit   the recurrent state rolls back to the last kept row (pulsar_qwen_s4_spec_rollback over the
 *            capture), the bank's position counter moves, the logits are the kept row's.
 *   absorb   the MTP layer catches up: its stage back to the snapshot the draft chain took, the kept
 *            rows' MTP rows (stack at p + i, the token after it) in one PREFILL step, the last kept
 *            row's stack parked in mtp_pend for the next draft.
 *   draft    the lockstep row (pending stack at p - 1, the carry) headed gives d_1; each further step
 *            feeds the MTP's own streams with d_j; the chain stops at K or when the draft's probability
 *            falls under tau (TensorFold's rule; K = 4, tau = 0.7 measured on sparky 2026-09-29, see the
 *            loop's record in L270).  Sampled drafts are draws from the MTP's q, recorded for the
 *            core's accept walk.
 *
 * L272 P1 S4: the batched lane verifies several banks in one step (one run each through decode_mixed,
 * every kernel at its decode-width arm while the step stays within PULSAR_QWEN_SPEC_ROWS); each bank's
 * commit rolls its own run back, and the absorb and the draft serve each bank in turn.  L284: any number of
 * banks -- the server's allocator fits their drafts to the step's rows (fused_heads_max = SPEC_ROWS), so N
 * decoders keep speculating with shallower drafts. */
#include "pulsar_engine_internal.h"
#include "family_qwen.h"
#include "spec_internal.h"
#include "qwen_forward.h"

#include <math.h>

/* ---- the drafter's schedule ----------------------------------------------------------------------- */

/* The draft schedule: the chain continues while the last draft's MTP probability is >= tau, up to K
 * (TensorFold's rule).  Swept on sparky over three code prompts (2026-09-29): K = 4, tau = 0.7 gives
 * 58.7 / 58.9 / 59.2 tok/s vs 53.4 for a fixed K = 3; TensorFold's own 6 / 0.6 is 53.6 here -- a verify
 * row costs more on this engine (4 rows read up to 40 distinct experts).  tau = 0 is a fixed depth of K.
 * PULSAR_QWEN_MTP_TAU / PULSAR_QWEN_MTP_K override them.  NOT the adaptive controller (spec_depth.h,
 * DSpark's L107 rule): measured 2026-09-29 it LOST to the fixed cap of 4 (52-59 vs 59.5-60.4 tok/s at tau
 * 0.6-0.8) -- the confidence stop already adapts inside every round, and the rule's down-signal fires on
 * chains that stopped early by design; so the target's depth policy is NULL. */
static uint32_t mtp_depth_default(const pulsar_engine *) {
    static const uint32_t K = [] {
        const char *k = getenv("PULSAR_QWEN_MTP_K");
        const int v = k && k[0] ? atoi(k) : 4;
        return (uint32_t)(v < 1 ? 1 : v > (int)PULSAR_QWEN_SPEC_DRAFT_MAX ? (int)PULSAR_QWEN_SPEC_DRAFT_MAX : v);
    }();
    return K;
}

static float mtp_tau(void) {
    static const float tau = [] {
        const char *t = getenv("PULSAR_QWEN_MTP_TAU");
        return t && t[0] ? (float)atof(t) : 0.7f;
    }();
    return tau;
}

/* ---- the target hooks ------------------------------------------------------------------------- */

/* The round's verify rows: the base at saved_len, then the drafts. */
static void qwen_verify_rows(const pulsar_session *s, const pulsar_spec_round *r, int32_t *tok, int32_t *pos,
                             int32_t *bank) {
    const int32_t b = (int32_t)s->qwen->live_bank;
    for (uint32_t i = 0; i < r->n_batch; i++) {
        tok[i] = i == 0 ? r->base : r->pend[i - 1];
        pos[i] = r->saved_len + (int32_t)i;
        bank[i] = b;
    }
}

/* The single lane's verify: the trunk over the round's rows, every row headed into the host rows the
 * core reads; the row argmaxes for the walk. */
static bool qwen_verify_single(pulsar_session *s, pulsar_spec_round *r, pulsar_spec_rows *out, char *err,
                               size_t errlen) {
    pulsar_qwen_state *q = s->qwen;
    const uint32_t V = g_qwen_shape.n_vocab, R = r->n_batch;
    if (!q->mtp || R > PULSAR_QWEN_SPEC_DRAFT_MAX + 1u || R > PULSAR_QWEN_SPEC_ROWS) {
        snprintf(err, errlen, "%s: a verify step of %u rows is outside the capture", PULSAR_QWEN_ARCH, R);
        return false;
    }
    if (q->bank_pos[q->live_bank] != (uint32_t)r->saved_len) {
        snprintf(err, errlen, "%s: verify at %d, the bank's state holds %u tokens", PULSAR_QWEN_ARCH, r->saved_len,
                 q->bank_pos[q->live_bank]);
        return false;
    }
    int32_t tok[PULSAR_QWEN_SPEC_DRAFT_MAX + 1], pos[PULSAR_QWEN_SPEC_DRAFT_MAX + 1], bank[PULSAR_QWEN_SPEC_DRAFT_MAX + 1];
    qwen_verify_rows(s, r, tok, pos, bank);
    if (!qwen_forward(s, PULSAR_QWEN_STEP_PREFILL, tok, pos, bank, R, qwen_heads_span(0, R), q->spec_logits, R)) {
        snprintf(err, errlen, "%s: the verify step refused (see the log for the op)", PULSAR_QWEN_ARCH);
        return false;
    }
    for (uint32_t i = 0; i < r->K && i < 16u; i++) r->row_tops[i] = (int)qwen_argmax(q->spec_logits + (size_t)i * V, V);
    out->read = pulsar_spec_row_read_block;
    out->ud = out;
    out->compact = NULL;
    out->block.rows = q->spec_logits;
    out->block.row0 = 0;
    out->block.vocab = V;
    return true;
}

/* The walk kept `commit` drafts: the installed bank's run of the last verify back to its last kept row
 * (rows 1 + commit .. of the run undone; the rollback finds the bank's run in the capture, wherever it sat
 * in the step -- row0), the bank's position counter there, the kept row's logits fresh. */
static bool qwen_spec_commit(pulsar_session *s, pulsar_spec_round *r, uint32_t commit, uint32_t) {
    pulsar_engine *e = s->engine;
    pulsar_qwen_state *q = s->qwen;
    int32_t tok[PULSAR_QWEN_SPEC_DRAFT_MAX + 1], pos[PULSAR_QWEN_SPEC_DRAFT_MAX + 1], bank[PULSAR_QWEN_SPEC_DRAFT_MAX + 1];
    qwen_verify_rows(s, r, tok, pos, bank);
    pulsar_qwen_step vst{};
    vst.shape = &g_qwen_shape;
    vst.w = e->qwen_weights;
    vst.plan = &e->plan;
    vst.st = q;
    vst.mode = PULSAR_QWEN_STEP_PREFILL;
    vst.n_rows = r->n_batch;
    vst.tokens = tok;
    vst.pos = pos;
    vst.bank = bank;
    const uint32_t keep = 1u + commit;
    if (!pulsar_qwen_s4_spec_rollback(&vst, keep)) return false;
    qwen_bank_set_pos(q, q->live_bank, (uint32_t)r->saved_len + keep);
    s->logits_stale = false;
    return true;
}

/* The L155 trim (the caller's accepted[] bound below what the round committed): a recurrent state has no
 * rewind, so the bank's view is invalidated and its next sync prefills it (a grid checkpoint at or below
 * the cut would do; S4). */
static void qwen_spec_cut(pulsar_session *s, int) {
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->logits_stale = true;
}

const pulsar_spec_target_ops k_qwen_spec_target = {
    /* .snapshot      = */ NULL,   /* the verify step captures its own undo state (qwen_forward verify) */
    /* .restore       = */ NULL,
    /* .release       = */ NULL,
    /* .round_note    = */ NULL,   /* one logits readback: the block */
    /* .arm_verify    = */ NULL,   /* a heads-on-every-row decode_mixed IS the verify */
    /* .verify_single = */ qwen_verify_single,
    /* .verify_rows   = */ NULL,   /* the block holds every row */
    /* .commit        = */ qwen_spec_commit,
    /* .cut           = */ qwen_spec_cut,
    /* .depth         = */ NULL,   /* a fixed depth: the drafter's K and tau */
    /* .readback      = */ NULL,   /* its batched steps read back full rows */
};

/* ---- the MTP drafter --------------------------------------------------------------------------- */

/* One bank's committed rows: the MTP layer's stage back to the snapshot its draft chain took, the kept rows'
 * MTP rows (stack at row i, next[i]) in one PREFILL step, the last kept row's stack parked for the next draft. */
static bool mtp_absorb_bank(pulsar_session *s, const uint32_t *rows, uint32_t b, const int32_t *next, uint32_t n) {
    pulsar_engine *e = s->engine;
    pulsar_qwen_state *q = s->qwen;
    const pulsar_qwen_shape *sh = &g_qwen_shape;
    const uint32_t il_mtp = e->plan.n_layer;
    for (uint32_t i = 1; i < n; i++)
        if (rows[i] != rows[0] + i) {
            fprintf(stderr, "pulsar: %s: bank %u's absorbed rows are not one run -- refusing\n", PULSAR_QWEN_ARCH, b);
            return false;
        }
    if (b >= q->n_banks || q->bank_pos[b] < n) return false;
    const uint64_t hc = pulsar_qwen_hc_dim(sh) * PULSAR_QWEN_STREAM_ELT_SIZE;
    const uint64_t itb = pulsar_qwen_index_tail_bytes(sh);
    const uint32_t p0 = q->bank_pos[b] - n;   /* the commit moved the counter past the kept rows */
    bool ok = true;
    if (q->mtp_stage_dirty[b]) {
        ok = pulsar_gpu_tensor_copy_async(q->layer[il_mtp].idx_tail, (uint64_t)b * itb, q->mtp_stage, (uint64_t)b * itb,
                                          itb) != 0;
        q->mtp_stage_dirty[b] = false;
    }
    if (ok && n > 1) {
        int32_t pos[PULSAR_QWEN_SPEC_ROWS], bank[PULSAR_QWEN_SPEC_ROWS];
        for (uint32_t i = 0; i + 1 < n; i++) { pos[i] = (int32_t)(p0 + i); bank[i] = (int32_t)b; }
        ok = pulsar_gpu_tensor_copy_async(q->mtp_h, 0, q->streams, (uint64_t)rows[0] * hc, (uint64_t)(n - 1) * hc) != 0 &&
             qwen_mtp_forward(s, PULSAR_QWEN_STEP_PREFILL, next, pos, bank, n - 1, 0, 0, NULL);
    }
    if (ok) ok = pulsar_gpu_tensor_copy_async(q->mtp_pend, (uint64_t)b * hc, q->streams, (uint64_t)rows[n - 1] * hc, hc) != 0;
    if (!ok) {
        q->mtp_pend_pos[b] = UINT32_MAX;
        fprintf(stderr, "pulsar: %s: the MTP absorb failed; bank %u drafts nothing until its next row\n", PULSAR_QWEN_ARCH, b);
        return false;
    }
    q->mtp_pend_pos[b] = p0 + n - 1;
    return true;
}

/* The committed rows of the last verify, any number of banks: each bank's rows in turn (L272 P1 S4). */
static bool mtp_absorb_banked(pulsar_session *s, const uint32_t *rows, const uint32_t *banks, const int32_t *next,
                              uint32_t n) {
    bool ok = true;
    for (uint32_t i = 0; i < n;) {
        uint32_t j = i + 1;
        while (j < n && banks[j] == banks[i]) j++;
        ok = mtp_absorb_bank(s, rows + i, banks[i], next + i, j - i) && ok;
        i = j;
    }
    return ok;
}

/* The chain for bank `b` after the carry `x` at the bank's next position, into a redraft
 * record: refined[0] = x, refined[1..keep] = the drafts, conf[] their MTP probabilities, and for sampled
 * drafts q(d) with q's support (compact, or the MTP logits scattered to the vocabulary for the walk to
 * rebuild q under the same params). */
static bool mtp_draft_record(pulsar_session *s, uint32_t b, int32_t x, uint32_t K, float temperature, int top_k,
                             float top_p, float min_p, uint64_t *rng, spec_redraft_req *q, char *err, size_t errlen) {
    pulsar_engine *e = s->engine;
    pulsar_qwen_state *st = s->qwen;
    const pulsar_qwen_shape *sh = &g_qwen_shape;
    const uint32_t il_mtp = e->plan.n_layer, V = sh->n_vocab;
    const uint64_t hc = pulsar_qwen_hc_dim(sh) * PULSAR_QWEN_STREAM_ELT_SIZE;
    const uint64_t itb = pulsar_qwen_index_tail_bytes(sh);
    const uint32_t p = st->bank_pos[b];   /* the carry's position */
    const bool sampled = temperature > 0.0f;
    float *L = st->spec_logits;
    q->keep = 0;
    q->have_conf = true;
    q->sample_drafts = sampled;
    q->refined[0] = x;
    if (!st->mtp) {
        snprintf(err, errlen, "%s: speculative decoding needs the MTP layer (the sidecar shard)", PULSAR_QWEN_ARCH);
        return false;
    }
    if (sampled && !rng) {
        snprintf(err, errlen, "%s: sampled speculation needs the request's rng", PULSAR_QWEN_ARCH);
        return false;
    }
    /* no pending row (a failed MTP step, a fresh restore): this round verifies the carry alone */
    if (st->mtp_pend_pos[b] == UINT32_MAX || st->mtp_pend_pos[b] + 1u != p) return true;
    /* the drafts and the carry must fit the context */
    if ((uint64_t)p + 1u + K > (uint64_t)st->ctx) K = (uint32_t)st->ctx > p + 1u ? (uint32_t)st->ctx - p - 1u : 0u;
    if (K == 0) return true;
    const int32_t bb = (int32_t)b;
    int32_t tp = (int32_t)p - 1;
    /* the lockstep row (stack at p - 1, x), headed: d_1 */
    bool ok = pulsar_gpu_tensor_copy_async(st->mtp_h, 0, st->mtp_pend, (uint64_t)b * hc, hc) != 0 &&
              qwen_mtp_forward(s, PULSAR_QWEN_STEP_DECODE, &x, &tp, &bb, 1, 0, 1, L);
    st->mtp_pend_pos[b] = UINT32_MAX;   /* consumed; the absorb re-parks */
    /* the MTP layer's stage after its last TRUE row: the chain below writes draft rows into it */
    if (ok) {
        ok = pulsar_gpu_tensor_copy_async(st->mtp_stage, (uint64_t)b * itb, st->layer[il_mtp].idx_tail,
                                          (uint64_t)b * itb, itb) != 0;
        st->mtp_stage_dirty[b] = true;
    }
    float conf = 1.0f;
    pulsar_sample_dist qd{};
    /* one draft from the MTP row in L: its argmax (greedy), or a draw from q (sampled; conf = q(d)) */
    auto draft = [&](uint32_t j) -> bool {
        if (!sampled) {
            q->refined[j] = qwen_mtp_argmax(e, L, &conf);
            q->conf[j - 1] = conf;
            q->qn[j - 1] = 0;
            return true;
        }
        if (!qwen_mtp_dist(s, L, temperature, top_k, top_p, min_p, &qd)) return false;
        const int d = pulsar_sample_dist_draw(&qd, rng);
        if (d < 0) { pulsar_sample_dist_free(&qd); return false; }
        conf = pulsar_sample_dist_prob(&qd, d);
        q->refined[j] = (int32_t)d;
        q->conf[j - 1] = conf;
        q->q_drawn[j - 1] = conf;
        if (pulsar_spec_q_compact(qd.n)) {
            q->qn[j - 1] = qd.n;
            memcpy(q->qids[j - 1], qd.ids, (size_t)qd.n * sizeof(int32_t));
            memcpy(q->qprobs[j - 1], qd.probs, (size_t)qd.n * sizeof(float));
        } else {
            /* too wide to store: the walk rebuilds q from a row under the stored params -- the MTP
             * logits scattered to the vocabulary, every other id -inf (the same support) */
            if (!pulsar_spec_redraft_qrows_reserve(q, V)) { pulsar_sample_dist_free(&qd); return false; }
            float *row = q->qrows + (size_t)(j - 1) * V;
            for (uint32_t i = 0; i < V; i++) row[i] = -INFINITY;
            const pulsar_qwen_weights *w = e->qwen_weights;
            for (uint32_t i = 0; i < w->n_draft; i++) row[w->draft_ids[i]] = L[i];
            q->qn[j - 1] = 0;
        }
        pulsar_sample_dist_free(&qd);
        return true;
    };
    uint32_t k = 0;
    if (ok) ok = draft(1);
    if (ok) k = 1;
    for (uint32_t j = 2; ok && j <= K; j++) {
        if (conf < mtp_tau()) break;      /* the schedule: stop on an unsure draft */
        tp = (int32_t)p + (int32_t)j - 2;
        ok = pulsar_gpu_tensor_copy_async(st->mtp_h, 0, st->mtp_streams, 0, hc) != 0 &&
             qwen_mtp_forward(s, PULSAR_QWEN_STEP_DECODE, &q->refined[j - 1], &tp, &bb, 1, 0, 1, L);
        if (ok) ok = draft(j);
        if (ok) k = j;
    }
    if (!ok) {
        snprintf(err, errlen, "%s: the MTP draft chain failed (see the log)", PULSAR_QWEN_ARCH);
        return false;
    }
    q->keep = k;
    return true;
}

/* The single lane's draft: the chain into a record, the record stamped into the bank's pendings. */
static uint32_t mtp_draft(pulsar_session *s, int next_base, bool, float temperature, int top_k, float top_p,
                          float min_p, uint64_t *rng) {
    spec_redraft_req rec;
    memset(&rec, 0, sizeof rec);
    rec.valid = true;
    rec.next_base = next_base;
    rec.n_draft = pulsar_spec_cur_depth(s);
    rec.temperature = temperature;
    rec.top_k = top_k;
    rec.top_p = top_p;
    rec.min_p = min_p;
    char err[200];
    uint32_t keep = 0;
    if (mtp_draft_record(s, s->qwen->live_bank, (int32_t)next_base, rec.n_draft, temperature, top_k, top_p, min_p, rng,
                         &rec, err, sizeof err)) {
        pulsar_spec_redraft_stamp(s, &rec);
        keep = rec.keep;
    } else {
        fprintf(stderr, "pulsar: %s (the round takes a plain step next)\n", err);
        pulsar_spec_drop_pendings(&s->spec);
    }
    free(rec.qrows);
    return keep;
}

/* The batched lane's draft: each bank's round in turn (L272 P1 S4; the chain is one MTP layer a step, so the
 * banks' chains need not share a step to stay cheap beside the verify). */
static int mtp_draft_batch(pulsar_session *s, pulsar_spec_round **rounds, const uint32_t *banks, uint64_t **rngs, int n,
                           char *err, size_t errlen) {
    for (int i = 0; i < n; i++) {
        spec_redraft_req *q = &rounds[i]->redraft;
        if (!q->valid || q->n_draft == 0) continue;
        if (banks[i] >= s->qwen->n_banks) {
            snprintf(err, errlen, "%s: the draft's bank %u is outside the pool", PULSAR_QWEN_ARCH, banks[i]);
            return -1;
        }
        if (!mtp_draft_record(s, banks[i], q->next_base, q->n_draft, q->temperature, q->top_k, q->top_p, q->min_p,
                              rngs[i], q, err, errlen))
            return -1;
        q->done = true;
    }
    return 0;
}

const pulsar_drafter_ops k_mtp_drafter = {
    /* .name          = */ "MTP",
    /* .depth_default = */ mtp_depth_default,
    /* .prime         = */ NULL,   /* the trunk's stacks are the conditioning; the prefill parks the last one */
    /* .absorb        = */ NULL,   /* a round's rows together: the kept rows' MTP rows are one PREFILL step */
    /* .absorb_banked = */ mtp_absorb_banked,
    /* .draft         = */ mtp_draft,
    /* .harvest       = */ NULL,
    /* .draft_batch   = */ mtp_draft_batch,
};

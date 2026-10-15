/* spec_qwen.cpp -- Qwen4-exp's verify hooks and its MTP drafter behind the speculation round API
 * (L272 P1 S3).  What qwen_session_generate_speculative did in one private loop (L251 / L270 / L271)
 * is now the core's round (session_spec.cpp) with these hooks under it:
 *
 *   verify   the trunk runs [base, d_1 .. d_k] at p .. p + k as ONE PREFILL step of one bank with every
 *            row headed and the per-row state capture armed (qwen_forward verify = true; the server's
 *            lane reaches the same step through decode_mixed with heads on every row).  k + 1 <= 16 (a copy
 *            chain's, L288; an MTP chain's <= 8), so every kernel takes its decode-width arm and each row's
 *            logits are the bytes one-token decode would give.
 *   commit   the bank's declared state is set to the last kept row's (pulsar_state_commit over the
 *            capture, L295 F3: L288 the GDN recurrent state, which the verify left untouched, advanced by the
 *            kept rows' updates; the rest rolled back), the bank's position counter moves, the logits are the
 *            kept row's.
 *   absorb   the MTP layer catches up: its stage back to the snapshot the draft chain took, the kept
 *            rows' MTP rows (stack at p + i, the token after it) in one PREFILL step of decode rows (L288),
 *            the last kept row's stack parked in mtp_pend for the next draft.
 *   draft    the lockstep row (pending stack at p - 1, the carry) headed gives d_1; each further step
 *            feeds the MTP's own streams with d_j; the chain is the shared stop rule (spec_ops.h) run token
 *            by token: it ends at the depth or at the first draft whose head was unsure (top probability
 *            under tau), that draft still verified.  Sampled drafts are draws from the MTP's q, recorded for
 *            the core's accept walk.
 *
 * L272 P1 S4: the batched lane verifies several banks in one step (one run each through decode_mixed,
 * every kernel at its decode-width arm while the step stays within PULSAR_QWEN_SPEC_ROWS); each bank's
 * commit rolls its own run back.  L288: the absorb of every bank's kept rows is one MTP step (decode rows), and
 * the draft chains of every bank run in lockstep -- one MTP step a position for all of them, the draft head once
 * a step, not once a bank.  L284: any number of banks -- the server's allocator fits their drafts to the step's
 * rows (fused_heads_max = SPEC_ROWS), so N decoders keep speculating with shallower drafts. */
#include "pulsar_engine_internal.h"
#include "family_qwen.h"
#include "spec_internal.h"
#include "qwen_forward.h"

#include <math.h>

static_assert(PULSAR_QWEN_SPEC_DRAFT_MAX + 1u <= PULSAR_QWEN_SPEC_ROWS, "a bank's verify run fits the verify width");

/* ---- the target hooks ------------------------------------------------------------------------- */

/* The single lane's verify: the trunk over the round's rows, every row headed into the host rows the
 * core reads; the row argmaxes for the walk. */
static bool qwen_verify_single(pulsar_session *s, pulsar_spec_round *r, pulsar_spec_rows *out, char *err,
                               size_t errlen) {
    pulsar_qwen_state *q = s->qwen;
    const uint32_t V = g_qwen_shape.n_vocab, R = r->n_batch;
    if (R > PULSAR_QWEN_SPEC_DRAFT_MAX + 1u) {
        snprintf(err, errlen, "%s: a verify step of %u rows is outside the capture", PULSAR_QWEN_ARCH, R);
        return false;
    }
    /* the batched lane's verify (decode_mixed, every row headed) over this round's rows alone */
    pulsar_multiseq_req reqs[PULSAR_QWEN_SPEC_DRAFT_MAX + 1];
    pulsar_spec_round_fill_reqs(r, s->states->live_bank, reqs);
    if (s->engine->family->session->decode_mixed(s, reqs, R, q->spec_logits, (int)(R * V), NULL,
                                                 PULSAR_MSEQ_HEAD_ALL_ROWS, err, errlen) != 0)
        return false;
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
    pulsar_state_set *ss = s->states;
    pulsar_multiseq_req reqs[PULSAR_QWEN_SPEC_DRAFT_MAX + 1];
    int32_t tok[PULSAR_QWEN_SPEC_DRAFT_MAX + 1], pos[PULSAR_QWEN_SPEC_DRAFT_MAX + 1], bank[PULSAR_QWEN_SPEC_DRAFT_MAX + 1];
    if (r->n_batch > PULSAR_QWEN_SPEC_DRAFT_MAX + 1u) return false;
    pulsar_spec_round_fill_reqs(r, ss->live_bank, reqs);
    if (!qwen_verify_rows(s, reqs, r->n_batch, tok, pos, bank, NULL, 0)) return false;
    const uint32_t keep = 1u + commit;
    if (!pulsar_state_commit(ss, ss->live_bank, tok, keep)) return false;
    pulsar_state_set_pos(ss, ss->live_bank, (uint32_t)r->saved_len + keep);
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
    /* .readback      = */ NULL,   /* its batched steps read back full rows */
};

/* ---- the MTP drafter --------------------------------------------------------------------------- */

/* The committed rows of the last verify, any number of banks (each bank's rows one run, in the step's row order):
 * every bank's MTP layer back to the stage its draft chain snapshotted, the kept rows' MTP rows (stack at row i,
 * next[i]) of EVERY bank in one step (L288: one run a bank, decode rows -- it was one prefill step a bank, at the
 * prompt arms), each bank's last kept row's stack parked for its next draft. */
static bool mtp_absorb_banked(pulsar_session *s, const uint32_t *rows, const uint32_t *banks, const int32_t *next,
                              uint32_t n) {
    pulsar_engine *e = s->engine;
    pulsar_qwen_state *q = s->qwen;
    pulsar_state_set *ss = s->states;
    const pulsar_qwen_shape *sh = &g_qwen_shape;
    const uint32_t il_mtp = e->plan.n_layer;
    const uint64_t hc = pulsar_qwen_hc_dim(sh) * PULSAR_QWEN_STREAM_ELT_SIZE;
    const uint64_t itb = pulsar_qwen_index_tail_bytes(sh);
    int32_t tok[PULSAR_QWEN_SPEC_ROWS], pos[PULSAR_QWEN_SPEC_ROWS], bank[PULSAR_QWEN_SPEC_ROWS];
    uint32_t m = 0;   /* the step's rows */
    bool ok = n <= PULSAR_QWEN_SPEC_ROWS;
    for (uint32_t i = 0; ok && i < n;) {
        const uint32_t b = banks[i];
        uint32_t j = i + 1;
        while (j < n && banks[j] == b) j++;
        const uint32_t k = j - i;
        for (uint32_t r = i + 1; ok && r < j; r++) ok = rows[r] == rows[i] + (r - i);
        if (!ok || b >= ss->n_banks || ss->bank_pos[b] < k) {
            fprintf(stderr, "pulsar: %s: bank %u's absorbed rows are not one run of its kept rows -- refusing\n",
                    PULSAR_QWEN_ARCH, b);
            ok = false;
            break;
        }
        const uint32_t p0 = ss->bank_pos[b] - k;   /* the commit moved the counter past the kept rows */
        if (q->mtp_stage_dirty[b]) {
            ok = pulsar_gpu_tensor_copy_async(pulsar_state_tensor(ss, il_mtp, PULSAR_QWEN_STATE_IDX_TAIL),
                                              (uint64_t)b * itb, q->mtp_stage,
                                              (uint64_t)b * itb, itb) != 0;
            q->mtp_stage_dirty[b] = false;
        }
        if (ok && k > 1)
            ok = pulsar_gpu_tensor_copy_async(q->mtp_h, (uint64_t)m * hc, q->streams, (uint64_t)rows[i] * hc,
                                              (uint64_t)(k - 1) * hc) != 0;
        for (uint32_t r = 0; r + 1 < k; r++, m++) {
            tok[m] = next[i + r];
            pos[m] = (int32_t)(p0 + r);
            bank[m] = (int32_t)b;
        }
        i = j;
    }
    if (ok && m > 0) ok = qwen_mtp_forward(s, PULSAR_QWEN_STEP_PREFILL, tok, pos, bank, m, 0, 0, NULL, NULL, true);
    for (uint32_t i = 0; i < n;) {
        const uint32_t b = banks[i];
        uint32_t j = i + 1;
        while (j < n && banks[j] == b) j++;
        bool parked = ok && b < ss->n_banks && ss->bank_pos[b] >= j - i;
        if (parked)
            parked = pulsar_gpu_tensor_copy_async(pulsar_state_tensor(ss, il_mtp, PULSAR_QWEN_STATE_MTP_PEND),
                                                  (uint64_t)b * hc, q->streams, (uint64_t)rows[j - 1] * hc, hc) != 0;
        if (b < ss->n_banks) q->mtp_pend_pos[b] = parked ? ss->bank_pos[b] - 1u : UINT32_MAX;
        ok = ok && parked;
        i = j;
    }
    if (!ok)
        fprintf(stderr, "pulsar: %s: the MTP absorb failed; its banks draft nothing until their next row\n",
                PULSAR_QWEN_ARCH);
    return ok;
}

/* The draft chains of `n` banks (L288: in lockstep -- step j runs every bank still drafting as ONE MTP step, one
 * row a bank, every row headed and picked in one launch; the batched lane drafted bank by bank before, so the
 * draft head ran once per bank per step).  Bank banks[i]'s chain follows its record's carry x at the bank's next
 * position, into the record: refined[0] = x, refined[1..keep] = the drafts, conf[] the head's top probability at
 * each position, and for sampled drafts q(d) with q's support (compact, or the MTP logits scattered to the
 * vocabulary for the walk to rebuild q under the same params).  The stop rule runs position by position: the
 * first draft whose confidence is under tau is its chain's last, so no MTP step is spent past it.  A row's bytes
 * do not depend on the rows beside it (one row a bank, the decode arms, a per-row pick), so each chain is the one
 * its bank drafts alone. */
static bool mtp_draft_chains(pulsar_session *s, const uint32_t *banks, uint64_t *const *rngs,
                             spec_redraft_req *const *reqs, uint32_t n, char *err, size_t errlen) {
    pulsar_engine *e = s->engine;
    pulsar_qwen_state *st = s->qwen;
    pulsar_state_set *ss = s->states;
    const pulsar_qwen_shape *sh = &g_qwen_shape;
    const pulsar_qwen_weights *w = e->qwen_weights;
    const uint32_t il_mtp = e->plan.n_layer, nc = sh->ngram_size - 1u;
    const uint64_t hc = pulsar_qwen_hc_dim(sh) * PULSAR_QWEN_STREAM_ELT_SIZE;
    const uint64_t itb = pulsar_qwen_index_tail_bytes(sh);
    const float tau = pulsar_spec_tau(e);
    if (!st->mtp) {
        snprintf(err, errlen, "%s: speculative decoding needs the MTP layer (the sidecar shard)", PULSAR_QWEN_ARCH);
        return false;
    }
    if (n > PULSAR_QWEN_HEAD_ROWS_MAX || nc != 2u) {
        snprintf(err, errlen, "%s: %u draft chains in one lockstep (at most %u) or an n-gram context of %u",
                 PULSAR_QWEN_ARCH, n, PULSAR_QWEN_HEAD_ROWS_MAX, nc);
        return false;
    }
    struct chain {
        uint32_t b, p, K;       /* the bank, the carry's position, the chain's depth */
        int32_t pctx[2];        /* the bank's n-gram context, for the PLE prefetch */
        bool sampled, live;     /* live: the chain takes the next step */
        uint32_t row;           /* its row in the last step */
        float run;              /* its survival: the product of its drafts' confidences */
        bool copy;              /* L288: its drafts are a copy chain; it takes step 1 only (its MTP row) */
    } c[PULSAR_QWEN_HEAD_ROWS_MAX];
    for (uint32_t i = 0; i < n; i++) {
        spec_redraft_req *q = reqs[i];
        chain &ch = c[i];
        ch.b = banks[i];
        ch.p = ss->bank_pos[ch.b];
        ch.sampled = q->temperature > 0.0f;
        ch.live = false;
        ch.copy = q->copied;   /* L288: the record holds a copy chain (pulsar_spec_draft_req_init) */
        ch.run = 1.0f;
        if (!ch.copy) {
            q->keep = 0;
            q->have_conf = true;
            q->refined[0] = q->next_base;
        }
        if (ch.sampled && !rngs[i]) {
            snprintf(err, errlen, "%s: sampled speculation needs the request's rng", PULSAR_QWEN_ARCH);
            return false;
        }
        /* no pending row (a failed MTP step, a fresh restore): this round verifies the carry alone */
        if (st->mtp_pend_pos[ch.b] == UINT32_MAX || st->mtp_pend_pos[ch.b] + 1u != ch.p) continue;
        /* the drafts and the carry must fit the context */
        if (q->n_draft > PULSAR_QWEN_SPEC_DRAFT_MAX) {
            snprintf(err, errlen, "%s: a chain of %u drafts is past the verify capture's %u", PULSAR_QWEN_ARCH,
                     q->n_draft, PULSAR_QWEN_SPEC_DRAFT_MAX);
            return false;
        }
        ch.K = q->n_draft;
        if ((uint64_t)ch.p + 1u + ch.K > (uint64_t)ss->ctx) ch.K = ss->ctx > ch.p + 1u ? ss->ctx - ch.p - 1u : 0u;
        if (ch.K == 0) continue;
        ch.live = true;
        /* L289: the verify will embed [x, d_1 ..] after the bank's n-gram context: their PLE rows are read into the
         * page cache while the chain runs (each as it is known), so the verify's gather finds them there */
        const int32_t *ngram = (const int32_t *)pulsar_state_host(ss, PULSAR_QWEN_STATE_NGRAM, ch.b);
        ch.pctx[0] = ngram[0];
        ch.pctx[1] = ngram[1];
        pulsar_qwen_s4_ple_prefetch(w, ch.pctx, &q->refined[0], ch.copy ? 1u + q->keep : 1u);
    }
    /* L288 pass 2: the price a further step must clear (session_spec.cpp, the priced depth), and every survival the
     * round has drafted so far -- what outranks a chain for the shared verify width */
    uint32_t n_live = 0;
    for (uint32_t i = 0; i < n; i++) n_live += c[i].live ? 1u : 0u;
    const pulsar_spec_chain_price price = pulsar_spec_chain_price_for(s, n_live);
    float drafted[PULSAR_QWEN_HEAD_ROWS_MAX * PULSAR_QWEN_SPEC_DRAFT_MAX];
    uint32_t n_drafted = 0;
    /* L288 copy drafts: a record that holds a copy chain (pulsar_spec_draft_req_init) keeps it; its drafts outrank the
     * MTP drafts by their survival like any other.  It still takes step 1 when its bank has a parked MTP row -- the
     * lockstep row writes the bank's MTP row at p - 1, which the next round's chain conditions on -- and its pick
     * there is dropped. */
    for (uint32_t i = 0; i < n; i++) {
        if (!c[i].copy) continue;
        float run = 1.0f;
        for (uint32_t k = 0; k < reqs[i]->keep; k++) drafted[n_drafted++] = run *= reqs[i]->conf[k];
    }
    bool any_sampled = false;
    for (uint32_t i = 0; i < n; i++) any_sampled |= c[i].live && !c[i].copy && c[i].sampled;
    /* L289: the picks are the device's; the rows come back only when a chain samples (its q) */
    float *L = any_sampled ? st->spec_logits : NULL;
    qwen_mtp_pick pick[PULSAR_QWEN_HEAD_ROWS_MAX];
    /* position j of chain i from its row's pick: its confidence (whether it is the chain's last), then its draft:
     * the pick (greedy) or a draw from q over the row in L (sampled) */
    auto draft = [&](uint32_t i, uint32_t j) -> bool {
        if (c[i].copy) {   /* its drafts are the copy; step 1 wrote its MTP row */
            c[i].live = false;
            return true;
        }
        spec_redraft_req *q = reqs[i];
        const qwen_mtp_pick &pk = pick[c[i].row];
        const float *row = L ? L + (size_t)c[i].row * w->n_draft : NULL;
        q->conf[j - 1] = pk.conf;
        q->keep = j;
        c[i].run *= pk.conf;
        drafted[n_drafted++] = c[i].run;   /* j <= K <= PULSAR_QWEN_SPEC_DRAFT_MAX (refused above) */
        if (pulsar_spec_conf_stops(pk.conf, tau) || j == c[i].K) c[i].live = false;   /* the chain's last */
        if (!c[i].sampled) {
            q->refined[j] = pk.id;
            pulsar_qwen_s4_ple_prefetch(w, c[i].pctx, &q->refined[j], 1);
            return true;
        }
        pulsar_sample_dist qd{};
        if (!qwen_mtp_dist(s, row, q->temperature, q->top_k, q->top_p, q->min_p, &qd)) return false;
        /* draft j sits j places past the carry: the key the target's draw for that token reads */
        const int d = pulsar_sample_dist_draw(&qd, pulsar_sample_key(*rngs[i], j));
        pulsar_sample_dist_free(&qd);
        if (d < 0) return false;
        q->refined[j] = (int32_t)d;
        pulsar_qwen_s4_ple_prefetch(w, c[i].pctx, &q->refined[j], 1);
        return true;
    };
    int32_t tok[PULSAR_QWEN_HEAD_ROWS_MAX], pos[PULSAR_QWEN_HEAD_ROWS_MAX], bank[PULSAR_QWEN_HEAD_ROWS_MAX];
    bool ok = true;
    for (uint32_t j = 1; ok; j++) {
        /* the step's rows: step 1 is every chain's lockstep row (the pending stack at p - 1, x); step j > 1 feeds
         * each live chain's MTP streams of its last row with d_{j-1} at p + j - 2 */
        uint32_t m = 0;
        bool headed = false;   /* a chain that drafts from this step's pick rides it (a copy chain needs no head) */
        for (uint32_t i = 0; ok && i < n; i++) {
            chain &ch = c[i];
            if (!ch.live) continue;
            headed |= !ch.copy;
            ok = j == 1 ? pulsar_gpu_tensor_copy_async(st->mtp_h, (uint64_t)m * hc,
                                                       pulsar_state_tensor(ss, il_mtp, PULSAR_QWEN_STATE_MTP_PEND),
                                                       (uint64_t)ch.b * hc, hc) != 0
                        : pulsar_gpu_tensor_copy_async(st->mtp_h, (uint64_t)m * hc, st->mtp_streams,
                                                       (uint64_t)ch.row * hc, hc) != 0;
            tok[m] = reqs[i]->refined[j - 1];
            pos[m] = (int32_t)(ch.p + j) - 2;
            bank[m] = (int32_t)ch.b;
            ch.row = m++;
        }
        if (!ok || m == 0) break;
        const double step_t0 = now_sec();
        ok = headed ? qwen_mtp_forward(s, PULSAR_QWEN_STEP_DECODE, tok, pos, bank, m, 0, m, L, pick)
                    : qwen_mtp_forward(s, PULSAR_QWEN_STEP_DECODE, tok, pos, bank, m, 0, 0, NULL, NULL);
        if (j == 1) {
            for (uint32_t i = 0; i < n; i++) {
                if (!c[i].live) continue;
                const uint32_t b = c[i].b;
                st->mtp_pend_pos[b] = UINT32_MAX;   /* consumed; the absorb re-parks */
                /* the MTP layer's stage after its last TRUE row: the chain below writes draft rows into it */
                if (!ok) continue;
                ok = pulsar_gpu_tensor_copy_async(st->mtp_stage, (uint64_t)b * itb,
                                                  pulsar_state_tensor(ss, il_mtp, PULSAR_QWEN_STATE_IDX_TAIL),
                                                  (uint64_t)b * itb, itb) != 0;
                st->mtp_stage_dirty[b] = true;
            }
        }
        for (uint32_t i = 0; ok && i < n; i++)
            if (c[i].live) ok = draft(i, j);
        /* the step's wall, picks and draws included (the picks' readback waited for the device): the drafter's step
         * price.  Off TP only, as the lane fits are (a group's ranks must price alike). */
        if (ok && headed && !e->tp) lane_cost_fit_observe(&e->draft_step_cost, m, (now_sec() - step_t0) * 1e3);
        /* the priced depth: the live chains that pay the next lockstep step between them, each its row, and that
         * fewer drafts than the shared verify carries already outrank (pulsar_spec_chain_step) */
        if (ok) {
            float run[PULSAR_QWEN_HEAD_ROWS_MAX];
            uint32_t ahead[PULSAR_QWEN_HEAD_ROWS_MAX], at[PULSAR_QWEN_HEAD_ROWS_MAX], nl = 0;
            bool go[PULSAR_QWEN_HEAD_ROWS_MAX];
            for (uint32_t i = 0; i < n; i++) {
                if (!c[i].live) continue;
                run[nl] = c[i].run;
                ahead[nl] = 0;
                for (uint32_t k = 0; k < n_drafted; k++) ahead[nl] += drafted[k] >= c[i].run ? 1u : 0u;
                at[nl++] = i;
            }
            (void)pulsar_spec_chain_step(&price, run, ahead, nl, go);
            for (uint32_t x = 0; x < nl; x++) c[at[x]].live = go[x];
        }
    }
    if (!ok) {
        snprintf(err, errlen, "%s: the MTP draft chain failed (see the log)", PULSAR_QWEN_ARCH);
        return false;
    }
    return true;
}

/* The single lane's draft: the chain into a record, the record stamped into the bank's pendings. */
static uint32_t mtp_draft(pulsar_session *s, int next_base, bool, float temperature, int top_k, float top_p,
                          float min_p, uint64_t *rng) {
    spec_redraft_req rec;
    memset(&rec, 0, sizeof rec);
    pulsar_spec_draft_req_init(s, &rec, next_base, temperature, top_k, top_p, min_p);
    char err[200];
    uint32_t keep = 0;
    spec_redraft_req *req = &rec;
    const uint32_t b = s->states->live_bank;
    if (mtp_draft_chains(s, &b, &rng, &req, 1, err, sizeof err)) {
        pulsar_spec_redraft_stamp(s, &rec);
        keep = rec.keep;
    } else {
        fprintf(stderr, "pulsar: %s (the round takes a plain step next)\n", err);
        pulsar_spec_drop_pendings(&s->spec);
    }
    return keep;
}

/* The batched lane's draft: every bank's chain in lockstep (mtp_draft_chains), PULSAR_QWEN_HEAD_ROWS_MAX banks at a
 * time. */
static int mtp_draft_batch(pulsar_session *s, pulsar_spec_round **rounds, const uint32_t *banks, uint64_t **rngs, int n,
                           char *err, size_t errlen) {
    uint32_t b[PULSAR_QWEN_HEAD_ROWS_MAX];
    uint64_t *r[PULSAR_QWEN_HEAD_ROWS_MAX];
    spec_redraft_req *q[PULSAR_QWEN_HEAD_ROWS_MAX];
    uint32_t m = 0;
    for (int i = 0; i <= n; i++) {
        if (i < n) {
            spec_redraft_req *rq = &rounds[i]->redraft;
            if (!rq->valid || rq->n_draft == 0) continue;
            if (banks[i] >= s->states->n_banks) {
                snprintf(err, errlen, "%s: the draft's bank %u is outside the pool", PULSAR_QWEN_ARCH, banks[i]);
                return -1;
            }
            b[m] = banks[i];
            r[m] = rngs[i];
            q[m++] = rq;
        }
        if (m == PULSAR_QWEN_HEAD_ROWS_MAX || (i == n && m > 0)) {
            if (!mtp_draft_chains(s, b, r, q, m, err, errlen)) return -1;
            for (uint32_t k = 0; k < m; k++) q[k]->done = true;
            m = 0;
        }
    }
    return 0;
}

/* The MTP drafter's numbers.  Swept on sparky over three code prompts (2026-09-29, L270): depth 4 with a
 * confidence stop at 0.7 gave 58.7 / 58.9 / 59.2 tok/s vs 53.4 for a fixed 3; TensorFold's own 6 / 0.6 was 53.6
 * here -- a verify row costs more on this engine (4 rows read up to 40 distinct experts).  The adaptive depth
 * controller (spec_depth.h) LOST to the fixed depth (52-59 vs 59.5-60.4 tok/s at tau 0.6-0.8): the in-round stop
 * already adapts inside every round, and the rule's down-signal fires on chains that stopped early by design.
 * L284: dropping the stopping draft instead of verifying it cost 4-11% here (code greedy / chat greedy / chat
 * T0.7, 3 reps median: 49.1 / 49.8 / 45.6 tok/s at 0.7, 50.9 / 50.6 / 47.4 at 0.5, vs 54.9 / 52.2 / 50.6
 * verified), which is why the shared rule verifies it.  L288 (2026-10-11, work/qwen-draft step 3, U-e4-d5, tau 0.7,
 * serve_matrix 512 tok x3, code / codex / prose / story at T0 and T0.7): depth 4 / 5 / 6 summed 504 / 498 / 500 tok/s
 * over the eight single-stream cells (code greedy 77.8 / 79.8 / 79.8 at 3.88 / 4.29 / 4.48 tokens a round; codex
 * and story T0.7 lose 4-8 % deeper), 2 streams 86.7 / 83.7 / 83.8, 4 streams 101.4 / 102.5 / 101.7: a deeper
 * chain buys tokens on code but each verify row costs more than it returns elsewhere, so 4 stays.  L288
 * (2026-10-12, sdet accept rule, td305 + U-e4-d5): this drafter accepts as well as TensorFold's on the same tokens
 * (teacher-forced, tools/qwen/accept); the deeper chains TF runs pay there because a draft costs 3.43 ms against
 * 4.45 here (verify row 2.49 + MTP step 1.74, 0.98 of it this draft head's MXFP8 read).  Depth 6 vs 4: single
 * stream -1..+4 %, 4 / 8 streams -3 % on both models.  L288 (2026-10-13, tf-gap pass 2): the chain runs to the
 * session's depth while its survival pays one more MTP step and its verify row at the lane's yield and the shared
 * verify width still has room for it (pulsar_spec_chain_price), the stop rule beside it; the depth is the shared
 * adaptive controller's (spec_depth.h) between 4 and 7 -- up a level after a round that converted all its depth but
 * one draft (up_slack 1: replayed, code 99.7 against 99.0 when only a whole chain climbs, the chat start 76.1 vs
 * 76.5), down after one that converted less than half.  Replayed on the unstopped depth-8 traces (td305; the 2026-10-13
 * costs: base 27.5, row 2.4, step 1.3 ms) against a fixed 7: the chat reply's first 18 rounds 76.5 vs 70.6 tok/s
 * (fixed 4: 77.3 -- TensorFold's 64-token chat fixture lives there), the whole prose reply 72.4 vs 72.9, code
 * 99.0 vs 99.8 and 75.3 vs 74.6; the confidence is miscalibrated early in a reply (drafts at 0.999 rejected) and no
 * survival price separates those rounds, a depth earned by full rounds does.  The L270 loss above was the
 * controller at a fixed 4 with a 2.7 ms draft step.  A copy chain (a repeat of the bank's history) runs to
 * depth_max, the verify capture, and does not move the depth. */
static const pulsar_spec_depth_policy k_mtp_depth = {4, 7, 0.0f, 0, 0.0f, 0u, 0u, 1u};

const pulsar_drafter_ops k_mtp_drafter = {
    /* .name          = */ "MTP",
    /* .depth         = */ 4u,   /* where a session starts; k_mtp_depth moves it 4..7 (7: the L288 sweep's best code
                                  * cell, 97.5 tok/s against 96.2 at 8 and 94.5 at 6) */
    /* .depth_max     = */ PULSAR_QWEN_SPEC_DRAFT_MAX,
    /* .tau           = */ 0.7f,
    /* .adapt         = */ &k_mtp_depth,
    /* .prime         = */ NULL,   /* the trunk's stacks are the conditioning; the prefill parks the last one */
    /* .absorb        = */ NULL,   /* a round's rows together: the kept rows' MTP rows are one PREFILL step */
    /* .absorb_banked = */ mtp_absorb_banked,
    /* .draft         = */ mtp_draft,
    /* .harvest       = */ NULL,
    /* .draft_batch   = */ mtp_draft_batch,
};

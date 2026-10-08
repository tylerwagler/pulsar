#include "pulsar_engine_internal.h"
#include "pulsar_nvtx.h"
#include "spec_depth.h"
#include "spec_internal.h"



/* --- L107 adaptive draft depth -------------------------------------------
 * The 2026-08-25 depth sweep measured opposite optima per regime (prose 2,
 * structured 5; the shipped static 3 loses ~5%/~9.5% respectively), and the
 * conf-head calibration run measured the head monotone in both regimes with
 * conf>=0.9 -> 1.000 realized accept. Post-draft conf-sched trimming cannot
 * capture this (draft cost is paid before the trim; the sweep ran WITH the
 * trimmer on), so depth itself moves: +/-1 per round in spec_round_end.
 *   UP:   the whole drafted chain was verified AND accepted (commit == depth,
 *         which implies the trimmer kept everything) and the tail position's
 *         confidence clears SPEC_DEPTH_CONF_UP (calibrated >=0.82 accept) --
 *         the drafter was not the bottleneck this round, so probe deeper.
 *   DOWN: less than half the drafted depth converted (2*commit < depth) --
 *         drafting work is outrunning acceptance, back off.
 * Bounds [SPEC_DEPTH_MIN, SPEC_DEPTH_MAX] are the sweep's measured range;
 * depth 6 lost on BOTH regimes, so probing past it is priced as pure waste.
 * Distribution-preserving by construction (verification is exact at any
 * depth); NOT byte-identical on greedy prose -- verify-batch width shifts
 * accumulation ~1 ULP, the same known-flip class as conf-sched itself. */
uint32_t pulsar_spec_cur_depth(const pulsar_session *s) {
    int d = s->spec.spec_adaptive_depth;
    if (d <= 0) d = spec_drafter(s) ? (int)spec_drafter(s)->depth_default(s->engine) : 0;
    if (d < 1) d = 1;
    if (d > 16) d = 16;
    return (uint32_t)d;
}

/* The target's undo of a round: the snapshot back on the installed bank, then released. */
static void spec_round_undo(pulsar_session *s, pulsar_spec_round *r) {
    if (spec_target(s)->restore) (void)spec_target(s)->restore(s, r);
    if (spec_target(s)->release) spec_target(s)->release(s, r);
}

/* The drafter's context advances by a committed verify row -- now, or in the batched lane's ONE pass
 * after every bank's walk (seed_defer, L260: the row is recorded against the installed bank). */
static bool spec_absorb_defer(pulsar_session *s, uint32_t row, int32_t next) {
    if (s->seed_defer.n >= PULSAR_SPEC_LOGITS_ROWS + 1) return false;
    s->seed_defer.src_row[s->seed_defer.n] = row;
    s->seed_defer.bank[s->seed_defer.n] = pulsar_session_live_bank(s);
    s->seed_defer.next_tok[s->seed_defer.n] = next;
    s->seed_defer.n++;
    return true;
}

/* The round's committed rows [row0, row0 + n) go to the drafter: recorded for the batched lane's one
 * pass after every bank's walk (seed_defer, L260), else row by row (a drafter with a per-row absorb,
 * DSpark), else as one call (a drafter that pairs rows with their successors, Qwen's MTP). */
static bool spec_absorb_rows(pulsar_session *s, uint32_t row0, const int32_t *next, uint32_t n) {
    bool ok = true;
    if (s->seed_defer.active) {
        for (uint32_t m = 0; ok && m < n; m++) ok = spec_absorb_defer(s, row0 + m, next[m]);
        return ok;
    }
    if (spec_drafter(s)->absorb) {
        for (uint32_t m = 0; ok && m < n; m++) ok = spec_drafter(s)->absorb(s, row0 + m, next[m]);
        return ok;
    }
    uint32_t rows[PULSAR_SPEC_LOGITS_ROWS + 1], banks[PULSAR_SPEC_LOGITS_ROWS + 1];
    const uint32_t bank = pulsar_session_live_bank(s);
    for (uint32_t m = 0; m < n; m++) { rows[m] = row0 + m; banks[m] = bank; }
    return spec_drafter(s)->absorb_banked(s, rows, banks, next, n);
}



/* --- Terminal yield-quench controller (spec-decode Item 4) ---------------
 * Controller design after the Entrpi ds4 yield quench (v0.1.1, MIT): per
 * request, every fused spec step accrues debt = breakeven yield minus
 * realized yield; when the request has provably lost more than a small
 * budget of plain tokens to speculation AND its recent yield is still below
 * breakeven, speculation turns off for the REMAINDER of that request
 * (terminal; a new request re-arms).
 *
 * Breakeven derivation (calibrated 2026-07-17 against per-step
 * PULSAR_DSPARK_STATS traces of the production v5mx serving path — 2585 steps
 * across prose/structured x greedy/T1.0, least-squares, resid rms 7.2 ms;
 * offline method after Entrpi's dspark_trace_replay, tool:
 * temp/quench/quench_replay.py):
 *   fused spec step   ~= FLAT + ROW * n_batch   milliseconds
 *     FLAT: pooled fit 101.6 ms, SHIPPED 95.7 ms — see the greedy-fit
 *           paragraph below for why the lower bound is the one compiled in
 *           (batched projections + shared + drafter dense/markov + sampled-q
 *           readback/dist walk — flat in verify rows; the old 53.6 ms nsys
 *           figure was the draft=3 build before temperature-matched drafting)
 *     ROW:  pooled fit 19.15 ms, SHIPPED 18.37 ms (marginal verify row —
 *           bandwidth-bound; matches the 2026-07-09 nsys 19.7 ms/row audit)
 *   plain decode token = PLAIN(pos), piecewise-linear through the measured
 *     served-plain depth table (2026-07-15, medians of 3): 59.7 ms @0.3k,
 *     67.3 @2.3k, 68.7 @9.3k, 74.5 @38k. Depth-dependence matters: spec step
 *     cost is ~flat in depth while plain slows, so a scalar PLAIN would
 *     overprice speculation exactly in the deep cells where it wins most
 *     (e.g. sweep-prose greedy @2.3k, +11.7%).
 * The breakeven yield of a step that verified K drafts at frontier pos is
 *   guard = (FLAT + ROW*(1+K)) / PLAIN(pos)     [plain tokens]
 * ~2.9 at full depth shallow, ~2.0 for a draft-only step. A request whose
 * committed tokens/step run below guard would have been faster plain.
 * Charging the ACTUAL n_batch (post conf-sched trim) rather than a scalar
 * guard prices exactly the steps the trimmer already shortened; committed
 * likewise is the post-trim realized yield. Measured operating points:
 * t2x prose y~2.05 vs guard ~2.8 (loses -> quench), structured y~5.1-5.3 vs
 * guard ~3.5 (wins by >1.5 -> never quenches).
 *
 * FLAT/ROW are the greedy-prose fit (the LOWER bound across the four
 * calibrated cells; the sampled cells fit ~6-11 ms higher FLAT). Deliberate:
 * underpricing the spec step biases against quenching, which keeps the
 * borderline deep greedy cells (sweep prose @2.3k: y~3.2 vs guard 3.06,
 * wins +11.7% measured) strictly on the no-quench side of the model.
 *
 * WARMUP: the first steps of every request are drafter pipeline fill —
 * n_batch ramps 1->2->3 with near-zero commits while the pendings build and
 * the prompt window seeds (a one-time ~40 ms not in the step model). That is
 * a fixed startup cost every request pays, not evidence about proposal
 * quality — charging it was measured (2026-07-17, first Load-2 gate run) to
 * book ~4.9 debt by step 6 at 2.3k ctx and spuriously quench the WINNING
 * sweep-prose cell (16.6 -> 14.8 t/s). The controller therefore ignores the
 * first PULSAR_QUENCH_WARMUP steps entirely, and MINEV=8 (Entrpi's replay
 * default) delays the verdict until the EWMA reflects steady state.
 * Re-validated offline over all 17 Load-1 traces + deep synthetic steady
 * states: losers (shallow y~2.05, deep y~2.5) fire at tokens ~11-25;
 * winners (struct, deep y>=3.2) never fire.
 *
 * Debt is deliberately NOT clamped below (matches Entrpi's shipped default):
 * unclamped, debt is exactly the request's NET plain-token-equivalents lost,
 * so the quench fires iff the request is genuinely >= BUDGET behind plain —
 * banked credit is real measured savings being spent, not optimism. Entrpi
 * measured the zero-clamped variant false-quenching long bursty winners, and
 * our offline replay selftest reproduces the same false quench for any
 * finite credit cap on a net-positive bursty request. Budget = 4 plain-token
 * equivalents (~250 ms): large enough that per-step yield variance on a
 * winning request cannot cross it (compounded with the EWMA and MINEV
 * conditions), small enough that a 400-token losing request recovers nearly
 * all of the loss.
 *
 * The step side of the guard is MEASURED (L263): pulsar_engine::spec_cost is
 * an exponentially weighted least-squares fit of a decode round's wall time
 * on the rows its forward carried, round = flat + row * rows, fed by the
 * server's round loop (the single lane off TP feeds its own steps).  Nothing
 * about a deployment is compiled in -- the 2026-07/09 refits that lived here
 * (FLAT 95.7 -> 57.0 -> 45.0, ROW 18.37 -> 7.17, a plain-by-depth table) went
 * stale with every kernel landing and could only be right for one machine.
 * The plain side needs no table: a bank that stops drafting still rides a
 * round with ONE row, so plain costs the same round at n = 1, at whatever
 * depth is being served.  A round shared by B banks shares its flat cost, so
 * the guard for a bank with n rows is (flat/B + row*n) / (flat/B + row);
 * B = 1 is the single lane's (flat + row*n) / (flat + row).  Until the fit
 * is valid the controller does not price at all: no number, no quench.
 * The decision therefore reads the machine's clock through the fit (the
 * quench point is not fixed for a given token stream); both paths sample the
 * exact target distribution, so only speed is at stake.  The remaining
 * constants are the controller's own (EWMA weight, warm-up, minimum
 * evidence, budget), not a deployment's. */
#define PULSAR_QUENCH_ALPHA      0.125f   /* EWMA weight (Entrpi default) */
#define PULSAR_QUENCH_WARMUP     3u      /* ramp steps charged to no one (below) */
#define PULSAR_QUENCH_MINEV      8u      /* min spec steps before quench */
#define PULSAR_QUENCH_BUDGET     4.0f    /* plain-token equivalents */

/* The break-even yield for a bank with `n_batch` rows in a round shared by
 * `banks` banks (see above).  Integer microseconds in, so every rank of a TP
 * group computes the same value from the same wire terms. */
static float spec_quench_guard(const pulsar_spec_cost_fit *c, int banks, uint32_t n_batch) {
    const float share = (float)c->flat_us / (float)(banks > 0 ? banks : 1);
    const float row = (float)c->row_us;
    return (share + row * (float)n_batch) / (share + row);
}

/* L263: one observation into the fit.  Decay 255/256 per round (a window of
 * ~256 rounds, 15-25 s of decode: at c1 the rows spread only 2..6, so a
 * 64-round window let the slope swing 1..6 ms/row between adjacent windows
 * -- the pair, 2026-10-05); valid once 16 rounds are in, the row counts have
 * spread (an EW variance of at least 1/4: adaptive depth and concurrency
 * move them every few rounds) and both terms are positive -- anything else
 * is noise, not a price. */
#define SPEC_COST_DECAY   (1.0 - 1.0 / 256.0)
#define SPEC_COST_MIN_OBS 16u
#define SPEC_COST_MIN_VAR 0.25
void spec_cost_fit_observe(pulsar_spec_cost_fit *f, uint32_t rows, double ms) {
    if (!f || rows == 0 || !(ms > 0.0)) return;
    const double x = (double)rows, y = ms;
    f->w   = f->w   * SPEC_COST_DECAY + 1.0;
    f->sx  = f->sx  * SPEC_COST_DECAY + x;
    f->sy  = f->sy  * SPEC_COST_DECAY + y;
    f->sxx = f->sxx * SPEC_COST_DECAY + x * x;
    f->sxy = f->sxy * SPEC_COST_DECAY + x * y;
    f->n++;
    f->valid = false;
    const double mx = f->sx / f->w, my = f->sy / f->w;
    const double vx = f->sxx / f->w - mx * mx;
    const double cxy = f->sxy / f->w - mx * my;
    if (f->n < SPEC_COST_MIN_OBS || vx < SPEC_COST_MIN_VAR) return;
    const double row = cxy / vx, flat = my - row * mx;
    if (!(row > 0.0) || !(flat > 0.0)) return;
    f->flat_us = (int32_t)(flat * 1000.0 + 0.5);
    f->row_us = (int32_t)(row * 1000.0 + 0.5);
    f->valid = f->flat_us > 0 && f->row_us > 0;
}

/* All-zero == armed, matching the xcalloc'd session. */
void spec_quench_reset(pulsar_session *s) {
    s->spec.spec_quench_debt = 0.0f;
    s->spec.spec_quench_ewma = 0.0f;
    s->spec.spec_quench_steps = 0;
    s->spec.spec_quenched = false;
}

/* A request boundary: the lookahead goes, the quench re-arms. */
void spec_lookahead_reset(pulsar_session *s) {
    s->spec.spec_carry_valid = false;
    pulsar_spec_drop_pendings(&s->spec);
    spec_quench_reset(s);
}












/* Fused DSpark loop (P2, PULSAR_DSPARK_FUSED=1): ONE batched target forward per
 * step instead of Step-1 decode + separate verify. The forward runs over
 * [first_token, pending_drafts...] (drafts made LAST step -- EAGLE pipeline
 * inversion), so position 0 is the base decode, positions 1..K verify the
 * drafts, and the batched anchor-hidden capture gives the drafter its
 * conditioning at whatever position ends up last-accepted. Drafting for the
 * NEXT step then conditions on the hidden that PRODUCED the next base token
 * (matching the reference generate.py forward_spec dataflow; the deleted
 * per-round loop conditioned on the hidden AFTER re-evaluating the base token
 * -- a one-position train/inference mismatch this path removed).
 * Greedy-only (generate.cpp gates on temperature<=0).
 * Partial/zero accepts restore the frontier and replay the committed prefix
 * (Stage A; the Stage-B transactional state removes the replay). */

/* plan-34 inc 6 (step 2a): the accept walk, extracted PURE from the fused
 * loop. Consumes logit rows through `read_row` (row index is round-local:
 * 0..K-1 are the draft rows), decides the accepted prefix and -- for the
 * sampled rule -- draws the rejecting position's residual carry. Touches NO
 * session state beyond the sampling scratch; this is the function the
 * batched lane calls once per bank over a SHARED forward's rows, with
 * read_row pointing into that batch's ALL_ROWS logits at the bank's offset.
 * The rules are verbatim from the fused loop (greedy argmax match; sampled
 * p/q with the draft-time-params q rebuild -- see the walk comment there). */

/* L149 phase 2: build row `row`'s target distribution from the compact
 * prefilter block when the request is in the sparse min-p contract and the
 * row's candidate set fit the cap. false = caller reads the full row. */
static bool spec_compact_dist(const int32_t *compact, uint32_t row,
                              float temperature, int top_k, float top_p, float min_p,
                              pulsar_sample_scratch *scratch, pulsar_sample_dist *out) {
    if (!compact || row >= PULSAR_SPEC_LOGITS_ROWS ||
        !(temperature > 0.0f && top_k <= 0 && top_p == 1.0f &&
          min_p >= PULSAR_SAMPLE_SPARSE_MINP_MIN && min_p <= 1.0f))
        return false;
    const int32_t *h = compact + (size_t)row * PULSAR_DSPARK_PREFILTER_ROW_I32;
    const uint32_t n = (uint32_t)h[0];
    if (n == 0 || n > PULSAR_DSPARK_PREFILTER_CAP) return false;
    float max_logit;
    memcpy(&max_logit, &h[2], sizeof(max_logit));
    return pulsar_sample_dist_build_prefiltered(h + 3,
                                                (const float *)(h + 3 + PULSAR_DSPARK_PREFILTER_CAP),
                                                n, max_logit, temperature, min_p,
                                                scratch, out) != 0;
}

static int spec_accept_walk(pulsar_session *s,
                            pulsar_spec_row_read_fn read_row, void *read_ud,
                            const int32_t *compact, uint32_t row0,
                            const int *row_tops,
                            const int32_t *pend, uint32_t K, bool pend_sampled,
                            float temperature, int top_k, float top_p, float min_p,
                            uint64_t *rng, int *out_carry_tok) {
    int commit = 0;
    int carry_tok = -1;
    if (temperature <= 0.0f || K == 0) {
        while (commit < (int)K && row_tops[commit] == (int)pend[commit]) commit++;
    } else {
        if (!s->spec_row_scratch)
            s->spec_row_scratch = (float *)xmalloc((size_t)spec_vocab(s) * sizeof(float));
        float *row_logits = s->spec_row_scratch;
        while (commit < (int)K) {
            pulsar_sample_dist dist;
            if (!spec_compact_dist(compact, row0 + (uint32_t)commit, temperature, top_k,
                                   top_p, min_p, &s->sample_scratch, &dist)) {
                if (!read_row(read_ud, (uint32_t)commit, row_logits) ||
                    !pulsar_sample_dist_build(row_logits, spec_vocab(s), temperature, top_k,
                                              top_p, min_p, &s->sample_scratch, &dist)) {
                    *out_carry_tok = -1;
                    return -1;
                }
            }
            const bool accepted_row = pend_sampled
                ? pulsar_sample_dist_accept_pq(&dist, (int)pend[commit],
                                            s->spec.pend_q[commit], rng)
                : pulsar_sample_dist_accept(&dist, (int)pend[commit], rng);
            if (accepted_row) {
                pulsar_sample_dist_free(&dist);
                commit++;
                continue;
            }
            if (pend_sampled) {
                pulsar_sample_dist qd;
                const uint32_t qn = s->spec.pend_qn[commit];
                if (pulsar_spec_q_compact(qn)) {
                    /* L149: q_X exactly as built at draft time (drafting loop) */
                    memset(&qd, 0, sizeof(qd));
                    qd.n = qn;
                    qd.ids = (int *)xmalloc((size_t)qn * sizeof(int));
                    qd.probs = (float *)xmalloc((size_t)qn * sizeof(float));
                    memcpy(qd.ids, s->spec.pend_qids[commit], (size_t)qn * sizeof(int));
                    memcpy(qd.probs, s->spec.pend_qprobs[commit],
                           (size_t)qn * sizeof(float));
                } else if (!pulsar_sample_dist_build(s->pend_qrows +
                                                         (size_t)commit * spec_vocab(s),
                                                     spec_vocab(s), s->spec.pend_temp,
                                                     s->spec.pend_top_k,
                                                     s->spec.pend_top_p,
                                                     s->spec.pend_min_p,
                                                     &s->sample_scratch, &qd)) {
                    /* the stored q row refused: the proposal this draft was
                     * drawn from cannot be rebuilt, so the walk fails */
                    pulsar_sample_dist_free(&dist);
                    *out_carry_tok = -1;
                    return -1;
                }
                carry_tok = pulsar_sample_dist_draw_residual(&dist, &qd,
                                                          &s->sample_scratch, rng);
                pulsar_sample_dist_free(&qd);
            } else {
                carry_tok = pulsar_sample_dist_draw_excluding(&dist, (int)pend[commit], rng);
            }
            pulsar_sample_dist_free(&dist);
            break;
        }
    }
    *out_carry_tok = carry_tok;
    return commit;
}






/** inc-6: one speculative ROUND's state, threaded between round_begin (rows
 * assembled, frontier snapshotted, checkpoint pushed), the verify forward
 * (classic in the fused loop; the SHARED decode_mixed ALL_ROWS forward in
 * the batched lane), and round_end (walk, state, emit, redraft). */
/* L108 P2: a deferred device draft is finished into the installed bank's pendings before anything reads
 * them (round_begin, the bank save, the conf peek); the drafter's business, a no-op without one. */
void pulsar_session_spec_chain_harvest(pulsar_session *s) {
    if (spec_drafter(s) && spec_drafter(s)->harvest) spec_drafter(s)->harvest(s);
}

/* Assemble the round: load/guard the pendings, seed the prompt window if
 * fresh, snapshot the frontier, push first_token + pendings onto the
 * checkpoint. Body verbatim from the fused loop; the locals it declared are
 * now reference-bound round fields so the batched lane can run one round
 * per bank around a shared forward. Returns 0 or -1 (snapshot failure,
 * session poisoned) exactly as before. */
static int spec_round_begin(pulsar_session *s, int first_token,
                            int max_tokens, int accepted_cap,
                            float temperature, int top_k, float top_p, float min_p,
                            pulsar_spec_round *r,
                            char *err, size_t errlen) {
    if (first_token < 0 || (uint32_t)first_token >= spec_vocab(s)) {
        /* -1 is what pulsar_session_spec_next_base returns when the sampler
         * refused the row; the base token feeds the embedding gather, so a
         * non-id must stop here. */
        snprintf(err, errlen, "spec round: base token %d is not a vocab id", first_token);
        return -1;
    }
    {
        /* the round is reused across rounds; its redraft record's lazily-owned
         * fallback rows (L150) survive the reset, everything else is cleared */
        float *qrows = r->redraft.qrows;
        void *snap = r->snap;
        memset(r, 0, sizeof(*r));
        r->redraft.qrows = qrows;
        r->snap = snap;
        r->base = (int32_t)first_token;
    }
    /* the target may choose the step's logits readback from the rounds that ride it (L149 / L219) */
    if (spec_target(s)->round_note) spec_target(s)->round_note(s, temperature, top_k, top_p, min_p);
    uint32_t &K = r->K;
    uint32_t &n_batch = r->n_batch;
    int &saved_len = r->saved_len;
    bool &pend_sampled = r->pend_sampled;
    int32_t (&pend)[16] = r->pend;
    float (&pend_conf)[16] = r->pend_conf;
    (void)pend_conf; (void)pend_sampled; (void)n_batch;
    /* Pending drafts continue from the greedy base we predicted last step; if
     * the caller committed something else (tool injection, sampling change),
     * they are stale. */
    pulsar_session_spec_chain_harvest(s);   /* L108 P2 */
    K = s->spec.n_pend;
    if (K > 16u) K = 16u;
    if (K && s->spec.pend_base != (int32_t)first_token) K = 0;
    /* Position guard — ACCEPTANCE, not exactness. The drafts were conditioned on
     * the old position; pend_base is a token VALUE, so a plain eval
     * that advances the session (tool injection, </think> recovery) followed by
     * a first_token that happens to collide with it would otherwise resurrect
     * them. Dropping them is a throughput choice: a draft conditioned on the
     * wrong position is near-worthless and would just burn a verify row. It is
     * NOT what keeps the output exact — both accept rules are proposal-agnostic
     * (see the walk below), so q_oldpos is still exactly the distribution the
     * draft was drawn from and the rule still yields p at the new position.
     * Mirrors carry_pos_match. (checkpoint.len is still the drafting-step value
     * here: this runs before the first_token push below.) */
    if (K && s->spec.pend_pos != (int32_t)s->checkpoint.len) K = 0;
    /* Params guard — also acceptance, not exactness. Drafts drawn under params X
     * and verified under params Y are still verified EXACTLY: the residual
     * rebuilds q under the stored params X (see the walk below), so the accept
     * denominator and the residual name one proposal q_X, and the p/q rule
     * returns exactly p_Y for any q. What a param change costs is acceptance —
     * q_X is a poor proposal for p_Y — so we drop them and draft afresh.
     * Mirrors the spec_carry_* params guard in pulsar_session_generate_speculative.
     *
     * The same holds for argmax drafts (pend_sampled == false), which
     * carry no q: a temp<=0 -> temp>0 change cannot misroute them, because the
     * walk picks its rule from pend_sampled, not from the live temperature, so
     * they still meet the deterministic rule — itself exact for an arbitrary
     * proposal. The guard just keeps a badly-matched proposal from wasting a
     * row. */
    const bool pending_params_match =
        s->spec.pend_temp == temperature && s->spec.pend_top_k == top_k &&
        s->spec.pend_top_p == top_p && s->spec.pend_min_p == min_p;
    if (K && !pending_params_match) K = 0;
    /* Proposal rule the pendings were drafted under; the verify walk must apply
     * the matching rule (see pend_sampled). */
    pend_sampled = K ? s->spec.pend_sampled : false;
    if ((int)K > accepted_cap - 1) K = accepted_cap > 1 ? (uint32_t)(accepted_cap - 1) : 0;
    if ((int)K > max_tokens - 1) K = max_tokens > 1 ? (uint32_t)(max_tokens - 1) : 0;
    for (uint32_t i = 0; i < K; i++) pend[i] = s->spec.pend[i];
    /* Conf carried unconditionally (L107 controller). */
    for (uint32_t i = 0; i < K; i++) pend_conf[i] = s->spec.pend_conf[i];
    pulsar_spec_drop_pendings(&s->spec);
    s->spec.spec_carry_valid = false;
    n_batch = 1u + K;

    /* a fresh drafter context primes itself from the prompt's captured features (DSpark: the last <= 128
     * prompt positions replayed into its ring); a refusal fails the round with the target untouched */
    if (spec_drafter(s)->prime && spec_drafter(s)->prime(s, err, errlen) != 0) return -1;

    /* the target's snapshot of what the verify rows will mutate (a family whose verify captures its
     * own undo state inside the forward has none) */
    if (spec_target(s)->snapshot && !spec_target(s)->snapshot(s, r, err, errlen)) {
        s->checkpoint_valid = false;
        return -1;
    }

    saved_len = s->checkpoint.len;
    token_vec_push(&s->checkpoint, first_token);
    for (uint32_t i = 0; i < K; i++) token_vec_push(&s->checkpoint, (int)pend[i]);
    return 0;
}

/* Finish the round over an already-run verify forward: accept walk, EOS
 * clamp, metrics, quench, logits refresh, carry, state seed/rollback, emit,
 * pendings redraft. Body verbatim from the fused loop with three seams for
 * the batched lane: `read_row`/`read_ud` name the forward's logit rows
 * (classic: spec_logits; batched: the shared ALL_ROWS block at the bank's
 * offset), and `row0` is this round's first row in the forward's
 * capture/comp-save buffers (classic: 0). Owns r->frontier's lifetime on
 * every path. Returns tokens emitted, or -1 (session poisoned). */
static int spec_round_end(pulsar_session *s, pulsar_spec_round *r,
                          int first_token,
                          float temperature, int top_k, float top_p, float min_p,
                          uint64_t *rng,
                          const pulsar_spec_rows *src, uint32_t row0,
                          bool defer_redraft,
                          double t0,
                          const int32_t *forced_truth,
                          int *accepted, int accepted_cap,
                          char *err, size_t errlen) {
    pulsar_engine *e = s->engine;
    static int dspark_stats_env = -1;
    const int dspark_stats = gpu_graph_env_flag("PULSAR_DSPARK_STATS", &dspark_stats_env);
    int n_accept = 0;
    const uint32_t K = r->K;
    const uint32_t n_batch = r->n_batch;
    const int saved_len = r->saved_len;
    const bool pend_sampled = r->pend_sampled;
    int32_t (&pend)[16] = r->pend;
    float (&pend_conf)[16] = r->pend_conf;
    int (&row_tops)[16] = r->row_tops;
    (void)pend_conf; (void)t0; (void)dspark_stats;
    /* Accept the longest prefix the target agrees with. Greedy: row i's
     * argmax must equal pend[i]. Sampled: exact speculative sampling under the
     * request's FILTERED target distribution p_i, with the rule matched to how
     * the draft was PROPOSED:
     *   - sampled proposal (the production path at temperature > 0): pend[i]
     *     was drawn from a temperature-matched q_i, so accept w.p.
     *     min(1, p_i/q_i) and on rejection draw the residual (p_i - q_i)+.
     *     Acceptance is NOT capped at p_i(mode).
     *   - argmax proposal (temperature == 0): the deterministic rule -- accept
     *     w.p. p_i(pend[i]), residual p_i with pend[i] excluded. Capped at
     *     p_i(mode).
     * The rejected row's replacement becomes the carry token. All three paths
     * yield the exact per-token target distribution. */
    int carry_tok = -1;
    int commit_rc;
    if (forced_truth) {
        /* TEACHER-FORCED end (gate-facing, L182): the walk is replaced by
         * equality against the true continuation -- draft i is accepted iff it
         * IS truth[i], the carry is the first true token the drafts did not
         * cover, and no draw is made.  The context the redraft conditions on
         * is the corpus, so the drafter's next proposal is a deterministic
         * function of (model, drafter, context).  Everything after this point
         * -- controllers, metrics, the trim, the redraft -- runs as in
         * production. */
        commit_rc = 0;
        while ((uint32_t)commit_rc < K && pend[commit_rc] == forced_truth[commit_rc]) commit_rc++;
        carry_tok = (int)forced_truth[commit_rc];
    } else {
        commit_rc = spec_accept_walk(s, src->read, src->ud, src->compact, row0,
                                     row_tops, pend, K, pend_sampled,
                                     temperature, top_k, top_p, min_p,
                                     rng, &carry_tok);
    }
    s->spec.qrows_n = 0u;   /* L260: the walk was the last reader of this round's q rows */
    if (commit_rc < 0) {
        s->checkpoint.len = saved_len;
        spec_round_undo(s, r);
        snprintf(err, errlen, "DSpark sampled-accept logits readback failed");
        s->checkpoint_valid = false;
        return -1;
    }
    int commit = commit_rc;

    /* Ghost-token guard: never commit drafts PAST an accepted EOS. The
     * emission loop below stops at EOS, but the checkpoint trim uses commit —
     * without this clamp the bank history keeps invisible tokens after the
     * stop: the exact-frontier warm gate misses next turn, and the
     * thinking-live continuation embeds the ghosts into the next prompt. The
     * EOS row itself stays committed (it was evaluated in the batch, matching
     * the batched lane's convention); hit_eos below invalidates the carry, so
     * nothing ever samples from a post-EOS row.  "EOS" is the family's whole
     * stop set (pulsar_token_is_stop; L284: the round cut only at the one eos
     * id, so a Qwen draft of <|endoftext|> committed past it and the server
     * invalidated the bank for the ghost tail). */
    if (pulsar_token_is_stop(e, first_token)) {
        commit = 0;
    } else {
        for (int i = 0; i < commit; i++) {
            if (pulsar_token_is_stop(e, (int)pend[i])) { commit = i + 1; break; }
        }
    }

    /* Prometheus /metrics spec-decode counters (server /metrics endpoint). The
     * base token is always emitted; K drafts were verified this step and the
     * accepted prefix is [0,commit). num_drafts counts draft rounds only.
     * verified_per_pos is the per-position ATTEMPT count: position i was put to
     * the target iff i < K, which under the L107 adaptive depth is not every
     * round. Without it a scraper can only form accepted[i]/num_drafts, a joint
     * prefix probability that the depth schedule moves; with it the per-position
     * rate is accepted[i]/verified[i] at any schedule (pulsar.h). */
    e->spec_gen_tokens += 1u + (uint64_t)commit;
    s->spec.spec_gen_tokens += 1u + (uint64_t)commit;
    if (K > 0) {
        e->spec_draft_tokens += K;
        e->spec_accepted_tokens += (uint64_t)commit;
        e->spec_num_drafts += 1u;
        for (uint32_t i = 0; i < K && i < 16u; i++) e->spec_verified_per_pos[i]++;
        for (int i = 0; i < commit && i < 16; i++) e->spec_accepted_per_pos[i]++;
        s->spec.spec_draft_tokens += K;
        s->spec.spec_accepted_tokens += (uint64_t)commit;
        s->spec.spec_num_drafts += 1u;
    }

    /* L107 adaptive draft depth (constants + rationale at spec_cur_depth).
     * Runs BEFORE the redraft below so the next chain is drafted at the new
     * depth. commit == depth implies the trimmer kept the whole chain AND the
     * target accepted all of it (commit <= K <= depth always). A tail conf of
     * -1 (head didn't run, e.g. conf-sched disabled) passes the UP check: the
     * full-accept signal alone then drives the climb. Counts-only decision,
     * deterministic for a fixed stream, same property as yield-quench. */
    if (K > 0 && spec_target(s)->depth) {
        /* the rule is spec_depth.h's; the numbers are the target's (pulsar_spec_target_ops::depth; a
         * NULL policy = a fixed depth) */
        const uint32_t depth = pulsar_spec_cur_depth(s);
        pulsar_spec_depth_state ds = {s->spec.spec_depth_down_forgiven, s->spec.spec_depth_rounds_since_up,
                                      s->spec.spec_depth_climb_cooldown};
        const int next = pulsar_spec_depth_next(spec_target(s)->depth, &ds, depth, K, commit, pend_conf);
        s->spec.spec_depth_down_forgiven = ds.down_forgiven;
        s->spec.spec_depth_rounds_since_up = ds.rounds_since_up;
        s->spec.spec_depth_climb_cooldown = ds.climb_cooldown;
        if (dspark_stats && (uint32_t)next != depth)
            fprintf(stderr, "pulsar: adaptive-k depth %u -> %d (commit=%d K=%u tail=%.2f)\n",
                    depth, next, commit, K, (double)pend_conf[K - 1]);
        s->spec.spec_adaptive_depth = next;
    }

    /* Yield-quench controller update (see the constants block up top). Uses
     * the ACTUAL verify width n_batch (post conf-sched trim last step) and the
     * ACTUAL committed yield 1+commit — counts only, so the decision is
     * deterministic for a fixed stream. Once latched, generate_speculative
     * routes this request's remaining tokens down the plain-decode path and
     * the drafting block below is skipped; both paths sample the exact target
     * distribution, so quenching changes speed, never marginals. */
    if (!s->spec.spec_quenched) {
        s->spec.spec_quench_steps++;
        bool fire = false;
        if (e->spec_cost.valid && s->spec.spec_quench_steps > PULSAR_QUENCH_WARMUP) {
            const float margin = (1.0f + (float)commit) -
                                 spec_quench_guard(&e->spec_cost, s->spec.spec_round_banks, n_batch);
            s->spec.spec_quench_ewma = (1.0f - PULSAR_QUENCH_ALPHA) * s->spec.spec_quench_ewma +
                                  PULSAR_QUENCH_ALPHA * margin;
            s->spec.spec_quench_debt -= margin;   /* unclamped: NET tokens lost */
            fire = s->spec.spec_quench_steps >= PULSAR_QUENCH_MINEV &&
                   s->spec.spec_quench_ewma < 0.0f &&
                   s->spec.spec_quench_debt > PULSAR_QUENCH_BUDGET;
        }
        if (fire) {
            s->spec.spec_quenched = true;
            fprintf(stderr,
                    "pulsar: spec yield-quench pos=%d steps=%u debt=%.2f ewma=%.2f "
                    "-> plain decode for request remainder%s\n",
                    saved_len + 1 + commit, s->spec.spec_quench_steps,
                    (double)s->spec.spec_quench_debt, (double)s->spec.spec_quench_ewma,
                    "");
        }
    }

    /* Refresh s->logits to the last committed position's distribution. */
    if (!src->read(src->ud, (uint32_t)commit, s->logits)) {
        s->checkpoint.len = saved_len;
        spec_round_undo(s, r);
        snprintf(err, errlen, "DSpark fused logits readback failed");
        s->checkpoint_valid = false;
        return -1;
    }

    /* Finalize the carry token — the next base, drawn from the refreshed
     * s->logits (= row[commit]) when the walk did not already draw a residual:
     * greedy -> argmax (the old next_base); sampled full-accept -> the bonus
     * draw from the last accepted row's distribution. */
    if (carry_tok < 0) {
        if (temperature <= 0.0f) {
            carry_tok = sample_argmax(s->logits, spec_vocab(s));
        } else {
            pulsar_sample_dist bonus;
            if (pulsar_sample_dist_build(s->logits, spec_vocab(s), temperature, top_k,
                                         top_p, min_p, &s->sample_scratch, &bonus)) {
                carry_tok = pulsar_sample_dist_draw(&bonus, rng);
                pulsar_sample_dist_free(&bonus);
            }
        }
        if (carry_tok < 0) {
            /* the committed row has no drawable distribution (the sampler said
             * why, once): no carry can be honest, so the round fails */
            s->checkpoint.len = saved_len;
            spec_round_undo(s, r);
            snprintf(err, errlen, "DSpark fused: the sampler refused the carry row");
            s->checkpoint_valid = false;
            return -1;
        }
    }

    /* the drafter's conditioning features follow the last committed row through the absorbs below */
    const bool features_ready = true;
    /* The target leaves the installed bank's state at saved_len + 1 + commit (a full accept needs
     * nothing of DeepSeek; a partial one restores its snapshot and rolls the recurrent state forward
     * from the inputs saved during the verify -- bit-identical, no replay), then the drafter absorbs
     * the committed rows 0..commit from its own captured features (row j = f(h_j)), each with the
     * token that follows it: the next committed token, or the carry after the last. */
    s->checkpoint.len = saved_len;
    bool ok_state = spec_target(s)->commit(s, r, (uint32_t)commit, row0);
    if (ok_state) {
        s->checkpoint.len = saved_len + 1 + commit;
        int32_t next[PULSAR_SPEC_LOGITS_ROWS + 1];
        uint32_t n_abs = 0;
        for (int m = 0; m <= commit && m < (int)n_batch; m++) next[n_abs++] = m < commit ? pend[m] : (int32_t)carry_tok;
        ok_state = spec_absorb_rows(s, row0, next, n_abs);
    }
    if (spec_target(s)->release) spec_target(s)->release(s, r);
    if (!ok_state) {
        snprintf(err, errlen, "spec round: the state update after the walk failed");
        s->checkpoint_valid = false;
        return -1;
    }

    /* Emit first_token + accepted drafts. */
    accepted[n_accept++] = first_token;
    bool hit_eos = pulsar_token_is_stop(e, first_token);
    for (int i = 0; i < commit && n_accept < accepted_cap && !hit_eos; i++) {
        accepted[n_accept++] = (int)pend[i];
        if (pulsar_token_is_stop(e, (int)pend[i])) hit_eos = true;
    }

    /* L155 guard: the committed frontier (saved_len + 1 + commit) must equal
     * what the caller receives.  EOS is already handled -- the ghost-token
     * guard above clamps commit at an accepted EOS, so no post-EOS draft is
     * ever committed (I first read this as a live leak; it is not).  The one
     * way the two can still differ is accepted_cap binding below 1 + commit,
     * which no current caller does (the server passes its array size).  Enforce
     * the invariant anyway, where both counts are known: rewind() clamps the
     * compressor frontier and drops the carry and drafter window (right -- the
     * carry was conditioned on positions that no longer exist).  `exposed_end`
     * is an arbitrary position, so the bank is left stale (L264): a further
     * block on this session refuses until a sync restores a grid checkpoint,
     * which is the fail-closed answer for a caller that bound accepted_cap.
     * The server's tripwire (server_sched.cpp) checks the same equality after
     * every round. */
    const int exposed_end = saved_len + n_accept;
    const bool trimmed = exposed_end < s->checkpoint.len;
    if (trimmed) {
        /* rewind() drops the drafter window, so this bank's seed rows go too:
         * the deferred batch must not advance counters the rewind has reset. */
        if (s->seed_defer.active) {
            const uint32_t cur = pulsar_session_live_bank(s);
            uint32_t k = 0;
            for (uint32_t i = 0; i < s->seed_defer.n; i++) {
                if (s->seed_defer.bank[i] == cur) continue;
                s->seed_defer.src_row[k] = s->seed_defer.src_row[i];
                s->seed_defer.bank[k] = s->seed_defer.bank[i];
                k++;
            }
            s->seed_defer.n = k;
        }
        spec_target(s)->cut(s, exposed_end);
    }

    /* The carry IS the next base (already correctly distributed). Persist it
     * so the next generate_speculative call forwards it as batch position 0;
     * pre-draft the NEXT block conditioned on it. */
    const int next_base = carry_tok;
    s->spec.spec_carry_token = (int32_t)carry_tok;
    s->spec.spec_carry_valid = !hit_eos && !trimmed;
    s->spec.spec_carry_pos = (int32_t)s->checkpoint.len;
    s->spec.spec_carry_temp = temperature;
    s->spec.spec_carry_top_k = top_k;
    s->spec.spec_carry_top_p = top_p;
    s->spec.spec_carry_min_p = min_p;
    uint32_t n_draft = pulsar_spec_cur_depth(s);   /* L107: session depth, not the static engine width */
    if (n_draft > 16u) n_draft = 16u;
    if (hit_eos || trimmed || pulsar_token_is_stop(e, next_base) || n_draft == 0 || s->spec.spec_quenched) {
        /* Quenched: don't draft the next chain — the carry persisted above is
         * still the correctly-distributed next base, which the next
         * generate_speculative call consumes before routing plain.  (After an
         * L155 trim there is no carry: the generation ended at the cap.) */
        if (dspark_stats && t0 > 0.0)
            fprintf(stderr, "pulsar: dspark fused n_batch=%u committed=%d nodraft step_ms=%.1f\n",
                    n_batch, commit, (now_sec() - t0) * 1000.0);
        return n_accept;
    }

    if (defer_redraft) {
        /* L150: the batched lane drafts every bank in ONE pass after all the
         * walks (pulsar_session_spec_redraft_batch). Record what this bank's
         * redraft needs and leave the shadow with no pendings: a bank the batch
         * cannot serve simply takes a plain n=1 step next round. */
        pulsar_spec_drop_pendings(&s->spec);
        spec_redraft_req *q = &r->redraft;
        q->valid = features_ready;
        q->done = false;
        q->next_base = (int32_t)next_base;
        q->n_draft = n_draft;
        q->n_batch = n_batch;
        q->commit = commit;
        q->temperature = temperature;
        q->top_k = top_k;
        q->top_p = top_p;
        q->min_p = min_p;
        return n_accept;
    }
    const uint32_t keep = spec_drafter(s)->draft(s, next_base, features_ready, temperature, top_k, top_p, min_p, rng);
    /* t0 == 0 marks a lane whose forward ran outside this round (the batched
     * lane) -- step_ms would print the epoch, so skip the line there. */
    if (dspark_stats && t0 > 0.0)
        fprintf(stderr, "pulsar: dspark fused n_batch=%u committed=%d pend=%u step_ms=%.1f\n",
                n_batch, commit, keep, (now_sec() - t0) * 1000.0);
    return n_accept;
}


/* The single lane (the classic API: CLI, eval, warmup): begin, the target's own verify forward over
 * the round's rows, end -- the batched lane runs the same begin and end around a shared forward. */
static int spec_single_round(pulsar_session *s, int first_token,
                             int max_tokens,
                             float temperature, int top_k,
                             float top_p, float min_p,
                             uint64_t *rng,
                             int *accepted, int accepted_cap,
                             char *err, size_t errlen) {
    pulsar_engine *e = s->engine;
    const double t0 = now_sec();
    pulsar_spec_round r;
    memset(&r, 0, sizeof r);
    {
        const int brc = spec_round_begin(s, first_token, max_tokens, accepted_cap,
                                         temperature, top_k, top_p, min_p, &r,
                                         err, errlen);
        if (brc != 0) { free(r.snap); return brc; }
    }
    pulsar_spec_rows src;
    memset(&src, 0, sizeof src);
    if (!spec_target(s)->verify_single(s, &r, &src, err, errlen)) {
        s->checkpoint.len = r.saved_len;
        spec_round_undo(s, &r);
        s->checkpoint_valid = false;
        free(r.snap);
        return -1;
    }
    const uint32_t n_rows = pulsar_spec_round_n_rows(&r);
    s->spec.spec_round_banks = 1;
    const int na = spec_round_end(s, &r, first_token,
                                  temperature, top_k, top_p, min_p, rng,
                                  &src, 0u, false, t0, NULL, accepted, accepted_cap, err, errlen);
    free(r.snap);
    /* L263: this lane's own cost observation -- the whole step, redraft
     * included.  Off TP only: a group's ranks must hold the same fit, and this
     * lane has no frame to carry one, so on a group neither rank observes
     * here (the batched lane ships the leader's with every round's end). */
    if (na >= 0 && !e->tp) pulsar_engine_spec_cost_observe(e, n_rows, (now_sec() - t0) * 1e3);
    return na;
}

/* Speculative generation that OWNS sampling: draws the base token from the
 * request's filtered distribution (or forwards the carry left by the previous
 * call), runs the fused draft/verify step with exact sampled acceptance, and
 * leaves the next correctly-distributed base as the carry. temperature <= 0
 * degenerates to the greedy argmax-equality path (byte-identical to the old
 * eval_speculative_block behavior). Returns the number of tokens emitted. */
int pulsar_session::generate_speculative(float temperature, int top_k,
                                     float top_p, float min_p, uint64_t *rng,
                                     int max_tokens,
                                     int *accepted, int accepted_cap,
                                     char *err, size_t errlen) {
    auto *s = this;
    if (!s || max_tokens <= 0 || accepted_cap <= 0 || !accepted) return 0;
    /* Same stale-classic-state guard as pulsar_session_eval: the spec loop
     * decodes and emits against the graph's scalar frontier counters, which
     * hold a cross-bank superset after a multiseq step. */
    if (s->mseq_dirty) {
        snprintf(err, errlen,
                 "speculative generate after a multiseq decode step: classic "
                 "per-bank state is stale; re-sync the session first");
        return 0;
    }
    int first;
    const bool carry_params_match =
        s->spec.spec_carry_temp == temperature && s->spec.spec_carry_top_k == top_k &&
        s->spec.spec_carry_top_p == top_p && s->spec.spec_carry_min_p == min_p;
    /* the carry is only valid at the exact position it was drawn at; any
     * session advance outside this path (sync, plain eval, tool injection)
     * means s->logits no longer matches the carry's source distribution */
    const bool carry_pos_match = s->spec.spec_carry_pos == (int32_t)s->checkpoint.len;
    if (s->spec.spec_carry_valid && carry_params_match && carry_pos_match) {
        first = (int)s->spec.spec_carry_token;
        s->spec.spec_carry_valid = false;
    } else {
        /* no carry, or params changed mid-stream (e.g. tool-call payloads
         * force greedy): redraw from the current distribution */
        s->spec.spec_carry_valid = false;
        first = sample_top_p_min_p(s->logits, spec_vocab(s), temperature, top_k,
                                   top_p, min_p, rng, &s->sample_scratch);
        if (first < 0) {
            snprintf(err, errlen, "the sampler refused the live logits row");
            return -1;
        }
    }
    if (pulsar_token_is_stop(s->engine, first)) {
        /* never forward EOS through the target (matches the old caller loops,
         * which broke before eval) */
        accepted[0] = first;
        return 1;
    }
    /* Yield-quenched requests run plain for their remainder — the same route
     * as a drafterless engine, chosen per request. The carry consumed above is
     * already correctly distributed, so this is a pure speed decision. */
    if (!spec_drafter(s) || s->spec.spec_quenched) {
        if (pulsar_session_family_eval(s, first, err, errlen) != 0) return -1;
        accepted[0] = first;
        return 1;
    }
    return spec_single_round(s, first, max_tokens,
                                              temperature, top_k, top_p, min_p, rng,
                                              accepted, accepted_cap, err, errlen);
}



int pulsar_session::eval_speculative_block(int first_token,
                                        int max_tokens,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
    auto *s = this;
    if (!s || max_tokens <= 0 || accepted_cap <= 0 || !accepted) return 0;
    /* Same stale-classic-state guard as pulsar_session_eval (which the no-dspark
     * fallback below would otherwise hit one frame deeper). */
    if (s->mseq_dirty) {
        snprintf(err, errlen,
                 "speculative block eval after a multiseq decode step: classic "
                 "per-bank state is stale; re-sync the session first");
        return -1;
    }
    if (!spec_drafter(s) || s->spec.spec_quenched) {
        if (pulsar_session_family_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
    /* The fused loop -- one batched forward/step plus transactional no-replay
     * rollback -- is the only loop.  It measured 16.4 vs 15.2 t/s against the
     * old Step1+verify loop with byte-identical deterministic output, and that
     * legacy loop is DELETED along with PULSAR_DSPARK_LEGACY_LOOP, which had no
     * setter anywhere and so could never have restored it. */
    /* an externally chosen first_token invalidates any pending carry */
    s->spec.spec_carry_valid = false;
    return spec_single_round(s, first_token, max_tokens,
                                                 0.0f, 0, 1.0f, 0.0f, NULL,
                                                 accepted, accepted_cap, err, errlen);
}

/* ---- plan-34 inc 6: the batched-lane round API (pulsar.h) --------------- */

pulsar_spec_round *pulsar_spec_round_new(void) {
    pulsar_spec_round *r = (pulsar_spec_round *)xmalloc(sizeof(pulsar_spec_round));
    memset(r, 0, sizeof(*r));
    return r;
}

void pulsar_spec_round_free(pulsar_spec_round *r) {
    if (!r) return;
    free(r->redraft.qrows);   /* L150: the lazily-owned fallback rows */
    free(r->snap);            /* the target's snapshot handle */
    free(r);
}

/* The carry-or-sample head of generate_speculative, verbatim: forward the
 * carry when it is valid at these params and this position, else draw fresh
 * from the live logits. The caller owns the EOS short-circuit. */
int pulsar_session_spec_next_base_local(pulsar_session *s, float temperature,
                               int top_k, float top_p, float min_p,
                               uint64_t *rng) {
    int first;
    const bool carry_params_match =
        s->spec.spec_carry_temp == temperature && s->spec.spec_carry_top_k == top_k &&
        s->spec.spec_carry_top_p == top_p && s->spec.spec_carry_min_p == min_p;
    const bool carry_pos_match = s->spec.spec_carry_pos == (int32_t)s->checkpoint.len;
    if (s->spec.spec_carry_valid && carry_params_match && carry_pos_match) {
        first = (int)s->spec.spec_carry_token;
        s->spec.spec_carry_valid = false;
    } else {
        s->spec.spec_carry_valid = false;
        first = sample_top_p_min_p(s->logits, spec_vocab(s), temperature, top_k,
                                   top_p, min_p, rng, &s->sample_scratch);
    }
    return first;
}

int pulsar_session_spec_round_begin_local(pulsar_session *s, pulsar_spec_round *r,
                                 int first_token, int max_tokens, int accepted_cap,
                                 float temperature, int top_k, float top_p,
                                 float min_p, char *err, size_t errlen) {
    return spec_round_begin(s, first_token, max_tokens, accepted_cap,
                            temperature, top_k, top_p, min_p, r, err, errlen);
}

uint32_t pulsar_spec_round_n_rows(const pulsar_spec_round *r) {
    return r->n_batch;
}

uint32_t pulsar_spec_round_fill_reqs(const pulsar_spec_round *r, uint32_t bank,
                                  int first_token, pulsar_multiseq_req *out) {
    for (uint32_t i = 0; i < r->n_batch; i++) {
        out[i].bank = bank;
        out[i].pos = r->saved_len + (int32_t)i;
        out[i].token = i == 0 ? first_token : (int)r->pend[i - 1];
    }
    return r->n_batch;
}

/** Row source over the server-held ALL_ROWS logits block.
 *
 * The batched lane runs one spec round per bank around ONE shared forward, so
 * a round reads its rows out of the shared block at an offset rather than
 * owning a logits buffer of its own. */
bool pulsar_spec_row_read_block(void *ud, uint32_t row, float *out) {
    const pulsar_spec_rows *b = (const pulsar_spec_rows *)ud;
    memcpy(out, b->block.rows + ((size_t)b->block.row0 + row) * b->block.vocab,
           (size_t)b->block.vocab * sizeof(float));
    return true;
}

/* The batched lane's round end: the target reads this round's rows out of the last shared forward
 * (its own readback or the caller's block), then the one round_end runs. */
static int spec_round_end_block(pulsar_session *s, pulsar_spec_round *r,
                                int first_token,
                                float temperature, int top_k, float top_p,
                                float min_p, uint64_t *rng,
                                const float *rows, uint32_t row0,
                                const int32_t *forced_truth,
                                int *accepted, int accepted_cap,
                                char *err, size_t errlen) {
    pulsar_spec_rows src;
    memset(&src, 0, sizeof src);
    if (spec_target(s)->verify_rows) {
        if (!spec_target(s)->verify_rows(s, r, rows, row0, &src, err, errlen)) return -1;
    } else {
        /* the block holds every row: draft i is judged against round-local row i (the row that
         * PREDICTS it), its argmax computed here */
        for (uint32_t i = 0; i < r->K && i < 16u; i++)
            r->row_tops[i] = (int)sample_argmax(rows + ((size_t)row0 + i) * spec_vocab(s), spec_vocab(s));
        src.read = pulsar_spec_row_read_block;
        src.ud = &src;
        src.block.rows = rows;
        src.block.row0 = row0;
        src.block.vocab = spec_vocab(s);
    }
    return spec_round_end(s, r, first_token,
                          temperature, top_k, top_p, min_p, rng,
                          &src, row0,
                          true /* L150: redraft deferred to the batch */,
                          0.0 /* t0: step_ms diagnostic reads 0 in this lane */,
                          forced_truth, accepted, accepted_cap, err, errlen);
}



int pulsar_session_spec_round_end_local(pulsar_session *s, pulsar_spec_round *r,
                               int first_token,
                               float temperature, int top_k, float top_p,
                               float min_p, uint64_t *rng,
                               const float *rows, uint32_t row0,
                               int *accepted, int accepted_cap,
                               char *err, size_t errlen) {
    return spec_round_end_block(s, r, first_token, temperature, top_k, top_p,
                                min_p, rng, rows, row0, NULL, accepted, accepted_cap,
                                err, errlen);
}

int pulsar_session_spec_round_end_forced(pulsar_session *s, pulsar_spec_round *r,
                                      int first_token,
                                      float temperature, int top_k, float top_p,
                                      float min_p, uint64_t *rng,
                                      const float *rows, uint32_t row0,
                                      const int32_t *truth,
                                      int *accepted, int accepted_cap,
                                      char *err, size_t errlen) {
    if (!truth) {
        snprintf(err, errlen, "spec round_end_forced: no truth");
        return -1;
    }
    return spec_round_end_block(s, r, first_token, temperature, top_k, top_p,
                                min_p, rng, rows, row0, truth, accepted, accepted_cap,
                                err, errlen);
}




/* =====================================================================
 * L150: batched redraft -- one drafter pass for every bank of the tick.
 *
 * The batched lane's round_end records each bank's redraft request and
 * defers. This runs the draft forward over all requesting banks' rows at once
 * (gpu_graph_dspark_draft_forward_banks: rope/store/visibility positions per
 * row, bank-major rings), the markov refine with one weight stream per
 * position across banks, and the confidence head over every row, then leaves
 * each bank's results in its round. It NEVER switches banks -- it reads the
 * banks' ring counters from the saved carries (g->ms_dspark_n_raw) and the
 * rings from the slabs -- so the server's bank_switch bookkeeping stays the
 * single authority; pulsar_session_spec_redraft_commit stamps a bank's shadow
 * under the server's switch.
 *
 * Byte-exact per bank with the serialized single-bank redraft (the kernels are
 * exact by construction; each bank draws from its own rng in the same order);
 * the greedy identity gate pins it. Diagnostics of the single-bank path
 * (DSPARK_Q lines, the on-policy and ring dumps, dspark_dump_step) are not
 * emitted here: they are dev instruments of the classic lane. */

/* L150: every live round drafts in ONE drafter pass over all the banks, no bank switch. */
int pulsar_session_spec_redraft_batch_local(pulsar_session *s, pulsar_spec_round **rounds,
                                      const uint32_t *banks, uint64_t **rngs, int n,
                                      char *err, size_t errlen) {
    if (!rounds || !banks || !rngs || n <= 0) return 0;
    if (!spec_drafter(s)) {
        snprintf(err, errlen, "redraft batch: no drafter is loaded");
        return -1;
    }
    return spec_drafter(s)->draft_batch(s, rounds, banks, rngs, n, err, errlen);
}

/* L150: stamp the CURRENT bank's shadow from its batched redraft result. The
 * caller has switched to the round's bank and saves it afterwards. Mirrors the
 * tail of spec_round_redraft field for field; the greedy identity gate is what
 * keeps the two in step. */
void pulsar_session_spec_redraft_commit_local(pulsar_session *s, pulsar_spec_round *r) {
    spec_redraft_req *q = &r->redraft;
    if (!q->valid || !q->done) return;
    pulsar_spec_redraft_stamp(s, q);
    q->done = false;
    q->valid = false;
}

bool pulsar_spec_redraft_qrows_reserve(spec_redraft_req *q, uint32_t vocab) {
    if (!q->qrows) q->qrows = (float *)xmalloc((size_t)16 * vocab * sizeof(float));
    return q->qrows != NULL;
}

void pulsar_spec_redraft_stamp(pulsar_session *s, spec_redraft_req *q) {
    s->spec.pend_base = q->next_base;
    s->spec.dspark_chain_unharvested = false;
    s->spec.n_pend = q->keep;
    for (uint32_t i = 0; i < q->keep; i++) s->spec.pend[i] = q->refined[i + 1];
    s->spec.pend_sampled = q->sample_drafts;
    s->spec.qrows_n = q->sample_drafts ? q->n_draft : 0u;   /* L260: the rows this commit writes */
    s->spec.pend_pos = (int32_t)s->checkpoint.len;
    s->spec.pend_temp = q->temperature;
    s->spec.pend_top_k = q->top_k;
    s->spec.pend_top_p = q->top_p;
    s->spec.pend_min_p = q->min_p;
    for (uint32_t i = 0; i < q->keep; i++)
        s->spec.pend_conf[i] = q->have_conf ? q->conf[i] : -1.0f;
    if (q->sample_drafts) {
        bool need_rows = false;
        for (uint32_t i = 0; i < q->n_draft; i++) {
            s->spec.pend_q[i] = q->q_drawn[i];
            s->spec.pend_qn[i] = q->qn[i];
            if (pulsar_spec_q_compact(q->qn[i])) {
                memcpy(s->spec.pend_qids[i], q->qids[i], (size_t)q->qn[i] * sizeof(int32_t));
                memcpy(s->spec.pend_qprobs[i], q->qprobs[i], (size_t)q->qn[i] * sizeof(float));
            } else {
                need_rows = true;
            }
        }
        if (need_rows && q->qrows) {
            const uint32_t need = q->n_draft * spec_vocab(s);
            if (s->pend_qrows_cap < need) {
                free(s->pend_qrows);
                s->pend_qrows = (float *)xmalloc((size_t)need * sizeof(float));
                s->pend_qrows_cap = need;
            }
            for (uint32_t i = 0; i < q->n_draft; i++)
                if (q->qn[i] == 0)
                    memcpy(s->pend_qrows + (size_t)i * spec_vocab(s),
                           q->qrows + (size_t)i * spec_vocab(s),
                           (size_t)spec_vocab(s) * sizeof(float));
        }
    }
}

/* ---- L260: the batched lane's per-bank bookkeeping, one call per phase ----
 * Each is the per-bank sequence the server ran, in the same order on the same
 * state: restore, the phase, save.  The public wrappers (engine_api.cpp) send
 * ONE frame per phase and collect ONE verdict, pulsar_spec_steps_verdict. */
static void spec_step_reset(pulsar_spec_step *st) {
    st->status = PULSAR_SPEC_STEP_OK;
    st->n_accepted = 0;
    st->err[0] = '\0';
}

void pulsar_session_spec_assemble_batch_local(pulsar_session *s, pulsar_spec_step *steps, int n,
                                              uint32_t row_budget,
                                              pulsar_multiseq_req *reqs, uint32_t *n_rows_out) {
    uint32_t rows = 0;
    for (int i = 0; i < n; i++) {
        pulsar_spec_step *st = &steps[i];
        spec_step_reset(st);
        st->first_token = -1;
        st->row0 = 0;
        st->n_rows = 0;
        if (!spec_bank_restore(s, st->bank)) {
            st->status = PULSAR_SPEC_STEP_RESTORE_FAILED;
            continue;
        }
        const uint32_t cap_rows = st->k_alloc >= 0 ? 1u + (uint32_t)st->k_alloc
                                                   : pulsar_session_spec_next_rows_max(s);
        if (s->pos() >= s->ctx()) {
            st->status = PULSAR_SPEC_STEP_LENGTH;
        } else if (rows + cap_rows > row_budget) {
            /* Sit out BEFORE the base draw or begin touch anything: the carry
             * (possibly a rejection residual, whose exact emission the
             * acceptance proof needs) and the pendings stay intact. */
            st->status = PULSAR_SPEC_STEP_SKIPPED;
        } else {
            const int first = pulsar_session_spec_next_base_local(s, st->temperature, st->top_k,
                                                                  st->top_p, st->min_p, st->rng);
            st->first_token = first;
            if (first < 0) {
                st->status = PULSAR_SPEC_STEP_FAILED;
                snprintf(st->err, sizeof st->err, "sampler refused a degenerate logits row (L188)");
            } else if (pulsar_token_is_stop(s->engine, first)) {
                st->status = PULSAR_SPEC_STEP_EOS;
            } else if (pulsar_session_spec_round_begin_local(s, st->round, first, st->max_tokens,
                                                             st->accepted_cap, st->temperature,
                                                             st->top_k, st->top_p, st->min_p,
                                                             st->err, sizeof st->err) != 0) {
                st->status = PULSAR_SPEC_STEP_FAILED;
            } else if (rows + pulsar_spec_round_n_rows(st->round) > row_budget) {
                /* Unreachable: begin only trims below cap_rows.  The backstop
                 * runs BEFORE fill_reqs writes, so no change to begin's row
                 * math can overflow reqs[]. */
                pulsar_session_spec_round_abort_local(s, st->round);
                st->status = PULSAR_SPEC_STEP_SKIPPED;
            } else {
                /* The worker has no reqs: its rows ride the mixed-batch frame. */
                st->n_rows = reqs ? pulsar_spec_round_fill_reqs(st->round, st->bank, first, reqs + rows)
                                  : pulsar_spec_round_n_rows(st->round);
                st->row0 = rows;
                rows += st->n_rows;
            }
        }
        spec_bank_save(s, st->bank);
    }
    *n_rows_out = rows;
}

void pulsar_session_spec_round_end_batch_local(pulsar_session *s, pulsar_spec_step *steps, int n,
                                               const float *rows) {
    /* L260: each bank's walk records its committed capture rows; one banked seed
     * after the loop shares the drafter's main_proj / attn_kv reads across every
     * bank (they were re-read per committed row -- 16 to 32 times a step at c16). */
    s->seed_defer.active = s->engine->family->banks->pooled(s);
    s->seed_defer.n = 0;
    s->spec.spec_round_banks = n;   /* L263: the round's flat cost is shared n ways */
    for (int i = 0; i < n; i++) {
        pulsar_spec_step *st = &steps[i];
        spec_step_reset(st);
        st->pos_before = pulsar_spec_round_saved_len(st->round);
        st->pos_after = -1;
        if (!spec_bank_restore(s, st->bank)) {
            st->status = PULSAR_SPEC_STEP_RESTORE_FAILED;
            continue;
        }
        const int na = pulsar_session_spec_round_end_local(s, st->round, st->first_token,
                                                           st->temperature, st->top_k, st->top_p,
                                                           st->min_p, st->rng, rows, st->row0,
                                                           st->accepted, st->accepted_cap,
                                                           st->err, sizeof st->err);
        if (na < 0) st->status = PULSAR_SPEC_STEP_FAILED;
        else st->n_accepted = na;
        st->pos_after = s->pos();
        spec_bank_save(s, st->bank);
    }
    if (!s->seed_defer.active) return;
    s->seed_defer.active = false;
    if (!spec_drafter(s)->absorb_banked(s, s->seed_defer.src_row, s->seed_defer.bank, s->seed_defer.next_tok,
                                        s->seed_defer.n)) {
        /* No counter advanced: every bank that had rows refuses its round, as a
         * failed one-row seed did ("DSpark fused state update failed"). */
        for (int i = 0; i < n; i++) {
            pulsar_spec_step *st = &steps[i];
            bool had = false;
            for (uint32_t k = 0; k < s->seed_defer.n && !had; k++) had = s->seed_defer.bank[k] == st->bank;
            if (!had || st->status != PULSAR_SPEC_STEP_OK) continue;
            st->status = PULSAR_SPEC_STEP_FAILED;
            st->round->redraft.valid = false;
            snprintf(st->err, sizeof st->err, "the batched drafter absorb failed");
        }
    }
    s->seed_defer.n = 0;
}

void pulsar_session_spec_redraft_commit_batch_local(pulsar_session *s, pulsar_spec_step *steps, int n) {
    for (int i = 0; i < n; i++) {
        pulsar_spec_step *st = &steps[i];
        spec_step_reset(st);
        if (!spec_bank_restore(s, st->bank)) {
            st->status = PULSAR_SPEC_STEP_RESTORE_FAILED;
            continue;
        }
        pulsar_session_spec_redraft_commit_local(s, st->round);
        spec_bank_save(s, st->bank);
    }
}

/* FNV-1a over 32-bit words. */
static uint32_t spec_fp(uint32_t h, uint32_t v) {
    for (int b = 0; b < 4; b++) h = (h ^ ((v >> (8 * b)) & 0xffu)) * 16777619u;
    return h;
}

int pulsar_spec_steps_verdict(pulsar_spec_phase phase, const pulsar_spec_step *steps, int n,
                              uint32_t n_rows) {
    /* Only what the phase itself decides, which both ranks hold the same way:
     * a field the phase does not write is the caller's leftover on the leader
     * and the wire's value on the worker. */
    uint32_t h = spec_fp(2166136261u, (uint32_t)phase);
    h = spec_fp(h, (uint32_t)n);
    h = spec_fp(h, n_rows);
    for (int i = 0; i < n; i++) {
        const pulsar_spec_step *st = &steps[i];
        h = spec_fp(h, st->bank);
        h = spec_fp(h, (uint32_t)st->status);
        if (phase == PULSAR_SPEC_PHASE_ASSEMBLE) {
            h = spec_fp(h, (uint32_t)st->first_token);
            h = spec_fp(h, st->row0);
            h = spec_fp(h, st->n_rows);
        } else if (phase == PULSAR_SPEC_PHASE_ROUND_END) {
            h = spec_fp(h, (uint32_t)st->pos_after);
            h = spec_fp(h, (uint32_t)st->n_accepted);
            for (int t = 0; t < st->n_accepted; t++) h = spec_fp(h, (uint32_t)st->accepted[t]);
        }
    }
    return (int)(h & 0x3fffffffu) + 1;
}

int pulsar_session_spec_redraft_peek(const pulsar_spec_round *r, int32_t ids[17], float conf[16],
                                     uint32_t *n_draft, uint32_t *keep, int *sampled) {
    if (!r || !r->redraft.valid || !r->redraft.done) return 0;
    const spec_redraft_req *q = &r->redraft;
    if (ids) memcpy(ids, q->refined, sizeof(q->refined));
    if (conf) memcpy(conf, q->conf, sizeof(q->conf));
    if (n_draft) *n_draft = q->n_draft;
    if (keep) *keep = q->keep;
    if (sampled) *sampled = q->sample_drafts ? 1 : 0;
    return 1;
}

uint32_t pulsar_session_bank_pending_confs(const pulsar_session *s, uint32_t bank,
                                        float out[16]) {
    /* L108 P2: the live session's pending count may still be an
     * in-flight chain; complete it before peeking. Logically a lazy
     * read-completion, not observable-state mutation. */
    pulsar_session_spec_chain_harvest((pulsar_session *)s);

    const pulsar_spec_carry_state *sp = s ? pulsar_spec_bank_shadow((pulsar_session *)s, bank) : NULL;
    if (!sp) return 0;
    uint32_t n = sp->n_pend;
    if (n > 16u) n = 16u;
    for (uint32_t i = 0; i < n; i++) out[i] = sp->pend_conf[i];
    return n;
}

const pulsar_spec_carry_state *pulsar_spec_bank_shadow(pulsar_session *s, uint32_t bank) {
    if (!s) return NULL;
    if (bank == pulsar_session_live_bank(s)) return &s->spec;
    /* a non-live bank's saved shadow: the core's carry, for every family (L272 P2) */
    return s->bank_carry && bank < s->bank_carry_n && s->bank_carry[bank].valid ? &s->bank_carry[bank].spec : NULL;
}

/* L112 observability: the adaptive draft controller's CURRENT depth for a bank -- the live state for the
 * installed bank, the saved carry otherwise.  0 = no drafter or nothing valid.  Pure host reads,
 * worker-thread safe at publish time. */
int pulsar_spec_bank_depth(pulsar_session *s, uint32_t bank) {
    if (!s || !spec_drafter(s)) return 0;
    const pulsar_spec_carry_state *sp = pulsar_spec_bank_shadow(s, bank);
    if (!sp) return 0;
    int d = sp->spec_adaptive_depth;
    if (d <= 0) d = (int)spec_drafter(s)->depth_default(s->engine);
    if (d < 1) d = 1;
    if (d > 16) d = 16;
    return d;
}

/* The session's speculative shadow into / out of a bank carry: s->spec by value, plus the full q rows
 * its sampled pendings may still read (pulsar_spec_q_compact says which positions keep a compact q
 * instead); a harvest first, so no in-flight chain is ever saved (L108 P2). */
static void spec_copy_pending_qrows(float *dst, uint32_t dst_cap, const float *src, uint32_t src_cap,
                                    const pulsar_spec_carry_state &sp, uint32_t vocab) {
    if (!dst || !src) return;
    for (uint32_t j = 0; j < sp.qrows_n && j < 16u; j++) {
        if (pulsar_spec_q_compact(sp.pend_qn[j])) continue;
        const uint64_t end = (uint64_t)(j + 1u) * vocab;
        if (end > dst_cap || end > src_cap) return;   /* no row was drafted there: nothing to carry */
        memcpy(dst + (size_t)j * vocab, src + (size_t)j * vocab, (size_t)vocab * sizeof(float));
    }
}

void pulsar_spec_shadow_save(pulsar_session *s, pulsar_spec_carry_state *dst, float **dst_rows, uint32_t *dst_cap) {
    if (s->pend_qrows && *dst_cap < s->pend_qrows_cap) {
        *dst_rows = (float *)xrealloc(*dst_rows, (size_t)s->pend_qrows_cap * sizeof(float));
        *dst_cap = s->pend_qrows_cap;
    }
    pulsar_session_spec_chain_harvest(s);
    *dst = s->spec;
    spec_copy_pending_qrows(*dst_rows, *dst_cap, s->pend_qrows, s->pend_qrows_cap, *dst, spec_vocab(s));
}

void pulsar_spec_shadow_restore(pulsar_session *s, const pulsar_spec_carry_state *src, const float *src_rows,
                                uint32_t src_cap) {
    if (s->pend_qrows_cap < src_cap) {
        s->pend_qrows = (float *)xrealloc(s->pend_qrows, (size_t)src_cap * sizeof(float));
        s->pend_qrows_cap = src_cap;
    }
    spec_copy_pending_qrows(s->pend_qrows, s->pend_qrows_cap, src_rows, src_cap, *src, spec_vocab(s));
    s->spec = *src;
}

int pulsar_spec_round_saved_len(const pulsar_spec_round *r) {
    return r ? r->saved_len : -1;
}

uint32_t pulsar_session_spec_next_rows_max(const pulsar_session *s) {
    /* Upper bound on the next round's n_batch: begin's guards (base-token,
     * position, params, caps) only ever TRIM K from n_pend, so
     * 1 + pending is safe to budget against before consuming anything. */
    uint32_t k = s->spec.n_pend;
    if (k > 16u) k = 16u;
    return 1u + k;
}


void pulsar_session_spec_arm_capture_local(pulsar_session *s, uint32_t n_rows) {
    if (spec_target(s)->arm_verify) spec_target(s)->arm_verify(s, n_rows);
}

void pulsar_session_spec_round_abort_local(pulsar_session *s, pulsar_spec_round *r) {
    /* Mirror the fused loop's forward-failure branch: drop the pushed rows,
     * restore the frontier, release the snapshot. */
    s->checkpoint.len = r->saved_len;
    spec_round_undo(s, r);
    s->checkpoint_valid = false;
}

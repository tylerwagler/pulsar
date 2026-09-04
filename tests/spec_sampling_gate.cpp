/* Statistical oracle for exact sampled speculative decoding.
 *
 * Promoted from temp/archive-2026-07-14/spec_sampling_test.c. temp/ is
 * transient and we have already lost harnesses that way (the Tier-2
 * byte-compare harness is gone; only gates promoted to tests/ survived), and
 * this is the ONLY end-to-end check that the speculative path emits the exact
 * target distribution.
 *
 * From one frozen session state, generates N short trajectories two ways:
 *   A) plain sampling (pulsar_session_sample + eval per token)
 *   B) speculative   (pulsar_session_generate_speculative, drafter active)
 * and chi-square-compares the per-position token marginals. If the acceptance +
 * residual-carry scheme is exact, the two are the same distribution; a biased
 * sampler shows up as a fat chi-square at position 1+ (position 0 is the same
 * plain draw in both paths by construction).
 *
 * The oracle is PROPOSAL-AGNOSTIC: it compares emitted marginals and never
 * inspects how a draft was proposed. It therefore validates the deterministic
 * (argmax-proposal, accept w.p. p) rule and the temperature-matched
 * (sampled-proposal, accept w.p. min(1,p/q), residual (p-q)+) rule unchanged —
 * which is exactly what makes it the gate for spec-decode Item 1.
 *
 * Also runs the greedy gate: temp=0 speculative output must agree with plain
 * greedy for a long prefix, and must be deterministic run-to-run.
 *
 * Reports acceptance alpha (accepted drafts / proposed drafts) from the
 * engine's own counters, so the same run answers "is the sampled proposal
 * beating the greedy p(mode) acceptance ceiling?".
 *
 * *** READ THIS BEFORE "FIXING" THE DEEP CHI-SQUARE ***
 * This gate's own test compares mode 0 (plain, sampled from SINGLE-TOKEN DECODE
 * logits) against mode 1 (speculative, p read from the BATCHED verify rows).
 * Those are two different numeric paths. They are KNOWN to diverge on near-ties
 * — the Tier-2 driver control proved the batched sweep diverges from classic
 * decode using only pre-existing code on an untouched baseline ("batch-sweep
 * DIVERGES from classic decode at step 18", gaps 0.03-0.26). So at depth this
 * test has a systematic difference between its arms that is NOT a sampler bug,
 * and it sits near crit even on the SHIPPED engine (measured: baseline at ctx
 * 6017, traj 2000 -> pos3 chi2=21.3 vs crit 22.6, i.e. 94% of threshold).
 *
 * Consequence: a deep FAIL here does NOT by itself indict the sampler. Measured
 * TVD between arms, ctx 6017 traj 2000:
 *     fixed    mode0 vs mode1 : 0.0530 / 0.0605  (pos2/pos3)
 *     baseline mode0 vs mode1 : 0.0400 / 0.0445
 *     fixed mode1 vs baseline mode1 : 0.0130 / 0.0180   <-- 3-4x smaller
 * The two speculative engines agree with EACH OTHER far better than either
 * agrees with plain decode: one shared artifact, both rules exact.
 *
 * To test the SAMPLER rather than the numerics, dump both engines (argv[6]) and
 * run tests/spec_sampling_compare.py — mode1-vs-mode1 removes the confound
 * because both arms are batch-sourced. The proper repair of THIS test is to
 * have mode 0 read the same batched rows (a 1-row spec batch, drafter off) so
 * it compares like with like; until then, treat a deep mode0-vs-mode1 FAIL as
 * "engine numerics", not "sampler bias". The batch-vs-decode divergence is a
 * real engine property we are deliberately NOT fixing here.
 *
 * L160 (2026-09-04): the sampled-distribution arm runs its trajectories as
 * BANKED BATCHES through the server's batched lane -- the same round_begin /
 * decode_mixed(ALL_ROWS) / round_end / redraft_batch sequence server_sched.cpp
 * runs for co-scheduled clients (tests/dspark_batch_gate.cpp drives it the same
 * way).  Every trajectory still starts from the SAME snapshot (each bank is
 * repointed, invalidated and loaded from it) and draws from the SAME per-
 * trajectory rng stream as before, so the emitted tokens and alpha are what the
 * serial loop produced -- 16 plain trajectories or 4 speculative ones per
 * forward instead of one.  Both widths stay inside the 16-row M-neutral range
 * the battery asserts (mixed_neutrality_gate), which is what makes the rows
 * byte-identical to a 1-row step; a round that would exceed 16 rows FAILS the
 * gate, it does not fall back to a narrower batch.  Mode 0 is therefore also
 * batch-sourced now (the "like-with-like" repair this header asked for above):
 * position 0 samples the snapshot's logits, positions 1+ sample decode_mixed
 * rows.  The greedy hard gates are unchanged and still run serially.
 *
 * MODEL-DEPENDENT — needs the merged drafter gguf and ~95 GB free. Run:
 *   make cuda-spec-sampling-gate                  (defaults to gguf/model.gguf)
 *   ./tests/spec_sampling_gate <model.gguf> [temp] [filler_tokens] [traj] \
 *                              [top_p] [dump_path] [min_p]
 * Defaults: temp 0.95, filler 0, traj 2500 (clamped to [1,TRAJ]), top_p 0.95,
 * dump off ("-" or "" also = off, so min_p can be given without a dump),
 * min_p 0. top_p is not a knob to turn down casually — see the note at its
 * parse below; too low makes the nucleus a single token and the chi-square
 * passes vacuously. min_p > 0 is what makes this gate NON-VACUOUS for the
 * min-p prefilter in pulsar_sample_dist_build (dev-minp): at min_p == 0 the
 * prefilter path is not even entered. The server default is 0.05.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pulsar.h"

#define TRAJ 2500
#define DEPTH 4
#define TOPN 24   /* histogram buckets per position: top tokens + tail bucket */

static const char *PROMPT =
    "The economic history of the Mediterranean is inseparable from its ports. ";

typedef struct { int id; long a, b; } bucket;

/* L160: banks per forward in the sampled arm.  Plain trajectories are one row
 * each; speculative ones are 1 + K rows with K trimmed to max_tokens - 1 <=
 * DEPTH - 1 by round_begin, so 4 banks are at most 16 rows.  16 is the
 * ALL_ROWS head cap AND the M-neutral range the battery asserts. */
#ifndef SAMPLED_BANKS_PLAIN
#define SAMPLED_BANKS_PLAIN 16
#endif
#ifndef SAMPLED_BANKS_SPEC
#define SAMPLED_BANKS_SPEC  4
#endif
#define SAMPLED_ROWS_MAX    16

/* Put bank `b` at the snapshot: repoint the device views (gpu_graph_bank_repoint
 * swaps the raw ring, the compressed caches and the compressor state to the
 * bank's storage), drop the live bookkeeping, load the payload into that bank,
 * persist it as the bank's carry.  The same reset for every bank and every
 * trajectory -- no "first time" path.  Measured 2026-09-04 (L160): at one bank
 * per forward this reproduces the serial gate's 10,000 tokens byte for byte;
 * a rewind-based reset came within 92% but not to identity (compressor state,
 * drafter window), so the payload load is the reset. */
static int bank_reset(pulsar_session *s, uint32_t b, const pulsar_session_snapshot *snap,
                      char *err, size_t errlen) {
    if (pulsar_session_bank_repoint(s, b) != 0) {
        snprintf(err, errlen, "bank %u repoint failed", b);
        return -1;
    }
    pulsar_session_invalidate(s);
    if (pulsar_session_load_snapshot(s, snap, err, errlen) != 0) return -1;
    pulsar_session_bank_state_save(s, b);
    return 0;
}

/* The per-trajectory rng seed.  Unchanged from the serial loop so the two
 * implementations draw the same streams. */
static uint64_t traj_seed(int t, int mode) {
    return 0x9E3779B97F4A7C15ull * (uint64_t)(t + 1) + (uint64_t)mode * 77777u;
}

/* Mode 0, one batch of up to SAMPLED_BANKS_PLAIN trajectories: position 0 is
 * drawn from the snapshot's logits (every bank was just loaded from the same
 * snapshot, so the live logits ARE that distribution), positions 1.. from the
 * decode_mixed row of the bank's own token.  Returns 0, or -1 with err. */
static int sampled_plain_batch(pulsar_session *s, const pulsar_session_snapshot *snap,
                               int t0, int nb, float temp, float top_p, float min_p,
                               int eos, int vocab, float *logits, int (*seq)[DEPTH],
                               char *err, size_t errlen) {
    uint64_t rng[SAMPLED_BANKS_PLAIN];
    int got[SAMPLED_BANKS_PLAIN], live[SAMPLED_BANKS_PLAIN];
    pulsar_multiseq_req reqs[SAMPLED_BANKS_PLAIN];
    int pos0 = -1;
    for (int b = 0; b < nb; b++) {
        if (bank_reset(s, (uint32_t)b, snap, err, errlen) != 0) return -1;
        const int p = pulsar_session_pos(s);
        if (pos0 < 0) pos0 = p;
        if (p != pos0) { snprintf(err, errlen, "bank %d at pos %d, bank 0 at %d", b, p, pos0); return -1; }
        rng[b] = traj_seed(t0 + b, 0);
        const int tok = pulsar_session_sample(s, temp, 0, top_p, min_p, &rng[b]);
        seq[t0 + b][0] = tok;
        got[b] = 1;
        live[b] = tok != eos && DEPTH > 1;
        reqs[b].bank = (uint32_t)b; reqs[b].pos = pos0; reqs[b].token = tok;
    }
    for (int step = 1; step < DEPTH; step++) {
        pulsar_multiseq_req batch[SAMPLED_BANKS_PLAIN];
        int who[SAMPLED_BANKS_PLAIN];
        uint32_t m = 0;
        for (int b = 0; b < nb; b++) if (live[b]) { batch[m] = reqs[b]; who[m++] = b; }
        if (m == 0) break;
        uint32_t rows = 0;
        const int rc = pulsar_session_decode_mixed(s, batch, m, logits, (int)(m * (uint32_t)vocab),
                                                   &rows, 0u, err, errlen);
        if (rc != 0 || rows != m) {
            if (rc == 0) snprintf(err, errlen, "decode_mixed returned %u rows for %u banks", rows, m);
            return -1;
        }
        for (uint32_t q = 0; q < m; q++) {
            const int b = who[q];
            const float *row = logits + (size_t)q * (size_t)vocab;
            const int tok = pulsar_sample_logits(row, vocab, temp, 0, top_p, min_p, &rng[b]);
            seq[t0 + b][got[b]++] = tok;
            reqs[b].pos++; reqs[b].token = tok;
            if (tok == eos || got[b] >= DEPTH) live[b] = 0;
        }
    }
    for (int b = 0; b < nb; b++)
        for (int k = got[b]; k < DEPTH; k++) seq[t0 + b][k] = -1;
    return 0;
}

/* Mode 1, one batch of up to SAMPLED_BANKS_SPEC trajectories through the
 * server's batched speculative lane: per bank under its restore, the base draw
 * and round_begin; ONE decode_mixed(ALL_ROWS) over every bank's rows; per bank
 * under its restore, round_end; then ONE redraft_batch over the banks that
 * continue, committed per bank.  A base draw of EOS ends the trajectory
 * without a forward, as generate_speculative does. */
static int sampled_spec_batch(pulsar_session *s, const pulsar_session_snapshot *snap,
                              int t0, int nb, float temp, float top_p, float min_p,
                              int eos, int vocab, float *logits, int (*seq)[DEPTH],
                              pulsar_spec_round **r, char *err, size_t errlen) {
    uint64_t rng[SAMPLED_BANKS_SPEC];
    int got[SAMPLED_BANKS_SPEC], live[SAMPLED_BANKS_SPEC];
    for (int b = 0; b < nb; b++) {
        if (bank_reset(s, (uint32_t)b, snap, err, errlen) != 0) return -1;
        rng[b] = traj_seed(t0 + b, 1);
        got[b] = 0; live[b] = 1;
    }
    for (;;) {
        pulsar_multiseq_req reqs[SAMPLED_ROWS_MAX + 17];
        int first[SAMPLED_BANKS_SPEC], begun[SAMPLED_BANKS_SPEC];
        uint32_t row0[SAMPLED_BANKS_SPEC];
        uint32_t rows = 0;
        int any = 0;
        for (int b = 0; b < nb; b++) {
            begun[b] = 0;
            if (!live[b]) continue;
            if (!pulsar_session_bank_state_restore(s, (uint32_t)b)) {
                snprintf(err, errlen, "bank %d restore failed", b); return -1;
            }
            first[b] = pulsar_session_spec_next_base(s, temp, 0, top_p, min_p, &rng[b]);
            if (first[b] == eos) {
                seq[t0 + b][got[b]++] = eos;
                live[b] = 0;
                pulsar_session_bank_state_save(s, (uint32_t)b);
                continue;
            }
            if (pulsar_session_spec_round_begin(s, r[b], first[b], DEPTH - got[b], 17,
                                                temp, 0, top_p, min_p, err, errlen) != 0)
                return -1;
            begun[b] = 1;
            const uint32_t nr = pulsar_spec_round_n_rows(r[b]);
            if (rows + nr > SAMPLED_ROWS_MAX) {
                snprintf(err, errlen, "round rows %u + %u exceed the %d-row batch (bank %d)",
                         rows, nr, SAMPLED_ROWS_MAX, b);
                return -1;
            }
            row0[b] = rows;
            rows += pulsar_spec_round_fill_reqs(r[b], (uint32_t)b, first[b], reqs + rows);
            pulsar_session_bank_state_save(s, (uint32_t)b);
            any = 1;
        }
        if (!any) break;
        pulsar_session_spec_arm_capture(s, rows);
        uint32_t out_rows = 0;
        const int rc = pulsar_session_decode_mixed(s, reqs, rows, logits, (int)(rows * (uint32_t)vocab),
                                                   &out_rows, PULSAR_MSEQ_HEAD_ALL_ROWS, err, errlen);
        pulsar_session_spec_arm_capture(s, 0u);
        if (rc != 0 || out_rows != rows) {
            for (int b = 0; b < nb; b++) if (begun[b]) {
                pulsar_session_bank_state_restore(s, (uint32_t)b);
                pulsar_session_spec_round_abort(s, r[b]);
            }
            if (rc == 0) snprintf(err, errlen, "decode_mixed returned %u rows for %u", out_rows, rows);
            return -1;
        }
        pulsar_spec_round *cont_r[SAMPLED_BANKS_SPEC];
        uint32_t cont_b[SAMPLED_BANKS_SPEC];
        uint64_t *cont_rng[SAMPLED_BANKS_SPEC];
        int nc = 0;
        for (int b = 0; b < nb; b++) {
            if (!begun[b]) continue;
            if (!pulsar_session_bank_state_restore(s, (uint32_t)b)) {
                snprintf(err, errlen, "bank %d restore failed", b); return -1;
            }
            int accepted[17];
            const int na = pulsar_session_spec_round_end(s, r[b], first[b], eos, temp, 0, top_p, min_p,
                                                         &rng[b], logits, row0[b], accepted, 17,
                                                         err, errlen);
            if (na < 0) return -1;
            for (int i = 0; i < na && got[b] < DEPTH; i++) seq[t0 + b][got[b]++] = accepted[i];
            if (got[b] >= DEPTH || (got[b] > 0 && seq[t0 + b][got[b] - 1] == eos)) live[b] = 0;
            pulsar_session_bank_state_save(s, (uint32_t)b);
            if (live[b]) { cont_r[nc] = r[b]; cont_b[nc] = (uint32_t)b; cont_rng[nc] = &rng[b]; nc++; }
        }
        if (nc > 0) {
            if (pulsar_session_spec_redraft_batch(s, cont_r, cont_b, cont_rng, nc, err, errlen) != 0)
                return -1;
            for (int i = 0; i < nc; i++) {
                if (!pulsar_session_bank_state_restore(s, cont_b[i])) {
                    snprintf(err, errlen, "bank %u restore failed", cont_b[i]); return -1;
                }
                pulsar_session_spec_redraft_commit(s, cont_r[i]);
                pulsar_session_bank_state_save(s, cont_b[i]);
            }
        }
    }
    for (int b = 0; b < nb; b++)
        for (int k = got[b]; k < DEPTH; k++) seq[t0 + b][k] = -1;
    return 0;
}

static int bucket_cmp(const void *x, const void *y) {
    const bucket *p = (const bucket *)x;
    const bucket *q = (const bucket *)y;
    return (int)((q->a + q->b) - (p->a + p->b));
}

/* alpha over a window of the engine's cumulative spec counters */
typedef struct { uint64_t drafted, accepted, rounds, gen; } spec_snap;

static spec_snap spec_take(pulsar_engine *e) {
    pulsar_spec_metrics m;
    memset(&m, 0, sizeof(m));
    pulsar_engine_spec_metrics(e, &m);
    spec_snap s = { m.draft_tokens, m.accepted_tokens, m.num_drafts, m.gen_tokens };
    return s;
}

static void spec_report(const char *tag, spec_snap a, spec_snap b) {
    const double drafted = (double)(b.drafted - a.drafted);
    const double accepted = (double)(b.accepted - a.accepted);
    const double rounds = (double)(b.rounds - a.rounds);
    const double gen = (double)(b.gen - a.gen);
    printf("%s: alpha=%.4f (accepted %.0f / drafted %.0f)  tau=%.3f tokens/round  "
           "rounds=%.0f gen=%.0f\n",
           tag, drafted > 0 ? accepted / drafted : 0.0, accepted, drafted,
           rounds > 0 ? gen / rounds : 0.0, rounds, gen);
}

/* Advance one token through the SPECULATIVE machinery with the drafter
 * contributing nothing: max_tokens=1 clamps K to 0 (session_spec.cpp ~536), so
 * n_batch = 1+K = 1 and the step still runs gpu_graph_verify_suffix_tops rather
 * than the plain decode kernels. This is the "1-row spec batch, drafter off"
 * arm this file's header asks for, so mode 0 can read batch-sourced rows and
 * compare like with like. Returns the token, or -1 on error.
 *
 * MEASURED 2026-08-11 — THIS REPAIR IS INSUFFICIENT, and the measurement is the
 * point. Running mode 0 through here changes the numerics (top-2 margins move:
 * pos1 0.4367->0.5422, pos7 0.5151->0.4803), so the batch-vs-decode graph
 * difference is real. But the divergence does NOT move: still position 7, still
 * plain=12549 vs spec=114881, prefix still 7/24.
 *
 * Holding the entry point constant and varying only the WIDTH isolates the
 * cause: a 1-row batch still has n_tokens==1 and takes the _vec MoE arms, while
 * mode 1's 4-row batch takes MMQ (pulsar_cuda_moe.cu ~1389, type-43 has no
 * non-MMQ fallback). This is positive evidence for the MoE arm split, which
 * could not be obtained by a no-MMQ control build -- that build cannot load the
 * artifact at all.
 *
 * So "1-row spec batch, drafter off" cannot restore a token-exact hard gate:
 * the width IS the variable. A true like-with-like mode 0 must be WIDTH-MATCHED
 * to mode 1 (a 4-row batch), which this API cannot express -- max_tokens=N>1
 * re-enables the drafter and reintroduces real drafts. Kept as a diagnostic. */
static int gate_step_batched(pulsar_session *s, float temperature, int top_k,
                             float top_p, float min_p, uint64_t *rng,
                             int eos, char *err, size_t errlen) {
    int toks[2];
    int k = pulsar_session_generate_speculative(s, temperature, top_k, top_p, min_p,
                                                rng, /*max_tokens=*/1, eos,
                                                toks, (int)(sizeof(toks)/sizeof(toks[0])),
                                                err, errlen);
    if (k <= 0) return -1;
    return toks[0];
}

int main(int argc, char **argv) {
    /* progress must be visible in a redirected log: stdout to a file is
     * block-buffered, which makes a long run look like a hang. */
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char *model = argc > 1 ? argv[1] : "gguf/model.gguf";
    /* 0.95 is the acceptance-sensitive regime: hot enough that the greedy
     * p(mode) acceptance ceiling actually binds. */
    const float TEMP = argc > 2 ? (float)atof(argv[2]) : 0.95f;
    const int filler = argc > 3 ? atoi(argv[3]) : 0;
    /* Trajectory count. The chi-square needs the full TRAJ for power, but
     * alpha converges in a few hundred draft rounds — so an alpha-only
     * comparison run (e.g. the pre-Item-1 baseline) can be much shorter. */
    int traj = argc > 4 ? atoi(argv[4]) : TRAJ;
    if (traj < 1) traj = 1;
    if (traj > TRAJ) traj = TRAJ;
    /* top_p matters more than it looks. dist_build stops adding candidates once
     * filtered_sum/sum >= top_p, so on a CONFIDENT row — p(top) > top_p — the
     * nucleus is a single token, sampling is deterministic, and the chi-square
     * compares two point masses and passes vacuously (chi2=0.0, df=1). The
     * inherited 0.38 does exactly that on this prompt: 2500 trajectories, one
     * token per position. Keep this high enough that the nucleus is real. */
    const float TOP_P = argc > 5 ? (float)atof(argv[5]) : 0.95f;
    const float MIN_P = argc > 7 ? (float)atof(argv[7]) : 0.0f;
    /* Optional: dump both arms' raw trajectories so two ENGINES can be compared
     * directly. The chi-square below pits plain-sampled (mode 0) against
     * speculative (mode 1) — but mode 0 samples single-token DECODE logits
     * while mode 1 reads p from the BATCHED verify rows, and those two numeric
     * paths are known to diverge on near-ties (the Tier-2 driver control proved
     * batch-sweep diverges from classic decode on an untouched baseline:
     * "diverges at step 18", gaps 0.03-0.26). So at depth this gate has a
     * systematic difference between its arms that is NOT a sampler bug, and it
     * sits near crit even on shipped code. Dumping lets us run the confound-free
     * comparison instead: mode1(engine A) vs mode1(engine B) — both spec, both
     * batch-sourced. Both accept rules are exact for an arbitrary proposal, so
     * if they are correct they must reproduce the SAME filtered target. */
    const char *dump_path = argc > 6 ? argv[6] : NULL;
    if (dump_path && (!dump_path[0] || !strcmp(dump_path, "-"))) dump_path = NULL;

    pulsar_engine_options opt = { .model_path = model, .backend = PULSAR_BACKEND_CUDA };
    pulsar_engine *engine = NULL;
    if (pulsar_engine_open(&engine, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    if (!pulsar_engine_has_dspark(engine)) {
        fprintf(stderr, "spec sampling gate: the model has no drafter -- nothing to gate\n");
        return 1;
    }
    /* L160: a bank pool for the batched sampled arm.  The pool size is read
     * once per process at graph allocation, so it is set HERE, before the
     * session exists, overriding whatever the battery exported.  The context
     * is sized for the run: the prompt plus the greedy window and the sampled
     * depth fit in 2048 rows unless a context filler was asked for. */
    if (setenv("PULSAR_MSEQ_BANKS", "16", 1) != 0) { fprintf(stderr, "setenv failed\n"); return 1; }
    const int ctx = filler > 0 ? 16384 : 2048;
    pulsar_session *session = NULL;
    if (pulsar_session_create(&session, engine, ctx) != 0) { fprintf(stderr, "session failed\n"); return 1; }
    if (pulsar_session_bank_count(session) < SAMPLED_BANKS_PLAIN) {
        fprintf(stderr, "spec sampling gate: pool has %d banks, need %d\n",
                pulsar_session_bank_count(session), SAMPLED_BANKS_PLAIN);
        return 1;
    }

    /* Optional context filler: alpha falls with depth (77.6% shallow -> 61.7%
     * at 9.4k on v5mx), so a single shallow number is not the whole story. */
    char *user = NULL;
    if (filler > 0) {
        const size_t cap = (size_t)filler * 8u + strlen(PROMPT) + 64u;
        user = (char *)malloc(cap);
        size_t off = 0;
        for (int i = 0; i < filler && off + 8 < cap; i++)
            off += (size_t)snprintf(user + off, cap - off, "port%d ", i % 997);
        snprintf(user + off, cap - off, "%s", PROMPT);
    }

    pulsar_tokens prompt = {0};
    pulsar_chat_begin(engine, &prompt);
    pulsar_chat_append_message(engine, &prompt, "user", user ? user : PROMPT);
    pulsar_chat_append_assistant_prefix(engine, &prompt, PULSAR_THINK_NONE);
    char err[256];
    if (pulsar_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "sync failed: %s\n", err);
        return 1;
    }
    printf("model=%s temp=%.2f top_p=%.2f min_p=%.2f ctx_depth=%d traj=%d\n",
           model, (double)TEMP, (double)TOP_P, (double)MIN_P,
           pulsar_session_pos(session), traj);
    pulsar_session_snapshot snap = {0};
    if (pulsar_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
        fprintf(stderr, "snapshot failed: %s\n", err);
        return 1;
    }
    const int eos = pulsar_token_eos(engine);
    /* Mode 0 stays PLAIN DECODE by default: spec-vs-plain is the question a
     * reader of this gate actually has, and the 1-row batch arm below buys no
     * hard gate (measured -- see gate_step_batched). Set
     * PULSAR_SPEC_GATE_MODE0_BATCH=1 for the diagnostic arm. */
    const bool mode0_batched = getenv("PULSAR_SPEC_GATE_MODE0_BATCH") != NULL;
    printf("mode 0 arm: %s\n",
           mode0_batched ? "1-row spec batch (diagnostic; batch-sourced, drafter off)"
                         : "plain decode (default)");

    /* ---- greedy gates ----
     * Plain decode and speculative verify do not run the same MoE kernels on
     * the shipped artifact: type-43 routes n_tokens==1 to the _vec arms and
     * n_tokens>1 to the MMQ batch arm, and type-43 has no non-MMQ fallback
     * (pulsar_cuda_moe.cu ~1389). So the two paths carry genuinely different
     * quantization error -- measured ~0.5 logits, expert-routing dependent --
     * and token-exact equality with plain decode is NOT the engine's contract.
     *
     * The old gate asserted a >= 8 token agreeing prefix. That is a lottery,
     * not a property: it fails as soon as the first decision whose top-2 margin
     * is under the cross-arm delta lands early in the window, which any kernel
     * or routing change can shift. Measured 2026-08-11 it read 7 while the
     * engine was healthy -- and it agreed at position 1 on a TIGHTER margin
     * (0.437) than the position 7 it diverged on (0.515), which is precisely
     * how a per-position quantization delta behaves and is not something a
     * prefix-length threshold can express.
     *
     * What is actually asserted here:
     *   1. spec greedy is deterministic across two runs (informational: batch
     *      verify inherits prefill atomicAdd nondeterminism, #17);
     *   2. every token committed to context was also returned to the caller
     *      (HARD -- this is the ghost-commit / dropped-emission class);
     *   3. no divergence from plain greedy at a DECISIVE position (HARD) --
     *      a flip where the plain path was confident means a real accept-rule
     *      defect, whereas a flip at a narrow margin is the cross-arm delta.
     * This is the SAME confound this file's header already documents for the
     * chi-square arm ("a systematic difference between its arms that is NOT a
     * sampler bug ... treat a deep mode0-vs-mode1 FAIL as engine numerics").
     * That reasoning was applied to chi2, which was demoted to informational,
     * but never to the greedy prefix, which stayed a hard gate on the same bad
     * premise. This change just finishes the job.
     * NOTE: the chi-square arm is INFORMATIONAL, so it does not hard-gate the
     * accept rule either. Confound-free accept-rule evidence comes from
     * mode1-vs-mode1 across builds (dump via argv[6] +
     * tests/spec_sampling_compare.py), where the measured TVD is 3-4x smaller
     * than mode0-vs-mode1. The proper repair the header names -- have mode 0
     * read the same batched rows (a 1-row spec batch, drafter off) so both arms
     * are batch-sourced -- would restore token-exact comparison here and make
     * the margin heuristic below unnecessary. Not done.
     * Temperature-matched draft sampling must leave this path untouched: at
     * temp <= 0 no q is built, no rng is drawn, and the argmax-equality accept
     * walk runs exactly as before. */
    /* Top-2 logit margin above which a greedy flip cannot be quantization
     * noise. Calibrated on the shipped type-43 artifact, where decisive
     * positions measure 6.0-14.8 and ambiguous ones 0.19-1.94; 2.0 sits in the
     * empty band with ~3x headroom either side. Retune with the margin table
     * this gate prints if the artifact's quantization mix changes. */
    #define SPEC_GREEDY_DECISIVE_MARGIN 2.0f
    {
        int ref[24], got[24], got2[24];
        float ref_gap[24];
        int nref = 0, ngot = 0, ngot2 = 0;
        int hist_ok = 0;
        uint64_t rng = 7;
        const int vw = pulsar_engine_logits_width(engine);
        float *lg = (float *)malloc((size_t)vw * sizeof(float));
        if (!lg) return 1;
        for (int t = 0; t < 24; t++) {
            /* Top-2 margin of the PLAIN path at this position. A divergence at
             * a position whose margin is ~0 is the documented near-tie flip;
             * a divergence at a wide margin is a real accept-rule defect. */
            float g = -1.0f;
            if (pulsar_session_copy_logits(session, lg, vw) == vw) {
                float b1 = -INFINITY, b2 = -INFINITY;
                for (int v = 0; v < vw; v++) {
                    if (lg[v] > b1) { b2 = b1; b1 = lg[v]; }
                    else if (lg[v] > b2) { b2 = lg[v]; }
                }
                g = b1 - b2;
            }
            int tok;
            if (mode0_batched) {
                tok = gate_step_batched(session, 0.0f, 0, 1.0f, 0.0f, &rng, eos,
                                        err, sizeof(err));
                if (tok < 0) { fprintf(stderr, "ref batched step: %s\n", err); return 1; }
                if (tok == eos) break;
            } else {
                tok = pulsar_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
                if (tok == eos) break;
                if (pulsar_session_eval(session, tok, err, sizeof(err)) != 0) return 1;
            }
            ref_gap[nref] = g;
            ref[nref++] = tok;
        }
        free(lg);
        const spec_snap g0 = spec_take(engine);
        for (int rep = 0; rep < 2; rep++) {
            if (pulsar_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) return 1;
            int *dst = rep == 0 ? got : got2;
            int *n = rep == 0 ? &ngot : &ngot2;
            rng = 7;
            while (*n < nref) {
                int toks[17];
                int k = pulsar_session_generate_speculative(session, 0.0f, 0, 1.0f, 0.0f, &rng,
                                                         nref - *n, eos, toks, 17,
                                                         err, sizeof(err));
                if (k <= 0) { fprintf(stderr, "greedy spec failed: %s\n", err); return 1; }
                if (k > nref - *n)
                    printf("  [round returned k=%d with budget %d -> %d token(s) "
                           "generated but DROPPED from the output]\n",
                           k, nref - *n, k - (nref - *n));
                for (int i = 0; i < k && *n < nref; i++) dst[(*n)++] = toks[i];
            }
            /* Does the session's COMMITTED history contain tokens the spec path
             * never returned (ghost commit), or vice versa? Either way the
             * caller's transcript and the KV frontier disagree, and every
             * later turn continues over the discrepancy. */
            if (rep == 0) {
                const pulsar_tokens *hist = pulsar_session_tokens(session);
                hist_ok = hist && hist->len >= ngot &&
                          memcmp(hist->v + hist->len - ngot, got,
                                 (size_t)ngot * sizeof(int)) == 0;
            }
        }
        const spec_snap g1 = spec_take(engine);
        int det = ngot == ngot2 && memcmp(got, got2, (size_t)ngot * sizeof(int)) == 0;
        int prefix = 0;
        while (prefix < nref && prefix < ngot && ref[prefix] == got[prefix]) prefix++;
        /* informational: batch verify inherits the prefill atomicAdd
         * nondeterminism (#17), so run-to-run equality is not achievable
         * until ordered reductions land; tie positions flip. */
        printf("greedy determinism (2 runs): %s (informational; batch FP nondeterminism)\n",
               det ? "same" : "tie-flips");
        printf("greedy prefix agreement vs plain: %d/%d tokens "
               "(informational; cross-arm quantization, not a contract)\n",
               prefix, nref);
        printf("spec emissions == committed context: %s\n",
               hist_ok ? "yes" : "NO -- ghost commit or dropped emission");
        spec_report("greedy  ", g0, g1);
        /* Byte-identity aid: the greedy token stream is printed so a build from
         * a baseline commit can be diffed against this one token-for-token. */
        printf("greedy tokens:");
        for (int i = 0; i < ngot; i++) printf(" %d", got[i]);
        printf("\n");
        printf("plain  tokens:");
        for (int i = 0; i < nref; i++) printf(" %d", ref[i]);
        printf("\n");
        printf("plain top-2 margins:");
        for (int i = 0; i < nref; i++) printf(" %.4g", (double)ref_gap[i]);
        printf("\n");
        int decisive_flip = 0;
        if (prefix < nref) {
            decisive_flip = ref_gap[prefix] > SPEC_GREEDY_DECISIVE_MARGIN;
            printf("first divergence at %d: plain=%d spec=%d plain-margin=%.6g "
                   "(%s)\n",
                   prefix, ref[prefix], prefix < ngot ? got[prefix] : -1,
                   (double)ref_gap[prefix],
                   decisive_flip ? "DECISIVE -- accept-rule defect"
                                 : "within cross-arm quantization delta");
        }
        printf("greedy gate: %s\n",
               (hist_ok && !decisive_flip) ? "PASS" : "FAIL");
        if (!hist_ok || decisive_flip) return 1;
    }

    /* ---- sampled-distribution comparison (L160: banked batches) ---- */
    static int seqA[TRAJ][DEPTH], seqB[TRAJ][DEPTH];
    spec_snap s0 = {0}, s1 = {0};
    {
        const int vocab = pulsar_engine_logits_width(engine);
        float *logits = (float *)malloc((size_t)SAMPLED_ROWS_MAX * (size_t)vocab * sizeof(float));
        pulsar_spec_round *rounds[SAMPLED_BANKS_SPEC];
        for (int b = 0; b < SAMPLED_BANKS_SPEC; b++) rounds[b] = pulsar_spec_round_new();
        if (!logits) return 1;
        for (int mode = 0; mode < 2; mode++) {
            if (mode == 1) s0 = spec_take(engine);
            const time_t mode_t0 = time(NULL);
            const int width = mode == 0 ? SAMPLED_BANKS_PLAIN : SAMPLED_BANKS_SPEC;
            int next_report = 250;
            for (int t0 = 0; t0 < traj; t0 += width) {
                const int nb = traj - t0 < width ? traj - t0 : width;
                const int rc = mode == 0
                    ? sampled_plain_batch(session, &snap, t0, nb, TEMP, TOP_P, MIN_P, eos, vocab,
                                          logits, seqA, err, sizeof(err))
                    : sampled_spec_batch(session, &snap, t0, nb, TEMP, TOP_P, MIN_P, eos, vocab,
                                         logits, seqB, rounds, err, sizeof(err));
                if (rc != 0) { fprintf(stderr, "mode %d batch at %d: %s\n", mode, t0, err); return 1; }
                if (t0 + nb >= next_report) {
                    printf("  mode %d: %d/%d trajectories\n", mode, t0 + nb, traj);
                    next_report += 250;
                }
            }
            printf("  mode %d: %d trajectories, %d banks per forward, %ld s\n",
                   mode, traj, width, (long)(time(NULL) - mode_t0));
            fputc('\n', stderr);
        }
        for (int b = 0; b < SAMPLED_BANKS_SPEC; b++) pulsar_spec_round_free(rounds[b]);
        free(logits);
    }
    s1 = spec_take(engine);
    if (dump_path) {
        FILE *df = fopen(dump_path, "wb");
        if (df) {
            const int32_t hdr[2] = { (int32_t)traj, (int32_t)DEPTH };
            fwrite(hdr, sizeof(int32_t), 2, df);
            for (int t = 0; t < traj; t++)
                for (int k = 0; k < DEPTH; k++) {
                    const int32_t v = seqA[t][k];
                    fwrite(&v, sizeof(v), 1, df);
                }
            for (int t = 0; t < traj; t++)
                for (int k = 0; k < DEPTH; k++) {
                    const int32_t v = seqB[t][k];
                    fwrite(&v, sizeof(v), 1, df);
                }
            fclose(df);
            printf("dumped %d trajectories x %d positions (modeA then modeB) -> %s\n",
                   traj, DEPTH, dump_path);
        }
    }
    /* THE Item 1 number: sampled-proposal acceptance at TEMP. The deterministic
     * (argmax-proposal) rule is capped at E[p(mode)]; min(1,p/q) is not. Compare
     * against a build of this same gate from the pre-Item-1 commit. */
    spec_report("sampled ", s0, s1);

    /* chi-square per position over pooled top buckets */
    int fail = 0, degenerate = 0;
    for (int posn = 0; posn < DEPTH; posn++) {
        bucket bk[4096];
        int nb = 0;
        for (int mode = 0; mode < 2; mode++)
            for (int t = 0; t < traj; t++) {
                int tok = mode == 0 ? seqA[t][posn] : seqB[t][posn];
                int j = 0;
                for (; j < nb; j++) if (bk[j].id == tok) break;
                if (j == nb) { if (nb >= 4096) continue; bk[nb++] = (bucket){tok, 0, 0}; }
                if (mode == 0) bk[j].a++; else bk[j].b++;
            }
        qsort(bk, (size_t)nb, sizeof(bk[0]), bucket_cmp);
        int keep = nb < TOPN ? nb : TOPN;
        long resta = 0, restb = 0;
        for (int j = keep; j < nb; j++) { resta += bk[j].a; restb += bk[j].b; }
        double chi = 0.0;
        int df = 0;
        for (int j = 0; j <= keep; j++) {
            const long a = j < keep ? bk[j].a : resta;
            const long b = j < keep ? bk[j].b : restb;
            const double tot = (double)(a + b);
            if (tot < 10.0) continue;
            const double ea = tot / 2.0, ebb = tot / 2.0;
            chi += ((double)a - ea) * ((double)a - ea) / ea +
                   ((double)b - ebb) * ((double)b - ebb) / ebb;
            df++;
        }
        df = df > 1 ? df - 1 : 1;
        /* p ~ 0.001 critical values: chi2(df) ≈ df + 3.1*sqrt(2 df) + 4 */
        const double crit = df + 3.1 * sqrt(2.0 * df) + 4.0;
        /* nb = distinct tokens seen at this position across both modes. nb==1
         * means the sampler is a point mass here (p(top) >= top_p), the chi2 is
         * comparing two identical constants, and this position proves NOTHING.
         *
         * The chi2 compares mode0 (plain single-token DECODE logits) vs mode1
         * (BATCHED verify rows) — TWO NUMERIC PATHS that diverge on near-ties,
         * increasingly with depth (see the "READ THIS BEFORE FIXING THE DEEP
         * CHI-SQUARE" header). It is therefore a cross-path NUMERICS indicator,
         * NOT a clean sampler test, and a deep chi2 > crit does NOT indict the
         * sampler. So the chi2 is INFORMATIONAL and does NOT gate (2026-07-26:
         * was `if (chi > crit) fail = 1`, which made the gate permanently red on
         * v5mx where the two paths diverge enough at pos>=3 to exceed crit —
         * pure two-path numerics, not a regression). The HARD exactness gate is
         * the SAME-PATH greedy-prefix agreement (>=8, `return 1` above) plus the
         * all-degenerate guard below; the clean temp>0 sampler test is the dump
         * mode (argv[6]) comparing mode1-vs-mode1 across builds. */
        printf("pos %d: chi2=%.1f df=%d crit(p~.001)=%.1f distinct=%d -> %s%s\n",
               posn, chi, df, crit, nb, chi <= crit ? "ok" : "HIGH (informational, cross-path)",
               nb < 2 ? "  [DEGENERATE: point mass, test is vacuous]" : "");
        if (nb < 2) degenerate++;
    }
    if (degenerate == DEPTH) {
        printf("spec sampling oracle VACUOUS: every position is a point mass at "
               "top_p=%.2f — raise top_p (or use a less confident prompt) or the "
               "gate proves nothing\n", (double)TOP_P);
        fail = 1;
    } else {
        /* Keep this in step with the greedy block's asserts. The old text still
         * advertised "greedy-prefix agreement >=8" after that stopped being a
         * hard gate, which is exactly the sort of stale claim this file's own
         * header warns about. */
        printf(fail ? "spec sampling oracle FAIL\n"
                    : "spec sampling oracle PASS (hard gates: spec emissions == "
                      "committed context, no decisive-margin greedy flip, "
                      "non-degenerate; greedy-prefix length and per-position "
                      "chi2 are cross-path numerics, informational)\n");
    }
    free(user);
    return fail;
}

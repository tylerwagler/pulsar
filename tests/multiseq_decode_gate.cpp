/* Tier-2 MULTISEQ decode gate: co-scheduling neutrality, mixed-entry byte
 * identity, the ALL_ROWS head mode, the stale-classic guard.
 *
 * MODEL-DEPENDENT: run on the GB10 via `make cuda-multiseq-gate` (memory
 * discipline in the Makefile target) -- NOT part of `make test`.
 *
 * WHAT THIS GATE ASSERTS
 * ----------------------
 *  1. CO-SCHEDULING NEUTRALITY (the reason the gate exists): a session's
 *     emitted token stream must not depend on WHICH other sessions share its
 *     batch, nor on how many -- bank k's stream over STEPS greedy steps at
 *     N=2 must be token-identical to bank k's stream at N=3.  N=1 is not in
 *     the comparison: a 1-row batch dispatches the single-row kernel tiers,
 *     so N=1 against N>=2 compares two kernel paths rather than
 *     co-scheduling (cuda-row-neutrality-gate compares those, on logits).
 *  2. MIXED-ENTRY IDENTITY: pulsar_session_decode_mixed (heap descriptors,
 *     the server worker's entry) must equal pulsar_session_decode_multiseq
 *     (stack descriptors) for a decode-only batch -- token streams over
 *     GATE_MIX_STEPS steps and the step-1 logits, byte for byte, at every
 *     width 2..MAXN.  The risk is descriptor addressing and lifetime, which
 *     the first step exposes; the stream leg is a short walk past it.
 *  3. ALL_ROWS HEAD MODE changes only the emission set, never the values.
 *  4. STALE CLASSIC STATE fails loud: after a multiseq step the classic
 *     single-bank entries must refuse, and a fresh pulsar_session_sync (the
 *     documented recovery, available to public callers) clears the condition.
 *
 * FIXTURE.  One session for assertions 1-3 (the stale guard takes its own,
 * see there).  Each prompt is prefilled ONCE,
 * cold, into its bank and a snapshot of that bank is taken; the N=MAXN run
 * decodes from those banks as populated.  Every later populate is a snapshot
 * load into the bank (repoint, invalidate, load, bank_state_save -- the reset
 * L160 measured byte-identical to a fresh trajectory), so the gate prefills
 * MAXN + 1 times instead of once per bank per run.  Assertion 2 at N=MAXN
 * compares a snapshot-loaded run against the cold-prefilled one, so it also
 * witnesses load == cold prefill on this tree (cuda-fork-gate is the
 * dedicated check for the fork form of that identity).
 *
 * WHAT THIS GATE IS BLIND TO (do not add it here -- it is S6's job in the
 * frontier gate): per-row POSITIONS.  reqs[k].pos below is a function of the
 * bank k and the step j, never of the batch width n, so a kernel that ignored
 * positions[] entirely would apply the same wrong rotation at both widths and
 * the neutrality comparison would still pass.  Batched-vs-classic divergence
 * is neither asserted nor measured here: the batch sweep and classic decode
 * were never bit-identical (near-tie argmax flips walk the streams apart),
 * and the single-row and multi-row tiers are compared on logits by
 * cuda-row-neutrality-gate.
 *
 * DSPARK: `PULSAR_GATE_NO_DSPARK=1` opens the engine with speculation disabled
 * (the pulsar-bench/pulsar-eval/agent and `pulsar-server --no-dspark` config,
 * and a different allocation shape).  The driver must work there -- it is
 * plain decode by contract at n >= 2 and must not depend on the speculation
 * machinery having been initialized.  Run via `make cuda-multiseq-gate-nodspark`.
 *
 * usage: PULSAR_MSEQ_BANKS=3 ./tests/multiseq_decode_gate MODEL [MAXN] [STEPS]
 *        (from the repo root -- reads tests/long_context_story_prompt.txt;
 *         MAXN >= 2; assertion 1 needs MAXN >= 3)
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "gate_fixture.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GATE_MAX_N 8
#define GATE_MAX_STEPS 2048
#define GATE_MIX_STEPS 64   /* assertion 2's stream leg (capped by STEPS) */

static pulsar_engine *g_e;
static pulsar_tokens g_toks;
static int g_fail;
static pulsar_session_snapshot g_snap[GATE_MAX_N];   /* prompt k, prefilled */
static int g_first_tok[GATE_MAX_N];                  /* argmax after that prefill */

/* Distinct prompts: different offsets AND different lengths so co-scheduled
 * banks sit at unrelated positions from step one. */
static const int g_prompt_off[GATE_MAX_N] = {0, 401, 907, 233, 601, 811, 101, 499};
static const int g_prompt_len[GATE_MAX_N] = {130, 258, 511, 187, 342, 419, 275, 158};

#define CHECK(cond, ...) do { \
        if (!(cond)) { \
            fprintf(stderr, "MULTISEQ GATE FAIL: " __VA_ARGS__); \
            fprintf(stderr, "\n"); \
            g_fail = 1; \
        } \
    } while (0)

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* The cold path, once per prompt: prefill prompt k into bank k through the
 * classic path and snapshot the bank. */
static bool bank_prefill_and_snapshot(pulsar_session *s, int k) {
    if (!gate_populate_bank(s, (uint32_t)k, g_toks.v + g_prompt_off[k], g_prompt_len[k],
                            &g_first_tok[k], "populate"))
        return false;
    char err[256];
    if (pulsar_session_save_snapshot(s, &g_snap[k], err, sizeof err) != 0) {
        fprintf(stderr, "snapshot of prompt %d failed: %s\n", k, err);
        return false;
    }
    return true;
}

/* Put prompt k into bank `bank` from its snapshot: repoint the device views,
 * drop the live bookkeeping, load, persist as the bank's carry. */
static bool bank_load(pulsar_session *s, uint32_t bank, int k, const char *what) {
    char err[256];
    if (pulsar_session_bank_repoint(s, bank) != 0) {
        fprintf(stderr, "%s: bank %u repoint failed\n", what, bank);
        return false;
    }
    pulsar_session_invalidate(s);
    if (pulsar_session_load_snapshot(s, &g_snap[k], err, sizeof err) != 0) {
        fprintf(stderr, "%s: bank %u load of prompt %d failed: %s\n", what, bank, k, err);
        return false;
    }
    pulsar_session_bank_state_save(s, bank);
    return true;
}

/* One run at batch width n: banks 0..n-1 hold prompts 0..n-1 (loaded here
 * unless banks_ready says they already do), then `steps` self-fed greedy
 * steps through the engine entry.  streams[k] gets steps+1 tokens; [0] is the
 * prefill continuation.  use_mixed routes the step through
 * pulsar_session_decode_mixed instead of pulsar_session_decode_multiseq.
 * first_logits_out (optional, n*vocab floats) receives the STEP-1 logits. */
static bool multi_run(pulsar_session *s, int n, int steps, int **streams, double *secs,
                      bool use_mixed, float *first_logits_out, bool banks_ready) {
    if (!gate_pool_fits(s, (uint32_t)n)) return false;
    const int vocab = (int)PULSAR_N_VOCAB;   /* the engine's logits row width */
    char err[256];
    for (int k = 0; k < n; k++) {
        if (!banks_ready && !bank_load(s, (uint32_t)k, k, "populate")) return false;
        streams[k][0] = g_first_tok[k];
    }
    float *logits = (float *)malloc((size_t)n * vocab * sizeof(float));
    if (!logits) return false;
    bool ok = true;
    const double t0 = now_s();
    for (int j = 1; ok && j <= steps; j++) {
        pulsar_multiseq_req reqs[GATE_MAX_N];
        for (int k = 0; k < n; k++) {
            reqs[k].bank = (uint32_t)k;
            reqs[k].pos = g_prompt_len[k] + (j - 1);
            reqs[k].token = streams[k][j - 1];
        }
        const int rc = use_mixed
            ? pulsar_session_decode_mixed(s, reqs, (uint32_t)n,
                                          logits, n * vocab, NULL, 0u, err, sizeof(err))
            : pulsar_session_decode_multiseq(s, reqs, (uint32_t)n,
                                             logits, n * vocab, err, sizeof(err));
        if (rc != 0) {
            fprintf(stderr, "%s step %d failed (rc=%d): %s\n",
                    use_mixed ? "mixed" : "multiseq", j, rc, err);
            ok = false;
            break;
        }
        if (j == 1 && first_logits_out)
            memcpy(first_logits_out, logits, (size_t)n * vocab * sizeof(float));
        for (int k = 0; k < n; k++)
            streams[k][j] = (int)argmax_f32(logits + (size_t)k * vocab, (uint64_t)vocab);
    }
    if (secs) *secs = now_s() - t0;
    free(logits);
    return ok;
}

/* HARD GATE: the classic single-bank entries must FAIL LOUD after a multiseq
 * step, not silently corrupt.
 *
 * After a multiseq step the graph's scalar frontier counters hold a
 * cross-bank SUPERSET, not any single bank's truth.  pulsar_session_eval
 * decodes against those scalars unconditionally: it would emit its
 * compressor row at the superset index and attend over the rows below it --
 * a previous tenant's bytes when the live bank's frontier is lower.  Wrong
 * logits, silently.  pulsar_session_decode_multiseq is public, so a server
 * is exactly the caller that hits this.
 *
 * Asserted: (1) eval fails after a multiseq step, (2) the speculative entries
 * fail too (same counters), (3) a fresh pulsar_session_sync -- the documented
 * escape hatch, available to public callers -- clears the condition and eval
 * works again.  On its OWN session: the pre-check needs one that has never
 * run a multiseq step (the dirty state it asserts on later is per session and
 * only a sync clears it), and the recovery leaves it classic. */
static bool check_stale_classic_fails_loud(void) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) return false;
    pulsar_gpu_graph *g = &s->graph;
    if ((int)gpu_graph_bank_pool_count(g) < 2) {
        printf("STALE-GUARD: skipped (pool %u < 2)\n", gpu_graph_bank_pool_count(g));
        pulsar_session_free(s);
        return true;
    }
    const int vocab = pulsar_engine_logits_width(g_e);
    char err[256];
    bool ok = true;
    int last[2] = {0, 0};
    for (int k = 0; ok && k < 2; k++) {
        ok = bank_load(s, (uint32_t)k, k, "stale-guard populate");
        last[k] = g_first_tok[k];
    }
    float *logits = ok ? (float *)malloc((size_t)2 * vocab * sizeof(float)) : NULL;
    if (ok && !logits) ok = false;
    if (ok) {
        /* Sanity: eval works BEFORE any multiseq step (so a later failure is
         * the guard firing, not an unrelated broken session). */
        if (pulsar_session_eval(s, last[1], err, sizeof(err)) != 0) {
            CHECK(0, "stale-guard: eval failed BEFORE any multiseq step (%s)", err);
            ok = false;
        }
    }
    if (ok) {
        /* Re-establish bank 1's frontier: the eval above advanced it. */
        ok = bank_load(s, 1u, 1, "stale-guard re-populate");
        if (!ok) CHECK(0, "stale-guard: bank 1 re-load failed");
    }
    if (ok) {
        pulsar_multiseq_req reqs[2];
        for (int k = 0; k < 2; k++) {
            reqs[k].bank = (uint32_t)k;
            reqs[k].pos = g_prompt_len[k];
            reqs[k].token = last[k];
        }
        const int rc = pulsar_session_decode_multiseq(s, reqs, 2, logits,
                                                      2 * vocab, err, sizeof(err));
        if (rc != 0) {
            CHECK(0, "stale-guard: multiseq step failed (rc=%d): %s", rc, err);
            ok = false;
        }
    }
    if (ok) {
        err[0] = '\0';
        const int rc = pulsar_session_eval(s, last[0], err, sizeof(err));
        CHECK(rc != 0,
              "STALE CLASSIC STATE NOT CAUGHT: pulsar_session_eval SUCCEEDED after "
              "a multiseq step -- it decoded against cross-bank superset frontier "
              "counters and silently corrupted KV");
        if (rc != 0) printf("STALE-GUARD: eval after multiseq fails loud OK (%s)\n", err);

        err[0] = '\0';
        int acc[4];
        const int rcs = pulsar_session_eval_speculative_block(s, last[0], 4, -1, acc,
                                                              4, err, sizeof(err));
        CHECK(rcs <= 0,
              "STALE CLASSIC STATE NOT CAUGHT: pulsar_session_eval_speculative_block "
              "SUCCEEDED after a multiseq step (returned %d)", rcs);
        if (rcs <= 0) printf("STALE-GUARD: spec block eval after multiseq fails loud OK\n");
    }
    if (ok) {
        /* The escape hatch: a fresh sync rebuilds per-bank state (reset zeroes
         * the counters) and classic decode must work again.  This is the one
         * populate in the gate that MUST be a real sync -- the recovery path
         * under test. */
        pulsar_tokens p = { .v = g_toks.v + g_prompt_off[0], .len = g_prompt_len[0],
                            .cap = g_prompt_len[0] };
        if (pulsar_session_sync(s, &p, err, sizeof(err)) != 0) {
            CHECK(0, "stale-guard: re-sync after multiseq failed: %s", err);
        } else {
            CHECK(pulsar_session_eval(s, pulsar_session_argmax(s), err, sizeof(err)) == 0,
                  "stale-guard: eval still refused after a re-sync -- the "
                  "documented recovery path does not clear the condition (%s)",
                  err);
            printf("STALE-GUARD: re-sync clears the condition, classic eval "
                   "works again OK\n");
        }
    }
    free(logits);
    pulsar_session_free(s);
    return ok;
}

/* HARD GATE: the ALL_ROWS head mode may change ONLY the emission set, never
 * the values.  Two identical states (banks 0 and 1 loaded from their
 * snapshots) run the same verify-shaped batch (bank0 = a 3-row run, bank1 =
 * 1 row); the last-of-run call emits 2 rows, the ALL_ROWS call emits 4, and
 * the rows both modes emit must be BYTE-identical (same forward, same hidden
 * rows, same head -- the mode only selects which rows are headed).  This is
 * the contract the batched speculative verify's accept walk stands on. */
static bool check_all_rows_head_mode(pulsar_session *s) {
    if (gpu_graph_bank_pool_count(&s->graph) < 2) {
        fprintf(stderr, "all-rows gate: pool too small\n");
        return false;
    }
    const int vocab = (int)PULSAR_N_VOCAB;
    float *rows_lor = (float *)malloc((size_t)2 * vocab * sizeof(float));
    float *rows_all = (float *)malloc((size_t)4 * vocab * sizeof(float));
    if (!rows_lor || !rows_all) { free(rows_lor); free(rows_all); return false; }
    bool ok = true;
    for (int mode = 0; ok && mode < 2; mode++) {
        char err[256];
        for (int k = 0; ok && k < 2; k++)
            ok = bank_load(s, (uint32_t)k, k, "all-rows populate");
        if (!ok) break;
        /* bank0: 3-row teacher-forced run (draft shape); bank1: 1 row.  The
         * token VALUES only need to be legal and identical across the two
         * modes -- the invariant under test is emission, not quality. */
        pulsar_multiseq_req reqs[4];
        for (int j = 0; j < 3; j++) {
            reqs[j].bank = 0u;
            reqs[j].pos = g_prompt_len[0] + j;
            reqs[j].token = g_first_tok[0];
        }
        reqs[3].bank = 1u;
        reqs[3].pos = g_prompt_len[1];
        reqs[3].token = g_first_tok[1];
        uint32_t got = 0;
        const int rc = pulsar_session_decode_mixed(s, reqs, 4u,
                mode == 0 ? rows_lor : rows_all,
                (mode == 0 ? 2 : 4) * vocab, &got,
                mode == 0 ? 0u : PULSAR_MSEQ_HEAD_ALL_ROWS,
                err, sizeof(err));
        if (rc != 0) {
            fprintf(stderr, "all-rows gate: decode_mixed mode %d failed (rc=%d): %s\n",
                    mode, rc, err);
            ok = false;
        } else if (got != (mode == 0 ? 2u : 4u)) {
            fprintf(stderr, "all-rows gate: mode %d emitted %u rows (want %u)\n",
                    mode, got, mode == 0 ? 2u : 4u);
            ok = false;
        }
    }
    if (ok) {
        const size_t rb = (size_t)vocab * sizeof(float);
        const int d_last = memcmp(rows_all + (size_t)2 * vocab, rows_lor, rb);
        const int d_b1   = memcmp(rows_all + (size_t)3 * vocab, rows_lor + (size_t)1 * vocab, rb);
        CHECK(d_last == 0,
              "ALL_ROWS row 2 (bank0 run-last) != last-of-run row 0 -- the head "
              "mode changed VALUES, not just the emission set");
        CHECK(d_b1 == 0,
              "ALL_ROWS row 3 (bank1) != last-of-run row 1 -- the head mode "
              "changed VALUES, not just the emission set");
        int finite = 1;
        for (int j = 0; j < 2 * vocab; j += 977)
            if (!(rows_all[j] == rows_all[j])) finite = 0;
        CHECK(finite, "ALL_ROWS intermediate rows contain NaN");
        if (d_last == 0 && d_b1 == 0 && finite)
            printf("ALL-ROWS HEAD MODE: emission-only (rows byte-identical to "
                   "last-of-run; 4 rows emitted, intermediates finite)\n");
    }
    free(rows_lor);
    free(rows_all);
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL [MAXN] [STEPS]\n", argv[0]);
        return 2;
    }
    const int maxn = argc > 2 ? atoi(argv[2]) : 3;
    const int steps = argc > 3 ? atoi(argv[3]) : 512;
    if (maxn < 2 || maxn > GATE_MAX_N || steps < 1 || steps > GATE_MAX_STEPS) {
        fprintf(stderr, "bad MAXN/STEPS (MAXN 2..%d, STEPS 1..%d)\n", GATE_MAX_N, GATE_MAX_STEPS);
        return 2;
    }
    const int mix_steps = steps < GATE_MIX_STEPS ? steps : GATE_MIX_STEPS;

    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    /* PULSAR_GATE_NO_DSPARK: run the whole gate with speculation DISABLED, the
     * way pulsar-bench/pulsar-eval/pulsar-agent and `pulsar-server --no-dspark`
     * open the engine.  A real serving config and a DIFFERENT allocation
     * shape: the DSpark graph state is never allocated, and the driver once
     * rejected every step here because the shared multi-row logits slab was
     * allocated only as a side effect of DSpark init.  Read once at startup
     * (the test's own config, not a hot path). */
    if (getenv("PULSAR_GATE_NO_DSPARK") != NULL) {
        opt.dspark_disable = true;
        printf("CONFIG: DSpark DISABLED (dspark_disable=1) -- the driver must "
               "work with no speculation machinery allocated\n");
    }
    if (pulsar_engine_open(&g_e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    if (getenv("PULSAR_GATE_NO_DSPARK") != NULL && pulsar_engine_has_dspark(g_e)) {
        fprintf(stderr, "MULTISEQ GATE FAIL: PULSAR_GATE_NO_DSPARK set but the "
                        "engine still reports a drafter -- the no-dspark case "
                        "is not actually being exercised\n");
        return 1;
    }

    int need = 0;
    for (int k = 0; k < maxn; k++) {
        if (g_prompt_off[k] + g_prompt_len[k] > need) need = g_prompt_off[k] + g_prompt_len[k];
    }
    if (!gate_load_story(g_e, &g_toks, need)) return 1;

    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) { fprintf(stderr, "session create failed\n"); return 1; }
    if (!gate_pool_fits(s, (uint32_t)maxn)) return 1;

    /* The cold prefills: prompt k into bank k, snapshotted. */
    {
        const double t0 = now_s();
        for (int k = 0; k < maxn; k++) {
            if (!bank_prefill_and_snapshot(s, k)) { fprintf(stderr, "prefill %d failed\n", k); return 1; }
        }
        printf("prefilled %d prompts (%d tokens) in %.1fs; every later populate is a "
               "snapshot load\n", maxn, need, now_s() - t0);
    }

    /* Multiseq runs at N = MAXN..2 (the banks already hold the prompts for the
     * first); every stream is kept for the cross-N gate. */
    int *multi[GATE_MAX_N][GATE_MAX_N];   /* [n-1][bank] */
    bool have[GATE_MAX_N];
    const int vocab_w = (int)PULSAR_N_VOCAB;
    float *ref_l1[GATE_MAX_N];            /* [n-1] step-1 logits, multiseq entry */
    memset(multi, 0, sizeof(multi));
    memset(have, 0, sizeof(have));
    memset(ref_l1, 0, sizeof(ref_l1));
    for (int n = maxn; n >= 2; n--) {
        for (int k = 0; k < n; k++) multi[n - 1][k] = (int *)malloc((size_t)(steps + 1) * sizeof(int));
        ref_l1[n - 1] = (float *)malloc((size_t)n * vocab_w * sizeof(float));
        double secs = 0.0;
        if (!multi_run(s, n, steps, multi[n - 1], &secs, false, ref_l1[n - 1], n == maxn)) {
            CHECK(0, "N=%d: multi run failed", n);
            continue;
        }
        have[n - 1] = true;
        printf("N=%d: %d steps x %d sessions in %.1fs -> aggregate %.2f tok/s "
               "(%.2f tok/s/session)\n",
               n, steps, n, secs, (double)n * steps / secs, (double)steps / secs);
    }

    /* HARD GATE 2: the mixed-descriptor entry must be BYTE-IDENTICAL to the
     * multiseq entry for a decode-only batch -- the per-bank token STREAM over
     * mix_steps steps and the STEP-1 logits (memcmp of the float rows).  A
     * heap-scratch addressing/lifetime bug surfaces here. */
    for (int n = 2; n <= maxn; n++) {
        if (!have[n - 1]) continue;
        int *mix[GATE_MAX_N];
        memset(mix, 0, sizeof(mix));
        for (int k = 0; k < n; k++) mix[k] = (int *)malloc((size_t)(mix_steps + 1) * sizeof(int));
        float *mix_l1 = (float *)malloc((size_t)n * vocab_w * sizeof(float));
        if (!multi_run(s, n, mix_steps, mix, NULL, true, mix_l1, false)) {
            CHECK(0, "N=%d: mixed-entry run failed", n);
        } else {
            int sdiff = -1;
            for (int k = 0; k < n && sdiff < 0; k++)
                for (int j = 0; j <= mix_steps; j++)
                    if (mix[k][j] != multi[n - 1][k][j]) { sdiff = j; break; }
            CHECK(sdiff < 0,
                  "MIXED ENTRY NOT byte-identical: N=%d token stream diverges from "
                  "the multiseq entry at step %d -- heap descriptor addressing is wrong",
                  n, sdiff);
            const size_t lb = (size_t)n * vocab_w * sizeof(float);
            const int ldiff = memcmp(mix_l1, ref_l1[n - 1], lb);
            CHECK(ldiff == 0,
                  "MIXED ENTRY NOT byte-identical: N=%d STEP-1 logits differ from "
                  "the multiseq entry (memcmp != 0)", n);
            if (sdiff < 0 && ldiff == 0)
                printf("MIXED-ENTRY EQUIV: N=%d decode_mixed == decode_multiseq "
                       "(streams over %d tokens + step-1 logits byte-identical%s)\n",
                       n, mix_steps + 1,
                       n == maxn ? "; snapshot-loaded banks == cold-prefilled" : "");
        }
        for (int k = 0; k < n; k++) free(mix[k]);
        free(mix_l1);
    }

    /* HARD GATE 1: co-scheduling neutrality -- bank k's stream must not
     * depend on which/how many OTHER sessions share the batch (n >= 2). */
    for (int n = 3; n <= maxn; n++) {
        if (!have[n - 1] || !have[1]) continue;
        for (int k = 0; k < 2; k++) {
            int diff = -1;
            for (int j = 0; j <= steps; j++) {
                if (multi[n - 1][k][j] != multi[1][k][j]) { diff = j; break; }
            }
            CHECK(diff < 0,
                  "co-scheduling NOT neutral: bank %d's stream at N=%d differs "
                  "from its stream at N=2 at step %d (%d vs %d) -- a batchmate "
                  "changed another session's tokens",
                  k, n, diff,
                  diff >= 0 ? multi[n - 1][k][diff] : -1,
                  diff >= 0 ? multi[1][k][diff] : -1);
            if (diff < 0) {
                printf("CO-SCHED NEUTRALITY: bank %d identical at N=2 and N=%d "
                       "(%d tokens)\n", k, n, steps + 1);
            }
        }
    }

    if (!check_all_rows_head_mode(s)) CHECK(0, "all-rows head-mode gate failed to run");

    if (!check_stale_classic_fails_loud()) CHECK(0, "stale-classic guard check failed to run");

    for (int n = 0; n < GATE_MAX_N; n++) free(ref_l1[n]);
    for (int n = 0; n < GATE_MAX_N; n++)
        for (int k = 0; k < GATE_MAX_N; k++) free(multi[n][k]);
    for (int k = 0; k < maxn; k++) pulsar_session_snapshot_free(&g_snap[k]);
    pulsar_session_free(s);
    pulsar_engine_close(g_e);
    if (g_fail) { fprintf(stderr, "MULTISEQ DECODE GATE: FAIL\n"); return 1; }
    printf("MULTISEQ DECODE GATE: PASS\n");
    return 0;
}

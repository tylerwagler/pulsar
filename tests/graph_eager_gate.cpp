/* L357: a replayed step graph computes what the eager step computes, on the core step driver's lane (every family
 * that stages its steps) -- the intent of the DeepSeek-only tests/head_graph_replay_gate.cpp moved to the driver.
 *
 *   ./tests/graph_eager_gate <model>
 *
 * The driver runs a pure DECODE step and a pure VERIFY step (every row a verify row, every row headed) as one captured
 * CUDA graph per key (src/engine/step_driver.cpp driver_graph_key: the step's kind, its rows, its head list, the
 * family's key; the L119 segment cache captures a key's 12th use and replays it after).  A decode step and a verify
 * step of the SAME row count head the same rows, so a key that lost the row kind would replay one kind's graph for
 * the other -- the verify's capture of the recurrent state missing, or present where it must not be.
 *
 * One session, ROWS banks, each prefilled with its own stretch of the story prompt.  The script alternates
 *   decode  one row a bank (pulsar_session_decode_mixed, max_head_runs 0): ROWS rows, all headed;
 *   verify  one row a bank, every row headed (PULSAR_MSEQ_HEAD_ALL_ROWS): ROWS rows, all headed -- then each bank's
 *           run committed with its one row kept (the round's commit, pulsar_state_commit + the bank's position: the
 *           one reach past the public API, as the driver's spec commit does it), so every step advances every bank
 * for STEPS_PER_KIND steps of each kind, every token teacher-forced from the story, so every step's inputs are the
 * same in every run:
 *   run A  EAGER: the segment cache off (pulsar_gpu_seg_set_enabled(0); asserted: nothing captured or replayed);
 *   run B  GRAPHS: the production lane (asserted: both kinds' graphs replayed);
 *   run C  EAGER again: run A's determinism, the comparison's premise.
 * PASS: every headed row of every step is byte-identical between B and A, and between C and A.  A family whose
 * driver does not stage its steps runs them eagerly only (step_driver.cpp graphable): the runner skips it by name
 * (NEED_STEP_GRAPHS) and the gate itself refuses one (no replay to grade).  A runner gate (tests/gates_runner.cpp
 * `graph-eager-gate`). */
#include "pulsar.h"
#include "gate_entry.h"
#include "gate_fixture.h"   /* the declared state: the verify step's commit */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static int n_fail = 0;
#define CHECK(c, ...) GATE_CHECK(n_fail, c, __VA_ARGS__)

#define ROWS           4u    /* both kinds' row count (and the banks) */
#define STEPS_PER_KIND 20u   /* > the capture threshold (12): each kind captures, then replays 8 times */
#define N_STEPS        (2u * STEPS_PER_KIND)

/* each bank's stretch of the story and its prompt length */
static const int bank_off[ROWS] = {0, 1200, 2400, 3600};
static const int bank_pre[ROWS] = {300, 170, 230, 100};

/* step t's kind: even steps decode, odd steps verify */
static bool step_verifies(uint32_t t) { return (t & 1u) != 0; }

/** One run's record: every headed row's logits and which steps replayed a graph. */
struct run_rec {
    std::vector<float> lg;          ///< [N_STEPS][ROWS][W]
    uint8_t replayed[N_STEPS] = {};
    uint64_t replays = 0, captures = 0;
};

/* The script on a fresh session, the segment cache on (graphs) or off (eager). */
static bool run_script(pulsar_engine *e, const pulsar_tokens *story, bool graphs, int W, run_rec *r) {
    r->lg.assign((size_t)N_STEPS * ROWS * W, 0.0f);
    /* a fresh cache: a key carries the driver's address, which a new session may reuse */
    pulsar_gpu_seg_reset();
    const int was_on = pulsar_gpu_seg_set_enabled(graphs ? 1 : 0);
    pulsar_engine_set_bank_pool(ROWS);
    pulsar_session *s = NULL;
    bool ok = pulsar_session_create(&s, e, 4096) == 0 && s && gate_bank_pool_fits(s, (int)ROWS);
    if (ok && (!s->states || !s->states->verify_rows)) {
        fprintf(stderr, "graph-eager: %s's session has no verify capture (a drafter loads it) -- refusing\n",
                pulsar_engine_family_name(e));
        ok = false;
    }
    for (uint32_t b = 0; ok && b < ROWS; b++) ok = gate_bank_sync(s, b, story->v + bank_off[b], bank_pre[b], NULL,
                                                                  "graph-eager bank");
    int pos[ROWS];
    for (uint32_t b = 0; b < ROWS; b++) pos[b] = bank_pre[b];
    uint64_t rep0 = 0, cap0 = 0;
    pulsar_gpu_seg_counts(&rep0, &cap0);
    char err[256] = "";
    for (uint32_t t = 0; ok && t < N_STEPS; t++) {
        const bool verify = step_verifies(t);
        pulsar_multiseq_req rq[ROWS];
        int32_t tok[ROWS];
        for (uint32_t b = 0; b < ROWS; b++) {
            tok[b] = story->v[bank_off[b] + pos[b]];
            rq[b] = {b, TokPos{pos[b]}, tok[b]};
        }
        uint64_t rb = 0, cb = 0, ra = 0, ca = 0;
        pulsar_gpu_seg_counts(&rb, &cb);
        uint32_t nr = 0;
        err[0] = '\0';
        float *lg = r->lg.data() + (size_t)t * ROWS * W;
        const int rc = pulsar_session_decode_mixed(s, rq, ROWS, lg, (int)(ROWS * W), &nr,
                                                   verify ? PULSAR_MSEQ_HEAD_ALL_ROWS : 0u, err, sizeof(err));
        if (rc != 0 || nr != ROWS) {
            fprintf(stderr, "graph-eager: step %u (%s, rc %d, %u rows headed): %s\n", t, verify ? "verify" : "decode",
                    rc, nr, err);
            ok = false;
            break;
        }
        pulsar_gpu_seg_counts(&ra, &ca);
        r->replayed[t] = ra != rb;
        /* the verify's commit: each bank's one row kept, its position past it (a decode step moved it already) */
        for (uint32_t b = 0; ok && verify && b < ROWS; b++) {
            ok = pulsar_state_commit(s->states, b, &tok[b], 1u);
            if (ok) pulsar_state_set_pos(s->states, b, (uint32_t)pos[b] + 1u);
            else fprintf(stderr, "graph-eager: step %u: bank %u's verify commit refused\n", t, b);
        }
        for (uint32_t b = 0; b < ROWS; b++) pos[b]++;
    }
    uint64_t rep1 = 0, cap1 = 0;
    pulsar_gpu_seg_counts(&rep1, &cap1);
    r->replays = rep1 - rep0;
    r->captures = cap1 - cap0;
    if (s) pulsar_session_free(s);
    pulsar_gpu_seg_set_enabled(was_on);
    return ok;
}

/* b against a, per kind and headed row: steps differing (and the first), worst |d|; returns the (kind, row) pairs
 * with any difference */
static int compare(const char *what, const run_rec &a, const run_rec &b, int W, bool show_replays) {
    int bad = 0;
    for (int kind = 0; kind < 2; kind++) {
        uint32_t replayed = 0;
        for (uint32_t t = (uint32_t)kind; t < N_STEPS; t += 2) replayed += b.replayed[t];
        printf("  %-4s %-6s", what, kind ? "verify" : "decode");
        if (show_replays) printf(" graph replays %2u/%u |", replayed, STEPS_PER_KIND);
        for (uint32_t k = 0; k < ROWS; k++) {
            uint32_t diff = 0;
            int first = -1;
            double maxd = 0;
            for (uint32_t t = (uint32_t)kind; t < N_STEPS; t += 2) {
                const size_t at = ((size_t)t * ROWS + k) * W;
                const float *x = a.lg.data() + at, *y = b.lg.data() + at;
                if (gate_first_diff(x, y, W) < 0) continue;
                diff++;
                if (first < 0) first = (int)t;
                for (int i = 0; i < W; i++) maxd = fmax(maxd, fabs((double)x[i] - (double)y[i]));
            }
            printf(" row %u: %2u differ", k, diff);
            if (diff) printf(" (first step %d, max|d| %.2e)", first, maxd);
            printf(" |");
            bad += diff != 0;
        }
        printf("\n");
    }
    return bad;
}

int GATE_ENTRY(int argc, char **argv) {
    n_fail = 0;
    if (argc < 2) { fprintf(stderr, "usage: %s <model>\n", argv[0]); return 2; }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "graph-eager: %s did not open\n", argv[1]); return 2; }
    const pulsar_driver_ops *ops = e->family->driver;
    if (!ops || !ops->stage) {
        fprintf(stderr, "graph-eager: %s's steps are not on a staged driver lane (no pulsar_driver_ops::stage): they "
                        "never graph, so there is no replay to grade -- refusing\n", pulsar_engine_family_name(e));
        gate_engine_close(e);
        return 1;
    }
    const int W = pulsar_engine_logits_width(e);
    pulsar_tokens story;
    int need = 0;
    for (uint32_t b = 0; b < ROWS; b++)
        need = need > bank_off[b] + bank_pre[b] + (int)N_STEPS ? need : bank_off[b] + bank_pre[b] + (int)N_STEPS;
    if (!gate_load_story(e, &story, need)) { gate_engine_close(e); return 2; }
    printf("graph-eager: %s, %u banks, %u decode and %u verify steps of %u rows, alternating\n",
           pulsar_engine_family_name(e), ROWS, STEPS_PER_KIND, STEPS_PER_KIND, ROWS);
    run_rec run[3];
    const char *names[3] = {"A (eager)", "B (graphs)", "C (eager)"};
    bool ok = true;
    for (int i = 0; ok && i < 3; i++) {
        const double t0 = gate_now_s();
        ok = run_script(e, &story, i == 1, W, &run[i]);
        printf("run %-10s %s in %.1f s: %llu graph replays, %llu captures\n", names[i], ok ? "done" : "FAILED",
               gate_now_s() - t0, (unsigned long long)run[i].replays, (unsigned long long)run[i].captures);
    }
    CHECK(ok, "the three runs of the script ran");
    if (ok) {
        CHECK(run[0].replays == 0 && run[0].captures == 0 && run[2].replays == 0 && run[2].captures == 0,
              "the eager runs captured and replayed nothing (they ran the eager steps)");
        uint32_t rep[2] = {0, 0};
        for (uint32_t t = 0; t < N_STEPS; t++) rep[step_verifies(t) ? 1 : 0] += run[1].replayed[t];
        CHECK(rep[0] > 0 && rep[1] > 0, "the graph run replayed captured steps of both kinds (decode %u, verify %u of "
              "%u each)", rep[0], rep[1], STEPS_PER_KIND);
        printf("every headed row against run A, by kind:\n");
        const int nondet = compare("C/A", run[0], run[2], W, false);
        const int wrong = compare("B/A", run[0], run[1], W, true);
        CHECK(nondet == 0, "the eager script is deterministic (run C == run A at every step and row)");
        CHECK(wrong == 0, "the graph lane equals the eager lane at every step and row (%d (kind, row) pair(s) differ)",
              wrong);
    }
    pulsar_tokens_free(&story);
    gate_engine_close(e);
    printf(n_fail ? "GRAPH-EAGER GATE FAIL (%d)\n" : "GRAPH-EAGER GATE PASS\n", n_fail);
    return n_fail != 0;
}

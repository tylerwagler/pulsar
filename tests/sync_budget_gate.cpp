/* sync-budget-gate (L302 step 2): the host<->device synchronisations a speculative round pays, per phase, held to a
 * budget -- pulsar's counterpart of DeepInfer's forbid_cuda_sync (utils.py, tests/test_cuda_sync_guards.py), which
 * raises on any sync in a forbidden scope.  Here every blocking copy and every stream drain is counted at the one
 * place it happens (pulsar_gpu_sync_stats, src/cuda/pulsar_cuda_runtime.cu) and the gate runs the server's batched
 * round API -- assemble, ONE shared verify (decode_mixed, every row headed, nothing read back), round_end (the device
 * walk), ONE redraft pass, commit -- at 1 and 4 banks, greedy and seeded T0.7, and asserts each phase's syncs per
 * round against the budget below.  A change that adds a readback or a drain to the round's path fails here by name;
 * one that removes some lowers the measured numbers, and the budget is lowered with it.
 *
 *   ./tests/sync_budget_gate <model> [rounds] [--report]
 *
 * --report prints the table without asserting (what a budget is set from). */
#include "pulsar.h"
#include "pulsar_gpu.h"
#include "gate_entry.h"
#include "gate_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static int n_fail = 0;
#define CHECK(c, ...) GATE_CHECK(n_fail, c, __VA_ARGS__)

enum { PH_ASSEMBLE, PH_FORWARD, PH_ROUND_END, PH_REDRAFT, PH_COUNT };
static const char *const k_phase[PH_COUNT] = {"assemble", "forward", "round_end", "redraft+commit"};

/* The budget: per round, per phase, the most stream drains + blocking copies (both directions) a lane may pay at 1 to
 * 4 banks, greedy or sampled -- per drafter, as its lane pays them.  Measured 2026-10-20 (L302: the device sampler,
 * walk and block chains), the worst round of 32 at each of the four configurations plus ~25% (a drafter's depth is
 * priced by measured step costs, so its round shapes move a little run to run).  Lower them as the round loses syncs;
 * never raise one without a row saying why.
 *   DSpark (vexp)      worst 12 / 6 / 14 / 16   (assemble pays 3 a bank)
 *   MTP (Qwen, mini)   worst  2 / 4 /  5 / 42   (each chained draft reads its token back)
 *   block (DFlash2,    worst  2 / 4 /  1 /  5   (the sampled chains: one launch, one read)
 *          LiLiCorr) */
struct budget { int assemble, forward, round_end, redraft; };
static budget budget_of(pulsar_drafter_kind k) {
    switch (k) {
    case PULSAR_DRAFTER_DSPARK: return {12, 6, 18, 20};
    case PULSAR_DRAFTER_MTP: return {3, 4, 6, 52};
    case PULSAR_DRAFTER_BLOCK: return {3, 4, 1, 6};
    default: return {0, 0, 0, 0};
    }
}

static uint64_t syncs_of(const pulsar_gpu_sync_stats &a, const pulsar_gpu_sync_stats &b) {
    return (b.stream_syncs - a.stream_syncs) + (b.device_syncs - a.device_syncs) + (b.d2h - a.d2h) + (b.h2d - a.h2d);
}

static void run(pulsar_engine *e, int NB, float temp, int rounds_max, bool report) {
    pulsar_session *s = NULL;
    char err[256] = "";
    bool ok = pulsar_session_create(&s, e, 4096) == 0 && s;
    static const char *const prompts[4] = {
        "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n",
        "The quick brown fox jumps over the lazy dog. Once upon a time, in a village by the sea,",
        "Write a function that returns the n-th Fibonacci number in Python.\n",
        "List the planets of the solar system in order from the sun:\n1."};
    for (int b = 0; ok && b < NB; b++) {
        pulsar_tokens p;
        memset(&p, 0, sizeof p);
        pulsar_tokenize_text(e, prompts[b % 4], &p);
        ok = pulsar_session_bank_state_restore(s, (uint32_t)b) && pulsar_session_sync(s, &p, err, sizeof(err)) == 0;
        if (ok) pulsar_session_bank_state_save(s, (uint32_t)b);
        pulsar_tokens_free(&p);
    }
    CHECK(ok, "%d bank(s) prefilled: %s", NB, err);
    std::vector<pulsar_spec_round *> rd(NB);
    std::vector<uint64_t> rngs(NB);
    for (int b = 0; b < NB; b++) { rd[b] = pulsar_spec_round_new(); rngs[b] = 0x1302u + (uint64_t)b; }
    const uint32_t width = pulsar_engine_fused_heads_max(e);
    std::vector<pulsar_multiseq_req> reqs(width);
    static int acc[32][PULSAR_SPEC_LOGITS_ROWS + 1];
    uint64_t sum[PH_COUNT] = {}, worst[PH_COUNT] = {}, d2h_bytes = 0;
    int rounds = 0, rows_total = 0;
    std::vector<bool> done(NB, false);   /* a bank whose stream ended (its base token is EOS, or the context) */
    for (int tick = 0; ok && rounds < rounds_max && tick < 4 * rounds_max; tick++) {
        pulsar_spec_step st[32];
        uint32_t bank_of[32];
        int m = 0;
        for (int b = 0; b < NB; b++) {
            if (done[b]) continue;
            memset(&st[m], 0, sizeof st[m]);
            st[m].bank = (uint32_t)b;
            st[m].round = rd[b];
            st[m].temperature = temp;
            st[m].top_p = 1.0f;
            st[m].min_p = 0.05f;
            st[m].rng = &rngs[b];
            st[m].max_tokens = 1 << 20;
            st[m].k_alloc = -1;
            st[m].accepted_cap = (int)(PULSAR_SPEC_LOGITS_ROWS + 1);
            bank_of[m++] = (uint32_t)b;
        }
        pulsar_gpu_sync_stats c0, c1, c2, c3, c4;
        pulsar_gpu_sync_stats_get(&c0);
        uint32_t rows = 0;
        (void)pulsar_session_spec_assemble_batch(s, st, m, width, reqs.data(), &rows);
        pulsar_gpu_sync_stats_get(&c1);
        int live = 0;
        char why[200] = "";
        for (int i = 0; i < m; i++) {
            if (st[i].status == PULSAR_SPEC_STEP_OK) { st[live] = st[i]; bank_of[live++] = bank_of[i]; }
            else if (st[i].status == PULSAR_SPEC_STEP_EOS || st[i].status == PULSAR_SPEC_STEP_LENGTH) done[bank_of[i]] = true;
            else if (!why[0]) snprintf(why, sizeof why, "bank %u status %d: %.150s", st[i].bank, (int)st[i].status, st[i].err);
        }
        m = live;
        if (m == 0 && !why[0]) break;   /* every bank's stream ended: the rounds measured so far stand */
        if (m == 0 || rows == 0) {
            ok = false;
            snprintf(err, sizeof err, "no bank verified at tick %d (%.200s)", tick, why);
            break;
        }
        uint32_t gotr = 0;
        pulsar_session_spec_arm_capture(s, rows);
        ok = pulsar_session_decode_mixed(s, reqs.data(), rows, NULL, 0, &gotr, PULSAR_MSEQ_HEAD_ALL_ROWS, err,
                                         sizeof(err)) == 0 && gotr == rows;
        pulsar_session_spec_arm_capture(s, 0u);
        pulsar_gpu_sync_stats_get(&c2);
        if (!ok) break;
        for (int i = 0; i < m; i++) {
            st[i].accepted = acc[i];
            st[i].accepted_cap = (int)(PULSAR_SPEC_LOGITS_ROWS + 1);
        }
        ok = pulsar_session_spec_round_end_batch(s, st, m) == 0;
        pulsar_gpu_sync_stats_get(&c3);
        for (int i = 0; ok && i < m; i++) {
            ok = st[i].status == PULSAR_SPEC_STEP_OK;
            if (!ok) snprintf(err, sizeof err, "round_end bank %u: %s", st[i].bank, st[i].err);
        }
        if (!ok) break;
        pulsar_spec_round *rr[32];
        uint64_t *rp[32];
        for (int i = 0; i < m; i++) { rr[i] = rd[bank_of[i]]; rp[i] = &rngs[bank_of[i]]; }
        ok = pulsar_session_spec_redraft_batch(s, rr, bank_of, rp, m, err, sizeof(err)) == 0 &&
             pulsar_session_spec_redraft_commit_batch(s, st, m) == 0;
        pulsar_gpu_sync_stats_get(&c4);
        if (tick < 2) continue;   /* the first rounds warm the lane (graph captures, prime) */
        const uint64_t ph[PH_COUNT] = {syncs_of(c0, c1), syncs_of(c1, c2), syncs_of(c2, c3), syncs_of(c3, c4)};
        for (int p = 0; p < PH_COUNT; p++) {
            sum[p] += ph[p];
            if (ph[p] > worst[p]) worst[p] = ph[p];
        }
        d2h_bytes += c4.d2h_bytes - c0.d2h_bytes;
        rows_total += (int)rows;
        rounds++;
    }
    CHECK(ok && rounds > 0, "%d bank(s) T%.1f: %d rounds measured: %s", NB, (double)temp, rounds, err);
    if (rounds > 0) {
        printf("  %d bank(s) T%.1f, %d rounds, %.1f verify rows/round, %.0f B read back/round:", NB, (double)temp,
               rounds, (double)rows_total / rounds, (double)d2h_bytes / rounds);
        for (int p = 0; p < PH_COUNT; p++)
            printf("  %s %.2f (max %llu)", k_phase[p], (double)sum[p] / rounds, (unsigned long long)worst[p]);
        printf("\n");
        if (!report) {
            const budget b = budget_of(pulsar_engine_drafter(e));
            const int lim[PH_COUNT] = {b.assemble, b.forward, b.round_end, b.redraft};
            for (int p = 0; p < PH_COUNT; p++)
                CHECK((int)worst[p] <= lim[p], "%d bank(s) T%.1f %s: at most %llu syncs a round, budget %d", NB,
                      (double)temp, k_phase[p], (unsigned long long)worst[p], lim[p]);
        }
    }
    for (int b = 0; b < NB; b++) pulsar_spec_round_free(rd[b]);
    if (s) pulsar_session_free(s);
}

int GATE_ENTRY(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model> [rounds] [--report]\n", argv[0]); return 2; }
    const int rounds = argc > 2 && argv[2][0] != '-' ? atoi(argv[2]) : 48;
    bool report = false;
    for (int i = 2; i < argc; i++) report |= !strcmp(argv[i], "--report");
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    n_fail = 0;
    pulsar_engine_set_bank_pool(4u);   /* the widest run's banks, before the first session */
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    if (!pulsar_engine_has_spec_rounds(e)) { fprintf(stderr, "no drafter in this artifact\n"); gate_engine_close(e); return 1; }
    printf("sync-budget-gate: syncs a round by phase (stream drains + blocking copies)%s\n",
           report ? " -- report only" : "");
    for (int nb : {1, 4})
        for (float t : {0.0f, 0.7f}) run(e, nb, t, rounds, report);
    gate_engine_close(e);
    printf("sync-budget-gate: %s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}

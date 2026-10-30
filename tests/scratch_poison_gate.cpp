/* L307 GATE: poisoned disposable scratch changes no output.
 *
 * Flare's defect class (NVIDIA flare-local-inference, docs/glm-batched-speculation-defect.md: 95.8% wrong selections on
 * Spark): a captured graph -- or any step -- that reads disposable scratch it did not write this step computes the
 * right answer exactly while the previous step left the right bytes behind, which is what eager runs and a repeating
 * shape do.  Flare's remedy, here: poison the reusable storage after its last permitted use and alternate the graph
 * families between replays.  Armed (pulsar_gpu_scratch_poison_set), every step's entry fills the scratch it owns -- the
 * CUDA scratch slots (every family), the driver families' op arenas (Qwen, Nemotron-H), the block drafter's arena (its
 * absorb and its draft pass) -- with a cycling byte pattern (0xFF, 0x00, 0x7F) before it uploads anything.
 *
 * The script: the server's batched speculative lane (assemble, ONE shared verify with every row headed, round_end, ONE
 * redraft pass, commit) over NB banks, greedy, the drafter at a fixed depth with no stop rule (its chains then depend
 * only on the bytes), and every tick ALTERNATES which banks run (a cycle of bank subsets) and the shared verify's row
 * budget (a cycle of widths) -- so the verify step, the output head and the draft pass change shape (bank count, row
 * count, run structure) from one tick to the next, and every captured graph is replayed between replays of others.
 *   run A  unpoisoned;   run C  unpoisoned again (A's determinism: the comparison's premise);   run B  POISONED, last
 *   (so nothing it leaves in a reused allocation can reach an unpoisoned run).
 * Recorded per tick: the verify rows (bank, position, token: the drafts the drafter proposed), every headed row's logits
 * (FNV-1a over the bytes), and each bank's accepted tokens; at the end each bank's stream.
 * PASS: C == A and B == A in every record, byte for byte; the graphed lanes replayed in B (rule 5: the gate ran the
 * production lane) and B poisoned steps (A and C none).
 *
 * Mutation-tested (make scratch-poison-gate-mutant): a build whose Qwen step clears its MoE non-finite flag only on its
 * first step -- an unpoisoned run never sees the difference (the flag stays clear), a poisoned run must FAIL.
 *
 *   [PULSAR_GATE_DRAFTER=dflash2] ./tests/scratch_poison_gate MODEL [tokens]
 * Also a runner gate (tests/gates_runner.cpp `scratch-poison-gate`, on every hosted model with speculative rounds). */
#include "pulsar.h"
#include "pulsar_gpu.h"
#include "gate_entry.h"
#include "gate_util.h"
#include "pulsar_engine_internal.h"   /* the drafter's numbers (fixed for the script), the yield quench */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

namespace {

constexpr int NB = 4;
/* the banks a tick runs (bit b = bank b) and the verify's row budget, each a cycle of its own length so the pairs
 * keep changing */
constexpr uint32_t kMasks[] = {0xFu, 0x1u, 0x6u, 0xBu, 0x2u, 0xDu, 0x8u, 0x7u, 0x5u, 0xEu, 0x3u};
constexpr uint32_t kWidths[] = {0u /* the engine's widest */, 3u, 7u, 2u, 11u, 5u, 16u};

int n_fail = 0;
#define CHECK(c, ...) GATE_CHECK(n_fail, c, __VA_ARGS__)

struct tick_rec {
    uint32_t rows = 0;
    std::vector<int32_t> req;      /* (bank, pos, token) per verify row */
    uint64_t logits = 0;           /* FNV-1a over every headed row's logits */
    std::vector<int32_t> acc;      /* (bank, n, tokens...) per verifying bank */
};

struct run_rec {
    std::vector<tick_rec> ticks;
    std::vector<int32_t> out[NB];
    uint64_t replays = 0, captures = 0, poisoned = 0, poison_bytes = 0;
    bool ok = false;
    std::string err;
};

std::vector<pulsar_tokens> prompts(pulsar_engine *e) {
    std::vector<pulsar_tokens> P;
    pulsar_tokens story;
    if (!gate_load_story(e, &story, 1400)) return P;
    const int lo[NB] = {0, 300, 700, 1040}, hi[NB] = {96, 460, 1033, 1390};
    for (int b = 0; b < NB; b++) {
        pulsar_tokens t;
        memset(&t, 0, sizeof t);
        for (int i = lo[b]; i < hi[b] && i < story.len; i++) pulsar_tokens_push(&t, story.v[i]);
        P.push_back(t);
    }
    pulsar_tokens_free(&story);
    return P;
}

/* the yield quench disarmed (it latches on a drafter that loses and drafts nothing for the next round) */
void unquench(pulsar_session *s) {
    s->spec.spec_quenched = false;
    s->spec.spec_quench_steps = 0;
    s->spec.spec_quench_debt = 0.0f;
    s->spec.spec_quench_ewma = 0.0f;
}

run_rec run(pulsar_engine *e, const std::vector<pulsar_tokens> &P, int n, bool poison) {
    run_rec R;
    const int W = pulsar_engine_logits_width(e);
    const uint32_t widest = pulsar_engine_fused_heads_max(e);
    pulsar_engine_set_bank_pool(NB);
    pulsar_session *s = NULL;
    char err[256] = "";
    /* poisoned from the banks' prefill on: every step of the run */
    const int was = pulsar_gpu_scratch_poison_set(poison ? 1 : 0);
    uint64_t r0 = 0, c0 = 0, p0 = 0, pb0 = 0;
    pulsar_gpu_seg_counts(&r0, &c0);
    pulsar_gpu_scratch_poison_counts(&p0, &pb0);
    bool ok = pulsar_session_create(&s, e, 4096) == 0 && s;
    for (int b = 0; ok && b < NB; b++) {
        ok = pulsar_session_bank_state_restore(s, (uint32_t)b) && pulsar_session_sync(s, &P[b], err, sizeof err) == 0;
        if (ok) pulsar_session_bank_state_save(s, (uint32_t)b);
    }
    if (!ok) snprintf(err + strlen(err), sizeof err - strlen(err), " (the banks' prefill)");
    pulsar_spec_round *rd[NB];
    uint64_t rngs[NB] = {};
    bool done[NB] = {};
    for (int b = 0; b < NB; b++) rd[b] = pulsar_spec_round_new();
    std::vector<pulsar_multiseq_req> reqs(widest);
    static int acc[NB][PULSAR_SPEC_LOGITS_ROWS + 1];
    for (uint32_t t = 0; ok && t < (uint32_t)(8 * n); t++) {
        bool any = false;
        for (int b = 0; b < NB; b++) any |= !done[b];
        if (!any) break;
        uint32_t mask = kMasks[t % (sizeof kMasks / sizeof kMasks[0])];
        bool hit = false;
        for (int b = 0; b < NB; b++) hit |= (mask >> b & 1u) && !done[b];
        if (!hit) mask = 0xFu;   /* the cycle's banks are all done: whichever are left */
        uint32_t width = kWidths[t % (sizeof kWidths / sizeof kWidths[0])];
        if (width == 0u || width > widest) width = widest;
        pulsar_spec_step st[NB];
        uint32_t bank_of[NB];
        int m = 0;
        for (int b = 0; ok && b < NB; b++) {
            if (done[b] || !(mask >> b & 1u)) continue;
            ok = pulsar_session_bank_state_restore(s, (uint32_t)b);
            unquench(s);
            pulsar_session_bank_state_save(s, (uint32_t)b);
            memset(&st[m], 0, sizeof st[m]);
            st[m].bank = (uint32_t)b;
            st[m].round = rd[b];
            st[m].temperature = 0.0f;
            st[m].top_p = 1.0f;
            st[m].rng = &rngs[b];
            st[m].max_tokens = n - (int)R.out[b].size();
            st[m].k_alloc = -1;
            st[m].accepted_cap = (int)(PULSAR_SPEC_LOGITS_ROWS + 1);
            bank_of[m++] = (uint32_t)b;
        }
        if (!ok) {
            snprintf(err, sizeof err, "tick %u: a bank would not restore", t);
            break;
        }
        uint32_t rows = 0;
        ok = pulsar_session_spec_assemble_batch(s, st, m, width, reqs.data(), &rows) == 0;
        tick_rec T;
        int live = 0;
        for (int i = 0; ok && i < m; i++) {
            if (st[i].status == PULSAR_SPEC_STEP_EOS) {
                R.out[bank_of[i]].push_back(st[i].first_token);
                done[bank_of[i]] = true;
            } else if (st[i].status == PULSAR_SPEC_STEP_OK) {
                st[live] = st[i];
                bank_of[live++] = bank_of[i];
            } else if (st[i].status != PULSAR_SPEC_STEP_SKIPPED) {
                ok = false;
                snprintf(err, sizeof err, "tick %u assemble bank %u: status %d %s", t, st[i].bank, st[i].status,
                         st[i].err);
            }
        }
        if (!ok) break;
        m = live;
        T.rows = rows;
        for (uint32_t k = 0; k < rows; k++) {
            T.req.push_back((int32_t)reqs[k].bank);
            T.req.push_back(reqs[k].pos.raw());
            T.req.push_back(reqs[k].token);
        }
        if (m == 0 || rows == 0) {
            R.ticks.push_back(T);
            continue;
        }
        std::vector<float> logits((size_t)rows * W);
        uint32_t got = 0;
        pulsar_session_spec_arm_capture(s, rows);
        ok = pulsar_session_decode_mixed(s, reqs.data(), rows, logits.data(), (int)rows * W, &got,
                                         PULSAR_MSEQ_HEAD_ALL_ROWS, err, sizeof err) == 0 && got == rows;
        pulsar_session_spec_arm_capture(s, 0u);
        if (!ok) {
            snprintf(err + strlen(err), sizeof err - strlen(err), " (tick %u: the %u-row verify)", t, rows);
            break;
        }
        T.logits = gate_fnv1a(logits.data(), logits.size() * sizeof(float));
        for (int i = 0; i < m; i++) {
            st[i].accepted = acc[i];
            st[i].accepted_cap = (int)(PULSAR_SPEC_LOGITS_ROWS + 1);
            st[i].temperature = 0.0f;
            st[i].greedy_rows = 0u;
        }
        ok = pulsar_session_spec_round_end_batch(s, st, m) == 0;
        for (int i = 0; ok && i < m; i++) {
            ok = st[i].status == PULSAR_SPEC_STEP_OK;
            if (!ok) {
                snprintf(err, sizeof err, "tick %u round_end bank %u: %s", t, st[i].bank, st[i].err);
                break;
            }
            const uint32_t b = bank_of[i];
            T.acc.push_back((int32_t)b);
            T.acc.push_back(st[i].n_accepted);
            for (int k = 0; k < st[i].n_accepted; k++) {
                T.acc.push_back(acc[i][k]);
                R.out[b].push_back(acc[i][k]);
            }
            if ((int)R.out[b].size() >= n ||
                (st[i].n_accepted && pulsar_token_is_stop(e, acc[i][st[i].n_accepted - 1])))
                done[b] = true;
        }
        R.ticks.push_back(T);
        if (!ok) break;
        pulsar_spec_round *rr[NB];
        uint64_t *rp[NB];
        for (int i = 0; i < m; i++) {
            rr[i] = rd[bank_of[i]];
            rp[i] = &rngs[bank_of[i]];
        }
        ok = pulsar_session_spec_redraft_batch(s, rr, bank_of, rp, m, err, sizeof err) == 0 &&
             pulsar_session_spec_redraft_commit_batch(s, st, m) == 0;
        if (!ok) snprintf(err + strlen(err), sizeof err - strlen(err), " (tick %u: the redraft)", t);
    }
    uint64_t r1 = 0, c1 = 0, p1 = 0, pb1 = 0;
    pulsar_gpu_seg_counts(&r1, &c1);
    pulsar_gpu_scratch_poison_counts(&p1, &pb1);
    (void)pulsar_gpu_scratch_poison_set(was);
    R.replays = r1 - r0;
    R.captures = c1 - c0;
    R.poisoned = p1 - p0;
    R.poison_bytes = pb1 - pb0;
    for (int b = 0; b < NB; b++) pulsar_spec_round_free(rd[b]);
    if (s) pulsar_session_free(s);
    R.ok = ok;
    R.err = err;
    return R;
}

/* b against a: the first tick whose rows, drafts, logits or accepted tokens differ, and each bank's stream */
int compare(const char *what, const run_rec &a, const run_rec &b) {
    int bad = 0;
    const size_t n = a.ticks.size() < b.ticks.size() ? a.ticks.size() : b.ticks.size();
    size_t first = n;
    const char *field = "";
    for (size_t t = 0; t < n && first == n; t++) {
        const tick_rec &x = a.ticks[t], &y = b.ticks[t];
        if (x.rows != y.rows) field = "the verify's row count";
        else if (x.req != y.req) field = "the verify rows (the drafts)";
        else if (x.logits != y.logits) field = "the headed logits";
        else if (x.acc != y.acc) field = "the accepted tokens";
        else continue;
        first = t;
    }
    if (first == n && a.ticks.size() != b.ticks.size()) field = "the tick count";
    const bool same = first == n && a.ticks.size() == b.ticks.size();
    printf("  %s: %zu vs %zu ticks, %s", what, a.ticks.size(), b.ticks.size(), same ? "every record identical" : "");
    if (!same) printf("first difference at tick %zu: %s", first, field);
    printf("\n");
    bad += !same;
    for (int k = 0; k < NB; k++) {
        if (a.out[k] == b.out[k]) continue;
        size_t d = 0;
        while (d < a.out[k].size() && d < b.out[k].size() && a.out[k][d] == b.out[k][d]) d++;
        printf("  %s: bank %d's stream differs at token %zu (%zu vs %zu tokens)\n", what, k, d, a.out[k].size(),
               b.out[k].size());
        bad++;
    }
    return bad;
}

}  // namespace

int GATE_ENTRY(int argc, char **argv) {
    n_fail = 0;
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL [tokens]\n", argv[0]);
        return 2;
    }
    const int n = argc > 2 ? atoi(argv[2]) : 96;
    pulsar_engine_options o;
    memset(&o, 0, sizeof o);
    o.model_path = argv[1];
    o.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &o) != 0) {
        fprintf(stderr, "engine open failed\n");
        return 1;
    }
    if (!pulsar_engine_has_spec_rounds(e) || !e->drafter_ops || pulsar_engine_fused_heads_max(e) < 2u) {
        printf("SCRATCH-POISON GATE: FAIL (the model has no batched speculative lane to run)\n");
        gate_engine_close(e);
        return 1;
    }
    std::vector<pulsar_tokens> P = prompts(e);
    if ((int)P.size() != NB) {
        gate_engine_close(e);
        return 1;
    }
    /* the drafter at a fixed depth, no stop rule, no adaptive controller and no verify pick: its chains depend on the
     * bytes alone (the engine's own numbers put back at the end -- a runner gate borrows the engine) */
    const pulsar_drafter_ops *base = e->drafter_ops;
    const float tau0 = e->spec_tau;
    const int depth0 = e->spec_depth;
    struct pulsar_verify_curve *curve0 = e->verify_curve;
    pulsar_drafter_ops fixed = *base;
    fixed.depth = base->depth_max < 4u ? base->depth_max : 4u;
    fixed.adapt = NULL;
    e->drafter_ops = &fixed;
    e->spec_tau = -1.0f;
    e->spec_depth = 0;
    e->verify_curve = NULL;
    printf("scratch-poison gate: %s, drafter %s at depth %u, %d banks, %d tokens a stream; the tick cycles %zu bank "
           "subsets x %zu verify widths (widest %u)\n",
           argv[1], base->name, fixed.depth, NB, n, sizeof kMasks / sizeof kMasks[0],
           sizeof kWidths / sizeof kWidths[0], pulsar_engine_fused_heads_max(e));
    run_rec A, B, C;
    const char *names[3] = {"A (unpoisoned)", "C (unpoisoned)", "B (POISONED)"};
    run_rec *runs[3] = {&A, &C, &B};
    for (int i = 0; i < 3; i++) {
        const double t0 = gate_now_s();
        *runs[i] = run(e, P, n, i == 2);
        const run_rec &r = *runs[i];
        size_t toks = 0;
        for (int b = 0; b < NB; b++) toks += r.out[b].size();
        printf("run %-15s %s in %.1f s: %zu ticks, %zu tokens, %llu graph replays, %llu captures, %llu steps "
               "poisoned (%.1f MiB filled)%s%s\n",
               names[i], r.ok ? "done" : "FAILED", gate_now_s() - t0, r.ticks.size(), toks,
               (unsigned long long)r.replays, (unsigned long long)r.captures, (unsigned long long)r.poisoned,
               (double)r.poison_bytes / 1048576.0, r.err.empty() ? "" : ": ", r.err.c_str());
    }
    e->drafter_ops = base;
    e->spec_tau = tau0;
    e->spec_depth = depth0;
    e->verify_curve = curve0;
    CHECK(A.ok && C.ok, "the unpoisoned runs completed (%s%s%s)", A.err.c_str(), A.err.empty() ? "" : "; ",
          C.err.c_str());
    CHECK(B.ok, "the poisoned run completed: %s", B.ok ? "every step ran" : B.err.c_str());
    CHECK(A.poisoned == 0 && C.poisoned == 0 && B.poisoned > 0,
          "only run B poisoned its steps (A %llu, B %llu, C %llu)", (unsigned long long)A.poisoned,
          (unsigned long long)B.poisoned, (unsigned long long)C.poisoned);
    CHECK(B.replays > 0, "run B replayed captured graphs between the poisonings (%llu replays)",
          (unsigned long long)B.replays);
    printf("every record against run A:\n");
    const int nondet = compare("C/A", A, C);
    const int wrong = compare("B/A", A, B);
    CHECK(nondet == 0, "the unpoisoned runs agree (run C == run A: the comparison's premise)");
    CHECK(wrong == 0, "the poisoned run equals the unpoisoned run in every record (%d difference(s))", wrong);
    for (pulsar_tokens &t : P) pulsar_tokens_free(&t);
    gate_engine_close(e);
    printf("SCRATCH-POISON GATE: %s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}

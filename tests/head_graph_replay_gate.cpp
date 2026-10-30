/* L292 GATE: a replayed output-head graph computes what the eager head computes, and leaves the next step what
 * the eager head leaves it.
 *
 * The batched output head is the one captured CUDA graph on the DeepSeek path (gpu_graph_encode_output_head_batch,
 * the L119 segment cache): a key's 12th use is captured and every later use replays it.  Two ways it went wrong:
 *   - the KEY: the head's dense GEMMs choose their arm by ROW KIND (L167, pulsar_gpu_matmul_decode_prefix), so a
 *     pure decode step and a fused step whose head has the same row count compute different arithmetic; a key
 *     without the kinds replays the first captured kind for both (the ARM row below);
 *   - the HOST STATE: an eagerly-encoded head left its flat_hc bf16 plane keyed and valid, and the next fused step's
 *     decode-prefix GEMM over batch_flat_hc took that tighter window -- the previous head's rows -- for its own;
 *     a replayed head acquires no slot, so the next step's bytes depended on whether this head replayed (rows 0/1
 *     of the fused phases below).
 *
 * One session, 4 banks: banks 0, 1, 3 decode, bank 2 takes prompt chunks.  The same step script runs three times:
 *   W1  PHASE_STEPS pure decode steps, 3 rows (banks 0, 1, 3): head 3 rows, all decode;
 *   F1  PHASE_STEPS fused steps: 2 decode rows (banks 0, 1) + a PF_ROWS prompt chunk on bank 2 with its last row
 *       headed: head 3 rows, decode prefix 2;
 *   --  the segment cache is reset (pulsar_gpu_seg_reset);
 *   F2  PHASE_STEPS fused steps, as F1;
 *   W2  PHASE_STEPS pure decode steps, as W1.
 * W1 then F1 hands the fused head whatever W1 captured; F2 then W2 hands the decode head whatever F2 captured.
 * Every token is teacher-forced, so the logits never feed the KV and every step's inputs are the same in every run.
 *   run A  EAGER: the cache is reset before every step, so no key reaches its capture threshold (asserted: no
 *          replay, no capture);
 *   run B  GRAPHS: the production lane (asserted: graphs replayed);
 *   run C  EAGER again: run A's determinism (the comparison's premise).
 * PASS: every headed row of every step is byte-identical between B and A, and between C and A.
 *   usage: PULSAR_MSEQ_BANKS=4 ./tests/head_graph_replay_gate MODEL
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "gate_entry.h"
#include "gate_fixture.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PHASE_STEPS 30u          /* > 2 x the capture threshold (12): both carrier parities capture, then replay */
#define PF_ROWS     32u          /* bank 2's prompt chunk per fused step */
#define N_PHASE     4u
#define N_STEPS     (N_PHASE * PHASE_STEPS)
#define HEAD_ROWS   3u           /* every step heads 3 rows: the two shapes share a row count */
#define ARM_ROW     2u           /* the row whose kind differs: bank 3's decode row / bank 2's prompt row */

static const char *const phase_name[N_PHASE] = {
    "W1 decode (cold cache)", "F1 fused after decode", "F2 fused (cache reset)", "W2 decode after fused"};
static bool phase_fused(uint32_t ph) { return ph == 1u || ph == 2u; }

/* where each bank's tokens come from in the story, and its classic prefix */
static const int bank_off[4] = {0, 2400, 3100, 2800};
static const int bank_pre[4] = {300, 170, 100, 230};

static pulsar_engine *g_e;
static pulsar_tokens g_toks;

/* one run's record: every headed row's logits and top-1, and which steps replayed a graph */
struct run_rec {
    float *lg;                  /* N_STEPS x HEAD_ROWS x V */
    int top[N_STEPS][HEAD_ROWS];
    uint8_t replayed[N_STEPS];
    uint64_t replays, captures;
};
static const float *row_of(const run_rec *r, uint32_t t, uint32_t k) {
    return r->lg + ((uint64_t)t * HEAD_ROWS + k) * PULSAR_N_VOCAB;
}

static bool run_script(bool eager, run_rec *r) {
    const uint64_t V = PULSAR_N_VOCAB;
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) { fprintf(stderr, "session create failed\n"); return false; }
    bool ok = gate_pool_fits(s, 4);
    for (uint32_t b = 0; ok && b < 4; b++)
        ok = gate_populate_bank(s, b, g_toks.v + bank_off[b], bank_pre[b], NULL, "head-graph bank");
    int pos[4] = {bank_pre[0], bank_pre[1], bank_pre[2], bank_pre[3]};
    pulsar_multiseq_req rq[2 + PF_ROWS];
    char err[256];
    uint64_t rep0 = 0, cap0 = 0;
    pulsar_gpu_seg_counts(&rep0, &cap0);
    for (uint32_t t = 0; ok && t < N_STEPS; t++) {
        const uint32_t ph = t / PHASE_STEPS;
        if (eager || t == 2u * PHASE_STEPS) pulsar_gpu_seg_reset();
        uint64_t rb = 0, cb = 0;
        pulsar_gpu_seg_counts(&rb, &cb);
        uint32_t nr = 0, n = 0;
        const uint32_t dec_banks[3] = {0, 1, 3};
        const uint32_t n_dec = phase_fused(ph) ? 2u : 3u;
        for (uint32_t j = 0; j < n_dec; j++) {
            const uint32_t b = dec_banks[j];
            rq[n].bank = b;
            rq[n].pos = TokPos{pos[b]};
            rq[n].token = g_toks.v[bank_off[b] + pos[b]];
            pos[b]++;
            n++;
        }
        float *lg = r->lg + (uint64_t)t * HEAD_ROWS * V;
        int rc;
        if (phase_fused(ph)) {
            for (uint32_t j = 0; j < PF_ROWS; j++, n++) {
                rq[n].bank = 2;
                rq[n].pos = TokPos{pos[2]};
                rq[n].token = g_toks.v[bank_off[2] + pos[2]];
                pos[2]++;
            }
            pulsar_fused_shape sh;
            memset(&sh, 0, sizeof sh);
            sh.n_dec = n_dec;
            sh.n_pf = 1;
            sh.head_last[0] = 1;
            rc = pulsar_session_decode_fused(s, rq, n, &sh, lg, (int)(HEAD_ROWS * V), &nr, err, sizeof err);
        } else {
            rc = pulsar_session_decode_mixed(s, rq, n, lg, (int)(HEAD_ROWS * V), &nr, 0u, err, sizeof err);
        }
        if (rc != 0) { fprintf(stderr, "step %u (%s): %s\n", t, phase_name[ph], err); ok = false; break; }
        if (nr != HEAD_ROWS) { fprintf(stderr, "step %u headed %u rows, want %u\n", t, nr, HEAD_ROWS); ok = false; break; }
        uint64_t ra = 0, ca = 0;
        pulsar_gpu_seg_counts(&ra, &ca);
        r->replayed[t] = ra != rb;
        for (uint32_t k = 0; k < HEAD_ROWS; k++) r->top[t][k] = (int)argmax_f32(lg + k * V, V);
    }
    uint64_t rep1 = 0, cap1 = 0;
    pulsar_gpu_seg_counts(&rep1, &cap1);
    r->replays = rep1 - rep0;
    r->captures = cap1 - cap0;
    pulsar_session_free(s);
    return ok;
}

/* b against a, per phase and headed row: steps differing (and the first), worst |d|, top-1 changes */
static int compare(const char *what, const run_rec *a, const run_rec *b, bool show_replays) {
    const uint64_t V = PULSAR_N_VOCAB;
    int bad = 0;
    for (uint32_t ph = 0; ph < N_PHASE; ph++) {
        uint32_t replayed = 0;
        for (uint32_t t = ph * PHASE_STEPS; t < (ph + 1u) * PHASE_STEPS; t++) replayed += b->replayed[t];
        printf("  %-4s %-24s", what, phase_name[ph]);
        if (show_replays) printf(" graph replays %2u/%u |", replayed, PHASE_STEPS);
        for (uint32_t k = 0; k < HEAD_ROWS; k++) {
            uint32_t diff = 0, flips = 0;
            int first = -1;
            double maxd = 0;
            for (uint32_t t = ph * PHASE_STEPS; t < (ph + 1u) * PHASE_STEPS; t++) {
                const float *x = row_of(a, t, k), *y = row_of(b, t, k);
                if (gate_first_diff(x, y, (long)V) < 0) continue;
                diff++;
                if (first < 0) first = (int)(t - ph * PHASE_STEPS);
                for (uint64_t i = 0; i < V; i++) maxd = fmax(maxd, fabs((double)x[i] - (double)y[i]));
                flips += a->top[t][k] != b->top[t][k];
            }
            printf(" row %u: %2u differ", k, diff);
            if (diff) printf(" (first step %d, max|d| %.2e, top-1 %u)", first, maxd, flips);
            printf(" |");
            bad += diff != 0;
        }
        printf("\n");
    }
    return bad;
}

int GATE_ENTRY(int argc, char **argv) {
    g_e = NULL;
    memset(&g_toks, 0, sizeof g_toks);
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }
    pulsar_engine_options o;
    memset(&o, 0, sizeof o);
    o.model_path = argv[1];
    o.backend = PULSAR_BACKEND_CUDA;
    if (gate_engine_open(&g_e, &o) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    int need = 0;
    for (uint32_t b = 0; b < 4; b++) {
        const int steps = b == 2 ? (int)(2u * PHASE_STEPS * PF_ROWS) : b == 3 ? (int)(2u * PHASE_STEPS) : (int)N_STEPS;
        need = need > bank_off[b] + bank_pre[b] + steps ? need : bank_off[b] + bank_pre[b] + steps;
    }
    if (!gate_load_story(g_e, &g_toks, need)) { gate_engine_close(g_e); return 1; }
    const uint64_t V = PULSAR_N_VOCAB;
    run_rec *run[3];
    for (int i = 0; i < 3; i++) {
        run[i] = (run_rec *)calloc(1, sizeof(run_rec));
        run[i]->lg = (float *)malloc((uint64_t)N_STEPS * HEAD_ROWS * V * sizeof(float));
    }
    bool ok = true;
    const char *names[3] = {"A (eager)", "B (graphs)", "C (eager)"};
    for (int i = 0; ok && i < 3; i++) {
        const double t0 = gate_now_s();
        ok = run_script(i != 1, run[i]);
        printf("run %-10s %s in %.1f s: %llu graph replays, %llu captures\n", names[i], ok ? "done" : "FAILED",
               gate_now_s() - t0, (unsigned long long)run[i]->replays, (unsigned long long)run[i]->captures);
    }
    int fails = 0;
    if (!ok) fails++;
    else {
        GATE_CHECK(fails, run[0]->replays == 0 && run[0]->captures == 0 && run[2]->replays == 0 && run[2]->captures == 0,
                   "the eager runs captured and replayed nothing (they ran the eager head)");
        GATE_CHECK(fails, run[1]->replays > 0, "the graph run replayed captured heads (%llu)",
                   (unsigned long long)run[1]->replays);
        printf("F1 step 0 fingerprints (cross-binary evidence): ");
        for (int i = 0; i < 3; i++)
            for (uint32_t k = 0; k < HEAD_ROWS; k++)
                printf("%c%u=%016llx ", 'A' + i, k,
                       (unsigned long long)gate_fnv1a(row_of(run[i], PHASE_STEPS, k), V * sizeof(float)));
        printf("\nevery headed row against run A, by phase:\n");
        const int nondet = compare("C/A", run[0], run[2], false);
        const int wrong = compare("B/A", run[0], run[1], true);
        GATE_CHECK(fails, nondet == 0, "the eager script is deterministic (run C == run A in every phase)");
        GATE_CHECK(fails, wrong == 0, "the graph lane equals the eager lane in every phase and row (%d (phase, row) "
                                      "pair(s) differ)", wrong);
    }
    for (int i = 0; i < 3; i++) { free(run[i]->lg); free(run[i]); }
    pulsar_tokens_free(&g_toks);
    gate_engine_close(g_e);
    printf("HEAD-GRAPH REPLAY GATE: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

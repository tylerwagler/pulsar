/* plan-34 phase-2 increment 2 — cuBLASLt ALGO-STABILITY gate.
 *
 * THE INVARIANT: a decode bank's step logits must be INDEPENDENT of the total
 * row count M of the batched step it rides in. If they are not, then co-scheduling
 * a big prefill chunk (inc 4, M -> hundreds) would silently perturb a co-scheduled
 * decode bank — risk #1 in plan-34 (the session.c cuBLASLt M-dependence class:
 * the MXFP8 GEMM switches from an M-independent custom kernel at M<=4 to a
 * cuBLASLt HEURISTIC algo at M>=5, and the Lt shape cache is keyed on ntok, so a
 * different M can resolve a different algo -> different reduction -> different bytes).
 *
 * METHOD: bank 0 is the fixed TARGET. Populate banks 0..M-1 from distinct prompts
 * (per-bank isolation means bank 0's output depends only on bank 0's KV), then run
 * ONE batched step through pulsar_session_decode_mixed at width M and capture bank 0's
 * logit row. Repeat for M in {1,2,4,5,8,12,16} on a FRESH session each time (no
 * idempotency/poisoning artifacts). Bank 0's prompt/pos/token are identical every
 * time; only M (and which OTHER banks pad the batch) changes.
 *
 * ASSERTIONS:
 *  - HARD: bank 0's logits are byte-identical across the BATCHED tier M in
 *    {2,4,5,8,12,16} — i.e. adding rows, INCLUDING crossing the M=4->5
 *    custom->cuBLASLt boundary and the 8->9 MoE GEMV cap (L152), does not
 *    perturb the target. This is the property inc 4 relies on.
 *  - REPORTED: M=1 vs M=2 (the single-row kernel tier the inc-1 multiseq gate
 *    excludes by construction) — informational, not fatal by itself.
 * The first differing float index + the two values are printed so a real algo
 * divergence (large, systematic) is never hidden behind a pass/fail bit.
 *
 * The comp caches have one format each (packed attn / MXFP4 indexer).
 * MODEL-DEPENDENT, needs PULSAR_MSEQ_BANKS>=16. Run under GPU discipline.
 *   usage: PULSAR_MSEQ_BANKS=16 ./tests/algo_stability_gate MODEL
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "gate_fixture.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 16 = PULSAR_GPU_MNEUTRAL_ROWS_MAX: the whole armed M-neutral range.  The
 * sweep stopped at 8 until 2026-09-02 (L152/L153) -- the same blind spot every
 * other gate had, and where the 8-vs-9-row MoE cap defect lived. */
#define GATE_MAX_N 16
static pulsar_engine *g_e;
static pulsar_tokens g_toks;
static int g_fail;

static const int g_prompt_off[GATE_MAX_N] = {0, 401, 907, 233, 601, 811, 101, 499,
                                             1301, 1499, 1201, 1601, 1109, 1705, 1401, 1000};
static const int g_prompt_len[GATE_MAX_N] = {130, 258, 511, 187, 342, 419, 275, 158,
                                             177, 233, 140, 201, 169, 150, 120, 205};

/* Run ONE batched decode step at width M on a fresh session; copy bank 0's logit
 * row (PULSAR_N_VOCAB floats) into row0_out. Returns false on any engine failure. */
static bool bank0_logits_at_width(int M, float *row0_out) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) return false;
    const int vocab = (int)PULSAR_N_VOCAB;
    bool ok = gate_pool_fits(s, (uint32_t)M);
    char err[256];
    int argtok[GATE_MAX_N];
    for (int k = 0; ok && k < M; k++)
        ok = gate_populate_bank(s, (uint32_t)k, g_toks.v + g_prompt_off[k], g_prompt_len[k],
                                &argtok[k], "populate");
    float *logits = ok ? (float *)malloc((size_t)M * vocab * sizeof(float)) : NULL;
    if (ok && !logits) ok = false;
    if (ok) {
        pulsar_multiseq_req reqs[GATE_MAX_N];
        for (int k = 0; k < M; k++) {
            reqs[k].bank = (uint32_t)k;
            reqs[k].pos = g_prompt_len[k];      /* bank 0 ALWAYS at the same pos */
            reqs[k].token = argtok[k];
        }
        const int rc = pulsar_session_decode_mixed(s, reqs, (uint32_t)M, logits,
                                                M * vocab, NULL, 0u, err, sizeof err);
        if (rc != 0) { fprintf(stderr, "decode_mixed(M=%d) failed rc=%d: %s\n", M, rc, err); ok = false; }
        else memcpy(row0_out, logits, (size_t)vocab * sizeof(float));  /* row 0 == bank 0 */
    }
    free(logits);
    pulsar_session_free(s);
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }
    pulsar_engine_options opt; memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1]; opt.backend = PULSAR_BACKEND_CUDA;
    if (pulsar_engine_open(&g_e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    printf("CONFIG: packed attn comp cache + MXFP4 indexer cache (the only formats)\n");

    int need = 0;
    for (int k = 0; k < GATE_MAX_N; k++)
        if (g_prompt_off[k] + g_prompt_len[k] > need) need = g_prompt_off[k] + g_prompt_len[k];
    if (!gate_load_story(g_e, &g_toks, need)) return 1;

    const int vocab = (int)PULSAR_N_VOCAB;
    /* 12 and 16 added 2026-09-02: the armed range above 8 rows had no width
     * sweep in the battery (L152 lived at 9..16). */
    const int widths[] = {1, 2, 4, 5, 8, 12, 16};
    const int nW = (int)(sizeof(widths) / sizeof(widths[0]));
    float *row[GATE_MAX_N]; memset(row, 0, sizeof row);
    for (int wi = 0; wi < nW; wi++) {
        row[wi] = (float *)malloc((size_t)vocab * sizeof(float));
        if (!bank0_logits_at_width(widths[wi], row[wi])) {
            fprintf(stderr, "ALGO-STABILITY GATE FAIL: width %d run failed\n", widths[wi]);
            g_fail = 1; goto done;
        }
    }

    /* Reference = M=2 (the smallest BATCHED-tier width; M=1 is the single-row
     * kernel tier). HARD: every batched-tier width byte-identical to it. */
    int ref;   /* widths[1] == 2 */
    ref = 1;
    for (int wi = 0; wi < nW; wi++) {
        const long d = gate_first_diff(row[wi], row[ref], vocab);
        const bool batched = widths[wi] >= 2;
        if (d < 0) {
            printf("ALGO-STABILITY: bank0 logits M=%d == M=2 (byte-identical)\n", widths[wi]);
        } else if (batched) {
            fprintf(stderr, "ALGO-STABILITY GATE FAIL: bank0 logits M=%d DIFFER from "
                    "M=2 at float %ld (%.9g vs %.9g) — adding rows perturbed a "
                    "co-scheduled decode bank (cuBLASLt M-dependence)\n",
                    widths[wi], d, row[wi][d], row[ref][d]);
            g_fail = 1;
        } else {
            printf("ALGO-STABILITY (informational): bank0 logits M=%d differ from M=2 "
                   "at float %ld (%.9g vs %.9g) — the single-row kernel tier (the "
                   "inc-1 multiseq gate excludes N=1 for the same reason)\n",
                   widths[wi], d, row[wi][d], row[ref][d]);
        }
    }

done:
    for (int wi = 0; wi < nW; wi++) free(row[wi]);
    pulsar_engine_close(g_e);
    if (g_fail) { fprintf(stderr, "ALGO-STABILITY GATE: FAIL\n"); return 1; }
    printf("ALGO-STABILITY GATE: PASS\n");
    return 0;
}

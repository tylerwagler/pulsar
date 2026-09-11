/* L168/L218 GATE: the ratio-2 compressor state after a whole-prompt prefill is
 * the state the decode store builds -- for prompts whose length is NOT a
 * multiple of the ratio.
 *
 * CSA2 (DeepSeek-V4.1): the kv sources 2/8/14 pool two tokens into one
 * compressed row.  Their state is 2 rows, slot pos % 2 holding the pending
 * token's kv / score projection; a complete group is consumed at its emit and
 * the slots are cleared, so at any EVEN position the state is canonically
 * empty (kv 0, score -inf) and at an odd position slot 0 holds the group's
 * first token and slot 1 is empty.  The boundary-aligned prefill arm builds
 * that state from the chunk's trailing partial group; the per-position store
 * builds it token by token.  The 0731 machinery this gate used to pin -- an
 * 8-row two-group window rebuilt from a re-projected tail, an indexer
 * compressor with its own state, a rewind projection ring (L171) -- is gone.
 *
 * Asserted per kv source at r = 1 (L = 32 + 1) against prefill(L - 1) + 1
 * classic decode step of the same token (the state the decode path builds):
 *   PLACEMENT (bit-level): after prefill(L), slot 0 is populated (score
 *     finite, kv non-zero) and slot 1 empty -- and the decode-built state has
 *     the same shape.  After prefill(L + 1) both slots are empty on both paths.
 *   PENDING ROW (printed, not asserted): slot 0 from the prefill forward vs
 *     the decode forward of the same token differ at the compressor input
 *     (E4M3 requantization, MoE routing), which no threshold separates from a
 *     wrong token.
 * Whole prompts shorter than a group (L = 1: slot 0 populated; L = 2: empty)
 * check placement only.  The ratio-1 source (20) keeps no state.
 *
 *   ./tests/comp_state_gate MODEL
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "pulsar_gpu.h"
#include "gate_entry.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define L_ALIGNED 32   /* 16 complete groups; L = 32 + 1; the templated prompt is ~41 tokens */

static const char *PROMPT =
    "The economic history of the Mediterranean is inseparable from its ports. Grain from Egypt, "
    "timber from the Levant, silver from Iberia and wine from the Aegean all moved by sea.";

static int g_fail = 0;

typedef struct {
    float *kv;     /* ratio * width */
    float *sc;
    uint32_t width;
    uint32_t rows;
    int ok;
} state_rows;

static int read_state(pulsar_gpu_graph *g, uint32_t il, state_rows *out) {
    const uint32_t bank = g->banks.n_banks ? g->banks.cur_bank : 0u;
    pulsar_gpu_tensor *kv = gpu_graph_bank_attn_state_kv_view(g, il, bank);
    pulsar_gpu_tensor *sc = gpu_graph_bank_attn_state_score_view(g, il, bank);
    out->ok = 0;
    if (!kv || !sc) { pulsar_gpu_tensor_free(kv); pulsar_gpu_tensor_free(sc); return 0; }
    const uint32_t width = PULSAR_N_HEAD_DIM;
    const uint32_t rows = pulsar_layer_compress_ratio(il);
    const uint64_t n = (uint64_t)rows * width;
    if (pulsar_gpu_tensor_bytes(kv) < n * sizeof(float) || pulsar_gpu_tensor_bytes(sc) < n * sizeof(float)) {
        fprintf(stderr, "comp_state_gate: kv source %u state is %llu bytes, expected >= %llu\n",
                il, (unsigned long long)pulsar_gpu_tensor_bytes(kv), (unsigned long long)(n * sizeof(float)));
        pulsar_gpu_tensor_free(kv); pulsar_gpu_tensor_free(sc);
        return 0;
    }
    out->width = width;
    out->rows = rows;
    out->kv = (float *)malloc(n * sizeof(float));
    out->sc = (float *)malloc(n * sizeof(float));
    int rc = out->kv && out->sc &&
             pulsar_gpu_tensor_read_f32(kv, 0, out->kv, n) &&
             pulsar_gpu_tensor_read_f32(sc, 0, out->sc, n);
    pulsar_gpu_tensor_free(kv); pulsar_gpu_tensor_free(sc);
    out->ok = rc;
    return rc;
}

static void free_state(state_rows *s) { free(s->kv); free(s->sc); s->kv = s->sc = NULL; s->ok = 0; }

/* A row is POPULATED when every score is finite and some kv is non-zero;
 * EMPTY when every score is -inf and every kv is 0.  Anything else is a
 * malformed row and fails on its own. */
static int row_kind(const state_rows *s, uint32_t row) {
    const float *kv = s->kv + (uint64_t)row * s->width;
    const float *sc = s->sc + (uint64_t)row * s->width;
    int finite = 0, ninf = 0, nz = 0;
    for (uint32_t j = 0; j < s->width; j++) {
        if (sc[j] == -INFINITY) ninf++; else if (isfinite(sc[j])) finite++;
        if (kv[j] != 0.0f) nz++;
    }
    if (finite == (int)s->width && nz > 0) return 1;          /* populated */
    if (ninf == (int)s->width && nz == 0) return 0;           /* empty */
    return -1;                                                /* malformed */
}

static double rel_l1(const float *a, const float *b, uint64_t n) {
    double num = 0.0, den = 0.0;
    for (uint64_t i = 0; i < n; i++) { num += fabs((double)a[i] - (double)b[i]); den += fabs((double)a[i]); }
    return den > 0.0 ? num / den : (num > 0.0 ? 1.0 : 0.0);
}

static int sync_prefix(pulsar_session *s, const pulsar_tokens *full, int len, char *err, size_t errlen) {
    pulsar_tokens p = *full;      /* borrowed view of the first len tokens */
    p.len = len;
    if (s->graph.banks.n_banks) {
        if (pulsar_session_bank_repoint(s, 0) != 0) { snprintf(err, errlen, "repoint 0"); return 1; }
    }
    pulsar_session_invalidate(s);
    if (pulsar_session_sync(s, &p, err, errlen) != 0) return 1;
    return 0;
}

static const char *kind_name(int k) { return k == 1 ? "populated" : k == 0 ? "empty" : "malformed"; }

/* PLACEMENT at position L: the pending slots [0, L % ratio) populated, the
 * rest empty.  B (the decode-built state) may be NULL for a placement-only
 * pass.  Returns 1 when the layer was checked. */
static int check_placement(const state_rows *A, const state_rows *B, uint32_t il, int L, const char *tag) {
    const uint32_t phase = (uint32_t)L % A->rows;
    for (uint32_t row = 0; row < A->rows; row++) {
        const int want = row < phase ? 1 : 0;
        const int got = row_kind(A, row);
        if (got != want) {
            printf("  FAIL kv source %2u %s L=%d: state slot %u is %s, expected %s\n", il, tag, L, row,
                   kind_name(got), kind_name(want));
            g_fail = 1;
        }
        if (B) {
            const int gotB = row_kind(B, row);
            if (gotB != want) {
                printf("  FAIL kv source %2u %s L=%d: DECODE-built state slot %u is %s, expected %s (fixture broken)\n",
                       il, tag, L, row, kind_name(gotB), kind_name(want));
                g_fail = 1;
            }
        }
    }
    return 1;
}

int GATE_ENTRY(int argc, char **argv) {
    g_fail = 0;
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }
    pulsar_engine_options opt = { .model_path = argv[1], .backend = PULSAR_BACKEND_CUDA };
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    pulsar_session *s = NULL;
    pulsar_tokens prompt = {0};
    int rc = 1;
    state_rows *A = NULL;
    {
        if (pulsar_session_create(&s, e, 4096) != 0) { fprintf(stderr, "session failed\n"); goto done; }
        pulsar_chat_begin(e, &prompt);
        pulsar_chat_append_message(e, &prompt, "user", PROMPT);
        pulsar_chat_append_assistant_prefix(e, &prompt, PULSAR_THINK_NONE);
        if (prompt.len < L_ALIGNED + 2) {
            fprintf(stderr, "comp_state_gate: prompt is %d tokens, need >= %d\n", prompt.len, L_ALIGNED + 2);
            goto done;
        }
        const uint32_t n_layer = PULSAR_N_LAYER;
        A = (state_rows *)calloc((size_t)n_layer, sizeof(state_rows));
        if (!A) goto done;
        char err[256];
        int total_checked = 0;
        /* L odd (pending token) and L even (consumed group), each against
         * prefill(L - 1) + one decode step of the same token. */
        for (int L = L_ALIGNED + 1; L <= L_ALIGNED + 2; L++) {
            /* Path A: one prefill of L tokens */
            if (sync_prefix(s, &prompt, L, err, sizeof(err))) { fprintf(stderr, "sync(%d): %s\n", L, err); goto done; }
            if (pulsar_session_pos(s) != L) { fprintf(stderr, "pos after sync(%d) = %d\n", L, pulsar_session_pos(s)); goto done; }
            for (uint32_t il = 0; il < n_layer; il++) {
                if (!gpu_graph_layer_has_comp_state(il)) continue;
                read_state(&s->graph, il, &A[il]);
            }
            /* Path B: prefill of L - 1 tokens, then one classic decode step of the same token */
            if (sync_prefix(s, &prompt, L - 1, err, sizeof(err))) { fprintf(stderr, "sync(%d): %s\n", L - 1, err); goto done; }
            if (pulsar_session_eval(s, prompt.v[L - 1], err, sizeof(err)) != 0) { fprintf(stderr, "eval: %s\n", err); goto done; }
            if (pulsar_session_pos(s) != L) { fprintf(stderr, "pos after decode = %d, expected %d\n", pulsar_session_pos(s), L); goto done; }
            int checked = 0;
            double min_pend = 0.0, worst_pend = 0.0;
            for (uint32_t il = 0; il < n_layer; il++) {
                if (!gpu_graph_layer_has_comp_state(il)) continue;
                state_rows *a = &A[il];
                if (!a->ok) continue;
                state_rows b;
                if (!read_state(&s->graph, il, &b)) { fprintf(stderr, "read decode state kv source %u\n", il); goto done; }
                checked += check_placement(a, &b, il, L, "prefill-vs-decode");
                if ((uint32_t)L % a->rows) {
                    /* PENDING ROW: printed, not asserted (see header). */
                    const double d_kv = rel_l1(b.kv, a->kv, a->width);
                    const double d_sc = rel_l1(b.sc, a->sc, a->width);
                    const double d = d_kv > d_sc ? d_kv : d_sc;
                    if (d > worst_pend) worst_pend = d;
                    if (min_pend == 0.0 || d < min_pend) min_pend = d;
                }
                free_state(a); free_state(&b);
            }
            if ((uint32_t)L % 2u)
                printf("L=%d (pending token): %d compressor states checked; pending slot relL1 %.3g..%.3g "
                       "(informational: prefill vs decode forward)%s\n", L, checked, min_pend, worst_pend, g_fail ? "" : "  OK");
            else
                printf("L=%d (group boundary): %d compressor states checked, both slots empty on both paths%s\n",
                       L, checked, g_fail ? "" : "  OK");
            total_checked += checked;
        }
        if (total_checked == 0) { fprintf(stderr, "comp_state_gate: no ratio-2 compressor state found\n"); goto done; }
        /* Whole prompts shorter than a group: placement only (there is no
         * shorter prefill to decode from). */
        for (int L = 1; L <= 2; L++) {
            if (sync_prefix(s, &prompt, L, err, sizeof(err))) { fprintf(stderr, "sync(%d): %s\n", L, err); goto done; }
            int checked = 0;
            for (uint32_t il = 0; il < n_layer; il++) {
                if (!gpu_graph_layer_has_comp_state(il)) continue;
                state_rows a;
                if (!read_state(&s->graph, il, &a)) continue;
                checked += check_placement(&a, NULL, il, L, "short-prompt");
                free_state(&a);
            }
            printf("short prompt L=%d: %d compressor states checked, placement%s\n", L, checked, g_fail ? " FAIL" : " OK");
        }
        /* L177: the speculative compact-prefilter slab must hold every row the
         * lane can admit (PULSAR_SPEC_LOGITS_ROWS).  It held 16 after the L117
         * budget went to 32; the host mirror is sized by the constant. */
        {
            pulsar_gpu_graph *g = &s->graph;
            const uint64_t need = (uint64_t)PULSAR_SPEC_LOGITS_ROWS * PULSAR_DSPARK_PREFILTER_ROW_I32 * sizeof(int32_t);
            const uint64_t have = g->dspark_prefilter_sel ? pulsar_gpu_tensor_bytes(g->dspark_prefilter_sel) : 0ull;
            if (have < need) {
                printf("  FAIL spec compact-prefilter slab holds %llu of the %u admitted rows (%llu < %llu bytes)\n",
                       (unsigned long long)(have / ((uint64_t)PULSAR_DSPARK_PREFILTER_ROW_I32 * sizeof(int32_t))),
                       (unsigned)PULSAR_SPEC_LOGITS_ROWS, (unsigned long long)have, (unsigned long long)need);
                g_fail = 1;
            } else {
                printf("spec compact-prefilter slab: %u rows, matches the lane's row budget  OK\n", (unsigned)PULSAR_SPEC_LOGITS_ROWS);
            }
        }
        rc = g_fail ? 1 : 0;
    }
done:
    if (A) { for (uint32_t i = 0; i < PULSAR_N_LAYER; i++) free_state(&A[i]); free(A); }
    pulsar_tokens_free(&prompt);
    if (s) pulsar_session_free(s);
    gate_engine_close(e);
    printf("COMP STATE GATE: %s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}

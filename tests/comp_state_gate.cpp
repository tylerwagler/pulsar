/* L168 GATE: the ratio-4 compressor state after a whole-prompt prefill has the
 * layout the decode store builds -- for prompts whose length is NOT a multiple
 * of the ratio.
 *
 * The state is 8 rows: rows 0..3 hold the last COMPLETE group of four tokens
 * (one per phase), rows 4..7 the partial group being built (row 4 + pos % 4),
 * which is what compressor_store_kernel writes on every decode step and what
 * the pool reads when the group completes.  After a zero-prefix prefill the
 * engine rebuilds that state from the chunk's tail, re-projected as decode
 * rows (gpu_graph_refresh_ratio4_compressor_state).  Until L168 the rebuild
 * wrote the last FOUR rows into rows 0..3 and left 4..7 empty, so a prompt of
 * length 4q + r (r != 0) lost its partial group and got a "complete group"
 * made of the wrong tokens -- three of every four single-chunk prompts.
 *
 * Asserted per layer, for the attention compressor and the indexer compressor,
 * at r = 1, 2, 3 (L = 32 + r), against prefill(L - r) + r classic decode steps
 * of the same tokens (the state the decode path builds for the same prefix):
 *   PLACEMENT (bit-level): after prefill(L), rows 4..4+r-1 are populated (score
 *     finite, kv non-zero), rows 4+r..7 are empty (score -inf, kv 0), rows 0..3
 *     populated -- and the decode-built state has the same shape.
 *   COMPLETE GROUP (bounded): rows 0..3 after prefill(L) are within
 *     COMP_GROUP_TOL relL1 of rows 0..3 after prefill(L - r).  Same four tokens,
 *     rebuilt by the same code, but the prefill lane is not chunk-mate neutral:
 *     with one more token in the chunk the hidden states of tokens 28..31 are
 *     byte-identical through layer 22 and differ by 0.4-6% relL1 from layer 24
 *     on (grouped MoE problem sizes change; measured 2026-09-04).  The wrong-
 *     token layout of the pre-L168 rebuild read 0.47-1.2 on the same rows, so
 *     0.25 separates the two with ~4x margin each way; both numbers are printed.
 * The partial rows are compared the same way and PRINTED, not asserted: they
 * come from the prefill forward vs the decode forward of the same token and
 * differ by 5-44% relL1 at the compressor input (E4M3 requantization, MoE
 * routing), which no threshold separates from a wrong token.
 *
 * Mutation-validated against the pre-L168 rebuild: PLACEMENT fails (rows 4..
 * empty) and COMPLETE GROUP fails (0.47-1.2: r of the four rows are other tokens).
 *
 * RING ALIASING (L171, bit-level): after prefill(9) the rewind projection ring
 * holds positions 1..8 of the chunk's PREFILL-arm projections.  The rebuild
 * used to re-project the tail (tokens 4..8) into the same scratch buffer's
 * rows 0..4 before the ring deposit read rows 1..8, so ring positions 1..4
 * were byte-identical to the rebuild's state rows for tokens 5..8.  The gate
 * asserts, per layer and compressor, that positions 1..8 are deposited and
 * that no ring KV row 1..4 equals the state KV row it aliased (rows 1, 2, 3 of
 * the complete group and partial row 4; the score half differs by the bias).
 * Mutation-validated: the pre-L171 helper fails it on every ratio-4 layer.
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

#define COMP_GROUP_TOL 0.25
#define STATE_ROWS 8u
#define L_ALIGNED 32   /* 8 complete groups; L = 32 + r; the templated prompt is ~41 tokens */

static const char *PROMPT =
    "The economic history of the Mediterranean is inseparable from its ports. Grain from Egypt, "
    "timber from the Levant, silver from Iberia and wine from the Aegean all moved by sea.";

static int g_fail = 0;

typedef struct {
    float *kv;     /* STATE_ROWS * width */
    float *sc;
    uint32_t width;
    int ok;
} state_rows;

static int read_state(pulsar_gpu_graph *g, uint32_t il, int indexer, state_rows *out) {
    const uint32_t bank = g->banks.n_banks ? g->banks.cur_bank : 0u;
    pulsar_gpu_tensor *kv = indexer ? gpu_graph_bank_index_state_kv_view(g, il, bank)
                                    : gpu_graph_bank_attn_state_kv_view(g, il, bank);
    pulsar_gpu_tensor *sc = indexer ? gpu_graph_bank_index_state_score_view(g, il, bank)
                                    : gpu_graph_bank_attn_state_score_view(g, il, bank);
    out->ok = 0;
    if (!kv || !sc) { pulsar_gpu_tensor_free(kv); pulsar_gpu_tensor_free(sc); return 0; }
    const uint32_t width = indexer ? 2u * PULSAR_N_INDEXER_HEAD_DIM : 2u * PULSAR_N_HEAD_DIM;
    const uint64_t n = (uint64_t)STATE_ROWS * width;
    if (pulsar_gpu_tensor_bytes(kv) < n * sizeof(float) || pulsar_gpu_tensor_bytes(sc) < n * sizeof(float)) {
        fprintf(stderr, "comp_state_gate: layer %u %s state is %llu bytes, expected >= %llu\n",
                il, indexer ? "indexer" : "attn",
                (unsigned long long)pulsar_gpu_tensor_bytes(kv), (unsigned long long)(n * sizeof(float)));
        pulsar_gpu_tensor_free(kv); pulsar_gpu_tensor_free(sc);
        return 0;
    }
    out->width = width;
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

/* Check one compressor (attn or indexer) at one r: returns layers checked. */
static int check_one(pulsar_gpu_graph *gA, state_rows *A, state_rows *B, uint32_t il, int indexer, uint32_t r, int has_full,
                     double *worst_full, double *min_part, double *worst_part) {
    (void)gA;
    const char *what = indexer ? "indexer" : "attn";
    const uint32_t w = A->width;
    /* PLACEMENT on the prefill(L) state (has_full: the chunk has a complete group) */
    for (uint32_t row = 0; row < STATE_ROWS; row++) {
        const int want = (row < 4u) ? has_full : (row < 4u + r ? 1 : 0);
        const int got = row_kind(A, row);
        if (got != want) {
            printf("  FAIL layer %2u %-7s r=%u: state row %u is %s, expected %s\n", il, what, r, row,
                   got == 1 ? "populated" : got == 0 ? "empty" : "malformed", want ? "populated" : "empty");
            g_fail = 1;
        }
        const int gotB = B ? row_kind(B, row) : want;
        if (gotB != want) {
            printf("  FAIL layer %2u %-7s r=%u: DECODE-built state row %u is %s, expected %s (fixture broken)\n",
                   il, what, r, row, gotB == 1 ? "populated" : gotB == 0 ? "empty" : "malformed",
                   want ? "populated" : "empty");
            g_fail = 1;
        }
    }
    if (!B) return 1;   /* short-prompt pass: placement only */
    /* COMPLETE GROUP: rows 0..3 within COMP_GROUP_TOL of the aligned prefill's rebuild */
    const double d_full_kv = rel_l1(B->kv, A->kv, 4ull * w);
    const double d_full_sc = rel_l1(B->sc, A->sc, 4ull * w);
    const double d_full = d_full_kv > d_full_sc ? d_full_kv : d_full_sc;
    if (d_full > *worst_full) *worst_full = d_full;
    if (d_full > COMP_GROUP_TOL) {
        printf("  FAIL layer %2u %-7s r=%u: complete group vs prefill(L-%u): relL1 kv %.3g, score %.3g (tol %.2f)\n",
               il, what, r, r, d_full_kv, d_full_sc, COMP_GROUP_TOL);
        g_fail = 1;
    }
    /* PARTIAL rows vs the decode-built state: printed, not asserted (see header). */
    const double d_part_kv = rel_l1(B->kv + 4ull * w, A->kv + 4ull * w, (uint64_t)r * w);
    const double d_part_sc = rel_l1(B->sc + 4ull * w, A->sc + 4ull * w, (uint64_t)r * w);
    const double d_part = d_part_kv > d_part_sc ? d_part_kv : d_part_sc;
    if (d_part > *worst_part) *worst_part = d_part;
    if (*min_part == 0.0 || d_part < *min_part) *min_part = d_part;
    return 1;
}

/* L171: ring row at absolute position `pos` (slot pos % PULSAR_REWIND_RING_DEPTH). */
static int read_ring_row(pulsar_gpu_graph *g, uint32_t il, int indexer, uint32_t pos, uint32_t width,
                         float *kv, float *sc) {
    pulsar_gpu_tensor *rk = indexer ? g->layer_index_proj_kv[il] : g->layer_attn_proj_kv[il];
    pulsar_gpu_tensor *rs = indexer ? g->layer_index_proj_sc[il] : g->layer_attn_proj_sc[il];
    if (!rk || !rs) return 0;
    const uint64_t off = (uint64_t)(pos % PULSAR_REWIND_RING_DEPTH) * width;
    return pulsar_gpu_tensor_read_f32(rk, off, kv, width) && pulsar_gpu_tensor_read_f32(rs, off, sc, width);
}

/* After prefill(L_RING = 9): ring positions 1..4 must not be the rebuild's
 * rows.  Tail = tokens 4..8 (n_full 4 + rem 1): scratch row r held token 4+r,
 * so ring position p (scratch row p) aliased state row p for p = 1..3 and
 * state row 4 (partial, token 8) for p = 4. */
#define L_RING 9
static int check_ring_alias(pulsar_session *s, int indexer, int *n_checked) {
    pulsar_gpu_graph *g = &s->graph;
    const char *what = indexer ? "indexer" : "attn";
    int fails = 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (pulsar_layer_compress_ratio(il) != 4u) continue;
        state_rows st;
        if (!read_state(g, il, indexer, &st)) continue;
        const uint32_t w = st.width;
        float *rkv = (float *)malloc(w * sizeof(float));
        float *rsc = (float *)malloc(w * sizeof(float));
        int layer_ok = rkv && rsc;
        /* Positions 1..8 must be deposited at all (kv non-zero): a missing
         * deposit would make the aliasing check below pass vacuously. */
        for (uint32_t p = 1; layer_ok && p <= 8; p++) {
            if (!read_ring_row(g, il, indexer, p, w, rkv, rsc)) { layer_ok = 0; break; }
            int nz = 0;
            for (uint32_t j = 0; j < w; j++) nz += rkv[j] != 0.0f;
            if (nz == 0) {
                printf("  FAIL layer %2u %-7s ring position %u is empty: the chunk tail was not deposited\n", il, what, p);
                fails++;
            }
        }
        /* The KV half is copied verbatim into both the state and the ring
         * (the score half gets the positional bias added in the state), so
         * aliasing shows as byte-identical KV rows. */
        for (uint32_t p = 1; layer_ok && p <= 4; p++) {
            if (!read_ring_row(g, il, indexer, p, w, rkv, rsc)) { layer_ok = 0; break; }
            const uint32_t srow = p < 4u ? p : 4u;
            int same = 1;
            for (uint32_t j = 0; j < w && same; j++)
                if (rkv[j] != st.kv[(uint64_t)srow * w + j]) same = 0;
            if (same) {
                printf("  FAIL layer %2u %-7s ring position %u KV is byte-identical to state row %u: the deposit read "
                       "the rebuild's re-projection (L171)\n", il, what, p, srow);
                fails++;
            }
        }
        if (layer_ok) (*n_checked)++;
        free(rkv); free(rsc); free_state(&st);
    }
    return fails;
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
    state_rows *A = NULL, *B = NULL;
    {
        if (pulsar_session_create(&s, e, 4096) != 0) { fprintf(stderr, "session failed\n"); goto done; }
        pulsar_chat_begin(e, &prompt);
        pulsar_chat_append_message(e, &prompt, "user", PROMPT);
        pulsar_chat_append_assistant_prefix(e, &prompt, PULSAR_THINK_NONE);
        if (prompt.len < L_ALIGNED + 4) {
            fprintf(stderr, "comp_state_gate: prompt is %d tokens, need >= %d\n", prompt.len, L_ALIGNED + 4);
            goto done;
        }
        const uint32_t n_layer = PULSAR_N_LAYER;
        A = (state_rows *)calloc((size_t)n_layer * 2u, sizeof(state_rows));
        B = (state_rows *)calloc((size_t)n_layer * 2u, sizeof(state_rows));
        if (!A || !B) goto done;
        char err[256];
        int total_checked = 0;
        for (uint32_t r = 1; r <= 3; r++) {
            const int L = L_ALIGNED + (int)r;
            /* Path A: one prefill of L tokens */
            if (sync_prefix(s, &prompt, L, err, sizeof(err))) { fprintf(stderr, "sync(%d): %s\n", L, err); goto done; }
            if (pulsar_session_pos(s) != L) { fprintf(stderr, "pos after sync(%d) = %d\n", L, pulsar_session_pos(s)); goto done; }
            for (uint32_t il = 0; il < n_layer; il++) {
                if (pulsar_layer_compress_ratio(il) != 4u) continue;
                read_state(&s->graph, il, 0, &A[il * 2u]);
                read_state(&s->graph, il, 1, &A[il * 2u + 1u]);
            }
            /* Path B: prefill of L - r tokens, then r classic decode steps of the same tokens */
            if (sync_prefix(s, &prompt, L - (int)r, err, sizeof(err))) { fprintf(stderr, "sync(%d): %s\n", L - (int)r, err); goto done; }
            for (int t = L - (int)r; t < L; t++) {
                if (pulsar_session_eval(s, prompt.v[t], err, sizeof(err)) != 0) { fprintf(stderr, "eval: %s\n", err); goto done; }
            }
            if (pulsar_session_pos(s) != L) { fprintf(stderr, "pos after decode = %d, expected %d\n", pulsar_session_pos(s), L); goto done; }
            int checked = 0;
            double worst_full = 0.0, min_part = 0.0, worst_part = 0.0;
            for (uint32_t il = 0; il < n_layer; il++) {
                if (pulsar_layer_compress_ratio(il) != 4u) continue;
                for (int ix = 0; ix < 2; ix++) {
                    state_rows *a = &A[il * 2u + (uint32_t)ix];
                    if (!a->ok) continue;
                    state_rows *b = &B[il * 2u + (uint32_t)ix];
                    if (!read_state(&s->graph, il, ix, b)) { fprintf(stderr, "read decode state layer %u\n", il); goto done; }
                    checked += check_one(&s->graph, a, b, il, ix, r, 1, &worst_full, &min_part, &worst_part);
                    free_state(a); free_state(b);
                }
            }
            printf("r=%u (L=%d): %d compressor states checked; complete group worst relL1 %.3g (tol %.2f); "
                   "partial rows %.3g..%.3g (informational: prefill vs decode forward)%s\n",
                   r, L, checked, worst_full, COMP_GROUP_TOL, min_part, worst_part, g_fail ? "" : "  OK");
            total_checked += checked;
        }
        if (total_checked == 0) { fprintf(stderr, "comp_state_gate: no ratio-4 compressor state found\n"); goto done; }
        /* L175: whole prompts shorter than the window (L = 1, 2, 3): no complete
         * group, rows 0..3 empty, rows 4..4+L-1 the partial group.  Placement
         * only (there is no shorter prefill to decode from). */
        for (int L = 1; L <= 3; L++) {
            if (sync_prefix(s, &prompt, L, err, sizeof(err))) { fprintf(stderr, "sync(%d): %s\n", L, err); goto done; }
            int checked = 0;
            double d0 = 0.0, d1 = 0.0, d2 = 0.0;
            for (uint32_t il = 0; il < n_layer; il++) {
                if (pulsar_layer_compress_ratio(il) != 4u) continue;
                for (int ix = 0; ix < 2; ix++) {
                    state_rows a;
                    if (!read_state(&s->graph, il, ix, &a)) continue;
                    checked += check_one(&s->graph, &a, NULL, il, ix, (uint32_t)L, 0, &d0, &d1, &d2);
                    free_state(&a);
                }
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
        /* L171: ring aliasing after a 9-token whole prompt */
        {
            if (sync_prefix(s, &prompt, L_RING, err, sizeof(err))) { fprintf(stderr, "sync(%d): %s\n", L_RING, err); goto done; }
            int n_ring = 0;
            const int f = check_ring_alias(s, 0, &n_ring) + check_ring_alias(s, 1, &n_ring);
            if (f) g_fail = 1;
            if (n_ring == 0) { fprintf(stderr, "comp_state_gate: no projection ring found\n"); goto done; }
            printf("ring aliasing (L=%d): %d compressor rings checked, %d aliased rows%s\n", L_RING, n_ring, f, f ? "" : "  OK");
        }
        rc = g_fail ? 1 : 0;
    }
done:
    if (A) { for (uint32_t i = 0; i < 2u * PULSAR_N_LAYER; i++) free_state(&A[i]); free(A); }
    if (B) { for (uint32_t i = 0; i < 2u * PULSAR_N_LAYER; i++) free_state(&B[i]); free(B); }
    pulsar_tokens_free(&prompt);
    if (s) pulsar_session_free(s);
    gate_engine_close(e);
    printf("COMP STATE GATE: %s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}

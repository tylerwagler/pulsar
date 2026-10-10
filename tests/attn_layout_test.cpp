/* HOST gate for the attention layout table (src/engine/model_layout.cpp).
 *
 * No model, no device: pulsar_attn_layout_install is a pure function of the
 * artifact's declared metadata, so the two-profile mode table is fully
 * checkable on any box.  That matters because the FAILURE MODE is silent-ish:
 * a wrong mode makes a layer skip its compressor (or run one it does not have)
 * and the only symptom is wrong attention much later.
 *
 * The metadata below is NOT copied from the shape profiles -- it is read out of
 * the real artifacts with gate-baseline/l218-v41/read_attn_layout.py, so a
 * profile that drifted from its artifact cannot make this gate agree with it:
 *
 *   V4.1  ds41flash-mxfp4src-full384-bf16head-dspark-v2.gguf
 *         block_count 40
 *         compress_ratios  [0,0, 2 x18, 1 x20]
 *         kv_source_layers     [2, 8, 14, 20]
 *         index_source_layers  [2, 8, 14, 20, 24, 28, 32, 36]
 *         candidate_source_layer 20
 *
 *   0731  oracle-zeroq8-99gb.gguf
 *         block_count 43
 *         compress_ratios  [0,0, 4,128, 4,128, ... 4]   (44 entries; see below)
 *         kv_source_layers / index_source_layers / candidate_source_layer
 *           -- ALL THREE KEYS ARE ABSENT from the artifact.  weights.cpp falls
 *              back to the profile's sets for 0731, which is exactly how a
 *              ratio-128 layer ends up with a ratio-4 "index source": the
 *              fallback's index set is the 21 even layers.  That asymmetry is
 *              the thing this gate exists to pin.
 *
 * The 0731 ratio array declares 44 entries for a 43-layer model: index 43 is a
 * trailing 0 that no layer reads (the table is built for il < n_layer).  It is
 * kept here verbatim rather than trimmed, so the gate records what the artifact
 * actually says.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/wait.h>
#include <unistd.h>

#include "pulsar_engine_internal.h"

static unsigned g_checks, g_failures;

static void check(int ok, const char *what, unsigned il, const char *detail) {
    g_checks++;
    if (ok) return;
    g_failures++;
    printf("FAIL: layer %u: %s (%s)\n", il, what, detail);
}

/* ---- V4.1, as the artifact declares it ---------------------------------- */
static const uint32_t V41_RATIOS[40] = {
    0, 0,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
};
static const uint32_t V41_KV[4]    = { 2, 8, 14, 20 };
static const uint32_t V41_IDX[8]   = { 2, 8, 14, 20, 24, 28, 32, 36 };
#define V41_CAND 20

/* The mode each layer must end up in, derived by hand from the semantics in the
 * pulsar_attn_mode doc block: a kv source that also indexes is FULL; one that
 * does not is FULL_UNINDEXED; an index source that is not a kv source is
 * REINDEX; neither is REUSE. */
static const pulsar_attn_mode V41_WANT[40] = {
    PULSAR_ATTN_WINDOW, PULSAR_ATTN_WINDOW,
    PULSAR_ATTN_FULL, PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE,
    PULSAR_ATTN_REUSE, PULSAR_ATTN_FULL, PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE,
    PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE, PULSAR_ATTN_FULL, PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE,
    PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE, PULSAR_ATTN_FULL,
    PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE,
    PULSAR_ATTN_REINDEX, PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE,
    PULSAR_ATTN_REINDEX, PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE,
    PULSAR_ATTN_REINDEX, PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE,
    PULSAR_ATTN_REINDEX, PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE, PULSAR_ATTN_REUSE,
};

/* ---- 0731, as the artifact declares it ---------------------------------- */
static const uint32_t V4_RATIOS[44] = {
    0, 0,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4,
    0,
};
/* The profile's sets, which is what the engine falls back to because the 0731
 * artifact carries none of the three keys. */
static const uint32_t V4_KV[41] = {
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42,
};
static const uint32_t V4_IDX[21] = {
    2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
    22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42,
};
#define V4_CAND (-1)

/* Even layers 2..42 are FULL (they compress AND index); odd layers 3..41 are
 * FULL_UNINDEXED (they compress and publish no top-k).  Layers 0-1 are window
 * only.  This is S2's asymmetry: 41 kv sources, 21 index sources. */
static void want_v4(unsigned il, pulsar_attn_mode *mode, const char **why) {
    if (il < 2)                       { *mode = PULSAR_ATTN_WINDOW;         *why = "window-only layer"; }
    else if ((il & 1u) == 0u)         { *mode = PULSAR_ATTN_FULL;           *why = "ratio-4 CSA source: compresses and indexes"; }
    else                              { *mode = PULSAR_ATTN_FULL_UNINDEXED; *why = "ratio-128 HCA source: compresses, no indexer"; }
}

static void install_and_grade(const pulsar_shape *shape, const char *what,
                              const uint32_t *ratios, const uint32_t *kv, uint32_t n_kv,
                              const uint32_t *idx, uint32_t n_index, int32_t cand) {
    g_pulsar_shape = *shape;
    pulsar_attn_layout_install(ratios, kv, n_kv, idx, n_index, cand);
    printf("--- %s: %u layers\n", what, (unsigned)PULSAR_N_LAYER);

    unsigned counts[5] = {0};
    /* Counted rather than checked per layer: "an indexer-running layer is ratio
     * 4" is VACUOUSLY true of today's data (it follows from the artifact's own
     * source sets), so a per-layer guard there can be deleted without any gate
     * noticing.  The biconditional below is the live statement -- every ratio-4
     * layer runs an indexer AND every indexer-running layer is ratio 4 -- and it
     * fails the moment a profile or artifact breaks the coupling. */
    unsigned n_idx_own = 0, n_ratio4 = 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const pulsar_layer_attn *a = pulsar_layer_attn_layout(il);
        counts[a->mode]++;

        pulsar_attn_mode want_mode = a->mode;
        const char *why = "";
        if (shape->variant == PULSAR_VARIANT_V4) want_v4(il, &want_mode, &why);
        else if (il < 40)                        want_mode = V41_WANT[il];

        char detail[160];
        snprintf(detail, sizeof detail, "got %s, want %s -- %s", pulsar_attn_mode_name(a->mode), pulsar_attn_mode_name(want_mode), why);
        check(a->mode == want_mode, "wrong attention mode", il, detail);

        /* The compressor's STATE geometry must follow coff (pulsar_compress_coff),
         * not be hardcoded to V4.1's no-overlap width.  Pinned as concrete
         * numbers rather than by re-deriving the formula, so a change to the
         * formula that keeps the two sites agreeing still has to face this. */
        {
            const uint32_t w    = pulsar_comp_row_width(a->ratio, (uint32_t)shape->n_head_dim);
            const uint32_t rows = a->ratio > 1u ? pulsar_comp_state_rows(a->ratio) : 0u;
            const uint32_t hd   = (uint32_t)shape->n_head_dim;
            uint32_t want_w = hd, want_rows = a->ratio > 1u ? a->ratio : 0u;
            if (a->ratio == 4u) { want_w = 2u * hd; want_rows = 8u; }   /* overlap: two-group */
            snprintf(detail, sizeof detail, "ratio %u: got %ux%u, want %ux%u",
                     a->ratio, w, rows, want_w, want_rows);
            check(w == want_w && rows == want_rows, "compressor state geometry is wrong", il, detail);
            if (shape->variant == PULSAR_VARIANT_V41) {
                /* The inertness proof for the coff-aware geometry: V4.1 uses no
                 * ratio that overlaps, so coff is 1 everywhere and its state is
                 * bit-for-bit what it was before. */
                snprintf(detail, sizeof detail, "ratio %u has coff %u", a->ratio, pulsar_compress_coff(a->ratio));
                check(pulsar_compress_coff(a->ratio) == 1u,
                      "V4.1 must never overlap -- the state geometry change is only inert if it does not", il, detail);
            }
            /* The INDEXER's OWN compressor is a SECOND Compressor: V4 builds
             * `Compressor(args, ratio, head_dim=n_indexer_head_dim, rotate=True)`
             * inside the Indexer, so at ratio 4 it is the overlap variant at
             * head_dim 128 and its lane is coff*ratio rows of
             * coff*index_head_dim.  Pinned as concrete numbers before the kernel
             * exists, for the same reason the main compressor's is -- and it also
             * pins the COUPLING the reference creates: an indexer that owns its
             * compressor only exists where that compressor overlaps. */
            if (shape->indexer_own_compressor) {
                const uint32_t ihd = (uint32_t)shape->n_indexer_head_dim;
                const uint32_t iw  = pulsar_comp_row_width(a->ratio, ihd);
                const uint32_t ir  = a->ratio > 1u ? pulsar_comp_state_rows(a->ratio) : 0u;
                if (pulsar_attn_runs_indexer(a->mode)) {
                    n_idx_own++;
                    snprintf(detail, sizeof detail, "indexer lane %ux%u, want %ux%u (head_dim %u)",
                             iw, ir, 2u * ihd, 8u, ihd);
                    check(a->ratio != 4u || (iw == 2u * ihd && ir == 8u),
                          "the indexer's own compressor state geometry is wrong", il, detail);
                }
                if (a->ratio == 4u) n_ratio4++;
            }
        }

        /* A layer's own ratio must equal the ratio of every source it reads.
         * The install enforces this, so reaching here with a mismatch would mean
         * the check was relaxed; assert the postcondition independently. */
        if (a->mode != PULSAR_ATTN_WINDOW) {
            snprintf(detail, sizeof detail, "ratio %u, kv_source %u (ratio %u)", a->ratio, a->kv_source, ratios[a->kv_source]);
            check(ratios[a->kv_source] == a->ratio, "kv source ratio differs from the layer's", il, detail);
            if (pulsar_attn_reads_index(a->mode)) {
                snprintf(detail, sizeof detail, "ratio %u, index_source %u (ratio %u)", a->ratio, a->index_source, ratios[a->index_source]);
                check(ratios[a->index_source] == a->ratio, "index source ratio differs from the layer's", il, detail);
            }
        }
    }

    if (shape->indexer_own_compressor) {
        char idetail[96];
        snprintf(idetail, sizeof idetail, "%u indexer-running vs %u ratio-4 layers", n_idx_own, n_ratio4);
        check(n_idx_own > 0u && n_idx_own == n_ratio4,
              "an indexer that owns its compressor exists exactly on the overlapping layers", 0, idetail);
        printf("    indexer's own compressor: %u layers (ratio-4 layers %u)\n", n_idx_own, n_ratio4);
    }
    unsigned total = 0;
    for (unsigned m = 0; m < 5; m++) total += counts[m];
    printf("    WINDOW %u  FULL %u  REINDEX %u  REUSE %u  FULL_UNINDEXED %u  (total %u)\n",
           counts[PULSAR_ATTN_WINDOW], counts[PULSAR_ATTN_FULL], counts[PULSAR_ATTN_REINDEX],
           counts[PULSAR_ATTN_REUSE], counts[PULSAR_ATTN_FULL_UNINDEXED], total);
    check(total == PULSAR_N_LAYER, "the mode counts do not cover the backbone", PULSAR_N_LAYER - 1, "count");
}

/* L269 W2: a LAYER-SUBSET FIXTURE of V4.1 (tools/container build.py --layers 1,2,3,14,20,21,24, what
 * /mnt/models/v41-tp1-fixture keeps): the block table must be the source layers' own modes, the sources and the
 * candidate re-bound to the kept blocks, and a fixture that drops a source its kept readers read must refuse
 * (the install exits; run in a child). */
static const uint32_t FIX_SRC[7] = { 1, 2, 3, 14, 20, 21, 24 };

static void fixture_install(const uint32_t *src, uint32_t n, const uint32_t *ratios, const uint32_t *kv, uint32_t n_kv,
                            const uint32_t *idx, uint32_t n_idx, int32_t cand) {
    g_pulsar_shape = PULSAR_SHAPE_V41;
    if (!pulsar_layer_subset_install(src, n)) exit(2);
    pulsar_attn_layout_install(ratios, kv, n_kv, idx, n_idx, cand);
}

static int fixture_refuses(const uint32_t *src, uint32_t n, const uint32_t *ratios, const uint32_t *kv, uint32_t n_kv,
                           const uint32_t *idx, uint32_t n_idx, int32_t cand) {
    fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) {
        fixture_install(src, n, ratios, kv, n_kv, idx, n_idx, cand);
        _exit(0);   /* installed: the refusal did not fire */
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) != 0;
}

static void fixture_checks(void) {
    static const uint32_t ratios[7] = { 0, 2, 2, 2, 1, 1, 1 };
    static const uint32_t kv[3] = { 1, 3, 4 };            /* source 2, 14, 20 */
    static const uint32_t idx[4] = { 1, 3, 4, 6 };        /* source 2, 14, 20, 24 */
    fixture_install(FIX_SRC, 7, ratios, kv, 3, idx, 4, 4);
    printf("--- V4.1 layer-subset fixture: %u blocks of source layers 1,2,3,14,20,21,24\n", (unsigned)PULSAR_N_LAYER);
    static const pulsar_attn_mode want[7] = {
        PULSAR_ATTN_WINDOW, PULSAR_ATTN_FULL, PULSAR_ATTN_REUSE, PULSAR_ATTN_FULL, PULSAR_ATTN_FULL,
        PULSAR_ATTN_REUSE, PULSAR_ATTN_REINDEX,
    };
    for (uint32_t b = 0; b < 7; b++) {
        const pulsar_layer_attn *a = pulsar_layer_attn_layout(b);
        char detail[160];
        snprintf(detail, sizeof detail, "source layer %u: got %s, want %s (V4.1 layer %u's own mode)", FIX_SRC[b],
                 pulsar_attn_mode_name(a->mode), pulsar_attn_mode_name(want[b]), FIX_SRC[b]);
        check(a->mode == want[b] && a->mode == V41_WANT[FIX_SRC[b]], "fixture block mode", b, detail);
        check(pulsar_layer_source(b) == FIX_SRC[b], "fixture block's source layer", b, "pulsar_layer_source");
    }
    check(pulsar_layer_attn_layout(2)->kv_source == 1 && pulsar_layer_attn_layout(5)->kv_source == 4,
          "a REUSE block reads its source's block", 2, "kv_source");
    check(pulsar_layer_attn_layout(4)->candidate_source, "block 4 (source 20) publishes the candidate mask", 4, "cand");
    check(pulsar_layer_attn_layout(6)->uses_candidates, "block 6 (source 24) scores inside the candidate mask", 6,
          "uses_candidates");
    check(pulsar_layer_subset(), "the fixture is announced as one", 0, "pulsar_layer_subset");

    /* refusals: a reader whose source the fixture drops, and an index layer whose candidate source it drops */
    static const uint32_t no2[6] = { 1, 3, 14, 20, 21, 24 };
    static const uint32_t r_no2[6] = { 0, 2, 2, 1, 1, 1 };
    static const uint32_t kv_no2[2] = { 2, 3 }, idx_no2[3] = { 2, 3, 5 };
    check(fixture_refuses(no2, 6, r_no2, kv_no2, 2, idx_no2, 3, 3),
          "a fixture keeping layer 3 without its kv source 2 refuses", 1, "reader of a dropped source");
    static const uint32_t no20[4] = { 1, 2, 14, 24 };
    static const uint32_t r_no20[4] = { 0, 2, 2, 1 };
    static const uint32_t kv_no20[2] = { 1, 2 }, idx_no20[3] = { 1, 2, 3 };
    check(fixture_refuses(no20, 4, r_no20, kv_no20, 2, idx_no20, 3, -1),
          "a fixture keeping layer 24 without the kv / candidate source 20 refuses", 3, "dropped source 20");
    static const uint32_t bad_order[3] = { 2, 1, 3 };
    g_pulsar_shape = PULSAR_SHAPE_V41;
    check(!pulsar_layer_subset_install(bad_order, 3), "a map that is not ascending refuses", 0, "ascending");
    g_pulsar_shape = PULSAR_SHAPE_V41;
    check(pulsar_layer_subset_install(NULL, 0) && !pulsar_layer_subset(), "an empty map is the identity", 0, "n = 0");
}

int main(void) {
    printf("attention layout gate: both profiles, from real artifact metadata\n");

    install_and_grade(&PULSAR_SHAPE_V41, "V4.1 (ds41flash ... -v2)", V41_RATIOS,
                      V41_KV, 4, V41_IDX, 8, V41_CAND);
    install_and_grade(&PULSAR_SHAPE_V4, "0731 (oracle-zeroq8-99gb)", V4_RATIOS,
                      V4_KV, 41, V4_IDX, 21, V4_CAND);

    /* The specific row the old unconditional ratio check refused: a ratio-128
     * layer whose "index source" is the last even layer.  Named here so that if
     * the asymmetry is ever re-broken, the failure says which layer and why. */
    const pulsar_layer_attn *hca = pulsar_layer_attn_layout(3);
    check(hca->ratio == 128, "layer 3 should be 0731's first ratio-128 HCA layer", 3, "ratio");
    check(hca->index_source == 2, "layer 3's index source is the last even layer", 3, "index_source");
    check(hca->mode == PULSAR_ATTN_FULL_UNINDEXED, "layer 3 compresses without an indexer", 3, pulsar_attn_mode_name(hca->mode));
    check(pulsar_attn_owns_kv(hca->mode), "FULL_UNINDEXED still owns its compressor", 3, pulsar_attn_mode_name(hca->mode));
    check(!pulsar_attn_runs_indexer(hca->mode), "FULL_UNINDEXED runs no indexer", 3, pulsar_attn_mode_name(hca->mode));
    check(pulsar_attn_reads_index(hca->mode) == false,
          "FULL_UNINDEXED must NOT read an index source -- this is what makes the ratio check skippable for it",
          3, pulsar_attn_mode_name(hca->mode));
    check(pulsar_attn_reads_index(PULSAR_ATTN_REUSE) == true,
          "REUSE consumes a top-k it did not compute, so it does read an index source",
          3, "predicate");

    fixture_checks();

    printf("attention layout gate: %u checks, %u failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

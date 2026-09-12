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
#include <string.h>
#include <stdint.h>

#include "pulsar_engine_internal.h"

static unsigned g_checks, g_failures;

static const char *mode_name(pulsar_attn_mode m) {
    switch (m) {
    case PULSAR_ATTN_WINDOW:         return "WINDOW";
    case PULSAR_ATTN_FULL:           return "FULL";
    case PULSAR_ATTN_REINDEX:        return "REINDEX";
    case PULSAR_ATTN_REUSE:          return "REUSE";
    case PULSAR_ATTN_FULL_UNINDEXED: return "FULL_UNINDEXED";
    }
    return "?";
}

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
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const pulsar_layer_attn *a = pulsar_layer_attn_layout(il);
        counts[a->mode]++;

        pulsar_attn_mode want_mode = a->mode;
        const char *why = "";
        if (shape->variant == PULSAR_VARIANT_V4) want_v4(il, &want_mode, &why);
        else if (il < 40)                        want_mode = V41_WANT[il];

        char detail[160];
        snprintf(detail, sizeof detail, "got %s, want %s -- %s", mode_name(a->mode), mode_name(want_mode), why);
        check(a->mode == want_mode, "wrong attention mode", il, detail);

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

    unsigned total = 0;
    for (unsigned m = 0; m < 5; m++) total += counts[m];
    printf("    WINDOW %u  FULL %u  REINDEX %u  REUSE %u  FULL_UNINDEXED %u  (total %u)\n",
           counts[PULSAR_ATTN_WINDOW], counts[PULSAR_ATTN_FULL], counts[PULSAR_ATTN_REINDEX],
           counts[PULSAR_ATTN_REUSE], counts[PULSAR_ATTN_FULL_UNINDEXED], total);
    check(total == PULSAR_N_LAYER, "the mode counts do not cover the backbone", PULSAR_N_LAYER - 1, "count");
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
    check(hca->mode == PULSAR_ATTN_FULL_UNINDEXED, "layer 3 compresses without an indexer", 3, mode_name(hca->mode));
    check(pulsar_attn_owns_kv(hca->mode), "FULL_UNINDEXED still owns its compressor", 3, mode_name(hca->mode));
    check(!pulsar_attn_runs_indexer(hca->mode), "FULL_UNINDEXED runs no indexer", 3, mode_name(hca->mode));
    check(pulsar_attn_reads_index(hca->mode) == false,
          "FULL_UNINDEXED must NOT read an index source -- this is what makes the ratio check skippable for it",
          3, mode_name(hca->mode));
    check(pulsar_attn_reads_index(PULSAR_ATTN_REUSE) == true,
          "REUSE consumes a top-k it did not compute, so it does read an index source",
          3, "predicate");

    printf("attention layout gate: %u checks, %u failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

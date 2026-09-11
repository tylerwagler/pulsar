#include "pulsar_engine_internal.h"

/* Model-layout queries: per-layer attention compression ratios and routed
 * expert counts. Split out of util.cpp in the C++ port. */

bool pulsar_backend_uses_graph(pulsar_backend backend) {
    return backend == PULSAR_BACKEND_CUDA;
}



/* The CSA2 attention layout table.  pulsar_attn_layout_install is its only
 * writer; the two accessors below are its only readers. */
static pulsar_layer_attn g_pulsar_attn_layout[PULSAR_MAX_LAYER];



/* Attention compression is read from GGUF metadata after validating that it
 * matches the exact layout expected for the loaded model shape. */
uint32_t pulsar_layer_compress_ratio(uint32_t il) {
    if (il >= PULSAR_N_LAYER) pulsar_die("DeepSeek4 layer index is outside the loaded model layout");
    return g_pulsar_attn_layout[il].ratio;
}



const pulsar_layer_attn *pulsar_layer_attn_layout(uint32_t il) {
    if (il >= PULSAR_N_LAYER) pulsar_die("DeepSeek4 layer index is outside the loaded model layout");
    return &g_pulsar_attn_layout[il];
}



/* Physically-present routed-expert count for a layer. For an un-pruned model
 * (or any layer whose keep_count was not set) this is the full n_expert; for a
 * REAP ds4-compact-v1 model the pruned layers report their dense survivor
 * count. Only the expert *weight* tensors are trimmed to this; the router and
 * bias stay padded to n_expert. */
uint32_t pulsar_layer_n_expert(uint32_t il) {
    if (il >= PULSAR_N_LAYER) pulsar_die("DeepSeek4 layer index is outside the loaded model layout");
    const uint32_t v = g_pulsar_layer_expert_count[il];
    return v ? v : PULSAR_N_EXPERT;
}



/* REAP compact detection: any layer whose physical routed-expert count was
 * trimmed below the architecture's n_expert marks the model as pruned. */
bool pulsar_engine::is_pruned() const {
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t v = g_pulsar_layer_expert_count[il];
        if (v != 0 && v != PULSAR_N_EXPERT) return true;
    }
    return false;
}



/* V4.1 CSA2 (L218): layers 0-1 sliding window only; the encoder's 2-19 share
 * ratio-2 compressed KV from sources 2/8/14; the decoder's 20-39 read one
 * ratio-1 (one latent per token) cache from layer 20.  The 0731 4/128 pattern
 * (CSA/HCA alternating) is gone with that checkpoint. */
static uint32_t expected_layer_compress_ratio(uint32_t il) {
    if (il < 2) return 0;
    return il < 20 ? 2u : 1u;
}



/* The profile's expectation of one source set: the artifact must name exactly
 * these layers, in this order. */
static void expect_source_set(const char *what,
                              const uint32_t *got, uint32_t n_got,
                              const uint32_t *want, uint32_t n_want) {
    if (n_got != n_want) {
        fprintf(stderr, "pulsar: %s names %u layers, %s expects %u\n", what, n_got, PULSAR_MODEL_SHAPE_NAME, n_want);
        exit(1);
    }
    for (uint32_t i = 0; i < n_got; i++) {
        if (got[i] != want[i]) {
            fprintf(stderr, "pulsar: %s[%u] is layer %u, %s expects %u\n", what, i, got[i], PULSAR_MODEL_SHAPE_NAME, want[i]);
            exit(1);
        }
        if (i > 0 && got[i] <= got[i - 1]) {
            fprintf(stderr, "pulsar: %s is not ascending at [%u]\n", what, i);
            exit(1);
        }
        if (got[i] >= PULSAR_N_LAYER) {
            fprintf(stderr, "pulsar: %s[%u] = %u is outside the backbone\n", what, i, got[i]);
            exit(1);
        }
    }
}



/* The most recent source at or below `il`, or PULSAR_NO_LAYER.  This is the
 * reference's rule: a layer reads whatever the last source published. */
static uint32_t latest_source(uint32_t il, const uint32_t *sources, uint32_t n) {
    uint32_t s = PULSAR_NO_LAYER;
    for (uint32_t i = 0; i < n && sources[i] <= il; i++) s = sources[i];
    return s;
}



void pulsar_attn_layout_install(const uint32_t *ratios,
                                const uint32_t *kv_sources, uint32_t n_kv,
                                const uint32_t *index_sources, uint32_t n_index,
                                int32_t candidate_source) {
    const pulsar_shape *sh = &g_pulsar_shape;
    expect_source_set("deepseek4.attention.kv_source_layers", kv_sources, n_kv,
                      sh->kv_source_layer, sh->n_kv_source);
    expect_source_set("deepseek4.attention.index_source_layers", index_sources, n_index,
                      sh->index_source_layer, sh->n_index_source);
    if (candidate_source != sh->candidate_source_layer) {
        fprintf(stderr, "pulsar: deepseek4.attention.candidate_source_layer is %d, %s expects %d\n",
                candidate_source, PULSAR_MODEL_SHAPE_NAME, sh->candidate_source_layer);
        exit(1);
    }

    memset(g_pulsar_attn_layout, 0, sizeof(g_pulsar_attn_layout));
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        pulsar_layer_attn *a = &g_pulsar_attn_layout[il];
        const uint32_t expected = expected_layer_compress_ratio(il);
        if (ratios[il] != expected) {
            fprintf(stderr,
                    "pulsar: unexpected DeepSeek4 compression ratio at layer %u for %s: got %u, expected %u\n",
                    il, PULSAR_MODEL_SHAPE_NAME, ratios[il], expected);
            exit(1);
        }
        a->ratio = ratios[il];
        a->kv_source = latest_source(il, kv_sources, n_kv);
        a->index_source = latest_source(il, index_sources, n_index);
        const bool is_kv = a->kv_source == il;
        const bool is_index = a->index_source == il;

        if (a->ratio == 0) {
            /* A window-only layer must not be a source and must not sit past
             * one: the reference slices the shared cache with the reader's own
             * ratio, so a ratio-0 layer after a source has no meaning. */
            if (is_kv || is_index || a->kv_source != PULSAR_NO_LAYER || a->index_source != PULSAR_NO_LAYER) {
                fprintf(stderr, "pulsar: layer %u has compress ratio 0 but lies inside a CSA2 source span\n", il);
                exit(1);
            }
            a->mode = PULSAR_ATTN_WINDOW;
            continue;
        }
        if (a->kv_source == PULSAR_NO_LAYER || a->index_source == PULSAR_NO_LAYER) {
            fprintf(stderr, "pulsar: layer %u compresses (ratio %u) but no kv/index source precedes it\n", il, a->ratio);
            exit(1);
        }
        if (ratios[a->kv_source] != a->ratio || ratios[a->index_source] != a->ratio) {
            fprintf(stderr, "pulsar: layer %u (ratio %u) reads sources %u/%u of ratio %u/%u\n",
                    il, a->ratio, a->kv_source, a->index_source, ratios[a->kv_source], ratios[a->index_source]);
            exit(1);
        }
        if (is_kv && !is_index) {
            /* Its attention would select rows with a top-k computed over a
             * different cache; the reference never ships such a layer. */
            fprintf(stderr, "pulsar: kv source layer %u is not an index source\n", il);
            exit(1);
        }
        a->mode = is_kv ? PULSAR_ATTN_FULL : is_index ? PULSAR_ATTN_REINDEX : PULSAR_ATTN_REUSE;
        a->candidate_source = candidate_source >= 0 && (uint32_t)candidate_source == il;
        a->uses_candidates = is_index && candidate_source >= 0 && (uint32_t)candidate_source < il;
        if (a->candidate_source && !is_index) {
            fprintf(stderr, "pulsar: candidate source layer %u runs no indexer\n", il);
            exit(1);
        }
        if (a->uses_candidates && g_pulsar_attn_layout[candidate_source].kv_source != a->kv_source) {
            /* The mask is over the candidate source's compressed positions; it
             * only means something to an indexer scoring the same cache. */
            fprintf(stderr, "pulsar: layer %u would apply layer %d's candidate mask to a different KV cache\n",
                    il, candidate_source);
            exit(1);
        }
    }
}

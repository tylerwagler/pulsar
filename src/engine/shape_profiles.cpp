#include "pulsar_engine_internal.h"



/* DeepSeek-V4-Flash (0731): 43 layers, hidden 4096, 256 routed experts of
 * intermediate 2048, q_lora 1024, a 64-head indexer, 3 hash-routed layers,
 * rms eps 1e-6.  CSA/HCA alternate 4/128 above layer 1, and EVERY compressed
 * layer owns its own compressor and indexer -- so its kv and index source sets
 * are the whole compressed backbone.  That is the degenerate case of the CSA2
 * table: pulsar_attn_layout_install resolves each layer's source to itself,
 * mode FULL, with no candidate pool and no ratio-0 layer inside a source span.
 *
 * Geometry here is 0731's as it shipped on dev; do not "tidy" these numbers.
 * The per-model constants that used to hide inside shared macros are named per
 * profile now (PULSAR_V4_RMS_EPS vs PULSAR_V41_RMS_EPS) -- see
 * plans/96-SPIKE0-results.md S2 and plans/96-two-profiles-one-engine.md. */
const pulsar_shape PULSAR_SHAPE_V4 = {
    .name = "DeepSeek V4 Flash",
    .variant = PULSAR_VARIANT_V4,
    .n_layer = 43,
    .n_embd = 4096,
    .n_vocab = 129280,
    .n_head = 64,
    .n_head_kv = 1,
    .n_head_dim = 512,
    .n_value_dim = 512,
    .n_rot = 64,
    .n_out_group = 8,
    .n_lora_q = 1024,
    .n_lora_o = 1024,
    .n_expert = 256,
    .n_expert_used = 6,
    .n_expert_shared = 1,
    .n_hash_layer = 3,
    /* The 0731 drafter routes with the target's expert count and top-k. */
    .n_dspark_expert = 256,
    .n_dspark_expert_used = 6,
    .n_ff_exp = 2048,
    .n_swa = 128,
    .n_indexer_head = 64,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 512,
    /* Every compressed layer (2..42) owns its own compressor. */
    .n_kv_source = 41,
    .kv_source_layer = {
        2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41,
        42,
    },
    .n_index_source = 21,
    /* Only the ratio-4 (CSA) layers carry an indexer.  The ratio-128 (HCA)
     * layers have a compressor and NO indexer at all -- verified in the real
     * artifact, where blk.2.* carries indexer + indexer_compressor tensors and
     * blk.3.* carries neither.  So HCA publishes compressed KV and no top-k,
     * which the layout table cannot express yet: pulsar_attn_layout_install
     * refuses such a layer by name rather than mis-moding it as an indexed
     * FULL.  The mode and its consumers land together -- see
     * plans/96-two-profiles-one-engine.md s9. */
    .index_source_layer = {
        2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
        22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
        42,
    },
    .candidate_source_layer = -1,   /* no hierarchical candidate pool */
    .candidate_topk_blocks = 0,
    .candidate_block_size = 0,
    .n_hc = 4,
    .n_hc_sinkhorn_iter = 20,
    .compressor_ape = true,           /* both compressors carry an ape */
    .indexer_own_compressor = true,   /* the indexer compresses its own index key */
    .hc_head_mix = true,              /* the head computes its own HC coefficients */
    .kv_row_style = PULSAR_KV_ROWS_UNIFIED,
    .rms_eps = PULSAR_V4_RMS_EPS,
    .hc_eps = PULSAR_DEFAULT_HC_EPS,
    .expert_weight_scale = 1.5f,
    .swiglu_clamp_exp = PULSAR_DEFAULT_SWIGLU_CLAMP_EXP,
    .rope_freq_base = PULSAR_DEFAULT_ROPE_FREQ_BASE,
    .rope_scale_factor = PULSAR_DEFAULT_ROPE_SCALE_FACTOR,
    .rope_yarn_beta_fast = PULSAR_DEFAULT_ROPE_YARN_BETA_FAST,
    .rope_yarn_beta_slow = PULSAR_DEFAULT_ROPE_YARN_BETA_SLOW,
    .compress_rope_freq_base = PULSAR_DEFAULT_COMPRESS_ROPE_FREQ_BASE,
    .rope_orig_ctx = PULSAR_DEFAULT_ROPE_ORIG_CTX,
};




/* DeepSeek-V4.1-Flash (L218): 40 layers = 20-layer causal encoder + 20-layer
 * decoder, hidden 5120, 384 routed experts of intermediate 2304, q_lora 1280,
 * a 32-head indexer, no hash-routed layers, rms eps 1e-20. */
const pulsar_shape PULSAR_SHAPE_V41 = {
    .name = "DeepSeek V4.1 Flash",
    .variant = PULSAR_VARIANT_V41,
    .n_layer = 40,
    .n_embd = 5120,
    .n_vocab = 129280,
    .n_head = 64,
    .n_head_kv = 1,
    .n_head_dim = 512,
    .n_value_dim = 512,
    .n_rot = 64,
    .n_out_group = 8,
    .n_lora_q = 1280,
    .n_lora_o = 1024,
    .n_expert = 384,
    .n_expert_used = 6,
    .n_expert_shared = 1,
    .n_hash_layer = 0,          /* V4.1 routes every layer through gate + bias */
    .n_dspark_expert = 128,
    .n_dspark_expert_used = 3,
    .n_ff_exp = 2304,
    .n_swa = 128,
    .n_indexer_head = 32,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 512,
    .n_kv_source = 4,
    .kv_source_layer = { 2, 8, 14, 20 },
    .n_index_source = 8,
    .index_source_layer = { 2, 8, 14, 20, 24, 28, 32, 36 },
    .candidate_source_layer = 20,
    .candidate_topk_blocks = 2048,
    .candidate_block_size = 8,
    .n_hc = 4,
    .n_hc_sinkhorn_iter = 20,
    .compressor_ape = false,          /* plain projections */
    .indexer_own_compressor = false,  /* the index key projects from the kv source's latent */
    .hc_head_mix = false,             /* collapses with the carried pre */
    .kv_row_style = PULSAR_KV_ROWS_CSA2,
    .rms_eps = PULSAR_V41_RMS_EPS,
    .hc_eps = PULSAR_DEFAULT_HC_EPS,
    .expert_weight_scale = 1.5f,
    .swiglu_clamp_exp = PULSAR_DEFAULT_SWIGLU_CLAMP_EXP,
    .rope_freq_base = PULSAR_DEFAULT_ROPE_FREQ_BASE,
    .rope_scale_factor = PULSAR_DEFAULT_ROPE_SCALE_FACTOR,
    .rope_yarn_beta_fast = PULSAR_DEFAULT_ROPE_YARN_BETA_FAST,
    .rope_yarn_beta_slow = PULSAR_DEFAULT_ROPE_YARN_BETA_SLOW,
    .compress_rope_freq_base = PULSAR_DEFAULT_COMPRESS_ROPE_FREQ_BASE,
    .rope_orig_ctx = PULSAR_DEFAULT_ROPE_ORIG_CTX,
};






/* Resolved at load by pulsar_select_shape_from_metadata, which either sets this
 * or exits; the initialiser is only a pre-selection placeholder.
 *
 * History worth keeping: a second profile was removed 2026-08-15 because every
 * width-dependent kernel needed a second arm -- and one of them (the generic
 * HC-norm) could not emit E4M3 at all and returned well-formed garbage.  The
 * two profiles carried now are handled the other way round: the per-layer
 * differences are read from the derived layout table, not from a second arm per
 * kernel.  That failure mode is exactly what the table and the battery guard. */
pulsar_shape g_pulsar_shape = PULSAR_SHAPE_V41;



uint32_t g_pulsar_layer_expert_count[PULSAR_MAX_LAYER] = {0};


int g_pulsar_lock_fd = -1;


#include "pulsar_engine_internal.h"



/* DeepSeek-V4.1-Flash (L218): 40 layers = 20-layer causal encoder + 20-layer
 * decoder, hidden 5120, 384 routed experts of intermediate 2304, q_lora 1280,
 * a 32-head indexer, no hash-routed layers, rms eps 1e-20.  Every V4-Flash
 * (0731) number this struct used to carry is gone with the checkpoint the
 * engine no longer serves; a 0731 artifact refuses at load on n_layer. */
const pulsar_shape PULSAR_SHAPE_FLASH = {
    .name = "DeepSeek V4.1 Flash",
    .variant = PULSAR_VARIANT_FLASH,
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
    .rms_eps = PULSAR_DEFAULT_RMS_EPS,
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






/* The only shape this engine serves.  DeepSeek V4 Pro was removed 2026-08-15:
 * we can only ever serve Flash, and carrying a second profile meant every
 * width-dependent kernel needed a second arm -- one of which (the generic
 * HC-norm) could not emit E4M3 at all and returned well-formed garbage. */
pulsar_shape g_pulsar_shape = PULSAR_SHAPE_FLASH;



uint32_t g_pulsar_layer_expert_count[PULSAR_MAX_LAYER] = {0};


int g_pulsar_lock_fd = -1;


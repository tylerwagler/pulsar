/* family_qwen.h -- the Qwen4-exp family (Qwen3.8-Flash-Next): shape, container
 * contract, weights, session state, and the per-layer op interface that the
 * L251 streams implement (S1 owns this header; S2 GDN, S3 QSA, S4 MoE/GR/PLE/
 * head fill the op table; S5 the tokenizer/renderer; S6 the container).
 *
 * THE FORWARD (HF transformers `qwen4_exp`, modeling_qwen4_exp.py), which the
 * family's step driver (family_qwen.cpp) walks in exactly this order:
 *
 *   streams = embed_tokens(ids) repeated into n_hc (4) streams       [op embed, S4]
 *   for il in plan:
 *     if il == ple_layer (1):  streams += PLE(streams, ngram rows)      [op ple, S4]
 *     x  = GR_attn.read(streams)          (hc_norm, low-rank read gate) [op gr_read  ATTN, S4]
 *     y  = GDN(x) | QSA(x)                by plan kind                  [op gdn S2 | op qsa S3]
 *     streams = streams + y (x) inj_attn  (2 sigmoid(W_inj norm/4))     [op gr_write ATTN, S4]
 *     x  = GR_mlp.read(streams)                                         [op gr_read  MLP, S4]
 *     y  = MoE(x)  (softmax top-10 of 512 + sigmoid-gated shared)       [op moe, S4]
 *     streams = streams + y (x) inj_mlp                                 [op gr_write MLP, S4]
 *   logits = lm_head(mixer.read(streams))   (no final RMSNorm: the mixer's
 *                                            hc_norm is the last norm)   [op head, S4]
 *
 * Every op reads and writes the step's activation slots (pulsar_qwen_step).
 * A NULL op in pulsar_qwen_ops is "not implemented yet": the driver refuses
 * the step by naming the FIRST missing op in forward order and the stream
 * that owns it -- no stub runs, nothing falls back (rules 1, 9). */
#ifndef PULSAR_FAMILY_QWEN_H
#define PULSAR_FAMILY_QWEN_H

#include <stdint.h>
#include <stdbool.h>

#include "family.h"

/* ---- 1. Shape -------------------------------------------------------------
 *
 * The known geometry.  Load reads every field from the artifact (section 2)
 * and refuses any value that differs from PULSAR_QWEN_SHAPE_FLASH_NEXT: the
 * kernels are specialised to these numbers exactly as DeepSeek's are to
 * theirs, and a second geometry is a second profile, added here on purpose. */
typedef struct {
    const char *name;
    uint32_t n_layer;          ///< num_hidden_layers (48)
    uint32_t n_embd;           ///< hidden_size (2560)
    uint32_t n_vocab;          ///< vocab_size (248320): the embed rows AND the logits width (lm_head is untied)
    uint32_t n_hc;             ///< hc_count: residual streams (4)
    uint32_t n_hc_lowrank;     ///< hc_lowrank: the gated-residual read gate's rank (320)
    /* Gated DeltaNet (linear_attention layers) */
    uint32_t gdn_n_k_head;     ///< linear_num_key_heads (16)
    uint32_t gdn_n_v_head;     ///< linear_num_value_heads (48)
    uint32_t gdn_k_dim;        ///< linear_key_head_dim (128)
    uint32_t gdn_v_dim;        ///< linear_value_head_dim (128)
    uint32_t gdn_conv_kernel;  ///< linear_conv_kernel_dim (4)
    /* QSA (full_attention layers) */
    uint32_t n_head;           ///< num_attention_heads (24)
    uint32_t n_head_kv;        ///< num_key_value_heads (2)
    uint32_t head_dim;         ///< head_dim (256)
    uint32_t n_rot;            ///< head_dim * partial_rotary_factor (64)
    uint32_t mrope_section[3]; ///< rope_parameters.mrope_section (11, 11, 10), interleaved
    float    rope_theta;       ///< rope_parameters.rope_theta (1e7)
    uint32_t idx_n_head;       ///< indexer_n_heads (4)
    uint32_t idx_n_head_kv;    ///< indexer_kv_heads (1)
    uint32_t idx_head_dim;     ///< indexer_head_dim (128)
    uint32_t idx_block;        ///< indexer_compress_ratio: tokens pooled per key block (4)
    uint32_t idx_budget;       ///< indexer_budget in tokens (2048 = 512 blocks + the partial tail)
    /* MoE */
    uint32_t n_expert;         ///< num_experts (512)
    uint32_t n_expert_used;    ///< num_experts_per_tok (10)
    uint32_t n_ff_exp;         ///< moe_intermediate_size (640)
    uint32_t n_ff_shexp;       ///< shared_expert_intermediate_size (640)
    /* PLE / n-gram (one layer) */
    uint32_t ple_layer;        ///< the 0-based layer carrying PLE: ple_layer_ids is 1-BASED in the config ([2] -> 1)
    uint32_t ngram_size;       ///< ngram_size (3): bigrams + trigrams; also the PLE conv dilation
    uint32_t ngram_heads;      ///< heads_per_ngram (8) per n-gram order -> 16 tables
    uint32_t ple_embed_dim;    ///< ple_embed_dim (2560) = 16 heads x 160
    uint32_t ple_conv_kernel;  ///< ple_conv_kernel_size (4)
    uint64_t ngram_vocab_base; ///< ngram_vocab_size_base (20,000,000): each table is a prime just above it
    uint32_t ngram_split_parts;///< split_ngram_parts (128): HF's shard count per table
    /* misc */
    float    rms_eps;          ///< rms_norm_eps (1e-6)
    uint32_t max_position;     ///< max_position_embeddings (262144)
    uint32_t eos_id;           ///< eos_token_id (248044), also the PLE context reset token
    uint32_t n_mtp_layer;      ///< mtp_num_hidden_layers (1)
    /* L266 step 7: this rank of a tensor-parallel group (0 of 0 / 1 = one GPU).  Under TP the four head counts
     * above are THIS RANK's (gdn_n_k_head 8, gdn_n_v_head 24, n_head 12, n_head_kv 1): the load binds the
     * container against the whole model, then pulsar_qwen_tp_load halves them, so every derived width,
     * state size and scratch layout below is the rank's.  n_vocab and n_expert stay whole. */
    uint32_t tp_rank, tp_ranks;
} pulsar_qwen_shape;

extern const pulsar_qwen_shape PULSAR_QWEN_SHAPE_FLASH_NEXT;

/** The loaded Qwen shape.  Valid after the Qwen family's load; nothing reads
 * it on a DeepSeek engine. */
extern pulsar_qwen_shape g_qwen_shape;

/* Derived widths, one definition each. */
static inline uint32_t pulsar_qwen_gdn_qk_dim(const pulsar_qwen_shape *s) { return s->gdn_n_k_head * s->gdn_k_dim; }   /* 2048 */
static inline uint32_t pulsar_qwen_gdn_v_total(const pulsar_qwen_shape *s) { return s->gdn_n_v_head * s->gdn_v_dim; } /* 6144 */
/** The GDN causal conv runs over q|k|v: 2 x 2048 + 6144 = 10240 channels. */
static inline uint32_t pulsar_qwen_gdn_conv_dim(const pulsar_qwen_shape *s) {
    return 2u * pulsar_qwen_gdn_qk_dim(s) + pulsar_qwen_gdn_v_total(s);
}
static inline uint32_t pulsar_qwen_hc_dim(const pulsar_qwen_shape *s) { return s->n_hc * s->n_embd; }                 /* 10240 */
/** The tensor-parallel degree (1 = one GPU). */
static inline uint32_t pulsar_qwen_tp(const pulsar_qwen_shape *s) { return s->tp_ranks > 1 ? s->tp_ranks : 1u; }
/** The QSA projection widths of this rank's heads: q + gate per query head, k / v per KV head, the o_proj input. */
static inline uint32_t pulsar_qwen_qsa_q_in(const pulsar_qwen_shape *s) { return s->n_head * 2u * s->head_dim; }
static inline uint32_t pulsar_qwen_qsa_kv_in(const pulsar_qwen_shape *s) { return s->n_head_kv * s->head_dim; }
static inline uint32_t pulsar_qwen_qsa_out_dim(const pulsar_qwen_shape *s) { return s->n_head * s->head_dim; }
/** Tokens of PLE conv history per sequence: (kernel - 1) * dilation = 9. */
static inline uint32_t pulsar_qwen_ple_conv_state_len(const pulsar_qwen_shape *s) {
    return (s->ple_conv_kernel - 1u) * s->ngram_size;
}

/* ---- 2. Container contract (S6 emits, this family reads) ------------------
 *
 * `general.architecture` = "qwen4_exp" selects this family.  Every other
 * config key is the HF text_config key VERBATIM under the "qwen4_exp."
 * prefix (nested dicts flattened with '.'), so the builder copies keys
 * mechanically and a reader can grep the HF config for any of them:
 *   qwen4_exp.num_hidden_layers (u32), qwen4_exp.hidden_size (u32),
 *   qwen4_exp.vocab_size (u32), qwen4_exp.hc_count, qwen4_exp.hc_lowrank,
 *   qwen4_exp.layer_types (array of string: "linear_attention"|"full_attention")
 *   -- THE layer plan's source --, qwen4_exp.linear_num_key_heads,
 *   qwen4_exp.linear_num_value_heads, qwen4_exp.linear_key_head_dim,
 *   qwen4_exp.linear_value_head_dim, qwen4_exp.linear_conv_kernel_dim,
 *   qwen4_exp.num_attention_heads, qwen4_exp.num_key_value_heads,
 *   qwen4_exp.head_dim, qwen4_exp.partial_rotary_factor (f32),
 *   qwen4_exp.rope_parameters.mrope_section (array u32),
 *   qwen4_exp.rope_parameters.rope_theta (f32), qwen4_exp.indexer_n_heads,
 *   qwen4_exp.indexer_kv_heads, qwen4_exp.indexer_head_dim,
 *   qwen4_exp.indexer_compress_ratio, qwen4_exp.indexer_budget,
 *   qwen4_exp.num_experts, qwen4_exp.num_experts_per_tok,
 *   qwen4_exp.moe_intermediate_size, qwen4_exp.shared_expert_intermediate_size,
 *   qwen4_exp.ple_layer_ids (array u32, 1-based as in HF), qwen4_exp.ngram_size,
 *   qwen4_exp.heads_per_ngram, qwen4_exp.ple_embed_dim,
 *   qwen4_exp.ple_conv_kernel_size, qwen4_exp.ngram_vocab_size_base (u64),
 *   qwen4_exp.split_ngram_parts, qwen4_exp.rms_norm_eps (f32),
 *   qwen4_exp.max_position_embeddings, qwen4_exp.eos_token_id,
 *   qwen4_exp.mtp_num_hidden_layers.
 * The three int64 BUFFERS HF keeps under ple.ple_embedding are config, not
 * weights, and ride as u64 arrays: qwen4_exp.ple_layer_multipliers
 * [ngram_size] (the hash's odd multipliers), qwen4_exp.ple_ngram_heads_vocab_sizes
 * [(ngram_size - 1) * heads_per_ngram] (each table's prime size) and
 * qwen4_exp.ple_ngram_heads_offsets [same] (each table's first row).
 * Integer keys are u32 (u64 allowed everywhere); floats f32 or f64.
 *
 * TENSOR NAMES: each container entry's `gguf_name` (the name the engine binds)
 * is the HF checkpoint name VERBATIM, e.g.
 * "model.language_model.layers.3.self_attn.q_proj.weight"; the table lives in
 * family_qwen.cpp (qwen_bind_*), and dims are checked in GGUF ne order
 * (innermost first, i.e. the HF shape reversed).  The expert stacks stay fused
 * as in HF (gate_up_proj [512,1280,2560], down_proj [512,2560,640]).  The PLE
 * n-gram tables (ple.ple_embedding.ngram_embedding.shard_*) are NOT container
 * tensors: they are a disk-table side artifact read by the L242 row path (S4).
 * The binder checks presence and logical dims; each op owner adds the type
 * admission for its tensors (EXL3 dense, MXFP8, BF16 ...) when its op lands,
 * so a tensor in a format no arm reads is refused at load, not at first use. */
#define PULSAR_QWEN_ARCH "qwen4_exp"

/* ---- 3. Weights ------------------------------------------------------------ */

/** One gated-residual (HC) unit: attn_hyper_connection / mlp_hyper_connection
 * per layer, and the top-level hyper_connection_mixer (no inject). */
typedef struct {
    pulsar_tensor *hc_norm;        ///< [n_hc * n_embd] grouped RMSNorm weight
    pulsar_tensor *mix_down;       ///< input_mix_weight_down [n_hc*n_embd -> hc_lowrank]
    pulsar_tensor *mix_up;         ///< input_mix_weight_up   [hc_lowrank -> n_hc*n_embd]
    pulsar_tensor *inject;         ///< block_inject_weight   [n_hc*n_embd -> n_hc]; NULL on the mixer
} pulsar_qwen_gr_weights;

typedef struct {
    pulsar_qwen_gr_weights gr_attn;
    pulsar_qwen_gr_weights gr_mlp;
    /* GDN (PULSAR_LAYER_QWEN_GDN) */
    pulsar_tensor *gdn_in_qkv;     ///< in_proj_qkv [n_embd -> conv_dim]
    pulsar_tensor *gdn_in_z;       ///< in_proj_z   [n_embd -> v_total] (output gate)
    pulsar_tensor *gdn_in_a;       ///< in_proj_a   [n_embd -> n_v_head]
    pulsar_tensor *gdn_in_b;       ///< in_proj_b   [n_embd -> n_v_head]
    pulsar_tensor *gdn_conv;       ///< conv1d.weight [conv_dim, 1, kernel]
    pulsar_tensor *gdn_a_log;      ///< A_log [n_v_head]
    pulsar_tensor *gdn_dt_bias;    ///< dt_bias [n_v_head]
    pulsar_tensor *gdn_norm;       ///< gated RMSNorm weight [v_dim]
    pulsar_tensor *gdn_out;        ///< out_proj [v_total -> n_embd]
    /* QSA (PULSAR_LAYER_QWEN_QSA) */
    pulsar_tensor *attn_q;         ///< q_proj [n_embd -> 2 * n_head * head_dim] (query | sigmoid gate)
    pulsar_tensor *attn_k;         ///< k_proj [n_embd -> n_head_kv * head_dim]
    pulsar_tensor *attn_v;         ///< v_proj [n_embd -> n_head_kv * head_dim]
    pulsar_tensor *attn_q_norm;    ///< q_norm [head_dim]
    pulsar_tensor *attn_k_norm;    ///< k_norm [head_dim]
    pulsar_tensor *attn_o;         ///< o_proj [n_head * head_dim -> n_embd]
    pulsar_tensor *idx_qk;         ///< indexer.index_qk_proj [n_embd -> (idx_n_head + idx_n_head_kv) * idx_head_dim]
    pulsar_tensor *idx_q_norm;     ///< indexer.q_layernorm [idx_head_dim]
    pulsar_tensor *idx_k_norm;     ///< indexer.k_layernorm [idx_head_dim]
    /* MoE (every layer) */
    pulsar_tensor *moe_router;     ///< mlp.gate.weight [n_embd -> n_expert]
    pulsar_tensor *moe_gate_up;    ///< mlp.experts.gate_up_proj [n_expert][n_embd -> 2 * n_ff_exp]
    pulsar_tensor *moe_down;       ///< mlp.experts.down_proj    [n_expert][n_ff_exp -> n_embd]
    /** L251 MTP: the SPLIT expert form -- mlp.experts.gate_proj / up_proj [n_expert][n_embd -> n_ff_exp],
     *  each slice with its own suh (the MTP layer, turboderp's EXL3).  Set exactly when moe_gate_up is NULL. */
    pulsar_tensor *moe_gate, *moe_up;
    pulsar_tensor *sh_gate;        ///< mlp.shared_expert.gate_proj [n_embd -> n_ff_shexp]
    pulsar_tensor *sh_up;          ///< mlp.shared_expert.up_proj   [n_embd -> n_ff_shexp]
    pulsar_tensor *sh_down;        ///< mlp.shared_expert.down_proj [n_ff_shexp -> n_embd]
    pulsar_tensor *sh_gate_scalar; ///< mlp.shared_expert_gate [n_embd -> 1] (sigmoid)
    /* PLE (ple_layer only; NULL elsewhere) */
    pulsar_tensor *ple_key;        ///< ple.key_proj   [ple_embed_dim -> n_hc*n_embd]
    pulsar_tensor *ple_value;      ///< ple.value_proj [ple_embed_dim -> n_embd]
    pulsar_tensor *ple_norm_key;   ///< ple.norm_key   [n_hc*n_embd]
    pulsar_tensor *ple_norm_query; ///< ple.norm_query [n_hc*n_embd]
    pulsar_tensor *ple_norm_conv;  ///< ple.norm_conv  [n_hc*n_embd]
    pulsar_tensor *ple_conv;       ///< ple.conv1d.weight [n_hc*n_embd, 1, ple_conv_kernel] (dilation ngram_size)
} pulsar_qwen_layer_weights;

#define PULSAR_QWEN_MAX_NGRAM        4u   ///< n-gram orders the PLE tables may carry (Flash-Next: 3)
#define PULSAR_QWEN_MAX_NGRAM_HEADS 32u   ///< n-gram tables (Flash-Next: 2 orders x 8 heads = 16)

/** L251 MTP: the one multi-token-prediction layer's head-side tensors (the sidecar shard; the spec is
 *  l251/docs/MTP-SPEC-2026-09-29.md).  The layer itself is bound as a QSA layer in
 *  pulsar_qwen_weights::layer[n_layer] (split experts, no PLE), so the trunk's ops run it unchanged. */
typedef struct {
    bool present;                  ///< the artifact carries mtp.* (all of it, or the load refuses)
    pulsar_tensor *norm_embd;      ///< mtp.pre_fc_norm_embedding.weight [n_embd] ((1 + w) RMSNorm)
    pulsar_tensor *norm_hidden;    ///< mtp.pre_fc_norm_hidden.weight [n_hc * n_embd] (ONE RMS over all of it)
    pulsar_tensor *fc_embd;        ///< mtp.fc_embedding.weight [n_embd -> n_embd]
    pulsar_tensor *fc_hidden;      ///< mtp.fc_hidden.weight [n_embd -> n_embd], applied per stream
    pulsar_qwen_gr_weights mixer;  ///< mtp.hyper_connection_mixer (no inject), then the trunk's lm_head
} pulsar_qwen_mtp_weights;

typedef struct {
    pulsar_tensor *token_embd;     ///< model.language_model.embed_tokens.weight [n_embd x n_vocab]
    pulsar_tensor *output;         ///< lm_head.weight [n_embd -> n_vocab]
    pulsar_qwen_gr_weights mixer;         ///< model.language_model.hyper_connection_mixer
    /** [0, n_layer) the trunk; [n_layer] the MTP layer when mtp.present (a QSA layer, split experts) */
    pulsar_qwen_layer_weights layer[PULSAR_FAMILY_MAX_LAYER];
    pulsar_qwen_mtp_weights mtp;
    /* The PLE hash's config (section 2): read at load, used by the host-side
     * n-gram id hashing (S4). */
    uint64_t ple_multipliers[PULSAR_QWEN_MAX_NGRAM];       ///< [ngram_size]
    uint64_t ple_head_vocab[PULSAR_QWEN_MAX_NGRAM_HEADS];  ///< table size (a prime) per n-gram head
    uint64_t ple_head_offset[PULSAR_QWEN_MAX_NGRAM_HEADS]; ///< first row of each head's table
    /** S4: the PLE row file (the container's pulsar.ple_rows.*), its pread pool
     * and the host gather in flight; opened by pulsar_qwen_s4_load. */
    struct pulsar_qwen_ple_io *ple_io;
    /** L251: the lm_head as MXFP8 (mxfp8_lt), made on the device from `output` at the first head
     *  step (the bf16 head is 1.27 GB read per decode row; this is half); freed at unload. */
    struct pulsar_gpu_tensor *head_mx;
    /** L251 MTP: the DRAFT head -- lm_head rows [0, N) and [eos_id, n_vocab) (the added / special ids,
     *  the stop tokens among them) gathered into one MXFP8 matrix, made at the first MTP head.  The
     *  drafter's argmax runs over these; verification keeps the full head, so the output is unchanged
     *  and only the draft cost moves.  N = PULSAR_QWEN_MTP_DRAFT_VOCAB (default 65536; 0 = the full
     *  head).  draft_ids[i] is row i's token id. */
    struct pulsar_gpu_tensor *draft_head_mx, *draft_ids_dev;
    int32_t *draft_ids;
    uint32_t n_draft;
    /** L266: every tensor's name, type and size, FNV-1a -- the artifact a Qwen segment was written by
     *  (pulsar_ckpt_store::artifact): two quantisations of the model never load each other's KV. */
    uint64_t artifact_digest;
    /** L268: the vision tower (model.visual.*, bf16 through the mapping; vision_qwen.cpp), when the artifact carries
     *  it, and the <|image_pad|> id its image blocks are runs of (-1 = no tower) */
    bool vision_present;
    pulsar_qwen_vision_weights_dev vision;
    int32_t vision_pad_id;
} pulsar_qwen_weights;

/* ---- 4. Session state -------------------------------------------------------
 *
 * Per bank (sequence) and per layer, sized from the shape and the session's
 * ctx_size (tokens of KV per bank).  Tyler's target is 1.5-2M tokens of KV on
 * one Spark beside ~66 GB of weights: FP8 KV (below) is ~13 KB/token across
 * the 12 QSA layers, and each bank carries a FIXED ~118 MB of recurrent state
 * (36 GDN layers x 3.1 MB fp32 + conv + PLE), whatever its length.  The KV
 * slabs are demand-paged (cudaMallocManaged), like DeepSeek's compressed
 * slabs: reserve the capacity, pay for what a bank touches.
 *
 * FORMATS -- each owned by the stream that PRODUCES the bytes (rule 3).  The
 * sizes below are the allocation authority; an owner who changes a format
 * changes its size HERE, in the same commit, and nowhere else. */

/** S2: the DeltaNet recurrent state, fp32 [n_v_head][k_dim][v_dim] per layer
 * per bank (mamba_ssm_dtype float32 -- the source numerics; narrowing it is a
 * graded change, rule 7). */
static inline uint64_t pulsar_qwen_gdn_state_bytes(const pulsar_qwen_shape *s) {
    return (uint64_t)s->gdn_n_v_head * s->gdn_k_dim * s->gdn_v_dim * sizeof(float);
}
/** S2: the causal-conv history, f32 [conv_kernel - 1][conv_dim] per layer per bank. */
static inline uint64_t pulsar_qwen_gdn_conv_bytes(const pulsar_qwen_shape *s) {
    return (uint64_t)(s->gdn_conv_kernel - 1u) * pulsar_qwen_gdn_conv_dim(s) * sizeof(float);
}
/** S3: one token's K and V for one QSA layer: n_head_kv x head_dim E4M3 each
 * (K post-norm post-RoPE, V raw), then one E8M0 byte per 32 elements -- 8 per
 * head, 2 x 2 x 8 = 32 B.  2 x 2 x 256 + 32 = **1056 B**, the kernel's
 * PULSAR_QSA_KV_TOKEN_BYTES.  (This said 1040 -- it charged a f32 per (head,
 * K|V) instead of head_dim/32 scale bytes; the kernel refused every sequence's
 * cache as "short" until it was corrected, 2026-09-27.) */
static inline uint64_t pulsar_qwen_kv_row_bytes(const pulsar_qwen_shape *s) {
    return 2ull * s->n_head_kv * s->head_dim + 2ull * s->n_head_kv * (s->head_dim / 32ull);
}
/** S3: the per-bank QSA cache capacity for a requested context of `ctx` tokens.
 *  The indexer pools idx_block tokens into ONE key block (the kernel's name for
 *  the same fact is PULSAR_QSA_BLOCK), so the cache -- and the `cap` handed to
 *  the kernel -- must be a whole number of blocks; the partial tail block is
 *  never addressed.  A session is sized in raw tokens (a depth+margin is not
 *  block-aligned), so this rounding is what keeps the ring legal: rounding HERE,
 *  once, keeps qwen_state_alloc's two allocations, the per-bank views, the op's
 *  scratch reservation and `seqs[b].cap` all derived from the same number. */
static inline uint32_t pulsar_qwen_qsa_cap(const pulsar_qwen_shape *s, uint32_t ctx) {
    return (ctx + s->idx_block - 1u) / s->idx_block * s->idx_block;
}
/** S3: one pooled indexer key block (idx_block tokens, post-norm, post-RoPE),
 * bf16 [idx_n_head_kv][idx_head_dim] = 256 B per 4 tokens. */
static inline uint64_t pulsar_qwen_index_row_bytes(const pulsar_qwen_shape *s) {
    return (uint64_t)s->idx_n_head_kv * s->idx_head_dim * 2u;
}
/** S3: the partial tail block's raw (pre-pool) index keys, f32 [idx_block][idx_n_head_kv * idx_head_dim]. */
static inline uint64_t pulsar_qwen_index_tail_bytes(const pulsar_qwen_shape *s) {
    return (uint64_t)s->idx_block * s->idx_n_head_kv * s->idx_head_dim * sizeof(float);
}
/** S4: the PLE dilated-conv history, f32 [conv_state_len][n_hc * n_embd] per bank. */
static inline uint64_t pulsar_qwen_ple_conv_bytes(const pulsar_qwen_shape *s) {
    return (uint64_t)pulsar_qwen_ple_conv_state_len(s) * pulsar_qwen_hc_dim(s) * sizeof(float);
}
/** Activation slot formats (the ops' shared contract; each set by the stream
 * that PRODUCES the slot).
 *   streams  bf16 [rows][n_hc][n_embd] -- S4 (embed, PLE, GR write): the
 *            source's residual dtype and the DeepSeek lane's (pulsar_hc_t):
 *            bf16 storage, f32 math, one rounding per write.
 *   x        bf16 [rows][n_embd] -- S4 (GR read): the block input rounded to
 *            bf16 as the source rounds `mixed_input`; the bf16-weight readers
 *            (router, shared-expert gate) read it here.  ITS E4M3 ENCODING is
 *            emitted by the same producer into the activation cache slot of `x`
 *            (pulsar_gpu_mxfp8_act_cache_e4m3_slot, armed + noted, f32 plane
 *            noted skipped): every A8 consumer -- EXL3 dense / MXFP8 Linears of
 *            GDN, QSA, the MoE -- reads it with pulsar_gpu_mxfp8_act_cache_get_e4m3
 *            and never encodes x itself (rule 3).
 *   y        f32 [rows][n_embd] -- written by GDN / QSA / MoE, read by GR write. */
#define PULSAR_QWEN_STREAM_ELT_SIZE 2u   ///< streams, bf16
#define PULSAR_QWEN_X_ELT_SIZE      2u   ///< x, bf16 (+ the armed E4M3 slot)
#define PULSAR_QWEN_Y_ELT_SIZE      4u   ///< y, f32
/** Rows the head may emit per step (the logits slab): the batched lane's bound. */
#define PULSAR_QWEN_HEAD_ROWS_MAX   16u


/** One layer's persistent state across all banks.  A GDN layer owns gdn_*;
 * a QSA layer owns kv/idx_*; the PLE layer additionally owns ple_conv.
 * Bank b's slice of a pooled tensor is at offset b * <per-bank bytes> (bank-major).
 * L284 #3: the two demand-paged ones are a tensor PER BANK -- on GB10 only a cudaFree returns physical, so
 * a bank's touched pages come back only when its own tensors go (qwen_bank_kv_free).  The arrays are
 * present exactly on a QSA layer; an entry is NULL while its bank is evicted. */
typedef struct {
    pulsar_gpu_tensor *gdn_state;   ///< [n_banks] x pulsar_qwen_gdn_state_bytes
    pulsar_gpu_tensor *gdn_conv;    ///< [n_banks] x pulsar_qwen_gdn_conv_bytes
    pulsar_gpu_tensor **kv;         ///< [n_banks] each [ctx] x pulsar_qwen_kv_row_bytes (managed)
    pulsar_gpu_tensor **idx_keys;   ///< [n_banks] each [ceil(ctx / idx_block)] x pulsar_qwen_index_row_bytes (managed)
    pulsar_gpu_tensor *idx_tail;    ///< [n_banks] x pulsar_qwen_index_tail_bytes
    pulsar_gpu_tensor *ple_conv;    ///< [n_banks] x pulsar_qwen_ple_conv_bytes (ple_layer only)
} pulsar_qwen_layer_state;

/** Which op a scratch arena belongs to (pulsar_qwen_ops::scratch_bytes). */
typedef enum {
    PULSAR_QWEN_OP_EMBED = 0,
    PULSAR_QWEN_OP_PLE,
    PULSAR_QWEN_OP_GR_READ,
    PULSAR_QWEN_OP_GDN,
    PULSAR_QWEN_OP_QSA,
    PULSAR_QWEN_OP_GR_WRITE,
    PULSAR_QWEN_OP_MOE,
    PULSAR_QWEN_OP_HEAD,
    PULSAR_QWEN_OP_COUNT,
} pulsar_qwen_op_id;

/** L251 MTP: the draft depth the verify capture holds (a verify step is at most this + 1 rows). */
#define PULSAR_QWEN_SPEC_DRAFT_MAX 6u
/** L272 P1 S4: the rows one verify step carries, every bank's run together -- the head's row cap, which is
 *  also the widest step whose kernels keep their decode-width arms (each row's logits the bytes a one-token
 *  decode gives).  N banks verify together while N x (K + 1) <= this. */
#define PULSAR_QWEN_SPEC_ROWS PULSAR_QWEN_HEAD_ROWS_MAX

/** L251 MTP: what a verify step keeps so a rejected draft rolls back (qwen_spec_*): the recurrent
 *  state after each row (GDN recurrent + conv, PLE conv), each QSA layer's index stage before the step
 *  and the step's raw index keys (the stage is one open block per bank, and rows past the accepted ones
 *  may have overwritten its slots), and each bank's n-gram context before the step.  L272 P1 S4: a step
 *  carries one RUN per bank (rows grouped by bank, consecutive positions); every per-row state is at the
 *  row's step index (each run's last row's slot unused -- its state is the pool's), every per-run one at
 *  the run's index, and the runs are recorded so a rollback finds its bank's.  Allocated with the MTP
 *  layer; `ord[il]` is layer il's index among the layers of its kind. */
typedef struct {
    pulsar_gpu_tensor *gdn_rec;     ///< [n_gdn][SPEC_ROWS] x pulsar_qwen_gdn_state_bytes
    pulsar_gpu_tensor *gdn_conv;    ///< [n_gdn][SPEC_ROWS] x pulsar_qwen_gdn_conv_bytes
    pulsar_gpu_tensor *ple;         ///< [SPEC_ROWS] x pulsar_qwen_ple_conv_bytes
    pulsar_gpu_tensor *qsa_stage;   ///< [n_qsa][SPEC_ROWS runs] x pulsar_qwen_index_tail_bytes
    pulsar_gpu_tensor *qsa_keys;    ///< [n_qsa][SPEC_ROWS][PULSAR_QSA_IDX_IN] f32, the rows' index projections
    uint8_t ord[PULSAR_FAMILY_MAX_LAYER];
    uint32_t n_gdn, n_qsa;
    /** The last verify step's runs: run k is rows [run_first[k], run_first[k + 1]) of bank run_bank[k] from
     *  position run_pos0[k]; ngram_before[k] is that bank's n-gram context before the step. */
    uint32_t n_runs;
    uint32_t run_first[PULSAR_QWEN_SPEC_ROWS + 1];
    int32_t run_bank[PULSAR_QWEN_SPEC_ROWS];
    int32_t run_pos0[PULSAR_QWEN_SPEC_ROWS];
    int32_t ngram_before[PULSAR_QWEN_SPEC_ROWS][PULSAR_QWEN_MAX_NGRAM];
} pulsar_qwen_spec_capture;

/** A Qwen session's device state. */
typedef struct pulsar_qwen_state {
    uint32_t n_banks;       ///< sequences with their own state (1 = a single-sequence session)
    uint32_t ctx;           ///< KV capacity per bank, tokens
    uint32_t max_rows;      ///< activation rows per step (the prefill chunk)
    pulsar_qwen_layer_state layer[PULSAR_FAMILY_MAX_LAYER];
    /* per-step activation slots (rows <= max_rows) */
    pulsar_gpu_tensor *streams;     ///< [max_rows][n_hc][n_embd] PULSAR_QWEN_STREAM_ELT_SIZE
    pulsar_gpu_tensor *x;           ///< [max_rows][n_embd] block input (GR read -> mixer / MoE)
    pulsar_gpu_tensor *y;           ///< [max_rows][n_embd] block output (mixer / MoE -> GR write)
    pulsar_gpu_tensor *logits;      ///< [PULSAR_QWEN_HEAD_ROWS_MAX][n_vocab] f32
    pulsar_gpu_tensor *row_pos;     ///< [max_rows] i32 positions
    pulsar_gpu_tensor *row_bank;    ///< [max_rows] i32 bank of each row
    pulsar_gpu_tensor *scratch[PULSAR_QWEN_OP_COUNT];  ///< each op's own arena, sized by its scratch_bytes (NULL = none)
    /* L251 MTP (mtp.present): the MTP layer's KV / index state is layer[n_layer]; these are its
     * activation slots.  Each bank's LAST trunk row waits in mtp_pend until its next token exists (the MTP
     * row at position p pairs the trunk stack at p with x_{p+1}). */
    bool mtp;
    pulsar_gpu_tensor *mtp_streams; ///< [max_rows][n_hc][n_embd] bf16, the MTP layer's streams
    pulsar_gpu_tensor *mtp_h;       ///< [max_rows + 1][n_hc][n_embd] bf16, the trunk stacks the MTP rows read
    pulsar_gpu_tensor *mtp_tok;     ///< [max_rows] i32, x_{p+1} per MTP row
    pulsar_gpu_tensor *mtp_ws;      ///< pulsar_qwen_mtp_combine_workspace_bytes
    pulsar_gpu_tensor *mtp_pend;    ///< [n_banks][n_hc][n_embd] bf16, the trunk stack awaiting its next token
    uint32_t *mtp_pend_pos;         ///< [n_banks] the pending row's position; UINT32_MAX = none
    uint64_t mtp_probe_n, mtp_probe_hit;   ///< PULSAR_QWEN_MTP_PROBE counters (qwen_session_eval)
    pulsar_qwen_spec_capture spec;  ///< the verify capture (mtp only)
    /** L272 P1: [n_banks] the MTP layer's index stage after each bank's last TRUE row, saved before a draft
     *  chain writes draft rows into it, and whether the chain has (L272 P1 S4: per bank). */
    pulsar_gpu_tensor *mtp_stage;
    bool *mtp_stage_dirty;
    pulsar_gpu_tensor *run_first_dev;   ///< L272 P1 S4: [SPEC_ROWS + 1] i32, a multi-run step's run offsets (GDN)
    float *spec_logits;             ///< host [SPEC_ROWS][n_vocab]: a round's verify / draft rows (mtp only)
    /* host-side sequence state */
    int32_t *ngram_ctx;     ///< [n_banks][ngram_size - 1] last token ids per bank (PLE hashing; reset at EOS)
    uint32_t *bank_pos;     ///< [n_banks] tokens each bank's state holds (written by qwen_bank_set_pos)
    /** L270: [n_banks] the most tokens each bank has held -- its KV pages up to here are resident (a rewind
     * or a reset frees none of them; only qwen_bank_kv_free does, and it zeroes this). */
    uint32_t *kv_hw;
    /* The bank pool (L251, family_qwen_banks.cpp): the session's host view (checkpoint,
     * logits) describes `live_bank`; every other bank's view waits in its carry. */
    uint32_t live_bank;     ///< the bank sync / eval run on
    /* L266 step 5: the grid checkpoints (kv_state_qwen.cpp; held by pointer because kv_state.h comes
     * after this header).  prefill_pos[b] ends bank b's PREFILL-ONLY history -- the cold prefill's
     * bytes; decode never advances it -- and bounds every checkpoint and resume. */
    struct pulsar_ckpt_store *ckpt;
    uint32_t *prefill_pos;  ///< [n_banks]
    bool *frontier_stale;   ///< [n_banks] a segment load wrote the pools without the lanes
    uint32_t n_trunk_layers;   ///< the plan's layers (the MTP layer's state sits at this index)
    /* L266 step 7: the session's share of the engine's tensor-parallel transport (NULL tp = one GPU): the
     * row all-reduce's stage ticket and the exchange seq, advanced identically on every rank. */
    struct pulsar_tp *tp;
    void *tp_slab_dev, *tp_bulk_dev;
    pulsar_gpu_tensor *tp_ticket;
    uint64_t tp_seq;
    pulsar_gpu_tensor *tp_vocab_own;   ///< the vocab gather's own-slice scratch (pulsar_tp_vocab_gather)
    uint64_t tp_vocab_seq;
} pulsar_qwen_state;


/** L266 step 7: tensor parallelism.  _load (at the family's load, the model bound): this rank's head counts
 *  into g_qwen_shape, the rank folded into the artifact digest.  _slices (the family's tp_slices op, L272 P4b):
 *  declares the rank's slices -- the dense heads' columns / rows, the GDN conv's channels, its whole experts --
 *  into the core's plan, which builds them and is the residency rule (tp_slice.cpp). */
bool pulsar_qwen_tp_load(pulsar_engine *e);
bool pulsar_qwen_tp_slices(pulsar_engine *e, pulsar_tp_plan *plan);

/** The Qwen family's bank-pool operations (family.h pulsar_family_bank_ops). */
extern const pulsar_family_bank_ops k_qwen_bank_ops;


/** L270: the one writer of a bank's position; it keeps the bank's KV high-water. */
static inline void qwen_bank_set_pos(pulsar_qwen_state *st, uint32_t bank, uint32_t pos) {
    st->bank_pos[bank] = pos;
    if (pos > st->kv_hw[bank]) st->kv_hw[bank] = pos;
}
/** L270: the demand-paged KV bytes `rows` positions take in one bank (every QSA layer's KV + pooled
 * indexer keys, the MTP layer's included) -- the allocation's own row functions. */
uint64_t qwen_kv_bytes_at(const pulsar_qwen_state *st, uint64_t rows);
/** L284 #3: bank `bank`'s physical residency -- its own KV + pooled-index tensors on every TRUNK QSA layer.
 *  The MTP layer's stay resident: a segment chain does not carry them (kv_state_qwen.cpp qwen_pools), so a
 *  restore could not bring them back.  _free releases them (cudaFree) and zeroes the bank's KV high-water;
 *  _alloc re-backs whichever are missing (idempotent), and on a failure frees what it made (the bank is
 *  whole or wholly evicted); _evicted is true when any is missing. */
void qwen_bank_kv_free(pulsar_qwen_state *st, uint32_t bank);
bool qwen_bank_kv_alloc(pulsar_qwen_state *st, uint32_t bank);
bool qwen_bank_kv_evicted(const pulsar_qwen_state *st, uint32_t bank);

/* ---- 5. The step and the op table ------------------------------------------ */

typedef enum {
    /** Each row is its own bank's NEXT token (one row per bank, any mix of banks). */
    PULSAR_QWEN_STEP_DECODE = 0,
    /** Rows are runs of consecutive positions, one bank each (pulsar_qwen_step::run_first): a prefill chunk is
     *  one run; a verify of several banks (L272 P1 S4) is one run per bank. */
    PULSAR_QWEN_STEP_PREFILL = 1,
} pulsar_qwen_step_mode;

typedef enum { PULSAR_QWEN_GR_ATTN = 0, PULSAR_QWEN_GR_MLP = 1 } pulsar_qwen_gr_side;


/** L284 P15: the importance-matrix collection's observer (imatrix_qwen.cpp).  The trunk's ops note the input rows of
 *  every linear they run through the core's front doors (step_linear, the shared expert, the routed MoE) into it;
 *  all of them read device rows (bf16) and synchronise first, so a collection is slow and bit-exact otherwise. */
struct pulsar_imatrix_tap;
/** A dense linear's input: x_dev [rows][cols] bf16 under the tensor's name. */
void pulsar_imatrix_note_dense(pulsar_imatrix_tap *tap, const pulsar_tensor *w, const uint16_t *x_dev, uint32_t rows,
                               uint32_t cols);
/** The routed gate / up input: x_dev [rows][cols] bf16 and each row's `k` picks sel_dev [rows][k], under the gate's
 *  name and, for a gate + up pair (`up` non-NULL), the up's too. */
void pulsar_imatrix_note_routed_in(pulsar_imatrix_tap *tap, const pulsar_tensor *gate, const pulsar_tensor *up,
                                   const uint16_t *x_dev, const int32_t *sel_dev, uint32_t rows, uint32_t k,
                                   uint32_t n_expert, uint32_t cols);
/** The routed down input: mid_dev [rows * k][mid] bf16 (a pick's SwiGLU row, route weight folded in) under the
 *  down's name. */
void pulsar_imatrix_note_routed_mid(pulsar_imatrix_tap *tap, const pulsar_tensor *down, const uint16_t *mid_dev,
                                    const int32_t *sel_dev, uint32_t rows, uint32_t k, uint32_t n_expert,
                                    uint32_t mid);

/** Everything an op needs for one step.  Built by the driver, read-only to
 * the ops (the tensors' CONTENTS are what ops write). */
typedef struct {
    const pulsar_model *model;    ///< tensor_map_base() for the weights
    const pulsar_qwen_shape *shape;
    const pulsar_qwen_weights *w;
    const pulsar_layer_plan *plan;
    pulsar_qwen_state *st;
    pulsar_qwen_step_mode mode;
    uint32_t n_rows;
    const int32_t *tokens;                 ///< [n_rows] host token ids
    const int32_t *pos;                    ///< [n_rows] host positions
    const int32_t *bank;                   ///< [n_rows] host bank ids (PREFILL: all equal)
    /** L268: [n_rows][6] host multi-axis rope positions (pulsar_qsa_io::row_rope), or NULL = text positions -- NULL
     *  whenever no row's bank holds an image block, so a text step is the bytes it always was */
    const uint32_t *rope;
    /** The residual streams the step runs on: st->streams for the trunk, st->mtp_streams for the MTP
     *  layer (L251 MTP).  Every op reads and writes THIS, never st->streams directly. */
    pulsar_gpu_tensor *streams;
    /** The mixer the head reads through: the trunk's, or mtp.mixer for the MTP head. */
    const pulsar_qwen_gr_weights *mixer;
    /** L251 MTP: the step's rows [0, n_dec) are a VERIFY (PREFILL mode, <= SPEC_ROWS rows): the recurrent ops
     *  also write those rows' per-row states and the QSA ops their stages + raw keys into st->spec.  L284 #2:
     *  rows [n_dec, n_rows) of the same step are prompt runs, never captured. */
    bool verify;
    /** L272 P1 S4: PREFILL rows as runs -- run k is rows [run_first[k], run_first[k + 1]) of one bank at
     *  consecutive positions: the verify's runs (one a bank) first, then (L284 #2, the fused step) its prompt runs
     *  from row n_dec, one bank each.  n_runs 1 = the classic one-bank chunk, or a one-bank verify. */
    uint32_t n_runs;
    const uint32_t *run_first;
    /** L251 MTP: the head runs the DRAFT head (pulsar_qwen_weights::draft_head_mx): n_draft logits a row. */
    bool draft_head;
    /** L284 #2: the row-kind boundary.  Rows [0, n_dec) take the decode arms (what decode / verify rows are
     *  graded against), rows [n_dec, n_rows) the prompt arms (the same bytes at every row count, L266).  The
     *  step's builder sets it (pulsar_qwen_step_n_dec); the ops cut their arm-sensitive launches at it. */
    uint32_t n_dec;
    /** L284 P15: the importance-matrix observer while a collection runs, else NULL (the trunk's steps only). */
    pulsar_imatrix_tap *tap;
} pulsar_qwen_step;

/** L284 #2: n_dec for a step -- a DECODE step is all decode rows; a PREFILL step's decode rows are its verify
 *  rows [0, n_verify) and the rest are prompt rows, so a prompt cut anywhere (or behind a verify, the fused
 *  step) is byte-identical to one prefilled whole. */
static inline uint32_t pulsar_qwen_step_n_dec(pulsar_qwen_step_mode mode, uint32_t n_verify, uint32_t n_rows) {
    return mode == PULSAR_QWEN_STEP_PREFILL ? n_verify : n_rows;
}
/** L284 #2: the verify runs of a PREFILL step -- its runs that start before n_dec (each lies wholly inside). */
static inline uint32_t pulsar_qwen_step_verify_runs(const pulsar_qwen_step *st) {
    uint32_t k = 0;
    while (k < st->n_runs && st->run_first[k] < st->n_dec) k++;
    return k;
}

/** A per-layer op: reads/writes the step's slots for layer il.  Returns false
 * after printing what failed; the driver refuses the step. */
typedef bool (*pulsar_qwen_layer_fn)(const pulsar_qwen_step *st, uint32_t il);

/** The Qwen forward's ops, in forward order.  NULL = not implemented: the
 * driver names the first NULL one it would call.  Owners in brackets. */
typedef struct {
    /** tokens -> streams (embed_tokens, repeated to n_hc streams).  [S4] */
    bool (*embed)(const pulsar_qwen_step *st);
    /** layer ple_layer: streams += PLE(streams, n-gram rows of the step's
     * tokens); also advances the PLE conv state and the n-gram context.  [S4] */
    pulsar_qwen_layer_fn ple;
    /** streams -> x through layer il's gated-residual read (side).  [S4] */
    bool (*gr_read)(const pulsar_qwen_step *st, uint32_t il, pulsar_qwen_gr_side side);
    /** x -> y, Gated DeltaNet; reads+advances gdn_state/gdn_conv.  [S2] */
    pulsar_qwen_layer_fn gdn;
    /** x -> y, QSA; appends K/V + index keys, attends.  [S3] */
    pulsar_qwen_layer_fn qsa;
    /** streams += y (x) injection weights of layer il's side.  [S4] */
    bool (*gr_write)(const pulsar_qwen_step *st, uint32_t il, pulsar_qwen_gr_side side);
    /** x -> y, router + routed experts + gated shared expert.  [S4] */
    pulsar_qwen_layer_fn moe;
    /** Step rows [row0, row0 + n) of streams -> logits rows [out0, out0 + n): mixer read + lm_head, out0 + n <=
     * PULSAR_QWEN_HEAD_ROWS_MAX.  The driver calls it once per span of its head list (consecutive rows of one
     * row kind): every row of a DECODE step or a verify, the last row of a PREFILL chunk.  [S4] */
    bool (*head)(const pulsar_qwen_step *st, uint32_t row0, uint32_t n, uint32_t out0);
    /** Bytes of scratch op `op` needs at max_rows rows (0 = none).  Called
     * once per session create; NULL = the op needs none. */
    uint64_t (*scratch_bytes)(pulsar_qwen_op_id op, const pulsar_qwen_shape *s, uint32_t max_rows, uint32_t ctx);
} pulsar_qwen_ops;

/** THE op table.  Defined in family_qwen.cpp; each stream fills its entries
 * there when its op lands (and deletes nothing else). */
extern const pulsar_qwen_ops g_qwen_ops;

/* ---- S4's ops (src/engine/family_qwen_s4.cpp) --------------------------------
 * embed, PLE, the gated-residual read / write, the MoE block and the head
 * (mixer + lm_head), their scratch, and at load: the admission of every tensor
 * those ops read (by layout -- a format no arm reads is refused at load) and the
 * PLE row file. */
bool pulsar_qwen_s4_embed(const pulsar_qwen_step *st);
bool pulsar_qwen_s4_ple(const pulsar_qwen_step *st, uint32_t il);
bool pulsar_qwen_s4_gr_read(const pulsar_qwen_step *st, uint32_t il, pulsar_qwen_gr_side side);
bool pulsar_qwen_s4_gr_write(const pulsar_qwen_step *st, uint32_t il, pulsar_qwen_gr_side side);
bool pulsar_qwen_s4_moe(const pulsar_qwen_step *st, uint32_t il);
bool pulsar_qwen_s4_head(const pulsar_qwen_step *st, uint32_t row0, uint32_t n, uint32_t out0);
/** L251 MTP: the input combine into st->streams (the MTP streams) from the trunk stack rows `h`
 *  (device, [n_rows][n_hc][n_embd] bf16) and st->tokens (x_{p+1} per row). */
bool pulsar_qwen_s4_mtp_combine(const pulsar_qwen_step *st, const void *h);
/** The combine's workspace bytes (a fixed size; the session allocates it once). */
uint64_t pulsar_qwen_s4_mtp_combine_ws_bytes(void);
/** L251 MTP: after a verify step `vst` of R rows at positions p .. p + R - 1, keep rows 0 .. keep - 1
 *  and roll every recurrent state of the bank back to "after row keep - 1": GDN recurrent + conv and
 *  the PLE conv from the per-row capture, each QSA layer's index stage from its snapshot plus the kept
 *  rows' raw keys, the n-gram context by replaying the kept tokens.  keep == R is a no-op. */
bool pulsar_qwen_s4_spec_rollback(const pulsar_qwen_step *vst, uint32_t keep);

/* S2's op, landed on the integration branch in family_qwen_s4.cpp (see the note
 * there); moves to family_qwen_s2.cpp when S2 rebases. */
bool pulsar_qwen_s2_gdn(const pulsar_qwen_step *st, uint32_t il);
bool pulsar_qwen_s3_qsa(const pulsar_qwen_step *st, uint32_t il);
uint64_t pulsar_qwen_s4_scratch_bytes(pulsar_qwen_op_id op, const pulsar_qwen_shape *s, uint32_t max_rows, uint32_t ctx);
bool pulsar_qwen_s4_load(pulsar_engine *e, const pulsar_engine_options *opt);
/** Called by the step driver after every forward that got past its op check:
 *  waits out a PLE gather a failed step left in flight, and after a completed
 *  step reads (and clears) the MoE non-finite flag.  Returns ok && no NaN. */
bool pulsar_qwen_s4_step_end(const pulsar_qwen_step *st, bool ok);
void pulsar_qwen_s4_unload(pulsar_qwen_weights *w);

/** Name and owner of an op, for the refusal line ("gdn", "S2 work/l251-gdn"). */
const char *pulsar_qwen_op_name(pulsar_qwen_op_id op);
const char *pulsar_qwen_op_owner(pulsar_qwen_op_id op);

/** The first op a forward over `plan` would call that `ops` lacks, in the
 * driver's own order (PULSAR_QWEN_OP_COUNT: none).  *at_layer is its layer, or
 * UINT32_MAX for embed/head.  The step driver refuses with exactly this. */
pulsar_qwen_op_id pulsar_qwen_first_missing_op(const pulsar_qwen_ops *ops, const pulsar_layer_plan *plan,
                                               const pulsar_qwen_shape *shape, uint32_t *at_layer);

/** Bytes a session's state takes at (n_banks, ctx, max_rows, with the MTP layer or not): the
 * allocation code run dry, so the price and the allocation are one function.
 * *managed_bytes (optional) is the demand-paged subset (the KV slabs). */
uint64_t pulsar_qwen_state_price(const pulsar_qwen_shape *s, const pulsar_layer_plan *plan,
                                 uint32_t n_banks, uint32_t ctx, uint32_t max_rows, bool mtp,
                                 uint64_t *managed_bytes);

#endif /* PULSAR_FAMILY_QWEN_H */

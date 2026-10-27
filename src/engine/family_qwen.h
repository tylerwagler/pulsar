/* family_qwen.h -- the Qwen4-exp family (Qwen3.8-Flash-Next): shape, container
 * contract, weights, session state, and the per-layer op interface that the
 * L251 streams implement (S1 owns this header; S2 GDN, S3 QSA, S4 MoE/GR/PLE/
 * head fill the op table; S5 the tokenizer/renderer; S6 the container).
 *
 * THE FORWARD (HF transformers `qwen4_exp`, modeling_qwen4_exp.py), which the
 * core step driver (step_driver.h, L295 F4) runs in exactly this order -- embed,
 * then one block a layer by the plan's kind (family_qwen.cpp qwen_layer), then
 * the head:
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
 * The driver refuses a session whose block table lacks what the plan needs
 * (pulsar_driver_first_missing) -- no stub runs, nothing falls back (rules 1, 9). */
#ifndef PULSAR_FAMILY_QWEN_H
#define PULSAR_FAMILY_QWEN_H

#include <stdint.h>
#include <stdbool.h>

#include "pulsar_gpu.h"

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
    /* L280: the QSA rope as loaded -- rope_parameters' YaRN, or the --rope-yarn option's over a native container
     * (yarn.factor 0 = the native rope) -- and its table (pulsar_qsa_rope_init, made once at load, the one every
     * QSA call passes).  Not part of the known geometry: qwen_shape_check does not compare them. */
    pulsar_qsa_yarn yarn;
    pulsar_qsa_rope qsa_rope;
    uint64_t rope_id;          ///< the rope's identity (family.h rope_id): 0 native, else a digest of `yarn`
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

/** L255: one Tessera projection stack's kernel-ready planes (src/cuda/mmq/pulsar_tessera.h), in this order,
 *  each the tensor mlp.experts.tessera.\<proj\>.\<plane\> (plane names in family_qwen.cpp). */
enum {
    PULSAR_QWEN_TESS_WORDS = 0,   ///< I32  [n_expert][words_stride]
    PULSAR_QWEN_TESS_TABLE,       ///< BF16 [n_expert][16384] composed table
    PULSAR_QWEN_TESS_INIT,        ///< I32  [n_expert][in]
    PULSAR_QWEN_TESS_HAS_INIT,    ///< I32  [n_expert]
    PULSAR_QWEN_TESS_WSCALE,      ///< F32  [n_expert][out]
    PULSAR_QWEN_TESS_RUNS,        ///< I32  [n_expert][8]
    PULSAR_QWEN_TESS_BDESC,       ///< I32  [n_expert][in / 32][12]
    PULSAR_QWEN_TESS_GEOM,        ///< I32  [2]: tile_words, slot_words (read on the host)
    PULSAR_QWEN_TESS_PLANES
};
enum { PULSAR_QWEN_TESS_GATE = 0, PULSAR_QWEN_TESS_UP, PULSAR_QWEN_TESS_DOWN, PULSAR_QWEN_TESS_PROJS };

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
    /** L255: the routed experts in Tessera's value family -- set (every plane of all three projections)
     *  exactly when the artifact carries the layer's experts that way, and then every EXL3 expert tensor above
     *  is NULL: a layer declares its experts once, and the load refuses a partial set or both forms. */
    pulsar_tensor *moe_tess[PULSAR_QWEN_TESS_PROJS][PULSAR_QWEN_TESS_PLANES];
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
    pulsar_qwen_gr_weights mixer;  ///< mtp.hyper_connection_mixer (no inject), then the draft head
    /** L288 qwen-exl3-head: the drafter's OWN head, mtp.draft_head.weight [n_embd -> n_draft], when the artifact
     *  carries one -- stored in whatever format the container gives it (a format the dense bf16-rows arm serves,
     *  EXL3 in the built containers), its row i the logit of mtp.draft_head.ids[i] (I32 [n_draft]), which the load
     *  checks against qwen_draft_vocab's ids.  NULL: the draft rows are gathered from a bf16 lm_head into MXFP8
     *  (pulsar_qwen_weights::draft_head_mx).  Both or neither are present. */
    pulsar_tensor *draft_head;
    pulsar_tensor *draft_head_ids;   ///< mtp.draft_head.ids I32 [n_draft]: the ids of draft_head's rows, in order
} pulsar_qwen_mtp_weights;

typedef struct {
    pulsar_tensor *token_embd;     ///< model.language_model.embed_tokens.weight [n_embd x n_vocab]
    pulsar_tensor *output;         ///< lm_head.weight [n_embd -> n_vocab]
    pulsar_qwen_gr_weights mixer;         ///< model.language_model.hyper_connection_mixer
    /** [0, n_layer) the trunk; [n_layer] the MTP layer when mtp.present (a QSA layer, split experts) */
    pulsar_qwen_layer_weights layer[PULSAR_FAMILY_MAX_LAYER];
    pulsar_qwen_mtp_weights mtp;
    /** L299: the block drafter (spec_block.h: DFlash2) when the open selected it -- then the MTP tensors are
     *  unloaded and mtp.present is false; NULL otherwise (its tensors, if the container holds them, unloaded). */
    struct pulsar_block_drafter *block;
    /* The PLE hash's config (section 2): read at load, used by the host-side
     * n-gram id hashing (S4). */
    uint64_t ple_multipliers[PULSAR_QWEN_MAX_NGRAM];       ///< [ngram_size]
    uint64_t ple_head_vocab[PULSAR_QWEN_MAX_NGRAM_HEADS];  ///< table size (a prime) per n-gram head
    uint64_t ple_head_offset[PULSAR_QWEN_MAX_NGRAM_HEADS]; ///< first row of each head's table
    /** S4: the PLE row file (the container's pulsar.ple_rows.*), its pread pool
     * and the host gather in flight; opened by pulsar_qwen_s4_load. */
    struct pulsar_qwen_ple_io *ple_io;
    /** L251: the lm_head as MXFP8 (mxfp8_lt), made on the device from a bf16 `output` at the first head
     *  step (the bf16 head is 1.27 GB read per decode row; this is half); freed at unload.  L288
     *  qwen-exl3-head: NULL when the container stores `output` in a format the dense arm reads itself. */
    struct pulsar_gpu_tensor *head_mx;
    /** L251 MTP: the DRAFT head -- the lm_head rows of draft_ids (ascending: [0, 65536), L288's
     *  frequency-ranked ids above it, and the added / special ids [eos_id, n_vocab), the stop tokens among
     *  them; chosen at load by qwen_draft_vocab) gathered into one MXFP8 matrix, made at the first MTP head
     *  when the artifact carries no mtp.draft_head of its own (else NULL; the ids are the same either way).
     *  The drafter's argmax runs over these; verification keeps the full head, so the output is unchanged
     *  and only the draft cost moves.  draft_ids[i] is row i's token id. */
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
 * (36 GDN layers x 3.1 MB fp32 + conv + PLE), whatever its length.
 *
 * L295 F3: the family DECLARES this state (family_qwen.cpp qwen_state_plan, the
 * kinds in state_kinds.h) and the core owns it -- allocation, the demand-paged
 * pools, the checkpoint walk, segments, payloads, the verify capture and its
 * commit.  The ops reach a layer's state through the step's `states`.
 *
 * FORMATS -- each owned by the stream that PRODUCES the bytes (rule 3).  The
 * sizes below are the declaration's authority; an owner who changes a format
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
/** Rows the head may emit per step (the logits slab): the batched lane's bound -- the core driver's (32, the shared
 *  verify width, L288 C1). */
#define PULSAR_QWEN_HEAD_ROWS_MAX   PULSAR_DRIVER_HEADS_MAX


/** L295 F3: the ids of Qwen's state declarations (the plan's table, in this order; state_kinds.h kinds).  A GDN
 *  layer carries GDN_REC + GDN_CONV, a QSA layer KV + IDX_KEYS + IDX_TAIL, the PLE layer PLE_CONV as well; the MTP
 *  layer's slot (n_layer, a drafter slot) a QSA layer's + MTP_PEND; NGRAM is session-wide host state. */
enum {
    PULSAR_QWEN_STATE_GDN_REC = 0,  ///< FIXED, DELTA (L288): the DeltaNet recurrent state; a verify row keeps its update (pulsar_gdn_upd_floats f32)
    PULSAR_QWEN_STATE_GDN_CONV,     ///< FIXED, SNAPSHOT: the GDN causal-conv tail
    PULSAR_QWEN_STATE_KV,           ///< POOL, one token a row: a QSA layer's K/V (pulsar_qwen_kv_row_bytes)
    PULSAR_QWEN_STATE_IDX_KEYS,     ///< POOL, idx_block tokens a row: the pooled indexer keys
    PULSAR_QWEN_STATE_IDX_TAIL,     ///< STAGED, STAGE: the open indexer block; a verify row's record is its index_qk projection (PULSAR_QSA_IDX_IN f32, the key after the query heads)
    PULSAR_QWEN_STATE_PLE_CONV,     ///< FIXED, SNAPSHOT: the PLE layer's dilated-conv history
    PULSAR_QWEN_STATE_MTP_PEND,     ///< FIXED: the MTP slot's pending trunk stack, bf16 [n_hc][n_embd]
    PULSAR_QWEN_STATE_NGRAM,        ///< host FIXED, REPLAY: the n-gram context (the last ngram_size - 1 ids, reset at EOS)
    PULSAR_QWEN_STATE_COUNT
};
/** Build Qwen's state declaration into e->state_plan (the load, after the shape is the rank's and the MTP layer is
 *  known). */
void qwen_state_plan(pulsar_engine *e);

/** Which op a scratch arena belongs to (the driver's scratch, pulsar_driver_ops::scratch_bytes). */
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

/** L272 P1 S4: the rows one verify step carries, every bank's run together -- the head's row cap, at every width
 *  of which the kernels keep their decode arms (each row's logits the bytes a one-token decode gives; L288 C1: in
 *  16-row blocks past 16).  The family's verify width (fused_heads_max, L284): the server's allocator fits N banks'
 *  drafts within it, one base row each. */
#define PULSAR_QWEN_SPEC_ROWS PULSAR_QWEN_HEAD_ROWS_MAX

/** L296 N9: the MTP drafter -- the chained-MTP drafter (spec_mtp.h) at this family's numbers (family_qwen.cpp). */
extern const struct pulsar_drafter_ops PULSAR_QWEN_MTP_DRAFTER;

struct pulsar_qwen_step;
/** A Qwen session's own device state (L295 F3 / F4): the driver's buffers (the base: the logits slab, the step's
 *  positions and banks, the op scratch by PULSAR_QWEN_OP_*, the TP share, the verify rows) and the declared state
 *  (pulsar_session::states) are the core's; this is the step's activation slots and the MTP drafter's lane. */
struct pulsar_qwen_state : pulsar_driver_state {
    /* per-step activation slots (rows <= max_rows) */
    pulsar_gpu_tensor *streams;     ///< [max_rows][n_hc][n_embd] PULSAR_QWEN_STREAM_ELT_SIZE
    pulsar_gpu_tensor *x;           ///< [max_rows][n_embd] block input (GR read -> mixer / MoE)
    pulsar_gpu_tensor *y;           ///< [max_rows][n_embd] block output (mixer / MoE -> GR write)
    /** The trunk's step object (the driver fills its core part each step). */
    struct pulsar_qwen_step *trunk;
    /* L251 MTP (mtp.present): the MTP layer's KV / index state is the declared state's slot n_layer; these are its
     * activation slots.  The chained-MTP lane (spec_mtp.h, L296 N9) parks each bank's LAST trunk row in the MTP_PEND
     * state until its next token exists (the MTP row at position p pairs the trunk stack at p with x_{p+1}) and stages
     * a step's conditioning rows; the module step is this family's (family_qwen.cpp). */
    bool mtp;
    pulsar_gpu_tensor *mtp_streams; ///< [max_rows][n_hc][n_embd] bf16, the MTP layer's streams
    pulsar_gpu_tensor *mtp_tok;     ///< [max_rows] i32, x_{p+1} per MTP row
    pulsar_gpu_tensor *mtp_ws;      ///< pulsar_qwen_mtp_combine_workspace_bytes
    pulsar_gpu_tensor *mtp_top;     ///< L289: pulsar_qwen_draft_top_ws_bytes, a draft row's pick + confidence
    /** L272 P1: [n_banks] the MTP layer's index stage after each bank's last TRUE row, saved before a draft
     *  chain writes draft rows into it (the lane's stage_save / stage_restore; L272 P1 S4: per bank). */
    pulsar_gpu_tensor *mtp_stage;
    /** L289: [n_banks][2] the n-gram context a draft chain's PLE prefetch advances (the bank's, at the chain's start) */
    int32_t *mtp_pctx;
};

/** L266 step 7: tensor parallelism.  _load (at the family's load, the model bound): this rank's head counts
 *  into g_qwen_shape, the rank folded into the artifact digest.  _slices (the family's tp_slices op, L272 P4b):
 *  declares the rank's slices -- the dense heads' columns / rows, the GDN conv's channels, its whole experts --
 *  into the core's plan, which builds them and is the residency rule (tp_slice.cpp). */
bool pulsar_qwen_tp_load(pulsar_engine *e);
bool pulsar_qwen_tp_slices(pulsar_engine *e, pulsar_tp_plan *plan);

/* ---- 5. The step and the blocks ---------------------------------------------- */

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

/** Everything a Qwen op needs for one step: the core's step (step_driver.h: rows, runs, the declared state, the
 *  driver's buffers) and the family's own part, fixed per lane -- the trunk's (pulsar_qwen_state::trunk), or the MTP
 *  layer's (qwen_mtp_forward).  Read-only to the ops (the tensors' CONTENTS are what ops write). */
struct pulsar_qwen_step : pulsar_driver_step {
    const pulsar_qwen_shape *shape;
    const pulsar_qwen_weights *w;
    pulsar_qwen_state *st;        ///< the session's Qwen state (the driver's buffers are its base)
    /** The residual streams the step runs on: st->streams for the trunk, st->mtp_streams for the MTP
     *  layer (L251 MTP).  Every op reads and writes THIS, never st->streams directly. */
    pulsar_gpu_tensor *streams;
    /** The mixer the head reads through: the trunk's, or mtp.mixer for the MTP head. */
    const pulsar_qwen_gr_weights *mixer;
    /** L251 MTP: the head runs the DRAFT head (the artifact's mtp.draft_head, or the gathered
     *  pulsar_qwen_weights::draft_head_mx): n_draft logits a row.  The MTP layer's lane. */
    bool draft_head;
};

/** A per-layer op: reads/writes the step's slots for layer il.  Returns false
 * after printing what failed; the driver refuses the step. */
typedef bool (*pulsar_qwen_layer_fn)(const pulsar_qwen_step *st, uint32_t il);

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
/** L289: the draft head's greedy pick over logits rows [0, n) of an MTP step that headed n rows (L288: every bank's
 *  draft row of a step, n <= PULSAR_QWEN_HEAD_ROWS_MAX), reduced on the device into st->st->mtp_top (enqueued after
 *  the head); pulsar_qwen_s4_draft_top_read, after the step's drain, gives each row's vocabulary id and softmax
 *  probability -- the bytes the host scan gave -- and refuses a NaN row. */
bool pulsar_qwen_s4_draft_top(const pulsar_qwen_step *st, uint32_t n);
bool pulsar_qwen_s4_draft_top_read(const pulsar_qwen_state *q, const pulsar_qwen_weights *w, uint32_t n, int32_t *id,
                                   float *conf);
/** Its workspace bytes (the session allocates it once, beside the combine's). */
uint64_t pulsar_qwen_s4_draft_top_ws_bytes(void);
/** L289: warm the page cache with the PLE rows of `n` tokens a later step will embed after the n-gram context `ctx`
 *  (oldest first; advanced over the tokens here, as the embed will) -- the verify's rows, issued while the draft
 *  chain runs, so the verify's own gather reads cached rows.  A hint: the gather still reads every row. */
void pulsar_qwen_s4_ple_prefetch(const pulsar_qwen_weights *w, int32_t ctx[2], const int32_t *tokens, uint32_t n);
/** L329 (b2): the family's pulsar_driver_ops::read_ahead -- the PLE rows of prompt[from, to) (the n-gram context the
 *  prompt's own two tokens before `from`) queued on the pool's background class under `tag`.  0 when the read-ahead
 *  is off (PULSAR_PLE_READ_AHEAD=0) or a token is outside the vocabulary (the embed refuses it by name). */
uint32_t pulsar_qwen_s4_ple_read_ahead(const pulsar_qwen_weights *w, uint64_t tag, const int32_t *prompt, uint32_t from,
                                       uint32_t to);
/** L329 (b2): the PLE rows' pread pool (pulsar_driver_ops::read_ahead_pool). */
pulsar_engram_io *pulsar_qwen_s4_ple_pool(const pulsar_qwen_weights *w);
/** L295 F3: the n-gram context's verify commit -- `ctx` (oldest first, the bank's before the step) advanced over the
 *  kept tokens. */
bool pulsar_qwen_s4_ngram_replay(const pulsar_qwen_weights *w, int32_t ctx[2], const int32_t *tokens, uint32_t keep);

/* S2's op, landed on the integration branch in family_qwen_s4.cpp (see the note
 * there); moves to family_qwen_s2.cpp when S2 rebases. */
bool pulsar_qwen_s2_gdn(const pulsar_qwen_step *st, uint32_t il);
bool pulsar_qwen_s3_qsa(const pulsar_qwen_step *st, uint32_t il);
uint64_t pulsar_qwen_s4_scratch_bytes(pulsar_qwen_op_id op, const pulsar_qwen_shape *s, uint32_t max_rows, uint32_t ctx);
bool pulsar_qwen_s4_load(pulsar_engine *e, const pulsar_engine_options *opt);
/** Called by the step driver after every forward that started (pulsar_driver_ops::step_end):
 *  waits out a PLE gather a failed step left in flight, and after a completed
 *  step reads (and clears) the MoE non-finite flag.  Returns ok && no NaN. */
bool pulsar_qwen_s4_step_end(const pulsar_qwen_step *st, bool ok);
void pulsar_qwen_s4_unload(pulsar_qwen_weights *w);

#endif /* PULSAR_FAMILY_QWEN_H */

/* nemotron_h.h -- the Nemotron-H family (L296 N1): NVIDIA Nemotron 3.5 Super VL's language model as a family module.
 *
 * Built in the plugin layout (src/families/<name>/, roadmap "The plugin contract") on Inkling's pattern: the family
 * declares its shape and its tensor roles and binds through the core's role-driven binder (src/engine/role_bind.h)
 * by the bind records the builder writes (tools/container/nemotron_h.py) -- no name is parsed.
 *
 * THE MODEL (config.json llm_config; plan 121 section 1): HF blocks, each ONE pre-norm residual sublayer
 *     x += mixer(RMSNorm(x)),   mixer in { Mamba-2 (SSD), GQA attention (NoPE), latent MoE }
 * 88 blocks in the full model: 40 Mamba-2, 40 MoE, 8 attention.  PULSAR GROUPS a mixer with the MoE after it into one
 * PULSAR_LAYER_BLOCKS layer -- {MAMBA2 | ATTN_FULL} x {MOE | NONE}: 48 layers (32 {M,E}, 8 {M,-}, 8 {*,E}).  The latent
 * MoE: router logits f32 on the full hidden, sigmoid, a selection-only bias, top-22 of 512 renormalised x 5.0; the
 * routed experts run on a 1024-wide latent (fc1 4096 -> 1024, up 1024 -> 2688, relu^2, down 2688 -> 1024, the
 * weighted sum, fc2 1024 -> 4096) beside one ungated relu^2 shared expert on the full hidden.  Untied embed / head,
 * no positional encoding anywhere.  One MTP module ({attention, MoE}, applied recursively by the card).
 *
 * WHAT IS HERE (N1): the shape (the container's nemotron_h.* keys, every geometry field pinned to the profile, every
 * switch pinned to the value the blocks implement, every training-only key named as such, any unknown key refused),
 * the layer plan, the role-driven binding of every tensor with its format admitted through the core's front doors,
 * the byte census, and the tokenizer + chat front (N8, nemotron_h_tokenizer.cpp).  The forward is NOT here: the
 * Mamba-2 block (N2), the state declaration (N3), the latent relu^2 MoE (N4) and the driver ops (N7) are not built,
 * so a session refuses by name (nemotron_h_family.cpp).
 *
 * FORMAT-NEUTRAL: every linear (Mamba in / out, attention, the latent projections, the shared expert, the drafter's
 * eh_proj, the head) is admitted through the format registry at bf16 rows -- BF16 today, an EXL3 dense-arm rate or
 * MXFP8 as the registry serves it; the routed up / down stacks through the registry's routed roles (BF16 today; the
 * up-only arm a quant format needs is the format verdict's, L296).
 *
 * THE CONTAINER CONTRACT (tools/container/nemotron_h.py emits, this family reads):
 *   general.architecture = "nemotron_h";
 *   nemotron_h.<key>   llm_config verbatim (kv.config_kvs), except time_step_limit, written as
 *                      nemotron_h.time_step_limit.min (0) + .max_inf (true) -- the builder refuses any other;
 *   pulsar.mtp_present (the container holds the drafter), pulsar.vision_present (false in v1).
 *   Every tensor carries a bind record: role (+ block `layers` / `mtp`, layer = the PULSAR layer, index).  Linears
 *   are stored [in, out] in ne order; the conv [kernel, channels]; routed experts are pulsar.experts stacks
 *   [in, out, n_expert] (up: latent -> ff, down: ff -> latent).
 */
#ifndef PULSAR_FAMILY_NEMOTRON_H_H
#define PULSAR_FAMILY_NEMOTRON_H_H

#include "engine/pulsar_engine_internal.h"
#include "engine/role_bind.h"

#define PULSAR_NEMOTRON_H_ARCH "nemotron_h"
/** The most HF blocks a container's layers_block_type may list (the full model has 88). */
#define PULSAR_NEMOTRON_H_MAX_BLOCKS 128u

/** The family descriptor (nemotron_h_family.cpp); family.cpp registers it. */
extern const pulsar_family PULSAR_FAMILY_NEMOTRON_H;

/* ---- 1. Shape -------------------------------------------------------------------------------------------------- */

/** The loaded shape.  The geometry is checked field by field against PULSAR_NEMOTRON_H_SHAPE_SUPER (the blocks are
 *  declared for these numbers; a second geometry is a second profile, added on purpose); the PLAN -- the block kinds,
 *  and so the grouped layers -- is the container's (the 5-block mini and the 88-block model are one profile). */
typedef struct {
    const char *name;
    /* geometry: pinned */
    uint32_t n_embd;           ///< hidden_size (4096)
    uint32_t n_vocab;          ///< vocab_size (131072): the embed rows and the logits width (every id real)
    uint32_t n_head;           ///< num_attention_heads (32)
    uint32_t n_head_kv;        ///< num_key_value_heads (2)
    uint32_t head_dim;         ///< head_dim (128); the score scale is 1/sqrt(head_dim) (no positional encoding)
    uint32_t ssm_n_head;       ///< mamba_num_heads (128)
    uint32_t ssm_head_dim;     ///< mamba_head_dim (64): d_inner = 8192 = expand x n_embd
    uint32_t ssm_state;        ///< ssm_state_size (128)
    uint32_t ssm_groups;       ///< n_groups (8): B / C shared by ssm_n_head / ssm_groups heads; the out norm's groups
    uint32_t conv_k;           ///< conv_kernel (4): over x | B | C (d_inner + 2 groups x state channels), with bias
    uint32_t expand;           ///< expand (2)
    uint32_t n_expert;         ///< n_routed_experts (512)
    uint32_t n_expert_used;    ///< num_experts_per_tok (22)
    uint32_t n_ff_exp;         ///< moe_intermediate_size == intermediate_size (2688): a routed expert's width
    uint32_t latent;           ///< moe_latent_size (1024): the routed experts' input / output width
    uint32_t n_ff_shared;      ///< moe_shared_expert_intermediate_size (5376)
    float rms_eps;             ///< layer_norm_epsilon (1e-5): every RMSNorm, the gated out norm included
    float route_scale;         ///< routed_scaling_factor (5.0): on the renormalised weights' sum
    uint64_t max_context;      ///< max_position_embeddings (1048576)
    uint32_t eos_id;           ///< eos_token_id (11, <|im_end|>)
    /* the plan: the container's */
    uint32_t n_blocks;         ///< len(layers_block_type)
    uint32_t n_layer;          ///< the grouped layers
    uint64_t mamba_layers;     ///< bit il: layer il's mixer is Mamba-2 (else attention)
    uint64_t moe_layers;       ///< bit il: layer il has the MoE (else its mixer alone)
    uint32_t n_mtp;            ///< num_nextn_predict_layers (0 or 1: one {attention, MoE} module)
    bool mtp_written;          ///< pulsar.mtp_present: the container holds the drafter
} pulsar_nemotron_h_shape;

extern const pulsar_nemotron_h_shape PULSAR_NEMOTRON_H_SHAPE_SUPER;

/** Read the shape from the container's keys and check it (nemotron_h_shape.cpp).  Every refusal is said and counted
 *  (pulsar_load_refuse); false = at least one. */
bool pulsar_nemotron_h_read_shape(const pulsar_model *m, pulsar_nemotron_h_shape *s);
/** The layer plan from the shape: PULSAR_LAYER_BLOCKS with each layer's mixer (MAMBA2 / ATTN_FULL) and FFN (MOE /
 *  NONE). */
void pulsar_nemotron_h_plan(const pulsar_nemotron_h_shape *s, pulsar_layer_plan *plan);

/* ---- 2. Weights ------------------------------------------------------------------------------------------------- */

/** One pulsar layer -- a trunk layer, or the drafter's (`mtp`) layer 0.  Bind roles in the comments; a slot the
 *  layer's kinds do not have stays NULL. */
typedef struct {
    /* the Mamba-2 mixer (PULSAR_MIXER_MAMBA2) */
    pulsar_tensor *ssm_pre_norm;     ///< ssm_pre_norm [n_embd]
    pulsar_tensor *ssm_in;           ///< ssm_in [n_embd -> z | x | B | C | dt] (d_inner + conv channels + heads)
    pulsar_tensor *ssm_conv;         ///< ssm_conv, ne [conv_k, conv channels]
    pulsar_tensor *ssm_conv_bias;    ///< ssm_conv_bias [conv channels]
    pulsar_tensor *ssm_dt_bias;      ///< ssm_dt_bias [heads]
    pulsar_tensor *ssm_a_log;        ///< ssm_a_log [heads]: A = -exp(A_log)
    pulsar_tensor *ssm_d;            ///< ssm_d [heads]: the skip
    pulsar_tensor *ssm_out_norm;     ///< ssm_out_norm [d_inner]: RMSNorm(y * silu(z)) per group of d_inner / groups
    pulsar_tensor *ssm_out;          ///< ssm_out [d_inner -> n_embd]
    /* the attention mixer (PULSAR_MIXER_ATTN_FULL) */
    pulsar_tensor *attn_norm;        ///< attn_norm [n_embd]
    pulsar_tensor *attn_q;           ///< attn_q [n_embd -> n_head * head_dim]
    pulsar_tensor *attn_k;           ///< attn_k [n_embd -> n_head_kv * head_dim]
    pulsar_tensor *attn_v;           ///< attn_v [n_embd -> n_head_kv * head_dim]
    pulsar_tensor *attn_o;           ///< attn_o [n_head * head_dim -> n_embd]
    /* the latent MoE (PULSAR_FFN_MOE) */
    pulsar_tensor *ffn_norm;         ///< ffn_norm [n_embd]
    pulsar_tensor *router;           ///< router [n_embd -> n_expert]
    pulsar_tensor *router_select_bias;   ///< router_select_bias f32 [n_expert]: selection only
    pulsar_tensor *latent_in;        ///< latent_in [n_embd -> latent]
    pulsar_tensor *latent_out;       ///< latent_out [latent -> n_embd]
    pulsar_tensor *exp_up;           ///< expert_up stack [latent, n_ff_exp, n_expert]
    pulsar_tensor *exp_down;         ///< expert_down stack [n_ff_exp, latent, n_expert]
    pulsar_tensor *sh_up;            ///< shared_up, index 0 [n_embd -> n_ff_shared]
    pulsar_tensor *sh_down;          ///< shared_down, index 0 [n_ff_shared -> n_embd]
} pulsar_nemotron_h_block;

/** The MTP module: one {attention, MoE} layer plus its input side and its out norm. */
typedef struct {
    pulsar_nemotron_h_block block;
    pulsar_tensor *embed_norm;       ///< draft_embed_norm [n_embd] (enorm)
    pulsar_tensor *hidden_norm;      ///< draft_hidden_norm [n_embd] (hnorm)
    pulsar_tensor *in_proj;          ///< draft_in_proj [2 n_embd -> n_embd] over [enorm(embed(t+1)) | hnorm(h_t)]
    pulsar_tensor *out_norm;         ///< draft_out_norm [n_embd] (final_layernorm)
} pulsar_nemotron_h_draft;

/** The family's state (pulsar_engine::family_state). */
typedef struct {
    pulsar_nemotron_h_shape shape;
    pulsar_tensor *embed;            ///< embed [n_vocab rows of n_embd]
    pulsar_tensor *output_norm;      ///< output_norm [n_embd] (norm_f)
    pulsar_tensor *head;             ///< head [n_embd -> n_vocab] (untied)
    pulsar_nemotron_h_block layer[PULSAR_FAMILY_MAX_LAYER];
    pulsar_nemotron_h_draft mtp;
    /** What the binding priced (role_bind.h): every byte the core stages, by role. */
    pulsar_role_census census;
    /** The tokenizer (nemotron_h_tokenizer.cpp): F5's shared BPE engine in its HF configuration (the "tekken"
     *  pre-tokenizer, ignore_merges), loaded at open from tokenizer.json + generation_config.json beside the
     *  container's shards; the chat front is src/lib/nemotron_chat (the engine's table here, the server's family
     *  table in src/server/family_nemotron_h.cpp). */
    struct bpe_tokenizer *tok;
} pulsar_nemotron_h_model;

/** The family's tokenizer table (nemotron_h_tokenizer.cpp) and its load: tokenizer.json and generation_config.json
 *  beside the shards of `model_path`.  false = refused (said). */
extern const pulsar_family_tokenizer PULSAR_NEMOTRON_H_TOKENIZER;
bool pulsar_nemotron_h_load_tokenizer(pulsar_nemotron_h_model *w, const char *model_path);

/** Bind every tensor the shape declares, admit its format, price it (nemotron_h_weights.cpp).  false = refused. */
bool pulsar_nemotron_h_bind(const pulsar_model *m, pulsar_nemotron_h_model *w);

/** The loaded family's state, or NULL on another family's engine. */
static inline const pulsar_nemotron_h_model *pulsar_nemotron_h_of(const pulsar_engine *e) {
    return e && e->family && e->family->id == PULSAR_FAMILY_ID_NEMOTRON_H
               ? (const pulsar_nemotron_h_model *)e->family_state
               : NULL;
}

#endif /* PULSAR_FAMILY_NEMOTRON_H_H */

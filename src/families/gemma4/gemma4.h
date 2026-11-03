/* gemma4.h -- the Gemma 4 family (L338): Google's Gemma 4 26B-A4B as a family module.
 *
 * THE MODEL (config.json text_config; HF transformers modeling_gemma4.py, the reference our BF16 grades ran): 30
 * layers in five blocks of 5 sliding-window (1024 keys) + 1 global attention layer, each
 *     r = h;  h = r + post_attention_norm( o( attn(input_norm(r)) ) )
 *     r = h;  d = post_ffn_norm_1( mlp(pre_ffn_norm(r)) )                 the dense GeGLU MLP
 *             m = post_ffn_norm_2( experts_{router(r)}(pre_ffn_norm_2(r)) )   128 routed GeGLU experts, top 8
 *             h = (r + post_ffn_norm(d + m)) * layer_scalar               the scalar scales the whole stream
 * with attention scale 1 (q / k RMS-normed per head, v RMS-normed without a weight), a sliding layer's 8 kv heads of
 * 256 under the default rope (theta 1e4), a global layer's 2 kv heads of 512 with K = V (v is k_proj's output before
 * k_norm) under the proportional rope (64 of 256 frequency pairs rotate at theta 1e6 over the 512 head, the rest pass
 * through), the router's input rmsnorm(r) (no weight) * scale * hidden^-1/2 -> softmax -> top 8 -> renormalised ->
 * per_expert_scale; GELU tanh; an embedding scaled by sqrt(hidden) (bf16-rounded); a head tied to the embedding with a
 * final logit soft-cap (tanh(x / 30) * 30) -- run here as the head exllamav3 quantizes from the embedding (6 bpw, the
 * pack's `lm_head`; Tyler 2026-11-01: pulsar's BF16 head GEMM needs sm_80+, the Turing build runs the EXL3 one).  RMSNorm is x * (mean(x^2) + eps)^-1/2 * w (w, not 1 + w).
 *
 * WHAT IS HERE (E2): the shape struct (read from the container's gemma4.* keys, every geometry field pinned to the
 * known profile, every switch pinned, any unknown key refused), the layer plan (PULSAR_LAYER_BLOCKS: window vs full
 * attention, PULSAR_FFN_DENSE_MOE on every layer), the role-driven binding of every tensor with its format admitted
 * through the core's front doors, and the byte census; the state declaration and the driver ops (E4,
 * gemma4_forward.cpp); the tokenizer table and chat front (gemma4_tokenizer.cpp; the server's family table is
 * src/server/family_gemma4.cpp).
 *
 * THE CONTAINER CONTRACT (tools/container/gemma4.py emits, this family reads):
 *   general.architecture = "gemma4";
 *   gemma4.<key>   text_config verbatim (nested dicts flattened with '.', None skipped);
 *   pulsar.vision_present (false in v1: the tower is not written).
 *   Every tensor carries a bind record: role (+ block `layers`, layer).  Linears are stored [in, out] in ne order;
 *   routed experts are pulsar.experts stacks [in, out, n_expert], gate and up apart (the builder splits HF's fused
 *   stack per expert, as exllamav3's pack holds them); the head is the pack's EXL3 `lm_head.weight`.  The dense MLP and the experts are declared at the EXL3 pack's
 *   STORED width -- exl3_out_stored(2112) = 2176, exl3_out_stored(704) = 768 -- whose padded gate / up columns decode
 *   to exactly zero (the builder checks svh = 0 there), so the stored width is the model's width.
 */
/** @file gemma4.h
 *  @brief The Gemma 4 family module's declarations: shape, weights and the family's state. */
#ifndef PULSAR_FAMILY_GEMMA4_H
#define PULSAR_FAMILY_GEMMA4_H

#include "engine/pulsar_engine_internal.h"
#include "engine/role_bind.h"
#include "engine/exl3_trellis.h"

/// The `general.architecture` value this family claims.
#define PULSAR_GEMMA4_ARCH "gemma4"

/** The family descriptor (gemma4_family.cpp); family.cpp registers it. */
extern const pulsar_family PULSAR_FAMILY_GEMMA4;

/* ---- 1. Shape -------------------------------------------------------------------------------------------------- */

/** The loaded shape.  The geometry is checked field by field against PULSAR_GEMMA4_SHAPE_26B_A4B (the kernels the
 *  blocks run are declared for these numbers; a second geometry is a second profile, added on purpose); the PLAN
 *  facts -- how many layers, which are global -- are the container's. */
typedef struct {
    /// The profile's display name ("Gemma-4-26B-A4B"); the boot line and model_name print it.
    const char *name;
    /* geometry: pinned */
    uint32_t n_embd;           ///< hidden_size (2816)
    uint32_t n_vocab;          ///< vocab_size (262144): the embed rows, the tied head's rows, the logits width
    uint32_t n_head;           ///< num_attention_heads (16), every layer
    uint32_t swa_n_head_kv;    ///< num_key_value_heads (8), a sliding layer
    uint32_t swa_head_dim;     ///< head_dim (256), a sliding layer
    uint32_t n_head_kv;        ///< num_global_key_value_heads (2), a global layer
    uint32_t head_dim;         ///< global_head_dim (512), a global layer
    uint32_t window;           ///< sliding_window (1024): a sliding layer's keys, self included
    uint32_t n_ff;             ///< intermediate_size (2112): the dense MLP (stored exl3_out_stored = 2176)
    uint32_t n_ff_exp;         ///< moe_intermediate_size (704): a routed expert (stored 768)
    uint32_t n_expert;         ///< num_experts (128)
    uint32_t n_expert_used;    ///< top_k_experts (8)
    float rms_eps;             ///< rms_norm_eps (1e-6)
    float softcap;             ///< final_logit_softcapping (30): logits = tanh(x / 30) * 30
    float swa_rope_theta;      ///< rope_parameters.sliding_attention.rope_theta (1e4), the default rope over 256
    float rope_theta;          ///< rope_parameters.full_attention.rope_theta (1e6), the proportional rope over 512
    float rope_partial;        ///< rope_parameters.full_attention.partial_rotary_factor (0.25): 64 of 256 pairs rotate
    uint64_t max_context;      ///< max_position_embeddings (262144)
    uint32_t bos_id;           ///< bos_token_id (2)
    uint32_t eos_id;           ///< eos_token_id (1)
    uint32_t pad_id;           ///< pad_token_id (0)
    /* the plan: the container's */
    uint32_t n_layer;          ///< num_hidden_layers
    uint64_t full_layers;      ///< bit il = layer_types[il] is "full_attention" (the rest "sliding_attention")
    bool vision_written;       ///< pulsar.vision_present: the container holds the tower (refused in v1)
} pulsar_gemma4_shape;

/// The one profile (Gemma 4 26B-A4B): the geometry every container must equal.
extern const pulsar_gemma4_shape PULSAR_GEMMA4_SHAPE_26B_A4B;

/** The dense MLP's width as the container stores it (the EXL3 pack's 128-padded width; gemma4.h header). */
/// @param s  the shape
/// @return exl3_out_stored(n_ff)
static inline uint32_t pulsar_gemma4_ff_stored(const pulsar_gemma4_shape *s) { return (uint32_t)exl3_out_stored(s->n_ff); }
/** A routed expert's width as the container stores it. */
/// @param s  the shape
/// @return exl3_out_stored(n_ff_exp)
static inline uint32_t pulsar_gemma4_ff_exp_stored(const pulsar_gemma4_shape *s) {
    return (uint32_t)exl3_out_stored(s->n_ff_exp);
}
/** Whether layer `il` is a global (full-attention) layer. */
/// @param s   the shape
/// @param il  the layer
/// @return true for a global layer
static inline bool pulsar_gemma4_is_full(const pulsar_gemma4_shape *s, uint32_t il) { return s->full_layers >> il & 1u; }

/** Read the shape from the container's keys and check it (gemma4_shape.cpp).  Every refusal is said and counted
 *  (pulsar_load_refuse); false = at least one. */
bool pulsar_gemma4_read_shape(const pulsar_model *m, pulsar_gemma4_shape *s);
/** The layer plan from the shape: PULSAR_LAYER_BLOCKS, window or full attention, PULSAR_FFN_DENSE_MOE. */
void pulsar_gemma4_plan(const pulsar_gemma4_shape *s, pulsar_layer_plan *plan);

/* ---- 2. Weights ------------------------------------------------------------------------------------------------- */

/** L338 P2: the largest LoRA rank the runtime adapter takes (the scratch is sized before any engine exists: the LoRA
 *  workspace holds max_rows x this many f32); a container's larger rank is refused at load. */
#define PULSAR_GEMMA4_LORA_RANK_MAX 64u

/** L338 P2: one targeted projection's LoRA pair (adapters.py: adapter.layers.N.<proj>.lora_{A,B}), applied at runtime
 *  as y += scale * B (A x) at the projection's LOGICAL widths -- A f32 [r][in] (ne in, r), B f32 [out][r] (ne r, out);
 *  both NULL when the container carries no adapter or the layer has no such projection. */
typedef struct {
    pulsar_tensor *a;   ///< lora_A: f32 [r][in]
    pulsar_tensor *b;   ///< lora_B: f32 [out][r]
} pulsar_gemma4_lora;

/** One layer.  Bind roles in the comments. */
typedef struct {
    pulsar_tensor *attn_norm;           ///< attn_norm [n_embd] (input_layernorm)
    pulsar_tensor *attn_q;              ///< attn_q [n_embd -> n_head * hd]
    pulsar_tensor *attn_k;              ///< attn_k [n_embd -> n_kv * hd]
    pulsar_tensor *attn_v;              ///< attn_v [n_embd -> n_kv * hd]; NULL on a global layer (K = V)
    pulsar_tensor *attn_o;              ///< attn_o [n_head * hd -> n_embd]
    pulsar_tensor *attn_q_norm;         ///< attn_q_norm [hd]
    pulsar_tensor *attn_k_norm;         ///< attn_k_norm [hd]
    pulsar_tensor *attn_post_norm;      ///< attn_post_norm [n_embd] (post_attention_layernorm)
    /* the dense MLP */
    pulsar_tensor *ffn_pre_norm;        ///< ffn_pre_norm [n_embd] (pre_feedforward_layernorm)
    pulsar_tensor *ffn_gate;            ///< ffn_gate [n_embd -> ff_stored]
    pulsar_tensor *ffn_up;              ///< ffn_up [n_embd -> ff_stored]
    pulsar_tensor *ffn_down;            ///< ffn_down [ff_stored -> n_embd]
    pulsar_tensor *ffn_post_norm_1;     ///< ffn_post_norm_1 [n_embd]: the MLP's output norm
    /* the MoE */
    pulsar_tensor *moe_pre_norm;        ///< moe_pre_norm [n_embd] (pre_feedforward_layernorm_2)
    pulsar_tensor *router;              ///< router [n_embd -> n_expert]
    pulsar_tensor *router_scale;        ///< router_scale [n_embd]: the router input's per-channel scale
    pulsar_tensor *router_expert_scale; ///< router_expert_scale [n_expert]: each chosen expert's weight scale
    pulsar_tensor *exp_gate;            ///< expert_gate stack [n_embd, ff_exp_stored, n_expert]
    pulsar_tensor *exp_up;              ///< expert_up stack
    pulsar_tensor *exp_down;            ///< expert_down stack [ff_exp_stored, n_embd, n_expert]
    pulsar_tensor *ffn_post_norm_2;     ///< ffn_post_norm_2 [n_embd]: the experts' output norm
    pulsar_tensor *ffn_post_norm;       ///< ffn_post_norm [n_embd]: of the sum (post_feedforward_layernorm)
    pulsar_tensor *layer_scalar;        ///< layer_scalar [1]: the whole stream's scale at the layer's end
    /* L338 P2: the adapter's LoRA pairs on the projections it targets (zero without an adapter) */
    pulsar_gemma4_lora lora_q;          ///< attn_q_lora_a / _b
    pulsar_gemma4_lora lora_k;          ///< attn_k_lora_a / _b
    pulsar_gemma4_lora lora_v;          ///< attn_v_lora_a / _b (none on a global layer)
    pulsar_gemma4_lora lora_o;          ///< attn_o_lora_a / _b
    pulsar_gemma4_lora lora_gate;       ///< ffn_gate_lora_a / _b (logical width n_ff, not the stored)
    pulsar_gemma4_lora lora_up;         ///< ffn_up_lora_a / _b
    pulsar_gemma4_lora lora_down;       ///< ffn_down_lora_a / _b
} pulsar_gemma4_block;

/** The family's state (pulsar_engine::family_state). */
typedef struct {
    /// The loaded, checked shape (pulsar_gemma4_read_shape) every bind and forward reads.
    pulsar_gemma4_shape shape;
    pulsar_tensor *embed;               ///< embed [n_vocab rows of n_embd]: the lookups
    pulsar_tensor *output_norm;         ///< output_norm [n_embd]
    pulsar_tensor *head;                ///< head [n_embd -> n_vocab]: the pack's EXL3 head (quantized from the tied embedding)
    /// The layers; entries [0, shape.n_layer) are bound, the rest stay zero.
    pulsar_gemma4_block layer[PULSAR_FAMILY_MAX_LAYER];
    /** What the binding priced (role_bind.h): every byte the core stages, by role. */
    pulsar_role_census census;
    /** The digest a session's disk segments are keyed by: the directory's names, types and sizes (FNV-1a). */
    uint64_t artifact_digest;
    /** The tokenizer table (gemma4_tokenizer.cpp): the shared BPE engine's SentencePiece-BPE row, from the container's
     *  tokenizer.json + generation_config.json.  Owned; freed by unload. */
    struct bpe_tokenizer *tok;
    /* L338 P2: the runtime adapter (pulsar.adapter in the container; adapters.py) -- absent: every field zero */
    bool adapter;                       ///< the container carries a LoRA adapter (pulsar.adapter = "jev")
    uint32_t lora_r;                    ///< its rank (pulsar.adapter.lora_r, <= PULSAR_GEMMA4_LORA_RANK_MAX)
    float lora_scale;                   ///< alpha / r (pulsar.adapter.lora_alpha / lora_r): y += scale * B (A x)
    pulsar_tensor *decision_head;       ///< decision_head: f32 [slots][n_embd] (ne n_embd, slots), read on the host
    pulsar_tensor *decision_head_bias;  ///< decision_head_bias: f32 [slots]
    uint32_t decision_slots;            ///< the head's slots
    /* L338 P3: the block drafter (pulsar.drafter in the container: z-lab's plain DFlash; spec_block.h) when the open
     * loads it -- NULL: none in the container, or --drafter none */
    struct pulsar_block_drafter *block; ///< owned (a plain record; free() is its teardown)
    bool block_written;                 ///< the container carries a block drafter (loaded or not)
} pulsar_gemma4_model;

/** The family's tokenizer table and chat front (gemma4_tokenizer.cpp over src/lib/gemma4_chat). */
extern const pulsar_family_tokenizer PULSAR_GEMMA4_TOKENIZER;
/** Load the tokenizer from tokenizer.json + generation_config.json beside the shards and check it against the shape:
 *  the "gemma4" pipeline, the eos among the stop ids, n_vocab ids (gemma4_tokenizer.cpp).
 *  @param w           the family state: its shape is read; tok is written
 *  @param model_path  the container's directory, or a shard path inside it
 *  @return true = loaded; false = refused (said) */
bool pulsar_gemma4_load_tokenizer(pulsar_gemma4_model *w, const char *model_path);

/** Bind every tensor the shape declares, admit its format, price it (gemma4_weights.cpp).  false = refused. */
bool pulsar_gemma4_bind(const pulsar_model *m, pulsar_gemma4_model *w);

/** The loaded family's state, or NULL on another family's engine. */
/** @param e  any engine (NULL allowed)
 *  @return the Gemma 4 state, or NULL */
static inline const pulsar_gemma4_model *pulsar_gemma4_of(const pulsar_engine *e) {
    return e && e->family && e->family->id == PULSAR_FAMILY_ID_GEMMA4 ? (const pulsar_gemma4_model *)e->family_state
                                                                      : NULL;
}

/* ---- 3. The forward on the core row-step driver (gemma4_forward.cpp) ------------------------------------------ */

/** The ids of the family's state declarations (state_kinds.h kinds), in the plan's table order: a window layer carries
 *  KV_WIN, a global layer KV_FULL. */
enum pulsar_gemma4_state_id {
    PULSAR_GEMMA4_STATE_KV_WIN = 0,   ///< RING of window slots: a window layer's K | V rows (F1's BF16 rows, 8 x 256)
    PULSAR_GEMMA4_STATE_KV_FULL,      ///< POOL, one token a row: a global layer's K | V rows (2 x 512)
    PULSAR_GEMMA4_STATE_COUNT
};

/** Build the family's state declaration into e->state_plan (the load, after the plan and the bind). */
void pulsar_gemma4_state_plan(pulsar_engine *e);

/** The family's ops on the core row-step driver (step_driver.h): block[PULSAR_LAYER_BLOCKS] composes each layer. */
extern const pulsar_driver_ops PULSAR_GEMMA4_DRIVER;

#endif /* PULSAR_FAMILY_GEMMA4_H */

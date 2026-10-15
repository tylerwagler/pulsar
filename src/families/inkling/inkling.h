/* inkling.h -- the Inkling family (L294 / L295 E1): Thinking Machines' Inkling-Small as a family module.
 *
 * The first family built in the plugin layout (src/families/<name>/, roadmap "The plugin contract"): it declares
 * its shape and its tensor roles, and binds through the core's role-driven binder (src/engine/role_bind.h) by the
 * bind records the builder writes (tools/container/inkling.py) -- no name is parsed.  Outside this directory the
 * family is one registry row (family.cpp), one id (pulsar.h) and its gate.
 *
 * THE MODEL (config.json text_config; plan 120): 42 layers of
 *     x += SConv_attn( wo_ud . GQA(q, SConv_k(k), SConv_v(v); relative bias r . P, log-scaling) )
 *     x += SConv_ffn ( dense MLP (layers < dense_mlp_idx) | MoE (sigmoid top-6 of 256 + 2 sink shared experts) )
 * with 35 sliding-window (512) layers and 7 global ones (`local_layer_ids` lists the window layers), no RoPE, an
 * embed RMSNorm, untied embed / unembed and muP logits (/16); 8 chained MTP layers; an hmlp image tower.
 *
 * WHAT IS HERE (E1): the shape struct (read from the container's inkling.* keys, every geometry field pinned to the
 * known profile, every switch pinned, any unknown key refused), the layer plan (PULSAR_LAYER_BLOCKS with a neutral
 * pulsar_layer_desc per layer: window vs full attention, dense vs MoE), the role-driven binding of every tensor with
 * its format admitted through the core's front doors, and the byte census.  The forward is NOT here: it runs on the
 * core row-step driver (L295 F4) over the shared blocks -- F1 attention, F2 router, F3 state kinds, E3 SConv, E5
 * head.  Until F4 lands a session refuses by name (inkling_family.cpp).
 *
 * THE CONTAINER CONTRACT (tools/container/inkling.py emits, this family reads):
 *   general.architecture = "inkling";
 *   inkling.<key>         text_config verbatim (nested dicts flattened with '.', None skipped);
 *   inkling.vision.<key>  vision_config, present when the recipe writes the tower;
 *   inkling.mtp.<key>     mtp_config, present when the checkpoint has one;
 *   inkling.eos_token_id; pulsar.mtp_present / pulsar.vision_present (whether the drafter / tower are written).
 *   Every tensor carries a bind record: role (+ block `layers` / `mtp`, layer, index).  Linears are stored [in, out]
 *   in ne order; the short convs [kernel, channels]; routed experts are pulsar.experts stacks [in, out, n_expert],
 *   gate and up apart (the builder de-interleaves the checkpoint's w13; the loader never does).
 */
#ifndef PULSAR_FAMILY_INKLING_H
#define PULSAR_FAMILY_INKLING_H

#include "engine/pulsar_engine_internal.h"
#include "engine/role_bind.h"

#define PULSAR_INKLING_ARCH "inkling"
/** The most MTP layers a drafter block can hold (Inkling-Small has 8) and shared experts a MoE layer (2). */
#define PULSAR_INKLING_MTP_MAX 8u
#define PULSAR_INKLING_SHARED_MAX 2u
/** The hmlp tower's linears (Inkling-Small: 4). */
#define PULSAR_INKLING_VISION_MAX 4u

/** The family descriptor (inkling_family.cpp); family.cpp registers it. */
extern const pulsar_family PULSAR_FAMILY_INKLING;

/* ---- 1. Shape -------------------------------------------------------------------------------------------------- */

/** The loaded shape.  The geometry is checked field by field against PULSAR_INKLING_SHAPE_SMALL (the kernels the
 *  blocks run are declared for these numbers; a second geometry is a second profile, added on purpose); the PLAN
 *  facts -- how many layers, which are windowed, which are dense, the drafter's -- are the container's (the 3-layer
 *  mini and the 42-layer model are one profile). */
typedef struct {
    const char *name;
    /* geometry: pinned */
    uint32_t n_embd;           ///< hidden_size (4096)
    uint32_t n_vocab;          ///< vocab_size (201024): the embed rows and the logits width
    uint32_t n_vocab_real;     ///< unpadded_vocab_size (200058): the ids past it are padding the head masks (E5)
    uint32_t n_head;           ///< num_attention_heads (32), a global layer
    uint32_t n_head_kv;        ///< num_key_value_heads (8)
    uint32_t head_dim;         ///< head_dim (128)
    uint32_t rel_extent;       ///< rel_extent (1024): a global layer's relative-bias distances
    uint32_t swa_n_head;       ///< swa_num_attention_heads (32), a window layer
    uint32_t swa_n_head_kv;    ///< swa_num_key_value_heads (8)
    uint32_t swa_head_dim;     ///< swa_head_dim (128)
    uint32_t window;           ///< sliding_window_size (512): a window layer's keys AND its relative extent
    uint32_t d_rel;            ///< d_rel (16): the relative query's width per head
    uint32_t conv_k;           ///< sconv_kernel_size (4)
    uint32_t n_ff_dense;       ///< dense_intermediate_size (16384)
    uint32_t n_ff_exp;         ///< intermediate_size (2048): a routed / shared expert's width
    uint32_t n_expert;         ///< n_routed_experts (256)
    uint32_t n_expert_used;    ///< num_experts_per_tok (6)
    uint32_t n_shared;         ///< n_shared_experts (2): sink-gated by the router's last rows
    float rms_eps;             ///< rms_norm_eps (1e-6)
    float route_scale;         ///< route_scale (8)
    float logit_div;           ///< logits_mup_width_multiplier (16): logits / 16
    float log_scale_alpha;     ///< log_scaling_alpha (0.1)
    uint32_t log_scale_floor;  ///< log_scaling_n_floor (128000): tau = 1 + alpha ln(max(1, (pos + 1) / floor))
    uint64_t max_context;      ///< model_max_length (1048576)
    uint32_t eos_id;           ///< eos_token_id (200006)
    /* the tower's (pinned when written) */
    uint32_t vis_patch;        ///< patch_size (40)
    uint32_t vis_temporal;     ///< temporal_patch_size (2)
    uint32_t vis_channels;     ///< n_channels (3)
    uint32_t vis_layers;       ///< n_layers (4): the hmlp tower's linears
    /** The tower's linear widths, in -> out per layer: vLLM's plan_out_scales (inkling/common/towers.py) for the
     *  pinned patch / temporal / channels / layers above -- each layer folds a (t, h, w) block into channels and
     *  rounds the width up to 64: (5x5x3 = 75 -> 128), (2x2 -> 512 -> 320), (4x4 -> 5120 -> 4800), (t 2 -> 9600 ->
     *  n_embd).  The config names none of them; they follow from the pinned keys, so the profile carries them. */
    uint32_t vis_in[PULSAR_INKLING_VISION_MAX];
    uint32_t vis_mid[PULSAR_INKLING_VISION_MAX - 1];   ///< the inner outputs; the last linear writes n_embd
    uint32_t n_mtp_max;       ///< num_nextn_predict_layers of the profile (8); the loaded value is n_mtp
    /* the plan: the container's */
    uint32_t n_layer;          ///< num_hidden_layers
    uint32_t n_dense;          ///< dense_mlp_idx: layers [0, n_dense) run the dense MLP, the rest the MoE
    uint64_t window_layers;    ///< bit il = local_layer_ids holds il
    uint32_t n_mtp;            ///< mtp_config.num_nextn_predict_layers (0: no mtp_config)
    uint64_t mtp_window_layers;///< bit n = mtp_config.local_layer_ids holds n
    bool mtp_written;          ///< pulsar.mtp_present: the container holds the drafter
    bool vision_written;       ///< pulsar.vision_present: the container holds the tower
} pulsar_inkling_shape;

extern const pulsar_inkling_shape PULSAR_INKLING_SHAPE_SMALL;

/** Read the shape from the container's keys and check it (inkling_shape.cpp).  Every refusal is said and counted
 *  (pulsar_load_refuse); false = at least one. */
bool pulsar_inkling_read_shape(const pulsar_model *m, pulsar_inkling_shape *s);
/** The layer plan from the shape: PULSAR_LAYER_BLOCKS with each layer's mixer / FFN / window. */
void pulsar_inkling_plan(const pulsar_inkling_shape *s, pulsar_layer_plan *plan);

/* ---- 2. Weights ------------------------------------------------------------------------------------------------- */

/** One transformer block -- a trunk layer, or the drafter's (`mtp`) layer n.  Bind roles in the comments. */
typedef struct {
    pulsar_tensor *attn_norm;        ///< attn_norm [n_embd]
    pulsar_tensor *attn_q;           ///< attn_q [n_embd -> n_head * head_dim]
    pulsar_tensor *attn_k;           ///< attn_k [n_embd -> n_head_kv * head_dim]
    pulsar_tensor *attn_v;           ///< attn_v [n_embd -> n_head_kv * head_dim]
    pulsar_tensor *attn_rel_proj;    ///< attn_rel_proj [n_embd -> n_head * d_rel]: the relative query r
    pulsar_tensor *attn_o;           ///< attn_o [n_head * head_dim -> n_embd]
    pulsar_tensor *attn_q_norm;      ///< attn_q_norm [head_dim]
    pulsar_tensor *attn_k_norm;      ///< attn_k_norm [head_dim]
    pulsar_tensor *attn_rel_basis;   ///< attn_rel_basis P, ne [extent, d_rel]: bias(dist) = r . P[:, dist]
    pulsar_tensor *conv_k;           ///< short_conv_k, ne [conv_k, n_head_kv * head_dim]
    pulsar_tensor *conv_v;           ///< short_conv_v, ne [conv_k, n_head_kv * head_dim]
    pulsar_tensor *conv_attn_out;    ///< short_conv_attn_out, ne [conv_k, n_embd]
    pulsar_tensor *conv_ffn_out;     ///< short_conv_ffn_out, ne [conv_k, n_embd]
    pulsar_tensor *ffn_norm;         ///< ffn_norm [n_embd]
    /* the dense MLP (PULSAR_FFN_DENSE) */
    pulsar_tensor *ffn_gate;         ///< ffn_gate [n_embd -> n_ff_dense]
    pulsar_tensor *ffn_up;           ///< ffn_up [n_embd -> n_ff_dense]
    pulsar_tensor *ffn_down;         ///< ffn_down [n_ff_dense -> n_embd]
    pulsar_tensor *ffn_out_scale;    ///< ffn_out_scale [1]
    /* the MoE (PULSAR_FFN_MOE) */
    pulsar_tensor *router;           ///< router [n_embd -> n_expert + n_shared]: routed logits, then the sink rows
    pulsar_tensor *router_select_bias;   ///< router_select_bias f32 [n_expert]: selection only
    pulsar_tensor *router_out_scale;     ///< router_out_scale f32 [1]
    pulsar_tensor *exp_gate;         ///< expert_gate stack [n_embd, n_ff_exp, n_expert]
    pulsar_tensor *exp_up;           ///< expert_up stack
    pulsar_tensor *exp_down;         ///< expert_down stack [n_ff_exp, n_embd, n_expert]
    pulsar_tensor *sh_gate[PULSAR_INKLING_SHARED_MAX];   ///< shared_gate, index j
    pulsar_tensor *sh_up[PULSAR_INKLING_SHARED_MAX];     ///< shared_up, index j
    pulsar_tensor *sh_down[PULSAR_INKLING_SHARED_MAX];   ///< shared_down, index j
} pulsar_inkling_block;

/** One MTP layer: a dense block plus its input side. */
typedef struct {
    pulsar_inkling_block block;
    pulsar_tensor *embed_norm;       ///< draft_embed_norm [n_embd]
    pulsar_tensor *hidden_norm;      ///< draft_hidden_norm [n_embd]
    pulsar_tensor *in_proj;          ///< draft_in_proj [2 n_embd -> n_embd]: [hidden_norm(h) | embed_norm(e)]
} pulsar_inkling_draft_layer;

/** The family's state (pulsar_engine::family_state). */
typedef struct {
    pulsar_inkling_shape shape;
    pulsar_tensor *embed;            ///< embed [n_vocab rows of n_embd]
    pulsar_tensor *embed_norm;       ///< embed_norm [n_embd]
    pulsar_tensor *output_norm;      ///< output_norm [n_embd]
    pulsar_tensor *head;             ///< head [n_embd -> n_vocab] (untied)
    pulsar_inkling_block layer[PULSAR_FAMILY_MAX_LAYER];
    pulsar_inkling_draft_layer mtp[PULSAR_INKLING_MTP_MAX];
    pulsar_tensor *vis_linear[PULSAR_INKLING_VISION_MAX];     ///< vision_linear, index i
    pulsar_tensor *vis_norm[PULSAR_INKLING_VISION_MAX - 1];   ///< vision_norm, index i (between linears)
    pulsar_tensor *vis_final_norm;   ///< vision_final_norm [n_embd]
    /** What the binding priced (role_bind.h): every byte the core stages, by role. */
    pulsar_role_census census;
    /** The frontend seam (L295 F5 / E6): the tokenizer the family's tokenizer table will dispatch to -- F5's shared
     *  BPE engine (src/lib/bpe_tokenizer.h on work/l295-f5-frontend) in its tiktoken configuration, loaded at open
     *  from the container directory's tokenizer files, with Inkling's renderer and output parser
     *  (src/lib/inkling_chat.h) behind the server's family table.  NULL until F5 lands: the family declares no
     *  PULSAR_FAMILY_CAP_CHAT and no tokenizer table, so the front ends refuse the engine at startup by name
     *  (pulsar_engine_has_tokenizer); eos_id above is the stop id that table's eos() answers. */
    struct bpe_tokenizer *tok;
} pulsar_inkling_model;

/** Bind every tensor the shape declares, admit its format, price it (inkling_weights.cpp).  false = refused. */
bool pulsar_inkling_bind(const pulsar_model *m, pulsar_inkling_model *w);

/** The loaded family's state, or NULL on another family's engine. */
static inline const pulsar_inkling_model *pulsar_inkling_of(const pulsar_engine *e) {
    return e && e->family && e->family->id == PULSAR_FAMILY_ID_INKLING ? (const pulsar_inkling_model *)e->family_state
                                                                       : NULL;
}

#endif /* PULSAR_FAMILY_INKLING_H */

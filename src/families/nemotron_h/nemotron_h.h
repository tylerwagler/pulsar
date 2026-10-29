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
 * the byte census, and the tokenizer + chat front (N8, nemotron_h_tokenizer.cpp).  THE FORWARD (N3-N7,
 * nemotron_h_forward.cpp): the family's ops on the core row-step driver -- block[PULSAR_LAYER_BLOCKS] composes each
 * layer from its pulsar_layer_desc over shared blocks (the Mamba-2 block N2, F1 attention, F2's router with the
 * 512 / 22 instance, the rows MoE front door's latent wrapper and relu^2 option N4, E5's head), and the declared
 * state (N3: SSM FIXED / DELTA, conv FIXED / SNAPSHOT, attention KV POOL; the drafter slot's KV POOL and parked row).
 * THE DRAFTER (N9): the MTP module declared to the core's chained-MTP drafter (src/engine/spec_mtp.h) -- the family
 * says what one module step is (eh_proj over [enorm(embed) | hnorm(norm_f(h))], its {attention, MoE} on the drafter
 * slot, final_layernorm, the target's head); the lane drafts it recursively, the driver's verify target verifies.
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
 *   pulsar.mtp_present (the container holds the drafter), pulsar.vision_present (the tower: section 2b, L331).
 *   Every tensor carries a bind record: role (+ block `layers` / `mtp`, layer = the PULSAR layer, index).  Linears
 *   are stored [in, out] in ne order; the conv [kernel, channels]; routed experts are pulsar.experts stacks
 *   [in, out, n_expert] (up: latent -> ff, down: ff -> latent).
 */
#ifndef PULSAR_FAMILY_NEMOTRON_H_H
#define PULSAR_FAMILY_NEMOTRON_H_H

#include "engine/pulsar_engine_internal.h"
#include "engine/role_bind.h"
#include "engine/spec_block.h"
#include "engine/vit_encoder.h"

#include <vector>

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
    bool vision_written;       ///< L331 pulsar.vision_present: the container holds the vision tower
    bool block_written;        ///< L298 pulsar.drafter = "lilicorr": the container holds the LiLiCorr block drafter
    /** L298: HF block b's place in the plan -- pulsar layer block_layer[b], its mixer (PULSAR_BLOCK_TAP_MIXER) or its
     *  MoE (PULSAR_BLOCK_TAP_FFN): where a block drafter's tap after block b reads the residual */
    uint8_t block_layer[PULSAR_NEMOTRON_H_MAX_BLOCKS];
    uint8_t block_sub[PULSAR_NEMOTRON_H_MAX_BLOCKS];
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

/* ---- 2b. The vision tower (L331; nemotron_h_vision.cpp, nemotron_h_image.cpp) ---------------------------------------
 *
 * THE MODEL (the checkpoint's modeling_radio.py + modeling_nemotron_h_omni.py, image_processing_nemotron_h_omni.py):
 *   image   decode (PNG / JPEG, the engine's codec), the processor's tile: (w_p, h_p) patches inside
 *           [min_patches, max_patches] with the aspect ratio, each side even (pulsar_nemotron_h_tile); torch's
 *           antialiased bicubic to 16 w_p x 16 h_p when the size changes (vision_torch_bicubic_aa_rgb); (p / 255 - mean)
 *           / std in f32; bf16 (the tower's dtype); 16 x 16 patches, element c * 256 + y * 16 + x
 *   tower   C-RADIO ViT-H/16: res = [10 cls / register tokens | patch Linear 768 -> 1280 (no bias) + CPE positions]
 *           (the 128 x 128 table bilinear-resized to M x M, M = max(h_p, w_p), the top-left h_p x w_p crop); 32 timm
 *           blocks (engine/vit_encoder.h: 16 heads of 80, MLP 5120, LayerNorm 1e-6)
 *   project the 10 prefix rows dropped; vision_final_layernorm (LayerNorm 1e-6: the checkpoint's own path -- L331);
 *           pixel shuffle 2x2 (LLM row (i, j) = patches (2i, 2j), (2i, 2j+1), (2i+1, 2j), (2i+1, 2j+1), 4 x 1280);
 *           mlp1 = RMSNorm(5120, 1e-5) -> Linear 5120 -> 20480 -> relu^2 -> Linear 20480 -> 4096 (no biases)
 *   prompt  `<img>` + (h_p w_p / 4) x `<image>` + `</img>`; the BLOCK is the run of `<image>`; NoPE: no positions.
 * The patches run in MERGE-BLOCK order (llm row (i, j), then a, b): the ViT has no relative position, so its rows are
 * a permutation of the reference's, and the pixel shuffle is then a view.
 *
 * THE CONTAINER: pulsar.vision_present true, the facts under nemotron_h_vision.* (pinned to the profile below), the
 * tensors by bind record -- the ViT's as Parse's (vit_patch, vit_cls, vit_pos at the top; vit_norm1 .. vit_fc2_bias in
 * block `vit`), the projector's vit_final_norm(_bias), proj_norm, proj_fc1, proj_fc2: Linears and the positions /
 * class tokens / proj_norm BF16 as stored, every LayerNorm weight and bias f32 (the shared row ops' operand). */
#define PULSAR_NEMOTRON_H_VIT_DEPTH 32u

/** The tower's pinned facts (nemotron_h_vision.*). */
typedef struct {
    uint32_t width, depth, heads, head_dim, mlp, patch, prefix, pos_grid;
    float eps;
    uint32_t merge, proj_hidden;
    float proj_eps;
    uint32_t min_patches, max_patches;
    float mean[3], std[3];
} pulsar_nemotron_h_vision_shape;

extern const pulsar_nemotron_h_vision_shape PULSAR_NEMOTRON_H_VISION_SHAPE;

struct pulsar_nemotron_h_vision_run;

/** The tower's tensors and its state (bound when the container holds it). */
typedef struct {
    bool present;                    ///< the container holds the tower (pulsar.vision_present) and it bound
    pulsar_nemotron_h_vision_shape shape;
    pulsar_tensor *patch, *cls, *pos;
    pulsar_vit_block vit[PULSAR_NEMOTRON_H_VIT_DEPTH];
    pulsar_tensor *final_norm, *final_norm_b, *proj_norm, *proj_fc1, *proj_fc2;
    int32_t image_id, start_id, end_id;   ///< `<image>`, `<img>`, `</img>` (the tokenizer's added tokens)
    std::vector<float> pos_field;    ///< the position table, host f32 [pos_grid][pos_grid][width]
    std::vector<float> cls_rows;     ///< the prefix rows, host f32 [prefix][width]
    struct pulsar_nemotron_h_vision_run *run;
} pulsar_nemotron_h_vision;

/** The processor's tile for a w x h image: (*wp, *hp) patches -- _compute_target_patches at the per-image budget
 *  (max_patches), each side even. */
void pulsar_nemotron_h_tile(const pulsar_nemotron_h_vision_shape *s, int w, int h, uint32_t *wp, uint32_t *hp);

/** One image ready for the tower. */
typedef struct {
    uint32_t wp, hp;                 ///< the tile in patches
    std::vector<float> pixels;       ///< the processor's pixel_values, f32 [3][16 hp][16 wp]
    std::vector<uint16_t> patches;   ///< bf16 [wp hp][768], merge-block order
} pulsar_nemotron_h_image;

/** Decode, tile, resize, normalise, patch.  false + `err` on an image the tower does not take. */
bool pulsar_nemotron_h_image_prepare(const pulsar_nemotron_h_vision_shape *s, const uint8_t *bytes, size_t len,
                                     pulsar_nemotron_h_image *img, char *err, size_t errlen);
/** The CPE position rows of a wp x hp tile, f32 [wp hp][width] in merge-block order, from the table. */
void pulsar_nemotron_h_positions(const pulsar_nemotron_h_vision_shape *s, const std::vector<float> &field, uint32_t wp,
                                 uint32_t hp, std::vector<float> *out);
/** Patch row index (merge-block order) of raster patch (r, c) in a wp-wide tile. */
static inline uint32_t pulsar_nemotron_h_patch_slot(uint32_t r, uint32_t c, uint32_t wp) {
    return (((r / 2u) * (wp / 2u) + c / 2u) * 4u) + (r % 2u) * 2u + (c % 2u);
}

/** Read the tower's facts and bind its tensors (when pulsar.vision_present); the tokenizer's image tokens.  false =
 *  refused (said). */
bool pulsar_nemotron_h_vision_bind(const pulsar_model *m, pulsar_role_binder *b, pulsar_nemotron_h_vision *v);
bool pulsar_nemotron_h_vision_tokens(pulsar_nemotron_h_vision *v, const struct bpe_tokenizer *tok);
/** The tower's device scratch (after the GPU is up), sized for the largest tile.  false = refused. */
bool pulsar_nemotron_h_vision_run_create(const pulsar_model *m, pulsar_nemotron_h_vision *v);
void pulsar_nemotron_h_vision_run_free(pulsar_nemotron_h_vision *v);
/** One image's LLM rows: bf16 [*n_rows][4096] (malloc'd).  `dbg` (optional): the tower's residual after the last block,
 *  f32 [prefix + patches][width], raster order of the patches (the gate's stage).  false + `err`. */
bool pulsar_nemotron_h_vision_encode(const pulsar_model *m, const pulsar_nemotron_h_vision *v, const uint8_t *bytes,
                                     size_t len, uint16_t **rows, int *n_rows, std::vector<float> *dbg, char *err,
                                     size_t errlen);

/** The family's image front (family.h pulsar_family_vision). */
extern const pulsar_family_vision PULSAR_NEMOTRON_H_IMAGE_FRONT;

/** The family's state (pulsar_engine::family_state). */
typedef struct {
    pulsar_nemotron_h_shape shape;
    pulsar_tensor *embed;            ///< embed [n_vocab rows of n_embd]
    pulsar_tensor *output_norm;      ///< output_norm [n_embd] (norm_f)
    pulsar_tensor *head;             ///< head [n_embd -> n_vocab] (untied)
    pulsar_nemotron_h_block layer[PULSAR_FAMILY_MAX_LAYER];
    pulsar_nemotron_h_draft mtp;
    /** L331: the vision tower (present when the container holds it; else images are refused by name). */
    pulsar_nemotron_h_vision vision;
    /** What the binding priced (role_bind.h): every byte the core stages, by role. */
    pulsar_role_census census;
    /** N9: the MTP module drafts (the container holds it and the open selected it): the engine's drafter is
     *  PULSAR_NEMOTRON_H_DRAFTER, the plan declares the drafter's slot, sessions verify. */
    bool drafting;
    /** L298: the LiLiCorr block drafter drafts instead (the container holds it and the open selected it): the
     *  engine's drafter is the core's block drafter (spec_block.h) over this declaration and these weights; the
     *  MTP module, if the container holds one, is declared unloaded. */
    struct pulsar_block_drafter *block;
    /** The digest a session's disk segments are keyed by: the directory's names, types and sizes (FNV-1a). */
    uint64_t artifact_digest;
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

/* ---- 3. The forward on the core row-step driver (nemotron_h_forward.cpp) -------------------------------------- */

/** The ids of the family's state declarations (state_kinds.h kinds), in the plan's table order.  A Mamba-2 layer
 *  carries SSM + CONV, an attention layer KV; the drafter's slot (after the trunk's, when the container holds the MTP
 *  module) KV. */
enum pulsar_nemotron_h_state_id {
    PULSAR_NEMOTRON_H_STATE_SSM = 0,   ///< FIXED, DELTA: the selective-scan state f32 [heads][head_dim][d_state]
    PULSAR_NEMOTRON_H_STATE_CONV,      ///< FIXED, SNAPSHOT: the x | B | C convolution's tail bf16 [conv_k - 1][channels]
    PULSAR_NEMOTRON_H_STATE_KV,        ///< POOL, one token a row: an attention layer's K | V rows (F1's BF16 rows)
    /** FIXED, no capture, on the drafter's slot (N9): a bank's parked conditioning row -- norm_f of the trunk's last
     *  row, bf16 [n_embd] -- until the token after it arrives (spec_mtp.h, the chained-MTP lane) */
    PULSAR_NEMOTRON_H_STATE_MTP_PEND,
    PULSAR_NEMOTRON_H_STATE_COUNT
};

/** Build the family's state declaration into e->state_plan (the load, after the plan and the bind). */
void pulsar_nemotron_h_state_plan(pulsar_engine *e);

/** The family's ops on the core row-step driver (step_driver.h): block[PULSAR_LAYER_BLOCKS] composes each layer
 *  from its pulsar_layer_desc; its MTP module is declared to the chained-MTP lane (spec_mtp.h). */
extern const pulsar_driver_ops PULSAR_NEMOTRON_H_DRIVER;
/** N9: the drafter -- the chained-MTP drafter (spec_mtp.h) at the card's numbers. */
extern const struct pulsar_drafter_ops PULSAR_NEMOTRON_H_DRAFTER;

#endif /* PULSAR_FAMILY_NEMOTRON_H_H */

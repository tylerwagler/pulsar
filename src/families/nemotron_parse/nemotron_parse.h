/* nemotron_parse.h -- NVIDIA Nemotron-3.5 Parse (L321) as a family module: a page image in, the model's text out.
 *
 * THE MODEL (hf_nemotron_parse_modeling.py, the checkpoint's own code; tools/container/nemotron_parse.py):
 *   image   Pillow BILINEAR down to fit 2048 x 1664 (only when larger), white-padded at the bottom / right to 16-row /
 *           64-column multiples, CLIP-normalised (f32), cut into 16 x 16 patches (channel, row, column order)
 *   ViT     C-RADIOv4-SO400M: patch Linear 768 -> 1152 (no bias) + the learned 128 x 128 position field resampled to
 *           the page's content grid (content-normalised: antialiased bilinear interpolate, then grid_sample at each
 *           patch's content-clamped centre -- computed on the host, torch's arithmetic), 10 class tokens before the
 *           patches, then 27 timm blocks: x += proj(MHA(LN1 x)); x += fc2(GELU(fc1(LN2 x))) -- 16 heads of 72,
 *           bidirectional, qkv / proj / fc biases, LayerNorm eps 1e-6
 *   neck    patches: LN(conv1 x) (1x1, 1152 -> 1024); four horizontally adjacent rows merged by conv2 (a Linear
 *           4096 -> 1024, the builder's [out, kw, ch]) + LN; the summary: LN(Linear(class tokens 0 | 1)) -- the
 *           decoder's visual prefix, the summary row last
 *   decoder mBART, decoder-only over [prefix | prompt | generated]: LN_emb(embeddings), 10 x {x += o(GQA(LN x));
 *           x += fc2(GELU(fc1(LN x)))}, LN -> the tied head; 16 query / 8 KV heads of 64, CAUSAL over the whole
 *           sequence (the prefix too), no positional encoding, text embeddings x 32 (sqrt(1024)), LayerNorm eps 1e-5
 *   recipe  greedy, repetition penalty 1.1 over the prompt and generated ids (transformers'
 *           RepetitionPenaltyLogitsProcessor), until eos (2) or 9000 new tokens
 * The output is transformers' batch_decode(ids, skip_special_tokens=False): the prompt and the generated text, the
 * model's own markup (classes, boxes, markdown) for the checkpoint's postprocessing to read.
 *
 * THE CONTAINER CONTRACT (tools/container/nemotron_parse.py emits, this family reads): general.architecture =
 * "nemotron_parse"; the pinned facts under nemotron_parse.*; every tensor a bind record (role + block `vit` / `layers`,
 * layer); Linears f16 [in, out] in ne order, norms / biases / class tokens / the position field f32; the side file
 * tokenizer.json (the decode table and the prompt's added tokens).
 */
/** @file nemotron_parse.h
 *  @brief NVIDIA Nemotron-3.5 Parse as a family module (a page image in, its text out): shape, weights, decode table,
 *  page front, forward. */
#ifndef PULSAR_FAMILY_NEMOTRON_PARSE_H
#define PULSAR_FAMILY_NEMOTRON_PARSE_H

#include "engine/pulsar_engine_internal.h"
#include "engine/role_bind.h"

#include <string>
#include <unordered_map>
#include <vector>

/// The `general.architecture` value this family claims.
#define PULSAR_NEMOTRON_PARSE_ARCH "nemotron_parse"
/// The most ViT blocks the model state holds (the profile: 27).
#define PULSAR_NEMOTRON_PARSE_VIT_MAX 32u
/// The most decoder layers the model state holds (the profile: 10).
#define PULSAR_NEMOTRON_PARSE_DEC_MAX 16u

/// The family descriptor (nemotron_parse_family.cpp); family.cpp registers it.
extern const pulsar_family PULSAR_FAMILY_NEMOTRON_PARSE;

/* ---- 1. Shape -------------------------------------------------------------------------------------------------- */

/** The loaded shape: the container's nemotron_parse.* facts, every one pinned to PULSAR_NEMOTRON_PARSE_SHAPE
 *  (nemotron_parse_shape.cpp).  The profile's values are in parentheses. */
typedef struct {
    /** @var pulsar_nemotron_parse_shape::vit_width
     *  vit.width (1152): the ViT's residual width
     *  @var pulsar_nemotron_parse_shape::vit_depth
     *  vit.depth (27): the ViT's blocks
     *  @var pulsar_nemotron_parse_shape::vit_heads
     *  vit.heads (16): the ViT's attention heads
     *  @var pulsar_nemotron_parse_shape::vit_head_dim
     *  vit.head_dim (72)
     *  @var pulsar_nemotron_parse_shape::vit_mlp
     *  vit.mlp (4304): the ViT MLP's inner width (fc1's output)
     *  @var pulsar_nemotron_parse_shape::patch
     *  vit.patch (16): a patch's side in pixels; a patch row is 3 * patch * patch (768) values
     *  @var pulsar_nemotron_parse_shape::n_cls
     *  vit.n_cls (10): the class tokens before the patches (tokens 0 and 1 feed the summary)
     *  @var pulsar_nemotron_parse_shape::pos_grid
     *  vit.pos_grid (128): the learned position field's side, in patches */
    uint32_t vit_width, vit_depth, vit_heads, vit_head_dim, vit_mlp, patch, n_cls, pos_grid;
    /// vit.norm_eps (1e-6): the ViT LayerNorms' epsilon
    float vit_eps;
    /** @var pulsar_nemotron_parse_shape::neck_width
     *  neck.width (1024): the neck's output width (= the decoder's width)
     *  @var pulsar_nemotron_parse_shape::merge
     *  neck.merge (4): horizontally adjacent patches conv2 merges into one prefix row */
    uint32_t neck_width, merge;
    /// neck.norm_eps (1e-6): the neck LayerNorms' epsilon
    float neck_eps;
    /** @var pulsar_nemotron_parse_shape::width
     *  dec.width (1024): the decoder's residual width; its head_dim is width / heads (64)
     *  @var pulsar_nemotron_parse_shape::n_layer
     *  dec.layers (10): the decoder's layers
     *  @var pulsar_nemotron_parse_shape::heads
     *  dec.heads (16): the decoder's query heads
     *  @var pulsar_nemotron_parse_shape::kv_heads
     *  dec.kv_heads (8): the decoder's KV heads
     *  @var pulsar_nemotron_parse_shape::ffn
     *  dec.ffn (4096): the decoder MLP's inner width
     *  @var pulsar_nemotron_parse_shape::vocab
     *  dec.vocab (72512): the embedding / tied head rows, padded past the tokenizer's last id */
    uint32_t width, n_layer, heads, kv_heads, ffn, vocab;
    /** @var pulsar_nemotron_parse_shape::eps
     *  dec.norm_eps (1e-5): the decoder LayerNorms' epsilon
     *  @var pulsar_nemotron_parse_shape::embed_scale
     *  dec.embed_scale (32 = sqrt(width)): the text embeddings' multiplier */
    float eps, embed_scale;
    /** @var pulsar_nemotron_parse_shape::max_seq
     *  max_sequence (16384): the decoder's context in tokens (prefix + prompt + generated); each layer's KV pool holds it
     *  @var pulsar_nemotron_parse_shape::bos
     *  bos (0): pinned; the forward does not read it (the default prompt spells its own `<s>`)
     *  @var pulsar_nemotron_parse_shape::eos
     *  eos (2): generation stops after sampling it
     *  @var pulsar_nemotron_parse_shape::pad
     *  pad (1): pinned; nothing pads (one page per call) */
    uint32_t max_seq, bos, eos, pad;
    /** @var pulsar_nemotron_parse_shape::canvas_h
     *  image.canvas_h (2048): the largest page height in pixels; a taller image is resized down
     *  @var pulsar_nemotron_parse_shape::canvas_w
     *  image.canvas_w (1664): the largest page width in pixels
     *  @var pulsar_nemotron_parse_shape::align_h
     *  image.align_h (16): the padded page's height is a multiple of it
     *  @var pulsar_nemotron_parse_shape::align_w
     *  image.align_w (64): the padded page's width is a multiple of it (patch * merge) */
    uint32_t canvas_h, canvas_w, align_h, align_w;
    /** @var pulsar_nemotron_parse_shape::mean
     *  image.mean.0..2: the CLIP normalisation's per-channel (R, G, B) mean, on p / 255
     *  @var pulsar_nemotron_parse_shape::std
     *  image.std.0..2: the CLIP normalisation's per-channel standard deviation */
    float mean[3], std[3];
    /// gen.repetition_penalty (1.1): applied to every id already in the sequence (prompt and generated)
    float rep_penalty;
    /// gen.max_new_tokens (9000): the default cap on generated tokens (pulsar_parse_options::max_new_tokens overrides)
    uint32_t max_new;
    /// gen.prompt: the default prompt, added tokens only, NUL-terminated
    char prompt[256];
} pulsar_nemotron_parse_shape;

/// The one profile (Nemotron-3.5 Parse), the values every container fact must equal.
extern const pulsar_nemotron_parse_shape PULSAR_NEMOTRON_PARSE_SHAPE;

/** Read the container's pinned facts and check them against the profile (every one is pinned).  false = refused. */
bool pulsar_nemotron_parse_read_shape(const pulsar_model *m, pulsar_nemotron_parse_shape *s);

/* ---- 2. Weights + the decode table ----------------------------------------------------------------------------- */

/** One ViT (timm) block; f32 norms and biases, f16 Linears [in -> out]; bind roles vit_* in block `vit`. */
typedef struct {
    /** @var pulsar_nemotron_parse_vit_block::norm1
     *  LN1 weight [vit_width]
     *  @var pulsar_nemotron_parse_vit_block::norm1_b
     *  LN1 bias [vit_width]
     *  @var pulsar_nemotron_parse_vit_block::qkv
     *  qkv [vit_width -> 3 * vit_heads * vit_head_dim]: q | k | v
     *  @var pulsar_nemotron_parse_vit_block::qkv_b
     *  qkv bias [3 * vit_heads * vit_head_dim]
     *  @var pulsar_nemotron_parse_vit_block::proj
     *  the attention output [vit_heads * vit_head_dim -> vit_width]
     *  @var pulsar_nemotron_parse_vit_block::proj_b
     *  proj bias [vit_width]
     *  @var pulsar_nemotron_parse_vit_block::norm2
     *  LN2 weight [vit_width]
     *  @var pulsar_nemotron_parse_vit_block::norm2_b
     *  LN2 bias [vit_width]
     *  @var pulsar_nemotron_parse_vit_block::fc1
     *  fc1 [vit_width -> vit_mlp] (GELU after)
     *  @var pulsar_nemotron_parse_vit_block::fc1_b
     *  fc1 bias [vit_mlp]
     *  @var pulsar_nemotron_parse_vit_block::fc2
     *  fc2 [vit_mlp -> vit_width]
     *  @var pulsar_nemotron_parse_vit_block::fc2_b
     *  fc2 bias [vit_width] */
    pulsar_tensor *norm1, *norm1_b, *qkv, *qkv_b, *proj, *proj_b, *norm2, *norm2_b, *fc1, *fc1_b, *fc2, *fc2_b;
} pulsar_nemotron_parse_vit_block;

/** One mBART decoder layer; f32 norms and biases, f16 Linears [in -> out]; bind roles in block `layers`. */
typedef struct {
    /** @var pulsar_nemotron_parse_dec_layer::attn_norm
     *  attn_norm: the attention LayerNorm's weight [width]
     *  @var pulsar_nemotron_parse_dec_layer::attn_norm_b
     *  attn_norm_bias [width]
     *  @var pulsar_nemotron_parse_dec_layer::q
     *  attn_q [width -> width]
     *  @var pulsar_nemotron_parse_dec_layer::q_b
     *  attn_q_bias [width]
     *  @var pulsar_nemotron_parse_dec_layer::k
     *  attn_k [width -> kv_heads * head_dim]
     *  @var pulsar_nemotron_parse_dec_layer::k_b
     *  attn_k_bias [kv_heads * head_dim]
     *  @var pulsar_nemotron_parse_dec_layer::v
     *  attn_v [width -> kv_heads * head_dim]
     *  @var pulsar_nemotron_parse_dec_layer::v_b
     *  attn_v_bias [kv_heads * head_dim]
     *  @var pulsar_nemotron_parse_dec_layer::o
     *  attn_o [width -> width], read over bf16 rows
     *  @var pulsar_nemotron_parse_dec_layer::o_b
     *  attn_o_bias [width] */
    pulsar_tensor *attn_norm, *attn_norm_b, *q, *q_b, *k, *k_b, *v, *v_b, *o, *o_b;
    /** @var pulsar_nemotron_parse_dec_layer::ffn_norm
     *  ffn_norm: the MLP LayerNorm's weight [width]
     *  @var pulsar_nemotron_parse_dec_layer::ffn_norm_b
     *  ffn_norm_bias [width]
     *  @var pulsar_nemotron_parse_dec_layer::up
     *  ffn_up (fc1) [width -> ffn] (GELU after; no gate)
     *  @var pulsar_nemotron_parse_dec_layer::up_b
     *  ffn_up_bias [ffn]
     *  @var pulsar_nemotron_parse_dec_layer::down
     *  ffn_down (fc2) [ffn -> width]
     *  @var pulsar_nemotron_parse_dec_layer::down_b
     *  ffn_down_bias [width] */
    pulsar_tensor *ffn_norm, *ffn_norm_b, *up, *up_b, *down, *down_b;
} pulsar_nemotron_parse_dec_layer;

/** The tokenizer as Parse needs it: ids -> bytes (transformers' decode, skip_special_tokens=False: an added token is
 *  its content; a vocabulary token is its ByteLevel string mapped back to bytes, or its UTF-8 when a character has no
 *  byte), and an added token's content -> its id (the prompt is added tokens only). */
struct pulsar_nemotron_parse_vocab {
    /// id -> the bytes it decodes to (before the sequence's UTF-8 repair); empty for a padding id with no token
    std::vector<std::string> bytes;
    /// an added token's content -> its id (the prompt encoder's table)
    std::unordered_map<std::string, int> added;
};

/// The forward's device state (nemotron_parse_forward.cpp): built by after_gpu, freed by unload.
struct pulsar_nemotron_parse_run;

/** The family's state (pulsar_engine::family_state): allocated with new by load, deleted by unload. */
typedef struct {
    /// The loaded, checked shape (pulsar_nemotron_parse_read_shape).
    pulsar_nemotron_parse_shape shape;
    /** @var pulsar_nemotron_parse_model::patch
     *  vit_patch: the patch Linear [3 * patch * patch -> vit_width] (no bias), f16
     *  @var pulsar_nemotron_parse_model::cls
     *  vit_cls: the class tokens, f32 [n_cls][vit_width] (read from the mapping into the residual)
     *  @var pulsar_nemotron_parse_model::pos
     *  vit_pos: the learned position field, f32 [pos_grid * pos_grid][vit_width] (copied to pos_field at bind) */
    pulsar_tensor *patch, *cls, *pos;
    /// The ViT blocks; entries [0, shape.vit_depth) are bound.
    pulsar_nemotron_parse_vit_block vit[PULSAR_NEMOTRON_PARSE_VIT_MAX];
    /** @var pulsar_nemotron_parse_model::neck_proj
     *  neck_proj (conv1, 1x1): [vit_width -> neck_width]
     *  @var pulsar_nemotron_parse_model::neck_proj_b
     *  neck_proj_bias [neck_width]
     *  @var pulsar_nemotron_parse_model::neck_norm1
     *  neck_norm1: the LayerNorm after conv1, weight [neck_width]
     *  @var pulsar_nemotron_parse_model::neck_norm1_b
     *  neck_norm1_bias [neck_width]
     *  @var pulsar_nemotron_parse_model::neck_merge
     *  neck_merge (conv2 as a Linear, no bias): [merge * neck_width -> neck_width]
     *  @var pulsar_nemotron_parse_model::neck_norm2
     *  neck_norm2: the LayerNorm after the merge, weight [neck_width]
     *  @var pulsar_nemotron_parse_model::neck_norm2_b
     *  neck_norm2_bias [neck_width] */
    pulsar_tensor *neck_proj, *neck_proj_b, *neck_norm1, *neck_norm1_b, *neck_merge, *neck_norm2, *neck_norm2_b;
    /** @var pulsar_nemotron_parse_model::summary
     *  neck_summary: [2 * vit_width -> neck_width] over class tokens 0 | 1
     *  @var pulsar_nemotron_parse_model::summary_b
     *  neck_summary_bias [neck_width]
     *  @var pulsar_nemotron_parse_model::neck_norm3
     *  neck_norm3: the summary's LayerNorm, weight [neck_width]
     *  @var pulsar_nemotron_parse_model::neck_norm3_b
     *  neck_norm3_bias [neck_width] */
    pulsar_tensor *summary, *summary_b, *neck_norm3, *neck_norm3_b;
    /** @var pulsar_nemotron_parse_model::embed
     *  embed: the text embedding, f16 [vocab rows of width]; also the tied head
     *  @var pulsar_nemotron_parse_model::embed_norm
     *  embed_norm: LN_emb's weight [width], over the whole [prefix | text] sequence
     *  @var pulsar_nemotron_parse_model::embed_norm_b
     *  embed_norm_bias [width]
     *  @var pulsar_nemotron_parse_model::output_norm
     *  output_norm: the final LayerNorm's weight [width], before the head
     *  @var pulsar_nemotron_parse_model::output_norm_b
     *  output_norm_bias [width] */
    pulsar_tensor *embed, *embed_norm, *embed_norm_b, *output_norm, *output_norm_b;
    /// The decoder layers; entries [0, shape.n_layer) are bound.
    pulsar_nemotron_parse_dec_layer layer[PULSAR_NEMOTRON_PARSE_DEC_MAX];
    /// What the binding priced (role_bind.h): every byte the core stages, by role.
    pulsar_role_census census;
    /// The decode table (pulsar_nemotron_parse_load_vocab); owned, deleted by unload.
    pulsar_nemotron_parse_vocab *vocab;
    std::vector<int> prompt_ids;          ///< the default prompt (shape.prompt), as ids
    std::vector<float> pos_field;          ///< the position field, host f32 [pos_grid][pos_grid][vit_width]
    /// The forward's device state: built by after_gpu (pulsar_nemotron_parse_run_create), freed by unload; NULL before.
    struct pulsar_nemotron_parse_run *run;
} pulsar_nemotron_parse_model;

/** Bind every tensor the shape declares, admit its format, price it, and copy the position field to the host
 *  (nemotron_parse_weights.cpp).
 *  @param m  the container
 *  @param w  the family state: its shape is read; the tensors, the census and pos_field are written
 *  @return true = bound; false = refused (said) */
bool pulsar_nemotron_parse_bind(const pulsar_model *m, pulsar_nemotron_parse_model *w);
/** The decode table and the default prompt's ids, from tokenizer.json beside the shards.  false = refused (said). */
bool pulsar_nemotron_parse_load_vocab(pulsar_nemotron_parse_model *w, const char *model_path);
/** `text` (added tokens only, back to back) as ids; false + `err` when a piece is not an added token. */
bool pulsar_nemotron_parse_encode_prompt(const pulsar_nemotron_parse_vocab *v, const char *text, std::vector<int> *ids,
                                          char *err, size_t errlen);
/** ids -> text, transformers' decode (skip_special_tokens=False). */
std::string pulsar_nemotron_parse_decode(const pulsar_nemotron_parse_vocab *v, const int *ids, size_t n);

/* ---- 3. The page: pixels -> patch rows + position rows (nemotron_parse_image.cpp) -------------------------------- */

/** A prepared page (pulsar_nemotron_parse_page_prepare): the padded canvas, the content inside it, the patch rows. */
typedef struct {
    /** @var pulsar_nemotron_parse_page::h
     *  the padded canvas height in pixels (a multiple of align_h) */
    uint32_t h, w;              ///< the padded canvas (multiples of 16 and 64)
    /** @var pulsar_nemotron_parse_page::ch
     *  the content height in pixels (the resized image, at the top) */
    uint32_t ch, cw;            ///< the content (the resized image) inside it, top-left
    std::vector<uint16_t> patches;   ///< f16 [(h / 16) * (w / 16)][768]
} pulsar_nemotron_parse_page;

/** Decode (PNG / JPEG), resize (Pillow BILINEAR, only when larger than the canvas), pad, normalise, patch.  false +
 *  `err` on an image the model does not take. */
bool pulsar_nemotron_parse_page_prepare(const pulsar_nemotron_parse_shape *s, const uint8_t *bytes, size_t len,
                                        pulsar_nemotron_parse_page *page, char *err, size_t errlen);
/** The position rows of a (canvas, content) geometry: f32 [(h / 16) * (w / 16)][vit_width] from the field (torch's
 *  content-normalised interpolate + grid_sample). */
void pulsar_nemotron_parse_positions(const pulsar_nemotron_parse_shape *s, const std::vector<float> &field, uint32_t h,
                                     uint32_t w, uint32_t ch, uint32_t cw, std::vector<float> *out);

/* ---- 4. The forward (nemotron_parse_forward.cpp) ----------------------------------------------------------------- */

bool pulsar_nemotron_parse_run_create(pulsar_engine *e);
void pulsar_nemotron_parse_run_free(pulsar_nemotron_parse_model *w);

/** The parser entries (pulsar.h pulsar_engine_parse_page). */
extern const pulsar_family_parser PULSAR_NEMOTRON_PARSE_PARSER;

#endif /* PULSAR_FAMILY_NEMOTRON_PARSE_H */

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
#ifndef PULSAR_FAMILY_NEMOTRON_PARSE_H
#define PULSAR_FAMILY_NEMOTRON_PARSE_H

#include "engine/pulsar_engine_internal.h"
#include "engine/role_bind.h"

#include <string>
#include <unordered_map>
#include <vector>

#define PULSAR_NEMOTRON_PARSE_ARCH "nemotron_parse"
#define PULSAR_NEMOTRON_PARSE_VIT_MAX 32u
#define PULSAR_NEMOTRON_PARSE_DEC_MAX 16u

extern const pulsar_family PULSAR_FAMILY_NEMOTRON_PARSE;

/* ---- 1. Shape -------------------------------------------------------------------------------------------------- */

typedef struct {
    uint32_t vit_width, vit_depth, vit_heads, vit_head_dim, vit_mlp, patch, n_cls, pos_grid;
    float vit_eps;
    uint32_t neck_width, merge;
    float neck_eps;
    uint32_t width, n_layer, heads, kv_heads, ffn, vocab;
    float eps, embed_scale;
    uint32_t max_seq, bos, eos, pad;
    uint32_t canvas_h, canvas_w, align_h, align_w;
    float mean[3], std[3];
    float rep_penalty;
    uint32_t max_new;
    char prompt[256];
} pulsar_nemotron_parse_shape;

extern const pulsar_nemotron_parse_shape PULSAR_NEMOTRON_PARSE_SHAPE;

/** Read the container's pinned facts and check them against the profile (every one is pinned).  false = refused. */
bool pulsar_nemotron_parse_read_shape(const pulsar_model *m, pulsar_nemotron_parse_shape *s);

/* ---- 2. Weights + the decode table ----------------------------------------------------------------------------- */

typedef struct {
    pulsar_tensor *norm1, *norm1_b, *qkv, *qkv_b, *proj, *proj_b, *norm2, *norm2_b, *fc1, *fc1_b, *fc2, *fc2_b;
} pulsar_nemotron_parse_vit_block;

typedef struct {
    pulsar_tensor *attn_norm, *attn_norm_b, *q, *q_b, *k, *k_b, *v, *v_b, *o, *o_b;
    pulsar_tensor *ffn_norm, *ffn_norm_b, *up, *up_b, *down, *down_b;
} pulsar_nemotron_parse_dec_layer;

/** The tokenizer as Parse needs it: ids -> bytes (transformers' decode, skip_special_tokens=False: an added token is
 *  its content; a vocabulary token is its ByteLevel string mapped back to bytes, or its UTF-8 when a character has no
 *  byte), and an added token's content -> its id (the prompt is added tokens only). */
struct pulsar_nemotron_parse_vocab {
    std::vector<std::string> bytes;
    std::unordered_map<std::string, int> added;
};

struct pulsar_nemotron_parse_run;

typedef struct {
    pulsar_nemotron_parse_shape shape;
    pulsar_tensor *patch, *cls, *pos;
    pulsar_nemotron_parse_vit_block vit[PULSAR_NEMOTRON_PARSE_VIT_MAX];
    pulsar_tensor *neck_proj, *neck_proj_b, *neck_norm1, *neck_norm1_b, *neck_merge, *neck_norm2, *neck_norm2_b;
    pulsar_tensor *summary, *summary_b, *neck_norm3, *neck_norm3_b;
    pulsar_tensor *embed, *embed_norm, *embed_norm_b, *output_norm, *output_norm_b;
    pulsar_nemotron_parse_dec_layer layer[PULSAR_NEMOTRON_PARSE_DEC_MAX];
    pulsar_role_census census;
    pulsar_nemotron_parse_vocab *vocab;
    std::vector<int> prompt_ids;          ///< the default prompt (shape.prompt), as ids
    std::vector<float> pos_field;          ///< the position field, host f32 [pos_grid][pos_grid][vit_width]
    struct pulsar_nemotron_parse_run *run;
} pulsar_nemotron_parse_model;

bool pulsar_nemotron_parse_bind(const pulsar_model *m, pulsar_nemotron_parse_model *w);
/** The decode table and the default prompt's ids, from tokenizer.json beside the shards.  false = refused (said). */
bool pulsar_nemotron_parse_load_vocab(pulsar_nemotron_parse_model *w, const char *model_path);
/** `text` (added tokens only, back to back) as ids; false + `err` when a piece is not an added token. */
bool pulsar_nemotron_parse_encode_prompt(const pulsar_nemotron_parse_vocab *v, const char *text, std::vector<int> *ids,
                                          char *err, size_t errlen);
/** ids -> text, transformers' decode (skip_special_tokens=False). */
std::string pulsar_nemotron_parse_decode(const pulsar_nemotron_parse_vocab *v, const int *ids, size_t n);

/* ---- 3. The page: pixels -> patch rows + position rows (nemotron_parse_image.cpp) -------------------------------- */

typedef struct {
    uint32_t h, w;              ///< the padded canvas (multiples of 16 and 64)
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

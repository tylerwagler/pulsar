/* nemotron_enc.h -- the Nemotron-3.5 retrieval encoders (L321): NVIDIA Nemotron-3.5-Embed-8B and -Rerank-8B as one
 * ENCODER family module.
 *
 * Built in the plugin layout (src/families/<name>/) on Nemotron-H's pattern: the family declares its shape and its
 * tensor roles and binds through the core's role-driven binder (src/engine/role_bind.h) by the bind records the builder
 * writes (tools/container/nemotron_enc.py) -- no name is parsed.  Outside this directory the family is one registry row
 * (family.cpp), one id (pulsar.h) and its gates.
 *
 * THE MODEL (config.json text_config; retriever/README.md): a Ministral-3 trunk run BIDIRECTIONALLY (is_causal false)
 *     h = embed(ids)                                      (no embedding norm, no BOS / EOS: the template writes none)
 *     34 x { h += o( GQA_bidir( rope(q(rmsnorm(h))), rope(k(.)), v(.) ) );  h += down( silu(gate(rmsnorm(h))) * up(.) ) }
 *     x = rmsnorm(h) * norm;   pooled = mean over the input's rows (f32)
 *     Embed:  out = pooled / |pooled|                     Rerank: out = score . pooled  (score [1, 4096], f32)
 * 32 q / 8 kv heads of 128, hidden 4096, ffn 14336, vocab 131072; the rope is YaRN at theta 1e6, factor 16 over an
 * original 16384, beta_fast 32 / beta_slow 1, attention factor exactly 1 (mscale = mscale_all_dim = 1); transformers'
 * llama-4 q scale 1 + 0.1 ln(1 + floor(pos / 16384)) is exactly 1 below position 16384, so an input is capped there
 * (pulsar_engine_encode_max_tokens) and the scale never applies.
 *
 * THE TEMPLATE (chat_template.jinja, as shipped; no BOS / EOS): Embed renders "query: " or "passage: " + the text (the
 * system prefix, rstrip(" ") + " "); Rerank renders "query: " + query + "\n\npassage: " + document.  In client text
 * '[' and '<' are each followed by a U+200C, which the tokenizer's normaliser drops again, so client text never forms
 * a special token; the renderer writes the same and also hands the client text over as a span.
 *
 * THE CONTAINER CONTRACT (tools/container/nemotron_enc.py emits, this family reads):
 *   general.architecture = "nemotron_enc";
 *   nemotron_enc.<key>   text_config verbatim (kv.config_kvs: nested dicts flattened with '.');
 *   pulsar.encoder ("embed" | "rerank"), pulsar.pooling ("mean"), pulsar.normalize (Embed true, Rerank false),
 *   pulsar.vision_present (false in v1: the Pixtral tower is not written);
 *   every tensor carries a bind record: role (+ block `layers`, layer).  Linears are stored [in, out] in ne order (an
 *   exl3m_k* dense-arm rate, or BF16); norms, the embedding and the score head BF16.
 *   side file read: tokenizer.json (the loader's one stop id is nemotron_enc.eos_token_id: the checkpoints ship no
 *   generation_config.json, and an encoder never stops).
 */
#ifndef PULSAR_FAMILY_NEMOTRON_ENC_H
#define PULSAR_FAMILY_NEMOTRON_ENC_H

#include "engine/pulsar_engine_internal.h"
#include "engine/role_bind.h"

#define PULSAR_NEMOTRON_ENC_ARCH "nemotron_enc"

/** The family descriptor (nemotron_enc_family.cpp); family.cpp registers it. */
extern const pulsar_family PULSAR_FAMILY_NEMOTRON_ENC;

/* ---- 1. Shape -------------------------------------------------------------------------------------------------- */

/** The loaded shape.  The geometry is checked field by field against PULSAR_NEMOTRON_ENC_SHAPE_8B; the layer count is
 *  the container's (a mini and the 34-layer model are one profile); the kind is the builder's fact. */
typedef struct {
    const char *name;
    uint32_t n_embd;           ///< hidden_size (4096)
    uint32_t n_vocab;          ///< vocab_size (131072): the embedding rows (every id real)
    uint32_t n_head;           ///< num_attention_heads (32)
    uint32_t n_head_kv;        ///< num_key_value_heads (8)
    uint32_t head_dim;         ///< head_dim (128); the score scale is 1/sqrt(head_dim)
    uint32_t n_ff;             ///< intermediate_size (14336)
    float rms_eps;             ///< rms_norm_eps (1e-5)
    float rope_theta;          ///< rope_parameters.rope_theta (1e6)
    float rope_factor;         ///< rope_parameters.factor (16)
    uint32_t rope_orig_ctx;    ///< rope_parameters.original_max_position_embeddings (16384): also the input cap
    float rope_beta_fast;      ///< rope_parameters.beta_fast (32)
    float rope_beta_slow;      ///< rope_parameters.beta_slow (1)
    uint32_t eos_id;           ///< eos_token_id (2): the tokenizer loader's stop id, never sampled
    /* the container's */
    uint32_t n_layer;          ///< num_hidden_layers
    pulsar_encoder_kind kind;  ///< pulsar.encoder: EMBED or RERANK
} pulsar_nemotron_enc_shape;

extern const pulsar_nemotron_enc_shape PULSAR_NEMOTRON_ENC_SHAPE_8B;

/** Read the shape from the container's keys and check it (nemotron_enc_shape.cpp).  Every refusal is said and counted
 *  (pulsar_load_refuse); false = at least one. */
bool pulsar_nemotron_enc_read_shape(const pulsar_model *m, pulsar_nemotron_enc_shape *s);
/** The layer plan: PULSAR_LAYER_BLOCKS, {ATTN_FULL, DENSE} on every layer (the encoder's attention is bidirectional:
 *  the plan's mixer is the block, its direction the forward's -- an encoder has no decode). */
void pulsar_nemotron_enc_plan(const pulsar_nemotron_enc_shape *s, pulsar_layer_plan *plan);

/* ---- 2. Weights ------------------------------------------------------------------------------------------------- */

/** One trunk layer; bind roles in the comments. */
typedef struct {
    pulsar_tensor *attn_norm;   ///< attn_norm [n_embd]
    pulsar_tensor *attn_q;      ///< attn_q [n_embd -> n_head * head_dim]
    pulsar_tensor *attn_k;      ///< attn_k [n_embd -> n_head_kv * head_dim]
    pulsar_tensor *attn_v;      ///< attn_v [n_embd -> n_head_kv * head_dim]
    pulsar_tensor *attn_o;      ///< attn_o [n_head * head_dim -> n_embd]
    pulsar_tensor *ffn_norm;    ///< ffn_norm [n_embd]
    pulsar_tensor *ffn_gate;    ///< ffn_gate [n_embd -> n_ff]
    pulsar_tensor *ffn_up;      ///< ffn_up [n_embd -> n_ff]
    pulsar_tensor *ffn_down;    ///< ffn_down [n_ff -> n_embd]
} pulsar_nemotron_enc_layer;

/** The forward's device state (nemotron_enc_forward.cpp): built by after_gpu, freed by unload. */
struct pulsar_nemotron_enc_run;

/** The family's state (pulsar_engine::family_state). */
typedef struct {
    pulsar_nemotron_enc_shape shape;
    pulsar_tensor *embed;        ///< embed [n_vocab rows of n_embd]
    pulsar_tensor *output_norm;  ///< output_norm [n_embd]
    pulsar_tensor *score;        ///< score [n_embd -> 1] (Rerank only)
    pulsar_nemotron_enc_layer layer[PULSAR_FAMILY_MAX_LAYER];
    /** The score head widened to f32 at load (Rerank): the host's dot with the pooled row. */
    float *score_f32;
    /** What the binding priced (role_bind.h): every byte the core stages, by role. */
    pulsar_role_census census;
    /** The tokenizer: F5's shared BPE engine in its HF configuration (tekken, the U+200C Replace normaliser), loaded
     *  at open from tokenizer.json + config.json beside the container's shards. */
    struct bpe_tokenizer *tok;
    struct pulsar_nemotron_enc_run *run;
} pulsar_nemotron_enc_model;

/** Bind every tensor the shape declares, admit its format, price it, widen the score head (nemotron_enc_weights.cpp).
 *  false = refused. */
bool pulsar_nemotron_enc_bind(const pulsar_model *m, pulsar_nemotron_enc_model *w);
/** The tokenizer beside the shards of `model_path` (nemotron_enc_weights.cpp).  false = refused (said). */
bool pulsar_nemotron_enc_load_tokenizer(pulsar_nemotron_enc_model *w, const char *model_path);

/* ---- 3. The forward (nemotron_enc_forward.cpp) ------------------------------------------------------------------ */

/** The device state for inputs of up to the shape's cap: scratch, the YaRN table, the layer's attention declaration.
 *  false = refused (said). */
bool pulsar_nemotron_enc_run_create(pulsar_engine *e);
void pulsar_nemotron_enc_run_free(pulsar_nemotron_enc_model *w);

/** The encoder entries (pulsar.h pulsar_engine_encode*). */
extern const pulsar_family_encoder PULSAR_NEMOTRON_ENC_ENCODER;

#endif /* PULSAR_FAMILY_NEMOTRON_ENC_H */

/* nemotron_enc_shape.cpp -- the Nemotron-3.5 encoders' shape: the container's nemotron_enc.* keys and the builder's
 * pulsar.* facts -> pulsar_nemotron_enc_shape, checked against the known profile, and the layer plan (nemotron_enc.h
 * section 1).
 *
 * Rule 1 for a config, as Nemotron-H's: a key the declaration reads is required; a geometry value other than the
 * profile's, a switch other than the value the forward implements, and a key this table does not know are each
 * refused by name; a key with no meaning for the forward is NAMED here with why.  The table is the ONE list of what
 * the family understands: the reader, the pins and the unknown-key check all walk it. */
/** @file nemotron_enc_shape.cpp
 *  @brief The Nemotron-3.5 encoders' shape: the container's nemotron_enc.* keys read and pinned, and the layer plan. */
#include "nemotron_enc.h"

#include <math.h>
#include <stddef.h>

const pulsar_nemotron_enc_shape PULSAR_NEMOTRON_ENC_SHAPE_8B = {
    /* .name           = */ "Nemotron-3.5-8B encoder",
    /* .n_embd         = */ 4096,
    /* .n_vocab        = */ 131072,
    /* .n_head         = */ 32,
    /* .n_head_kv      = */ 8,
    /* .head_dim       = */ 128,
    /* .n_ff           = */ 14336,
    /* .rms_eps        = */ 1e-5f,
    /* .rope_theta     = */ 1e6f,
    /* .rope_factor    = */ 16.0f,
    /* .rope_orig_ctx  = */ 16384,
    /* .rope_beta_fast = */ 32.0f,
    /* .rope_beta_slow = */ 1.0f,
    /* .eos_id         = */ 2,
    /* .n_layer        = */ 34,
    /* .kind           = */ PULSAR_ENCODER_EMBED,
};

/// The prefix of every config refusal.
#define NE "nemotron_enc: "

namespace {

/** What a key is: a number read into the shape (pinned to the profile, or the layer count), a switch pinned to the
 *  one value the forward implements, or a key (or, ending in '.', a key prefix) named as meaningless to it. */
enum ne_kind : uint8_t { K_U32, K_F32, K_BOOL, K_STR, K_U32_EQ, K_F32_EQ, K_IGNORED };

/** One entry of the family's key table: which container key, how it is read and checked, and where it lands. */
struct ne_key {
    const char *key;      ///< the container key (K_IGNORED ending in '.': every key under it)
    /// How the key is read and checked
    ne_kind kind;
    size_t off;           ///< the shape field (K_U32 / K_F32)
    bool pinned;          ///< the field must equal the profile's
    bool want_bool;       ///< K_BOOL's one value
    const char *want;     ///< K_STR's one value; K_IGNORED's reason
    double want_num;      ///< K_U32_EQ / K_F32_EQ's one value
};

/// A shape field's byte offset, for k_keys.
#define F(field) offsetof(pulsar_nemotron_enc_shape, field)
/// A u32 geometry key, read into `field` and pinned to the profile.
#define PIN(k, field) {"nemotron_enc." k, K_U32, F(field), true, false, NULL, 0}
/// An f32 geometry key, read into `field` and pinned to the profile.
#define PIN_F(k, field) {"nemotron_enc." k, K_F32, F(field), true, false, NULL, 0}
/// A bool switch that must be `v`.
#define SW_B(k, v) {"nemotron_enc." k, K_BOOL, 0, false, v, NULL, 0}
/// A string switch that must be `v`.
#define SW_S(k, v) {"nemotron_enc." k, K_STR, 0, false, false, v, 0}
/// An integer switch that must equal `v`.
#define SW_U(k, v) {"nemotron_enc." k, K_U32_EQ, 0, false, false, NULL, v}
/// A float switch that must equal `v`.
#define SW_F(k, v) {"nemotron_enc." k, K_F32_EQ, 0, false, false, NULL, v}
/// A key (or, ending in '.', a prefix) named as meaningless to the forward, for the reason `why`.
#define IGN(k, why) {"nemotron_enc." k, K_IGNORED, 0, false, false, why, 0}
/// The ONE list of keys the family understands; the reader, the profile pins and the unknown-key check walk it.
const ne_key k_keys[] = {
    /* geometry, pinned */
    PIN("hidden_size", n_embd),
    PIN("vocab_size", n_vocab),
    PIN("num_attention_heads", n_head),
    PIN("num_key_value_heads", n_head_kv),
    PIN("head_dim", head_dim),
    PIN("intermediate_size", n_ff),
    PIN("eos_token_id", eos_id),
    PIN_F("rms_norm_eps", rms_eps),
    /* the rope: YaRN, pinned (the table nemotron_enc_forward.cpp builds is for these numbers) */
    PIN_F("rope_parameters.rope_theta", rope_theta),
    PIN_F("rope_parameters.factor", rope_factor),
    PIN("rope_parameters.original_max_position_embeddings", rope_orig_ctx),
    PIN_F("rope_parameters.beta_fast", rope_beta_fast),
    PIN_F("rope_parameters.beta_slow", rope_beta_slow),
    SW_S("rope_parameters.rope_type", "yarn"),
    SW_S("rope_parameters.type", "yarn"),
    SW_F("rope_parameters.mscale", 1.0),          /* with mscale_all_dim 1: the attention factor is exactly 1 */
    SW_F("rope_parameters.mscale_all_dim", 1.0),
    SW_B("rope_parameters.apply_yarn_scaling", false),
    /* transformers' llama-4 q scale: 1 + beta ln(1 + floor(pos / 16384)) -- exactly 1 below the input cap */
    SW_F("rope_parameters.llama_4_scaling_beta", 0.1),
    SW_F("llama_4_scaling.beta", 0.1),
    SW_U("llama_4_scaling.original_max_position_embeddings", 16384),
    /* the plan, the container's */
    {"nemotron_enc.num_hidden_layers", K_U32, F(n_layer), false, false, NULL, 0},
    /* the switches: the one value the forward implements */
    SW_S("model_type", "ministral3"),
    SW_S("dtype", "bfloat16"),
    SW_S("hidden_act", "silu"),
    SW_S("pooling", "avg"),                       /* the mean over the input's rows */
    SW_B("is_causal", false),                     /* bidirectional: every row attends every row of its input */
    SW_B("is_encoder_decoder", false),
    SW_B("tie_word_embeddings", false),
    /* named, with no meaning for the forward */
    IGN("_name_or_path", "the checkpoint's provenance (pulsar.source.hf is the pin)"),
    IGN("architectures", "the trunk's class name; pulsar.encoder is the builder's fact"),
    IGN("nemo_version", "the training framework's version"),
    IGN("attention_dropout", "training only"),
    IGN("initializer_range", "initialisation only"),
    IGN("chunk_size_feed_forward", "a transformers memory option"),
    IGN("output_attentions", "a transformers output option"),
    IGN("output_hidden_states", "a transformers output option"),
    IGN("return_dict", "a transformers output option"),
    IGN("use_cache", "a transformers generation option"),
    IGN("bos_token_id", "the template renders no BOS"),
    IGN("pad_token_id", "padding: no batch here pads (each input is its own run)"),
    IGN("max_position_embeddings", "the rope's reach; an input is capped at the original context (the llama-4 scale)"),
    IGN("id2label.", "the score head's label name (one logit)"),
    IGN("label2id.", "the score head's label name (one logit)"),
};
#undef F
#undef PIN
#undef PIN_F
#undef SW_B
#undef SW_S
#undef SW_U
#undef SW_F
#undef IGN

/** The container's metadata entry named `key` (linear scan).
 *  @param m    the container
 *  @param key  the full key name
 *  @return the entry, or NULL when absent */
const pulsar_kv *ne_kv(const pulsar_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++)
        if (pulsar_streq(m->kv[i].key, key)) return &m->kv[i];
    return NULL;
}

/** Say one config refusal and count it (pulsar_load_refuse).
 *  @param fmt   a printf format taking two strings: `key`, then `more`
 *  @param key   the key refused
 *  @param more  the detail appended to the message
 *  @return always false */
bool refuse(const char *fmt, const char *key, const char *more = "") {
    fprintf(stderr, "pulsar: " NE);
    fprintf(stderr, fmt, key, more);
    fprintf(stderr, " -- refusing\n");
    pulsar_load_refuse();
    return false;
}

/** @param key  a table key (non-empty)
 *  @return true when it ends in '.': a prefix entry naming every key under it */
bool is_prefix(const char *key) { return key[strlen(key) - 1] == '.'; }

/* the key is the table entry's (a prefix entry: the key lies under it) */
/** @param k  a container key
 *  @param d  a table entry
 *  @return true when k is d's key, or lies strictly under d's prefix */
bool matches(pulsar_str k, const ne_key &d) {
    if (!is_prefix(d.key)) return pulsar_streq(k, d.key);
    const size_t n = strlen(d.key);
    return k.len > n && memcmp(k.ptr, d.key, n) == 0;
}

/* the builder's facts: the encoder's kind, its pooling and its normalisation (which must be the kind's), no tower */
/** @param m  the container
 *  @param s  receives the kind
 *  @return true = read and consistent; false = refused (said and counted) */
bool read_facts(const pulsar_model *m, pulsar_nemotron_enc_shape *s) {
    bool ok = true;
    pulsar_str kind = {NULL, 0}, pool = {NULL, 0};
    bool normalize = false, vision = true;
    if (!model_get_string(m, "pulsar.encoder", &kind) || !model_get_string(m, "pulsar.pooling", &pool) ||
        !model_get_bool(m, "pulsar.normalize", &normalize) || !model_get_bool(m, "pulsar.vision_present", &vision))
        return refuse("%s / pulsar.pooling / pulsar.normalize / pulsar.vision_present: the container does not say what "
                      "the encoder returns%s", "pulsar.encoder");
    if (pulsar_streq(kind, "embed")) s->kind = PULSAR_ENCODER_EMBED;
    else if (pulsar_streq(kind, "rerank")) s->kind = PULSAR_ENCODER_RERANK;
    else ok = refuse("%s is not \"embed\" or \"rerank\"%s", "pulsar.encoder");
    if (!pulsar_streq(pool, "mean")) ok = refuse("%s is not \"mean\" (the one pooling the forward implements)%s",
                                                 "pulsar.pooling");
    if (ok && normalize != (s->kind == PULSAR_ENCODER_EMBED))
        ok = refuse("%s disagrees with the kind (an embedder L2-normalises, a reranker scores the raw mean)%s",
                    "pulsar.normalize");
    if (vision) ok = refuse("%s is true: the Pixtral tower is not part of this declaration (v1: text)%s",
                            "pulsar.vision_present");
    return ok;
}

}  // namespace

/** @param m  the container
 *  @param s  receives the shape (the profile's values first, every read field overwritten)
 *  @return true = read and checked; false = at least one refusal (each said and counted) */
bool pulsar_nemotron_enc_read_shape(const pulsar_model *m, pulsar_nemotron_enc_shape *s) {
    *s = PULSAR_NEMOTRON_ENC_SHAPE_8B;   /* the name and the profile's values; every read field is overwritten */
    bool ok = read_facts(m, s);
    /* every nemotron_enc.* key is one of the table's */
    for (uint64_t i = 0; i < m->n_kv; i++) {
        const pulsar_str k = m->kv[i].key;
        if (k.len < 13 || memcmp(k.ptr, "nemotron_enc.", 13) != 0) continue;
        bool known = false;
        for (const ne_key &d : k_keys) known |= matches(k, d);
        if (!known) {
            char key[160];
            snprintf(key, sizeof(key), "%.*s", (int)(k.len < 150 ? k.len : 150), k.ptr);
            ok = refuse("%s is a config key the encoder declaration does not know (a switch the forward does not "
                        "implement)%s", key);
        }
    }
    for (const ne_key &d : k_keys) {
        if (d.kind == K_IGNORED) continue;
        const pulsar_kv *kv = ne_kv(m, d.key);
        if (!kv) {
            ok = refuse("required config key %s is missing%s", d.key);
            continue;
        }
        uint8_t *field = (uint8_t *)s + d.off;
        const uint8_t *want = (const uint8_t *)&PULSAR_NEMOTRON_ENC_SHAPE_8B + d.off;
        char got[96] = "", exp[96] = "";
        bool match = true, read = true;
        switch (d.kind) {
        case K_U32:
        case K_U32_EQ: {
            uint64_t v = 0;
            read = model_get_u64_compat(m, d.key, &v) && v <= UINT32_MAX;
            if (!read) break;
            uint64_t w = (uint64_t)d.want_num;
            if (d.kind == K_U32) {
                *(uint32_t *)field = (uint32_t)v;
                w = *(const uint32_t *)want;
            }
            match = (d.kind == K_U32_EQ || d.pinned) ? v == w : true;
            snprintf(got, sizeof(got), "%llu", (unsigned long long)v);
            snprintf(exp, sizeof(exp), "%llu", (unsigned long long)w);
            break;
        }
        case K_F32:
        case K_F32_EQ: {
            float v = 0.0f;
            read = model_get_f32_compat(m, d.key, &v);
            if (!read) break;
            const float w = d.kind == K_F32 ? *(const float *)want : (float)d.want_num;
            if (d.kind == K_F32) *(float *)field = v;
            match = fabsf(v - w) <= fabsf(w) * 1.0e-6f;
            snprintf(got, sizeof(got), "%.9g", (double)v);
            snprintf(exp, sizeof(exp), "%.9g", (double)w);
            break;
        }
        case K_BOOL: {
            bool v = false;
            read = model_get_bool(m, d.key, &v);
            match = v == d.want_bool;
            snprintf(got, sizeof(got), "%s", v ? "true" : "false");
            snprintf(exp, sizeof(exp), "%s", d.want_bool ? "true" : "false");
            break;
        }
        case K_STR: {
            pulsar_str v = {NULL, 0};
            read = model_get_string(m, d.key, &v);
            match = read && pulsar_streq(v, d.want);
            snprintf(got, sizeof(got), "'%.*s'", (int)(v.len < 80 ? v.len : 80), v.ptr ? v.ptr : "");
            snprintf(exp, sizeof(exp), "'%s'", d.want);
            break;
        }
        case K_IGNORED: break;
        }
        if (!read) {
            ok = refuse("config key %s has the wrong type%s", d.key);
        } else if (!match) {
            char why[224];
            snprintf(why, sizeof(why), " = %s; %s %s", got,
                     d.pinned ? PULSAR_NEMOTRON_ENC_SHAPE_8B.name : "the forward implements", exp);
            ok = refuse("%s%s", d.key, why);
        }
    }
    if (s->n_layer == 0 || s->n_layer > PULSAR_FAMILY_MAX_LAYER) {
        char why[64];
        snprintf(why, sizeof(why), " = %u (1 .. %u)", s->n_layer, PULSAR_FAMILY_MAX_LAYER);
        ok = refuse("%s%s", "nemotron_enc.num_hidden_layers", why);
        s->n_layer = 0;
    }
    return ok;
}

/** @param s     the checked shape
 *  @param plan  receives n_layer layer descriptions */
void pulsar_nemotron_enc_plan(const pulsar_nemotron_enc_shape *s, pulsar_layer_plan *plan) {
    memset(plan, 0, sizeof(*plan));
    plan->n_layer = s->n_layer;
    for (uint32_t il = 0; il < s->n_layer; il++) {
        plan->kind[il] = PULSAR_LAYER_BLOCKS;
        plan->desc[il].mixer = PULSAR_MIXER_ATTN_FULL;
        plan->desc[il].ffn = PULSAR_FFN_DENSE;
    }
}

/* gemma4_shape.cpp -- the Gemma 4 family's shape: the container's gemma4.* keys -> pulsar_gemma4_shape, checked
 * against the known profile, and the layer plan (gemma4.h section 1).
 *
 * Rule 1 for a config: a key the declaration reads is required; a geometry value other than the profile's, a switch
 * other than the value the blocks implement, and a key this table does not know are each refused by name.  The
 * table below is the ONE list of what the family understands: the reader, the pins and the unknown-key check all
 * walk it. */
/** @file gemma4_shape.cpp
 *  @brief Gemma 4's shape: the container's gemma4.* keys read, checked against the profile, and the layer plan. */
#include "gemma4.h"

#include <math.h>
#include <stddef.h>

const pulsar_gemma4_shape PULSAR_GEMMA4_SHAPE_26B_A4B = {
    /* .name           = */ "Gemma-4-26B-A4B",
    /* .n_embd         = */ 2816,
    /* .n_vocab        = */ 262144,
    /* .n_head         = */ 16,
    /* .swa_n_head_kv  = */ 8,
    /* .swa_head_dim   = */ 256,
    /* .n_head_kv      = */ 2,
    /* .head_dim       = */ 512,
    /* .window         = */ 1024,
    /* .n_ff           = */ 2112,
    /* .n_ff_exp       = */ 704,
    /* .n_expert       = */ 128,
    /* .n_expert_used  = */ 8,
    /* .rms_eps        = */ 1e-6f,
    /* .softcap        = */ 30.0f,
    /* .swa_rope_theta = */ 10000.0f,
    /* .rope_theta     = */ 1000000.0f,
    /* .rope_partial   = */ 0.25f,
    /* .max_context    = */ 262144,
    /* .bos_id         = */ 2,
    /* .eos_id         = */ 1,
    /* .pad_id         = */ 0,
};

/// The prefix of every config refusal.
#define GEM "gemma4: "

namespace {

/** What a key is: a number read into the shape (pinned to the profile, or a plan fact read as is), a switch pinned
 *  to the one value the blocks implement (a bool, a string, or a count that must be zero), the plan's list, or a key
 *  known and not read (training-only: inference has no dropout, no init, no cache flag to honour). */
enum gem_kind : uint8_t { K_U32, K_U64, K_F32, K_BOOL, K_STR, K_ZERO, K_LIST, K_INFO };

/** One entry of the family's key table: which container key, how it is read and checked, and where it lands. */
struct gem_key {
    const char *key;      ///< the container key
    /// How the key is read and checked
    gem_kind kind;
    size_t off;           ///< the shape field (K_U32 / K_U64 / K_F32)
    bool pinned;          ///< the field must equal the profile's
    bool want_bool;       ///< K_BOOL's one value
    const char *want_str; ///< K_STR's one value
};

/// A shape field's byte offset, for k_keys.
#define F(field) offsetof(pulsar_gemma4_shape, field)
/// The rope blocks' key prefix.
#define ROPE "gemma4.rope_parameters."
/// The ONE list of keys the family understands; the reader, the profile pins and the unknown-key check walk it.
const gem_key k_keys[] = {
    /* geometry, pinned */
    {"gemma4.hidden_size", K_U32, F(n_embd), true, false, NULL},
    {"gemma4.vocab_size", K_U32, F(n_vocab), true, false, NULL},
    {"gemma4.num_attention_heads", K_U32, F(n_head), true, false, NULL},
    {"gemma4.num_key_value_heads", K_U32, F(swa_n_head_kv), true, false, NULL},
    {"gemma4.head_dim", K_U32, F(swa_head_dim), true, false, NULL},
    {"gemma4.num_global_key_value_heads", K_U32, F(n_head_kv), true, false, NULL},
    {"gemma4.global_head_dim", K_U32, F(head_dim), true, false, NULL},
    {"gemma4.sliding_window", K_U32, F(window), true, false, NULL},
    {"gemma4.intermediate_size", K_U32, F(n_ff), true, false, NULL},
    {"gemma4.moe_intermediate_size", K_U32, F(n_ff_exp), true, false, NULL},
    {"gemma4.num_experts", K_U32, F(n_expert), true, false, NULL},
    {"gemma4.top_k_experts", K_U32, F(n_expert_used), true, false, NULL},
    {"gemma4.rms_norm_eps", K_F32, F(rms_eps), true, false, NULL},
    {"gemma4.final_logit_softcapping", K_F32, F(softcap), true, false, NULL},
    {ROPE "sliding_attention.rope_theta", K_F32, F(swa_rope_theta), true, false, NULL},
    {ROPE "full_attention.rope_theta", K_F32, F(rope_theta), true, false, NULL},
    {ROPE "full_attention.partial_rotary_factor", K_F32, F(rope_partial), true, false, NULL},
    {"gemma4.max_position_embeddings", K_U64, F(max_context), true, false, NULL},
    {"gemma4.bos_token_id", K_U32, F(bos_id), true, false, NULL},
    {"gemma4.eos_token_id", K_U32, F(eos_id), true, false, NULL},
    {"gemma4.pad_token_id", K_U32, F(pad_id), true, false, NULL},
    /* the plan, the container's */
    {"gemma4.num_hidden_layers", K_U32, F(n_layer), false, false, NULL},
    {"gemma4.layer_types", K_LIST, 0, false, false, NULL},
    /* the switches: the one value each block implements */
    {"gemma4.model_type", K_STR, 0, false, false, "gemma4_text"},
    {"gemma4.dtype", K_STR, 0, false, false, "bfloat16"},
    {"gemma4.hidden_activation", K_STR, 0, false, false, "gelu_pytorch_tanh"},
    {"gemma4.use_bidirectional_attention", K_STR, 0, false, false, "vision"},   /* vision tokens only; v1 has none */
    {ROPE "sliding_attention.rope_type", K_STR, 0, false, false, "default"},
    {ROPE "full_attention.rope_type", K_STR, 0, false, false, "proportional"},
    {"gemma4.attention_k_eq_v", K_BOOL, 0, false, true, NULL},
    {"gemma4.enable_moe_block", K_BOOL, 0, false, true, NULL},
    {"gemma4.tie_word_embeddings", K_BOOL, 0, false, true, NULL},
    {"gemma4.attention_bias", K_BOOL, 0, false, false, NULL},
    {"gemma4.use_double_wide_mlp", K_BOOL, 0, false, false, NULL},
    {"gemma4.hidden_size_per_layer_input", K_ZERO, 0, false, false, NULL},   /* no per-layer inputs */
    {"gemma4.num_kv_shared_layers", K_ZERO, 0, false, false, NULL},          /* every layer has its own KV */
    /* known, not read */
    {"gemma4.attention_dropout", K_INFO, 0, false, false, NULL},
    {"gemma4.initializer_range", K_INFO, 0, false, false, NULL},
    {"gemma4.use_cache", K_INFO, 0, false, false, NULL},
    {"gemma4.vocab_size_per_layer_input", K_INFO, 0, false, false, NULL},    /* the per-layer inputs' table (none) */
};
#undef F
#undef ROPE

/** The container's metadata entry named `key` (linear scan).
 *  @param m    the container
 *  @param key  the full key name
 *  @return the entry, or NULL when absent */
const pulsar_kv *gem_kv(const pulsar_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++)
        if (pulsar_streq(m->kv[i].key, key)) return &m->kv[i];
    return NULL;
}

/** Say one config refusal and count it (pulsar_load_refuse).
 *  @param fmt   a printf format taking two strings: `key`, then `more`
 *  @param key   the key refused
 *  @param more  the detail appended to the message
 *  @return always false, so callers can `return refuse(...)` */
bool refuse(const char *fmt, const char *key, const char *more = "") {
    fprintf(stderr, "pulsar: " GEM);
    fprintf(stderr, fmt, key, more);
    fprintf(stderr, " -- refusing\n");
    pulsar_load_refuse();
    return false;
}

/** The plan's layer types: `n` strings, each "sliding_attention" or "full_attention", as a bitmask of the full ones.
 *  @param m     the container
 *  @param n     the layer count
 *  @param mask  receives bit il for each full-attention layer
 *  @return true = read; false = refused (said and counted) */
bool read_layer_types(const pulsar_model *m, uint32_t n, uint64_t *mask) {
    const char *key = "gemma4.layer_types";
    pulsar_array_ref arr;
    *mask = 0;
    if (!model_get_array(m, key, &arr) || arr.type != PULSAR_META_STRING || arr.len != n)
        return refuse("%s is missing or not an array of one string per layer%s", key);
    pulsar_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t il = 0; il < n; il++) {
        pulsar_str t = {NULL, 0};
        if (!cursor_string(&c, &t)) return refuse("%s is unreadable%s", key);
        if (pulsar_streq(t, "full_attention")) {
            *mask |= 1ull << il;
        } else if (!pulsar_streq(t, "sliding_attention")) {
            char why[96];
            snprintf(why, sizeof(why), "[%u] = '%.*s': sliding_attention or full_attention", il, (int)(t.len < 40 ? t.len : 40),
                     t.ptr);
            return refuse("%s%s", key, why);
        }
    }
    if (*mask == 0) return refuse("%s names no full_attention layer%s", key);
    return true;
}

}  // namespace

/** @param m  the container
 *  @param s  receives the shape (the profile's values first, every read field overwritten)
 *  @return true = read and checked; false = at least one refusal (each said and counted) */
bool pulsar_gemma4_read_shape(const pulsar_model *m, pulsar_gemma4_shape *s) {
    *s = PULSAR_GEMMA4_SHAPE_26B_A4B;   /* the name and the profile's values; every read field is overwritten */
    bool ok = true;
    if (!model_get_bool(m, "pulsar.vision_present", &s->vision_written))
        ok = refuse("%s: the container does not say whether it holds the tower%s", "pulsar.vision_present");
    if (s->vision_written) ok = refuse("%s is true: the Gemma 4 family serves text only (no image front)%s", "pulsar.vision_present");
    /* every gemma4.* key is one of the table's */
    for (uint64_t i = 0; i < m->n_kv; i++) {
        const pulsar_str k = m->kv[i].key;
        if (k.len < 7 || memcmp(k.ptr, "gemma4.", 7) != 0) continue;
        bool known = false;
        for (const gem_key &d : k_keys) known |= pulsar_streq(k, d.key);
        if (!known) {
            char key[160];
            snprintf(key, sizeof(key), "%.*s", (int)(k.len < 150 ? k.len : 150), k.ptr);
            ok = refuse("%s is a config key the Gemma 4 declaration does not know (a switch no block here "
                        "implements)%s", key);
        }
    }
    for (const gem_key &d : k_keys) {
        if (d.kind == K_INFO || d.kind == K_LIST) continue;   /* known-and-unread; the list after the numbers */
        if (!gem_kv(m, d.key)) {
            ok = refuse("required config key %s is missing%s", d.key);
            continue;
        }
        uint8_t *field = (uint8_t *)s + d.off;
        const uint8_t *want = (const uint8_t *)&PULSAR_GEMMA4_SHAPE_26B_A4B + d.off;
        char got[96] = "", exp[96] = "";
        bool match = true, read = true;
        switch (d.kind) {
        case K_U32:
        case K_U64: {
            uint64_t v = 0;
            read = model_get_u64_compat(m, d.key, &v) && (d.kind == K_U64 || v <= UINT32_MAX);
            if (!read) break;
            const uint64_t w = d.kind == K_U64 ? *(const uint64_t *)want : *(const uint32_t *)want;
            if (d.kind == K_U64) *(uint64_t *)field = v; else *(uint32_t *)field = (uint32_t)v;
            match = !d.pinned || v == w;
            snprintf(got, sizeof(got), "%llu", (unsigned long long)v);
            snprintf(exp, sizeof(exp), "%llu", (unsigned long long)w);
            break;
        }
        case K_ZERO: {
            uint64_t v = 0;
            read = model_get_u64_compat(m, d.key, &v);
            match = v == 0;
            snprintf(got, sizeof(got), "%llu", (unsigned long long)v);
            snprintf(exp, sizeof(exp), "0");
            break;
        }
        case K_F32: {
            float v = 0.0f;
            read = model_get_f32_compat(m, d.key, &v);
            if (!read) break;
            const float w = *(const float *)want;
            *(float *)field = v;
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
            match = read && pulsar_streq(v, d.want_str);
            snprintf(got, sizeof(got), "'%.*s'", (int)(v.len < 80 ? v.len : 80), v.ptr ? v.ptr : "");
            snprintf(exp, sizeof(exp), "'%s'", d.want_str);
            break;
        }
        default: break;
        }
        if (!read) {
            ok = refuse("config key %s has the wrong type%s", d.key);
        } else if (!match) {
            char why[224];
            snprintf(why, sizeof(why), " = %s; %s %s", got, d.pinned ? PULSAR_GEMMA4_SHAPE_26B_A4B.name : "the blocks implement",
                     exp);
            ok = refuse("%s%s", d.key, why);
        }
    }
    /* the plan */
    if (s->n_layer == 0 || s->n_layer > PULSAR_FAMILY_MAX_LAYER) {
        char why[64];
        snprintf(why, sizeof(why), " = %u: 1 .. %u layers", s->n_layer, PULSAR_FAMILY_MAX_LAYER);
        ok = refuse("%s%s", "gemma4.num_hidden_layers", why);
        s->n_layer = 0;
    }
    if (s->n_layer) ok &= read_layer_types(m, s->n_layer, &s->full_layers);
    return ok;
}

/** @param s     the checked shape
 *  @param plan  receives the n_layer layer descriptions */
void pulsar_gemma4_plan(const pulsar_gemma4_shape *s, pulsar_layer_plan *plan) {
    memset(plan, 0, sizeof(*plan));
    plan->n_layer = s->n_layer;
    for (uint32_t il = 0; il < s->n_layer; il++) {
        const bool full = pulsar_gemma4_is_full(s, il);
        plan->kind[il] = PULSAR_LAYER_BLOCKS;
        plan->desc[il].mixer = full ? PULSAR_MIXER_ATTN_FULL : PULSAR_MIXER_ATTN_WINDOW;
        plan->desc[il].ffn = PULSAR_FFN_DENSE_MOE;
        plan->desc[il].window = full ? 0u : s->window;
    }
}

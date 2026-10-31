/* nemotron_parse_shape.cpp -- Parse's shape: the container's nemotron_parse.* facts (the builder pins every config
 * value they come from) read and pinned again against the profile (nemotron_parse.h section 1).  Every key is one of
 * the table's; every value the profile's -- a second geometry is a second profile, added on purpose. */
/** @file nemotron_parse_shape.cpp
 *  @brief Parse's shape: the container's nemotron_parse.* facts read and pinned against the profile. */
#include "nemotron_parse.h"

#include <math.h>
#include <stddef.h>

const pulsar_nemotron_parse_shape PULSAR_NEMOTRON_PARSE_SHAPE = {
    /* vit  */ 1152, 27, 16, 72, 4304, 16, 10, 128, 1e-6f,
    /* neck */ 1024, 4, 1e-6f,
    /* dec  */ 1024, 10, 16, 8, 4096, 72512, 1e-5f, 32.0f,
    /* ids  */ 16384, 0, 2, 1,
    /* page */ 2048, 1664, 16, 64,
    /* mean */ {0.48145466f, 0.4578275f, 0.40821073f},
    /* std  */ {0.26862954f, 0.26130258f, 0.27577711f},
    /* gen  */ 1.1f, 9000,
    /* prompt */ "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>",
};

/// The prefix of every fact refusal.
#define NP "nemotron_parse: "

namespace {

/** How a fact is read: an unsigned integer, a float, or a string (the prompt). */
enum np_kind : uint8_t { K_U32, K_F32, K_STR };
/** One fact of the table: the container key, its kind, and its field in the shape. */
struct np_key {
    /// the container key
    const char *key;
    /// how it is read
    np_kind kind;
    /// the field's offset in pulsar_nemotron_parse_shape
    size_t off;
};

/// A u32 fact read into field `f`.
#define U(k, f) {"nemotron_parse." k, K_U32, offsetof(pulsar_nemotron_parse_shape, f)}
/// An f32 fact read into field `f`.
#define F(k, f) {"nemotron_parse." k, K_F32, offsetof(pulsar_nemotron_parse_shape, f)}
/// The ONE list of facts the family reads; every one is required and pinned to the profile.
const np_key k_keys[] = {
    U("vit.width", vit_width), U("vit.depth", vit_depth), U("vit.heads", vit_heads), U("vit.head_dim", vit_head_dim),
    U("vit.mlp", vit_mlp), U("vit.patch", patch), U("vit.n_cls", n_cls),
    U("vit.pos_grid", pos_grid), F("vit.norm_eps", vit_eps),
    U("neck.width", neck_width), U("neck.merge", merge), F("neck.norm_eps", neck_eps),
    U("dec.width", width), U("dec.layers", n_layer), U("dec.heads", heads), U("dec.kv_heads", kv_heads),
    U("dec.ffn", ffn), U("dec.vocab", vocab), F("dec.norm_eps", eps), F("dec.embed_scale", embed_scale),
    U("max_sequence", max_seq), U("bos", bos), U("eos", eos), U("pad", pad),
    U("image.canvas_h", canvas_h), U("image.canvas_w", canvas_w), U("image.align_h", align_h),
    U("image.align_w", align_w),
    F("image.mean.0", mean[0]), F("image.mean.1", mean[1]), F("image.mean.2", mean[2]),
    F("image.std.0", std[0]), F("image.std.1", std[1]), F("image.std.2", std[2]),
    F("gen.repetition_penalty", rep_penalty), U("gen.max_new_tokens", max_new),
    {"nemotron_parse.gen.prompt", K_STR, offsetof(pulsar_nemotron_parse_shape, prompt)},
};
#undef U
#undef F

/** Say one fact refusal and count it (pulsar_load_refuse).
 *  @param fmt   a printf format taking two strings: `key`, then `more`
 *  @param key   the key refused
 *  @param more  the detail appended to the message
 *  @return always false */
bool refuse(const char *fmt, const char *key, const char *more = "") {
    fprintf(stderr, "pulsar: " NP);
    fprintf(stderr, fmt, key, more);
    fprintf(stderr, " -- refusing\n");
    pulsar_load_refuse();
    return false;
}

}  // namespace

/** @param m  the container
 *  @param s  receives the shape (the profile's values first, every read fact overwritten)
 *  @return true = read and every fact the profile's; false = at least one refusal (each said and counted) */
bool pulsar_nemotron_parse_read_shape(const pulsar_model *m, pulsar_nemotron_parse_shape *s) {
    *s = PULSAR_NEMOTRON_PARSE_SHAPE;
    bool ok = true;
    for (uint64_t i = 0; i < m->n_kv; i++) {
        const pulsar_str k = m->kv[i].key;
        if (k.len < 15 || memcmp(k.ptr, "nemotron_parse.", 15) != 0) continue;
        bool known = false;
        for (const np_key &d : k_keys) known |= pulsar_streq(k, d.key);
        if (!known) {
            char key[160];
            snprintf(key, sizeof(key), "%.*s", (int)(k.len < 150 ? k.len : 150), k.ptr);
            ok = refuse("%s is a fact the Parse declaration does not know%s", key);
        }
    }
    for (const np_key &d : k_keys) {
        uint8_t *field = (uint8_t *)s + d.off;
        const uint8_t *want = (const uint8_t *)&PULSAR_NEMOTRON_PARSE_SHAPE + d.off;
        char why[200] = "";
        switch (d.kind) {
        case K_U32: {
            uint64_t v = 0;
            if (!model_get_u64_compat(m, d.key, &v)) {
                ok = refuse("required key %s is missing or not an integer%s", d.key);
                break;
            }
            if (v != *(const uint32_t *)want) {
                snprintf(why, sizeof why, " = %llu; the profile has %u", (unsigned long long)v, *(const uint32_t *)want);
                ok = refuse("%s%s", d.key, why);
            }
            *(uint32_t *)field = (uint32_t)v;
            break;
        }
        case K_F32: {
            float v = 0.0f;
            if (!model_get_f32_compat(m, d.key, &v)) {
                ok = refuse("required key %s is missing or not a number%s", d.key);
                break;
            }
            const float w = *(const float *)want;
            if (!(fabsf(v - w) <= fabsf(w) * 1e-6f)) {
                snprintf(why, sizeof why, " = %.9g; the profile has %.9g", (double)v, (double)w);
                ok = refuse("%s%s", d.key, why);
            }
            *(float *)field = v;
            break;
        }
        case K_STR: {
            pulsar_str v = {NULL, 0};
            if (!model_get_string(m, d.key, &v) || v.len >= sizeof(s->prompt)) {
                ok = refuse("required key %s is missing, not a string or longer than the prompt buffer%s", d.key);
                break;
            }
            snprintf(s->prompt, sizeof s->prompt, "%.*s", (int)v.len, v.ptr);
            break;
        }
        }
    }
    return ok;
}

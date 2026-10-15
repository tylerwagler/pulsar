/* inkling_shape.cpp -- the Inkling family's shape: the container's inkling.* keys -> pulsar_inkling_shape, checked
 * against the known profile, and the layer plan (inkling.h section 1).
 *
 * Rule 1 for a config: a key the declaration reads is required; a geometry value other than the profile's, a switch
 * other than the value the blocks implement, and a key this table does not know are each refused by name.  The
 * table below is the ONE list of what the family understands: the reader, the pins and the unknown-key check all
 * walk it. */
#include "inkling.h"

#include <math.h>
#include <stddef.h>

const pulsar_inkling_shape PULSAR_INKLING_SHAPE_SMALL = {
    /* .name            = */ "Inkling-Small",
    /* .n_embd          = */ 4096,
    /* .n_vocab         = */ 201024,
    /* .n_vocab_real    = */ 200058,
    /* .n_head          = */ 32,
    /* .n_head_kv       = */ 8,
    /* .head_dim        = */ 128,
    /* .rel_extent      = */ 1024,
    /* .swa_n_head      = */ 32,
    /* .swa_n_head_kv   = */ 8,
    /* .swa_head_dim    = */ 128,
    /* .window          = */ 512,
    /* .d_rel           = */ 16,
    /* .conv_k          = */ 4,
    /* .n_ff_dense      = */ 16384,
    /* .n_ff_exp        = */ 2048,
    /* .n_expert        = */ 256,
    /* .n_expert_used   = */ 6,
    /* .n_shared        = */ 2,
    /* .rms_eps         = */ 1e-6f,
    /* .route_scale     = */ 8.0f,
    /* .logit_div       = */ 16.0f,
    /* .log_scale_alpha = */ 0.1f,
    /* .log_scale_floor = */ 128000,
    /* .max_context     = */ 1048576,
    /* .eos_id          = */ 200006,
    /* .vis_patch       = */ 40,
    /* .vis_temporal    = */ 2,
    /* .vis_channels    = */ 3,
    /* .vis_layers      = */ 4,
    /* .vis_in          = */ {75, 512, 5120, 9600},
    /* .vis_mid         = */ {128, 320, 4800},
    /* .n_mtp_max       = */ 8,
};

#define INK "inkling: "

namespace {

/** What a key is: a number read into the shape (pinned to the profile, or a plan fact read as is), a switch pinned
 *  to the one value the blocks implement, a list the plan reads, or a key that must be absent. */
enum ink_kind : uint8_t { K_U32, K_U64, K_F32, K_BOOL, K_STR, K_LIST, K_ABSENT, K_EQ_EXP };
/** Which block of keys: the text config (always), the tower's (when written), the drafter's (when present). */
enum ink_group : uint8_t { G_TEXT, G_VISION, G_MTP };

struct ink_key {
    const char *key;      ///< the container key
    ink_kind kind;
    ink_group group;
    size_t off;           ///< the shape field (K_U32 / K_U64 / K_F32)
    bool pinned;          ///< the field must equal the profile's
    bool want_bool;       ///< K_BOOL's one value
    const char *want_str; ///< K_STR's one value
};

#define F(field) offsetof(pulsar_inkling_shape, field)
const ink_key k_keys[] = {
    /* geometry, pinned */
    {"inkling.hidden_size", K_U32, G_TEXT, F(n_embd), true, false, NULL},
    {"inkling.vocab_size", K_U32, G_TEXT, F(n_vocab), true, false, NULL},
    {"inkling.unpadded_vocab_size", K_U32, G_TEXT, F(n_vocab_real), true, false, NULL},
    {"inkling.num_attention_heads", K_U32, G_TEXT, F(n_head), true, false, NULL},
    {"inkling.num_key_value_heads", K_U32, G_TEXT, F(n_head_kv), true, false, NULL},
    {"inkling.head_dim", K_U32, G_TEXT, F(head_dim), true, false, NULL},
    {"inkling.rel_extent", K_U32, G_TEXT, F(rel_extent), true, false, NULL},
    {"inkling.swa_num_attention_heads", K_U32, G_TEXT, F(swa_n_head), true, false, NULL},
    {"inkling.swa_num_key_value_heads", K_U32, G_TEXT, F(swa_n_head_kv), true, false, NULL},
    {"inkling.swa_head_dim", K_U32, G_TEXT, F(swa_head_dim), true, false, NULL},
    {"inkling.sliding_window_size", K_U32, G_TEXT, F(window), true, false, NULL},
    {"inkling.d_rel", K_U32, G_TEXT, F(d_rel), true, false, NULL},
    {"inkling.sconv_kernel_size", K_U32, G_TEXT, F(conv_k), true, false, NULL},
    {"inkling.dense_intermediate_size", K_U32, G_TEXT, F(n_ff_dense), true, false, NULL},
    {"inkling.intermediate_size", K_U32, G_TEXT, F(n_ff_exp), true, false, NULL},
    {"inkling.n_routed_experts", K_U32, G_TEXT, F(n_expert), true, false, NULL},
    {"inkling.num_experts_per_tok", K_U32, G_TEXT, F(n_expert_used), true, false, NULL},
    {"inkling.n_shared_experts", K_U32, G_TEXT, F(n_shared), true, false, NULL},
    {"inkling.rms_norm_eps", K_F32, G_TEXT, F(rms_eps), true, false, NULL},
    {"inkling.route_scale", K_F32, G_TEXT, F(route_scale), true, false, NULL},
    {"inkling.logits_mup_width_multiplier", K_F32, G_TEXT, F(logit_div), true, false, NULL},
    {"inkling.log_scaling_alpha", K_F32, G_TEXT, F(log_scale_alpha), true, false, NULL},
    {"inkling.log_scaling_n_floor", K_U32, G_TEXT, F(log_scale_floor), true, false, NULL},
    {"inkling.model_max_length", K_U64, G_TEXT, F(max_context), true, false, NULL},
    {"inkling.eos_token_id", K_U32, G_TEXT, F(eos_id), true, false, NULL},
    /* the plan, the container's */
    {"inkling.num_hidden_layers", K_U32, G_TEXT, F(n_layer), false, false, NULL},
    {"inkling.dense_mlp_idx", K_U32, G_TEXT, F(n_dense), false, false, NULL},
    {"inkling.local_layer_ids", K_LIST, G_TEXT, 0, false, false, NULL},
    /* the switches: the one value each block implements */
    {"inkling.torch_dtype", K_STR, G_TEXT, 0, false, false, "bfloat16"},
    {"inkling.q_bias", K_BOOL, G_TEXT, 0, false, false, NULL},
    {"inkling.o_bias", K_BOOL, G_TEXT, 0, false, false, NULL},
    {"inkling.use_embed_norm", K_BOOL, G_TEXT, 0, false, true, NULL},
    {"inkling.use_sconv", K_BOOL, G_TEXT, 0, false, true, NULL},
    {"inkling.gate_activation", K_STR, G_TEXT, 0, false, false, "sigmoid"},
    {"inkling.use_gate_bias", K_BOOL, G_TEXT, 0, false, true, NULL},
    {"inkling.shared_expert_sink", K_BOOL, G_TEXT, 0, false, true, NULL},
    {"inkling.norm_after_topk", K_BOOL, G_TEXT, 0, false, true, NULL},
    {"inkling.use_global_scale", K_BOOL, G_TEXT, 0, false, true, NULL},
    {"inkling.final_logit_softcapping", K_ABSENT, G_TEXT, 0, false, false, NULL},   /* null in the config: no softcap */
    {"inkling.moe_intermediate_size", K_EQ_EXP, G_TEXT, 0, false, false, NULL},     /* when named, == intermediate_size */
    /* the hmlp tower */
    {"inkling.vision.vision_encoder_type", K_STR, G_VISION, 0, false, false, "hmlp"},
    {"inkling.vision.decoder_dmodel", K_U32, G_VISION, F(n_embd), true, false, NULL},
    {"inkling.vision.patch_size", K_U32, G_VISION, F(vis_patch), true, false, NULL},
    {"inkling.vision.temporal_patch_size", K_U32, G_VISION, F(vis_temporal), true, false, NULL},
    {"inkling.vision.n_channels", K_U32, G_VISION, F(vis_channels), true, false, NULL},
    {"inkling.vision.n_layers", K_U32, G_VISION, F(vis_layers), true, false, NULL},
    {"inkling.vision.use_vision_norm", K_BOOL, G_VISION, 0, false, true, NULL},
    /* the chained MTP drafter */
    {"inkling.mtp.num_nextn_predict_layers", K_U32, G_MTP, F(n_mtp), false, false, NULL},
    {"inkling.mtp.chain_hidden_post_norm", K_BOOL, G_MTP, 0, false, false, NULL},
    {"inkling.mtp.local_layer_ids", K_LIST, G_MTP, 0, false, false, NULL},
};
#undef F

const pulsar_kv *ink_kv(const pulsar_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++)
        if (pulsar_streq(m->kv[i].key, key)) return &m->kv[i];
    return NULL;
}

bool refuse(const char *fmt, const char *key, const char *more = "") {
    fprintf(stderr, "pulsar: " INK);
    fprintf(stderr, fmt, key, more);
    fprintf(stderr, " -- refusing\n");
    pulsar_load_refuse();
    return false;
}

/* An array of ascending distinct layer ids, each below n, as a bitmask. */
bool read_ids(const pulsar_model *m, const char *key, uint32_t n, uint64_t *mask) {
    pulsar_array_ref arr;
    *mask = 0;
    if (!model_get_array(m, key, &arr) || (arr.type != PULSAR_META_UINT32 && arr.type != PULSAR_META_UINT64))
        return refuse("%s is missing or not an array of unsigned integers%s", key);
    pulsar_cursor c = cursor_at(m, arr.data_pos);
    int64_t last = -1;
    for (uint64_t i = 0; i < arr.len; i++) {
        uint64_t v = 0;
        uint32_t v32 = 0;
        const bool ok = arr.type == PULSAR_META_UINT64 ? cursor_u64(&c, &v) : (cursor_u32(&c, &v32) && ((v = v32), true));
        if (!ok || (int64_t)v <= last || v >= n || v >= 64u) {
            char why[96];
            snprintf(why, sizeof(why), " (entry %llu: ascending, distinct, below %u)", (unsigned long long)i, n);
            return refuse("%s is not a list of layer ids%s", key, why);
        }
        last = (int64_t)v;
        *mask |= 1ull << v;
    }
    return true;
}

}  // namespace

bool pulsar_inkling_read_shape(const pulsar_model *m, pulsar_inkling_shape *s) {
    *s = PULSAR_INKLING_SHAPE_SMALL;   /* the name and the profile's values; every read field is overwritten */
    s->n_mtp = 0;
    bool ok = true;
    if (!model_get_bool(m, "pulsar.mtp_present", &s->mtp_written) ||
        !model_get_bool(m, "pulsar.vision_present", &s->vision_written))
        ok = refuse("%s / pulsar.vision_present: the container does not say whether it holds the drafter and the "
                    "tower%s", "pulsar.mtp_present");
    const bool has_mtp = ink_kv(m, "inkling.mtp.num_nextn_predict_layers") != NULL;
    /* every inkling.* key is one of the table's */
    for (uint64_t i = 0; i < m->n_kv; i++) {
        const pulsar_str k = m->kv[i].key;
        if (k.len < 8 || memcmp(k.ptr, "inkling.", 8) != 0) continue;
        bool known = false;
        for (const ink_key &d : k_keys) known |= pulsar_streq(k, d.key);
        if (!known) {
            char key[160];
            snprintf(key, sizeof(key), "%.*s", (int)(k.len < 150 ? k.len : 150), k.ptr);
            ok = refuse("%s is a config key the Inkling declaration does not know (a switch no block here "
                        "implements)%s", key);
        }
    }
    for (const ink_key &d : k_keys) {
        const bool needed = d.group == G_TEXT || (d.group == G_VISION && s->vision_written) || (d.group == G_MTP && has_mtp);
        const pulsar_kv *kv = ink_kv(m, d.key);
        if (d.kind == K_ABSENT) {
            if (kv) ok = refuse("%s is set; the Inkling declaration implements it unset%s", d.key);
            continue;
        }
        if (d.kind == K_EQ_EXP || d.kind == K_LIST) continue;   /* below, after the numbers */
        if (!needed) {
            if (kv) ok = refuse("%s is present but its block is not written%s", d.key);
            continue;
        }
        if (!kv) {
            ok = refuse("required config key %s is missing%s", d.key);
            continue;
        }
        uint8_t *field = (uint8_t *)s + d.off;
        const uint8_t *want = (const uint8_t *)&PULSAR_INKLING_SHAPE_SMALL + d.off;
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
            snprintf(why, sizeof(why), " = %s; %s %s", got, d.pinned ? PULSAR_INKLING_SHAPE_SMALL.name : "the blocks implement",
                     exp);
            ok = refuse("%s%s", d.key, why);
        }
    }
    /* the plan */
    if (s->n_layer == 0 || s->n_layer > PULSAR_FAMILY_MAX_LAYER) {
        char why[64];
        snprintf(why, sizeof(why), " = %u: 1 .. %u layers", s->n_layer, PULSAR_FAMILY_MAX_LAYER);
        ok = refuse("%s%s", "inkling.num_hidden_layers", why);
        s->n_layer = 0;
    }
    if (s->n_dense > s->n_layer) ok = refuse("%s exceeds num_hidden_layers%s", "inkling.dense_mlp_idx");
    if (s->n_layer) ok &= read_ids(m, "inkling.local_layer_ids", s->n_layer, &s->window_layers);
    uint64_t ff = 0;
    if (ink_kv(m, "inkling.moe_intermediate_size") &&
        (!model_get_u64_compat(m, "inkling.moe_intermediate_size", &ff) || ff != s->n_ff_exp))
        ok = refuse("%s is named and is not intermediate_size (the experts are intermediate_size wide; L294)%s",
                    "inkling.moe_intermediate_size");
    if (has_mtp) {
        if (s->n_mtp == 0 || s->n_mtp != PULSAR_INKLING_SHAPE_SMALL.n_mtp_max) {
            char why[64];
            snprintf(why, sizeof(why), " = %u; %s has %u", s->n_mtp, s->name, PULSAR_INKLING_SHAPE_SMALL.n_mtp_max);
            ok = refuse("%s%s", "inkling.mtp.num_nextn_predict_layers", why);
            s->n_mtp = 0;
        } else {
            ok &= read_ids(m, "inkling.mtp.local_layer_ids", s->n_mtp, &s->mtp_window_layers);
        }
    }
    if (s->mtp_written && !has_mtp) ok = refuse("%s is true but the container has no inkling.mtp config%s", "pulsar.mtp_present");
    return ok;
}

void pulsar_inkling_plan(const pulsar_inkling_shape *s, pulsar_layer_plan *plan) {
    memset(plan, 0, sizeof(*plan));
    plan->n_layer = s->n_layer;
    for (uint32_t il = 0; il < s->n_layer; il++) {
        const bool window = s->window_layers >> il & 1u;
        plan->kind[il] = PULSAR_LAYER_BLOCKS;
        plan->desc[il].mixer = window ? PULSAR_MIXER_ATTN_WINDOW : PULSAR_MIXER_ATTN_FULL;
        plan->desc[il].ffn = il < s->n_dense ? PULSAR_FFN_DENSE : PULSAR_FFN_MOE;
        plan->desc[il].window = window ? s->window : 0u;
    }
}

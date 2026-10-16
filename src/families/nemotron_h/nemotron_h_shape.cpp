/* nemotron_h_shape.cpp -- the Nemotron-H family's shape: the container's nemotron_h.* keys -> pulsar_nemotron_h_shape,
 * checked against the known profile, and the layer plan (nemotron_h.h section 1).
 *
 * Rule 1 for a config: a key the declaration reads is required; a geometry value other than the profile's, a switch
 * other than the value the blocks implement, and a key this table does not know are each refused by name.  A key
 * with no meaning for the forward (an initialiser, a dropout, a generation default) is NAMED here with why, so a new
 * key is never silently taken as one of them.  The table below is the ONE list of what the family understands: the
 * reader, the pins and the unknown-key check all walk it. */
#include "nemotron_h.h"

#include <math.h>
#include <stddef.h>

const pulsar_nemotron_h_shape PULSAR_NEMOTRON_H_SHAPE_SUPER = {
    /* .name           = */ "Nemotron-3.5-Super",
    /* .n_embd         = */ 4096,
    /* .n_vocab        = */ 131072,
    /* .n_head         = */ 32,
    /* .n_head_kv      = */ 2,
    /* .head_dim       = */ 128,
    /* .ssm_n_head     = */ 128,
    /* .ssm_head_dim   = */ 64,
    /* .ssm_state      = */ 128,
    /* .ssm_groups     = */ 8,
    /* .conv_k         = */ 4,
    /* .expand         = */ 2,
    /* .n_expert       = */ 512,
    /* .n_expert_used  = */ 22,
    /* .n_ff_exp       = */ 2688,
    /* .latent         = */ 1024,
    /* .n_ff_shared    = */ 5376,
    /* .rms_eps        = */ 1e-5f,
    /* .route_scale    = */ 5.0f,
    /* .max_context    = */ 1048576,
    /* .eos_id         = */ 11,
};

#define NH "nemotron_h: "

namespace {

/** What a key is: a number read into the shape (pinned to the profile, or a plan fact), a switch pinned to the one
 *  value the blocks implement, the block-kind lists the plan reads, a key that must be absent, or a key named as
 *  meaningless to the forward (with why). */
enum nh_kind : uint8_t { K_U32, K_U64, K_F32, K_BOOL, K_STR, K_U32_EQ, K_F32_EQ, K_KINDS, K_MTP_KINDS, K_ABSENT, K_IGNORED };

struct nh_key {
    const char *key;      ///< the container key
    nh_kind kind;
    size_t off;           ///< the shape field (K_U32 / K_U64 / K_F32)
    bool pinned;          ///< the field must equal the profile's
    bool want_bool;       ///< K_BOOL's one value
    const char *want;     ///< K_STR's one value; K_IGNORED's reason
    double want_num;      ///< K_U32_EQ / K_F32_EQ's one value
};

#define F(field) offsetof(pulsar_nemotron_h_shape, field)
#define PIN(k, field) {"nemotron_h." k, K_U32, F(field), true, false, NULL, 0}
#define SW_B(k, v) {"nemotron_h." k, K_BOOL, 0, false, v, NULL, 0}
#define SW_S(k, v) {"nemotron_h." k, K_STR, 0, false, false, v, 0}
#define SW_U(k, v) {"nemotron_h." k, K_U32_EQ, 0, false, false, NULL, v}
#define IGN(k, why) {"nemotron_h." k, K_IGNORED, 0, false, false, why, 0}
const nh_key k_keys[] = {
    /* geometry, pinned */
    PIN("hidden_size", n_embd),
    PIN("vocab_size", n_vocab),
    PIN("num_attention_heads", n_head),
    PIN("num_key_value_heads", n_head_kv),
    PIN("head_dim", head_dim),
    PIN("mamba_num_heads", ssm_n_head),
    PIN("mamba_head_dim", ssm_head_dim),
    PIN("ssm_state_size", ssm_state),
    PIN("n_groups", ssm_groups),
    PIN("conv_kernel", conv_k),
    PIN("expand", expand),
    PIN("n_routed_experts", n_expert),
    PIN("num_experts_per_tok", n_expert_used),
    PIN("moe_intermediate_size", n_ff_exp),
    PIN("intermediate_size", n_ff_exp),      /* the experts' width under its other name: both pinned to it */
    PIN("moe_latent_size", latent),
    PIN("moe_shared_expert_intermediate_size", n_ff_shared),
    PIN("eos_token_id", eos_id),
    {"nemotron_h.layer_norm_epsilon", K_F32, F(rms_eps), true, false, NULL, 0},
    {"nemotron_h.routed_scaling_factor", K_F32, F(route_scale), true, false, NULL, 0},
    {"nemotron_h.max_position_embeddings", K_U64, F(max_context), true, false, NULL, 0},
    /* the plan, the container's */
    {"nemotron_h.layers_block_type", K_KINDS, 0, false, false, NULL, 0},
    {"nemotron_h.num_nextn_predict_layers", K_U32, F(n_mtp), false, false, NULL, 0},
    {"nemotron_h.mtp_layers_block_type", K_MTP_KINDS, 0, false, false, NULL, 0},
    /* the switches: the one value each block implements */
    SW_S("model_type", "nemotron_h"),
    SW_S("dtype", "bfloat16"),
    SW_S("mlp_hidden_act", "relu2"),          /* routed and shared experts: relu(x)^2, no gate */
    SW_S("mamba_hidden_act", "silu"),         /* the conv's activation (and the gate's silu(z)) */
    SW_S("mamba_ssm_cache_dtype", "float32"), /* the SSM state is f32 (plan 121: FIXED / DELTA) */
    SW_B("attention_bias", false),
    SW_B("mlp_bias", false),
    SW_B("use_bias", false),
    SW_B("mamba_proj_bias", false),
    SW_B("use_conv_bias", true),
    SW_B("norm_topk_prob", true),
    SW_B("tie_word_embeddings", false),
    SW_B("residual_in_fp32", false),
    SW_B("moe_shared_expert_overlap", false),
    SW_U("n_shared_experts", 1),
    SW_U("n_group", 1),                       /* no group-limited selection */
    SW_U("topk_group", 1),
    {"nemotron_h.time_step_limit.min", K_F32_EQ, 0, false, false, NULL, 0.0},   /* no dt clamp: [0, inf] (vLLM) */
    SW_B("time_step_limit.max_inf", true),
    {"nemotron_h.sliding_window", K_ABSENT, 0, false, false, NULL, 0},         /* null in the config: full attention */
    {"nemotron_h.time_step_limit", K_ABSENT, 0, false, false, NULL, 0},        /* the builder writes it as the two above */
    /* named, with no meaning for the forward */
    IGN("architectures", "the language model's class name; model_type is the pin"),
    IGN("attention_dropout", "training only"),
    IGN("hidden_dropout", "training only"),
    IGN("initializer_range", "initialisation only"),
    IGN("rescale_prenorm_residual", "initialisation only"),
    IGN("time_step_min", "dt_bias initialisation (transformers 5.19 also clamps dt at it; vLLM, the graded reference, "
                         "uses time_step_limit)"),
    IGN("time_step_max", "dt_bias initialisation only"),
    IGN("time_step_floor", "dt_bias initialisation only"),
    IGN("chunk_size", "vLLM's SSD prefill chunk: a schedule, not the model (pulsar's scan is its own)"),
    IGN("use_cache", "a transformers generation option"),
    IGN("use_mamba_kernels", "a transformers kernel choice"),
    IGN("num_logits_to_keep", "a transformers generation option"),
    IGN("bos_token_id", "the template renders no BOS (the tokenizer's file holds the id)"),
    IGN("pad_token_id", "padding: no batch here pads"),
};
#undef F
#undef PIN
#undef SW_B
#undef SW_S
#undef SW_U
#undef IGN

const pulsar_kv *nh_kv(const pulsar_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++)
        if (pulsar_streq(m->kv[i].key, key)) return &m->kv[i];
    return NULL;
}

bool refuse(const char *fmt, const char *key, const char *more = "") {
    fprintf(stderr, "pulsar: " NH);
    fprintf(stderr, fmt, key, more);
    fprintf(stderr, " -- refusing\n");
    pulsar_load_refuse();
    return false;
}

/* A string array of block kinds: each one of `allowed` (ascending index = HF block). */
bool read_kinds(const pulsar_model *m, const char *key, char kinds[][12], uint32_t max, uint32_t *n) {
    pulsar_array_ref arr;
    *n = 0;
    if (!model_get_array(m, key, &arr) || arr.type != PULSAR_META_STRING)
        return refuse("%s is missing or not an array of strings%s", key);
    if (arr.len == 0 || arr.len > max) {
        char why[64];
        snprintf(why, sizeof(why), " (%llu entries; 1 .. %u)", (unsigned long long)arr.len, max);
        return refuse("%s has a length outside the declaration%s", key, why);
    }
    pulsar_cursor c = cursor_at(m, arr.data_pos);
    for (uint64_t i = 0; i < arr.len; i++) {
        pulsar_str s = {NULL, 0};
        if (!cursor_string(&c, &s)) return refuse("%s: an entry is not a string%s", key);
        if (!pulsar_streq(s, "mamba") && !pulsar_streq(s, "attention") && !pulsar_streq(s, "moe")) {
            char why[96];
            snprintf(why, sizeof(why), "[%llu] = '%.*s' (mamba / attention / moe)", (unsigned long long)i,
                     (int)(s.len < 40 ? s.len : 40), s.ptr);
            return refuse("%s%s: not a block kind the declaration implements", key, why);
        }
        snprintf(kinds[i], sizeof(kinds[i]), "%.*s", (int)s.len, s.ptr);
    }
    *n = (uint32_t)arr.len;
    return true;
}

/* HF blocks -> pulsar layers: a mixer opens a layer, a moe closes the open one.  The same grouping as the builder's
 * (tools/container/nemotron_h.py group_layers); the bind records carry its result, and the binder proves the two
 * agree tensor by tensor. */
bool group(const char kinds[][12], uint32_t n, const char *key, uint32_t *n_layer, uint64_t *mamba, uint64_t *moe) {
    *n_layer = 0;
    *mamba = *moe = 0;
    bool open = false;
    for (uint32_t b = 0; b < n; b++) {
        if (!strcmp(kinds[b], "moe")) {
            if (!open) {
                char why[64];
                snprintf(why, sizeof(why), "[%u] = 'moe' with no mixer before it", b);
                return refuse("%s%s (a layer is a mixer and the MoE after it)", key, why);
            }
            *moe |= 1ull << (*n_layer - 1);
            open = false;
            continue;
        }
        if (*n_layer == PULSAR_FAMILY_MAX_LAYER) {
            char why[64];
            snprintf(why, sizeof(why), " groups into more than %u layers", PULSAR_FAMILY_MAX_LAYER);
            return refuse("%s%s", key, why);
        }
        if (!strcmp(kinds[b], "mamba")) *mamba |= 1ull << *n_layer;
        (*n_layer)++;
        open = true;
    }
    return true;
}

}  // namespace

bool pulsar_nemotron_h_read_shape(const pulsar_model *m, pulsar_nemotron_h_shape *s) {
    *s = PULSAR_NEMOTRON_H_SHAPE_SUPER;   /* the name and the profile's values; every read field is overwritten */
    bool ok = true;
    bool vision = false;
    if (!model_get_bool(m, "pulsar.mtp_present", &s->mtp_written) || !model_get_bool(m, "pulsar.vision_present", &vision))
        ok = refuse("%s / pulsar.vision_present: the container does not say whether it holds the drafter and the "
                    "tower%s", "pulsar.mtp_present");
    if (vision) ok = refuse("%s is true: the vision tower is not part of this declaration (v1: text; N11)%s",
                            "pulsar.vision_present");
    /* every nemotron_h.* key is one of the table's */
    for (uint64_t i = 0; i < m->n_kv; i++) {
        const pulsar_str k = m->kv[i].key;
        if (k.len < 11 || memcmp(k.ptr, "nemotron_h.", 11) != 0) continue;
        bool known = false;
        for (const nh_key &d : k_keys) known |= pulsar_streq(k, d.key);
        if (!known) {
            char key[160];
            snprintf(key, sizeof(key), "%.*s", (int)(k.len < 150 ? k.len : 150), k.ptr);
            ok = refuse("%s is a config key the Nemotron-H declaration does not know (a switch no block here "
                        "implements)%s", key);
        }
    }
    for (const nh_key &d : k_keys) {
        const pulsar_kv *kv = nh_kv(m, d.key);
        if (d.kind == K_ABSENT) {
            if (kv) ok = refuse("%s is set; the Nemotron-H declaration implements it unset%s", d.key);
            continue;
        }
        if (d.kind == K_IGNORED || d.kind == K_KINDS || d.kind == K_MTP_KINDS) continue;   /* the lists: below */
        if (!kv) {
            ok = refuse("required config key %s is missing%s", d.key);
            continue;
        }
        uint8_t *field = (uint8_t *)s + d.off;
        const uint8_t *want = (const uint8_t *)&PULSAR_NEMOTRON_H_SHAPE_SUPER + d.off;
        char got[96] = "", exp[96] = "";
        bool match = true, read = true;
        switch (d.kind) {
        case K_U32:
        case K_U64:
        case K_U32_EQ: {
            uint64_t v = 0;
            read = model_get_u64_compat(m, d.key, &v) && (d.kind == K_U64 || v <= UINT32_MAX);
            if (!read) break;
            uint64_t w = (uint64_t)d.want_num;
            if (d.kind == K_U64) {
                *(uint64_t *)field = v;
                w = *(const uint64_t *)want;
            } else if (d.kind == K_U32) {
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
            match = w == 0.0f ? v == 0.0f : fabsf(v - w) <= fabsf(w) * 1.0e-6f;
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
        default: break;
        }
        if (!read) {
            ok = refuse("config key %s has the wrong type%s", d.key);
        } else if (!match) {
            char why[224];
            snprintf(why, sizeof(why), " = %s; %s %s", got,
                     d.pinned ? PULSAR_NEMOTRON_H_SHAPE_SUPER.name : "the blocks implement", exp);
            ok = refuse("%s%s", d.key, why);
        }
    }
    /* the plan: the block kinds, grouped */
    char kinds[PULSAR_NEMOTRON_H_MAX_BLOCKS][12];
    uint32_t n = 0;
    if (read_kinds(m, "nemotron_h.layers_block_type", kinds, PULSAR_NEMOTRON_H_MAX_BLOCKS, &n) &&
        group(kinds, n, "nemotron_h.layers_block_type", &s->n_layer, &s->mamba_layers, &s->moe_layers)) {
        s->n_blocks = n;
    } else {
        ok = false;
        s->n_layer = 0;
    }
    const bool has_mtp_kinds = nh_kv(m, "nemotron_h.mtp_layers_block_type") != NULL;
    if (s->n_mtp > 1) {
        char why[64];
        snprintf(why, sizeof(why), " = %u: one module (applied recursively) or none", s->n_mtp);
        ok = refuse("%s%s", "nemotron_h.num_nextn_predict_layers", why);
    } else if (s->n_mtp == 1) {
        /* the drafter is one {attention, MoE} layer: the binder's declaration of it */
        uint32_t nm = 0, nl = 0;
        uint64_t mm = 0, me = 0;
        if (!has_mtp_kinds) {
            ok = refuse("%s is missing (num_nextn_predict_layers is 1)%s", "nemotron_h.mtp_layers_block_type");
        } else if (!read_kinds(m, "nemotron_h.mtp_layers_block_type", kinds, PULSAR_NEMOTRON_H_MAX_BLOCKS, &nm) ||
                   !group(kinds, nm, "nemotron_h.mtp_layers_block_type", &nl, &mm, &me)) {
            ok = false;
        } else if (nl != 1 || mm != 0 || me != 1) {
            ok = refuse("%s is not one {attention, moe} layer (the declaration's drafter)%s",
                        "nemotron_h.mtp_layers_block_type");
        }
    }
    if (s->mtp_written && s->n_mtp == 0)
        ok = refuse("%s is true but num_nextn_predict_layers is 0%s", "pulsar.mtp_present");
    return ok;
}

void pulsar_nemotron_h_plan(const pulsar_nemotron_h_shape *s, pulsar_layer_plan *plan) {
    memset(plan, 0, sizeof(*plan));
    plan->n_layer = s->n_layer;
    for (uint32_t il = 0; il < s->n_layer; il++) {
        plan->kind[il] = PULSAR_LAYER_BLOCKS;
        plan->desc[il].mixer = s->mamba_layers >> il & 1u ? PULSAR_MIXER_MAMBA2 : PULSAR_MIXER_ATTN_FULL;
        plan->desc[il].ffn = s->moe_layers >> il & 1u ? PULSAR_FFN_MOE : PULSAR_FFN_NONE;
    }
}

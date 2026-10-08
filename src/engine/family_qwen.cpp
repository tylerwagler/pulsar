/* family_qwen.cpp -- the Qwen4-exp family (Qwen3.8-Flash-Next), L251 S1.
 *
 * What is here: config validation against the known shape, the layer plan from
 * the artifact's layer_types, the weight binding by HF name, the session state
 * (DeltaNet recurrent + conv state, FP8 KV and pooled index keys, PLE conv
 * state; sized for the 1.5-2M-token target), and the step driver that walks
 * the plan through the op table (family_qwen.h).  What is NOT here: any op.
 * Until S2-S4 fill g_qwen_ops, a step refuses by naming the first op it would
 * call and the stream that owns it; nothing computes a number (rules 1, 9). */
#include "pulsar_engine_internal.h"
#include "exl3_trellis.h"
#include "lib/qwen_tokenizer.h"
#include "tp/pulsar_tp.h"
#include "spec_internal.h"
#include "qwen_forward.h"

const pulsar_qwen_shape PULSAR_QWEN_SHAPE_FLASH_NEXT = {
    /* .name              = */ "Qwen3.8-Flash-Next",
    /* .n_layer           = */ 48,
    /* .n_embd            = */ 2560,
    /* .n_vocab           = */ 248320,
    /* .n_hc              = */ 4,
    /* .n_hc_lowrank      = */ 320,
    /* .gdn_n_k_head      = */ 16,
    /* .gdn_n_v_head      = */ 48,
    /* .gdn_k_dim         = */ 128,
    /* .gdn_v_dim         = */ 128,
    /* .gdn_conv_kernel   = */ 4,
    /* .n_head            = */ 24,
    /* .n_head_kv         = */ 2,
    /* .head_dim          = */ 256,
    /* .n_rot             = */ 64,
    /* .mrope_section     = */ {11, 11, 10},
    /* .rope_theta        = */ 1.0e7f,
    /* .idx_n_head        = */ 4,
    /* .idx_n_head_kv     = */ 1,
    /* .idx_head_dim      = */ 128,
    /* .idx_block         = */ 4,
    /* .idx_budget        = */ 2048,
    /* .n_expert          = */ 512,
    /* .n_expert_used     = */ 10,
    /* .n_ff_exp          = */ 640,
    /* .n_ff_shexp        = */ 640,
    /* .ple_layer         = */ 1,
    /* .ngram_size        = */ 3,
    /* .ngram_heads       = */ 8,
    /* .ple_embed_dim     = */ 2560,
    /* .ple_conv_kernel   = */ 4,
    /* .ngram_vocab_base  = */ 20000000ull,
    /* .ngram_split_parts = */ 128,
    /* .rms_eps           = */ 1.0e-6f,
    /* .max_position      = */ 262144,
    /* .eos_id            = */ 248044,
    /* .n_mtp_layer       = */ 1,
};

pulsar_qwen_shape g_qwen_shape = PULSAR_QWEN_SHAPE_FLASH_NEXT;

/* ---- ops ---------------------------------------------------------------- */

/* THE op table.  Every entry is NULL until its stream lands it; a stream fills
 * its own entries here and nothing else. */
const pulsar_qwen_ops g_qwen_ops = {
    /* .embed         = */ pulsar_qwen_s4_embed,      /* S4 work/l251-moe */
    /* .ple           = */ pulsar_qwen_s4_ple,        /* S4 work/l251-moe */
    /* .gr_read       = */ pulsar_qwen_s4_gr_read,    /* S4 work/l251-moe */
    /* .gdn           = */ pulsar_qwen_s2_gdn,
    /* .qsa           = */ pulsar_qwen_s3_qsa,
    /* .gr_write      = */ pulsar_qwen_s4_gr_write,   /* S4 work/l251-moe */
    /* .moe           = */ pulsar_qwen_s4_moe,        /* S4 work/l251-moe */
    /* .head          = */ pulsar_qwen_s4_head,       /* S4 work/l251-moe */
    /* .scratch_bytes = */ pulsar_qwen_s4_scratch_bytes,   /* S4's ops; S2 / S3 chain theirs in */
};

const char *pulsar_qwen_op_name(pulsar_qwen_op_id op) {
    switch (op) {
    case PULSAR_QWEN_OP_EMBED:    return "embed";
    case PULSAR_QWEN_OP_PLE:      return "ple";
    case PULSAR_QWEN_OP_GR_READ:  return "gr_read";
    case PULSAR_QWEN_OP_GDN:      return "gdn";
    case PULSAR_QWEN_OP_QSA:      return "qsa";
    case PULSAR_QWEN_OP_GR_WRITE: return "gr_write";
    case PULSAR_QWEN_OP_MOE:      return "moe";
    case PULSAR_QWEN_OP_HEAD:     return "head";
    case PULSAR_QWEN_OP_COUNT:    break;
    }
    return "unknown";
}

const char *pulsar_qwen_op_owner(pulsar_qwen_op_id op) {
    switch (op) {
    case PULSAR_QWEN_OP_GDN: return "S2 (work/l251-gdn)";
    case PULSAR_QWEN_OP_QSA: return "S3 (work/l251-attn)";
    case PULSAR_QWEN_OP_EMBED:
    case PULSAR_QWEN_OP_PLE:
    case PULSAR_QWEN_OP_GR_READ:
    case PULSAR_QWEN_OP_GR_WRITE:
    case PULSAR_QWEN_OP_MOE:
    case PULSAR_QWEN_OP_HEAD:     return "S4 (work/l251-moe)";
    case PULSAR_QWEN_OP_COUNT:    break;
    }
    return "unknown";
}

static bool qwen_op_present(const pulsar_qwen_ops *ops, pulsar_qwen_op_id op) {
    switch (op) {
    case PULSAR_QWEN_OP_EMBED:    return ops->embed != NULL;
    case PULSAR_QWEN_OP_PLE:      return ops->ple != NULL;
    case PULSAR_QWEN_OP_GR_READ:  return ops->gr_read != NULL;
    case PULSAR_QWEN_OP_GDN:      return ops->gdn != NULL;
    case PULSAR_QWEN_OP_QSA:      return ops->qsa != NULL;
    case PULSAR_QWEN_OP_GR_WRITE: return ops->gr_write != NULL;
    case PULSAR_QWEN_OP_MOE:      return ops->moe != NULL;
    case PULSAR_QWEN_OP_HEAD:     return ops->head != NULL;
    case PULSAR_QWEN_OP_COUNT:    break;
    }
    return false;
}

/* The ops a forward over `plan` calls, in the order it calls them, and the
 * first of them that is missing (PULSAR_QWEN_OP_COUNT: none).  The walk is the
 * driver's own order (qwen_forward), so the refusal names exactly the op the
 * step would have reached first. */
pulsar_qwen_op_id pulsar_qwen_first_missing_op(const pulsar_qwen_ops *ops, const pulsar_layer_plan *plan,
                                               const pulsar_qwen_shape *shape, uint32_t *at_layer) {
    *at_layer = UINT32_MAX;
    if (!qwen_op_present(ops, PULSAR_QWEN_OP_EMBED)) return PULSAR_QWEN_OP_EMBED;
    for (uint32_t il = 0; il < plan->n_layer; il++) {
        const pulsar_qwen_op_id seq[] = {
            il == shape->ple_layer ? PULSAR_QWEN_OP_PLE : PULSAR_QWEN_OP_COUNT,
            PULSAR_QWEN_OP_GR_READ,
            plan->kind[il] == PULSAR_LAYER_QWEN_GDN ? PULSAR_QWEN_OP_GDN : PULSAR_QWEN_OP_QSA,
            PULSAR_QWEN_OP_GR_WRITE,
            PULSAR_QWEN_OP_GR_READ,
            PULSAR_QWEN_OP_MOE,
            PULSAR_QWEN_OP_GR_WRITE,
        };
        for (size_t k = 0; k < sizeof(seq) / sizeof(seq[0]); k++) {
            if (seq[k] == PULSAR_QWEN_OP_COUNT) continue;
            if (!qwen_op_present(ops, seq[k])) {
                *at_layer = il;
                return seq[k];
            }
        }
    }
    if (!qwen_op_present(ops, PULSAR_QWEN_OP_HEAD)) return PULSAR_QWEN_OP_HEAD;
    return PULSAR_QWEN_OP_COUNT;
}

/* ---- load: config ---------------------------------------------------------- */

#define QWEN_KEY(k) PULSAR_QWEN_ARCH "." k

static bool qcfg_u64(const pulsar_model *m, const char *key, uint64_t *out) {
    if (model_get_u64_compat(m, key, out)) return true;
    fprintf(stderr, "pulsar: %s: required config key %s is missing or not an unsigned integer\n",
            PULSAR_QWEN_ARCH, key);
    return false;
}

static bool qcfg_u32(const pulsar_model *m, const char *key, uint32_t *out) {
    uint64_t v = 0;
    if (!qcfg_u64(m, key, &v)) return false;
    if (v > UINT32_MAX) {
        fprintf(stderr, "pulsar: %s: %s = %llu does not fit 32 bits\n", PULSAR_QWEN_ARCH, key,
                (unsigned long long)v);
        return false;
    }
    *out = (uint32_t)v;
    return true;
}

static bool qcfg_f32(const pulsar_model *m, const char *key, float *out) {
    if (model_get_f32_compat(m, key, out)) return true;
    fprintf(stderr, "pulsar: %s: required config key %s is missing or not a number\n",
            PULSAR_QWEN_ARCH, key);
    return false;
}

/* An unsigned-integer array key of exactly n entries (u32 or u64 elements). */
static bool qcfg_u64_array(const pulsar_model *m, const char *key, uint64_t *out, uint32_t n) {
    pulsar_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != PULSAR_META_UINT32 && arr.type != PULSAR_META_UINT64) || arr.len != n) {
        fprintf(stderr, "pulsar: %s: required config key %s is missing or not an array of %u "
                        "unsigned integers\n", PULSAR_QWEN_ARCH, key, n);
        return false;
    }
    pulsar_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t i = 0; i < n; i++) {
        if (arr.type == PULSAR_META_UINT64) {
            if (!cursor_u64(&c, &out[i])) return false;
        } else {
            uint32_t v = 0;
            if (!cursor_u32(&c, &v)) return false;
            out[i] = v;
        }
    }
    return true;
}

/* Every field of the loaded shape against the known one, by name. */
static bool qwen_shape_check(const pulsar_qwen_shape *got, const pulsar_qwen_shape *want) {
    bool ok = true;
#define QCHK_U(field) do { if (got->field != want->field) { \
        fprintf(stderr, "pulsar: %s: %s = %llu, %s has %llu\n", PULSAR_QWEN_ARCH, #field, \
                (unsigned long long)got->field, want->name, (unsigned long long)want->field); ok = false; } } while (0)
#define QCHK_F(field) do { if (fabsf(got->field - want->field) > fabsf(want->field) * 1.0e-6f) { \
        fprintf(stderr, "pulsar: %s: %s = %.9g, %s has %.9g\n", PULSAR_QWEN_ARCH, #field, \
                (double)got->field, want->name, (double)want->field); ok = false; } } while (0)
    QCHK_U(n_layer); QCHK_U(n_embd); QCHK_U(n_vocab); QCHK_U(n_hc); QCHK_U(n_hc_lowrank);
    QCHK_U(gdn_n_k_head); QCHK_U(gdn_n_v_head); QCHK_U(gdn_k_dim); QCHK_U(gdn_v_dim);
    QCHK_U(gdn_conv_kernel);
    QCHK_U(n_head); QCHK_U(n_head_kv); QCHK_U(head_dim); QCHK_U(n_rot);
    QCHK_U(mrope_section[0]); QCHK_U(mrope_section[1]); QCHK_U(mrope_section[2]);
    QCHK_F(rope_theta);
    QCHK_U(idx_n_head); QCHK_U(idx_n_head_kv); QCHK_U(idx_head_dim); QCHK_U(idx_block);
    QCHK_U(idx_budget);
    QCHK_U(n_expert); QCHK_U(n_expert_used); QCHK_U(n_ff_exp); QCHK_U(n_ff_shexp);
    QCHK_U(ple_layer); QCHK_U(ngram_size); QCHK_U(ngram_heads); QCHK_U(ple_embed_dim);
    QCHK_U(ple_conv_kernel); QCHK_U(ngram_vocab_base); QCHK_U(ngram_split_parts);
    QCHK_F(rms_eps);
    QCHK_U(max_position); QCHK_U(eos_id); QCHK_U(n_mtp_layer);
#undef QCHK_U
#undef QCHK_F
    return ok;
}

static bool qwen_read_shape(const pulsar_model *m, pulsar_qwen_shape *s) {
    *s = PULSAR_QWEN_SHAPE_FLASH_NEXT;   /* the name; every number is overwritten below */
    bool ok = true;
    ok &= qcfg_u32(m, QWEN_KEY("num_hidden_layers"), &s->n_layer);
    ok &= qcfg_u32(m, QWEN_KEY("hidden_size"), &s->n_embd);
    ok &= qcfg_u32(m, QWEN_KEY("vocab_size"), &s->n_vocab);
    ok &= qcfg_u32(m, QWEN_KEY("hc_count"), &s->n_hc);
    ok &= qcfg_u32(m, QWEN_KEY("hc_lowrank"), &s->n_hc_lowrank);
    ok &= qcfg_u32(m, QWEN_KEY("linear_num_key_heads"), &s->gdn_n_k_head);
    ok &= qcfg_u32(m, QWEN_KEY("linear_num_value_heads"), &s->gdn_n_v_head);
    ok &= qcfg_u32(m, QWEN_KEY("linear_key_head_dim"), &s->gdn_k_dim);
    ok &= qcfg_u32(m, QWEN_KEY("linear_value_head_dim"), &s->gdn_v_dim);
    ok &= qcfg_u32(m, QWEN_KEY("linear_conv_kernel_dim"), &s->gdn_conv_kernel);
    ok &= qcfg_u32(m, QWEN_KEY("num_attention_heads"), &s->n_head);
    ok &= qcfg_u32(m, QWEN_KEY("num_key_value_heads"), &s->n_head_kv);
    ok &= qcfg_u32(m, QWEN_KEY("head_dim"), &s->head_dim);
    float partial = 0.0f;
    ok &= qcfg_f32(m, QWEN_KEY("partial_rotary_factor"), &partial);
    s->n_rot = (uint32_t)lroundf((float)s->head_dim * partial);
    uint64_t sec[3] = {0, 0, 0};
    ok &= qcfg_u64_array(m, QWEN_KEY("rope_parameters.mrope_section"), sec, 3);
    for (int i = 0; i < 3; i++) s->mrope_section[i] = (uint32_t)sec[i];
    ok &= qcfg_f32(m, QWEN_KEY("rope_parameters.rope_theta"), &s->rope_theta);
    ok &= qcfg_u32(m, QWEN_KEY("indexer_n_heads"), &s->idx_n_head);
    ok &= qcfg_u32(m, QWEN_KEY("indexer_kv_heads"), &s->idx_n_head_kv);
    ok &= qcfg_u32(m, QWEN_KEY("indexer_head_dim"), &s->idx_head_dim);
    ok &= qcfg_u32(m, QWEN_KEY("indexer_compress_ratio"), &s->idx_block);
    ok &= qcfg_u32(m, QWEN_KEY("indexer_budget"), &s->idx_budget);
    ok &= qcfg_u32(m, QWEN_KEY("num_experts"), &s->n_expert);
    ok &= qcfg_u32(m, QWEN_KEY("num_experts_per_tok"), &s->n_expert_used);
    ok &= qcfg_u32(m, QWEN_KEY("moe_intermediate_size"), &s->n_ff_exp);
    ok &= qcfg_u32(m, QWEN_KEY("shared_expert_intermediate_size"), &s->n_ff_shexp);
    /* ple_layer_ids is 1-based in HF (`layer_idx + 1 in ple_layer_ids`), and
     * this engine carries exactly one PLE layer. */
    uint64_t ple_id = 0;
    if (qcfg_u64_array(m, QWEN_KEY("ple_layer_ids"), &ple_id, 1)) {
        if (ple_id == 0) {
            fprintf(stderr, "pulsar: %s: ple_layer_ids holds 0, but the ids are 1-based\n", PULSAR_QWEN_ARCH);
            ok = false;
        }
        s->ple_layer = (uint32_t)(ple_id - 1u);
    } else {
        ok = false;
    }
    ok &= qcfg_u32(m, QWEN_KEY("ngram_size"), &s->ngram_size);
    ok &= qcfg_u32(m, QWEN_KEY("heads_per_ngram"), &s->ngram_heads);
    ok &= qcfg_u32(m, QWEN_KEY("ple_embed_dim"), &s->ple_embed_dim);
    ok &= qcfg_u32(m, QWEN_KEY("ple_conv_kernel_size"), &s->ple_conv_kernel);
    ok &= qcfg_u64(m, QWEN_KEY("ngram_vocab_size_base"), &s->ngram_vocab_base);
    ok &= qcfg_u32(m, QWEN_KEY("split_ngram_parts"), &s->ngram_split_parts);
    ok &= qcfg_f32(m, QWEN_KEY("rms_norm_eps"), &s->rms_eps);
    ok &= qcfg_u32(m, QWEN_KEY("max_position_embeddings"), &s->max_position);
    ok &= qcfg_u32(m, QWEN_KEY("eos_token_id"), &s->eos_id);
    ok &= qcfg_u32(m, QWEN_KEY("mtp_num_hidden_layers"), &s->n_mtp_layer);
    return ok;
}

/* The plan from `layer_types`, HF's own strings: "linear_attention" is a GDN
 * layer, "full_attention" a QSA layer, anything else is refused. */
static bool qwen_read_plan(const pulsar_model *m, uint32_t n_layer, pulsar_layer_plan *plan) {
    const char *key = QWEN_KEY("layer_types");
    pulsar_array_ref arr;
    if (!model_get_array(m, key, &arr) || arr.type != PULSAR_META_STRING || arr.len != n_layer) {
        fprintf(stderr, "pulsar: %s: %s must be an array of %u strings\n", PULSAR_QWEN_ARCH, key, n_layer);
        return false;
    }
    if (n_layer > PULSAR_FAMILY_MAX_LAYER) {
        fprintf(stderr, "pulsar: %s: %u layers exceed the plan's %u\n", PULSAR_QWEN_ARCH, n_layer,
                PULSAR_FAMILY_MAX_LAYER);
        return false;
    }
    memset(plan, 0, sizeof(*plan));
    plan->n_layer = n_layer;
    pulsar_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t il = 0; il < n_layer; il++) {
        pulsar_str t = {NULL, 0};
        if (!cursor_string(&c, &t)) {
            fprintf(stderr, "pulsar: %s: %s[%u] is unreadable\n", PULSAR_QWEN_ARCH, key, il);
            return false;
        }
        if (pulsar_streq(t, "linear_attention")) {
            plan->kind[il] = PULSAR_LAYER_QWEN_GDN;
        } else if (pulsar_streq(t, "full_attention")) {
            plan->kind[il] = PULSAR_LAYER_QWEN_QSA;
        } else {
            fprintf(stderr, "pulsar: %s: %s[%u] = '%.*s' is neither linear_attention nor full_attention\n",
                    PULSAR_QWEN_ARCH, key, il, (int)t.len, t.ptr);
            return false;
        }
    }
    return true;
}

/* ---- load: weights ---------------------------------------------------------- */

#define QWEN_TEXT "model.language_model."

/* Bind one tensor by its HF name, checking its logical dims in ne order
 * (the HF shape reversed).  nd dims are checked; missing -> refused. */
static pulsar_tensor *qbind(const pulsar_model *m, bool *ok, const char *name,
                            uint32_t nd, uint64_t d0, uint64_t d1 = 0, uint64_t d2 = 0) {
    pulsar_tensor *t = pulsar_tensor_bind(m, PULSAR_QWEN_ARCH, name);   /* L272 P4a: the core's mechanics */
    if (!t || !pulsar_tensor_dims(t, PULSAR_QWEN_ARCH, nd, d0, d1, d2)) *ok = false;
    return t;
}

static void qwen_bind_gr(const pulsar_model *m, bool *ok, const pulsar_qwen_shape *s,
                         const char *prefix, pulsar_qwen_gr_weights *gr, bool with_inject) {
    const uint64_t hc = pulsar_qwen_hc_dim(s);
    char name[256];
    snprintf(name, sizeof(name), "%shc_norm.weight", prefix);
    gr->hc_norm = qbind(m, ok, name, 1, hc);
    snprintf(name, sizeof(name), "%sinput_mix_weight_down.weight", prefix);
    gr->mix_down = qbind(m, ok, name, 2, hc, s->n_hc_lowrank);
    snprintf(name, sizeof(name), "%sinput_mix_weight_up.weight", prefix);
    gr->mix_up = qbind(m, ok, name, 2, s->n_hc_lowrank, hc);
    if (with_inject) {
        snprintf(name, sizeof(name), "%sblock_inject_weight.weight", prefix);
        gr->inject = qbind(m, ok, name, 2, hc, s->n_hc);
    }
}

/* The PLE side tables that HF keeps as int64 buffers: the hash multipliers,
 * and each n-gram head's table size and row offset.  They are config, not
 * weights -- the container carries them as u64 arrays (family_qwen.h s2). */
static bool qwen_read_ple_tables(const pulsar_model *m, const pulsar_qwen_shape *s, pulsar_qwen_weights *w) {
    const uint32_t n_heads = (s->ngram_size - 1u) * s->ngram_heads;
    if (s->ngram_size > PULSAR_QWEN_MAX_NGRAM || n_heads > PULSAR_QWEN_MAX_NGRAM_HEADS) {
        fprintf(stderr, "pulsar: %s: n-gram size %u / %u heads exceed the engine's %u / %u\n",
                PULSAR_QWEN_ARCH, s->ngram_size, n_heads, PULSAR_QWEN_MAX_NGRAM, PULSAR_QWEN_MAX_NGRAM_HEADS);
        return false;
    }
    bool ok = qcfg_u64_array(m, QWEN_KEY("ple_layer_multipliers"), w->ple_multipliers, s->ngram_size);
    ok &= qcfg_u64_array(m, QWEN_KEY("ple_ngram_heads_vocab_sizes"), w->ple_head_vocab, n_heads);
    ok &= qcfg_u64_array(m, QWEN_KEY("ple_ngram_heads_offsets"), w->ple_head_offset, n_heads);
    return ok;
}

/* One layer's tensors under `lp` (the layer's name prefix, ending in '.'): the trunk's
 * "model.language_model.layers.<il>." or the MTP layer's "mtp.layers.0.".  `split` binds the
 * routed experts as separate gate / up stacks (the MTP layer) instead of the fused gate_up. */
static void qwen_bind_layer(const pulsar_model *m, bool *ok, const pulsar_qwen_shape *s, pulsar_layer_kind kind,
                            const char *lp, bool ple, bool split, pulsar_qwen_layer_weights *L) {
    const uint64_t E = s->n_embd, hc = pulsar_qwen_hc_dim(s);
    const uint64_t conv = pulsar_qwen_gdn_conv_dim(s), vt = pulsar_qwen_gdn_v_total(s);
    const uint64_t q_out = 2ull * s->n_head * s->head_dim, kv_out = (uint64_t)s->n_head_kv * s->head_dim;
    const uint64_t o_in = (uint64_t)s->n_head * s->head_dim;
    const uint64_t idx_out = (uint64_t)(s->idx_n_head + s->idx_n_head_kv) * s->idx_head_dim;
    char name[256];
    auto nm = [&](const char *suffix) -> const char * {
        snprintf(name, sizeof(name), "%s%s", lp, suffix);   /* the directory keeps its own copy */
        return name;
    };
    qwen_bind_gr(m, ok, s, nm("attn_hyper_connection."), &L->gr_attn, true);
    qwen_bind_gr(m, ok, s, nm("mlp_hyper_connection."), &L->gr_mlp, true);
    if (kind == PULSAR_LAYER_QWEN_GDN) {
        L->gdn_in_qkv  = qbind(m, ok, nm("linear_attn.in_proj_qkv.weight"), 2, E, conv);
        L->gdn_in_z    = qbind(m, ok, nm("linear_attn.in_proj_z.weight"), 2, E, vt);
        L->gdn_in_a    = qbind(m, ok, nm("linear_attn.in_proj_a.weight"), 2, E, s->gdn_n_v_head);
        L->gdn_in_b    = qbind(m, ok, nm("linear_attn.in_proj_b.weight"), 2, E, s->gdn_n_v_head);
        L->gdn_conv    = qbind(m, ok, nm("linear_attn.conv1d.weight"), 3, s->gdn_conv_kernel, 1, conv);
        L->gdn_a_log   = qbind(m, ok, nm("linear_attn.A_log"), 1, s->gdn_n_v_head);
        L->gdn_dt_bias = qbind(m, ok, nm("linear_attn.dt_bias"), 1, s->gdn_n_v_head);
        L->gdn_norm    = qbind(m, ok, nm("linear_attn.norm.weight"), 1, s->gdn_v_dim);
        L->gdn_out     = qbind(m, ok, nm("linear_attn.out_proj.weight"), 2, vt, E);
    } else {
        L->attn_q      = qbind(m, ok, nm("self_attn.q_proj.weight"), 2, E, q_out);
        L->attn_k      = qbind(m, ok, nm("self_attn.k_proj.weight"), 2, E, kv_out);
        L->attn_v      = qbind(m, ok, nm("self_attn.v_proj.weight"), 2, E, kv_out);
        L->attn_q_norm = qbind(m, ok, nm("self_attn.q_norm.weight"), 1, s->head_dim);
        L->attn_k_norm = qbind(m, ok, nm("self_attn.k_norm.weight"), 1, s->head_dim);
        L->attn_o      = qbind(m, ok, nm("self_attn.o_proj.weight"), 2, o_in, E);
        L->idx_qk      = qbind(m, ok, nm("self_attn.indexer.index_qk_proj.weight"), 2, E, idx_out);
        L->idx_q_norm  = qbind(m, ok, nm("self_attn.indexer.q_layernorm.weight"), 1, s->idx_head_dim);
        L->idx_k_norm  = qbind(m, ok, nm("self_attn.indexer.k_layernorm.weight"), 1, s->idx_head_dim);
    }
    L->moe_router     = qbind(m, ok, nm("mlp.gate.weight"), 2, E, s->n_expert);
    if (split) {
        L->moe_gate   = qbind(m, ok, nm("mlp.experts.gate_proj"), 3, E, s->n_ff_exp, s->n_expert);
        L->moe_up     = qbind(m, ok, nm("mlp.experts.up_proj"), 3, E, s->n_ff_exp, s->n_expert);
        L->moe_down   = qbind(m, ok, nm("mlp.experts.down_proj"), 3, s->n_ff_exp, E, s->n_expert);
    } else {
        L->moe_gate_up = qbind(m, ok, nm("mlp.experts.gate_up_proj"), 3, E, 2ull * s->n_ff_exp, s->n_expert);
        L->moe_down    = qbind(m, ok, nm("mlp.experts.down_proj"), 3, s->n_ff_exp, E, s->n_expert);
    }
    L->sh_gate        = qbind(m, ok, nm("mlp.shared_expert.gate_proj.weight"), 2, E, s->n_ff_shexp);
    L->sh_up          = qbind(m, ok, nm("mlp.shared_expert.up_proj.weight"), 2, E, s->n_ff_shexp);
    L->sh_down        = qbind(m, ok, nm("mlp.shared_expert.down_proj.weight"), 2, s->n_ff_shexp, E);
    L->sh_gate_scalar = qbind(m, ok, nm("mlp.shared_expert_gate.weight"), 2, E, 1);
    if (ple) {
        L->ple_key        = qbind(m, ok, nm("ple.key_proj.weight"), 2, s->ple_embed_dim, hc);
        L->ple_value      = qbind(m, ok, nm("ple.value_proj.weight"), 2, s->ple_embed_dim, E);
        L->ple_norm_key   = qbind(m, ok, nm("ple.norm_key.weight"), 1, hc);
        L->ple_norm_query = qbind(m, ok, nm("ple.norm_query.weight"), 1, hc);
        L->ple_norm_conv  = qbind(m, ok, nm("ple.norm_conv.weight"), 1, hc);
        L->ple_conv       = qbind(m, ok, nm("ple.conv1d.weight"), 3, s->ple_conv_kernel, 1, hc);
    }
}

/* L251 MTP: the sidecar shard's layer and head-side tensors, bound when the artifact carries
 * mtp.fc_hidden.weight -- then ALL of them or the load refuses (a partial MTP is a broken artifact,
 * not a reason to run without one).  The layer goes to slot n_layer (family_qwen.h). */
static bool qwen_bind_mtp(const pulsar_model *m, const pulsar_qwen_shape *s, const pulsar_layer_plan *plan,
                          pulsar_qwen_weights *w) {
    w->mtp.present = model_find_tensor(m, "mtp.fc_hidden.weight") != NULL;
    if (!w->mtp.present) return true;
    if (plan->n_layer + 1u > PULSAR_FAMILY_MAX_LAYER) {
        fprintf(stderr, "pulsar: %s: no layer slot for the MTP layer past %u -- refusing\n", PULSAR_QWEN_ARCH,
                plan->n_layer);
        return false;
    }
    bool ok = true;
    const uint64_t E = s->n_embd, hc = pulsar_qwen_hc_dim(s);
    w->mtp.norm_embd   = qbind(m, &ok, "mtp.pre_fc_norm_embedding.weight", 1, E);
    w->mtp.norm_hidden = qbind(m, &ok, "mtp.pre_fc_norm_hidden.weight", 1, hc);
    w->mtp.fc_embd     = qbind(m, &ok, "mtp.fc_embedding.weight", 2, E, E);
    w->mtp.fc_hidden   = qbind(m, &ok, "mtp.fc_hidden.weight", 2, E, E);
    qwen_bind_gr(m, &ok, s, "mtp.hyper_connection_mixer.", &w->mtp.mixer, false);
    qwen_bind_layer(m, &ok, s, PULSAR_LAYER_QWEN_QSA, "mtp.layers.0.", false, true, &w->layer[plan->n_layer]);
    if (!ok) fprintf(stderr, "pulsar: %s: the artifact carries mtp.* but not all of it -- refusing\n", PULSAR_QWEN_ARCH);
    return ok;
}

static bool qwen_bind_weights(const pulsar_model *m, const pulsar_qwen_shape *s,
                              const pulsar_layer_plan *plan, pulsar_qwen_weights *w) {
    bool ok = true;
    /* the trunk's routed experts as the builder wrote them (L266): "fused" (one gate_up slice per
     * expert, our quant) or "split" (gate and up apart, turboderp's packs -- their suh differ, so
     * they cannot be fused without a re-quant).  The builder's fact, not a probe of the names. */
    pulsar_str gu = {NULL, 0};
    if (!model_get_string(m, "pulsar.expert_gate_up", &gu)) {
        fprintf(stderr, "pulsar: %s: the artifact has no pulsar.expert_gate_up -- refusing\n", PULSAR_QWEN_ARCH);
        return false;
    }
    const bool split = gu.len == 5 && memcmp(gu.ptr, "split", 5) == 0;
    if (!split && !(gu.len == 5 && memcmp(gu.ptr, "fused", 5) == 0)) {
        fprintf(stderr, "pulsar: %s: pulsar.expert_gate_up is %.*s, neither fused nor split -- refusing\n",
                PULSAR_QWEN_ARCH, (int)gu.len, gu.ptr);
        return false;
    }
    const uint64_t E = s->n_embd, V = s->n_vocab;
    w->token_embd = qbind(m, &ok, QWEN_TEXT "embed_tokens.weight", 2, E, V);
    w->output = qbind(m, &ok, "lm_head.weight", 2, E, V);
    qwen_bind_gr(m, &ok, s, QWEN_TEXT "hyper_connection_mixer.", &w->mixer, false);
    for (uint32_t il = 0; il < plan->n_layer; il++) {
        char lp[128];
        snprintf(lp, sizeof(lp), QWEN_TEXT "layers.%u.", il);
        qwen_bind_layer(m, &ok, s, plan->kind[il], lp, il == s->ple_layer, split, &w->layer[il]);
    }
    ok &= qwen_read_ple_tables(m, s, w);
    ok &= qwen_bind_mtp(m, s, plan, w);
    return ok;
}

/* The checkpoint's tokenizer.json / generation_config.json, read verbatim from the container
 * directory (L251 S5).  The tokenizer's loader is graded against HF on exactly these files
 * (make qwen-chat-gate), so the family takes them as shipped rather than a re-encoding. */
static bool qwen_read_text(const std::string &path, std::string *out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char buf[1 << 16];
    size_t n;
    out->clear();
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out->append(buf, n);
    const bool ok = !ferror(f);
    fclose(f);
    return ok;
}

static bool qwen_load_tokenizer(pulsar_engine *e, const char *model_path) {
    std::string dir = model_path ? model_path : "";
    struct stat st;
    if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        const size_t slash = dir.find_last_of('/');
        dir = slash == std::string::npos ? "." : dir.substr(0, slash);
    }
    std::string tj, gc;
    const std::string tp = dir + "/tokenizer.json", gp = dir + "/generation_config.json";
    if (!qwen_read_text(tp, &tj) || !qwen_read_text(gp, &gc)) {
        fprintf(stderr, "pulsar: %s: the container carries no %s / %s -- copy the checkpoint's own two files "
                        "beside the shards; refusing\n", PULSAR_QWEN_ARCH, tp.c_str(), gp.c_str());
        return false;
    }
    char err[256] = "";
    e->qwen_tok = qwen_tokenizer_load(tj.data(), tj.size(), gc.data(), gc.size(), err, sizeof(err));
    if (!e->qwen_tok) {
        fprintf(stderr, "pulsar: %s: tokenizer refused: %s\n", PULSAR_QWEN_ARCH, err);
        return false;
    }
    const std::vector<int> &stop = qwen_tokenizer_stop_ids(e->qwen_tok);
    fprintf(stderr, "pulsar: %s: tokenizer %d ids from %s; stop ids", PULSAR_QWEN_ARCH,
            qwen_tokenizer_n_tokens(e->qwen_tok), tp.c_str());
    for (int id : stop) fprintf(stderr, " %d", id);
    fprintf(stderr, "\n");
    return true;
}

static bool qwen_family_load(pulsar_engine *e, const pulsar_engine_options *opt) {
    /* The options that name a DeepSeek-only feature refuse here, by name,
     * rather than being ignored. */
    if (opt->expert_overlay && opt->expert_overlay[0]) {
        fprintf(stderr, "pulsar: %s: --expert-overlay is a DeepSeek routed-expert tool; refusing\n",
                PULSAR_QWEN_ARCH);
        return false;
    }
    if (opt->directional_steering_file && opt->directional_steering_file[0]) {
        fprintf(stderr, "pulsar: %s: directional steering is not implemented for this family; refusing\n",
                PULSAR_QWEN_ARCH);
        return false;
    }
    pulsar_qwen_shape s;
    if (!qwen_read_shape(&e->model, &s)) return false;
    if (!qwen_shape_check(&s, &PULSAR_QWEN_SHAPE_FLASH_NEXT)) {
        fprintf(stderr, "pulsar: %s: the artifact is not %s -- refusing\n", PULSAR_QWEN_ARCH,
                PULSAR_QWEN_SHAPE_FLASH_NEXT.name);
        return false;
    }
    g_qwen_shape = s;
    if (!qwen_read_plan(&e->model, s.n_layer, &e->plan)) return false;
    if (e->plan.kind[s.ple_layer] == PULSAR_LAYER_NONE) return false;
    e->qwen_weights = (pulsar_qwen_weights *)xcalloc(1, sizeof(*e->qwen_weights));
    {
        uint64_t h = 1469598103934665603ull;
        auto mix = [&h](const void *p, size_t n) {
            for (size_t i = 0; i < n; i++) { h ^= ((const uint8_t *)p)[i]; h *= 1099511628211ull; }
        };
        for (uint64_t i = 0; i < e->model.n_tensors; i++) {
            const pulsar_tensor *t = &e->model.tensors[i];
            mix(t->name.ptr, t->name.len);
            mix(&t->type, sizeof(t->type));
            mix(&t->bytes, sizeof(t->bytes));
        }
        e->qwen_weights->artifact_digest = h;
    }
    /* L272 B7: the whole-artifact scans DeepSeek's bind runs -- a tensor type no reader takes, a NaN
     * E8M0 scale, a non-finite EXL3 scale -- run for this family's artifact too. */
    pulsar_load_refusals_reset();   /* L272: the loader's one failure policy (family.cpp) */
    weights_reject_unsupported_types(&e->model);
    weights_reject_bad_e8m0(&e->model);
    if (pulsar_load_refusals()) {
        fprintf(stderr, "pulsar: %s: %u refusal(s) in the artifact scans -- the model does not load\n", PULSAR_QWEN_ARCH,
                pulsar_load_refusals());
        return false;
    }
    if (!qwen_bind_weights(&e->model, &g_qwen_shape, &e->plan, e->qwen_weights)) {
        fprintf(stderr, "pulsar: %s: the artifact does not bind -- refusing\n", PULSAR_QWEN_ARCH);
        return false;
    }
    if (!pulsar_qwen_s4_load(e, opt)) return false;
    if (!pulsar_qwen_tp_load(e)) return false;
    if (!qwen_load_tokenizer(e, opt->model_path)) return false;
    /* L268: the vision tower, when the artifact carries it -- then images are served (the front, the tower, the
     * multi-axis rope through QSA); a text-only artifact refuses them by name */
    if (!qwen_vision_bind(&e->model, &e->qwen_weights->vision, &e->qwen_weights->vision_present)) return false;
    e->qwen_weights->vision_pad_id = -1;
    if (e->qwen_weights->vision_present) {
        e->qwen_weights->vision_pad_id = qwen_tokenizer_added_id(e->qwen_tok, "<|image_pad|>");
        if (e->qwen_weights->vision_pad_id < 0 || g_qwen_shape.n_embd != PULSAR_QWEN_VISION_OUT) {
            fprintf(stderr, "pulsar: %s: the vision tower needs <|image_pad|> in the tokenizer and a %u-wide model "
                            "(pad %d, n_embd %u) -- refusing\n", PULSAR_QWEN_ARCH, (unsigned)PULSAR_QWEN_VISION_OUT,
                    e->qwen_weights->vision_pad_id, g_qwen_shape.n_embd);
            return false;
        }
        e->vision_ready = true;
    }
    /* L272 P1: the MTP layer is the drafter behind the round API (spec_qwen.cpp); the drafter option
     * (--no-dspark) turns drafting off for this family as it does for DeepSeek's */
    if (e->qwen_weights->mtp.present && !opt->dspark_disable) e->drafter_ops = &k_mtp_drafter;
    fprintf(stderr, "pulsar: %s: %s, %u GDN + %u QSA layers, PLE at layer %u\n",
            PULSAR_QWEN_ARCH, g_qwen_shape.name,
            pulsar_layer_plan_count(&e->plan, PULSAR_LAYER_QWEN_GDN),
            pulsar_layer_plan_count(&e->plan, PULSAR_LAYER_QWEN_QSA), g_qwen_shape.ple_layer);
    return true;
}

/* L266 step 7: tensor parallelism at load (family_qwen.h). */
bool pulsar_qwen_tp_load(pulsar_engine *e) {
    pulsar_model *m = &e->model;
    if (m->tp_n_ranks <= 1) return true;
    pulsar_qwen_shape *s = &g_qwen_shape;
    const uint32_t nr = m->tp_n_ranks;
    if (nr != 2 || m->tp_rank < 0 || m->tp_rank >= 2 || s->gdn_n_k_head % nr || s->gdn_n_v_head % nr ||
        s->n_head_kv % nr || s->n_head % nr || s->n_expert % nr || s->n_vocab % nr) {
        fprintf(stderr, "pulsar: %s: tensor parallelism is built for 2 ranks over whole heads, experts and vocab "
                        "rows (rank %d of %u) -- refusing\n", PULSAR_QWEN_ARCH, m->tp_rank, nr);
        return false;
    }
    /* the rank's heads: every derived width, state size and scratch layout follows */
    s->gdn_n_k_head /= nr;
    s->gdn_n_v_head /= nr;
    s->n_head /= nr;
    s->n_head_kv /= nr;
    s->tp_rank = (uint32_t)m->tp_rank;
    s->tp_ranks = nr;
    /* a rank's KV is its heads' only: its segments never load into another rank or one GPU */
    pulsar_qwen_weights *w = e->qwen_weights;
    const uint64_t rk[2] = {(uint64_t)m->tp_rank, nr};
    for (int i = 0; i < 2; i++)
        for (int b = 0; b < 8; b++) { w->artifact_digest ^= (uint8_t)(rk[i] >> (8 * b)); w->artifact_digest *= 1099511628211ull; }
    uint32_t e0 = 0, e1 = 0;
    (void)pulsar_tp_owned_range(m->tp_rank, nr, s->n_expert, &e0, &e1);
    fprintf(stderr, "pulsar: %s TP rank %d/%u: GDN %u key + %u value heads, QSA %u query + %u KV heads, experts "
                    "[%u, %u)\n", PULSAR_QWEN_ARCH, m->tp_rank, nr, s->gdn_n_k_head, s->gdn_n_v_head, s->n_head,
            s->n_head_kv, e0, e1);
    return true;
}

/* The ops that exist, announced once at open (rule 5): which of the forward's
 * ops this build carries, so a log says what a Qwen engine CAN run. */
static bool qwen_family_after_gpu(pulsar_engine *e) {
    char have[256] = "", missing[256] = "";
    for (int op = 0; op < PULSAR_QWEN_OP_COUNT; op++) {
        char *dst = qwen_op_present(&g_qwen_ops, (pulsar_qwen_op_id)op) ? have : missing;
        const size_t n = strlen(dst);
        snprintf(dst + n, 256 - n, "%s%s", n ? " " : "", pulsar_qwen_op_name((pulsar_qwen_op_id)op));
    }
    uint32_t at = 0;
    const pulsar_qwen_op_id first = pulsar_qwen_first_missing_op(&g_qwen_ops, &e->plan, &g_qwen_shape, &at);
    fprintf(stderr, "pulsar: %s ops present: [%s]; missing: [%s]%s%s\n", PULSAR_QWEN_ARCH,
            have, missing, first == PULSAR_QWEN_OP_COUNT ? "" : "; a step refuses at ",
            first == PULSAR_QWEN_OP_COUNT ? "" : pulsar_qwen_op_name(first));
    return true;
}

/* ---- engine facts ------------------------------------------------------------- */

static uint32_t qwen_logits_width(const pulsar_engine *) { return g_qwen_shape.n_vocab; }
/* L272 B15: the drafter the opened artifact carries -- the MTP layer, when its sidecar shard loaded. */
static pulsar_drafter_kind qwen_drafter(pulsar_engine *e) {
    return e->qwen_weights && e->qwen_weights->mtp.present ? PULSAR_DRAFTER_MTP : PULSAR_DRAFTER_NONE;
}
/* The routed experts' precision tier in EXL3's value space (family.h quant_bits): 20 + the highest
 * rate present in half bits.  Qwen admits EXL3 experts only, so the tier is never 2 or 4. */
static int qwen_quant_bits(pulsar_engine *e) {
    int bits = 0;
    if (!e->qwen_weights) return 0;
    for (uint32_t il = 0; il < e->plan.n_layer; il++) {
        const pulsar_qwen_layer_weights &L = e->qwen_weights->layer[il];
        for (const pulsar_tensor *t : {L.moe_gate_up, L.moe_gate, L.moe_up, L.moe_down}) {
            const int k2 = t ? exl3_type_k2(t->type) : 0;
            if (k2 && 20 + k2 > bits) bits = 20 + k2;
        }
    }
    return bits;
}
static const char *qwen_model_name(const pulsar_engine *) { return g_qwen_shape.name; }
static const char *qwen_served_model_id(const pulsar_engine *) { return "qwen3.8-flash-next"; }
static pulsar_chat_format qwen_chat_format(const pulsar_engine *) { return PULSAR_CHAT_QWEN; }
/* Disk-KV compatibility id: 0 and 1 are DeepSeek's two profiles. */
static int qwen_model_id(const pulsar_engine *) { return 2; }
/* no position scaling yet (L280): the trained positions are the limit */
static uint64_t qwen_trained_context(const pulsar_engine *) { return g_qwen_shape.max_position; }

/* ---- session state ------------------------------------------------------------ */

static uint64_t qwen_ceil_div(uint64_t a, uint64_t b) { return (a + b - 1u) / b; }

/* L284 #3: one bank's demand-paged tensors on QSA layer il, each made where it is missing.  The sizes are the
 * op's per-bank views' (st->ctx is the block-rounded capacity; pulsar_qwen_qsa_cap). */
static bool qwen_layer_bank_alloc(pulsar_qwen_state *st, uint32_t il, uint32_t bank) {
    const pulsar_qwen_shape *s = &g_qwen_shape;
    pulsar_qwen_layer_state *L = &st->layer[il];
    if (!L->kv[bank]) L->kv[bank] = pulsar_gpu_tensor_alloc_managed((uint64_t)st->ctx * pulsar_qwen_kv_row_bytes(s));
    if (!L->idx_keys[bank])
        L->idx_keys[bank] = pulsar_gpu_tensor_alloc_managed(qwen_ceil_div(st->ctx, s->idx_block) *
                                                            pulsar_qwen_index_row_bytes(s));
    return L->kv[bank] && L->idx_keys[bank];
}

static void qwen_layer_bank_free(pulsar_qwen_state *st, uint32_t il, uint32_t bank) {
    pulsar_qwen_layer_state *L = &st->layer[il];
    pulsar_gpu_tensor_free(L->kv[bank]);
    pulsar_gpu_tensor_free(L->idx_keys[bank]);
    L->kv[bank] = NULL;
    L->idx_keys[bank] = NULL;
}

void qwen_bank_kv_free(pulsar_qwen_state *st, uint32_t bank) {
    for (uint32_t il = 0; il < st->n_trunk_layers; il++)
        if (st->layer[il].kv) qwen_layer_bank_free(st, il, bank);
    st->kv_hw[bank] = 0;   /* its pages are gone; the restore's loads raise it again */
}

bool qwen_bank_kv_alloc(pulsar_qwen_state *st, uint32_t bank) {
    for (uint32_t il = 0; il < st->n_trunk_layers; il++) {
        if (st->layer[il].kv && !qwen_layer_bank_alloc(st, il, bank)) {
            qwen_bank_kv_free(st, bank);   /* never half-backed */
            return false;
        }
    }
    return true;
}

bool qwen_bank_kv_evicted(const pulsar_qwen_state *st, uint32_t bank) {
    for (uint32_t il = 0; il < st->n_trunk_layers; il++) {
        const pulsar_qwen_layer_state *L = &st->layer[il];
        if (L->kv && (!L->kv[bank] || !L->idx_keys[bank])) return true;
    }
    return false;
}

static void qwen_state_free(pulsar_qwen_state *st) {
    if (!st) return;
    for (uint32_t il = 0; il < PULSAR_FAMILY_MAX_LAYER; il++) {
        pulsar_qwen_layer_state *L = &st->layer[il];
        pulsar_gpu_tensor_free(L->gdn_state);
        pulsar_gpu_tensor_free(L->gdn_conv);
        if (L->kv)
            for (uint32_t b = 0; b < st->n_banks; b++) qwen_layer_bank_free(st, il, b);
        free(L->kv);
        free(L->idx_keys);
        pulsar_gpu_tensor_free(L->idx_tail);
        pulsar_gpu_tensor_free(L->ple_conv);
    }
    pulsar_gpu_tensor_free(st->streams);
    pulsar_gpu_tensor_free(st->x);
    pulsar_gpu_tensor_free(st->y);
    pulsar_gpu_tensor_free(st->logits);
    pulsar_gpu_tensor_free(st->row_pos);
    pulsar_gpu_tensor_free(st->row_bank);
    pulsar_gpu_tensor_free(st->mtp_streams);
    pulsar_gpu_tensor_free(st->mtp_h);
    pulsar_gpu_tensor_free(st->mtp_tok);
    pulsar_gpu_tensor_free(st->mtp_ws);
    pulsar_gpu_tensor_free(st->mtp_pend);
    free(st->mtp_pend_pos);
    free(st->spec_logits);
    pulsar_gpu_tensor_free(st->spec.gdn_rec);
    pulsar_gpu_tensor_free(st->spec.gdn_conv);
    pulsar_gpu_tensor_free(st->spec.ple);
    pulsar_gpu_tensor_free(st->spec.qsa_stage);
    pulsar_gpu_tensor_free(st->spec.qsa_keys);
    pulsar_gpu_tensor_free(st->mtp_stage);
    free(st->mtp_stage_dirty);
    pulsar_gpu_tensor_free(st->run_first_dev);
    for (int op = 0; op < PULSAR_QWEN_OP_COUNT; op++) pulsar_gpu_tensor_free(st->scratch[op]);
    free(st->ngram_ctx);
    free(st->bank_pos);
    free(st->kv_hw);
    pulsar_gpu_tensor_free(st->tp_ticket);
    pulsar_gpu_tensor_free(st->tp_vocab_own);
    if (st->ckpt) pulsar_ckpt_release(st->ckpt);
    free(st->ckpt);
    free(st->prefill_pos);
    free(st->frontier_stale);
    free(st);
}

/* Allocate a session's state.  The same code prices it: under
 * pulsar_gpu_tensor_dry_begin the allocators total the bytes instead
 * (qwen_session_cost_bytes). */
/* The layer kind of state slot il: the plan's for the trunk, QSA for the MTP layer at n_layer. */
static pulsar_layer_kind qwen_state_kind(const pulsar_layer_plan *plan, uint32_t il) {
    return il < plan->n_layer ? plan->kind[il] : PULSAR_LAYER_QWEN_QSA;
}

static pulsar_qwen_state *qwen_state_alloc(const pulsar_qwen_shape *s, const pulsar_layer_plan *plan,
                                           uint32_t n_banks, uint32_t ctx, uint32_t max_rows, bool mtp) {
    pulsar_qwen_state *st = (pulsar_qwen_state *)xcalloc(1, sizeof(*st));
    st->n_banks = n_banks;
    st->ctx = pulsar_qwen_qsa_cap(s, ctx);   /* a whole number of indexer blocks; see the helper */
    st->max_rows = max_rows;
    st->mtp = mtp;
    bool ok = true;
    const uint64_t nb = n_banks;
    const uint32_t n_state = plan->n_layer + (mtp ? 1u : 0u);   /* the MTP layer's KV at slot n_layer */
    for (uint32_t il = 0; ok && il < n_state; il++) {
        pulsar_qwen_layer_state *L = &st->layer[il];
        if (qwen_state_kind(plan, il) == PULSAR_LAYER_QWEN_GDN) {
            L->gdn_state = pulsar_gpu_tensor_alloc(nb * pulsar_qwen_gdn_state_bytes(s));
            L->gdn_conv  = pulsar_gpu_tensor_alloc(nb * pulsar_qwen_gdn_conv_bytes(s));
            ok = L->gdn_state && L->gdn_conv;
        } else {
            /* Demand-paged: a bank pays for the KV it touches.  Sized from st->ctx, NOT the raw
             * parameter: st->ctx is the block-rounded QSA capacity (see pulsar_qwen_qsa_cap), and
             * the op views exactly st->ctx tokens per bank -- allocating the raw ctx made the KV
             * view overshoot its tensor and the op refused with "a QSA bank cache view failed".  One
             * tensor per bank (L284 #3), so a bank's pages can be returned (qwen_bank_kv_free). */
            L->kv       = (pulsar_gpu_tensor **)xcalloc(n_banks, sizeof(*L->kv));
            L->idx_keys = (pulsar_gpu_tensor **)xcalloc(n_banks, sizeof(*L->idx_keys));
            for (uint32_t b = 0; ok && b < n_banks; b++) ok = qwen_layer_bank_alloc(st, il, b);
            L->idx_tail = ok ? pulsar_gpu_tensor_alloc(nb * pulsar_qwen_index_tail_bytes(s)) : NULL;
            ok = ok && L->idx_tail;
        }
        if (ok && il == s->ple_layer) {
            L->ple_conv = pulsar_gpu_tensor_alloc(nb * pulsar_qwen_ple_conv_bytes(s));
            ok = L->ple_conv != NULL;
        }
    }
    const uint64_t rows = max_rows;
    if (ok) {
        st->streams  = pulsar_gpu_tensor_alloc(rows * pulsar_qwen_hc_dim(s) * PULSAR_QWEN_STREAM_ELT_SIZE);
        st->x        = pulsar_gpu_tensor_alloc(rows * s->n_embd * PULSAR_QWEN_X_ELT_SIZE);
        st->y        = pulsar_gpu_tensor_alloc(rows * s->n_embd * PULSAR_QWEN_Y_ELT_SIZE);
        st->logits   = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_QWEN_HEAD_ROWS_MAX * s->n_vocab * sizeof(float));
        st->row_pos  = pulsar_gpu_tensor_alloc(rows * sizeof(int32_t));
        st->row_bank = pulsar_gpu_tensor_alloc(rows * sizeof(int32_t));
        ok = st->streams && st->x && st->y && st->logits && st->row_pos && st->row_bank;
    }
    if (ok && mtp) {
        const uint64_t hc = pulsar_qwen_hc_dim(s) * PULSAR_QWEN_STREAM_ELT_SIZE;
        st->mtp_streams = pulsar_gpu_tensor_alloc(rows * hc);
        st->mtp_h       = pulsar_gpu_tensor_alloc((rows + 1u) * hc);
        st->mtp_tok     = pulsar_gpu_tensor_alloc(rows * sizeof(int32_t));
        st->mtp_ws      = pulsar_gpu_tensor_alloc(pulsar_qwen_s4_mtp_combine_ws_bytes());
        st->mtp_pend    = pulsar_gpu_tensor_alloc(nb * hc);
        ok = st->mtp_streams && st->mtp_h && st->mtp_tok && st->mtp_ws && st->mtp_pend;
    }
    st->mtp_pend_pos = (uint32_t *)xmalloc(n_banks * sizeof(uint32_t));
    for (uint32_t b = 0; b < n_banks; b++) st->mtp_pend_pos[b] = UINT32_MAX;
    if (mtp) st->spec_logits = (float *)xmalloc((size_t)PULSAR_QWEN_SPEC_ROWS * s->n_vocab * sizeof(float));
    if (ok && mtp) {
        /* the verify capture (L272 P1 S4: every bank's run of one step): per-row recurrent states at the row's
         * step index, each QSA layer's stage per run + the rows' raw keys; the MTP layer's stage per bank */
        pulsar_qwen_spec_capture &sp = st->spec;
        for (uint32_t il = 0; il < plan->n_layer; il++)
            sp.ord[il] = (uint8_t)(plan->kind[il] == PULSAR_LAYER_QWEN_GDN ? sp.n_gdn++ : sp.n_qsa++);
        sp.ord[plan->n_layer] = (uint8_t)sp.n_qsa;           /* the MTP layer's stage slot */
        const uint64_t D = PULSAR_QWEN_SPEC_ROWS;
        sp.gdn_rec   = pulsar_gpu_tensor_alloc((uint64_t)sp.n_gdn * D * pulsar_qwen_gdn_state_bytes(s));
        sp.gdn_conv  = pulsar_gpu_tensor_alloc((uint64_t)sp.n_gdn * D * pulsar_qwen_gdn_conv_bytes(s));
        sp.ple       = pulsar_gpu_tensor_alloc(D * pulsar_qwen_ple_conv_bytes(s));
        sp.qsa_stage = pulsar_gpu_tensor_alloc((uint64_t)sp.n_qsa * D * pulsar_qwen_index_tail_bytes(s));
        sp.qsa_keys  = pulsar_gpu_tensor_alloc((uint64_t)sp.n_qsa * D * PULSAR_QSA_IDX_IN * sizeof(float));
        st->mtp_stage = pulsar_gpu_tensor_alloc((uint64_t)n_banks * pulsar_qwen_index_tail_bytes(s));
        st->run_first_dev = pulsar_gpu_tensor_alloc((uint64_t)(D + 1u) * sizeof(int32_t));
        ok = sp.gdn_rec && sp.gdn_conv && sp.ple && sp.qsa_stage && sp.qsa_keys && st->mtp_stage && st->run_first_dev;
    }
    if (mtp) st->mtp_stage_dirty = (bool *)xcalloc(n_banks, sizeof(bool));
    if (ok && mtp) {
    }
    for (int op = 0; ok && op < PULSAR_QWEN_OP_COUNT; op++) {
        const uint64_t b = g_qwen_ops.scratch_bytes
                               ? g_qwen_ops.scratch_bytes((pulsar_qwen_op_id)op, s, max_rows, ctx) : 0;
        if (b) ok = (st->scratch[op] = pulsar_gpu_tensor_alloc(b)) != NULL;
    }
    /* L266 step 7: under TP, the row all-reduce's stage ticket (one u32) -- allocated here so the dry run
     * prices it with the rest of the state */
    if (ok && pulsar_qwen_tp(s) > 1) {
        /* and the vocab gather's own slice (its row lane's; sized whatever the transport, so the price holds) */
        const uint64_t vb = pulsar_tp_vocab_own_bytes_for(pulsar_qwen_tp(s), (uint64_t)s->n_embd * sizeof(float),
                                                          s->n_vocab, PULSAR_QWEN_HEAD_ROWS_MAX);
        ok = (st->tp_ticket = pulsar_gpu_tensor_alloc(sizeof(uint32_t))) != NULL &&
             (st->tp_vocab_own = pulsar_gpu_tensor_alloc(vb)) != NULL;
    }
    st->ngram_ctx = (int32_t *)xcalloc((size_t)n_banks * (s->ngram_size - 1u), sizeof(int32_t));
    st->bank_pos = (uint32_t *)xcalloc(n_banks, sizeof(uint32_t));
    st->kv_hw = (uint32_t *)xcalloc(n_banks, sizeof(uint32_t));
    st->prefill_pos = (uint32_t *)xcalloc(n_banks, sizeof(uint32_t));
    st->frontier_stale = (bool *)xcalloc(n_banks, sizeof(bool));
    st->n_trunk_layers = plan->n_layer;
    /* L266 step 5: the grid checkpoints (kv_state_qwen.cpp), ckpt_slots x ~118 MB per bank -- priced with
     * the rest under the dry run */
    st->ckpt = (pulsar_ckpt_store *)xcalloc(1, sizeof(pulsar_ckpt_store));
    if (ok && n_banks > PULSAR_MSEQ_MAX) {
        fprintf(stderr, "pulsar: %s: %u banks, the checkpoint store holds at most %u -- refusing\n", PULSAR_QWEN_ARCH,
                n_banks, (unsigned)PULSAR_MSEQ_MAX);
        ok = false;
    }
    if (ok) ok = pulsar_ckpt_alloc(st->ckpt, &PULSAR_KV_STATE_QWEN, st, n_banks);
    if (!ok) {
        fprintf(stderr, "pulsar: %s: session state allocation failed (%u banks x %u tokens)\n",
                PULSAR_QWEN_ARCH, n_banks, ctx);
        qwen_state_free(st);
        return NULL;
    }
    return st;
}

/* A bank starts fresh: its recurrent, conv and index-tail state is zero, and
 * its n-gram context is the reset token (HF resets the n-gram context at EOS,
 * and a sequence start is a reset). */
static bool qwen_state_reset_bank(pulsar_qwen_state *st, const pulsar_qwen_shape *s,
                                  const pulsar_layer_plan *plan, uint32_t bank) {
    bool ok = true;
    const uint32_t n_state = plan->n_layer + (st->mtp ? 1u : 0u);
    for (uint32_t il = 0; ok && il < n_state; il++) {
        pulsar_qwen_layer_state *L = &st->layer[il];
        struct { pulsar_gpu_tensor *t; uint64_t bytes; } z[] = {
            { L->gdn_state, pulsar_qwen_gdn_state_bytes(s) },
            { L->gdn_conv,  pulsar_qwen_gdn_conv_bytes(s) },
            { L->idx_tail,  pulsar_qwen_index_tail_bytes(s) },
            { L->ple_conv,  pulsar_qwen_ple_conv_bytes(s) },
        };
        for (size_t k = 0; ok && k < sizeof(z) / sizeof(z[0]); k++) {
            if (!z[k].t) continue;
            pulsar_gpu_tensor *v = pulsar_gpu_tensor_view(z[k].t, (uint64_t)bank * z[k].bytes, z[k].bytes);
            ok = v && pulsar_gpu_tensor_fill_f32(v, 0.0f, z[k].bytes / sizeof(float)) != 0;
            pulsar_gpu_tensor_free(v);
        }
    }
    for (uint32_t i = 0; i + 1u < s->ngram_size; i++)
        st->ngram_ctx[(size_t)bank * (s->ngram_size - 1u) + i] = (int32_t)s->eos_id;
    qwen_bank_set_pos(st, bank, 0);
    st->prefill_pos[bank] = 0;
    st->frontier_stale[bank] = false;
    st->mtp_pend_pos[bank] = UINT32_MAX;
    if (st->ckpt && st->ckpt->ops) pulsar_ckpt_drop_bank(st->ckpt, bank);   /* its history is gone */
    return ok;
}


/* The family's state, built into a session the core allocated (L272 P2: the core sets ctx_size, the prefill
 * cap and the logits row, and measures the bytes this allocates). */
static int qwen_session_create(pulsar_session *s) {
    pulsar_engine *e = s->engine;
    const uint32_t n_banks = gpu_graph_bank_pool_n();
    s->qwen = qwen_state_alloc(&g_qwen_shape, &e->plan, n_banks, (uint32_t)s->ctx_size, s->prefill_cap,
                               e->qwen_weights->mtp.present);
    if (!s->qwen) return 1;
    for (uint32_t b = 0; b < n_banks; b++) {
        if (!qwen_state_reset_bank(s->qwen, &g_qwen_shape, &e->plan, b)) {
            fprintf(stderr, "pulsar: %s: could not clear bank %u's state\n", PULSAR_QWEN_ARCH, b);
            qwen_state_free(s->qwen);
            s->qwen = NULL;
            return 1;
        }
    }
    s->qwen->ckpt->artifact = e->qwen_weights->artifact_digest;
    if (e->tp) {   /* L266 step 7: the engine's lanes; the ticket is the state's (priced with it), zeroed here */
        const uint32_t zero = 0;
        s->qwen->tp = e->tp;
        s->qwen->tp_slab_dev = e->tp_slab_dev;
        s->qwen->tp_bulk_dev = e->tp_bulk_dev;
        if (!s->qwen->tp_ticket || !pulsar_gpu_tensor_write(s->qwen->tp_ticket, 0, &zero, sizeof(zero))) {
            fprintf(stderr, "pulsar: %s: the TP stage ticket is missing\n", PULSAR_QWEN_ARCH);
            qwen_state_free(s->qwen);
            s->qwen = NULL;
            return 1;
        }
    }
    fprintf(stderr, "pulsar: %s session: %u bank(s) x %d tokens, %u-row steps "
                    "(%.1f MiB fixed per bank + %.1f KiB per token)\n",
            PULSAR_QWEN_ARCH, n_banks, s->ctx_size, s->prefill_cap,
            (double)(pulsar_layer_plan_count(&e->plan, PULSAR_LAYER_QWEN_GDN) *
                     (pulsar_qwen_gdn_state_bytes(&g_qwen_shape) + pulsar_qwen_gdn_conv_bytes(&g_qwen_shape)) +
                     pulsar_qwen_ple_conv_bytes(&g_qwen_shape)) / 1048576.0,
            (double)(pulsar_layer_plan_count(&e->plan, PULSAR_LAYER_QWEN_QSA) *
                     (pulsar_qwen_kv_row_bytes(&g_qwen_shape) +
                      pulsar_qwen_index_row_bytes(&g_qwen_shape) / g_qwen_shape.idx_block)) / 1024.0);
    return 0;
}

static void qwen_session_destroy(pulsar_session *s) {
    if (s->qwen && s->qwen->mtp_probe_n)
        fprintf(stderr, "pulsar: %s: MTP probe: draft position 1 agreed with the trunk's argmax %llu of %llu (%.1f%%)\n",
                PULSAR_QWEN_ARCH, (unsigned long long)s->qwen->mtp_probe_hit, (unsigned long long)s->qwen->mtp_probe_n,
                100.0 * (double)s->qwen->mtp_probe_hit / (double)s->qwen->mtp_probe_n);
    qwen_state_free(s->qwen);
    s->qwen = NULL;
}

uint64_t pulsar_qwen_state_price(const pulsar_qwen_shape *s, const pulsar_layer_plan *plan,
                                 uint32_t n_banks, uint32_t ctx, uint32_t max_rows, bool mtp,
                                 uint64_t *managed_bytes) {
    pulsar_gpu_tensor_dry_begin();
    pulsar_qwen_state *st = qwen_state_alloc(s, plan, n_banks, ctx, max_rows, mtp);
    uint64_t bytes = 0, managed = 0;
    pulsar_gpu_tensor_dry_end(&bytes, &managed);
    const bool ok = st != NULL;
    qwen_state_free(st);
    if (managed_bytes) *managed_bytes = ok ? managed : 0;
    return ok ? bytes : 0;
}

uint64_t qwen_kv_bytes_at(const pulsar_qwen_state *st, uint64_t rows) {
    const pulsar_qwen_shape *s = &g_qwen_shape;
    uint64_t bytes = 0;
    for (uint32_t il = 0; il < PULSAR_FAMILY_MAX_LAYER; il++) {
        if (!st->layer[il].kv) continue;
        bytes += rows * pulsar_qwen_kv_row_bytes(s) + qwen_ceil_div(rows, s->idx_block) * pulsar_qwen_index_row_bytes(s);
    }
    return bytes;
}

/* L270: one bank's demand-paged share at ctx_size -- the managed bytes of the allocation's own dry
 * run (the KV and index pools are cudaMallocManaged: VA reserved, physical on touch). */
uint64_t qwen_demand_paged_bytes(pulsar_engine *e, int ctx_size) {
    if (!pulsar_backend_uses_graph(e->backend) || !e->gpu_ready || ctx_size <= 0) return 0;
    uint64_t managed = 0;
    pulsar_qwen_state_price(&g_qwen_shape, &e->plan, 1u, (uint32_t)ctx_size, pulsar_prefill_cap_for_prompt(ctx_size, e->prefill_chunk),
                            e->qwen_weights->mtp.present, &managed);
    return managed;
}

static uint64_t qwen_session_cost_bytes(pulsar_engine *e, int ctx_size, int n_banks) {
    if (!pulsar_backend_uses_graph(e->backend) || !e->gpu_ready) return 0;
    return pulsar_qwen_state_price(&g_qwen_shape, &e->plan, (uint32_t)n_banks, (uint32_t)ctx_size,
                                   pulsar_prefill_cap_for_prompt(ctx_size, e->prefill_chunk), e->qwen_weights->mtp.present, NULL);
}

/* ---- the step driver ------------------------------------------------------------ */

/* One step: the plan, in the forward's order (family_qwen.h).  Refuses by
 * name before touching any state when an op the step needs is missing. */
/* L266 step 7: under TP a rank's mixer (its heads) and MoE (its experts, rank 0's shared expert) leave a
 * PARTIAL block output in y; the group's sum is the block's.  One exchange each, f32 [rows][n_embd]; the
 * MTP layer's take the slot after the trunk's.  One GPU: nothing to do. */
static bool qwen_tp_allreduce_y(pulsar_session *s, uint32_t il, uint32_t n_rows, const char *what) {
    pulsar_qwen_state *q = s->qwen;
    if (!q->tp) return true;
    const pulsar_tp_rows x = {q->tp, q->tp_slab_dev, q->tp_bulk_dev, q->tp_ticket, &q->tp_seq, g_qwen_shape.n_embd};
    pulsar_gpu_tensor *y = pulsar_gpu_tensor_view(q->y, 0, (uint64_t)n_rows * g_qwen_shape.n_embd * PULSAR_QWEN_Y_ELT_SIZE);
    const bool ok = y && pulsar_tp_allreduce_rows(&x, il, n_rows, y, NULL, what);
    pulsar_gpu_tensor_free(y);
    return ok;
}

/* L268: the step's multi-axis rope positions (pulsar_image_rope3 over each row's bank's image records) -- NULL when
 * no row's bank holds a gridded image block: text positions, the bytes a text step always ran.  Per row, its own
 * (T, H, W) and its indexer block's first token's (pos - 3), which may be an earlier step's. */
static const uint32_t *qwen_rope_table(const pulsar_session *s, const int32_t *pos, const int32_t *bank, uint32_t n,
                                       std::vector<uint32_t> *out) {
    bool any = false;
    for (uint32_t r = 0; r < n && !any; r++) {
        const pulsar_image_identity *id = pulsar_session_bank_images(s, (uint32_t)bank[r], s->qwen->live_bank);
        for (uint32_t i = 0; id && i < id->n && !any; i++) any = id->b[i].grid_h && id->b[i].grid_w;
    }
    if (!any) return NULL;
    out->resize((size_t)n * 6u);
    for (uint32_t r = 0; r < n; r++) {
        const pulsar_image_identity *id = pulsar_session_bank_images(s, (uint32_t)bank[r], s->qwen->live_bank);
        const uint32_t p = (uint32_t)pos[r], back = PULSAR_QSA_BLOCK - 1u;
        pulsar_image_rope3(id, p, &(*out)[(size_t)r * 6u]);
        pulsar_image_rope3(id, p >= back ? p - back : 0u, &(*out)[(size_t)r * 6u + 3u]);
    }
    return out->data();
}

/* L268: image rows into the step's streams -- each row in all of them (the expansion follows the substitution) */
static bool qwen_write_image_rows(void *ud, const uint16_t *rows, uint32_t n_rows, uint32_t row0, uint32_t n_tokens) {
    return pulsar_image_write_stream_rows((pulsar_gpu_tensor *)ud, rows, n_rows, row0, n_tokens, g_qwen_shape.n_embd,
                                          g_qwen_shape.n_hc);
}

qwen_heads qwen_heads_span(uint32_t row0, uint32_t n) {
    qwen_heads h;
    h.n = n;
    for (uint32_t i = 0; i < n && i < PULSAR_QWEN_HEAD_ROWS_MAX; i++) h.row[i] = row0 + i;
    return h;
}

bool qwen_forward(pulsar_session *s, pulsar_qwen_step_mode mode, const int32_t *tokens,
                  const int32_t *pos, const int32_t *bank, uint32_t n_rows,
                  const qwen_heads &heads, float *logits_out, uint32_t n_verify) {
    pulsar_engine *e = s->engine;
    const pulsar_qwen_ops *ops = &g_qwen_ops;
    uint32_t at = 0;
    const pulsar_qwen_op_id miss = pulsar_qwen_first_missing_op(ops, &e->plan, &g_qwen_shape, &at);
    if (miss != PULSAR_QWEN_OP_COUNT) {
        char where[48] = "";
        if (at != UINT32_MAX) snprintf(where, sizeof(where), " at layer %u (%s)", at,
                                       pulsar_layer_kind_name(e->plan.kind[at]));
        fprintf(stderr, "pulsar: %s: op '%s'%s is not implemented yet (owner: %s) -- refusing the "
                        "%s step\n", PULSAR_QWEN_ARCH, pulsar_qwen_op_name(miss), where,
                pulsar_qwen_op_owner(miss), mode == PULSAR_QWEN_STEP_PREFILL ? "prefill" : "decode");
        return false;
    }
    if (n_rows == 0 || n_rows > s->qwen->max_rows || heads.n > PULSAR_QWEN_HEAD_ROWS_MAX) {
        fprintf(stderr, "pulsar: %s: a step of %u rows heading %u (at most %u rows, %u heads) -- refusing\n",
                PULSAR_QWEN_ARCH, n_rows, heads.n, s->qwen->max_rows, PULSAR_QWEN_HEAD_ROWS_MAX);
        return false;
    }
    pulsar_qwen_step st{};
    st.model = &e->model;
    st.shape = &g_qwen_shape;
    st.w = e->qwen_weights;
    st.plan = &e->plan;
    st.st = s->qwen;
    st.mode = mode;
    st.n_rows = n_rows;
    st.tokens = tokens;
    st.pos = pos;
    st.bank = bank;
    st.streams = s->qwen->streams;
    st.mixer = &e->qwen_weights->mixer;
    st.verify = n_verify > 0;
    st.n_dec = pulsar_qwen_step_n_dec(mode, n_verify, n_rows);
    std::vector<uint32_t> rope;
    st.rope = qwen_rope_table(s, pos, bank, n_rows, &rope);
    if (st.verify && (mode != PULSAR_QWEN_STEP_PREFILL || n_verify > n_rows || n_verify > PULSAR_QWEN_SPEC_ROWS ||
                      !s->qwen->mtp)) {
        fprintf(stderr, "pulsar: %s: a verify step of %u rows is outside the capture -- refusing\n", PULSAR_QWEN_ARCH,
                n_verify);
        return false;
    }
    /* L272 P1 S4 / L284 #2: a PREFILL step's rows as runs -- one bank each, consecutive positions, a bank in one
     * run.  The verify rows [0, n_verify) are one run a bank (every row headed); a run never crosses n_verify; the
     * rows after it are the fused step's prompt runs, at most PULSAR_FUSED_PF_MAX of them. */
    uint32_t run_first[PULSAR_QWEN_SPEC_ROWS + PULSAR_FUSED_PF_MAX + 1] = {0};
    uint32_t n_runs = 0;
    if (mode == PULSAR_QWEN_STEP_PREFILL) {
        n_runs = 1;
        uint32_t n_prompt = n_verify == 0 ? 1u : 0u;   /* the runs from n_verify on */
        for (uint32_t r = 1; r < n_rows; r++) {
            if (bank[r] == bank[r - 1] && r != n_verify) {
                if (pos[r] == pos[r - 1] + 1) continue;
                fprintf(stderr, "pulsar: %s: prefill row %u (bank %d pos %d) does not follow its run -- refusing\n",
                        PULSAR_QWEN_ARCH, r, bank[r], pos[r]);
                return false;
            }
            bool seen = false;
            for (uint32_t k = 0; k < n_runs; k++) seen |= bank[run_first[k]] == bank[r];
            if (r >= n_verify) n_prompt++;
            if (seen || n_prompt > PULSAR_FUSED_PF_MAX) {
                fprintf(stderr, "pulsar: %s: prefill row %u (bank %d) starts a run %s -- refusing\n", PULSAR_QWEN_ARCH,
                        r, bank[r], seen ? "of a bank the step already carries" : "past the step's run bound");
                return false;
            }
            run_first[n_runs++] = r;
        }
        run_first[n_runs] = n_rows;
    }
    st.n_runs = n_runs;
    st.run_first = n_runs ? run_first : NULL;
    const uint32_t n_vruns = st.verify ? pulsar_qwen_step_verify_runs(&st) : 0u;
    if (st.verify) {
        /* the verify's runs, and each bank's n-gram context before the step: a rollback finds its bank's run */
        pulsar_qwen_spec_capture &sp = s->qwen->spec;
        const uint32_t nc = g_qwen_shape.ngram_size - 1u;
        sp.n_runs = n_vruns;
        for (uint32_t k = 0; k <= n_vruns; k++) sp.run_first[k] = run_first[k];
        for (uint32_t k = 0; k < n_vruns; k++) {
            sp.run_bank[k] = bank[run_first[k]];
            sp.run_pos0[k] = pos[run_first[k]];
            for (uint32_t i = 0; i < nc; i++)
                sp.ngram_before[k][i] = s->qwen->ngram_ctx[(size_t)sp.run_bank[k] * nc + i];
        }
    }
    bool ok = pulsar_gpu_tensor_write(s->qwen->row_pos, 0, pos, (uint64_t)n_rows * sizeof(int32_t)) != 0 &&
              pulsar_gpu_tensor_write(s->qwen->row_bank, 0, bank, (uint64_t)n_rows * sizeof(int32_t)) != 0;
    if (ok && n_vruns > 1) {   /* the GDN's ragged verify runs */
        int32_t rf[PULSAR_QWEN_SPEC_ROWS + 1];
        for (uint32_t k = 0; k <= n_vruns; k++) rf[k] = (int32_t)run_first[k];
        ok = pulsar_gpu_tensor_write(s->qwen->run_first_dev, 0, rf, (uint64_t)(n_vruns + 1u) * sizeof(int32_t)) != 0;
    }
    if (ok) ok = pulsar_gpu_begin_commands() != 0;
    if (ok) ok = ops->embed(&st);
    /* L268: a sync's prefill chunk takes the image rows of the blocks it owns over their pads' embeddings (HF
     * masked_scatter), each into all 4 streams (the hyper-connection expansion follows the substitution) */
    if (ok && mode == PULSAR_QWEN_STEP_PREFILL && s->sync_images && n_runs == 1 && bank[0] == (int32_t)s->qwen->live_bank)
        ok = pulsar_image_merge_chunk(e, s->sync_prompt->v, s->sync_prompt->len, s->sync_images, s->sync_n_images,
                                      (uint32_t)pos[0], n_rows, qwen_write_image_rows, s->qwen->streams);
    for (uint32_t il = 0; ok && il < e->plan.n_layer; il++) {
        if (il == g_qwen_shape.ple_layer) ok = ops->ple(&st, il);
        if (ok) ok = ops->gr_read(&st, il, PULSAR_QWEN_GR_ATTN);
        if (ok) ok = e->plan.kind[il] == PULSAR_LAYER_QWEN_GDN ? ops->gdn(&st, il) : ops->qsa(&st, il);
        if (ok) ok = qwen_tp_allreduce_y(s, il, n_rows, "mixer");
        if (ok) ok = ops->gr_write(&st, il, PULSAR_QWEN_GR_ATTN);
        if (ok) ok = ops->gr_read(&st, il, PULSAR_QWEN_GR_MLP);
        if (ok) ok = ops->moe(&st, il);
        if (ok) ok = qwen_tp_allreduce_y(s, il, n_rows, "moe");
        if (ok) ok = ops->gr_write(&st, il, PULSAR_QWEN_GR_MLP);
        /* Diagnostic tap: the HC streams after layer il -- `n_hc` x n_embd f32 per row,
         * the same quantity the streamed reference keeps in `hs` ([seq, hc_count,
         * n_embd]).  It is how a reference-gate miss gets attributed to a layer instead
         * of to "the model".  The dump helpers re-check the arming themselves
         * (PULSAR_CUDA_GRAPH_DUMP_PREFIX, plus _LAYER/_NAME/_POS), so an unarmed run
         * does no work and allocates nothing. */
        if (ok && gpu_graph_debug_dump_enabled()) {
            /* The WHOLE step's streams ([n_rows][n_hc][n_embd], bf16 -> widened f32).  The
             * last row alone is enough to compare stream error, but the layer-correctness
             * test needs every position: it feeds the engine's stream at layer L into the
             * REFERENCE's layer L+1 and compares with the engine's L+1, which isolates a
             * layer's own arithmetic from the error it inherited.  Volume is the caller's
             * problem -- filter with PULSAR_CUDA_GRAPH_DUMP_LAYER/_POS, since the dump
             * helper synchronizes mid-graph. */
            gpu_graph_debug_dump_hc_tensor("qwen_h", s->qwen->streams,
                                          (uint64_t)n_rows * pulsar_qwen_hc_dim(&g_qwen_shape),
                                          il, (uint32_t)pos[n_rows - 1]);
        }
    }
    /* the head list into its logits rows (L284 #2): a span of consecutive decode rows is one call (a verify heads
     * every row), a prompt row is its own call -- the one-row head its classic chunk makes */
    for (uint32_t i = 0, j; ok && i < heads.n; i = j) {
        for (j = i + 1; j < heads.n && heads.row[i] < st.n_dec && heads.row[j] == heads.row[j - 1] + 1u &&
                        heads.row[j] < st.n_dec; j++) {}
        ok = ops->head(&st, heads.row[i], j - i, i);
    }
    if (ok) ok = pulsar_gpu_end_commands() != 0;
    else (void)pulsar_gpu_synchronize();
    /* S4's step end: settles a PLE gather a failed step left in flight and, on a
     * completed step, reads the MoE non-finite flag (rule 9: a NaN names its layer) */
    ok = pulsar_qwen_s4_step_end(&st, ok);
    if (!ok) {
        /* the step may have advanced any of its banks' state (n-gram context, conv,
         * recurrent) before it failed: those banks start over, and bank 0's
         * checkpoint no longer describes its state */
        for (uint32_t r = 0; r < n_rows; r++)
            if (!qwen_state_reset_bank(s->qwen, &g_qwen_shape, &e->plan, (uint32_t)bank[r]))
                fprintf(stderr, "pulsar: %s: could not clear bank %d after a failed step\n", PULSAR_QWEN_ARCH, bank[r]);
        s->checkpoint_valid = false;
    }
    if (ok && heads.n)
        ok = pulsar_gpu_tensor_read(s->qwen->logits, 0, logits_out,
                                    (uint64_t)heads.n * g_qwen_shape.n_vocab * sizeof(float)) != 0;
    return ok;
}

/* ---- L251 MTP: the drafter's forward and its lockstep with the trunk ----------------------------
 * (l251/docs/MTP-SPEC-2026-09-29.md.)  An MTP row at position p reads the trunk's pre-mixer stack at
 * p and the token x_{p+1}; it writes the MTP layer's KV at p and predicts x_{p+2}.  Every trunk row
 * gets its MTP row once its next token exists: in a prompt chunk that is the next prompt token, and
 * the chunk's LAST row waits in mtp_pend (per bank) for the token the next step feeds. */

/* The MTP layer over rows [0, n_rows) of mtp_h (trunk stacks), tokens = x_{p+1} per row; heads rows
 * [head_row0, head_row0 + head_n) into logits_out.  Its failure leaves the trunk untouched; the MTP
 * rows' banks lose their pending row (their MTP cache has a hole the drafter must not read past). */
bool qwen_mtp_forward(pulsar_session *s, pulsar_qwen_step_mode mode, const int32_t *tokens,
                      const int32_t *pos, const int32_t *bank, uint32_t n_rows,
                      uint32_t head_row0, uint32_t head_n, float *logits_out) {
    pulsar_engine *e = s->engine;
    const pulsar_qwen_ops *ops = &g_qwen_ops;
    const uint32_t il = e->plan.n_layer;                /* the MTP layer's slot */
    pulsar_qwen_step st{};
    st.model = &e->model;
    st.shape = &g_qwen_shape;
    st.w = e->qwen_weights;
    st.plan = &e->plan;
    st.st = s->qwen;
    st.mode = mode;
    st.n_rows = n_rows;
    st.tokens = tokens;
    st.pos = pos;
    st.bank = bank;
    std::vector<uint32_t> rope;   /* L268: the MTP layer ropes as the trunk does at the same positions */
    st.rope = qwen_rope_table(s, pos, bank, n_rows, &rope);
    st.streams = s->qwen->mtp_streams;
    st.mixer = &e->qwen_weights->mtp.mixer;
    st.draft_head = true;                               /* the drafter's head: n_draft logits a row */
    st.n_dec = pulsar_qwen_step_n_dec(mode, 0u, n_rows);
    bool ok = pulsar_gpu_tensor_write(s->qwen->row_pos, 0, pos, (uint64_t)n_rows * sizeof(int32_t)) != 0 &&
              pulsar_gpu_tensor_write(s->qwen->row_bank, 0, bank, (uint64_t)n_rows * sizeof(int32_t)) != 0;
    if (ok) ok = pulsar_gpu_begin_commands() != 0;
    if (ok) ok = pulsar_qwen_s4_mtp_combine(&st, pulsar_gpu_tensor_device_ptr(s->qwen->mtp_h));
    if (ok) ok = ops->gr_read(&st, il, PULSAR_QWEN_GR_ATTN);
    if (ok) ok = ops->qsa(&st, il);
    if (ok) ok = qwen_tp_allreduce_y(s, il, n_rows, "MTP mixer");
    if (ok) ok = ops->gr_write(&st, il, PULSAR_QWEN_GR_ATTN);
    if (ok) ok = ops->gr_read(&st, il, PULSAR_QWEN_GR_MLP);
    if (ok) ok = ops->moe(&st, il);
    if (ok) ok = qwen_tp_allreduce_y(s, il, n_rows, "MTP moe");
    if (ok) ok = ops->gr_write(&st, il, PULSAR_QWEN_GR_MLP);
    if (ok && head_n) ok = ops->head(&st, head_row0, head_n, 0);
    if (ok) ok = pulsar_gpu_end_commands() != 0;
    else (void)pulsar_gpu_synchronize();
    ok = pulsar_qwen_s4_step_end(&st, ok);
    if (!ok) {
        fprintf(stderr, "pulsar: %s: the MTP step failed; its banks' drafter state is dropped\n", PULSAR_QWEN_ARCH);
        for (uint32_t r = 0; r < n_rows; r++) s->qwen->mtp_pend_pos[bank[r]] = UINT32_MAX;
        return false;
    }
    if (head_n)
        ok = pulsar_gpu_tensor_read(s->qwen->logits, 0, logits_out,
                                    (uint64_t)head_n * e->qwen_weights->n_draft * sizeof(float)) != 0;
    return ok;
}

/* The drafter's greedy token from one MTP head row (n_draft logits); *prob (optional) is its softmax
 * probability over the draft head -- the confidence the draft schedule reads. */
int32_t qwen_mtp_argmax(const pulsar_engine *e, const float *row, float *prob) {
    const uint32_t n = e->qwen_weights->n_draft;
    uint32_t a = 0;
    for (uint32_t i = 1; i < n; i++) if (row[i] > row[a]) a = i;
    if (prob) {
        double z = 0.0;
        const float m = row[a];
        for (uint32_t i = 0; i < n; i++) z += exp((double)(row[i] - m));
        *prob = (float)(1.0 / z);
    }
    return e->qwen_weights->draft_ids[a];
}

/* After a trunk step over rows (tokens, pos, bank) -- step rows [row0, row0 + n), the trunk's streams there (L284
 * #2: a fused step's prompt run sits behind its verify rows) -- give every row whose next token is now known its
 * MTP row, and park each bank's last row in mtp_pend.  PREFILL: one bank, consecutive positions --
 * the bank's pending row (position P - 1, if any) pairs with tokens[0], row i with tokens[i + 1], and
 * the last row is parked.  DECODE: one row per bank -- each bank's pending row pairs with the token
 * this step fed, and this step's row is parked.  With head_n = 1 and a DECODE step of one row, the
 * MTP row is headed into mtp_logits: the draft for the token after the one the trunk just predicted
 * is NOT this -- it is the prediction of the SAME next token (x_{p+1}), made from the stack at p - 1. */
static bool qwen_mtp_absorb(pulsar_session *s, pulsar_qwen_step_mode mode, const int32_t *tokens,
                            const int32_t *pos, const int32_t *bank, uint32_t n, uint32_t row0, uint32_t head_n,
                            float *mtp_logits) {
    pulsar_qwen_state *q = s->qwen;
    const uint64_t hc = pulsar_qwen_hc_dim(&g_qwen_shape) * PULSAR_QWEN_STREAM_ELT_SIZE;
    int32_t *mt = (int32_t *)xmalloc((size_t)(n + 1) * sizeof(int32_t));
    int32_t *mp = (int32_t *)xmalloc((size_t)(n + 1) * sizeof(int32_t));
    int32_t *mb = (int32_t *)xmalloc((size_t)(n + 1) * sizeof(int32_t));
    uint32_t k = 0;
    bool ok = true;
    if (mode == PULSAR_QWEN_STEP_PREFILL) {
        const uint32_t b = (uint32_t)bank[0];
        if (q->mtp_pend_pos[b] != UINT32_MAX && q->mtp_pend_pos[b] + 1u == (uint32_t)pos[0]) {
            ok = pulsar_gpu_tensor_copy_async(q->mtp_h, 0, q->mtp_pend, (uint64_t)b * hc, hc) != 0;
            mt[k] = tokens[0]; mp[k] = pos[0] - 1; mb[k] = (int32_t)b; k++;
        }
        if (ok && n > 1) ok = pulsar_gpu_tensor_copy_async(q->mtp_h, (uint64_t)k * hc, q->streams, (uint64_t)row0 * hc,
                                                          (uint64_t)(n - 1u) * hc) != 0;
        for (uint32_t i = 0; i + 1u < n; i++) { mt[k] = tokens[i + 1]; mp[k] = pos[i]; mb[k] = (int32_t)b; k++; }
        if (ok) ok = pulsar_gpu_tensor_copy_async(q->mtp_pend, (uint64_t)b * hc, q->streams,
                                                  (uint64_t)(row0 + n - 1u) * hc, hc) != 0;
        if (ok) q->mtp_pend_pos[b] = (uint32_t)pos[n - 1];
    } else {
        for (uint32_t j = 0; ok && j < n; j++) {
            const uint32_t b = (uint32_t)bank[j];
            if (q->mtp_pend_pos[b] != UINT32_MAX && q->mtp_pend_pos[b] + 1u == (uint32_t)pos[j]) {
                ok = pulsar_gpu_tensor_copy_async(q->mtp_h, (uint64_t)k * hc, q->mtp_pend, (uint64_t)b * hc, hc) != 0;
                mt[k] = tokens[j]; mp[k] = pos[j] - 1; mb[k] = (int32_t)b; k++;
            }
            if (ok) ok = pulsar_gpu_tensor_copy_async(q->mtp_pend, (uint64_t)b * hc, q->streams,
                                                      (uint64_t)(row0 + j) * hc, hc) != 0;
            if (ok) q->mtp_pend_pos[b] = (uint32_t)pos[j];
        }
    }
    if (ok && k > 0) {
        const uint32_t hn = head_n && k == n && mode == PULSAR_QWEN_STEP_DECODE ? head_n : 0;
        ok = qwen_mtp_forward(s, mode, mt, mp, mb, k, 0, hn, mtp_logits);
    }
    free(mt);
    free(mp);
    free(mb);
    return ok;
}

/* One prefill chunk of the live bank (pulsar_prefill_chunk_fn): rows [pos0, pos0 + rows) of the
 * prompt through the forward (the MTP layer absorbs them too), the bank's position advanced, and --
 * while the prefill CONTINUES the bank's prefill-only history (L266 step 5) -- that history extended. */
struct qwen_prefill_ctx {
    int32_t *pos, *bank;
    uint32_t live;
    bool canonical;
};

static bool qwen_prefill_chunk(pulsar_session *s, const pulsar_tokens *prompt, uint32_t pos0, uint32_t rows,
                               bool last, void *ud) {
    qwen_prefill_ctx *c = (qwen_prefill_ctx *)ud;
    for (uint32_t r = 0; r < rows; r++) c->pos[r] = (int32_t)(pos0 + r);
    bool ok = qwen_forward(s, PULSAR_QWEN_STEP_PREFILL, prompt->v + pos0, c->pos, c->bank, rows,
                           qwen_heads_span(rows - 1u, last ? 1u : 0u), s->logits);
    if (ok && s->qwen->mtp)
        ok = qwen_mtp_absorb(s, PULSAR_QWEN_STEP_PREFILL, prompt->v + pos0, c->pos, c->bank, rows, 0, 0, NULL);
    if (ok) qwen_bank_set_pos(s->qwen, c->live, pos0 + rows);
    if (ok && c->canonical) s->qwen->prefill_pos[c->live] = pos0 + rows;
    return ok;
}

/* Prefill the live bank from `start` through the core loop (prefill_loop.cpp, L272 P2): prefill_cap
 * chunks, progress, and the cancel hook at every chunk boundary.  When the prefill continues the bank's
 * prefill-only history, the chunk that crosses the prompt's last grid point is cut there and the state
 * captured -- the checkpoint the next divergent turn resumes from.  A cut changes no byte: every prompt
 * chunk takes the prefill arms (session_contract_gate C1), which is also why an interrupted sync
 * resumes exactly.  Returns the loop's 0 / PULSAR_SESSION_SYNC_INTERRUPTED / 1. */
static int qwen_prefill(pulsar_session *s, const pulsar_tokens *prompt, uint32_t start) {
    qwen_prefill_ctx c;
    c.live = s->qwen->live_bank;
    c.canonical = s->qwen->prefill_pos[c.live] == start && !s->qwen->frontier_stale[c.live];
    pulsar_ckpt_store *ck = s->qwen->ckpt;
    const uint32_t grid_end = pulsar_ckpt_grid_floor(ck, (uint32_t)prompt->len);
    const uint32_t capture_at = c.canonical && grid_end > start ? grid_end : 0u;
    c.pos = (int32_t *)xmalloc((size_t)s->prefill_cap * sizeof(int32_t));
    c.bank = (int32_t *)xmalloc((size_t)s->prefill_cap * sizeof(int32_t));
    for (uint32_t r = 0; r < s->prefill_cap; r++) c.bank[r] = (int32_t)c.live;
    const int rc = pulsar_prefill_loop(s, prompt, start, s->prefill_cap, capture_at, ck, c.live, qwen_prefill_chunk, &c);
    free(c.pos);
    free(c.bank);
    return rc;
}

/* The family's sync is the core's default (sync_driver.cpp, L272 P2) over three ops: the bank's
 * position is the state's authority, a reset clears the bank's recurrent and attention state, and the
 * prefill is qwen_prefill (the core loop, with Qwen's capture on its prefill-only history). */
static bool qwen_sync_state_agrees(pulsar_session *s) {
    return s->qwen->bank_pos[s->qwen->live_bank] == (uint32_t)s->checkpoint.len;
}

static bool qwen_sync_reset_bank(pulsar_session *s) {
    return qwen_state_reset_bank(s->qwen, &g_qwen_shape, &s->engine->plan, s->qwen->live_bank);
}

static const pulsar_sync_ops k_qwen_sync = {
    /* .name         = */ PULSAR_QWEN_ARCH,
    /* .state_agrees = */ qwen_sync_state_agrees,
    /* .reset_bank   = */ qwen_sync_reset_bank,
    /* .prefill      = */ qwen_prefill,
};

static int qwen_session_sync(pulsar_session *s, const pulsar_tokens *prompt,
                             const pulsar_image_ref *images, int n_images, char *err, size_t errlen) {
    return pulsar_session_sync_default(s, prompt, images, n_images, &k_qwen_sync, err, errlen);
}

static int qwen_session_eval(pulsar_session *s, int token, char *err, size_t errlen) {
    if (!s->checkpoint_valid || s->checkpoint.len >= s->ctx_size) {
        if (err) snprintf(err, errlen, "%s: eval needs a synced session with room left", PULSAR_QWEN_ARCH);
        return 1;
    }
    if (!pulsar_session_token_is_id(s, token, err, errlen)) return 1;   /* L188 (L272 B2) */
    const uint32_t live = s->qwen->live_bank;
    if (s->qwen->bank_pos[live] != (uint32_t)s->checkpoint.len) {
        if (err) snprintf(err, errlen, "%s: eval: bank %u holds %u tokens, the checkpoint %d", PULSAR_QWEN_ARCH, live,
                          s->qwen->bank_pos[live], s->checkpoint.len);
        return 1;
    }
    const int32_t tok = token, pos = s->checkpoint.len, bank = (int32_t)live;
    if (!qwen_forward(s, PULSAR_QWEN_STEP_DECODE, &tok, &pos, &bank, 1, qwen_heads_span(0, 1), s->logits)) {
        if (err) snprintf(err, errlen, "%s: decode refused (see the log for the op)", PULSAR_QWEN_ARCH);
        return 1;
    }
    if (s->qwen->mtp) {
        /* DIAGNOSTIC (PULSAR_QWEN_MTP_PROBE set): head the MTP row this step absorbs -- (stack at p - 1,
         * x_p), a prediction of x_{p+1} -- and count how often its argmax is the trunk's own argmax for
         * x_{p+1}: draft-position-1 acceptance under greedy decoding.  Printed at session destroy. */
        static const bool probe = getenv("PULSAR_QWEN_MTP_PROBE") && getenv("PULSAR_QWEN_MTP_PROBE")[0];
        float *ml = probe ? (float *)xmalloc((size_t)g_qwen_shape.n_vocab * sizeof(float)) : NULL;
        const bool had = s->qwen->mtp_pend_pos[live] != UINT32_MAX && s->qwen->mtp_pend_pos[live] + 1u == (uint32_t)pos;
        const bool ok = qwen_mtp_absorb(s, PULSAR_QWEN_STEP_DECODE, &tok, &pos, &bank, 1, 0, probe ? 1u : 0u, ml);
        if (ok && probe && had) {
            uint32_t at = 0;
            for (uint32_t v = 1; v < g_qwen_shape.n_vocab; v++) if (s->logits[v] > s->logits[at]) at = v;
            s->qwen->mtp_probe_n++;
            s->qwen->mtp_probe_hit += (uint32_t)qwen_mtp_argmax(s->engine, ml) == at;
        }
        free(ml);
        if (!ok) {
            if (err) snprintf(err, errlen, "%s: the MTP step refused (see the log)", PULSAR_QWEN_ARCH);
            return 1;
        }
    }
    token_vec_push(&s->checkpoint, token);
    qwen_bank_set_pos(s->qwen, live, (uint32_t)s->checkpoint.len);
    s->logits_stale = false;
    return 0;
}

static int qwen_session_decode_multiseq(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n,
                                        float *logits, int logits_cap, char *err, size_t errlen) {
    const uint32_t nv = g_qwen_shape.n_vocab;
    if (!reqs || n == 0 || n > PULSAR_QWEN_HEAD_ROWS_MAX || n > s->qwen->max_rows || !logits ||
        logits_cap < 0 || (uint64_t)logits_cap < (uint64_t)n * nv) {
        if (err) snprintf(err, errlen, "%s: batched decode of %u rows refused (bad args)", PULSAR_QWEN_ARCH, n);
        return 1;
    }
    int32_t tok[PULSAR_QWEN_HEAD_ROWS_MAX], pos[PULSAR_QWEN_HEAD_ROWS_MAX], bank[PULSAR_QWEN_HEAD_ROWS_MAX];
    for (uint32_t i = 0; i < n; i++) {
        /* One row per bank, at that bank's next position: a recurrent state
         * advances exactly one token per row. */
        if (reqs[i].bank >= s->qwen->n_banks || (uint32_t)reqs[i].pos != s->qwen->bank_pos[reqs[i].bank]) {
            if (err) snprintf(err, errlen, "%s: row %u (bank %u pos %d) is not its bank's next position",
                              PULSAR_QWEN_ARCH, i, reqs[i].bank, reqs[i].pos);
            return 1;
        }
        for (uint32_t j = 0; j < i; j++) {
            if (reqs[j].bank == reqs[i].bank) {
                if (err) snprintf(err, errlen, "%s: bank %u has two rows in one decode step", PULSAR_QWEN_ARCH,
                                  reqs[i].bank);
                return 1;
            }
        }
        tok[i] = reqs[i].token;
        pos[i] = reqs[i].pos;
        bank[i] = (int32_t)reqs[i].bank;
    }
    if (!qwen_forward(s, PULSAR_QWEN_STEP_DECODE, tok, pos, bank, n, qwen_heads_span(0, n), logits) ||
        (s->qwen->mtp && !qwen_mtp_absorb(s, PULSAR_QWEN_STEP_DECODE, tok, pos, bank, n, 0, 0, NULL))) {
        /* the failed step reset the banks it touched (qwen_forward): fatal for these rows */
        if (err) snprintf(err, errlen, "%s: decode refused (see the log for the op)", PULSAR_QWEN_ARCH);
        return -1;
    }
    for (uint32_t i = 0; i < n; i++) {
        qwen_bank_set_pos(s->qwen, reqs[i].bank, s->qwen->bank_pos[reqs[i].bank] + 1u);
        if (reqs[i].bank == s->qwen->live_bank) {   /* the live bank moved past the host view */
            s->checkpoint_valid = false;
            s->logits_stale = true;
        }
    }
    return 0;
}

/* A verify's rows reqs[0, n): one run a bank, each run starting at its bank's next position and each further row
 * following the one before -- the speculation lane's verify, alone (decode_mixed) or in front of a fused step's
 * prompt run.  Fills tok / pos / bk; false (nothing moved) names the first row that is not. */
static bool qwen_verify_rows(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n, int32_t *tok,
                             int32_t *pos, int32_t *bk, char *err, size_t errlen) {
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t b = reqs[i].bank;
        const bool run_start = i == 0 || reqs[i - 1].bank != b;
        const uint32_t want = run_start ? (b < s->qwen->n_banks ? s->qwen->bank_pos[b] : 0u) : (uint32_t)reqs[i - 1].pos + 1u;
        if (b >= s->qwen->n_banks || (uint32_t)reqs[i].pos != want) {
            if (err) snprintf(err, errlen, "%s: verify row %u (bank %u pos %d) is not its run's next position",
                              PULSAR_QWEN_ARCH, i, b, reqs[i].pos);
            return false;
        }
        tok[i] = reqs[i].token;
        pos[i] = reqs[i].pos;
        bk[i] = (int32_t)b;
    }
    return true;
}

/* The server's batched lane: its plain decode is this entry with one row per bank and no prefill
 * runs (max_head_runs 0), which is exactly the batched decode.  A step that carries a PREFILL run (a
 * bank with more than one row) is refused before anything moves: the family declares no
 * PULSAR_FAMILY_CAP_MIXED_PREFILL, so the server's mixed lane never sends one -- its prompts ride the
 * fused step (qwen_session_decode_fused) or the sync. */
static int qwen_session_decode_mixed(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows,
                                     float *logits, int logits_cap, uint32_t *out_n_rows,
                                     uint32_t max_head_runs, char *err, size_t errlen) {
    if (out_n_rows) *out_n_rows = 0;
    if (max_head_runs == PULSAR_MSEQ_HEAD_ALL_ROWS) {
        /* L272 P1 S3/S4: the speculation lane's verify -- one run per bank, each [p, p + k] at its bank's next
         * position, every row headed into the caller's block, the per-row state capture armed (the target's
         * commit rolls each bank's rejected rows back, pulsar_qwen_s4_spec_rollback).  At most SPEC_ROWS rows,
         * so every kernel keeps its decode-width arm. */
        const uint32_t nv = g_qwen_shape.n_vocab;
        if (!reqs || n_rows == 0 || n_rows > PULSAR_QWEN_SPEC_ROWS || !logits || logits_cap < 0 ||
            (uint64_t)logits_cap < (uint64_t)n_rows * nv || !s->qwen->mtp) {
            if (err) snprintf(err, errlen, "%s: a verify step of %u rows refused (bad args, over %u rows, or no MTP layer)",
                              PULSAR_QWEN_ARCH, n_rows, PULSAR_QWEN_SPEC_ROWS);
            return 1;
        }
        int32_t tok[PULSAR_QWEN_SPEC_ROWS], pos[PULSAR_QWEN_SPEC_ROWS], bk[PULSAR_QWEN_SPEC_ROWS];
        if (!qwen_verify_rows(s, reqs, n_rows, tok, pos, bk, err, errlen)) return 1;
        if (!qwen_forward(s, PULSAR_QWEN_STEP_PREFILL, tok, pos, bk, n_rows, qwen_heads_span(0, n_rows), logits,
                          n_rows)) {
            if (err) snprintf(err, errlen, "%s: the verify step refused (see the log for the op)", PULSAR_QWEN_ARCH);
            return -1;
        }
        if (out_n_rows) *out_n_rows = n_rows;
        return 0;
    }
    for (uint32_t i = 0; reqs && i < n_rows; i++)
        for (uint32_t j = 0; j < i; j++)
            if (reqs[j].bank == reqs[i].bank) {
                if (err) snprintf(err, errlen, "%s: a fused prefill run (bank %u, %u+ rows) is not implemented; "
                                  "prefill goes through sync", PULSAR_QWEN_ARCH, reqs[i].bank, 2u);
                return 1;
            }
    const int rc = qwen_session_decode_multiseq(s, reqs, n_rows, logits, logits_cap, err, errlen);
    if (rc == 0 && out_n_rows) *out_n_rows = n_rows;
    return rc;
}

/* L284 #2: the fused step (contract in pulsar.h) -- a verify's rows [0, n_dec) and the shape's n_pf prompt runs
 * [n_dec, n_rows) in one forward.  The verify rows are exactly the speculation lane's verify (decode_mixed with
 * every row headed): headed in place, their per-row state captured for the commit's rollback, their bank positions
 * left for the commit to move.  Each prompt run is exactly a sync's prefill chunk of the same rows on its bank, at
 * any width: a run from position 0 resets its bank first (as a cold sync does), the MTP layer absorbs it, the
 * bank's position moves past it, and so does its prefill-only history when the run continues it.  A run with
 * head_last set has its last row headed into the next logits row after the verify's, in run order.  The record
 * of each chunk in its bank's history is the core's pulsar_session_note_prefilled.  1 = refused before anything
 * moved, -1 = the step failed with state moved (its banks' state is fatal). */
static int qwen_session_decode_fused(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows,
                                     const pulsar_fused_shape *shape, float *logits, int logits_cap,
                                     uint32_t *out_n_rows, char *err, size_t errlen) {
    if (out_n_rows) *out_n_rows = 0;
    pulsar_engine *e = s->engine;
    pulsar_qwen_state *q = s->qwen;
    const uint32_t nv = g_qwen_shape.n_vocab;
    if (!reqs || !shape || !logits || n_rows == 0 || n_rows > q->max_rows) {
        if (err) snprintf(err, errlen, "%s: fused step: bad args (%u rows, at most %u)", PULSAR_QWEN_ARCH, n_rows,
                          q->max_rows);
        return 1;
    }
    const uint32_t n_dec = shape->n_dec, heads = pulsar_fused_shape_heads(shape);
    if (n_dec >= n_rows || shape->n_pf == 0 || shape->n_pf > PULSAR_FUSED_PF_MAX) {
        if (err) snprintf(err, errlen, "%s: a fused step is a verify and 1..%u prompt runs (n_dec %u of %u rows, %u "
                          "prompt runs)", PULSAR_QWEN_ARCH, PULSAR_FUSED_PF_MAX, n_dec, n_rows, shape->n_pf);
        return 1;
    }
    if (n_dec > PULSAR_QWEN_SPEC_ROWS || (n_dec > 0 && !q->mtp) ||
        n_dec + heads > e->family->session->fused_heads_max || logits_cap < 0 || (uint64_t)logits_cap < (uint64_t)(n_dec + heads) * nv) {
        if (err) snprintf(err, errlen, "%s: fused step: %u verify rows (at most %u, with the MTP layer) + %u headed "
                          "(at most %u rows headed), logits capacity %d", PULSAR_QWEN_ARCH, n_dec, PULSAR_QWEN_SPEC_ROWS,
                          heads, e->family->session->fused_heads_max, logits_cap);
        return 1;
    }
    std::vector<int32_t> tok(n_rows), pos(n_rows), bk(n_rows);
    if (!qwen_verify_rows(s, reqs, n_dec, tok.data(), pos.data(), bk.data(), err, errlen)) return 1;
    /* the prompt runs: each on a bank no other row of the step is on, consecutive positions, text rows */
    uint32_t first[PULSAR_FUSED_PF_MAX + 1];
    uint32_t n_pf = 0;
    const pulsar_family_vision *vis = e->family->vision;
    for (uint32_t i = n_dec; i < n_rows; i++) {
        const uint32_t b = reqs[i].bank;
        const bool starts = i == n_dec || b != reqs[i - 1].bank;
        bool taken = false;
        for (uint32_t j = 0; starts && j < i; j++) taken |= reqs[j].bank == b;
        if (starts && n_pf < PULSAR_FUSED_PF_MAX) first[n_pf] = i;
        n_pf += starts ? 1u : 0u;
        const char *why = n_pf > shape->n_pf                                       ? "starts a run past the shape's"
                          : taken                                                 ? "is on a bank the step already carries"
                          : !starts && reqs[i].pos != reqs[i - 1].pos + 1         ? "does not follow its run"
                          : !pulsar_session_token_is_id(s, reqs[i].token, NULL, 0) ? "is not a token id"
                          : vis && vis->is_sentinel(e, reqs[i].token)               ? "is an image block's (a sync's)"
                                                                                  : NULL;
        if (why) {
            if (err) snprintf(err, errlen, "%s: fused prompt row %u (bank %u pos %d) %s", PULSAR_QWEN_ARCH, i, b,
                              reqs[i].pos, why);
            return 1;
        }
        tok[i] = reqs[i].token;
        pos[i] = reqs[i].pos;
        bk[i] = (int32_t)b;
    }
    if (n_pf != shape->n_pf) {
        if (err) snprintf(err, errlen, "%s: fused step: %u prompt runs, the shape says %u", PULSAR_QWEN_ARCH, n_pf,
                          shape->n_pf);
        return 1;
    }
    first[n_pf] = n_rows;
    /* each run at its bank's next position or from 0 (a run from 0 must not inherit a dead conversation's image
     * records: its rope would read them) */
    for (uint32_t r = 0; r < n_pf; r++) {
        const uint32_t b = reqs[first[r]].bank, m = first[r + 1] - first[r];
        const int32_t p0 = reqs[first[r]].pos;
        const pulsar_image_identity *img = b < q->n_banks ? pulsar_session_bank_images(s, b, q->live_bank) : NULL;
        const char *why = b >= q->n_banks                            ? "is outside the pool"
                          : p0 != 0 && (uint32_t)p0 != q->bank_pos[b] ? "holds a different position"
                          : (uint64_t)p0 + m > q->ctx                 ? "has no room"
                          : p0 == 0 && img && img->n                  ? "still holds image blocks (invalidate it first)"
                                                                      : NULL;
        if (why) {
            if (err) snprintf(err, errlen, "%s: fused prompt run %u on bank %u at %d + %u rows: the bank %s",
                              PULSAR_QWEN_ARCH, r, b, p0, m, why);
            return 1;
        }
    }
    /* a run from 0 starts its bank over, as a cold sync does; whether each run continues its bank's prefill-only
     * history is read before the step moves it */
    bool canonical[PULSAR_FUSED_PF_MAX];
    for (uint32_t r = 0; r < n_pf; r++) {
        const uint32_t b = reqs[first[r]].bank;
        if (reqs[first[r]].pos == 0 && !qwen_state_reset_bank(q, &g_qwen_shape, &e->plan, b)) {
            if (err) snprintf(err, errlen, "%s: fused step: bank %u could not be reset", PULSAR_QWEN_ARCH, b);
            return -1;
        }
        canonical[r] = q->prefill_pos[b] == (uint32_t)reqs[first[r]].pos && !q->frontier_stale[b];
    }
    qwen_heads hl = qwen_heads_span(0, n_dec);
    for (uint32_t r = 0; r < n_pf; r++)
        if (shape->head_last[r]) hl.row[hl.n++] = first[r + 1] - 1u;
    bool ok = qwen_forward(s, PULSAR_QWEN_STEP_PREFILL, tok.data(), pos.data(), bk.data(), n_rows, hl, logits, n_dec);
    for (uint32_t r = 0; ok && q->mtp && r < n_pf; r++)   /* each run's MTP rows, as its sync chunk's absorb */
        ok = qwen_mtp_absorb(s, PULSAR_QWEN_STEP_PREFILL, tok.data() + first[r], pos.data() + first[r],
                             bk.data() + first[r], first[r + 1] - first[r], first[r], 0, NULL);
    if (!ok) {
        if (err) snprintf(err, errlen, "%s: the fused step refused (see the log for the op)", PULSAR_QWEN_ARCH);
        return -1;
    }
    for (uint32_t r = 0; r < n_pf; r++) {
        const uint32_t b = reqs[first[r]].bank, end = (uint32_t)reqs[first[r]].pos + (first[r + 1] - first[r]);
        qwen_bank_set_pos(q, b, end);
        if (canonical[r]) q->prefill_pos[b] = end;
        if (b == q->live_bank) {   /* the live bank moved past the host view */
            s->checkpoint_valid = false;
            s->logits_stale = true;
        }
    }
    if (out_n_rows) *out_n_rows = hl.n;
    return 0;
}

/* The HOST view forgets the live bank's history; its device state is left alone.  The next sync
 * of that bank finds the checkpoint invalid and resets the bank before it prefills, so a stale
 * state is never decoded -- and the server's invalidate (a bank provision, an eviction, a stop
 * string mid-batch) cannot wipe OTHER banks' conversations, which resetting every bank did. */
static void qwen_session_invalidate(pulsar_session *s) {
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->logits_stale = true;
    s->live_images.n = 0;   /* the forgotten history's image records go with it (a fused run from 0 reads them) */
}

uint32_t qwen_argmax(const float *v, uint32_t n) {
    uint32_t a = 0;
    for (uint32_t i = 1; i < n; i++) if (v[i] > v[a]) a = i;
    return a;
}

/* L270: the MTP's sampled distribution q over its draft vocabulary, ids mapped to the vocabulary. */
bool qwen_mtp_dist(pulsar_session *s, const float *row, float temperature, int top_k, float top_p, float min_p,
                   pulsar_sample_dist *q) {
    const pulsar_qwen_weights *w = s->engine->qwen_weights;
    if (!pulsar_sample_dist_build(row, w->n_draft, temperature, top_k, top_p, min_p, &s->sample_scratch, q)) return false;
    for (uint32_t i = 0; i < q->n; i++) q->ids[i] = (int)w->draft_ids[q->ids[i]];
    return true;
}

static const pulsar_family_session_ops k_qwen_session_ops = {
    /* .create          = */ qwen_session_create,
    /* .destroy         = */ qwen_session_destroy,
    /* .cost_bytes      = */ qwen_session_cost_bytes,
    /* .sync            = */ qwen_session_sync,
    /* .eval            = */ qwen_session_eval,
    /* .decode_multiseq = */ qwen_session_decode_multiseq,
    /* .decode_mixed    = */ qwen_session_decode_mixed,
    /* .invalidate      = */ qwen_session_invalidate,
    /* .decode_fused    = */ qwen_session_decode_fused,
    /* .fused_heads_max = */ PULSAR_QWEN_HEAD_ROWS_MAX,   /* the logits slab's rows */
};

/* The trunk's layers and the MTP layer each take an exchange slot (L266 step 7). */
static void qwen_tp_shape(const pulsar_engine *e, uint32_t *n_layer, uint32_t *n_embd, uint32_t *n_vocab) {
    *n_layer = e->plan.n_layer + (e->qwen_weights && e->qwen_weights->mtp.present ? 1u : 0u);
    *n_embd = g_qwen_shape.n_embd;
    *n_vocab = g_qwen_shape.n_vocab;
}

const pulsar_family PULSAR_FAMILY_QWEN4_EXP = {
    /* .id           = */ PULSAR_FAMILY_ID_QWEN4_EXP,
    /* .arch         = */ PULSAR_QWEN_ARCH,
    /* .name         = */ "Qwen4-exp",
    /* .caps         = */ PULSAR_FAMILY_CAP_BANKS | PULSAR_FAMILY_CAP_SEGMENTS | PULSAR_FAMILY_CAP_PAYLOAD | PULSAR_FAMILY_CAP_TP |
                          PULSAR_FAMILY_CAP_SPEC | PULSAR_FAMILY_CAP_CHAT | PULSAR_FAMILY_CAP_VISION,
    /* .load         = */ qwen_family_load,
    /* .after_gpu    = */ qwen_family_after_gpu,
    /* .logits_width = */ qwen_logits_width,
    /* .model_name   = */ qwen_model_name,
    /* .served_id    = */ qwen_served_model_id,
    /* .chat_format  = */ qwen_chat_format,
    /* .model_id     = */ qwen_model_id,
    /* .trained_ctx  = */ qwen_trained_context,
    /* .tp_shape     = */ qwen_tp_shape,
    /* .drafter      = */ qwen_drafter,
    /* .quant_bits   = */ qwen_quant_bits,
    /* .spec         = */ &k_qwen_spec_target,
    /* .session      = */ &k_qwen_session_ops,
    /* .tokenizer    = */ &k_qwen_tokenizer,
    /* .tp_slices    = */ pulsar_qwen_tp_slices,
    /* .act_kind     = */ PULSAR_ACT_KIND_ROWS,
    /* .vision       = */ &PULSAR_QWEN_IMAGE_FRONT,   /* L268: vision_qwen.cpp */
    /* .banks        = */ &k_qwen_bank_ops,
};

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
#include "lib/qwen_tokenizer.h"

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
    pulsar_tensor *t = model_find_tensor(m, name);
    if (!t) {
        fprintf(stderr, "pulsar: %s: required tensor %s is missing\n", PULSAR_QWEN_ARCH, name);
        *ok = false;
        return NULL;
    }
    const uint64_t want[3] = {d0, d1, d2};
    bool dims_ok = t->ndim == nd;
    for (uint32_t i = 0; dims_ok && i < nd; i++) dims_ok = t->dim[i] == want[i];
    if (!dims_ok) {
        fprintf(stderr, "pulsar: %s: tensor %s has ne [", PULSAR_QWEN_ARCH, name);
        for (uint32_t i = 0; i < t->ndim; i++) fprintf(stderr, "%s%llu", i ? ", " : "", (unsigned long long)t->dim[i]);
        fprintf(stderr, "], want [");
        for (uint32_t i = 0; i < nd; i++) fprintf(stderr, "%s%llu", i ? ", " : "", (unsigned long long)want[i]);
        fprintf(stderr, "]\n");
        *ok = false;
    }
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
    const uint64_t E = s->n_embd, V = s->n_vocab;
    w->token_embd = qbind(m, &ok, QWEN_TEXT "embed_tokens.weight", 2, E, V);
    w->output = qbind(m, &ok, "lm_head.weight", 2, E, V);
    qwen_bind_gr(m, &ok, s, QWEN_TEXT "hyper_connection_mixer.", &w->mixer, false);
    for (uint32_t il = 0; il < plan->n_layer; il++) {
        char lp[128];
        snprintf(lp, sizeof(lp), QWEN_TEXT "layers.%u.", il);
        qwen_bind_layer(m, &ok, s, plan->kind[il], lp, il == s->ple_layer, false, &w->layer[il]);
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
    if (!qwen_bind_weights(&e->model, &g_qwen_shape, &e->plan, e->qwen_weights)) {
        fprintf(stderr, "pulsar: %s: the artifact does not bind -- refusing\n", PULSAR_QWEN_ARCH);
        return false;
    }
    if (!pulsar_qwen_s4_load(e, opt)) return false;
    if (!qwen_load_tokenizer(e, opt->model_path)) return false;
    fprintf(stderr, "pulsar: %s: %s, %u GDN + %u QSA layers, PLE at layer %u\n",
            PULSAR_QWEN_ARCH, g_qwen_shape.name,
            pulsar_layer_plan_count(&e->plan, PULSAR_LAYER_QWEN_GDN),
            pulsar_layer_plan_count(&e->plan, PULSAR_LAYER_QWEN_QSA), g_qwen_shape.ple_layer);
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
static const char *qwen_model_name(const pulsar_engine *) { return g_qwen_shape.name; }
static pulsar_chat_format qwen_chat_format(const pulsar_engine *) { return PULSAR_CHAT_QWEN; }
/* Disk-KV compatibility id: 0 and 1 are DeepSeek's two profiles. */
static int qwen_model_id(const pulsar_engine *) { return 2; }

/* ---- session state ------------------------------------------------------------ */

static uint64_t qwen_ceil_div(uint64_t a, uint64_t b) { return (a + b - 1u) / b; }

static void qwen_state_free(pulsar_qwen_state *st) {
    if (!st) return;
    for (uint32_t il = 0; il < PULSAR_FAMILY_MAX_LAYER; il++) {
        pulsar_qwen_layer_state *L = &st->layer[il];
        pulsar_gpu_tensor_free(L->gdn_state);
        pulsar_gpu_tensor_free(L->gdn_conv);
        pulsar_gpu_tensor_free(L->kv);
        pulsar_gpu_tensor_free(L->idx_keys);
        pulsar_gpu_tensor_free(L->idx_tail);
        pulsar_gpu_tensor_free(L->ple_conv);
    }
    pulsar_gpu_tensor_free(st->streams);
    pulsar_gpu_tensor_free(st->x);
    pulsar_gpu_tensor_free(st->y);
    pulsar_gpu_tensor_free(st->logits);
    pulsar_gpu_tensor_free(st->row_pos);
    pulsar_gpu_tensor_free(st->row_bank);
    for (int op = 0; op < PULSAR_QWEN_OP_COUNT; op++) pulsar_gpu_tensor_free(st->scratch[op]);
    free(st->ngram_ctx);
    free(st->bank_pos);
    free(st);
}

/* Allocate a session's state.  The same code prices it: under
 * pulsar_gpu_tensor_dry_begin the allocators total the bytes instead
 * (qwen_session_cost_bytes). */
static pulsar_qwen_state *qwen_state_alloc(const pulsar_qwen_shape *s, const pulsar_layer_plan *plan,
                                           uint32_t n_banks, uint32_t ctx, uint32_t max_rows) {
    pulsar_qwen_state *st = (pulsar_qwen_state *)xcalloc(1, sizeof(*st));
    st->n_banks = n_banks;
    st->ctx = pulsar_qwen_qsa_cap(s, ctx);   /* a whole number of indexer blocks; see the helper */
    st->max_rows = max_rows;
    bool ok = true;
    const uint64_t nb = n_banks;
    for (uint32_t il = 0; ok && il < plan->n_layer; il++) {
        pulsar_qwen_layer_state *L = &st->layer[il];
        if (plan->kind[il] == PULSAR_LAYER_QWEN_GDN) {
            L->gdn_state = pulsar_gpu_tensor_alloc(nb * pulsar_qwen_gdn_state_bytes(s));
            L->gdn_conv  = pulsar_gpu_tensor_alloc(nb * pulsar_qwen_gdn_conv_bytes(s));
            ok = L->gdn_state && L->gdn_conv;
        } else {
            /* Demand-paged: a bank pays for the KV it touches.  Sized from st->ctx, NOT the raw
             * parameter: st->ctx is the block-rounded QSA capacity (see pulsar_qwen_qsa_cap), and
             * the op views exactly st->ctx tokens per bank -- allocating the raw ctx made the KV
             * view overshoot its tensor and the op refused with "a QSA bank cache view failed". */
            L->kv       = pulsar_gpu_tensor_alloc_managed(nb * st->ctx * pulsar_qwen_kv_row_bytes(s));
            L->idx_keys = pulsar_gpu_tensor_alloc_managed(nb * qwen_ceil_div(st->ctx, s->idx_block) *
                                                          pulsar_qwen_index_row_bytes(s));
            L->idx_tail = pulsar_gpu_tensor_alloc(nb * pulsar_qwen_index_tail_bytes(s));
            ok = L->kv && L->idx_keys && L->idx_tail;
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
    for (int op = 0; ok && op < PULSAR_QWEN_OP_COUNT; op++) {
        const uint64_t b = g_qwen_ops.scratch_bytes
                               ? g_qwen_ops.scratch_bytes((pulsar_qwen_op_id)op, s, max_rows, ctx) : 0;
        if (b) ok = (st->scratch[op] = pulsar_gpu_tensor_alloc(b)) != NULL;
    }
    st->ngram_ctx = (int32_t *)xcalloc((size_t)n_banks * (s->ngram_size - 1u), sizeof(int32_t));
    st->bank_pos = (uint32_t *)xcalloc(n_banks, sizeof(uint32_t));
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
    for (uint32_t il = 0; ok && il < plan->n_layer; il++) {
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
    st->bank_pos[bank] = 0;
    return ok;
}

static uint32_t qwen_prefill_cap(const pulsar_engine *e, int ctx_size) {
    return pulsar_prefill_cap_for_prompt(ctx_size, e->prefill_chunk);
}

static int qwen_session_create(pulsar_session **out, pulsar_engine *e, int ctx_size) {
    if (!pulsar_backend_uses_graph(e->backend) || !e->gpu_ready) return 1;
    const uint32_t n_banks = gpu_graph_bank_pool_n();
    pulsar_session *s = (pulsar_session *)xcalloc(1, sizeof(*s));
    s->engine = e;
    s->ctx_size = ctx_size;
    s->prefill_cap = qwen_prefill_cap(e, ctx_size);
    const uint64_t alloc_before = pulsar_gpu_tensor_alloc_bytes_current();
    s->qwen = qwen_state_alloc(&g_qwen_shape, &e->plan, n_banks, (uint32_t)ctx_size, s->prefill_cap);
    if (!s->qwen) {
        free(s);
        return 1;
    }
    for (uint32_t b = 0; b < n_banks; b++) {
        if (!qwen_state_reset_bank(s->qwen, &g_qwen_shape, &e->plan, b)) {
            fprintf(stderr, "pulsar: %s: could not clear bank %u's state\n", PULSAR_QWEN_ARCH, b);
            qwen_state_free(s->qwen);
            free(s);
            return 1;
        }
    }
    s->logits = (float *)xmalloc((size_t)g_qwen_shape.n_vocab * sizeof(s->logits[0]));
    s->qwen->carry = (pulsar_qwen_bank_carry *)xcalloc(n_banks, sizeof(pulsar_qwen_bank_carry));
    s->resident_bytes = pulsar_gpu_tensor_alloc_bytes_current() - alloc_before;
    fprintf(stderr, "pulsar: %s session: %u bank(s) x %d tokens, %u-row steps, %.2f GiB of state "
                    "(%.1f MiB fixed per bank + %.1f KiB per token)\n",
            PULSAR_QWEN_ARCH, n_banks, ctx_size, s->prefill_cap,
            (double)s->resident_bytes / 1073741824.0,
            (double)(pulsar_layer_plan_count(&e->plan, PULSAR_LAYER_QWEN_GDN) *
                     (pulsar_qwen_gdn_state_bytes(&g_qwen_shape) + pulsar_qwen_gdn_conv_bytes(&g_qwen_shape)) +
                     pulsar_qwen_ple_conv_bytes(&g_qwen_shape)) / 1048576.0,
            (double)(pulsar_layer_plan_count(&e->plan, PULSAR_LAYER_QWEN_QSA) *
                     (pulsar_qwen_kv_row_bytes(&g_qwen_shape) +
                      pulsar_qwen_index_row_bytes(&g_qwen_shape) / g_qwen_shape.idx_block)) / 1024.0);
    *out = s;
    return 0;
}

static void qwen_session_destroy(pulsar_session *s) {
    if (s->qwen && s->qwen->carry) {
        for (uint32_t b = 0; b < s->qwen->n_banks; b++) {
            token_vec_free(&s->qwen->carry[b].checkpoint);
            free(s->qwen->carry[b].logits);
        }
        free(s->qwen->carry);
        s->qwen->carry = NULL;
    }
    qwen_state_free(s->qwen);
    token_vec_free(&s->checkpoint);
    pulsar_sample_scratch_free(&s->sample_scratch);
    free(s->logits);
    free(s);
}

uint64_t pulsar_qwen_state_price(const pulsar_qwen_shape *s, const pulsar_layer_plan *plan,
                                 uint32_t n_banks, uint32_t ctx, uint32_t max_rows,
                                 uint64_t *managed_bytes) {
    pulsar_gpu_tensor_dry_begin();
    pulsar_qwen_state *st = qwen_state_alloc(s, plan, n_banks, ctx, max_rows);
    uint64_t bytes = 0, managed = 0;
    pulsar_gpu_tensor_dry_end(&bytes, &managed);
    const bool ok = st != NULL;
    qwen_state_free(st);
    if (managed_bytes) *managed_bytes = ok ? managed : 0;
    return ok ? bytes : 0;
}

static uint64_t qwen_session_cost_bytes(pulsar_engine *e, int ctx_size, int n_banks) {
    if (!pulsar_backend_uses_graph(e->backend) || !e->gpu_ready) return 0;
    return pulsar_qwen_state_price(&g_qwen_shape, &e->plan, (uint32_t)n_banks, (uint32_t)ctx_size,
                                   qwen_prefill_cap(e, ctx_size), NULL);
}

/* ---- the step driver ------------------------------------------------------------ */

/* One step: the plan, in the forward's order (family_qwen.h).  Refuses by
 * name before touching any state when an op the step needs is missing. */
static bool qwen_forward(pulsar_session *s, pulsar_qwen_step_mode mode, const int32_t *tokens,
                         const int32_t *pos, const int32_t *bank, uint32_t n_rows,
                         uint32_t head_row0, uint32_t head_n, float *logits_out) {
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
    pulsar_qwen_step st;
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
    bool ok = pulsar_gpu_tensor_write(s->qwen->row_pos, 0, pos, (uint64_t)n_rows * sizeof(int32_t)) != 0 &&
              pulsar_gpu_tensor_write(s->qwen->row_bank, 0, bank, (uint64_t)n_rows * sizeof(int32_t)) != 0;
    if (ok) ok = pulsar_gpu_begin_commands() != 0;
    if (ok) ok = ops->embed(&st);
    for (uint32_t il = 0; ok && il < e->plan.n_layer; il++) {
        if (il == g_qwen_shape.ple_layer) ok = ops->ple(&st, il);
        if (ok) ok = ops->gr_read(&st, il, PULSAR_QWEN_GR_ATTN);
        if (ok) ok = e->plan.kind[il] == PULSAR_LAYER_QWEN_GDN ? ops->gdn(&st, il) : ops->qsa(&st, il);
        if (ok) ok = ops->gr_write(&st, il, PULSAR_QWEN_GR_ATTN);
        if (ok) ok = ops->gr_read(&st, il, PULSAR_QWEN_GR_MLP);
        if (ok) ok = ops->moe(&st, il);
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
    if (ok && head_n) ok = ops->head(&st, head_row0, head_n);
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
    if (ok && head_n)
        ok = pulsar_gpu_tensor_read(s->qwen->logits, 0, logits_out,
                                    (uint64_t)head_n * g_qwen_shape.n_vocab * sizeof(float)) != 0;
    return ok;
}

/* Prefill `n` tokens of the live bank from position `start`, in prefill_cap
 * chunks; the last chunk heads its last row into s->logits. */
static bool qwen_prefill(pulsar_session *s, const pulsar_tokens *prompt, uint32_t start) {
    const uint32_t n = (uint32_t)prompt->len - start;
    const uint32_t live = s->qwen->live_bank;
    int32_t *pos = (int32_t *)xmalloc((size_t)s->prefill_cap * sizeof(int32_t));
    int32_t *bank = (int32_t *)xmalloc((size_t)s->prefill_cap * sizeof(int32_t));
    for (uint32_t r = 0; r < s->prefill_cap; r++) bank[r] = (int32_t)live;
    bool ok = true;
    for (uint32_t off = 0; ok && off < n; off += s->prefill_cap) {
        const uint32_t rows = n - off < s->prefill_cap ? n - off : s->prefill_cap;
        for (uint32_t r = 0; r < rows; r++) pos[r] = (int32_t)(start + off + r);
        const bool last = off + rows == n;
        ok = qwen_forward(s, PULSAR_QWEN_STEP_PREFILL, prompt->v + start + off, pos, bank, rows,
                          rows - 1u, last ? 1u : 0u, s->logits);
        if (ok) s->qwen->bank_pos[live] = start + off + rows;
    }
    s->qwen->logits_fresh = ok;
    free(pos);
    free(bank);
    return ok;
}

/* Make the live bank hold exactly `prompt`.  A recurrent state cannot be cut back,
 * so the one path is: continue when the prompt extends what the bank holds,
 * otherwise clear the bank and prefill from 0.  (Resuming from saved state
 * snapshots is the prefix-reuse follow-up; until it exists a divergent prompt
 * re-prefills.) */
static int qwen_session_sync(pulsar_session *s, const pulsar_tokens *prompt,
                             const pulsar_image_ref *, int n_images, char *err, size_t errlen) {
    if (n_images > 0) {
        if (err) snprintf(err, errlen, "%s: images are not implemented for this family", PULSAR_QWEN_ARCH);
        return 1;
    }
    if (!prompt || prompt->len <= 0 || prompt->len > s->ctx_size) {
        if (err) snprintf(err, errlen, "%s: prompt length %d outside [1, %d]", PULSAR_QWEN_ARCH,
                          prompt ? prompt->len : -1, s->ctx_size);
        return 1;
    }
    int common = 0;
    if (s->checkpoint_valid) {
        while (common < s->checkpoint.len && common < prompt->len &&
               s->checkpoint.v[common] == prompt->v[common]) common++;
    }
    /* bank_pos[live] is the state's authority; the checkpoint must agree with it
     * to be continued (a batched step on the live bank moves the state, not the
     * checkpoint, and clears checkpoint_valid). */
    const uint32_t live = s->qwen->live_bank;
    if (s->checkpoint_valid && s->qwen->bank_pos[live] != (uint32_t)s->checkpoint.len) s->checkpoint_valid = false;
    const bool extends = s->checkpoint_valid && common == s->checkpoint.len && common < prompt->len;
    /* the same prompt is a no-op only while the logits are its next-token row: after the batched
     * lane (note_committed) they are stale, and a recurrent state cannot rewind one token to redo
     * the last row -- so that case prefills cold */
    if (s->checkpoint_valid && common == s->checkpoint.len && common == prompt->len && s->qwen->logits_fresh) return 0;
    uint32_t start = 0;
    if (extends) {
        start = (uint32_t)common;
    } else if (!qwen_state_reset_bank(s->qwen, &g_qwen_shape, &s->engine->plan, live)) {
        if (err) snprintf(err, errlen, "%s: could not clear the session's state", PULSAR_QWEN_ARCH);
        return 1;
    }
    s->checkpoint_valid = false;
    if (!qwen_prefill(s, prompt, start)) {
        if (err) snprintf(err, errlen, "%s: prefill refused (see the log for the op)", PULSAR_QWEN_ARCH);
        return 1;
    }
    s->checkpoint.len = 0;
    for (int i = 0; i < prompt->len; i++) token_vec_push(&s->checkpoint, prompt->v[i]);
    s->checkpoint_valid = true;
    return 0;
}

static int qwen_session_eval(pulsar_session *s, int token, char *err, size_t errlen) {
    if (!s->checkpoint_valid || s->checkpoint.len >= s->ctx_size) {
        if (err) snprintf(err, errlen, "%s: eval needs a synced session with room left", PULSAR_QWEN_ARCH);
        return 1;
    }
    const uint32_t live = s->qwen->live_bank;
    if (s->qwen->bank_pos[live] != (uint32_t)s->checkpoint.len) {
        if (err) snprintf(err, errlen, "%s: eval: bank %u holds %u tokens, the checkpoint %d", PULSAR_QWEN_ARCH, live,
                          s->qwen->bank_pos[live], s->checkpoint.len);
        return 1;
    }
    const int32_t tok = token, pos = s->checkpoint.len, bank = (int32_t)live;
    if (!qwen_forward(s, PULSAR_QWEN_STEP_DECODE, &tok, &pos, &bank, 1, 0, 1, s->logits)) {
        if (err) snprintf(err, errlen, "%s: decode refused (see the log for the op)", PULSAR_QWEN_ARCH);
        return 1;
    }
    token_vec_push(&s->checkpoint, token);
    s->qwen->bank_pos[live] = (uint32_t)s->checkpoint.len;
    s->qwen->logits_fresh = true;
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
    if (!qwen_forward(s, PULSAR_QWEN_STEP_DECODE, tok, pos, bank, n, 0, n, logits)) {
        /* the failed step reset the banks it touched (qwen_forward): fatal for these rows */
        if (err) snprintf(err, errlen, "%s: decode refused (see the log for the op)", PULSAR_QWEN_ARCH);
        return -1;
    }
    for (uint32_t i = 0; i < n; i++) {
        s->qwen->bank_pos[reqs[i].bank]++;
        if (reqs[i].bank == s->qwen->live_bank) {   /* the live bank moved past the host view */
            s->checkpoint_valid = false;
            s->qwen->logits_fresh = false;
        }
    }
    return 0;
}

/* The server's batched lane: its plain decode is this entry with one row per bank and no prefill
 * runs (max_head_runs 0), which is exactly the batched decode.  A step that fuses a PREFILL run (a
 * bank with more than one row) is refused before anything moves (rc 1: the lane then retries
 * decode-only; prefill keeps the sync path). */
static int qwen_session_decode_mixed(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows,
                                     float *logits, int logits_cap, uint32_t *out_n_rows,
                                     uint32_t max_head_runs, char *err, size_t errlen) {
    if (out_n_rows) *out_n_rows = 0;
    if (max_head_runs == PULSAR_MSEQ_HEAD_ALL_ROWS) {
        if (err) snprintf(err, errlen, "%s: a verify step (heads on every row) is not implemented", PULSAR_QWEN_ARCH);
        return 1;
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

/* The HOST view forgets the live bank's history; its device state is left alone.  The next sync
 * of that bank finds the checkpoint invalid and resets the bank before it prefills, so a stale
 * state is never decoded -- and the server's invalidate (a bank provision, an eviction, a stop
 * string mid-batch) cannot wipe OTHER banks' conversations, which resetting every bank did. */
static void qwen_session_invalidate(pulsar_session *s) {
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->qwen->logits_fresh = false;
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
};

const pulsar_family PULSAR_FAMILY_QWEN4_EXP = {
    /* .id           = */ PULSAR_FAMILY_ID_QWEN4_EXP,
    /* .arch         = */ PULSAR_QWEN_ARCH,
    /* .name         = */ "Qwen4-exp",
    /* .drafter      = */ PULSAR_DRAFTER_MTP,
    /* .caps         = */ PULSAR_FAMILY_CAP_BANKS,
    /* .load         = */ qwen_family_load,
    /* .after_gpu    = */ qwen_family_after_gpu,
    /* .logits_width = */ qwen_logits_width,
    /* .model_name   = */ qwen_model_name,
    /* .chat_format  = */ qwen_chat_format,
    /* .model_id     = */ qwen_model_id,
    /* .session      = */ &k_qwen_session_ops,
    /* .banks        = */ &k_qwen_bank_ops,
};

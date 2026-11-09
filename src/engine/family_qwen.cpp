/* family_qwen.cpp -- the Qwen4-exp family (Qwen3.8-Flash-Next), L251 S1.
 *
 * What is here: config validation against the known shape, the layer plan from
 * the artifact's layer_types, the weight binding by HF name, the DECLARATION of
 * the session state (L295 F3: DeltaNet recurrent + conv state, FP8 KV and pooled
 * index keys, PLE conv state -- the core's state_kinds.cpp builds and runs it),
 * the family's own session state, and its blocks on the core step driver (L295
 * F4: one block per layer kind, the block drafter's taps -- step_driver.cpp
 * runs every step).  What is NOT here: any op (family_qwen_s4.cpp). */
#include "pulsar_engine_internal.h"
#include "exl3_trellis.h"
#include "lib/bpe_tokenizer.h"
#include "tp/pulsar_tp.h"
#include "spec_block.h"
#include "cuda/pulsar_cuda_blockdraft.h"
#include "cuda/pulsar_cuda_gdn.h"

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
};

pulsar_qwen_shape g_qwen_shape = PULSAR_QWEN_SHAPE_FLASH_NEXT;

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
    QCHK_U(max_position); QCHK_U(eos_id);
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

/* L255: one layer's routed experts in Tessera's value family -- every plane of gate, up and down
 * (family_qwen.h PULSAR_QWEN_TESS_*), dims in ne order against the stack.  The words' per-expert stride is the
 * encoder's (it follows the column rates), so only their expert count is checked; the launcher checks the rest
 * of the geometry at every call. */
static const char *const kTessProj[PULSAR_QWEN_TESS_PROJS] = {"gate", "up", "down"};
static const char *const kTessPlane[PULSAR_QWEN_TESS_PLANES] = {"words", "table", "init",  "has_init",
                                                                "wscale", "runs", "bdesc", "geom"};

static void qwen_bind_tessera(const pulsar_model *m, bool *ok, const pulsar_qwen_shape *s, const char *lp,
                              pulsar_qwen_layer_weights *L) {
    const uint64_t nE = s->n_expert;
    for (int p = 0; p < PULSAR_QWEN_TESS_PROJS; ++p) {
        const uint64_t in = p == PULSAR_QWEN_TESS_DOWN ? s->n_ff_exp : s->n_embd;
        const uint64_t out = p == PULSAR_QWEN_TESS_DOWN ? s->n_embd : s->n_ff_exp;
        char name[256];
        auto nm = [&](int plane) -> const char * {
            snprintf(name, sizeof(name), "%smlp.experts.tessera.%s.%s", lp, kTessProj[p], kTessPlane[plane]);
            return name;
        };
        pulsar_tensor **P = L->moe_tess[p];
        P[PULSAR_QWEN_TESS_WORDS] = model_find_tensor(m, nm(PULSAR_QWEN_TESS_WORDS));
        if (!P[PULSAR_QWEN_TESS_WORDS] || P[PULSAR_QWEN_TESS_WORDS]->ndim != 2 ||
            P[PULSAR_QWEN_TESS_WORDS]->dim[1] != nE) {
            fprintf(stderr, "pulsar: %s: %s is missing or not [%llu][words] -- refusing\n", PULSAR_QWEN_ARCH, name,
                    (unsigned long long)nE);
            *ok = false;
        }
        P[PULSAR_QWEN_TESS_TABLE]    = qbind(m, ok, nm(PULSAR_QWEN_TESS_TABLE), 2, 16384, nE);
        P[PULSAR_QWEN_TESS_INIT]     = qbind(m, ok, nm(PULSAR_QWEN_TESS_INIT), 2, in, nE);
        P[PULSAR_QWEN_TESS_HAS_INIT] = qbind(m, ok, nm(PULSAR_QWEN_TESS_HAS_INIT), 1, nE);
        P[PULSAR_QWEN_TESS_WSCALE]   = qbind(m, ok, nm(PULSAR_QWEN_TESS_WSCALE), 2, out, nE);
        P[PULSAR_QWEN_TESS_RUNS]     = qbind(m, ok, nm(PULSAR_QWEN_TESS_RUNS), 2, 8, nE);
        P[PULSAR_QWEN_TESS_BDESC]    = qbind(m, ok, nm(PULSAR_QWEN_TESS_BDESC), 3, 12, in / 32, nE);
        P[PULSAR_QWEN_TESS_GEOM]     = qbind(m, ok, nm(PULSAR_QWEN_TESS_GEOM), 1, 2);
    }
}

/* One layer's tensors under `lp` (the layer's name prefix, "model.language_model.layers.<il>.").  `split` binds the
 * routed experts as separate gate / up stacks (turboderp's packs, L266) instead of the fused gate_up. */
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
    if (model_find_tensor(m, nm("mlp.experts.tessera.gate.words"))) {
        /* L255: this layer's experts are Tessera planes -- and nothing else may declare them */
        if (split || model_find_tensor(m, nm("mlp.experts.gate_up_proj")) ||
            model_find_tensor(m, nm("mlp.experts.down_proj"))) {
            fprintf(stderr, "pulsar: %s: %s declares its routed experts as Tessera planes AND as EXL3 stacks (or in "
                            "a split-stack container, which has no Tessera arm) -- refusing\n", PULSAR_QWEN_ARCH, lp);
            *ok = false;
        }
        qwen_bind_tessera(m, ok, s, lp, L);
    } else if (split) {
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
    return ok;
}

/* L305: Qwen drafts with its block drafter (pulsar.drafter, DFlash2: role-recorded blocks) or not at all.  The
 * checkpoint's MTP module (mtp.*) is retired (Tyler 2026-10-20: DFlash2 won every width of the L299 scoreboard): a
 * container carrying any of its tensors is refused -- nothing here reads them, and a container that ships them is not
 * one this family's builder writes -- and so is a --drafter naming anything but the container's block drafter.
 * --no-dspark loads no drafter; the block drafter is then declared unloaded (never staged; its memory stays the KV
 * pool's). */
static bool qwen_select_drafter(pulsar_engine *e, const pulsar_engine_options *opt, bool *use_block) {
    pulsar_model *m = &e->model;
    uint64_t n_mtp = 0;
    for (uint64_t i = 0; i < m->n_tensors; i++)
        n_mtp += m->tensors[i].name.len > 4 && memcmp(m->tensors[i].name.ptr, "mtp.", 4) == 0;
    if (n_mtp) {
        fprintf(stderr, "pulsar: %s: the container carries %llu tensors of the retired MTP module (mtp.*; L305: this "
                        "family's drafter is its block drafter) -- rebuild it without them (tools/container/build.py "
                        "drops them); refusing\n", PULSAR_QWEN_ARCH, (unsigned long long)n_mtp);
        return false;
    }
    const char *bk = pulsar_block_drafter_kind_of(m);
    if (!bk) return false;
    const bool block_in = bk[0] != 0;
    const char *want = opt->drafter;
    *use_block = false;
    if (opt->dspark_disable) {
    } else if (!want || !want[0] || (block_in && !strcmp(want, bk))) {
        *use_block = block_in;
    } else {
        fprintf(stderr, "pulsar: %s: --drafter %s names no drafter the container holds (it holds: %s) -- refusing\n",
                PULSAR_QWEN_ARCH, want, block_in ? bk : "none");
        return false;
    }
    if (*use_block && m->tp_n_ranks > 1) {
        fprintf(stderr, "pulsar: %s: the block drafter runs on one GPU (no tensor-parallel lane yet) -- refusing\n",
                PULSAR_QWEN_ARCH);
        return false;
    }
    return true;
}

/* L299: the block drafter's declaration and taps (the HC-mean of the streams after each tapped layer's MLP write,
 * qwen_layer), when selected */
static bool qwen_block_read(pulsar_engine *e, const pulsar_qwen_shape *s, pulsar_qwen_weights *w) {
    w->block = (pulsar_block_drafter *)xcalloc(1, sizeof(*w->block));
    pulsar_block_drafter *d = w->block;
    if (!pulsar_block_drafter_read(&e->model, d)) return false;
    for (uint32_t j = 0; j < d->n_taps; j++) {
        if (d->tap_block[j] >= s->n_layer) {
            fprintf(stderr, "pulsar: %s: the drafter taps layer %u of a %u-layer target -- refusing\n", PULSAR_QWEN_ARCH,
                    d->tap_block[j], s->n_layer);
            return false;
        }
        d->tap_layer[j] = d->tap_block[j];        /* HF layer i is the trunk's layer i */
        d->tap_sub[j] = PULSAR_BLOCK_TAP_FFN;     /* its completed output: after the MLP side's gated-residual write */
    }
    if (d->hidden != s->n_embd || d->vocab != s->n_vocab) {
        fprintf(stderr, "pulsar: %s: the drafter's hidden %u / vocabulary %u are not the target's %u / %u -- refusing\n",
                PULSAR_QWEN_ARCH, d->hidden, d->vocab, s->n_embd, s->n_vocab);
        return false;
    }
    return true;
}

/* L299: the block drafter's tensors (role-recorded blocks; the rest of the container binds by name) bound when
 * selected, else declared unloaded */
static bool qwen_block_bind(pulsar_engine *e, pulsar_qwen_weights *w) {
    const char *kind = pulsar_block_drafter_kind_of(&e->model);
    if (!kind || !kind[0]) return kind != NULL;
    pulsar_role_binder *b = pulsar_role_binder_open_blocks(&e->model, PULSAR_QWEN_ARCH, pulsar_block_drafter_blocks(kind));
    if (w->block) {
        (void)pulsar_block_drafter_bind(b, w->block);
    } else {
        uint64_t n = 0;
        for (const char *const *bl = pulsar_block_drafter_blocks(kind); *bl; bl++) n += pulsar_role_binder_unload(b, *bl);
        fprintf(stderr, "pulsar: %s: the %s block drafter (%llu tensors) is not the selected drafter: unloaded\n",
                PULSAR_QWEN_ARCH, kind, (unsigned long long)n);
    }
    pulsar_role_census c;
    return pulsar_role_binder_close(b, &c) && !pulsar_load_refusals();
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
    e->qwen_tok = bpe_tokenizer_load_hf(tj.data(), tj.size(), gc.data(), gc.size(), err, sizeof(err));
    if (!e->qwen_tok) {
        fprintf(stderr, "pulsar: %s: tokenizer refused: %s\n", PULSAR_QWEN_ARCH, err);
        return false;
    }
    const std::vector<int> &stop = bpe_tokenizer_stop_ids(e->qwen_tok);
    fprintf(stderr, "pulsar: %s: tokenizer %d ids from %s; stop ids", PULSAR_QWEN_ARCH,
            bpe_tokenizer_n_tokens(e->qwen_tok), tp.c_str());
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
    bool use_block = false;
    if (!qwen_select_drafter(e, opt, &use_block)) return false;
    if (use_block && !qwen_block_read(e, &s, e->qwen_weights)) return false;
    weights_reject_unsupported_types(&e->model);
    weights_reject_bad_e8m0(&e->model);
    if (pulsar_load_refusals()) {
        fprintf(stderr, "pulsar: %s: %u refusal(s) in the artifact scans -- the model does not load\n", PULSAR_QWEN_ARCH,
                pulsar_load_refusals());
        return false;
    }
    if (!qwen_bind_weights(&e->model, &g_qwen_shape, &e->plan, e->qwen_weights) ||
        !qwen_block_bind(e, e->qwen_weights)) {
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
        e->qwen_weights->vision_pad_id = bpe_tokenizer_added_id(e->qwen_tok, "<|image_pad|>");
        if (e->qwen_weights->vision_pad_id < 0 || g_qwen_shape.n_embd != PULSAR_QWEN_VISION_OUT) {
            fprintf(stderr, "pulsar: %s: the vision tower needs <|image_pad|> in the tokenizer and a %u-wide model "
                            "(pad %d, n_embd %u) -- refusing\n", PULSAR_QWEN_ARCH, (unsigned)PULSAR_QWEN_VISION_OUT,
                    e->qwen_weights->vision_pad_id, g_qwen_shape.n_embd);
            return false;
        }
        e->vision_ready = true;
    }
    /* L299: the block drafter (DFlash2) behind the round API, the target's input embedding its block's; the drafter
     * option (--no-dspark) turns drafting off for this family as it does for DeepSeek's */
    if (pulsar_block_drafter *d = e->qwen_weights->block) {
        d->embed = e->qwen_weights->token_embd;
        e->block_drafter = d;
        e->drafter_ops = &d->ops;
        char desc[320];
        pulsar_block_drafter_describe(d, desc, sizeof desc);
        fprintf(stderr, "pulsar: %s: %s\n", PULSAR_QWEN_ARCH, desc);
    }
    qwen_state_plan(e);   /* L295 F3: the shape is the rank's, the drafter selected */
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

/* The lane, announced once at open (rule 5): the core driver over Qwen's blocks, and what its table lacks for the
 * plan (a session refuses with exactly that). */
static bool qwen_family_after_gpu(pulsar_engine *e) {
    uint32_t at = 0;
    const char *miss = pulsar_driver_first_missing(e->family->driver, &e->plan, &at);
    fprintf(stderr, "pulsar: %s: the core step driver over %u qwen_gdn + %u qwen_qsa blocks%s%s%s\n", PULSAR_QWEN_ARCH,
            pulsar_layer_plan_count(&e->plan, PULSAR_LAYER_QWEN_GDN), pulsar_layer_plan_count(&e->plan, PULSAR_LAYER_QWEN_QSA),
            e->block_drafter ? ", the block drafter's lane (draft_kv RING)" : "",
            miss ? "; MISSING " : "", miss ? miss : "");
    return true;
}

/* ---- engine facts ------------------------------------------------------------- */

static uint32_t qwen_logits_width(const pulsar_engine *) { return g_qwen_shape.n_vocab; }
/* L272 B15: the drafter the open selected -- (L299) the block drafter, or none. */
static pulsar_drafter_kind qwen_drafter(pulsar_engine *e) {
    return e->block_drafter ? PULSAR_DRAFTER_BLOCK : PULSAR_DRAFTER_NONE;
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

/* L295 F3: Qwen's state, DECLARED (state_kinds.h; the ids are family_qwen.h's PULSAR_QWEN_STATE_*).  The core
 * builds, prices, pages, walks and commits it; what follows is only what the model IS.
 *
 *   per GDN layer (36)  the DeltaNet recurrent state (FIXED, 3.1 MB fp32; a verify row keeps its update, L288, and
 *                       the commit applies the kept ones) and the causal-conv tail (FIXED, the state after each
 *                       verify row);
 *   per QSA layer (12)  KV rows (POOL, 1 token a row, 1056 B, E4M3 + E8M0) and pooled indexer keys (POOL, idx_block
 *                       tokens a row, 256 B) -- demand-paged per bank -- and the open index block (STAGED; a verify
 *                       row's record is its index_qk projection, the key after the query heads);
 *   the PLE layer       its dilated-conv history (FIXED, snapshot per verify row);
 *   the block drafter   its context rings, on its own drafter slots (spec_block.h pulsar_block_drafter_plan);
 *   session-wide        the n-gram context, host state (the last two ids, reset at EOS): replayed over a verify's
 *                       kept tokens, and carried in the checkpoint slot's tail so a restore needs no token replay.
 *
 * THE RESUME GRID: Qwen's recurrent GDN and per-row QSA are split-invariant by construction, and every prompt chunk
 * takes the prefill arms at every row count with a pinned GEMM (L266: session_contract_gate C1) -- so a prompt cut
 * ANYWHERE is the cold prefill's bytes.  The grid is therefore free; 128 keeps the slots on the server's and
 * DeepSeek's cadence and closes an indexer block (the core checks every pool's rows divide it).
 *
 * THE PREFILL FRONTIER: a checkpoint is the cold prefill of its tokens only if every row below it was prefilled
 * (decode rows are the decode arms'); the core's prefill_pos[bank] ends the bank's prefill-only history, decode
 * never advances it, and a capture requires bank_pos == G == prefill_pos. */
#define QWEN_RESUME_GRID 128u
/* ~118 MB a slot (the GDN recurrent state), so 4 -- not DeepSeek's 16 x ~3.8 MB, which here would be ~1.9 GB a bank
 * (Tyler 2026-10-08: "4"); an edit up to 2 turns back resumes from a slot, not from 0 (the ladder is checkpoint.cpp's). */
#define QWEN_CKPT_SLOTS 4u

/* DELTA (L288): apply the kept rows' recurrent updates -- the verify left the state as it found it. */
static bool qwen_gdn_commit(void *state, uint32_t bank, const void *rows, uint32_t keep) {
    return pulsar_gdn_commit((float *)state, (int)bank, (const float *)rows, (int)keep,
                             (int)pulsar_qwen_tp(&g_qwen_shape), 0) == 0;
}

/* The n-gram context: a fresh bank's is the reset token (HF resets it at EOS, and a sequence start is a reset); a
 * verify's commit replays the kept tokens over the bank's context before the step. */
static void qwen_ngram_init(const void *, void *state) {
    int32_t *ctx = (int32_t *)state;
    for (uint32_t i = 0; i + 1u < g_qwen_shape.ngram_size; i++) ctx[i] = (int32_t)g_qwen_shape.eos_id;
}

static bool qwen_ngram_replay(const void *e, void *state, const int32_t *tokens, uint32_t keep) {
    return pulsar_qwen_s4_ngram_replay(((const pulsar_engine *)e)->qwen_weights, (int32_t *)state, tokens, keep);
}

void qwen_state_plan(pulsar_engine *e) {
    const pulsar_qwen_shape *s = &g_qwen_shape;
    const pulsar_layer_plan *lp = &e->plan;
    pulsar_state_plan *p = &e->state_plan;
    *p = pulsar_state_plan{};
    p->name = "qwen4-exp";   /* the kv-state's name since L266: mixed into every segment's digest */
    p->resume_grid = QWEN_RESUME_GRID;
    p->split_invariant = true;   /* THE RESUME GRID above */
    p->ckpt_slots = QWEN_CKPT_SLOTS;
    pulsar_state_decl d[PULSAR_QWEN_STATE_COUNT] = {};
    d[PULSAR_QWEN_STATE_GDN_REC] = {"gdn_state", PULSAR_STATE_FIXED, PULSAR_STATE_CAPTURE_DELTA, false, false, 0,
                                    pulsar_qwen_gdn_state_bytes(s),
                                    pulsar_gdn_upd_floats((int)pulsar_qwen_tp(s)) * sizeof(float), 0, qwen_gdn_commit,
                                    NULL, NULL};
    d[PULSAR_QWEN_STATE_GDN_CONV] = {"gdn_conv", PULSAR_STATE_FIXED, PULSAR_STATE_CAPTURE_SNAPSHOT, false, false, 0,
                                     pulsar_qwen_gdn_conv_bytes(s), 0, 0, NULL, NULL, NULL};
    d[PULSAR_QWEN_STATE_KV] = {"kv", PULSAR_STATE_POOL, PULSAR_STATE_CAPTURE_NONE, false, false, 1u,
                               pulsar_qwen_kv_row_bytes(s), 0, 0, NULL, NULL, NULL};
    d[PULSAR_QWEN_STATE_IDX_KEYS] = {"idx_keys", PULSAR_STATE_POOL, PULSAR_STATE_CAPTURE_NONE, false, false,
                                     s->idx_block, pulsar_qwen_index_row_bytes(s), 0, 0, NULL, NULL, NULL};
    d[PULSAR_QWEN_STATE_IDX_TAIL] = {"idx_tail", PULSAR_STATE_STAGED, PULSAR_STATE_CAPTURE_STAGE, false, false,
                                     s->idx_block, pulsar_qwen_index_tail_bytes(s) / s->idx_block,
                                     (uint64_t)PULSAR_QSA_IDX_IN * sizeof(float),
                                     (uint64_t)PULSAR_QSA_IDX_HEADS * PULSAR_QSA_IDX_DIM * sizeof(float), NULL, NULL,
                                     NULL};
    d[PULSAR_QWEN_STATE_PLE_CONV] = {"ple_conv", PULSAR_STATE_FIXED, PULSAR_STATE_CAPTURE_SNAPSHOT, false, false, 0,
                                     pulsar_qwen_ple_conv_bytes(s), 0, 0, NULL, NULL, NULL};
    d[PULSAR_QWEN_STATE_NGRAM] = {"ngram_ctx", PULSAR_STATE_FIXED, PULSAR_STATE_CAPTURE_REPLAY, true, false, 0,
                                  (uint64_t)(s->ngram_size - 1u) * sizeof(int32_t), 0, 0, NULL, qwen_ngram_replay,
                                  qwen_ngram_init};
    for (uint32_t id = 0; id < PULSAR_QWEN_STATE_COUNT; id++) (void)pulsar_state_plan_add(p, d[id]);
    const uint32_t gdn = 1u << PULSAR_QWEN_STATE_GDN_REC | 1u << PULSAR_QWEN_STATE_GDN_CONV;
    const uint32_t qsa = 1u << PULSAR_QWEN_STATE_KV | 1u << PULSAR_QWEN_STATE_IDX_KEYS | 1u << PULSAR_QWEN_STATE_IDX_TAIL;
    for (uint32_t il = 0; il < lp->n_layer; il++)
        p->slot[il] = (lp->kind[il] == PULSAR_LAYER_QWEN_GDN ? gdn : qsa) |
                      (il == s->ple_layer ? 1u << PULSAR_QWEN_STATE_PLE_CONV : 0u);
    p->n_trunk = lp->n_layer;
    p->n_slots = lp->n_layer;
    /* L299: the block drafter's slots (its context rings; spec_block.h) */
    if (e->block_drafter) pulsar_block_drafter_plan(e, p);
    p->session = 1u << PULSAR_QWEN_STATE_NGRAM;
}

/* A Qwen session's state: the driver's buffers are its base (L295 F4). */
static pulsar_qwen_state *qwen_st(const pulsar_session *s) { return static_cast<pulsar_qwen_state *>(s->driver); }

static void qwen_state_free(pulsar_qwen_state *st) {
    if (!st) return;
    pulsar_gpu_tensor_free(st->streams);
    pulsar_gpu_tensor_free(st->x);
    pulsar_gpu_tensor_free(st->y);
    free(st->trunk);
    free(st);
}

/* The family's own session state (the driver's create, step_driver.h; the driver then builds its buffers and the
 * declared state): the step's activation slots and the trunk's step object.  Allocates only through
 * the tensor allocators -- the create is also the session's price (session_cost_bytes_banked). */
static pulsar_driver_state *qwen_create(pulsar_session *s, uint32_t n_banks) {
    pulsar_engine *e = s->engine;
    const pulsar_qwen_shape *sh = &g_qwen_shape;
    pulsar_qwen_state *st = (pulsar_qwen_state *)xcalloc(1, sizeof(*st));
    const uint64_t rows = s->prefill_cap;
    st->streams = pulsar_gpu_tensor_alloc(rows * pulsar_qwen_hc_dim(sh) * PULSAR_QWEN_STREAM_ELT_SIZE);
    st->x       = pulsar_gpu_tensor_alloc(rows * sh->n_embd * PULSAR_QWEN_X_ELT_SIZE);
    st->y       = pulsar_gpu_tensor_alloc(rows * sh->n_embd * PULSAR_QWEN_Y_ELT_SIZE);
    bool ok = st->streams && st->x && st->y;
    /* the trunk's lane: the step's own part, fixed for the session */
    st->trunk = (pulsar_qwen_step *)xcalloc(1, sizeof(pulsar_qwen_step));
    st->trunk->shape = sh;
    st->trunk->w = e->qwen_weights;
    st->trunk->st = st;
    if (!ok) {
        fprintf(stderr, "pulsar: %s: session state allocation failed (%u banks x %d tokens)\n", PULSAR_QWEN_ARCH,
                n_banks, s->ctx_size);
        qwen_state_free(st);
        return NULL;
    }
    return st;
}

static void qwen_destroy(pulsar_session *s) { qwen_state_free(qwen_st(s)); }

/* the declared state's verify capture exists with a drafter (L299: the block drafter) */
static bool qwen_verifies(const pulsar_session *s) { return s->engine->block_drafter != NULL; }

static pulsar_driver_step *qwen_trunk(pulsar_session *s) { return qwen_st(s)->trunk; }

/* ---- the blocks on the core driver (step_driver.h, L295 F4) ------------------------------------------------------
 *
 * The core driver runs every step (embed, the plan block by block, the head, the drafter's absorb, the banks'
 * positions); Qwen supplies its ops, one block for each of its layer kinds, and the block drafter's taps. */

static const pulsar_qwen_step *QS(const pulsar_driver_step *st) { return static_cast<const pulsar_qwen_step *>(st); }

/* One layer of THE FORWARD (family_qwen.h): the PLE injection on its layer, the attention side's gated-residual
 * read, the mixer (the plan's kind), its TP exchange, the write; the MLP side's read, the MoE, its exchange, the
 * write.  L266 step 7: under TP a rank's mixer (its heads) and MoE (its experts, rank 0's shared expert) leave a
 * PARTIAL block output in y; the group's sum is the block's -- one exchange each in the layer's slot. */
static bool qwen_layer(const pulsar_qwen_step *st, uint32_t il, pulsar_qwen_layer_fn mixer) {
    bool ok = true;
    if (il == g_qwen_shape.ple_layer) ok = pulsar_qwen_s4_ple(st, il);
    if (ok) ok = pulsar_qwen_s4_gr_read(st, il, PULSAR_QWEN_GR_ATTN);
    if (ok) ok = mixer(st, il);
    if (ok) ok = pulsar_driver_tp_allreduce(st, il, st->st->y, g_qwen_shape.n_embd, "mixer");
    if (ok) ok = pulsar_qwen_s4_gr_write(st, il, PULSAR_QWEN_GR_ATTN);
    if (ok) ok = pulsar_qwen_s4_gr_read(st, il, PULSAR_QWEN_GR_MLP);
    if (ok) ok = pulsar_qwen_s4_moe(st, il);
    if (ok) ok = pulsar_driver_tp_allreduce(st, il, st->st->y, g_qwen_shape.n_embd, "moe");
    if (ok) ok = pulsar_qwen_s4_gr_write(st, il, PULSAR_QWEN_GR_MLP);
    /* L299: the block drafter's taps after a tapped layer's completed output -- the mean of its n_hc streams */
    if (ok) {
        const uint64_t HC = pulsar_qwen_hc_dim(&g_qwen_shape);
        const uint16_t *h = (const uint16_t *)pulsar_gpu_tensor_device_ptr(st->st->streams);
        ok = pulsar_block_tap_rows(st->st, il, PULSAR_BLOCK_TAP_FFN, st->n_rows, [&](uint16_t *dst, uint64_t ld) {
            return pulsar_blockdraft_stream_mean_launch(h, HC, g_qwen_shape.n_hc, st->n_rows, g_qwen_shape.n_embd, dst,
                                                        ld, 0) == 0;
        });
    }
    return ok;
}

static bool qwen_block_gdn(const pulsar_driver_step *st, uint32_t il) { return qwen_layer(QS(st), il, pulsar_qwen_s2_gdn); }
static bool qwen_block_qsa(const pulsar_driver_step *st, uint32_t il) { return qwen_layer(QS(st), il, pulsar_qwen_s3_qsa); }
static bool qwen_embed(const pulsar_driver_step *st) { return pulsar_qwen_s4_embed(QS(st)); }
static bool qwen_head(const pulsar_driver_step *st, uint32_t row0, uint32_t n, uint32_t out0) {
    return pulsar_qwen_s4_head(QS(st), row0, n, out0);
}
static bool qwen_step_end(const pulsar_driver_step *st, bool ok) { return pulsar_qwen_s4_step_end(QS(st), ok); }
static uint64_t qwen_scratch_bytes(uint32_t arena, uint32_t max_rows, uint32_t ctx) {
    return pulsar_qwen_s4_scratch_bytes((pulsar_qwen_op_id)arena, &g_qwen_shape, max_rows, ctx);
}
static uint64_t qwen_artifact(const pulsar_engine *e) { return e->qwen_weights->artifact_digest; }

/* Diagnostic tap: the HC streams after layer il -- `n_hc` x n_embd f32 per row, the same quantity the streamed
 * reference keeps in `hs` ([seq, hc_count, n_embd]).  It is how a reference-gate miss gets attributed to a layer
 * instead of to "the model".  The WHOLE step's streams ([n_rows][n_hc][n_embd], bf16 -> widened f32): the
 * layer-correctness test feeds the engine's stream at layer L into the REFERENCE's layer L+1 and compares with the
 * engine's L+1, which isolates a layer's own arithmetic from the error it inherited.  Volume is the caller's problem
 * -- filter with PULSAR_CUDA_GRAPH_DUMP_LAYER/_POS, since the dump helper synchronizes mid-graph. */
static void qwen_dump(const pulsar_driver_step *st, uint32_t il) {
    gpu_graph_debug_dump_hc_tensor("qwen_h", QS(st)->st->streams, (uint64_t)st->n_rows * pulsar_qwen_hc_dim(&g_qwen_shape),
                                   il, (uint32_t)st->pos[st->n_rows - 1]);
}

/* L268: the step's multi-axis rope positions (pulsar_image_rope3 over each row's bank's image records) -- NULL when
 * no row's bank holds a gridded image block: text positions, the bytes a text step always ran.  Per row, its own
 * (T, H, W) and its indexer block's first token's (pos - 3), which may be an earlier step's. */
static const uint32_t *qwen_rope_table(const pulsar_session *s, const int32_t *pos, const int32_t *bank, uint32_t n,
                                       std::vector<uint32_t> *out) {
    bool any = false;
    for (uint32_t r = 0; r < n && !any; r++) {
        const pulsar_image_identity *id = pulsar_session_bank_images(s, (uint32_t)bank[r], s->states->live_bank);
        for (uint32_t i = 0; id && i < id->n && !any; i++) any = id->b[i].grid_h && id->b[i].grid_w;
    }
    if (!any) return NULL;
    out->resize((size_t)n * 6u);
    for (uint32_t r = 0; r < n; r++) {
        const pulsar_image_identity *id = pulsar_session_bank_images(s, (uint32_t)bank[r], s->states->live_bank);
        const uint32_t p = (uint32_t)pos[r], back = PULSAR_QSA_BLOCK - 1u;
        pulsar_image_rope3(id, p, &(*out)[(size_t)r * 6u]);
        pulsar_image_rope3(id, p >= back ? p - back : 0u, &(*out)[(size_t)r * 6u + 3u]);
    }
    return out->data();
}

/* L268: image rows into the step's streams -- each row in all of them (the expansion follows the substitution) */
static bool qwen_image_rows(const pulsar_driver_step *st, const uint16_t *rows, uint32_t n_rows, uint32_t row0,
                            uint32_t n_tokens) {
    return pulsar_image_write_stream_rows(QS(st)->st->streams, rows, n_rows, row0, n_tokens, g_qwen_shape.n_embd,
                                          g_qwen_shape.n_hc);
}

static const pulsar_driver_ops k_qwen_driver = {
    /* .create        = */ qwen_create,
    /* .destroy       = */ qwen_destroy,
    /* .verifies      = */ qwen_verifies,
    /* .trunk         = */ qwen_trunk,
    /* .scratch_bytes = */ qwen_scratch_bytes,
    /* .n_scratch     = */ PULSAR_QWEN_OP_COUNT,
    /* .embed         = */ qwen_embed,
    /* .block         = */ {NULL, NULL, qwen_block_gdn, qwen_block_qsa},   /* by pulsar_layer_kind: GDN, QSA */
    /* .head          = */ qwen_head,
    /* .step_end      = */ qwen_step_end,
    /* .absorb        = */ pulsar_drafter_lane_absorb,   /* L299: the block lane's lockstep (none: nothing) */
    /* .mtp           = */ NULL,
    /* .rope          = */ qwen_rope_table,
    /* .image_rows    = */ qwen_image_rows,
    /* .dump          = */ qwen_dump,
    /* .artifact      = */ qwen_artifact,
};
static_assert(PULSAR_LAYER_QWEN_GDN == 2 && PULSAR_LAYER_QWEN_QSA == 3, "the block table's slots are the kinds' values");

static const pulsar_sync_ops k_qwen_sync = PULSAR_DRIVER_SYNC_OPS(PULSAR_QWEN_ARCH);

/* L299: the family's own state beside qwen_weights (which the engine's close frees): the block drafter */
static void qwen_unload(pulsar_engine *e) {
    if (e->qwen_weights) {
        free(e->qwen_weights->block);
        e->qwen_weights->block = NULL;
    }
    e->block_drafter = NULL;
}
static const pulsar_family_session_ops k_qwen_session_ops = PULSAR_DRIVER_SESSION_OPS(&k_qwen_sync);

/* The trunk's layers each take an exchange slot (L266 step 7). */
static void qwen_tp_shape(const pulsar_engine *e, uint32_t *n_layer, uint32_t *n_embd, uint32_t *n_vocab) {
    *n_layer = e->plan.n_layer;
    *n_embd = g_qwen_shape.n_embd;
    *n_vocab = g_qwen_shape.n_vocab;
}

const pulsar_family PULSAR_FAMILY_QWEN4_EXP = {
    /* .id           = */ PULSAR_FAMILY_ID_QWEN4_EXP,
    /* .arch         = */ PULSAR_QWEN_ARCH,
    /* .name         = */ "Qwen4-exp",
    /* .caps         = */ PULSAR_FAMILY_CAP_BANKS | PULSAR_FAMILY_CAP_SEGMENTS | PULSAR_FAMILY_CAP_PAYLOAD | PULSAR_FAMILY_CAP_TP |
                          PULSAR_FAMILY_CAP_SPEC | PULSAR_FAMILY_CAP_CHAT | PULSAR_FAMILY_CAP_VISION |
                          PULSAR_FAMILY_CAP_IMATRIX,
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
    /* .spec         = */ &PULSAR_DRIVER_SPEC_TARGET,   /* L295 F4: the driver's verify target */
    /* .session      = */ &k_qwen_session_ops,          /* L295 F4: the core driver's (step_driver.cpp) */
    /* .tokenizer    = */ &k_qwen_tokenizer,
    /* .tp_slices    = */ pulsar_qwen_tp_slices,
    /* .act_kind     = */ PULSAR_ACT_KIND_ROWS,
    /* .vision       = */ &PULSAR_QWEN_IMAGE_FRONT,   /* L268: vision_qwen.cpp */
    /* .banks        = */ &PULSAR_BANK_OPS_DECLARED,   /* L295 F3: state_kinds.cpp */
    /* .imatrix      = */ &k_qwen_imatrix,   /* L284 P15: imatrix_qwen.cpp */
    /* .driver       = */ &k_qwen_driver,    /* L295 F4 */
    /* .unload       = */ qwen_unload,       /* L299: the block drafter */
};

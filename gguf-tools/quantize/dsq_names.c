#include "dsq_internal.h"

/* =====
 * GGUF tensor mapping and quantization policy
 */

expert_tensor parse_expert_tensor(const char *name) {
    expert_tensor e = {0};
    int layer = -1;
    char kind[16];
    int rest = 0;
    if (sscanf(name, "blk.%d.ffn_%15[^_]_exps.weight%n", &layer, kind, &rest) == 2
        && rest == (int)strlen(name))
    {
        if (strcmp(kind, "gate") == 0 || strcmp(kind, "down") == 0 || strcmp(kind, "up") == 0) {
            e.is_expert = true;
            e.layer = layer;
            e.part = strcmp(kind, "gate") == 0 ? EXP_W1 : strcmp(kind, "down") == 0 ? EXP_W2 : EXP_W3;
        }
    } else if (sscanf(name, "dspark.%d.ffn_%15[^_]_exps.weight%n", &layer, kind, &rest) == 2
               && rest == (int)strlen(name) && layer >= 0 && layer <= 2)
    {
        if (strcmp(kind, "gate") == 0 || strcmp(kind, "down") == 0 || strcmp(kind, "up") == 0) {
            e.is_expert = true;
            e.is_mtp = true;
            e.layer = layer;
            e.part = strcmp(kind, "gate") == 0 ? EXP_W1 : strcmp(kind, "down") == 0 ? EXP_W2 : EXP_W3;
        }
    } else if (sscanf(name, "mtp.0.ffn_%15[^_]_exps.weight%n", kind, &rest) == 1
               && rest == (int)strlen(name))
    {
        if (strcmp(kind, "gate") == 0 || strcmp(kind, "down") == 0 || strcmp(kind, "up") == 0) {
            e.is_expert = true;
            e.is_mtp = true;
            e.layer = 0;
            e.part = strcmp(kind, "gate") == 0 ? EXP_W1 : strcmp(kind, "down") == 0 ? EXP_W2 : EXP_W3;
        }
    }
    return e;
}

const char *expert_part_name(expert_part p) {
    switch (p) {
        case EXP_W1: return "w1";
        case EXP_W2: return "w2";
        case EXP_W3: return "w3";
        default: die("bad expert part");
    }
    return "";
}

static const name_map top_map[] = {
    { "token_embd.weight",      "embed.weight" },
    { "output_norm.weight",     "norm.weight" },
    { "output.weight",          "head.weight" },
    { "output_hc_base.weight",  "hc_head_base" },
    { "output_hc_fn.weight",    "hc_head_fn" },
    { "output_hc_scale.weight", "hc_head_scale" },
};

static const name_map layer_map[] = {
    { "hc_attn_base.weight",              "hc_attn_base" },
    { "hc_attn_fn.weight",                "hc_attn_fn" },
    { "hc_attn_scale.weight",             "hc_attn_scale" },
    { "hc_ffn_base.weight",               "hc_ffn_base" },
    { "hc_ffn_fn.weight",                 "hc_ffn_fn" },
    { "hc_ffn_scale.weight",              "hc_ffn_scale" },
    { "attn_sinks.weight",                "attn.attn_sink" },
    { "attn_q_a.weight",                  "attn.wq_a.weight" },
    { "attn_q_b.weight",                  "attn.wq_b.weight" },
    { "attn_q_a_norm.weight",             "attn.q_norm.weight" },
    { "attn_kv.weight",                   "attn.wkv.weight" },
    { "attn_kv_a_norm.weight",            "attn.kv_norm.weight" },
    { "attn_output_a.weight",             "attn.wo_a.weight" },
    { "attn_output_b.weight",             "attn.wo_b.weight" },
    { "attn_compressor_ape.weight",       "attn.compressor.ape" },
    { "attn_compressor_kv.weight",        "attn.compressor.wkv.weight" },
    { "attn_compressor_gate.weight",      "attn.compressor.wgate.weight" },
    { "attn_compressor_norm.weight",      "attn.compressor.norm.weight" },
    { "indexer.attn_q_b.weight",          "attn.indexer.wq_b.weight" },
    { "indexer.proj.weight",              "attn.indexer.weights_proj.weight" },
    { "indexer_compressor_ape.weight",    "attn.indexer.compressor.ape" },
    { "indexer_compressor_kv.weight",     "attn.indexer.compressor.wkv.weight" },
    { "indexer_compressor_gate.weight",   "attn.indexer.compressor.wgate.weight" },
    { "indexer_compressor_norm.weight",   "attn.indexer.compressor.norm.weight" },
    { "attn_norm.weight",                 "attn_norm.weight" },
    { "ffn_norm.weight",                  "ffn_norm.weight" },
    { "ffn_gate_shexp.weight",            "ffn.shared_experts.w1.weight" },
    { "ffn_up_shexp.weight",              "ffn.shared_experts.w3.weight" },
    { "ffn_down_shexp.weight",            "ffn.shared_experts.w2.weight" },
    { "ffn_gate_inp.weight",              "ffn.gate.weight" },
    { "exp_probs_b.bias",                 "ffn.gate.bias" },
    { "ffn_gate_tid2eid.weight",          "ffn.gate.tid2eid" },
    { "hc_head_base.weight",              "hc_head_base" },
    { "hc_head_fn.weight",                "hc_head_fn" },
    { "hc_head_scale.weight",             "hc_head_scale" },
    { "main_proj.weight",                 "main_proj.weight" },
    { "main_norm.weight",                 "main_norm.weight" },
    { "norm.weight",                      "norm.weight" },
    { "markov_head.markov_w1.weight",     "markov_head.markov_w1.weight" },
    { "markov_head.markov_w2.weight",     "markov_head.markov_w2.weight" },
    { "confidence_head.proj.weight",      "confidence_head.proj.weight" },
};

/* Map a dspark.N.* or mtp.N.* GGUF tensor name to its HF name (mtp.N.*). */
static bool dspark_hf_name(const char *gguf_name, char *hf_out, size_t hf_sz) {
    /* dspark.main_proj.weight → mtp.0.main_proj.weight */
    const char *prefixes[] = { "dspark.", "mtp.", NULL };
    for (int pi = 0; prefixes[pi]; pi++) {
        size_t plen = strlen(prefixes[pi]);
        if (strncmp(gguf_name, prefixes[pi], plen) != 0) continue;
        const char *rest = gguf_name + plen;
        /* Check for numbered sub-module: dspark.0.xxx or dspark.1.xxx or dspark.2.xxx */
        int idx;
        if (sscanf(rest, "%d.", &idx) == 1 && idx >= 0 && idx <= 2) {
            const char *suffix = strchr(rest, '.') + 1;
            /* Try layer_map suffix mapping first (attn_q_a.weight → attn.wq_a.weight) */
            for (size_t i = 0; i < sizeof(layer_map) / sizeof(layer_map[0]); i++) {
                if (strcmp(suffix, layer_map[i].gguf) == 0) {
                    snprintf(hf_out, hf_sz, "mtp.%d.%s", idx, layer_map[i].hf);
                    return true;
                }
            }
            /* Fallback: identity (GGUF name == HF name under mtp.N.*) */
            snprintf(hf_out, hf_sz, "mtp.%d.%s", idx, suffix);
            return true;
        }
        /* Top-level dspark tensors: dspark.main_proj.weight → mtp.0.main_proj.weight */
        snprintf(hf_out, hf_sz, "mtp.0.%s", rest);
        return true;
    }
    return false;
}

char *hf_name_for_regular(const char *gguf_name) {
    for (size_t i = 0; i < sizeof(top_map) / sizeof(top_map[0]); i++) {
        if (strcmp(gguf_name, top_map[i].gguf) == 0) return xstrdup(top_map[i].hf);
    }
    char hf_buf[512];
    if (dspark_hf_name(gguf_name, hf_buf, sizeof(hf_buf))) {
        return xstrdup(hf_buf);
    }
    int layer = -1;
    const char *p = gguf_name;
    if (sscanf(p, "blk.%d.", &layer) != 1) {
        fprintf(stderr, "error: cannot map GGUF tensor to HF tensor: %s\n", gguf_name);
        exit(1);
    }
    const char *rest = strchr(p + 4, '.');
    if (!rest) die("bad layer tensor name");
    rest++;
    for (size_t i = 0; i < sizeof(layer_map) / sizeof(layer_map[0]); i++) {
        if (strcmp(rest, layer_map[i].gguf) == 0) {
            char buf[512];
            snprintf(buf, sizeof(buf), "layers.%d.%s", layer, layer_map[i].hf);
            return xstrdup(buf);
        }
    }
    fprintf(stderr, "error: cannot map GGUF tensor to HF tensor: %s\n", gguf_name);
    exit(1);
}

static bool is_attention_projection(const char *name) {
    return strstr(name, ".attn_kv.weight") || strstr(name, ".attn_q_a.weight") ||
           strstr(name, ".attn_q_b.weight") || strstr(name, ".attn_output_a.weight") ||
           strstr(name, ".attn_output_b.weight");
}

static bool is_attention_tensor(const char *name) {
    return strstr(name, ".attn") || strstr(name, "attn_") || strstr(name, ".indexer") || strstr(name, "indexer_");
}

static bool is_shared_expert(const char *name) {
    return strstr(name, "_shexp.") != NULL;
}

static bool is_output_tensor(const char *name) {
    return str_starts(name, "output.");
}

int tensor_n_dims(const tensor_meta *t) {
    int n = t->n_dims;
    while (n > 1 && t->ne[n - 1] == 1) n--;
    return n;
}

ds4q_type policy_type(const quant_policy *p, const char *name, const tensor_meta *tmpl) {
    for (int i = 0; i < p->n_overrides; i++) {
        if (strcmp(name, p->overrides[i].prefix) == 0 || str_starts(name, p->overrides[i].prefix)) {
            return p->overrides[i].type;
        }
    }
    expert_tensor e = parse_expert_tensor(name);
    if (e.is_expert) {
        if (e.part == EXP_W1 && p->routed_w1 != DS4Q_TYPE_COUNT) return p->routed_w1;
        if (e.part == EXP_W2 && p->routed_w2 != DS4Q_TYPE_COUNT) return p->routed_w2;
        if (e.part == EXP_W3 && p->routed_w3 != DS4Q_TYPE_COUNT) return p->routed_w3;
        return tmpl->type;
    }
    if (tmpl->type != DS4Q_TYPE_F32 && tmpl->type != DS4Q_TYPE_F16 &&
        tmpl->type != DS4Q_TYPE_BF16 && !ds4q_can_quantize(tmpl->type)) {
        return tmpl->type;
    }
    if (tensor_n_dims(tmpl) <= 1) return tmpl->type;
    if (strcmp(name, "token_embd.weight") == 0 && p->embedding != DS4Q_TYPE_COUNT) return p->embedding;
    if (is_output_tensor(name) && p->output != DS4Q_TYPE_COUNT) return p->output;
    if (is_shared_expert(name) && p->shared != DS4Q_TYPE_COUNT) return p->shared;
    if (is_attention_projection(name) && p->attention_proj != DS4Q_TYPE_COUNT) return p->attention_proj;
    if (is_attention_tensor(name) && p->attention != DS4Q_TYPE_COUNT) return p->attention;
    if (p->dense != DS4Q_TYPE_COUNT) return p->dense;
    return tmpl->type;
}

ds4q_type parse_type(const char *raw) {
    char wanted[64];
    size_t n = 0;
    for (const char *p = raw; *p && n + 1 < sizeof(wanted); p++) {
        if (*p != '-' && *p != '_') wanted[n++] = (char)tolower((unsigned char)*p);
    }
    wanted[n] = '\0';
    if (strcmp(wanted, "copy") == 0 || strcmp(wanted, "template") == 0) return DS4Q_TYPE_COUNT;
    for (int i = 0; i < DS4Q_TYPE_COUNT; i++) {
        char name[64];
        size_t m = 0;
        const char *tn = ds4q_type_name((ds4q_type)i);
        if (!tn) continue;
        for (const char *p = tn; *p && m + 1 < sizeof(name); p++) {
            if (*p != '-' && *p != '_') name[m++] = (char)tolower((unsigned char)*p);
        }
        name[m] = '\0';
        if (strcmp(name, wanted) == 0) return (ds4q_type)i;
    }
    fprintf(stderr, "error: unknown quant type: %s\n", raw);
    exit(1);
}

bool is_quantizable_target(ds4q_type type) {
    return type == DS4Q_TYPE_F32 || type == DS4Q_TYPE_F16 || type == DS4Q_TYPE_BF16 ||
           type == DS4Q_TYPE_CUTLASS_MXFP4 || type == DS4Q_TYPE_MXFP8_LT ||
           type == DS4Q_TYPE_IQ2_XXS_MMQ ||
           ds4q_can_quantize(type);
}

/* Whether a GGUF tensor may be stored as MXFP8_LT (type 41): only the workhorse
 * weights that route through the engine's LT-aware FP8 resolver.  Matches on the
 * base name (second-to-last dotted token) so both blk.N.* and dspark.N.* are
 * covered.  Mirrors _WORKHORSE_BASES in tools/mxfp8_prestore/repack_mxfp8_lt.py
 * byte-for-byte -- keep the two lists in sync. */
bool is_mxfp8_lt_workhorse(const char *name) {
    static const char *bases[] = {
        "attn_q_a", "attn_q_b", "attn_kv", "attn_output_a", "attn_output_b",
        "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
        "output", "main_proj", NULL,
    };
    const char *last = strrchr(name, '.');
    if (!last) return false;               /* need at least one dot (base.suffix) */
    const char *base_start = name;
    for (const char *p = name; p < last; p++) if (*p == '.') base_start = p + 1;
    const size_t base_len = (size_t)(last - base_start);
    for (int i = 0; bases[i]; i++) {
        if (strlen(bases[i]) == base_len && strncmp(base_start, bases[i], base_len) == 0)
            return true;
    }
    return false;
}

/* --format-map: load a prisma_alloc.py manifest ({"tensor.name": "FMT", ...})
 * as exact-name overrides appended to the policy's override list, so the
 * allocator's chosen per-tensor formats drive the output GGUF directly. CLI
 * --tensor-type overrides are appended before this and therefore win. */
void policy_load_format_map(quant_policy *p, const char *path) {
    size_t len = 0;
    char *text = read_file(path, &len);
    json_doc d = json_parse_text(text, len);
    if (d.len < 1 || d.v[0].type != JT_OBJECT) die("--format-map is not a JSON object");
    int loaded = 0;
    for (int i = 1; i < d.len && d.v[i].parent == 0;) {
        const int k = i;
        const int v = i + 1;
        if (v >= d.len || d.v[v].parent != 0 || d.v[v].type != JT_STRING) {
            die("--format-map entries must be \"tensor\": \"TYPE\" strings");
        }
        char *name = json_strdup_tok(&d, k);
        char *fmt = json_strdup_tok(&d, v);
        p->overrides = xrealloc(p->overrides,
                                (size_t)(p->n_overrides + 1) * sizeof(p->overrides[0]));
        p->overrides[p->n_overrides].prefix = name;
        p->overrides[p->n_overrides].type = parse_type(fmt);
        p->n_overrides++;
        loaded++;
        free(fmt);
        i = json_skip(&d, v);
    }
    json_free(&d);
    free(text);
    fprintf(stderr, "format-map: %d tensor overrides loaded from %s\n", loaded, path);
}


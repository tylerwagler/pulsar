#include "dsq_internal.h"

/* =====
 * CLI
 */

typedef struct {
    char *hf_dir;
    char *template_gguf;
    char *out_gguf;
    char *compare_gguf;
    char *compare_tensor;
    char *imatrix_file;
    char *reap_survivors;
    char *dspark_template;
    quant_policy policy;
    int n_experts;
    int n_threads;
    bool dry_run;
    bool overwrite;
    bool imatrix_strict;
    char *mse_probe_file;   /* ds4-spark: --mse-probe OUT.json (MSE oracle) */
    int probe_sample;       /* experts sampled per tensor (0 => default 8) */
} params;

static void usage(const char *argv0) {
    printf("usage: %s --hf DIR --template MODEL.gguf --out OUT.gguf [options]\n", argv0);
    printf("\nDeepSeek V4 Flash/Pro safetensors -> GGUF quantizer in plain C.\n\n");
    printf("options:\n");
    printf("  --hf DIR               Hugging Face model directory with model.safetensors.index.json\n");
    printf("  --template FILE        existing DS4 GGUF used for metadata, tensor order, shapes\n");
    printf("  --out FILE             output GGUF path\n");
    printf("  --compare-gguf FILE    reference GGUF for --compare-tensor, default template\n");
    printf("  --compare-tensor NAME  regenerate one tensor, byte-compare, and exit\n");
    printf("  --overwrite            replace --out if it already exists\n");
    printf("  --dry-run              print output plan without reading HF tensor data\n");
    printf("  --mse-probe FILE       MSE oracle: per-(tensor,format) imatrix-weighted error -> JSON;\n");
    printf("                         requires --imatrix (there is no unweighted probe)\n");
    printf("  --probe-sample N       experts sampled per tensor for --mse-probe (default 8)\n");
    printf("  --imatrix FILE         legacy .dat imatrix from ds4 --imatrix-out; required by every\n");
    printf("                         iq2_xxs target (a tensor without an entry refuses)\n");
    printf("  --imatrix-strict       also fail when a q2_K tensor has no matching vector\n");
    printf("  --experts TYPE         set routed w1/w2/w3 expert tensors to TYPE\n");
    printf("  --routed-w1 TYPE       routed gate expert tensor type\n");
    printf("  --routed-w2 TYPE       routed down expert tensor type\n");
    printf("  --routed-w3 TYPE       routed up expert tensor type\n");
    printf("  --attention-proj TYPE  attn_q/kv/output projection type\n");
    printf("  --attention TYPE       other 2D attention/indexer/compressor type\n");
    printf("  --shared TYPE          shared expert tensor type\n");
    printf("  --embedding TYPE       token embedding type\n");
    printf("  --output TYPE          output.* tensor type\n");
    printf("  --dense TYPE           remaining 2D+ non-routed tensor type\n");
    printf("  --tensor-type PFX=TYPE exact tensor-name or prefix override; may repeat\n");
    printf("  --format-map FILE      JSON manifest of per-tensor formats (prisma_alloc.py output)\n");
    printf("  --dspark-template FILE emit the DSpark drafter into the SAME artifact:\n");
    printf("                         its tensors are appended after the main ones and its\n");
    printf("                         binding KVs merged in. Replaces merge_dspark_gguf.py.\n");
    printf("  --reap-survivors JSON  REAP survivor map: emit an already-pruned artifact\n");
    printf("                         (expert tensors trimmed to survivors, router/bias kept\n");
    printf("                          padded with -1e30 sentinels).\n");
    printf("                          the template must be built with the same map.\n");
    printf("  --n-experts N          routed expert count, default template metadata\n");
    printf("  --threads N            expert worker count, default 8\n");
    printf("\nTYPE examples: f16, f32, bf16, q2_k, iq2_xxs, fp8_e4m3, mxfp4\n");
    printf("               cutlass_mxfp4 (routed experts only: tensor-core B layout, type 40)\n");
}

static char *need_value(int argc, char **argv, int *i, const char *arg) {
    if (++*i >= argc) {
        fprintf(stderr, "error: missing value for %s\n", arg);
        exit(1);
    }
    return argv[*i];
}

static bool file_exists(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    fclose(fp);
    return true;
}

static params parse_args(int argc, char **argv) {
    params p = {0};
    p.policy.routed_w1 = p.policy.routed_w2 = p.policy.routed_w3 = DS4Q_TYPE_COUNT;
    p.policy.attention_proj = p.policy.attention = p.policy.shared = DS4Q_TYPE_COUNT;
    p.policy.embedding = p.policy.output = p.policy.dense = DS4Q_TYPE_COUNT;
    p.n_experts = 0;
    p.n_threads = 8;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else if (strcmp(arg, "--hf") == 0) {
            p.hf_dir = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--template") == 0) {
            p.template_gguf = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--out") == 0) {
            p.out_gguf = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--compare-gguf") == 0) {
            p.compare_gguf = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--compare-tensor") == 0) {
            p.compare_tensor = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--overwrite") == 0) {
            p.overwrite = true;
        } else if (strcmp(arg, "--dry-run") == 0) {
            p.dry_run = true;
        } else if (strcmp(arg, "--mse-probe") == 0) {
            p.mse_probe_file = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--probe-sample") == 0) {
            p.probe_sample = atoi(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--imatrix") == 0) {
            p.imatrix_file = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--reap-survivors") == 0) {
            p.reap_survivors = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--dspark-template") == 0) {
            p.dspark_template = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--imatrix-strict") == 0) {
            p.imatrix_strict = true;
        } else if (strcmp(arg, "--experts") == 0 || strcmp(arg, "--routed") == 0) {
            ds4q_type t = parse_type(need_value(argc, argv, &i, arg));
            p.policy.routed_w1 = p.policy.routed_w2 = p.policy.routed_w3 = t;
        } else if (strcmp(arg, "--routed-w1") == 0 || strcmp(arg, "--routed-gate") == 0) {
            p.policy.routed_w1 = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--routed-w2") == 0 || strcmp(arg, "--routed-down") == 0) {
            p.policy.routed_w2 = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--routed-w3") == 0 || strcmp(arg, "--routed-up") == 0) {
            p.policy.routed_w3 = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--attention-proj") == 0 || strcmp(arg, "--attn-proj") == 0) {
            p.policy.attention_proj = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--attention") == 0) {
            p.policy.attention = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--shared") == 0) {
            p.policy.shared = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--embedding") == 0) {
            p.policy.embedding = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--output") == 0) {
            p.policy.output = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--dense") == 0) {
            p.policy.dense = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--tensor-type") == 0) {
            char *spec = need_value(argc, argv, &i, arg);
            char *eq = strchr(spec, '=');
            if (!eq || eq == spec || !eq[1]) die("bad --tensor-type, expected NAME=TYPE");
            *eq = '\0';
            p.policy.overrides = xrealloc(p.policy.overrides, (size_t)(p.policy.n_overrides + 1) * sizeof(p.policy.overrides[0]));
            p.policy.overrides[p.policy.n_overrides++] = (type_override){ xstrdup(spec), parse_type(eq + 1) };
        } else if (strcmp(arg, "--format-map") == 0) {
            policy_load_format_map(&p.policy, need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--n-experts") == 0) {
            p.n_experts = atoi(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--threads") == 0) {
            p.n_threads = atoi(need_value(argc, argv, &i, arg));
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg);
            exit(1);
        }
    }
    if (!p.hf_dir) die("--hf is required");
    if (!p.template_gguf) die("--template is required");
    if (p.mse_probe_file && !p.imatrix_file)
        die("--mse-probe requires --imatrix: the probe reports imatrix-weighted error and has no unweighted path");
    if (!p.dry_run && !p.compare_tensor && !p.out_gguf) die("--out is required unless --dry-run or --compare-tensor is used");
    if (p.compare_tensor && !p.compare_gguf) p.compare_gguf = p.template_gguf;
    if (p.out_gguf && file_exists(p.out_gguf) && !p.overwrite) die("output exists; use --overwrite");
    return p;
}

static void free_gguf_file(gguf_file *g) {
    free(g->path);
    free(g->kv_raw);
    for (uint64_t i = 0; i < g->n_tensors; i++) free(g->tensors[i].name);
    free(g->tensors);
    hmap_free(&g->tensor_map);
    memset(g, 0, sizeof(*g));
}

static void compare_one_tensor(st_db *db, const gguf_file *tmpl, const output_context *out_ctx,
                               const params *p, const imatrix_store *imatrix, const reap_map *reap) {
    int idx = hmap_get(&tmpl->tensor_map, p->compare_tensor);
    if (idx < 0) {
        fprintf(stderr, "error: tensor not found in template: %s\n", p->compare_tensor);
        exit(1);
    }
    fprintf(stderr, "regenerating %s as %s\n",
            p->compare_tensor, ds4q_type_name(out_ctx->tensors[idx].type));
    byte_buf generated = generate_tensor(db, p->compare_tensor, &tmpl->tensors[idx],
                                         out_ctx->tensors[idx].type, p->n_experts, p->n_threads, imatrix, reap);
    gguf_file ref = load_gguf_metadata(p->compare_gguf);
    byte_buf reference = read_gguf_tensor_data(&ref, p->compare_gguf, p->compare_tensor);
    printf("tensor: %s\n", p->compare_tensor);
    printf("type: %s\n", ds4q_type_name(out_ctx->tensors[idx].type));
    printf("generated_bytes: %zu\n", generated.size);
    printf("reference_bytes: %zu\n", reference.size);
    printf("generated_fnv1a64: %016" PRIx64 "\n", fnv1a64_bytes(generated.data, generated.size));
    printf("reference_fnv1a64: %016" PRIx64 "\n", fnv1a64_bytes(reference.data, reference.size));
    size_t mismatches = 0;
    size_t first = SIZE_MAX;
    const size_t n = generated.size < reference.size ? generated.size : reference.size;
    for (size_t i = 0; i < n; i++) {
        if (generated.data[i] != reference.data[i]) {
            if (first == SIZE_MAX) first = i;
            mismatches++;
        }
    }
    if (generated.size != reference.size) {
        if (first == SIZE_MAX) first = n;
        mismatches += generated.size > reference.size ? generated.size - reference.size : reference.size - generated.size;
    }
    if (!mismatches) {
        printf("byte_compare: OK\n");
    } else {
        printf("byte_compare: FAIL mismatches=%zu first=%zu\n", mismatches, first);
    }
    free(generated.data);
    free(reference.data);
    free_gguf_file(&ref);
}

/* The template's reap.* arrays only carry per-layer counts and policies, so two
 * survivor maps that keep the same NUMBER of experts per layer -- but different
 * ones -- shape an identical template and pass the expert-count cross-check,
 * while routing every token to a different expert. Nothing downstream notices:
 * the artifact loads, generates, and is quietly wrong. The template therefore
 * records the hash of the map that shaped it, and this refuses any other. */
static void check_reap_binding(const gguf_file *tmpl, const char *map_path) {
    if (tmpl->reap_sha256 && !map_path) {
        fprintf(stderr, "error: template is REAP-shaped (reap.survivors.sha256 %s)"
                " but no --reap-survivors was given\n", tmpl->reap_sha256);
        exit(1);
    }
    if (!map_path) return;
    if (!tmpl->reap_sha256) {
        if (tmpl->reap_enabled)
            die("template has reap.enabled but no reap.survivors.sha256; rebuild it "
                "with build_main_template.py --reap-survivors so the two are bound");
        die("--reap-survivors given but the template is not REAP-shaped");
    }
    char *have = sha256_file_hex(map_path);
    if (strcmp(have, tmpl->reap_sha256) != 0) {
        fprintf(stderr,
                "error: survivor map does not match the one this template was built from\n"
                "  template %s\n  supplied %s (%s)\n",
                tmpl->reap_sha256, have, map_path);
        exit(1);
    }
    fprintf(stderr, "reap: survivor map matches the template (sha256 %.16s...)\n", have);
    free(have);
}

int main(int argc, char **argv) {
    params p = parse_args(argc, argv);
    imatrix_store imatrix = {0};
    if (p.imatrix_file) imatrix_load(&imatrix, p.imatrix_file, p.imatrix_strict);
    reap_map reap = {0};
    if (p.reap_survivors) reap_load(&reap, p.reap_survivors);

    gguf_file tmpl = load_gguf_metadata(p.template_gguf);
    /* Before build_output_context: the drafter's tensors have to be part of the
     * plan so they get offsets in the same data region. */
    if (p.dspark_template) gguf_append_dspark(&tmpl, p.dspark_template);
    check_reap_binding(&tmpl, p.reap_survivors);
    if (p.n_experts <= 0) {
        if (tmpl.n_experts > 0) {
            p.n_experts = tmpl.n_experts;
            fprintf(stderr, "using %d routed experts from template metadata\n", p.n_experts);
        } else {
            p.n_experts = 256;
            fprintf(stderr, "warning: template has no deepseek4.expert_count; using Flash default %d routed experts\n", p.n_experts);
        }
    } else {
        fprintf(stderr, "using %d routed experts from --n-experts\n", p.n_experts);
    }
    output_context out_ctx = build_output_context(&tmpl, &p.policy, &imatrix);
    print_plan(&tmpl, &out_ctx);
    if (p.dry_run) return 0;

    st_db db;
    db_open(&db, p.hf_dir);
    if (p.mse_probe_file) {
        run_mse_probe(&db, &tmpl, &imatrix, p.n_experts, p.n_threads,
                      p.probe_sample, p.mse_probe_file);
        db_close(&db); imatrix_free(&imatrix); free_gguf_file(&tmpl);
        free(out_ctx.tensors);
        return 0;
    }
    if (p.compare_tensor) {
        compare_one_tensor(&db, &tmpl, &out_ctx, &p, &imatrix, &reap);
        db_close(&db);
        imatrix_free(&imatrix);
        free_gguf_file(&tmpl);
        free(out_ctx.tensors);
        return 0;
    }
    write_full_gguf(&db, &tmpl, &out_ctx, p.out_gguf, p.n_experts, p.n_threads, &imatrix, &reap);
    fprintf(stderr, "wrote %s\n", p.out_gguf);

    db_close(&db);
    imatrix_free(&imatrix);
    free_gguf_file(&tmpl);
    free(out_ctx.tensors);
    for (int i = 0; i < p.policy.n_overrides; i++) free(p.policy.overrides[i].prefix);
    free(p.policy.overrides);
    return 0;
}

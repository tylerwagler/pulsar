#include "pulsar_engine_internal.h"
#include "exl3_trellis.h"
#include "spec_internal.h"
#include "tp/pulsar_tp.h"
#include "tp/pulsar_tp_gpu.h"
#include "lib/qwen_tokenizer.h"


int pulsar_engine::routed_quant_bits() {
    auto *e = this;
    return e ? e->family->quant_bits(e) : 0;
}


bool pulsar_engine::has_dspark() {
    auto *e = this;
    return e && e->dspark_ready;
}

int pulsar_engine_dspark_draft_tokens(pulsar_engine *e) {
    return e->has_dspark() ? e->dspark_draft_tokens : 0;
}


const pulsar_tokens *pulsar_session::tokens() {
    auto *s = this;
    return s ? &s->checkpoint : NULL;
}




int pulsar_dump_text_tokenization(const char *model_path, const char *text, FILE *fp) {
    pulsar_model model;
    pulsar_vocab vocab;
    token_vec tokens = {0};

    if (!fp) fp = stdout;
    if (!model_open(&model, model_path, false)) {
        model_close(&model);
        return 1;
    }
    vocab.vocab_load(&model);
    vocab.tokenize_rendered_chat_vocab(text ? text : "", &tokens);

    dump_tokens_fp(fp, &vocab, &tokens);
    token_vec_free(&tokens);
    vocab.vocab_free();
    model_close(&model);
    return 0;
}


static bool imatrix_read_text_file(const char *path, char **out, size_t *len_out) {
    *out = NULL;
    *len_out = 0;
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "pulsar: failed to stat imatrix dataset %s: %s\n", path, strerror(errno));
        return false;
    }
    if (st.st_size < 0 || (uint64_t)st.st_size > SIZE_MAX - 1) {
        fprintf(stderr, "pulsar: imatrix dataset is too large: %s\n", path);
        return false;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "pulsar: failed to open imatrix dataset %s: %s\n", path, strerror(errno));
        return false;
    }
    size_t n = (size_t)st.st_size;
    char *buf = (char *)xmalloc(n + 1);
    if (n != 0 && fread(buf, 1, n, fp) != n) {
        fprintf(stderr, "pulsar: failed to read imatrix dataset %s\n", path);
        fclose(fp);
        free(buf);
        return false;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "pulsar: failed to close imatrix dataset %s: %s\n", path, strerror(errno));
        free(buf);
        return false;
    }
    buf[n] = '\0';
    *out = buf;
    *len_out = n;
    return true;
}


static char *imatrix_trim_block(char *p, char *end) {
    while (p < end && isspace((unsigned char)*p)) p++;
    while (end > p && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return p;
}


/* The prompt delimiter is a LINE: it matches only at the start of the dataset
 * or right after a '\n', so a prompt that quotes the marker mid-line is not
 * split there (upstream antirez/ds4 0a62b396). */
static char *imatrix_find_marker(char *dataset, char *cursor, const char *marker) {
    const size_t marker_len = strlen(marker);
    char *p = cursor;
    while ((p = strstr(p, marker)) != NULL) {
        if (p == dataset || p[-1] == '\n') return p;
        p += marker_len;
    }
    return NULL;
}


int pulsar_engine::collect_imatrix(const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens) {
    auto *e = this;
    if (!e || !dataset_path || !output_path) return 1;
    if (e->backend != PULSAR_BACKEND_CUDA || !e->gpu_ready) {
        fprintf(stderr, "pulsar: imatrix collection requires a CUDA device (GPU init failed)\n");
        return 1;
    }
    if (ctx_size <= 0) ctx_size = 32768;

    char *dataset = NULL;
    size_t dataset_len = 0;
    if (!imatrix_read_text_file(dataset_path, &dataset, &dataset_len)) return 1;

    /* L284 P15: the dataset walk and the caps are the core's; the family runs one prompt through its forward */
    const pulsar_family_imatrix *fi = e->family->imatrix;
    void *collection = fi->begin(e, dataset_path, ctx_size);
    if (!collection) {
        free(dataset);
        return 1;
    }
    bool ok = true;

    int prompts_done = 0;
    int tokens_done = 0;
    char *cursor = dataset;
    const char *marker_lit = "===== DS4_IMATRIX_PROMPT";
    while (*cursor) {
        char *start = cursor;
        char *marker = imatrix_find_marker(dataset, cursor, marker_lit);
        if (marker) {
            char *nl = strchr(marker, '\n');
            if (!nl) break;
            start = nl + 1;
        } else if (prompts_done != 0) {
            break;
        }

        char *next = imatrix_find_marker(dataset, start, marker_lit);
        char *end = next ? next : dataset + dataset_len;
        char saved = *end;
        char *prompt_text = imatrix_trim_block(start, end);
        if (prompt_text[0] != '\0') {
            token_vec prompt = {0};
            pulsar_tokenize_rendered_chat(e, prompt_text, &prompt);
            if (prompt.len > ctx_size) prompt.len = ctx_size;
            if (max_tokens > 0 && prompt.len > max_tokens - tokens_done) {
                prompt.len = max_tokens - tokens_done;
            }
            if (prompt.len > 0) {
                ok = fi->prompt(e, collection, &prompt);
                if (!ok) {
                    fprintf(stderr, "pulsar: imatrix prefill failed at prompt %d\n", prompts_done + 1);
                    token_vec_free(&prompt);
                    *end = saved;
                    break;
                }
                prompts_done++;
                tokens_done += prompt.len;
                if (prompts_done % 10 == 0) {
                    fprintf(stderr,
                            "pulsar: imatrix prompts=%d tokens=%d routes=%llu\r",
                            prompts_done,
                            tokens_done,
                            (unsigned long long)fi->routes(collection));
                    fflush(stderr);
                }
            }
            token_vec_free(&prompt);
        }
        *end = saved;
        if (!next) break;
        cursor = next;
        if (max_prompts > 0 && prompts_done >= max_prompts) break;
        if (max_tokens > 0 && tokens_done >= max_tokens) break;
    }
    fputc('\n', stderr);

    if (ok) {
        ok = fi->save(e, collection, output_path);
        if (ok) {
            fprintf(stderr,
                    "pulsar: wrote imatrix %s from %d prompts, %d tokens, %llu routed expert observations\n",
                    output_path,
                    prompts_done,
                    tokens_done,
                    (unsigned long long)fi->routes(collection));
        }
    }

    fi->end(e, collection);
    free(dataset);
    return ok ? 0 : 1;
}


/* Register every mapping of a model with the staged-fd route -- the only route
 * onto the GPU here, since cudaHostRegister reports "operation not supported" on
 * GB10.  A GGUF has one mapping; a safetensors checkpoint has one per layer, and
 * the route is per-mapping, so registering only the first shard would leave 47
 * layers unreadable. */
static void register_model_fds(const pulsar_model *m) {
    if (m->n_shards) {
        for (uint64_t i = 0; i < m->n_shards; i++) {
            (void)pulsar_gpu_set_model_fd_for_map(m->shard_fd[i], m->shard_map[i]);
        }
    } else {
        (void)pulsar_gpu_set_model_fd_for_map(m->fd, m->map);
    }
}

/* L241 4g-2 expert tensor-parallel: this rank's half of every expert of one layer's three routed stacks --
 * gate/up by intermediate ROWS (OUT), down by intermediate (input) COLUMNS (IN), the same owned range for all
 * three, so the SwiGLU halves line up and the rank's down output is a partial the FFN exchange sums.  Both ranks
 * do identical work (every selected expert, half width), so there is no skew by construction.  How a half is
 * built is the format's (tp_slice.cpp: a CUTLASS MXFP4 stack's half; another format refuses there by name). */
static bool tp_declare_expert_half(pulsar_tp_plan *p, pulsar_model *m, const pulsar_layer_weights *L, int rank,
                                   uint32_t nr) {
    if (!L->ffn_gate_exps || !L->ffn_up_exps || !L->ffn_down_exps) return true;   /* no routed experts */
    const uint64_t mid = L->ffn_gate_exps->dim[1];
    uint32_t lo = 0, hi = 0;
    if (L->ffn_down_exps->dim[0] != mid || L->ffn_up_exps->dim[1] != mid || !pulsar_tp_owned_range(rank, nr, (uint32_t)mid, &lo, &hi)) {
        fprintf(stderr, "pulsar: expert tensor-parallel: the routed stacks' intermediate %llu does not split over %u "
                        "ranks -- refusing\n", (unsigned long long)mid, nr);
        return false;
    }
    return pulsar_tp_plan_add1(p, m, L->ffn_gate_exps, PULSAR_TP_AXIS_OUT, lo, hi) &&
           pulsar_tp_plan_add1(p, m, L->ffn_up_exps, PULSAR_TP_AXIS_OUT, lo, hi) &&
           pulsar_tp_plan_add1(p, m, L->ffn_down_exps, PULSAR_TP_AXIS_IN, lo, hi);
}

/* L241 4g-2: the shared expert, split like a Megatron MLP -- gate/up by OUTPUT rows (column-parallel: this rank's
 * half of the intermediate), down by INPUT columns (row-parallel: the matching half of the reduction).  The rank's
 * shared output is then a PARTIAL that rides the FFN's existing exchange with the routed partial, so the split
 * costs no exchange of its own.  The range is the one authority every split shares (pulsar_tp_owned_range over
 * the shared width).  For the target's layers and the drafter's.  (An MXFP8 down's K-half is keyed by the engine
 * and the tensor OBJECT, never its abs_offset: a safetensors checkpoint is one shard per layer with identical
 * layouts, so every layer's down projection sits at the SAME offset -- tp_slice.cpp.) */
static bool tp_declare_shared_split(pulsar_tp_plan *p, pulsar_model *m, const pulsar_layer_weights *L, int rank,
                                    uint32_t nr) {
    if (!L->ffn_gate_shexp || !L->ffn_up_shexp || !L->ffn_down_shexp) return false;
    uint32_t lo = 0, hi = 0;
    if (!pulsar_tp_owned_range(rank, nr, (uint32_t)L->ffn_gate_shexp->dim[1], &lo, &hi) || hi <= lo) return false;
    return pulsar_tp_plan_add1(p, m, L->ffn_gate_shexp, PULSAR_TP_AXIS_OUT, lo, hi) &&
           pulsar_tp_plan_add1(p, m, L->ffn_up_shexp, PULSAR_TP_AXIS_OUT, lo, hi) &&
           pulsar_tp_plan_add1(p, m, L->ffn_down_shexp, PULSAR_TP_AXIS_IN, lo, hi);
}

/* The DeepSeek family's load (family.h pulsar_family::load): the tokenizer, the
 * shape profile and its metadata validation, the expert overlay, and the
 * target / drafter / vision weight binding -- the steps pulsar_engine::open ran
 * inline until L251, moved here verbatim and in the same order.  Every refusal
 * prints its reason; the caller tears the engine down. */
/* L272: the loader's one failure policy -- each stage below reports every refusal it finds
 * (pulsar_load_refuse) and the load stops at the stage's end, so a bad artifact fails
 * pulsar_engine_open instead of exiting the process. */
static bool ds4_load_stage_ok(const char *stage) {
    const uint32_t n = pulsar_load_refusals();
    if (n == 0) return true;
    fprintf(stderr, "pulsar: deepseek4: %u refusal(s) in %s -- the model does not load\n", n, stage);
    return false;
}

bool pulsar_ds4_family_load(pulsar_engine *e, const pulsar_engine_options *opt) {
    pulsar_load_refusals_reset();
    if (!opt->inspect_only) e->vocab.vocab_load(&e->model);
    if (!ds4_load_stage_ok("the tokenizer")) return false;
    config_validate_model(&e->model);
    if (!ds4_load_stage_ok("the configuration")) return false;
    if (opt->expert_overlay && opt->expert_overlay[0]) {
        const char *sep = strrchr(opt->expert_overlay, ':');
        if (!sep || sep == opt->expert_overlay || !sep[1]) {
            fprintf(stderr, "pulsar: --expert-overlay expects FILE:PREFIX (e.g. donor.gguf:blk.17.)\n");
            return false;
        }
        char overlay_path[4096];
        const size_t path_len = (size_t)(sep - opt->expert_overlay);
        if (path_len >= sizeof(overlay_path)) {
            fprintf(stderr, "pulsar: --expert-overlay path is too long\n");
            return false;
        }
        memcpy(overlay_path, opt->expert_overlay, path_len);
        overlay_path[path_len] = '\0';
        if (!model_open(&e->overlay_model, overlay_path, pulsar_backend_uses_graph(opt->backend))) {
            model_close(&e->overlay_model);
            fprintf(stderr, "pulsar: the --expert-overlay donor %s does not open -- refusing\n", overlay_path);
            return false;
        }
        e->overlay_ready = true;
        /* PREFIX is a comma-separated list so several layers can be swapped
         * in one run (e.g. compose "anchor + candidate" from a cheap base
         * without materializing the combined model as a file). */
        char prefixes[2048];
        const size_t plist_len = strlen(sep + 1);
        if (plist_len >= sizeof(prefixes)) {
            fprintf(stderr, "pulsar: --expert-overlay prefix list is too long\n");
            return false;
        }
        memcpy(prefixes, sep + 1, plist_len + 1);
        uint32_t swapped = 0;
        for (char *p = strtok(prefixes, ","); p; p = strtok(NULL, ",")) {
            const uint32_t n = model_apply_expert_overlay(&e->model, &e->overlay_model, p);
            if (!ds4_load_stage_ok("the expert overlay")) return false;
            if (n == 0) {
                fprintf(stderr, "pulsar: --expert-overlay prefix '%s' matched no routed-expert tensors\n",
                        p);
                return false;
            }
            swapped += n;
        }
        fprintf(stderr, "pulsar: expert overlay: %u tensors swapped in from %s (prefixes %s)\n",
                swapped, overlay_path, sep + 1);
    }
    weights_bind(&e->weights, &e->model);
    if (!ds4_load_stage_ok("the weights")) return false;
    /* the drafter binds before the inspect-only exit so --inspect proves the
     * whole artifact binds, drafter included */
    if (!opt->dspark_disable && model_find_tensor(&e->model, "dspark.main_proj.weight")) {
        /* Drafter merged into the main GGUF: bind from the main model and
         * alias dspark_model to it by value (same map/fd; every dspark call
         * site reads e->dspark_model, and close is guarded on dspark_external
         * so the shared mapping is only torn down once). */
        dspark_weights_bind(&e->dspark_weights, &e->model);
        if (!ds4_load_stage_ok("the DSpark drafter")) return false;
        e->dspark_model = e->model;
        e->dspark_external = false;
        e->dspark_ready = true;
        e->drafter_ops = &k_dspark_drafter;   /* L272 P1: the drafter behind the round API */
        fprintf(stderr, "pulsar: DSpark drafter found in model (draft=%d, markov_w2 %s)\n",
                e->dspark_draft_tokens, tensor_type_name(e->dspark_weights.markov_w2->type));
    }
    /* Vision-Exp tower: bound and layout-validated here so a wrong or
     * half-present vision stack refuses at load rather than at first image.
     * Absent tower is normal for text-only artifacts and simply leaves
     * vision_ready false; the image path is refused until it is true. */
    const bool vision = vision_weights_bind(&e->vision_weights, &e->model);
    if (!ds4_load_stage_ok("the vision tower")) return false;
    if (vision) {
        e->vision_ready = true;
        fprintf(stderr, "pulsar: Vision-Exp tower bound (%u blocks, dim %u, %u heads, inter %u, "
                "patch %u, aligner %ux%d -> %u)\n",
                e->vision_weights.n_layers, (unsigned)PULSAR_VISION_DIM,
                (unsigned)PULSAR_VISION_HEADS, (unsigned)PULSAR_VISION_INTER,
                (unsigned)PULSAR_VISION_PATCH,
                (unsigned)PULSAR_VISION_DOWNSAMPLE, (unsigned)PULSAR_VISION_DOWNSAMPLE,
                (unsigned)PULSAR_N_EMBD);
    }
    /* The layer plan: one kind for every layer.  The per-layer attention mode
     * (SWA / compressed / indexed) stays in pulsar_attn_layout, its authority. */
    e->plan.n_layer = PULSAR_N_LAYER;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) e->plan.kind[il] = PULSAR_LAYER_DS4_BLOCK;
    return true;
}

/* The DeepSeek family's device-side step of open (family.h after_gpu): the MoE tier announce and the rank's
 * attention groups.  Runs after the GPU, the transport and the core's TP plan (pulsar_tp_plan_run) are up. */
/* Slice 4g (L241): the attention OUTPUT GROUPS this rank owns, from the one range authority every split shares.
 * The unit is the group (8 heads and one LoRA-down block each), never the head: a rank's heads are whole groups,
 * so its attn_q_b rows and its attn_output_a rows are contiguous and 128-row aligned.  One box: [0, n).  The rank
 * is the model's (set at open from the options; the transport asserts the same), so record mode reads it too. */
static bool ds4_tp_groups(pulsar_engine *e) {
    const int tp_rk = e->model.tp_rank;
    const uint32_t tp_nr = e->model.tp_n_ranks ? e->model.tp_n_ranks : 1u;
    if (!pulsar_tp_owned_range(tp_rk, tp_nr, PULSAR_N_OUT_GROUP, &e->tp_group_lo, &e->tp_group_hi) ||
        e->tp_group_hi <= e->tp_group_lo) {
        fprintf(stderr, "pulsar: TP rank %d/%u owns no attention output group (%u groups per "
                        "layer; a group of more than %u ranks cannot split attention) -- refusing\n",
                tp_rk, tp_nr, (unsigned)PULSAR_N_OUT_GROUP, (unsigned)PULSAR_N_OUT_GROUP);
        return false;
    }
    return true;
}

/* The DeepSeek family's tensor-parallel slices (pulsar_family::tp_slices, L272 P4b): on each layer the owned
 * attention row slices are registered with the backend, once; the attention block then addresses them by
 * offset like any other weight.  Every slice goes through the core's operations (tp_slice.cpp). */
bool pulsar_ds4_tp_slices(pulsar_engine *e, pulsar_tp_plan *plan) {
    if (!ds4_tp_groups(e)) return false;
    const int tp_rk = e->model.tp_rank;
    const uint32_t tp_nr = e->model.tp_n_ranks ? e->model.tp_n_ranks : 1u;
    const uint32_t group_heads = PULSAR_N_HEAD / PULSAR_N_OUT_GROUP;
    const uint64_t q_lo = (uint64_t)e->tp_group_lo * group_heads * PULSAR_N_HEAD_DIM;
    const uint64_t q_hi = (uint64_t)e->tp_group_hi * group_heads * PULSAR_N_HEAD_DIM;
    const uint64_t a_lo = (uint64_t)e->tp_group_lo * PULSAR_N_LORA_O;
    const uint64_t a_hi = (uint64_t)e->tp_group_hi * PULSAR_N_LORA_O;
    /* the attention head split, for a target layer and a drafter block alike: attn_q_b's and attn_output_a's rows
     * of the owned groups, attn_output_b's K-half over the same group columns (4g-2) */
    auto attention = [&](pulsar_model *m, const pulsar_layer_weights *L) {
        return L->attn_q_a && L->attn_q_b && L->attn_output_a && L->attn_output_b &&
               pulsar_tp_plan_add1(plan, m, L->attn_q_b, PULSAR_TP_AXIS_OUT, q_lo, q_hi) &&
               pulsar_tp_plan_add1(plan, m, L->attn_output_a, PULSAR_TP_AXIS_OUT, a_lo, a_hi) &&
               pulsar_tp_plan_add1(plan, m, L->attn_output_b, PULSAR_TP_AXIS_IN, a_lo, a_hi);
    };
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const pulsar_layer_weights *L = &e->weights.layer[il];
        if (!attention(&e->model, L)) {
            fprintf(stderr, "pulsar: layer %u: the attention projections do not split -- refusing\n", il);
            return false;
        }
        /* 4g-3: the attention INPUT side is row-split too -- q_a, kv, the compressor and indexer projections, each
         * rank its floor row range, gathered in one row-lane exchange per layer (tp_attn_input_split) */
        for (const pulsar_tensor *w : { L->attn_q_a, L->attn_kv, L->attn_compressor_kv, L->attn_compressor_gate,
                                        L->indexer_compressor_kv, L->indexer_compressor_gate, L->indexer_proj }) {
            if (!w) continue;
            const uint64_t out = w->ndim >= 2 ? w->dim[w->ndim - 1] : 0;
            uint32_t lo = 0, hi = 0;
            if (!pulsar_tp_owned_range(tp_rk, tp_nr, (uint32_t)out, &lo, &hi) ||
                !pulsar_tp_plan_add1(plan, &e->model, w, PULSAR_TP_AXIS_OUT, lo, hi)) {
                fprintf(stderr, "pulsar: layer %u: %.*s does not row-split -- refusing\n", il, (int)w->name.len,
                        w->name.ptr);
                return false;
            }
        }
        if (!tp_declare_shared_split(plan, &e->model, L, tp_rk, tp_nr) ||
            !tp_declare_expert_half(plan, &e->model, L, tp_rk, tp_nr)) {
            fprintf(stderr, "pulsar: layer %u: the experts do not split -- refusing\n", il);
            return false;
        }
    }
    /* 4g-1b: the drafter's blocks split exactly like a target layer's, over the drafter's own mapping */
    for (uint32_t dl = 0; e->dspark_ready && dl < 3u; dl++) {
        const pulsar_layer_weights *DL = &e->dspark_weights.layer[dl];
        if (!tp_declare_expert_half(plan, &e->dspark_model, DL, tp_rk, tp_nr) ||
            !tp_declare_shared_split(plan, &e->dspark_model, DL, tp_rk, tp_nr) || !attention(&e->dspark_model, DL)) {
            fprintf(stderr, "pulsar: drafter block %u does not split -- refusing\n", dl);
            return false;
        }
    }
    const pulsar_layer_weights *L0 = &e->weights.layer[0];
    uint32_t sx_lo = 0, sx_hi = 0;
    (void)pulsar_tp_owned_range(tp_rk, tp_nr, (uint32_t)L0->ffn_gate_shexp->dim[1], &sx_lo, &sx_hi);
    fprintf(stderr, "pulsar: TP rank %d/%u owns attention output groups [%u,%u) of %u = heads [%u,%u) and "
                    "shared-expert intermediate [%u,%u) of %u (%u layers + %u drafter blocks)\n",
            tp_rk, tp_nr, e->tp_group_lo, e->tp_group_hi, (unsigned)PULSAR_N_OUT_GROUP,
            e->tp_group_lo * group_heads, e->tp_group_hi * group_heads, sx_lo, sx_hi,
            (unsigned)L0->ffn_gate_shexp->dim[1], (unsigned)PULSAR_N_LAYER, e->dspark_ready ? 3u : 0u);
    return true;
}

bool pulsar_ds4_family_after_gpu(pulsar_engine *e) {
    /* One MoE-tier boot line so a silent slow tier is no longer silent:
     * the resolved expert weight types per layer (grouped-CUTLASS type-40
     * on both sides vs the per-expert MMQ/mixed path).  Reuses the
     * bound-layer expert tensors; log-only, runs once at open on every GPU
     * serve.  PULSAR_DUMP_MOE_TYPES=1 additionally prints every routed
     * layer's gate/down type pair -- that census is what established the
     * artifact is 40 and 43 only. */
    {
        /* Count the tier PER LAYER.  Reporting the first bound layer's type
         * as "the" tier is actively misleading on a heterogeneous model:
         * v5mx has only 5 uniform-MXFP4 layers, so the old line printed
         * "per-expert-tiled" while a dozen layers were in fact running the
         * grouped CUTLASS path (2026-07-21 profile).  This observability
         * exists to catch a SILENT fall to a slow tier, so it has to be
         * truthful about the mix or it defeats its own purpose. */
        uint32_t n_grouped = 0, n_tiled = 0, n_routed = 0;
        const pulsar_layer_weights *ml = NULL;
        for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
            const pulsar_layer_weights *l = &e->weights.layer[il];
            if (!l->ffn_gate_exps || !l->ffn_down_exps) continue;
            if (!ml) ml = l;                    /* first routed layer, for the type sample */
            n_routed++;
            /* The grouped/GEMV CUTLASS dispatch is entered only when BOTH
             * gate and down experts are type-40 (see the moe.cu batch
             * predicate); any other mix takes the per-expert tiled path. */
            if (l->ffn_gate_exps->type == PULSAR_TENSOR_CUTLASS_MXFP4 &&
                l->ffn_down_exps->type == PULSAR_TENSOR_CUTLASS_MXFP4) n_grouped++;
            else n_tiled++;
        }
        if (ml) {
            const uint32_t gt = ml->ffn_gate_exps->type;
            const uint32_t dt = ml->ffn_down_exps->type;
            /* The "mxfp4 tile=NT%u" field is gone with the type-39 gate/up
             * kernel it described: nothing reads that tile width now that
             * every MXFP4 layer goes through CUTLASS. */
            fprintf(stderr,
                    "pulsar: MoE expert tier: %u/%u layers grouped-CUTLASS, %u/%u per-expert-tiled "
                    "(first routed layer gate=%s(%u) down=%s(%u))\n",
                    n_grouped, n_routed, n_tiled, n_routed,
                    tensor_type_name(gt), gt, tensor_type_name(dt), dt);
        }
    }

    /* the rank's attention output groups (one GPU: all of them); the slices are the core's plan (L272 P4b) */
    return ds4_tp_groups(e);
}

int pulsar_engine::open(pulsar_engine **out, const pulsar_engine_options *opt) {
    pulsar_engine *e = (pulsar_engine *)xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->dspark_model.fd = -1;
    e->backend = opt->backend;
    e->prefill_chunk = opt->prefill_chunk;
    /* Slice 4f (L237): the rank this process loads the model FOR, decided from
     * the options before any weight is staged -- the transport is created
     * after the load, and on GB10 staging IS residency (a rank does not stage
     * the stored expert stacks).  Rank 0 of 1 when the pair is off.  The
     * transport's rank is asserted equal once it exists, so the two readers of
     * this fact cannot disagree.  An expert overlay swaps expert stacks from a donor file and
     * is staged by its own path, which has no half-expert notion: refused
     * under TP rather than staged whole on every rank (rule 9). */
    int tp_rank_at_load = 0;
    uint32_t tp_n_ranks_at_load = 1;
    if (opt->tp_peers) {
        if (opt->tp_rank < 0 || opt->tp_nranks < 2 || opt->tp_rank >= opt->tp_nranks) {
            fprintf(stderr, "pulsar: n-way TP needs --tp-rank R and --tp-nranks N "
                            "with 0 <= R < N (got rank=%d nranks=%d)\n",
                    opt->tp_rank, opt->tp_nranks);
            free(e);
            *out = NULL;
            return 1;
        }
        tp_rank_at_load = opt->tp_rank;
        tp_n_ranks_at_load = (uint32_t)opt->tp_nranks;
    } else if (opt->tp_role != 0) {
        tp_rank_at_load = opt->tp_role == 2 ? 1 : 0;
        tp_n_ranks_at_load = 2;
    }
    if (tp_n_ranks_at_load > 1 && opt->expert_overlay) {
        fprintf(stderr, "pulsar: --expert-overlay is not supported under tensor parallelism "
                        "(the overlay's experts have no owner) -- refusing\n");
        free(e);
        *out = NULL;
        return 1;
    }
    /* Default draft depth 3: the measured v5mx optimum (2026-07-17 k-sweep on
     * the shipped ds4flash build at the tau=0.25 conf-sched default, quench
     * disarmed, conf-sched trimming active). k=3 beats k=5 by +15% structured
     * to +32% prose served decode; distribution-preserving (exact verify) —
     * byte-identical on structured, near-tie-equivalent on greedy prose (the
     * verify-width change flips ~1-ULP argmax ties, same class as yield-quench).
     * The DSpark drafter forward is autoregressive, so its cost scales with the
     * chain length ON TOP of the verify rows — ms/accepted-token stays flat
     * ~41-46 ms across k, i.e. depth never amortizes, so shallower wins.
     * The prior default 5 was a compact-model figure (2026-07-09, conf3 head,
     * tau 0.35) that does not hold on shipped v5mx at the tau=0.25 default. */
    e->dspark_draft_tokens = opt->dspark_draft_tokens > 0 ? opt->dspark_draft_tokens : 3;
    if (e->dspark_draft_tokens > 16) e->dspark_draft_tokens = 16;
    if ((opt->directional_steering_attn != 0.0f || opt->directional_steering_ffn != 0.0f) &&
        (!opt->directional_steering_file || !opt->directional_steering_file[0]))
    {
        fprintf(stderr, "pulsar: directional steering needs --dir-steering-file\n");
        free(e);
        *out = NULL;
        return 1;
    }
    if (opt->directional_steering_file && opt->directional_steering_file[0]) {
        e->directional_steering_file = pulsar_strdup(opt->directional_steering_file);
        e->directional_steering_attn_scale = opt->directional_steering_attn;
        e->directional_steering_ffn_scale = opt->directional_steering_ffn;
    }
    pulsar_acquire_instance_lock();

    const bool graph_backend = pulsar_backend_uses_graph(opt->backend);
    if (graph_backend) pulsar_linux_graph_backend_set_oom_score(opt->backend);
    if (!model_open(&e->model, opt->model_path, graph_backend)) {   /* L278: refused by name, never an exit */
        e->destroy();
        *out = NULL;
        return 1;
    }
    /* Slice 4f: the model knows the rank it is loaded for from here on; the
     * merged drafter aliases e->model by value below and inherits it. */
    e->model.tp_rank = tp_rank_at_load;
    e->model.tp_n_ranks = tp_n_ranks_at_load;
    e->family = pulsar_family_for_model(&e->model);
    if (!e->family) {
        e->destroy();
        *out = NULL;
        return 1;
    }
    /* A feature the family does not declare is refused before anything of it
     * is built: TP here, an overlay in the family's own load. */
    if (tp_n_ranks_at_load > 1 && !pulsar_family_require(e, PULSAR_FAMILY_CAP_TP, "tensor parallelism")) {
        e->destroy();
        *out = NULL;
        return 1;
    }
    if (!e->family->load(e, opt)) {
        e->destroy();
        *out = NULL;
        return 1;
    }
    fprintf(stderr, "pulsar: model family %s (%s): %u layers\n",
            e->family->name, e->family->arch, e->plan.n_layer);

    /* L272 P4b: the rank's TP plan -- declared by the family, its operations and the residency rule the core's --
     * before staging and before the inspect-only exit (tests/tp_plan_test.cpp records it from there) */
    if (tp_n_ranks_at_load > 1 && !pulsar_tp_plan_build(e)) {
        e->destroy();
        *out = NULL;
        return 1;
    }
    /* the family's load binds everything (drafter and vision tower included)
     * before the inspect-only exit: --inspect proves the whole artifact binds */
    if (opt->inspect_only) {
        *out = e;
        return 0;
    }

    if (graph_backend) {
        e->gpu_ready = pulsar_gpu_init() != 0;
        if (!e->gpu_ready) {
            fprintf(stderr, "pulsar: %s backend unavailable; aborting startup\n",
                    pulsar_backend_name(e->backend));
            e->destroy();
            *out = NULL;
            return 1;
        }
        (void)pulsar_gpu_set_model_fd(e->model.fd);
        const int model_map_ok =
            pulsar_gpu_set_model_map_range(e->model.map,
                                        e->model.size,
                                        e->model.tensor_data_pos,
                                        e->model.size - e->model.tensor_data_pos,
                                        e->model.max_tensor_bytes);
        if (!model_map_ok) {
            fprintf(stderr,
                    "pulsar: %s failed to map model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    pulsar_backend_name(e->backend));
            e->destroy();
            *out = NULL;
            return 1;
        }
        if (e->dspark_ready && e->dspark_external &&
            !pulsar_gpu_set_model_map_range(e->dspark_model.map,
                                           e->dspark_model.size,
                                           e->dspark_model.tensor_data_pos,
                                           e->dspark_model.size - e->dspark_model.tensor_data_pos,
                                           e->dspark_model.max_tensor_bytes))
        {
            fprintf(stderr,
                    "pulsar: %s failed to map DSpark model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    pulsar_backend_name(e->backend));
            e->destroy();
            *out = NULL;
            return 1;
        }
        register_model_fds(&e->model);
        if (!accelerator_cache_model_tensors(e->backend, &e->model,
                                             NULL, NULL, 0,
                                             e->dspark_ready ? NULL : "dspark.")) {
            fprintf(stderr, "pulsar: %s failed to prepare optional model cache\n",
                    pulsar_backend_name(e->backend));
            e->destroy();
            *out = NULL;
            return 1;
        }
        if (e->dspark_ready && e->dspark_external) {
            /* An external drafter is its own model: it stages its own expert
             * stacks, for the same rank. */
            e->dspark_model.tp_rank = tp_rank_at_load;
            e->dspark_model.tp_n_ranks = tp_n_ranks_at_load;
            register_model_fds(&e->dspark_model);
            if (!accelerator_cache_model_tensors(e->backend, &e->dspark_model,
                                                 NULL, NULL, 0, NULL)) {
                fprintf(stderr, "pulsar: %s failed to prepare optional DSpark model cache\n",
                        pulsar_backend_name(e->backend));
                e->destroy();
                *out = NULL;
                return 1;
            }
            /* The main model's fd no longer needs restoring here: the fd route
             * is a per-mapping table now, so registering the drafter's mapping
             * leaves the main model's entry alone. */
        }
        if (e->overlay_ready &&
            !accelerator_prepare_expert_overlay(e->backend, &e->model,
                                                &e->overlay_model)) {
            fprintf(stderr, "pulsar: %s failed to prepare expert-overlay spans\n",
                    pulsar_backend_name(e->backend));
            e->destroy();
            *out = NULL;
            return 1;
        }
        fprintf(stderr, "pulsar: %s backend initialized for graph diagnostics\n",
                pulsar_backend_name(e->backend));
    }

    /* Two-rank TP (slice 4b, prefill big-gate arm).  Built here, after the
     * model/weights are bound and the GPU is up, because the identity needs the
     * resolved shape and the slab needs the CUDA runtime.  Only the graph
     * backend can host a TP pair; requesting TP on the CPU backend is a loud
     * refusal, not a quiet single-box fallback (rule 4). */
    if (opt->tp_role != 0 || opt->tp_peers) {
        if (!graph_backend) {
            fprintf(stderr, "pulsar: tensor parallelism requires the CUDA/graph "
                            "backend, not %s\n", pulsar_backend_name(e->backend));
            e->destroy();
            *out = NULL;
            return 1;
        }
        char tperr[512];
        pulsar_tp_options tp_opt;
        memset(&tp_opt, 0, sizeof(tp_opt));
        tp_opt.role = opt->tp_role == 2 ? PULSAR_TP_ROLE_WORKER
                                        : PULSAR_TP_ROLE_LEADER;
        tp_opt.port = opt->tp_port > 0 ? opt->tp_port : 5588;
        /* n-way (full mesh) vs legacy 2-rank.  n-way: tp_peers carries every
         * rank's "host:port" and tp_rank/tp_nranks are explicit; legacy pair
         * derives rank from tp_role (leader=0, worker=1), n_ranks=2. */
        if (opt->tp_peers) tp_opt.peers = opt->tp_peers;
        else               tp_opt.peer = opt->tp_peer;
        /* The rank and group size were decided ONCE, before staging (slice 4f);
         * the transport is built for that same identity. */
        tp_opt.rank = tp_rank_at_load;
        tp_opt.n_ranks = (int)tp_n_ranks_at_load;
        tp_opt.build = opt->build_id;
        uint32_t tp_layers = 0, tp_embd = 0, tp_vocab = 0;
        e->family->tp_shape(e, &tp_layers, &tp_embd, &tp_vocab);
        pulsar_tp_identity id;
        /* mapped_bytes, not size: a safetensors model's size is one shard (L272 B13) */
        pulsar_tp_identity_init_defaults(&id,
                                         (uint64_t)e->model.mapped_bytes,
                                         (uint32_t)e->model_id(),
                                         tp_layers,
                                         tp_embd,
                                         tp_vocab,
                                         (uint32_t)e->routed_quant_bits(),
                                         0);
        int tp_ok = opt->tp_peers
                        ? pulsar_tp_create_mesh(&e->tp, &tp_opt, &id, tperr, sizeof(tperr))
                        : pulsar_tp_create(&e->tp, &tp_opt, &id, tperr, sizeof(tperr));
        if (!tp_ok) {
            fprintf(stderr, "pulsar: tensor parallelism bring-up failed: %s\n",
                    tperr);
            e->destroy();
            *out = NULL;
            return 1;
        }
        if (opt->tp_kv_dir && opt->tp_kv_dir[0]) e->tp_kv_dir = pulsar_strdup(opt->tp_kv_dir);
        e->tp_slab_bytes = pulsar_tp_slab_bytes(tp_layers, tp_embd);
        /* The bulk lane (v14): a pair over RDMA moves its prefill exchanges
         * GPU-direct through a buffer of out | in[0] | in[1], each one prefill
         * chunk of f32 rows; attach registers it beside the slab.  A chunk
         * larger than that is cut by gpu_graph_tp_allreduce_rows. */
        if (pulsar_tp_is_rdma(e->tp) && pulsar_tp_n_ranks(e->tp) == 2) {
            const uint64_t rows = e->prefill_chunk ? e->prefill_chunk : PULSAR_PREFILL_CHUNK_DEFAULT;
            e->tp_bulk_bytes = 3u * rows * (uint64_t)tp_embd * sizeof(float);
            if (!pulsar_tp_gpu_slab_alloc_hostpin(e->tp_bulk_bytes, &e->tp_bulk_base, tperr, sizeof(tperr)) ||
                !(e->tp_bulk_dev = pulsar_tp_gpu_slab_device_ptr(e->tp_bulk_base, tperr, sizeof(tperr)))) {
                fprintf(stderr, "pulsar: tensor parallelism bulk buffer (%llu bytes) setup failed: %s\n",
                        (unsigned long long)e->tp_bulk_bytes, tperr);
                e->destroy();
                *out = NULL;
                return 1;
            }
            pulsar_tp_set_bulk(e->tp, e->tp_bulk_base, e->tp_bulk_bytes);
        }
        if (!pulsar_tp_gpu_slab_alloc_hostpin(e->tp_slab_bytes,
                                              &e->tp_slab_base,
                                              tperr, sizeof(tperr)) ||
            !pulsar_tp_attach_slab(e->tp, e->tp_slab_base, tperr, sizeof(tperr)) ||
            !(e->tp_slab_dev = pulsar_tp_gpu_slab_device_ptr(e->tp_slab_base, tperr, sizeof(tperr)))) {
            fprintf(stderr, "pulsar: tensor parallelism slab setup failed: %s\n",
                    tperr);
            e->destroy();
            *out = NULL;
            return 1;
        }
        if (pulsar_tp_row_lane(e->tp)) {
            pulsar_tp_row_lane_layout_t L;
            pulsar_tp_row_lane_layout(e->tp, &L);
            pulsar_gpu_tp_err_word_set((const volatile uint32_t *)((uint8_t *)e->tp_slab_base + L.err_off));
        }
        /* Slice 4f: the weights were staged for one identity and the transport
         * came up as another only if the two derivations drifted -- a bug, and
         * one that would compute the wrong experts on both ranks in silence. */
        if (pulsar_tp_rank(e->tp) != e->model.tp_rank ||
            pulsar_tp_n_ranks(e->tp) != e->model.tp_n_ranks) {
            fprintf(stderr, "pulsar: TP identity drift: the model was staged for rank %d/%u but "
                            "the transport came up as rank %d/%u -- refusing\n",
                    e->model.tp_rank, e->model.tp_n_ranks,
                    pulsar_tp_rank(e->tp), pulsar_tp_n_ranks(e->tp));
            e->destroy();
            *out = NULL;
            return 1;
        }
        fprintf(stderr, "pulsar: TP rank %d/%d armed (prefill big-gate), slab %zu bytes, %.2f GiB of stored tensors "
                        "unstaged (the plan's)\n",
                pulsar_tp_rank(e->tp), pulsar_tp_n_ranks(e->tp), e->tp_slab_bytes,
                (double)pulsar_model_unstaged_expert_bytes(&e->model) / 1073741824.0);
    }

    /* The rank's TP slices (the core's plan), then the family's own device-side registrations and announces. */
    if (graph_backend && (!pulsar_tp_plan_run(e) || !e->family->after_gpu(e))) {
        e->destroy();
        *out = NULL;
        return 1;
    }

    /* Every weight the device reads now lives in device memory (the staged spans,
     * and under TP this rank's expert halves), so the checkpoint's host pages are
     * dead weight: ~11 GiB per rank on the pair stayed resident after the TP half
     * build read the expert stacks through the mapping, and with swappiness 60 the
     * kernel swapped server memory out to keep them (2026-09-30).  Release them;
     * a later host read of the mapping (metadata, a debug range check) refaults
     * from disk.  Kept when the device reads through the host mapping, or when a
     * non-graph backend computes from it. */
    if (pulsar_backend_uses_graph(e->backend) && e->gpu_ready) {
        if (pulsar_gpu_model_reads_host_pages()) {
            fprintf(stderr, "pulsar: checkpoint host pages KEPT: the device reads weights through the "
                            "host mapping\n");
        } else {
            uint64_t released = pulsar_model_release_host_pages(&e->model);
            if (e->dspark_ready && e->dspark_external)
                released += pulsar_model_release_host_pages(&e->dspark_model);
            if (e->overlay_ready) released += pulsar_model_release_host_pages(&e->overlay_model);
            fprintf(stderr, "pulsar: released %.2f GiB of checkpoint pages from host memory after load "
                            "(weights are device-resident)\n", (double)released / 1073741824.0);
        }
    }

    *out = e;
    return 0;
}


void pulsar_engine::summary() {
    auto *e = this;
    model_summary(&e->model);
}


int pulsar_engine::vocab_size() {
    auto *e = this;
    if (!e) return 0;
    /* the tokenizer TABLE length (L272 B3: Qwen never loads e->vocab; its table is the checkpoint's) */
    return e->qwen_tok ? qwen_tokenizer_n_tokens(e->qwen_tok) : e->vocab.n_vocab;
}


/* The engine's logits ROW WIDTH — the shape profile's n_vocab, which is what
 * every logits buffer the engine writes is strided by.  This is NOT
 * pulsar_engine_vocab_size (the tokenizer table length): the loader never checks
 * the two against each other, and sizing a logits buffer from the tokenizer
 * length is exactly the mismatch that produced an unbounded-logits write. */
int pulsar_engine::logits_width() const {
    auto *e = this;
    return e ? (int)e->family->logits_width(e) : 0;
}


const char *pulsar_engine::model_name() {
    auto *e = this;
    return e->family->model_name(e);
}

void pulsar_engine::spec_metrics(pulsar_spec_metrics *out) {
    auto *e = this;
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!e) return;
    out->accepted_tokens = e->spec_accepted_tokens;
    out->draft_tokens = e->spec_draft_tokens;
    out->num_drafts = e->spec_num_drafts;
    out->gen_tokens = e->spec_gen_tokens;
    for (int i = 0; i < 16; i++) out->accepted_per_pos[i] = e->spec_accepted_per_pos[i];
    for (int i = 0; i < 16; i++) out->verified_per_pos[i] = e->spec_verified_per_pos[i];
    out->max_draft = e->dspark_draft_tokens > PULSAR_SPEC_DEPTH_MAX
                         ? e->dspark_draft_tokens : PULSAR_SPEC_DEPTH_MAX;   /* L107: waterfall covers the adaptive range */
    out->has_drafter = e->drafter_ops != NULL;
}



uint64_t pulsar_engine::weights_resident_bytes() {
    auto *e = this;
    if (!e) return 0;
    /* The checkpoint(s) are mmap'd read-only and shared across every session, so
     * this is a single resident copy competing with per-session KV for the
     * unified memory budget.  A merged/embedded drafter lives inside e->model
     * and is already counted; an external drafter and an expert overlay map
     * their own files and are added when present.
     *
     * mapped_bytes, NOT size: a safetensors model's `size` is ONE SHARD, so the
     * server's admission budget read 0.88 GiB of weights for a 92 GB checkpoint
     * and over-stated the budget by ~85 GiB.  The measured min() masked it at
     * runtime, which is precisely why it had to be fixed here rather than
     * noticed: the static formula is the bound that is supposed to hold when the
     * measured one reads inflated. */
    uint64_t bytes = e->model.mapped_bytes - pulsar_model_unstaged_expert_bytes(&e->model);
    if (e->dspark_ready && e->dspark_external) {
        bytes += e->dspark_model.mapped_bytes - pulsar_model_unstaged_expert_bytes(&e->dspark_model);
    }
    if (e->overlay_ready) bytes += e->overlay_model.mapped_bytes;
    /* Under TP the stored expert stacks are unstaged (subtracted above) and this
     * rank's HALF of every expert is built on the device at open instead: ~73 GiB
     * per rank on V4-Flash that the admission budget read as free memory
     * (2026-09-29: static bound 89 GiB on a box with 16 GiB for KV); Qwen builds
     * its dense slices and expert halves the same way (L272 B9).  The small
     * shared-expert K-slice repacks are left to the process overhead reserve. */
    bytes += e->tp_built_bytes;
    return bytes;
}


int pulsar_engine::model_id() {
    auto *e = this;
    return e->family->model_id(e);
}


void pulsar_engine::destroy() {
    auto *e = this;
    if (!e) return;
    /* Tear down the TP pair before releasing the GPU/model (stop the peer, drop
     * the transport and its registered MR, then free the host-pinned slab). */
    if (e->tp) {
        if (pulsar_tp_rank(e->tp) == 0) {
            /* The cross-rank logits identity tally (L243): every logits-producing
             * frame the leader collected, and how many peer acks agreed.  The
             * pair grading tool reads this line as LEG A; a mismatch already
             * refused by name when it happened. */
            uint64_t frames = 0, matched = 0;
            pulsar_tp_identity_stats(e->tp, &frames, &matched);
            fprintf(stderr, "pulsar: tp: cross-rank logits identity: %llu/%llu worker frames matched\n",
                    (unsigned long long)matched, (unsigned long long)frames);
        }
        (void)pulsar_tp_send_stop(e->tp);
        pulsar_tp_free(e->tp);
        e->tp = NULL;
    }
    free(e->tp_kv_dir);
    e->tp_kv_dir = NULL;
    if (e->tp_slab_base) {
        pulsar_gpu_tp_err_word_set(NULL);
        pulsar_tp_gpu_slab_free_hostpin(e->tp_slab_base);
        e->tp_slab_base = NULL;
        e->tp_slab_dev = NULL;
        e->tp_slab_bytes = 0;
    }
    if (e->tp_bulk_base) {   /* after pulsar_tp_free deregistered it */
        pulsar_tp_gpu_slab_free_hostpin(e->tp_bulk_base);
        e->tp_bulk_base = NULL;
        e->tp_bulk_dev = NULL;
        e->tp_bulk_bytes = 0;
    }
    weights_free(&e->weights);
    pulsar_tp_plan_free(e);   /* the rank's built slices, before the device goes */
    if (e->qwen_weights) pulsar_qwen_s4_unload(e->qwen_weights);
    if (e->qwen_tok) qwen_tokenizer_free(e->qwen_tok);
    e->qwen_tok = NULL;
    free(e->qwen_weights);
    e->vocab.vocab_free();
    /* Tear down GPU state (which cudaHostUnregisters the mmap'd weight ranges)
     * before munmap'ing the model — unmapping still-registered pages is UB. */
    pulsar_gpu_cleanup();
    if (e->dspark_ready && e->dspark_external) model_close(&e->dspark_model);
    if (e->overlay_ready) model_close(&e->overlay_model);
    model_close(&e->model);
    pulsar_release_instance_lock();
    free(e->directional_steering_dirs);
    free(e->directional_steering_file);
    free(e);
}
/* The per-session tensor-parallel scratch (4g-2): the vocab gather's own-slice
 * buffer (pulsar_gpu_graph::tp_vocab_own) -- PULSAR_SPEC_LOGITS_ROWS rows at
 * the widest rank range, rounded up to whole row-lane messages -- and the row
 * lane's stage+publish ticket (tp_stage_ticket, zeroed).  Part of the create,
 * so the admission price (session_cost_bytes_banked: the create run dry)
 * carries it.  Nothing on a box with no row-lane pair. */
static bool session_alloc_tp_scratch(pulsar_gpu_graph *g, pulsar_tp *tp) {
    if (!tp || !pulsar_tp_row_lane(tp)) return true;
    const uint64_t vb = pulsar_tp_vec_bytes(tp);
    const uint64_t bytes = pulsar_tp_vocab_own_bytes(tp, (uint32_t)PULSAR_N_VOCAB, (uint32_t)PULSAR_SPEC_LOGITS_ALLOC_ROWS);
    g->tp_vocab_own = pulsar_gpu_tensor_alloc(bytes);
    if (!g->tp_vocab_own) {
        fprintf(stderr, "pulsar: tp vocab gather scratch (%llu bytes) allocation failed\n",
                (unsigned long long)bytes);
        return false;
    }
    g->tp_ain_own = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_TP_BATCH_MAX_ROWS * vb);
    if (!g->tp_ain_own) {
        fprintf(stderr, "pulsar: tp attention-input gather scratch allocation failed\n");
        return false;
    }
    const uint32_t zero = 0u;
    g->tp_stage_ticket = pulsar_gpu_tensor_alloc(sizeof(zero));
    if (!g->tp_stage_ticket || !pulsar_gpu_tensor_write(g->tp_stage_ticket, 0, &zero, sizeof(zero))) {
        fprintf(stderr, "pulsar: tp row-lane stage ticket allocation failed\n");
        return false;
    }
    return true;
}



/* Every session is created by its engine's family (family.h).  The ONE build both the create and the price
 * run: the core allocates the session and its own state (the prefill cap, the logits row at the family's
 * width), the family builds its state for n_banks into it.  NULL when the family refused. */
static pulsar_session *session_build(pulsar_engine *e, int ctx_size, uint32_t n_banks) {
    pulsar_session *s = (pulsar_session *)xcalloc(1, sizeof(*s));
    s->engine = e;
    s->ctx_size = ctx_size;
    s->prefill_cap = pulsar_prefill_cap_for_prompt(ctx_size, e->prefill_chunk);
    s->logits = (float *)xmalloc((size_t)e->logits_width() * sizeof(s->logits[0]));
    if (e->family->session->create(s, n_banks) != 0) {
        free(s->logits);
        free(s);
        return NULL;
    }
    return s;
}

/* L272 P2: every family's session begins here -- session_build at the engine's bank pool, and the core
 * measures what that allocated on the GPU (the allocator's delta across the create, so callers can
 * reconcile admission estimates against reality). */
int pulsar_session::create(pulsar_session **out, pulsar_engine *e, int ctx_size) {
    if (!out || !e || ctx_size <= 0) return 1;
    *out = NULL;
    if (!pulsar_backend_uses_graph(e->backend) || !e->gpu_ready) return 1;
    const uint64_t alloc_before = pulsar_gpu_tensor_alloc_bytes_current();
    pulsar_session *s = session_build(e, ctx_size, gpu_graph_bank_pool_n());
    if (!s) return 1;
    s->resident_bytes = pulsar_gpu_tensor_alloc_bytes_current() - alloc_before;
    /* Slice 4e: the mirror id both ranks agree on by construction -- the engine's create ordinal, assigned
     * here, where every family's session begins (L266: it was DeepSeek's create, so a Qwen session was never
     * mirrored).  A session created with no pair armed keeps 0 and stays out of the mirror. */
    if (e->tp) s->tp_session_id = ++e->tp_session_seq;
    *out = s;
    return 0;
}


/* The DeepSeek family's session state: the pulsar_gpu_graph (SWA rings, compressed KV and frontiers, bank
 * slabs), steering, the TP scratch and the drafter's buffers, built into a session the core allocated.
 * pulsar_session::create's body until L251. */
int pulsar_ds4_session_create(pulsar_session *s, uint32_t n_banks) {
    pulsar_engine *e = s->engine;
    const int ctx_size = s->ctx_size;
    s->prefill_frontier = 0;   /* L195: nothing prefilled yet */
    const uint32_t raw_cap = gpu_graph_raw_cap_for_context(ctx_size, s->prefill_cap);
    const pulsar_layer_weights *shape_layer = weights_first_bound_layer(&e->weights);
    if (!shape_layer) {
        fprintf(stderr, "pulsar: no transformer layers are loaded\n");
        return 1;
    }
    s->graph = (pulsar_gpu_graph *)xcalloc(1, sizeof(*s->graph));   /* L272 P6: the family's own */
    if (!gpu_graph_alloc_raw_cap(s->graph, &e->weights, shape_layer,
                                   raw_cap, (uint32_t)ctx_size, s->prefill_cap,
                                   n_banks, e->dspark_ready))
    {
        free(s->graph);
        s->graph = NULL;
        return 1;
    }
    if (!gpu_graph_load_directional_steering(s->graph,
                                               e->directional_steering_file,
                                               e->directional_steering_attn_scale,
                                               e->directional_steering_ffn_scale)) {
        pulsar_ds4_session_destroy(s);
        return 1;
    }
    /* Borrow the engine's TP transport into the graph so the prefill big-gate
     * call sites can reach it without threading the engine through every
     * gpu_graph entry point (slice 4b).  NULL when the pair is not armed. */
    s->graph->tp = e->tp;
    s->graph->tp_group_lo = e->tp_group_lo;
    s->graph->tp_group_hi = e->tp_group_hi;
    s->graph->tp_slab_dev = e->tp_slab_dev;
    s->graph->tp_bulk_dev = e->tp_bulk_dev;
    s->graph->tp_kslice_key = e->tp ? (const void *)e : NULL;
    if (!session_alloc_tp_scratch(s->graph, e->tp)) {
        pulsar_ds4_session_destroy(s);
        return 1;
    }
    if (e->dspark_ready) {
        if (!gpu_graph_init_dspark_target(s->graph, e->dspark_weights.target_layer_ids)) {
            fprintf(stderr, "pulsar: failed to allocate DSpark graph buffers\n");
            pulsar_ds4_session_destroy(s);
            return 1;
        }
    }
    return 0;
}


uint64_t pulsar_session_resident_bytes(const pulsar_session *s) {
    return s ? s->resident_bytes : 0;
}


/* The price of pulsar_session::create at (ctx_size, n_banks) IS the create: session_build run with the
 * tensor primitives in dry mode (pulsar_gpu_tensor_dry_begin) and torn down, so the number admission
 * control charges and the number the allocator commits come from one piece of code for every family --
 * the server checks them equal after every create.  *managed_bytes (optional) is the demand-paged
 * (cudaMallocManaged) subset.  Not in the price, on either side: allocations made on demand after create
 * (gpu_graph_ensure_batch_ffn_out, the multiseq descriptors, the batched-copy descriptor tables); the
 * server's memory floor absorbs those. */
uint64_t pulsar_engine::session_cost_bytes_banked(int ctx_size, int n_banks, uint64_t *managed_bytes) {
    auto *e = this;
    if (managed_bytes) *managed_bytes = 0;
    if (!e || ctx_size <= 0 || n_banks < 1) return 0;
    if (!pulsar_backend_uses_graph(e->backend) || !e->gpu_ready) return 0;
    pulsar_gpu_tensor_dry_begin();
    pulsar_session *s = session_build(e, ctx_size, (uint32_t)n_banks);
    const bool ok = s != NULL;
    if (s) s->destroy();
    uint64_t bytes = 0, managed = 0;
    pulsar_gpu_tensor_dry_end(&bytes, &managed);
    if (managed_bytes && ok) *managed_bytes = managed;
    return ok ? bytes : 0;
}

uint64_t pulsar_engine::session_cost_bytes(int ctx_size) {
    return session_cost_bytes_banked(ctx_size, (int)gpu_graph_bank_pool_n());
}

uint64_t pulsar_engine::demand_paged_bytes_per_bank(int ctx_size) {
    auto *e = this;
    if (!e || ctx_size <= 0) return 0;
    if (!pulsar_backend_uses_graph(e->backend) || !e->gpu_ready) return 0;
    return gpu_graph_demand_paged_bytes_per_bank((uint32_t)ctx_size);
}


/* L272 P2: the family frees its state, the core the session's own -- one host half for every family
 * (Qwen's destroy had kept its own copy of it, without the speculation scratch the round API allocates). */
void pulsar_session::destroy() {
    auto *s = this;
    if (!s) return;
    s->engine->family->session->destroy(s);
    token_vec_free(&s->checkpoint);
    pulsar_sample_scratch_free(&s->sample_scratch);
    s->bank_carry_free();
    free(s->pend_qrows);
    free(s->spec_row_scratch);
    free(s->logits);
    free(s);
}


void pulsar_ds4_session_destroy(pulsar_session *s) {
    if (!s->graph) return;
    /* a priced (dry) session never ran: its placeholders go without resetting the live session's
     * segment graph */
    if (pulsar_gpu_tensor_dry_active()) gpu_graph_release(s->graph);
    else gpu_graph_free(s->graph);
    free(s->graph);
    s->graph = NULL;
}


void pulsar_session::set_progress(pulsar_session_progress_fn fn, void *ud) {
    auto *s = this;
    if (!s) return;
    s->progress = fn;
    s->progress_ud = ud;
}


void pulsar_session::set_display_progress(pulsar_session_progress_fn fn, void *ud) {
    auto *s = this;
    if (!s) return;
    s->display_progress = fn;
    s->display_progress_ud = ud;
}


void pulsar_session::set_cancel(pulsar_session_cancel_fn fn, void *ud) {
    auto *s = this;
    if (!s) return;
    s->cancel = fn;
    s->cancel_ud = ud;
}


bool pulsar_session_cancelled(pulsar_session *s) {
    /* A MIRRORED session stops at a chunk boundary only TOGETHER (v15): a
     * leader that stopped after k chunks alone would leave its workers running
     * the rest of the sync, waiting at an exchange the leader never joins (the
     * reason slice 4e disabled this hook outright -- which also froze the
     * server's per-chunk yield: no progress published, no disconnect honoured,
     * every other session stalled for the whole prompt).  Every rank polls at
     * the same boundaries; the leader decides with its own hook and ships the
     * verdict, each worker reads it here. */
    if (!s) return false;
    if (pulsar_session_is_mirrored(s)) {
        pulsar_tp *tp = s->engine->tp;
        if (pulsar_tp_rank(tp) == 0) {
            const bool stop = s->cancel && s->cancel(s->cancel_ud);
            if (!pulsar_tp_send_chunk_verdict(tp, s->tp_session_id, stop ? 1 : 0)) {
                pulsar_tp_mark_failed(tp);
                fprintf(stderr, "pulsar: tp: could not ship the chunk verdict -- stopping the prefill\n");
                return true;
            }
            return stop;
        }
        int stop = 0;
        char err[256];
        if (!pulsar_tp_recv_chunk_verdict(tp, s->tp_session_id, &stop, err, sizeof(err))) {
            fprintf(stderr, "pulsar: %s\n", err);
            return true;
        }
        return stop != 0;
    }
    return s->cancel && s->cancel(s->cancel_ud);
}


int pulsar_session::common_prefix(const pulsar_tokens *prompt) {
    auto *s = this;
    if (!s->checkpoint_valid) return 0;
    int n = s->checkpoint.len < prompt->len ? s->checkpoint.len : prompt->len;
    int i = 0;
    while (i < n && s->checkpoint.v[i] == prompt->v[i]) i++;
    return i;
}

/* L115: token-boundary-insensitive common prefix.  Generated text freezes
 * SAMPLED token boundaries into the live checkpoint; a client echo of the
 * same bytes re-tokenizes CANONICALLY, so consecutive turns disagree on ids
 * while agreeing on every byte (live `))`+`**` vs echoed `))**`).  An
 * id-exact compare declares divergence at the earliest such seam and a
 * conversation re-pays its whole history forever.  This walk advances two
 * (token, byte-offset) cursors: equal ids on a shared boundary take the
 * fast path; on mismatch it compares bytes until the boundaries realign
 * (another shared boundary — a seam crossed) or a byte truly differs.
 * Returns the LARGEST (a_n, b_n) with bytes(a[0..a_n)) == bytes(b[0..b_n))
 * ending on a shared boundary.  Zero-length token texts (control/special
 * ids) only match by ID: a boundary mismatch involving one stops the walk
 * conservatively — role markers must never byte-alias into content. */
void pulsar_tokens_prefix_match(pulsar_engine *e,
                                const int *a, int a_len,
                                const int *b, int b_len,
                                pulsar_prefix_match *out) {
    out->live_cut = 0;
    out->prompt_cut = 0;
    out->seamed = false;
    if (!e || !a || !b) return;
    bool seamed = false;
    int i = 0, j = 0;          /* token cursors */
    size_t oa = 0, ob = 0;     /* byte offsets inside the current tokens */
    int best_i = 0, best_j = 0;
    const char *ta = NULL, *tb = NULL;
    size_t la = 0, lb = 0;
    while (i < a_len && j < b_len) {
        if (oa == 0 && ob == 0) {
            best_i = i;
            best_j = j;
            if (a[i] == b[j]) { i++; j++; continue; }   /* aligned fast path */
            seamed = true;   /* same bytes ahead, different boundaries */
        }
        if (oa == 0) {
            ta = pulsar_token_text(e, a[i], &la);
            if (!ta || la == 0) break;   /* control/special: id-only match */
        }
        if (ob == 0) {
            tb = pulsar_token_text(e, b[j], &lb);
            if (!tb || lb == 0) break;
        }
        while (oa < la && ob < lb) {
            if (ta[oa] != tb[ob]) goto done;   /* true byte divergence */
            oa++; ob++;
        }
        if (oa == la) { i++; oa = 0; }
        if (ob == lb) { j++; ob = 0; }
    }
    if (oa == 0 && ob == 0) { best_i = i; best_j = j; }
done:
    out->live_cut = best_i;
    out->prompt_cut = best_j;
    out->seamed = seamed;
}

void pulsar_session::prefix_match(const pulsar_tokens *prompt,
                                 pulsar_prefix_match *out) {
    auto *s = this;
    out->live_cut = 0;
    out->prompt_cut = 0;
    out->seamed = false;
    if (!s->checkpoint_valid || !prompt) return;
    pulsar_tokens_prefix_match(s->engine,
                               s->checkpoint.v, s->checkpoint.len,
                               prompt->v, prompt->len, out);
}


int pulsar_session::argmax() {
    auto *s = this;
    return sample_argmax(s->logits, (uint32_t)s->engine->logits_width());
}


int pulsar_session::argmax_excluding(int excluded_id) {
    auto *s = this;
    if (!s || !s->logits) return -1;
    /* THE row-max rule (sample_argmax): first finite value seeds, lowest id
     * wins a tie, -1 when nothing is finite. */
    int best = -1;
    float best_logit = 0.0f;
    for (uint32_t i = 0, nv = (uint32_t)s->engine->logits_width(); i < nv; i++) {
        if ((int)i == excluded_id) continue;
        const float v = s->logits[i];
        if (!isfinite(v)) continue;
        if (best < 0 || v > best_logit) {
            best = (int)i;
            best_logit = v;
        }
    }
    return best;
}


int pulsar_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng) {
    if (!logits || n_vocab <= 0) return PULSAR_SAMPLE_REFUSED;   /* was token 0 (L188) */
    /* No session here, so no scratch to borrow: this public entry keeps the
     * malloc path. Session callers below pass their own. */
    return sample_top_p_min_p(logits, (uint32_t)n_vocab, temperature, top_k, top_p,
                              min_p, rng, NULL);
}


int pulsar_session::sample(float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    auto *s = this;
    return sample_top_p_min_p(s->logits, (uint32_t)s->engine->logits_width(), temperature, top_k, top_p,
                              min_p, rng, &s->sample_scratch);
}


/* Row-based twins of the two session logprob readers below, in the same
 * relation pulsar_sample_logits has to pulsar_session::sample: score a
 * caller-supplied logits row instead of the session's own.  The batched decode
 * entries (pulsar_session_decode_multiseq / _mixed) hand each bank's row back to
 * the caller and deliberately leave s->logits alone, so for those steps the
 * returned row is the ONLY place that position's distribution exists.  Same
 * arithmetic as before, moved verbatim — the session methods now delegate. */
int pulsar_logits_top_logprobs(const float *logits, int n_vocab,
                            pulsar_token_score *out, int k) {
    if (!logits || !out || k <= 0 || n_vocab <= 0) return 0;
    if (k > n_vocab) k = n_vocab;
    for (int i = 0; i < k; i++) {
        out[i].id = -1;
        out[i].logit = PULSAR_NEG_INF;
        out[i].logprob = PULSAR_NEG_INF;
    }

    /* Row max by THE rule (sample_argmax): the first finite value seeds.  A
     * -1e30 seed is finite, so an all-non-finite row used to pass the
     * isfinite(max_logit) check below and score against log(0). */
    float max_logit = 0.0f;
    bool have_max = false;
    for (int i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        if (!have_max || v > max_logit) {
            max_logit = v;
            have_max = true;
        }
        for (int j = 0; j < k; j++) {
            if (out[j].id < 0 || v > out[j].logit) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j].id = i;
                out[j].logit = v;
                break;
            }
        }
    }
    if (!have_max) return 0;

    double sum = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    for (int i = 0; i < k && out[i].id >= 0; i++) {
        out[i].logprob = isfinite(out[i].logit) ? (float)((double)out[i].logit - logsum) : PULSAR_NEG_INF;
    }
    return k;
}


int pulsar_logits_token_logprob(const float *logits, int n_vocab, int token,
                             pulsar_token_score *out) {
    if (!logits || !out || n_vocab <= 0 || token < 0 || token >= n_vocab) return 0;

    float max_logit = 0.0f;
    bool have_max = false;
    for (int i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        if (!have_max || v > max_logit) {
            max_logit = v;
            have_max = true;
        }
    }
    if (!have_max) return 0;

    double sum = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    out->id = token;
    out->logit = logits[token];
    out->logprob = isfinite(out->logit) ? (float)((double)out->logit - logsum) : PULSAR_NEG_INF;
    return 1;
}


int pulsar_session::top_logprobs(pulsar_token_score *out, int k) {
    auto *s = this;
    if (!s) return 0;
    return pulsar_logits_top_logprobs(s->logits, s->engine->logits_width(), out, k);
}


int pulsar_session::token_logprob(int token, pulsar_token_score *out) {
    auto *s = this;
    if (!s) return 0;
    return pulsar_logits_token_logprob(s->logits, s->engine->logits_width(), token, out);
}


int pulsar_session::copy_logits(float *out, int cap) {
    auto *s = this;
    if (!s || !out || cap < s->engine->logits_width()) return 0;
    memcpy(out, s->logits, (size_t)s->engine->logits_width() * sizeof(out[0]));
    return s->engine->logits_width();
}


int pulsar_session::set_logits(const float *logits, int n) {
    auto *s = this;
    if (!s || !logits || n != s->engine->logits_width()) return 1;
    memcpy(s->logits, logits, (size_t)s->engine->logits_width() * sizeof(s->logits[0]));
    return 0;
}


bool pulsar_session_token_is_id(const pulsar_session *s, int token, char *err, size_t errlen) {
    if (token >= 0 && token < pulsar_engine_vocab_size(s->engine)) return true;
    snprintf(err, errlen, "eval: token %d is not a vocab id (a refused sample must fail the "
                          "request, not be evaluated)", token);
    return false;
}

/* Trim the committed history back to pos WITHOUT touching the KV content
 * below it. The caller owns the invariant that positions >= pos were never
 * exposed to the client (ghost tokens from a mid-block speculative stop);
 * the next prefill/eval overwrites their rows. Restored 2026-08-19: this was
 * deleted as callerless the same morning, then the spec mid-block stop path
 * turned out to need exactly it (it was full-session invalidate before,
 * which threw away the whole live KV on every mid-block tool-call stop). */
void pulsar_session::rewind(int pos) {
    auto *s = this;
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    s->trim_history(pos);
    s->logits_stale = true;
    /* The compressed frontier of every kv source follows the position law on
     * the bank this session's checkpoint describes (other banks hold other
     * slots' positions).  Clamp DOWN only -- a counter can only be AHEAD of a
     * rewound position, and a lagging one (mid-admission prefill) must never
     * be raised here.  Rows beyond the clamp are invisible (readers cap at
     * n_comp) and are rewritten by the next emit at that index; the clamp also
     * drops the grid checkpoints above pos (gpu_graph_set_n_comp). */
    const uint32_t rw_bank = gpu_graph_cur_bank(s->graph);
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (!gpu_graph_layer_is_kv_source(il)) continue;
        const uint32_t want = (uint32_t)pos / pulsar_layer_compress_ratio(il);
        if (gpu_graph_n_comp(s->graph, rw_bank, il) > want) gpu_graph_set_n_comp(s->graph, rw_bank, il, want);
    }
    /* The VALUE half (L264).  The recurrent lanes and the raw window at pos are
     * re-established in exactly two cases: position 0, whose state is the
     * canonical empty group, and a grid checkpoint AT pos.  Anywhere else the
     * bank is marked stale -- its lanes and window describe a position it no
     * longer stands at -- and every compressor store refuses until a sync
     * restores a checkpoint at or below the prefill frontier (or rebuilds from
     * 0).  Nothing decodes from a rewound position directly: a rewind ends a
     * generation (a stop's ghost tail, L073) or is the first half of a sync. */
    bool ok = true;
    if (pos == 0) {
        /* Position 0 is where a bank changes conversations (invalidate, a cold
         * rebuild, an evicted bank's reuse): none of its checkpoints describes
         * the next one, including those a spill kept while its frontier sat at 0
         * and the clamp above therefore could not drop. */
        pulsar_ckpt_drop_bank(&s->graph->ckpt, rw_bank);
        ok = gpu_graph_compressor_state_reset(s->graph, rw_bank);
    }
    else if (pulsar_ckpt_best(&s->graph->ckpt, rw_bank, (uint32_t)pos) == (uint32_t)pos)
        ok = pulsar_ckpt_restore(&s->graph->ckpt, rw_bank, (uint32_t)pos);
    else {
        s->graph->ms_comp_state_stale[rw_bank] = true;
        return;
    }
    if (!ok) {
        fprintf(stderr, "pulsar: rewind to %d on bank %u: state copy failed -- checkpoint invalidated\n", pos, rw_bank);
        s->checkpoint_valid = false;
        return;
    }
    s->graph->ms_comp_state_stale[rw_bank] = false;
}


void pulsar_session::trim_history(int pos) {
    auto *s = this;
    /* The image blocks above the new end are gone from the live KV: their records go with them or a later request
     * could reuse rows that no longer exist (L226).  L281: the blocks BELOW it survive, records and all -- before,
     * one cut block dropped the whole identity and the next image request rebuilt every block from 0. */
    pulsar_image_identity_trim(&s->live_images, (uint32_t)(pos < 0 ? 0 : pos));
    s->checkpoint.len = pos;
    spec_lookahead_reset(s);
    /* Rewound positions' drafter rows are stale; empty the window (it refills
     * from the prompt capture on the next prefill, or from commits). */
    for (int i = 0; i < 3; i++) s->graph->dspark_n_raw[i] = 0;
    s->graph->dspark_prompt_n = 0;
    if (s->prefill_frontier > pos) s->prefill_frontier = pos;   /* L195: a prefill above the new frontier never happened */
}


bool pulsar_session::restore_checkpoint(uint32_t G) {
    auto *s = this;
    const uint32_t bank = gpu_graph_cur_bank(s->graph);
    if (!s->checkpoint_valid || G == 0u || G > (uint32_t)s->checkpoint.len ||
        pulsar_ckpt_best(&s->graph->ckpt, bank, G) != G) return false;
    if (!pulsar_ckpt_restore(&s->graph->ckpt, bank, G)) {
        /* The device copies were issued in part: the bank's state is no longer
         * any position's.  Nothing reads it before a rebuild. */
        s->checkpoint_valid = false;
        return false;
    }
    s->trim_history((int)G);
    s->logits_stale = true;
    /* The checkpoint was captured where a prefill reached G, and a frontier
     * that fell below G since would have dropped it. */
    s->prefill_frontier = (int)G;
    return true;
}


int pulsar_session::pos() {
    auto *s = this;
    return s->checkpoint.len;
}


int pulsar_session::ctx() {
    auto *s = this;
    return s->ctx_size;
}


int pulsar_session_resume_origin(pulsar_session *s) { return s->resume_origin; }

int pulsar_session_prefill_cap(pulsar_session *s) {
    return s ? (int)s->prefill_cap : 0;
}




#include "pulsar_engine_internal.h"
#include "exl3_trellis.h"
#include "tp/pulsar_tp.h"
#include "tp/pulsar_tp_gpu.h"


int pulsar_engine::routed_quant_bits() {
    auto *e = this;
    if (!e) return 0;
    /* Report the routed-expert precision tier actually present, derived from
     * the loaded tensor types (was hardcoded 2, which under-reported the mixed
     * IQ2 + MXFP4/type-40 build as pure 2-bit). Any 4-bit routed format
     * (MXFP4 E2M1 / CUTLASS type-40) anywhere in gate/up/down makes this a
     * 4-bit-tier model; otherwise the 2-bit floor (IQ2_XXS / Q2_K); 0 if no
     * routed experts. The kvstore snapshot-compat guards accept {2,4} and
     * pulsar_engine_model_id() is a compile-time constant, so this is the only
     * model-variant discriminator in the disk-KV key — a value change
     * invalidates old snapshots (one-time re-prefill; fine in dev). */
    /* EXL3 (L245) is its own value space -- 20 + the rate in half-bit units
     * (24 = K2, 25 = K2.5, 26 = K3; the highest rate present wins) -- so an
     * EXL3 artifact never shares a KV key with the IQ2 (2) or MXFP4 (4) tier
     * of the same model id.  pulsar_kvstore_quant_bits_valid() enumerates the
     * accepted set; the kvstore refuses to store or match anything else. */
    int bits = 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const pulsar_tensor *proj[3] = {
            e->weights.layer[il].ffn_gate_exps,
            e->weights.layer[il].ffn_up_exps,
            e->weights.layer[il].ffn_down_exps,
        };
        for (int k = 0; k < 3; k++) {
            const pulsar_tensor *t = proj[k];
            if (!t) continue;
            if (t->type == PULSAR_TENSOR_CUTLASS_MXFP4)
                return 4;
            const int k2 = exl3_type_k2(t->type);
            if (k2) {
                if (20 + k2 > bits) bits = 20 + k2;
                continue;
            }
            if (bits == 0) bits = 2;
        }
    }
    return bits;
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


void pulsar_engine::dump_tokens(const pulsar_tokens *tokens) {
    auto *e = this;
    e->vocab.dump_tokens(tokens);  /* the pulsar_vocab member */
}


int pulsar_dump_text_tokenization(const char *model_path, const char *text, FILE *fp) {
    pulsar_model model;
    pulsar_vocab vocab;
    token_vec tokens = {0};

    if (!fp) fp = stdout;
    model_open(&model, model_path, false);
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

    const pulsar_model *model = &e->model;
    const pulsar_weights *weights = &e->weights;
    const uint32_t prefill_cap =
        gpu_graph_prefill_cap_for_prompt(ctx_size, e->prefill_chunk);
    const uint32_t raw_cap = gpu_graph_raw_cap_for_context(ctx_size, prefill_cap);

    pulsar_gpu_graph g;
    bool ok = gpu_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size, prefill_cap,
                                        gpu_graph_bank_pool_n(), false);
    if (!ok) {
        fprintf(stderr, "pulsar: failed to allocate imatrix GPU graph runtime\n");
        free(dataset);
        return 1;
    }

    pulsar_imatrix_collector collector;
    if (!imatrix_collector_init(&collector, prefill_cap, dataset_path)) {
        fprintf(stderr, "pulsar: failed to allocate imatrix collector\n");
        gpu_graph_free(&g);
        free(dataset);
        return 1;
    }

    fprintf(stderr,
            "pulsar: collecting routed-MoE imatrix from %s (model=%s, layers=%u, experts=%u, ctx=%d, chunk=%u)\n",
            dataset_path, PULSAR_MODEL_SHAPE_NAME, PULSAR_N_LAYER, PULSAR_N_EXPERT, ctx_size, prefill_cap);

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
                if (!gpu_graph_reset_prefill_state(&g)) {
                    fprintf(stderr, "pulsar: failed to reset imatrix graph state\n");
                    ok = false;
                } else if ((uint32_t)prompt.len > prefill_cap) {
                    ok = gpu_graph_prefill_chunked_range(&g, model, weights,
                                                           &prompt, 0,
                                                           (uint32_t)prompt.len,
                                                           NULL, false,
                                                           NULL, NULL,
                                                           NULL, NULL,
                                                           &collector,
                                                           NULL, NULL, NULL);
                } else {
                    ok = gpu_graph_prefill_layer_major(&g, model, weights,
                                                         &prompt, 0,
                                                         (uint32_t)prompt.len,
                                                         NULL, false,
                                                         &collector,
                                                         NULL, NULL);
                }
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
                            (unsigned long long)collector.observed_routes);
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
        ok = imatrix_collector_save(&collector, weights, output_path);
        if (ok) {
            fprintf(stderr,
                    "pulsar: wrote imatrix %s from %d prompts, %d tokens, %llu routed expert observations\n",
                    output_path,
                    prompts_done,
                    tokens_done,
                    (unsigned long long)collector.observed_routes);
        }
    }

    imatrix_collector_free(&collector);
    gpu_graph_free(&g);
    free(dataset);
    return ok ? 0 : 1;
}


int pulsar_engine::generate_argmax(const pulsar_tokens  *prompt,
        int                n_predict,
        int                ctx_size,
        pulsar_token_emit_fn  emit,
        pulsar_generation_done_fn done,
        void              *emit_ud,
        pulsar_session_progress_fn progress,
        void              *progress_ud) {
    auto *e = this;
    const pulsar_model *model = &e->model;
    const pulsar_vocab *vocab = &e->vocab;
    const pulsar_weights *weights = &e->weights;

    /* The raw whole-graph pipeline builds its own graph with no TP transport
     * and no owned head-group span: under a pair it cannot gather the
     * attention `low` rows, big-gate the FFN or all-reduce the owned experts,
     * and the first layer's guard refuses with a message about the GRAPH.
     * Say it here, once, by name (rule 9): generation under TP rides the
     * session lane, whose operations the group mirrors (slice 4e). */
    if (e->tp) {
        fprintf(stderr, "pulsar: raw whole-graph generation refused under tensor parallelism "
                        "(rank %d/%u): the path has no TP transport -- generation on a TP "
                        "engine rides the session lane\n",
                pulsar_tp_rank(e->tp), pulsar_tp_n_ranks(e->tp));
        return 1;
    }

    if (pulsar_backend_uses_graph(e->backend)) {
        if (!e->gpu_ready) {
            fprintf(stderr, "pulsar: %s generation requested but the graph backend is unavailable\n",
                    pulsar_backend_name(e->backend));
            return 1;
        }
        return generate_gpu_graph_raw_swa(model, vocab, weights, prompt,
                                            n_predict, ctx_size,
                                            e->prefill_chunk,
                                            e->directional_steering_file,
                                            e->directional_steering_attn_scale,
                                            e->directional_steering_ffn_scale,
                                            emit, done, emit_ud,
                                            progress, progress_ud);
    }
    return 1;
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

/* L241 4g-2: the shared expert, split like a Megatron MLP -- gate/up by
 * OUTPUT rows (column-parallel: this rank's half of the intermediate), down by
 * INPUT columns (row-parallel: the matching half of the reduction).  The rank's
 * shared output is then a PARTIAL that rides the FFN's existing exchange with
 * the routed partial, so the split costs no exchange of its own.  The range is
 * the one authority every split shares (pulsar_tp_owned_range over the shared
 * width).  Registered once at open, for the target's layers and the drafter's.
 * The K-half's key is (engine, the tensor OBJECT's address) -- never its
 * abs_offset: a safetensors checkpoint is one shard per layer with identical
 * layouts, so every layer's down projection sits at the SAME offset, and an
 * offset key made all 43 layers resolve to whichever registered last. */
static bool tp_register_shared_split(const void *kslice_key, const pulsar_model *m,
                                     const pulsar_layer_weights *L, int rank, uint32_t nr) {
    if (!L->ffn_gate_shexp || !L->ffn_up_shexp || !L->ffn_down_shexp) return false;
    const uint64_t in_dim = L->ffn_gate_shexp->dim[0];
    const uint64_t shared_dim = L->ffn_gate_shexp->dim[1];
    uint32_t lo = 0, hi = 0;
    if (!pulsar_tp_owned_range(rank, nr, (uint32_t)shared_dim, &lo, &hi) || hi <= lo) return false;
    return pulsar_gpu_register_fp8_lt_row_slice(tensor_map_base(m, L->ffn_gate_shexp),
                                                L->ffn_gate_shexp->abs_offset, in_dim, shared_dim, lo, hi) &&
           pulsar_gpu_register_fp8_lt_row_slice(tensor_map_base(m, L->ffn_up_shexp),
                                                L->ffn_up_shexp->abs_offset, in_dim, shared_dim, lo, hi) &&
           pulsar_gpu_register_fp8_lt_kslice(tensor_map_base(m, L->ffn_down_shexp),
                                             L->ffn_down_shexp->abs_offset, shared_dim,
                                             L->ffn_down_shexp->dim[1], lo, hi,
                                             kslice_key, pulsar_tp_kslice_key_offset(L->ffn_down_shexp));
}

int pulsar_engine::open(pulsar_engine **out, const pulsar_engine_options *opt) {
    pulsar_engine *e = (pulsar_engine *)xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->dspark_model.fd = -1;
    e->backend = opt->backend;
    e->prefill_chunk = opt->prefill_chunk;
    /* Slice 4f (L237): the rank this process loads the model FOR, decided from
     * the options before any weight is staged -- the transport is created
     * after the load, and on GB10 staging IS residency (a rank stages only its
     * owned experts).  Rank 0 of 1 when the pair is off.  The transport's rank
     * is asserted equal once it exists, so the two readers of this fact cannot
     * disagree.  An expert overlay swaps expert stacks from a donor file and
     * is staged by its own path, which has no ownership notion: refused under
     * TP rather than staged whole on every rank (rule 9). */
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
    model_open(&e->model, opt->model_path, graph_backend);
    /* Slice 4f: the model knows the rank it is loaded for from here on; the
     * merged drafter aliases e->model by value below and inherits it. */
    e->model.tp_rank = tp_rank_at_load;
    e->model.tp_n_ranks = tp_n_ranks_at_load;
    if (!opt->inspect_only) e->vocab.vocab_load(&e->model);
    config_validate_model(&e->model);
    if (opt->expert_overlay && opt->expert_overlay[0]) {
        const char *sep = strrchr(opt->expert_overlay, ':');
        if (!sep || sep == opt->expert_overlay || !sep[1]) {
            fprintf(stderr, "pulsar: --expert-overlay expects FILE:PREFIX (e.g. donor.gguf:blk.17.)\n");
            e->destroy();
            *out = NULL;
            return 1;
        }
        char overlay_path[4096];
        const size_t path_len = (size_t)(sep - opt->expert_overlay);
        if (path_len >= sizeof(overlay_path)) {
            fprintf(stderr, "pulsar: --expert-overlay path is too long\n");
            e->destroy();
            *out = NULL;
            return 1;
        }
        memcpy(overlay_path, opt->expert_overlay, path_len);
        overlay_path[path_len] = '\0';
        model_open(&e->overlay_model, overlay_path, graph_backend);
        e->overlay_ready = true;
        /* PREFIX is a comma-separated list so several layers can be swapped
         * in one run (e.g. compose "anchor + candidate" from a cheap base
         * without materializing the combined model as a file). */
        char prefixes[2048];
        const size_t plist_len = strlen(sep + 1);
        if (plist_len >= sizeof(prefixes)) {
            fprintf(stderr, "pulsar: --expert-overlay prefix list is too long\n");
            e->destroy();
            *out = NULL;
            return 1;
        }
        memcpy(prefixes, sep + 1, plist_len + 1);
        uint32_t swapped = 0;
        for (char *p = strtok(prefixes, ","); p; p = strtok(NULL, ",")) {
            const uint32_t n = model_apply_expert_overlay(&e->model, &e->overlay_model, p);
            if (n == 0) {
                fprintf(stderr, "pulsar: --expert-overlay prefix '%s' matched no routed-expert tensors\n",
                        p);
                e->destroy();
                *out = NULL;
                return 1;
            }
            swapped += n;
        }
        fprintf(stderr, "pulsar: expert overlay: %u tensors swapped in from %s (prefixes %s)\n",
                swapped, overlay_path, sep + 1);
    }
    weights_bind(&e->weights, &e->model);
    /* the drafter binds before the inspect-only exit so --inspect proves the
     * whole artifact binds, drafter included */
    if (!opt->dspark_disable && model_find_tensor(&e->model, "dspark.main_proj.weight")) {
        /* Drafter merged into the main GGUF: bind from the main model and
         * alias dspark_model to it by value (same map/fd; every dspark call
         * site reads e->dspark_model, and close is guarded on dspark_external
         * so the shared mapping is only torn down once). */
        dspark_weights_bind(&e->dspark_weights, &e->model);
        e->dspark_model = e->model;
        e->dspark_external = false;
        e->dspark_ready = true;
        fprintf(stderr, "pulsar: DSpark drafter found in model (draft=%d, markov_w2 %s)\n",
                e->dspark_draft_tokens, tensor_type_name(e->dspark_weights.markov_w2->type));
    }
    /* Vision-Exp tower: bound and layout-validated here so a wrong or
     * half-present vision stack refuses at load rather than at first image.
     * Absent tower is normal for text-only artifacts and simply leaves
     * vision_ready false; the image path is refused until it is true. */
    if (vision_weights_bind(&e->vision_weights, &e->model)) {
        e->vision_ready = true;
        fprintf(stderr, "pulsar: Vision-Exp tower bound (%u blocks, dim %u, %u heads, inter %u, "
                "patch %u, aligner %ux%d -> %u)\n",
                e->vision_weights.n_layers, (unsigned)PULSAR_VISION_DIM,
                (unsigned)PULSAR_VISION_HEADS, (unsigned)PULSAR_VISION_INTER,
                (unsigned)PULSAR_VISION_PATCH,
                (unsigned)PULSAR_VISION_DOWNSAMPLE, (unsigned)PULSAR_VISION_DOWNSAMPLE,
                (unsigned)PULSAR_N_EMBD);
    }

    /* the vision tower binds before the inspect-only exit for the same reason
     * the drafter does: --inspect proves the whole artifact binds */
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
        pulsar_tp_identity id;
        pulsar_tp_identity_init_defaults(&id,
                                         (uint64_t)e->model.size,
                                         (uint32_t)e->model_id(),
                                         PULSAR_N_LAYER,
                                         PULSAR_N_EMBD,
                                         PULSAR_N_VOCAB,
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
        if (opt->tp_spill_dir && opt->tp_spill_dir[0]) e->tp_spill_dir = pulsar_strdup(opt->tp_spill_dir);
        e->tp_slab_bytes = pulsar_tp_slab_bytes(PULSAR_N_LAYER, PULSAR_N_EMBD);
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
        fprintf(stderr, "pulsar: TP rank %d/%d armed (prefill big-gate), slab %zu bytes, "
                        "%.2f GiB of peer-owned experts not resident\n",
                pulsar_tp_rank(e->tp), pulsar_tp_n_ranks(e->tp), e->tp_slab_bytes,
                (double)pulsar_model_peer_expert_bytes(&e->model) / 1073741824.0);
    }

    /* Slice 4g (L241): the attention OUTPUT GROUPS this rank owns, from the one
     * range authority every split shares.  The unit is the group (8 heads and
     * one LoRA-down block each), never the head: a rank's heads are whole
     * groups, so its attn_q_b rows and its attn_output_a rows are contiguous
     * and 128-row aligned.  On a group each layer's two owned row slices are
     * registered with the backend here, once; the attention block then
     * addresses them by offset like any other weight.  One box: [0, n). */
    if (graph_backend) {
        const int tp_rk = e->tp ? pulsar_tp_rank(e->tp) : 0;
        const uint32_t tp_nr = e->tp ? pulsar_tp_n_ranks(e->tp) : 1u;
        if (!pulsar_tp_owned_range(tp_rk, tp_nr, PULSAR_N_OUT_GROUP, &e->tp_group_lo, &e->tp_group_hi) ||
            e->tp_group_hi <= e->tp_group_lo) {
            fprintf(stderr, "pulsar: TP rank %d/%u owns no attention output group (%u groups per "
                            "layer; a group of more than %u ranks cannot split attention) -- refusing\n",
                    tp_rk, tp_nr, (unsigned)PULSAR_N_OUT_GROUP, (unsigned)PULSAR_N_OUT_GROUP);
            e->destroy();
            *out = NULL;
            return 1;
        }
        if (tp_nr > 1) {
            const uint32_t group_heads = PULSAR_N_HEAD / PULSAR_N_OUT_GROUP;
            const uint64_t q_out_full = (uint64_t)PULSAR_N_HEAD * PULSAR_N_HEAD_DIM;
            const uint64_t q_lo = (uint64_t)e->tp_group_lo * group_heads * PULSAR_N_HEAD_DIM;
            const uint64_t q_hi = (uint64_t)e->tp_group_hi * group_heads * PULSAR_N_HEAD_DIM;
            const uint64_t group_dim = (uint64_t)group_heads * PULSAR_N_HEAD_DIM;
            const uint64_t a_out_full = (uint64_t)PULSAR_N_OUT_GROUP * PULSAR_N_LORA_O;
            const uint64_t a_lo = (uint64_t)e->tp_group_lo * PULSAR_N_LORA_O;
            const uint64_t a_hi = (uint64_t)e->tp_group_hi * PULSAR_N_LORA_O;
            uint32_t registered = 0;
            for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
                const pulsar_layer_weights *L = &e->weights.layer[il];
                if (!L->attn_q_a || !L->attn_q_b || !L->attn_output_a) {
                    fprintf(stderr, "pulsar: layer %u has no attention projections to split -- refusing\n", il);
                    e->destroy();
                    *out = NULL;
                    return 1;
                }
                const uint64_t q_rank = L->attn_q_a->dim[1];
                if (!pulsar_gpu_register_fp8_lt_row_slice(tensor_map_base(&e->model, L->attn_q_b),
                                                          L->attn_q_b->abs_offset, q_rank, q_out_full, q_lo, q_hi) ||
                    !pulsar_gpu_register_fp8_lt_row_slice(tensor_map_base(&e->model, L->attn_output_a),
                                                          L->attn_output_a->abs_offset, group_dim, a_out_full, a_lo, a_hi)) {
                    fprintf(stderr, "pulsar: layer %u: the owned attention row slices could not be "
                                    "registered -- refusing\n", il);
                    e->destroy();
                    *out = NULL;
                    return 1;
                }
                registered += 2;
                if (!tp_register_shared_split(e, &e->model, L, tp_rk, tp_nr)) {
                    fprintf(stderr, "pulsar: layer %u: the owned shared-expert split could not be "
                                    "registered -- refusing\n", il);
                    e->destroy();
                    *out = NULL;
                    return 1;
                }
                registered += 3;
            }
            for (uint32_t dl = 0; e->dspark_ready && dl < 3u; dl++) {
                if (!tp_register_shared_split(e, &e->dspark_model, &e->dspark_weights.layer[dl], tp_rk, tp_nr)) {
                    fprintf(stderr, "pulsar: drafter block %u: the owned shared-expert split could not be "
                                    "registered -- refusing\n", dl);
                    e->destroy();
                    *out = NULL;
                    return 1;
                }
                registered += 3;
            }
            const pulsar_layer_weights *L0 = &e->weights.layer[0];
            uint32_t sx_lo = 0, sx_hi = 0;
            (void)pulsar_tp_owned_range(tp_rk, tp_nr, (uint32_t)L0->ffn_gate_shexp->dim[1], &sx_lo, &sx_hi);
            fprintf(stderr, "pulsar: TP rank %d/%u owns attention output groups [%u,%u) of %u = heads "
                            "[%u,%u) (attn_q_b rows [%llu,%llu), attn_output_a rows [%llu,%llu)) and "
                            "shared-expert intermediate [%u,%u) of %u (gate/up row slices, down K-half, "
                            "%u layers + %u drafter blocks): %u slices registered\n",
                    tp_rk, tp_nr, e->tp_group_lo, e->tp_group_hi, (unsigned)PULSAR_N_OUT_GROUP,
                    e->tp_group_lo * group_heads, e->tp_group_hi * group_heads,
                    (unsigned long long)q_lo, (unsigned long long)q_hi,
                    (unsigned long long)a_lo, (unsigned long long)a_hi,
                    sx_lo, sx_hi, (unsigned)L0->ffn_gate_shexp->dim[1],
                    (unsigned)PULSAR_N_LAYER, e->dspark_ready ? 3u : 0u, registered);
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
    return e ? e->vocab.n_vocab : 0;
}


/* The engine's logits ROW WIDTH — the shape profile's n_vocab, which is what
 * every logits buffer the engine writes is strided by.  This is NOT
 * pulsar_engine_vocab_size (the tokenizer table length): the loader never checks
 * the two against each other, and sizing a logits buffer from the tokenizer
 * length is exactly the mismatch that produced an unbounded-logits write. */
int pulsar_engine::logits_width() const {
    auto *e = this;
    return e ? (int)PULSAR_N_VOCAB : 0;
}


const char *pulsar_engine::model_name() {
    auto *e = this;
    (void)e;
    return PULSAR_MODEL_SHAPE_NAME;
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
    out->has_dspark = e->dspark_ready;
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
    uint64_t bytes = e->model.mapped_bytes - pulsar_model_peer_expert_bytes(&e->model);
    if (e->dspark_ready && e->dspark_external) {
        bytes += e->dspark_model.mapped_bytes - pulsar_model_peer_expert_bytes(&e->dspark_model);
    }
    if (e->overlay_ready) bytes += e->overlay_model.mapped_bytes;
    return bytes;
}


int pulsar_engine::model_id() {
    auto *e = this;
    (void)e;
    return (int)PULSAR_MODEL_VARIANT;
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
        pulsar_tp_timing_report(e->tp);   /* TEMPORARY INSTRUMENT (4g-2 step 1) */
        (void)pulsar_tp_send_stop(e->tp);
        pulsar_tp_free(e->tp);
        e->tp = NULL;
    }
    free(e->tp_spill_dir);
    e->tp_spill_dir = NULL;
    if (e->tp_slab_base) {
        pulsar_tp_gpu_slab_free_hostpin(e->tp_slab_base);
        e->tp_slab_base = NULL;
        e->tp_slab_dev = NULL;
        e->tp_slab_bytes = 0;
    }
    weights_free(&e->weights);
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


int pulsar_session::create(pulsar_session **out, pulsar_engine *e, int ctx_size) {
    if (!out || !e || ctx_size <= 0) return 1;
    if (!pulsar_backend_uses_graph(e->backend) || !e->gpu_ready) return 1;

    pulsar_session *s = (pulsar_session *)xcalloc(1, sizeof(*s));
    s->engine = e;
    s->ctx_size = ctx_size;
    s->prefill_frontier = 0;   /* L195: nothing prefilled yet */
    s->prefill_cap = gpu_graph_prefill_cap_for_prompt(ctx_size,
                                                        e->prefill_chunk);
    const uint32_t raw_cap = gpu_graph_raw_cap_for_context(ctx_size, s->prefill_cap);
    const pulsar_layer_weights *shape_layer = weights_first_bound_layer(&e->weights);
    if (!shape_layer) {
        fprintf(stderr, "pulsar: no transformer layers are loaded\n");
        free(s);
        return 1;
    }
    /* Measure the true GPU cost of this session (allocator delta across the
     * create) so callers can reconcile admission estimates against reality. */
    const uint64_t alloc_before = pulsar_gpu_tensor_alloc_bytes_current();
    if (!gpu_graph_alloc_raw_cap(&s->graph, &e->weights, shape_layer,
                                   raw_cap, (uint32_t)ctx_size, s->prefill_cap,
                                   gpu_graph_bank_pool_n(), e->dspark_ready))
    {
        free(s);
        return 1;
    }
    if (!gpu_graph_load_directional_steering(&s->graph,
                                               e->directional_steering_file,
                                               e->directional_steering_attn_scale,
                                               e->directional_steering_ffn_scale)) {
        gpu_graph_free(&s->graph);
        free(s);
        return 1;
    }
    /* Borrow the engine's TP transport into the graph so the prefill big-gate
     * call sites can reach it without threading the engine through every
     * gpu_graph entry point (slice 4b).  NULL when the pair is not armed. */
    s->graph.tp = e->tp;
    s->graph.tp_group_lo = e->tp_group_lo;
    s->graph.tp_group_hi = e->tp_group_hi;
    s->graph.tp_slab_dev = e->tp_slab_dev;
    s->graph.tp_kslice_key = e->tp ? (const void *)e : NULL;
    /* Slice 4e: the mirror id both ranks agree on by construction.  Assigned
     * here, at the one place a session begins, from the engine's ordinal; a
     * session created with no pair armed keeps 0 and stays out of the mirror. */
    if (e->tp) s->tp_session_id = ++e->tp_session_seq;
    s->logits = (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(s->logits[0]));
    if (e->dspark_ready) {
        if (!gpu_graph_init_dspark_target(&s->graph, e->dspark_weights.target_layer_ids)) {
            fprintf(stderr, "pulsar: failed to allocate DSpark graph buffers\n");
            gpu_graph_free(&s->graph);
            free(s->logits);
            free(s);
            return 1;
        }
    }
    s->resident_bytes = pulsar_gpu_tensor_alloc_bytes_current() - alloc_before;
    *out = s;
    return 0;
}


uint64_t pulsar_session_resident_bytes(const pulsar_session *s) {
    return s ? s->resident_bytes : 0;
}


/* The price of pulsar_session::create at this context size IS the create: the
 * same three allocation steps run with the tensor primitives in dry mode
 * (pulsar_gpu_tensor_dry_begin), so the number admission control charges and
 * the number the allocator commits come from one piece of code -- the server
 * checks them equal after every create.  Not in the price, on either side:
 * allocations made on demand after create (gpu_graph_ensure_batch_ffn_out,
 * the multiseq descriptors, the batched-copy descriptor tables); the server's
 * memory floor absorbs those. */
uint64_t pulsar_engine::session_cost_bytes_banked(int ctx_size, int n_banks) {
    auto *e = this;
    if (!e || ctx_size <= 0 || n_banks < 1) return 0;
    if (!pulsar_backend_uses_graph(e->backend) || !e->gpu_ready) return 0;
    const uint32_t prefill_cap = gpu_graph_prefill_cap_for_prompt(ctx_size,
                                                                  e->prefill_chunk);
    const uint32_t raw_cap = gpu_graph_raw_cap_for_context(ctx_size, prefill_cap);
    const pulsar_layer_weights *shape_layer = weights_first_bound_layer(&e->weights);
    if (!shape_layer) return 0;
    pulsar_gpu_graph g;
    pulsar_gpu_tensor_dry_begin();
    const bool ok =
        gpu_graph_alloc_raw_cap(&g, &e->weights, shape_layer, raw_cap,
                                (uint32_t)ctx_size, prefill_cap, (uint32_t)n_banks,
                                e->dspark_ready) &&
        gpu_graph_load_directional_steering(&g, e->directional_steering_file,
                                            e->directional_steering_attn_scale,
                                            e->directional_steering_ffn_scale) &&
        (!e->dspark_ready ||
         gpu_graph_init_dspark_target(&g, e->dspark_weights.target_layer_ids));
    uint64_t bytes = 0;
    pulsar_gpu_tensor_dry_end(&bytes, NULL);
    gpu_graph_release(&g);
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

uint64_t pulsar_session::touched_kv_bytes() const {
    auto *s = this;
    if (!s) return 0;
    return gpu_graph_touched_kv_bytes(&s->graph);
}


void pulsar_session::destroy() {
    auto *s = this;
    if (!s) return;
    gpu_graph_free(&s->graph);
    token_vec_free(&s->checkpoint);
    pulsar_sample_scratch_free(&s->sample_scratch);
    s->bank_carry_free();
    free(s->dspark_pending_qrows);
    free(s->spec_row_scratch);
    free(s->logits);
    free(s);
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


static bool pulsar_session_cancelled(pulsar_session *s) {
    /* Slice 4e increment 5 (L238): a MIRRORED session never cancels inside an
     * operation.  The hook is polled at prefill chunk boundaries; a leader
     * that stopped after k chunks would leave its workers running the full
     * sync and waiting at a big gate the leader never joins -- a data-plane
     * hang the control plane cannot see.  Under TP, cancellation is between
     * operations only: the driver simply issues no further frame. */
    if (pulsar_session_is_mirrored(s)) return false;
    return s && s->cancel && s->cancel(s->cancel_ud);
}


static bool pulsar_session_cancelled_cb(void *ud) {
    return pulsar_session_cancelled((pulsar_session *)ud);
}


/* Identity of an image SET as it enters the KV: order, positions, byte lengths
 * and the bytes themselves.  The sentinel block's token IDs encode only its
 * geometry (vocab_size + role), so two different images of the same size produce
 * the SAME token prefix -- reuse must be decided on the pixels, not the ids.
 * FNV-1a, and 0 is reserved for "no images". */
static uint64_t image_set_fingerprint(const pulsar_image_ref *images, int n) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t *nul = (const uint8_t *)"\0";
    for (int i = 0; i < n; i++) {
        const uint8_t *p = (images && images[i].bytes) ? images[i].bytes : nul;
        const size_t len = (images && images[i].bytes) ? images[i].len : 0;
        const uint64_t fields[2] = { (uint64_t)(images ? images[i].start_pos : -1), (uint64_t)len };
        for (int f = 0; f < 2; f++) {
            for (int b = 0; b < 8; b++) {
                h ^= (fields[f] >> (8 * b)) & 0xffu;
                h *= 1099511628211ull;
            }
        }
        for (size_t b = 0; b < len; b++) {
            h ^= p[b];
            h *= 1099511628211ull;
        }
    }
    return h ? h : 1u;
}



static void pulsar_session_note_prefill_progress(void *ud, const char *event, int current, int total) {
    pulsar_sync_progress *p = (pulsar_sync_progress *)ud;
    if (!p || !p->session || !p->prompt) return;
    if (!strcmp(event, "prefill_chunk") && current > 0 && current <= p->prompt->len) {
        p->session->checkpoint.len = 0;
        for (int i = 0; i < current; i++) token_vec_push(&p->session->checkpoint, p->prompt->v[i]);
        p->session->checkpoint_valid = true;
        p->session->prefill_frontier = current;   /* L195: a prefill wrote up to here */
    }
    if (p->user) p->user(p->user_ud, event, current, total);
}


/* Bring the live backend state to exactly the supplied token prefix.
 *
 * pulsar-server and the REPL are stateless at the text/API layer but stateful here:
 * they resend or rebuild the full transcript, and this function decides whether
 * the live checkpoint is a prefix.  A matching prefix is extended in one of two
 * ways:
 *
 *   - long suffix: batched layer-major prefill, aligned to absolute chunk
 *     boundaries so compressor/indexer rows finalize in the same order as a
 *     cold prompt;
 *   - short suffix: ordinary one-token decode, which is faster below the
 *     measured crossover and preserves exact autoregressive semantics.
 *
 * A non-matching prompt discards the checkpoint and prefills from token zero.
 */
int pulsar_session::sync(const pulsar_tokens *prompt, const pulsar_image_ref *images,
                         int n_images, char *err, size_t errlen) {
    auto *s = this;
    s->resume_origin = -1;
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return 1;
    }
    if (pulsar_session_cancelled(s)) {
        snprintf(err, errlen, "interrupted");
        return PULSAR_SESSION_SYNC_INTERRUPTED;
    }
    pulsar_engine *e = s->engine;
    const char *backend_name = pulsar_backend_name(e->backend);

    /* The image path is refused until the artifact actually carries a bound,
     * layout-validated tower -- which is what vision_ready has always meant.
     * An image request is then a COLD prefill from token 0: the reference merges
     * only on the start_pos == 0 pass and asserts that no sentinel id survives a
     * continuation, so neither cache-reuse path may run (the extend path and the
     * L115 seam rescue are both gated on checkpoint_valid). */
    /* Exclusive end of the LAST image block in this prompt (0 when none): the
     * resume path below must never RE-EVALUATE a row inside it.  A sentinel row
     * only carries values because the merge wrote them; a re-evaluated one would
     * be zero-masked by the embedder (its id is out of vocab). */
    int image_barrier = 0;
    uint32_t resume_floor = 0;   /* the grid point a licensed image reuse must start from */
    if (n_images > 0) {
        if (!e->vision_ready) {
            snprintf(err, errlen, "this model has no vision tower bound; it cannot accept images");
            return 1;
        }
        for (int i = 0; i < n_images; i++) {
            if (!images || !images[i].bytes || images[i].len == 0 || images[i].start_pos < 0) {
                snprintf(err, errlen, "image %d has no bytes or a bad span position", i);
                return 1;
            }
            int span_len = 0;
            if (!vision_span_extent(prompt->v, prompt->len, (int)PULSAR_N_VOCAB,
                                    images[i].start_pos, &span_len) ||
                span_len <= 0) {
                snprintf(err, errlen, "image %d at %d is not a sentinel block in this prompt",
                         i, images[i].start_pos);
                return 1;
            }
            if (images[i].start_pos + span_len > image_barrier)
                image_barrier = images[i].start_pos + span_len;
        }
        /* L226: an image ALREADY inside the live KV is not a reason to redo the
         * prompt.  Reuse is licensed when this session's checkpoint holds those
         * blocks AND holds THESE IMAGES: the fingerprint is what makes that true,
         * since block ids are geometry (see image_set_fingerprint).  Licensing it
         * is all that is needed -- the carry path below and the L115 seam rescue
         * then do the work, and `image_barrier` (see the resume) keeps either one
         * from re-evaluating a merged row.  Anything else -- a new image, a
         * different image of the same size, a block the checkpoint does not cover
         * -- clears checkpoint_valid exactly as before, so the cold rebuild (the
         * one pass that merges) runs.
         *
         * No sentinel id can reach a cache surface this way.  The server never
         * plans a cold store for an image request (server_jobs.cpp gates the
         * whole disk/prefix resolver on !image_request), and a LATER prompt whose
         * sentinel ids outlive their images is still refused by the scan below. */
        const uint64_t request_fp = image_set_fingerprint(images, n_images);
        /* A bank whose compressor state is STALE (a mid-group rewind with no state
         * coverage) can only be joined at a group boundary: the per-row producer
         * refuses the store anywhere else, so extending it fails the request
         * outright (measured: "store at 170 would extend a stale pending group --
         * refusing" -> HTTP 400 on the third turn of an image conversation).  The
         * cold rebuild is the pass that rebuilds that state from scratch, so a
         * stale bank declines reuse. */
        const bool bank_extendable =
            !s->graph.ms_comp_state_stale[gpu_graph_cur_bank(&s->graph)];
        /* The resume must start at a PULSAR_RESUME_GRID multiple: that is what
         * makes a resumed prefill reproduce the cold one byte for byte (chunk
         * boundaries and kernel calls are the cold prefill's, and every grid point
         * is an empty compressor group), and it must start at or above the barrier
         * so no merged row is re-evaluated.  Both together mean reuse is legal only
         * when such a grid point fits inside the checkpoint AND the raw ring can
         * still replay from it.  Otherwise there is no byte-identical extension to
         * take and the cold rebuild runs -- which is the common answer early in a
         * conversation, when the images are near the frontier. */
        resume_floor = image_barrier > 0
            ? (((uint32_t)image_barrier + PULSAR_RESUME_GRID - 1u) / PULSAR_RESUME_GRID) * PULSAR_RESUME_GRID
            : 0u;
        const uint32_t raw_reach = s->graph.raw_cap > s->graph.raw_window
                                 ? s->graph.raw_cap - s->graph.raw_window : 0u;
        const bool grid_reachable =
            resume_floor > 0 && resume_floor <= (uint32_t)s->checkpoint.len &&
            (uint32_t)s->checkpoint.len - resume_floor <= raw_reach;
        const bool images_all_live =
            s->checkpoint_valid &&
            bank_extendable &&
            grid_reachable &&
            s->live_image_fp == request_fp &&
            s->live_image_barrier == image_barrier;
        if (!images_all_live) {
            s->checkpoint_valid = false;
            if (!bank_extendable)
                fprintf(stderr, "pulsar: image request: %d image(s) are live but this bank's "
                                "compressor state is stale -- rebuilding cold\n", n_images);
        } else {
            fprintf(stderr, "pulsar: image request: %d image(s) already live (prefix %d tokens, "
                            "blocks end at %d) -- reuse licensed\n",
                    n_images, s->checkpoint.len, image_barrier);
        }
    } else {
        /* A prompt carrying sentinel ids with no image to fill them would prefill
         * rows whose embeddings never arrived -- the embedder zero-masks an
         * out-of-vocab id, and only the merge puts anything there -- so refuse it
         * here instead of silently serving a wrong answer.  A tokenizer never
         * emits an id at or above vocab_size, so ANY such id is a sentinel.  The
         * PLACEHOLDER id is in-vocab and would embed as ordinary text, so it is
         * refused too: it is a renderer artifact that
         * pulsar_expand_image_placeholders() must have replaced. */
        for (int i = 0; i < prompt->len; i++) {
            if (prompt->v[i] >= (int)PULSAR_N_VOCAB || prompt->v[i] == e->vocab.image_id) {
                snprintf(err, errlen, "prompt token %d is image sentinel id %d, but the request "
                                      "carries no images", i, prompt->v[i]);
                return 1;
            }
        }
    }

    /* L226: this sync re-establishes whatever a salvaged rewind left open -- the
     * carry path re-prefills from a grid point at or above the salvage floor and
     * the rebuild path prefills from 0 -- so a decode is legal again once it
     * returns.  Cleared here rather than in either arm because BOTH make the
     * session decodable again (and the carry path returns early). */
    s->kv_salvaged = false;
    /* a sync begins a new request: any carry left by a max-tokens/stop-string
     * truncated generation belongs to the previous request's distribution.
     * (position stamping alone misses a same-length full rebuild.) */
    s->spec.spec_carry_valid = false;
    /* Same argument, same blind spot: the pendings' position stamp cannot see a
     * rebuild that lands on the same length, and a sampled draft's q belongs to
     * the previous request's distribution. Dropping them here costs one draft
     * round at the start of a request and is the only guard that covers it. */
    pulsar_spec_drop_pendings(&s->spec);
    /* A sync begins a new request: re-arm the terminal yield quench. */
    spec_quench_reset(s);

    if (s->checkpoint_valid &&
        prompt->len >= s->checkpoint.len &&
        pulsar_tokens_starts_with(prompt, &s->checkpoint))
    {
        /* L148 self-heal.  The checkpoint says the installed bank holds
         * checkpoint.len tokens; the bank's compressed frontier says how far
         * its KV actually reached.  A frontier AHEAD of the checkpoint means
         * the bank kept decoding after this checkpoint was recorded and the
         * history was never folded back (the plain batched lane keeps
         * generated tokens server-side and folds them late; the spec lane
         * appends per round).  Trusting the checkpoint then either returns
         * early (suffix 0 -- the L148 repro: a 32-token repeat prompt on a
         * bank at 288, "frontier not position-true ... n_comp 72 want 8") or
         * prefills a suffix onto rows the frontier already counts.  A counter
         * can only be AHEAD of a position it must be rewound to, so rewind to
         * the checkpoint first: L120's clamp, applied where the stale copy is
         * consumed.  Cost: one host loop over the layers per sync. */
        {
            const uint32_t bank = gpu_graph_cur_bank(&s->graph);
            bool ahead = false;
            for (uint32_t il = 0; il < PULSAR_N_LAYER && !ahead; il++) {
                if (!gpu_graph_layer_is_kv_source(il)) continue;
                if (gpu_graph_n_comp(&s->graph, bank, il) > (uint32_t)s->checkpoint.len / pulsar_layer_compress_ratio(il))
                    ahead = true;
            }
            if (ahead) s->rewind(s->checkpoint.len);
        }
        /* L183/L194/L195/L218: a resume is a COLD PREFILL FROM A GRID POINT.  A
         * prefill chunk's bytes depend on the chunk's row count and on a row's
         * offset within the call (L183), so a suffix evaluated from an off-grid
         * checkpoint is a different computation from the cold prefill of the same
         * tokens.  The rule: G = the last multiple of PULSAR_RESUME_GRID at or
         * below the PREFILL frontier (decode rows are the decode kernels'; the
         * tokens generated since are recomputed), rewind to G -- at an even
         * position every compressor holds the empty group, so the rewind leaves
         * the cold prefill's state and nothing needs warming up -- then evaluate
         * [G, N): every chunk boundary and every kernel call is the cold
         * prefill's.  Cost: < 128 + generated tokens recomputed.  The raw window
         * the resumed prefill attends over must still be in the ring: a
         * generation longer than the ring's reach since the last prefill leaves
         * only the cold prefill from 0, said once. */
        if (prompt->len > s->checkpoint.len && s->prefill_cap != 0) {
            const uint32_t bank = gpu_graph_cur_bank(&s->graph);
            const uint32_t ck = (uint32_t)s->checkpoint.len;
            uint32_t pf = s->prefill_frontier < 0 ? 0u : (uint32_t)s->prefill_frontier;
            if (pf > ck) pf = ck;
            uint32_t G = (pf / PULSAR_RESUME_GRID) * PULSAR_RESUME_GRID;   /* may move to a reachable grid point below */
            const uint32_t reach = s->graph.raw_cap > s->graph.raw_window
                                 ? s->graph.raw_cap - s->graph.raw_window : 0u;
            /* When the prefill grid point is past the ring's reach (a generation
             * longer than ~4k tokens since the last prefill), resume from the
             * NEWEST grid point within reach instead of from 0.  The tokens below
             * it were generated by this very session and stay as decode wrote
             * them, so that one turn is not byte-identical to a fresh request
             * (the same standard vLLM / llama.cpp hold everywhere); the
             * alternative was a 50 s cold prefill (dogfood 2026-09-06 14:02).
             * Said once per resume. */
            bool exact = true;
            if (G != 0 && ck - G > reach) {
                const uint32_t lo = ck - reach;   /* oldest position whose attention window is still in the ring */
                const uint32_t G2 = ((lo + PULSAR_RESUME_GRID - 1u) / PULSAR_RESUME_GRID) * PULSAR_RESUME_GRID;
                if (G2 >= PULSAR_RESUME_GRID && G2 < ck) { G = G2; exact = false; }
                else G = 0;   /* nothing reachable: cold */
            }
            /* L226: an image request's reuse may not RE-EVALUATE a row inside an
             * image block, and it must still start on the resume grid so the result
             * is the cold prefill's byte for byte.  The licence above proved
             * `resume_floor` fits inside the checkpoint and is reachable, so raising
             * G to it is the same move the ring-reach rule already makes -- a NEWER
             * start, never an older one. */
            if (resume_floor > 0 && G < resume_floor) G = resume_floor;
            if (G == ck) {
                s->resume_origin = (int)ck;   /* the checkpoint is a prefill grid point: nothing to redo */
            } else if (G == 0) {
                if (ck >= PULSAR_RESUME_GRID) {
                    fprintf(stderr, "pulsar: resume at %u on bank %u: no grid point within the raw ring's reach -- "
                                    "prefilling the prompt from 0\n", ck, bank);
                }
                s->rewind(0);
                s->resume_origin = 0;
            } else {
                s->rewind((int)G);
                if (!s->checkpoint_valid) {
                    /* The compressor state at G could not be re-established: no
                     * projection-ring coverage and no boundary stash for that row
                     * (L221).  Fall back to the cold path the `G == 0` arm above
                     * already takes, rather than failing the request -- a rebuild
                     * is slower, not wrong, and the alternative is decoding the
                     * group that straddles G against a lane this rewind wiped. */
                    fprintf(stderr, "pulsar: resume at %u on bank %u: compressor state at grid point %u "
                                    "could not be re-established -- prefilling the prompt from 0\n",
                            ck, bank, G);
                    s->rewind(0);
                    s->resume_origin = 0;
                } else {
                    s->resume_origin = (int)G;
                    fprintf(stderr, "pulsar: resume at %u from grid point %u on bank %u (%u tokens recomputed)%s\n",
                            ck, G, bank, ck - G,
                            exact ? "" : " -- past the prefill frontier's reach: the generated tokens below stay as decoded, "
                                         "not identical to a cold prefill");
                }
            }
        }
        const int suffix = prompt->len - s->checkpoint.len;
        if (suffix > 0) {
            bool cancelled = false;
            pulsar_sync_progress progress = {
                .session = s,
                .prompt = prompt,
                .user = s->progress,
                .user_ud = s->progress_ud,
            };
            bool ok = gpu_graph_prefill_chunked_range(&s->graph,
                                                        &e->model,
                                                        &e->weights,
                                                        prompt,
                                                        (uint32_t)s->checkpoint.len,
                                                        (uint32_t)suffix,
                                                        s->logits,
                                                        false,
                                                        pulsar_session_note_prefill_progress,
                                                        &progress,
                                                        s->display_progress,
                                                        s->display_progress_ud,
                                                        NULL,
                                                        pulsar_session_cancelled_cb,
                                                        s,
                                                        &cancelled);
            if (cancelled) {
                snprintf(err, errlen, "interrupted");
                s->checkpoint_valid = true;
                return PULSAR_SESSION_SYNC_INTERRUPTED;
            }
            if (!ok) {
                snprintf(err, errlen, "%s resumed prefill failed while extending checkpoint", backend_name);
                s->checkpoint_valid = false;
                return 1;
            }
            pulsar_tokens_copy(&s->checkpoint, prompt);
            s->checkpoint_valid = true;
            s->prefill_frontier = prompt->len;   /* L195 */
            return 0;
        }

        /* L131: suffix == 0 means the checkpoint already IS the prompt --
         * nothing to evaluate.  The single-token fallback that used to live
         * here is gone with its encoder; every positive suffix takes the
         * batched branch above. */
        return 0;
    }

    /* L115 seam rescue: before surrendering to a full rebuild, check whether
     * the id mismatch is only sampled-vs-canonical TOKEN BOUNDARY drift.
     * If the prompt's bytes match the live history's bytes up to a shared
     * boundary past the id divergence, the live KV IS this conversation's
     * true history (it carries the boundaries the model actually sampled) —
     * keep it: rewind to the matched live token, stitch
     * live[0..live_n) + prompt[prompt_n..], and re-enter sync, which now
     * takes the extend path.  One recursion level by construction (the
     * stitched prompt starts_with the rewound checkpoint). */
    if (s->checkpoint_valid) {
        pulsar_prefix_match m;
        s->prefix_match(prompt, &m);
        const int live_n = m.live_cut, prompt_n = m.prompt_cut;
        /* Fires for every shape that reaches here with reusable live bytes:
         *   - SEAM: live_n > id-common (sampled vs canonical boundaries);
         *   - SHORTER ECHO: the client strips generated reasoning, so live
         *     carries a tail the prompt does not (live_n < checkpoint.len)
         *     -- measured 2026-08-28, live 390,258 vs echo 390,018;
         *   - ROLLBACK/COMPACTION: the prompt is a strict prefix of live.
         * All three are the same conversation, so the rewind+stitch below
         * beats a rebuild; stitching is never worse (prompt_n >= 0).
         *
         * L226: an IMAGE request may take this route too, and it is the route
         * that matters for a multi-turn image conversation -- the client replays
         * the visible reply, so the prompt is SHORTER than the live history and
         * the exact-prefix license above can never fire.  The images ride along
         * into the re-entry only when every block lies WHOLLY below the stitch
         * point: the stitched prompt's first live_n tokens are the live
         * checkpoint's, so that is exactly the condition under which the blocks
         * survive the stitch intact.  A block that the stitch would cut (or an
         * image the client swapped for a different one) declines the stitch and
         * keeps the cold rebuild -- today's behaviour -- rather than merging at a
         * start_pos the stitched prompt no longer has. */
        bool images_survive_stitch = true;
        for (int i = 0; n_images > 0 && i < n_images && images_survive_stitch; i++) {
            int span_len = 0;
            if (!images || images[i].start_pos < 0 ||
                !vision_span_extent(s->checkpoint.v, live_n, (int)PULSAR_N_VOCAB,
                                    images[i].start_pos, &span_len) ||
                images[i].start_pos + span_len > live_n)
            {
                images_survive_stitch = false;
            }
        }
        if (live_n > 0 && images_survive_stitch) {
            s->rewind(live_n);
            pulsar_tokens stitched;
            memset(&stitched, 0, sizeof(stitched));
            stitched.v = (int *)xmalloc(
                    (size_t)(live_n + (prompt->len - prompt_n)) * sizeof(int));
            stitched.cap = live_n + (prompt->len - prompt_n);
            memcpy(stitched.v, s->checkpoint.v, (size_t)live_n * sizeof(int));
            memcpy(stitched.v + live_n, prompt->v + prompt_n,
                   (size_t)(prompt->len - prompt_n) * sizeof(int));
            stitched.len = stitched.cap;
            const int rc = s->sync(&stitched, n_images > 0 ? images : NULL,
                                   n_images > 0 ? n_images : 0, err, errlen);
            free(stitched.v);
            return rc;
        }
    }

    /* The images are BORROWED for exactly this prefill.  The driver reads them
     * off the graph so that four prefill signatures do not grow a parameter that
     * only this caller can ever fill; the scope clears the borrow on every exit,
     * including the interrupted ones, so no later decode can see it. */
    struct vision_scope {
        pulsar_gpu_graph *g;
        const pulsar_vision_request *prev;
        vision_scope(pulsar_gpu_graph *g_, const pulsar_vision_request *r)
            : g(g_), prev(g_->vision_req) { g->vision_req = r; }
        ~vision_scope() { g->vision_req = prev; }
    };
    pulsar_vision_request vreq = { images, n_images, &e->vision_weights };
    vision_scope vscope(&s->graph, n_images > 0 ? &vreq : NULL);

    bool ok;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    if (!gpu_graph_reset_prefill_state(&s->graph)) {
        snprintf(err, errlen, "%s prefill state reset failed", backend_name);
        return 1;
    }
    /* The rebuild path is the one place this session's per-bank truth is
     * legitimately re-established: reset_prefill_state zeroes
     * ms_n_comp[cur_bank] and the prefill below
     * refills them from zero against the installed bank.  Other banks hold
     * other slots' positions and a reset here says nothing about them.
     *
     * (Before stage 1b this zeroed the scalar twins, whose second job was
     * clearing a stale multiseq superset; there is no superset any more, so
     * that job went with them.  The prefix-resume path above still cannot be
     * reached while dirty — decode_multiseq clears checkpoint_valid, which
     * that path gates on.) */
    s->mseq_dirty = false;
    if (s->prefill_cap < (uint32_t)prompt->len) {
        bool cancelled = false;
        pulsar_sync_progress progress = {
            .session = s,
            .prompt = prompt,
            .user = s->progress,
            .user_ud = s->progress_ud,
        };
        ok = gpu_graph_prefill_chunked(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false,
                                         pulsar_session_note_prefill_progress, &progress,
                                         s->display_progress,
                                         s->display_progress_ud,
                                         pulsar_session_cancelled_cb,
                                         s,
                                         &cancelled);
        if (cancelled) {
            snprintf(err, errlen, "interrupted");
            s->checkpoint_valid = s->checkpoint.len > 0;
            return PULSAR_SESSION_SYNC_INTERRUPTED;
        }
    } else {
        bool cancelled = false;
        ok = gpu_graph_prefill_raw_swa(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false,
                                         s->display_progress,
                                         s->display_progress_ud,
                                         pulsar_session_cancelled_cb,
                                         s,
                                         &cancelled);
        if (cancelled) {
            snprintf(err, errlen, "interrupted");
            return PULSAR_SESSION_SYNC_INTERRUPTED;
        }
    }
    if (!ok) {
        snprintf(err, errlen, "%s prefill failed", backend_name);
        s->checkpoint_valid = false;
        return 1;
    }
    pulsar_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    s->prefill_frontier = prompt->len;   /* L195 */
    /* A rebuild replaces what the checkpoint describes, so the image identity
     * goes with it: this pass merged THESE images (barrier included), or there
     * are none in the prompt at all.  A carry/extension keeps the previous
     * identity -- its blocks are still in the prefix -- and rewind() clears it
     * when a truncation drops one. */
    s->live_image_fp = n_images > 0 ? image_set_fingerprint(images, n_images) : 0;
    s->live_image_barrier = n_images > 0 ? image_barrier : 0;
    return 0;
}


/* Return true when canonicalization would replace already-sampled tokens.
 *
 * A DS4 session checkpoint is more than a token vector: the backend state also
 * contains raw SWA rows, compressed KV rows, indexer rows, and compressor
 * frontiers.  Replacing any part of the live tail requires restoring that whole
 * frontier first.  Extending exactly at the live end is safe; rewriting behind
 * it is not an in-place operation. */
bool pulsar_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common) {
    if (live_len < 0 || canonical_len < 0 || common < 0) return true;
    if (common > live_len || common > canonical_len) return true;
    return common < live_len;
}


/* Replace the live suffix after a shared prefix.
 *
 * This is used after parsing a generated tool call.  The model may have emitted
 * DSML in an order that is semantically valid but not byte-for-byte equal to the
 * canonical prompt we will see on the next request.  Rewriting only the token
 * checkpoint is not enough: the backend still contains raw and compressed rows
 * for the old suffix.  Until we have a real frontier snapshot at the
 * rewrite point, any replacement behind the live end reports that a rebuild is
 * needed without mutating the session.  The server may still find an older disk KV
 * checkpoint before falling back to a full replay. */
pulsar_session_rewrite_result pulsar_session::rewrite_from_common(const pulsar_tokens *prompt, int common,
        char *err, size_t errlen) {
    auto *s = this;
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return PULSAR_SESSION_REWRITE_ERROR;
    }
    if (!s->checkpoint_valid) {
        snprintf(err, errlen, "session has no valid checkpoint");
        return PULSAR_SESSION_REWRITE_ERROR;
    }
    if (common < 0 || common > s->checkpoint.len || common > prompt->len) {
        snprintf(err, errlen, "invalid rewrite prefix");
        return PULSAR_SESSION_REWRITE_ERROR;
    }
    for (int i = 0; i < common; i++) {
        if (s->checkpoint.v[i] != prompt->v[i]) {
            snprintf(err, errlen, "rewrite prefix does not match live checkpoint");
            return PULSAR_SESSION_REWRITE_ERROR;
        }
    }

    if (common == s->checkpoint.len) {
        return s->sync(prompt, NULL, 0, err, errlen) == 0 ?
            PULSAR_SESSION_REWRITE_OK : PULSAR_SESSION_REWRITE_ERROR;
    }

    if (pulsar_session_rewrite_requires_rebuild(s->checkpoint.len, prompt->len, common)) {
        snprintf(err, errlen, "rewrite needs rebuild: common=%d live=%d canonical=%d",
                 common, s->checkpoint.len, prompt->len);
        return PULSAR_SESSION_REWRITE_REBUILD_NEEDED;
    }

    snprintf(err, errlen, "unexpected canonical rewrite state");
    return PULSAR_SESSION_REWRITE_ERROR;
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
    return sample_argmax(s->logits, PULSAR_N_VOCAB);
}


int pulsar_session::argmax_excluding(int excluded_id) {
    auto *s = this;
    if (!s || !s->logits) return -1;
    /* THE row-max rule (sample_argmax): first finite value seeds, lowest id
     * wins a tie, -1 when nothing is finite. */
    int best = -1;
    float best_logit = 0.0f;
    for (uint32_t i = 0; i < PULSAR_N_VOCAB; i++) {
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
    return sample_top_p_min_p(s->logits, PULSAR_N_VOCAB, temperature, top_k, top_p,
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
    return pulsar_logits_top_logprobs(s->logits, (int)PULSAR_N_VOCAB, out, k);
}


int pulsar_session::token_logprob(int token, pulsar_token_score *out) {
    auto *s = this;
    if (!s) return 0;
    return pulsar_logits_token_logprob(s->logits, (int)PULSAR_N_VOCAB, token, out);
}


int pulsar_session::copy_logits(float *out, int cap) {
    auto *s = this;
    if (!s || !out || cap < (int)PULSAR_N_VOCAB) return 0;
    memcpy(out, s->logits, (size_t)PULSAR_N_VOCAB * sizeof(out[0]));
    return (int)PULSAR_N_VOCAB;
}


int pulsar_session::set_logits(const float *logits, int n) {
    auto *s = this;
    if (!s || !logits || n != (int)PULSAR_N_VOCAB) return 1;
    memcpy(s->logits, logits, (size_t)PULSAR_N_VOCAB * sizeof(s->logits[0]));
    return 0;
}


int pulsar_session::eval(int token, char *err, size_t errlen) {
    auto *s = this;
    if (!s) return 1;
    /* Fail loud rather than corrupt: after a multiseq step the graph's scalar
     * frontier counters hold a cross-bank superset, so this decode would emit
     * its compressor row at the superset index and attend over another bank's
     * rows — wrong logits, silently.  pulsar_session_sync re-establishes per-bank
     * state (rebuild path) and clears the flag. */
    if (s->mseq_dirty) {
        snprintf(err, errlen,
                 "session eval after a multiseq decode step: this session's "
                 "per-bank state needs re-establishing; re-sync the session "
                 "first");
        return 1;
    }
    /* L226: same shape for a salvaged rewind -- the KV above the salvage floor
     * was dropped, so decoding from here would continue a truncated history.
     * A sync re-prefills above the floor and clears this. */
    if (s->kv_salvaged) {
        snprintf(err, errlen,
                 "session eval after a salvaged rewind: the compressor could not follow the rollback, "
                 "so the KV is only valid to the last prefill frontier; re-sync the session first");
        return 1;
    }
    pulsar_engine *e = s->engine;
    /* L188: a refused sample is -1 (PULSAR_SAMPLE_REFUSED); the embed kernel
     * would clamp it to token 0 and the step would look like a good one.  The
     * check lives HERE, once, for every caller that feeds a sampled token back
     * (server lanes, CLI, agent, eval): a non-id fails the request. */
    if (token < 0 || token >= pulsar_engine_vocab_size(e)) {
        snprintf(err, errlen, "eval: token %d is not a vocab id (a refused sample must fail the "
                              "request, not be evaluated)", token);
        return 1;
    }
    /* Steady-state decode must reuse preallocated scratch, never touch the host
     * heap. The guard is a no-op unless PULSAR_ALLOC_GUARD is set, so this is
     * free in production; armed, it makes any xmalloc/xrealloc inside the decode
     * fatal. Only the eval is guarded -- token_vec_push below legitimately grows
     * the checkpoint. */
    /* ONE LANE.  This used to call gpu_graph_eval_token_raw_swa -- a whole
     * parallel single-token graph encoder -- while the server decoded through
     * gpu_graph_decode_multiseq_batch.  Two lanes meant every tool built on the
     * classic API (pulsar-bench, pulsar-eval, pulsar-cli, several gates) measured
     * code production never executes, which is how a dead fusion survived and
     * how five instruments in a row measured nothing (L129).
     *
     * A 1-row batch on this session's own bank is the same work: bank 0 maps to
     * the classic tensors when no pool is allocated (gpu_graph_bank_raw_pool
     * falls back to layer_raw_cache, gpu_graph_bank_pool_count reports 1), so
     * this costs no extra slab and no extra memory.
     *
     * The classic flags stay untouched deliberately, and that is SOUND rather
     * than convenient: decode_multiseq must invalidate because its scalar
     * frontier counters end up holding a cross-bank superset. A superset over
     * exactly one touched bank IS that bank's truth -- and the mseq_dirty guard
     * above already establishes that the counters were this bank's truth on
     * entry. Both conditions are required; neither alone is enough. Nothing
     * here weakens pulsar_session_decode_mixed's contract for its own callers. */
    int     ms_tok[1]  = { token };
    int32_t ms_pos[1]  = { (int32_t)s->checkpoint.len };
    int32_t ms_bank[1] = { (int32_t)(s->graph.banks.n_banks ? s->graph.banks.cur_bank : 0u) };
    /* rc: 0 = recoverable pre-arm reject, 1 = success, else fatal mid-sweep. */
    const int ms_rc = gpu_graph_decode_multiseq_batch(&s->graph, &e->model, &e->weights,
                                                      ms_tok, ms_pos, ms_bank, 1u,
                                                      s->logits, NULL, 0u,
                                                      /*capture_cur=*/true);
    const bool decode_ok = (ms_rc == 1);
    if (!decode_ok) {
        snprintf(err, errlen, "%s decode failed", pulsar_backend_name(e->backend));
        s->checkpoint_valid = false;
        return 1;
    }
    token_vec_push(&s->checkpoint, token);
    /* a token evaluated outside the speculative path (tool injection, plain
     * fallback loops) advances the state past any in-flight carry */
    s->spec.spec_carry_valid = false;
    return 0;
}


void pulsar_session::note_committed_tokens(const int *toks, int n) {
    auto *s = this;
    if (!s || !toks || n <= 0) return;
    for (int i = 0; i < n; i++) token_vec_push(&s->checkpoint, toks[i]);
}


void pulsar_session::invalidate() {
    auto *s = this;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    pulsar_spec_drop_pendings(&s->spec);
    s->spec.spec_carry_valid = false;
    spec_quench_reset(s);
    /* L218: a new conversation starts from the empty compressor group; the
     * verify-save span described the dead one's positions. */
    {
        const uint32_t b = gpu_graph_cur_bank(&s->graph);
        s->graph.ms_spec_save_rows[b] = 0u;
        s->graph.ms_comp_state_stale[b] = false;
        /* plan-33 inc C: a partial cut's boundary-row stash describes the replay
         * that followed it.  A new conversation's emits start over row 0, so the
         * hook must not byte-restore the dead conversation's row. */
        s->graph.ms_emit_keep[b] = 0u;
    }
    /* The drafter's context-KV ring must not survive into a new prompt: it was
     * never reset before, so in the server every request after the first
     * attended over the PREVIOUS request's window rows for its first ~128
     * generated tokens (and the drafter is near-useless without a valid
     * window: masked-window eval 4.7% vs 86% top-1). Positions are
     * drafter-relative, so restarting at 0 is exact. */
    for (int i = 0; i < 3; i++) s->graph.dspark_n_raw[i] = 0;
    s->graph.dspark_prompt_n = 0;
    s->prefill_frontier = 0;   /* L195: the history is gone */
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
    /* The length BEFORE this rewind: the compressor-state rebuild needs it to
     * tell a rewind that stays inside the group a coff-1 lane is already filling
     * (its committed slots are still valid and must be KEPT) from one that
     * crosses a group boundary (those slots belong to a newer group and the
     * group's rows must be rebuilt).  See gpu_graph_compressor_state_rewind. */
    const uint32_t prev_len = (uint32_t)s->checkpoint.len;
    /* The image blocks above the new end are gone from the live KV: their
     * identity must go with them or a later request could reuse rows that no
     * longer exist (L226). */
    if (pos < s->live_image_barrier) {
        s->live_image_fp = 0;
        s->live_image_barrier = 0;
    }
    s->checkpoint.len = pos;
    pulsar_spec_drop_pendings(&s->spec);
    s->spec.spec_carry_valid = false;
    spec_quench_reset(s);
    /* Rewound positions' drafter rows are stale; empty the window (it refills
     * from the prompt capture on the next prefill, or from commits). */
    for (int i = 0; i < 3; i++) s->graph.dspark_n_raw[i] = 0;
    s->graph.dspark_prompt_n = 0;
    if (s->prefill_frontier > pos) s->prefill_frontier = pos;   /* L195: a prefill above the new frontier never happened */
    /* The compressed frontier of every kv source follows the position law on
     * the bank this session's checkpoint describes (other banks hold other
     * slots' positions).  Clamp DOWN only -- a counter can only be AHEAD of a
     * rewound position, and a lagging one (mid-admission prefill) must never
     * be raised here.  Rows beyond the clamp are invisible (readers cap at
     * n_comp) and are rewritten by the next emit at that index.  L120's
     * production signature ("frontier not position-true") was exactly this
     * clamp missing on the per-bank row. */
    const uint32_t rw_bank = gpu_graph_cur_bank(&s->graph);
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (!gpu_graph_layer_is_kv_source(il)) continue;
        const uint32_t want = (uint32_t)pos / pulsar_layer_compress_ratio(il);
        if (gpu_graph_n_comp(&s->graph, rw_bank, il) > want) gpu_graph_n_comp(&s->graph, rw_bank, il) = want;
    }
    /* Value half (L218): the ratio-2 sources' pending group.  At an even
     * position it is the empty group; inside a group it is rebuilt from the
     * last verify round's saved projections (the two callers that land here
     * -- the spec trim and the server's ghost rewind -- stay inside that
     * round), else the bank is marked stale and refuses a mid-group store. */
    if (!gpu_graph_compressor_state_rewind(&s->graph, rw_bank, (uint32_t)pos, prev_len)) {
        /* L226: the compressor produces its rows on PREFILL, so a rewind into the
         * GENERATED region -- above the last prefill frontier -- has no rows to
         * rebuild from and cannot be re-established.  Invalidating the whole
         * checkpoint there costs a rebuild of the entire conversation; measured on
         * a served agentic session, EVERY tool round did exactly that (34 s of
         * prefill for a ~700-token span).  Salvage instead: roll back to the last
         * prefill frontier, rounded to the resume grid so the re-prefill that
         * follows is the engine's own byte-identical resume, keep the tokens below
         * it (they are correct), and let the next sync re-prefill above it.  Only
         * if the state cannot be established THERE either does the checkpoint get
         * invalidated. */
        /* Candidates, nearest first: the prefill frontier itself (the state there
         * is a prefill's own), then one grid step below it, and so on.  A step is
         * worth trying because the ring only reaches PULSAR_REWIND_RING_DEPTH
         * back and a bank that was just forked or spilled starts with an empty
         * one, so the frontier itself sometimes misses while a step below it --
         * which the ring still covers, or which is a group boundary -- rebuilds.
         * Four steps is the useful range; beyond that the ring cannot help and
         * the rebuild we are avoiding is what is left. */
        const int pf = s->prefill_frontier;
        const int base = (pf > 0 && pf < pos) ? pf : pos - (int)PULSAR_RESUME_GRID;
        bool salvaged = false;
        for (int step = 0; step < 4 && !salvaged; step++) {
            const int cand = base - step * (int)PULSAR_RESUME_GRID;
            if (cand <= 0) break;
            const uint32_t floor = (uint32_t)((cand / (int)PULSAR_RESUME_GRID) * (int)PULSAR_RESUME_GRID);
            if (floor == 0u || floor >= (uint32_t)pos) continue;
            if (!gpu_graph_compressor_state_rewind(&s->graph, rw_bank, floor, prev_len)) continue;
            s->checkpoint.len = (int)floor;
            s->prefill_frontier = (int)floor;
            for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
                if (!gpu_graph_layer_is_kv_source(il)) continue;
                const uint32_t want = floor / pulsar_layer_compress_ratio(il);
                if (gpu_graph_n_comp(&s->graph, rw_bank, il) > want) gpu_graph_n_comp(&s->graph, rw_bank, il) = want;
            }
            if (floor < (uint32_t)s->live_image_barrier) {
                s->live_image_fp = 0;
                s->live_image_barrier = 0;
            }
            s->kv_salvaged = true;
            salvaged = true;
            fprintf(stderr, "pulsar: rewind to %u: compressor state not re-establishable (generated region) -- "
                            "salvaged to %u (%d grid step%s below the prefill frontier); the next sync "
                            "re-prefills above it\n",
                    (unsigned)pos, (unsigned)floor, step,
                    step == 1 ? "" : "s");
        }
        if (!salvaged) {
            fprintf(stderr, "pulsar: rewind to %u: compressor state could not be re-established -- checkpoint invalidated\n",
                    (unsigned)pos);
            s->checkpoint_valid = false;
        }
    }
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


/* Multi-session serving: is interrupting pulsar_session_sync() at a chunk
 * boundary (cancel callback) and re-issuing the sync bit-identical to letting
 * it run to completion?
 *
 * Two conditions must hold, and the return value encodes both:
 *
 *   - gpu_graph_prefill_chunked_range caps resumed (start != 0) chunks at
 *     raw_cap. If this session's cold chunks are larger (prefill_cap >
 *     raw_cap), a resumed prefill would re-chunk on different boundaries,
 *     changing batch shapes and therefore cuBLASLt algo selection; exact
 *     replay is lost. Return 0: the caller must not interrupt at all.
 *
 *   - There USED to be a third condition: below a crossover, sync extended the
 *     checkpoint by single-token decode evals instead of a batched chunk, so
 *     interrupting with less than that left would change which path evaluated
 *     the tail. L131 deleted the single-token encoder, so the tail is always a
 *     batched chunk and the hazard cannot arise. Any positive suffix resumes
 *     exactly; the minimum is simply 1. */
uint32_t pulsar_session::prefill_quantum_min_suffix() const {
    auto *s = this;
    if (!s) return 0;
    if (s->graph.prefill_cap > s->graph.raw_cap) return 0;
    /* A cold (start==0) chunk loop trims each non-final chunk end DOWN to the
     * compress-ratio LCM, while a resumed (start!=0) loop snaps to absolute
     * prefill_cap boundaries. The two produce the same chunk ends only when
     * prefill_cap itself is LCM-aligned (true for the 4096/8192 defaults; a
     * hand-set --prefill-chunk may not be). */
    uint32_t align = 1;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t r = pulsar_layer_compress_ratio(il);
        if (r > 1 && align % r != 0) {
            uint32_t a = align, b = r;
            while (b) { const uint32_t t = a % b; a = b; b = t; }
            align *= r / a;
        }
    }
    if (align > 1 && s->graph.prefill_cap % align != 0) return 0;
    return 1u;
}


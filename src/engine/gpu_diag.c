#include "ds4_engine_internal.h"



float rms_abs_diff(const float *a, const float *b, uint64_t n) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)b[i];
        ss += d * d;
    }
    return n ? (float)sqrt(ss / (double)n) : 0.0f;
}



uint64_t argmax_f32(const float *x, uint64_t n) {
    uint64_t best = 0;
    for (uint64_t i = 1; i < n; i++) {
        if (x[i] > x[best]) best = i;
    }
    return best;
}




void print_vec_stats(const char *name, const float *x, uint64_t n) {
    float minv = DS4_POS_INF;
    float maxv = DS4_NEG_INF;
    double ss = 0.0;

    for (uint64_t i = 0; i < n; i++) {
        const float v = x[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        ss += (double)v * v;
    }

    printf("%s: min=%g max=%g rms=%g\n",
        name, minv, maxv, sqrt(ss / (double)n));
}



void gpu_graph_debug_dump_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f32,
        uint32_t          il,
        uint32_t          pos) {
    if (!t || n_f32 == 0 || !gpu_graph_debug_wants(name, il, pos)) return;
    const char *prefix = getenv("DS4_CUDA_GRAPH_DUMP_PREFIX");

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    float *buf = xmalloc((size_t)n_f32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_f32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, buf, n_f32)) {
            fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume GPU command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}



void gpu_graph_debug_dump_f16_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f16,
        uint32_t          il,
        uint32_t          pos) {
    if (!t || n_f16 == 0 || !gpu_graph_debug_wants(name, il, pos)) return;
    const char *prefix = getenv("DS4_CUDA_GRAPH_DUMP_PREFIX");

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    uint16_t *hbuf = xmalloc((size_t)n_f16 * sizeof(hbuf[0]));
    float *fbuf = xmalloc((size_t)n_f16 * sizeof(fbuf[0]));
    if (ds4_gpu_tensor_read(t, 0, hbuf, n_f16 * sizeof(hbuf[0])) != 0) {
        for (uint64_t i = 0; i < n_f16; i++) fbuf[i] = f16_to_f32(hbuf[i]);
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, fbuf, n_f16)) {
            fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(fbuf);
    free(hbuf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume GPU command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}



void gpu_graph_debug_dump_i32_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_i32,
        uint32_t          il,
        uint32_t          pos) {
    if (!t || n_i32 == 0 || !gpu_graph_debug_wants(name, il, pos)) return;
    const char *prefix = getenv("DS4_CUDA_GRAPH_DUMP_PREFIX");

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    int32_t *buf = xmalloc((size_t)n_i32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_i32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.i32", prefix, name, il, pos);
        FILE *fp = fopen(path, "wb");
        if (fp) {
            if (fwrite(buf, sizeof(buf[0]), (size_t)n_i32, fp) == (size_t)n_i32) {
                fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
            }
            fclose(fp);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume GPU command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}



bool gpu_graph_needs_ffn_out(const ds4_gpu_graph *g, uint32_t il, uint32_t pos) {
    return gpu_graph_directional_steering_ffn_enabled(g) ||
           g->materialize_ffn_out ||
           gpu_graph_debug_wants("ffn_out", il, pos);
}



bool gpu_graph_ensure_ffn_out(ds4_gpu_graph *g) {
    if (!g->ffn_out) {
        g->ffn_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    }
    return g->ffn_out != NULL;
}



bool gpu_graph_ensure_batch_ffn_out(ds4_gpu_graph *g) {
    if (!g->batch_ffn_out) {
        g->batch_ffn_out = ds4_gpu_tensor_alloc((uint64_t)g->prefill_cap * DS4_N_EMBD * sizeof(float));
    }
    return g->batch_ffn_out != NULL;
}



/* =========================================================================
 * GPU Release Graph Allocation.
 * ========================================================================= */

/* Derived capacities and tensor dimensions for one session's GPU graph.
 * Computed by gpu_graph_compute_dims and shared by the allocator
 * (gpu_graph_alloc_raw_cap) and the sizing estimate (gpu_graph_session_bytes)
 * so admission control and the allocator can never disagree about the derived
 * quantities.  The per-buffer byte expressions still appear in both functions;
 * the server's estimate-vs-actual reconciliation (>10% drift warning) is the
 * enforcement that they stay in sync. */
typedef struct {
    uint32_t raw_cap;
    uint32_t raw_window;
    uint32_t ctx_size;
    uint32_t prefill_cap;
    uint32_t comp_cap;
    uint32_t attn_comp_stage_cap;          /* only meaningful under DS4_ATTN_PACK */
    uint32_t layer_comp_cap[DS4_MAX_LAYER];
    uint64_t hc_dim;
    uint64_t mix_hc;
    uint64_t q_rank;
    uint64_t q_dim;
    uint64_t low_dim;
    uint64_t shared_dim;
    uint64_t routed_mid_dim;
    uint64_t vocab_dim;
    uint64_t comp_width_max;
    uint64_t indexer_q_dim;
} gpu_graph_dims;

static void gpu_graph_compute_dims(
        gpu_graph_dims *d,
        const ds4_weights       *weights,
        const ds4_layer_weights *layer,
        uint32_t                 raw_cap,
        uint32_t                 ctx_size,
        uint32_t                 prefill_cap) {
    memset(d, 0, sizeof(*d));
    if (raw_cap == 0) raw_cap = 1;
    if (ctx_size == 0) ctx_size = raw_cap;
    if (prefill_cap == 0) prefill_cap = 1;
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > ctx_size) raw_window = ctx_size;
    if (raw_window == 0) raw_window = 1;
    if (raw_cap < raw_window) raw_cap = raw_window;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    d->raw_cap = raw_cap;
    d->raw_window = raw_window;
    d->ctx_size = ctx_size;
    d->prefill_cap = prefill_cap;

    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    d->comp_cap = ctx_size / min_ratio + 2u;
    if (d->comp_cap < 2u) d->comp_cap = 2u;
    d->attn_comp_stage_cap = prefill_cap / min_ratio + 2u;
    if (d->attn_comp_stage_cap < 2u) d->attn_comp_stage_cap = 2u;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) {
            d->layer_comp_cap[il] = 0;
        } else {
            d->layer_comp_cap[il] = ctx_size / ratio + 2u;
            if (d->layer_comp_cap[il] < 2u) d->layer_comp_cap[il] = 2u;
        }
    }

    d->hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    d->mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    d->q_rank = layer->attn_q_a->dim[1];
    d->q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    d->low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    d->shared_dim = layer->ffn_gate_shexp->dim[1];
    d->routed_mid_dim = layer->ffn_gate_exps->dim[1];
    d->vocab_dim = weights->output ? weights->output->dim[1] : DS4_N_VOCAB;
    d->comp_width_max = 2ull * (DS4_N_HEAD_DIM > DS4_N_INDEXER_HEAD_DIM
        ? DS4_N_HEAD_DIM
        : DS4_N_INDEXER_HEAD_DIM);
    d->indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
}



/* TRUE total per-session GPU byte cost: the sum of every ds4_gpu_tensor that
 * gpu_graph_alloc_raw_cap (and, when enable_spec, gpu_graph_init_dspark_target)
 * allocates for one session.  This is what admission control must price a
 * session at — the packed KV estimate (ds4_context_memory_estimate_packed)
 * covers only the persistent KV rows plus the indexer score/mask scratch and
 * undercounts the real cost by roughly an order of magnitude (2026-07-13
 * incident: three ctx=65536 slots admitted at 0.5 GiB each consumed the whole
 * GB10 and hard-locked the machine).
 *
 * KEEP IN SYNC with the allocators below (gpu_graph_alloc_raw_cap,
 * gpu_graph_bank_slabs_alloc, gpu_graph_init_dspark_target — same order,
 * same expressions).
 * The server reconciles this estimate against the measured allocation delta
 * after every session create and logs a loud warning on >10% drift, so a
 * missed buffer surfaces on the first live run instead of as an
 * under-admission OOM.
 *
 * EXCLUSION LIST — intentionally unaccounted per-session allocations (each
 * negligible, absorbed by DS4_SERVER_MEM_FLOOR_BYTES; do not re-derive):
 *   - the lazy gpu_graph_ensure_ffn_out/gpu_graph_ensure_batch_ffn_out
 *     buffers (only allocated under steering/diagnostics);
 *   - directional-steering dirs (gpu_graph_load_directional_steering,
 *     ~DS4_N_LAYER*DS4_N_EMBD floats — these ARE inside the measured create
 *     delta, so they show up in reconciliation but never in this estimate);
 *   - the spec snapshot/restore descriptor tables that
 *     ds4_gpu_batched_copy_prepare lazily cudaMallocs on the first fused
 *     spec step (~KBs; raw cudaMalloc, outside both this estimate and the
 *     ds4_gpu_tensor byte counter that reconciliation measures). */
uint64_t gpu_graph_session_bytes(
        const ds4_weights       *weights,
        const ds4_layer_weights *layer,
        uint32_t                 raw_cap,
        uint32_t                 ctx_size,
        uint32_t                 prefill_cap,
        bool                     enable_spec) {
    return gpu_graph_session_bytes_banked(weights, layer, raw_cap, ctx_size,
                                          prefill_cap, enable_spec,
                                          gpu_graph_bank_pool_n());
}

uint64_t gpu_graph_session_bytes_banked(
        const ds4_weights       *weights,
        const ds4_layer_weights *layer,
        uint32_t                 raw_cap,
        uint32_t                 ctx_size,
        uint32_t                 prefill_cap,
        bool                     enable_spec,
        uint32_t                 n_banks_in) {
    gpu_graph_dims dz;
    gpu_graph_compute_dims(&dz, weights, layer, raw_cap, ctx_size, prefill_cap);
    const uint64_t pc = dz.prefill_cap;
    const uint64_t f32 = sizeof(float);

    /* Persistent KV caches (raw ring, packed attn comp, indexer comp) plus the
     * indexer_scores/comp_mask working pair — shared with the managed-vs-device
     * KV placement policy the allocator itself uses.  In bank-pool mode the
     * persistent KV slabs (and the per-bank compressor state lanes below)
     * scale with the pool size; everything else is shared by all banks. */
    const uint64_t n_banks = n_banks_in < 1u ? 1u : n_banks_in;
    uint64_t kv_cache_bytes = 0;
    uint64_t total = gpu_graph_context_bytes_for_kv_policy(
            dz.ctx_size, dz.raw_cap, dz.prefill_cap, &kv_cache_bytes);
    if (n_banks > 1u) total += (n_banks - 1u) * kv_cache_bytes;

    /* Per-layer attention/indexer state — one lane per bank — and the
     * single-lane spec shadows (spec never runs against a shared pool). */
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_state = (uint64_t)coff * DS4_N_HEAD_DIM *
                                    coff * ratio * f32;
        total += n_banks * 2ull * attn_state;             /* layer_attn_state_kv/score */
        if (enable_spec) total += 2ull * attn_state;      /* spec_attn_state_kv/score */
        if (ratio == 4) {
            const uint64_t index_state = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM *
                                         coff * ratio * f32;
            total += n_banks * 2ull * index_state;        /* layer_index_state_kv/score */
            if (enable_spec) total += 2ull * index_state; /* spec_index_state_kv/score */
        }
    }

    /* Single-token graph buffers. */
    total += 2ull * dz.hc_dim * f32;                      /* cur_hc, flat_hc */
    total += 2ull * dz.mix_hc * f32;                      /* hc_mix, hc_split (views free) */
    total += 2ull * DS4_N_EMBD * f32;                     /* attn_cur, attn_norm */
    total += 2ull * dz.q_rank * f32;                      /* qr, qr_norm */
    total += dz.q_dim * f32;                              /* q */
    total += 2ull * DS4_N_HEAD_DIM * f32;                 /* kv_raw, kv */
    total += 2ull * dz.comp_width_max * f32;              /* comp_kv_cur, comp_sc_cur */
    if (gpu_graph_attn_pack_enabled()) {
        total += (uint64_t)dz.attn_comp_stage_cap * DS4_N_HEAD_DIM * f32; /* attn_comp_stage */
        total += (uint64_t)dz.comp_cap * DS4_N_HEAD_DIM * f32;            /* attn_comp_dequant */
    }
    if (gpu_graph_idx_fp4_enabled()) {
        total += (uint64_t)dz.comp_cap * DS4_N_INDEXER_HEAD_DIM * f32;    /* idx_comp_stage */
    }
    total += dz.indexer_q_dim * f32;                      /* indexer_q */
    total += (uint64_t)DS4_N_INDEXER_HEAD * f32;          /* indexer_weights */
    total += (uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u) *
             pc * sizeof(uint32_t);                       /* comp_selected */
    total += dz.q_dim * f32;                              /* heads */
    total += dz.low_dim * f32;                            /* attn_low */
    total += (uint64_t)DS4_N_EMBD * f32;                  /* attn_out */
    total += dz.hc_dim * f32;                             /* after_attn_hc */
    total += 2ull * DS4_N_EMBD * f32;                     /* ffn_cur, ffn_norm */
    total += 3ull * dz.shared_dim * f32;                  /* shared_gate/up/mid */
    total += (uint64_t)DS4_N_EMBD * f32;                  /* shared_out */
    total += 2ull * DS4_N_EXPERT * f32;                   /* router_logits, router_probs */
    total += (uint64_t)DS4_N_EXPERT_USED * (sizeof(int) + f32); /* router_selected/weights */
    total += 3ull * DS4_N_EXPERT_USED * dz.routed_mid_dim * f32; /* routed_gate/up/mid */
    total += (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * f32;     /* routed_down */
    total += (uint64_t)DS4_N_EMBD * f32;                  /* routed_out */
    total += dz.hc_dim * f32;                             /* after_ffn_hc */
    total += 2ull * DS4_N_HC * f32;                       /* output_pre, output_weights */
    total += 2ull * DS4_N_EMBD * f32;                     /* output_embd, output_norm */
    total += dz.vocab_dim * f32;                          /* logits */
    total += pc * sizeof(int32_t);                        /* prefill_tokens */
    /* Multi-row logits readback slab.  Allocated unconditionally (NOT under
     * enable_spec): besides the DSpark draft/verify passes it is the output
     * buffer of every batched multi-row head — gpu_graph_verify_suffix_tops
     * and the Tier-2 batched multi-session decode driver
     * (gpu_graph_decode_multiseq_batch) both read their rows out of it, and
     * those paths must work with speculation disabled (--no-dspark,
     * ds4-bench/ds4-eval/agent, or any model without a merged drafter). */
    total += 16ull * DS4_N_VOCAB * f32;                   /* spec_logits */

    /* Batch (prefill working set) buffers — the pc-scaled bulk that dominates
     * the non-KV cost (~4 GiB at pc=4096 on Flash). */
    total += 3ull * pc * dz.hc_dim * f32;                 /* batch_cur/next/flat_hc */
    total += 2ull * pc * dz.mix_hc * f32;                 /* batch_hc_mix/split */
    total += 2ull * pc * DS4_N_EMBD * f32;                /* batch_attn_cur/norm */
    total += 2ull * pc * dz.q_rank * f32;                 /* batch_qr/qr_norm */
    total += pc * dz.q_dim * f32;                         /* batch_q */
    total += 2ull * pc * DS4_N_HEAD_DIM * f32;            /* batch_kv_raw/kv */
    total += 2ull * pc * dz.comp_width_max * f32;         /* batch_comp_kv/sc */
    total += pc * dz.indexer_q_dim * f32;                 /* batch_indexer_q */
    total += pc * DS4_N_INDEXER_HEAD * f32;               /* batch_indexer_weights */
    total += pc * dz.q_dim * f32;                         /* batch_heads */
    total += pc * dz.low_dim * f32;                       /* batch_attn_low */
    total += pc * DS4_N_EMBD * f32;                       /* batch_attn_out */
    total += pc * dz.hc_dim * f32;                        /* batch_after_attn_hc */
    total += 2ull * pc * DS4_N_EMBD * f32;                /* batch_ffn_cur/norm */
    total += 3ull * pc * dz.shared_dim * f32;             /* batch_shared_gate/up/mid */
    total += pc * DS4_N_EMBD * f32;                       /* batch_shared_out */
    total += 2ull * pc * DS4_N_EXPERT * f32;              /* batch_router_logits/probs */
    total += pc * DS4_N_EXPERT_USED * (sizeof(int) + f32); /* batch_router_selected/weights */
    total += 3ull * pc * DS4_N_EXPERT_USED * dz.routed_mid_dim * f32; /* batch_routed_gate/up/mid */
    total += pc * DS4_N_EXPERT_USED * DS4_N_EMBD * f32;   /* batch_routed_down */
    total += pc * DS4_N_EMBD * f32;                       /* batch_routed_out */

    /* DSpark drafter graph state (gpu_graph_init_dspark_target), allocated by
     * ds4_session_create whenever the engine has a drafter loaded — the
     * production merged GGUF auto-enables it. */
    if (enable_spec) {
        total += 3ull * DS4_N_EMBD * f32;                 /* dspark_target_h[3] */
        total += 3ull * 17 * DS4_N_EMBD * f32;            /* dspark_target_h_batch[3] */
        /* Option F: dspark_raw_cache[3] + dspark_prompt_h[3] are BANKED slabs
         * (n_banks lanes) so the N=2 spec-time-slice lane keeps a warm ring per
         * bank; the rest of the drafter state is shared across banks. */
        total += (uint64_t)n_banks * 3ull * DS4_DSPARK_DRAFT_WINDOW * DS4_N_HEAD_DIM * f32; /* dspark_raw_cache[3] */
        total += (uint64_t)DS4_N_EMBD * f32;              /* dspark_main_x */
        total += 3ull * pc * DS4_N_EMBD * f32;            /* dspark_bulk_h[3] */
        total += (uint64_t)n_banks * 3ull * DS4_DSPARK_DRAFT_WINDOW * DS4_N_EMBD * f32; /* dspark_prompt_h[3] */
        const uint64_t attn_w = 2ull * DS4_N_HEAD_DIM;
        const uint64_t idx_w = 2ull * DS4_N_INDEXER_HEAD_DIM;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio == 0) continue;
            total += 2ull * 17 * attn_w * f32;            /* spec_comp_kv/sc_save */
            if (ratio == 4) total += 2ull * 17 * idx_w * f32; /* spec_icomp_kv/sc_save */
        }
        total += (uint64_t)DS4_N_HEAD_DIM * f32;          /* spec_comp_scratch_row */
        total += 3ull * DS4_N_EMBD * f32;                 /* dspark_concat */
        total += (uint64_t)DS4_N_EMBD * f32;              /* dspark_proj_out */
        total += 3ull * DS4_N_HEAD_DIM * f32;             /* dspark_seed_kv/norm/rot */
        total += (uint64_t)DS4_N_VOCAB * f32;             /* dspark_markov_logits */
    }
    return total;
}



/* DS4_KV_MANAGED: measurement override for the managed-vs-device placement of
 * the PERSISTENT KV caches (raw ring + attn/index comp caches), for the
 * multi-session UVM over-provisioning design (plan addendum 2026-07-14 —
 * first-touch fault cost must be measured before committing to
 * always-managed slot KV). Unset/empty → the size-based policy
 * (ds4_gpu_should_use_managed_kv_cache) decides, as before. "1"/"on" →
 * force cudaMallocManaged; "0"/"off"/"false" → force device cudaMalloc.
 * Read once per process (static cache) and consulted only at graph
 * allocation (session create) — never on a token/layer path. */
static int gpu_graph_kv_managed_override(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *v = getenv("DS4_KV_MANAGED");
        if (!v || !v[0]) {
            cached = -1;
        } else if (strcmp(v, "0") == 0 || strcasecmp(v, "off") == 0 ||
                   strcasecmp(v, "false") == 0) {
            cached = 0;
        } else if (strcmp(v, "1") == 0 || strcasecmp(v, "on") == 0 ||
                   strcasecmp(v, "true") == 0) {
            cached = 1;
        } else {
            /* Unrecognized values fall back to the size policy instead of
             * silently force-flipping a measurement flag (a typo'd "of[f]"
             * must not force-manage the KV of a 128 GB box). */
            fprintf(stderr,
                    "ds4: DS4_KV_MANAGED=\"%s\" not recognized "
                    "(want 1/on/true or 0/off/false); using size policy\n", v);
            cached = -1;
        }
    }
    return cached;
}



/* DS4_MSEQ_BANKS: Tier-2 bank-pool size for the next graph allocation.
 * 1 (default) keeps the classic single-session cache layout; 2..DS4_MSEQ_MAX
 * allocates the fixed per-bank slabs (ds4_bank_slabs) and installs bank-0
 * views, so every existing single-session path runs unmodified against
 * bank 0.  Read once per process at graph allocation — never on a
 * token/layer path.  Interim wiring: later increments make the server pass
 * the pool size explicitly instead. */
uint32_t gpu_graph_bank_pool_n(void) {
    static long cached = -1;
    if (cached < 0) {
        const char *v = getenv("DS4_MSEQ_BANKS");
        long n = 1;
        if (v && v[0]) {
            char *end = NULL;
            n = strtol(v, &end, 10);
            /* Tolerate trailing whitespace (e.g. "2\n" from a shell here-doc),
             * like the DS4_CUDA_DECODE_INDEXER_SPARSE_THRESHOLD parser. */
            while (end && isspace((unsigned char)*end)) end++;
            if (end == v || (end && *end != '\0') || n < 1) {
                fprintf(stderr,
                        "ds4: DS4_MSEQ_BANKS=\"%s\" not recognized (want 1..%u); "
                        "bank pool disabled\n", v, DS4_MSEQ_MAX);
                n = 1;
            } else if (n > (long)DS4_MSEQ_MAX) {
                fprintf(stderr, "ds4: DS4_MSEQ_BANKS=%ld clamped to %u\n",
                        n, DS4_MSEQ_MAX);
                n = (long)DS4_MSEQ_MAX;
            }
        }
        cached = n;
    }
    return (uint32_t)cached;
}



/* Allocate the fixed per-bank KV slabs (layout: ds4_bank_slabs in
 * ds4_engine_internal.h; design adapted from Entrpi/ds4 v0.2 — citation at
 * the struct).  Per-bank byte expressions MUST match the single-session
 * branches in gpu_graph_alloc_raw_cap below and the pricing in
 * gpu_graph_session_bytes.  Compressor state lanes are primed for every bank
 * here (kv = 0, score = -inf), exactly like a fresh single-session graph, so
 * a later bank admit starts from the same state a cold session would. */
static bool gpu_graph_bank_slabs_alloc(
        ds4_gpu_graph      *g,
        uint32_t              n_banks,
        bool                  managed_kv_cache,
        const gpu_graph_dims *dz) {
    ds4_bank_slabs *b = &g->banks;
    const uint64_t raw_elem = gpu_graph_raw_f16_enabled() ? sizeof(uint16_t)
                                                          : sizeof(float);
    b->n_banks = n_banks;
    b->cur_bank = 0;
    b->raw_bank_bytes = (uint64_t)dz->raw_cap * DS4_N_HEAD_DIM * raw_elem;
    bool ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        /* uint32 row-ABI audit: batched kernels address rows as
         * seq_id * cap + local in uint32; reject the geometry up front. */
        if ((uint64_t)n_banks * dz->raw_cap > 4294967296ull ||
            (uint64_t)n_banks * dz->layer_comp_cap[il] > 4294967296ull) {
            fprintf(stderr,
                    "ds4: bank pool geometry overflows the uint32 row ABI "
                    "(%u banks x raw_cap %u / comp_cap %u)\n",
                    n_banks, dz->raw_cap, dz->layer_comp_cap[il]);
            return false;
        }
        b->raw[il] = gpu_graph_alloc_kv_cache_tensor(
                managed_kv_cache, (uint64_t)n_banks * b->raw_bank_bytes);
        ok = b->raw[il] != NULL;
        if (!ok || ratio == 0) continue;

        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
        const uint64_t attn_rows = (uint64_t)coff * ratio;
        const uint64_t attn_lane = attn_width * attn_rows * sizeof(float);
        b->comp_bank_bytes[il] = (uint64_t)dz->layer_comp_cap[il] *
                                 gpu_graph_attn_comp_cache_row_bytes();
        b->astate_bank_bytes[il] = attn_lane;
        /* ctx-scaled comp slabs are always managed in pool mode: unified-
         * memory demand paging keeps short banks from paying worst-case
         * residency (the VMM adaptation, see ds4_bank_slabs). */
        b->comp[il] = ds4_gpu_tensor_alloc_managed(
                (uint64_t)n_banks * b->comp_bank_bytes[il]);
        b->askv[il] = ds4_gpu_tensor_alloc((uint64_t)n_banks * attn_lane);
        b->assc[il] = ds4_gpu_tensor_alloc((uint64_t)n_banks * attn_lane);
        ok = b->comp[il] && b->askv[il] && b->assc[il] &&
             gpu_tensor_fill_f32(b->askv[il], 0.0f,
                                 (uint64_t)n_banks * attn_width * attn_rows) &&
             gpu_tensor_fill_f32(b->assc[il], DS4_NEG_INF,
                                 (uint64_t)n_banks * attn_width * attn_rows);
        if (ok && ratio == 4) {
            const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
            const uint64_t index_rows = (uint64_t)coff * ratio;
            const uint64_t index_lane = index_width * index_rows * sizeof(float);
            const uint64_t index_row_bytes = gpu_graph_idx_fp4_enabled()
                ? DS4_ENGINE_IDXFP4_ROWBYTES
                : (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            b->index_bank_bytes[il] = (uint64_t)dz->layer_comp_cap[il] *
                                      index_row_bytes;
            b->istate_bank_bytes[il] = index_lane;
            b->index[il] = ds4_gpu_tensor_alloc_managed(
                    (uint64_t)n_banks * b->index_bank_bytes[il]);
            b->iskv[il] = ds4_gpu_tensor_alloc((uint64_t)n_banks * index_lane);
            b->issc[il] = ds4_gpu_tensor_alloc((uint64_t)n_banks * index_lane);
            ok = b->index[il] && b->iskv[il] && b->issc[il] &&
                 gpu_tensor_fill_f32(b->iskv[il], 0.0f,
                                     (uint64_t)n_banks * index_width * index_rows) &&
                 gpu_tensor_fill_f32(b->issc[il], DS4_NEG_INF,
                                     (uint64_t)n_banks * index_width * index_rows);
        }
    }
    /* Option B-full increment 1: build the per-bank raw-ring descriptor tables
     * (see ds4_bank_slabs).  Uniform allocation this increment, so the base is
     * a prefix-sum of EQUAL caps (b * raw_cap) and every cap entry is raw_cap —
     * the degenerate case the batched kernels reduce to Option A on.  The comp
     * tables stay NULL (comp is still uniform-scalar addressed).  Built once
     * here because the pool geometry is fixed for the graph's lifetime. */
    if (ok) {
        uint32_t base_h[DS4_MSEQ_MAX];
        uint32_t cap_h[DS4_MSEQ_MAX];
        for (uint32_t bk = 0; bk < n_banks && bk < DS4_MSEQ_MAX; bk++) {
            base_h[bk] = bk * dz->raw_cap;   /* n_banks*raw_cap < 2^32 audited above */
            cap_h[bk]  = dz->raw_cap;
        }
        b->raw_base_b = ds4_gpu_tensor_alloc((uint64_t)n_banks * sizeof(uint32_t));
        b->raw_cap_b  = ds4_gpu_tensor_alloc((uint64_t)n_banks * sizeof(uint32_t));
        ok = b->raw_base_b && b->raw_cap_b &&
             ds4_gpu_tensor_write(b->raw_base_b, 0, base_h,
                                  (uint64_t)n_banks * sizeof(uint32_t)) &&
             ds4_gpu_tensor_write(b->raw_cap_b, 0, cap_h,
                                  (uint64_t)n_banks * sizeof(uint32_t));
    }
    return ok;
}



/* Re-install the graph's per-layer cache views onto `bank`.  Pure host-side
 * pointer surgery (view wrappers are freed/recreated; no device work), so the
 * caller must guarantee the device is idle with respect to the previous
 * bank's views.  The spec frontier copy tables bake raw device pointers of
 * the state views, so they are dropped for lazy rebuild. */
bool gpu_graph_bank_repoint(ds4_gpu_graph *g, uint32_t bank) {
    if (!g || g->banks.n_banks == 0 || bank >= g->banks.n_banks) return false;
    ds4_bank_slabs *b = &g->banks;
    if (bank == b->cur_bank) return true;
    bool ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        ds4_gpu_tensor_free(g->layer_raw_cache[il]);
        g->layer_raw_cache[il] = ds4_gpu_tensor_view(
                b->raw[il], (uint64_t)bank * b->raw_bank_bytes, b->raw_bank_bytes);
        ok = g->layer_raw_cache[il] != NULL;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (!ok || ratio == 0) continue;
        ds4_gpu_tensor_free(g->layer_attn_comp_cache[il]);
        ds4_gpu_tensor_free(g->layer_attn_state_kv[il]);
        ds4_gpu_tensor_free(g->layer_attn_state_score[il]);
        g->layer_attn_comp_cache[il] = ds4_gpu_tensor_view(
                b->comp[il], (uint64_t)bank * b->comp_bank_bytes[il],
                b->comp_bank_bytes[il]);
        g->layer_attn_state_kv[il] = ds4_gpu_tensor_view(
                b->askv[il], (uint64_t)bank * b->astate_bank_bytes[il],
                b->astate_bank_bytes[il]);
        g->layer_attn_state_score[il] = ds4_gpu_tensor_view(
                b->assc[il], (uint64_t)bank * b->astate_bank_bytes[il],
                b->astate_bank_bytes[il]);
        ok = g->layer_attn_comp_cache[il] && g->layer_attn_state_kv[il] &&
             g->layer_attn_state_score[il];
        if (ok && ratio == 4) {
            ds4_gpu_tensor_free(g->layer_index_comp_cache[il]);
            ds4_gpu_tensor_free(g->layer_index_state_kv[il]);
            ds4_gpu_tensor_free(g->layer_index_state_score[il]);
            g->layer_index_comp_cache[il] = ds4_gpu_tensor_view(
                    b->index[il], (uint64_t)bank * b->index_bank_bytes[il],
                    b->index_bank_bytes[il]);
            g->layer_index_state_kv[il] = ds4_gpu_tensor_view(
                    b->iskv[il], (uint64_t)bank * b->istate_bank_bytes[il],
                    b->istate_bank_bytes[il]);
            g->layer_index_state_score[il] = ds4_gpu_tensor_view(
                    b->issc[il], (uint64_t)bank * b->istate_bank_bytes[il],
                    b->istate_bank_bytes[il]);
            ok = g->layer_index_comp_cache[il] && g->layer_index_state_kv[il] &&
                 g->layer_index_state_score[il];
        }
    }
    /* Option F: swap the per-bank DSpark drafter ring views (present only when
     * the drafter is loaded and the ring was banked).  Same host-only pointer
     * surgery as the KV views above, so the spec path transparently reads the
     * active bank's warm window. */
    if (ok && b->dspark_raw[0]) {
        for (int i = 0; i < 3 && ok; i++) {
            ds4_gpu_tensor_free(g->dspark_raw_cache[i]);
            ds4_gpu_tensor_free(g->dspark_prompt_h[i]);
            g->dspark_raw_cache[i] = ds4_gpu_tensor_view(
                    b->dspark_raw[i], (uint64_t)bank * b->dspark_raw_bank_bytes,
                    b->dspark_raw_bank_bytes);
            g->dspark_prompt_h[i] = ds4_gpu_tensor_view(
                    b->dspark_prompt[i], (uint64_t)bank * b->dspark_prompt_bank_bytes,
                    b->dspark_prompt_bank_bytes);
            ok = g->dspark_raw_cache[i] && g->dspark_prompt_h[i];
        }
    }
    /* Stale pointer hygiene (mirrors gpu_graph_free). */
    ds4_gpu_batched_copy_free(g->spec_snap_copies);
    ds4_gpu_batched_copy_free(g->spec_restore_copies);
    g->spec_snap_copies = NULL;
    g->spec_restore_copies = NULL;
    g->spec_frontier_copy_n = 0;
    g->spec_frontier_copy_init = 0;
    if (ok) b->cur_bank = bank;
    return ok;
}



/* ===== Tier-2 banked multiseq step machinery (increment 2) ==============
 *
 * Pool/view accessors + per-bank frontier bookkeeping + the multiseq step
 * arm/disarm pair.  Contracts at the declarations (ds4_engine_internal.h)
 * and at the ms_* fields in ds4_gpu_graph. */

uint32_t gpu_graph_bank_pool_count(const ds4_gpu_graph *g) {
    return g->banks.n_banks ? g->banks.n_banks : 1u;
}

ds4_gpu_tensor *gpu_graph_bank_raw_pool(ds4_gpu_graph *g, uint32_t il) {
    if (!g || il >= DS4_N_LAYER) return NULL;
    return g->banks.n_banks ? g->banks.raw[il] : g->layer_raw_cache[il];
}

ds4_gpu_tensor *gpu_graph_bank_attn_comp_pool(ds4_gpu_graph *g, uint32_t il) {
    if (!g || il >= DS4_N_LAYER) return NULL;
    return g->banks.n_banks ? g->banks.comp[il] : g->layer_attn_comp_cache[il];
}

ds4_gpu_tensor *gpu_graph_bank_index_comp_pool(ds4_gpu_graph *g, uint32_t il) {
    if (!g || il >= DS4_N_LAYER) return NULL;
    return g->banks.n_banks ? g->banks.index[il] : g->layer_index_comp_cache[il];
}

/* One accessor body for all six per-(bank,layer) view kinds: a fresh view of
 * the bank's lane in the slab, or (pool disabled) a fresh view wrapping the
 * whole classic tensor — bank 0 only, so a stale bank id cannot silently
 * alias the single session's state. */
static ds4_gpu_tensor *bank_lane_view(ds4_gpu_graph *g,
                                      ds4_gpu_tensor *slab,
                                      const uint64_t *lane_bytes,
                                      ds4_gpu_tensor *classic,
                                      uint32_t il, uint32_t bank) {
    if (!g || il >= DS4_N_LAYER) return NULL;
    if (g->banks.n_banks == 0) {
        if (bank != 0 || !classic) return NULL;
        return ds4_gpu_tensor_view(classic, 0, ds4_gpu_tensor_bytes(classic));
    }
    if (bank >= g->banks.n_banks || !slab) return NULL;
    return ds4_gpu_tensor_view(slab, (uint64_t)bank * lane_bytes[il], lane_bytes[il]);
}

ds4_gpu_tensor *gpu_graph_bank_attn_comp_view(ds4_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_lane_view(g, g->banks.comp[il], g->banks.comp_bank_bytes,
                          g->layer_attn_comp_cache[il], il, bank);
}

ds4_gpu_tensor *gpu_graph_bank_index_comp_view(ds4_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_lane_view(g, g->banks.index[il], g->banks.index_bank_bytes,
                          g->layer_index_comp_cache[il], il, bank);
}

ds4_gpu_tensor *gpu_graph_bank_attn_state_kv_view(ds4_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_lane_view(g, g->banks.askv[il], g->banks.astate_bank_bytes,
                          g->layer_attn_state_kv[il], il, bank);
}

ds4_gpu_tensor *gpu_graph_bank_attn_state_score_view(ds4_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_lane_view(g, g->banks.assc[il], g->banks.astate_bank_bytes,
                          g->layer_attn_state_score[il], il, bank);
}

ds4_gpu_tensor *gpu_graph_bank_index_state_kv_view(ds4_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_lane_view(g, g->banks.iskv[il], g->banks.istate_bank_bytes,
                          g->layer_index_state_kv[il], il, bank);
}

ds4_gpu_tensor *gpu_graph_bank_index_state_score_view(ds4_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_lane_view(g, g->banks.issc[il], g->banks.istate_bank_bytes,
                          g->layer_index_state_score[il], il, bank);
}

void gpu_graph_bank_counters_capture(ds4_gpu_graph *g, uint32_t bank) {
    if (!g || bank >= DS4_MSEQ_MAX) return;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->ms_n_comp[bank][il] = g->layer_n_comp[il];
        g->ms_n_index_comp[bank][il] = g->layer_n_index_comp[il];
    }
    /* Option F: the drafter-ring frontier is per-bank too (device rings live in
     * banks.dspark_*), so it rides the same capture/install hand-off. */
    for (int i = 0; i < 3; i++) g->ms_dspark_n_raw[bank][i] = g->dspark_n_raw[i];
    g->ms_dspark_prompt_n[bank] = g->dspark_prompt_n;
    g->ms_dspark_prompt_lo[bank] = g->dspark_prompt_lo;
}

void gpu_graph_bank_counters_install(ds4_gpu_graph *g, uint32_t bank) {
    if (!g || bank >= DS4_MSEQ_MAX) return;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_n_comp[il] = g->ms_n_comp[bank][il];
        g->layer_n_index_comp[il] = g->ms_n_index_comp[bank][il];
    }
    for (int i = 0; i < 3; i++) g->dspark_n_raw[i] = g->ms_dspark_n_raw[bank][i];
    g->dspark_prompt_n = g->ms_dspark_prompt_n[bank];
    g->dspark_prompt_lo = g->ms_dspark_prompt_lo[bank];
}

bool gpu_graph_multiseq_step_begin(ds4_gpu_graph *g, const int32_t *pos,
                                   const int32_t *seq, uint32_t n_rows,
                                   bool capture_cur) {
    if (!g || !pos || !seq || n_rows == 0 || n_rows > g->prefill_cap) {
        fprintf(stderr, "ds4: multiseq step rejected: bad args (n_rows=%u)\n",
                n_rows);
        return false;
    }
    if (g->batch_multiseq) {
        fprintf(stderr, "ds4: multiseq step rejected: step already armed\n");
        return false;
    }
    const uint32_t n_banks = gpu_graph_bank_pool_count(g);
    /* Constraint (relaxed for the batched-decode driver): contiguous per-bank
     * runs (each bank at most one run), positions consecutive WITHIN each
     * bank's run, and every run starting at a position > 0.  Banks may sit at
     * unrelated positions — the multi-session shape; the upstream batch
     * stages (RoPE q/kv/indexer-q/inverse, compressor loop, raw scatter,
     * attention, indexer) are all per-row-position driven.  Position-0 rows
     * stay rejected: admission (from-zero) prefill runs on the classic
     * single-bank view path in v1, and negative positions would wrap the
     * uint32 casts below. */
    bool bank_seen[DS4_MSEQ_MAX] = {false};
    int32_t prev_bank = -1;
    for (uint32_t t = 0; t < n_rows; t++) {
        if (seq[t] < 0 || (uint32_t)seq[t] >= n_banks ||
            (uint32_t)seq[t] >= DS4_MSEQ_MAX) {
            fprintf(stderr, "ds4: multiseq step rejected: row %u bank %d "
                            "out of pool (n_banks=%u)\n", t, seq[t], n_banks);
            return false;
        }
        if (seq[t] != prev_bank) {
            if (bank_seen[seq[t]]) {
                fprintf(stderr, "ds4: multiseq step rejected: bank %d rows "
                                "not contiguous\n", seq[t]);
                return false;
            }
            bank_seen[seq[t]] = true;
            prev_bank = seq[t];
            if (pos[t] <= 0) {
                fprintf(stderr, "ds4: multiseq step rejected: bank %d first "
                                "position %d <= 0 (admission prefill is "
                                "single-bank classic)\n", seq[t], pos[t]);
                return false;
            }
        } else if ((int64_t)pos[t] != (int64_t)pos[t - 1] + 1) {
            /* int64 arithmetic: pos[t-1] == INT32_MAX would make the int32
             * increment signed overflow (UB) before the per-row bound below
             * could reject it. */
            fprintf(stderr, "ds4: multiseq step rejected: bank %d positions "
                            "not consecutive within its run (row %u: %d, "
                            "want %lld)\n", seq[t], t, pos[t],
                    (long long)((int64_t)pos[t - 1] + 1));
            return false;
        }
        /* Bound EVERY row (not just each run's first): positions are cast to
         * uint32 downstream (ring slot, visible-comp, RoPE), and the
         * position-derived arithmetic below adds 1. */
        if (pos[t] <= 0 || pos[t] == INT32_MAX) {
            fprintf(stderr, "ds4: multiseq step rejected: bank %d row %u "
                            "position %d out of range (want 0 < pos < "
                            "INT32_MAX)\n", seq[t], t, pos[t]);
            return false;
        }
    }
    /* capture_cur: the classic scalar counters are the CURRENT bank's truth
     * (its per-bank slots may be stale — e.g. still holding a previous
     * step's values).  The capture is COMMITTED only after every rejection
     * point below has passed: a rejected begin must leave ms_n_comp[cur]
     * untouched too (the scalars can hold a previous step's cross-bank
     * superset, which would corrupt the bank's frontier record).  Until
     * then the validation loop reads the scalars directly for cur_bank. */
    const uint32_t cur_bank = g->banks.n_banks ? g->banks.cur_bank : 0u;
    /* DRIVER CONTRACT check: each batched bank's committed frontier is
     * position-true at its first row — floor(first_pos / ratio) compressed
     * rows.  A bank that lags (mid-admission-prefill) must never be
     * co-scheduled: its rows would clamp against the superset and silently
     * diverge from single-session output. */
    for (uint32_t t = 0; t < n_rows; t++) {
        if (t > 0 && seq[t] == seq[t - 1]) continue;
        const uint32_t b = (uint32_t)seq[t];
        const uint32_t p = (uint32_t)pos[t];
        const bool use_scalars = capture_cur && b == cur_bank;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint32_t have_comp = use_scalars ? g->layer_n_comp[il]
                                                   : g->ms_n_comp[b][il];
            const uint32_t have_index = use_scalars ? g->layer_n_index_comp[il]
                                                    : g->ms_n_index_comp[b][il];
            if (have_comp != p / ratio ||
                (ratio == 4 && have_index != p / ratio)) {
                fprintf(stderr,
                        "ds4: multiseq step rejected: bank %u frontier not "
                        "position-true at layer %u (pos %u ratio %u: "
                        "n_comp %u want %u, n_index_comp %u)\n",
                        b, il, p, ratio, have_comp, p / ratio, have_index);
                return false;
            }
        }
    }
    /* Lazy descriptor storage: host mirrors + device arrays, prefill_cap
     * entries (a few KB) — never allocated in single-session serving.  On a
     * (transient) device-alloc failure everything is released and reset so a
     * later step re-attempts instead of failing forever. */
    if (!g->ms_positions) {
        g->ms_positions = xmalloc((size_t)g->prefill_cap * sizeof(int32_t));
        g->ms_seq_id = xmalloc((size_t)g->prefill_cap * sizeof(int32_t));
        g->batch_positions =
            ds4_gpu_tensor_alloc((uint64_t)g->prefill_cap * sizeof(int32_t));
        g->batch_seq_id =
            ds4_gpu_tensor_alloc((uint64_t)g->prefill_cap * sizeof(int32_t));
        if (!g->batch_positions || !g->batch_seq_id) {
            fprintf(stderr, "ds4: multiseq descriptor alloc failed\n");
            ds4_gpu_tensor_free(g->batch_positions);
            ds4_gpu_tensor_free(g->batch_seq_id);
            g->batch_positions = NULL;
            g->batch_seq_id = NULL;
            free(g->ms_positions);
            free(g->ms_seq_id);
            g->ms_positions = NULL;
            g->ms_seq_id = NULL;
            return false;
        }
    }
    memcpy(g->ms_positions, pos, (size_t)n_rows * sizeof(int32_t));
    memcpy(g->ms_seq_id, seq, (size_t)n_rows * sizeof(int32_t));
    if (!ds4_gpu_tensor_write(g->batch_positions, 0, pos,
                              (uint64_t)n_rows * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(g->batch_seq_id, 0, seq,
                              (uint64_t)n_rows * sizeof(int32_t))) {
        fprintf(stderr, "ds4: multiseq descriptor upload failed\n");
        return false;
    }
    /* Superset refresh — the ONLY write of the scalar counters during a
     * multiseq step (top of step, before any launch, never mid-forward).
     * The value is the step's emit-inclusive visibility bound: max over
     * rows of (pos+1)/ratio, which every batched bank's frontier reaches
     * once its own emits land (verified by step_end).  Validate EVERY
     * layer's capacity before writing ANY scalar: a rejected begin must
     * leave the graph's classic counters untouched (a partial overwrite
     * would inflate the frontiers a recovering classic caller decodes
     * with). */
    uint32_t sup[DS4_MAX_LAYER];
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        sup[il] = 0;
        if (ratio == 0) continue;
        for (uint32_t t = 0; t < n_rows; t++) {
            const uint32_t v = ((uint32_t)pos[t] + 1u) / ratio;
            if (v > sup[il]) sup[il] = v;
        }
        if (sup[il] > g->layer_comp_cap[il]) {
            fprintf(stderr,
                    "ds4: multiseq step rejected: superset %u exceeds comp "
                    "cap %u at layer %u\n", sup[il], g->layer_comp_cap[il], il);
            return false;
        }
    }
    /* All rejection points passed: NOW commit the cur-bank capture (the
     * scalars are still the pre-step classic values here — the superset
     * write below is the first scalar mutation). */
    if (capture_cur) gpu_graph_bank_counters_capture(g, cur_bank);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        g->layer_n_comp[il] = sup[il];
        if (ratio == 4) g->layer_n_index_comp[il] = sup[il];
    }
    g->batch_multiseq_rows = n_rows;
    g->batch_multiseq = true;
    return true;
}

bool gpu_graph_multiseq_step_end(ds4_gpu_graph *g) {
    if (!g || !g->batch_multiseq) return false;
    g->batch_multiseq = false;
    const uint32_t n_rows = g->batch_multiseq_rows;
    g->batch_multiseq_rows = 0;
    /* Self-check (host ints only): every batched bank's frontier advanced to
     * exactly its position-derived value — (last_pos+1)/ratio — and the
     * scalar superset (untouched since step top) equals the max over the
     * batched banks.  A miss here is the silent-KV-corruption class; fail
     * loud so the driver aborts the session instead of serving garbage. */
    bool ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        uint32_t sup = 0;
        for (uint32_t t = 0; t < n_rows && ok; t++) {
            if (t + 1 < n_rows && g->ms_seq_id[t + 1] == g->ms_seq_id[t]) continue;
            const uint32_t b = (uint32_t)g->ms_seq_id[t];
            const uint32_t want = ((uint32_t)g->ms_positions[t] + 1u) / ratio;
            if (want > sup) sup = want;
            if (g->ms_n_comp[b][il] != want ||
                (ratio == 4 && g->ms_n_index_comp[b][il] != want)) {
                fprintf(stderr,
                        "ds4: multiseq step_end FAILED: bank %u layer %u "
                        "frontier %u/%u want %u (ratio %u)\n",
                        b, il, g->ms_n_comp[b][il],
                        g->ms_n_index_comp[b][il], want, ratio);
                ok = false;
            }
        }
        if (ok && g->layer_n_comp[il] != sup) {
            fprintf(stderr,
                    "ds4: multiseq step_end FAILED: layer %u superset %u "
                    "mutated mid-step (want %u)\n",
                    il, g->layer_n_comp[il], sup);
            ok = false;
        }
        if (ok && ratio == 4 && g->layer_n_index_comp[il] != sup) {
            fprintf(stderr,
                    "ds4: multiseq step_end FAILED: layer %u indexer superset "
                    "%u mutated mid-step (want %u)\n",
                    il, g->layer_n_index_comp[il], sup);
            ok = false;
        }
    }
    return ok;
}



/* Allocate the GPU graph state for a chosen raw-cache capacity.  The model
 * weights are not copied here; tensors reference the mapped GGUF. */
bool gpu_graph_alloc_raw_cap(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer,
        uint32_t                raw_cap,
        uint32_t                ctx_size,
        uint32_t                prefill_cap,
        bool                    enable_spec) {
    memset(g, 0, sizeof(*g));
    g->comp_ratio_override = -1;
    gpu_graph_dims dz;
    gpu_graph_compute_dims(&dz, weights, layer, raw_cap, ctx_size, prefill_cap);
    raw_cap = dz.raw_cap;
    ctx_size = dz.ctx_size;
    prefill_cap = dz.prefill_cap;
    g->raw_cap = dz.raw_cap;
    g->raw_window = dz.raw_window;
    g->prefill_cap = dz.prefill_cap;

    /* DS4_ATTN_PACK validation lives up here: no allocations have happened
     * yet, so these early returns need no cleanup. */
    if (getenv("DS4_ATTN_MX") != NULL) {
        /* Removed 2026-07-10: superseded by DS4_ATTN_PACK (bit-exact, smaller
         * rows; MX re-quantized the rope dims and cost drafter acceptance).
         * Refuse loudly instead of silently running a different format. */
        fprintf(stderr,
                "ds4: DS4_ATTN_MX has been removed (superseded by DS4_ATTN_PACK); "
                "unset DS4_ATTN_MX (use DS4_ATTN_PACK=0 for the classic f32 comp cache)\n");
        return false;
    }
    if (gpu_graph_attn_pack_enabled() &&
        (DS4_N_ROT != 64u || ((DS4_N_HEAD_DIM - DS4_N_ROT) % 64u) != 0u)) {
        fprintf(stderr,
                "ds4: DS4_ATTN_PACK requires n_rot 64 and 64-aligned nope dims "
                "(head_dim %u / n_rot %u)\n",
                (unsigned)DS4_N_HEAD_DIM, (unsigned)DS4_N_ROT);
        return false;
    }
    g->comp_cap = dz.comp_cap;
    if (gpu_graph_attn_pack_enabled()) {
        g->attn_comp_stage_cap = dz.attn_comp_stage_cap;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_comp_cap[il] = dz.layer_comp_cap[il];
    }

    const uint64_t hc_dim = dz.hc_dim;
    const uint64_t mix_hc = dz.mix_hc;
    const uint64_t q_rank = dz.q_rank;
    const uint64_t q_dim = dz.q_dim;
    const uint64_t low_dim = dz.low_dim;
    const uint64_t shared_dim = dz.shared_dim;
    const uint64_t routed_mid_dim = dz.routed_mid_dim;
    const uint64_t vocab_dim = dz.vocab_dim;
    const uint64_t comp_width_max = dz.comp_width_max;
    const uint64_t indexer_q_dim = dz.indexer_q_dim;
    const uint64_t pc = prefill_cap;
    uint64_t kv_cache_bytes = 0;
    const uint64_t context_bytes =
        gpu_graph_context_bytes_for_kv_policy(ctx_size, raw_cap, prefill_cap, &kv_cache_bytes);
    const int managed_override = gpu_graph_kv_managed_override();
    const bool managed_kv_cache = managed_override >= 0
        ? managed_override != 0
        : ds4_gpu_should_use_managed_kv_cache(kv_cache_bytes, context_bytes) != 0;
    if (managed_override >= 0) {
        fprintf(stderr,
                "ds4: DS4_KV_MANAGED override: KV caches use %s allocation "
                "(measurement flag; policy would have chosen %s)\n",
                managed_kv_cache ? "MANAGED (cudaMallocManaged)"
                                 : "DEVICE (cudaMalloc)",
                ds4_gpu_should_use_managed_kv_cache(kv_cache_bytes,
                                                    context_bytes) != 0
                    ? "managed" : "device");
    }
    if (managed_kv_cache) {
        /*
         * CUDA device allocations are fastest, but a million-token KV cache is
         * large enough to starve DGX Spark's unified CPU/GPU memory once the
         * model cache and driver allocations are present.  For this one
         * long-lived cache class, managed memory restores the old demand-paged
         * behavior.  It can be slower, but it keeps oversized contexts from
         * turning memory pressure into a machine-wide lockup.
         */
        fprintf(stderr,
                "ds4: CUDA using managed KV cache for ctx=%u "
                "(kv cache %.2f GiB, context buffers %.2f GiB); "
                "this may degrade performance but is needed for very large contexts\n",
                ctx_size,
                (double)kv_cache_bytes / 1073741824.0,
                (double)context_bytes / 1073741824.0);
    }

    /* Tier-2 bank pool: allocate the per-bank slabs first; the per-layer
     * cache pointers below then become bank-0 views instead of owning
     * allocations, and all single-session code runs unmodified. */
    const uint32_t n_banks = gpu_graph_bank_pool_n();
    if (n_banks >= 2u &&
        !gpu_graph_bank_slabs_alloc(g, n_banks, managed_kv_cache, &dz)) {
        gpu_graph_free(g);
        return false;
    }
    const bool banked = g->banks.n_banks != 0;

    g->cur_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->flat_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->hc_mix = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_split = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_pre = ds4_gpu_tensor_view(g->hc_split, 0, (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_post = ds4_gpu_tensor_view(g->hc_split,
                                       (uint64_t)DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_comb = ds4_gpu_tensor_view(g->hc_split,
                                       2ull * DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    g->attn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->attn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->qr = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->qr_norm = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->q = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->kv_raw = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    g->kv = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    bool state_init_ok = true;
    const uint64_t raw_elem_bytes = gpu_graph_raw_f16_enabled() ? sizeof(uint16_t)
                                                                : sizeof(float);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_raw_cache[il] = banked
            ? ds4_gpu_tensor_view(g->banks.raw[il], 0, g->banks.raw_bank_bytes)
            : gpu_graph_alloc_kv_cache_tensor(
                    managed_kv_cache,
                    (uint64_t)raw_cap * DS4_N_HEAD_DIM * raw_elem_bytes);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
            const uint64_t attn_rows = (uint64_t)coff * ratio;
            const uint64_t comp_row_bytes = gpu_graph_attn_comp_cache_row_bytes();
            if (banked) {
                g->layer_attn_comp_cache[il] = ds4_gpu_tensor_view(
                        g->banks.comp[il], 0, g->banks.comp_bank_bytes[il]);
                g->layer_attn_state_kv[il] = ds4_gpu_tensor_view(
                        g->banks.askv[il], 0, g->banks.astate_bank_bytes[il]);
                g->layer_attn_state_score[il] = ds4_gpu_tensor_view(
                        g->banks.assc[il], 0, g->banks.astate_bank_bytes[il]);
            } else {
                g->layer_attn_comp_cache[il] = gpu_graph_alloc_kv_cache_tensor(
                        managed_kv_cache,
                        (uint64_t)g->layer_comp_cap[il] * comp_row_bytes);
                g->layer_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->layer_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            }
            if (enable_spec) {
                g->spec_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->spec_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            }
            /* Banked mode primes every bank's state lanes at slab alloc. */
            if (!banked && g->layer_attn_state_kv[il]) {
                state_init_ok = state_init_ok &&
                                gpu_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows);
            }
            if (!banked && g->layer_attn_state_score[il]) {
                state_init_ok = state_init_ok &&
                                gpu_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows);
            }

            if (ratio == 4) {
                const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
                const uint64_t index_rows = (uint64_t)coff * ratio;
                const uint64_t index_row_bytes = gpu_graph_idx_fp4_enabled()
                    ? DS4_ENGINE_IDXFP4_ROWBYTES
                    : (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
                if (banked) {
                    g->layer_index_comp_cache[il] = ds4_gpu_tensor_view(
                            g->banks.index[il], 0, g->banks.index_bank_bytes[il]);
                    g->layer_index_state_kv[il] = ds4_gpu_tensor_view(
                            g->banks.iskv[il], 0, g->banks.istate_bank_bytes[il]);
                    g->layer_index_state_score[il] = ds4_gpu_tensor_view(
                            g->banks.issc[il], 0, g->banks.istate_bank_bytes[il]);
                } else {
                    g->layer_index_comp_cache[il] = gpu_graph_alloc_kv_cache_tensor(
                            managed_kv_cache,
                            (uint64_t)g->layer_comp_cap[il] * index_row_bytes);
                    g->layer_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->layer_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                }
                if (enable_spec) {
                    g->spec_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->spec_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                }
                if (!banked && g->layer_index_state_kv[il]) {
                    state_init_ok = state_init_ok &&
                                    gpu_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows);
                }
                if (!banked && g->layer_index_state_score[il]) {
                    state_init_ok = state_init_ok &&
                                    gpu_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows);
                }
            }
        }
    }
    g->comp_kv_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    g->comp_sc_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    if (gpu_graph_attn_pack_enabled()) {
        /* f32 staging: the compressor writes real f32 rows here, then the commit
         * step packs them to the persistent DS4_ATTN_PACK comp cache. */
        g->attn_comp_stage = ds4_gpu_tensor_alloc((uint64_t)g->attn_comp_stage_cap *
                                                  DS4_N_HEAD_DIM * sizeof(float));
        /* f32 shadow the prefill attention consumers (and session save) read
         * after the persistent packed comp cache is dequantized (see
         * gpu_graph_attn_comp_read_cache). */
        g->attn_comp_dequant = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap *
                                                    DS4_N_HEAD_DIM * sizeof(float));
    }
    if (gpu_graph_idx_fp4_enabled()) {
        if (DS4_N_INDEXER_HEAD_DIM != 128u) {
            /* The packed loader and QAT+pack kernels hard-code the 68-byte
             * head_dim-128 row; fail loud here instead of deep in a launch. */
            fprintf(stderr,
                    "ds4: DS4_IDX_FP4 requires indexer head_dim 128 (model has %u)\n",
                    DS4_N_INDEXER_HEAD_DIM);
            gpu_graph_free(g);
            return false;
        }
        /* f32 emit/repack staging for the packed indexer cache: comp-cap rows so
         * the compressor writers can keep their absolute row indices. */
        g->idx_comp_stage = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap *
                                                 DS4_N_INDEXER_HEAD_DIM * sizeof(float));
        ds4_gpu_indexer_set_fp4(1);
    }
    g->indexer_q = ds4_gpu_tensor_alloc(indexer_q_dim * sizeof(float));
    g->indexer_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_INDEXER_HEAD * sizeof(float));
    /* DS4_PREFILL_SLICE: these two are the only ctx-scaling f32 work buffers
     * with a prefill_cap token dimension; under slicing they only ever hold
     * one <=slice-token span at a time. */
    const uint64_t score_rows = (gpu_graph_prefill_slice() != 0u &&
                                 (uint64_t)gpu_graph_prefill_slice() < pc)
        ? (uint64_t)gpu_graph_prefill_slice() : pc;
    g->indexer_scores = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * score_rows * sizeof(float));
    g->comp_mask = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * score_rows * sizeof(float));
    g->comp_selected = ds4_gpu_tensor_alloc((uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u) *
                                              pc * sizeof(uint32_t));
    g->heads = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->attn_low = ds4_gpu_tensor_alloc(low_dim * sizeof(float));
    g->attn_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_attn_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->ffn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->ffn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->shared_gate = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_up = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_mid = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->router_logits = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_probs = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_selected = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(int));
    g->router_weights = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(float));
    g->routed_gate = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_up = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_mid = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_down = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->routed_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_ffn_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->output_pre = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_embd = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->output_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->logits = ds4_gpu_tensor_alloc(vocab_dim * sizeof(float));
    g->prefill_tokens = ds4_gpu_tensor_alloc(pc * sizeof(int32_t));
    /* Shared multi-row logits slab (16 rows).  Unconditional, NOT gated on
     * speculation: every batched multi-row output head writes its rows here —
     * the DSpark draft/verify passes, gpu_graph_verify_suffix_tops, and the
     * Tier-2 batched multi-session decode driver.  It used to be allocated
     * only by gpu_graph_init_dspark_target (session create, dspark_ready
     * only), which left the multiseq driver rejecting every step whenever
     * speculation was off. */
    g->spec_logits = ds4_gpu_tensor_alloc(16ull * DS4_N_VOCAB * sizeof(float));
    g->batch_cur_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_next_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_flat_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_hc_mix = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_hc_split = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_attn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_attn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_qr = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_qr_norm = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_q = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_kv_raw = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_kv = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_comp_kv = ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_comp_sc = ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_indexer_q = ds4_gpu_tensor_alloc(pc * indexer_q_dim * sizeof(float));
    g->batch_indexer_weights = ds4_gpu_tensor_alloc(pc * DS4_N_INDEXER_HEAD * sizeof(float));
    g->batch_heads = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_attn_low = ds4_gpu_tensor_alloc(pc * low_dim * sizeof(float));
    g->batch_attn_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_after_attn_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_ffn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_ffn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_shared_gate = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_up = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_mid = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_router_logits = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_probs = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_selected = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(int));
    g->batch_router_weights = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(float));
    g->batch_routed_gate = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_up = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_mid = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_down = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->batch_routed_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));

    bool layer_cache_ok = true;
    for (uint32_t il = 0; layer_cache_ok && il < DS4_N_LAYER; il++) {
        layer_cache_ok = g->layer_raw_cache[il] != NULL;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (layer_cache_ok && ratio != 0) {
            layer_cache_ok = g->layer_attn_comp_cache[il] != NULL &&
                             g->layer_attn_state_kv[il] != NULL &&
                             g->layer_attn_state_score[il] != NULL &&
                             (!enable_spec ||
                              (g->spec_attn_state_kv[il] != NULL &&
                               g->spec_attn_state_score[il] != NULL));
        }
        if (layer_cache_ok && ratio == 4) {
            layer_cache_ok = g->layer_index_comp_cache[il] != NULL &&
                             (!gpu_graph_idx_fp4_enabled() || g->idx_comp_stage != NULL) &&
                             g->layer_index_state_kv[il] != NULL &&
                             g->layer_index_state_score[il] != NULL &&
                             (!enable_spec ||
                              (g->spec_index_state_kv[il] != NULL &&
                               g->spec_index_state_score[il] != NULL));
        }
    }

    const bool ok = state_init_ok && layer_cache_ok &&
                    g->cur_hc && g->flat_hc && g->hc_mix && g->hc_split &&
                    g->hc_pre && g->hc_post && g->hc_comb &&
                    g->attn_cur && g->attn_norm && g->qr && g->qr_norm &&
                    g->q && g->kv_raw && g->kv &&
                    g->comp_kv_cur && g->comp_sc_cur &&
                    (!gpu_graph_attn_pack_enabled() ||
                     (g->attn_comp_stage && g->attn_comp_dequant)) &&
                    g->indexer_q && g->indexer_weights && g->indexer_scores &&
                    g->comp_mask && g->comp_selected &&
                    g->heads && g->attn_low && g->attn_out &&
                    g->after_attn_hc && g->ffn_cur && g->ffn_norm &&
                    g->shared_gate && g->shared_up && g->shared_mid &&
                    g->shared_out &&
                    g->router_logits && g->router_probs && g->router_selected && g->router_weights &&
                    g->routed_gate && g->routed_up && g->routed_mid &&
                    g->routed_down && g->routed_out &&
                    g->after_ffn_hc &&
                    g->output_pre && g->output_weights && g->output_embd &&
                    g->output_norm && g->logits &&
                    g->prefill_tokens && g->spec_logits &&
                    g->batch_cur_hc && g->batch_next_hc && g->batch_flat_hc &&
                    g->batch_hc_mix && g->batch_hc_split &&
                    g->batch_attn_cur && g->batch_attn_norm &&
                    g->batch_qr && g->batch_qr_norm && g->batch_q &&
                    g->batch_kv_raw && g->batch_kv &&
                    g->batch_comp_kv && g->batch_comp_sc &&
                    g->batch_indexer_q && g->batch_indexer_weights &&
                    g->batch_heads && g->batch_attn_low && g->batch_attn_out &&
                    g->batch_after_attn_hc &&
                    g->batch_ffn_cur && g->batch_ffn_norm &&
                    g->batch_shared_gate && g->batch_shared_up &&
                    g->batch_shared_mid && g->batch_shared_out &&
                    g->batch_router_logits && g->batch_router_probs &&
                    g->batch_router_selected && g->batch_router_weights &&
                    g->batch_routed_gate && g->batch_routed_up &&
                    g->batch_routed_mid && g->batch_routed_down &&
                    g->batch_routed_out;
    if (!ok) gpu_graph_free(g);
    return ok;
}

bool gpu_graph_init_dspark_target(ds4_gpu_graph *g, const uint32_t target_layer_ids[3]) {
    bool ok = true;
    for (int i = 0; i < 3; i++) {
        g->dspark_target_layer_ids[i] = target_layer_ids[i];
        g->dspark_target_h[i] = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        /* Fused-loop batch capture: one anchor hidden per verify-batch position
         * (17 = the spec block clamp of 16 drafts + first_token). */
        g->dspark_target_h_batch[i] = ds4_gpu_tensor_alloc(
            (uint64_t)17 * DS4_N_EMBD * sizeof(float));
        /* Option F: bank the drafter ring when the pool is enabled — one
         * bank-major slab, dspark_raw_cache[i] becomes bank 0's view (repoint
         * swaps it). */
        if (g->banks.n_banks != 0) {
            g->banks.dspark_raw_bank_bytes =
                (uint64_t)DS4_DSPARK_DRAFT_WINDOW * DS4_N_HEAD_DIM * sizeof(float);
            g->banks.dspark_raw[i] = ds4_gpu_tensor_alloc(
                (uint64_t)g->banks.n_banks * g->banks.dspark_raw_bank_bytes);
            g->dspark_raw_cache[i] = g->banks.dspark_raw[i]
                ? ds4_gpu_tensor_view(g->banks.dspark_raw[i], 0,
                                      g->banks.dspark_raw_bank_bytes)
                : NULL;
        } else {
            g->dspark_raw_cache[i] = ds4_gpu_tensor_alloc(
                (uint64_t)DS4_DSPARK_DRAFT_WINDOW * DS4_N_HEAD_DIM * sizeof(float));
        }
        g->dspark_n_raw[i] = 0;
        ok = ok && g->dspark_target_h[i] && g->dspark_target_h_batch[i] &&
             g->dspark_raw_cache[i];
    }
    g->dspark_capture_batch_n = 0;
    g->dspark_main_x = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    ok = ok && g->dspark_main_x;
    /* Bulk prefill capture buffers: always allocated when the drafter is
     * loaded (~200 MB at prefill_cap 4096) -- the prompt-window seeding
     * reduces every chunk through them; DS4_DSPARK_PREFILL_DUMP additionally
     * streams them to disk for retraining data. */
    g->dspark_bulk_n = 0;
    for (int i = 0; i < 3; i++) {
        g->dspark_bulk_h[i] = ds4_gpu_tensor_alloc(
            (uint64_t)g->prefill_cap * DS4_N_EMBD * sizeof(float));
        ok = ok && g->dspark_bulk_h[i];
    }
    /* Prompt-window ring: last <=128 prompt positions' anchor hiddens. */
    g->dspark_prompt_n = 0;
    for (int i = 0; i < 3; i++) {
        if (g->banks.n_banks != 0) {
            g->banks.dspark_prompt_bank_bytes =
                (uint64_t)DS4_DSPARK_DRAFT_WINDOW * DS4_N_EMBD * sizeof(float);
            g->banks.dspark_prompt[i] = ds4_gpu_tensor_alloc(
                (uint64_t)g->banks.n_banks * g->banks.dspark_prompt_bank_bytes);
            g->dspark_prompt_h[i] = g->banks.dspark_prompt[i]
                ? ds4_gpu_tensor_view(g->banks.dspark_prompt[i], 0,
                                      g->banks.dspark_prompt_bank_bytes)
                : NULL;
        } else {
            g->dspark_prompt_h[i] = ds4_gpu_tensor_alloc(
                (uint64_t)DS4_DSPARK_DRAFT_WINDOW * DS4_N_EMBD * sizeof(float));
        }
        ok = ok && g->dspark_prompt_h[i];
    }
    /* Stage-B no-replay rollback: per-position compressor projection saves for
     * every compressed layer (attn comp_width <= 2*DS4_N_HEAD_DIM; indexer width
     * = 2*DS4_N_INDEXER_HEAD_DIM) + one emit-sink scratch row. ~8 MB total. */
    {
        const uint64_t attn_w = 2ull * DS4_N_HEAD_DIM;
        const uint64_t idx_w = 2ull * DS4_N_INDEXER_HEAD_DIM;
        for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            g->spec_comp_kv_save[il] = NULL;
            g->spec_comp_sc_save[il] = NULL;
            g->spec_icomp_kv_save[il] = NULL;
            g->spec_icomp_sc_save[il] = NULL;
            if (ratio == 0) continue;
            g->spec_comp_kv_save[il] = ds4_gpu_tensor_alloc(17ull * attn_w * sizeof(float));
            g->spec_comp_sc_save[il] = ds4_gpu_tensor_alloc(17ull * attn_w * sizeof(float));
            ok = ok && g->spec_comp_kv_save[il] && g->spec_comp_sc_save[il];
            if (ratio == 4) {
                g->spec_icomp_kv_save[il] = ds4_gpu_tensor_alloc(17ull * idx_w * sizeof(float));
                g->spec_icomp_sc_save[il] = ds4_gpu_tensor_alloc(17ull * idx_w * sizeof(float));
                ok = ok && g->spec_icomp_kv_save[il] && g->spec_icomp_sc_save[il];
            }
        }
        g->spec_comp_scratch_row = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        ok = ok && g->spec_comp_scratch_row;
        g->spec_comp_save_n = 0;
        g->dspark_concat = ds4_gpu_tensor_alloc(3ull * DS4_N_EMBD * sizeof(float));
        g->dspark_proj_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->dspark_seed_kv = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        g->dspark_seed_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        g->dspark_seed_rot = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        g->dspark_markov_logits = ds4_gpu_tensor_alloc((uint64_t)DS4_N_VOCAB * sizeof(float));
        ok = ok && g->dspark_concat && g->dspark_proj_out && g->dspark_seed_kv &&
             g->dspark_seed_norm && g->dspark_seed_rot && g->dspark_markov_logits;
    }
    /* spec_logits is NOT allocated here: it is the shared multi-row logits
     * slab, allocated unconditionally by gpu_graph_alloc_raw_cap (the batched
     * decode driver needs it with speculation disabled). */
    return ok;
}


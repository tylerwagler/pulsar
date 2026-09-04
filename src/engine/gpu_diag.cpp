#include "pulsar_engine_internal.h"
#include <map>
#include <string>






uint64_t argmax_f32(const float *x, uint64_t n) {
    uint64_t best = 0;
    for (uint64_t i = 1; i < n; i++) {
        if (x[i] > x[best]) best = i;
    }
    return best;
}







/* ---- f16-viability range sweep (diagnostic) -----------------------------
 *
 * PULSAR_CUDA_RANGE_SWEEP=1 turns every debug-dump point into a range probe
 * instead of a file dump, so one prefill answers "can this f32 staging buffer
 * be f16?" for every named tensor at every layer.  Env is read once. */
namespace {
struct range_acc {
    double amax = 0.0, amin = 0.0;
    double n_over = 0, n_sub = 0, n_inf = 0, n_nan = 0;
    unsigned long long calls = 0;
    unsigned long long n_elem = 0;   /* elements per call (the buffer's live extent) */
    double int8_vs_e4m3 = -1.0;      /* relative L2, our int8 acts vs the source's E4M3 */
};
std::map<std::string, range_acc> g_range;

bool range_sweep_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("PULSAR_CUDA_RANGE_SWEEP"); on = (e && e[0] == '1') ? 1 : 0; }
    return on != 0;
}

void range_sweep_report(void) {
    fprintf(stderr, "\n=== f16 RANGE SWEEP (f16 max 65504, min normal 6.1035e-05) ===\n");
    fprintf(stderr, "%-26s %12s %12s %10s %12s %10s %8s %6s\n",
            "tensor", "max|v|", "min|v|!=0", "n>65504", "n<6.1e-5", "n_inf", "n_NaN", "calls");
    fprintf(stderr, "%-26s %12s %10s\n", "", "MiB/call", "elems");
    for (const auto &kv : g_range) {
        const range_acc &a = kv.second;
        fprintf(stderr, "%-26s %12.4g %12.4g %10.0f %12.0f %10.0f %8.0f %6llu%s\n",
                kv.first.c_str(), a.amax, a.amin, a.n_over, a.n_sub, a.n_inf, a.n_nan, a.calls,
                (a.n_over > 0 || a.n_nan > 0) ? "   <-- F16 UNSAFE" : "");
        fprintf(stderr, "%-26s %12.2f %10llu   int8-vs-E4M3 relL2 %7.3f%%\n", "",
                (double)a.n_elem * 4.0 / (1024.0 * 1024.0), a.n_elem,
                a.int8_vs_e4m3 >= 0.0 ? 100.0 * a.int8_vs_e4m3 : -1.0);
    }
    fprintf(stderr, "=== END RANGE SWEEP ===\n");
}
}  /* namespace */

/* "Will anything OBSERVE these f32 bytes?" -- the question every f32
 * store-skip actually has to ask, which is NOT the same question as "should
 * this tensor be written to a dump file".
 *
 * There are TWO observers, and only one of them is the file dump:
 * gpu_graph_debug_dump_tensor's FIRST branch (below) is the range sweep, and
 * it reads the tensor and returns BEFORE gpu_graph_debug_wants is ever
 * consulted.  So a skip gated on debug_wants alone hands the sweep dead bytes
 * -- including its int8_vs_e4m3 column, which is the metric narrowing
 * decisions are made from, and which has already been wrong once (1836c39).
 *
 * Every store-skip gates on THESE, never on the dump predicates directly.
 * Named for the invariant rather than for either observer, so adding a third
 * observer is one edit here instead of a hunt through the skip sites. */
bool gpu_graph_f32_store_observed(const char *name, uint32_t il, uint32_t pos) {
    return range_sweep_on() || gpu_graph_debug_wants(name, il, pos);
}

bool gpu_graph_f32_store_observed_any(void) {
    return range_sweep_on() || gpu_graph_debug_dump_enabled();
}


void gpu_graph_debug_dump_tensor(
        const char       *name,
        pulsar_gpu_tensor *t,
        uint64_t          n_f32,
        uint32_t          il,
        uint32_t          pos) {
    if (range_sweep_on()) {
        if (!t || n_f32 == 0) return;
        double s5[6] = {0,0,0,0,0,0};
        if (!pulsar_gpu_tensor_range_stats(t, n_f32, s5)) return;
        static bool hooked = false;
        if (!hooked) { atexit(range_sweep_report); hooked = true; }
        range_acc &a = g_range[name];
        a.n_elem = n_f32;
        const double dv = pulsar_gpu_tensor_int8_vs_e4m3(t, n_f32);
        if (dv >= 0.0 && dv > a.int8_vs_e4m3) a.int8_vs_e4m3 = dv;
        if (s5[0] > a.amax) a.amax = s5[0];
        if (s5[1] > 0.0 && (a.amin == 0.0 || s5[1] < a.amin)) a.amin = s5[1];
        a.n_over += s5[2]; a.n_sub += s5[3]; a.n_inf += s5[4]; a.n_nan += s5[5];
        a.calls++;
        return;
    }
    if (!t || n_f32 == 0 || !gpu_graph_debug_wants(name, il, pos)) return;
    const char *prefix = getenv("PULSAR_CUDA_GRAPH_DUMP_PREFIX");

    if (pulsar_gpu_synchronize() == 0) {
        fprintf(stderr, "pulsar: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    float *buf = (float *)xmalloc((size_t)n_f32 * sizeof(buf[0]));
    /* Reads ELEMENTS, not bytes, and widens whatever they are stored as.  This
     * was a raw byte read that assumed f32; heads (L033) is the first narrowed
     * tensor on this path, and the failure mode was a dump full of plausible
     * garbage rather than an error. */
    if (pulsar_gpu_tensor_read_f32(t, 0, buf, n_f32) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, buf, n_f32)) {
            fprintf(stderr, "pulsar: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(buf);

    if (pulsar_gpu_begin_commands() == 0) {
        fprintf(stderr, "pulsar: failed to resume GPU command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}


/* Same dump for an HC residual CARRIER (BF16 storage; task #62). The plain f32
 * dump above would read n*sizeof(float) from a buffer holding n*PULSAR_HC_ELT_SIZE
 * bytes — a 2x out-of-bounds read — so carriers MUST come through here. */
void gpu_graph_debug_dump_hc_tensor(
        const char       *name,
        pulsar_gpu_tensor *t,
        uint64_t          n_elems,
        uint32_t          il,
        uint32_t          pos) {
    if (!t || n_elems == 0 || !gpu_graph_debug_wants(name, il, pos)) return;
    const char *prefix = getenv("PULSAR_CUDA_GRAPH_DUMP_PREFIX");

    if (pulsar_gpu_synchronize() == 0) {
        fprintf(stderr, "pulsar: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    float *buf = (float *)xmalloc((size_t)n_elems * sizeof(buf[0]));
    if (pulsar_read_hc_carrier_f32(t, 0, buf, n_elems) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, buf, n_elems)) {
            fprintf(stderr, "pulsar: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(buf);

    if (pulsar_gpu_begin_commands() == 0) {
        fprintf(stderr, "pulsar: failed to resume GPU command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

/* Same dump for the stored Q buffer (f16 since L045).  As with the HC
 * carrier, the plain f32 dump above would read n*sizeof(float) from a
 * buffer holding n*PULSAR_Q_ELT_SIZE -- a 2x out-of-bounds read -- so Q
 * MUST come through here.
 *
 * Q is also deliberately out of the range-sweep census that the f32 dump
 * runs: that census reads f32 and exists to decide what to narrow, which
 * for Q is now an answered question. */
void gpu_graph_debug_dump_q_tensor(
        const char       *name,
        pulsar_gpu_tensor *t,
        uint64_t          n_elems,
        uint32_t          il,
        uint32_t          pos) {
    if (!t || n_elems == 0 || !gpu_graph_debug_wants(name, il, pos)) return;
    const char *prefix = getenv("PULSAR_CUDA_GRAPH_DUMP_PREFIX");

    if (pulsar_gpu_synchronize() == 0) {
        fprintf(stderr, "pulsar: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    float *buf = (float *)xmalloc((size_t)n_elems * sizeof(buf[0]));
    if (pulsar_read_q_f32(t, 0, buf, n_elems) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, buf, n_elems)) {
            fprintf(stderr, "pulsar: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(buf);

    if (pulsar_gpu_begin_commands() == 0) {
        fprintf(stderr, "pulsar: failed to resume GPU command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}






void gpu_graph_debug_dump_i32_tensor(
        const char       *name,
        pulsar_gpu_tensor *t,
        uint64_t          n_i32,
        uint32_t          il,
        uint32_t          pos) {
    if (!t || n_i32 == 0 || !gpu_graph_debug_wants(name, il, pos)) return;
    const char *prefix = getenv("PULSAR_CUDA_GRAPH_DUMP_PREFIX");

    if (pulsar_gpu_synchronize() == 0) {
        fprintf(stderr, "pulsar: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    int32_t *buf = (int32_t *)xmalloc((size_t)n_i32 * sizeof(buf[0]));
    if (pulsar_gpu_tensor_read(t, 0, buf, n_i32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.i32", prefix, name, il, pos);
        FILE *fp = fopen(path, "wb");
        if (fp) {
            if (fwrite(buf, sizeof(buf[0]), (size_t)n_i32, fp) == (size_t)n_i32) {
                fprintf(stderr, "pulsar: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
            }
            fclose(fp);
        }
    }
    free(buf);

    if (pulsar_gpu_begin_commands() == 0) {
        fprintf(stderr, "pulsar: failed to resume GPU command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}



bool gpu_graph_needs_ffn_out(const pulsar_gpu_graph *g, uint32_t il, uint32_t pos) {
    return gpu_graph_directional_steering_ffn_enabled(g) ||
           gpu_graph_f32_store_observed("ffn_out", il, pos);
}



bool gpu_graph_ensure_batch_ffn_out(pulsar_gpu_graph *g) {
    if (!g->batch_ffn_out) {
        g->batch_ffn_out = pulsar_gpu_tensor_alloc((uint64_t)g->prefill_cap * PULSAR_N_EMBD * sizeof(float));
    }
    return g->batch_ffn_out != NULL;
}



/* =========================================================================
 * GPU Release Graph Allocation.
 * ========================================================================= */

/** Derived capacities and tensor dimensions for one session's GPU graph.
 * Computed by gpu_graph_compute_dims and shared by the allocator
 * (gpu_graph_alloc_raw_cap) and the sizing estimate (gpu_graph_session_bytes)
 * so admission control and the allocator can never disagree about the derived
 * quantities.  The per-buffer byte expressions still appear in both functions;
 * the server's estimate-vs-actual reconciliation (>10% drift warning) is the
 * enforcement that they stay in sync. */
typedef struct {
    uint32_t raw_cap;      ///< positions the raw KV ring holds
    uint32_t raw_window;   ///< positions retained per layer in that ring
    uint32_t ctx_size;     ///< session context size the graph is sized for
    uint32_t prefill_cap;  ///< maximum rows one prefill chunk may carry
    uint32_t comp_cap;     ///< worst-case compressed rows per layer (the ratio-4 bound)
    uint32_t attn_comp_stage_cap;  ///< rows the attention compressor staging buffer holds; only meaningful under PULSAR_ATTN_PACK
    /** Per-layer compressed capacity, sized from each layer's ACTUAL ratio --
     * a ratio-128 layer needs far fewer rows than the ratio-4 bound in
     * comp_cap, and sizing every layer at that bound wastes most of it. */
    uint32_t layer_comp_cap[PULSAR_MAX_LAYER];
    uint64_t hc_dim;          ///< HC carrier width
    uint64_t mix_hc;          ///< width of the HC mix projection output
    uint64_t q_rank;          ///< low-rank query latent width
    uint64_t q_dim;           ///< query width in head space
    uint64_t low_dim;         ///< width of the attention output's low-rank 'a' projection
    uint64_t shared_dim;      ///< shared-expert intermediate width
    uint64_t routed_mid_dim;  ///< routed-expert intermediate width
    uint64_t vocab_dim;       ///< output head width
    uint64_t comp_width_max;  ///< widest compressed row across layers; sizes the shared staging
    uint64_t indexer_q_dim;   ///< indexer query width
} gpu_graph_dims;

static void gpu_graph_compute_dims(
        gpu_graph_dims *d,
        const pulsar_weights       *weights,
        const pulsar_layer_weights *layer,
        uint32_t                 raw_cap,
        uint32_t                 ctx_size,
        uint32_t                 prefill_cap) {
    memset(d, 0, sizeof(*d));
    if (raw_cap == 0) raw_cap = 1;
    if (ctx_size == 0) ctx_size = raw_cap;
    if (prefill_cap == 0) prefill_cap = 1;
    uint32_t raw_window = PULSAR_N_SWA;
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
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    d->comp_cap = ctx_size / min_ratio + 2u;
    if (d->comp_cap < 2u) d->comp_cap = 2u;
    d->attn_comp_stage_cap = prefill_cap / min_ratio + 2u;
    if (d->attn_comp_stage_cap < 2u) d->attn_comp_stage_cap = 2u;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) {
            d->layer_comp_cap[il] = 0;
        } else {
            d->layer_comp_cap[il] = ctx_size / ratio + 2u;
            if (d->layer_comp_cap[il] < 2u) d->layer_comp_cap[il] = 2u;
        }
    }

    d->hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    d->mix_hc = 2ull * PULSAR_N_HC + (uint64_t)PULSAR_N_HC * PULSAR_N_HC;
    d->q_rank = layer->attn_q_a->dim[1];
    d->q_dim = (uint64_t)PULSAR_N_HEAD * PULSAR_N_HEAD_DIM;
    d->low_dim = (uint64_t)PULSAR_N_OUT_GROUP * PULSAR_N_LORA_O;
    d->shared_dim = layer->ffn_gate_shexp->dim[1];
    d->routed_mid_dim = layer->ffn_gate_exps->dim[1];
    d->vocab_dim = weights->output ? weights->output->dim[1] : PULSAR_N_VOCAB;
    d->comp_width_max = 2ull * (PULSAR_N_HEAD_DIM > PULSAR_N_INDEXER_HEAD_DIM
        ? PULSAR_N_HEAD_DIM
        : PULSAR_N_INDEXER_HEAD_DIM);
    d->indexer_q_dim = (uint64_t)PULSAR_N_INDEXER_HEAD * PULSAR_N_INDEXER_HEAD_DIM;
}



/* TRUE total per-session GPU byte cost: the sum of every pulsar_gpu_tensor that
 * gpu_graph_alloc_raw_cap (and, when enable_spec, gpu_graph_init_dspark_target)
 * allocates for one session.  This is what admission control must price a
 * session at — the packed KV estimate (pulsar_context_memory_estimate_packed)
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
 * negligible, absorbed by PULSAR_SERVER_MEM_FLOOR_BYTES; do not re-derive):
 *   - the lazy gpu_graph_ensure_batch_ffn_out buffer (only allocated under
 *     steering/diagnostics);
 *   - directional-steering dirs (gpu_graph_load_directional_steering,
 *     ~PULSAR_N_LAYER*PULSAR_N_EMBD floats — these ARE inside the measured create
 *     delta, so they show up in reconciliation but never in this estimate);
 *   - the spec snapshot/restore descriptor tables that
 *     pulsar_gpu_batched_copy_prepare lazily cudaMallocs on the first fused
 *     spec step (~KBs; raw cudaMalloc, outside both this estimate and the
 *     pulsar_gpu_tensor byte counter that reconciliation measures). */
uint64_t gpu_graph_session_bytes(
        const pulsar_weights       *weights,
        const pulsar_layer_weights *layer,
        uint32_t                 raw_cap,
        uint32_t                 ctx_size,
        uint32_t                 prefill_cap,
        bool                     enable_spec) {
    return gpu_graph_session_bytes_banked(weights, layer, raw_cap, ctx_size,
                                          prefill_cap, enable_spec,
                                          gpu_graph_bank_pool_n());
}

uint64_t gpu_graph_session_bytes_banked(
        const pulsar_weights       *weights,
        const pulsar_layer_weights *layer,
        uint32_t                 raw_cap,
        uint32_t                 ctx_size,
        uint32_t                 prefill_cap,
        bool                     enable_spec,
        uint32_t                 n_banks_in) {
    gpu_graph_dims dz;
    gpu_graph_compute_dims(&dz, weights, layer, raw_cap, ctx_size, prefill_cap);
    const uint64_t pc = dz.prefill_cap;
    const uint64_t f32 = sizeof(float);
    const uint64_t hc = PULSAR_HC_ELT_SIZE;   /* HC residual carrier element (BF16); task #62 */

    /* Persistent KV caches (raw ring, packed attn comp, indexer comp) plus the
     * indexer_scores working buffer — shared with the managed-vs-device
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
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_state = (uint64_t)coff * PULSAR_N_HEAD_DIM *
                                    coff * ratio * f32;
        total += n_banks * 2ull * attn_state;             /* layer_attn_state_kv/score */
        /* inc 6: banked pools carry a snapshot lane PER BANK (batched spec
         * verify snapshots every decode bank before the shared forward). */
        if (enable_spec) total += (n_banks > 0 ? n_banks : 1) * 2ull * attn_state;
        if (ratio == 4) {
            const uint64_t index_state = (uint64_t)coff * PULSAR_N_INDEXER_HEAD_DIM *
                                         coff * ratio * f32;
            total += n_banks * 2ull * index_state;        /* layer_index_state_kv/score */
            if (enable_spec) total += (n_banks > 0 ? n_banks : 1) * 2ull * index_state;
        }
    }

    /* Single-token graph buffers. */
    total += dz.hc_dim * hc;                              /* cur_hc (carrier) */
    total += dz.hc_dim * f32;                             /* flat_hc (RMSNorm out, f32) */
    total += dz.mix_hc * f32;                             /* hc_split (views free) */
    total += (uint64_t)PULSAR_N_EMBD * f32;                  /* attn_norm */
    total += (uint64_t)PULSAR_N_HEAD_DIM * f32;              /* kv */
    total += (uint64_t)dz.attn_comp_stage_cap * PULSAR_N_HEAD_DIM * f32; /* attn_comp_stage */
    total += (uint64_t)dz.comp_cap * PULSAR_N_INDEXER_HEAD_DIM * f32;    /* idx_comp_stage */
    total += (uint64_t)(PULSAR_N_INDEXER_TOP_K ? PULSAR_N_INDEXER_TOP_K : 1u) *
             pc * sizeof(uint32_t);                       /* comp_selected */
    total += (uint64_t)PULSAR_N_EMBD * f32;                  /* ffn_norm */
    total += 2ull * PULSAR_N_HC * f32;                       /* output_pre, output_weights */
    total += 2ull * PULSAR_N_EMBD * f32;                     /* output_embd, output_norm */
    total += dz.vocab_dim * f32;                          /* logits */
    total += pc * sizeof(int32_t);                        /* prefill_tokens */
    /* Multi-row logits readback slab.  Allocated unconditionally (NOT under
     * enable_spec): besides the DSpark draft/verify passes it is the output
     * buffer of every batched multi-row head — gpu_graph_verify_suffix_tops
     * and the Tier-2 batched multi-session decode driver
     * (gpu_graph_decode_multiseq_batch) both read their rows out of it, and
     * those paths must work with speculation disabled (--no-dspark,
     * pulsar-bench/pulsar-eval/agent, or any model without a merged drafter). */
    total += (uint64_t)PULSAR_SPEC_LOGITS_ROWS * PULSAR_N_VOCAB * f32; /* spec_logits */

    /* Batch (prefill working set) buffers — the pc-scaled bulk that dominates
     * the non-KV cost (~4 GiB at pc=4096 on Flash). */
    total += 2ull * pc * dz.hc_dim * hc;                  /* batch_cur/next_hc (carriers) */
    total += pc * dz.hc_dim * f32;                        /* batch_flat_hc (RMSNorm out, f32) */
    total += 2ull * pc * dz.mix_hc * f32;                 /* batch_hc_mix/split */
    total += pc * PULSAR_N_EMBD * f32;                       /* batch_attn_norm */
    if (gpu_graph_f32_store_observed_any())
        total += pc * PULSAR_N_EMBD * f32;                   /* batch_attn_cur (dump-only) */
    total += 2ull * pc * dz.q_rank * f32;                 /* batch_qr/qr_norm */
    total += pc * dz.q_dim * PULSAR_Q_ELT_SIZE;           /* batch_q */
    total += 2ull * pc * PULSAR_N_HEAD_DIM * f32;            /* batch_kv_raw/kv */
    total += 2ull * pc * dz.comp_width_max * f32;         /* batch_comp_kv/sc */
    total += pc * dz.indexer_q_dim * f32;                 /* batch_indexer_q (rope staging) */
    total += pc * (uint64_t)PULSAR_N_INDEXER_HEAD * PULSAR_ENGINE_IDXFP4_ROWBYTES; /* batch_indexer_qp (packed) */
    total += pc * PULSAR_N_INDEXER_HEAD * f32;               /* batch_indexer_weights */
    total += pc * dz.q_dim * PULSAR_HEADS_ELT_SIZE;       /* batch_heads */
    total += pc * dz.low_dim * f32;                       /* batch_attn_low */
    total += pc * PULSAR_N_EMBD * f32;                       /* batch_attn_out */
    total += pc * dz.hc_dim * hc;                         /* batch_after_attn_hc (carrier) */
    total += 2ull * pc * PULSAR_N_EMBD * f32;                /* batch_ffn_cur/norm */
    total += pc * dz.shared_dim * (f32 + 2ull * PULSAR_SHARED_ACT_ELT_SIZE); /* batch_shared_mid f32 + gate/up f16 (L033) */
    total += pc * PULSAR_N_EMBD * f32;                       /* batch_shared_out */
    total += 2ull * pc * PULSAR_N_EXPERT * f32;              /* batch_router_logits/probs */
    total += pc * PULSAR_N_EXPERT_USED * (sizeof(int) + f32); /* batch_router_selected/weights */
    total += 2ull * pc * PULSAR_N_EXPERT_USED * dz.routed_mid_dim * f32; /* batch_routed_up/mid */
    total += pc * PULSAR_N_EXPERT_USED * PULSAR_N_EMBD * f32;   /* batch_routed_down */
    total += pc * PULSAR_N_EMBD * f32;                       /* batch_routed_out */

    /* DSpark drafter graph state (gpu_graph_init_dspark_target), allocated by
     * pulsar_session_create whenever the engine has a drafter loaded — the
     * production merged GGUF auto-enables it. */
    if (enable_spec) {
        total += 3ull * PULSAR_N_EMBD * f32;                 /* dspark_target_h[3] */
        total += 3ull * 17 * PULSAR_N_EMBD * f32;            /* dspark_target_h_batch[3] */
        /* Option F: dspark_raw_cache[3] + dspark_prompt_h[3] are BANKED slabs
         * (n_banks lanes) so the N=2 spec-time-slice lane keeps a warm ring per
         * bank; the rest of the drafter state is shared across banks. */
        total += (uint64_t)n_banks * 3ull * PULSAR_DSPARK_DRAFT_WINDOW
                 * PULSAR_ENGINE_ATTN_PACK_ROWBYTES;          /* dspark_raw_cache[3], packed */
        total += (uint64_t)PULSAR_N_EMBD * f32;              /* dspark_main_x */
        total += 3ull * pc * PULSAR_N_EMBD * f32;            /* dspark_bulk_h[3] */
        total += (uint64_t)n_banks * 3ull * PULSAR_DSPARK_DRAFT_WINDOW * PULSAR_N_EMBD * f32; /* dspark_prompt_h[3] */
        const uint64_t attn_w = 2ull * PULSAR_N_HEAD_DIM;
        const uint64_t idx_w = 2ull * PULSAR_N_INDEXER_HEAD_DIM;
        for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
            const uint32_t ratio = pulsar_layer_compress_ratio(il);
            if (ratio == 0) continue;
            total += 2ull * (PULSAR_SPEC_LOGITS_ROWS + 1u) * attn_w * f32;            /* spec_comp_kv/sc_save */
            if (ratio == 4) total += 2ull * (PULSAR_SPEC_LOGITS_ROWS + 1u) * idx_w * f32; /* spec_icomp_kv/sc_save */
        }
        total += (uint64_t)PULSAR_N_HEAD_DIM * f32;          /* spec_comp_scratch_row */
        total += 3ull * PULSAR_N_EMBD * f32;                 /* dspark_concat */
        total += (uint64_t)PULSAR_N_EMBD * f32;              /* dspark_proj_out */
        total += 3ull * PULSAR_N_HEAD_DIM * f32;             /* dspark_seed_kv/norm/rot */
        total += (uint64_t)PULSAR_N_VOCAB * f32;             /* dspark_markov_logits */
    }
    return total;
}



/* Tier-2 overcommit (task #55, increment 1): the DEMAND-PAGED (cudaMallocManaged,
 * physical-on-touch) bytes of ONE bank's ctx-scaled comp + index caches at the
 * given context.  This is exactly the comp/index portion of
 * gpu_graph_kv_cache_bytes_for_context (steering.cpp) MINUS the eager raw ring —
 * the part the overcommit auto-size reserves as VA only and does NOT charge at
 * admission (the eager floor is charged; physical materializes as the frontier
 * grows, tracked by gpu_graph_touched_kv_bytes).  Row widths track the ACTUAL
 * packed storage (PULSAR_ATTN_PACK attn comp + MXFP4 indexer), matching the slab
 * allocator in gpu_graph_bank_slabs_alloc. */
uint64_t gpu_graph_demand_paged_bytes_per_bank(uint32_t ctx_size) {
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = PULSAR_ENGINE_IDXFP4_ROWBYTES;
    uint64_t bytes = 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t comp_cap = (uint64_t)(ctx_size / ratio + 2u);
        bytes += comp_cap * attn_row;
        if (ratio == 4) bytes += comp_cap * idx_row;
    }
    return bytes;
}



/* PULSAR_KV_MANAGED: measurement override for the managed-vs-device placement of
 * the PERSISTENT KV caches (raw ring + attn/index comp caches), for the
 * multi-session UVM over-provisioning design (plan addendum 2026-07-14 —
 * first-touch fault cost must be measured before committing to
 * always-managed slot KV). Unset/empty → the size-based policy
 * (pulsar_gpu_should_use_managed_kv_cache) decides, as before. "1"/"on" →
 * force cudaMallocManaged; "0"/"off"/"false" → force device cudaMalloc.
 * Read once per process (static cache) and consulted only at graph
 * allocation (session create) — never on a token/layer path. */
static int gpu_graph_kv_managed_override(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *v = getenv("PULSAR_KV_MANAGED");
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
                    "pulsar: PULSAR_KV_MANAGED=\"%s\" not recognized "
                    "(want 1/on/true or 0/off/false); using size policy\n", v);
            cached = -1;
        }
    }
    return cached;
}



/* PULSAR_MSEQ_BANKS: Tier-2 bank-pool size for the next graph allocation.
 * 1 (default) keeps the classic single-session cache layout; 2..PULSAR_MSEQ_MAX
 * allocates the fixed per-bank slabs (pulsar_bank_slabs) and installs bank-0
 * views, so every existing single-session path runs unmodified against
 * bank 0.  Read once per process at graph allocation — never on a
 * token/layer path.  Interim wiring: later increments make the server pass
 * the pool size explicitly instead. */
uint32_t gpu_graph_bank_pool_n(void) {
    static long cached = -1;
    if (cached < 0) {
        const char *v = getenv("PULSAR_MSEQ_BANKS");
        long n = 1;
        if (v && v[0]) {
            char *end = NULL;
            n = strtol(v, &end, 10);
            /* Tolerate trailing whitespace (e.g. "2\n" from a shell here-doc),
             * like the PULSAR_CUDA_DECODE_INDEXER_SPARSE_THRESHOLD parser. */
            while (end && isspace((unsigned char)*end)) end++;
            if (end == v || (end && *end != '\0') || n < 1) {
                fprintf(stderr,
                        "pulsar: PULSAR_MSEQ_BANKS=\"%s\" not recognized (want 1..%u); "
                        "bank pool disabled\n", v, PULSAR_MSEQ_MAX);
                n = 1;
            } else if (n > (long)PULSAR_MSEQ_MAX) {
                fprintf(stderr, "pulsar: PULSAR_MSEQ_BANKS=%ld clamped to %u\n",
                        n, PULSAR_MSEQ_MAX);
                n = (long)PULSAR_MSEQ_MAX;
            }
        }
        cached = n;
    }
    return (uint32_t)cached;
}



/* Allocate the fixed per-bank KV slabs (layout: pulsar_bank_slabs in
 * pulsar_engine_internal.h; design adapted from Entrpi/ds4 v0.2 — citation at
 * the struct).  Per-bank byte expressions MUST match the single-session
 * branches in gpu_graph_alloc_raw_cap below and the pricing in
 * gpu_graph_session_bytes.  Compressor state lanes are primed for every bank
 * here (kv = 0, score = -inf), exactly like a fresh single-session graph, so
 * a later bank admit starts from the same state a cold session would. */
static bool gpu_graph_bank_slabs_alloc(
        pulsar_gpu_graph      *g,
        uint32_t              n_banks,
        bool                  managed_kv_cache,
        const gpu_graph_dims *dz,
        bool                  enable_spec) {
    pulsar_bank_slabs *b = &g->banks;
    /* The raw KV ring is PULSAR_ATTN_PACK rows: 584 B at head_dim 512, the same
      * 448 E4M3 + 8 scale + 64 bf16 layout the compressed pool uses. It was
      * __half (1024 B) until 2026-08-17, which spent 2 bytes per element on nope
      * dims that hold E4M3 precision, in a dtype the source model has nowhere. */
    b->n_banks = n_banks;
    b->cur_bank = 0;
    b->raw_bank_bytes = (uint64_t)dz->raw_cap * PULSAR_ENGINE_ATTN_PACK_ROWBYTES;
    bool ok = true;
    for (uint32_t il = 0; il < PULSAR_N_LAYER && ok; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        /* uint32 row-ABI audit: batched kernels address rows as
         * seq_id * cap + local in uint32; reject the geometry up front. */
        if ((uint64_t)n_banks * dz->raw_cap > 4294967296ull ||
            (uint64_t)n_banks * dz->layer_comp_cap[il] > 4294967296ull) {
            fprintf(stderr,
                    "pulsar: bank pool geometry overflows the uint32 row ABI "
                    "(%u banks x raw_cap %u / comp_cap %u)\n",
                    n_banks, dz->raw_cap, dz->layer_comp_cap[il]);
            return false;
        }
        b->raw[il] = gpu_graph_alloc_kv_cache_tensor(
                managed_kv_cache, (uint64_t)n_banks * b->raw_bank_bytes);
        ok = b->raw[il] != NULL;
        if (!ok || ratio == 0) continue;

        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_width = (uint64_t)coff * PULSAR_N_HEAD_DIM;
        const uint64_t attn_rows = (uint64_t)coff * ratio;
        const uint64_t attn_lane = attn_width * attn_rows * sizeof(float);
        b->comp_bank_bytes[il] = (uint64_t)dz->layer_comp_cap[il] *
                                 gpu_graph_attn_comp_cache_row_bytes();
        b->astate_bank_bytes[il] = attn_lane;
        /* Increment 2a: one cudaMallocManaged PER BANK (not one n_banks*bytes
         * slab) so the eviction guard can cudaFree a single idle bank's physical
         * directly. Uniform stride (comp_bank_bytes[il] is bank-independent). */
        void *comp_ptr_h[PULSAR_MSEQ_MAX];
        for (uint32_t bk = 0; ok && bk < n_banks; bk++) {
            b->comp[il][bk] = pulsar_gpu_tensor_alloc_managed(b->comp_bank_bytes[il]);
            ok = b->comp[il][bk] != NULL;
            if (ok) comp_ptr_h[bk] = pulsar_gpu_tensor_device_ptr(b->comp[il][bk]);
        }
        b->askv[il] = pulsar_gpu_tensor_alloc((uint64_t)n_banks * attn_lane);
        b->assc[il] = pulsar_gpu_tensor_alloc((uint64_t)n_banks * attn_lane);
        if (enable_spec) {
            /* No fill: a snapshot always writes a lane before its restore
             * reads it, and nothing else reads these. */
            b->spec_askv[il] = pulsar_gpu_tensor_alloc((uint64_t)n_banks * attn_lane);
            b->spec_assc[il] = pulsar_gpu_tensor_alloc((uint64_t)n_banks * attn_lane);
            ok = ok && b->spec_askv[il] && b->spec_assc[il];
        }
        /* Device base-pointer table (indexed by seq_id) the batched READ kernels
         * use instead of base + seq_id*comp_cap over one slab. */
        if (ok) b->comp_bases[il] = pulsar_gpu_tensor_alloc((uint64_t)n_banks * sizeof(void *));
        ok = ok && b->askv[il] && b->assc[il] && b->comp_bases[il] &&
             pulsar_gpu_tensor_write(b->comp_bases[il], 0, comp_ptr_h,
                                  (uint64_t)n_banks * sizeof(void *)) &&
             gpu_tensor_fill_f32(b->askv[il], 0.0f,
                                 (uint64_t)n_banks * attn_width * attn_rows) &&
             gpu_tensor_fill_f32(b->assc[il], PULSAR_NEG_INF,
                                 (uint64_t)n_banks * attn_width * attn_rows);
        if (ok && ratio == 4) {
            const uint64_t index_width = (uint64_t)coff * PULSAR_N_INDEXER_HEAD_DIM;
            const uint64_t index_rows = (uint64_t)coff * ratio;
            const uint64_t index_lane = index_width * index_rows * sizeof(float);
            const uint64_t index_row_bytes = PULSAR_ENGINE_IDXFP4_ROWBYTES;
            b->index_bank_bytes[il] = (uint64_t)dz->layer_comp_cap[il] *
                                      index_row_bytes;
            b->istate_bank_bytes[il] = index_lane;
            void *index_ptr_h[PULSAR_MSEQ_MAX];
            for (uint32_t bk = 0; ok && bk < n_banks; bk++) {
                b->index[il][bk] = pulsar_gpu_tensor_alloc_managed(b->index_bank_bytes[il]);
                ok = b->index[il][bk] != NULL;
                if (ok) index_ptr_h[bk] = pulsar_gpu_tensor_device_ptr(b->index[il][bk]);
            }
            b->iskv[il] = pulsar_gpu_tensor_alloc((uint64_t)n_banks * index_lane);
            b->issc[il] = pulsar_gpu_tensor_alloc((uint64_t)n_banks * index_lane);
            if (enable_spec) {
                b->spec_iskv[il] = pulsar_gpu_tensor_alloc((uint64_t)n_banks * index_lane);
                b->spec_issc[il] = pulsar_gpu_tensor_alloc((uint64_t)n_banks * index_lane);
                ok = ok && b->spec_iskv[il] && b->spec_issc[il];
            }
            if (ok) b->index_bases[il] = pulsar_gpu_tensor_alloc((uint64_t)n_banks * sizeof(void *));
            ok = ok && b->iskv[il] && b->issc[il] && b->index_bases[il] &&
                 pulsar_gpu_tensor_write(b->index_bases[il], 0, index_ptr_h,
                                      (uint64_t)n_banks * sizeof(void *)) &&
                 gpu_tensor_fill_f32(b->iskv[il], 0.0f,
                                     (uint64_t)n_banks * index_width * index_rows) &&
                 gpu_tensor_fill_f32(b->issc[il], PULSAR_NEG_INF,
                                     (uint64_t)n_banks * index_width * index_rows);
            /* L120 value-half: committed-projection ring lanes (32 slots x
             * width-256 rows; attn and indexer widths are both 256 at
             * ratio 4).  No fill: a rewind replay only reads slots inside
             * the deposited [lo, hi) span. */
            b->pring_bank_bytes = (uint64_t)PULSAR_REWIND_RING_DEPTH * attn_width * sizeof(float);
            b->apkv[il] = pulsar_gpu_tensor_alloc((uint64_t)n_banks * b->pring_bank_bytes);
            b->apsc[il] = pulsar_gpu_tensor_alloc((uint64_t)n_banks * b->pring_bank_bytes);
            b->ipkv[il] = pulsar_gpu_tensor_alloc((uint64_t)n_banks * b->pring_bank_bytes);
            b->ipsc[il] = pulsar_gpu_tensor_alloc((uint64_t)n_banks * b->pring_bank_bytes);
            ok = ok && b->apkv[il] && b->apsc[il] && b->ipkv[il] && b->ipsc[il];
        }
        if (pulsar_layer_compress_ratio(il) == 128u) {
            /* L124: undo lanes (32 x head_dim f32, kv + sc).  No fill: only
             * slots inside the host ring's recorded entries are ever read. */
            b->rulane_bank_bytes = 32ull * PULSAR_N_HEAD_DIM * sizeof(float);
            b->rukv[il] = pulsar_gpu_tensor_alloc((uint64_t)n_banks * b->rulane_bank_bytes);
            b->rusc[il] = pulsar_gpu_tensor_alloc((uint64_t)n_banks * b->rulane_bank_bytes);
            ok = ok && b->rukv[il] && b->rusc[il];
        }
    }
    /* plan-33 inc C: the partial-fork boundary-row stash (one packed comp row +
     * one packed index row per (bank, layer); a few hundred KB total). */
    if (ok) {
        const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
        const uint64_t idx_row = PULSAR_ENGINE_IDXFP4_ROWBYTES;
        g->emit_stash_comp = pulsar_gpu_tensor_alloc((uint64_t)n_banks * PULSAR_N_LAYER * attn_row);
        g->emit_stash_index = pulsar_gpu_tensor_alloc((uint64_t)n_banks * PULSAR_N_LAYER * idx_row);
        ok = g->emit_stash_comp && g->emit_stash_index;
    }
    return ok;
}



/* Write one bank's entry in the comp/index base-pointer tables (device arrays).
 * A NULL slab nulls the entry so a stray batched-kernel read of an evicted bank
 * faults instead of touching freed pages. */
static bool bank_bases_set(pulsar_gpu_graph *g, uint32_t il, uint32_t bank,
                           pulsar_gpu_tensor *comp_slab, pulsar_gpu_tensor *index_slab) {
    pulsar_bank_slabs *b = &g->banks;
    bool ok = true;
    if (b->comp_bases[il]) {
        void *cp = comp_slab ? pulsar_gpu_tensor_device_ptr(comp_slab) : NULL;
        ok = pulsar_gpu_tensor_write(b->comp_bases[il], (uint64_t)bank * sizeof(void *),
                                  &cp, sizeof(void *)) != 0;
    }
    if (ok && b->index_bases[il]) {
        void *ip = index_slab ? pulsar_gpu_tensor_device_ptr(index_slab) : NULL;
        ok = pulsar_gpu_tensor_write(b->index_bases[il], (uint64_t)bank * sizeof(void *),
                                  &ip, sizeof(void *)) != 0;
    }
    return ok;
}

/* Tier-2 task #55 increment 2b — EVICTION RECLAIM primitive. Free ONE idle bank's
 * own comp/index physical by a DIRECT cudaFree of its per-bank split allocations
 * (the only primitive that returns physical on GB10; Step-1/2a reclaim gate).
 * Nulls the slab pointers and their base-table entries, and ZEROES this bank's
 * frontier counters (ms_n_comp/ms_n_index_comp) so touched_kv_bytes stops counting
 * a freed bank — else the guard's projected never drops after a spill and it
 * cascades, evicting every idle bank on one breach (review finding 2). The disk
 * snapshot preserves the real counts for restore. MUST NOT be the installed (cur)
 * bank — the caller repoints away first. The eager raw ring + state lanes
 * (contiguous, bounded floor) are NOT freed and survive the cycle in place.
 *
 * Contract (review finding 4 — no ambiguous half-evicted state): returns FALSE
 * ONLY on a precondition refusal (bad args / cur bank) where NOTHING was freed and
 * the bank stays live. Once freeing begins it returns TRUE — the bank IS physically
 * evicted; a base-table device-write failure is logged HARD but does not un-evict
 * the bank (the stale table entry is never read while the bank is evicted, and
 * restore rebuilds it), so the caller can unconditionally mark it spilled. */
bool gpu_graph_bank_free_physical(pulsar_gpu_graph *g, uint32_t bank) {
    if (!g || g->banks.n_banks == 0 || bank >= g->banks.n_banks) return false;
    if (bank == g->banks.cur_bank) return false;   /* precondition: never free cur */
    pulsar_bank_slabs *b = &g->banks;
    bool table_ok = true;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        pulsar_gpu_tensor_free(b->comp[il][bank]);
        b->comp[il][bank] = NULL;
        if (ratio == 4) {
            pulsar_gpu_tensor_free(b->index[il][bank]);
            b->index[il][bank] = NULL;
        }
        table_ok = bank_bases_set(g, il, bank, NULL, NULL) && table_ok;
        /* Finding 2: a freed bank contributes 0 resident KV. */
        g->ms_n_comp[bank][il] = 0;
        g->ms_n_index_comp[bank][il] = 0;
    }
    g->ms_proj_ring_lo[bank] = 0u;
    g->ms_proj_ring_hi[bank] = 0u;
    g->ms_r128_undo_head[bank] = 0u;
    g->ms_r128_undo_n[bank] = 0u;
    /* plan-33: an evicted bank's boundary stash is meaningless — disarm the
     * emit-restore hook so a later cold refill cannot restore stale bytes. */
    g->ms_emit_keep[bank] = 0u;
    if (!table_ok) {
        fprintf(stderr,
                "pulsar: WARNING free_physical bank %u: base-table NULL device-write "
                "failed (CUDA); bank IS evicted, table is rebuilt on restore\n", bank);
    }
    return true;   /* bank is physically evicted (slabs freed) */
}

/* Tier-2 task #55 increment 2b — RESTORE alloc primitive. Reallocate ONE evicted
 * bank's comp/index physical (fresh cudaMallocManaged: VA reserved, physical on
 * touch) and rebuild its base-table entries to the new pointers. The caller then
 * reloads the bank's KV (H2D from the disk snapshot) into these. Idempotent: a
 * slab already present is left untouched. Returns false on OOM. */
bool gpu_graph_bank_alloc_physical(pulsar_gpu_graph *g, uint32_t bank) {
    if (!g || g->banks.n_banks == 0 || bank >= g->banks.n_banks) return false;
    pulsar_bank_slabs *b = &g->banks;
    bool ok = true;
    for (uint32_t il = 0; il < PULSAR_N_LAYER && ok; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        if (!b->comp[il][bank]) {
            b->comp[il][bank] = pulsar_gpu_tensor_alloc_managed(b->comp_bank_bytes[il]);
            ok = b->comp[il][bank] != NULL;
        }
        if (ok && ratio == 4 && !b->index[il][bank]) {
            b->index[il][bank] = pulsar_gpu_tensor_alloc_managed(b->index_bank_bytes[il]);
            ok = b->index[il][bank] != NULL;
        }
        if (ok) ok = bank_bases_set(g, il, bank, b->comp[il][bank],
                                    ratio == 4 ? b->index[il][bank] : NULL);
    }
    if (!ok) {
        /* TRANSACTIONAL: roll the bank back to fully-evicted rather than leaving
         * layers 0..k allocated.  A half-allocated bank is the dangerous state —
         * callers (server_bank_restore_spilled) return false, so the slot stays
         * marked spilled, but the bank now LOOKS partly live.  Returning it to a
         * clean evicted state keeps gpu_graph_bank_is_evicted's answer truthful
         * and lets a later retry start from scratch. */
        for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
            if (pulsar_layer_compress_ratio(il) == 0) continue;
            if (b->comp[il][bank])  { pulsar_gpu_tensor_free(b->comp[il][bank]);  b->comp[il][bank] = NULL; }
            if (b->index[il][bank]) { pulsar_gpu_tensor_free(b->index[il][bank]); b->index[il][bank] = NULL; }
            (void)bank_bases_set(g, il, bank, NULL, NULL);
        }
    }
    return ok;
}

/* True when `bank`'s comp/index physical is currently freed (evicted) — the
 * server checks this before restoring on a returning request. */
bool gpu_graph_bank_is_evicted(const pulsar_gpu_graph *g, uint32_t bank) {
    if (!g || g->banks.n_banks == 0 || bank >= g->banks.n_banks) return false;
    /* Scan EVERY compressed layer, not just the first.  The old version sampled
     * layer 0 and returned, on the assumption that slabs are freed in lockstep.
     * That holds for eviction, but NOT for a FAILED (re)allocation:
     * gpu_graph_bank_alloc_physical can fail partway and leave layers 0..k
     * allocated, at which point a layer-0 sample reports the bank LIVE while it
     * is really half-built — and bank_fork_copy would then read NULL/garbage
     * slabs from it (silent cross-conversation KV corruption).  A bank is only
     * "live" when every compressed layer has physical. */
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (pulsar_layer_compress_ratio(il) == 0) continue;
        if (g->banks.comp[il][bank] == NULL) return true;
    }
    return false;
}

/* plan-33 inc C: base alignment for a partial cut = LCM of the layer compress
 * ratios (128 on Flash): a multiple-of-LCM cut leaves every ratio-128 layer with
 * an EMPTY in-progress group; only ratio-4 layers straddle (boundary row). */
uint32_t pulsar_partial_fork_base_align(void) {
    static uint32_t a = 0;
    if (a == 0u) {
        uint32_t m = 4u;
        for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
            const uint32_t r = pulsar_layer_compress_ratio(il);
            if (r > m) m = r;
        }
        a = m;
    }
    return a;
}

/* plan-33 inc C: byte-REPLACE the recomputed ratio-4 boundary row with the stash.
 * Fires after any ratio-4 emit that wrote rows starting below the bank's keep
 * threshold (R/4+1); self-deactivates once emits move past. Byte-copy — NEVER
 * re-encode (MXFP4 QAT is non-idempotent; MXFP8 pack byte-copy is trivially
 * bit-exact too). Same-stream D2D: ordered after the emit's store and before any
 * later attention read. No-op when the pool/stash is absent or keep==0. */
bool gpu_graph_emit_keep_restore(pulsar_gpu_graph *g, uint32_t il, uint32_t bank,
                                 uint32_t row0, uint32_t rows, bool indexer) {
    if (!g || rows == 0u || bank >= PULSAR_MSEQ_MAX) return true;
    const uint32_t keep = g->ms_emit_keep[bank];
    if (keep == 0u || row0 >= keep) return true;
    if (pulsar_layer_compress_ratio(il) != 4u) return true;
    pulsar_gpu_tensor *stash = indexer ? g->emit_stash_index : g->emit_stash_comp;
    if (!stash) return true;
    const uint64_t row_bytes = indexer
        ? (PULSAR_ENGINE_IDXFP4_ROWBYTES)
        : gpu_graph_attn_comp_cache_row_bytes();
    const uint32_t keep4 = keep - 1u;              /* the boundary row index R/4 */
    pulsar_gpu_tensor *cache = indexer ? gpu_graph_bank_index_comp_view(g, il, bank)
                                    : gpu_graph_bank_attn_comp_view(g, il, bank);
    if (!cache) return false;
    const bool ok = pulsar_gpu_tensor_copy(cache, (uint64_t)keep4 * row_bytes,
                                        stash,
                                        ((uint64_t)bank * PULSAR_N_LAYER + il) * row_bytes,
                                        row_bytes) != 0;
    pulsar_gpu_tensor_free(cache);
    return ok;
}

/* Tier-2 PATH-A PARTIAL-CUT FORK (plan-33 increment C, the risky core). Clone
 * bank src's KV TRUNCATED at position R into dst (src==dst = in-place truncate:
 * no copies, counters/stash only). Preconds: pool on, R >= align, R % align == 0,
 * R+4 <= src_len (the boundary row R/4 pools [R-4, R+4) — all inside the
 * validated prefix). Wrapped-ring guard: if src's ring has scrolled past
 * R - raw_window, the replay's attention would read scrolled-out raw rows —
 * REFUSE (caller cold-prefills). Per layer: raw [0,R) (or the whole wrapped
 * ring); ratio-4 comp/index rows [0, R/4+1) — ONE row past the counter (the
 * byte-valid boundary row), counters set to R/4 so it is present-but-invisible;
 * ratio-128 rows [0, R/128) (group closed exactly at a 128-multiple cut); state
 * lanes copied for hygiene (the replay re-seeds ratio-4 state from raw). The
 * boundary rows are stashed (packed bytes, from src) and ms_emit_keep[dst] =
 * R/4+1 arms the emit-restore hook. Caller validates tokens + pins src FIRST. */
bool gpu_graph_bank_fork_copy_cut(pulsar_gpu_graph *g, uint32_t src, uint32_t dst,
                                  uint32_t R, uint32_t src_len) {
    if (!g || g->banks.n_banks == 0) return false;
    if (src >= g->banks.n_banks || dst >= g->banks.n_banks) return false;
    const uint32_t align = pulsar_partial_fork_base_align();
    if (R < align || (R % align) != 0u || (uint64_t)R + 4u > src_len) return false;
    if (gpu_graph_bank_is_evicted(g, src)) return false;
    if (src != dst && gpu_graph_bank_is_evicted(g, dst) &&
        !gpu_graph_bank_alloc_physical(g, dst)) return false;
    pulsar_bank_slabs *b = &g->banks;
    /* Wrapped-ring window guard: ring holds positions [oldest, src_len); the
     * replay from R reads raw rows [R - raw_window, R). */
    const uint32_t rcap = g->raw_cap;
    const uint64_t oldest = src_len > rcap ? (uint64_t)src_len - rcap : 0u;
    if ((uint64_t)R < oldest + g->raw_window) return false;   /* scrolled out */
    if (!g->emit_stash_comp || !g->emit_stash_index) return false;
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = PULSAR_ENGINE_IDXFP4_ROWBYTES;
    const uint32_t keep4 = R / 4u;
    const uint64_t raw_row_bytes = b->raw_bank_bytes / rcap;
    bool ok = true;
    for (uint32_t il = 0; il < PULSAR_N_LAYER && ok; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (src != dst) {
            const uint64_t raw_bytes = (uint64_t)(src_len <= rcap ? R : rcap) * raw_row_bytes;
            if (raw_bytes)
                ok = pulsar_gpu_tensor_copy(b->raw[il], (uint64_t)dst * b->raw_bank_bytes,
                                         b->raw[il], (uint64_t)src * b->raw_bank_bytes,
                                         raw_bytes) != 0;
            if (ok && ratio != 0) {
                const uint64_t crows = ratio == 4u ? (uint64_t)keep4 + 1u
                                                   : (uint64_t)R / ratio;
                if (crows)
                    ok = pulsar_gpu_tensor_copy(b->comp[il][dst], 0, b->comp[il][src], 0,
                                             crows * attn_row) != 0;
                if (ok) ok = pulsar_gpu_tensor_copy(b->askv[il], (uint64_t)dst * b->astate_bank_bytes[il],
                                                 b->askv[il], (uint64_t)src * b->astate_bank_bytes[il],
                                                 b->astate_bank_bytes[il]) != 0;
                if (ok) ok = pulsar_gpu_tensor_copy(b->assc[il], (uint64_t)dst * b->astate_bank_bytes[il],
                                                 b->assc[il], (uint64_t)src * b->astate_bank_bytes[il],
                                                 b->astate_bank_bytes[il]) != 0;
                if (ok && ratio == 4u) {
                    ok = pulsar_gpu_tensor_copy(b->index[il][dst], 0, b->index[il][src], 0,
                                             ((uint64_t)keep4 + 1u) * idx_row) != 0;
                    if (ok) ok = pulsar_gpu_tensor_copy(b->iskv[il], (uint64_t)dst * b->istate_bank_bytes[il],
                                                     b->iskv[il], (uint64_t)src * b->istate_bank_bytes[il],
                                                     b->istate_bank_bytes[il]) != 0;
                    if (ok) ok = pulsar_gpu_tensor_copy(b->issc[il], (uint64_t)dst * b->istate_bank_bytes[il],
                                                     b->issc[il], (uint64_t)src * b->istate_bank_bytes[il],
                                                     b->istate_bank_bytes[il]) != 0;
                }
            }
        }
        if (!ok) break;
        /* Boundary-row stash (from SRC's rows — identical to dst's copy, and the
         * only source for the src==dst truncate) + counters at frontier R. */
        if (ratio == 4u) {
            ok = pulsar_gpu_tensor_copy(g->emit_stash_comp,
                                     ((uint64_t)dst * PULSAR_N_LAYER + il) * attn_row,
                                     b->comp[il][src], (uint64_t)keep4 * attn_row,
                                     attn_row) != 0;
            if (ok) ok = pulsar_gpu_tensor_copy(g->emit_stash_index,
                                     ((uint64_t)dst * PULSAR_N_LAYER + il) * idx_row,
                                     b->index[il][src], (uint64_t)keep4 * idx_row,
                                     idx_row) != 0;
            g->ms_n_comp[dst][il] = keep4;
            g->ms_n_index_comp[dst][il] = keep4;
        } else if (ratio != 0) {
            g->ms_n_comp[dst][il] = R / ratio;
            g->ms_n_index_comp[dst][il] = 0u;
        } else {
            g->ms_n_comp[dst][il] = 0u;
            g->ms_n_index_comp[dst][il] = 0u;
        }
    }
    /* L120 value-half: a partial fork's cut invalidates the projection ring
     * span (ring rows above R are the trunk's future); degraded until dst
     * decodes/prefills 8 fresh positions. */
    g->ms_proj_ring_lo[dst] = 0u;
    g->ms_proj_ring_hi[dst] = 0u;
    g->ms_r128_undo_head[dst] = 0u;
    g->ms_r128_undo_n[dst] = 0u;
    if (ok) g->ms_emit_keep[dst] = keep4 + 1u;
    return ok;
}

/* Tier-2 PATH-A FULL-PREFIX FORK (plan-33 increment A). Device-side D2D clone of
 * bank `src`'s entire committed KV into bank `dst`: per layer the raw ring (whole
 * bank region — position-indexed, stale slots harmlessly copied), the comp
 * frontier rows (ms_n_comp[src] rows at offset 0 of the split alloc), the ratio-4
 * index frontier rows, and the attn/index compressor state lanes; the per-bank
 * frontier counters are mirrored src->dst. No captured-graph invalidation (pure
 * D2D + host counters). The CALLER (pulsar_session_bank_fork) has already memcmp-
 * validated the request prefix against src's committed history and pinned src
 * against eviction — this routine performs the copy only. Refuses if src is
 * evicted (no physical to clone; the caller restores from disk first) and
 * reallocs dst if it was freed. Returns false on a bad geometry / copy error. */
bool gpu_graph_bank_fork_copy(pulsar_gpu_graph *g, uint32_t src, uint32_t dst) {
    if (!g || g->banks.n_banks == 0) return false;
    if (src >= g->banks.n_banks || dst >= g->banks.n_banks || src == dst) return false;
    if (gpu_graph_bank_is_evicted(g, src)) return false;   /* caller restores src first */
    if (gpu_graph_bank_is_evicted(g, dst) && !gpu_graph_bank_alloc_physical(g, dst)) return false;
    pulsar_bank_slabs *b = &g->banks;
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = PULSAR_ENGINE_IDXFP4_ROWBYTES;
    bool ok = true;
    for (uint32_t il = 0; il < PULSAR_N_LAYER && ok; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        /* Raw ring: copy the whole bank region (bounded by raw_cap; the ring is
         * position-indexed so any stale slots are never read at dst's pos). */
        ok = pulsar_gpu_tensor_copy(b->raw[il], (uint64_t)dst * b->raw_bank_bytes,
                                 b->raw[il], (uint64_t)src * b->raw_bank_bytes,
                                 b->raw_bank_bytes) != 0;
        g->ms_n_comp[dst][il] = g->ms_n_comp[src][il];
        g->ms_n_index_comp[dst][il] = g->ms_n_index_comp[src][il];
        if (!ok || ratio == 0) continue;
        const uint64_t csz = (uint64_t)g->ms_n_comp[src][il] * attn_row;
        if (ok && csz) ok = pulsar_gpu_tensor_copy(b->comp[il][dst], 0, b->comp[il][src], 0, csz) != 0;
        if (ok) ok = pulsar_gpu_tensor_copy(b->askv[il], (uint64_t)dst * b->astate_bank_bytes[il],
                                         b->askv[il], (uint64_t)src * b->astate_bank_bytes[il],
                                         b->astate_bank_bytes[il]) != 0;
        if (ok) ok = pulsar_gpu_tensor_copy(b->assc[il], (uint64_t)dst * b->astate_bank_bytes[il],
                                         b->assc[il], (uint64_t)src * b->astate_bank_bytes[il],
                                         b->astate_bank_bytes[il]) != 0;
        if (ratio == 4) {
            const uint64_t isz = (uint64_t)g->ms_n_index_comp[src][il] * idx_row;
            if (ok && isz) ok = pulsar_gpu_tensor_copy(b->index[il][dst], 0, b->index[il][src], 0, isz) != 0;
            if (ok) ok = pulsar_gpu_tensor_copy(b->iskv[il], (uint64_t)dst * b->istate_bank_bytes[il],
                                             b->iskv[il], (uint64_t)src * b->istate_bank_bytes[il],
                                             b->istate_bank_bytes[il]) != 0;
            if (ok) ok = pulsar_gpu_tensor_copy(b->issc[il], (uint64_t)dst * b->istate_bank_bytes[il],
                                             b->issc[il], (uint64_t)src * b->istate_bank_bytes[il],
                                             b->istate_bank_bytes[il]) != 0;
        }
    }
    /* L120 value-half: the fork does not carry the projection ring; the
     * forked bank runs degraded (no rewind window replay) until it deposits
     * 8 fresh positions.  Safe: degraded == pre-fix behavior. */
    g->ms_proj_ring_lo[dst] = 0u;
    g->ms_proj_ring_hi[dst] = 0u;
    g->ms_r128_undo_head[dst] = 0u;
    g->ms_r128_undo_n[dst] = 0u;
    return ok;
}



/* Re-install the graph's per-layer cache views onto `bank`.  Pure host-side
 * pointer surgery (view wrappers are freed/recreated; no device work), so the
 * caller must guarantee the device is idle with respect to the previous
 * bank's views.  The spec frontier copy tables bake raw device pointers of
 * the state views, so they are dropped for lazy rebuild. */
bool gpu_graph_bank_repoint(pulsar_gpu_graph *g, uint32_t bank) {
    if (!g || g->banks.n_banks == 0 || bank >= g->banks.n_banks) return false;
    pulsar_bank_slabs *b = &g->banks;
    if (bank == b->cur_bank) return true;
    bool ok = true;
    for (uint32_t il = 0; il < PULSAR_N_LAYER && ok; il++) {
        pulsar_gpu_tensor_free(g->layer_raw_cache[il]);
        g->layer_raw_cache[il] = pulsar_gpu_tensor_view(
                b->raw[il], (uint64_t)bank * b->raw_bank_bytes, b->raw_bank_bytes);
        ok = g->layer_raw_cache[il] != NULL;
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (!ok || ratio == 0) continue;
        pulsar_gpu_tensor_free(g->layer_attn_comp_cache[il]);
        pulsar_gpu_tensor_free(g->layer_attn_state_kv[il]);
        pulsar_gpu_tensor_free(g->layer_attn_state_score[il]);
        g->layer_attn_comp_cache[il] = pulsar_gpu_tensor_view(
                b->comp[il][bank], 0, b->comp_bank_bytes[il]);
        g->layer_attn_state_kv[il] = pulsar_gpu_tensor_view(
                b->askv[il], (uint64_t)bank * b->astate_bank_bytes[il],
                b->astate_bank_bytes[il]);
        g->layer_attn_state_score[il] = pulsar_gpu_tensor_view(
                b->assc[il], (uint64_t)bank * b->astate_bank_bytes[il],
                b->astate_bank_bytes[il]);
        ok = g->layer_attn_comp_cache[il] && g->layer_attn_state_kv[il] &&
             g->layer_attn_state_score[il];
        /* L124: the ratio-128 undo lanes follow the live views. */
        if (ok && ratio == 128u && b->rukv[il]) {
            pulsar_gpu_tensor_free(g->layer_r128_undo_kv[il]);
            pulsar_gpu_tensor_free(g->layer_r128_undo_sc[il]);
            g->layer_r128_undo_kv[il] = pulsar_gpu_tensor_view(
                    b->rukv[il], (uint64_t)bank * b->rulane_bank_bytes, b->rulane_bank_bytes);
            g->layer_r128_undo_sc[il] = pulsar_gpu_tensor_view(
                    b->rusc[il], (uint64_t)bank * b->rulane_bank_bytes, b->rulane_bank_bytes);
            ok = g->layer_r128_undo_kv[il] && g->layer_r128_undo_sc[il];
        }
        /* inc 6: the spec frontier snapshot lanes follow the live views, so
         * the snapshot machinery (incl. its re-prepared copy tables) is
         * bank-correct with no call-site changes. */
        if (ok && b->spec_askv[il]) {
            pulsar_gpu_tensor_free(g->spec_attn_state_kv[il]);
            pulsar_gpu_tensor_free(g->spec_attn_state_score[il]);
            g->spec_attn_state_kv[il] = pulsar_gpu_tensor_view(
                    b->spec_askv[il], (uint64_t)bank * b->astate_bank_bytes[il],
                    b->astate_bank_bytes[il]);
            g->spec_attn_state_score[il] = pulsar_gpu_tensor_view(
                    b->spec_assc[il], (uint64_t)bank * b->astate_bank_bytes[il],
                    b->astate_bank_bytes[il]);
            ok = g->spec_attn_state_kv[il] && g->spec_attn_state_score[il];
        }
        if (ok && ratio == 4) {
            pulsar_gpu_tensor_free(g->layer_index_comp_cache[il]);
            pulsar_gpu_tensor_free(g->layer_index_state_kv[il]);
            pulsar_gpu_tensor_free(g->layer_index_state_score[il]);
            g->layer_index_comp_cache[il] = pulsar_gpu_tensor_view(
                    b->index[il][bank], 0, b->index_bank_bytes[il]);
            g->layer_index_state_kv[il] = pulsar_gpu_tensor_view(
                    b->iskv[il], (uint64_t)bank * b->istate_bank_bytes[il],
                    b->istate_bank_bytes[il]);
            g->layer_index_state_score[il] = pulsar_gpu_tensor_view(
                    b->issc[il], (uint64_t)bank * b->istate_bank_bytes[il],
                    b->istate_bank_bytes[il]);
            if (b->spec_iskv[il]) {
                pulsar_gpu_tensor_free(g->spec_index_state_kv[il]);
                pulsar_gpu_tensor_free(g->spec_index_state_score[il]);
                g->spec_index_state_kv[il] = pulsar_gpu_tensor_view(
                        b->spec_iskv[il], (uint64_t)bank * b->istate_bank_bytes[il],
                        b->istate_bank_bytes[il]);
                g->spec_index_state_score[il] = pulsar_gpu_tensor_view(
                        b->spec_issc[il], (uint64_t)bank * b->istate_bank_bytes[il],
                        b->istate_bank_bytes[il]);
            }
            /* L120 value-half: the projection rings follow the live views. */
            pulsar_gpu_tensor_free(g->layer_attn_proj_kv[il]);
            pulsar_gpu_tensor_free(g->layer_attn_proj_sc[il]);
            pulsar_gpu_tensor_free(g->layer_index_proj_kv[il]);
            pulsar_gpu_tensor_free(g->layer_index_proj_sc[il]);
            g->layer_attn_proj_kv[il] = pulsar_gpu_tensor_view(
                    b->apkv[il], (uint64_t)bank * b->pring_bank_bytes, b->pring_bank_bytes);
            g->layer_attn_proj_sc[il] = pulsar_gpu_tensor_view(
                    b->apsc[il], (uint64_t)bank * b->pring_bank_bytes, b->pring_bank_bytes);
            g->layer_index_proj_kv[il] = pulsar_gpu_tensor_view(
                    b->ipkv[il], (uint64_t)bank * b->pring_bank_bytes, b->pring_bank_bytes);
            g->layer_index_proj_sc[il] = pulsar_gpu_tensor_view(
                    b->ipsc[il], (uint64_t)bank * b->pring_bank_bytes, b->pring_bank_bytes);
            ok = g->layer_index_comp_cache[il] && g->layer_index_state_kv[il] &&
                 g->layer_index_state_score[il] &&
                 g->layer_attn_proj_kv[il] && g->layer_attn_proj_sc[il] &&
                 g->layer_index_proj_kv[il] && g->layer_index_proj_sc[il];
        }
    }
    /* Option F: swap the per-bank DSpark drafter ring views (present only when
     * the drafter is loaded and the ring was banked).  Same host-only pointer
     * surgery as the KV views above, so the spec path transparently reads the
     * active bank's warm window. */
    if (ok && b->dspark_raw[0]) {
        for (int i = 0; i < 3 && ok; i++) {
            pulsar_gpu_tensor_free(g->dspark_raw_cache[i]);
            pulsar_gpu_tensor_free(g->dspark_prompt_h[i]);
            g->dspark_raw_cache[i] = pulsar_gpu_tensor_view(
                    b->dspark_raw[i], (uint64_t)bank * b->dspark_raw_bank_bytes,
                    b->dspark_raw_bank_bytes);
            g->dspark_prompt_h[i] = pulsar_gpu_tensor_view(
                    b->dspark_prompt[i], (uint64_t)bank * b->dspark_prompt_bank_bytes,
                    b->dspark_prompt_bank_bytes);
            ok = g->dspark_raw_cache[i] && g->dspark_prompt_h[i];
        }
    }
    /* Stale pointer hygiene (mirrors gpu_graph_free). */
    pulsar_gpu_batched_copy_free(g->spec_snap_copies);
    pulsar_gpu_batched_copy_free(g->spec_restore_copies);
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
 * arm/disarm pair.  Contracts at the declarations (pulsar_engine_internal.h)
 * and at the ms_* fields in pulsar_gpu_graph. */

uint32_t gpu_graph_bank_pool_count(const pulsar_gpu_graph *g) {
    return g->banks.n_banks ? g->banks.n_banks : 1u;
}

pulsar_gpu_tensor *gpu_graph_bank_raw_pool(pulsar_gpu_graph *g, uint32_t il) {
    if (!g || il >= PULSAR_N_LAYER) return NULL;
    return g->banks.n_banks ? g->banks.raw[il] : g->layer_raw_cache[il];
}

/* Nominal comp/index operand for the batched attention/indexer wrappers. With
 * per-bank split allocations there is no single slab; the batched (descriptor)
 * path addresses banks through the base-pointer table (below), so this returns
 * bank 0's allocation as the nominal typed operand (its per-bank size drives the
 * wrappers' buffer-size validation). Pool disabled → the classic tensor. */
pulsar_gpu_tensor *gpu_graph_bank_attn_comp_pool(pulsar_gpu_graph *g, uint32_t il) {
    if (!g || il >= PULSAR_N_LAYER) return NULL;
    return g->banks.n_banks ? g->banks.comp[il][0] : g->layer_attn_comp_cache[il];
}

pulsar_gpu_tensor *gpu_graph_bank_index_comp_pool(pulsar_gpu_graph *g, uint32_t il) {
    if (!g || il >= PULSAR_N_LAYER) return NULL;
    return g->banks.n_banks ? g->banks.index[il][0] : g->layer_index_comp_cache[il];
}

/* Per-bank comp/index base-pointer tables (device arrays of n_banks pointers,
 * indexed by seq_id) the batched READ kernels use in place of base +
 * seq_id*comp_cap. NULL when the pool is disabled. */
pulsar_gpu_tensor *gpu_graph_bank_attn_comp_bases(pulsar_gpu_graph *g, uint32_t il) {
    if (!g || il >= PULSAR_N_LAYER) return NULL;
    return g->banks.n_banks ? g->banks.comp_bases[il] : NULL;
}

pulsar_gpu_tensor *gpu_graph_bank_index_comp_bases(pulsar_gpu_graph *g, uint32_t il) {
    if (!g || il >= PULSAR_N_LAYER) return NULL;
    return g->banks.n_banks ? g->banks.index_bases[il] : NULL;
}

/* One accessor body for all six per-(bank,layer) view kinds: a fresh view of
 * the bank's lane in the slab, or (pool disabled) a fresh view wrapping the
 * whole classic tensor — bank 0 only, so a stale bank id cannot silently
 * alias the single session's state. */
static pulsar_gpu_tensor *bank_lane_view(pulsar_gpu_graph *g,
                                      pulsar_gpu_tensor *slab,
                                      const uint64_t *lane_bytes,
                                      pulsar_gpu_tensor *classic,
                                      uint32_t il, uint32_t bank) {
    if (!g || il >= PULSAR_N_LAYER) return NULL;
    if (g->banks.n_banks == 0) {
        if (bank != 0 || !classic) return NULL;
        return pulsar_gpu_tensor_view(classic, 0, pulsar_gpu_tensor_bytes(classic));
    }
    if (bank >= g->banks.n_banks || !slab) return NULL;
    return pulsar_gpu_tensor_view(slab, (uint64_t)bank * lane_bytes[il], lane_bytes[il]);
}

/* Per-bank comp/index view: with split allocations the bank's whole allocation
 * IS its lane (offset 0), so it bypasses the contiguous-slab bank_lane_view. */
static pulsar_gpu_tensor *bank_split_view(pulsar_gpu_graph *g,
                                       pulsar_gpu_tensor *bank_slab,
                                       const uint64_t *lane_bytes,
                                       pulsar_gpu_tensor *classic,
                                       uint32_t il, uint32_t bank) {
    if (!g || il >= PULSAR_N_LAYER) return NULL;
    if (g->banks.n_banks == 0) {
        if (bank != 0 || !classic) return NULL;
        return pulsar_gpu_tensor_view(classic, 0, pulsar_gpu_tensor_bytes(classic));
    }
    if (bank >= g->banks.n_banks || !bank_slab) return NULL;
    return pulsar_gpu_tensor_view(bank_slab, 0, lane_bytes[il]);
}

pulsar_gpu_tensor *gpu_graph_bank_attn_comp_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_split_view(g, g->banks.n_banks ? g->banks.comp[il][bank] : NULL,
                           g->banks.comp_bank_bytes,
                           g->layer_attn_comp_cache[il], il, bank);
}

pulsar_gpu_tensor *gpu_graph_bank_index_comp_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_split_view(g, g->banks.n_banks ? g->banks.index[il][bank] : NULL,
                           g->banks.index_bank_bytes,
                           g->layer_index_comp_cache[il], il, bank);
}

pulsar_gpu_tensor *gpu_graph_bank_attn_state_kv_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_lane_view(g, g->banks.askv[il], g->banks.astate_bank_bytes,
                          g->layer_attn_state_kv[il], il, bank);
}

pulsar_gpu_tensor *gpu_graph_bank_attn_state_score_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_lane_view(g, g->banks.assc[il], g->banks.astate_bank_bytes,
                          g->layer_attn_state_score[il], il, bank);
}

pulsar_gpu_tensor *gpu_graph_bank_index_state_kv_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_lane_view(g, g->banks.iskv[il], g->banks.istate_bank_bytes,
                          g->layer_index_state_kv[il], il, bank);
}

pulsar_gpu_tensor *gpu_graph_bank_index_state_score_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_lane_view(g, g->banks.issc[il], g->banks.istate_bank_bytes,
                          g->layer_index_state_score[il], il, bank);
}

void gpu_graph_bank_counters_capture(pulsar_gpu_graph *g, uint32_t bank) {
    if (!g || bank >= PULSAR_MSEQ_MAX) return;
    /* STAGE 1b: the frontier is no longer copied here. ms_n_comp[bank] IS the
     * storage now -- this loop had become a self-copy (callers capture the
     * LIVE bank, and gpu_graph_n_comp resolves to exactly that row). The other
     * fields below still have scalar twins and still ride the hand-off. */
    /* L120 value-half: the projection-ring span rides the same hand-off. */
    g->ms_proj_ring_lo[bank] = g->proj_ring_lo;
    g->ms_proj_ring_hi[bank] = g->proj_ring_hi;
    memcpy(g->ms_r128_undo_pos[bank], g->r128_undo_pos, sizeof g->r128_undo_pos);
    g->ms_r128_undo_head[bank] = g->r128_undo_head;
    g->ms_r128_undo_n[bank] = g->r128_undo_n;
    /* Option F: the drafter-ring frontier is per-bank too (device rings live in
     * banks.dspark_*), so it rides the same capture/install hand-off. */
    for (int i = 0; i < 3; i++) g->ms_dspark_n_raw[bank][i] = g->dspark_n_raw[i];
    g->ms_dspark_prompt_n[bank] = g->dspark_prompt_n;
    g->ms_dspark_prompt_lo[bank] = g->dspark_prompt_lo;
}

bool gpu_graph_proj_ring_deposit(pulsar_gpu_graph *g, uint32_t il, uint32_t pos,
                                 const pulsar_gpu_tensor *kv_row,
                                 const pulsar_gpu_tensor *sc_row,
                                 bool indexer) {
    pulsar_gpu_tensor *dk = indexer ? g->layer_index_proj_kv[il] : g->layer_attn_proj_kv[il];
    pulsar_gpu_tensor *ds = indexer ? g->layer_index_proj_sc[il] : g->layer_attn_proj_sc[il];
    if (!dk || !ds) return true;   /* no ring on this layer (ratio != 4) */
    /* Ratio-4 widths: attn = 2*head_dim (512) = 4 KiB rows; indexer =
     * 2*indexer_head_dim (128) = 1 KiB rows.  Lanes are attn-sized. */
    const uint64_t row_bytes = (indexer ? 2ull * PULSAR_N_INDEXER_HEAD_DIM
                                        : 2ull * PULSAR_N_HEAD_DIM) * sizeof(float);
    const uint64_t off = (uint64_t)(pos % PULSAR_REWIND_RING_DEPTH) * row_bytes;
    return pulsar_gpu_tensor_copy_async(dk, off, kv_row, 0, row_bytes) != 0 &&
           pulsar_gpu_tensor_copy_async(ds, off, sc_row, 0, row_bytes) != 0;
}

bool gpu_graph_r128_undo_capture(pulsar_gpu_graph *g, uint32_t il, uint32_t pos) {
    pulsar_gpu_tensor *uk = g->layer_r128_undo_kv[il];
    pulsar_gpu_tensor *us = g->layer_r128_undo_sc[il];
    if (!uk || !us) return true;   /* no lane on this layer */
    /* Save the CURRENT state slot (the value the imminent store destroys):
     * ratio-128 state rows are width head_dim, slot = pos %% 128; the lane
     * row is pos %% 32 (unique within any restorable window -- ghost
     * overshoot <= 16 < 32, same argument as the L120 projection ring). */
    const uint64_t row_bytes = (uint64_t)PULSAR_N_HEAD_DIM * sizeof(float);
    const uint64_t state_off = (uint64_t)(pos % 128u) * row_bytes;
    const uint64_t lane_off = (uint64_t)(pos % PULSAR_REWIND_RING_DEPTH) * row_bytes;
    return pulsar_gpu_tensor_copy_async(uk, lane_off, g->layer_attn_state_kv[il],
                                        state_off, row_bytes) != 0 &&
           pulsar_gpu_tensor_copy_async(us, lane_off, g->layer_attn_state_score[il],
                                        state_off, row_bytes) != 0;
}

void gpu_graph_r128_undo_note_pos(pulsar_gpu_graph *g, uint32_t pos) {
    /* Once per position, after every ratio-128 layer captured.  Consecutive
     * duplicate pushes (a position re-stored after a same-target rewind)
     * are fine: restore is idempotent per lane row. */
    g->r128_undo_pos[g->r128_undo_head] = pos;
    g->r128_undo_head = (g->r128_undo_head + 1u) % PULSAR_REWIND_RING_DEPTH;
    if (g->r128_undo_n < 32u) g->r128_undo_n++;
}

void gpu_graph_proj_ring_note_pos(pulsar_gpu_graph *g, uint32_t pos) {
    /* Gap => span restarts.  Deliberately conservative: after a rewind the
     * next deposit lands below the stale hi and restarts the span, so slots
     * claimed under the stale hi (ghost deposits) can never be read. */
    if (g->proj_ring_hi != pos) g->proj_ring_lo = pos;
    g->proj_ring_hi = pos + 1u;
    if (g->proj_ring_lo + PULSAR_REWIND_RING_DEPTH < g->proj_ring_hi)
        g->proj_ring_lo = g->proj_ring_hi - PULSAR_REWIND_RING_DEPTH;
}

void gpu_graph_bank_counters_install(pulsar_gpu_graph *g, uint32_t bank) {
    if (!g || bank >= PULSAR_MSEQ_MAX) return;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        /* STAGE 1b: self-copy. gpu_graph_bank_repoint() has already set
         * cur_bank to `bank`, so this wrote ms_n_comp[bank] onto itself. */
    }
    for (int i = 0; i < 3; i++) g->dspark_n_raw[i] = g->ms_dspark_n_raw[bank][i];
    g->dspark_prompt_n = g->ms_dspark_prompt_n[bank];
    g->dspark_prompt_lo = g->ms_dspark_prompt_lo[bank];
    g->proj_ring_lo = g->ms_proj_ring_lo[bank];
    g->proj_ring_hi = g->ms_proj_ring_hi[bank];
    memcpy(g->r128_undo_pos, g->ms_r128_undo_pos[bank], sizeof g->r128_undo_pos);
    g->r128_undo_head = g->ms_r128_undo_head[bank];
    g->r128_undo_n = g->ms_r128_undo_n[bank];
}

/* Tier-2 overcommit (task #55, increment 1): EXACT touched (physically resident)
 * demand-paged KV bytes across the whole pool — Σ over live banks Σ over layers
 * of (comp frontier rows × comp row bytes + index frontier rows × index row
 * bytes).  Deterministic from the position-driven compressor frontier; no
 * cudaMemGetInfo / MemAvailable.  This is the number the increment-2 eviction
 * guard triggers on, and the accounting-exactness gate proves it tracks the real
 * physical delta.  The CURRENT bank's frontier is live in layer_n_comp /
 * layer_n_index_comp; idle banks keep their frontier in ms_n_comp / ms_n_index_comp
 * (captured on switch-away).  Pool disabled (n_banks==0) → pool_count 1, cur 0 →
 * the classic single-session frontier (layer_n_comp) is summed. Only the ctx-
 * scaled comp/index are counted; the eager raw ring + state lanes are the fixed
 * floor and are already resident (not part of the growing touched set). */
/* Exact touched (physically resident) demand-paged comp/index KV of ONE bank,
 * from its compressor frontier.  cur bank reads the live layer_n_comp; idle banks
 * read their captured ms_n_comp.  The increment-2b guard uses this for the
 * per-bank Δ projection and the smallest-frontier victim tie-break. */
uint64_t gpu_graph_bank_touched_kv_bytes(const pulsar_gpu_graph *g, uint32_t bank) {
    if (!g) return 0;
    const uint32_t nb = gpu_graph_bank_pool_count(g);
    if (bank >= nb) return 0;
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = PULSAR_ENGINE_IDXFP4_ROWBYTES;
    const uint32_t cur = g->banks.n_banks ? g->banks.cur_bank : 0u;
    uint64_t bytes = 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t ncomp = (bank == cur) ? gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il)
                                             : g->ms_n_comp[bank][il];
        bytes += (uint64_t)ncomp * attn_row;
        if (ratio == 4) {
            const uint32_t nidx = (bank == cur) ? gpu_graph_n_index_comp(g, gpu_graph_cur_bank(g), il)
                                                : g->ms_n_index_comp[bank][il];
            bytes += (uint64_t)nidx * idx_row;
        }
    }
    return bytes;
}

uint64_t gpu_graph_touched_kv_bytes(const pulsar_gpu_graph *g) {
    if (!g) return 0;
    const uint32_t nb = gpu_graph_bank_pool_count(g);
    uint64_t bytes = 0;
    for (uint32_t b = 0; b < nb; b++) bytes += gpu_graph_bank_touched_kv_bytes(g, b);
    return bytes;
}

/* Tier-2 task #55 increment 2b — CONSERVATIVE per-bank comp/index growth over one
 * decode quantum of `q` tokens: Σ_layers( ceil(q/ratio)·comp_row + q·index_row ).
 * The index term charges q (not ceil(q/ratio)) rows — a deliberate over-estimate
 * so the guard fires EARLY (safe side). Position-independent, so total Δ =
 * n_live_growing_banks × this. */
uint64_t gpu_graph_quantum_growth_bytes_per_bank(uint32_t q) {
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = PULSAR_ENGINE_IDXFP4_ROWBYTES;
    uint64_t bytes = 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)((q + ratio - 1u) / ratio) * attn_row;
        if (ratio == 4) bytes += (uint64_t)q * idx_row;
    }
    return bytes;
}

bool gpu_graph_multiseq_step_begin(pulsar_gpu_graph *g, const int32_t *pos,
                                   const int32_t *seq, uint32_t n_rows,
                                   bool capture_cur) {
    if (!g || !pos || !seq || n_rows == 0 || n_rows > g->prefill_cap) {
        fprintf(stderr, "pulsar: multiseq step rejected: bad args (n_rows=%u)\n",
                n_rows);
        return false;
    }
    if (g->batch_multiseq) {
        fprintf(stderr, "pulsar: multiseq step rejected: step already armed\n");
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
    bool bank_seen[PULSAR_MSEQ_MAX] = {false};
    int32_t prev_bank = -1;
    for (uint32_t t = 0; t < n_rows; t++) {
        if (seq[t] < 0 || (uint32_t)seq[t] >= n_banks ||
            (uint32_t)seq[t] >= PULSAR_MSEQ_MAX) {
            fprintf(stderr, "pulsar: multiseq step rejected: row %u bank %d "
                            "out of pool (n_banks=%u)\n", t, seq[t], n_banks);
            return false;
        }
        if (seq[t] != prev_bank) {
            if (bank_seen[seq[t]]) {
                fprintf(stderr, "pulsar: multiseq step rejected: bank %d rows "
                                "not contiguous\n", seq[t]);
                return false;
            }
            bank_seen[seq[t]] = true;
            prev_bank = seq[t];
            if (pos[t] <= 0) {
                fprintf(stderr, "pulsar: multiseq step rejected: bank %d first "
                                "position %d <= 0 (admission prefill is "
                                "single-bank classic)\n", seq[t], pos[t]);
                return false;
            }
        } else if ((int64_t)pos[t] != (int64_t)pos[t - 1] + 1) {
            /* int64 arithmetic: pos[t-1] == INT32_MAX would make the int32
             * increment signed overflow (UB) before the per-row bound below
             * could reject it. */
            fprintf(stderr, "pulsar: multiseq step rejected: bank %d positions "
                            "not consecutive within its run (row %u: %d, "
                            "want %lld)\n", seq[t], t, pos[t],
                    (long long)((int64_t)pos[t - 1] + 1));
            return false;
        }
        /* Bound EVERY row (not just each run's first): positions are cast to
         * uint32 downstream (ring slot, visible-comp, RoPE), and the
         * position-derived arithmetic below adds 1. */
        if (pos[t] <= 0 || pos[t] == INT32_MAX) {
            fprintf(stderr, "pulsar: multiseq step rejected: bank %d row %u "
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
        for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
            const uint32_t ratio = pulsar_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint32_t have_comp = use_scalars ? gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il)
                                                   : g->ms_n_comp[b][il];
            const uint32_t have_index = use_scalars ? gpu_graph_n_index_comp(g, gpu_graph_cur_bank(g), il)
                                                    : g->ms_n_index_comp[b][il];
            if (have_comp != p / ratio ||
                (ratio == 4 && have_index != p / ratio)) {
                fprintf(stderr,
                        "pulsar: multiseq step rejected: bank %u frontier not "
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
        g->ms_positions = (int32_t *)xmalloc((size_t)g->prefill_cap * sizeof(int32_t));
        g->ms_seq_id = (int32_t *)xmalloc((size_t)g->prefill_cap * sizeof(int32_t));
        g->batch_positions =
            pulsar_gpu_tensor_alloc((uint64_t)g->prefill_cap * sizeof(int32_t));
        g->batch_seq_id =
            pulsar_gpu_tensor_alloc((uint64_t)g->prefill_cap * sizeof(int32_t));
        if (!g->batch_positions || !g->batch_seq_id) {
            fprintf(stderr, "pulsar: multiseq descriptor alloc failed\n");
            pulsar_gpu_tensor_free(g->batch_positions);
            pulsar_gpu_tensor_free(g->batch_seq_id);
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
    if (!pulsar_gpu_tensor_write(g->batch_positions, 0, pos,
                              (uint64_t)n_rows * sizeof(int32_t)) ||
        !pulsar_gpu_tensor_write(g->batch_seq_id, 0, seq,
                              (uint64_t)n_rows * sizeof(int32_t))) {
        fprintf(stderr, "pulsar: multiseq descriptor upload failed\n");
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
    uint32_t sup[PULSAR_MAX_LAYER];
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        sup[il] = 0;
        if (ratio == 0) continue;
        for (uint32_t t = 0; t < n_rows; t++) {
            const uint32_t v = ((uint32_t)pos[t] + 1u) / ratio;
            if (v > sup[il]) sup[il] = v;
        }
        if (sup[il] > g->layer_comp_cap[il]) {
            fprintf(stderr,
                    "pulsar: multiseq step rejected: superset %u exceeds comp "
                    "cap %u at layer %u\n", sup[il], g->layer_comp_cap[il], il);
            return false;
        }
    }
    /* All rejection points passed: NOW commit the cur-bank capture. */
    if (capture_cur) gpu_graph_bank_counters_capture(g, cur_bank);
    /* STAGE 1b: the cross-bank SUPERSET is no longer written anywhere.
     *
     * This loop used to publish max-over-banks into the scalars, which the step
     * then treated as a read-only working value. With the scalars gone that
     * write lands on ms_n_comp[cur_bank] -- i.e. it overwrites one real bank's
     * frontier with the maximum across all of them, and step_end catches it as
     * "bank N layer 2 frontier 18/18 want 17" whenever another bank is ahead.
     *
     * sup[] is still computed above because the comp-cap rejection needs it;
     * it is a bound check, not stored state. Every consumer of the frontier
     * during the step reads its own bank: the per-row loop through
     * ms_n_comp[ms_seq_id[t]], the kernels through the per-row comp_counts[]
     * device array. */
    g->batch_multiseq_rows = n_rows;
    g->batch_multiseq = true;
    /* plan-34 inc 2/3/4: arm the M-independent custom GEMMs for the DECODE PREFIX
     * of this step. Row layout (inc 4): decode rows first — one 1-row run per
     * decode bank — then the K-row PREFILL run(s). n_dec = the count of leading
     * rows that live in length-1 runs; every dense GEMM runs those rows through the
     * M-independent custom kernel (byte-identical to a decode-only step of that
     * width — gate-4 neutrality) and the trailing prefill rows through the fast
     * tensor-core path. Decode-only (every run length 1) => n_dec == n_rows (arm
     * all, == inc-2). Pure prefill (first run length>1) => n_dec == 0 (arm none,
     * == inc-3). The scan stops at the first multi-row run: the inc-4 scheduler
     * lays decode rows strictly before the prefill run (a length-1 run appearing
     * AFTER a prefill run is not an inc-4 layout and would be treated as prefill —
     * documented, enforced by the row builder / gate). */
    /* L146: a step whose WHOLE batch fits the M-neutral row cap is M-neutral in
     * full, whatever its run structure.  The prefix scan below exists for the
     * mixed step whose trailing run is a K-row prefill chunk (K up to the
     * prefill cap) -- those rows want the tensor-core path.  A production
     * verify batch -- three slots x (1 + draft depth) rows, no prefill run --
     * has NO length-1 runs, so the scan armed nothing: every dense projection
     * took cuBLASLt at grids of 4-32 CTAs (output_a 291 us where the grouped
     * nt kernel takes ~150, output_b 266 vs ~155) and the MXFP4 MoE took the
     * grouped path, padding 54 rows to ~7,000 (gather + swiglu-pack 3.1 ms per
     * layer for ~1 MB of activations) -- L145.  At <= PULSAR_GPU_MNEUTRAL_ROWS_MAX
     * rows there is no run the tensor-core path serves better, and the
     * per-row-exact kernels are the ones the decode prefix already uses, so arm
     * them all.  Every row's output is then independent of its batchmates --
     * the property gate 4 asserts for decode rows, extended to draft rows. */
    uint32_t n_dec = 0;
    if (n_rows <= PULSAR_GPU_MNEUTRAL_ROWS_MAX) {
        n_dec = n_rows;
    } else {
        for (uint32_t t = 0; t < n_rows; ) {
            uint32_t rl = 1;
            while (t + rl < n_rows && seq[t + rl] == seq[t]) rl++;
            if (rl == 1) { n_dec++; t++; } else break;
        }
    }
    pulsar_gpu_matmul_set_batch_mneutral((int)n_dec);
    return true;
}

bool gpu_graph_multiseq_step_end(pulsar_gpu_graph *g) {
    if (!g || !g->batch_multiseq) return false;
    pulsar_gpu_matmul_set_batch_mneutral(0);
    g->batch_multiseq = false;
    const uint32_t n_rows = g->batch_multiseq_rows;
    g->batch_multiseq_rows = 0;
    /* Self-check (host ints only): every batched bank's frontier advanced to
     * exactly its position-derived value — (last_pos+1)/ratio — and the
     * position-derived value -- (last_pos+1)/ratio.  A miss here is the
     * silent-KV-corruption class; fail loud so the driver aborts the session
     * instead of serving garbage.
     * STAGE 1b: the companion "scalar superset" check is gone with the scalar
     * (see below) -- there is no longer a shared value for a bank to corrupt. */
    bool ok = true;
    for (uint32_t il = 0; il < PULSAR_N_LAYER && ok; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
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
                        "pulsar: multiseq step_end FAILED: bank %u layer %u "
                        "frontier %u/%u want %u (ratio %u)\n",
                        b, il, g->ms_n_comp[b][il],
                        g->ms_n_index_comp[b][il], want, ratio);
                ok = false;
            }
        }
        /* STAGE 1b: the "scalar superset mutated mid-step" checks are gone with
         * the scalar. They existed to catch corruption of a SHARED value that
         * several banks' rows wrote through; there is no shared value now, so
         * the failure mode is unrepresentable rather than merely detected.
         * The per-bank assertions above are the real check and they remain. */
        (void)sup;
    }
    return ok;
}



/* Allocate the GPU graph state for a chosen raw-cache capacity.  The model
 * weights are not copied here; tensors reference the mapped GGUF. */
bool gpu_graph_alloc_raw_cap(
        pulsar_gpu_graph *g,
        const pulsar_weights     *weights,
        const pulsar_layer_weights *layer,
        uint32_t                raw_cap,
        uint32_t                ctx_size,
        uint32_t                prefill_cap,
        bool                    enable_spec) {
    memset(g, 0, sizeof(*g));
    gpu_graph_dims dz;
    gpu_graph_compute_dims(&dz, weights, layer, raw_cap, ctx_size, prefill_cap);
    raw_cap = dz.raw_cap;
    ctx_size = dz.ctx_size;
    prefill_cap = dz.prefill_cap;
    g->raw_cap = dz.raw_cap;
    g->raw_window = dz.raw_window;
    g->prefill_cap = dz.prefill_cap;

    /* PULSAR_ATTN_PACK validation lives up here: no allocations have happened
     * yet, so these early returns need no cleanup. */
    if (getenv("PULSAR_ATTN_MX") != NULL) {
        /* Removed 2026-07-10: superseded by PULSAR_ATTN_PACK (bit-exact, smaller
         * rows; MX re-quantized the rope dims and cost drafter acceptance).
         * Refuse loudly instead of silently running a different format. */
        fprintf(stderr,
                "pulsar: PULSAR_ATTN_MX has been removed (superseded by PULSAR_ATTN_PACK); "
                "unset PULSAR_ATTN_MX\n");
        return false;
    }
    if (PULSAR_N_ROT != 64u || ((PULSAR_N_HEAD_DIM - PULSAR_N_ROT) % 64u) != 0u) {
        fprintf(stderr,
                "pulsar: the packed comp cache requires n_rot 64 and 64-aligned nope dims "
                "(head_dim %u / n_rot %u)\n",
                (unsigned)PULSAR_N_HEAD_DIM, (unsigned)PULSAR_N_ROT);
        return false;
    }
    g->comp_cap = dz.comp_cap;
    g->attn_comp_stage_cap = dz.attn_comp_stage_cap;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
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
        : pulsar_gpu_should_use_managed_kv_cache(kv_cache_bytes, context_bytes) != 0;
    if (managed_override >= 0) {
        fprintf(stderr,
                "pulsar: PULSAR_KV_MANAGED override: KV caches use %s allocation "
                "(measurement flag; policy would have chosen %s)\n",
                managed_kv_cache ? "MANAGED (cudaMallocManaged)"
                                 : "DEVICE (cudaMalloc)",
                pulsar_gpu_should_use_managed_kv_cache(kv_cache_bytes,
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
                "pulsar: CUDA using managed KV cache for ctx=%u "
                "(kv cache %.2f GiB, context buffers %.2f GiB); "
                "this may degrade performance but is needed for very large contexts\n",
                ctx_size,
                (double)kv_cache_bytes / 1073741824.0,
                (double)context_bytes / 1073741824.0);
    }

    /* ROW GEOMETRY AGREEMENT, checked once, before anything is sized by it.
     *
     * The packed KV row and the indexer FP4 row are each defined twice -- once
     * here and once in the backend -- because head_dim is a runtime shape that
     * neither side can name across the seam.  Until now the two were kept in
     * step by a comment.  Every cache stride, every session-payload span and
     * every attention read is derived from these numbers, so a divergence is not
     * a subtle bug: it is out-of-bounds reads at a wrong stride, which is
     * exactly how the f32/packed mixup produced NaNs from a clean compile.
     *
     * Ask the backend and refuse to start if it disagrees.  This is cheap (two
     * calls per graph) and it converts a silent divergence into a startup
     * failure that names both numbers. */
    {
        const uint64_t eng_attn = (uint64_t)PULSAR_ENGINE_ATTN_PACK_ROWBYTES;
        const uint64_t gpu_attn = pulsar_gpu_attn_pack_rowbytes(PULSAR_N_HEAD_DIM);
        const uint64_t eng_idx  = (uint64_t)PULSAR_ENGINE_IDXFP4_ROWBYTES;
        const uint64_t gpu_idx  = pulsar_gpu_mxkv_fp4_rowbytes(PULSAR_N_INDEXER_HEAD_DIM);
        if (eng_attn != gpu_attn || eng_idx != gpu_idx) {
            fprintf(stderr,
                    "pulsar: KV row geometry disagrees across the backend seam -- "
                    "attn engine=%llu gpu=%llu (head_dim %u), "
                    "index engine=%llu gpu=%llu (head_dim %u). "
                    "PULSAR_ENGINE_ATTN_PACK_ROWBYTES / PULSAR_ENGINE_IDXFP4_ROWBYTES "
                    "must match PULSAR_ATTN_PACK_ROWBYTES / PULSAR_MXKV_FP4_ROWBYTES.\n",
                    (unsigned long long)eng_attn, (unsigned long long)gpu_attn,
                    (unsigned)PULSAR_N_HEAD_DIM,
                    (unsigned long long)eng_idx, (unsigned long long)gpu_idx,
                    (unsigned)PULSAR_N_INDEXER_HEAD_DIM);
            gpu_graph_free(g);
            return false;
        }
    }

    /* One KV row format since the L111 unification.  Name it once in the log
     * (the arm a measurement ran is a fact the log must witness), and refuse
     * a set PULSAR_KV4: the switch is DEAD -- an env that once chose formats
     * and now silently does nothing is how an A/B grades the wrong thing
     * (ATTN_MX removal set the pattern). */
    {
        if (getenv("PULSAR_KV4")) {
            fprintf(stderr,
                    "pulsar: PULSAR_KV4 is set but the switch was REMOVED (L111 "
                    "unification 2026-08-27: every KV buffer is the 384 B NVFP4 "
                    "row); unset it -- refusing to start\n");
            gpu_graph_free(g);
            return false;
        }
        fprintf(stderr, "pulsar: KV rows = NVFP4 (%llu B/row, unified: raw ring + "
                        "comp pool + drafter + chunk)\n",
                (unsigned long long)pulsar_gpu_attn_pack_rowbytes(PULSAR_N_HEAD_DIM));
    }

    /* Tier-2 bank pool: allocate the per-bank slabs first; the per-layer
     * cache pointers below then become bank-0 views instead of owning
     * allocations, and all single-session code runs unmodified. */
    const uint32_t n_banks = gpu_graph_bank_pool_n();
    if (n_banks >= 2u &&
        !gpu_graph_bank_slabs_alloc(g, n_banks, managed_kv_cache, &dz, enable_spec)) {
        gpu_graph_free(g);
        return false;
    }
    const bool banked = g->banks.n_banks != 0;

    g->cur_hc = pulsar_gpu_tensor_alloc_elt(hc_dim, PULSAR_HC_ELT_SIZE, PULSAR_HC_ELT_FMT);   /* HC residual carrier (BF16); task #62 */
    g->hc_split = pulsar_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_post = pulsar_gpu_tensor_view(g->hc_split,
                                       (uint64_t)PULSAR_N_HC * sizeof(float),
                                       (uint64_t)PULSAR_N_HC * sizeof(float));
    g->hc_comb = pulsar_gpu_tensor_view(g->hc_split,
                                       2ull * PULSAR_N_HC * sizeof(float),
                                       (uint64_t)PULSAR_N_HC * PULSAR_N_HC * sizeof(float));
    g->attn_norm = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_N_EMBD * sizeof(float));
    g->kv = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_N_HEAD_DIM * sizeof(float));
    bool state_init_ok = true;
    /* PULSAR_ATTN_PACK rows -- see the bank sizing above. */
    const uint64_t raw_row_bytes_pack = PULSAR_ENGINE_ATTN_PACK_ROWBYTES;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        g->layer_raw_cache[il] = banked
            ? pulsar_gpu_tensor_view(g->banks.raw[il], 0, g->banks.raw_bank_bytes)
            : gpu_graph_alloc_kv_cache_tensor(
                    managed_kv_cache,
                    (uint64_t)raw_cap * raw_row_bytes_pack);
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t attn_width = (uint64_t)coff * PULSAR_N_HEAD_DIM;
            const uint64_t attn_rows = (uint64_t)coff * ratio;
            const uint64_t comp_row_bytes = gpu_graph_attn_comp_cache_row_bytes();
            if (banked) {
                g->layer_attn_comp_cache[il] = pulsar_gpu_tensor_view(
                        g->banks.comp[il][0], 0, g->banks.comp_bank_bytes[il]);
                g->layer_attn_state_kv[il] = pulsar_gpu_tensor_view(
                        g->banks.askv[il], 0, g->banks.astate_bank_bytes[il]);
                g->layer_attn_state_score[il] = pulsar_gpu_tensor_view(
                        g->banks.assc[il], 0, g->banks.astate_bank_bytes[il]);
            } else {
                g->layer_attn_comp_cache[il] = gpu_graph_alloc_kv_cache_tensor(
                        managed_kv_cache,
                        (uint64_t)g->layer_comp_cap[il] * comp_row_bytes);
                g->layer_attn_state_kv[il] = pulsar_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->layer_attn_state_score[il] = pulsar_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            }
            if (enable_spec) {
                if (banked) {
                    g->spec_attn_state_kv[il] = pulsar_gpu_tensor_view(
                            g->banks.spec_askv[il], 0, g->banks.astate_bank_bytes[il]);
                    g->spec_attn_state_score[il] = pulsar_gpu_tensor_view(
                            g->banks.spec_assc[il], 0, g->banks.astate_bank_bytes[il]);
                } else {
                    g->spec_attn_state_kv[il] = pulsar_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                    g->spec_attn_state_score[il] = pulsar_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                }
            }
            /* Banked mode primes every bank's state lanes at slab alloc. */
            if (!banked && g->layer_attn_state_kv[il]) {
                state_init_ok = state_init_ok &&
                                gpu_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows);
            }
            if (!banked && g->layer_attn_state_score[il]) {
                state_init_ok = state_init_ok &&
                                gpu_tensor_fill_f32(g->layer_attn_state_score[il], PULSAR_NEG_INF, attn_width * attn_rows);
            }

            if (ratio == 4) {
                const uint64_t index_width = (uint64_t)coff * PULSAR_N_INDEXER_HEAD_DIM;
                const uint64_t index_rows = (uint64_t)coff * ratio;
                const uint64_t index_row_bytes = PULSAR_ENGINE_IDXFP4_ROWBYTES;
                if (banked) {
                    g->layer_index_comp_cache[il] = pulsar_gpu_tensor_view(
                            g->banks.index[il][0], 0, g->banks.index_bank_bytes[il]);
                    g->layer_index_state_kv[il] = pulsar_gpu_tensor_view(
                            g->banks.iskv[il], 0, g->banks.istate_bank_bytes[il]);
                    g->layer_index_state_score[il] = pulsar_gpu_tensor_view(
                            g->banks.issc[il], 0, g->banks.istate_bank_bytes[il]);
                } else {
                    g->layer_index_comp_cache[il] = gpu_graph_alloc_kv_cache_tensor(
                            managed_kv_cache,
                            (uint64_t)g->layer_comp_cap[il] * index_row_bytes);
                    g->layer_index_state_kv[il] = pulsar_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->layer_index_state_score[il] = pulsar_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                }
                if (enable_spec) {
                    if (banked) {
                        g->spec_index_state_kv[il] = pulsar_gpu_tensor_view(
                                g->banks.spec_iskv[il], 0, g->banks.istate_bank_bytes[il]);
                        g->spec_index_state_score[il] = pulsar_gpu_tensor_view(
                                g->banks.spec_issc[il], 0, g->banks.istate_bank_bytes[il]);
                    } else {
                        g->spec_index_state_kv[il] = pulsar_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                        g->spec_index_state_score[il] = pulsar_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    }
                }
                if (!banked && g->layer_index_state_kv[il]) {
                    state_init_ok = state_init_ok &&
                                    gpu_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows);
                }
                if (!banked && g->layer_index_state_score[il]) {
                    state_init_ok = state_init_ok &&
                                    gpu_tensor_fill_f32(g->layer_index_state_score[il], PULSAR_NEG_INF, index_width * index_rows);
                }
                /* L120 value-half: committed-projection rings (32 x width-256
                 * rows), attn + indexer.  Banked: bank-0 views, repointed
                 * with the state views. */
                const uint64_t pring_bytes = (uint64_t)PULSAR_REWIND_RING_DEPTH * attn_width * sizeof(float);
                if (banked) {
                    g->layer_attn_proj_kv[il] = pulsar_gpu_tensor_view(
                            g->banks.apkv[il], 0, g->banks.pring_bank_bytes);
                    g->layer_attn_proj_sc[il] = pulsar_gpu_tensor_view(
                            g->banks.apsc[il], 0, g->banks.pring_bank_bytes);
                    g->layer_index_proj_kv[il] = pulsar_gpu_tensor_view(
                            g->banks.ipkv[il], 0, g->banks.pring_bank_bytes);
                    g->layer_index_proj_sc[il] = pulsar_gpu_tensor_view(
                            g->banks.ipsc[il], 0, g->banks.pring_bank_bytes);
                } else {
                    g->layer_attn_proj_kv[il] = pulsar_gpu_tensor_alloc(pring_bytes);
                    g->layer_attn_proj_sc[il] = pulsar_gpu_tensor_alloc(pring_bytes);
                    g->layer_index_proj_kv[il] = pulsar_gpu_tensor_alloc(pring_bytes);
                    g->layer_index_proj_sc[il] = pulsar_gpu_tensor_alloc(pring_bytes);
                }
                state_init_ok = state_init_ok &&
                                g->layer_attn_proj_kv[il] && g->layer_attn_proj_sc[il] &&
                                g->layer_index_proj_kv[il] && g->layer_index_proj_sc[il];
            }
            if (ratio == 128u) {
                /* L124: undo lanes (32 x head_dim f32 rows, kv + sc). */
                const uint64_t rulane_bytes = 32ull * PULSAR_N_HEAD_DIM * sizeof(float);
                if (banked) {
                    g->layer_r128_undo_kv[il] = pulsar_gpu_tensor_view(
                            g->banks.rukv[il], 0, g->banks.rulane_bank_bytes);
                    g->layer_r128_undo_sc[il] = pulsar_gpu_tensor_view(
                            g->banks.rusc[il], 0, g->banks.rulane_bank_bytes);
                } else {
                    g->layer_r128_undo_kv[il] = pulsar_gpu_tensor_alloc(rulane_bytes);
                    g->layer_r128_undo_sc[il] = pulsar_gpu_tensor_alloc(rulane_bytes);
                }
                state_init_ok = state_init_ok &&
                                g->layer_r128_undo_kv[il] && g->layer_r128_undo_sc[il];
            }
        }
    }
    /* f32 staging: the compressor writes real f32 rows here, then the commit
     * step packs them to the persistent packed comp cache. */
    g->attn_comp_stage = pulsar_gpu_tensor_alloc((uint64_t)g->attn_comp_stage_cap *
                                              PULSAR_N_HEAD_DIM * sizeof(float));
    {
        if (PULSAR_N_INDEXER_HEAD_DIM != 128u) {
            /* The packed loader and QAT+pack kernels hard-code the 68-byte
             * head_dim-128 row; fail loud here instead of deep in a launch. */
            fprintf(stderr,
                    "pulsar: PULSAR_IDX_FP4 requires indexer head_dim 128 (model has %u)\n",
                    PULSAR_N_INDEXER_HEAD_DIM);
            gpu_graph_free(g);
            return false;
        }
        /* f32 emit/repack staging for the packed indexer cache: comp-cap rows so
         * the compressor writers can keep their absolute row indices. */
        g->idx_comp_stage = pulsar_gpu_tensor_alloc((uint64_t)g->comp_cap *
                                                 PULSAR_N_INDEXER_HEAD_DIM * sizeof(float));
    }
    /* PULSAR_PREFILL_SLICE: these two are the only ctx-scaling f32 work buffers
     * with a prefill_cap token dimension; under slicing they only ever hold
     * one <=slice-token span at a time. */
    const uint64_t score_rows = (gpu_graph_prefill_slice() != 0u &&
                                 (uint64_t)gpu_graph_prefill_slice() < pc)
        ? (uint64_t)gpu_graph_prefill_slice() : pc;
    g->indexer_scores = pulsar_gpu_tensor_alloc((uint64_t)g->comp_cap * score_rows * sizeof(float));
    g->comp_selected = pulsar_gpu_tensor_alloc((uint64_t)(PULSAR_N_INDEXER_TOP_K ? PULSAR_N_INDEXER_TOP_K : 1u) *
                                              pc * sizeof(uint32_t));
    g->ffn_norm = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_N_EMBD * sizeof(float));
    g->output_pre = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_N_HC * sizeof(float));
    g->output_weights = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_N_HC * sizeof(float));
    g->output_embd = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_N_EMBD * sizeof(float));
    g->output_norm = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_N_EMBD * sizeof(float));
    g->logits = pulsar_gpu_tensor_alloc(vocab_dim * sizeof(float));
    g->prefill_tokens = pulsar_gpu_tensor_alloc(pc * sizeof(int32_t));
    /* Shared multi-row logits slab (16 rows).  Unconditional, NOT gated on
     * speculation: every batched multi-row output head writes its rows here —
     * the DSpark draft/verify passes, gpu_graph_verify_suffix_tops, and the
     * Tier-2 batched multi-session decode driver.  It used to be allocated
     * only by gpu_graph_init_dspark_target (session create, dspark_ready
     * only), which left the multiseq driver rejecting every step whenever
     * speculation was off. */
    g->spec_logits = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_SPEC_LOGITS_ROWS * PULSAR_N_VOCAB * sizeof(float));
    g->batch_cur_hc = pulsar_gpu_tensor_alloc_elt(pc * hc_dim, PULSAR_HC_ELT_SIZE, PULSAR_HC_ELT_FMT);   /* HC residual carrier */
    g->batch_next_hc = pulsar_gpu_tensor_alloc_elt(pc * hc_dim, PULSAR_HC_ELT_SIZE, PULSAR_HC_ELT_FMT);   /* HC residual carrier */
    g->batch_flat_hc = pulsar_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_hc_mix = pulsar_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_hc_split = pulsar_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    /* Dump-only carrier (L090.1): its WRITE has been dump-gated NULL since the
     * dead-store pass, but the 64 MiB allocation never followed.  Confirmed by
     * the D2 hand census: zero non-debug readers.  Allocate it only when a
     * dump can actually want it, and keep the budget line in step -- the
     * session ledger asserts est == actual. */
    g->batch_attn_cur = gpu_graph_f32_store_observed_any()
            ? pulsar_gpu_tensor_alloc(pc * PULSAR_N_EMBD * sizeof(float)) : NULL;
    g->batch_attn_norm = pulsar_gpu_tensor_alloc(pc * PULSAR_N_EMBD * sizeof(float));
    g->batch_qr = pulsar_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_qr_norm = pulsar_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_q = pulsar_gpu_tensor_alloc_elt(pc * q_dim, PULSAR_Q_ELT_SIZE, PULSAR_Q_ELT_FMT);
    g->batch_kv_raw = pulsar_gpu_tensor_alloc(pc * PULSAR_N_HEAD_DIM * sizeof(float));
    g->batch_kv = pulsar_gpu_tensor_alloc(pc * PULSAR_N_HEAD_DIM * sizeof(float));
    g->batch_kv_pack = pulsar_gpu_tensor_alloc(pc * PULSAR_ENGINE_ATTN_PACK_ROWBYTES);
    g->batch_comp_kv = pulsar_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_comp_sc = pulsar_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_indexer_q = pulsar_gpu_tensor_alloc(pc * indexer_q_dim * sizeof(float));
    g->batch_indexer_qp = pulsar_gpu_tensor_alloc(pc * (uint64_t)PULSAR_N_INDEXER_HEAD *
                                                  PULSAR_ENGINE_IDXFP4_ROWBYTES);
    g->batch_indexer_weights = pulsar_gpu_tensor_alloc(pc * PULSAR_N_INDEXER_HEAD * sizeof(float));
    /* The stored width is PULSAR_HEADS_ELT_SIZE, not sizeof(float): the alloc,
     * the byte budget and the kernels must agree on ONE authority or the flip
     * to bf16 silently under-allocates. */
    g->batch_heads = pulsar_gpu_tensor_alloc_elt(pc * q_dim, PULSAR_HEADS_ELT_SIZE, PULSAR_HEADS_ELT_FMT);
    g->batch_attn_low = pulsar_gpu_tensor_alloc(pc * low_dim * sizeof(float));
    g->batch_attn_out = pulsar_gpu_tensor_alloc(pc * PULSAR_N_EMBD * sizeof(float));
    g->batch_after_attn_hc = pulsar_gpu_tensor_alloc_elt(pc * hc_dim, PULSAR_HC_ELT_SIZE, PULSAR_HC_ELT_FMT);   /* HC residual carrier */
    g->batch_ffn_cur = pulsar_gpu_tensor_alloc(pc * PULSAR_N_EMBD * sizeof(float));
    g->batch_ffn_norm = pulsar_gpu_tensor_alloc(pc * PULSAR_N_EMBD * sizeof(float));
    /* L033 increment 2: gate/up staging is F16.  Sole producer is the mxfp8
     * GEMM (writes __half whenever the out tensor's esz is 2, on every arm —
     * the L045 machinery), sole consumer is the swiglu fold's f16-load
     * instantiation; decode's fused shared-expert path uses its own f32
     * scratch and never sees these.  NOT bit-exact (swiglu's inputs are
     * f16-rounded): graded by cuda-reference-gate, not the byte gate. */
    g->batch_shared_gate = pulsar_gpu_tensor_alloc_elt(pc * shared_dim, PULSAR_SHARED_ACT_ELT_SIZE, PULSAR_SHARED_ACT_ELT_FMT);
    g->batch_shared_up = pulsar_gpu_tensor_alloc_elt(pc * shared_dim, PULSAR_SHARED_ACT_ELT_SIZE, PULSAR_SHARED_ACT_ELT_FMT);
    g->batch_shared_mid = pulsar_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_out = pulsar_gpu_tensor_alloc(pc * PULSAR_N_EMBD * sizeof(float));
    g->batch_router_logits = pulsar_gpu_tensor_alloc(pc * PULSAR_N_EXPERT * sizeof(float));
    g->batch_router_probs = pulsar_gpu_tensor_alloc(pc * PULSAR_N_EXPERT * sizeof(float));
    g->batch_router_selected = pulsar_gpu_tensor_alloc(pc * PULSAR_N_EXPERT_USED * sizeof(int));
    g->batch_router_weights = pulsar_gpu_tensor_alloc(pc * PULSAR_N_EXPERT_USED * sizeof(float));
    g->batch_routed_up = pulsar_gpu_tensor_alloc(pc * PULSAR_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_mid = pulsar_gpu_tensor_alloc(pc * PULSAR_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_down = pulsar_gpu_tensor_alloc(pc * PULSAR_N_EXPERT_USED * PULSAR_N_EMBD * sizeof(float));
    g->batch_routed_out = pulsar_gpu_tensor_alloc(pc * PULSAR_N_EMBD * sizeof(float));

    bool layer_cache_ok = true;
    for (uint32_t il = 0; layer_cache_ok && il < PULSAR_N_LAYER; il++) {
        layer_cache_ok = g->layer_raw_cache[il] != NULL;
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
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
                             g->idx_comp_stage != NULL &&
                             g->layer_index_state_kv[il] != NULL &&
                             g->layer_index_state_score[il] != NULL &&
                             (!enable_spec ||
                              (g->spec_index_state_kv[il] != NULL &&
                               g->spec_index_state_score[il] != NULL));
        }
    }

    const bool ok = state_init_ok && layer_cache_ok &&
                    g->cur_hc && g->hc_split &&
                    g->hc_post && g->hc_comb &&
                    g->attn_norm && g->kv &&
                    g->attn_comp_stage &&
                    g->indexer_scores &&
                    g->comp_selected &&
                    g->ffn_norm &&
                    g->output_pre && g->output_weights && g->output_embd &&
                    g->output_norm && g->logits &&
                    g->prefill_tokens && g->spec_logits &&
                    g->batch_cur_hc && g->batch_next_hc && g->batch_flat_hc &&
                    g->batch_hc_mix && g->batch_hc_split &&
                    (g->batch_attn_cur || !gpu_graph_f32_store_observed_any()) &&
                    g->batch_attn_norm &&
                    g->batch_qr && g->batch_qr_norm && g->batch_q &&
                    g->batch_kv_raw && g->batch_kv &&
                    g->batch_comp_kv && g->batch_comp_sc &&
                    g->batch_indexer_q && g->batch_indexer_qp && g->batch_indexer_weights &&
                    g->batch_heads && g->batch_attn_low && g->batch_attn_out &&
                    g->batch_after_attn_hc &&
                    g->batch_ffn_cur && g->batch_ffn_norm &&
                    g->batch_shared_gate && g->batch_shared_up &&
                    g->batch_shared_mid && g->batch_shared_out &&
                    g->batch_router_logits && g->batch_router_probs &&
                    g->batch_router_selected && g->batch_router_weights &&
                    g->batch_routed_up &&
                    g->batch_routed_mid && g->batch_routed_down &&
                    g->batch_routed_out;
    if (!ok) gpu_graph_free(g);
    return ok;
}

bool gpu_graph_init_dspark_target(pulsar_gpu_graph *g, const uint32_t target_layer_ids[3]) {
    bool ok = true;
    for (int i = 0; i < 3; i++) {
        g->dspark_target_layer_ids[i] = target_layer_ids[i];
        g->dspark_target_h[i] = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_N_EMBD * sizeof(float));
        /* Fused-loop batch capture: one anchor hidden per verify-batch position.
         * Sized from the slab authority (L117: a 32-row sweep reduces up to
         * n_tokens rows into this; the old literal 17 would overflow). */
        g->dspark_target_h_batch[i] = pulsar_gpu_tensor_alloc(
            (uint64_t)(PULSAR_SPEC_LOGITS_ROWS + 1u) * PULSAR_N_EMBD * sizeof(float));
        /* Option F: bank the drafter ring when the pool is enabled — one
         * bank-major slab, dspark_raw_cache[i] becomes bank 0's view (repoint
         * swaps it). */
        if (g->banks.n_banks != 0) {
            g->banks.dspark_raw_bank_bytes =
                (uint64_t)PULSAR_DSPARK_DRAFT_WINDOW * PULSAR_ENGINE_ATTN_PACK_ROWBYTES;
            g->banks.dspark_raw[i] = pulsar_gpu_tensor_alloc(
                (uint64_t)g->banks.n_banks * g->banks.dspark_raw_bank_bytes);
            g->dspark_raw_cache[i] = g->banks.dspark_raw[i]
                ? pulsar_gpu_tensor_view(g->banks.dspark_raw[i], 0,
                                      g->banks.dspark_raw_bank_bytes)
                : NULL;
        } else {
            g->dspark_raw_cache[i] = pulsar_gpu_tensor_alloc(
                (uint64_t)PULSAR_DSPARK_DRAFT_WINDOW * PULSAR_ENGINE_ATTN_PACK_ROWBYTES);
        }
        g->dspark_n_raw[i] = 0;
        ok = ok && g->dspark_target_h[i] && g->dspark_target_h_batch[i] &&
             g->dspark_raw_cache[i];
    }
    g->dspark_capture_batch_n = 0;
    g->dspark_main_x = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_N_EMBD * sizeof(float));
    ok = ok && g->dspark_main_x;
    /* Bulk prefill capture buffers: always allocated when the drafter is
     * loaded (~200 MB at prefill_cap 4096) -- the prompt-window seeding
     * reduces every chunk through them; PULSAR_DSPARK_PREFILL_DUMP additionally
     * streams them to disk for retraining data. */
    g->dspark_bulk_n = 0;
    for (int i = 0; i < 3; i++) {
        g->dspark_bulk_h[i] = pulsar_gpu_tensor_alloc(
            (uint64_t)g->prefill_cap * PULSAR_N_EMBD * sizeof(float));
        ok = ok && g->dspark_bulk_h[i];
    }
    /* plan-92 P0 teacher dump: allocated only when the collection mode is on
     * (env read once here, not per token -- [[no-hot-path-flags]]). ~1.1 MiB
     * at prefill_cap 4096. */
    {
        const char *dd = getenv("PULSAR_DISTILL_DUMP");
        if (dd && dd[0]) {
            g->distill_top_ids = pulsar_gpu_tensor_alloc(
                (uint64_t)g->prefill_cap * 64 * sizeof(int32_t));
            g->distill_top_vals = pulsar_gpu_tensor_alloc(
                (uint64_t)g->prefill_cap * 64 * sizeof(uint16_t));
            g->distill_tail_lse = pulsar_gpu_tensor_alloc(
                (uint64_t)g->prefill_cap * sizeof(uint16_t));
            g->distill_inexact = pulsar_gpu_tensor_alloc(sizeof(int32_t));
            ok = ok && g->distill_top_ids && g->distill_top_vals &&
                 g->distill_tail_lse && g->distill_inexact;
            if (ok) {
                const int32_t z = 0;
                ok = pulsar_gpu_tensor_write(g->distill_inexact, 0, &z, sizeof(z)) != 0;
            }
        }
    }
    /* Prompt-window ring: last <=128 prompt positions' anchor hiddens. */
    g->dspark_prompt_n = 0;
    for (int i = 0; i < 3; i++) {
        if (g->banks.n_banks != 0) {
            g->banks.dspark_prompt_bank_bytes =
                (uint64_t)PULSAR_DSPARK_DRAFT_WINDOW * PULSAR_N_EMBD * sizeof(float);
            g->banks.dspark_prompt[i] = pulsar_gpu_tensor_alloc(
                (uint64_t)g->banks.n_banks * g->banks.dspark_prompt_bank_bytes);
            g->dspark_prompt_h[i] = g->banks.dspark_prompt[i]
                ? pulsar_gpu_tensor_view(g->banks.dspark_prompt[i], 0,
                                      g->banks.dspark_prompt_bank_bytes)
                : NULL;
        } else {
            g->dspark_prompt_h[i] = pulsar_gpu_tensor_alloc(
                (uint64_t)PULSAR_DSPARK_DRAFT_WINDOW * PULSAR_N_EMBD * sizeof(float));
        }
        ok = ok && g->dspark_prompt_h[i];
    }
    /* Stage-B no-replay rollback: per-position compressor projection saves for
     * every compressed layer (attn comp_width <= 2*PULSAR_N_HEAD_DIM; indexer width
     * = 2*PULSAR_N_INDEXER_HEAD_DIM) + one emit-sink scratch row. ~8 MB total. */
    {
        const uint64_t attn_w = 2ull * PULSAR_N_HEAD_DIM;
        const uint64_t idx_w = 2ull * PULSAR_N_INDEXER_HEAD_DIM;
        for (uint32_t il = 0; il < PULSAR_N_LAYER && ok; il++) {
            const uint32_t ratio = pulsar_layer_compress_ratio(il);
            g->spec_comp_kv_save[il] = NULL;
            g->spec_comp_sc_save[il] = NULL;
            g->spec_icomp_kv_save[il] = NULL;
            g->spec_icomp_sc_save[il] = NULL;
            if (ratio == 0) continue;
            g->spec_comp_kv_save[il] = pulsar_gpu_tensor_alloc((PULSAR_SPEC_LOGITS_ROWS + 1ull) * attn_w * sizeof(float));
            g->spec_comp_sc_save[il] = pulsar_gpu_tensor_alloc((PULSAR_SPEC_LOGITS_ROWS + 1ull) * attn_w * sizeof(float));
            ok = ok && g->spec_comp_kv_save[il] && g->spec_comp_sc_save[il];
            if (ratio == 4) {
                g->spec_icomp_kv_save[il] = pulsar_gpu_tensor_alloc((PULSAR_SPEC_LOGITS_ROWS + 1ull) * idx_w * sizeof(float));
                g->spec_icomp_sc_save[il] = pulsar_gpu_tensor_alloc((PULSAR_SPEC_LOGITS_ROWS + 1ull) * idx_w * sizeof(float));
                ok = ok && g->spec_icomp_kv_save[il] && g->spec_icomp_sc_save[il];
            }
        }
        /* Shared emit sink for BOTH compressors (L090.2): sized by the attention
         * head dim but also handed to the indexer roll-forward.  Safe only
         * while the indexer row fits; both are runtime shape values, so say it
         * loudly instead of assuming it quietly. */
        if (PULSAR_N_INDEXER_HEAD_DIM > PULSAR_N_HEAD_DIM) {
            fprintf(stderr, "pulsar: spec_comp_scratch_row sized for head_dim=%u but "
                            "indexer head_dim=%u exceeds it -- refusing graph alloc\n",
                    (unsigned)PULSAR_N_HEAD_DIM, (unsigned)PULSAR_N_INDEXER_HEAD_DIM);
            return false;
        }
        g->spec_comp_scratch_row = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_N_HEAD_DIM * sizeof(float));
        ok = ok && g->spec_comp_scratch_row;
        g->spec_comp_save_n = 0;
        g->dspark_concat = pulsar_gpu_tensor_alloc(3ull * PULSAR_N_EMBD * sizeof(float));
        g->dspark_proj_out = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_N_EMBD * sizeof(float));
        g->dspark_seed_kv = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_N_HEAD_DIM * sizeof(float));
        g->dspark_seed_norm = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_N_HEAD_DIM * sizeof(float));
        g->dspark_seed_rot = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_N_HEAD_DIM * sizeof(float));
        /* L150: the drafter scratch holds every bank of a batched redraft
         * (PULSAR_DSPARK_BANKS_MAX banks of refined logits / chain ids, a
         * confidence row per drafted row); the single-bank paths use slot 0. */
        g->dspark_markov_logits = pulsar_gpu_tensor_alloc(
            (uint64_t)PULSAR_DSPARK_BANKS_MAX * PULSAR_N_VOCAB * sizeof(float));
        g->dspark_conf_scores = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_SPEC_LOGITS_ROWS * sizeof(float));
        g->dspark_conf_tokens = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_SPEC_LOGITS_ROWS * sizeof(int32_t));
        g->dspark_bank_meta = pulsar_gpu_tensor_alloc(2ull * PULSAR_DSPARK_BANKS_MAX * sizeof(int32_t));
        ok = ok && g->dspark_bank_meta;
        g->dspark_embed_tokens = pulsar_gpu_tensor_alloc(16ull * sizeof(int32_t));
        g->dspark_refined_ids = pulsar_gpu_tensor_alloc(17ull * PULSAR_DSPARK_BANKS_MAX * sizeof(int32_t));
        g->dspark_refined2_ids = pulsar_gpu_tensor_alloc(17ull * PULSAR_DSPARK_BANKS_MAX * sizeof(int32_t));
        g->dspark_prefilter_sel = pulsar_gpu_tensor_alloc(
            16ull * PULSAR_DSPARK_PREFILTER_ROW_I32 * sizeof(int32_t));
        g->dspark_row_meta = pulsar_gpu_tensor_alloc(7ull * PULSAR_SPEC_LOGITS_ROWS * sizeof(int32_t));
        ok = ok && g->dspark_row_meta;
        g->spec_compact_host = (int32_t *)xmalloc(
            (size_t)PULSAR_SPEC_LOGITS_ROWS * PULSAR_DSPARK_PREFILTER_ROW_I32 * sizeof(int32_t));
        g->spec_compact_rows = 0;
        g->spec_compact_armed = false;
        g->spec_compact_acc_n = 0;
        ok = ok && g->dspark_concat && g->dspark_proj_out && g->dspark_seed_kv &&
             g->dspark_seed_norm && g->dspark_seed_rot && g->dspark_markov_logits &&
             g->dspark_conf_scores && g->dspark_conf_tokens && g->dspark_embed_tokens &&
             g->dspark_refined_ids && g->dspark_refined2_ids && g->dspark_prefilter_sel;
    }
    /* spec_logits is NOT allocated here: it is the shared multi-row logits
     * slab, allocated unconditionally by gpu_graph_alloc_raw_cap (the batched
     * decode driver needs it with speculation disabled). */
    return ok;
}


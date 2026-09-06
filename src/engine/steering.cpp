#include "pulsar_engine_internal.h"



bool gpu_graph_directional_steering_attn_enabled(const pulsar_gpu_graph *g) {
    return g && g->directional_steering_dirs && g->directional_steering_attn_scale != 0.0f;
}



bool gpu_graph_directional_steering_ffn_enabled(const pulsar_gpu_graph *g) {
    return g && g->directional_steering_dirs && g->directional_steering_ffn_scale != 0.0f;
}



static bool gpu_graph_apply_directional_steering(
        pulsar_gpu_graph  *g,
        pulsar_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows,
        float             scale) {
    if (!g || !g->directional_steering_dirs || scale == 0.0f) return true;
    return pulsar_gpu_directional_steering_project_tensor(x,
                                            g->directional_steering_dirs,
                                            il,
                                            PULSAR_N_EMBD,
                                            rows,
                                            scale) != 0;
}



bool gpu_graph_apply_directional_steering_attn(
        pulsar_gpu_graph  *g,
        pulsar_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return gpu_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_attn_scale : 0.0f);
}



bool gpu_graph_apply_directional_steering_ffn(
        pulsar_gpu_graph  *g,
        pulsar_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return gpu_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_ffn_scale : 0.0f);
}



/* The comp + index caches of one bank at ctx_size, in their stored row
 * formats (packed attn comp, MXFP4 indexer), per-layer capacity from
 * gpu_graph_comp_cap -- the same capacity gpu_graph_compute_dims hands the
 * allocator.  Three readers: the KV-policy sizing below, the overcommit
 * split (gpu_graph_demand_paged_bytes_per_bank) and the boot-line estimate
 * (pulsar_context_memory_estimate). */
uint64_t gpu_graph_comp_index_bytes_for_context(uint32_t ctx_size) {
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = PULSAR_ENGINE_IDXFP4_ROWBYTES;
    uint64_t bytes = 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t comp_cap = gpu_graph_comp_cap(ctx_size, ratio);
        bytes += comp_cap * attn_row;
        if (ratio == 4) bytes += comp_cap * idx_row;
    }
    return bytes;
}

uint64_t gpu_graph_raw_ring_bytes_for_context(uint32_t raw_cap) {
    /* Stored formats, not f32 upper bounds: an f32-priced size overshoots ~3x
     * and trips the managed-KV (demand-paged) policy at the 512k+ contexts
     * where performance matters most.  The raw ring is PULSAR_ATTN_PACK rows,
     * as gpu_graph_bank_slabs_alloc sizes it (raw_bank_bytes). */
    return (uint64_t)PULSAR_N_LAYER * raw_cap * PULSAR_ENGINE_ATTN_PACK_ROWBYTES;
}

uint32_t gpu_graph_comp_cap_max(uint32_t ctx_size) {
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    return gpu_graph_comp_cap(ctx_size, min_ratio);
}



uint64_t gpu_graph_context_bytes_for_kv_policy(
        uint32_t  ctx_size,
        uint32_t  raw_cap,
        uint32_t  prefill_cap,
        uint64_t *kv_cache_bytes_out) {
    const uint64_t comp_cap = gpu_graph_comp_cap_max(ctx_size);
    const uint64_t kv_cache_bytes = gpu_graph_raw_ring_bytes_for_context(raw_cap) +
                                    gpu_graph_comp_index_bytes_for_context(ctx_size);
    if (kv_cache_bytes_out) *kv_cache_bytes_out = kv_cache_bytes;
    /* indexer_scores token rows shrink to the slice size under
     * PULSAR_PREFILL_SLICE (see gpu_graph_prefill_slice / the gpu_diag alloc). */
    uint64_t score_rows = (uint64_t)prefill_cap;
    if (gpu_graph_prefill_slice() != 0u && (uint64_t)gpu_graph_prefill_slice() < score_rows) {
        score_rows = (uint64_t)gpu_graph_prefill_slice();
    }
    uint64_t bytes = kv_cache_bytes +
                     comp_cap * score_rows * sizeof(float);  /* one indexer_scores buffer */
    return bytes;
}



pulsar_gpu_tensor *gpu_graph_alloc_kv_cache_tensor(bool managed, uint64_t bytes) {
    return managed ? pulsar_gpu_tensor_alloc_managed(bytes) : pulsar_gpu_tensor_alloc(bytes);
}



/* =========================================================================
 * GPU Diagnostic Dump Hooks.
 * =========================================================================
 *
 * The release path calls these after important stages, but they are no-ops
 * unless PULSAR_CUDA_GRAPH_DUMP_PREFIX is set.  Dumping synchronizes and restarts
 * the command batch, so it is intentionally isolated here.
 */

/* Is graph dumping enabled AT ALL this process?  Read once and cached.  Exposed
 * so graph allocation can skip buffers that exist ONLY to be dumped: the routed
 * MoE gate/up staging slabs are written by the routed-expert kernels and read by
 * nothing except gpu_graph_debug_dump_tensor, yet cost ~400 MB per session at
 * the default prefill_cap and ~2-4 MB/token of pointless device writes. */
/* The graph-dump knobs, parsed ONCE (L159 inc 4).  The filters used to be
 * re-read from the environment on every dump call -- three getenv scans per
 * tensor per layer whenever dumping was armed.  One struct, one parse. */
struct gpu_graph_dump_cfg {
    int         enabled;      ///< PULSAR_CUDA_GRAPH_DUMP_PREFIX set
    const char *prefix;       ///< file prefix
    const char *name;         ///< PULSAR_CUDA_GRAPH_DUMP_NAME substring filter, or NULL
    long        layer;        ///< PULSAR_CUDA_GRAPH_DUMP_LAYER, -1 = all
    long        pos;          ///< PULSAR_CUDA_GRAPH_DUMP_POS, -1 = all
};
static const gpu_graph_dump_cfg *gpu_graph_dump_cfg_get(void) {
    static gpu_graph_dump_cfg cfg;
    static int parsed = 0;
    if (!parsed) {
        parsed = 1;
        const char *p = getenv("PULSAR_CUDA_GRAPH_DUMP_PREFIX");
        cfg.enabled = (p && p[0]) ? 1 : 0;
        cfg.prefix = cfg.enabled ? p : NULL;
        const char *n = getenv("PULSAR_CUDA_GRAPH_DUMP_NAME");
        cfg.name = (n && n[0]) ? n : NULL;
        const char *l = getenv("PULSAR_CUDA_GRAPH_DUMP_LAYER");
        cfg.layer = (l && l[0] && strcmp(l, "all") != 0) ? (long)strtoul(l, NULL, 10) : -1;
        const char *q = getenv("PULSAR_CUDA_GRAPH_DUMP_POS");
        cfg.pos = (q && q[0]) ? (long)strtoul(q, NULL, 10) : -1;
    }
    return &cfg;
}
bool gpu_graph_debug_dump_enabled(void) {
    return gpu_graph_dump_cfg_get()->enabled != 0;
}
const char *gpu_graph_debug_dump_prefix(void) {
    return gpu_graph_dump_cfg_get()->prefix;
}
bool gpu_graph_debug_wants(const char *name, uint32_t il, uint32_t pos) {
    const gpu_graph_dump_cfg *c = gpu_graph_dump_cfg_get();
    if (!c->enabled) return false;
    if (c->name && strstr(c->name, name) == NULL) return false;
    if (c->layer >= 0 && (uint32_t)c->layer != il) return false;
    if (c->pos >= 0 && (uint32_t)c->pos != pos) return false;
    return true;
}


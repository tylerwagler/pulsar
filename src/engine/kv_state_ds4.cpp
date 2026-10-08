/* DeepSeek V4's KV state (kv_state.h): what a bank's state at a grid point G is.
 *
 *   - the compressed and index-K rows [0, G/ratio) of every kv source: APPEND-ONLY, never rewritten
 *     while the frontier stays at or above them -- the POOLS, referenced through the frontier,
 *     serialized by a segment;
 *   - the raw window [G - raw_window, G) of every layer: the ring slot of a position is rewritten
 *     raw_cap positions later, so it is copied into the SLOT;
 *   - an overlapping (coff 2, 0731 ratio 4) source's recurrent lanes, attention and indexer: the
 *     carry half holds the projections of [G - ratio, G), which nothing else retains -- copied;
 *   - a coff-1 (ratio 128) lane: canonical-empty at a 128 boundary (every slot is rewritten before
 *     the next emit reads it), so it is reset, not stored.
 * Nothing else carries across a position: the HC residual is per token, logits are not kept (a
 * resume always evaluates at least one token past G), and the drafter rings are rebuilt from the
 * prompt the way every rewind already does.
 *
 * THE RESUME GRID (L195/L218): a continuation is a cold prefill from G = the last multiple of the
 * grid at or below the PREFILL frontier (decode rows are the decode kernels' and can never equal a
 * cold prefill's).  At any even position the ratio-2 compressors hold no pending group and the
 * ratio-1 compressor no state at all, so the state at G IS the cold prefill's.  128 is a multiple of
 * 32, the period of the one remaining chunk-mate mechanism, the HC-mix GEMM's dependence on a row's
 * offset within the call (censuses 14/15, 2026-09-06, at n_embd 4096; RE-CENSUS at 5120 before
 * moving this). */
#include "pulsar_engine_internal.h"

#include <stdio.h>

#define DS4_RESUME_GRID 128u
static_assert(DS4_RESUME_GRID % 2u == 0u, "a grid point must be a complete ratio-2 group");
static_assert(DS4_RESUME_GRID % 128u == 0u, "every compress ratio (4, 128) must divide the grid");

/* The newest checkpoints are the prompt ends of the turns a client is most likely to continue;
 * the older ones thin out into a ladder reaching back into the history a client rewrites (L261:
 * tool results 12-20k tokens back).  ~3.8 MB a slot. */
#define DS4_CKPT_SLOTS 16u
#define DS4_CKPT_RECENT 8u
static_assert(DS4_CKPT_SLOTS <= PULSAR_CKPT_SLOTS_MAX, "the store's slot bound");

static pulsar_gpu_graph *G_(void *state) { return (pulsar_gpu_graph *)state; }

static bool ds4_walk(void *state, int dir, pulsar_gpu_tensor *slab, uint64_t slot_off, uint32_t G,
                     uint64_t *bytes_out) {
    pulsar_gpu_graph *g = G_(state);
    const uint64_t raw_row = pulsar_kv_row_bytes(PULSAR_KV_ROW_RING);
    const uint32_t W = g->raw_window;
    uint64_t off = slot_off;
    bool ok = true;
    /* The window's first slot in the ring, and how many rows run before the wrap. */
    const uint32_t first = (G - W) % g->raw_cap;
    const uint32_t run = (g->raw_cap - first) < W ? (g->raw_cap - first) : W;
    for (uint32_t il = 0; il < PULSAR_N_LAYER && ok; il++) {
        pulsar_gpu_tensor *raw = g->layer_raw_cache[il];
        if (dir == 0) {
            ok = pulsar_gpu_tensor_copy_async(slab, off, raw, (uint64_t)first * raw_row, (uint64_t)run * raw_row) != 0 &&
                 (run == W || pulsar_gpu_tensor_copy_async(slab, off + (uint64_t)run * raw_row, raw, 0,
                                                           (uint64_t)(W - run) * raw_row) != 0);
        } else if (dir > 0) {
            ok = pulsar_gpu_tensor_copy_async(raw, (uint64_t)first * raw_row, slab, off, (uint64_t)run * raw_row) != 0 &&
                 (run == W || pulsar_gpu_tensor_copy_async(raw, 0, slab, off + (uint64_t)run * raw_row,
                                                           (uint64_t)(W - run) * raw_row) != 0);
        }
        off += (uint64_t)W * raw_row;
        if (!gpu_graph_layer_is_kv_source(il)) continue;
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio <= 1u || pulsar_compress_coff(ratio) == 1u) continue;
        pulsar_gpu_tensor *lanes[4] = {
            g->layer_attn_state_kv[il], g->layer_attn_state_score[il],
            g->layer_index_state_kv[il], g->layer_index_state_score[il],
        };
        for (int k = 0; k < 4 && ok; k++) {
            if (!lanes[k]) continue;
            const uint64_t n = pulsar_gpu_tensor_bytes(lanes[k]);
            if (dir == 0) ok = pulsar_gpu_tensor_copy_async(slab, off, lanes[k], 0, n) != 0;
            else if (dir > 0) ok = pulsar_gpu_tensor_copy_async(lanes[k], 0, slab, off, n) != 0;
            off += n;
        }
    }
    if (bytes_out) *bytes_out = off - slot_off;
    return ok;
}

static uint32_t ds4_min_checkpoint(void *state) { return G_(state)->raw_window; }

static bool ds4_stands_at(void *state, uint32_t G, char *why, size_t whylen) {
    pulsar_gpu_graph *g = G_(state);
    const uint32_t bank = gpu_graph_cur_bank(g);
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (!gpu_graph_layer_is_kv_source(il)) continue;
        const uint32_t want = G / pulsar_layer_compress_ratio(il);
        if (gpu_graph_n_comp(g, bank, il) != want) {
            snprintf(why, whylen, "layer %u holds %u compressed rows, a frontier at %u holds %u", il,
                     gpu_graph_n_comp(g, bank, il), G, want);
            return false;
        }
    }
    return true;
}

static bool ds4_holds(void *state, uint32_t G) {
    pulsar_gpu_graph *g = G_(state);
    const uint32_t bank = gpu_graph_cur_bank(g);
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++)
        if (gpu_graph_layer_is_kv_source(il) && gpu_graph_n_comp(g, bank, il) < G / pulsar_layer_compress_ratio(il))
            return false;
    return true;
}

static void ds4_set_frontier(pulsar_gpu_graph *g, uint32_t bank, uint32_t G) {
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++)
        if (gpu_graph_layer_is_kv_source(il)) gpu_graph_set_n_comp(g, bank, il, G / pulsar_layer_compress_ratio(il));
}

static bool ds4_prepare_restore(void *state, uint32_t G) {
    pulsar_gpu_graph *g = G_(state);
    const uint32_t bank = gpu_graph_cur_bank(g);
    /* Frontier first: lowering it drops the checkpoints above G (not this one, which sits exactly
     * at G/ratio).  Then every lane to the canonical boundary state; the walk brings the coff-2
     * lanes back from the slot, and a coff-1 lane at G IS the canonical state. */
    ds4_set_frontier(g, bank, G);
    return gpu_graph_compressor_state_reset(g, bank);
}

static void ds4_restored(void *state) {
    pulsar_gpu_graph *g = G_(state);
    g->ms_comp_state_stale[gpu_graph_cur_bank(g)] = false;
}

static void ds4_set_frontier_stale(void *state, uint32_t G) {
    pulsar_gpu_graph *g = G_(state);
    const uint32_t bank = gpu_graph_cur_bank(g);
    ds4_set_frontier(g, bank, G);
    g->ms_comp_state_stale[bank] = true;
}

static uint32_t ds4_pools(void *state, pulsar_kv_pool *out, uint32_t cap) {
    pulsar_gpu_graph *g = G_(state);
    uint32_t n = 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (!gpu_graph_layer_is_kv_source(il)) continue;
        const uint32_t r = pulsar_layer_compress_ratio(il);
        if (n < cap) out[n] = { g->layer_attn_comp_cache[il], r, pulsar_kv_row_bytes(PULSAR_KV_ROW_COMP) };
        n++;
        if (!gpu_graph_layer_has_index_pool(il)) continue;
        if (n < cap) out[n] = { g->layer_index_comp_cache[il], r, pulsar_kv_row_bytes(PULSAR_KV_ROW_INDEX) };
        n++;
    }
    return n;
}

/* A fused chunk to T: the graph's frontier counters are the step's own (the graph steps each run's rows); a bank
 * whose compressor group is stale holds lanes that describe no prefill, so it is not captured (the sync's
 * grid-point resume rebuilds it). */
static bool ds4_noted_at(void *state, uint32_t, bool *capture, char *, size_t) {
    pulsar_gpu_graph *g = G_(state);
    *capture = !g->ms_comp_state_stale[gpu_graph_cur_bank(g)];
    return true;
}

const pulsar_kv_state_ops PULSAR_KV_STATE_DS4 = {
    /* .name               = */ "deepseek-v4",
    /* .resume_grid        = */ DS4_RESUME_GRID,
    /* .ckpt_slots         = */ DS4_CKPT_SLOTS,
    /* .ckpt_recent        = */ DS4_CKPT_RECENT,
    /* .walk               = */ ds4_walk,
    /* .min_checkpoint     = */ ds4_min_checkpoint,
    /* .stands_at          = */ ds4_stands_at,
    /* .holds              = */ ds4_holds,
    /* .prepare_restore    = */ ds4_prepare_restore,
    /* .restored           = */ ds4_restored,
    /* .set_frontier_stale = */ ds4_set_frontier_stale,
    /* .pools              = */ ds4_pools,
    /* .frontier_at        = */ NULL,   /* the payload is the graph's own (session_payload.cpp) */
    /* .install_frontier   = */ NULL,
    /* .trailing_pools     = */ NULL,
    /* .noted_at           = */ ds4_noted_at,
};

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
 * A session payload's FRONTIER T (L284, the kv-state payload) is the same state at any position: the
 * frontier walk adds the coff-1 lanes, which hold T's open group off a 128 boundary, and the drafter's rings
 * (a restored session speculates on as the saved one would).
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

/* ~3.8 MB a slot, so 16 (the ladder over them is checkpoint.cpp's). */
#define DS4_CKPT_SLOTS 16u
static_assert(DS4_CKPT_SLOTS <= PULSAR_CKPT_SLOTS_MAX, "the store's slot bound");

static pulsar_gpu_graph *G_(void *state) { return (pulsar_gpu_graph *)state; }

/* The slot at grid point G, or -- `frontier` -- the state at ANY position G (a session payload's frontier,
 * session_payload.cpp): the same window and coff-2 lanes plus every other lane of a kv source, since a coff-1
 * (ratio 128) lane is canonical-empty only on a 128 boundary and holds the open group's projections anywhere
 * else.  A frontier inside the first window carries positions [0, G) at the window's tail; the slot's head is
 * left as it is (a payload stages it over zeros), and a restore writes no ring slot of a position below 0. */
static bool ds4_walk(void *state, int dir, bool frontier, pulsar_gpu_tensor *slab, uint64_t slot_off, uint32_t G,
                     uint64_t *bytes_out) {
    pulsar_gpu_graph *g = G_(state);
    const uint64_t raw_row = pulsar_kv_row_bytes(PULSAR_KV_ROW_RING);
    const uint32_t W = g->raw_window;
    uint64_t off = slot_off;
    bool ok = true;
    /* one device lane, whole */
    auto lane = [&](pulsar_gpu_tensor *t) {
        if (!t || !ok) return;
        const uint64_t lb = pulsar_gpu_tensor_bytes(t);
        if (dir == 0) ok = pulsar_gpu_tensor_copy_async(slab, off, t, 0, lb) != 0;
        else if (dir > 0) ok = pulsar_gpu_tensor_copy_async(t, 0, slab, off, lb) != 0;
        off += lb;
    };
    /* The window's positions [G - n, G): the first one's slot in the ring, and how many run before the wrap. */
    const uint32_t n = G < W ? G : W, skip = W - n;
    const uint32_t first = n ? (G - n) % g->raw_cap : 0u;
    const uint32_t run = (g->raw_cap - first) < n ? (g->raw_cap - first) : n;
    for (uint32_t il = 0; il < PULSAR_N_LAYER && ok; il++) {
        pulsar_gpu_tensor *raw = g->layer_raw_cache[il];
        const uint64_t at = off + (uint64_t)skip * raw_row;
        if (dir == 0 && n) {
            ok = pulsar_gpu_tensor_copy_async(slab, at, raw, (uint64_t)first * raw_row, (uint64_t)run * raw_row) != 0 &&
                 (run == n || pulsar_gpu_tensor_copy_async(slab, at + (uint64_t)run * raw_row, raw, 0,
                                                           (uint64_t)(n - run) * raw_row) != 0);
        } else if (dir > 0 && n) {
            ok = pulsar_gpu_tensor_copy_async(raw, (uint64_t)first * raw_row, slab, at, (uint64_t)run * raw_row) != 0 &&
                 (run == n || pulsar_gpu_tensor_copy_async(raw, 0, slab, at + (uint64_t)run * raw_row,
                                                           (uint64_t)(n - run) * raw_row) != 0);
        }
        off += (uint64_t)W * raw_row;
        if (!gpu_graph_layer_is_kv_source(il)) continue;
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (!frontier && (ratio <= 1u || pulsar_compress_coff(ratio) == 1u)) continue;
        lane(g->layer_attn_state_kv[il]);
        lane(g->layer_attn_state_score[il]);
        lane(g->layer_index_state_kv[il]);
        lane(g->layer_index_state_score[il]);
    }
    /* The frontier's drafter (DSpark, when loaded): its raw KV ring and prompt-window ring, the installed bank's
     * views, and their counters as a host tail -- the state the next speculative round drafts from (Qwen's MTP
     * stage and pending row are its slot's).  A grid checkpoint does not carry it: a resume re-captures the
     * prompt window as its prefill runs. */
    if (frontier && g->dspark_raw_cache[0]) {
        for (int i = 0; i < 3; i++) {
            lane(g->dspark_raw_cache[i]);
            lane(g->dspark_prompt_h[i]);
        }
        uint32_t c[5] = { g->dspark_n_raw[0], g->dspark_n_raw[1], g->dspark_n_raw[2], g->dspark_prompt_n,
                          g->dspark_prompt_lo };
        if (ok && dir == 0) ok = pulsar_gpu_tensor_write(slab, off, c, sizeof(c)) != 0;
        else if (ok && dir > 0) {
            ok = pulsar_gpu_tensor_read(slab, off, c, sizeof(c)) != 0;
            for (int i = 0; ok && i < 3; i++) g->dspark_n_raw[i] = c[i];
            if (ok) { g->dspark_prompt_n = c[3]; g->dspark_prompt_lo = c[4]; }
        }
        off += sizeof(c);
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
    /* the bank's Engram history is whatever it held before the restore: the step that resumes from G re-seeds
     * it from its prompt (gpu_graph_engram_stage), and a step that does not refuses instead of hashing stale ids */
    gpu_graph_engram_forget(g, gpu_graph_cur_bank(g));
}

static void ds4_set_frontier_stale(void *state, uint32_t G) {
    pulsar_gpu_graph *g = G_(state);
    const uint32_t bank = gpu_graph_cur_bank(g);
    ds4_set_frontier(g, bank, G);
    g->ms_comp_state_stale[bank] = true;
}

/* Every pool's rows are WHOLE: an emit writes a row when its group closes, and the open group is the lanes'. */

static bool ds4_stale(void *state, uint32_t bank) {
    const pulsar_gpu_graph *g = G_(state);
    return bank < gpu_graph_bank_pool_count(g) && bank < PULSAR_MSEQ_MAX && g->ms_comp_state_stale[bank];
}

static uint32_t ds4_pools(void *state, pulsar_kv_pool *out, uint32_t cap) {
    pulsar_gpu_graph *g = G_(state);
    uint32_t n = 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (!gpu_graph_layer_is_kv_source(il)) continue;
        const uint32_t r = pulsar_layer_compress_ratio(il);
        if (n < cap) out[n] = { g->layer_attn_comp_cache[il], r, pulsar_kv_row_bytes(PULSAR_KV_ROW_COMP), true };
        n++;
        if (!gpu_graph_layer_has_index_pool(il)) continue;
        if (n < cap) out[n] = { g->layer_index_comp_cache[il], r, pulsar_kv_row_bytes(PULSAR_KV_ROW_INDEX), true };
        n++;
    }
    return n;
}

/* L284: the payload's frontier (session_payload.cpp).  The lanes describe T when every pool stands at T (a
 * decode row emits as a prefill row does) and no off-grid rewind left the compressor group stale. */
static bool ds4_frontier_at(void *state, uint32_t T, char *why, size_t whylen) {
    pulsar_gpu_graph *g = G_(state);
    if (g->ms_comp_state_stale[gpu_graph_cur_bank(g)]) {
        snprintf(why, whylen, "the bank's state is stale (rewound off its grid checkpoints); sync it first");
        return false;
    }
    return ds4_stands_at(state, T, why, whylen);
}

/* The counters at T; the frontier walk brings the window and every lane back.  The prefill frontier is not the
 * graph's: it is the session's (pulsar_session::prefill_frontier), which the payload's load sets. */
static bool ds4_install_frontier(void *state, uint32_t T, uint32_t) {
    pulsar_gpu_graph *g = G_(state);
    ds4_set_frontier(g, gpu_graph_cur_bank(g), T);
    return true;
}

/* A prompt chunk to T -- the planner's, or a fused step's: the graph's frontier counters are the chunk's own (the
 * graph steps each run's rows); a bank whose compressor group is stale holds lanes that describe no prefill, so it
 * is not captured (the sync's grid-point resume rebuilds it; a planner chunk never runs on one -- the compressor
 * store refuses). */
static bool ds4_noted_at(void *state, uint32_t, bool *capture, char *, size_t) {
    pulsar_gpu_graph *g = G_(state);
    *capture = !g->ms_comp_state_stale[gpu_graph_cur_bank(g)];
    return true;
}

const pulsar_kv_state_ops PULSAR_KV_STATE_DS4 = {
    /* .name               = */ "deepseek-v4",
    /* .resume_grid        = */ DS4_RESUME_GRID,
    /* .split_invariant    = */ false,   /* L183: only a grid cut is the cold prefill's */
    /* .ckpt_slots         = */ DS4_CKPT_SLOTS,
    /* .walk               = */ ds4_walk,
    /* .min_checkpoint     = */ ds4_min_checkpoint,
    /* .stands_at          = */ ds4_stands_at,
    /* .holds              = */ ds4_holds,
    /* .prepare_restore    = */ ds4_prepare_restore,
    /* .restored           = */ ds4_restored,
    /* .set_frontier_stale = */ ds4_set_frontier_stale,
    /* .stale              = */ ds4_stale,
    /* .pools              = */ ds4_pools,
    /* .frontier_at        = */ ds4_frontier_at,
    /* .install_frontier   = */ ds4_install_frontier,
    /* .trailing_pools     = */ NULL,   /* no row trails its position */
    /* .noted_at           = */ ds4_noted_at,
};

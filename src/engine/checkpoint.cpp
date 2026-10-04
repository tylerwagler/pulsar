/* L264 grid checkpoints: the state of a bank at a prefill grid point that its
 * compressed pools do not already hold, so a later request can resume there by
 * copying it back instead of rebuilding it.
 *
 * What "the state at G" is (G a multiple of PULSAR_RESUME_GRID, so a group
 * boundary of every compressor):
 *   - the compressed and index-K rows [0, G/ratio): append-only, never rewritten
 *     while the frontier stays at or above them -- referenced, not copied;
 *   - the raw window [G - raw_window, G) of every layer: the ring slot of a
 *     position is rewritten raw_cap positions later, so it is copied;
 *   - an overlapping (coff 2, 0731 ratio 4) source's recurrent lanes, attention
 *     and indexer: the carry half holds the projections of [G - ratio, G), which
 *     nothing else retains -- copied;
 *   - a coff-1 (ratio 128) lane: canonical-empty at a 128 boundary (every slot is
 *     rewritten before the next emit reads it), so it is reset, not stored.
 * Nothing else carries across a position: the HC residual is per token, logits
 * are not kept (a resume always evaluates at least one token past G), and the
 * drafter rings are rebuilt from the prompt the way every rewind already does.
 *
 * Validity is the frontier: a checkpoint at G is good while its bank holds rows
 * [0, G/ratio) unmodified.  gpu_graph_set_n_comp drops it when a counter falls
 * below that, and an operation that replaces a bank's rows wholesale (a payload
 * load, an eviction) drops all of the bank's checkpoints. */
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <string.h>

/* The layout of one slot, walked identically by sizing, capture and restore so
 * there is one authority for it: per layer the raw window, then for a coff-2
 * source its attention lane (kv, score) and indexer lane (kv, score).  `dir`
 * < 0 sizes only; 0 copies graph -> slot; > 0 copies slot -> graph. */
static bool ckpt_walk(pulsar_gpu_graph *g, int dir, pulsar_gpu_tensor *slab, uint64_t slot_off,
                      uint32_t G, uint64_t *bytes_out) {
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

bool gpu_graph_ckpt_alloc(pulsar_gpu_graph *g, uint32_t n_banks) {
    if (n_banks < 1u) n_banks = 1u;
    if (n_banks > PULSAR_MSEQ_MAX) return false;
    uint64_t slot = 0;
    (void)ckpt_walk(g, -1, NULL, 0, g->raw_window, &slot);
    /* The batched copies want 256-B aligned starts; keep every slot on one. */
    g->ckpt_slot_bytes = (slot + 255u) & ~(uint64_t)255u;
    memset(g->ckpt_pos, 0, sizeof(g->ckpt_pos));
    for (uint32_t b = 0; b < n_banks; b++) {
        g->ckpt_slab[b] = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_CKPT_SLOTS * g->ckpt_slot_bytes);
        if (!g->ckpt_slab[b]) {
            fprintf(stderr, "pulsar: grid checkpoint slab for bank %u (%llu bytes) allocation failed\n",
                    b, (unsigned long long)((uint64_t)PULSAR_CKPT_SLOTS * g->ckpt_slot_bytes));
            return false;
        }
    }
    return true;
}

void gpu_graph_ckpt_release(pulsar_gpu_graph *g) {
    for (uint32_t b = 0; b < PULSAR_MSEQ_MAX; b++) {
        pulsar_gpu_tensor_free(g->ckpt_slab[b]);
        g->ckpt_slab[b] = NULL;
    }
    memset(g->ckpt_pos, 0, sizeof(g->ckpt_pos));
}

/* The slot a capture at G writes: the one already holding G, else an empty one,
 * else the victim of the retention rule -- never one of the PULSAR_CKPT_RECENT
 * newest, and among the older ones the checkpoint whose removal opens the
 * smallest gap between its neighbours (the newest is G itself, about to land). */
static uint32_t ckpt_pick_slot(const pulsar_gpu_graph *g, uint32_t bank, uint32_t G) {
    const uint32_t *pos = g->ckpt_pos[bank];
    for (uint32_t s = 0; s < PULSAR_CKPT_SLOTS; s++) if (pos[s] == G) return s;
    for (uint32_t s = 0; s < PULSAR_CKPT_SLOTS; s++) if (pos[s] == 0u) return s;
    uint32_t order[PULSAR_CKPT_SLOTS];
    for (uint32_t s = 0; s < PULSAR_CKPT_SLOTS; s++) order[s] = s;
    for (uint32_t i = 1; i < PULSAR_CKPT_SLOTS; i++) {          /* ascending by position */
        const uint32_t v = order[i];
        uint32_t j = i;
        for (; j > 0 && pos[order[j - 1]] > pos[v]; j--) order[j] = order[j - 1];
        order[j] = v;
    }
    const uint32_t older = PULSAR_CKPT_SLOTS - (PULSAR_CKPT_RECENT - 1u);   /* G is the newest */
    uint32_t victim = order[0];
    uint32_t best_gap = UINT32_MAX;
    for (uint32_t i = 0; i < older; i++) {
        const uint32_t lo = i == 0 ? 0u : pos[order[i - 1]];
        const uint32_t hi = pos[order[i + 1]];
        if (hi - lo < best_gap) { best_gap = hi - lo; victim = order[i]; }
    }
    return victim;
}

bool gpu_graph_ckpt_capture(pulsar_gpu_graph *g, uint32_t G) {
    const uint32_t bank = gpu_graph_cur_bank(g);
    if (G < g->raw_window || G % PULSAR_RESUME_GRID != 0u || !g->ckpt_slab[bank]) {
        fprintf(stderr, "pulsar: grid checkpoint at %u on bank %u refused: not a grid point the slab can hold\n",
                G, bank);
        return false;
    }
    /* The precondition is what makes the copy the state AT G: every source
     * holds exactly the rows a prefill to G emits, no more (a row above would be
     * a decode's, which the frontier says nothing about) and no fewer. */
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (!gpu_graph_layer_is_kv_source(il)) continue;
        const uint32_t want = G / pulsar_layer_compress_ratio(il);
        if (gpu_graph_n_comp(g, bank, il) != want) {
            fprintf(stderr, "pulsar: grid checkpoint at %u on bank %u refused: layer %u holds %u compressed rows, "
                            "a frontier at %u holds %u\n", G, bank, il, gpu_graph_n_comp(g, bank, il), G, want);
            return false;
        }
    }
    const uint32_t s = ckpt_pick_slot(g, bank, G);
    g->ckpt_pos[bank][s] = 0u;   /* not a checkpoint until the copy is issued whole */
    if (!ckpt_walk(g, 0, g->ckpt_slab[bank], (uint64_t)s * g->ckpt_slot_bytes, G, NULL)) {
        fprintf(stderr, "pulsar: grid checkpoint at %u on bank %u: copy failed\n", G, bank);
        return false;
    }
    g->ckpt_pos[bank][s] = G;
    return true;
}

bool gpu_graph_ckpt_restore(pulsar_gpu_graph *g, uint32_t G) {
    const uint32_t bank = gpu_graph_cur_bank(g);
    uint32_t s = PULSAR_CKPT_SLOTS;
    for (uint32_t k = 0; k < PULSAR_CKPT_SLOTS; k++) if (G != 0u && g->ckpt_pos[bank][k] == G) s = k;
    if (s == PULSAR_CKPT_SLOTS) return false;
    /* Frontier first: lowering it drops the checkpoints above G (and not this
     * one, which sits exactly at G/ratio). */
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (!gpu_graph_layer_is_kv_source(il)) continue;
        gpu_graph_set_n_comp(g, bank, il, G / pulsar_layer_compress_ratio(il));
    }
    /* Every lane to the canonical boundary state, then the coff-2 lanes from the
     * slot: a coff-1 lane at G IS the canonical state. */
    if (!gpu_graph_compressor_state_reset(g, bank) ||
        !ckpt_walk(g, 1, g->ckpt_slab[bank], (uint64_t)s * g->ckpt_slot_bytes, G, NULL)) {
        fprintf(stderr, "pulsar: grid checkpoint restore to %u on bank %u: copy failed\n", G, bank);
        return false;
    }
    /* The stale flag described a lane that is now exact. */
    g->ms_comp_state_stale[bank] = false;
    for (uint32_t k = 0; k < PULSAR_CKPT_SLOTS; k++)
        if (g->ckpt_pos[bank][k] > G) g->ckpt_pos[bank][k] = 0u;
    return true;
}

uint32_t gpu_graph_ckpt_best(const pulsar_gpu_graph *g, uint32_t bank, uint32_t limit) {
    if (bank >= PULSAR_MSEQ_MAX) return 0u;
    uint32_t best = 0u;
    for (uint32_t k = 0; k < PULSAR_CKPT_SLOTS; k++) {
        const uint32_t p = g->ckpt_pos[bank][k];
        if (p != 0u && p <= limit && p > best) best = p;
    }
    return best;
}

bool gpu_graph_ckpt_locate(pulsar_gpu_graph *g, uint32_t G, pulsar_gpu_tensor **slab, uint64_t *off) {
    const uint32_t bank = gpu_graph_cur_bank(g);
    for (uint32_t k = 0; k < PULSAR_CKPT_SLOTS; k++) {
        if (G == 0u || g->ckpt_pos[bank][k] != G) continue;
        *slab = g->ckpt_slab[bank];
        *off = (uint64_t)k * g->ckpt_slot_bytes;
        return true;
    }
    return false;
}

bool gpu_graph_ckpt_claim(pulsar_gpu_graph *g, uint32_t G, pulsar_gpu_tensor **slab, uint64_t *off,
                          uint32_t *slot) {
    const uint32_t bank = gpu_graph_cur_bank(g);
    if (G < g->raw_window || G % PULSAR_RESUME_GRID != 0u || !g->ckpt_slab[bank]) return false;
    const uint32_t k = ckpt_pick_slot(g, bank, G);
    g->ckpt_pos[bank][k] = 0u;   /* filled by the caller, committed after */
    *slab = g->ckpt_slab[bank];
    *off = (uint64_t)k * g->ckpt_slot_bytes;
    *slot = k;
    return true;
}

void gpu_graph_ckpt_commit(pulsar_gpu_graph *g, uint32_t slot, uint32_t G) {
    if (slot < PULSAR_CKPT_SLOTS) g->ckpt_pos[gpu_graph_cur_bank(g)][slot] = G;
}

void gpu_graph_ckpt_drop_bank(pulsar_gpu_graph *g, uint32_t bank) {
    if (bank < PULSAR_MSEQ_MAX) memset(g->ckpt_pos[bank], 0, sizeof(g->ckpt_pos[bank]));
}

void gpu_graph_ckpt_frontier_lowered(pulsar_gpu_graph *g, uint32_t bank, uint32_t il, uint32_t rows) {
    const uint32_t ratio = pulsar_layer_compress_ratio(il);
    for (uint32_t k = 0; k < PULSAR_CKPT_SLOTS; k++) {
        const uint32_t p = g->ckpt_pos[bank][k];
        if (p != 0u && rows < p / ratio) g->ckpt_pos[bank][k] = 0u;
    }
}

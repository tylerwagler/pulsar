/* Qwen3.8-Flash-Next's KV state (kv_state.h, L266 step 5): what a bank's state at a grid point G is.
 *
 *   - every QSA layer's KV rows and pooled indexer keys [0, G): position-indexed, rewritten only by a
 *     prefill or decode AT their position -- the POOLS, referenced through the frontier;
 *   - every GDN layer's recurrent state and causal-conv tail, the PLE layer's dilated-conv history,
 *     every QSA layer's open index block (the MTP layer's too), the MTP layer's pending trunk row:
 *     overwritten as positions advance -- copied into the SLOT (~118 MB);
 *   - the n-gram context (the last two token ids, reset at EOS): host state, carried in the slot's
 *     tail so a restore needs no token replay.
 * The MTP layer's KV lags the trunk by one row (row p is written once x_{p+1} exists), so at G it holds
 * rows [0, G - 1) and the slot's pending row is the trunk stack at G - 1.
 *
 * THE RESUME GRID: Qwen's recurrent GDN and per-row QSA are split-invariant by construction, and every
 * prompt chunk takes the prefill arms at every row count with a pinned GEMM (L266:
 * qwen_chunk_neutrality_gate) -- so a prompt cut ANYWHERE is the cold prefill's bytes.  The grid is
 * therefore free; 128 keeps the slots on the server's and DeepSeek's cadence and is a multiple of the
 * indexer block (4).
 *
 * THE PREFILL FRONTIER: a checkpoint is the cold prefill of its tokens only if every row below it was
 * prefilled (decode rows are the decode arms').  pulsar_qwen_state::prefill_pos[bank] is the end of the
 * bank's prefill-only history; decode never advances it, and a capture requires bank_pos == G ==
 * prefill_pos. */
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <string.h>

#define QWEN_RESUME_GRID 128u
static_assert(QWEN_RESUME_GRID % 4u == 0u, "a grid point must close an indexer block");

/* ~118 MB a slot: the newest is the last prompt end, one older one survives the next turn. */
#define QWEN_CKPT_SLOTS 2u
#define QWEN_CKPT_RECENT 1u
static_assert(QWEN_CKPT_SLOTS <= PULSAR_CKPT_SLOTS_MAX, "the store's slot bound");

static pulsar_qwen_state *Q_(void *state) { return (pulsar_qwen_state *)state; }

/* The slot's host tail: the n-gram context of the bank at G. */
static uint64_t qwen_host_tail_bytes(void) {
    return (uint64_t)(g_qwen_shape.ngram_size - 1u) * sizeof(int32_t);
}

static bool qwen_walk(void *state, int dir, pulsar_gpu_tensor *slab, uint64_t slot_off, uint32_t,
                      uint64_t *bytes_out) {
    pulsar_qwen_state *st = Q_(state);
    const pulsar_qwen_shape *s = &g_qwen_shape;
    const uint32_t bank = st->live_bank;
    uint64_t off = slot_off;
    bool ok = true;
    /* one bank-major tensor's slice: bank b at b * bytes */
    auto lane = [&](pulsar_gpu_tensor *t, uint64_t bytes) {
        if (!t || !ok) return;
        if (dir == 0) ok = pulsar_gpu_tensor_copy_async(slab, off, t, (uint64_t)bank * bytes, bytes) != 0;
        else if (dir > 0) ok = pulsar_gpu_tensor_copy_async(t, (uint64_t)bank * bytes, slab, off, bytes) != 0;
        off += bytes;
    };
    for (uint32_t il = 0; il < PULSAR_FAMILY_MAX_LAYER; il++) {
        pulsar_qwen_layer_state *L = &st->layer[il];
        lane(L->gdn_state, pulsar_qwen_gdn_state_bytes(s));
        lane(L->gdn_conv, pulsar_qwen_gdn_conv_bytes(s));
        lane(L->idx_tail, pulsar_qwen_index_tail_bytes(s));
        lane(L->ple_conv, pulsar_qwen_ple_conv_bytes(s));
    }
    if (st->mtp) lane(st->mtp_pend, pulsar_qwen_hc_dim(s) * PULSAR_QWEN_STREAM_ELT_SIZE);
    const uint64_t tail = qwen_host_tail_bytes();
    int32_t *ng = st->ngram_ctx + (size_t)bank * (s->ngram_size - 1u);
    if (ok && dir == 0) ok = pulsar_gpu_tensor_write(slab, off, ng, tail) != 0;
    else if (ok && dir > 0) ok = pulsar_gpu_tensor_read(slab, off, ng, tail) != 0;
    off += tail;
    if (bytes_out) *bytes_out = off - slot_off;
    return ok;
}

static uint32_t qwen_min_checkpoint(void *) { return QWEN_RESUME_GRID; }

static bool qwen_stands_at(void *state, uint32_t G, char *why, size_t whylen) {
    pulsar_qwen_state *st = Q_(state);
    const uint32_t bank = st->live_bank;
    if (st->bank_pos[bank] != G || st->prefill_pos[bank] != G) {
        snprintf(why, whylen, "bank %u holds %u tokens, %u of them prefilled, not exactly %u prefilled", bank,
                 st->bank_pos[bank], st->prefill_pos[bank], G);
        return false;
    }
    return true;
}

static bool qwen_holds(void *state, uint32_t G) {
    pulsar_qwen_state *st = Q_(state);
    return st->prefill_pos[st->live_bank] >= G;
}

static bool qwen_prepare_restore(void *state, uint32_t G) {
    pulsar_qwen_state *st = Q_(state);
    const uint32_t bank = st->live_bank;
    /* the pools' rows below G are the history the checkpoint references; the walk brings every
     * overwritten lane back, so only the counters move here */
    st->bank_pos[bank] = G;
    st->prefill_pos[bank] = G;
    st->mtp_pend_pos[bank] = st->mtp ? G - 1u : UINT32_MAX;
    st->logits_fresh = false;
    return true;
}

static void qwen_restored(void *state) { Q_(state)->frontier_stale[Q_(state)->live_bank] = false; }

static void qwen_set_frontier_stale(void *state, uint32_t G) {
    pulsar_qwen_state *st = Q_(state);
    const uint32_t bank = st->live_bank;
    st->bank_pos[bank] = G;
    st->prefill_pos[bank] = G;
    st->frontier_stale[bank] = true;
}

/* Every QSA layer's KV (one token a row) and pooled indexer keys (idx_block a row), the installed
 * bank's slice; the MTP layer's KV lags a row, so it is not a pool (a segment does not carry it). */
static uint32_t qwen_pools(void *state, pulsar_kv_pool *out, uint32_t cap) {
    pulsar_qwen_state *st = Q_(state);
    const pulsar_qwen_shape *s = &g_qwen_shape;
    const uint32_t bank = st->live_bank;
    const uint64_t kv_row = pulsar_qwen_kv_row_bytes(s), ix_row = pulsar_qwen_index_row_bytes(s);
    const uint64_t ix_rows = (st->ctx + s->idx_block - 1u) / s->idx_block;
    uint32_t n = 0;
    for (uint32_t il = 0; il < st->n_trunk_layers; il++) {
        pulsar_qwen_layer_state *L = &st->layer[il];
        if (!L->kv) continue;
        if (n < cap) out[n] = { L->kv, 1u, kv_row, (uint64_t)bank * st->ctx * kv_row };
        n++;
        if (n < cap) out[n] = { L->idx_keys, s->idx_block, ix_row, (uint64_t)bank * ix_rows * ix_row };
        n++;
    }
    return n;
}

const pulsar_kv_state_ops PULSAR_KV_STATE_QWEN = {
    /* .name               = */ "qwen4-exp",
    /* .resume_grid        = */ QWEN_RESUME_GRID,
    /* .ckpt_slots         = */ QWEN_CKPT_SLOTS,
    /* .ckpt_recent        = */ QWEN_CKPT_RECENT,
    /* .walk               = */ qwen_walk,
    /* .min_checkpoint     = */ qwen_min_checkpoint,
    /* .stands_at          = */ qwen_stands_at,
    /* .holds              = */ qwen_holds,
    /* .prepare_restore    = */ qwen_prepare_restore,
    /* .restored           = */ qwen_restored,
    /* .set_frontier_stale = */ qwen_set_frontier_stale,
    /* .pools              = */ qwen_pools,
};

/* family_qwen_banks.cpp -- the Qwen4-exp bank pool (L251): what the server's per-bank bookkeeping
 * needs from a family whose banks are not DeepSeek's graph pool (family.h pulsar_family_bank_ops).
 *
 * The device side is already per bank -- every bank has its own recurrent GDN state, QSA KV cache,
 * index tail, PLE conv state and n-gram context (pulsar_qwen_state), and the batched decode takes
 * one row per bank.  What was missing is the HOST side: the session carries ONE view (checkpoint,
 * logits) and sync / eval ran on bank 0.  Here the view belongs to `live_bank`; save copies it into
 * the bank's carry, restore makes a bank live and brings its carry back.  No device work.
 *
 * What a recurrent state cannot do is refused in engine_api.cpp, each with the value the server
 * already has a path for: forks (PULSAR_FORK_RING_SCROLLED -- permanently infeasible, so the router
 * takes a fresh bank or prefills cold in place) and per-bank KV spill files.
 *
 * PHYSICAL RESIDENCY (L284 #3) is DeepSeek's contract.  Each QSA layer's KV and pooled index keys are a
 * managed tensor per bank (family_qwen.h), so the server's 2b guard spills a Qwen victim as it does a
 * DeepSeek one: persist its segment chain, free_physical (cudaFree -- on GB10 the only thing that returns
 * physical), and on its return alloc_physical + load the chain (server bank_restore_spilled).  What stays
 * resident is the always-resident state: GDN recurrent + conv, PLE conv, the index tails, the checkpoint
 * slots -- and the MTP layer's KV, which a segment does not carry (kv_state_qwen.cpp qwen_pools; the
 * payload's trailing_pools op is what could carry it later). */
#include "pulsar_engine_internal.h"
#include "family_qwen.h"
#include "spec_internal.h"

#include <string.h>

static int qwen_bank_count(pulsar_session *s) {
    return s && s->qwen ? (int)s->qwen->n_banks : 0;
}

/* The host view is the core's carry (session_banks.cpp, L272 P2): save and restore are the core's, around
 * which bank is live. */
static void qwen_bank_save(pulsar_session *s, uint32_t bank) {
    if (!s || !s->qwen || bank >= s->qwen->n_banks) return;
    pulsar_bank_carry_save_view(s, bank);
}

static bool qwen_bank_restore(pulsar_session *s, uint32_t bank) {
    if (!s || !s->qwen || bank >= s->qwen->n_banks) return false;
    s->qwen->live_bank = bank;
    if (pulsar_bank_carry_restore_view(s, bank)) return true;
    /* a bank nothing was saved for: an empty, INVALID view, so its first sync resets the bank and
     * prefills cold (never continues whatever the device state holds) */
    s->checkpoint.len = 0;
    s->checkpoint_valid = false;
    s->logits_stale = true;
    spec_lookahead_reset(s);   /* no shadow was saved: the fresh bank has no pendings, carry or quench */
    return true;
}

static pulsar_ckpt_store *qwen_bank_kv_store(pulsar_session *s) { return s && s->qwen ? s->qwen->ckpt : NULL; }

static uint32_t qwen_bank_prefill_frontier(pulsar_session *s, uint32_t bank) {
    return s && s->qwen && bank < s->qwen->n_banks ? s->qwen->prefill_pos[bank] : 0u;
}

static uint32_t qwen_bank_live(pulsar_session *s) { return s && s->qwen ? s->qwen->live_bank : 0u; }

/* L270: the demand-paged KV accounting.  A bank's touched KV is its high-water's rows (the pages stay
 * resident once written, until free_physical returns the bank's own tensors); a quantum of q tokens grows
 * it by at most q rows (plus a partial index block). */
uint64_t qwen_demand_paged_bytes(pulsar_engine *e, int ctx_size);

static uint64_t qwen_bank_touched_kv_bytes(pulsar_session *s, uint32_t bank) {
    if (!s->qwen || bank >= s->qwen->n_banks) return 0;
    return qwen_kv_bytes_at(s->qwen, s->qwen->kv_hw[bank]);
}

static uint64_t qwen_bank_growth_bytes(pulsar_session *s, uint32_t q) {
    return s->qwen ? qwen_kv_bytes_at(s->qwen, q) : 0;
}

/* L284 #3: physical residency.  Only the live bank is refused -- sync / eval / the segment walk run on it;
 * a bank the batched lane steps is the server's to exclude (it restores a spilled bank before any step,
 * and the QSA op refuses a row of an evicted bank by name). */
static bool qwen_bank_free_physical(pulsar_session *s, uint32_t bank) {
    if (!s->qwen || bank >= s->qwen->n_banks) return false;
    if (bank == s->qwen->live_bank) {
        fprintf(stderr, "pulsar: %s: free_physical refuses bank %u: it is the live bank\n", PULSAR_QWEN_ARCH, bank);
        return false;
    }
    qwen_bank_kv_free(s->qwen, bank);
    return true;
}

static bool qwen_bank_alloc_physical(pulsar_session *s, uint32_t bank) {
    if (!s->qwen || bank >= s->qwen->n_banks) return false;
    return qwen_bank_kv_alloc(s->qwen, bank);
}

static bool qwen_bank_is_evicted(const pulsar_session *s, uint32_t bank) {
    return s->qwen && bank < s->qwen->n_banks && qwen_bank_kv_evicted(s->qwen, bank);
}

const pulsar_family_bank_ops k_qwen_bank_ops = {
    /* .count          = */ qwen_bank_count,
    /* .save           = */ qwen_bank_save,
    /* .restore        = */ qwen_bank_restore,
    /* .kv_store         = */ qwen_bank_kv_store,
    /* .prefill_frontier = */ qwen_bank_prefill_frontier,
    /* .live             = */ qwen_bank_live,
    /* .demand_paged_bytes = */ qwen_demand_paged_bytes,
    /* .touched_kv_bytes   = */ qwen_bank_touched_kv_bytes,
    /* .growth_bytes       = */ qwen_bank_growth_bytes,
    /* .free_physical      = */ qwen_bank_free_physical,
    /* .alloc_physical     = */ qwen_bank_alloc_physical,
    /* .is_evicted         = */ qwen_bank_is_evicted,
};

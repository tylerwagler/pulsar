#include "pulsar_engine_internal.h"

/* Tier-2 task #55 increment 2b — per-bank physical evict/restore.  The caller
 * (server guard) must have snapshotted the bank's KV to DISK before evict (host
 * RAM reclaims nothing on unified memory) and repointed away from it; and after
 * restore-alloc it reloads the KV H2D from that snapshot. */
bool pulsar_session::bank_free_physical(uint32_t bank) {
    auto *s = this;
    if (!s) return false;
    return gpu_graph_bank_free_physical(&s->graph, bank);
}

bool pulsar_session::bank_alloc_physical(uint32_t bank) {
    auto *s = this;
    if (!s) return false;
    return gpu_graph_bank_alloc_physical(&s->graph, bank);
}

bool pulsar_session::bank_is_evicted(uint32_t bank) const {
    auto *s = this;
    if (!s) return false;
    return gpu_graph_bank_is_evicted(&s->graph, bank);
}

uint64_t pulsar_session::bank_touched_kv_bytes(uint32_t bank) {
    auto *s = this;
    if (!s) return 0;
    return gpu_graph_bank_touched_kv_bytes(&s->graph, bank);
}

uint64_t pulsar_session::quantum_growth_bytes_per_bank(uint32_t q) {
    auto *s = this;
    (void)s;
    return gpu_graph_quantum_growth_bytes_per_bank(q);
}

/* ===== Tier-2 PATH A: per-bank HOST carry (pulsar_bank_carry) ================
 *
 * gpu_graph_bank_repoint swaps only the DEVICE cache views; the session's HOST
 * per-conversation state (checkpoint history, host logits, the DSpark fused-
 * loop / spec-carry shadow) is single-instance and must be saved for the bank
 * we leave and restored for the bank we enter.  save/restore below are that
 * host half; the graph frontier counters ride gpu_graph_bank_counters_*, and
 * (Option F) the per-bank drafter ring rides gpu_graph_bank_repoint. */

void pulsar_bank_carry::free_one() {
    auto *c = this;
    if (!c) return;
    token_vec_free(&c->checkpoint);
    free(c->logits);
    free(c->dspark_pending_qrows);
    memset(c, 0, sizeof(*c));
}

void pulsar_session::bank_carry_free() {
    auto *s = this;
    if (!s || !s->bank_carry) return;
    for (uint32_t i = 0; i < s->bank_carry_n; i++) s->bank_carry[i].free_one();
    free(s->bank_carry);
    s->bank_carry = NULL;
    s->bank_carry_n = 0;
}

static bool bank_carry_ensure(pulsar_session *s) {
    const uint32_t n = gpu_graph_bank_pool_count(&s->graph);
    if (s->bank_carry && s->bank_carry_n == n) return true;
    if (s->bank_carry) s->bank_carry_free();
    s->bank_carry = (pulsar_bank_carry *)xcalloc(n, sizeof(*s->bank_carry));
    s->bank_carry_n = n;
    return true;
}

int pulsar_session::bank_count() {
    auto *s = this;
    return s ? (int)gpu_graph_bank_pool_count(&s->graph) : 0;
}

int pulsar_session::bank_repoint(uint32_t bank) {
    auto *s = this;
    if (!s || bank >= gpu_graph_bank_pool_count(&s->graph)) return 1;
    /* Pool disabled: bank 0 is the classic tensors, nothing to repoint. */
    if (s->graph.banks.n_banks == 0) return bank == 0 ? 0 : 1;
    return gpu_graph_bank_repoint(&s->graph, bank) ? 0 : 1;
}

/* L260: the full q rows a bank carries -- rows j < dspark_qrows_n (the positions whose rows may still be read,
 * which outlives the pendings: round_begin drops them before the in-flight round's walk reads the rows) of
 * sampled drafts whose q did not fit the compact form (pulsar_spec_q_compact) -- copied from src to dst.  The
 * rest of the n_draft x PULSAR_N_VOCAB capacity is never read, and copying it on every bank switch was ~2.6 MB
 * per save and per restore at depth 5 under sampling (the c10 host-bookkeeping cost). */
static void copy_pending_qrows(float *dst, uint32_t dst_cap, const float *src, uint32_t src_cap,
                               const pulsar_spec_carry_state &sp) {
    if (!sp.dspark_pending_sampled) return;
    const uint32_t n = sp.dspark_qrows_n < 16u ? sp.dspark_qrows_n : 16u;
    for (uint32_t j = 0; j < n; j++) {
        if (pulsar_spec_q_compact(sp.dspark_pending_qn[j])) continue;
        const uint64_t end = (uint64_t)(j + 1u) * PULSAR_N_VOCAB;
        if (end > dst_cap || end > src_cap) return;   /* no row was drafted there: nothing to carry */
        memcpy(dst + (size_t)j * PULSAR_N_VOCAB, src + (size_t)j * PULSAR_N_VOCAB,
               (size_t)PULSAR_N_VOCAB * sizeof(float));
    }
}

void pulsar_session::bank_state_save(uint32_t bank) {
    auto *s = this;
    if (!s || bank >= gpu_graph_bank_pool_count(&s->graph)) return;
    if (!bank_carry_ensure(s)) return;
    /* Graph frontier counters (attn/index comp; Option F also the drafter ring
     * counters) are captured on the graph side so a later install re-arms this
     * bank's per-bank truth. */
    gpu_graph_bank_counters_capture(&s->graph, bank);
    pulsar_bank_carry *c = &s->bank_carry[bank];
    /* heap-backed deep copies */
    pulsar_tokens_copy(&c->checkpoint, &s->checkpoint);
    if (!c->logits) c->logits = (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(float));
    memcpy(c->logits, s->logits, (size_t)PULSAR_N_VOCAB * sizeof(float));
    if (s->dspark_pending_qrows && c->dspark_pending_qrows_cap < s->dspark_pending_qrows_cap) {
        c->dspark_pending_qrows = (float *)xrealloc(c->dspark_pending_qrows,
                                                    (size_t)s->dspark_pending_qrows_cap * sizeof(float));
        c->dspark_pending_qrows_cap = s->dspark_pending_qrows_cap;
    }
    /* scalar mirrors */
    c->checkpoint_valid       = s->checkpoint_valid;
    c->logits_stale           = s->logits_stale;
    c->prefill_frontier       = s->prefill_frontier;   /* L195 */
    c->live_image_fp          = s->live_image_fp;      /* L226: travels with the checkpoint */
    c->live_image_barrier     = s->live_image_barrier;
    /* Whole speculative/DSpark shadow in one assignment — a new field added to
     * pulsar_spec_carry_state is carried here for free (the old field-by-field
     * mirror was a silent-corruption footgun: miss one and the entering bank
     * inherits another conversation's speculative state).
     * s->mseq_dirty is NOT saved: it describes the graph's scalar frontier
     * counters, not this bank's conversation, and _restore re-establishes
     * per-bank frontier truth and clears it unconditionally. */
    pulsar_session_spec_chain_harvest(s);   /* L108 P2: never save an in-flight chain */
    c->spec = s->spec;
    /* the q rows the saved pendings can read (after the harvest, which only finishes greedy chains) */
    copy_pending_qrows(c->dspark_pending_qrows, c->dspark_pending_qrows_cap, s->dspark_pending_qrows,
                       s->dspark_pending_qrows_cap, c->spec);
    c->valid = true;
}

bool pulsar_session::bank_state_restore(uint32_t bank) {
    auto *s = this;
    if (!s || bank >= gpu_graph_bank_pool_count(&s->graph)) return false;
    /* Point device views (incl. Option F drafter ring) at this bank, and
     * re-arm its frontier counters — this is what makes clearing mseq_dirty
     * cheap and safe (per-bank truth is re-established without a re-prefill). */
    if (s->graph.banks.n_banks != 0 && !gpu_graph_bank_repoint(&s->graph, bank))
        return false;
    gpu_graph_bank_counters_install(&s->graph, bank);
    if (!s->bank_carry || bank >= s->bank_carry_n || !s->bank_carry[bank].valid) {
        /* No saved host state for a fresh bank: the counters_install above set
         * the (zeroed) frontier; leave the session's host shadow as the caller
         * primed it (a fresh sync just ran).  Clear the multiseq poison so
         * classic work resumes. */
        s->mseq_dirty = false;
        return true;
    }
    pulsar_bank_carry *c = &s->bank_carry[bank];
    pulsar_tokens_copy(&s->checkpoint, &c->checkpoint);
    memcpy(s->logits, c->logits, (size_t)PULSAR_N_VOCAB * sizeof(float));
    if (s->dspark_pending_qrows_cap < c->dspark_pending_qrows_cap) {
        s->dspark_pending_qrows = (float *)xrealloc(s->dspark_pending_qrows,
                                                    (size_t)c->dspark_pending_qrows_cap * sizeof(float));
        s->dspark_pending_qrows_cap = c->dspark_pending_qrows_cap;
    }
    copy_pending_qrows(s->dspark_pending_qrows, s->dspark_pending_qrows_cap, c->dspark_pending_qrows,
                       c->dspark_pending_qrows_cap, c->spec);
    s->checkpoint_valid       = c->checkpoint_valid;
    s->logits_stale           = c->logits_stale;
    s->prefill_frontier       = c->prefill_frontier;   /* L195 */
    s->live_image_fp          = c->live_image_fp;      /* L226 */
    s->live_image_barrier     = c->live_image_barrier;
    /* Mirror of the save above: one assignment restores the whole shadow. */
    s->spec = c->spec;
    /* Cheap resume: per-bank frontier truth is now installed, so the multiseq
     * superset poison no longer applies to this bank. */
    s->mseq_dirty = false;
    return true;
}



/* ===== Tier-2 per-bank frontier READERS (server routing/metrics) ==========
 *
 * A bank-pooled server shares one pulsar_session across N conversation banks, so
 * pulsar_session_pos / _tokens / _common_prefix (which read the single live host
 * checkpoint) describe ONLY the currently-installed bank.  Reading them for a
 * non-live bank returns the wrong conversation's frontier.  These readers give
 * the correct per-bank answer without repointing the device or disturbing the
 * live bank: the live bank (banks.cur_bank) reads the authoritative live
 * checkpoint; every other bank reads its saved host carry (which the server
 * keeps current for idle banks by bank_state_save'ing at job end).  Pure host
 * reads — no CUDA, safe on the worker thread at routing/publish time. */
static const pulsar_tokens *bank_frontier_tokens(pulsar_session *s, uint32_t bank) {
    if (!s || bank >= gpu_graph_bank_pool_count(&s->graph)) return NULL;
    const uint32_t cur = s->graph.banks.n_banks ? s->graph.banks.cur_bank : 0u;
    if (bank == cur) return s->checkpoint_valid ? &s->checkpoint : NULL;
    if (s->bank_carry && bank < s->bank_carry_n &&
        s->bank_carry[bank].valid && s->bank_carry[bank].checkpoint_valid)
        return &s->bank_carry[bank].checkpoint;
    return NULL;
}

/* L112 observability: the adaptive draft controller's CURRENT depth for a
 * bank -- live spec state for the live bank, the saved carry otherwise
 * (same live-vs-carry rule as bank_frontier_tokens above). In a bankless
 * session, bank 0 reads the live state. 0 = no drafter or nothing valid.
 * Pure host reads, worker-thread safe at publish time. */
int pulsar_session::bank_spec_depth(uint32_t bank) {
    auto *s = this;
    if (!s->engine || !s->engine->has_dspark()) return 0;
    const uint32_t pool = gpu_graph_bank_pool_count(&s->graph);
    const pulsar_spec_carry_state *sp = NULL;
    if (pool == 0) {
        if (bank == 0) sp = &s->spec;
    } else if (bank < pool) {
        const uint32_t cur = s->graph.banks.n_banks ? s->graph.banks.cur_bank : 0u;
        if (bank == cur) sp = &s->spec;
        else if (s->bank_carry && bank < s->bank_carry_n && s->bank_carry[bank].valid)
            sp = &s->bank_carry[bank].spec;
    }
    if (!sp) return 0;
    int d = sp->spec_adaptive_depth;
    if (d <= 0) d = s->engine->dspark_draft_tokens;
    if (d < 1) d = 1;
    if (d > 16) d = 16;
    return d;
}

int pulsar_session::bank_pos(uint32_t bank) {
    auto *s = this;
    const pulsar_tokens *t = bank_frontier_tokens(s, bank);
    return t ? t->len : 0;
}

/* L264: how far the bank's PREFILL reached (decode rows past it are the decode
 * kernels'; L195) -- the end of the last prompt it served, the same live-vs-
 * carry rule as bank_frontier_tokens.  0 when nothing valid. */
int pulsar_session::bank_prefill_frontier(uint32_t bank) {
    auto *s = this;
    if (!bank_frontier_tokens(s, bank)) return 0;
    const uint32_t cur = s->graph.banks.n_banks ? s->graph.banks.cur_bank : 0u;
    const int pf = bank == cur ? s->prefill_frontier : s->bank_carry[bank].prefill_frontier;
    return pf > 0 ? pf : 0;
}

const pulsar_tokens *pulsar_session::bank_tokens(uint32_t bank) {
    auto *s = this;
    return bank_frontier_tokens(s, bank);
}

#include "pulsar_engine_internal.h"
#include "spec_internal.h"

/* Tier-2 task #55 increment 2b — per-bank physical evict/restore.  The caller
 * (server guard) must have snapshotted the bank's KV to DISK before evict (host
 * RAM reclaims nothing on unified memory) and repointed away from it; and after
 * restore-alloc it reloads the KV H2D from that snapshot. */
bool pulsar_session::bank_free_physical(uint32_t bank) {
    auto *s = this;
    if (!s) return false;
    if (const pulsar_family_bank_ops *ops = FAMILY_BANKS(s)) return ops->free_physical(s, bank);   /* L284 #3 */
    return gpu_graph_bank_free_physical(s->graph, bank);
}

bool pulsar_session::bank_alloc_physical(uint32_t bank) {
    auto *s = this;
    if (!s) return false;
    if (const pulsar_family_bank_ops *ops = FAMILY_BANKS(s)) return ops->alloc_physical(s, bank);
    return gpu_graph_bank_alloc_physical(s->graph, bank);
}

bool pulsar_session::bank_is_evicted(uint32_t bank) const {
    auto *s = this;
    if (!s) return false;
    if (const pulsar_family_bank_ops *ops = FAMILY_BANKS(s)) return ops->is_evicted(s, bank);
    return gpu_graph_bank_is_evicted(s->graph, bank);
}

uint64_t pulsar_session::bank_touched_kv_bytes(uint32_t bank) {
    auto *s = this;
    if (!s) return 0;
    return gpu_graph_bank_touched_kv_bytes(s->graph, bank);
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
    free(c->pend_qrows);
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

/* L272 P2: the bank carry is the core's, for every family -- one array, one save and one restore of the
 * host view, one live-or-carry reader.  A family adds only its device side around them (DeepSeek: the
 * graph's frontier counters and views; Qwen: which bank is live).  How many banks: the family's pool,
 * or DeepSeek's graph pool. */
static uint32_t session_bank_n(pulsar_session *s) {
    return FAMILY_BANKS(s) ? (uint32_t)FAMILY_BANKS(s)->count(s) : gpu_graph_bank_pool_count(s->graph);
}

static bool bank_carry_ensure(pulsar_session *s) {
    const uint32_t n = session_bank_n(s);
    if (s->bank_carry && s->bank_carry_n == n) return true;
    if (s->bank_carry) s->bank_carry_free();
    s->bank_carry = (pulsar_bank_carry *)xcalloc(n, sizeof(*s->bank_carry));
    s->bank_carry_n = n;
    return true;
}

int pulsar_session::bank_count() {
    auto *s = this;
    return s ? (int)gpu_graph_bank_pool_count(s->graph) : 0;
}

int pulsar_session::bank_repoint(uint32_t bank) {
    auto *s = this;
    if (!s || bank >= gpu_graph_bank_pool_count(s->graph)) return 1;
    /* Pool disabled: bank 0 is the classic tensors, nothing to repoint. */
    if (s->graph->banks.n_banks == 0) return bank == 0 ? 0 : 1;
    return gpu_graph_bank_repoint(s->graph, bank) ? 0 : 1;
}


void pulsar_session::bank_state_save(uint32_t bank) {
    auto *s = this;
    if (!s || bank >= gpu_graph_bank_pool_count(s->graph)) return;
    if (!bank_carry_ensure(s)) return;
    /* Graph frontier counters (attn/index comp; Option F also the drafter ring
     * counters) are captured on the graph side so a later install re-arms this
     * bank's per-bank truth. */
    gpu_graph_bank_counters_capture(s->graph, bank);
    pulsar_bank_carry_save_view(s, bank);
}

void pulsar_bank_carry_save_view(pulsar_session *s, uint32_t bank) {
    if (!s || bank >= session_bank_n(s) || !bank_carry_ensure(s)) return;
    pulsar_bank_carry *c = &s->bank_carry[bank];
    /* heap-backed deep copies */
    pulsar_tokens_copy(&c->checkpoint, &s->checkpoint);
    const size_t nv = (size_t)s->engine->logits_width();
    if (!c->logits) c->logits = (float *)xmalloc(nv * sizeof(float));
    memcpy(c->logits, s->logits, nv * sizeof(float));
    /* scalar mirrors */
    c->checkpoint_valid       = s->checkpoint_valid;
    c->logits_stale           = s->logits_stale;
    c->prefill_frontier       = s->prefill_frontier;   /* L195 */
    c->live_images            = s->live_images;        /* L226 / L281: travels with the checkpoint */
    /* Whole speculative/DSpark shadow in one assignment — a new field added to
     * pulsar_spec_carry_state is carried here for free (the old field-by-field
     * mirror was a silent-corruption footgun: miss one and the entering bank
     * inherits another conversation's speculative state).
     * s->mseq_dirty is NOT saved: it describes the graph's scalar frontier
     * counters, not this bank's conversation, and _restore re-establishes
     * per-bank frontier truth and clears it unconditionally. */
    pulsar_spec_shadow_save(s, &c->spec, &c->pend_qrows, &c->pend_qrows_cap);
    c->valid = true;
}

bool pulsar_session::bank_state_restore(uint32_t bank) {
    auto *s = this;
    if (!s || bank >= gpu_graph_bank_pool_count(s->graph)) return false;
    /* Point device views (incl. Option F drafter ring) at this bank, and
     * re-arm its frontier counters — this is what makes clearing mseq_dirty
     * cheap and safe (per-bank truth is re-established without a re-prefill). */
    if (s->graph->banks.n_banks != 0 && !gpu_graph_bank_repoint(s->graph, bank))
        return false;
    gpu_graph_bank_counters_install(s->graph, bank);
    /* No saved host state for a fresh bank: the counters_install above set the
     * (zeroed) frontier; the session's host shadow stays as the caller primed it
     * (a fresh sync just ran).  Either way, per-bank frontier truth is now
     * installed, so the multiseq superset poison no longer applies to this bank. */
    (void)pulsar_bank_carry_restore_view(s, bank);
    s->mseq_dirty = false;
    return true;
}

bool pulsar_bank_carry_restore_view(pulsar_session *s, uint32_t bank) {
    if (!s || !s->bank_carry || bank >= s->bank_carry_n || !s->bank_carry[bank].valid) return false;
    const pulsar_bank_carry *c = &s->bank_carry[bank];
    pulsar_tokens_copy(&s->checkpoint, &c->checkpoint);
    memcpy(s->logits, c->logits, (size_t)s->engine->logits_width() * sizeof(float));
    s->checkpoint_valid       = c->checkpoint_valid;
    s->logits_stale           = c->logits_stale;
    s->prefill_frontier       = c->prefill_frontier;   /* L195 */
    s->live_images            = c->live_images;        /* L226 / L281 */
    /* Mirror of the save above: the whole shadow and its q rows. */
    pulsar_spec_shadow_restore(s, &c->spec, c->pend_qrows, c->pend_qrows_cap);
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
const pulsar_tokens *pulsar_bank_history(pulsar_session *s, uint32_t bank) {
    if (!s || bank >= session_bank_n(s)) return NULL;
    if (bank == pulsar_session_live_bank(s)) return s->checkpoint_valid ? &s->checkpoint : NULL;
    if (s->bank_carry && bank < s->bank_carry_n &&
        s->bank_carry[bank].valid && s->bank_carry[bank].checkpoint_valid)
        return &s->bank_carry[bank].checkpoint;
    return NULL;
}



/* L260 fusion / L284 #2: the record of a fused step's prompt chunk in the INSTALLED bank's history, for every
 * family -- what a classic sync of the same tokens leaves behind.  The tokens join the checkpoint (it started
 * empty, an invalidated bank, or as the valid prefix the prompt extends -- left invalid, the bank's frontier read
 * 0 and the next chunk restarted the prompt over rows it had already committed: "frontier not position-true ...
 * n_comp 61 want 0"); the prefill frontier moves to its end (L195); a chunk that ends on the resume grid leaves a
 * checkpoint, as the chunk loop takes one at every grid chunk end, so the fused lane's prompts resume like the
 * classic sync's (the server's planners end chunks on grid points) -- when the model's store says the state there
 * is a prefill's (pulsar_kv_state_ops::noted_at); and the finishing chunk's headed row becomes the session's
 * next-token logits.  The family's state already moved with the step; a record that disagrees with it refuses
 * before anything changes. */
int pulsar_session::note_prefilled(const int *toks, int n, int head) {
    auto *s = this;
    if (!toks || n <= 0) return 1;
    if (head >= 0 && (!s->fused_logits || (uint32_t)head >= s->fused_heads)) {
        fprintf(stderr, "pulsar: note_prefilled: head row %d of a fused step that headed %u -- refusing\n",
                head, s->fused_logits ? s->fused_heads : 0u);
        return 1;
    }
    pulsar_ckpt_store *ck = pulsar_session_kv_store(s);
    const uint32_t bank = pulsar_session_live_bank(s);
    const uint32_t from = (uint32_t)s->checkpoint.len, G = from + (uint32_t)n;
    /* the shared capture rule (P13): the state stands at G or the record is refused, and G on the grid in a
     * prefill's state is captured -- before the record changes anything */
    if (!ck || !ck->ops || !pulsar_ckpt_landed(ck, bank, G)) {
        fprintf(stderr, "pulsar: note_prefilled: bank %u's chunk [%u, %u) %s -- refusing\n", bank, from, G,
                ck && ck->ops ? "did not land (above)" : "has no checkpoint store");
        return 1;
    }
    if (from == 0) s->live_images.n = 0;   /* a history from 0 holds no image block (a fused chunk is text) */
    /* a prompt chunk is a request's, as a sync's prefill is: the lookahead and a latched quench are the
     * previous request's (pulsar_session_family_sync does the same for the classic lane) */
    spec_lookahead_reset(s);
    pulsar_session_note_committed(s, toks, n);
    s->checkpoint_valid = true;
    s->prefill_frontier = s->checkpoint.len;   /* L195: a prefill wrote up to here */
    if (head >= 0) {
        const size_t w = (size_t)s->engine->logits_width();
        memcpy(s->logits, s->fused_logits + (size_t)(s->fused_n_dec + (uint32_t)head) * w, w * sizeof(s->logits[0]));
        s->logits_stale = false;   /* the next-token row of the history, as a sync's last chunk leaves it */
    }
    return 0;
}

/* L264: how far the bank's PREFILL reached (decode rows past it are the decode
 * kernels'; L195) -- the end of the last prompt it served, the same live-vs-
 * carry rule as pulsar_bank_history.  0 when nothing valid. */
int pulsar_session::bank_prefill_frontier(uint32_t bank) {
    auto *s = this;
    if (!pulsar_bank_history(s, bank)) return 0;
    const int pf = bank == pulsar_session_live_bank(s) ? s->prefill_frontier : s->bank_carry[bank].prefill_frontier;
    return pf > 0 ? pf : 0;
}


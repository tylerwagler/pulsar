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

static bool bank_carry_ensure(pulsar_session *s);   /* defined below */

/* Set d's token contents to EXACTLY toks[0..n) (reusing d's owned buffer). The
 * fork paths use this to stamp a dst's committed frontier from the VALIDATED
 * request prefix, rather than relying on truncating a copied/live checkpoint —
 * that truncate silently no-ops when the target starts shorter than n (a fresh
 * dst==cur carry at len 0), which is exactly what made the server re-prefill
 * from 0 and skip zero work (the fork "fired" but delivered no TTFT gain). */
static void token_vec_set_prefix(token_vec *d, const int *toks, int n) {
    d->len = 0;
    for (int i = 0; i < n; i++) token_vec_push(d, toks[i]);
}

/* L115: stamp a fork destination's checkpoint from the LIVE history rather
 * than from the request's tokens.  With token-boundary seams the two agree
 * byte-for-byte while disagreeing in ids, and the checkpoint must describe
 * the KV that was actually cloned — which carries the boundaries the model
 * sampled.  Stamping from the request would label live KV with canonical
 * ids it was not built from.  Handles `d` aliasing `src`. */
static void token_vec_set_prefix_from(token_vec *d, const token_vec *src, int n) {
    if (d == src) { if (d->len > n) d->len = n; return; }
    token_vec_set_prefix(d, src->v, n);
}

/* Tier-2 PATH-A (plan-33 inc A) — deep-copy one bank carry into another (src's
 * conversation continues on dst). Reuses d's owned heap buffers; no alias/leak. */
void pulsar_bank_carry::copy(const pulsar_bank_carry *sc) {
    auto *d = this;
    token_vec d_ck = d->checkpoint;              /* keep d's own token buffer */
    float   *d_logits = d->logits;
    float   *d_qrows = d->dspark_pending_qrows;
    uint32_t d_qcap = d->dspark_pending_qrows_cap;
    *d = *sc;                                    /* scalars + (aliased) heap ptrs — fixed below */
    d->checkpoint = d_ck;
    pulsar_tokens_copy(&d->checkpoint, &sc->checkpoint);
    d->logits = d_logits ? d_logits : (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(float));
    if (sc->logits) memcpy(d->logits, sc->logits, (size_t)PULSAR_N_VOCAB * sizeof(float));
    d->dspark_pending_qrows = d_qrows;
    d->dspark_pending_qrows_cap = d_qcap;
    if (sc->dspark_pending_qrows && sc->dspark_pending_qrows_cap > 0) {
        if (d->dspark_pending_qrows_cap < sc->dspark_pending_qrows_cap) {
            d->dspark_pending_qrows = (float *)xrealloc(d->dspark_pending_qrows,
                (size_t)sc->dspark_pending_qrows_cap * sizeof(float));
            d->dspark_pending_qrows_cap = sc->dspark_pending_qrows_cap;
        }
        memcpy(d->dspark_pending_qrows, sc->dspark_pending_qrows,
               (size_t)sc->dspark_pending_qrows_cap * sizeof(float));
    }
}

/* Tier-2 PATH-A FULL-PREFIX FORK (plan-33 increment A). Clone bank `src`'s
 * committed KV + host carry into bank `dst` so dst continues src's conversation,
 * then the caller re-prefills only the divergent suffix. STRUCTURAL ANTI-
 * CONTAMINATION: memcmp the request's tokens[0..n_cached) against src's committed
 * history BEFORE any device write — a mismatch (or a shorter/absent history)
 * REFUSES (return non-zero) so the caller cold-prefills; the fork never runs on a
 * non-matching source. Increment A is full-prefix only: n_cached must equal src's
 * committed length (a shorter shared prefix is the partial case, increment C).
 * Pins src against the eviction guard for the duration of the clone. Returns 0 on
 * success, non-zero on refusal/failure (caller falls back to cold prefill). */
int pulsar_session::bank_fork(uint32_t src, uint32_t dst,
                          const int *tokens, int n_tokens, int n_cached) {
    auto *s = this;
    if (!s || !tokens || n_cached < 0) return 1;
    pulsar_gpu_graph *g = &s->graph;
    if (g->banks.n_banks == 0 || src >= g->banks.n_banks ||
        dst >= g->banks.n_banks || src == dst) return 1;
    /* 1. VALIDATE against src's committed history BEFORE any device write. */
    const uint32_t cur = g->banks.n_banks ? g->banks.cur_bank : 0u;
    const token_vec *hist = NULL;
    if (src == cur) {
        hist = s->checkpoint_valid ? &s->checkpoint : NULL;
    } else if (s->bank_carry && src < s->bank_carry_n &&
               s->bank_carry[src].valid && s->bank_carry[src].checkpoint_valid) {
        hist = &s->bank_carry[src].checkpoint;
    }
    if (!hist || (int)hist->len != n_cached) return 1;      /* full-prefix only */
    /* L115: validate by BYTES, not ids.  Generated text freezes the token
     * boundaries the model sampled into the live history; the client's echo
     * re-tokenizes the same bytes canonically, so an id-exact compare
     * refuses to fork a conversation onto ITS OWN history — measured
     * 2026-08-28 as `warm-advance-in-place refused (token-mismatch)` at 328k
     * and 390k, one of which cost a 108,360-token re-prefill.  Byte equality
     * keeps the anti-contamination guarantee exactly as strong: identical
     * text is the same conversation, whatever the boundaries. */
    {
        pulsar_prefix_match m;
        pulsar_tokens_prefix_match(s->engine, hist->v, hist->len,
                                   tokens, n_tokens, &m);
        if (m.live_cut != hist->len) return 1;              /* mismatch -> cold */
    }
    /* 2. Eviction pin src (the guard's victim picker must not free it mid-clone). */
    g->fork_pin[src] = 1u;
    /* 3. src must have physical to clone (inc A refuses an evicted source). */
    if (gpu_graph_bank_is_evicted(g, src)) { g->fork_pin[src] = 0u; return 1; }
    if (src == cur) gpu_graph_bank_counters_capture(g, src);  /* current frontier */
    /* 4-5. Device D2D clone (allocs dst physical if freed, mirrors counters). */
    if (!gpu_graph_bank_fork_copy(g, src, dst)) { g->fork_pin[src] = 0u; return 1; }
    /* 6. Copy src host carry -> dst so dst owns src's conversation state. */
    if (!bank_carry_ensure(s)) { g->fork_pin[src] = 0u; return 1; }
    if (src == cur) s->bank_state_save(src);      /* refresh from live session */
    if (src < s->bank_carry_n && dst < s->bank_carry_n) {
        s->bank_carry[dst].copy(&s->bank_carry[src]);
    }
    /* dst owns tokens[0..n_cached) as its committed frontier. SET it explicitly
     * from the validated request prefix (do not trust the copied carry alone). */
    if (dst < s->bank_carry_n) {
        pulsar_bank_carry *c = &s->bank_carry[dst];
        token_vec_set_prefix_from(&c->checkpoint, hist, n_cached);
        c->checkpoint_valid = true;
        c->valid = true;
    }
    /* BUG FIX (server dst==cur): provision_bank installs the fresh dst as cur and
     * invalidates the live session (s->checkpoint.len -> 0) BEFORE this fork runs.
     * bank_frontier_tokens(dst) then reads the LIVE s->checkpoint, so unless we
     * SET it here bank_pos(dst) stays 0 and the server re-prefills the WHOLE
     * prompt (fork fires, zero work skipped). Install dst's graph frontier and
     * stamp s->checkpoint to tokens[0..n_cached). */
    if (dst == cur) {
        gpu_graph_bank_counters_install(g, dst);
        token_vec_set_prefix_from(&s->checkpoint, hist, n_cached);
        s->checkpoint_valid = true;
        /* L195/L196: the live prefill frontier is the carry's twin while a bank
         * is current; provision's invalidate zeroed it, so a resume on the
         * forked bank found no grid point and re-prefilled the whole prompt
         * (dogfood 2026-09-06 17:41: 292k tokens for a 36-token turn).  Install
         * it from the carry dst just inherited, bounded by the fork frontier. */
        s->prefill_frontier = dst < s->bank_carry_n ? s->bank_carry[dst].prefill_frontier : 0;
        if (s->prefill_frontier > n_cached) s->prefill_frontier = n_cached;
        s->spec.spec_carry_valid = false;
        pulsar_spec_drop_pendings(&s->spec);
        s->mseq_dirty = false;
    }
    /* 7. The drafter ring is NOT cloned: zero dst's ring counters so the spec
     * path re-warms instead of reading another conversation's window. */
    for (int i = 0; i < 3; i++) g->ms_dspark_n_raw[dst][i] = 0u;
    g->ms_dspark_prompt_n[dst] = 0u;
    g->ms_dspark_prompt_lo[dst] = 0u;
    /* 8. (dst suffix KV-growth admission charge is the server's job in a later
     *    increment; the engine clone copies existing KV, touching nothing new.) */
    g->fork_pin[src] = 0u;
    return 0;
}

bool pulsar_session::bank_fork_pinned(uint32_t bank) const {
    auto *s = this;
    if (!s || bank >= gpu_graph_bank_pool_count(&s->graph)) return false;
    return s->graph.fork_pin[bank] != 0u;
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

const pulsar_tokens *pulsar_session::bank_tokens(uint32_t bank) {
    auto *s = this;
    return bank_frontier_tokens(s, bank);
}

int pulsar_session::bank_common_prefix(uint32_t bank,
                                   const pulsar_tokens *prompt) {
    auto *s = this;
    const pulsar_tokens *t = bank_frontier_tokens(s, bank);
    if (!t || !prompt) return 0;
    int n = t->len < prompt->len ? t->len : prompt->len;
    int i = 0;
    while (i < n && t->v[i] == prompt->v[i]) i++;
    return i;
}

/* L115: the prefix-reuse authority against one bank's committed history. */
void pulsar_session::bank_prefix_match(uint32_t bank, const pulsar_tokens *prompt,
                                       pulsar_prefix_match *out) {
    auto *s = this;
    out->live_cut = 0;
    out->prompt_cut = 0;
    out->seamed = false;
    const pulsar_tokens *t = bank_frontier_tokens(s, bank);
    if (!t || !prompt) return;
    pulsar_tokens_prefix_match(s->engine, t->v, t->len, prompt->v, prompt->len, out);
}

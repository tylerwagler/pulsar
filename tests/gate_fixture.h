/* Shared fixture for the multi-bank engine gates (fused_step_gate, mixed_zero_prefill_gate, prefill_chunk_census_probe,
 * and the declared-state readers bank_isolation_gate, kv_accounting_gate, mtp_spec_gate).  Before 2026-09-02 (L153)
 * each gate carried its own copy of: read the story prompt, tokenize it, and populate a bank -- repoint the
 * graph's views, invalidate, classic sync, capture the bank's frontier counters, take the argmax.  Three
 * copies of a fixture that MUST agree (the gates compare each other's shapes) with nothing enforcing it.
 *
 * L278: this is the ENGINE-INTERNAL part -- the helpers that reach into DeepSeek V4's session graph (s->graph,
 * gpu_graph_*), for the standalone gates that grade that family's mechanisms (none is in the runner since L357), and
 * the reader of a bank's declared state (state_kinds.h: every family on the core step driver, never DeepSeek), for the
 * gates that compare state bytes.  Everything that needs only the public API lives in tests/gate_util.h, included
 * here, so a family-generic gate that compares only logits takes that header and none of this.
 *
 * Header-only, static inline: the gates are single-TU programs. */
#pragma once

#include "gate_util.h"
#include "pulsar_engine_internal.h"

#include <string>
#include <vector>

/* L183: prefill toks[start, end) as a classic CONTINUATION at `start` -- the
 * chunked prefill entered at the checkpoint, which is what pulsar_session_sync
 * did before L183.  Since L183 the public sync recomputes a resume from the
 * last chunk-grid boundary (a resume is a cold prefill), so a gate that
 * compares the MIXED lane's prefill of [start, end) against the classic lane
 * on the SAME chunking needs this entry; through sync it would compare the
 * mixed continuation against a cold prefill and measure chunk dependence
 * instead of lane parity (that is what the chunk-neutrality gate measures).
 * Leaves the session's checkpoint at `end` and its logits at the last row. */
static inline bool gate_prefill_suffix_classic(pulsar_session *s, const pulsar_tokens *toks,
                                               int start, int end, char *err, size_t errlen) {
    if (!s || !toks || start < 0 || end <= start || end > toks->len) {
        snprintf(err, errlen, "classic suffix: bad range [%d, %d)", start, end);
        return false;
    }
    if (s->checkpoint.len != start) {
        snprintf(err, errlen, "classic suffix: checkpoint at %d, suffix starts at %d", s->checkpoint.len, start);
        return false;
    }
    if (!gpu_graph_prefill_chunked_range(s->graph, &s->engine->model, &s->engine->weights, toks,
                                         (uint32_t)start, (uint32_t)(end - start), s->dev.live, NULL)) {
        snprintf(err, errlen, "classic suffix prefill [%d, %d) failed", start, end);
        return false;
    }
    s->dev.host_stale = true;   /* the head wrote the live row on the device (L302) */
    pulsar_tokens p = *toks; p.len = end;
    pulsar_tokens_copy(&s->checkpoint, &p);
    s->checkpoint_valid = true;
    return true;
}

/* Populate bank `bank` of session s from the token view [v, v+len): repoint
 * the graph at the bank (when a pool exists), invalidate (no prefix reuse
 * across banks), classic-sync the prompt, capture the bank's frontier
 * counters, and hand back the next token (argmax) through *argtok when asked.
 * The session does not take ownership of v. */
static inline bool gate_populate_bank(pulsar_session *s, uint32_t bank, const int *v, int len,
                                      int *argtok, const char *what) {
    pulsar_gpu_graph *g = s->graph;
    char err[256];
    if (g->banks.n_banks && !gpu_graph_bank_repoint(g, bank)) {
        fprintf(stderr, "%s: bank %u repoint failed\n", what, bank);
        return false;
    }
    pulsar_session_invalidate(s);
    pulsar_tokens p = { .v = (int *)v, .len = len, .cap = len };
    if (pulsar_session_sync(s, &p, err, sizeof err) != 0) {
        fprintf(stderr, "%s: bank %u sync failed: %s\n", what, bank, err);
        return false;
    }
    gpu_graph_bank_counters_capture(g, bank);
    if (argtok) *argtok = pulsar_session_argmax(s);
    return true;
}

/* The pool must hold at least `need` banks; false + a message otherwise.  L357: through the public count -- the
 * DeepSeek graph it read (s->graph) is NULL on every other family, so this segfaulted there. */
static inline bool gate_pool_fits(pulsar_session *s, uint32_t need) {
    const uint32_t have = (uint32_t)pulsar_session_bank_count(s);
    if (have >= need) return true;
    fprintf(stderr, "pool too small: %u < %u (set PULSAR_MSEQ_BANKS)\n", have, need);
    return false;
}

/* ---- a bank's declared state (L357; was mtp_spec_gate's bank_state) ------------------------------------------- */

/** One piece of a bank's declared state: one (slot, decl), named "<decl>@<slot>" so a difference says where it is. */
struct gate_state_piece {
    std::string name;            ///< "<decl>@<slot>"
    std::vector<uint8_t> bytes;  ///< the bank's bytes of it
};

/** gate_bank_state's `rows` for the WHOLE bank: every decl's whole per-bank bytes, each pool's whole tensor (the rows
 *  past the frontier too, where a misrouted write of another bank's position would land). */
#define GATE_STATE_WHOLE UINT32_MAX

/** The declared state's layer slots (trunk then drafter), 0 for a session without one (DeepSeek). */
static inline uint32_t gate_state_slots(const pulsar_session *s) {
    return s && s->states ? s->states->plan->n_slots : 0u;
}

/** Bank `b`'s declared state, read off the device as the frontier `rows` describes it: slots [k0, k1), each resident
 *  decl's bank bytes (FIXED, RING whole; STAGED: the open block's rows below the frontier -- a rejected verify row past
 *  it is no state) and the whole rows below the frontier of each pool (a drafter slot trails the trunk by one row: its
 *  caller passes one row fewer).  rows == GATE_STATE_WHOLE reads every decl and pool whole.  The bank's pending fold
 *  rows are flushed first (state_kinds.h THE FOLD: a reader outside a trunk step flushes).  Empty when the session has
 *  no declared state or a read failed (gate_state_diff then names it). */
static inline std::vector<gate_state_piece> gate_bank_state(pulsar_session *s, uint32_t b, uint32_t k0, uint32_t k1,
                                                            uint32_t rows) {
    std::vector<gate_state_piece> out;
    if (!s || !s->states || b >= s->states->n_banks) return out;
    if (!pulsar_state_fold_flush(s->states, b) || !pulsar_gpu_synchronize()) return out;
    const pulsar_state_set *ss = s->states;
    const pulsar_state_plan *p = ss->plan;
    const bool whole = rows == GATE_STATE_WHOLE;
    bool ok = k1 <= p->n_slots;
    for (uint32_t k = k0; ok && k < k1; k++) {
        for (uint32_t id = 0; ok && id < p->n_decl; id++) {
            const pulsar_state_decl *d = &p->decl[id];
            if (!(p->slot[k] >> id & 1u)) continue;
            const pulsar_gpu_tensor *t;
            uint64_t off, n;
            if (d->kind == PULSAR_STATE_POOL) {
                t = pulsar_state_pool(ss, k, id, b);
                n = whole ? (t ? pulsar_gpu_tensor_bytes(t) : 0u) : (uint64_t)rows / d->tokens * d->bytes;
                off = 0;
            } else {
                n = d->kind == PULSAR_STATE_STAGED && !whole ? (uint64_t)(rows % d->tokens) * d->bytes
                                                             : pulsar_state_decl_bank_bytes(d);
                t = pulsar_state_tensor(ss, k, id);
                off = (uint64_t)b * pulsar_state_decl_bank_bytes(d);
            }
            gate_state_piece pc;
            pc.name = std::string(d->name) + "@" + std::to_string(k);
            pc.bytes.resize(n);
            ok = t && (n == 0 || pulsar_gpu_tensor_read(t, off, pc.bytes.data(), n) != 0);
            out.push_back(std::move(pc));
        }
    }
    if (!ok) out.clear();
    return out;
}

/** The pieces' bytes (*bytes, a's) and the first piece that differs: "" when a and b are byte-identical, else its name
 *  and the differing span ("the state did not read" when either is empty or their pieces do not pair up). */
static inline std::string gate_state_diff(const std::vector<gate_state_piece> &a,
                                          const std::vector<gate_state_piece> &b, size_t *bytes) {
    *bytes = 0;
    for (const gate_state_piece &p : a) *bytes += p.bytes.size();
    if (a.empty() || a.size() != b.size()) return "the state did not read";
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].name != b[i].name || a[i].bytes.size() != b[i].bytes.size()) return a[i].name;
        size_t at = 0, last = 0, n = 0;
        for (size_t x = 0; x < a[i].bytes.size(); x++)
            if (a[i].bytes[x] != b[i].bytes[x]) {
                if (n++ == 0) at = x;
                last = x;
            }
        if (n) return a[i].name + " (" + std::to_string(n) + " bytes from offset " + std::to_string(at) + " to " +
                      std::to_string(last) + " of " + std::to_string(a[i].bytes.size()) + ")";
    }
    return "";
}

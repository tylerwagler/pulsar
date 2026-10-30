/* state_kinds.cpp -- declared session state (state_kinds.h, L295 F3): the core's half of a family's state
 * declaration -- allocation and its price, the bank's physical residency, the kv-state answer the grid checkpoints,
 * segments and payloads read (kv_state.h), the verify capture and its commit, and the bank pool.
 *
 * Nothing here names a family.  What used to be Qwen's own copy of each (qwen_state_alloc, kv_state_qwen.cpp,
 * family_qwen_banks.cpp, the S4 spec rollback) is this file over Qwen's declaration (family_qwen.cpp
 * qwen_state_plan), walked in the same order, so its slot, segment and payload bytes are the ones it wrote. */
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <string.h>

static uint64_t ceil_div(uint64_t a, uint64_t b) { return (a + b - 1u) / b; }
static void *dptr(pulsar_gpu_tensor *t) { return pulsar_gpu_tensor_device_ptr(t); }
static pulsar_state_set *SS(void *state) { return (pulsar_state_set *)state; }
static bool has(uint32_t mask, uint32_t id) { return (mask >> id) & 1u; }

uint64_t pulsar_state_decl_bank_bytes(const pulsar_state_decl *d) {
    switch (d->kind) {
    case PULSAR_STATE_FIXED:  return d->bytes;
    case PULSAR_STATE_STAGED: return (uint64_t)d->tokens * d->bytes;
    case PULSAR_STATE_RING:   return (uint64_t)(d->tokens - 1u + PULSAR_STATE_VERIFY_ROWS) * d->bytes;
    case PULSAR_STATE_POOL:   break;
    }
    return 0;
}

uint32_t pulsar_state_plan_add(pulsar_state_plan *p, const pulsar_state_decl &d) {
    if (p->n_decl >= PULSAR_STATE_DECLS_MAX) return UINT32_MAX;   /* the create refuses the plan by name */
    p->decl[p->n_decl] = d;
    return p->n_decl++;
}

/* The bytes a capture row of decl d takes. */
static uint64_t capture_row_bytes(const pulsar_state_decl *d) {
    return d->capture == PULSAR_STATE_CAPTURE_SNAPSHOT ? d->bytes : d->capture_bytes;
}

/* A plan the core can run, or the first thing that is not (rule 9: refused by name at the session's create). */
static const char *plan_refusal(const pulsar_state_plan *p) {
    static char why[200];
    if (p->n_decl > PULSAR_STATE_DECLS_MAX || p->n_slots > PULSAR_STATE_SLOTS_MAX || p->n_trunk > p->n_slots)
        return "the table or the slots are past the core's bounds";
    if (p->resume_grid == 0u || p->ckpt_slots < 1u || p->ckpt_slots > PULSAR_CKPT_SLOTS_MAX)
        return "the resume grid is 0 or the checkpoint slots are outside [1, PULSAR_CKPT_SLOTS_MAX]";
    for (uint32_t id = 0; id < p->n_decl; id++) {
        const pulsar_state_decl &d = p->decl[id];
        const bool on_slot = [&] { for (uint32_t k = 0; k < p->n_slots; k++) if (has(p->slot[k], id)) return true; return false; }();
        const char *bad = NULL;
        if (d.bytes == 0u) bad = "has no bytes";
        else if ((d.kind == PULSAR_STATE_POOL || d.kind == PULSAR_STATE_STAGED || d.kind == PULSAR_STATE_RING) &&
                 d.tokens == 0u) bad = "has no tokens";
        else if ((d.kind == PULSAR_STATE_POOL || d.kind == PULSAR_STATE_STAGED) && p->resume_grid % d.tokens)
            bad = "has rows the resume grid does not close";
        else if (d.host && (d.kind != PULSAR_STATE_FIXED || on_slot)) bad = "is host state that is not session-wide FIXED";
        else if (!d.host && has(p->session, id)) bad = "is session-wide device state (a layer slot holds it)";
        else if (d.kind != PULSAR_STATE_POOL && !d.host && pulsar_state_decl_bank_bytes(&d) % sizeof(float))
            bad = "is not a whole number of floats a bank (the reset fills f32 zeros)";
        else if ((d.capture == PULSAR_STATE_CAPTURE_SNAPSHOT || d.capture == PULSAR_STATE_CAPTURE_DELTA) &&
                 (d.kind != PULSAR_STATE_FIXED || d.host)) bad = "captures rows but is not device FIXED state";
        else if (d.capture == PULSAR_STATE_CAPTURE_DELTA && (!d.commit || !d.capture_bytes)) bad = "is DELTA without a commit";
        else if (d.capture == PULSAR_STATE_CAPTURE_REPLAY && (!d.host || !d.replay)) bad = "is REPLAY without host state and a replay";
        else if (d.capture == PULSAR_STATE_CAPTURE_STAGE &&
                 (d.kind != PULSAR_STATE_STAGED || d.capture_off + d.bytes > d.capture_bytes))
            bad = "is STAGE but not STAGED state, or its block row is outside the record";
        if (bad) {
            snprintf(why, sizeof why, "decl %u '%s' %s", id, d.name ? d.name : "?", bad);
            return why;
        }
    }
    for (uint32_t k = 0; k < p->n_slots; k++)
        if (p->slot[k] >> p->n_decl) return "a slot names a decl past the table";
    if (p->session >> p->n_decl) return "the session set names a decl past the table";
    return NULL;
}

/* A POOL bank's capacity: the session's tokens rounded up to whole rows of every pool and whole blocks of every
 * staged block (the kernels address only whole ones; L251's "a whole number of indexer blocks"). */
static uint32_t pool_capacity(const pulsar_state_plan *p, uint32_t ctx) {
    uint64_t align = 1;
    for (uint32_t id = 0; id < p->n_decl; id++) {
        const pulsar_state_decl &d = p->decl[id];
        if (d.kind != PULSAR_STATE_POOL && d.kind != PULSAR_STATE_STAGED) continue;
        uint64_t a = align, b = d.tokens;
        while (b) { const uint64_t t = a % b; a = b; b = t; }
        align = align / a * d.tokens;
    }
    return (uint32_t)(ceil_div(ctx, align) * align);
}

/* Bank b's pools on slot k, each made where it is missing. */
static bool slot_bank_alloc(pulsar_state_set *ss, uint32_t k, uint32_t b) {
    const pulsar_state_plan *p = ss->plan;
    bool ok = true;
    for (uint32_t id = 0; ok && id < p->n_decl; id++) {
        if (!has(p->slot[k], id) || p->decl[id].kind != PULSAR_STATE_POOL) continue;
        pulsar_gpu_tensor *&t = ss->pool[k][id][b];
        if (!t) t = pulsar_gpu_tensor_alloc_managed(ceil_div(ss->ctx, p->decl[id].tokens) * p->decl[id].bytes);
        ok = t != NULL;
    }
    return ok;
}

static void slot_bank_free(pulsar_state_set *ss, uint32_t k, uint32_t b) {
    for (uint32_t id = 0; id < ss->plan->n_decl; id++) {
        if (!ss->pool[k][id]) continue;
        pulsar_gpu_tensor_free(ss->pool[k][id][b]);
        ss->pool[k][id][b] = NULL;
    }
}

void pulsar_state_bank_free(pulsar_state_set *ss, uint32_t bank) {
    for (uint32_t k = 0; k < ss->plan->n_trunk; k++) slot_bank_free(ss, k, bank);
    ss->kv_hw[bank] = 0;   /* its pages are gone; the restore's loads raise it again */
}

bool pulsar_state_bank_alloc(pulsar_state_set *ss, uint32_t bank) {
    for (uint32_t k = 0; k < ss->plan->n_trunk; k++) {
        if (!slot_bank_alloc(ss, k, bank)) {
            pulsar_state_bank_free(ss, bank);   /* never half-backed */
            return false;
        }
    }
    return true;
}

bool pulsar_state_bank_evicted(const pulsar_state_set *ss, uint32_t bank) {
    for (uint32_t k = 0; k < ss->plan->n_trunk; k++)
        for (uint32_t id = 0; id < ss->plan->n_decl; id++)
            if (ss->pool[k][id] && !ss->pool[k][id][bank]) return true;
    return false;
}

KvBytes pulsar_state_pool_bytes_at(const pulsar_state_set *ss, KvRows rows) {
    const pulsar_state_plan *p = ss->plan;
    uint64_t bytes = 0;   /* each pool's rows times its row's bytes: the conversion itself, so raw inside */
    for (uint32_t k = 0; k < p->n_slots; k++)
        for (uint32_t id = 0; id < p->n_decl; id++)
            if (has(p->slot[k], id) && p->decl[id].kind == PULSAR_STATE_POOL)
                bytes += ceil_div(rows.raw(), p->decl[id].tokens) * p->decl[id].bytes;
    return KvBytes{bytes};
}

void pulsar_state_destroy(pulsar_state_set *ss) {
    if (!ss) return;
    for (uint32_t k = 0; k < PULSAR_STATE_SLOTS_MAX; k++) {
        for (uint32_t id = 0; id < PULSAR_STATE_DECLS_MAX; id++) {
            pulsar_gpu_tensor_free(ss->t[k][id]);
            if (!ss->pool[k][id]) continue;
            for (uint32_t b = 0; b < ss->n_banks; b++) pulsar_gpu_tensor_free(ss->pool[k][id][b]);
            free(ss->pool[k][id]);
        }
    }
    for (uint32_t id = 0; id < PULSAR_STATE_DECLS_MAX; id++) {
        free(ss->host[id]);
        pulsar_gpu_tensor_free(ss->cap_rows[id]);
        pulsar_gpu_tensor_free(ss->cap_runs[id]);
        free(ss->cap_host[id]);
    }
    free(ss->bank_pos);
    free(ss->kv_hw);
    free(ss->prefill_pos);
    free(ss->frontier_stale);
    if (ss->ckpt.ops) pulsar_ckpt_release(&ss->ckpt);
    free(ss);
}

bool pulsar_state_reset_bank(pulsar_state_set *ss, uint32_t bank) {
    const pulsar_state_plan *p = ss->plan;
    bool ok = true;
    for (uint32_t k = 0; ok && k < p->n_slots; k++) {
        for (uint32_t id = 0; ok && id < p->n_decl; id++) {
            if (!has(p->slot[k], id) || p->decl[id].kind == PULSAR_STATE_POOL || !ss->t[k][id]) continue;
            const uint64_t bb = pulsar_state_decl_bank_bytes(&p->decl[id]);
            pulsar_gpu_tensor *v = pulsar_gpu_tensor_view(ss->t[k][id], (uint64_t)bank * bb, bb);
            ok = v && pulsar_gpu_tensor_fill_f32(v, 0.0f, bb / sizeof(float)) != 0;
            pulsar_gpu_tensor_free(v);
        }
    }
    for (uint32_t id = 0; id < p->n_decl; id++) {
        if (!has(p->session, id)) continue;
        void *h = pulsar_state_host(ss, id, bank);
        if (p->decl[id].init) p->decl[id].init(ss->engine, h);
        else memset(h, 0, p->decl[id].bytes);
    }
    pulsar_state_set_pos(ss, bank, 0);
    ss->prefill_pos[bank] = 0;
    ss->frontier_stale[bank] = false;
    if (p->drafter_at) p->drafter_at(ss->ud, bank, 0u);
    if (ss->ckpt.ops) pulsar_ckpt_drop_bank(&ss->ckpt, bank);   /* its history is gone */
    return ok;
}

uint64_t pulsar_state_capture_offset(const pulsar_state_set *ss, uint32_t slot, uint32_t id) {
    return (uint64_t)ss->ord[slot][id] * ss->verify_rows * capture_row_bytes(&ss->plan->decl[id]);
}

void *pulsar_state_capture_rows(const pulsar_state_set *ss, uint32_t slot, uint32_t id) {
    if (!ss->cap_rows[id]) return NULL;
    return (uint8_t *)dptr(ss->cap_rows[id]) + pulsar_state_capture_offset(ss, slot, id);
}

void pulsar_state_verify_begin(pulsar_state_set *ss, uint32_t n_runs, const uint32_t *run_first, const int32_t *bank,
                               const int32_t *pos) {
    const pulsar_state_plan *p = ss->plan;
    ss->n_runs = n_runs;
    for (uint32_t k = 0; k <= n_runs; k++) ss->run_first[k] = run_first[k];
    for (uint32_t k = 0; k < n_runs; k++) {
        ss->run_bank[k] = bank[run_first[k]];
        ss->run_pos0[k] = pos[run_first[k]];
        for (uint32_t id = 0; id < p->n_decl; id++)
            if (ss->cap_host[id])
                memcpy(ss->cap_host[id] + (size_t)k * p->decl[id].bytes,
                       pulsar_state_host(ss, id, (uint32_t)ss->run_bank[k]), p->decl[id].bytes);
    }
}

bool pulsar_state_verify_capture(pulsar_state_set *ss) {
    const pulsar_state_plan *p = ss->plan;
    bool ok = true;
    for (uint32_t k = 0; ok && k < p->n_trunk; k++) {
        for (uint32_t id = 0; ok && id < p->n_decl; id++) {
            if (!has(p->slot[k], id) || p->decl[id].capture != PULSAR_STATE_CAPTURE_STAGE) continue;
            const uint64_t blk = pulsar_state_decl_bank_bytes(&p->decl[id]);
            const uint64_t o = ss->ord[k][id];
            for (uint32_t r = 0; ok && r < ss->n_runs; r++)
                ok = pulsar_gpu_tensor_copy_async(ss->cap_runs[id], (o * ss->verify_rows + r) * blk, ss->t[k][id],
                                                  (uint64_t)ss->run_bank[r] * blk, blk) != 0;
        }
    }
    if (!ok) fprintf(stderr, "pulsar: %s: the verify capture of a staged block failed\n", p->name);
    return ok;
}

static bool commit_fail(const pulsar_state_set *ss, const char *what) {
    fprintf(stderr, "pulsar: %s: %s\n", ss->plan->name, what);
    return false;
}

bool pulsar_state_commit(pulsar_state_set *ss, uint32_t bank, const int32_t *tokens, uint32_t keep) {
    const pulsar_state_plan *p = ss->plan;
    uint32_t run = ss->n_runs;
    for (uint32_t k = 0; k < ss->n_runs; k++)
        if ((uint32_t)ss->run_bank[k] == bank) run = k;
    if (run == ss->n_runs) return commit_fail(ss, "a commit names a bank the last verify step did not carry");
    const uint32_t f = ss->run_first[run], R = ss->run_first[run + 1] - f;
    if (keep == 0 || keep > R) return commit_fail(ss, "a commit keeps 1 .. the run's rows");
    const uint64_t VR = ss->verify_rows;
    /* L288: a DELTA state advances by the kept rows -- the verify left it as it found it -- at every keep, a full
     * accept included */
    bool ok = true;
    for (uint32_t k = 0; ok && k < p->n_trunk; k++)
        for (uint32_t id = 0; ok && id < p->n_decl; id++)
            if (has(p->slot[k], id) && p->decl[id].capture == PULSAR_STATE_CAPTURE_DELTA)
                ok = p->decl[id].commit(dptr(ss->t[k][id]), bank,
                                        (const uint8_t *)dptr(ss->cap_rows[id]) +
                                            (ss->ord[k][id] * VR + f) * p->decl[id].capture_bytes,
                                        keep);
    if (!ok) return commit_fail(ss, "a verify commit of a delta state failed");
    if (keep == R) return true;
    const uint32_t p0 = (uint32_t)ss->run_pos0[run], pnext = p0 + keep;
    for (uint32_t k = 0; ok && k < p->n_trunk; k++) {
        for (uint32_t id = 0; ok && id < p->n_decl; id++) {
            const pulsar_state_decl &d = p->decl[id];
            if (!has(p->slot[k], id)) continue;
            const uint64_t o = ss->ord[k][id];
            if (d.capture == PULSAR_STATE_CAPTURE_SNAPSHOT) {
                ok = pulsar_gpu_tensor_copy_async(ss->t[k][id], (uint64_t)bank * d.bytes, ss->cap_rows[id],
                                                  (o * VR + f + (keep - 1u)) * d.bytes, d.bytes) != 0;
            } else if (d.capture == PULSAR_STATE_CAPTURE_STAGE) {
                /* the block as the step found it, then the kept rows of the block that is open at pnext */
                const uint64_t blk = pulsar_state_decl_bank_bytes(&d);
                ok = pulsar_gpu_tensor_copy_async(ss->t[k][id], (uint64_t)bank * blk, ss->cap_runs[id],
                                                  (o * VR + run) * blk, blk) != 0;
                const uint32_t blk0 = pnext / d.tokens * d.tokens;
                for (uint32_t pp = blk0 > p0 ? blk0 : p0; ok && pp < pnext; pp++)
                    ok = pulsar_gpu_tensor_copy_async(ss->t[k][id], (uint64_t)bank * blk + (pp % d.tokens) * d.bytes,
                                                      ss->cap_rows[id],
                                                      (o * VR + f + (pp - p0)) * d.capture_bytes + d.capture_off,
                                                      d.bytes) != 0;
            }
        }
    }
    if (!ok) return commit_fail(ss, "a verify rollback copy failed");
    /* host state: the bank's bytes before the step, advanced over the kept tokens */
    for (uint32_t id = 0; ok && id < p->n_decl; id++) {
        if (!ss->cap_host[id]) continue;
        void *h = pulsar_state_host(ss, id, bank);
        memcpy(h, ss->cap_host[id] + (size_t)run * p->decl[id].bytes, p->decl[id].bytes);
        ok = p->decl[id].replay(ss->engine, h, tokens, keep);
    }
    return ok;
}

/* ---- the kv-state answer (kv_state.h) over the plan ------------------------------------------------------------ */

/* One layout at a grid point and at a payload's frontier (`frontier` changes nothing): every non-pool lane of every
 * slot in the slot's id order, then the session's host decls -- the bank's slice of each. */
static bool decl_walk(void *state, int dir, bool, pulsar_gpu_tensor *slab, uint64_t slot_off, uint32_t,
                      uint64_t *bytes_out) {
    pulsar_state_set *ss = SS(state);
    const pulsar_state_plan *p = ss->plan;
    const uint32_t bank = ss->live_bank;
    uint64_t off = slot_off;
    bool ok = true;
    for (uint32_t k = 0; k < p->n_slots; k++) {
        for (uint32_t id = 0; id < p->n_decl; id++) {
            pulsar_gpu_tensor *t = ss->t[k][id];
            if (!has(p->slot[k], id) || p->decl[id].kind == PULSAR_STATE_POOL || !t || !ok) continue;
            const uint64_t bytes = pulsar_state_decl_bank_bytes(&p->decl[id]);
            if (dir == 0) ok = pulsar_gpu_tensor_copy_async(slab, off, t, (uint64_t)bank * bytes, bytes) != 0;
            else if (dir > 0) ok = pulsar_gpu_tensor_copy_async(t, (uint64_t)bank * bytes, slab, off, bytes) != 0;
            off += bytes;
        }
    }
    for (uint32_t id = 0; id < p->n_decl; id++) {
        if (!has(p->session, id)) continue;
        void *h = pulsar_state_host(ss, id, bank);
        const uint64_t bytes = p->decl[id].bytes;
        if (ok && dir == 0) ok = pulsar_gpu_tensor_write(slab, off, h, bytes) != 0;
        else if (ok && dir > 0) ok = pulsar_gpu_tensor_read(slab, off, h, bytes) != 0;
        off += bytes;
    }
    if (bytes_out) *bytes_out = off - slot_off;
    return ok;
}

static uint32_t decl_min_checkpoint(void *state) { return SS(state)->plan->resume_grid; }

static bool decl_stands_at(void *state, uint32_t G, char *why, size_t whylen) {
    pulsar_state_set *ss = SS(state);
    const uint32_t bank = ss->live_bank;
    if (ss->bank_pos[bank] != G || ss->prefill_pos[bank] != G) {
        snprintf(why, whylen, "bank %u holds %u tokens, %u of them prefilled, not exactly %u prefilled", bank,
                 ss->bank_pos[bank], ss->prefill_pos[bank], G);
        return false;
    }
    return true;
}

static bool decl_holds(void *state, uint32_t G) { return SS(state)->prefill_pos[SS(state)->live_bank] >= G; }

static bool decl_prepare_restore(void *state, uint32_t G) {
    pulsar_state_set *ss = SS(state);
    const uint32_t bank = ss->live_bank;
    /* the pools' rows below G are the history the checkpoint references; the walk brings every overwritten lane
     * back, so only the counters move here */
    pulsar_state_set_pos(ss, bank, G);
    ss->prefill_pos[bank] = G;
    if (ss->plan->drafter_at) ss->plan->drafter_at(ss->ud, bank, G);   /* L289: the walk writes the true tail */
    return true;   /* the logits: the caller's (s->logits_stale after a restore) */
}

static void decl_restored(void *state) { SS(state)->frontier_stale[SS(state)->live_bank] = false; }

static void decl_set_frontier_stale(void *state, uint32_t G) {
    pulsar_state_set *ss = SS(state);
    const uint32_t bank = ss->live_bank;
    pulsar_state_set_pos(ss, bank, G);
    ss->prefill_pos[bank] = G;
    ss->frontier_stale[bank] = true;
}

static bool decl_stale(void *state, uint32_t bank) {
    const pulsar_state_set *ss = SS(state);
    return bank < ss->n_banks && ss->frontier_stale[bank];
}

/* Bank `bank`'s pools of slots [k0, k1), each slot's in id order. */
static uint32_t slot_pools(pulsar_state_set *ss, uint32_t bank, uint32_t k0, uint32_t k1, pulsar_kv_pool *out,
                           uint32_t cap) {
    const pulsar_state_plan *p = ss->plan;
    uint32_t n = 0;
    for (uint32_t k = k0; k < k1; k++) {
        for (uint32_t id = 0; id < p->n_decl; id++) {
            const pulsar_state_decl &d = p->decl[id];
            if (!has(p->slot[k], id) || d.kind != PULSAR_STATE_POOL) continue;
            if (n < cap) out[n] = {ss->pool[k][id][bank], d.tokens, d.bytes, d.whole_rows};
            n++;
        }
    }
    return n;
}

/* Every trunk slot's pools; a drafter slot's trail a row, so no segment carries them (decl_trailing_pools). */
static uint32_t decl_pools(void *state, pulsar_kv_pool *out, uint32_t cap) {
    return slot_pools(SS(state), SS(state)->live_bank, 0, SS(state)->plan->n_trunk, out, cap);
}

/* L313: any bank's pools, trunk then drafter (decl_pools then decl_trailing_pools) */
static uint32_t decl_bank_pools(void *state, uint32_t bank, pulsar_kv_pool *out, uint32_t cap) {
    pulsar_state_set *ss = SS(state);
    if (bank >= ss->n_banks) return 0;
    return slot_pools(ss, bank, 0, ss->plan->n_slots, out, cap);
}

/* L284: the payload's frontier (session_payload.cpp).  At T the lanes describe T when the bank's counter is T, no
 * segment load left them stale, and the drafter's lane (its hook) describes T too. */
static bool decl_frontier_at(void *state, uint32_t T, char *why, size_t whylen) {
    pulsar_state_set *ss = SS(state);
    const uint32_t bank = ss->live_bank;
    if (ss->bank_pos[bank] != T || ss->frontier_stale[bank]) {
        snprintf(why, whylen, "bank %u holds %u tokens%s, not a live frontier at %u", bank, ss->bank_pos[bank],
                 ss->frontier_stale[bank] ? " (stale: a segment chain without its last restore)" : "", T);
        return false;
    }
    return !ss->plan->drafter_frontier || ss->plan->drafter_frontier(ss->ud, bank, T, why, whylen);
}

static bool decl_install_frontier(void *state, uint32_t T, uint32_t prefill) {
    pulsar_state_set *ss = SS(state);
    const uint32_t bank = ss->live_bank;
    pulsar_state_set_pos(ss, bank, T);
    ss->prefill_pos[bank] = prefill;
    if (ss->plan->drafter_at) ss->plan->drafter_at(ss->ud, bank, T);   /* the walk writes the true stage */
    return true;
}

static uint32_t decl_trailing_pools(void *state, pulsar_kv_pool *out, uint32_t cap) {
    return slot_pools(SS(state), SS(state)->live_bank, SS(state)->plan->n_trunk, SS(state)->plan->n_slots, out, cap);
}

/* A prompt chunk to T -- the core loop's, or a fused step's (L284 #2): the bank's counter is the chunk's; a capture
 * at T is the cold prefill's exactly when T ends the bank's prefill-only history and no segment load left the lanes
 * stale (decl_stands_at's own rule). */
static bool decl_noted_at(void *state, uint32_t T, bool *capture, char *why, size_t whylen) {
    pulsar_state_set *ss = SS(state);
    const uint32_t bank = ss->live_bank;
    if (ss->bank_pos[bank] != T) {
        snprintf(why, whylen, "bank %u holds %u tokens, the record says %u", bank, ss->bank_pos[bank], T);
        return false;
    }
    *capture = ss->prefill_pos[bank] == T && !ss->frontier_stale[bank];
    return true;
}

pulsar_state_set *pulsar_state_create(const void *engine, const pulsar_state_plan *plan, void *ud, uint32_t n_banks,
                                      uint32_t ctx, uint32_t verify_rows) {
    const char *why = plan_refusal(plan);
    if (!why && n_banks > PULSAR_MSEQ_MAX) why = "more banks than the checkpoint store holds (PULSAR_MSEQ_MAX)";
    if (!why && verify_rows > PULSAR_STATE_VERIFY_ROWS) why = "a verify capture past PULSAR_STATE_VERIFY_ROWS";
    if (why) {
        fprintf(stderr, "pulsar: %s: the state declaration is refused: %s\n", plan->name, why);
        return NULL;
    }
    pulsar_state_set *ss = (pulsar_state_set *)xcalloc(1, sizeof(*ss));
    ss->plan = plan;
    ss->engine = engine;
    ss->ud = ud;
    ss->n_banks = n_banks;
    ss->ctx = pool_capacity(plan, ctx);
    ss->verify_rows = verify_rows;
    const uint64_t nb = n_banks;
    bool ok = true;
    for (uint32_t k = 0; ok && k < plan->n_slots; k++) {
        /* Demand-paged pools first, one tensor per bank (L284 #3: a bank's pages return only when its own tensors
         * go), each sized from ss->ctx -- the capacity the ops view; then the slot's resident state, bank-major. */
        for (uint32_t id = 0; id < plan->n_decl; id++)
            if (has(plan->slot[k], id) && plan->decl[id].kind == PULSAR_STATE_POOL)
                ss->pool[k][id] = (pulsar_gpu_tensor **)xcalloc(n_banks, sizeof(pulsar_gpu_tensor *));
        for (uint32_t b = 0; ok && b < n_banks; b++) ok = slot_bank_alloc(ss, k, b);
        for (uint32_t id = 0; ok && id < plan->n_decl; id++) {
            if (!has(plan->slot[k], id) || plan->decl[id].kind == PULSAR_STATE_POOL) continue;
            ok = (ss->t[k][id] = pulsar_gpu_tensor_alloc(nb * pulsar_state_decl_bank_bytes(&plan->decl[id]))) != NULL;
        }
    }
    for (uint32_t id = 0; id < plan->n_decl; id++)
        if (has(plan->session, id)) ss->host[id] = (uint8_t *)xcalloc(n_banks, plan->decl[id].bytes);
    /* the verify capture: per decl, [the trunk slots carrying it][verify rows] */
    for (uint32_t id = 0; ok && verify_rows && id < plan->n_decl; id++) {
        const pulsar_state_decl &d = plan->decl[id];
        if (d.capture == PULSAR_STATE_CAPTURE_NONE) continue;
        if (d.capture == PULSAR_STATE_CAPTURE_REPLAY) {
            ss->cap_host[id] = (uint8_t *)xcalloc(verify_rows, d.bytes);
            continue;
        }
        uint32_t n_l = 0;
        for (uint32_t k = 0; k < plan->n_trunk; k++)
            if (has(plan->slot[k], id)) ss->ord[k][id] = (uint8_t)n_l++;
        if (!n_l) continue;
        const uint64_t rows = (uint64_t)n_l * verify_rows;
        if (d.capture == PULSAR_STATE_CAPTURE_STAGE)
            ok = (ss->cap_runs[id] = pulsar_gpu_tensor_alloc(rows * pulsar_state_decl_bank_bytes(&d))) != NULL;
        if (ok) ok = (ss->cap_rows[id] = pulsar_gpu_tensor_alloc(rows * capture_row_bytes(&d))) != NULL;
    }
    ss->bank_pos = (uint32_t *)xcalloc(n_banks, sizeof(uint32_t));
    ss->kv_hw = (uint32_t *)xcalloc(n_banks, sizeof(uint32_t));
    ss->prefill_pos = (uint32_t *)xcalloc(n_banks, sizeof(uint32_t));
    ss->frontier_stale = (bool *)xcalloc(n_banks, sizeof(bool));
    ss->ops = {
        /* .name               = */ plan->name,
        /* .resume_grid        = */ plan->resume_grid,
        /* .split_invariant    = */ plan->split_invariant,
        /* .ckpt_slots         = */ plan->ckpt_slots,
        /* .walk               = */ decl_walk,
        /* .min_checkpoint     = */ decl_min_checkpoint,
        /* .stands_at          = */ decl_stands_at,
        /* .holds              = */ decl_holds,
        /* .prepare_restore    = */ decl_prepare_restore,
        /* .restored           = */ decl_restored,
        /* .set_frontier_stale = */ decl_set_frontier_stale,
        /* .stale              = */ decl_stale,
        /* .pools              = */ decl_pools,
        /* .frontier_at        = */ decl_frontier_at,
        /* .install_frontier   = */ decl_install_frontier,
        /* .trailing_pools     = */ decl_trailing_pools,
        /* .noted_at           = */ decl_noted_at,
        /* .bank_pools         = */ decl_bank_pools,
    };
    /* the grid checkpoints, ckpt_slots x the walk's bytes a bank -- priced with the rest under the dry run */
    if (ok) ok = pulsar_ckpt_alloc(&ss->ckpt, &ss->ops, ss, n_banks);
    for (uint32_t b = 0; ok && b < n_banks; b++) {
        ok = pulsar_state_reset_bank(ss, b);
        if (!ok) fprintf(stderr, "pulsar: %s: could not clear bank %u's state\n", plan->name, b);
    }
    if (!ok) {
        fprintf(stderr, "pulsar: %s: session state allocation failed (%u banks x %u tokens)\n", plan->name, n_banks,
                ctx);
        pulsar_state_destroy(ss);
        return NULL;
    }
    return ss;
}

/* ---- the bank pool (family.h pulsar_family_bank_ops) over the declared state ----------------------------------
 *
 * The device side is per bank already (every bank has its own declared state); the HOST side is the session's one
 * view (checkpoint, logits), which belongs to `live_bank`: save copies it into the bank's carry, restore makes a
 * bank live and brings its carry back.  What a recurrent state cannot do is refused in engine_api.cpp (forks,
 * per-bank KV spill files), each with the value the server has a path for.
 *
 * PHYSICAL RESIDENCY (L284 #3) is DeepSeek's contract: a bank's trunk pools are its own managed tensors, so the
 * server's 2b guard spills a victim (its segment chain persisted, free_physical) and brings it back
 * (alloc_physical + the chain).  What stays resident is the rest: every non-pool decl, the checkpoint slots, and a
 * drafter slot's pools (no segment carries them, so no restore could bring them back). */

static int decl_bank_count(pulsar_session *s) { return s && s->states ? (int)s->states->n_banks : 0; }

/* The host view is the core's carry (session_banks.cpp, L272 P2). */
static void decl_bank_save(pulsar_session *s, uint32_t bank) {
    if (!s || !s->states || bank >= s->states->n_banks) return;
    pulsar_bank_carry_save_view(s, bank);
}

static bool decl_bank_restore(pulsar_session *s, uint32_t bank) {
    if (!s || !s->states || bank >= s->states->n_banks) return false;
    s->states->live_bank = bank;
    /* a bank nothing was saved for: the fresh view (its first sync resets the bank and prefills cold) */
    if (!pulsar_bank_carry_restore_view(s, bank)) pulsar_bank_carry_fresh_view(s);
    return true;
}

static pulsar_ckpt_store *decl_bank_kv_store(pulsar_session *s) { return s && s->states ? &s->states->ckpt : NULL; }

static uint32_t decl_bank_prefill_frontier(pulsar_session *s, uint32_t bank) {
    return s && s->states && bank < s->states->n_banks ? s->states->prefill_pos[bank] : 0u;
}

static uint32_t decl_bank_live(pulsar_session *s) { return s && s->states ? s->states->live_bank : 0u; }

/* L270: one bank's demand-paged share at ctx_size -- the managed bytes of the session's own dry create (the pools
 * are cudaMallocManaged: VA reserved, physical on touch). */
static KvBytes decl_demand_paged_bytes(pulsar_engine *e, int ctx_size) {
    uint64_t managed = 0;
    e->session_cost_bytes_banked(ctx_size, 1, &managed);
    return KvBytes{managed};
}

/* L270: a bank's touched pools are its high-water's rows (the pages stay resident once written, until
 * free_physical returns the bank's own tensors); a quantum of q tokens grows them by at most q rows. */
static KvBytes decl_bank_touched_kv_bytes(pulsar_session *s, uint32_t bank) {
    if (!s->states || bank >= s->states->n_banks) return KvBytes{};
    return pulsar_state_pool_bytes_at(s->states, KvRows{s->states->kv_hw[bank]});
}

static KvBytes decl_bank_growth_bytes(pulsar_session *s, KvRows q) {
    return s->states ? pulsar_state_pool_bytes_at(s->states, q) : KvBytes{};
}

/* L284 #3: only the live bank is refused -- sync / eval / the segment walk run on it; a bank the batched lane
 * steps is the server's to exclude (it restores a spilled bank before any step, and an op refuses a row of an
 * evicted bank by name). */
static bool decl_bank_free_physical(pulsar_session *s, uint32_t bank) {
    if (!s->states || bank >= s->states->n_banks) return false;
    if (bank == s->states->live_bank) {
        fprintf(stderr, "pulsar: %s: free_physical refuses bank %u: it is the live bank\n", s->states->plan->name, bank);
        return false;
    }
    pulsar_state_bank_free(s->states, bank);
    return true;
}

static bool decl_bank_alloc_physical(pulsar_session *s, uint32_t bank) {
    if (!s->states || bank >= s->states->n_banks) return false;
    return pulsar_state_bank_alloc(s->states, bank);
}

static bool decl_bank_is_evicted(const pulsar_session *s, uint32_t bank) {
    return s->states && bank < s->states->n_banks && pulsar_state_bank_evicted(s->states, bank);
}

/* every bank has its own state */
static bool decl_bank_pooled(const pulsar_session *s) { return s->states && s->states->n_banks > 0; }

const pulsar_family_bank_ops PULSAR_BANK_OPS_DECLARED = {
    /* .count              = */ decl_bank_count,
    /* .save               = */ decl_bank_save,
    /* .restore            = */ decl_bank_restore,
    /* .kv_store           = */ decl_bank_kv_store,
    /* .prefill_frontier   = */ decl_bank_prefill_frontier,
    /* .live               = */ decl_bank_live,
    /* .demand_paged_bytes = */ decl_demand_paged_bytes,
    /* .touched_kv_bytes   = */ decl_bank_touched_kv_bytes,
    /* .growth_bytes       = */ decl_bank_growth_bytes,
    /* .free_physical      = */ decl_bank_free_physical,
    /* .alloc_physical     = */ decl_bank_alloc_physical,
    /* .is_evicted         = */ decl_bank_is_evicted,
    /* .repoint            = */ NULL,   /* nothing on the device to point: the repoint is a restore */
    /* .rewind             = */ NULL,   /* a recurrent state has no rewind (no PULSAR_FAMILY_CAP_REWIND) */
    /* .restore_checkpoint = */ NULL,   /* a grid checkpoint is restored only as the sync resumes from it */
    /* .pooled             = */ decl_bank_pooled,
};

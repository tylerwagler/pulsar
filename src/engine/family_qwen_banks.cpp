/* family_qwen_banks.cpp -- the Qwen4-exp bank pool (L251): what the server's per-bank bookkeeping
 * needs from a family whose banks are not DeepSeek's graph pool (family.h pulsar_family_bank_ops).
 *
 * The device side is already per bank -- every bank has its own recurrent GDN state, QSA KV cache,
 * index tail, PLE conv state and n-gram context (pulsar_qwen_state), and the batched decode takes
 * one row per bank.  What was missing is the HOST side: the session carries ONE view (checkpoint,
 * logits) and sync / eval ran on bank 0.  Here the view belongs to `live_bank`; save copies it into
 * the bank's carry, restore makes a bank live and brings its carry back.  No device work at all.
 *
 * What a recurrent state cannot do is refused in engine_api.cpp, each with the value the server
 * already has a path for: forks (PULSAR_FORK_RING_SCROLLED -- permanently infeasible, so the router
 * takes a fresh bank or prefills cold in place), per-bank KV spill and physical residency (the
 * touched-KV count is 0, so the guard never spills). */
#include "pulsar_engine_internal.h"
#include "family_qwen.h"
#include "spec_internal.h"

#include <string.h>

static uint32_t qwen_bank_logits_width(const pulsar_session *s) {
    return s->engine->family->logits_width(s->engine);
}

static int qwen_bank_count(pulsar_session *s) {
    return s && s->qwen ? (int)s->qwen->n_banks : 0;
}

static void qwen_bank_save(pulsar_session *s, uint32_t bank) {
    if (!s || !s->qwen || bank >= s->qwen->n_banks) return;
    pulsar_qwen_bank_carry *c = &s->qwen->carry[bank];
    pulsar_tokens_copy(&c->checkpoint, &s->checkpoint);
    const uint32_t nv = qwen_bank_logits_width(s);
    if (!c->logits) c->logits = (float *)xmalloc((size_t)nv * sizeof(float));
    memcpy(c->logits, s->logits, (size_t)nv * sizeof(float));
    c->checkpoint_valid = s->checkpoint_valid;
    c->logits_stale = s->logits_stale;
    if (!c->spec) c->spec = (pulsar_spec_carry_state *)xcalloc(1, sizeof(*c->spec));
    pulsar_spec_shadow_save(s, c->spec, &c->pend_qrows, &c->pend_qrows_cap);   /* L272 P1 */
    c->valid = true;
}

/* L272 P1: a non-live bank's saved speculative shadow for the round API's readers. */
static const pulsar_spec_carry_state *qwen_bank_spec_carry(pulsar_session *s, uint32_t bank) {
    if (!s || !s->qwen || bank >= s->qwen->n_banks || !s->qwen->carry[bank].valid) return NULL;
    return s->qwen->carry[bank].spec;
}

static bool qwen_bank_restore(pulsar_session *s, uint32_t bank) {
    if (!s || !s->qwen || bank >= s->qwen->n_banks) return false;
    s->qwen->live_bank = bank;
    const pulsar_qwen_bank_carry *c = &s->qwen->carry[bank];
    if (!c->valid) {
        /* a bank nothing was saved for: an empty, INVALID view, so its first sync resets the bank
         * and prefills cold (never continues whatever the device state holds) */
        s->checkpoint.len = 0;
        s->checkpoint_valid = false;
        s->logits_stale = true;
        pulsar_spec_drop_pendings(&s->spec);   /* no shadow was saved: the fresh bank has no pendings or carry */
        s->spec.spec_carry_valid = false;
        return true;
    }
    pulsar_tokens_copy(&s->checkpoint, &c->checkpoint);
    memcpy(s->logits, c->logits, (size_t)qwen_bank_logits_width(s) * sizeof(float));
    s->checkpoint_valid = c->checkpoint_valid;
    s->logits_stale = c->logits_stale;
    pulsar_spec_shadow_restore(s, c->spec, c->pend_qrows, c->pend_qrows_cap);   /* L272 P1 */
    return true;
}

/* The per-bank history the router reads: the live bank's checkpoint, or another bank's carry --
 * the same live-vs-carry rule as DeepSeek's bank_frontier_tokens.  Pure host reads. */
static const pulsar_tokens *qwen_bank_tokens(pulsar_session *s, uint32_t bank) {
    if (!s || !s->qwen || bank >= s->qwen->n_banks) return NULL;
    if (bank == s->qwen->live_bank) return s->checkpoint_valid ? &s->checkpoint : NULL;
    const pulsar_qwen_bank_carry *c = &s->qwen->carry[bank];
    return c->valid && c->checkpoint_valid ? &c->checkpoint : NULL;
}

/* The batched lane fed `toks` to the live bank's device state; the host view records them.  The
 * logits are no longer the next-token row (the lane kept its own), and the view is valid exactly
 * when it agrees with what the bank's state holds. */
static void qwen_bank_note_committed(pulsar_session *s, const int *toks, int n) {
    if (!s || !s->qwen || !toks || n <= 0) return;
    for (int i = 0; i < n; i++) pulsar_tokens_push(&s->checkpoint, toks[i]);
    s->logits_stale = true;
    s->checkpoint_valid = s->qwen->bank_pos[s->qwen->live_bank] == (uint32_t)s->checkpoint.len;
}

static pulsar_ckpt_store *qwen_bank_kv_store(pulsar_session *s) { return s && s->qwen ? s->qwen->ckpt : NULL; }

static uint32_t qwen_bank_prefill_frontier(pulsar_session *s, uint32_t bank) {
    return s && s->qwen && bank < s->qwen->n_banks ? s->qwen->prefill_pos[bank] : 0u;
}

static uint32_t qwen_bank_live(pulsar_session *s) { return s && s->qwen ? s->qwen->live_bank : 0u; }

/* L270: the demand-paged KV accounting.  A bank's touched KV is its high-water's rows (the pages stay
 * resident once written: the banks share one managed tensor per layer); a quantum of q tokens grows it by
 * at most q rows (plus a partial index block). */
uint64_t qwen_demand_paged_bytes(pulsar_engine *e, int ctx_size);

static uint64_t qwen_bank_touched_kv_bytes(pulsar_session *s, uint32_t bank) {
    if (!s->qwen || bank >= s->qwen->n_banks) return 0;
    return qwen_kv_bytes_at(s->qwen, s->qwen->kv_hw[bank]);
}

static uint64_t qwen_bank_growth_bytes(pulsar_session *s, uint32_t q) {
    return s->qwen ? qwen_kv_bytes_at(s->qwen, q) : 0;
}

const pulsar_family_bank_ops k_qwen_bank_ops = {
    /* .count          = */ qwen_bank_count,
    /* .save           = */ qwen_bank_save,
    /* .restore        = */ qwen_bank_restore,
    /* .tokens         = */ qwen_bank_tokens,
    /* .note_committed = */ qwen_bank_note_committed,
    /* .kv_store         = */ qwen_bank_kv_store,
    /* .prefill_frontier = */ qwen_bank_prefill_frontier,
    /* .live             = */ qwen_bank_live,
    /* .demand_paged_bytes = */ qwen_demand_paged_bytes,
    /* .touched_kv_bytes   = */ qwen_bank_touched_kv_bytes,
    /* .growth_bytes       = */ qwen_bank_growth_bytes,
    /* .spec_carry         = */ qwen_bank_spec_carry,
};

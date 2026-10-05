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
    c->logits_fresh = s->qwen->logits_fresh;
    c->valid = true;
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
        s->qwen->logits_fresh = false;
        return true;
    }
    pulsar_tokens_copy(&s->checkpoint, &c->checkpoint);
    memcpy(s->logits, c->logits, (size_t)qwen_bank_logits_width(s) * sizeof(float));
    s->checkpoint_valid = c->checkpoint_valid;
    s->qwen->logits_fresh = c->logits_fresh;
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
    s->qwen->logits_fresh = false;
    s->checkpoint_valid = s->qwen->bank_pos[s->qwen->live_bank] == (uint32_t)s->checkpoint.len;
}

static pulsar_ckpt_store *qwen_bank_kv_store(pulsar_session *s) { return s && s->qwen ? s->qwen->ckpt : NULL; }

static uint32_t qwen_bank_prefill_frontier(pulsar_session *s, uint32_t bank) {
    return s && s->qwen && bank < s->qwen->n_banks ? s->qwen->prefill_pos[bank] : 0u;
}

static uint32_t qwen_bank_live(pulsar_session *s) { return s && s->qwen ? s->qwen->live_bank : 0u; }

const pulsar_family_bank_ops k_qwen_bank_ops = {
    /* .count          = */ qwen_bank_count,
    /* .save           = */ qwen_bank_save,
    /* .restore        = */ qwen_bank_restore,
    /* .tokens         = */ qwen_bank_tokens,
    /* .note_committed = */ qwen_bank_note_committed,
    /* .kv_store         = */ qwen_bank_kv_store,
    /* .prefill_frontier = */ qwen_bank_prefill_frontier,
    /* .live             = */ qwen_bank_live,
};

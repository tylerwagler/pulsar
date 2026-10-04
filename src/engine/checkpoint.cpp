/* L264 grid checkpoints, the model-neutral half (kv_state.h): per bank, a few slots each holding
 * "the state at grid point G" that the bank's append-only pools do not already hold -- what that
 * state IS, and how big, is the model's pulsar_kv_state_ops; here are only the slots, the
 * retention rule and the validity rule.
 *
 * Validity is the frontier: a checkpoint at G is good while its bank's history reaches G (the
 * pools' rows below G are append-only, so a slot references them instead of copying).  The model
 * calls pulsar_ckpt_frontier_lowered wherever its frontier falls, and an operation that replaces a
 * bank's history wholesale drops all of its checkpoints. */
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <string.h>

bool pulsar_ckpt_alloc(pulsar_ckpt_store *st, const pulsar_kv_state_ops *ops, void *state, uint32_t n_banks) {
    if (n_banks < 1u) n_banks = 1u;
    if (n_banks > PULSAR_MSEQ_MAX || ops->ckpt_slots < 1u || ops->ckpt_slots > PULSAR_CKPT_SLOTS_MAX ||
        ops->ckpt_recent < 1u || ops->ckpt_recent > ops->ckpt_slots || ops->resume_grid == 0u) return false;
    st->ops = ops;
    st->state = state;
    uint64_t slot = 0;
    (void)ops->walk(state, -1, NULL, 0, ops->min_checkpoint(state), &slot);
    /* The batched copies want 256-B aligned starts; keep every slot on one. */
    st->slot_bytes = (slot + 255u) & ~(uint64_t)255u;
    memset(st->pos, 0, sizeof(st->pos));
    for (uint32_t b = 0; b < n_banks; b++) {
        st->slab[b] = pulsar_gpu_tensor_alloc((uint64_t)ops->ckpt_slots * st->slot_bytes);
        if (!st->slab[b]) {
            fprintf(stderr, "pulsar: grid checkpoint slab for bank %u (%u x %llu bytes, %s) allocation failed\n",
                    b, ops->ckpt_slots, (unsigned long long)st->slot_bytes, ops->name);
            return false;
        }
    }
    return true;
}

void pulsar_ckpt_release(pulsar_ckpt_store *st) {
    for (uint32_t b = 0; b < PULSAR_MSEQ_MAX; b++) {
        pulsar_gpu_tensor_free(st->slab[b]);
        st->slab[b] = NULL;
    }
    memset(st->pos, 0, sizeof(st->pos));
}

/* The slot a capture at G writes: the one already holding G, else an empty one, else the victim
 * of the retention rule -- never one of the ckpt_recent newest, and among the older ones the
 * checkpoint whose removal opens the smallest gap between its neighbours (the newest is G itself,
 * about to land). */
static uint32_t ckpt_pick_slot(const pulsar_ckpt_store *st, uint32_t bank, uint32_t G) {
    const uint32_t n = st->ops->ckpt_slots;
    const uint32_t *pos = st->pos[bank];
    for (uint32_t s = 0; s < n; s++) if (pos[s] == G) return s;
    for (uint32_t s = 0; s < n; s++) if (pos[s] == 0u) return s;
    uint32_t order[PULSAR_CKPT_SLOTS_MAX] = {0};
    for (uint32_t s = 0; s < n; s++) order[s] = s;
    for (uint32_t i = 1; i < n; i++) {          /* ascending by position */
        const uint32_t v = order[i];
        uint32_t j = i;
        for (; j > 0 && pos[order[j - 1]] > pos[v]; j--) order[j] = order[j - 1];
        order[j] = v;
    }
    const uint32_t older = n - (st->ops->ckpt_recent - 1u);   /* G is the newest */
    uint32_t victim = order[0];
    uint32_t best_gap = UINT32_MAX;
    for (uint32_t i = 0; i < older; i++) {
        const uint32_t lo = i == 0 ? 0u : pos[order[i - 1]];
        const uint32_t hi = i + 1 < n ? pos[order[i + 1]] : G;
        if (hi - lo < best_gap) { best_gap = hi - lo; victim = order[i]; }
    }
    return victim;
}

static bool ckpt_grid_ok(const pulsar_ckpt_store *st, uint32_t bank, uint32_t G) {
    return bank < PULSAR_MSEQ_MAX && st->slab[bank] && G % st->ops->resume_grid == 0u &&
           G >= st->ops->min_checkpoint(st->state);
}

bool pulsar_ckpt_capture(pulsar_ckpt_store *st, uint32_t bank, uint32_t G) {
    if (!ckpt_grid_ok(st, bank, G)) {
        fprintf(stderr, "pulsar: grid checkpoint at %u on bank %u refused: not a grid point the slab can hold\n",
                G, bank);
        return false;
    }
    /* The precondition is what makes the copy the state AT G: every pool holds exactly the rows a
     * prefill to G writes, no more (a row above would be a decode's) and no fewer. */
    char why[192];
    if (!st->ops->stands_at(st->state, G, why, sizeof(why))) {
        fprintf(stderr, "pulsar: grid checkpoint at %u on bank %u refused: %s\n", G, bank, why);
        return false;
    }
    const uint32_t s = ckpt_pick_slot(st, bank, G);
    st->pos[bank][s] = 0u;   /* not a checkpoint until the copy is issued whole */
    if (!st->ops->walk(st->state, 0, st->slab[bank], (uint64_t)s * st->slot_bytes, G, NULL)) {
        fprintf(stderr, "pulsar: grid checkpoint at %u on bank %u: copy failed\n", G, bank);
        return false;
    }
    st->pos[bank][s] = G;
    return true;
}

bool pulsar_ckpt_restore(pulsar_ckpt_store *st, uint32_t bank, uint32_t G) {
    if (bank >= PULSAR_MSEQ_MAX || G == 0u) return false;
    uint32_t s = PULSAR_CKPT_SLOTS_MAX;
    for (uint32_t k = 0; k < st->ops->ckpt_slots; k++) if (st->pos[bank][k] == G) s = k;
    if (s == PULSAR_CKPT_SLOTS_MAX) return false;
    if (!st->ops->prepare_restore(st->state, G) ||
        !st->ops->walk(st->state, 1, st->slab[bank], (uint64_t)s * st->slot_bytes, G, NULL)) {
        fprintf(stderr, "pulsar: grid checkpoint restore to %u on bank %u: copy failed\n", G, bank);
        return false;
    }
    st->ops->restored(st->state);
    for (uint32_t k = 0; k < st->ops->ckpt_slots; k++)
        if (st->pos[bank][k] > G) st->pos[bank][k] = 0u;
    return true;
}

uint32_t pulsar_ckpt_best(const pulsar_ckpt_store *st, uint32_t bank, uint32_t limit) {
    if (bank >= PULSAR_MSEQ_MAX || !st->ops) return 0u;
    uint32_t best = 0u;
    for (uint32_t k = 0; k < st->ops->ckpt_slots; k++) {
        const uint32_t p = st->pos[bank][k];
        if (p != 0u && p <= limit && p > best) best = p;
    }
    return best;
}

void pulsar_ckpt_frontier_lowered(pulsar_ckpt_store *st, uint32_t bank, uint32_t tokens) {
    if (bank >= PULSAR_MSEQ_MAX) return;
    for (uint32_t k = 0; k < PULSAR_CKPT_SLOTS_MAX; k++)
        if (st->pos[bank][k] > tokens) st->pos[bank][k] = 0u;
}

void pulsar_ckpt_drop_bank(pulsar_ckpt_store *st, uint32_t bank) {
    if (bank < PULSAR_MSEQ_MAX) memset(st->pos[bank], 0, sizeof(st->pos[bank]));
}

bool pulsar_ckpt_locate(pulsar_ckpt_store *st, uint32_t bank, uint32_t G, pulsar_gpu_tensor **slab, uint64_t *off) {
    if (bank >= PULSAR_MSEQ_MAX || G == 0u) return false;
    for (uint32_t k = 0; k < st->ops->ckpt_slots; k++) {
        if (st->pos[bank][k] != G) continue;
        *slab = st->slab[bank];
        *off = (uint64_t)k * st->slot_bytes;
        return true;
    }
    return false;
}

bool pulsar_ckpt_claim(pulsar_ckpt_store *st, uint32_t bank, uint32_t G, pulsar_gpu_tensor **slab, uint64_t *off,
                       uint32_t *slot) {
    if (!ckpt_grid_ok(st, bank, G)) return false;
    const uint32_t k = ckpt_pick_slot(st, bank, G);
    st->pos[bank][k] = 0u;   /* filled by the caller, committed after */
    *slab = st->slab[bank];
    *off = (uint64_t)k * st->slot_bytes;
    *slot = k;
    return true;
}

void pulsar_ckpt_commit(pulsar_ckpt_store *st, uint32_t bank, uint32_t slot, uint32_t G) {
    if (bank < PULSAR_MSEQ_MAX && slot < st->ops->ckpt_slots) st->pos[bank][slot] = G;
}

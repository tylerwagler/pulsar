/* L264 grid checkpoints, the model-neutral half (kv_state.h): per bank, a few slots each holding
 * "the state at grid point G" that the bank's append-only pools do not already hold -- what that
 * state IS, and how big, is the model's pulsar_kv_state_ops; here are only the slots, the
 * retention rule and the validity rule.
 *
 * Validity is the frontier: a checkpoint at G is good while its bank's history reaches G (the
 * pools' rows below G are append-only, so a slot references them instead of copying).  The model
 * calls pulsar_ckpt_frontier_lowered wherever its frontier falls, and an operation that replaces a
 * bank's history wholesale drops all of its checkpoints.
 *
 * L289 / L311: the RECORDS -- a pool of exact-position states (the frontier walk at any T) every bank draws from: a
 * bank's prompt end and its last message endpoint.  The same validity rule; a record is replaced by the bank's next
 * one of its kind, else evicted least recently used when the pool is full. */
#include "pulsar_engine_internal.h"
#include "lib/pulsar_stall.h"

#include <stdio.h>
#include <string.h>

/* L289: a record's bytes: the model's frontier walk, aligned as a slot is.  Nothing else: every pool's rows below T
 * are never rewritten while the history reaches T, and a row T leaves open (a part-filled indexer block, the MTP
 * layer's lagging row) is written whole by the token that closes it, from the lanes the walk holds, before any step
 * reads it (qsa_block_keys_kernel). */
static uint64_t ckpt_end_bytes(const pulsar_ckpt_store *st) {
    uint64_t front = 0;
    (void)st->ops->walk(st->state, -1, true, NULL, 0, st->ops->min_checkpoint(st->state), &front);
    return (front + 255u) & ~(uint64_t)255u;
}

bool pulsar_ckpt_alloc(pulsar_ckpt_store *st, const pulsar_kv_state_ops *ops, void *state, uint32_t n_banks) {
    if (n_banks < 1u) n_banks = 1u;
    if (n_banks > PULSAR_MSEQ_MAX || ops->ckpt_slots < 1u || ops->ckpt_slots > PULSAR_CKPT_SLOTS_MAX ||
        ops->resume_grid == 0u) return false;
    st->ops = ops;
    st->state = state;
    uint64_t slot = 0;
    (void)ops->walk(state, -1, false, NULL, 0, ops->min_checkpoint(state), &slot);
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
    /* L289 / L311: the record pool, allocated with the ladder so a session's dry run prices it */
    const uint64_t end = ckpt_end_bytes(st);
    st->n_rec = pulsar_ckpt_records_for(n_banks);
    st->rec_clock = 0;
    for (uint32_t i = 0; i < st->n_rec; i++) {
        pulsar_ckpt_record *r = &st->rec[i];
        r->T = 0u;
        r->slab = pulsar_gpu_tensor_alloc(end);
        if (!r->slab) {
            fprintf(stderr, "pulsar: record %u of %u (%llu bytes, %s) allocation failed\n", i, st->n_rec,
                    (unsigned long long)end, ops->name);
            return false;
        }
        r->cap = end;
    }
    return true;
}

void pulsar_ckpt_release(pulsar_ckpt_store *st) {
    for (uint32_t b = 0; b < PULSAR_MSEQ_MAX; b++) {
        pulsar_gpu_tensor_free(st->slab[b]);
        st->slab[b] = NULL;
    }
    for (uint32_t i = 0; i < PULSAR_CKPT_RECORDS_MAX; i++) {
        pulsar_ckpt_record *r = &st->rec[i];
        pulsar_gpu_tensor_free(r->slab);
        free(r->logits);
        free(r->tokens);
        memset(r, 0, sizeof(*r));
    }
    st->n_rec = 0;
    memset(st->pos, 0, sizeof(st->pos));
}

/* The slot a capture at G writes: the one already holding G, else an empty one, else the victim
 * of THE ladder rule (P13, every family's; only the slot count n is the model's): the newest n / 2
 * checkpoints are kept whole -- the prompt ends of the turns a client continues or edits next -- and
 * among the older ones goes the checkpoint whose removal opens the smallest gap between its
 * neighbours, so they thin into a ladder reaching back into the history a client rewrites (L261:
 * tool results 12-20k tokens back).  DeepSeek's 16 keep 8 newest + G and thin 7; Qwen's 2 keep the
 * newest + G, i.e. the previous turn's prompt end survives the next one's. */
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
    const uint32_t older = n - n / 2u;   /* the candidates: all but the newest n / 2 (G lands above them) */
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

/* Capture the installed `bank`'s state at grid point G (pulsar_ckpt_landed's).  False (said) on a violated
 * precondition. */
static bool ckpt_capture(pulsar_ckpt_store *st, uint32_t bank, uint32_t G) {
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
    pulsar_stall_mark("grid checkpoint capture");
    st->pos[bank][s] = 0u;   /* not a checkpoint until the copy is issued whole */
    if (!st->ops->walk(st->state, 0, false, st->slab[bank], (uint64_t)s * st->slot_bytes, G, NULL)) {
        fprintf(stderr, "pulsar: grid checkpoint at %u on bank %u: copy failed\n", G, bank);
        return false;
    }
    st->pos[bank][s] = G;
    return true;
}

bool pulsar_ckpt_landed(pulsar_ckpt_store *st, uint32_t bank, uint32_t T) {
    bool capture = false;
    char why[192] = "";
    if (!st->ops->noted_at(st->state, T, &capture, why, sizeof(why))) {
        fprintf(stderr, "pulsar: bank %u's prompt chunk to %u is not its state: %s\n", bank, T, why);
        return false;
    }
    if (!capture || T % st->ops->resume_grid != 0u || T < st->ops->min_checkpoint(st->state)) return true;
    return ckpt_capture(st, bank, T);
}

/* The bank's records above `tokens` go (the frontier rule), and with `tokens` 0 every one of the bank's. */
static void ckpt_records_drop_above(pulsar_ckpt_store *st, uint32_t bank, uint32_t tokens) {
    for (uint32_t i = 0; i < st->n_rec; i++)
        if (st->rec[i].T != 0u && st->rec[i].bank == bank && st->rec[i].T > tokens) st->rec[i].T = 0u;
}

bool pulsar_ckpt_restore(pulsar_ckpt_store *st, uint32_t bank, uint32_t G) {
    if (bank >= PULSAR_MSEQ_MAX || G == 0u) return false;
    uint32_t s = PULSAR_CKPT_SLOTS_MAX;
    for (uint32_t k = 0; k < st->ops->ckpt_slots; k++) if (st->pos[bank][k] == G) s = k;
    if (s == PULSAR_CKPT_SLOTS_MAX) return false;
    if (!st->ops->prepare_restore(st->state, G) ||
        !st->ops->walk(st->state, 1, false, st->slab[bank], (uint64_t)s * st->slot_bytes, G, NULL)) {
        fprintf(stderr, "pulsar: grid checkpoint restore to %u on bank %u: copy failed\n", G, bank);
        return false;
    }
    st->ops->restored(st->state);
    for (uint32_t k = 0; k < st->ops->ckpt_slots; k++)
        if (st->pos[bank][k] > G) st->pos[bank][k] = 0u;
    ckpt_records_drop_above(st, bank, G);
    return true;
}

/* The record a capture writes (the bank's previous one of its kind already dropped): a free one, else the least
 * recently used. */
static pulsar_ckpt_record *ckpt_record_pick(pulsar_ckpt_store *st) {
    pulsar_ckpt_record *lru = NULL;
    for (uint32_t i = 0; i < st->n_rec; i++) {
        pulsar_ckpt_record *r = &st->rec[i];
        if (r->T == 0u) return r;
        if (!lru || r->used < lru->used) lru = r;
    }
    return lru;
}

bool pulsar_ckpt_record_capture(pulsar_ckpt_store *st, uint32_t bank, pulsar_ckpt_record_kind kind, uint32_t T,
                                const int *tokens, const float *logits, uint32_t width) {
    if (bank >= PULSAR_MSEQ_MAX || T == 0u || !tokens || st->n_rec == 0u ||
        (kind == PULSAR_CKPT_REC_END) != (logits != NULL && width != 0u))
        return false;
    /* the bank's previous record of this kind describes an older position: gone whatever happens below */
    for (uint32_t i = 0; i < st->n_rec; i++)
        if (st->rec[i].T != 0u && st->rec[i].bank == bank && st->rec[i].kind == kind) st->rec[i].T = 0u;
    char why[192];
    if (!st->ops->frontier_at(st->state, T, why, sizeof(why))) {
        fprintf(stderr, "pulsar: %s at %u on bank %u not recorded: %s\n",
                kind == PULSAR_CKPT_REC_END ? "prompt end" : "message endpoint", T, bank, why);
        return false;
    }
    pulsar_ckpt_record *r = ckpt_record_pick(st);
    r->T = 0u;   /* not a record until the copy is issued whole (an evicted LRU record is gone from here) */
    const uint64_t need = ckpt_end_bytes(st);
    if (r->cap < need) {   /* a drafter loaded after the store adds its rings to the frontier */
        pulsar_gpu_tensor_free(r->slab);
        r->slab = pulsar_gpu_tensor_alloc(need);
        r->cap = r->slab ? need : 0u;
        if (!r->slab) {
            fprintf(stderr, "pulsar: record for bank %u (%llu bytes) allocation failed\n", bank,
                    (unsigned long long)need);
            return false;
        }
    }
    if (logits) {
        if (st->end_width != width) {   /* one logits width a store (the engine's) */
            for (uint32_t i = 0; i < st->n_rec; i++) {
                free(st->rec[i].logits);
                st->rec[i].logits = NULL;
                if (st->rec[i].kind == PULSAR_CKPT_REC_END) st->rec[i].T = 0u;
            }
            st->end_width = width;
        }
        if (!r->logits) r->logits = (float *)xmalloc((size_t)width * sizeof(float));
    } else {
        free(r->logits);   /* a recycled prompt end's row is no endpoint's */
        r->logits = NULL;
    }
    if (r->tokens_cap < T) {
        free(r->tokens);
        r->tokens = (int32_t *)xmalloc((size_t)T * sizeof(int32_t));
        r->tokens_cap = T;
    }
    if (!st->ops->walk(st->state, 0, true, r->slab, 0, T, NULL)) {
        fprintf(stderr, "pulsar: record at %u on bank %u: copy failed\n", T, bank);
        return false;
    }
    if (logits) memcpy(r->logits, logits, (size_t)width * sizeof(float));
    for (uint32_t i = 0; i < T; i++) r->tokens[i] = (int32_t)tokens[i];
    r->bank = bank;
    r->kind = kind;
    r->used = ++st->rec_clock;
    r->T = T;
    return true;
}

const pulsar_ckpt_record *pulsar_ckpt_record_of(const pulsar_ckpt_store *st, uint32_t bank,
                                                pulsar_ckpt_record_kind kind) {
    for (uint32_t i = 0; st && i < st->n_rec; i++)
        if (st->rec[i].T != 0u && st->rec[i].bank == bank && st->rec[i].kind == kind) return &st->rec[i];
    return NULL;
}

/* Are the record's ids tokens[0, T)? */
static bool ckpt_record_matches(const pulsar_ckpt_record *r, const int *tokens, uint32_t len) {
    if (r->T == 0u || !tokens || len < r->T) return false;
    for (uint32_t i = 0; i < r->T; i++)
        if ((int32_t)tokens[i] != r->tokens[i]) return false;
    return true;
}

const pulsar_ckpt_record *pulsar_ckpt_record_best(const pulsar_ckpt_store *st, uint32_t bank, const int *tokens,
                                                  uint32_t len, uint32_t limit, uint32_t prompt_len) {
    const pulsar_ckpt_record *best = NULL;
    for (uint32_t i = 0; st && i < st->n_rec; i++) {
        const pulsar_ckpt_record *r = &st->rec[i];
        if (r->T == 0u || r->bank != bank || r->T > limit || (best && r->T <= best->T)) continue;
        if (r->T == prompt_len && !r->logits) continue;   /* a resume AT T reads the row the prefill left there */
        if (ckpt_record_matches(r, tokens, len)) best = r;
    }
    return best;
}

bool pulsar_ckpt_record_restore(pulsar_ckpt_store *st, uint32_t bank, const pulsar_ckpt_record *r, float *logits,
                                uint32_t width) {
    if (!r || r < st->rec || r >= st->rec + st->n_rec || r->T == 0u || r->bank != bank ||
        (logits && (!r->logits || width != st->end_width)))
        return false;
    const uint32_t T = r->T;
    pulsar_ckpt_record *w = &st->rec[r - st->rec];
    /* the counters at T, prefilled to T (Qwen: the MTP layer's pending row at T - 1, its stage clean); the walk
     * brings every lane back */
    if (!st->ops->install_frontier(st->state, T, T) || !st->ops->walk(st->state, 1, true, w->slab, 0, T, NULL)) {
        fprintf(stderr, "pulsar: record restore to %u on bank %u: copy failed\n", T, bank);
        w->T = 0u;
        return false;
    }
    st->ops->restored(st->state);
    for (uint32_t k = 0; k < st->ops->ckpt_slots; k++)
        if (st->pos[bank][k] > T) st->pos[bank][k] = 0u;
    ckpt_records_drop_above(st, bank, T);
    w->used = ++st->rec_clock;
    if (logits) memcpy(logits, w->logits, (size_t)width * sizeof(float));
    return true;
}

/* L313: the most pools any bank has (bank_pools' out) */
#define CKPT_ADOPT_POOLS (2u * PULSAR_KV_POOLS_MAX)

bool pulsar_ckpt_adopt(pulsar_ckpt_store *st, uint32_t dst, uint32_t src, uint32_t T, const pulsar_ckpt_record *r) {
    const char *why = NULL;
    pulsar_gpu_tensor *slab = NULL;
    uint64_t off = 0;
    if (dst >= PULSAR_MSEQ_MAX || src >= PULSAR_MSEQ_MAX || dst == src || T == 0u || !st->ops->bank_pools)
        why = "not two banks of the pool";
    else if (r) {
        if (r < st->rec || r >= st->rec + st->n_rec || r->bank != src || r->T != T) why = "the record is not the source's at T";
        else slab = r->slab;
    } else if (!pulsar_ckpt_locate(st, src, T, &slab, &off)) {
        why = "the source holds no grid checkpoint at T";
    }
    static pulsar_kv_pool sp[CKPT_ADOPT_POOLS], dp[CKPT_ADOPT_POOLS];   /* worker thread only, as every store op */
    uint32_t n = 0;
    if (!why) {
        n = st->ops->bank_pools(st->state, src, sp, CKPT_ADOPT_POOLS);
        const uint32_t m = st->ops->bank_pools(st->state, dst, dp, CKPT_ADOPT_POOLS);
        if (n == 0 || n != m || n > CKPT_ADOPT_POOLS) why = "the banks' pools do not pair";
        for (uint32_t i = 0; !why && i < n; i++) {
            const uint64_t rows = (T + sp[i].tokens_per_row - 1u) / sp[i].tokens_per_row;
            if (!sp[i].rows || !dp[i].rows) why = "a bank's physical is freed";
            else if (sp[i].tokens_per_row != dp[i].tokens_per_row || sp[i].row_bytes != dp[i].row_bytes ||
                     rows * dp[i].row_bytes > pulsar_gpu_tensor_bytes(dp[i].rows) ||
                     rows * sp[i].row_bytes > pulsar_gpu_tensor_bytes(sp[i].rows))
                why = "a pool's rows do not fit";
        }
    }
    if (why) {
        fprintf(stderr, "pulsar: adopting bank %u's %s at %u into bank %u refused: %s\n", src,
                r ? "record" : "checkpoint", T, dst, why);
        return false;
    }
    /* dst's history is replaced wholesale: its checkpoints and records go; then the counters at T, the source's rows
     * below T, the source's state at T -- the same three moves as a restore on the source itself */
    pulsar_ckpt_drop_bank(st, dst);
    bool ok = r ? st->ops->install_frontier(st->state, T, T) : st->ops->prepare_restore(st->state, T);
    for (uint32_t i = 0; ok && i < n; i++) {
        const uint64_t rows = (T + sp[i].tokens_per_row - 1u) / sp[i].tokens_per_row;
        ok = pulsar_gpu_tensor_copy_async(dp[i].rows, 0, sp[i].rows, 0, rows * sp[i].row_bytes) != 0;
    }
    ok = ok && st->ops->walk(st->state, 1, r != NULL, slab, off, T, NULL);
    if (!ok) {
        fprintf(stderr, "pulsar: adopting bank %u's state at %u into bank %u: copy failed\n", src, T, dst);
        return false;
    }
    st->ops->restored(st->state);
    /* and dst keeps the checkpoint it now stands at, so its own next resume finds it */
    if (!r) {
        pulsar_gpu_tensor *ds = NULL;
        uint64_t doff = 0;
        uint32_t slot = 0;
        if (pulsar_ckpt_claim(st, dst, T, &ds, &doff, &slot) &&
            pulsar_gpu_tensor_copy_async(ds, doff, slab, off, st->slot_bytes) != 0)
            pulsar_ckpt_commit(st, dst, slot, T);
    } else {
        /* L326: lending the record is a use of it -- stamped before dst's copy picks its slot, so neither that copy
         * nor the next capture evicts the record a bank was just served from (it was the LRU victim first) */
        pulsar_ckpt_record *lent = &st->rec[r - st->rec];
        lent->used = ++st->rec_clock;
        pulsar_ckpt_record *w = ckpt_record_pick(st);
        if (w != r && w->cap >= r->cap) {
            w->T = 0u;
            if (r->logits) {
                if (!w->logits) w->logits = (float *)xmalloc((size_t)st->end_width * sizeof(float));
                memcpy(w->logits, r->logits, (size_t)st->end_width * sizeof(float));
            } else {
                free(w->logits);
                w->logits = NULL;
            }
            if (w->tokens_cap < T) {
                free(w->tokens);
                w->tokens = (int32_t *)xmalloc((size_t)T * sizeof(int32_t));
                w->tokens_cap = T;
            }
            memcpy(w->tokens, r->tokens, (size_t)T * sizeof(int32_t));
            if (pulsar_gpu_tensor_copy_async(w->slab, 0, r->slab, 0, r->cap) != 0) {
                w->bank = dst;
                w->kind = r->kind;
                w->used = ++st->rec_clock;
                w->T = T;
            }
        }
    }
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
    ckpt_records_drop_above(st, bank, tokens);
}

void pulsar_ckpt_drop_bank(pulsar_ckpt_store *st, uint32_t bank) {
    if (bank >= PULSAR_MSEQ_MAX) return;
    memset(st->pos[bank], 0, sizeof(st->pos[bank]));
    ckpt_records_drop_above(st, bank, 0u);
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

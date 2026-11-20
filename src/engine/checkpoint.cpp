/** @file checkpoint.cpp
 *  @brief L264 grid checkpoints, the model-neutral half (kv_state.h): per-bank slots, retention and validity. */
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
 * one of its kind, else evicted least recently used when the pool is full.
 *
 * L347 I4: the slots and records live where pulsar_ckpt_host_tier puts them, decided once at alloc (kv_state.h,
 * pulsar_ckpt_store::host).  Every copy between the state and a slot or record goes through ckpt_out / ckpt_in /
 * ckpt_copy below, the only functions that read the placement; the device tier's arm of each is the copy the store
 * made before I4. */
#include "pulsar_engine_internal.h"
#include "lib/pulsar_stall.h"

#include <stdio.h>
#include <string.h>

/* L289: a record's bytes: the model's frontier walk, aligned as a slot is.  Nothing else: every pool's rows below T
 * are never rewritten while the history reaches T, and a row T leaves open (a part-filled indexer block, the MTP
 * layer's lagging row) is written whole by the token that closes it, from the lanes the walk holds, before any step
 * reads it (qsa_block_keys_kernel). */
/// @param st the store (bound to its ops)
/// @return one record slab's bytes: the frontier walk at min_checkpoint, rounded up to 256
static TensorBytes ckpt_end_bytes(const pulsar_ckpt_store *st) {
    TensorBytes front{};
    (void)st->ops->walk(st->state, -1, true, NULL, SlabOff{}, st->ops->min_checkpoint(st->state), &front);
    return pulsar_bytes_align(front, 256u);
}

/* ---- L347 I4: the host tier's tickets and copies ---------------------------------------------------------------
 * The lane's ticket table: the staging buffers first, then every (bank, slot), then every record -- fixed indices, so
 * a ticket names the bytes it guards for the store's life. */
/** Tickets a host-tier store's lane holds. */
#define CKPT_TICKETS (PULSAR_CKPT_STAGES + PULSAR_MSEQ_MAX * PULSAR_CKPT_SLOTS_MAX + PULSAR_CKPT_RECORDS_MAX)

/// @param bank the bank
/// @param s    the slot
/// @return the ticket of host slot (bank, s)
static uint32_t ckpt_ticket_slot(uint32_t bank, uint32_t s) {
    return PULSAR_CKPT_STAGES + bank * PULSAR_CKPT_SLOTS_MAX + s;
}

/// @param st the store
/// @param r  one of its records
/// @return the record's ticket
static uint32_t ckpt_ticket_rec(const pulsar_ckpt_store *st, const pulsar_ckpt_record *r) {
    return PULSAR_CKPT_STAGES + PULSAR_MSEQ_MAX * PULSAR_CKPT_SLOTS_MAX + (uint32_t)(r - st->rec);
}

/** Where a slot's or a record's bytes are: one tier's half set, the other's NULL. */
struct ckpt_place {
    pulsar_gpu_tensor *slab;   ///< device tier: the tensor
    SlabOff off;               ///< device tier: the byte offset in it
    uint8_t *host;             ///< host tier: the pinned bytes
    uint32_t ticket;           ///< host tier: the bytes' ticket
    TensorBytes bytes;         ///< the bytes a host-tier copy moves
};

/// @param st   the store
/// @param bank the bank
/// @param s    the slot
/// @return where slot s of `bank` is
static ckpt_place ckpt_slot_place(const pulsar_ckpt_store *st, uint32_t bank, uint32_t s) {
    const SlabOff off = SlabOff::after(st->slot_bytes.times(s));   /* slot s sits at s * slot_bytes */
    if (st->host)
        return {NULL, SlabOff{}, pulsar_slab_at(st->host_slab[bank], off), ckpt_ticket_slot(bank, s), st->slot_bytes};
    return {st->slab[bank], off, NULL, 0u, st->slot_bytes};
}

/// @param st the store
/// @param r  one of its records
/// @return where the record is
static ckpt_place ckpt_rec_place(const pulsar_ckpt_store *st, const pulsar_ckpt_record *r) {
    if (st->host) return {NULL, SlabOff{}, r->host, ckpt_ticket_rec(st, r), r->cap};
    return {r->slab, SlabOff{}, NULL, 0u, r->cap};
}

/* Grow the host tier's staging buffers to `need` bytes (a record grew: a drafter loaded after the store).  The old
 * buffers' last device -> host copies are waited for first; their free waits for the stream. */
/// @param st   the store (host tier)
/// @param need bytes a copy will move through a staging buffer
/// @return false (said) on a failed ticket or allocation
static bool ckpt_stage_fit(pulsar_ckpt_store *st, TensorBytes need) {
    if (st->stage_bytes >= need) return true;
    for (uint32_t k = 0; k < PULSAR_CKPT_STAGES; k++) {
        if (st->stage[k] && pulsar_gpu_lane_wait(st->lane, k) == 0) return false;
        pulsar_gpu_tensor_free(st->stage[k]);
        st->stage[k] = pulsar_gpu_tensor_alloc(need);
        if (!st->stage[k]) {
            fprintf(stderr, "pulsar: checkpoint staging buffer %u (%llu bytes, %s) allocation failed\n", k,
                    (unsigned long long)need.raw(), st->ops->name);
            st->stage_bytes = TensorBytes{};
            return false;
        }
    }
    st->stage_bytes = need;
    return true;
}

/* The installed bank's state at G -> `p` (`frontier`: the record walk).  Device tier: the walk into the slab on the
 * session stream.  Host tier: the walk into the next staging buffer on the session stream -- once that buffer's last
 * device -> host copy has read it, a wait queued on the stream, not the host -- then the lane copies the buffer to
 * p.host behind whatever the session queues next, and records p's ticket and the buffer's.  The request goes on at
 * once: whoever reads p later waits on p's ticket. */
/// @param st       the store
/// @param p        the destination
/// @param frontier the record walk (false: the grid walk)
/// @param G        the position, in tokens
/// @return false on a failed walk or copy
static bool ckpt_out(pulsar_ckpt_store *st, const ckpt_place &p, bool frontier, uint32_t G) {
    if (!st->host) return st->ops->walk(st->state, 0, frontier, p.slab, p.off, pulsar_ckpt_pos(G), NULL);
    if (!ckpt_stage_fit(st, p.bytes)) return false;
    const uint32_t k = st->stage_next++ % PULSAR_CKPT_STAGES;
    const uint32_t done[2] = {p.ticket, k};
    return pulsar_gpu_lane_order(st->lane, k) != 0 &&
           st->ops->walk(st->state, 0, frontier, st->stage[k], SlabOff{}, pulsar_ckpt_pos(G), NULL) &&
           pulsar_gpu_lane_d2h(st->lane, st->stage[k], TensorBytes{0u}, p.host, p.bytes, done, 2u) != 0;
}

/* `p` -> the installed bank's state at G.  Device tier: the walk from the slab.  Host tier: the session stream waits
 * for p's ticket (its capture) and the next staging buffer's (its last device -> host copy), copies p.host into the
 * buffer and walks it in -- all queued, the host does not wait; *staged names the buffer, which holds p's bytes until
 * the next copy takes it (ckpt_copy reads it). */
/// @param st       the store
/// @param p        the source
/// @param frontier the record walk (false: the grid walk)
/// @param G        the position, in tokens
/// @param staged   receives the staging buffer used (host tier)
/// @return false on a failed walk or copy
static bool ckpt_in(pulsar_ckpt_store *st, const ckpt_place &p, bool frontier, uint32_t G, uint32_t *staged) {
    if (!st->host) return st->ops->walk(st->state, 1, frontier, p.slab, p.off, pulsar_ckpt_pos(G), NULL);
    if (!ckpt_stage_fit(st, p.bytes)) return false;
    const uint32_t k = st->stage_next++ % PULSAR_CKPT_STAGES;
    *staged = k;
    return pulsar_gpu_lane_order(st->lane, k) != 0 &&
           pulsar_gpu_lane_h2d(st->lane, st->stage[k], TensorBytes{0u}, p.host, p.bytes, p.ticket) != 0 &&
           st->ops->walk(st->state, 1, frontier, st->stage[k], SlabOff{}, pulsar_ckpt_pos(G), NULL);
}

/* `src`'s bytes -> `dst`, right after a ckpt_in(src) that staged them in buffer `staged` (host tier: the lane copies
 * the buffer out, as a capture does; device tier: a device copy on the session stream). */
/// @param st     the store
/// @param dst    the destination
/// @param src    the source just walked in
/// @param staged the staging buffer ckpt_in used (host tier)
/// @return false on a failed copy
static bool ckpt_copy(pulsar_ckpt_store *st, const ckpt_place &dst, const ckpt_place &src, uint32_t staged) {
    if (!st->host) return pulsar_gpu_tensor_copy_async(dst.slab, dst.off.before(), src.slab, src.off.before(),
                                                       src.bytes) != 0;
    const uint32_t done[2] = {dst.ticket, staged};
    return pulsar_gpu_lane_d2h(st->lane, st->stage[staged], TensorBytes{0u}, dst.host, src.bytes, done,
                               2u) != 0;
}

int pulsar_ckpt_host_tier(void) {
    const int integrated = pulsar_gpu_integrated();
    return integrated < 0 ? -1 : integrated == 0 ? 1 : 0;
}

TensorBytes pulsar_ckpt_host_bytes(const pulsar_ckpt_store *st) {
    if (!st->host) return TensorBytes{};
    TensorBytes bytes{};
    for (uint32_t b = 0; b < PULSAR_MSEQ_MAX; b++)
        if (st->host_slab[b]) bytes += st->slot_bytes.times(st->ops->ckpt_slots);
    for (uint32_t i = 0; i < st->n_rec; i++)
        if (st->rec[i].host) bytes += st->rec[i].cap;
    return bytes;
}

bool pulsar_ckpt_alloc(pulsar_ckpt_store *st, const pulsar_kv_state_ops *ops, void *state, uint32_t n_banks) {
    const int tier = pulsar_ckpt_host_tier();
    if (tier < 0) {
        fprintf(stderr, "pulsar: grid checkpoints (%s): the device did not say whether its memory is the host's -- "
                        "refusing to place them\n", ops->name);
        return false;
    }
    return pulsar_ckpt_alloc_tier(st, ops, state, n_banks, tier == 1);
}

bool pulsar_ckpt_alloc_tier(pulsar_ckpt_store *st, const pulsar_kv_state_ops *ops, void *state, uint32_t n_banks,
                            bool host) {
    if (n_banks < 1u) n_banks = 1u;
    if (n_banks > PULSAR_MSEQ_MAX || ops->ckpt_slots < 1u || ops->ckpt_slots > PULSAR_CKPT_SLOTS_MAX ||
        ops->resume_grid == 0u) return false;
    st->ops = ops;
    st->state = state;
    st->host = host;
    TensorBytes slot{};
    (void)ops->walk(state, -1, false, NULL, SlabOff{}, ops->min_checkpoint(state), &slot);
    /* The batched copies want 256-B aligned starts; keep every slot on one. */
    st->slot_bytes = pulsar_bytes_align(slot, 256u);
    memset(st->pos, 0, sizeof(st->pos));
    const TensorBytes end = ckpt_end_bytes(st);
    /* A dry run (the session's price) prices the device memory alone: the host tier's staging buffers, not its pinned
     * slabs, records or lane, which take no VRAM -- so a discrete card's per-bank price carries no checkpoint. */
    const bool dry = pulsar_gpu_tensor_dry_active() != 0;
    if (st->host) {
        if (!ckpt_stage_fit(st, end > st->slot_bytes ? end : st->slot_bytes)) return false;
        if (!dry && !(st->lane = pulsar_gpu_lane_create(CKPT_TICKETS))) {
            fprintf(stderr, "pulsar: grid checkpoints (%s): the copy lane could not be created\n", ops->name);
            return false;
        }
    }
    const TensorBytes slab_bytes = st->slot_bytes.times(ops->ckpt_slots);
    for (uint32_t b = 0; b < n_banks; b++) {
        const bool ok = st->host ? dry || (st->host_slab[b] = (uint8_t *)pulsar_gpu_host_alloc(slab_bytes)) != NULL
                                 : (st->slab[b] = pulsar_gpu_tensor_alloc(slab_bytes)) != NULL;
        if (!ok) {
            fprintf(stderr, "pulsar: grid checkpoint slab for bank %u (%u x %llu bytes, %s, %s) allocation failed\n",
                    b, ops->ckpt_slots, (unsigned long long)st->slot_bytes.raw(), ops->name,
                    st->host ? "pinned host RAM" : "device");
            return false;
        }
    }
    /* L289 / L311: the record pool, allocated with the ladder so a session's dry run prices it */
    st->n_rec = pulsar_ckpt_records_for(n_banks);
    st->rec_clock = 0;
    for (uint32_t i = 0; i < st->n_rec; i++) {
        pulsar_ckpt_record *r = &st->rec[i];
        r->T = 0u;
        const bool ok = st->host ? dry || (r->host = (uint8_t *)pulsar_gpu_host_alloc(end)) != NULL
                                 : (r->slab = pulsar_gpu_tensor_alloc(end)) != NULL;
        if (!ok) {
            fprintf(stderr, "pulsar: record %u of %u (%llu bytes, %s, %s) allocation failed\n", i, st->n_rec,
                    (unsigned long long)end.raw(), ops->name, st->host ? "pinned host RAM" : "device");
            return false;
        }
        r->cap = end;
    }
    return true;
}

void pulsar_ckpt_release(pulsar_ckpt_store *st) {
    /* the host tier: every copy still touching a pinned byte or a staging buffer lands before anything is freed */
    if (st->lane)
        for (uint32_t t = 0; t < CKPT_TICKETS; t++) (void)pulsar_gpu_lane_wait(st->lane, t);
    pulsar_gpu_lane_destroy(st->lane);
    st->lane = NULL;
    for (uint32_t k = 0; k < PULSAR_CKPT_STAGES; k++) {
        pulsar_gpu_tensor_free(st->stage[k]);
        st->stage[k] = NULL;
    }
    st->stage_bytes = TensorBytes{};
    for (uint32_t b = 0; b < PULSAR_MSEQ_MAX; b++) {
        pulsar_gpu_tensor_free(st->slab[b]);
        st->slab[b] = NULL;
        pulsar_gpu_host_free(st->host_slab[b]);
        st->host_slab[b] = NULL;
    }
    for (uint32_t i = 0; i < PULSAR_CKPT_RECORDS_MAX; i++) {
        pulsar_ckpt_record *r = &st->rec[i];
        pulsar_gpu_tensor_free(r->slab);
        pulsar_gpu_host_free(r->host);
        free(r->logits);
        free(r->tokens);
        free(r->img);   /* L350 */
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
/// @param st   the store
/// @param bank the bank
/// @param G    the grid point to capture, in tokens
/// @return the slot index to write, < ops->ckpt_slots
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

/** Can `bank`'s slab hold a checkpoint at G: an allocated bank, G on the resume grid and >= min_checkpoint?
 *  @param st   the store
 *  @param bank the bank
 *  @param G    the grid point, in tokens
 *  @return true when it can */
static bool ckpt_grid_ok(const pulsar_ckpt_store *st, uint32_t bank, uint32_t G) {
    return bank < PULSAR_MSEQ_MAX && (st->host ? st->host_slab[bank] != NULL : st->slab[bank] != NULL) &&
           G % st->ops->resume_grid == 0u &&
           G >= pulsar_ckpt_raw(st->ops->min_checkpoint(st->state));
}

/* The slot of `bank` holding G; PULSAR_CKPT_SLOTS_MAX when none. */
/// @param st   the store
/// @param bank the bank
/// @param G    the grid point, in tokens
/// @return the slot index, or PULSAR_CKPT_SLOTS_MAX
static uint32_t ckpt_find(const pulsar_ckpt_store *st, uint32_t bank, uint32_t G) {
    if (bank >= PULSAR_MSEQ_MAX || G == 0u || !st->ops) return PULSAR_CKPT_SLOTS_MAX;
    for (uint32_t k = 0; k < st->ops->ckpt_slots; k++)
        if (st->pos[bank][k] == G) return k;
    return PULSAR_CKPT_SLOTS_MAX;
}

/* Capture the installed `bank`'s state at grid point G (pulsar_ckpt_landed's).  False (said) on a violated
 * precondition. */
/// @param st   the store
/// @param bank the installed bank
/// @param G    the grid point, in tokens
/// @return true when captured
static bool ckpt_capture(pulsar_ckpt_store *st, uint32_t bank, uint32_t G) {
    if (!ckpt_grid_ok(st, bank, G)) {
        fprintf(stderr, "pulsar: grid checkpoint at %u on bank %u refused: not a grid point the slab can hold\n",
                G, bank);
        return false;
    }
    /* The precondition is what makes the copy the state AT G: every pool holds exactly the rows a
     * prefill to G writes, no more (a row above would be a decode's) and no fewer. */
    char why[192];
    if (!st->ops->stands_at(st->state, pulsar_ckpt_pos(G), why, sizeof(why))) {
        fprintf(stderr, "pulsar: grid checkpoint at %u on bank %u refused: %s\n", G, bank, why);
        return false;
    }
    const uint32_t s = ckpt_pick_slot(st, bank, G);
    pulsar_stall_mark("grid checkpoint capture");
    st->pos[bank][s] = 0u;   /* not a checkpoint until the copy is issued whole */
    if (!ckpt_out(st, ckpt_slot_place(st, bank, s), false, G)) {
        fprintf(stderr, "pulsar: grid checkpoint at %u on bank %u: copy failed\n", G, bank);
        return false;
    }
    st->pos[bank][s] = G;
    return true;
}

bool pulsar_ckpt_landed(pulsar_ckpt_store *st, BankId bank, TokPos T_at) {
    const uint32_t b = (uint32_t)bank.raw();   /* raw: raw inside; none() is out of range there, as before */
    const uint32_t T = (uint32_t)T_at.before().raw();   /* raw: L334 P2b: the store keeps raw uint32 inside (0 = empty slot / free record); typed at its API */
    bool capture = false;
    char why[192] = "";
    if (!st->ops->noted_at(st->state, T_at, &capture, why, sizeof(why))) {
        fprintf(stderr, "pulsar: bank %u's prompt chunk to %u is not its state: %s\n", b, T, why);
        return false;
    }
    if (!capture || T % st->ops->resume_grid != 0u || T < pulsar_ckpt_raw(st->ops->min_checkpoint(st->state)))
        return true;
    return ckpt_capture(st, b, T);
}

/* The bank's records above `tokens` go (the frontier rule), and with `tokens` 0 every one of the bank's. */
/// @param st     the store
/// @param bank   the bank
/// @param tokens the history length kept, in tokens
static void ckpt_records_drop_above(pulsar_ckpt_store *st, BankId bank, uint32_t tokens) {
    for (uint32_t i = 0; i < st->n_rec; i++)
        if (st->rec[i].T != 0u && st->rec[i].bank == bank && st->rec[i].T > tokens) st->rec[i].T = 0u;
}

bool pulsar_ckpt_restore(pulsar_ckpt_store *st, BankId bank, TokPos G_at) {
    const uint32_t b = (uint32_t)bank.raw();   /* raw: raw inside; none() is out of range there, as before */
    const uint32_t G = (uint32_t)G_at.before().raw();   /* raw: L334 P2b: the store keeps raw uint32 inside (0 = empty slot / free record); typed at its API */
    const uint32_t s = ckpt_find(st, b, G);
    if (s == PULSAR_CKPT_SLOTS_MAX) return false;
    uint32_t staged = 0;
    if (!st->ops->prepare_restore(st->state, G_at) || !ckpt_in(st, ckpt_slot_place(st, b, s), false, G, &staged)) {
        fprintf(stderr, "pulsar: grid checkpoint restore to %u on bank %u: copy failed\n", G, b);
        return false;
    }
    st->ops->restored(st->state);
    for (uint32_t k = 0; k < st->ops->ckpt_slots; k++)
        if (st->pos[b][k] > G) st->pos[b][k] = 0u;
    ckpt_records_drop_above(st, bank, G);
    return true;
}

/* The record a capture writes (the bank's previous one of its kind already dropped): a free one, else the least
 * recently used. */
/// @param st the store
/// @return the record to overwrite (the store's); NULL only when the pool is empty (n_rec 0)
static pulsar_ckpt_record *ckpt_record_pick(pulsar_ckpt_store *st) {
    pulsar_ckpt_record *lru = NULL;
    for (uint32_t i = 0; i < st->n_rec; i++) {
        pulsar_ckpt_record *r = &st->rec[i];
        if (r->T == 0u) return r;
        if (!lru || r->used < lru->used) lru = r;
    }
    return lru;
}

bool pulsar_ckpt_record_capture(pulsar_ckpt_store *st, BankId bank, pulsar_ckpt_record_kind kind, TokPos T_at,
                                const TokId *tokens, const float *logits, uint32_t width,
                                const struct pulsar_image_identity *img) {
    const uint32_t b = (uint32_t)bank.raw();   /* raw: raw inside; none() is out of range there, as before */
    const uint32_t T = (uint32_t)T_at.before().raw();   /* raw: L334 P2b: the store keeps raw uint32 inside (0 = empty slot / free record); typed at its API */
    if (b >= PULSAR_MSEQ_MAX || T == 0u || !tokens || st->n_rec == 0u ||
        (kind == PULSAR_CKPT_REC_END) != (logits != NULL && width != 0u))
        return false;
    /* the bank's previous record of this kind describes an older position: gone whatever happens below */
    for (uint32_t i = 0; i < st->n_rec; i++)
        if (st->rec[i].T != 0u && st->rec[i].bank == bank && st->rec[i].kind == kind) st->rec[i].T = 0u;
    char why[192];
    if (!st->ops->frontier_at(st->state, T_at, why, sizeof(why))) {
        fprintf(stderr, "pulsar: %s at %u on bank %u not recorded: %s\n",
                kind == PULSAR_CKPT_REC_END ? "prompt end" : "message endpoint", T, b, why);
        return false;
    }
    pulsar_ckpt_record *r = ckpt_record_pick(st);
    r->T = 0u;   /* not a record until the copy is issued whole (an evicted LRU record is gone from here) */
    const TensorBytes need = ckpt_end_bytes(st);
    if (r->cap < need) {   /* a drafter loaded after the store adds its rings to the frontier */
        bool ok = true;
        if (st->host) {   /* its last copy lands before its bytes go */
            ok = pulsar_gpu_lane_wait(st->lane, ckpt_ticket_rec(st, r)) != 0;
            pulsar_gpu_host_free(r->host);
            r->host = ok ? (uint8_t *)pulsar_gpu_host_alloc(need) : NULL;
            ok = r->host != NULL;
        } else {
            pulsar_gpu_tensor_free(r->slab);
            r->slab = pulsar_gpu_tensor_alloc(need);
            ok = r->slab != NULL;
        }
        r->cap = ok ? need : TensorBytes{};
        if (!ok) {
            fprintf(stderr, "pulsar: record for bank %u (%llu bytes) allocation failed\n", b,
                    (unsigned long long)need.raw());
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
        r->tokens = (TokId *)xmalloc((size_t)T * sizeof(TokId));
        r->tokens_cap = T;
    }
    if (!ckpt_out(st, ckpt_rec_place(st, r), true, T)) {
        fprintf(stderr, "pulsar: record at %u on bank %u: copy failed\n", T, b);
        return false;
    }
    if (logits) memcpy(r->logits, logits, (size_t)width * sizeof(float));
    for (uint32_t i = 0; i < T; i++) r->tokens[i] = tokens[i];
    /* L350: the images the state holds below T -- the match authority beside the ids */
    if (img && img->n > 0) {
        if (!r->img) r->img = (pulsar_image_identity *)xmalloc(sizeof(pulsar_image_identity));
        *r->img = *img;
        pulsar_image_identity_trim(r->img, T);
    } else if (r->img) {
        r->img->n = 0;
    }
    r->bank = bank;
    r->kind = kind;
    r->used = ++st->rec_clock;
    r->T = T;
    return true;
}

const pulsar_ckpt_record *pulsar_ckpt_record_of(const pulsar_ckpt_store *st, BankId bank,
                                                pulsar_ckpt_record_kind kind) {
    for (uint32_t i = 0; st && i < st->n_rec; i++)
        if (st->rec[i].T != 0u && st->rec[i].bank == bank && st->rec[i].kind == kind) return &st->rec[i];
    return NULL;
}

/* Are the record's ids tokens[0, T) -- and (L350) its images the request's below T? */
/// @param r      the record
/// @param tokens the prompt's ids
/// @param len    ids in `tokens`
/// @param want   the request's image blocks over the prompt, or NULL (then only a record holding no image)
/// @return true when the record is live, its T ids are the prompt's first T and its images are the request's
static bool ckpt_record_matches(const pulsar_ckpt_record *r, const TokId *tokens, uint32_t len,
                                const pulsar_image_identity *want) {
    if (r->T == 0u || !tokens || len < r->T) return false;
    for (uint32_t i = 0; i < r->T; i++)
        if (tokens[i] != r->tokens[i]) return false;
    const uint32_t held = r->img ? r->img->n : 0u;
    if (!want) return held == 0u;
    /* a block of the request straddling T would resume with its merged rows cut */
    for (uint32_t i = 0; i < want->n; i++)
        if (want->b[i].start < r->T && want->b[i].end > r->T) return false;
    pulsar_image_identity below = *want;
    pulsar_image_identity_trim(&below, r->T);
    if (below.n != held) return false;
    return held == 0u || pulsar_image_identity_equal(r->img, &below);
}

const pulsar_ckpt_record *pulsar_ckpt_record_best(const pulsar_ckpt_store *st, BankId bank, const TokId *tokens,
                                                  TokCount len_n, TokPos limit_at, TokCount prompt_n,
                                                  const struct pulsar_image_identity *want) {
    /* L334 P2b: the store keeps raw uint32 inside (0 = empty slot / free record); typed at its API */
    const uint32_t len = (uint32_t)len_n.raw(), limit = (uint32_t)limit_at.before().raw(),   /* raw: a bare local for the arithmetic below */
                   prompt_len = (uint32_t)prompt_n.raw();   /* raw: into a bare variable */
    const pulsar_ckpt_record *best = NULL;
    for (uint32_t i = 0; st && i < st->n_rec; i++) {
        const pulsar_ckpt_record *r = &st->rec[i];
        if (r->T == 0u || r->bank != bank || r->T > limit || (best && r->T <= best->T)) continue;
        if (r->T == prompt_len && !r->logits) continue;   /* a resume AT T reads the row the prefill left there */
        if (ckpt_record_matches(r, tokens, len, want)) best = r;
    }
    return best;
}

bool pulsar_ckpt_record_restore(pulsar_ckpt_store *st, BankId bank, const pulsar_ckpt_record *r, float *logits,
                                uint32_t width) {
    const uint32_t b = (uint32_t)bank.raw();   /* raw: raw inside; none() is out of range there, as before */
    if (!r || r < st->rec || r >= st->rec + st->n_rec || r->T == 0u || r->bank != bank ||
        (logits && (!r->logits || width != st->end_width)))
        return false;
    const uint32_t T = r->T;
    pulsar_ckpt_record *w = &st->rec[r - st->rec];
    /* the counters at T, prefilled to T (Qwen: the MTP layer's pending row at T - 1, its stage clean); the walk
     * brings every lane back */
    uint32_t staged = 0;
    if (!st->ops->install_frontier(st->state, pulsar_ckpt_pos(T), TokCount{(int)T}) ||
        !ckpt_in(st, ckpt_rec_place(st, w), true, T, &staged)) {
        fprintf(stderr, "pulsar: record restore to %u on bank %u: copy failed\n", T, b);
        w->T = 0u;
        return false;
    }
    st->ops->restored(st->state);
    for (uint32_t k = 0; k < st->ops->ckpt_slots; k++)
        if (st->pos[b][k] > T) st->pos[b][k] = 0u;
    ckpt_records_drop_above(st, bank, T);
    w->used = ++st->rec_clock;
    if (logits) memcpy(logits, w->logits, (size_t)width * sizeof(float));
    return true;
}

/* L313: the most pools any bank has (bank_pools' out) */
/** The most pools adopt() asks bank_pools for (twice PULSAR_KV_POOLS_MAX). */
#define CKPT_ADOPT_POOLS (2u * PULSAR_KV_POOLS_MAX)

bool pulsar_ckpt_adopt(pulsar_ckpt_store *st, BankId dst, BankId src, TokPos T_at, const pulsar_ckpt_record *r,
                       bool *moved) {
    const uint32_t d = (uint32_t)dst.raw(), sb = (uint32_t)src.raw();   /* raw: raw inside; none() is out of range */
    const uint32_t T = (uint32_t)T_at.before().raw();   /* raw: L334 P2b: the store keeps raw uint32 inside (0 = empty slot / free record); typed at its API */
    const char *why = NULL;
    ckpt_place from = {};
    *moved = false;
    if (d >= PULSAR_MSEQ_MAX || sb >= PULSAR_MSEQ_MAX || d == sb || T == 0u || !st->ops->bank_pools)
        why = "not two banks of the pool";
    else if (st->bank_adapter && (st->bank_adapter[sb] != 0) != (st->bank_adapter[d] != 0))
        why = "one bank runs the runtime adapter and the other does not (their states are different models')";
    else if (r) {
        if (r < st->rec || r >= st->rec + st->n_rec || r->bank != src || r->T != T) why = "the record is not the source's at T";
        else from = ckpt_rec_place(st, r);
    } else {
        const uint32_t s = ckpt_find(st, sb, T);
        if (s == PULSAR_CKPT_SLOTS_MAX) why = "the source holds no grid checkpoint at T";
        else from = ckpt_slot_place(st, sb, s);
    }
    static pulsar_kv_pool sp[CKPT_ADOPT_POOLS], dp[CKPT_ADOPT_POOLS];   /* worker thread only, as every store op */
    uint32_t n = 0;
    if (!why) {
        n = st->ops->bank_pools(st->state, src, sp, CKPT_ADOPT_POOLS);
        const uint32_t m = st->ops->bank_pools(st->state, dst, dp, CKPT_ADOPT_POOLS);
        if (n == 0 || n != m || n > CKPT_ADOPT_POOLS) why = "the banks' pools do not pair";
        for (uint32_t i = 0; !why && i < n; i++) {
            const uint64_t rows = (uint64_t)pulsar_comp_rows_cover(TokCount{(int)T}, sp[i].tokens_per_row).raw();   /* raw: a bare local for the arithmetic below */
            if (!sp[i].rows || !dp[i].rows) why = "a bank's physical is freed";
            else if (sp[i].tokens_per_row != dp[i].tokens_per_row || sp[i].row_bytes != dp[i].row_bytes ||
                     dp[i].row_bytes.times(rows) > pulsar_gpu_tensor_bytes(dp[i].rows) ||
                     sp[i].row_bytes.times(rows) > pulsar_gpu_tensor_bytes(sp[i].rows))
                why = "a pool's rows do not fit";
        }
    }
    if (why) {
        fprintf(stderr, "pulsar: adopting bank %u's %s at %u into bank %u refused: %s\n", sb,
                r ? "record" : "checkpoint", T, d, why);
        return false;
    }
    /* dst's history is replaced wholesale: its checkpoints and records go; then the counters at T, the source's rows
     * below T, the source's state at T -- the same three moves as a restore on the source itself */
    *moved = true;   /* L375 O43: from here dst is no longer what its view describes */
    pulsar_ckpt_drop_bank(st, dst);
    if (st->ops->history_replaced) st->ops->history_replaced(st->state);   /* L375 O42: before the walk below */
    bool ok = r ? st->ops->install_frontier(st->state, pulsar_ckpt_pos(T), TokCount{(int)T})
                 : st->ops->prepare_restore(st->state, pulsar_ckpt_pos(T));
    for (uint32_t i = 0; ok && i < n; i++) {
        const uint64_t rows = (uint64_t)pulsar_comp_rows_cover(TokCount{(int)T}, sp[i].tokens_per_row).raw();   /* raw: a bare local for the arithmetic below */
        ok = pulsar_gpu_tensor_copy_async(dp[i].rows, TensorBytes{0u}, sp[i].rows, TensorBytes{0u},
                                          sp[i].row_bytes.times(rows)) != 0;
    }
    uint32_t staged = 0;
    ok = ok && ckpt_in(st, from, r != NULL, T, &staged);
    if (!ok) {
        fprintf(stderr, "pulsar: adopting bank %u's state at %u into bank %u: copy failed\n", sb, T, d);
        return false;
    }
    st->ops->restored(st->state);
    /* and dst keeps the checkpoint it now stands at, so its own next resume finds it */
    if (!r) {
        /* a slot the device writes: no host wait for it (its earlier copies are ordered on the lane) */
        if (ckpt_grid_ok(st, d, T)) {
            const uint32_t slot = ckpt_pick_slot(st, d, T);
            st->pos[d][slot] = 0u;
            if (ckpt_copy(st, ckpt_slot_place(st, d, slot), from, staged)) pulsar_ckpt_commit(st, dst, slot, T_at);
        }
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
                w->tokens = (TokId *)xmalloc((size_t)T * sizeof(TokId));
                w->tokens_cap = T;
            }
            memcpy(w->tokens, r->tokens, (size_t)T * sizeof(TokId));
            /* L354: the record says which images it merged: r's, not whatever w last held */
            if (r->img && r->img->n > 0) {
                if (!w->img) w->img = (pulsar_image_identity *)xmalloc(sizeof(pulsar_image_identity));
                *w->img = *r->img;
            } else if (w->img) {
                w->img->n = 0;
            }
            if (ckpt_copy(st, ckpt_rec_place(st, w), from, staged)) {
                w->bank = dst;
                w->kind = r->kind;
                w->used = ++st->rec_clock;
                w->T = T;
            }
        }
    }
    return true;
}

/* The deepest checkpoint G <= limit (raw, 0 = none): pulsar_ckpt_best's and pulsar_ckpt_deepest's one scan. */
/// @param st    the store
/// @param bank  the bank
/// @param limit the highest grid point acceptable, in tokens
/// @return the grid point in tokens, 0 when none
static uint32_t ckpt_best_raw(const pulsar_ckpt_store *st, uint32_t bank, uint32_t limit) {
    if (bank >= PULSAR_MSEQ_MAX || !st->ops) return 0u;
    uint32_t best = 0u;
    for (uint32_t k = 0; k < st->ops->ckpt_slots; k++) {
        const uint32_t p = st->pos[bank][k];
        if (p != 0u && p <= limit && p > best) best = p;
    }
    return best;
}

TokPos pulsar_ckpt_best(const pulsar_ckpt_store *st, BankId bank, TokPos limit) {
    const uint32_t b = (uint32_t)bank.raw();   /* raw: raw inside; none() is out of range there, as before */
    const uint32_t g = ckpt_best_raw(st, b, (uint32_t)limit.before().raw());   /* raw: into ckpt_best_raw()'s bare parameter */
    return g ? TokPos{(int)g} : TokPos::none();
}

TokPos pulsar_ckpt_deepest(const pulsar_ckpt_store *st, BankId bank) {
    const uint32_t b = (uint32_t)bank.raw();   /* raw: raw inside; none() is out of range there, as before */
    const uint32_t g = ckpt_best_raw(st, b, UINT32_MAX);
    return g ? TokPos{(int)g} : TokPos::none();
}

void pulsar_ckpt_frontier_lowered(pulsar_ckpt_store *st, BankId bank, TokCount tokens_n) {
    const uint32_t b = (uint32_t)bank.raw();   /* raw: raw inside; none() is out of range there, as before */
    const uint32_t tokens = (uint32_t)tokens_n.raw();   /* raw: L334 P2b: the store keeps raw uint32 inside (0 = empty slot / free record); typed at its API */
    if (b >= PULSAR_MSEQ_MAX) return;
    for (uint32_t k = 0; k < PULSAR_CKPT_SLOTS_MAX; k++)
        if (st->pos[b][k] > tokens) st->pos[b][k] = 0u;
    ckpt_records_drop_above(st, bank, tokens);
}

void pulsar_ckpt_drop_bank(pulsar_ckpt_store *st, BankId bank) {
    const uint32_t b = (uint32_t)bank.raw();   /* raw: raw inside; none() is out of range there, as before */
    if (b >= PULSAR_MSEQ_MAX) return;
    memset(st->pos[b], 0, sizeof(st->pos[b]));
    ckpt_records_drop_above(st, bank, 0u);
}

/* A slot handed to the host (a reader or a writer): its place, after a host wait on its ticket on the host tier. */
/// @param st   the store
/// @param bank the bank
/// @param s    the slot
/// @param at   receives where its bytes are
/// @return false (said) when its ticket failed
static bool ckpt_hand_to_host(pulsar_ckpt_store *st, uint32_t bank, uint32_t s, pulsar_ckpt_slot_at *at) {
    const ckpt_place p = ckpt_slot_place(st, bank, s);
    if (st->host && pulsar_gpu_lane_wait(st->lane, p.ticket) == 0) {
        fprintf(stderr, "pulsar: grid checkpoint slot %u of bank %u: its last copy failed\n", s, bank);
        return false;
    }
    *at = {p.slab, p.off, p.host, s};
    return true;
}

bool pulsar_ckpt_holds(const pulsar_ckpt_store *st, BankId bank, TokPos G) {
    const uint32_t b = (uint32_t)bank.raw();   /* raw: raw inside; none() is out of range there, as before */
    return ckpt_find(st, b, (uint32_t)G.before().raw()) != PULSAR_CKPT_SLOTS_MAX;   /* raw: into ckpt_find()'s bare parameter */
}

bool pulsar_ckpt_locate(pulsar_ckpt_store *st, BankId bank, TokPos G, pulsar_ckpt_slot_at *at) {
    const uint32_t b = (uint32_t)bank.raw();   /* raw: raw inside; none() is out of range there, as before */
    const uint32_t s = ckpt_find(st, b, (uint32_t)G.before().raw());   /* raw: into ckpt_find()'s bare parameter */
    return s != PULSAR_CKPT_SLOTS_MAX && ckpt_hand_to_host(st, b, s, at);
}

bool pulsar_ckpt_claim(pulsar_ckpt_store *st, BankId bank, TokPos G_at, pulsar_ckpt_slot_at *at) {
    const uint32_t b = (uint32_t)bank.raw();   /* raw: raw inside; none() is out of range there, as before */
    const uint32_t G = (uint32_t)G_at.before().raw();   /* raw: L334 P2b: the store keeps raw uint32 inside (0 = empty slot / free record); typed at its API */
    if (!ckpt_grid_ok(st, b, G)) return false;
    const uint32_t k = ckpt_pick_slot(st, b, G);
    st->pos[b][k] = 0u;   /* filled by the caller, committed after */
    return ckpt_hand_to_host(st, b, k, at);
}

bool pulsar_ckpt_slot_get(const pulsar_ckpt_store *st, const pulsar_ckpt_slot_at *at, void *dst) {
    if (st->host) {
        memcpy(dst, at->host, st->slot_bytes.raw());   /* raw: a memcpy length / offset */
        return true;
    }
    const pulsar_gpu_copy_span span = {at->slab, at->off.before(), dst, st->slot_bytes};
    return pulsar_gpu_copy_out(&span, 1u) != 0;
}

bool pulsar_ckpt_slot_put(const pulsar_ckpt_store *st, const pulsar_ckpt_slot_at *at, const void *src) {
    if (st->host) {
        memcpy(at->host, src, st->slot_bytes.raw());   /* raw: a memcpy length / offset */
        return true;
    }
    return pulsar_gpu_tensor_write(at->slab, at->off.before(), src, st->slot_bytes) != 0;
}

void pulsar_ckpt_commit(pulsar_ckpt_store *st, BankId bank, uint32_t slot, TokPos G) {
    const uint32_t b = (uint32_t)bank.raw();   /* raw: raw inside; none() is out of range there, as before */
    if (b < PULSAR_MSEQ_MAX && slot < st->ops->ckpt_slots) st->pos[b][slot] = (uint32_t)G.before().raw();   /* raw: a bare operand */
}

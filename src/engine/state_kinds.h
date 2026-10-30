/* state_kinds.h -- declared session state (L295 F3): a family DECLARES the state its layers keep, the core
 * derives everything that state needs.
 *
 * A family names each piece of per-bank state once, as a pulsar_state_decl of one of four KINDS, and says which
 * layer slots carry which declarations (pulsar_state_plan).  From that table alone the core
 *   - allocates it, and prices it (the session's create run dry, session.cpp);
 *   - pages it: a POOL is demand-paged one tensor per bank, so free_physical / alloc_physical return and re-back a
 *     bank's touched pages, and the touched / growth bytes admission reads are the pools' row sizes;
 *   - answers the kv-state questions (kv_state.h) -- the slot walk, the pools, stands_at, the frontier -- so the grid
 *     checkpoints (prefix reuse at chunk boundaries), the disk segments and the session payload need nothing more;
 *   - captures a verify step's undo state and commits / rolls back each bank's run of it (speculation).
 *
 * THE KINDS
 *   (a) POOL    per-token rows, append-only, demand-paged per bank: full-attention KV, pooled index keys.  Row r
 *               holds tokens [r * tokens, (r + 1) * tokens); a segment carries them, a checkpoint references them.
 *   (b) RING    a sliding window's KV: `tokens` = the window W, the ring holds W - 1 + PULSAR_STATE_VERIFY_ROWS slots (a
 *               verify of k rows writes slots the window still needs only past W - 1 + k), resident, walked whole
 *               into a checkpoint.  Rolling a verify back needs no copy: a rejected row's slot is rewritten when
 *               its position is decoded again before any row reads it.
 *   (c) FIXED   recurrent state of a fixed size a bank (GDN recurrent + conv, PLE conv, a short-conv stream's
 *               tail), overwritten as positions advance, walked into a checkpoint.  Its verify capture is one of
 *               SNAPSHOT (row r of the verify writes the state after it; a commit of `keep` rows copies row keep-1
 *               back), DELTA (row r writes its update and the verify leaves the state alone; the commit applies the
 *               kept rows -- the decl's `commit`) or, for a HOST decl, REPLAY (the bank's bytes before the step;
 *               the commit puts them back and replays the kept tokens -- the decl's `replay`).
 *   (d) STAGED  an open block of `tokens` rows that fills position by position and is flushed into a POOL when it
 *               closes (the indexer's partial block).  Capture STAGE: each run's block before the step and each
 *               row's input record; the commit restores the block and rewrites the kept rows of the block open at
 *               the new frontier.
 *
 * LAYOUT.  A device decl's state is one tensor per (slot, decl), bank-major (bank b at b * its per-bank bytes);
 * a POOL is one managed tensor per (slot, decl, bank); a host decl (session-wide only) one host array.  A capture
 * is one tensor per decl, [the slots carrying it][verify rows] -- slot k's rows at k * verify_rows.  Slots
 * [n_trunk, n_slots) are a drafter's layers: they trail the trunk by one row, so their pools are no segment's
 * (a payload carries them as trailing pools), they are never evicted and never captured.
 *
 * THE WALK ORDER (the checkpoint slot's bytes, a payload's lanes): slots in order, each slot's decls in id order,
 * then the session decls in id order.  Pools the same way. */
#ifndef PULSAR_STATE_KINDS_H
#define PULSAR_STATE_KINDS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../lib/pulsar_units.h"

/* Included from pulsar_engine_internal.h after kv_state.h and family.h; not standalone. */

/** Declarations one family's table carries (a slot's set is a mask of their ids). */
#define PULSAR_STATE_DECLS_MAX 16u
/** Layer slots one plan carries: the trunk's layers and a drafter's. */
#define PULSAR_STATE_SLOTS_MAX PULSAR_FAMILY_MAX_LAYER
/** The rows one verify step carries, every bank's run together: the depth of every capture, and the core
 *  driver's head width (step_driver.h). */
#define PULSAR_STATE_VERIFY_ROWS PULSAR_DRIVER_HEADS_MAX

/** What a piece of state IS (the header's four kinds). */
typedef enum : uint8_t {
    PULSAR_STATE_POOL   = 1,  ///< (a) per-token rows, demand-paged per bank
    PULSAR_STATE_RING   = 2,  ///< (b) a sliding window's ring of W - 1 + verify rows slots
    PULSAR_STATE_FIXED  = 3,  ///< (c) fixed recurrent state
    PULSAR_STATE_STAGED = 4,  ///< (d) an open block flushed into a pool when it closes
} pulsar_state_kind;

/** How a verify step keeps a decl's undo state (the header's captures). */
typedef enum : uint8_t {
    PULSAR_STATE_CAPTURE_NONE     = 0,
    PULSAR_STATE_CAPTURE_SNAPSHOT = 1,  ///< FIXED: the state after each row
    PULSAR_STATE_CAPTURE_DELTA    = 2,  ///< FIXED: each row's update; the commit applies the kept ones
    PULSAR_STATE_CAPTURE_REPLAY   = 3,  ///< FIXED host: the state before the step; the commit replays the kept tokens
    PULSAR_STATE_CAPTURE_STAGE    = 4,  ///< STAGED: the block before the step + each row's input record
} pulsar_state_capture;

/** One piece of per-bank state. */
typedef struct pulsar_state_decl {
    const char *name;              ///< printed in refusals ("gdn_state", "kv", "sconv")
    pulsar_state_kind kind;
    pulsar_state_capture capture;
    bool host;                     ///< FIXED, session-wide only: host memory (n-gram context)
    bool whole_rows;               ///< POOL: a row is written once its last token lands (kv_state.h pulsar_kv_pool)
    uint32_t tokens;               ///< POOL: tokens a row; STAGED: rows of the open block; RING: the window W
    uint64_t bytes;                ///< POOL: a row; RING: a slot; FIXED: one bank's state; STAGED: one block row
    uint64_t capture_bytes;        ///< DELTA: one row's update; STAGE: one row's input record
    uint64_t capture_off;          ///< STAGE: where the block row sits in the record
    /** DELTA: apply rows [0, keep) of `rows` (device, capture_bytes a row) to bank `bank` of `state` (device, the
     *  decl's whole tensor), on the session stream.  false = refused (said). */
    bool (*commit)(void *state, uint32_t bank, const void *rows, uint32_t keep);
    /** REPLAY: advance a bank's host state `state` (restored to its bytes before the step) over the kept tokens.
     *  `e` is the engine (const pulsar_engine *).  false = refused (said). */
    bool (*replay)(const void *e, void *state, const int32_t *tokens, uint32_t keep);
    /** A host decl's fresh bank (NULL = zero). */
    void (*init)(const void *e, void *state);
} pulsar_state_decl;

/** A family's state declaration: built once at load, read-only after.  `slot[k]` is the mask of decl ids layer
 *  slot k carries; `session` the mask held once a bank outside any layer (host decls only). */
typedef struct pulsar_state_plan {
    const char *name;              ///< the kv-state's name (refusals; mixed into the segment / payload digest)
    uint32_t resume_grid;          ///< kv_state.h: every POOL / STAGED decl's tokens divide it
    bool split_invariant;          ///< kv_state.h: a prompt cut anywhere is the cold prefill's bytes
    uint32_t ckpt_slots;           ///< grid checkpoints a bank keeps (<= PULSAR_CKPT_SLOTS_MAX)
    pulsar_state_decl decl[PULSAR_STATE_DECLS_MAX];
    uint32_t n_decl;
    uint32_t n_slots;              ///< [0, n_trunk) the trunk's layers, [n_trunk, n_slots) a drafter's
    uint32_t n_trunk;
    uint32_t slot[PULSAR_STATE_SLOTS_MAX];
    uint32_t session;
    /** The drafter lane's own counters the state walk must keep in step (NULL = no drafter lane): a restore or a
     *  payload put bank `bank` at T (T 0 = a fresh bank) -- the drafter's lagging row is the trunk's at T - 1. */
    void (*drafter_at)(void *ud, uint32_t bank, uint32_t T);
    /** ... and does the drafter's state describe frontier T (nothing drafted past it)?  `why` names the violation. */
    bool (*drafter_frontier)(void *ud, uint32_t bank, uint32_t T, char *why, size_t whylen);
} pulsar_state_plan;

/** Bytes one bank of a non-POOL decl takes (FIXED: bytes; STAGED: the block; RING: W - 1 + the verify width slots). */
uint64_t pulsar_state_decl_bank_bytes(const pulsar_state_decl *d);
/** Declare decl `id` (returns it): the table's next entry. */
uint32_t pulsar_state_plan_add(pulsar_state_plan *p, const pulsar_state_decl &d);

/** A session's declared state (the plan's instance for n_banks banks). */
typedef struct pulsar_state_set {
    const pulsar_state_plan *plan;
    const void *engine;            ///< the decls' replay / init context
    void *ud;                      ///< the drafter hooks' (the family's session state)
    uint32_t n_banks;
    uint32_t ctx;                  ///< a POOL bank's capacity in tokens: the session's, rounded to whole rows / blocks
    uint32_t verify_rows;          ///< the captures' depth (0 = none: no drafter, no verify)
    pulsar_gpu_tensor *t[PULSAR_STATE_SLOTS_MAX][PULSAR_STATE_DECLS_MAX];       ///< FIXED / STAGED / RING
    pulsar_gpu_tensor **pool[PULSAR_STATE_SLOTS_MAX][PULSAR_STATE_DECLS_MAX];   ///< POOL: [n_banks], NULL = evicted
    uint8_t *host[PULSAR_STATE_DECLS_MAX];                                      ///< session host decls: [n_banks]
    /* the verify capture */
    pulsar_gpu_tensor *cap_rows[PULSAR_STATE_DECLS_MAX];   ///< SNAPSHOT / DELTA / STAGE rows
    pulsar_gpu_tensor *cap_runs[PULSAR_STATE_DECLS_MAX];   ///< STAGE: each run's block before the step
    uint8_t *cap_host[PULSAR_STATE_DECLS_MAX];             ///< REPLAY: each run's bank bytes before the step
    uint8_t ord[PULSAR_STATE_SLOTS_MAX][PULSAR_STATE_DECLS_MAX];   ///< a slot's index among the slots carrying a decl
    /** The last verify step's runs: run k is step rows [run_first[k], run_first[k + 1]) of bank run_bank[k] from
     *  position run_pos0[k]. */
    uint32_t n_runs;
    uint32_t run_first[PULSAR_STATE_VERIFY_ROWS + 1];
    int32_t run_bank[PULSAR_STATE_VERIFY_ROWS];
    int32_t run_pos0[PULSAR_STATE_VERIFY_ROWS];
    /* per bank */
    uint32_t *bank_pos;            ///< tokens each bank's state holds (pulsar_state_set_pos writes it)
    uint32_t *kv_hw;               ///< L270: the most tokens each bank has held -- its touched pool pages
    uint32_t *prefill_pos;         ///< L266: the end of each bank's prefill-only history (the cold prefill's bytes)
    bool *frontier_stale;          ///< a segment load wrote the pools without the lanes
    uint32_t live_bank;            ///< the bank sync / eval / the store's walk run on
    pulsar_ckpt_store ckpt;        ///< the grid checkpoints and prompt ends (kv_state.h)
    pulsar_kv_state_ops ops;       ///< this set's kv-state answer: the core's walk over the plan
} pulsar_state_set;

/** Build the plan's state for `n_banks` banks of `ctx` tokens, with a verify capture of `verify_rows` rows (0 =
 *  none).  Every bank starts fresh (pulsar_state_reset_bank).  Allocates only through the tensor allocators, so a
 *  dry create prices it.  NULL (said) on a refusal or a failed allocation. */
pulsar_state_set *pulsar_state_create(const void *engine, const pulsar_state_plan *plan, void *ud, uint32_t n_banks,
                                      uint32_t ctx, uint32_t verify_rows);
void pulsar_state_destroy(pulsar_state_set *ss);
/** A bank starts over: its device state zeroed, its host decls at their init, its counters 0, its drafter row
 *  dropped and its checkpoints gone. */
bool pulsar_state_reset_bank(pulsar_state_set *ss, uint32_t bank);
/** L270: the one writer of a bank's position; it keeps the bank's high-water. */
static inline void pulsar_state_set_pos(pulsar_state_set *ss, uint32_t bank, uint32_t pos) {
    ss->bank_pos[bank] = pos;
    if (pos > ss->kv_hw[bank]) ss->kv_hw[bank] = pos;
}
/** The tensor of a FIXED / STAGED / RING decl on slot `slot` (every bank's). */
static inline pulsar_gpu_tensor *pulsar_state_tensor(const pulsar_state_set *ss, uint32_t slot, uint32_t id) {
    return ss->t[slot][id];
}
/** Bank `bank`'s tensor of a POOL decl on slot `slot` (NULL while the bank is evicted). */
static inline pulsar_gpu_tensor *pulsar_state_pool(const pulsar_state_set *ss, uint32_t slot, uint32_t id,
                                                   uint32_t bank) {
    return ss->pool[slot][id] ? ss->pool[slot][id][bank] : NULL;
}
/** Bank `bank`'s bytes of a session host decl. */
static inline void *pulsar_state_host(const pulsar_state_set *ss, uint32_t id, uint32_t bank) {
    return ss->host[id] + (size_t)bank * ss->plan->decl[id].bytes;
}
/** The verify capture's rows of decl `id` on slot `slot` (device; step row r at + r * the row's bytes): where a
 *  verify's op writes them -- as a pointer, or as its tensor and the byte offset into it (a copy's destination). */
void *pulsar_state_capture_rows(const pulsar_state_set *ss, uint32_t slot, uint32_t id);
uint64_t pulsar_state_capture_offset(const pulsar_state_set *ss, uint32_t slot, uint32_t id);
static inline pulsar_gpu_tensor *pulsar_state_capture_tensor(const pulsar_state_set *ss, uint32_t id) {
    return ss->cap_rows[id];
}

/** A verify step begins: its runs (`n_runs` runs, run k rows [run_first[k], run_first[k + 1]) of bank[run_first[k]]
 *  from pos[run_first[k]]), each run's host REPLAY state kept.  Then, on the step's stream inside its commands,
 *  pulsar_state_verify_capture keeps each run's STAGE blocks before any op writes them. */
void pulsar_state_verify_begin(pulsar_state_set *ss, uint32_t n_runs, const uint32_t *run_first, const int32_t *bank,
                               const int32_t *pos);
bool pulsar_state_verify_capture(pulsar_state_set *ss);
/** Keep the first `keep` rows of bank `bank`'s run of the last verify step and set every captured state of the
 *  bank to "after row keep - 1" (the header's captures); `tokens` are the run's tokens.  The bank's position
 *  counter is the caller's.  false (said) = refused or a copy failed. */
bool pulsar_state_commit(pulsar_state_set *ss, uint32_t bank, const int32_t *tokens, uint32_t keep);

/** L270: the demand-paged bytes `rows` positions take in one bank (every slot's pools): the declared families' one
 *  conversion of rows to bytes, each pool rounding its own ceil(rows / tokens-per-row) (L333). */
KvBytes pulsar_state_pool_bytes_at(const pulsar_state_set *ss, KvRows rows);
/** L284 #3: a bank's physical residency -- its trunk pools.  _free releases them (and zeroes its high-water),
 *  _alloc re-backs whichever are missing (idempotent; on a failure the bank is left wholly evicted), _evicted is
 *  true while any is missing. */
void pulsar_state_bank_free(pulsar_state_set *ss, uint32_t bank);
bool pulsar_state_bank_alloc(pulsar_state_set *ss, uint32_t bank);
bool pulsar_state_bank_evicted(const pulsar_state_set *ss, uint32_t bank);

/** The bank pool of a family whose state is declared (family.h pulsar_family_bank_ops), over pulsar_session::states. */
extern const pulsar_family_bank_ops PULSAR_BANK_OPS_DECLARED;

#endif /* PULSAR_STATE_KINDS_H */

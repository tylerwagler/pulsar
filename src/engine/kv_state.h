/* kv_state.h -- the model-specific half of the KV state model (L264, made generic in L265).
 *
 * L264's model of a session's KV is architecture-neutral except for one question: WHAT IS THE
 * STATE OF A BANK AT A POSITION?  Everything else is shared code that never names a layer:
 *
 *   - grid checkpoints: per bank, a few slots holding "the state at grid point G".  ONE ladder for
 *     every family (P13): a prefill walk -- the core loop, DeepSeek's planner, a fused step's prompt
 *     chunk -- captures at EVERY chunk end on the grid where the state is a prefill's
 *     (pulsar_ckpt_landed), its final chunk is cut at the prompt's last grid point
 *     (pulsar_ckpt_final_cut), and a capture into a full store keeps the newest half of it whole
 *     and thins the older half by merging the smallest gap.  Only the slot COUNT is the model's (its
 *     slot size prices it).  A checkpoint is valid while the bank still holds its history up to G
 *     (the frontier rule); a resume restores the deepest one at or below the prefill frontier and
 *     prefills from there (checkpoint.cpp, session.cpp);
 *   - disk segments: the tokens [G_prev, G), every APPEND-ONLY row pool's rows for that span, and
 *     the checkpoint at G where the bank holds it (L323) -- a chain of them rebuilds a bank (session_payload.cpp, then
 *     src/lib/pulsar_segstore + pulsar_kvchain, the server, the agent, TP mirroring).
 *
 * A model answers the question with a pulsar_kv_state_ops: which of its state is append-only (row
 * POOLS -- referenced by a checkpoint through the frontier, serialized by a segment) and which is
 * overwritten as positions advance (copied into a checkpoint SLOT by its walk), how big a slot is,
 * which grid its kernels make a resume exact on, and how many slots a bank can afford.
 *
 * DeepSeek V4 (kv_state_ds4.cpp): pools = every kv source's compressed rows (ratio 4 / 128) and
 * index-K rows (ratio 4); slot = the raw SWA window [G - W, G) of every layer + the overlapping
 * (coff-2) sources' compressor and indexer lanes (~3.8 MB); grid 128; 16 slots.
 *
 * L295 F3: a family whose state is DECLARED (state_kinds.h -- Qwen3.8-Flash-Next's is family_qwen.cpp
 * qwen_state_plan) gets its answer from the core: the declaration's POOL kinds are the pools, its FIXED / STAGED /
 * RING kinds the slot, walked in the declaration's order (pulsar_state_set::ops).  The ops take an opaque `state`,
 * and the store below is a plain struct any family's state embeds. */
#ifndef PULSAR_KV_STATE_H
#define PULSAR_KV_STATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Included from pulsar_engine_internal.h (pulsar_gpu_tensor, PULSAR_MSEQ_MAX); not standalone. */

/** The most checkpoint slots any model keeps per bank (the store's array bound). */
#define PULSAR_CKPT_SLOTS_MAX 16u
/** The most append-only row pools any model has (DeepSeek V4: <= 2 per layer). */
#define PULSAR_KV_POOLS_MAX 128u

/** One append-only row pool of the INSTALLED bank: row r holds tokens [r * tokens_per_row,
 *  (r + 1) * tokens_per_row), rows never change once the frontier is past them, and the
 *  rows are contiguous in `rows` (a tensor the installed bank owns) at r * row_bytes. */
typedef struct {
    struct pulsar_gpu_tensor *rows;
    uint32_t tokens_per_row;
    uint64_t row_bytes;
    /** A row is written once, when its last token lands (DeepSeek's compressor emit: the open group is the
     *  slot's lanes), so a frontier T holds rows [0, T / tokens_per_row); false = the open row is written as
     *  its tokens arrive and a payload carries it too.  A grid span ends on whole rows either way. */
    bool whole_rows;
} pulsar_kv_pool;

typedef struct pulsar_kv_state_ops {
    const char *name;            ///< printed in refusals ("deepseek-v4", "qwen4-exp")
    uint32_t resume_grid;        ///< tokens; every pool's tokens_per_row divides it
    /** L284: a prompt cut ANYWHERE is the cold prefill's bytes (Qwen: split-invariant kernels, L266 C1), so a
     *  history prefilled whole continues from its end wherever that falls; false = only a cut on resume_grid is
     *  (DeepSeek: a chunk's bytes depend on a row's offset in the call, L183), so a history ending off the grid
     *  resumes from a checkpoint below it.  The one fact pulsar_session_bank_continues reads. */
    bool split_invariant;
    uint32_t ckpt_slots;         ///< slots per bank, <= PULSAR_CKPT_SLOTS_MAX (the ladder's rule is checkpoint.cpp's)
    /** The ONE slot layout, walked identically by sizing, capture and restore: `dir` < 0 sizes
     *  only (*bytes = the slot's size), 0 copies the installed bank's state -> slot, > 0 copies
     *  slot -> state.  `frontier`: the state at ANY position G (a payload's frontier), not at a grid
     *  point -- a superset of the grid slot where the model holds state a grid point has canonical or
     *  rebuilds (DeepSeek: its coff-1 lanes and its drafter's rings).  Sized when a payload is (a
     *  drafter may load after the store).  Device copies on the session stream; false on a failed copy. */
    bool (*walk)(void *state, int dir, bool frontier, struct pulsar_gpu_tensor *slab, uint64_t off, uint32_t G,
                 uint64_t *bytes);
    /** The smallest G a checkpoint can describe (DeepSeek: the raw window). */
    uint32_t (*min_checkpoint)(void *state);
    /** Does the installed bank stand exactly at prefill point G -- every pool's frontier at
     *  G / tokens_per_row and nothing decoded past it?  `why` names the first violation. */
    bool (*stands_at)(void *state, uint32_t G, char *why, size_t whylen);
    /** Does the installed bank hold every pool's rows up to G (a segment's save precondition)? */
    bool (*holds)(void *state, uint32_t G);
    /** Before a restore's walk: every pool's frontier to G, every non-slot lane to its canonical
     *  boundary state. */
    bool (*prepare_restore)(void *state, uint32_t G);
    /** After a restore: the state is exact at G (clears the model's "stale" mark). */
    void (*restored)(void *state);
    /** A segment load wrote the pools' rows to G without the lanes: mark the bank stale until the
     *  chain's last checkpoint is restored. */
    void (*set_frontier_stale)(void *state, uint32_t G);
    /** Is `bank` (any bank, installed or not) stale -- its lanes describing no position until a restore
     *  (set_frontier_stale; DeepSeek also a rewind off the grid)?  Out of range = false. */
    bool (*stale)(void *state, uint32_t bank);
    /** The installed bank's append-only pools, in a fixed order; returns how many. */
    uint32_t (*pools)(void *state, pulsar_kv_pool *out, uint32_t cap);
    /* ---- L284: the session payload, every model's (session_payload.cpp, "the kv-state payload"). */
    /** Does the installed bank stand at frontier T with every lane describing T -- nothing stale, nothing
     *  speculative written past its true rows?  `why` names the first violation.  (The prefill frontier
     *  the payload carries is pulsar_session_bank_prefill_frontier's.) */
    bool (*frontier_at)(void *state, uint32_t T, char *why, size_t whylen);
    /** Before a payload load's frontier walk (dir > 0): the bank's counters at T with `prefill` of it
     *  prefilled, where the model keeps that count; restored() follows the walk. */
    bool (*install_frontier)(void *state, uint32_t T, uint32_t prefill);
    /** Append-only rows that TRAIL the frontier (a row is written once the token after it exists, so no
     *  grid span closes them): a payload carries them to its frontier beside the pools; a segment does
     *  not.  Same contract as pools; NULL = none (DeepSeek). */
    uint32_t (*trailing_pools)(void *state, pulsar_kv_pool *out, uint32_t cap);
    /** A prompt chunk took the installed bank's history to T -- a prefill walk's chunk landed, or a fused step's
     *  (L284 #2, pulsar_session_note_prefilled) -- and pulsar_ckpt_landed asks about it.  False (`why` filled)
     *  when the state does not stand at T -- the record would describe another state; else *capture says whether
     *  the state at T is a prefill's, so a grid checkpoint there is the cold prefill's (DeepSeek: no stale
     *  compressor group; Qwen: T is the prefill-only history's end). */
    bool (*noted_at)(void *state, uint32_t T, bool *capture, char *why, size_t whylen);
    /** L313: every append-only row pool of ANY bank -- pools() then trailing_pools(), the same order -- each
     *  bank's own tensor (a NULL `rows` while the bank's physical is freed).  What a cross-bank restore copies. */
    uint32_t (*bank_pools)(void *state, uint32_t bank, pulsar_kv_pool *out, uint32_t cap);
} pulsar_kv_state_ops;

/** The kinds of record (pulsar_ckpt_store::rec). */
typedef enum : uint8_t {
    PULSAR_CKPT_REC_END = 1,        ///< L289: a classic sync's prompt end, with its logits
    PULSAR_CKPT_REC_ENDPOINT = 2,   ///< L311: the prompt's last message endpoint (its last turn marker's start)
} pulsar_ckpt_record_kind;

/** The most records one store pools (two a bank at the pool's widest). */
#define PULSAR_CKPT_RECORDS_MAX (2u * PULSAR_MSEQ_MAX)

/** One record of the pool: bank `bank`'s state at T. */
typedef struct pulsar_ckpt_record {
    uint32_t bank;                     ///< the bank whose history it is a position of
    uint32_t T;                        ///< 0 = a free record
    pulsar_ckpt_record_kind kind;
    uint64_t used;                     ///< the pool's clock at its capture or last restore (the eviction order)
    struct pulsar_gpu_tensor *slab;    ///< the frontier walk's bytes
    uint64_t cap;                      ///< slab's bytes (it grows when a drafter loads after the store)
    float *logits;                     ///< host, end_width floats: a prompt end's next-token row (NULL otherwise)
    /** host: the T token ids the record's state is the prefill of -- the match authority at a resume (a history
     *  rewound inside T's last compressor group and re-prefilled with other ids keeps the record's rows, not its
     *  tokens) */
    int32_t *tokens;
    uint32_t tokens_cap;
} pulsar_ckpt_record;

/** A model's grid checkpoints: per bank, ops->ckpt_slots slots of slot_bytes.  pos[bank][s] is
 *  the grid point slot s holds, 0 = empty.  A checkpoint at G is valid while its bank's history
 *  reaches G: pulsar_ckpt_frontier_lowered drops it when the history falls below. */
typedef struct pulsar_ckpt_store {
    const pulsar_kv_state_ops *ops;
    void *state;
    /** The artifact the state was computed by, mixed into every segment's layout digest so a segment
     *  of another quantisation of the same model is refused, never loaded (L266: Qwen's containers
     *  share one store identity).  0 = the store identity alone decides (DeepSeek's routed tier). */
    uint64_t artifact;
    struct pulsar_gpu_tensor *slab[PULSAR_MSEQ_MAX];
    uint64_t slot_bytes;
    uint32_t pos[PULSAR_MSEQ_MAX][PULSAR_CKPT_SLOTS_MAX];
    /** L289 / L311: the RECORDS, one pool every bank draws from -- each the state a prefill left at an EXACT
     *  position T of a bank's history (not a grid point): the model's FRONTIER walk at T and the token ids it is the
     *  prefill of.  Two kinds: a PROMPT END (the state a classic sync's prefill left at the prompt's last token, with
     *  the logits row it headed) and a MESSAGE ENDPOINT (L311: where the prompt's last turn marker begins, the
     *  history before the generation header -- the branch point of an agent's side request).  Valid by the frontier
     *  rule like a grid checkpoint (T 0 once the bank's history falls below T, or the bank is dropped).  A record
     *  resumes a prompt that carries its tokens: the same prompt AT T with its logits (every family: the state is the
     *  prefill's own, copied whole), and on a split-invariant family any longer one from T. */
    pulsar_ckpt_record rec[PULSAR_CKPT_RECORDS_MAX];
    uint32_t n_rec;                         ///< records in the pool (pulsar_ckpt_records_for)
    uint64_t rec_clock;                     ///< the pool's use clock (a capture or restore stamps rec[].used)
    uint32_t end_width;                     ///< the logits width a prompt end keeps (the engine's)
} pulsar_ckpt_store;

bool pulsar_ckpt_alloc(pulsar_ckpt_store *st, const pulsar_kv_state_ops *ops, void *state, uint32_t n_banks);
void pulsar_ckpt_release(pulsar_ckpt_store *st);
/** THE capture rule (P13), every walk's: a prompt chunk took the installed `bank` to T.  The model confirms the
 *  state stands at T (ops->noted_at; false, said, when not), and a T on the grid where the state is a prefill's
 *  is captured (false, said, when the capture fails). */
bool pulsar_ckpt_landed(pulsar_ckpt_store *st, uint32_t bank, uint32_t T);
/** Restore the installed `bank` to its checkpoint at G; the checkpoints above G go.  False when
 *  none at G or a copy failed. */
bool pulsar_ckpt_restore(pulsar_ckpt_store *st, uint32_t bank, uint32_t G);
/** The deepest checkpoint G <= limit `bank` holds; 0 when none. */
uint32_t pulsar_ckpt_best(const pulsar_ckpt_store *st, uint32_t bank, uint32_t limit);
/** `bank`'s history now reaches only `tokens`: the checkpoints above it go (and a prompt end above it -- a model
 *  whose pools hold R tokens a whole row passes the deepest position its intact rows serve, R * rows + R - 1). */
void pulsar_ckpt_frontier_lowered(pulsar_ckpt_store *st, uint32_t bank, uint32_t tokens);
/** Drop every checkpoint of `bank`: its history was replaced wholesale. */
void pulsar_ckpt_drop_bank(pulsar_ckpt_store *st, uint32_t bank);
/** The slot holding G (a save reads it). */
bool pulsar_ckpt_locate(pulsar_ckpt_store *st, uint32_t bank, uint32_t G, struct pulsar_gpu_tensor **slab,
                        uint64_t *off);
/** A slot for an incoming G, emptied until pulsar_ckpt_commit marks it (a load writes it). */
bool pulsar_ckpt_claim(pulsar_ckpt_store *st, uint32_t bank, uint32_t G, struct pulsar_gpu_tensor **slab,
                       uint64_t *off, uint32_t *slot);
void pulsar_ckpt_commit(pulsar_ckpt_store *st, uint32_t bank, uint32_t slot, uint32_t G);
/** The grid G <= pos at which a resume of a history prefilled to `pos` starts. */
static inline uint32_t pulsar_ckpt_grid_floor(const pulsar_ckpt_store *st, uint32_t pos) {
    return pos / st->ops->resume_grid * st->ops->resume_grid;
}
/** L311: how many records a store of n_banks pools -- a prompt end and a message endpoint for most banks: n_banks
 *  plus half as many again (at least two, so a one-bank session keeps both of its own). */
static inline uint32_t pulsar_ckpt_records_for(uint32_t n_banks) {
    const uint32_t n = n_banks + (n_banks / 2u > 1u ? n_banks / 2u : 1u);
    return n < PULSAR_CKPT_RECORDS_MAX ? n : PULSAR_CKPT_RECORDS_MAX;
}
/** L289 / L311: capture the installed `bank`'s state at T as a record of `kind` -- a prompt end with the logits the
 *  prefill left there (`width` floats), a message endpoint with none (logits NULL).  It replaces the bank's record of
 *  that kind, in a free record or the least recently used one.  The model confirms a live frontier at T
 *  (ops->frontier_at); false, said, when not or when a copy fails (the bank then holds no record of that kind). */
bool pulsar_ckpt_record_capture(pulsar_ckpt_store *st, uint32_t bank, pulsar_ckpt_record_kind kind, uint32_t T,
                                const int *tokens, const float *logits, uint32_t width);
/** The deepest of `bank`'s records whose ids are tokens[0, T) (of `len`) with T <= limit and -- when T == prompt_len,
 *  a resume AT T -- that holds the logits that resume reads.  NULL when none. */
const pulsar_ckpt_record *pulsar_ckpt_record_best(const pulsar_ckpt_store *st, uint32_t bank, const int *tokens,
                                                  uint32_t len, uint32_t limit, uint32_t prompt_len);
/** `bank`'s record of `kind`, NULL when none. */
const pulsar_ckpt_record *pulsar_ckpt_record_of(const pulsar_ckpt_store *st, uint32_t bank,
                                                pulsar_ckpt_record_kind kind);
/** `bank`'s prompt end T, 0 when none. */
static inline uint32_t pulsar_ckpt_end(const pulsar_ckpt_store *st, uint32_t bank) {
    const pulsar_ckpt_record *r = pulsar_ckpt_record_of(st, bank, PULSAR_CKPT_REC_END);
    return r ? r->T : 0u;
}
/** Put the installed `bank` back at its record `r` (one of the bank's own): the counters at r->T (prefilled to T),
 *  every lane and open row from the record, the grid checkpoints and records above T dropped; *logits (width floats,
 *  when non-NULL; the record must keep them) is the row the prefill left at T.  False when the record is not the
 *  bank's or a copy failed (its state then describes no position). */
bool pulsar_ckpt_record_restore(pulsar_ckpt_store *st, uint32_t bank, const pulsar_ckpt_record *r, float *logits,
                                uint32_t width);
/** L313: put the INSTALLED bank `dst` at ANOTHER bank's state at T -- `src`'s grid checkpoint at T (r NULL) or its
 *  record `r` at T: every pool's rows below T copied from src's own pools (ops->bank_pools), then src's checkpoint
 *  walked into dst, exactly what a restore on src would leave there (a published checkpoint is any bank's to restore
 *  while its own bank's history still reaches T, the frontier rule).  dst's own checkpoints and records go (its
 *  history is replaced); it keeps a copy of the one it now stands at.  False, said, when the banks' pools do not pair,
 *  one is freed, or a copy failed (dst's state then describes no position). */
bool pulsar_ckpt_adopt(pulsar_ckpt_store *st, uint32_t dst, uint32_t src, uint32_t T, const pulsar_ckpt_record *r);
/** Where a walk's FINAL chunk [pos0, end) ends (P13, every walk's): at the prompt's last grid point when that lies
 *  strictly inside it -- so the prefill leaves the checkpoint the next turn resumes from -- else at `end`.  The
 *  caller keeps an image block whole around it. */
static inline uint32_t pulsar_ckpt_final_cut(const pulsar_ckpt_store *st, uint32_t pos0, uint32_t end) {
    const uint32_t g = pulsar_ckpt_grid_floor(st, end);
    return g > pos0 && g < end ? g : end;
}

/** DeepSeek V4's answer (kv_state_ds4.cpp); its state is the pulsar_gpu_graph. */
extern const pulsar_kv_state_ops PULSAR_KV_STATE_DS4;

#endif

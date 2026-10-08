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
 *     the checkpoint at G -- a chain of them rebuilds a bank (session_payload.cpp, then
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
 * Qwen3.8-Flash-Next (`qwen4_exp`, the L251 lane -- family_qwen.h on work/l251-serve) maps as:
 *   pools  per QSA layer (12): KV rows, 1 token/row, pulsar_qwen_kv_row_bytes (1056 B, E4M3 + E8M0);
 *          pooled indexer keys, idx_block (4) tokens/row, pulsar_qwen_index_row_bytes (256 B);
 *   slot   per GDN layer (36): the DeltaNet recurrent state (pulsar_qwen_gdn_state_bytes, 3.1 MB
 *          fp32) and the causal-conv tail (pulsar_qwen_gdn_conv_bytes); per QSA layer: the partial
 *          index tail block (pulsar_qwen_index_tail_bytes); the PLE layer's dilated-conv history
 *          (pulsar_qwen_ple_conv_bytes) -- ~118 MB, so the slot budget is ~2 per bank, not 16;
 *          the n-gram context is a function of the tokens and needs no copy;
 *   grid   a multiple of idx_block (4) that its GDN chunked-prefill kernel makes a resume exact on
 *          -- a census of Qwen's own kernels decides it (L183/L195 for V4), not this file.
 * Qwen's state lives in pulsar_qwen_state, not pulsar_gpu_graph: the ops take an opaque `state`,
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
} pulsar_kv_state_ops;

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
/** `bank`'s history now reaches only `tokens`: the checkpoints above it go. */
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
/** Where a walk's FINAL chunk [pos0, end) ends (P13, every walk's): at the prompt's last grid point when that lies
 *  strictly inside it -- so the prefill leaves the checkpoint the next turn resumes from -- else at `end`.  The
 *  caller keeps an image block whole around it. */
static inline uint32_t pulsar_ckpt_final_cut(const pulsar_ckpt_store *st, uint32_t pos0, uint32_t end) {
    const uint32_t g = pulsar_ckpt_grid_floor(st, end);
    return g > pos0 && g < end ? g : end;
}

/** DeepSeek V4's answer (kv_state_ds4.cpp); its state is the pulsar_gpu_graph. */
extern const pulsar_kv_state_ops PULSAR_KV_STATE_DS4;
/** Qwen3.8-Flash-Next's (kv_state_qwen.cpp, L266); its state is the pulsar_qwen_state. */
extern const pulsar_kv_state_ops PULSAR_KV_STATE_QWEN;

#endif

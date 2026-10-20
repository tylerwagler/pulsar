/* step_driver.h -- the core row-step driver (L295 F4): a family supplies its blocks, the core runs every step.
 *
 * What a session operation does to the rows of a step is family-neutral: build the rows (one a bank for a decode,
 * runs of consecutive positions for a prefill chunk, a verify, or a fused step's prompt runs), check them against
 * the banks' positions, upload them, run the forward -- embed, then the layer plan block by block, then the head
 * over the rows the caller asked for -- read the logits back, let the drafter's lane absorb the rows, and move each
 * bank's position and prefill frontier.  That is this driver (step_driver.cpp): eval_row, decode_multiseq,
 * decode_mixed (and the speculation lane's verify through it), decode_fused, the sync's prefill chunk, and the
 * speculation round's verify target (verify, commit, cut).  Its per-session buffers are pulsar_driver_state; the
 * state the steps advance is the family's declared state (state_kinds.h, pulsar_session::states).
 *
 * A family supplies (pulsar_driver_ops): its session state (which EXTENDS pulsar_driver_state), its trunk step
 * object (which EXTENDS pulsar_driver_step with the family's own fields -- weights, activation slots), the embed,
 * ONE BLOCK OP PER LAYER KIND (indexed by pulsar_layer_kind: the layer plan is the authority for which runs at
 * layer il), the head, the drafter lane's absorb, and its scratch arenas.  The ops read the step and write its
 * slots; the driver never names a layer's internals.  DeepSeek keeps its own forward (no driver ops).
 *
 * Included from pulsar_engine_internal.h after family.h, before family_qwen.h (whose state and step derive from
 * these); the driver's entry points are declared beside the sync driver's there.  Not standalone. */
#ifndef PULSAR_STEP_DRIVER_H
#define PULSAR_STEP_DRIVER_H

#include <stdbool.h>
#include <stdint.h>
#include <vector>

/** The rows one step heads into the logits slab -- also the widest verify (every bank's run together), so the
 *  depth of the declared state's verify capture (state_kinds.h PULSAR_STATE_VERIFY_ROWS) and the fused step's head
 *  budget (pulsar_family_session_ops::fused_heads_max).  The shared verify width, PULSAR_SPEC_ROW_BUDGET (32),
 *  DeepSeek's too (L288 C1, 2026-10-14: it was 16, the M-independent width).  Decode rows past
 *  PULSAR_GPU_MNEUTRAL_ROWS_MAX keep the decode arms in 16-row blocks (the kernels block them: the dense EXL3 GEMV,
 *  the W8A16 rows kernel, the GR read; the routed EXL3 decode GEMM is M-independent at any width), so every row of a
 *  wide verify is the bytes a one-token decode gives (verify-width-probe-qwen: identical rows).  The width a round
 *  actually carries is the server's pick (pulsar_engine_set_spec_width, server_sched.cpp width_price_pick). */
#define PULSAR_DRIVER_HEADS_MAX ((uint32_t)PULSAR_SPEC_ROW_BUDGET)
/** The scratch arenas a family's ops may own. */
#define PULSAR_DRIVER_SCRATCH_MAX 16u

/** How a step's rows are laid out. */
typedef enum {
    /** Each row is its own bank's NEXT token (one row per bank, any mix of banks). */
    PULSAR_STEP_DECODE = 0,
    /** Rows are runs of consecutive positions, one bank each (pulsar_driver_step::run_first): a prefill chunk is one
     *  run; a verify of several banks (L272 P1 S4) one run per bank, a fused step's prompt runs after them. */
    PULSAR_STEP_PREFILL = 1,
} pulsar_step_mode;

/** The step rows a forward heads, in logits-row order: logits row i is step row row[i] (L284 #2: a fused step heads
 *  its verify rows, then its prompt runs' last rows).  n > PULSAR_DRIVER_HEADS_MAX is refused. */
struct pulsar_driver_heads {
    uint32_t n = 0;
    uint32_t row[PULSAR_DRIVER_HEADS_MAX] = {};
};
/** Step rows [row0, row0 + n). */
pulsar_driver_heads pulsar_driver_heads_span(uint32_t row0, uint32_t n);

struct pulsar_state_set;
struct pulsar_imatrix_tap;
struct pulsar_tp;
struct pulsar_mtp_ops;
struct pulsar_mtp_lane;
struct pulsar_block_lane;

/** The core's per-session driver buffers.  A family's session state EXTENDS it (C++ derivation; the family's
 *  create allocates the derived object, the driver fills this part). */
struct pulsar_driver_state {
    uint32_t max_rows;                  ///< rows one step carries (the session's prefill chunk)
    pulsar_gpu_tensor *logits;          ///< [PULSAR_DRIVER_HEADS_MAX][logits width] f32: the head's rows
    pulsar_gpu_tensor *row_pos;         ///< [max_rows] i32: the step's positions (enqueued, never waited for)
    pulsar_gpu_tensor *row_bank;        ///< [max_rows] i32: the step's banks
    pulsar_gpu_tensor *run_first_dev;   ///< [PULSAR_DRIVER_HEADS_MAX + 1] i32: a ragged verify's run offsets (with a verify capture)
    pulsar_gpu_tensor *scratch[PULSAR_DRIVER_SCRATCH_MAX];   ///< each op arena, sized by the family's scratch_bytes (NULL = none)
    /* L266 step 7: the session's share of the engine's tensor-parallel transport (NULL tp = one GPU): the row
     * all-reduce's stage ticket and the exchange seq, advanced identically on every rank; the vocab gather's own
     * slice. */
    struct pulsar_tp *tp;
    void *tp_slab_dev, *tp_bulk_dev;
    pulsar_gpu_tensor *tp_ticket;
    uint64_t tp_seq;
    pulsar_gpu_tensor *tp_vocab_own;
    uint64_t tp_vocab_seq;
    /** L296 N9: the chained-MTP drafter's lane (spec_mtp.h) when the family's ops declare modules, else NULL */
    struct pulsar_mtp_lane *mtp;
    /** L298: the block drafter's lane (spec_block.h) when the engine loaded one, else NULL */
    struct pulsar_block_lane *block;
};

/** Everything an op needs for one step: the core's part.  A family's step EXTENDS it with what its ops read of
 *  their own (weights, the activation slots the lane runs on); the driver fills this part every step.  Read-only to
 *  the ops (the tensors' CONTENTS are what ops write). */
struct pulsar_driver_step {
    const pulsar_model *model;            ///< tensor_map_base() for the weights
    const pulsar_layer_plan *plan;
    struct pulsar_state_set *states;      ///< the session's declared state (state_kinds.h)
    pulsar_driver_state *buf;             ///< the session's driver buffers
    pulsar_step_mode mode;
    uint32_t n_rows;
    const int32_t *tokens;                ///< [n_rows] host token ids
    const int32_t *pos;                   ///< [n_rows] host positions
    const int32_t *bank;                  ///< [n_rows] host bank ids
    /** L268: the family's per-row rope positions (pulsar_driver_ops::rope), or NULL = text positions */
    const uint32_t *rope;
    /** L251 MTP: rows [0, n_dec) are a VERIFY (PREFILL mode, <= PULSAR_DRIVER_HEADS_MAX rows): the ops write their
     *  per-row captures into the declared state's (pulsar_state_capture_rows); the core kept each run's open blocks
     *  (pulsar_state_verify_capture).  L284 #2: rows [n_dec, n_rows) are prompt runs, never captured. */
    bool verify;
    /** L272 P1 S4: PREFILL rows as runs -- run k is rows [run_first[k], run_first[k + 1]) of one bank at consecutive
     *  positions: the verify's runs (one a bank) first, then (L284 #2) the fused step's prompt runs from row n_dec.
     *  n_runs 1 = the classic one-bank chunk, or a one-bank verify; 0 on a DECODE step. */
    uint32_t n_runs;
    const uint32_t *run_first;
    /** L284 #2: the row-kind boundary.  Rows [0, n_dec) take the decode arms (what decode / verify rows are graded
     *  against), rows [n_dec, n_rows) the prompt arms (the same bytes at every row count, L266). */
    uint32_t n_dec;
    /** L284 P15: the importance-matrix observer while a collection runs, else NULL (the trunk's steps only). */
    struct pulsar_imatrix_tap *tap;
};

/** L284 #2: n_dec for a step -- a DECODE step is all decode rows; a PREFILL step's decode rows are its verify rows
 *  [0, n_verify) and the rest are prompt rows, so a prompt cut anywhere (or behind a verify, the fused step) is
 *  byte-identical to one prefilled whole. */
static inline uint32_t pulsar_step_n_dec(pulsar_step_mode mode, uint32_t n_verify, uint32_t n_rows) {
    return mode == PULSAR_STEP_PREFILL ? n_verify : n_rows;
}
/** L284 #2: the verify runs of a PREFILL step -- its runs that start before n_dec (each lies wholly inside). */
static inline uint32_t pulsar_step_verify_runs(const pulsar_driver_step *st) {
    uint32_t k = 0;
    while (k < st->n_runs && st->run_first[k] < st->n_dec) k++;
    return k;
}

/** One layer's block: reads and writes the step's slots for layer il.  false after printing what failed; the
 *  driver refuses the step. */
typedef bool (*pulsar_block_fn)(const pulsar_driver_step *st, uint32_t il);

/** What a family supplies to run on the driver (pulsar_family::driver). */
typedef struct pulsar_driver_ops {
    /** The family's session state for `n_banks` banks (its own activation slots and drafter lane; the driver fills
     *  the pulsar_driver_state part and builds the declared state after it).  Allocates only through the tensor
     *  allocators (the create is also the session's price).  NULL (said) on a failure, nothing left allocated. */
    pulsar_driver_state *(*create)(pulsar_session *s, uint32_t n_banks);
    /** Free what create made (the driver freed its own part and the declared state first). */
    void (*destroy)(pulsar_session *s);
    /** Does the session verify (a drafter lane exists): the declared state's capture and the verify buffers are
     *  built.  Read once, at create. */
    bool (*verifies)(const pulsar_session *s);
    /** The family's trunk step object (its own fields set; the driver fills the core part each step). */
    pulsar_driver_step *(*trunk)(pulsar_session *s);
    /** Bytes of scratch arena `arena` (< n_scratch) at max_rows rows and a pool capacity of ctx tokens (0 = none). */
    uint64_t (*scratch_bytes)(uint32_t arena, uint32_t max_rows, uint32_t ctx);
    uint32_t n_scratch;
    /** tokens -> the step's slots */
    bool (*embed)(const pulsar_driver_step *st);
    /** ONE block op per layer kind, indexed by pulsar_layer_kind (NULL = the family has no such layer: a plan that
     *  holds one is refused at create).  The plan's kind[il] is the only dispatch key; block[PULSAR_LAYER_BLOCKS]
     *  composes layer il from the plan's desc[il] (family.h pulsar_layer_desc) -- one layer description, the plan. */
    pulsar_block_fn block[PULSAR_LAYER_KIND_COUNT];
    /** Step rows [row0, row0 + n) -> logits rows [out0, out0 + n) of the driver's slab, out0 + n <=
     *  PULSAR_DRIVER_HEADS_MAX.  Called once per span of the head list (consecutive rows of one row kind). */
    bool (*head)(const pulsar_driver_step *st, uint32_t row0, uint32_t n, uint32_t out0);
    /** After every forward that started: settle what the step left in flight; on a completed step its own checks
     *  (rule 9).  Returns ok && the step is good. */
    bool (*step_end)(const pulsar_driver_step *st, bool ok);
    /** After a trunk step over rows (tokens, pos, bank) -- step rows [row0, row0 + n) -- the drafter lane's lockstep:
     *  every row whose next token is now known gets its drafter row (PREFILL: one bank's run; DECODE: one row a bank).
     *  `eval`: the session's single-row eval (a family's diagnostics may head it).  NULL = no drafter lane; a family
     *  that drafts with its MTP modules sets the chained-MTP lane's (spec_mtp.h pulsar_mtp_absorb). */
    bool (*absorb)(pulsar_session *s, pulsar_step_mode mode, const int32_t *tokens, const int32_t *pos,
                   const int32_t *bank, uint32_t n, uint32_t row0, bool eval);
    /** L296 N9: the family's MTP module declaration (spec_mtp.h), or NULL = none.  When it declares modules the
     *  driver builds the chained-MTP drafter's lane (pulsar_driver_state::mtp) before the declared state. */
    const struct pulsar_mtp_ops *mtp;
    /** Optional (L268): the step's multi-axis rope positions into *out, or NULL = text positions. */
    const uint32_t *(*rope)(const pulsar_session *s, const int32_t *pos, const int32_t *bank, uint32_t n,
                            std::vector<uint32_t> *out);
    /** Optional (L268): a sync chunk's image rows into the step's slots (pulsar_image_merge_chunk's writer).  NULL =
     *  the family serves no images (a sync with images is refused). */
    bool (*image_rows)(const pulsar_driver_step *st, const uint16_t *rows, uint32_t n_rows, uint32_t row0,
                       uint32_t n_tokens);
    /** Optional: the per-layer diagnostic tap, called after layer il while PULSAR_CUDA_GRAPH_DUMP_PREFIX is armed. */
    void (*dump)(const pulsar_driver_step *st, uint32_t il);
    /** The artifact digest the session's segments are keyed by (pulsar_ckpt_store::artifact). */
    uint64_t (*artifact)(const pulsar_engine *e);
    /** Optional (L296 perf): the step's HOST HALF -- every upload its launches read (token ids, sequence tables,
     *  attention descriptors), run by the driver before the step's launches on every step.  A family that supplies
     *  it makes embed, every block and the head LAUNCH-ONLY: their launches read the device and the step's shape,
     *  nothing a step's host values decide.  *graph_key = what those launches depend on beyond the driver's own key
     *  (the session's buffers, the step's kind, the row count, the head list), nonzero; 0 = this step's launches
     *  must run eager.  A DECODE step, or a pure verify step (every row a verify row; the key must then carry the
     *  step's run structure too), with a key runs its launches as ONE captured CUDA graph per key
     *  (pulsar_gpu_seg_*), replayed after; the launches are the same either way, so the bytes are (nemotron-fwd-gate
     *  X7: graphed == eager; mtp-spec-gate on the verify lane).  NULL = the ops upload as they go and every step
     *  runs eager. */
    bool (*stage)(const pulsar_driver_step *st, uint64_t *graph_key);
} pulsar_driver_ops;

/** The first thing a forward over `plan` would need that `ops` lacks -- "embed", the block of a layer kind (its
 *  layer in *at_layer), "head" -- or NULL when nothing is missing.  The driver refuses a session with exactly this. */
const char *pulsar_driver_first_missing(const pulsar_driver_ops *ops, const pulsar_layer_plan *plan,
                                        uint32_t *at_layer);
/** L266 step 7: a block's tensor-parallel exchange -- the group's sum of the step's partial output rows `y` (f32
 *  [n_rows][width]) in exchange slot `slot` (the layer's; a drafter layer's after the trunk's).  One GPU: nothing. */
bool pulsar_driver_tp_allreduce(const pulsar_driver_step *st, uint32_t slot, pulsar_gpu_tensor *y, uint32_t width,
                                const char *what);
/** The speculation round's verify target for a driver family (spec_ops.h): the verify is a heads-on-every-row
 *  decode_mixed, the commit the declared state's (pulsar_state_commit), the cut an invalidate. */
extern const pulsar_spec_target_ops PULSAR_DRIVER_SPEC_TARGET;

#endif /* PULSAR_STEP_DRIVER_H */

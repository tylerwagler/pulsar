/** @file spec_mtp.h
 *  @brief The chained-MTP drafter: one drafter plugin for every family whose artifact carries MTP modules. */
/* spec_mtp.h -- the chained-MTP drafter (L295 F6, L296 N9): one drafter plugin for every family whose artifact carries
 * multi-token-prediction modules, on the core row-step driver (step_driver.h).
 *
 * WHAT AN MTP MODULE IS.  A module at position p reads a CONDITIONING ROW -- the trunk's output for the token at p (or,
 * deeper in a chain, the previous module step's output) -- and the token at p + 1, writes its own state at p (its
 * attention KV: the declared state's drafter slot), and predicts the token at p + 2 through a head.  A family DECLARES
 * its modules (pulsar_mtp_ops); THIS FILE's lane owns everything else:
 *
 *   lockstep   after every trunk step (the driver's absorb op = pulsar_mtp_absorb) each row whose next token is now
 *              known gets its module row -- in a prompt chunk the next prompt token, in a decode step the token the
 *              step fed -- and each bank's last row is PARKED (its conditioning row in the family's pending decl, the
 *              lane's pend_pos) until its token arrives;
 *   draft      a chain of up to the session's depth: step 1 is the parked row with the round's carry x (the lockstep
 *              row at p - 1, headed: d_1), step j > 1 the previous step's own output row with d_{j-1} at p + j - 2.
 *              Step j runs module min(j, n_modules) - 1: ONE module is applied recursively (Nemotron's card: "a
 *              shared MTP head predicts future tokens recursively"; vLLM reuses the one module per step), N modules
 *              each take their own step and the last repeats past them.  The chain is the shared stop rule
 *              (spec_ops.h) and the priced depth (L288) run token by token; every bank's chain runs in LOCKSTEP --
 *              one module step a position for all of them, the head once a step;
 *   absorb     after a round's commit, the kept verify rows' module rows (their trunk rows, the tokens after them) of
 *              every bank in ONE step of decode rows, and each bank's last kept row parked;
 *   harvest    nothing deferred (the chain's picks are read as it runs).
 *
 * The lane's per-bank counters are kept in step with the declared state by the plan's drafter hooks
 * (pulsar_mtp_drafter_at / pulsar_mtp_drafter_frontier: a restore or a payload at T parks the trunk row at T - 1; a
 * frontier is only a payload's while nothing is drafted past it).  A module's position-addressed state needs no undo:
 * a draft row at p + j is rewritten by the absorb before any true row reads it.  A family whose drafter slot holds
 * state position addressing cannot undo (Qwen's index stage) saves it before a chain writes draft rows and gets it
 * back before the absorb (stage_save / stage_restore).
 *
 * A family's drafter numbers (depth, the stop threshold, the adaptive depth's bounds) are its own pulsar_drafter_ops
 * built with PULSAR_MTP_DRAFTER_OPS; the functions are this file's.  Include after pulsar_engine_internal.h. */
#ifndef PULSAR_SPEC_MTP_H
#define PULSAR_SPEC_MTP_H

#include "spec_internal.h"

/** The deepest chain: the round record holds a base and 15 drafts (spec_redraft_req), and a bank's verify run (base +
 *  drafts) fits the verify width. */
#define PULSAR_MTP_DRAFT_MAX 15u
static_assert(PULSAR_MTP_DRAFT_MAX + 1u <= PULSAR_DRIVER_HEADS_MAX, "a bank's verify run fits the verify width");

/** One headed row's greedy pick: the vocabulary id of the head's first maximum and its softmax probability (the
 *  draft's confidence). */
typedef struct {
    /// The vocabulary id of the head row's first maximum.
    int32_t id;
    /// Its softmax probability, in [0, 1].
    float conf;
} pulsar_mtp_pick;

/** One module step the lane asks of the family: `n` rows, row i the module's input for (conditioning row i of
 *  `cond`, tokens[i]) at pos[i] of bank[i]. */
typedef struct {
    uint32_t module;            ///< which declared module runs (< modules(e)); its state is drafter slot n_trunk + module
    pulsar_step_mode mode;      ///< DECODE: one row a bank; PREFILL: one bank's run of consecutive positions
    /** PREFILL only: the rows are runs of any number of banks (one run a bank) taken as DECODE rows -- the decode
     *  arms, the verify's row kind (the absorb of every bank's kept rows); without it a PREFILL step is one bank's
     *  prompt rows at the prompt arms (a sync chunk's lockstep). */
    bool decode_rows;
    const int32_t *tokens;      ///< [n] the token after each conditioning row's: the module's input token
    const int32_t *pos;         ///< [n] the position the row writes (its conditioning row's)
    const int32_t *bank;        ///< [n]
    /// Rows in the step (the length of tokens / pos / bank).
    uint32_t n;
    const pulsar_gpu_tensor *cond;   ///< the conditioning rows: row i at i * cond_bytes (the lane's)
    uint32_t head_n;            ///< rows [0, head_n) are headed (0 = none)
    pulsar_mtp_pick *pick;      ///< [head_n] their picks, reduced on the device (NULL = none)
    float *logits;              ///< host [head_n][head_width]: their logits rows (NULL = not read back)
} pulsar_mtp_step;

/** What a family declares to have its MTP module(s) drafted (pulsar_driver_ops::mtp). */
typedef struct pulsar_mtp_ops {
    /** The FIXED decl (capture NONE, `bytes` = one conditioning row) on drafter slot n_trunk where a bank's parked
     *  row lives -- walked with the bank (checkpoints, payloads) like any FIXED state. */
    uint32_t pend_decl;
    /** The modules the opened artifact carries and drafts with (0 = none: no lane, no drafter). */
    uint32_t (*modules)(const pulsar_engine *e);
    /** Bytes of one conditioning row (the lane's staging rows and the pending decl's `bytes`). */
    uint64_t (*cond_bytes)(const pulsar_engine *e);
    /** Floats a headed row's logits hold: the head's width (the vocabulary, or a draft head's rows). */
    uint32_t (*head_width)(const pulsar_engine *e);
    /** The conditioning rows of the LAST TRUNK STEP's rows [row0, row0 + n) into `dst` at byte `off` (one row each,
     *  cond_bytes apart), on the session stream: what the trunk hands a module (Nemotron: norm_f of the residual,
     *  vLLM's target hidden; Qwen: the HC streams).  Valid until the next trunk step. */
    bool (*trunk_rows)(pulsar_session *s, uint32_t row0, uint32_t n, pulsar_gpu_tensor *dst, uint64_t off);
    /** After a module step: its rows' OUTPUT conditioning (row i at i * cond_bytes) -- the next chain step's input. */
    pulsar_gpu_tensor *(*out_rows)(pulsar_session *s);
    /** One module step.  false (said) = refused or failed; the lane drops the rows' banks' parked rows. */
    bool (*forward)(pulsar_session *s, const pulsar_mtp_step *m);
    /** L302: the last module step's headed rows on the device -- row r at r * head_width floats -- that a sampled
     *  chain draws its drafts from (THE sampler, the request's params), and *id_map the token id of each row index
     *  (int32 [head_width]; NULL = the index is the id). */
    const pulsar_gpu_tensor *(*q_rows)(pulsar_session *s, const pulsar_gpu_tensor **id_map);
    /** Optional.  A chain is about to write draft rows on `bank`'s drafter slot into state position addressing cannot
     *  undo: keep it (Qwen: the index stage); stage_restore puts it back before the bank's next true rows. */
    bool (*stage_save)(pulsar_session *s, uint32_t bank);
    /** Optional, paired with stage_save: put `bank`'s saved stage back before its next true rows. */
    bool (*stage_restore)(pulsar_session *s, uint32_t bank);
    /** Optional, a hint: ids the next verify of `bank` will embed, in order, as the chain drafts them -- the first
     *  call of a chain (`chain_start`) the carry (and a copy chain's drafts), then each draft (Qwen: the PLE rows'
     *  page-cache prefetch, its n-gram context advanced along the chain). */
    void (*prefetch)(pulsar_session *s, uint32_t bank, const int32_t *ids, uint32_t n, bool chain_start);
    /** A module row's bytes are the same whatever rows share its step (every op of the module step at an
     *  M-independent arm), so the drafter slot a speculative stream leaves is the plain stream's byte for byte
     *  (tests/mtp_spec_gate M3 holds the family to it; false = graded there, not asserted -- it moves drafts only,
     *  never an emitted token). */
    bool row_exact;
} pulsar_mtp_ops;

/** L296 N9: a diagnostic -- while armed, every lockstep module step (pulsar_mtp_absorb) heads its rows (at most
 *  PULSAR_DRIVER_HEADS_MAX) and appends their logits rows and positions here (what the draft grade reads: the first
 *  draft step's distribution at every position, teacher-forced by the trunk). */
typedef struct {
    float *logits;      ///< host [cap][head_width]
    int32_t *pos;       ///< host [cap]: the position each row wrote (its conditioning row's)
    /** @var pulsar_mtp_probe::n
     *  Rows appended so far (<= cap). */
    /// Rows the logits / pos buffers hold.
    uint32_t cap, n;
} pulsar_mtp_probe;

/** The lane's per-session state (pulsar_driver_state::mtp; the driver builds it when the family's ops declare
 *  modules). */
struct pulsar_mtp_lane {
    /// The family's MTP ops (static; not owned).
    const pulsar_mtp_ops *ops;
    /// ops->modules(e) at build: the module count the lane drafts with (> 0).
    uint32_t modules;
    /// Banks the session holds: the length of pend_pos / stage_dirty.
    uint32_t n_banks;
    /// ops->cond_bytes(e): bytes of one conditioning row.
    uint64_t cond_bytes;
    /// ops->head_width(e): floats in one headed row's logits.
    uint32_t head_width;
    pulsar_gpu_tensor *cond;    ///< [max_rows] the conditioning rows a module step reads
    uint32_t *pend_pos;         ///< [n_banks] the parked row's position; UINT32_MAX = none
    bool *stage_dirty;          ///< [n_banks] a chain wrote draft rows into the saved stage (stage_save)
    int32_t vtok[PULSAR_DRIVER_HEADS_MAX];   ///< the last verify step's tokens, by step row (pulsar_mtp_note_verify)
    /// Valid entries of vtok (<= PULSAR_DRIVER_HEADS_MAX).
    uint32_t n_vtok;
    pulsar_mtp_probe *probe;    ///< NULL = off
};

/** The driver's verify step over rows [0, n) is about to run with these tokens (step_driver.cpp): the absorb after
 *  its commit pairs a bank whose row is still parked -- no chain consumed it: the round's first after a sync, or one
 *  whose drafter did not run -- with its run's base token, so the module's state has no hole there. */
/// @param l the lane
/// @param tokens [n] the verify step's tokens, by step row
/// @param n rows (only the first PULSAR_DRIVER_HEADS_MAX are kept)
static inline void pulsar_mtp_note_verify(pulsar_mtp_lane *l, const int32_t *tokens, uint32_t n) {
    l->n_vtok = n < PULSAR_DRIVER_HEADS_MAX ? n : PULSAR_DRIVER_HEADS_MAX;
    for (uint32_t r = 0; r < l->n_vtok; r++) l->vtok[r] = tokens[r];
}

/** Build / free the lane (the driver's session create / destroy). */
/// @param s the session
/// @param ops the family's MTP ops (static)
/// @param n_banks banks the session holds
/// @param max_rows the widest step's rows (the conditioning staging)
/// @return the lane, freed with pulsar_mtp_lane_free; NULL (said) when the plan's MTP declaration is inconsistent or allocation failed
pulsar_mtp_lane *pulsar_mtp_lane_create(pulsar_session *s, const pulsar_mtp_ops *ops, uint32_t n_banks,
                                        uint32_t max_rows);
/// Free a lane from pulsar_mtp_lane_create (NULL: no-op).
/// @param l the lane
void pulsar_mtp_lane_free(pulsar_mtp_lane *l);

/** The driver's absorb op (pulsar_driver_ops::absorb) for a family that drafts with its MTP module(s). */
/// @param s the session
/// @param mode PREFILL: one bank's run; DECODE: one row a bank
/// @param tokens [n] the step's tokens
/// @param pos [n] their positions
/// @param bank [n] their banks
/// @param n rows
/// @param row0 the step row of the first
/// @param eval the single-row eval (unused here)
/// @return false = refused or failed (said); true with no lane
bool pulsar_mtp_absorb(pulsar_session *s, pulsar_step_mode mode, const int32_t *tokens, const int32_t *pos,
                       const int32_t *bank, uint32_t n, uint32_t row0, bool eval);

/** The plan's drafter hooks (pulsar_state_plan::drafter_at / drafter_frontier; ud = the driver state). */
/// @param ud the driver state
/// @param bank the bank
/// @param T the bank's new frontier (tokens)
void pulsar_mtp_drafter_at(void *ud, uint32_t bank, uint32_t T);
/// Check that a bank's lane state is at frontier T (not mid-draft), before a payload walks it.
/// @param ud the driver state
/// @param bank the bank
/// @param T the frontier
/// @param why receives the reason when not
/// @param whylen its size
/// @return true when at T (or no lane)
bool pulsar_mtp_drafter_frontier(void *ud, uint32_t bank, uint32_t T, char *why, size_t whylen);

/** The drafter's functions (PULSAR_MTP_DRAFTER_OPS). */
/// pulsar_drafter_ops::absorb_banked: the kept rows of the last verify, every bank.
/// @param s the session
/// @param rows [n] step rows
/// @param banks [n] their banks
/// @param next [n] the token after each
/// @param n rows
/// @return false = refused or failed (said)
bool pulsar_mtp_absorb_banked(pulsar_session *s, const uint32_t *rows, const uint32_t *banks, const int32_t *next,
                              uint32_t n);
/// pulsar_drafter_ops::draft for the installed bank.
/// @param s the session
/// @param next_base the carry
/// @param features_ready unused by this lane
/// @param temperature request temperature
/// @param top_k request top-k
/// @param top_p request top-p
/// @param min_p request min-p
/// @param rng the request's stream
/// @return drafts kept, or PULSAR_DRAFT_REFUSED (said; the pendings dropped): the round refuses (L306)
uint32_t pulsar_mtp_draft(pulsar_session *s, int next_base, bool features_ready, float temperature, int top_k,
                          float top_p, float min_p, uint64_t *rng);
/// pulsar_drafter_ops::draft_batch.
/// @param s the session
/// @param rounds [n] the rounds
/// @param banks [n] their banks
/// @param rngs [n] their streams
/// @param n rounds
/// @param err receives the reason
/// @param errlen its size
/// @return 0, or -1 refused
int pulsar_mtp_draft_batch(pulsar_session *s, pulsar_spec_round **rounds, const uint32_t *banks, uint64_t **rngs,
                           int n, char *err, size_t errlen);

/** A family's MTP drafter: its numbers (spec_ops.h pulsar_drafter_ops), the lane's functions. */
#define PULSAR_MTP_DRAFTER_OPS(NAME, DEPTH, DEPTH_MAX, TAU, ADAPT)                                                    \
    { (NAME), (DEPTH), (DEPTH_MAX), (TAU), (ADAPT), NULL, NULL, pulsar_mtp_absorb_banked, pulsar_mtp_draft, NULL,     \
      pulsar_mtp_draft_batch, NULL }

#endif /* PULSAR_SPEC_MTP_H */

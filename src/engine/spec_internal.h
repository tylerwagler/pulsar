/** @file spec_internal.h
 *  @brief The round API's private types, shared by session_spec.cpp and the target / drafter implementations. */
/* spec_internal.h -- the round API's private types, shared by the core (session_spec.cpp) and the
 * target / drafter implementations (spec_dspark.cpp; Qwen's in L272 P1 S3).  Not a standalone header:
 * include after pulsar_engine_internal.h. */
#ifndef PULSAR_SPEC_INTERNAL_H
#define PULSAR_SPEC_INTERNAL_H

#include "pulsar_engine_internal.h"
#include "spec_ops.h"

/* L150: what a bank's redraft needs, recorded by round_end in the batched lane
 * (which defers drafting), consumed by pulsar_session_spec_redraft_batch. */
/** One bank's deferred redraft (L150): the request round_end records in the batched lane, and the chain the
 *  batched redraft fills in for pulsar_session_spec_redraft_commit.  Lives in its round (pulsar_spec_round::redraft). */
typedef struct {
    bool     valid;         ///< round_end reached the drafting point with the drafter's features ready
    int32_t  next_base;     ///< the carry: batch position 0 of the next round
    uint32_t n_draft;       ///< the bank's depth at round_end (pulsar_spec_cur_depth), <= 16
    /// The rows the round that recorded this verified (pulsar_spec_round::n_batch).  Recorded, not read today.
    uint32_t n_batch;
    /// The drafts that round committed.  Recorded, not read today.
    int      commit;
    /** @var spec_redraft_req::top_p
     *  The request's top-p filter for sampled drafts. */
    /** @var spec_redraft_req::min_p
     *  The request's min-p filter for sampled drafts. */
    /// The request's sampling temperature for the next chain (> 0: drafts are sampled, see sample_drafts).
    float    temperature, top_p, min_p;
    /// The request's top-k filter for sampled drafts.
    int      top_k;
    /* results, filled by the batched redraft; committed into the bank's shadow
     * by pulsar_session_spec_redraft_commit under the server's bank switch */
    /// The batched redraft has filled this record (or it is a copy chain, drafted at init).
    bool     done;
    /// temperature > 0: the chain is drawn from the proposal q, not argmax.
    bool     sample_drafts;
    /// Drafts kept: refined[1..keep] after the stop rule and the verify pick; 0 = no chain.
    uint32_t keep;
    /// conf[0..keep) holds the drafts' confidences (false: the drafter gave none).
    bool     have_conf;
    bool     copied;        ///< L288: the drafts are a copy chain (pulsar_spec_copy_chain), not the drafter's
    /// The chain as token ids: [0] = next_base, [1..keep] = the drafts in order.
    int32_t  refined[17];
    /// conf[i]: draft i+1's confidence (valid when have_conf; negative = none).
    float    conf[16];
} spec_redraft_req;

/** A speculation round in flight on one bank: its base row, drafts and the target's pre-verify snapshot, from
 *  round_begin to round_end; freed with its snapshot when the round ends. */
typedef struct pulsar_spec_round {
    int32_t base;           ///< the round's base token (verify row 0, at saved_len)
    uint32_t K;             ///< draft depth this round: tokens the drafter proposed
    uint32_t n_batch;       ///< rows the verify forward evaluates (K + the base row)
    int saved_len;          ///< checkpoint length before the round, to trim back to on rejection
    int32_t pend[16];       ///< the drafted token ids, in draft order
    float pend_conf[16];    ///< confidence head's score per draft position
    bool copy;              ///< L288: the drafts are a copy chain (their verdict feeds the copy acceptance)
    /** The target's state snapshot taken before the verify (pulsar_spec_target_ops snapshot / restore /
     * release own its contents; malloc'd, freed with the round), so a partial accept can roll back. */
    void *snap;
    spec_redraft_req redraft;   ///< L150: batched-lane deferred redraft (see the typedef)
} pulsar_spec_round;

/* ---- the core's helpers the implementations use ---- */
/** The width of every host logits row the round API reads or stores (the engine's logits width). */
/// @param s the session
/// @return floats per logits row
static inline uint32_t spec_vocab(const pulsar_session *s) { return (uint32_t)s->engine->logits_width(); }
/** The session's current draft depth (the adaptive controller's, else the starting depth), 1..16. */
/// @param s the session
/// @return drafts per chain, 1 .. 16
uint32_t pulsar_spec_cur_depth(const pulsar_session *s);
/** The draft stop threshold in force: --spec-tau / PULSAR_SPEC_TAU, else the drafter's; 0 = no stop. */
/// @param e the engine
/// @return the threshold in force, in [0, 1]; 0 = no stop
float pulsar_spec_tau(const pulsar_engine *e);
/** THE stop rule (spec_ops.h): a draft whose confidence is under tau is its chain's last (tau <= 0: none is). */
/// @param conf the draft's confidence (negative = none: never stops)
/// @param tau the threshold
/// @return true when this draft ends its chain
static inline bool pulsar_spec_conf_stops(float conf, float tau) { return tau > 0.0f && conf < tau; }
/** The stop rule over a drafted chain: how many of its n drafts are kept -- up to and including the first that
 *  stops it. */
/// @param conf [n] the chain's confidences
/// @param n drafts in the chain
/// @param tau the threshold
/// @return drafts kept, 0 .. n
static inline uint32_t pulsar_spec_conf_keep(const float *conf, uint32_t n, float tau) {
    for (uint32_t k = 0; k < n; k++)
        if (pulsar_spec_conf_stops(conf[k], tau)) return k + 1u;
    return n;
}
/** L301: the verify pick (verify_curve.cpp has the rule) over a round's drafted records: every draft of the nq records
 *  ranked by survival (the product of the confidences up to it; a confidence under 0 counts as 1), the drafts a
 *  round of `n_chains` base rows verifies at its best tokens per ms on the measured curve -- each record's keep cut
 *  to its share, a prefix.  No curve (a tensor-parallel group, a classic server): the row budget alone. */
/// @param s the session (its engine's verify curve)
/// @param n_chains base rows the round verifies
/// @param qs [nq] the records; each one's keep is cut in place
/// @param nq records
void pulsar_spec_keep_picked(const pulsar_session *s, uint32_t n_chains, spec_redraft_req *const *qs, uint32_t nq);
/** L301: which of `n_live` chains drafting in lockstep take the next step: the k best by survival (run[]) whose next
 *  drafts -- each worth at most its chain's survival -- raise the round's best tokens per ms the most once the step
 *  that drafts them (the drafter's measured step at k chains) and `spent_ms` already drafting are paid.  `drafted`
 *  holds the survivals the round has drafted so far (n_drafted, any order); the round verifies `n_chains` base
 *  rows.  go[i] set for those that go on; returns how many.  No curve: a chain goes on while fewer drafts than the
 *  row budget outrank it. */
/// @param s the session
/// @param n_chains base rows the round verifies
/// @param drafted [n_drafted] survivals drafted so far
/// @param n_drafted their count
/// @param run [n_live] each live chain's survival so far
/// @param n_live chains drafting
/// @param spent_ms drafting wall already spent this round, ms
/// @param go [n_live] out: set for the chains that take the step
/// @return how many go[] are set
uint32_t pulsar_spec_chain_step(const pulsar_session *s, uint32_t n_chains, const float *drafted, uint32_t n_drafted,
                                const float *run, uint32_t n_live, double spent_ms, bool *go);
/** L301: a verified round's calibration, into the engine's per-position accumulators (pulsar_spec_calibration):
 *  conf[i] the draft's confidence (under 0: none, skipped), label[i] its exact acceptance under the coupled draws --
 *  greedy: the row's argmax is the draft (0 / 1); sampled: the target's probability of the draft -- and hit[i]
 *  whether the walk accepted it, for the n positions the walk reached.  `copy`: the drafts were a copy chain. */
/// @param e the engine (its accumulators)
/// @param copy the drafts were a copy chain
/// @param conf [n] confidences
/// @param label [n] exact acceptance labels
/// @param hit [n] walk verdicts (0 / 1)
/// @param n positions the walk reached
void pulsar_spec_calibration_note(pulsar_engine *e, bool copy, const float *conf, const float *label,
                                  const uint8_t *hit, uint32_t n);
/** L288: the copy chain of `bank` -- when its last PULSAR_SPEC_COPY_MATCH tokens (its history, then `next_base`)
 *  occurred earlier in its history, the tokens that followed the latest such copy, at most `max`, into out[]; their
 *  count, 0 for none (session_spec.cpp has the rule). */
#define PULSAR_SPEC_COPY_MATCH 8u
/// The copy chain of `bank` (see the block above PULSAR_SPEC_COPY_MATCH).
/// @param s the session
/// @param bank the bank whose history is scanned
/// @param next_base the carry (the history's next token)
/// @param max the most drafts
/// @param out [max] receives the drafts
/// @return drafts written, 0 = none
uint32_t pulsar_spec_copy_chain(pulsar_session *s, uint32_t bank, int32_t next_base, uint32_t max, int32_t *out);
/** L308: an estimate unobserved for this many committed tokens is stale (Flare's 32-token horizon). */
#define PULSAR_SPEC_COPY_REPROBE 32u
/** L288 / L308: the confidences of a copy chain's first n drafts from the request's copy estimate `sp` (the one
 *  estimate: copy_accept, copy_unseen), into conf[0..n), and how many of them the stop rule at tau admits (0 = the
 *  request does not copy now).  None measured yet: 1 each.  A fresh estimate: copy_accept each, none admitted under
 *  tau.  A stale one (copy_unseen >= PULSAR_SPEC_COPY_REPROBE): the first draft at its upper bound, 1, the rest at
 *  the estimate -- a probe, cut by the stop rule (session_spec.cpp has the rationale).  *reprobe = the chain is a
 *  probe the fresh estimate would have refused. */
/// @param sp the request's carry state (its copy estimate)
/// @param tau the stop threshold
/// @param n drafts to price
/// @param conf [n] out: their confidences
/// @param reprobe out: the chain is a stale-estimate probe
/// @return drafts admitted, 0 .. n
uint32_t pulsar_spec_copy_price(const pulsar_spec_carry_state *sp, float tau, uint32_t n, float *conf, bool *reprobe);
/** L308: a verified copy chain's verdict, `commit` of its `K` drafts accepted, into the estimate (a stale or
 *  unmeasured one is re-seeded by it); the estimate is fresh again. */
/// @param sp the request's carry state
/// @param commit drafts accepted
/// @param K drafts verified
void pulsar_spec_copy_observe(pulsar_spec_carry_state *sp, int commit, uint32_t K);
/** L308: `tokens` committed by a round that verified no copy chain: the estimate ages. */
/// @param sp the request's carry state
/// @param tokens tokens committed (saturates copy_unseen)
void pulsar_spec_copy_age(pulsar_spec_carry_state *sp, uint32_t tokens);
/** A draft record for the installed bank's next chain after `next_base` at the session's depth and these
 *  params -- what a drafter's draft / draft_batch fills (the rest of the record is the caller's). */
/// @param s the session
/// @param q the record, reset (a copy chain may be placed in it)
/// @param next_base the carry
/// @param temperature request temperature (> 0 = sampled drafts)
/// @param top_k request top-k
/// @param top_p request top-p
/// @param min_p request min-p
void pulsar_spec_draft_req_init(pulsar_session *s, spec_redraft_req *q, int next_base, float temperature, int top_k,
                                float top_p, float min_p);
/** Stamp a drafter's result record into the installed bank's pendings (the one way pendings are written). */
/// @param s the session
/// @param q the filled record
void pulsar_spec_redraft_stamp(pulsar_session *s, spec_redraft_req *q);
/** `bank`'s speculative shadow for a read: the live state for the installed bank, the saved carry's for
 *  another, NULL when nothing is saved. */
/// @param s the session
/// @param bank the bank
/// @return its shadow (owned by the session), or NULL
const pulsar_spec_carry_state *pulsar_spec_bank_shadow(pulsar_session *s, uint32_t bank);
/** The adaptive draft controller's CURRENT depth for `bank` (0 = no drafter or nothing valid). */
/// @param s the session
/// @param bank the bank
/// @return the depth, or 0
int pulsar_spec_bank_depth(pulsar_session *s, uint32_t bank);
/** Save / restore the session's speculative shadow (s->spec) into / from a bank carry's copy -- every family's
 *  bank save and restore call these. */
/// @param s the session
/// @param dst the carry's copy, overwritten
void pulsar_spec_shadow_save(pulsar_session *s, pulsar_spec_carry_state *dst);
/// The restore half of pulsar_spec_shadow_save.
/// @param s the session
/// @param src the carry's copy
void pulsar_spec_shadow_restore(pulsar_session *s, const pulsar_spec_carry_state *src);
/// @param s the session
/// @return the family's verify target hooks
static inline const pulsar_spec_target_ops *spec_target(const pulsar_session *s) { return s->engine->family->spec; }
/// @param s the session
/// @return the engine's drafter (NULL = none)
static inline const pulsar_drafter_ops *spec_drafter(const pulsar_session *s) { return s->engine->drafter_ops; }
/** The bank save / restore the batched lane runs per step: the family's bank ops. */
/// @param s the session
/// @param bank the bank to install
/// @return false = the family refused
static inline bool spec_bank_restore(pulsar_session *s, uint32_t bank) { return s->engine->family->banks->restore(s, bank); }
/// Save the installed bank through the family's bank ops.
/// @param s the session
/// @param bank the bank
static inline void spec_bank_save(pulsar_session *s, uint32_t bank) { s->engine->family->banks->save(s, bank); }

/* ---- the implementations ---- */
extern const pulsar_spec_target_ops k_ds4_spec_target;   ///< DeepSeek V4's verify hooks (spec_dspark.cpp)
extern const pulsar_drafter_ops k_dspark_drafter;        ///< the DSpark drafter (spec_dspark.cpp)

#endif /* PULSAR_SPEC_INTERNAL_H */

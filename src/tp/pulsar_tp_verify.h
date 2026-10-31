/** @file
 *  TP speculative-verify grading: the (full_accept, replay_n) decision rule and the leader round-trip. */
#ifndef PULSAR_TP_VERIFY_H
#define PULSAR_TP_VERIFY_H

#include "tp/pulsar_tp.h"

#include <stdbool.h>
#include <stdint.h>

/*
 * Speculative-verify reference grading (host half of the reference pass; the
 * seed of engine-slice 4e).  The leader keeps a reference (golden) run over
 * each draft block and, before committing, must turn per-position
 * draft-vs-reference agreement into the (full_accept, replay_n) decision the
 * transport broadcasts on VERIFY_COMMIT.
 *
 * Everything here is pure host C++: the reference computation itself is
 * funneled through a hook (the engine binds real graded rows later), so the
 * decision rule and the leader round-trip are unit-tested off the GPU --
 * the same shape as the gate scheduler's write/read hooks.
 *
 * Decision rule (batched-verify convention): the draft and the reference
 * agree on an accepted prefix.  All-equal -> full accept (1, n).  On the
 * first mismatch at index k, stop BEFORE it: accept k rows (replay_n = k)
 * so the engine re-runs from the reference token at k.  A first-position
 * mismatch is an empty accept (replay_n 0).
 */

/// The leader's reference-grading hook for a verify round (see the file comment above).
typedef struct {
    /// Opaque context handed back to grade_matches.
    void *ud;
    /* For each of n drafts, report in match[i] whether the leader's reference
     * run agrees with the draft token at position i.  Called on the caller's
     * thread with a (n) match array; never after the round returns. */
    /// Fill match[i] (i < n) with whether the leader's reference agrees with drafts[i].
    void (*grade_matches)(void *ud, const int *drafts, uint32_t n, bool *match);
} pulsar_tp_verify_hooks;

/// A verify round's outcome, as broadcast on VERIFY_COMMIT.
typedef struct {
    /// 1 when every draft row matched (accept all n), else 0.
    int full_accept;   /* 1 when every draft row matched (accept all n) */
    /// Accepted rows: the matched prefix length; == n iff full_accept.
    uint32_t replay_n; /* accepted rows; == n iff full_accept */
} pulsar_tp_verify_decision;

/* Decide (full_accept, replay_n) from per-position matches.  Pure: no I/O. */
/// Turn per-position matches into the verify decision (the rule in the file comment).
/// @param match per-draft agreement, n entries (NULL or n == 0 gives an empty accept)
/// @param n     drafts in the block
/// @param d     receives the decision
void pulsar_tp_verify_decide(const bool *match, uint32_t n,
                             pulsar_tp_verify_decision *d);

/* Leader verify round-trip: send the drafts, wait the engine ack, grade via
 * hooks, broadcast the commit.  Returns 1 on success; 0 on transport failure
 * with err set.  The worker consumes the drafts (VERIFY) and the commit
 * (VERIFY_COMMIT) in the same lockstep the stress test pins. */
/// The leader's verify round: VERIFY, wait the ack, grade, send VERIFY_COMMIT.
/// @param tp         the transport (leader)
/// @param session_id the mirrored session
/// @param drafts     the draft tokens, n entries
/// @param n          drafts in the block
/// @param hooks      the reference-grading hook
/// @param err        receives the failure reason (may be NULL)
/// @param errlen     size of `err`
/// @return 1 on success, 0 on bad arguments or transport failure
int pulsar_tp_verify_leader_round(pulsar_tp *tp, uint64_t session_id,
                                  const int *drafts, uint32_t n,
                                  const pulsar_tp_verify_hooks *hooks,
                                  char *err, size_t errlen);

#endif

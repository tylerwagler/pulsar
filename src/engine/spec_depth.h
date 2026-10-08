/* spec_depth.h -- the adaptive draft depth (L107), as a rule any drafter can run.
 *
 * One round's update of the draft-depth cap from what the round converted.  The rule is DSpark's
 * (spec_dspark.cpp's k_dspark_depth, the 2026-08-25 sweep and its v2..v5 refinements), moved here in
 * L251 so it has one home and a gate (tests/spec_depth_gate.cpp pins DSpark's schedule).  The Qwen MTP
 * drafter measured it and does NOT use it (spec_qwen.cpp: its in-round stop wins); a drafter opts in
 * through pulsar_drafter_ops::adapt.  Pure host
 * logic: no engine state, no CUDA.
 *
 *   UP:    commit == depth (the whole chain was kept and accepted), no cooldown running, and the tail
 *          draft's confidence clears conf_up (a negative confidence -- no head ran -- passes).
 *   DOWN:  2 * commit < depth (less than half the depth converted) -- except that at veto_depth a
 *          still-confident tail (>= veto_conf) forgives ONE down-signal; a second consecutive one backs
 *          off regardless.  A down within excursion_rounds of the last up is a FAILED EXCURSION and
 *          starts a cooldown of cooldown_rounds in which UP is suppressed.
 * The result is clamped to [min, max].
 *
 * DSpark's history of the refinements (kept with the rule they shaped):
 *   v2  the down-veto -- single bad rounds at depth 5 with tail conf ~0.98 knocked depth down and cost
 *       ~2 t/s until the climb back; a still-believing calibrated head (>= 0.90 -> the measured
 *       1.000-accept band) forgives ONE down-signal.
 *   v3  the veto only at depth 5, the measured structured optimum: below 5 it regressed prose (occasional
 *       >= 0.90 tails, each forgiven round paying a deep draft that converts nothing); at 6 it delays the
 *       return to 5, and 6 lost on both regimes.
 *   v4  a cooldown after a failed excursion: raw-completion prose oscillated 2 -> 3 -> 4 -> crash
 *       (29 transitions / 192 tokens, -14%); tail conf does not separate good climbs from bad.
 *   v5  only a FAILED EXCURSION (a down within 2 rounds of the last up) starts it -- v4's blanket
 *       cooldown cost ~0.9 t/s on both server workloads by suppressing profitable climbs. */
#ifndef PULSAR_SPEC_DEPTH_H
#define PULSAR_SPEC_DEPTH_H

#include <stdint.h>

typedef struct {
    int min, max;             ///< the depth bounds
    float conf_up;            ///< UP needs the tail draft's confidence >= this
    int veto_depth;           ///< the one depth whose down-signal a confident tail may forgive (0 = never)
    float veto_conf;          ///< ... when the last verified draft's confidence >= this
    uint8_t cooldown_rounds;  ///< UP suppressed this long after a failed excursion
    uint8_t excursion_rounds; ///< a down within this many rounds of an up is a failed excursion
} pulsar_spec_depth_policy;

typedef struct {
    bool down_forgiven;       ///< one down-signal was vetoed; the next backs off regardless
    uint8_t rounds_since_up;  ///< saturating at 255
    uint8_t climb_cooldown;   ///< rounds left in which UP is suppressed
} pulsar_spec_depth_state;

/** The next depth after a round drafted at `depth`, verified K drafts (K <= depth) and committed
 *  `commit` of them.  conf[i] is draft i's confidence (negative: none).  Updates *st. */
static inline int pulsar_spec_depth_next(const pulsar_spec_depth_policy *p, pulsar_spec_depth_state *st,
                                         uint32_t depth, uint32_t K, int commit, const float *conf) {
    if (st->climb_cooldown) st->climb_cooldown--;
    if (st->rounds_since_up < 255u) st->rounds_since_up++;
    int next = (int)depth;
    if (2u * (uint32_t)commit < depth) {
        if (p->veto_depth > 0 && depth == (uint32_t)p->veto_depth && K > 0 && conf[K - 1] >= p->veto_conf &&
            !st->down_forgiven) {
            st->down_forgiven = true;
        } else {
            st->down_forgiven = false;
            if (st->rounds_since_up <= p->excursion_rounds) st->climb_cooldown = p->cooldown_rounds;
            next--;
        }
    } else if ((uint32_t)commit == depth && st->climb_cooldown == 0u &&
               (conf[depth - 1] >= p->conf_up || conf[depth - 1] < 0.0f)) {
        st->down_forgiven = false;
        st->rounds_since_up = 0u;
        next++;
    } else {
        st->down_forgiven = false;
    }
    if (next < p->min) next = p->min;
    if (next > p->max) next = p->max;
    return next;
}

#endif /* PULSAR_SPEC_DEPTH_H */

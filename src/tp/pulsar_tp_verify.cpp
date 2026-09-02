/*
 * Speculative-verify reference grading (host half of engine-slice 4e).
 * See pulsar_tp_verify.h for the contract: pure host C++, the reference
 * computation goes through hooks, covered by tests/tp_verify_test.
 */

#include "tp/pulsar_tp_verify.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void pulsar_tp_verify_decide(const bool *match, uint32_t n,
                             pulsar_tp_verify_decision *d) {
    if (!d) return;
    d->full_accept = 0;
    d->replay_n = 0;
    if (!match || n == 0) return;
    uint32_t i = 0;
    while (i < n && match[i]) i++;
    if (i == n) {
        d->full_accept = 1;
        d->replay_n = n;
        return;
    }
    d->replay_n = i; /* stop before the first mismatch */
}

int pulsar_tp_verify_leader_round(pulsar_tp *tp, uint64_t session_id,
                                  const int *drafts, uint32_t n,
                                  const pulsar_tp_verify_hooks *hooks,
                                  char *err, size_t errlen) {
    if (!tp || !drafts || !hooks || !hooks->grade_matches) {
        if (err && errlen) snprintf(err, errlen, "tp verify: null arg or hook");
        return 0;
    }
    if (!pulsar_tp_send_verify(tp, session_id, drafts, n)) {
        if (err && errlen) snprintf(err, errlen, "tp verify: send_verify failed");
        return 0;
    }
    if (!pulsar_tp_wait_command_ack(tp, session_id, "verify", err, errlen))
        return 0;
    bool *match = (bool *)malloc(n ? (size_t)n : 1u);
    if (!match) {
        if (err && errlen) snprintf(err, errlen, "tp verify: out of memory");
        return 0;
    }
    hooks->grade_matches(hooks->ud, drafts, n, match);
    pulsar_tp_verify_decision d;
    pulsar_tp_verify_decide(match, n, &d);
    free(match);
    if (!pulsar_tp_send_verify_commit(tp, d.full_accept, (int32_t)d.replay_n)) {
        if (err && errlen)
            snprintf(err, errlen, "tp verify: send_verify_commit failed");
        return 0;
    }
    return 1;
}

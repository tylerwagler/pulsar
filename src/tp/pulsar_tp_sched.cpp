/*
 * Gate scheduler — host half of the CUDA gate machinery (slice 4b).
 * See pulsar_tp_sched.h for the contract and the exchange-counter `e` rule.
 */

#include "tp/pulsar_tp_sched.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

struct pulsar_tp_sched {
    pulsar_tp *tp;
    pulsar_tp_sched_hooks hooks;
    uint64_t next_e;            /* next per-exchange counter (1-based) */
    char last_err[256];
};

static int sched_fail(pulsar_tp_sched *s, char *err, size_t errlen,
                      const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(s->last_err, sizeof(s->last_err), fmt, ap);
    va_end(ap);
    if (err && errlen) snprintf(err, errlen, "%s", s->last_err);
    return 0;
}

pulsar_tp_sched *pulsar_tp_sched_new(pulsar_tp *tp,
                                     const pulsar_tp_sched_hooks *hooks,
                                     char *err, size_t errlen) {
    if (!tp || !hooks || !hooks->write_partial || !hooks->read_partial) {
        if (err && errlen)
            snprintf(err, errlen, "tp sched: null transport or incomplete hooks");
        return NULL;
    }
    pulsar_tp_sched *s = (pulsar_tp_sched *)calloc(1, sizeof(*s));
    if (!s) {
        if (err && errlen) snprintf(err, errlen, "tp sched: out of memory");
        return NULL;
    }
    s->tp = tp;
    s->hooks = *hooks;
    s->next_e = 1;
    return s;
}

void pulsar_tp_sched_free(pulsar_tp_sched *s) {
    free(s);
}

int pulsar_tp_sched_decode_token(pulsar_tp_sched *s, char *err, size_t errlen) {
    const uint32_t n_layer = pulsar_tp_n_layer(s->tp);
    for (uint32_t layer = 0; layer < n_layer; layer++) {
        for (uint32_t gate = 0; gate < PULSAR_TP_GATES_PER_LAYER; gate++) {
            const uint64_t e = s->next_e++;
            if (!s->hooks.write_partial(s->hooks.ud, layer, gate, e))
                return sched_fail(s, err, errlen,
                                  "tp sched: write_partial l=%u g=%u e=%llu",
                                  layer, gate, (unsigned long long)e);
            if (!pulsar_tp_gate_exchange(s->tp, layer, gate, e))
                return sched_fail(s, err, errlen,
                                  "tp sched: gate_exchange l=%u g=%u e=%llu",
                                  layer, gate, (unsigned long long)e);
            if (!s->hooks.read_partial(s->hooks.ud, layer, gate, e))
                return sched_fail(s, err, errlen,
                                  "tp sched: read_partial l=%u g=%u e=%llu",
                                  layer, gate, (unsigned long long)e);
        }
    }
    return 1;
}

int pulsar_tp_sched_prefill_chunk(pulsar_tp_sched *s, uint64_t seq,
                                  const void *chunk_out, void *chunk_in,
                                  char *err, size_t errlen) {
    const uint32_t n_layer = pulsar_tp_n_layer(s->tp);
    const uint64_t vec = pulsar_tp_vec_bytes(s->tp);
    if (!chunk_out || !chunk_in)
        return sched_fail(s, err, errlen, "tp sched: null prefill buffers");
    const char *o = (const char *)chunk_out;
    char *i = (char *)chunk_in;
    for (uint32_t layer = 0; layer < n_layer; layer++) {
        const char *ol = o + (uint64_t)layer * vec;
        char *il = i + (uint64_t)layer * vec;
        if (!pulsar_tp_big_gate_exchange(s->tp, layer, seq, ol, il, vec))
            return sched_fail(s, err, errlen,
                              "tp sched: big_gate l=%u seq=%llu",
                              layer, (unsigned long long)seq);
    }
    return 1;
}

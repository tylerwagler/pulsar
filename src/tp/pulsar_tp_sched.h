#ifndef PULSAR_TP_SCHED_H
#define PULSAR_TP_SCHED_H

#include "tp/pulsar_tp.h"

/*
 * Gate scheduler (slice 4b) — the host-side half of the two-rank CUDA gate
 * machinery.  Drives the transport's per-gate exchange for a pair.  The two
 * CUDA-touching steps — write the local partial into the slab out-slot, and
 * consume the peer's landed partial from the in-slot — are funneled through
 * two hook pointers, so every bit of scheduling/lockstep logic here is host-
 * runnable and unit-tested (tests/tp_sched_test) before the real .cu
 * implementations land in slice 4c.  All hook calls happen on the calling
 * (engine worker) thread; nothing here touches CUDA.
 *
 * Exchange-counter convention: a decode token fires 86 exchanges
 * (GATES_PER_LAYER * n_layer).  The scheduler numbers them with a monotonic
 * per-exchange counter `e` (1-based) that both ranks compute identically from
 * their own run, and passes `e` to every hook and to pulsar_tp_gate_exchange.
 * Under the DS identity schedule (gates_per_token 0) the RDMA path derives the
 * slab slot as slot = (e-1) % n_slots, so e uniquely identifies (layer, gate)
 * and stays monotonic across tokens — do NOT pass a token index as `e`.
 */

typedef struct {
    const char *name;   /* hook-set name, diagnostics only */
    void *ud;           /* opaque context (engine tensors / CUDA state) */
    /* Publish this exchange's f32 partial (vec_bytes) into slab out-slot
     * [layer,gate].  4c's implementation reads the engine's hidden state,
     * and, when a GPU kernel produces it, signals the gate-ready flags. */
    int (*write_partial)(void *ud, uint32_t layer, uint32_t gate, uint64_t e);
    /* Consume the peer's landed partial from slab in-slot [layer,gate]
     * (e.g. accumulate into the engine's hidden state). */
    int (*read_partial)(void *ud, uint32_t layer, uint32_t gate, uint64_t e);
} pulsar_tp_sched_hooks;

typedef struct pulsar_tp_sched pulsar_tp_sched;

pulsar_tp_sched *pulsar_tp_sched_new(pulsar_tp *tp,
                                     const pulsar_tp_sched_hooks *hooks,
                                     char *err, size_t errlen);
void pulsar_tp_sched_free(pulsar_tp_sched *s);

/* Run one full decode step: for every layer, GATE_ATTN then GATE_FFN, one
 * exchange each, in slot order.  Both ranks call this for the same token,
 * strictly in lockstep.  Returns 1 on success, 0 on failure with err set. */
int pulsar_tp_sched_decode_token(pulsar_tp_sched *s, char *err, size_t errlen);

/* Prefill: one big_gate_exchange per layer for a whole chunk.  chunk_out /
 * chunk_in are the caller's per-layer buffers (n_layer * vec_bytes each);
 * this is the amortized gate path (86 exchanges per CHUNK, not per token).
 * Both ranks call it for the same chunk in lockstep. */
int pulsar_tp_sched_prefill_chunk(pulsar_tp_sched *s, uint64_t seq,
                                  const void *chunk_out, void *chunk_in,
                                  char *err, size_t errlen);

#endif

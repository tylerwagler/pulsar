/*
 * Pulsar TP transport core — slice 1.  See pulsar_tp.h for the contract.
 * Layout and identity rules mirror upstream antirez/ds4 ds4_tp.c so the
 * transport lift (slice 3) is mechanical.
 */

#include "tp/pulsar_tp.h"

#include <stdio.h>

void pulsar_tp_slab_layout_init(uint32_t n_layer, uint32_t n_embd, pulsar_tp_slab *s) {
    const uint64_t vec = (uint64_t)n_embd * sizeof(float);
    const uint64_t slots = (uint64_t)n_layer * PULSAR_TP_GATES_PER_LAYER;
    s->out_off = 0;
    s->in_off = slots * vec;
    s->in_flags_off = s->in_off + slots * vec;
    s->token_off = s->in_flags_off + slots * 8;
    s->out_flags_off = s->token_off + 16;
    s->gpu_flags_off = s->out_flags_off + slots * 8;
    s->batch_out_off = s->gpu_flags_off + slots * 4;
    s->batch_in_off = s->batch_out_off + (uint64_t)n_layer * PULSAR_TP_BATCH_MAX_ROWS * vec;
    s->slab_bytes = s->batch_in_off + (uint64_t)n_layer * PULSAR_TP_BATCH_MAX_ROWS * vec;
}

uint64_t pulsar_tp_slab_bytes(uint32_t n_layer, uint32_t n_embd) {
    const uint64_t vec = (uint64_t)n_embd * sizeof(float);
    const uint64_t slots = (uint64_t)n_layer * PULSAR_TP_GATES_PER_LAYER;
    return slots * vec * 2 +    /* out + in vectors */
           slots * 8 * 2 +      /* in seq flags + out flag staging */
           16 +                 /* token slot */
           slots * 4 +          /* GPU-written gate-ready flags */
           (uint64_t)n_layer * PULSAR_TP_BATCH_MAX_ROWS * vec * 2; /* batch out+in */
}

static uint64_t tp_slot(uint32_t layer, uint32_t gate) {
    return (uint64_t)layer * PULSAR_TP_GATES_PER_LAYER + gate;
}

uint64_t pulsar_tp_slab_out_offset(const pulsar_tp_slab *s, uint32_t layer, uint32_t gate, uint64_t vec_bytes) {
    return s->out_off + tp_slot(layer, gate) * vec_bytes;
}

uint64_t pulsar_tp_slab_in_offset(const pulsar_tp_slab *s, uint32_t layer, uint32_t gate, uint64_t vec_bytes) {
    return s->in_off + tp_slot(layer, gate) * vec_bytes;
}

uint64_t pulsar_tp_slab_batch_out_offset(const pulsar_tp_slab *s, uint32_t layer, uint64_t vec_bytes) {
    return s->batch_out_off + (uint64_t)layer * PULSAR_TP_BATCH_MAX_ROWS * vec_bytes;
}

uint64_t pulsar_tp_slab_batch_in_offset(const pulsar_tp_slab *s, uint32_t layer, uint64_t vec_bytes) {
    return s->batch_in_off + (uint64_t)layer * PULSAR_TP_BATCH_MAX_ROWS * vec_bytes;
}

int pulsar_tp_identity_check(const pulsar_tp_identity *mine,
                             const pulsar_tp_identity *theirs,
                             char *err, size_t errlen) {
    if (theirs->gguf_bytes != mine->gguf_bytes ||
        theirs->model_id != mine->model_id ||
        theirs->n_layer != mine->n_layer ||
        theirs->n_embd != mine->n_embd ||
        theirs->n_vocab != mine->n_vocab ||
        theirs->quant_bits != mine->quant_bits ||
        theirs->gate_slot_start != mine->gate_slot_start ||
        theirs->gate_slot_step != mine->gate_slot_step ||
        theirs->gates_per_token != mine->gates_per_token) {
        if (errlen)
            snprintf(err, errlen,
                     "tp: model mismatch (peer gguf=%llu id=%u layers=%u embd=%u "
                     "vocab=%u qbits=%u)",
                     (unsigned long long)theirs->gguf_bytes, theirs->model_id,
                     theirs->n_layer, theirs->n_embd, theirs->n_vocab,
                     theirs->quant_bits);
        return -1;
    }
    return 0;
}

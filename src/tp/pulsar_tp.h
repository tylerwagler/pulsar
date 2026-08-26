#ifndef PULSAR_TP_H
#define PULSAR_TP_H

/*
 * Two-rank tensor-parallel transport core (ledger L102, plan 102, and
 * docs/tensor-parallel-port.md).  Slice 1 of the port: the wire/identity
 * contract and the slab-layout arithmetic.  Pure host C++, no sockets, no
 * CUDA, unit-tested by tests/tp_core_test.
 *
 * The layout and hello mirror upstream antirez/ds4 (ds4_tp.c) field for
 * field so the later full transport lift stays mechanical; the two ranks run
 * the same logical model, each holding one contiguous half of the routed
 * experts, exchanging f32 partial layer outputs through a registered slab
 * (RDMA SEND/RECV when verbs are available, full-duplex TCP otherwise).
 *
 * Successive slices wire the options into the engine CLIs (slice 2), import
 * the socket/RDMA transport exercising this contract (slice 3), then add
 * the CUDA gate machinery on the engine worker thread.
 */

#include <stddef.h>
#include <stdint.h>

#define PULSAR_TP_MAGIC UINT32_C(0x44533454)     /* "DS4T", same wire magic as upstream */
#define PULSAR_TP_PROTOCOL_VERSION 7u

enum { PULSAR_TP_GATE_ATTN = 0, PULSAR_TP_GATE_FFN = 1, PULSAR_TP_GATES_PER_LAYER = 2 };
#define PULSAR_TP_BATCH_MAX_ROWS 8u

typedef enum {
    PULSAR_TP_ROLE_NONE = 0,
    PULSAR_TP_ROLE_LEADER = 1,
    PULSAR_TP_ROLE_WORKER = 2,
} pulsar_tp_role;

typedef struct {
    pulsar_tp_role role;
    const char *peer;   /* worker: leader's control/RDMA address; NULL on the leader (it listens) */
    int port;           /* control port */
} pulsar_tp_options;

/* Engine identity exchanged in the hello so a mismatched pair aborts before
 * any inference runs.  Field-for-field mirror of upstream ds4_tp_identity. */
typedef struct {
    uint64_t gguf_bytes;
    uint32_t model_id;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t quant_bits;
    uint32_t ctx_size;
    /* Decode gate schedule, used to place recvs into the right slab slot.
     * per_token 0 falls back to the identity mapping over all slots. */
    uint32_t gate_slot_start;
    uint32_t gate_slot_step;
    uint32_t gates_per_token;
} pulsar_tp_identity;

/* Fixed-size wire prefix of the hello; the RDMA key block follows it. */
typedef struct {
    uint32_t magic;      /* also detects byte-order mismatch */
    uint32_t version;
    uint32_t role;
    uint32_t rdma_ok;    /* this side has a usable verbs device */
    uint64_t gguf_bytes;
    uint32_t model_id;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t quant_bits;
    uint32_t ctx_size;
    uint32_t gate_slot_start;
    uint32_t gate_slot_step;
    uint32_t gates_per_token;
    uint32_t pad;
} pulsar_tp_hello_fixed;

/* Registered-slab layout.  S = n_layer * GATES_PER_LAYER slots, all offsets
 * are from the base VA; vec_bytes = n_embd * 4 (f32 partials, never
 * quantized on the wire).  Region order matches upstream ds4_tp_slab_layout:
 *
 *   out vectors    S*vec   written by local GPU kernels
 *   in  vectors    S*vec   RDMA/TCP-written peer partials
 *   in  seq flags  S*8     written after each in vector
 *   token slot     16      {seq u64, token i32, pad} leader->worker
 *   out flag sats  S*8
 *   gpu flags      S*4     GPU-written gate-ready flags
 *   batch out      n_layer*MAX_ROWS*vec   (verify-block partial rows)
 *   batch in       n_layer*MAX_ROWS*vec
 */
typedef struct {
    uint64_t out_off;
    uint64_t in_off;
    uint64_t in_flags_off;
    uint64_t token_off;
    uint64_t out_flags_off;
    uint64_t gpu_flags_off;
    uint64_t batch_out_off;
    uint64_t batch_in_off;
    uint64_t slab_bytes;
} pulsar_tp_slab;

void pulsar_tp_slab_layout_init(uint32_t n_layer, uint32_t n_embd, pulsar_tp_slab *out);
uint64_t pulsar_tp_slab_bytes(uint32_t n_layer, uint32_t n_embd);
uint64_t pulsar_tp_slab_out_offset(const pulsar_tp_slab *s, uint32_t layer, uint32_t gate, uint64_t vec_bytes);
uint64_t pulsar_tp_slab_in_offset(const pulsar_tp_slab *s, uint32_t layer, uint32_t gate, uint64_t vec_bytes);
uint64_t pulsar_tp_slab_batch_out_offset(const pulsar_tp_slab *s, uint32_t layer, uint64_t vec_bytes);
uint64_t pulsar_tp_slab_batch_in_offset(const pulsar_tp_slab *s, uint32_t layer, uint64_t vec_bytes);

/* Identity acceptance: 0 when compat, -1 with a message in err.  The shape
 * and gate schedule must match; ctx_size is exchanged but NOT compared
 * (mirrors upstream: each side reads the peer's ctx, it is not equality). */
int pulsar_tp_identity_check(const pulsar_tp_identity *mine,
                             const pulsar_tp_identity *theirs,
                             char *err, size_t errlen);

#endif

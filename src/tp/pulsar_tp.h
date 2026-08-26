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

#include <stdbool.h>
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

/* ------------------------------------------------------------------------
 * Transport (slice 3).  Public surface of the ported upstream ds4_tp.c: the
 * connection/lockstep transport that runs the slice-1 slab contract.  The
 * wire format is byte-identical to upstream antirez/ds4 (magic 0x44533454,
 * protocol version 7, the hello frame above, and the ds4_tp.h frame type
 * numbers).  Functions return 1 on success / 0 on failure and snprintf into
 * (err, errlen) when err is non-NULL, matching the tree convention.
 * ---------------------------------------------------------------------- */

typedef struct pulsar_tp pulsar_tp;   /* opaque; defined in pulsar_tp.cpp */

/* Deferred to the engine-wiring slice (not ported here); the upstream
 * versions couple to ds4_* CLI/engine/distributed types this fork dropped:
 *   pulsar_tp_parse_cli_arg / pulsar_tp_usage / pulsar_tp_enabled /
 *   pulsar_tp_validate_engine_options / pulsar_tp_adopt_distributed_options
 * The slice-2 engine options (--tp-role/--tp-peer/--tp-port) already keep the
 * fail-loud open guard until one of those arms in.  pulsar_tp_worker_run
 * (upstream ds4_tp_worker_run) comes with the engine-worker wiring slice. */

/* Connection bring-up.  The leader listens on (peer ? peer : 0.0.0.0):port
 * and accepts one worker; the worker dials peer:port with retry.  Both then
 * exchange and validate identities (multi-field equality plus roles must
 * differ).  Blocking; call after the engine is loaded (identity needs its
 * shape).  Returns 1 on success, 0 on failure with err set. */
int pulsar_tp_create(pulsar_tp **out, const pulsar_tp_options *opt,
                     const pulsar_tp_identity *id, char *err, size_t errlen);
void pulsar_tp_free(pulsar_tp *tp);
int pulsar_tp_rank(const pulsar_tp *tp);            /* 0 leader, 1 worker */
bool pulsar_tp_is_rdma(const pulsar_tp *tp);
uint32_t pulsar_tp_peer_ctx(const pulsar_tp *tp);
bool pulsar_tp_failed(const pulsar_tp *tp);
void pulsar_tp_mark_failed(pulsar_tp *tp);

/* GPU-written gate-ready flags region of the slab (u32 per slot). */
uint64_t pulsar_tp_slab_gpu_flags_offset(const pulsar_tp_slab *s);

/* Register the slab base with the transport.  The engine allocates one
 * contiguous (GPU-visible on GB10) block and hands its base VA here; the
 * transport registers it with the NIC (RDMA) and exchanges remote keys, or
 * keeps it as plain TCP staging.  Returns 1 on success, 0 on failure. */
int pulsar_tp_attach_slab(pulsar_tp *tp, void *base, char *err, size_t errlen);

/* One gate: send out[layer][gate] to the peer's in[layer][gate] and wait for
 * the peer's partial for `seq` to land.  Returns 0 on failure. */
int pulsar_tp_gate_exchange(pulsar_tp *tp, uint32_t layer, uint32_t gate,
                            uint64_t seq);

/* Verify-block batch gate: exchange `rows` (<= PULSAR_TP_BATCH_MAX_ROWS) row
 * partials for one layer in one bulk transfer.  Returns 0 on failure. */
int pulsar_tp_batch_gate_exchange(pulsar_tp *tp, uint32_t layer, uint32_t rows,
                                  uint64_t seq);

/* Prefill batch gate: arbitrary-size symmetric payload exchange between the
 * caller's out/in buffers (slab staging free to use).  Returns 0 on failure. */
int pulsar_tp_big_gate_exchange(pulsar_tp *tp, uint32_t layer, uint64_t seq,
                                const void *out, void *in, uint64_t bytes);

/* Lockstep mirroring (leader side) and worker loop primitives. */
typedef struct {
    uint64_t session_id;
    int32_t token;
    uint32_t reserved;
} pulsar_tp_batch_item;

int pulsar_tp_send_session_create(pulsar_tp *tp, uint64_t session_id,
                                  int ctx_size);
int pulsar_tp_send_session_destroy(pulsar_tp *tp, uint64_t session_id);
int pulsar_tp_send_sync(pulsar_tp *tp, uint64_t session_id,
                        const int *tokens, uint32_t n_tokens);
int pulsar_tp_send_eval(pulsar_tp *tp, uint64_t session_id,
                        uint64_t seq, int token);
int pulsar_tp_send_rewind(pulsar_tp *tp, uint64_t session_id, int pos);
int pulsar_tp_send_invalidate(pulsar_tp *tp, uint64_t session_id);
int pulsar_tp_send_eval_batch(pulsar_tp *tp, const pulsar_tp_batch_item *items,
                              uint32_t count);
int pulsar_tp_send_mixed_batch(pulsar_tp *tp, uint64_t prefill_session_id,
                               const int *prompt, uint32_t prompt_count,
                               const pulsar_tp_batch_item *items,
                               uint32_t count);
int pulsar_tp_send_command_ack(pulsar_tp *tp, uint64_t session_id, int status);
int pulsar_tp_wait_command_ack(pulsar_tp *tp, uint64_t session_id,
                               const char *operation,
                               char *err, size_t errlen);
int pulsar_tp_send_stop(pulsar_tp *tp);

/* Worker: blocks for the next mirrored command.  For FRAME_SYNC the token
 * array is returned in *tokens / *n_tokens (malloc'd, caller frees), for
 * FRAME_EVAL seq/token are filled.  Frame type numbers are upstream's wire
 * numbers — do not renumber. */
typedef enum {
    PULSAR_TP_FRAME_ERROR = -1,
    PULSAR_TP_FRAME_SYNC = 1,
    PULSAR_TP_FRAME_EVAL = 2,
    PULSAR_TP_FRAME_REWIND = 3,
    PULSAR_TP_FRAME_INVALIDATE = 4,
    PULSAR_TP_FRAME_STOP = 5,
    PULSAR_TP_FRAME_HASH = 6,
    PULSAR_TP_FRAME_RDMA_INFO = 7,
    PULSAR_TP_FRAME_SYNC_ACK = 8,
    PULSAR_TP_FRAME_RDMA_READY = 9,
    PULSAR_TP_FRAME_LOGITS = 10,
    PULSAR_TP_FRAME_VERIFY = 11,
    PULSAR_TP_FRAME_VERIFY_COMMIT = 12,
    PULSAR_TP_FRAME_SESSION_CREATE = 13,
    PULSAR_TP_FRAME_SESSION_DESTROY = 14,
    PULSAR_TP_FRAME_EVAL_BATCH = 15,
    PULSAR_TP_FRAME_MIXED_BATCH = 16,
    PULSAR_TP_FRAME_COMMAND_ACK = 17,
} pulsar_tp_frame_type;

typedef struct {
    pulsar_tp_frame_type type;
    uint64_t session_id;
    uint64_t seq;
    int value;
    int *tokens;          /* malloc'd for FRAME_SYNC/VERIFY/MIXED_BATCH */
    uint32_t n_tokens;
    pulsar_tp_batch_item *items;  /* malloc'd for EVAL_BATCH/MIXED_BATCH */
    uint32_t n_items;
} pulsar_tp_command;

int pulsar_tp_recv_command(pulsar_tp *tp, pulsar_tp_command *command,
                           char *err, size_t errlen);
void pulsar_tp_command_free(pulsar_tp_command *command);

/* Debug lockstep check: both sides send their hidden-state hash for a token
 * and compare.  Returns 0 on transport failure, -1 on hash mismatch. */
int pulsar_tp_hash_check(pulsar_tp *tp, uint64_t seq, uint64_t hash,
                         char *err, size_t errlen);

/* Vocab-split output head: the worker ships its logits half to the leader
 * after every eval (and after a sync) on the control socket. */
int pulsar_tp_send_logits_half(pulsar_tp *tp, const float *half,
                               uint32_t count);
int pulsar_tp_recv_logits_half(pulsar_tp *tp, float *half, uint32_t count);

/* Speculative verify mirroring.  The leader announces a draft block right
 * before both ranks run the expert-split batch verify; the worker then blocks
 * on the commit frame, which carries the leader's decision. */
int pulsar_tp_send_verify(pulsar_tp *tp, uint64_t session_id,
                          const int *drafts, uint32_t n);
int pulsar_tp_send_verify_commit(pulsar_tp *tp, int32_t full_accept,
                                 int32_t replay_n);
int pulsar_tp_recv_verify_commit(pulsar_tp *tp, int32_t *full_accept,
                                 int32_t *replay_n);

/* Decode gate schedule -> slab slot.  per_token 0 falls back to the identity
 * mapping slot = (seq-1) % n_slots; otherwise slot = gate_slot_start +
 * ((seq-1) % gates_per_token) * gate_slot_step.  Both ranks compute this
 * from the exchanged identity so their recv placement matches exactly. */
uint32_t pulsar_tp_gate_slot(uint32_t n_slots, uint64_t seq,
                             uint32_t gate_slot_start, uint32_t gate_slot_step,
                             uint32_t gates_per_token);

#endif

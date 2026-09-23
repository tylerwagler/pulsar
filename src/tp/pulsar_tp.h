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
#define PULSAR_TP_PROTOCOL_VERSION 10u           /* v10: batch header carries max_head_runs; v9: row payload + RNG_STATE; v8: rank + n_ranks in the hello */

enum { PULSAR_TP_GATE_ATTN = 0, PULSAR_TP_GATE_FFN = 1, PULSAR_TP_GATES_PER_LAYER = 2 };
/** Layer tag for exchanges that are NOT per-layer (slice 4d's vocab gather).
 *  Deliberately outside any real layer index so a lane that ran a vocab gather
 *  where its peer ran a per-layer MoE gate is caught as a desync by name rather
 *  than passing because the (layer, seq) pair happened to coincide. */
#define PULSAR_TP_NON_LAYER_TAG UINT32_C(0xFFF0)
#define PULSAR_TP_BATCH_MAX_ROWS 8u

typedef enum {
    PULSAR_TP_ROLE_NONE = 0,
    PULSAR_TP_ROLE_LEADER = 1,
    PULSAR_TP_ROLE_WORKER = 2,
} pulsar_tp_role;

typedef struct {
    pulsar_tp_role role;    /* legacy 2-rank: LEADER/WORKER; n-way: any (ranks are symmetric) */
    int rank;               /* this rank's index in the group; -1 = unset */
    int n_ranks;            /* group size; 0/1 = unset */
    const char *peer;       /* legacy 2-rank: worker's dial target (NULL on the leader) */
    const char *peers;      /* n-way: ordered "host:port,..." list for ALL n ranks */
    int port;               /* this rank's own control/listen port */
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
    uint32_t rank;       /* this rank's index in the TP group (n-way) */
    uint32_t n_ranks;    /* group size */
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

/* Fill the two-rank identity for this engine's artifact and set the DS
 * gate-schedule defaults (43-layer DS fires ATTN then FFN on every layer, so
 * gates_per_token stays 0 -> the transport's identity slot mapping).  The
 * engine wiring (slice 4a+) supplies the exact values from g_pulsar_shape at
 * bind time plus the GGUF file size; this is the pure filler + default keeper,
 * unit-tested by tp_core_test. */
void pulsar_tp_identity_init_defaults(pulsar_tp_identity *id,
                                      uint64_t gguf_bytes,
                                      uint32_t model_id,
                                      uint32_t n_layer,
                                      uint32_t n_embd,
                                      uint32_t n_vocab,
                                      uint32_t quant_bits,
                                      uint32_t ctx_size);

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
uint32_t pulsar_tp_n_ranks(const pulsar_tp *tp);    /* ranks in this TP group */

/* Owned slice of a dimension for `rank` in a group of `n_ranks`, floor-
 * partitioned over [0,n_total): lo = rank*n/n_ranks, hi = (rank+1)*n/n_ranks
 * (uint64 mid).  Deterministic; disjoint and complete across ranks (rank r's hi
 * == rank r+1's lo).  n_ranks<=1 -> [0,n_total) (full path); n_total==0 ->
 * [0,0).  Returns 1 on success, 0 on bad args.
 *
 * This is the single authority (rule 4) for "which slice does a rank own", and
 * it is deliberately generic: the routed-EXPERT split (256 experts over the
 * group) and the 4d vocab split (n_vocab over the group) are the SAME rule, so
 * they share one implementation and cannot drift.  Callers never recompute it. */
int pulsar_tp_owned_range(int rank, uint32_t n_ranks, uint32_t n_total,
                          uint32_t *lo, uint32_t *hi);

/* Slice 4f (L237): the BYTE span of a rank's owned slice of an expert stack
 * whose experts are stored back-to-back at `expert_bytes` each:
 * off = lo*expert_bytes, bytes = (hi-lo)*expert_bytes, from the range authority
 * above.  The loader stages ONLY these bytes, the admission budget charges ONLY
 * these bytes and the MoE dispatch resolves ONLY these bytes, so the three
 * cannot drift.  n_ranks<=1 -> the whole stack.  Returns 1 on success, 0 on
 * bad args. */
int pulsar_tp_owned_byte_span(int rank, uint32_t n_ranks, uint32_t n_total,
                              uint64_t expert_bytes, uint64_t *off, uint64_t *bytes);

/* n-way full-mesh bring-up: connects every rank (n_ranks) to every other with
 * rank-ordered dial/accept.  opt->peers is the ordered "host:port,..." list for
 * all n ranks; opt->rank/opt->n_ranks are explicit.  Returns 1 on success. */
int pulsar_tp_create_mesh(pulsar_tp **out, const pulsar_tp_options *opt,
                          const pulsar_tp_identity *id, char *err, size_t errlen);

/* All-gather + local sum across the whole group: `out` starts as this rank's
 * owned partial and returns the combined sum of every rank's partial.  `in` is
 * caller scratch (clobbered).  n=2 is byte-identical to the old pairwise
 * exchange + add.  Returns 0 on failure. */
int pulsar_tp_allreduce_sum(pulsar_tp *tp, uint32_t layer, uint64_t seq,
                            void *out, const void *in, uint64_t bytes);

/* n-way VOCAB ALL-GATHER (slice 4d).  Rank r contributes the vocab range
 * [r*V/n, (r+1)*V/n) of `n_rows` rows; every rank ends with the full
 * [n_rows, n_total] block in `full_out`.  This CONCATENATES in rank order -- it
 * is NOT the sum pulsar_tp_allreduce_sum performs, and the two must never be
 * confused (a sum here would multiply the logits by the group size).
 *
 * Shapes: `full_out` is [n_rows, n_total] with row pitch n_total.  `own_slice`
 * and `scratch` are [n_rows, stride] PACKED, stride = ceil(n_total/n_ranks):
 * the ranges are PADDED to a uniform stride because the group exchange carries
 * ONE byte count per round and n_total need not divide by n_ranks (129280 does
 * not divide by 3 or 5).  Everything past a rank's real range is never sent on
 * the wire as data -- it pads the transfer only -- and the caller must leave
 * those tail elements ZEROED, since a short last range is never written by the
 * head.  `scratch` is clobbered.  Returns 0 on failure.
 *
 * The gather runs the same ascending-rank loop on every rank, one exchange per
 * peer per round, so `seq` must be identical on all ranks for a given round and
 * distinct between rounds (the transport's desync guard keys on it). */
int pulsar_tp_allgather_vocab(pulsar_tp *tp, uint32_t layer, uint64_t seq,
                              float *full_out, const float *own_slice,
                              float *scratch, uint32_t n_rows, uint32_t n_total);

bool pulsar_tp_is_rdma(const pulsar_tp *tp);
uint32_t pulsar_tp_peer_ctx(const pulsar_tp *tp);
bool pulsar_tp_failed(const pulsar_tp *tp);
void pulsar_tp_mark_failed(pulsar_tp *tp);
uint32_t pulsar_tp_n_layer(const pulsar_tp *tp);     /* decoded from the hello */
uint64_t pulsar_tp_vec_bytes(const pulsar_tp *tp);   /* n_embd * 4 (f32 partials) */

/* The registered slab's per-layer BATCH regions (n_layer blocks, each
 * PULSAR_TP_BATCH_MAX_ROWS vectors).  A caller whose payload fits that many
 * rows should stage HERE rather than in its own buffer: the RDMA big gate rides
 * DIRECT when its out/in pointers already lie inside the slab, instead of
 * copying the payload through these very regions to reach registered memory.
 * NULL when no slab is attached or `layer` is out of range. */
void *pulsar_tp_slab_batch_out(const pulsar_tp *tp, uint32_t layer);
void *pulsar_tp_slab_batch_in(const pulsar_tp *tp, uint32_t layer);

/* Is [ptr, ptr+bytes) wholly inside the registered slab?  This is the RDMA big
 * gate's DIRECT test: when both payloads answer yes they are already registered
 * and ride the QP with no copy through the staging regions; when either says no
 * the payload is staged through those regions first.  ONE authority for that
 * rule.  Exposed so the decision can be asserted on a single box (a real
 * transport plus a real slab is enough) even though engaging it needs a pair. */
bool pulsar_tp_in_slab(const pulsar_tp *tp, const void *ptr, uint64_t bytes);

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

/* Lockstep mirroring (leader side) and worker loop primitives.
 *
 * RETURN CONVENTION FOR THIS WHOLE FILE: nonzero (1) means SUCCESS and zero
 * means failure -- the inverse of the usual C shape, and it holds for every
 * pulsar_tp_send_*, for pulsar_tp_recv_command, and for
 * pulsar_tp_wait_command_ack.  The engine's first mirroring pass read it
 * backwards at seven call sites, which made every successful send look like a
 * refusal, so it is stated once here rather than inferred per call site.  A
 * refusal that carries a reason puts it in `err`; the allgather/allreduce and
 * gate exchanges above follow the same rule. */
/** One row of a mirrored batched decode: the engine's own row contract
 * (pulsar_multiseq_req) plus the session the row belongs to.  The layout was
 * fixed HERE, while FRAME_EVAL_BATCH/FRAME_MIXED_BATCH still had no user -- it
 * used to be {session_id, token, reserved}, which cannot carry a row's bank and
 * position, so no caller could have been written against it.  Frame NUMBERS are
 * still never reused; only this payload changed, once, before first use. */
typedef struct {
    uint64_t session_id;  ///< the mirrored session this row belongs to
    int32_t  bank;        ///< true bank id in that session's pool
    int32_t  pos;         ///< absolute position of `token`
    int32_t  token;       ///< input token id decoded at `pos`
    uint32_t reserved;    ///< pad to 24 bytes: keeps every field naturally aligned
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
/** The SAME row payload as pulsar_tp_send_eval_batch, on a distinct frame type.
 * The two frames exist so the worker can prove which engine operation the leader
 * is in: `decode_mixed` and `decode_multiseq` are byte-identical for a
 * decode-only batch, so a driver that diverged between them would otherwise
 * decode the same rows through a different contract in silence.  The payload
 * was redefined to rows-only along with the batch item: it used to carry a
 * separate prefill prompt (`prefill_session_id` + token array), which came from
 * upstream's mixed step -- OUR decode_mixed takes its prompt as rows in the same
 * `pulsar_multiseq_req` list (a K-row run for one bank), so there is nothing
 * else to send.  The header's shape was fixed while it still had no user. */
int pulsar_tp_send_mixed_batch(pulsar_tp *tp,
                               const pulsar_tp_batch_item *items,
                               uint32_t count, uint32_t max_head_runs);
/** Ship this rank's rng state.  Speculation's accept walk draws from the
 * CALLER's rng, so two ranks seeded independently would accept different tokens
 * and commit different session state; the leader's state is the pair's stream.
 * Fire-and-forget, like the void operations: a `void`-shaped draw site has
 * nowhere to put a peer's refusal, and the worker's frame check reports a
 * divergence through the next acked operation instead of hanging on it. */
int pulsar_tp_send_rng_state(pulsar_tp *tp, uint64_t session_id, uint64_t state);
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
    /* RETIRED 2026-09-22: the leader-collect vocab half that slice 4d's
     * all-gather replaced.  The NUMBER is never reused -- these are wire
     * values, and a future frame must take a fresh one. */
    PULSAR_TP_FRAME_LOGITS = 10,
    PULSAR_TP_FRAME_VERIFY = 11,
    PULSAR_TP_FRAME_VERIFY_COMMIT = 12,
    PULSAR_TP_FRAME_SESSION_CREATE = 13,
    PULSAR_TP_FRAME_SESSION_DESTROY = 14,
    PULSAR_TP_FRAME_EVAL_BATCH = 15,
    PULSAR_TP_FRAME_MIXED_BATCH = 16,
    PULSAR_TP_FRAME_COMMAND_ACK = 17,
    /* RDMA gate-window armed ack: the first gate exchange uses this to make
     * both ranks arm their receive window before either SENDS (a rank that
     * sends before the peer's window is armed silently loses the first N
     * messages under UC, shifting the whole pairing by +1). */
    PULSAR_TP_FRAME_RDMA_GATE_ARMED = 18,
    /* Slice 4e: one rank's rng state, so a pair can share ONE speculation
     * stream.  Fresh number (18 is taken, 10 is retired and never reused). */
    PULSAR_TP_FRAME_RNG_STATE = 19,
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

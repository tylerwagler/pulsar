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
#include "pulsar.h"   /* pulsar_image_ref (SYNC_MM) */

#define PULSAR_TP_MAGIC UINT32_C(0x44533454)     /* "DS4T", same wire magic as upstream */
#define PULSAR_TP_PROTOCOL_VERSION 15u           /* v15: CHUNK_VERDICT -- a mirrored prefill yields at a chunk boundary on both ranks; v14: the bulk lane -- rdma info carries a bulk buffer + second QP; v13: a NODE frame after bring-up carries each rank's host, build and RDMA device; v12: SESSION_CREATE carries the bank-pool size; v11: the command ack carries a logits digest (L243); v10: batch header carries max_head_runs; v9: row payload + RNG_STATE; v8: rank + n_ranks in the hello */

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
    const char *build;      /* this rank's build id (git describe), carried to every
                             * peer in the NODE frame; NULL/"" = not stamped */
} pulsar_tp_options;

/* What one rank is, as the group knows it: exchanged once per peer in the NODE
 * frame at the end of bring-up (protocol v13), after each rank has opened its
 * RDMA device, so the device is known.  Every string is NUL-terminated and may
 * be empty.  This exists for operators (/health's tp block), never for the data
 * plane: nothing here is compared or acted on. */
#define PULSAR_TP_NODE_STR 64
typedef struct {
    int rank;
    char host[PULSAR_TP_NODE_STR];        /* gethostname() on that rank */
    char build[PULSAR_TP_NODE_STR];       /* pulsar_tp_options.build on that rank */
    char addr[PULSAR_TP_NODE_STR];        /* its TP endpoint ("host:port"), if known */
    char rdma_device[PULSAR_TP_NODE_STR]; /* the HCA it opened; "" under TCP */
    int rdma_port;                        /* the HCA port; 0 under TCP */
} pulsar_tp_node;

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
 *   in  seq flags  S*8     [0] = row lane done word (proxy-written exchange id)
 *   token slot     16      {seq u64, token i32, pad} leader->worker
 *   out flag sats  S*8     [0..2] = row lane descriptor (GPU-written, id last)
 *   gpu flags      S*4     [0] = row lane device spin-timeout latch
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
/* The NODE frame's record for `rank` (this rank's own included).  Returns 1 and
 * fills *out when the group knows that rank, 0 otherwise.  The records are
 * written once during bring-up and never again, so any thread may read them
 * for the life of the transport. */
int pulsar_tp_node_info(const pulsar_tp *tp, int rank, pulsar_tp_node *out);

/* Owned slice of a dimension for `rank` in a group of `n_ranks`, floor-
 * partitioned over [0,n_total): lo = rank*n/n_ranks, hi = (rank+1)*n/n_ranks
 * (uint64 mid).  Deterministic; disjoint and complete across ranks (rank r's hi
 * == rank r+1's lo).  n_ranks<=1 -> [0,n_total) (full path); n_total==0 ->
 * [0,0).  Returns 1 on success, 0 on bad args.
 *
 * This is the single authority (rule 4) for "which slice does a rank own", and
 * it is deliberately generic: the attention output groups, the shared- and
 * routed-expert intermediate halves and the 4d vocab split are the SAME rule,
 * so they share one implementation and cannot drift.  Callers never recompute
 * it. */
int pulsar_tp_owned_range(int rank, uint32_t n_ranks, uint32_t n_total,
                          uint32_t *lo, uint32_t *hi);


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

/* n-way ROW ALL-GATHER (slice 4d for the vocab, 4g for the attention `low`).
 * A row of `n_units * unit` floats is partitioned over the group in UNITS by
 * the range authority above: rank r owns units [lo, hi) = owned_range(r, n,
 * n_units), i.e. elements [lo*unit, hi*unit).  Every rank ends with the full
 * [n_rows, n_units*unit] block in `full_out`.  This CONCATENATES in rank order
 * -- it is NOT the sum pulsar_tp_allreduce_sum performs, and the two must never
 * be confused (a sum here would multiply the logits by the group size).
 *
 * `unit` is the indivisible element run: 1 for the vocab (129280 tokens over
 * n), the LoRA rank for the attention output groups (8 groups x 1024, so the
 * split follows the groups the kernels compute, never a bare element count).
 *
 * Shapes: `full_out` is [n_rows, n_units*unit] with that row pitch.
 * `own_slice` and `scratch` are [n_rows, stride] PACKED, stride =
 * ceil(n_units/n_ranks) * unit: the ranges are PADDED to a uniform stride
 * because the group exchange carries ONE byte count per round and n_units need
 * not divide by n_ranks.  Everything past a rank's real range is never sent on
 * the wire as data -- it pads the transfer only -- and the caller must leave
 * those tail elements ZEROED, since a short last range is never written by the
 * producer.  `scratch` is clobbered.  Returns 0 on failure.
 *
 * The gather runs the same ascending-rank loop on every rank, one exchange per
 * peer per round, so `seq` must be identical on all ranks for a given round and
 * distinct between rounds (the transport's desync guard keys on it). */
int pulsar_tp_allgather_rows(pulsar_tp *tp, uint32_t layer, uint64_t seq,
                             float *full_out, const float *own_slice,
                             float *scratch, uint32_t n_rows,
                             uint32_t n_units, uint32_t unit);

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

/* Pin the CALLING thread to the big core reserved for the engine's launch
 * thread (chosen with the row-lane proxy's core at attach; announced there).
 * The driver calls it once from the thread that will launch the engine's work
 * -- AFTER creating its other threads, which would otherwise inherit the
 * one-core mask.  A hint: no row lane or no core found = no-op; a refused
 * pin is said once and the thread runs unpinned.  tp may be NULL. */
void pulsar_tp_pin_launch_thread(pulsar_tp *tp);

/* One gate: send out[layer][gate] to the peer's in[layer][gate] and wait for
 * the peer's partial for `seq` to land.  Returns 0 on failure. */
int pulsar_tp_gate_exchange(pulsar_tp *tp, uint32_t layer, uint32_t gate,
                            uint64_t seq);

/* The ROW LANE (L241 4g-2): the pair's decode and verify exchanges, fully
 * asynchronous.  An exchange swaps `rows` (1..PULSAR_TP_BATCH_MAX_ROWS)
 * vectors of vec_bytes with the peer and never makes the engine thread wait:
 *
 *  - the engine thread calls pulsar_tp_row_lane_begin, which numbers the
 *    exchange (monotonic id) and its messages (one per row, monotonic seq --
 *    message t lives in slab slot (t-1) % n_slots, out-slots to send, in-slots
 *    to receive), and arms the pre-posted RDMA receive window when a big gate
 *    drained it (the ARMED handshake stays on the engine thread: the proxy
 *    never touches the control socket);
 *  - the engine then enqueues the GPU half (pulsar_gpu_tp_stage_publish,
 *    pulsar_gpu_tp_combine_*): stage own rows into the out-slots and, from the
 *    last block to finish, publish the descriptor; combine once the done word
 *    reaches the exchange id;
 *  - the proxy thread (verbs only, NO CUDA -- port rule 1) polls the
 *    descriptor, posts the sends, waits for the peer's rows AND its own send
 *    completions, re-posts the window, then writes the done word.
 *
 * Ring invariants (checked by pulsar_tp_row_lane): WINDOW + MAX_ROWS <=
 * n_slots, and at most one exchange in flight -- the GPU combines exchange e
 * before it stages e+1 -- so no slot is rewritten or re-posted while live.
 * Both ranks number exchanges identically by lockstep.  A proxy timeout
 * latches failed(); a device spin timeout sets the slab's error word; the
 * engine refuses at its next host sync through pulsar_tp_row_lane_check. */
typedef struct {
    uint64_t out_off, in_off;   /* gate-slot rings (message slots) */
    uint64_t desc_off;          /* u64[3] {exchange id (written last), first msg, rows} */
    uint64_t done_off;          /* u64: last exchange the proxy completed */
    uint64_t err_off;           /* u32: device spin timeout latch */
    uint64_t vec_bytes;
    uint32_t n_slots;
    uint64_t timeout_ns;        /* device spin bound = the transport timeout */
} pulsar_tp_row_lane_layout_t;

bool pulsar_tp_row_lane(const pulsar_tp *tp);
void pulsar_tp_row_lane_layout(const pulsar_tp *tp, pulsar_tp_row_lane_layout_t *out);
int pulsar_tp_row_lane_begin(pulsar_tp *tp, uint32_t rows, uint64_t *first_msg, uint64_t *exch);
/* At a host sync point (the stream is drained): 1 when every enqueued
 * exchange completed cleanly, 0 (refused by name) otherwise. */
int pulsar_tp_row_lane_check(pulsar_tp *tp);

/* THE BULK LANE (v14): prefill-sized exchanges on the row lane's machinery.
 * The engine attaches a caller-owned, host-pinned, GPU-mapped buffer BEFORE
 * pulsar_tp_attach_slab (pulsar_tp_set_bulk); attach registers it and brings up
 * a second UC QP whose receive queue holds only zero-length imm receives, so
 * bulk traffic never mixes with the gates' 16 KB receives.  Laid out
 * out | in[0] | in[1], each cap_bytes (a multiple of vec_bytes).  Per exchange:
 * the GPU stages its rows into `out`, publishes {exch, bytes, BULK|buf} on the
 * row lane's descriptor (the same exchange numbering), the proxy RDMA-writes
 * `out` into the PEER's in[buf] (the last write WITH_IMM = the exchange id),
 * and the combine adds in[buf] once `done` reaches the exchange.  Receive
 * buffers alternate per bulk exchange: the peer can only be one exchange ahead,
 * so the buffer it writes next is never the one being read.  A payload larger
 * than cap_bytes is the caller's to cut. */
typedef struct { uint64_t cap_bytes, out_off, in_off[2]; } pulsar_tp_bulk_layout_t;
void pulsar_tp_set_bulk(pulsar_tp *tp, void *base, uint64_t bytes);
bool pulsar_tp_bulk_lane(const pulsar_tp *tp);
void pulsar_tp_bulk_layout(const pulsar_tp *tp, pulsar_tp_bulk_layout_t *out);
/* Number one bulk exchange of `bytes` (<= cap): *exch on the row lane's
 * sequence, *buf the receive buffer it lands in.  0 = refused (printed). */
int pulsar_tp_bulk_begin(pulsar_tp *tp, uint64_t bytes, uint64_t *exch, uint32_t *buf);
/* The row-lane descriptor's bulk flag (word 2; low bit = buffer). */
#define PULSAR_TP_DESC_BULK_FLAG (UINT64_C(1) << 63)

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

/* SESSION_CREATE carries the leader's Tier-2 bank-pool size (v12): the pool is
 * sized by the SERVER at startup (auto-fit or PULSAR_MSEQ_BANKS), a step a
 * worker process never runs, so a worker created its sessions with no bank
 * pool and refused the first batched speculative redraft.  The worker adopts
 * `n_banks` (command->seq, >= 1) before creating.  */
int pulsar_tp_send_session_create(pulsar_tp *tp, uint64_t session_id,
                                  int ctx_size, uint32_t n_banks);
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
/* The bank frames (increment 2).  Leader -> workers. */
int pulsar_tp_send_bank_state_save(pulsar_tp *tp, uint64_t session_id, uint32_t bank);
int pulsar_tp_send_bank_state_restore(pulsar_tp *tp, uint64_t session_id, uint32_t bank);
int pulsar_tp_send_bank_repoint(pulsar_tp *tp, uint64_t session_id, uint32_t bank);
/* `partial` selects BANK_FORK_PARTIAL over BANK_FORK; the payload is the same. */
int pulsar_tp_send_bank_fork(pulsar_tp *tp, int partial, uint64_t session_id,
                             uint32_t src, uint32_t dst,
                             const int *tokens, uint32_t n_tokens, int n_cached);
/* Increment 3.  `common` rides the token header beside the tokens. */
int pulsar_tp_send_rewrite_from_common(pulsar_tp *tp, uint64_t session_id,
                                       const int *tokens, uint32_t n_tokens, int common);
int pulsar_tp_send_note_committed(pulsar_tp *tp, uint64_t session_id,
                                  const int *tokens, uint32_t n_tokens);
int pulsar_tp_send_set_logits(pulsar_tp *tp, uint64_t session_id,
                              const float *logits, uint32_t n);
/* The one payload every speculative frame shares.  Which of i0..i3 a frame
 * uses is documented on its sender; unused fields are 0. */
typedef struct {
    uint64_t session_id;
    int32_t  bank;        /* the round's bank (the live bank when the leader sent it) */
    int32_t  i0, i1, i2, i3;
    float    temperature, top_p, min_p;
    int32_t  top_k;
    uint64_t rng;         /* the rng state the call consumes, as it was BEFORE the call */
    uint32_t count;       /* REDRAFT_BATCH: rounds; banks[count] then rngs[count] follow */
    uint32_t reserved;
} pulsar_tp_spec_command;

/* Increment 4: one sender for every speculative frame; `banks`/`rngs` are
 * REDRAFT_BATCH's arrays (count entries each) and NULL elsewhere. */
int pulsar_tp_send_spec(pulsar_tp *tp, uint32_t frame_type, const pulsar_tp_spec_command *cmd,
                        const uint32_t *banks, const uint64_t *rngs);
/* Increment 6.  free/alloc physical ride the value payload; save/load carry
 * the snapshot key. */
int pulsar_tp_send_bank_free_physical(pulsar_tp *tp, uint64_t session_id, uint32_t bank);
int pulsar_tp_send_bank_alloc_physical(pulsar_tp *tp, uint64_t session_id, uint32_t bank);
int pulsar_tp_send_bank_kv(pulsar_tp *tp, int load, uint64_t session_id, uint32_t bank, const char *key);
/* Increment 7: sync with images (n_images > 0; use pulsar_tp_send_sync otherwise). */
int pulsar_tp_send_sync_mm(pulsar_tp *tp, uint64_t session_id, const int *tokens, uint32_t n_tokens,
                           const pulsar_image_ref *images, uint32_t n_images);
int pulsar_tp_send_command_ack(pulsar_tp *tp, uint64_t session_id, int status);

/* ---- The cross-rank logits identity check (L243, protocol v11) --------------
 * Every mirrored operation that PRODUCES logits (eval, batch decode, mixed
 * batch) ends with each rank holding the full vocab vector (slice 4d's
 * all-gather + assembly).  Those must be the same bytes on every rank -- a wrong
 * range partition, a wrong gather or a wrong assembly makes them differ.  The
 * worker folds its assembled logits into a 64-bit digest and rides it on the
 * ack; the leader digests its own and REFUSES the operation (and marks the
 * group failed -- the ranks no longer agree on the model's output) when any
 * peer's differs.  The instrument runs the production lane on every frame; the
 * leader prints the tally at close (`cross-rank logits identity: N/N worker
 * frames matched`) and the pair grading tool reads that line as LEG A.
 *
 * The digest is word-wise (uint64 lanes, four independent chains so the
 * multiply latency does not serialise it), seeded with the row count so a
 * missing or extra row differs even when the bytes present agree.  It is a
 * structural identity check, not a cryptographic one. */
#define PULSAR_TP_ACK_HAS_DIGEST 1u
uint64_t pulsar_tp_logits_digest(const float *logits, uint32_t n_rows, uint32_t width);
/* The ack a logits-producing operation answers with on SUCCESS: `status` is 0
 * for the acked operations (eval, batch decode, mixed batch) and the POSITIVE
 * verdict for a verdict operation that ran (generate_speculative: tokens + 1).
 * A failed one answers pulsar_tp_send_command_ack (the status is the verdict;
 * there are no logits to digest). */
int pulsar_tp_send_command_ack_digest(pulsar_tp *tp, uint64_t session_id, int status,
                                      uint64_t digest);
/* The leader's collect for a logits-producing operation: like
 * pulsar_tp_wait_command_ack, plus every peer's ack must CARRY a digest equal
 * to `own_digest`.  A peer ack without a digest, or with a different one, is a
 * refusal that also marks the group failed.  Counted in
 * pulsar_tp_identity_stats. */
int pulsar_tp_wait_command_ack_digest(pulsar_tp *tp, uint64_t session_id,
                                      const char *operation, uint64_t own_digest,
                                      char *err, size_t errlen);
/* The verdict collect for a verdict operation that also produces logits
 * (generate_speculative: the CLI's whole loop as one frame, every rank ending
 * on the same last row).  Like pulsar_tp_wait_command_status, plus every
 * POSITIVE verdict (the operation ran) must carry a digest equal to
 * `own_digest`; a zero verdict (the run failed on that rank) is plain, and is
 * read as a split against the leader's positive one by the caller. */
int pulsar_tp_wait_command_status_digest(pulsar_tp *tp, uint64_t session_id,
                                         const char *operation, int *status,
                                         uint64_t own_digest, char *err, size_t errlen);
/* PIPELINED identity check (L241 4g-2): instead of waiting for the peers'
 * digest acks of a step before the next one starts, the leader records its own
 * digest here and returns; the acks are read -- and the digests compared --
 * the next time ANY ack is collected (they arrive in frame order, so they are
 * the next ones in the socket) or at pulsar_tp_settle_deferred_ack.  A
 * mismatch then fails that later call, names `operation` (the step whose
 * logits differed), and marks the group failed, exactly as the direct check
 * does.  One deferral at a time: deferring with one pending refuses (0). */
int pulsar_tp_defer_command_ack_digest(pulsar_tp *tp, uint64_t session_id,
                                       const char *operation, uint64_t own_digest);
/* Read and check the deferred acks now (1 = none pending, or they matched). */
int pulsar_tp_settle_deferred_ack(pulsar_tp *tp, char *err, size_t errlen);
/* This rank will not run (or could not finish) a step its peer is running:
 * mark the group failed, latch the row lane's error word (this rank's spinning
 * kernels exit), and send the peer the abort on the data socket so its proxy
 * latches the same -- a refusal fails the pair in milliseconds instead of
 * leaving the peer's GPU spinning until the transport timeout.  Idempotent. */
void pulsar_tp_row_lane_abort(pulsar_tp *tp, const char *why);
/* The leader's own body of a mirrored step failed: give the peers 5 s to
 * answer (they failed it too), else abort the row lane (they are spinning on
 * exchanges this rank will not join).  Call before collecting their acks;
 * pulsar_tp_drain_command_acks does. */
void pulsar_tp_own_step_failed(pulsar_tp *tp, const char *operation);
/* The leader's drain for a logits-producing operation whose OWN body failed:
 * the peers' acks are read (an unread ack would shift every later frame) in
 * either shape and their verdicts ignored -- the local failure is the result. */
void pulsar_tp_drain_command_acks(pulsar_tp *tp);
/* frames = peer acks checked for identity so far, matched = how many agreed. */
void pulsar_tp_identity_stats(const pulsar_tp *tp, uint64_t *frames, uint64_t *matched);
/* Collect one ack per peer and return the VERDICT they agree on in *status
 * (1 on success).  Unlike pulsar_tp_wait_command_ack, a nonzero status is not
 * a failure here -- a fork refusal code is a legitimate result -- but the
 * peers must all report the SAME status, and a NEGATIVE status is a worker
 * refusal (unknown session, failed pair), never a verdict.  Returns 0 on a
 * dead link, a disagreement between peers, or a refusal (err says which). */
int pulsar_tp_wait_command_status(pulsar_tp *tp, uint64_t session_id,
                                  const char *operation, int *status,
                                  char *err, size_t errlen);
int pulsar_tp_wait_command_ack(pulsar_tp *tp, uint64_t session_id,
                               const char *operation,
                               char *err, size_t errlen);
int pulsar_tp_send_stop(pulsar_tp *tp);
/* A MIRRORED prefill's chunk-boundary decision (v15).  Every rank reaches the
 * same boundaries (identical chunk plan); the leader evaluates its own cancel
 * hook there -- a client disconnect, or the scheduler's one-chunk yield -- and
 * sends the verdict; each worker reads it at the same boundary and stops, or
 * not, with it.  Both ranks therefore stop at the same row, and the next
 * mirrored sync resumes both from there.  The worker's read is bounded by the
 * transport timeout (the leader is at the same boundary); a missing or foreign
 * frame fails the group. */
int pulsar_tp_send_chunk_verdict(pulsar_tp *tp, uint64_t session_id, int stop);
int pulsar_tp_recv_chunk_verdict(pulsar_tp *tp, uint64_t session_id, int *stop,
                                 char *err, size_t errlen);

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
    /* RETIRED 2026-09-23 (increment 4): the rng-state frame.  A single
     * per-session rng could not follow the server, which interleaves banks
     * between the base draw and the accept walk; every speculative frame now
     * carries the rng it consumes.  The number is never reused. */
    PULSAR_TP_FRAME_RNG_STATE = 19,
    /* Slice 4e increment 2 (L238): the bank surface.  save is void
     * (fire-and-forget); restore, repoint and the two forks return a VERDICT
     * the ranks must AGREE on (a fork legitimately refuses with a code that
     * routes the caller to a cold prefill), collected with
     * pulsar_tp_wait_command_status. */
    PULSAR_TP_FRAME_BANK_STATE_SAVE = 20,
    PULSAR_TP_FRAME_BANK_STATE_RESTORE = 21,
    PULSAR_TP_FRAME_BANK_REPOINT = 22,
    PULSAR_TP_FRAME_BANK_FORK = 23,
    PULSAR_TP_FRAME_BANK_FORK_PARTIAL = 24,
    /* Increment 3: the rest of the server's mutating surface.  REWRITE is a
     * verdict frame whose result enum includes -1 (ERROR), so the wire status
     * is result + 1 (a negative wire status stays a worker refusal).
     * NOTE_COMMITTED is void.  SET_LOGITS carries the vector itself. */
    PULSAR_TP_FRAME_REWRITE_FROM_COMMON = 25,
    PULSAR_TP_FRAME_NOTE_COMMITTED = 26,
    PULSAR_TP_FRAME_SET_LOGITS = 27,
    /* Increment 4: the speculative round family.  Every frame that draws
     * carries the rng it consumes (a single per-session state cannot follow
     * the server, which interleaves banks between the base draw and the
     * accept walk).  Verdicts that can be -1 ride the wire as value + 1. */
    PULSAR_TP_FRAME_SPEC_NEXT_BASE = 28,        /* verdict: first token + 1 */
    PULSAR_TP_FRAME_SPEC_ROUND_BEGIN = 29,      /* verdict: 0 ok, 1 failed */
    PULSAR_TP_FRAME_SPEC_ARM_CAPTURE = 30,      /* void */
    PULSAR_TP_FRAME_SPEC_ROUND_END = 31,        /* verdict: accepted count + 1 */
    PULSAR_TP_FRAME_SPEC_ROUND_ABORT = 32,      /* void */
    PULSAR_TP_FRAME_SPEC_REDRAFT_BATCH = 33,    /* verdict: 0 ok, 1 failed */
    PULSAR_TP_FRAME_SPEC_REDRAFT_COMMIT = 34,   /* void */
    PULSAR_TP_FRAME_GENERATE_SPECULATIVE = 35,  /* verdict: tokens generated + 1 */
    /* Increment 6: the eviction guard's spill path.  KV is replicated per
     * rank, so each rank spills its own bank to its own disk; the frames name
     * the snapshot by the KEY of the file the leader wrote (its basename with
     * any ".tmp.<pid>" stripped) and the worker mirrors it under its own spill
     * directory.  All four are verdicts (0 ok, 1 failed). */
    PULSAR_TP_FRAME_BANK_FREE_PHYSICAL = 36,
    PULSAR_TP_FRAME_BANK_ALLOC_PHYSICAL = 37,
    PULSAR_TP_FRAME_BANK_KV_SAVE = 38,
    PULSAR_TP_FRAME_BANK_KV_LOAD = 39,
    /* Increment 7: a sync that carries IMAGES -- the tokens (with the vision
     * sentinel blocks already expanded by the leader), then a per-image table
     * of {start_pos, len}, then the concatenated image bytes.  Every rank's
     * own replicated tower encodes the same bytes; nothing image-shaped is
     * gathered.  Acked like SYNC. */
    PULSAR_TP_FRAME_SYNC_MM = 40,
    /* v15: the leader's per-chunk "stop here?" inside a mirrored prefill
     * (pulsar_tp_send_chunk_verdict).  Read by the worker's prefill loop at the
     * same chunk boundary, never by the command loop. */
    PULSAR_TP_FRAME_CHUNK_VERDICT = 41,
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
    /* BANK_FORK / BANK_FORK_PARTIAL: the banks and the shared-prefix length;
     * the request tokens ride `tokens`/`n_tokens`.  The bank of a save /
     * restore / repoint rides `value`. */
    int32_t bank_src;
    int32_t bank_dst;
    int32_t n_cached;
    /* SET_LOGITS: the leader's live logits row (malloc'd, n_logits floats). */
    float *logits;
    uint32_t n_logits;
    /* The speculative frames: the shared payload, plus REDRAFT_BATCH's arrays
     * (malloc'd, spec.count each). */
    pulsar_tp_spec_command spec;
    uint32_t *spec_banks;
    uint64_t *spec_rngs;
    /* BANK_KV_SAVE / BANK_KV_LOAD: the snapshot key (malloc'd, NUL-terminated);
     * the bank rides `value`. */
    char *spill_key;
    /* SYNC_MM: the images (malloc'd table whose `bytes` point into
     * `image_bytes`, one malloc'd block). */
    pulsar_image_ref *images;
    uint32_t n_images;
    uint8_t *image_bytes;
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

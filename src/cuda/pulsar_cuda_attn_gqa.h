/* GQA attention block (L295 F1): the family-neutral grouped-query attention
 * the shared block library offers a family's layer plan.  Everything BETWEEN
 * the attention's projections: the caller runs the q / k / v (and, for a
 * relative bias, r) Linears before it and the o_proj after it.
 *
 * What a family declares (pulsar_gqa_layer), all at RUNTIME:
 *   geometry      n_q query heads, n_kv KV heads (n_q % n_kv == 0, the group
 *                 G = n_q / n_kv <= 64), head_dim (a built tile: 64 or 128)
 *   KV rows       the row format (pulsar_gqa_kv_fmt; BF16 today)
 *   span          FULL (every earlier key) or WINDOW (the last `window` keys,
 *                 self included, held in a ring of `cap` >= window slots)
 *   scale         the softmax scale on q.k (1/head_dim for unit-RMS q and k)
 *   qk norm       optional per-head RMSNorm of q and k (weight used as is)
 *   score mod     optional additive relative bias: per (row, query head, d)
 *                 bias = bf16(r[row, h, :] . P[:, d]) for 0 <= d < extent,
 *                 0 beyond, where d = query position - key position, r is a
 *                 rel_dim-wide per-head projection the caller supplies and P
 *                 the layer's [rel_dim][extent] table;
 *                 optional log-length temperature tau(pos) = 1 + alpha *
 *                 ln(max(1, (pos + 1) / floor)), applied (as a bf16 product)
 *                 to the normed q and to the bias -- i.e. to the whole logit
 *   output gate   optional y *= sigmoid(gate) per (row, head, dim)
 *   rope          optional rotary position encoding of the normed q and k
 *                 at the row's position: the NeoX half-split rotation of
 *                 the whole head, base `rope_theta` (or, L321, a per-pair
 *                 inverse-frequency table: YaRN), in the HF / vLLM bf16
 *                 arithmetic (cos and sin rounded to bf16, each product and
 *                 the sum one bf16 rounding)
 *   block         optional (L298, a block drafter's query block): every row
 *                 of a sequence's run also attends the LATER rows of its run
 *                 (bidirectional inside the run, the span's window to the
 *                 left), and the run's rows are drafts -- they are never
 *                 stored into the pool
 *
 * The arithmetic, per (row, query head):
 *   q = bf16(rope(bf16(rmsnorm(q_in) * w_q)) * tau) (norm / rope / tau when declared)
 *   k = rope(bf16(rmsnorm(k_in) * w_k)),  v = v_in  (the KV row stores these)
 *   s(x) = (q . k_x) * scale + bias(pos - x)        for the keys x it attends
 *   y = sum_x softmax(s)_x v_x  [* sigmoid(gate)]   -> bf16 (o_proj's input)
 *
 * ONE ARITHMETIC PER ROW.  A row's keys are cut at ABSOLUTE positions into
 * splits of 64 and chunks of 256 (4 splits).  A chunk is reduced in a fixed
 * order (online softmax over its splits, P V on the tensor cores with P as a
 * bf16 hi + lo pair); the row's result folds its chunks in chunk order with
 * one recurrence and ends in one epilogue.  Nothing a row computes depends on
 * which other rows share its call, its CTA or its MMA tile, so decode (one row
 * per sequence), verify (a run of rows), chunked prefill and any mix of them
 * give the SAME bytes -- asserted by tests/gqa_attn_gate.  Two schedules of
 * that arithmetic, chosen by the call's row count only: below 64 rows every
 * (chunk, KV head, tile) is a CTA and a combine kernel folds the chunks; at 64
 * or more a CTA walks its tile's chunks and folds in registers.
 *
 * KV rows (one per token per layer, the caller owns the pools; F3 will):
 *   BF16  K head 0 .. K head n_kv-1, then V head 0 .. V head n_kv-1, each
 *         head_dim bf16 -- n_kv * 2 * head_dim * 2 B per token.
 *   E4M3  (L298) the same order, each element ONE E4M3 byte at scale 1.0 (the
 *         stored bf16 value converted with satfinite rounding; vLLM's
 *         kv_cache_dtype fp8_e4m3 with unit k / v scales) -- n_kv * 2 *
 *         head_dim B per token.  The tile loader widens it to bf16 (exact);
 *         a call's own rows are read in the same format, so every key and
 *         value the attention sees is the stored one.
 *   (a block-scaled E4M3 row, one E8M0 per 32 elements, would be one more
 *   value, a row-bytes line and a tile loader.)
 * A FULL sequence's token at position p is slot p (cap = token capacity); a
 * WINDOW sequence's is slot p % cap.  A call's own rows read the keys it
 * produced from its workspace and write the cache AFTER the attention (only
 * the last cap positions of a ring), so a ring needs no slack for a prefill
 * chunk.  Rolling back a verify of k rows on a ring is free when
 * cap >= window - 1 + k (no position a later row still sees was overwritten).
 */
#ifndef PULSAR_CUDA_ATTN_GQA_H
#define PULSAR_CUDA_ATTN_GQA_H

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
typedef struct CUstream_st *cudaStream_t;
#endif

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** The KV row format (the format table's attention-row entry). */
typedef enum {
    PULSAR_GQA_KV_BF16 = 0,   /**< K then V heads, head_dim bf16 each */
    PULSAR_GQA_KV_E4M3 = 1,   /**< K then V heads, head_dim E4M3 bytes each, unit scale */
} pulsar_gqa_kv_fmt;

/** Which keys a row attends. */
typedef enum {
    PULSAR_GQA_FULL   = 0,    /**< every key at a position <= the row's */
    PULSAR_GQA_WINDOW = 1,    /**< the last `window` keys, self included, in a ring */
} pulsar_gqa_span;

/** One attention layer as its family declares it.  Weights are device
 *  pointers, bf16 as stored. */
typedef struct {
    uint32_t          n_q;          /**< query heads */
    uint32_t          n_kv;         /**< KV heads; n_q % n_kv == 0, n_q / n_kv <= 64 */
    uint32_t          head_dim;     /**< 64 or 128 */
    pulsar_gqa_kv_fmt kv_fmt;
    pulsar_gqa_span   span;
    uint32_t          window;       /**< WINDOW: keys per row (self included); FULL: 0 */
    float             scale;        /**< softmax scale on q . k */
    int               qk_norm;      /**< 1: per-head RMSNorm of q and k */
    float             norm_eps;
    const uint16_t   *q_norm_w;     /**< bf16 [head_dim] (qk_norm) */
    const uint16_t   *k_norm_w;     /**< bf16 [head_dim] (qk_norm) */
    uint32_t          rel_extent;   /**< relative bias distances 0 .. extent-1; 0 = no bias */
    uint32_t          rel_dim;      /**< the per-head r width: 16 (one MMA k-step; the only built width) */
    const uint16_t   *rel_proj;     /**< bf16 [rel_dim][rel_extent] */
    float             temp_floor;   /**< log-length temperature floor; 0 = no temperature */
    float             temp_alpha;
    int               out_gate;     /**< 1: y *= sigmoid(gate) */
    float             rope_theta;   /**< > 0: NeoX rope of q and k at base rope_theta; 0 = none */
    int               block;        /**< 1: a run's rows attend its later rows too and are never stored (no bias) */
    const float      *rope_inv_freq;/**< L321: device f32 [head_dim / 2]: pair p's inverse frequency in place of
                                         1 / rope_theta^(2p / head_dim) -- a scaled rope (YaRN) the family builds on
                                         the host (src/lib/pulsar_yarn.h); needs rope_theta > 0; NULL = plain NeoX */
} pulsar_gqa_layer;

/** One sequence's KV pool for ONE layer. */
typedef struct {
    void    *kv;       /**< device, cap * pulsar_gqa_kv_token_bytes(layer) */
    uint64_t bytes;
    uint32_t cap;      /**< FULL: token capacity; WINDOW: ring slots (>= window) */
} pulsar_gqa_seq;

/** A strided bf16 input: element (row, head, i) at ptr[row * ld + head * head_stride + i]. */
typedef struct {
    const void *ptr;
    uint64_t    ld;
    uint32_t    head_stride;
} pulsar_gqa_in;

typedef struct {
    pulsar_gqa_in q;          /**< [rows] x n_q heads x head_dim */
    pulsar_gqa_in k;          /**< [rows] x n_kv heads x head_dim (after any conv the family runs) */
    pulsar_gqa_in v;          /**< [rows] x n_kv heads x head_dim */
    pulsar_gqa_in r;          /**< [rows] x n_q heads x rel_dim (rel_extent > 0) */
    pulsar_gqa_in gate;       /**< [rows] x n_q heads x head_dim (out_gate) */
    void         *out_bf16;   /**< [rows][ld_out] bf16: head h at columns h * head_dim .. */
    uint64_t      ld_out;
    float        *tap_out_f32;/**< gate tap, f32 [rows][n_q * head_dim], or NULL in the lane */
    void         *workspace;  /**< device, >= pulsar_gqa_workspace_bytes */
    uint64_t      workspace_bytes;
} pulsar_gqa_io;

/** Bytes of one token's KV row for this layer (every KV head, K and V). */
uint64_t pulsar_gqa_kv_token_bytes(const pulsar_gqa_layer *layer);

/** Workspace for ANY call of at most n_rows rows whose positions are < max_ctx. */
uint64_t pulsar_gqa_workspace_bytes(const pulsar_gqa_layer *layer, uint32_t n_rows, uint32_t max_ctx);

/** Run the layer's attention for n_rows rows on `stream`.  Row r is the token
 *  at position row_pos[r] of sequence row_seq[r] (host arrays).  Each
 *  sequence's rows are ONE contiguous run of consecutive positions starting
 *  at its frontier (every earlier position was written by an earlier call).
 *  Writes the rows' KV into the sequences' pools.
 *  @return 0 on success, -1 on a refused contract (named on stderr, nothing
 *  launched), -3 on a launch failure. */
int pulsar_gqa_forward(const pulsar_gqa_layer *layer,
                       const pulsar_gqa_seq *seqs, uint32_t n_seqs,
                       const uint32_t *row_seq, const uint32_t *row_pos, uint32_t n_rows,
                       const pulsar_gqa_io *io, cudaStream_t stream);

/* ---- the same call in two halves (L296 perf): the host plan, then the launches -------------------------------
 * pulsar_gqa_forward is pulsar_gqa_plan into a host stage + one upload + pulsar_gqa_launch.  A caller that runs its
 * step's launches as a captured CUDA graph plans every attention call of the step first (its host half), uploads
 * the descriptors itself, and launches from them: a launch reads nothing from the host but the plan's shape. */

/** A planned call: what its launches depend on. */
typedef struct {
    uint32_t n_rows;
    uint32_t n_tiles;
    uint32_t grid_chunks;   /**< the split schedule's chunk CTAs per (KV head, tile) */
    uint32_t ctx_bound;     /**< the workspace's context: the call's bound, else its own last position + 1 */
    int      fold;          /**< 1: the folded schedule (n_rows >= 64) */
} pulsar_gqa_call;

/** Bytes of a call's descriptors at n_rows rows (pulsar_gqa_plan's `desc`). */
uint64_t pulsar_gqa_desc_bytes(uint32_t n_rows);

/** The host half: the rows' descriptors into `desc` (host, >= pulsar_gqa_desc_bytes(n_rows)) and the call's shape
 *  into *out.  The contract and the refusals are pulsar_gqa_forward's.  ctx_bound > 0 BOUNDS the call: every row's
 *  position is below it (else refused) and the launches are sized by it, not by the positions -- the bytes are the
 *  unbounded call's (a chunk CTA past its tile's last row exits; the combine folds each row's own chunks), so a
 *  graph captured at one bound replays at every position under it.  0 = the call's own positions (the forward's).
 *  @return 0, or -1 on a refused contract (named on stderr). */
int pulsar_gqa_plan(const pulsar_gqa_layer *layer, const pulsar_gqa_seq *seqs, uint32_t n_seqs,
                    const uint32_t *row_seq, const uint32_t *row_pos, uint32_t n_rows, uint32_t ctx_bound,
                    void *desc, uint64_t desc_cap, pulsar_gqa_call *out);

/** The launches of a planned call from its descriptors on the device (`desc_dev`, 256-byte aligned: the plan's
 *  bytes, uploaded by the caller before this runs).  io->workspace is sized by pulsar_gqa_workspace_bytes at the
 *  call's rows and ctx_bound; its descriptor region is unused (the descriptors are desc_dev's).
 *  @return 0, -1 on a refused contract, -3 on a launch failure. */
int pulsar_gqa_launch(const pulsar_gqa_layer *layer, const pulsar_gqa_call *call, const void *desc_dev,
                      const pulsar_gqa_io *io, cudaStream_t stream);

/* ---- the KV rows alone (L298) ----------------------------------------------------------------------------------
 * A block drafter's context rows have no queries: their K and V come from the target's hidden states and land in
 * the pool, which the drafter's query blocks read.  The store is the attention call's own first and last stage --
 * the K head norm and rope of gqa_prep, the pool write of gqa_commit -- so a row stored here is byte for byte the
 * row a forward over the same k / v would have stored.  The contract on rows, sequences and the ring is the
 * forward's (each sequence's rows one run of consecutive positions; a ring keeps the run's last cap). */

/** Workspace for a store of at most n_rows rows. */
uint64_t pulsar_gqa_store_workspace_bytes(const pulsar_gqa_layer *layer, uint32_t n_rows);

/** Store n_rows rows' K (normed, roped as declared) and V into the sequences' pools on `stream`.  io->k and io->v
 *  are read; q, r, gate and out are not.  `block` must be 0 (a block's rows are never stored).
 *  @return 0, -1 on a refused contract (named on stderr, nothing launched), -3 on a launch failure. */
int pulsar_gqa_store(const pulsar_gqa_layer *layer, const pulsar_gqa_seq *seqs, uint32_t n_seqs,
                     const uint32_t *row_seq, const uint32_t *row_pos, uint32_t n_rows, const pulsar_gqa_io *io,
                     cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* PULSAR_CUDA_ATTN_GQA_H */

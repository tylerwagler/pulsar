/* Short convolution block (L295 E3): a family-neutral depthwise causal
 * convolution over a row stream, with a recurrent state of the last
 * kernel - 1 input rows per sequence.  The shared block library offers it to a
 * family's layer plan; nothing here knows which family declares it.
 *
 * What a family declares (pulsar_sconv_decl), per stream:
 *   channels   C, the stream's width (a multiple of 8)
 *   kernel     K taps, 2 .. PULSAR_SCONV_KMAX; tap K-1 multiplies the current row
 *   weight     bf16 [C][K] as a checkpoint stores a depthwise Conv1d ([C][1][K])
 *   residual   1: the block's output adds its own input (y = x + conv(x))
 *   bias       NULL, or bf16 [C]: the tap sum starts from it (Conv1d's bias)
 *   act        PULSAR_SCONV_ACT_NONE, or PULSAR_SCONV_ACT_SILU: the sum goes
 *              through SiLU before its one rounding (exclusive with residual:
 *              no family declares both, so the order is not built)
 *
 * The arithmetic, per (row, channel) -- ONE arithmetic, whoever calls it
 * (pulsar_cuda_sconv.cuh is its single authority, for this kernel and for any
 * epilogue that fuses it later):
 *   acc = bias ? bias[c] : 0;  for i = 0 .. K-1:  acc = fmaf(w[c][i], x[t - (K-1) + i][c], acc)
 *                                                       (f32, oldest tap first)
 *   y[t][c] = bf16_rn(residual ? acc + x[t][c] : act == SILU ? acc / (1 + exp(-acc)) : acc)
 * where x at a position before the sequence's first token is 0 (a fresh
 * sequence's state is zero).  Inputs, weights, state and output are bf16;
 * every tap widens exactly, so the only roundings are the f32 fma chain, the
 * SiLU's f32 ops (the engine's fast-math exp and divide) and the one bf16
 * store.  Nothing a row computes depends on the rows beside it, its call, its
 * CTA or the call's row count: decode (1 row a sequence), verify (a
 * run of rows), a prefill chunk cut anywhere and any batch of them give the
 * SAME bytes -- asserted by tests/sconv_gate.
 *
 * STATE (the caller owns the pools; F3's "fixed recurrent with per-row
 * capture" kind).  One slot per (sequence, stream):
 *   bf16 [K-1][C], SHIFT order: row i holds the stream's INPUT x at position
 *   p - (K-1) + i, where p is the sequence's next position (row K-2 is the
 *   newest).  Channels contiguous; the slot's rows are state_ld apart.
 *   Slot s of a pool starts at  state + s * state_stride  (elements; both
 *   strides multiples of 8, state_ld >= C, state_stride >= (K-2) * state_ld
 *   + C).  So several streams can share ONE record a sequence and layer:
 *   Inkling's four streams as [K-1][10240] with the streams at channel
 *   offsets 0 / 1024 / 2048 / 6144 (each stream's `state` pointer offset by
 *   its channels, state_ld 10240) -- one 60 KiB record to price, page,
 *   snapshot and restore.  A FRESH sequence starts from a zeroed slot.
 * A call reads each sequence's slot and leaves it holding the state after the
 * sequence's last row (the slot is advanced in place).  With capture set, the
 * call also writes, for every call row r, the state AFTER row r -- the same
 * layout, rows capture_ld apart -- at capture + r * capture_stride (a packed
 * record per row works the same way): a verify that accepts
 * its rows up to row r rolls back by copying that record into the slot
 * (pulsar_sconv_restore).  Restoring to BEFORE a call's first row is the
 * caller's own snapshot (a speculative verify always keeps its first row).
 * A prefix snapshot at a chunk boundary is a copy of the slot.
 *
 * Inkling (the first user) declares four streams a layer, K = 4, residual = 1:
 * K and V after their projections (1024 channels each, read strided from the
 * fused qkvr row) before the k-norm; the attention-output delta and the
 * MLP-output delta (4096 each) before the residual add.  3 x 10240 x 2 B =
 * 60 KiB of state a layer and sequence.
 *
 * A Mamba-2 mixer (pulsar_cuda_mamba2.h, the second user) declares one stream
 * over its x | B | C columns with a bias and SiLU, residual 0. */
#ifndef PULSAR_CUDA_SCONV_H
#define PULSAR_CUDA_SCONV_H

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

/** The widest kernel built (taps). */
#define PULSAR_SCONV_KMAX 8
/** The most streams one call carries. */
#define PULSAR_SCONV_STREAMS_MAX 4

/** The activation a stream's tap sum goes through before its rounding. */
enum {
    PULSAR_SCONV_ACT_NONE = 0,
    PULSAR_SCONV_ACT_SILU = 1,  /**< acc / (1 + exp(-acc)), f32 */
};

/** One stream's convolution as its family declares it. */
typedef struct {
    uint32_t        channels;   /**< C, a multiple of 8 */
    uint32_t        kernel;     /**< K taps, 2 .. PULSAR_SCONV_KMAX */
    int             residual;   /**< 1: y = x + conv(x); 0: y = conv(x) */
    const uint16_t *weight;     /**< device bf16 [C][K], 16-byte aligned; tap K-1 multiplies the current row */
    const uint16_t *bias;       /**< NULL, or device bf16 [C], 16-byte aligned: the tap sum's start */
    uint32_t        act;        /**< PULSAR_SCONV_ACT_*; SILU needs residual 0 */
} pulsar_sconv_decl;

/** One stream of a call: where its rows, output, state pool and capture live.
 *  Every pointer is a device pointer, 16-byte aligned; every stride is in
 *  bf16 elements and a multiple of 8. */
typedef struct {
    const pulsar_sconv_decl *decl;
    const uint16_t *x;          /**< input rows: row r's channels at x + r * ld_x (may be a strided column range) */
    uint64_t        ld_x;       /**< >= C */
    uint16_t       *y;          /**< output rows at y + r * ld_y; must not overlap the input rows */
    uint64_t        ld_y;       /**< >= C */
    uint16_t       *state;      /**< the pool: slot s at state + s * state_stride, its row i at + i * state_ld */
    uint64_t        state_ld;       /**< row stride inside a slot, >= C */
    uint64_t        state_stride;   /**< slot stride, >= (K-2) * state_ld + C */
    uint16_t       *capture;    /**< NULL, or the state after call row r at capture + r * capture_stride, rows capture_ld apart */
    uint64_t        capture_ld;     /**< >= C when capture is set */
    uint64_t        capture_stride; /**< >= (K-2) * capture_ld + C when capture is set */
} pulsar_sconv_stream;

/** The call's rows: n_seqs sequences, each ONE contiguous run of its next
 *  positions.  Sequence s is the call's rows [seq_first[s], seq_first[s+1])
 *  on pool slot seq_slot[s]; distinct sequences of one call on distinct slots.
 *  Decode is n_seqs runs of 1 row, a prefill chunk 1 run, a verify runs of
 *  2..k rows; any mix in one call. */
typedef struct {
    uint32_t       n_seqs;
    uint32_t       n_rows;        /**< seq_first[n_seqs] */
    uint32_t       max_seq_rows;  /**< >= every sequence's row count (sizes the grid) */
    const int32_t *seq_first;     /**< device [n_seqs + 1], ascending, seq_first[0] = 0, every run >= 1 row */
    const int32_t *seq_slot;      /**< device [n_seqs] */
} pulsar_sconv_rows;

/** Elements of one stream's state, (K-1) * C: a slot / capture record at row stride C. */
uint64_t pulsar_sconv_state_elems(const pulsar_sconv_decl *decl);

/** Run n_streams (1 .. PULSAR_SCONV_STREAMS_MAX) convolutions over the same
 *  rows in one launch on `stream` (Inkling's K and V ride together).  Advances
 *  every sequence's slot of every stream; writes the captures when set.
 *  @return 0 on success, -1 on a refused contract (named on stderr, nothing
 *  launched), -3 on a launch failure. */
int pulsar_sconv_forward(const pulsar_sconv_stream *streams, uint32_t n_streams, const pulsar_sconv_rows *rows,
                         cudaStream_t stream);

/** Roll a slot back to the state after call row `row` of a captured call: the
 *  stream's K-1 rows of C, from capture + row * capture_stride (rows capture_ld
 *  apart) to state + slot * state_stride (rows state_ld apart).  A caller whose
 *  streams share one packed record may copy the record whole instead.
 *  @return 0, or -1 refused / -3 on a copy failure. */
int pulsar_sconv_restore(const pulsar_sconv_stream *s, uint32_t slot, uint32_t row, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* PULSAR_CUDA_SCONV_H */

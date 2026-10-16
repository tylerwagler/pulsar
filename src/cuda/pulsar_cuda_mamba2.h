/* Mamba-2 (SSD) mixer block (L296 N2): a family-neutral selective state-space
 * mixer -- everything BETWEEN its two projections.  The in_proj and out_proj
 * GEMMs are the caller's (the format front doors), so the block is the same
 * whether a model's Linears are EXL3, NVFP4 or bf16: it reads the in_proj's
 * bf16 output row and writes the out_proj's bf16 input row.  Nothing here
 * knows which family declares it; geometry is runtime.
 *
 * What a family declares (pulsar_mamba2_geom + pulsar_mamba2_layer):
 *   heads H, head_dim P, d_state N, groups G (B and C shared by H / G heads),
 *   conv kernel K over the CC = H*P + 2*G*N channels x | B | C (with its
 *   bias, SiLU), the gated norm's group (channels per RMS group) and eps, and
 *   per head A_log, dt_bias, D; the norm weight [H*P].  All bf16, as the
 *   checkpoints store them (the conv is E3's block: NVIDIA keeps it bf16 in
 *   its NVFP4 build too).
 *
 * THE MATH, per row t (one arithmetic, whoever calls; the reference is vLLM's
 * mamba_mixer2, the card's serving path -- rounding points mirror it):
 *   xbc        = bf16( SiLU( conv_bias + sum_i w_i * xBC[t - (K-1) + i] ) )    (E3, pulsar_cuda_sconv.h)
 *   x, B, C    = split(xbc, H*P, G*N, G*N);  head h reads group g = h / (H / G)
 *   dt[h]      = softplus(f32(dt_raw[h]) + f32(dt_bias[h]))                 double, rounded to f32
 *   dA[h]      = exp(dt[h] * A[h]),  A[h] = -exp(A_log[h])                   double, rounded to f32
 *   S[h][p][n] = fmaf(dt[h] * B[g][n], x[h][p], dA[h] * S[h][p][n])           f32, each op rounded
 *   y[h][p]    = bf16( fmaf(D[h], x[h][p], sum_n S[h][p][n] * C[g][n]) )      the sum in a fixed tree
 *   v          = y * SiLU(z)                                                  f32
 *   out        = bf16( w * bf16( v * rsqrt(mean_group(v^2) + eps) ) )        per norm group
 * The decay and the softplus are evaluated in DOUBLE: dA multiplies the state
 * every row, so its error compounds over a head's memory (as Qwen's GDN,
 * pulsar_cuda_gdn.h).  The recurrence is a per-row scan in a fixed order: a
 * sequence's bytes never depend on how its rows were cut into calls, which
 * other sequences ride with it, or the call's row count -- decode (1 row a
 * sequence), a prefill chunk cut anywhere, one-call prefill and verify runs
 * give the SAME bytes (tests/mamba2_gate asserts it).  There is no chunked
 * (SSD matmul) prefill: it is a different association of the same sums and
 * would break that invariance; prefill measures do not call for it (L296 N2).
 *
 * Built instance: head_dim 64 x d_state 128 (the thread map holds a head's
 * [64][128] state slice in registers).  Heads, groups, the norm group, the
 * conv kernel and the channel count are runtime; tensor parallelism is a
 * rank's heads and groups (the caller offsets the pointers and passes the
 * rank's geometry).  Anything else is refused by name.
 *
 * STATE (the caller owns the pools; F3's kinds, src/engine/state_kinds.h):
 *   conv  bf16 [K-1][CC] a slot, slot s at conv_state + s * (K-1) * CC -- E3's
 *         record (rows CC apart).  FIXED with a SNAPSHOT capture: a verify
 *         writes the state after each call row (record r at
 *         conv_capture + r * (K-1) * CC); a commit of `keep` rows copies record
 *         keep-1 back.
 *   ssm   f32 [H][P][N] a slot (vLLM's layout), slot s at
 *         ssm_state + s * pulsar_mamba2_ssm_state_floats.  FIXED with a DELTA
 *         capture: with ssm_capture set the call READS each sequence's state
 *         and leaves it as it was, and stores each call row's update -- record
 *         r at ssm_capture + r * pulsar_mamba2_upd_floats: x [H*P], B [G*N],
 *         dt [H], dA [H] (f32) -- and pulsar_mamba2_commit advances a slot by
 *         any prefix of a sequence's rows with exactly the forward's
 *         operations, so the state is BIT-IDENTICAL to a forward over them.
 *   A fresh sequence starts from zeroed slots.  Both captures or neither. */
#ifndef PULSAR_CUDA_MAMBA2_H
#define PULSAR_CUDA_MAMBA2_H

#include "pulsar_cuda_sconv.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** The block's geometry as a family declares it (a TP rank's: its heads and groups). */
typedef struct {
    uint32_t n_heads;      /**< H, a multiple of n_groups */
    uint32_t head_dim;     /**< P (built: 64) */
    uint32_t d_state;      /**< N (built: 128) */
    uint32_t n_groups;     /**< G: B and C groups, each shared by H / G heads */
    uint32_t conv_kernel;  /**< K taps of the x | B | C convolution, 2 .. PULSAR_SCONV_KMAX */
    uint32_t norm_group;   /**< channels per gated-RMSNorm group: a multiple of 256, <= 8192, dividing H*P */
    float    norm_eps;
} pulsar_mamba2_geom;

/** One layer: its geometry and its (bf16, device, 16-byte aligned) parameters. */
typedef struct {
    pulsar_mamba2_geom geom;
    const uint16_t *conv_w;   /**< [CC][K], tap K-1 on the current row (Conv1d's [CC][1][K]) */
    const uint16_t *conv_b;   /**< [CC] */
    const uint16_t *A_log;    /**< [H] */
    const uint16_t *dt_bias;  /**< [H] */
    const uint16_t *D;        /**< [H] */
    const uint16_t *norm_w;   /**< [H*P], used as is (no 1 + w) */
} pulsar_mamba2_layer;

/** One call: n_seqs sequences, each ONE contiguous run of its next positions.
 *  Sequence s is call rows [seq_first[s], seq_first[s+1]) on pool slot
 *  seq_slot[s]; distinct sequences on distinct slots.  Decode is n_seqs runs of
 *  1 row, a prefill chunk 1 run, a verify runs of 2..k rows; any mix.
 *  Activation pointers are device bf16, 16-byte aligned, strides in elements
 *  and multiples of 8.  The three inputs are usually column ranges of the one
 *  in_proj row [z | x B C | dt]. */
typedef struct {
    uint32_t       n_seqs;
    uint32_t       n_rows;        /**< seq_first[n_seqs] */
    uint32_t       max_seq_rows;  /**< >= every sequence's row count */
    const int32_t *seq_first;     /**< device [n_seqs + 1], ascending from 0, every run >= 1 row */
    const int32_t *seq_slot;      /**< device [n_seqs] */
    const uint16_t *z;   uint64_t ld_z;     /**< [rows][H*P], the gate */
    const uint16_t *xbc; uint64_t ld_xbc;   /**< [rows][CC], the conv's input */
    const uint16_t *dt;  uint64_t ld_dt;    /**< [rows][H], dt before its bias (any alignment, ld >= H) */
    uint16_t *conv_state;     /**< bf16 pool, pulsar_mamba2_conv_state_elems a slot */
    float    *ssm_state;      /**< f32 pool, pulsar_mamba2_ssm_state_floats a slot, 16-byte aligned */
    uint16_t *conv_capture;   /**< NULL, or the SNAPSHOT records: one conv slot per call row */
    float    *ssm_capture;    /**< NULL, or the DELTA records: pulsar_mamba2_upd_floats per call row */
    void     *scratch;        /**< >= pulsar_mamba2_scratch_bytes(geom, n_rows), 16-byte aligned */
    size_t    scratch_bytes;
    uint16_t *out; uint64_t ld_out;         /**< [rows][H*P], the out_proj's input; must not overlap the inputs */
} pulsar_mamba2_call;

/** CC = H*P + 2*G*N, the convolution's channels. */
uint32_t pulsar_mamba2_conv_channels(const pulsar_mamba2_geom *g);
/** bf16 elements of one conv slot / SNAPSHOT record: (K-1) * CC. */
uint64_t pulsar_mamba2_conv_state_elems(const pulsar_mamba2_geom *g);
/** f32 elements of one SSM slot: H * P * N. */
uint64_t pulsar_mamba2_ssm_state_floats(const pulsar_mamba2_geom *g);
/** f32 elements of one row's DELTA record: H*P + G*N + 2*H, padded to 4. */
uint64_t pulsar_mamba2_upd_floats(const pulsar_mamba2_geom *g);
/** Scratch bytes a call of `rows` rows needs (the conv output, y, dt and dA). */
size_t pulsar_mamba2_scratch_bytes(const pulsar_mamba2_geom *g, uint32_t rows);

/** Run the block for one layer on `stream` (conv, [dt / dA: a run longer than
 *  16 rows only -- else the scan stages them], scan, norm).  Advances every sequence's conv slot and (without ssm_capture) its
 *  SSM slot; writes the captures when set.
 *  @return 0, -1 on a refused contract (named on stderr, nothing launched),
 *  -3 on a launch failure. */
int pulsar_mamba2_forward(const pulsar_mamba2_layer *layer, const pulsar_mamba2_call *call, cudaStream_t stream);

/** Advance SSM slot `slot` by the first `keep` rows of a verify sequence:
 *  `upd` is that sequence's first row's DELTA record (rows
 *  pulsar_mamba2_upd_floats apart).  Bit-identical to a forward over those
 *  rows.  keep >= 1.  @return 0, -1 refused, -3 on a launch failure. */
int pulsar_mamba2_commit(const pulsar_mamba2_geom *g, float *ssm_state, uint32_t slot, const float *upd, uint32_t keep,
                         cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* PULSAR_CUDA_MAMBA2_H */

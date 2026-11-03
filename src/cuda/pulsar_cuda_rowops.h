/** @file
 *  Row operations contract: the residual add, the norms and the bf16 / f16 emits over rows. */
/* Row operations between a layer plan's blocks (L295, Inkling's forward the first user): the residual add, the
 * RMSNorm, and the bf16 emit of a linear's f32 output -- family-neutral, over bf16 rows.
 *
 * A layer composed from the shared block library (attention, SConv, router + MoE, dense MLP) needs the same glue
 * between its blocks whatever the family: the bf16 residual stream gains a block's bf16 delta, the next block reads
 * the RMSNorm of the stream, and a linear's f32 output (the bf16-rows front door's, pulsar_rows_linear_launch) becomes
 * the bf16 row its consumer reads.  Nothing here knows which family calls it.
 *
 * THE ARITHMETIC (one per op; a row's bytes never depend on the rows beside it, the call's row count or its CTA):
 *   add       res[t][c] = bf16_rn(res[t][c] + delta[t][c])          (f32 add of the two widened values, one rounding)
 *   rmsnorm   y[t][c]   = bf16_rn((x[t][c] * rstd) * w[c]),  rstd = rsqrtf(sum_c x^2 / width + eps)
 *             -- vLLM's rmsnorm / add_rmsnorm (one rounding; the sum over the ROUNDED residual when it follows an add).
 *             The squares are one f32 fma chain per thread over its 8-element chunks in order, the warps fold by
 *             xor-shuffle and warp 0 adds the warp sums in order: the arithmetic of the head's embedding norm
 *             (pulsar_cuda_head.cu), on any row.
 *   emit      y[t][c]   = bf16_rn(x[t][c])  or  bf16_rn(x[t][c] * s)  for a device scalar s (f32 or bf16 stored) --
 *             the producer-stage encode of a linear whose arm has no bf16 epilogue yet (rule 3: the ONE encoder for
 *             that buffer, at the producer).
 L338 (fused chains: the ops above composed on one row in one CTA, byte for byte the unfused sequence):
 *   norm-add-norms  res = add(res, rmsnorm(x) * w_x), then up to three rmsnorms of the updated row (one rstd)
 *   sum2-norm-add   two f32 branches emitted and normed, added, the sum normed, res = bf16_rn(bf16_rn(res + it) * s)
 *                   for a device bf16 scalar s (Gemma 4's layer_scalar: torch's add, then its mul), then its rmsnorm
 * L321 (the encoder-decoder families: LayerNorm, biased linears, GELU, an f32 residual -- Nemotron Parse first;
 * weights and biases f32, and the rows they write bf16 or f16 by the consumer's format, rule 3):
 *   res32_add   res[t][c] += x[t][c] + bias[c]  in f32, for a linear's f32 output x -- an f32 RESIDUAL stream (a ViT's:
 *               its few very-high-norm tokens make a bf16 residual drop the small updates every other row gets, and
 *               drift -- Parse's RADIO tower measured it) -- then, when w is given, y = layernorm(res)
 *   layernorm32 y = (x - mean) * rstd * w + b,  mean = sum_c x / width, rstd = rsqrtf(sum_c (x - mean)^2 / width + eps)
 *               -- torch's LayerNorm, two passes (the mean, then the centred squares), each an f32 chain per thread
 *               folded as rmsnorm's -- from bf16 rows or f32 rows (+ an f32 bias first: a linear's output and its
 *               bias, normalised with no rounding between), to bf16 / f16 rows and / or f32 rows
 *   bias_emit   y[t][c] = act(x[t][c] + bias[c]),  act = none, GELU (erf: 0.5 v (1 + erf(v / sqrt 2))) or (L331) relu^2
 *               (max(v, 0)^2, one rounding of the f32 square), as bf16 / f16
 *   gather      y[t][c] = f32(table[ids[t]][c]) * scale  as f32 rows, from an f16 table (a tied f16 embedding / head)
 *
 * Every pointer is a device pointer, 16-byte aligned; widths and leading dimensions are in elements, multiples of 8. */
#ifndef PULSAR_CUDA_ROWOPS_H
#define PULSAR_CUDA_ROWOPS_H

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
/// The CUDA runtime's stream handle, declared for CUDA-free includers (same definition as cuda_runtime.h's).
typedef struct CUstream_st *cudaStream_t;
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** How an emit's scalar is stored. */
typedef enum {
    PULSAR_ROWS_SCALE_NONE = 0,   /**< no scalar */
    PULSAR_ROWS_SCALE_F32 = 1,    /**< one f32 */
    PULSAR_ROWS_SCALE_BF16 = 2,   /**< one bf16 */
} pulsar_rows_scale;

/** y = rmsnorm(x) * w for n_rows rows of `width`: x bf16 at x + t * ld_x, w bf16 [width], y bf16 at y + t * ld_y.
 *  y may not overlap x.  @return 0, -1 refused (named on stderr, nothing launched), -3 launch failure. */
/// @param x      device bf16 input rows (16-byte aligned)
/// @param ld_x   x row pitch in elements (a multiple of 8, >= width)
/// @param n_rows rows (one CTA each)
/// @param width  row width (a multiple of 8)
/// @param w      device bf16 [width] norm weight (16-byte aligned)
/// @param eps    the norm's epsilon
/// @param y      device bf16 output rows (16-byte aligned)
/// @param ld_y   y row pitch in elements (a multiple of 8)
/// @param stream stream to launch on
int pulsar_rows_rmsnorm_launch(const uint16_t *x, uint64_t ld_x, uint32_t n_rows, uint32_t width, const uint16_t *w,
                               float eps, uint16_t *y, uint64_t ld_y, cudaStream_t stream);

/** res += delta (in place, one rounding), then, when `w` is given, y = rmsnorm(res) * w of the updated row.
 *  res and delta bf16 rows (ld_res / ld_delta apart), y bf16 rows (ld_y apart; unread when w is NULL).
 *  @return 0, -1 refused, -3 launch failure. */
/// @param res      device bf16 residual rows, updated in place
/// @param ld_res   res row pitch in elements (a multiple of 8)
/// @param delta    device bf16 rows added to res
/// @param ld_delta delta row pitch in elements (a multiple of 8)
/// @param n_rows   rows
/// @param width    row width (a multiple of 8)
/// @param w        device bf16 [width] norm weight, or NULL: the add alone
/// @param eps      the norm's epsilon
/// @param y        device bf16 normed output rows (with w)
/// @param ld_y     y row pitch in elements
/// @param stream   stream to launch on
int pulsar_rows_add_launch(uint16_t *res, uint64_t ld_res, const uint16_t *delta, uint64_t ld_delta, uint32_t n_rows,
                           uint32_t width, const uint16_t *w, float eps, uint16_t *y, uint64_t ld_y,
                           cudaStream_t stream);

/** L338: res = bf16(res + bf16(rmsnorm(x) * w_x)) in place, then y[j] = bf16(rmsnorm(res) * w[j]) for each j whose
 *  w[j] is given -- a block's post-norm, the residual add and the next blocks' input norms in one pass, bit for bit
 *  rows_rmsnorm then rows_add (and rows_rmsnorm again for the extra norms).  Width at most 8192.
 *  @return 0, -1 refused, -3 launch failure. */
/// @param x      device bf16 delta rows, normed before the add (16-byte aligned)
/// @param ld_x   x pitch in elements (a multiple of 8, >= width)
/// @param w_x    device bf16 [width] the delta's norm weight
/// @param res    device bf16 residual rows, updated in place
/// @param ld_res res pitch in elements
/// @param n_rows rows (one CTA each)
/// @param width  row width (a multiple of 8, <= 8192)
/// @param w      [3] device bf16 [width] norm weights of the updated residual, NULL = that output unused
/// @param y      [3] their output rows
/// @param ld_y   [3] their pitches in elements
/// @param eps    the norms' epsilon
/// @param stream stream to launch on
int pulsar_rows_norm_add_norms_launch(const uint16_t *x, uint64_t ld_x, const uint16_t *w_x, uint16_t *res,
                                      uint64_t ld_res, uint32_t n_rows, uint32_t width, const uint16_t *const w[3],
                                      uint16_t *const y[3], const uint64_t ld_y[3], float eps, cudaStream_t stream);

/** L338: two f32 branch outputs into the residual, bit for bit rows_emit (each), rows_rmsnorm (each), rows_add with
 *  w_s, then the scaled add with w_n: s = bf16(bf16(rmsnorm(bf16(a)) * w_a) + bf16(rmsnorm(bf16(b)) * w_b));
 *  res = bf16(bf16(res + bf16(rmsnorm(s) * w_s)) * scale); y = bf16(rmsnorm(res) * w_n).  Width at most 8192.
 *  @return 0, -1 refused, -3 launch failure. */
/// @param a      device f32 rows of branch a (16-byte aligned)
/// @param ld_a   a pitch in elements (a multiple of 8, >= width)
/// @param w_a    device bf16 [width] branch a's norm weight
/// @param b      device f32 rows of branch b
/// @param ld_b   b pitch in elements
/// @param w_b    branch b's norm weight
/// @param w_s    the branch sum's norm weight
/// @param res    device bf16 residual rows, updated in place
/// @param ld_res res pitch in elements
/// @param scale  device bf16 scalar the summed residual is multiplied by
/// @param w_n    the updated residual's norm weight
/// @param y      device bf16 rows of that norm
/// @param ld_y   y pitch in elements
/// @param n_rows rows (one CTA each)
/// @param width  row width (a multiple of 8, <= 8192)
/// @param eps    the norms' epsilon
/// @param stream stream to launch on
int pulsar_rows_sum2_norm_add_launch(const float *a, uint64_t ld_a, const uint16_t *w_a, const float *b,
                                     uint64_t ld_b, const uint16_t *w_b, const uint16_t *w_s, uint16_t *res,
                                     uint64_t ld_res, const uint16_t *scale, const uint16_t *w_n, uint16_t *y,
                                     uint64_t ld_y, uint32_t n_rows, uint32_t width, float eps, cudaStream_t stream);

/** y = bf16(x [* s]) for n_rows rows of `width`: x f32 rows ld_x apart (elements; a multiple of 4), y bf16 rows ld_y
 *  apart.  `scale` is a device pointer to one value stored as `kind` (NULL with PULSAR_ROWS_SCALE_NONE).
 *  @return 0, -1 refused, -3 launch failure. */
/// @param x      device f32 input rows
/// @param ld_x   x row pitch in elements (a multiple of 4)
/// @param n_rows rows
/// @param width  row width (a multiple of 4)
/// @param scale  device pointer to the scalar, or NULL with PULSAR_ROWS_SCALE_NONE
/// @param kind   how the scalar is stored
/// @param y      device bf16 output rows
/// @param ld_y   y row pitch in elements
/// @param stream stream to launch on
int pulsar_rows_emit_bf16_launch(const float *x, uint64_t ld_x, uint32_t n_rows, uint32_t width, const void *scale,
                                 pulsar_rows_scale kind, uint16_t *y, uint64_t ld_y, cudaStream_t stream);

/** L321: an emit's activation. */
typedef enum {
    /** identity */
    PULSAR_ROWS_ACT_NONE = 0,
    PULSAR_ROWS_ACT_GELU = 1,   /**< exact (erf) GELU, torch's default */
    PULSAR_ROWS_ACT_RELU2 = 2,  /**< L331: relu(v)^2 (Nemotron's ReLU-squared: the vision projector's mlp1) */
} pulsar_rows_act;

/** L321: the 16-bit rows an op writes -- the consumer's operand format (bf16 rows, or the f16 rows of an fp16
 *  tensor-core consumer: pulsar_rows_linear_f16_launch, pulsar_fmha_f16_launch). */
typedef enum {
    /** bf16 rows */
    PULSAR_ROWS_OUT_BF16 = 0,
    /** f16 rows */
    PULSAR_ROWS_OUT_F16 = 1,
} pulsar_rows_out;

/** y = act(x + bias) for n_rows rows of `width` (x f32 rows ld_x apart, a multiple of 4; bias f32 [width] or NULL), as
 *  y_fmt rows ld_y apart.  @return 0, -1 refused, -3 launch failure. */
/// @param x      device f32 input rows
/// @param ld_x   x row pitch in elements (a multiple of 4)
/// @param n_rows rows
/// @param width  row width (a multiple of 4)
/// @param bias   device f32 [width], or NULL
/// @param act    the activation applied after the bias
/// @param y      device 16-bit output rows
/// @param ld_y   y row pitch in elements
/// @param y_fmt  y's element format
/// @param stream stream to launch on
int pulsar_rows_bias_emit_launch(const float *x, uint64_t ld_x, uint32_t n_rows, uint32_t width, const float *bias,
                                 pulsar_rows_act act, uint16_t *y, uint64_t ld_y, pulsar_rows_out y_fmt,
                                 cudaStream_t stream);

/** res += x + bias in f32 (res and x f32 rows ld_res / ld_x apart; bias f32 [width] or NULL), then, when `w` is given,
 *  y = layernorm(res) * w + b as y_fmt rows.  @return 0, -1 refused, -3 launch failure. */
/// @param res    device f32 residual rows, updated in place
/// @param ld_res res row pitch in elements
/// @param x      device f32 rows added to res
/// @param ld_x   x row pitch in elements
/// @param bias   device f32 [width] added with x, or NULL
/// @param n_rows rows
/// @param width  row width
/// @param w      device f32 [width] LayerNorm weight, or NULL: the add alone
/// @param b      device f32 [width] LayerNorm bias (with w)
/// @param eps    the norm's epsilon
/// @param y      device 16-bit normed output rows (with w)
/// @param ld_y   y row pitch in elements
/// @param y_fmt  y's element format
/// @param stream stream to launch on
int pulsar_rows_res32_add_launch(float *res, uint64_t ld_res, const float *x, uint64_t ld_x, const float *bias,
                                 uint32_t n_rows, uint32_t width, const float *w, const float *b, float eps, uint16_t *y,
                                 uint64_t ld_y, pulsar_rows_out y_fmt, cudaStream_t stream);

/** layernorm(x [+ x_bias]) * w + b for n_rows rows of `width`: x bf16 rows (x_f32 false) or f32 rows, ld_x apart, and
 *  x_bias f32 [width] (f32 rows only) or NULL; out to y (y_fmt rows, ld_y) and / or yf (f32, ld_yf) -- at least one;
 *  neither may overlap x.  @return 0, -1 refused, -3 launch failure. */
/// @param x      device input rows (bf16 or f32 by x_f32)
/// @param x_f32  true: x holds f32 rows
/// @param ld_x   x row pitch in elements
/// @param x_bias device f32 [width] added to f32 rows, or NULL
/// @param n_rows rows
/// @param width  row width
/// @param w      device f32 [width] LayerNorm weight
/// @param b      device f32 [width] LayerNorm bias
/// @param eps    the norm's epsilon
/// @param y      device 16-bit output rows, or NULL
/// @param ld_y   y row pitch in elements
/// @param y_fmt  y's element format
/// @param yf     device f32 output rows, or NULL
/// @param ld_yf  yf row pitch in elements
/// @param stream stream to launch on
int pulsar_rows_layernorm32_launch(const void *x, bool x_f32, uint64_t ld_x, const float *x_bias, uint32_t n_rows,
                                   uint32_t width, const float *w, const float *b, float eps, uint16_t *y, uint64_t ld_y,
                                   pulsar_rows_out y_fmt, float *yf, uint64_t ld_yf, cudaStream_t stream);

/** y[t] = table[ids[t]] * scale as f32 rows for n device ids (an id outside [0, n_vocab) writes a NaN row), table f16
 *  [n_vocab][width] (16-byte aligned).  @return 0, -1 refused, -3 launch failure. */
/// @param table   device f16 [n_vocab][width] (16-byte aligned)
/// @param n_vocab rows in the table
/// @param width   row width (a multiple of 8)
/// @param ids     device [n] token ids
/// @param n       output rows
/// @param scale   multiplies every element
/// @param y       device f32 output rows
/// @param ld_y    y row pitch in elements
/// @param stream  stream to launch on
int pulsar_rows_gather_f16_launch(const uint16_t *table, uint32_t n_vocab, uint32_t width, const int32_t *ids,
                                  uint32_t n, float scale, float *y, uint64_t ld_y, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* PULSAR_CUDA_ROWOPS_H */

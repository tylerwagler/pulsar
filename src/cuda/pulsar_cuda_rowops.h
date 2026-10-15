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
 *
 * Every pointer is a device pointer, 16-byte aligned; widths and leading dimensions are in elements, multiples of 8. */
#ifndef PULSAR_CUDA_ROWOPS_H
#define PULSAR_CUDA_ROWOPS_H

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

/** How an emit's scalar is stored. */
typedef enum {
    PULSAR_ROWS_SCALE_NONE = 0,   /**< no scalar */
    PULSAR_ROWS_SCALE_F32 = 1,    /**< one f32 */
    PULSAR_ROWS_SCALE_BF16 = 2,   /**< one bf16 */
} pulsar_rows_scale;

/** y = rmsnorm(x) * w for n_rows rows of `width`: x bf16 at x + t * ld_x, w bf16 [width], y bf16 at y + t * ld_y.
 *  y may not overlap x.  @return 0, -1 refused (named on stderr, nothing launched), -3 launch failure. */
int pulsar_rows_rmsnorm_launch(const uint16_t *x, uint64_t ld_x, uint32_t n_rows, uint32_t width, const uint16_t *w,
                               float eps, uint16_t *y, uint64_t ld_y, cudaStream_t stream);

/** res += delta (in place, one rounding), then, when `w` is given, y = rmsnorm(res) * w of the updated row.
 *  res and delta bf16 rows (ld_res / ld_delta apart), y bf16 rows (ld_y apart; unread when w is NULL).
 *  @return 0, -1 refused, -3 launch failure. */
int pulsar_rows_add_launch(uint16_t *res, uint64_t ld_res, const uint16_t *delta, uint64_t ld_delta, uint32_t n_rows,
                           uint32_t width, const uint16_t *w, float eps, uint16_t *y, uint64_t ld_y,
                           cudaStream_t stream);

/** y = bf16(x [* s]) for n_rows rows of `width`: x f32 rows ld_x apart (elements; a multiple of 4), y bf16 rows ld_y
 *  apart.  `scale` is a device pointer to one value stored as `kind` (NULL with PULSAR_ROWS_SCALE_NONE).
 *  @return 0, -1 refused, -3 launch failure. */
int pulsar_rows_emit_bf16_launch(const float *x, uint64_t ld_x, uint32_t n_rows, uint32_t width, const void *scale,
                                 pulsar_rows_scale kind, uint16_t *y, uint64_t ld_y, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* PULSAR_CUDA_ROWOPS_H */

/* The block drafter's own small ops (L298): what a DFlash-family block drafter (spec_block.h) needs beyond the shared
 * GEMM front doors (pulsar_rows_linear_*) and the GQA block (pulsar_cuda_attn_gqa.h).  Format- and family-neutral.
 * Nothing in FlashInfer / b12x / CUTLASS / cuBLASLt covers these shapes (a per-token dynamic grouped convolution of
 * two taps inside an 8-row block, a row gather, a k = 8 selection over a vocabulary row), so they are small kernels
 * here; every GEMM and the attention are the libraries' / the shared blocks'.
 *
 *   grouped conv   DFlash2's dynamic grouped convolution (vLLM qwen3_dflash2.py _dflash2_grouped_conv_kernel):
 *                  out[r, c] = (base[0, c] + delta[r, 0, g]) * x[r, c]
 *                            + sum_{t >= 1, p(r) >= t} (base[t, c] + delta[r, t, g]) * x[r - t, c],
 *                  g = c / group, p(r) = r % block (rows are blocks of `block` rows, the convolution never crosses one),
 *                  every operand widened to f32, the taps summed in order with fused multiply-adds (Triton's), one
 *                  bf16 rounding at the end
 *   swiglu         h = bf16(silu(g) * u) over bf16 g, u (vLLM silu_and_mul: f32 silu(x) = x / (1 + exp(-x)))
 *   row copy       bf16 rows -> a column window of wider rows (the target's tapped hidden states into the
 *                  drafter's concatenated context row)
 *   stream mean    (L299) rows of n_streams bf16 streams [row][stream][width] -> their mean, bf16, into a column
 *                  window of wider rows: a hyper-connection trunk's tap (the streams summed in stream order in f32,
 *                  times 1 / n_streams, one bf16 rounding -- the reference's bf16(streams.mean(-2)), exact up to that
 *                  rounding while the streams' exponents span under 2^16)
 *   top-k          per row of bf16 logits (the head's rows as its producer emits them): the k largest rows (ties to
 *                  the lower row), and their log-softmax values over the whole row (f32); a head over a subset of the
 *                  vocabulary maps each row to its id (L299: `id_map`, the head's rows' ids) */
#ifndef PULSAR_CUDA_BLOCKDRAFT_H
#define PULSAR_CUDA_BLOCKDRAFT_H

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
typedef struct CUstream_st *cudaStream_t;
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** The largest k pulsar_rows_topk_logprob_launch selects. */
#define PULSAR_BLOCKDRAFT_TOPK_MAX 16

/** out [n_rows][width] bf16 (ld_out) = the grouped conv of x [n_rows][width] bf16 (ld_x) with the per-row tap
 *  deltas delta [n_rows] x [taps][width / group] bf16 (row r at delta + r * ld_delta) and the base kernel
 *  base [taps][width] bf16.  n_rows % block == 0; out may not overlap x.  @return 0, -1 refused, -3 launch failure. */
int pulsar_blockdraft_conv_launch(const uint16_t *x, uint64_t ld_x, const uint16_t *delta, uint64_t ld_delta,
                                  const uint16_t *base, uint32_t taps, uint32_t group, uint32_t block, uint32_t n_rows,
                                  uint32_t width, uint16_t *out, uint64_t ld_out, cudaStream_t stream);

/** h [n_rows][width] bf16 (ld_h) = bf16(silu(g) * u) for g, u bf16 rows (ld_g, ld_u).  @return 0, -1, -3. */
int pulsar_blockdraft_swiglu_launch(const uint16_t *g, uint64_t ld_g, const uint16_t *u, uint64_t ld_u,
                                    uint32_t n_rows, uint32_t width, uint16_t *h, uint64_t ld_h, cudaStream_t stream);

/** dst row r [0, width) (dst + r * ld_dst) = src row r (src + r * ld_src), bf16, n_rows rows; width % 8 == 0 and both
 *  16-byte aligned rows.  @return 0, -1, -3. */
int pulsar_blockdraft_rows_copy_launch(const uint16_t *src, uint64_t ld_src, uint32_t n_rows, uint32_t width,
                                       uint16_t *dst, uint64_t ld_dst, cudaStream_t stream);

/** dst row r [0, width) (dst + r * ld_dst) = bf16 of the mean of src row r's n_streams streams, stream i at
 *  src + r * ld_src + i * width (bf16).  width % 8 == 0, 16-byte aligned rows.  @return 0, -1, -3. */
int pulsar_blockdraft_stream_mean_launch(const uint16_t *src, uint64_t ld_src, uint32_t n_streams, uint32_t n_rows,
                                         uint32_t width, uint16_t *dst, uint64_t ld_dst, cudaStream_t stream);

/** For each of n_rows rows of `logits` [row][ld] bf16 (the first n columns are the row): ids [row][k] = the k largest
 *  values in descending order (ties: the lower column first) as ids -- the column itself, or id_map[column] when the
 *  head's rows are a subset of the vocabulary (id_map: device [n] i32; NULL = the columns ARE the ids) --, lp [row][k]
 *  = v - max - log(sum exp(v - max)) over the row (f32).  1 <= k <= PULSAR_BLOCKDRAFT_TOPK_MAX.  @return 0, -1, -3. */
int pulsar_rows_topk_logprob_launch(const uint16_t *logits, uint64_t ld, uint32_t n_rows, uint32_t n, uint32_t k,
                                    const int32_t *id_map, int32_t *ids, float *lp, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* PULSAR_CUDA_BLOCKDRAFT_H */

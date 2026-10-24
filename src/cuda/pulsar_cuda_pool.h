/* Pooling over a sequence's rows (L321: the Nemotron-3.5 encoders' mean pool) -- family-neutral, over bf16 rows.
 *
 * An encoder's output is one vector per input: the mean of its last hidden rows.  The sum is f32 and runs in ROW
 * ORDER -- one f32 add per row per column, the column a thread's own -- so a column's bits depend on the rows' values
 * only, never on the grid, and a host loop in the same order reproduces them exactly.  (bf16 sums overflow nothing
 * here but lose the mean's low bits; fp16 sums overflow outright -- an fp16 reranker scored -inf over ~1.6k tokens,
 * retriever/README.md -- so the accumulator is f32 by contract.)
 *
 * THE ARITHMETIC:  out[c] = (sum_{t = 0 .. n-1, in order} f32(x[t][c])) / n      (IEEE f32 add and divide)
 *
 * Every pointer is a device pointer, 16-byte aligned; widths and leading dimensions are in elements, multiples of 8. */
#ifndef PULSAR_CUDA_POOL_H
#define PULSAR_CUDA_POOL_H

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

/** out = the f32 mean of n_rows bf16 rows of `width` (x + t * ld_x), out f32 [width].
 *  @return 0, -1 refused (named on stderr, nothing launched), -3 launch failure. */
int pulsar_rows_mean_f32_launch(const uint16_t *x, uint64_t ld_x, uint32_t n_rows, uint32_t width, float *out,
                                cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

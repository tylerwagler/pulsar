/** @file
 *  Pooling over a sequence's rows: the row-order f32 mean of bf16 rows. */
/* Pooling over a sequence's rows (L321) -- see pulsar_cuda_pool.h for the contract and the arithmetic. */
#include "pulsar_cuda_pool.h"

#include <cuda_bf16.h>
#include <stdio.h>

namespace {

/// Block size of rows_mean_kernel (one thread a column).
constexpr int kThreads = 256;

/* one thread per column: the rows in order, one IEEE f32 add each (the engine builds with --use_fast_math, so the add
 * and the divide are the _rn intrinsics -- no contraction, no approximate reciprocal) */
/** Column mean over n bf16 rows: out[c] = (sum_t x[t][c]) / n, the adds in row order.  Grid ceil(width / kThreads);
 *  block kThreads; no smem.
 *  @param x     device bf16 rows, row t at x + t * ld
 *  @param ld    row pitch in elements
 *  @param n     rows to pool (> 0)
 *  @param width columns
 *  @param out   device f32 [width] */
__global__ void __launch_bounds__(kThreads)
rows_mean_kernel(const uint16_t *__restrict__ x, uint64_t ld, uint32_t n, uint32_t width, float *__restrict__ out) {
    const uint32_t c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= width) return;
    float acc = 0.0f;
    for (uint32_t t = 0; t < n; t++) acc = __fadd_rn(acc, __uint_as_float((uint32_t)x[(uint64_t)t * ld + c] << 16));
    out[c] = __fdiv_rn(acc, (float)n);
}

/// @param p a pointer
/// @return true when p is 16-byte aligned
bool aligned16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

}  // namespace

extern "C" int pulsar_rows_mean_f32_launch(const uint16_t *x, uint64_t ld_x, uint32_t n_rows, uint32_t width, float *out,
                                           cudaStream_t stream) {
    if (!x || !out || n_rows == 0 || width == 0 || width % 8 || ld_x < width || ld_x % 8 || !aligned16(x) ||
        !aligned16(out)) {
        fprintf(stderr, "pulsar: rows mean: a null / misaligned pointer, no rows, or a width / ld not a multiple of 8 "
                        "(rows %u, width %u, ld %llu) -- refusing\n", n_rows, width, (unsigned long long)ld_x);
        return -1;
    }
    rows_mean_kernel<<<(width + kThreads - 1) / kThreads, kThreads, 0, stream>>>(x, ld_x, n_rows, width, out);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: rows mean: launch failed: %s\n", cudaGetErrorString(e));
        return -3;
    }
    return 0;
}

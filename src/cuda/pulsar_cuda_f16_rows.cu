/* The f16 rows arm (L321) -- see pulsar_cuda_f16_rows.h for the contract and the arithmetic. */
#include "pulsar_cuda_f16_rows.h"
#include "pulsar_cuda_lt_fixed.h"

#include "pulsar_gpu.h"   /* PULSAR_GPU_MNEUTRAL_ROWS_MAX */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>

namespace {

constexpr int kThreads = 256;
constexpr int kHiExp = 14;   ///< the row's largest |value| lands just under 2^14 (the EXL3 prompt arm's prescale)

size_t up256(size_t b) { return (b + 255u) & ~(size_t)255u; }

__device__ __forceinline__ float pow2(int p) { return __int_as_float((p + 127) << 23); }

/* one CTA per row: the row's amax, then the scaled f16 plane and the row's inverse scale */
__global__ void __launch_bounds__(kThreads)
prep_kernel(const __nv_bfloat16 *__restrict__ x, __half *__restrict__ xh, float *__restrict__ inv, int K) {
    __shared__ float s_amax[kThreads / 32];
    const int m = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const __nv_bfloat16 *xr = x + (size_t)m * K;
    float amax = 0.0f;
    for (int k = threadIdx.x; k < K; k += kThreads) amax = fmaxf(amax, fabsf(__bfloat162float(xr[k])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) s_amax[warp] = amax;
    __syncthreads();
    float rmax = 0.0f;
#pragma unroll
    for (int w = 0; w < kThreads / 32; ++w) rmax = fmaxf(rmax, s_amax[w]);
    int p = rmax > 0.0f ? kHiExp - (((__float_as_int(rmax) >> 23) & 0xff) - 126) : 0;
    p = max(-100, min(100, p));
    const float up = pow2(p);
    __half *xo = xh + (size_t)m * K;
    for (int k = threadIdx.x; k < K; k += kThreads) xo[k] = __float2half_rn(__bfloat162float(xr[k]) * up);
    if (threadIdx.x == 0) inv[m] = pow2(-p);
}

/* y[m][n] *= inv[m] (an exact power of two) */
__global__ void __launch_bounds__(kThreads)
unscale_kernel(float *__restrict__ y, const float *__restrict__ inv, int M, int N) {
    const size_t i = (size_t)blockIdx.x * kThreads + threadIdx.x;
    if (i < (size_t)M * N) y[i] *= inv[i / (size_t)N];
}

}  // namespace

size_t pulsar_f16_rows_workspace_bytes(int rows, int in) {
    return rows > 0 && in > 0 ? up256((size_t)rows * in * sizeof(__half)) + up256((size_t)rows * sizeof(float)) : 0;
}

int pulsar_f16_rows_linear_launch(const void *w, const uint16_t *x_bf16, int rows, int in, int out, float *y,
                                  bool prompt, void *ws, size_t ws_bytes, cudaStream_t stream) {
    const char *tag = "pulsar: f16 rows linear";
    if (!w || !x_bf16 || !y || rows <= 0 || in <= 0 || out <= 0 || in % 8 || out % 8 || ((uintptr_t)w & 15u) ||
        !ws || ws_bytes < pulsar_f16_rows_workspace_bytes(rows, in)) {
        fprintf(stderr, "%s: %d rows %d -> %d, a null / misaligned weight, a width not a multiple of 8, or a workspace "
                        "of %zu B < %zu B -- refusing\n", tag, rows, in, out, ws_bytes,
                pulsar_f16_rows_workspace_bytes(rows, in));
        return -1;
    }
    __half *xh = static_cast<__half *>(ws);
    float *inv = reinterpret_cast<float *>(static_cast<uint8_t *>(ws) + up256((size_t)rows * in * sizeof(__half)));
    prep_kernel<<<rows, kThreads, 0, stream>>>(reinterpret_cast<const __nv_bfloat16 *>(x_bf16), xh, inv, in);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "%s: prep launch failed\n", tag);
        return -3;
    }
    /* prompt rows: one call on the prompt class; decode rows: blocks of the M-neutral width on the decode class */
    const int blk = prompt ? rows : (int)PULSAR_GPU_MNEUTRAL_ROWS_MAX;
    for (int r0 = 0; r0 < rows; r0 += blk) {
        const int m = rows - r0 < blk ? rows - r0 : blk;
        const int rc = pulsar_lt_fixed_gemm(PULSAR_LT_A_NK, prompt ? PULSAR_LT_PROMPT : PULSAR_LT_DECODE, w, in, out, 0,
                                            out, xh + (size_t)r0 * in, m, y + (size_t)r0 * out, out, stream, tag);
        if (rc) return rc;
    }
    const size_t n = (size_t)rows * out;
    unscale_kernel<<<(unsigned)((n + kThreads - 1) / kThreads), kThreads, 0, stream>>>(y, inv, rows, out);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "%s: unscale launch failed\n", tag);
        return -3;
    }
    return 0;
}

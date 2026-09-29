// SPDX-License-Identifier: MIT
// The dense EXL3 arm for PROMPT CHUNKS (L251): M > 16 rows of ds4_exl3_dense_launch.
//
// The decode arm (ds4_exl3_dense.cu) is a GEMV: every CTA decodes its trellis tiles for 16 rows and
// reads the activation planes for 32 outputs, so a 4096-row prompt re-decodes each weight 256 times
// and re-reads each activation N/32 times.  This is the GEMM, TensorFold's shape for the same format
// (tensorfold/cuda/exl3/prefill.py: "W_q decoded once a chunk into a ... fp16 GEMM"):
//
//   1. DECODE  the trellis ONCE into an fp16 K x N matrix W_q (row-major), with the GEMV's own tile
//              decode (mul1_pair: the same fp16 values, bit for bit).
//   2. PREP    per row: x * suh, H128 per 128-block, then ONE power-of-two prescale for the whole row
//              (so a GEMM can sum across blocks), rounded to one fp16 plane: 11 significant bits,
//              8x finer than the bf16 row it came from (exllamav3's and TensorFold's operand; a second
//              "lo" plane doubled the GEMM for precision the input does not carry).
//   3. GEMM    one cuBLAS tensor-core GEMM, Z = X W_q, f32 accumulate and output.
//   4. EPILOGUE per row: times 2^-P, the output H128 per 128-block, times svh.
//
// A prefilled row and a decoded row agree to rounding, not to the bit (Tyler 2026-09-29, "Don't use
// a decode kernel for prefill"); graded against double by tests/exl3_dense_gate at every rate.
// The scratch (W_q, the planes, the row scales) is one device buffer grown to the largest call.

#include "ds4_exl3_dev.cuh"
#include "engine/exl3_trellis.h"

#include <cstdio>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kHiExp = 14;   ///< the row's largest |value| lands just under 2^14 (the GEMV's prescale)

__device__ __forceinline__ float pow2f(int e) { return __int_as_float((127 + e) << 23); }

/* 1. one warp per 16 x 16 tile (kt, nt): lane t holds positions 8t..8t+7 -- k rows (t%4)*2 + {0,1,8,9}
 * of columns t/4 (positions 8t..8t+3) and t/4 + 8 (8t+4..8t+7), the mma B-fragment order. */
template <int K2>
__global__ void __launch_bounds__(256)
dense_decode_kernel(const uint32_t *__restrict__ trellis, __half *__restrict__ wq, int K, int N) {
    constexpr int W32 = exl3dev::Rate<K2>::words32;
    const int64_t tile = (int64_t)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int ntn = N >> 4;
    if (tile >= (int64_t)(K >> 4) * ntn) return;
    const int kt = (int)(tile / ntn), nt = (int)(tile % ntn);
    const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const uint32_t *t = trellis + (size_t)tile * W32;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        int lo, hi, sh;
        exl3dev::run4_window<K2>(8 * lane + 4 * h, lo, hi, sh);
        uint32_t st[4];
        exl3dev::run4_split<K2>(EXL3_RUN4_JOIN(t[lo], t[hi], sh), 8 * lane + 4 * h, st);
        const uint32_t b0 = exl3dev::mul1_pair(st[0], st[1]);   /* k rows 2c, 2c+1 */
        const uint32_t b1 = exl3dev::mul1_pair(st[2], st[3]);   /* k rows 2c+8, 2c+9 */
        const int col = nt * 16 + g + 8 * h, k0 = kt * 16 + 2 * c;
        const __half2 p0 = *reinterpret_cast<const __half2 *>(&b0), p1 = *reinterpret_cast<const __half2 *>(&b1);
        wq[(size_t)(k0 + 0) * N + col] = __low2half(p0);
        wq[(size_t)(k0 + 1) * N + col] = __high2half(p0);
        wq[(size_t)(k0 + 8) * N + col] = __low2half(p1);
        wq[(size_t)(k0 + 9) * N + col] = __high2half(p1);
    }
}

/* 2. one CTA (8 warps) per row: the rotated row into shared memory block by block, the row amax,
 * then the fp16 plane and the row's inverse prescale. */
__global__ void __launch_bounds__(256)
dense_prep_kernel(const __nv_bfloat16 *__restrict__ x, const __half *__restrict__ suh,
                  __half *__restrict__ xhi, float *__restrict__ inv, int K) {
    extern __shared__ float s_row[];              /* [K] */
    __shared__ float s_amax[8];
    const int m = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float amax = 0.0f;
    for (int blk = warp; blk < (K >> 7); blk += 8) {
        const int k = blk * 128 + lane * 4;
        const uint2 xw = *reinterpret_cast<const uint2 *>(x + (size_t)m * K + k);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&xw.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&xw.y));
        const uint2 sw = *reinterpret_cast<const uint2 *>(suh + k);
        const __half2 s01 = *reinterpret_cast<const __half2 *>(&sw.x), s23 = *reinterpret_cast<const __half2 *>(&sw.y);
        float v[4] = {a.x * __low2float(s01), a.y * __high2float(s01), b.x * __low2float(s23), b.y * __high2float(s23)};
        exl3dev::had128(v);
#pragma unroll
        for (int i = 0; i < 4; ++i) { s_row[k + i] = v[i]; amax = fmaxf(amax, fabsf(v[i])); }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) s_amax[warp] = amax;
    __syncthreads();
    float rmax = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) rmax = fmaxf(rmax, s_amax[w]);
    int p = rmax > 0.0f ? kHiExp - (((__float_as_int(rmax) >> 23) & 0xff) - 126) : 0;
    p = max(-100, min(100, p));
    const float up = pow2f(p);
    for (int k = threadIdx.x * 2; k < K; k += 512) {
        const float a0 = s_row[k] * up, a1 = s_row[k + 1] * up;
        *reinterpret_cast<__half2 *>(xhi + (size_t)m * K + k) = __floats2half2_rn(a0, a1);
    }
    if (threadIdx.x == 0) inv[m] = pow2f(-p);
}

/* 4. one warp per (row, 128-block of outputs), in place on y. */
__global__ void __launch_bounds__(256)
dense_epilogue_kernel(float *__restrict__ y, const float *__restrict__ inv, const __half *__restrict__ svh,
                      int M, int N) {
    const int64_t task = (int64_t)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int n_blk = N >> 7;
    if (task >= (int64_t)M * n_blk) return;
    const int m = (int)(task / n_blk), blk = (int)(task % n_blk), lane = threadIdx.x & 31, n0 = blk * 128 + lane * 4;
    float4 *p = reinterpret_cast<float4 *>(y + (size_t)m * N + n0);
    const float4 q = *p;
    const float s = inv[m];
    float z[4] = {q.x * s, q.y * s, q.z * s, q.w * s};
    exl3dev::had128(z);
    const __half2 v01 = *reinterpret_cast<const __half2 *>(svh + n0), v23 = *reinterpret_cast<const __half2 *>(svh + n0 + 2);
    *p = make_float4(z[0] * __low2float(v01), z[1] * __high2float(v01), z[2] * __low2float(v23), z[3] * __high2float(v23));
}

uint8_t *g_scratch = nullptr;
size_t g_scratch_bytes = 0;
cublasHandle_t g_blas = nullptr;

size_t up256(size_t b) { return (b + 255u) & ~(size_t)255u; }

} // namespace

int qwen_exl3_dense_prefill_launch(const void *w, int k2, const void *x_bf16, float *y, int M, int K, int N,
                                   uint64_t trellis_bytes, cudaStream_t stream) {
    const char *tag = "qwen_exl3_dense_prefill_launch";
    if (!w || !x_bf16 || !y || M <= 0 || N % 128 || K % EXL3_HAD_BLOCK) {
        fprintf(stderr, "%s: bad arguments (M=%d K=%d N=%d) -- refusing\n", tag, M, K, N);
        return -1;
    }
    const size_t wq_b = up256((size_t)K * N * sizeof(__half)), pl_b = up256((size_t)M * K * sizeof(__half));
    const size_t need = wq_b + pl_b + up256((size_t)M * sizeof(float));
    if (need > g_scratch_bytes) {
        if (g_scratch) cudaFree(g_scratch);
        g_scratch = nullptr;
        g_scratch_bytes = 0;
        if (cudaMalloc(&g_scratch, need) != cudaSuccess) {
            fprintf(stderr, "%s: scratch of %zu bytes refused -- refusing\n", tag, need);
            return -1;
        }
        g_scratch_bytes = need;
    }
    if (!g_blas && cublasCreate(&g_blas) != CUBLAS_STATUS_SUCCESS) {
        g_blas = nullptr;
        fprintf(stderr, "%s: cuBLAS unavailable -- refusing\n", tag);
        return -1;
    }
    __half *wq = reinterpret_cast<__half *>(g_scratch);
    __half *xhi = reinterpret_cast<__half *>(g_scratch + wq_b);
    float *inv = reinterpret_cast<float *>(g_scratch + wq_b + pl_b);
    const uint32_t *tr = static_cast<const uint32_t *>(w);
    const __half *suh = reinterpret_cast<const __half *>(static_cast<const uint8_t *>(w) + trellis_bytes);
    const __half *svh = suh + K;

    const int64_t tiles = (int64_t)(K >> 4) * (N >> 4);
    const unsigned dgrid = (unsigned)((tiles + 7) / 8);
    switch (k2) {
    case 4:  dense_decode_kernel<4><<<dgrid, 256, 0, stream>>>(tr, wq, K, N); break;
    case 6:  dense_decode_kernel<6><<<dgrid, 256, 0, stream>>>(tr, wq, K, N); break;
    case 8:  dense_decode_kernel<8><<<dgrid, 256, 0, stream>>>(tr, wq, K, N); break;
    case 10: dense_decode_kernel<10><<<dgrid, 256, 0, stream>>>(tr, wq, K, N); break;
    default:
        fprintf(stderr, "%s: rate k2=%d has no instance (4, 6, 8, 10) -- refusing\n", tag, k2);
        return -1;
    }
    dense_prep_kernel<<<M, 256, (size_t)K * sizeof(float), stream>>>(static_cast<const __nv_bfloat16 *>(x_bf16), suh,
                                                                       xhi, inv, K);
    /* column-major view: Y^T (N x M, ld N) = W_q^T (N x K, ld N) . X^T (K x M, ld K) */
    const float one = 1.0f, zero = 0.0f;
    cublasSetStream(g_blas, stream);
    cublasStatus_t st = cublasGemmEx(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, wq, CUDA_R_16F, N, xhi,
                                     CUDA_R_16F, K, &zero, y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cuBLAS GEMM failed (%d) -- refusing\n", tag, (int)st);
        return -3;
    }
    const int64_t etasks = (int64_t)M * (N >> 7);
    dense_epilogue_kernel<<<(unsigned)((etasks + 7) / 8), 256, 0, stream>>>(y, inv, svh, M, N);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

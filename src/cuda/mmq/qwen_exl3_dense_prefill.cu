// SPDX-License-Identifier: MIT
/** @file
 *  The dense EXL3 arm for prompt chunks: decode W once to fp16, then one fixed-algorithm cuBLASLt GEMM. */
// The dense EXL3 arm for PROMPT CHUNKS (L251): every prompt chunk of ds4_exl3_dense_launch (decode rows keep the GEMV).
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
//   3. GEMM    one cuBLASLt tensor-core GEMM, Z = X W_q, f32 accumulate and output -- a row's bytes
//              independent of how many rows share the call (L266): the fixed-algorithm GEMM
//              (pulsar_cuda_lt_fixed.h, L321: the contract now shared with the f16 rows arm) -- the
//              reduction scheme pinned to NONE and ONE algorithm per (K, N), chosen at the production
//              prefill chunk's row count and checked to support each other row count.
//              cublasGemmEx(CUBLAS_GEMM_DEFAULT) sat here and picked a kernel (and split-K) by M, so a
//              prompt cut at G got different bytes from the same prompt prefilled whole --
//              the chunk-neutrality gate (now tests/session_contract_gate) measured 1-2 nats of logit difference.  DeepSeek's bf16
//              GEMM met the same fact in L183 (pulsar_cuda_matmul.cu bf16_lt_matmul).
//   4. EPILOGUE per row: times 2^-P, the output H128 per 128-block, times svh.
//
// A prefilled row and a decoded row agree to rounding, not to the bit (Tyler 2026-09-29, "Don't use
// a decode kernel for prefill"); graded against double by tests/exl3_dense_gate at every rate.
// The scratch (W_q, the planes, the row scales) is one device buffer grown to the largest call.

#include "ds4_exl3_dev.cuh"
#include "engine/exl3_trellis.h"

#include <cstdio>
#include "cuda/pulsar_cuda_lt_fixed.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kHiExp = 14;   ///< the row's largest |value| lands just under 2^14 (the GEMV's prescale)

/// 2^e as an f32 built from its exponent bits.
/// @param e  the exponent, in [-126, 127]
/// @return   2^e
__device__ __forceinline__ float pow2f(int e) { return __int_as_float((127 + e) << 23); }

/* 1. one warp per 16 x 16 tile (kt, nt): lane t holds positions 8t..8t+7 -- k rows (t%4)*2 + {0,1,8,9}
 * of columns t/4 (positions 8t..8t+3) and t/4 + 8 (8t+4..8t+7), the mma B-fragment order. */
/// DECODE at rate K2: the trellis into fp16 W_q.  Grid ceil(K/16 x N/16 / 8), 256 threads: one warp per 16 x 16
/// tile, tiles in the trellis's (kt, nt) order.
/// @param trellis  the weight's trellis words, device
/// @param wq       out: W_q fp16 [K][N], row-major
/// @param K        input width (a multiple of 128)
/// @param N        stored output width (a multiple of 128)
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
/// PREP: x * suh, H128 per block, one power-of-two prescale per row, one fp16 plane.  Grid M, 256 threads,
/// K * 4 bytes of dynamic smem (the rotated row; the launcher opts in past 48 KiB).
/// @param x    bf16 input rows [M][K] (8-byte aligned rows)
/// @param suh  the input sign/scale vector, fp16 [K]
/// @param xhi  out: the fp16 operand plane [M][K]
/// @param inv  out: each row's inverse prescale 2^-P, f32 [M]
/// @param K    input width (a multiple of 128)
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

/* 4. one warp per (row, 128-block of outputs), in place on y (row width N, the logical outputs).  The blocks past
 * n_main -- one, the stored width's last when N is not a 128 multiple (exl3_out_stored, L295 F7) -- come from `tail`
 * [M][128] instead, and only their columns below N are written. */
/// EPILOGUE: times 2^-P, the output H128 and svh, in place on y.  Grid ceil(M x stored N / 128 / 8), 256 threads.
/// @param y       f32 [M][N], the GEMM's columns below n_main; rewritten finished
/// @param tail    the last stored 128-block's GEMM output f32 [M][128] when N is not a 128 multiple, else NULL
/// @param inv     each row's inverse prescale [M]
/// @param svh     the output sign/scale vector, fp16 [stored N]
/// @param M       rows
/// @param N       logical output width (y's row)
/// @param n_main  N rounded down to 128: the columns read from y
__global__ void __launch_bounds__(256)
dense_epilogue_kernel(float *__restrict__ y, const float *__restrict__ tail, const float *__restrict__ inv,
                      const __half *__restrict__ svh, int M, int N, int n_main) {
    const int64_t task = (int64_t)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int n_blk = (int)(exl3_out_stored((uint64_t)N) >> 7);
    if (task >= (int64_t)M * n_blk) return;
    const int m = (int)(task / n_blk), blk = (int)(task % n_blk), lane = threadIdx.x & 31, n0 = blk * 128 + lane * 4;
    float4 *p = reinterpret_cast<float4 *>(y + (size_t)m * N + n0);
    const float4 q = n0 < n_main ? *p : *reinterpret_cast<const float4 *>(tail + (size_t)m * 128 + (n0 - n_main));
    const float s = inv[m];
    float z[4] = {q.x * s, q.y * s, q.z * s, q.w * s};
    exl3dev::had128(z);
    const __half2 v01 = *reinterpret_cast<const __half2 *>(svh + n0), v23 = *reinterpret_cast<const __half2 *>(svh + n0 + 2);
    if (n0 < N)
        *p = make_float4(z[0] * __low2float(v01), z[1] * __high2float(v01), z[2] * __low2float(v23), z[3] * __high2float(v23));
}

/// The arm's one device scratch (W_q, the plane, the row scales, the tail), grown to the largest call; never freed.
uint8_t *g_scratch = nullptr;
/// Bytes of g_scratch.
size_t g_scratch_bytes = 0;

/// @param b  bytes
/// @return   b rounded up to a multiple of 256
size_t up256(size_t b) { return (b + 255u) & ~(size_t)255u; }

} // namespace

int qwen_exl3_dense_prefill_launch(const void *w, int k2, const void *x_bf16, float *y, int M, int K, int Ny,
                                   uint64_t trellis_bytes, cudaStream_t stream) {
    const char *tag = "qwen_exl3_dense_prefill_launch";
    if (!w || !x_bf16 || !y || M <= 0 || Ny <= 0 || Ny % 4 || K % EXL3_HAD_BLOCK) {
        fprintf(stderr, "%s: bad arguments (M=%d K=%d N=%d) -- refusing\n", tag, M, K, Ny);
        return -1;
    }
    /* N: the stored width the trellis and svh carry; Ny: the logical one, y's row (L295 F7) */
    const int N = (int)exl3_out_stored((uint64_t)Ny), n_main = Ny / 128 * 128;
    const size_t wq_b = up256((size_t)K * N * sizeof(__half)), pl_b = up256((size_t)M * K * sizeof(__half));
    const size_t tail_b = Ny == N ? 0 : up256((size_t)M * 128 * sizeof(float));
    const size_t need = wq_b + pl_b + up256((size_t)M * sizeof(float)) + tail_b;
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
    __half *wq = reinterpret_cast<__half *>(g_scratch);
    __half *xhi = reinterpret_cast<__half *>(g_scratch + wq_b);
    float *inv = reinterpret_cast<float *>(g_scratch + wq_b + pl_b);
    float *tail = Ny != N ? reinterpret_cast<float *>(g_scratch + wq_b + pl_b + up256((size_t)M * sizeof(float))) : nullptr;
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
    case 12: dense_decode_kernel<12><<<dgrid, 256, 0, stream>>>(tr, wq, K, N); break;
    case 16: dense_decode_kernel<16><<<dgrid, 256, 0, stream>>>(tr, wq, K, N); break;
    default:
        fprintf(stderr, "%s: rate k2=%d has no instance (K = 2, 3, 4, 5, 6, 8) -- refusing\n", tag, k2);
        return -1;
    }
    /* the prep holds one row's K f32 in shared memory: past the 48 KiB a launch gets by default (K > 12288 -- L295:
     * Inkling's dense down, K 16384) the kernel opts in once to the larger carve-out.  The arithmetic is unchanged. */
    const size_t prep_smem = (size_t)K * sizeof(float);
    static size_t prep_smem_set = 48u * 1024u;
    if (prep_smem > prep_smem_set) {
        if (cudaFuncSetAttribute(dense_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)prep_smem) !=
            cudaSuccess) {
            fprintf(stderr, "%s: the prep's %zu B of shared memory (K=%d) is refused by the device -- refusing\n", tag,
                    prep_smem, K);
            (void)cudaGetLastError();
            return -1;
        }
        prep_smem_set = prep_smem;
    }
    dense_prep_kernel<<<M, 256, prep_smem, stream>>>(static_cast<const __nv_bfloat16 *>(x_bf16), suh, xhi, inv, K);
    /* the whole GEMM, or (a logical width short of the stored one) its 128-multiple head into y and its last stored
     * block into the tail */
    if (n_main) {
        const int rc = pulsar_lt_fixed_gemm(PULSAR_LT_A_KN, PULSAR_LT_PROMPT, wq, K, N, 0, n_main, xhi, M, y, Ny, stream,
                                            tag);
        if (rc) return rc;
    }
    if (Ny != N) {
        const int rc = pulsar_lt_fixed_gemm(PULSAR_LT_A_KN, PULSAR_LT_PROMPT, wq, K, N, n_main, 128, xhi, M, tail, 128,
                                            stream, tag);
        if (rc) return rc;
    }
    const int64_t etasks = (int64_t)M * (N >> 7);
    dense_epilogue_kernel<<<(unsigned)((etasks + 7) / 8), 256, 0, stream>>>(y, tail, inv, svh, M, Ny, n_main);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

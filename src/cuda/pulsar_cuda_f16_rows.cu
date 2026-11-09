/** @file
 *  The f16 rows arm (L321): prep, fixed-algorithm cuBLASLt GEMM, unscale. */
/* The f16 rows arm (L321) -- see pulsar_cuda_f16_rows.h for the contract and the arithmetic. */
#include "pulsar_cuda_f16_rows.h"
#include "pulsar_cuda_lt_fixed.h"

#include "pulsar_gpu.h"   /* PULSAR_GPU_MNEUTRAL_ROWS_MAX */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>

namespace {

/// Threads per CTA of both kernels.
constexpr int kThreads = 256;
constexpr int kHiExp = 14;   ///< the row's largest |value| lands just under 2^14 (the EXL3 prompt arm's prescale)

/// @param b a byte count.
/// @return b rounded up to a multiple of 256.
size_t up256(size_t b) { return (b + 255u) & ~(size_t)255u; }

/// @param p an exponent, -126 .. 127 (callers clamp to +-100).
/// @return 2^p, built from the exponent bits.
__device__ __forceinline__ float pow2(int p) { return __int_as_float((p + 127) << 23); }

/// @param v an element.
/// @return v as f32.
__device__ __forceinline__ float as_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
/// @param v an element.
/// @return v.
__device__ __forceinline__ float as_f32(float v) { return v; }
/* L338: the source's element type (bf16 rows, or the runtime LoRA's f32 intermediate) and its row pitch are
 * parameters; the arithmetic is the one the arm had */
/* one CTA per row: the row's amax, then the scaled f16 plane and the row's inverse scale */
/// Grid M (one CTA per row), kThreads threads: xh[m] = f16(x[m] * 2^p), p putting the row's largest |value| just
/// under 2^kHiExp; inv[m] = 2^-p.
/// @tparam T the source element (bf16 or f32).
/// @param x the rows [M][ld_x].
/// @param ld_x their pitch.
/// @param xh out: the scaled rows, f16 [M][K].
/// @param inv out: each row's inverse scale, f32 [M].
/// @param K row width.
template <class T>
__global__ void __launch_bounds__(kThreads)
prep_kernel(const T *__restrict__ x, size_t ld_x, __half *__restrict__ xh, float *__restrict__ inv, int K) {
    __shared__ float s_amax[kThreads / 32];
    const int m = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const T *xr = x + (size_t)m * ld_x;
    float amax = 0.0f;
    for (int k = threadIdx.x; k < K; k += kThreads) amax = fmaxf(amax, fabsf(as_f32(xr[k])));
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
    for (int k = threadIdx.x; k < K; k += kThreads) xo[k] = __float2half_rn(as_f32(xr[k]) * up);
    if (threadIdx.x == 0) inv[m] = pow2(-p);
}

/* y[m][n] *= inv[m] (an exact power of two) */
/// Grid ceil(M N / kThreads), kThreads threads.
/// @param y the GEMM output, f32 [M][N], scaled in place.
/// @param inv each row's inverse scale, [M].
/// @param M rows.
/// @param N columns.
__global__ void __launch_bounds__(kThreads)
unscale_kernel(float *__restrict__ y, const float *__restrict__ inv, int M, int N) {
    const size_t i = (size_t)blockIdx.x * kThreads + threadIdx.x;
    if (i < (size_t)M * N) y[i] *= inv[i / (size_t)N];
}

/* L338: y[t][o] += (scale * inv1[t] * inv2[t]) * u[t][o] on the marked rows */
/// Grid (ceil(out / kThreads), M), kThreads threads.
/// @param y     the linear's f32 rows (ld_y apart), accumulated into.
/// @param ld_y  their pitch.
/// @param u     the LoRA's unscaled output, f32 [M][out].
/// @param out   the output width.
/// @param scale alpha / r.
/// @param inv1  the input prep's inverse scales [M].
/// @param inv2  the intermediate prep's [M].
/// @param rows  the marked rows, u8 [M], or NULL (every row).
__global__ void __launch_bounds__(kThreads)
lora_add_kernel(float *__restrict__ y, size_t ld_y, const float *__restrict__ u, int out, float scale,
                const float *__restrict__ inv1, const float *__restrict__ inv2, const uint8_t *__restrict__ rows) {
    const int t = blockIdx.y, o = blockIdx.x * kThreads + threadIdx.x;
    if (o >= out || (rows && !rows[t])) return;
    y[(size_t)t * ld_y + o] += (scale * (inv1[t] * inv2[t])) * u[(size_t)t * out + o];
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
    prep_kernel<<<rows, kThreads, 0, stream>>>(reinterpret_cast<const __nv_bfloat16 *>(x_bf16), (size_t)in, xh, inv, in);
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

int pulsar_f16_rows_linear_h_launch(const void *w, const uint16_t *x_f16, int rows, int in, int out, float *y,
                                    bool prompt, cudaStream_t stream) {
    const char *tag = "pulsar: f16 rows linear (f16 operand)";
    if (!w || !x_f16 || !y || rows <= 0 || in <= 0 || out <= 0 || in % 8 || out % 8 || ((uintptr_t)w & 15u) ||
        ((uintptr_t)x_f16 & 15u)) {
        fprintf(stderr, "%s: %d rows %d -> %d, a null / misaligned weight or operand, or a width not a multiple of 8 "
                        "-- refusing\n", tag, rows, in, out);
        return -1;
    }
    const int blk = prompt ? rows : (int)PULSAR_GPU_MNEUTRAL_ROWS_MAX;
    for (int r0 = 0; r0 < rows; r0 += blk) {
        const int m = rows - r0 < blk ? rows - r0 : blk;
        const int rc = pulsar_lt_fixed_gemm(PULSAR_LT_A_NK, prompt ? PULSAR_LT_PROMPT : PULSAR_LT_DECODE, w, in, out, 0,
                                            out, x_f16 + (size_t)r0 * in, m, y + (size_t)r0 * out, out, stream, tag);
        if (rc) return rc;
    }
    return 0;
}

size_t pulsar_rows_lora_workspace_bytes(int rows, int in, int r, int out) {
    if (rows <= 0 || in <= 0 || r <= 0 || out <= 0) return 0;
    const size_t R = (size_t)(rows < PULSAR_ROWS_LORA_BLOCK ? rows : PULSAR_ROWS_LORA_BLOCK);
    return up256(R * in * sizeof(__half)) + up256(R * sizeof(float)) + up256(R * r * sizeof(float)) +
           up256(R * r * sizeof(__half)) + up256(R * sizeof(float)) + up256(R * out * sizeof(float));
}

int pulsar_rows_lora_launch(const uint16_t *x, uint64_t ld_x, int n_rows, const pulsar_rows_lora *l, float *y,
                            uint64_t ld_y, const uint8_t *rows, bool prompt, void *ws, size_t ws_bytes,
                            cudaStream_t stream) {
    const char *tag = "pulsar: rows lora";
    if (!x || !l || !l->A || !l->B || !y || !ws || n_rows <= 0 || l->r == 0 || l->in == 0 || l->out == 0 ||
        l->r % 8 || l->in % 8 || l->out % 8 || l->in > ld_x || l->out > ld_y || ((uintptr_t)l->A & 15u) ||
        ((uintptr_t)l->B & 15u) || !isfinite(l->scale) ||
        ws_bytes < pulsar_rows_lora_workspace_bytes(n_rows, (int)l->in, (int)l->r, (int)l->out)) {
        fprintf(stderr, "%s: %d rows, in %u <= ld_x %llu, rank %u, out %u <= ld_y %llu (each a multiple of 8), 16-byte "
                        "aligned f16 pairs, a finite scale and a workspace of %zu B (have %zu) -- refusing\n", tag, n_rows,
                l ? l->in : 0u, (unsigned long long)ld_x, l ? l->r : 0u, l ? l->out : 0u, (unsigned long long)ld_y,
                l ? pulsar_rows_lora_workspace_bytes(n_rows, (int)l->in, (int)l->r, (int)l->out) : (size_t)0, ws_bytes);
        return -1;
    }
    const int in = (int)l->in, r = (int)l->r, out = (int)l->out, R = n_rows < PULSAR_ROWS_LORA_BLOCK ? n_rows : PULSAR_ROWS_LORA_BLOCK;
    uint8_t *b = static_cast<uint8_t *>(ws);
    __half *xh = reinterpret_cast<__half *>(b);           b += up256((size_t)R * in * sizeof(__half));
    float *inv1 = reinterpret_cast<float *>(b);           b += up256((size_t)R * sizeof(float));
    float *t = reinterpret_cast<float *>(b);              b += up256((size_t)R * r * sizeof(float));
    __half *th = reinterpret_cast<__half *>(b);           b += up256((size_t)R * r * sizeof(__half));
    float *inv2 = reinterpret_cast<float *>(b);           b += up256((size_t)R * sizeof(float));
    float *u = reinterpret_cast<float *>(b);
    const pulsar_lt_rows cls = prompt ? PULSAR_LT_PROMPT : PULSAR_LT_DECODE;
    for (int b0 = 0; b0 < n_rows; b0 += R) {
        const int M = n_rows - b0 < R ? n_rows - b0 : R;
        const int blk = prompt ? M : (int)PULSAR_GPU_MNEUTRAL_ROWS_MAX;
        prep_kernel<<<M, kThreads, 0, stream>>>(reinterpret_cast<const __nv_bfloat16 *>(x) + (size_t)b0 * ld_x,
                                                (size_t)ld_x, xh, inv1, in);
        if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "%s: prep launch failed\n", tag); return -3; }
        for (int r0 = 0; r0 < M; r0 += blk) {
            const int m = M - r0 < blk ? M - r0 : blk;
            const int rc = pulsar_lt_fixed_gemm(PULSAR_LT_A_NK, cls, l->A, in, r, 0, r, xh + (size_t)r0 * in, m,
                                                t + (size_t)r0 * r, r, stream, tag);
            if (rc) return rc;
        }
        prep_kernel<<<M, kThreads, 0, stream>>>(t, (size_t)r, th, inv2, r);
        if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "%s: intermediate prep launch failed\n", tag); return -3; }
        for (int r0 = 0; r0 < M; r0 += blk) {
            const int m = M - r0 < blk ? M - r0 : blk;
            const int rc = pulsar_lt_fixed_gemm(PULSAR_LT_A_NK, cls, l->B, r, out, 0, out, th + (size_t)r0 * r, m,
                                                u + (size_t)r0 * out, out, stream, tag);
            if (rc) return rc;
        }
        lora_add_kernel<<<dim3((unsigned)((out + kThreads - 1) / kThreads), (unsigned)M), kThreads, 0, stream>>>(
            y + (size_t)b0 * ld_y, (size_t)ld_y, u, out, l->scale, inv1, inv2, rows ? rows + b0 : nullptr);
        if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "%s: add launch failed\n", tag); return -3; }
    }
    return 0;
}

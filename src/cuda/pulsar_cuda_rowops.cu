/** @file
 *  Row operations between a layer plan's blocks: residual add, RMSNorm, LayerNorm, emits and gathers. */
/* Row operations between a layer plan's blocks (L295): contracts in pulsar_cuda_rowops.h. */
#include "pulsar_cuda_rowops.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

namespace {

/// Block size of the one-CTA-a-row kernels (rows_norm_kernel, rows_ln32_kernel; 8 warps).
constexpr int kNormThreads = 256;
/// Block size of the element-wise emit / gather kernels.
constexpr int kEmitThreads = 256;

/// bf16 bits (low 16 of b) -> f32.
/// @param b bf16 bits in the low half (high half zero)
/// @return the value as f32
__device__ __forceinline__ float bf2f(uint32_t b) { return __uint_as_float(b << 16); }
/// f32 -> bf16 bits, round to nearest even.
/// @param f the value
/// @return its bf16 bits in the low half
__device__ __forceinline__ uint32_t f2bf(float f) { return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(f)); }

/* One CTA a row.  Thread i owns the 8-element chunks i, i + 256, ... of the row.  With a delta, the thread first
 * writes its chunks' rounded sums back to the residual (and reads them from registers for the squares, so the norm
 * is of the stored bytes); its squares are an f32 fma chain over its chunks in order, the warps fold by
 * xor-shuffle, warp 0 adds the eight warp sums in order. */
/** Optional residual add then optional RMS norm, bf16.  Grid n_rows (blockIdx.x = row); block kNormThreads; static
 *  smem only.  With delta: res[t] = bf16(res[t] + delta[t]) stored.  With w:
 *  y[t] = bf16((res[t] * rstd) * w) of the stored row.
 *  @param res      device bf16 rows (the residual; updated in place with delta), 16-byte aligned
 *  @param ld_res   res row pitch in elements (a multiple of 8)
 *  @param delta    device bf16 rows to add, or NULL
 *  @param ld_delta delta row pitch in elements
 *  @param width    row width (a multiple of 8)
 *  @param w        device bf16 [width] norm weight as uint4 chunks, or NULL: the add alone
 *  @param eps      the norm's epsilon
 *  @param y        device bf16 output rows (read only with w)
 *  @param ld_y     y row pitch in elements */
__global__ void __launch_bounds__(kNormThreads)
rows_norm_kernel(uint16_t *__restrict__ res, uint64_t ld_res, const uint16_t *__restrict__ delta, uint64_t ld_delta,
                 uint32_t width, const uint4 *__restrict__ w, float eps, uint16_t *__restrict__ y, uint64_t ld_y) {
    __shared__ float warp_ss[kNormThreads / 32];
    __shared__ float s_rstd;
    const uint64_t t = blockIdx.x;
    const uint32_t chunks = width / 8;
    uint4 *x4 = (uint4 *)(res + t * ld_res);
    const uint4 *d4 = delta ? (const uint4 *)(delta + t * ld_delta) : nullptr;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < chunks; i += kNormThreads) {
        uint4 v = x4[i];
        uint32_t u[4] = {v.x, v.y, v.z, v.w};
        if (d4) {
            const uint4 dv = d4[i];
            const uint32_t du[4] = {dv.x, dv.y, dv.z, dv.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t lo = f2bf(__fadd_rn(bf2f(u[k] & 0xffffu), bf2f(du[k] & 0xffffu)));
                const uint32_t hi = f2bf(__fadd_rn(bf2f(u[k] >> 16), bf2f(du[k] >> 16)));
                u[k] = lo | (hi << 16);
            }
            v = make_uint4(u[0], u[1], u[2], u[3]);
            x4[i] = v;
        }
        if (w) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float a = bf2f(u[k] & 0xffffu), b = bf2f(u[k] >> 16);
                ss = __fmaf_rn(a, a, ss);
                ss = __fmaf_rn(b, b, ss);
            }
        }
    }
    if (!w) return;   /* an add alone */
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
    if ((threadIdx.x & 31) == 0) warp_ss[threadIdx.x / 32] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int k = 0; k < kNormThreads / 32; ++k) tot = __fadd_rn(tot, warp_ss[k]);
        s_rstd = rsqrtf(__fadd_rn(__fdiv_rn(tot, (float)width), eps));
    }
    __syncthreads();
    const float rstd = s_rstd;
    uint4 *y4 = (uint4 *)(y + t * ld_y);
    for (uint32_t i = threadIdx.x; i < chunks; i += kNormThreads) {
        const uint4 v = x4[i], wv = w[i];   /* this thread's own chunks: its writes above are visible to it */
        const uint32_t u[4] = {v.x, v.y, v.z, v.w}, wu[4] = {wv.x, wv.y, wv.z, wv.w};
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float lo = __fmul_rn(__fmul_rn(bf2f(u[k] & 0xffffu), rstd), bf2f(wu[k] & 0xffffu));
            const float hi = __fmul_rn(__fmul_rn(bf2f(u[k] >> 16), rstd), bf2f(wu[k] >> 16));
            o[k] = f2bf(lo) | (f2bf(hi) << 16);
        }
        y4[i] = make_uint4(o[0], o[1], o[2], o[3]);
    }
}

/* Four elements a thread: x f32 -> bf16, times the scalar when there is one. */
/** f32 rows -> bf16, times an optional device scalar.  Grid (ceil(width / (4 * kEmitThreads)), rows); block
 *  kEmitThreads; four columns a thread (width a multiple of 4, 16-byte aligned f32 rows).
 *  @param x     device f32 rows, row t at x + t * ld_x
 *  @param ld_x  x row pitch in elements
 *  @param width columns
 *  @param scale device pointer to one scalar stored as `kind`, or NULL with PULSAR_ROWS_SCALE_NONE
 *  @param kind  pulsar_rows_scale value
 *  @param y     device bf16 output rows
 *  @param ld_y  y row pitch in elements */
__global__ void __launch_bounds__(kEmitThreads)
rows_emit_kernel(const float *__restrict__ x, uint64_t ld_x, uint32_t width, const void *__restrict__ scale, int kind,
                 uint16_t *__restrict__ y, uint64_t ld_y) {
    const uint64_t t = blockIdx.y;
    const uint32_t c = (blockIdx.x * kEmitThreads + threadIdx.x) * 4u;
    if (c >= width) return;
    float s = 1.0f;
    if (kind == PULSAR_ROWS_SCALE_F32) s = *(const float *)scale;
    else if (kind == PULSAR_ROWS_SCALE_BF16) s = bf2f(*(const uint16_t *)scale);
    const float4 v = *(const float4 *)(x + t * ld_x + c);
    const float f[4] = {v.x, v.y, v.z, v.w};
    uint32_t o[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) o[k] = f2bf(kind == PULSAR_ROWS_SCALE_NONE ? f[k] : __fmul_rn(f[k], s));
    *(uint2 *)(y + t * ld_y + c) = make_uint2(o[0] | (o[1] << 16), o[2] | (o[3] << 16));
}

/* L321: a block-wide f32 sum -- the warps fold by xor-shuffle, warp 0 adds the warp sums in order. */
/** Block-wide f32 sum over kNormThreads threads (xor-shuffle per warp, warp 0 adds the warp sums in order).
 *  Every thread of the block must call it (it syncs twice).
 *  @param v      this thread's partial
 *  @param warp_s shared scratch [kNormThreads / 32]
 *  @param out    shared slot the total is published through
 *  @return the block total, in every thread */
__device__ float ln_block_sum(float v, float *warp_s, float *out) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) warp_s[threadIdx.x / 32] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int k = 0; k < kNormThreads / 32; ++k) tot = __fadd_rn(tot, warp_s[k]);
        *out = tot;
    }
    __syncthreads();
    return *out;
}

/* the 16-bit row element: bf16 or f16 (pulsar_rows_out), round to nearest even */
/// f32 -> the 16-bit output element (round to nearest even).
/// @param f   the value
/// @param f16 nonzero: f16, else bf16
/// @return the element bits in the low half
__device__ __forceinline__ uint32_t f2out(float f, int f16) {
    return f16 ? (uint32_t)__half_as_ushort(__float2half_rn(f)) : f2bf(f);
}

/* Four elements a thread: act(x + bias) as bf16 / f16. */
/** act(x + bias) f32 -> bf16 / f16.  Grid (ceil(width / (4 * kEmitThreads)), rows); block kEmitThreads; four columns
 *  a thread.
 *  @param x     device f32 rows, row t at x + t * ld_x
 *  @param ld_x  x row pitch in elements
 *  @param width columns
 *  @param bias  device f32 [width], or NULL
 *  @param act   pulsar_rows_act value (NONE or exact-erf GELU)
 *  @param y     device 16-bit output rows
 *  @param ld_y  y row pitch in elements
 *  @param f16   nonzero: f16 output, else bf16 */
__global__ void __launch_bounds__(kEmitThreads)
rows_bias_emit_kernel(const float *__restrict__ x, uint64_t ld_x, uint32_t width, const float *__restrict__ bias,
                      int act, uint16_t *__restrict__ y, uint64_t ld_y, int f16) {
    const uint64_t t = blockIdx.y;
    const uint32_t c = (blockIdx.x * kEmitThreads + threadIdx.x) * 4u;
    if (c >= width) return;
    const float4 v = *(const float4 *)(x + t * ld_x + c);
    const float f[4] = {v.x, v.y, v.z, v.w};
    uint32_t o[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        float z = bias ? __fadd_rn(f[k], bias[c + k]) : f[k];
        if (act == PULSAR_ROWS_ACT_GELU) z = __fmul_rn(__fmul_rn(0.5f, z), __fadd_rn(1.0f, erff(__fmul_rn(z, 0.70710678118654752f))));
        else if (act == PULSAR_ROWS_ACT_RELU2) { z = fmaxf(z, 0.0f); z = __fmul_rn(z, z); }
        o[k] = f2out(z, f16);
    }
    *(uint2 *)(y + t * ld_y + c) = make_uint2(o[0] | (o[1] << 16), o[2] | (o[3] << 16));
}

/* L321: the f32-residual LayerNorm, one CTA a row, thread i owning columns i, i + 256, ... (f32 or bf16 input): with
 * x, res += x + bias first (f32, stored); without x, an f32 row read plus `bias` (not stored: a linear's output and its
 * bias); then the sum, the mean, the centred squares -- f32 chains per thread in column order, folded as ln_block_sum
 * -- and the outputs (bf16 / f16 and / or f32). */
/** LayerNorm with an f32 residual stream (or a bf16 input row).  Grid n_rows; block kNormThreads; static smem only.
 *  IN_F32 with x: res += x + bias, stored; IN_F32 without x: the row read plus bias, not stored; !IN_F32: the bf16 row
 *  as is.  Then, when w is set, out = (u - mean) * rstd * w + b to y (16-bit) and / or yf (f32).
 *  @tparam IN_F32 res_ holds f32 rows (else bf16)
 *  @param res_   device residual rows (f32 or bf16 by IN_F32)
 *  @param ld_res res_ row pitch in elements
 *  @param x      device f32 rows added to the residual, or NULL
 *  @param ld_x   x row pitch in elements
 *  @param bias   device f32 [width], or NULL
 *  @param width  columns
 *  @param w      device f32 [width] LayerNorm weight, or NULL: the add alone
 *  @param b      device f32 [width] LayerNorm bias (read with w)
 *  @param eps    the norm's epsilon
 *  @param y      device 16-bit output rows, or NULL
 *  @param ld_y   y row pitch in elements
 *  @param f16    nonzero: y is f16, else bf16
 *  @param yf     device f32 output rows, or NULL
 *  @param ld_yf  yf row pitch in elements */
template <bool IN_F32>
__global__ void __launch_bounds__(kNormThreads)
rows_ln32_kernel(void *__restrict__ res_, uint64_t ld_res, const float *__restrict__ x, uint64_t ld_x,
                 const float *__restrict__ bias, uint32_t width, const float *__restrict__ w,
                 const float *__restrict__ b, float eps, uint16_t *__restrict__ y, uint64_t ld_y, int f16,
                 float *__restrict__ yf, uint64_t ld_yf) {
    __shared__ float warp_s[kNormThreads / 32];
    __shared__ float s_sum, s_sq;
    const uint64_t t = blockIdx.x;
    float *rf = IN_F32 ? (float *)res_ + t * ld_res : nullptr;
    const uint16_t *rb = IN_F32 ? nullptr : (const uint16_t *)res_ + t * ld_res;
    float sum = 0.0f;
    for (uint32_t c = threadIdx.x; c < width; c += kNormThreads) {
        float v;
        if constexpr (IN_F32) {
            v = rf[c];
            if (x) {
                v = __fadd_rn(v, __fadd_rn(x[t * ld_x + c], bias ? bias[c] : 0.0f));
                rf[c] = v;
            } else if (bias) {
                v = __fadd_rn(v, bias[c]);
            }
        } else {
            v = bf2f(rb[c]);
        }
        sum = __fadd_rn(sum, v);
    }
    if (!w) return;   /* an add alone */
    const float mean = __fdiv_rn(ln_block_sum(sum, warp_s, &s_sum), (float)width);
    float sq = 0.0f;
    const float *rbias = IN_F32 && !x ? bias : nullptr;   /* the unstored bias, added again where the row is read */
    for (uint32_t c = threadIdx.x; c < width; c += kNormThreads) {   /* this thread's own columns: its writes visible */
        const float a = __fsub_rn(IN_F32 ? (rbias ? __fadd_rn(rf[c], rbias[c]) : rf[c]) : bf2f(rb[c]), mean);
        sq = __fmaf_rn(a, a, sq);
    }
    const float rstd = rsqrtf(__fadd_rn(__fdiv_rn(ln_block_sum(sq, warp_s, &s_sq), (float)width), eps));
    for (uint32_t c = threadIdx.x; c < width; c += kNormThreads) {
        const float u = IN_F32 ? (rbias ? __fadd_rn(rf[c], rbias[c]) : rf[c]) : bf2f(rb[c]);
        const float v = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(u, mean), rstd), w[c]), b[c]);
        if (y) y[t * ld_y + c] = (uint16_t)f2out(v, f16);
        if (yf) yf[t * ld_yf + c] = v;
    }
}

/* Eight elements a thread: one f16 table row, scaled, as f32 (a NaN row for an id outside the table). */
/** Embedding gather from an f16 table, scaled, to f32.  Grid (ceil(width / (8 * kEmitThreads)), rows); block
 *  kEmitThreads; eight columns a thread.  An id outside [0, n_vocab) writes a NaN row.
 *  @param table   device f16 [n_vocab][width] as uint4 chunks
 *  @param n_vocab rows in the table
 *  @param width   row width (a multiple of 8)
 *  @param ids     device token ids, one per output row
 *  @param scale   multiplies every element
 *  @param y       device f32 output rows (16-byte aligned)
 *  @param ld_y    y row pitch in elements */
__global__ void __launch_bounds__(kEmitThreads)
rows_gather_f16_kernel(const uint4 *__restrict__ table, uint32_t n_vocab, uint32_t width, const int32_t *__restrict__ ids,
                       float scale, float *__restrict__ y, uint64_t ld_y) {
    const uint64_t t = blockIdx.y;
    const uint32_t c8 = blockIdx.x * kEmitThreads + threadIdx.x;
    if (8u * c8 >= width) return;
    const int32_t id = ids[t];
    float o[8];
    if (id < 0 || (uint32_t)id >= n_vocab) {
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = __int_as_float(0x7fc00000);
    } else {
        const uint4 v = table[(uint64_t)id * (width / 8u) + c8];
        const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            o[2 * k] = __fmul_rn(__half2float(__ushort_as_half((unsigned short)(u[k] & 0xffffu))), scale);
            o[2 * k + 1] = __fmul_rn(__half2float(__ushort_as_half((unsigned short)(u[k] >> 16))), scale);
        }
    }
    float4 *y4 = (float4 *)(y + t * ld_y + 8u * c8);
    y4[0] = make_float4(o[0], o[1], o[2], o[3]);
    y4[1] = make_float4(o[4], o[5], o[6], o[7]);
}

/** Checks the last launch.
 *  @param who the launch's name for the message
 *  @return 0, or -3 after naming the CUDA error on stderr */
int launched(const char *who) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: rows %s: launch failed: %s\n", who, cudaGetErrorString(e));
        return -3;
    }
    return 0;
}

/// @param p a pointer
/// @return true when p is 16-byte aligned
bool aligned16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

/* the shared contract of the norm and the add: widths and strides in 8s, aligned rows, a grid of n_rows CTAs */
/** Validates the norm / add contract (rows in 1 .. 2^31 - 1, width and pitches multiples of 8, 16-byte aligned
 *  rows; y required with w); names the refusal on stderr.
 *  @param who      the caller's name for the message
 *  @param res      residual / input rows
 *  @param ld_res   its pitch in elements
 *  @param delta    rows to add, or NULL
 *  @param ld_delta its pitch in elements
 *  @param n_rows   rows (the grid)
 *  @param width    row width
 *  @param w        norm weight, or NULL
 *  @param y        output rows (required with w)
 *  @param ld_y     its pitch in elements
 *  @return true when the arguments meet the contract */
bool norm_args_ok(const char *who, const void *res, uint64_t ld_res, const void *delta, uint64_t ld_delta,
                  uint32_t n_rows, uint32_t width, const void *w, const void *y, uint64_t ld_y) {
    const bool ok = res && n_rows > 0 && n_rows <= 0x7fffffffu && width > 0 && width % 8 == 0 && ld_res >= width &&
                    ld_res % 8 == 0 && aligned16(res) && (!delta || (ld_delta >= width && ld_delta % 8 == 0 &&
                    aligned16(delta))) && (!w || (aligned16(w) && y && ld_y >= width && ld_y % 8 == 0 && aligned16(y)));
    if (!ok)
        fprintf(stderr, "pulsar: rows %s: %u rows of width %u (a multiple of 8) with 16-byte-aligned rows and leading "
                        "dimensions (multiples of 8, >= the width) -- refusing\n", who, n_rows, width);
    return ok;
}


/* ---- L338: fused row chains (Gemma 4's residual stream between its blocks) -----------------------------------------
 *
 * Each is a sequence of rows_norm_kernel's operations on one row in one CTA, bit for bit: a thread owns the 8-element
 * chunks i, i + 256, ... (held in registers, kChainChunks at most: width <= 8 * 256 * kChainChunks), squares summed as
 * an f32 fma chain over its chunks in order, the warps folded by xor-shuffle, warp 0 adding the eight warp sums in
 * order, rstd = rsqrt(sum / width + eps), y = bf16((x * rstd) * w); a sum rounds once, a scaled sum twice. */

/// Chunks of 8 a thread holds in a fused chain (width <= 8 * kNormThreads * kChainChunks).
constexpr int kChainChunks = 4;

/// rows_norm_kernel's square sum of one held row: this thread's chunks in order.
/// @param u  [kChainChunks][4] the thread's chunks (bf16 pairs)
/// @param nc chunks this thread holds
/// @return the thread's partial
__device__ __forceinline__ float chain_ss(const uint32_t (&u)[kChainChunks][4], int nc) {
    float ss = 0.0f;
#pragma unroll
    for (int c = 0; c < kChainChunks; ++c) {
        if (c >= nc) break;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float a = bf2f(u[c][k] & 0xffffu), b = bf2f(u[c][k] >> 16);
            ss = __fmaf_rn(a, a, ss);
            ss = __fmaf_rn(b, b, ss);
        }
    }
    return ss;
}

/// rows_norm_kernel's rstd from every thread's partial (two block syncs; every thread calls it).
/// @param ss     this thread's partial
/// @param width  the row width
/// @param eps    the norm's epsilon
/// @param warp_s shared [kNormThreads / 32]
/// @param out    shared slot the rstd is published through
/// @return rstd, in every thread
__device__ __forceinline__ float chain_rstd(float ss, uint32_t width, float eps, float *warp_s, float *out) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
    if ((threadIdx.x & 31) == 0) warp_s[threadIdx.x / 32] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int k = 0; k < kNormThreads / 32; ++k) tot = __fadd_rn(tot, warp_s[k]);
        *out = rsqrtf(__fadd_rn(__fdiv_rn(tot, (float)width), eps));
    }
    __syncthreads();
    return *out;
}

/// One normed chunk: bf16((x * rstd) * w) of 8 elements.
/// @param u    the chunk (bf16 pairs)
/// @param wv   the weight's chunk
/// @param rstd the row's rstd
/// @param o    receives the normed chunk
__device__ __forceinline__ void chain_norm(const uint32_t (&u)[4], uint4 wv, float rstd, uint32_t (&o)[4]) {
    const uint32_t wu[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float lo = __fmul_rn(__fmul_rn(bf2f(u[k] & 0xffffu), rstd), bf2f(wu[k] & 0xffffu));
        const float hi = __fmul_rn(__fmul_rn(bf2f(u[k] >> 16), rstd), bf2f(wu[k] >> 16));
        o[k] = f2bf(lo) | (f2bf(hi) << 16);
    }
}

/// One summed chunk: bf16(a + b), and with a scale bf16(bf16(a + b) * s) -- rows_norm_kernel's add.
/// @param a     the residual's chunk, updated
/// @param b     the delta's chunk
/// @param scale device bf16 scalar, or NULL
__device__ __forceinline__ void chain_add(uint32_t (&a)[4], const uint32_t (&b)[4], const uint16_t *scale) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        uint32_t lo = f2bf(__fadd_rn(bf2f(a[k] & 0xffffu), bf2f(b[k] & 0xffffu)));
        uint32_t hi = f2bf(__fadd_rn(bf2f(a[k] >> 16), bf2f(b[k] >> 16)));
        if (scale) {
            const float s = bf2f(*scale);
            lo = f2bf(__fmul_rn(bf2f(lo), s));
            hi = f2bf(__fmul_rn(bf2f(hi), s));
        }
        a[k] = lo | (hi << 16);
    }
}

/// The row-chain outputs of rows_norm_add_norms_kernel: up to three norms of the updated residual.
struct chain_outs {
    const uint4 *w[3];   ///< norm weights (NULL: unused slot)
    uint16_t *y[3];      ///< their output rows
    uint64_t ld[3];      ///< their pitches in elements
};

/** res = bf16(res + bf16(rmsnorm(x) * w_x)), then y_j = bf16(rmsnorm(res) * w_j) for each given j -- a block's
 *  post-norm, the residual add and the next blocks' input norms in one pass (the norms of one row share its rstd: the
 *  same reduction).  Grid n_rows; block kNormThreads.
 *  @param x      device bf16 delta rows (normed before the add)
 *  @param ld_x   x pitch in elements
 *  @param w_x    device bf16 [width] the delta's norm weight
 *  @param res    device bf16 residual rows, updated in place
 *  @param ld_res res pitch
 *  @param width  row width
 *  @param eps    the norms' epsilon
 *  @param outs   the residual's norms */
__global__ void __launch_bounds__(kNormThreads)
rows_norm_add_norms_kernel(const uint16_t *__restrict__ x, uint64_t ld_x, const uint4 *__restrict__ w_x,
                           uint16_t *__restrict__ res, uint64_t ld_res, uint32_t width, float eps, chain_outs outs) {
    __shared__ float warp_s[kNormThreads / 32];
    __shared__ float s_out;
    const uint64_t t = blockIdx.x;
    const uint32_t chunks = width / 8;
    const int nc = threadIdx.x < chunks ? (int)((chunks - threadIdx.x + kNormThreads - 1) / kNormThreads) : 0;
    const uint4 *x4 = (const uint4 *)(x + t * ld_x);
    uint4 *r4 = (uint4 *)(res + t * ld_res);
    uint32_t u[kChainChunks][4], h[kChainChunks][4];
#pragma unroll
    for (int c = 0; c < kChainChunks; ++c) {
        if (c >= nc) break;
        const uint4 v = x4[threadIdx.x + c * kNormThreads];
        u[c][0] = v.x; u[c][1] = v.y; u[c][2] = v.z; u[c][3] = v.w;
    }
    const float r0 = chain_rstd(chain_ss(u, nc), width, eps, warp_s, &s_out);
#pragma unroll
    for (int c = 0; c < kChainChunks; ++c) {
        if (c >= nc) break;
        const uint32_t i = threadIdx.x + c * kNormThreads;
        uint32_t d[4];
        chain_norm(u[c], w_x[i], r0, d);
        const uint4 v = r4[i];
        h[c][0] = v.x; h[c][1] = v.y; h[c][2] = v.z; h[c][3] = v.w;
        chain_add(h[c], d, nullptr);
        r4[i] = make_uint4(h[c][0], h[c][1], h[c][2], h[c][3]);
    }
    const float r1 = chain_rstd(chain_ss(h, nc), width, eps, warp_s, &s_out);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        if (!outs.w[j]) continue;
        uint4 *y4 = (uint4 *)(outs.y[j] + t * outs.ld[j]);
#pragma unroll
        for (int c = 0; c < kChainChunks; ++c) {
            if (c >= nc) break;
            const uint32_t i = threadIdx.x + c * kNormThreads;
            uint32_t o[4];
            chain_norm(h[c], outs.w[j][i], r1, o);
            y4[i] = make_uint4(o[0], o[1], o[2], o[3]);
        }
    }
}

/** Two f32 branch outputs into the residual: e_a = bf16(a), e_b = bf16(b); s = bf16(bf16(rmsnorm(e_a) * w_a) +
 *  bf16(rmsnorm(e_b) * w_b)); res = bf16(bf16(res + bf16(rmsnorm(s) * w_s)) * scale); y = bf16(rmsnorm(res) * w_n)
 *  -- Gemma 4's FFN tail (the dense MLP and the experts, their post-norms, the sum's norm, the layer's scaled add and
 *  the next reader's norm) in one pass.  Grid n_rows; block kNormThreads.
 *  @param a      device f32 rows of branch a
 *  @param ld_a   a pitch in elements
 *  @param w_a    device bf16 [width] branch a's norm weight
 *  @param b      device f32 rows of branch b
 *  @param ld_b   b pitch
 *  @param w_b    branch b's norm weight
 *  @param w_s    the sum's norm weight
 *  @param res    device bf16 residual rows, updated in place
 *  @param ld_res res pitch
 *  @param scale  device bf16 scalar the summed residual is multiplied by
 *  @param w_n    the updated residual's norm weight
 *  @param y      device bf16 rows of that norm
 *  @param ld_y   y pitch
 *  @param width  row width
 *  @param eps    the norms' epsilon */
__global__ void __launch_bounds__(kNormThreads)
rows_sum2_norm_add_kernel(const float *__restrict__ a, uint64_t ld_a, const uint4 *__restrict__ w_a,
                          const float *__restrict__ b, uint64_t ld_b, const uint4 *__restrict__ w_b,
                          const uint4 *__restrict__ w_s, uint16_t *__restrict__ res, uint64_t ld_res,
                          const uint16_t *__restrict__ scale, const uint4 *__restrict__ w_n, uint16_t *__restrict__ y,
                          uint64_t ld_y, uint32_t width, float eps) {
    __shared__ float warp_s[kNormThreads / 32];
    __shared__ float s_out;
    const uint64_t t = blockIdx.x;
    const uint32_t chunks = width / 8;
    const int nc = threadIdx.x < chunks ? (int)((chunks - threadIdx.x + kNormThreads - 1) / kNormThreads) : 0;
    uint32_t ua[kChainChunks][4], ub[kChainChunks][4];
#pragma unroll
    for (int c = 0; c < kChainChunks; ++c) {
        if (c >= nc) break;
        const uint32_t i = threadIdx.x + c * kNormThreads;
        const float4 *pa = (const float4 *)(a + t * ld_a + (uint64_t)i * 8u);
        const float4 *pb = (const float4 *)(b + t * ld_b + (uint64_t)i * 8u);
        const float4 a0 = pa[0], a1 = pa[1], b0 = pb[0], b1 = pb[1];
        ua[c][0] = f2bf(a0.x) | (f2bf(a0.y) << 16); ua[c][1] = f2bf(a0.z) | (f2bf(a0.w) << 16);
        ua[c][2] = f2bf(a1.x) | (f2bf(a1.y) << 16); ua[c][3] = f2bf(a1.z) | (f2bf(a1.w) << 16);
        ub[c][0] = f2bf(b0.x) | (f2bf(b0.y) << 16); ub[c][1] = f2bf(b0.z) | (f2bf(b0.w) << 16);
        ub[c][2] = f2bf(b1.x) | (f2bf(b1.y) << 16); ub[c][3] = f2bf(b1.z) | (f2bf(b1.w) << 16);
    }
    /* L349: the four norm weights and the residual row load here, ahead of the four reductions -- each was a
     * dependent round trip after its reduction on a one-CTA-a-row kernel's critical path */
    uint4 *r4 = (uint4 *)(res + t * ld_res);
    uint4 pwa[kChainChunks], pwb[kChainChunks], pws[kChainChunks], pwn[kChainChunks], pres[kChainChunks];
#pragma unroll
    for (int c = 0; c < kChainChunks; ++c) {
        if (c >= nc) break;
        const uint32_t i = threadIdx.x + c * kNormThreads;
        pwa[c] = w_a[i]; pwb[c] = w_b[i]; pws[c] = w_s[i]; pwn[c] = w_n[i]; pres[c] = r4[i];
    }
    const float ra = chain_rstd(chain_ss(ua, nc), width, eps, warp_s, &s_out);
    const float rb = chain_rstd(chain_ss(ub, nc), width, eps, warp_s, &s_out);
#pragma unroll
    for (int c = 0; c < kChainChunks; ++c) {   /* ua := the sum of the two post-norms */
        if (c >= nc) break;
        uint32_t da[4], db[4];
        chain_norm(ua[c], pwa[c], ra, da);
        chain_norm(ub[c], pwb[c], rb, db);
        chain_add(da, db, nullptr);
#pragma unroll
        for (int k = 0; k < 4; ++k) ua[c][k] = da[k];
    }
    const float rs = chain_rstd(chain_ss(ua, nc), width, eps, warp_s, &s_out);
#pragma unroll
    for (int c = 0; c < kChainChunks; ++c) {   /* ub := the updated residual */
        if (c >= nc) break;
        const uint32_t i = threadIdx.x + c * kNormThreads;
        uint32_t d[4];
        chain_norm(ua[c], pws[c], rs, d);
        const uint4 v = pres[c];
        ub[c][0] = v.x; ub[c][1] = v.y; ub[c][2] = v.z; ub[c][3] = v.w;
        chain_add(ub[c], d, scale);
        r4[i] = make_uint4(ub[c][0], ub[c][1], ub[c][2], ub[c][3]);
    }
    const float rn = chain_rstd(chain_ss(ub, nc), width, eps, warp_s, &s_out);
    uint4 *y4 = (uint4 *)(y + t * ld_y);
#pragma unroll
    for (int c = 0; c < kChainChunks; ++c) {
        if (c >= nc) break;
        const uint32_t i = threadIdx.x + c * kNormThreads;
        uint32_t o[4];
        chain_norm(ub[c], pwn[c], rn, o);
        y4[i] = make_uint4(o[0], o[1], o[2], o[3]);
    }
}

/// A fused chain's shape: rows, a width of whole chunks within kChainChunks a thread, 16-byte rows.
/// @param who   the launch's name
/// @param n_rows rows
/// @param width the row width
/// @param eps   the epsilon
/// @return true when launchable (else said)
bool chain_args_ok(const char *who, uint32_t n_rows, uint32_t width, float eps) {
    const bool ok = n_rows > 0 && n_rows <= 0x7fffffffu && width > 0 && width % 8 == 0 &&
                    width <= 8u * kNormThreads * kChainChunks && eps >= 0.0f;
    if (!ok)
        fprintf(stderr, "pulsar: rows %s: %u rows of width %u (a multiple of 8, at most %u), eps >= 0 -- refusing\n",
                who, n_rows, width, 8u * kNormThreads * kChainChunks);
    return ok;
}

/// A bf16 row operand: present, 16-byte aligned, a pitch of whole chunks >= the width.
/// @param p the rows
/// @param ld their pitch
/// @param width the row width
/// @return true when usable
bool chain_rows_ok(const void *p, uint64_t ld, uint32_t width) { return p && aligned16(p) && ld >= width && ld % 8 == 0; }

} // namespace

extern "C" int pulsar_rows_rmsnorm_launch(const uint16_t *x, uint64_t ld_x, uint32_t n_rows, uint32_t width,
                                          const uint16_t *w, float eps, uint16_t *y, uint64_t ld_y,
                                          cudaStream_t stream) {
    if (!w || !(eps >= 0.0f) || !norm_args_ok("rmsnorm", x, ld_x, nullptr, 0, n_rows, width, w, y, ld_y)) {
        if (!w || !(eps >= 0.0f)) fprintf(stderr, "pulsar: rows rmsnorm: no weight or a negative eps -- refusing\n");
        return -1;
    }
    rows_norm_kernel<<<n_rows, kNormThreads, 0, stream>>>(const_cast<uint16_t *>(x), ld_x, nullptr, 0, width,
                                                          (const uint4 *)w, eps, y, ld_y);
    return launched("rmsnorm");
}

extern "C" int pulsar_rows_add_launch(uint16_t *res, uint64_t ld_res, const uint16_t *delta, uint64_t ld_delta,
                                      uint32_t n_rows, uint32_t width, const uint16_t *w, float eps, uint16_t *y,
                                      uint64_t ld_y, cudaStream_t stream) {
    if (!delta || (w && !(eps >= 0.0f)) ||
        !norm_args_ok("add", res, ld_res, delta, ld_delta, n_rows, width, w, y, ld_y)) {
        if (!delta || (w && !(eps >= 0.0f)))
            fprintf(stderr, "pulsar: rows add: no delta or a negative eps -- refusing\n");
        return -1;
    }
    rows_norm_kernel<<<n_rows, kNormThreads, 0, stream>>>(res, ld_res, delta, ld_delta, width, (const uint4 *)w, eps,
                                                          y, ld_y);
    return launched("add");
}

extern "C" int pulsar_rows_emit_bf16_launch(const float *x, uint64_t ld_x, uint32_t n_rows, uint32_t width,
                                            const void *scale, pulsar_rows_scale kind, uint16_t *y, uint64_t ld_y,
                                            cudaStream_t stream) {
    if (!x || !y || n_rows == 0 || n_rows > 65535u || width == 0 || width % 8 || ld_x < width || ld_x % 4 ||
        ld_y < width || ld_y % 8 || !aligned16(x) || !aligned16(y) ||
        (kind == PULSAR_ROWS_SCALE_NONE) != (scale == nullptr) || kind < PULSAR_ROWS_SCALE_NONE ||
        kind > PULSAR_ROWS_SCALE_BF16) {
        fprintf(stderr, "pulsar: rows emit: %u rows (1 .. 65535) of width %u (a multiple of 8), aligned rows, ld_x %llu "
                        "(a multiple of 4) / ld_y %llu (of 8) >= the width, a scalar exactly when one is declared "
                        "-- refusing\n", n_rows, width, (unsigned long long)ld_x, (unsigned long long)ld_y);
        return -1;
    }
    const dim3 grid((width / 4 + kEmitThreads - 1) / kEmitThreads, n_rows);
    rows_emit_kernel<<<grid, kEmitThreads, 0, stream>>>(x, ld_x, width, scale, (int)kind, y, ld_y);
    return launched("emit");
}

extern "C" int pulsar_rows_bias_emit_launch(const float *x, uint64_t ld_x, uint32_t n_rows, uint32_t width,
                                            const float *bias, pulsar_rows_act act, uint16_t *y, uint64_t ld_y,
                                            pulsar_rows_out y_fmt, cudaStream_t stream) {
    if (!x || !y || n_rows == 0 || n_rows > 65535u || width == 0 || width % 8 || ld_x < width || ld_x % 4 ||
        ld_y < width || ld_y % 8 || !aligned16(x) || !aligned16(y) || (bias && !aligned16(bias)) ||
        (act != PULSAR_ROWS_ACT_NONE && act != PULSAR_ROWS_ACT_GELU && act != PULSAR_ROWS_ACT_RELU2) ||
        (y_fmt != PULSAR_ROWS_OUT_BF16 && y_fmt != PULSAR_ROWS_OUT_F16)) {
        fprintf(stderr, "pulsar: rows bias emit: %u rows (1 .. 65535) of width %u (a multiple of 8), aligned rows, "
                        "ld_x %llu (a multiple of 4) / ld_y %llu (of 8) >= the width, a known activation -- refusing\n",
                n_rows, width, (unsigned long long)ld_x, (unsigned long long)ld_y);
        return -1;
    }
    const dim3 grid((width / 4 + kEmitThreads - 1) / kEmitThreads, n_rows);
    rows_bias_emit_kernel<<<grid, kEmitThreads, 0, stream>>>(x, ld_x, width, bias, (int)act, y, ld_y,
                                                             y_fmt == PULSAR_ROWS_OUT_F16);
    return launched("bias emit");
}

extern "C" int pulsar_rows_gather_f16_launch(const uint16_t *table, uint32_t n_vocab, uint32_t width, const int32_t *ids,
                                             uint32_t n, float scale, float *y, uint64_t ld_y, cudaStream_t stream) {
    if (!table || !ids || !y || n == 0 || n > 65535u || n_vocab == 0 || width == 0 || width % 8 || ld_y < width ||
        ld_y % 4 || !aligned16(table) || !aligned16(y) || !isfinite(scale)) {
        fprintf(stderr, "pulsar: rows gather f16: %u ids (1 .. 65535) of width %u (a multiple of 8), aligned table and "
                        "rows, a finite scale -- refusing\n", n, width);
        return -1;
    }
    const dim3 grid((width / 8 + kEmitThreads - 1) / kEmitThreads, n);
    rows_gather_f16_kernel<<<grid, kEmitThreads, 0, stream>>>((const uint4 *)table, n_vocab, width, ids, scale, y, ld_y);
    return launched("gather f16");
}

extern "C" int pulsar_rows_res32_add_launch(float *res, uint64_t ld_res, const float *x, uint64_t ld_x, const float *bias,
                                            uint32_t n_rows, uint32_t width, const float *w, const float *b, float eps,
                                            uint16_t *y, uint64_t ld_y, pulsar_rows_out y_fmt, cudaStream_t stream) {
    if (!res || !x || n_rows == 0 || n_rows > 0x7fffffffu || width == 0 || width % 4 || ld_res < width || ld_res % 4 ||
        ld_x < width || ld_x % 4 || (!w != !b) || (w && (!y || ld_y < width || !(eps >= 0.0f))) ||
        (y_fmt != PULSAR_ROWS_OUT_BF16 && y_fmt != PULSAR_ROWS_OUT_F16)) {
        fprintf(stderr, "pulsar: rows res32 add: %u rows of width %u (a multiple of 4), leading dimensions >= the width "
                        "(multiples of 4), a norm weight with its bias and its output -- refusing\n", n_rows, width);
        return -1;
    }
    rows_ln32_kernel<true><<<n_rows, kNormThreads, 0, stream>>>(res, ld_res, x, ld_x, bias, width, w, b, eps, y, ld_y,
                                                                y_fmt == PULSAR_ROWS_OUT_F16, nullptr, 0);
    return launched("res32 add");
}

extern "C" int pulsar_rows_layernorm32_launch(const void *x, bool x_f32, uint64_t ld_x, const float *x_bias,
                                              uint32_t n_rows, uint32_t width, const float *w, const float *b, float eps,
                                              uint16_t *y, uint64_t ld_y, pulsar_rows_out y_fmt, float *yf, uint64_t ld_yf,
                                              cudaStream_t stream) {
    if (!x || !w || !b || (!y && !yf) || (x_bias && !x_f32) || n_rows == 0 || n_rows > 0x7fffffffu || width == 0 ||
        ld_x < width ||
        (y && ld_y < width) || (yf && ld_yf < width) || !(eps >= 0.0f) ||
        (y_fmt != PULSAR_ROWS_OUT_BF16 && y_fmt != PULSAR_ROWS_OUT_F16)) {
        fprintf(stderr, "pulsar: rows layernorm32: %u rows of width %u, a weight and a bias, an output, leading dimensions "
                        ">= the width -- refusing\n", n_rows, width);
        return -1;
    }
    if (x_f32)
        rows_ln32_kernel<true><<<n_rows, kNormThreads, 0, stream>>>(const_cast<void *>(x), ld_x, nullptr, 0, x_bias,
                                                                    width, w, b, eps, y, ld_y,
                                                                    y_fmt == PULSAR_ROWS_OUT_F16, yf, ld_yf);
    else
        rows_ln32_kernel<false><<<n_rows, kNormThreads, 0, stream>>>(const_cast<void *>(x), ld_x, nullptr, 0, nullptr,
                                                                     width, w, b, eps, y, ld_y,
                                                                     y_fmt == PULSAR_ROWS_OUT_F16, yf, ld_yf);
    return launched("layernorm32");
}

extern "C" int pulsar_rows_norm_add_norms_launch(const uint16_t *x, uint64_t ld_x, const uint16_t *w_x, uint16_t *res,
                                                 uint64_t ld_res, uint32_t n_rows, uint32_t width,
                                                 const uint16_t *const w[3], uint16_t *const y[3],
                                                 const uint64_t ld_y[3], float eps, cudaStream_t stream) {
    if (!chain_args_ok("norm-add-norms", n_rows, width, eps)) return -1;
    bool ok = chain_rows_ok(x, ld_x, width) &&
              w_x && aligned16(w_x) && chain_rows_ok(res, ld_res, width) && w && y && ld_y;
    chain_outs o{};
    for (int j = 0; ok && j < 3; ++j) {
        if (!w[j]) continue;
        ok = aligned16(w[j]) && chain_rows_ok(y[j], ld_y[j], width);
        o.w[j] = (const uint4 *)w[j];
        o.y[j] = y[j];
        o.ld[j] = ld_y[j];
    }
    if (!ok) {
        fprintf(stderr, "pulsar: rows norm-add-norms: a missing or misaligned operand -- refusing\n");
        return -1;
    }
    rows_norm_add_norms_kernel<<<n_rows, kNormThreads, 0, stream>>>(x, ld_x, (const uint4 *)w_x, res, ld_res, width,
                                                                    eps, o);
    return launched("norm-add-norms");
}

extern "C" int pulsar_rows_sum2_norm_add_launch(const float *a, uint64_t ld_a, const uint16_t *w_a, const float *b,
                                                uint64_t ld_b, const uint16_t *w_b, const uint16_t *w_s, uint16_t *res,
                                                uint64_t ld_res, const uint16_t *scale, const uint16_t *w_n,
                                                uint16_t *y, uint64_t ld_y, uint32_t n_rows, uint32_t width, float eps,
                                                cudaStream_t stream) {
    if (!chain_args_ok("sum2-norm-add", n_rows, width, eps)) return -1;
    const bool ok = a && aligned16(a) && ld_a >= width &&
                    ld_a % 8 == 0 && b && aligned16(b) && ld_b >= width && ld_b % 8 == 0 && w_a && aligned16(w_a) &&
                    w_b && aligned16(w_b) && w_s && aligned16(w_s) && chain_rows_ok(res, ld_res, width) && scale &&
                    w_n && aligned16(w_n) && chain_rows_ok(y, ld_y, width);
    if (!ok) {
        fprintf(stderr, "pulsar: rows sum2-norm-add: a missing or misaligned operand -- refusing\n");
        return -1;
    }
    rows_sum2_norm_add_kernel<<<n_rows, kNormThreads, 0, stream>>>(a, ld_a, (const uint4 *)w_a, b, ld_b,
                                                                   (const uint4 *)w_b, (const uint4 *)w_s, res, ld_res,
                                                                   scale, (const uint4 *)w_n, y, ld_y, width, eps);
    return launched("sum2-norm-add");
}

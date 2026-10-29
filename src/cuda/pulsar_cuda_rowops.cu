/* Row operations between a layer plan's blocks (L295): contracts in pulsar_cuda_rowops.h. */
#include "pulsar_cuda_rowops.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

namespace {

constexpr int kNormThreads = 256;
constexpr int kEmitThreads = 256;

__device__ __forceinline__ float bf2f(uint32_t b) { return __uint_as_float(b << 16); }
__device__ __forceinline__ uint32_t f2bf(float f) { return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(f)); }

/* One CTA a row.  Thread i owns the 8-element chunks i, i + 256, ... of the row.  With a delta, the thread first
 * writes its chunks' rounded sums back to the residual (and reads them from registers for the squares, so the norm
 * is of the stored bytes); its squares are an f32 fma chain over its chunks in order, the warps fold by
 * xor-shuffle, warp 0 adds the eight warp sums in order. */
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
__device__ __forceinline__ uint32_t f2out(float f, int f16) {
    return f16 ? (uint32_t)__half_as_ushort(__float2half_rn(f)) : f2bf(f);
}

/* Four elements a thread: act(x + bias) as bf16 / f16. */
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

int launched(const char *who) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: rows %s: launch failed: %s\n", who, cudaGetErrorString(e));
        return -3;
    }
    return 0;
}

bool aligned16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

/* the shared contract of the norm and the add: widths and strides in 8s, aligned rows, a grid of n_rows CTAs */
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

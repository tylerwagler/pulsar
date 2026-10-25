/* Row operations between a layer plan's blocks (L295): contracts in pulsar_cuda_rowops.h. */
#include "pulsar_cuda_rowops.h"

#include <cuda_bf16.h>
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

/* L321: one CTA a row, thread i owning the 8-element chunks i, i + 256, ... .  With x (a linear's f32 output), the
 * thread first writes res + (x + bias) rounded once back to the residual; then, with w, the LayerNorm of the stored
 * row: the sum, the mean, the centred squares (re-read from the thread's own chunks), each an f32 chain per thread in
 * chunk order, the warps folded by xor-shuffle and warp 0 adding the warp sums in order. */
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

__global__ void __launch_bounds__(kNormThreads)
rows_ln_kernel(uint16_t *__restrict__ res, uint64_t ld_res, const float *__restrict__ x, uint64_t ld_x,
               const float *__restrict__ bias, uint32_t width, const float *__restrict__ w, const float *__restrict__ b,
               float eps, uint16_t *__restrict__ y, uint64_t ld_y) {
    __shared__ float warp_s[kNormThreads / 32];
    __shared__ float s_sum, s_sq;
    const uint64_t t = blockIdx.x;
    const uint32_t chunks = width / 8;
    uint4 *r4 = (uint4 *)(res + t * ld_res);
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < chunks; i += kNormThreads) {
        uint4 v = r4[i];
        uint32_t u[4] = {v.x, v.y, v.z, v.w};
        if (x) {
            const float *xr = x + t * ld_x + 8u * i;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t c = 8u * i + 2u * k;
                const float d0 = __fadd_rn(xr[2 * k], bias ? bias[c] : 0.0f);
                const float d1 = __fadd_rn(xr[2 * k + 1], bias ? bias[c + 1] : 0.0f);
                const uint32_t lo = f2bf(__fadd_rn(bf2f(u[k] & 0xffffu), d0));
                const uint32_t hi = f2bf(__fadd_rn(bf2f(u[k] >> 16), d1));
                u[k] = lo | (hi << 16);
            }
            v = make_uint4(u[0], u[1], u[2], u[3]);
            r4[i] = v;
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) sum = __fadd_rn(__fadd_rn(sum, bf2f(u[k] & 0xffffu)), bf2f(u[k] >> 16));
    }
    if (!w) return;   /* a bias add alone */
    const float mean = __fdiv_rn(ln_block_sum(sum, warp_s, &s_sum), (float)width);
    float sq = 0.0f;
    for (uint32_t i = threadIdx.x; i < chunks; i += kNormThreads) {
        const uint4 v = r4[i];   /* this thread's own chunks: its writes above are visible to it */
        const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float a = __fsub_rn(bf2f(u[k] & 0xffffu), mean), c = __fsub_rn(bf2f(u[k] >> 16), mean);
            sq = __fmaf_rn(a, a, sq);
            sq = __fmaf_rn(c, c, sq);
        }
    }
    const float rstd = rsqrtf(__fadd_rn(__fdiv_rn(ln_block_sum(sq, warp_s, &s_sq), (float)width), eps));
    uint4 *y4 = (uint4 *)(y + t * ld_y);
    for (uint32_t i = threadIdx.x; i < chunks; i += kNormThreads) {
        const uint4 v = r4[i];
        const uint32_t u[4] = {v.x, v.y, v.z, v.w};
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t c = 8u * i + 2u * k;
            const float lo = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(bf2f(u[k] & 0xffffu), mean), rstd), w[c]), b[c]);
            const float hi =
                __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(bf2f(u[k] >> 16), mean), rstd), w[c + 1]), b[c + 1]);
            o[k] = f2bf(lo) | (f2bf(hi) << 16);
        }
        y4[i] = make_uint4(o[0], o[1], o[2], o[3]);
    }
}

/* Four elements a thread: bf16(act(x + bias)). */
__global__ void __launch_bounds__(kEmitThreads)
rows_bias_emit_kernel(const float *__restrict__ x, uint64_t ld_x, uint32_t width, const float *__restrict__ bias,
                      int act, uint16_t *__restrict__ y, uint64_t ld_y) {
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
        o[k] = f2bf(z);
    }
    *(uint2 *)(y + t * ld_y + c) = make_uint2(o[0] | (o[1] << 16), o[2] | (o[3] << 16));
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

extern "C" int pulsar_rows_layernorm_launch(const uint16_t *x, uint64_t ld_x, uint32_t n_rows, uint32_t width,
                                            const float *w, const float *b, float eps, uint16_t *y, uint64_t ld_y,
                                            cudaStream_t stream) {
    if (!w || !b || !aligned16(w) || !aligned16(b) || !(eps >= 0.0f) ||
        !norm_args_ok("layernorm", x, ld_x, nullptr, 0, n_rows, width, w, y, ld_y)) {
        if (!w || !b || !(eps >= 0.0f)) fprintf(stderr, "pulsar: rows layernorm: no weight / bias or a negative eps -- refusing\n");
        return -1;
    }
    rows_ln_kernel<<<n_rows, kNormThreads, 0, stream>>>(const_cast<uint16_t *>(x), ld_x, nullptr, 0, nullptr, width, w,
                                                        b, eps, y, ld_y);
    return launched("layernorm");
}

extern "C" int pulsar_rows_bias_add_launch(uint16_t *res, uint64_t ld_res, const float *x, uint64_t ld_x,
                                           const float *bias, uint32_t n_rows, uint32_t width, const float *w,
                                           const float *b, float eps, uint16_t *y, uint64_t ld_y, cudaStream_t stream) {
    if (!x || ld_x < width || ld_x % 4 || !aligned16(x) || (bias && !aligned16(bias)) || (!w != !b) ||
        (w && !(eps >= 0.0f)) || !norm_args_ok("bias add", res, ld_res, nullptr, 0, n_rows, width, w, y, ld_y)) {
        if (!x || (!w != !b) || (w && !(eps >= 0.0f)))
            fprintf(stderr, "pulsar: rows bias add: no input, a norm weight without its bias, or a negative eps -- "
                            "refusing\n");
        return -1;
    }
    rows_ln_kernel<<<n_rows, kNormThreads, 0, stream>>>(res, ld_res, x, ld_x, bias, width, w, b, eps, y, ld_y);
    return launched("bias add");
}

extern "C" int pulsar_rows_bias_emit_launch(const float *x, uint64_t ld_x, uint32_t n_rows, uint32_t width,
                                            const float *bias, pulsar_rows_act act, uint16_t *y, uint64_t ld_y,
                                            cudaStream_t stream) {
    if (!x || !y || n_rows == 0 || n_rows > 65535u || width == 0 || width % 8 || ld_x < width || ld_x % 4 ||
        ld_y < width || ld_y % 8 || !aligned16(x) || !aligned16(y) || (bias && !aligned16(bias)) ||
        (act != PULSAR_ROWS_ACT_NONE && act != PULSAR_ROWS_ACT_GELU)) {
        fprintf(stderr, "pulsar: rows bias emit: %u rows (1 .. 65535) of width %u (a multiple of 8), aligned rows, "
                        "ld_x %llu (a multiple of 4) / ld_y %llu (of 8) >= the width, a known activation -- refusing\n",
                n_rows, width, (unsigned long long)ld_x, (unsigned long long)ld_y);
        return -1;
    }
    const dim3 grid((width / 4 + kEmitThreads - 1) / kEmitThreads, n_rows);
    rows_bias_emit_kernel<<<grid, kEmitThreads, 0, stream>>>(x, ld_x, width, bias, (int)act, y, ld_y);
    return launched("bias emit");
}

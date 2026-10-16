// SPDX-License-Identifier: MIT
// L298: NVFP4 W4A16 against raw bf16 rows (contract, layout and arithmetic in pulsar_cuda_nvfp4.h).
//
// The kernel is the BF16 rows kernel (pulsar_cuda_bf16_rows.cu) with the weight decoded in registers.  A CTA is 8
// warps: R = 8 / S row groups of 16 weight rows (the MMA's M) times S K slices.  Lane (g, c) holds, per 64-input
// step, inputs [16 c, 16 c + 16) of rows r0 = 16-row group + g and r1 = r0 + 8: 8 bytes of codes and ONE E4M3 scale
// each (the 16 inputs are exactly one scale block).  A 256-entry shared table maps a code byte to its two e2m1 values
// as bf16x2; one bf16x2 multiply by the scale gives the dequantized pair, exact.  The activation fragments come
// straight from global (L1-resident: every warp of the CTA reads the same rows).  The loads of U steps are issued
// before any of them is consumed -- the decode lane is memory-bound, so bytes in flight are the whole game.
//
// Per step and n8 tile: 4 MMAs, each into a fresh fragment added into the row's f32 accumulator, in k order -- the
// rows kernel's order exactly.  Slices k > 0 park their accumulators in shared memory; slice 0 adds them in slice
// order, multiplies by the global scale and stores.

#include "pulsar_cuda_nvfp4.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdio.h>

namespace {

constexpr int kWarps = 8;
constexpr int kTargetCtas = 96;   ///< two CTAs a GB10 SM: the K split's goal (pulsar_nvfp4_ksplit)
constexpr int kUnroll = 8;        ///< 64-input steps whose loads are in flight together

__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/** e2m1 code (sign, 2 exponent bits, 1 mantissa bit) -> its bf16 bits: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and negatives. */
__device__ __forceinline__ uint32_t e2m1_bf16(uint32_t v) {
    const uint32_t e = (v >> 1) & 3u, m = v & 1u;
    const uint32_t mag = e ? ((e + 126u) << 7) | (m << 6) : (m ? 126u << 7 : 0u);
    return ((v & 8u) << 12) | mag;
}

/** An E4M3 scale byte as bf16x2 (both halves), exact. */
__device__ __forceinline__ uint32_t e4m3_bf16x2(uint32_t b) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    const __nv_bfloat16 v = __float2bfloat16(__half2float(__half(h)));
    const uint32_t u = *reinterpret_cast<const uint16_t *>(&v);
    return u | (u << 16);
}

__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
    __nv_bfloat162 r = __hmul2(*reinterpret_cast<const __nv_bfloat162 *>(&a), *reinterpret_cast<const __nv_bfloat162 *>(&b));
    return *reinterpret_cast<uint32_t *>(&r);
}

/* One CTA: rows [blockIdx.x * 16 R, +16 R) of W for nr <= 8 NT activation rows, K split S ways across its warps. */
template <int NT, int S>
__global__ void __launch_bounds__(32 * kWarps)
nvfp4_rows_kernel(const uint8_t *__restrict__ Wq, const uint8_t *__restrict__ SF, const float *__restrict__ G,
                  int out, int in, const __nv_bfloat16 *__restrict__ X, int nr, float *__restrict__ Y,
                  __nv_bfloat16 *__restrict__ Yb) {
    constexpr int R = kWarps / S;
    __shared__ uint32_t lut[256];
    __shared__ float part[S > 1 ? S - 1 : 1][R][32][NT * 4];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) lut[i] = e2m1_bf16(i & 15u) | (e2m1_bf16(i >> 4) << 16);
    __syncthreads();

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int rg = warp % R, kq = warp / R;
    const int r0 = (int)blockIdx.x * 16 * R + rg * 16 + g, r1 = r0 + 8;
    const bool v0 = r0 < out, v1 = r1 < out;
    const int nt = (nr + 7) >> 3;
    const int kspan = in / S, kbeg = kq * kspan, kend = kbeg + kspan;
    /* the lane's codes: 8 bytes of each row per step, at input k + 16 c */
    const uint8_t *q0 = Wq + (size_t)(v0 ? r0 : 0) * (in / 2) + 8 * c;
    const uint8_t *q1 = Wq + (size_t)(v1 ? r1 : 0) * (in / 2) + 8 * c;
    /* the scale word of a row at a 64-input step: its 4 scale blocks, the lane's is byte c */
    const int kb4 = in / 64;
    auto sf_row = [&](int r) {
        return SF + (size_t)(r / 128) * kb4 * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4;
    };
    const uint8_t *s0 = sf_row(v0 ? r0 : 0), *s1 = sf_row(v1 ? r1 : 0);
    /* the lane's activation rows (row g of each n8 tile; a row past the call's reads row 0 and is zeroed) */
    const __nv_bfloat16 *xr[NT];
    bool xv[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        xv[j] = j * 8 + g < nr;
        xr[j] = X + (size_t)(xv[j] ? j * 8 + g : 0) * in + 16 * c;
    }

    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;
    const uint2 z2 = make_uint2(0u, 0u);
    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);

    for (int k0 = kbeg; k0 < kend; k0 += 64 * kUnroll) {
        uint2 a0[kUnroll], a1[kUnroll];
        uint32_t w0[kUnroll], w1[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int k = k0 + 64 * u;
            const bool in_k = k < kend;
            a0[u] = v0 && in_k ? __ldcs(reinterpret_cast<const uint2 *>(q0 + k / 2)) : z2;
            a1[u] = v1 && in_k ? __ldcs(reinterpret_cast<const uint2 *>(q1 + k / 2)) : z2;
            w0[u] = v0 && in_k ? __ldcs(reinterpret_cast<const uint32_t *>(s0 + (size_t)(k / 64) * 512)) : 0u;
            w1[u] = v1 && in_k ? __ldcs(reinterpret_cast<const uint32_t *>(s1 + (size_t)(k / 64) * 512)) : 0u;
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int k = k0 + 64 * u;
            if (k >= kend) break;
            const uint32_t sc0 = e4m3_bf16x2((w0[u] >> (8 * c)) & 0xffu), sc1 = e4m3_bf16x2((w1[u] >> (8 * c)) & 0xffu);
            /* [2s] = inputs 4s, 4s+1; [2s+1] = 4s+2, 4s+3 of the lane's 16 (the rows kernel's A words) */
            uint32_t A0[8], A1[8];
            const uint32_t b0[2] = {a0[u].x, a0[u].y}, b1[2] = {a1[u].x, a1[u].y};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                A0[i] = bmul2(lut[(b0[i >> 2] >> (8 * (i & 3))) & 0xffu], sc0);
                A1[i] = bmul2(lut[(b1[i >> 2] >> (8 * (i & 3))) & 0xffu], sc1);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                if (j >= nt) break;
                const uint4 p0 = xv[j] ? __ldg(reinterpret_cast<const uint4 *>(xr[j] + k)) : z4;
                const uint4 p1 = xv[j] ? __ldg(reinterpret_cast<const uint4 *>(xr[j] + k) + 1) : z4;
                const uint32_t B[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const uint32_t af[4] = {A0[2 * s], A1[2 * s], A0[2 * s + 1], A1[2 * s + 1]};
                    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    mma_bf16_16816(d, af, B[2 * s], B[2 * s + 1]);
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[j][e] += d[e];
                }
            }
        }
    }

    if constexpr (S > 1) {
        if (kq > 0) {
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) part[kq - 1][rg][lane][j * 4 + e] = acc[j][e];
        }
        __syncthreads();
        if (kq > 0) return;
#pragma unroll
        for (int q = 0; q < S - 1; ++q)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[j][e] += part[q][rg][lane][j * 4 + e];
    }
    const float gs = *G;
    /* acc[j][0..1] = row r0, call rows 8j + 2c, +1; acc[j][2..3] = row r1 */
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int t = 8 * j + 2 * c + (e & 1), row = e < 2 ? r0 : r1;
            if (t < nr && (e < 2 ? v0 : v1)) {
                const float v = acc[j][e] * gs;
                if (Y) Y[(size_t)t * out + row] = v;
                if (Yb) Yb[(size_t)t * out + row] = __float2bfloat16(v);
            }
        }
}

template <int NT>
void launch(int S, dim3 grid, const uint8_t *q, const uint8_t *sf, const float *g, int out, int in,
            const __nv_bfloat16 *X, int nr, float *Y, __nv_bfloat16 *Yb, cudaStream_t st) {
    switch (S) {
    case 1: nvfp4_rows_kernel<NT, 1><<<grid, 32 * kWarps, 0, st>>>(q, sf, g, out, in, X, nr, Y, Yb); break;
    case 2: nvfp4_rows_kernel<NT, 2><<<grid, 32 * kWarps, 0, st>>>(q, sf, g, out, in, X, nr, Y, Yb); break;
    case 4: nvfp4_rows_kernel<NT, 4><<<grid, 32 * kWarps, 0, st>>>(q, sf, g, out, in, X, nr, Y, Yb); break;
    default: nvfp4_rows_kernel<NT, 8><<<grid, 32 * kWarps, 0, st>>>(q, sf, g, out, in, X, nr, Y, Yb); break;
    }
}

bool aligned16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

} // namespace

int pulsar_nvfp4_ksplit(int in, int out) {
    int S = 1;
    while (S < kWarps && (long)(out + 16 * (kWarps / S) - 1) / (16 * (kWarps / S)) < kTargetCtas && in % (64 * 2 * S) == 0)
        S *= 2;
    return S;
}

int pulsar_nvfp4_rows_linear_launch(const void *w, const uint16_t *x_bf16, int rows, int in, int out, float *y,
                                    uint16_t *y_bf16, cudaStream_t stream) {
    uint64_t vb = 0, sb = 0, stride = 0;
    if (!w || !x_bf16 || (!y && !y_bf16) || rows <= 0 || out <= 0 ||
        !pulsar_nvfp4_layout((uint64_t)in, (uint64_t)out, &vb, &sb, &stride) || !aligned16(w) || !aligned16(x_bf16)) {
        fprintf(stderr, "pulsar: NVFP4 linear (bf16 rows): %d -> %d at %d rows needs in %% 64, an output and 16-byte "
                        "aligned weight and rows -- refusing\n", in, out, rows);
        return -1;
    }
    const int S = pulsar_nvfp4_ksplit(in, out), R = kWarps / S;
    static bool announced = false;
    if (!__atomic_exchange_n(&announced, true, __ATOMIC_RELAXED))
        fprintf(stderr, "pulsar: NVFP4 linear (bf16 rows) = the W4A16 rows kernel (e2m1 x e4m3 decoded to bf16 in "
                        "registers, the BF16 rows kernel's order, K split by shape), 16-row blocks, M-independent at "
                        "every width (first: %d -> %d at %d rows, K split %d)\n", in, out, rows, S);
    const uint8_t *q = static_cast<const uint8_t *>(w);
    const uint8_t *sf = q + vb;
    const float *g = reinterpret_cast<const float *>(q + vb + sb);
    const dim3 grid((unsigned)((out + 16 * R - 1) / (16 * R)));
    const __nv_bfloat16 *X = reinterpret_cast<const __nv_bfloat16 *>(x_bf16);
    __nv_bfloat16 *Yb = reinterpret_cast<__nv_bfloat16 *>(y_bf16);
    for (int r0 = 0; r0 < rows; r0 += 16) {
        const int m = rows - r0 < 16 ? rows - r0 : 16;
        float *yo = y ? y + (size_t)r0 * out : nullptr;
        __nv_bfloat16 *ybo = Yb ? Yb + (size_t)r0 * out : nullptr;
        if (m > 8) launch<2>(S, grid, q, sf, g, out, in, X + (size_t)r0 * in, m, yo, ybo, stream);
        else launch<1>(S, grid, q, sf, g, out, in, X + (size_t)r0 * in, m, yo, ybo, stream);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: NVFP4 linear (bf16 rows): launch failed: %s\n", cudaGetErrorString(e));
        return -3;
    }
    return 0;
}

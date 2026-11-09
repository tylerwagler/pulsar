// SPDX-License-Identifier: MIT
/** @file
 *  NVFP4 W4A16 tensor-core GEMV / skinny GEMM against raw bf16 rows. */
// L298: NVFP4 W4A16 against raw bf16 rows (contract, layout and arithmetic in pulsar_cuda_nvfp4.h).
//
// The kernel is a memory-bound W4A16 GEMV / skinny GEMM: the weight is read once for all of the call's rows (up to
// 16 a launch, the MMA's N), decoded in registers, and multiplied on the tensor cores.  Its techniques follow
// FlashInfer's native SM12x bf16 x fp4 kernels (flashinfer/gemm/kernels/native_bf16_fp4, and b12x's block-scaled A16
// GEMM in flashinfer/experimental/b12x): the 128x4-tiled scale plane read in place, the e2m1 / e4m3 -> bf16x2
// conversions and an exact bf16x2 scale multiply, bf16 m16n8k16 MMAs with f32 accumulation, streaming (.cs) weight
// loads issued well ahead of their use, and a K split for the narrow outputs.  Two choices are ours: the weight is
// the MMA's A operand (16 weight rows) and the call's rows its N, so at M <= 8 no MMA row is padding; and a warp's 16
// rows are the ones whose scale words share two 32-byte sectors (group_row), so no scale byte is fetched in vain.
//
// A CTA is 8 warps: R = 8 / S row groups times S K slices.  Slices > 0 park their accumulators in shared memory;
// slice 0 adds them in slice order, applies the global scale and stores (f32 and / or bf16).

#include "pulsar_cuda_nvfp4.h"
#include "pulsar_cuda_arch.cuh"

#include <cuda_bf16.h>
#include <stdio.h>

namespace {

/// Warps a CTA (R = kWarps / S row groups x S K slices); the block is 32 * kWarps threads.
constexpr int kWarps = 8;
/* Measured on GB10 at the LiLiCorr drafter's shapes (2026-10-16, tests/nvfp4_rows_gate --bench): at decode widths 4
 * super-steps in flight and a K split aiming at 192 CTAs beat 1 / 2 super-steps and 96 / 384 CTAs (a few % to ~10%).
 * Neither the loads in flight (U) nor the n8 tiles a launch (NT) touch the summation order, so a row's bytes are the
 * same at every launch geometry. */
constexpr int kTargetCtas = 192;   ///< the K split's goal (pulsar_nvfp4_ksplit): four CTAs a GB10 SM
constexpr int kBlockRows = 64;     ///< the widest launch: 8 n8 tiles (prompt chunks; the weight read once a 64 rows)


/* The decode is the SM12x family's conversions (as FlashInfer's native SM12x bf16 x fp4 kernels decode,
 * flashinfer/gemm/kernels/native_bf16_fp4): cvt.rn.bf16x2.e2m1x2 turns a code byte into its two values (the low
 * nibble the low half), cvt.rn.bf16x2.e4m3x2 a scale byte into bf16 (exact), mul.rn.bf16x2 the pair by it (exact:
 * at most 2 x 4 significant bits).  They exist only in the family-specific target (sm_120f); the portable compute_120
 * PTX the build also embeds is never what a GB10 runs, and traps if anything ever JITs it. */
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1200
#define NVFP4_HAVE_CVT 1
#endif

/** One byte of the scale word as bf16x2 (both halves): `sel` the prmt selector b * 0x1111 replicating byte b. */
/** @param w   the 32-bit scale word (four e4m3 block scales)
 *  @param sel prmt selector b * 0x1111 picking byte b
 *  @return that scale as bf16x2, both halves equal (traps without the sm_12x family conversions) */
__device__ __forceinline__ uint32_t scale_bf16x2(uint32_t w, uint32_t sel) {
    uint32_t r = 0;
#ifdef NVFP4_HAVE_CVT
    asm("{ .reg .b16 sq; .reg .b32 e; prmt.b32 e, %1, %1, %2; cvt.u16.u32 sq, e; cvt.rn.bf16x2.e4m3x2 %0, sq; }"
        : "=r"(r) : "r"(w), "r"(sel));
#else
    (void)w; (void)sel;
    __trap();
#endif
    return r;
}

/** The low byte of `q` (two e2m1 codes) times the scale pair, as bf16x2. */
/** @param q  two e2m1 codes in the low byte (low nibble the low half)
 *  @param sc the scale pair (bf16x2) from scale_bf16x2
 *  @return the two decoded values times the scale, bf16x2 (traps without the sm_12x family conversions) */
__device__ __forceinline__ uint32_t dq(uint32_t q, uint32_t sc) {
    uint32_t r = 0;
#ifdef NVFP4_HAVE_CVT
    asm("{ .reg .b8 b; .reg .b32 v; cvt.u8.u32 b, %1; cvt.rn.bf16x2.e2m1x2 v, b; mul.rn.bf16x2 %0, v, %2; }"
        : "=r"(r) : "r"(q), "r"(sc));
#else
    (void)q; (void)sc;
    __trap();
#endif
    return r;
}

/* The rows a 16-row group owns: group G (of a 128-row tile T = G / 8, w = G % 8) is rows T 128 + q 32 + 4 w + j
 * (q, j < 4; logical row 4 q + j): the 16 rows whose scale words share two 32-byte sectors of the 128x4 tiling, so
 * a warp's scale loads use every byte they fetch. */
/// Physical weight row of a 16-row group's logical row (see the comment above).
/// @param G the group (8 a 128-row tile)
/// @param i the logical row within the group, 0..15
/// @return the weight row index
__device__ __forceinline__ int group_row(int G, int i) { return (G >> 3) * 128 + (i >> 2) * 32 + 4 * (G & 7) + (i & 3); }

/* One CTA: R = 8 / S row groups (blockIdx.x R + rg) of W for nr <= 8 NT activation rows, K split S ways across its
 * warps, U super-steps' loads in flight.  A super-step is 256 inputs: lane c holds inputs [64 c, 64 c + 64) of each of its two rows -- 32 contiguous
 * code bytes (a row's 128 bytes across the 4 lanes) and one scale word (the 4 blocks) -- and the same 64 inputs of
 * its activation row.  Sub-step s (16 of them) is one m16n8k16 MMA over inputs 64 c + 4 s + {0, 1 | 2, 3} of every
 * lane c, into a fresh fragment added into the row's f32 accumulator: super-steps in order, sub-steps in order. */
/** Up to kMaxSet Linears of one shape and one input, one launch (blockIdx.y picks one): each one's slice and its
 *  outputs (f32 and / or bf16, NULL = not written) for the launch's 16-row block. */
struct nvfp4_set {
    /// per Linear: its NVFP4 weight (codes, then the scale plane at scales_at, the global scale at global_at)
    const uint8_t *w[PULSAR_NVFP4_MULTI_MAX];
    /// per Linear: f32 output rows [nr][out] for this block, or NULL
    float *y[PULSAR_NVFP4_MULTI_MAX];
    /// per Linear: bf16 output rows [nr][out] for this block, or NULL
    __nv_bfloat16 *yb[PULSAR_NVFP4_MULTI_MAX];
};

/** NVFP4 W4A16 rows GEMM: Y[t][row] = gs * sum_k W[row][k] * X[t][k] for up to 8 * NT rows.  Grid
 *  (ceil(out / 128) * 8 / R, n Linears); block 32 * kWarps; static smem part[S - 1][R][32][4 NT] f32 for the K slices.
 *  @tparam NT n8 activation tiles (rows <= 8 * NT)
 *  @tparam S  K slices across the CTA's warps (1, 2, 4, 8; in % (256 S) == 0, see pulsar_nvfp4_ksplit)
 *  @tparam U  super-steps (256 inputs each) whose loads are in flight
 *  @param set       the Linears' weights and outputs (blockIdx.y picks one)
 *  @param scales_at byte offset of the 128x4-tiled e4m3 scale plane in each weight
 *  @param global_at byte offset of the f32 global scale in each weight
 *  @param out       output width (weight rows)
 *  @param in        input width (a multiple of 64)
 *  @param X         bf16 activation rows [nr][in], 16-byte aligned
 *  @param nr        activation rows in this launch */
template <int NT, int S, int U>
__global__ void __launch_bounds__(32 * kWarps)
nvfp4_rows_kernel(const __grid_constant__ nvfp4_set set, size_t scales_at, size_t global_at, int out, int in,
                  const __nv_bfloat16 *__restrict__ X, int nr) {
    constexpr int R = kWarps / S;
    const uint8_t *__restrict__ Wq = set.w[blockIdx.y];
    const uint8_t *__restrict__ SF = Wq + scales_at;
    const float *__restrict__ Gs = reinterpret_cast<const float *>(Wq + global_at);
    float *__restrict__ Y = set.y[blockIdx.y];
    __nv_bfloat16 *__restrict__ Yb = set.yb[blockIdx.y];
    __shared__ float part[S > 1 ? S - 1 : 1][R][32][NT * 4];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int rg = warp % R, kq = warp / R;
    const int G = (int)blockIdx.x * R + rg;
    const int r0 = group_row(G, g), r1 = group_row(G, g + 8);
    const bool v0 = r0 < out, v1 = r1 < out;
    const int nt = (nr + 7) >> 3;
    const int kspan = in / S, kbeg = kq * kspan, kend = kbeg + kspan;
    const uint8_t *q0 = Wq + (size_t)(v0 ? r0 : 0) * (in / 2) + 32 * c;
    const uint8_t *q1 = Wq + (size_t)(v1 ? r1 : 0) * (in / 2) + 32 * c;
    /* a row's scale word of the 64 inputs at k: 512 bytes a 4-block column of its tile; the lane's is column c on */
    const int kb4 = in / 64;
    auto sf_row = [&](int r) {
        return SF + (size_t)(r / 128) * kb4 * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + (size_t)c * 512;
    };
    const uint8_t *s0 = sf_row(v0 ? r0 : 0), *s1 = sf_row(v1 ? r1 : 0);
    const __nv_bfloat16 *xr[NT];
    bool xv[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        xv[j] = j * 8 + g < nr;
        xr[j] = X + (size_t)(xv[j] ? j * 8 + g : 0) * in + 64 * c;
    }

    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;
    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);

    for (int k0 = kbeg; k0 < kend; k0 += 256 * U) {
        uint4 a0[U][2], a1[U][2];
        uint32_t w0[U], w1[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int k = k0 + 256 * u;
            const bool lv = k + 64 * c < kend;   /* a span of 64-multiples: a last super-step's high lanes are empty */
            const uint4 *p0 = reinterpret_cast<const uint4 *>(q0 + k / 2), *p1 = reinterpret_cast<const uint4 *>(q1 + k / 2);
            a0[u][0] = v0 && lv ? __ldcs(p0) : z4;
            a0[u][1] = v0 && lv ? __ldcs(p0 + 1) : z4;
            a1[u][0] = v1 && lv ? __ldcs(p1) : z4;
            a1[u][1] = v1 && lv ? __ldcs(p1 + 1) : z4;
            w0[u] = v0 && lv ? __ldcs(reinterpret_cast<const uint32_t *>(s0 + (size_t)(k / 64) * 512)) : 0u;
            w1[u] = v1 && lv ? __ldcs(reinterpret_cast<const uint32_t *>(s1 + (size_t)(k / 64) * 512)) : 0u;
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int k = k0 + 256 * u;
            if (k >= kend) break;
            const bool lv = k + 64 * c < kend;
            uint32_t sc0[4], sc1[4];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                sc0[b] = scale_bf16x2(w0[u], 0x1111u * b);
                sc1[b] = scale_bf16x2(w1[u], 0x1111u * b);
            }
            const uint32_t c0[8] = {a0[u][0].x, a0[u][0].y, a0[u][0].z, a0[u][0].w,
                                    a0[u][1].x, a0[u][1].y, a0[u][1].z, a0[u][1].w};
            const uint32_t c1[8] = {a1[u][0].x, a1[u][0].y, a1[u][0].z, a1[u][0].w,
                                    a1[u][1].x, a1[u][1].y, a1[u][1].z, a1[u][1].w};
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                uint4 xq[NT];
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    xq[j] = j < nt && xv[j] && lv ? __ldg(reinterpret_cast<const uint4 *>(xr[j] + k) + q) : z4;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int s = 2 * q + h;   /* code bytes 2s, 2s+1: word q, bytes 2h, 2h+1; scale block s / 4 */
                    const uint32_t af[4] = {dq(c0[q] >> (16 * h), sc0[s >> 2]), dq(c1[q] >> (16 * h), sc1[s >> 2]),
                                            dq(c0[q] >> (16 * h + 8), sc0[s >> 2]), dq(c1[q] >> (16 * h + 8), sc1[s >> 2])};
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        if (j >= nt) break;
                        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        pulsar_mma_bf16_16816(d, af, h ? xq[j].z : xq[j].x, h ? xq[j].w : xq[j].y);
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc[j][e] += d[e];
                    }
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
    const float gs = *Gs;
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

/** Launches nvfp4_rows_kernel at the K split S (switch to the compiled instance).
 *  @tparam NT n8 activation tiles
 *  @tparam U  super-steps in flight
 *  @param S    K slices (1, 2, 4; anything else runs 8)
 *  @param grid the launch grid
 *  @param set  the Linears' weights and outputs
 *  @param vb   scale plane offset (the code bytes)
 *  @param gat  global scale offset
 *  @param out  output width
 *  @param in   input width
 *  @param X    bf16 activation rows
 *  @param nr   rows this launch
 *  @param st   stream */
template <int NT, int U>
void launch(int S, dim3 grid, const nvfp4_set &set, size_t vb, size_t gat, int out, int in, const __nv_bfloat16 *X,
            int nr, cudaStream_t st) {
    switch (S) {
    case 1: nvfp4_rows_kernel<NT, 1, U><<<grid, 32 * kWarps, 0, st>>>(set, vb, gat, out, in, X, nr); break;
    case 2: nvfp4_rows_kernel<NT, 2, U><<<grid, 32 * kWarps, 0, st>>>(set, vb, gat, out, in, X, nr); break;
    case 4: nvfp4_rows_kernel<NT, 4, U><<<grid, 32 * kWarps, 0, st>>>(set, vb, gat, out, in, X, nr); break;
    default: nvfp4_rows_kernel<NT, 8, U><<<grid, 32 * kWarps, 0, st>>>(set, vb, gat, out, in, X, nr); break;
    }
}

/// @param p a pointer
/// @return true when p is 16-byte aligned
bool aligned16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

} // namespace

int pulsar_nvfp4_ksplit(int in, int out) {
    int S = 1;
    while (S < kWarps && (long)(out + 127) / 128 * S < kTargetCtas && in % (256 * 2 * S) == 0)
        S *= 2;
    return S;
}

int pulsar_nvfp4_rows_multi_launch(const void *const *w, int n, const uint16_t *x_bf16, int rows, int in, int out,
                                   float *const *y, uint16_t *const *y_bf16, cudaStream_t stream) {
    PULSAR_REQUIRE_CC(120, "the NVFP4 rows linear (sm_12x e2m1 / e4m3 conversions)", -1);   /* L321: refuse below the lane's arch, before its kernels trap */
    uint64_t vb = 0, sb = 0, stride = 0;
    bool ok = w && n >= 1 && n <= PULSAR_NVFP4_MULTI_MAX && x_bf16 && rows > 0 && out > 0 &&
              pulsar_nvfp4_layout((uint64_t)in, (uint64_t)out, &vb, &sb, &stride) && aligned16(x_bf16);
    for (int j = 0; ok && j < n; ++j)
        ok = w[j] && aligned16(w[j]) && ((y && y[j]) || (y_bf16 && y_bf16[j]));
    if (!ok) {
        fprintf(stderr, "pulsar: NVFP4 linear (bf16 rows): %d Linear(s) (1..%d) %d -> %d at %d rows need in %% 64, an "
                        "output each and 16-byte aligned weights and rows -- refusing\n", n, PULSAR_NVFP4_MULTI_MAX, in,
                out, rows);
        return -1;
    }
    const int S = pulsar_nvfp4_ksplit(in, out), R = kWarps / S;
    static bool announced = false;
    if (!__atomic_exchange_n(&announced, true, __ATOMIC_RELAXED))
        fprintf(stderr, "pulsar: NVFP4 linear (bf16 rows) = the W4A16 tensor-core kernel (e2m1 x e4m3 decoded to bf16 in "
                        "registers, f32 accumulation in a fixed order, K split by shape), up to 64 rows a launch, deterministic and "
                        "M-independent at every width (first: %d x %d -> %d at %d rows, K split %d)\n", n, in, out, rows,
                S);
    /* every 16-row group of every 128-row tile, for each Linear */
    const dim3 grid((unsigned)((out + 127) / 128 * 8 / R), (unsigned)n);
    const __nv_bfloat16 *X = reinterpret_cast<const __nv_bfloat16 *>(x_bf16);
    for (int r0 = 0; r0 < rows; r0 += kBlockRows) {
        const int m = rows - r0 < kBlockRows ? rows - r0 : kBlockRows;
        nvfp4_set set{};
        for (int j = 0; j < n; ++j) {
            set.w[j] = static_cast<const uint8_t *>(w[j]);
            set.y[j] = y && y[j] ? y[j] + (size_t)r0 * out : nullptr;
            set.yb[j] = y_bf16 && y_bf16[j] ? reinterpret_cast<__nv_bfloat16 *>(y_bf16[j]) + (size_t)r0 * out : nullptr;
        }
        /* decode widths: 4 super-steps in flight; prompt widths (the weight re-read a 64 rows): 2, for registers */
        const __nv_bfloat16 *Xr = X + (size_t)r0 * in;
        if (m > 32) launch<8, 2>(S, grid, set, vb, vb + sb, out, in, Xr, m, stream);
        else if (m > 16) launch<4, 2>(S, grid, set, vb, vb + sb, out, in, Xr, m, stream);
        else if (m > 8) launch<2, 4>(S, grid, set, vb, vb + sb, out, in, Xr, m, stream);
        else launch<1, 4>(S, grid, set, vb, vb + sb, out, in, Xr, m, stream);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: NVFP4 linear (bf16 rows): launch failed: %s\n", cudaGetErrorString(e));
        return -3;
    }
    return 0;
}

int pulsar_nvfp4_rows_linear_launch(const void *w, const uint16_t *x_bf16, int rows, int in, int out, float *y,
                                    uint16_t *y_bf16, cudaStream_t stream) {
    PULSAR_REQUIRE_CC(120, "the NVFP4 rows linear (sm_12x e2m1 / e4m3 conversions)", -1);   /* L321: refuse below the lane's arch, before its kernels trap */
    return pulsar_nvfp4_rows_multi_launch(&w, 1, x_bf16, rows, in, out, &y, &y_bf16, stream);
}

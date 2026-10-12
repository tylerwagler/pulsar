// SPDX-License-Identifier: MIT
// The EXL3 dense-Linear arm (L251).  See ds4_exl3_dense.cuh for the contract
// and where the rotations go; src/engine/exl3_trellis.h for the format; the
// tile decode, codebook and Hadamard are ds4_exl3_dev.cuh's, shared with the
// routed-expert arms (exl3_moe_prefill.cu).
//
// Two launches per call (task #44: it was three -- prep, GEMV, epilogue):
//
//   1. PREP    the input side of the factorization, once per call, one warp
//              per (row, 128-block): x * suh, H128, a power-of-two prescale P
//              that puts the block's largest |value| just under 2^14, and the
//              split v = hi + lo into two fp16 planes (two roundings, ~22
//              significant bits) plus 2^-P, into the arm's workspace.  Once per
//              call instead of once per GEMV CTA: (N / 32) x splits CTAs would
//              each repeat their k-range's share (measured, task #44: an
//              in-CTA prep loses from 4 rows up, and sharing it across a
//              4-CTA cluster through distributed shared memory was 1.5-8x
//              slower still).
//   2. GEMV + EPILOGUE  A CTA owns 32 outputs (two n-tiles), one split's
//              k-range of at most kMaxBlocks Hadamard blocks, and all M <= 16
//              rows (the MMA's M: rows past M are zero, so M = 1..16 is one
//              kernel and a row's result cannot depend on the width).  Warp w
//              issues its trellis words for k-tile w of every block of the
//              range and its A tiles (cp.async from the prep planes) at entry.
//              On the tensor cores at every width: the format's tile order is
//              the mma.m16n8k16 B-fragment order (exllamav3 packed it that
//              way): lane t of a warp owns positions 8t..8t+7 of a tile, i.e.
//              tile rows (t%4)*2 + {0, 1, 8, 9} (k) of columns t/4 and t/4 + 8
//              (n), so a lane decodes its two four-position runs straight into
//              the two B registers of each n8 half, as fp16 pairs.  Two MMAs
//              per n8 half (hi, lo) into one fresh fp32 fragment -- the weight
//              is exact in fp16, so every product is exact and the result is
//              f32-class -- scaled back by 2^-P into the row's accumulator; the
//              warps reduced in warp order into the split's partial.  Then the
//              LAST CTA of a 128-output block (its 4 n-column CTAs x S splits
//              arrive on one counter, which atomicInc wraps back to zero) sums
//              the splits in split order -- fixed by the split index, never by
//              arrival -- rotates each row's 128-block and scales by svh: the
//              epilogue that was a third launch.  The counters are a static
//              device pool, one slice per (host thread, device, stream):
//              kernels on one stream are serialised, so a slice never serves
//              two live launches.
//
// A launch may carry two projections of one shape and rate over the same rows
// (gridDim.z / gridDim.y of the prep; a shared expert's gate and up), each with
// exactly the arithmetic it has alone.

#include "ds4_exl3_dense.cuh"
#include "ds4_exl3_dev.cuh"
#include "cuda/pulsar_cuda_mx.cuh"
#include "engine/exl3_trellis.h"

#include <cstdio>
#include <cuda_bf16.h>

namespace {

constexpr int kOut       = 32;   ///< outputs per CTA = two n-tiles
constexpr int kWarps     = 8;    ///< a Hadamard block's 8 k-tiles, one per warp
constexpr int kMaxBlocks = 5;    ///< Hadamard blocks per CTA k-range, at most = k-tiles per warp
constexpr int kRows      = 16;   ///< the MMA's M: every row block is 16 rows
/** CTAs the split count aims the grid at, at least: GB10 is 48 SMs, and a
 *  256-thread CTA of this kernel fits two per SM. */
constexpr int kTargetCtas = 288;
/** The prescale puts a block's largest |value| just under 2^kHiExp. */
constexpr int kHiExp     = 14;

/** The split count: a function of the weight's shape alone -- enough splits
 *  that no CTA's k-range exceeds kMaxBlocks, and enough CTAs to fill the GPU. */
int dense_splits(int K, int N) {
    const int blocks = K / EXL3_HAD_BLOCK;
    const int ctas = N / kOut;
    int s = (blocks + kMaxBlocks - 1) / kMaxBlocks;
    const int fill = (kTargetCtas + ctas - 1) / ctas;
    if (s < fill) s = fill;
    return s > blocks ? blocks : s;
}

/** Counters per (thread, device, stream) slice -- one per 128-output block of
 *  each projection, so n_proj * N <= kMaxCounters * 128 -- and the slices. */
constexpr int kMaxCounters = 1024;
constexpr int kCounterSlices = 64;
__device__ unsigned g_dense_ctr[kCounterSlices][kMaxCounters];   /* zero at load; every launch leaves it zero */

/** One projection's workspace: the split partials f32 [S][M][N], the prep planes hi and lo fp16 [M][K] and
 *  the inverse prescales f32 [M][K/128].  Every piece 16-byte aligned. */
struct Workspace {
    size_t part, hi, lo, inv, bytes;
    Workspace(int S, int M, int K, int N) {
        part = 0;
        hi = (size_t)S * M * N * sizeof(float);
        lo = hi + (size_t)M * K * sizeof(__half);
        inv = lo + (size_t)M * K * sizeof(__half);
        bytes = inv + (((size_t)M * (K / EXL3_HAD_BLOCK) * sizeof(float) + 15) & ~(size_t)15);
    }
};

/** One projection of a launch: its weight's planes, its workspace, its output, its counters.  A launch runs one
 *  or two (gridDim.z) projections of the same shape and rate over the same rows, each with exactly the
 *  arithmetic it has alone. */
struct dense_proj {
    const uint32_t *trellis;
    const __half *suh, *svh;
    float *part, *y;
    __half *hi, *lo;                 ///< the prep planes
    float *inv;
    unsigned *ctr;
};

__device__ __forceinline__ float pow2f(int e) { return __int_as_float((127 + e) << 23); }   /* e in [-126, 127] */

__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&a)[4], const __half *p) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(s));
}

/** 16 bytes global -> shared, asynchronously; src_bytes 0 writes zeros. */
__device__ __forceinline__ void cp_async16(void *smem, const void *gmem, int src_bytes) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(src_bytes));
}

/** A 16 x 16 fp16 tile in shared memory as 32 chunks of 8 halves: chunk
 *  (row, half) at 16-byte slot row * 2 + (half ^ bit 2 of row), so the 8 rows
 *  one ldmatrix 8x8 reads fall in 8 distinct 16-byte bank groups. */
__device__ __forceinline__ int tile_chunk(int row, int half) { return row * 2 + (half ^ ((row >> 2) & 1)); }

/** The PREP of one (row, 128-block), one warp: lane l owns k = 4l..4l+3 of the block (x the row's bf16 values
 *  there, su_p suh's).  hw / lw: the hi and lo fp16 pairs of the lane's four values; returns 2^-P. */
__device__ __forceinline__ float prep_block(const __nv_bfloat16 *__restrict__ x, const __half *__restrict__ su_p,
                                            uint32_t (&hw)[2], uint32_t (&lw)[2]) {
    /* L251 / ac69748f: the activation is the block input's bf16 row, NOT an A8 slot (rule 3: the producer emits
     * bf16, the consumer reads bf16).  suh, H128, the power-of-two prescale and the fp16 hi + lo split are the
     * tensor-core operand the rest of the arm takes. */
    const uint2 xw = *reinterpret_cast<const uint2 *>(x);
    const float2 b01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&xw.x));
    const float2 b23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&xw.y));
    const float xv[4] = {b01.x, b01.y, b23.x, b23.y};
    const uint2 su2 = *reinterpret_cast<const uint2 *>(su_p);
    const __half2 s01 = *reinterpret_cast<const __half2 *>(&su2.x);
    const __half2 s23 = *reinterpret_cast<const __half2 *>(&su2.y);
    const float su[4] = {__low2float(s01), __high2float(s01), __low2float(s23), __high2float(s23)};
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = xv[e] * su[e];
    exl3dev::had128(v);
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    /* amax = m * 2^e, m in [0.5, 1): P = kHiExp - e puts it under 2^kHiExp */
    int p = amax > 0.0f ? kHiExp - (((__float_as_int(amax) >> 23) & 0xff) - 126) : 0;
    p = max(-100, min(100, p));
    const float up = pow2f(p);
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const float a0 = v[2 * e] * up, a1 = v[2 * e + 1] * up;
        const __half2 h2 = __floats2half2_rn(a0, a1);
        const __half2 l2 = __floats2half2_rn(a0 - __low2float(h2), a1 - __high2float(h2));
        hw[e] = *reinterpret_cast<const uint32_t *>(&h2);
        lw[e] = *reinterpret_cast<const uint32_t *>(&l2);
    }
    return pow2f(-p);
}

/* 1. PREP: one warp per (row, 128-block), projection blockIdx.y. */
__global__ void __launch_bounds__(256)
exl3_dense_prep_kernel(const dense_proj p0, const dense_proj p1, const __nv_bfloat16 *__restrict__ xb, int rows, int K) {
    const dense_proj &p = blockIdx.y ? p1 : p0;
    const int64_t task = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int blocks = K >> 7;
    if (task >= (int64_t)rows * blocks) return;            /* whole warps: task is per warp */
    const int m = (int)(task / blocks), blk = (int)(task - (int64_t)m * blocks);
    const int k = blk * EXL3_HAD_BLOCK + (threadIdx.x & 31) * 4;
    uint32_t hw[2], lw[2];
    const float inv = prep_block(xb + (size_t)m * (size_t)K + (size_t)k, p.suh + k, hw, lw);
    const size_t off = (size_t)m * K + k;
    *reinterpret_cast<uint2 *>(p.hi + off) = make_uint2(hw[0], hw[1]);
    *reinterpret_cast<uint2 *>(p.lo + off) = make_uint2(lw[0], lw[1]);
    if ((threadIdx.x & 31) == 0) p.inv[(size_t)m * blocks + blk] = inv;
}

/* 2. GEMV + EPILOGUE.  grid (N / 32, S, projections): CTA (x, split) owns outputs 32x..32x+31 and the split's
 * k-range. */
template <int K2>
__global__ void __launch_bounds__(kOut * kWarps)
exl3_dense_kernel(const dense_proj p0, const dense_proj p1, const __nv_bfloat16 *__restrict__ xb, int M, int K, int N,
                  int S) {
    const dense_proj &p = blockIdx.z ? p1 : p0;
    constexpr int W32 = exl3dev::Rate<K2>::words32;
    constexpr int kTile = kRows * 16;                             /* halves per 16 x 16 A tile */
    static_assert(kWarps * kRows * kOut * 4 <= kWarps * kMaxBlocks * 2 * kTile * 2, "the reduction view must fit the A slots");
    __shared__ __align__(16) __half s_a[kWarps][kMaxBlocks][2][kTile];   /* per k-tile of each block: hi, lo */
    __shared__ float s_inv[kMaxBlocks][kRows];
    __shared__ bool s_last;

    const int lane   = threadIdx.x;
    const int warp   = threadIdx.y;
    const int tid    = warp * kOut + lane;
    const int ntn    = N >> 4;
    const int split  = blockIdx.y;
    const int nr     = M;                                         /* 1..kRows */
    const int blocks = K >> 7;
    const int b_lo   = (int)((int64_t)split * blocks / S);
    const int nb     = (int)((int64_t)(split + 1) * blocks / S) - b_lo;   /* 1..kMaxBlocks */
    const int g      = lane >> 2;                                 /* fragment rows g, g + 8 */

    /* the lane's two runs of every tile: positions 8t..8t+3 (n8 half 0) and
     * 8t+4..8t+7 (half 1) -- where each run's window sits, once for every tile */
    int lo[2], hi[2], sh[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) exl3dev::run4_window<K2>(8 * lane + 4 * h, lo[h], hi[h], sh[h]);

    /* 1. this warp's trellis words, k-tile `warp` of each block, both n-tiles and halves */
    uint32_t wd[kMaxBlocks][2][2][2];
#pragma unroll
    for (int i = 0; i < kMaxBlocks; ++i) {
        if (i < nb) {
            const int kt = (b_lo + i) * 8 + warp;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const uint32_t *t = p.trellis + ((size_t)kt * (size_t)ntn + (size_t)(blockIdx.x * 2 + j)) * W32;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    wd[i][j][h][0] = t[lo[h]];
                    wd[i][j][h][1] = t[hi[h]];
                }
            }
        }
    }

    /* the A tiles: tile (w, i) is k-tile w of block i of the range, rows past M zero */
    {
        /* warp w's own tiles from the prep planes (lane = chunk (row l/2, half l%2); rows past M zero) */
        const int crow = lane >> 1, chalf = lane & 1;
        const bool creal = crow < nr;
#pragma unroll
        for (int i = 0; i < kMaxBlocks; ++i) {
            if (i < nb) {
                const int kt = (b_lo + i) * 8 + warp;
                const size_t src = (size_t)(creal ? crow : 0) * K + (size_t)kt * 16 + chalf * 8;
                cp_async16(&s_a[warp][i][0][tile_chunk(crow, chalf) * 8], p.hi + src, creal ? 16 : 0);
                cp_async16(&s_a[warp][i][1][tile_chunk(crow, chalf) * 8], p.lo + src, creal ? 16 : 0);
            }
        }
        for (int t = tid; t < nr * nb; t += kOut * kWarps) {
            const int m = t / nb, i = t - m * nb;
            s_inv[i][m] = p.inv[(size_t)m * blocks + b_lo + i];
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();

    /* 3. per k-tile: the A fragments (ldmatrix x4: lane l addresses row
     * (l%8) + 8 (bit 3 of l), k half bit 4 of l), each B pair decoded once,
     * two MMAs per n8 half into a fresh fragment, scaled back into the rows'
     * accumulators.  Lane (g, c = lane%4) holds rows g, g + 8 and columns 2c,
     * 2c + 1 of each n8 half. */
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8, ahalf = (lane >> 4) & 1;
    float acc[2][2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[j][h][e] = 0.0f;
#pragma unroll
    for (int i = 0; i < kMaxBlocks; ++i) {
        if (i < nb) {
            const float inv_a = g < nr ? s_inv[i][g] : 0.0f;
            const float inv_b = g + 8 < nr ? s_inv[i][g + 8] : 0.0f;
            uint32_t ahi[4], alo[4];
            ldmatrix_x4(ahi, &s_a[warp][i][0][tile_chunk(arow, ahalf) * 8]);
            ldmatrix_x4(alo, &s_a[warp][i][1][tile_chunk(arow, ahalf) * 8]);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t st[4];
                    exl3dev::run4_split<K2>(EXL3_RUN4_JOIN(wd[i][j][h][0], wd[i][j][h][1], sh[h]), 8 * lane + 4 * h, st);
                    const uint32_t b0 = exl3dev::mul1_pair(st[0], st[1]);   /* k rows 2c, 2c+1 */
                    const uint32_t b1 = exl3dev::mul1_pair(st[2], st[3]);   /* k rows 2c+8, 2c+9 */
                    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    mma_16816(d, ahi, b0, b1);
                    mma_16816(d, alo, b0, b1);
                    acc[j][h][0] = fmaf(d[0], inv_a, acc[j][h][0]);
                    acc[j][h][1] = fmaf(d[1], inv_a, acc[j][h][1]);
                    acc[j][h][2] = fmaf(d[2], inv_b, acc[j][h][2]);
                    acc[j][h][3] = fmaf(d[3], inv_b, acc[j][h][3]);
                }
            }
        }
    }

    /* cross-warp reduction in warp order, one warp per row, over the A slots
     * (every warp is past its last read of them), into the split's partial */
    __syncthreads();
    float *s_red = reinterpret_cast<float *>(&s_a[0][0][0][0]);   /* [warp][row][32 outputs] */
    const int c2 = (lane & 3) * 2;
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int rr = g + 8 * (e >> 1), col = j * 16 + h * 8 + c2 + (e & 1);
                s_red[(warp * kRows + rr) * kOut + col] = acc[j][h][e];
            }
    __syncthreads();
    for (int r = warp; r < nr; r += kWarps) {
        float z = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) z += s_red[(w * kRows + r) * kOut + lane];
        p.part[((size_t)split * (size_t)M + (size_t)r) * (size_t)N + (size_t)(blockIdx.x * kOut + lane)] = z;
    }

    /* 4. the last of the 128-output block's 4 x S CTAs finishes it: splits in split order, H128, svh */
    __threadfence();
    __syncthreads();
    const int ob = blockIdx.x >> 2;                               /* the 128-output block */
    if (tid == 0) s_last = atomicInc(&p.ctr[ob], 4u * (unsigned)S - 1u) == 4u * (unsigned)S - 1u;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    const int n0 = ob * 128 + lane * 4;
    for (int m = warp; m < nr; m += kWarps) {
        float z[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int s = 0; s < S; ++s) {
            const float4 v = __ldcg(reinterpret_cast<const float4 *>(p.part + ((size_t)s * (size_t)M + (size_t)m) * (size_t)N + n0));
            z[0] += v.x; z[1] += v.y; z[2] += v.z; z[3] += v.w;
        }
        exl3dev::had128(z);
        const __half2 v01 = *reinterpret_cast<const __half2 *>(p.svh + n0);
        const __half2 v23 = *reinterpret_cast<const __half2 *>(p.svh + n0 + 2);
        *reinterpret_cast<float4 *>(p.y + (size_t)m * (size_t)N + n0) =
            make_float4(z[0] * __low2float(v01), z[1] * __high2float(v01), z[2] * __low2float(v23), z[3] * __high2float(v23));
    }
}

/** The calling thread's counter slice for (device, stream): kernels on one
 *  stream are serialised and two threads never share a slice, so no two live
 *  launches meet on a counter.  nullptr once the pool is spent (a refusal). */
unsigned *counter_slice(cudaStream_t stream) {
    struct slice { int dev; cudaStream_t s; unsigned *p; };
    thread_local slice t_slices[8];
    thread_local int t_n = 0;
    static int s_next = 0;   /* slices handed out to every thread */
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return nullptr;
    for (int i = 0; i < t_n; ++i)
        if (t_slices[i].dev == dev && t_slices[i].s == stream) return t_slices[i].p;
    if (t_n == 8) return nullptr;
    const int slot = __atomic_fetch_add(&s_next, 1, __ATOMIC_RELAXED);
    void *base = nullptr;
    if (slot >= kCounterSlices || cudaGetSymbolAddress(&base, g_dense_ctr) != cudaSuccess) return nullptr;
    t_slices[t_n] = slice{dev, stream, static_cast<unsigned *>(base) + (size_t)slot * kMaxCounters};
    return t_slices[t_n++].p;
}

void gemv_launch(int k2, const dim3 &grid, const dim3 &block, cudaStream_t stream, const dense_proj &p0,
                 const dense_proj &p1, const __nv_bfloat16 *xb, int M, int K, int N, int S) {
    switch (k2) {
    case 4:  exl3_dense_kernel<4><<<grid, block, 0, stream>>>(p0, p1, xb, M, K, N, S); break;
    case 6:  exl3_dense_kernel<6><<<grid, block, 0, stream>>>(p0, p1, xb, M, K, N, S); break;
    case 8:  exl3_dense_kernel<8><<<grid, block, 0, stream>>>(p0, p1, xb, M, K, N, S); break;
    case 10: exl3_dense_kernel<10><<<grid, block, 0, stream>>>(p0, p1, xb, M, K, N, S); break;
    case 12: exl3_dense_kernel<12><<<grid, block, 0, stream>>>(p0, p1, xb, M, K, N, S); break;
    default: exl3_dense_kernel<16><<<grid, block, 0, stream>>>(p0, p1, xb, M, K, N, S); break;  /* k2 == 16, checked */
    }
}

} // namespace

bool ds4_exl3_dense_rate_supported(int k2) { return exl3_arm_has_rate(EXL3_ARM_DENSE, k2); }

int ds4_exl3_dense_splits(int K, int N) {
    if (K <= 0 || N <= 0 || K % EXL3_HAD_BLOCK || N % EXL3_HAD_BLOCK) return 0;
    return dense_splits(K, N);
}

size_t ds4_exl3_dense_workspace_bytes(int M, int K, int N) {
    const int S = ds4_exl3_dense_splits(K, N);
    if (S == 0 || M <= 0) return 0;
    return Workspace(S, M < kRows ? M : kRows, K, N).bytes;   /* M > kRows is the prefill GEMM's: none */
}

namespace {

/* The launch of n_proj (1 or 2) projections of one (K, N, k2) over the same M <= kRows rows. */
int dense_launch(const char *tag, const void *const w[2], int n_proj, int k2, const void *xb, float *const y[2], int M,
                 int K, int N, void *workspace, size_t workspace_bytes, cudaStream_t stream, bool prompt) {
    uint64_t trellis_bytes = 0, scale_bytes = 0, stride = 0;
    if (!w[0] || (n_proj == 2 && !w[1]) || !xb || !y[0] || (n_proj == 2 && !y[1]) || !workspace || M <= 0) {
        fprintf(stderr, "%s: null pointer or M=%d -- refusing\n", tag, M);
        return -1;
    }
    if (!exl3_expert_layout((uint64_t)(K > 0 ? K : 0), (uint64_t)(N > 0 ? N : 0), k2, &trellis_bytes, &scale_bytes, &stride)) {
        fprintf(stderr, "%s: K=%d N=%d k2=%d is not an EXL3 layout (both dims %% 128, a valid rate) -- refusing\n",
                tag, K, N, k2);
        return -1;
    }
    if (!ds4_exl3_dense_rate_supported(k2)) {
        fprintf(stderr, "%s: rate k2=%d has no instance (K = 2, 3, 4, 5, 6, 8) -- refusing\n", tag, k2);
        return -1;
    }
    for (int j = 0; j < n_proj; ++j)
        if ((uintptr_t)w[j] % 16 || (uintptr_t)y[j] % 16) {
            fprintf(stderr, "%s: w / y must be 16-byte aligned -- refusing\n", tag);
            return -1;
        }
    if ((uintptr_t)xb % 8 || (uintptr_t)workspace % 16) {
        fprintf(stderr, "%s: the workspace must be 16-byte aligned and xb 8-byte (the uint2 row loads) -- refusing\n", tag);
        return -1;
    }
    /* L251: a prompt chunk takes the tensor-core GEMM (qwen_exl3_dense_prefill.cu): this
     * arm's GEMV re-decodes the trellis per 16 rows and re-reads the activation planes per 32 outputs,
     * a decode kernel doing a GEMM's work.  Decode widths keep the GEMV bit for bit. */
    if (prompt || M > kRows) {
        for (int j = 0; j < n_proj; ++j) {
            const int rc = qwen_exl3_dense_prefill_launch(w[j], k2, xb, y[j], M, K, N, trellis_bytes, stream);
            if (rc) return rc;
        }
        return 0;
    }
    const size_t one = ds4_exl3_dense_workspace_bytes(M, K, N);
    if (workspace_bytes < (size_t)n_proj * one) {
        fprintf(stderr, "%s: workspace %zu bytes < %zu for %d projection(s) at M=%d K=%d N=%d -- refusing\n", tag,
                workspace_bytes, (size_t)n_proj * one, n_proj, M, K, N);
        return -1;
    }
    if (n_proj * (N / 128) > kMaxCounters) {
        fprintf(stderr, "%s: %d x N=%d is past the epilogue's %d counters -- refusing\n", tag, n_proj, N, kMaxCounters);
        return -1;
    }
    unsigned *ctr = counter_slice(stream);
    if (!ctr) {
        fprintf(stderr, "%s: no epilogue counter slice left for this (thread, device, stream) -- the pool holds %d "
                        "-- refusing\n", tag, kCounterSlices);
        return -1;
    }
    const int S = dense_splits(K, N);
    const Workspace L(S, M, K, N);
    dense_proj p[2] = {};
    for (int j = 0; j < n_proj; ++j) {
        uint8_t *ws = static_cast<uint8_t *>(workspace) + (size_t)j * one;
        const __half *suh = reinterpret_cast<const __half *>(static_cast<const uint8_t *>(w[j]) + trellis_bytes);
        p[j] = dense_proj{static_cast<const uint32_t *>(w[j]), suh, suh + K, reinterpret_cast<float *>(ws + L.part), y[j],
                          reinterpret_cast<__half *>(ws + L.hi), reinterpret_cast<__half *>(ws + L.lo),
                          reinterpret_cast<float *>(ws + L.inv), ctr + (size_t)j * (size_t)(N / 128)};
    }
    const __nv_bfloat16 *xb16 = static_cast<const __nv_bfloat16 *>(xb);
    const dim3 grid((unsigned)(N / kOut), (unsigned)S, (unsigned)n_proj), block(kOut, kWarps, 1);
    const int64_t tasks = (int64_t)M * (K / EXL3_HAD_BLOCK);
    exl3_dense_prep_kernel<<<dim3((unsigned)((tasks + 7) / 8), (unsigned)n_proj), 256, 0, stream>>>(p[0], p[1], xb16, M, K);
    gemv_launch(k2, grid, block, stream, p[0], p[1], xb16, M, K, N, S);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

} // namespace

int ds4_exl3_dense_launch(const void *w, int k2, const void *xb, float *y,
                          int M, int K, int N, void *workspace, size_t workspace_bytes, cudaStream_t stream, bool prompt) {
    const void *const ws[2] = {w, nullptr};
    float *const ys[2] = {y, nullptr};
    return dense_launch("ds4_exl3_dense_launch", ws, 1, k2, xb, ys, M, K, N, workspace, workspace_bytes, stream, prompt);
}

int ds4_exl3_dense_pair_launch(const void *w0, const void *w1, int k2, const void *xb, float *y0, float *y1,
                               int M, int K, int N, void *workspace, size_t workspace_bytes, cudaStream_t stream,
                               bool prompt) {
    const void *const ws[2] = {w0, w1};
    float *const ys[2] = {y0, y1};
    return dense_launch("ds4_exl3_dense_pair_launch", ws, 2, k2, xb, ys, M, K, N, workspace, workspace_bytes, stream,
                        prompt);
}

// SPDX-License-Identifier: MIT
// The EXL3 dense-Linear arm (L251).  See ds4_exl3_dense.cuh for the contract
// and where the rotations go; src/engine/exl3_trellis.h for the format; the
// tile decode, codebook and Hadamard are ds4_exl3_dev.cuh's, shared with the
// routed-expert arm (ds4_exl3_gemv.cu).
//
// Three launches per slab of rows:
//
//   1. PREP (the input side of the factorization, once per row and 128-block):
//      x * scale * suh, H128, then a power-of-two prescale P that puts the
//      block's largest |value| just under 2^14, and the split v = hi + lo into
//      two fp16 planes (two roundings, ~22 significant bits) plus 2^-P per
//      (row, block), into the arm's workspace.  Doing it once per call instead
//      of once per GEMV CTA is the difference between one pass over M x K and
//      (N / 32) x splits passes (1280 for DeltaNet in_proj_qkv).
//   2. GEMV, on the tensor cores at EVERY width.  The format's tile order is
//      the mma.m16n8k16 B-fragment order (exllamav3 packed it that way): lane t
//      of a warp owns positions 8t..8t+7 of a tile, i.e. tile rows (t%4)*2 +
//      {0, 1, 8, 9} (k) of columns t/4 and t/4 + 8 (n), so a lane decodes its
//      two four-position runs straight into the two B registers of each n8
//      half, as fp16 pairs.  The A operand is a 16-row block of the prep
//      planes: a row block is ALWAYS 16 rows (rows past M read as zero), so
//      M = 1..16 run one kernel and a row's result cannot depend on the width.
//      Two MMAs per n8 half (hi, lo) into one fresh fp32 fragment -- the weight
//      is exact in fp16, so every product is exact and the result is f32-class
//      -- scaled back by 2^-P into the row's accumulator.  A CTA owns 32
//      outputs (two n-tiles), one split's k-range of at most kMaxBlocks blocks
//      and one row block; warp w owns k-tile w of every block of the range and
//      issues ALL its loads at entry -- trellis words into registers, its A
//      tiles into a warp-private shared slot by cp.async -- so the whole
//      k-range is in flight at once and nothing synchronises the CTA until
//      the fixed-order cross-warp reduction into the split partial.
//   3. EPILOGUE: the splits summed in split order, the output H128, svh.

#include "ds4_exl3_dense.cuh"
#include "ds4_exl3_dev.cuh"
#include "cuda/pulsar_cuda_mx.cuh"
#include "engine/exl3_trellis.h"

#include <cstdio>

namespace {

constexpr int kOut       = 32;   ///< outputs per CTA = two n-tiles
constexpr int kWarps     = 8;    ///< a Hadamard block's 8 k-tiles, one per warp
constexpr int kMaxBlocks = 5;    ///< Hadamard blocks per CTA k-range, at most = k-tiles per warp
constexpr int kRows      = 16;   ///< the MMA's M: every row block is 16 rows
/** Rows per launch triple: prefill rows go through slab by slab, so the
 *  workspace stays a few MB-tens of MB at any M. */
constexpr int kSlabRows  = 128;
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

/** The workspace of one slab of `rows`: the split partials, the hi and lo
 *  planes, the per-(row, block) inverse prescale.  Every piece 16-byte
 *  aligned (K and N are multiples of 128). */
struct Workspace {
    size_t part, hi, lo, inv, bytes;
    Workspace(int S, int rows, int K, int N) {
        part = 0;
        hi   = part + (size_t)S * rows * N * sizeof(float);
        lo   = hi + (size_t)rows * K * sizeof(__half);
        inv  = lo + (size_t)rows * K * sizeof(__half);
        bytes = inv + (size_t)rows * (K / EXL3_HAD_BLOCK) * sizeof(float);
    }
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

/* 1. PREP: one warp per (row, 128-block) of the slab; lane l owns k = 4l..4l+3. */
__global__ void __launch_bounds__(256)
exl3_dense_prep_kernel(const __half *__restrict__ suh, const uint8_t *__restrict__ xq, const uint8_t *__restrict__ sx,
                       __half *__restrict__ xhi, __half *__restrict__ xlo, float *__restrict__ inv,
                       int x_row0, int rows, int K) {
    const int64_t task = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int blocks = K >> 7;
    if (task >= (int64_t)rows * blocks) return;            /* whole warps: task is per warp */
    const int m = (int)(task / blocks), blk = (int)(task - (int64_t)m * blocks);
    const int lane = threadIdx.x & 31;
    const int k = blk * EXL3_HAD_BLOCK + lane * 4;
    const int row = x_row0 + m;                            /* the slot's row */
    const uint32_t q = *reinterpret_cast<const uint32_t *>(xq + (size_t)row * (size_t)K + (size_t)k);
    const uint32_t sb = sx[pulsar_mx_sfoff(row, k >> 5, pulsar_mx_kbp(K))];
    const uint2 su2 = *reinterpret_cast<const uint2 *>(suh + k);
    const float sc = __int_as_float((int)(sb << 23));      /* ue8m0 -> 2^(b-127), the MXFP8 GEMV's form */
    const __half2 s01 = *reinterpret_cast<const __half2 *>(&su2.x);
    const __half2 s23 = *reinterpret_cast<const __half2 *>(&su2.y);
    const float su[4] = {__low2float(s01), __high2float(s01), __low2float(s23), __high2float(s23)};
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = exl3dev::e4m3_to_f32((uint8_t)(q >> (8 * e))) * sc * su[e];
    exl3dev::had128(v);
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    /* amax = m * 2^e, m in [0.5, 1): P = kHiExp - e puts it under 2^kHiExp */
    int p = amax > 0.0f ? kHiExp - (((__float_as_int(amax) >> 23) & 0xff) - 126) : 0;
    p = max(-100, min(100, p));
    const float up = pow2f(p);
    uint32_t hw[2], lw[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const float a0 = v[2 * e] * up, a1 = v[2 * e + 1] * up;
        const __half2 h2 = __floats2half2_rn(a0, a1);
        const __half2 l2 = __floats2half2_rn(a0 - __low2float(h2), a1 - __high2float(h2));
        hw[e] = *reinterpret_cast<const uint32_t *>(&h2);
        lw[e] = *reinterpret_cast<const uint32_t *>(&l2);
    }
    const size_t off = (size_t)m * K + k;
    *reinterpret_cast<uint2 *>(xhi + off) = make_uint2(hw[0], hw[1]);
    *reinterpret_cast<uint2 *>(xlo + off) = make_uint2(lw[0], lw[1]);
    if (lane == 0) inv[(size_t)m * blocks + blk] = pow2f(-p);
}

/* 2. GEMV */
template <int K2>
__global__ void __launch_bounds__(kOut * kWarps)
exl3_dense_gemv_kernel(const uint32_t *__restrict__ trellis,
                       const __half *__restrict__ xhi, const __half *__restrict__ xlo, const float *__restrict__ inv,
                       float *__restrict__ part, int M, int K, int N, int S) {
    constexpr int W32 = exl3dev::Rate<K2>::words32;
    constexpr int kTile = kRows * 16;                             /* halves per 16 x 16 A tile */
    static_assert(kWarps * kRows * kOut * 4 <= kWarps * kMaxBlocks * 2 * kTile * 2, "the reduction view must fit the A slots");
    __shared__ __align__(16) __half s_a[kWarps][kMaxBlocks][2][kTile];   /* per warp, per k-tile: hi, lo */

    const int lane   = threadIdx.x;
    const int warp   = threadIdx.y;
    const int ntn    = N >> 4;
    const int split  = blockIdx.y;
    const int row0   = blockIdx.z * kRows;
    const int nr     = min(kRows, M - row0);
    const int blocks = K >> 7;
    const int b_lo   = (int)((int64_t)split * blocks / S);
    const int nb     = (int)((int64_t)(split + 1) * blocks / S) - b_lo;   /* 1..kMaxBlocks */
    const int g      = lane >> 2;                                 /* fragment rows g, g + 8 */

    /* the lane's two runs of every tile: positions 8t..8t+3 (n8 half 0) and
     * 8t+4..8t+7 (half 1) -- where each run's window sits, once for every tile */
    int lo[2], hi[2], sh[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) exl3dev::run4_window<K2>(8 * lane + 4 * h, lo[h], hi[h], sh[h]);

    /* 1. every load this warp makes, issued now: per k-tile (`warp` of each
     * block) the trellis words of both n-tiles and halves, the A tile's hi and
     * lo chunks (lane = chunk (row l/2, half l%2); rows past M read as zero),
     * and the two inverse prescales of the lane's fragment rows */
    uint32_t wd[kMaxBlocks][2][2][2];
    float inv_a[kMaxBlocks], inv_b[kMaxBlocks];
    const int crow = lane >> 1, chalf = lane & 1;
    const bool creal = crow < nr;
#pragma unroll
    for (int i = 0; i < kMaxBlocks; ++i) {
        if (i < nb) {
            const int kt = (b_lo + i) * 8 + warp;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const uint32_t *t = trellis + ((size_t)kt * (size_t)ntn + (size_t)(blockIdx.x * 2 + j)) * W32;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    wd[i][j][h][0] = t[lo[h]];
                    wd[i][j][h][1] = t[hi[h]];
                }
            }
            const size_t src = (size_t)(row0 + (creal ? crow : 0)) * K + (size_t)kt * 16 + chalf * 8;
            cp_async16(&s_a[warp][i][0][tile_chunk(crow, chalf) * 8], xhi + src, creal ? 16 : 0);
            cp_async16(&s_a[warp][i][1][tile_chunk(crow, chalf) * 8], xlo + src, creal ? 16 : 0);
            inv_a[i] = g < nr     ? inv[(size_t)(row0 + g) * blocks + b_lo + i] : 0.0f;
            inv_b[i] = g + 8 < nr ? inv[(size_t)(row0 + g + 8) * blocks + b_lo + i] : 0.0f;
        }
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
    __syncwarp();

    /* 2. per k-tile: the A fragments (ldmatrix x4: lane l addresses row
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
                    acc[j][h][0] = fmaf(d[0], inv_a[i], acc[j][h][0]);
                    acc[j][h][1] = fmaf(d[1], inv_a[i], acc[j][h][1]);
                    acc[j][h][2] = fmaf(d[2], inv_b[i], acc[j][h][2]);
                    acc[j][h][3] = fmaf(d[3], inv_b[i], acc[j][h][3]);
                }
            }
        }
    }

    /* 3. cross-warp reduction in warp order, one warp per row, over the A
     * slots (every warp is past its last read of them) */
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
        part[((size_t)split * (size_t)M + (size_t)(row0 + r)) * (size_t)N + (size_t)(blockIdx.x * kOut + lane)] = z;
    }
}

/* 3. EPILOGUE: one warp per (row, 128-block of outputs): the splits summed in
 * split order, the output Hadamard, svh.  Lane l owns outputs 4l..4l+3. */
__global__ void __launch_bounds__(256)
exl3_dense_epilogue_kernel(const float *__restrict__ part, const __half *__restrict__ svh,
                           float *__restrict__ y, int M, int N, int S) {
    const int64_t task = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int n_blk = N >> 7;
    if (task >= (int64_t)M * n_blk) return;              /* whole warps: task is per warp */
    const int m = (int)(task / n_blk);
    const int blk = (int)(task - (int64_t)m * n_blk);
    const int lane = threadIdx.x & 31;
    const int n0 = blk * 128 + lane * 4;
    float z[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int s = 0; s < S; ++s) {
        const float4 p = *reinterpret_cast<const float4 *>(part + ((size_t)s * (size_t)M + (size_t)m) * (size_t)N + n0);
        z[0] += p.x; z[1] += p.y; z[2] += p.z; z[3] += p.w;
    }
    exl3dev::had128(z);
    const __half2 v01 = *reinterpret_cast<const __half2 *>(svh + n0);
    const __half2 v23 = *reinterpret_cast<const __half2 *>(svh + n0 + 2);
    *reinterpret_cast<float4 *>(y + (size_t)m * (size_t)N + n0) =
        make_float4(z[0] * __low2float(v01), z[1] * __high2float(v01), z[2] * __low2float(v23), z[3] * __high2float(v23));
}

} // namespace

bool ds4_exl3_dense_rate_supported(int k2) { return k2 == 4 || k2 == 6 || k2 == 8 || k2 == 10; }

int ds4_exl3_dense_splits(int K, int N) {
    if (K <= 0 || N <= 0 || K % EXL3_HAD_BLOCK || N % EXL3_HAD_BLOCK) return 0;
    return dense_splits(K, N);
}

size_t ds4_exl3_dense_workspace_bytes(int M, int K, int N) {
    const int S = ds4_exl3_dense_splits(K, N);
    if (S == 0 || M <= 0) return 0;
    return Workspace(S, M < kSlabRows ? M : kSlabRows, K, N).bytes;
}

int ds4_exl3_dense_launch(const void *w, int k2, const void *xq, const void *sx, float *y,
                          int M, int K, int N, void *workspace, size_t workspace_bytes, cudaStream_t stream) {
    const char *tag = "ds4_exl3_dense_launch";
    uint64_t trellis_bytes = 0, scale_bytes = 0, stride = 0;
    if (!w || !xq || !sx || !y || !workspace || M <= 0) {
        fprintf(stderr, "%s: null pointer or M=%d -- refusing\n", tag, M);
        return -1;
    }
    if (!exl3_expert_layout((uint64_t)(K > 0 ? K : 0), (uint64_t)(N > 0 ? N : 0), k2, &trellis_bytes, &scale_bytes, &stride)) {
        fprintf(stderr, "%s: K=%d N=%d k2=%d is not an EXL3 layout (both dims %% 128, a valid rate) -- refusing\n",
                tag, K, N, k2);
        return -1;
    }
    if (!ds4_exl3_dense_rate_supported(k2)) {
        fprintf(stderr, "%s: rate k2=%d has no instance (K = 2, 3, 4, 5) -- refusing\n", tag, k2);
        return -1;
    }
    if ((uintptr_t)w % 16 || (uintptr_t)xq % 16 || (uintptr_t)y % 16 || (uintptr_t)workspace % 16) {
        fprintf(stderr, "%s: w / xq / y / workspace must be 16-byte aligned -- refusing\n", tag);
        return -1;
    }
    const size_t need = ds4_exl3_dense_workspace_bytes(M, K, N);
    if (workspace_bytes < need) {
        fprintf(stderr, "%s: workspace %zu bytes < %zu for M=%d K=%d N=%d -- refusing\n", tag, workspace_bytes, need, M, K, N);
        return -1;
    }
    const int S = dense_splits(K, N);
    const uint32_t *t = static_cast<const uint32_t *>(w);
    const __half *suh = reinterpret_cast<const __half *>(static_cast<const uint8_t *>(w) + trellis_bytes);
    const __half *svh = suh + K;
    const uint8_t *q = static_cast<const uint8_t *>(xq), *s = static_cast<const uint8_t *>(sx);
    uint8_t *ws = static_cast<uint8_t *>(workspace);
    const dim3 block(kOut, kWarps, 1);
    /* rows past a slab of kSlabRows go through the same three kernels slab by
     * slab (a row's arithmetic does not depend on its slab), so the workspace
     * never outgrows one slab */
    for (int m0 = 0; m0 < M; m0 += kSlabRows) {
        const int rows = M - m0 < kSlabRows ? M - m0 : kSlabRows;
        const Workspace L(S, rows, K, N);
        float *part = reinterpret_cast<float *>(ws + L.part);
        __half *xhi = reinterpret_cast<__half *>(ws + L.hi), *xlo = reinterpret_cast<__half *>(ws + L.lo);
        float *inv = reinterpret_cast<float *>(ws + L.inv);
        const int64_t prep_tasks = (int64_t)rows * (K / EXL3_HAD_BLOCK);
        exl3_dense_prep_kernel<<<(unsigned)((prep_tasks + 7) / 8), 256, 0, stream>>>(suh, q, s, xhi, xlo, inv, m0, rows, K);
        const dim3 grid((unsigned)(N / kOut), (unsigned)S, (unsigned)((rows + kRows - 1) / kRows));
        switch (k2) {
        case 4:  exl3_dense_gemv_kernel<4><<<grid, block, 0, stream>>>(t, xhi, xlo, inv, part, rows, K, N, S); break;
        case 6:  exl3_dense_gemv_kernel<6><<<grid, block, 0, stream>>>(t, xhi, xlo, inv, part, rows, K, N, S); break;
        case 8:  exl3_dense_gemv_kernel<8><<<grid, block, 0, stream>>>(t, xhi, xlo, inv, part, rows, K, N, S); break;
        default: exl3_dense_gemv_kernel<10><<<grid, block, 0, stream>>>(t, xhi, xlo, inv, part, rows, K, N, S); break;  /* k2 == 10, checked above */
        }
        const int64_t tasks = (int64_t)rows * (N >> 7);
        exl3_dense_epilogue_kernel<<<(unsigned)((tasks + 7) / 8), 256, 0, stream>>>(part, svh, y + (size_t)m0 * N, rows, N, S);
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

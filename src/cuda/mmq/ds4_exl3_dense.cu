// SPDX-License-Identifier: MIT
/** @file
 *  The EXL3 dense-Linear arm (L251): prep, then the tensor-core GEMV with its epilogue. */
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
//   2. GEMV + EPILOGUE  A CTA owns 32 outputs (two n-tiles) over the whole
//              of K and all M <= 16 rows (the MMA's M: rows past M are zero, so
//              M = 1..16 is one kernel and a row's result cannot depend on the
//              width).  L338 (exllamav3's GEMV structure, exl3_gemv_kernel.cuh):
//              16 warps, warp w taking k-tile w % 8 of every other 128-block
//              through a register prefetch ring of coalesced trellis words (one
//              word per lane per 32 of a tile, each lane's window by shuffle),
//              its A fragments straight from the prep planes one step ahead;
//              no shared-memory staging, no barrier until the reduction
//              (split-K with staged tiles held Turing to one CTA an SM and a
//              quarter of its issue slots); a projection too narrow to fill
//              the GPU splits its K (dense_splits, the shape's alone).
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
//              the splits in split order, rotates each row's 128-block and
//              scales by svh: the
//              epilogue that was a third launch.  The counters are a static
//              device pool, one slice per (host thread, device, stream):
//              kernels on one stream are serialised, so a slice never serves
//              two live launches.
//
// A call may carry up to kMaxProj projections of one input over the same rows
// (L288 exl3-dense: a GDN block's in_proj_qkv + in_proj_z, an attention's
// q / k / v / index, a shared expert's gate and up), each with its own output
// width and exactly the arithmetic it has alone: ONE prep launch for all of
// them (gridDim.y), then ONE GEMV launch per rate among them, whose flat grid
// runs each projection's (n-tile, split) CTAs in turn.  Fewer, fuller launches:
// a 512-wide projection alone is 16 x 18 CTAs of a 48-SM GPU, latency-bound.

#include "ds4_exl3_dense.cuh"
#include "ds4_exl3_dev.cuh"
#include "cuda/pulsar_cuda_mx.cuh"
#include "engine/exl3_trellis.h"
#include "cuda/pulsar_cuda_arch.cuh"

#include <climits>
#include <cstdio>
#include <cuda_bf16.h>

namespace {

constexpr int kOut       = 32;   ///< outputs per CTA = two n-tiles
constexpr int kWarps     = 16;   ///< warps per CTA: k-tile w % 8 of every other 128-block (two block parities)
constexpr int kPF        = 4;    ///< the trellis words' register prefetch ring, in k-steps
constexpr int kRows      = 16;   ///< the MMA's M: every row block is 16 rows
/** L341: rows one GEMV launch carries -- kRows-row passes over the trellis words the CTA reads once (the verify
 *  width, PULSAR_SPEC_ROW_BUDGET: 17..32 rows were two launches, each reading the weight). */
constexpr int kMaxRows   = 2 * kRows;
/** The prescale puts a block's largest |value| just under 2^kHiExp. */
constexpr int kHiExp     = 14;

/** CTAs a projection's grid should reach before it is split: GB10's 48 SMs (a 512-thread CTA of this kernel is
 *  one an SM).  A constant, so the split -- and the bytes -- are the shape's alone, on every device. */
constexpr int kFillCtas = 48;
/** The fewest 128-blocks a split's k-range may hold: two steps for each of the two warp block parities. */
constexpr int kMinSplitBlocks = 4;

/** The split count, a function of the weight's shape alone (L338 occupancy follow-up): 1 -- a CTA covers the whole
 *  of K, exllamav3's GEMV structure -- once the projection's 32-output CTAs fill kFillCtas; else the fewest splits
 *  that fill them, capped so each split keeps kMinSplitBlocks blocks.  S = 1 left GB10's SMs idle on the narrow
 *  projections (Qwen's shared gate / up: 20 CTAs; 2560 -> 1280: 40), ~1 ms a single-stream Qwen round (gpuq #588). */
/// @param K  input width (a multiple of 128)
/// @param N  stored output width (a multiple of 32)
/// @return   the split count S, 1 .. K / (128 x kMinSplitBlocks)
int dense_splits(int K, int N) {
    const int ctas = N / kOut, blocks = K / EXL3_HAD_BLOCK;
    if (ctas >= kFillCtas) return 1;
    int s = (kFillCtas + ctas - 1) / ctas;
    const int smax = blocks / kMinSplitBlocks;
    if (s > smax) s = smax;
    return s < 1 ? 1 : s;
}

/** Counters per (thread, device, stream) slice -- one per 128-output block of
 *  each projection, so n_proj * N <= kMaxCounters * 128 -- and the slices.
 *  L288 qwen-exl3-head: 2048, the widest Linear served being a head (Qwen's
 *  lm_head 2560 -> 248320 = 1940 blocks); 512 KB for the 64 slices. */
constexpr int kMaxCounters = 2048;
/// Counter slices in the pool: one per (host thread, device, stream) that ever launches (counter_slice).
constexpr int kCounterSlices = 64;
/// The epilogue's arrival counters, [slice][128-output block]; zero at load and left zero by every launch.
__device__ unsigned g_dense_ctr[kCounterSlices][kMaxCounters];   /* zero at load; every launch leaves it zero */

/** One projection's workspace: the split partials f32 [S][M][N], the prep planes hi and lo fp16 [M][K] and
 *  the inverse prescales f32 [M][K/128].  Every piece 16-byte aligned. */
struct Workspace {
    /** @var hi
     *  offset of the hi fp16 plane [M][K] */
    /** @var lo
     *  offset of the lo fp16 plane [M][K] */
    /** @var inv
     *  offset of the inverse prescales f32 [M][K/128] */
    /** @var bytes
     *  the projection's total bytes (the last piece padded to 16) */
    /// Offset of the split partials f32 [S][M][N] (0: they come first); every offset is from the projection's base.
    size_t part, hi, lo, inv, bytes;
    /// The layout at S splits, M rows, input width K, stored output width N.
    /// @param S  split count
    /// @param M  rows
    /// @param K  input width
    /// @param N  stored output width
    Workspace(int S, int M, int K, int N) {
        part = 0;
        hi = (size_t)S * M * N * sizeof(float);
        lo = hi + (size_t)M * K * sizeof(__half);
        inv = lo + (size_t)M * K * sizeof(__half);
        bytes = inv + (((size_t)M * (K / EXL3_HAD_BLOCK) * sizeof(float) + 15) & ~(size_t)15);
    }
};

/** One projection of a launch: its weight's planes, its workspace, its output, its counters.  A launch runs a set
 *  of up to kMaxProj projections of one rate over the same rows (a flat grid; proj_of names a CTA's projection),
 *  each with exactly the arithmetic it has alone. */
struct dense_proj {
    /// The weight's trellis words (the slice's first plane), device.
    const uint32_t *trellis;
    /** @var svh
     *  the output sign/scale vector, fp16 [N], device */
    /// The input sign/scale vector suh, fp16 [K], device (the plane after the trellis).
    const __half *suh, *svh;
    /** @var y
     *  the f32 output [M][Ny], or NULL when yb is set */
    /// The split partials f32 [S][M][N], in the workspace.
    float *part, *y;
    __nv_bfloat16 *yb;               ///< the bf16 output instead of y (decode rows), or NULL
    /** @var hi
     *  the prep's hi fp16 plane [M][K] */
    __half *hi, *lo;                 ///< the prep planes
    /// Inverse prescales 2^-P f32 [M][K/128] (the prep's).
    float *inv;
    /// Its N / 128 epilogue counters, in the caller's counter slice.
    unsigned *ctr;
    /** @var N
     *  its stored output width (exl3_out_stored, a multiple of 128) */
    int N, S;                        ///< its STORED outputs (exl3_out_stored) and its split count (dense_splits(K, N))
    int Ny;                          ///< its logical outputs: y's row width, the columns stored (<= N)
};

/** Projections one call carries at most. */
constexpr int kMaxProj = 4;

/** A launch's projections: GEMV CTA c belongs to the first j with c < end[j] (unused slots end at INT_MAX
 *  past the last), its local index c - end[j - 1] = split * (N / 32) + n-tile -- the order one projection's own
 *  (N / 32, S) grid runs in. */
struct dense_set {
    /// The projections; slots past the launch's count are unused.
    dense_proj p[kMaxProj];
    /// Running CTA totals: projection j owns flat CTAs [end[j-1], end[j]); unused slots INT_MAX.
    int end[kMaxProj];
};

/// The projection index of flat GEMV CTA c (P.end's first entry above c).
/// @param P  the launch's set
/// @param c  flat CTA index (blockIdx.x)
/// @return   j, 0 .. kMaxProj - 1
__device__ __forceinline__ int proj_of(const dense_set &P, int c) {
    return (c >= P.end[0]) + (c >= P.end[1]) + (c >= P.end[2]);
}
/// Projection j of the set, by a select chain (a constant-bank read, no local-memory index).
/// @param P  the launch's set
/// @param j  projection index, 0 .. kMaxProj - 1
/// @return   P.p[j]
__device__ __forceinline__ const dense_proj &proj_at(const dense_set &P, int j) {
    return j == 0 ? P.p[0] : j == 1 ? P.p[1] : j == 2 ? P.p[2] : P.p[3];
}

/// 2^e as an f32 built from its exponent bits; e in [-126, 127].
/// @param e  the exponent
/// @return   2^e
__device__ __forceinline__ float pow2f(int e) { return __int_as_float((127 + e) << 23); }   /* e in [-126, 127] */


/** L338: one m16n8k16 A fragment of a prep plane straight from global memory -- lane (g = l/4, c = l%4) holds rows
 *  g and g + 8 at k = 2c, 2c + 1 and 2c + 8, 2c + 9 of the 16-wide k-tile at k0 (the registers ldmatrix .x4 gave
 *  from the staged tile: the same values in the same registers), rows at or past nr zero and not read.  A warp's A
 *  tiles are its own (k-tile `warp` of each block), so nothing is staged in shared memory: on Turing (64 KB an SM)
 *  the 40 KB of staged tiles held the GEMV to one CTA an SM. */
/// @param a     out: the fragment
/// @param plane the prep plane (hi or lo), fp16 [M][K]
/// @param K     its row pitch
/// @param k0    the k-tile's first column
/// @param g     the lane's fragment row (lane / 4)
/// @param c2    the lane's fragment column pair (2 * (lane % 4))
/// @param nr    real rows
__device__ __forceinline__ void a_frag(uint32_t (&a)[4], const __half *__restrict__ plane, int K, int k0, int g, int c2,
                                       int nr) {
    const __half *r0 = plane + (size_t)g * K + k0 + c2, *r1 = plane + (size_t)(g + 8) * K + k0 + c2;
    a[0] = g < nr ? __ldg(reinterpret_cast<const unsigned *>(r0)) : 0u;
    a[1] = g + 8 < nr ? __ldg(reinterpret_cast<const unsigned *>(r1)) : 0u;
    a[2] = g < nr ? __ldg(reinterpret_cast<const unsigned *>(r0 + 8)) : 0u;
    a[3] = g + 8 < nr ? __ldg(reinterpret_cast<const unsigned *>(r1 + 8)) : 0u;
}

/** The PREP of one (row, 128-block), one warp: lane l owns k = 4l..4l+3 of the block (x the row's bf16 values
 *  there, su_p suh's).  hw / lw: the hi and lo fp16 pairs of the lane's four values; returns 2^-P. */
/// @param x     the lane's 4 bf16 inputs (8-byte aligned)
/// @param su_p  suh at the same 4 positions (8-byte aligned)
/// @param hw    out: the hi fp16 pairs
/// @param lw    out: the lo fp16 pairs
/// @return      2^-P, the block's inverse prescale (the same on every lane)
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
/// PREP: suh, H128, the power-of-two prescale and the fp16 hi + lo split of every (row, 128-block), into each
/// projection's hi / lo / inv workspace planes.  Grid (ceil(rows * K/128 / 8), n_proj), 256 threads: one warp per
/// (row, block) task, projection blockIdx.y.
/// @param P     the call's projections
/// @param xb    the bf16 input rows [rows][K]
/// @param rows  M
/// @param K     input width (a multiple of 128)
__global__ void __launch_bounds__(256)
exl3_dense_prep_kernel(const dense_set P, const __nv_bfloat16 *__restrict__ xb, int rows, int K) {
    pulsar_pdl_trigger();   /* L340: the GEMV may launch now and fill its trellis-word ring under this grid */
    const dense_proj &p = proj_at(P, (int)blockIdx.y);
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

/* 2. GEMV + EPILOGUE.  A flat grid over the set's projections (dense_set): CTA x of projection j owns its outputs
 * 32x..32x+31 over the whole of K. */
/// GEMV + EPILOGUE at rate K2 (2 x bits per weight).  Flat grid over the set's n-tile pairs, block (32, kWarps):
/// each CTA 32 outputs x its split's k-range (all of K at S = 1 -- exllamav3's GEMV structure, exl3_gemv_kernel.cuh;
/// S > 1 only for a projection too narrow to fill the GPU) x all rows; warp w takes k-tile w % 8 of the range's
/// 128-blocks w / 8, w / 8 + 2, ... through a register
/// prefetch ring (kPF steps: one coalesced trellis word per lane per 32 of each tile, the lane's windows by
/// shuffle) with its A fragments and prescales one step ahead, never synchronising until the reduction; the warps
/// reduced in warp order (dynamic smem, kWarps x M x 32 f32) into the CTA's partial; the last CTA of a 128-output
/// block (on its counter) applies H128 and svh, writes y or yb.
/// @param P   the projections of one rate (the prep planes carry the input)
/// L341 (on this structure): PASSES kRows-row passes share each step's decoded B fragments -- every pass its own
/// A fragments, prescales and accumulators (rows 16 x pass ..), the reduction one pass at a time through a kRows-row
/// buffer -- so a row's arithmetic is exactly a 16-row launch's of its block and the words are read once for them all.
/// L340: the wait for the prep's planes follows the ring's first trellis loads (programmatic dependent launch).
/// @param M   rows, 1..PASSES x kRows
/// @param K   input width
template <int K2, int PASSES>
__global__ void __launch_bounds__(kOut * kWarps)
exl3_dense_kernel(const dense_set P, int M, int K) {
    extern __shared__ float s_red[];                              /* [kWarps][min(M, kRows)][kOut], one pass at a time */
    __shared__ bool s_last;
    const int c = (int)blockIdx.x, j = proj_of(P, c);
    const dense_proj &p = proj_at(P, j);
    const int cl = c - (j == 0 ? 0 : j == 1 ? P.end[0] : j == 2 ? P.end[1] : P.end[2]);
    const int N = p.N, ntn = N >> 4, S = p.S, ntiles = N / kOut;
    const int bx = cl % ntiles, split = cl / ntiles;              /* the 32-output tile, the k-range */
    constexpr int W32 = exl3dev::Rate<K2>::words32;
    constexpr int NV = (W32 + 31) / 32;                           /* a tile's words per lane */
    const int lane = threadIdx.x, warp = threadIdx.y, tid = warp * kOut + lane;
    const int blocks = K >> 7;
    const int b_lo = (int)((int64_t)split * blocks / S);          /* the split's 128-blocks: b_lo .. b_lo + nb - 1 */
    const int nb = (int)((int64_t)(split + 1) * blocks / S) - b_lo;
    int nrp[PASSES];                                              /* each pass's real rows, 0..kRows */
#pragma unroll
    for (int q = 0; q < PASSES; ++q) nrp[q] = M - q * kRows < 0 ? 0 : (M - q * kRows < kRows ? M - q * kRows : kRows);
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    const int kti = warp & 7, b0 = warp >> 3;                     /* k-tile within a block, first block */
    const int nsteps = nb > b0 ? (nb - b0 + 1) / 2 : 0;           /* blocks b_lo + b0, + 2, ... of the split */

    /* the lane's eight positions 8t..8t+7 of every tile: ONE 64-bit window when the rate's eight-position runs fit
     * one (run8_slack >= 0; K = 2, 3, 4 and 6 bits), else two four-position runs (n8 half 0, half 1) */
    constexpr bool R8 = exl3dev::run8_slack(K2) >= 0;
    int lo[2], hi[2], sh[2];
    if constexpr (R8) {
        exl3dev::run8_window<K2>(8 * lane, lo[0], hi[0], sh[0]);
        lo[1] = lo[0]; hi[1] = hi[0]; sh[1] = sh[0];
    } else {
#pragma unroll
        for (int h = 0; h < 2; ++h) exl3dev::run4_window<K2>(8 * lane + 4 * h, lo[h], hi[h], sh[h]);
    }

    /* step s: block b0 + 2s, k-tile kti -- its two tiles' words (coalesced), its A fragments and prescales */
    auto ld_w = [&](int s, uint32_t (&tw)[2][NV]) {
        const int kt = (b_lo + b0 + 2 * s) * 8 + kti;
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            const uint32_t *t = p.trellis + ((size_t)kt * (size_t)ntn + (size_t)(bx * 2 + jj)) * W32;
#pragma unroll
            for (int v = 0; v < NV; ++v) tw[jj][v] = v * 32 + lane < W32 ? __ldcs(t + v * 32 + lane) : 0u;
        }
    };
    auto ld_a = [&](int s, int q, uint32_t (&ah)[4], uint32_t (&al)[4], float &ia, float &ib) {   /* pass q's rows */
        const int b = b_lo + b0 + 2 * s, k0 = (b * 8 + kti) * 16, nr = nrp[q];
        const size_t r0 = (size_t)q * kRows;
        a_frag(ah, p.hi + r0 * K, K, k0, g, c2, nr);
        a_frag(al, p.lo + r0 * K, K, k0, g, c2, nr);
        ia = g < nr ? __ldg(p.inv + (r0 + g) * blocks + b) : 0.0f;
        ib = g + 8 < nr ? __ldg(p.inv + (r0 + g + 8) * blocks + b) : 0.0f;
    };
    /* word `w` of a tile from the lanes holding it */
    auto word = [&](const uint32_t (&v)[NV], int w) -> uint32_t {
        if constexpr (NV == 1) {
            return __shfl_sync(0xffffffffu, v[0], w);
        } else {
            const uint32_t x0 = __shfl_sync(0xffffffffu, v[0], w & 31), x1 = __shfl_sync(0xffffffffu, v[1], w & 31);
            return w < 32 ? x0 : x1;
        }
    };

    uint32_t pf[kPF][2][NV];
#pragma unroll
    for (int d = 0; d < kPF; ++d)
        if (d < nsteps) ld_w(d, pf[d]);
    pulsar_pdl_wait();   /* L340: above, only the weight; below, the prep's planes and prescales */
    /* pass 0's next-step fragments are double-buffered (nhi..); a later pass reloads its own right after its MMAs --
     * the same values, issued later, so the two-pass body stays inside the 128 registers 512 threads allow */
    uint32_t ahi[PASSES][4] = {}, alo[PASSES][4] = {}, nhi[4] = {}, nlo[4] = {};
    float ia[PASSES] = {}, ib[PASSES] = {}, nia = 0.0f, nib = 0.0f;
#pragma unroll
    for (int q = 0; q < PASSES; ++q)
        if (nsteps > 0) ld_a(0, q, ahi[q], alo[q], ia[q], ib[q]);

    float acc[PASSES][2][2][4];
#pragma unroll
    for (int q = 0; q < PASSES; ++q)
#pragma unroll
        for (int jj = 0; jj < 2; ++jj)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[q][jj][h][e] = 0.0f;

    for (int s0 = 0; s0 < nsteps; s0 += kPF) {
#pragma unroll
        for (int d = 0; d < kPF; ++d) {
            const int s = s0 + d;
            if (s >= nsteps) break;
            uint32_t tw[2][NV];
#pragma unroll
            for (int jj = 0; jj < 2; ++jj)
#pragma unroll
                for (int v = 0; v < NV; ++v) tw[jj][v] = pf[d][jj][v];
            if (s + kPF < nsteps) ld_w(s + kPF, pf[d]);
            if (s + 1 < nsteps) ld_a(s + 1, 0, nhi, nlo, nia, nib);
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                uint32_t st8[8];
                if constexpr (R8)
                    exl3dev::run8_split<K2>(EXL3_RUN4_JOIN(word(tw[jj], lo[0]), word(tw[jj], hi[0]), sh[0]), 8 * lane, st8);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t st[4];
                    if constexpr (R8) {
#pragma unroll
                        for (int e = 0; e < 4; ++e) st[e] = st8[4 * h + e];
                    } else {
                        exl3dev::run4_split<K2>(EXL3_RUN4_JOIN(word(tw[jj], lo[h]), word(tw[jj], hi[h]), sh[h]),
                                                8 * lane + 4 * h, st);
                    }
                    const uint32_t bb0 = exl3dev::mul1_pair(st[0], st[1]);   /* k rows 2c, 2c+1 */
                    const uint32_t bb1 = exl3dev::mul1_pair(st[2], st[3]);   /* k rows 2c+8, 2c+9 */
#pragma unroll
                    for (int q = 0; q < PASSES; ++q) {   /* L341: the decoded B, every pass's rows */
                        float dd[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        pulsar_mma_f16_16816(dd, ahi[q], bb0, bb1);
                        pulsar_mma_f16_16816(dd, alo[q], bb0, bb1);
                        acc[q][jj][h][0] = fmaf(dd[0], ia[q], acc[q][jj][h][0]);
                        acc[q][jj][h][1] = fmaf(dd[1], ia[q], acc[q][jj][h][1]);
                        acc[q][jj][h][2] = fmaf(dd[2], ib[q], acc[q][jj][h][2]);
                        acc[q][jj][h][3] = fmaf(dd[3], ib[q], acc[q][jj][h][3]);
                    }
                }
            }
#pragma unroll
            for (int e = 0; e < 4; ++e) { ahi[0][e] = nhi[e]; alo[0][e] = nlo[e]; }
            ia[0] = nia;
            ib[0] = nib;
#pragma unroll
            for (int q = 1; q < PASSES; ++q)
                if (s + 1 < nsteps) ld_a(s + 1, q, ahi[q], alo[q], ia[q], ib[q]);
        }
    }

    /* cross-warp reduction in warp order, one warp per row, into the CTA's partial -- a pass at a time (L341) */
#pragma unroll
    for (int q = 0; q < PASSES; ++q) {
        const int nr = nrp[q];
        if (nr == 0) break;
        if (q > 0) __syncthreads();   /* every warp is past the last pass's reads of the buffer */
#pragma unroll
        for (int jj = 0; jj < 2; ++jj)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int rr = g + 8 * (e >> 1), col = jj * 16 + h * 8 + c2 + (e & 1);
                    if (rr < nr) s_red[((size_t)warp * nr + rr) * kOut + col] = acc[q][jj][h][e];
                }
        __syncthreads();
        for (int r = warp; r < nr; r += kWarps) {
            float z = 0.0f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w) z += s_red[((size_t)w * nr + r) * kOut + lane];
            p.part[((size_t)split * (size_t)M + (size_t)(q * kRows + r)) * (size_t)N + (size_t)(bx * kOut + lane)] = z;
        }
    }

    /* 4. the last of the 128-output block's 4 x S CTAs finishes it: the splits in split order (fixed by the split
     *    index, never by arrival; split 0's value is the start, so S = 1 is the unsplit kernel's bytes), H128, svh */
    __threadfence();
    __syncthreads();
    const int ob = bx >> 2;                               /* the 128-output block */
    if (tid == 0) s_last = atomicInc(&p.ctr[ob], 4u * (unsigned)S - 1u) == 4u * (unsigned)S - 1u;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    const int n0 = ob * 128 + lane * 4;
    for (int mm = warp; mm < M; mm += kWarps) {
        const float4 v = __ldcg(reinterpret_cast<const float4 *>(p.part + (size_t)mm * (size_t)N + n0));
        float z[4] = {v.x, v.y, v.z, v.w};
        for (int sp = 1; sp < S; ++sp) {
            const float4 u = __ldcg(reinterpret_cast<const float4 *>(p.part + ((size_t)sp * (size_t)M + (size_t)mm) *
                                                                              (size_t)N + n0));
            z[0] += u.x; z[1] += u.y; z[2] += u.z; z[3] += u.w;
        }
        exl3dev::had128(z);   /* the whole warp: a stored block mixes its pad columns in */
        const __half2 v01 = *reinterpret_cast<const __half2 *>(p.svh + n0);
        const __half2 v23 = *reinterpret_cast<const __half2 *>(p.svh + n0 + 2);
        if (n0 < p.Ny) {      /* L295 F7: a stored pad column (exl3_out_stored) is never written */
            const float4 o = make_float4(z[0] * __low2float(v01), z[1] * __high2float(v01), z[2] * __low2float(v23),
                                         z[3] * __high2float(v23));
            if (p.yb) {       /* L296 dec2: the bf16 rows the consumer reads, emitted here */
                const __nv_bfloat162 b01 = __floats2bfloat162_rn(o.x, o.y), b23 = __floats2bfloat162_rn(o.z, o.w);
                *reinterpret_cast<uint2 *>(p.yb + (size_t)mm * (size_t)p.Ny + n0) =
                    make_uint2(*reinterpret_cast<const uint32_t *>(&b01), *reinterpret_cast<const uint32_t *>(&b23));
            } else {
                *reinterpret_cast<float4 *>(p.y + (size_t)mm * (size_t)p.Ny + n0) = o;
            }
        }
    }
}

/** The calling thread's counter slice for (device, stream): kernels on one
 *  stream are serialised and two threads never share a slice, so no two live
 *  launches meet on a counter.  nullptr once the pool is spent (a refusal). */
/// @param stream  the stream the launch will run on
/// @return         that slice's kMaxCounters counters, or nullptr when the pool is spent
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

/// Launch exl3_dense_kernel at rate k2 (4, 6, 8, 10, 12 or 16, checked by the caller) with grid / block on stream.
/// @param k2      rate (2 x bits per weight)
/// @param grid    flat CTA count of the set
/// @param block   (kOut, kWarps, 1)
/// @param stream  the launch stream
/// @param P       the set's projections, all at rate k2
/// @param M       rows, 1..kMaxRows
/// @param K       input width
/// @return the launch's error
cudaError_t gemv_launch(int k2, const dim3 &grid, const dim3 &block, cudaStream_t stream, const dense_set &P, int M,
                        int K) {
    static_assert(kMaxRows == 2 * kRows, "the instances below are one and two passes");
    const size_t smem = (size_t)kWarps * (size_t)(M < kRows ? M : kRows) * kOut * sizeof(float);   /* a pass's rows */
    const bool two = M > kRows;   /* L341: the pass count is the instance's (M <= kMaxRows, checked) */
    void (*k)(const dense_set, int, int) =
        k2 == 4  ? (two ? exl3_dense_kernel<4, 2>  : exl3_dense_kernel<4, 1>)
      : k2 == 6  ? (two ? exl3_dense_kernel<6, 2>  : exl3_dense_kernel<6, 1>)
      : k2 == 8  ? (two ? exl3_dense_kernel<8, 2>  : exl3_dense_kernel<8, 1>)
      : k2 == 10 ? (two ? exl3_dense_kernel<10, 2> : exl3_dense_kernel<10, 1>)
      : k2 == 12 ? (two ? exl3_dense_kernel<12, 2> : exl3_dense_kernel<12, 1>)
                 : (two ? exl3_dense_kernel<16, 2> : exl3_dense_kernel<16, 1>);   /* k2 == 16, checked */
    /* L340: under the previous kernel's tail (the prep, or the last rate's GEMV): the kernel waits before its planes */
    return pulsar_launch_pdl(k, grid, block, smem, stream, P, M, K);
}

/* One projection's workspace bytes at (M <= kMaxRows, K, N), 16-byte aligned so the next one's starts aligned. */
/// One projection's workspace bytes at (M <= kMaxRows, K, stored N), rounded up to 16.
/// @param M  rows, <= kMaxRows
/// @param K  input width
/// @param N  stored output width
/// @return   bytes
size_t proj_ws_bytes(int M, int K, int N) {
    return (Workspace(dense_splits(K, N), M, K, N).bytes + 15) & ~(size_t)15;
}

} // namespace

bool ds4_exl3_dense_rate_supported(int k2) { return exl3_arm_has_rate(EXL3_ARM_DENSE, k2); }

int ds4_exl3_dense_splits(int K, int N) {
    if (K <= 0 || N <= 0 || K % EXL3_HAD_BLOCK || N % 4) return 0;
    return dense_splits(K, (int)exl3_out_stored((uint64_t)N));
}

size_t ds4_exl3_dense_multi_workspace_bytes(const ds4_exl3_dense_proj *p, int n, int M, int K) {
    if (!p || n < 1 || n > DS4_EXL3_DENSE_MAX_PROJ || M <= 0) return 0;
    size_t b = 0;
    for (int j = 0; j < n; ++j) {
        if (ds4_exl3_dense_splits(K, p[j].N) == 0) return 0;
        /* M > kMaxRows: kMaxRows-row launches in turn (decode rows); the partials are the stored width's */
        b += proj_ws_bytes(M < kMaxRows ? M : kMaxRows, K, (int)exl3_out_stored((uint64_t)p[j].N));
    }
    return b;
}

size_t ds4_exl3_dense_workspace_bytes(int M, int K, int N) {
    const ds4_exl3_dense_proj p{nullptr, 0, N, nullptr, nullptr};
    return ds4_exl3_dense_multi_workspace_bytes(&p, 1, M, K);
}

int ds4_exl3_dense_multi_launch(const ds4_exl3_dense_proj *p, int n, const void *xb, int M, int K, void *workspace,
                                size_t workspace_bytes, cudaStream_t stream, bool prompt) {
    const char *tag = "ds4_exl3_dense_launch";
    static_assert(DS4_EXL3_DENSE_MAX_PROJ == kMaxProj, "the header's projection cap is the kernel's");
    if (!p || n < 1 || n > kMaxProj || !xb || !workspace || M <= 0) {
        fprintf(stderr, "%s: %d projections (1..%d), a null pointer or M=%d -- refusing\n", tag, n, kMaxProj, M);
        return -1;
    }
    uint64_t trellis_bytes[kMaxProj] = {};
    int n_ctr = 0;
    for (int j = 0; j < n; ++j) {
        uint64_t scale_bytes = 0, stride = 0;
        const int N = p[j].N;
        if (!p[j].w || !p[j].y == !p[j].yb || (p[j].yb && prompt)) {
            fprintf(stderr, "%s: projection %d has no weight, or not exactly one output (f32 y, or bf16 yb on decode "
                            "rows) -- refusing\n", tag, j);
            return -1;
        }
        if (N <= 0 || N % 4 ||
            !exl3_linear_layout((uint64_t)(K > 0 ? K : 0), (uint64_t)N, p[j].k2, &trellis_bytes[j], &scale_bytes,
                                &stride)) {
            fprintf(stderr, "%s: K=%d N=%d k2=%d is not an EXL3 dense layout (K %% 128, N %% 4 -- the epilogue's "
                            "float4 stores; N stored at its 128 multiple -- and a valid rate) -- refusing\n",
                    tag, K, N, p[j].k2);
            return -1;
        }
        if (!ds4_exl3_dense_rate_supported(p[j].k2)) {
            fprintf(stderr, "%s: rate k2=%d has no instance (K = 2, 3, 4, 5, 6, 8) -- refusing\n", tag, p[j].k2);
            return -1;
        }
        if ((uintptr_t)p[j].w % 16 || (uintptr_t)p[j].y % 16 || (uintptr_t)p[j].yb % 8) {
            fprintf(stderr, "%s: w / y must be 16-byte aligned, yb 8-byte -- refusing\n", tag);
            return -1;
        }
        n_ctr += (int)(exl3_out_stored((uint64_t)N) / 128);
    }
    if ((uintptr_t)xb % 8 || (uintptr_t)workspace % 16) {
        fprintf(stderr, "%s: the workspace must be 16-byte aligned and xb 8-byte (the uint2 row loads) -- refusing\n", tag);
        return -1;
    }
    /* L251: a prompt chunk takes the tensor-core GEMM (qwen_exl3_dense_prefill.cu): this
     * arm's GEMV re-decodes the trellis per 16 rows and re-reads the activation planes per 32 outputs,
     * a decode kernel doing a GEMM's work.  Decode rows keep the GEMV bit for bit at every width. */
    if (prompt) {
        for (int j = 0; j < n; ++j) {
            const int rc = qwen_exl3_dense_prefill_launch(p[j].w, p[j].k2, xb, p[j].y, M, K, p[j].N, trellis_bytes[j],
                                                          stream);
            if (rc) return rc;
        }
        return 0;
    }
    /* decode rows are the GEMV; its m16n8k16 MMA and cp.async run on sm_75 through their arch bodies (L338,
     * pulsar_cuda_arch.cuh) */
    if (M > kMaxRows) {
        /* L288 C1: decode rows past one launch's kMaxRows run kMaxRows-row launches in turn -- each the call it is
         * alone, so every row is the bytes a one-row decode gives at ANY width (within a launch the kRows-row passes
         * keep that, L341).  The GEMM above lost here: it decodes the whole weight to fp16 for a handful of rows
         * (+11 ms a 32-row Qwen verify, L288 C1).  The launches reuse the workspace in stream order. */
        static bool announced = false;
        if (!__atomic_exchange_n(&announced, true, __ATOMIC_RELAXED))
            fprintf(stderr, "pulsar: EXL3 dense decode rows past %d run as %d-row GEMV launches (first call: M=%d K=%d)\n",
                    kMaxRows, kMaxRows, M, K);
        for (int r0 = 0; r0 < M; r0 += kMaxRows) {
            ds4_exl3_dense_proj q[kMaxProj];
            for (int j = 0; j < n; ++j) {
                q[j] = p[j];
                if (p[j].y) q[j].y = p[j].y + (size_t)r0 * p[j].N;
                if (p[j].yb) q[j].yb = p[j].yb + (size_t)r0 * p[j].N;
            }
            const int m = M - r0 < kMaxRows ? M - r0 : kMaxRows;
            const int rc = ds4_exl3_dense_multi_launch(q, n, static_cast<const __nv_bfloat16 *>(xb) + (size_t)r0 * K, m, K,
                                                       workspace, workspace_bytes, stream, false);
            if (rc) return rc;
        }
        return 0;
    }
    const size_t need = ds4_exl3_dense_multi_workspace_bytes(p, n, M, K);
    if (workspace_bytes < need) {
        fprintf(stderr, "%s: workspace %zu bytes < %zu for %d projection(s) at M=%d K=%d -- refusing\n", tag,
                workspace_bytes, need, n, M, K);
        return -1;
    }
    if (n_ctr > kMaxCounters) {
        fprintf(stderr, "%s: %d 128-output blocks are past the epilogue's %d counters -- refusing\n", tag, n_ctr,
                kMaxCounters);
        return -1;
    }
    unsigned *ctr = counter_slice(stream);
    if (!ctr) {
        fprintf(stderr, "%s: no epilogue counter slice left for this (thread, device, stream) -- the pool holds %d "
                        "-- refusing\n", tag, kCounterSlices);
        return -1;
    }
    /* every projection's planes and partials, and its own counters */
    dense_set all{};
    {
        uint8_t *ws = static_cast<uint8_t *>(workspace);
        unsigned *c = ctr;
        for (int j = 0; j < n; ++j) {
            const int N = (int)exl3_out_stored((uint64_t)p[j].N), S = dense_splits(K, N);   /* the stored width */
            const Workspace L(S, M, K, N);
            const __half *suh = reinterpret_cast<const __half *>(static_cast<const uint8_t *>(p[j].w) + trellis_bytes[j]);
            all.p[j] = dense_proj{static_cast<const uint32_t *>(p[j].w), suh, suh + K, reinterpret_cast<float *>(ws + L.part),
                                  p[j].y, reinterpret_cast<__nv_bfloat16 *>(p[j].yb),
                                  reinterpret_cast<__half *>(ws + L.hi), reinterpret_cast<__half *>(ws + L.lo),
                                  reinterpret_cast<float *>(ws + L.inv), c, N, S, p[j].N};
            ws += proj_ws_bytes(M, K, N);
            c += N / 128;
        }
    }
    const __nv_bfloat16 *xb16 = static_cast<const __nv_bfloat16 *>(xb);
    /* ONE prep for every projection (the input is shared; each takes its own suh) */
    const int64_t tasks = (int64_t)M * (K / EXL3_HAD_BLOCK);
    exl3_dense_prep_kernel<<<dim3((unsigned)((tasks + 7) / 8), (unsigned)n), 256, 0, stream>>>(all, xb16, M, K);
    if (n > 1) {
        static bool announced = false;
        if (!__atomic_exchange_n(&announced, true, __ATOMIC_RELAXED))
            fprintf(stderr, "pulsar: EXL3 dense multi launch = one prep + one GEMV per rate over up to %d projections of "
                            "one input (first call: %d at K=%d, M=%d)\n", kMaxProj, n, K, M);
    }
    if (M > kRows) {
        static bool announced = false;
        if (!__atomic_exchange_n(&announced, true, __ATOMIC_RELAXED))
            fprintf(stderr, "pulsar: EXL3 dense decode rows past %d run as %d-row passes over trellis words read once "
                            "(L341; first call: M=%d K=%d)\n", kRows, kRows, M, K);
    }
    /* ONE GEMV per rate among them, in the order the projections were given */
    bool done[kMaxProj] = {};
    for (int j0 = 0; j0 < n; ++j0) {
        if (done[j0]) continue;
        dense_set g{};
        int m = 0, ctas = 0;
        for (int j = j0; j < n; ++j) {
            if (done[j] || p[j].k2 != p[j0].k2) continue;
            done[j] = true;
            g.p[m] = all.p[j];
            ctas += (all.p[j].N / kOut) * all.p[j].S;
            g.end[m++] = ctas;
        }
        for (int i = m; i < kMaxProj; ++i) g.end[i] = INT_MAX;
        const cudaError_t le = gemv_launch(p[j0].k2, dim3((unsigned)ctas), dim3(kOut, kWarps, 1), stream, g, M, K);
        if (le != cudaSuccess) {
            fprintf(stderr, "%s: GEMV launch failed: %s\n", tag, cudaGetErrorString(le));
            return -3;
        }
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

int ds4_exl3_dense_launch(const void *w, int k2, const void *xb, float *y,
                          int M, int K, int N, void *workspace, size_t workspace_bytes, cudaStream_t stream, bool prompt) {
    const ds4_exl3_dense_proj p{w, k2, N, y, nullptr};
    return ds4_exl3_dense_multi_launch(&p, 1, xb, M, K, workspace, workspace_bytes, stream, prompt);
}

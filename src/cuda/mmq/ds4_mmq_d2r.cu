// SPDX-License-Identifier: MIT
// ds4_mmq_d2r.cu - gated D2R IQ2_XXS MoE GEMM production path.

#include "ds4_mmq_d2r.cuh"

#include <type_traits>

#include "ds4_cuda_env.cuh"
#include "ds4_act_block.cuh"
#include "mma.cuh"          /* ggml_cuda_mma::tile / load_ldmatrix */
#include "ds4_mxfp8_mma.cuh"
#include "cuda/pulsar_cuda_mx.cuh"
#include "pulsar_gpu.h"     /* pulsar_gpu_matmul_batch_decode_rows: L167's row-kind predicate */

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

/* Lifted verbatim from the vendored vecdotq.cuh (upstream llama.cpp).
 *
 * Eight lines of bit twiddling with no dependencies, and this file is its only
 * LIVE caller -- the other three are in mmq-load-tiles.cuh, which is on the
 * deletion list (ledger L066: nothing calls the vendored mul_mat_q kernels).
 * Copying it is what lets this translation unit stop including mmq.cuh, and a
 * copy this small is cheaper than keeping a 1363-line header alive to supply
 * it. */
static __device__ __forceinline__ uint32_t ds4_unpack_ksigns(const uint8_t v) {
    // v is a 7 bit int, with the 8th sign being encodable as popcnt
    // with xor we can "correct" the bit instead of having to mask
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    // broadcast over uint to allow for 0x08040201 / 0x80402010 as selectors
    return s * 0x01010101;
}

namespace {


constexpr int kMTile      = 128;
constexpr int kNTile      = 64;
constexpr int kWarps      = 8;
constexpr int kThreads    = 32 * kWarps;
constexpr int kStages     = 2;  // act PREFETCH DISTANCE (issue at k for k+2)
/* Act ring DEPTH.  Depth must EXCEED the prefetch distance or the issue at
 * iter k targets ((k+dist) % depth) == the very slot being read at k -- true
 * at ANY depth when depth == distance, which is why the old 2/2 scheme needed
 * a second full-block barrier between the MMA fold (reads) and the prefetch
 * issue (writes).  At depth 3 / distance 2 the write slot (k+2)%3 never
 * aliases the read slot k%3, and the remaining per-iteration barrier bounds
 * warp spread to one iteration so no third slot is ever live (L099). */
constexpr int kActDepth   = 3;
constexpr int kNFrag      = kNTile / 8;
/* L200: the DECODE tile width.  A layer routes a handful of assignment slots
 * across ~200 experts, so at decode a 64-wide tile stages 63 activation columns
 * that do not exist -- 27.6 of the 47 KiB shared budget -- and carries 8
 * accumulator fragments where one column needs 1.  Those two are what
 * independently cap the kernel at 2 blocks/SM (L199).  The width is a
 * PERFORMANCE choice only: it decides which block computes a column, never how,
 * so every width is bit-identical.  See the dispatch for when each is used. */
constexpr int kNTileNarrow = 8;
/* L202: the weights are no longer staged.  Under the k-major artifact layout
 * (type 44) the 16 rows one k step needs are 128 contiguous bytes, so the lane
 * that loads a code word is the lane that consumes it and the whole raw ring --
 * its 16.9 KiB of shared, its cp.async traffic and the shared round trip that
 * was 37.7% of this kernel's stall samples (L201) -- is gone. */
/* Act staging geometry, as a function of the tile width alone.  One column is
 * sizeof(block_mx_act_mmq) == 144 B == 9 cp.async chunks of 16 B, so a stage is
 * Cols*9 chunk-issues split over the CTA's 256 threads; a thread's item index
 * splits into (column, chunk) by Shift == log2(Cols).  The helpers below deduce
 * NFRAG from the s_act extent, so one body serves every width (L200). */
template <int NFRAG>
struct D2RAct {
    static constexpr int Cols  = NFRAG * 8;
    static constexpr int Items = Cols * 9;
    static constexpr int Trips = (Items + kThreads - 1) / kThreads;
    static constexpr int Shift = (Cols == 8) ? 3 : ((Cols == 16) ? 4 : ((Cols == 32) ? 5 : 6));
    static_assert(Cols == 8 || Cols == 16 || Cols == 32 || Cols == 64,
                  "D2R act staging supports tile widths 8, 16, 32 and 64");
    static_assert((1 << Shift) == Cols, "Shift must be log2(Cols)");
};

static_assert(kStages == 2, "D2R act prefetch distance is 2");
static_assert(kActDepth > kStages, "act ring depth must exceed the prefetch distance -- equal means the issue aliases the current read slot (L099)");
static_assert(kThreads == 256, "D2R CTA is fixed at 256 threads");
static_assert(D2RAct<kNFrag>::Trips == 3, "unexpected act issue trip count (wide tile)");
static_assert(D2RAct<kNTileNarrow / 8>::Trips == 1, "unexpected act issue trip count (narrow tile)");

struct alignas(16) SmemInvariants {
    const char *w_base;
    /* L202: EXPERT-based, not tensor-based -- this block's expert offset is
     * folded in once here so the inner loop indexes with (k256, pair, row)
     * alone.  Layout (type 44): d[(e*nb + k)*M + row],
     * q[((e*nb + k)*8 + pair)*M + row]. */
    const half *iq2_dq_base;
    const uint2 *iq2_qs_base;
    const char *act_tile_base;
    float *out;
    uint32_t sc_off_bytes;
    uint32_t qs_off_bytes;
    uint32_t act_k128_stride_bytes;
    int nb;
    int k128_iters;
    int M;
    int cta_row0;
    int col_lo;
    int col_count;
};

static_assert(sizeof(SmemInvariants) <= 128, "shared invariant table must stay small");












__device__ __forceinline__ void zero_16B(void *dst) {
    int4 z = make_int4(0, 0, 0, 0);
    *reinterpret_cast<int4 *>(dst) = z;
}

__device__ __forceinline__ void cp_async_16B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(smem), "l"(src));
    } else {
        zero_16B(dst);
    }
#else
    if (pred) {
        *reinterpret_cast<int4 *>(dst) = *reinterpret_cast<const int4 *>(src);
    } else {
        zero_16B(dst);
    }
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

template <int KeepGroups>
__device__ __forceinline__ void cp_async_wait_group() {
    static_assert(KeepGroups >= 0 && KeepGroups <= 7, "bad cp.async wait_group depth");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" :: "n"(KeepGroups));
#endif
}

__device__ __forceinline__ void cp_async_wait_keep(int keep_groups) {
    switch (keep_groups) {
        case 0: cp_async_wait_group<0>(); break;
        case 1: cp_async_wait_group<1>(); break;
        case 2: cp_async_wait_group<2>(); break;
        case 3: cp_async_wait_group<3>(); break;
        default: cp_async_wait_group<4>(); break;
    }
}


__device__ __forceinline__ int d2r_lane() {
#if defined(__CUDA_ARCH__)
    uint32_t lane;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(lane));
    return (int)lane;
#else
    return (int)threadIdx.x;
#endif
}

__device__ __forceinline__ int d2r_warp() {
#if defined(__CUDA_ARCH__)
    uint32_t warp;
    asm volatile("mov.u32 %0, %%tid.y;" : "=r"(warp));
    return (int)warp;
#else
    return (int)threadIdx.y;
#endif
}

__device__ __forceinline__ int d2r_tid() {
    return (d2r_warp() << 5) | d2r_lane();
}

__device__ __forceinline__ int d2r_group() {
    return d2r_lane() >> 2;
}

__device__ __forceinline__ int d2r_tig() {
    return d2r_lane() & 3;
}

__device__ __forceinline__ int d2r_act_stage(int k128_iter) {
    return k128_iter % kActDepth;
}

template <bool FullTile, int NFRAG>
__device__ __forceinline__ void issue_act_prefetch_one(
        block_mx_act_mmq (&s_act)[kActDepth][NFRAG][8],
        const char * __restrict__ act_iter_base,
        int col_count, int stage, int t) {
    constexpr int cols = D2RAct<NFRAG>::Cols;
    const int col_local = t & (cols - 1);
    const int chunk = t >> D2RAct<NFRAG>::Shift;
    const int nf = col_local >> 3;
    const int c = col_local & 7;
    /* L207: a column PAST col_count is zero for the whole block's life -- the
     * kernel zeroes s_act once at entry -- so skip it entirely rather than
     * re-storing zeros into it every stage.  At decode the narrow tile has one
     * real assignment slot of eight, so this was 63 of every 72 shared stores
     * on the activation path doing nothing.  (The FullTile arm has every column
     * valid by construction and is unchanged.) */
    if constexpr (!FullTile) {
        if (col_local >= col_count) {
            return;
        }
    }
    void *dst = (char *)&s_act[stage][nf][c] + chunk * 16;
    const void *src = act_iter_base + (uint64_t)col_local * sizeof(block_mx_act_mmq) + chunk * 16;
    cp_async_16B(dst, src, true);
}

template <bool FullTile, int Iter, int NFRAG>
__device__ __forceinline__ void issue_act_prefetch_unrolled(
        block_mx_act_mmq (&s_act)[kActDepth][NFRAG][8],
        const char * __restrict__ act_iter_base,
        int col_count, int stage, int tid) {
    if constexpr (Iter < D2RAct<NFRAG>::Trips) {
        const int t = tid + Iter * kThreads;
        if constexpr ((Iter + 1) * kThreads <= D2RAct<NFRAG>::Items) {
            issue_act_prefetch_one<FullTile>(s_act, act_iter_base, col_count, stage, t);
        } else {
            if (t < D2RAct<NFRAG>::Items) {
                issue_act_prefetch_one<FullTile>(s_act, act_iter_base, col_count, stage, t);
            }
        }
        issue_act_prefetch_unrolled<FullTile, Iter + 1>(
            s_act, act_iter_base, col_count, stage, tid);
    }
}

template <bool FullTile, int NFRAG>
__device__ __forceinline__ void issue_act_prefetch(
        block_mx_act_mmq (&s_act)[kActDepth][NFRAG][8],
        const volatile SmemInvariants &s_inv,
        int stage, int k128_iter, int tid) {
    const char *act_tile_base = s_inv.act_tile_base;
    const uint32_t k128_stride = s_inv.act_k128_stride_bytes;
    int col_count = D2RAct<NFRAG>::Cols;
    if constexpr (!FullTile) {
        col_count = s_inv.col_count;
    }
    const char *act_iter_base = act_tile_base + (uint64_t)k128_iter * (uint64_t)k128_stride;
    issue_act_prefetch_unrolled<FullTile, 0>(s_act, act_iter_base, col_count, stage, tid);
    cp_async_commit();
}

template <int NFRAG>
__device__ __forceinline__ void issue_act_prefetch_one_fast(
        block_mx_act_mmq (&s_act)[kActDepth][NFRAG][8],
        const char * __restrict__ act_iter_base,
        int stage, int t) {
    constexpr int cols = D2RAct<NFRAG>::Cols;
    const int col_local = t & (cols - 1);
    const int c = col_local & 7;
    const int nf = col_local >> 3;
    const int chunk = t >> D2RAct<NFRAG>::Shift;
    void *dst = (char *)&s_act[stage][nf][c] + chunk * 16;
    const void *src = act_iter_base + (uint64_t)col_local * sizeof(block_mx_act_mmq) + chunk * 16;
    cp_async_16B(dst, src, true);
}

template <int Iter, int NFRAG>
__device__ __forceinline__ void issue_act_prefetch_fast_unrolled(
        block_mx_act_mmq (&s_act)[kActDepth][NFRAG][8],
        const char * __restrict__ act_iter_base,
        int stage, int tid) {
    if constexpr (Iter < D2RAct<NFRAG>::Trips) {
        const int t = tid + Iter * kThreads;
        if constexpr ((Iter + 1) * kThreads <= D2RAct<NFRAG>::Items) {
            issue_act_prefetch_one_fast(s_act, act_iter_base, stage, t);
        } else {
            if (t < D2RAct<NFRAG>::Items) {
                issue_act_prefetch_one_fast(s_act, act_iter_base, stage, t);
            }
        }
        issue_act_prefetch_fast_unrolled<Iter + 1>(s_act, act_iter_base, stage, tid);
    }
}

template <int NFRAG>
__device__ __forceinline__ void issue_act_prefetch_fast(
        block_mx_act_mmq (&s_act)[kActDepth][NFRAG][8],
        const volatile SmemInvariants &s_inv,
        int stage, int k128_iter) {
    const char *act_iter_base =
        s_inv.act_tile_base + (uint64_t)k128_iter * (uint64_t)s_inv.act_k128_stride_bytes;
    issue_act_prefetch_fast_unrolled<0>(s_act, act_iter_base, stage, d2r_tid());
    cp_async_commit();
}













constexpr size_t kSmemActStageBytes = (size_t)kNFrag * 8 * sizeof(block_mx_act_mmq);
constexpr size_t kSmemInvBytes = sizeof(SmemInvariants);
constexpr size_t kSmemIQ2GridBytes = 256u * sizeof(uint2);
/* kActDepth, not kStages: s_act is declared [kActDepth][...], so the old
 * accounting under-counted the act ring by a whole stage (L200). */
constexpr size_t kSmemIQ2StaticBytes = (size_t)kActDepth * kSmemActStageBytes +
                                       kSmemIQ2GridBytes + kSmemInvBytes;
static_assert(kSmemIQ2StaticBytes <= 48ull * 1024ull,
              "IQ2 D2R static shared memory exceeds 48 KiB");
/* The narrow tile exists to get under the 3-blocks/SM shared cliff (102.4 KiB
 * per SM), which is the whole point of L200 -- assert it rather than hope. */
constexpr size_t kSmemIQ2NarrowBytes =
    (size_t)kActDepth * (size_t)(kNTileNarrow / 8) * 8 * sizeof(block_mx_act_mmq) +
    kSmemIQ2GridBytes + kSmemInvBytes;
static_assert(kSmemIQ2NarrowBytes * 4 <= 102400ull,
              "narrow D2R tile must leave room for 4 blocks per SM");















template <int NFrag, typename TileB>
__device__ __forceinline__ void load_B_tile(
        TileB &B,
        const block_mx_act_mmq (&s_act)[kActDepth][NFrag][8],
        int stage, int nf, int k_in_act) {
    const int *base = reinterpret_cast<const int *>(&s_act[stage][nf][0].qs[k_in_act]);
    ggml_cuda_mma::load_ldmatrix(B, base, sizeof(block_mx_act_mmq) / sizeof(int));
}




/* L198/L199: what one half of a grid entry contributes to one A quad.
 *
 * Every one of the IQ2_XXS grid's 2048 bytes is POSITIVE and drawn from exactly
 * {8, 25, 43} (verified against the vendored table itself in
 * ds4_iq2_grid_magnitudes_ok below -- the sign lives in the ksigns field, not in
 * the grid).  A weight's E4M3 byte is therefore one of only three values per
 * row, so the dequant needs to name WHICH, not compute it: `sel` carries one
 * byte-permute nibble per weight (0/1/2 -> the magnitude, 3 -> a zero byte) and
 * `sign80` the E4M3 sign bit.  `amax_idx` is the largest magnitude index in this
 * half, which is all the block amax needs -- the magnitudes are ordered, so the
 * larger index IS the larger magnitude.
 *
 * This replaced, per 16 weights per lane, a 16-step int8->float->fmax scan and
 * 16 int8->float->E4M3 conversions (census L198: 0.27% of the kernel's
 * instructions were the MMA itself). */
struct iq2_half_e4m3 {
    uint32_t sel;       ///< four byte-permute nibbles, each 0/1/2, or 3 for zero
    uint32_t sign80;    ///< 0x80 in each byte whose weight is negative
    int       amax_idx; ///< 0..2, or 3 ("no weights here", magnitude 0)
};

__device__ __forceinline__ static float iq2_mag_value(int idx) {
    return idx == 0 ? 8.0f : (idx == 1 ? 25.0f : (idx == 2 ? 43.0f : 0.0f));
}

__device__ __forceinline__ iq2_half_e4m3 iq2_decode_half_e4m3(
        uint2 code, const uint2 * __restrict__ s_gsel, int chunk, bool row_ok) {
    iq2_half_e4m3 h;
    if (!row_ok) {
        h.sel      = 0x3333u;   /* every nibble picks the zero byte */
        h.sign80   = 0u;
        h.amax_idx = 3;         /* iq2_mag_value(3) == 0.0f */
        return h;
    }
    const int group = chunk >> 1;
    const int hi = chunk & 1;
    const uint8_t aux = (uint8_t)(code.x >> (8 * group));
    const uint2 packed = s_gsel[aux];
    const uint32_t half = hi ? packed.y : packed.x;
    const uint32_t signs8 = ds4_unpack_ksigns((uint8_t)(code.y >> (7 * group)));
    const uint32_t selbits = hi ? 0x80402010u : 0x08040201u;
    h.sel      = half & 0xFFFFu;
    h.sign80   = __vcmpne4(signs8 & selbits, 0) & 0x80808080u;
    h.amax_idx = (int)(half >> 16);
    return h;
}


/* A8 (gap B): the same tile, but as MXFP8 operands for the block-scaled MMA.
 *
 * This is NOT new numerics.  The block-amax -> shared-exponent -> e4m3 encode is
 * exactly what mxfp8_quant_act_kernel does on the dense activation path; it is
 * called here through pulsar_cuda_mx.cuh so there is ONE definition of the
 * exponent arithmetic rather than a fourth copy.  What differs from the dense
 * case is only where the values come from (IQ2 codes dequantised in-register,
 * not f32 in memory) and where they go (MMA fragment registers plus the sfa
 * scale register, not an mx_sfoff-swizzled slab).
 *
 * REDUCTION WIDTH.  Per lane, x[0]/x[2] hold 8 of row `group`'s 32 k-values and
 * x[1]/x[3] hold 8 of row `group+8`'s, so a row's 32 values live in the 4 lanes
 * sharing `group` (d2r_group() = lane>>2, d2r_tig() = lane&3 -- consecutive).
 * The block amax is therefore a QUAD reduction, xor 1 and 2, not a warp-wide
 * one -- the same narrowing the attention epilogue uses.
 *
 * SCALE REGISTER.  Each lane supplies ONE byte, and which row it belongs to
 * follows the layout the hardware expects: lanes with tig == 1 supply row
 * group+8, every other lane supplies row group.  Copied from the working
 * instance in pulsar_cuda_indexer_mxfp4.cu:420 -- a wrong guess here is a
 * silent wrong-operand bug, since the GEMM would read a well-formed scale
 * belonging to the wrong row.
 *
 * The effective weight is grid_value * d * ls * 0.125 (the IQ2 grid is 8x: the
 * {8,25,43} magnitudes are 8x the real {1, 3.125, 5.375}).  That PRODUCT is
 * what gets MX-quantised -- there is no scale left to fold afterwards, which is
 * the point: the hardware applies it.
 *
 * This comment used to read "NOT YET WIRED: the B tile still stages q8_1, so
 * nothing calls this."  Both halves are false now: the B tile stages E4M3
 * (ds4_quantize_e4m3.cu), and make_iq2_A_tile_e4m3 below builds its quads from
 * this.  Left as a note because the stale version was read as evidence that
 * the routed path still ran q8_1 activations -- see ledger L065.
 *
 * L199: the three E4M3 bytes a row can produce, in magnitude order, with a zero
 * in byte 3 so a byte-permute nibble of 3 yields a zero weight.  Built ONCE per
 * row-pair; a weight is then a byte-permute plus a sign XOR.  Bit-identical to
 * converting each weight: 8/25/43 are exact in float, and e4m3(-x) is the sign
 * flip of e4m3(x) (the conversion rounds magnitudes symmetrically), so the
 * sign-after-convert order does not change a byte -- including the saturating
 * and signed-zero cases. */
__device__ __forceinline__ static uint32_t iq2_mag_triple_e4m3(float scale) {
    const float mag[3] = {8.0f, 25.0f, 43.0f};
    uint32_t out = 0;
#pragma unroll
    for (int m = 0; m < 3; ++m) {
        const __nv_fp8_e4m3 e = (__nv_fp8_e4m3)(mag[m] * scale);
        out |= ((uint32_t)*(const uint8_t *)&e) << (8 * m);
    }
    return out;
}

/* L202: the A tile is read STRAIGHT FROM GLOBAL.  Under the k-major layout the
 * word this lane wants sits at q[(k256*8 + pair)*M + row]; the warp's lanes ask
 * for rows warp_row0+0..7 (and +8..15), four lanes per row, so the hardware
 * broadcasts and the warp's whole request is 128 contiguous bytes.  Same for the
 * block scale at d[k256*M + row].  Measured on the access pattern alone: 88 GB/s
 * reading the old layout this way, 182 staging it, 218 reading this one (L201). */
template <typename TileA>
__device__ __forceinline__ void make_iq2_A_tile_e4m3(
        TileA &A, uint32_t &sfa,
        uint2 code0, uint2 code1, float d0, float d1,
        const uint2 * __restrict__ s_gsel,
        bool row0_ok, bool row1_ok, int tig) {

    const iq2_half_e4m3 h0a = iq2_decode_half_e4m3(code0, s_gsel, tig,     row0_ok);
    const iq2_half_e4m3 h1a = iq2_decode_half_e4m3(code1, s_gsel, tig,     row1_ok);
    const iq2_half_e4m3 h0b = iq2_decode_half_e4m3(code0, s_gsel, tig + 4, row0_ok);
    const iq2_half_e4m3 h1b = iq2_decode_half_e4m3(code1, s_gsel, tig + 4, row1_ok);

    const int ls0 = (int)(code0.y >> 27) | 1;
    const int ls1 = (int)(code1.y >> 27) | 1;
    const float dA0 = d0 * (float)ls0 * 0.125f;
    const float dA1 = d1 * (float)ls1 * 0.125f;

    /* L199: the block amax is the largest MAGNITUDE this lane holds, and the
     * grid's three magnitudes are ordered, so the larger index is the larger
     * magnitude -- no scan over the values. */
    float a0 = iq2_mag_value(max(h0a.amax_idx, h0b.amax_idx)) * fabsf(dA0);
    float a1 = iq2_mag_value(max(h1a.amax_idx, h1b.amax_idx)) * fabsf(dA1);
    /* quad reduction: lanes group*4 + 0..3 hold this row's whole 32-block */
    a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, 1));
    a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, 2));
    a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, 1));
    a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, 2));

    const int se0 = pulsar_mx_shared_exp(a0);
    const int se1 = pulsar_mx_shared_exp(a1);
    const float r0 = dA0 * exp2f(-(float)se0);
    const float r1 = dA1 * exp2f(-(float)se1);

    const uint32_t mag0 = iq2_mag_triple_e4m3(r0);
    const uint32_t mag1 = iq2_mag_triple_e4m3(r1);
    A.x[0] = (int)(__byte_perm(mag0, 0u, h0a.sel) ^ h0a.sign80);
    A.x[1] = (int)(__byte_perm(mag1, 0u, h1a.sel) ^ h1a.sign80);
    A.x[2] = (int)(__byte_perm(mag0, 0u, h0b.sel) ^ h0b.sign80);
    A.x[3] = (int)(__byte_perm(mag1, 0u, h1b.sel) ^ h1b.sign80);

    sfa = (uint32_t)pulsar_mx_scale_byte((tig == 1) ? se1 : se0);
}




/* MXFP8 arm of the k32 pair: same decomposition as mma_fold_iq2_k32_pair_t, but
 * both operands are E4M3 and the HARDWARE applies the ue8m0 scales, so there is
 * NO fold -- the accumulator is f32 and the MMA reads/writes it in place.
 *
 * WHERE THE SCALES COME FROM, which is the easy thing to get wrong:
 *   sfa -- from make_iq2_A_tile_e4m3, keyed on `tig` (tig == 1 supplies row
 *          group+8, every other lane supplies row group).
 *   sfb -- from the COLUMN GROUP, i.e. s_act[stage][nf][group], NOT from the
 *          accumulator columns c0/c1.  For
 *          m16n8k32 the B fragment's 8 columns map one per 4-lane group, so
 *          lane group g carries column g's scale.  Using the c0/c1 pair here
 *          would hand the MMA a well-formed scale for the WRONG column.
 * The ue8m0 byte is stored as a float by ds4_gather_mmq_e4m3 (which reuses
 * block_mx_act_mmq's d4 slot to keep the staging layout identical), hence the
 * cast back on read.
 *
 * GUARDED (!FullTile) CASE: out-of-range ROWS are already zeroed inside the A
 * tile, and out-of-range COLUMNS accumulate values the output store never
 * writes, since it bounds-checks each fragment element.  So no masking fold is
 * needed, unlike the integer path which had to mask after the MMA. */
template <bool FullTile, typename TileA, typename TileB, typename TileC,
          int T0, int T1, int NFrag>
__device__ __forceinline__ void mma_iq2_k32_pair_e4m3(
        float (&acc)[NFrag][TileC::ne],
        const uint2 (&c0)[4], const uint2 (&c1)[4], float d0, float d1,
        const uint2 * __restrict__ s_gsel,
        const block_mx_act_mmq (&s_act)[kActDepth][NFrag][8],
        int act_stage, bool raw_row0_ok, bool raw_row1_ok,
        int group, int tig, const volatile SmemInvariants &s_inv) {
    static_assert(T1 == T0 + 1, "expected adjacent k32 pair");
    static_assert(TileC::ne == 4, "expected m16n8 accumulator fragment");
    TileA A0;
    TileA A1;
    uint32_t sfa0 = 0;
    uint32_t sfa1 = 0;
    make_iq2_A_tile_e4m3(A0, sfa0, c0[T0 & 3], c1[T0 & 3], d0, d1, s_gsel,
                         raw_row0_ok, raw_row1_ok, tig);
    make_iq2_A_tile_e4m3(A1, sfa1, c0[T1 & 3], c1[T1 & 3], d0, d1, s_gsel,
                         raw_row0_ok, raw_row1_ok, tig);

    constexpr int k_in_act_0 = (T0 & 3) * 32;
    constexpr int k_in_act_1 = (T1 & 3) * 32;

    int nf_live = NFrag;
    if constexpr (!FullTile) {
        nf_live = (s_inv.col_count + 7) >> 3;
    }

#pragma unroll
    for (int nf = 0; nf < NFrag; ++nf) {
        if constexpr (!FullTile) {
            if (nf >= nf_live) {
                break;
            }
        }
        TileB B0;
        TileB B1;
        load_B_tile(B0, s_act, act_stage, nf, k_in_act_0);
        load_B_tile(B1, s_act, act_stage, nf, k_in_act_1);
        const uint32_t sfb0 = (uint32_t)s_act[act_stage][nf][group].d4[T0 & 3];
        const uint32_t sfb1 = (uint32_t)s_act[act_stage][nf][group].d4[T1 & 3];

        ds4_mma_m16n8k32_e4m3(acc[nf][0], acc[nf][1], acc[nf][2], acc[nf][3],
                              (uint32_t)A0.x[0], (uint32_t)A0.x[1],
                              (uint32_t)A0.x[2], (uint32_t)A0.x[3],
                              (uint32_t)B0.x[0], (uint32_t)B0.x[1], sfa0, sfb0);
        ds4_mma_m16n8k32_e4m3(acc[nf][0], acc[nf][1], acc[nf][2], acc[nf][3],
                              (uint32_t)A1.x[0], (uint32_t)A1.x[1],
                              (uint32_t)A1.x[2], (uint32_t)A1.x[3],
                              (uint32_t)B1.x[0], (uint32_t)B1.x[1], sfa1, sfb1);
    }
}

template <bool FullTile, typename TileA, typename TileB, typename TileC,
          int NFrag>
__device__ __forceinline__ void mma_fold_iq2_k128(
        float (&acc)[NFrag][TileC::ne],
        const uint2 * __restrict__ s_gsel,
        const block_mx_act_mmq (&s_act)[kActDepth][NFrag][8],
        int k128_iter, const volatile SmemInvariants &s_inv) {
    const uint2 * __restrict__ q_expert = (const uint2 *)s_inv.iq2_qs_base;
    const half * __restrict__ d_expert = (const half *)s_inv.iq2_dq_base;
    const int warp = d2r_warp();
    const int group = d2r_group();
    /* L202: the absolute row is now needed on BOTH arms -- it addresses the
     * weights directly, it is no longer just a bounds check. */
    const int warp_row0 = s_inv.cta_row0 + (warp << 4);
    const int abs_row0 = warp_row0 + group;
    bool row0_ok = true;
    bool row1_ok = true;
    if constexpr (!FullTile) {
        row0_ok = abs_row0 < s_inv.M;
        row1_ok = (abs_row0 + 8) < s_inv.M;
    }
    const int tig = d2r_tig();
    const int k256 = k128_iter >> 1;
    const int act_stage = d2r_act_stage(k128_iter);
    const int half_pair_base = (k128_iter & 1) ? 4 : 0;

    /* L203: fetch this whole k128 iteration's weights BEFORE any of its MMAs.
     *
     * L202 deleted the shared ring, which removed the transpose (mio_throttle
     * 21.6% -> 1.3%) but also removed the software prefetch that came with it:
     * loading each word where it is consumed puts every load in the dependency
     * chain, and DRAM latency went from 8.5% to 56.4% of stall samples.  The
     * four k32 tiles of one k128 iteration read eight independent words, so
     * issuing all eight together lets them overlap each other and leaves only
     * the first dequant waiting.  The block scale depends on (row, k256), not
     * on the word, so it is fetched once here instead of eight times. */
    /* L205: 32-bit indexing WITHIN the expert.  One expert holds nb*8*M code
     * words and nb*M scales -- 262144 and 32768 at the shipped shapes -- so the
     * whole inner loop addresses in 32 bits and the 64-bit offset registers and
     * their IMAD.WIDE pairs leave the hot path.  Registers are what caps this
     * kernel's occupancy (L202), so an address temporary is not free.  The
     * launcher refuses a shape that would not fit rather than wrapping. */
    const uint32_t rowM = (uint32_t)s_inv.M;
    const uint32_t d_off = (uint32_t)k256 * rowM + (uint32_t)abs_row0;
    const float d0 = row0_ok ? __half2float(d_expert[d_off])      : 0.0f;
    const float d1 = row1_ok ? __half2float(d_expert[d_off + 8u]) : 0.0f;
    uint2 c0[4];
    uint2 c1[4];
    uint32_t q_off = ((uint32_t)k256 * 8u + (uint32_t)half_pair_base) * rowM + (uint32_t)abs_row0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        c0[i] = row0_ok ? q_expert[q_off]      : make_uint2(0, 0);
        c1[i] = row1_ok ? q_expert[q_off + 8u] : make_uint2(0, 0);
        q_off += rowM;
    }

    if (half_pair_base == 0) {
        mma_iq2_k32_pair_e4m3<FullTile, TileA, TileB, TileC, 0, 1>(
            acc, c0, c1, d0, d1, s_gsel, s_act, act_stage,
            row0_ok, row1_ok, group, tig, s_inv);
        mma_iq2_k32_pair_e4m3<FullTile, TileA, TileB, TileC, 2, 3>(
            acc, c0, c1, d0, d1, s_gsel, s_act, act_stage,
            row0_ok, row1_ok, group, tig, s_inv);
    } else {
        mma_iq2_k32_pair_e4m3<FullTile, TileA, TileB, TileC, 4, 5>(
            acc, c0, c1, d0, d1, s_gsel, s_act, act_stage,
            row0_ok, row1_ok, group, tig, s_inv);
        mma_iq2_k32_pair_e4m3<FullTile, TileA, TileB, TileC, 6, 7>(
            acc, c0, c1, d0, d1, s_gsel, s_act, act_stage,
            row0_ok, row1_ok, group, tig, s_inv);
    }
}

template <bool FullTile, typename TileA, typename TileB, typename TileC, int NFrag>
__device__ __forceinline__ void iq2_d2r_mainloop(
        float (&acc)[NFrag][TileC::ne],
        block_mx_act_mmq (&s_act)[kActDepth][NFrag][8],
        const uint2 * __restrict__ s_gsel,
        const volatile SmemInvariants &s_inv) {
#pragma unroll
    for (int pf = 0; pf < kStages; ++pf) {
        if (pf < s_inv.k128_iters) {
            if constexpr (FullTile) {
                issue_act_prefetch_fast(s_act, s_inv, d2r_act_stage(pf), pf);
            } else {
                issue_act_prefetch<false>(s_act, s_inv, d2r_act_stage(pf), pf, d2r_tid());
            }
        }
    }

    for (int k128_iter = 0;; ++k128_iter) {
        if (k128_iter >= s_inv.k128_iters) {
            break;
        }
        /* Per-iteration wait (L099, simplified by L202).  Only the ACT stream
         * commits now: the prologue issues act(0..kStages-1) and iteration j
         * issues act(j+kStages) AFTER this wait, so at iteration j the group
         * this iteration consumes is act(j) and the only group committed after
         * it is act(j+1) -- issued by iteration j-1, or by the prologue when
         * j+1 < kStages.  Keep that one and wait for the rest. */
        const int keep = ((k128_iter + 1) < s_inv.k128_iters) ? 1 : 0;
        cp_async_wait_keep(keep);
        __syncthreads();
        if constexpr (FullTile) {
            mma_fold_iq2_k128<true, TileA, TileB, TileC>(
                acc, s_gsel, s_act, k128_iter, s_inv);
        } else {
            mma_fold_iq2_k128<false, TileA, TileB, TileC>(
                acc, s_gsel, s_act, k128_iter, s_inv);
        }

        /* No second block barrier (L099).  The act issue below writes slot
         * (k+kStages) % kActDepth, disjoint from the slot any warp can still be
         * reading (depth > distance).  The raw issue that used to need a
         * __syncwarp() here is gone with the ring (L202). */
        const int pf_iter = k128_iter + kStages;
        if (pf_iter < s_inv.k128_iters) {
            if constexpr (FullTile) {
                issue_act_prefetch_fast(s_act, s_inv, d2r_act_stage(pf_iter), pf_iter);
            } else {
                issue_act_prefetch<false>(s_act, s_inv, d2r_act_stage(pf_iter), pf_iter, d2r_tid());
            }
        }
    }
}






















template <int TileN>
__global__ __launch_bounds__(kThreads, 2)
void d2r_build_worklist_kernel(const int32_t * __restrict__ expert_bounds,
                               int * __restrict__ work,
                               int * __restrict__ n_items_out,
                               int n_experts) {
    static_assert(TileN > 0, "worklist tile width must be positive");
    __shared__ int scan[kThreads];
    __shared__ int running;
    __shared__ int chunk_base;

    const int tid = (int)threadIdx.x;
    if (tid == 0) {
        running = 0;
    }
    __syncthreads();

    for (int base = 0; base < n_experts; base += kThreads) {
        const int expert = base + tid;
        int tiles = 0;
        if (expert < n_experts) {
            const int count = expert_bounds[expert + 1] - expert_bounds[expert];
            tiles = count > 0 ? ((count + TileN - 1) / TileN) : 0;
        }
        scan[tid] = tiles;
        __syncthreads();

#pragma unroll
        for (int offset = 1; offset < kThreads; offset <<= 1) {
            const int add = tid >= offset ? scan[tid - offset] : 0;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }

        if (tid == 0) {
            chunk_base = running;
        }
        __syncthreads();

        if (expert < n_experts && tiles > 0) {
            const int exclusive = tid == 0 ? 0 : scan[tid - 1];
            const int out_base = chunk_base + exclusive;
            for (int jt = 0; jt < tiles; ++jt) {
                work[out_base + jt] = (expert << 16) | jt;
            }
        }
        __syncthreads();

        if (tid == 0) {
            running += scan[kThreads - 1];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *n_items_out = running;
    }
}



/* IQ2 routed-expert GEMM: E4M3 operands into the block-scaled MXFP8 MMA with a
 * hardware-applied ue8m0 scale.  There is no second arm and no selector.
 *
 * There used to be: PULSAR_MOE_IQ2_ARM chose between q8_1 int8 + integer MMA +
 * a software scale fold, and this.  The comparison concluded in favour of MXFP8
 * (it is what the source model computes in), so per [[no-hot-path-flags]] the
 * losing arm was DELETED rather than left behind as a flag -- along with
 * make_iq2_A_tile, the fold helpers, and the arm-selecting kernel argument.
 * Callers stage E4M3; nothing here can disagree with them about the format. */
/* L109 N2 verdict (corrected 2026-08-25 late): minBlocks 3 was reverted on a
 * COMPOSITE A/B later shown to be confounded (N1's acceptance tax + N4's
 * pipeline regression dominated it) -- N2 was never measured solo and its
 * true cost is UNKNOWN. The spill risk on a 128-register kernel is real, so
 * 2 stays until someone runs the clean single-lever A/B this one never got. */
/* L200: NT is the assignment-slot tile width.  The 2-vs-4 minimum-blocks hint
 * is the point of the narrow instantiation: with one accumulator fragment
 * instead of eight the compiler can fit 4 blocks/SM in the register file, and
 * the narrow act staging fits them in shared. */
template <int NT>
__global__ __launch_bounds__(kThreads, NT == 64 ? 2 : 4)
void gateup_iq2_d2r_pair_kernel(const void * __restrict__ gate_soa,
                                const void * __restrict__ up_soa,
                                const block_mx_act_mmq * __restrict__ act,
                                const int32_t * __restrict__ ids_dst,
                                const int32_t * __restrict__ expert_bounds,
                                const int * __restrict__ work,
                                const int * __restrict__ n_items_ptr,
                                float * __restrict__ out_gate,
                                float * __restrict__ out_up,
                                int M, int K, int n_assign, int E) {
#if defined(TURING_MMA_AVAILABLE)
    const int n_items = *n_items_ptr;
    if ((int)blockIdx.y >= n_items) {
        return;
    }

    const int packed = work[blockIdx.y];
    const int expert = packed >> 16;
    const int jt = packed & 0xFFFF;
    const int leg = (int)blockIdx.z;
    if (expert >= E || leg >= 2) {
        return;
    }

    constexpr int NFrag = NT / 8;
    const int col_lo = expert_bounds[expert] + jt * NT;
    const int col_hi_full = expert_bounds[expert + 1];
    const int col_tile_hi = (col_hi_full < col_lo + NT) ? col_hi_full : (col_lo + NT);
    if (col_lo >= col_tile_hi) {
        return;
    }

    /* A/B: `int` here means "one 32-bit register holding four E4M3 bytes",
     * not an integer element -- the packing is done by make_iq2_A_tile_e4m3 and
     * load_B_tile.  C is genuinely f32 (the FFMA accumulator).  It was
     * tile<16,8,int> -- a leftover of the deleted s8/s32 arm -- which was
     * harmless only because tile::ne is T-independent and only ne/get_i/get_j
     * are ever read; the stated type now matches what flows. */
    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile<8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, float>;

    /* s_act / act_iter_base / act_stage etc. were s_q8 / q8_iter_base / q8_stage
     * until 2026-08-18.  The bytes have been E4M3 since the A8 campaign; only
     * the names still said int8, and a name that outlives its format is not
     * cosmetic -- reading `block_q8_1_mmq` as evidence the MoE still ran q8_1 is
     * exactly how L065 got written and retracted.  See ds4_act_block.cuh. */
    __shared__ __align__(16) block_mx_act_mmq s_act[kActDepth][NFrag][8];
    __shared__ __align__(16) uint2 s_gsel[256];
    __shared__ __align__(16) volatile SmemInvariants s_inv;
    /* Scatter-index staging: one output column per tile lane. */
    __shared__ int s_out_cols[NT];

    const void *W_soa = leg == 0 ? gate_soa : up_soa;
    float *out = leg == 0 ? out_gate : out_up;
    const int cta_row0 = (int)blockIdx.x * kMTile;
    const int warp_row0 = cta_row0 + d2r_warp() * 16;
    const int nb = K >> 8;

    const bool full_warp_tile = (warp_row0 + 15 < M) && (col_lo + NT <= col_hi_full);

    /* L199: stage the byte-permute selectors and per-half amax index rather than
     * the grid values themselves -- same 2 KiB, and the dequant becomes a
     * lookup.  See iq2_decode_half_e4m3. */
    for (int i = d2r_tid(); i < 256; i += kThreads) {
        const uint64_t g = iq2xxs_grid[i];
        uint32_t sel[2] = {0u, 0u};
        uint32_t amax[2] = {0u, 0u};
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const uint32_t v = (uint32_t)((g >> (8 * b)) & 0xFFu);
            const uint32_t idx = (v == 8u) ? 0u : ((v == 25u) ? 1u : 2u);
            const int half = b >> 2;
            sel[half] |= idx << (4 * (b & 3));
            amax[half] = amax[half] > idx ? amax[half] : idx;
        }
        s_gsel[i] = make_uint2(sel[0] | (amax[0] << 16), sel[1] | (amax[1] << 16));
    }
    /* L207: zero the activation ring once, so the per-stage issue can skip the
     * columns this tile does not have instead of re-zeroing them (see
     * issue_act_prefetch_one).  Slots for real columns are overwritten by their
     * cp.async before any read. */
    {
        int *act_words = reinterpret_cast<int *>(&s_act[0][0][0]);
        constexpr int act_n_words = (int)(sizeof(s_act) / sizeof(int));
#pragma unroll 1
        for (int i = d2r_tid(); i < act_n_words; i += kThreads) {
            act_words[i] = 0;
        }
    }
    if (d2r_tid() == 0) {
        const uint64_t nblk = (uint64_t)E * (uint64_t)M * (uint64_t)nb;
        const uint64_t dq_bytes = (nblk * 2ull + 63ull) & ~63ull;
        s_inv.w_base = (const char *)W_soa;
        /* L202: fold this block's expert offset in once.  Type 44 orders each
         * plane (expert, k, word, row), so the expert stride is nb*M halves in
         * the d plane and nb*8*M words in the q plane. */
        s_inv.iq2_dq_base = reinterpret_cast<const half *>(W_soa) +
                            (uint64_t)expert * (uint64_t)nb * (uint64_t)M;
        s_inv.iq2_qs_base =
            reinterpret_cast<const uint2 *>(reinterpret_cast<const char *>(W_soa) + dq_bytes) +
            (uint64_t)expert * (uint64_t)nb * 8ull * (uint64_t)M;
        s_inv.act_tile_base = reinterpret_cast<const char *>(act) + (uint64_t)col_lo * sizeof(block_mx_act_mmq);
        s_inv.out = out;
        s_inv.sc_off_bytes = 0;
        s_inv.qs_off_bytes = 0;
        s_inv.act_k128_stride_bytes = (uint32_t)((uint64_t)n_assign * sizeof(block_mx_act_mmq));
        s_inv.nb = nb;
        s_inv.k128_iters = K >> 7;
        s_inv.M = M;
        s_inv.cta_row0 = cta_row0;
        s_inv.col_lo = col_lo;
        s_inv.col_count = col_tile_hi - col_lo;
    }
    if (d2r_lane() == 0) {
    }
    if (d2r_tid() < col_tile_hi - col_lo) {
        s_out_cols[d2r_tid()] = ids_dst[col_lo + d2r_tid()];
    }
    __syncthreads();

    float acc[NFrag][tile_C::ne] = {};

    if (full_warp_tile) {
        iq2_d2r_mainloop<true, tile_A, tile_B, tile_C>(
            acc, s_act, s_gsel, s_inv);
    } else {
        iq2_d2r_mainloop<false, tile_A, tile_B, tile_C>(
            acc, s_act, s_gsel, s_inv);
    }

    const int out_col_lo = s_inv.col_lo;
    const int out_col_hi = out_col_lo + s_inv.col_count;
    const int out_warp_row0 = s_inv.cta_row0 + (d2r_warp() << 4);
    const int out_M = s_inv.M;
    float *out_base = s_inv.out;
#pragma unroll
    for (int nf = 0; nf < NFrag; ++nf) {
        const int col_frag0 = out_col_lo + nf * 8;
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = out_warp_row0 + tile_C::get_i(l);
            const int col = col_frag0 + tile_C::get_j(l);
            if (row < out_M && col < out_col_hi) {
                const int out_col = s_out_cols[col - out_col_lo];
                out_base[(uint64_t)out_col * (uint64_t)out_M + (uint64_t)row] = acc[nf][l];
            }
        }
    }
#else
    (void)gate_soa;
    (void)up_soa;
    (void)act;
    (void)ids_dst;
    (void)expert_bounds;
    (void)work;
    (void)n_items_ptr;
    (void)out_gate;
    (void)out_up;
    (void)M;
    (void)K;
    (void)n_assign;
    (void)E;
#endif
}


static int64_t d2r_work_capacity_for_tile(
        int64_t ncols_max, int n_experts, int tile_n) {
    if (ncols_max <= 0 || n_experts <= 0 || tile_n <= 0) {
        return 0;
    }
    return (ncols_max + tile_n - 1) / tile_n + (int64_t)n_experts;
}

/* L200: which tile width a launch uses.  Below the threshold an expert almost
 * certainly holds a single tile at either width, so the narrow tile costs no
 * extra weight traffic and buys 2x the resident blocks; above it an expert
 * spans several narrow tiles and would RE-READ its whole weight tile once per
 * tile, which is the one way this choice can lose.  ne_get_rows is the total
 * assignment slots (tokens x top-k), so 128 covers the entire decode and
 * speculative-verify regime (<= 16 rows, the M-neutral range) with room to
 * spare, and prefill chunks land far above it. */
constexpr int64_t kD2RNarrowMaxSlots = 128;

static bool d2r_use_narrow_tile(int64_t ne_get_rows) {
    return ne_get_rows > 0 && ne_get_rows <= kD2RNarrowMaxSlots;
}

/* The worklist is one entry per (expert, tile), so the NARROW tile produces the
 * most entries: size every scratch allocation for it, whichever width a given
 * launch then picks (L200). */
static int64_t d2r_work_capacity(int64_t ncols_max, int n_experts) {
    return d2r_work_capacity_for_tile(ncols_max, n_experts, kNTileNarrow);
}


} // namespace


/* L199: the dequant names a weight's magnitude with a 2-bit nibble, which is
 * sound only while the vendored grid holds exactly {8, 25, 43}.  Check the table
 * itself once rather than trust the comment that named them: a fourth magnitude
 * would otherwise be packed silently as 43.  Refusing here fails the model load
 * closed -- D2R is the only IQ2 arm (VENDOR.md). */
static bool ds4_iq2_grid_magnitudes_ok() {
    uint64_t grid[256];
    if (cudaMemcpyFromSymbol(grid, iq2xxs_grid, sizeof(grid)) != cudaSuccess) {
        fprintf(stderr, "ds4_mmq_d2r: could not read iq2xxs_grid to validate it\n");
        return false;
    }
    for (int i = 0; i < 256; ++i) {
        for (int b = 0; b < 8; ++b) {
            const uint32_t v = (uint32_t)((grid[i] >> (8 * b)) & 0xFFu);
            if (v != 8u && v != 25u && v != 43u) {
                fprintf(stderr,
                        "ds4_mmq_d2r: iq2xxs_grid[%d] byte %d is %u, not one of "
                        "{8,25,43} -- the D2R magnitude index cannot name it\n",
                        i, b, v);
                return false;
            }
        }
    }
    return true;
}


bool ds4_mmq_iq2_xxs_moe_d2r_available(int cc) {
    static int cached_cc = -1;
    static int cached = 0;
    if (cached_cc != cc) {
        cached_cc = cc;
        cached = (GGML_CUDA_CC_IS_NVIDIA(cc) &&
                  ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE &&
                  ds4_iq2_grid_magnitudes_ok()) ? 1 : 0;
    }
    return cached != 0;
}


size_t ds4_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(int64_t ncols_max, int n_experts) {
    const int64_t capacity = d2r_work_capacity(ncols_max, n_experts);
    if (capacity <= 0 || capacity > (int64_t)(INT_MAX - 1)) {
        return 0;
    }
    return (size_t)capacity * sizeof(int) + sizeof(int);
}



/* ==== L210: the DECODE tier -- a GEMV over the k-major artifact ==============
 *
 * The D2R tile above is an m16n8k32 block-scaled MMA whose N dimension is the
 * assignments of ONE expert.  At decode a routed expert receives ~1 token
 * whatever the row count (6 of 256 experts per token), so the tile runs at
 * ~1/8 N-fill on every served shape and reads its weights at ~60% of DRAM
 * roofline (L208 census: 7.8 ms/token where the bytes take 4.8).  The right
 * shape at decode is a GEMV.
 *
 * Type 44 is K-MAJOR (rows fastest): the word for (row, k256, cw) sits at
 * q[(k256*8 + cw)*M + row] and its scale at d[k256*M + row].  A warp-per-row
 * GEMV (what the row-major forks run) would stride M between a lane's
 * consecutive words; instead a WARP OWNS 32 CONSECUTIVE ROWS and each load is
 * one 256-byte contiguous slab, the layout's natural access.  The block's 8
 * warps split K (k256 blocks w, w+8, ...) and reduce through shared memory in
 * fixed warp order, so the result is deterministic.  Gate and up ride the same
 * pass and share the activation reads.  Activations are the producer's E4M3 +
 * ue8m0 blocks ([k128][assignment]), dequantised once per block to f32.
 *
 * Numerics: the weight is the EXACT f32 value d * ls * 0.125 * {8,25,43} * sign;
 * the D2R tile rounds it to E4M3 under a shared exponent first.  Different
 * bytes by construction -- the reference gate grades it, the byte gates
 * re-anchor (rows/L210.md).  Output contract is the D2R kernel's:
 * out[ids_dst[col] * M + row], f32. */
constexpr int kDecodeGemvRows       = 32;    ///< rows per warp == lanes
constexpr int kDecodeGemvWarps      = 8;     ///< K split
constexpr int kDecodeGemvMaxK       = 5120;  ///< shared f32 activation slab (V4.1 n_embd, L218; 20 KB)

__device__ __forceinline__ static float d2r_e4m3_to_f32(uint8_t bits) {
    return (float)(*reinterpret_cast<const __nv_fp8_e4m3 *>(&bits));
}

__global__ void __launch_bounds__(kDecodeGemvRows * kDecodeGemvWarps)
gateup_iq2_decode_gemv_kernel(const void * __restrict__ gate_soa,
                              const void * __restrict__ up_soa,
                              const block_mx_act_mmq * __restrict__ act,
                              const int32_t * __restrict__ ids_dst,
                              const int32_t * __restrict__ expert_bounds,
                              float * __restrict__ out_gate,
                              float * __restrict__ out_up,
                              int M, int K, int n_assign, int E) {
    __shared__ float    s_x[kDecodeGemvMaxK];
    __shared__ float    s_red[kDecodeGemvWarps][kDecodeGemvRows][2];
    __shared__ uint64_t s_grid[256];   /* the IQ2 grid: constant memory serialises divergent indices (L199) */
    __shared__ int      s_expert;

    const int col  = blockIdx.y;                 /* assignment (expert-sorted) */
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * kDecodeGemvRows + lane;
    const int row  = blockIdx.x * kDecodeGemvRows + lane;
    if (col >= n_assign) return;

    /* Which expert owns this assignment: expert_bounds is E+1 ascending offsets. */
    if (tid == 0) {
        int lo = 0, hi = E - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (expert_bounds[mid] <= col) lo = mid; else hi = mid - 1;
        }
        s_expert = lo;
    }
    /* The grid table: 256 x 8 bytes, one per thread, indexed by data-dependent
     * bytes below -- from shared memory that is one wavefront per lookup, from
     * __constant__ it would be one per distinct byte in the warp. */
    s_grid[tid] = iq2xxs_grid[tid];
    /* Stage this assignment's activation row as f32: block i holds k in
     * [128 i, 128 i + 128) as 4 groups of 32 e4m3 under one ue8m0 byte each
     * (d4[] carries the byte as a float). */
    const int n_k128 = K >> 7;
    for (int i = tid; i < n_k128 * 4; i += kDecodeGemvRows * kDecodeGemvWarps) {
        const int blk = i >> 2, grp = i & 3;
        const block_mx_act_mmq &b = act[(uint64_t)blk * (uint64_t)n_assign + (uint64_t)col];
        const float sc = exp2f(b.d4[grp] - 127.0f);
        const int8_t *q = b.qs + grp * 32;
        float *xo = s_x + blk * 128 + grp * 32;
#pragma unroll
        for (int j = 0; j < 32; ++j) xo[j] = d2r_e4m3_to_f32((uint8_t)q[j]) * sc;
    }
    __syncthreads();

    const int expert = s_expert;
    const int nb = K >> 8;                       /* k256 blocks per row */
    const uint64_t nblk = (uint64_t)E * (uint64_t)M * (uint64_t)nb;
    const uint64_t dq_bytes = (nblk * 2ull + 63ull) & ~63ull;
    const half  *dg = reinterpret_cast<const half *>(gate_soa) + (uint64_t)expert * nb * (uint64_t)M;
    const half  *du = reinterpret_cast<const half *>(up_soa)   + (uint64_t)expert * nb * (uint64_t)M;
    const uint2 *qg = reinterpret_cast<const uint2 *>(reinterpret_cast<const char *>(gate_soa) + dq_bytes) +
                      (uint64_t)expert * nb * 8ull * (uint64_t)M;
    const uint2 *qu = reinterpret_cast<const uint2 *>(reinterpret_cast<const char *>(up_soa) + dq_bytes) +
                      (uint64_t)expert * nb * 8ull * (uint64_t)M;
    const bool row_ok = row < M;

    float acc_g = 0.0f, acc_u = 0.0f;
    for (int b256 = warp; b256 < nb; b256 += kDecodeGemvWarps) {
        if (!row_ok) break;
        const float dgb = __half2float(dg[(uint64_t)b256 * (uint64_t)M + row]);
        const float dub = __half2float(du[(uint64_t)b256 * (uint64_t)M + row]);
        const uint2 *qgb = qg + ((uint64_t)b256 * 8ull) * (uint64_t)M + row;
        const uint2 *qub = qu + ((uint64_t)b256 * 8ull) * (uint64_t)M + row;
        const float *xb = s_x + b256 * 256;
        /* All 16 code words of this k256 block (8 gate, 8 up; 256 contiguous bytes
         * per warp each) are fetched before any of them is used: the round-3
         * profile was 77% long_scoreboard with one or two loads in flight per
         * warp.  Same lesson as L203 on the tile. */
        uint2 cgw[8], cuw[8];
#pragma unroll
        for (int cw = 0; cw < 8; ++cw) {
            cgw[cw] = qgb[(uint64_t)cw * (uint64_t)M];
            cuw[cw] = qub[(uint64_t)cw * (uint64_t)M];
        }
#pragma unroll
        for (int cw = 0; cw < 8; ++cw) {
            const uint2 cg = cgw[cw];
            const uint2 cu = cuw[cw];
            /* The activations come out of shared memory as two float4 broadcasts
             * per 8-weight group -- ONE shared load per 4 MACs per matrix. The
             * first cut loaded one float per weight and saturated the LSU pipe
             * (ncu: memory 9.6%, short_scoreboard 49%, tex_throttle 27%). */
            const float4 *x4 = reinterpret_cast<const float4 *>(xb + cw * 32);
            float sg = 0.0f, su = 0.0f;
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const uint64_t grid_g = s_grid[(cg.x >> (8 * g)) & 0xffu];
                const uint64_t grid_u = s_grid[(cu.x >> (8 * g)) & 0xffu];
                const uint32_t sgn_g = ds4_unpack_ksigns((uint8_t)((cg.y >> (7 * g)) & 0x7fu));
                const uint32_t sgn_u = ds4_unpack_ksigns((uint8_t)((cu.y >> (7 * g)) & 0x7fu));
                const float4 xa = x4[g * 2], xb4 = x4[g * 2 + 1];
                const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb4.x, xb4.y, xb4.z, xb4.w};
                const uint32_t glo = (uint32_t)grid_g, ghi = (uint32_t)(grid_g >> 32);
                const uint32_t ulo = (uint32_t)grid_u, uhi = (uint32_t)(grid_u >> 32);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    /* byte j of the 8-byte grid entry, zero-extended in one op */
                    const uint32_t bg = __byte_perm(j < 4 ? glo : ghi, 0u, 0x4440u | (uint32_t)(j & 3));
                    const uint32_t bu = __byte_perm(j < 4 ? ulo : uhi, 0u, 0x4440u | (uint32_t)(j & 3));
                    /* sign = bit j of the sign byte, moved to the float sign bit */
                    const float mg = __uint_as_float(__float_as_uint((float)bg) | (((sgn_g >> j) & 1u) << 31));
                    const float mu = __uint_as_float(__float_as_uint((float)bu) | (((sgn_u >> j) & 1u) << 31));
                    sg = fmaf(mg, xv[j], sg);
                    su = fmaf(mu, xv[j], su);
                }
            }
            const float lsg = (float)((int)(cg.y >> 27) | 1) * 0.125f;
            const float lsu = (float)((int)(cu.y >> 27) | 1) * 0.125f;
            acc_g = fmaf(dgb * lsg, sg, acc_g);
            acc_u = fmaf(dub * lsu, su, acc_u);
        }
    }
    s_red[warp][lane][0] = acc_g;
    s_red[warp][lane][1] = acc_u;
    __syncthreads();
    if (warp == 0 && row_ok) {
        float g = 0.0f, u = 0.0f;
#pragma unroll
        for (int w = 0; w < kDecodeGemvWarps; ++w) { g += s_red[w][lane][0]; u += s_red[w][lane][1]; }
        const uint64_t o = (uint64_t)ids_dst[col] * (uint64_t)M + (uint64_t)row;
        out_gate[o] = g;
        out_up[o] = u;
    }
}

int ds4_mmq_iq2_xxs_moe_d2r_pair_launch(const void *gate_soa,
                                         const void *up_soa,
                                         int64_t soa_blocks,
                                         const void *act,
                                         const int32_t *ids_dst,
                                         const int32_t *expert_bounds,
                                         float *out_gate,
                                         float *out_up,
                                         int M,
                                         int K,
                                         int64_t ne_get_rows,
                                         int n_experts,
                                         void *worklist_scratch,
                                         size_t worklist_scratch_bytes,
                                         cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_moe_d2r_pair_launch";
    const int dev = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_iq2_xxs_moe_d2r_available(cc)) {
        return 1;
    }
    if (!gate_soa || !up_soa || !act || !ids_dst || !expert_bounds || !out_gate || !out_up ||
        !worklist_scratch || M <= 0 || K <= 0 || K % 256 != 0 || ne_get_rows <= 0 ||
        ne_get_rows > INT_MAX || n_experts <= 0) {
        return -1;
    }

    /* L205: the kernel addresses within one expert in 32 bits (nb*8*M code words
     * in the q plane).  Refuse a shape that would not fit rather than wrap. */
    if ((uint64_t)(K >> 8) * 8ull * (uint64_t)M > (uint64_t)UINT32_MAX) {
        fprintf(stderr, "%s: expert too large for 32-bit intra-expert indexing (K=%d M=%d)\n",
                tag, K, M);
        return -1;
    }

    const int64_t expected_soa_blocks =
        (int64_t)n_experts * (int64_t)M * (int64_t)(K >> 8);
    if (soa_blocks < expected_soa_blocks) {
        return -1;
    }

    /* L210: the decode tier.  A decode step's few assignments (<= 16 tokens x
     * 6 experts) is the shape where the MMA tile runs N-starved; the GEMV reads
     * the same k-major bytes at roofline.  Different arithmetic from the tile
     * (exact f32 weights, split-K fma order), so WHICH rows take it is a
     * numerics boundary and follows L167's rule, stated once at the routed MoE
     * FFN dispatch (pulsar_cuda_moe.cu, "small-batch FFN"): DECODE rows take
     * one arithmetic at ANY width, so the whole M-neutral range is one kernel
     * and no cap sits inside it; PREFILL rows never take it, so a chunk's bytes
     * do not depend on its size and a mixed step's prefill pass equals a solo
     * prefill.  The predicate is the same global that rule reads: the mixed
     * step's two-pass split leaves it at n_tokens for the decode pass and 0 for
     * the prefill pass.  The setter bounds decode rows at
     * PULSAR_GPU_MNEUTRAL_ROWS_MAX, so the grid's y extent is bounded with it.
     * The byte gates re-anchor on the decode blob; the reference gate grades
     * prefill logits and never sees this arm (rows/L210.md). */
    if (pulsar_gpu_matmul_batch_decode_rows() > 0) {
        if (K > kDecodeGemvMaxK) {
            fprintf(stderr, "%s: decode GEMV tier holds K <= %d in shared memory, got K=%d -- refusing\n",
                    tag, kDecodeGemvMaxK, (int)K);
            return -1;
        }
        static int announced = 0;
        if (!announced) {
            announced = 1;
            fprintf(stderr, "pulsar: L210 routed IQ2 gate/up decode tier = k-major GEMV (decode rows, any width)\n");
        }
        const dim3 grid((unsigned)((M + kDecodeGemvRows - 1) / kDecodeGemvRows), (unsigned)ne_get_rows, 1);
        const dim3 block(kDecodeGemvRows, kDecodeGemvWarps, 1);
        gateup_iq2_decode_gemv_kernel<<<grid, block, 0, stream>>>(
            gate_soa, up_soa, (const block_mx_act_mmq *)act, ids_dst, expert_bounds,
            out_gate, out_up, M, K, (int)ne_get_rows, n_experts);
        const cudaError_t gerr = cudaGetLastError();
        if (gerr != cudaSuccess) {
            fprintf(stderr, "%s: decode GEMV launch failed: %s\n", tag, cudaGetErrorString(gerr));
            return -3;
        }
        return 0;
    }
    const bool narrow = d2r_use_narrow_tile(ne_get_rows);
    const int64_t capacity64 = d2r_work_capacity_for_tile(
        ne_get_rows, n_experts, narrow ? kNTileNarrow : kNTile);
    if (capacity64 <= 0 || capacity64 > (int64_t)(INT_MAX - 1)) {
        return -1;
    }
    const size_t needed = (size_t)capacity64 * sizeof(int) + sizeof(int);
    if (worklist_scratch_bytes < needed) {
        return -1;
    }

    int *work = (int *)worklist_scratch;
    int *n_items = work + capacity64;

    if (narrow) {
        d2r_build_worklist_kernel<kNTileNarrow><<<1, kThreads, 0, stream>>>(
            expert_bounds, work, n_items, n_experts);
    } else {
        d2r_build_worklist_kernel<kNTile><<<1, kThreads, 0, stream>>>(
            expert_bounds, work, n_items, n_experts);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: worklist builder launch failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    /* Same expert-major schedule as the down launch (see comment there). */
    const dim3 grid((unsigned)((M + kMTile - 1) / kMTile), (unsigned)capacity64, 2);
    const dim3 block(32, kWarps, 1);
    if (narrow) {
        gateup_iq2_d2r_pair_kernel<kNTileNarrow><<<grid, block, 0, stream>>>(
            gate_soa, up_soa, (const block_mx_act_mmq *)act, ids_dst, expert_bounds, work, n_items,
            out_gate, out_up, M, K, (int)ne_get_rows, n_experts);
    } else {
        gateup_iq2_d2r_pair_kernel<kNTile><<<grid, block, 0, stream>>>(
            gate_soa, up_soa, (const block_mx_act_mmq *)act, ids_dst, expert_bounds, work, n_items,
            out_gate, out_up, M, K, (int)ne_get_rows, n_experts);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: main kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}



/* pulsar (plan 41b): SINGLE-tensor IQ2_XXS D2R, for a routed DOWN whose tensor
 * is IQ2 rather than Q2_K (our v5mx).  Upstream has a Q2_K single-tensor D2R
 * (down_q2k_d2r_kernel) and an IQ2 PAIR D2R, but no IQ2 single -- so an IQ2
 * down fell back to stock mul_mat_q at 33.24 ms/layer against Entrpi's 9.78 ms
 * Q2_K D2R.  That was the entire remaining gap after alignment.
 *
 * No new kernel is needed.  gateup_iq2_d2r_pair_kernel already separates the
 * two projections along blockIdx.z ("leg"), selecting
 *     W_soa = leg == 0 ? gate_soa : up_soa
 *     out   = leg == 0 ? out_gate : out_up
 * so launching the SAME kernel with gridDim.z = 1 pins leg to 0 and computes
 * exactly one tensor.  up_soa/out_up are passed as the gate pointers and are
 * never dereferenced.
 *
 * Appended to cuda/mmq/ds4_mmq_d2r.cu; declared in ds4_mmq_d2r.cuh.
 */
int ds4_mmq_iq2_xxs_moe_d2r_single_launch(const void *W_soa,
                                          int64_t soa_blocks,
                                          const void *act,
                                          const int32_t *ids_dst,
                                          const int32_t *expert_bounds,
                                          float *out,
                                          int M,
                                          int K,
                                          int64_t ne_get_rows,
                                          int n_experts,
                                          void *worklist_scratch,
                                          size_t worklist_scratch_bytes,
                                           cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_moe_d2r_single_launch";
    const int dev = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_iq2_xxs_moe_d2r_available(cc)) {
        return -1;
    }
    if (!W_soa || !act || !ids_dst || !expert_bounds || !out) {
        return -1;
    }
    if (M <= 0 || K <= 0 || K % 256 != 0 || n_experts <= 0 || ne_get_rows <= 0) {
        return -1;
    }

    /* L205: the kernel addresses within one expert in 32 bits (nb*8*M code words
     * in the q plane).  Refuse a shape that would not fit rather than wrap. */
    if ((uint64_t)(K >> 8) * 8ull * (uint64_t)M > (uint64_t)UINT32_MAX) {
        fprintf(stderr, "%s: expert too large for 32-bit intra-expert indexing (K=%d M=%d)\n",
                tag, K, M);
        return -1;
    }

    const int64_t expected_soa_blocks =
        (int64_t)n_experts * (int64_t)M * (int64_t)(K >> 8);
    if (soa_blocks < expected_soa_blocks) {
        return -1;
    }
    const bool narrow = d2r_use_narrow_tile(ne_get_rows);
    const int64_t capacity64 = d2r_work_capacity_for_tile(
        ne_get_rows, n_experts, narrow ? kNTileNarrow : kNTile);
    if (capacity64 <= 0 || capacity64 > (int64_t)(INT_MAX - 1)) {
        return -1;
    }
    const size_t need = (size_t)capacity64 * sizeof(int) + sizeof(int);
    if (!worklist_scratch || worklist_scratch_bytes < need) {
        fprintf(stderr, "%s: worklist scratch too small\n", tag);
        return -1;
    }
    int *work = (int *)worklist_scratch;
    int *n_items = work + capacity64;

    if (narrow) {
        d2r_build_worklist_kernel<kNTileNarrow><<<1, kThreads, 0, stream>>>(
            expert_bounds, work, n_items, n_experts);
    } else {
        d2r_build_worklist_kernel<kNTile><<<1, kThreads, 0, stream>>>(
            expert_bounds, work, n_items, n_experts);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: worklist builder launch failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    /* z = 1: leg is pinned to 0, so only W_soa / out are ever touched. */
    const dim3 grid((unsigned)((M + kMTile - 1) / kMTile), (unsigned)capacity64, 1);
    const dim3 block(32, kWarps, 1);
    /* Callers MUST stage E4M3 -- there is no int8 arm left to fall back to.
     * This launch once read ds4_d2r_iq2_arm() itself, which ran the E4M3 MMA
     * against q8_1 bytes whenever the env was set (the arm-1 garbage).  With
     * one format there is nothing left to disagree about. */
    if (narrow) {
        gateup_iq2_d2r_pair_kernel<kNTileNarrow><<<grid, block, 0, stream>>>(
            W_soa, W_soa, (const block_mx_act_mmq *)act, ids_dst, expert_bounds, work, n_items,
            out, out, M, K, (int)ne_get_rows, n_experts);
    } else {
        gateup_iq2_d2r_pair_kernel<kNTile><<<grid, block, 0, stream>>>(
            W_soa, W_soa, (const block_mx_act_mmq *)act, ids_dst, expert_bounds, work, n_items,
            out, out, M, K, (int)ne_get_rows, n_experts);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: main kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

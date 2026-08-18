// SPDX-License-Identifier: MIT
// ds4_mmq_d2r.cu - gated D2R IQ2_XXS MoE GEMM production path.

#include "ds4_mmq_d2r.cuh"

#include <type_traits>

#include "ds4_cuda_env.cuh"
#include "ds4_act_block.cuh"
#include "mma.cuh"          /* ggml_cuda_mma::tile / load_ldmatrix */
#include "ds4_mxfp8_mma.cuh"
#include "cuda/pulsar_cuda_mx.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

/* The MoE input staging now follows this selector (ds4_mmq.cu picks
 * ds4_quantize_mmq_e4m3_cuda when the arm is 1), so the arm is usable.  The
 * macro stays as the interlock: if the staging is ever decoupled from the
 * selector again, set it back to 0 so the arm REFUSES rather than reading int8
 * bytes as e4m3 -- which would be silently wrong, since every shape still
 * fits. */



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
constexpr int kStages     = 2;
constexpr int kNFrag      = kNTile / 8;
constexpr int kRawStages  = 2;  // k256 raw slots; NT=64 stays under 48 KiB.
constexpr int kIQ2RawRowsPerWarp = 16;
constexpr int kIQ2RawPairsPerRow = 8;
constexpr int kIQ2RawQCodeChunks = kIQ2RawRowsPerWarp * kIQ2RawPairsPerRow;
constexpr int kIQ2RawQCodeTrips = (kIQ2RawQCodeChunks + 31) / 32;
constexpr int kQ8PrefetchItems = kNFrag * 8 * 9;
constexpr int kQ8PrefetchTrips = (kQ8PrefetchItems + kThreads - 1) / kThreads;

static_assert(kNTile == 64, "D2R production path is CFG1 NT64 only");

static_assert(kStages == 2, "D2R raw-ring schedule expects exactly two q8 stages");
static_assert(kThreads == 256, "D2R CTA is fixed at 256 threads");
static_assert(kQ8PrefetchTrips == 3, "unexpected q8 issue trip count");
static_assert(kIQ2RawQCodeTrips == 4, "unexpected IQ2 raw-ring issue trip count");

struct alignas(16) SmemInvariants {
    const char *w_base;
    const half *iq2_dq_base;
    const uint2 *iq2_qs_base;
    const char *q8_tile_base;
    float *out;
    uint32_t sc_off_bytes;
    uint32_t qs_off_bytes;
    uint32_t q8_k128_stride_bytes;
    int nb;
    int k128_iters;
    int M;
    int cta_row0;
    int col_lo;
    int col_count;
    union {
        uint32_t warp_pair0_blk[kWarps];
        uint32_t warp_row0_blk[kWarps];
    };
};

static_assert(sizeof(SmemInvariants) <= 128, "shared invariant table must stay small");












__device__ __forceinline__ void zero_16B(void *dst) {
    int4 z = make_int4(0, 0, 0, 0);
    *reinterpret_cast<int4 *>(dst) = z;
}

__device__ __forceinline__ void zero_8B(void *dst) {
    uint2 z = make_uint2(0, 0);
    *reinterpret_cast<uint2 *>(dst) = z;
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

__device__ __forceinline__ void cp_async_8B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
                     :: "r"(smem), "l"(src));
    } else {
        zero_8B(dst);
    }
#else
    if (pred) {
        *reinterpret_cast<uint2 *>(dst) = *reinterpret_cast<const uint2 *>(src);
    } else {
        zero_8B(dst);
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

__device__ __forceinline__ int d2r_q8_stage(int k128_iter) {
    return k128_iter & (kStages - 1);
}

__device__ __forceinline__ int d2r_raw_stage(int k256_iter) {
    return k256_iter & (kRawStages - 1);
}

template <bool FullTile>
__device__ __forceinline__ void issue_q8_prefetch_one(
        block_mx_act_mmq (&s_q8)[kStages][kNFrag][8],
        const char * __restrict__ q8_iter_base,
        int col_count, int stage, int t) {
    constexpr int cols = kNFrag * 8;
    static_assert(cols == 64, "NT64 q8 prefetch mapping expects 64 columns");
    const int col_local = t & (cols - 1);
    const int chunk = t >> 6;
    const int nf = col_local >> 3;
    const int c = col_local & 7;
    const bool valid = FullTile ? true : (col_local < col_count);
    void *dst = (char *)&s_q8[stage][nf][c] + chunk * 16;
    const void *src = q8_iter_base + (uint64_t)col_local * sizeof(block_mx_act_mmq) + chunk * 16;
    cp_async_16B(dst, src, valid);
}

template <bool FullTile, int Iter>
__device__ __forceinline__ void issue_q8_prefetch_unrolled(
        block_mx_act_mmq (&s_q8)[kStages][kNFrag][8],
        const char * __restrict__ q8_iter_base,
        int col_count, int stage, int tid) {
    if constexpr (Iter < kQ8PrefetchTrips) {
        const int t = tid + Iter * kThreads;
        if constexpr ((Iter + 1) * kThreads <= kQ8PrefetchItems) {
            issue_q8_prefetch_one<FullTile>(s_q8, q8_iter_base, col_count, stage, t);
        } else {
            if (t < kQ8PrefetchItems) {
                issue_q8_prefetch_one<FullTile>(s_q8, q8_iter_base, col_count, stage, t);
            }
        }
        issue_q8_prefetch_unrolled<FullTile, Iter + 1>(
            s_q8, q8_iter_base, col_count, stage, tid);
    }
}

template <bool FullTile>
__device__ __forceinline__ void issue_q8_prefetch(
        block_mx_act_mmq (&s_q8)[kStages][kNFrag][8],
        const volatile SmemInvariants &s_inv,
        int stage, int k128_iter, int tid) {
    const char *q8_tile_base = s_inv.q8_tile_base;
    const uint32_t k128_stride = s_inv.q8_k128_stride_bytes;
    int col_count = kNTile;
    if constexpr (!FullTile) {
        col_count = s_inv.col_count;
    }
    const char *q8_iter_base = q8_tile_base + (uint64_t)k128_iter * (uint64_t)k128_stride;
    issue_q8_prefetch_unrolled<FullTile, 0>(s_q8, q8_iter_base, col_count, stage, tid);
    cp_async_commit();
}

__device__ __forceinline__ void issue_q8_prefetch_one_fast(
        block_mx_act_mmq (&s_q8)[kStages][kNFrag][8],
        const char * __restrict__ q8_iter_base,
        int stage, int t) {
    constexpr int cols = kNFrag * 8;
    static_assert(cols == 64, "NT64 q8 prefetch mapping expects 64 columns");
    const int col_local = t & (cols - 1);
    const int c = col_local & 7;
    const int nf = col_local >> 3;
    const int chunk = t >> 6;
    void *dst = (char *)&s_q8[stage][nf][c] + chunk * 16;
    const void *src = q8_iter_base + (uint64_t)col_local * sizeof(block_mx_act_mmq) + chunk * 16;
    cp_async_16B(dst, src, true);
}

template <int Iter>
__device__ __forceinline__ void issue_q8_prefetch_fast_unrolled(
        block_mx_act_mmq (&s_q8)[kStages][kNFrag][8],
        const char * __restrict__ q8_iter_base,
        int stage, int tid) {
    if constexpr (Iter < kQ8PrefetchTrips) {
        const int t = tid + Iter * kThreads;
        if constexpr ((Iter + 1) * kThreads <= kQ8PrefetchItems) {
            issue_q8_prefetch_one_fast(s_q8, q8_iter_base, stage, t);
        } else {
            if (t < kQ8PrefetchItems) {
                issue_q8_prefetch_one_fast(s_q8, q8_iter_base, stage, t);
            }
        }
        issue_q8_prefetch_fast_unrolled<Iter + 1>(s_q8, q8_iter_base, stage, tid);
    }
}

__device__ __forceinline__ void issue_q8_prefetch_fast(
        block_mx_act_mmq (&s_q8)[kStages][kNFrag][8],
        const volatile SmemInvariants &s_inv,
        int stage, int k128_iter) {
    const char *q8_iter_base =
        s_inv.q8_tile_base + (uint64_t)k128_iter * (uint64_t)s_inv.q8_k128_stride_bytes;
    issue_q8_prefetch_fast_unrolled<0>(s_q8, q8_iter_base, stage, d2r_tid());
    cp_async_commit();
}













struct alignas(16) IQ2RawWarpStage {
    uint2 qs[kIQ2RawRowsPerWarp][kIQ2RawPairsPerRow];
    half dq[kIQ2RawRowsPerWarp];
};

static_assert(sizeof(IQ2RawWarpStage) ==
              kIQ2RawRowsPerWarp * kIQ2RawPairsPerRow * sizeof(uint2) +
              kIQ2RawRowsPerWarp * sizeof(half),
              "unexpected IQ2 raw ring stage size");


constexpr size_t kSmemQ8StageBytes = (size_t)kNFrag * 8 * sizeof(block_mx_act_mmq);
constexpr size_t kSmemInvBytes = sizeof(SmemInvariants);
constexpr size_t kSmemIQ2RawBytes = (size_t)kWarps * kRawStages * sizeof(IQ2RawWarpStage);
constexpr size_t kSmemIQ2GridBytes = 256u * sizeof(uint2);
constexpr size_t kSmemIQ2StaticBytes = (size_t)kStages * kSmemQ8StageBytes +
                                       kSmemIQ2RawBytes + kSmemIQ2GridBytes + kSmemInvBytes;
static_assert(kSmemIQ2StaticBytes <= 48ull * 1024ull,
              "IQ2 D2R static shared memory exceeds 48 KiB");















template <int NFrag, typename TileB>
__device__ __forceinline__ void load_B_tile(
        TileB &B,
        const block_mx_act_mmq (&s_q8)[kStages][NFrag][8],
        int stage, int nf, int k_in_q8) {
    const int *base = reinterpret_cast<const int *>(&s_q8[stage][nf][0].qs[k_in_q8]);
    ggml_cuda_mma::load_ldmatrix(B, base, sizeof(block_mx_act_mmq) / sizeof(int));
}




__device__ __forceinline__ bool iq2_raw_row_valid(int warp_row0, int row, int M) {
    return warp_row0 + row < M;
}

template <bool FullTile, int Iter>
__device__ __forceinline__ void issue_iq2_raw_codes_iter(
        IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ qs_base,
        uint32_t warp_row0_blk, int nb, int raw_stage, int k256_iter,
        int warp_row0, int M, int warp, int lane) {
    if constexpr (Iter < kIQ2RawQCodeTrips) {
        const int t = lane + Iter * 32;
        if (t < kIQ2RawQCodeChunks) {
            const int row = t >> 3;
            const int pair = t & 7;
            const bool row_valid = FullTile ? true : iq2_raw_row_valid(warp_row0, row, M);
            const bool valid = k256_iter < nb && row_valid;
            const uint64_t blk = valid ? ((uint64_t)warp_row0_blk + (uint64_t)row * (uint64_t)nb +
                                           (uint64_t)k256_iter) : 0ull;
            void *dst = &s_raw[warp][raw_stage].qs[row][pair];
            const void *src = qs_base + blk * 8ull + (uint64_t)pair;
            cp_async_8B(dst, src, valid);
        }
        issue_iq2_raw_codes_iter<FullTile, Iter + 1>(
            s_raw, qs_base, warp_row0_blk, nb, raw_stage, k256_iter,
            warp_row0, M, warp, lane);
    }
}

template <bool FullTile>
__device__ __forceinline__ void issue_iq2_raw_prefetch(
        IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const volatile SmemInvariants &s_inv,
        int raw_stage, int k256_iter, int warp, int lane, half dq_val) {
    const uint2 *qs_base = s_inv.iq2_qs_base;
    const uint32_t warp_row0_blk = s_inv.warp_row0_blk[warp];
    const int nb = s_inv.nb;
    int warp_row0 = 0;
    int M = 0;
    if constexpr (!FullTile) {
        warp_row0 = s_inv.cta_row0 + (warp << 4);
        M = s_inv.M;
    }
    issue_iq2_raw_codes_iter<FullTile, 0>(
        s_raw, qs_base, warp_row0_blk, nb, raw_stage, k256_iter,
        warp_row0, M, warp, lane);
    if (lane < kIQ2RawRowsPerWarp) {
        s_raw[warp][raw_stage].dq[lane] = dq_val;
    }
    cp_async_commit();
}

/* The per-block dq halves cannot ride the cp.async ring (2-byte elements,
 * nb-strided rows), so they go global->register->smem.  Issuing the LDG here
 * and passing the value into issue_iq2_raw_prefetch* one k128 iteration later
 * hides the load latency behind a fold; fused LDG.U16->STS.U16 was 52% of the
 * kernel's long-scoreboard stalls (cmd2rncu15b PC sampling). */
template <bool FullTile>
__device__ __forceinline__ half iq2_raw_dq_preload(
        const volatile SmemInvariants &s_inv, int k256_iter) {
    const int lane = d2r_lane();
    half dq = __float2half(0.0f);
    if (lane < kIQ2RawRowsPerWarp) {
        bool valid = k256_iter < s_inv.nb;
        if constexpr (!FullTile) {
            const int warp_row0 = s_inv.cta_row0 + (d2r_warp() << 4);
            valid = valid && iq2_raw_row_valid(warp_row0, lane, s_inv.M);
        }
        if (valid) {
            const uint64_t blk = (uint64_t)s_inv.warp_row0_blk[d2r_warp()] +
                                 (uint64_t)lane * (uint64_t)s_inv.nb + (uint64_t)k256_iter;
            dq = s_inv.iq2_dq_base[blk];
        }
    }
    return dq;
}

template <int Iter>
__device__ __forceinline__ void issue_iq2_raw_codes_iter_fast(
        IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ qs_base,
        uint32_t warp_row0_blk, int nb, int raw_stage, int k256_iter) {
    if constexpr (Iter < kIQ2RawQCodeTrips) {
        const int lane = d2r_lane();
        const int t = lane + Iter * 32;
        if (t < kIQ2RawQCodeChunks) {
            const int row = t >> 3;
            const int pair = t & 7;
            const bool valid = k256_iter < nb;
            const uint64_t blk = valid ? ((uint64_t)warp_row0_blk + (uint64_t)row * (uint64_t)nb +
                                           (uint64_t)k256_iter) : 0ull;
            void *dst = &s_raw[d2r_warp()][raw_stage].qs[row][pair];
            const void *src = qs_base + blk * 8ull + (uint64_t)pair;
            cp_async_8B(dst, src, valid);
        }
        issue_iq2_raw_codes_iter_fast<Iter + 1>(
            s_raw, qs_base, warp_row0_blk, nb, raw_stage, k256_iter);
    }
}

__device__ __forceinline__ void issue_iq2_raw_prefetch_fast(
        IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const volatile SmemInvariants &s_inv,
        int raw_stage, int k256_iter, half dq_val) {
    const int warp = d2r_warp();
    const int lane = d2r_lane();
    const uint32_t warp_row0_blk = s_inv.warp_row0_blk[warp];
    issue_iq2_raw_codes_iter_fast<0>(
        s_raw, s_inv.iq2_qs_base, warp_row0_blk, s_inv.nb, raw_stage, k256_iter);
    if (lane < kIQ2RawRowsPerWarp) {
        s_raw[warp][raw_stage].dq[lane] = dq_val;
    }
    cp_async_commit();
}

__device__ __forceinline__ uint32_t iq2_decode_signed_half(
        uint2 code, const uint2 * __restrict__ s_grid, int chunk) {
    const int group = chunk >> 1;
    const int hi = chunk & 1;
    const uint8_t aux = (uint8_t)(code.x >> (8 * group));
    const uint2 grid_pos = s_grid[aux];
    const uint32_t signs8 = ds4_unpack_ksigns((uint8_t)(code.y >> (7 * group)));
    const uint32_t sel = hi ? 0x80402010u : 0x08040201u;
    const uint32_t s = __vcmpne4(signs8 & sel, 0);
    const uint32_t grid_half = hi ? grid_pos.y : grid_pos.x;
    return __vsub4(grid_half ^ s, s);
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
 * (ds4_quantize_e4m3.cu), and make_iq2_A_tile_e4m3 below calls this four times
 * per pair.  Left as a note because the stale version was read as evidence that
 * the routed path still ran q8_1 activations -- see ledger L065. */
__device__ __forceinline__ static uint32_t iq2_pack_e4m3_quad(
        uint32_t signed4, float scale) {
    uint32_t out = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const float v = (float)(int8_t)((signed4 >> (8 * b)) & 0xFFu) * scale;
        const __nv_fp8_e4m3 e = (__nv_fp8_e4m3)v;
        out |= ((uint32_t)*(const uint8_t *)&e) << (8 * b);
    }
    return out;
}

template <int T, typename TileA>
__device__ __forceinline__ void make_iq2_A_tile_e4m3(
        TileA &A, uint32_t &sfa,
        const IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ s_grid,
        int warp, int raw_stage, bool row0_ok, bool row1_ok,
        int group, int tig) {
    const IQ2RawWarpStage &raw = s_raw[warp][raw_stage];
    constexpr int pair = T;
    const int row0 = group;
    const int row1 = group + 8;
    const uint2 code0 = row0_ok ? raw.qs[row0][pair] : make_uint2(0, 0);
    const uint2 code1 = row1_ok ? raw.qs[row1][pair] : make_uint2(0, 0);

    const uint32_t w0a = row0_ok ? iq2_decode_signed_half(code0, s_grid, tig)     : 0u;
    const uint32_t w1a = row1_ok ? iq2_decode_signed_half(code1, s_grid, tig)     : 0u;
    const uint32_t w0b = row0_ok ? iq2_decode_signed_half(code0, s_grid, tig + 4) : 0u;
    const uint32_t w1b = row1_ok ? iq2_decode_signed_half(code1, s_grid, tig + 4) : 0u;

    const float d0 = row0_ok ? __half2float(raw.dq[row0]) : 0.0f;
    const float d1 = row1_ok ? __half2float(raw.dq[row1]) : 0.0f;
    const int ls0 = (int)(code0.y >> 27) | 1;
    const int ls1 = (int)(code1.y >> 27) | 1;
    const float dA0 = d0 * (float)ls0 * 0.125f;
    const float dA1 = d1 * (float)ls1 * 0.125f;

    float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        a0 = fmaxf(a0, fabsf((float)(int8_t)((w0a >> (8 * b)) & 0xFFu)));
        a0 = fmaxf(a0, fabsf((float)(int8_t)((w0b >> (8 * b)) & 0xFFu)));
        a1 = fmaxf(a1, fabsf((float)(int8_t)((w1a >> (8 * b)) & 0xFFu)));
        a1 = fmaxf(a1, fabsf((float)(int8_t)((w1b >> (8 * b)) & 0xFFu)));
    }
    a0 *= fabsf(dA0);
    a1 *= fabsf(dA1);
    /* quad reduction: lanes group*4 + 0..3 hold this row's whole 32-block */
    a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, 1));
    a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, 2));
    a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, 1));
    a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, 2));

    const int se0 = pulsar_mx_shared_exp(a0);
    const int se1 = pulsar_mx_shared_exp(a1);
    const float r0 = dA0 * exp2f(-(float)se0);
    const float r1 = dA1 * exp2f(-(float)se1);

    A.x[0] = (int)iq2_pack_e4m3_quad(w0a, r0);
    A.x[1] = (int)iq2_pack_e4m3_quad(w1a, r1);
    A.x[2] = (int)iq2_pack_e4m3_quad(w0b, r0);
    A.x[3] = (int)iq2_pack_e4m3_quad(w1b, r1);

    sfa = (uint32_t)pulsar_mx_scale_byte((tig == 1) ? se1 : se0);
}




/* MXFP8 arm of the k32 pair: same decomposition as mma_fold_iq2_k32_pair_t, but
 * both operands are E4M3 and the HARDWARE applies the ue8m0 scales, so there is
 * NO fold -- the accumulator is f32 and the MMA reads/writes it in place.
 *
 * WHERE THE SCALES COME FROM, which is the easy thing to get wrong:
 *   sfa -- from make_iq2_A_tile_e4m3, keyed on `tig` (tig == 1 supplies row
 *          group+8, every other lane supplies row group).
 *   sfb -- from the COLUMN GROUP, i.e. s_q8[stage][nf][group], NOT from the
 *          accumulator columns c0/c1.  For
 *          m16n8k32 the B fragment's 8 columns map one per 4-lane group, so
 *          lane group g carries column g's scale.  Using the c0/c1 pair here
 *          would hand the MMA a well-formed scale for the WRONG column.
 * The ue8m0 byte is stored as a float by ds4_quantize_mmq_e4m3 (which reuses
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
        const IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ s_grid,
        const block_mx_act_mmq (&s_q8)[kStages][NFrag][8],
        int raw_stage, int q8_stage, bool raw_row0_ok, bool raw_row1_ok,
        int warp, int group, int tig, const volatile SmemInvariants &s_inv) {
    static_assert(T1 == T0 + 1, "expected adjacent k32 pair");
    static_assert(TileC::ne == 4, "expected m16n8 accumulator fragment");
    TileA A0;
    TileA A1;
    uint32_t sfa0 = 0;
    uint32_t sfa1 = 0;
    make_iq2_A_tile_e4m3<T0>(A0, sfa0, s_raw, s_grid, warp, raw_stage,
                             raw_row0_ok, raw_row1_ok, group, tig);
    make_iq2_A_tile_e4m3<T1>(A1, sfa1, s_raw, s_grid, warp, raw_stage,
                             raw_row0_ok, raw_row1_ok, group, tig);

    constexpr int k_in_q8_0 = (T0 & 3) * 32;
    constexpr int k_in_q8_1 = (T1 & 3) * 32;

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
        load_B_tile(B0, s_q8, q8_stage, nf, k_in_q8_0);
        load_B_tile(B1, s_q8, q8_stage, nf, k_in_q8_1);
        const uint32_t sfb0 = (uint32_t)s_q8[q8_stage][nf][group].d4[T0 & 3];
        const uint32_t sfb1 = (uint32_t)s_q8[q8_stage][nf][group].d4[T1 & 3];

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
        const IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ s_grid,
        const block_mx_act_mmq (&s_q8)[kStages][NFrag][8],
        int k128_iter, const volatile SmemInvariants &s_inv) {
    const int warp = d2r_warp();
    const int group = d2r_group();
    int warp_row0 = 0;
    if constexpr (!FullTile) {
        warp_row0 = s_inv.cta_row0 + (warp << 4);
    }
    bool row0_ok = true;
    bool row1_ok = true;
    if constexpr (!FullTile) {
        row0_ok = (warp_row0 + group) < s_inv.M;
        row1_ok = (warp_row0 + group + 8) < s_inv.M;
    }
    const int tig = d2r_tig();
    const int raw_stage = d2r_raw_stage(k128_iter >> 1);
    const int q8_stage = d2r_q8_stage(k128_iter);
    const int half_pair_base = (k128_iter & 1) ? 4 : 0;

    if (half_pair_base == 0) {
        mma_iq2_k32_pair_e4m3<FullTile, TileA, TileB, TileC, 0, 1>(
            acc, s_raw, s_grid, s_q8, raw_stage, q8_stage,
            row0_ok, row1_ok, warp, group, tig, s_inv);
        mma_iq2_k32_pair_e4m3<FullTile, TileA, TileB, TileC, 2, 3>(
            acc, s_raw, s_grid, s_q8, raw_stage, q8_stage,
            row0_ok, row1_ok, warp, group, tig, s_inv);
    } else {
        mma_iq2_k32_pair_e4m3<FullTile, TileA, TileB, TileC, 4, 5>(
            acc, s_raw, s_grid, s_q8, raw_stage, q8_stage,
            row0_ok, row1_ok, warp, group, tig, s_inv);
        mma_iq2_k32_pair_e4m3<FullTile, TileA, TileB, TileC, 6, 7>(
            acc, s_raw, s_grid, s_q8, raw_stage, q8_stage,
            row0_ok, row1_ok, warp, group, tig, s_inv);
    }
}

template <bool FullTile, typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void iq2_d2r_mainloop(
        float (&acc)[kNFrag][TileC::ne],
        block_mx_act_mmq (&s_q8)[kStages][kNFrag][8],
        IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ s_grid,
        const volatile SmemInvariants &s_inv) {
#pragma unroll
    for (int pf = 0; pf < kStages; ++pf) {
        if (pf < s_inv.k128_iters) {
            if constexpr (FullTile) {
                issue_q8_prefetch_fast(s_q8, s_inv, d2r_q8_stage(pf), pf);
            } else {
                issue_q8_prefetch<false>(s_q8, s_inv, d2r_q8_stage(pf), pf, d2r_tid());
            }
        }
    }

#pragma unroll
    for (int pf = 0; pf < kRawStages; ++pf) {
        if (pf < s_inv.nb) {
            const half dq0 = iq2_raw_dq_preload<FullTile>(s_inv, pf);
            if constexpr (FullTile) {
                issue_iq2_raw_prefetch_fast(s_raw, s_inv, d2r_raw_stage(pf), pf, dq0);
            } else {
                issue_iq2_raw_prefetch<false>(
                    s_raw, s_inv, d2r_raw_stage(pf), pf, d2r_warp(), d2r_lane(), dq0);
            }
        }
    }

    half dq_pend = __float2half(0.0f);
    for (int k128_iter = 0;; ++k128_iter) {
        if (k128_iter >= s_inv.k128_iters) {
            break;
        }
        if ((k128_iter & 1) == 0) {
            int keep_raw = 0;
            if constexpr (kRawStages > 1) {
                keep_raw = ((k128_iter >> 1) + 1 < s_inv.nb) ? 1 : 0;
            }
            cp_async_wait_keep(keep_raw);
        }
        __syncthreads();
        if ((k128_iter & 1) == 0) {
            /* dq LDG for the raw prefetch issued at the NEXT (odd) iteration:
             * a full fold of distance between the load and its smem store. */
            const int raw_pf = (k128_iter >> 1) + kRawStages;
            if (raw_pf < s_inv.nb) {
                dq_pend = iq2_raw_dq_preload<FullTile>(s_inv, raw_pf);
            }
        }

        if constexpr (FullTile) {
            mma_fold_iq2_k128<true, TileA, TileB, TileC>(
                acc, s_raw, s_grid, s_q8, k128_iter, s_inv);
        } else {
            mma_fold_iq2_k128<false, TileA, TileB, TileC>(
                acc, s_raw, s_grid, s_q8, k128_iter, s_inv);
        }

        __syncthreads();
        const int pf_iter = k128_iter + kStages;
        if (pf_iter < s_inv.k128_iters) {
            if constexpr (FullTile) {
                issue_q8_prefetch_fast(s_q8, s_inv, d2r_q8_stage(pf_iter), pf_iter);
            } else {
                issue_q8_prefetch<false>(s_q8, s_inv, d2r_q8_stage(pf_iter), pf_iter, d2r_tid());
            }
        }
        if ((k128_iter & 1) != 0) {
            const int raw_pf = (k128_iter >> 1) + kRawStages;
            if (raw_pf < s_inv.nb) {
                if constexpr (FullTile) {
                    issue_iq2_raw_prefetch_fast(s_raw, s_inv, d2r_raw_stage(raw_pf), raw_pf, dq_pend);
                } else {
                    issue_iq2_raw_prefetch<false>(
                        s_raw, s_inv, d2r_raw_stage(raw_pf), raw_pf, d2r_warp(), d2r_lane(), dq_pend);
                }
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
__global__ __launch_bounds__(kThreads, 2)
void gateup_iq2_d2r_pair_kernel(const void * __restrict__ gate_soa,
                                const void * __restrict__ up_soa,
                                const block_mx_act_mmq * __restrict__ q8,
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

    const int col_lo = expert_bounds[expert] + jt * kNTile;
    const int col_hi_full = expert_bounds[expert + 1];
    const int col_tile_hi = (col_hi_full < col_lo + kNTile) ? col_hi_full : (col_lo + kNTile);
    if (col_lo >= col_tile_hi) {
        return;
    }

    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile<8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, int>;

    __shared__ __align__(16) block_mx_act_mmq s_q8[kStages][kNFrag][8];
    __shared__ __align__(16) IQ2RawWarpStage s_raw[kWarps][kRawStages];
    __shared__ __align__(16) uint2 s_grid[256];
    __shared__ __align__(16) volatile SmemInvariants s_inv;
    /* Scatter-index staging: one output column per tile lane. */
    __shared__ int s_out_cols[kNTile];

    const void *W_soa = leg == 0 ? gate_soa : up_soa;
    float *out = leg == 0 ? out_gate : out_up;
    const int cta_row0 = (int)blockIdx.x * kMTile;
    const int warp_row0 = cta_row0 + d2r_warp() * 16;
    const int nb = K >> 8;

    const bool full_warp_tile = (warp_row0 + 15 < M) && (col_lo + kNTile <= col_hi_full);

    for (int i = d2r_tid(); i < 256; i += kThreads) {
        s_grid[i] = reinterpret_cast<const uint2 *>(iq2xxs_grid)[i];
    }
    if (d2r_tid() == 0) {
        const uint64_t nblk = (uint64_t)E * (uint64_t)M * (uint64_t)nb;
        const uint64_t dq_bytes = (nblk * 2ull + 63ull) & ~63ull;
        s_inv.w_base = (const char *)W_soa;
        s_inv.iq2_dq_base = reinterpret_cast<const half *>(W_soa);
        s_inv.iq2_qs_base =
            reinterpret_cast<const uint2 *>(reinterpret_cast<const char *>(W_soa) + dq_bytes);
        s_inv.q8_tile_base = reinterpret_cast<const char *>(q8) + (uint64_t)col_lo * sizeof(block_mx_act_mmq);
        s_inv.out = out;
        s_inv.sc_off_bytes = 0;
        s_inv.qs_off_bytes = 0;
        s_inv.q8_k128_stride_bytes = (uint32_t)((uint64_t)n_assign * sizeof(block_mx_act_mmq));
        s_inv.nb = nb;
        s_inv.k128_iters = K >> 7;
        s_inv.M = M;
        s_inv.cta_row0 = cta_row0;
        s_inv.col_lo = col_lo;
        s_inv.col_count = col_tile_hi - col_lo;
    }
    if (d2r_lane() == 0) {
        const uint64_t expert_row = (uint64_t)expert * (uint64_t)M;
        const uint64_t warp_row0_base = (expert_row + (uint64_t)warp_row0) * (uint64_t)nb;
        s_inv.warp_row0_blk[d2r_warp()] = (uint32_t)warp_row0_base;
    }
    if (d2r_tid() < col_tile_hi - col_lo) {
        s_out_cols[d2r_tid()] = ids_dst[col_lo + d2r_tid()];
    }
    __syncthreads();

    float acc[kNFrag][tile_C::ne] = {};

    if (full_warp_tile) {
        iq2_d2r_mainloop<true, tile_A, tile_B, tile_C>(
            acc, s_q8, s_raw, s_grid, s_inv);
    } else {
        iq2_d2r_mainloop<false, tile_A, tile_B, tile_C>(
            acc, s_q8, s_raw, s_grid, s_inv);
    }

    const int out_col_lo = s_inv.col_lo;
    const int out_col_hi = out_col_lo + s_inv.col_count;
    const int out_warp_row0 = s_inv.cta_row0 + (d2r_warp() << 4);
    const int out_M = s_inv.M;
    float *out_base = s_inv.out;
#pragma unroll
    for (int nf = 0; nf < kNFrag; ++nf) {
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
    (void)q8;
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

static int64_t d2r_work_capacity(int64_t ncols_max, int n_experts) {
    return d2r_work_capacity_for_tile(ncols_max, n_experts, kNTile);
}


} // namespace


bool ds4_mmq_iq2_xxs_moe_d2r_available(int cc) {
    static int cached_cc = -1;
    static int cached = 0;
    if (cached_cc != cc) {
        cached_cc = cc;
        cached = (GGML_CUDA_CC_IS_NVIDIA(cc) &&
                  ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE) ? 1 : 0;
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



int ds4_mmq_iq2_xxs_moe_d2r_pair_launch(const void *gate_soa,
                                         const void *up_soa,
                                         int64_t soa_blocks,
                                         const void *q8,
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
    if (!gate_soa || !up_soa || !q8 || !ids_dst || !expert_bounds || !out_gate || !out_up ||
        !worklist_scratch || M <= 0 || K <= 0 || K % 256 != 0 || ne_get_rows <= 0 ||
        ne_get_rows > INT_MAX || n_experts <= 0) {
        return -1;
    }

    const int64_t expected_soa_blocks =
        (int64_t)n_experts * (int64_t)M * (int64_t)(K >> 8);
    if (soa_blocks < expected_soa_blocks) {
        return -1;
    }

    const int64_t capacity64 = d2r_work_capacity(ne_get_rows, n_experts);
    if (capacity64 <= 0 || capacity64 > (int64_t)(INT_MAX - 1)) {
        return -1;
    }
    const size_t needed = (size_t)capacity64 * sizeof(int) + sizeof(int);
    if (worklist_scratch_bytes < needed) {
        return -1;
    }

    int *work = (int *)worklist_scratch;
    int *n_items = work + capacity64;

    d2r_build_worklist_kernel<kNTile><<<1, kThreads, 0, stream>>>(
        expert_bounds, work, n_items, n_experts);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: worklist builder launch failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    /* Same expert-major schedule as the down launch (see comment there). */
    const dim3 grid((unsigned)((M + kMTile - 1) / kMTile), (unsigned)capacity64, 2);
    const dim3 block(32, kWarps, 1);
    gateup_iq2_d2r_pair_kernel<<<grid, block, 0, stream>>>(
        gate_soa, up_soa, (const block_mx_act_mmq *)q8, ids_dst, expert_bounds, work, n_items,
        out_gate, out_up, M, K, (int)ne_get_rows, n_experts);
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
                                          const void *q8,
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
    if (!W_soa || !q8 || !ids_dst || !expert_bounds || !out) {
        return -1;
    }
    if (M <= 0 || K <= 0 || K % 256 != 0 || n_experts <= 0 || ne_get_rows <= 0) {
        return -1;
    }

    const int64_t expected_soa_blocks =
        (int64_t)n_experts * (int64_t)M * (int64_t)(K >> 8);
    if (soa_blocks < expected_soa_blocks) {
        return -1;
    }
    const int64_t capacity64 = d2r_work_capacity(ne_get_rows, n_experts);
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

    d2r_build_worklist_kernel<kNTile><<<1, kThreads, 0, stream>>>(
        expert_bounds, work, n_items, n_experts);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: worklist builder launch failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    /* z = 1: leg is pinned to 0, so only W_soa / out are ever touched. */
    const dim3 grid((unsigned)((M + kMTile - 1) / kMTile), (unsigned)capacity64, 1);
    const dim3 block(32, kWarps, 1);
    gateup_iq2_d2r_pair_kernel<<<grid, block, 0, stream>>>(
        W_soa, W_soa, (const block_mx_act_mmq *)q8, ids_dst, expert_bounds, work, n_items,
        /* Callers MUST stage E4M3 -- there is no int8 arm left to fall back
         * to.  This launch once read ds4_d2r_iq2_arm() itself, which ran the
         * E4M3 MMA against q8_1 bytes whenever the env was set (the arm-1
         * garbage).  With one format there is nothing left to disagree about. */
        out, out, M, K, (int)ne_get_rows, n_experts);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: main kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

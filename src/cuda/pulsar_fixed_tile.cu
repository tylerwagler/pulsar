// pulsar_fixed_tile.cu -- L151-C stage 1: fixed-tile tensor-core GEMMs for the
// M-neutral dense step (see pulsar_fixed_tile.h for the contract and the
// stage-0 pricing).  CUTLASS unit: compiled like pulsar_mxfp4_cutlass.cu.
//
// WHY A FIXED TILE IS NEUTRAL.  Each output element is a dot product over K
// accumulated in f32 by one thread's MMA fragments, in a K order fixed by the
// tile's K loop; the tile scheduler walks N; M only decides how many rows are
// real (the rest of the 128-row A tile is zero-filled by TMA/predication).  No
// split-K, no M-dependent algorithm choice.  So row r of an M-row call is the
// same bytes as row r of any other call that contains it -- the stage-0 probe
// verified this for every M in 1..16, and it is what gates 4/5 of the mixed
// neutrality gate assert in the engine.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "pulsar_gpu.h"
#include "pulsar_cuda_mx.cuh"
#include "pulsar_fixed_tile.h"
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

using namespace cute;

/* ---- MXFP8 x MXFP8, block-scaled (sm120 kind::f8f6f4), 128 x TN x 128 ---- */
using MxA   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using MxB   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using MxSF  = cutlass::float_ue8m0_t;

template <int TN, class OutT>
struct FT {
    using TileShape    = Shape<_128, Int<TN>, _128>;   // 128 rows: the SF atom is 128x4
    using ClusterShape = Shape<_1, _1, _1>;
    static constexpr int AlignOut = 128 / cutlass::sizeof_bits<OutT>::value;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
        TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
        float, float, OutT, cutlass::layout::RowMajor, AlignOut, OutT, cutlass::layout::RowMajor, AlignOut,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
        MxA, cutlass::layout::RowMajor, 32, MxB, cutlass::layout::ColumnMajor, 32, float,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using BlkCfg     = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    static typename Gemm::Arguments args(OutT *D, const uint8_t *A, const MxSF *Asf,
                                         const uint8_t *B, const MxSF *Bsf, int M, int N, int K) {
        auto sA = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, K, 1});
        auto sB = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, K, 1});
        auto sC = cutlass::make_cute_packed_stride(typename GemmKernel::StrideC{}, {M, N, 1});
        auto sD = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M, N, 1});
        auto lSFA = BlkCfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
        auto lSFB = BlkCfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
        return typename Gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
            { reinterpret_cast<const typename MxA::DataType *>(A), sA,
              reinterpret_cast<const typename MxB::DataType *>(B), sB,
              Asf, lSFA, Bsf, lSFB },
            { {1.0f, 0.0f}, D, sC, D, sD } };
    }

    static int run(OutT *D, const uint8_t *A, const MxSF *Asf, const uint8_t *B, const MxSF *Bsf,
                   int M, int N, int K, void *ws, size_t ws_bytes) {
        auto a = args(D, A, Asf, B, Bsf, M, N, K);
        Gemm gemm;
        const size_t need = Gemm::get_workspace_size(a);
        if (need > ws_bytes) return -1;
        if (gemm.can_implement(a) != cutlass::Status::kSuccess) return 1;
        if (gemm.initialize(a, need ? ws : nullptr) != cutlass::Status::kSuccess) return 2;
        return gemm.run() == cutlass::Status::kSuccess ? 0 : 3;
    }
};

/* The engine's VEC32 scale swizzle (pulsar_mx_sfoff) and CUTLASS's SFA/SFB
 * layouts are the same function -- the stage-0 probe found 0 of 1M weight
 * scale bytes displaced.  This is load-bearing (the kernel reads the A8 slot's
 * scales and the LT slab directly), so it is asserted here once per K, on the
 * host, for both operands, rather than trusted. */
template <int TN>
static int ft_layouts_match(int N, int K) {
    using Cfg = typename FT<TN, float>::BlkCfg;
    const int KB = K / 32, KBp = pulsar_mx_rup(KB, 4);
    auto lA = Cfg::tile_atom_to_shape_SFA(make_shape(128, N, K, 1));
    auto lB = Cfg::tile_atom_to_shape_SFB(make_shape(128, N, K, 1));
    const int rows_a[] = {0, 1, 17, 31, 32, 33, 63, 64, 96, 127};
    const int kbs[]    = {0, 1, 2, 3, 4, 5, 7, 8, KB - 1};
    for (int ri = 0; ri < 10; ri++) for (int ki = 0; ki < 9; ki++) {
        const int kb = kbs[ki]; if (kb < 0 || kb >= KB) continue;
        const int m = rows_a[ri];
        const int want = pulsar_mx_sfoff(m, kb, KBp);
        const int gotA = (int)lA(make_coord(m, kb * 32, 0));
        if (gotA != want) { fprintf(stderr, "pulsar: fixed-tile SFA layout differs from the VEC32 swizzle at (m=%d,kb=%d): cutlass %d vs %d -- refusing the fixed tile for K=%d\n", m, kb, gotA, want, K); return 0; }
        const int n = (m * 977) % N;   // spread rows across the N range too
        const int wantB = pulsar_mx_sfoff(n, kb, KBp);
        const int gotB = (int)lB(make_coord(n, kb * 32, 0));
        if (gotB != wantB) { fprintf(stderr, "pulsar: fixed-tile SFB layout differs from the VEC32 swizzle at (n=%d,kb=%d): cutlass %d vs %d -- refusing the fixed tile for K=%d\n", n, kb, gotB, wantB, K); return 0; }
    }
    return 1;
}

extern "C" int pulsar_ft_mxfp8_shape_ok(uint64_t in_dim, uint64_t out_dim) {
    return in_dim % 128 == 0 && out_dim % 64 == 0 && in_dim * out_dim >= (16ull << 20);
}

extern "C" int pulsar_ft_mxfp8(void *out, int out_f16,
                               const uint8_t *xq, const uint8_t *sx,
                               const uint8_t *wdata, const uint8_t *wscale,
                               int M, int N, int K, void *ws, size_t ws_bytes) {
    if (!out || !xq || !sx || !wdata || !wscale || M < 1 || M > 16) return 0;
    if (!pulsar_ft_mxfp8_shape_ok((uint64_t)K, (uint64_t)N)) return 0;
    const int tn128 = N >= 16384;
    /* one-time layout assertion per (K, N) -- a handful of shapes */
    {
        static uint64_t ok_keys[16]; static int n_ok = 0;
        const uint64_t key = ((uint64_t)(uint32_t)K << 32) | (uint32_t)N;
        int known = 0;
        for (int i = 0; i < n_ok; i++) if (ok_keys[i] == key) { known = 1; break; }
        if (!known) {
            if (!(tn128 ? ft_layouts_match<128>(N, K) : ft_layouts_match<64>(N, K))) return 0;
            if (n_ok < 16) ok_keys[n_ok++] = key;
        }
    }
    const MxSF *Asf = reinterpret_cast<const MxSF *>(sx);
    const MxSF *Bsf = reinterpret_cast<const MxSF *>(wscale);
    int rc;
    if (out_f16) {
        rc = tn128 ? FT<128, cutlass::half_t>::run((cutlass::half_t *)out, xq, Asf, wdata, Bsf, M, N, K, ws, ws_bytes)
                   : FT<64,  cutlass::half_t>::run((cutlass::half_t *)out, xq, Asf, wdata, Bsf, M, N, K, ws, ws_bytes);
    } else {
        rc = tn128 ? FT<128, float>::run((float *)out, xq, Asf, wdata, Bsf, M, N, K, ws, ws_bytes)
                   : FT<64,  float>::run((float *)out, xq, Asf, wdata, Bsf, M, N, K, ws, ws_bytes);
    }
    if (rc != 0) {
        static int said = 0;
        if (!said) { said = 1; fprintf(stderr, "pulsar: fixed-tile MXFP8 GEMM refused/failed (rc=%d, M=%d N=%d K=%d out_f16=%d)\n", rc, M, N, K, out_f16); }
        return 0;
    }
    return 1;
}

/* ---- bf16 x bf16 -> f32, 64 x 128 x 64, 4 stages (sm80-style mma.sync) ----
 * The sm120 TMA builder takes only f8f6f4 elements; this tile measured
 * 198 GB/s on the 1 GB head (cuBLAS 224), flat from 1 to 16 rows. */
using FBGemm = cutlass::gemm::device::GemmUniversal<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4, 8, 8>;

extern "C" int pulsar_ft_bf16_shape_ok(uint64_t in_dim, uint64_t out_dim) {
    return in_dim % 64 == 0 && out_dim % 128 == 0 && in_dim * out_dim * 2ull >= (512ull << 20);
}

extern "C" int pulsar_ft_bf16(float *out, const uint16_t *xb, const uint16_t *w,
                              int M, int N, int K, void *ws, size_t ws_bytes) {
    if (!out || !xb || !w || M < 1 || M > 16) return 0;
    if (!pulsar_ft_bf16_shape_ok((uint64_t)K, (uint64_t)N)) return 0;
    FBGemm::Arguments a(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, {1.0f, 0.0f},
                        (const cutlass::bfloat16_t *)xb, (const cutlass::bfloat16_t *)w, out, out,
                        0, 0, 0, 0, (int64_t)K, (int64_t)K, (int64_t)N, (int64_t)N);
    FBGemm gemm;
    const size_t need = FBGemm::get_workspace_size(a);
    int rc = need > ws_bytes ? -1
           : gemm.can_implement(a) != cutlass::Status::kSuccess ? 1
           : gemm.initialize(a, need ? ws : nullptr) != cutlass::Status::kSuccess ? 2
           : gemm.run() == cutlass::Status::kSuccess ? 0 : 3;
    if (rc != 0) {
        static int said = 0;
        if (!said) { said = 1; fprintf(stderr, "pulsar: fixed-tile bf16 GEMM refused/failed (rc=%d, M=%d N=%d K=%d)\n", rc, M, N, K); }
        return 0;
    }
    return 1;
}

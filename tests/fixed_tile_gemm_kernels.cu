// tests/fixed_tile_gemm_kernels.cu -- L151-C stage 0 / L169: the CUTLASS side
// of the fixed-tile probe.  A block-scaled MXFP8 x MXFP8 GEMM on sm120 with ONE
// tile configuration (128 x 128 x 128) regardless of M, fed the SAME E4M3
// operands the engine's dense path uses: activations quantised with the shared
// pulsar_mx_* helpers (bit-identical bytes to mxfp8_quant_act_kernel), weights
// = the MXFP8_LT slabs straight from the mmap (data reused as CUTLASS K-major
// B; the E8M0 scale byte re-laid into CUTLASS's SFB layout).
//
// L169 ("Try split-K"): the same mainloop and epilogue are instantiated twice --
// once under CUTLASS's default persistent scheduler (the stage-0 kernel, one CTA
// per 128-column output tile, no reduction) and once under its stream-K
// scheduler (cutlass::gemm::StreamKScheduler -> PersistentTileSchedulerSm100StreamK
// on sm120), which the probe drives in two modes:
//   split-K  S in {2,4,8}: DecompositionMode::SplitK, splits = S.  K's tile
//            count is cut into S contiguous slices (k_tiles / S each, the first
//            k_tiles % S slices one tile longer -- set_params_basic), one CTA
//            per (output tile, slice).  ReductionMode::Deterministic: slice 0
//            stores its f32 partial to the workspace, slice s waits on a per-
//            tile barrier until exactly the k-tiles of slices 0..s-1 have
//            arrived, then adds its partial IN PLACE (workspace += partial), and
//            the last slice adds the workspace into its own accumulator and
//            runs the epilogue (sm90_tile_scheduler_stream_k.hpp fixup_helper).
//            The sum is therefore (((p0 + p1) + p2) + ...) + p_{S-1} in a fixed
//            order for every element, every launch: M-independent by
//            construction, and a DIFFERENT number from the unsplit tile's
//            single-accumulator sum (the probe reports that distance).
//   stream-K: DecompositionMode::StreamK, splits = 1.  Units = the persistent
//            grid (sm_count CTAs); each unit walks a fixed, contiguous span of
//            the (tile, k-tile) iteration space, so the peer set and the
//            reduction order per output tile are again a function of (N, K,
//            sm_count) only, not of M (M <= 128 is one row of tiles).
// Both are driven with hw_info.sm_count fixed from the device so the
// decomposition cannot drift with a query; the probe reads the decomposition
// back (ft_plan_decomp) and asserts it is the same at every M.
//
// Plain C interface so the driver (fixed_tile_gemm_probe.cpp) can stay a C++23
// engine client while this unit compiles as the CUTLASS unit does (nvcc, C++17).
//
// PRICING ONLY.  Nothing here is reachable from the engine.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "pulsar_gpu.h"
#include "pulsar_cuda_internal.h"   /* pulsar_gpu_tensor::ptr -- the driver hands tensors, not raw pointers */
#include "pulsar_cuda_mx.cuh"
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "fixed_tile_gemm_probe.h"

using namespace cute;

using ElementA   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementSF  = cutlass::float_ue8m0_t;
using ElementAcc = float;
using LayoutA    = cutlass::layout::RowMajor;     // A[M,K], K contiguous
using LayoutB    = cutlass::layout::ColumnMajor;  // B[N,K] as K-major per n == the LT data slab
using LayoutC    = cutlass::layout::RowMajor;
constexpr int AlignA = 32, AlignB = 32;
constexpr int AlignC = 128 / cutlass::sizeof_bits<float>::value;

using TileShape    = Shape<_128, _128, _128>;   /* 128 rows: the SF atom is 128x4; a 64-row tile cannot describe its TMA load */
using ClusterShape = Shape<_1, _1, _1>;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc, float, LayoutC, AlignC, float, LayoutC, AlignC,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using BlkCfg = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

/* One mainloop + epilogue, two tile schedulers (the 4th GemmUniversal parameter). */
template <class SchedTag>
struct FT {
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, SchedTag>;
    using Gemm       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using Arguments  = typename Gemm::Arguments;

    static Arguments args(float *D, const uint8_t *A_data, const ElementSF *A_sf,
                          const uint8_t *B_data, const ElementSF *B_sf, int M, int N, int K, int sm_count) {
        auto strideA = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, K, 1});
        auto strideB = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, K, 1});
        auto strideC = cutlass::make_cute_packed_stride(typename GemmKernel::StrideC{}, {M, N, 1});
        auto strideD = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M, N, 1});
        auto lSFA = BlkCfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
        auto lSFB = BlkCfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
        Arguments a{
            cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
            { reinterpret_cast<const typename ElementA::DataType *>(A_data), strideA,
              reinterpret_cast<const typename ElementB::DataType *>(B_data), strideB,
              A_sf, lSFA, B_sf, lSFB },
            { {1.0f, 0.0f}, D, strideC, D, strideD } };
        a.hw_info.device_id = 0;
        a.hw_info.sm_count = sm_count;   /* fixed: no per-call device query, no decomposition drift */
        return a;
    }
};
using FT1 = FT<void>;                              /* stage-0 kernel: one CTA per output tile */
using FTK = FT<cutlass::gemm::StreamKScheduler>;   /* split-K / stream-K: adds the fixup reduction */
using SKArgs   = typename FTK::GemmKernel::TileSchedulerArguments;
using SKDecomp = decltype(SKArgs::decomposition_mode);
using SKReduce = decltype(SKArgs::reduction_mode);

/* Variants: index -> (kernel, split count).  splits 1 + StreamK mode is the
 * stream-K variant; splits 1 on FT1 is the plain tile. */
enum { V_S1 = 0, V_S2, V_S4, V_S8, V_SK, V_N };
static const int         ft_splits[V_N] = {1, 2, 4, 8, 1};
static const char *const ft_names[V_N]  = {"128^3 S=1", "S=2", "S=4", "S=8", "streamK"};
extern "C" int ft_nvariants(void) { return V_N; }
extern "C" const char *ft_variant_name(int v) { return v >= 0 && v < V_N ? ft_names[v] : "?"; }

static int ft_check(cudaError_t e, const char *what) {
    if (e == cudaSuccess) return 0;
    fprintf(stderr, "fixed-tile probe: %s: %s\n", what, cudaGetErrorString(e));
    return 1;
}

extern "C" int ft_device_info(int *sm_count, unsigned long long *l2_bytes) {
    int sm = 0, l2 = 0;
    if (ft_check(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0), "sm count") ||
        ft_check(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0), "l2 size")) return 1;
    if (sm_count) *sm_count = sm;
    if (l2_bytes) *l2_bytes = (unsigned long long)l2;
    return 0;
}

/* Activation pack: one warp per (m, kb); identical bytes to the engine's
 * mxfp8_quant_act_kernel (same helpers, fmax is order-free), SF written through
 * the CUTLASS SFA layout object. */
template <class TSFA>
__global__ static void ft_pack_a(uint8_t *A_data, TSFA tSFA, const float *x, int M, int K) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    const int KB = K / 32;
    if (warp >= M * KB) return;
    const int m = warp / KB, kb = warp % KB;
    const float v = x[(size_t)m * K + kb * 32 + lane];
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const int se = pulsar_mx_shared_exp(a);
    const __nv_fp8_e4m3 e = pulsar_mx_encode(v, se);
    A_data[(size_t)m * K + kb * 32 + lane] = e.__x;
    if (lane == 0) tSFA(m, kb * 32, 0) = ElementSF::bitcast(pulsar_mx_scale_byte(se));
}

/* Weight scale re-layout: LT swizzle (pulsar_mx_sfoff) -> CUTLASS SFB layout.
 * Also counts how many (n, kb) land at a DIFFERENT linear byte than the LT
 * slab put them: 0 means the two swizzles are the same function and the LT
 * scale slab is directly usable as SFB. */
template <class TSFB>
__global__ static void ft_relayout_sfb(TSFB tSFB, const uint8_t *sfb_base, const uint8_t *lt_scale,
                                       int N, int KB, int KBp, unsigned long long *n_mismatch) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)N * KB) return;
    const int n = (int)(idx / KB), kb = (int)(idx % KB);
    const int lt_off = pulsar_mx_sfoff(n, kb, KBp);
    tSFB(n, kb * 32, 0) = ElementSF::bitcast(lt_scale[lt_off]);
    const long ct_off = (const uint8_t *)&tSFB(n, kb * 32, 0) - sfb_base;
    if (ct_off != lt_off) atomicAdd(n_mismatch, 1ull);
}

struct ft_weight {
    int N, K, KB, KBp;
    uint8_t *B_data;          // LT data slab, device
    uint8_t *lt_scale;        // LT swizzled scale slab, device
    ElementSF *B_sf;          // CUTLASS-layout SFB
    unsigned long long sfb_mismatch;
};

struct ft_act {
    int N, K, KB, sm_count;
    uint8_t *A_data;          // [16, K] e4m3
    ElementSF *A_sf;          // SFA for up to 128 rows
    void *ws; size_t ws_bytes;   // one workspace: launches are serial on one stream
};

extern "C" ft_weight *ft_prepare(const uint8_t *host_lt_data, const uint8_t *host_lt_scale, int N, int K,
                                 unsigned long long *sfb_mismatch_out) {
    if (K % 128 != 0 || K < 128) { fprintf(stderr, "fixed-tile probe: K=%d not a multiple of 128\n", K); return nullptr; }
    ft_weight *c = new ft_weight();
    memset(c, 0, sizeof *c);
    c->N = N; c->K = K; c->KB = K / 32; c->KBp = pulsar_mx_rup(c->KB, 4);
    const size_t data_bytes = (size_t)N * K;
    const size_t lt_scale_bytes = (size_t)pulsar_mx_rup(N, 128) * c->KBp;
    if (ft_check(cudaMalloc(&c->B_data, data_bytes), "malloc B") ||
        ft_check(cudaMalloc(&c->lt_scale, lt_scale_bytes), "malloc lt scale") ||
        ft_check(cudaMemcpy(c->B_data, host_lt_data, data_bytes, cudaMemcpyHostToDevice), "copy B") ||
        ft_check(cudaMemcpy(c->lt_scale, host_lt_scale, lt_scale_bytes, cudaMemcpyHostToDevice), "copy scale"))
        return nullptr;
    /* SFB layout: one block-scaled config for both kernels (same mainloop type). */
    auto lSFB = BlkCfg::tile_atom_to_shape_SFB(make_shape(128, N, K, 1));
    const size_t sfb_bytes = (size_t)cosize(lSFB);
    if (ft_check(cudaMalloc(&c->B_sf, sfb_bytes), "malloc SFB") ||
        ft_check(cudaMemset(c->B_sf, 0, sfb_bytes), "zero SFB")) return nullptr;
    unsigned long long *d_mis = nullptr;
    if (ft_check(cudaMalloc(&d_mis, sizeof *d_mis), "malloc mis") || ft_check(cudaMemset(d_mis, 0, sizeof *d_mis), "zero mis")) return nullptr;
    {
        auto tSFB = make_tensor(make_gmem_ptr(c->B_sf), lSFB);
        const long total = (long)N * c->KB;
        ft_relayout_sfb<<<(unsigned)((total + 255) / 256), 256>>>(tSFB, (const uint8_t *)c->B_sf, c->lt_scale, N, c->KB, c->KBp, d_mis);
        if (ft_check(cudaGetLastError(), "relayout launch") || ft_check(cudaDeviceSynchronize(), "relayout sync")) return nullptr;
        cudaMemcpy(&c->sfb_mismatch, d_mis, sizeof c->sfb_mismatch, cudaMemcpyDeviceToHost);
        cudaFree(d_mis);
    }
    if (sfb_mismatch_out) *sfb_mismatch_out = c->sfb_mismatch;
    return c;
}

/* The scheduler arguments of a variant on the stream-K kernel. */
static void ft_sk_args(SKArgs &s, int variant) {
    s.splits = ft_splits[variant];
    s.decomposition_mode = ft_splits[variant] > 1 ? SKDecomp::SplitK : SKDecomp::StreamK;
    s.reduction_mode = SKReduce::Deterministic;
}

extern "C" ft_act *ft_act_prepare(int N, int K) {
    if (K % 128 != 0 || K < 128) { fprintf(stderr, "fixed-tile probe: K=%d not a multiple of 128\n", K); return nullptr; }
    ft_act *a = new ft_act();
    memset(a, 0, sizeof *a);
    a->N = N; a->K = K; a->KB = K / 32;
    if (ft_device_info(&a->sm_count, nullptr)) return nullptr;
    auto lSFA = BlkCfg::tile_atom_to_shape_SFA(make_shape(128, N, K, 1));
    if (ft_check(cudaMalloc(&a->A_data, (size_t)16 * K), "malloc A") ||
        ft_check(cudaMalloc(&a->A_sf, (size_t)cosize(lSFA)), "malloc SFA") ||
        ft_check(cudaMemset(a->A_sf, 0, (size_t)cosize(lSFA)), "zero SFA")) return nullptr;
    /* workspace: max over M <= 16 and variants (split-K partials + barriers) */
    for (int M = 1; M <= 16; M++) {
        size_t w = FT1::Gemm::get_workspace_size(FT1::args(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, a->sm_count));
        if (w > a->ws_bytes) a->ws_bytes = w;
        for (int v = V_S2; v < V_N; v++) {
            auto ak = FTK::args(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, a->sm_count);
            ft_sk_args(ak.scheduler, v);
            w = FTK::Gemm::get_workspace_size(ak);
            if (w > a->ws_bytes) a->ws_bytes = w;
        }
    }
    if (a->ws_bytes && ft_check(cudaMalloc(&a->ws, a->ws_bytes), "malloc ws")) return nullptr;
    return a;
}

extern "C" int ft_pack(ft_act *a, const pulsar_gpu_tensor *x, int M) {
    if (!a || !x || M < 1 || M > 16) return 4;
    if (x->bytes < (uint64_t)M * a->K * sizeof(float)) return 5;
    auto lSFA = BlkCfg::tile_atom_to_shape_SFA(make_shape(M, a->N, a->K, 1));
    auto tSFA = make_tensor(make_gmem_ptr(a->A_sf), lSFA);
    const long warps = (long)M * a->KB;
    ft_pack_a<<<(unsigned)((warps * 32 + 255) / 256), 256>>>(a->A_data, tSFA, (const float *)x->ptr, M, a->K);
    return ft_check(cudaGetLastError(), "pack launch") ? 10 : 0;
}

struct ft_plan {
    int variant;
    void *ws;
    FT1::Gemm g1; FT1::Arguments a1;
    FTK::Gemm gk; FTK::Arguments ak;
};

template <class F>
static int ft_plan_init(typename F::Gemm &g, typename F::Arguments &a, void *ws) {
    if (F::Gemm::can_implement(a) != cutlass::Status::kSuccess) return 1;
    return g.initialize(a, ws) == cutlass::Status::kSuccess ? 0 : 2;   /* TMA descriptors, scheduler params, smem attribute */
}

extern "C" int ft_plan_make(ft_weight *w, ft_act *a, int variant, int M, pulsar_gpu_tensor *D, ft_plan **out) {
    if (!w || !a || !D || !out || M < 1 || M > 16 || w->N != a->N || w->K != a->K) return 4;
    if (D->bytes < (uint64_t)M * w->N * sizeof(float)) return 5;
    if (variant < 0 || variant >= V_N) return 6;
    ft_plan *p = new ft_plan();
    p->variant = variant; p->ws = a->ws;
    int rc;
    if (variant == V_S1) {
        p->a1 = FT1::args((float *)D->ptr, a->A_data, a->A_sf, w->B_data, w->B_sf, M, w->N, w->K, a->sm_count);
        rc = ft_plan_init<FT1>(p->g1, p->a1, a->ws);
    } else {
        p->ak = FTK::args((float *)D->ptr, a->A_data, a->A_sf, w->B_data, w->B_sf, M, w->N, w->K, a->sm_count);
        ft_sk_args(p->ak.scheduler, variant);
        rc = ft_plan_init<FTK>(p->gk, p->ak, a->ws);
    }
    if (rc) { delete p; return rc; }
    *out = p;
    return 0;
}

/* Per launch: the barrier workspace back to zero (split-K/stream-K arrivals
 * are counters; a no-op for the plain tile) and the kernel.  Both on the
 * per-thread default stream, like the engine's launches. */
extern "C" int ft_plan_run(ft_plan *p) {
    if (!p) return 4;
    if (p->variant == V_S1) {
        if (FT1::GemmKernel::initialize_workspace(p->a1, p->ws) != cutlass::Status::kSuccess) return 3;
        return p->g1.run() == cutlass::Status::kSuccess ? 0 : 3;
    }
    if (FTK::GemmKernel::initialize_workspace(p->ak, p->ws) != cutlass::Status::kSuccess) return 3;
    return p->gk.run() == cutlass::Status::kSuccess ? 0 : 3;
}

extern "C" void ft_plan_decomp(const ft_plan *p, ft_decomp *d) {
    memset(d, 0, sizeof *d);
    if (!p) return;
    if (p->variant == V_S1) {
        const dim3 g = FT1::Gemm::get_grid_shape(p->g1.params());
        d->splits = 1; d->ctas = g.x * g.y * g.z;
        return;
    }
    const auto &prm = p->gk.params();
    const auto &sk = prm.scheduler.sk_params_;
    d->splits    = sk.divmod_splits_.divisor;   /* EFFECTIVE (after adjust_split_count) */
    d->sk_units  = sk.sk_units_;
    d->sk_tiles  = sk.sk_tiles_;
    d->big_units = sk.big_units_;
    const dim3 g = FTK::Gemm::get_grid_shape(prm);
    d->ctas = g.x * g.y * g.z;
}

extern "C" void ft_plan_release(ft_plan *p) { delete p; }

extern "C" int ft_sync(void) { return ft_check(cudaDeviceSynchronize(), "sync"); }

extern "C" void ft_release(ft_weight *c) {
    if (!c) return;
    cudaFree(c->B_data); cudaFree(c->lt_scale); cudaFree(c->B_sf);
    delete c;
}

extern "C" void ft_act_release(ft_act *a) {
    if (!a) return;
    cudaFree(a->A_data); cudaFree(a->A_sf);
    if (a->ws) cudaFree(a->ws);
    delete a;
}

/* ---- stage 0.4: the bf16 family (router, output head) --------------------
 * A plain (non-block-scaled) tensor-core GEMM with a fixed tile, sm80-style
 * mma.sync m16n8k16 through cutlass::gemm::device::GemmUniversal (no split-K,
 * identity swizzle: fixed reduction order, M never consulted).
 * A = activations rounded f32 -> bf16 with the engine's RNE (f32_to_bf16_kernel),
 * B = the bf16 weight [N][K] straight from the mmap (K-major == ColumnMajor). */
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

/* Stage 0.4b: the 64x{64,128}x32 / 4-stage tile measured 170 GB/s on the 1 GB
 * head against cuBLAS's 224 (rows/L151.md).  The sm120 TMA builder refuses
 * bf16 (f8f6f4 elements only), so the lever is the sm80-style tile itself:
 * "64" now means 64x128x64 / 4 stages, "128" means 128x128x64 / 3 stages. */
template <int V> struct FBShape;
template <> struct FBShape<0> { static constexpr int TM = 64, TN_ = 128, TK = 64, ST = 4; };   // stage-0.4b winner
template <> struct FBShape<1> { static constexpr int TM = 64, TN_ = 64,  TK = 64, ST = 3; };   // smaller smem -> more CTAs/SM
template <> struct FBShape<2> { static constexpr int TM = 32, TN_ = 128, TK = 64, ST = 3; };   // 32-row tile
template <> struct FBShape<3> { static constexpr int TM = 64, TN_ = 128, TK = 64, ST = 2; };   // fewer stages -> more CTAs/SM
extern "C" int fb_nvariants(void) { return 4; }
extern "C" const char *fb_variant_name(int v) { static const char *n[] = {"64x128x64/4", "64x64x64/3", "32x128x64/3", "64x128x64/2"}; return v >= 0 && v < 4 ? n[v] : "?"; }

template <int V>
struct FB {
    using S = FBShape<V>;
    using Gemm = cutlass::gemm::device::GemmUniversal<
        cutlass::bfloat16_t, cutlass::layout::RowMajor,
        cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<S::TM, S::TN_, S::TK>,
        cutlass::gemm::GemmShape<S::TM / 2, S::TN_ / 2, S::TK>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        S::ST, 8 /* AlignA */, 8 /* AlignB */>;
};

__global__ static void fb_f32_to_bf16(uint16_t *out, const float *x, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const uint32_t u = __float_as_uint(x[i]);
        out[i] = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
}

struct fb_weight { int N, K; uint16_t *W; };                       // bf16 [N][K]
struct fb_act    { int N, K; uint16_t *A; void *ws; size_t ws_bytes; };   // bf16 [16][K]

template <int V>
static typename FB<V>::Gemm::Arguments fb_args(int N, int K, const uint16_t *A, const uint16_t *W, int M, float *D) {
    return typename FB<V>::Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, {1.0f, 0.0f},
        A, W, D, D,
        0, 0, 0, 0,
        (int64_t)K, (int64_t)K, (int64_t)N, (int64_t)N);
}

extern "C" fb_weight *fb_prepare(const uint8_t *host_w_bf16, int N, int K) {
    if (K % 8 != 0 || N % 8 != 0) { fprintf(stderr, "fixed-tile probe (bf16): N=%d K=%d not multiples of 8\n", N, K); return nullptr; }
    fb_weight *c = new fb_weight();
    memset(c, 0, sizeof *c);
    c->N = N; c->K = K;
    const size_t wbytes = (size_t)N * K * 2;
    if (ft_check(cudaMalloc(&c->W, wbytes), "malloc W bf16") ||
        ft_check(cudaMemcpy(c->W, host_w_bf16, wbytes, cudaMemcpyHostToDevice), "copy W bf16")) return nullptr;
    return c;
}

extern "C" fb_act *fb_act_prepare(int N, int K) {
    if (K % 8 != 0 || N % 8 != 0) { fprintf(stderr, "fixed-tile probe (bf16): N=%d K=%d not multiples of 8\n", N, K); return nullptr; }
    fb_act *a = new fb_act();
    memset(a, 0, sizeof *a);
    a->N = N; a->K = K;
    if (ft_check(cudaMalloc(&a->A, (size_t)16 * K * 2), "malloc A bf16")) return nullptr;
    for (int M = 1; M <= 16; M++) {
        size_t w;
        w = FB<0>::Gemm::get_workspace_size(fb_args<0>(N, K, nullptr, nullptr, M, nullptr)); if (w > a->ws_bytes) a->ws_bytes = w;
        w = FB<1>::Gemm::get_workspace_size(fb_args<1>(N, K, nullptr, nullptr, M, nullptr)); if (w > a->ws_bytes) a->ws_bytes = w;
        w = FB<2>::Gemm::get_workspace_size(fb_args<2>(N, K, nullptr, nullptr, M, nullptr)); if (w > a->ws_bytes) a->ws_bytes = w;
        w = FB<3>::Gemm::get_workspace_size(fb_args<3>(N, K, nullptr, nullptr, M, nullptr)); if (w > a->ws_bytes) a->ws_bytes = w;
    }
    if (a->ws_bytes && ft_check(cudaMalloc(&a->ws, a->ws_bytes), "malloc ws bf16")) return nullptr;
    return a;
}

extern "C" int fb_pack(fb_act *a, const pulsar_gpu_tensor *x, int M) {
    if (!a || !x || M < 1 || M > 16) return 4;
    if (x->bytes < (uint64_t)M * a->K * sizeof(float)) return 5;
    const uint64_t n = (uint64_t)M * a->K;
    fb_f32_to_bf16<<<(unsigned)((n + 255) / 256), 256>>>(a->A, (const float *)x->ptr, n);
    return ft_check(cudaGetLastError(), "bf16 pack launch") ? 10 : 0;
}

struct fb_plan {
    int variant;
    FB<0>::Gemm g0; FB<1>::Gemm g1; FB<2>::Gemm g2; FB<3>::Gemm g3;
};

template <int V>
static int fb_plan_init(typename FB<V>::Gemm &g, fb_weight *w, fb_act *a, int M, float *D) {
    auto args = fb_args<V>(w->N, w->K, a->A, w->W, M, D);
    if (FB<V>::Gemm::can_implement(args) != cutlass::Status::kSuccess) return 1;
    return g.initialize(args, a->ws) == cutlass::Status::kSuccess ? 0 : 2;
}

extern "C" int fb_plan_make(fb_weight *w, fb_act *a, int variant, int M, pulsar_gpu_tensor *D, fb_plan **out) {
    if (!w || !a || !D || !out || M < 1 || M > 16 || w->N != a->N || w->K != a->K) return 4;
    if (D->bytes < (uint64_t)M * w->N * sizeof(float)) return 5;
    if (variant < 0 || variant > 3) return 6;
    fb_plan *p = new fb_plan();
    p->variant = variant;
    float *Dp = (float *)D->ptr;
    int rc;
    switch (variant) {
    case 0:  rc = fb_plan_init<0>(p->g0, w, a, M, Dp); break;
    case 1:  rc = fb_plan_init<1>(p->g1, w, a, M, Dp); break;
    case 2:  rc = fb_plan_init<2>(p->g2, w, a, M, Dp); break;
    default: rc = fb_plan_init<3>(p->g3, w, a, M, Dp); break;
    }
    if (rc) { delete p; return rc; }
    *out = p;
    return 0;
}

extern "C" int fb_plan_run(fb_plan *p) {
    if (!p) return 4;
    cutlass::Status s;
    switch (p->variant) {
    case 0:  s = p->g0.run(); break;
    case 1:  s = p->g1.run(); break;
    case 2:  s = p->g2.run(); break;
    default: s = p->g3.run(); break;
    }
    return s == cutlass::Status::kSuccess ? 0 : 3;
}

extern "C" void fb_plan_release(fb_plan *p) { delete p; }

extern "C" void fb_release(fb_weight *c) {
    if (!c) return;
    cudaFree(c->W);
    delete c;
}

extern "C" void fb_act_release(fb_act *a) {
    if (!a) return;
    cudaFree(a->A);
    if (a->ws) cudaFree(a->ws);
    delete a;
}

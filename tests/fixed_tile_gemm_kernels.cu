// tests/fixed_tile_gemm_kernels.cu -- L151-C stage 0: the CUTLASS side of the
// fixed-tile probe.  A block-scaled MXFP8 x MXFP8 GEMM on sm120 with ONE tile
// configuration (128 x TN x 128, TN in {64,128}) regardless of M, fed the SAME
// E4M3 operands the engine's dense path uses: activations quantised with the
// shared pulsar_mx_* helpers (bit-identical bytes to mxfp8_quant_act_kernel),
// weights = the MXFP8_LT slabs straight from the mmap (data reused as CUTLASS
// K-major B; the E8M0 scale byte re-laid into CUTLASS's SFB layout).  Plain C
// interface so the driver (fixed_tile_gemm_probe.cpp) can stay a C++23 engine
// client while this unit compiles as the CUTLASS unit does (nvcc, C++17).
//
// PRICING ONLY.  Nothing here is reachable from the engine.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
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

template <int TN>
struct FT {
    using TileShape    = Shape<_128, Int<TN>, _128>;   /* 128 rows: the SF atom is 128x4; a 64-row tile cannot describe its TMA load */
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
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using BlkCfg     = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    static typename Gemm::Arguments args(float *D, const uint8_t *A_data, const ElementSF *A_sf,
                                         const uint8_t *B_data, const ElementSF *B_sf, int M, int N, int K) {
        auto strideA = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, K, 1});
        auto strideB = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, K, 1});
        auto strideC = cutlass::make_cute_packed_stride(typename GemmKernel::StrideC{}, {M, N, 1});
        auto strideD = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M, N, 1});
        auto lSFA = BlkCfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
        auto lSFB = BlkCfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
        return typename Gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
            { reinterpret_cast<const typename ElementA::DataType *>(A_data), strideA,
              reinterpret_cast<const typename ElementB::DataType *>(B_data), strideB,
              A_sf, lSFA, B_sf, lSFB },
            { {1.0f, 0.0f}, D, strideC, D, strideD } };
    }
};

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

struct ft_ctx {
    int N, K, KB, KBp;
    int m_max;                // rows the A side and the workspaces are sized for (L183: production M)
    std::vector<char> checked;  // can_implement verified per M
    uint8_t *B_data;          // LT data slab, device
    uint8_t *lt_scale;        // LT swizzled scale slab, device
    ElementSF *B_sf;          // CUTLASS-layout SFB
    uint8_t *A_data;          // [m_max, K] e4m3
    ElementSF *A_sf;          // SFA for rup(m_max, 128) rows
    void *ws64, *ws128;
    size_t ws64_bytes, ws128_bytes;
    unsigned long long sfb_mismatch;
};

static int ft_check(cudaError_t e, const char *what) {
    if (e == cudaSuccess) return 0;
    fprintf(stderr, "fixed-tile probe: %s: %s\n", what, cudaGetErrorString(e));
    return 1;
}

extern "C" ft_ctx *ft_prepare(const uint8_t *host_lt_data, const uint8_t *host_lt_scale, int N, int K,
                              int m_max, unsigned long long *sfb_mismatch_out) {
    if (K % 128 != 0 || K < 128) { fprintf(stderr, "fixed-tile probe: K=%d not a multiple of 128\n", K); return nullptr; }
    if (m_max < 1) { fprintf(stderr, "fixed-tile probe: m_max=%d\n", m_max); return nullptr; }
    ft_ctx *c = new ft_ctx();
    c->N = N; c->K = K; c->KB = K / 32; c->KBp = pulsar_mx_rup(c->KB, 4);
    c->m_max = m_max;
    c->checked.assign((size_t)m_max + 2u, 0);
    const size_t data_bytes = (size_t)N * K;
    const size_t lt_scale_bytes = (size_t)pulsar_mx_rup(N, 128) * c->KBp;
    if (ft_check(cudaMalloc(&c->B_data, data_bytes), "malloc B") ||
        ft_check(cudaMalloc(&c->lt_scale, lt_scale_bytes), "malloc lt scale") ||
        ft_check(cudaMemcpy(c->B_data, host_lt_data, data_bytes, cudaMemcpyHostToDevice), "copy B") ||
        ft_check(cudaMemcpy(c->lt_scale, host_lt_scale, lt_scale_bytes, cudaMemcpyHostToDevice), "copy scale"))
        return nullptr;
    /* SFB layout: same block-scaled config for both tile widths (SF vector 32,
     * 128x4 atoms); take TN=128's and assert TN=64's has the same cosize. */
    auto lSFB = FT<128>::BlkCfg::tile_atom_to_shape_SFB(make_shape(128, N, K, 1));
    auto lSFB64 = FT<64>::BlkCfg::tile_atom_to_shape_SFB(make_shape(128, N, K, 1));
    const size_t sfb_bytes = (size_t)cosize(lSFB);
    if ((size_t)cosize(lSFB64) != sfb_bytes) { fprintf(stderr, "fixed-tile probe: SFB cosize differs between TN variants\n"); return nullptr; }
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
    /* A side: m_max rows of data, SFA sized for the 128-row atom over m_max rows. */
    auto lSFA = FT<128>::BlkCfg::tile_atom_to_shape_SFA(make_shape(pulsar_mx_rup(m_max, 128), N, K, 1));
    if (ft_check(cudaMalloc(&c->A_data, (size_t)m_max * K), "malloc A") ||
        ft_check(cudaMalloc(&c->A_sf, (size_t)cosize(lSFA)), "malloc SFA") ||
        ft_check(cudaMemset(c->A_sf, 0, (size_t)cosize(lSFA)), "zero SFA")) return nullptr;
    /* workspaces: max over the decode range and the top of the probed range */
    auto consider = [&](int M) {
        if (M < 1 || M > m_max) return;
        auto a64 = FT<64>::args(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K);
        auto a128 = FT<128>::args(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K);
        const size_t w64 = FT<64>::Gemm::get_workspace_size(a64), w128 = FT<128>::Gemm::get_workspace_size(a128);
        if (w64 > c->ws64_bytes) c->ws64_bytes = w64;
        if (w128 > c->ws128_bytes) c->ws128_bytes = w128;
    };
    for (int M = 1; M <= 16; M++) consider(M);
    for (int M = m_max - 2; M <= m_max; M++) consider(M);
    if (c->ws64_bytes && ft_check(cudaMalloc(&c->ws64, c->ws64_bytes), "malloc ws64")) return nullptr;
    if (c->ws128_bytes && ft_check(cudaMalloc(&c->ws128, c->ws128_bytes), "malloc ws128")) return nullptr;
    return c;
}

template <int TN>
static int ft_run_tn(ft_ctx *c, const float *x_dev, int M, float *D_dev) {
    auto lSFA = FT<TN>::BlkCfg::tile_atom_to_shape_SFA(make_shape(M, c->N, c->K, 1));
    auto tSFA = make_tensor(make_gmem_ptr(c->A_sf), lSFA);
    const long warps = (long)M * c->KB;
    ft_pack_a<<<(unsigned)((warps * 32 + 255) / 256), 256>>>(c->A_data, tSFA, x_dev, M, c->K);
    if (cudaGetLastError() != cudaSuccess) return 10;
    auto a = FT<TN>::args(D_dev, c->A_data, c->A_sf, c->B_data, c->B_sf, M, c->N, c->K);
    typename FT<TN>::Gemm gemm;
    if (!c->checked[(size_t)M]) {
        if (gemm.can_implement(a) != cutlass::Status::kSuccess) return 1;
        c->checked[(size_t)M] = 1;
    }
    if (gemm.initialize(a, TN == 64 ? c->ws64 : c->ws128) != cutlass::Status::kSuccess) return 2;
    return gemm.run() == cutlass::Status::kSuccess ? 0 : 3;
}

extern "C" int ft_run(ft_ctx *c, int tn, const pulsar_gpu_tensor *x, int M, pulsar_gpu_tensor *D) {
    if (!c || !x || !D || M < 1 || M > c->m_max) return 4;
    if (x->bytes < (uint64_t)M * c->K * sizeof(float) || D->bytes < (uint64_t)M * c->N * sizeof(float)) return 5;
    const float *x_dev = (const float *)x->ptr;
    float *D_dev = (float *)D->ptr;
    return tn == 64 ? ft_run_tn<64>(c, x_dev, M, D_dev) : ft_run_tn<128>(c, x_dev, M, D_dev);
}

extern "C" int ft_sync(void) { return ft_check(cudaDeviceSynchronize(), "sync"); }

extern "C" void ft_release(ft_ctx *c) {
    if (!c) return;
    cudaFree(c->B_data); cudaFree(c->lt_scale); cudaFree(c->B_sf); cudaFree(c->A_data); cudaFree(c->A_sf);
    if (c->ws64) cudaFree(c->ws64);
    if (c->ws128) cudaFree(c->ws128);
    delete c;
}

/* ---- stage 0.4: the bf16 family (router, output head) --------------------
 * A plain (non-block-scaled) tensor-core GEMM with a fixed 64 x TN x 32 tile,
 * sm80-style mma.sync m16n8k16 through cutlass::gemm::device::GemmUniversal
 * (no split-K, identity swizzle: fixed reduction order, M never consulted).
 * A = activations rounded f32 -> bf16 with the engine's RNE (f32_to_bf16_kernel),
 * B = the bf16 weight [N][K] straight from the mmap (K-major == ColumnMajor). */
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

/* Stage 0.4b: the 64x{64,128}x32 / 4-stage tile measured 170 GB/s on the 1 GB
 * head against cuBLAS's 224 (rows/L151.md).  The sm120 TMA builder refuses
 * bf16 (f8f6f4 elements only), so the lever is the sm80-style tile itself:
 * "64" now means 64x128x64 / 4 stages, "128" means 128x128x64 / 3 stages. */
template <int TN> struct FBShape;
template <> struct FBShape<64>  { static constexpr int TM = 64,  TN_ = 128, TK = 64, ST = 4; };
template <> struct FBShape<128> { static constexpr int TM = 128, TN_ = 128, TK = 64, ST = 3; };

template <int TN>
struct FB {
    using S = FBShape<TN>;
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

struct fb_ctx {
    int N, K;
    uint16_t *W;      // bf16 [N][K]
    uint16_t *A;      // bf16 [m_max][K]
    void *ws64, *ws128;
    size_t ws64_bytes, ws128_bytes;
    int m_max;
    std::vector<char> checked;
};

template <int TN>
static typename FB<TN>::Gemm::Arguments fb_args(fb_ctx *c, const uint16_t *A, int M, float *D) {
    return typename FB<TN>::Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, {M, c->N, c->K}, 1, {1.0f, 0.0f},
        A, c->W, D, D,
        0, 0, 0, 0,
        (int64_t)c->K, (int64_t)c->K, (int64_t)c->N, (int64_t)c->N);
}

extern "C" fb_ctx *fb_prepare(const uint8_t *host_w_bf16, int N, int K, int m_max) {
    if (K % 8 != 0 || N % 8 != 0) { fprintf(stderr, "fixed-tile probe (bf16): N=%d K=%d not multiples of 8\n", N, K); return nullptr; }
    if (m_max < 1) { fprintf(stderr, "fixed-tile probe (bf16): m_max=%d\n", m_max); return nullptr; }
    fb_ctx *c = new fb_ctx();
    c->N = N; c->K = K;
    c->m_max = m_max;
    c->checked.assign((size_t)m_max + 2u, 0);
    const size_t wbytes = (size_t)N * K * 2;
    if (ft_check(cudaMalloc(&c->W, wbytes), "malloc W bf16") ||
        ft_check(cudaMemcpy(c->W, host_w_bf16, wbytes, cudaMemcpyHostToDevice), "copy W bf16") ||
        ft_check(cudaMalloc(&c->A, (size_t)m_max * K * 2), "malloc A bf16")) return nullptr;
    auto consider = [&](int M) {
        if (M < 1 || M > m_max) return;
        const size_t w64 = FB<64>::Gemm::get_workspace_size(fb_args<64>(c, nullptr, M, nullptr));
        const size_t w128 = FB<128>::Gemm::get_workspace_size(fb_args<128>(c, nullptr, M, nullptr));
        if (w64 > c->ws64_bytes) c->ws64_bytes = w64;
        if (w128 > c->ws128_bytes) c->ws128_bytes = w128;
    };
    for (int M = 1; M <= 16; M++) consider(M);
    for (int M = m_max - 2; M <= m_max; M++) consider(M);
    if (c->ws64_bytes && ft_check(cudaMalloc(&c->ws64, c->ws64_bytes), "malloc ws64 bf16")) return nullptr;
    if (c->ws128_bytes && ft_check(cudaMalloc(&c->ws128, c->ws128_bytes), "malloc ws128 bf16")) return nullptr;
    return c;
}

template <int TN>
static int fb_run_tn(fb_ctx *c, const float *x_dev, int M, float *D_dev) {
    const uint64_t n = (uint64_t)M * c->K;
    fb_f32_to_bf16<<<(unsigned)((n + 255) / 256), 256>>>(c->A, x_dev, n);
    if (cudaGetLastError() != cudaSuccess) return 10;
    typename FB<TN>::Gemm gemm;
    auto a = fb_args<TN>(c, c->A, M, D_dev);
    if (!c->checked[(size_t)M]) {
        if (gemm.can_implement(a) != cutlass::Status::kSuccess) return 1;
        c->checked[(size_t)M] = 1;
    }
    if (gemm.initialize(a, TN == 64 ? c->ws64 : c->ws128) != cutlass::Status::kSuccess) return 2;
    return gemm.run() == cutlass::Status::kSuccess ? 0 : 3;
}

/* L183: write the engine's bf16 activation PLANE for x (the bf16-weight GEMM
 * reads only a producer-emitted plane since L159; the probe is the producer
 * here, with the same RNE rounding the engine's kernels use). */
extern "C" int fb_emit_plane(const pulsar_gpu_tensor *x, int M, int K, void *xb) {
    if (!x || !xb || M < 1 || K < 1 || x->bytes < (uint64_t)M * K * sizeof(float)) return 4;
    const uint64_t n = (uint64_t)M * K;
    fb_f32_to_bf16<<<(unsigned)((n + 255) / 256), 256>>>((uint16_t *)xb, (const float *)x->ptr, n);
    return cudaGetLastError() == cudaSuccess ? 0 : 10;
}

extern "C" int fb_run(fb_ctx *c, int tn, const pulsar_gpu_tensor *x, int M, pulsar_gpu_tensor *D) {
    if (!c || !x || !D || M < 1 || M > c->m_max) return 4;
    if (x->bytes < (uint64_t)M * c->K * sizeof(float) || D->bytes < (uint64_t)M * c->N * sizeof(float)) return 5;
    return tn == 64 ? fb_run_tn<64>(c, (const float *)x->ptr, M, (float *)D->ptr)
                    : fb_run_tn<128>(c, (const float *)x->ptr, M, (float *)D->ptr);
}

extern "C" void fb_release(fb_ctx *c) {
    if (!c) return;
    cudaFree(c->W); cudaFree(c->A);
    if (c->ws64) cudaFree(c->ws64);
    if (c->ws128) cudaFree(c->ws128);
    delete c;
}

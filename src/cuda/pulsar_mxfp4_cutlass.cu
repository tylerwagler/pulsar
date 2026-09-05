// pulsar_mxfp4_cutlass.cu — CUTLASS MXFP4 tensor-core expert FFN for the ds4 MoE (sm_120f).
// Weights arrive pre-packed in CUTLASS B layout (from the offline converter); activations are
// E4M3 (+ue8m0 block scales) -- W4A8, matching the source model's activation format.
// Path: gather the producer's E4M3 x -> gate/up GEMM -> fused SwiGLU + E4M3 pack -> down GEMM.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "pulsar_gpu.h"
#include "pulsar_cuda_internal.h"   /* cuda_ok: every launch and memset on the hot path is checked here (L189) */
#include "pulsar_cuda_mx.cuh"   /* the single source for pulsar_mx_sfoff */
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

using namespace cute;

// ---- GEMM: W4A8 mixed block-scaled — A=activation MXFP8 (E4M3, RowMajor) x B=weight MXFP4
// (E2M1, ColumnMajor) -> f32 D. This is the native DeepSeek-V4-Flash expert scheme (expert_dtype
// fp4 x activation e4m3, dynamic UE8M0 block scale). The sm120 CollectiveBuilder selects the
// kind::f8f6f4 (MX_F4F6F8) instruction (cute atom SM120_16x8x32_TN<e2m1,e4m3,f32>, mma_sm120.hpp:157).
// The weight-side SF (SFB) swizzle is BYTE-IDENTICAL to the former all-fp4 (mxf4nvf4) config
// (verified: same tile_atom_to_shape_SFB shape/stride/size), so the type-40 weight repack and the
// shipped blk 0-2 stay valid unchanged; only the activation element (fp4->fp8) and its packer change. ----
using ElementA   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB   = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementD   = float;
using ElementC   = float;
using ElementAcc = float;
using LayoutA    = cutlass::layout::RowMajor;
using LayoutB    = cutlass::layout::ColumnMajor;
using LayoutC    = cutlass::layout::RowMajor;
using LayoutD    = cutlass::layout::RowMajor;
constexpr int AlignA = 32, AlignB = 32;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
/* Occupancy sweep knob (2026-08-07).  ncu says both CUTLASS MXFP4 kernels run
 * 168 registers x 384 threads = 64512 regs/block against 65536 per SM, i.e.
 * exactly ONE block resides and occupancy pins at 25% -- the same cliff that
 * cost attention 1.42x until launch_bounds fixed it.  Here the register count
 * is set by the f32 accumulator, which is TileM x TileN: 128x128 f32 over 384
 * threads is ~43 regs of accumulator before anything else.
 * Shrinking the tile trades compute efficiency (more tiles, more weight
 * re-reads) for residency, so it must be MEASURED, not assumed.
 * Override: -DPULSAR_MXFP4_TILE_M=64 etc. */
#ifndef PULSAR_MXFP4_TILE_M
#define PULSAR_MXFP4_TILE_M 128
#endif
#ifndef PULSAR_MXFP4_TILE_N
#define PULSAR_MXFP4_TILE_N 128
#endif
#ifndef PULSAR_MXFP4_TILE_K
#define PULSAR_MXFP4_TILE_K 128
#endif
using TileShape    = Shape<cute::Int<PULSAR_MXFP4_TILE_M>,
                           cute::Int<PULSAR_MXFP4_TILE_N>,
                           cute::Int<PULSAR_MXFP4_TILE_K>>;
using ClusterShape = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc, ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ElementSF = cutlass::float_ue8m0_t;

// ---- GROUPED (ptr-array) MXFP4 GEMM: one launch runs every active expert's GEMM. ----
// Same element/tile config as the per-expert path above, but the builders take POINTER
// layout tags (LayoutA*, ...), which selects the SM120 blockscaled *array* collective and
// the KernelPtrArrayTmaWarpSpecializedCooperative schedule (KernelScheduleAuto -> cooperative
// for grouped; verified accepted by sm120_blockscaled_mma_builder.inl:226-230). Per-group
// {M,N,K}, A/B/D + SFA/SFB pointer arrays, and per-group strides/SF-layouts are all built on
// DEVICE from the sorted-pairs/offsets buffers -- no host readback, zero host sync.
using GProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
using GCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc, ElementC, LayoutC*, AlignC, ElementD, LayoutD*, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using GCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA, LayoutA*, AlignA, ElementB, LayoutB*, AlignB, ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename GCollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GGemmKernel = cutlass::gemm::kernel::GemmUniversal<GProblemShape, GCollectiveMainloop, GCollectiveEpilogue>;
using GGemm = cutlass::gemm::device::GemmUniversalAdapter<GGemmKernel>;

using GProbElem  = typename GProblemShape::UnderlyingProblemShape;   // Shape<int,int,int>
using GStrideA   = typename GGemmKernel::InternalStrideA;
using GStrideB   = typename GGemmKernel::InternalStrideB;
using GStrideC   = typename GGemmKernel::InternalStrideC;
using GStrideD   = typename GGemmKernel::InternalStrideD;
using GLayoutSFA = typename GGemmKernel::CollectiveMainloop::InternalLayoutSFA;
using GLayoutSFB = typename GGemmKernel::CollectiveMainloop::InternalLayoutSFB;
using GSm1xxBlkScaledConfig = typename GGemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using GElemA  = typename GGemm::ElementA;
using GElemB  = typename GGemm::ElementB;
using GElemSF = typename GGemmKernel::CollectiveMainloop::ElementSF;
using GElemC  = typename GGemm::ElementC;
using GElemD  = typename GGemm::EpilogueOutputOp::ElementOutput;

// ---- device activation packer: f32 [M,K] RowMajor -> A data (E4M3, 1 byte/elem) + SF (swizzled UE8M0).
// W4A8: activations are MXFP8 (E4M3) with a DYNAMIC per-32-element UE8M0 block scale, matching the
// DeepSeek-V4-Flash source scheme (activation_scheme dynamic, fmt e4m3, scale_fmt ue8m0). The scale
// exponent convention mirrors the engine's cuBLASLt MXFP8 packer (mxfp8_quant_act_kernel): se =
// floor(log2(amax)) - 7, data = v * 2^-se, SF byte = se+127; CUTLASS reconstructs v = data*2^(se-127+127-... )
// i.e. data * 2^(SF-127) = v exactly. The SF is written through the CUTLASS tile-atom SFA layout object
// (identical swizzle to the weight SFB), NOT the cuBLASLt VEC32 swizzle. ----


/* SwiGLU with the E4M3+E8M0 epilogue: one warp = four consecutive 32-blocks
 * of one row of `mid`; the gate/up products are combined in registers,
 * xor-shuffle amax within each block, then encoded straight into the down
 * GEMM's A operand (data + tile-atom SFA).  `mid` never touches memory as f32.
 * Same per-block amax, same shared exponent and the same
 * cutlass::float_e4m3_t(v*inv) encode as the other E4M3 producers. */
template<class TSFA>
__global__ void swiglu_pack_e4m3_warp_kernel(uint8_t *A_data, TSFA tSFA,
                                             const float *gate, const float *up,
                                             const float *w, float clamp, int M, int K){
  const int nblk = K/32;
  const long total_blk = (long)M*nblk;
  const int lane = threadIdx.x & 31;
  const long grp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;
  const long blk0 = grp*4;
  if (blk0 >= total_blk) return;
  const int m = (int)(blk0 / nblk), kb0 = (int)(blk0 % nblk);
  const size_t base = (size_t)m*K + (size_t)kb0*32;

  const float4 g = reinterpret_cast<const float4*>(gate+base)[lane];
  const float4 u = reinterpret_cast<const float4*>(up+base)[lane];
  const float wv = w[m];              /* row weight: i/mid_dim == m when K==mid_dim */
  float4 v;
  v.x = pulsar_swiglu_elem(g.x,u.x,wv,clamp);
  v.y = pulsar_swiglu_elem(g.y,u.y,wv,clamp);
  v.z = pulsar_swiglu_elem(g.z,u.z,wv,clamp);
  v.w = pulsar_swiglu_elem(g.w,u.w,wv,clamp);

  float mx = fmaxf(fmaxf(fabsf(v.x),fabsf(v.y)), fmaxf(fabsf(v.z),fabsf(v.w)));
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));

  int se=-127; if(mx>0.f){ int e=(int)floorf(log2f(mx)); se=e-7; }
  if(se<-127)se=-127; if(se>127)se=127;
  const float inv=exp2f((float)-se);

  cutlass::float_e4m3_t ob[4];
  ob[0]=cutlass::float_e4m3_t(v.x*inv); ob[1]=cutlass::float_e4m3_t(v.y*inv);
  ob[2]=cutlass::float_e4m3_t(v.z*inv); ob[3]=cutlass::float_e4m3_t(v.w*inv);
  reinterpret_cast<uint32_t*>(reinterpret_cast<cutlass::float_e4m3_t*>(A_data)+base)[lane]
      = *reinterpret_cast<const uint32_t*>(ob);
  if((lane & 7)==0) tSFA(m, (kb0+(lane>>3))*32, 0)=ElementSF::bitcast((uint8_t)(se+127));
}


/* SwiGLU straight into the E4M3 staging the down GEMM reads.  The warp packer
 * needs K a 128-multiple and 16-byte-aligned gate/up rows; any other shape is
 * refused -- there is no unfused pair behind it. */
static int swiglu_pack_activation(uint8_t *A_data, ElementSF *A_sf,
                                  const float *gate, const float *up, const float *w,
                                  float clamp, int M, int K){
  const bool shape_ok = ((K/32) % 4) == 0;
  const bool align_ok = ((uintptr_t)gate % 16)==0 && ((uintptr_t)up % 16)==0;
  if (!shape_ok || !align_ok) {
    fprintf(stderr, "pulsar: grouped MoE swiglu->E4M3: K=%d not a 128-multiple or operands unaligned -- refusing\n", K);
    return 1;
  }
  auto layoutSF = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, 0, K, 1));
  auto tSFA = make_tensor(make_gmem_ptr(A_sf), layoutSF);
  const int t = 128;
  const long groups = (long)M*(K/32)/4;
  const long thr = groups*32, bw = (thr+t-1)/t;
  static int announced = 0;
  if (!announced) {
    announced = 1;
    fprintf(stderr, "pulsar: grouped MoE swiglu->E4M3 fused (M=%d K=%d)\n", M, K);
  }
  swiglu_pack_e4m3_warp_kernel<<<(unsigned)bw,t>>>(A_data, tSFA, gate, up, w, clamp, M, K);
  return cuda_ok(cudaGetLastError(), "grouped MoE swiglu->E4M3 pack launch") ? 0 : 1;
}

/* Where the ENGINE's activation cache keeps the E8M0 byte for (row, kb): the
 * 128x4 SF atom swizzle, pulsar_mx_sfoff from pulsar_cuda_mx.cuh -- the one
 * authority, shared with every producer and reader of that plane. */

/* Gather ALREADY-E4M3 activations straight into the grouped GEMM's A operand.
 *
 * The norm that produced this activation emitted E4M3 + ue8m0 into the engine's
 * activation cache, in exactly the scheme pack_act_e4m3_rowmajor_* uses --
 * se = floor(log2(amax)) - 7, byte se+127, payload v * 2^-se -- and the two
 * encoders were verified byte-identical over a 4M-value conversion sweep and a
 * full 512x4096 block encode.  So there is nothing to recompute: move the codes.
 *
 * That turns the activation half of the gather from a 4-byte copy plus a
 * re-encode pass into a 1-byte copy, on the layers where the routed experts are
 * CUTLASS MXFP4.
 *
 * The two sides disagree only on WHERE the scale byte lives -- the cache uses
 * mx_sfoff, CUTLASS its own tile atom -- so read through one and write through
 * the other.  The byte is never recomputed.
 *
 * Padding rows carry row_src < 0 and get zero payload and a zero scale, which is
 * what the f32 path produced from its pre-zeroed rows. */
template<class TSFA>
__global__ void gather_act_e4m3_kernel(uint8_t *A_data, TSFA tSFA,
                                       const uint8_t *src_q, const uint8_t *src_sf, int src_kbp,
                                       const int32_t *row_src, int M, int K){
  const int nblk = K / 32;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)M * nblk) return;
  const int m = (int)(idx / nblk), kb = (int)(idx % nblk);
  const int src = row_src[m];
  int4 *d = reinterpret_cast<int4*>(A_data + (size_t)m * K + (size_t)kb * 32);
  if (src < 0) {
    const int4 z = make_int4(0, 0, 0, 0);
    d[0] = z; d[1] = z;
    tSFA(m, kb * 32, 0) = ElementSF::bitcast((uint8_t)0);
    return;
  }
  const int4 *s = reinterpret_cast<const int4*>(src_q + (size_t)src * K + (size_t)kb * 32);
  d[0] = s[0]; d[1] = s[1];
  tSFA(m, kb * 32, 0) = ElementSF::bitcast(src_sf[pulsar_mx_sfoff(src, kb, src_kbp)]);
}

/* Returns 0 on success, 1 when the gather launch failed (named; L189). */
static int gather_activation_e4m3(uint8_t *A_data, ElementSF *A_sf,
                                  const void *src_q, const void *src_sf, int src_kbp,
                                  const int32_t *row_src, int M, int K){
  auto layoutSF = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, 0, K, 1));
  auto tSFA = make_tensor(make_gmem_ptr(A_sf), layoutSF);
  const int nb = M * (K / 32), t = 128, b = (nb + t - 1) / t;
  gather_act_e4m3_kernel<<<b,t>>>(A_data, tSFA, (const uint8_t*)src_q, (const uint8_t*)src_sf,
                                  src_kbp, row_src, M, K);
  return cuda_ok(cudaGetLastError(), "grouped MoE E4M3 activation gather launch") ? 0 : 1;
}

static typename Gemm::Arguments make_gemm_args(float *D, const uint8_t *A_data, const ElementSF *A_sf,
                    const uint8_t *B_data, const ElementSF *B_sf, int M, int N, int K){
  auto strideA = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M,K,1});
  auto strideB = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N,K,1});
  auto strideC = cutlass::make_cute_packed_stride(typename GemmKernel::StrideC{}, {M,N,1});
  auto strideD = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M,N,1});
  auto lSFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
  auto lSFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
  return typename Gemm::Arguments{
    cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K,1},
    { reinterpret_cast<const ElementA::DataType*>(A_data), strideA,
      reinterpret_cast<const ElementB::DataType*>(B_data), strideB,
      A_sf, lSFA, B_sf, lSFB },
    { {1.0f, 0.0f}, D, strideC, D, strideD } };   // C=D ptr (beta=0, unused) to keep epilogue happy
}

// Workspace bytes CUTLASS needs for a GEMM of this shape (queried once per distinct
// (M,N,K) at scratch-sizing time; NOT malloc'd here -- caller folds this into one
// scratch allocation so the hot path never touches the CUDA allocator).
static size_t gemm_workspace_bytes(int M, int N, int K){
  auto args = make_gemm_args(nullptr,nullptr,nullptr,nullptr,nullptr,M,N,K);
  return Gemm::get_workspace_size(args);
}


static size_t align_up_bytes(size_t n, size_t a){ return (n + a - 1) / a * a; }

// ---- Single-projection W4A8 GEMM (for MIXED type-40 + iq2 layers). One logical matmul:
// out[T,out_dim] = x[T,in_dim] . W[out_dim,in_dim]^T, where W is a type-40 CUTLASS-packed MXFP4
// weight for ONE expert (data at W_d, swizzled SFB at W_sf) and x is f32 activations for the T
// tokens routed to that expert (gathered contiguously by the caller). Activations are packed to
// E4M3 (dynamic UE8M0) exactly as the full FFN's projections -- so this projection is bit-for-bit
// the same function the uniform grouped path computes for a single GEMM. No allocation, no sync;
// launches into the caller's current stream. SF pointer typed as raw bytes at the boundary. ----
static size_t proj_scratch_layout(int T, int in_dim, int out_dim,
                                  size_t *xA_off, size_t *xSF_off, size_t *xSF_n, size_t *ws_off, size_t *ws_bytes){
  const size_t a=256; size_t off=0;
  *xA_off=off; off=align_up_bytes(off+(size_t)T*in_dim, a);                 /* E4M3: 1 byte/elem */
  *xSF_n=(size_t)((T+127)/128*128)*((in_dim/32+3)/4*4);
  *xSF_off=off; off=align_up_bytes(off+(*xSF_n)*sizeof(ElementSF), a);
  *ws_bytes=gemm_workspace_bytes(T, out_dim, in_dim);
  *ws_off=off; off=align_up_bytes(off+(*ws_bytes), a);
  return off;
}
size_t pulsar_cutlass_proj_scratch_bytes(int T, int in_dim, int out_dim){
  size_t a,b,c,d,e; return proj_scratch_layout(T,in_dim,out_dim,&a,&b,&c,&d,&e);
}

// ============================================================================================
// GROUPED MXFP4 PREFILL FFN — single ptr-array grouped GEMM per logical matmul (gate/up/down),
// replacing routed_moe_launch_cutlass's blocking per-expert host loop. All per-group problem
// shapes, pointers, strides and SF-layouts are built on DEVICE from the sorted-pairs/offsets
// buffers; there is NO host readback and NO host sync.
//
// SF-LAYOUT INVARIANT (the highest-risk part): the SM120 blockscaled SF atom spans 128 rows
// (Blk_MN). CUTLASS grouped GEMM reads each group's A-side SF starting at ptr_SFA[g] with its
// own tile_atom_to_shape_SFA(M_g,K) layout, so every group's rows MUST start on a 128-row
// boundary in the packed activation buffer. We therefore GATHER each expert's tokens to a
// 128-padded row offset (padded_offsets[e], a multiple of 128) and pack the whole padded buffer
// with ONE tile_atom_to_shape_SFA(padded_total,K). Because tile_to_shape(..., Step<_2,_1>) makes
// the M-tile the outer (slowest) dimension, M-tile t begins at exactly t*per_Mtile_sf elements,
// so ptr_SFA[e] = SFA_base + (padded_offsets[e]/128)*per_Mtile_sf indexes group e's SF exactly,
// and the per-group tile_atom_to_shape_SFA(count[e],K) view reads its real rows [0,count[e]).
// Getting this offset/stride wrong reads scrambled scales — the standalone self-check below
// exercises a multi-expert problem list against the same double-precision oracle to catch it.
// ============================================================================================

static int grouped_sm_count(){
  static int sc = -1;
  if (sc < 0){ int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); sc = v > 0 ? v : 1; }
  return sc;
}

// Physical SF-A element count for exactly one 128-row M-tile at inner dim K (host + device).
static __host__ __device__ long grouped_per_mtile_sfA(int K){
  return (long)cute::size(cute::filter_zeros(GSm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(128, 0, K, 1))));
}

// One thread per expert e in [0,n_total). Builds every device array the grouped GEMM needs for
// ONE logical matmul of constant (N,K) across groups (M_g = counts[e]). ptr_A/ptr_SFA index the
// shared packed-activation buffer at the group's 128-padded row; ptr_B/ptr_SFB index the model
// weight block for expert e; ptr_D indexes the padded output at the same padded row.
__global__ static void g_build_arrays(
    GProbElem *prob,
    const GElemA **ptrA, GStrideA *dA, const GElemSF **ptrSFA, GLayoutSFA *lSFA,
    const GElemB **ptrB, GStrideB *dB, const GElemSF **ptrSFB, GLayoutSFB *lSFB,
    const GElemC **ptrC, GStrideC *dC, GElemD **ptrD, GStrideD *dD,
    const uint32_t *counts, const uint32_t *padded_off,
    const uint8_t *A_data, const uint8_t *A_sf, long per_mtile_sfA,
    const uint8_t *B_base, uint64_t B_stride, uint64_t B_data_bytes,
    GElemD *D_base, int N, int K, int n_total){
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_total) return;
  int M = (int)counts[e];
  uint32_t roff = padded_off[e];
  prob[e] = make_shape(M, N, K);
  dA[e] = cutlass::make_cute_packed_stride(GStrideA{}, cute::make_shape(M, K, 1));
  dB[e] = cutlass::make_cute_packed_stride(GStrideB{}, cute::make_shape(N, K, 1));
  dC[e] = cutlass::make_cute_packed_stride(GStrideC{}, cute::make_shape(M, N, 1));
  dD[e] = cutlass::make_cute_packed_stride(GStrideD{}, cute::make_shape(M, N, 1));
  lSFA[e] = GSm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  lSFB[e] = GSm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
  ptrA[e]   = reinterpret_cast<const GElemA*>(A_data + (size_t)roff * K);   /* E4M3 A: K bytes/row */
  ptrSFA[e] = reinterpret_cast<const GElemSF*>(A_sf + (size_t)(roff / 128u) * per_mtile_sfA);
  ptrB[e]   = reinterpret_cast<const GElemB*>(B_base + (size_t)e * B_stride);
  ptrSFB[e] = reinterpret_cast<const GElemSF*>(B_base + (size_t)e * B_stride + B_data_bytes);
  ptrC[e]   = nullptr;                                    // beta = 0, C unused
  ptrD[e]   = D_base + (size_t)roff * N;
}

/** Per-group device arrays for one logical GEMM shape in a grouped launch.
 *
 * A grouped GEMM runs many independent problems in one kernel, so every
 * operand becomes an ARRAY indexed by group: one problem shape, one pointer,
 * and one stride or scale-factor layout per group. All of these live in device
 * memory and are filled by a setup kernel.
 */
struct GArrays {
  GProbElem   *prob;      ///< per-group problem shape (M, N, K)
  const GElemA **ptrA;    ///< per-group A operand (packed activations)
  GStrideA *dA;           ///< per-group A stride
  const GElemB **ptrB;    ///< per-group B operand (expert weights)
  GStrideB *dB;           ///< per-group B stride
  const GElemSF **ptrSFA; ///< per-group A scale factors
  GLayoutSFA *lSFA;       ///< per-group A scale-factor layout
  const GElemSF **ptrSFB; ///< per-group B scale factors
  GLayoutSFB *lSFB;       ///< per-group B scale-factor layout
  const GElemC **ptrC;    ///< per-group C operand (accumulator source)
  GStrideC *dC;           ///< per-group C stride
  GElemD      **ptrD;     ///< per-group D operand (output)
  GStrideD *dD;           ///< per-group D stride
};

static size_t g_arrays_bytes(int n_total){
  size_t a = 256, off = 0;
  auto add = [&](size_t n){ size_t r = off; off = align_up_bytes(off + n, a); (void)r; return r; };
  add(sizeof(GProbElem)*n_total);
  add(sizeof(GElemA*)*n_total); add(sizeof(GStrideA)*n_total);
  add(sizeof(GElemB*)*n_total); add(sizeof(GStrideB)*n_total);
  add(sizeof(GElemSF*)*n_total); add(sizeof(GLayoutSFA)*n_total);
  add(sizeof(GElemSF*)*n_total); add(sizeof(GLayoutSFB)*n_total);
  add(sizeof(GElemC*)*n_total); add(sizeof(GStrideC)*n_total);
  add(sizeof(GElemD*)*n_total); add(sizeof(GStrideD)*n_total);
  return off;
}

static GArrays g_arrays_place(uint8_t *base, int n_total){
  GArrays g{}; size_t a = 256, off = 0;
  auto add = [&](size_t n)->uint8_t*{ uint8_t *p = base + off; off = align_up_bytes(off + n, a); return p; };
  g.prob   = (GProbElem*)  add(sizeof(GProbElem)*n_total);
  g.ptrA   = (const GElemA**)add(sizeof(GElemA*)*n_total); g.dA = (GStrideA*)add(sizeof(GStrideA)*n_total);
  g.ptrB   = (const GElemB**)add(sizeof(GElemB*)*n_total); g.dB = (GStrideB*)add(sizeof(GStrideB)*n_total);
  g.ptrSFA = (const GElemSF**)add(sizeof(GElemSF*)*n_total); g.lSFA = (GLayoutSFA*)add(sizeof(GLayoutSFA)*n_total);
  g.ptrSFB = (const GElemSF**)add(sizeof(GElemSF*)*n_total); g.lSFB = (GLayoutSFB*)add(sizeof(GLayoutSFB)*n_total);
  g.ptrC   = (const GElemC**)add(sizeof(GElemC*)*n_total); g.dC = (GStrideC*)add(sizeof(GStrideC)*n_total);
  g.ptrD   = (GElemD**)     add(sizeof(GElemD*)*n_total); g.dD = (GStrideD*)add(sizeof(GStrideD)*n_total);
  return g;
}

static size_t grouped_gemm_workspace_bytes(int n_total, int sm_count){
  cutlass::KernelHardwareInfo hw; hw.device_id = 0; hw.sm_count = sm_count;
  typename GGemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
  args.problem_shape = {n_total, nullptr, nullptr};
  args.hw_info = hw;
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta  = 0.0f;
  return GGemm::get_workspace_size(args);
}

// Launch one grouped blockscaled MXFP4 GEMM over n_total groups (inactive experts carry M=0 and
// contribute zero M-tiles). Deterministic per problem list; caller owns ordering.
static int run_grouped_gemm(int n_total, const GArrays &g, void *workspace, int sm_count){
  cutlass::KernelHardwareInfo hw; hw.device_id = 0; hw.sm_count = sm_count;
  typename GGemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {n_total, g.prob, nullptr},
    {g.ptrA, g.dA, g.ptrB, g.dB, g.ptrSFA, g.lSFA, g.ptrSFB, g.lSFB},
    {{}, g.ptrC, g.dC, g.ptrD, g.dD},
    hw};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta  = 0.0f;
  GGemm gemm;
  /* Each failure is named (L189): the callers fold every non-zero into "grouped
   * GEMM failed", and an unchecked launch error used to surface under the NEXT
   * kernel's cuda_ok. */
  cutlass::Status st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) {
    fprintf(stderr, "pulsar: grouped CUTLASS GEMM can_implement refused (n_groups=%d): %s\n",
            n_total, cutlass::cutlassGetStatusString(st));
    return 1;
  }
  st = gemm.initialize(args, workspace);
  if (st != cutlass::Status::kSuccess) {
    fprintf(stderr, "pulsar: grouped CUTLASS GEMM initialize failed (n_groups=%d): %s\n",
            n_total, cutlass::cutlassGetStatusString(st));
    return 2;
  }
  st = gemm.run();
  if (st != cutlass::Status::kSuccess) {
    fprintf(stderr, "pulsar: grouped CUTLASS GEMM run failed (n_groups=%d): %s\n",
            n_total, cutlass::cutlassGetStatusString(st));
    return 3;
  }
  return cuda_ok(cudaGetLastError(), "grouped CUTLASS GEMM launch") ? 0 : 4;
}

/** Byte offsets carving one scratch allocation for the GROUPED (MoE) FFN.
 *
 * Same contract as the per-expert scratch layout had (deleted with the per-expert loop, L158), with two additions:
 * buffers are sized to `padded_total` rows rather than the token count, since
 * each expert's rows are padded up to a tile boundary, and the per-group
 * ::GArrays tables need space of their own.
 *
 * @warning Sizing and execution both go through grouped_scratch_layout(); that
 * shared function is the only thing keeping them consistent.
 */
struct pulsar_grouped_scratch_layout {
  size_t xA_off,       ///< packed E4M3 activations, padded_total rows
         xSF_off,      ///< scale factors for xA
         gate_off,     ///< gate GEMM float output
         up_off,       ///< up GEMM float output
         midA_off,     ///< packed E4M3 SwiGLU product, input to the down GEMM
         midSF_off;    ///< scale factors for midA
  size_t gu_arr_off,   ///< GArrays table for the fused gate/up grouped GEMM
         dn_arr_off,   ///< GArrays table for the down grouped GEMM
         ws_gu_off,    ///< CUTLASS workspace for the gate/up grouped GEMM
         ws_dn_off;    ///< CUTLASS workspace for the down grouped GEMM
  size_t xSF_bytes,    ///< size of the xA scale-factor table
         midSF_bytes,  ///< size of the midA scale-factor table
         ws_bytes,     ///< size of one grouped workspace (both are equal)
         total_bytes;  ///< allocation size the whole layout requires
};

static pulsar_grouped_scratch_layout grouped_scratch_layout(int padded_total, int n_total,
                                                         int in_dim, int mid_dim, int out_dim){
  pulsar_grouped_scratch_layout L{};
  const size_t a = 256; size_t off = 0;
  int sm = grouped_sm_count();
  int tiles = padded_total / 128;
  L.xA_off = off;  off = align_up_bytes(off + (size_t)padded_total*in_dim, a);   /* E4M3: 1 byte/elem */
  L.xSF_bytes = (size_t)tiles * grouped_per_mtile_sfA(in_dim) * sizeof(ElementSF);
  L.xSF_off = off; off = align_up_bytes(off + L.xSF_bytes, a);
  L.gate_off = off; off = align_up_bytes(off + (size_t)padded_total*mid_dim*sizeof(float), a);
  L.up_off   = off; off = align_up_bytes(off + (size_t)padded_total*mid_dim*sizeof(float), a);
  L.midA_off = off; off = align_up_bytes(off + (size_t)padded_total*mid_dim, a);   /* E4M3: 1 byte/elem */
  L.midSF_bytes = (size_t)tiles * grouped_per_mtile_sfA(mid_dim) * sizeof(ElementSF);
  L.midSF_off = off; off = align_up_bytes(off + L.midSF_bytes, a);
  L.gu_arr_off = off; off = align_up_bytes(off + g_arrays_bytes(n_total), a);
  L.dn_arr_off = off; off = align_up_bytes(off + g_arrays_bytes(n_total), a);
  L.ws_bytes = grouped_gemm_workspace_bytes(n_total, sm);
  L.ws_gu_off = off; off = align_up_bytes(off + L.ws_bytes, a);
  L.ws_dn_off = off; off = align_up_bytes(off + L.ws_bytes, a);
  L.total_bytes = off;
  return L;
}

size_t pulsar_cutlass_grouped_moe_scratch_bytes(
    int padded_total, int n_total_expert, int in_dim, int mid_dim, int out_dim){
  return grouped_scratch_layout(padded_total, n_total_expert, in_dim, mid_dim, out_dim).total_bytes;
}

// Grouped MXFP4 FFN. x_gathered/w_gathered are the per-slot activations gathered to 128-padded
// row offsets (padding rows must be pre-zeroed by the caller). Writes ffn_out[padded_total,out_dim]
// (the caller scatters the real rows into the flat down buffer, then moe_sum reduces). No host sync.
int pulsar_cutlass_grouped_moe(
    float *ffn_out, const float *x_gathered, const float *w_gathered,
    const uint8_t *gate_w, const uint8_t *up_w, const uint8_t *down_w,
    uint64_t gate_stride, uint64_t gate_data_bytes,
    uint64_t down_stride, uint64_t down_data_bytes,
    float clamp, int n_total_expert,
    int in_dim, int mid_dim, int out_dim,
    const uint32_t *counts, const uint32_t *padded_offsets, int padded_total,
    uint8_t *scratch, size_t scratch_bytes,
    const void *act_q, const void *act_sf, int act_kbp, const int32_t *row_src_tok){
  pulsar_grouped_scratch_layout L = grouped_scratch_layout(padded_total, n_total_expert, in_dim, mid_dim, out_dim);
  if (!scratch || scratch_bytes < L.total_bytes) return -1;
  int sm = grouped_sm_count();
  uint8_t   *xA   = scratch + L.xA_off;
  ElementSF *xSF  = reinterpret_cast<ElementSF*>(scratch + L.xSF_off);
  float     *gate = reinterpret_cast<float*>(scratch + L.gate_off);
  float     *up   = reinterpret_cast<float*>(scratch + L.up_off);
  uint8_t   *midA = scratch + L.midA_off;
  ElementSF *midSF= reinterpret_cast<ElementSF*>(scratch + L.midSF_off);
  GArrays gu = g_arrays_place(scratch + L.gu_arr_off, n_total_expert);
  GArrays dn = g_arrays_place(scratch + L.dn_arr_off, n_total_expert);
  void *ws_gu = L.ws_bytes ? (void*)(scratch + L.ws_gu_off) : nullptr;
  void *ws_dn = L.ws_bytes ? (void*)(scratch + L.ws_dn_off) : nullptr;

  /* A missed zeroing here reads stale swizzle slots in the GEMM (see the E4M3
   * slot note in pulsar_cuda_matmul.cu) -- so the memsets are checked (L189). */
  if (!cuda_ok(cudaMemsetAsync(xSF, 0, L.xSF_bytes), "grouped MoE xSF memset")) return 3;
  if (!cuda_ok(cudaMemsetAsync(midSF, 0, L.midSF_bytes), "grouped MoE midSF memset")) return 3;

  // Fill the whole padded A operand once (one global SF layout; per-group slices below).
  // The A operand is the E4M3 the producing norm emitted, permuted into the
  // padded layout; without a producer encoding the call refuses.
  if (!(act_q && act_sf && row_src_tok)) {
    static int said = 0;
    if (!said) { said = 1; fprintf(stderr, "pulsar: grouped MoE: no producer E4M3 for x (in_dim=%d padded=%d; act_q=%d act_sf=%d row_src=%d) -- refusing\n", in_dim, padded_total, act_q != nullptr, act_sf != nullptr, row_src_tok != nullptr); }
    return -2;
  }
  (void)x_gathered;
  if (gather_activation_e4m3(xA, xSF, act_q, act_sf, act_kbp, row_src_tok, padded_total, in_dim) != 0) return 3;

  const int bt = 128, bb = (n_total_expert + bt - 1) / bt;
  long pmt_in  = grouped_per_mtile_sfA(in_dim);
  long pmt_mid = grouped_per_mtile_sfA(mid_dim);

  // gate arrays (D = gate) and up arrays (share A/SFA, D = up) — built as two calls.
  g_build_arrays<<<bb,bt>>>(gu.prob, gu.ptrA,gu.dA,gu.ptrSFA,gu.lSFA, gu.ptrB,gu.dB,gu.ptrSFB,gu.lSFB,
      gu.ptrC,gu.dC,gu.ptrD,gu.dD, counts,padded_offsets, xA,(const uint8_t*)xSF,pmt_in,
      gate_w,gate_stride,gate_data_bytes, gate, mid_dim, in_dim, n_total_expert);
  if (!cuda_ok(cudaGetLastError(), "grouped MoE g_build_arrays (gate)")) return 3;
  if (run_grouped_gemm(n_total_expert, gu, ws_gu, sm) != 0) return 3;

  g_build_arrays<<<bb,bt>>>(gu.prob, gu.ptrA,gu.dA,gu.ptrSFA,gu.lSFA, gu.ptrB,gu.dB,gu.ptrSFB,gu.lSFB,
      gu.ptrC,gu.dC,gu.ptrD,gu.dD, counts,padded_offsets, xA,(const uint8_t*)xSF,pmt_in,
      up_w,gate_stride,gate_data_bytes, up, mid_dim, in_dim, n_total_expert);
  if (!cuda_ok(cudaGetLastError(), "grouped MoE g_build_arrays (up)")) return 3;
  if (run_grouped_gemm(n_total_expert, gu, ws_gu, sm) != 0) return 3;

  if (swiglu_pack_activation(midA, midSF, gate, up, w_gathered, clamp, padded_total, mid_dim) != 0) return 3;
  g_build_arrays<<<bb,bt>>>(dn.prob, dn.ptrA,dn.dA,dn.ptrSFA,dn.lSFA, dn.ptrB,dn.dB,dn.ptrSFB,dn.lSFB,
      dn.ptrC,dn.dC,dn.ptrD,dn.dD, counts,padded_offsets, midA,(const uint8_t*)midSF,pmt_mid,
      down_w,down_stride,down_data_bytes, ffn_out, out_dim, mid_dim, n_total_expert);
  if (!cuda_ok(cudaGetLastError(), "grouped MoE g_build_arrays (down)")) return 3;
  if (run_grouped_gemm(n_total_expert, dn, ws_dn, sm) != 0) return 3;
  return 0;
}

// ---- Grouped SINGLE-projection W4A8 GEMM (for MIXED type-40 + iq2/q2k layers). ----
// One device-built ptr-array grouped GEMM over the 128-padded gathered activations, for ONE
// logical matmul out[padded_total,out_dim] = x_gathered . W^T (W = type-40 expert weight, data at
// W_base+e*W_stride, swizzled SFB at +W_data_bytes). This is exactly the gate/up/down sub-step of
// pulsar_cutlass_grouped_moe factored out: same pack + same g_build_arrays gather order + same
// run_grouped_gemm, so it is bit-identical to the per-expert single-projection path it replaces,
// with NO host readback and NO per-expert sync. Padding rows (padded_offsets leave <128-row gaps)
// must be pre-zeroed by the caller. `ws` is folded into scratch (shape-independent grouped WS). ----
static size_t grouped_proj_layout(int padded_total, int n_total_expert, int in_dim,
                                  size_t *xA_off, size_t *xSF_off, size_t *xSF_bytes,
                                  size_t *arr_off, size_t *ws_off, size_t *ws_bytes) {
  const size_t a = 256; size_t off = 0;
  int tiles = padded_total / 128;
  *xA_off = off;  off = align_up_bytes(off + (size_t)padded_total * in_dim, a);   /* E4M3: 1 byte/elem */
  *xSF_bytes = (size_t)tiles * grouped_per_mtile_sfA(in_dim) * sizeof(ElementSF);
  *xSF_off = off; off = align_up_bytes(off + *xSF_bytes, a);
  *arr_off = off; off = align_up_bytes(off + g_arrays_bytes(n_total_expert), a);
  *ws_bytes = grouped_gemm_workspace_bytes(n_total_expert, grouped_sm_count());
  *ws_off = off;  off = align_up_bytes(off + *ws_bytes, a);
  return off;
}
size_t pulsar_cutlass_grouped_proj_scratch_bytes(int padded_total, int n_total_expert, int in_dim, int out_dim){
  (void)out_dim;   /* grouped WS is shape-independent; D buffer is caller-owned */
  size_t a,b,c,d,e,f; return grouped_proj_layout(padded_total, n_total_expert, in_dim, &a,&b,&c,&d,&e,&f);
}
int pulsar_cutlass_grouped_proj(
    float *out, const float *x_gathered,
    const uint8_t *W_base, uint64_t W_stride, uint64_t W_data_bytes,
    int n_total_expert, int in_dim, int out_dim,
    const uint32_t *counts, const uint32_t *padded_offsets, int padded_total,
    uint8_t *scratch, size_t scratch_bytes, int reuse_packed_a,
    const void *act_q, const void *act_sf, int act_kbp, const int32_t *row_src_tok){
  size_t xA_off,xSF_off,xSF_bytes,arr_off,ws_off,ws_bytes;
  size_t need = grouped_proj_layout(padded_total, n_total_expert, in_dim,
                                    &xA_off,&xSF_off,&xSF_bytes,&arr_off,&ws_off,&ws_bytes);
  if (!scratch || scratch_bytes < need) return -1;
  int sm = grouped_sm_count();
  uint8_t   *xA  = scratch + xA_off;
  ElementSF *xSF = reinterpret_cast<ElementSF*>(scratch + xSF_off);
  GArrays g = g_arrays_place(scratch + arr_off, n_total_expert);
  void *ws = ws_bytes ? (void*)(scratch + ws_off) : nullptr;
  /* The gate/up pair runs this GEMM twice on ONE gathered activation; the
   * layout above is a pure function of (padded_total, in_dim), so the second
   * call finds the first call's xA/xSF at the same offsets and skips the
   * whole encode (it used to pack the identical values twice). g_build_arrays
   * only READS xA/xSF, so nothing between the calls can have touched them. */
  if (!reuse_packed_a) {
    if (!cuda_ok(cudaMemsetAsync(xSF, 0, xSF_bytes), "grouped projection xSF memset")) return 3;
    /* Producer handover (L089): when the norm already emitted these rows as
     * E4M3, permute THOSE bytes into the CUTLASS layout instead of gathering
     * f32 and re-encoding it.  Identical shape to pulsar_cutlass_grouped_moe's
     * entry -- this is the one that the mixed-type (gate type != down type)
     * layers lacked, which is why they alone kept gathering raw f32 and were
     * the last tier holding the ffn f32 store on.  The transcode rides the row
     * permutation that has to happen regardless, so it costs ~nothing. */
    if (!(act_q && act_sf && row_src_tok)) {
      static int said = 0;
      if (!said) { said = 1; fprintf(stderr, "pulsar: grouped projection: no producer E4M3 for the activation (in_dim=%d padded=%d) -- refusing\n", in_dim, padded_total); }
      return -2;
    }
    (void)x_gathered;
    if (gather_activation_e4m3(xA, xSF, act_q, act_sf, act_kbp, row_src_tok, padded_total, in_dim) != 0) return 3;
  }
  long pmt = grouped_per_mtile_sfA(in_dim);
  const int bt = 128, bb = (n_total_expert + bt - 1) / bt;
  g_build_arrays<<<bb,bt>>>(g.prob, g.ptrA,g.dA,g.ptrSFA,g.lSFA, g.ptrB,g.dB,g.ptrSFB,g.lSFB,
      g.ptrC,g.dC,g.ptrD,g.dD, counts, padded_offsets, xA,(const uint8_t*)xSF, pmt,
      W_base, W_stride, W_data_bytes, out, out_dim, in_dim, n_total_expert);
  if (!cuda_ok(cudaGetLastError(), "grouped projection g_build_arrays")) return 3;
  return run_grouped_gemm(n_total_expert, g, ws, sm) == 0 ? 0 : 3;
}

// ---- Small-batch expert FFN via direct fp4 GEMV (spec-decode verify, n_tokens 2..4). ----
// The grouped CUTLASS path costs ~2.8 ms per rich layer at n_tokens=3: per-expert GEMM
// launches at M<=3 run far off roofline, behind a blocking per-layer offsets readback.
// These GEMVs read the packed weights directly: one launch for gate+up+swiglu, one for
// down, no readback, no sort.  Activations are the producer's E4M3 + swizzled E8M0 -- the
// SAME operand format the GEMM path reads, so the two paths differ in launch shape, not
// operand numerics.
// Data layout (the converter's pack step): B is ColumnMajor packed E2M1 -- logical
// (n,k) lives at nibble n + k*N, so byte (n + k*N)/2. A thread owning row-pair
// (2p, 2p+1) owns whole bytes, and a warp reads 32 consecutive bytes at each k ->
// coalesced. SF is the swizzled tile-atom layout, indexed with the same layout object
// the packers use (callable on device).

__device__ __constant__ static float kE2M1_GEMV[16] =
    {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
     -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

__device__ __forceinline__ static float gemv_sf_val(uint8_t b) {
  return __int_as_float((uint32_t)b << 23);   /* 2^(e-127) */
}

// v3: the CUTLASS B data section is ROW-MAJOR K-contiguous packed nibbles --
// verified empirically (temp/fp4gemv_test.cu): the converter's pack step's data
// section is an IDENTITY COPY of the source's row-major e2m1 bytes, i.e. logical
// (n,k) lives at nibble k + n*K, byte n*(K/2) + k/2. (pack_weight_f32's manual
// "n + k*N" math -- and its byte-for-byte comment -- does NOT match pack_source;
// v1/v2 of these kernels inherited that assumption and read scrambled weights on
// real models while passing a self-consistent synthetic test.) Row-major K lets
// each warp own one output row with lanes reading consecutive uint32s (8 nibbles)
// along k -- fully coalesced, no k-split, no partial buffers: one launch for
// gate+up+swiglu, one for down. SF stays in the swizzled tile-atom layout,
// indexed with the same layout object the packers use (SF sections of both
// packers agree byte-for-byte).

/* One CTA = one slot x 32 consecutive output columns (one MX block of mid).
 * 8 warps; warp w owns columns n0+w, n0+w+8, n0+w+16, n0+w+24 in turn.  Within
 * a column the lane->k mapping (lane l takes k = l*8 + 256*t, 8 at a time) and
 * the xor-shuffle tree are the dot product; the column loop only sequences
 * them, so a column's f32 result does not depend on which columns share its
 * CTA.
 *
 * Reads the activation as the E4M3 + swizzled-E8M0 pair the producing norm
 * emitted: (float)e4m3 * s with s the block's shared scale.  A lane's 8
 * consecutive k start at a multiple of 8 and so never straddle a 32-element
 * block, which is what lets one scale serve the inner unroll.
 *
 * Epilogue, chosen by the consumer's format:
 *   EMIT_E4M3 -- the 32 SwiGLU results are deposited in shared memory and
 *     warp 0 encodes the block with pulsar_mx_emit_block into (midq, midsf) at
 *     pulsar_mx_sfoff(slot, n0/32, mid_kbp): the small FFN's down GEMV reads
 *     that pair.  Requires N % 32 == 0 (every CTA is a whole block).
 *   !EMIT_E4M3 -- lane 0 stores the f32 to mid[slot][n]; the MoE stage encodes
 *     it (pulsar_cutlass_gemv_gateup).  Tolerates N % 8 == 0. */
template <class SFL, bool EMIT_E4M3>
__global__ static void expert_gemv_gu_swiglu_kernel(
    float *mid,               // [n_slots, N] f32 out            (!EMIT_E4M3)
    uint8_t *midq,            // [n_slots, N] E4M3 out            (EMIT_E4M3)
    uint8_t *midsf,           // ue8m0 plane, pulsar_mx_sfoff     (EMIT_E4M3)
    int mid_kbp,              // its blocks-per-row pitch         (EMIT_E4M3)
    const __nv_fp8_e4m3 *xq8, // [n_tokens, K] E4M3 activations
    const uint8_t *xsf, int xkbp,
    const int32_t *sel,       // [n_slots] expert ids
    const float *rw,          // [n_slots] routing weights
    const uint8_t *gate_base, const uint8_t *up_base,
    uint64_t stride, uint64_t data_bytes, SFL sfl, float clampv,
    int n_expert, unsigned n_total, int K, int N) {
  __shared__ float lut[16];
  __shared__ float vblk[32];
  if (threadIdx.x < 16) lut[threadIdx.x] = kE2M1_GEMV[threadIdx.x];
  __syncthreads();
  const int slot = (int)blockIdx.y;
  const int lane = (int)(threadIdx.x & 31u);
  const int warp = (int)(threadIdx.x >> 5);
  const int n0 = (int)(blockIdx.x * 32u);
  const int e = sel[slot];
  const bool valid = !(e < 0 || (unsigned)e >= n_total);
  const uint8_t *ge = valid ? gate_base + (size_t)e * stride : nullptr;
  const uint8_t *ue = valid ? up_base + (size_t)e * stride : nullptr;
  const int xrow = slot / n_expert;
  const __nv_fp8_e4m3 *xt8 = xq8 + (size_t)xrow * K;
  for (int i = 0; i < 4; i++) {
    const int n = n0 + i * 8 + warp;
    if (n >= N) break;                 /* N % 8 == 0: uniform across the CTA */
    float v = 0.f;                     /* an invalid expert contributes zero */
    if (valid) {
      const uint8_t *gd = ge + (size_t)n * (K / 2);
      const uint8_t *ud = ue + (size_t)n * (K / 2);
      const uint8_t *gsf = ge + data_bytes;
      const uint8_t *usf = ue + data_bytes;
      float g = 0.f, u = 0.f;
      for (int k0 = lane * 8; k0 < K; k0 += 32 * 8) {
        const uint32_t wg = *(const uint32_t *)(gd + (k0 >> 1));
        const uint32_t wu = *(const uint32_t *)(ud + (k0 >> 1));
        const float sg = gemv_sf_val(gsf[sfl(n, k0 & ~31, 0)]);
        const float su = gemv_sf_val(usf[sfl(n, k0 & ~31, 0)]);
        const float sa = gemv_sf_val(xsf[pulsar_mx_sfoff(xrow, k0 >> 5, xkbp)]);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
          const float xv = __half2float((__half)xt8[k0 + j]) * sa;
          g += lut[(wg >> (4 * j)) & 0xFu] * sg * xv;
          u += lut[(wu >> (4 * j)) & 0xFu] * su * xv;
        }
      }
      for (int sh = 16; sh > 0; sh >>= 1) {
        g += __shfl_xor_sync(0xffffffffu, g, sh);
        u += __shfl_xor_sync(0xffffffffu, u, sh);
      }
      /* swiglu identical to swiglu_kernel above (clamp then silu(gate)*up*rweight) */
      if (clampv > 1.0e-6f) {
        if (g > clampv) g = clampv;
        if (u > clampv) u = clampv;
        if (u < -clampv) u = -clampv;
      }
      v = (g / (1.f + expf(-g))) * u * rw[slot];
    }
    if constexpr (EMIT_E4M3) {
      if (lane == 0) vblk[n - n0] = v;
    } else {
      if (lane == 0) mid[(size_t)slot * N + n] = v;
    }
  }
  if constexpr (EMIT_E4M3) {
    __syncthreads();
    if (warp == 0) {
      pulsar_mx_emit_block(vblk[lane], (uint32_t)(n0 + lane), (uint32_t)slot, (uint32_t)N,
                           mid_kbp, (__nv_fp8_e4m3 *)midq, midsf);
    }
  }
}

/* Reads mid as the E4M3 + swizzled-E8M0 pair the SwiGLU stage emitted
 * (pulsar_gpu_mxfp8_act_cache_encode_f32 in the MoE, or the gate/up epilogue
 * above in the small FFN): 1 byte read per element, the source's own format. */
template <class SFL>
__global__ static void expert_gemv_down_kernel(
    float *down_out,          // [n_slots, N]
    const uint8_t *midq8,     // [n_slots, K] E4M3 mid
    const uint8_t *midsf,     // E8M0 in the VEC32 swizzle (pulsar_mx_sfoff(slot, kb, xkbp))
    int xkbp,                 // blocks-per-row pitch of that swizzle
    const int32_t *sel,       // [n_slots] expert ids
    const uint8_t *down_base,
    uint64_t stride, uint64_t data_bytes, SFL sfl,
    unsigned n_total, int K, int N) {
  __shared__ float lut[16];
  if (threadIdx.x < 16) lut[threadIdx.x] = kE2M1_GEMV[threadIdx.x];
  __syncthreads();
  const int slot = (int)blockIdx.y;
  const int lane = (int)(threadIdx.x & 31u);
  const int n = (int)(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
  if (n >= N) return;
  const int e = sel[slot];
  float *o = down_out + (size_t)slot * N;
  if (e < 0 || (unsigned)e >= n_total) { if (lane == 0) o[n] = 0.f; return; }
  const uint8_t *de = down_base + (size_t)e * stride;
  const uint8_t *dd = de + (size_t)n * (K / 2);
  const uint8_t *dsf = de + data_bytes;
  const uint8_t *xt8 = midq8 + (size_t)slot * K;
  float a = 0.f;
  for (int k0 = lane * 8; k0 < K; k0 += 32 * 8) {
    const uint32_t w = *(const uint32_t *)(dd + (k0 >> 1));
    const float sc = gemv_sf_val(dsf[sfl(n, k0 & ~31, 0)]);
    const float sa = gemv_sf_val(midsf[pulsar_mx_sfoff(slot, k0 >> 5, xkbp)]);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      const float xv = __half2float((__half)*(const __nv_fp8_e4m3 *)&xt8[k0 + j]) * sa;
      a += lut[(w >> (4 * j)) & 0xFu] * sc * xv;
    }
  }
  for (int sh = 16; sh > 0; sh >>= 1) a += __shfl_xor_sync(0xffffffffu, a, sh);
  if (lane == 0) o[n] = a;
}




/* Persistent E4M3 staging for the small FFN's mid, grown on demand and reused
 * across layers/calls. */
/* thread_local: one GPU-submitting thread owns its own scratch; a second
 * decode thread must not share this grow/free/realloc buffer (double-free +
 * silent overwrite). Matches the g_act_slots / DsparkReduceBufs convention. */
static thread_local float *g_fp4_gemv_actbuf = nullptr;
static thread_local size_t g_fp4_gemv_actbuf_floats = 0;

// Small-batch (n_tokens 2..4) rich-expert FFN over the packed CUTLASS weights.
// gate/up GEMV emits mid as E4M3 + ue8m0 from its epilogue; the down GEMV reads that
// pair.  down_out gets one pre-weighted FFN result per (token, slot); the caller sums
// the n_expert slices per token (moe_sum_kernel).
int pulsar_cutlass_expert_ffn_gemv_small(
    float *down_out,
    const int32_t *selected, const float *rweights,
    const uint8_t *gate_w, const uint8_t *up_w, const uint8_t *down_w,
    uint64_t gate_stride, uint64_t gate_data_bytes,
    uint64_t down_stride, uint64_t down_data_bytes,
    float clamp, int n_tokens, int n_expert, unsigned n_total_expert,
    int in_dim, int mid_dim, int out_dim,
    const void *act_q, const void *act_sf, int act_kbp) {
  if (in_dim % 256 || mid_dim % 256 || out_dim % 8) return 1;
  if ((gate_stride & 3u) || (down_stride & 3u)) return 1;   /* uint32 row loads */
  const unsigned n_slots = (unsigned)(n_tokens * n_expert);
  /* Both legs read their scale plane through pulsar_mx_sfoff's swizzle: the
   * gate/up leg from the producer's plane, the down leg from the one the
   * gate/up epilogue writes.  Staging sizes are in BYTES (payload + scales)
   * expressed in the buffer's float units.
   *
   * The swizzled plane is NOT nblk bytes: mx_sfoff tiles it 128 rows x 4 blocks
   * into 512-byte groups, so it is sized ceil(rows/128) * (kbp/4) * 512.  The
   * tiling leaves holes past n_slots rows, and they are never read: the down
   * GEMV indexes (slot < n_slots, kb < mid_dim/32) only, and mid_dim % 256 == 0
   * makes mid_kbp == mid_dim/32, so the epilogue writes every position it
   * reads.  No memset. */
  const int   x_kbp      = pulsar_mx_rup((int)(in_dim / 32u), 4);
  const size_t midq_elems = (size_t)n_slots * mid_dim;
  const int   mid_kbp      = pulsar_mx_rup((int)(mid_dim / 32u), 4);
  const size_t midsf_bytes = pulsar_mx_sf_slab_bytes(n_slots, mid_kbp);
  const size_t midq_bytes = midq_elems + midsf_bytes;
  const size_t need = (midq_bytes + 3u) / 4u;
  if (need > g_fp4_gemv_actbuf_floats) {
    if (g_fp4_gemv_actbuf) { pulsar_gpu_seg_note_device_free(); cudaFree(g_fp4_gemv_actbuf); }
    g_fp4_gemv_actbuf = nullptr;
    if (cudaMalloc(&g_fp4_gemv_actbuf, need * sizeof(float)) != cudaSuccess) {
      g_fp4_gemv_actbuf_floats = 0;
      return 1;
    }
    g_fp4_gemv_actbuf_floats = need;
  }
  uint8_t *midq8 = (uint8_t *)g_fp4_gemv_actbuf;
  uint8_t *midsf = midq8 + midq_elems;
  auto sfl_gu = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, mid_dim, in_dim, 1));
  auto sfl_dn = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, out_dim, mid_dim, 1));
  /* x is the producer's E4M3 + ue8m0 or nothing.  Both sides are the
   * pulsar_mx_sfoff swizzle: the producer writes through pulsar_mx_emit_block,
   * expert_gemv_gu_swiglu_kernel reads through pulsar_mx_sfoff, and the
   * producer's act_kbp must equal the x_kbp computed here -- a swizzled reader
   * handed a linear plane computes a well-formed WRONG answer, so a pitch
   * mismatch refuses like a missing encoding does. */
  if (act_q == nullptr || act_sf == nullptr || act_kbp != x_kbp) {
    static int said = 0;
    if (!said) { said = 1; fprintf(stderr, "pulsar: small-batch FFN GEMV: no producer E4M3 for x (in_dim=%d n_tokens=%d kbp %d vs %d) -- refusing\n", in_dim, n_tokens, act_kbp, x_kbp); }
    return 1;
  }
  static int announced = 0;
  if (!announced) {
    announced = 1;
    fprintf(stderr, "pulsar: fp4 small-batch FFN: gate/up emits E4M3 mid from the epilogue "
                    "(mid_dim=%d n_slots=%u)\n", mid_dim, n_slots);
  }
  {
    dim3 g((unsigned)(mid_dim / 32), n_slots);
    expert_gemv_gu_swiglu_kernel<decltype(sfl_gu), true><<<g, 256>>>(
        nullptr, midq8, midsf, mid_kbp,
        (const __nv_fp8_e4m3 *)act_q, (const uint8_t *)act_sf, x_kbp, selected, rweights,
        gate_w, up_w, gate_stride, gate_data_bytes, sfl_gu, clamp,
        n_expert, n_total_expert, in_dim, mid_dim);
  }
  {
    dim3 g((unsigned)((out_dim + 7) / 8), n_slots);
    expert_gemv_down_kernel<decltype(sfl_dn)><<<g, 256>>>(
        down_out, midq8, midsf, mid_kbp, selected,
        down_w, down_stride, down_data_bytes, sfl_dn,
        n_total_expert, mid_dim, out_dim);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}


/* gate/up W4A8 GEMV -> mid[n_slots,mid_dim] = silu(clamp(gate))*clamp(up)*rw (pair layout). */
int pulsar_cutlass_gemv_gateup(
    float *mid, const int32_t *selected, const float *rweights,
    const uint8_t *gate_w, const uint8_t *up_w, uint64_t gate_stride, uint64_t gate_data_bytes,
    float clamp, int n_tokens, int n_expert, unsigned n_total_expert, int in_dim, int mid_dim,
    const void *act_q, const void *act_sf, int act_kbp) {
  if (in_dim % 256 || mid_dim % 8 || (gate_stride & 3u)) return 1;
  const unsigned n_slots = (unsigned)(n_tokens * n_expert);
  auto sfl_gu = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, mid_dim, in_dim, 1));
  dim3 g((unsigned)((mid_dim + 31) / 32), n_slots);

  /* The activation is the E4M3 the producing norm emitted, or the call refuses. */
  if (!act_q || !act_sf) {
    static int said_gu = 0;
    if (!said_gu) { said_gu = 1; fprintf(stderr, "pulsar: fp4 decode gate/up GEMV: no producer E4M3 for x (in_dim=%d n_tokens=%d) -- refusing\n", in_dim, n_tokens); }
    return 3;
  }
  /* Say it once: the announce is the evidence this lane ran. */
  static int announced_gemv_a8 = 0;
  if (!announced_gemv_a8) {
    announced_gemv_a8 = 1;
    fprintf(stderr, "pulsar: fp4 decode GEMV = producer's E4M3 (no re-encode) "
                    "for in_dim=%d mid_dim=%d\n", in_dim, mid_dim);
  }
  expert_gemv_gu_swiglu_kernel<decltype(sfl_gu), false><<<g, 256>>>(
      mid, nullptr, nullptr, 0,
      (const __nv_fp8_e4m3 *)act_q, (const uint8_t *)act_sf, act_kbp,
      selected, rweights, gate_w, up_w,
      gate_stride, gate_data_bytes, sfl_gu, clamp, n_expert, n_total_expert, in_dim, mid_dim);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
/* down W4A8 GEMV -> down_out[n_slots,out_dim] (pair layout, NO routing weight -- applied at gate/up). */
int pulsar_cutlass_gemv_down(
    float *down_out, const int32_t *selected,
    const uint8_t *down_w, uint64_t down_stride, uint64_t down_data_bytes,
    int n_tokens, int n_expert, unsigned n_total_expert, int mid_dim, int out_dim,
    const void *mid_q, const void *mid_sf, int mid_kbp) {
  if (mid_dim % 256 || out_dim % 8 || (down_stride & 3u)) return 1;
  const unsigned n_slots = (unsigned)(n_tokens * n_expert);
  /* mid arrives as the MoE stage's E4M3 encoding (slot rows = pair slots,
   * VEC32 swizzle), or the call refuses. */
  if (!mid_q || !mid_sf || mid_kbp != pulsar_mx_rup(mid_dim / 32, 4)) {
    static int said = 0;
    if (!said) { said = 1; fprintf(stderr, "pulsar: fp4 down GEMV: no producer E4M3 for mid (mid_dim=%d n_slots=%u) -- refusing\n", mid_dim, n_slots); }
    return 1;
  }
  auto sfl_dn = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, out_dim, mid_dim, 1));
  dim3 g((unsigned)((out_dim + 7) / 8), n_slots);
  expert_gemv_down_kernel<decltype(sfl_dn)><<<g, 256>>>(
      down_out, (const uint8_t *)mid_q, (const uint8_t *)mid_sf, mid_kbp, selected, down_w,
      down_stride, down_data_bytes, sfl_dn, n_total_expert, mid_dim, out_dim);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}


// Pack SOURCE-format MXFP4 (separate E2M1 [N,K/2] row-major + E8M0 [N,K/32]) — exactly as the
// DeepSeek-V4-Flash source stores rich experts — into CUTLASS B layout (ColumnMajor packed E2M1
// data + swizzled SFB). Host-side, lossless (copies nibbles+scale verbatim). This is the permanent
// source->CUTLASS packer; nothing consumes ds4's 17-byte format.

// Physical element count of the swizzled SF tensor for a weight of shape (N=out, K=in).


/* An #ifdef PULSAR_MXFP4_STANDALONE self-check main() lived here until the
 * 2026-08-22 types sweep.  Nothing ever defined the macro, its oracle modelled
 * activations as FP4 (pre-W4A8), and it no longer compiled -- its
 * pulsar_cutlass_grouped_moe call was four arguments short.  A silently absent
 * self-check over exactly the format it was meant to pin is worse than none;
 * the living oracle is temp/fp4gemv_test.cu plus the gate suite. */

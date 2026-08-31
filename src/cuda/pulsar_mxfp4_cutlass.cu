// pulsar_mxfp4_cutlass.cu — CUTLASS MXFP4 tensor-core expert FFN for the ds4 MoE (sm_120f).
// Weights arrive pre-packed in CUTLASS B layout (from the offline converter); activations are
// quantized to E4M3 (+ue8m0 block scales) on-device -- W4A8, matching the source model's
// activation format.  Path: pack/roundtrip(x) -> gate/up GEMM -> SwiGLU -> pack(mid) -> down GEMM.
// (This header claimed "activations packed to MXFP4" from the pre-W4A8 era until 2026-08-22.)
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "pulsar_gpu.h"
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

/* Vectorized twin (2026-07-26): the scalar kernel above is one thread per 32-block
 * doing 32 serial scalar LDG.32 for the amax + 32 scalar STG.8 — issue/latency
 * bound (the roofline named it 444 ms = 6.6% of prefill, the biggest dense/MLA
 * sub-cost). This reads the block as 8x float4 (LDG.128) and writes as 2x int4
 * (STG.128). BIT-EXACT to the scalar kernel: one thread still owns the whole
 * 32-block, the amax folds in the SAME sequential order 0..31 (.x/.y/.z/.w per
 * float4), and each element's `cutlass::float_e4m3_t(v*inv)` encode is byte-for-
 * byte the same conversion on the same float. Only the load/store WIDTH changes.
 * Alignment: per-expert K is a 128-multiple, kb*32 floats = 128 B, so
 * act+m*K+kb*32 is 16-B aligned (float4) and the e4m3 out is 16-B aligned (int4).
 * Also the fallback when K is not a 128-multiple (see can_warp). */
template<class TSFA>
__global__ void pack_act_e4m3_rowmajor_vec(uint8_t *A_data, TSFA tSFA, const float *act, int M, int K){
  int nblk=K/32; long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)M*nblk) return;
  int m=(int)(idx/nblk), kb=(int)(idx%nblk);
  const float4 *x4=reinterpret_cast<const float4*>(act+(size_t)m*K+(size_t)kb*32);
  float4 v[8];
  #pragma unroll
  for(int i=0;i<8;i++) v[i]=x4[i];
  float mx=0.f;
  #pragma unroll
  for(int i=0;i<8;i++){ mx=fmaxf(mx,fabsf(v[i].x)); mx=fmaxf(mx,fabsf(v[i].y));
                        mx=fmaxf(mx,fabsf(v[i].z)); mx=fmaxf(mx,fabsf(v[i].w)); }
  int se=-127; if(mx>0.f){ int e=(int)floorf(log2f(mx)); se=e-7; }
  if(se<-127)se=-127; if(se>127)se=127;
  float inv=exp2f((float)-se);
  cutlass::float_e4m3_t ob[32];
  #pragma unroll
  for(int i=0;i<8;i++){ ob[i*4+0]=cutlass::float_e4m3_t(v[i].x*inv);
                        ob[i*4+1]=cutlass::float_e4m3_t(v[i].y*inv);
                        ob[i*4+2]=cutlass::float_e4m3_t(v[i].z*inv);
                        ob[i*4+3]=cutlass::float_e4m3_t(v[i].w*inv); }
  int4 *outp=reinterpret_cast<int4*>(reinterpret_cast<cutlass::float_e4m3_t*>(A_data)+(size_t)m*K+(size_t)kb*32);
  const int4 *obp=reinterpret_cast<const int4*>(ob);
  outp[0]=obp[0]; outp[1]=obp[1];
  tSFA(m, kb*32, 0)=ElementSF::bitcast((uint8_t)(se+127));
}
/* Warp-per-4-blocks twin (2026-08-08).  The vec kernel above still gives one
 * THREAD the whole 32-block, so a lane reads 128 contiguous bytes while its
 * neighbour starts 128 B away: each of its 8 LDG.128 spans 4 KB across the warp
 * and lands in 32 separate sectors, and every sector gets requested twice (a
 * 32 B sector holds two of the thread's consecutive float4).  Measured on a
 * real prefill: 57344 blocks x 128 threads x 32 elem = 235 M elements, 1.18 GB
 * moved in 7.72 ms = 154 GB/s against ~273 GB/s of bandwidth.
 *
 * Here a WARP owns four consecutive 32-blocks: lane L takes elements 4L..4L+3
 * as one float4, so the warp's load is 512 B of contiguous memory in a single
 * coalesced access, and the store is 128 B the same way.  The amax for a block
 * is then a shuffle across the eight lanes that span it rather than a serial
 * scan.  (Same restructuring as idx_pack_q_kernel in the indexer scorer, which
 * this pattern came from.)
 *
 * STILL BIT-EXACT to both twins.  fmaxf is associative AND commutative on this
 * data, so folding the amax by shuffle instead of in index order 0..31 cannot
 * change it -- unlike a sum, a max reassociates exactly.  NaN does not break
 * that either: fabsf(NaN) is NaN and fmaxf(x, NaN) returns x, so a NaN is
 * skipped in any order.  Each element's encode is the same
 * cutlass::float_e4m3_t(v*inv) on the same float; only the load/store width and
 * the reduction order change.
 *
 * Requires K/32 divisible by 4 (i.e. K a 128-multiple) so a warp's four blocks
 * never straddle a row; pack_activation checks that and falls back if not. */
template<class TSFA>
__global__ void pack_act_e4m3_rowmajor_warp(uint8_t *A_data, TSFA tSFA, const float *act, int M, int K){
  const int nblk = K/32;
  const long total_blk = (long)M*nblk;
  const int lane = threadIdx.x & 31;
  const long grp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;  // group of 4 blocks
  const long blk0 = grp*4;
  if (blk0 >= total_blk) return;
  const int m = (int)(blk0 / nblk), kb0 = (int)(blk0 % nblk);

  const float4 v = reinterpret_cast<const float4*>(act+(size_t)m*K+(size_t)kb0*32)[lane];
  float mx = fmaxf(fmaxf(fabsf(v.x),fabsf(v.y)), fmaxf(fabsf(v.z),fabsf(v.w)));
  // lanes 8b..8b+7 span block b: xor over the low three lane bits reduces within it
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));

  int se=-127; if(mx>0.f){ int e=(int)floorf(log2f(mx)); se=e-7; }
  if(se<-127)se=-127; if(se>127)se=127;
  const float inv=exp2f((float)-se);

  cutlass::float_e4m3_t ob[4];
  ob[0]=cutlass::float_e4m3_t(v.x*inv); ob[1]=cutlass::float_e4m3_t(v.y*inv);
  ob[2]=cutlass::float_e4m3_t(v.z*inv); ob[3]=cutlass::float_e4m3_t(v.w*inv);
  reinterpret_cast<uint32_t*>(reinterpret_cast<cutlass::float_e4m3_t*>(A_data)
                              +(size_t)m*K+(size_t)kb0*32)[lane]
      = *reinterpret_cast<const uint32_t*>(ob);
  if((lane & 7)==0) tSFA(m, (kb0+(lane>>3))*32, 0)=ElementSF::bitcast((uint8_t)(se+127));
}

// LOSSY dequant->fp4 weight packer still needs the E2M1 nearest-value helper (below); keep it.
__device__ __constant__ float d_kE2M1[16] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f, 0.f,-0.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
__device__ __forceinline__ uint8_t d_to_e2m1(float v){ float best=1e30f; uint8_t bn=0; for(uint8_t n=0;n<16;n++){ float d=fabsf(v-d_kE2M1[n]); if(d<best){best=d;bn=n;} } return bn; }
// mid = silu(clamp(gate)) * clamp(up) * routing_weight.
// TWIN of swiglu_kernel in pulsar_cuda_hc_router.cu, and deliberately a
// separate copy: this TU deliberately does not take pulsar_cuda_internal.h
// (it is the one CUTLASS-including TU and keeps its include surface minimal),
// so the two cannot share a definition without coupling them. They MUST agree on the arithmetic above.
// Two differences that are intentional, not drift: the routing weight is a
// per-row array here and a scalar there, and the engine's twin carries an
// E4M3 + E8M0 epilogue that this one does not -- which is why the CUTLASS
// down path re-quantises mid (the grouped/GEMM path via pack_activation, the
// GEMV path via e4m3_act_roundtrip_kernel) instead of reading the producer's
// encoding.
// This cited `pulsar_cuda.cu:10827-10835` until 2026-08-17 to assert the
// agreement. That file does not exist -- the engine was split up long ago --
// so the reference had been unfollowable, and the invariant uncheckable,
// for however long it took anyone to look.
/* ONE definition of the per-element math, shared by the scalar and vector
 * kernels below so the two can never drift.  fminf/fmaxf form, matching the
 * hc_router twin EXACTLY: identical to the if-chain on finite inputs, but
 * agrees on NaN too (fminf(NaN,c)=c, where the if-chain left NaN) -- the
 * twins must not diverge on any input. */
__device__ __forceinline__ static float swiglu_elem(float g, float u, float wv, float clamp){
  if(clamp>1.0e-6f){ g=fminf(g,clamp); u=fminf(fmaxf(u,-clamp),clamp); }
  const float s=g/(1.f+expf(-g));
  return s*u*wv;
}

__global__ void swiglu_kernel(float *mid, const float *gate, const float *up, const float *w, float clamp, int mid_dim, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  mid[i]=swiglu_elem(gate[i], up[i], w[i/mid_dim], clamp);
}

/* Vector-4 twin (L128).  ncu on the decode path measured this kernel at
 * 47.3% SM with a long-scoreboard stall ratio of 13.1 -- occupancy was fine
 * (80%), it was simply waiting on memory: one element per thread means three
 * 4-byte global accesses and a 64-bit divide per output float.  Four elements
 * per thread turns those into 16-byte transactions and amortises the divide.
 * BIT-EXACT by construction: same swiglu_elem per element, elementwise with
 * no reduction, so nothing reassociates -- only the access granularity and
 * the thread mapping change.  Taken only when mid_dim and n are multiples of
 * 4 and all three pointers are 16-byte aligned; otherwise the scalar kernel
 * runs unchanged. */
__global__ void swiglu_v4_kernel(float4 *mid, const float4 *gate, const float4 *up, const float *w, float clamp, int mid_dim_q, long quads){
  long q=(long)blockIdx.x*blockDim.x+threadIdx.x; if(q>=quads) return;
  const float4 g=gate[q], u=up[q];
  const float wv=w[q/mid_dim_q];   /* all four share a pair when mid_dim%4==0 */
  float4 o;
  o.x=swiglu_elem(g.x,u.x,wv,clamp);
  o.y=swiglu_elem(g.y,u.y,wv,clamp);
  o.z=swiglu_elem(g.z,u.z,wv,clamp);
  o.w=swiglu_elem(g.w,u.w,wv,clamp);
  mid[q]=o;
}

/* Dispatch: vector when the shape and alignment allow, scalar otherwise. */
static inline bool swiglu_v4_ok(const void *a, const void *b, const void *c, int mid_dim, long n){
  return (mid_dim % 4)==0 && (n % 4)==0 &&
         ((uintptr_t)a % 16)==0 && ((uintptr_t)b % 16)==0 && ((uintptr_t)c % 16)==0;
}
static inline void swiglu_launch(float *mid, const float *gate, const float *up, const float *w, float clamp, int mid_dim, long n){
  const int t=256;
  if(swiglu_v4_ok(mid,gate,up,mid_dim,n)){
    const long quads=n/4; const long b=(quads+t-1)/t;
    swiglu_v4_kernel<<<(unsigned)b,t>>>((float4*)mid,(const float4*)gate,(const float4*)up,w,clamp,mid_dim/4,quads);
  } else {
    const long b=(n+t-1)/t;
    swiglu_kernel<<<(unsigned)b,t>>>(mid,gate,up,w,clamp,mid_dim,n);
  }
}

/* L129 lever 1 — SwiGLU WITH the E4M3+E8M0 epilogue its hc_router twin has.
 *
 * The comment below this kernel has named the gap for months: the CUTLASS
 * down path re-quantises `mid` precisely because this TU's swiglu lacked the
 * epilogue.  The profile priced it: pack_act_e4m3_rowmajor_warp is 3.5% of
 * decode at an 85 long-scoreboard stall ratio, and it exists only to read
 * back f32 values this kernel just wrote.  Fusing removes the whole
 * round-trip -- one launch and one full pass over `mid` per FFN.
 *
 * Structure is pack_act_e4m3_rowmajor_warp's exactly (one warp = four
 * consecutive 32-blocks of one row, xor-shuffle amax within each block), with
 * the loads replaced by gate/up and swiglu_elem applied first.  BIT-EXACT vs
 * swiglu-then-pack: the packer re-read f32 values that were stored exactly,
 * so the in-register values are the same floats, the same per-block amax over
 * the same set, and the same cutlass::float_e4m3_t(v*inv) encode.
 *
 * `mid_f32` is still written (NULL to skip) because dumps and the non-fused
 * fallback read it; the epilogue is additive. */
template<class TSFA>
__global__ void swiglu_pack_e4m3_warp_kernel(float *mid_f32, uint8_t *A_data, TSFA tSFA,
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
  v.x = swiglu_elem(g.x,u.x,wv,clamp);
  v.y = swiglu_elem(g.y,u.y,wv,clamp);
  v.z = swiglu_elem(g.z,u.z,wv,clamp);
  v.w = swiglu_elem(g.w,u.w,wv,clamp);
  if (mid_f32) reinterpret_cast<float4*>(mid_f32+base)[lane] = v;

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


// A_data is E4M3: M*K bytes (1 byte/elem), NOT M*K/2. SFA via the CUTLASS tile-atom layout.
static void pack_activation(uint8_t *A_data, ElementSF *A_sf, const float *x, int M, int K){
  auto layoutSF = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, 0, K, 1));
  auto tSFA = make_tensor(make_gmem_ptr(A_sf), layoutSF);
  int nb=M*(K/32), t=128, b=(nb+t-1)/t;
  /* The warp kernel gives a warp four consecutive 32-blocks, so they must not
   * straddle a row.  Fall back rather than mis-index if K is ever not a
   * 128-multiple (per-expert K is, but this is the only thing guaranteeing it). */
  const int can_warp = ((K/32) % 4) == 0;
  if (!can_warp) pack_act_e4m3_rowmajor_vec<<<b,t>>>(A_data, tSFA, x, M, K);
  else {
    const long groups = (long)M*(K/32)/4;          /* one warp per group */
    const long thr = groups*32, bw = (thr+t-1)/t;
    pack_act_e4m3_rowmajor_warp<<<(unsigned)bw,t>>>(A_data, tSFA, x, M, K);
  }
}

/* L129 lever 1: swiglu straight into the E4M3 staging the down GEMM reads.
 * Falls back to the unfused pair whenever the warp packer's own shape rule
 * (K a 128-multiple) or 16-byte alignment does not hold, so the fast path
 * adds no constraint the old pair did not already have. */
static void swiglu_pack_activation(float *mid_f32, uint8_t *A_data, ElementSF *A_sf,
                                   const float *gate, const float *up, const float *w,
                                   float clamp, int M, int K){
  const bool shape_ok = ((K/32) % 4) == 0;
  const bool align_ok = ((uintptr_t)gate % 16)==0 && ((uintptr_t)up % 16)==0 &&
                        ((uintptr_t)mid_f32 % 16)==0;
  if (!shape_ok || !align_ok) {
    swiglu_launch(mid_f32, gate, up, w, clamp, K, (long)M*K);
    pack_activation(A_data, A_sf, mid_f32, M, K);
    return;
  }
  auto layoutSF = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, 0, K, 1));
  auto tSFA = make_tensor(make_gmem_ptr(A_sf), layoutSF);
  const int t = 128;
  const long groups = (long)M*(K/32)/4;
  const long thr = groups*32, bw = (thr+t-1)/t;
  /* The f32 mid store is DEAD on this path and is skipped.  Both callers
   * consume only (A_data, A_sf) after this returns -- the down GEMM reads the
   * E4M3 staging -- and `mid` is CUTLASS-local scratch, never registered in
   * the activation cache, so no consumer can look it up and no
   * act_f32_absent_hazard applies.  Same move the hc_router twin and qr_norm
   * already make, with the same announce discipline.  The unfused fallback
   * above still writes it, because pack_activation reads it back. */
  static int announced = 0;
  if (!announced) {
    announced = 1;
    fprintf(stderr, "pulsar: swiglu->E4M3 fused; mid f32 store SKIPPED "
                    "(%.1f MiB/call at M=%d K=%d)\n",
            (double)((size_t)M * K * sizeof(float)) / 1048576.0, M, K);
  }
  swiglu_pack_e4m3_warp_kernel<<<(unsigned)bw,t>>>(nullptr, A_data, tSFA, gate, up, w, clamp, M, K);
}

/* Where the ENGINE's activation cache keeps the E8M0 byte for (row, kb): the
 * 128x4 SF atom swizzle from pulsar_cuda_matmul.cu (mx_sfoff).  Duplicated here
 * rather than exported because it is the on-disk-free layout of a device buffer
 * this TU only READS -- and the two must agree, so it is spelled out identically
 * and asserted by the bit-exactness of the gather against the pack path. */

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

static void gather_activation_e4m3(uint8_t *A_data, ElementSF *A_sf,
                                   const void *src_q, const void *src_sf, int src_kbp,
                                   const int32_t *row_src, int M, int K){
  auto layoutSF = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, 0, K, 1));
  auto tSFA = make_tensor(make_gmem_ptr(A_sf), layoutSF);
  const int nb = M * (K / 32), t = 128, b = (nb + t - 1) / t;
  gather_act_e4m3_kernel<<<b,t>>>(A_data, tSFA, (const uint8_t*)src_q, (const uint8_t*)src_sf,
                                  src_kbp, row_src, M, K);
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

// One MXFP4 GEMM: D[M,N] = A[M,K](act,packed) . B[N,K](weight,packed). A_sf/B_sf = swizzled SF
// buffers. workspace must be at least gemm_workspace_bytes(M,N,K) bytes, caller-owned.
static int run_gemm(float *D, const uint8_t *A_data, const ElementSF *A_sf,
                    const uint8_t *B_data, const ElementSF *B_sf, int M, int N, int K,
                    void *workspace){
  auto args = make_gemm_args(D,A_data,A_sf,B_data,B_sf,M,N,K);
  Gemm gemm;
  /* can_implement is pure host-side argument validation and is deterministic
   * per (M,N,K); the expert loop re-ran it for every expert's every GEMM. Cache
   * validated shapes (single-threaded GPU submission thread; the shape set is
   * tiny -- a few (T,mid,in)/(T,out,mid) combos). */
  {
    static uint64_t seen[64];
    static int n_seen = 0;
    const uint64_t key = ((uint64_t)(uint32_t)M << 42) ^ ((uint64_t)(uint32_t)N << 21) ^ (uint32_t)K;
    bool hit = false;
    for (int i = 0; i < n_seen; i++) if (seen[i] == key) { hit = true; break; }
    if (!hit) {
      if (gemm.can_implement(args)!=cutlass::Status::kSuccess) return 1;
      if (n_seen < 64) seen[n_seen++] = key;
    }
  }
  if (gemm.initialize(args, workspace)!=cutlass::Status::kSuccess) return 2;
  auto st = gemm.run();
  return st==cutlass::Status::kSuccess ? 0 : 3;
}

/** Byte offsets carving ONE scratch allocation into everything the dense FFN
 * needs beyond the weight and activation pointers it is called with: packed
 * activation buffers, their scale-factor tables, the three GEMMs' float
 * outputs, and the three GEMMs' CUTLASS workspaces (usually zero-size, but not
 * guaranteed).
 *
 * @warning The sizing path and the execution path both derive their offsets
 * from cutlass_ffn_scratch_layout(). That single function is what keeps them
 * in lock-step -- there is no runtime check that a caller computing offsets
 * some other way agrees with it, and a disagreement is silent memory
 * corruption rather than a failure.
 */
struct pulsar_cutlass_ffn_scratch_layout {
  size_t xA_off,          ///< packed E4M3 activations (1 byte/elem)
         xSF_off,         ///< scale factors for xA
         midA_off,        ///< packed E4M3 SwiGLU product, input to the down GEMM
         midSF_off,       ///< scale factors for midA
         gate_off,        ///< gate GEMM float output
         up_off,          ///< up GEMM float output
         mid_off;         ///< SwiGLU product, f32, before packing to midA
  size_t ws_gate_off,     ///< CUTLASS workspace for the gate GEMM
         ws_up_off,       ///< CUTLASS workspace for the up GEMM
         ws_down_off;     ///< CUTLASS workspace for the down GEMM
  size_t xSF_n,           ///< scale-factor entries for xA, tile-padded
         midSF_n;         ///< scale-factor entries for midA, tile-padded
  size_t ws_gate_bytes,   ///< size of the gate workspace
         ws_up_bytes,     ///< size of the up workspace (same shape as gate)
         ws_down_bytes;   ///< size of the down workspace
  size_t total_bytes;     ///< allocation size the whole layout requires
};

static size_t align_up_bytes(size_t n, size_t a){ return (n + a - 1) / a * a; }

static pulsar_cutlass_ffn_scratch_layout cutlass_ffn_scratch_layout(int T, int in_dim, int mid_dim, int out_dim){
  pulsar_cutlass_ffn_scratch_layout L{};
  const size_t align = 256;
  size_t off = 0;
  L.xA_off = off; off = align_up_bytes(off + (size_t)T*in_dim, align);   /* E4M3: 1 byte/elem */
  L.xSF_n = (size_t)((T+127)/128*128)*((in_dim/32+3)/4*4);
  L.xSF_off = off; off = align_up_bytes(off + L.xSF_n*sizeof(ElementSF), align);
  L.midA_off = off; off = align_up_bytes(off + (size_t)T*mid_dim, align);  /* E4M3: 1 byte/elem */
  L.midSF_n = (size_t)((T+127)/128*128)*((mid_dim/32+3)/4*4);
  L.midSF_off = off; off = align_up_bytes(off + L.midSF_n*sizeof(ElementSF), align);
  L.gate_off = off; off = align_up_bytes(off + (size_t)T*mid_dim*sizeof(float), align);
  L.up_off   = off; off = align_up_bytes(off + (size_t)T*mid_dim*sizeof(float), align);
  L.mid_off  = off; off = align_up_bytes(off + (size_t)T*mid_dim*sizeof(float), align);
  L.ws_gate_bytes = gemm_workspace_bytes(T, mid_dim, in_dim);
  L.ws_up_bytes   = L.ws_gate_bytes;   // identical (M,N,K) shape to gate
  L.ws_down_bytes = gemm_workspace_bytes(T, out_dim, mid_dim);
  L.ws_gate_off = off; off = align_up_bytes(off + L.ws_gate_bytes, align);
  L.ws_up_off   = off; off = align_up_bytes(off + L.ws_up_bytes, align);
  L.ws_down_off = off; off = align_up_bytes(off + L.ws_down_bytes, align);
  L.total_bytes = off;
  return L;
}

// Scratch bytes pulsar_cutlass_expert_ffn_scratch() needs for the worst case of T tokens routed
// to a single expert. Callers size one buffer for their layer's (in_dim,mid_dim,out_dim) shape
// at T=max_tokens_per_expert once (e.g. via cuda_tmp_alloc) and reuse it across every expert
// and every layer that shares that shape -- this is deliberately NOT malloc'd per call.
size_t pulsar_cutlass_expert_ffn_scratch_bytes(int T, int in_dim, int mid_dim, int out_dim){
  return cutlass_ffn_scratch_layout(T, in_dim, mid_dim, out_dim).total_bytes;
}

// Core FFN, no allocation and no synchronization: out[T,out_dim] = down(swiglu(x.Wg^T, x.Wu^T)).Wd^T.
// Weights pre-packed (data+sf) in B layout; x is f32 [T,in_dim]. `scratch` must be at least
// pulsar_cutlass_expert_ffn_scratch_bytes(T,in_dim,mid_dim,out_dim) bytes, 256-byte aligned.
// Launches into the caller's current stream (legacy default stream); the caller is responsible
// for ordering/synchronizing against its own subsequent work, same as every other kernel in
// this engine's decode/prefill graphs.
// SF weight pointers are typed as raw bytes at this extern "C" boundary (not ElementSF*) so
// callers outside this TU -- e.g. pulsar_cuda_moe.cu, which has no CUTLASS header visibility --
// can call this without depending on CUTLASS types. ElementSF (cutlass::float_ue8m0_t) is a
// 1-byte POD; the reinterpret below is exact.
int pulsar_cutlass_expert_ffn_scratch(
    float *out, const float *x,
    const uint8_t *Wg_d, const uint8_t *Wg_sf,
    const uint8_t *Wu_d, const uint8_t *Wu_sf,
    const uint8_t *Wd_d, const uint8_t *Wd_sf,
    const float *weights, float clamp,
    int T, int in_dim, int mid_dim, int out_dim,
    uint8_t *scratch, size_t scratch_bytes){
  pulsar_cutlass_ffn_scratch_layout L = cutlass_ffn_scratch_layout(T, in_dim, mid_dim, out_dim);
  if (!scratch || scratch_bytes < L.total_bytes) return -1;
  const ElementSF *Wg_sf_e = reinterpret_cast<const ElementSF*>(Wg_sf);
  const ElementSF *Wu_sf_e = reinterpret_cast<const ElementSF*>(Wu_sf);
  const ElementSF *Wd_sf_e = reinterpret_cast<const ElementSF*>(Wd_sf);
  uint8_t *xA = scratch + L.xA_off;
  ElementSF *xSF = reinterpret_cast<ElementSF*>(scratch + L.xSF_off);
  uint8_t *midA = scratch + L.midA_off;
  ElementSF *midSF = reinterpret_cast<ElementSF*>(scratch + L.midSF_off);
  float *gate = reinterpret_cast<float*>(scratch + L.gate_off);
  float *up   = reinterpret_cast<float*>(scratch + L.up_off);
  float *mid  = reinterpret_cast<float*>(scratch + L.mid_off);
  void *ws_gate = L.ws_gate_bytes ? scratch + L.ws_gate_off : nullptr;
  void *ws_up   = L.ws_up_bytes   ? scratch + L.ws_up_off   : nullptr;
  void *ws_down = L.ws_down_bytes ? scratch + L.ws_down_off : nullptr;
  /* Async: blocking cudaMemset serializes the host against ALL in-flight device
   * work -- in the per-expert FFN loop that was 2 full device syncs per expert
   * per rich layer, dominating small-batch verify encode time. The null-stream
   * async memset is still ordered before the pack/GEMM launches below. */
  if (L.xSF_n) cudaMemsetAsync(xSF,0,L.xSF_n*sizeof(ElementSF));
  if (L.midSF_n) cudaMemsetAsync(midSF,0,L.midSF_n*sizeof(ElementSF));
  int rc=0;
  pack_activation(xA,xSF,x,T,in_dim);
  rc|=run_gemm(gate,xA,xSF,Wg_d,Wg_sf_e,T,mid_dim,in_dim,ws_gate);
  rc|=run_gemm(up,  xA,xSF,Wu_d,Wu_sf_e,T,mid_dim,in_dim,ws_up);
  swiglu_pack_activation(mid,midA,midSF,gate,up,weights,clamp,T,mid_dim);
  rc|=run_gemm(out, midA,midSF,Wd_d,Wd_sf_e,T,out_dim,mid_dim,ws_down);
  return rc;
}

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
// ---- extern-C expert FFN (standalone/test convenience): allocates+frees its own scratch and
// synchronizes before returning. The engine never calls this -- it calls the scratch variant
// above with a pre-sized, reused buffer, since a per-call cudaMalloc/cudaFree/cudaDeviceSynchronize
// here is exactly the hot-path cost the engine path exists to avoid. ----
int pulsar_cutlass_expert_ffn(
    float *out, const float *x,
    const uint8_t *Wg_d, const ElementSF *Wg_sf,
    const uint8_t *Wu_d, const ElementSF *Wu_sf,
    const uint8_t *Wd_d, const ElementSF *Wd_sf,
    const float *weights, float clamp,
    int T, int in_dim, int mid_dim, int out_dim){
  size_t scratch_bytes = pulsar_cutlass_expert_ffn_scratch_bytes(T, in_dim, mid_dim, out_dim);
  uint8_t *scratch=nullptr;
  cudaMalloc(&scratch, scratch_bytes);
  int rc = pulsar_cutlass_expert_ffn_scratch(out,x,Wg_d,reinterpret_cast<const uint8_t*>(Wg_sf),
                                          Wu_d,reinterpret_cast<const uint8_t*>(Wu_sf),
                                          Wd_d,reinterpret_cast<const uint8_t*>(Wd_sf),
                                          weights,clamp,
                                          T,in_dim,mid_dim,out_dim,scratch,scratch_bytes);
  cudaDeviceSynchronize();
  cudaFree(scratch);
  return rc;
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
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 1;
  if (gemm.initialize(args, workspace) != cutlass::Status::kSuccess) return 2;
  return gemm.run() == cutlass::Status::kSuccess ? 0 : 3;
}

/** Byte offsets carving one scratch allocation for the GROUPED (MoE) FFN.
 *
 * Same contract as ::pulsar_cutlass_ffn_scratch_layout, with two additions:
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
         mid_off,      ///< SwiGLU product, f32
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
  L.mid_off  = off; off = align_up_bytes(off + (size_t)padded_total*mid_dim*sizeof(float), a);
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
  float     *mid  = reinterpret_cast<float*>(scratch + L.mid_off);
  uint8_t   *midA = scratch + L.midA_off;
  ElementSF *midSF= reinterpret_cast<ElementSF*>(scratch + L.midSF_off);
  GArrays gu = g_arrays_place(scratch + L.gu_arr_off, n_total_expert);
  GArrays dn = g_arrays_place(scratch + L.dn_arr_off, n_total_expert);
  void *ws_gu = L.ws_bytes ? (void*)(scratch + L.ws_gu_off) : nullptr;
  void *ws_dn = L.ws_bytes ? (void*)(scratch + L.ws_dn_off) : nullptr;

  cudaMemsetAsync(xSF, 0, L.xSF_bytes);
  cudaMemsetAsync(midSF, 0, L.midSF_bytes);

  // Fill the whole padded A operand once (one global SF layout; per-group slices below).
  // Preferred: move the E4M3 the producing norm already emitted. Fall back to
  // encoding the gathered f32 when no cached encoding was handed over.
  if (act_q && act_sf && row_src_tok)
    gather_activation_e4m3(xA, xSF, act_q, act_sf, act_kbp, row_src_tok, padded_total, in_dim);
  else
    pack_activation(xA, xSF, x_gathered, padded_total, in_dim);

  const int bt = 128, bb = (n_total_expert + bt - 1) / bt;
  long pmt_in  = grouped_per_mtile_sfA(in_dim);
  long pmt_mid = grouped_per_mtile_sfA(mid_dim);

  // gate arrays (D = gate) and up arrays (share A/SFA, D = up) — built as two calls.
  g_build_arrays<<<bb,bt>>>(gu.prob, gu.ptrA,gu.dA,gu.ptrSFA,gu.lSFA, gu.ptrB,gu.dB,gu.ptrSFB,gu.lSFB,
      gu.ptrC,gu.dC,gu.ptrD,gu.dD, counts,padded_offsets, xA,(const uint8_t*)xSF,pmt_in,
      gate_w,gate_stride,gate_data_bytes, gate, mid_dim, in_dim, n_total_expert);
  if (run_grouped_gemm(n_total_expert, gu, ws_gu, sm) != 0) return 3;

  g_build_arrays<<<bb,bt>>>(gu.prob, gu.ptrA,gu.dA,gu.ptrSFA,gu.lSFA, gu.ptrB,gu.dB,gu.ptrSFB,gu.lSFB,
      gu.ptrC,gu.dC,gu.ptrD,gu.dD, counts,padded_offsets, xA,(const uint8_t*)xSF,pmt_in,
      up_w,gate_stride,gate_data_bytes, up, mid_dim, in_dim, n_total_expert);
  if (run_grouped_gemm(n_total_expert, gu, ws_gu, sm) != 0) return 3;

  swiglu_pack_activation(mid, midA, midSF, gate, up, w_gathered, clamp, padded_total, mid_dim);
  g_build_arrays<<<bb,bt>>>(dn.prob, dn.ptrA,dn.dA,dn.ptrSFA,dn.lSFA, dn.ptrB,dn.dB,dn.ptrSFB,dn.lSFB,
      dn.ptrC,dn.dC,dn.ptrD,dn.dD, counts,padded_offsets, midA,(const uint8_t*)midSF,pmt_mid,
      down_w,down_stride,down_data_bytes, ffn_out, out_dim, mid_dim, n_total_expert);
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
    cudaMemsetAsync(xSF, 0, xSF_bytes);
    /* Producer handover (L089): when the norm already emitted these rows as
     * E4M3, permute THOSE bytes into the CUTLASS layout instead of gathering
     * f32 and re-encoding it.  Identical shape to pulsar_cutlass_grouped_moe's
     * entry -- this is the one that the mixed-type (gate type != down type)
     * layers lacked, which is why they alone kept gathering raw f32 and were
     * the last tier holding the ffn f32 store on.  The transcode rides the row
     * permutation that has to happen regardless, so it costs ~nothing. */
    if (act_q && act_sf && row_src_tok)
      gather_activation_e4m3(xA, xSF, act_q, act_sf, act_kbp, row_src_tok, padded_total, in_dim);
    else
      pack_activation(xA, xSF, x_gathered, padded_total, in_dim);
  }
  long pmt = grouped_per_mtile_sfA(in_dim);
  const int bt = 128, bb = (n_total_expert + bt - 1) / bt;
  g_build_arrays<<<bb,bt>>>(g.prob, g.ptrA,g.dA,g.ptrSFA,g.lSFA, g.ptrB,g.dB,g.ptrSFB,g.lSFB,
      g.ptrC,g.dC,g.ptrD,g.dD, counts, padded_offsets, xA,(const uint8_t*)xSF, pmt,
      W_base, W_stride, W_data_bytes, out, out_dim, in_dim, n_total_expert);
  return run_grouped_gemm(n_total_expert, g, ws, sm) == 0 ? 0 : 3;
}

// ---- Small-batch expert FFN via direct fp4 GEMV (spec-decode verify, n_tokens 2..4). ----
// The grouped CUTLASS path costs ~2.8 ms per rich layer at n_tokens=3: per-expert GEMM
// launches at M<=3 run far off roofline, behind a blocking per-layer offsets readback.
// These GEMVs read the packed weights directly: one launch for gate+up+swiglu, one for
// down, no readback, no sort.  Activations are E4M3 (the A8 arms; the producer's grouped
// encoding when armed, an on-the-spot roundtrip otherwise) -- the SAME format the GEMM
// path quantizes to, so the two paths differ in launch shape, not operand numerics.
// Data layout (see pulsar_cutlass_pack_source): B is ColumnMajor packed E2M1 -- logical
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
// verified empirically (temp/fp4gemv_test.cu): pulsar_cutlass_pack_source's data
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

/* A8=true reads the activation as the E4M3 + swizzled-E8M0 pair the producing
 * norm already emitted, instead of the f32 buffer e4m3_act_roundtrip_kernel
 * builds by deriving that same encoding and immediately throwing the bytes
 * away.  Bit-exact, not merely equivalent: the round-trip writes
 * (float)e4m3(v*inv)*s and this computes (float)e4m3 * s from the same block
 * scale, so the value entering the dot product is identical.  A lane's 8
 * consecutive k start at a multiple of 8 and so never straddle a 32-element
 * block, which is what lets one scale serve the inner unroll. */
template <class SFL, bool A8 = false>
__global__ static void expert_gemv_gu_swiglu_kernel(
    float *mid,               // [n_slots, N]
    const float *xq,          // [n_tokens, K] f32 activations (A8=false)
    const __nv_fp8_e4m3 *xq8, // [n_tokens, K] E4M3 activations (A8=true)
    const uint8_t *xsf, int xkbp,
    const int32_t *sel,       // [n_slots] expert ids
    const float *rw,          // [n_slots] routing weights
    const uint8_t *gate_base, const uint8_t *up_base,
    uint64_t stride, uint64_t data_bytes, SFL sfl, float clampv,
    int n_expert, unsigned n_total, int K, int N) {
  __shared__ float lut[16];
  if (threadIdx.x < 16) lut[threadIdx.x] = kE2M1_GEMV[threadIdx.x];
  __syncthreads();
  const int slot = (int)blockIdx.y;
  const int lane = (int)(threadIdx.x & 31u);
  const int n = (int)(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
  if (n >= N) return;
  const int e = sel[slot];
  float *m = mid + (size_t)slot * N;
  if (e < 0 || (unsigned)e >= n_total) { if (lane == 0) m[n] = 0.f; return; }
  const uint8_t *ge = gate_base + (size_t)e * stride;
  const uint8_t *ue = up_base + (size_t)e * stride;
  const uint8_t *gd = ge + (size_t)n * (K / 2);
  const uint8_t *ud = ue + (size_t)n * (K / 2);
  const uint8_t *gsf = ge + data_bytes;
  const uint8_t *usf = ue + data_bytes;
  const int xrow = slot / n_expert;
  const float *xt = A8 ? nullptr : xq + (size_t)xrow * K;
  const __nv_fp8_e4m3 *xt8 = A8 ? xq8 + (size_t)xrow * K : nullptr;
  float g = 0.f, u = 0.f;
  for (int k0 = lane * 8; k0 < K; k0 += 32 * 8) {
    const uint32_t wg = *(const uint32_t *)(gd + (k0 >> 1));
    const uint32_t wu = *(const uint32_t *)(ud + (k0 >> 1));
    const float sg = gemv_sf_val(gsf[sfl(n, k0 & ~31, 0)]);
    const float su = gemv_sf_val(usf[sfl(n, k0 & ~31, 0)]);
    const float sa = A8 ? gemv_sf_val(xsf[pulsar_mx_sfoff(xrow, k0 >> 5, xkbp)]) : 0.f;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      const float xv = A8 ? (__half2float((__half)xt8[k0 + j]) * sa) : xt[k0 + j];
      g += lut[(wg >> (4 * j)) & 0xFu] * sg * xv;
      u += lut[(wu >> (4 * j)) & 0xFu] * su * xv;
    }
  }
  for (int sh = 16; sh > 0; sh >>= 1) {
    g += __shfl_xor_sync(0xffffffffu, g, sh);
    u += __shfl_xor_sync(0xffffffffu, u, sh);
  }
  if (lane == 0) {
    /* swiglu identical to swiglu_kernel above (clamp then silu(gate)*up*rweight) */
    if (clampv > 1.0e-6f) {
      if (g > clampv) g = clampv;
      if (u > clampv) u = clampv;
      if (u < -clampv) u = -clampv;
    }
    m[n] = (g / (1.f + expf(-g))) * u * rw[slot];
  }
}

/* A8=true reads mid as the E4M3 + E8M0 pair e4m3_act_pack_kernel emitted, rather
 * than the f32 buffer the round-trip leaves behind after deriving exactly those
 * bytes. Same value into the dot product, 1 byte read instead of 4. */
template <class SFL, bool A8 = false>
__global__ static void expert_gemv_down_kernel(
    float *down_out,          // [n_slots, N]
    const float *midq,        // [n_slots, K] f32 mid (A8=false)
    const uint8_t *midq8,     // [n_slots, K] E4M3 mid (A8=true)
    const uint8_t *midsf,     // [n_slots, K/32] E8M0
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
  const float *xt = A8 ? nullptr : midq + (size_t)slot * K;
  const uint8_t *xt8 = A8 ? midq8 + (size_t)slot * K : nullptr;
  const uint8_t *xsf = A8 ? midsf + (size_t)slot * (K / 32) : nullptr;
  float a = 0.f;
  for (int k0 = lane * 8; k0 < K; k0 += 32 * 8) {
    const uint32_t w = *(const uint32_t *)(dd + (k0 >> 1));
    const float sc = gemv_sf_val(dsf[sfl(n, k0 & ~31, 0)]);
    const float sa = A8 ? gemv_sf_val(xsf[k0 >> 5]) : 0.f;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      const float xv = A8 ? (__half2float((__half)*(const __nv_fp8_e4m3 *)&xt8[k0 + j]) * sa)
                          : xt[k0 + j];
      a += lut[(w >> (4 * j)) & 0xFu] * sc * xv;
    }
  }
  for (int sh = 16; sh > 0; sh >>= 1) a += __shfl_xor_sync(0xffffffffu, a, sh);
  if (lane == 0) o[n] = a;
}


/* W4A8 activation round-trip for the decode/small-batch GEMV: quantize f32 -> E4M3 (per-32 dynamic
 * UE8M0 block scale, EXACTLY as pack_act_e4m3_rowmajor) then dequantize back to f32, so the GEMV
 * (f32 act . fp4 weight dot) computes the SAME function as the prefill W4A8 grouped GEMM
 * (E4M3 act x MXFP4 weight). Keeps decode numerics consistent with prefill -- no fp4-vs-E4M3 drift
 * across the prefill/decode boundary, fully source-faithful. */
__global__ static void e4m3_act_roundtrip_kernel(float *xq, const float *x, long nblk32) {
  const long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblk32) return;
  const float *src = x + b * 32;
  float *dst = xq + b * 32;
  float mx = 0.f;
  for (int i = 0; i < 32; i++) mx = fmaxf(mx, fabsf(src[i]));
  const int se = pulsar_mx_shared_exp(mx);   /* ONE definition, shared header */
  const float inv = exp2f((float)-se), s = exp2f((float)se);
  for (int i = 0; i < 32; i++) dst[i] = (float)((cutlass::float_e4m3_t)(src[i] * inv)) * s;
}

/* Packing twin of e4m3_act_roundtrip_kernel: IDENTICAL recipe -- same per-32
 * amax, same se, same encode -- but it keeps the E4M3 byte and the E8M0
 * exponent instead of multiplying straight back to f32.  The round-trip derives
 * the compact form and throws it away; the consumer then reads 4 bytes where 1
 * plus a shared scale would do.
 *
 * Bit-exact against the round-trip by construction: the GEMV computes
 * (float)e4m3 * exp2(se), which is exactly what dst[i] held. */
__global__ static void e4m3_act_pack_kernel(uint8_t *q, uint8_t *sf, const float *x, long nblk32) {
  const long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblk32) return;
  const float *src = x + b * 32;
  uint8_t *dq = q + b * 32;
  float mx = 0.f;
  for (int i = 0; i < 32; i++) mx = fmaxf(mx, fabsf(src[i]));
  const int se = pulsar_mx_shared_exp(mx);   /* ONE definition, shared header */
  const float inv = exp2f((float)-se);
  sf[b] = pulsar_mx_scale_byte(se);
  for (int i = 0; i < 32; i++) {
    const cutlass::float_e4m3_t e = (cutlass::float_e4m3_t)(src[i] * inv);
    dq[i] = *(const uint8_t *)&e;
  }
}

/* SWIZZLE-writing twin of e4m3_act_pack_kernel.  IDENTICAL per-32 amax, se and
 * encode -- so its values are bit-identical to both the round-trip and the
 * linear packer -- but the scale byte lands at pulsar_mx_sfoff(row, kb, kbp)
 * instead of at sf[b].
 *
 * ⚠ BOTH PACKERS EXIST BECAUSE THE TWO A8 ARMS DISAGREE ABOUT THE SCALE PLANE.
 * expert_gemv_down_kernel reads it LINEARLY (xsf[k0>>5], offset per slot);
 * expert_gemv_gu_swiglu_kernel reads it through the SWIZZLE, because its A8 arm
 * was built to consume the PRODUCER's cache.  Handing a swizzled reader a
 * linear plane compiles clean and computes a well-formed WRONG answer -- that
 * mistake was one build away from shipping on 2026-08-23.
 *
 * ⚠ THE SCALE SLAB MUST BE ZEROED BEFORE THIS RUNS.  mx_sfoff leaves holes
 * whenever rows or blocks are not multiples of 128/4, and the GEMM reads those
 * slots; the producer-side cache zeroes for exactly this reason. */
__global__ static void e4m3_act_pack_swizzled_kernel(uint8_t *q, uint8_t *sf,
                                                     const float *x,
                                                     int nblk_per_row, int kbp,
                                                     long nblk32) {
  const long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblk32) return;
  const int row = (int)(b / nblk_per_row);
  const int kb  = (int)(b % nblk_per_row);
  const float *src = x + b * 32;
  uint8_t *dq = q + b * 32;
  float mx = 0.f;
  for (int i = 0; i < 32; i++) mx = fmaxf(mx, fabsf(src[i]));
  const int se = pulsar_mx_shared_exp(mx);
  const float inv = exp2f((float)-se);
  sf[pulsar_mx_sfoff(row, kb, kbp)] = pulsar_mx_scale_byte(se);
  for (int i = 0; i < 32; i++) {
    const cutlass::float_e4m3_t e = (cutlass::float_e4m3_t)(src[i] * inv);
    dq[i] = *(const uint8_t *)&e;
  }
}

/* Persistent activation round-trip buffers, grown on demand and reused across
 * layers/calls -- single GPU-submission thread, same convention as the other
 * static caches. */
/* thread_local: one GPU-submitting thread owns its own scratch; a second
 * decode thread must not share this grow/free/realloc buffer (double-free +
 * silent overwrite). Matches the g_act_slots / DsparkReduceBufs convention. */
static thread_local float *g_fp4_gemv_actbuf = nullptr;
static thread_local size_t g_fp4_gemv_actbuf_floats = 0;

// Small-batch (n_tokens 2..4) rich-expert FFN over the packed CUTLASS weights.
// down_out gets one pre-weighted FFN result per (token, slot); the caller sums the
// n_expert slices per token (moe_sum_kernel). mid_scratch: [n_tokens*n_expert, mid_dim].
int pulsar_cutlass_expert_ffn_gemv_small(
    float *down_out, float *mid_scratch, const float *x,
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
  /* BOTH legs are native-format now.  The gate/up leg reads its scale plane
   * through pulsar_mx_sfoff's swizzle, so it is fed by
   * e4m3_act_pack_swizzled_kernel; the down leg reads a linear plane and keeps
   * e4m3_act_pack_kernel.  Sizes are in BYTES (payload + scales) expressed in
   * the buffer's float units.
   *
   * The swizzled plane is NOT nblk bytes: mx_sfoff tiles it 128 rows x 4 blocks
   * into 512-byte groups, so it needs ceil(rows/128) * (kbp/4) * 512 and must be
   * ZEROED (the tiling leaves holes the GEMM still reads). */
  const int   x_nblk_row = (int)(in_dim / 32u);
  const int   x_kbp      = pulsar_mx_rup(x_nblk_row, 4);
  const size_t xq_elems  = (size_t)n_tokens * in_dim;
  const size_t xsf_bytes = (size_t)((n_tokens + 127u) / 128u) * (size_t)(x_kbp / 4) * 512u;
  const size_t xq_bytes  = xq_elems + xsf_bytes;
  const size_t midq_elems = (size_t)n_slots * mid_dim;
  const size_t midq_bytes = midq_elems + midq_elems / 32u;
  const size_t need = (xq_bytes + 3u) / 4u + (midq_bytes + 3u) / 4u;
  if (need > g_fp4_gemv_actbuf_floats) {
    if (g_fp4_gemv_actbuf) { pulsar_gpu_seg_note_device_free(); cudaFree(g_fp4_gemv_actbuf); }
    g_fp4_gemv_actbuf = nullptr;
    if (cudaMalloc(&g_fp4_gemv_actbuf, need * sizeof(float)) != cudaSuccess) {
      g_fp4_gemv_actbuf_floats = 0;
      return 1;
    }
    g_fp4_gemv_actbuf_floats = need;
  }
  uint8_t *xq8   = (uint8_t *)g_fp4_gemv_actbuf;
  uint8_t *xsf   = xq8 + xq_elems;
  uint8_t *midq8 = xq8 + ((xq_bytes + 3u) & ~(size_t)3u);
  uint8_t *midsf = midq8 + midq_elems;
  auto sfl_gu = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, mid_dim, in_dim, 1));
  auto sfl_dn = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, out_dim, mid_dim, 1));
  /* HANDOVER (L089): when the producing norm already emitted this x as E4M3 +
   * ue8m0, read THOSE bytes and do not pack.  x is then never dereferenced by
   * this call -- which is the whole point.  This arm's read of the raw f32 was
   * L089's "sixth reader", the lone consumer sitting outside every moe.cu
   * guard, and the reason the ffn f32 store-skip is still cut at n<=8.
   *
   * ⚠ THE HIT IS ONLY VALID BECAUSE BOTH SIDES ARE THE SWIZZLE, and that was
   * checked on both ends rather than assumed: the producer writes through
   * pulsar_mx_emit_block -> pulsar_mx_sfoff, and expert_gemv_gu_swiglu_kernel
   * reads through pulsar_mx_sfoff, with act_kbp = mx_rup(in_dim/32,4) = the
   * x_kbp computed here.  Handing a swizzled reader a linear plane compiles
   * clean and computes a well-formed WRONG answer; the reverse assumption --
   * that nothing needed the swizzle -- is what broke the engine in 7df3b75. */
  const bool have_handover = (act_q != nullptr && act_sf != nullptr && act_kbp == x_kbp);
  const __nv_fp8_e4m3 *gu_q  = have_handover ? (const __nv_fp8_e4m3 *)act_q
                                             : (const __nv_fp8_e4m3 *)xq8;
  const uint8_t       *gu_sf = have_handover ? (const uint8_t *)act_sf : xsf;
  if (!have_handover) {
    /* MISS: pack from x ourselves.  Announce once -- a silently-missed
     * handover must not be able to masquerade as a hit, the same reasoning as
     * the sibling arms' "no re-encode" announce. */
    static int announced_pack = 0;
    if (!announced_pack) {
      announced_pack = 1;
      fprintf(stderr, "pulsar: small-batch FFN GEMV packs x itself (handover miss) "
                      "in_dim=%d n_tokens=%d\n", in_dim, n_tokens);
    }
    const long nb = (long)(xq_elems / 32);
    /* Zero first: the swizzle leaves holes and the GEMM reads them. */
    if (cudaMemsetAsync(xsf, 0, xsf_bytes) != cudaSuccess) return 1;
    e4m3_act_pack_swizzled_kernel<<<(unsigned)((nb + 127) / 128), 128>>>(
        xq8, xsf, x, x_nblk_row, x_kbp, nb);   /* W4A8: E4M3 acts, swizzled SF */
  }
  {
    dim3 g((unsigned)((mid_dim + 7) / 8), n_slots);
    expert_gemv_gu_swiglu_kernel<decltype(sfl_gu), true><<<g, 256>>>(
        mid_scratch, nullptr, gu_q, gu_sf, x_kbp, selected, rweights,
        gate_w, up_w, gate_stride, gate_data_bytes, sfl_gu, clamp,
        n_expert, n_total_expert, in_dim, mid_dim);
  }
  {
    const long nb = (long)(midq_elems / 32);
    e4m3_act_pack_kernel<<<(unsigned)((nb + 127) / 128), 128>>>(midq8, midsf, mid_scratch, nb);   /* W4A8: E4M3 acts */
  }
  {
    dim3 g((unsigned)((out_dim + 7) / 8), n_slots);
    expert_gemv_down_kernel<decltype(sfl_dn), true><<<g, 256>>>(
        down_out, nullptr, midq8, midsf, selected,
        down_w, down_stride, down_data_bytes, sfl_dn,
        n_total_expert, mid_dim, out_dim);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}


/* ---- Single-projection W4A8 GEMV for MIXED type-40 layers at DECODE/small-batch (n<=4). ----
 * Decode is memory-bound; the M=1 CUTLASS tensor-core GEMM wastes launch + pack + TC-underfill
 * overhead. These reuse the lean expert_gemv_* kernels (fp4 weight read directly, dequant via LUT,
 * E4M3-roundtripped f32 activations = same function as the prefill grouped GEMM). One launch over
 * all (token,expert) slots, no per-expert loop, no host sync. `mid`/`down_out` are the pair-layout
 * f32 accumulators the caller composes with the dp4a side. Persistent actbuf grown on demand. */
static int gemv_actbuf_ensure(size_t need_floats) {
  if (need_floats <= g_fp4_gemv_actbuf_floats) return 1;
  if (g_fp4_gemv_actbuf) { pulsar_gpu_seg_note_device_free(); cudaFree(g_fp4_gemv_actbuf); }
  g_fp4_gemv_actbuf = nullptr;
  if (cudaMalloc(&g_fp4_gemv_actbuf, need_floats * sizeof(float)) != cudaSuccess) { g_fp4_gemv_actbuf_floats = 0; return 0; }
  g_fp4_gemv_actbuf_floats = need_floats;
  return 1;
}
/* gate/up W4A8 GEMV -> mid[n_slots,mid_dim] = silu(clamp(gate))*clamp(up)*rw (pair layout). */
int pulsar_cutlass_gemv_gateup(
    float *mid, const float *x, const int32_t *selected, const float *rweights,
    const uint8_t *gate_w, const uint8_t *up_w, uint64_t gate_stride, uint64_t gate_data_bytes,
    float clamp, int n_tokens, int n_expert, unsigned n_total_expert, int in_dim, int mid_dim,
    const void *act_q, const void *act_sf, int act_kbp) {
  if (in_dim % 256 || mid_dim % 8 || (gate_stride & 3u)) return 1;
  const unsigned n_slots = (unsigned)(n_tokens * n_expert);
  auto sfl_gu = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, mid_dim, in_dim, 1));
  dim3 g((unsigned)((mid_dim + 7) / 8), n_slots);

  /* Preferred: read the E4M3 the producing norm already emitted. The fallback
   * derives the same encoding here and dequantises it straight back to f32 --
   * correct, and pure waste when the bytes already exist. A miss is not an
   * error; it means no cached encoding was handed over. */
  if (act_q && act_sf) {
    /* Say it once. A bit-exactness check cannot tell "the A8 arm ran and agrees"
     * from "the A8 arm never ran", and a silently-missed cache lookup produces
     * exactly the second -- the failure this file's sibling diagnostics were
     * added for on 2026-08-17. */
    static int announced_gemv_a8 = 0;
    if (!announced_gemv_a8) {
      announced_gemv_a8 = 1;
      fprintf(stderr, "pulsar: fp4 decode GEMV = producer's E4M3 (no re-encode) "
                      "for in_dim=%d mid_dim=%d\n", in_dim, mid_dim);
    }
    expert_gemv_gu_swiglu_kernel<decltype(sfl_gu), true><<<g, 256>>>(
        mid, nullptr, (const __nv_fp8_e4m3 *)act_q, (const uint8_t *)act_sf, act_kbp,
        selected, rweights, gate_w, up_w,
        gate_stride, gate_data_bytes, sfl_gu, clamp, n_expert, n_total_expert, in_dim, mid_dim);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
  }
  const size_t xq_floats = (size_t)n_tokens * in_dim;
  if (!gemv_actbuf_ensure(xq_floats)) return 1;
  float *xq = g_fp4_gemv_actbuf;
  { const long nb = (long)(xq_floats / 32);
    e4m3_act_roundtrip_kernel<<<(unsigned)((nb + 127) / 128), 128>>>(xq, x, nb); }
  expert_gemv_gu_swiglu_kernel<decltype(sfl_gu), false><<<g, 256>>>(
      mid, xq, nullptr, nullptr, 0, selected, rweights, gate_w, up_w,
      gate_stride, gate_data_bytes, sfl_gu, clamp, n_expert, n_total_expert, in_dim, mid_dim);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
/* down W4A8 GEMV -> down_out[n_slots,out_dim] (pair layout, NO routing weight -- applied at gate/up). */
int pulsar_cutlass_gemv_down(
    float *down_out, const float *mid, const int32_t *selected,
    const uint8_t *down_w, uint64_t down_stride, uint64_t down_data_bytes,
    int n_tokens, int n_expert, unsigned n_total_expert, int mid_dim, int out_dim) {
  if (mid_dim % 256 || out_dim % 8 || (down_stride & 3u)) return 1;
  const unsigned n_slots = (unsigned)(n_tokens * n_expert);
  const size_t midq_elems = (size_t)n_slots * mid_dim;
  const long nb = (long)(midq_elems / 32);
  /* mid must be quantised here -- the gate/up GEMV gives one warp one output
   * element, so a 32-wide MX block spans four CTAs and no epilogue can see its
   * amax.  What CAN go is the dequantise: pack once and let the GEMV read the
   * bytes, instead of writing 4 bytes per element that hold 1 byte of value. */
  /* Bytes needed: midq_elems E4M3 payload + nb ue8m0 scale BYTES (one per
   * 32-value block), expressed in the buffer's float units.  This budgeted
   * nb*4 -- four bytes per scale byte -- until the types sweep: conservative,
   * never corrupting, but a unit error all the same. */
  const size_t need_floats = (midq_elems + (size_t)nb + 3u) / 4u;
  if (!gemv_actbuf_ensure(need_floats)) return 1;
  uint8_t *midq8 = (uint8_t *)g_fp4_gemv_actbuf;
  uint8_t *midsf = midq8 + midq_elems;
  auto sfl_dn = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, out_dim, mid_dim, 1));
  e4m3_act_pack_kernel<<<(unsigned)((nb + 127) / 128), 128>>>(midq8, midsf, mid, nb);
  dim3 g((unsigned)((out_dim + 7) / 8), n_slots);
  expert_gemv_down_kernel<decltype(sfl_dn), true><<<g, 256>>>(
      down_out, nullptr, midq8, midsf, selected, down_w,
      down_stride, down_data_bytes, sfl_dn, n_total_expert, mid_dim, out_dim);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}


// Pack SOURCE-format MXFP4 (separate E2M1 [N,K/2] row-major + E8M0 [N,K/32]) — exactly as the
// DeepSeek-V4-Flash source stores rich experts — into CUTLASS B layout (ColumnMajor packed E2M1
// data + swizzled SFB). Host-side, lossless (copies nibbles+scale verbatim). This is the permanent
// source->CUTLASS packer; nothing consumes ds4's 17-byte format.
void pulsar_cutlass_pack_source(uint8_t *Bd, ElementSF *Bsf, const uint8_t *e2m1, const uint8_t *e8m0, int N, int K){
  auto lB   = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N,K,1});
  auto layB = make_layout(make_shape(N,K,1), lB);
  auto lSFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1,N,K,1));
  auto tB   = make_tensor(recast_ptr<cutlass::float_e2m1_t>(Bd), layB);
  auto tSFB = make_tensor(Bsf, lSFB);
  int nblk=K/32, rowbytes=K/2;
  for(int n=0;n<N;n++) for(int kb=0;kb<nblk;kb++){
    tSFB(n, kb*32, 0) = ElementSF::bitcast(e8m0[(size_t)n*nblk + kb]);   // E8M0 scale
    const uint8_t *row = e2m1 + (size_t)n*rowbytes + kb*16;              // E2M1: 16 bytes = 32 nibbles
    for(int i=0;i<16;i++){ uint8_t byte=row[i]; int k=kb*32+i*2;
      tB(n,k,0)   = cutlass::float_e2m1_t::bitcast(byte & 0xF);
      tB(n,k+1,0) = cutlass::float_e2m1_t::bitcast(byte >> 4); }
  }
}

// Physical element count of the swizzled SF tensor for a weight of shape (N=out, K=in).
size_t pulsar_cutlass_weight_sf_count(int N, int K){
  auto lSFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1,N,K,1));
  return cute::size(cute::filter_zeros(lSFB));
}

#ifdef PULSAR_MXFP4_REPACK_CLI
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
// Offline converter CLI. Two modes:
//   per-expert: e2m1.bin e8m0.bin N K out_data.bin out_sf.bin
//   stacked   : --stacked e2m1_stacked.bin e8m0_stacked.bin N K n_expert out_blob.bin
//               (in: [ne,N,K/2] + [ne,N,K/32]; out: per expert data||sf, expert-major)
int main(int argc, char **argv){
  if(argc>=8 && strcmp(argv[1],"--stacked")==0){
    int N=atoi(argv[4]), K=atoi(argv[5]), ne=atoi(argv[6]);
    size_t e2each=(size_t)N*(K/2), e8each=(size_t)N*(K/32);
    std::vector<uint8_t> e2(e2each*ne), e8(e8each*ne);
    FILE*f1=fopen(argv[2],"rb"); if(!f1||fread(e2.data(),1,e2.size(),f1)!=e2.size()){ fprintf(stderr,"e2m1 read fail\n"); return 1; } fclose(f1);
    FILE*f2=fopen(argv[3],"rb"); if(!f2||fread(e8.data(),1,e8.size(),f2)!=e8.size()){ fprintf(stderr,"e8m0 read fail\n"); return 1; } fclose(f2);
    size_t sfn=pulsar_cutlass_weight_sf_count(N,K);
    FILE*fo=fopen(argv[7],"wb"); if(!fo){ fprintf(stderr,"out open fail\n"); return 1; }
    std::vector<uint8_t> Bd((size_t)N*K/2); std::vector<ElementSF> Bsf(sfn);
    for(int i=0;i<ne;i++){
      std::fill(Bd.begin(),Bd.end(),(uint8_t)0);
      std::fill(Bsf.begin(),Bsf.end(),ElementSF::bitcast(127));
      pulsar_cutlass_pack_source(Bd.data(), Bsf.data(), e2.data()+(size_t)i*e2each, e8.data()+(size_t)i*e8each, N, K);
      fwrite(Bd.data(),1,Bd.size(),fo); fwrite(Bsf.data(),sizeof(ElementSF),sfn,fo);
    }
    fclose(fo);
    printf("stacked N=%d K=%d ne=%d -> per-expert data=%zuB sf=%zuB total=%zuB\n",
           N,K,ne, Bd.size(), sfn*sizeof(ElementSF), (size_t)ne*(Bd.size()+sfn*sizeof(ElementSF)));
    return 0;
  }
  if(argc<7){ fprintf(stderr,"usage: %s e2m1.bin e8m0.bin N K out_data.bin out_sf.bin\n       %s --stacked e2m1_stacked.bin e8m0_stacked.bin N K n_expert out_blob.bin\n",argv[0],argv[0]); return 1; }
  int N=atoi(argv[3]), K=atoi(argv[4]);
  size_t e2n=(size_t)N*(K/2), e8n=(size_t)N*(K/32);
  std::vector<uint8_t> e2(e2n), e8(e8n);
  FILE*f1=fopen(argv[1],"rb"); if(!f1||fread(e2.data(),1,e2n,f1)!=e2n){ fprintf(stderr,"e2m1 read fail %s\n",argv[1]); return 1; } fclose(f1);
  FILE*f2=fopen(argv[2],"rb"); if(!f2||fread(e8.data(),1,e8n,f2)!=e8n){ fprintf(stderr,"e8m0 read fail %s\n",argv[2]); return 1; } fclose(f2);
  std::vector<uint8_t> Bd((size_t)N*K/2,0);
  size_t sfn=pulsar_cutlass_weight_sf_count(N,K);
  std::vector<ElementSF> Bsf(sfn, ElementSF::bitcast(127));
  pulsar_cutlass_pack_source(Bd.data(), Bsf.data(), e2.data(), e8.data(), N, K);
  FILE*fd=fopen(argv[5],"wb"); fwrite(Bd.data(),1,Bd.size(),fd); fclose(fd);
  FILE*fs=fopen(argv[6],"wb"); fwrite(Bsf.data(),sizeof(ElementSF),sfn,fs); fclose(fs);
  printf("packed N=%d K=%d -> data=%zuB sf=%zuB\n",N,K,Bd.size(),sfn*sizeof(ElementSF));
  return 0;
}
#endif

/* An #ifdef PULSAR_MXFP4_STANDALONE self-check main() lived here until the
 * 2026-08-22 types sweep.  Nothing ever defined the macro, its oracle modelled
 * activations as FP4 (pre-W4A8), and it no longer compiled -- its
 * pulsar_cutlass_grouped_moe call was four arguments short.  A silently absent
 * self-check over exactly the format it was meant to pin is worse than none;
 * the living oracle is temp/fp4gemv_test.cu plus the gate suite. */

#pragma once

/* Device-pass capability tiers (L321): the ONE place a kernel asks which instructions its compile target has.
 *
 * A TU compiled for an older target (sm_75, Turing) still has to compile and link, because the arch trap
 * (cuda_tu_archs_ok) demands device code for the GPU's SM major from every TU.  A primitive whose instruction the
 * target lacks keeps its signature and traps instead -- never a second implementation (rule 1): the host refuses the
 * lane by name before anything launches it, so the trap is only the proof that nothing did.  The host pass has no
 * __CUDA_ARCH__ and sees every tier as present, which keeps host stubs and template instantiations identical across
 * targets; a GB10 build (sm_120f) has every tier, so its device code is the same source it was before these guards.
 *
 *   PULSAR_SM80  mma.sync m16n8k16 (f16 and bf16), cp.async, __reduce_*_sync
 *   PULSAR_SM89  mma.sync with e4m3 operands, cvt.*.e4m3x2
 *   PULSAR_SM90  thread-block clusters (__cluster_dims__, cooperative_groups::this_cluster)
 *
 * Family-specific (sm_120f) instructions keep their own __CUDA_ARCH_FAMILY_SPECIFIC__ tests beside their asm. */

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define PULSAR_SM80 1
#else
#define PULSAR_SM80 0
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
#define PULSAR_SM89 1
#else
#define PULSAR_SM89 0
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
#define PULSAR_SM90 1
#else
#define PULSAR_SM90 0
#endif

#ifdef __CUDACC__
#include <cstdint>

/* Full-warp integer reductions.  sm_80 has them as one instruction (redux.sync); below it the same value comes from a
 * butterfly of shuffles.  Integer min/max/add are exact and order-free, so both bodies return identical bits: these
 * are one function per target, not two answers. */
template <typename T>
__device__ __forceinline__ T pulsar_warp_add(T v) {
#if PULSAR_SM80
    return __reduce_add_sync(0xffffffffu, v);
#else
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
#endif
}

template <typename T>
__device__ __forceinline__ T pulsar_warp_min(T v) {
#if PULSAR_SM80
    return __reduce_min_sync(0xffffffffu, v);
#else
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = min(v, (T)__shfl_xor_sync(0xffffffffu, v, o));
    return v;
#endif
}

template <typename T>
__device__ __forceinline__ T pulsar_warp_max(T v) {
#if PULSAR_SM80
    return __reduce_max_sync(0xffffffffu, v);
#else
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = max(v, (T)__shfl_xor_sync(0xffffffffu, v, o));
    return v;
#endif
}
#endif

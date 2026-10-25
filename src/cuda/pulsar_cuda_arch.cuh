#pragma once

#include <cuda_runtime.h>
#include <cstdio>

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

/* Host side: what the DEVICE has.  The tiers above say what an IMAGE was compiled with; a launch decision must ask the
 * device, or a build for an older target selects a lane its card cannot run (L321: an sm_75 build on an sm_75 card chose
 * the routed IQ2 MMQ lane, whose block-scaled MMA exists only in the sm_120f image; its other arm returned NaN). */
/* The current device's compute capability as major * 10 + minor (75 Turing, 89 Ada, 121 GB10), asked once per process;
 * 0 if unknown.  Inline (one cache per program) so a model-free gate that links a single kernel object needs no
 * runtime object for it. */
inline int pulsar_cuda_device_cc(void) {
    static int cc = -1;
    if (cc < 0) {
        int dev = 0, ma = 0, mi = 0;
        cc = (cudaGetDevice(&dev) == cudaSuccess &&
              cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
              cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess) ? ma * 10 + mi : 0;
    }
    return cc;
}
inline int pulsar_cuda_device_major(void) { return pulsar_cuda_device_cc() / 10; }

/* A lane whose kernels need compute capability `cc` (major * 10 + minor) refuses on a lesser device -- on the host, by
 * name, once per call site -- returning `ret`; its kernels' trapping arms (the tiers above) then never run.  (L321) */
#define PULSAR_REQUIRE_CC(cc, lane, ret)                                                                          \
    do {                                                                                                         \
        if (pulsar_cuda_device_cc() < (cc)) {                                                                     \
            static bool said_ = false;                                                                           \
            if (!said_) {                                                                                        \
                said_ = true;                                                                                    \
                fprintf(stderr, "pulsar: %s needs sm_%d or newer; this device is sm_%d -- refusing\n", (lane), (cc), \
                        pulsar_cuda_device_cc());                                                                \
            }                                                                                                    \
            return ret;                                                                                          \
        }                                                                                                        \
    } while (0)

/* mma.sync.kind::mxf8f6f4.block_scale (MXFP8 experts, MXFP4 indexer): the sm_12x family only. */
static inline bool pulsar_cuda_block_scale_mma(int major) { return major == 12; }

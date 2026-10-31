/** @file
 *  Device-pass capability tiers (PULSAR_SM80 / SM89 / SM90): the one place a kernel asks what its target has. */
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
/// 1 when the device pass targets sm_80+ (and in the host pass): m16n8k16 f16/bf16 MMA, cp.async, __reduce_*_sync.
#define PULSAR_SM80 1
#else
#define PULSAR_SM80 0
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
/// 1 when the device pass targets sm_89+ (and in the host pass): e4m3 MMA operands, cvt e4m3x2.
#define PULSAR_SM89 1
#else
#define PULSAR_SM89 0
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
/// 1 when the device pass targets sm_90+ (and in the host pass): thread-block clusters.
#define PULSAR_SM90 1
#else
#define PULSAR_SM90 0
#endif

#ifdef __CUDACC__
#include <cstdint>

/* Full-warp integer reductions.  sm_80 has them as one instruction (redux.sync); below it the same value comes from a
 * butterfly of shuffles.  Integer min/max/add are exact and order-free, so both bodies return identical bits: these
 * are one function per target, not two answers. */
/* The m16n8k16 tensor-core MMAs, ONE definition each (L335: the bf16 one was copied into three TUs and the f16 one into
 * two).  sm_80+; below it the body traps -- the launchers refuse such a device by name first (the tiers above). */
/** One mma.sync m16n8k16 bf16 x bf16 -> f32, accumulating into d (A row-major 16x16, B col-major 16x8).
 *  @param d  the thread's four f32 accumulators (in / out)
 *  @param a  the thread's A fragment, four bf16x2 registers
 *  @param b0 B fragment register 0 (bf16x2)
 *  @param b1 B fragment register 1 (bf16x2) */
__device__ __forceinline__ void pulsar_mma_bf16_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if PULSAR_SM80
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void)d; (void)a; (void)b0; (void)b1;
    __trap();
#endif
}

/** One mma.sync m16n8k16 f16 x f16 -> f32, accumulating into d (A row-major 16x16, B col-major 16x8).
 *  @param d  the thread's four f32 accumulators (in / out)
 *  @param a  the thread's A fragment, four f16x2 registers
 *  @param b0 B fragment register 0 (f16x2: k rows 2c, 2c+1)
 *  @param b1 B fragment register 1 (f16x2: k rows 2c+8, 2c+9) */
__device__ __forceinline__ void pulsar_mma_f16_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if PULSAR_SM80
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void)d; (void)a; (void)b0; (void)b1;
    __trap();
#endif
}

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
/// @return the current device's compute capability, major * 10 + minor, cached on first call; 0 if unknown.
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
/// @return the current device's compute-capability major (0 if unknown).
inline int pulsar_cuda_device_major(void) { return pulsar_cuda_device_cc() / 10; }

/* Per-device host state (L335): a process static caching a DEVICE answer -- a function attribute, an SM count, a
 * device symbol's address -- is right for the first device that asks and wrong on the next one a TP rank or a second
 * engine runs on.  Such a cache is an array indexed by the calling thread's current device. */
/** Devices a per-device host cache (pulsar_cuda_device_slot) holds. */
#define PULSAR_CUDA_DEVICE_SLOTS 16
/// @return the calling thread's current device, 0 .. PULSAR_CUDA_DEVICE_SLOTS - 1; -1 when it cannot be read or is
///         past the caches (said once)
inline int pulsar_cuda_device_slot(void) {
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        (void)cudaGetLastError();
        dev = -1;
    }
    if (dev < 0 || dev >= PULSAR_CUDA_DEVICE_SLOTS) {
        static bool said = false;
        if (!said) {
            said = true;
            fprintf(stderr, "pulsar: device %d is past the %d per-device host caches -- refusing\n", dev,
                    PULSAR_CUDA_DEVICE_SLOTS);
        }
        return -1;
    }
    return dev;
}

/* A kernel's dynamic-shared-memory opt-in, per device (L335): every launcher that opted in "once" through a function
 * static set the attribute on the first device only -- and two ignored the return code. */
/** Opts `func` in to `bytes` of dynamic shared memory on the current device, once per (device, larger size).
 *  @param func  the kernel
 *  @param set   the caller's per-device record of the bytes each device was given (static, zero-initialised)
 *  @param bytes the launch's dynamic shared memory
 *  @return true when the device grants it (now or before); false refused (the caller says why) */
inline bool pulsar_cuda_smem_opt_in(const void *func, int (&set)[PULSAR_CUDA_DEVICE_SLOTS], size_t bytes) {
    const int d = pulsar_cuda_device_slot();
    if (d < 0) return false;
    if ((size_t)set[d] >= bytes) return true;
    if (cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    set[d] = (int)bytes;
    return true;
}

/* A lane whose kernels need compute capability `cc` (major * 10 + minor) refuses on a lesser device -- on the host, by
 * name, once per call site -- returning `ret`; its kernels' trapping arms (the tiers above) then never run.  (L321) */
/// Refuses the enclosing function (`return ret`) on a device below compute capability `cc`, naming `lane` once.
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
/// @param major a compute-capability major.
/// @return true when it has mma.sync kind::mxf8f6f4.block_scale (the sm_12x family).
static inline bool pulsar_cuda_block_scale_mma(int major) { return major == 12; }

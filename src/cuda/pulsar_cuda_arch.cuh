/** @file
 *  Device-pass capability tiers (PULSAR_SM80 / SM89 / SM90): the one place a kernel asks what its target has. */
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <utility>

/* Device-pass capability tiers (L321): the ONE place a kernel asks which instructions its compile target has.
 *
 * A TU compiled for an older target (sm_75, Turing) still has to compile and link, because the arch trap
 * (cuda_tu_archs_ok) demands device code for the GPU's SM major from every TU.  A primitive whose instruction the
 * target lacks keeps its signature and traps instead -- never a second implementation (rule 1): the host refuses the
 * lane by name before anything launches it, so the trap is only the proof that nothing did.  The host pass has no
 * __CUDA_ARCH__ and sees every tier as present, which keeps host stubs and template instantiations identical across
 * targets; a GB10 build (sm_120f) has every tier, so its device code is the same source it was before these guards.
 *
 *   PULSAR_SM80  mma.sync m16n8k16 (f16 and bf16), cp.async, __reduce_*_sync -- the f16 MMA and cp.async have sm_75 bodies
 *                (L338): cp.async a synchronous copy (the same bytes); the MMA two m16n8k8, a graded numerics change
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
 * two).  The bf16 one is sm_80+ (below it the body traps -- the launchers refuse such a device by name first, the tiers
 * above); the f16 one runs on sm_75 as two m16n8k8 (pulsar_mma_f16_16816_k8, L338). */
/** The f16 m16n8k16 MMA as two m16n8k8 MMAs (k 0..7, then k 8..15) -- the sm_75 body of pulsar_mma_f16_16816, defined on
 *  every target (m16n8k8 f16 is sm_75+) so the k8 split gate can run it beside the k16 instruction on one card.
 *  @param d  the thread's four f32 accumulators (in / out)
 *  @param a  the thread's A fragment, four f16x2 registers (a[0] a[1]: k 0..7; a[2] a[3]: k 8..15)
 *  @param b0 B fragment register 0 (k 0..7)
 *  @param b1 B fragment register 1 (k 8..15) */
__device__ __forceinline__ void pulsar_mma_f16_16816_k8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                                        uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[2]), "r"(a[3]), "r"(b1));
}
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
 *  sm_75 (L338, plan 123): Turing has the fp16 MMA at m16n8k8 only, so the same products run as two m16n8k8 MMAs, k 0..7
 *  then k 8..15 -- the k16 fragment's registers split along k exactly so (A a[0] a[1] / B b0 hold k 0..7, a[2] a[3] / b1
 *  k 8..15), the split exllamav3's sm_75 port and the GQA block's (L321) use; f32 accumulate as on sm_80.  NOT the k16
 *  instruction's bits: each half rounds its partial sum once more, so 35 % of the accumulators differ by an ulp or a few
 *  (measured on GB10, which has both) -- a numerics change on sm_75, the only arithmetic Turing has for this shape,
 *  graded by tests/mma_k8_split_gate.cu (both within 2^-21 * sum |terms| of the exact product: k16 4.8, split 6.2 units of
 *  2^-24) and by the EXL3 lanes' own gates on the card.
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
    pulsar_mma_f16_16816_k8(d, a, b0, b1);
#endif
}

/* cp.async, ONE definition (L338: the EXL3 MoE and dense-decode TUs each carried a copy).  sm_80+ issues the async copy
 * and its group commit / wait; sm_75 has no cp.async, so the copy is a synchronous 16-byte store with cp.async's exact
 * zero-fill (bytes past src_bytes are 0) and the commit / wait have nothing to wait for.  The shared-memory bytes a
 * consumer reads after its __syncthreads are the same on both targets: the callers' barriers between filling a buffer
 * and reading it (and between reading it and refilling it) are what order the copy, not the wait. */
/** 16 bytes global -> shared (cp.async.cg on sm_80+; a synchronous store on sm_75), zero-filling past `src_bytes`.
 *  @param smem       16-byte aligned shared destination
 *  @param gmem       16-byte aligned global source (not read when src_bytes is 0)
 *  @param src_bytes  bytes read, 0 .. 16; the rest of the 16 are zero */
__device__ __forceinline__ void pulsar_cp_async16(void *smem, const void *gmem, int src_bytes) {
#if PULSAR_SM80
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(src_bytes));
#else
    if (src_bytes == 16) {
        *(int4 *)smem = *(const int4 *)gmem;
    } else {
        uint8_t *d = (uint8_t *)smem;
        const uint8_t *g = (const uint8_t *)gmem;
        for (int i = 0; i < 16; ++i) d[i] = i < src_bytes ? g[i] : (uint8_t)0;
    }
#endif
}
/** Commit the cp.async issued so far as one group (sm_80+); nothing to commit on sm_75 (the copies are done). */
__device__ __forceinline__ void pulsar_cp_async_commit(void) {
#if PULSAR_SM80
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
/** Wait until at most `N` committed cp.async groups are pending (sm_80+); nothing pending on sm_75.
 *  @tparam N  groups allowed to stay in flight */
template <int N>
__device__ __forceinline__ void pulsar_cp_async_wait(void) {
#if PULSAR_SM80
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
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

/* L340 (after TensorFold d93002949, FlashInfer 80ece849c): programmatic dependent launch.  A kernel launched with
 * pulsar_launch_pdl may start while the kernel before it on the stream is still running; everything it does before
 * pulsar_pdl_wait() must not read what that kernel writes (nor write what it reads) -- weights are the use: their
 * loads go in flight under the producer's tail.  pulsar_pdl_trigger() lets the dependent launch before this grid ends.
 * Byte-exact by construction: the same kernels and the same math, only the overlap.  sm_90+; on an older device the
 * host never sets the attribute, the launch is serialised as any other, and the wait is already satisfied -- so the
 * device bodies below are empty there, which is the same semantics, not a second path. */
#ifdef __CUDACC__
/// Wait until the grid this one depends on (the previous kernel on the stream) has completed and its writes are
/// visible.  A no-op when launched without pulsar_launch_pdl.
__device__ __forceinline__ void pulsar_pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}
/// Let the kernel launched after this one with pulsar_launch_pdl start (it still waits in pulsar_pdl_wait for this
/// grid's completion before reading its output).
__device__ __forceinline__ void pulsar_pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}
/// @brief Launch `kernel` so it may overlap the stream's previous kernel (programmatic stream serialisation, sm_90+;
/// an ordinary launch on an older device).  The kernel must call pulsar_pdl_wait() before it reads that kernel's
/// output.
/// @param kernel  the __global__ function
/// @param grid    grid
/// @param block   block
/// @param smem    dynamic shared memory bytes
/// @param stream  the stream
/// @param args    the kernel's arguments
/// @return the launch's cudaError_t
template <typename... KArgs, typename... Args>
inline cudaError_t pulsar_launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                                     Args &&...args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pulsar_cuda_device_major() >= 9 ? 1u : 0u;
    return cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
}
#endif

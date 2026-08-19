// SPDX-License-Identifier: MIT
// ds4_cuda_env.cuh - the CUDA environment the MMQ sources actually use.
//
// This REPLACES the vendored common.cuh (1,669 lines, MIT, copyright 2023-2026
// The ggml authors), which was kept alive by nineteen symbols.  The bodies below
// are upstream's, carried over unchanged where they are non-obvious, so this is
// still derived work and VENDOR.md still tracks it -- what changed is that the
// file now contains only what is reachable.
//
// WHAT WAS DROPPED, and why it is safe to drop:
//
//   * The AMD/HIP/MUSA half.  Every predicate upstream writes as
//     "NVIDIA-and-X or CDNA or RDNA or MTHREADS" collapses to its NVIDIA arm
//     here: `cc` is read from the DEVICE at runtime, and this engine runs on
//     GB10.  The AMD branches were unreachable, not merely uncompiled.  The
//     NVIDIA arms below are byte-for-byte upstream's, pre-Turing cases included.
//   * ggml_backend_cuda_context, ggml_cuda_pool and ggml_cuda_pool_alloc --
//     retired with the pool itself; MMQ scratch is the CUDA_SCRATCH_MMQ arena.
//   * Ten of the thirteen cuda_device_info fields.  Only cc, warp_size and
//     smpbo are ever read (plus device_count); the rest were filled by our own
//     stub and read by nobody.
//   * Everything else in common.cuh: cuBLAS handles, stream pools, graph
//     capture, tensor helpers, the fattn/mmvq surface, ~40 unused predicates.
//
// If a future vendor bump needs one of those back, take it from upstream rather
// than reconstructing it here.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <array>

#include "ds4_ggml_stubs.h"   // GGML_ASSERT/ABORT/UNUSED, ggml_type, GGML_CUDA_MAX_DEVICES
#define GGML_COMMON_DECL_CUDA   // selects the CUDA declarations in ggml-common.h
#define GGML_COMMON_IMPL_CUDA   // ...and EMITS the tables (iq2xxs_grid et al) as
                                // `static const __device__`.  DECL alone declares
                                // the block layouts but defines no table, and the
                                // table block is #if'd out rather than erroring.
#include "ggml-common.h"      // block_q*/block_iq* layouts + QK8_1, still vendored

#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#define WARP_SIZE 32

// last row of quant. matrices is a multiple of this to avoid out-of-bounds reads
#define MATRIX_ROW_PADDING 512

#define GGML_CUDA_CC_PASCAL          600
#define GGML_CUDA_CC_DP4A            610 // minimum compute capability for __dp4a
#define GGML_CUDA_CC_VOLTA           700
#define GGML_CUDA_CC_TURING          750
#define GGML_CUDA_CC_AMPERE          800
#define GGML_CUDA_CC_BLACKWELL      1200

// Upstream encodes non-NVIDIA vendors as large cc offsets (AMD 0x1000000,
// MTHREADS 0x0100000), so "is NVIDIA" is a range test rather than a flag.  Kept
// verbatim: ds4_mmq_d2r.cu still asks, and a hand-simplified `true` would be a
// different question with the same spelling.
#define GGML_CUDA_CC_OFFSET_MTHREADS 0x0100000
#define GGML_CUDA_CC_IS_NVIDIA(cc)   (cc < GGML_CUDA_CC_OFFSET_MTHREADS)

// Raise a kernel's dynamic-shared-memory cap once per device.  Used by the
// mm_ids launchers.
#define CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes)                                                       \
    do {                                                                                                   \
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = { false };                         \
        const int   id                                                = ggml_cuda_get_device();            \
        if (!shared_memory_limit_raised[id]) {                                                             \
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes)); \
            shared_memory_limit_raised[id] = true;                                                         \
        }                                                                                                  \
    } while (0)

#define GGML_CUDA_CC_RUBIN         1300

// ----------------------------------------------------------------------------
// FEATURE MACROS.  ⚠ THESE ARE LOAD-BEARING AND THEIR ABSENCE IS SILENT.
//
// These are preprocessor macros, not the predicate FUNCTIONS above, and the
// device code selects whole kernel bodies on them:
// ds4_mmq_d2r.cu:905 is `#if defined(TURING_MMA_AVAILABLE)` around the entire
// IQ2 tensor-core GEMM.
//
// Leaving one undefined does NOT fail to compile.  It takes the #else arm, and
// the build is clean, and the engine runs, and every logit is wrong.  That is
// exactly what happened when this header first replaced common.cuh: the first
// version omitted TURING_MMA_AVAILABLE, the D2R kernel compiled to its fallback,
// and the prefill gate came back 129280/129280 logits different at every depth
// with decode acceptance down from 0.4237 to 0.1786.
//
// The empty-shim-and-compile method that found the other nineteen symbols CANNOT
// find these: an undefined identifier is an error, an undefined `#if defined()`
// is just false.  If you touch this block, grep `_AVAILABLE` across src/cuda/mmq
// and diff the set against what is defined here.
//
// AMD_MFMA_AVAILABLE / AMD_WMMA_AVAILABLE are deliberately absent: upstream
// defines them only under GGML_USE_HIP, so they were never defined in this build
// either, and the code that tests them takes the same arm it always did.
// ----------------------------------------------------------------------------

#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
#define FP16_AVAILABLE
#endif

#if defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610
#define FAST_FP16_AVAILABLE
#endif

// The Volta instructions are in principle available on Turing or newer but they
// are effectively unusable:
#if __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
#define VOLTA_MMA_AVAILABLE
#endif

#if __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
#define TURING_MMA_AVAILABLE
#endif

#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define AMPERE_MMA_AVAILABLE
#endif

#if __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL && __CUDA_ARCH__ < GGML_CUDA_CC_RUBIN
#define BLACKWELL_MMA_AVAILABLE
#endif

// Tautological today, on purpose: it fires if a later edit removes the defines
// above, which is the failure this whole comment block exists to prevent.  The
// D2R IQ2 GEMM is the engine's expert matmul -- if it is compiling to a stub we
// want a build error, not a gate run.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && !defined(TURING_MMA_AVAILABLE)
#error "TURING_MMA_AVAILABLE undefined on a Turing+ arch: the D2R IQ2 GEMM would silently compile to its #else stub."
#endif

// ----------------------------------------------------------------------------
// Error reporting.  ggml_cuda_error is defined in ds4_ggml_stubs.cu.
// ----------------------------------------------------------------------------

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg);

#define CUDA_CHECK_GEN(err, success, error_fn)                                      \
     do {                                                                           \
        auto err_ = (err);                                                          \
        if (err_ != (success)) {                                                    \
            ggml_cuda_error(#err, __func__, __FILE__, __LINE__, error_fn(err_));    \
        }                                                                           \
    } while (0)

#define CUDA_CHECK(err) CUDA_CHECK_GEN(err, cudaSuccess, cudaGetErrorString)

// ----------------------------------------------------------------------------
// Which archs is this binary compiled for?  Upstream's recursive constexpr fold
// over __CUDA_ARCH_LIST__, unchanged -- the predicates below ask "is the DEVICE
// at least arch X AND did we emit code for it", and dropping the second half
// would silently answer yes for a device we cannot run on.
// ----------------------------------------------------------------------------

#ifdef __CUDA_ARCH_LIST__
constexpr int ggml_cuda_highest_compiled_arch_impl(const int /*arch*/, const int cur) {
    if (cur == 0) {
        return -1;
    }
    return cur;
}

template<class ... Archs>
constexpr int ggml_cuda_highest_compiled_arch_impl(const int arch, const int cur, const int first, Archs... rest) {
    if (first <= arch && first > cur) {
        return ggml_cuda_highest_compiled_arch_impl(arch, first, rest...);
    } else {
        return ggml_cuda_highest_compiled_arch_impl(arch, cur, rest...);
    }
}

constexpr int ggml_cuda_highest_compiled_arch(const int arch) {
    return ggml_cuda_highest_compiled_arch_impl(arch, 0, __CUDA_ARCH_LIST__);
}
#else
static int ggml_cuda_highest_compiled_arch(const int arch) {
    return arch;
}
#endif // __CUDA_ARCH_LIST__

// ----------------------------------------------------------------------------
// Feature predicates.  NVIDIA-only forms of upstream's; see the header note.
// ----------------------------------------------------------------------------

// For feature selection of external libraries, e.g. cuBLAS.  Hardware check:
// asks about the DEVICE, not about what we compiled.
static bool fp16_mma_hardware_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA;
}

static bool turing_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING;
}

static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
    return 32;
}

static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    return 16;
#else
    return 8;
#endif
}

// ----------------------------------------------------------------------------
// Device-side helpers.
// ----------------------------------------------------------------------------

[[noreturn]]
static __device__ void no_device_code(
    const char * file_name, const int line, const char * function_name, const int arch, const char * arch_list) {
    printf("%s:%d: ERROR: CUDA kernel %s has no device code compatible with CUDA arch %d. compiled for: %s\n",
           file_name, line, function_name, arch, arch_list);
    __trap();

    GGML_UNUSED(no_device_code); // suppress unused function warning
}

#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))
#else
#define NO_DEVICE_CODE //GGML_ABORT("NO_DEVICE_CODE not valid in host code.")
#endif // __CUDA_ARCH__

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_sum(int x) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    return __reduce_add_sync(0xffffffff, x);
#else
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
#endif
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_any(int x) {
    if (width == ggml_cuda_get_physical_warp_size()) {
        return __any_sync(0xffffffff, x);
    } else {
#pragma unroll
        for (int offset = width/2; offset > 0; offset >>= 1) {
            x = __shfl_xor_sync(0xffffffff, x, offset, width) || x;
        }
        return x;
    }
}

template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    static_assert(
        nbytes <= ggml_cuda_get_max_cpy_bytes() || alignment == 0,
        "You are misusing the alignment parameter for ggml_cuda_memcpy_1. "
        "The intent is for the parameter is only as a workaround if either one of the pointers is not properly aligned. "
        "If you use it to do more bytes per copy than ggml_cuda_max_cpy_bytes() the reads and writes may not be coalesced. "
        "Call ggml_cuda_memcpy_1 in a loop instead.");
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nb_per_cpy == 0, "bad number of bytes per copy");
        }
    }
}

// ----------------------------------------------------------------------------
// Device info.  Trimmed to the three fields anything reads -- ds4_ggml_stubs.cu
// fills these and nothing else now.
// ----------------------------------------------------------------------------

struct ggml_cuda_device_info {
    int device_count;

    struct cuda_device_info {
        int     cc;         // compute capability, major*100 + minor*10
        int     warp_size;  // threads in a dispatch
        size_t  smpbo;      // max. shared memory per block, with opt-in
    };

    cuda_device_info devices[GGML_CUDA_MAX_DEVICES] = {};
};

const ggml_cuda_device_info & ggml_cuda_info();

void ggml_cuda_set_device(int device);
int ggml_cuda_get_device();

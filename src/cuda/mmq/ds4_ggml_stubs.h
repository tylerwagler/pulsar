// SPDX-License-Identifier: MIT
// ds4_ggml_stubs.h - minimal ggml-API stubs for ds4's vendored mmq kernels.
//
// The mma.cuh / mmid.cuh / mmid.cu files in this directory are vendored
// verbatim from llama.cpp's ggml-cuda backend (MIT, copyright 2023-2026 The
// ggml authors). They transitively #include "ggml.h", "ggml-impl.h",
// "ggml-cuda.h" - in ds4 those names resolve to thin redirect headers in this
// directory which all #include this stubs file.
//
// This file declares the minimum surface of the ggml API that the vendored
// CUDA code references, EXCLUDING what ds4_cuda_env.cuh provides
// (compute-capability constants, ggml_cuda_device_info, CUDA_CHECK,
// ggml_cuda_get_device / ggml_cuda_set_device / ggml_cuda_info).  That file
// replaced the vendored common.cuh on 2026-08-18; ggml_cuda_pool,
// ggml_cuda_pool_alloc and ggml_backend_cuda_context are gone entirely.
//
// What is left here is a MACRO SHIM: the vendored headers spell their
// assertions, unused-tags and padding the ggml way, and this is where those
// spellings resolve.  It provides no ggml TYPE VOCABULARY at all -- see below.
//
// Things this header DOES provide:
//   * GGML_ASSERT / GGML_ABORT / GGML_UNUSED / GGML_UNUSED_VARS / GGML_PAD
//   * GGML_CUDA_MAX_DEVICES
//   * the CUDA-graph #undefs (ds4 manages its own streams)
//
// Things it does NOT, and why:
//   * enum ggml_type -- REMOVED 2026-09-21.  The 43-code enum existed because
//     the deleted mmq dense dispatcher switched on its case labels; what
//     survived was ds4_mmq_should_use(), whose only callers pass one layout and
//     which now takes the ENGINE's id (PULSAR_TENSOR_IQ2_XXS_MMQ_K).  The
//     engine's own tensor vocabulary had already stopped being ggml's, so
//     keeping a second, foreign one here was a live trap: the callers passed a
//     bare `16` (ggml's IQ2 code) with a comment explaining that "our 43" is not
//     a ggml type at all.
//   * enum ggml_glu_op, struct ggml_tensor, ggml_nbytes() and ggml_time_us() --
//     removed earlier; all four existed only for common.cuh, deleted 2026-08-18
//     (L066).
//   * the inline size traits (ggml_type_size / ggml_blck_size /
//     ggml_is_quantized and their lookup tables) -- removed in the 2026-08-22
//     types sweep (L093).
//   * GGML_MAX_DIMS / GGML_MAX_SRC / GGML_CUDA_NAME / GGML_CUDA_MAX_STREAMS /
//     GGML_LOG_DEBUG -- removed 2026-09-21; a grep over src/cuda/mmq found no
//     reference outside this file.

//   * (the inline size-trait lookups were removed in L093 -- see below)
//
// Things ggml-common.h (vendored) owns:
//   * ggml_half / ggml_half2 typedefs
//   * GGML_EXTENSION macro
//   * block_q*, block_iq* struct definitions

#pragma once

#include <cassert>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// ----------------------------------------------------------------------------
// Macros
// ----------------------------------------------------------------------------

#ifndef GGML_ASSERT
#define GGML_ASSERT(cond) \
    do { if (!(cond)) { \
        fprintf(stderr, "GGML_ASSERT(%s) failed at %s:%d\n", #cond, __FILE__, __LINE__); \
        abort(); \
    } } while (0)
#endif

#ifndef GGML_ABORT
#define GGML_ABORT(fmt, ...) \
    do { \
        fprintf(stderr, "GGML_ABORT: " fmt " at %s:%d\n", ##__VA_ARGS__, __FILE__, __LINE__); \
        abort(); \
    } while (0)
#endif

#ifndef GGML_UNUSED
#define GGML_UNUSED(x) ((void)(x))
#endif

// Variadic GGML_UNUSED_VARS: drop up to 12 unused names without warnings.
#ifndef GGML_UNUSED_VARS
#define GGML_UNUSED_VARS_1(_1)                                             GGML_UNUSED(_1)
#define GGML_UNUSED_VARS_2(_1,_2)                                          GGML_UNUSED(_1); GGML_UNUSED(_2)
#define GGML_UNUSED_VARS_3(_1,_2,_3)                                       GGML_UNUSED_VARS_2(_1,_2); GGML_UNUSED(_3)
#define GGML_UNUSED_VARS_4(_1,_2,_3,_4)                                    GGML_UNUSED_VARS_3(_1,_2,_3); GGML_UNUSED(_4)
#define GGML_UNUSED_VARS_5(_1,_2,_3,_4,_5)                                 GGML_UNUSED_VARS_4(_1,_2,_3,_4); GGML_UNUSED(_5)
#define GGML_UNUSED_VARS_6(_1,_2,_3,_4,_5,_6)                              GGML_UNUSED_VARS_5(_1,_2,_3,_4,_5); GGML_UNUSED(_6)
#define GGML_UNUSED_VARS_7(_1,_2,_3,_4,_5,_6,_7)                           GGML_UNUSED_VARS_6(_1,_2,_3,_4,_5,_6); GGML_UNUSED(_7)
#define GGML_UNUSED_VARS_8(_1,_2,_3,_4,_5,_6,_7,_8)                        GGML_UNUSED_VARS_7(_1,_2,_3,_4,_5,_6,_7); GGML_UNUSED(_8)
#define GGML_UNUSED_VARS_9(_1,_2,_3,_4,_5,_6,_7,_8,_9)                     GGML_UNUSED_VARS_8(_1,_2,_3,_4,_5,_6,_7,_8); GGML_UNUSED(_9)
#define GGML_UNUSED_VARS_10(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10)                GGML_UNUSED_VARS_9(_1,_2,_3,_4,_5,_6,_7,_8,_9); GGML_UNUSED(_10)
#define GGML_UNUSED_VARS_11(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11)            GGML_UNUSED_VARS_10(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10); GGML_UNUSED(_11)
#define GGML_UNUSED_VARS_12(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12)        GGML_UNUSED_VARS_11(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11); GGML_UNUSED(_12)
#define GGML_UNUSED_VARS_PICK(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,NAME,...) NAME
#define GGML_UNUSED_VARS(...) \
    GGML_UNUSED_VARS_PICK(__VA_ARGS__, \
        GGML_UNUSED_VARS_12, GGML_UNUSED_VARS_11, GGML_UNUSED_VARS_10, \
        GGML_UNUSED_VARS_9, GGML_UNUSED_VARS_8, GGML_UNUSED_VARS_7, \
        GGML_UNUSED_VARS_6, GGML_UNUSED_VARS_5, GGML_UNUSED_VARS_4, \
        GGML_UNUSED_VARS_3, GGML_UNUSED_VARS_2, GGML_UNUSED_VARS_1)(__VA_ARGS__)
#endif

#ifndef GGML_PAD
#define GGML_PAD(x, n) (((x) + (n) - 1) / (n) * (n))
#endif

#ifndef GGML_CUDA_MAX_DEVICES
#define GGML_CUDA_MAX_DEVICES 16
#endif

// Cuda-graphs are explicitly disabled - ds4 manages its own streams.
#undef GGML_CUDA_USE_GRAPHS
#undef GGML_HIP_GRAPHS
#undef GGML_MUSA_GRAPHS

// GGML_EXTENSION: ggml-common.h provides the canonical definition. We leave
// it undefined here so the vendored header's `#define GGML_EXTENSION
// __extension__` wins.

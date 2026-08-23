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
// ⚠ SOME OF WHAT IS BELOW IS NOW DEAD.  struct ggml_tensor, ggml_nbytes and
// ggml_glu_op existed for common.cuh's concurrent-event and fusion structs and
// no live code references them any more -- a grep finds ggml_tensor only in
// comments.  Not removed in the same commit that deleted common.cuh, so that
// the deletion stays reviewable; trimming them is the obvious follow-on.
//
// Things this header DOES provide:
//   * GGML_ASSERT / GGML_ABORT / GGML_UNUSED / GGML_UNUSED_VARS / GGML_PAD
//   * GGML_MAX_DIMS / GGML_MAX_SRC / GGML_CUDA_NAME / GGML_CUDA_MAX_DEVICES /
//     GGML_CUDA_MAX_STREAMS / GGML_LOG_DEBUG
//   * enum ggml_type (all 21 mmq type codes - we only USE a subset for V4
//     Flash but the switch in mmq.cu's downstream replacement must compile)
//   * enum ggml_glu_op (just for the unused mm_fusion_args fields)
//   * struct ggml_tensor (complete enough for common.cuh's
//     ggml_cuda_concurrent_event::is_valid() to compile - we never call it)
//   * int64_t ggml_nbytes(const ggml_tensor *) (stub - never called)
//   * int64_t ggml_time_us() (used by USE_CUDA_GRAPH paths we disable)
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

#ifndef GGML_MAX_DIMS
#define GGML_MAX_DIMS 4
#endif

#ifndef GGML_MAX_SRC
#define GGML_MAX_SRC  10
#endif

#ifndef GGML_CUDA_NAME
#define GGML_CUDA_NAME "DS4_CUDA"
#endif

#ifndef GGML_CUDA_MAX_DEVICES
#define GGML_CUDA_MAX_DEVICES 16
#endif

#ifndef GGML_CUDA_MAX_STREAMS
#define GGML_CUDA_MAX_STREAMS 8
#endif

#ifndef GGML_LOG_DEBUG
#define GGML_LOG_DEBUG(...) ((void)0)
#endif

// Cuda-graphs are explicitly disabled - ds4 manages its own streams.
#undef GGML_CUDA_USE_GRAPHS
#undef GGML_HIP_GRAPHS
#undef GGML_MUSA_GRAPHS

// GGML_EXTENSION: ggml-common.h provides the canonical definition. We leave
// it undefined here so the vendored header's `#define GGML_EXTENSION
// __extension__` wins.

// ----------------------------------------------------------------------------
// Quantization type enum.
//
// Order matches llama.cpp's enum ggml_type. Values are pinned because the
// mmq switch uses them as case labels.
// ----------------------------------------------------------------------------

enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 / Q4_3 deprecated
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_MXFP4   = 39,
    GGML_TYPE_NVFP4   = 40,
    GGML_TYPE_Q1_0    = 41,
    GGML_TYPE_Q2_0    = 42,  // added upstream after the 5c0e946 pin (L008)
    GGML_TYPE_COUNT   = 43,
};

/* ggml_glu_op, struct ggml_tensor, ggml_nbytes and ggml_time_us stood here.
 *
 * Every one existed to satisfy common.cuh -- the GLU enum for mm_fusion_args,
 * the tensor for ggml_cuda_concurrent_event::is_valid(), ggml_nbytes for the
 * same, ggml_time_us for USE_CUDA_GRAPH paths we disable.  common.cuh was
 * deleted on 2026-08-18 (L066) and nothing replaced those consumers, so all
 * four went from "compiled but never called" to "not referenced at all": a grep
 * across src/cuda/mmq finds ggml_tensor only inside two comments in ds4_mmq.cu
 * describing what we deliberately do NOT vendor. */

// The inline size traits (ggml_type_size / ggml_blck_size / ggml_is_quantized
// and their q8/K-quant lookup tables) lived here until the 2026-08-22
// launched-vs-defined sweep (L093).  Their last callers -- two dead
// `blck = ggml_blck_size(type)` locals -- were removed in the types sweep the
// same week, leaving the whole cluster reachable from nothing (the mmvq code
// the L008 comment cited was itself removed in L066).

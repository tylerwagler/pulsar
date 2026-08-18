// SPDX-License-Identifier: MIT
// Implementations of the ggml-API stubs declared in ds4_ggml_stubs.h plus
// the bodies of ggml_backend_cuda_context / ggml_cuda_info /
// the device-info accessor that ds4_cuda_env.cuh declares without
// defining.
//
// Phase 0: pool is plain cudaMallocAsync / cudaFreeAsync. Phase 4 swaps
// in ds4's existing cuda_tmp_alloc slab allocator.

#include "ds4_cuda_env.cuh"   // pulls in ds4_ggml_stubs.h via redirect headers

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>

// ----------------------------------------------------------------------------
// Device info singleton.
//
// Common.cuh declares `const ggml_cuda_device_info & ggml_cuda_info();` -
// we provide the body.  The struct now lives in ds4_cuda_env.cuh and is
//   { int device_count; cuda_device_info devices[GGML_CUDA_MAX_DEVICES]; }
// where cuda_device_info has { cc, warp_size, smpbo } -- see the note there on
// what was dropped from common.cuh's thirteen fields.
// ----------------------------------------------------------------------------

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info;
    static std::once_flag once;
    std::call_once(once, []{
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess) {
            fprintf(stderr, "ggml_cuda_info: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
            count = 0;
        }
        if (count > GGML_CUDA_MAX_DEVICES) count = GGML_CUDA_MAX_DEVICES;
        info.device_count = count;
        for (int i = 0; i < count; i++) {
            cudaDeviceProp p;
            CUDA_CHECK(cudaGetDeviceProperties(&p, i));
            /* Three fields, because three fields are read: cc by the MMQ gate
             * and the D2R availability checks, warp_size and smpbo by the
             * mm_ids launchers.  The other ten upstream carries were filled
             * here and read by nobody. */
            info.devices[i].cc        = p.major * 100 + p.minor * 10;
            info.devices[i].warp_size = p.warpSize;
            info.devices[i].smpbo     = p.sharedMemPerBlockOptin;
        }
    });
    return info;
}

// ggml_cuda_get_device / ggml_cuda_set_device are declared (not defined) in
// ds4_cuda_env.cuh. We provide thin wrappers.

int ggml_cuda_get_device() {
    int dev = 0;
    cudaGetDevice(&dev);
    return dev;
}

void ggml_cuda_set_device(int device) {
    int cur = -1;
    cudaGetDevice(&cur);
    if (cur != device) {
        CUDA_CHECK(cudaSetDevice(device));
    }
}

/* ggml_time_us's body was here; its declaration and its only would-be
 * callers (USE_CUDA_GRAPH paths) went with common.cuh. */

// ----------------------------------------------------------------------------
// ggml_cuda_error: invoked by the CUDA_CHECK / CUBLAS_CHECK macros defined
// in ds4_cuda_env.cuh on the error path. Marked [[noreturn]] in the
// declaration there - abort() satisfies that contract.
// ----------------------------------------------------------------------------

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "CUDA error: %s\n  call: %s\n  in: %s at %s:%d\n", msg, stmt, func, file, line);
    fflush(stderr);
    abort();
}

// ----------------------------------------------------------------------------
// Concrete pool wrapping cudaMallocAsync / cudaFreeAsync.
// ----------------------------------------------------------------------------

/* The ggml pool shim lived here: a thread-local stream, ds4_pool_set_stream /
 * ds4_pool_get_stream, a ds4_naive_pool over cudaMallocAsync/cudaFreeAsync, and
 * ggml_backend_cuda_context's pool factory and destructor.
 *
 * All of it existed to give ds4_mmq.cu per-call scratch.  It now takes that
 * scratch from the CUDA_SCRATCH_MMQ arena slot instead -- one persistent
 * reservation, no allocation on the stream -- so the pool has no callers, and
 * with it goes task #22's stream-ordering requirement, which existed only
 * because the allocation WAS on the stream.
 *
 * This is what unhooks ggml_backend_cuda_context, and through it common.cuh
 * (L066 step 3). */


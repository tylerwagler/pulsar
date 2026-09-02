/* GPU-side slab support for the two-rank TP port (slice 4b ground work).
 * See pulsar_tp_gpu.h for the slab-class rationale (runbook step 4 verdict).
 */

#include "tp/pulsar_tp_gpu.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int pulsar_tp_gpu_slab_alloc_hostpin(size_t bytes, void **base,
                                     char *err, size_t errlen) {
    if (!base) {
        if (err && errlen) snprintf(err, errlen, "tp gpu: null base out");
        return 0;
    }
    *base = NULL;
    if (posix_memalign(base, 4096, bytes) != 0 || !*base) {
        if (err && errlen)
            snprintf(err, errlen, "tp gpu: posix_memalign(%zu) failed", bytes);
        return 0;
    }
    const cudaError_t ce = cudaHostRegister(*base, bytes, cudaHostRegisterMapped);
    if (ce != cudaSuccess) {
        if (err && errlen)
            snprintf(err, errlen, "tp gpu: cudaHostRegister(%zu): %s", bytes,
                     cudaGetErrorString(ce));
        free(*base);
        *base = NULL;
        return 0;
    }
    return 1;
}

void pulsar_tp_gpu_slab_free_hostpin(void *base) {
    if (!base) return;
    cudaHostUnregister(base);
    free(base);
}

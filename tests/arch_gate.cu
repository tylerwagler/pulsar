/* Arch gate (L321) -- model-free, links the PRODUCTION objects.
 *
 * A lane whose kernel exists only for some devices decides by the DEVICE (pulsar_cuda_device_major), never by what
 * the image was compiled for.  The case that made this a gate: an sm_75 build on an sm_75 card chose the routed IQ2
 * MMQ lane, whose block-scaled MMA only the sm_120f image has, and its other arm returned NaN.  On whatever card runs
 * this:
 *   1. routed IQ2 MMQ   ds4_mmq_should_use(IQ2_XXS_MMQ_K) == pulsar_cuda_block_scale_mma(major)
 *   2. indexer mxfp4    on a device without the block-scaled MMA, a well-shaped call refuses before it touches its
 *                       pointers (they point at nothing) and leaves no error behind.
 * It names the device and which way each lane went. */
#include "pulsar_gpu.h"
#include "cuda/pulsar_cuda_arch.cuh"
#include "cuda/pulsar_cuda_internal.h"   /* PULSAR_IDX_MXFP4_HEADS_V4 */
#include "cuda/mmq/ds4_mmq.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>

int main(void) {
    if (cudaSetDevice(0) != cudaSuccess) {
        fprintf(stderr, "ARCH FAIL: no GPU\n");
        return 1;
    }
    const int major = pulsar_cuda_device_major();
    const bool bs = pulsar_cuda_block_scale_mma(major);
    printf("arch: device major %d, block-scaled MMA %s\n", major, bs ? "present" : "absent");
    int fail = 0;

    const int mmq = ds4_mmq_should_use((uint32_t)PULSAR_TENSOR_IQ2_XXS_MMQ_K, 8, 256);
    printf("  routed IQ2 MMQ: %s\n", mmq ? "taken" : "declined");
    if (mmq != (bs ? 1 : 0)) {
        fprintf(stderr, "ARCH FAIL: routed IQ2 MMQ %s on a device whose block-scaled MMA is %s\n",
                mmq ? "taken" : "declined", bs ? "present" : "absent");
        fail = 1;
    }

    if (bs) {
        printf("  indexer mxfp4: device has the MMA -- its refusal is not reachable here (the scorer gates cover it)\n");
    } else {
        float *nowhere = (float *)(uintptr_t)0x10;
        const pulsar_mxkv_pack_t *pnowhere = (const pulsar_mxkv_pack_t *)(uintptr_t)0x10;
        const int r = pulsar_gpu_indexer_scores_mxfp4(nowhere, pnowhere, nowhere, pnowhere, 64u, 4u, 0u,
                                                      PULSAR_IDX_MXFP4_HEADS_V4, 128u, 4u, 1.0f, 1);
        const cudaError_t e = cudaDeviceSynchronize();
        printf("  indexer mxfp4: %s, device %s\n", r ? "LAUNCHED" : "refused", e == cudaSuccess ? "clean" : cudaGetErrorString(e));
        if (r || e != cudaSuccess) {
            fprintf(stderr, "ARCH FAIL: indexer mxfp4 on a device without the block-scaled MMA\n");
            fail = 1;
        }
    }

    puts(fail ? "ARCH GATE FAIL" : "ARCH GATE PASS");
    return fail;
}

/* Qwen3.8-Flash-Next (L251 S4): the token embedding into the four residual
 * streams, and the two engine-facing lookups the family ops need -- a weight's
 * device pointer and an EXL3 expert stack's pointer table -- so the ops
 * (src/engine/family_qwen_s4.cpp) stay CUDA-free.  Contracts in
 * pulsar_cuda_qwen.h. */
#include "pulsar_cuda_internal.h"
#include "pulsar_cuda_qwen.h"

#include <cuda_bf16.h>
#include <stdio.h>

namespace {

/* one CTA per token: the embed row, copied into the 4 streams (bf16 -> bf16, exact:
 * the source's `inputs_embeds.repeat(1, 1, hc_count)`) */
__global__ void __launch_bounds__(256)
qwen_embed_kernel(const uint4 *__restrict__ table, const int32_t *__restrict__ tokens,
                  uint4 *__restrict__ streams) {
    constexpr int kRow = PULSAR_QWEN_HIDDEN * 2 / 16;   /* uint4 per 2560-wide bf16 row */
    const int t = blockIdx.x;
    const uint4 *src = table + (size_t)tokens[t] * kRow;
    uint4 *dst = streams + (size_t)t * PULSAR_QWEN_HC * kRow;
    for (int i = threadIdx.x; i < kRow; i += blockDim.x) {
        const uint4 v = src[i];
#pragma unroll
        for (int s = 0; s < PULSAR_QWEN_HC; ++s) dst[s * kRow + i] = v;
    }
}

} // namespace

extern "C" const void *pulsar_gpu_weight_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
    return cuda_model_range_ptr(model_map, offset, bytes, what);
}

extern "C" const void *const *pulsar_qwen_expert_table(const void *stack, uint32_t n_expert, uint64_t stride,
                                                       uint64_t split) {
    return exl3_expert_table(stack, n_expert, stride, split);
}

extern "C" int pulsar_qwen_embed_launch(const uint16_t *table, const int32_t *tokens, int T, int n_vocab,
                                        uint16_t *streams, cudaStream_t stream) {
    if (!table || !tokens || !streams || T <= 0 || n_vocab <= 0 || (((uintptr_t)table | (uintptr_t)streams) & 15u)) {
        fprintf(stderr, "pulsar: qwen embed: a null or misaligned pointer, or no rows -- refusing\n");
        return -1;
    }
    qwen_embed_kernel<<<T, 256, 0, stream>>>((const uint4 *)table, tokens, (uint4 *)streams);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: qwen embed: launch failed: %s\n", cudaGetErrorString(e));
        return -3;
    }
    return 0;
}

/* L446: reference points for Gemma 4's global-layer prompt attention on Turing (not a gate; not in any tier).
 *
 *   mma        the achievable fp16 m16n8k8 (f32 accumulate) tensor-core rate of this device, a loop of independent MMAs
 *   cutlass    the vendored CUTLASS example-41 kernel (src/cuda/fmha/, xformers' memory-efficient attention), Sm75 fp16,
 *              instantiated here at head_dim 512 (kMaxK 512: more than one value iteration, the output accumulated in an
 *              f32 buffer) with the bottom-right causal mask: one 4096-row chunk after `start` cached keys.  Gemma's
 *              grouping (16 q heads on 2 KV heads) is replicated K / V (16 heads): the same FLOPs, 8x the KV bytes --
 *              the kernel is compute-bound at these shapes, so it is a fair ceiling check, not a candidate as is.
 * FLOPs: 4 * n_q * head_dim per (row, key) attended. */
#include "fmha/kernel_forward.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

__global__ void mma_k8_kernel(float *sink, int iters) {
    float d[8][4] = {};
    uint32_t a0 = threadIdx.x * 0x3c003c00u, a1 = a0 ^ 0x1111u, b = 0x3c003c00u ^ threadIdx.x;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++)
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                         : "+f"(d[j][0]), "+f"(d[j][1]), "+f"(d[j][2]), "+f"(d[j][3])
                         : "r"(a0), "r"(a1), "r"(b));
    }
    float s = 0;
    for (int j = 0; j < 8; j++) s += d[j][0] + d[j][1] + d[j][2] + d[j][3];
    if (s == 1234.5f) *sink = s;
}

static double mma_tflops() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    float *sink;
    CK(cudaMalloc(&sink, 4));
    const int iters = 4096, blocks = prop.multiProcessorCount * 8, threads = 256;
    mma_k8_kernel<<<blocks, threads>>>(sink, 16);
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    mma_k8_kernel<<<blocks, threads>>>(sink, iters);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    const double flop = (double)blocks * (threads / 32) * iters * 8 * (2.0 * 16 * 8 * 8);
    CK(cudaFree(sink));
    return flop / (ms * 1e-3) / 1e12;
}

template <int QB, int KB>
static double time_cutlass(uint32_t start, uint32_t rows, int reps) {
    using Attention = AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, QB, KB, 512, false, false>;
    const uint32_t HD = 512, NH = 16, keys = start + rows;
    const uint64_t w = (uint64_t)NH * HD;
    std::vector<__half> h((uint64_t)keys * w);
    for (uint64_t i = 0; i < h.size(); i++) h[i] = __float2half(((int)((i * 2654435761u) >> 20) % 2001 - 1000) * 1e-3f);
    __half *q, *kv, *y;
    float *acc;
    CK(cudaMalloc(&q, (uint64_t)rows * w * 2)); CK(cudaMalloc(&kv, (uint64_t)keys * w * 2));
    CK(cudaMalloc(&y, (uint64_t)rows * w * 2)); CK(cudaMalloc(&acc, (uint64_t)rows * w * 4));
    CK(cudaMemcpy(q, h.data(), (uint64_t)rows * w * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(kv, h.data(), (uint64_t)keys * w * 2, cudaMemcpyHostToDevice));
    typename Attention::Params p;
    p.query_ptr = (cutlass::half_t *)q;
    p.key_ptr = (cutlass::half_t *)kv;
    p.value_ptr = (cutlass::half_t *)kv;   /* Gemma's global layers: K == V */
    p.output_ptr = (cutlass::half_t *)y;
    p.output_accum_ptr = Attention::kNeedsOutputAccumulatorBuffer ? acc : nullptr;
    p.logsumexp_ptr = nullptr;
    p.scale = 1.f;
    p.num_heads = NH; p.num_batches = 1; p.head_dim = HD; p.head_dim_value = HD;
    p.num_queries = rows; p.num_keys = keys;
    p.custom_mask_type = Attention::CausalFromBottomRight;
    p.q_strideH = p.k_strideH = p.v_strideH = HD;
    p.q_strideM = p.k_strideM = p.v_strideM = p.o_strideM = (int32_t)w;
    p.q_strideB = (int64_t)w * rows; p.k_strideB = p.v_strideB = (int64_t)w * keys;
    if (!Attention::check_supported(p)) { fprintf(stderr, "cutlass %dx%d: unsupported\n", QB, KB); exit(2); }
    constexpr auto kernel = attention_kernel_batched_impl<Attention>;
    constexpr int smem = (int)sizeof(typename Attention::SharedStorage);
    if (smem > 0xc000) CK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<p.getBlocksGrid(), p.getThreadsGrid(), smem>>>(p);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int r = 0; r < reps; r++) kernel<<<p.getBlocksGrid(), p.getThreadsGrid(), smem>>>(p);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaFree(q)); CK(cudaFree(kv)); CK(cudaFree(y)); CK(cudaFree(acc));
    return ms / reps;
}

static double flop(uint32_t start, uint32_t rows) {
    double keys = 0;
    for (uint32_t i = start; i < start + rows; i++) keys += i + 1;
    return 4.0 * 16 * 512 * keys;
}

int main() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int khz = 0;
    CK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
    printf("l446_attn_ref_bench: %s, %d SMs, %.2f GHz, L2 %d MB\n", prop.name, prop.multiProcessorCount, khz * 1e-6,
           prop.l2CacheSize >> 20);
    printf("mma.sync m16n8k8 f16 (f32 acc): %.1f TFLOP/s achievable\n", mma_tflops());
    const uint32_t starts[] = {0, 8192, 16384, 32768, 57344};
    for (uint32_t s : starts) {
        const double f = flop(s, 4096);
        const double a = time_cutlass<32, 128>(s, 4096, 3), b = time_cutlass<64, 64>(s, 4096, 3);
        printf("cutlass global start %6u: 32x128 %8.2f ms %6.2f TFLOP/s | 64x64 %8.2f ms %6.2f TFLOP/s\n", s, a,
               f / (a * 1e-3) / 1e12, b, f / (b * 1e-3) / 1e12);
    }
    return 0;
}

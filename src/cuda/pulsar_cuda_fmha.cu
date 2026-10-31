/** @file
 *  Bidirectional fused attention over f16 rows (L321): pulsar's wrapper of CUTLASS example 41. */
/* Bidirectional fused attention over f16 rows (L321) -- see pulsar_cuda_fmha.h for the contract. */
#include "pulsar_cuda_fmha.h"

#include "fmha/kernel_forward.h"

#include <cstdio>

namespace {

/* the Sm75 instance: 64 queries x 128 keys a block (measured on the RTX 5000 at a Parse page, 12298 rows x 16 heads of
 * 72: 61.8 ms; 32 x 128 63.6, 64 x 64 134.6, 32 x 64 155.2), heads up to 128 kept in registers (one value iteration),
 * no dropout, no bias */
/// CUTLASS example 41's AttentionKernel: f16, Sm75, aligned, 64 queries x 128 keys a block, head_dim <= 128, no
/// dropout, no bias.
using Attention = AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 128, false, false>;
static_assert(!Attention::kNeedsOutputAccumulatorBuffer, "the output stays in registers");

/// @param p a pointer.
/// @return true when p is 16-byte aligned.
bool aligned16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

}  // namespace

extern "C" int pulsar_fmha_f16_launch(const uint16_t *q, uint64_t ld_q, const uint16_t *k, uint64_t ld_k,
                                      const uint16_t *v, uint64_t ld_v, uint32_t n_rows, uint32_t n_heads,
                                      uint32_t head_dim, float scale, uint16_t *y, uint64_t ld_y, cudaStream_t stream) {
    const uint64_t w = (uint64_t)n_heads * head_dim;
    if (!q || !k || !v || !y || n_rows == 0 || n_rows > 0x7fffffffu || n_heads == 0 || head_dim == 0 || head_dim % 8 ||
        head_dim > 128 || ld_q < w || ld_k < w || ld_v < w || ld_y < w || ld_q % 8 || ld_k % 8 || ld_v % 8 ||
        ld_y % 8 || !aligned16(q) || !aligned16(k) || !aligned16(v) || !aligned16(y) || !(scale > 0.f)) {
        fprintf(stderr, "pulsar: fmha: %u rows x %u heads of %u (a multiple of 8, at most 128), aligned rows and "
                        "leading dimensions (multiples of 8, >= the heads' width), a positive scale -- refusing\n",
                n_rows, n_heads, head_dim);
        return -1;
    }
    typename Attention::Params p;
    p.query_ptr = (cutlass::half_t *)q;
    p.key_ptr = (cutlass::half_t *)k;
    p.value_ptr = (cutlass::half_t *)v;
    p.output_ptr = (cutlass::half_t *)y;
    p.logsumexp_ptr = nullptr;
    p.output_accum_ptr = nullptr;
    p.scale = scale;
    p.num_heads = (int32_t)n_heads;
    p.num_batches = 1;
    p.head_dim = (int32_t)head_dim;
    p.head_dim_value = (int32_t)head_dim;
    p.num_queries = (int32_t)n_rows;
    p.num_keys = (int32_t)n_rows;
    p.custom_mask_type = Attention::NoCustomMask;
    p.q_strideH = p.k_strideH = p.v_strideH = (int32_t)head_dim;
    p.q_strideM = (int32_t)ld_q;
    p.k_strideM = (int32_t)ld_k;
    p.v_strideM = (int32_t)ld_v;
    p.q_strideB = (int64_t)ld_q * n_rows;
    p.k_strideB = (int64_t)ld_k * n_rows;
    p.v_strideB = (int64_t)ld_v * n_rows;
    p.o_strideM = (int32_t)ld_y;
    if (!Attention::check_supported(p)) {
        fprintf(stderr, "pulsar: fmha: the kernel does not support this call (alignment / strides) -- refusing\n");
        return -1;
    }
    constexpr auto kernel = attention_kernel_batched_impl<Attention>;
    constexpr int smem = (int)sizeof(typename Attention::SharedStorage);
    static bool smem_set = false;
    if (smem > 0xc000 && !smem_set) {
        if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) {
            (void)cudaGetLastError();
            fprintf(stderr, "pulsar: fmha: %d B of shared memory refused by the device -- refusing\n", smem);
            return -1;
        }
        smem_set = true;
    }
    kernel<<<p.getBlocksGrid(), p.getThreadsGrid(), smem, stream>>>(p);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: fmha: launch failed: %s\n", cudaGetErrorString(e));
        return -3;
    }
    return 0;
}

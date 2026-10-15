/* The short convolution's per-(row, channel) arithmetic (L295 E3) -- the ONE
 * authority for it.  The standalone block (pulsar_cuda_sconv.cu) runs it, and
 * an epilogue that fuses the convolution later (the qkv prep, the residual add
 * + norm) calls the same two functions on the same f32 values, so a fused lane
 * stays byte-identical to the block.  Contract in pulsar_cuda_sconv.h. */
#ifndef PULSAR_CUDA_SCONV_CUH
#define PULSAR_CUDA_SCONV_CUH

#include <cuda_bf16.h>
#include <stdint.h>

/** bf16 bits -> f32 (exact). */
__device__ __forceinline__ float pulsar_sconv_bf2f(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }

/** f32 -> bf16 bits, round to nearest even (CUDA's conversion). */
__device__ __forceinline__ uint16_t pulsar_sconv_f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }

/** The tap sum for one (row, channel): hist[0 .. K-2] are the inputs at the
 *  K-1 positions before the row, oldest first (0 before the sequence's
 *  start); x is the row's own input; w[0 .. K-1] the channel's taps.  An f32
 *  fma chain, oldest tap first. */
template <int K>
__device__ __forceinline__ float pulsar_sconv_taps(const float *w, const float *hist, float x) {
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < K - 1; ++i) acc = __fmaf_rn(w[i], hist[i], acc);
    return __fmaf_rn(w[K - 1], x, acc);
}

/** The block's output for one (row, channel) from its tap sum: the residual
 *  (when declared) added in f32, then one bf16 rounding. */
__device__ __forceinline__ uint16_t pulsar_sconv_out(float acc, float x, int residual) {
    return pulsar_sconv_f2bf(residual ? __fadd_rn(acc, x) : acc);
}

#endif /* PULSAR_CUDA_SCONV_CUH */

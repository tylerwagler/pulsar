#ifndef PULSAR_CUDA_MX_CUH
#define PULSAR_CUDA_MX_CUH

#include <cuda_fp8.h>
#include <stdint.h>

/* Producer-side MX (E4M3 data + shared E8M0 block scale) emission.
 *
 * A8 -- every activation reaching a GEMM as E4M3 -- is a PRODUCER-side
 * property.  The kernel that computes an activation already holds it in a
 * register, and with a 256-thread block whose epilogue walks columns as
 * `i = threadIdx.x; i += blockDim.x`, warp w covers columns 32w..32w+31 of each
 * pass: exactly one aligned 32-element MX block.  So the block amax is a warp
 * shuffle the producer can do for free, and the standalone quantize pass that
 * would otherwise read the f32 back is not moved but DELETED.
 *
 * ⚠ MUST MATCH pulsar_cuda_matmul.cu's mx_sfoff / mxfp8_quant_act_kernel
 * EXACTLY.  These epilogues REPLACE that pass, so a divergence in the swizzle
 * or in the exponent arithmetic is a SILENT WRONG-OPERAND bug, not a rounding
 * difference -- the GEMM would read a well-formed value from the wrong slot.
 * This header exists so there is ONE definition to keep in step with the
 * quantiser instead of a copy per producer file; it was extracted from
 * pulsar_cuda_hc_router.cu (b7ba56b), verified inert by SASS compare.
 *
 * CALLER CONTRACT, and it is easy to get wrong:
 *   - ALL 32 LANES of the warp must call this, including lanes whose column is
 *     out of range -- the shuffle reduction is warp-wide.  Pass v = 0.0f for a
 *     dead lane and skip only the data store, exactly as the hc epilogue does.
 *   - The warp's 32 columns must be one ALIGNED block (col % 32 == 0 for lane
 *     0), or the amax belongs to a block this warp does not own.
 *   - The scale slab must be zeroed first: mx_sfoff leaves holes when rows or
 *     blocks are not multiples of 128/4, and a producer that fills only the
 *     live (row, kb) pairs leaves stale swizzle slots for the GEMM to read.
 *     pulsar_gpu_mxfp8_act_cache_e4m3_slot() does that memset. */

__device__ __forceinline__ static int pulsar_mx_sfoff(int row, int kb, int KBp) {
    return ((row / 128) * (KBp / 4) + (kb / 4)) * 512
           + (row % 32) * 16 + ((row % 128) / 32) * 4 + (kb % 4);
}

/* One warp owns one 32-element MX block: shuffle-max |v| to the block amax, then
 * the shared power-of-two exponent, then each lane stores its E4M3 value and
 * lane 0 the E8M0 byte.  Bit-exact against the standalone quantiser -- fmaxf is
 * exact and max is order-independent, so the amax, se and E4M3 bytes are
 * identical however the reduction is ordered.
 *
 * `stride` is the row pitch in ELEMENTS (the GEMM's in_dim), not bytes. */
__device__ __forceinline__ static void pulsar_mx_emit_block(
        float v, uint32_t col, uint32_t row, uint32_t stride, int KBp,
        __nv_fp8_e4m3 *data, unsigned char *scale) {
    float a = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    int se = -127;
    if (a > 0.f) { int e = (int)floorf(log2f(a)); se = e - 7; }
    if (se < -127) se = -127;
    if (se >  127) se =  127;
    data[(size_t)row * stride + col] = (__nv_fp8_e4m3)(v * exp2f((float)-se));
    if ((threadIdx.x & 31u) == 0u) {
        scale[pulsar_mx_sfoff((int)row, (int)(col >> 5), KBp)] = (unsigned char)(se + 127);
    }
}

#endif /* PULSAR_CUDA_MX_CUH */

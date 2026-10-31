/** @file
 *  Block-scaled MXFP8 (e4m3 x e4m3, ue8m0 scales) m16n8k32 MMA for the routed-expert GEMMs. */
#ifndef DS4_MXFP8_MMA_CUH
/// Include guard.
#define DS4_MXFP8_MMA_CUH

#include <cstdint>

/* Block-scaled MXFP8 MMA for the routed-expert GEMMs (A8 campaign, gap B).
 *
 * WHY THIS AND NOT THE INTEGER MMA IT REPLACES.  Upstream computes every GEMM
 * input as MXFP8 -- e4m3 values with a shared ue8m0 (power-of-two) scale per 32
 * -- so the expert GEMM should ISSUE that, rather than reproduce it in another
 * container.  The D2R kernels currently run
 *   mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32          (mma.cuh:946)
 * with q8_1 int8 activations and an s32 accumulator folded to f32 afterwards.
 * int8 is not a lower-precision version of the source format, it is a DIFFERENT
 * one.
 *
 * THE SHAPES ALREADY MATCH, which is what makes the swap tractable: same
 * m16n8k32, same operand register counts (4 A regs, 2 B regs, 4 accumulators),
 * both operands 8-bit -- so the fragment register layout and all the shared
 * memory staging carry over.  What changes is the interpretation (s8 -> e4m3),
 * two added scale operands, and an accumulator that is f32 NATIVELY instead of
 * s32 plus a scale fold.
 *
 * Modelled on the working instance in pulsar_cuda_indexer_mxfp4.cu, which uses
 * .e4m3.e2m1 (e4m3 Q against fp4 K); experts want .e4m3.e4m3. */

/* __CUDA_ARCH__ ALONE IS NOT THE DISCRIMINATOR.  At -arch=sm_120f nvcc runs two
 * device passes and __CUDA_ARCH__ is 1200 in BOTH, so gating on it alone emits
 * this asm into the compute_120 PTX (JIT-fallback) pass, where ptxas rejects it
 * and the whole TU fails to build.  __CUDA_ARCH_FAMILY_SPECIFIC__ is defined
 * ONLY in the sm_120f SASS pass; the other two cover the sm_12Xa targets.
 * Copied deliberately from pulsar_cuda_indexer_mxfp4.cu:50 -- this was learned
 * the hard way once and must not be re-learned. */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) &&                     \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) ||                               \
     defined(__CUDA_ARCH_SPECIFIC__) ||                                      \
     defined(__CUDA_ARCH_FEAT_SM120_ALL))
/// 1 in a device pass with the sm_12x block-scaled MMA (family- or arch-specific target), else 0.
#define DS4_MXFP8_MMA 1
#else
/// 0: this pass has no block-scaled MMA (the host pass, or an older target); ds4_mma_m16n8k32_e4m3 traps.
#define DS4_MXFP8_MMA 0
#endif

/* D = A(e4m3, ue8m0-scaled) x B(e4m3, ue8m0-scaled) + C, f32 accumulate.
 *
 * sfa/sfb carry the ue8m0 scale bytes; the (bid, tid) selector pair picks which
 * byte of the scale register applies to this warp's rows/columns.  scale_vec::1X
 * means one scale factor per k32 block, which is exactly the MX block size, so
 * the hardware applies the same per-32 scale the producer computed.
 *
 * The accumulators are "+f" READ-WRITE and referenced twice in the template,
 * not separate "=f" outputs plus "f" inputs.  That is not style: declaring them
 * separately lets the compiler allocate two accumulator sets and copy between
 * them on every MMA, which measured as pure register churn in the indexer. */
/// D += A B with per-32 ue8m0 scales on both operands (traps without DS4_MXFP8_MMA).
/// @param d0 accumulator 0, in/out.
/// @param d1 accumulator 1, in/out.
/// @param d2 accumulator 2, in/out.
/// @param d3 accumulator 3, in/out.
/// @param a0 A fragment register 0.
/// @param a1 A fragment register 1.
/// @param a2 A fragment register 2.
/// @param a3 A fragment register 3.
/// @param b0 B fragment register 0.
/// @param b1 B fragment register 1.
/// @param sfa A's ue8m0 scale bytes.
/// @param sfb B's ue8m0 scale bytes.
__device__ __forceinline__ static void ds4_mma_m16n8k32_e4m3(
        float &d0, float &d1, float &d2, float &d3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1,
        uint32_t sfa, uint32_t sfb) {
#if DS4_MXFP8_MMA
    const uint16_t bid = 0, tid = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(sfa), "h"(bid), "h"(tid),
          "r"(sfb), "h"(bid), "h"(tid));
#else
    /* Trap, not NaN (L321).  NaN was chosen over zeros so a JITed image could
     * not pass for working, but NaN still RETURNS: on an sm_75 card it flowed
     * through the experts as numbers.  The host now asks the device
     * (pulsar_cuda_block_scale_mma) before choosing this lane; reaching here
     * means that refusal was bypassed, and the kernel stops.  Same choice as
     * pulsar_cuda_indexer_mxfp4.cu's PTX-pass arm. */
    (void)d0; (void)d1; (void)d2; (void)d3;
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
    (void)sfa; (void)sfb;
    __trap();
#endif
}

#endif /* DS4_MXFP8_MMA_CUH */

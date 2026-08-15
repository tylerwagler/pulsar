/* MMQ activation staging in E4M3 + ue8m0 instead of q8_1 int8 (A8 campaign,
 * gap B).
 *
 * WHY.  Upstream computes every GEMM input as MXFP8 -- e4m3 values with a
 * shared power-of-two scale per 32.  q8_1 is not a lower-precision version of
 * that, it is a DIFFERENT format: an affine int8 grid with an f32 scale.  The
 * routed-expert GEMMs are the last place it is used.
 *
 * ⚠ THE STAGING STRUCT IS REUSED VERBATIM, and that is the whole reason this is
 * one kernel rather than a staging rewrite.  block_q8_1_mmq already carries
 * "1 scale per 32 values" (d4[4] over qs[128]) -- exactly the MX block size.  So
 * qs[] holds e4m3 BIT PATTERNS where it held int8, and d4[i] carries the ue8m0
 * byte (as a float, wasting 3 bytes per 32 values to keep the layout bit-for-bit
 * identical).  Nothing downstream moves: load_B_tile's ldmatrix is byte-agnostic
 * and its stride is sizeof(block_q8_1_mmq)-derived, and the prefetch/staging
 * paths are untouched.  Paying 12 bytes per 144-byte block to avoid touching a
 * hand-tuned staging pipeline is the right trade.
 *
 * The quantise arithmetic itself is NOT new: block amax -> se = floor(log2(amax))
 * - 7 -> e4m3 encode is what mxfp8_quant_act_kernel does on the dense path, and
 * it is called here through pulsar_cuda_mx.cuh so the exponent arithmetic keeps
 * one definition.  Only the destination layout differs.
 *
 * ⚠ NOT YET WIRED: nothing calls this.  The D2R kernels still stage q8_1 and
 * still run the integer MMA; this is the shared activation half of the two-arm
 * comparison (MXFP8 e4m3xe4m3 vs f16 containers), and BOTH arms consume it --
 * the f16 arm widens these same e4m3 values losslessly on load. */

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "cuda/pulsar_cuda_mx.cuh"

#include <cuda_fp8.h>

/* One thread per 4 values, mirroring quantize_mmq_q8_1's decomposition so the
 * write indices stay identical.  vals_per_scale is 32 (the MX block), so the
 * amax exchange runs over 32/4 = 8 threads: offsets 4, 2, 1. */
template <bool scatter>
static __global__ void ds4_quantize_mmq_e4m3(
        const float * __restrict__ x, const int32_t * __restrict__ ids,
        void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2, const int n_expert_used) {

    constexpr int vals_per_scale = 32;

    const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) {
        return;
    }
    const int64_t i00 = i0;

    int64_t base_idx;
    if constexpr (scatter) {
        base_idx = (int64_t)blockIdx.x * s02;      /* one physical row per token */
    } else {
        const int64_t i2  = blockIdx.z % ne2;
        const int64_t i3  = blockIdx.z / ne2;
        const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;
        base_idx = i3 * s03 + i2 * s02 + i01 * s01;
    }

    const float4 *x4 = (const float4 *)x;
    block_q8_1_mmq *y = (block_q8_1_mmq *)vy;

    const int64_t k_block = i0 / QK8_1_MMQ;
    const int64_t iqs     = i0 % QK8_1_MMQ;

    const float4 xi = i0 < ne00 ? x4[(base_idx + i00) / 4]
                                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    /* Shared power-of-two exponent, identical to the dense activation path. */
    const int se = pulsar_mx_shared_exp(amax);
    const float r = exp2f(-(float)se);

    uchar4 q;
    q.x = (unsigned char)__nv_cvt_float_to_fp8(xi.x * r, __NV_SATFINITE, __NV_E4M3);
    q.y = (unsigned char)__nv_cvt_float_to_fp8(xi.y * r, __NV_SATFINITE, __NV_E4M3);
    q.z = (unsigned char)__nv_cvt_float_to_fp8(xi.z * r, __NV_SATFINITE, __NV_E4M3);
    q.w = (unsigned char)__nv_cvt_float_to_fp8(xi.w * r, __NV_SATFINITE, __NV_E4M3);

    const int nwrite = scatter ? n_expert_used : 1;
#pragma unroll
    for (int slot = 0; slot < nwrite; ++slot) {
        int64_t ib;
        if constexpr (scatter) {
            const int64_t i = ids[(int64_t)blockIdx.x * n_expert_used + slot];
            ib = k_block * ne1 + i;
        } else {
            const int64_t ib0 = blockIdx.z * ((int64_t)gridDim.x * gridDim.y * blockDim.x / QK8_1);
            ib = ib0 + k_block * ne1 + blockIdx.x;
        }
        uchar4 *yqs4 = (uchar4 *)y[ib].qs;
        yqs4[iqs / 4] = q;
        if (iqs % 32 == 0) {
            /* ue8m0 byte, carried in the f32 scale slot the layout already has.
             * Stored as a float so the struct is untouched; the consumer reads
             * it back with a cast when building the MMA's sfb operand. */
            y[ib].d4[iqs / 32] = (float)pulsar_mx_scale_byte(se);
        }
    }
    GGML_UNUSED(n_expert_used);
}

void ds4_quantize_mmq_e4m3_cuda(
        const float *x, const int32_t *ids, void *vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int n_expert_used, const bool scatter, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % QK8_1_MMQ == 0);
    /* ne1 takes the largest values, so it is the grid's x dimension -- mirrored
     * from quantize_mmq_q8_1_cuda so the write indices above stay valid. */
    const int64_t block_num_y = (ne0 + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) /
                                (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    if (scatter) {
        ds4_quantize_mmq_e4m3<true><<<num_blocks, block_size, 0, stream>>>(
                x, ids, vy, ne00, s01, s02, s03, ne0, (int)ne1, (int)ne2, n_expert_used);
    } else {
        ds4_quantize_mmq_e4m3<false><<<num_blocks, block_size, 0, stream>>>(
                x, ids, vy, ne00, s01, s02, s03, ne0, (int)ne1, (int)ne2, n_expert_used);
    }
}

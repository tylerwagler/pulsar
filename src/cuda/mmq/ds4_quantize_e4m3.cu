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
 * This is the shared activation half of the two-arm comparison (MXFP8 e4m3xe4m3
 * vs f16 containers), and BOTH arms consume it -- the f16 arm widens these same
 * e4m3 values losslessly on load.  It is the ONLY staging the routed-expert path
 * has: the q8_1 fallback that used to sit beside it is gone, so IQ2 experts are
 * E4M3 or they fail.
 *
 * Two entry points, same output bytes.  ds4_quantize_mmq_e4m3_cuda encodes from
 * f32; ds4_gather_mmq_e4m3_cuda copies an encoding the producer already made.
 * Prefer the gather -- see its comment for why it is a copy and not a re-encode. */

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

/* GATHER variant: the rows this is about to stage were ALREADY encoded to E4M3
 * by the norm that produced them, into the activation cache, with the same
 * shared exponent and the same conversion.  Verified, not assumed:
 * pulsar_mx_shared_exp is literally the function the cache uses, and
 * (__nv_fp8_e4m3)(v) vs __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3)
 * agree on 6M values including both infinities, NaN, and the 448/449
 * saturation boundary.  So there is nothing to recompute here -- move the
 * bytes.  Read drops from 4 B/element to 1, and the amax shuffle, the log2 and
 * the encode all disappear.
 *
 * The only thing that changes is WHERE the scale byte comes from: the cache
 * keeps it in the pulsar_mx_sfoff swizzle, this container wants it widened into
 * d4[] as a float.  Read through one, write through the other.
 *
 * Index arithmetic is deliberately identical to the encoding kernel above --
 * same thread-per-4-values decomposition, same k_block/iqs, same ib -- so the
 * two write the same bytes to the same places by construction. */
template <bool scatter>
static __global__ void ds4_gather_mmq_e4m3(
        const uint8_t * __restrict__ src_q, const uint8_t * __restrict__ src_sf,
        const int src_kbp,
        const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2, const int n_expert_used) {

    const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) {
        return;
    }
    const int64_t i00 = i0;

    int64_t src_row;
    if constexpr (scatter) {
        src_row = blockIdx.x;
    } else {
        src_row = ids ? ids[blockIdx.x] : blockIdx.x;
    }

    block_q8_1_mmq *y = (block_q8_1_mmq *)vy;
    const int64_t k_block = i0 / QK8_1_MMQ;
    const int64_t iqs     = i0 % QK8_1_MMQ;

    /* Cache rows are [row][ne00] E4M3, so the 4 bytes this thread owns sit
     * contiguously and 4-aligned -- one uchar4 load. Past ne00 the encoding
     * kernel produced zeros from its zero-filled float4; match that. */
    uchar4 q = make_uchar4(0, 0, 0, 0);
    unsigned char sb = 0;
    if (i0 < ne00) {
        q = *(const uchar4 *)(src_q + (size_t)src_row * ne00 + i00);
        sb = src_sf[pulsar_mx_sfoff((int)src_row, (int)(i00 >> 5), src_kbp)];
    }

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
            y[ib].d4[iqs / 32] = (float)sb;
        }
    }
    GGML_UNUSED(s01); GGML_UNUSED(s02); GGML_UNUSED(s03);
    GGML_UNUSED(ne2); GGML_UNUSED(n_expert_used);
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

/* Same launch geometry as ds4_quantize_mmq_e4m3_cuda -- the two kernels share
 * their index arithmetic, so they must share their grid too. */
void ds4_gather_mmq_e4m3_cuda(
        const void *src_q, const void *src_sf, const int src_kbp,
        const int32_t *ids, void *vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int n_expert_used, const bool scatter, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % QK8_1_MMQ == 0);
    const int64_t block_num_y = (ne0 + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) /
                                (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    if (scatter) {
        ds4_gather_mmq_e4m3<true><<<num_blocks, block_size, 0, stream>>>(
                (const uint8_t *)src_q, (const uint8_t *)src_sf, src_kbp,
                ids, vy, ne00, s01, s02, s03, ne0, (int)ne1, (int)ne2, n_expert_used);
    } else {
        ds4_gather_mmq_e4m3<false><<<num_blocks, block_size, 0, stream>>>(
                (const uint8_t *)src_q, (const uint8_t *)src_sf, src_kbp,
                ids, vy, ne00, s01, s02, s03, ne0, (int)ne1, (int)ne2, n_expert_used);
    }
}

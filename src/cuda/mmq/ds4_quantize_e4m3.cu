/* MMQ activation staging in E4M3 + ue8m0.
 *
 * The routed-expert MMQ GEMMs read block_mx_act_mmq: qs[] holds E4M3 bit
 * patterns and d4[i] carries the block's ue8m0 byte as a float, one scale per
 * 32 values -- the MX block.  The struct is byte-for-byte the q8_1 staging
 * shape (d4[4] over qs[128]), which is what lets load_B_tile's ldmatrix and its
 * sizeof-derived stride stay untouched; 12 spare bytes per 144-byte block buy a
 * hand-tuned staging pipeline that never changes.
 *
 * The bytes are the producer's.  The norm that made this activation encoded it
 * into the activation cache ([row][ne00] E4M3 plus a ue8m0 plane in the
 * pulsar_mx_sfoff swizzle); this file only moves them into the staging
 * container.  There is no encode here, so there is no second definition of the
 * exponent arithmetic to drift. */

#include "ds4_act_block.cuh"
#include "ds4_cuda_env.cuh"
#include "cuda/pulsar_cuda_mx.cuh"

/* One thread per 4 values, the quantize_mmq_q8_1 decomposition, so the write
 * indices match the block placement the D2R staging, prefetch and load_B_tile
 * assume.  The only thing that changes between source and destination is WHERE
 * the scale byte lives: the cache keeps it in the pulsar_mx_sfoff swizzle, this
 * container wants it widened into d4[] as a float. */
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

    block_mx_act_mmq *y = (block_mx_act_mmq *)vy;
    const int64_t k_block = i0 / DS4_ACT_BLOCK_VALS;
    const int64_t iqs     = i0 % DS4_ACT_BLOCK_VALS;

    /* Cache rows are [row][ne00] E4M3, so the 4 bytes this thread owns sit
     * contiguously and 4-aligned -- one uchar4 load.  Past ne00 the staging
     * holds zeros. */
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
            /* Blocks per z-slice: each thread packs 4 values, DS4_ACT_BLOCK_VALS
             * values per block (every live launch has gridDim.z == 1). */
            const int64_t ib0 = blockIdx.z * ((int64_t)gridDim.x * gridDim.y * blockDim.x * 4 / DS4_ACT_BLOCK_VALS);
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

void ds4_gather_mmq_e4m3_cuda(
        const void *src_q, const void *src_sf, const int src_kbp,
        const int32_t *ids, void *vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int n_expert_used, const bool scatter, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % DS4_ACT_BLOCK_VALS == 0);
    /* ne1 takes the largest values, so it is the grid's x dimension -- the
     * quantize_mmq_q8_1_cuda geometry the write indices above assume. */
    const int64_t block_num_y = (ne0 + 4 * DS4_ACT_QUANT_BLOCK - 1) /
                                (4 * DS4_ACT_QUANT_BLOCK);
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(DS4_ACT_QUANT_BLOCK, 1, 1);
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

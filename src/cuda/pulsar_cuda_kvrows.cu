/* The two V4.1 KV row formats (L218): the packers.
 *
 * Every layer keeps a sliding-window ring of WINDOW rows and every kv source a
 * pool of MAIN rows; both hold, byte for byte, what the reference stores after
 * its in-place quantise (the dequantised bf16 values), so a reader that
 * multiplies code by scale in fp32 reproduces the reference's tensor exactly:
 *
 *   WINDOW  act_quant(kv, 32, scale ue8m0):  per 32 dims, amax floored at 1e-4,
 *           scale 2^ceil(log2(amax * (1/448))), E4M3 codes clamped to +-448.
 *           Row = [head_dim E4M3][head_dim/32 E8M0] = 528 B at 512.
 *   MAIN    fp4_act_quant(latent, 16, scale e4m3): per 16 dims, amax floored at
 *           6 * 2^-9, scale e4m3(amax / 6) (RNE, saturating), E2M1 codes
 *           clamped to +-6.  Row = [head_dim/2 E2M1 nibbles][head_dim/16 E4M3]
 *           = 288 B at 512.
 *
 * The E8M0 exponent is derived the way the reference's fast_round_scale
 * derives it (pulsar_e8m0_round_up), the divisions are IEEE (__fdiv_rn) so
 * the RNE ties land where the reference's do, and values are rounded to bf16
 * before quantising: the reference quantises a bf16 tensor and our staging
 * carries fp32.
 *
 * Quantise EXACTLY ONCE, here.  Every later move of a row (ring scatter, fork,
 * evict/restore, session payload) is a byte move; there is no re-encode and no
 * conversion from any other format. */
#include "pulsar_cuda_internal.h"
#include <cuda_fp8.h>

__device__ __forceinline__ static float kvrow_bf16r(float v) { return __bfloat162float(__float2bfloat16(v)); }

/* One block of 512 threads per row: thread d owns dim d. */
__global__ static void winkv_pack_kernel(float *x, const float *src, uint8_t *out,
                                         uint32_t out_row0, uint32_t n_rows, uint32_t head_dim,
                                         const int32_t * __restrict__ positions,
                                         const int32_t * __restrict__ seq_id,
                                         uint32_t n_banks, uint32_t raw_cap) {
    const uint32_t row = blockIdx.x, d = threadIdx.x;
    if (row >= n_rows || d >= head_dim) return;
    const uint64_t dst_row = pulsar_kv_ring_slot(row, out_row0, raw_cap, n_banks, positions, seq_id);
    if (dst_row == PULSAR_KV_RING_DEAD_ROW) return;   /* dead row stores nothing (block-uniform) */
    const float v = kvrow_bf16r(src[(uint64_t)row * head_dim + d]);
    /* per-32 amax over the warp (one warp = one block of 32 dims) */
    float a = fabsf(v);
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 16));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 8));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 4));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
    a = fmaxf(a, 1e-4f);
    const uint32_t e8 = pulsar_e8m0_round_up(a * (1.0f / 448.0f));
    const float s = pulsar_e8m0_scale(e8);
    /* IEEE division whatever the TU's fast-math setting: the reference divides
     * exactly and E4M3's RNE ties sit on exactly representable ratios */
    const __nv_fp8_e4m3 q(fminf(448.0f, fmaxf(-448.0f, __fdiv_rn(v, s))));
    uint8_t *outr = out + dst_row * PULSAR_WINKV_ROWBYTES(head_dim);
    outr[d] = *(const uint8_t *)&q;
    if ((d & 31u) == 0u) outr[head_dim + d / 32u] = (uint8_t)e8;
    if (x) x[(uint64_t)row * head_dim + d] = (float)q * s;
}

/* One block of 512 threads per row: thread d owns dim d; lane pairs pack nibbles. */
__global__ static void mainkv_pack_kernel(float *x, const float *src, uint8_t *out,
                                          uint32_t out_row0, uint32_t n_rows, uint32_t head_dim) {
    const uint32_t row = blockIdx.x, d = threadIdx.x;
    if (row >= n_rows || d >= head_dim) return;
    const float v = kvrow_bf16r(src[(uint64_t)row * head_dim + d]);
    /* per-16 amax over the half-warp */
    float a = fabsf(v);
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 8));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 4));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
    a = fmaxf(a, 6.0f * 0.001953125f);                 /* 6 * 2^-9 */
    const __nv_fp8_e4m3 se(__fdiv_rn(a, 6.0f));        /* RNE, saturating -- T.Cast(FP8, amax / 6) */
    const float s = (float)se;
    const uint8_t nib = dsv4_e2m1fn_encode_dev(fminf(6.0f, fmaxf(-6.0f, __fdiv_rn(v, s))));
    const uint8_t other = (uint8_t)__shfl_xor_sync(0xffffffffu, (uint32_t)nib, 1);
    uint8_t *outr = out + ((uint64_t)out_row0 + row) * PULSAR_MAINKV_ROWBYTES(head_dim);
    if ((d & 1u) == 0u) outr[d >> 1] = (uint8_t)(nib | (other << 4));   /* low nibble = even dim */
    if ((d & 15u) == 0u) outr[head_dim / 2u + d / 16u] = *(const uint8_t *)&se;
    if (x) x[(uint64_t)row * head_dim + d] = dsv4_e2m1fn_decode_dev(nib, s);
}


static bool kvrow_shape_ok(uint32_t head_dim) {
    return head_dim == 512u;   /* one block of 512 threads per row; the graph alloc refuses other shapes */
}

int pulsar_gpu_winkv_pack_tensor(pulsar_gpu_tensor *x, const pulsar_gpu_tensor *src, pulsar_gpu_tensor *packed,
                                 uint32_t out_row0, uint32_t n_rows, uint32_t head_dim,
                                 const pulsar_gpu_tensor *positions, const pulsar_gpu_tensor *seq_id,
                                 uint32_t n_banks, uint32_t raw_cap) {
    if (!src || !packed || n_rows == 0 || !kvrow_shape_ok(head_dim) ||
        src->bytes < (uint64_t)n_rows * head_dim * sizeof(float) ||
        (x && x->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) ||
        (positions && positions->bytes < (uint64_t)n_rows * sizeof(int32_t)) ||
        (seq_id && seq_id->bytes < (uint64_t)n_rows * sizeof(int32_t))) {
        fprintf(stderr, "pulsar: window KV pack: bad operands (rows %u, head_dim %u) -- refusing\n", n_rows, head_dim);
        return 0;
    }
    const uint64_t rows_needed = raw_cap ? (uint64_t)(seq_id ? n_banks : 1u) * raw_cap : (uint64_t)out_row0 + n_rows;
    if (packed->bytes < rows_needed * PULSAR_WINKV_ROWBYTES(head_dim)) {
        fprintf(stderr, "pulsar: window KV pack: destination holds %llu B, %llu rows of %llu B needed -- refusing\n",
                (unsigned long long)packed->bytes, (unsigned long long)rows_needed,
                (unsigned long long)PULSAR_WINKV_ROWBYTES(head_dim));
        return 0;
    }
    winkv_pack_kernel<<<n_rows, head_dim>>>(x ? (float *)x->ptr : NULL, (const float *)src->ptr, (uint8_t *)packed->ptr,
                                            out_row0, n_rows, head_dim,
                                            positions ? (const int32_t *)positions->ptr : NULL,
                                            seq_id ? (const int32_t *)seq_id->ptr : NULL, n_banks, raw_cap);
    return cuda_ok(cudaGetLastError(), "window KV pack launch");
}

int pulsar_gpu_mainkv_pack_tensor(pulsar_gpu_tensor *x, const pulsar_gpu_tensor *src, pulsar_gpu_tensor *packed,
                                  uint32_t out_row0, uint32_t n_rows, uint32_t head_dim) {
    if (!src || !packed || n_rows == 0 || !kvrow_shape_ok(head_dim) ||
        src->bytes < (uint64_t)n_rows * head_dim * sizeof(float) ||
        (x && x->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) ||
        packed->bytes < ((uint64_t)out_row0 + n_rows) * PULSAR_MAINKV_ROWBYTES(head_dim)) {
        fprintf(stderr, "pulsar: main KV pack: bad operands (rows %u+%u, head_dim %u, dst %llu B) -- refusing\n",
                out_row0, n_rows, head_dim, (unsigned long long)(packed ? packed->bytes : 0ull));
        return 0;
    }
    mainkv_pack_kernel<<<n_rows, head_dim>>>(x ? (float *)x->ptr : NULL, (const float *)src->ptr, (uint8_t *)packed->ptr,
                                             out_row0, n_rows, head_dim);
    return cuda_ok(cudaGetLastError(), "main KV pack launch");
}

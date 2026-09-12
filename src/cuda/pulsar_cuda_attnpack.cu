/* 0731's unified NVFP4 attention KV row: the packer and the ring scatter.
 *
 * WHY A SEPARATE TU.  The two-profile engine stores two different KV row
 * families (see pulsar_kv_row_style): V4.1's WINDOW/MAIN pair in
 * pulsar_cuda_kvrows.cu, and 0731's single 384 B NVFP4 row here.  They are
 * siblings selected at ONE place (the dispatchers in pulsar_cuda_norm_kv.cu),
 * not branches interleaved in one kernel, so a change to either family is
 * provably local.  Restored from dev, where this code shipped behind
 * tests/kv4_pack_gate.cpp; that TU has 0 dev commits, so nothing was
 * resurrected stale.
 *
 * WHAT IS THE SAME AS dev AND WHAT IS NOT.  dev's attn_pack_ring_slot was
 * byte-for-byte pulsar_kv_ring_slot, which L218 generalised for both families --
 * so this file calls the shared helper and the shared dead-row sentinel rather
 * than carrying dev's copy.  Everything else (the recipe) is dev's, unchanged:
 * the row scale, the per-16 E4M3 codes, the E2M1 pairs, the bf16 rope tail.
 *
 * THE RECIPE IS THE ORACLE'S.  tests/attn_pack_fixture.h is a host replica of
 * this row and tests/attn_pack_fixture_test.cpp runs 11 checks against it with
 * no device.  tests/kv4_pack_gate.cpp is the device-vs-host gate.  A change
 * here that the fixture does not follow is a change to the format.
 *
 * plans/96-two-profiles-one-engine.md s12. */

#include "pulsar_cuda_internal.h"

/* Native e4m3 conversions.  dev proved these bit-identical to a 7-iteration
 * binary search over all 2^32 finite float bit patterns (4278190080 checked,
 * zero mismatches, every RNE tie and the subnormal range); the ctor IS the
 * cvt.rn.satfinite encoding, so the byte pattern and the round-trip agree by
 * construction.  Static: this TU is their only caller. */
__device__ static uint8_t dsv4_e4m3fn_encode_dev(float x) {
    __nv_fp8_e4m3 f(x);
    return *(const uint8_t *)&f;
}

__device__ static float dsv4_e4m3fn_dequant_dev(float x) {
    return (float)__nv_fp8_e4m3(x);
}

/* Quantise `n_rows` f32 rows of `src` and store them as 0731 unified rows at
 * rows [out_row0, out_row0+n_rows) -- or, when raw_cap != 0, at the ring slot
 * pulsar_kv_ring_slot derives.  `x`, when non-NULL, receives the DECODED values
 * so the f32 staging keeps holding exactly what the packed row decodes to
 * (dev: "the single pack-store fp8 recipe").
 *
 * 64 threads per row.  The per-16 amax is an atomicMax on the int view --
 * non-negative floats order-match their bit patterns, and max is
 * order-independent, so the result is deterministic. */
__global__ static void attn_pack_store_kernel(float *x, const float *src, uint8_t *out,
                                              uint32_t out_row0, uint32_t n_rows,
                                              uint32_t head_dim, uint32_t n_rot,
                                              const int32_t * __restrict__ positions,
                                              const int32_t * __restrict__ seq_id,
                                              uint32_t n_banks, uint32_t raw_cap) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;      /* 64 threads */
    if (row >= n_rows) return;
    const uint64_t dst_row = pulsar_kv_ring_slot(row, out_row0, raw_cap, n_banks,
                                                positions, seq_id);
    if (dst_row == PULSAR_KV_RING_DEAD_ROW) return;   /* dead row stores nothing */
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t nib_bytes = n_nope / 2u;
    const uint32_t nblk = n_nope / PULSAR_KV4_NV_BLOCK;
    const uint64_t rowbytes = PULSAR_ATTN_PACK_ROWBYTES(head_dim);
    const float *sr = src + (uint64_t)row * head_dim;
    float *xr = x ? (x + (uint64_t)row * head_dim) : NULL;
    uint8_t *outr = out + dst_row * rowbytes;
    uint8_t *sc = outr + nib_bytes;
    __shared__ float samax[PULSAR_KV4_NV_NBLK(512u)];   /* 28 at head_dim 512 */
    __shared__ float sscale[PULSAR_KV4_NV_NBLK(512u)];
    __shared__ float srow;

    for (uint32_t bk = tid; bk < nblk; bk += blockDim.x) samax[bk] = 0.0f;
    __syncthreads();
    for (uint32_t d = tid; d < n_nope; d += blockDim.x) {
        atomicMax((int *)&samax[d / PULSAR_KV4_NV_BLOCK], __float_as_int(fabsf(sr[d])));
    }
    __syncthreads();
    if (tid == 0) {
        float ra = 0.0f;
        for (uint32_t bk = 0; bk < nblk; bk++) ra = fmaxf(ra, samax[bk]);
        /* Row scale keyed so every block scale amax/(6*row_scale) fits E4M3's
         * [0, 448]; the 1e-4 amax floor matches the retired fp8 recipe's. */
        const float rs = fmaxf(ra, 1.0e-4f) * (1.0f / (6.0f * 448.0f));
        srow = rs;
        *(float *)(sc + nblk) = rs;   /* 4-aligned: nib 224 + 28 codes */
    }
    __syncthreads();
    for (uint32_t bk = tid; bk < nblk; bk += blockDim.x) {
        /* The DECODED scale (e4m3 roundtrip x row scale) is what both the encode
         * below and every reader use; a round-down clips the block's extremes
         * into the top code -- the standard NVFP4 trade, measured in the L111
         * verdict. */
        const float t = fminf(448.0f, samax[bk] * (1.0f / 6.0f) / srow);
        sc[bk] = dsv4_e4m3fn_encode_dev(t);
        sscale[bk] = dsv4_e4m3fn_dequant_dev(t) * srow;
    }
    __syncthreads();

    /* Nibble pairs: thread t owns packed bytes t, t+64, ... (dims 2t, 2t+1).
     * dsv4_e2m1fn_encode_dev is the tree's ONE reference E2M1 encoder
     * (round-to-nearest, tie to the even code).  A zero block decodes zero
     * whatever its code; guard the quotient so it encodes code 0, not NaN. */
    for (uint32_t i = tid; i < nib_bytes; i += blockDim.x) {
        const uint32_t d0 = i * 2u;
        const float s0 = sscale[d0 / PULSAR_KV4_NV_BLOCK];
        const float s1 = sscale[(d0 + 1u) / PULSAR_KV4_NV_BLOCK];
        const uint32_t v0 = dsv4_e2m1fn_encode_dev(s0 > 0.0f ? sr[d0] / s0 : 0.0f);
        const uint32_t v1 = dsv4_e2m1fn_encode_dev(s1 > 0.0f ? sr[d0 + 1u] / s1 : 0.0f);
        outr[i] = (uint8_t)(v0 | (v1 << 4));
        if (xr) {
            xr[d0]      = attn_kv4_e2m1(v0, s0);
            xr[d0 + 1u] = attn_kv4_e2m1(v1, s1);
        }
    }
    /* bf16 rope tail, roundtripped in place so the f32 staging keeps holding
     * exactly what the packed row decodes to. */
    __nv_bfloat16 *rope = (__nv_bfloat16 *)(outr + nib_bytes + nblk + 4u);
    for (uint32_t d = tid; d < n_rot; d += blockDim.x) {
        const __nv_bfloat16 hb = __float2bfloat16(sr[n_nope + d]);
        rope[d] = hb;
        if (xr) xr[n_nope + d] = __bfloat162float(hb);
    }
}

/* Quantise + store n_rows rows into a contiguous run (no ring): the form the
 * comp pool and the current-chunk pack buffer use. */
int pulsar_gpu_attn_pack_store_tensor(pulsar_gpu_tensor *x, const pulsar_gpu_tensor *src,
                                      pulsar_gpu_tensor *packed, uint32_t out_row0,
                                      uint32_t n_rows, uint32_t head_dim) {
    if (!src || !packed || n_rows == 0 ||
        head_dim != 512u ||   /* the shared samax/sscale are sized for 28 blocks */
        ((head_dim - PULSAR_ATTN_PACK_NROT) % PULSAR_KV4_NV_BLOCK) != 0 ||
        src->bytes < (uint64_t)n_rows * head_dim * sizeof(float) ||
        (x && x->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) ||
        packed->bytes < ((uint64_t)out_row0 + n_rows) * PULSAR_ATTN_PACK_ROWBYTES(head_dim)) {
        fprintf(stderr, "pulsar: 0731 KV pack: bad operands (rows %u, head_dim %u) -- refusing\n",
                n_rows, head_dim);
        return 0;
    }
    attn_pack_store_kernel<<<n_rows, 64>>>(x ? (float *)x->ptr : NULL,
                                           (const float *)src->ptr,
                                           (uint8_t *)packed->ptr,
                                           out_row0, n_rows, head_dim, PULSAR_ATTN_PACK_NROT,
                                           NULL, NULL, 0u, 0u);
    return cuda_ok(cudaGetLastError(), "0731 KV pack launch");
}

/* Quantise + store one row into a ring slot (raw_cap != 0): the single-row
 * decode store.  seq_id/positions are NULL -- the slot is `row % raw_cap`. */
int pulsar_gpu_attn_pack_ring_store_tensor(pulsar_gpu_tensor *raw_cache, const pulsar_gpu_tensor *kv,
                                           uint32_t raw_cap, uint32_t row, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * PULSAR_ATTN_PACK_ROWBYTES(head_dim) ||
        kv->bytes < (uint64_t)head_dim * sizeof(float)) {
        return 0;
    }
    /* x = NULL: kv is const here, so the row is packed WITHOUT the in-place
     * round-trip the decode store does. */
    attn_pack_store_kernel<<<1, 64>>>(NULL, (const float *)kv->ptr,
                                      (uint8_t *)raw_cache->ptr,
                                      row, 1u, head_dim, PULSAR_ATTN_PACK_NROT,
                                      NULL, NULL, 1u, raw_cap);
    return cuda_ok(cudaGetLastError(), "0731 raw pack store launch");
}

/* Quantise + store a batch into banked ring slots: the prefill store. */
int pulsar_gpu_attn_pack_ring_store_batch_tensor(pulsar_gpu_tensor *raw_cache, const pulsar_gpu_tensor *kv,
                                                 uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens,
                                                 uint32_t head_dim, const pulsar_gpu_tensor *positions,
                                                 const pulsar_gpu_tensor *seq_id, uint32_t n_banks) {
    const bool descr = positions != NULL;
    if (!raw_cache || !kv || raw_cap == 0 ||
        head_dim != 512u ||
        raw_cache->bytes < (descr ? n_banks : 1u) * (uint64_t)raw_cap * PULSAR_ATTN_PACK_ROWBYTES(head_dim) ||
        kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float)) {
        return 0;
    }
    if (n_tokens == 0) return 1;
    attn_pack_store_kernel<<<n_tokens, 64>>>(NULL, (const float *)kv->ptr,
                                             (uint8_t *)raw_cache->ptr,
                                             pos0, n_tokens, head_dim, PULSAR_ATTN_PACK_NROT,
                                             descr ? (const int32_t *)positions->ptr : NULL,
                                             descr ? (const int32_t *)seq_id->ptr : NULL,
                                             descr ? n_banks : 1u, raw_cap);
    return cuda_ok(cudaGetLastError(), "0731 raw pack batch launch");
}

/* Scatter ALREADY-PACKED rows into the ring: a pure byte move.  Shares
 * pulsar_kv_ring_slot with attn_pack_store_kernel, so the destination rule
 * cannot drift between the two.  Copying bytes (rather than re-quantising) is
 * what makes the ring hold exactly what attention read. */
__global__ static void attn_pack_scatter_kernel(const uint8_t *__restrict__ src, uint8_t *out,
                                                uint32_t out_row0, uint32_t n_rows,
                                                uint32_t head_dim,
                                                const int32_t *__restrict__ positions,
                                                const int32_t *__restrict__ seq_id,
                                                uint32_t n_banks, uint32_t raw_cap) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    const uint64_t dst_row = pulsar_kv_ring_slot(row, out_row0, raw_cap, n_banks,
                                                positions, seq_id);
    if (dst_row == PULSAR_KV_RING_DEAD_ROW) return;   /* dead row stores nothing */
    const uint64_t rowbytes = PULSAR_ATTN_PACK_ROWBYTES(head_dim);
    const uint8_t *sr = src + (uint64_t)row * rowbytes;
    uint8_t *dr = out + dst_row * rowbytes;
    for (uint32_t b = threadIdx.x; b < (uint32_t)rowbytes; b += blockDim.x) dr[b] = sr[b];
}

int pulsar_gpu_attn_pack_scatter_tensor(pulsar_gpu_tensor *raw_cache, const pulsar_gpu_tensor *packed,
                                        uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens,
                                        uint32_t head_dim, const pulsar_gpu_tensor *positions,
                                        const pulsar_gpu_tensor *seq_id, uint32_t n_banks) {
    const bool descr = positions != NULL || seq_id != NULL;
    if (descr &&
        (!positions || !seq_id || n_banks == 0 ||
         positions->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         seq_id->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         (uint64_t)n_banks * raw_cap > 4294967296ull)) {
        fprintf(stderr,
                "pulsar: banked 0731 packed raw store rejected: bad descriptor args "
                "(n_tokens=%u n_banks=%u raw_cap=%u)\n",
                n_tokens, n_banks, raw_cap);
        return 0;
    }
    const uint64_t kv_banks = descr ? n_banks : 1u;
    const uint64_t rowbytes = PULSAR_ATTN_PACK_ROWBYTES(head_dim);
    if (!raw_cache || !packed || raw_cap == 0 ||
        head_dim <= PULSAR_ATTN_PACK_NROT ||
        ((head_dim - PULSAR_ATTN_PACK_NROT) % PULSAR_KV4_NV_BLOCK) != 0 ||
        raw_cache->bytes < kv_banks * raw_cap * rowbytes ||
        packed->bytes < (uint64_t)n_tokens * rowbytes) return 0;
    if (n_tokens == 0) return 1;
    attn_pack_scatter_kernel<<<n_tokens, 64>>>((const uint8_t *)packed->ptr,
                                               (uint8_t *)raw_cache->ptr,
                                               pos0, n_tokens, head_dim,
                                               descr ? (const int32_t *)positions->ptr : NULL,
                                               descr ? (const int32_t *)seq_id->ptr : NULL,
                                               descr ? n_banks : 1u, raw_cap);
    return cuda_ok(cudaGetLastError(), "0731 raw pack scatter launch");
}

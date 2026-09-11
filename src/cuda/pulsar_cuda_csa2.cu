/* CSA2 compressor kernels (L218, DeepSeek-V4.1).  Their own translation unit so
 * the kernel unit test (tests/csa2_compressor_kernel_test.cu) can #include the
 * shipped code and drive the REAL kernels against a CPU oracle with two runtime
 * symbols stubbed, the pattern tests/idx_mxfp4_kernel_test.cu set. */
#include "pulsar_cuda_internal.h"

/* ============================================================================
 * CSA2 compressor (L218, DeepSeek-V4.1): the reference's Compressor.forward.
 *
 * A group of `ratio` tokens becomes one latent: softmax over the group's score
 * projections, weighted sum of its kv projections (both fp32, the reference
 * promotes the ratio-2 weights to fp32), the pooled row ROUNDED TO BF16, then
 * RMSNorm against attn_compressor_norm in fp32 with the weighted result rounded
 * to bf16 again -- `self.norm(kv.to(dtype))`.  At ratio 1 there is no pooling:
 * the projection (a bf16 Linear in the reference, so rounded to bf16 here) is
 * normed.  The latent leaves this kernel PRE-RoPE as fp32 holding bf16-exact
 * values: the index-key projection reads it unrotated and the attention row
 * pack rotates its own copy, exactly as the reference orders those two.
 *
 * The softmax is written in the reference's order: normalise the weights
 * first, then sum kv * weight (torch's `(kv * score.softmax(dim)).sum(dim)`),
 * not (sum kv * e) / den -- the two differ in the last bit.
 *
 * One block per emitted latent row, 256 threads over head_dim (512 -> two
 * elements per thread); `src` selects where the group's rows come from:
 *   SRC_ROWS   the group is rows [g*ratio, g*ratio+ratio) of kv/sc (prefill)
 *   SRC_STATE  the group is the state lane's `ratio` rows (a decode emit)
 * ============================================================================ */
enum { CSA2_SRC_ROWS = 0, CSA2_SRC_STATE = 1 };

__device__ __forceinline__ static float csa2_bf16_round(float v) {
    return __bfloat162float(__float2bfloat16(v));   /* RNE, the reference's .to(bf16) */
}

template <bool NORM_BF16>
__global__ static void csa2_compressor_pool_norm_kernel(
        float *latent,            /* [n_groups][head_dim] out */
        const float *kv,          /* SRC_ROWS: [n_groups * ratio][head_dim]; SRC_STATE: unused */
        const float *sc,
        const float *state_kv,    /* SRC_STATE: [ratio][head_dim] */
        const float *state_sc,
        const void *norm_w,       /* attn_compressor_norm [head_dim] */
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t n_groups,
        uint32_t src,
        float eps) {
    const uint32_t g = blockIdx.x;
    if (g >= n_groups) return;
    __shared__ float red[256];
    float pooled[4];              /* head_dim <= 4 * blockDim.x */
    float sumsq = 0.0f;
    uint32_t k = 0;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x, k++) {
        float v;
        if (ratio == 1u) {
            /* a bf16 Linear's output: the fp32 GEMM result rounded once */
            v = csa2_bf16_round(kv[(uint64_t)g * head_dim + d]);
        } else {
            const float *gk = src == CSA2_SRC_STATE ? state_kv : kv + (uint64_t)g * ratio * head_dim;
            const float *gs = src == CSA2_SRC_STATE ? state_sc : sc + (uint64_t)g * ratio * head_dim;
            float m = -INFINITY;
            for (uint32_t r = 0; r < ratio; r++) m = fmaxf(m, gs[(uint64_t)r * head_dim + d]);
            float den = 0.0f;
            for (uint32_t r = 0; r < ratio; r++) den += expf(gs[(uint64_t)r * head_dim + d] - m);
            float acc = 0.0f;
            for (uint32_t r = 0; r < ratio; r++) {
                const float w = expf(gs[(uint64_t)r * head_dim + d] - m) / den;
                acc += gk[(uint64_t)r * head_dim + d] * w;
            }
            v = csa2_bf16_round(acc);
        }
        pooled[k] = v;
        sumsq += v * v;
    }
    /* RMSNorm.forward: var = mean(x^2) over head_dim, x * rsqrt(var + eps),
     * (weight * x).to(bf16).  Block tree reduction, fixed order. */
    red[threadIdx.x] = sumsq;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)head_dim + eps);
    k = 0;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x, k++) {
        const float w = pulsar_w_load_f32_or_bf16<NORM_BF16>(norm_w, d);
        latent[(uint64_t)g * head_dim + d] = csa2_bf16_round(w * (pooled[k] * inv));
    }
}

/* Store `n_tokens` positions' kv / score rows into the state lane at slot
 * (pos0 + t) %% ratio.  No pooling, no shift: the slot layout IS the group. */
__global__ static void csa2_compressor_store_kernel(
        float *state_kv, float *state_sc,
        const float *kv, const float *sc,
        uint32_t head_dim, uint32_t ratio, uint32_t pos0, uint32_t src_row0, uint32_t n_tokens) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * head_dim;
    if (gid >= n) return;
    const uint32_t t = (uint32_t)(gid / head_dim);
    const uint32_t d = (uint32_t)(gid - (uint64_t)t * head_dim);
    const uint32_t slot = (pos0 + t) % ratio;
    state_kv[(uint64_t)slot * head_dim + d] = kv[(uint64_t)(src_row0 + t) * head_dim + d];
    state_sc[(uint64_t)slot * head_dim + d] = sc[(uint64_t)(src_row0 + t) * head_dim + d];
}

static bool csa2_norm_args_ok(const void *model_map, uint64_t model_size, uint64_t norm_offset,
                              uint32_t norm_type, uint32_t head_dim, const void **norm_w) {
    /* ds4 types: 0 = F32, 30 = BF16 (source format). */
    if (!model_map || (norm_type != 0u && norm_type != 30u) || head_dim == 0 || head_dim > 1024u) return false;
    const uint64_t norm_bytes = (uint64_t)head_dim * pulsar_w_elt_bytes(norm_type == 30u);
    if (norm_offset > model_size || norm_bytes > model_size - norm_offset) return false;
    *norm_w = cuda_model_range_ptr(model_map, norm_offset, norm_bytes, "attn_compressor_norm");
    return *norm_w != NULL;
}

static int csa2_pool_norm_launch(float *latent, const float *kv, const float *sc,
                                 const float *state_kv, const float *state_sc, const void *norm_w,
                                 int norm_bf16, uint32_t head_dim, uint32_t ratio, uint32_t n_groups,
                                 uint32_t src, float eps) {
    if (n_groups == 0) return 1;
    if (norm_bf16)
        csa2_compressor_pool_norm_kernel<true><<<n_groups, 256>>>(latent, kv, sc, state_kv, state_sc, norm_w,
                                                                  head_dim, ratio, n_groups, src, eps);
    else
        csa2_compressor_pool_norm_kernel<false><<<n_groups, 256>>>(latent, kv, sc, state_kv, state_sc, norm_w,
                                                                   head_dim, ratio, n_groups, src, eps);
    return cuda_ok(cudaGetLastError(), "csa2 compressor pool+norm launch");
}

int pulsar_gpu_csa2_compressor_prefill_tensor(
        pulsar_gpu_tensor       *latent,
        const pulsar_gpu_tensor *kv,
        const pulsar_gpu_tensor *sc,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        float                   rms_eps) {
    const void *norm_w = NULL;
    if (!latent || !kv || ratio == 0 || n_tokens == 0 ||
        !csa2_norm_args_ok(model_map, model_size, norm_offset, norm_type, head_dim, &norm_w)) {
        fprintf(stderr, "pulsar: csa2 compressor prefill: bad arguments (ratio %u, n_tokens %u, norm type %u) -- refusing\n",
                ratio, n_tokens, norm_type);
        return 0;
    }
    if (pos0 % ratio != 0u) {
        fprintf(stderr, "pulsar: csa2 compressor prefill: position %u is not a group boundary (ratio %u) -- refusing\n",
                pos0, ratio);
        return 0;
    }
    const uint32_t n_groups = n_tokens / ratio;
    const uint32_t rem = n_tokens - n_groups * ratio;
    const uint64_t row_bytes = (uint64_t)head_dim * sizeof(float);
    if (kv->bytes < (uint64_t)n_tokens * row_bytes || latent->bytes < (uint64_t)n_groups * row_bytes ||
        (ratio > 1u && (!sc || sc->bytes < (uint64_t)n_tokens * row_bytes))) {
        fprintf(stderr, "pulsar: csa2 compressor prefill: operand too small (kv %llu B for %u rows) -- refusing\n",
                (unsigned long long)kv->bytes, n_tokens);
        return 0;
    }
    if (ratio > 1u) {
        const uint64_t state_bytes = (uint64_t)ratio * row_bytes;
        if (!state_kv || !state_score || state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
            fprintf(stderr, "pulsar: csa2 compressor prefill: state lane missing or too small at ratio %u -- refusing\n", ratio);
            return 0;
        }
        /* The chunk starts on a group boundary, so the state it leaves is
         * exactly its trailing partial group: reset, then store the remainder. */
        const uint64_t state_n = (uint64_t)ratio * head_dim;
        if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))), "csa2 state kv zero")) return 0;
        fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
        if (!cuda_ok(cudaGetLastError(), "csa2 state score fill launch")) return 0;
        if (rem) {
            const uint64_t n = (uint64_t)rem * head_dim;
            csa2_compressor_store_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    head_dim, ratio, pos0 + n_groups * ratio, n_groups * ratio, rem);
            if (!cuda_ok(cudaGetLastError(), "csa2 state store launch")) return 0;
        }
    }
    return csa2_pool_norm_launch((float *)latent->ptr, (const float *)kv->ptr,
                                 ratio > 1u ? (const float *)sc->ptr : NULL, NULL, NULL, norm_w,
                                 norm_type == 30u, head_dim, ratio, n_groups, CSA2_SRC_ROWS, rms_eps);
}

int pulsar_gpu_csa2_compressor_update_tensor(
        pulsar_gpu_tensor       *latent,
        const pulsar_gpu_tensor *kv_cur,
        const pulsar_gpu_tensor *sc_cur,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        float                   rms_eps,
        int                    *emitted) {
    const void *norm_w = NULL;
    if (emitted) *emitted = 0;
    if (!latent || !kv_cur || !emitted || ratio == 0 ||
        !csa2_norm_args_ok(model_map, model_size, norm_offset, norm_type, head_dim, &norm_w)) {
        fprintf(stderr, "pulsar: csa2 compressor update: bad arguments (ratio %u, norm type %u) -- refusing\n", ratio, norm_type);
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)head_dim * sizeof(float);
    if (kv_cur->bytes < row_bytes || latent->bytes < row_bytes) return 0;
    if (ratio == 1u) {
        *emitted = 1;
        return csa2_pool_norm_launch((float *)latent->ptr, (const float *)kv_cur->ptr, NULL, NULL, NULL, norm_w,
                                     norm_type == 30u, head_dim, 1u, 1u, CSA2_SRC_ROWS, rms_eps);
    }
    if (!pulsar_gpu_csa2_compressor_store_tensor(kv_cur, sc_cur, state_kv, state_score, head_dim, ratio, pos)) return 0;
    if ((pos + 1u) % ratio != 0u) return 1;   /* the group is still filling */
    *emitted = 1;
    if (!csa2_pool_norm_launch((float *)latent->ptr, NULL, NULL,
                               (const float *)state_kv->ptr, (const float *)state_score->ptr, norm_w,
                               norm_type == 30u, head_dim, ratio, 1u, CSA2_SRC_STATE, rms_eps)) return 0;
    /* The group is consumed: leave the canonical empty state (kv 0, score
     * -inf), the state a boundary-aligned prefill leaves too, so "empty at a
     * group boundary" is literally true on every path (rewind, resume, the
     * comp-state gate's prefill-vs-decode comparison), not merely equivalent
     * up to slots the next group would overwrite. */
    const uint64_t state_n = (uint64_t)ratio * head_dim;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))), "csa2 state kv clear")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    return cuda_ok(cudaGetLastError(), "csa2 state score clear launch");
}

int pulsar_gpu_csa2_compressor_store_tensor(
        const pulsar_gpu_tensor *kv_row,
        const pulsar_gpu_tensor *sc_row,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos) {
    const uint64_t row_bytes = (uint64_t)head_dim * sizeof(float);
    if (!kv_row || !sc_row || !state_kv || !state_score || head_dim == 0 || ratio < 2u ||
        kv_row->bytes < row_bytes || sc_row->bytes < row_bytes ||
        state_kv->bytes < (uint64_t)ratio * row_bytes || state_score->bytes < (uint64_t)ratio * row_bytes) {
        fprintf(stderr, "pulsar: csa2 compressor store: bad operands (ratio %u, head_dim %u) -- refusing\n", ratio, head_dim);
        return 0;
    }
    const uint64_t n = head_dim;
    csa2_compressor_store_kernel<<<(n + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv_row->ptr, (const float *)sc_row->ptr,
            head_dim, ratio, pos, 0u, 1u);
    return cuda_ok(cudaGetLastError(), "csa2 compressor store launch");
}

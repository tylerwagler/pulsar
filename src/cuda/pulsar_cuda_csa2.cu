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
        const float *kv,          /* SRC_ROWS: [n_groups * ratio][coff * head_dim] */
        const float *sc,
        const float *state_kv,    /* SRC_STATE / overlap: [coff * ratio][coff * head_dim] */
        const float *state_sc,
        const void *norm_w,       /* attn_compressor_norm [head_dim] */
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t coff,
        uint32_t n_groups,
        uint32_t src,
        uint32_t pos0,            /* SRC_ROWS only: where this batch starts */
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
        } else if (coff == 2u) {
            /* The overlap case, the reference's overlap_transform.  coff 2 is
             * exactly ratio 4 (pulsar_compress_coff), so there are always
             * 2*ratio = 8 positions.  Position p < ratio comes from the PREVIOUS
             * group's row p read at its FIRST half; p >= ratio from THIS group's
             * row p-ratio read at its SECOND half -- the 2*head_dim projection is
             * split, which is the whole point of the mode.
             *
             * WHERE those rows live depends on `src`, and the two callers differ
             * exactly there:
             *   SRC_ROWS  (batched prefill): a rows array indexed by group, with
             *             the previous group at g-1.  Group 0 has no previous
             *             group, and what stands in for it depends on where the
             *             batch starts: at the sequence start the reference PADS
             *             (kv 0, score -inf), mid-sequence it is the state CARRY.
             *             That is why pos0 has to reach this kernel; inferring it
             *             from the state alone cannot tell the two apart.
             *   SRC_STATE (per-token update): there is no rows array at all --
             *             kv/sc are NULL -- because THIS group is the lane's own
             *             current half.  Position p is then simply lane row p, at
             *             its first half (p < ratio, the carry the last shift
             *             left) or its second half (p >= ratio, this group).
             *             n_groups is 1 here, so g is always 0.
             * Reading `kv` unconditionally is what a first version did; on the
             * update path that is a NULL dereference, and it only ever ran on a
             * device. */
            const uint32_t width = 2u * head_dim;
            const uint32_t npos = 2u * ratio;
            const bool from_state = src == CSA2_SRC_STATE;
            float sk[8], ss[8];
            for (uint32_t p = 0; p < npos; p++) {
                const uint32_t r = p % ratio;
                if (from_state) {
                    if (p < ratio && pos0 == 0u) {
                        sk[p] = 0.0f;               /* the sequence start: pad */
                        ss[p] = -INFINITY;
                    } else {
                        const uint32_t off = p < ratio ? 0u : head_dim;
                        sk[p] = state_kv[(uint64_t)p * width + off + d];
                        ss[p] = state_sc[(uint64_t)p * width + off + d];
                    }
                } else if (p < ratio) {
                    if (g > 0u) {
                        sk[p] = kv[(uint64_t)((g - 1u) * ratio + r) * width + d];
                        ss[p] = sc[(uint64_t)((g - 1u) * ratio + r) * width + d];
                    } else if (pos0 == 0u) {
                        sk[p] = 0.0f;
                        ss[p] = -INFINITY;
                    } else {
                        sk[p] = state_kv[(uint64_t)r * width + d];
                        ss[p] = state_sc[(uint64_t)r * width + d];
                    }
                } else {
                    sk[p] = kv[(uint64_t)(g * ratio + r) * width + head_dim + d];
                    ss[p] = sc[(uint64_t)(g * ratio + r) * width + head_dim + d];
                }
            }
            float m = -INFINITY;
            for (uint32_t p = 0; p < npos; p++) m = fmaxf(m, ss[p]);
            float den = 0.0f;
            for (uint32_t p = 0; p < npos; p++) den += expf(ss[p] - m);
            float acc = 0.0f;
            for (uint32_t p = 0; p < npos; p++) acc += sk[p] * (expf(ss[p] - m) / den);
            v = csa2_bf16_round(acc);
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

/* Add each position's within-group ape row to its score projection.
 *
 * Every one of the reference's four ape uses is `ape[position % ratio]` -- the
 * carry stash (positions cutoff-ratio..cutoff-1), the partial stash (`ape
 * [:remainder]`), the batched pool (`+ self.ape` over unflattened groups) and
 * the decode add -- so folding it in ONCE, here, on the score rows, is exactly
 * equivalent to doing it at each site, and it keeps the compressor kernels free
 * of a parameter they would each have to apply in the same way.
 *
 * Width is coff * head_dim: the ape is as wide as the projection it biases. */
template <bool APE_BF16>
__global__ static void csa2_comp_ape_add_kernel(
        float *sc, const void *ape, uint32_t width, uint32_t ratio, uint32_t pos0, uint32_t n_tokens) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * width;
    if (gid >= n) return;
    const uint32_t t = (uint32_t)(gid / width);
    const uint32_t d = (uint32_t)(gid - (uint64_t)t * width);
    sc[gid] += pulsar_w_load_f32_or_bf16<APE_BF16>(ape, (uint64_t)((pos0 + t) % ratio) * width + d);
}

/* Store `n_tokens` positions' kv / score rows into the state lane starting at
 * `slot_base + (pos0 + t) %% ratio`.  No pooling, no shift.
 *
 * slot_base is what separates the two lane conventions:
 *   coff 1 (V4.1): the lane IS the group; slot_base 0, slot = pos % ratio, and
 *                  the caller resets the lane between groups.
 *   coff 2 (V4):   the lane is TWO groups -- rows 0..ratio-1 the carry from the
 *                  previous group, rows ratio..2ratio-1 the group being filled --
 *                  so a running token stores at slot_base = ratio and the
 *                  prefill's carry stores at slot_base = 0, both of which the
 *                  caller names explicitly rather than the kernel inferring. */
__global__ static void csa2_compressor_store_kernel(
        float *state_kv, float *state_sc,
        const float *kv, const float *sc,
        uint32_t head_dim, uint32_t ratio, uint32_t coff, uint32_t slot_base,
        uint32_t pos0, uint32_t src_row0, uint32_t n_tokens) {
    const uint32_t width = coff * head_dim;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * width;
    if (gid >= n) return;
    const uint32_t t = (uint32_t)(gid / width);
    const uint32_t d = (uint32_t)(gid - (uint64_t)t * width);
    const uint32_t slot = slot_base + (pos0 + t) % ratio;
    state_kv[(uint64_t)slot * width + d] = kv[(uint64_t)(src_row0 + t) * width + d];
    state_sc[(uint64_t)slot * width + d] = sc[(uint64_t)(src_row0 + t) * width + d];
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
                                 int norm_bf16, uint32_t head_dim, uint32_t ratio, uint32_t coff,
                                 uint32_t n_groups, uint32_t src, uint32_t pos0, float eps) {
    if (n_groups == 0) return 1;
    if (norm_bf16)
        csa2_compressor_pool_norm_kernel<true><<<n_groups, 256>>>(latent, kv, sc, state_kv, state_sc, norm_w,
                                                                  head_dim, ratio, coff, n_groups, src, pos0, eps);
    else
        csa2_compressor_pool_norm_kernel<false><<<n_groups, 256>>>(latent, kv, sc, state_kv, state_sc, norm_w,
                                                                   head_dim, ratio, coff, n_groups, src, pos0, eps);
    return cuda_ok(cudaGetLastError(), "csa2 compressor pool+norm launch");
}

/* Shift the overlap lane's current half down into the carry half: the reference
 * does `kv_state[:ratio] = kv_state[ratio:]` after a group completes.  The two
 * halves are contiguous and disjoint, so this is one device-to-device copy per
 * plane and no kernel. */
static int csa2_lane_clear_current(float *state_kv, float *state_sc, uint32_t head_dim, uint32_t ratio);

static int csa2_overlap_shift(float *state_kv, float *state_sc, uint32_t head_dim, uint32_t ratio) {
    const uint32_t width = 2u * head_dim;
    const size_t bytes = (size_t)ratio * width * sizeof(float);
    if (!cuda_ok(cudaMemcpyAsync(state_kv, state_kv + (uint64_t)ratio * width, bytes,
                                 cudaMemcpyDeviceToDevice, 0), "csa2 overlap shift kv")) return 0;
    if (!cuda_ok(cudaMemcpyAsync(state_sc, state_sc + (uint64_t)ratio * width, bytes,
                                 cudaMemcpyDeviceToDevice, 0), "csa2 overlap shift score")) return 0;
    /* Canonicalise the half the copy just abandoned to the empty value (kv 0,
     * score -inf) rather than leaving the completed group's rows behind.  The
     * reference leaves them -- nothing reads a slot before overwriting it -- so
     * this is observationally free, and it is what makes "at a group boundary
     * the current half is empty" LITERALLY true on both the per-token and the
     * batched prefill path.  The coff-1 code makes the same choice for the same
     * reason (the comp-state gate compares prefill against decode states). */
    if (!csa2_lane_clear_current(state_kv, state_sc, head_dim, ratio)) return 0;
    return 1;
}

/* Reset the overlap lane's CURRENT half (rows ratio..2ratio-1) to the empty
 * value.  The carry half is untouched. */
static int csa2_lane_clear_current(float *state_kv, float *state_sc, uint32_t head_dim, uint32_t ratio) {
    const uint32_t width = 2u * head_dim;
    const uint64_t cur_floats = (uint64_t)ratio * width;
    const uint64_t cur_bytes = cur_floats * sizeof(float);
    float *kv_cur = state_kv + (uint64_t)ratio * width;
    float *sc_cur = state_sc + (uint64_t)ratio * width;
    if (!cuda_ok(cudaMemsetAsync(kv_cur, 0, (size_t)cur_bytes), "csa2 current half kv clear")) return 0;
    fill_f32_kernel<<<(cur_floats + 255) / 256, 256>>>(sc_cur, cur_floats, -INFINITY);
    return cuda_ok(cudaGetLastError(), "csa2 current half score clear launch");
}

int pulsar_gpu_csa2_comp_ape_add_tensor(
        pulsar_gpu_tensor *sc,
        const void        *model_map,
        uint64_t           model_size,
        uint64_t           ape_offset,
        uint32_t           ape_type,
        uint32_t           width,
        uint32_t           ratio,
        uint32_t           pos0,
        uint32_t           n_tokens) {
    /* The ape is a model-mapped table, not a device tensor: every other mapped
     * weight on this path (the norm, the matmul operands) reaches the kernels as
     * (map, size, offset, type) through cuda_model_range_ptr, and an ape that
     * arrived as a pulsar_gpu_tensor* would have had to be a second copy of the
     * model.  ds4 types: 0 = F32, 30 = BF16 -- the same pair the compressor's
     * norm accepts, and the reference stores the ape in fp32. */
    if (!sc || !model_map || width == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 30u) ||
        sc->bytes < (uint64_t)n_tokens * width * sizeof(float)) {
        fprintf(stderr, "pulsar: csa2 comp ape add: bad operands (width %u, ratio %u, n_tokens %u, type %u) -- refusing\n",
                width, ratio, n_tokens, ape_type);
        return 0;
    }
    const uint64_t ape_bytes = (uint64_t)ratio * width * pulsar_w_elt_bytes(ape_type == 30u);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset) {
        fprintf(stderr, "pulsar: csa2 comp ape add: ape range is outside the model map -- refusing\n");
        return 0;
    }
    const void *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    const uint64_t n = (uint64_t)n_tokens * width;
    if (ape_type == 30u)
        csa2_comp_ape_add_kernel<true><<<(n + 255) / 256, 256>>>((float *)sc->ptr, ape,
                                                                width, ratio, pos0, n_tokens);
    else
        csa2_comp_ape_add_kernel<false><<<(n + 255) / 256, 256>>>((float *)sc->ptr, ape,
                                                                 width, ratio, pos0, n_tokens);
    return cuda_ok(cudaGetLastError(), "csa2 comp ape add launch");
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
    const uint32_t coff = pulsar_compress_coff(ratio);
    const uint32_t width = coff * head_dim;
    const uint32_t n_groups = n_tokens / ratio;
    const uint32_t rem = n_tokens - n_groups * ratio;
    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    if (kv->bytes < (uint64_t)n_tokens * row_bytes ||
        latent->bytes < (uint64_t)n_groups * head_dim * sizeof(float) ||
        (ratio > 1u && (!sc || sc->bytes < (uint64_t)n_tokens * row_bytes))) {
        fprintf(stderr, "pulsar: csa2 compressor prefill: operand too small (kv %llu B for %u rows of %u) -- refusing\n",
                (unsigned long long)kv->bytes, n_tokens, width);
        return 0;
    }
    if (ratio > 1u) {
        const uint64_t lane_floats = (uint64_t)coff * ratio * width;
        const uint64_t lane_bytes = lane_floats * sizeof(float);
        if (!state_kv || !state_score || state_kv->bytes < lane_bytes || state_score->bytes < lane_bytes) {
            fprintf(stderr, "pulsar: csa2 compressor prefill: state lane missing or too small at ratio %u (need %llu B) -- refusing\n",
                    ratio, (unsigned long long)lane_bytes);
            return 0;
        }
        if (pos0 == 0u) {
            /* The reference registers a fresh state at the sequence start; at
             * pos0 != 0 the lane is LIVE and clearing it would destroy the carry
             * group 0 is about to read. */
            if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)lane_bytes), "csa2 state kv zero")) return 0;
            fill_f32_kernel<<<(lane_floats + 255) / 256, 256>>>((float *)state_score->ptr, lane_floats, -INFINITY);
            if (!cuda_ok(cudaGetLastError(), "csa2 state score fill launch")) return 0;
        }
        if (coff == 2u) {
            /* POOL FIRST.  At pos0 != 0 group 0 reads the incoming carry out of
             * rows 0..ratio-1, and the carry store below overwrites exactly those
             * rows -- so the order here is load-bearing, not stylistic. */
            if (!csa2_pool_norm_launch((float *)latent->ptr, (const float *)kv->ptr, (const float *)sc->ptr,
                                       (const float *)state_kv->ptr, (const float *)state_score->ptr,
                                       norm_w, norm_type == 30u, head_dim, ratio, coff, n_groups,
                                       CSA2_SRC_ROWS, pos0, rms_eps)) return 0;
            /* rows 0..ratio-1: this chunk's LAST FULL GROUP, which is the next
             * call's carry; rows ratio..: the trailing partial group.  Both keep
             * their full 2*head_dim width -- the halves are read separately. */
            if (n_groups > 0u) {
                const uint64_t n = (uint64_t)ratio * width;
                csa2_compressor_store_kernel<<<(n + 255) / 256, 256>>>(
                        (float *)state_kv->ptr, (float *)state_score->ptr,
                        (const float *)kv->ptr, (const float *)sc->ptr,
                        head_dim, ratio, coff, 0u, 0u, (n_groups - 1u) * ratio, ratio);
                if (!cuda_ok(cudaGetLastError(), "csa2 overlap carry store launch")) return 0;
            }
            /* The current half is emptied first so the slots past the partial
             * group hold the empty value, exactly as the per-token path leaves
             * them after its last shift. */
            if (!csa2_lane_clear_current((float *)state_kv->ptr, (float *)state_score->ptr, head_dim, ratio)) return 0;
            if (rem) {
                const uint64_t n = (uint64_t)rem * width;
                csa2_compressor_store_kernel<<<(n + 255) / 256, 256>>>(
                        (float *)state_kv->ptr, (float *)state_score->ptr,
                        (const float *)kv->ptr, (const float *)sc->ptr,
                        head_dim, ratio, coff, ratio, 0u, n_groups * ratio, rem);
                if (!cuda_ok(cudaGetLastError(), "csa2 overlap partial store launch")) return 0;
            }
            return 1;
        }
        /* coff 1: the lane IS the group, so reset and leave the trailing partial. */
        const uint64_t state_n = (uint64_t)ratio * head_dim;
        if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))), "csa2 state kv zero")) return 0;
        fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
        if (!cuda_ok(cudaGetLastError(), "csa2 state score fill launch")) return 0;
        if (rem) {
            const uint64_t n = (uint64_t)rem * head_dim;
            csa2_compressor_store_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    head_dim, ratio, 1u, 0u, pos0 + n_groups * ratio, n_groups * ratio, rem);
            if (!cuda_ok(cudaGetLastError(), "csa2 state store launch")) return 0;
        }
    }
    return csa2_pool_norm_launch((float *)latent->ptr, (const float *)kv->ptr,
                                 ratio > 1u ? (const float *)sc->ptr : NULL, NULL, NULL, norm_w,
                                 norm_type == 30u, head_dim, ratio, coff, n_groups, CSA2_SRC_ROWS, pos0, rms_eps);
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
    const uint32_t coff = pulsar_compress_coff(ratio);
    const uint32_t width = coff * head_dim;
    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    if (kv_cur->bytes < row_bytes || latent->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    if (ratio == 1u) {
        *emitted = 1;
        return csa2_pool_norm_launch((float *)latent->ptr, (const float *)kv_cur->ptr, NULL, NULL, NULL, norm_w,
                                     norm_type == 30u, head_dim, 1u, 1u, 1u, CSA2_SRC_ROWS, pos, rms_eps);
    }
    if (!pulsar_gpu_csa2_compressor_store_tensor(kv_cur, sc_cur, state_kv, state_score, head_dim, ratio, pos)) return 0;
    if ((pos + 1u) % ratio != 0u) return 1;   /* the group is still filling */
    *emitted = 1;
    if (!csa2_pool_norm_launch((float *)latent->ptr, NULL, NULL,
                               (const float *)state_kv->ptr, (const float *)state_score->ptr, norm_w,
                               norm_type == 30u, head_dim, ratio, coff, 1u, CSA2_SRC_STATE,
                               pos + 1u - ratio, rms_eps)) return 0;
    if (coff == 2u) {
        /* the group just pooled becomes the next group's carry -- and that is
         * ALL: the current half is overwritten slot by slot as the next group
         * fills, so an "empty" lane is not a state the overlap case has. */
        return csa2_overlap_shift((float *)state_kv->ptr, (float *)state_score->ptr, head_dim, ratio);
    }
    /* coff 1: the group is consumed, so leave the canonical empty state (kv 0,
     * score -inf) -- the state a boundary-aligned prefill leaves too, so "empty
     * at a group boundary" is literally true on every path (rewind, resume, the
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
    const uint32_t coff = pulsar_compress_coff(ratio);
    const uint32_t width = coff * head_dim;
    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t lane_bytes = (uint64_t)coff * ratio * width * sizeof(float);
    if (!kv_row || !sc_row || !state_kv || !state_score || head_dim == 0 || ratio < 2u ||
        kv_row->bytes < row_bytes || sc_row->bytes < row_bytes ||
        state_kv->bytes < lane_bytes || state_score->bytes < lane_bytes) {
        fprintf(stderr, "pulsar: csa2 compressor store: bad operands (ratio %u, head_dim %u, coff %u) -- refusing\n",
                ratio, head_dim, coff);
        return 0;
    }
    /* A running token goes into the CURRENT half (rows ratio..); coff 1 has no
     * halves, so its slot is simply pos %% ratio. */
    const uint32_t slot_base = coff == 2u ? ratio : 0u;
    const uint64_t n = width;
    csa2_compressor_store_kernel<<<(n + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv_row->ptr, (const float *)sc_row->ptr,
            head_dim, ratio, coff, slot_base, pos, 0u, 1u);
    return cuda_ok(cudaGetLastError(), "csa2 compressor store launch");
}

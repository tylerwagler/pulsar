#include "pulsar_cuda_internal.h"
#include "pulsar_cuda_mx.cuh"






/* Parallel hc4 split for the fused decode kernels: lanes 0..31 of warp 0
 * cooperate (guards inside; all lanes must enter for the __syncwarp()s).
 * BIT-IDENTICAL to hc4_split_one: rows (and columns) are normalized
 * INDEPENDENTLY in the serial version, so distributing one row/column per lane
 * with serial per-lane reductions preserves every element's exact float
 * operation sequence -- only genuinely independent work runs concurrently.
 * This removes the ~1k-cycle single-thread critical section (24 expf + 20
 * Sinkhorn iterations) that ran while 255 threads idled, x2 per layer per
 * decode token. c is a per-block shared[16] scratch. */
__device__ static void hc4_split_par(float *out, const float *mix, const float *scale,
                                     const float *base, uint32_t sinkhorn_iters, float epsv,
                                     uint32_t lane, float *c) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    if (lane < 4u) {
        float z = mix[lane] * pre_scale + base[lane];
        out[lane] = 1.0f / (1.0f + expf(-z)) + epsv;
    } else if (lane < 8u) {
        float z = mix[lane] * post_scale + base[lane];
        out[lane] = 2.0f / (1.0f + expf(-z));
    }
    if (lane < 4u) {
        const uint32_t r = lane;
        float v[4];
        float m = -INFINITY;
        for (int col = 0; col < 4; col++) {
            v[col] = mix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
            m = fmaxf(m, v[col]);
        }
        float ss = 0.0f;
        for (int col = 0; col < 4; col++) {
            v[col] = expf(v[col] - m);
            ss += v[col];
        }
        for (int col = 0; col < 4; col++) c[r * 4 + col] = v[col] / ss + epsv;
    }
    __syncwarp();
    if (lane < 4u) {
        const uint32_t col = lane;
        float ss = epsv;
        for (int r = 0; r < 4; r++) ss += c[r * 4 + col];
        for (int r = 0; r < 4; r++) c[r * 4 + col] /= ss;
    }
    __syncwarp();
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        if (lane < 4u) {
            const uint32_t r = lane;
            float ss = epsv;
            for (int col = 0; col < 4; col++) ss += c[r * 4 + col];
            for (int col = 0; col < 4; col++) c[r * 4 + col] /= ss;
        }
        __syncwarp();
        if (lane < 4u) {
            const uint32_t col = lane;
            float ss = epsv;
            for (int r = 0; r < 4; r++) ss += c[r * 4 + col];
            for (int r = 0; r < 4; r++) c[r * 4 + col] /= ss;
        }
        __syncwarp();
    }
    if (lane < 16u) out[8 + lane] = c[lane];
}





__global__ static void hc_weighted_sum_kernel(float *out, const pulsar_hc_t *x, const float *w, uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens, uint32_t weight_stride_f32) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_tokens;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint32_t t = gid / n_embd;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) {
        acc += pulsar_hc_load(x, (uint64_t)t * n_hc * n_embd + (uint64_t)h * n_embd + d) *
               w[(uint64_t)t * weight_stride_f32 + h];
    }
    out[(uint64_t)t * n_embd + d] = acc;
}



__global__ static void hc_expand_kernel(
        pulsar_hc_t *out_hc,
        const float *block_out,
        const float *block_add,
        const pulsar_hc_t *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride,
        int has_add) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n_elem) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t dst_hc = tmp % n_hc;
    uint32_t t = tmp / n_hc;

    float block_v = block_out[(uint64_t)t * n_embd + d];
    if (has_add) block_v += block_add[(uint64_t)t * n_embd + d];
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        float res_v = pulsar_hc_load(residual_hc, (uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d);
        acc += comb_v * res_v;
    }
    pulsar_hc_store(out_hc, (uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d, acc);
}



__global__ static void hc_split_weighted_sum_fused_kernel(
        float *out,
        float *split,
        const float *mix,
        const pulsar_hc_t *residual_hc,
        const float *scale,
        const float *base,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv) {
    uint32_t t = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    __shared__ float hc4_c[16];
    if (d < 32u) hc4_split_par(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv, d, hc4_c);
    __syncthreads();
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += pulsar_hc_load(residual_hc, (uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col) * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
    }
}



/* BIT-EXACT register-staging rewrite (2026-07-21).  At decode n_rows==1, so this
 * runs as ONE block on a 48-SM GPU and every stall is exposed.  The old shape
 * wrote `acc` to global `out`, then -- after a full block barrier -- read the
 * SAME value straight back to scale it, i.e. a read-after-write round trip
 * through memory for a value that was already in a register.  With BLK a
 * compile-time constant the per-thread column set {d, d+BLK, ...} is a
 * compile-time-bounded VEC, so the accs live in registers across the barrier
 * and the second pass loads nothing.
 *
 * BIT-EXACT: `v` reloaded from `out` was the f32 that `acc` already held (one
 * store + one load of the same 32-bit pattern -- no rounding in between), the
 * column order and the 256-partial pairwise tree are untouched, and `out` is
 * still written with the same values.  Only the reload disappears.
 *
 * `residual_hc` is an HC CARRIER (BF16 storage under task #62) and is read
 * through pulsar_hc_load exactly as the generic kernel below does. */
/* The MX emit helpers moved to pulsar_cuda_mx.cuh so the swizzle has ONE
 * definition to keep in step with pulsar_cuda_matmul.cu's quantiser -- a second
 * producer (dsv4_qkv_rms_norm_rows_kernel) needs the same code, and a divergent
 * copy is a silent wrong-operand bug rather than a rounding difference. */


/* NWBF16: storage of norm_w (attn_norm / ffn_norm), bf16 in source. Promoted
 * to f32 before it multiplies, so an f32 tensor stays bit-exact. */
template <uint32_t BLK, uint32_t VEC, bool NWBF16>
__global__ static void hc_split_weighted_sum_norm_fused_kernel(
        float *out,
        float *norm_out,
        __nv_fp8_e4m3 *norm_out_q,
        unsigned char *norm_out_sf,
        int norm_out_kbp,
        float *split,
        const float *mix,
        const pulsar_hc_t *residual_hc,
        const float *scale,
        const float *base,
        const void *norm_w,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv,
        float norm_eps) {
    const uint32_t t = blockIdx.x;
    const uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    __shared__ float hc4_c[16];
    if (d < 32u) hc4_split_par(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv, d, hc4_c);
    __syncthreads();

    const uint64_t rbase = (uint64_t)t * 4u * n_embd;
    const uint64_t obase = (uint64_t)t * n_embd;
    float accs[VEC];
    float sum = 0.0f;
    #pragma unroll
    for (uint32_t u = 0; u < VEC; u++) {
        const uint32_t col = d + u * BLK;
        if (col < n_embd) {
            float acc = 0.0f;
            #pragma unroll
            for (uint32_t h = 0; h < 4; h++) {
                acc += pulsar_hc_load(residual_hc, rbase + (uint64_t)h * n_embd + col) * sp[h];
            }
            if (out) out[obase + col] = acc;
            accs[u] = acc;
            sum += acc * acc;
        } else {
            accs[u] = 0.0f;
        }
    }

    __shared__ float partial[BLK];
    partial[d] = sum;
    __syncthreads();
    for (uint32_t stride = BLK >> 1; stride > 0; stride >>= 1) {
        if (d < stride) partial[d] += partial[d + stride];
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / (float)n_embd + norm_eps);
    #pragma unroll
    for (uint32_t u = 0; u < VEC; u++) {
        const uint32_t col = d + u * BLK;
        const float v = (col < n_embd)
                ? (accs[u] * norm_scale * pulsar_w_load_f32_or_bf16<NWBF16>(norm_w, col)) : 0.0f;
        if (col < n_embd) {
            norm_out[obase + col] = v;
        }
        /* Warp-uniform: every lane must reach the shuffle.  BLK is a multiple of
         * 32 and columns are contiguous within a warp, so a warp spans exactly
         * one MX block and lanes past n_embd contribute 0 to the max. */
        if (norm_out_q) {
            pulsar_mx_emit_block(v, col, t, n_embd, norm_out_kbp, norm_out_q, norm_out_sf);
        }
    }
}


/* Generic fallback for n_embd > BLK*VEC (byte-identical to the pre-2026-07-21
 * kernel: same column order, same tree, same global round trip). */



__global__ static void output_hc_weights_kernel(
        float *out,
        const float *pre,
        const float *scale,
        const float *base,
        uint32_t n_hc,
        uint32_t n_tokens,
        float epsv) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_tokens * n_hc;
    if (gid >= n) return;
    uint32_t h = gid % n_hc;
    float z = pre[gid] * scale[0] + base[h];
    out[gid] = 1.0f / (1.0f + expf(-z)) + epsv;
}



__device__ static float softplus_dev(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}



__device__ __forceinline__ static bool router_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}



__global__ static void router_select_warp_topk_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row_in_block = threadIdx.y;
    const uint32_t t = blockIdx.x * blockDim.y + row_in_block;
    if (t >= n_tokens || lane >= 32u) return;

    const float *log = logits + (uint64_t)t * 256u;
    float *prob = probs ? probs + (uint64_t)t * 256u : NULL;
    int32_t *sel = selected + (uint64_t)t * 6u;
    float *w = weights + (uint64_t)t * 6u;
    __shared__ float sprob[4][256];
    float local_prob[8];
    float local_score[8];

    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        const uint32_t e = lane + j * 32u;
        const float p = sqrtf(softplus_dev(log[e]));
        local_prob[j] = p;
        local_score[j] = p + (has_bias ? bias[e] : 0.0f);
        sprob[row_in_block][e] = p;
        if (prob) prob[e] = p;
    }
    __syncwarp();

    if (hash_mode) {
        if (lane == 0) {
            int32_t tok = tokens ? tokens[t] : token_scalar;
            if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
            const int32_t *row = hash + (uint64_t)tok * 6u;
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) {
                const int32_t e = row[j];
                sel[j] = e;
                const float v = (e >= 0 && e < 256) ? sprob[row_in_block][(uint32_t)e] : 0.0f;
                w[j] = v;
                sum += v;
            }
            sum = fmaxf(sum, 6.103515625e-5f);
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) w[j] = w[j] / sum * 1.5f;
        }
        return;
    }

    float out_prob[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t out_idx[6] = {0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t k = 0; k < 6u; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            const float s = local_score[j];
            if (router_score_better(s, e, best_score, best_idx)) {
                best_score = s;
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
            const float other_prob = __shfl_xor_sync(0xffffffffu, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(0xffffffffu, best_idx, mask);
            if (router_score_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            if (e == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            out_idx[k] = best_idx;
            out_prob[k] = best_prob;
        }
    }

    if (lane == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) {
            sel[j] = (int32_t)out_idx[j];
            w[j] = out_prob[j];
            sum += out_prob[j];
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) w[j] = w[j] / sum * 1.5f;
    }
}



/* out_q/out_sf, when non-NULL, receive the E4M3 + E8M0 encoding of the SwiGLU
 * result straight from this epilogue, so the MXFP8 shared_down GEMM does not
 * wait on a separate quantize pass over batch_shared_mid.  The launcher only
 * supplies them when n is a multiple of the 256-thread block, so no lane takes
 * the `i >= n` exit before the warp-wide shuffle in pulsar_mx_emit_block.
 *
 * This kernel is launched FLAT over n = rows * mid_dim, so the MX (row, col)
 * has to be recovered by division.  mid_dim is a multiple of 32, so a warp's
 * 32 consecutive i never straddle a row and are 32-aligned within it. */
__global__ static void swiglu_kernel(float *out, const float *gate, const float *up, uint32_t n, float clamp, float weight,
                                     __nv_fp8_e4m3 *out_q, unsigned char *out_sf, int out_kbp, uint32_t mid_dim) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    float s = g / (1.0f + expf(-g));
    const float v = s * u * weight;
    /* `out` is NULL when the launcher was told the f32 store is dead -- the
     * MXFP8 consumer reads the encoding below instead.  The branch is uniform
     * across the whole launch, so it costs nothing in a bandwidth-bound
     * kernel, and it removes the widest store this kernel makes. */
    if (out) out[i] = v;
    if (out_q) {
        pulsar_mx_emit_block(v, i % mid_dim, i / mid_dim, mid_dim, out_kbp, out_q, out_sf);
    }
}



__global__ static void add_kernel(float *out, const float *a, const float *b, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}



__global__ static void directional_steering_project_kernel(
        float       *x,
        const float *directions,
        uint32_t     layer,
        uint32_t     width,
        uint32_t     rows,
        float        scale) {
    const uint32_t row = blockIdx.x;
    if (row >= rows || width == 0) return;

    float *xr = x + (uint64_t)row * width;
    const float *dir = directions + (uint64_t)layer * width;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        sum += xr[i] * dir[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }

    const float coeff = scale * partial[0];
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        xr[i] -= coeff * dir[i];
    }
}


int pulsar_gpu_swiglu_mx_tensor(pulsar_gpu_tensor *out, const pulsar_gpu_tensor *gate, const pulsar_gpu_tensor *up,
                                uint32_t n, float clamp, float weight,
                                void *out_q, void *out_sf, int out_kbp, uint32_t mid_dim,
                                int skip_f32) {
    if (!out || !gate || !up ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        gate->bytes < (uint64_t)n * sizeof(float) ||
        up->bytes < (uint64_t)n * sizeof(float)) return 0;
    /* skip_f32 without an encoding to replace it would write NOTHING and leave
     * the consumer reading stale bytes.  Same failure shape as a skipped
     * emission, so it gets the same treatment: refuse, do not silently
     * downgrade to storing f32 (that would hide a caller bug behind a
     * correct-looking run). */
    if (skip_f32 && !out_q) {
        fprintf(stderr, "pulsar: swiglu asked to skip the f32 store with no E4M3 slot "
                        "(n=%u mid_dim=%u) -- refusing\n", n, mid_dim);
        return 0;
    }
    /* FAIL LOUD rather than silently skip the emission: the caller arms the
     * activation cache off the same slot pointer, so a skipped emission leaves
     * the GEMM reading a memset-zero E4M3 buffer -- a well-formed WRONG answer.
     * n must fill whole blocks (no lane exits before the warp shuffle) and
     * mid_dim must be a whole number of MX blocks. */
    if (out_q && ((n % 256u) != 0u || mid_dim == 0u || (mid_dim % 32u) != 0u)) {
        fprintf(stderr, "pulsar: swiglu cannot emit MX for n=%u mid_dim=%u "
                        "(need n %% 256 == 0 and mid_dim %% 32 == 0)\n", n, mid_dim);
        return 0;
    }
    /* Announce once per shape.  A byte-identical gate cannot tell "the store
     * was skipped" from "the skip never fired", and a silently-inert
     * optimisation looks exactly like a working one -- same reason the A8 GEMV
     * arms announce themselves in pulsar_cuda_matmul.cu. */
    if (skip_f32) {
        static uint32_t seen_n[8] = {0};
        static int n_seen = 0;
        int known = 0;
        for (int i = 0; i < n_seen; i++) if (seen_n[i] == n) { known = 1; break; }
        if (!known && n_seen < 8) {
            seen_n[n_seen++] = n;
            fprintf(stderr, "pulsar: swiglu f32 store SKIPPED (n=%u mid_dim=%u, %.1f MiB/layer)\n",
                    n, mid_dim, (double)n * sizeof(float) / (1024.0 * 1024.0));
        }
    }
    swiglu_kernel<<<(n + 255) / 256, 256>>>(skip_f32 ? NULL : (float *)out->ptr,
                                            (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight,
                                            (__nv_fp8_e4m3 *)out_q, (unsigned char *)out_sf, out_kbp, mid_dim);
    return cuda_ok(cudaGetLastError(), "swiglu launch");
}


int pulsar_gpu_shared_gate_up_swiglu_mxfp8_tensor(
        pulsar_gpu_tensor       *gate,
        pulsar_gpu_tensor       *up,
        pulsar_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        float                   clamp,
        void                   *mid_q,
        void                   *mid_sf,
        int                     mid_kbp) {
    return pulsar_gpu_matmul_mxfp8_tensor(gate, model_map, model_size,
                                        gate_offset, in_dim, out_dim, x, 1, /* out_f16: */ 0) &&
           pulsar_gpu_matmul_mxfp8_tensor(up, model_map, model_size,
                                        up_offset, in_dim, out_dim, x, 1, /* out_f16: */ 0) &&
           /* One row here, so n == out_dim and the MX row width IS out_dim.
            * skip_f32 = 0 deliberately: this is the DECODE path, where the
            * store is one row (8 KiB, not 32 MiB) and so is worth nothing,
            * while the arms that could read it are the n==1 GEMVs rather than
            * the single cuBLASLt path prefill takes.  The prefill call site in
            * gpu_prefill.cpp is where the store is actually dead. */
           pulsar_gpu_swiglu_mx_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f,
                                       mid_q, mid_sf, mid_kbp, (uint32_t)out_dim, 0);
}


int pulsar_gpu_add_tensor(pulsar_gpu_tensor *out, const pulsar_gpu_tensor *a, const pulsar_gpu_tensor *b, uint32_t n) {
    if (!out || !a || !b ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float)) return 0;
    add_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)a->ptr, (const float *)b->ptr, n);
    return cuda_ok(cudaGetLastError(), "add launch");
}


int pulsar_gpu_directional_steering_project_tensor(
        pulsar_gpu_tensor       *x,
        const pulsar_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    if (!x || !directions || width == 0 || rows == 0 || scale == 0.0f) return 0;
    const uint64_t x_bytes = (uint64_t)width * rows * sizeof(float);
    const uint64_t dir_bytes = (uint64_t)(layer + 1u) * width * sizeof(float);
    if (x->bytes < x_bytes || directions->bytes < dir_bytes) return 0;

    uint32_t nth = 256u;
    while (nth > width && nth > 1u) nth >>= 1;
    directional_steering_project_kernel<<<rows, nth>>>(
            (float *)x->ptr,
            (const float *)directions->ptr,
            layer,
            width,
            rows,
            scale);
    return cuda_ok(cudaGetLastError(), "directional steering launch");
}


int pulsar_gpu_router_select_tensor(pulsar_gpu_tensor *selected, pulsar_gpu_tensor *weights, pulsar_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t token, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const pulsar_gpu_tensor *logits) {
    if (!selected || !weights || !logits || !model_map || n_expert_groups > 1u || n_group_used > 0u) return 0;
    if (n_expert != 256u || n_expert_used != 6u || fabsf(expert_weight_scale - 1.5f) > 1.0e-6f) return 0;
    int32_t tok = (int32_t)token;
    int ok = 1;
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (ok && has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < 256u * sizeof(float)) ok = 0;
        else bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, 256u * sizeof(float), "router_bias");
        if (!bias) ok = 0;
    }
    if (ok && hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) ok = 0;
        else hash = (const int32_t *)cuda_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) ok = 0;
    }
    if (ok) {
        dim3 block(32, 4, 1);
        router_select_warp_topk_kernel<<<1, block>>>((int32_t *)selected->ptr, (float *)weights->ptr, probs ? (float *)probs->ptr : NULL,
                                                     bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                                     has_bias && !hash_mode, hash_mode);
        ok = cuda_ok(cudaGetLastError(), "router_select launch");
    }
    return ok;
}


int pulsar_gpu_router_select_batch_tensor(pulsar_gpu_tensor *selected, pulsar_gpu_tensor *weights, pulsar_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const pulsar_gpu_tensor *logits, const pulsar_gpu_tensor *tokens, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_tokens) {
    if (n_expert != 256u || n_expert_used != 6u || fabsf(expert_weight_scale - 1.5f) > 1.0e-6f) return 0;
    if (!selected || !weights || !logits || !tokens || !model_map || n_tokens == 0 ||
        n_expert_groups > 1u || n_group_used > 0u ||
        logits->bytes < (uint64_t)n_tokens * 256u * sizeof(float) ||
        (probs && probs->bytes < (uint64_t)n_tokens * 256u * sizeof(float)) ||
        selected->bytes < (uint64_t)n_tokens * 6u * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * 6u * sizeof(float)) {
        return 0;
    }
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < 256u * sizeof(float)) return 0;
        bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, 256u * sizeof(float), "router_bias");
        if (!bias) return 0;
    }
    if (hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) return 0;
        hash = (const int32_t *)cuda_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) return 0;
    }
    dim3 block(32, 4, 1);
    router_select_warp_topk_kernel<<<(n_tokens + 3u) / 4u, block>>>((int32_t *)selected->ptr,
                                                                    (float *)weights->ptr,
                                                                    probs ? (float *)probs->ptr : NULL,
                                                                    bias,
                                                                    hash,
                                                                    (const float *)logits->ptr,
                                                                    (const int32_t *)tokens->ptr,
                                                                    0,
                                                                    hash_rows,
                                                                    n_tokens,
                                                                    has_bias && !hash_mode,
                                                                    hash_mode);
    return cuda_ok(cudaGetLastError(), "router_select launch");
}




int pulsar_gpu_hc_weighted_sum_tensor(pulsar_gpu_tensor *out, const pulsar_gpu_tensor *residual_hc, const pulsar_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !residual_hc || !weights || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float)));
    /* n_tokens comes from the OUTPUT; bound the carrier input too, mirroring
     * pulsar_gpu_hc_split_weighted_sum_tensor. The BF16 narrowing removed the
     * accidental 2x margin that made an over-read harmless (task #62). */
    if (residual_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd * PULSAR_HC_ELT_SIZE ||
        weights->bytes < (uint64_t)n_tokens * n_hc * sizeof(float)) return 0;
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256>>>(
        (float *)out->ptr, (const pulsar_hc_t *)residual_hc->ptr, (const float *)weights->ptr,
        n_embd, n_hc, n_tokens, n_hc);
    return cuda_ok(cudaGetLastError(), "hc_weighted_sum launch");
}




int pulsar_gpu_hc_split_weighted_sum_tensor(
        pulsar_gpu_tensor       *out,
        pulsar_gpu_tensor       *split,
        const pulsar_gpu_tensor *mix,
        const pulsar_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc != 4) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * PULSAR_HC_ELT_SIZE;
    if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
        scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset) {
        return 0;
    }
    uint64_t n_rows = out->bytes / out_row_bytes;
    if (mix->bytes < n_rows * mix_bytes ||
        split->bytes < n_rows * mix_bytes ||
        residual_hc->bytes < n_rows * residual_row_bytes) {
        return 0;
    }
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, mix_bytes, "hc_base");
    if (!scale || !base) return 0;
    hc_split_weighted_sum_fused_kernel<<<(uint32_t)n_rows, 256>>>(
            (float *)out->ptr,
            (float *)split->ptr,
            (const float *)mix->ptr,
            (const pulsar_hc_t *)residual_hc->ptr,
            scale,
            base,
            n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps);
    return cuda_ok(cudaGetLastError(), "hc split weighted sum launch");
}


int pulsar_gpu_hc_split_weighted_sum_norm_f16_tensor(
        pulsar_gpu_tensor       *out,
        pulsar_gpu_tensor       *norm_out,
        void                    *norm_out_q,
        void                    *norm_out_sf,
        int                      norm_out_kbp,
        pulsar_gpu_tensor       *split,
        const pulsar_gpu_tensor *mix,
        const pulsar_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_rows_in,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps,
        int                     norm_w_bf16) {
    /* `out` is OPTIONAL: it is the pre-norm carrier, which nothing reads except
     * a debug dump, so callers pass NULL unless a dump was requested. */
    if (!norm_out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc != 4) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * PULSAR_HC_ELT_SIZE;
    if (n_rows_in == 0 ||
        norm_out->bytes < (uint64_t)n_rows_in * out_row_bytes ||
        (out && out->bytes < (uint64_t)n_rows_in * out_row_bytes) ||
        scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset ||
        norm_weight_offset > model_size ||
        (uint64_t)n_embd * pulsar_w_elt_bytes(norm_w_bf16) > model_size - norm_weight_offset) {
        return 0;
    }
    const uint64_t n_rows = n_rows_in;
    if (mix->bytes < n_rows * mix_bytes ||
        split->bytes < n_rows * mix_bytes ||
        residual_hc->bytes < n_rows * residual_row_bytes) {
        return 0;
    }
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset,
            3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset,
            mix_bytes, "hc_base");
    const void *norm_w = cuda_model_range_ptr(model_map, norm_weight_offset,
            (uint64_t)n_embd * pulsar_w_elt_bytes(norm_w_bf16), "hc_norm_weight");
    if (!scale || !base || !norm_w) return 0;
#define PULSAR_HCFUSED_BLK 256u
#define PULSAR_HCFUSED_VEC 16u
    /* Flash is n_embd == 4096 == BLK*VEC exactly, so the templated kernel always
     * applies.  There WAS a generic fallback for wider models; it is deleted.
     * It could not emit E4M3 at all -- its parameter list had no norm_out_q --
     * while the caller still marked the activation slot valid, so every MXFP8
     * consumer downstream read the previous layer's data against zeroed scales.
     * Fail loud rather than ever silently skip the emit again. */
    if (n_embd > PULSAR_HCFUSED_BLK * PULSAR_HCFUSED_VEC) {
        fprintf(stderr, "pulsar: hc fused norm cannot handle n_embd=%u (max %u)\n",
                n_embd, PULSAR_HCFUSED_BLK * PULSAR_HCFUSED_VEC);
        return 0;
    }
    {
#define PULSAR_HCFUSED_LAUNCH(NW)                                                    \
        hc_split_weighted_sum_norm_fused_kernel<PULSAR_HCFUSED_BLK, PULSAR_HCFUSED_VEC, NW> \
                <<<(uint32_t)n_rows, PULSAR_HCFUSED_BLK>>>(                           \
                out ? (float *)out->ptr : NULL,                                       \
                (float *)norm_out->ptr,                                              \
                (__nv_fp8_e4m3 *)norm_out_q,                                         \
                (unsigned char *)norm_out_sf,                                        \
                norm_out_kbp,                                                        \
                (float *)split->ptr,                                                 \
                (const float *)mix->ptr,                                             \
                (const pulsar_hc_t *)residual_hc->ptr,                               \
                scale,                                                               \
                base,                                                                \
                norm_w,                                                              \
                n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps, norm_eps)
        if (norm_w_bf16) PULSAR_HCFUSED_LAUNCH(true);
        else             PULSAR_HCFUSED_LAUNCH(false);
#undef PULSAR_HCFUSED_LAUNCH
        return cuda_ok(cudaGetLastError(), "hc split weighted sum norm launch");
    }
}




int pulsar_gpu_output_hc_weights_tensor(
        pulsar_gpu_tensor       *out,
        const pulsar_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps) {
    if (!out || !pre || !model_map || n_hc == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
    if (row_bytes == 0 || out->bytes < row_bytes || out->bytes % row_bytes != 0 ||
        pre->bytes < out->bytes ||
        scale_offset > model_size || sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || row_bytes > model_size - base_offset) {
        return 0;
    }
    const uint64_t n_tokens = out->bytes / row_bytes;
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, sizeof(float), "output_hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, row_bytes, "output_hc_base");
    if (!scale || !base) return 0;
    uint64_t n = n_tokens * n_hc;
    output_hc_weights_kernel<<<(n + 255) / 256, 256>>>(
            (float *)out->ptr,
            (const float *)pre->ptr,
            scale,
            base,
            n_hc,
            (uint32_t)n_tokens,
            eps);
    return cuda_ok(cudaGetLastError(), "output hc weights launch");
}


int pulsar_gpu_hc_expand_tensor(pulsar_gpu_tensor *out_hc, const pulsar_gpu_tensor *block_out, const pulsar_gpu_tensor *residual_hc, const pulsar_gpu_tensor *post, const pulsar_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !post || !comb || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * PULSAR_HC_ELT_SIZE));
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((pulsar_hc_t *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const pulsar_hc_t *)residual_hc->ptr,
                                                    (const float *)post->ptr,
                                                    (const float *)comb->ptr,
                                                    n_embd, n_hc, n_tokens,
                                                    n_hc, n_hc * n_hc, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand launch");
}


int pulsar_gpu_hc_expand_split_tensor(pulsar_gpu_tensor *out_hc, const pulsar_gpu_tensor *block_out, const pulsar_gpu_tensor *residual_hc, const pulsar_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * PULSAR_HC_ELT_SIZE));
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((pulsar_hc_t *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const pulsar_hc_t *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand_split launch");
}



int pulsar_gpu_hc_expand_add_split_tensor(pulsar_gpu_tensor *out_hc, const pulsar_gpu_tensor *block_out, const pulsar_gpu_tensor *block_add, const pulsar_gpu_tensor *residual_hc, const pulsar_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !block_add || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * PULSAR_HC_ELT_SIZE));
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((pulsar_hc_t *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_add->ptr,
                                                    (const pulsar_hc_t *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 1);
    return cuda_ok(cudaGetLastError(), "hc_expand_add_split launch");
}



int pulsar_gpu_shared_down_hc_expand_mxfp8_tensor(
        pulsar_gpu_tensor       *out_hc,
        pulsar_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *shared_mid,
        const pulsar_gpu_tensor *routed_out,
        const pulsar_gpu_tensor *residual_hc,
        const pulsar_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    return pulsar_gpu_matmul_mxfp8_tensor(shared_out, model_map, model_size,
                                        weight_offset, in_dim, out_dim,
                                        shared_mid, 1, /* out_f16: */ 0) &&
           pulsar_gpu_hc_expand_add_split_tensor(out_hc, shared_out, routed_out,
                                                residual_hc, split, n_embd, n_hc);
}



int pulsar_gpu_matmul_fp8_hc_expand_tensor(
        pulsar_gpu_tensor       *out_hc,
        pulsar_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        const pulsar_gpu_tensor *residual_hc,
        const pulsar_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    return cuda_matmul_fp8_hc_expand_tensor_labeled(out_hc, block_out,
                                                    model_map, model_size,
                                                    weight_offset,
                                                    in_dim, out_dim,
                                                    x,
                                                    NULL,
                                                    residual_hc,
                                                    split,
                                                    n_embd, n_hc,
                                                    "fp8_hc_expand");
}



/* ---------------------------------------------------------------------------
 * Fused plain-RMSNorm + f16 HC-mix GEMV (2026-07-21, BIT-EXACT).
 *
 * The decode graph ran, three times per token per layer-pair:
 *     rms_norm_plain_tensor(flat_hc, cur_hc, hc_dim)      // grid 1  x 256
 *     matmul_f16_tensor(hc_mix, hc_attn_fn, flat_hc, 1)   // grid 24 x 256
 * (that second call is the GEMV of the day; the f16 weight path was deleted in
 * the F16 sweep and the entry point is now pulsar_gpu_matmul_bf16_tensor.  The
 * argument shape and the reduction it performs are unchanged.)
 * On a 48-SM GB10 that is one SM followed by 24 SMs, with a 64 KB f32 store and
 * a 24-way re-read of it in between, for 24 (or 4, at the output head) dot
 * products.  The nsys profile put the pair at ~5.4% of decode; the GEMV alone
 * moved 786 KB in 28.3 us = 27.8 GB/s, ~12% of achievable -- i.e. it was never
 * bandwidth, it was 8 warps on 24 SMs with no memory-level parallelism.
 *
 * This kernel keeps the SAME two reductions but runs the norm redundantly in
 * each of the out_dim blocks, so the serial 1-block stage disappears, the
 * 64 KB scratch round trip disappears (flat_hc has no other consumer -- every
 * call site feeds it straight into this GEMV), one launch disappears, and both
 * loops get compile-time strides so UNROLL loads are in flight per warp.
 *
 * `x` is an HC residual CARRIER (cur_hc / after_attn_hc), so it is read through
 * pulsar_hc_load -- BF16 storage promoted to f32 -- matching rms_norm_plain_kernel
 * exactly (task #62).  Reading it as float* would be silent corruption.
 *
 * WHY IT IS BIT-EXACT, not merely close:
 *   - The sum-of-squares keeps thread t on exactly {t, t+BLK, ...} in
 *     increasing order and reuses the identical 256-entry pairwise tree, so
 *     `scale` is the same float rsqrtf produced before.
 *   - flat[i] was stored as f32 and reloaded; here x[i]*scale is consumed from
 *     the register that held that same f32.  Store/load of an f32 is the
 *     identity, so the multiplicand is bit-identical.
 *   - The dot product keeps the same 256 partials over the same index subsets
 *     in the same order and the same pairwise tree as the generic GEMV --
 *     matmul_f16_kernel when this was written, matmul_bf16_kernel and
 *     matmul_f32_kernel today.  All of them stride thread t over
 *     {t, t+BLK, ...}, land in a 256-entry shared partial[], and fold it with
 *     the same halving tree, so the contract survived the f16 deletion.
 * There is no split-K, no atomic, and no cross-thread re-association anywhere.
 * ------------------------------------------------------------------------- */
template <uint32_t BLK, uint32_t UNROLL, typename WT>
__global__ static void hc_norm_mix_kernel(
        float *out,
        const WT *w,
        const pulsar_hc_t *x,
        uint32_t n,
        uint32_t out_dim,
        float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    const uint32_t tid = threadIdx.x;
    __shared__ float partial[BLK];

    /* stage 1: plain RMSNorm scale over x -- same order as rms_norm_plain_kernel */
    float sum = 0.0f;
    uint32_t i = tid;
    for (; i + (UNROLL - 1u) * BLK < n; i += BLK * UNROLL) {
        float v[UNROLL];
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) v[u] = pulsar_hc_load(x, i + u * BLK);
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) sum += v[u] * v[u];
    }
    for (; i < n; i += BLK) {
        float v = pulsar_hc_load(x, i);
        sum += v * v;
    }
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t stride = BLK >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    __syncthreads();   /* partial[] is reused below */

    /* stage 2: dot(w[row], normed x) -- same order as the generic GEMV */
    const WT *wr = w + (uint64_t)row * n;
    float dot = 0.0f;
    i = tid;
    for (; i + (UNROLL - 1u) * BLK < n; i += BLK * UNROLL) {
        float xv[UNROLL];
        float wv[UNROLL];
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) xv[u] = pulsar_hc_load(x, i + u * BLK);
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) wv[u] = pulsar_wt_load(wr, i + u * BLK);
        /* pinned roundings: __fmul_rn reproduces the f32 that rms_norm_plain
         * stored into flat_hc, __fmaf_rn reproduces the generic GEMV's
         * contracted `sum += w * x`.  Written explicitly so no future
         * contraction/reassociation decision by nvcc can silently move this
         * off byte-identical. */
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) dot = __fmaf_rn(wv[u], __fmul_rn(xv[u], scale), dot);
    }
    for (; i < n; i += BLK) {
        dot = __fmaf_rn(pulsar_wt_load(wr, i), __fmul_rn(pulsar_hc_load(x, i), scale), dot);
    }
    partial[tid] = dot;
    __syncthreads();
    for (uint32_t stride = BLK >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[row] = partial[0];
}


/* w_type: ds4 tensor type of the mix weight -- 1 F16, 30 BF16, 0 F32.
 *
 * Was F16-only, which made this fusion silently unreachable the moment hc_*_fn
 * moved to another storage: gpu_graph_norm_mix_plain fell back to a separate
 * rms_norm_plain + matmul pair with a 64 KB f32 scratch round trip between
 * them, the very cost this kernel exists to remove (~5.4% of decode by its own
 * measurement). The fusion has nothing to do with the weight's width, so it is
 * templated on storage rather than gated on one type. */
int pulsar_gpu_hc_norm_mix_tensor(
        pulsar_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        float                   eps,
        uint32_t                w_type) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    const uint64_t elt = (w_type == 0u) ? 4u : 2u;      /* F32 : F16/BF16 */
    const uint64_t weight_bytes = out_dim * in_dim * elt;
    if (weight_bytes > model_size - weight_offset) return 0;
    /* x is an HC residual carrier: PULSAR_HC_ELT_SIZE bytes per sample. */
    if (x->bytes < in_dim * PULSAR_HC_ELT_SIZE || out->bytes < out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "hc_mix");
    if (!wptr) return 0;
#define PULSAR_HCMIX(WT, CAST)                                          \
    hc_norm_mix_kernel<256, 8, WT><<<(uint32_t)out_dim, 256>>>(          \
            (float *)out->ptr, (const CAST)wptr, (const pulsar_hc_t *)x->ptr, \
            (uint32_t)in_dim, (uint32_t)out_dim, eps)
    /* Fail closed on an unexpected type.  This used to fall through to a
     * __half instantiation, so ANY type that was not BF16 or F32 got read as
     * f16 -- silently, at the wrong element width, which is precisely the
     * failure the typed-loader note in pulsar_cuda_internal.h was written after
     * (the embed kernels reading an f16 table as f32 on 2026-08-15).  The
     * artifact's hc_*_fn family is F32 and its drafter twin BF16, so the f16
     * arm was unreachable AND wrong for whatever would have reached it. */
    if (w_type == 30u)      PULSAR_HCMIX(__nv_bfloat16, __nv_bfloat16 *);
    else if (w_type == 0u)  PULSAR_HCMIX(float, float *);
    else {
        fprintf(stderr, "pulsar: hc_mix weight type %u is neither BF16 nor F32\n", w_type);
        return 0;
    }
#undef PULSAR_HCMIX
    return cuda_ok(cudaGetLastError(), "hc norm mix launch");
}

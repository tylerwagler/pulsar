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



/* NHC: the hidden-carrier count as a COMPILE-TIME value, or 0 to take the
 * runtime loop.  The point is memory-level parallelism, not instruction count.
 *
 * The 2026-08-24 SOL sweep found this kernel at 87.5% occupancy with ZERO
 * barrier stalls and 31.6M long_scoreboard stalls -- i.e. plenty of warps, all
 * of them waiting on global loads, at only 25.7% of memory throughput.  The
 * cause is visible in the source: `n_hc` arrives as a kernel ARGUMENT, so the
 * src_hc loop cannot be unrolled, and its `residual_hc` loads (each n_embd
 * apart, individually coalesced) issue ONE AT A TIME, each waiting on the last.
 * Templating on the count lets all NHC loads be in flight together.
 *
 * BIT-EXACT: the accumulation order is unchanged (acc += c0*r0; acc += c1*r1;
 * ...), so this is an issue-order change only.  It is graded by the byte-exact
 * prefill gate, not the reference gate. */
template <int NHC>
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
    if (NHC > 0) {
        /* Gather first, accumulate second: the loads have no dependence on each
         * other, so hoisting them out of the accumulate lets the compiler keep
         * NHC of them in flight.  The adds then run in the SAME order as the
         * rolled loop, which is what keeps this bit-exact. */
        float comb_v[NHC > 0 ? NHC : 1];
        float res_v[NHC > 0 ? NHC : 1];
#pragma unroll
        for (int src_hc = 0; src_hc < NHC; src_hc++) {
            comb_v[src_hc] = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
            res_v[src_hc] = pulsar_hc_load(residual_hc, (uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d);
        }
        /* PLAIN `acc += a*b`, DELIBERATELY, and this is a numerics decision --
         * read before "fixing" it back to __fmaf_rn.
         *
         * With --use_fast_math the compiler REASSOCIATES these four products
         * into a tree once they are visible.  That is not a defect: pairwise
         * summation has O(log n) error growth where sequential has O(n), so the
         * tree is the more accurate arithmetic.  Graded against the B300 source
         * logits it is CLOSER to the source at 6 of 9 depths, including all
         * three known-high outliers (story 512 0.643->0.569, story 30464
         * 0.255->0.173, code 3840 0.196->0.190).
         *
         * An earlier revision pinned this with __fmaf_rn purely to keep the
         * byte gate green, which preserved our own historical rounding instead
         * of reducing our distance from the source.  Tyler, 2026-08-24: "I
         * would love to disagree with our past if it puts us closer to the
         * original model."  The prefill baseline was re-anchored for this
         * change; see PREFILL_BASELINE_REF in the Makefile. */
#pragma unroll
        for (int src_hc = 0; src_hc < NHC; src_hc++) acc += comb_v[src_hc] * res_v[src_hc];
    } else {
        for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
            float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
            float res_v = pulsar_hc_load(residual_hc, (uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d);
            acc += comb_v * res_v;   /* matches the unrolled arm above */
        }
    }
    pulsar_hc_store(out_hc, (uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d, acc);
}



/* One dispatch point for the three hc_expand callers.  PULSAR_N_HC is 4 on the
 * shipped artifact; anything else takes the runtime-loop instantiation, so a
 * differently-shaped model still runs (just without the unrolled gather). */
static void hc_expand_launch(uint32_t blocks, uint32_t threads,
                             pulsar_hc_t *out_hc, const float *block_out,
                             const float *block_add, const pulsar_hc_t *residual_hc,
                             const float *post, const float *comb,
                             uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens,
                             uint32_t post_stride, uint32_t comb_stride, int has_add) {
    if (n_hc == 4u) {
        hc_expand_kernel<4><<<blocks, threads>>>(out_hc, block_out, block_add, residual_hc,
                                                post, comb, n_embd, n_hc, n_tokens,
                                                post_stride, comb_stride, has_add);
    } else {
        hc_expand_kernel<0><<<blocks, threads>>>(out_hc, block_out, block_add, residual_hc,
                                                post, comb, n_embd, n_hc, n_tokens,
                                                post_stride, comb_stride, has_add);
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
 * to f32 before it multiplies, so an f32 tensor stays bit-exact.
 *
 * SINGLE-PASS mHC (L218, DeepSeek-V4.1 Block.forward): the coefficients a
 * sublayer derives from its own input are used by the NEXT sublayer's collapse
 * -- attention collapses with the previous FFN's `pre` (an initial one-hot
 * before layer 0), the FFN with this attention's, the output head with the
 * last FFN's.  `pre_carry` [n_rows][4] is that hand-over: each row reads the
 * pre it was handed, collapses with it, then overwrites the slot with the pre
 * it derived here for whoever collapses next.  The read happens in the same
 * threads before the write, and no other block touches the row, so one buffer
 * carries the value through the whole sweep.  `post` and `comb` (split[4..])
 * stay with THIS sublayer's expand, as in the reference's hc_post. */
template <uint32_t BLK, uint32_t VEC, bool NWBF16>
__global__ static void hc_split_weighted_sum_norm_fused_kernel(
        float *out,
        float *norm_out,
        __nv_fp8_e4m3 *norm_out_q,
        unsigned char *norm_out_sf,
        int norm_out_kbp,
        __nv_bfloat16 *norm_out_b,
        uint32_t norm_f32_keep_from,
        float *split,
        float *pre_carry,
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
    float *pc = pre_carry + (uint64_t)t * 4u;
    __shared__ float hc4_c[16];
    __shared__ float pre_in[4];
    if (d < 4u) pre_in[d] = pc[d];                 /* the pre handed to this sublayer */
    if (d < 32u) hc4_split_par(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv, d, hc4_c);
    __syncthreads();
    if (d < 4u) pc[d] = sp[d];                     /* this sublayer's pre, for the next one */

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
                acc += pulsar_hc_load(residual_hc, rbase + (uint64_t)h * n_embd + col) * pre_in[h];
            }
            /* hc_pre returns y.to(x.dtype): the collapsed row is bf16 before
             * the norm sees it (RMSNorm then does its math in fp32 on it) */
            acc = __bfloat162float(__float2bfloat16(acc));
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
        /* (weight * x).to(bf16): the normed row is bf16 -- the value every
         * consumer sees, f32 plane, bf16 plane and the E4M3 quant alike */
        const float v = (col < n_embd)
                ? __bfloat162float(__float2bfloat16(accs[u] * norm_scale * pulsar_w_load_f32_or_bf16<NWBF16>(norm_w, col)))
                : 0.0f;
        if (col < n_embd) {
            /* Row-conditional f32: below keep_from every consumer reads an
             * emitted encoding (E4M3 slot / bf16 slot), so the f32 write is a
             * dead store.  Rows >= keep_from stay f32 for the readers that
             * genuinely want those bytes -- the ratio-4 compressor rebuild
             * reads the LAST FOUR rows through an offset view.  The engine
             * passes 0 (store all) unless its skip predicate holds. */
            if (t >= norm_f32_keep_from) norm_out[obase + col] = v;
            /* The bf16 copy, from the same register value -- an emission
             * beside f32/E4M3, so the BF16-weight GEMMs stop paying a
             * separate whole-tensor convert (L086 T3). */
            if (norm_out_b) norm_out_b[obase + col] = __float2bfloat16(v);
        }
        /* Warp-uniform: every lane must reach the shuffle.  BLK is a multiple of
         * 32 and columns are contiguous within a warp, so a warp spans exactly
         * one MX block and lanes past n_embd contribute 0 to the max. */
        if (norm_out_q) {
            pulsar_mx_emit_block(v, col, t, n_embd, norm_out_kbp, norm_out_q, norm_out_sf);
        }
    }
}





/* make_identity_pre_mix: the pre handed to layer 0's attention -- copy 0
 * with weight 1, the rest 0 -- one row per token of the sweep. */
__global__ static void hc_pre_identity_kernel(float *pre, uint32_t n_rows, uint32_t n_hc) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_rows * n_hc) return;
    pre[gid] = (gid % n_hc) == 0u ? 1.0f : 0.0f;
}



__device__ static float softplus_dev(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}



__device__ __forceinline__ static bool router_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}



/* One warp per token: each lane holds NE/32 experts, the top-K selection is a
 * K-round warp argmax with ties broken toward the lower expert id (torch.topk
 * order on equal scores).  Scores select (with the correction bias), the raw
 * sqrt(softplus) probabilities weight; the normalisation is the reference's
 * `weights / (weights.sum() + 1e-20) * route_scale` in fp32.  V4.1 (L218):
 * 384 experts / top-6 on the target, 128 / top-3 on the DSpark drafter --
 * the two instantiations below; there are no hash-routed layers any more. */
template <uint32_t NE, uint32_t TOPK>
__global__ static void router_select_warp_topk_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const float *logits,
        uint32_t n_tokens,
        float route_scale) {
    static_assert(NE % 32u == 0u, "experts per lane must be whole");
    constexpr uint32_t PER = NE / 32u;
    const uint32_t lane = threadIdx.x;
    const uint32_t row_in_block = threadIdx.y;
    const uint32_t t = blockIdx.x * blockDim.y + row_in_block;
    if (t >= n_tokens || lane >= 32u) return;

    const float *log = logits + (uint64_t)t * NE;
    float *prob = probs ? probs + (uint64_t)t * NE : NULL;
    int32_t *sel = selected + (uint64_t)t * TOPK;
    float *w = weights + (uint64_t)t * TOPK;
    float local_prob[PER];
    float local_score[PER];

    #pragma unroll
    for (uint32_t j = 0; j < PER; j++) {
        const uint32_t e = lane + j * 32u;
        const float p = sqrtf(softplus_dev(log[e]));
        local_prob[j] = p;
        local_score[j] = p + bias[e];
        if (prob) prob[e] = p;
    }

    float out_prob[TOPK];
    uint32_t out_idx[TOPK];
    #pragma unroll
    for (uint32_t k = 0; k < TOPK; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < PER; j++) {
            const uint32_t e = lane + j * 32u;
            const float sc = local_score[j];
            if (router_score_better(sc, e, best_score, best_idx)) {
                best_score = sc;
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
        for (uint32_t j = 0; j < PER; j++) {
            const uint32_t e = lane + j * 32u;
            if (e == best_idx) local_score[j] = -INFINITY;
        }
        out_idx[k] = best_idx;
        out_prob[k] = best_prob;
    }

    if (lane == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < TOPK; j++) {
            sel[j] = (int32_t)out_idx[j];
            w[j] = out_prob[j];
            sum += out_prob[j];
        }
        const float inv = route_scale / (sum + 1.0e-20f);
        #pragma unroll
        for (uint32_t j = 0; j < TOPK; j++) w[j] = w[j] * inv;
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
/* Gate/up loads for the templated swiglu below.  The float overload is a
 * plain load — the float instantiation is instruction-identical to the
 * pre-template kernel.  The __half overload DEFUSES the narrow-store edge
 * (L033 increment 1): the gate clamp below is UPPER-bound-only, so a
 * large-negative gate that overflowed f16 to -inf would give
 * s = -inf/(1+expf(+inf)) = -inf/inf = NaN where the f32 path yields -0.
 * Pulling ±inf to ±max-finite keeps swiglu(-huge) ≈ -0 (the f32 answer) and
 * is the IDENTITY on every finite stored value; NaN is deliberately left to
 * propagate exactly as the f32 path would. */
__device__ static inline float swiglu_act_load(const float *p, uint32_t i) { return p[i]; }
__device__ static inline float swiglu_act_load(const __half *p, uint32_t i) {
    float v = __half2float(p[i]);
    if (isinf(v)) v = copysignf(65504.0f, v);
    return v;
}

template <typename AT>
__global__ static void swiglu_kernel(float *out, const AT *gate, const AT *up, uint32_t n, float clamp, float weight,
                                     __nv_fp8_e4m3 *out_q, unsigned char *out_sf, int out_kbp, uint32_t mid_dim) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = pulsar_swiglu_elem(swiglu_act_load(gate, i), swiglu_act_load(up, i), weight, clamp);
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
    /* gate/up carry their own element size (L033 increment 2: prefill stages
     * them f16; decode's fused path passes f32 scratch and takes the float
     * instantiation untouched).  They must AGREE — one f16 and one f32 means a
     * caller narrowed half a pair, the [[L035]] shape. */
    const uint32_t act_esz = pulsar_tensor_esz(gate);
    if (!out || !gate || !up ||
        act_esz != pulsar_tensor_esz(up) ||
        (act_esz != sizeof(float) && act_esz != sizeof(__half)) ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        gate->bytes < (uint64_t)n * act_esz ||
        up->bytes < (uint64_t)n * act_esz) {
        if (gate && up && pulsar_tensor_esz(gate) != pulsar_tensor_esz(up))
            fprintf(stderr, "pulsar: swiglu gate/up element sizes disagree (%u vs %u) -- refusing\n",
                    pulsar_tensor_esz(gate), pulsar_tensor_esz(up));
        return 0;
    }
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
    if (act_esz == sizeof(__half)) {
        /* Announce once: a byte gate cannot distinguish "reading the f16
         * staging" from "the narrowing never went live" (same rule as the
         * skip announcement above). */
        static int announced = 0;
        if (!announced) {
            announced = 1;
            fprintf(stderr, "pulsar: swiglu reading F16 gate/up staging (n=%u mid_dim=%u)\n",
                    n, mid_dim);
        }
        swiglu_kernel<<<(n + 255) / 256, 256>>>(skip_f32 ? NULL : (float *)out->ptr,
                                                (const __half *)gate->ptr, (const __half *)up->ptr, n, clamp, weight,
                                                (__nv_fp8_e4m3 *)out_q, (unsigned char *)out_sf, out_kbp, mid_dim);
    } else {
        swiglu_kernel<<<(n + 255) / 256, 256>>>(skip_f32 ? NULL : (float *)out->ptr,
                                                (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight,
                                                (__nv_fp8_e4m3 *)out_q, (unsigned char *)out_sf, out_kbp, mid_dim);
    }
    return cuda_ok(cudaGetLastError(), "swiglu launch");
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



int pulsar_gpu_router_select_batch_tensor(pulsar_gpu_tensor *selected, pulsar_gpu_tensor *weights, pulsar_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, const pulsar_gpu_tensor *logits, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_tokens) {
    if (!selected || !weights || !logits || !model_map || n_tokens == 0 ||
        logits->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        (probs && probs->bytes < (uint64_t)n_tokens * n_expert * sizeof(float)) ||
        selected->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(float)) {
        return 0;
    }
    /* the correction bias (exp_probs_b) is part of every V4.1 router, target
     * and drafter alike -- there is no bias-less arm */
    const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
    if (bias_offset > model_size || model_size - bias_offset < bias_bytes) return 0;
    const float *bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, bias_bytes, "router_bias");
    if (!bias) return 0;
    dim3 block(32, 4, 1);
    const dim3 grid((n_tokens + 3u) / 4u);
    int32_t *sel = (int32_t *)selected->ptr;
    float *w = (float *)weights->ptr;
    float *pr = probs ? (float *)probs->ptr : NULL;
    const float *lg = (const float *)logits->ptr;
    /* the two routers this engine serves: the V4.1 target and its DSpark drafter */
    if (n_expert == 384u && n_expert_used == 6u) {
        router_select_warp_topk_kernel<384u, 6u><<<grid, block>>>(sel, w, pr, bias, lg, n_tokens, expert_weight_scale);
    } else if (n_expert == 128u && n_expert_used == 3u) {
        router_select_warp_topk_kernel<128u, 3u><<<grid, block>>>(sel, w, pr, bias, lg, n_tokens, expert_weight_scale);
    } else {
        fprintf(stderr, "pulsar: router_select has no arm for %u experts / top-%u -- refusing\n", n_expert, n_expert_used);
        return 0;
    }
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





int pulsar_gpu_hc_split_weighted_sum_norm_f16_tensor(
        pulsar_gpu_tensor       *out,
        pulsar_gpu_tensor       *norm_out,
        void                    *norm_out_q,
        void                    *norm_out_sf,
        int                      norm_out_kbp,
        void                    *norm_out_b,
        uint32_t                 norm_f32_keep_from,
        pulsar_gpu_tensor       *split,
        pulsar_gpu_tensor       *pre_carry,
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
    if (!norm_out || !split || !pre_carry || !mix || !residual_hc || !model_map ||
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
        pre_carry->bytes < n_rows * (uint64_t)n_hc * sizeof(float) ||
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
#define PULSAR_HCFUSED_VEC 20u   /* 256 x 20 = 5120 = V4.1 n_embd (L218); the body bounds-checks col < n_embd */
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
                (__nv_bfloat16 *)norm_out_b,                                         \
                norm_f32_keep_from,                                                  \
                (float *)split->ptr,                                                 \
                (float *)pre_carry->ptr,                                             \
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




int pulsar_gpu_hc_pre_identity_tensor(pulsar_gpu_tensor *pre, uint32_t n_rows, uint32_t n_hc) {
    if (!pre || n_rows == 0 || n_hc == 0 || pre->bytes < (uint64_t)n_rows * n_hc * sizeof(float)) return 0;
    const uint32_t n = n_rows * n_hc;
    hc_pre_identity_kernel<<<(n + 255u) / 256u, 256u>>>((float *)pre->ptr, n_rows, n_hc);
    return cuda_ok(cudaGetLastError(), "hc pre identity launch");
}



int pulsar_gpu_hc_expand_split_tensor(pulsar_gpu_tensor *out_hc, const pulsar_gpu_tensor *block_out, const pulsar_gpu_tensor *residual_hc, const pulsar_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * PULSAR_HC_ELT_SIZE));
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_launch((uint32_t)((n_elem + 255) / 256), 256,
                      (pulsar_hc_t *)out_hc->ptr,
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
    hc_expand_launch((uint32_t)((n_elem + 255) / 256), 256,
                      (pulsar_hc_t *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_add->ptr,
                                                    (const pulsar_hc_t *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 1);
    return cuda_ok(cudaGetLastError(), "hc_expand_add_split launch");
}

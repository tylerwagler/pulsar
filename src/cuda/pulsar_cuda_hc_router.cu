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
 * L219: the thread mapping is one thread per (t,d), computing ALL NHC
 * destinations.  The old per-(t,dst,d) mapping re-read the same NHC residual
 * values and the same block_out element once per destination -- 4x the residual
 * traffic, and residual_hc is the bulk of this op's bytes ([t][hc][embd], read
 * once here instead of four times).  Gathering the residuals once and looping
 * the destinations in registers changes nothing about the per-output
 * accumulation order, so it stays bit-exact by construction.
 *
 * BIT-EXACT: the per-output accumulation order is unchanged (acc starts at
 * block_v*post[dst], then src_hc 0..NHC-1 in order), so this is an issue-order
 * and dedupe change only.  It is graded by the byte-exact prefill gate, not the
 * reference gate. */
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
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if constexpr (NHC > 0) {
        const uint64_t n_elem = (uint64_t)n_tokens * n_embd;
        if (gid >= n_elem) return;
        const uint32_t d = (uint32_t)(gid % n_embd);
        const uint32_t t = (uint32_t)(gid / n_embd);

        float block_v = block_out[(uint64_t)t * n_embd + d];
        if (has_add) block_v += block_add[(uint64_t)t * n_embd + d];
        float res_v[NHC];
#pragma unroll
        for (int src = 0; src < NHC; src++) {
            res_v[src] = pulsar_hc_load(residual_hc,
                    (uint64_t)t * (uint64_t)NHC * n_embd + (uint64_t)src * n_embd + d);
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
        for (int dst = 0; dst < NHC; dst++) {
            float acc = block_v * post[(uint64_t)t * post_stride + dst];
#pragma unroll
            for (int src = 0; src < NHC; src++) {
                const float comb_v = comb[(uint64_t)t * comb_stride + dst + (uint64_t)src * (uint64_t)NHC];
                acc += comb_v * res_v[src];
            }
            pulsar_hc_store(out_hc,
                    (uint64_t)t * (uint64_t)NHC * n_embd + (uint64_t)dst * n_embd + d, acc);
        }
    } else {
        const uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
        if (gid >= n_elem) return;
        const uint32_t d = (uint32_t)(gid % n_embd);
        const uint64_t tmp = gid / n_embd;
        const uint32_t dst_hc = (uint32_t)(tmp % n_hc);
        const uint32_t t = (uint32_t)(tmp / n_hc);

        float block_v = block_out[(uint64_t)t * n_embd + d];
        if (has_add) block_v += block_add[(uint64_t)t * n_embd + d];
        float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
        for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
            float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
            float res_v = pulsar_hc_load(residual_hc, (uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d);
            acc += comb_v * res_v;   /* matches the unrolled arm above */
        }
        pulsar_hc_store(out_hc, (uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d, acc);
    }
}



/* One dispatch point for the three hc_expand callers.  PULSAR_N_HC is 4 on the
 * shipped artifact; anything else takes the runtime-loop instantiation, so a
 * differently-shaped model still runs (just without the unrolled gather).
 * The grid differs by arm: NHC=4 maps one thread per (t,d), the runtime arm one
 * per (t,dst,d). */
static void hc_expand_launch(uint32_t threads,
                             pulsar_hc_t *out_hc, const float *block_out,
                             const float *block_add, const pulsar_hc_t *residual_hc,
                             const float *post, const float *comb,
                             uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens,
                             uint32_t post_stride, uint32_t comb_stride, int has_add) {
    const uint64_t elems = n_hc == 4u ? (uint64_t)n_tokens * n_embd
                                      : (uint64_t)n_tokens * n_hc * n_embd;
    const uint32_t blocks = (uint32_t)((elems + threads - 1u) / threads);
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
/* Where the collapse takes its coefficients -- see pulsar_gpu.h's setter.  A
 * host-side fact pushed once at load, beside the KV row style and for the same
 * reason: it is a property of the loaded profile, not of the call.  It reaches
 * the kernel as a LAUNCH ARGUMENT rather than a __device__ symbol because the
 * setter runs during weight binding, before the graph's first command batch. */
static bool g_hc_head_mix = false;
void pulsar_gpu_set_hc_head_mix(bool on) { g_hc_head_mix = on; }

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
        float norm_eps,
        int head_mix) {
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
                /* 0731 collapses with the split this sublayer just derived;
                 * V4.1 with the pre its predecessor handed on. */
                acc += pulsar_hc_load(residual_hc, rbase + (uint64_t)h * n_embd + col) *
                       (head_mix ? sp[h] : pre_in[h]);
            }
            /* hc_pre returns y.to(x.dtype): the collapsed row is bf16 before
             * the norm sees it (RMSNorm then does its math in fp32 on it).
             * That narrowing is V4.1's reference; 0731's kernel -- the one that
             * served this artifact -- does not narrow, and narrowing here cost
             * V4 a bf16 ULP in attn_norm that grew into a different answer. */
            if (!head_mix) acc = __bfloat162float(__float2bfloat16(acc));
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
                ? (head_mix
                       ? (accs[u] * norm_scale * pulsar_w_load_f32_or_bf16<NWBF16>(norm_w, col))
                       : __bfloat162float(__float2bfloat16(accs[u] * norm_scale * pulsar_w_load_f32_or_bf16<NWBF16>(norm_w, col))))
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
 * order on equal scores).  Scores select (with the correction bias, or with the
 * vision image-token bias on an image slot), the raw sqrt(softplus)
 * probabilities weight; the normalisation is the reference's `weights /
 * (weights.sum() + 1e-20) * route_scale` in fp32.  Three instantiations:
 * 0731's 256 / top-6 target and drafter, V4.1's 384 / top-6 target, and V4.1's
 * 128 / top-3 DSpark drafter. */
/* The K-round warp argmax, ONE implementation for both callers: the dense arm
 * below and the hash kernel's IMAGE arm (a vision image slot on a hash layer
 * must not take the tid2eid path -- that table has a row per real token only,
 * model.py).  `local_score` is consumed (a taken expert is marked -inf) and both
 * arrays are [NE/32] per lane; lane 0 writes the selection and the normalised
 * weights. */
template <uint32_t NE, uint32_t TOPK>
__device__ __forceinline__ static void router_topk_select_warp(
        float (&local_prob)[NE / 32u],
        float (&local_score)[NE / 32u],
        int32_t *sel, float *w, float route_scale) {
    constexpr uint32_t PER = NE / 32u;
    const uint32_t lane = threadIdx.x;
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
        /* dev's arithmetic, exactly: floor the denominator at 2^-14, then DIVIDE
         * and multiply per element.  Folding the scale into a reciprocal
         * (`w * (scale/(sum+1e-20))`) rounds differently from `w/sum*scale` and
         * produced a routing-weight vector that differed from dev on ~1/3 of
         * entries while the logits, the probs and the top-k selection -- which
         * depend only on the scores -- all matched.  The batched kernel and the
         * single-token kernel must also agree with each other; :734 already has
         * this form. */
        sum = fmaxf(sum, 6.103515625e-5f);
        #pragma unroll
        for (uint32_t j = 0; j < TOPK; j++) w[j] = w[j] / sum * route_scale;
    }
}

template <uint32_t NE, uint32_t TOPK>
__global__ static void router_select_warp_topk_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        int has_bias,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t n_tokens,
        const float *vl_bias,
        uint32_t n_vocab,
        int has_vl_bias,
        float route_scale) {
    static_assert(NE % 32u == 0u, "experts per lane must be whole");
    constexpr uint32_t PER = NE / 32u;
    const uint32_t lane = threadIdx.x;
    const uint32_t row_in_block = threadIdx.y;
    const uint32_t t = blockIdx.x * blockDim.y + row_in_block;
    if (t >= n_tokens || lane >= 32u) return;

    /* Vision-Exp: an image slot carries a token id at or above the vocabulary
     * -- the reference emits out-of-vocab sentinels for every image position
     * (model.py: image ids are vocab_size + type).  Such a position routes
     * with bias_vl INSTEAD of the text bias, and on a hash layer it must NOT
     * take the tid2eid path, because that table only has a row per real token
     * (model.py: topk(scores + bias_vl) where the image mask holds).  Only the
     * SELECTION is biased: the routing weights come from the unbiased scores,
     * matching the reference's softmax over the selected logits. */
    const int32_t tok_id = tokens ? tokens[t] : token_scalar;
    const int is_image = has_vl_bias && tok_id >= (int32_t)n_vocab;
    const int use_bias = is_image ? 1 : has_bias;
    const float *use_bias_row = is_image ? vl_bias : bias;

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
        local_score[j] = p + (use_bias ? use_bias_row[e] : 0.0f);
        if (prob) prob[e] = p;
    }

    router_topk_select_warp<NE, TOPK>(local_prob, local_score, sel, w, route_scale);
}

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

/* The HASH-ROUTED arm -- 0731's leading layers (pulsar_shape::n_hash_layer),
 * which name their experts by TOKEN ID instead of selecting them from the
 * logits.  Restored for the two-profile engine (PLAN 96 s16); V4.1 has no such
 * layer.
 *
 * The logits are still needed: the table names the experts, but the WEIGHT is
 * the expert's sqrt(softplus(logit)) -- the bias-free probability, the same
 * value the top-k arm stores in `probs`, NOT the bias-corrected score.  So this
 * kernel takes no text bias at all.
 *
 * A separate kernel rather than an arm inside router_select_warp_topk_kernel:
 * this one needs an NE-wide shared tile of probabilities, and paying 4-6 KB of
 * shared in the V4.1 instantiations would cost occupancy on a path that never
 * hash-routes.  Being separate, V4.1's router is bit-for-bit untouched.
 *
 * dev's constants are kept exactly: the sum floor 2^-14 (not the top-k arm's
 * 1e-20) and the route scale. */
template <uint32_t NE, uint32_t TOPK>
__global__ static void router_select_hash_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        const float *vl_bias,
        uint32_t n_vocab,
        int has_vl_bias,
        float route_scale) {
    static_assert(NE % 32u == 0u, "experts per lane must be whole");
    constexpr uint32_t PER = NE / 32u;
    const uint32_t lane = threadIdx.x;
    const uint32_t row_in_block = threadIdx.y;
    const uint32_t t = blockIdx.x * blockDim.y + row_in_block;
    if (t >= n_tokens || lane >= 32u) return;

    const float *log = logits + (uint64_t)t * NE;
    int32_t *sel = selected + (uint64_t)t * TOPK;
    float *w = weights + (uint64_t)t * TOPK;
    const int32_t tok_id = tokens ? tokens[t] : token_scalar;
    const int is_image = has_vl_bias && tok_id >= (int32_t)n_vocab;
    __shared__ float sprob[4][NE];   /* 4 = the block's y dim */

    #pragma unroll
    for (uint32_t j = 0; j < PER; j++) {
        const uint32_t e = lane + j * 32u;
        const float p = sqrtf(softplus_dev(log[e]));
        sprob[row_in_block][e] = p;
        if (probs) probs[(uint64_t)t * NE + e] = p;
    }
    __syncwarp();

    /* Vision-Exp: an image slot does NOT take the table path -- its id is
     * vocab_size + type and the table has a row per real token only -- so it
     * runs the same vl-biased top-k the dense arm runs (dev's per-row rule). */
    if (is_image) {
        float local_prob[PER];
        float local_score[PER];
        #pragma unroll
        for (uint32_t j = 0; j < PER; j++) {
            const uint32_t e = lane + j * 32u;
            local_prob[j] = sprob[row_in_block][e];
            local_score[j] = sprob[row_in_block][e] + vl_bias[e];
        }
        router_topk_select_warp<NE, TOPK>(local_prob, local_score, sel, w, route_scale);
        return;
    }

    /* One lane does the table walk and the normalisation -- there is one row of
     * table per token, so spreading it over the warp would need a broadcast for
     * no gain. */
    if (lane == 0) {
        int32_t tok = tok_id;
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;   /* fail closed on a bad id */
        const int32_t *row = hash + (uint64_t)tok * TOPK;
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < TOPK; j++) {
            const int32_t e = row[j];
            sel[j] = e;
            const float v = (e >= 0 && (uint32_t)e < NE) ? sprob[row_in_block][(uint32_t)e] : 0.0f;
            w[j] = v;
            sum += v;
        }
        /* dev's floor, not the top-k arm's 1e-20 */
        sum = fmaxf(sum, 6.103515625e-5f);
        #pragma unroll
        for (uint32_t j = 0; j < TOPK; j++) w[j] = w[j] / sum * route_scale;
    }
}


int pulsar_gpu_router_select_batch_tensor(
        pulsar_gpu_tensor *selected, pulsar_gpu_tensor *weights, pulsar_gpu_tensor *probs,
        const void *model_map, uint64_t model_size,
        uint64_t bias_offset, bool has_bias,
        uint64_t tid2eid_offset, uint32_t tid2eid_rows,
        const pulsar_gpu_tensor *logits, const pulsar_gpu_tensor *tokens,
        uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_tokens,
        uint64_t vl_bias_offset, uint32_t n_vocab, bool has_vl_bias) {
    if (!selected || !weights || !logits || !model_map || n_tokens == 0 ||
        logits->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        (probs && probs->bytes < (uint64_t)n_tokens * n_expert * sizeof(float)) ||
        selected->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(float)) {
        return 0;
    }
    /* The correction bias is OPTIONAL.  Both the target's blk.N.exp_probs_b.bias
     * and the drafter's dspark.N.exp_probs_b.bias are absent from shipped
     * artifacts -- Vision-Exp's serving file among them -- and `has_bias` is the
     * only thing that can distinguish "absent" from "offset 0", which is a real
     * tensor offset.  A routed layer without it scores sqrt(softplus(logit))
     * alone.  (dev had the same arm; this file used to insist the bias was part
     * of every router, which made those artifacts unloadable.) */
    const float *bias = NULL;
    if (has_bias) {
        const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
        if (bias_offset > model_size || model_size - bias_offset < bias_bytes) return 0;
        bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, bias_bytes, "router_bias");
        if (!bias) return 0;
    }
    /* Vision-Exp: the image-token bias, one n_expert-wide row per layer.  A
     * text-only artifact has no such tensor and this stays NULL, which leaves
     * the kernel's behaviour bit-identical to before. */
    const float *vl_bias = NULL;
    if (has_vl_bias) {
        const uint64_t vl_bytes = (uint64_t)n_expert * sizeof(float);
        if (vl_bias_offset > model_size || model_size - vl_bias_offset < vl_bytes) return 0;
        vl_bias = (const float *)cuda_model_range_ptr(model_map, vl_bias_offset, vl_bytes, "router_bias_vl");
        if (!vl_bias) return 0;
    }
    const int hb = has_bias ? 1 : 0;
    const int hv = has_vl_bias ? 1 : 0;
    const int32_t *tokp = tokens ? (const int32_t *)tokens->ptr : NULL;
    dim3 block(32, 4, 1);
    const dim3 grid((n_tokens + 3u) / 4u);
    int32_t *sel = (int32_t *)selected->ptr;
    float *w = (float *)weights->ptr;
    float *pr = probs ? (float *)probs->ptr : NULL;
    const float *lg = (const float *)logits->ptr;

    /* The HASH-routed arm takes precedence when the artifact carries a token-id
     * table: the ids come from the table, not from a top-k over the logits.
     * tid2eid_rows is the table's row count (the vocab size); 0 means the model
     * has no such table, so no separate presence flag is needed. */
    if (tid2eid_rows != 0) {
        if (n_expert != 256u || n_expert_used != 6u) {
            fprintf(stderr, "pulsar: a hash-routed layer with %u experts / top-%u has no "
                            "arm -- refusing\n", n_expert, n_expert_used);
            return 0;
        }
        if (!tokens) {
            /* The lookup is per TOKEN.  dev's kernel fell back to a scalar id
             * here; guessing one would route every row of a decode batch to the
             * same experts, silently.  The engine's only caller passes the
             * prompt tokens, so a NULL here means a lane we have not met --
             * refuse and name it rather than mis-route. */
            fprintf(stderr, "pulsar: hash routing needs the token ids and none were "
                            "passed -- refusing\n");
            return 0;
        }
        const uint64_t table_bytes = (uint64_t)tid2eid_rows * n_expert_used * sizeof(int32_t);
        if (tid2eid_offset > model_size || model_size - tid2eid_offset < table_bytes ||
            tokens->bytes < (uint64_t)n_tokens * sizeof(int32_t)) return 0;
        const int32_t *hash = (const int32_t *)cuda_model_range_ptr(model_map, tid2eid_offset,
                                                                   table_bytes, "router_tid2eid");
        if (!hash) return 0;
        router_select_hash_kernel<256u, 6u><<<grid, block>>>(
                sel, w, pr, hash, lg, tokp, 0,
                tid2eid_rows, n_tokens, vl_bias, n_vocab, hv, expert_weight_scale);
        return cuda_ok(cudaGetLastError(), "router_select hash launch");
    }

    /* the routers this engine serves: the V4.1 target, 0731's target and its
     * DSpark drafter (the drafter routes with its OWN width -- 128/top-3 on
     * V4.1 against a 384/top-6 target) */
    if (n_expert == 384u && n_expert_used == 6u) {
        router_select_warp_topk_kernel<384u, 6u><<<grid, block>>>(sel, w, pr, bias, hb, lg, tokp, 0, n_tokens, vl_bias, n_vocab, hv, expert_weight_scale);
    } else if (n_expert == 256u && n_expert_used == 6u) {
        router_select_warp_topk_kernel<256u, 6u><<<grid, block>>>(sel, w, pr, bias, hb, lg, tokp, 0, n_tokens, vl_bias, n_vocab, hv, expert_weight_scale);
    } else if (n_expert == 128u && n_expert_used == 3u) {
        router_select_warp_topk_kernel<128u, 3u><<<grid, block>>>(sel, w, pr, bias, hb, lg, tokp, 0, n_tokens, vl_bias, n_vocab, hv, expert_weight_scale);
    } else {
        fprintf(stderr, "pulsar: router_select has no arm for %u experts / top-%u -- refusing\n", n_expert, n_expert_used);
        return 0;
    }
    return cuda_ok(cudaGetLastError(), "router_select launch");
}





/* ---- the HC HEAD MIX (0731) -----------------------------------------------
 *
 * The final collapse is SHARED with V4.1 (pulsar_gpu_hc_weighted_sum_tensor
 * below); what differs is where its n_hc coefficients come from.  V4.1 reads
 * the pre its last FFN handed on; 0731 computes them: a fused RMSNorm + mix
 * GEMV, then a sigmoid with a per-stream bias and a row scale.  Restored from
 * dev (PLAN 96 s17); these are the only two 0731-specific HC kernels.
 *
 * Reachable only when the artifact bound output_hc_* / dspark.2.hc_head_*, so
 * V4.1 never runs them. */

__global__ static void output_hc_weights_kernel(
        float *out,
        const float *pre,
        const float *scale,
        const float *base,
        uint32_t n_hc,
        uint32_t n_tokens,
        float epsv) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = n_tokens * n_hc;
    if (gid >= n) return;
    const uint32_t h = gid % n_hc;
    const float z = pre[gid] * scale[0] + base[h];
    out[gid] = 1.0f / (1.0f + expf(-z)) + epsv;
}

/* Fused RMSNorm + mix GEMV.  Byte-identical to the rms_norm_plain -> matmul
 * pair it replaces, which ran a 1-block kernel and then a 24-block kernel with a
 * 64 KB f32 scratch round trip between them (~5.4% of decode, by dev's own
 * measurement).
 *
 * The roundings are PINNED so that no future contraction or reassociation
 * decision by nvcc can quietly move this off byte-identical: __fmul_rn
 * reproduces the f32 that rms_norm_plain stored, and __fmaf_rn reproduces the
 * generic GEMV's contracted `sum += w * x`.  Do not "tidy" these into plain
 * operators. */
template <uint32_t BLK, uint32_t UNROLL, typename WT>
__global__ static void hc_norm_mix_kernel(
        float *out,
        const WT *w,
        const pulsar_hc_t *x,
        uint32_t n,
        uint32_t out_dim,
        float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= out_dim) return;          /* block-uniform: depends only on blockIdx */
    const uint32_t tid = threadIdx.x;
    __shared__ float partial[BLK];

    /* stage 1: plain RMSNorm scale over x -- same order as rms_norm_plain */
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
        const float v = pulsar_hc_load(x, i);
        sum += v * v;
    }
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t stride = BLK >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    __syncthreads();                     /* partial[] is reused below */

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

/* The sigmoid step: out = sigmoid(pre * scale + base[h]) + eps.  `scale` is a
 * single f32 (a per-ROW scale, not per-stream), `base` is n_hc wide. */
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
    if (out->bytes < row_bytes || out->bytes % row_bytes != 0 || pre->bytes < out->bytes ||
        scale_offset > model_size || sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || row_bytes > model_size - base_offset) {
        return 0;
    }
    const uint64_t n_tokens = out->bytes / row_bytes;
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, sizeof(float), "output_hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, row_bytes, "output_hc_base");
    if (!scale || !base) return 0;
    const uint64_t n = n_tokens * n_hc;
    output_hc_weights_kernel<<<(uint32_t)((n + 255) / 256), 256>>>(
            (float *)out->ptr, (const float *)pre->ptr, scale, base,
            n_hc, (uint32_t)n_tokens, eps);
    return cuda_ok(cudaGetLastError(), "output hc weights launch");
}

/* w_type is the ds4 tensor type of the mix weight: 30 BF16, 0 F32.  Templated
 * on STORAGE rather than gated on one type: it used to be F16-only, which made
 * the fusion silently unreachable the moment hc_*_fn moved storage.  Anything
 * else REFUSES rather than falling through to an f16 read at the wrong width. */
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
    const uint64_t elt = (w_type == PULSAR_TENSOR_F32) ? 4u : 2u;   /* F32 : BF16 */
    const uint64_t weight_bytes = out_dim * in_dim * elt;
    if (weight_bytes > model_size - weight_offset) return 0;
    /* x is an HC residual carrier: PULSAR_HC_ELT_SIZE bytes per sample. */
    if (x->bytes < in_dim * PULSAR_HC_ELT_SIZE || out->bytes < out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "hc_mix");
    if (!wptr) return 0;
#define PULSAR_HCMIX(WT, CAST)                                              \
    hc_norm_mix_kernel<256, 8, WT><<<(uint32_t)out_dim, 256>>>(             \
            (float *)out->ptr, (const CAST)wptr, (const pulsar_hc_t *)x->ptr, \
            (uint32_t)in_dim, (uint32_t)out_dim, eps)
    if (w_type == PULSAR_TENSOR_BF16)      PULSAR_HCMIX(__nv_bfloat16, __nv_bfloat16 *);
    else if (w_type == PULSAR_TENSOR_F32)  PULSAR_HCMIX(float, float *);
    else {
        fprintf(stderr, "pulsar: hc_mix weight type %u is neither BF16 nor F32\n", w_type);
        return 0;
    }
#undef PULSAR_HCMIX
    return cuda_ok(cudaGetLastError(), "hc norm mix launch");
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
/* The unroll must land EXACTLY on n_embd where it can.  A group past the end is
 * not free: every unrolled step calls pulsar_mx_emit_block, and a dead group
 * still runs through its shuffles, so a width that does not divide n_embd
 * perturbs the emitted E4M3 plane.  That is the s51 bug, and dev's V4 build is
 * exactly 256*16 == 4096 == n_embd with no dead group at all.  V4.1's 5120
 * needs 20.  Both are selected here, on the tensor's own shape. */
#define PULSAR_HCFUSED_VEC_4096 16u
#define PULSAR_HCFUSED_VEC_5120 20u
    /* Fail loud rather than ever silently skip the emit again.  There WAS a
     * generic fallback for wider models; it is deleted -- it could not emit
     * E4M3 at all (its parameter list had no norm_out_q) while the caller still
     * marked the activation slot valid, so every MXFP8 consumer downstream read
     * the previous layer's data against zeroed scales. */
    if (n_embd > PULSAR_HCFUSED_BLK * PULSAR_HCFUSED_VEC_5120) {
        fprintf(stderr, "pulsar: hc fused norm cannot handle n_embd=%u (max %u)\n",
                n_embd, PULSAR_HCFUSED_BLK * PULSAR_HCFUSED_VEC_5120);
        return 0;
    }
    {
#define PULSAR_HCFUSED_LAUNCH(VEC, NW)                                               \
        hc_split_weighted_sum_norm_fused_kernel<PULSAR_HCFUSED_BLK, VEC, NW>          \
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
                n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps, norm_eps,     \
                g_hc_head_mix ? 1 : 0)
#define PULSAR_HCFUSED_LAUNCH_NW(VEC)                                                \
        do { if (norm_w_bf16) PULSAR_HCFUSED_LAUNCH(VEC, true);                       \
             else             PULSAR_HCFUSED_LAUNCH(VEC, false); } while (0)
        if (n_embd <= PULSAR_HCFUSED_BLK * PULSAR_HCFUSED_VEC_4096)
            PULSAR_HCFUSED_LAUNCH_NW(PULSAR_HCFUSED_VEC_4096);
        else
            PULSAR_HCFUSED_LAUNCH_NW(PULSAR_HCFUSED_VEC_5120);
#undef PULSAR_HCFUSED_LAUNCH_NW
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
    const float *base = (const float *)split->ptr;
    hc_expand_launch(256,
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
    const float *base = (const float *)split->ptr;
    hc_expand_launch(256,
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

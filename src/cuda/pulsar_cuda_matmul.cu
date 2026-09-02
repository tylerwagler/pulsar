#include "pulsar_cuda_internal.h"
#include "pulsar_cuda_mx.cuh"   /* the single source for pulsar_mx_sfoff */



/* token_embd is BF16 (source format). It was F16 until 2026-08-16, and briefly
 * read through a loader whose non-bf16 branch decoded F32 -- 4 bytes per element
 * off a 2-byte table, a gigabyte past the end, faulting asynchronously so the
 * error surfaced in an unrelated CUDA call. Typed pointer now: a width mismatch
 * is a compile error. */
__global__ static void embed_token_hc_kernel(pulsar_hc_t *out, const __nv_bfloat16 *w, uint32_t token, uint32_t n_vocab, uint32_t n_embd, uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    uint32_t e = i % n_embd;
    uint32_t tok = token < n_vocab ? token : n_vocab - 1;  ///< clamp: an OOB token id is a wild global read
    pulsar_hc_store(out, i, pulsar_wt_load(w, (uint64_t)tok * n_embd + e));
}



__global__ static void embed_tokens_hc_kernel(
        pulsar_hc_t *out,
        const int32_t *tokens,
        const __nv_bfloat16 *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    pulsar_hc_store(out, gid, pulsar_wt_load(w, (uint64_t)tok * n_embd + d));
}





/* Activation load, keyed on the ACTIVATION's own storage type.  A BF16 weight
 * is fed a BF16 activation buffer and an f32 weight an f32 one, so the pointer
 * type is the whole contract and a caller cannot pair them wrongly.
 *
 * WHY BF16 AT ALL.  The source model runs a BF16 activation stream
 * (ds4-source-numerics: torch_dtype bfloat16, E4M3 only where the weight is
 * quantised, and NO f32 activations anywhere), so multiplying a BF16 weight by
 * an f32 activation is over-precision on one side of the product: it does not
 * buy fidelity, it diverges from the reference.
 *
 * WHY A BUFFER RATHER THAN ROUNDING IN-REGISTER.  f161083 narrowed at the load
 * -- pulsar_act_bf16(x[i]) -- which was correct arithmetic and the wrong shape,
 * and it MEASURED as +2.9% step time.  In a GEMV every one of out_dim/8 warps
 * walks the whole activation row, so narrowing at the load re-rounds the same
 * in_dim values once per warp: hundreds of redundant conversions per launch.
 * Converting once into a buffer the whole launch shares does it in_dim times
 * total, and has the second, larger benefit that the nt, n==1 and cuBLASLt arms
 * then read THE SAME BYTES instead of three separately-derived encodings of the
 * same number. */
__device__ __forceinline__ static float pulsar_at_load(const __nv_bfloat16 *x, uint64_t i) {
    return __bfloat162float(x[i]);
}


/* Small-batch (2..4 token) f16 GEMV: one weight-row read serves all NT tokens,
 * replacing the cuBLAS GemmEx path which is latency-bound at these shapes and
 * needs an f32->f16 activation convert + tmp alloc per call. Per-token loop
 * structure and shared-memory reduction match the n=1 kernel exactly, so each
 * token's output is bit-identical to that kernel run on the token alone. */
/* Weight load for the nt kernel, one overload per storage. Overloads rather
 * than a bool template: the pointer type IS the contract, so a mismatched
 * width cannot be passed silently (which is exactly how the embed kernels came
 * to read an f16 table as f32 on 2026-08-15). */
template <int NT, typename WT, typename AT = float>
__global__ static void matmul_nt_kernel(
        float *out,
        const WT *w,
        const AT *x,
        uint64_t in_dim,
        uint64_t out_dim) {
    uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out_dim) return;

    float sum[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) sum[t] = 0.0f;
    const WT *wr = w + row * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const float wv = pulsar_wt_load(wr, i);
        #pragma unroll
        for (int t = 0; t < NT; t++) sum[t] += wv * pulsar_at_load(x, t * in_dim + i);
    }

    __shared__ float partial[256];
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        partial[threadIdx.x] = sum[t];
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) out[t * out_dim + row] = partial[0];
        __syncthreads();
    }
}




/* L109 N1 verdict (2026-08-25, tried and REVERTED same day): split-K for the
 * skinny hc outputs ran at 9.7us vs the 62us one-block-per-column launches --
 * a 6x kernel win worth ~1 ms/token -- but the f32 reassociation (known-flip
 * class) perturbed the drafter's target-anchor conditioning and measured
 * -8% structured tokens/step in the server decomposition A/B (prose neutral).
 * The acceptance tax exceeds the kernel gain. A bit-exact variant does not
 * exist at grid=24 (any repartition of the serial lane sums reassociates).
 * Reopen only with a drafter retuned on the new numerics (L092 refit), or if
 * the hc family stops feeding drafter anchors. */

/* BF16 is the high 16 bits of f32, so conversions are pure bit ops (no header). */
__device__ __forceinline__ static float bf16_to_f32(uint16_t b) {
    return __uint_as_float((uint32_t)b << 16);
}


__global__ static void f32_to_bf16_kernel(uint16_t *out, const float *x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const uint32_t u = __float_as_uint(x[i]);
        out[i] = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);  ///< round-to-nearest-even
    }
}


__global__ static void matmul_bf16_kernel(
        float *out,
        const uint16_t *w,
        const __nv_bfloat16 *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    float sum = 0.0f;
    const uint16_t *wr = w + row * in_dim;
    const __nv_bfloat16 *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += bf16_to_f32(wr[i]) * __bfloat162float(xr[i]);
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}



/* MXFP8 weight block (33B: [E8M0][32 e4m3]) . raw f32 activation block.
 * Decode-side attn-output dot: the e4m3 weight is expanded to float and MAC'd
 * against the unquantized activation, so the only quantization error left on
 * this path is the weight's own (e4m3-packing the activations pushed the
 * perplexity gate; int8 requant is what this change removes). The contiguous
 * 32-element inner loop is what nvcc vectorizes — a lane-per-element
 * (mmvq-style) mapping profiled 2.2x slower. Returns wscale*sum(e4m3*x). */


/* De-interleaved MXFP8 block dot: contiguous 32-E4M3 block (aligned) + separate
 * E8M0 scale byte. Reads 8 aligned uint32 (4 fp8 each) -> vectorized, vs the raw
 * 33B interleaved kernels' misaligned byte-wise weight reads. */
__device__ __forceinline__ static float dev_dot_fp8mx_deint_block(
        const __nv_fp8_e4m3 *blk, unsigned char scale_byte, const float *xb) {
    const float wscale = __int_as_float((uint32_t)scale_byte << 23);  ///< E8M0
    float s = 0.0f;
#pragma unroll
    for (int g = 0; g < 8; g++) {
        const uint32_t p = ((const uint32_t *)blk)[g];
        const __nv_fp8_e4m3 *q = (const __nv_fp8_e4m3 *)&p;
#pragma unroll
        for (int j = 0; j < 4; j++) s += __half2float((__half)q[j]) * xb[g * 4 + j];
    }
    return wscale * s;
}


/* Register-operand core: the weight block already staged as 8 packed words and
 * its E8M0 scale decoded.  The pointer form below is this function with the
 * load in front of it -- ONE arithmetic body, so a kernel that stages a weight
 * block once and dots it against N tokens (grouped_fp8mx_a_nt_a8_kernel) gets
 * every token's value from the same expression the one-token kernels use. */
__device__ __forceinline__ static float dev_dot_fp8mx_deint_block_a8_w(
        const uint32_t wp[8], float sw,
        const __nv_fp8_e4m3 *xblk, unsigned char xscale_byte) {
    const float sa = __int_as_float((uint32_t)xscale_byte << 23);  ///< E8M0
    float s = 0.0f;
    /* Same (g, j) traversal as the f32 helper so the summation order matches. */
#pragma unroll
    for (int g = 0; g < 8; g++) {
        const uint32_t xp = ((const uint32_t *)xblk)[g];
        const __nv_fp8_e4m3 *wq = (const __nv_fp8_e4m3 *)&wp[g];
        const __nv_fp8_e4m3 *xq = (const __nv_fp8_e4m3 *)&xp;
#pragma unroll
        for (int j = 0; j < 4; j++)
            s += __half2float((__half)wq[j]) * __half2float((__half)xq[j]);
    }
    return sw * sa * s;
}

__device__ __forceinline__ static float dev_dot_fp8mx_deint_block_a8(
        const __nv_fp8_e4m3 *wblk, unsigned char wscale_byte,
        const __nv_fp8_e4m3 *xblk, unsigned char xscale_byte) {
    uint32_t wp[8];
#pragma unroll
    for (int g = 0; g < 8; g++) wp[g] = ((const uint32_t *)wblk)[g];
    return dev_dot_fp8mx_deint_block_a8_w(wp, __int_as_float((uint32_t)wscale_byte << 23),
                                          xblk, xscale_byte);
}


/* Stage one 32-element activation block into registers from whatever width the
 * activation is STORED at (L033).
 *
 * T=float keeps the exact eight-float4 sequence the two grouped-"a" GEMVs had
 * before they were templated -- same instructions, same order -- so their f32
 * arms stay bit-identical.  A narrowed T issues four 128-bit loads of eight
 * elements each and widens on the way into the block; the dot helpers below
 * still see f32, so no arithmetic depends on the storage width.
 *
 * Both grouped-"a" GEMVs staged this with the same copy-pasted loop.  One
 * helper now, so a future width lands in one place instead of two that have to
 * be kept agreeing by hand. */
template <typename T>
__device__ __forceinline__ static void stage_x_block32(const T *xr, float *xb);

template <>
__device__ __forceinline__ void stage_x_block32<__nv_bfloat16>(const __nv_bfloat16 *xr, float *xb) {
    /* 8 bf16 is 16 B: four 128-bit transactions per 32-wide block.  The f32
     * arm (eight of them) went with the last f32-activation caller; the primary
     * template is declared only, so a new width fails to link, not silently. */
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const uint4 raw = *(const uint4 *)&xr[(uint32_t)k * 8u];
        const __nv_bfloat16 *h = (const __nv_bfloat16 *)&raw;
#pragma unroll
        for (int j = 0; j < 8; j++) xb[k * 8 + j] = __bfloat162float(h[j]);
    }
}

/* T is the activation's STORED type, deduced from the pointer.  Callers on f32
 * buffers are unchanged; (float)x[i] is the identity for T=float. */
template <typename T>
__device__ __forceinline__ static float dev_dot_fp8mx_f32_block(
        const unsigned char *wblk, const T *x, uint64_t bn) {
    const float wscale = __int_as_float((uint32_t)wblk[0] << 23);  ///< E8M0
    float s = 0.0f;
    for (uint64_t i = 0; i < bn; i++) {
        const __half wh = (__half)(*(const __nv_fp8_e4m3 *)&wblk[1 + i]);
        s += __half2float(wh) * (float)x[i];
    }
    return wscale * s;
}



/* Same dot against an activation block already staged in registers (full
 * 32-element block only). */
__device__ __forceinline__ static float dev_dot_fp8mx_xreg_block(
        const unsigned char *wblk, const float *xb) {
    const float wscale = __int_as_float((uint32_t)wblk[0] << 23);  ///< E8M0
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; i++) {
        const __half wh = (__half)(*(const __nv_fp8_e4m3 *)&wblk[1 + i]);
        s += __half2float(wh) * xb[i];
    }
    return wscale * s;
}



/* Fused attn-output kernels: MXFP8 weight rows dotted against RAW f32
 * activations. Each warp covers PULSAR_FP8MX_ROWS consecutive output rows and
 * loads every 32-float activation block into registers ONCE for all of them,
 * so per-row activation traffic drops to int8-path parity (per-row f32 reads
 * straight from global measured ~13% slower end-to-end at decode; a shared-
 * memory staging variant measured worse still). */
/* Sweep knob (2026-08-30).  The 4 above amortises one activation-block load
 * across 4 output rows, and the comment records that 1 row/warp measured ~13%
 * SLOWER end-to-end -- but that was the f32-activation era.  Activations are
 * E4M3 now (1 byte, not 4), so the traffic this amortisation saves is worth 4x
 * less than when it was measured, while ncu (2026-08-30) puts the a8 kernel at
 * 1.33 waves/SM with long_scoreboard 41 and memory only 24% busy: starved of
 * parallelism with bandwidth to spare.  Fewer rows per warp trades activation
 * traffic for waves, and is BIT-EXACT (a row's lane-strided accumulation over K
 * does not depend on how many other rows its warp also carries).
 * Override: -DPULSAR_FP8MX_ROWS=2 */
#ifndef PULSAR_FP8MX_ROWS
#define PULSAR_FP8MX_ROWS 4
#endif
/* Output rows a whole 8-warp block covers -- every grid below derives from this
 * rather than spelling 32, which silently meant "8 warps x 4 rows". */
#define PULSAR_FP8MX_ROWS_PER_BLOCK (8u * (unsigned)PULSAR_FP8MX_ROWS)

/* The A8 twin gets its OWN rows-per-warp, and the split is physical rather than
 * a tuning whim: it reads E4M3 activations (1 byte/element) where the f32 twin
 * reads 4, so the activation traffic that 4-rows-per-warp amortises costs a
 * quarter as much here -- and that amortisation is the only thing 4 buys.
 *
 * Measured 2026-08-30 (nsys, pulsar-bench, 3 alternating reps, kernel GPU time):
 *   ROWS_A8=4   848.11 ms median (spread 4.9)
 *   ROWS_A8=1   814.42 ms median (spread 0.3)   -4.0%, arms do not overlap
 * total GPU 8042 -> 8022 ms. BIT-EXACT: probe hash 7d6ef2dfd081fca4 unchanged
 * across every arm, because a row's lane-strided accumulation over K does not
 * depend on how many other rows its warp carries.
 *
 * ncu said why: at 4 rows this kernel ran 1.33 waves/SM with long_scoreboard 41
 * and memory only 24% busy -- stalled on loads, bandwidth to spare, too few
 * blocks to hide the latency. Fewer rows per warp buys waves.
 *
 * The f32 twin and the _nt_ variant deliberately KEEP 4: the header's "~13%
 * slower for 1 row/warp" was measured in the f32-activation era and still
 * applies to them, and _nt_ also serves the server's 2..4-row speculative
 * verify batches, which this sweep did not measure. */
#ifndef PULSAR_FP8MX_ROWS_A8
#define PULSAR_FP8MX_ROWS_A8 1
#endif
#define PULSAR_FP8MX_ROWS_A8_PER_BLOCK (8u * (unsigned)PULSAR_FP8MX_ROWS_A8)



/* A8 adds the activation as E4M3 + its own ue8m0 scale, so the attn-output "b"
 * projection multiplies in the source's format. Only the ACCUMULATE differs --
 * the hc_expand epilogue (residual mix, split, carrier store) is identical, so
 * it is a template parameter rather than a second kernel. A8 implies DEINT: the
 * raw 33B reader has no E4M3-activation form and the launcher never pairs them.
 * xdata/xscale are the per-slot cache layout (flat [in_dim] at n_tok 1, scale
 * at pulsar_mx_sfoff(0, kb, xKBp)) -- NOT the group-major grouped layout the "a"
 * projection uses. Two caches, two layouts, one letter apart in the call. */
template<bool DEINT, bool A8 = false>
__global__ static void matmul_fp8mx_hc_expand_warp8_kernel(
        pulsar_hc_t *out_hc,
        float *block_out,
        const float *block_add,
        const pulsar_hc_t *residual_hc,
        const float *split,
        const unsigned char *w,
        const __nv_fp8_e4m3 *wdata,
        const unsigned char *wscale,
        int KBp,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t blocks,
        int has_add,
        const __nv_fp8_e4m3 *xdata = nullptr,
        const unsigned char *xscale = nullptr,
        int xKBp = 0) {
    const uint64_t row0 = ((uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u)) * PULSAR_FP8MX_ROWS;
    const uint32_t lane = threadIdx.x & 31u;
    if (row0 >= out_dim) return;
    const uint32_t nr = out_dim - row0 < PULSAR_FP8MX_ROWS ? (uint32_t)(out_dim - row0)
                                                        : (uint32_t)PULSAR_FP8MX_ROWS;
    const unsigned char *wr = w + row0 * blocks * 33u;
    float acc[PULSAR_FP8MX_ROWS] = {};
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        if (bn == 32u) {
            float xb[32];
            if (!A8) {
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    *(float4 *)&xb[k * 4] = *(const float4 *)&x[i0 + (uint32_t)k * 4u];
                }
            }
            const unsigned char xsb = A8 ? xscale[pulsar_mx_sfoff(0, (int)b, xKBp)] : (unsigned char)0;
#pragma unroll
            for (uint32_t r = 0; r < PULSAR_FP8MX_ROWS; r++) {
                if (r >= nr) continue;
                if (A8) {
                    const uint32_t rw = (uint32_t)(row0 + r);
                    acc[r] += dev_dot_fp8mx_deint_block_a8(
                            wdata + (uint64_t)rw * in_dim + i0,
                            wscale[pulsar_mx_sfoff((int)rw, (int)b, KBp)],
                            xdata + i0, xsb);
                } else if (DEINT) {
                    const uint32_t rw = (uint32_t)(row0 + r);
                    acc[r] += dev_dot_fp8mx_deint_block(wdata + (uint64_t)rw * in_dim + i0,
                                                        wscale[pulsar_mx_sfoff((int)rw, (int)b, KBp)], xb);
                } else {
                    acc[r] += dev_dot_fp8mx_xreg_block(wr + (r * blocks + b) * 33u, xb);
                }
            }
        } else {
            for (uint32_t r = 0; r < nr; r++) {
                acc[r] += dev_dot_fp8mx_f32_block(wr + (r * blocks + b) * 33u, x + i0, bn);
            }
        }
    }
    for (uint32_t r = 0; r < nr; r++) {
        const float red = warp_sum_f32(acc[r]);
        if (lane != 0) continue;
        const uint32_t d = (uint32_t)(row0 + r);
        block_out[d] = red;
        float block_v = red;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = pulsar_hc_load(residual_hc, (uint64_t)src_hc * n_embd + d);
                hc_acc += comb_v * res_v;
            }
            pulsar_hc_store(out_hc, (uint64_t)dst_hc * n_embd + d, hc_acc);
        }
    }
}



/* XT is the activation's STORED element type (L033); f32 instantiation is the
 * pre-template code exactly. */
template<bool DEINT, typename XT>
__global__ static void grouped_fp8mx_a_warp8_kernel(
        float *low,
        const unsigned char *w,
        const __nv_fp8_e4m3 *wdata,
        const unsigned char *wscale,
        int KBp,
        const XT *x,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks) {
    const uint64_t row0 = ((uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u)) * PULSAR_FP8MX_ROWS;
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row0 >= low_dim || tok >= n_tokens) return;
    const uint32_t nr = low_dim - row0 < PULSAR_FP8MX_ROWS ? (uint32_t)(low_dim - row0)
                                                        : (uint32_t)PULSAR_FP8MX_ROWS;
    const unsigned char *wr = w + row0 * blocks * 33u;
    const uint64_t group = row0 / rank;
    float acc[PULSAR_FP8MX_ROWS] = {};

    if ((row0 + nr - 1) / rank == group) {
        /* Common case (rank % PULSAR_FP8MX_ROWS == 0): all rows share the
         * group's activation row, so its blocks are loaded once per warp. */
        const XT *xr = x + (tok * (uint64_t)n_groups + group) * group_dim;
        for (uint64_t b = lane; b < blocks; b += 32u) {
            const uint64_t i0 = b * 32;
            const uint64_t bn = group_dim - i0 < 32 ? group_dim - i0 : 32;
            if (bn == 32u) {
                float xb[32];
                stage_x_block32<XT>(xr + i0, xb);
#pragma unroll
                for (uint32_t r = 0; r < PULSAR_FP8MX_ROWS; r++) {
                    if (r >= nr) continue;
                    if (DEINT) {
                        const uint32_t rw = (uint32_t)(row0 + r);
                        acc[r] += dev_dot_fp8mx_deint_block(wdata + (uint64_t)rw * group_dim + i0,
                                                            wscale[pulsar_mx_sfoff((int)rw, (int)b, KBp)], xb);
                    } else {
                        acc[r] += dev_dot_fp8mx_xreg_block(wr + (r * blocks + b) * 33u, xb);
                    }
                }
            } else {
                for (uint32_t r = 0; r < nr; r++) {
                    acc[r] += dev_dot_fp8mx_f32_block(wr + (r * blocks + b) * 33u, xr + i0, bn);
                }
            }
        }
    } else {
        for (uint32_t r = 0; r < nr; r++) {
            const XT *xr = x + (tok * (uint64_t)n_groups + (row0 + r) / rank) * group_dim;
            for (uint64_t b = lane; b < blocks; b += 32u) {
                const uint64_t i0 = b * 32;
                const uint64_t bn = group_dim - i0 < 32 ? group_dim - i0 : 32;
                acc[r] += dev_dot_fp8mx_f32_block(wr + (r * blocks + b) * 33u, xr + i0, bn);
            }
        }
    }
    for (uint32_t r = 0; r < nr; r++) {
        const float red = warp_sum_f32(acc[r]);
        if (lane == 0) low[tok * low_dim + row0 + r] = red;
    }
}


/* A8 twin of grouped_fp8mx_a_warp8_kernel: both operands E4M3 with their own
 * ue8m0 block scales, so the attn-output "a" projection multiplies in the
 * source's activation format instead of against f32.
 *
 * Activation layout is the GROUPED cache's, which is NOT the f32 input's:
 * mxfp8_quant_act_grouped_kernel writes data GROUP-MAJOR,
 * [(g * n_tokens + tok) * group_dim + i], while the f32 heads it reads are
 * token-major [(tok * n_groups + g) * group_dim + i]. Getting that transpose
 * wrong reads a well-formed activation belonging to another group, which is a
 * wrong answer with no symptom -- so it is spelled out here rather than
 * inferred at the call site.
 *
 * Only the common case: the caller guarantees a de-interleaved weight,
 * rank % PULSAR_FP8MX_ROWS == 0 (a row quad never straddles a group) and
 * group_dim % 32 == 0 (whole blocks, so bn is always 32). Anything else keeps
 * the f32 path, which is a FALLBACK not a format choice -- see the launcher. */

__global__ static void grouped_fp8mx_a_warp8_a8_kernel(
        float *low,
        const __nv_fp8_e4m3 *wdata,
        const unsigned char *wscale,
        int KBp,
        const __nv_fp8_e4m3 *xdata,
        const unsigned char *xscale,
        int xKBp,
        uint64_t scale_slab,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks) {
    const uint64_t row0 = ((uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u)) * PULSAR_FP8MX_ROWS_A8;
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row0 >= low_dim || tok >= n_tokens) return;
    const uint32_t nr = low_dim - row0 < PULSAR_FP8MX_ROWS_A8 ? (uint32_t)(low_dim - row0)
                                                        : (uint32_t)PULSAR_FP8MX_ROWS_A8;
    const uint64_t group = row0 / rank;
    float acc[PULSAR_FP8MX_ROWS_A8] = {};

    const __nv_fp8_e4m3 *xr = xdata + (group * (uint64_t)n_tokens + tok) * group_dim;
    const unsigned char *xs = xscale + group * scale_slab;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const unsigned char xsb = xs[pulsar_mx_sfoff((int)tok, (int)b, xKBp)];
#pragma unroll
        for (uint32_t r = 0; r < PULSAR_FP8MX_ROWS_A8; r++) {
            if (r >= nr) continue;
            const uint32_t rw = (uint32_t)(row0 + r);
            acc[r] += dev_dot_fp8mx_deint_block_a8(
                    wdata + (uint64_t)rw * group_dim + i0,
                    wscale[pulsar_mx_sfoff((int)rw, (int)b, KBp)],
                    xr + i0, xsb);
        }
    }
    for (uint32_t r = 0; r < nr; r++) {
        const float red = warp_sum_f32(acc[r]);
        if (lane == 0) low[tok * low_dim + row0 + r] = red;
    }
}


/* N-token twin of grouped_fp8mx_a_warp8_a8_kernel (L141).  That kernel
 * launches grid.y = n_tokens: every token walks the whole attn_output_a
 * (8192 x 4096 E4M3 = 33.5 MB) on its own, so a 3-row step read it three
 * times -- 147 us at one row, 419 at three (L140), while the dense nt_a8
 * projections amortise at 1.05x.  The f32 nt fusion two kernels down used to
 * do this for 2..4 tokens and became unreachable when A8 took every n.
 *
 * Here each warp-row stages a weight block ONCE and dots it against all NT
 * tokens.  Per token, the block order (lane-strided b, then warp_sum) and the
 * dot body are the one-token kernel's exactly, so each row of `low` is
 * bit-identical to what the per-token grid produced -- which is also what
 * keeps a co-scheduled bank's output independent of its batchmates (gate 4).
 * n_tokens >= NT is the dispatcher's contract (it launches NT == n_tokens).
 * Activation layout is the grouped cache's, group-major, as above. */
template <int NT>
__global__ static void grouped_fp8mx_a_nt_a8_kernel(
        float *low,
        const __nv_fp8_e4m3 *wdata,
        const unsigned char *wscale,
        int KBp,
        const __nv_fp8_e4m3 *xdata,
        const unsigned char *xscale,
        int xKBp,
        uint64_t scale_slab,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks) {
    const uint64_t row0 = ((uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u)) * PULSAR_FP8MX_ROWS_A8;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row0 >= low_dim) return;
    const uint32_t nr = low_dim - row0 < PULSAR_FP8MX_ROWS_A8 ? (uint32_t)(low_dim - row0)
                                                        : (uint32_t)PULSAR_FP8MX_ROWS_A8;
    const uint64_t group = row0 / rank;
    float acc[PULSAR_FP8MX_ROWS_A8][NT];
#pragma unroll
    for (uint32_t r = 0; r < PULSAR_FP8MX_ROWS_A8; r++)
#pragma unroll
        for (int t = 0; t < NT; t++) acc[r][t] = 0.0f;

    /* token t's activation row sits at xg + t * group_dim (group-major). */
    const __nv_fp8_e4m3 *xg = xdata + group * (uint64_t)n_tokens * group_dim;
    const unsigned char *xs = xscale + group * scale_slab;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        unsigned char xsb[NT];
#pragma unroll
        for (int t = 0; t < NT; t++) xsb[t] = xs[pulsar_mx_sfoff(t, (int)b, xKBp)];
#pragma unroll
        for (uint32_t r = 0; r < PULSAR_FP8MX_ROWS_A8; r++) {
            if (r >= nr) continue;
            const uint32_t rw = (uint32_t)(row0 + r);
            const uint32_t *wblk = (const uint32_t *)(wdata + (uint64_t)rw * group_dim + i0);
            uint32_t wp[8];
#pragma unroll
            for (int g = 0; g < 8; g++) wp[g] = wblk[g];
            const float sw = __int_as_float(
                    (uint32_t)wscale[pulsar_mx_sfoff((int)rw, (int)b, KBp)] << 23);
#pragma unroll
            for (int t = 0; t < NT; t++)
                acc[r][t] += dev_dot_fp8mx_deint_block_a8_w(wp, sw,
                                                            xg + (uint64_t)t * group_dim + i0,
                                                            xsb[t]);
        }
    }
    for (uint32_t r = 0; r < nr; r++) {
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float red = warp_sum_f32(acc[r][t]);
            if (lane == 0) low[(uint64_t)t * low_dim + row0 + r] = red;
        }
    }
}


/* Small-batch (2..4 token) variant of the grouped o_a GEMV: one launch whose
 * weight blocks are read once and served from L1 across the NT tokens' dots,
 * replacing the per-group block-scaled tensor-core GEMMs that dominated the
 * spec-verify launch storm (8 GEMMs/layer at 2-4 rows each). Per-(row,token)
 * block order and dot helper match grouped_fp8mx_a_warp8_kernel's DEINT fast
 * path exactly, so each token's output is bit-identical to the n=1 kernel.
 * Caller guarantees: deint weight available, rank % PULSAR_FP8MX_ROWS_A8 == 0 (row
 * quads never straddle a group), group_dim % 32 == 0 (whole blocks). */
template <int NT, typename XT>
__global__ static void grouped_fp8mx_a_nt_kernel(
        float *low,
        const __nv_fp8_e4m3 *wdata,
        const unsigned char *wscale,
        int KBp,
        const XT *x,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint64_t blocks) {
    const uint64_t row0 = ((uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u)) * PULSAR_FP8MX_ROWS;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row0 >= low_dim) return;
    const uint64_t group = row0 / rank;
    float acc[PULSAR_FP8MX_ROWS][NT];
#pragma unroll
    for (uint32_t r = 0; r < PULSAR_FP8MX_ROWS; r++)
#pragma unroll
        for (int t = 0; t < NT; t++) acc[r][t] = 0.0f;
    const XT *xg = x + group * group_dim;
    const uint64_t tok_stride = (uint64_t)n_groups * group_dim;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
#pragma unroll
        for (int t = 0; t < NT; t++) {
            float xb[32];
            const XT *xr = xg + (uint64_t)t * tok_stride + i0;
            stage_x_block32<XT>(xr, xb);
#pragma unroll
            for (uint32_t r = 0; r < PULSAR_FP8MX_ROWS; r++) {
                const uint32_t rw = (uint32_t)(row0 + r);
                acc[r][t] += dev_dot_fp8mx_deint_block(wdata + (uint64_t)rw * group_dim + i0,
                                                       wscale[pulsar_mx_sfoff((int)rw, (int)b, KBp)], xb);
            }
        }
    }
#pragma unroll
    for (uint32_t r = 0; r < PULSAR_FP8MX_ROWS; r++) {
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float red = warp_sum_f32(acc[r][t]);
            if (lane == 0) low[(uint64_t)t * low_dim + row0 + r] = red;
        }
    }
}



int pulsar_gpu_embed_token_hc_tensor(pulsar_gpu_tensor *out_hc, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !model_map || weight_offset >= model_size || n_vocab == 0) return 0;
    /* The kernel writes n_embd*n_hc carrier samples; validate like the batched
     * sibling does. Before the BF16 narrowing an undersized out_hc still had 2x
     * slack — it does not any more, so this check is load-bearing (task #62). */
    if (out_hc->bytes < (uint64_t)n_embd * n_hc * PULSAR_HC_ELT_SIZE) return 0;
    uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "token_embd");
    if (!wptr) return 0;
    uint32_t n = n_embd * n_hc;
    embed_token_hc_kernel<<<(n + 255) / 256, 256>>>((pulsar_hc_t *)out_hc->ptr,
            (const __nv_bfloat16 *)wptr, token, n_vocab, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed token launch");
}



int pulsar_gpu_embed_tokens_hc_tensor(
        pulsar_gpu_tensor       *out_hc,
        const pulsar_gpu_tensor *tokens_t,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !tokens_t || !model_map ||
        weight_offset > model_size ||
        (uint64_t)n_vocab * n_embd * sizeof(uint16_t) > model_size - weight_offset ||
        tokens_t->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
        out_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd * PULSAR_HC_ELT_SIZE) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset,
                                            (uint64_t)n_vocab * n_embd * sizeof(uint16_t),
                                            "token_embd");
    if (!wptr) return 0;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    embed_tokens_hc_kernel<<<(n + 255) / 256, 256>>>(
        (pulsar_hc_t *)out_hc->ptr,
        (const int32_t *)tokens_t->ptr,
        (const __nv_bfloat16 *)wptr,
        n_vocab, n_tokens, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed tokens launch");
}


static int g_cublaslt_ready = 0;


static int cublaslt_ensure(void) {
    if (g_cublaslt_ready) return 1;
    if (cublasLtCreate(&g_cublaslt) != CUBLAS_STATUS_SUCCESS) return 0;
    g_cublaslt_ready = 1; return 1;
}


/* mx_rup moved to pulsar_cuda_mx.cuh (shared with the cutlass packers). */
#define mx_rup pulsar_mx_rup
/* Reservation arithmetic for the scratch arena: every cuda_arena_take here
 * 256-aligns, so the reservation has to budget the ALIGNED size of each
 * slice.  Budgeting the raw sizes is how a caller reserves less than it
 * then asks for and trips the arena's latch on the last take. */
static inline size_t mx_a256(size_t b) { return (b + 255u) & ~(size_t)255u; }




/* X[rows,K] f32 -> E4M3 data[K,rows]col + swizzled E8M0 scale. one warp per (row,kb). */
__global__ static void mxfp8_quant_act_kernel(const float *X, int rows, int K, int KBp,
                                              __nv_fp8_e4m3 *data, unsigned char *scale) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    int KB = K / 32; if (warp >= rows * KB) return;
    int row = warp / KB, kb = warp % KB;
    float v = X[(size_t)row * K + kb * 32 + lane], a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const int se = pulsar_mx_shared_exp(a);
    data[(size_t)(kb * 32 + lane) + (size_t)row * K] = pulsar_mx_encode(v, se);
    if (lane == 0) scale[pulsar_mx_sfoff(row, kb, KBp)] = pulsar_mx_scale_byte(se);
}


/* Grouped variant for the attn-output "a" projection. The activation tensor is
 * heads[tok][group][K] but each group is an independent GEMM, so the E4M3 data
 * is regrouped as n_groups slabs of [K, n_tokens] col-major and the E8M0 scale
 * as n_groups independent swizzles of scale_slab bytes (token = scale row). */
/* T is the activation's STORED type (L033).  Both callers quantise heads. */
template <typename T>
__global__ static void mxfp8_quant_act_grouped_kernel(const T *X, int n_tokens, int n_groups,
                                                      int K, int KBp, __nv_fp8_e4m3 *data,
                                                      unsigned char *scale, size_t scale_slab) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    int KB = K / 32; if (warp >= n_tokens * n_groups * KB) return;
    int row = warp / KB, kb = warp % KB;  ///< row = tok * n_groups + g
    int g = row % n_groups, tok = row / n_groups;
    float v = (float)X[(size_t)row * K + kb * 32 + lane], a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const int se = pulsar_mx_shared_exp(a);
    data[((size_t)g * n_tokens + tok) * K + kb * 32 + lane] = pulsar_mx_encode(v, se);
    if (lane == 0) scale[(size_t)g * scale_slab + pulsar_mx_sfoff(tok, kb, KBp)] = pulsar_mx_scale_byte(se);
}


/* GGUF MXFP8 weight (row-major [out,in], 33B blocks: [E8M0][32xE4M3]) ->
 * E4M3 data[in,out]col + swizzled E8M0 scale. one warp per (out,kb). */


/* CONCURRENCY (multi-stream decode, mid-roadmap): this map + the g_fp8_fc_*
 * front cache are mutated on the cold submit path (lazy insert / rehash), so a
 * second decode thread would hit iterator invalidation. thread_local is the
 * intended fix, but it must be done together with g_mxfp8_lt_offsets below
 * (which decides the pre-stored fast path) so the two do not desync -- not a
 * blind keyword change. Do it as part of the multi-stream work. */
static std::unordered_map<uint64_t, fp8_mx_weight> g_fp8_mx_by_offset;

/* Offsets whose MXFP8 weight is PRE-STORED in the mmap in the exact device
 * layout (de-interleaved E4M3 data + pulsar_mx_sfoff-swizzled E8M0 scale, contiguous:
 * [data (in*out B)][scale]). For these the resolver skips cudaMalloc+convert and
 * points cuBLASLt straight at g_model_device_base+offset. Populated once at load
 * (cold path); the resolved fp8_mx_weight is then cached in g_fp8_mx_by_offset
 * exactly like a converted weight, so the per-token hot path never probes this
 * set. */
static std::unordered_set<uint64_t> g_mxfp8_lt_offsets;

/* Direct-mapped front cache for cuda_fp8_mx_weight (file-scope so backend
 * cleanup can invalidate it together with g_fp8_mx_by_offset). */
constexpr uint32_t FP8_FC = 2048u;
static uint64_t g_fp8_fc_off[FP8_FC];  ///< zero-init; real offsets are never 0
static const fp8_mx_weight *g_fp8_fc_ptr[FP8_FC];


/* lazily de-interleave + swizzle an MXFP8 weight into device buffers, cached by offset. */
static const fp8_mx_weight *cuda_fp8_mx_weight(const void *model_map, uint64_t offset, uint64_t weight_bytes,
                                               uint64_t in_dim, uint64_t out_dim, const char *label) {
    /* The same ~300 weight offsets are resolved once per layer every token on
     * the launch-serializing host thread. A tiny direct-mapped cache in front
     * of the unordered_map skips the probe on the hot repeat; a miss or hash
     * collision just falls through (benign), and the cached pointer is
     * re-validated (map references are stable across inserts). */
    constexpr uint32_t FC = FP8_FC;
    uint64_t *fc_off = g_fp8_fc_off;
    const fp8_mx_weight **fc_ptr = g_fp8_fc_ptr;
    const uint32_t slot = (uint32_t)(((offset >> 5) ^ (offset >> 17)) & (FC - 1u));
    if (offset != 0 && fc_off[slot] == offset) {
        const fp8_mx_weight *p = fc_ptr[slot];
        if (p && p->host_base == model_map && p->in_dim == in_dim && p->out_dim == out_dim) return p;
    }
    auto it = g_fp8_mx_by_offset.find(offset);
    if (it != g_fp8_mx_by_offset.end() && it->second.host_base == model_map &&
        it->second.in_dim == in_dim && it->second.out_dim == out_dim) {
        fc_off[slot] = offset; fc_ptr[slot] = &it->second;
        return &it->second;
    }
    int KB = (int)(in_dim / 32), KBp = mx_rup(KB, 4);
    size_t data_bytes = in_dim * out_dim;
    size_t scale_bytes = (size_t)mx_rup((int)out_dim, 128) * KBp;

    /* Pre-stored MXFP8_LT: the de-interleaved E4M3 data and swizzled E8M0 scale
     * are already laid out contiguously in the mmap at [offset .. offset+data ..
     * +scale]. Skip the cudaMalloc+convert entirely and hand cuBLASLt the
     * device-accessible mmap pointers (g_model_device_base+offset). Byte-for-byte
     * identical to what mxfp8_weight_convert_kernel would have produced. */
    if (g_mxfp8_lt_offsets.count(offset)) {
        __nv_fp8_e4m3 *ltdata =
            (__nv_fp8_e4m3 *)cuda_model_range_ptr(model_map, offset, data_bytes, "fp8_mx_lt data");
        unsigned char *ltscale =
            (unsigned char *)cuda_model_range_ptr(model_map, offset + data_bytes, scale_bytes, "fp8_mx_lt scale");
        if (ltdata && ltscale) {
            fp8_mx_weight w = { model_map, offset, in_dim, out_dim, ltdata, ltscale };
            g_fp8_mx_by_offset[offset] = w;
            const fp8_mx_weight *wp = &g_fp8_mx_by_offset[offset];
            fc_off[slot] = offset; fc_ptr[slot] = wp;
            (void)label;
            return wp;
        }
        fprintf(stderr, "pulsar: MXFP8_LT weight at offset %llu did not resolve to "
                "device-accessible mmap pointers\n", (unsigned long long)offset);
        return NULL;
    }

    /* Every MXFP8 weight is pre-stored, so reaching here means an offset was
     * registered as FP8 but not as MXFP8_LT -- i.e. a plain type-38 tensor.
     * Those are refused at load (gguf.cpp), so this is unreachable and stays
     * only to say so rather than returning a silent NULL.
     *
     * What USED to be here: cudaMalloc of the data and scale buffers plus
     * mxfp8_weight_convert_kernel, de-interleaving the 33B-interleaved blocks
     * into exactly the bytes an MXFP8_LT tensor already contains -- a second
     * resident copy of every plain weight beside the mmap.  Deleted 2026-08-17
     * once the artifact reached 390/390 pre-stored; see L060. */
    fprintf(stderr, "pulsar: FP8 weight at offset %llu is not pre-stored MXFP8_LT; "
            "plain type-38 weights are no longer converted at runtime -- repack with "
            "tools/mxfp8_prestore/repack_mxfp8_lt.py\n", (unsigned long long)offset);
    (void)label; (void)weight_bytes; (void)KB;
    return NULL;
}


/* ---- "quantize once, N GEMMs" activation cache --------------------------
 *
 * In the batched prefill path a single normalized activation (batch_attn_norm,
 * [n_tok x n_embd] f32) feeds up to SEVEN block-scaled MXFP8 projections per
 * ratio-4 layer (q_a, kv, attn compressor kv+gate, indexer compressor kv+gate,
 * indexer_proj).  mxfp8_quant_act_kernel is a PURE function of
 * (x, n_tok, in_dim) -- see the kernel: per-warp, no atomics, no state -- so
 * running it once per consumer re-derives byte-identical E4M3 data and E8M0
 * scales while re-reading the whole f32 activation each time.  Cache the first
 * result and hand it to the remaining GEMMs: bit-identical by construction,
 * and it drops ~6/7 of the activation quant traffic on those layers.
 *
 * ALIASING: the cached buffers must NOT come from cuda_tmp_alloc.  That is one
 * process-global scratch region (pulsar_cuda_runtime.cu) that later callers --
 * attention, MoE, the indexer top-k -- freely overwrite or realloc, and such
 * calls DO run between the consumers above.  Dedicated cudaMalloc'd storage is
 * the only safe home for a value that must survive across unrelated kernels.
 *
 * COHERENCE: the cache is keyed on (x->ptr, n_tok, in_dim), which is NOT enough
 * on its own -- batch_attn_norm keeps its pointer and shape while its CONTENTS
 * change every layer.  So a hit additionally requires that the engine has armed
 * the cache for that exact buffer since the last write to it
 * (pulsar_gpu_mxfp8_act_cache_arm, called immediately after the norm that produces
 * it).  Arming invalidates, so a stale hit would require a write with no arm.
 *
 * THREADING: per-thread state, matching the per-thread CUDA streams -- two
 * sessions prefilling concurrently must not share a quantized activation.
 *
 * TWO ENCODINGS, ONE ARMING.  The same activation also feeds BF16 GEMMs (the
 * steering/plain matmul arm), and that conversion is likewise pure, so the
 * slot carries a bf16 copy (valid_b / xb -- see f32_to_bf16_kernel) alongside
 * the E4M3 one and both fill lazily: a layer pays for only the encodings it
 * actually uses.  (This paragraph once described an F16 copy for "F16 cuBLAS
 * GEMMs" and was truncated mid-clause; the last F16 weight left the model and
 * the field is __nv_bfloat16 *xb -- rewritten by L106 V8.) */
/* MULTI-SLOT.  One armed buffer sufficed while only batch_attn_norm and
 * batch_ffn_norm carried producer-emitted encodings, because those two never
 * had to be live at the same instant.  Driving every activation to E4M3 breaks
 * that: a layer holds several at once -- batch_attn_norm (4096) feeds q_a/kv
 * while batch_qr_norm (1024) feeds q_b, and batch_ffn_norm (4096) feeds the
 * router while batch_shared_mid (2048) feeds shared_down.  With a single slot,
 * arming the second silently invalidated the first, and its later consumers
 * re-quantized from f32: a performance loss today, and a WRONG-OPERAND bug the
 * moment that f32 store goes away.
 *
 * ⚠ The single-slot design had an ACCIDENTAL safety property -- arming anything
 * invalidated everything else, so a buffer rewritten with no arm could never
 * serve a stale hit.  Slots keyed independently lose that, so disarm() still
 * clears ALL of them.  batch_ffn_norm is scratch that the output head
 * legitimately reuses under the same (ptr, n_tok, in_dim) key, which is exactly
 * how the C1 stale-logits bug happened, so there are FOUR disarms and they are
 * two different mechanisms -- named by function, because the line numbers this
 * comment used to carry had rotted so far that two of them pointed past the end
 * of the file:
 *
 *   ENCODE EXITS, which is what makes arm/disarm pair within one encode --
 *     gpu_graph_encode_layer_attention_batch, gpu_graph_encode_layer_ffn_batch
 *     (both gpu_prefill.cpp)
 *   HEAD ENTRIES, the redundant second lock on the same door --
 *     gpu_graph_encode_output_head_batch,
 *     gpu_graph_encode_dspark_output_head_batch (both gpu_decode.cpp)
 *
 * (L131 removed the caveat that used to sit here: gpu_graph_encode_decode_layer
 * armed four buffers and disarmed NONE, which was safe only by the accident
 * that its single-token buffers were separate allocations from the batch_*
 * ones -- an aliasing accident rather than an invariant, per the decode audit's
 * D3. That encoder is gone, so the pairing rule now holds everywhere without an
 * exception.) */
#define PULSAR_ACT_SLOTS 6

/** One slot of the per-thread activation quantisation cache.
 *
 * Several GEMMs in a layer consume the SAME activation buffer, so the MXFP8
 * (and bf16) encoding of it is computed once and reused. The cache key is the
 * triple (buffer pointer, token count, input width) -- which is exactly why
 * the arm/disarm discipline documented above matters: a later, unrelated
 * buffer that lands on the same address with the same shape would present a
 * matching key and silently be served a previous tensor's encoding.
 */
struct mxfp8_act_cache_t {
    const void    *key_ptr;  ///< armed activation buffer (NULL = disarmed)
    uint64_t       key_ntok;    ///< token count of the armed buffer; part of the key
    uint64_t       key_in_dim;  ///< input width of the armed buffer; part of the key
    int            valid;  ///< xq/sx hold the MXFP8 quant of that (ptr,shape)
    __nv_fp8_e4m3 *xq;       ///< cached E4M3 activations
    size_t         xq_cap;   ///< bytes allocated for xq
    unsigned char *sx;       ///< cached block scales for xq
    size_t         sx_cap;   ///< bytes allocated for sx
    int            valid_b;  ///< xb holds the bf16 conversion of that (ptr,shape)
    __nv_bfloat16 *xb;       ///< cached bf16 conversion
    size_t         xb_cap;   ///< bytes allocated for xb
    /** The producer emitted the E4M3 encoding and SKIPPED the f32 store, so the
     * f32 buffer holds stale bytes from a previous call.  Any arm that would
     * read it must fail loudly instead: a skipped store read as an operand is a
     * well-formed WRONG answer, not a crash.  See the invariant at
     * act_f32_absent_hazard(). */
    int            f32_absent;
    /** Rows < this had their f32 store skipped; rows >= are present.  A
     * full skip sets it to key_ntok.  Exists because the first flight of
     * the bf16 backstop refused the ratio-4 tail rebuild -- a read of the
     * four rows the skip deliberately KEPT -- and a boolean cannot tell
     * the kept tail from the skipped body. */
    uint32_t       f32_keep_from;
    uint64_t       lru;  ///< eviction stamp; 0 = never used
};
static thread_local mxfp8_act_cache_t g_act_slots[PULSAR_ACT_SLOTS];
static thread_local uint64_t g_act_clock;
/* Slot most recently armed.  The note_*() calls always follow their arm() with
 * no intervening arm, so they act on this rather than re-deriving the key. */
static thread_local mxfp8_act_cache_t *g_act_cur;

static mxfp8_act_cache_t *act_slot_find(const void *ptr, uint64_t n_tok, uint64_t in_dim) {
    if (!ptr) return NULL;
    for (int i = 0; i < PULSAR_ACT_SLOTS; i++) {
        mxfp8_act_cache_t *s = &g_act_slots[i];
        if (s->key_ptr == ptr && s->key_ntok == n_tok && s->key_in_dim == in_dim) {
            s->lru = ++g_act_clock;
            return s;
        }
    }
    return NULL;
}

/* Consumer-side lookup that tolerates a WIDER cache block.
 *
 * The exact-key act_slot_find above is correct for ARMING -- the producer owns
 * the whole (ptr, n_tok, in_dim) block it just wrote.  It is WRONG for
 * CONSUMING, and that asymmetry is what failed GATE 4 on 2026-08-17: in a fused
 * mixed step the norm arms the cache at the FULL batch width (66 rows in the
 * gate), then the inc-4 prefix split asks the matmul for just the decode rows
 * (n_dec=2), misses on n_tok, and silently drops to the f32 arm -- while those
 * same decode rows in a decode-only step hit the cache and take A8.  Same op,
 * two numerics, chosen by batch width: an M-DEPENDENCE hiding inside a cache
 * key, which is exactly what GATE 4 exists to catch.
 *
 * Taking a prefix is safe because the encoding is ROW-LOCAL in both halves: the
 * quantiser reduces amax per (row, kb) warp and stores data at row*K + k, and
 * pulsar_mx_sfoff(row, kb, KBp) is a function of row, kb and KBp only -- never of the
 * total row count.  So rows [0, need) of a width-N block are BYTE-IDENTICAL to
 * a width-need block, and consuming the prefix makes the A8 arms genuinely
 * M-independent rather than only appearing so at the widths a gate happens to
 * exercise.  ⚠ The n==1 A8 arm had the SAME latent defect and no gate caught
 * it, because this gate co-schedules two decode banks and so never ran the
 * n_dec==1 mixed case.
 *
 * Tightest fit wins, so a decode-only step still reads its own exact block. */
static mxfp8_act_cache_t *act_slot_find_rows(const void *ptr, uint64_t need, uint64_t in_dim) {
    if (!ptr || need == 0) return NULL;
    mxfp8_act_cache_t *best = NULL;
    for (int i = 0; i < PULSAR_ACT_SLOTS; i++) {
        mxfp8_act_cache_t *s = &g_act_slots[i];
        if (s->key_ptr == ptr && s->key_in_dim == in_dim && s->key_ntok >= need) {
            if (!best || s->key_ntok < best->key_ntok) best = s;
        }
    }
    if (best) best->lru = ++g_act_clock;
    return best;
}

/* D3: tell a LEGITIMATE miss from a BROKEN one.
 *
 * Every consumer below reads the cache and, on a miss, quietly multiplies
 * against f32 instead.  That makes the activation FORMAT a function of cache
 * state rather than of what the engine meant -- W8A8 and W8A32 chosen by a
 * lookup, with silence as the only signal.  The per-shape "W8A8" announcements
 * do not close this: they fire when A8 DOES run, so the failure mode is a line
 * that never appears, and nobody watches for an absent log line.
 *
 * The two misses are not the same event:
 *
 *   NO SLOT AT ALL -- the producer never armed this buffer, so it never claimed
 *     to have emitted E4M3.  f32 is the correct and intended answer (the
 *     drafter's own forwards are the live example).  Not an error.
 *
 *   A SLOT EXISTS AND IS VALID, BUT COVERS FEWER ROWS THAN ASKED -- the
 *     producer DID declare E4M3 for this buffer and this consumer cannot use
 *     the declaration.  There is no correct f32 answer here: the rows past
 *     key_ntok were never encoded, so falling back silently changes the format
 *     of an operand the engine believes is E4M3.  This is exactly the shape
 *     that failed GATE 4 on 2026-08-17 (arm at full batch width, consume at the
 *     decode prefix) and the shape act_slot_find_rows exists to prevent.
 *
 * So the second case is a defect, and it now says so and fails the launch
 * rather than returning a differently-typed number.  Fail closed, one format or
 * an error -- the same stance routed_moe_launch takes for a declined MMQ. */
static bool act_slot_a8_declared_short(const void *ptr, uint64_t need, uint64_t in_dim) {
    if (!ptr || need == 0) return false;
    for (int i = 0; i < PULSAR_ACT_SLOTS; i++) {
        const mxfp8_act_cache_t *s = &g_act_slots[i];
        if (s->key_ptr == ptr && s->key_in_dim == in_dim && s->valid &&
            s->key_ntok < need) return true;
    }
    return false;
}

/* Report it once per (in,out) shape and fail.  Keyed on the pair for the reason
 * the W8A8 announcements are: several decode GEMVs share in_dim 4096, so an
 * in_dim-only key would report one and hide the rest. */
static int act_a8_contract_fail(const char *what, uint64_t need,
                                uint64_t in_dim, uint64_t out_dim) {
    static uint64_t seen[16] = {0};
    static int n_seen = 0;
    const uint64_t key = (in_dim << 32) ^ out_dim;
    int known = 0;
    for (int i = 0; i < n_seen; i++) if (seen[i] == key) { known = 1; break; }
    if (!known && n_seen < 16) {
        seen[n_seen++] = key;
        fprintf(stderr,
                "pulsar: A8 CONTRACT VIOLATION in %s -- the producer declared E4M3 for "
                "this activation but the cached block is narrower than the %llu rows "
                "asked for (in_dim=%llu out_dim=%llu).  Refusing to silently multiply "
                "against f32 instead.\n",
                what, (unsigned long long)need,
                (unsigned long long)in_dim, (unsigned long long)out_dim);
    }
    return 0;  ///< every consumer of this returns int; 0 = launch failed
}

/* Find this key's slot, or take over the least-recently-used one.  Acquiring
 * RESETS the validity bits: the caller is about to (re)write that buffer, so
 * whatever the slot held is stale by definition.  Buffers are grow-only and
 * survive eviction -- only the key and the validity bits are reassigned. */
static mxfp8_act_cache_t *act_slot_acquire(const void *ptr, uint64_t n_tok, uint64_t in_dim) {
    if (!ptr || n_tok == 0 || in_dim == 0 || (in_dim % 32) != 0) return NULL;
    mxfp8_act_cache_t *s = act_slot_find(ptr, n_tok, in_dim);
    if (!s) {
        s = &g_act_slots[0];
        for (int i = 1; i < PULSAR_ACT_SLOTS; i++) {
            if (g_act_slots[i].lru < s->lru) s = &g_act_slots[i];
        }
        s->key_ptr    = ptr;
        s->key_ntok   = n_tok;
        s->key_in_dim = in_dim;
        s->lru        = ++g_act_clock;
    }
    s->valid = 0; s->valid_b = 0; s->f32_absent = 0; s->f32_keep_from = 0;
    return s;
}

void pulsar_gpu_mxfp8_act_cache_arm(const pulsar_gpu_tensor *x, uint64_t n_tok, uint64_t in_dim) {
    g_act_cur = x ? act_slot_acquire(x->ptr, n_tok, in_dim) : NULL;
}

void pulsar_gpu_mxfp8_act_cache_disarm(void) {
    for (int i = 0; i < PULSAR_ACT_SLOTS; i++) {
        g_act_slots[i].key_ptr    = NULL;
        g_act_slots[i].valid      = 0;
        g_act_slots[i].valid_b    = 0;
        g_act_slots[i].f32_absent = 0;
        g_act_slots[i].f32_keep_from = 0;
    }
    g_act_cur = NULL;
}

/* Returns 1 when this call is about to read an f32 buffer whose store was
 * SKIPPED -- either because no slot covers these rows, or because the covering
 * slot's encoding went invalid (eviction: PULSAR_ACT_SLOTS is finite and the
 * cache is LRU).
 *
 * The invariant that makes one check sufficient, verified arm by arm in
 * cuda_matmul_mxfp8_tensor_labeled: when a covering slot is `valid`, EVERY arm
 * takes an A8 variant -- the n==1 GEMV, the n_tok 2..8 NT kernels
 * (mxfp8_mmvq_deint_nt_a8_kernel, chosen on `ac8nt && ac8nt->valid`), the pair
 * kernel (which gives up its launch fusion rather than run W8A32 on an armed
 * buffer), and the cuBLASLt path (which reuses ac->xq/ac->sx).  f32 is read
 * ONLY when no covering slot is valid.  So the entire hazard is
 * `f32_absent && !valid`, independent of which arm would have run.
 *
 * This is a BACKSTOP, not the mechanism.  In a correct layer it never fires:
 * the producer skips only when it is itself emitting the encoding, and the
 * arm/note follow the emission with no intervening arm.  It exists because that
 * adjacency is a LIFETIME ORDER inside the layer, and reordering the layer
 * would otherwise turn a silent wrong answer loose rather than a loud stop. */
static int act_f32_absent_hazard(const void *ptr, uint64_t n_tok, uint64_t in_dim) {
    if (!ptr) return 0;
    const mxfp8_act_cache_t *cover = act_slot_find_rows(ptr, n_tok, in_dim);
    if (cover && cover->valid && cover->xq && cover->sx) return 0;  ///< served from cache
    /* Matching the BASE pointer is not enough.  Consumers reach these buffers
     * through offset VIEWS -- gpu_prefill.cpp:143 takes the last four rows of
     * batch_attn_norm for the ratio-4 compressor rebuild and hands them to a
     * plain matmul at n_tok=4.  A view keys no slot, so an equality test sees
     * nothing and the GEMM quantizes from bytes that were never written.  Test
     * CONTAINMENT in the skipped buffer's extent instead.
     * (Cf. the standing lesson that consumers bypass the accessor: the bug
     * always lives in the gap between the API and the raw buffer.) */
    for (int i = 0; i < PULSAR_ACT_SLOTS; i++) {
        const mxfp8_act_cache_t *s = &g_act_slots[i];
        if (!s->f32_absent || !s->key_ptr) continue;
        const char *base = (const char *)s->key_ptr;
        const char *end  = base + s->key_ntok * s->key_in_dim * sizeof(float);
        const char *p    = (const char *)ptr;
        if (p < base || p >= end) continue;
        /* Row-granular: the skip may deliberately KEEP a tail (the ratio-4
         * rebuild reads the last four rows of attn_norm).  Absent rows are
         * [0, f32_keep_from); a read is hazardous only if it TOUCHES them.
         * The read's row extent is computed in the SLOT's stride -- if the
         * reader's in_dim disagrees with the slot's, fall back to refusing,
         * because a stride mismatch means we cannot reason about rows at all. */
        const uint64_t off = (uint64_t)(p - base);
        if (in_dim != s->key_in_dim ||
            (off % (s->key_in_dim * sizeof(float))) != 0) return 1;
        const uint64_t start_row = off / (s->key_in_dim * sizeof(float));
        if (start_row < s->f32_keep_from) return 1;
    }
    return 0;
}

/* Do rows [0, n_tok) of this buffer have their f32 stores present?  Returns
 * the first PRESENT row (0 = all present).  For gather-style consumers that
 * read arbitrary rows, ANY nonzero return is fatal -- they must be served by
 * an encoding instead.  Exported for the MoE tiers (L089 FFN half). */
uint32_t pulsar_gpu_act_f32_first_present_row(const void *ptr, uint64_t n_tok,
                                              uint64_t in_dim) {
    mxfp8_act_cache_t *s = act_slot_find_rows(ptr, n_tok, in_dim);
    return (s && s->f32_absent) ? s->f32_keep_from : 0u;
}

void pulsar_gpu_mxfp8_act_cache_note_f32_skipped(uint32_t keep_from) {
    if (g_act_cur) {
        g_act_cur->f32_absent    = 1;
        g_act_cur->f32_keep_from = keep_from;
    }
}

/* Grow-only device buffer for the cache. cudaFree implicitly synchronizes, so
 * a growth cannot pull the old pointer out from under an in-flight kernel. */
static int mxfp8_act_cache_reserve(void **buf, size_t *cap, size_t need, const char *what) {
    if (*cap >= need) return 1;
    void *p = NULL;
    if (*buf) { pulsar_gpu_seg_note_device_free();  ///< stale baked pointers
                (void)cudaFree(*buf); *buf = NULL; *cap = 0; }
    if (cudaMalloc(&p, need) != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "pulsar: MXFP8 activation cache alloc failed for %s (%.2f MiB)\n",
                what ? what : "act", (double)need / 1048576.0);
        return 0;
    }
    *buf = p; *cap = need;
    return 1;
}


/* Reserve the cache's E4M3 slots for (n_tok, in_dim) and hand back BOTH device
 * pointers, so a PRODUCER kernel can emit the MX encoding from its own epilogue
 * instead of a separate pass reading the f32 back.  `sf_pitch` returns the KBp
 * the swizzle was sized with -- the producer must use the same one.
 *
 * The quantize pass this replaces is not moved, it is eliminated: the producer
 * already holds the value in a register, and its warp already spans exactly one
 * 32-element MX block, so the block max is a shuffle it can do for free. */
int pulsar_gpu_mxfp8_act_cache_e4m3_slot(const pulsar_gpu_tensor *x,
                                         uint64_t n_tok, uint64_t in_dim,
                                         void **data_out, void **scale_out,
                                         int *sf_pitch) {
    if (!x || n_tok == 0 || in_dim == 0 || (in_dim % 32) != 0 ||
        !data_out || !scale_out || !sf_pitch) {
        return 0;
    }
    mxfp8_act_cache_t *s = act_slot_acquire(x->ptr, n_tok, in_dim);
    if (!s) return 0;
    const int ntok = (int)n_tok;
    const int KBp  = mx_rup((int)(in_dim / 32), 4);
    const size_t sx_bytes = (size_t)mx_rup(ntok, 128) * (size_t)KBp;
    if (!mxfp8_act_cache_reserve((void **)&s->xq, &s->xq_cap,
                                 (size_t)(n_tok * in_dim), "act data") ||
        !mxfp8_act_cache_reserve((void **)&s->sx, &s->sx_cap,
                                 sx_bytes, "act scale")) {
        return 0;
    }
    /* The quantizer memsets the scale slab because pulsar_mx_sfoff leaves holes when
     * rows/blocks are not multiples of 128/4; a producer filling only the live
     * (row, kb) pairs must do the same or the GEMM reads stale swizzle slots. */
    if (cudaMemsetAsync(s->sx, 0, sx_bytes, 0) != cudaSuccess) return 0;
    *data_out  = s->xq;
    *scale_out = s->sx;
    *sf_pitch  = KBp;
    return 1;
}

/* BF16 half of the activation slot: producer-side reservation and validity.
 *
 * matmul_bf16_tensor stages a bf16 copy of its f32 activation on every call
 * (exact-key cached, convert-on-miss).  These two let the PRODUCER of that
 * activation write the bf16 copy from registers instead -- the same move the
 * f16 and E4M3 slots already made -- so the standalone convert kernel never
 * runs for buffers whose producer emits.  The note takes the explicit key
 * rather than riding g_act_cur: arm/disarm pairing is delicate (L035) and
 * this file should not grow a second dependent on it. */
int pulsar_gpu_bf16_act_slot(const pulsar_gpu_tensor *x,
                             uint64_t n_tok, uint64_t in_dim, void **xb_out) {
    if (!x || n_tok == 0 || in_dim == 0 || !xb_out) return 0;
    mxfp8_act_cache_t *s = act_slot_acquire(x->ptr, n_tok, in_dim);
    if (!s) return 0;
    if (!mxfp8_act_cache_reserve((void **)&s->xb, &s->xb_cap,
                                 (size_t)(n_tok * in_dim) * sizeof(__nv_bfloat16),
                                 "act bf16")) {
        return 0;
    }
    *xb_out = s->xb;
    return 1;
}

void pulsar_gpu_bf16_act_note(const pulsar_gpu_tensor *x,
                              uint64_t n_tok, uint64_t in_dim) {
    mxfp8_act_cache_t *s = x ? act_slot_find(x->ptr, n_tok, in_dim) : NULL;
    if (s) s->valid_b = 1;
}

/* GROUPED activation cache -- the attn-output "a" projection only.
 *
 * Separate from the slot array above because the grouped encoding has a
 * different shape (per-group data AND a per-group scale slab) and exactly ONE
 * producer/consumer pair, so a single entry is enough.  The reason it needs a
 * cache at all is the same as the non-grouped case: cuda_attention_output_a_mx_gemm
 * carves xq/sx out of cuda_tmp_alloc, which is one shared scratch region that
 * later callers freely overwrite -- a producer cannot fill a buffer that does
 * not outlive the kernels between it and the GEMM. */
/** The GROUPED counterpart of ::mxfp8_act_cache_t, for per-head activations.
 *
 * Single slot rather than an array: the grouped path has one live activation
 * tensor at a time, so there is nothing to evict between. The key gains the
 * group count and group width, since the same buffer reshaped differently is a
 * different encoding.
 */
struct mxfp8_gact_cache_t {
    const void    *key_ptr;      ///< armed activation buffer (NULL = disarmed)
    uint32_t       key_ntok;     ///< token count; part of the key
    uint32_t       key_ngroups;  ///< group (head) count; part of the key
    uint64_t       key_gdim;     ///< per-group width; part of the key
    int            valid;        ///< xq/sx hold the encoding of that key
    __nv_fp8_e4m3 *xq;           ///< cached E4M3 activations
    size_t         xq_cap;       ///< bytes allocated for xq
    unsigned char *sx;           ///< cached block scales for xq
    size_t         sx_cap;       ///< bytes allocated for sx
    size_t         scale_slab;   ///< stride between one group's scales and the next
    int            kbp;          ///< scale-table K-blocks per group, as the kernel indexes them
};
static thread_local mxfp8_gact_cache_t g_gact;

static mxfp8_gact_cache_t *gact_find(const void *ptr, uint32_t ntok,
                                     uint32_t ngroups, uint64_t gdim) {
    if (!ptr || g_gact.key_ptr != ptr || g_gact.key_ntok != ntok ||
        g_gact.key_ngroups != ngroups || g_gact.key_gdim != gdim) {
        return NULL;
    }
    return &g_gact;
}

int pulsar_gpu_mxfp8_gact_slot(const pulsar_gpu_tensor *heads, uint32_t n_tokens,
                               uint32_t n_groups, uint64_t group_dim,
                               void **data_out, void **scale_out,
                               int *sf_pitch, uint64_t *scale_slab) {
    if (!heads || !heads->ptr || n_tokens == 0 || n_groups == 0 ||
        group_dim == 0 || (group_dim % 32) != 0 ||
        !data_out || !scale_out || !sf_pitch || !scale_slab) {
        return 0;
    }
    const int KBp = mx_rup((int)(group_dim / 32), 4);
    const size_t slab = (size_t)mx_rup((int)n_tokens, 128) * (size_t)KBp;
    const size_t data_bytes  = (size_t)n_tokens * n_groups * group_dim;
    const size_t scale_bytes = (size_t)n_groups * slab;
    if (!mxfp8_act_cache_reserve((void **)&g_gact.xq, &g_gact.xq_cap, data_bytes, "gact data") ||
        !mxfp8_act_cache_reserve((void **)&g_gact.sx, &g_gact.sx_cap, scale_bytes, "gact scale")) {
        return 0;
    }
    /* Same reason as the non-grouped slot: pulsar_mx_sfoff leaves holes, and the
     * producers here fill only the (row, kb) pairs they own -- and they own
     * them in TWO passes (attn_f16 the nope blocks, rope_tail the rope tail),
     * so a stale byte between them would survive into the GEMM. */
    if (cudaMemsetAsync(g_gact.sx, 0, scale_bytes, 0) != cudaSuccess) return 0;
    g_gact.key_ptr     = heads->ptr;
    g_gact.key_ntok    = n_tokens;
    g_gact.key_ngroups = n_groups;
    g_gact.key_gdim    = group_dim;
    g_gact.scale_slab  = slab;
    g_gact.kbp         = KBp;
    g_gact.valid       = 0;
    *data_out   = g_gact.xq;
    *scale_out  = g_gact.sx;
    *sf_pitch   = KBp;
    *scale_slab = (uint64_t)slab;
    return 1;
}

/* Declare the grouped encoding current.  BOTH producers must have run: the
 * attention epilogue owns the nope blocks and rope_tail owns the rope tail it
 * rewrites in place afterwards, so noting after only one leaves the GEMM
 * reading zeroed scale bytes for the other half. */
void pulsar_gpu_mxfp8_gact_note(void) {
    if (g_gact.key_ptr) g_gact.valid = 1;
}

void pulsar_gpu_mxfp8_gact_disarm(void) {
    g_gact.key_ptr = NULL;
    g_gact.valid   = 0;
}

/* Declare the E4M3 encoding current (producer filled the slots above). */
void pulsar_gpu_mxfp8_act_cache_note_mxfp8(void) {
    if (g_act_cur && g_act_cur->key_ptr) g_act_cur->valid = 1;
}

/* Read side of the cache for a consumer that is NOT this file's GEMM: hand back
 * the E4M3 payload and scale a producer already emitted, so the consumer can
 * skip its own quantize pass.  Returns 0 when there is no valid encoding for
 * this exact (ptr, n_tok, in_dim), which the caller must treat as "encode it
 * yourself" -- never as an error. */
int pulsar_gpu_mxfp8_act_cache_get_e4m3_ptr(const void *ptr,
                                            uint64_t n_tok,
                                            uint64_t in_dim,
                                            const void **data,
                                            const void **scale,
                                            int *kbp) {
    if (!ptr || !data || !scale || !kbp) return 0;
    /* Consumer lookup must be prefix-tolerant (act_slot_find_rows), not the
     * exact-key act_slot_find: this reader (the MoE A8 path) can ask for fewer
     * rows than the norm armed in a fused mixed step, and the encoding is
     * row-local so rows [0,n_tok) of a wider block are byte-identical. Exact
     * matching there would miss and silently drop to a re-encode -- the same
     * M-dependence GATE 4 caught for the in-file GEMM consumers. Same slot at
     * equal widths, so bit-exact on today's paths. */
    mxfp8_act_cache_t *s = act_slot_find_rows(ptr, n_tok, in_dim);
    if (!s || !s->valid || !s->xq || !s->sx) return 0;
    *data  = s->xq;
    *scale = s->sx;
    *kbp   = mx_rup((int)(in_dim / 32), 4);
    return 1;
}

int pulsar_gpu_mxfp8_act_cache_get_e4m3(const pulsar_gpu_tensor *x,
                                        uint64_t n_tok,
                                        uint64_t in_dim,
                                        const void **data,
                                        const void **scale,
                                        int *kbp) {
    return x ? pulsar_gpu_mxfp8_act_cache_get_e4m3_ptr(x->ptr, n_tok, in_dim, data, scale, kbp) : 0;
}



static int cuda_matmul_fp8_mx_tensor_labeled(pulsar_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const pulsar_gpu_tensor *x,
        uint64_t n_tok, const char *label) {
    if (!out || !x || !model_map || in_dim % 32 != 0 || !cublaslt_ensure()) return 0;
    /* Derived from the destination, never passed in: the two cannot
     * disagree if only one of them exists. */
    const int out_f16 = (pulsar_tensor_esz(out) == sizeof(__half));

    uint64_t KB = in_dim / 32, weight_bytes = out_dim * KB * 33;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const size_t out_esz = out_f16 ? sizeof(__half) : sizeof(float);
    if (x->bytes < n_tok * in_dim * sizeof(float) || out->bytes < n_tok * out_dim * out_esz) return 0;
    const fp8_mx_weight *w = cuda_fp8_mx_weight(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
    if (!w) return 0;
    int ntok = (int)n_tok, KBp = mx_rup((int)KB, 4);
    size_t sx_bytes = (size_t)mx_rup(ntok, 128) * KBp;
    size_t wz = 32u << 20;
    /* xq, sx and the cuBLASLt workspace must be DISTINCT buffers, but
     * cuda_tmp_alloc hands out one shared scratch region (later calls alias or
     * realloc/free earlier ones), so these three come from one arena. */
    /* Armed activation cache (see mxfp8_act_cache_t above): reuse the E4M3 data
     * and E8M0 scales this activation was already quantized into, and take only
     * the cuBLASLt workspace from the shared tmp region. */
    mxfp8_act_cache_t *ac = act_slot_find(x->ptr, n_tok, in_dim);
    if (ac) {
        if (!mxfp8_act_cache_reserve((void **)&ac->xq, &ac->xq_cap,
                                     in_dim * (size_t)ntok, "act data") ||
            !mxfp8_act_cache_reserve((void **)&ac->sx, &ac->sx_cap,
                                     sx_bytes, "act scale")) {
            ac->valid = 0;  ///< fall back to the per-GEMM quantization
            ac = NULL;
        }
    }
    __nv_fp8_e4m3 *xq;
    unsigned char *sx;
    void *ws;
    cuda_arena ar;
    if (ac) {
        /* Cache armed: the E4M3 data and its E8M0 scales already exist, so the
         * arena reserves the cuBLASLt workspace and nothing else. */
        if (!cuda_arena_begin(&ar, wz, "fp8_mx scratch")) return 0;
        xq = ac->xq;
        sx = ac->sx;
        ws = cuda_arena_take(&ar, wz, 256);
        if (!ws) return 0;
    } else {
        const size_t xq_bytes = in_dim * (size_t)ntok;
        if (!cuda_arena_begin(&ar, mx_a256(xq_bytes) + mx_a256(sx_bytes) + wz,
                              "fp8_mx scratch")) return 0;
        xq = (__nv_fp8_e4m3 *)cuda_arena_take(&ar, xq_bytes, 256);
        sx = (unsigned char *)cuda_arena_take(&ar, sx_bytes, 256);
        ws = cuda_arena_take(&ar, wz, 256);
        if (!ws) return 0;  ///< take() latches: one check covers all three
    }
    if (!ac || !ac->valid) {
        /* About to quantize FROM f32.  If that store was skipped, these bytes
         * are whatever the last call left behind -- stop instead of computing a
         * well-formed wrong answer.  (Reached when the outer dispatch was
         * bypassed, or when the slot went invalid between arm and use.) */
        if (act_f32_absent_hazard(x->ptr, n_tok, in_dim)) {
            fprintf(stderr, "pulsar: fp8_mx '%s' quantizing from an f32 activation whose store "
                            "was SKIPPED (in_dim=%llu n_tok=%llu) -- refusing.\n",
                    label ? label : "?", (unsigned long long)in_dim, (unsigned long long)n_tok);
            return 0;
        }
        cudaMemsetAsync(sx, 0, sx_bytes, 0);
        int warps = ntok * (int)KB;
        mxfp8_quant_act_kernel<<<(warps * 32 + 255) / 256, 256>>>((const float *)x->ptr, ntok, (int)in_dim, KBp, xq, sx);
        if (!cuda_ok(cudaGetLastError(), "fp8_mx act quant")) return 0;
        if (ac) ac->valid = 1;
    }
    /* Shape-keyed handle cache: the desc/layout/preference build plus the
     * heuristic query cost ~100us of host time PER GEMM and only the two
     * scale pointers change between calls of the same (in,out,ntok) shape.
     * The workhorse prefill runs a handful of shapes x 43 layers per chunk,
     * so cache the handles and the chosen algo, and update just the scale
     * pointers per call. Round-robin eviction; entries live for the process
     * (bounded by the slot count). */
    struct lt_shape_cache {
        uint64_t in_dim, out_dim; int ntok; int valid;
        cublasLtMatmulDesc_t op;
        cublasLtMatrixLayout_t la, lb, ld;
        cublasLtMatmulHeuristicResult_t h;
        int out_f16;  ///< D layout dtype: 0 = CUDA_R_32F, 1 = CUDA_R_16F (L045)
};
    /** thread_local for the same reason as the dspark reduce buffers: round-robin
     * eviction below DESTROYS the cuBLASLt descriptors in the slot it takes, so
     * as a process global a second submitting thread could destroy handles this
     * one is about to hand to cublasLtMatmul -- and `cache_next` is a plain
     * non-atomic counter two threads would both advance onto the same slot.
     * Per-thread costs one heuristic search per thread per shape; the entries
     * are small metadata objects, not device memory. */
    static thread_local lt_shape_cache cache[16];
    static thread_local int cache_next;
    lt_shape_cache *e = NULL;
    for (int i = 0; i < 16; i++) {
        if (cache[i].valid && cache[i].in_dim == in_dim &&
            cache[i].out_dim == out_dim && cache[i].ntok == ntok &&
            cache[i].out_f16 == out_f16) { e = &cache[i]; break; }
    }
    if (!e) {
        lt_shape_cache ne = {};
        ne.in_dim = in_dim; ne.out_dim = out_dim; ne.ntok = ntok; ne.out_f16 = out_f16;
        if (cublasLtMatmulDescCreate(&ne.op, CUBLAS_COMPUTE_32F, CUDA_R_32F)) return 0;
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        cublasLtMatmulMatrixScale_t mo = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mo, sizeof(mo));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mo, sizeof(mo));
        /* The heuristic must see a fully-populated desc (including scale
         * pointers) or it selects a non-MX algo an order of magnitude
         * slower; the pointers are re-set per call below. */
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w->scale, sizeof(w->scale));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sx, sizeof(sx));
        cublasLtMatrixLayoutCreate(&ne.la, CUDA_R_8F_E4M3, in_dim, out_dim, in_dim);
        cublasLtMatrixLayoutCreate(&ne.lb, CUDA_R_8F_E4M3, in_dim, ntok, in_dim);
        /* The heuristic is selected AGAINST this layout, so an f16 D gets its own
         * algo -- that is where the measured -39% on q_b comes from, not from
         * writing fewer bytes alone. */
        cublasLtMatrixLayoutCreate(&ne.ld, out_f16 ? CUDA_R_16F : CUDA_R_32F, out_dim, ntok, out_dim);
        cublasLtMatmulPreference_t pf; cublasLtMatmulPreferenceCreate(&pf);
        cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wz, sizeof(wz));
        /* determinism: forbid split-K reduction algos (atomic/parallel
         * reduction order varies run-to-run and vs the decode GEMV path);
         * NONE-scheme algos accumulate in a fixed order. */
        {
            uint32_t red = CUBLASLT_REDUCTION_SCHEME_NONE;
            cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &red, sizeof(red));
        }
        int got = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(g_cublaslt, ne.op, ne.la, ne.lb, ne.ld, ne.ld, pf, 1, &ne.h, &got);
        cublasLtMatmulPreferenceDestroy(pf);
        if (hs != CUBLAS_STATUS_SUCCESS || !got) {
            cublasLtMatrixLayoutDestroy(ne.la); cublasLtMatrixLayoutDestroy(ne.lb);
            cublasLtMatrixLayoutDestroy(ne.ld); cublasLtMatmulDescDestroy(ne.op);
            return 0;
        }
        ne.valid = 1;
        e = &cache[cache_next];
        cache_next = (cache_next + 1) & 15;
        if (e->valid) {
            cublasLtMatrixLayoutDestroy(e->la); cublasLtMatrixLayoutDestroy(e->lb);
            cublasLtMatrixLayoutDestroy(e->ld); cublasLtMatmulDescDestroy(e->op);
        }
        *e = ne;
    }
    cublasLtMatmulDescSetAttribute(e->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w->scale, sizeof(w->scale));
    cublasLtMatmulDescSetAttribute(e->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sx, sizeof(sx));
    int ok = 0;
    if (ws) {
        float al = 1.f, be = 0.f;
        cublasStatus_t st = cublasLtMatmul(g_cublaslt, e->op, &al, w->data, e->la, xq, e->lb, &be,
                                           out->ptr, e->ld, out->ptr, e->ld, &e->h.algo, ws, wz,
                                           cudaStreamPerThread);
        ok = (st == CUBLAS_STATUS_SUCCESS);
        if (!ok) fprintf(stderr, "pulsar: cuBLASLt MXFP8 matmul failed: status %d\n", (int)st);
    }
    return ok;
}





/* Prefill attn-output "a" projection as n_groups block-scaled MXFP8xMXFP8 GEMMs.
 *
 * The projection is block-diagonal: group g's [rank, group_dim] weight slice only
 * sees activation rows (tok, g). cuBLASLt's VEC32_UE8M0 scale layout tiles rows in
 * 128-row blocks, so a batched/strided formulation can't share one swizzled scale
 * buffer across groups; instead the weight cache is sliced per group (exact when
 * rank % 128 == 0: both data columns and scale tiles are contiguous per group)
 * and the activations are quantized into per-group data + scale slabs. Each of
 * the n_groups (8/16) GEMMs writes straight into low[tok][g*rank + r] via ldd,
 * so no epilogue pass is needed. */
static int cuda_attention_output_a_mx_gemm(
        pulsar_gpu_tensor *low,
        const void *model_map,
        uint64_t model_size,
        uint64_t out_a_offset,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        const pulsar_gpu_tensor *heads,
        uint32_t n_tokens) {
    if (group_dim % 32 != 0 || rank % 128 != 0 || !cublaslt_ensure()) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t KB = group_dim / 32;
    const uint64_t weight_bytes = low_dim * KB * 33;
    if (out_a_offset > model_size || weight_bytes > model_size - out_a_offset) return 0;
    const fp8_mx_weight *w = cuda_fp8_mx_weight(model_map, out_a_offset, weight_bytes,
                                                group_dim, low_dim, "attn_out_a");
    if (!w) return 0;
    const int KBp = mx_rup((int)KB, 4);
    const size_t x_scale_slab = (size_t)mx_rup((int)n_tokens, 128) * KBp;
    const size_t w_scale_slab = ((size_t)rank / 128) * (size_t)KBp * 128;
    const size_t data_bytes = (size_t)n_tokens * n_groups * group_dim;
    const size_t scale_bytes = (size_t)n_groups * x_scale_slab;
    size_t wz = 32u << 20;
    __nv_fp8_e4m3 *xq;
    unsigned char *sx;
    void *ws;
    cuda_arena ar;
    /* Producer-emitted grouped encoding (attn_f16 epilogue + rope_tail): the
     * quantize pass is not moved, it is skipped entirely.  The slab geometry
     * the producers wrote must match what this GEMM is about to describe to
     * cuBLASLt, so check it rather than trust the key. */
    mxfp8_gact_cache_t *gc = gact_find(heads->ptr, n_tokens, n_groups, group_dim);
    if (gc && gc->valid && gc->kbp == KBp && gc->scale_slab == x_scale_slab) {
        /* Announce once per process, like the attention tier line.  Whether
         * this path is TAKEN is the whole question for the fusion's value: it
         * needs raw_prefix_tokens == n_tokens, and long prefills with
         * compressed KV go through the per-token indexed loop instead.  A
         * silent fallback is indistinguishable from a working fusion in every
         * measurement except a kernel-count profile, so say so. */
        static int announced_gact = 0;
        if (!announced_gact) {
            announced_gact = 1;
            fprintf(stderr, "pulsar: attn-out 'a' activation = producer-emitted E4M3 "
                            "(grouped quantise pass skipped)\n");
        }
        xq = gc->xq;
        sx = gc->sx;
        if (!cuda_arena_begin(&ar, wz, "attn_out_a mx scratch")) return 0;
        ws = cuda_arena_take(&ar, wz, 256);
        if (!ws) return 0;
    } else {
        if (!cuda_arena_begin(&ar, mx_a256(data_bytes) + mx_a256(scale_bytes) + wz,
                              "attn_out_a mx scratch")) return 0;
        xq = (__nv_fp8_e4m3 *)cuda_arena_take(&ar, data_bytes, 256);
        sx = (unsigned char *)cuda_arena_take(&ar, scale_bytes, 256);
        ws = cuda_arena_take(&ar, wz, 256);
        if (!ws) return 0;  ///< take() latches: one check covers all three
        cudaMemsetAsync(sx, 0, scale_bytes, 0);
        int warps = (int)n_tokens * (int)n_groups * (int)KB;
        mxfp8_quant_act_grouped_kernel<<<(warps * 32 + 255) / 256, 256>>>(
                (const pulsar_heads_t *)heads->ptr, (int)n_tokens, (int)n_groups,
                (int)group_dim, KBp, xq, sx, x_scale_slab);
        if (!cuda_ok(cudaGetLastError(), "attn_out_a act quant")) return 0;
    }
    /* Same shape-keyed handle/algo cache as the main MXFP8 GEMM above (and
     * the same gotcha: the heuristic must see scale pointers on the desc or
     * it picks a non-MX algo). The per-group loop swaps only scale pointers,
     * which is already cache-shaped. */
    struct lt_group_cache {
        uint64_t group_dim, rank; uint32_t n_groups; int ntok; int valid;
        cublasLtMatmulDesc_t op;
        cublasLtMatrixLayout_t la, lb, ld;
        cublasLtMatmulHeuristicResult_t h;
    };
    /** thread_local -- same destroy-on-evict hazard as the shape cache above. */
    static thread_local lt_group_cache cache[8];
    static thread_local int cache_next;
    lt_group_cache *e = NULL;
    for (int i = 0; i < 8; i++) {
        if (cache[i].valid && cache[i].group_dim == group_dim && cache[i].rank == rank &&
            cache[i].n_groups == n_groups && cache[i].ntok == (int)n_tokens) { e = &cache[i]; break; }
    }
    if (!e) {
        lt_group_cache ne = {};
        ne.group_dim = group_dim; ne.rank = rank; ne.n_groups = n_groups; ne.ntok = (int)n_tokens;
        if (cublasLtMatmulDescCreate(&ne.op, CUBLAS_COMPUTE_32F, CUDA_R_32F)) return 0;
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        cublasLtMatmulMatrixScale_t mo = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mo, sizeof(mo));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mo, sizeof(mo));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w->scale, sizeof(w->scale));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sx, sizeof(sx));
        cublasLtMatrixLayoutCreate(&ne.la, CUDA_R_8F_E4M3, group_dim, rank, group_dim);
        cublasLtMatrixLayoutCreate(&ne.lb, CUDA_R_8F_E4M3, group_dim, n_tokens, group_dim);
        cublasLtMatrixLayoutCreate(&ne.ld, CUDA_R_32F, rank, n_tokens, low_dim);
        cublasLtMatmulPreference_t pf; cublasLtMatmulPreferenceCreate(&pf);
        cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wz, sizeof(wz));
        /* determinism: forbid split-K reduction algos (atomic/parallel
         * reduction order varies run-to-run and vs the decode GEMV path);
         * NONE-scheme algos accumulate in a fixed order. */
        {
            uint32_t red = CUBLASLT_REDUCTION_SCHEME_NONE;
            cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &red, sizeof(red));
        }
        int got = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(g_cublaslt, ne.op, ne.la, ne.lb, ne.ld, ne.ld, pf, 1, &ne.h, &got);
        cublasLtMatmulPreferenceDestroy(pf);
        if (hs != CUBLAS_STATUS_SUCCESS || !got) {
            cublasLtMatrixLayoutDestroy(ne.la); cublasLtMatrixLayoutDestroy(ne.lb);
            cublasLtMatrixLayoutDestroy(ne.ld); cublasLtMatmulDescDestroy(ne.op);
            return 0;
        }
        ne.valid = 1;
        e = &cache[cache_next];
        cache_next = (cache_next + 1) & 7;
        if (e->valid) {
            cublasLtMatrixLayoutDestroy(e->la); cublasLtMatrixLayoutDestroy(e->lb);
            cublasLtMatrixLayoutDestroy(e->ld); cublasLtMatmulDescDestroy(e->op);
        }
        *e = ne;
    }
    cublasLtMatmulDesc_t op = e->op;
    cublasLtMatrixLayout_t la = e->la, lb = e->lb, ld = e->ld;
    cublasLtMatmulHeuristicResult_t h = e->h;
    int ok = 0;
    if (ws) {
        ok = 1;
        for (uint32_t g = 0; g < n_groups && ok; g++) {
            const __nv_fp8_e4m3 *ag = w->data + (size_t)g * rank * group_dim;
            const unsigned char *as = w->scale + (size_t)g * w_scale_slab;
            const __nv_fp8_e4m3 *bg = xq + (size_t)g * n_tokens * group_dim;
            const unsigned char *bs = sx + (size_t)g * x_scale_slab;
            float *dg = (float *)low->ptr + (size_t)g * rank;
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &as, sizeof(as));
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bs, sizeof(bs));
            float al = 1.f, be = 0.f;
            cublasStatus_t st = cublasLtMatmul(g_cublaslt, op, &al, ag, la, bg, lb, &be,
                                               dg, ld, dg, ld, &h.algo, ws, wz,
                                               cudaStreamPerThread);
            ok = (st == CUBLAS_STATUS_SUCCESS);
            if (!ok) fprintf(stderr, "pulsar: cuBLASLt attn_out_a MXFP8 matmul failed: status %d\n", (int)st);
        }
    }
    return ok;
}



/* Native FP8 decode (mmvq): read the MXFP8 weight 8-bit and dot with the f32
 * activation (single token). One warp per output row; lane L covers in-dim
 * positions {L, 32+L, ...}. No f16 expansion -> no decode bandwidth regression
 * vs the removed Q8_0 path. Used when MXFP8 tensor-cores don't apply (n_tok==1). */


/* Decode mmvq over the DE-INTERLEAVED cached weight (contiguous E4M3 data[out,in]
 * + swizzled E8M0 scale), the same buffers the prefill tensor-core path builds via
 * cuda_fp8_mx_weight(). Reading contiguous fp8 lets each lane pull a uint32 (4 E4M3)
 * per step -> 128-wide coalesced loads vs the raw 33B-interleaved kernel's misaligned
 * 1-byte/thread reads. Numerically identical (same fp8 bytes, same raw E8M0 scale byte).
 * Requires in_dim % 128 == 0 (all MLA/shared/head dims qualify); else fall back to raw. */
template <typename OT>
__global__ static void mxfp8_mmvq_deint_kernel(OT *out, const __nv_fp8_e4m3 *data,
                                               const unsigned char *scale, const float *x,
                                               int in_dim, int out_dim, int KBp) {
    int o = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (o >= out_dim) return;
    const __nv_fp8_e4m3 *row = data + (size_t)o * in_dim;
    float acc = 0.f;
    for (int base = 0; base < in_dim; base += 128) {
        int k = base + lane * 4;  ///< this lane's 4 in-positions
        uint32_t packed = *(const uint32_t *)(row + k);
        int kb = k >> 5;  ///< 32-elem block for these 4
        float sc = __int_as_float((uint32_t)scale[pulsar_mx_sfoff(o, kb, KBp)] << 23);  ///< 2^(e-127), no SFU
        const __nv_fp8_e4m3 *q = (const __nv_fp8_e4m3 *)&packed;
        const float *xk = x + k;
        #pragma unroll
        for (int j = 0; j < 4; j++) acc += __half2float((__half)q[j]) * sc * xk[j];
    }
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0) q_store<OT>(out, o, acc);
}


/* W8A8 twin of the de-interleaved mmvq: the activation arrives as E4M3 + a
 * swizzled E8M0 block scale instead of f32, so both operands are in the format
 * the SOURCE model computes with (dynamic e4m3, ue8m0 scale).  Weights are
 * unchanged.
 *
 * ⚠ THIS IS A FIDELITY CHANGE, NOT A SPEED ONE, AND THE DISTINCTION WAS GOT
 * WRONG ONCE ALREADY.  L044 argued a W8A8 GEMV "reads 1 B/elem instead of 4, so
 * it should be FASTER".  That counts the wrong bytes: in a GEMV the weight
 * matrix is out_dim x in_dim and is streamed once with no reuse, while the
 * activation is a single in_dim vector shared by every output warp and served
 * from cache.  At in_dim=4096, out_dim=1024 that is 4 MB of weights against
 * 16 KB of activations -- 0.4%.  Narrowing the activation saves in_dim*3 bytes
 * IN TOTAL, not per row, and costs a dequant per element.  Expect neutral to
 * slightly slower; the reason to do it is that f32 activations here are
 * OVER-PRECISION against the reference, not that they are expensive.
 *
 * Lane k-range: k = base + lane*4 with base stepping 128, so a lane's 4
 * elements always sit inside ONE 32-element MX block and both scales are
 * loaded once per lane per step.  The activation is a single row, so its
 * scale row is xrow -- the cache holds the whole (n_tok, in_dim) block, so a
 * per-token launch must say which row it is reading. */
template <typename OT>
__global__ static void mxfp8_mmvq_deint_a8_kernel(OT *out, const __nv_fp8_e4m3 *data,
                                                  const unsigned char *scale,
                                                  const __nv_fp8_e4m3 *xq,
                                                  const unsigned char *xs,
                                                  int in_dim, int out_dim, int KBp, int xKBp,
                                                  int xrow) {
    int o = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (o >= out_dim) return;
    const __nv_fp8_e4m3 *row = data + (size_t)o * in_dim;
    float acc = 0.f;
    for (int base = 0; base < in_dim; base += 128) {
        int k = base + lane * 4;
        uint32_t wpk = *(const uint32_t *)(row + k);
        uint32_t apk = *(const uint32_t *)(xq + k);
        int kb = k >> 5;
        float sw = __int_as_float((uint32_t)scale[pulsar_mx_sfoff(o, kb, KBp)] << 23);
        float sa = __int_as_float((uint32_t)xs[pulsar_mx_sfoff(xrow, kb, xKBp)] << 23);
        const float s = sw * sa;
        const __nv_fp8_e4m3 *qw = (const __nv_fp8_e4m3 *)&wpk;
        const __nv_fp8_e4m3 *qa = (const __nv_fp8_e4m3 *)&apk;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            acc += __half2float((__half)qw[j]) * __half2float((__half)qa[j]) * s;
        }
    }
    for (int s2 = 16; s2 > 0; s2 >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s2);
    if (lane == 0) q_store<OT>(out, o, acc);
}



/* Small-batch (2..4 token) variant of the de-interleaved mmvq for the spec-decode
 * verify forward. One weight-row read serves all NT tokens (per-token accumulators),
 * vs the tensor-core tile path (latency-bound at 2-4 rows) or NT GEMV relaunches
 * (NT x weight traffic). Per-token multiply/accumulate order matches
 * mxfp8_mmvq_deint_kernel exactly, so each token's output is bit-identical to the
 * n=1 kernel run on that token alone. */
template <int NT, typename OT>
__global__ static void mxfp8_mmvq_deint_nt_kernel(OT *out, const __nv_fp8_e4m3 *data,
                                                  const unsigned char *scale, const float *x,
                                                  int in_dim, int out_dim, int KBp) {
    int o = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (o >= out_dim) return;
    const __nv_fp8_e4m3 *row = data + (size_t)o * in_dim;
    float acc[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.f;
    for (int base = 0; base < in_dim; base += 128) {
        int k = base + lane * 4;
        uint32_t packed = *(const uint32_t *)(row + k);
        int kb = k >> 5;
        float sc = __int_as_float((uint32_t)scale[pulsar_mx_sfoff(o, kb, KBp)] << 23);
        const __nv_fp8_e4m3 *q = (const __nv_fp8_e4m3 *)&packed;
        const float *xk = x + k;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const float wj = __half2float((__half)q[j]) * sc;
            #pragma unroll
            for (int t = 0; t < NT; t++) acc[t] += wj * xk[(size_t)t * in_dim + j];
        }
    }
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        float a = acc[t];
        for (int s = 16; s > 0; s >>= 1) a += __shfl_xor_sync(0xffffffffu, a, s);
        if (lane == 0) q_store<OT>(out, (size_t)t * out_dim + o, a);
    }
}


/* A8 twin of the NT batched GEMV.  Restores the invariant the comment above
 * claims and A8 quietly broke: "each token's output is bit-identical to the n=1
 * kernel run on that token alone".
 *
 * That held while both arms read f32.  Then the n==1 path moved to E4M3
 * (mxfp8_mmvq_deint_a8_kernel) and this one did not, so from that commit the
 * SPEC-DECODE VERIFY BATCH multiplied in a different activation format than the
 * drafter's own forwards -- the drafter proposing under one numerics and the
 * target verifying under another, which is exactly the cross-arm mismatch that
 * cost 6.6 points of acceptance in the MoE (L056) before ea86645 closed it.
 *
 * Multiply order is copied from the n==1 A8 kernel and must stay that way:
 * qw[j] * qa[j] * s, with s = sw*sa hoisted, NOT (qw*sw)*(qa*sa).  The f32 NT
 * kernel hoists wj = w*sc for the same reason -- matching the arm it must agree
 * with is worth more than a saved multiply.
 *
 * The activation cache is row-major [rows, K] (mxfp8_quant_act_kernel stores at
 * row*K + k), so a lane's 4 elements stay contiguous per token and the scale row
 * is simply the token index. */
template <int NT, typename OT>
__global__ static void mxfp8_mmvq_deint_nt_a8_kernel(OT *out, const __nv_fp8_e4m3 *data,
                                                     const unsigned char *scale,
                                                     const __nv_fp8_e4m3 *xq,
                                                     const unsigned char *xs,
                                                     int in_dim, int out_dim, int KBp, int xKBp) {
    int o = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (o >= out_dim) return;
    const __nv_fp8_e4m3 *row = data + (size_t)o * in_dim;
    float acc[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.f;
    /* L109 N4 FINAL VERDICT (2026-08-25, two trials): a manual
     * software-pipelined weight load (prefetch next word+scale, guarded by an
     * in-loop bounds branch) measured -33% DECODE in a clean single-lever A/B
     * against the true baseline -- the hand pipeline plus its branch defeats
     * the compiler's own scheduling of this loop, which was already good.
     * Bit-exact both ways; the stall ncu reports here (20.7 cyc/warp L1TEX)
     * is cheaper than any register/branch price paid to hide it. Do not
     * re-add without a branch-free formulation A/B'd solo. */
    for (int base = 0; base < in_dim; base += 128) {
        int k = base + lane * 4;
        uint32_t wpk = *(const uint32_t *)(row + k);
        int kb = k >> 5;
        float sw = __int_as_float((uint32_t)scale[pulsar_mx_sfoff(o, kb, KBp)] << 23);
        const __nv_fp8_e4m3 *qw = (const __nv_fp8_e4m3 *)&wpk;
        #pragma unroll
        for (int t = 0; t < NT; t++) {
            uint32_t apk = *(const uint32_t *)(xq + (size_t)t * in_dim + k);
            float sa = __int_as_float((uint32_t)xs[pulsar_mx_sfoff(t, kb, xKBp)] << 23);
            const float s = sw * sa;
            const __nv_fp8_e4m3 *qa = (const __nv_fp8_e4m3 *)&apk;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                acc[t] += __half2float((__half)qw[j]) * __half2float((__half)qa[j]) * s;
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        float a = acc[t];
        for (int s2 = 16; s2 > 0; s2 >>= 1) a += __shfl_xor_sync(0xffffffffu, a, s2);
        if (lane == 0) q_store<OT>(out, (size_t)t * out_dim + o, a);
    }
}


/* PER-TENSOR routing: every offset registered at load (the MXFP8 workhorse
 * weights: attn_kv/q_a/q_b, attn_output_a/b, shared experts, output head)
 * takes the FP8 path; anything unregistered is rejected below. */
std::unordered_set<uint64_t> g_fp8_offsets;


void pulsar_gpu_register_fp8_weight(uint64_t weight_offset) { g_fp8_offsets.insert(weight_offset); }


void pulsar_gpu_register_fp8_lt_weight(uint64_t weight_offset) { g_mxfp8_lt_offsets.insert(weight_offset); }


/* Drop every process-global fp8 weight-cache entry. MUST run at backend
 * cleanup (pulsar_gpu_cleanup): pre-stored MXFP8_LT entries point straight into
 * the per-engine model arena that cleanup frees, and a subsequent engine open
 * in the same process typically mmaps the model at the SAME base address --
 * the cache's (host_base, offset, dims) guard then false-positives and serves
 * dangling pointers into freed memory (garbage/NaN activations, and an
 * illegal TMA access inside the cuBLASLt MXFP8 GEMM once a page is gone).
 *
 * Every entry is now arena-owned. The loop that cudaFree'd converted (non-LT)
 * buffers went with the convert path on 2026-08-17: nothing allocates them, so
 * freeing them was reachable only if a plain type-38 weight had been resolved,
 * which the loader now refuses. */
void cuda_fp8_weight_cache_clear(void) {
    g_fp8_mx_by_offset.clear();
    memset(g_fp8_fc_off, 0, sizeof(g_fp8_fc_off));
    memset(g_fp8_fc_ptr, 0, sizeof(g_fp8_fc_ptr));
    /* per-load registrations; the next engine open re-registers its own set */
    g_fp8_offsets.clear();
    g_mxfp8_lt_offsets.clear();
}



/* plan-34 phase-2 inc 2 — cuBLASLt ALGO-STABILITY. When armed (a batched
 * multiseq/mixed step), force the M-INDEPENDENT custom per-token GEMV kernels for
 * the whole batched row range [2..PULSAR_MSEQ_MAX=8] instead of switching to a
 * cuBLAS(Lt) tensor-core GEMM at n_tok>=5. cuBLAS(Lt) resolves an M-dependent
 * (ntok-keyed heuristic) algo, so a co-scheduled decode bank's logits would shift
 * with the batch width (measured: M=5/8 differ from M=2). The custom deint_nt /
 * f16_nt kernels compute each output row purely from THAT row's activations, so
 * they are byte-identical across M by construction. Armed once per step
 * (multiseq_step_begin) / cleared (step_end) — NEVER per token. Classic prefill
 * never arms it, so its large-M cuBLAS tensor-core path is unchanged, and the
 * decode-only lane (n_tok<=4, already custom) is bit-for-bit unchanged.
 *
 * plan-34 phase-2 inc 4 — GENERALIZED to a PREFIX ROW COUNT. In a fused mixed
 * step the row layout is [decode rows 0..n_dec) then one K-row prefill run
 * [n_dec..M). The decode prefix must stay M-independent (byte-identical to a
 * decode-only step of width n_dec — gate-4 neutrality); the prefill suffix takes
 * the fast tensor-core path (correctness, not byte-identity). g_mneutral_rows now
 * holds n_dec (0 = pure prefill / not armed; == n_tok = decode-only). Each dense
 * GEMM splits by recursing with the flag set to each range's PURE regime, so the
 * exact inc-2 (custom) and inc-3 (tensor-core) code paths are reused verbatim —
 * the decode prefix launch is LITERALLY the same nt<n_dec> a decode-only step
 * emits, and the prefill suffix is the same tensor-core GEMM a pure-prefill step
 * emits at that width. No kernel logic is duplicated. */
static int g_mneutral_rows = 0;
void pulsar_gpu_matmul_set_batch_mneutral(int n) {
    /* The armed nt-caps cover PULSAR_GPU_MNEUTRAL_ROWS_MAX rows, and the engine
     * static_asserts PULSAR_MSEQ_MAX against it -- so this branch is unreachable
     * from the batched lane.  It stays as a fail-loud guard for any OTHER caller:
     * rows past the cap would take a batch-shape-dependent GEMM, the "same op,
     * two numerics, chosen by width" shape this codebase keeps getting bitten by. */
    if (n > (int)PULSAR_GPU_MNEUTRAL_ROWS_MAX) {
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr, "pulsar: WARNING batched step of %d rows exceeds the "
                            "m-neutral cap of %u -- rows past the cap take a different "
                            "GEMM and neutrality is not guaranteed\n",
                            n, (unsigned)PULSAR_GPU_MNEUTRAL_ROWS_MAX);
        }
    }
    g_mneutral_rows = (n > 0) ? n : 0;
}
/* Queried cross-TU by the MoE dispatch (pulsar_cuda_moe.cu): the number of leading
 * decode rows that must take the M-independent per-token expert path (the trailing
 * prefill rows take the grouped GEMM). 0 = not armed. Nonzero = armed (inc-2/3
 * read it as a boolean; inc-4 MoE two-pass reads the count to place the split). */
int pulsar_gpu_matmul_batch_mneutral(void) { return g_mneutral_rows; }


static int cuda_matmul_mxfp8_tensor_labeled(pulsar_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const pulsar_gpu_tensor *x, uint64_t n_tok, const char *label) {
    /* Derived from the destination, never passed in: the two cannot
     * disagree if only one of them exists. */
    const int out_f16 = (pulsar_tensor_esz(out) == sizeof(__half));
    /* A pure bounds check.  It no longer guards a format mismatch -- out_f16 is
     * derived from `out` itself now, so the two cannot disagree -- but n_tok
     * can still exceed what the destination holds, and that is worth catching
     * loudly rather than scribbling past the end. */
    if (out) {
        const uint64_t need = out_dim * n_tok * (out_f16 ? sizeof(__half) : sizeof(float));
        if (out->bytes < need) {
            fprintf(stderr, "pulsar: mxfp8 matmul '%s' would write %llu bytes into a "
                            "%llu-byte output (out_f16=%d) -- refusing\n",
                    label ? label : "?", (unsigned long long)need,
                    (unsigned long long)out->bytes, out_f16);
            return 0;
        }
    }
    if (!out || !x || !model_map) return 0;
    /* Backstop for a producer that emitted E4M3 and skipped its f32 store.
     * Fail LOUD rather than multiply stale bytes: per the standing rule, a
     * wrong answer is worse than a stop. */
    if (act_f32_absent_hazard(x->ptr, n_tok, in_dim)) {
        fprintf(stderr, "pulsar: mxfp8 '%s' would read the f32 activation of a buffer whose "
                        "store was SKIPPED (in_dim=%llu out_dim=%llu n_tok=%llu) -- refusing. "
                        "The producer emitted E4M3 and dropped the f32; something invalidated "
                        "or evicted that encoding before this GEMM.\n",
                label ? label : "?", (unsigned long long)in_dim,
                (unsigned long long)out_dim, (unsigned long long)n_tok);
        return 0;
    }
    /* The mixed-batch split below recurses on OFFSET row pointers, which key no
     * slot, so BOTH halves quantize from f32 -- including the half whose store
     * was skipped.  The check above cannot see that case (the base pointer is
     * covered by a valid slot), so it is made here, where the split is decided. */
    if (g_mneutral_rows > 0 && (uint64_t)g_mneutral_rows < n_tok) {
        for (int i = 0; i < PULSAR_ACT_SLOTS; i++) {
            if (g_act_slots[i].key_ptr == x->ptr && g_act_slots[i].f32_absent) {
                fprintf(stderr, "pulsar: mxfp8 '%s' mixed-batch split (n_dec=%d of %llu) on a "
                                "buffer whose f32 store was SKIPPED -- refusing; the split "
                                "halves cannot reach the E4M3 cache.\n",
                        label ? label : "?", g_mneutral_rows, (unsigned long long)n_tok);
                return 0;
            }
        }
    }
    /* inc 4 prefix-split: 0<n_dec<n_tok => mixed decode+prefill batch. Run the
     * decode prefix [0,n_dec) in the M-independent (decode) regime and the prefill
     * suffix [n_dec,n_tok) in the tensor-core (prefill) regime, by recursing with
     * the flag set to each range's pure value. Offsets are row-major (float rows). */
    {
        const uint64_t n_dec = (uint64_t)g_mneutral_rows;
        if (n_dec > 0 && n_dec < n_tok) {
            /* Row strides are BYTES: the output half must follow the OUTPUT
             * element size, or the suffix lands at the wrong offset. */
            const uint64_t inb = in_dim * sizeof(float);
            const uint64_t outb = out_dim * (out_f16 ? sizeof(__half) : sizeof(float));
            pulsar_gpu_tensor out_pre = pulsar_tensor_subview(out, 0, out->bytes);
            pulsar_gpu_tensor x_pre   = pulsar_tensor_subview(x, 0, x->bytes);
            pulsar_gpu_tensor out_suf = pulsar_tensor_subview(out, n_dec * outb,
                                                             out->bytes - n_dec * outb);
            pulsar_gpu_tensor x_suf   = pulsar_tensor_subview(x, n_dec * inb,
                                                             x->bytes - n_dec * inb);
            const int saved = g_mneutral_rows;
            g_mneutral_rows = (int)n_dec;  ///< decode prefix: n_dec == n_tok' => all custom
            int r1 = cuda_matmul_mxfp8_tensor_labeled(&out_pre, model_map, model_size,
                    weight_offset, in_dim, out_dim, &x_pre, n_dec, label);
            g_mneutral_rows = 0;  ///< prefill suffix: tensor-core
            int r2 = cuda_matmul_mxfp8_tensor_labeled(&out_suf, model_map, model_size,
                    weight_offset, in_dim, out_dim, &x_suf, n_tok - n_dec, label);
            g_mneutral_rows = saved;
            return r1 && r2;
        }
    }
    if (g_fp8_offsets.count(weight_offset)) {
        const uint64_t fblocks = (in_dim + 31) / 32;
        const uint64_t fbytes = out_dim * fblocks * 33;
        /* No out->bytes term here: the guard at the top of this function
         * already bounds the destination, and does it in the ELEMENT SIZE
         * THIS CALL WILL WRITE.  A second copy that hardcoded sizeof(float)
         * lived here and refused every f16 output whose n_tok passed half the
         * buffer's capacity -- silently, since a refusal is "did not encode"
         * rather than an error.  That cost a bisect: it made the Q narrowing
         * look depth-dependent (fine at 512 and 2048, dead at 4096) when the
         * real variable was n_tok against the allocation. */
        if (weight_offset > model_size || fbytes > model_size - weight_offset ||
            x->bytes < n_tok * in_dim * sizeof(float)) return 0;
        /* Small batches (spec-decode verify, n_tok 2..4): batched GEMV over the
         * de-interleaved weight. One weight-row read serves all tokens, vs the
         * tensor-core tile path which is latency-bound at these shapes (the
         * measured "CUTLASS launch storm" that made verify(3) cost ~2x a decode
         * token). Bit-identical per token to the n=1 kernel, so verify logits match
         * the decode path's numerics -- and that holds WITHIN an activation
         * format, not across: the A8 twin above pairs with
         * mxfp8_mmvq_deint_a8_kernel, this f32 one with mxfp8_mmvq_deint_kernel.
         * The claim was silently false from the day A8 converted n==1 and left
         * this arm on f32 until 0a60a1a converted it too.  (A sentence ended
         * here describing an env override that dispatched tensor-core for all
         * n_tok>1; the override is gone and the sentence had been truncated
         * mid-clause for however long, so it goes with it.) */
        /* 2026-07-21: raising this default 4 -> 8 was TRIED and REVERTED. The
         * "bit-identical" claim above is GEMV-n vs GEMV-1 -- it does NOT extend to
         * the cuBLASLt dispatch this cap hands off to, which is what widening the
         * window actually swaps out. Measured at verify width 6: all 129280 logits
         * differ, max |d| 1.674, RMS 0.312 vs sigma 6.02, and 115/128 generated
         * tokens change. It also reaches real prefill -- the final chunk keeps the
         * exact remainder, so a 4102-token prompt at chunk 4096 ends on n_tok=6 and
         * its logits move (4100 -> n_tok=4 is byte-identical, control). The
         * prefill byte-exact gate pins chunk 4096 with depths 512/2048/4096/6144
         * DID once never land a 5..8 remainder -- depth 4102 was added precisely
         * to cover it (4102 = 4096 + 6), and it is the depth that moves whenever
         * this cap or the MoE kernel choice changes. Keep 4.
         * The width-6 verify win is real but belongs entirely to moe_gemv_cap
         * (see pulsar_cuda_moe.cu) -- these three caps cost ~28 ms/verify on top. */
        static const int gemv_max_n = 4;
        /* inc 2: raise the custom-nt cap for a batched step so armed widths keep
         * the M-independent kernel instead of cuBLASLt. Default cap (gemv_max_n=4)
         * is unchanged for classic prefill (never armed) and for the decode-only
         * lane (n_tok<=4), which take the identical cases 2/3/4 below. The armed
         * cap is PULSAR_GPU_MNEUTRAL_ROWS_MAX, static_assert-tied to
         * PULSAR_MSEQ_MAX (it drifted once: caps stayed 8 while MSEQ went 16). */
        const uint64_t nt_cap = (g_mneutral_rows > 0)
                ? (uint64_t)PULSAR_GPU_MNEUTRAL_ROWS_MAX : (uint64_t)gemv_max_n;
        /* nt_cap is 4 or MNEUTRAL_ROWS_MAX, so <= nt_cap already bounds the
         * row count; the explicit MNEUTRAL conjunct here was provably true (K13). */
        if (n_tok >= 2 && n_tok <= nt_cap && in_dim % 128 == 0) {
            const fp8_mx_weight *bw = cuda_fp8_mx_weight(model_map, weight_offset, fbytes,
                                                         in_dim, out_dim, label);
            if (bw) {
                const int KBp = mx_rup((int)(in_dim / 32), 4);
                const unsigned wpb = 8;
                dim3 grid(((unsigned)out_dim + wpb - 1) / wpb);
                /* A8 first, for the same reason the n==1 path takes it: the
                 * source multiplies dynamic E4M3 activations, and this arm
                 * serving the verify batch in f32 while n==1 served the
                 * drafter's forwards in E4M3 is a size-thresholded activation
                 * format -- the defect this codebase keeps re-finding. */
                const mxfp8_act_cache_t *ac8nt = act_slot_find_rows(x->ptr, n_tok, in_dim);
                if (!ac8nt && act_slot_a8_declared_short(x->ptr, n_tok, in_dim))
                    return act_a8_contract_fail("verify-batch GEMV", n_tok, in_dim, out_dim);
                if (ac8nt && ac8nt->valid) {
                    const int xKBp = mx_rup((int)(in_dim / 32), 4);
                    {
                        /* Keyed on the (in,out) pair and announced once per
                         * shape, exactly like the n==1 twin: a gate PASS cannot
                         * tell "the A8 arm ran" from "the A8 arm never ran", and
                         * a silently-missed cache would look identical. */
                        static uint64_t seen_nt[16] = {0};
                        static int n_seen_nt = 0;
                        const uint64_t key = (in_dim << 32) ^ out_dim;
                        int known = 0;
                        for (int i = 0; i < n_seen_nt; i++) if (seen_nt[i] == key) { known = 1; break; }
                        if (!known && n_seen_nt < 16) {
                            seen_nt[n_seen_nt++] = key;
                            fprintf(stderr, "pulsar: verify-batch GEMV W8A8 (E4M3 acts) for "
                                            "in_dim=%llu out_dim=%llu\n",
                                    (unsigned long long)in_dim, (unsigned long long)out_dim);
                        }
                    }
                    #define PULSAR_FP8_NT_A8(N, OT) mxfp8_mmvq_deint_nt_a8_kernel<N, OT><<<grid, wpb * 32>>>( \
                            (OT *)out->ptr, bw->data, bw->scale, ac8nt->xq, ac8nt->sx, \
                            (int)in_dim, (int)out_dim, KBp, xKBp)
                    switch (n_tok) {
                    case 2: if (out_f16) PULSAR_FP8_NT_A8(2, __half); else PULSAR_FP8_NT_A8(2, float); break;
                    case 3: if (out_f16) PULSAR_FP8_NT_A8(3, __half); else PULSAR_FP8_NT_A8(3, float); break;
                    case 4: if (out_f16) PULSAR_FP8_NT_A8(4, __half); else PULSAR_FP8_NT_A8(4, float); break;
                    case 5: if (out_f16) PULSAR_FP8_NT_A8(5, __half); else PULSAR_FP8_NT_A8(5, float); break;
                    case 6: if (out_f16) PULSAR_FP8_NT_A8(6, __half); else PULSAR_FP8_NT_A8(6, float); break;
                    case 7: if (out_f16) PULSAR_FP8_NT_A8(7, __half); else PULSAR_FP8_NT_A8(7, float); break;
                    case 8: if (out_f16) PULSAR_FP8_NT_A8(8, __half); else PULSAR_FP8_NT_A8(8, float); break;
                    case 9: if (out_f16) PULSAR_FP8_NT_A8(9, __half); else PULSAR_FP8_NT_A8(9, float); break;
                    case 10: if (out_f16) PULSAR_FP8_NT_A8(10, __half); else PULSAR_FP8_NT_A8(10, float); break;
                    case 11: if (out_f16) PULSAR_FP8_NT_A8(11, __half); else PULSAR_FP8_NT_A8(11, float); break;
                    case 12: if (out_f16) PULSAR_FP8_NT_A8(12, __half); else PULSAR_FP8_NT_A8(12, float); break;
                    case 13: if (out_f16) PULSAR_FP8_NT_A8(13, __half); else PULSAR_FP8_NT_A8(13, float); break;
                    case 14: if (out_f16) PULSAR_FP8_NT_A8(14, __half); else PULSAR_FP8_NT_A8(14, float); break;
                    case 15: if (out_f16) PULSAR_FP8_NT_A8(15, __half); else PULSAR_FP8_NT_A8(15, float); break;
                    default: if (out_f16) PULSAR_FP8_NT_A8(16, __half); else PULSAR_FP8_NT_A8(16, float); break;  ///< n_tok == 16 == PULSAR_GPU_MNEUTRAL_ROWS_MAX
                    }
                    #undef PULSAR_FP8_NT_A8
                    return cuda_ok(cudaGetLastError(), "fp8_mx mmvq deint nt a8");
                }
                #define PULSAR_FP8_NT(N, OT) mxfp8_mmvq_deint_nt_kernel<N, OT><<<grid, wpb * 32>>>( \
                        (OT *)out->ptr, bw->data, bw->scale, (const float *)x->ptr, \
                        (int)in_dim, (int)out_dim, KBp)
                switch (n_tok) {
                case 2: if (out_f16) PULSAR_FP8_NT(2, __half); else PULSAR_FP8_NT(2, float); break;
                case 3: if (out_f16) PULSAR_FP8_NT(3, __half); else PULSAR_FP8_NT(3, float); break;
                case 4: if (out_f16) PULSAR_FP8_NT(4, __half); else PULSAR_FP8_NT(4, float); break;
                case 5: if (out_f16) PULSAR_FP8_NT(5, __half); else PULSAR_FP8_NT(5, float); break;
                case 6: if (out_f16) PULSAR_FP8_NT(6, __half); else PULSAR_FP8_NT(6, float); break;
                case 7: if (out_f16) PULSAR_FP8_NT(7, __half); else PULSAR_FP8_NT(7, float); break;
                case 8: if (out_f16) PULSAR_FP8_NT(8, __half); else PULSAR_FP8_NT(8, float); break;
                case 9: if (out_f16) PULSAR_FP8_NT(9, __half); else PULSAR_FP8_NT(9, float); break;
                case 10: if (out_f16) PULSAR_FP8_NT(10, __half); else PULSAR_FP8_NT(10, float); break;
                case 11: if (out_f16) PULSAR_FP8_NT(11, __half); else PULSAR_FP8_NT(11, float); break;
                case 12: if (out_f16) PULSAR_FP8_NT(12, __half); else PULSAR_FP8_NT(12, float); break;
                case 13: if (out_f16) PULSAR_FP8_NT(13, __half); else PULSAR_FP8_NT(13, float); break;
                case 14: if (out_f16) PULSAR_FP8_NT(14, __half); else PULSAR_FP8_NT(14, float); break;
                case 15: if (out_f16) PULSAR_FP8_NT(15, __half); else PULSAR_FP8_NT(15, float); break;
                default: if (out_f16) PULSAR_FP8_NT(16, __half); else PULSAR_FP8_NT(16, float); break;  ///< n_tok == 16 == PULSAR_GPU_MNEUTRAL_ROWS_MAX
                }
                #undef PULSAR_FP8_NT
                return cuda_ok(cudaGetLastError(), "fp8_mx mmvq deint nt");
            }
        }
        /* Prefill (n_tok>1) uses the cuBLASLt MX tensor-core GEMM; decode falls
         * through to the per-token mmvq kernel below. */
        if (n_tok > 1 &&
                cuda_matmul_fp8_mx_tensor_labeled(out, model_map, model_size,
                weight_offset, in_dim, out_dim, x, n_tok, label)) return 1;
        const unsigned wpb = 8;  ///< output rows per block
        dim3 grid(((unsigned)out_dim + wpb - 1) / wpb);
        /* Prefer the de-interleaved cached weight (contiguous E4M3 -> coalesced 128-wide
         * loads, vs the raw 33B-interleaved kernel's misaligned 1-byte/thread reads).
         * Bit-exact vs the raw kernel (verified: rel 0 across all workhorse shapes);
         * ~+8% decode. */
        const fp8_mx_weight *w = (in_dim % 128 == 0 )
                ? cuda_fp8_mx_weight(model_map, weight_offset, fbytes, in_dim, out_dim, label)
                : NULL;
        if (w) {
            const int KBp = mx_rup((int)(in_dim / 32), 4);
            /* A8: if the producer already emitted this activation as E4M3 +
             * ue8m0, multiply in that format rather than against f32.  This is
             * the decode/spec path -- the ONLY place W8A32 still existed -- and
             * it is a FIDELITY change: the source computes with dynamic e4m3
             * activations, so f32 here is over-precision, not accuracy.  It is
             * NOT a speed win (the weight matrix dominates GEMV traffic by
             * ~out_dim x; see the kernel comment). */
            mxfp8_act_cache_t *ac8 = act_slot_find_rows(x->ptr, n_tok, in_dim);
            if (!ac8 && act_slot_a8_declared_short(x->ptr, n_tok, in_dim))
                return act_a8_contract_fail("decode GEMV", n_tok, in_dim, out_dim);
            if (ac8 && ac8->valid) {
                /* Say it once PER SHAPE, not once per process.  A gate PASS
                 * cannot distinguish "the A8 path ran and is correct" from "the
                 * A8 path never ran", and the second is what a mis-keyed cache
                 * would silently produce -- so a single global line proved only
                 * that SOME GEMV converted, which is the easy half. Keyed on
                 * in_dim, the log enumerates exactly which decode GEMVs reach
                 * W8A8: 4096 = attn_q_a/kv, 1024 = attn_q_b, 2048 = shared_down.
                 * Anything missing from that list is still multiplying against
                 * f32, which is the whole question when the goal is one
                 * activation format everywhere. */
                {
                    /* Keyed on the (in,out) PAIR, not in_dim alone: attn_q_a,
                     * attn_kv and the shared gate/up all read in_dim 4096, so
                     * an in_dim-only key reported the first and silently hid the
                     * other two -- an under-reporting diagnostic, which is the
                     * one failure mode a diagnostic must not have. */
                    static uint64_t seen[16] = {0};
                    static int n_seen = 0;
                    const uint64_t key = (in_dim << 32) ^ out_dim;
                    int known = 0;
                    for (int i = 0; i < n_seen; i++) if (seen[i] == key) { known = 1; break; }
                    if (!known && n_seen < 16) {
                        seen[n_seen++] = key;
                        fprintf(stderr, "pulsar: decode GEMV W8A8 (E4M3 acts) for "
                                        "in_dim=%llu out_dim=%llu\n",
                                (unsigned long long)in_dim, (unsigned long long)out_dim);
                    }
                }
                const int xKBp = mx_rup((int)(in_dim / 32), 4);
                /* The GEMV arms were the NINTH element-type defect, and the
                 * one that broke SERVING while every gate stayed green: both
                 * kernels are templated on OT, but these launches hardcoded
                 * (float *), deducing OT=float and writing 4-byte rows into
                 * the f16 Q buffer -- a 2x overwrite whose garbage both sides
                 * of every us-vs-us decode gate shared byte-for-byte.  Found
                 * by a serving curl, step-1 BOS cliff, 2026-08-22. */
                for (uint64_t t = 0; t < n_tok; t++) {
                    if (out_f16)
                        mxfp8_mmvq_deint_a8_kernel<<<grid, wpb * 32>>>(
                                (__half *)out->ptr + t * out_dim,
                                w->data, w->scale,
                                ac8->xq + t * in_dim, ac8->sx,
                                (int)in_dim, (int)out_dim, KBp, xKBp, (int)t);
                    else
                        mxfp8_mmvq_deint_a8_kernel<<<grid, wpb * 32>>>(
                                (float *)out->ptr + t * out_dim,
                                w->data, w->scale,
                                ac8->xq + t * in_dim, ac8->sx,
                                (int)in_dim, (int)out_dim, KBp, xKBp, (int)t);
                }
                return cuda_ok(cudaGetLastError(), "fp8_mx mmvq deint a8");
            }
            for (uint64_t t = 0; t < n_tok; t++) {
                if (out_f16)
                    mxfp8_mmvq_deint_kernel<<<grid, wpb * 32>>>((__half *)out->ptr + t * out_dim,
                            w->data, w->scale, (const float *)x->ptr + t * in_dim,
                            (int)in_dim, (int)out_dim, KBp);
                else
                    mxfp8_mmvq_deint_kernel<<<grid, wpb * 32>>>((float *)out->ptr + t * out_dim,
                            w->data, w->scale, (const float *)x->ptr + t * in_dim,
                            (int)in_dim, (int)out_dim, KBp);
            }
            return cuda_ok(cudaGetLastError(), "fp8_mx mmvq deint");
        }
        /* No raw arm any more.  It read 33B-interleaved blocks for plain
         * type-38 weights; the artifact is 390/390 pre-stored MXFP8_LT and the
         * loader refuses type 38, so nothing can select it.  A resolver failure
         * is now terminal here rather than silently slower. */
        fprintf(stderr, "pulsar: MXFP8 weight at offset %llu did not resolve\n",
                (unsigned long long)weight_offset);
        return 0;
    }
    fprintf(stderr,
            "pulsar: matmul %s at offset %llu is not a registered MXFP8 weight "
            "(legacy q8_0 weights are no longer supported)\n",
            label ? label : "mxfp8",
            (unsigned long long)weight_offset);
    return 0;
}



int pulsar_gpu_matmul_mxfp8_tensor(pulsar_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const pulsar_gpu_tensor *x, uint64_t n_tok) {
    return cuda_matmul_mxfp8_tensor_labeled(out, model_map, model_size, weight_offset,
                                           in_dim, out_dim, x, n_tok, "mxfp8");
}



int pulsar_gpu_matmul_mxfp8_pair_tensor(
        pulsar_gpu_tensor *out0,
        pulsar_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const pulsar_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out0_dim == 0 || out1_dim == 0 || n_tok == 0) {
        return 0;
    }
    /* A fused mxfp8_mmvq_deint_pair_kernel lived here until the 2026-08-22
     * launched-vs-defined sweep (L093).  It fired only when NO valid E4M3
     * encoding existed for x at n_tok==1 -- and gpu_decode.cpp arms exactly
     * that encoding off attn_norm before every call, so the fusion's only
     * remaining trigger was a cudaMalloc failure inside the act cache.  An
     * f32-activation kernel reachable only on an allocation-error path is a
     * W8A32/W8A8 split waiting for an OOM to expose it ("one activation
     * format, every batch size"); the per-weight calls below compute the same
     * function and derive their output type from the tensor. */
    if (act_slot_a8_declared_short(x->ptr, n_tok, in_dim) &&
        !act_slot_find_rows(x->ptr, n_tok, in_dim))
        return act_a8_contract_fail("qkv pair GEMV", n_tok, in_dim, out0_dim);
    return cuda_matmul_mxfp8_tensor_labeled(out0, model_map, model_size, weight0_offset,
                                           in_dim, out0_dim, x, n_tok, "mxfp8_pair0") &&
           cuda_matmul_mxfp8_tensor_labeled(out1, model_map, model_size, weight1_offset,
                                           in_dim, out1_dim, x, n_tok, "mxfp8_pair1");
}



int cuda_matmul_fp8_hc_expand_tensor_labeled(
        pulsar_gpu_tensor       *out_hc,
        pulsar_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        const pulsar_gpu_tensor *block_add,
        const pulsar_gpu_tensor *residual_hc,
        const pulsar_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label) {
    if (!out_hc || !block_out || !x || !residual_hc || !split || !model_map ||
        in_dim == 0 || out_dim == 0 || n_embd == 0 || n_hc == 0 ||
        out_dim != (uint64_t)n_embd) {
        return 0;
    }
    if (!g_fp8_offsets.count(weight_offset)) return 0;
    const uint64_t wstride = 33u;
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * wstride)) return 0;
    const uint64_t weight_bytes = out_dim * blocks * wstride;
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * PULSAR_HC_ELT_SIZE;  ///< residual_hc + out_hc are carriers
    const uint64_t split_bytes = (uint64_t)(2u * n_hc + n_hc * n_hc) * sizeof(float);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        block_out->bytes < out_dim * sizeof(float) ||
        residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes ||
        out_hc->bytes < hc_bytes ||
        (block_add && block_add->bytes < out_dim * sizeof(float))) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, label ? label : "fp8_hc_expand");
    if (!wptr) return 0;

    /* Decode uses the de-interleaved cached weight (coalesced vectorized loads); raw
     * 33B path is reached only when the dims rule out de-interleaving. */
    const fp8_mx_weight *dw = (in_dim % 32 == 0)
            ? cuda_fp8_mx_weight(model_map, weight_offset, weight_bytes, in_dim, out_dim,
                                 label ? label : "fp8_hc_expand")
            : NULL;
    const int KBp = mx_rup((int)(in_dim / 32), 4);
    const dim3 hg(((unsigned)out_dim + PULSAR_FP8MX_ROWS_PER_BLOCK - 1u) / PULSAR_FP8MX_ROWS_PER_BLOCK);
    const float *ba = block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr;
    /* A8: the "b" projection reads attn_low, which the "a" projection produced
     * one call earlier and encoded into the per-slot activation cache. If that
     * encoding is current, multiply in E4M3 instead of against f32 -- the last
     * decode consumer that was still W8A32. Cache miss falls through to f32,
     * which is a fallback, not a second format: the "a" projection emits
     * unconditionally, so a miss means the shapes disagreed and that is worth
     * finding, not silently living with. */
    if (dw) {
        const mxfp8_act_cache_t *ac8 = act_slot_find_rows(x->ptr, 1, in_dim);
        if (!ac8 && act_slot_a8_declared_short(x->ptr, 1, in_dim))
            return act_a8_contract_fail("attn-out 'b' GEMV", 1, in_dim, out_dim);
        if (ac8 && ac8->valid) {
            static int announced_ob8 = 0;
            if (!announced_ob8) {
                announced_ob8 = 1;
                fprintf(stderr, "pulsar: attn-out 'b' GEMV = W8A8 (E4M3 acts) for "
                                "in_dim=%llu out_dim=%llu\n",
                        (unsigned long long)in_dim, (unsigned long long)out_dim);
            }
            const int xKBp = mx_rup((int)(in_dim / 32), 4);
            matmul_fp8mx_hc_expand_warp8_kernel<true, true><<<hg, 256>>>(
                    (pulsar_hc_t *)out_hc->ptr, (float *)block_out->ptr, ba,
                    (const pulsar_hc_t *)residual_hc->ptr, (const float *)split->ptr,
                    (const unsigned char *)wptr, dw->data, dw->scale, KBp,
                    (const float *)x->ptr, in_dim, out_dim, n_embd, n_hc, blocks,
                    block_add ? 1 : 0, ac8->xq, ac8->sx, xKBp);
            return cuda_ok(cudaGetLastError(), "fp8_hc_expand a8 launch");
        }
    }
    if (dw) {
        matmul_fp8mx_hc_expand_warp8_kernel<true><<<hg, 256>>>(
                (pulsar_hc_t *)out_hc->ptr, (float *)block_out->ptr, ba,
                (const pulsar_hc_t *)residual_hc->ptr, (const float *)split->ptr,
                (const unsigned char *)wptr, dw->data, dw->scale, KBp, (const float *)x->ptr,
                in_dim, out_dim, n_embd, n_hc, blocks, block_add ? 1 : 0);
    } else {
        matmul_fp8mx_hc_expand_warp8_kernel<false><<<hg, 256>>>(
                (pulsar_hc_t *)out_hc->ptr, (float *)block_out->ptr, ba,
                (const pulsar_hc_t *)residual_hc->ptr, (const float *)split->ptr,
                (const unsigned char *)wptr, (const __nv_fp8_e4m3 *)NULL, (const unsigned char *)NULL, 0,
                (const float *)x->ptr, in_dim, out_dim, n_embd, n_hc, blocks, block_add ? 1 : 0);
    }
    return cuda_ok(cudaGetLastError(), "matmul_fp8_hc_expand launch");
}






/* L079/L087: the bf16 compute core, weight given as a resolved DEVICE pointer.
 * Two producers feed it: pulsar_gpu_matmul_bf16_tensor (native BF16 storage,
 * pointer straight into the mmap) and pulsar_gpu_matmul_f32_tensor (F32-source
 * storage, pointer into the once-converted bf16 copy -- see
 * f32_weight_bf16_resolve).  Everything numeric lives HERE, so the two weight
 * families cannot drift: same activation cache, same NT/GemmEx/GEMV arms,
 * same M-independence contract. */
static int matmul_bf16_wptr(pulsar_gpu_tensor *out, const uint16_t *w,
                            uint64_t in_dim, uint64_t out_dim,
                            const pulsar_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !w) return 0;
    /* inc 4 prefix-split (see the f16/mxfp8 twins): decode prefix [0,n_dec)
     * M-independent, prefill suffix [n_dec,n_tok) tensor-core, via pure-regime
     * recursion. */
    {
        const uint64_t n_dec = (uint64_t)g_mneutral_rows;
        if (n_dec > 0 && n_dec < n_tok) {
            const uint64_t inb = in_dim * sizeof(float), outb = out_dim * sizeof(float);
            pulsar_gpu_tensor out_pre = pulsar_tensor_subview(out, 0, out->bytes);
            pulsar_gpu_tensor x_pre   = pulsar_tensor_subview(x, 0, x->bytes);
            pulsar_gpu_tensor out_suf = pulsar_tensor_subview(out, n_dec * outb,
                                                             out->bytes - n_dec * outb);
            pulsar_gpu_tensor x_suf   = pulsar_tensor_subview(x, n_dec * inb,
                                                             x->bytes - n_dec * inb);
            const int saved = g_mneutral_rows;
            g_mneutral_rows = (int)n_dec;
            int r1 = matmul_bf16_wptr(&out_pre, w, in_dim, out_dim, &x_pre, n_dec);
            g_mneutral_rows = 0;
            int r2 = matmul_bf16_wptr(&out_suf, w, in_dim, out_dim, &x_suf, n_tok - n_dec);
            g_mneutral_rows = saved;
            return r1 && r2;
        }
    }
    /* Format check, not just size: these arms store f32 through an
     * untemplated float*, and an oversized narrowed batch buffer at small
     * n_tok would pass a pure byte bound (defect nine's shape). */
    if (pulsar_tensor_esz(x) != sizeof(float) ||
        pulsar_tensor_esz(out) != sizeof(float)) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;

    /* ONE bf16 activation for the whole call, produced before the arm is
     * chosen.  Every arm below reads these exact bytes, so "which kernel ran"
     * can no longer change the activation operand -- the failure mode this file
     * has now hit three times (the cuBLAS-rounds/nt-does-not split, the MoE
     * n==1 q8_1 split, and the A8 cache key).  It also replaces per-warp
     * rounding with one pass: a GEMV gives every out_dim/8 warps the whole
     * activation row, so narrowing at the load re-converted the same values
     * hundreds of times per launch and measured +2.9% step time.
     *
     * Caching stays on the EXACT key, unlike the A8 lookup next door.  It does
     * not need to be prefix-tolerant to be M-independent: bf16 conversion is
     * elementwise, so a cache miss converts exactly the rows this call owns and
     * gets byte-identical bytes to any other width that covers them.  A miss
     * costs one pass, never a different answer -- which is the property the A8
     * cache lacked, because there a miss changed the ARM and therefore the
     * arithmetic. */
    const uint64_t xb_count = n_tok * in_dim;
    /* Prefix-tolerant READ first -- the same move (and the same safety
     * argument) as the A8 consumer's act_slot_find_rows: the bf16 encoding is
     * elementwise and xb is row-major contiguous, so rows [0, n_tok) of a
     * WIDER valid block are byte-identical to a block staged at exactly
     * n_tok.  Not a rare case: the mixed-batch prefix split hands this
     * function the SAME base pointer at n_dec rows while the producer armed
     * the full batch width, so the exact-key lookup made every split prefix a
     * guaranteed miss (the T3 census showed them as the surviving
     * n_tok=1..8 converts). */
    __nv_bfloat16 *xb_pre = NULL;
    {
        mxfp8_act_cache_t *hw = act_slot_find_rows(x->ptr, n_tok, in_dim);
        if (hw && hw->valid_b && hw->xb) xb_pre = hw->xb;
    }
    mxfp8_act_cache_t *hb = xb_pre ? NULL : act_slot_find(x->ptr, n_tok, in_dim);
    if (hb && !mxfp8_act_cache_reserve((void **)&hb->xb, &hb->xb_cap,
                                       xb_count * sizeof(__nv_bfloat16), "act bf16")) {
        hb = NULL;
    }
    __nv_bfloat16 *xbb = xb_pre ? xb_pre
                       : hb     ? hb->xb
                                : (__nv_bfloat16 *)cuda_tmp_alloc(xb_count * sizeof(__nv_bfloat16),
                                                                  "bf16 activations");
    if (!xbb) return 0;
    /* The mxfp8 arms have carried this backstop since the f32 skips landed;
     * the bf16 arm converts FROM f32 and had none, so a skipped store plus an
     * unexpected miss here would convert unwritten bytes in silence. */
    if (!xb_pre && (!hb || !hb->valid_b) &&
        act_f32_absent_hazard(x->ptr, n_tok, in_dim)) {
        fprintf(stderr, "pulsar: bf16 activation convert would read a SKIPPED f32 "
                        "store (n_tok=%llu in_dim=%llu) -- refusing\n",
                (unsigned long long)n_tok, (unsigned long long)in_dim);
        return 0;
    }
    if (!xb_pre && (!hb || !hb->valid_b)) {
        /** Shape census for L086 T3 (producer-emits-bf16): each unique
         * (n_tok, in_dim) prints once, so the 169 convert launches the D1
         * profile counted become attributable to producers without a rerun.
         * Same announce discipline as the skip/tier prints in this file. */
        static uint64_t seen_shapes[8];
        static int n_seen = 0;
        const uint64_t shape_key = (n_tok << 32) | in_dim;
        int known = 0;
        for (int i = 0; i < n_seen; i++) if (seen_shapes[i] == shape_key) { known = 1; break; }
        if (!known && n_seen < 8) {
            seen_shapes[n_seen++] = shape_key;
            /* WHY did this convert run?  The T3b prefix-read should have
             * served base-pointer prefixes from the producer's entry, yet the
             * census shapes survived the nibble run unchanged -- so name the
             * miss instead of theorizing: does ANY slot cover this pointer,
             * and if one does, which half of the validity failed? */
            const mxfp8_act_cache_t *cov = act_slot_find_rows(x->ptr, n_tok, in_dim);
            int contained = 0;
            for (int i = 0; i < PULSAR_ACT_SLOTS; i++) {
                const mxfp8_act_cache_t *sl = &g_act_slots[i];
                if (!sl->key_ptr) continue;
                const char *b = (const char *)sl->key_ptr;
                if ((const char *)x->ptr >= b &&
                    (const char *)x->ptr < b + sl->key_ntok * sl->key_in_dim * sizeof(float)) {
                    contained = 1;
                    break;
                }
            }
            fprintf(stderr, "pulsar: bf16 act convert shape n_tok=%llu in_dim=%llu "
                            "(T3 census: cover=%d valid_b=%d xb=%d contained=%d)\n",
                    (unsigned long long)n_tok, (unsigned long long)in_dim,
                    cov != NULL, cov ? cov->valid_b : 0, cov ? (cov->xb != NULL) : 0,
                    contained);
        }
        f32_to_bf16_kernel<<<(xb_count + 255) / 256, 256>>>((uint16_t *)xbb,
                                                            (const float *)x->ptr, xb_count);
        if (!cuda_ok(cudaGetLastError(), "bf16 activation convert launch")) return 0;
        if (hb) hb->valid_b = 1;
    }
    const __nv_bfloat16 *xb16 = xbb;
    /* M-independence, same contract as the f16 and f32 arms.  All three arms now
     * read the SAME bf16 activation buffer built above, so they cannot disagree
     * on the operand at all and the only cross-arm difference left is
     * accumulation ORDER.  This comment read "the cuBLAS path below additionally
     * ROUNDS the activations to bf16, so its disagreement with the n=1 kernel is
     * larger than the f32 arm's, not smaller" until 2026-08-17, which was true
     * and was the bug. */
    {
        const uint64_t nt_cap = (g_mneutral_rows > 0)
                ? (uint64_t)PULSAR_GPU_MNEUTRAL_ROWS_MAX : 4u;
        if (n_tok >= 2 && n_tok <= nt_cap) {
            dim3 g((unsigned)out_dim);
            #define PULSAR_NT_LAUNCH(N) matmul_nt_kernel<N, __nv_bfloat16, __nv_bfloat16><<<g, 256>>>( \
                    (float *)out->ptr, (const __nv_bfloat16 *)w,                        \
                    xb16, in_dim, out_dim)
            switch (n_tok) {
            case 2: PULSAR_NT_LAUNCH(2); break;
            case 3: PULSAR_NT_LAUNCH(3); break;
            case 4: PULSAR_NT_LAUNCH(4); break;
            case 5: PULSAR_NT_LAUNCH(5); break;
            case 6: PULSAR_NT_LAUNCH(6); break;
            case 7: PULSAR_NT_LAUNCH(7); break;
            case 8: PULSAR_NT_LAUNCH(8); break;
            case 9: PULSAR_NT_LAUNCH(9); break;
            case 10: PULSAR_NT_LAUNCH(10); break;
            case 11: PULSAR_NT_LAUNCH(11); break;
            case 12: PULSAR_NT_LAUNCH(12); break;
            case 13: PULSAR_NT_LAUNCH(13); break;
            case 14: PULSAR_NT_LAUNCH(14); break;
            case 15: PULSAR_NT_LAUNCH(15); break;
            default: PULSAR_NT_LAUNCH(16); break;  ///< n_tok == 16 == PULSAR_GPU_MNEUTRAL_ROWS_MAX
            }
            #undef PULSAR_NT_LAUNCH
            return cuda_ok(cudaGetLastError(), "matmul_bf16 nt launch");
        }
    }
    if (g_cublas_ready && n_tok > 1) {
        const uint16_t *xb = (const uint16_t *)xb16;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(g_cublas,
                                         CUBLAS_OP_T, CUBLAS_OP_N,
                                         (int)out_dim, (int)n_tok, (int)in_dim,
                                         &alpha,
                                         w, CUDA_R_16BF, (int)in_dim,
                                         xb, CUDA_R_16BF, (int)in_dim,
                                         &beta,
                                         out->ptr, CUDA_R_32F, (int)out_dim,
                                         CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        return cublas_ok(st, "bf16 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_bf16_kernel<<<grid, 256>>>((float *)out->ptr, w, xb16, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_bf16 launch");
}


int pulsar_gpu_matmul_bf16_tensor(pulsar_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const pulsar_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "bf16");
    if (!wptr) return 0;
    return matmul_bf16_wptr(out, (const uint16_t *)wptr, in_dim, out_dim, x, n_tok);
}


/* L079/L087 phase 1: F32-source weights COMPUTE in bf16 on tensor cores.
 *
 * The source model's residual stream is BF16 -- there is no F32 compute
 * anywhere in the checkpoint's own math ([[ds4-source-numerics]]) -- so bf16
 * compute for the F32-stored family (hc_attn_fn/hc_ffn_fn/output_hc_fn/
 * *_ape) is CLOSER to source than the cublasSgemm/SIMT arms it replaces,
 * while the checkpoint storage stays exactly F32 (fidelity policy intact).
 * Priced by the 2026-08-25 fusion survey: this family was 14.4% of decode
 * GPU time (verify-batch NT kernels) + 6.2% of prefill (SIMT sgemm).
 *
 * The copy is converted ONCE per weight (immutable, keyed by offset --
 * same pattern as g_fp8_mx_by_offset next door) and lives beside the mmap:
 * the full F32 family is ~130 MiB of f32, so ~65 MiB of bf16 copies. */
static const uint16_t *f32_weight_bf16_resolve(const void *model_map,
                                               uint64_t model_size,
                                               uint64_t offset,
                                               uint64_t in_dim, uint64_t out_dim) {
    static std::unordered_map<uint64_t, uint16_t *> g_f32w_bf16;
    static uint64_t fc_off[4] = {~0ull, ~0ull, ~0ull, ~0ull};
    static const uint16_t *fc_ptr[4] = {};
    const int slot = (int)(offset & 3u);
    if (fc_off[slot] == offset) return fc_ptr[slot];
    auto it = g_f32w_bf16.find(offset);
    if (it != g_f32w_bf16.end()) {
        fc_off[slot] = offset; fc_ptr[slot] = it->second;
        return it->second;
    }
    const uint64_t n = in_dim * out_dim;
    const uint64_t f32_bytes = n * sizeof(float);
    if (offset > model_size || f32_bytes > model_size - offset) return NULL;
    const float *src = (const float *)cuda_model_range_ptr(model_map, offset,
                                                           f32_bytes, "f32->bf16 src");
    if (!src) return NULL;
    uint16_t *dst = NULL;
    if (cudaMalloc(&dst, n * sizeof(uint16_t)) != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "pulsar: f32->bf16 weight copy alloc failed (%.1f MiB) -- "
                        "REFUSING (no silent SIMT fallback)\n",
                (double)(n * 2) / 1048576.0);
        return NULL;
    }
    f32_to_bf16_kernel<<<(unsigned)((n + 255) / 256), 256>>>(dst, src, n);
    if (cudaGetLastError() != cudaSuccess) { (void)cudaFree(dst); return NULL; }
    g_f32w_bf16[offset] = dst;
    fc_off[slot] = offset; fc_ptr[slot] = dst;
    static uint64_t total = 0;
    total += n * 2;
    static int announced = 0;
    if (!announced) {
        announced = 1;
        fprintf(stderr, "pulsar: F32-source matmul family computes in BF16 tensor "
                        "cores (source residual stream is BF16; storage stays F32)\n");
    }
    return dst;
}






int pulsar_gpu_matmul_f32_tensor(pulsar_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const pulsar_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (out_dim > UINT64_MAX / in_dim) return 0;
    /* L079/L087 phase 1: resolve the once-converted bf16 copy and run the
     * SHARED bf16 core (see matmul_bf16_wptr).  What used to be here -- the
     * f32 NT kernels, cublasSgemm, and the f32 GEMV -- computed in f32 for
     * weights whose SOURCE math is bf16; the whole family now takes the
     * tensor-core path at every n_tok, so mixed-batch M-independence and the
     * n=1-vs-batch contract are the bf16 core's, uniformly.  Resolver failure
     * is TERMINAL (refuse-loud): a silent SIMT fallback would split the
     * family's numerics by allocation weather. */
    const uint16_t *w16 = f32_weight_bf16_resolve(model_map, model_size,
                                                  weight_offset, in_dim, out_dim);
    if (!w16) return 0;
    return matmul_bf16_wptr(out, w16, in_dim, out_dim, x, n_tok);
}



/* Decode grouped "a" projection: prefer the de-interleaved cached weight (vectorized
 * coalesced loads) over the raw 33B kernel. Bit-exact. */
/* L141: one A8 launch for every n the M-neutral lane can carry.  n == 1 keeps
 * the one-token kernel; 2..PULSAR_GPU_MNEUTRAL_ROWS_MAX stage each weight
 * block once for all rows.  Anything wider is not reachable today (above
 * a_gemv_max_n the tensor-core arm runs unless M-neutral, whose cap this
 * switch enumerates) and falls back to the per-token grid: a new caller gets
 * the old cost, never a different answer. */
static_assert(PULSAR_GPU_MNEUTRAL_ROWS_MAX == 16u,
              "grouped_fp8mx_a_a8_rows enumerates NT up to the M-neutral row cap");
static int launch_grouped_fp8mx_a_a8_rows(float *low, const fp8_mx_weight *dw, int KBp,
        const __nv_fp8_e4m3 *xq, const unsigned char *sx, int xKBp, uint64_t slab,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint32_t n_tokens,
        uint64_t blocks, uint64_t low_dim, const char *what) {
    const unsigned gx = ((unsigned)low_dim + PULSAR_FP8MX_ROWS_A8_PER_BLOCK - 1u)
                        / PULSAR_FP8MX_ROWS_A8_PER_BLOCK;
    #define PULSAR_OA_NT(N) grouped_fp8mx_a_nt_a8_kernel<N><<<dim3(gx, 1, 1), 256>>>( \
            low, dw->data, dw->scale, KBp, xq, sx, xKBp, slab,                        \
            group_dim, rank, n_groups, n_tokens, blocks)
    switch (n_tokens) {
    case 1:
        grouped_fp8mx_a_warp8_a8_kernel<<<dim3(gx, 1, 1), 256>>>(
                low, dw->data, dw->scale, KBp, xq, sx, xKBp, slab,
                group_dim, rank, n_groups, n_tokens, blocks);
        break;
    case 2: PULSAR_OA_NT(2); break;
    case 3: PULSAR_OA_NT(3); break;
    case 4: PULSAR_OA_NT(4); break;
    case 5: PULSAR_OA_NT(5); break;
    case 6: PULSAR_OA_NT(6); break;
    case 7: PULSAR_OA_NT(7); break;
    case 8: PULSAR_OA_NT(8); break;
    case 9: PULSAR_OA_NT(9); break;
    case 10: PULSAR_OA_NT(10); break;
    case 11: PULSAR_OA_NT(11); break;
    case 12: PULSAR_OA_NT(12); break;
    case 13: PULSAR_OA_NT(13); break;
    case 14: PULSAR_OA_NT(14); break;
    case 15: PULSAR_OA_NT(15); break;
    case 16: PULSAR_OA_NT(16); break;
    default:
        grouped_fp8mx_a_warp8_a8_kernel<<<dim3(gx, n_tokens, 1), 256>>>(
                low, dw->data, dw->scale, KBp, xq, sx, xKBp, slab,
                group_dim, rank, n_groups, n_tokens, blocks);
        break;
    }
    #undef PULSAR_OA_NT
    return cuda_ok(cudaGetLastError(), what);
}

static int launch_grouped_fp8mx_a(float *low, const void *model_map, uint64_t out_a_offset,
        uint64_t out_a_bytes, const unsigned char *out_a, uint64_t group_dim, uint64_t rank,
        uint32_t n_groups, uint32_t n_tokens, uint64_t blocks_a, uint64_t low_dim,
        const pulsar_heads_t *heads, const char *label) {
    const dim3 grid_a(((unsigned)low_dim + PULSAR_FP8MX_ROWS_PER_BLOCK - 1u) / PULSAR_FP8MX_ROWS_PER_BLOCK, (unsigned)n_tokens, 1);
    /* The A8 kernels carry PULSAR_FP8MX_ROWS_A8 rows per warp, not
     * PULSAR_FP8MX_ROWS, so they take their own grid (built in
     * launch_grouped_fp8mx_a_a8_rows) -- grid_a serves the f32 twin below and
     * keeps the 4-row geometry. */
    const fp8_mx_weight *dw = (group_dim % 32 == 0)
            ? cuda_fp8_mx_weight(model_map, out_a_offset, out_a_bytes, group_dim, low_dim, label) : NULL;
    const int KBp = mx_rup((int)(group_dim / 32), 4);

    /* A8: multiply the attn-output "a" projection in E4M3 rather than against
     * f32 heads. This is the largest byte consumer in the model and was the
     * last big decode GEMV still on W8A32.
     *
     * The encoding is made HERE rather than in the attention epilogue. Prefill
     * splits it across two producers (attn_f16 owns the nope blocks, rope_tail
     * the tail it rewrites) to avoid a separate pass over a
     * [n_tok x n_head x head_dim] tensor -- worth it at 4096 tokens. This path
     * only ever serves small n (decode, verify batches, mixed steps), where the
     * same tensor is ~128 KB and one quantise pass is noise next to the weight
     * traffic the GEMV is about to do. Doing it after the inverse rope also
     * means ONE producer owns the whole range, so there is no half-emitted
     * encoding to get wrong.
     *
     * Covers EVERY n this function serves, not just n==1: converting the n==1
     * kernel and leaving the 2..4 nt fusion on f32 would be a size-thresholded
     * activation format -- the exact defect removed from the mmvq pair path and
     * refused by name in the MoE down path. Format beats fusion -- and since
     * L141 the fusion is back IN the format: n >= 2 stages each weight block
     * once for all rows (grouped_fp8mx_a_nt_a8_kernel). */
    if (dw &&   /* dw != NULL implies group_dim%32==0 (K13) */
        rank % PULSAR_FP8MX_ROWS == 0 && low_dim % PULSAR_FP8MX_ROWS == 0) {
        const size_t slab = (size_t)mx_rup((int)n_tokens, 128) * (size_t)KBp;
        const size_t data_n  = (size_t)n_tokens * n_groups * group_dim;
        const size_t scale_n = (size_t)n_groups * slab;
        const size_t data_bytes_a8 = data_n * sizeof(__nv_fp8_e4m3);
        /* L106 K1: consume the producer-emitted grouped encoding when it is
         * current, exactly as the tensor-core arm does.  Before this probe,
         * only that arm consulted the cache, so the "a" activation was
         * f32->E4M3 above the gemv cap and f32->bf16->E4M3 at or below it --
         * a SIZE-THRESHOLDED ACTIVATION FORMAT, the defect this file refuses
         * by name two functions up, live on every spec verify with K <= 3.
         * The producer encodes from the attention accumulator registers; this
         * quantiser reads the bf16 heads store.  One probe removes the split:
         * every n now multiplies the same bytes the producer emitted, and the
         * quantise below remains only for buffers with no producer encoding
         * (decode g->heads, the drafter). */
        mxfp8_gact_cache_t *gc = gact_find(heads, n_tokens, n_groups, group_dim);
        if (gc && gc->valid && gc->kbp == KBp && gc->scale_slab == slab) {
            static int announced_gact_gemv = 0;
            if (!announced_gact_gemv) {
                announced_gact_gemv = 1;
                fprintf(stderr, "pulsar: attn-out 'a' GEMV = producer-emitted E4M3 "
                                "(grouped quantise pass skipped)\n");
            }
            return launch_grouped_fp8mx_a_a8_rows(low, dw, KBp,
                    gc->xq, gc->sx, gc->kbp, (uint64_t)gc->scale_slab,
                    group_dim, rank, n_groups, n_tokens, blocks_a, low_dim,
                    "attention_output_a a8 launch (gact)");
        }
        /** Soft failure on purpose: this path falls through to f32 rather than
         * failing the call, and the arena latches, so a refused reservation
         * arrives as a NULL take below and needs no separate branch. */
        cuda_arena ar;
        (void)cuda_arena_begin(&ar, mx_a256(data_bytes_a8) + mx_a256(scale_n),
                               "attn_out_a a8 acts");
        __nv_fp8_e4m3 *xq = (__nv_fp8_e4m3 *)cuda_arena_take(&ar, data_bytes_a8, 256);
        unsigned char *sx = (unsigned char *)cuda_arena_take(&ar, scale_n, 256);
        if (sx) {   /* take() latches, so sx != NULL implies xq != NULL */
            /* pulsar_mx_sfoff leaves holes; the GEMV reads only the (tok, kb) pairs the
             * quantiser fills, but zero the slab so a hole can never carry a
             * stale byte from a previous call's larger shape. */
            cudaMemsetAsync(sx, 0, scale_n, cudaStreamPerThread);
            const int warps = (int)n_tokens * (int)n_groups * (int)(group_dim / 32);
            mxfp8_quant_act_grouped_kernel<<<(warps * 32 + 255) / 256, 256>>>(
                    heads, (int)n_tokens, (int)n_groups, (int)group_dim, KBp, xq, sx, slab);
            if (cudaGetLastError() == cudaSuccess) {
                static int announced_oa8 = 0;
                if (!announced_oa8) {
                    announced_oa8 = 1;
                    fprintf(stderr, "pulsar: attn-out 'a' GEMV = W8A8 (E4M3 acts) for "
                                    "group_dim=%llu rank=%llu\n",
                            (unsigned long long)group_dim, (unsigned long long)rank);
                }
                return launch_grouped_fp8mx_a_a8_rows(low, dw, KBp, xq, sx, KBp, (uint64_t)slab,
                        group_dim, rank, n_groups, n_tokens, blocks_a, low_dim,
                        "attention_output_a a8 launch");
            }
            /* Quantise failed: fall through to f32 rather than run on a partly
             * written buffer. */
        }
    }
    /* Small verify batches: one nt launch, weight blocks L1-shared across tokens;
     * per-token bit-identical to the n=1 DEINT kernel below. */
    if (dw && n_tokens >= 2u && n_tokens <= 4u &&
        rank % PULSAR_FP8MX_ROWS == 0 && low_dim % PULSAR_FP8MX_ROWS == 0) {
        const dim3 g(((unsigned)low_dim + PULSAR_FP8MX_ROWS_PER_BLOCK - 1u) / PULSAR_FP8MX_ROWS_PER_BLOCK);
        switch (n_tokens) {
        case 2: grouped_fp8mx_a_nt_kernel<2, pulsar_heads_t><<<g, 256>>>(low, dw->data, dw->scale, KBp,
                heads, group_dim, rank, n_groups, blocks_a); break;
        case 3: grouped_fp8mx_a_nt_kernel<3, pulsar_heads_t><<<g, 256>>>(low, dw->data, dw->scale, KBp,
                heads, group_dim, rank, n_groups, blocks_a); break;
        default: grouped_fp8mx_a_nt_kernel<4, pulsar_heads_t><<<g, 256>>>(low, dw->data, dw->scale, KBp,
                heads, group_dim, rank, n_groups, blocks_a); break;
        }
        return cuda_ok(cudaGetLastError(), "attention_output_a nt launch");
    }
    if (!dw) {
        /* L106 K3: this used to soft-fall to the raw-33B DEINT=false arm.  A
         * NULL resolver here means the pre-stored MXFP8_LT weight failed to
         * resolve -- a should-never-happen path whose fallback silently ran a
         * different kernel on the same call: the dense dispatcher made the
         * twin case TERMINAL (L083 C3, "refuse-loud"), and this entry now
         * follows the same rule.  A loud failure beats a quiet arm change. */
        fprintf(stderr, "pulsar: attn-out 'a' weight did not resolve (deint) -- refusing "
                        "the raw fallback (group_dim=%llu rank=%llu)\n",
                (unsigned long long)group_dim, (unsigned long long)rank);
        return 0;
    }
    grouped_fp8mx_a_warp8_kernel<true, pulsar_heads_t><<<grid_a, 256>>>(low, out_a, dw->data, dw->scale, KBp,
            heads, group_dim, rank, n_groups, n_tokens, blocks_a);
    return cuda_ok(cudaGetLastError(), "attention_output_a launch");
}


int pulsar_gpu_attention_output_batch_tensor(
        pulsar_gpu_tensor       *out,
        pulsar_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *heads,
        uint32_t                n_tokens) {
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0) {
        return 0;
    }
    /* inc 4 prefix-split: recurse the whole attn-output stage (a-proj + b-proj)
     * over the decode prefix [0,n_dec) in the M-independent regime and the prefill
     * suffix [n_dec,n_tokens) in the tensor-core regime. Both the warp8/nt 'a' path
     * and the mxfp8 'b' path then run their PURE code for each range (the 'b' proj
     * recurses no further: its sub-range already has n_dec in {n_tok',0}). */
    {
        const uint64_t n_dec = (uint64_t)g_mneutral_rows;
        if (n_dec > 0 && n_dec < (uint64_t)n_tokens) {
            const uint64_t low_dim = (uint64_t)n_groups * rank;
            const uint64_t headb = (uint64_t)n_groups * group_dim * PULSAR_HEADS_ELT_SIZE;
            const uint64_t lowb  = low_dim * sizeof(float);
            const uint64_t outb  = out_dim * sizeof(float);
            pulsar_gpu_tensor out_pre = pulsar_tensor_subview(out, 0, out->bytes);
            pulsar_gpu_tensor low_pre = pulsar_tensor_subview(low, 0, low->bytes);
            pulsar_gpu_tensor hd_pre  = pulsar_tensor_subview(heads, 0, heads->bytes);
            pulsar_gpu_tensor out_suf = pulsar_tensor_subview(out, n_dec * outb, out->bytes - n_dec * outb);
            pulsar_gpu_tensor low_suf = pulsar_tensor_subview(low, n_dec * lowb, low->bytes - n_dec * lowb);
            pulsar_gpu_tensor hd_suf  = pulsar_tensor_subview(heads, n_dec * headb,
                                                             heads->bytes - n_dec * headb);
            const int saved = g_mneutral_rows;
            g_mneutral_rows = (int)n_dec;
            int r1 = pulsar_gpu_attention_output_batch_tensor(&out_pre, &low_pre, model_map,
                    model_size, out_a_offset, out_b_offset, group_dim, rank, n_groups,
                    out_dim, &hd_pre, (uint32_t)n_dec);
            g_mneutral_rows = 0;
            int r2 = pulsar_gpu_attention_output_batch_tensor(&out_suf, &low_suf, model_map,
                    model_size, out_a_offset, out_b_offset, group_dim, rank, n_groups,
                    out_dim, &hd_suf, n_tokens - (uint32_t)n_dec);
            g_mneutral_rows = saved;
            return r1 && r2;
        }
    }
    if (!g_fp8_offsets.count(out_a_offset) || !g_fp8_offsets.count(out_b_offset)) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t blocks_b = (low_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 33u;
    const uint64_t out_b_bytes = out_dim * blocks_b * 33u;
    if (out_a_offset > model_size || out_b_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        out_b_bytes > model_size - out_b_offset ||
        heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * PULSAR_HEADS_ELT_SIZE ||
        low->bytes < (uint64_t)n_tokens * low_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }

    /* "a" projection: prefill takes the block-scaled MXFP8xMXFP8 tensor-core
     * GEMMs; decode and small verify batches (n_tokens<=4) take the
     * register-blocked GEMV path (launch_grouped_fp8mx_a dispatches the nt
     * variant at 2..4 -- one launch vs 8 per-group GEMMs, bit-identical per
     * token to decode's kernel).  (A sentence ended here describing an env
     * override that restored the dispatch for all n_tokens>1; the override is
     * gone and the sentence was truncated mid-clause.  The same fragment,
     * near-verbatim, sat at the mxfp8 nt cap -- one edit left both behind.) */
    /* 2026-07-21: raising to 8 was TRIED and REVERTED (see the dense-matmul gate
     * for the measurement).  It shared its cap with that gate, so the two
     * defaults must move together. */
    static const int a_gemv_max_n = 4;
    int a_done = 0;
    /* plan-34 inc 2: a batched multiseq/mixed step must NOT take the M-dependent
     * tensor-core (cuBLASLt) 'a' GEMM -- fall to launch_grouped_fp8mx_a, whose
     * warp8/nt kernels are per-token M-independent (bit-identical to the n=1
     * DEINT kernel), so a co-scheduled decode bank's attn-output row is invariant
     * to the batch width. Classic prefill (not armed) keeps the tensor-core path. */
    if (n_tokens > 1 && (int)n_tokens > a_gemv_max_n && g_mneutral_rows == 0) {
        a_done = cuda_attention_output_a_mx_gemm(low, model_map, model_size, out_a_offset,
                                                 group_dim, rank, n_groups, heads, n_tokens);
    }
    if (!a_done) {
        const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
                cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
        if (!out_a) return 0;
        if (!launch_grouped_fp8mx_a((float *)low->ptr, model_map, out_a_offset, out_a_bytes, out_a,
                                    group_dim, rank, n_groups, n_tokens, blocks_a, low_dim,
                                    (const pulsar_heads_t *)heads->ptr, "attn_out_a")) return 0;
    }
    /* Emit `low` here too, for the same reason as the n==1 entry below: the "b"
     * GEMM consumes it next. This entry is what the SERVER actually takes --
     * with the drafter live, verify batches come through here, not through
     * attention_output_low_tensor -- so emitting only there converted the
     * benchmark and left production on f32. The bench has no drafter, so it
     * could not have shown that. */
    if (low_dim % 256 == 0) {
        void *lq = NULL, *lsf = NULL; int lkbp = 0;
        if (pulsar_gpu_mxfp8_act_cache_e4m3_slot(low, n_tokens, low_dim, &lq, &lsf, &lkbp)) {
            const int lwarps = (int)n_tokens * (int)(low_dim / 32);
            mxfp8_quant_act_kernel<<<(lwarps * 32 + 255) / 256, 256>>>(
                    (const float *)low->ptr, (int)n_tokens, (int)low_dim, lkbp,
                    (__nv_fp8_e4m3 *)lq, (unsigned char *)lsf);
            if (cudaGetLastError() == cudaSuccess) {
                pulsar_gpu_mxfp8_act_cache_arm(low, n_tokens, low_dim);
                pulsar_gpu_mxfp8_act_cache_note_mxfp8();
            }
        }
    }

    return cuda_matmul_mxfp8_tensor_labeled(out,
                                           model_map,
                                           model_size,
                                           out_b_offset,
                                           low_dim,
                                           out_dim,
                                           low,
                                           n_tokens,
                                           "attn_output_b");
}



int pulsar_gpu_attention_output_low_tensor(
        pulsar_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const pulsar_gpu_tensor *heads) {
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 || n_groups == 0) {
        return 0;
    }
    if (!g_fp8_offsets.count(out_a_offset)) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 33u;
    if (out_a_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        heads->bytes < (uint64_t)n_groups * group_dim * PULSAR_HEADS_ELT_SIZE ||
        low->bytes < low_dim * sizeof(float)) {
        return 0;
    }
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
    if (!out_a) return 0;

    if (!launch_grouped_fp8mx_a((float *)low->ptr, model_map, out_a_offset, out_a_bytes, out_a,
                                group_dim, rank, n_groups, 1, blocks_a, low_dim,
                                (const pulsar_heads_t *)heads->ptr, "attn_out_a")) return 0;

    /* Emit `low` so the "b" projection that consumes it next multiplies in
     * E4M3 too. Done here rather than as an epilogue on the "a" kernel: the
     * warp that reduces a row does not hold that row's block neighbours, so an
     * in-kernel encode would need a second pass regardless, and `low` is only
     * low_dim floats (32 KB) at decode. Feeds BOTH "b" arms -- the fused
     * hc_expand's A8 branch reads this slot, and the non-fused
     * cuda_matmul_mxfp8_tensor_labeled finds it on its own. A failure to get a
     * slot is not an error: "b" falls back to f32. */
    if (low_dim % 256 == 0) {
        void *lq = NULL, *lsf = NULL; int lkbp = 0;
        if (pulsar_gpu_mxfp8_act_cache_e4m3_slot(low, 1, low_dim, &lq, &lsf, &lkbp)) {
            const int lwarps = (int)(low_dim / 32);
            mxfp8_quant_act_kernel<<<(lwarps * 32 + 255) / 256, 256>>>(
                    (const float *)low->ptr, 1, (int)low_dim, lkbp,
                    (__nv_fp8_e4m3 *)lq, (unsigned char *)lsf);
            if (cudaGetLastError() == cudaSuccess) {
                pulsar_gpu_mxfp8_act_cache_arm(low, 1, low_dim);
                pulsar_gpu_mxfp8_act_cache_note_mxfp8();
            }
        }
    }
    return 1;
}


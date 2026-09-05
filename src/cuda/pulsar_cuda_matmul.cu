#include <atomic>
#include "pulsar_cuda_internal.h"
#include "pulsar_cuda_mx.cuh"   /* the single source for pulsar_mx_sfoff */





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
template <int NT, int R, typename WT, typename AT = float>
__global__ static void matmul_nt_kernel(
        float *out,
        const WT *w,
        const AT *x,
        uint64_t in_dim,
        uint64_t out_dim) {
    /* L151-D (2026-09-03): R output rows per block.  One block per row
     * re-read the whole [NT][in_dim] activation tile from L2 for every output
     * row (the 1 GB head at NT=16: 129280 rows x 128 KB = 16 GB of L2 traffic
     * per call, +0.5 ms per row in the L151 sweep).  With R rows per block the
     * per-token activation loads are issued once and applied to R weight rows.
     * R FOLLOWS THE ROW COUNT (nt_rows_per_block: 1 up to 4 rows, 2 up to 8,
     * 4 above, capped by out_dim) -- that is allowed because R never changes
     * a (row, token) accumulator's sequence: the per-(row, token) FMA
     * sequence and the 256-wide tree reduction below are exactly the R=1
     * kernel's, so every output is bit-identical at every M and every R.
     * (This comment said "R is a per-shape constant, never the batch" until
     * L191; the argument at nt_rows_per_block was the correct one.) */
    const uint64_t row0 = (uint64_t)blockIdx.x * R;
    if (row0 >= out_dim) return;

    float sum[R][NT];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        #pragma unroll
        for (int t = 0; t < NT; t++) sum[r][t] = 0.0f;
    }
    /* L144: the same per-thread walk (i = tid, tid + 256, ...) and the same
     * FMA sequence per accumulator, but the loads of U consecutive strides are
     * issued before any of their FMAs.  As one dependent load per FMA the
     * 16384-wide hc rows ran at 27 GB/s on a grid of 24 blocks -- a latency
     * chain, not a bandwidth limit.  Accumulation ORDER is untouched, so every
     * output is bit-identical (L109 N1's split-K changed the order and cost 8%
     * tokens/step -- 35220f6).  U shrinks with NT to keep xv[] in registers. */
    constexpr int U = NT <= 4 ? 8 : (NT <= 8 ? 4 : 2);
    const uint64_t stride = blockDim.x;
    uint64_t i = threadIdx.x;
    for (; i + (uint64_t)(U - 1) * stride < in_dim; i += (uint64_t)U * stride) {
        float xv[U][NT];
        #pragma unroll
        for (int u = 0; u < U; u++) {
            #pragma unroll
            for (int t = 0; t < NT; t++) xv[u][t] = pulsar_at_load(x, t * in_dim + i + (uint64_t)u * stride);
        }
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const uint64_t row = row0 + r;
            if (row >= out_dim) break;
            const WT *wr = w + row * in_dim;
            float wv[U];
            #pragma unroll
            for (int u = 0; u < U; u++) wv[u] = pulsar_wt_load(wr, i + (uint64_t)u * stride);
            #pragma unroll
            for (int u = 0; u < U; u++) {
                #pragma unroll
                for (int t = 0; t < NT; t++) sum[r][t] += wv[u] * xv[u][t];
            }
        }
    }
    for (; i < in_dim; i += stride) {
        float xv[NT];
        #pragma unroll
        for (int t = 0; t < NT; t++) xv[t] = pulsar_at_load(x, t * in_dim + i);
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const uint64_t row = row0 + r;
            if (row >= out_dim) break;
            const float wv = pulsar_wt_load(w + row * in_dim, i);
            #pragma unroll
            for (int t = 0; t < NT; t++) sum[r][t] += wv * xv[t];
        }
    }

    /* NT trees per row, in lockstep: the same partial[tid] += partial[tid +
     * stride] sequence per (row, token) as before, one __syncthreads per
     * level for all NT of them. */
    __shared__ float partial[NT][256];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        const uint64_t row = row0 + r;
        #pragma unroll
        for (int t = 0; t < NT; t++) partial[t][threadIdx.x] = sum[r][t];
        __syncthreads();
        for (uint32_t st = blockDim.x >> 1; st > 0; st >>= 1) {
            if (threadIdx.x < st) {
                #pragma unroll
                for (int t = 0; t < NT; t++) partial[t][threadIdx.x] += partial[t][threadIdx.x + st];
            }
            __syncthreads();
        }
        if (threadIdx.x < NT && row < out_dim) out[threadIdx.x * out_dim + row] = partial[threadIdx.x][0];
        __syncthreads();
    }
}

/* L151-D: rows per block for the generic nt kernel -- same rule and same
 * neutrality argument as the A8 twin PULSAR_FP8MX_ROWS_A8 (rows per warp,
 * fixed at build time): R never changes a (row, token)
 * accumulator's sequence, so it may follow the row count.  1 up to 4 rows
 * (roofline already), 2 up to 8, 4 above; capped by out_dim (the 1 GB head
 * takes 4, N >= 8192 takes 2, small shapes stay 1). */
static int nt_rows_per_block(int NT, uint64_t out_dim) {
    const int by_rows = NT <= 4 ? 1 : (NT <= 8 ? 2 : 4);
    const int by_shape = out_dim >= 65536 ? 4 : out_dim >= 8192 ? 2 : 1;
    return by_rows < by_shape ? by_rows : by_shape;
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
    /* L144: loads for 8 strides hoisted ahead of their FMAs; walk and
     * accumulation order unchanged -- see matmul_nt_kernel. */
    constexpr int U = 8;
    const uint64_t stride = blockDim.x;
    uint64_t i = threadIdx.x;
    for (; i + (uint64_t)(U - 1) * stride < in_dim; i += (uint64_t)U * stride) {
        float wv[U], xv[U];
        #pragma unroll
        for (int u = 0; u < U; u++) {
            wv[u] = bf16_to_f32(wr[i + (uint64_t)u * stride]);
            xv[u] = __bfloat162float(xr[i + (uint64_t)u * stride]);
        }
        #pragma unroll
        for (int u = 0; u < U; u++) sum += wv[u] * xv[u];
    }
    for (; i < in_dim; i += stride) {
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
 * group_dim % 32 == 0 (whole blocks, so bn is always 32). Anything else is
 * refused by the launcher (launch_grouped_fp8mx_a). */

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
        uint32_t x_tok_stride,   ///< rows per group in the ENCODING (the slot's row count, >= n_tokens)
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

    const __nv_fp8_e4m3 *xr = xdata + (group * (uint64_t)x_tok_stride + tok) * group_dim;
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
        uint32_t x_tok_stride,
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
    const __nv_fp8_e4m3 *xg = xdata + group * (uint64_t)x_tok_stride * group_dim;
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
 * cleanup can invalidate it together with g_fp8_mx_by_offset).
 *
 * L191: the tag and the pointer are one entry written in a fixed order --
 * pointer first, then tag with release; readers load the tag with acquire and
 * only then the pointer -- so a reader that matches the tag never pairs it
 * with the previous entry's pointer.  The weight caches are process-global on
 * purpose (one model, arena-owned device copies shared by every submitting
 * thread, cleared at engine close), so they are NOT thread_local: a per-thread
 * copy would resolve and convert every weight once per thread. */
constexpr uint32_t FP8_FC = 2048u;
struct fp8_fc_entry {
    std::atomic<uint64_t> tag;                  ///< 0 = empty; real offsets are never 0
    std::atomic<const fp8_mx_weight *> ptr;
};
static fp8_fc_entry g_fp8_fc[FP8_FC];


/* F32-source -> bf16 weight copies (f32_weight_bf16_resolve below), keyed by
 * the full identity of the weight (L191).  Declared here so
 * cuda_fp8_weight_cache_clear can clear them with the fp8 caches. */
struct f32w_key {
    const void *model_map;
    uint64_t offset, in_dim, out_dim;
    bool operator==(const f32w_key &o) const {
        return model_map == o.model_map && offset == o.offset && in_dim == o.in_dim && out_dim == o.out_dim;
    }
};
struct f32w_key_hash {
    size_t operator()(const f32w_key &k) const {
        uint64_t h = (uint64_t)(uintptr_t)k.model_map * 0x9E3779B97F4A7C15ull;
        h ^= k.offset + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
        h ^= (k.in_dim << 32 | k.out_dim) + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
        return (size_t)h;
    }
};
static std::unordered_map<f32w_key, uint16_t *, f32w_key_hash> g_f32w_bf16;
constexpr uint32_t F32W_FC = 8u;
struct f32w_fc_entry { f32w_key key; const uint16_t *ptr; };
static f32w_fc_entry g_f32w_fc[F32W_FC];

/* lazily de-interleave + swizzle an MXFP8 weight into device buffers, cached by offset. */
static const fp8_mx_weight *cuda_fp8_mx_weight(const void *model_map, uint64_t offset, uint64_t weight_bytes,
                                               uint64_t in_dim, uint64_t out_dim, const char *label) {
    /* The same ~300 weight offsets are resolved once per layer every token on
     * the launch-serializing host thread. A tiny direct-mapped cache in front
     * of the unordered_map skips the probe on the hot repeat; a miss or hash
     * collision just falls through (benign), and the cached pointer is
     * re-validated (map references are stable across inserts). */
    constexpr uint32_t FC = FP8_FC;
    fp8_fc_entry *fc = &g_fp8_fc[(uint32_t)(((offset >> 5) ^ (offset >> 17)) & (FC - 1u))];
    if (offset != 0 && fc->tag.load(std::memory_order_acquire) == offset) {
        const fp8_mx_weight *p = fc->ptr.load(std::memory_order_relaxed);
        if (p && p->host_base == model_map && p->in_dim == in_dim && p->out_dim == out_dim) return p;
    }
    auto it = g_fp8_mx_by_offset.find(offset);
    if (it != g_fp8_mx_by_offset.end() && it->second.host_base == model_map &&
        it->second.in_dim == in_dim && it->second.out_dim == out_dim) {
        fc->ptr.store(&it->second, std::memory_order_relaxed);
        fc->tag.store(offset, std::memory_order_release);
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
            fc->ptr.store(wp, std::memory_order_relaxed);
            fc->tag.store(offset, std::memory_order_release);
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
 * slot carries a bf16 plane (valid_b / xb, written by the producer's epilogue
 * via pulsar_gpu_bf16_act_slot) alongside the E4M3 one; a producer fills only
 * the planes its consumers read.  (This paragraph once described an F16 copy for "F16 cuBLAS
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

/* L159: the bf16 consumer's lookup.  A bf16 plane is elementwise and row-major,
 * so ANY row window of a valid plane is byte-identical to a plane produced for
 * exactly that window -- the same argument act_slot_find_rows makes for a
 * prefix, extended to an offset: the compressor's 4-row tail view of attn_norm
 * and the mixed-batch suffix both hand this a pointer INSIDE the producer's
 * buffer.  Returns the slot and the first row of the window. */
static mxfp8_act_cache_t *act_slot_find_window(const void *ptr, uint64_t need, uint64_t in_dim,
                                               uint64_t *row0_out) {
    if (!ptr || need == 0 || in_dim == 0) return NULL;
    const uint64_t rowb = in_dim * sizeof(float);
    mxfp8_act_cache_t *best = NULL; uint64_t best_row0 = 0;
    for (int i = 0; i < PULSAR_ACT_SLOTS; i++) {
        mxfp8_act_cache_t *s = &g_act_slots[i];
        if (!s->key_ptr || s->key_in_dim != in_dim) continue;
        const char *b = (const char *)s->key_ptr, *p = (const char *)ptr;
        if (p < b) continue;
        const uint64_t off = (uint64_t)(p - b);
        if (off % rowb) continue;
        const uint64_t row0 = off / rowb;
        if (row0 + need > s->key_ntok) continue;
        if (!best || s->key_ntok < best->key_ntok) { best = s; best_row0 = row0; }
    }
    if (best) { best->lru = ++g_act_clock; *row0_out = best_row0; }
    return best;
}

/* Two different misses on the activation slot, two different refusals:
 *
 *   NO SLOT AT ALL -- no producer armed this buffer.  act_a8_missing_fail.
 *
 *   A SLOT EXISTS AND IS VALID BUT COVERS FEWER ROWS THAN ASKED -- the producer
 *     declared E4M3 for this buffer and this consumer cannot use the
 *     declaration (armed at full batch width, consumed at the decode prefix).
 *     act_a8_contract_fail.
 *
 * Neither returns a number from a different computation: one format or an
 * error. */
static bool act_slot_a8_declared_short(const void *ptr, uint64_t need, uint64_t in_dim) {
    if (!ptr || need == 0) return false;
    for (int i = 0; i < PULSAR_ACT_SLOTS; i++) {
        const mxfp8_act_cache_t *s = &g_act_slots[i];
        if (s->key_ptr == ptr && s->key_in_dim == in_dim && s->valid &&
            s->key_ntok < need) return true;
    }
    return false;
}

/* NO E4M3 encoding exists for this activation and none was declared: refuse,
 * printed once per (in,out) shape, counted on every call (pulsar_shape_once;
 * the old 16-entry table went silent at the 17th shape -- L189). */
static int act_a8_missing_fail(const char *what, uint64_t need,
                               uint64_t in_dim, uint64_t out_dim) {
    static pulsar_shape_once seen = {};
    if (pulsar_shape_once_first(&seen, pulsar_shape_key(in_dim, out_dim), "act_a8_missing_fail")) {
        fprintf(stderr,
                "pulsar: NO E4M3 ENCODING for the activation of %s (in_dim=%llu out_dim=%llu "
                "n_tok=%llu) -- no producer armed the slot.  Arm the activation "
                "(pulsar_gpu_mxfp8_act_cache_arm) and have its producer emit E4M3 "
                "(pulsar_gpu_mxfp8_act_cache_encode_f32 for a synthesised x).  Refusing "
                "(refusal #%llu of this kind).\n",
                what ? what : "?", (unsigned long long)in_dim,
                (unsigned long long)out_dim, (unsigned long long)need,
                (unsigned long long)seen.n_calls);
    }
    return 0;
}

/* Report it once per (in,out) shape and fail.  Keyed on the pair for the reason
 * the W8A8 announcements are: several decode GEMVs share in_dim 4096, so an
 * in_dim-only key would report one and hide the rest. */
static int act_a8_contract_fail(const char *what, uint64_t need,
                                uint64_t in_dim, uint64_t out_dim) {
    static pulsar_shape_once seen = {};
    if (pulsar_shape_once_first(&seen, pulsar_shape_key(in_dim, out_dim), "act_a8_contract_fail")) {
        fprintf(stderr,
                "pulsar: A8 CONTRACT VIOLATION in %s -- the producer declared E4M3 for "
                "this activation but the cached block is narrower than the %llu rows "
                "asked for (in_dim=%llu out_dim=%llu).  Refusing to silently multiply "
                "against f32 instead (refusal #%llu of this kind).\n",
                what, (unsigned long long)need,
                (unsigned long long)in_dim, (unsigned long long)out_dim,
                (unsigned long long)seen.n_calls);
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
     * through offset VIEWS -- gpu_prefill.cpp:143-168 takes the last n_tail <= 7
     * rows (n_full + rem, L168) of batch_attn_norm for the ratio-4 compressor
     * rebuild and hands them to a plain matmul at n_tok = n_tail.  A view keys
     * no slot, so an equality test sees
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
         * rebuild reads the last n_tail <= 7 rows of attn_norm).  Absent rows are
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


void pulsar_gpu_mxfp8_act_cache_note_f32_skipped(uint32_t keep_from) {
    if (g_act_cur) {
        g_act_cur->f32_absent    = 1;
        g_act_cur->f32_keep_from = keep_from;
    }
}

/** L157: the same declaration for a buffer that was never arm()ed -- the
 * hidden-carrier norm has no E4M3 consumer, so it holds only a bf16 slot and
 * g_act_cur points at whatever was armed last.  Name the slot by its key. */
void pulsar_gpu_act_note_f32_skipped_for(const pulsar_gpu_tensor *x, uint64_t n_tok,
                                         uint64_t in_dim, uint32_t keep_from) {
    mxfp8_act_cache_t *s = x ? act_slot_find(x->ptr, n_tok, in_dim) : NULL;
    if (s) {
        s->f32_absent    = 1;
        s->f32_keep_from = keep_from;
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
    /* Every refusal below is named (L189): the callers used to reset their
     * pointers to NULL and continue, and the miss surfaced layers later under
     * the consuming GEMM's name. */
    if (!x || n_tok == 0 || in_dim == 0 || (in_dim % 32) != 0 ||
        !data_out || !scale_out || !sf_pitch) {
        fprintf(stderr, "pulsar: E4M3 act slot: bad request (x=%d n_tok=%llu in_dim=%llu; in_dim must be a "
                        "non-zero multiple of 32, every out pointer non-NULL) -- refusing\n",
                x != NULL, (unsigned long long)n_tok, (unsigned long long)in_dim);
        return 0;
    }
    mxfp8_act_cache_t *s = act_slot_acquire(x->ptr, n_tok, in_dim);
    if (!s) {
        fprintf(stderr, "pulsar: E4M3 act slot: no slot for buffer %p (n_tok=%llu in_dim=%llu) -- refusing\n",
                x->ptr, (unsigned long long)n_tok, (unsigned long long)in_dim);
        return 0;
    }
    const int ntok = (int)n_tok;
    const int KBp  = pulsar_mx_kbp((int)in_dim);
    const size_t sx_bytes = pulsar_mx_sf_slab_bytes(ntok, KBp);
    if (!mxfp8_act_cache_reserve((void **)&s->xq, &s->xq_cap,
                                 (size_t)(n_tok * in_dim), "act data") ||
        !mxfp8_act_cache_reserve((void **)&s->sx, &s->sx_cap,
                                 sx_bytes, "act scale")) {
        fprintf(stderr, "pulsar: E4M3 act slot: device reserve failed (data %llu B, scale %llu B; "
                        "n_tok=%llu in_dim=%llu) -- refusing\n",
                (unsigned long long)(n_tok * in_dim), (unsigned long long)sx_bytes,
                (unsigned long long)n_tok, (unsigned long long)in_dim);
        return 0;
    }
    /* The quantizer memsets the scale slab because pulsar_mx_sfoff leaves holes when
     * rows/blocks are not multiples of 128/4; a producer filling only the live
     * (row, kb) pairs must do the same or the GEMM reads stale swizzle slots.
     * cuda_ok: an unchecked failure here left the error sticky for the next
     * unrelated check to report under its own name (L189). */
    if (!cuda_ok(cudaMemsetAsync(s->sx, 0, sx_bytes, 0), "E4M3 act slot scale-slab memset")) return 0;
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

/* L159: forget every plane keyed on a buffer that is about to be freed.  A
 * per-call scratch tensor (the verify heads' output_norm) can be re-allocated
 * at the same address for a different buffer; a valid_b left behind would
 * then satisfy a window lookup with last call's rows. */
void pulsar_gpu_act_slot_drop(const pulsar_gpu_tensor *x) {
    if (!x || !x->ptr) return;
    for (int i = 0; i < PULSAR_ACT_SLOTS; i++) {
        mxfp8_act_cache_t *s = &g_act_slots[i];
        if (s->key_ptr != x->ptr) continue;
        if (g_act_cur == s) g_act_cur = NULL;
        s->key_ptr = NULL; s->valid = 0; s->valid_b = 0; s->f32_absent = 0; s->lru = 0;
    }
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

/* Consumer lookup for the grouped slot, prefix-tolerant like act_slot_find_rows:
 * the producer armed the FULL batch, a mixed-step prefix asks for fewer rows.
 * The encoding is row-local per group (data [g][tok][gdim], scales
 * sfoff(tok, kb)), so rows [0, need) of a wider block are the same bytes --
 * provided the consumer strides groups by the SLOT's row count (x_tok_stride),
 * not its own. */
static mxfp8_gact_cache_t *gact_find_rows(const void *ptr, uint32_t need,
                                          uint32_t ngroups, uint64_t gdim) {
    if (!ptr || g_gact.key_ptr != ptr || g_gact.key_ntok < need ||
        g_gact.key_ngroups != ngroups || g_gact.key_gdim != gdim) {
        return NULL;
    }
    return &g_gact;
}

/* L158 inc 4: re-base a VEC32 scale slab to a row window.  The mixed-batch
 * prefix split hands its prefill SUFFIX (rows [row0, row0+rows) of the
 * producer's full-width encoding) to cuBLASLt, whose scale pointer must
 * describe a slab starting at row 0; pulsar_mx_sfoff(row, ...) is not a
 * pointer offset for row0 % 128 != 0, so the bytes are COPIED into a fresh
 * slab (rows x KBp bytes per group -- a re-layout, not a quantise; the values
 * are the producer's).  Groups: n_groups slabs of src_slab / dst_slab bytes. */
__global__ static void mx_scale_rebase_kernel(unsigned char *dst, const unsigned char *src,
                                              int row0, int rows, int KBp, int n_groups,
                                              size_t dst_slab, size_t src_slab) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long per_group = (long)rows * KBp;
    if (idx >= per_group * n_groups) return;
    const int g = (int)(idx / per_group);
    const int rem = (int)(idx % per_group);
    const int r = rem / KBp, kb = rem % KBp;
    dst[(size_t)g * dst_slab + pulsar_mx_sfoff(r, kb, KBp)] =
        src[(size_t)g * src_slab + pulsar_mx_sfoff(row0 + r, kb, KBp)];
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
    const int KBp = pulsar_mx_kbp((int)group_dim);
    const size_t slab = pulsar_mx_sf_slab_bytes((int)n_tokens, KBp);
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

/* L158 inc 5: a slot for an OFFSET VIEW of an encoded activation.  The mixed-
 * batch prefix split hands its prefill suffix to consumers as row views, and a
 * view pointer keys no slot -- the reason every suffix used to be re-quantised
 * from f32.  This gives the view its own slot filled from the producer's:
 * data by a byte copy of rows [row0, row0+rows), scales re-based into a fresh
 * VEC32 slab (a re-layout, not a quantise; the bytes are the producer's).
 * Consumers then find the view's encoding by their ordinary lookup.  Refuses
 * when the full activation carries no valid encoding. */
int pulsar_gpu_mxfp8_act_cache_window(const pulsar_gpu_tensor *x_full, uint64_t row0, uint64_t rows,
                                      uint64_t in_dim, const pulsar_gpu_tensor *x_view) {
    if (!x_full || !x_view || rows == 0 || in_dim % 32 != 0) return 0;
    mxfp8_act_cache_t *src = act_slot_find_rows(x_full->ptr, row0 + rows, in_dim);
    if (!src || !src->valid || !src->xq || !src->sx) return 0;
    const int KBp = pulsar_mx_kbp((int)in_dim);
    const size_t sx_bytes = pulsar_mx_sf_slab_bytes((int)rows, KBp);
    /* keep the source's buffers: acquire may evict an LRU slot, never the one
     * we are reading from (it was just touched by the lookup). */
    __nv_fp8_e4m3 *src_q = src->xq; unsigned char *src_sf = src->sx;
    mxfp8_act_cache_t *dst = act_slot_acquire(x_view->ptr, rows, in_dim);
    if (!dst || dst == src) return 0;
    if (!mxfp8_act_cache_reserve((void **)&dst->xq, &dst->xq_cap, rows * in_dim, "act window data") ||
        !mxfp8_act_cache_reserve((void **)&dst->sx, &dst->sx_cap, sx_bytes, "act window scale")) return 0;
    if (cudaMemcpyAsync(dst->xq, src_q + row0 * in_dim, rows * in_dim, cudaMemcpyDeviceToDevice, 0) != cudaSuccess) return 0;
    cudaMemsetAsync(dst->sx, 0, sx_bytes, 0);
    const long n_pairs = (long)rows * KBp;
    mx_scale_rebase_kernel<<<(unsigned)((n_pairs + 255) / 256), 256>>>(dst->sx, src_sf, (int)row0, (int)rows, KBp, 1,
                                                                        sx_bytes, sx_bytes);
    if (!cuda_ok(cudaGetLastError(), "act window rebase")) return 0;
    dst->valid = 1;
    return 1;
}

/* PRODUCER-side encode of an f32 activation the caller itself produced and
 * owns: tests and tools that synthesise x, and the engine producers whose
 * kernel has no epilogue (the MoE's SwiGLU output).  Epilogue producers emit
 * directly and do not call this.  Arms the slot, encodes with the one
 * standalone encoder, notes it. */
int pulsar_gpu_mxfp8_act_cache_encode_f32(const pulsar_gpu_tensor *x, uint64_t n_tok, uint64_t in_dim) {
    void *q = NULL, *sf = NULL; int kbp = 0;
    if (!x || n_tok == 0 || in_dim % 32 != 0) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float)) return 0;
    if (!pulsar_gpu_mxfp8_act_cache_e4m3_slot(x, n_tok, in_dim, &q, &sf, &kbp)) return 0;
    const int warps = (int)n_tok * (int)(in_dim / 32);
    mxfp8_quant_act_kernel<<<(warps * 32 + 255) / 256, 256>>>((const float *)x->ptr, (int)n_tok, (int)in_dim, kbp,
                                                             (__nv_fp8_e4m3 *)q, (unsigned char *)sf);
    if (!cuda_ok(cudaGetLastError(), "act encode f32")) return 0;
    pulsar_gpu_mxfp8_act_cache_arm(x, n_tok, in_dim);
    pulsar_gpu_mxfp8_act_cache_note_mxfp8();
    return 1;
}

/* L158 inc 4: PRODUCER-side grouped encode of the attention output for a
 * producer whose attention kernel has no E4M3 epilogue (today: the drafter's
 * raw batch attention).  Run right after the inverse rope tail, before the
 * attn-out projection; disarm with pulsar_gpu_mxfp8_gact_disarm after it. */
int pulsar_gpu_mxfp8_gact_emit_heads(const pulsar_gpu_tensor *heads, uint32_t n_tokens,
                                     uint32_t n_groups, uint64_t group_dim) {
    void *q = NULL, *sf = NULL; int kbp = 0; uint64_t slab = 0;
    if (!pulsar_gpu_mxfp8_gact_slot(heads, n_tokens, n_groups, group_dim, &q, &sf, &kbp, &slab)) return 0;
    const int warps = (int)n_tokens * (int)n_groups * (int)(group_dim / 32);
    mxfp8_quant_act_grouped_kernel<<<(warps * 32 + 255) / 256, 256>>>(
            (const pulsar_heads_t *)heads->ptr, (int)n_tokens, (int)n_groups, (int)group_dim, kbp,
            (__nv_fp8_e4m3 *)q, (unsigned char *)sf, (size_t)slab);
    if (!cuda_ok(cudaGetLastError(), "gact emit heads")) return 0;
    pulsar_gpu_mxfp8_gact_note();
    return 1;
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
    *kbp   = pulsar_mx_kbp((int)in_dim);
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



/* L158 inc 4: the tensor-core dense arm over a ROW WINDOW of full tensors --
 * rows [row0, row0 + n_tok) of x and out.  row0 = 0 is the ordinary call; the
 * mixed-batch prefix split uses row0 = n_dec for its prefill suffix, so the
 * suffix reads the producer's full-width E4M3 slot (data by pointer offset,
 * scales re-based by mx_scale_rebase_kernel) instead of quantising an offset
 * f32 view -- the last way an f32 activation reached this GEMM. */
static int cuda_matmul_fp8_mx_window(pulsar_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const pulsar_gpu_tensor *x,
        uint64_t row0, uint64_t n_tok, const char *label) {
    if (!out || !x || !model_map || in_dim % 32 != 0 || !cublaslt_ensure()) return 0;
    /* Derived from the destination, never passed in: the two cannot
     * disagree if only one of them exists. */
    const int out_f16 = (pulsar_tensor_esz(out) == sizeof(__half));

    uint64_t KB = in_dim / 32, weight_bytes = out_dim * KB * 33;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const size_t out_esz = out_f16 ? sizeof(__half) : sizeof(float);
    const uint64_t rows_total = row0 + n_tok;
    if (x->bytes < rows_total * in_dim * sizeof(float) || out->bytes < rows_total * out_dim * out_esz) return 0;
    const fp8_mx_weight *w = cuda_fp8_mx_weight(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
    if (!w) return 0;
    int ntok = (int)n_tok, KBp = mx_rup((int)KB, 4);
    size_t sx_bytes = pulsar_mx_sf_slab_bytes(ntok, KBp);
    size_t wz = 32u << 20;
    /* xq, sx and the cuBLASLt workspace must be DISTINCT buffers, but
     * cuda_tmp_alloc hands out one shared scratch region (later calls alias or
     * realloc/free earlier ones), so these three come from one arena. */
    /* Armed activation cache (see mxfp8_act_cache_t above): reuse the E4M3 data
     * and E8M0 scales this activation was already quantized into, and take only
     * the cuBLASLt workspace from the shared tmp region. */
    /* The activation MUST already carry its E4M3 encoding: the producer emitted
     * it (norm epilogues, the attention epilogue, the drafter's producers) and
     * armed the slot for the full batch.  One format or an error. */
    mxfp8_act_cache_t *ac = act_slot_find_rows(x->ptr, rows_total, in_dim);
    if (!ac || !ac->valid || !ac->xq || !ac->sx)
        return act_a8_missing_fail(label ? label : "tensor-core GEMM", n_tok, in_dim, out_dim);
    __nv_fp8_e4m3 *xq = ac->xq + row0 * in_dim;
    unsigned char *sx;
    void *ws;
    cuda_arena ar;
    if (row0 == 0) {
        if (!cuda_arena_begin(&ar, wz, "fp8_mx scratch")) return 0;
        sx = ac->sx;
    } else {
        if (!cuda_arena_begin(&ar, mx_a256(sx_bytes) + wz, "fp8_mx scratch")) return 0;
        sx = (unsigned char *)cuda_arena_take(&ar, sx_bytes, 256);
        if (!sx) return 0;
        cudaMemsetAsync(sx, 0, sx_bytes, 0);
        const long n_pairs = (long)ntok * KBp;
        mx_scale_rebase_kernel<<<(unsigned)((n_pairs + 255) / 256), 256>>>(sx, ac->sx, (int)row0, ntok, KBp, 1,
                                                                            sx_bytes, sx_bytes /* unused for 1 group */);
        if (!cuda_ok(cudaGetLastError(), "fp8_mx scale rebase")) return 0;
    }
    ws = cuda_arena_take(&ar, wz, 256);
    if (!ws) return 0;
    void *optr = (char *)out->ptr + row0 * out_dim * out_esz;
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
                                           optr, e->ld, optr, e->ld, &e->h.algo, ws, wz,
                                           cudaStreamPerThread);
        ok = (st == CUBLAS_STATUS_SUCCESS);
        if (!ok) fprintf(stderr, "pulsar: cuBLASLt MXFP8 matmul failed: status %d\n", (int)st);
    }
    return ok;
}

static int cuda_matmul_fp8_mx_tensor_labeled(pulsar_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const pulsar_gpu_tensor *x,
        uint64_t n_tok, const char *label) {
    return cuda_matmul_fp8_mx_window(out, model_map, model_size, weight_offset, in_dim, out_dim, x, 0, n_tok, label);
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
        uint32_t n_tokens,
        uint32_t row0) {   ///< L158 inc 4: rows [row0, row0+n_tokens) of a FULL-width heads encoding
    if (group_dim % 32 != 0 || rank % 128 != 0 || !cublaslt_ensure()) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t KB = group_dim / 32;
    const uint64_t weight_bytes = low_dim * KB * 33;
    if (out_a_offset > model_size || weight_bytes > model_size - out_a_offset) return 0;
    const fp8_mx_weight *w = cuda_fp8_mx_weight(model_map, out_a_offset, weight_bytes,
                                                group_dim, low_dim, "attn_out_a");
    if (!w) return 0;
    const int KBp = mx_rup((int)KB, 4);
    size_t x_scale_slab = pulsar_mx_sf_slab_bytes((int)n_tokens, KBp);
    const size_t w_scale_slab = ((size_t)rank / 128) * (size_t)KBp * 128;
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
    mxfp8_gact_cache_t *gc = gact_find_rows(heads->ptr, row0 + n_tokens, n_groups, group_dim);
    if (!gc || !gc->valid || gc->kbp != KBp) {
        static int said_ga = 0;
        if (!said_ga) {
            said_ga = 1;
            fprintf(stderr, "pulsar: NO grouped E4M3 ENCODING for attn-out 'a' (tensor-core arm, "
                            "n_tokens=%u row0=%u) -- the attention producer did not emit it; the "
                            "f32 quantise fallback was deleted (L158).  Refusing.\n", n_tokens, row0);
        }
        return 0;
    }
    /* group-major encoding: group g's rows start at g * key_ntok (the SLOT's
     * row count, not this call's) -- see gact_find_rows. */
    const uint64_t x_gstride = (uint64_t)gc->key_ntok * group_dim;
    xq = gc->xq + (size_t)row0 * group_dim;
    if (row0 == 0) {
        x_scale_slab = gc->scale_slab;
        sx = gc->sx;
        if (!cuda_arena_begin(&ar, wz, "attn_out_a mx scratch")) return 0;
        ws = cuda_arena_take(&ar, wz, 256);
        if (!ws) return 0;
    } else {
        if (!cuda_arena_begin(&ar, mx_a256(scale_bytes) + wz, "attn_out_a mx scratch")) return 0;
        sx = (unsigned char *)cuda_arena_take(&ar, scale_bytes, 256);
        ws = cuda_arena_take(&ar, wz, 256);
        if (!ws) return 0;
        cudaMemsetAsync(sx, 0, scale_bytes, 0);
        const long n_pairs = (long)n_tokens * KBp * n_groups;
        mx_scale_rebase_kernel<<<(unsigned)((n_pairs + 255) / 256), 256>>>(sx, gc->sx, (int)row0, (int)n_tokens, KBp,
                                                                            (int)n_groups, x_scale_slab, gc->scale_slab);
        if (!cuda_ok(cudaGetLastError(), "attn_out_a scale rebase")) return 0;
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
            const __nv_fp8_e4m3 *bg = xq + (size_t)g * x_gstride;
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
 * qw[j] * qa[j] * s, with s = sw*sa hoisted, NOT (qw*sw)*(qa*sa) -- matching
 * the arm it must agree with is worth more than a saved multiply.  (The f32 NT
 * twin that hoisted wj = w*sc was deleted 2026-09-03, L158.)
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
    for (uint32_t i = 0; i < FP8_FC; i++) {
        g_fp8_fc[i].ptr.store(NULL, std::memory_order_relaxed);
        g_fp8_fc[i].tag.store(0, std::memory_order_release);
    }
    /* per-load registrations; the next engine open re-registers its own set */
    g_fp8_offsets.clear();
    g_mxfp8_lt_offsets.clear();
    /* L191: the F32-source -> bf16 copies were never cleared -- a second engine
     * open in one process served the first model's converted weights. */
    for (auto &kv : g_f32w_bf16) (void)cudaFree(kv.second);
    g_f32w_bf16.clear();
    for (uint32_t i = 0; i < F32W_FC; i++) { g_f32w_fc[i].key = f32w_key{}; g_f32w_fc[i].ptr = NULL; }
}



/* ROW KIND CHOOSES THE ARM; ROW COUNT NEVER DOES (L167).
 *
 * g_batch_decode_rows is the number of leading DECODE rows in the batch being
 * encoded -- a fact about the rows, declared by the lane that owns them
 * (pulsar_gpu_matmul_set_batch_decode_rows; the batched step, the classic
 * verify block, the drafter's forwards and seeds, the one-row output head).
 * Every dense dispatcher in this file and every MoE tier in
 * pulsar_cuda_moe.cu reads it the same way:
 *   - decode rows (0 < n_dec, n_dec >= n_tok): the M-INDEPENDENT arms -- the
 *     one-row GEMV at n_tok == 1, the nt kernels at 2..PULSAR_GPU_MNEUTRAL_
 *     ROWS_MAX (one weight read serving every row, each output row computed
 *     from its own activation alone, bit-identical to the one-row kernel).
 *     A decode row's bytes therefore depend on neither its batchmates nor the
 *     batch width.  The setter refuses a count past the cap, so a decode call
 *     is never wider than the nt instantiations.
 *   - prefill rows (n_dec == 0): the TENSOR-CORE arms -- cuBLAS(Lt), the
 *     grouped CUTLASS GEMMs -- at ANY n_tok, one row included.  This is the
 *     arm the B300 reference computes prefill rows with (the reference gate
 *     graded a 6-row remainder chunk moved onto nt as further from source).
 *   - mixed (0 < n_dec < n_tok): the batch is laid out [decode rows 0..n_dec)
 *     then one prefill run [n_dec..n_tok); each dispatcher splits there and
 *     recurses with n_dec on the prefix and 0 on the suffix (the suffix reads
 *     the producer's full-width slot through a row window).
 * Until L167 the arm was chosen by ROW COUNT with a global mode flag as a
 * tie-break (nt to 4 rows unarmed, to 16 armed, tensor-core above): a proxy
 * for kind that put 5..16-row prefill remainders on one arm or the other by
 * the caller's mode, and a 1-row prefill chunk on the decode GEMV.  The
 * prefill byte gate's depth 4102 (= 4096 + 6, a 6-row remainder) is the
 * depth that moves whenever a prefill row leaves the tensor-core arm.
 * Set once at the lane's entry, restored on exit -- never on a per-token
 * path (pulsar_decode_rows_scope in the engine). */
/* thread_local (L191): every dispatcher below reads this to choose the
 * arithmetic arm; a second submitting thread setting its own lane's row count
 * must not flip a concurrent decode onto the tensor-core arm.  The engine sets
 * it on the lane's entry thread and dispatches from that thread
 * (pulsar_decode_rows_scope). */
static thread_local int g_batch_decode_rows = 0;
int pulsar_gpu_matmul_set_batch_decode_rows(int n) {
    /* Refuse, do not warn: decode rows past the cap would take a
     * batch-shape-dependent GEMM.  The engine static_asserts PULSAR_MSEQ_MAX
     * against the cap, so the batched lane cannot reach this; a probe can. */
    if (n > (int)PULSAR_GPU_MNEUTRAL_ROWS_MAX) {
        fprintf(stderr, "pulsar: batched step declares %d decode rows; the row cap is %u -- "
                        "refusing (rows past the cap would take a batch-shape-dependent GEMM)\n",
                        n, (unsigned)PULSAR_GPU_MNEUTRAL_ROWS_MAX);
        return 0;
    }
    g_batch_decode_rows = (n > 0) ? n : 0;
    return 1;
}
/* Read cross-TU by the MoE dispatch (pulsar_cuda_moe.cu) to place its split,
 * and by the prefill encoder's f32-store skips (the split's offset views key no
 * slot, so the skips apply only when no decode prefix is in flight). */
int pulsar_gpu_matmul_batch_decode_rows(void) { return g_batch_decode_rows; }


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
    if (g_batch_decode_rows > 0 && (uint64_t)g_batch_decode_rows < n_tok) {
        for (int i = 0; i < PULSAR_ACT_SLOTS; i++) {
            if (g_act_slots[i].key_ptr == x->ptr && g_act_slots[i].f32_absent) {
                fprintf(stderr, "pulsar: mxfp8 '%s' mixed-batch split (n_dec=%d of %llu) on a "
                                "buffer whose f32 store was SKIPPED -- refusing; the split "
                                "halves cannot reach the E4M3 cache.\n",
                        label ? label : "?", g_batch_decode_rows, (unsigned long long)n_tok);
                return 0;
            }
        }
    }
    /* inc 4 prefix-split: 0<n_dec<n_tok => mixed decode+prefill batch. Run the
     * decode prefix [0,n_dec) in the M-independent (decode) regime and the prefill
     * suffix [n_dec,n_tok) in the tensor-core (prefill) regime, by recursing with
     * the flag set to each range's pure value. Offsets are row-major (float rows). */
    {
        const uint64_t n_dec = (uint64_t)g_batch_decode_rows;
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
            const int saved = g_batch_decode_rows;
            g_batch_decode_rows = (int)n_dec;  ///< decode prefix: n_dec == n_tok' => no further split
            int r1 = cuda_matmul_mxfp8_tensor_labeled(&out_pre, model_map, model_size,
                    weight_offset, in_dim, out_dim, &x_pre, n_dec, label);
            g_batch_decode_rows = 0;
            /* The suffix is the step's prefill run: tensor-core window arm at
             * any width (GATE 2 grades its rows for correctness).  L158 inc 4:
             * the window reads the producer's full-width slot; an offset f32
             * view keyed no slot and used to be quantised here -- that path is
             * gone. */
            (void)out_suf; (void)x_suf;
            int r2 = cuda_matmul_fp8_mx_window(out, model_map, model_size,
                    weight_offset, in_dim, out_dim, x, n_dec, n_tok - n_dec, label);
            g_batch_decode_rows = saved;
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
        /* Row kind (see the header at g_batch_decode_rows).  The split above
         * has already run, so n_dec is 0 (prefill rows) or >= n_tok (decode). */
        const int decode_kind = g_batch_decode_rows > 0;
        if (!decode_kind) {
            /* PREFILL rows, any n_tok (one included): the cuBLASLt MX
             * tensor-core GEMM, and only that -- a failure inside it (handle,
             * workspace arena, heuristic, the matmul itself) is refused here.
             * Until L164 it fell to the per-token GEMV; until L167 a 1-row
             * prefill chunk took the GEMV by row count. */
            if (cuda_matmul_fp8_mx_tensor_labeled(out, model_map, model_size,
                    weight_offset, in_dim, out_dim, x, n_tok, label)) return 1;
            fprintf(stderr, "pulsar: cuBLASLt MX GEMM failed for %s (prefill rows, n_tok=%llu "
                            "in_dim=%llu out_dim=%llu) -- refusing\n",
                    label ? label : "weights", (unsigned long long)n_tok,
                    (unsigned long long)in_dim, (unsigned long long)out_dim);
            return 0;
        }
        /* DECODE rows, n_tok 2..cap (spec-verify batches, drafter forwards):
         * batched GEMV over the de-interleaved weight.  One weight-row read
         * serves every row; each row's result is bit-identical to the n == 1
         * kernel's below, so a row's bytes do not depend on the batch width.
         * (Until L158 an f32-activation twin of each kernel sat behind this
         * arm for the no-slot case; the no-slot case refuses.) */
        if (n_tok >= 2) {
            if (in_dim % 128 != 0) {
                fprintf(stderr, "pulsar: mxfp8 '%s' decode GEMV needs in_dim %% 128 == 0 (in_dim=%llu) "
                                "-- refusing\n", label ? label : "?", (unsigned long long)in_dim);
                return 0;
            }
            const fp8_mx_weight *bw = cuda_fp8_mx_weight(model_map, weight_offset, fbytes,
                                                         in_dim, out_dim, label);
            if (bw) {
                const int KBp = pulsar_mx_kbp((int)in_dim);
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
                    const int xKBp = pulsar_mx_kbp((int)in_dim);
                    {
                        /* Keyed on the (in,out) pair and announced once per
                         * shape, exactly like the n==1 twin: a gate PASS cannot
                         * tell "the A8 arm ran" from "the A8 arm never ran", and
                         * a silently-missed cache would look identical. */
                        static pulsar_shape_once seen_nt = {};
                        if (pulsar_shape_once_first(&seen_nt, pulsar_shape_key(in_dim, out_dim),
                                                    "verify-batch GEMV W8A8 announce")) {
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
                /* L158: no E4M3 encoding and none declared -- refuse (the f32
                 * fallback that stood here is gone; see act_a8_missing_fail). */
                return act_a8_missing_fail("verify-batch GEMV", n_tok, in_dim, out_dim);
            }
            fprintf(stderr, "pulsar: mxfp8 '%s' decode GEMV: de-interleaved weight did not resolve "
                            "(in_dim=%llu out_dim=%llu) -- refusing\n",
                    label ? label : "?", (unsigned long long)in_dim, (unsigned long long)out_dim);
            return 0;
        }
        /* DECODE row, n_tok == 1: the one-row GEMV. */
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
            const int KBp = pulsar_mx_kbp((int)in_dim);
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
                    static pulsar_shape_once seen = {};
                    if (pulsar_shape_once_first(&seen, pulsar_shape_key(in_dim, out_dim),
                                                "decode GEMV W8A8 announce")) {
                        fprintf(stderr, "pulsar: decode GEMV W8A8 (E4M3 acts) for "
                                        "in_dim=%llu out_dim=%llu\n",
                                (unsigned long long)in_dim, (unsigned long long)out_dim);
                    }
                }
                const int xKBp = pulsar_mx_kbp((int)in_dim);
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
            /* L158: no E4M3 encoding and none declared -- refuse (the per-token
             * f32 GEMV that stood here is gone; see act_a8_missing_fail). */
            return act_a8_missing_fail("decode GEMV", n_tok, in_dim, out_dim);
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










/* L079/L087: the bf16 compute core, weight given as a resolved DEVICE pointer.
 * Two producers feed it: pulsar_gpu_matmul_bf16_tensor (native BF16 storage,
 * pointer straight into the mmap) and pulsar_gpu_matmul_f32_tensor (F32-source
 * storage, pointer into the once-converted bf16 copy -- see
 * f32_weight_bf16_resolve).  Everything numeric lives HERE, so the two weight
 * families cannot drift: same activation cache, same NT/cuBLASLt/GEMV arms,
 * same M-independence contract. */
/* L183: the prefill arm of the plain-weight family (router, compressor kv/gate,
 * HC mix, output head -- bf16 source, or F32 source through its once-converted
 * bf16 copy) runs cuBLASLt with ONE kernel per weight shape, chosen once at a
 * canonical row count and used at every row count.  Two things make the bytes
 * a row gets independent of how many rows share the call, and both are needed:
 *   - the reduction scheme is pinned to NONE (split-K reduces partial sums in
 *     an order that depends on the split), the MXFP8 arm's contract;
 *   - the ALGORITHM is pinned per (in_dim, out_dim): two non-split-K kernels
 *     still accumulate a 16384-long K differently (different MMA shapes and
 *     k-tiles), and cuBLASLt's heuristic picks by M -- measured on the HC mix
 *     (K 16384, N 24): 4091 rows and 4097 rows got different kernels and
 *     different bytes with the mask alone.
 * cublasGemmEx(CUBLAS_GEMM_DEFAULT) sat here until 2026-09-05, free to pick
 * split-K for these tall-skinny shapes below ~4095 rows -- and did: a token got
 * different bytes in a 6-row and a 4097-row chunk (L183 census), which is what
 * made a prompt's logits depend on its chunking (L180) and a warm-fork resume a
 * different computation from a cold prefill.  The canonical row count is the
 * production prefill chunk: the kernel that runs the 4096-row chunk runs the
 * remainder chunk and the frontier row too; cublasLtMatmulAlgoCheck confirms it
 * supports each row count before use, and an unsupported one refuses.  The
 * per-(shape, ntok) entries below cache the layouts and that check. */
#define PULSAR_BF16_LT_CANON_ROWS 4096
static int bf16_lt_matmul(void *out, const uint16_t *w, const uint16_t *xb,
                          uint64_t in_dim, uint64_t out_dim, uint64_t n_tok) {
    if (!cublaslt_ensure()) {
        fprintf(stderr, "pulsar: bf16 GEMM on prefill rows: cuBLASLt handle not ready -- refusing\n");
        return 0;
    }
    const size_t wz = 32u << 20;
    /* ONE algorithm per weight shape, chosen at the canonical row count. */
    struct bf16_lt_shape {
        uint64_t in_dim, out_dim; int valid;
        cublasLtMatmulDesc_t op;
        cublasLtMatrixLayout_t la;
        cublasLtMatmulAlgo_t algo;
    };
    static thread_local bf16_lt_shape shapes[16];
    static thread_local int shapes_next;
    bf16_lt_shape *sh = NULL;
    for (int i = 0; i < 16; i++)
        if (shapes[i].valid && shapes[i].in_dim == in_dim && shapes[i].out_dim == out_dim) { sh = &shapes[i]; break; }
    if (!sh) {
        bf16_lt_shape ns = {};
        ns.in_dim = in_dim; ns.out_dim = out_dim;
        if (cublasLtMatmulDescCreate(&ns.op, CUBLAS_COMPUTE_32F, CUDA_R_32F)) return 0;
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(ns.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
        cublasLtMatmulDescSetAttribute(ns.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
        /* the geometry cublasGemmEx had: D(out x ntok) = W^T(out x in) . X(in x ntok) */
        cublasLtMatrixLayoutCreate(&ns.la, CUDA_R_16BF, in_dim, out_dim, in_dim);
        cublasLtMatrixLayout_t lb_c, ld_c;
        cublasLtMatrixLayoutCreate(&lb_c, CUDA_R_16BF, in_dim, PULSAR_BF16_LT_CANON_ROWS, in_dim);
        cublasLtMatrixLayoutCreate(&ld_c, CUDA_R_32F, out_dim, PULSAR_BF16_LT_CANON_ROWS, out_dim);
        cublasLtMatmulPreference_t pf; cublasLtMatmulPreferenceCreate(&pf);
        cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wz, sizeof(wz));
        {
            uint32_t red = CUBLASLT_REDUCTION_SCHEME_NONE;
            cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &red, sizeof(red));
        }
        cublasLtMatmulHeuristicResult_t hr; int got = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(g_cublaslt, ns.op, ns.la, lb_c, ld_c, ld_c, pf, 1, &hr, &got);
        cublasLtMatmulPreferenceDestroy(pf);
        cublasLtMatrixLayoutDestroy(lb_c); cublasLtMatrixLayoutDestroy(ld_c);
        if (hs != CUBLAS_STATUS_SUCCESS || !got) {
            cublasLtMatrixLayoutDestroy(ns.la); cublasLtMatmulDescDestroy(ns.op);
            fprintf(stderr, "pulsar: bf16 GEMM (in_dim=%llu out_dim=%llu): no fixed-order cuBLASLt kernel at %d rows -- refusing\n",
                    (unsigned long long)in_dim, (unsigned long long)out_dim, PULSAR_BF16_LT_CANON_ROWS);
            return 0;
        }
        ns.algo = hr.algo;
        ns.valid = 1;
        sh = &shapes[shapes_next];
        shapes_next = (shapes_next + 1) & 15;
        if (sh->valid) { cublasLtMatrixLayoutDestroy(sh->la); cublasLtMatmulDescDestroy(sh->op); }
        *sh = ns;
    }
    /* per (shape, ntok): the row-count layouts and the check that the shape's
     * kernel supports this row count */
    struct bf16_lt_call {
        uint64_t in_dim, out_dim; int ntok; int valid;
        cublasLtMatrixLayout_t lb, ld;
    };
    static thread_local bf16_lt_call calls[32];
    static thread_local int calls_next;
    const int ntok = (int)n_tok;
    bf16_lt_call *e = NULL;
    for (int i = 0; i < 32; i++)
        if (calls[i].valid && calls[i].in_dim == in_dim && calls[i].out_dim == out_dim && calls[i].ntok == ntok) { e = &calls[i]; break; }
    if (!e) {
        bf16_lt_call ne = {};
        ne.in_dim = in_dim; ne.out_dim = out_dim; ne.ntok = ntok;
        cublasLtMatrixLayoutCreate(&ne.lb, CUDA_R_16BF, in_dim, ntok, in_dim);
        cublasLtMatrixLayoutCreate(&ne.ld, CUDA_R_32F, out_dim, ntok, out_dim);
        cublasLtMatmulHeuristicResult_t chk;
        cublasStatus_t cs = cublasLtMatmulAlgoCheck(g_cublaslt, sh->op, sh->la, ne.lb, ne.ld, ne.ld, &sh->algo, &chk);
        if (cs != CUBLAS_STATUS_SUCCESS || chk.state != CUBLAS_STATUS_SUCCESS || chk.workspaceSize > wz) {
            cublasLtMatrixLayoutDestroy(ne.lb); cublasLtMatrixLayoutDestroy(ne.ld);
            fprintf(stderr, "pulsar: bf16 GEMM (in_dim=%llu out_dim=%llu): the shape's fixed kernel (chosen at %d rows) "
                            "does not support %d rows -- refusing\n",
                    (unsigned long long)in_dim, (unsigned long long)out_dim, PULSAR_BF16_LT_CANON_ROWS, ntok);
            return 0;
        }
        ne.valid = 1;
        e = &calls[calls_next];
        calls_next = (calls_next + 1) & 31;
        if (e->valid) { cublasLtMatrixLayoutDestroy(e->lb); cublasLtMatrixLayoutDestroy(e->ld); }
        *e = ne;
    }
    cuda_arena ar;
    if (!cuda_arena_begin(&ar, wz, "bf16 lt scratch")) return 0;
    void *ws = cuda_arena_take(&ar, wz, 256);
    if (!ws) return 0;
    const float al = 1.0f, be = 0.0f;
    cublasStatus_t st = cublasLtMatmul(g_cublaslt, sh->op, &al, w, sh->la, xb, e->lb, &be,
                                       out, e->ld, out, e->ld, &sh->algo, ws, wz, cudaStreamPerThread);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "pulsar: cuBLASLt bf16 matmul failed: status %d\n", (int)st);
        return 0;
    }
    return 1;
}

static int matmul_bf16_wptr(pulsar_gpu_tensor *out, const uint16_t *w,
                            uint64_t in_dim, uint64_t out_dim,
                            const pulsar_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !w) return 0;
    /* inc 4 prefix-split (see the f16/mxfp8 twins): decode prefix [0,n_dec)
     * M-independent, prefill suffix [n_dec,n_tok) tensor-core, via pure-regime
     * recursion. */
    {
        const uint64_t n_dec = (uint64_t)g_batch_decode_rows;
        if (n_dec > 0 && n_dec < n_tok) {
            const uint64_t inb = in_dim * sizeof(float), outb = out_dim * sizeof(float);
            pulsar_gpu_tensor out_pre = pulsar_tensor_subview(out, 0, out->bytes);
            pulsar_gpu_tensor x_pre   = pulsar_tensor_subview(x, 0, x->bytes);
            pulsar_gpu_tensor out_suf = pulsar_tensor_subview(out, n_dec * outb,
                                                             out->bytes - n_dec * outb);
            pulsar_gpu_tensor x_suf   = pulsar_tensor_subview(x, n_dec * inb,
                                                             x->bytes - n_dec * inb);
            const int saved = g_batch_decode_rows;
            g_batch_decode_rows = (int)n_dec;
            int r1 = matmul_bf16_wptr(&out_pre, w, in_dim, out_dim, &x_pre, n_dec);
            g_batch_decode_rows = 0;
            int r2 = matmul_bf16_wptr(&out_suf, w, in_dim, out_dim, &x_suf, n_tok - n_dec);
            g_batch_decode_rows = saved;
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

    /* ONE bf16 activation for the whole call, READ before the arm is chosen.
     * Every arm below reads these exact bytes, so "which kernel ran" cannot
     * change the activation operand -- the failure mode this file hit three
     * times (the cuBLAS-rounds/nt-does-not split, the MoE n==1 q8_1 split, the
     * A8 cache key).  L159: the bytes come from the PRODUCER's epilogue
     * (pulsar_gpu_bf16_act_slot / note), through any row window of its plane.
     * The convert-on-miss that used to sit here (f32_to_bf16_kernel over
     * x->ptr, exact-key cached) was the last consumer-side activation
     * conversion in the engine and ran on the served lane for the output head
     * at every verify width and for the prefill warmup; it is deleted.  A miss
     * is a producer that did not emit, and that is an error, once, loudly. */
    uint64_t row0 = 0;
    const mxfp8_act_cache_t *hw = act_slot_find_window(x->ptr, n_tok, in_dim, &row0);
    if (!hw || !hw->valid_b || !hw->xb) {
        static int said = 0;
        if (!said) {
            said = 1;
            fprintf(stderr, "pulsar: NO bf16 ENCODING for the activation of a bf16-weight GEMM "
                            "(n_tok=%llu in_dim=%llu) -- no producer emitted it; the f32->bf16 "
                            "convert was deleted (L159).  Refusing.\n",
                    (unsigned long long)n_tok, (unsigned long long)in_dim);
        }
        return 0;
    }
    const __nv_bfloat16 *xb16 = hw->xb + row0 * in_dim;
    /* M-independence, same contract as the f16 and f32 arms.  All three arms now
     * read the SAME bf16 activation plane read above, so they cannot disagree
     * on the operand at all and the only cross-arm difference left is
     * accumulation ORDER.  This comment read "the cuBLAS path below additionally
     * ROUNDS the activations to bf16, so its disagreement with the n=1 kernel is
     * larger than the f32 arm's, not smaller" until 2026-08-17, which was true
     * and was the bug. */
    {
        /* Row kind (see the header at g_batch_decode_rows): prefill rows take
         * the fixed-order cuBLASLt arm at any n_tok, one included (L183);
         * decode rows take the nt kernels at 2..cap and the one-row kernel
         * below at 1. */
        const int decode_kind = g_batch_decode_rows > 0;
        if (!decode_kind) {
            return bf16_lt_matmul(out->ptr, w, (const uint16_t *)xb16, in_dim, out_dim, n_tok);
        }
        if (n_tok >= 2) {
            #define PULSAR_NT_LAUNCH_R(N, RR) matmul_nt_kernel<N, RR, __nv_bfloat16, __nv_bfloat16><<<g, 256>>>( \
                    (float *)out->ptr, (const __nv_bfloat16 *)w,                        \
                    xb16, in_dim, out_dim)
            #define PULSAR_NT_LAUNCH(N) do { const int R = nt_rows_per_block(N, out_dim); \
                                             dim3 g(((unsigned)out_dim + (unsigned)R - 1u) / (unsigned)R); \
                                             if (R == 4) PULSAR_NT_LAUNCH_R(N, 4); \
                                             else if (R == 2) PULSAR_NT_LAUNCH_R(N, 2); \
                                             else PULSAR_NT_LAUNCH_R(N, 1); } while (0)
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
            #undef PULSAR_NT_LAUNCH_R
            return cuda_ok(cudaGetLastError(), "matmul_bf16 nt launch");
        }
    }
    /* DECODE row, n_tok == 1: the one-row kernel.  (A cuBLAS handle that was
     * not ready used to land any n_tok here -- a fallback; prefill rows now
     * refuse instead.) */
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
 * The copy is converted ONCE per weight (immutable) and lives beside the mmap:
 * the full F32 family is ~130 MiB of f32, so ~65 MiB of bf16 copies.
 *
 * L191: keyed by (model_map, offset, in_dim, out_dim), not offset alone -- the
 * same offset in a second model, or the same bytes read at another shape, must
 * not be served the first conversion -- and cleared with the other weight
 * caches (cuda_fp8_weight_cache_clear), which it never was.  Process-global
 * like g_fp8_mx_by_offset, for the same reason (see the note there). */
static const uint16_t *f32_weight_bf16_resolve(const void *model_map,
                                               uint64_t model_size,
                                               uint64_t offset,
                                               uint64_t in_dim, uint64_t out_dim) {
    const f32w_key key{model_map, offset, in_dim, out_dim};
    f32w_fc_entry *fc = &g_f32w_fc[(uint32_t)(offset & (F32W_FC - 1u))];
    if (fc->ptr && fc->key == key) return fc->ptr;
    auto it = g_f32w_bf16.find(key);
    if (it != g_f32w_bf16.end()) {
        fc->key = key; fc->ptr = it->second;
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
    if (!cuda_ok(cudaGetLastError(), "f32->bf16 weight convert launch")) { (void)cudaFree(dst); return NULL; }
    g_f32w_bf16[key] = dst;
    fc->key = key; fc->ptr = dst;
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
/* L141: one A8 launch for every decode-row count the setter admits.  n == 1
 * keeps the one-token kernel; 2..PULSAR_GPU_MNEUTRAL_ROWS_MAX stage each
 * weight block once for all rows.  Prefill rows take the tensor-core 'a' arm
 * and never reach here; a wider call is refused (the per-token grid that used
 * to catch it was a fallback nobody measured). */
static_assert(PULSAR_GPU_MNEUTRAL_ROWS_MAX == 16u,
              "grouped_fp8mx_a_a8_rows enumerates NT up to the row cap");
static int launch_grouped_fp8mx_a_a8_rows(float *low, const fp8_mx_weight *dw, int KBp,
        const __nv_fp8_e4m3 *xq, const unsigned char *sx, int xKBp, uint64_t slab,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint32_t n_tokens,
        uint64_t blocks, uint64_t low_dim, const char *what, uint32_t x_tok_stride) {
    const unsigned gx = ((unsigned)low_dim + PULSAR_FP8MX_ROWS_A8_PER_BLOCK - 1u)
                        / PULSAR_FP8MX_ROWS_A8_PER_BLOCK;
    #define PULSAR_OA_NT(N) grouped_fp8mx_a_nt_a8_kernel<N><<<dim3(gx, 1, 1), 256>>>( \
            low, dw->data, dw->scale, KBp, xq, sx, xKBp, slab,                        \
            group_dim, rank, n_groups, n_tokens, x_tok_stride, blocks)
    switch (n_tokens) {
    case 1:
        grouped_fp8mx_a_warp8_a8_kernel<<<dim3(gx, 1, 1), 256>>>(
                low, dw->data, dw->scale, KBp, xq, sx, xKBp, slab,
                group_dim, rank, n_groups, n_tokens, x_tok_stride, blocks);
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
        #undef PULSAR_OA_NT
        fprintf(stderr, "pulsar: %s: n_tokens=%u is above the row cap %u -- refusing "
                        "(the tensor-core 'a' arm owns this width)\n",
                what, n_tokens, (unsigned)PULSAR_GPU_MNEUTRAL_ROWS_MAX);
        return 0;
    }
    return cuda_ok(cudaGetLastError(), what);
}

static int launch_grouped_fp8mx_a(float *low, const void *model_map, uint64_t out_a_offset,
        uint64_t out_a_bytes, uint64_t group_dim, uint64_t rank,
        uint32_t n_groups, uint32_t n_tokens, uint64_t blocks_a, uint64_t low_dim,
        const pulsar_heads_t *heads, const char *label) {
    const fp8_mx_weight *dw = (group_dim % 32 == 0)
            ? cuda_fp8_mx_weight(model_map, out_a_offset, out_a_bytes, group_dim, low_dim, label) : NULL;
    const int KBp = pulsar_mx_kbp((int)group_dim);

    /* W8A8 attn-output "a" projection: both operands E4M3 with their own ue8m0
     * block scales.  The activation is the grouped encoding the attention
     * producer emitted (its epilogue, or pulsar_gpu_mxfp8_gact_emit_heads for
     * a producer without one), read from the grouped cache for EVERY n this
     * function serves -- one activation format across row counts, no size
     * threshold.  n >= 2 stages each weight block once for all rows
     * (grouped_fp8mx_a_nt_a8_kernel). */
    if (dw &&   /* dw != NULL implies group_dim%32==0 (K13) */
        rank % PULSAR_FP8MX_ROWS == 0 && low_dim % PULSAR_FP8MX_ROWS == 0) {
        mxfp8_gact_cache_t *gc = gact_find_rows(heads, n_tokens, n_groups, group_dim);
        if (gc && gc->valid && gc->kbp == KBp) {
            static int announced_gact_gemv = 0;
            if (!announced_gact_gemv) {
                announced_gact_gemv = 1;
                fprintf(stderr, "pulsar: attn-out 'a' GEMV = producer-emitted E4M3\n");
            }
            return launch_grouped_fp8mx_a_a8_rows(low, dw, KBp,
                    gc->xq, gc->sx, gc->kbp, (uint64_t)gc->scale_slab,
                    group_dim, rank, n_groups, n_tokens, blocks_a, low_dim,
                    "attention_output_a a8 launch (gact)", gc->key_ntok);
        }
        /* No producer encoding -> refuse; there is no quantiser here. */
        static int said_gg = 0;
        if (!said_gg) {
            said_gg = 1;
            fprintf(stderr, "pulsar: NO grouped E4M3 ENCODING for attn-out 'a' (GEMV arm, n_tokens=%u "
                            "group_dim=%llu) -- the attention producer did not emit it "
                            "(pulsar_gpu_mxfp8_gact_emit_heads for producers without an epilogue).  "
                            "Refusing.\n", n_tokens, (unsigned long long)group_dim);
        }
        return 0;
    }
    if (!dw) {
        fprintf(stderr, "pulsar: attn-out 'a' weight did not resolve (deint) -- refusing "
                        "(group_dim=%llu rank=%llu)\n",
                (unsigned long long)group_dim, (unsigned long long)rank);
        return 0;
    }
    fprintf(stderr, "pulsar: attn-out 'a' shape (rank=%llu low_dim=%llu) not a multiple of %d -- refusing\n",
            (unsigned long long)rank, (unsigned long long)low_dim, (int)PULSAR_FP8MX_ROWS);
    return 0;
}


/* The "a" projection's output `low` is the "b" projection's activation; emit
 * its E4M3 encoding here, at the producer stage, so "b" reads a slot.  Done as
 * a pass after "a" rather than in "a"'s epilogue because the warp that reduces
 * a row does not hold that row's block neighbours.  A slot failure is an ERROR
 * now (L158): "b" has no f32 arm to fall back to. */
static int emit_low_e4m3(pulsar_gpu_tensor *low, uint32_t n_tokens, uint64_t low_dim) {
    if (low_dim % 256 != 0) {
        fprintf(stderr, "pulsar: attn-out low_dim=%llu cannot carry an E4M3 encoding (needs a multiple of 256) -- refusing\n",
                (unsigned long long)low_dim);
        return 0;
    }
    void *lq = NULL, *lsf = NULL; int lkbp = 0;
    if (!pulsar_gpu_mxfp8_act_cache_e4m3_slot(low, n_tokens, low_dim, &lq, &lsf, &lkbp)) return 0;
    const int lwarps = (int)n_tokens * (int)(low_dim / 32);
    mxfp8_quant_act_kernel<<<(lwarps * 32 + 255) / 256, 256>>>(
            (const float *)low->ptr, (int)n_tokens, (int)low_dim, lkbp,
            (__nv_fp8_e4m3 *)lq, (unsigned char *)lsf);
    if (!cuda_ok(cudaGetLastError(), "attn-out low e4m3 emit")) return 0;
    pulsar_gpu_mxfp8_act_cache_arm(low, n_tokens, low_dim);
    pulsar_gpu_mxfp8_act_cache_note_mxfp8();
    return 1;
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
        const uint64_t n_dec = (uint64_t)g_batch_decode_rows;
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
            const int saved = g_batch_decode_rows;
            g_batch_decode_rows = (int)n_dec;
            int r1 = pulsar_gpu_attention_output_batch_tensor(&out_pre, &low_pre, model_map,
                    model_size, out_a_offset, out_b_offset, group_dim, rank, n_groups,
                    out_dim, &hd_pre, (uint32_t)n_dec);
            g_batch_decode_rows = 0;
            /* The suffix is a prefill run inside a step wider than the cap:
             * its 'a' takes the tensor-core arm at any width, reading the
             * producer's full-width grouped encoding through a row window
             * (L158 inc 4); its `low` is then emitted and consumed as its own
             * producer/consumer pair.  An offset heads view keyed no encoding
             * and used to be quantised -- that path is gone. */
            (void)hd_suf;
            const uint32_t n_suf = n_tokens - (uint32_t)n_dec;
            int r2 = cuda_attention_output_a_mx_gemm(&low_suf, model_map, model_size, out_a_offset,
                                                     group_dim, rank, n_groups, heads, n_suf, (uint32_t)n_dec);
            if (r2) r2 = emit_low_e4m3(&low_suf, n_suf, low_dim);
            if (r2) r2 = cuda_matmul_mxfp8_tensor_labeled(&out_suf, model_map, model_size, out_b_offset,
                                                          low_dim, out_dim, &low_suf, n_suf, "attn_output_b");
            g_batch_decode_rows = saved;
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

    /* "a" projection by row kind (see the header at g_batch_decode_rows; the
     * split above leaves n_dec at 0 or >= n_tokens): prefill rows take the
     * block-scaled MXFP8xMXFP8 tensor-core GEMMs at any n_tokens, one
     * included; decode rows take launch_grouped_fp8mx_a -- the warp8/nt A8
     * kernels, one launch for all groups, each row bit-identical to the
     * n == 1 kernel, so a row's attn-output bytes do not depend on its
     * batchmates.  Either arm runs or the call refuses; the tensor-core arm's
     * failure used to fall to the GEMV. */
    if (g_batch_decode_rows == 0) {
        if (!cuda_attention_output_a_mx_gemm(low, model_map, model_size, out_a_offset,
                                             group_dim, rank, n_groups, heads, n_tokens, 0)) {
            fprintf(stderr, "pulsar: attn-out 'a' tensor-core GEMM failed (n_tokens=%u rank=%llu "
                            "group_dim=%llu) -- refusing\n",
                    n_tokens, (unsigned long long)rank, (unsigned long long)group_dim);
            return 0;
        }
    } else {
        if (!launch_grouped_fp8mx_a((float *)low->ptr, model_map, out_a_offset, out_a_bytes,
                                    group_dim, rank, n_groups, n_tokens, blocks_a, low_dim,
                                    (const pulsar_heads_t *)heads->ptr, "attn_out_a")) return 0;
    }
    /* Emit `low` here, as the split-step suffix arm above does after its own
     * "a" GEMM: the "b" GEMM consumes it next.  This entry is what the SERVER
     * actually takes -- with the drafter live, verify batches come through
     * here -- and an emission that once lived only on a since-deleted
     * single-row entry converted the benchmark and left production on f32.
     * The bench has no drafter, so it could not have shown that. */
    if (!emit_low_e4m3(low, n_tokens, low_dim)) return 0;

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




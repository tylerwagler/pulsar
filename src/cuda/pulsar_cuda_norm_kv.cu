#include "pulsar_cuda_internal.h"
#include "pulsar_cuda_rope.cuh"   /* the tail-rope math: ONE authority (L074) */
#include "pulsar_cuda_mx.cuh"
#include <cuda_fp8.h>
#include <cuda_bf16.h>



/* Plain (no-weight) RMSNorm. Input x is ALWAYS an HC residual carrier (the
 * hc_dim-wide flatten before each sublayer / the output head) — every caller
 * feeds cur_hc/after_attn_hc/batch_cur_hc etc. — so x loads through pulsar_hc_load
 * (BF16 storage promoted to f32). Output and the sum-of-squares stay f32. */
/* BIT-EXACT ILP rewrite (2026-07-21).  The decode call sites launch this with
 * grid==1 (one block, 256 threads, n == hc_dim == 16384) -- 1 of 48 SMs, 8
 * warps, and a scalar strided load loop the compiler will not unroll because
 * blockDim.x is a runtime value.  That leaves ~2 loads in flight per warp and
 * makes the kernel pure memory LATENCY.  Templating the block width makes the
 * stride a compile-time constant so the batch below issues UNROLL independent
 * loads before the first FMA.
 *
 * REDUCTION ORDER IS UNCHANGED, WHICH IS THE WHOLE POINT.  Thread t still owns
 * exactly {t, t+BLK, t+2*BLK, ...} and still accumulates them in increasing
 * order; batching UNROLL of those into registers first only reorders the LOADS,
 * never the adds.  The shared-memory pairwise tree is byte-for-byte the old one
 * (no warp shuffles, no split-K): every float op happens in the same sequence,
 * so this is bit-exact by construction, not by measurement. */
/* BF16 narrowing, byte-for-byte the expression f32_to_bf16_kernel uses in
 * pulsar_cuda_matmul.cu.  Written out rather than calling __float2bfloat16():
 * this epilogue REPLACES that kernel's output for the same values, so the two
 * must agree bit-for-bit, and "the intrinsic is also round-to-nearest-even"
 * is an assumption where copying the expression is a guarantee. */
__device__ __forceinline__ static uint16_t pulsar_f32_to_bf16_rne(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);   ///< round-to-nearest-even
}

template <uint32_t BLK, uint32_t UNROLL>
__global__ static void rms_norm_plain_kernel(float *out, uint16_t *out_b,
                                             const pulsar_hc_t *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const pulsar_hc_t *xr = x + (uint64_t)row * n;
    /* `out` may be NULL (L157): when the only consumer is the bf16 GEMM core and
     * no dump wants the f32, the f32 row is a dead store -- 64 KB of the row's
     * 160 KB of traffic at hc_dim.  The launcher passes NULL only with out_b
     * set and the skip declared to the activation cache (fail-loud on a miss). */
    float *orow = out ? out + (uint64_t)row * n : nullptr;
    /* Optional producer-side BF16 copy (L086 T3).  The consumer of this buffer
     * is a BF16 GEMM (pulsar_gpu_matmul_f32_tensor runs the shared bf16 core),
     * so emitting here deletes its convert pass rather than moving it: the
     * value is already in a register and already scaled. */
    uint16_t *brow = out_b ? (uint16_t *)out_b + (uint64_t)row * n : nullptr;
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    uint32_t i = tid;
    for (; i + (UNROLL - 1u) * BLK < n; i += BLK * UNROLL) {
        float v[UNROLL];
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) v[u] = pulsar_hc_load(xr, i + u * BLK);
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) sum += v[u] * v[u];
    }
    for (; i < n; i += BLK) {
        float v = pulsar_hc_load(xr, i);
        sum += v * v;
    }
    __shared__ float partial[BLK];
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t stride = BLK >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    i = tid;
    for (; i + (UNROLL - 1u) * BLK < n; i += BLK * UNROLL) {
        float v[UNROLL];
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) v[u] = pulsar_hc_load(xr, i + u * BLK);
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) {
            const float o = v[u] * scale;
            if (orow) orow[i + u * BLK] = o;
            if (brow) brow[i + u * BLK] = pulsar_f32_to_bf16_rne(o);
        }
    }
    for (; i < n; i += BLK) {
        const float o = pulsar_hc_load(xr, i) * scale;
        if (orow) orow[i] = o;
        if (brow) brow[i] = pulsar_f32_to_bf16_rne(o);
    }
}



/* out_q/out_sf, when non-NULL, additionally emit the E4M3 + ue8m0 encoding, so
 * a GEMM consuming this norm multiplies in the source's format instead of
 * against f32.  Same contract as pulsar_cuda_mx.cuh: every lane of a warp must
 * reach the emit, which the launcher guarantees by refusing n % 256 != 0. */
/* WBF16: the norm weight is bf16 (source format) rather than f32. Storage only
 * -- the value is promoted to f32 before it multiplies, so the arithmetic here
 * is identical either way and an f32 artifact stays bit-exact. */
template <bool WBF16>
__global__ static void rms_norm_weight_kernel(float *out, const float *x, const void *w, uint32_t n, uint32_t rows, float eps,
                                              __nv_fp8_e4m3 *out_q, unsigned char *out_sf, int out_kbp,
                                              __nv_bfloat16 *out_b) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out ? out + (uint64_t)row * n : NULL;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] * scale * pulsar_w_load_f32_or_bf16<WBF16>(w, i);
        if (orow) orow[i] = v;
        if (out_q) pulsar_mx_emit_block(v, i, row, n, out_kbp, out_q, out_sf);
        /* L159: the bf16 plane for a bf16-weight consumer (the output head),
         * RNE from the same f32 the row stores -- the consumer no longer
         * converts on its own. */
        if (out_b) out_b[(uint64_t)row * n + i] = __float2bfloat16(v);
    }
}



/* q_out_q/q_out_sf, when non-NULL, receive the E4M3 + E8M0 encoding of the Q
 * half straight from this epilogue, so the MXFP8 attn_q_b GEMM does not wait on
 * a separate quantize pass over batch_qr_norm.  Q ONLY: batch_kv is not a GEMM
 * input, it goes to the KV cache.  See pulsar_cuda_mx.cuh for the contract --
 * in particular every lane must reach pulsar_mx_emit_block(), which is why the
 * host only supplies the slots when q_n is a multiple of the block size (then
 * the strided loop runs the same number of times on every thread). */
/* QWBF16/KVWBF16: storage of q_w and kv_w. Separate flags -- they are separate
 * tensors. `which` is blockIdx.y, so the select below is block-uniform and the
 * branch costs nothing. */
template <bool QWBF16, bool KVWBF16>
__global__ static void dsv4_qkv_rms_norm_rows_kernel(
        float *q_out,
        const float *q,
        const void *q_w,
        uint32_t q_n,
        float *kv_out,
        const float *kv,
        const void *kv_w,
        uint32_t kv_n,
        uint32_t rows,
        float eps,
        __nv_fp8_e4m3 *q_out_q,
        unsigned char *q_out_sf,
        int q_out_kbp) {
    const uint32_t row = blockIdx.x;
    const uint32_t which = blockIdx.y;
    if (row >= rows || which > 1u) return;
    const uint32_t n = which == 0u ? q_n : kv_n;
    const float *xr = (which == 0u ? q : kv) + (uint64_t)row * n;
    /* q_out is NULL when the caller declared the f32 store dead (the MXFP8
     * consumers read the E4M3 emitted below instead).  Select the base BEFORE
     * offsetting: NULL + row*n is undefined behaviour, not a harmless NULL. */
    float *obase = (which == 0u ? q_out : kv_out);
    float *orow = obase ? obase + (uint64_t)row * n : NULL;
    const bool is_q = (which == 0u);
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    const int emit_mx = (q_out_q != NULL) && (which == 0u);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float wv = is_q ? pulsar_w_load_f32_or_bf16<QWBF16>(q_w, i)
                              : pulsar_w_load_f32_or_bf16<KVWBF16>(kv_w, i);
        const float v = xr[i] * scale * wv;
        if (orow) orow[i] = v;
        if (emit_mx) {
            pulsar_mx_emit_block(v, i, row, n, q_out_kbp, q_out_q, q_out_sf);
        }
    }
}










/* positions (both RoPE kernels): per-row absolute query positions for banked
 * multi-session batches; NULL degenerates to the classic consecutive pos0+t
 * rule bit-exactly (same arithmetic on the same value). */
template <typename QT>
__global__ static void head_rms_norm_rope_tail_kernel(
        QT *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float eps,
        const int32_t * __restrict__ positions) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    uint32_t t = row / n_head;
    const uint32_t rope_pos = positions ? (uint32_t)positions[t] : pos0 + t;
    QT *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = q_load<QT>(xr, i);
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t i = threadIdx.x; i < n_nope; i += blockDim.x) {
        q_store<QT>(xr, i, q_load<QT>(xr, i) * scale);
    }

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f)
        rope_corr_dims_dev(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow, &corr0, &corr1);
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2; pair += blockDim.x) {
        uint32_t i = pair * 2u;
        /* The rotation itself stays in f32 for both instantiations; only the
         * two stores narrow.  Rotating in f16 would compound the rounding
         * across the pair and is not what the fp16-storage change is. */
        QT *tail = xr + n_nope;
        float x0 = q_load<QT>(tail, i) * scale;
        float x1 = q_load<QT>(tail, i + 1) * scale;
        float r0, r1;
        rope_pair_rotate_core_dev(x0, x1, i, n_rot, rope_pos, inverse,
                                  freq_base, freq_scale, ext_factor, attn_factor,
                                  corr0, corr1, &r0, &r1);
        q_store<QT>(tail, i,     r0);
        q_store<QT>(tail, i + 1, r1);
    }
}




/* One rope rotation pair, in place at tail[i], tail[i+1], for the callers that
 * rotate a stored tail (rope_tail_kernel and the fused indexer rope+QAT
 * kernel).  T is the buffer's STORED element type and is deduced from the
 * pointer, never named at the call site.
 *
 * ⚠ THIS IS NOT THE AUTHORITY -- it used to claim it was, while the header
 * claimed the same thing and a third copy sat in the kernel above. All of them
 * now call rope_pair_rotate_core_dev in pulsar_cuda_rope.cuh, which is the one
 * place the YaRN math lives. Returns the rotated pair through r0/r1 for callers
 * with an epilogue.
 *
 * T changes WHERE the values live, never HOW they are rotated: the operands
 * widen to float on the way in, the YaRN core is the same f32 code for every
 * instantiation, and only the store narrows.  So T=float emits the identical
 * instruction sequence this function had before it was templated -- which is
 * the property the byte gate checks, and the reason the narrowing can land
 * one buffer at a time instead of all at once. */
template <typename T>
__device__ static void rope_tail_rotate_pair_dev(
        T *tail, uint32_t i, uint32_t n_rot, uint32_t rope_pos,
        uint32_t n_ctx_orig, int inverse,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, float *out_r0, float *out_r1) {
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f)
        rope_corr_dims_dev(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow, &corr0, &corr1);

    float r0, r1;
    rope_pair_rotate_core_dev((float)tail[i], (float)tail[i + 1], i, n_rot, rope_pos, inverse,
                              freq_base, freq_scale, ext_factor, attn_factor,
                              corr0, corr1, &r0, &r1);
    tail[i] = (T)r0;
    tail[i + 1] = (T)r1;
    /* r0/r1 leave as f32; the MX epilogue rounds them to T before quantising
     * (L195).  This used to say the opposite -- quantise the f32 pair so that
     * narrowing T "does not put a second rounding in front of the E4M3
     * emission" -- and that made the E4M3 a function of WHICH kernel emitted
     * it: the read-back encoder sees tail[] and only tail[].  One operand, one
     * encoding; the rounding is the price of that and the reference gate
     * grades it. */
    *out_r0 = r0;
    *out_r1 = r1;
}

template <typename T>
__global__ static void rope_tail_kernel(
        T *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t pos_stride,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        const int32_t * __restrict__ positions,
        /* Grouped E4M3 slots for the attn-output "a" projection.  This kernel
         * rewrites head dims [n_nope, head_dim) IN PLACE after the attention
         * epilogue has already emitted [0, n_nope), so it owns -- and must
         * emit -- exactly the MX blocks covering the rope tail.  NULL = f32
         * only.  See pulsar_cuda_attn_f16.cu's epilogue for the other half. */
        __nv_fp8_e4m3 * __restrict__ gact_data,
        unsigned char * __restrict__ gact_scale,
        int gact_kbp, uint32_t gact_slab, uint32_t n_groups) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    /* pairs is a multiple of n_rot/2 (32 here), so this exit takes WHOLE warps
     * and never strands a lane before the shuffles in the MX epilogue below. */
    if (gid >= pairs) return;
    uint32_t pair = gid % (n_rot / 2);
    uint32_t tmp = gid / (n_rot / 2);
    uint32_t h = tmp % n_head;
    uint32_t t = tmp / n_head;
    uint32_t n_nope = head_dim - n_rot;
    uint32_t i = pair * 2;

    const uint32_t rope_pos = positions ? (uint32_t)positions[t] : pos0 + t * pos_stride;
    T *tail = x + ((uint64_t)t * n_head + h) * head_dim + n_nope;
    float r0, r1;
    rope_tail_rotate_pair_dev(tail, i, n_rot, rope_pos, n_ctx_orig, inverse,
                              freq_base, freq_scale, ext_factor, attn_factor,
                              beta_fast, beta_slow, &r0, &r1);

    if (gact_data) {
        /* One warp is one (t, h) and covers pair 0..n_rot/2-1, i.e. head dims
         * [n_nope, head_dim) -- TWO 32-element MX blocks when n_rot is 64.
         * Lane L holds dims n_nope+2L and n_nope+2L+1, so lanes 0..15 own the
         * first block and 16..31 the second: the amax is a HALF-warp reduction
         * (xor 1,2,4,8), and the lane at each half's base writes the scale. */
        const uint32_t lane = threadIdx.x & 31u;
        /* L195: quantise the values the rotate STORED (the heads type), not the
         * f32 pair -- the read-back encoder sees the stored values, and the two
         * must agree byte for byte (see heads_round). */
        const float q0 = (float)(T)r0, q1 = (float)(T)r1;
        float a = fmaxf(fabsf(q0), fabsf(q1));
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 4));
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 8));
        const int se = pulsar_mx_shared_exp(a);
        const uint32_t hpg = n_head / n_groups;         /* heads per group */
        const uint32_t gd  = hpg * head_dim;            /* == group_dim */
        const uint32_t grp = h / hpg, hh = h % hpg;
        const uint32_t d0  = n_nope + i;                /* absolute head dim */
        __nv_fp8_e4m3 *dst = gact_data + ((size_t)grp * n_tok + t) * gd + hh * head_dim;
        dst[d0]      = pulsar_mx_encode(q0, se);
        dst[d0 + 1u] = pulsar_mx_encode(q1, se);
        if ((lane & 15u) == 0u) {
            const uint32_t kb = hh * (head_dim / 32u) + (d0 / 32u);
            gact_scale[(size_t)grp * gact_slab +
                       pulsar_mx_sfoff((int)t, (int)kb, gact_kbp)] = pulsar_mx_scale_byte(se);
        }
    }
}






__device__ static float dsv4_e4m3fn_dequant_dev(float x) {
    /* Native e4m3 round-trip (cvt.rn.satfinite). PROVEN bit-identical to the
     * former 7-iteration binary search (each step an exp2f) by an exhaustive
     * sweep of all 2^32 finite float bit patterns: 4278190080 checked, zero
     * mismatches, including every RNE tie and the subnormal range
     * (temp/fp8test.cu). NaN inputs differ (search clamped to 448, native
     * propagates NaN) -- activations are finite, never hit. */
    return (float)__nv_fp8_e4m3(x);
}



__device__ static float dsv4_e2m1fn_value_dev(int i) {
    switch (i & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}




/* Encode to an OCP E2M1 (float_e2m1_t) 4-bit nibble: [sign:1][magnitude:3].
 * The magnitude table matches dsv4_e2m1fn_value_dev, i.e. CUTLASS float_e2m1_t. */
__device__ static uint8_t dsv4_e2m1fn_encode_dev(float x) {
    float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_dev(0));
    for (int i = 1; i < 8; i++) {
        float diff = fabsf(ax - dsv4_e2m1fn_value_dev(i));
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return (uint8_t)((best & 7) | ((x < 0.0f) ? 0x8u : 0u));
}

__device__ static float dsv4_e2m1fn_decode_dev(uint8_t nib, float scale) {
    float val = dsv4_e2m1fn_value_dev(nib & 7);
    return (nib & 8u) ? (-val * scale) : (val * scale);
}

/* E8M0 microscale decode lived here (byte = exponent+127, 255 reserved NaN).
 * Its only caller in this TU was mxkv_dequant_kernel.  The readers that still
 * decode E8M0 -- attn_comp_pack_ld and the indexer scorer -- each derive the
 * scale inline from the stored byte, which is why nothing here needs it. */

/* ===== INVARIANT: FRESH data -> fast path; RE-ENCODED data -> exact path =====
 * The `exp2f(ceilf(log2f(amax/K)))` form used by the quantize kernels below is
 * NOT safe on already-quantized input, and the split between the two forms in
 * this file is DELIBERATE. Do not "simplify" it to one path.
 *
 * VERIFIED on GB10 (2026-07-21, SASS + full-binade sweeps): under
 * --use_fast_math, `log2f` lowers to MUFU.LG2, whose error is one-sided HIGH.
 * `lg2.approx(2^k)` is wrong for EVERY k in [-126,-1] — i.e. the entire range
 * real block scales occupy — so at `amax == K*2^k` exactly, ceilf() rounds up
 * one extra step and the scale comes out 2x TOO LARGE. (`ex2.approx` and
 * `div.approx` were both verified EXACT here; MUFU.LG2 is the sole culprit.
 * Host `-ffast-math` is unaffected: glibc log2f stays exact at pow2.)
 *
 * WHY THE FAST FORM IS FINE HERE: these kernels see FRESH f32 activations, so
 * hitting `amax == K*2^k` is a 1-in-2^23 mantissa coincidence (~2.4e-7/block for
 * K=448, ~1.2e-7 for K=6) — well under the noise floor of the quantization
 * itself, and deterministic, so no bit-exactness gate destabilizes.
 *
 * WHY RE-ENCODING IS CATASTROPHIC: quantized values live on a dyadic LATTICE.
 * After E2M1, block values are {0,.5,1,1.5,2,3,4,6}*scale, so a block whose max
 * sits on the top code has `amax == 6*scale` EXACTLY — a guaranteed hit, and
 * `v/scale` rounds to 6.0 for anything above 5.0, so ~1/3 of re-encoded FP4
 * blocks land on it (~5% for E4M3's 448). A competing GB10 fork shipped exactly
 * this bug and lost 10.5% of their fp4 indexer lanes to zero.
 *
 * **If you ever feed already-quantized data into `fp8_kv_quantize*`,
 * `attn_pack_store`, or `indexer_hadamard_fp4*`, the misround rate jumps from
 * 1e-7 to ~5% (E4M3) or ~33% (FP4) instantly.**  the old standalone
 * quantizer's `quantize_fp8` flag was hardcoded false at every call site for
 * exactly this reason -- and was deleted, with its kernel, in the L093 sweep:
 * the pack commit is the single fp8 quantizer.
 *
 * This is MEASURED, not theoretical.  On 2026-08-18 three KV paths were found
 * quantizing the same rows twice (prefill's ring store re-quantized the buffer
 * the pack had just round-tripped; the draft batch and the drafter seed did the
 * same).  Removing the second pass CHANGED THE BYTES: an 18-token prompt is one
 * chunk and never reads the ring, and its logits were identical either way,
 * while a 5530-token prompt that does read the ring moved 27.788 -> 27.321.
 * So the fast-math bucket is NOT value-idempotent, and the call site that
 * claimed the two passes "agree byte for byte" was wrong.  It cost 2.9% of
 * decode acceptance to stop doing it, and it was still worth it -- the ring now
 * holds what attention actually read.
 *
 * THE RULE IS THEREFORE: never re-encode.  MOVE PACKED BYTES.  Every KV path in
 * the engine now does -- prefill's ring scatter, session save/load (payload v5),
 * and the drafter seed.  The exact integer-math scale buckets that used to make
 * re-encoding survivable (`attn_pack_exact_e8_dev`,
 * `dsv4_e8m0_encode_scale_exact_dev`) are deleted, because nothing re-encodes
 * any more and an unreferenced safety net is just code that rots.  If a future
 * restore path genuinely cannot move bytes, recover one from git history rather
 * than reaching for the fresh quantizers above -- that is the whole point of
 * this comment.
 *
 * RESOLVED BY DELETION (2026-08-25, L106 K11): this note used to record that
 * the amax floor 7.052966104933725e-38f (= exactly 6.0f * 2^-126, ON a
 * misround point) made an empty block's stored E8M0 byte differ from what the
 * mxkv_pack RESTORE packer wrote for the same row — a latent flake for any
 * byte-comparing gate, deferred to the next golden re-baseline.  The restore
 * packer no longer exists: the v5 session payload moves packed bytes verbatim
 * both directions, so there is exactly ONE packer and no second path to agree
 * with.  The floor stays as-is — "folding" it now would change bytes for zero
 * benefit.  Value impact was always NIL (every lane encodes nibble 0 either
 * way).  (The E4M3 floor 1.0e-4f was checked and is NOT on a misround point.)
 * ============================================================================ */

__device__ static float model_scalar_dev(const void *base, uint64_t offset, uint32_t type, uint64_t idx) {
    const char *p = (const char *)base + offset;
    if (type == 1u) return __half2float(((const __half *)p)[idx]);
    return ((const float *)p)[idx];
}









__device__ static uint8_t dsv4_e4m3fn_encode_dev(float x) {
    /* Native e4m3 encode: the former (exp<<3)|mant index IS the e4m3fn bit
     * pattern (sign in 0x80), so the hardware cvt byte is the same encoding.
     * Bit-identity proven by the exhaustive round-trip sweep (see
     * dsv4_e4m3fn_dequant_dev). */
    __nv_fp8_e4m3 f(x);
    return *(const uint8_t *)&f;
}


#define PULSAR_FP8_KV_BLOCK 64u
#define PULSAR_FP8_KV_NBLK(HD) (((HD) + PULSAR_FP8_KV_BLOCK - 1u) / PULSAR_FP8_KV_BLOCK)
#define PULSAR_FP8_KV_ROWBYTES(HD) ((HD) + PULSAR_FP8_KV_NBLK(HD) * sizeof(float))



/* fp8_kv_quantize_kernel (the standalone in-place quantizer) lived here until
 * the 2026-08-22 launched-vs-defined sweep (L093): every caller passed
 * quantize_fp8=false -- the pack-store below has been the single fp8 quantizer
 * since the 2026-08-18 double-quantize audit -- so the kernel, its wrapper
 * pulsar_gpu_dsv4_fp8_kv_quantize_tensor, and the flag were dead dispatch.
 * The recipe lives on in attn_pack_store. */

/* PULSAR_ATTN_PACK store: quantize the nope dims of n_rows f32 rows of x with
 * the engine's ONE NVFP4 recipe (per-PULSAR_KV4_NV_BLOCK = 16 amax, one f32 row
 * scale keyed so every block scale fits E4M3, a per-block E4M3 scale code whose
 * DECODED value x row scale is what both the encode and every reader use, E2M1
 * nibbles by dsv4_e2m1fn_encode_dev -- L111), write the roundtripped f32 back
 * into x (so the stage/dumps show the same values the packed row decodes to),
 * and store the packed rows (see PULSAR_ATTN_PACK_* in pulsar_cuda_internal.h;
 * 384 B at head_dim 512) into `out` at rows [out_row0, out_row0+n_rows).  The rope tail
 * takes the same treatment one dtype up: bf16-roundtripped in place, then
 * stored.  Read-back is bit-identical to the f32 path. */
/* `x` is the OPTIONAL f32 staging to round-trip in place; NULL when the source
 * must not be modified (the raw-ring writers take a const kv).  `src` is what is
 * read.  positions/seq_id/n_banks/raw_cap give the ring scatter the raw cache
 * needs -- destination row is bank*raw_cap + pos%raw_cap.  raw_cap == 0
 * degenerates to consecutive rows at out_row0, which is what the compressed pool
 * wants, so ONE kernel now writes both KV caches in the one 384 B format. */
/* WHERE A BATCH ROW LANDS IN THE RING.  One definition, deliberately.
 *
 * attn_pack_store_kernel (quantise then store) and attn_pack_scatter_kernel
 * (copy already-packed bytes) must agree on this exactly: they write the same
 * ring from the same batch, and a disagreement puts a token's KV in the wrong
 * slot.  Nothing about that fails to compile, and it would surface as a
 * position-dependent wrong answer rather than a crash.  The mapping was copied
 * verbatim between the two when the scatter path was added (2026-08-18) with a
 * comment saying they must stay in step; this replaces the comment.
 *
 * Returns ATTN_PACK_DEAD_ROW when seq_id puts the row outside the pool.  A
 * helper cannot return on the caller's behalf, hence a sentinel -- and the
 * caller's early-out stays UNIFORM ACROSS THE BLOCK, which matters because the
 * store kernel __syncthreads() below: the decision depends only on blockIdx.x,
 * so either the whole block leaves or none of it does. */
#define ATTN_PACK_DEAD_ROW (~0ull)

__device__ __forceinline__ static uint64_t attn_pack_ring_slot(
        uint32_t row, uint32_t out_row0, uint32_t raw_cap, uint32_t n_banks,
        const int32_t *__restrict__ positions,
        const int32_t *__restrict__ seq_id) {
    if (!raw_cap) return (uint64_t)(out_row0 + row);
    if (seq_id && (uint32_t)seq_id[row] >= n_banks) return ATTN_PACK_DEAD_ROW;
    const uint32_t pos = positions ? (uint32_t)positions[row] : out_row0 + row;
    return (uint64_t)(seq_id ? (uint32_t)seq_id[row] * raw_cap : 0u) + pos % raw_cap;
}

__global__ static void attn_pack_store_kernel(float *x, const float *src, uint8_t *out,
                                              uint32_t out_row0, uint32_t n_rows,
                                              uint32_t head_dim, uint32_t n_rot,
                                              const int32_t * __restrict__ positions,
                                              const int32_t * __restrict__ seq_id,
                                              uint32_t n_banks, uint32_t raw_cap) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;      /* 64 threads */
    if (row >= n_rows) return;
    const uint64_t dst_row = attn_pack_ring_slot(row, out_row0, raw_cap, n_banks,
                                                positions, seq_id);
    if (dst_row == ATTN_PACK_DEAD_ROW) return;   /* dead row stores nothing */
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
    /* Per-16 amax.  Non-negative floats order-match their bit patterns, so
     * atomicMax on the int view is exact -- and max is order-independent, so
     * the result is deterministic. */
    for (uint32_t d = tid; d < n_nope; d += blockDim.x) {
        atomicMax((int *)&samax[d / PULSAR_KV4_NV_BLOCK], __float_as_int(fabsf(sr[d])));
    }
    __syncthreads();
    if (tid == 0) {
        float ra = 0.0f;
        for (uint32_t bk = 0; bk < nblk; bk++) ra = fmaxf(ra, samax[bk]);
        /* Row scale keyed so every block scale block_amax/(6*row_scale) fits
         * E4M3's [0, 448]; the 1e-4 amax floor matches the retired fp8
         * recipe's. */
        const float rs = fmaxf(ra, 1.0e-4f) * (1.0f / (6.0f * 448.0f));
        srow = rs;
        *(float *)(sc + nblk) = rs;   /* 4-aligned: nib 224 + 28 codes */
    }
    __syncthreads();
    for (uint32_t bk = tid; bk < nblk; bk += blockDim.x) {
        /* The DECODED scale (e4m3 roundtrip x row scale) is what both the
         * encode below and every reader use; a round-down clips the block's
         * extremes into the top code -- the standard NVFP4 trade, measured
         * in the L111 verdict. */
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

/* The QAT transform both indexer FP4 kernels run before they diverge: the
 * 128-point Hadamard butterfly, the 1/sqrt(128) scaling, and the per-32-block
 * absmax left in absbuf[block_base].  Returns this thread's transformed value.
 *
 * The packing kernel's contract is that it writes a BIT-IDENTICAL f32 result to
 * the unpacked one, and two hand-kept copies of this code is exactly how such a
 * contract rots silently.  One copy makes it structural.
 *
 * Called by all 128 threads of the block, after the kernels' own bounds check --
 * the __syncthreads() below are as uniform here as they were when inlined. */
/** One thread's result from the shared Hadamard + block-absmax step.
 *
 * Returned by value so the packing and non-packing indexer kernels can share
 * ONE copy of that code -- see the note above: the packing kernel's contract is
 * a bit-identical f32 result, and two hand-kept copies is exactly how such a
 * contract rots silently. */
struct indexer_had_t {
    float    v;           ///< this thread's transformed value
    uint32_t fp4_block;   ///< which 32-wide FP4 block it lands in
    uint32_t lane;        ///< its lane within that block
    uint32_t block_base;  ///< absbuf[block_base] holds the block's absmax
};

__device__ static inline indexer_had_t indexer_hadamard_block_absmax_dev(
        const float *xr, uint32_t tid, float *vals, float *absbuf) {
    vals[tid] = xr[tid];
    __syncthreads();

    for (uint32_t stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            uint32_t base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            float a = vals[base];
            float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        __syncthreads();
    }

    float v = vals[tid] * 0.08838834764831845f;
    uint32_t fp4_block = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t block_base = fp4_block * 32u;
    absbuf[tid] = fabsf(v);
    __syncthreads();

    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = fmaxf(absbuf[block_base + lane],
                                              absbuf[block_base + lane + stride]);
        }
        __syncthreads();
    }
    return { v, fp4_block, lane, block_base };
}


/* Fused indexer-q epilogue: rope the tail, then the Hadamard+FP4 QAT
 * round-trip, one 128-thread block per (token, head) row. Replaces the two
 * back-to-back launches over the same tensor (rope_tail_kernel then
 * indexer_hadamard_fp4_pack_kernel), saving a full read+write of the buffer. The
 * rotation is the SAME device function rope_tail_kernel runs and the QAT
 * body is the same code as indexer_hadamard_fp4_pack_kernel, in the same order,
 * so the PACKED row is bit-exact vs the two-launch sequence (this kernel writes
 * no f32 back -- see the note at its QAT phase); the __syncthreads
 * between the phases stands in for the old kernel boundary (one block owns
 * the whole row, so a block-local barrier is equivalent). */
__global__ static void indexer_rope_hadamard_fp4_pack_q_kernel(
        float *x, uint8_t *out, uint32_t n_rows, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
        uint32_t pos0, uint32_t n_ctx_orig, int inverse,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow,
        const int32_t * __restrict__ positions) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    float *xr = x + (uint64_t)row * head_dim;
    if (tid < n_rot / 2u) {
        const uint32_t t = row / n_head;
        const uint32_t rope_pos = positions ? (uint32_t)positions[t] : pos0 + t;
        float r0, r1;
        rope_tail_rotate_pair_dev(xr + (head_dim - n_rot), tid * 2u, n_rot, rope_pos,
                                  n_ctx_orig, inverse, freq_base, freq_scale,
                                  ext_factor, attn_factor, beta_fast, beta_slow,
                                  &r0, &r1);
    }
    __syncthreads();

    /* Hadamard + QAT, identical math to indexer_hadamard_fp4_pack_kernel --
     * but the E2M1 code + E8M0 scale ARE the output now.  The f32 dequant
     * writeback that used to live here (the "QAT round-trip") is gone: x is
     * producer-internal rope staging, and every consumer reads the packed
     * row, so there is exactly one Q operand encoding (L090.4). */
    __shared__ float vals[128];
    __shared__ float absbuf[128];
    __shared__ uint8_t nib_sh[128];
    indexer_had_t h = indexer_hadamard_block_absmax_dev(xr, tid, vals, absbuf);

    float amax = fmaxf(absbuf[h.block_base], 7.052966104933725e-38f);
    int e8 = (int)ceilf(log2f(amax / 6.0f)) + 127;
    e8 = e8 < 0 ? 0 : (e8 > 254 ? 254 : e8);
    float scale = exp2f((float)(e8 - 127));
    uint8_t nib = dsv4_e2m1fn_encode_dev(fminf(6.0f, fmaxf(-6.0f, h.v / scale)));
    nib_sh[tid] = nib;
    __syncthreads();

    uint8_t *outr = out + (uint64_t)row * PULSAR_MXKV_FP4_ROWBYTES(128u);
    if (tid < 64u) outr[tid] = (uint8_t)(nib_sh[2u * tid] | (nib_sh[2u * tid + 1u] << 4));
    if (h.lane == 0u) outr[64u + h.fp4_block] = (uint8_t)e8;
}

/* Same QAT transform as indexer_rope_hadamard_fp4_pack_q_kernel minus the rope,
 * emitting the row in MXKV FP4 layout — E2M1 nibble pairs low-nibble-first
 * followed by one E8M0 byte per 32-block; the dequantised f32 goes back into x
 * only under keep_f32 (observers -- every consumer reads the packed row).  The E8M0
 * exponent clamp only differs from the unpacked path outside [2^-127, 2^127]
 * scales, which the 7e-38 amax floor already makes unreachable. */
__global__ static void indexer_hadamard_fp4_pack_kernel(float *x, uint8_t *out,
                                                        uint32_t n_rows, uint32_t head_dim,
                                                        int keep_f32) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    __shared__ float vals[128];
    __shared__ float absbuf[128];
    __shared__ uint8_t nib_sh[128];
    float *xr = x + (uint64_t)row * head_dim;
    indexer_had_t h = indexer_hadamard_block_absmax_dev(xr, tid, vals, absbuf);

    float amax = fmaxf(absbuf[h.block_base], 7.052966104933725e-38f);
    int e8 = (int)ceilf(log2f(amax / 6.0f)) + 127;
    e8 = e8 < 0 ? 0 : (e8 > 254 ? 254 : e8);
    float scale = exp2f((float)(e8 - 127));
    uint8_t nib = dsv4_e2m1fn_encode_dev(fminf(6.0f, fmaxf(-6.0f, h.v / scale)));
    /* The dequantised writeback is for OBSERVERS only -- the packed rows below
     * are what every consumer reads (L094).  Skipped unless a dump or the
     * range sweep will actually look at the staging. */
    if (keep_f32) xr[tid] = dsv4_e2m1fn_decode_dev(nib, scale);
    nib_sh[tid] = nib;
    __syncthreads();

    uint8_t *outr = out + (uint64_t)row * PULSAR_MXKV_FP4_ROWBYTES(128u);
    if (tid < 64u) outr[tid] = (uint8_t)(nib_sh[2u * tid] | (nib_sh[2u * tid + 1u] << 4));
    if (h.lane == 0u) outr[64u + h.fp4_block] = (uint8_t)e8;
}









__global__ static void compressor_store_kernel(
        const float *kv,
        const float *sc,
        float *state_kv,
        float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens) {
    uint32_t coff = pulsar_compress_coff(ratio);
    uint32_t width = coff * head_dim;
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * width;
    if (gid >= n) return;
    uint32_t t = gid / width;
    uint32_t j = gid - (uint64_t)t * width;
    uint32_t pos_mod = (pos0 + t) % ratio;
    uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    state_kv[(uint64_t)dst_row * width + j] = kv[(uint64_t)t * width + j];
    state_score[(uint64_t)dst_row * width + j] =
        sc[(uint64_t)t * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)pos_mod * width + j);
}



__global__ static void compressor_set_rows_kernel(
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t width,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t src0,
        uint32_t dst0,
        uint32_t rows) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)rows * width;
    if (gid >= n) return;
    uint32_t r = gid / width;
    uint32_t j = gid - (uint64_t)r * width;
    uint32_t src = src0 + r;
    uint32_t dst = dst0 + r;
    uint32_t phase = (pos0 + src) % ratio;
    state_kv[(uint64_t)dst * width + j] = kv[(uint64_t)src * width + j];
    state_score[(uint64_t)dst * width + j] =
        sc[(uint64_t)src * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)phase * width + j);
}



__global__ static void compressor_prefill_pool_kernel(
        float *comp,
        const float *kv,
        const float *sc,
        const float *state_kv,
        const float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_comp,
        uint32_t replay) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t c = blockIdx.y;
    if (d >= head_dim || c >= n_comp) return;
    uint32_t coff = pulsar_compress_coff(ratio);
    uint32_t width = coff * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        if (replay && c == 0) {
            for (uint32_t r = 0; r < 4; r++) {
                vals[n_cand] = state_kv[(uint64_t)r * width + d];
                scores[n_cand] = state_score[(uint64_t)r * width + d];
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        } else if (c > 0) {
            uint32_t base = (c - 1u) * ratio;
            for (uint32_t r = 0; r < 4; r++) {
                uint32_t t = base + r;
                float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
                vals[n_cand] = kv[(uint64_t)t * width + d];
                scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        }
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < 4; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + head_dim + d);
            vals[n_cand] = kv[(uint64_t)t * width + head_dim + d];
            scores[n_cand] = sc[(uint64_t)t * width + head_dim + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < ratio; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
            vals[n_cand] = kv[(uint64_t)t * width + d];
            scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    comp[(uint64_t)c * head_dim + d] = den != 0.0f ? acc / den : 0.0f;
}



__global__ static void compressor_update_pool_kernel(
        float *row,
        const float *state_kv,
        const float *state_score,
        uint32_t head_dim,
        uint32_t ratio) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    uint32_t coff = pulsar_compress_coff(ratio);
    uint32_t width = coff * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)(ratio + r) * width + head_dim + d];
            scores[n_cand] = state_score[(uint64_t)(ratio + r) * width + head_dim + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        for (uint32_t r = 0; r < ratio; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    row[d] = den != 0.0f ? acc / den : 0.0f;
}



__global__ static void compressor_shift_ratio4_kernel(float *state_kv, float *state_score, uint32_t width) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t half = 4ull * width;
    if (i >= half) return;
    float v = state_kv[half + i];
    float s = state_score[half + i];
    state_kv[i] = v;
    state_score[i] = s;
    state_kv[half + i] = v;
    state_score[half + i] = s;
}





int pulsar_gpu_rms_norm_plain_rows_tensor(pulsar_gpu_tensor *out, void *out_b, const pulsar_gpu_tensor *x, uint32_t n, uint32_t rows, float eps,
                                          int skip_f32) {
    if (!out || !x || out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * PULSAR_HC_ELT_SIZE) return 0;   /* x is an HC residual carrier */
    /* L157: the f32 rows may be skipped only when the bf16 copy replaces them;
     * a skip with nothing to read instead would leave the consumer converting
     * unwritten bytes, which the bf16 core refuses loudly -- refuse it here too. */
    if (skip_f32 && !out_b) return 0;
    /* The F16 destination that used to sit here went out with the last F16
     * weight (2026-08-16) -- every caller passed NULL, so it could never fire.
     * `out_b` is its BF16 successor and is NOT the same thing: the consumer
     * (pulsar_gpu_matmul_f32_tensor -> the shared bf16 core) genuinely wants
     * BF16, and BF16 is the right 16-bit format here -- 8 exponent bits keep
     * f32's range, where F16's 5 never did.  NULL is still valid and still
     * emits nothing. */
    rms_norm_plain_kernel<256, 8><<<rows, 256>>>(skip_f32 ? nullptr : (float *)out->ptr,
                                                 (uint16_t *)out_b,
                                                 (const pulsar_hc_t *)x->ptr, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}


int pulsar_gpu_rms_norm_weight_mx_tensor(pulsar_gpu_tensor *out, const pulsar_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps,
        void *out_q, void *out_sf, int out_kbp, void *out_b, int w_bf16) {
    /* The weight is sized by ITS storage; x is an f32 buffer.  out may be NULL
     * when an E4M3 or bf16 plane is requested: the f32 row is then not stored
     * (its only readers are dumps, and the caller passes out only for them). */
    const uint64_t w_bytes = (uint64_t)n * pulsar_w_elt_bytes(w_bf16);
    if ((!out && !out_q && !out_b) || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < w_bytes ||
        (out && out->bytes < (uint64_t)n * sizeof(float)) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    const void *w = cuda_model_range_ptr(model_map, weight_offset, w_bytes, "rms_weight");
    if (!w) return 0;
    /* FAIL LOUD, not skip: the caller arms the activation cache off the same
     * slot pointer, so a silently skipped emission leaves the GEMM reading a
     * memset-zero E4M3 buffer -- a well-formed WRONG answer. */
    if (out_q && (n % 256u) != 0u) {
        fprintf(stderr, "pulsar: rms_norm_weight cannot emit MX for n=%u "
                        "(needs a multiple of 256)\n", n);
        return 0;
    }
    if (w_bf16)
        rms_norm_weight_kernel<true><<<1, 256>>>(out ? (float *)out->ptr : NULL, (const float *)x->ptr, w, n, 1, eps,
                                           (__nv_fp8_e4m3 *)out_q, (unsigned char *)out_sf, out_kbp,
                                              (__nv_bfloat16 *)out_b);
    else
        rms_norm_weight_kernel<false><<<1, 256>>>(out ? (float *)out->ptr : NULL, (const float *)x->ptr, w, n, 1, eps,
                                           (__nv_fp8_e4m3 *)out_q, (unsigned char *)out_sf, out_kbp,
                                              (__nv_bfloat16 *)out_b);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}

int pulsar_gpu_rms_norm_weight_tensor(pulsar_gpu_tensor *out, const pulsar_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps, int w_bf16) {
    return pulsar_gpu_rms_norm_weight_mx_tensor(out, x, model_map, model_size, weight_offset, n, eps, NULL, NULL, 0, NULL, w_bf16);
}


int pulsar_gpu_rms_norm_weight_rows_tensor(pulsar_gpu_tensor *out, const pulsar_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows, float eps, void *out_b, int w_bf16) {
    const uint64_t w_bytes = (uint64_t)n * pulsar_w_elt_bytes(w_bf16);
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < w_bytes ||
        out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    const void *w = cuda_model_range_ptr(model_map, weight_offset, w_bytes, "rms_weight");
    if (!w) return 0;
    if (w_bf16)
        rms_norm_weight_kernel<true><<<rows, 256>>>(out ? (float *)out->ptr : NULL, (const float *)x->ptr, w, n, rows, eps,
                                              NULL, NULL, 0, (__nv_bfloat16 *)out_b);
    else
        rms_norm_weight_kernel<false><<<rows, 256>>>(out ? (float *)out->ptr : NULL, (const float *)x->ptr, w, n, rows, eps,
                                              NULL, NULL, 0, (__nv_bfloat16 *)out_b);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}

int pulsar_gpu_rms_norm_weight_rows_mx_tensor(pulsar_gpu_tensor *out, const pulsar_gpu_tensor *x,
                                              const void *model_map, uint64_t model_size,
                                              uint64_t weight_offset, uint32_t n, uint32_t rows, float eps,
                                              void *out_q, void *out_sf, int out_kbp, void *out_b, int w_bf16) {
    /* L158: the rows twin of pulsar_gpu_rms_norm_weight_mx_tensor -- the same
     * kernel, `rows` blocks, with the E4M3 epilogue the drafter's per-row
     * norms (attn_norm, q_a_norm) needed so their GEMVs read a slot instead
     * of f32.  Same fail-loud rule as the single-row variant. */
    const uint64_t w_bytes = (uint64_t)n * pulsar_w_elt_bytes(w_bf16);
    if ((!out && !out_q && !out_b) || !x || !model_map || rows == 0 || weight_offset > model_size ||
        model_size - weight_offset < w_bytes ||
        (out && out->bytes < (uint64_t)n * rows * sizeof(float)) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    const void *w = cuda_model_range_ptr(model_map, weight_offset, w_bytes, "rms_weight");
    if (!w) return 0;
    if (out_q && (n % 256u) != 0u) {
        fprintf(stderr, "pulsar: rms_norm_weight (rows) cannot emit MX for n=%u "
                        "(needs a multiple of 256)\n", n);
        return 0;
    }
    if (w_bf16)
        rms_norm_weight_kernel<true><<<rows, 256>>>(out ? (float *)out->ptr : NULL, (const float *)x->ptr, w, n, rows, eps,
                                              (__nv_fp8_e4m3 *)out_q, (unsigned char *)out_sf, out_kbp,
                                              (__nv_bfloat16 *)out_b);
    else
        rms_norm_weight_kernel<false><<<rows, 256>>>(out ? (float *)out->ptr : NULL, (const float *)x->ptr, w, n, rows, eps,
                                              (__nv_fp8_e4m3 *)out_q, (unsigned char *)out_sf, out_kbp,
                                              (__nv_bfloat16 *)out_b);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight rows mx launch");
}


int pulsar_gpu_dsv4_qkv_rms_norm_rows_mx_tensor(
        pulsar_gpu_tensor       *q_out,
        const pulsar_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        pulsar_gpu_tensor       *kv_out,
        const pulsar_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps,
        void                   *q_out_q,
        void                   *q_out_sf,
        int                     q_out_kbp,
        int                     q_w_bf16,
        int                     kv_w_bf16,
        int                     q_skip_f32) {
    const uint64_t q_w_bytes = (uint64_t)q_n * pulsar_w_elt_bytes(q_w_bf16);
    const uint64_t kv_w_bytes = (uint64_t)kv_n * pulsar_w_elt_bytes(kv_w_bf16);
    if (!q_out || !q || !kv_out || !kv || !model_map ||
        q_weight_offset > model_size ||
        kv_weight_offset > model_size ||
        model_size - q_weight_offset < q_w_bytes ||
        model_size - kv_weight_offset < kv_w_bytes ||
        q_out->bytes < (uint64_t)q_n * rows * sizeof(float) ||
        q->bytes < (uint64_t)q_n * rows * sizeof(float) ||
        kv_out->bytes < (uint64_t)kv_n * rows * sizeof(float) ||
        kv->bytes < (uint64_t)kv_n * rows * sizeof(float)) {
        return 0;
    }
    const void *q_w = cuda_model_range_ptr(model_map,
            q_weight_offset, q_w_bytes, "q_rms_weight");
    const void *kv_w = cuda_model_range_ptr(model_map,
            kv_weight_offset, kv_w_bytes, "kv_rms_weight");
    if (!q_w || !kv_w) return 0;
    /* The strided epilogue keeps every lane of a warp live only when the row
     * divides evenly by the 256-thread block; otherwise some lanes exit before
     * the warp-wide shuffle in pulsar_mx_emit_block.  FAIL LOUD rather than
     * silently skip the emission: the caller arms the activation cache off the
     * same slot pointer, so a skipped emission would leave the GEMM reading a
     * memset-zero E4M3 buffer -- a well-formed WRONG answer, not an error. */
    if (q_out_q && ((q_n % 256u) != 0u || (q_n % 32u) != 0u)) {
        fprintf(stderr, "pulsar: qkv rms norm cannot emit MX for q_n=%u "
                        "(needs a multiple of 256)\n", q_n);
        return 0;
    }
    /* Same shape as the emission guard above: skipping the store without an
     * encoding to replace it writes NOTHING and leaves the consumer reading
     * whatever the previous call left behind.  Refuse rather than silently
     * downgrade to storing f32, which would hide the caller's bug. */
    if (q_skip_f32 && !q_out_q) {
        fprintf(stderr, "pulsar: qkv rms norm asked to skip the q f32 store with no "
                        "E4M3 slot (q_n=%u) -- refusing\n", q_n);
        return 0;
    }
    if (q_skip_f32) {
        static pulsar_shape_once seen = {};
        if (pulsar_shape_once_first(&seen, pulsar_shape_key(q_n, 0u), "qr_norm f32 skip announce")) {
            fprintf(stderr, "pulsar: qr_norm f32 store SKIPPED (q_n=%u rows=%u, %.1f MiB)\n",
                    q_n, rows, (double)q_n * rows * sizeof(float) / (1024.0 * 1024.0));
        }
    }
    dim3 grid(rows, 2u, 1u);
#define PULSAR_QKV_NORM_LAUNCH(A, B)                    \
    dsv4_qkv_rms_norm_rows_kernel<A, B><<<grid, 256>>>( \
            q_skip_f32 ? NULL : (float *)q_out->ptr,    \
            (const float *)q->ptr,                      \
            q_w,                                        \
            q_n,                                        \
            (float *)kv_out->ptr,                       \
            (const float *)kv->ptr,                     \
            kv_w,                                       \
            kv_n,                                       \
            rows,                                       \
            eps,                                        \
            (__nv_fp8_e4m3 *)q_out_q,                   \
            (unsigned char *)q_out_sf,                  \
            q_out_kbp)
    if (q_w_bf16 && kv_w_bf16)   PULSAR_QKV_NORM_LAUNCH(true, true);
    else if (q_w_bf16)           PULSAR_QKV_NORM_LAUNCH(true, false);
    else if (kv_w_bf16)          PULSAR_QKV_NORM_LAUNCH(false, true);
    else                         PULSAR_QKV_NORM_LAUNCH(false, false);
#undef PULSAR_QKV_NORM_LAUNCH
    return cuda_ok(cudaGetLastError(), "dsv4 qkv rms norm rows launch");
}





int pulsar_gpu_head_rms_norm_rope_tail_tensor(pulsar_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float eps, const pulsar_gpu_tensor *positions) {
    if (positions && positions->bytes < (uint64_t)n_tok * sizeof(int32_t)) return 0;
    /* Derived from the buffer, never passed in.  Passing it was how decode
     * came to hand an f16 Q to the f32 kernel. */
    const size_t esz = pulsar_tensor_esz(x);
    const int q_f16 = (esz == sizeof(__half));
    if (!x || n_rot > head_dim || (n_rot & 1u) ||
        x->bytes < (uint64_t)n_tok * n_head * head_dim * esz) return 0;
    const int32_t *pos = positions ? (const int32_t *)positions->ptr : NULL;
    if (q_f16)
        head_rms_norm_rope_tail_kernel<__half><<<n_tok * n_head, 256>>>((__half *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps, pos);
    else
        head_rms_norm_rope_tail_kernel<float><<<n_tok * n_head, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps, pos);
    return cuda_ok(cudaGetLastError(), "head_rms_norm_rope_tail launch");
}






/* PULSAR_ATTN_PACK quantize+store: fp8-roundtrip the nope dims of n_rows f32 rows
 * of x IN PLACE (the single pack-store fp8 recipe) and store
 * the packed rows into `packed` at rows [out_row0, out_row0+n_rows). */
/* Row geometry, straight from the macros the kernels index with.  Deliberately
 * NOT a re-derivation: if these ever stop being the same expression the kernels
 * use, the check they exist for is worthless. */
uint64_t pulsar_gpu_attn_pack_rowbytes(uint32_t head_dim) {
    return PULSAR_ATTN_PACK_ROWBYTES(head_dim);
}

uint64_t pulsar_gpu_mxkv_fp4_rowbytes(uint32_t head_dim) {
    return PULSAR_MXKV_FP4_ROWBYTES(head_dim);
}


int pulsar_gpu_attn_pack_quantize_store_tensor(pulsar_gpu_tensor *x,
                                                       pulsar_gpu_tensor *packed,
                                                       uint32_t out_row0,
                                                       uint32_t n_rows,
                                                       uint32_t head_dim,
                                                       uint32_t n_rot,
                                                       bool keep_f32) {
    if (!x || !packed || n_rows == 0 ||
        n_rot != PULSAR_ATTN_PACK_NROT || head_dim <= n_rot ||
        head_dim != 512u ||   /* shared samax/sscale are sized for 28 blocks */
        ((head_dim - n_rot) % PULSAR_KV4_NV_BLOCK) != 0 ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float) ||
        packed->bytes < ((uint64_t)out_row0 + n_rows) *
                        PULSAR_ATTN_PACK_ROWBYTES(head_dim)) {
        return 0;
    }
    attn_pack_store_kernel<<<n_rows, 64>>>(keep_f32 ? (float *)x->ptr : NULL,
                                          (const float *)x->ptr,
                                          (uint8_t *)packed->ptr,
                                           out_row0, n_rows, head_dim, n_rot,
                                           NULL, NULL, 0u, 0u);
    return cuda_ok(cudaGetLastError(), "attn_pack_store launch");
}





/* Fused rope + QAT + PACK for the indexer q projection: rope the f32 staging
 * in place, Hadamard + E2M1-quantize, and store MXKV FP4 packed rows into
 * `packed`.  There is no dequantized output: the packed rows are the ONLY Q
 * the scorers see, so the quantized values cannot fork from what a second
 * encode would produce -- the encode happens once, here. n_rows = n_tok * n_head. */
int pulsar_gpu_dsv4_indexer_rope_qat_tensor(pulsar_gpu_tensor *x,
        pulsar_gpu_tensor *packed,
        uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
        uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, const pulsar_gpu_tensor *positions) {
    const uint32_t n_rows = n_tok * n_head;
    if (!x || !packed || n_rows == 0 || head_dim != 128u || n_rot > head_dim || (n_rot & 1) ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float) ||
        packed->bytes < (uint64_t)n_rows * PULSAR_MXKV_FP4_ROWBYTES(128u)) {
        return 0;
    }
    if (pulsar_tensor_esz(x) != sizeof(float)) return 0;   /* staging is f32 by contract */
    if (positions && positions->bytes < (uint64_t)n_tok * sizeof(int32_t)) return 0;
    indexer_rope_hadamard_fp4_pack_q_kernel<<<n_rows, 128>>>((float *)x->ptr,
            (uint8_t *)packed->ptr,
            n_rows, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse ? 1 : 0,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow,
            positions ? (const int32_t *)positions->ptr : NULL);
    return cuda_ok(cudaGetLastError(), "indexer rope+qat+pack launch");
}

/* QAT + pack: roundtrip n_rows f32 rows of x in place (the same QAT the fused
 * rope+QAT entry applies) and store the MXKV FP4 packed rows into
 * `packed` at rows [out_row0, out_row0 + n_rows). */
int pulsar_gpu_dsv4_indexer_qat_pack_tensor(pulsar_gpu_tensor *x,
                                                    pulsar_gpu_tensor *packed,
                                                    uint32_t out_row0,
                                                    uint32_t n_rows,
                                                    uint32_t head_dim,
                                                    bool keep_f32) {
    const uint64_t rowbytes = PULSAR_MXKV_FP4_ROWBYTES(128u);
    if (!x || !packed || n_rows == 0 || head_dim != 128u ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float) ||
        packed->bytes < ((uint64_t)out_row0 + n_rows) * rowbytes) {
        return 0;
    }
    indexer_hadamard_fp4_pack_kernel<<<n_rows, 128>>>(
            (float *)x->ptr,
            (uint8_t *)packed->ptr + (uint64_t)out_row0 * rowbytes,
            n_rows, head_dim, keep_f32 ? 1 : 0);
    return cuda_ok(cudaGetLastError(), "indexer_hadamard_fp4_pack launch");
}


int pulsar_gpu_rope_tail_mx_tensor(pulsar_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, const pulsar_gpu_tensor *positions,
        void *gact_data, void *gact_scale, int gact_kbp, uint32_t gact_slab, uint32_t n_groups) {
    if (!x || n_rot > head_dim || (n_rot & 1)) return 0;
    /* Derived from the buffer, never passed in -- the same rule its f16-aware
     * twin head_rms_norm_rope_tail states above, for the same reason: passing
     * the width is how decode came to hand an f16 Q to the f32 kernel.
     *
     * This used to be `esz != sizeof(float) -> refuse`, guarding an untemplated
     * kernel against a narrowed buffer whose byte bound would still pass.  The
     * kernel is templated now, so the width selects an instantiation instead of
     * rejecting the call, and the bound below is computed from that same width
     * rather than from a hardcoded sizeof(float) -- which is what made the
     * bound too weak to stand alone in the first place. */
    const size_t esz = pulsar_tensor_esz(x);
    if (esz != sizeof(float) && esz != sizeof(__nv_bfloat16)) return 0;
    if (x->bytes < (uint64_t)n_tok * n_head * head_dim * esz) return 0;
    if (positions && positions->bytes < (uint64_t)n_tok * sizeof(int32_t)) return 0;
    /* The MX epilogue owns exactly the blocks covering [head_dim - n_rot,
     * head_dim), so that range must BE whole MX blocks and a warp must map to
     * one (t, h): n_rot/2 == 32 lanes.  Refuse rather than emit a partial
     * encoding the "a" GEMM would then read as if it were complete. */
    if (gact_data && (n_groups == 0u || (n_head % n_groups) != 0u ||
                      (n_rot % 64u) != 0u || ((head_dim - n_rot) % 32u) != 0u ||
                      (head_dim % 32u) != 0u || n_rot != 64u)) {
        fprintf(stderr, "pulsar: rope_tail cannot emit MX for n_head=%u n_rot=%u head_dim=%u n_groups=%u\n",
                n_head, n_rot, head_dim, n_groups);
        return 0;
    }
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    if (esz == sizeof(float))
        rope_tail_kernel<float><<<(pairs + 255) / 256, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, 1, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, positions ? (const int32_t *)positions->ptr : NULL,
                (__nv_fp8_e4m3 *)gact_data, (unsigned char *)gact_scale, gact_kbp, gact_slab, n_groups);
    else
        rope_tail_kernel<__nv_bfloat16><<<(pairs + 255) / 256, 256>>>((__nv_bfloat16 *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, 1, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, positions ? (const int32_t *)positions->ptr : NULL,
                (__nv_fp8_e4m3 *)gact_data, (unsigned char *)gact_scale, gact_kbp, gact_slab, n_groups);
    return cuda_ok(cudaGetLastError(), "rope_tail launch");
}

int pulsar_gpu_rope_tail_tensor(pulsar_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, const pulsar_gpu_tensor *positions) {
    return pulsar_gpu_rope_tail_mx_tensor(x, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse,
                                          freq_base, freq_scale, ext_factor, attn_factor,
                                          beta_fast, beta_slow, positions, NULL, NULL, 0, 0u, 0u);
}


int pulsar_gpu_store_raw_kv_tensor(pulsar_gpu_tensor *raw_cache, const pulsar_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim);



int pulsar_gpu_store_raw_kv_tensor(pulsar_gpu_tensor *raw_cache, const pulsar_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * PULSAR_ATTN_PACK_ROWBYTES(head_dim) ||
        kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    /* x = NULL: kv is const here, so the row is packed WITHOUT the in-place
     * round-trip the decode store does. */
    attn_pack_store_kernel<<<1, 64>>>(NULL, (const float *)kv->ptr,
                                      (uint8_t *)raw_cache->ptr,
                                      row, 1u, head_dim, PULSAR_ATTN_PACK_NROT,
                                      NULL, NULL, 1u, raw_cap);
    return cuda_ok(cudaGetLastError(), "raw pack store launch");
}


/* Scatter ALREADY-PACKED rows into the ring.  Shares attn_pack_ring_slot with
 * attn_pack_store_kernel -- see the note there on why that is one function and
 * not two identical blocks; the only difference here is that this one moves
 * bytes instead of quantising.
 *
 * That difference is the point.  The prefill producer packs its chunk once
 * (attn_pack_quantize_store_tensor, which also round-trips the f32 in place),
 * and the ring store then re-quantised THAT ALREADY-ROUND-TRIPPED buffer --
 * which is the exact pattern the warning at the top of this file calls out:
 * feeding already-quantized data to a fresh quantizer takes the misround rate
 * from 1e-7 to ~5% at E4M3 scale boundaries.  The old call site argued the two
 * agree because both use the same fast-math scale; the warning says that bucket
 * is not bit-idempotent.  Copying the bytes makes the question moot -- the ring
 * gets exactly what attention read, by construction rather than by argument. */
__global__ static void attn_pack_scatter_kernel(const uint8_t *__restrict__ src, uint8_t *out,
                                                uint32_t out_row0, uint32_t n_rows,
                                                uint32_t head_dim,
                                                const int32_t *__restrict__ positions,
                                                const int32_t *__restrict__ seq_id,
                                                uint32_t n_banks, uint32_t raw_cap) {
    uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    const uint64_t dst_row = attn_pack_ring_slot(row, out_row0, raw_cap, n_banks,
                                                positions, seq_id);
    if (dst_row == ATTN_PACK_DEAD_ROW) return;   /* dead row stores nothing */
    const uint64_t rowbytes = PULSAR_ATTN_PACK_ROWBYTES(head_dim);
    const uint8_t *sr = src + (uint64_t)row * rowbytes;
    uint8_t *dr = out + dst_row * rowbytes;
    for (uint32_t b = threadIdx.x; b < (uint32_t)rowbytes; b += blockDim.x) dr[b] = sr[b];
}

int pulsar_gpu_store_raw_kv_batch_packed_tensor(pulsar_gpu_tensor *raw_cache, const pulsar_gpu_tensor *packed,
                                                uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim,
                                                const pulsar_gpu_tensor *positions, const pulsar_gpu_tensor *seq_id,
                                                uint32_t n_banks) {
    const int descr = positions != NULL || seq_id != NULL;
    if (descr &&
        (!positions || !seq_id || n_banks == 0 ||
         positions->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         seq_id->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         (uint64_t)n_banks * raw_cap > 4294967296ull)) {
        fprintf(stderr,
                "pulsar: banked packed raw store rejected: bad descriptor args "
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
    return cuda_ok(cudaGetLastError(), "raw pack scatter launch");
}

int pulsar_gpu_store_raw_kv_batch_tensor(pulsar_gpu_tensor *raw_cache, const pulsar_gpu_tensor *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim,
                                                 const pulsar_gpu_tensor *positions, const pulsar_gpu_tensor *seq_id, uint32_t n_banks) {
    if (head_dim != 512u) {
        /* attn_pack_store_kernel's shared samax/sscale are sized for 28
         * per-16 blocks (head_dim 512) and the row macros hardcode NROT=64;
         * no caller passes anything else -- refuse rather than trust that
         * (the standard the quantize_store entry already applies). */
        return 0;
    }
    /* Descriptor (banked) mode: both arrays or neither; the raw cache operand
     * is the whole bank pool (byte bound scales by n_banks) and the uint32
     * row ABI (seq*raw_cap + slot) must not overflow.  pos0 is ignored when
     * positions != NULL.  Fail-loud, like the banked attention launchers. */
    const int descr = positions != NULL || seq_id != NULL;
    if (descr &&
        (!positions || !seq_id || n_banks == 0 ||
         positions->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         seq_id->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         (uint64_t)n_banks * raw_cap > 4294967296ull)) {
        fprintf(stderr,
                "pulsar: banked raw store rejected: bad descriptor args "
                "(n_tokens=%u n_banks=%u raw_cap=%u)\n",
                n_tokens, n_banks, raw_cap);
        return 0;
    }
    const uint64_t kv_banks = descr ? n_banks : 1u;
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < kv_banks * raw_cap * PULSAR_ATTN_PACK_ROWBYTES(head_dim) ||
        kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float)) return 0;
    {
        /* One block per row, 64 threads -- the packed format needs a per-64-block
         * amax, so this cannot be the flat one-thread-per-element scatter the f32
         * ring uses.  x = NULL: kv is const on this entry. */
        attn_pack_store_kernel<<<n_tokens, 64>>>(NULL, (const float *)kv->ptr,
                                                 (uint8_t *)raw_cache->ptr,
                                                 pos0, n_tokens, head_dim, PULSAR_ATTN_PACK_NROT,
                                                 descr ? (const int32_t *)positions->ptr : NULL,
                                                 descr ? (const int32_t *)seq_id->ptr : NULL,
                                                 descr ? n_banks : 1u, raw_cap);
        return cuda_ok(cudaGetLastError(), "raw pack store batch launch");
    }
}


int pulsar_gpu_compressor_store_batch_tensor(
        const pulsar_gpu_tensor *kv,
        const pulsar_gpu_tensor *sc,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t coff = pulsar_compress_coff(ratio);
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        /* Fail loud: this reject was silent and cost an L120 debugging cycle
         * (a caller passing the wrong width reads as a generic failure). */
        fprintf(stderr, "pulsar: compressor store rejected: kv=%llu/%llu state=%llu/%llu (head_dim=%u ratio=%u)\n",
                (unsigned long long)kv->bytes, (unsigned long long)kv_bytes,
                (unsigned long long)state_kv->bytes, (unsigned long long)state_bytes,
                head_dim, ratio);
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    uint64_t n = (uint64_t)n_tokens * width;
    compressor_store_kernel<<<(n + 255) / 256, 256>>>(
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (float *)state_kv->ptr,
            (float *)state_score->ptr,
            ape,
            0,
            ape_type,
            head_dim,
            ratio,
            pos0,
            n_tokens);
    return cuda_ok(cudaGetLastError(), "compressor store launch");
}



/* L120 value-half: the ratio-4 window shift as its own entry, for the
 * rewind-time window replay (session.cpp).  Same kernel the emit path
 * runs; the replay re-runs store+shift over committed projections only. */
int pulsar_gpu_compressor_shift_ratio4_tensor(
        pulsar_gpu_tensor *state_kv,
        pulsar_gpu_tensor *state_score,
        uint32_t           head_dim) {
    if (!state_kv || !state_score || head_dim == 0) return 0;
    const uint32_t width = pulsar_compress_coff(4u) * head_dim;   /* this entry is ratio-4 only */
    const uint64_t state_bytes = 8ull * width * sizeof(float);
    if (state_kv->bytes < state_bytes || state_score->bytes < state_bytes) return 0;
    const uint64_t half = 4ull * width;
    compressor_shift_ratio4_kernel<<<(half + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr, width);
    return cuda_ok(cudaGetLastError(), "compressor ratio4 shift (replay) launch");
}



int pulsar_gpu_compressor_update_tensor(
        const pulsar_gpu_tensor *kv_cur,
        const pulsar_gpu_tensor *sc_cur,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        pulsar_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || head_dim == 0 || ratio == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) ||
        /* ds4 types: 0 = F32, 30 = BF16 (source format). */
        (norm_type != 0u && norm_type != 30u)) {
        return 0;
    }
    const uint32_t coff = pulsar_compress_coff(ratio);
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)(comp_row + (emit ? 1u : 0u)) * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const int norm_bf16 = (norm_type == 30u);
    const uint64_t norm_bytes = (uint64_t)head_dim * pulsar_w_elt_bytes(norm_bf16);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv_cur->bytes < kv_bytes || sc_cur->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (emit && comp_cache->bytes < comp_bytes)) {
        fprintf(stderr, "pulsar: compressor update rejected: comp_row=%u emit=%u head_dim=%u (comp cache %llu B, "
                        "needs %llu) -- refusing\n", comp_row, (unsigned)emit, head_dim,
                (unsigned long long)(comp_cache ? comp_cache->bytes : 0ull), (unsigned long long)comp_bytes);
        return 0;
    }
    if (!pulsar_gpu_compressor_store_batch_tensor(kv_cur, sc_cur, state_kv, state_score,
                                                 model_map, model_size, ape_offset, ape_type,
                                                 head_dim, ratio, pos, 1)) {
        return 0;
    }
    if (!emit) return 1;
    pulsar_gpu_tensor *comp_row_view = pulsar_gpu_tensor_view(
            comp_cache,
            (uint64_t)comp_row * head_dim * sizeof(float),
            (uint64_t)head_dim * sizeof(float));
    if (!comp_row_view) return 0;
    compressor_update_pool_kernel<<<(head_dim + 255) / 256, 256>>>(
            (float *)comp_row_view->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            head_dim,
            ratio);
    int ok = cuda_ok(cudaGetLastError(), "compressor update pool launch");
    if (ok) ok = pulsar_gpu_rms_norm_weight_rows_tensor(comp_row_view, comp_row_view,
                                                       model_map, model_size, norm_offset,
                                                       head_dim, 1, rms_eps, NULL, norm_bf16);
    if (ok) ok = pulsar_gpu_rope_tail_tensor(comp_row_view, 1, 1, head_dim, n_rot,
                                            pos + 1u - ratio, n_ctx_orig, false,
                                            freq_base, freq_scale, ext_factor, attn_factor,
                                            beta_fast, beta_slow, NULL);
    pulsar_gpu_tensor_free(comp_row_view);
    if (ok && ratio == 4u) {
        uint64_t half = 4ull * width;
        compressor_shift_ratio4_kernel<<<(half + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr, width);
        ok = cuda_ok(cudaGetLastError(), "compressor ratio4 shift launch");
    }
    return ok;
}


int pulsar_gpu_compressor_prefill_tensor(
        pulsar_gpu_tensor       *comp_cache,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        const pulsar_gpu_tensor *kv,
        const pulsar_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) ||
        /* ds4 types: 0 = F32, 30 = BF16 (source format). */
        (norm_type != 0u && norm_type != 30u)) {
        return 0;
    }

    const uint32_t coff = pulsar_compress_coff(ratio);
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const int norm_bf16 = (norm_type == 30u);
    const uint64_t norm_bytes = (uint64_t)head_dim * pulsar_w_elt_bytes(norm_bf16);

    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (n_comp && comp_cache->bytes < comp_bytes)) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;

    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor state score fill launch")) return 0;

    if (ratio == 4u) {
        if (cutoff >= ratio) {
            uint32_t prev_start = cutoff - ratio;
            uint64_t n = (uint64_t)ratio * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    prev_start, 0, ratio);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill prev state launch")) return 0;
        }
        if (rem != 0) {
            uint64_t n = (uint64_t)rem * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    cutoff, ratio, rem);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill rem state launch")) return 0;
        }
    } else if (rem != 0) {
        uint64_t n = (uint64_t)rem * width;
        compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr,
                (const float *)kv->ptr, (const float *)sc->ptr,
                ape, 0, ape_type, width, ratio, pos0,
                cutoff, 0, rem);
        if (!cuda_ok(cudaGetLastError(), "compressor prefill rem state launch")) return 0;
    }
    if (n_comp != 0) {
        dim3 grid((head_dim + 255) / 256, n_comp, 1);
        compressor_prefill_pool_kernel<<<grid, 256>>>(
                (float *)comp_cache->ptr,
                (const float *)kv->ptr,
                (const float *)sc->ptr,
                (const float *)state_kv->ptr,
                (const float *)state_score->ptr,
                ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 0);
        if (!cuda_ok(cudaGetLastError(), "compressor prefill pool launch")) return 0;
        if (!pulsar_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                                   model_map, model_size, norm_offset,
                                                   head_dim, n_comp, rms_eps, NULL, norm_bf16)) return 0;
        if (n_rot != 0) {
            const uint32_t pairs = n_comp * (n_rot / 2u);
            rope_tail_kernel<<<(pairs + 255) / 256, 256>>>(
                    (float *)comp_cache->ptr, n_comp, 1, head_dim, n_rot,
                    pos0, ratio, n_ctx_orig, 0, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow, NULL,
                    NULL, NULL, 0, 0u, 0u);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill rope launch")) return 0;
        }
    }
    return 1;
}


int pulsar_gpu_compressor_prefill_ratio4_replay_tensor(
        pulsar_gpu_tensor       *comp_cache,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        const pulsar_gpu_tensor *kv,
        const pulsar_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || n_tokens == 0 || (n_tokens & 3u) != 0 || (pos0 & 3u) != 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) ||
        /* ds4 types: 0 = F32, 30 = BF16 (source format). */
        (norm_type != 0u && norm_type != 30u)) {
        return 0;
    }

    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const int norm_bf16 = (norm_type == 30u);
    const uint64_t norm_bytes = (uint64_t)head_dim * pulsar_w_elt_bytes(norm_bf16);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        comp_cache->bytes < comp_bytes) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    dim3 grid((head_dim + 255) / 256, n_comp, 1);
    compressor_prefill_pool_kernel<<<grid, 256>>>(
            (float *)comp_cache->ptr,
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 1);
    if (!cuda_ok(cudaGetLastError(), "compressor replay pool launch")) return 0;
    if (!pulsar_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                               model_map, model_size, norm_offset,
                                               head_dim, n_comp, rms_eps, NULL, norm_bf16)) return 0;
    if (n_rot != 0) {
        const uint32_t pairs = n_comp * (n_rot / 2u);
        rope_tail_kernel<<<(pairs + 255) / 256, 256>>>(
                (float *)comp_cache->ptr, n_comp, 1, head_dim, n_rot,
                pos0, ratio, n_ctx_orig, 0, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow, NULL,
                NULL, NULL, 0, 0u, 0u);
        if (!cuda_ok(cudaGetLastError(), "compressor replay rope launch")) return 0;
    }

    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor replay state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor replay state score fill launch")) return 0;
    uint32_t prev_start = n_tokens - ratio;
    uint64_t n = (uint64_t)ratio * width;
    compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv->ptr, (const float *)sc->ptr,
            ape, 0, ape_type, width, ratio, pos0,
            prev_start, 0, ratio);
    return cuda_ok(cudaGetLastError(), "compressor replay state launch");
}


int pulsar_gpu_compressor_prefill_state_ratio4_tensor(
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        const pulsar_gpu_tensor *kv_tail,
        const pulsar_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_full,
        uint32_t                rem) {
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t ratio = 4u;
    /* The tail is the last complete group (n_full = 4 rows, or 0 when the
     * chunk has none) followed by the partial group (rem = 0..3 rows), in
     * position order.  pos0 is the position of tail row 0 and must be
     * ratio-aligned: the layout below puts partial row r in state row 4 + r
     * and compressor_store_kernel puts a decode token in row 4 + pos % 4, so
     * the two agree only when r == phase.  The caller derives pos0 from an
     * aligned chunk start; an unaligned one is a caller bug, refused here
     * rather than laid out wrong (L168). */
    if ((n_full != 0u && n_full != ratio) || rem >= ratio || n_full + rem == 0u) {
        fprintf(stderr, "pulsar: compressor state rebuild: n_full=%u rem=%u is not a ratio-4 tail "
                        "(4 or 0 complete rows + 0..3 partial rows) -- refusing\n", n_full, rem);
        return 0;
    }
    if ((pos0 % ratio) != 0u) {
        fprintf(stderr, "pulsar: compressor state rebuild: tail position %u is not ratio-aligned -- "
                        "refusing (partial rows are laid out by phase)\n", pos0);
        return 0;
    }
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_tail = n_full + rem;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t tail_bytes = (uint64_t)n_tail * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv_tail->bytes < tail_bytes || sc_tail->bytes < tail_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor state score fill launch")) return 0;
    /* Same two placements pulsar_gpu_compressor_prefill_tensor makes: the
     * complete group at rows 0..3, the partial rows at 4 + phase. */
    if (n_full != 0u) {
        uint64_t n = (uint64_t)n_full * width;
        compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr,
                (const float *)kv_tail->ptr, (const float *)sc_tail->ptr,
                ape, 0, ape_type, width, ratio, pos0,
                0, 0, n_full);
        if (!cuda_ok(cudaGetLastError(), "compressor state set launch (complete group)")) return 0;
    }
    if (rem != 0u) {
        uint64_t n = (uint64_t)rem * width;
        compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr,
                (const float *)kv_tail->ptr, (const float *)sc_tail->ptr,
                ape, 0, ape_type, width, ratio, pos0,
                n_full, ratio, rem);
        if (!cuda_ok(cudaGetLastError(), "compressor state set launch (partial group)")) return 0;
    }
    return 1;
}



/* Read n_elems ELEMENTS of t, starting at elem_off, to the host as f32,
 * widening from the tensor's stored format (f32 / f16 / bf16).  A BYTES tensor
 * is refused: a narrowed buffer reinterpreted as f32 is plausible wrong
 * numbers, and this reader is the last stop before a human looks at them. */
int pulsar_gpu_tensor_read_f32(const pulsar_gpu_tensor *t, uint64_t elem_off,
                               float *out, uint64_t n_elems) {
    if (!t || !t->ptr || !out || n_elems == 0) return 0;
    const uint32_t esz = pulsar_tensor_esz(t);
    const pulsar_elt_fmt fmt = pulsar_tensor_fmt(t);
    if (t->bytes < (elem_off + n_elems) * esz) return 0;
    /* L106 K15: dispatch on the FORMAT, not the size -- esz==2 is ambiguous
     * between __half and __nv_bfloat16, and this reader used to resolve it as
     * bf16 unconditionally, so a routed f16 buffer would have decoded as
     * plausible wrong numbers.  BYTES (packed rows, int payloads) is refused
     * loudly: there is no widening that means anything. */
    if (fmt == PULSAR_ELT_BYTES) {
        fprintf(stderr, "pulsar: tensor_read_f32 refused: PULSAR_ELT_BYTES tensor "
                        "(packed/opaque rows are not widenable)\n");
        return 0;
    }
    if (fmt == PULSAR_ELT_F32)
        return pulsar_gpu_tensor_read(t, elem_off * esz, out, n_elems * sizeof(float)) != 0;
    if (fmt == PULSAR_ELT_F16) {
        __half *tmp16 = (__half *)malloc((size_t)n_elems * sizeof(*tmp16));
        if (!tmp16) return 0;
        if (pulsar_gpu_tensor_read(t, elem_off * esz, tmp16, n_elems * sizeof(*tmp16)) == 0) {
            free(tmp16); return 0;
        }
        for (uint64_t i = 0; i < n_elems; i++) out[i] = __half2float(tmp16[i]);
        free(tmp16);
        return 1;
    }
    if (esz != sizeof(__nv_bfloat16)) return 0;
    /* Widen on the host: this runs only behind the dump/range-sweep env gates,
     * so a staging buffer here costs nothing anyone measures, and it keeps the
     * conversion in the one place that knows the stored type. */
    __nv_bfloat16 *tmp = (__nv_bfloat16 *)malloc((size_t)n_elems * sizeof(*tmp));
    if (!tmp) return 0;
    if (pulsar_gpu_tensor_read(t, elem_off * esz, tmp, n_elems * sizeof(*tmp)) == 0) {
        free(tmp); return 0;
    }
    for (uint64_t i = 0; i < n_elems; i++) out[i] = __bfloat162float(tmp[i]);
    free(tmp);
    return 1;
}


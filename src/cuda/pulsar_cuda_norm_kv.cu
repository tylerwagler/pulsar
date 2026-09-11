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


/* One rope rotation pair, in place at tail[i], tail[i+1], for the callers that
 * rotate a stored tail (rope_tail_kernel and the fused indexer rope+pack
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


/* ===== INVARIANT: QUANTISE ONCE; EVERY LATER MOVE IS A BYTE MOVE ============
 * The three quantisers (winkv_pack, mainkv_pack in pulsar_cuda_kvrows.cu and
 * indexer_fp4_pack_row_dev below) derive their E8M0 scales from the fp32
 * product's bit fields (pulsar_e8m0_round_up -- the reference's
 * fast_round_scale) and divide with __fdiv_rn, so they are exact at the
 * boundaries whatever the TU's fast-math flags.  The invariant that matters is
 * about their INPUT, not their arithmetic:
 *
 * RE-ENCODING IS CATASTROPHIC: quantised values live on a dyadic lattice.
 * After E2M1, block values are {0,.5,1,1.5,2,3,4,6}*scale, so a block whose max
 * sits on the top code has `amax == 6*scale` EXACTLY and its re-derived scale
 * moves; ~1/3 of re-encoded FP4 blocks land on such a point (~5% for E4M3's
 * 448).  A competing GB10 fork shipped exactly this bug and lost 10.5% of
 * their fp4 indexer lanes to zero.
 *
 * MEASURED, not theoretical.  On 2026-08-18 three KV paths were found
 * quantising the same rows twice (prefill's ring store re-quantised the buffer
 * the pack had just round-tripped; the draft batch and the drafter seed did the
 * same).  Removing the second pass CHANGED THE BYTES on a 5530-token prompt
 * (27.788 -> 27.321) and cost 2.9% of decode acceptance, and it was still
 * worth it -- the ring now holds what attention actually read.
 *
 * THE RULE: never re-encode.  MOVE PACKED BYTES.  Every KV path in the engine
 * does -- prefill's ring scatter (winkv_scatter_kernel), session save/load and
 * bank snapshots (payload / KVB2 versions), fork and evict/restore, the
 * drafter seed.  There is no exact-re-encode safety net and no conversion
 * loader; if a future restore path genuinely cannot move bytes, it needs a
 * design, not a call into the quantisers above.
 * ============================================================================ */


/* The indexer's FP4 row (L218, DeepSeek-V4.1): the reference's
 * fp4_act_quant(x, 32, scale_dtype=e8m0) on a bf16 row -- per 32-element
 * block, amax floored at 6 * 2^-126, scale 2^ceil(log2(amax * (1/6))), values
 * clamped to +-6 and rounded to E2M1, the scale stored as one E8M0 byte.  The
 * value is rounded to bf16 first: the reference quantises the bf16 tensor the
 * projection (fp8 GEMM, bf16 out) and RoPE (fp32 math, copied back to bf16)
 * left, and our f32 staging carries those steps unrounded.  0731's 128-point
 * Hadamard rotation before the quant is gone with that checkpoint.
 *
 * Both kernels below run this ONE copy (the packing kernel's contract is a
 * bit-identical row to the rope-fused one; two hand-kept copies is how such a
 * contract rots).  Called by all 128 threads of the block after the kernels'
 * own bounds check -- the __syncthreads() are uniform. */
struct indexer_fp4_t {
    float    v;           ///< this thread's bf16-rounded value
    uint32_t fp4_block;   ///< which 32-wide FP4 block it lands in
    uint32_t lane;        ///< its lane within that block
    uint32_t block_base;  ///< absbuf[block_base] holds the block's absmax
};

__device__ static inline indexer_fp4_t indexer_block_absmax_dev(const float *xr, uint32_t tid, float *absbuf) {
    const float v = __bfloat162float(__float2bfloat16(xr[tid]));
    const uint32_t fp4_block = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t block_base = fp4_block * 32u;
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

__device__ static inline void indexer_fp4_pack_row_dev(const indexer_fp4_t &h, const float *absbuf,
                                                       uint8_t *nib_sh, uint8_t *outr, uint32_t tid,
                                                       float *keep_f32_slot) {
    const float amax = fmaxf(absbuf[h.block_base], 7.052966104933725e-38f);   /* 6 * 2^-126 */
    /* fast_round_scale(amax, 1/6): the product in fp32, the exponent from its
     * bit fields (pulsar_cuda_internal.h); IEEE division for the RNE ties */
    const uint32_t e8 = pulsar_e8m0_round_up(amax * (1.0f / 6.0f));
    const float scale = pulsar_e8m0_scale(e8);
    const uint8_t nib = dsv4_e2m1fn_encode_dev(fminf(6.0f, fmaxf(-6.0f, __fdiv_rn(h.v, scale))));
    /* The dequantised writeback is for OBSERVERS only -- the packed rows are
     * what every consumer reads (L094). */
    if (keep_f32_slot) *keep_f32_slot = dsv4_e2m1fn_decode_dev(nib, scale);
    nib_sh[tid] = nib;
    __syncthreads();
    if (tid < 64u) outr[tid] = (uint8_t)(nib_sh[2u * tid] | (nib_sh[2u * tid + 1u] << 4));
    if (h.lane == 0u) outr[64u + h.fp4_block] = (uint8_t)e8;
}

/* Fused indexer-q epilogue: rope the tail, then the FP4 pack, one 128-thread
 * block per (token, head) row -- the rotation is the SAME device function
 * rope_tail_kernel runs, and the __syncthreads between the phases stands in
 * for a kernel boundary (one block owns the whole row).  This kernel writes no
 * f32 back: x is producer-internal rope staging and every consumer reads the
 * packed row, so there is exactly one Q operand encoding (L090.4). */
__global__ static void indexer_rope_fp4_pack_q_kernel(
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
    __shared__ float absbuf[128];
    __shared__ uint8_t nib_sh[128];
    const indexer_fp4_t h = indexer_block_absmax_dev(xr, tid, absbuf);
    indexer_fp4_pack_row_dev(h, absbuf, nib_sh, out + (uint64_t)row * PULSAR_MXKV_FP4_ROWBYTES(128u), tid, NULL);
}

/* The same pack minus the rope (the kv source's index-K rows, already rotated
 * at their group positions), emitting MXKV FP4 rows -- E2M1 nibble pairs
 * low-nibble-first, then one E8M0 byte per 32-block. */
__global__ static void indexer_fp4_pack_kernel(float *x, uint8_t *out,
                                               uint32_t n_rows, uint32_t head_dim,
                                               int keep_f32) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;
    __shared__ float absbuf[128];
    __shared__ uint8_t nib_sh[128];
    float *xr = x + (uint64_t)row * head_dim;
    const indexer_fp4_t h = indexer_block_absmax_dev(xr, tid, absbuf);
    indexer_fp4_pack_row_dev(h, absbuf, nib_sh, out + (uint64_t)row * PULSAR_MXKV_FP4_ROWBYTES(128u), tid,
                             keep_f32 ? &xr[tid] : NULL);
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


/* Rope the indexer Q rows' tails in place and store their FP4 rows into
 * `packed`.  There is no dequantized output: the packed rows are the ONLY Q
 * the scorers see, so the quantized values cannot fork from what a second
 * encode would produce -- the encode happens once, here. n_rows = n_tok * n_head. */
int pulsar_gpu_indexer_rope_fp4_pack_tensor(pulsar_gpu_tensor *x,
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
    indexer_rope_fp4_pack_q_kernel<<<n_rows, 128>>>((float *)x->ptr,
            (uint8_t *)packed->ptr,
            n_rows, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse ? 1 : 0,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow,
            positions ? (const int32_t *)positions->ptr : NULL);
    return cuda_ok(cudaGetLastError(), "indexer rope+fp4 pack launch");
}

/* Pack n_rows f32 rows of x (the same quant the fused rope entry applies) into
 * MXKV FP4 rows of `packed` at [out_row0, out_row0 + n_rows). */
int pulsar_gpu_indexer_fp4_pack_tensor(pulsar_gpu_tensor *x,
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
    indexer_fp4_pack_kernel<<<n_rows, 128>>>(
            (float *)x->ptr,
            (uint8_t *)packed->ptr + (uint64_t)out_row0 * rowbytes,
            n_rows, head_dim, keep_f32 ? 1 : 0);
    return cuda_ok(cudaGetLastError(), "indexer fp4 pack launch");
}


int pulsar_gpu_rope_tail_mx_tensor(pulsar_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, const pulsar_gpu_tensor *positions,
        void *gact_data, void *gact_scale, int gact_kbp, uint32_t gact_slab, uint32_t n_groups) {
    if (!x || n_rot > head_dim || (n_rot & 1)) return 0;
    /* Derived from the buffer, never passed in -- the same rule its f16-aware
     * twin (0731's head_rms_norm_rope_tail) stated, for the same reason: passing
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


/* CSA2 (L218): rotate `n_rows` single-head f32 rows whose positions step by
 * `pos_stride` -- the compressed rows of a kv source, where row k stands for
 * the first token of its group at position pos0 + k * ratio (the reference:
 * `freqs_cis[: seqlen - seqlen % ratio : ratio]`). */
int pulsar_gpu_rope_tail_strided_tensor(pulsar_gpu_tensor *x, uint32_t n_rows, uint32_t head_dim, uint32_t n_rot,
                                        uint32_t pos0, uint32_t pos_stride, uint32_t n_ctx_orig,
                                        float freq_base, float freq_scale, float ext_factor, float attn_factor,
                                        float beta_fast, float beta_slow) {
    if (!x || n_rows == 0 || pos_stride == 0 || n_rot == 0 || (n_rot & 1u) || n_rot > head_dim ||
        pulsar_tensor_esz(x) != sizeof(float) || x->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        fprintf(stderr, "pulsar: strided rope tail: bad operands (rows %u, head_dim %u, n_rot %u, stride %u) -- refusing\n",
                n_rows, head_dim, n_rot, pos_stride);
        return 0;
    }
    const uint32_t pairs = n_rows * (n_rot / 2);
    rope_tail_kernel<float><<<(pairs + 255) / 256, 256>>>((float *)x->ptr, n_rows, 1u, head_dim, n_rot, pos0, pos_stride,
                                                          n_ctx_orig, 0, freq_base, freq_scale, ext_factor, attn_factor,
                                                          beta_fast, beta_slow, NULL, NULL, NULL, 0, 0u, 0u);
    return cuda_ok(cudaGetLastError(), "strided rope tail launch");
}


/* The raw-ring writers.  Every ring row is a WINDOW row (pulsar_gpu.h); the
 * quantise happens in pulsar_gpu_winkv_pack_tensor exactly once, and rows
 * already packed are moved as bytes.  Destination slot = the shared
 * pulsar_kv_ring_slot rule. */
int pulsar_gpu_store_raw_kv_tensor(pulsar_gpu_tensor *raw_cache, const pulsar_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 || row >= raw_cap ||
        raw_cache->bytes < (uint64_t)raw_cap * PULSAR_WINKV_ROWBYTES(head_dim) ||
        kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    /* x = NULL: kv is const here; the ring slot is pos % raw_cap with pos = row */
    return pulsar_gpu_winkv_pack_tensor(NULL, kv, raw_cache, row, 1u, head_dim, NULL, NULL, 1u, raw_cap);
}


/* Scatter ALREADY-PACKED window rows into the ring: the prefill producer packs
 * its chunk once (pulsar_gpu_winkv_pack_tensor into the pack buffer attention
 * reads), and the ring then receives THOSE bytes -- never a second quantise of
 * the same values (re-encoding a decoded row is not bit-idempotent at scale
 * boundaries; the ring must hold exactly what attention read). */
__global__ static void winkv_scatter_kernel(const uint8_t *__restrict__ src, uint8_t *out,
                                            uint32_t out_row0, uint32_t n_rows, uint32_t head_dim,
                                            const int32_t *__restrict__ positions,
                                            const int32_t *__restrict__ seq_id,
                                            uint32_t n_banks, uint32_t raw_cap) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    const uint64_t dst_row = pulsar_kv_ring_slot(row, out_row0, raw_cap, n_banks, positions, seq_id);
    if (dst_row == PULSAR_KV_RING_DEAD_ROW) return;   /* dead row stores nothing */
    const uint64_t rowbytes = PULSAR_WINKV_ROWBYTES(head_dim);
    const uint8_t *sr = src + (uint64_t)row * rowbytes;
    uint8_t *dr = out + dst_row * rowbytes;
    for (uint32_t b = threadIdx.x; b < (uint32_t)rowbytes; b += blockDim.x) dr[b] = sr[b];
}

/* Descriptor (banked) mode: both arrays or neither; the raw cache operand is
 * the whole bank pool (byte bound scales by n_banks) and the uint32 row ABI
 * (seq*raw_cap + slot) must not overflow.  pos0 is ignored when positions !=
 * NULL.  Fail-loud, like the banked attention launchers. */
static bool raw_store_descr_ok(const pulsar_gpu_tensor *positions, const pulsar_gpu_tensor *seq_id,
                               uint32_t n_tokens, uint32_t n_banks, uint32_t raw_cap, const char *what) {
    const bool descr = positions != NULL || seq_id != NULL;
    if (descr &&
        (!positions || !seq_id || n_banks == 0 ||
         positions->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         seq_id->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         (uint64_t)n_banks * raw_cap > 4294967296ull)) {
        fprintf(stderr, "pulsar: %s rejected: bad descriptor args (n_tokens=%u n_banks=%u raw_cap=%u)\n",
                what, n_tokens, n_banks, raw_cap);
        return false;
    }
    return true;
}

int pulsar_gpu_store_raw_kv_batch_packed_tensor(pulsar_gpu_tensor *raw_cache, const pulsar_gpu_tensor *packed,
                                                uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim,
                                                const pulsar_gpu_tensor *positions, const pulsar_gpu_tensor *seq_id,
                                                uint32_t n_banks) {
    if (!raw_store_descr_ok(positions, seq_id, n_tokens, n_banks, raw_cap, "banked packed raw store")) return 0;
    const bool descr = positions != NULL;
    const uint64_t kv_banks = descr ? n_banks : 1u;
    const uint64_t rowbytes = PULSAR_WINKV_ROWBYTES(head_dim);
    if (!raw_cache || !packed || raw_cap == 0 || (head_dim % PULSAR_WINKV_BLOCK) != 0u ||
        raw_cache->bytes < kv_banks * raw_cap * rowbytes ||
        packed->bytes < (uint64_t)n_tokens * rowbytes) return 0;
    if (n_tokens == 0) return 1;
    winkv_scatter_kernel<<<n_tokens, 64>>>((const uint8_t *)packed->ptr, (uint8_t *)raw_cache->ptr,
                                           pos0, n_tokens, head_dim,
                                           descr ? (const int32_t *)positions->ptr : NULL,
                                           descr ? (const int32_t *)seq_id->ptr : NULL,
                                           descr ? n_banks : 1u, raw_cap);
    return cuda_ok(cudaGetLastError(), "window row scatter launch");
}

int pulsar_gpu_store_raw_kv_batch_tensor(pulsar_gpu_tensor *raw_cache, const pulsar_gpu_tensor *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim,
                                         const pulsar_gpu_tensor *positions, const pulsar_gpu_tensor *seq_id, uint32_t n_banks) {
    if (!raw_store_descr_ok(positions, seq_id, n_tokens, n_banks, raw_cap, "banked raw store")) return 0;
    const bool descr = positions != NULL;
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (descr ? n_banks : 1u) * (uint64_t)raw_cap * PULSAR_WINKV_ROWBYTES(head_dim) ||
        kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float)) return 0;
    if (n_tokens == 0) return 1;
    /* x = NULL: kv is const on this entry */
    return pulsar_gpu_winkv_pack_tensor(NULL, kv, raw_cache, pos0, n_tokens, head_dim,
                                        positions, seq_id, descr ? n_banks : 1u, raw_cap);
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


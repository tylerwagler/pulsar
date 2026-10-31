/** @file
 *  Qwen3.8-Flash-Next Gated Residual: the 4-stream read, the write and the collapse-only mixer. */
/* Qwen3.8-Flash-Next Gated Residual (L251 S4): the 4-stream read, the write,
 * and the collapse-only top-level mixer.  Source: transformers'
 * Qwen4ExpTextGatedResidual; contracts in pulsar_cuda_qwen.h.
 *
 * The read, per token (4 streams x 2560, bf16 storage), three launches:
 *   1. norm    xn_s = x_s * rsqrt(mean(x_s^2) + eps) * (1 + w_s)   one CTA per
 *              (stream, token); xn leaves in the format W_down reads and the
 *              per-stream rstd is kept, so the last kernel recomputes xn with
 *              the same three operations instead of storing it; the inject
 *              dot W_inj . xn is taken here on the f32 xn as per-stream partials.
 *              At prefill widths (3') the down GEMM stages xn from the streams with
 *              the same three operations, so the norm does not write it there
 *   2. down    d = W_down xn (320 x 10240), split-K in 5 fixed splits (L288: on the tensor
 *              cores at decode widths too -- the rows kernel, W read once for <= 16 rows)
 *   3. up      mid: a = silu(d / 4) in shared memory, in the format W_up reads
 *              (each CTA derives it for its tokens), inj = 2 sigmoid(z / 4) with
 *              z the partials summed in stream order; then for each channel c
 *              the four rows s * 2560 + c of W_up (10240 x 320) in one CTA:
 *              g_s = sigmoid(W_up a), x = (g_0 xn_0 + g_1 xn_1 + g_2 xn_2 +
 *              g_3 xn_3) / 4, rounded to bf16 (the source's `mixed_input`
 *              dtype), emitted as the block input's bf16 row AND (when a slot
 *              is given) its E4M3 encoding -- one value, the encoding of the bf16
 * The write: streams_s += out * inj_s, f32 math, one bf16 rounding.
 *
 * Two low-rank weight formats (pulsar_qwen_lowrank): MXFP8 at every per-layer
 * site (xn and a are E4M3, the products are exact, the sums f32) and BF16 at
 * the top-level mixer, as the graded recipe stores it (xn and a are bf16 --
 * the source's own dtype there -- and the GEMVs are bf16 x bf16 into f32).
 * One template parameter, W8, picks the pair; the norm, the gate, the mean
 * and the emit are one code.
 *
 * Fusion.  The paper runs the read and the write as one kernel each; the read
 * here is three launches because the two low-rank GEMVs need the whole token
 * row and the whole weight, which one CTA cannot hold.  The down GEMV is the
 * byte cost (3.3 MB of MXFP8 per site).  Fusing the write with the next site's
 * norm (they touch the same row back to back) is the next step, measured in.
 */
#include "pulsar_cuda_qwen.h"
#include "pulsar_cuda_mx.cuh"
#include "pulsar_cuda_arch.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdio.h>

namespace {

/// Hidden width of one residual stream (2560).
constexpr int kH = PULSAR_QWEN_HIDDEN;
/// Residual streams per token (4).
constexpr int kS = PULSAR_QWEN_HC;
/// Width of the whole stream stack, kS * kH (10240): W_down's input, W_up's output.
constexpr int kHC = PULSAR_QWEN_HC_HIDDEN;
/// The Gated Residual's low rank (320): W_down's output, W_up's input.
constexpr int kR = PULSAR_QWEN_HC_LOWRANK;
constexpr float kEps = 1e-6f;                 ///< rms_norm_eps
/// Block size of the norm kernel (one CTA per (stream, token)).
constexpr int kNormThreads = 256;
constexpr int kNormPer = kH / kNormThreads;   ///< 10 columns per thread
constexpr int kDownSplit = 5;                 ///< W_down split-K: 320 blocks of 32 -> 64 per split
constexpr int kDownTB = 8;                    ///< tokens per down CTA
constexpr int kDownStageBytes = 48 * 1024;    ///< the W8A16 GEMV's staged activation slice, at most
constexpr int kUpTB = 16;                     ///< tokens per up CTA, BF16 mixer at prefill widths
constexpr int kUpTBDecode = 8;                ///< tokens per up CTA at decode widths: a verify of <= 8 rows reads
                                              ///< W_up once (L288 C2; it was 4), and still small enough that two
                                              ///< CTAs (the staged W_up tile + a; the W8 gated products reuse the
                                              ///< tile's memory) share an SM, so the 80 CTAs are one wave on 48
                                              ///< SMs.  The row arithmetic does not depend on the tile, so every
                                              ///< tile width is bit-identical.
static_assert(kH % kNormThreads == 0, "the norm walks a stream in whole passes of 256");
static_assert((kHC / 32) % kDownSplit == 0, "W_down's 32-blocks split evenly");
static_assert(kR % 32 == 0 && kR / 32 <= 12, "the low rank is whole 32-blocks");

/** bf16 to f32.
 *  @param v the bf16 value
 *  @return v widened to f32 (exact) */
__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }

/** Unpack 8 bf16 (one 16-byte chunk, element order) to f32.
 *  @param u the chunk
 *  @param f out: the 8 values */
__device__ __forceinline__ void bf16x8(const uint4 &u, float f[8]) {
    const __nv_bfloat162 *p = reinterpret_cast<const __nv_bfloat162 *>(&u);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 v = __bfloat1622float2(p[j]);
        f[2 * j] = v.x;
        f[2 * j + 1] = v.y;
    }
}
/* 8 bf16 . 8 bf16 accumulated in element order */
/** @param w   8 bf16 weights (one chunk)
 *  @param x   8 bf16 activations (one chunk)
 *  @param acc running sum
 *  @return acc plus the 8 products, fmaf in element order */
__device__ __forceinline__ float bf16_dot8(const uint4 &w, const uint4 &x, float acc) {
    float wf[8], xf[8];
    bf16x8(w, wf);
    bf16x8(x, xf);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc = fmaf(wf[j], xf[j], acc);
    return acc;
}

/* 32 E4M3 . 32 E4M3 in f32, element order 0..31: every product is exact (4 x 4
 * significant bits) and the sum rounds in a fixed order. */
/** @param w0 weight codes 0..15
 *  @param w1 weight codes 16..31
 *  @param x0 activation codes 0..15
 *  @param x1 activation codes 16..31
 *  @return the unscaled 32-element dot in f32 */
__device__ __forceinline__ float e4m3_dot32(const uint4 &w0, const uint4 &w1, const uint4 &x0, const uint4 &x1) {
    const uint32_t *wa = reinterpret_cast<const uint32_t *>(&w0), *wb = reinterpret_cast<const uint32_t *>(&w1);
    const uint32_t *xa = reinterpret_cast<const uint32_t *>(&x0), *xb = reinterpret_cast<const uint32_t *>(&x1);
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t wv = i < 4 ? wa[i] : wb[i - 4];
        const uint32_t xv = i < 4 ? xa[i] : xb[i - 4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const __half2_raw wr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(wv >> (16 * h)), __NV_E4M3);
            const __half2_raw xr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(xv >> (16 * h)), __NV_E4M3);
            const float2 wf = __half22float2(*reinterpret_cast<const __half2 *>(&wr));
            const float2 xf = __half22float2(*reinterpret_cast<const __half2 *>(&xr));
            s = fmaf(wf.x, xf.x, s);
            s = fmaf(wf.y, xf.y, s);
        }
    }
    return s;
}

/* 2^(e8m0_w + e8m0_x - 254), exact */
/** @param sw the weight block's E8M0 byte
 *  @param sx the activation block's E8M0 byte
 *  @return 2^(sw + sx - 254) */
__device__ __forceinline__ float mx_scale2(unsigned sw, unsigned sx) {
    return exp2f((float)((int)sw + (int)sx - 254));
}

/* One E8M0 byte as its scale: 2^(b-127).  mx_scale2's single-operand form. */
/** @param sb an E8M0 byte
 *  @return 2^(sb - 127) */
__device__ __forceinline__ float mx_scale1(unsigned sb) {
    return exp2f((float)((int)sb - 127));
}

/* 32 E4M3 weight codes against 32 BF16 activations (L251 / ac69748f, the W8A16 shape).
 * The W8 weights arrive exactly as they do in e4m3_dot32 above (2 uint4 = 32 bytes = 32 codes);
 * the activation is the block input's bf16 row, so 32 of them are 4 uint4 (2 bf16 per uint32).
 * The scale is the weight's E8M0 alone -- bf16 carries its own exponent, so there is no second
 * block scale to fold in.  The E4M3 decode is the same `__nv_cvt_fp8x2_to_halfraw2` the A8 path
 * uses, so the weights are read bit-identically to W8A8; only the activation operand changed. */
/** @param w0 weight codes 0..15
 *  @param w1 weight codes 16..31
 *  @param b0 bf16 activations 0..7
 *  @param b1 bf16 activations 8..15
 *  @param b2 bf16 activations 16..23
 *  @param b3 bf16 activations 24..31
 *  @return the dot in f32, unscaled by the weight's E8M0 (fmaf in element order) */
__device__ __forceinline__ float e4m3_bf16_dot32(const uint4 &w0, const uint4 &w1,
                                                const uint4 &b0, const uint4 &b1,
                                                const uint4 &b2, const uint4 &b3) {
    const uint32_t *wa = reinterpret_cast<const uint32_t *>(&w0);
    const uint32_t *wb = reinterpret_cast<const uint32_t *>(&w1);
    const uint32_t *xa = reinterpret_cast<const uint32_t *>(&b0);
    const uint32_t *xb = reinterpret_cast<const uint32_t *>(&b1);
    const uint32_t *xc = reinterpret_cast<const uint32_t *>(&b2);
    const uint32_t *xd = reinterpret_cast<const uint32_t *>(&b3);
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        /* weight word i holds E4M3 codes [4i, 4i+4); the matching bf16 are words 2i and 2i+1 */
        const uint32_t wv = i < 4 ? wa[i] : wb[i - 4];
        const int j = 2 * i;
        const uint32_t xv = j < 4 ? xa[j] : (j < 8 ? xb[j - 4] : (j < 12 ? xc[j - 8] : xd[j - 12]));
        const uint32_t yv = (j + 1) < 4 ? xa[j + 1] : ((j + 1) < 8 ? xb[j + 1 - 4]
                          : ((j + 1) < 12 ? xc[j + 1 - 8] : xd[j + 1 - 12]));
        const float2 a0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&xv));
        const float2 a1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&yv));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const __half2_raw wr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(wv >> (16 * h)), __NV_E4M3);
            const float2 wf = __half22float2(*reinterpret_cast<const __half2 *>(&wr));
            const float2 af = h == 0 ? a0 : a1;
            s = fmaf(wf.x, af.x, s);
            s = fmaf(wf.y, af.y, s);
        }
    }
    return s;
}

/* L288 C2: e4m3_bf16_dot32 in two halves -- the 32 E4M3 codes decoded once (the same conversions, element order
 * 0..31), then applied to any number of bf16 activation rows: the same 32 fmaf in the same order, so a row's dot
 * is bit-identical to e4m3_bf16_dot32's. */
/** @param w0 codes 0..15
 *  @param w1 codes 16..31
 *  @param wf out: the 32 decoded values (unscaled) */
__device__ __forceinline__ void e4m3x32_f32(const uint4 &w0, const uint4 &w1, float (&wf)[32]) {
    const uint32_t *wa = reinterpret_cast<const uint32_t *>(&w0), *wb = reinterpret_cast<const uint32_t *>(&w1);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t wv = i < 4 ? wa[i] : wb[i - 4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const __half2_raw wr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(wv >> (16 * h)), __NV_E4M3);
            const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&wr));
            wf[4 * i + 2 * h] = f.x;
            wf[4 * i + 2 * h + 1] = f.y;
        }
    }
}
/** The second half of e4m3_bf16_dot32: decoded weights against one row's 32 bf16 activations.
 *  @param wf the 32 decoded weights (e4m3x32_f32)
 *  @param b0 bf16 activations 0..7
 *  @param b1 bf16 activations 8..15
 *  @param b2 bf16 activations 16..23
 *  @param b3 bf16 activations 24..31
 *  @return the dot in f32, element order 0..31 */
__device__ __forceinline__ float f32_bf16_dot32(const float (&wf)[32], const uint4 &b0, const uint4 &b1,
                                                const uint4 &b2, const uint4 &b3) {
    const uint32_t x[16] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w,
                            b2.x, b2.y, b2.z, b2.w, b3.x, b3.y, b3.z, b3.w};
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&x[j]));
        s = fmaf(wf[2 * j], a.x, s);
        s = fmaf(wf[2 * j + 1], a.y, s);
    }
    return s;
}

/* The block reduction every kernel here uses: warp xor tree, then the warps
 * in order.  `red` holds one float per warp. */
/** @param v   this thread's partial
 *  @param red shared scratch, blockDim.x / 32 floats
 *  @return the block's sum (the same value in every thread) */
__device__ __forceinline__ float block_sum(float v, float *red) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();                               /* the previous use of red is done */
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float s = 0.0f;
    for (int w = 0; w < nw; ++w) s += red[w];
    return s;
}

/* 1. norm (+ the inject partials); xn leaves in bf16, whatever W_down's WEIGHT format is (L251 /
 * ac69748f: there is no E4M3 activation slot in this family, and the read's own activation is an
 * op-internal one, so it follows the family's rule too -- W8 weights against bf16 is the W8A16 arm). */
/** Grid = (kS, T): one CTA per (stream, token); block = kNormThreads; static smem kNormThreads / 32 floats.
 *  @param streams bf16 [T][kS][kH] residual streams
 *  @param norm_w  bf16 [kHC] hc_norm weight, applied as (1 + w)
 *  @param inject  bf16 [kS][kHC] block_inject_weight, or NULL (the mixer: no inject partials)
 *  @param xb      out, bf16 [T][kHC] xn; NULL at prefill widths (the GEMM stages xn itself)
 *  @param rstd    out, f32 [T][kS] each stream's rsqrt(mean(x^2) + eps)
 *  @param injp    out, f32 [T][kS stream][kS output] the inject dot's per-stream partials (written only with inject) */
__global__ void __launch_bounds__(kNormThreads)
qwen_gr_norm_kernel(const __nv_bfloat16 *__restrict__ streams, const __nv_bfloat16 *__restrict__ norm_w,
                    const __nv_bfloat16 *__restrict__ inject,
                    __nv_bfloat16 *__restrict__ xb,
                    float *__restrict__ rstd, float *__restrict__ injp) {
    __shared__ float red[kNormThreads / 32];
    const int s = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const __nv_bfloat16 *xs = streams + (size_t)t * kHC + (size_t)s * kH;
    float x[kNormPer];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < kNormPer; ++j) { x[j] = bf2f(xs[tid + kNormThreads * j]); ss = fmaf(x[j], x[j], ss); }
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss * (1.0f / kH) + kEps);
    float io[kS] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int j = 0; j < kNormPer; ++j) {
        const int c = tid + kNormThreads * j;
        const float xn = x[j] * r * (1.0f + bf2f(norm_w[s * kH + c]));
        if (xb) xb[(size_t)t * kHC + s * kH + c] = __float2bfloat16(xn);   /* NULL: the GEMM stages xn itself */
        if (inject) {
#pragma unroll
            for (int o = 0; o < kS; ++o) io[o] = fmaf(bf2f(inject[(size_t)o * kHC + s * kH + c]), xn, io[o]);
        }
    }
    if (tid == 0) rstd[t * kS + s] = r;
    if (inject) {
#pragma unroll
        for (int o = 0; o < kS; ++o) {
            const float v = block_sum(io[o], red);
            if (tid == 0) injp[((size_t)t * kS + s) * kS + o] = v;
        }
    }
}

/* 2. W_down, split-K: part[t][split][row] = the split's share of W_row . xn_t.
 * MXFP8: 32-blocks, (W_row,blk . x_t,blk) 2^(sw + sx); BF16: 8-element chunks.
 * One warp per row; lane l takes the split's blocks (chunks) l, l + 32, ... in
 * order; the xor tree sums lanes.  Row arithmetic is independent of T.  W8A16 (the served arm) stages
 * the CTA's tokens' activation slice in shared memory first (L271): every one of its 8 row-warps read
 * it from L2 inside the inner loop, which made the kernel latency-bound (~123 GB/s at one token) and
 * doubled its time at a 4-row MTP verify.  The products and their order are unchanged; `tb` (tokens per
 * CTA, gr_down_tokens_per_cta) only regroups tokens, each of whose sums is its own. */
/** Grid = (ceil(out / 8), n_split, ceil(T / tb)), block = 256 (8 warps, one weight row each); dynamic smem
 *  tb * (in / n_split) bf16 for the W8A16 arm (the staged activation slice), 0 otherwise.
 *  @tparam W8 weights mxfp8_lt (true) or bf16 (false)
 *  @tparam A8 activations E4M3 with their own scale slab (W8A8); false = bf16 activations
 *  @param wv      weight [out][in]: E4M3 codes (W8) or bf16
 *  @param wsf     the weight's E8M0 plane (W8), else unread
 *  @param xv      activations [T][in]: E4M3 (A8) or bf16
 *  @param xsf     the activations' E8M0 slab (A8), else unread
 *  @param x_kbp   pulsar_mx_kbp(in) of the activation slab (A8)
 *  @param out     weight rows
 *  @param in      weight columns (a multiple of 32 * n_split)
 *  @param T       tokens
 *  @param n_split split-K count
 *  @param tb      tokens per CTA, at most kDownTB (gr_down_tokens_per_cta)
 *  @param part    out, f32 [T][n_split][out] the splits' partial sums */
template <bool W8, bool A8 = W8>
__global__ void __launch_bounds__(256)
qwen_gr_down_kernel(const void *__restrict__ wv, const uint8_t *__restrict__ wsf,
                    const void *__restrict__ xv, const uint8_t *__restrict__ xsf, int x_kbp,
                    int out, int in, int T, int n_split, int tb, float *__restrict__ part) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    const bool live = row < out;   /* not an early return: the W8A16 arm stages behind a barrier */
    const int split = blockIdx.y;
    const int t0 = blockIdx.z * tb, nt = min(tb, T - t0);
    float acc[kDownTB];
#pragma unroll
    for (int tt = 0; tt < kDownTB; ++tt) acc[tt] = 0.0f;
    if constexpr (W8 && A8) {
        const uint8_t *wq = (const uint8_t *)wv, *xq = (const uint8_t *)xv;
        const int nblk = in / 32, per = nblk / n_split, b0 = split * per;
        const int w_kbp = pulsar_mx_kbp(in);
        for (int b = b0 + lane; live && b < b0 + per; b += 32) {
            const uint4 *wp = reinterpret_cast<const uint4 *>(wq + (size_t)row * in + (size_t)b * 32);
            const uint4 w0 = wp[0], w1 = wp[1];
            const unsigned sw = wsf[pulsar_mx_sfoff(row, b, w_kbp)];
#pragma unroll
            for (int tt = 0; tt < kDownTB; ++tt) {
                if (tt < nt) {
                    const int t = t0 + tt;
                    const uint4 *xp = reinterpret_cast<const uint4 *>(xq + (size_t)t * in + (size_t)b * 32);
                    const float d = e4m3_dot32(w0, w1, xp[0], xp[1]);
                    acc[tt] = fmaf(d, mx_scale2(sw, xsf[pulsar_mx_sfoff(t, b, x_kbp)]), acc[tt]);
                }
            }
        }
    } else if constexpr (W8) {
        /* W8A16 (L251 / ac69748f): mxfp8_lt weights against the block input's bf16 row.  The weights
         * are read bit-identically to the W8A8 branch above; the activation is bf16, so 32 of them
         * need 4 uint4 instead of 2 -- that asymmetry is the whole difference. */
        extern __shared__ uint4 xs[];   /* [tb][per * 4]: 32 bf16 = 4 uint4 per block */
        const uint8_t *wq = (const uint8_t *)wv;
        const __nv_bfloat16 *xb = (const __nv_bfloat16 *)xv;
        const int nblk = in / 32, per = nblk / n_split, b0 = split * per;
        const int w_kbp = pulsar_mx_kbp(in);
        for (int i = threadIdx.x; i < nt * per * 4; i += blockDim.x) {
            const int tt = i / (per * 4), r = i - tt * per * 4;
            xs[tt * per * 4 + r] = reinterpret_cast<const uint4 *>(xb + (size_t)(t0 + tt) * in + (size_t)b0 * 32)[r];
        }
        __syncthreads();
        for (int b = b0 + lane; live && b < b0 + per; b += 32) {
            const uint4 *wp = reinterpret_cast<const uint4 *>(wq + (size_t)row * in + (size_t)b * 32);
            const uint4 w0 = wp[0], w1 = wp[1];
            const float wsc = mx_scale1(wsf[pulsar_mx_sfoff(row, b, w_kbp)]);
#pragma unroll
            for (int tt = 0; tt < kDownTB; ++tt) {
                if (tt < nt) {
                    const uint4 *xp = &xs[tt * per * 4 + (b - b0) * 4];
                    acc[tt] = fmaf(e4m3_bf16_dot32(w0, w1, xp[0], xp[1], xp[2], xp[3]), wsc, acc[tt]);
                }
            }
        }
    } else {
        const uint4 *wr = reinterpret_cast<const uint4 *>((const __nv_bfloat16 *)wv + (size_t)row * in);
        const int nch = in / 8, per = nch / n_split, c0 = split * per;
        for (int ci = c0 + lane; live && ci < c0 + per; ci += 32) {
            const uint4 w = wr[ci];
#pragma unroll
            for (int tt = 0; tt < kDownTB; ++tt)
                if (tt < nt)
                    acc[tt] = bf16_dot8(w, reinterpret_cast<const uint4 *>((const __nv_bfloat16 *)xv + (size_t)(t0 + tt) * in)[ci],
                                        acc[tt]);
        }
    }
#pragma unroll
    for (int tt = 0; tt < kDownTB; ++tt) {
        float v = acc[tt];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (live && lane == 0 && tt < nt) part[((size_t)(t0 + tt) * n_split + split) * out + row] = v;
    }
}

/* 2'. PREFILL: the W8A16 product as a tensor-core GEMM, for prompt chunks (T > kDecodeRowsMax).
 * part[t][split][row] = sum over the split's k of x[t][k] * W[row][k], W = E4M3 * 2^(e8m0 - 127).
 * That weight is EXACT in bf16 (3 mantissa bits, a power-of-two scale), so it is dequantized into
 * shared memory as bf16 and the product runs as mma.m16n8k16 bf16 x bf16 -> f32, a fresh fragment per
 * 16-k step added into the f32 accumulator.
 *
 * Decode widths keep a decode kernel (the GEMV (2); L288: the rows kernel for the head and the read's
 * W_down): a decode kernel run over a prompt re-reads the activation once per 8 weight rows (40x per call for W_down), which was 22% of a 4096-row prefill.  The two kernels sum in
 * different orders, so a prefilled row and a decoded row agree to rounding, not to the bit (Tyler
 * 2026-09-29: "Don't use a decode kernel for prefill"); within this kernel a row's arithmetic depends
 * on nothing but its own x and W, so the chunk width changes no bit -- nor does the CTA's output
 * width.  CTA = 4 warps = 64 tokens x 8 NJ outputs (warp w: tokens 16w..16w+15, NJ n8 tiles); a wider
 * CTA re-stages the activation fewer times.  kMmaStage MX blocks staged per barrier. */
constexpr int kDecodeRowsMax = PULSAR_QWEN_DECODE_ROWS_MAX;   ///< widths at or below take the decode GEMV
/** @var kMmaTok
 *  Tokens per prefill-GEMM CTA (4 warps x 16).
 *  @var kMmaStage
 *  MX blocks of k staged per barrier. */
constexpr int kMmaTok = 64, kMmaStage = 4;
/** @var kMmaNJ
 *  n8 output tiles per CTA of the plain prefill Linear. */
constexpr int kMmaNJ = 8, kMmaOut = 8 * kMmaNJ;  ///< the plain Linear's CTA output width
/** @var kGrNJ
 *  n8 output tiles per CTA of the read's W_up GEMM (GATE form). */
constexpr int kGrNJ = 8, kGrOut = 8 * kGrNJ;     ///< the read's W_up (GATE): 16 channels x 4 streams
/** @var kDnNJ
 *  n8 output tiles per CTA of the read's W_down GEMM (NORM form). */
constexpr int kDnNJ = 10, kDnOut = 8 * kDnNJ;    ///< the read's W_down (NORM): 320 outputs in 4 CTAs
constexpr int kMmaKs  = kMmaStage * 32;          ///< k per stage
constexpr int kMmaPad = kMmaKs + 8;              ///< smem row stride in bf16: 16-byte rows, conflict-free ldmatrix

/** One mma.sync m16n8k16 bf16 x bf16 -> f32, accumulating into d (sm_80+; traps on older arches).
 *  @param d  the f32 accumulator fragment (in/out)
 *  @param a  the A fragment (4 regs of bf16 pairs)
 *  @param b0 B fragment register 0
 *  @param b1 B fragment register 1 */
__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if PULSAR_SM80
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

/** ldmatrix .x4: four 8x8 b16 matrices from shared memory.
 *  @param r out: the four fragment registers
 *  @param p this lane's shared-memory row address */
__device__ __forceinline__ void gr_ldsm_x4(uint32_t (&r)[4], const void *p) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

/** ldmatrix .x2: two 8x8 b16 matrices from shared memory.
 *  @param r0 out: fragment register 0
 *  @param r1 out: fragment register 1
 *  @param p  this lane's shared-memory row address */
__device__ __forceinline__ void gr_ldsm_x2(uint32_t &r0, uint32_t &r1, const void *p) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(s));
}

/* The GR operands of the NORM and GATE forms (3'): the read's streams, hc_norm, the norm's rstd, and
 * (GATE) the bf16 row out. */
/// The NORM / GATE forms' extra operands of qwen_w8a16_prefill_kernel (unused by PLAIN).
struct gr_gate_args {
    /** @var streams
     *  bf16 [T][kS][kH] residual streams (NORM: the staged x; GATE: the xn the gate multiplies).
     *  @var norm_w
     *  bf16 [kHC] hc_norm weight, (1 + w). */
    const __nv_bfloat16 *streams, *norm_w;
    /// f32 [T][kS] the norm kernel's per-stream rstd.
    const float *rstd;
    /// GATE: out, bf16 [T][kH] the block input row.
    __nv_bfloat16 *x_out;
};

/* PLAIN: x is the bf16 activation.  NORM (the read's W_down): x is the streams and the staging takes
 * xn = bf16(x * rstd * (1 + w)) -- the norm kernel's three operations, so the norm never writes xn at
 * prefill widths.  GATE (the read's W_up): the gate is the epilogue. */
/// Which operand / epilogue form qwen_w8a16_prefill_kernel compiles to (see the comment above).
enum gr_mma_form { GR_PLAIN, GR_NORM, GR_GATE };

/** The W8A16 prefill GEMM (2'): E4M3 weights dequantized exactly to bf16 in smem, mma.m16n8k16 into f32.
 *  Grid = (ceil(out / (8 * NJ)), n_split, ceil(T / kMmaTok)), block = 128 (4 warps x 16 tokens), static smem
 *  (kMmaTok + 8 * NJ) rows of kMmaPad bf16.
 *  @tparam FORM PLAIN, NORM (x is the streams, staged as xn) or GATE (the gate epilogue writes ga.x_out)
 *  @tparam NJ   n8 output tiles per CTA
 *  @param wq      E4M3 weight codes [out][in]
 *  @param wsf     the weight's E8M0 plane (mxfp8_lt swizzle)
 *  @param x       bf16 activations [T][in] (NORM: the streams [T][kHC])
 *  @param out     weight rows
 *  @param in      weight columns (whole MX blocks, split evenly by n_split)
 *  @param T       tokens
 *  @param n_split split-K count
 *  @param part    out, f32 [T][n_split][out] (unused under GATE)
 *  @param ga      the NORM / GATE operands */
template <gr_mma_form FORM, int NJ>
__global__ void __launch_bounds__(128)
qwen_w8a16_prefill_kernel(const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsf,
                      const __nv_bfloat16 *__restrict__ x, int out, int in, int T, int n_split,
                      float *__restrict__ part, gr_gate_args ga) {
    constexpr bool GATE = FORM == GR_GATE;
    constexpr int OUT = 8 * NJ;                  /* W rows per CTA */
    constexpr int CPS = OUT / kS;                /* GATE: channels per stream in the CTA */
    __shared__ __align__(16) __nv_bfloat16 sx[kMmaTok][kMmaPad];
    __shared__ __align__(16) __nv_bfloat16 sw[OUT][kMmaPad];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    /* GATE: the CTA's W rows are channels c0 .. c0 + CPS - 1 of the four streams (n8 tiles s * CPS / 8 ..
     * of stream s), so every lane ends holding z for all four streams of its (token, channel) pairs;
     * otherwise rows r0 .. r0 + OUT - 1 */
    static_assert(!GATE || (NJ % kS == 0 && CPS % 8 == 0), "the gate tile is whole n8 tiles per stream");
    static_assert(kH % 8 == 0, "a staged 8-element chunk lies in one stream");
    const int r0 = blockIdx.x * OUT, c0 = blockIdx.x * CPS, split = blockIdx.y, t0 = blockIdx.z * kMmaTok;
    const int nblk = in / 32, per = nblk / n_split, b_lo = split * per, b_hi = b_lo + per;
    const int w_kbp = pulsar_mx_kbp(in);
    float acc[NJ][4];
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;

    /* The staging is register-prefetched: a stage's raw x chunks and W codes are loaded into registers
     * while the previous stage's MMAs run, then converted into shared memory -- the same values the
     * MMAs read, only issued earlier.  Thread tid takes x chunks tid + 128 i and W chunks tid + 128 i. */
    constexpr int kXT = kMmaTok * kMmaStage * 4 / 128, kWT = OUT * kMmaStage * 4 / 128;
    static_assert(kXT * 128 == kMmaTok * kMmaStage * 4 && kWT * 128 == OUT * kMmaStage * 4, "whole passes");
    uint4 xr[kXT];
    uint2 wc[kWT];
    unsigned ws[kWT];
    auto load = [&](int b0) {
        const int nbs = min(kMmaStage, b_hi - b0);
#pragma unroll
        for (int i = 0; i < kXT; ++i) {
            const int c = tid + 128 * i, tr = c / (kMmaStage * 4), ch = c % (kMmaStage * 4), t = t0 + tr;
            xr[i] = t < T && ch < nbs * 4 ? *reinterpret_cast<const uint4 *>(x + (size_t)t * in + (size_t)b0 * 32 + ch * 8)
                                          : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int i = 0; i < kWT; ++i) {
            const int c = tid + 128 * i, wr = c / (kMmaStage * 4), ch = c % (kMmaStage * 4);
            const int row = GATE ? (wr / CPS) * kH + c0 + wr % CPS : r0 + wr;
            const bool ok = row < out && ch < nbs * 4;
            wc[i] = ok ? *reinterpret_cast<const uint2 *>(wq + (size_t)row * in + (size_t)b0 * 32 + ch * 8) : make_uint2(0u, 0u);
            ws[i] = ok ? wsf[pulsar_mx_sfoff(row, b0 + (ch >> 2), w_kbp)] : 0u;
        }
    };
    auto store = [&](int b0) {
        const int nbs = min(kMmaStage, b_hi - b0);
        /* x: kMmaTok rows x nbs blocks of 32 bf16 = 4 16-byte chunks a block; rows past T are zero */
#pragma unroll
        for (int i = 0; i < kXT; ++i) {
            const int c = tid + 128 * i, tr = c / (kMmaStage * 4), ch = c % (kMmaStage * 4), t = t0 + tr;
            if (ch >= nbs * 4) continue;
            uint4 v = xr[i];
            if constexpr (FORM == GR_NORM) {
                if (t < T) {
                    const int k = b0 * 32 + ch * 8, s = k / kH;
                    float xf[8], wf[8];
                    bf16x8(xr[i], xf);
                    bf16x8(*reinterpret_cast<const uint4 *>(ga.norm_w + k), wf);
                    const float r = ga.rstd[t * kS + s];
                    uint32_t o[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const __nv_bfloat162 h = __halves2bfloat162(__float2bfloat16(xf[2 * e] * r * (1.0f + wf[2 * e])),
                                                                    __float2bfloat16(xf[2 * e + 1] * r * (1.0f + wf[2 * e + 1])));
                        o[e] = *reinterpret_cast<const uint32_t *>(&h);
                    }
                    v = make_uint4(o[0], o[1], o[2], o[3]);
                }
            }
            *reinterpret_cast<uint4 *>(&sx[tr][ch * 8]) = v;
        }
        /* W: OUT rows x nbs blocks, 8 codes a thread-task, dequantized exactly into bf16 */
#pragma unroll
        for (int i = 0; i < kWT; ++i) {
            const int c = tid + 128 * i, wr = c / (kMmaStage * 4), ch = c % (kMmaStage * 4);
            if (ch >= nbs * 4) continue;
            const float s = mx_scale1(ws[i]);
            const uint8_t *cb = reinterpret_cast<const uint8_t *>(&wc[i]);
            uint32_t o[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                __nv_fp8_e4m3 q0, q1;
                q0.__x = cb[2 * e];
                q1.__x = cb[2 * e + 1];
                const __nv_bfloat162 v = __floats2bfloat162_rn(float(q0) * s, float(q1) * s);
                o[e] = *reinterpret_cast<const uint32_t *>(&v);
            }
            *reinterpret_cast<uint4 *>(&sw[wr][ch * 8]) = make_uint4(o[0], o[1], o[2], o[3]);
        }
    };
    load(b_lo);
    for (int b0 = b_lo; b0 < b_hi; b0 += kMmaStage) {
        const int nbs = min(kMmaStage, b_hi - b0);
        store(b0);
        __syncthreads();
        if (b0 + kMmaStage < b_hi) load(b0 + kMmaStage);
        for (int kk = 0; kk < nbs * 32; kk += 16) {
            uint32_t a[4];
            gr_ldsm_x4(a, &sx[warp * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                uint32_t bb0, bb1;
                gr_ldsm_x2(bb0, bb1, &sw[j * 8 + (lane & 7)][kk + ((lane >> 3) & 1) * 8]);
                /* a FRESH fragment per 16-k step, added into acc in f32: the tensor core's own
                 * accumulate is not IEEE f32 (measured: 2.9e-04 against double over K = 2560 when
                 * acc rode in the C operand, vs ~1e-6 this way) -- the dense EXL3 arm's pattern */
                float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                mma_bf16_16816(d, a, bb0, bb1);
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[j][e] += d[e];
            }
        }
        __syncthreads();
    }
    /* lane (g, c): acc[j][0..1] = token 16w+g, rows 8j+2c, +1; acc[j][2..3] = token 16w+g+8 */
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    if constexpr (GATE) {
        /* the up kernel's emit with z_s = acc[s * CPS / 8 + q][e]: x = (g_0 xn_0 + g_1 xn_1 + g_2 xn_2 + g_3 xn_3) / 4 */
#pragma unroll
        for (int q = 0; q < CPS / 8; ++q)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int t = t0 + warp * 16 + g + 8 * h, c = c0 + 8 * q + c2;
            if (t >= T) continue;
            float p[2][kS];
#pragma unroll
            for (int s = 0; s < kS; ++s) {
                const int row = s * kH + c;                 /* channels c, c + 1: one bf16 pair */
                const float2 xs = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(ga.streams + (size_t)t * kHC + row));
                const float2 nw = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(ga.norm_w + row));
                const float r = ga.rstd[t * kS + s];
                const int j = s * (CPS / 8) + q;
                p[0][s] = 1.0f / (1.0f + expf(-acc[j][2 * h])) * (xs.x * r * (1.0f + nw.x));
                p[1][s] = 1.0f / (1.0f + expf(-acc[j][2 * h + 1])) * (xs.y * r * (1.0f + nw.y));
            }
            float v[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) v[e] = (((p[e][0] + p[e][1]) + p[e][2]) + p[e][3]) * (1.0f / kS);
            *reinterpret_cast<__nv_bfloat162 *>(ga.x_out + (size_t)t * kH + c) = __floats2bfloat162_rn(v[0], v[1]);
        }
    } else {
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int t = t0 + warp * 16 + g + 8 * (e >> 1), row = r0 + j * 8 + c2 + (e & 1);
                if (t < T && row < out) part[((size_t)t * n_split + split) * out + row] = acc[j][e];
            }
    }
}

/* 3. mid + up + gate + mean: one warp per stream, lane = channel, TB
 * tokens per CTA (kUpTBDecode; kUpTB for the BF16 mixer at prefill widths -- the W8 prefill is 3').
 *   mid  a = silu(d / 4), d = the down's splits summed in split order, for the
 *        CTA's tokens, in shared memory -- in the format W_up reads (E4M3 per
 *        32 by the one producer encoder, or bf16).  Every CTA derives the same
 *        a (320 values a token: cheaper than a launch and a round trip);
 *        CTA x == 0 also finalises inj = 2 sigmoid(z / 4) from the norm's
 *        per-stream partials, in stream order.
 *   up   the k loop is outermost: each 32-block of the lane's W_up row is
 *        loaded once and applied to every token of the CTA (a token's z sums
 *        its blocks in order 0..9, so the row arithmetic does not depend on T);
 *        the gated products go to shared memory and one warp per token takes
 *        the stream mean and emits the row.
 * L288 C2: the W8 block is DECODED once (32 E4M3 codes -> f32, the conversions e4m3_bf16_dot32 makes) and
 * that f32 row applied to each of the CTA's tokens (f32_bf16_dot32: the same 32 fmaf in the same k order),
 * where every token used to decode the codes again; and the decode tile is 8 tokens, so a verify of up to 8
 * rows reads W_up once.  A token's arithmetic is unchanged, so the output is bit-identical (and still does
 * not depend on T). */
/* The W8 up's weight tile (decode widths), staged whole: the CTA's 4 x 32 rows of 320 E4M3 bytes, copied by
 * cp.async in coalesced 16-byte chunks at kernel entry so the load overlaps the mid.  The row
 * stride is padded to 336 B (84 words): a quarter-warp's 8 lanes, reading rows l .. l + 7 at the
 * same column, then hit 8 distinct 4-bank groups.  The arithmetic reads the same bytes in the same
 * order as a direct global load, so the output is bit-identical to it. */
constexpr int kUpRowBytes = kR;                               ///< 320 E4M3 codes
constexpr int kUpRowStride = kUpRowBytes + 16;                ///< 336 B
constexpr int kUpTileBytes = kS * 32 * kUpRowStride;          ///< 43,008 B dynamic shared memory
static_assert(kUpRowBytes % 16 == 0 && (kUpRowStride / 4) % 32 == 20, "the padded stride is conflict-free");

/** cp.async.cg of 16 bytes global -> shared (sm_80+; traps on older arches).
 *  @param smem shared destination, 16-B aligned
 *  @param gmem global source, 16-B aligned */
__device__ __forceinline__ void gr_cp_async16(void *smem, const void *gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
#if PULSAR_SM80
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
#else
    __trap();
#endif
}

/** Grid = (kH / 32, ceil(T / TB)), block = 32 * kS (warp s = stream s, lane = channel); W8: dynamic smem
 *  kUpTileBytes (the staged W_up tile, reused for the gated products), static s_a [TB][kR] bf16.
 *  @tparam W8 W_up mxfp8_lt (decode widths only) or bf16
 *  @tparam TB tokens per CTA (kUpTBDecode, or kUpTB for the bf16 mixer at prefill widths)
 *  @param wv      W_up [kHC][kR]: E4M3 codes (W8) or bf16
 *  @param wsf     W_up's E8M0 plane (W8)
 *  @param part    f32 [T][kDownSplit][kR] the down's split partials
 *  @param injp    f32 [T][kS][kS] the inject partials, or NULL (no inject)
 *  @param inj     out, f32 [T][kS] = 2 sigmoid(z / 4) (CTA x == 0 writes it; only with injp)
 *  @param streams bf16 [T][kS][kH] residual streams
 *  @param norm_w  bf16 [kHC] hc_norm weight
 *  @param rstd    f32 [T][kS] from the norm kernel
 *  @param T       tokens
 *  @param x_out   out, bf16 [T][kH] the block input row */
template <bool W8, int TB>
__global__ void __launch_bounds__(32 * kS)
qwen_gr_up_kernel(const void *__restrict__ wv, const uint8_t *__restrict__ wsf,
                  const float *__restrict__ part, const float *__restrict__ injp, float *__restrict__ inj,
                  const __nv_bfloat16 *__restrict__ streams, const __nv_bfloat16 *__restrict__ norm_w,
                  const float *__restrict__ rstd, int T,
                  __nv_bfloat16 *__restrict__ x_out) {
    constexpr int NB = kR / 32;
    /* the gated products: in the staged W_up tile's memory once the up has read it (W8 -- L288 C2: that keeps the
     * 8-token CTA at two an SM), their own array otherwise */
    __shared__ float prod_own[W8 ? 1 : TB][kS][32];
    /* L251 / ac69748f: `a` is bf16 whatever W_up's WEIGHT format is -- this was the last op-internal
     * E4M3 activation in the read.  W8 now means E4M3 weights against a bf16 activation, the same
     * W8A16 shape the down already uses, so e4m3_bf16_dot32's arithmetic serves here too. */
    __shared__ __align__(16) uint8_t s_a[TB][2 * kR];
    const int s = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c = blockIdx.x * 32 + lane;
    const int row = s * kH + c;
    const int t0 = blockIdx.y * TB, nt = min(TB, T - t0);
    /* W8 runs this kernel at decode widths only (its prefill is 3'), and stages W_up there */
    static_assert(!W8 || TB == kUpTBDecode, "the W8 up at prefill widths is the GEMM (3')");
    constexpr bool STAGE = W8;
    extern __shared__ __align__(16) uint8_t s_w[];
    static_assert(!W8 || sizeof(float) * TB * kS * 32 <= kUpTileBytes, "the gated products fit the W_up tile");
    float (*prod)[kS][32] = W8 ? reinterpret_cast<float (*)[kS][32]>(s_w) : prod_own;
    if constexpr (STAGE) {
        /* stream ss's 32 rows are one contiguous 10,240 B run of W_up */
        constexpr int kChunks = kUpRowBytes / 16;                 /* 20 per row */
        for (int i = threadIdx.x; i < kS * 32 * kChunks; i += 32 * kS) {
            const int rr = i / kChunks, ch = i % kChunks, ss = rr / 32, cc = rr % 32;
            const uint8_t *src = (const uint8_t *)wv + ((size_t)ss * kH + blockIdx.x * 32 + cc) * kR + ch * 16;
            gr_cp_async16(s_w + rr * kUpRowStride + ch * 16, src);
        }
#if PULSAR_SM80
        asm volatile("cp.async.commit_group;\n" ::: "memory");
#else
        __trap();
#endif
    }
    /* mid: warp s takes the (token, 32-group) pairs s, s + 4, ...; lane = element */
    for (int g = s; g < nt * NB; g += kS) {
        const int tt = g / NB, b = g % NB, r = b * 32 + lane;
        float d = 0.0f;
#pragma unroll
        for (int sp = 0; sp < kDownSplit; ++sp) d += part[((size_t)(t0 + tt) * kDownSplit + sp) * kR + r];
        const float zz = d * (1.0f / kS);
        const float av = zz / (1.0f + expf(-zz));
        reinterpret_cast<__nv_bfloat16 *>(s_a[tt])[r] = __float2bfloat16(av);
    }
    if (injp && blockIdx.x == 0 && (int)threadIdx.x < nt * kS) {
        const int tt = threadIdx.x / kS, j = threadIdx.x % kS, t = t0 + tt;
        float zi = 0.0f;
#pragma unroll
        for (int ss = 0; ss < kS; ++ss) zi += injp[((size_t)t * kS + ss) * kS + j];
        inj[t * kS + j] = 2.0f / (1.0f + expf(-zi * (1.0f / kS)));
    }
    if constexpr (STAGE) {
#if PULSAR_SM80
        asm volatile("cp.async.wait_all;\n" ::: "memory");
#else
        __trap();
#endif
    }
    __syncthreads();
    float z[TB];
#pragma unroll
    for (int tt = 0; tt < TB; ++tt) z[tt] = 0.0f;
    if constexpr (W8) {
        const int w_kbp = pulsar_mx_kbp(kR);
        const uint4 *wp = reinterpret_cast<const uint4 *>(s_w + (s * 32 + lane) * kUpRowStride);
        for (int b = 0; b < NB; ++b) {
            /* the block's 32 codes decoded once for all the CTA's tokens (L288 C2); 32 bf16 activations are FOUR
             * uint4, and the weight's E8M0 is the only scale left to apply (bf16 carries its own exponent) */
            float wf[32];
            e4m3x32_f32(wp[2 * b], wp[2 * b + 1], wf);
            const float sc = mx_scale1(wsf[pulsar_mx_sfoff(row, b, w_kbp)]);
#pragma unroll
            for (int tt = 0; tt < TB; ++tt) {
                if (tt < nt) {
                    const uint4 *ap = reinterpret_cast<const uint4 *>(s_a[tt]) + 4 * b;
                    z[tt] = fmaf(f32_bf16_dot32(wf, ap[0], ap[1], ap[2], ap[3]), sc, z[tt]);
                }
            }
        }
        __syncthreads();   /* every warp is past its last read of the tile: the products take its memory */
    } else {
        const uint4 *wp = reinterpret_cast<const uint4 *>((const __nv_bfloat16 *)wv + (size_t)row * kR);
        for (int ci = 0; ci < kR / 8; ++ci) {
            const uint4 w = wp[ci];
#pragma unroll
            for (int tt = 0; tt < TB; ++tt)
                if (tt < nt) z[tt] = bf16_dot8(w, reinterpret_cast<const uint4 *>(s_a[tt])[ci], z[tt]);
        }
    }
    const float w1n = 1.0f + bf2f(norm_w[row]);
#pragma unroll
    for (int tt = 0; tt < TB; ++tt) {
        if (tt < nt) {
            const int t = t0 + tt;
            const float g = 1.0f / (1.0f + expf(-z[tt]));
            const float xn = bf2f(streams[(size_t)t * kHC + row]) * rstd[t * kS + s] * w1n;
            prod[tt][s][lane] = g * xn;
        }
    }
    __syncthreads();
    for (int tt = s; tt < nt; tt += kS) {                 /* whole warps: tt depends on the warp only */
        const int t = t0 + tt;
        const float v = (((prod[tt][0][lane] + prod[tt][1][lane]) + prod[tt][2][lane]) + prod[tt][3][lane]) * (1.0f / kS);
        const __nv_bfloat16 vb = __float2bfloat16(v);
        x_out[(size_t)t * kH + c] = vb;   /* bf16 is the one encoding -- no E4M3 slot to fill */
    }
}

/** The write: streams_s += out * inj_s, f32 math, one bf16 rounding.  8 elements (one 16-B chunk) a thread:
 *  grid = ceil(T * kHC / 8 / 256), block = 256.
 *  @param streams bf16 [T][kS][kH], updated in place; 16-B aligned
 *  @param out     f32 [T][kH] the block's output; 16-B aligned
 *  @param inj     f32 [T][kS] the read's inject gates
 *  @param T       tokens */
__global__ void qwen_gr_write_kernel(__nv_bfloat16 *__restrict__ streams, const float *__restrict__ out,
                                     const float *__restrict__ inj, int T) {
    /* 8 consecutive elements a thread (one 16-byte chunk, inside one stream: kH % 8 == 0) */
    const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= (size_t)T * kHC) return;
    const size_t t = i / kHC;
    const int s = (int)((i / kH) % kS), c = (int)(i % kH);
    uint4 *sp = reinterpret_cast<uint4 *>(streams + i);
    float x[8];
    bf16x8(*sp, x);
    const float4 *op = reinterpret_cast<const float4 *>(out + t * kH + c);
    const float4 o0 = op[0], o1 = op[1];
    const float o[8] = {o0.x, o0.y, o0.z, o0.w, o1.x, o1.y, o1.z, o1.w};
    const float g = inj[t * kS + s];
    uint32_t r[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const __nv_bfloat162 v = __halves2bfloat162(__float2bfloat16(x[2 * e] + o[2 * e] * g),
                                                    __float2bfloat16(x[2 * e + 1] + o[2 * e + 1] * g));
        r[e] = *reinterpret_cast<const uint32_t *>(&v);
    }
    *sp = make_uint4(r[0], r[1], r[2], r[3]);
}

/* 3'. PREFILL (W8, T > kDecodeRowsMax): the up as a tensor-core GEMM.  The up kernel above reads its
 * W_up row once per 16-token CTA on the CUDA cores (0.49 s of a 4096-row prefill); here
 *   mid   a = silu(d / 4) for every (token, rank) into a bf16 [T][kR] buffer, and inj -- the same
 *         three operations as the up kernel's mid, so `a` is bit-identical to its shared copy;
 *   GEMM  z = W_up a, the W8A16 prefill kernel (2') with one split, in its GATE form: a CTA takes rows
 *         s * 2560 + c0 .. c0 + 15 of the four streams (n8 tiles 2s, 2s + 1), so each lane holds z_s for all four
 *         streams of its (token, channel) pairs and the epilogue is the gate -- x = (g_0 xn_0 + g_1 xn_1
 *         + g_2 xn_2 + g_3 xn_3) / 4, g_s = sigmoid(z_s), the up kernel's emit.  z never reaches memory
 *         (it was an f32 [T][10240] round trip, 168 MB a 4096-row chunk); a row's k order is unchanged.
 * Only z's summation order differs from the decode kernel, so a prefilled row agrees with a decoded one
 * to rounding (the same contract as 2'). */
/** One thread per (token, rank) element: grid = ceil(T * kR / 256), block = 256 (T * kR >= T * kS, so the
 *  inj threads are covered).
 *  @param part f32 [T][kDownSplit][kR] the down's split partials
 *  @param injp f32 [T][kS][kS] the inject partials, or NULL
 *  @param inj  out, f32 [T][kS] (only with injp)
 *  @param T    tokens
 *  @param a    out, bf16 [T][kR] silu(d / 4) */
__global__ void __launch_bounds__(256)
qwen_gr_mid_kernel(const float *__restrict__ part, const float *__restrict__ injp, float *__restrict__ inj,
                   int T, __nv_bfloat16 *__restrict__ a) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)T * kR) {
        const int t = (int)(i / kR), r = (int)(i % kR);
        float d = 0.0f;
#pragma unroll
        for (int sp = 0; sp < kDownSplit; ++sp) d += part[((size_t)t * kDownSplit + sp) * kR + r];
        const float zz = d * (1.0f / kS);
        a[i] = __float2bfloat16(zz / (1.0f + expf(-zz)));
    }
    if (injp && i < (int64_t)T * kS) {
        const int t = (int)(i / kS), j = (int)(i % kS);
        float zi = 0.0f;
#pragma unroll
        for (int ss = 0; ss < kS; ++ss) zi += injp[((size_t)t * kS + ss) * kS + j];
        inj[t * kS + j] = 2.0f / (1.0f + expf(-zi * (1.0f / kS)));
    }
}

/// The read's device scratch, carved by gr_ws_layout.
struct gr_ws {
    /// bf16 [T][kHC] xn (decode widths; the prefill GEMM stages its own).
    uint8_t *xn;                      /* the norm's bf16 activation row (L251 / ac69748f) */
    /** @var rstd
     *  f32 [T][kS] per-stream rstd.
     *  @var injp
     *  f32 [T][kS][kS] inject partials.
     *  @var part
     *  f32 [T][kDownSplit][kR] the down's split partials. */
    float *rstd, *injp, *part;
    /// bf16 [T][kR] silu(d / 4), the prefill W_up GEMM's input.
    __nv_bfloat16 *a;                 /* prefill only: silu(d / 4), [T][kR] */
};

/** Lay the read's workspace out for T rows, each piece 256-B aligned.
 *  @param T    tokens
 *  @param base workspace start, or nullptr to size only
 *  @param cap  workspace bytes (checked when base is non-null)
 *  @param o    out: the pieces, or nullptr
 *  @return bytes used, 0 when they exceed cap */
static size_t gr_ws_layout(int T, void *base, size_t cap, gr_ws *o) {
    size_t used = 0;
    bool failed = false;
    auto take = [&](size_t bytes) -> void * {
        const size_t off = (used + 255) & ~(size_t)255;
        if (off + bytes > (base ? cap : (size_t)-1)) { failed = true; return nullptr; }
        used = off + bytes;
        return base ? (uint8_t *)base + off : nullptr;
    };
    gr_ws m{};
    m.xn    = (uint8_t *)take((size_t)T * kHC * 2);
    m.rstd  = (float *)take((size_t)T * kS * 4);
    m.injp  = (float *)take((size_t)T * kS * kS * 4);
    m.part  = (float *)take((size_t)T * kDownSplit * kR * 4);
    /* the prefill GEMM's buffers at every T: a prompt chunk of any width takes it (L266) */
    m.a = (__nv_bfloat16 *)take((size_t)T * kR * 2);
    if (o) *o = m;
    return failed ? 0 : used;
}

/** Check the last launch; prints the failure with `what` and returns false on error.
 *  @param what step name for the message
 *  @return true when no launch error is pending */
static bool launch_ok(const char *what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: qwen GR %s: launch failed: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

} // namespace

extern "C" size_t pulsar_qwen_gr_workspace_bytes(int T) {
    return T > 0 ? gr_ws_layout(T, nullptr, 0, nullptr) : 0;
}

/* The W8A16 GEMV's tokens per CTA: kDownTB, or fewer when that many tokens' activation slices (in / n_split
 * bf16 each) would not stage in kDownStageBytes.  0 = a slice too wide to stage even alone. */
/// @param in      the GEMV's input width
/// @param n_split split-K factor (each CTA stages in / n_split bf16 per token)
/// @return tokens per CTA, or 0 when one token's slice does not stage
static int gr_down_tokens_per_cta(int in, int n_split) {
    const int slice = in / n_split * 2;
    const int tb = kDownStageBytes / slice;
    return tb < 1 ? 0 : tb < kDownTB ? tb : kDownTB;
}

namespace {
/* L288: the tensor-core rows kernel (below, with the head) -- the read's W_down takes it at decode widths */
constexpr int kHeadChunk = 64;                    ///< k per lane load: 16 codes a row
constexpr int kHeadUnroll = 4;                    ///< chunks per step: two 4-block scale words a row
/// k per rows-kernel step (256: 8 MX blocks, two scale words a row).
constexpr int kHeadKStep = kHeadChunk * kHeadUnroll;
/// Widest call the rows kernel takes (two n8 token tiles).
constexpr int kHeadRowsMax = 16;
constexpr int kRowsLinearOutMin = 256;             ///< a plain Linear this wide takes the rows kernel at decode widths
template <int NT>
int rows_launch(const uint8_t *wq, const uint8_t *sf, const uint16_t *x_bf16, int out, int in, int rows, int n_split,
                float *y, bool persistent, cudaStream_t stream);
} // namespace

/** One block of the GR read's kernels: the grouped norm, W_down (the prefill GEMM, the decode rows kernel, or the
 *  bf16 GEMV), the silu / 4 and W_up gate with the stream mean.
 *  @tparam W8 the down / up weights are MXFP8 (else BF16); the activations are bf16 either way
 *  @param w       the site's weights
 *  @param streams device bf16 [T][4][2560] this block's rows
 *  @param T       rows in this block (at most PULSAR_QWEN_DECODE_ROWS_MAX for decode rows)
 *  @param x_bf16  device bf16 [T][2560] out
 *  @param inj     device f32 [T][4] injection weights out, or NULL (the mixer)
 *  @param m       the carved workspace
 *  @param stream  stream to launch on
 *  @return 0 on success, nonzero when a launch failed or was refused */
template <bool W8>
static int gr_read_kernels(const pulsar_qwen_gr_dev *w, const uint16_t *streams, int T, uint16_t *x_bf16,
                            float *inj, const gr_ws &m, cudaStream_t stream) {
    /* W8 selects the WEIGHT format only; the activation is bf16 either way -- the W8A16 arm, on the
     * tensor cores (2') at prefill widths, where the GEMM's staging takes xn from the streams itself
     * (GR_NORM), so the norm emits only rstd and the inject partials there. */
    const bool gemm = W8 && w->prompt;   /* decode rows arrive <= kDecodeRowsMax at a time (the launcher's blocks) */
    const gr_gate_args ga{(const __nv_bfloat16 *)streams, (const __nv_bfloat16 *)w->norm_w, m.rstd, (__nv_bfloat16 *)x_bf16};
    qwen_gr_norm_kernel<<<dim3(kS, T), kNormThreads, 0, stream>>>(
        (const __nv_bfloat16 *)streams, (const __nv_bfloat16 *)w->norm_w, (const __nv_bfloat16 *)w->inject,
        gemm ? nullptr : (__nv_bfloat16 *)m.xn, m.rstd, m.injp);
    if (gemm) {
        qwen_w8a16_prefill_kernel<GR_NORM, kDnNJ><<<dim3((kR + kDnOut - 1) / kDnOut, kDownSplit, (T + kMmaTok - 1) / kMmaTok), 128, 0, stream>>>(
            (const uint8_t *)w->down.w, w->down.sf, (const __nv_bfloat16 *)streams, kR, kHC, T, kDownSplit, m.part, ga);
    } else if (W8) {
        /* L288: decode widths on the tensor cores too -- the W_down read once for all <= 16 rows (the GEMV re-read
         * it per 8-row group and re-decoded every code per row: 42 us at 9-16 rows vs 22 at <= 8) */
        static_assert(kHC / kDownSplit % kHeadKStep == 0 && kDecodeRowsMax <= kHeadRowsMax, "W_down splits into whole steps");
        const int rc = T > 8 ? rows_launch<2>((const uint8_t *)w->down.w, w->down.sf, (const uint16_t *)m.xn, kR, kHC, T,
                                              kDownSplit, m.part, false, stream)
                             : rows_launch<1>((const uint8_t *)w->down.w, w->down.sf, (const uint16_t *)m.xn, kR, kHC, T,
                                              kDownSplit, m.part, false, stream);
        if (rc) return rc;
    } else {
        const int tb = gr_down_tokens_per_cta(kHC, kDownSplit);
        qwen_gr_down_kernel<W8, false><<<dim3((kR + 7) / 8, kDownSplit, (T + tb - 1) / tb), 256,
                                         W8 ? (size_t)tb * kHC / kDownSplit * 2 : 0, stream>>>(
            w->down.w, w->down.sf, m.xn, nullptr, 0, kR, kHC, T, kDownSplit, tb, m.part);
    }
    if (gemm) {
        const int64_t nm = (int64_t)T * kR;
        qwen_gr_mid_kernel<<<(unsigned)((nm + 255) / 256), 256, 0, stream>>>(m.part, w->inject ? m.injp : nullptr, inj,
                                                                           T, m.a);
        /* the up GEMM with the gate in its epilogue: a CTA per 16 channels x the 4 streams, z stays in registers */
        static_assert(kH % (kGrOut / kS) == 0, "the gate tiles the channels by kGrOut / 4");
        qwen_w8a16_prefill_kernel<GR_GATE, kGrNJ><<<dim3(kH / (kGrOut / kS), 1, (T + kMmaTok - 1) / kMmaTok), 128, 0, stream>>>(
            (const uint8_t *)w->up.w, w->up.sf, m.a, kHC, kR, T, 1, nullptr, ga);
        return 0;
    }
    if (W8) {
        /* per device, return codes checked (L335: set once per process, both ignored) */
        static int smem_set[PULSAR_CUDA_DEVICE_SLOTS] = {};
        static bool carve_set[PULSAR_CUDA_DEVICE_SLOTS] = {};
        const int d = pulsar_cuda_device_slot();
        if (d < 0 || !pulsar_cuda_smem_opt_in((const void *)qwen_gr_up_kernel<true, kUpTBDecode>, smem_set, kUpTileBytes) ||
            (!carve_set[d] && cudaFuncSetAttribute(qwen_gr_up_kernel<true, kUpTBDecode>,
                                                   cudaFuncAttributePreferredSharedMemoryCarveout, 100) != cudaSuccess)) {
            (void)cudaGetLastError();
            fprintf(stderr, "pulsar: qwen GR up: the decode tile's %d B of shared memory is refused by the device -- "
                            "refusing\n", (int)kUpTileBytes);
            return -1;
        }
        carve_set[d] = true;
    }
    /* W8 is at decode widths here; the BF16 mixer takes the 16-token tile at prefill widths */
    const int tb = !W8 && w->prompt ? kUpTB : kUpTBDecode;
    const dim3 grid(kH / 32, (T + tb - 1) / tb);
    const size_t smem = W8 ? kUpTileBytes : 0;
    const float *injp = w->inject ? m.injp : nullptr;
    if constexpr (!W8) {
        if (tb == kUpTB) {
            qwen_gr_up_kernel<W8, kUpTB><<<grid, 32 * kS, smem, stream>>>(
                w->up.w, w->up.sf, m.part, injp, inj, (const __nv_bfloat16 *)streams, (const __nv_bfloat16 *)w->norm_w,
                m.rstd, T, (__nv_bfloat16 *)x_bf16);
            return 0;
        }
    }
    qwen_gr_up_kernel<W8, kUpTBDecode><<<grid, 32 * kS, smem, stream>>>(
        w->up.w, w->up.sf, m.part, injp, inj, (const __nv_bfloat16 *)streams, (const __nv_bfloat16 *)w->norm_w,
        m.rstd, T, (__nv_bfloat16 *)x_bf16);
    return 0;
}

extern "C" int pulsar_qwen_gr_read_launch(const pulsar_qwen_gr_dev *w, const uint16_t *streams, int T,
                                          uint16_t *x_bf16, float *inj,
                                          void *ws, size_t ws_bytes, cudaStream_t stream) {
    PULSAR_REQUIRE_CC(80, "the Qwen GR read (bf16 m16n8k16, cp.async)", -1);   /* L321: refuse below the lane's arch, before its kernels trap */
    if (!w || !w->norm_w || !w->down.w || !w->up.w || !streams || T <= 0 ||
        !x_bf16 || (w->inject && !inj)) {
        fprintf(stderr, "pulsar: qwen GR read: a null input -- refusing\n");
        return -1;
    }
    if ((w->down.sf == nullptr) != (w->up.sf == nullptr)) {
        fprintf(stderr, "pulsar: qwen GR read: W_down and W_up must both be MXFP8 or both BF16 -- refusing\n");
        return -1;
    }
    if (w->down.out != kR || w->down.in != kHC || w->up.out != kHC || w->up.in != kR) {
        fprintf(stderr, "pulsar: qwen GR read: low-rank %dx%d / %dx%d, built for %dx%d / %dx%d -- refusing\n",
                w->down.out, w->down.in, w->up.out, w->up.in, kR, kHC, kHC, kR);
        return -1;
    }
    if (((uintptr_t)w->down.w | (uintptr_t)w->up.w) & 15u) {
        fprintf(stderr, "pulsar: qwen GR read: low-rank weights must be 16-byte aligned -- refusing\n");
        return -1;
    }
    gr_ws m;
    if (!ws || gr_ws_layout(T, ws, ws_bytes, &m) == 0) {
        fprintf(stderr, "pulsar: qwen GR read: workspace %zu B < %zu B for %d rows -- refusing\n",
                ws_bytes, pulsar_qwen_gr_workspace_bytes(T), T);
        return -1;
    }
    const bool w8 = w->down.sf != nullptr;
    static int announced[2] = {0, 0};
    if (!announced[w8]) {
        announced[w8] = 1;
        fprintf(stderr, "pulsar: L251 qwen GR read = grouped norm -> %s W_down (split-K %d%s) -> silu/4 -> %s W_up "
                        "sigmoid gate, stream mean -> bf16 row; streams bf16, activation bf16\n",
                w8 ? "MXFP8" : "BF16", kDownSplit, w8 ? "; L288 tensor cores at every width" : "", w8 ? "MXFP8" : "BF16");
    }
    /* L251 / ac69748f: there is no scale slab to clear and no slot to fill -- xn is bf16, so the
     * norm writes every element the down GEMV reads and nothing else owns this buffer.
     * L288 C1: decode rows past kDecodeRowsMax (a wide verify) take the decode arms a block at a time -- every kernel
     * of the read indexes its row alone, the workspace's per-row planes from 0 reused in stream order -- so a row's
     * bytes are a one-row decode's at any width (the prefill GEMM's here cost a wide verify more than it saved). */
    int rc = 0;
    const int blk = w->prompt ? T : kDecodeRowsMax;
    for (int r0 = 0; rc == 0 && r0 < T; r0 += blk) {
        const int t = T - r0 < blk ? T - r0 : blk;
        const uint16_t *s = streams + (size_t)r0 * kHC;
        uint16_t *xb = x_bf16 + (size_t)r0 * kH;
        float *in = inj ? inj + (size_t)r0 * PULSAR_QWEN_HC : nullptr;
        rc = w8 ? gr_read_kernels<true>(w, s, t, xb, in, m, stream) : gr_read_kernels<false>(w, s, t, xb, in, m, stream);
    }
    if (rc) {
        fprintf(stderr, "pulsar: qwen GR read: the W_down of %d rows could not be launched -- refusing\n", T);
        return -3;
    }
    return launch_ok("read") ? 0 : -3;
}

extern "C" int pulsar_qwen_gr_write_launch(uint16_t *streams, const float *out, const float *inj, int T,
                                           cudaStream_t stream) {
    if (!streams || !out || !inj || T <= 0) {
        fprintf(stderr, "pulsar: qwen GR write: a null input -- refusing\n");
        return -1;
    }
    if (((uintptr_t)streams | (uintptr_t)out) & 15u) {
        fprintf(stderr, "pulsar: qwen GR write: streams and out must be 16-byte aligned -- refusing\n");
        return -1;
    }
    const size_t n8 = (size_t)T * kHC / 8;
    qwen_gr_write_kernel<<<(unsigned)((n8 + 255) / 256), 256, 0, stream>>>((__nv_bfloat16 *)streams, out, inj, T);
    return launch_ok("write") ? 0 : -3;
}

/* L251 S1/S2/S3: the plain MXFP8 dense Linear the recipe's mxfp8_lt tensors need
 * (GDN in_proj_a / _b, the indexer's index_qk_proj).  It IS the W_down arithmetic
 * above with n_split = 1: the split-K kernel's `part` buffer is then exactly
 * y [T][out] (index (t * 1 + 0) * out + row), so there is no reduction pass and
 * no workspace.  Split-K would only add partials to sum; with one split the row's
 * 80 blocks are walked by one warp in order. */
extern "C" int pulsar_qwen_mxfp8_linear_launch(const pulsar_qwen_lowrank *l, const uint16_t *x_bf16, int rows,
                                               float *y, void *ws, size_t ws_bytes, cudaStream_t stream) {
    PULSAR_REQUIRE_CC(80, "the Qwen MXFP8 linear (tensor-core rows kernel)", -1);   /* L321: refuse below the lane's arch, before its kernels trap */
    (void)ws;
    (void)ws_bytes;
    /* L251 / ac69748f: the bf16 activation, so the W8=false arm of the same kernel.  The E4M3 slot and
     * its per-32 scales are gone; the weight stays mxfp8_lt (E4M3 + E8M0), which is the weight tier. */
    if (!l || !l->w || !l->sf || !x_bf16 || rows <= 0 || l->in <= 0 || l->out <= 0 || l->in % 32 != 0) {
        fprintf(stderr, "pulsar: qwen mxfp8 linear: %d -> %d needs a bf16 activation of width %d -- refusing\n",
                l ? l->in : -1, l ? l->out : -1, l ? l->in : -1);
        return -1;
    }
    /* One split, so part[t][0][row] is y[t][row].  Decode widths take the W8A16 GEMV (2); prompt
     * chunks take the tensor-core GEMM (2'). */
    if (l->prompt || rows > kDecodeRowsMax) {
        const dim3 grid((l->out + kMmaOut - 1) / kMmaOut, 1, (rows + kMmaTok - 1) / kMmaTok);
        qwen_w8a16_prefill_kernel<GR_PLAIN, kMmaNJ><<<grid, 128, 0, stream>>>((const uint8_t *)l->w, l->sf, (const __nv_bfloat16 *)x_bf16,
                                                                   l->out, l->in, rows, 1, y, gr_gate_args{});
    } else if (l->out >= kRowsLinearOutMin && l->in % kHeadKStep == 0) {
        /* L288: a wide Linear (the MTP fc, the PLE projections) takes the rows kernel at decode widths -- its weight
         * read once for all <= 16 rows; the narrow ones (GDN in_proj_a / _b, 48 rows: 3 MMA tiles) keep the GEMV,
         * each row split over 32 lanes, one token a CTA (below) */
        if ((((uintptr_t)l->w | (uintptr_t)x_bf16) & 15u) != 0) {
            fprintf(stderr, "pulsar: qwen mxfp8 linear: %d -> %d needs 16-byte aligned weight and rows -- refusing\n",
                    l->in, l->out);
            return -1;
        }
        static int announced = 0;
        if (!announced) {
            announced = 1;
            fprintf(stderr, "pulsar: L288 qwen mxfp8 linear at decode widths = the tensor-core rows kernel for >= %d "
                            "outputs (first: %d -> %d), the GEMV below\n", kRowsLinearOutMin, l->in, l->out);
        }
        const int rc = rows > 8 ? rows_launch<2>((const uint8_t *)l->w, l->sf, x_bf16, l->out, l->in, rows, 1, y, true, stream)
                                : rows_launch<1>((const uint8_t *)l->w, l->sf, x_bf16, l->out, l->in, rows, 1, y, true, stream);
        if (rc) {
            fprintf(stderr, "pulsar: qwen mxfp8 linear: %d -> %d at %d rows could not be launched -- refusing\n", l->in,
                    l->out, rows);
            return -1;
        }
    } else {
        /* L288 C2: one token a CTA -- the narrow Linear is out / 8 CTAs a token, so a verify's rows run side by side
         * (it was every row in one CTA's loop: 6 CTAs on 48 SMs at every width); a token's sum is its own, so the
         * bytes do not move */
        const int tb = gr_down_tokens_per_cta(l->in, 1) ? 1 : 0;
        if (tb == 0) {
            fprintf(stderr, "pulsar: qwen mxfp8 linear: a %d-wide row does not stage in %d bytes -- refusing\n", l->in,
                    kDownStageBytes);
            return -1;
        }
        const dim3 grid((l->out + 7) / 8, 1, (rows + tb - 1) / tb);
        qwen_gr_down_kernel<true, false><<<grid, 256, (size_t)tb * l->in * 2, stream>>>(l->w, l->sf, x_bf16, nullptr, 0,
                                                                                      l->out, l->in, rows, 1, tb, y);
    }
    const cudaError_t qe = cudaGetLastError();
    if (qe != cudaSuccess) {
        fprintf(stderr, "pulsar: qwen mxfp8 linear launch: %s\n", cudaGetErrorString(qe));
        return -1;
    }
    return 0;
}

/* L288 (work/qwen-rows): the ROWS kernel -- an mxfp8_lt weight against 1 .. 16 bf16 rows on the tensor cores,
 * reading the weight ONCE per call for all rows.  Two users: the HEAD -- lm_head (~248k x 2560) and the MTP draft
 * head (~82k x 2560), at every width the head runs -- and the GR read's W_down (320 x 10240, split-K 5) at decode
 * widths (the read's GEMV took 42 us at 9-16 rows vs 22 at <= 8 for the same reason as below).
 *
 * Why not the GEMV (2): it stages at most kDownTB = 8 rows a CTA, so a 9..16-row verify re-read the 0.66 GB head
 * per 8-row group, and inside a group it re-decoded every E4M3 code once per row on the CUDA cores -- 3.0 ms at
 * ~4 rows (the 0.66 GB floor at 273 GB/s is 2.4 ms) and 7.7 ms at 9-16 (L288 a2a profile: 4 streams).  This
 * kernel: 2.82 / 2.87 / 2.90 ms at 1 / 4 / 16 rows, ~230 GB/s (qwen_s4_gate section I times it).
 *
 * Arithmetic.  W = E4M3 * 2^(e8m0 - 127) is EXACT in bf16 (3 mantissa bits, a power-of-two scale), and so is the
 * activation (the mixer's bf16 row), so every product of mma.m16n8k16.bf16 is exact; only the summation order
 * differs from the GEMV (a fresh 16-k fragment, added into an f32 accumulator in k order -- the prefill GEMM's
 * pattern, 2').  A token's column of the MMA reads only its own activation, so a row's logits do not depend on how
 * many rows ride with it: a verify row's bytes equal a one-row decode's.
 *
 * Mapping.  A warp owns 16 weight rows (the MMA's M) and the call's tokens are its N (one or two n8 tiles).  The k
 * order inside a 64-wide chunk is permuted -- any permutation applied to both operands is the same dot product --
 * so that lane (g, c) loads 16 CONTIGUOUS codes of each of its two rows (one uint4 each) and the matching 16 bf16
 * of its token from shared memory (two uint4): logical k {2c, 2c+1 | 2c+8, 2c+9} of 16-k step s is actual
 * k 16c + 4s + {0, 1 | 2, 3}.  The activation rows are staged once per CTA (persistent CTAs, a grid-stride walk
 * over 16-row tiles); the row stride is padded to 16 mod 128 bytes, so a quarter-warp's eight 16-byte loads hit
 * eight distinct bank groups. */
namespace {
/// Warps per rows-kernel CTA (each owns 16-row weight tiles).
constexpr int kHeadWarps = 8;

/* 4 E4M3 codes (one word, code i in byte i) -> bf16 pairs lo = codes 0, 1 and hi = codes 2, 3, each the code's value
 * times the block scale, EXACTLY.  The code's sign and 7 magnitude bits, placed at bf16 bits 15 and 4..10, read as a
 * bf16 of value code x 2^-120 -- the normal codes and the subnormal ones alike (a bf16 subnormal), zero staying
 * zero; two exact power-of-two products (2^120, then the block's 2^(e8m0 - 127)) give the value. */
/** @param v a 32-bit word
 *  @return v reinterpreted as a bf16 pair */
__device__ __forceinline__ __nv_bfloat162 bf2_bits(uint32_t v) { return *reinterpret_cast<const __nv_bfloat162 *>(&v); }
/** @param v a bf16 pair
 *  @return its 32 bits */
__device__ __forceinline__ uint32_t bf2_u32(__nv_bfloat162 v) { return *reinterpret_cast<const uint32_t *>(&v); }
/** @param w  4 E4M3 codes, code i in byte i
 *  @param sc the block scale as a bf16 pair (mx_scale_bf16x2)
 *  @param lo out: codes 0, 1 times the scale, a bf16 pair
 *  @param hi out: codes 2, 3 times the scale, a bf16 pair */
__device__ __forceinline__ void e4m3x4_bf16(uint32_t w, __nv_bfloat162 sc, uint32_t &lo, uint32_t &hi) {
    const __nv_bfloat162 k120 = bf2_bits(0x7B807B80u);             /* 2^120 in each half */
    const uint32_t a = __byte_perm(w, 0u, 0x1404u), b = __byte_perm(w, 0u, 0x3424u);   /* code << 8 per half */
    const uint32_t ma = (a & 0x80008000u) | ((a >> 4) & 0x07F007F0u), mb = (b & 0x80008000u) | ((b >> 4) & 0x07F007F0u);
    lo = bf2_u32(__hmul2(__hmul2(bf2_bits(ma), k120), sc));
    hi = bf2_u32(__hmul2(__hmul2(bf2_bits(mb), k120), sc));
}
/* 2^(b - 127) as a bf16 pair: the E8M0 byte IS the bf16 exponent field (b = 0: 2^-127, a bf16 subnormal) */
/** @param b an E8M0 byte
 *  @return 2^(b - 127) in both halves */
__device__ __forceinline__ __nv_bfloat162 mx_scale_bf16x2(uint32_t b) {
    const uint32_t h = b ? b << 7 : 0x0040u;
    return bf2_bits(h | h << 16);
}

/** The W8A16 rows kernel (decode widths, W read once for all rows; the head and the read's W_down).
 *  Grid = (CTAs walking 16-row tiles, n_split), block = 32 * kHeadWarps, dynamic smem NT * 8 staged token rows of
 *  (in / n_split) * 2 + 16 bytes.
 *  @tparam NT n8 token tiles (1 or 2: up to 8 or 16 rows)
 *  @param wq      E4M3 weight codes [out][in], 16-B aligned rows
 *  @param wsf     the weight's E8M0 plane (mxfp8_lt swizzle)
 *  @param x       bf16 activations [T][in]
 *  @param out     weight rows
 *  @param in      weight columns (n_split slices, each a whole number of kHeadKStep)
 *  @param T       tokens, at most NT * 8
 *  @param n_split split-K count (1: y is [T][out])
 *  @param y       out, f32 [T][n_split][out] */
template <int NT>
__global__ void __launch_bounds__(32 * kHeadWarps)
qwen_w8a16_rows_kernel(const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsf,
                       const __nv_bfloat16 *__restrict__ x, int out, int in, int T, int n_split,
                       float *__restrict__ y) {
    extern __shared__ __align__(16) uint8_t hx[];
    /* split-K (the GR read's W_down): CTA row blockIdx.y owns k [k_lo, k_lo + kin) and stages only that slice;
     * y is then part[t][split][row], the GEMV's layout (n_split 1: y[t][row]) */
    const int split = blockIdx.y, kin = in / n_split, k_lo = split * kin;
    const int stride = kin * 2 + 16;              /* bytes per staged token row: 16 mod 128 */
    const int n16 = kin / 8;
    for (int i = threadIdx.x; i < NT * 8 * n16; i += blockDim.x) {
        const int t = i / n16, q = i - t * n16;
        const uint4 v = t < T ? reinterpret_cast<const uint4 *>(x + (size_t)t * in + k_lo)[q] : make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4 *>(hx + (size_t)t * stride + q * 16) = v;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int kbp = pulsar_mx_kbp(in), n_tiles = (out + 15) / 16;
    const uint8_t *xs = hx + 32 * c;
    for (int tile = blockIdx.x * kHeadWarps + warp; tile < n_tiles; tile += gridDim.x * kHeadWarps) {
        const int r0 = tile * 16 + g, r1 = r0 + 8;
        const bool v0 = r0 < out, v1 = r1 < out;
        const uint4 *w0 = reinterpret_cast<const uint4 *>(wq + (size_t)(v0 ? r0 : 0) * in + 16 * c);
        const uint4 *w1 = reinterpret_cast<const uint4 *>(wq + (size_t)(v1 ? r1 : 0) * in + 16 * c);
        float acc[NT][4];
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;
        /* the step's codes (4 chunks x 2 rows) and its 8 MX blocks' scales (two words a row); the NEXT step's are
         * loaded before this step's MMAs, so a warp keeps a step's bytes in flight */
        uint4 a0[kHeadUnroll], a1[kHeadUnroll];
        uint32_t s0[2], s1[2];
        auto load = [&](int k0) {
            const int ch0 = k0 / kHeadChunk, kb = k0 / 32;   /* kb a multiple of 8: two aligned 4-block words */
#pragma unroll
            for (int u = 0; u < kHeadUnroll; ++u) {
                a0[u] = v0 ? __ldcs(w0 + (ch0 + u) * (kHeadChunk / 16)) : make_uint4(0u, 0u, 0u, 0u);
                a1[u] = v1 ? __ldcs(w1 + (ch0 + u) * (kHeadChunk / 16)) : make_uint4(0u, 0u, 0u, 0u);
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                s0[h] = v0 ? *reinterpret_cast<const uint32_t *>(wsf + pulsar_mx_sfoff(r0, kb + 4 * h, kbp)) : 0u;
                s1[h] = v1 ? *reinterpret_cast<const uint32_t *>(wsf + pulsar_mx_sfoff(r1, kb + 4 * h, kbp)) : 0u;
            }
        };
        load(k_lo);
        for (int k0 = k_lo; k0 < k_lo + kin; k0 += kHeadKStep) {
            uint32_t A0[kHeadUnroll][8], A1[kHeadUnroll][8];   /* bf16 pairs: [2s] = k 4s, 4s+1; [2s+1] = k 4s+2, 4s+3 */
#pragma unroll
            for (int u = 0; u < kHeadUnroll; ++u) {
                /* lane c's 16 codes of chunk u are k 64u + 16c .. +15: MX block 2u + c / 2 of the step */
                const int b = 2 * u + (c >> 1), sh = 8 * (b & 3);
                const __nv_bfloat162 sc0 = mx_scale_bf16x2((s0[b >> 2] >> sh) & 0xFFu);
                const __nv_bfloat162 sc1 = mx_scale_bf16x2((s1[b >> 2] >> sh) & 0xFFu);
                const uint32_t *p0 = reinterpret_cast<const uint32_t *>(&a0[u]);
                const uint32_t *p1 = reinterpret_cast<const uint32_t *>(&a1[u]);
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    e4m3x4_bf16(p0[w], sc0, A0[u][2 * w], A0[u][2 * w + 1]);
                    e4m3x4_bf16(p1[w], sc1, A1[u][2 * w], A1[u][2 * w + 1]);
                }
            }
            if (k0 + kHeadKStep < k_lo + kin) load(k0 + kHeadKStep);
#pragma unroll
            for (int u = 0; u < kHeadUnroll; ++u) {
                const int xoff = (k0 - k_lo + u * kHeadChunk) * 2;
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const uint8_t *xr = xs + (size_t)(j * 8 + g) * stride + xoff;
                    const uint4 q0 = *reinterpret_cast<const uint4 *>(xr);
                    const uint4 q1 = *reinterpret_cast<const uint4 *>(xr + 16);
                    const uint32_t B[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        const uint32_t af[4] = {A0[u][2 * s], A1[u][2 * s], A0[u][2 * s + 1], A1[u][2 * s + 1]};
                        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        mma_bf16_16816(d, af, B[2 * s], B[2 * s + 1]);
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc[j][e] += d[e];
                    }
                }
            }
        }
        /* acc[j][0..1] = row r0, tokens 8j + 2c, +1; acc[j][2..3] = row r1 */
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int t = 8 * j + 2 * c + (e & 1), row = e < 2 ? r0 : r1;
                if (t < T && (e < 2 ? v0 : v1)) y[((size_t)t * n_split + split) * out + row] = acc[j][e];
            }
    }
}

/* One launch of the rows kernel.  `persistent`: as many CTAs as fit the GPU, each walking 16-row tiles (the
 * head: thousands of tiles); otherwise one tile a warp (a split-K W_down: 20 tiles x 5 splits).  The dynamic
 * shared memory a CTA stages is NT x 8 rows of the split's slice. */
/** @tparam NT n8 token tiles (1 or 2)
 *  @param wq         E4M3 weight codes [out][in]
 *  @param sf         the weight's E8M0 plane
 *  @param x_bf16     bf16 activations [rows][in]
 *  @param out        weight rows
 *  @param in         weight columns
 *  @param rows       tokens, at most NT * 8
 *  @param n_split    split-K count
 *  @param y          out, f32 [rows][n_split][out]
 *  @param persistent cap the grid at the resident CTAs (the head) instead of one tile a warp
 *  @param stream     launch stream
 *  @return 0, or -1 when the device query, the smem attribute or the occupancy query fails */
template <int NT>
int rows_launch(const uint8_t *wq, const uint8_t *sf, const uint16_t *x_bf16, int out, int in, int rows, int n_split,
                float *y, bool persistent, cudaStream_t stream) {
    /* per device (L335: process statics, right for the first device only); the attribute grows to the widest slice */
    static int sms_of[PULSAR_CUDA_DEVICE_SLOTS] = {};
    static int smem_set[PULSAR_CUDA_DEVICE_SLOTS] = {};
    const size_t smem = (size_t)NT * 8 * (in / n_split * 2 + 16);
    const int d = pulsar_cuda_device_slot();
    if (d < 0) return -1;
    if (!sms_of[d] && cudaDeviceGetAttribute(&sms_of[d], cudaDevAttrMultiProcessorCount, d) != cudaSuccess) {
        sms_of[d] = 0;
        return -1;
    }
    const int sms = sms_of[d];
    if (!pulsar_cuda_smem_opt_in((const void *)qwen_w8a16_rows_kernel<NT>, smem_set, smem)) {
        fprintf(stderr, "pulsar: qwen mxfp8 rows: %d rows of a %d-wide slice do not fit an SM -- refusing\n", NT * 8,
                in / n_split);
        return -1;
    }
    const int tiles = (out + 15) / 16;
    int gx = (tiles + kHeadWarps - 1) / kHeadWarps;
    if (persistent) {
        int per = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, qwen_w8a16_rows_kernel<NT>, 32 * kHeadWarps, smem) !=
                cudaSuccess || per < 1)
            return -1;
        gx = gx < sms * per ? gx : sms * per;
    }
    qwen_w8a16_rows_kernel<NT><<<dim3(gx, n_split), 32 * kHeadWarps, smem, stream>>>(
        wq, sf, (const __nv_bfloat16 *)x_bf16, out, in, rows, n_split, y);
    return 0;
}
} // namespace

/* L251: the lm_head as MXFP8, made once at load from the container's bf16 head.  The bf16 head
 * is 1.27 GB read per decode row (5.1 ms of a 33 ms token, nsys); MXFP8 is half the bytes and
 * the W8A16 arms above read it.  One warp per (row, 32-block), the producers' own encoder
 * (pulsar_mx_emit_block: the shared exponent, E4M3, the swizzled E8M0 plane), so the result is
 * the same mxfp8_lt layout the container's own MXFP8 tensors carry. */
namespace {
/** One warp per (row, 32-block): 8 tasks a 256-thread block, grid = ceil(out * in / 32 / 8).
 *  @param w     bf16 weight [n][in] (the source head)
 *  @param rows  int32 [out] source rows to gather, or NULL (row r is w's row r)
 *  @param out   rows to encode
 *  @param in    columns (a multiple of 32)
 *  @param kbp   pulsar_mx_kbp(in)
 *  @param data  out, E4M3 [out][in]
 *  @param scale out, the swizzled E8M0 plane (zeroed by the caller first) */
__global__ void __launch_bounds__(256)
qwen_bf16_to_mxfp8_kernel(const __nv_bfloat16 *__restrict__ w, const int32_t *__restrict__ rows, int out, int in,
                          int kbp, __nv_fp8_e4m3 *__restrict__ data, unsigned char *__restrict__ scale) {
    const int64_t task = (int64_t)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int nblk = in / 32;
    if (task >= (int64_t)out * nblk) return;
    const int row = (int)(task / nblk), blk = (int)(task % nblk), lane = threadIdx.x & 31;
    const uint32_t col = (uint32_t)(blk * 32 + lane);
    const size_t src = rows ? (size_t)rows[row] : (size_t)row;      /* the gathered form: row r is w's rows[r] */
    const float v = __bfloat162float(w[src * in + col]);
    pulsar_mx_emit_block(v, col, (uint32_t)row, (uint32_t)in, kbp, data, scale);
}
} // namespace

extern "C" uint64_t pulsar_qwen_mxfp8_bytes(int out, int in) {
    if (out <= 0 || in <= 0 || in % 32) return 0;
    return (uint64_t)out * in + (uint64_t)pulsar_mx_sf_slab_bytes(out, pulsar_mx_kbp(in));
}

extern "C" int pulsar_qwen_bf16_to_mxfp8(const uint16_t *w, const int32_t *rows, int out, int in, void *dst,
                                         cudaStream_t stream) {
    const uint64_t bytes = pulsar_qwen_mxfp8_bytes(out, in);
    if (!w || !dst || !bytes) {
        fprintf(stderr, "pulsar: qwen bf16 -> MXFP8: %d x %d refused (in %% 32, non-null)\n", out, in);
        return -1;
    }
    uint8_t *data = (uint8_t *)dst, *scale = data + (uint64_t)out * in;
    /* the scale slab must start zeroed: sfoff leaves holes when rows or blocks are not multiples */
    if (cudaMemsetAsync(scale, 0, bytes - (uint64_t)out * in, stream) != cudaSuccess) return -1;
    const int64_t tasks = (int64_t)out * (in / 32);
    qwen_bf16_to_mxfp8_kernel<<<(unsigned)((tasks + 7) / 8), 256, 0, stream>>>(
        (const __nv_bfloat16 *)w, rows, out, in, pulsar_mx_kbp(in), (__nv_fp8_e4m3 *)data, scale);
    return launch_ok("bf16 -> mxfp8") ? 0 : -1;
}

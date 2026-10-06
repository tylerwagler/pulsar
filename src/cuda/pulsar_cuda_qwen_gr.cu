/* Qwen3.8-Flash-Next Gated Residual (L251 S4): the 4-stream read, the write,
 * and the collapse-only top-level mixer.  Source: transformers'
 * Qwen4ExpTextGatedResidual; contracts in pulsar_cuda_qwen.h.
 *
 * The read, per token (4 streams x 2560, bf16 storage), three launches:
 *   1. norm    xn_s = x_s * rsqrt(mean(x_s^2) + eps) * (1 + w_s)   one CTA per
 *              (stream, token); xn leaves in the format W_down reads and the
 *              per-stream rstd is kept, so the last kernel recomputes xn with
 *              the same three operations instead of storing it; the inject
 *              dot W_inj . xn is taken here on the f32 xn as per-stream partials
 *   2. down    d = W_down xn (320 x 10240), split-K in 5 fixed splits
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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdio.h>

namespace {

constexpr int kH = PULSAR_QWEN_HIDDEN;
constexpr int kS = PULSAR_QWEN_HC;
constexpr int kHC = PULSAR_QWEN_HC_HIDDEN;
constexpr int kR = PULSAR_QWEN_HC_LOWRANK;
constexpr float kEps = 1e-6f;                 ///< rms_norm_eps
constexpr int kNormThreads = 256;
constexpr int kNormPer = kH / kNormThreads;   ///< 10 columns per thread
constexpr int kDownSplit = 5;                 ///< W_down split-K: 320 blocks of 32 -> 64 per split
constexpr int kDownTB = 8;                    ///< tokens per down CTA
constexpr int kDownStageBytes = 48 * 1024;    ///< the W8A16 GEMV's staged activation slice, at most
constexpr int kUpTB = 16;                     ///< tokens per up CTA, BF16 mixer at prefill widths
constexpr int kUpTBDecode = 4;                ///< tokens per up CTA at decode widths: small enough that two
                                              ///< CTAs (staged W_up tile + a, prod) share an SM, so the 80
                                              ///< CTAs are one wave on 48 SMs.  The row arithmetic does not
                                              ///< depend on the tile, so the two widths are bit-identical.
static_assert(kH % kNormThreads == 0, "the norm walks a stream in whole passes of 256");
static_assert((kHC / 32) % kDownSplit == 0, "W_down's 32-blocks split evenly");
static_assert(kR % 32 == 0 && kR / 32 <= 12, "the low rank is whole 32-blocks");

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }

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
__device__ __forceinline__ float mx_scale2(unsigned sw, unsigned sx) {
    return exp2f((float)((int)sw + (int)sx - 254));
}

/* One E8M0 byte as its scale: 2^(b-127).  mx_scale2's single-operand form. */
__device__ __forceinline__ float mx_scale1(unsigned sb) {
    return exp2f((float)((int)sb - 127));
}

/* 32 E4M3 weight codes against 32 BF16 activations (L251 / ac69748f, the W8A16 shape).
 * The W8 weights arrive exactly as they do in e4m3_dot32 above (2 uint4 = 32 bytes = 32 codes);
 * the activation is the block input's bf16 row, so 32 of them are 4 uint4 (2 bf16 per uint32).
 * The scale is the weight's E8M0 alone -- bf16 carries its own exponent, so there is no second
 * block scale to fold in.  The E4M3 decode is the same `__nv_cvt_fp8x2_to_halfraw2` the A8 path
 * uses, so the weights are read bit-identically to W8A8; only the activation operand changed. */
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

/* The block reduction every kernel here uses: warp xor tree, then the warps
 * in order.  `red` holds one float per warp. */
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
        xb[(size_t)t * kHC + s * kH + c] = __float2bfloat16(xn);
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
 * Decode widths keep the GEMV (2): a decode kernel run over a prompt re-reads the activation once per
 * 8 weight rows (40x per call for W_down), which was 22% of a 4096-row prefill.  The two kernels sum in
 * different orders, so a prefilled row and a decoded row agree to rounding, not to the bit (Tyler
 * 2026-09-29: "Don't use a decode kernel for prefill"); within this kernel a row's arithmetic depends
 * on nothing but its own x and W, so the chunk width changes no bit.  CTA = 4 warps = 64 tokens x 32
 * outputs (warp w: tokens 16w..16w+15, four n8 tiles); kMmaStage MX blocks staged per barrier. */
constexpr int kDecodeRowsMax = 16;               ///< widths at or below take the decode GEMV
constexpr int kMmaTok = 64, kMmaOut = 32, kMmaStage = 4;
constexpr int kMmaKs  = kMmaStage * 32;          ///< k per stage
constexpr int kMmaPad = kMmaKs + 8;              ///< smem row stride in bf16: 16-byte rows, conflict-free ldmatrix

__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void gr_ldsm_x4(uint32_t (&r)[4], const void *p) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

__device__ __forceinline__ void gr_ldsm_x2(uint32_t &r0, uint32_t &r1, const void *p) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(s));
}

__device__ __forceinline__ void gr_cp_async16(void *smem, const void *gmem, int src_bytes) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(src_bytes));
}

__global__ void __launch_bounds__(128)
qwen_w8a16_prefill_kernel(const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsf,
                      const __nv_bfloat16 *__restrict__ x, int out, int in, int T, int n_split,
                      float *__restrict__ part) {
    __shared__ __align__(16) __nv_bfloat16 sx[kMmaTok][kMmaPad];
    __shared__ __align__(16) __nv_bfloat16 sw[kMmaOut][kMmaPad];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int r0 = blockIdx.x * kMmaOut, split = blockIdx.y, t0 = blockIdx.z * kMmaTok;
    const int nblk = in / 32, per = nblk / n_split, b_lo = split * per, b_hi = b_lo + per;
    const int w_kbp = pulsar_mx_kbp(in);
    float acc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;

    for (int b0 = b_lo; b0 < b_hi; b0 += kMmaStage) {
        const int nbs = min(kMmaStage, b_hi - b0);
        /* x: kMmaTok rows x nbs blocks of 32 bf16 = 4 16-byte chunks a block; rows past T read zero */
        for (int c = tid; c < kMmaTok * kMmaStage * 4; c += 128) {
            const int tr = c / (kMmaStage * 4), ch = c % (kMmaStage * 4), t = t0 + tr;
            if (ch < nbs * 4) {
                const bool ok = t < T;
                gr_cp_async16(&sx[tr][ch * 8], x + (size_t)(ok ? t : 0) * in + (size_t)b0 * 32 + ch * 8, ok ? 16 : 0);
            }
        }
        /* W: kMmaOut rows x nbs blocks, 8 codes a thread-task, dequantized exactly into bf16 */
        for (int c = tid; c < kMmaOut * kMmaStage * 4; c += 128) {
            const int wr = c / (kMmaStage * 4), ch = c % (kMmaStage * 4), row = r0 + wr;
            if (ch >= nbs * 4) continue;
            uint32_t o[4] = {0u, 0u, 0u, 0u};
            if (row < out) {
                const int b = b0 + (ch >> 2);
                const uint2 codes = *reinterpret_cast<const uint2 *>(wq + (size_t)row * in + (size_t)b0 * 32 + ch * 8);
                const float s = mx_scale1(wsf[pulsar_mx_sfoff(row, b, w_kbp)]);
                const uint8_t *cb = reinterpret_cast<const uint8_t *>(&codes);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    __nv_fp8_e4m3 q0, q1;
                    q0.__x = cb[2 * e];
                    q1.__x = cb[2 * e + 1];
                    const __nv_bfloat162 v = __floats2bfloat162_rn(float(q0) * s, float(q1) * s);
                    o[e] = *reinterpret_cast<const uint32_t *>(&v);
                }
            }
            *reinterpret_cast<uint4 *>(&sw[wr][ch * 8]) = make_uint4(o[0], o[1], o[2], o[3]);
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        for (int kk = 0; kk < nbs * 32; kk += 16) {
            uint32_t a[4];
            gr_ldsm_x4(a, &sx[warp * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
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
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int t = t0 + warp * 16 + g + 8 * (e >> 1), row = r0 + j * 8 + c2 + (e & 1);
            if (t < T && row < out) part[((size_t)t * n_split + split) * out + row] = acc[j][e];
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
 *        the stream mean and emits the row. */
/* The W8 up's weight tile (decode widths), staged whole: the CTA's 4 x 32 rows of 320 E4M3 bytes, copied by
 * cp.async in coalesced 16-byte chunks at kernel entry so the load overlaps the mid.  The row
 * stride is padded to 336 B (84 words): a quarter-warp's 8 lanes, reading rows l .. l + 7 at the
 * same column, then hit 8 distinct 4-bank groups.  The arithmetic reads the same bytes in the same
 * order as a direct global load, so the output is bit-identical to it. */
constexpr int kUpRowBytes = kR;                               ///< 320 E4M3 codes
constexpr int kUpRowStride = kUpRowBytes + 16;                ///< 336 B
constexpr int kUpTileBytes = kS * 32 * kUpRowStride;          ///< 43,008 B dynamic shared memory
static_assert(kUpRowBytes % 16 == 0 && (kUpRowStride / 4) % 32 == 20, "the padded stride is conflict-free");

__device__ __forceinline__ void gr_cp_async16(void *smem, const void *gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}

template <bool W8, int TB>
__global__ void __launch_bounds__(32 * kS)
qwen_gr_up_kernel(const void *__restrict__ wv, const uint8_t *__restrict__ wsf,
                  const float *__restrict__ part, const float *__restrict__ injp, float *__restrict__ inj,
                  const __nv_bfloat16 *__restrict__ streams, const __nv_bfloat16 *__restrict__ norm_w,
                  const float *__restrict__ rstd, int T,
                  __nv_bfloat16 *__restrict__ x_out) {
    constexpr int NB = kR / 32;
    __shared__ float prod[TB][kS][32];
    /* L251 / ac69748f: `a` is bf16 whatever W_up's WEIGHT format is -- this was the last op-internal
     * E4M3 activation in the read.  W8 now means E4M3 weights against a bf16 activation, the same
     * W8A16 shape the down already uses, so e4m3_bf16_dot32 serves here too. */
    __shared__ __align__(16) uint8_t s_a[TB][2 * kR];
    const int s = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c = blockIdx.x * 32 + lane;
    const int row = s * kH + c;
    const int t0 = blockIdx.y * TB, nt = min(TB, T - t0);
    /* W8 runs this kernel at decode widths only (its prefill is 3'), and stages W_up there */
    static_assert(!W8 || TB == kUpTBDecode, "the W8 up at prefill widths is the GEMM (3')");
    constexpr bool STAGE = W8;
    extern __shared__ __align__(16) uint8_t s_w[];
    if constexpr (STAGE) {
        /* stream ss's 32 rows are one contiguous 10,240 B run of W_up */
        constexpr int kChunks = kUpRowBytes / 16;                 /* 20 per row */
        for (int i = threadIdx.x; i < kS * 32 * kChunks; i += 32 * kS) {
            const int rr = i / kChunks, ch = i % kChunks, ss = rr / 32, cc = rr % 32;
            const uint8_t *src = (const uint8_t *)wv + ((size_t)ss * kH + blockIdx.x * 32 + cc) * kR + ch * 16;
            gr_cp_async16(s_w + rr * kUpRowStride + ch * 16, src);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
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
    if constexpr (STAGE) asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
    float z[TB];
#pragma unroll
    for (int tt = 0; tt < TB; ++tt) z[tt] = 0.0f;
    if constexpr (W8) {
        const int w_kbp = pulsar_mx_kbp(kR);
        const uint4 *wp = reinterpret_cast<const uint4 *>(s_w + (s * 32 + lane) * kUpRowStride);
        for (int b = 0; b < NB; ++b) {
            const uint4 w0 = wp[2 * b], w1 = wp[2 * b + 1];
            const unsigned sw = wsf[pulsar_mx_sfoff(row, b, w_kbp)];
#pragma unroll
            for (int tt = 0; tt < TB; ++tt) {
                if (tt < nt) {
                    /* 32 bf16 activations are FOUR uint4 where 32 E4M3 were two, and the weight's
                     * E8M0 is the only scale left to apply (bf16 carries its own exponent). */
                    const uint4 *ap = reinterpret_cast<const uint4 *>(s_a[tt]) + 4 * b;
                    z[tt] = fmaf(e4m3_bf16_dot32(w0, w1, ap[0], ap[1], ap[2], ap[3]),
                                 mx_scale1(sw), z[tt]);
                }
            }
        }
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

__global__ void qwen_gr_write_kernel(__nv_bfloat16 *__restrict__ streams, const float *__restrict__ out,
                                     const float *__restrict__ inj, int T) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * kHC) return;
    const size_t t = i / kHC;
    const int s = (int)((i / kH) % kS), c = (int)(i % kH);
    streams[i] = __float2bfloat16(bf2f(streams[i]) + out[t * kH + c] * inj[t * kS + s]);
}

/* 3'. PREFILL (W8, T > kDecodeRowsMax): the up as a tensor-core GEMM.  The up kernel above reads its
 * W_up row once per 16-token CTA on the CUDA cores (0.49 s of a 4096-row prefill); here
 *   mid   a = silu(d / 4) for every (token, rank) into a bf16 [T][kR] buffer, and inj -- the same
 *         three operations as the up kernel's mid, so `a` is bit-identical to its shared copy;
 *   GEMM  z [T][kHC] = W_up a, the W8A16 prefill kernel (2') with one split;
 *   gate  x = (g_0 xn_0 + g_1 xn_1 + g_2 xn_2 + g_3 xn_3) / 4, g_s = sigmoid(z_s), one thread per
 *         (token, channel), the up kernel's emit.
 * Only z's summation order differs from the decode kernel, so a prefilled row agrees with a decoded one
 * to rounding (the same contract as 2'). */
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

__global__ void __launch_bounds__(256)
qwen_gr_gate_kernel(const float *__restrict__ z, const __nv_bfloat16 *__restrict__ streams,
                    const __nv_bfloat16 *__restrict__ norm_w, const float *__restrict__ rstd, int T,
                    __nv_bfloat16 *__restrict__ x_out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)T * kH) return;
    const int t = (int)(i / kH), c = (int)(i % kH);
    float p[kS];
#pragma unroll
    for (int s = 0; s < kS; ++s) {
        const int row = s * kH + c;
        const float g = 1.0f / (1.0f + expf(-z[(size_t)t * kHC + row]));
        const float xn = bf2f(streams[(size_t)t * kHC + row]) * rstd[t * kS + s] * (1.0f + bf2f(norm_w[row]));
        p[s] = g * xn;
    }
    x_out[i] = __float2bfloat16((((p[0] + p[1]) + p[2]) + p[3]) * (1.0f / kS));
}

struct gr_ws {
    uint8_t *xn;                      /* the norm's bf16 activation row (L251 / ac69748f) */
    float *rstd, *injp, *part;
    __nv_bfloat16 *a;                 /* prefill only: silu(d / 4), [T][kR] */
    float *z;                         /* prefill only: W_up a, [T][kHC] */
};

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
    m.z = (float *)take((size_t)T * kHC * 4);
    if (o) *o = m;
    return failed ? 0 : used;
}

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
static int gr_down_tokens_per_cta(int in, int n_split) {
    const int slice = in / n_split * 2;
    const int tb = kDownStageBytes / slice;
    return tb < 1 ? 0 : tb < kDownTB ? tb : kDownTB;
}

template <bool W8>
static void gr_read_kernels(const pulsar_qwen_gr_dev *w, const uint16_t *streams, int T, uint16_t *x_bf16,
                            float *inj, const gr_ws &m, cudaStream_t stream) {
    qwen_gr_norm_kernel<<<dim3(kS, T), kNormThreads, 0, stream>>>(
        (const __nv_bfloat16 *)streams, (const __nv_bfloat16 *)w->norm_w, (const __nv_bfloat16 *)w->inject,
        (__nv_bfloat16 *)m.xn, m.rstd, m.injp);
    /* W8 selects the WEIGHT format only; the activation is bf16 either way -- the W8A16 arm, on the
     * tensor cores (2'). */
    if (W8 && (w->prompt || T > kDecodeRowsMax)) {
        qwen_w8a16_prefill_kernel<<<dim3((kR + kMmaOut - 1) / kMmaOut, kDownSplit, (T + kMmaTok - 1) / kMmaTok), 128, 0, stream>>>(
            (const uint8_t *)w->down.w, w->down.sf, (const __nv_bfloat16 *)m.xn, kR, kHC, T, kDownSplit, m.part);
    } else {
        const int tb = gr_down_tokens_per_cta(kHC, kDownSplit);
        static_assert(kHC / kDownSplit * 2 * kDownTB <= kDownStageBytes, "W_down stages a whole token tile");
        qwen_gr_down_kernel<W8, false><<<dim3((kR + 7) / 8, kDownSplit, (T + tb - 1) / tb), 256,
                                         W8 ? (size_t)tb * kHC / kDownSplit * 2 : 0, stream>>>(
            w->down.w, w->down.sf, m.xn, nullptr, 0, kR, kHC, T, kDownSplit, tb, m.part);
    }
    if (W8 && (w->prompt || T > kDecodeRowsMax)) {
        const int64_t nm = (int64_t)T * kR;
        qwen_gr_mid_kernel<<<(unsigned)((nm + 255) / 256), 256, 0, stream>>>(m.part, w->inject ? m.injp : nullptr, inj,
                                                                           T, m.a);
        qwen_w8a16_prefill_kernel<<<dim3((kHC + kMmaOut - 1) / kMmaOut, 1, (T + kMmaTok - 1) / kMmaTok), 128, 0, stream>>>(
            (const uint8_t *)w->up.w, w->up.sf, m.a, kHC, kR, T, 1, m.z);
        const int64_t ng = (int64_t)T * kH;
        qwen_gr_gate_kernel<<<(unsigned)((ng + 255) / 256), 256, 0, stream>>>(
            m.z, (const __nv_bfloat16 *)streams, (const __nv_bfloat16 *)w->norm_w, m.rstd, T, (__nv_bfloat16 *)x_bf16);
        return;
    }
    if (W8) {
        static bool attr = false;                 /* idempotent; set once, not per launch */
        if (!attr) {
            cudaFuncSetAttribute(qwen_gr_up_kernel<true, kUpTBDecode>, cudaFuncAttributeMaxDynamicSharedMemorySize, kUpTileBytes);
            cudaFuncSetAttribute(qwen_gr_up_kernel<true, kUpTBDecode>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
            attr = true;
        }
    }
    /* W8 is at decode widths here; the BF16 mixer takes the 16-token tile at prefill widths */
    const int tb = !W8 && (w->prompt || T > kDecodeRowsMax) ? kUpTB : kUpTBDecode;
    const dim3 grid(kH / 32, (T + tb - 1) / tb);
    const size_t smem = W8 ? kUpTileBytes : 0;
    const float *injp = w->inject ? m.injp : nullptr;
    if constexpr (!W8) {
        if (tb == kUpTB) {
            qwen_gr_up_kernel<W8, kUpTB><<<grid, 32 * kS, smem, stream>>>(
                w->up.w, w->up.sf, m.part, injp, inj, (const __nv_bfloat16 *)streams, (const __nv_bfloat16 *)w->norm_w,
                m.rstd, T, (__nv_bfloat16 *)x_bf16);
            return;
        }
    }
    qwen_gr_up_kernel<W8, kUpTBDecode><<<grid, 32 * kS, smem, stream>>>(
        w->up.w, w->up.sf, m.part, injp, inj, (const __nv_bfloat16 *)streams, (const __nv_bfloat16 *)w->norm_w,
        m.rstd, T, (__nv_bfloat16 *)x_bf16);
}

extern "C" int pulsar_qwen_gr_read_launch(const pulsar_qwen_gr_dev *w, const uint16_t *streams, int T,
                                          uint16_t *x_bf16, float *inj,
                                          void *ws, size_t ws_bytes, cudaStream_t stream) {
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
        fprintf(stderr, "pulsar: L251 qwen GR read = grouped norm -> %s W_down (split-K %d) -> silu/4 -> %s W_up "
                        "sigmoid gate, stream mean -> bf16 row; streams bf16, activation bf16\n",
                w8 ? "MXFP8" : "BF16", kDownSplit, w8 ? "MXFP8" : "BF16");
    }
    /* L251 / ac69748f: there is no scale slab to clear and no slot to fill -- xn is bf16, so the
     * norm writes every element the down GEMV reads and nothing else owns this buffer. */
    if (w8) gr_read_kernels<true>(w, streams, T, x_bf16, inj, m, stream);
    else    gr_read_kernels<false>(w, streams, T, x_bf16, inj, m, stream);
    return launch_ok("read") ? 0 : -3;
}

extern "C" int pulsar_qwen_gr_write_launch(uint16_t *streams, const float *out, const float *inj, int T,
                                           cudaStream_t stream) {
    if (!streams || !out || !inj || T <= 0) {
        fprintf(stderr, "pulsar: qwen GR write: a null input -- refusing\n");
        return -1;
    }
    const size_t n = (size_t)T * kHC;
    qwen_gr_write_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>((__nv_bfloat16 *)streams, out, inj, T);
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
        qwen_w8a16_prefill_kernel<<<grid, 128, 0, stream>>>((const uint8_t *)l->w, l->sf, (const __nv_bfloat16 *)x_bf16,
                                                            l->out, l->in, rows, 1, y);
    } else {
        const int tb = gr_down_tokens_per_cta(l->in, 1);
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

/* L251: the lm_head as MXFP8, made once at load from the container's bf16 head.  The bf16 head
 * is 1.27 GB read per decode row (5.1 ms of a 33 ms token, nsys); MXFP8 is half the bytes and
 * the W8A16 arms above read it.  One warp per (row, 32-block), the producers' own encoder
 * (pulsar_mx_emit_block: the shared exponent, E4M3, the swizzled E8M0 plane), so the result is
 * the same mxfp8_lt layout the container's own MXFP8 tensors carry. */
namespace {
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

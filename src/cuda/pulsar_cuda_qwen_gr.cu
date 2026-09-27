/* Qwen3.8-Flash-Next Gated Residual (L251 S4): the 4-stream read, the write,
 * and the collapse-only top-level mixer.  Source: transformers'
 * Qwen4ExpTextGatedResidual; contracts in pulsar_cuda_qwen.h.
 *
 * The read, per token (4 streams x 2560, bf16 storage):
 *   1. norm    xn_s = x_s * rsqrt(mean(x_s^2) + eps) * (1 + w_s)   one CTA per
 *              (stream, token); xn leaves as E4M3 (the W_down input) and the
 *              per-stream rstd is kept, so later kernels recompute xn with the
 *              same three operations instead of storing it; the inject dot
 *              W_inj . xn is taken here on the f32 xn as per-stream partials
 *   2. down    d = W_down xn (320 x 10240, MXFP8), split-K in 5 fixed splits
 *   3. mid     a = silu(d / 4) -> E4M3 (the W_up input); inj = 2 sigmoid(z / 4)
 *              with z the partials summed in stream order
 *   4. up      for each channel c, the four rows s * 2560 + c of W_up
 *              (10240 x 320, MXFP8) in one CTA: g_s = sigmoid(W_up a),
 *              x = (g_0 xn_0 + g_1 xn_1 + g_2 xn_2 + g_3 xn_3) / 4, rounded to
 *              bf16 (the source's `mixed_input` dtype), emitted as the block
 *              input's bf16 row AND its E4M3 encoding (one value, the encoding
 *              of the bf16 value)
 * The write: streams_s += out * inj_s, f32 math, one bf16 rounding.
 *
 * Two low-rank weight formats (pulsar_qwen_lowrank): MXFP8 at every per-layer
 * site (xn and a leave as E4M3, the products are exact, the sums f32) and BF16
 * at the top-level mixer, as the graded recipe stores it (xn and a leave as
 * bf16 -- the source's own dtype there -- and the GEMVs are bf16 x bf16 into
 * f32).  One template parameter, W8, picks the pair; the norm, the gate, the
 * mean and the emit are one code.
 *
 * Fusion.  The paper runs the read and the write as one kernel each; here the
 * read is four launches (the two low-rank GEMVs need the whole token row and
 * the whole weight, which one CTA cannot hold).  The down GEMV is the byte
 * cost (3.3 MB of MXFP8 per site); the rest is small.  Fusing the write with
 * the next site's norm (they touch the same row back to back) is the obvious
 * next step once a caller exists to measure it in.
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
constexpr int kUpTB = 16;                     ///< tokens per up CTA
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

/* 1. norm (+ the inject partials); xn leaves in the format W_down reads */
template <bool W8>
__global__ void __launch_bounds__(kNormThreads)
qwen_gr_norm_kernel(const __nv_bfloat16 *__restrict__ streams, const __nv_bfloat16 *__restrict__ norm_w,
                    const __nv_bfloat16 *__restrict__ inject,
                    __nv_fp8_e4m3 *__restrict__ xq, unsigned char *__restrict__ xsf, int kbp,
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
        if constexpr (W8) pulsar_mx_emit_block(xn, (uint32_t)(s * kH + c), (uint32_t)t, (uint32_t)kHC, kbp, xq, xsf);
        else xb[(size_t)t * kHC + s * kH + c] = __float2bfloat16(xn);
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
 * order; the xor tree sums lanes.  Row arithmetic is independent of T. */
template <bool W8>
__global__ void __launch_bounds__(256)
qwen_gr_down_kernel(const void *__restrict__ wv, const uint8_t *__restrict__ wsf,
                    const void *__restrict__ xv, const uint8_t *__restrict__ xsf, int x_kbp,
                    int out, int in, int T, int n_split, float *__restrict__ part) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= out) return;
    const int split = blockIdx.y;
    const int t0 = blockIdx.z * kDownTB, nt = min(kDownTB, T - t0);
    float acc[kDownTB];
#pragma unroll
    for (int tt = 0; tt < kDownTB; ++tt) acc[tt] = 0.0f;
    if constexpr (W8) {
        const uint8_t *wq = (const uint8_t *)wv, *xq = (const uint8_t *)xv;
        const int nblk = in / 32, per = nblk / n_split, b0 = split * per;
        const int w_kbp = pulsar_mx_kbp(in);
        for (int b = b0 + lane; b < b0 + per; b += 32) {
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
    } else {
        const uint4 *wr = reinterpret_cast<const uint4 *>((const __nv_bfloat16 *)wv + (size_t)row * in);
        const int nch = in / 8, per = nch / n_split, c0 = split * per;
        for (int ci = c0 + lane; ci < c0 + per; ci += 32) {
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
        if (lane == 0 && tt < nt) part[((size_t)(t0 + tt) * n_split + split) * out + row] = v;
    }
}

/* 3. a = silu(d / 4) -> the format W_up reads; inj = 2 sigmoid(z / 4) */
template <bool W8>
__global__ void __launch_bounds__(kR)
qwen_gr_mid_kernel(const float *__restrict__ part, __nv_fp8_e4m3 *__restrict__ aq, unsigned char *__restrict__ asf,
                   int a_kbp, __nv_bfloat16 *__restrict__ ab, const float *__restrict__ injp, float *__restrict__ inj) {
    const int t = blockIdx.x, r = threadIdx.x;
    float d = 0.0f;
#pragma unroll
    for (int sp = 0; sp < kDownSplit; ++sp) d += part[((size_t)t * kDownSplit + sp) * kR + r];
    const float z = d * (1.0f / kS);
    const float a = z / (1.0f + expf(-z));
    if constexpr (W8) pulsar_mx_emit_block(a, (uint32_t)r, (uint32_t)t, (uint32_t)kR, a_kbp, aq, asf);
    else ab[(size_t)t * kR + r] = __float2bfloat16(a);
    if (injp && r < kS) {
        float zi = 0.0f;
#pragma unroll
        for (int s = 0; s < kS; ++s) zi += injp[((size_t)t * kS + s) * kS + r];
        inj[t * kS + r] = 2.0f / (1.0f + expf(-zi * (1.0f / kS)));
    }
}

/* 4. up + gate + mean: one warp per stream, lane = channel, kUpTB tokens per
 * CTA.  The k loop is outermost: each 32-block of the lane's W_up row is
 * loaded once and applied to every token of the CTA (a token's z still sums
 * its blocks in order 0..9, so the row arithmetic does not depend on T); the
 * gated products go to shared memory and one warp per token takes the stream
 * mean and emits the row. */
template <bool W8>
__global__ void __launch_bounds__(32 * kS)
qwen_gr_up_kernel(const void *__restrict__ wv, const uint8_t *__restrict__ wsf,
                  const void *__restrict__ av, const uint8_t *__restrict__ asf, int a_kbp,
                  const __nv_bfloat16 *__restrict__ streams, const __nv_bfloat16 *__restrict__ norm_w,
                  const float *__restrict__ rstd, int T,
                  __nv_bfloat16 *__restrict__ x_out, __nv_fp8_e4m3 *__restrict__ xq, unsigned char *__restrict__ xsf,
                  int x_kbp) {
    constexpr int NB = kR / 32;
    __shared__ float prod[kUpTB][kS][32];
    const int s = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c = blockIdx.x * 32 + lane;
    const int row = s * kH + c;
    const int t0 = blockIdx.y * kUpTB, nt = min(kUpTB, T - t0);
    float z[kUpTB];
#pragma unroll
    for (int tt = 0; tt < kUpTB; ++tt) z[tt] = 0.0f;
    if constexpr (W8) {
        const int w_kbp = pulsar_mx_kbp(kR);
        const uint4 *wp = reinterpret_cast<const uint4 *>((const uint8_t *)wv + (size_t)row * kR);
        for (int b = 0; b < NB; ++b) {
            const uint4 w0 = wp[2 * b], w1 = wp[2 * b + 1];
            const unsigned sw = wsf[pulsar_mx_sfoff(row, b, w_kbp)];
#pragma unroll
            for (int tt = 0; tt < kUpTB; ++tt) {
                if (tt < nt) {
                    const int t = t0 + tt;
                    const uint4 *ap = reinterpret_cast<const uint4 *>((const uint8_t *)av + (size_t)t * kR) + 2 * b;
                    z[tt] = fmaf(e4m3_dot32(w0, w1, ap[0], ap[1]), mx_scale2(sw, asf[pulsar_mx_sfoff(t, b, a_kbp)]), z[tt]);
                }
            }
        }
    } else {
        const uint4 *wp = reinterpret_cast<const uint4 *>((const __nv_bfloat16 *)wv + (size_t)row * kR);
        for (int ci = 0; ci < kR / 8; ++ci) {
            const uint4 w = wp[ci];
#pragma unroll
            for (int tt = 0; tt < kUpTB; ++tt)
                if (tt < nt)
                    z[tt] = bf16_dot8(w, reinterpret_cast<const uint4 *>((const __nv_bfloat16 *)av + (size_t)(t0 + tt) * kR)[ci],
                                      z[tt]);
        }
    }
    const float w1n = 1.0f + bf2f(norm_w[row]);
#pragma unroll
    for (int tt = 0; tt < kUpTB; ++tt) {
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
        x_out[(size_t)t * kH + c] = vb;
        if (xq) pulsar_mx_emit_block(bf2f(vb), (uint32_t)c, (uint32_t)t, (uint32_t)kH, x_kbp, xq, xsf);   /* warp-uniform */
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

struct gr_ws {
    uint8_t *xn, *xn_sf, *a, *a_sf;   /* xn / a: E4M3 (W8) or bf16 -- sized for bf16 */
    float *rstd, *injp, *part;
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
    m.xn_sf = (uint8_t *)take(pulsar_mx_sf_slab_bytes(T, pulsar_mx_kbp(kHC)));
    m.a     = (uint8_t *)take((size_t)T * kR * 2);
    m.a_sf  = (uint8_t *)take(pulsar_mx_sf_slab_bytes(T, pulsar_mx_kbp(kR)));
    m.rstd  = (float *)take((size_t)T * kS * 4);
    m.injp  = (float *)take((size_t)T * kS * kS * 4);
    m.part  = (float *)take((size_t)T * kDownSplit * kR * 4);
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

template <bool W8>
static void gr_read_kernels(const pulsar_qwen_gr_dev *w, const uint16_t *streams, int T, uint16_t *x_bf16,
                            const pulsar_qwen_slot *x, float *inj, const gr_ws &m, cudaStream_t stream) {
    const int xn_kbp = pulsar_mx_kbp(kHC), a_kbp = pulsar_mx_kbp(kR);
    qwen_gr_norm_kernel<W8><<<dim3(kS, T), kNormThreads, 0, stream>>>(
        (const __nv_bfloat16 *)streams, (const __nv_bfloat16 *)w->norm_w, (const __nv_bfloat16 *)w->inject,
        (__nv_fp8_e4m3 *)m.xn, m.xn_sf, xn_kbp, (__nv_bfloat16 *)m.xn, m.rstd, m.injp);
    qwen_gr_down_kernel<W8><<<dim3((kR + 7) / 8, kDownSplit, (T + kDownTB - 1) / kDownTB), 256, 0, stream>>>(
        w->down.w, w->down.sf, m.xn, m.xn_sf, xn_kbp, kR, kHC, T, kDownSplit, m.part);
    qwen_gr_mid_kernel<W8><<<T, kR, 0, stream>>>(m.part, (__nv_fp8_e4m3 *)m.a, m.a_sf, a_kbp, (__nv_bfloat16 *)m.a,
                                                 w->inject ? m.injp : nullptr, inj);
    qwen_gr_up_kernel<W8><<<dim3(kH / 32, (T + kUpTB - 1) / kUpTB), 32 * kS, 0, stream>>>(
        w->up.w, w->up.sf, m.a, m.a_sf, a_kbp, (const __nv_bfloat16 *)streams, (const __nv_bfloat16 *)w->norm_w,
        m.rstd, T, (__nv_bfloat16 *)x_bf16, x ? (__nv_fp8_e4m3 *)x->q : nullptr, x ? x->sf : nullptr,
        x ? x->kbp : 0);
}

extern "C" int pulsar_qwen_gr_read_launch(const pulsar_qwen_gr_dev *w, const uint16_t *streams, int T,
                                          uint16_t *x_bf16, const pulsar_qwen_slot *x, float *inj,
                                          void *ws, size_t ws_bytes, cudaStream_t stream) {
    if (!w || !w->norm_w || !w->down.w || !w->up.w || !streams || T <= 0 ||
        !x_bf16 || (x && (!x->q || !x->sf || x->kbp != pulsar_mx_kbp(kH))) || (w->inject && !inj)) {
        fprintf(stderr, "pulsar: qwen GR read: a null input, or a block-input slot that is not %d wide -- refusing\n", kH);
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
                        "sigmoid gate, stream mean -> bf16 row%s; streams bf16\n",
                w8 ? "MXFP8 (E4M3 act)" : "BF16 (bf16 act)", kDownSplit, w8 ? "MXFP8" : "BF16",
                x ? " + E4M3" : "");
    }
    if (w8) {
        cudaMemsetAsync(m.xn_sf, 0, pulsar_mx_sf_slab_bytes(T, pulsar_mx_kbp(kHC)), stream);
        cudaMemsetAsync(m.a_sf, 0, pulsar_mx_sf_slab_bytes(T, pulsar_mx_kbp(kR)), stream);
    }
    if (x) cudaMemsetAsync(x->sf, 0, pulsar_mx_sf_slab_bytes(T, x->kbp), stream);   /* this launcher is the slot's producer */
    if (w8) gr_read_kernels<true>(w, streams, T, x_bf16, x, inj, m, stream);
    else    gr_read_kernels<false>(w, streams, T, x_bf16, x, inj, m, stream);
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

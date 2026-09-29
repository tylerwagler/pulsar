/* Qwen3.8-Flash-Next MTP (L251): the input combine that starts the MTP layer's streams.
 * The spec is l251/docs/MTP-SPEC-2026-09-29.md (vLLM v0.30.0 qwen4_exp/nvidia/mtp.py:293-313, which
 * TensorFold matches); contracts in pulsar_cuda_qwen.h.
 *
 * For each row (target position p, token x_{p+1}):
 *   hn   = GemmaRMSNorm_{4 x 2560}(h[p]; w_ph)   ONE rms over the whole pre-mixer stack, (1 + w) per
 *                                                 element, f32 math, rounded to bf16
 *   ein  = GemmaRMSNorm_2560(embed(x_{p+1}); w_pe)   rounded to bf16
 *   e    = fc_embedding(ein)                       rounded to bf16 (a Linear in the model dtype)
 *   s_i  = bf16( bf16(fc_hidden(hn_i)) + e )       i = 0..3: the shared fc per stream, e added to every
 *                                                 stream with unit weight (the layer's first combine
 *                                                 with no injection), one rounding
 * The two fc Linears are the recipe's mxfp8_lt through the W8A16 arm (pulsar_qwen_mxfp8_linear_launch:
 * the GEMV at decode widths, the tensor-core GEMM at prompt widths).  Rows go through in slabs of
 * kMtpSlab so the workspace is a fixed few tens of MB whatever the chunk. */
#include "pulsar_cuda_qwen.h"

#include <cuda_bf16.h>
#include <stdio.h>

namespace {

constexpr int kH = PULSAR_QWEN_HIDDEN;
constexpr int kS = PULSAR_QWEN_HC;
constexpr int kHC = PULSAR_QWEN_HC_HIDDEN;
constexpr float kEps = 1e-6f;                 ///< rms_norm_eps
constexpr int kThreads = 256;
constexpr int kMtpSlab = 256;                 ///< rows per combine pass

__device__ __forceinline__ float block_sum(float v, float *red) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float s = 0.0f;
    for (int w = 0; w < nw; ++w) s += red[w];
    return s;
}

/* hn [4 rows per token][2560] bf16: one CTA per token row, the rms over all 10240 values */
__global__ void __launch_bounds__(kThreads)
mtp_norm_hidden_kernel(const __nv_bfloat16 *__restrict__ h, const __nv_bfloat16 *__restrict__ w,
                       __nv_bfloat16 *__restrict__ hn) {
    __shared__ float red[kThreads / 32];
    const size_t row = blockIdx.x;
    const __nv_bfloat16 *x = h + row * kHC;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < kHC; i += kThreads) {
        const float v = __bfloat162float(x[i]);
        ss = fmaf(v, v, ss);
    }
    const float r = rsqrtf(block_sum(ss, red) / (float)kHC + kEps);
    for (int i = threadIdx.x; i < kHC; i += kThreads)
        hn[row * kHC + i] = __float2bfloat16(__bfloat162float(x[i]) * r * (1.0f + __bfloat162float(w[i])));
}

/* ein [T][2560] bf16: the token's embedding row, (1 + w) RMSNorm */
__global__ void __launch_bounds__(kThreads)
mtp_norm_embed_kernel(const __nv_bfloat16 *__restrict__ table, const int32_t *__restrict__ tokens, int n_vocab,
                      const __nv_bfloat16 *__restrict__ w, __nv_bfloat16 *__restrict__ ein) {
    __shared__ float red[kThreads / 32];
    const size_t row = blockIdx.x;
    const int tok = tokens[row];
    const __nv_bfloat16 *x = table + (size_t)(tok >= 0 && tok < n_vocab ? tok : 0) * kH;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < kH; i += kThreads) {
        const float v = __bfloat162float(x[i]);
        ss = fmaf(v, v, ss);
    }
    const float r = rsqrtf(block_sum(ss, red) / (float)kH + kEps);
    for (int i = threadIdx.x; i < kH; i += kThreads)
        ein[row * kH + i] = __float2bfloat16(__bfloat162float(x[i]) * r * (1.0f + __bfloat162float(w[i])));
}

/* out streams [T][4][2560] = bf16(bf16(fh[t*4+s]) + bf16(fe[t])) */
__global__ void mtp_add_kernel(const float *__restrict__ fh, const float *__restrict__ fe, int T,
                               __nv_bfloat16 *__restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * kHC) return;
    const size_t t = i / kHC;
    const int c = (int)(i % kH);
    const float a = __bfloat162float(__float2bfloat16(fh[i]));
    const float b = __bfloat162float(__float2bfloat16(fe[t * kH + c]));
    out[i] = __float2bfloat16(a + b);
}

struct mtp_ws { __nv_bfloat16 *hn, *ein; float *fh, *fe; };

size_t mtp_ws_layout(uint8_t *base, mtp_ws *o) {
    size_t used = 0;
    auto take = [&](size_t bytes) -> void * {
        const size_t off = (used + 255) & ~(size_t)255;
        used = off + bytes;
        return base ? base + off : nullptr;
    };
    mtp_ws m{};
    m.hn  = (__nv_bfloat16 *)take((size_t)kMtpSlab * kHC * 2);
    m.ein = (__nv_bfloat16 *)take((size_t)kMtpSlab * kH * 2);
    m.fh  = (float *)take((size_t)kMtpSlab * kHC * 4);
    m.fe  = (float *)take((size_t)kMtpSlab * kH * 4);
    if (o) *o = m;
    return used;
}

bool launch_ok(const char *what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: qwen MTP %s: launch failed: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

} // namespace

extern "C" size_t pulsar_qwen_mtp_combine_workspace_bytes(void) { return mtp_ws_layout(nullptr, nullptr); }

extern "C" int pulsar_qwen_mtp_combine_launch(const pulsar_qwen_mtp_dev *w, const uint16_t *h_streams,
                                              const int32_t *tokens, int T, uint16_t *out_streams,
                                              void *ws, size_t ws_bytes, cudaStream_t stream) {
    if (!w || !w->embd || !w->norm_embd || !w->norm_hidden || !w->fc_embd.w || !w->fc_embd.sf || !w->fc_hidden.w ||
        !w->fc_hidden.sf || !h_streams || !tokens || !out_streams || T <= 0 || w->n_vocab <= 0) {
        fprintf(stderr, "pulsar: qwen MTP combine: a null input or no rows -- refusing\n");
        return -1;
    }
    if (w->fc_embd.in != kH || w->fc_embd.out != kH || w->fc_hidden.in != kH || w->fc_hidden.out != kH) {
        fprintf(stderr, "pulsar: qwen MTP combine: fc %dx%d / %dx%d, built for %dx%d -- refusing\n", w->fc_embd.out,
                w->fc_embd.in, w->fc_hidden.out, w->fc_hidden.in, kH, kH);
        return -1;
    }
    mtp_ws m;
    if (!ws || ws_bytes < mtp_ws_layout(nullptr, nullptr)) {
        fprintf(stderr, "pulsar: qwen MTP combine: workspace %zu B < %zu B -- refusing\n", ws_bytes,
                mtp_ws_layout(nullptr, nullptr));
        return -1;
    }
    mtp_ws_layout((uint8_t *)ws, &m);
    static int announced = 0;
    if (!announced) {
        announced = 1;
        fprintf(stderr, "pulsar: L251 qwen MTP combine = joint 10240 (1+w) RMSNorm of the pre-mixer stack -> "
                        "fc_hidden per stream (mxfp8_lt) + fc_embedding((1+w) RMSNorm(embed(next))) -> bf16 streams\n");
    }
    for (int t0 = 0; t0 < T; t0 += kMtpSlab) {
        const int n = T - t0 < kMtpSlab ? T - t0 : kMtpSlab;
        mtp_norm_hidden_kernel<<<n, kThreads, 0, stream>>>((const __nv_bfloat16 *)h_streams + (size_t)t0 * kHC,
                                                           (const __nv_bfloat16 *)w->norm_hidden, m.hn);
        mtp_norm_embed_kernel<<<n, kThreads, 0, stream>>>((const __nv_bfloat16 *)w->embd, tokens + t0, w->n_vocab,
                                                          (const __nv_bfloat16 *)w->norm_embd, m.ein);
        if (!launch_ok("norms")) return -3;
        /* hn is [n * 4][2560]: fc_hidden applies to each stream as its own row */
        if (pulsar_qwen_mxfp8_linear_launch(&w->fc_hidden, (const uint16_t *)m.hn, n * kS, m.fh, nullptr, 0, stream) ||
            pulsar_qwen_mxfp8_linear_launch(&w->fc_embd, (const uint16_t *)m.ein, n, m.fe, nullptr, 0, stream))
            return -1;
        const size_t ne = (size_t)n * kHC;
        mtp_add_kernel<<<(unsigned)((ne + 255) / 256), 256, 0, stream>>>(
            m.fh, m.fe, n, (__nv_bfloat16 *)out_streams + (size_t)t0 * kHC);
        if (!launch_ok("add")) return -3;
    }
    return 0;
}

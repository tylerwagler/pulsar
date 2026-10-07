/* Qwen3.8-Flash-Next PLE injection at layer index 1 (L251 S4).  Source:
 * transformers' Qwen4ExpTextPLELayer; contracts in pulsar_cuda_qwen.h.  The
 * table rows arrive gathered (src/engine/qwen_ngram.cpp hashes the ids on the
 * host and reads the 16 rows per token from disk); this file is everything
 * after the gather:
 *
 *   emit   the 16 x 160 bf16 rows -> the E4M3 slot key_proj / value_proj read
 *          (160 = 5 x 32: no MX group straddles two heads' rows)
 *   k, v   the two dense Linears (EXL3 dense arm)
 *   gate   per (stream, token): k_s = norm_key(k)_s, q_s = norm_query(x)_s,
 *          g = <k_s, q_s> / sqrt(2560) -> sign(g) sqrt(max(|g|, 1e-6)),
 *          gv_s = sigmoid(g) v, gvn_s = norm_conv(gv)_s (the norm of gv is
 *          sigmoid(g)^2 sum v^2, taken on gv itself); gvn to the workspace
 *   conv   per (channel, token): y = silu(sum_m w[m] gvn[t - 9 + 3m]) over the
 *          batch rows of the same sequence and the 9-token state before them;
 *          stream += gv + y (f32 math, one bf16 rounding)
 *   state  the last 9 gvn of each sequence become its state
 */
#include "pulsar_cuda_qwen.h"
#include "pulsar_cuda_mx.cuh"
#include "mmq/ds4_exl3_dense.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdio.h>

namespace {

constexpr int kH = PULSAR_QWEN_HIDDEN;
constexpr int kS = PULSAR_QWEN_HC;
constexpr int kHC = PULSAR_QWEN_HC_HIDDEN;
constexpr int kTaps = PULSAR_QWEN_PLE_TAPS;
constexpr int kDil = PULSAR_QWEN_PLE_DIL;
constexpr int kState = PULSAR_QWEN_PLE_STATE;
constexpr float kEps = 1e-6f;
constexpr int kThreads = 256;
constexpr int kPer = kH / kThreads;
static_assert(PULSAR_QWEN_PLE_HEADS * PULSAR_QWEN_PLE_ROW == kH, "16 rows of 160 are the embedding");
static_assert(PULSAR_QWEN_PLE_ROW % 32 == 0, "a table row is whole MX groups");
static_assert(kHC % kThreads == 0, "the conv grid covers the channels in whole blocks");

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }

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


__global__ void __launch_bounds__(kThreads)
qwen_ple_gate_kernel(const float *__restrict__ key, const float *__restrict__ value,
                     const __nv_bfloat16 *__restrict__ streams, const __nv_bfloat16 *__restrict__ nk,
                     const __nv_bfloat16 *__restrict__ nq, const __nv_bfloat16 *__restrict__ nc,
                     float *__restrict__ gvn, float *__restrict__ sig) {
    __shared__ float red[kThreads / 32];
    const int s = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    float k[kPer], q[kPer];
    float ssk = 0.0f, ssq = 0.0f;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int c = tid + kThreads * j;
        k[j] = key[(size_t)t * kHC + s * kH + c];
        q[j] = bf2f(streams[(size_t)t * kHC + s * kH + c]);
        ssk = fmaf(k[j], k[j], ssk);
        ssq = fmaf(q[j], q[j], ssq);
    }
    const float rk = rsqrtf(block_sum(ssk, red) * (1.0f / kH) + kEps);
    const float rq = rsqrtf(block_sum(ssq, red) * (1.0f / kH) + kEps);
    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int ch = s * kH + tid + kThreads * j;
        const float kn = k[j] * rk * (1.0f + bf2f(nk[ch]));
        const float qn = q[j] * rq * (1.0f + bf2f(nq[ch]));
        dot = fmaf(kn, qn, dot);
    }
    const float g = block_sum(dot, red) * rsqrtf((float)kH);
    const float gs = g > 0.0f ? sqrtf(fmaxf(g, 1e-6f)) : (g < 0.0f ? -sqrtf(fmaxf(-g, 1e-6f)) : 0.0f);
    const float sg = 1.0f / (1.0f + expf(-gs));
    float gv[kPer];
    float ssv = 0.0f;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        gv[j] = sg * value[(size_t)t * kH + tid + kThreads * j];
        ssv = fmaf(gv[j], gv[j], ssv);
    }
    const float rc = rsqrtf(block_sum(ssv, red) * (1.0f / kH) + kEps);
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int ch = s * kH + tid + kThreads * j;
        gvn[(size_t)t * kHC + ch] = gv[j] * rc * (1.0f + bf2f(nc[ch]));
    }
    if (tid == 0) sig[t * kS + s] = sg;
}

__global__ void __launch_bounds__(kThreads)
qwen_ple_conv_kernel(const float *__restrict__ gvn, const float *__restrict__ value, const float *__restrict__ sig,
                     const __nv_bfloat16 *__restrict__ conv_w, const float *__restrict__ state,
                     const int32_t *__restrict__ row_seq, const int32_t *__restrict__ row_j,
                     const int32_t *__restrict__ seq_bank, __nv_bfloat16 *__restrict__ streams) {
    const int ch = blockIdx.x * kThreads + threadIdx.x, t = blockIdx.y;
    const int seq = seq_bank[row_seq[t]], j = row_j[t];
    const int s = ch / kH, c = ch % kH;
    float y = 0.0f;
#pragma unroll
    for (int m = 0; m < kTaps; ++m) {
        const int back = (kTaps - 1 - m) * kDil;          /* 9, 6, 3, 0 */
        const float tap = j >= back ? gvn[(size_t)(t - back) * kHC + ch]
                                    : state[((size_t)seq * kState + (kState + j - back)) * kHC + ch];
        y = fmaf(bf2f(conv_w[ch * kTaps + m]), tap, y);
    }
    const float out = sig[t * kS + s] * value[(size_t)t * kH + c] + y / (1.0f + expf(-y));
    const size_t i = (size_t)t * kHC + ch;
    streams[i] = __float2bfloat16(bf2f(streams[i]) + out);
}

__global__ void __launch_bounds__(kThreads)
qwen_ple_state_kernel(const float *__restrict__ gvn, const int32_t *__restrict__ seq_first,
                      const int32_t *__restrict__ seq_rows, const int32_t *__restrict__ seq_bank,
                      float *__restrict__ state) {
    const int ch = blockIdx.x * kThreads + threadIdx.x;
    const int f = seq_first[blockIdx.y], n = seq_rows[blockIdx.y], q = seq_bank[blockIdx.y];
    float nv[kState];
#pragma unroll
    for (int i = 0; i < kState; ++i) {
        const int p = n + i - kState;                     /* position relative to the batch's first row */
        nv[i] = p >= 0 ? gvn[(size_t)(f + p) * kHC + ch] : state[((size_t)q * kState + (n + i)) * kHC + ch];
    }
#pragma unroll
    for (int i = 0; i < kState; ++i) state[((size_t)q * kState + i) * kHC + ch] = nv[i];
}

/* the verify capture: the conv state after each row of a sequence but its last -- the state kernel's
 * rule with the sequence cut after that row -- at the row's batch index (L272 P1 S4: any number of
 * sequences, one per bank).  Launched before it, so the old state is intact.  grid (channels, T). */
__global__ void __launch_bounds__(kThreads)
qwen_ple_state_rows_kernel(const float *__restrict__ gvn, const int32_t *__restrict__ row_seq,
                           const int32_t *__restrict__ row_j, const int32_t *__restrict__ seq_first,
                           const int32_t *__restrict__ seq_rows, const int32_t *__restrict__ seq_bank,
                           const float *__restrict__ state, float *__restrict__ state_rows) {
    const int ch = blockIdx.x * kThreads + threadIdx.x, t = blockIdx.y;
    const int sq = row_seq[t], j = row_j[t];
    if (j >= seq_rows[sq] - 1) return;                    /* the sequence's last row: the pool's */
    const int f = seq_first[sq], q = seq_bank[sq], n = j + 1;
#pragma unroll
    for (int i = 0; i < kState; ++i) {
        const int p = n + i - kState;
        state_rows[((size_t)t * kState + i) * kHC + ch] =
            p >= 0 ? gvn[(size_t)(f + p) * kHC + ch] : state[((size_t)q * kState + (n + i)) * kHC + ch];
    }
}

struct ple_ws {
    float *key, *value, *gvn, *sig;
    void *lin;
    size_t lin_bytes;
};

static size_t ple_ws_layout(int T, void *base, size_t cap, ple_ws *o) {
    size_t used = 0;
    bool failed = false;
    auto take = [&](size_t bytes) -> void * {
        const size_t off = (used + 255) & ~(size_t)255;
        if (off + bytes > (base ? cap : (size_t)-1)) { failed = true; return nullptr; }
        used = off + bytes;
        return base ? (uint8_t *)base + off : nullptr;
    };
    ple_ws m{};
    m.key   = (float *)take((size_t)T * kHC * 4);
    m.value = (float *)take((size_t)T * kH * 4);
    m.gvn   = (float *)take((size_t)T * kHC * 4);
    m.sig   = (float *)take((size_t)T * kS * 4);
    const size_t lk = ds4_exl3_dense_workspace_bytes(T, kH, kHC);   /* a function of (rows, in, out) alone */
    const size_t lv = ds4_exl3_dense_workspace_bytes(T, kH, kH);
    m.lin_bytes = lk > lv ? lk : lv;
    m.lin = take(m.lin_bytes);
    if (o) *o = m;
    return failed ? 0 : used;
}

static bool launch_ok(const char *what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: qwen PLE %s: launch failed: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

} // namespace

extern "C" size_t pulsar_qwen_ple_workspace_bytes(int T) {
    return T > 0 ? ple_ws_layout(T, nullptr, 0, nullptr) : 0;
}

extern "C" int pulsar_qwen_ple_launch(const pulsar_qwen_ple_dev *w, const uint16_t *emb, uint16_t *streams, int T,
                                      const pulsar_qwen_rows *rows, float *conv_state,
                                      void *ws, size_t ws_bytes, cudaStream_t stream) {
    if (!w || !emb || !streams || T <= 0 || !rows || !rows->row_seq || !rows->row_j || !rows->seq_first ||
        !rows->seq_rows || !rows->seq_bank || rows->n_seq <= 0 || !conv_state || !w->norm_key || !w->norm_query || !w->norm_conv ||
        !w->conv_w) {
        fprintf(stderr, "pulsar: qwen PLE: a null input -- refusing\n");
        return -1;
    }
    if (w->key_proj.in != kH || w->key_proj.out != kHC || w->value_proj.in != kH || w->value_proj.out != kH) {
        fprintf(stderr, "pulsar: qwen PLE: key %d->%d / value %d->%d, built for %d->%d / %d->%d -- refusing\n",
                w->key_proj.in, w->key_proj.out, w->value_proj.in, w->value_proj.out, kH, kHC, kH, kH);
        return -1;
    }
    ple_ws m;
    if (!ws || ple_ws_layout(T, ws, ws_bytes, &m) == 0) {
        fprintf(stderr, "pulsar: qwen PLE: workspace %zu B < %zu B for %d rows -- refusing\n",
                ws_bytes, pulsar_qwen_ple_workspace_bytes(T), T);
        return -1;
    }
    static int announced = 0;
    if (!announced) {
        announced = 1;
        fprintf(stderr, "pulsar: L251 qwen PLE = gathered bf16 rows -> E4M3 -> EXL3 key K=%g / value K=%g -> "
                        "signed-sqrt stream gate, dilated conv (%d taps x %d, f32 state)\n",
                w->key_proj.k2 / 2.0, w->value_proj.k2 / 2.0, kTaps, kDil);
    }
    /* L251 / ac69748f: the gathered rows ARE bf16, so the key/value projections read them directly
     * and the E4M3 emit step (which existed only to build the A8 slot) is gone.  The bf16 row is the
     * one activation encoding, emitted by whatever produced `emb` (rule 3). */
    int rc = pulsar_rows_linear_launch(&w->key_proj, (const uint16_t *)emb, T, m.key, m.lin, m.lin_bytes, stream);
    if (!rc) rc = pulsar_rows_linear_launch(&w->value_proj, (const uint16_t *)emb, T, m.value, m.lin, m.lin_bytes, stream);
    if (rc) return rc;
    qwen_ple_gate_kernel<<<dim3(kS, T), kThreads, 0, stream>>>(
        m.key, m.value, (const __nv_bfloat16 *)streams, (const __nv_bfloat16 *)w->norm_key,
        (const __nv_bfloat16 *)w->norm_query, (const __nv_bfloat16 *)w->norm_conv, m.gvn, m.sig);
    qwen_ple_conv_kernel<<<dim3(kHC / kThreads, T), kThreads, 0, stream>>>(
        m.gvn, m.value, m.sig, (const __nv_bfloat16 *)w->conv_w, conv_state, rows->row_seq, rows->row_j,
        rows->seq_bank, (__nv_bfloat16 *)streams);
    if (rows->state_rows && T > 1)
        qwen_ple_state_rows_kernel<<<dim3(kHC / kThreads, T), kThreads, 0, stream>>>(
            m.gvn, rows->row_seq, rows->row_j, rows->seq_first, rows->seq_rows, rows->seq_bank, conv_state,
            rows->state_rows);
    qwen_ple_state_kernel<<<dim3(kHC / kThreads, rows->n_seq), kThreads, 0, stream>>>(
        m.gvn, rows->seq_first, rows->seq_rows, rows->seq_bank, conv_state);
    return launch_ok("inject") ? 0 : -3;
}

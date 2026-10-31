/** @file
 *  Qwen3.8-Flash-Next MTP: the input combine that starts the MTP layer's streams. */
/* Qwen3.8-Flash-Next MTP (L251): the input combine that starts the MTP layer's streams.
 * The spec is the L251 MTP spec (2026-09-29, private notes), from vLLM's reference (vLLM v0.30.0 qwen4_exp/nvidia/mtp.py:293-313, which
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

/// Hidden width of one residual stream (2560).
constexpr int kH = PULSAR_QWEN_HIDDEN;
/// Residual streams per token (4).
constexpr int kS = PULSAR_QWEN_HC;
/// Width of the whole pre-mixer stack, kS * kH (10240).
constexpr int kHC = PULSAR_QWEN_HC_HIDDEN;
constexpr float kEps = 1e-6f;                 ///< rms_norm_eps
/// Block size of the two RMSNorm kernels.
constexpr int kThreads = 256;
constexpr int kMtpSlab = 256;                 ///< rows per combine pass

/** Block-wide f32 sum: warp shuffle reduction, then every thread adds the per-warp partials in warp order (so all
 *  threads return the same value).  Syncs the block; blockDim.x a multiple of 32.
 *  @param v   this thread's partial
 *  @param red shared scratch, blockDim.x / 32 floats
 *  @return the block's sum */
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
/** Joint (1 + w) RMSNorm over a token's whole pre-mixer stack (kHC values), f32 math, bf16 out.
 *  Grid = rows (tokens of the slab), block = kThreads, static smem kThreads / 32 floats.
 *  @param h  bf16 [rows][kHC] pre-mixer streams
 *  @param w  bf16 [kHC] mtp.pre_fc_norm_hidden weight
 *  @param hn out, bf16 [rows][kHC] (read later as [rows * kS][kH]) */
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
/** (1 + w) RMSNorm of each next token's embedding row.  Grid = rows, block = kThreads, static smem kThreads / 32 floats.
 *  @param table   bf16 [n_vocab][kH] embed_tokens
 *  @param tokens  device int32 [rows] next-token ids; an id outside [0, n_vocab) reads row 0
 *  @param n_vocab rows in `table`
 *  @param w       bf16 [kH] mtp.pre_fc_norm_embedding weight
 *  @param ein     out, bf16 [rows][kH] */
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
/** The combine's add: each stream's fc_hidden output plus the token's fc_embedding output, each rounded to bf16
 *  first, the sum rounded once.  One thread per output element: grid = ceil(T * kHC / 256), block = 256.
 *  @param fh  f32 [T * kS][kH] fc_hidden(hn)
 *  @param fe  f32 [T][kH] fc_embedding(ein)
 *  @param T   rows (tokens) in this slab
 *  @param out bf16 [T][kS][kH] the MTP layer's starting streams */
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

/// The combine's per-slab device scratch (kMtpSlab rows), carved by mtp_ws_layout.
struct mtp_ws { __nv_bfloat16 *hn, *ein; float *fh, *fe; };

/** @var mtp_ws::hn
 *  bf16 [kMtpSlab][kHC] normed pre-mixer stack.
 *  @var mtp_ws::ein
 *  bf16 [kMtpSlab][kH] normed next-token embedding.
 *  @var mtp_ws::fh
 *  f32 [kMtpSlab * kS][kH] fc_hidden output.
 *  @var mtp_ws::fe
 *  f32 [kMtpSlab][kH] fc_embedding output. */

/** Lay the combine's workspace out from `base`, each piece 256-B aligned.
 *  @param base workspace start, or nullptr to only size it
 *  @param o    out: the pieces' pointers (nullptr when base is nullptr), or nullptr
 *  @return bytes the layout uses */
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

/** Check the last launch; prints the failure with `what` and returns false on error.
 *  @param what step name for the message
 *  @return true when no launch error is pending */
bool launch_ok(const char *what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: qwen MTP %s: launch failed: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

/* ---- L289: the draft head's greedy pick and confidence on the device ---------------------------------
 * Three launches over kTopBlocks contiguous chunks of each row (blockIdx.y = the row): each block's first
 * maximum; each block's double sum of exp((double)(row[i] - m)) at the row's maximum m; one thread a row folds
 * both in block order.  The pick is order-free (max, ties to the lower index); the sum's order is fixed, so a
 * row's record is the same bytes run to run and whatever rows ride with it (L288: one launch for every bank's
 * draft row of a step). */
/// Blocks (contiguous chunks) each draft-head row is split over (blockIdx.x).
constexpr int kTopBlocks = 64;
/// Block size of the max and sum kernels.
constexpr int kTopThreads = 256;

/* one row's scratch; the workspace is PULSAR_QWEN_DRAFT_TOP_ROWS_MAX records, then as many of these */
/// One row's per-block partials for the draft-top reduction.
struct top_ws {
    /// Each block's first maximum value.
    float best[kTopBlocks];
    /// Each block's index of that maximum.
    int32_t idx[kTopBlocks];
    /// Each block's NaN flag (nonzero: its chunk held a NaN).
    uint32_t nan[kTopBlocks];
    /// Each block's double sum of exp(row[i] - m) over its chunk.
    double z[kTopBlocks];
};

/** Max order: a larger value wins, ties to the lower index.
 *  @param v candidate value
 *  @param i candidate index
 *  @param bv current best value
 *  @param bi current best index
 *  @return true when (v, i) replaces (bv, bi) */
__device__ __forceinline__ bool top_takes(float v, int32_t i, float bv, int32_t bi) {
    return v > bv || (v == bv && i < bi);
}

/** Chunk length each of the kTopBlocks blocks scans.
 *  @param n row length
 *  @return ceil(n / kTopBlocks) */
__device__ __forceinline__ int top_chunk(int n) { return (n + kTopBlocks - 1) / kTopBlocks; }

/** This block's row scratch: after the PULSAR_QWEN_DRAFT_TOP_ROWS_MAX records, entry blockIdx.y.
 *  @param ws the draft-top workspace
 *  @return the row's top_ws */
__device__ __forceinline__ top_ws *top_row_ws(void *ws) {
    return (top_ws *)((pulsar_qwen_draft_top *)ws + PULSAR_QWEN_DRAFT_TOP_ROWS_MAX) + blockIdx.y;
}

/** Pass 1: each block's first maximum and NaN flag over its chunk.  Grid = (kTopBlocks, n_rows), block = kTopThreads.
 *  @param rows f32 [n_rows][n] draft-head logits
 *  @param n    row length
 *  @param wsv  the draft-top workspace (writes top_ws::best / idx / nan) */
__global__ void __launch_bounds__(kTopThreads) draft_top_max_kernel(const float *rows, int n, void *wsv) {
    const float *row = rows + (size_t)blockIdx.y * n;
    top_ws *ws = top_row_ws(wsv);
    __shared__ float sv[kTopThreads / 32];
    __shared__ int32_t si[kTopThreads / 32];
    __shared__ uint32_t sn[kTopThreads / 32];
    const int c = top_chunk(n), lo = blockIdx.x * c, hi = min(n, lo + c);
    float bv = -INFINITY;
    int32_t bi = INT32_MAX;
    uint32_t nan = 0;
    for (int i = lo + (int)threadIdx.x; i < hi; i += kTopThreads) {
        const float v = row[i];
        nan |= isnan(v) ? 1u : 0u;
        if (top_takes(v, i, bv, bi)) { bv = v; bi = i; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
        nan |= __shfl_xor_sync(0xffffffffu, nan, o);
        if (top_takes(ov, oi, bv, bi)) { bv = ov; bi = oi; }
    }
    const int warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) { sv[warp] = bv; si[warp] = bi; sn[warp] = nan; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < kTopThreads / 32; ++w) {
            if (top_takes(sv[w], si[w], bv, bi)) { bv = sv[w]; bi = si[w]; }
            nan |= sn[w];
        }
        ws->best[blockIdx.x] = bv;
        ws->idx[blockIdx.x] = bi;
        ws->nan[blockIdx.x] = nan;
    }
}

/* The row's maximum from the blocks' -- every block folds the same kTopBlocks entries in the same order. */
/** @param ws the row's partials
 *  @param bv out: the row's maximum value
 *  @param bi out: its (first) index */
__device__ __forceinline__ void top_fold(const top_ws *ws, float *bv, int32_t *bi) {
    *bv = ws->best[0];
    *bi = ws->idx[0];
    for (int b = 1; b < kTopBlocks; ++b)
        if (top_takes(ws->best[b], ws->idx[b], *bv, *bi)) { *bv = ws->best[b]; *bi = ws->idx[b]; }
}

/** Pass 2: each block's double sum of exp((double)(row[i] - m)) at the row's maximum m, a fixed tree order.
 *  Grid = (kTopBlocks, n_rows), block = kTopThreads, static smem kTopThreads doubles.
 *  @param rows f32 [n_rows][n] draft-head logits
 *  @param n    row length
 *  @param wsv  the draft-top workspace (reads pass 1, writes top_ws::z) */
__global__ void __launch_bounds__(kTopThreads) draft_top_sum_kernel(const float *rows, int n, void *wsv) {
    __shared__ double sz[kTopThreads];
    const float *row = rows + (size_t)blockIdx.y * n;
    const top_ws *ws = top_row_ws(wsv);
    float m;
    int32_t a;
    top_fold(ws, &m, &a);
    m = a < n ? row[a] : m;   /* the host rule's m is row[a] itself */
    const int c = top_chunk(n), lo = blockIdx.x * c, hi = min(n, lo + c);
    double z = 0.0;
    for (int i = lo + (int)threadIdx.x; i < hi; i += kTopThreads) z += exp((double)(row[i] - m));
    sz[threadIdx.x] = z;
    __syncthreads();
    for (int s = kTopThreads / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sz[threadIdx.x] += sz[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) top_row_ws(wsv)->z[blockIdx.x] = sz[0];
}

/** Pass 3: fold the partials in block order into the row's pulsar_qwen_draft_top record.  Grid = (1, n_rows), block = 1.
 *  @param wsv the draft-top workspace (record blockIdx.y written at its start) */
__global__ void draft_top_record_kernel(void *wsv) {
    const top_ws *ws = top_row_ws(wsv);
    pulsar_qwen_draft_top *rec = (pulsar_qwen_draft_top *)wsv + blockIdx.y;
    float m;
    int32_t a;
    top_fold(ws, &m, &a);
    double z = 0.0;
    uint32_t nan = 0;
    for (int b = 0; b < kTopBlocks; ++b) { z += ws->z[b]; nan |= ws->nan[b]; }
    rec->index = a;
    rec->conf = (float)(1.0 / z);
    rec->nan = nan;
    rec->pad = 0;
}

} // namespace

extern "C" size_t pulsar_qwen_draft_top_ws_bytes(void) {
    return PULSAR_QWEN_DRAFT_TOP_ROWS_MAX * (sizeof(pulsar_qwen_draft_top) + sizeof(top_ws));
}

extern "C" int pulsar_qwen_draft_top_launch(const float *rows, int n, int n_rows, void *ws, cudaStream_t stream) {
    if (!rows || !ws || n <= 0 || n_rows <= 0 || n_rows > PULSAR_QWEN_DRAFT_TOP_ROWS_MAX) {
        fprintf(stderr, "pulsar: qwen draft top: a null row / workspace, no logits or %d rows (1..%d) -- refusing\n",
                n_rows, PULSAR_QWEN_DRAFT_TOP_ROWS_MAX);
        return -1;
    }
    const dim3 grid(kTopBlocks, (unsigned)n_rows);
    draft_top_max_kernel<<<grid, kTopThreads, 0, stream>>>(rows, n, ws);
    draft_top_sum_kernel<<<grid, kTopThreads, 0, stream>>>(rows, n, ws);
    draft_top_record_kernel<<<dim3(1, (unsigned)n_rows), 1, 0, stream>>>(ws);
    return launch_ok("draft top") ? 0 : -3;
}

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

// SPDX-License-Identifier: MIT
/** @file
 *  LiLiCorr candidate-lattice head and greedy path walk: the glue kernels between its library calls. */
// L298: the LiLiCorr candidate-lattice head and its greedy path walk (pulsar_cuda_lilicorr.h has the model, the
// numerics and which library runs which op).  This TU is the glue between the library calls: every Linear is the rows
// front door's, the lattice attention's two products are strided-batched cuBLASLt GEMMs, and the kernels below do what
// no library does -- the gather and the candidate features, the epilogues, the biased softmax, the normalize-and-dot
// scores and the walk.
//
// Rounding: each kernel rounds to bf16 exactly where vLLM's torch ops hand a bf16 tensor to the next op (the reference,
// tools/lilicorr/reference.py, names them); the sums inside an op are f32.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>

#include "pulsar_cuda_bf16_rows.cuh"   /* pulsar_gpu_bf16_batched_gemm */
#include "pulsar_cuda_lilicorr.h"

namespace {

/// Drafted slots per bank (7).
constexpr int kS = PULSAR_LILICORR_SLOTS, kK = PULSAR_LILICORR_TOPK, kN = kS * kK;   // 7, 8, 56 lattice rows
/** @var kK
 *  Candidates per slot (8).
 *  @var kN
 *  Lattice rows per bank, slot-major: kS x kK = 56. */
/// The head's hidden width (1024).
constexpr int kH = PULSAR_LILICORR_HIDDEN, kHeads = PULSAR_LILICORR_HEADS, kD = kH / kHeads;
/** @var kHeads
 *  Lattice attention heads (8).
 *  @var kD
 *  Attention head width, kH / kHeads = 128. */
/// The drafter's / target's hidden width (4096): the token, pass and anchor inputs.
constexpr int kM = PULSAR_LILICORR_MODEL;
/// Threads per glue-kernel block (each holds kH / kThreads = 4 values of a row).
constexpr int kThreads = 256;
/// Epsilon of the RMSNorms and the feature LayerNorm (torch's default, 1e-5).
constexpr float kNormEps = 1e-5f;   // the head's RMSNorms and the feature LayerNorm: torch's default (none configured)

/// Widens a bf16 bit pattern to f32 (exact).
/// @param u bf16 bits.
/// @return the value as f32.
__device__ __forceinline__ float bf(uint16_t u) { return __bfloat162float(__ushort_as_bfloat16(u)); }
/// Rounds to bf16 (nearest-even) and back: the value a torch bf16 tensor would hold.
/// @param x f32 value.
/// @return x rounded to bf16, as f32.
__device__ __forceinline__ float rnd(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
/// Rounds to bf16 (nearest-even).
/// @param x f32 value.
/// @return the bf16 bit pattern.
__device__ __forceinline__ uint16_t bits(float x) { return __bfloat16_as_ushort(__float2bfloat16_rn(x)); }
/// SiLU, x * sigmoid(x), in f32.
/// @param x input.
/// @return x / (1 + exp(-x)).
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

/* A block's f32 sum (kThreads threads); every thread gets it. */
/// Block-wide f32 sum over the kThreads threads.
/// @param v this thread's addend (all kThreads threads must call; uses __syncthreads).
/// @return the block-wide f32 sum, warp shuffles then warp partials in order.
__device__ float block_sum(float v) {
    __shared__ float part[kThreads / 32];
    __shared__ float total;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < kThreads / 32; i++) t += part[i];
        total = t;
    }
    __syncthreads();
    const float r = total;
    __syncthreads();
    return r;
}

/* y = rmsnorm(x) * w over one row of kH held 4 a thread (vals[]): normalize in f32, round, times the bf16 weight,
 * round -- torch's rms_norm on a bf16 tensor (and vLLM's). */
/// y = bf16(bf16(x * rstd) * w): torch's rms_norm of one kH-wide bf16 row held 4 values a thread.
/// @param vals this thread's 4 values of the row (column threadIdx.x + j * kThreads).
/// @param w    the norm's weight, bf16 [kH].
/// @param y    out: the normed row, bf16 [kH].
__device__ void rmsnorm_row(const float (&vals)[kH / kThreads], const uint16_t *w, uint16_t *y) {
    float ss = 0.0f;
    for (int j = 0; j < kH / kThreads; j++) ss += vals[j] * vals[j];
    const float rstd = rsqrtf(block_sum(ss) / (float)kH + kNormEps);
    for (int j = 0; j < kH / kThreads; j++) {
        const int c = threadIdx.x + j * kThreads;
        y[c] = bits(rnd(vals[j] * rstd) * bf(w[c]));
    }
}

/* Row n of the lattice (bank b = n / 56, slot s, rank k): the candidate's embedding row into E, and its feature row
 * silu(feature_mlp.up(LN5(f))) into F1 (the down projection is a front-door GEMM). */
/// Gathers each lattice row's candidate embedding and computes its feature row silu(feature_mlp.up(LN5(f))).
/// Launch: grid N = 56 x banks (block = lattice row n), block kThreads.
/// @param cand  candidate ids [banks][7][8] (row n's is cand[n]).
/// @param lp    candidate log-probs, f32 [banks][7][8].
/// @param table the target's input embedding, bf16 [vocab][kM], rows 16-byte aligned.
/// @param ln_w  feature LayerNorm weight, bf16 [5].
/// @param ln_b  feature LayerNorm bias, bf16 [5].
/// @param up_w  feature_mlp up weight, bf16 [kH][5].
/// @param up_b  feature_mlp up bias, bf16 [kH].
/// @param E     out: gathered embedding rows, bf16 [N][kM].
/// @param F1    out: feature rows, bf16 [N][kH].
__global__ void __launch_bounds__(kThreads) k_gather(const int32_t *cand, const float *lp, const uint16_t *table,
                                                     const uint16_t *ln_w, const uint16_t *ln_b, const uint16_t *up_w,
                                                     const uint16_t *up_b, uint16_t *E, uint16_t *F1) {
    const int n = blockIdx.x, b = n / kN, s = (n % kN) / kK, k = n % kK;
    const int64_t id = cand[n];
    const uint4 *src = reinterpret_cast<const uint4 *>(table + id * kM);
    uint4 *dst = reinterpret_cast<uint4 *>(E + (int64_t)n * kM);
    for (int i = threadIdx.x; i < kM / 8; i += kThreads) dst[i] = src[i];
    /* the five features as LiLiCorrHead.score stacks them (f32), then .to(bf16) */
    const float *row = lp + ((int64_t)b * kS + s) * kK;
    float mx = row[0];
    for (int j = 1; j < kK; j++) mx = fmaxf(mx, row[j]);
    const float l = row[k];
    float f[5] = {rnd(l), rnd(expf(l)), rnd(l - mx), rnd((float)k / (float)(kK - 1)), k == 0 ? 1.0f : 0.0f};
    /* LayerNorm(5): f32 statistics (biased variance), affine in f32, one rounding */
    float mean = 0.0f;
    for (int j = 0; j < 5; j++) mean += f[j];
    mean /= 5.0f;
    float var = 0.0f;
    for (int j = 0; j < 5; j++) var += (f[j] - mean) * (f[j] - mean);
    const float rstd = rsqrtf(var / 5.0f + kNormEps);
    float fn[5];
    for (int j = 0; j < 5; j++) fn[j] = rnd((f[j] - mean) * rstd * bf(ln_w[j]) + bf(ln_b[j]));
    for (int c = threadIdx.x; c < kH; c += kThreads) {
        float acc = 0.0f;
        for (int j = 0; j < 5; j++) acc += fn[j] * bf(up_w[c * 5 + j]);
        F1[(int64_t)n * kH + c] = bits(silu(rnd(acc + bf(up_b[c]))));
    }
}

/* The lattice's input rows (block n < N): x = ((((tok + pass) + feat) + slot) + rank), each Linear's output and each
 * sum rounded, into H; X = attn_norm(x).  Blocks N .. N + B - 1: bank b's anchor a = bf16(context_proj + bias) * valid,
 * A = anchor_norm(a). */
/// The lattice input rows x (rounded sums of the projections and embeddings) and attn_norm(x); the extra blocks build each bank's normed anchor.
/// Launch: grid N + banks, block kThreads.  The Z* inputs are the front door's f32 Linear outputs, biases bf16 [kH].
/// @param N     lattice rows (56 x banks).
/// @param Zt    token_proj(E), f32 [N][kH].
/// @param bt    token_proj bias.
/// @param Zp    pass_hidden_proj(pass), f32 [banks][7][kH].
/// @param bp    pass_hidden_proj bias.
/// @param Zf    feature_mlp down output, f32 [N][kH].
/// @param bfd   feature_mlp down bias.
/// @param slot  slot embedding, bf16 [7][kH].
/// @param rank  rank embedding, bf16 [8][kH].
/// @param norm0 layer 0's attn_norm weight, bf16 [kH].
/// @param Zc    context_proj(anchor_hidden), f32 [banks][kH].
/// @param bc    context_proj bias.
/// @param valid anchor_valid [banks] (0 zeroes the anchor).
/// @param anorm anchor_norm weight, bf16 [kH].
/// @param H     out: the residual stream x, bf16 [N][kH].
/// @param X     out: attn_norm(x), bf16 [N][kH].
/// @param A     out: anchor_norm(a), bf16 [banks][kH].
__global__ void __launch_bounds__(kThreads) k_embed(int N, const float *Zt, const uint16_t *bt, const float *Zp,
                                                    const uint16_t *bp, const float *Zf, const uint16_t *bfd,
                                                    const uint16_t *slot, const uint16_t *rank, const uint16_t *norm0,
                                                    const float *Zc, const uint16_t *bc, const uint8_t *valid,
                                                    const uint16_t *anorm, uint16_t *H, uint16_t *X, uint16_t *A) {
    float v[kH / kThreads];
    const int n = blockIdx.x;
    if (n < N) {
        const int b = n / kN, s = (n % kN) / kK, k = n % kK;
        for (int j = 0; j < kH / kThreads; j++) {
            const int c = threadIdx.x + j * kThreads;
            float h = rnd(rnd(Zt[(int64_t)n * kH + c] + bf(bt[c])) + rnd(Zp[((int64_t)b * kS + s) * kH + c] + bf(bp[c])));
            h = rnd(h + rnd(Zf[(int64_t)n * kH + c] + bf(bfd[c])));
            h = rnd(h + bf(slot[s * kH + c]));
            h = rnd(h + bf(rank[k * kH + c]));
            H[(int64_t)n * kH + c] = bits(h);
            v[j] = h;
        }
        rmsnorm_row(v, norm0, X + (int64_t)n * kH);
    } else {
        const int b = n - N;
        const float keep = valid[b] ? 1.0f : 0.0f;
        for (int j = 0; j < kH / kThreads; j++) {
            const int c = threadIdx.x + j * kThreads;
            v[j] = rnd(Zc[(int64_t)b * kH + c] + bf(bc[c])) * keep;
        }
        rmsnorm_row(v, anorm, A + (int64_t)b * kH);
    }
}

/* in_proj's output + bias, rounded, split head-major: QKV[p][b * 8 + h][i][d] (p = q, k, v) for the batched GEMMs. */
/// in_proj output + bias, rounded and split head-major for the batched Q K^T / P V GEMMs.
/// Launch: grid N, block kThreads.
/// @param N    lattice rows (56 x banks).
/// @param Z    in_proj output, f32 [N][3 * kH].
/// @param bias in_proj bias, bf16 [3 * kH].
/// @param QKV  out: bf16 [3][banks * 8][56][128] (q, k, v head-major).
__global__ void __launch_bounds__(kThreads) k_qkv(int N, const float *Z, const uint16_t *bias, uint16_t *QKV) {
    const int n = blockIdx.x, b = n / kN, i = n % kN;
    for (int c = threadIdx.x; c < 3 * kH; c += kThreads) {
        const int p = c / kH, h = (c % kH) / kD, d = c % kD;
        QKV[(((int64_t)p * (N / kN) * kHeads + b * kHeads + h) * kN + i) * kD + d] = bits(Z[(int64_t)n * 3 * kH + c] +
                                                                                         bf(bias[c]));
    }
}

/* One attention row: softmax(q.k / sqrt(128) + bias) in f32 (SDPA with a bf16 additive mask), P rounded to bf16 for
 * the P V GEMM.  bias[h][i][j] = bf16(rel[h][slot(i) - slot(j) + 7] + same[h] * [slot(i) == slot(j)]). */
/// One attention row's biased softmax in f32 (slot-relative + same-slot bias), probabilities rounded to bf16.
/// Launch: grid banks x 8 x 56 (block = one score row, (bank, head) major), block 64 (thread = key column j < 56).
/// @param S    Q K^T, f32 [banks * 8][56][56].
/// @param rel  relative_slot_bias, bf16 [8][15].
/// @param same same_slot_bias, bf16 [8].
/// @param P    out: probabilities, bf16 [banks * 8][56][56].
__global__ void __launch_bounds__(64) k_softmax(const float *S, const uint16_t *rel, const uint16_t *same,
                                                uint16_t *P) {
    const int row = blockIdx.x, i = row % kN, h = (row / kN) % kHeads;
    const int j = threadIdx.x;
    const float scale = 1.0f / sqrtf((float)kD);
    float x = -INFINITY;
    if (j < kN) {
        const int si = i / kK, sj = j / kK;
        const float bias = si == sj ? rnd(bf(rel[h * (2 * kS + 1) + si - sj + kS]) + bf(same[h]))
                                    : bf(rel[h * (2 * kS + 1) + si - sj + kS]);
        x = S[(int64_t)row * kN + j] * scale + bias;
    }
    __shared__ float red[2];
    float m = x;
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((j & 31) == 0) red[j >> 5] = m;
    __syncthreads();
    m = fmaxf(red[0], red[1]);
    __syncthreads();
    const float e = j < kN ? expf(x - m) : 0.0f;
    float sum = e;
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if ((j & 31) == 0) red[j >> 5] = sum;
    __syncthreads();
    sum = red[0] + red[1];
    if (j < kN) P[(int64_t)row * kN + j] = bits(e / sum);
}

/* The attention output back to lattice rows, heads concatenated, rounded: Ob[n][h * 128 + d]. */
/// The attention output back to lattice rows, heads concatenated, rounded to bf16.
/// Launch: grid N, block kThreads.
/// @param O  P V, f32 [banks * 8][56][128].
/// @param Ob out: bf16 [N][kH].
__global__ void __launch_bounds__(kThreads) k_ocat(const float *O, uint16_t *Ob) {
    const int n = blockIdx.x, b = n / kN, i = n % kN;
    for (int c = threadIdx.x; c < kH; c += kThreads) {
        const int h = c / kD, d = c % kD;
        Ob[(int64_t)n * kH + c] = bits(O[(((int64_t)b * kHeads + h) * kN + i) * kD + d]);
    }
}

/* H = bf16(H + bf16(Z + bias)); X = rmsnorm(H) * w. */
/// H = bf16(H + bf16(Z + bias)); X = rmsnorm(H) * w.
/// Launch: grid N, block kThreads.
/// @param Z    the Linear's f32 output [N][kH].
/// @param bias its bias, bf16 [kH].
/// @param H    the residual stream, bf16 [N][kH]; updated in place.
/// @param w    the next norm's weight, bf16 [kH].
/// @param X    out: the normed rows, bf16 [N][kH].
__global__ void __launch_bounds__(kThreads) k_residual(const float *Z, const uint16_t *bias, uint16_t *H,
                                                       const uint16_t *w, uint16_t *X) {
    const int n = blockIdx.x;
    float v[kH / kThreads];
    for (int j = 0; j < kH / kThreads; j++) {
        const int c = threadIdx.x + j * kThreads;
        const float h = rnd(bf(H[(int64_t)n * kH + c]) + rnd(Z[(int64_t)n * kH + c] + bf(bias[c])));
        H[(int64_t)n * kH + c] = bits(h);
        v[j] = h;
    }
    rmsnorm_row(v, w, X + (int64_t)n * kH);
}

/* Y = bf16(silu(bf16(Z + bias))) over rows of `width`. */
/// Y = bf16(silu(bf16(Z + bias))) row by row.
/// Launch: grid rows, block kThreads.
/// @param Z     the Linear's f32 output [rows][width].
/// @param bias  its bias, bf16 [width].
/// @param width row width in elements.
/// @param Y     out: bf16 [rows][width].
__global__ void __launch_bounds__(kThreads) k_silu(const float *Z, const uint16_t *bias, int width, uint16_t *Y) {
    const int n = blockIdx.x;
    for (int c = threadIdx.x; c < width; c += kThreads)
        Y[(int64_t)n * width + c] = bits(silu(rnd(Z[(int64_t)n * width + c] + bf(bias[c]))));
}

/* factor_input_proj's input row [x | a | bf16(x * a)] (3072), a = the bank's normed anchor. */
/// Builds factor_input_proj's input row [x | a | bf16(x * a)] (3 * kH) per lattice row.
/// Launch: grid N, block kThreads.
/// @param X  output_norm(x), bf16 [N][kH].
/// @param A  anchor_norm(a), bf16 [banks][kH].
/// @param X3 out: bf16 [N][3 * kH].
__global__ void __launch_bounds__(kThreads) k_factor_in(const uint16_t *X, const uint16_t *A, uint16_t *X3) {
    const int n = blockIdx.x, b = n / kN;
    for (int c = threadIdx.x; c < kH; c += kThreads) {
        const uint16_t x = X[(int64_t)n * kH + c], a = A[(int64_t)b * kH + c];
        uint16_t *r = X3 + (int64_t)n * 3 * kH;
        r[c] = x;
        r[kH + c] = a;
        r[2 * kH + c] = bits(bf(x) * bf(a));
    }
}

/* F.normalize of bf16(Z + bias), row by row: nrm = bf16(||x||), y = bf16(x / max(nrm, eps)). */
/// F.normalize of bf16(Z + bias) row by row: y = bf16(x / max(bf16(||x||), eps)).
/// Launch: grid rows, block kThreads (rows of kH).
/// @param Z    the Linear's f32 output [rows][kH].
/// @param bias its bias, bf16 [kH].
/// @param eps  the norm's floor (vector_eps).
/// @param Y    out: unit rows, bf16 [rows][kH].
__global__ void __launch_bounds__(kThreads) k_normalize(const float *Z, const uint16_t *bias, float eps, uint16_t *Y) {
    const int n = blockIdx.x;
    float v[kH / kThreads], ss = 0.0f;
    for (int j = 0; j < kH / kThreads; j++) {
        const int c = threadIdx.x + j * kThreads;
        v[j] = rnd(Z[(int64_t)n * kH + c] + bf(bias[c]));
        ss += v[j] * v[j];
    }
    const float den = fmaxf(rnd(sqrtf(block_sum(ss))), eps);
    for (int j = 0; j < kH / kThreads; j++) Y[(int64_t)n * kH + threadIdx.x + j * kThreads] = bits(v[j] / den);
}

/* One bank: start[j] = bf16(sum_c bf16(aout[c] * in[0][j][c])), pair[s][i][j] = bf16(sum_c out[s][i][c] *
 * in[s+1][j][c]) (a bf16 matmul: f32 sum, one rounding), scores = logit_scale * them; then the walk. */
/// One bank's start and pair scores (normalized dot products, scaled), then the greedy walk (first maximum per slot).
/// Launch: grid banks, block kThreads (warp per dot product); 1.8 KB static smem.
/// @param Vout        normalize(out_head g), bf16 [banks][56][kH].
/// @param Vin         normalize(in_head g), bf16 [banks][56][kH].
/// @param Va          normalize(anchor_out_head a), bf16 [banks][kH].
/// @param cand        candidate ids [banks][7][8].
/// @param logit_scale the scores' scale (8).
/// @param scores      out: f32 [banks][7][8][8].
/// @param chain_ids   out: the walk's tokens [banks][7].
/// @param chain_rows  out: the walk's candidate index per slot [banks][7].
__global__ void __launch_bounds__(kThreads) k_scores(const uint16_t *Vout, const uint16_t *Vin, const uint16_t *Va,
                                                     const int32_t *cand, float logit_scale, float *scores,
                                                     int32_t *chain_ids, int32_t *chain_rows) {
    const int b = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __shared__ float sc[kS][kK][kK];
    const uint16_t *out = Vout + (int64_t)b * kN * kH, *in = Vin + (int64_t)b * kN * kH, *a = Va + (int64_t)b * kH;
    for (int t = warp; t < kK + (kS - 1) * kK * kK; t += kThreads / 32) {
        float acc = 0.0f;
        if (t < kK) {
            for (int c = lane; c < kH; c += 32) acc += rnd(bf(a[c]) * bf(in[t * kH + c]));
        } else {
            const int u = t - kK, s = u / (kK * kK), i = (u / kK) % kK, j = u % kK;
            const uint16_t *x = out + (s * kK + i) * kH, *y = in + ((s + 1) * kK + j) * kH;
            for (int c = lane; c < kH; c += 32) acc += bf(x[c]) * bf(y[c]);
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) {
            const float v = logit_scale * rnd(acc);
            if (t < kK) {
                for (int i = 0; i < kK; i++) sc[0][i][t] = v;
            } else {
                const int u = t - kK;
                sc[1 + u / (kK * kK)][(u / kK) % kK][u % kK] = v;
            }
        }
    }
    __syncthreads();
    float *dst = scores + (int64_t)b * kS * kK * kK;
    for (int i = threadIdx.x; i < kS * kK * kK; i += kThreads) dst[i] = (&sc[0][0][0])[i];
    if (threadIdx.x == 0) {
        int prev = 0;
        for (int s = 0; s < kS; s++) {
            int best = 0;
            for (int j = 1; j < kK; j++)
                if (sc[s][prev][j] > sc[s][prev][best]) best = j;   /* the first maximum (tl.max: ties to the left) */
            chain_rows[b * kS + s] = best;
            chain_ids[b * kS + s] = cand[(b * kS + s) * kK + best];
            prev = best;
        }
    }
}

/* The workspace carve: one pointer per buffer, in this order, each 256-byte aligned. */
/// One launch's workspace buffers (carve lays them out, each 256-byte aligned); N = 56 x banks.
struct ws_plan {
    uint16_t *E, *F1, *H, *X, *X3, *A, *QKV, *P, *Ob, *Vout, *Vin, *Va;
    float *Zt, *Zp, *Zf, *Zc, *Z, *Z2, *Za, *S, *O;
/** @var Zt
 *  f32 [N][kH] token_proj output.
 *  @var Zp
 *  f32 [banks][7][kH] pass_hidden_proj output.
 *  @var Zf
 *  f32 [N][kH] feature_mlp down output.
 *  @var Zc
 *  f32 [banks][kH] context_proj output.
 *  @var Z
 *  f32 [N][3 * kH] the per-layer Linear outputs (and out_head's).
 *  @var Z2
 *  f32 [N][kH] in_head output.
 *  @var Za
 *  f32 [banks][kH] anchor_out_head output.
 *  @var S
 *  f32 [banks * 8][56][56] attention scores Q K^T.
 *  @var O
 *  f32 [banks * 8][56][128] attention output P V. */
/** @var E
 *  bf16 [N][kM] gathered candidate embeddings.
 *  @var F1
 *  bf16 [N][kH] feature rows (feature_mlp up + SiLU).
 *  @var H
 *  bf16 [N][kH] the lattice residual stream.
 *  @var X
 *  bf16 [N][kH] the current normed rows.
 *  @var X3
 *  bf16 [N][3 * kH] factor_input_proj's input.
 *  @var A
 *  bf16 [banks][kH] the normed anchor.
 *  @var QKV
 *  bf16 [3][banks * 8][56][128] the head-major q, k, v.
 *  @var P
 *  bf16 [banks * 8][56][56] attention probabilities.
 *  @var Ob
 *  bf16 [N][2 * kH]: the attention output (kH wide) and the MLP / factor SiLU rows (up to 2 * kH).
 *  @var Vout
 *  bf16 [N][kH] normalized out_head rows.
 *  @var Vin
 *  bf16 [N][kH] normalized in_head rows.
 *  @var Va
 *  bf16 [banks][kH] normalized anchor_out_head rows. */
};

/// @param B    banks.
/// @param base the workspace (256-byte aligned), or NULL to only size it.
/// @param p    out: the buffers' pointers (NULL ones when base is NULL), or NULL.
/// @return the workspace bytes the plan takes.
uint64_t carve(uint32_t B, char *base, ws_plan *p) {
    const uint64_t N = (uint64_t)kN * B;
    uint64_t off = 0;
    auto take = [&](uint64_t bytes) -> char * {
        char *r = base ? base + off : nullptr;
        off += (bytes + 255) & ~(uint64_t)255;
        return r;
    };
    ws_plan q;
    q.E = (uint16_t *)take(N * kM * 2);
    q.F1 = (uint16_t *)take(N * kH * 2);
    q.H = (uint16_t *)take(N * kH * 2);
    q.X = (uint16_t *)take(N * kH * 2);
    q.X3 = (uint16_t *)take(N * 3 * kH * 2);
    q.A = (uint16_t *)take((uint64_t)B * kH * 2);
    q.QKV = (uint16_t *)take(3 * N * kH * 2);
    q.P = (uint16_t *)take((uint64_t)B * kHeads * kN * kN * 2);
    q.Ob = (uint16_t *)take(N * 2 * kH * 2);
    q.Vout = (uint16_t *)take(N * kH * 2);
    q.Vin = (uint16_t *)take(N * kH * 2);
    q.Va = (uint16_t *)take((uint64_t)B * kH * 2);
    q.Zt = (float *)take(N * kH * 4);
    q.Zp = (float *)take((uint64_t)B * kS * kH * 4);
    q.Zf = (float *)take(N * kH * 4);
    q.Zc = (float *)take((uint64_t)B * kH * 4);
    q.Z = (float *)take(N * 3 * kH * 4);
    q.Z2 = (float *)take(N * kH * 4);
    q.Za = (float *)take((uint64_t)B * kH * 4);
    q.S = (float *)take((uint64_t)B * kHeads * kN * kN * 4);
    q.O = (float *)take(N * kH * 4);
    if (p) *p = q;
    return off;
}

/// Checks one Linear binding: weight and bias set, in -> out widths, exactly one of BF16 / nvfp4; names a mismatch on stderr.
/// @param l    the binding.
/// @param in   expected input width.
/// @param out  expected output width.
/// @param name the role's name for the message.
/// @return true when it is usable.
bool lin_ok(const pulsar_lilicorr_linear &l, int in, int out, const char *name) {
    if (l.l.w && l.b && l.l.in == in && l.l.out == out && (l.l.bf16 != l.l.nvfp4)) return true;
    fprintf(stderr, "pulsar: LiLiCorr head: Linear %s must be %d -> %d, BF16 or nvfp4, with its bias (has %d -> %d, "
                    "bf16 %d nvfp4 %d, w %p b %p) -- refusing\n", name, in, out, l.l.in, l.l.out, (int)l.l.bf16,
            (int)l.l.nvfp4, l.l.w, (const void *)l.b);
    return false;
}

/// y = x W^T through the rows front door (no bias; the glue kernels add it).
/// @param l    the Linear.
/// @param x    bf16 input rows [rows][in].
/// @param rows rows.
/// @param y    out: f32 [rows][out].
/// @param st   the stream.
/// @return pulsar_rows_linear_launch's status (0 = launched).
int linear(const pulsar_lilicorr_linear &l, const uint16_t *x, int rows, float *y, cudaStream_t st) {
    return pulsar_rows_linear_launch(&l.l, x, rows, y, nullptr, 0, st);
}

/// Checks the last launch; names a failure on stderr.
/// @param what the launch's name for the message.
/// @return true when no CUDA error is pending.
bool launched(const char *what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    fprintf(stderr, "pulsar: LiLiCorr head: %s launch failed: %s\n", what, cudaGetErrorString(e));
    return false;
}

} // namespace

extern "C" uint64_t pulsar_lilicorr_head_ws_bytes(uint32_t n_banks) {
    if (n_banks == 0 || n_banks > PULSAR_LILICORR_BANKS_MAX) return 0;
    return carve(n_banks, nullptr, nullptr);
}

extern "C" int pulsar_lilicorr_head_launch(const pulsar_lilicorr_head *w, uint32_t n_banks, const int32_t *cand_ids,
                                           const float *cand_logprob, const uint16_t *embed_table,
                                           const uint16_t *pass_hidden, const uint16_t *anchor_hidden,
                                           const uint8_t *anchor_valid, float *scores, int32_t *chain_ids,
                                           int32_t *chain_rows, void *ws, uint64_t ws_bytes, cudaStream_t st) {
    const uint64_t need = pulsar_lilicorr_head_ws_bytes(n_banks);
    if (!w || !need || !cand_ids || !cand_logprob || !embed_table || !pass_hidden || !anchor_hidden || !anchor_valid ||
        !scores || !chain_ids || !chain_rows || !ws || ws_bytes < need || ((uintptr_t)ws & 255u)) {
        fprintf(stderr, "pulsar: LiLiCorr head: %u banks (1..%u), every input and output, and a 256-byte-aligned "
                        "workspace of %llu bytes (got %llu) -- refusing\n", n_banks, PULSAR_LILICORR_BANKS_MAX,
                (unsigned long long)need, (unsigned long long)ws_bytes);
        return -1;
    }
    bool ok = lin_ok(w->token_proj, kM, kH, "token_proj") && lin_ok(w->pass_hidden_proj, kM, kH, "pass_hidden_proj") &&
              lin_ok(w->context_proj, kM, kH, "context_proj") &&
              lin_ok(w->feature_mlp_down, kH, kH, "feature_mlp.down_proj") &&
              lin_ok(w->factor_input_proj, 3 * kH, kH, "factor_input_proj") && lin_ok(w->out_head, kH, kH, "out_head") &&
              lin_ok(w->in_head, kH, kH, "in_head") && lin_ok(w->anchor_out_head, kH, kH, "anchor_out_head");
    for (int li = 0; li < (int)PULSAR_LILICORR_LAYERS && ok; li++) {
        const pulsar_lilicorr_layer &L = w->layers[li];
        ok = lin_ok(L.lat_attn_in, kH, 3 * kH, "layers.attn.in_proj") &&
             lin_ok(L.lat_attn_out, kH, kH, "layers.attn.out_proj") && lin_ok(L.lat_mlp_up, kH, 2 * kH, "layers.mlp.up_proj") &&
             lin_ok(L.lat_mlp_down, 2 * kH, kH, "layers.mlp.down_proj") && L.lat_attn_norm && L.lat_mlp_norm;
    }
    if (!ok || !w->feature_norm_w || !w->feature_norm_b || !w->feature_mlp_up_w || !w->feature_mlp_up_b ||
        !w->slot_embedding || !w->rank_embedding || !w->relative_slot_bias || !w->same_slot_bias || !w->output_norm ||
        !w->anchor_norm) {
        if (ok) fprintf(stderr, "pulsar: LiLiCorr head: a norm, embedding or bias vector is missing -- refusing\n");
        return -1;
    }
    static bool announced = false;
    if (!__atomic_exchange_n(&announced, true, __ATOMIC_RELAXED))
        fprintf(stderr, "pulsar: LiLiCorr head: Linears through the rows front door (token_proj %s), lattice attention "
                        "= strided-batched cuBLASLt, glue kernels for features / epilogues / softmax / scores / walk "
                        "(first: %u banks)\n", w->token_proj.l.nvfp4 ? "nvfp4" : "bf16", n_banks);

    ws_plan p;
    carve(n_banks, (char *)ws, &p);
    const int B = (int)n_banks, N = kN * B, BH = B * kHeads;
    const int64_t qkv_part = (int64_t)N * kH;   /* one of q / k / v, head-major [B * 8][56][128] */

    k_gather<<<N, kThreads, 0, st>>>(cand_ids, cand_logprob, embed_table, w->feature_norm_w, w->feature_norm_b,
                                     w->feature_mlp_up_w, w->feature_mlp_up_b, p.E, p.F1);
    if (!launched("gather")) return -3;
    if (linear(w->token_proj, p.E, N, p.Zt, st) || linear(w->pass_hidden_proj, pass_hidden, B * kS, p.Zp, st) ||
        linear(w->feature_mlp_down, p.F1, N, p.Zf, st) || linear(w->context_proj, anchor_hidden, B, p.Zc, st))
        return -1;
    k_embed<<<N + B, kThreads, 0, st>>>(N, p.Zt, w->token_proj.b, p.Zp, w->pass_hidden_proj.b, p.Zf,
                                        w->feature_mlp_down.b, w->slot_embedding, w->rank_embedding,
                                        w->layers[0].lat_attn_norm, p.Zc, w->context_proj.b, anchor_valid,
                                        w->anchor_norm, p.H, p.X, p.A);
    if (!launched("embed")) return -3;
    for (int li = 0; li < (int)PULSAR_LILICORR_LAYERS; li++) {
        const pulsar_lilicorr_layer &L = w->layers[li];
        const uint16_t *next_norm = li + 1 < (int)PULSAR_LILICORR_LAYERS ? w->layers[li + 1].lat_attn_norm
                                                                          : w->output_norm;
        if (linear(L.lat_attn_in, p.X, N, p.Z, st)) return -1;
        k_qkv<<<N, kThreads, 0, st>>>(N, p.Z, L.lat_attn_in.b, p.QKV);
        if (!launched("qkv")) return -3;
        /* S = Q K^T per (bank, head); P = softmax(S / sqrt(128) + bias); O = P V */
        if (!pulsar_gpu_bf16_batched_gemm(p.S, kN, (int64_t)kN * kN, p.QKV, kD, (int64_t)kN * kD, p.QKV + qkv_part, kD,
                                          (int64_t)kN * kD, 1, kN, kN, kD, BH, st))
            return -1;
        k_softmax<<<BH * kN, 64, 0, st>>>(p.S, w->relative_slot_bias, w->same_slot_bias, p.P);
        if (!launched("softmax")) return -3;
        if (!pulsar_gpu_bf16_batched_gemm(p.O, kD, (int64_t)kN * kD, p.P, kN, (int64_t)kN * kN, p.QKV + 2 * qkv_part,
                                          kD, (int64_t)kN * kD, 0, kN, kD, kN, BH, st))
            return -1;
        k_ocat<<<N, kThreads, 0, st>>>(p.O, p.Ob);
        if (!launched("ocat")) return -3;
        if (linear(L.lat_attn_out, p.Ob, N, p.Z, st)) return -1;
        k_residual<<<N, kThreads, 0, st>>>(p.Z, L.lat_attn_out.b, p.H, L.lat_mlp_norm, p.X);
        if (!launched("attn residual")) return -3;
        if (linear(L.lat_mlp_up, p.X, N, p.Z, st)) return -1;
        k_silu<<<N, kThreads, 0, st>>>(p.Z, L.lat_mlp_up.b, 2 * kH, p.Ob);
        if (!launched("mlp silu")) return -3;
        if (linear(L.lat_mlp_down, p.Ob, N, p.Z, st)) return -1;
        k_residual<<<N, kThreads, 0, st>>>(p.Z, L.lat_mlp_down.b, p.H, next_norm, p.X);
        if (!launched("mlp residual")) return -3;
    }
    /* p.X = output_norm(x); p.A = anchor_norm(a) */
    k_factor_in<<<N, kThreads, 0, st>>>(p.X, p.A, p.X3);
    if (!launched("factor input")) return -3;
    if (linear(w->factor_input_proj, p.X3, N, p.Z, st)) return -1;
    k_silu<<<N, kThreads, 0, st>>>(p.Z, w->factor_input_proj.b, kH, p.Ob);
    if (!launched("factor silu")) return -3;
    if (linear(w->out_head, p.Ob, N, p.Z, st) || linear(w->in_head, p.Ob, N, p.Z2, st) ||
        linear(w->anchor_out_head, p.A, B, p.Za, st))
        return -1;
    k_normalize<<<N, kThreads, 0, st>>>(p.Z, w->out_head.b, w->vector_eps, p.Vout);
    k_normalize<<<N, kThreads, 0, st>>>(p.Z2, w->in_head.b, w->vector_eps, p.Vin);
    k_normalize<<<B, kThreads, 0, st>>>(p.Za, w->anchor_out_head.b, w->vector_eps, p.Va);
    if (!launched("normalize")) return -3;
    k_scores<<<B, kThreads, 0, st>>>(p.Vout, p.Vin, p.Va, cand_ids, w->logit_scale, scores, chain_ids, chain_rows);
    return launched("scores + walk") ? 0 : -3;
}

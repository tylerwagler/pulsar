/* Qwen3.8-Flash-Next full attention + QSA (L251 stream S3).
 *
 * The contract, the cache layout and the geometry live in src/pulsar_gpu.h
 * (PULSAR_QSA_*); this file is the one implementation.  Kernels, in launch
 * order, per call:
 *
 *   qsa_block_keys  every row that COMPLETES an indexer block (pos % 4 == 3):
 *                   mean-pool the block's 4 raw keys (this call's rows, or the
 *                   stage for tokens of earlier calls), RMSNorm, RoPE at the
 *                   block start, store bf16 -> bkey[pos / 4]
 *   qsa_prep        per row: q heads norm + RoPE -> f32 scratch; K norm + RoPE
 *                   and V -> E4M3/MX32 into the cache; indexer q norm + RoPE ->
 *                   f32 scratch; the raw indexer key -> stage when its block
 *                   stays open past this call
 *   qsa_score       rows with nb > 512 only: score every complete block
 *   (pulsar_gpu_indexer_topk_tensor -- the engine's top-512, reused as is)
 *   qsa_sel_sort    the 512 selected blocks ascending, so every row attends in
 *                   token order
 *   qsa_attn_split  (split of 128 listed tokens, KV head, row): the 12 GQA query
 *                   heads against the split's FP8 K/V -> an unnormalised partial
 *   qsa_combine     fold the splits IN ORDER, divide, times sigmoid(gate), emit
 *                   the o_proj E4M3 slot (and the f32 tap when asked)
 *
 * ONE ARITHMETIC PER ROW.  Nothing a row computes depends on what else is in
 * the call: splits are 128 LISTED tokens whatever the batch, the fold order is
 * the split order, the score of (row, block) is one thread's fixed-order dot,
 * and the engine's top-k is a strict total order on (score, index).  So decode
 * (one row per sequence) and chunked prefill (a run of rows) give identical
 * bytes -- asserted, not argued, by tests/qsa_attn_gate.cu.
 */
#include "pulsar_cuda_internal.h"
#include "pulsar_cuda_mx.cuh"

#include <algorithm>

namespace {

constexpr uint32_t QSA_SPLIT       = 128u;   /* listed tokens per attention split */
constexpr uint32_t QSA_TILE        = 32u;    /* tokens per smem tile */
constexpr uint32_t QSA_GQA         = PULSAR_QSA_N_HEAD / PULSAR_QSA_N_KV;
constexpr uint32_t QSA_BUDGET      = PULSAR_QSA_TOP_BLOCKS * PULSAR_QSA_BLOCK;   /* 2048 */
constexpr uint32_t QSA_MAX_LISTED  = QSA_BUDGET + PULSAR_QSA_BLOCK - 1u;         /* 2051 */
constexpr uint32_t QSA_MAX_SPLITS  = (QSA_MAX_LISTED + QSA_SPLIT - 1u) / QSA_SPLIT;  /* 17 */
constexpr uint32_t QSA_ATT_GROUP   = 256u;   /* rows per attention pass (bounds the partials) */
constexpr uint64_t QSA_SCORE_BUDGET = 64ull << 20;   /* bytes of scores per selection pass */
constexpr uint32_t QSA_SCORE_BLK   = 128u;   /* blocks per score CTA */
constexpr uint32_t QSA_KS          = PULSAR_QSA_HEAD_DIM + 4u;   /* smem row pitch (floats) */
constexpr uint32_t QSA_BKS         = PULSAR_QSA_IDX_DIM / 2u + 4u;  /* bkey smem pitch (u32 of bf16x2) */

static_assert(QSA_GQA == 12u, "GQA ratio");
static_assert(PULSAR_QSA_KV_TOKEN_BYTES ==
              2u * PULSAR_QSA_N_KV * (PULSAR_QSA_HEAD_DIM + PULSAR_QSA_HEAD_DIM / 32u),
              "KV token record = K and V rows of every KV head, data then scales");
static_assert(PULSAR_QSA_BKEY_BYTES == PULSAR_QSA_IDX_DIM * 2u, "bf16 block key");
static_assert(PULSAR_QSA_STAGE_BYTES == PULSAR_QSA_BLOCK * PULSAR_QSA_IDX_DIM * 4u, "f32 stage");
static_assert(PULSAR_QSA_Q_IN == PULSAR_QSA_N_HEAD * 2u * PULSAR_QSA_HEAD_DIM, "q + gate per head");
static_assert(PULSAR_QSA_IDX_IN == (PULSAR_QSA_IDX_HEADS + 1u) * PULSAR_QSA_IDX_DIM, "idx q + k");
static_assert(PULSAR_QSA_ROT_DIM == 64u, "the RoPE pairs (i, i+32) are one lane's elements 0 and 1");

/* Byte offsets inside one token's 1056-B record. */
__host__ __device__ constexpr uint32_t qsa_k_data(uint32_t g) { return g * PULSAR_QSA_HEAD_DIM; }
__host__ __device__ constexpr uint32_t qsa_v_data(uint32_t g) { return (PULSAR_QSA_N_KV + g) * PULSAR_QSA_HEAD_DIM; }
__host__ __device__ constexpr uint32_t qsa_k_scale(uint32_t g) {
    return 2u * PULSAR_QSA_N_KV * PULSAR_QSA_HEAD_DIM + g * (PULSAR_QSA_HEAD_DIM / 32u);
}
__host__ __device__ constexpr uint32_t qsa_v_scale(uint32_t g) { return qsa_k_scale(PULSAR_QSA_N_KV + g); }

enum : uint32_t {
    QSA_ROW_COMPLETES = 1u,   /* pos % 4 == 3: this row finishes block pos / 4 */
    QSA_ROW_STAGE     = 2u,   /* the row's block is still open after this call */
    QSA_ROW_SELECT    = 4u,   /* nb > 512: the row attends to a selection */
};

struct qsa_row {
    uint8_t       *kv;
    __nv_bfloat16 *bkey;
    float         *stage;
    uint32_t pos;
    uint32_t run_off;   /* rows of the same sequence before this one in the call */
    uint32_t flags;
    uint32_t nb;        /* complete blocks visible: (pos + 1) / 4 */
    uint32_t n_list;    /* tokens attended: pos + 1, or 2048 + (pos + 1) % 4 */
    uint32_t pad;
};

struct qsa_rope_tab { float inv[PULSAR_QSA_ROT_DIM / 2]; };

__device__ __forceinline__ float qsa_warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float qsa_warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* cos/sin of the rope angle for pair `i` at `pos`.  The angle is the f32
 * product transformers forms (inv_freq f32 times the position as f32); its
 * cos/sin are taken in double so a large angle is not the fast-math __sinf's. */
__device__ __forceinline__ void qsa_rope_cs(const qsa_rope_tab &tab, uint32_t pos, uint32_t i,
                                            float *c, float *s) {
    const float ang = __fmul_rn((float)pos, tab.inv[i]);
    double sd, cd;
    sincos((double)ang, &sd, &cd);
    *c = (float)cd;
    *s = (float)sd;
}

/* One warp, one head of NJ*32 elements: lane l holds x[l + 32 j].  RMSNorm with
 * weight (1 + w), then RoPE on the pair (x[l], x[l + 32]) -- lane l's j = 0, 1. */
template <int NJ>
__device__ __forceinline__ void qsa_norm_rope(float (&x)[NJ], const float *w, float c, float s) {
    const int lane = threadIdx.x & 31;
    float ss = 0.f;
    #pragma unroll
    for (int j = 0; j < NJ; j++) ss = __fmaf_rn(x[j], x[j], ss);
    ss = qsa_warp_sum(ss);
    const float r = rsqrtf(__fadd_rn(__fdiv_rn(ss, (float)(NJ * 32)), PULSAR_QSA_RMS_EPS));
    #pragma unroll
    for (int j = 0; j < NJ; j++) x[j] = __fmul_rn(__fmul_rn(x[j], r), __fadd_rn(1.0f, w[lane + 32 * j]));
    const float x0 = x[0], x1 = x[1];
    x[0] = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
    x[1] = __fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s));
}

/* 8 x 32-element MX blocks of one 256-wide head row: block j is every lane's x[j]. */
__device__ __forceinline__ void qsa_emit_row(const float (&x)[8], uint8_t *data, uint8_t *scales) {
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int se = pulsar_mx_shared_exp(qsa_warp_max(fabsf(x[j])));
        __nv_fp8_e4m3 q = pulsar_mx_encode(x[j], se);
        data[lane + 32 * j] = *reinterpret_cast<uint8_t *>(&q);
        if (lane == 0) scales[j] = pulsar_mx_scale_byte(se);
    }
}

/* ---- block keys ------------------------------------------------------------- */
__global__ void qsa_block_keys_kernel(const qsa_row *rows, uint32_t n_rows, const float *idx,
                                      const float *kw, qsa_rope_tab tab) {
    const uint32_t r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= n_rows) return;
    const qsa_row row = rows[r];
    if (!(row.flags & QSA_ROW_COMPLETES)) return;
    const int lane = threadIdx.x & 31;
    float x[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t k = 0; k < PULSAR_QSA_BLOCK; k++) {     /* tokens pos-3 .. pos, ascending */
        const uint32_t off = PULSAR_QSA_BLOCK - 1u - k;    /* distance back from this row */
        const float *src = off <= row.run_off
                ? idx + (uint64_t)(r - off) * PULSAR_QSA_IDX_IN + PULSAR_QSA_IDX_HEADS * PULSAR_QSA_IDX_DIM
                : row.stage + (uint64_t)k * PULSAR_QSA_IDX_DIM;   /* slot = (pos - off) % 4 = k */
        #pragma unroll
        for (int j = 0; j < 4; j++) x[j] = __fadd_rn(x[j], src[lane + 32 * j]);
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) x[j] = __fmul_rn(x[j], 0.25f);
    float c, s;
    qsa_rope_cs(tab, row.pos - (PULSAR_QSA_BLOCK - 1u), (uint32_t)lane, &c, &s);
    qsa_norm_rope<4>(x, kw, c, s);
    __nv_bfloat16 *dst = row.bkey + (uint64_t)(row.pos / PULSAR_QSA_BLOCK) * PULSAR_QSA_IDX_DIM;
    #pragma unroll
    for (int j = 0; j < 4; j++) dst[lane + 32 * j] = __float2bfloat16_rn(x[j]);
}

/* ---- prep: 32 warps per row ------------------------------------------------- *
 * warps 0..23 query heads, 24..25 K heads, 26..27 V heads, 28..31 indexer q heads
 * (warp 28 also stages the raw indexer key). */
__global__ void __launch_bounds__(1024) qsa_prep_kernel(
        const qsa_row *rows, const float *qg, const float *kin, const float *vin, const float *idx,
        const float *qw, const float *kw, const float *iqw, qsa_rope_tab tab,
        float *q_out, float *iq_out) {
    const uint32_t r = blockIdx.x;
    const qsa_row row = rows[r];
    const uint32_t w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    float c, s;
    qsa_rope_cs(tab, row.pos, (uint32_t)lane, &c, &s);
    if (w < PULSAR_QSA_N_HEAD) {
        const float *src = qg + (uint64_t)r * PULSAR_QSA_Q_IN + (uint64_t)w * 2u * PULSAR_QSA_HEAD_DIM;
        float x[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) x[j] = src[lane + 32 * j];
        qsa_norm_rope<8>(x, qw, c, s);
        float *dst = q_out + ((uint64_t)r * PULSAR_QSA_N_HEAD + w) * PULSAR_QSA_HEAD_DIM;
        #pragma unroll
        for (int j = 0; j < 8; j++) dst[lane + 32 * j] = x[j];
    } else if (w < PULSAR_QSA_N_HEAD + 2u * PULSAR_QSA_N_KV) {
        const uint32_t g = (w - PULSAR_QSA_N_HEAD) % PULSAR_QSA_N_KV;
        const bool is_k = w < PULSAR_QSA_N_HEAD + PULSAR_QSA_N_KV;
        const float *src = (is_k ? kin : vin) + (uint64_t)r * PULSAR_QSA_KV_IN + (uint64_t)g * PULSAR_QSA_HEAD_DIM;
        float x[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) x[j] = src[lane + 32 * j];
        if (is_k) qsa_norm_rope<8>(x, kw, c, s);
        uint8_t *rec = row.kv + (uint64_t)row.pos * PULSAR_QSA_KV_TOKEN_BYTES;
        qsa_emit_row(x, rec + (is_k ? qsa_k_data(g) : qsa_v_data(g)),
                     rec + (is_k ? qsa_k_scale(g) : qsa_v_scale(g)));
    } else {
        const uint32_t h = w - (PULSAR_QSA_N_HEAD + 2u * PULSAR_QSA_N_KV);
        const float *src = idx + (uint64_t)r * PULSAR_QSA_IDX_IN + (uint64_t)h * PULSAR_QSA_IDX_DIM;
        float x[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) x[j] = src[lane + 32 * j];
        qsa_norm_rope<4>(x, iqw, c, s);
        float *dst = iq_out + ((uint64_t)r * PULSAR_QSA_IDX_HEADS + h) * PULSAR_QSA_IDX_DIM;
        #pragma unroll
        for (int j = 0; j < 4; j++) dst[lane + 32 * j] = x[j];
        if (h == 0 && (row.flags & QSA_ROW_STAGE)) {
            const float *kraw = idx + (uint64_t)r * PULSAR_QSA_IDX_IN + PULSAR_QSA_IDX_HEADS * PULSAR_QSA_IDX_DIM;
            float *st = row.stage + (uint64_t)(row.pos % PULSAR_QSA_BLOCK) * PULSAR_QSA_IDX_DIM;
            #pragma unroll
            for (int j = 0; j < 4; j++) st[lane + 32 * j] = kraw[lane + 32 * j];
        }
    }
}

/* ---- block scores (selecting rows only) -------------------------------------- *
 * grid (block chunk, listed row); one thread per block.  score = sum over the 4
 * heads, in head order, of relu(q_h . k) with each dot a sequential f32 FMA
 * chain over the 128 dims.  Blocks past the row's nb score -inf, which the
 * top-k never prefers to a real (>= 0) score. */
__global__ void __launch_bounds__(QSA_SCORE_BLK) qsa_score_kernel(
        const qsa_row *rows, const uint32_t *list, const float *iq, float *scores, uint32_t nb_max) {
    __shared__ float qs[PULSAR_QSA_IDX_HEADS * PULSAR_QSA_IDX_DIM];
    __shared__ uint32_t ks[QSA_SCORE_BLK * QSA_BKS];
    const uint32_t r = list[blockIdx.y];
    const qsa_row row = rows[r];
    for (uint32_t i = threadIdx.x; i < PULSAR_QSA_IDX_HEADS * PULSAR_QSA_IDX_DIM; i += blockDim.x) {
        qs[i] = iq[(uint64_t)r * PULSAR_QSA_IDX_HEADS * PULSAR_QSA_IDX_DIM + i];
    }
    const uint32_t b0 = blockIdx.x * QSA_SCORE_BLK;
    /* coalesced: 16 B per thread per step, 16 steps per block key */
    const uint4 *src = reinterpret_cast<const uint4 *>(row.bkey);
    for (uint32_t i = threadIdx.x; i < QSA_SCORE_BLK * 16u; i += blockDim.x) {
        const uint32_t bb = i >> 4, part = i & 15u;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (b0 + bb < row.nb) v = src[(uint64_t)(b0 + bb) * 16u + part];
        *reinterpret_cast<uint4 *>(&ks[bb * QSA_BKS + part * 4u]) = v;
    }
    __syncthreads();
    const uint32_t b = b0 + threadIdx.x;
    if (b >= nb_max) return;
    float out = -INFINITY;
    if (b < row.nb) {
        float acc[PULSAR_QSA_IDX_HEADS] = {0.f, 0.f, 0.f, 0.f};
        const uint32_t *kr = &ks[threadIdx.x * QSA_BKS];
        for (uint32_t d2 = 0; d2 < PULSAR_QSA_IDX_DIM / 2u; d2 += 4u) {
            const uint4 kk = *reinterpret_cast<const uint4 *>(&kr[d2]);
            const uint32_t words[4] = {kk.x, kk.y, kk.z, kk.w};
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const float2 kf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&words[e]));
                const uint32_t d = 2u * (d2 + (uint32_t)e);
                #pragma unroll
                for (int h = 0; h < PULSAR_QSA_IDX_HEADS; h++) {
                    acc[h] = __fmaf_rn(qs[h * PULSAR_QSA_IDX_DIM + d], kf.x, acc[h]);
                    acc[h] = __fmaf_rn(qs[h * PULSAR_QSA_IDX_DIM + d + 1u], kf.y, acc[h]);
                }
            }
        }
        out = 0.f;
        #pragma unroll
        for (int h = 0; h < PULSAR_QSA_IDX_HEADS; h++) out = __fadd_rn(out, fmaxf(acc[h], 0.f));
    }
    scores[(uint64_t)blockIdx.y * nb_max + b] = out;
}

/* ---- the selection ascending (bitonic, 512 in smem) --------------------------- */
__global__ void __launch_bounds__(256) qsa_sel_sort_kernel(const uint32_t *list, const uint32_t *topk,
                                                          uint32_t *sel) {
    __shared__ uint32_t v[PULSAR_QSA_TOP_BLOCKS];
    const uint32_t g = blockIdx.x;
    for (uint32_t i = threadIdx.x; i < PULSAR_QSA_TOP_BLOCKS; i += blockDim.x) {
        v[i] = topk[(uint64_t)g * PULSAR_QSA_TOP_BLOCKS + i];
    }
    __syncthreads();
    for (uint32_t k = 2; k <= PULSAR_QSA_TOP_BLOCKS; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            for (uint32_t i = threadIdx.x; i < PULSAR_QSA_TOP_BLOCKS; i += blockDim.x) {
                const uint32_t p = i ^ j;
                if (p > i) {
                    const bool up = (i & k) == 0u;
                    const uint32_t a = v[i], b = v[p];
                    if ((a > b) == up) { v[i] = b; v[p] = a; }
                }
            }
            __syncthreads();
        }
    }
    uint32_t *dst = sel + (uint64_t)list[g] * PULSAR_QSA_TOP_BLOCKS;
    for (uint32_t i = threadIdx.x; i < PULSAR_QSA_TOP_BLOCKS; i += blockDim.x) dst[i] = v[i];
}

/* The token at listed position j of a row: every token for a row that attends
 * to all of them, else the selected blocks' tokens then the open tail. */
__device__ __forceinline__ uint32_t qsa_listed_token(const qsa_row &row, const uint32_t *sel, uint32_t j) {
    if (!(row.flags & QSA_ROW_SELECT)) return j;
    if (j < QSA_BUDGET) return sel[j / PULSAR_QSA_BLOCK] * PULSAR_QSA_BLOCK + (j % PULSAR_QSA_BLOCK);
    return row.nb * PULSAR_QSA_BLOCK + (j - QSA_BUDGET);
}

/* Dequantise one 32-token tile of K or V rows for KV head g into smem f32.
 * Thread (token tt, 32-element block cb) walks its 8 words starting at word cb,
 * so the 8 threads of a quarter-warp store to 8 different bank groups. */
__device__ __forceinline__ void qsa_load_tile(float *dst, const uint8_t *kv, const uint32_t *toks,
                                              uint32_t n, uint32_t data_off, uint32_t scale_off) {
    if (threadIdx.x >= QSA_TILE * 8u) return;
    const uint32_t tt = threadIdx.x >> 3, cb = threadIdx.x & 7u;   /* token, 32-elem block */
    float *o = dst + tt * QSA_KS + cb * 32u;
    if (tt >= n) {
        #pragma unroll
        for (int wi = 0; wi < 8; wi++) {
            *reinterpret_cast<float4 *>(o + (((uint32_t)wi + cb) & 7u) * 4u) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        return;
    }
    const uint8_t *rec = kv + (uint64_t)toks[tt] * PULSAR_QSA_KV_TOKEN_BYTES;
    const uint32_t *w32 = reinterpret_cast<const uint32_t *>(rec + data_off + cb * 32u);
    const float sc = exp2f((float)((int)rec[scale_off + cb] - 127));
    #pragma unroll
    for (int wi = 0; wi < 8; wi++) {
        const uint32_t k = ((uint32_t)wi + cb) & 7u;
        const uint32_t word = w32[k];
        float f[4];
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            __nv_fp8_e4m3 q;
            *reinterpret_cast<uint8_t *>(&q) = (uint8_t)(word >> (8 * e));
            f[e] = __fmul_rn((float)q, sc);
        }
        *reinterpret_cast<float4 *>(o + k * 4u) = make_float4(f[0], f[1], f[2], f[3]);
    }
}

/* ---- attention partials ------------------------------------------------------ *
 * grid (split, KV head, row in group); 12 warps = the 12 query heads of the KV
 * head.  Online softmax over the split's tokens in list order; writes the
 * split's (m, l) and its UNNORMALISED accumulator (base-2 exponent domain). */
__global__ void __launch_bounds__(QSA_GQA * 32) qsa_attn_split_kernel(
        const qsa_row *rows, uint32_t row0, const uint32_t *sel, const float *q,
        float *part, float2 *ml) {
    __shared__ __align__(16) float qs[QSA_GQA * PULSAR_QSA_HEAD_DIM];
    __shared__ __align__(16) float ts[QSA_TILE * QSA_KS];
    __shared__ uint32_t toks[QSA_SPLIT];
    const uint32_t s = blockIdx.x, g = blockIdx.y, rg = blockIdx.z, r = row0 + rg;
    const qsa_row row = rows[r];
    const uint32_t j0 = s * QSA_SPLIT;
    if (j0 >= row.n_list) return;
    const uint32_t n = min(QSA_SPLIT, row.n_list - j0);
    const uint32_t *rsel = sel + (uint64_t)r * PULSAR_QSA_TOP_BLOCKS;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) toks[i] = qsa_listed_token(row, rsel, j0 + i);
    const float *qsrc = q + ((uint64_t)r * PULSAR_QSA_N_HEAD + g * QSA_GQA) * PULSAR_QSA_HEAD_DIM;
    for (uint32_t i = threadIdx.x; i < QSA_GQA * PULSAR_QSA_HEAD_DIM; i += blockDim.x) qs[i] = qsrc[i];
    __syncthreads();

    const uint32_t h = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    /* scores in the base-2 domain: (q . k) / sqrt(256) * log2(e) */
    const float sc = 0.0625f * 1.4426950408889634f;
    float m = -INFINITY, l = 0.f;
    float acc[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) acc[k] = 0.f;
    const float4 *q4 = reinterpret_cast<const float4 *>(&qs[h * PULSAR_QSA_HEAD_DIM]);

    for (uint32_t t0 = 0; t0 < n; t0 += QSA_TILE) {
        const uint32_t nt = min(QSA_TILE, n - t0);
        qsa_load_tile(ts, row.kv, toks + t0, nt, qsa_k_data(g), qsa_k_scale(g));
        __syncthreads();
        float sv = -INFINITY;
        if (lane < nt) {
            const float4 *k4 = reinterpret_cast<const float4 *>(&ts[lane * QSA_KS]);
            float d = 0.f;
            for (uint32_t i = 0; i < PULSAR_QSA_HEAD_DIM / 4u; i++) {
                const float4 a = q4[i], b = k4[i];
                d = __fmaf_rn(a.x, b.x, d);
                d = __fmaf_rn(a.y, b.y, d);
                d = __fmaf_rn(a.z, b.z, d);
                d = __fmaf_rn(a.w, b.w, d);
            }
            sv = __fmul_rn(d, sc);
        }
        const float mn = fmaxf(m, qsa_warp_max(sv));
        const float corr = exp2f(__fsub_rn(m, mn));
        const float p = exp2f(__fsub_rn(sv, mn));
        l = __fadd_rn(__fmul_rn(l, corr), qsa_warp_sum(p));
        m = mn;
        __syncthreads();                       /* every warp is done with K */
        qsa_load_tile(ts, row.kv, toks + t0, nt, qsa_v_data(g), qsa_v_scale(g));
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 8; k++) acc[k] = __fmul_rn(acc[k], corr);
        for (uint32_t t = 0; t < nt; t++) {
            const float pt = __shfl_sync(0xffffffffu, p, (int)t);
            const float *vr = &ts[t * QSA_KS + lane];
            #pragma unroll
            for (int k = 0; k < 8; k++) acc[k] = __fmaf_rn(pt, vr[32 * k], acc[k]);
        }
        __syncthreads();                       /* every warp is done with V */
    }
    const uint64_t slot = ((uint64_t)rg * PULSAR_QSA_N_HEAD + g * QSA_GQA + h) * QSA_MAX_SPLITS + s;
    if (lane == 0) ml[slot] = make_float2(m, l);
    #pragma unroll
    for (int k = 0; k < 8; k++) part[slot * PULSAR_QSA_HEAD_DIM + lane + 32 * k] = acc[k];
}

/* ---- fold + gate + o_proj slot ------------------------------------------------ *
 * grid (row in group, query head); thread = dim.  The fold visits the splits in
 * order with ONE recurrence, so the result is a function of the row alone. */
__global__ void __launch_bounds__(PULSAR_QSA_HEAD_DIM) qsa_combine_kernel(
        const qsa_row *rows, uint32_t row0, const float *part, const float2 *ml, const float *qg,
        __nv_fp8_e4m3 *out, unsigned char *out_scale, int kbp, float *tap) {
    const uint32_t rg = blockIdx.x, h = blockIdx.y, d = threadIdx.x, r = row0 + rg;
    const qsa_row row = rows[r];
    const uint32_t ns = (row.n_list + QSA_SPLIT - 1u) / QSA_SPLIT;
    const uint64_t base = ((uint64_t)rg * PULSAR_QSA_N_HEAD + h) * QSA_MAX_SPLITS;
    float M = -INFINITY, L = 0.f, A = 0.f;
    for (uint32_t s = 0; s < ns; s++) {
        const float2 e = ml[base + s];
        const float a = part[(base + s) * PULSAR_QSA_HEAD_DIM + d];
        const float mn = fmaxf(M, e.x);
        const float c1 = exp2f(__fsub_rn(M, mn)), c2 = exp2f(__fsub_rn(e.x, mn));
        L = __fadd_rn(__fmul_rn(L, c1), __fmul_rn(e.y, c2));
        A = __fadd_rn(__fmul_rn(A, c1), __fmul_rn(a, c2));
        M = mn;
    }
    const float o = __fdiv_rn(A, L);
    const float gate = qg[(uint64_t)r * PULSAR_QSA_Q_IN + (uint64_t)h * 2u * PULSAR_QSA_HEAD_DIM
                          + PULSAR_QSA_HEAD_DIM + d];
    const float y = __fmul_rn(o, __fdiv_rn(1.0f, __fadd_rn(1.0f, __expf(-gate))));
    const uint32_t col = h * PULSAR_QSA_HEAD_DIM + d;
    if (tap) tap[(uint64_t)r * PULSAR_QSA_OUT_DIM + col] = y;
    pulsar_mx_emit_block(y, col, r, PULSAR_QSA_OUT_DIM, kbp, out, out_scale);
}

__global__ void qsa_sel_tap_kernel(const qsa_row *rows, const uint32_t *sel, uint32_t *tap, uint32_t n_rows) {
    const uint32_t r = blockIdx.x;
    if (r >= n_rows) return;
    const bool selecting = (rows[r].flags & QSA_ROW_SELECT) != 0u;
    for (uint32_t i = threadIdx.x; i < PULSAR_QSA_TOP_BLOCKS; i += blockDim.x) {
        tap[(uint64_t)r * PULSAR_QSA_TOP_BLOCKS + i] =
                selecting ? sel[(uint64_t)r * PULSAR_QSA_TOP_BLOCKS + i] : 0xffffffffu;
    }
}

/* ---- workspace geometry: ONE function sizes it and carves it ------------------- */
struct qsa_ws {
    qsa_row  *rows;
    uint32_t *list;
    float    *q;
    float    *iq;
    uint32_t *sel;
    float    *scores;
    uint32_t *topk;
    float    *part;
    float2   *ml;
    uint64_t  bytes;
    uint32_t  score_rows;   /* rows per selection pass */
};

static uint64_t qsa_up(uint64_t x) { return (x + 255u) & ~255ull; }

static qsa_ws qsa_ws_layout(uint8_t *base, uint32_t n_rows, uint32_t max_ctx) {
    qsa_ws w{};
    const uint32_t nb_max = max_ctx / PULSAR_QSA_BLOCK;
    const bool selects = nb_max > PULSAR_QSA_TOP_BLOCKS;
    const uint64_t score_row_bytes = (uint64_t)nb_max * sizeof(float);
    w.score_rows = selects ? (uint32_t)std::min<uint64_t>(n_rows, std::max<uint64_t>(1u, QSA_SCORE_BUDGET / score_row_bytes)) : 0u;
    const uint32_t att_rows = std::min(n_rows, QSA_ATT_GROUP);
    uint64_t off = 0;
    auto take = [&](uint64_t bytes) { uint8_t *p = base ? base + off : nullptr; off += qsa_up(bytes); return p; };
    w.rows   = (qsa_row *)take((uint64_t)n_rows * sizeof(qsa_row));
    w.list   = (uint32_t *)take((uint64_t)n_rows * sizeof(uint32_t));
    w.q      = (float *)take((uint64_t)n_rows * PULSAR_QSA_N_HEAD * PULSAR_QSA_HEAD_DIM * sizeof(float));
    w.iq     = (float *)take((uint64_t)n_rows * PULSAR_QSA_IDX_HEADS * PULSAR_QSA_IDX_DIM * sizeof(float));
    w.sel    = (uint32_t *)take((uint64_t)n_rows * PULSAR_QSA_TOP_BLOCKS * sizeof(uint32_t));
    w.scores = (float *)take((uint64_t)w.score_rows * score_row_bytes);
    w.topk   = (uint32_t *)take((uint64_t)w.score_rows * PULSAR_QSA_TOP_BLOCKS * sizeof(uint32_t));
    w.part   = (float *)take((uint64_t)att_rows * PULSAR_QSA_N_HEAD * QSA_MAX_SPLITS * PULSAR_QSA_HEAD_DIM * sizeof(float));
    w.ml     = (float2 *)take((uint64_t)att_rows * PULSAR_QSA_N_HEAD * QSA_MAX_SPLITS * sizeof(float2));
    w.bytes  = off;
    return w;
}

static int qsa_refuse(const char *why) {
    fprintf(stderr, "pulsar: qsa attention refused: %s\n", why);
    return 0;
}

static pulsar_shape_once g_qsa_announce;

}  // namespace

void pulsar_qsa_inv_freq(float inv_freq[PULSAR_QSA_ROT_DIM / 2]) {
    for (uint32_t i = 0; i < PULSAR_QSA_ROT_DIM / 2; i++) {
        inv_freq[i] = (float)(1.0 / pow(PULSAR_QSA_ROPE_THETA, (double)(2u * i) / (double)PULSAR_QSA_ROT_DIM));
    }
}

uint64_t pulsar_gpu_qsa_workspace_bytes(uint32_t n_rows, uint32_t max_ctx) {
    return qsa_ws_layout(nullptr, n_rows, max_ctx).bytes;
}

int pulsar_gpu_qsa_forward(const pulsar_qsa_layer *layer,
                           const pulsar_qsa_seq *seqs, uint32_t n_seqs,
                           const uint32_t *row_seq, const uint32_t *row_pos, uint32_t n_rows,
                           const pulsar_qsa_io *io, pulsar_gpu_tensor *workspace) {
    if (!layer || !seqs || !row_seq || !row_pos || !io || !workspace || n_rows == 0 || n_seqs == 0) {
        return qsa_refuse("a NULL argument or zero rows");
    }
    if (!layer->q_norm || layer->q_norm->bytes < PULSAR_QSA_HEAD_DIM * 4u ||
        !layer->k_norm || layer->k_norm->bytes < PULSAR_QSA_HEAD_DIM * 4u ||
        !layer->idx_q_norm || layer->idx_q_norm->bytes < PULSAR_QSA_IDX_DIM * 4u ||
        !layer->idx_k_norm || layer->idx_k_norm->bytes < PULSAR_QSA_IDX_DIM * 4u) {
        return qsa_refuse("a norm weight is missing or short");
    }
    const uint64_t R = n_rows;
    if (!io->qg || io->qg->bytes < R * PULSAR_QSA_Q_IN * 4u || !io->k || io->k->bytes < R * PULSAR_QSA_KV_IN * 4u ||
        !io->v || io->v->bytes < R * PULSAR_QSA_KV_IN * 4u || !io->idx || io->idx->bytes < R * PULSAR_QSA_IDX_IN * 4u) {
        return qsa_refuse("a projection input is missing or shorter than n_rows rows");
    }
    if (!io->out_e4m3 || !io->out_scale || io->out_sf_pitch != pulsar_mx_kbp(PULSAR_QSA_OUT_DIM)) {
        return qsa_refuse("no o_proj E4M3 slot (or its scale pitch is not the 6144-wide KBp)");
    }
    if ((io->tap_out_f32 && io->tap_out_f32->bytes < R * PULSAR_QSA_OUT_DIM * 4u) ||
        (io->tap_sel && io->tap_sel->bytes < R * PULSAR_QSA_TOP_BLOCKS * 4u)) {
        return qsa_refuse("a tap is shorter than n_rows rows");
    }

    /* rows -> descriptors; every sequence is ONE run of consecutive positions */
    std::vector<qsa_row> h_rows(n_rows);
    std::vector<uint32_t> run_first(n_seqs, UINT32_MAX), run_last(n_seqs, UINT32_MAX);
    uint32_t max_ctx = 0;
    for (uint32_t r = 0; r < n_rows; r++) {
        const uint32_t sq = row_seq[r];
        if (sq >= n_seqs) return qsa_refuse("row_seq names no sequence");
        const pulsar_qsa_seq &S = seqs[sq];
        if (!S.kv || !S.bkey || !S.stage || S.cap == 0 || S.cap % PULSAR_QSA_BLOCK ||
            S.kv->bytes < (uint64_t)S.cap * PULSAR_QSA_KV_TOKEN_BYTES ||
            S.bkey->bytes < (uint64_t)(S.cap / PULSAR_QSA_BLOCK) * PULSAR_QSA_BKEY_BYTES ||
            S.stage->bytes < PULSAR_QSA_STAGE_BYTES) {
            return qsa_refuse("a sequence's cache is missing, short, or its cap is not a multiple of 4");
        }
        if (row_pos[r] >= S.cap) return qsa_refuse("a row's position is past its sequence's cap");
        if (run_first[sq] == UINT32_MAX) {
            run_first[sq] = r;
        } else if (run_last[sq] != r - 1u || row_pos[r] != row_pos[r - 1u] + 1u) {
            return qsa_refuse("a sequence's rows are not one contiguous run of consecutive positions");
        }
        run_last[sq] = r;
        max_ctx = std::max(max_ctx, row_pos[r] + 1u);
    }
    uint32_t n_select = 0;
    for (uint32_t r = 0; r < n_rows; r++) {
        const uint32_t sq = row_seq[r], p = row_pos[r];
        qsa_row &d = h_rows[r];
        d.kv    = (uint8_t *)seqs[sq].kv->ptr;
        d.bkey  = (__nv_bfloat16 *)seqs[sq].bkey->ptr;
        d.stage = (float *)seqs[sq].stage->ptr;
        d.pos = p;
        d.run_off = r - run_first[sq];
        d.nb = (p + 1u) / PULSAR_QSA_BLOCK;
        const uint32_t block_end = (p / PULSAR_QSA_BLOCK) * PULSAR_QSA_BLOCK + PULSAR_QSA_BLOCK - 1u;
        d.flags = (p % PULSAR_QSA_BLOCK == PULSAR_QSA_BLOCK - 1u ? QSA_ROW_COMPLETES : 0u)
                | (row_pos[run_last[sq]] < block_end ? QSA_ROW_STAGE : 0u)
                | (d.nb > PULSAR_QSA_TOP_BLOCKS ? QSA_ROW_SELECT : 0u);
        d.n_list = d.nb > PULSAR_QSA_TOP_BLOCKS ? QSA_BUDGET + (p + 1u) % PULSAR_QSA_BLOCK : p + 1u;
        d.pad = 0;
        if (d.flags & QSA_ROW_SELECT) n_select++;
    }
    const qsa_ws need = qsa_ws_layout(nullptr, n_rows, max_ctx);
    if (!workspace->ptr || workspace->bytes < need.bytes) {
        fprintf(stderr, "pulsar: qsa attention refused: workspace %llu B < %llu B for %u rows at ctx %u\n",
                (unsigned long long)workspace->bytes, (unsigned long long)need.bytes, n_rows, max_ctx);
        return 0;
    }
    const qsa_ws ws = qsa_ws_layout((uint8_t *)workspace->ptr, n_rows, max_ctx);
    std::vector<uint32_t> h_list;
    h_list.reserve(n_select);
    for (uint32_t r = 0; r < n_rows; r++) if (h_rows[r].flags & QSA_ROW_SELECT) h_list.push_back(r);

    if (pulsar_shape_once_first(&g_qsa_announce, pulsar_shape_key(n_rows, n_select ? 1u : 0u), "qsa attention")) {
        fprintf(stderr, "pulsar: qsa attention: %u rows (%u selecting top-%u blocks), KV E4M3 MX32, "
                        "splits of %u listed tokens\n", n_rows, n_select, PULSAR_QSA_TOP_BLOCKS, QSA_SPLIT);
    }

    if (!cuda_ok(cudaMemcpyAsync(ws.rows, h_rows.data(), R * sizeof(qsa_row), cudaMemcpyHostToDevice),
                 "qsa rows upload")) return 0;
    if (n_select && !cuda_ok(cudaMemcpyAsync(ws.list, h_list.data(), (uint64_t)n_select * 4u,
                                             cudaMemcpyHostToDevice), "qsa list upload")) return 0;
    qsa_rope_tab tab;
    pulsar_qsa_inv_freq(tab.inv);

    qsa_block_keys_kernel<<<(n_rows + 7u) / 8u, 256>>>(ws.rows, n_rows, (const float *)io->idx->ptr,
                                                       (const float *)layer->idx_k_norm->ptr, tab);
    if (!cuda_ok(cudaGetLastError(), "qsa block keys launch")) return 0;
    qsa_prep_kernel<<<n_rows, 1024>>>(ws.rows, (const float *)io->qg->ptr, (const float *)io->k->ptr,
                                      (const float *)io->v->ptr, (const float *)io->idx->ptr,
                                      (const float *)layer->q_norm->ptr, (const float *)layer->k_norm->ptr,
                                      (const float *)layer->idx_q_norm->ptr, tab, ws.q, ws.iq);
    if (!cuda_ok(cudaGetLastError(), "qsa prep launch")) return 0;

    for (uint32_t g0 = 0; g0 < n_select; g0 += ws.score_rows) {
        const uint32_t gn = std::min(ws.score_rows, n_select - g0);
        uint32_t nb_max = 0;
        for (uint32_t i = 0; i < gn; i++) nb_max = std::max(nb_max, h_rows[h_list[g0 + i]].nb);
        dim3 grid((nb_max + QSA_SCORE_BLK - 1u) / QSA_SCORE_BLK, gn);
        qsa_score_kernel<<<grid, QSA_SCORE_BLK>>>(ws.rows, ws.list + g0, ws.iq, ws.scores, nb_max);
        if (!cuda_ok(cudaGetLastError(), "qsa score launch")) return 0;
        pulsar_gpu_tensor sc{}, tk{};
        sc.ptr = ws.scores; sc.bytes = (uint64_t)gn * nb_max * sizeof(float);
        tk.ptr = ws.topk;   tk.bytes = (uint64_t)gn * PULSAR_QSA_TOP_BLOCKS * sizeof(uint32_t);
        if (!pulsar_gpu_indexer_topk_tensor(&tk, &sc, nb_max, gn, PULSAR_QSA_TOP_BLOCKS)) {
            return qsa_refuse("the indexer top-k refused the block scores");
        }
        qsa_sel_sort_kernel<<<gn, 256>>>(ws.list + g0, ws.topk, ws.sel);
        if (!cuda_ok(cudaGetLastError(), "qsa selection sort launch")) return 0;
    }

    for (uint32_t r0 = 0; r0 < n_rows; r0 += QSA_ATT_GROUP) {
        const uint32_t gn = std::min(QSA_ATT_GROUP, n_rows - r0);
        uint32_t ns_max = 0;
        for (uint32_t i = 0; i < gn; i++) {
            ns_max = std::max(ns_max, (h_rows[r0 + i].n_list + QSA_SPLIT - 1u) / QSA_SPLIT);
        }
        qsa_attn_split_kernel<<<dim3(ns_max, PULSAR_QSA_N_KV, gn), QSA_GQA * 32>>>(ws.rows, r0, ws.sel, ws.q,
                                                                                  ws.part, ws.ml);
        if (!cuda_ok(cudaGetLastError(), "qsa attention split launch")) return 0;
        qsa_combine_kernel<<<dim3(gn, PULSAR_QSA_N_HEAD), PULSAR_QSA_HEAD_DIM>>>(
                ws.rows, r0, ws.part, ws.ml, (const float *)io->qg->ptr,
                (__nv_fp8_e4m3 *)io->out_e4m3, (unsigned char *)io->out_scale, io->out_sf_pitch,
                io->tap_out_f32 ? (float *)io->tap_out_f32->ptr : nullptr);
        if (!cuda_ok(cudaGetLastError(), "qsa combine launch")) return 0;
    }
    if (io->tap_sel) {
        qsa_sel_tap_kernel<<<n_rows, 128>>>(ws.rows, ws.sel, (uint32_t *)io->tap_sel->ptr, n_rows);
        if (!cuda_ok(cudaGetLastError(), "qsa selection tap launch")) return 0;
    }
    return 1;
}

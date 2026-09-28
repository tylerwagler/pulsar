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
 *   qsa_attn_split  (split of 64 listed tokens, KV head, row): the 12 GQA query
 *                   heads against the split's FP8 K/V on the tensor cores -> an
 *                   unnormalised partial
 *   qsa_combine     fold the splits IN ORDER, divide, times sigmoid(gate), emit
 *                   the o_proj E4M3 slot (and the f32 tap when asked)
 *
 * ONE ARITHMETIC PER ROW.  Nothing a row computes depends on what else is in
 * the call: splits are 64 LISTED tokens whatever the batch, the fold order is
 * the split order, the score of (row, block) is one thread's fixed-order dot,
 * and the engine's top-k is a strict total order on (score, index).  So decode
 * (one row per sequence) and chunked prefill (a run of rows) give identical
 * bytes -- asserted, not argued, by tests/qsa_attn_gate.cu.
 */
#include "pulsar_cuda_internal.h"
#include "pulsar_cuda_mx.cuh"

#include <algorithm>

namespace {

constexpr uint32_t QSA_SPLIT       = 64u;    /* listed tokens per attention split */
constexpr uint32_t QSA_GQA         = PULSAR_QSA_N_HEAD / PULSAR_QSA_N_KV;
constexpr uint32_t QSA_BUDGET      = PULSAR_QSA_TOP_BLOCKS * PULSAR_QSA_BLOCK;   /* 2048 */
constexpr uint32_t QSA_MAX_LISTED  = QSA_BUDGET + PULSAR_QSA_BLOCK - 1u;         /* 2051 */
constexpr uint32_t QSA_MAX_SPLITS  = (QSA_MAX_LISTED + QSA_SPLIT - 1u) / QSA_SPLIT;  /* 33 */
constexpr uint64_t QSA_SCORE_BUDGET = 64ull << 20;   /* bytes of scores per selection pass */
constexpr uint32_t QSA_SCORE_BLK   = 128u;   /* blocks per score CTA */
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
__device__ __forceinline__ void qsa_norm_rope(float (&x)[NJ], const uint16_t *w, float c, float s) {
    const int lane = threadIdx.x & 31;
    float ss = 0.f;
    #pragma unroll
    for (int j = 0; j < NJ; j++) ss = __fmaf_rn(x[j], x[j], ss);
    ss = qsa_warp_sum(ss);
    const float r = rsqrtf(__fadd_rn(__fdiv_rn(ss, (float)(NJ * 32)), PULSAR_QSA_RMS_EPS));
    #pragma unroll
    for (int j = 0; j < NJ; j++)   /* bf16 -> f32 is a left shift of the stored bits */
        x[j] = __fmul_rn(__fmul_rn(x[j], r), __fadd_rn(1.0f, __uint_as_float((uint32_t)w[lane + 32 * j] << 16)));
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
                                      const uint16_t *kw, qsa_rope_tab tab) {
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
        const uint16_t *qw, const uint16_t *kw, const uint16_t *iqw, qsa_rope_tab tab,
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
 * grid (block chunk, row group); one thread per block.  A group is up to
 * QSA_SCORE_ROWS consecutive listed rows of ONE sequence (a prefill run), so a
 * chunk of block keys is read once for all of them; decode rows are groups of
 * one.  score = sum over the 4 heads, in head order, of relu(q_h . k) with each
 * dot a sequential f32 FMA chain over the 128 dims -- per (row, block), whatever
 * the grouping.  Blocks past a row's nb score -inf, which the top-k never
 * prefers to a real (>= 0) score. */
constexpr uint32_t QSA_SCORE_ROWS = 4u;

__device__ __forceinline__ float qsa_block_score(const float *qs, const uint32_t *kr) {
    float acc[PULSAR_QSA_IDX_HEADS] = {0.f, 0.f, 0.f, 0.f};
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
    float out = 0.f;
    #pragma unroll
    for (int h = 0; h < PULSAR_QSA_IDX_HEADS; h++) out = __fadd_rn(out, fmaxf(acc[h], 0.f));
    return out;
}

/* groups[i] = (first listed row of the pass << 8) | rows; scores are indexed by
 * the row's place in the pass. */
__global__ void __launch_bounds__(QSA_SCORE_BLK) qsa_score_kernel(
        const qsa_row *rows, const uint32_t *list, const uint32_t *groups, const float *iq, float *scores,
        uint32_t nb_max) {
    __shared__ float qs[QSA_SCORE_ROWS * PULSAR_QSA_IDX_HEADS * PULSAR_QSA_IDX_DIM];
    __shared__ uint32_t ks[QSA_SCORE_BLK * QSA_BKS];
    const uint32_t first = groups[blockIdx.y] >> 8, cnt = groups[blockIdx.y] & 0xffu;
    constexpr uint32_t QW = PULSAR_QSA_IDX_HEADS * PULSAR_QSA_IDX_DIM;
    uint32_t nb_hi = 0;
    for (uint32_t i = 0; i < cnt; i++) nb_hi = max(nb_hi, rows[list[first + i]].nb);
    for (uint32_t i = threadIdx.x; i < cnt * QW; i += blockDim.x) {
        qs[i] = iq[(uint64_t)list[first + i / QW] * QW + i % QW];
    }
    const uint32_t b0 = blockIdx.x * QSA_SCORE_BLK;
    /* coalesced: 16 B per thread per step, 16 steps per block key */
    const uint4 *src = reinterpret_cast<const uint4 *>(rows[list[first]].bkey);
    for (uint32_t i = threadIdx.x; i < QSA_SCORE_BLK * 16u; i += blockDim.x) {
        const uint32_t bb = i >> 4, part = i & 15u;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (b0 + bb < nb_hi) v = src[(uint64_t)(b0 + bb) * 16u + part];
        *reinterpret_cast<uint4 *>(&ks[bb * QSA_BKS + part * 4u]) = v;
    }
    __syncthreads();
    const uint32_t b = b0 + threadIdx.x;
    if (b >= nb_max) return;
    for (uint32_t i = 0; i < cnt; i++) {
        const uint32_t nb = rows[list[first + i]].nb;
        scores[(uint64_t)(first + i) * nb_max + b] =
                b < nb ? qsa_block_score(&qs[i * QW], &ks[threadIdx.x * QSA_BKS]) : -INFINITY;
    }
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

/* ---- attention on the tensor cores -------------------------------------------- *
 * One split = 64 LISTED tokens of one row against the 12 GQA query heads of KV
 * head g (MMA rows 12..15 are zero); 4 warps.
 *
 * f32-class arithmetic on f16 MMAs (m16n8k16, f32 accumulate):
 *   - K and V come from E4M3, which f16 holds exactly; the E8M0 scales stay out
 *     of the f16 operands and are applied in f32: QK is summed per 32-dim scale
 *     block and each block's partial is scaled into the score; for PV the V
 *     scale of (token, dim block) is folded into P before the split.
 *   - every f32 operand (q, and P times the V scale) enters as an f16 hi + lo
 *     pair, two MMAs -- ~22 significant bits, graded against double by the gate.
 * A split's softmax is its own: m = max over the split, p = 2^(s - m), l = sum p
 * in a fixed order.  The row's result folds the splits IN SPLIT ORDER with ONE
 * recurrence (qsa_fold) and ends in ONE epilogue (qsa_gated).
 *
 * Two schedules of that arithmetic, chosen by batch shape only:
 *   qsa_attn_kernel<false>  grid (split, KV head, row): each CTA one split, the
 *                           partials to global, qsa_combine_kernel folds -- the
 *                           parallelism a few decode rows need;
 *   qsa_attn_kernel<true>   grid (1, KV head, row): each CTA walks its row's
 *                           splits and folds in registers -- no partials, for
 *                           prefill-sized calls.
 * Same device functions, same operations in the same order, so the bytes are
 * identical (tests/qsa_attn_gate G6: one row per call vs chunked prefill). */
constexpr uint32_t QSA_TPITCH = PULSAR_QSA_HEAD_DIM + 16u;   /* bytes per smem K/V row */
constexpr uint32_t QSA_SPITCH = QSA_SPLIT + 4u;              /* floats per score row */
constexpr uint32_t QSA_FOLD_ROWS = 64u;                      /* calls at least this wide fold in the CTA */
static_assert(QSA_SPLIT == 64u, "the warp layout below tiles 64 tokens: 4 warps x 2 n-tiles (QK), 4 k-steps (PV)");

__device__ __forceinline__ void qsa_mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
/* f32 pair -> (hi, lo) packed f16x2 words: hi = f16(x), lo = f16(x - hi). */
__device__ __forceinline__ void qsa_split2(float x0, float x1, uint32_t &hi, uint32_t &lo) {
    const __half h0 = __float2half_rn(x0), h1 = __float2half_rn(x1);
    const __half l0 = __float2half_rn(__fsub_rn(x0, __half2float(h0)));
    const __half l1 = __float2half_rn(__fsub_rn(x1, __half2float(h1)));
    const __half2 H = __halves2half2(h0, h1), L = __halves2half2(l0, l1);
    hi = *reinterpret_cast<const uint32_t *>(&H);
    lo = *reinterpret_cast<const uint32_t *>(&L);
}
/* two E4M3 bytes (low byte = first element) -> packed f16x2, exact */
__device__ __forceinline__ uint32_t qsa_e4m3x2(uint16_t two) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)two, __NV_E4M3);
    uint32_t u;
    memcpy(&u, &h, 4);
    return u;
}

/* The fold of one split (m, l, a) into a row's running (M, L, A): THE recurrence. */
__device__ __forceinline__ void qsa_fold(float &M, float &L, float &A, float m, float l, float a) {
    const float mn = fmaxf(M, m);
    const float c1 = exp2f(__fsub_rn(M, mn)), c2 = exp2f(__fsub_rn(m, mn));
    L = __fadd_rn(__fmul_rn(L, c1), __fmul_rn(l, c2));
    A = __fadd_rn(__fmul_rn(A, c1), __fmul_rn(a, c2));
    M = mn;
}
/* The epilogue: normalise, times the sigmoid output gate. */
__device__ __forceinline__ float qsa_gated(float A, float L, float gate) {
    return __fmul_rn(__fdiv_rn(A, L), __fdiv_rn(1.0f, __fadd_rn(1.0f, __expf(-gate))));
}

/* One K or V tile (64 listed tokens, KV head g) into smem as raw E4M3 plus its
 * 8 scales per token; rows past n are zero (their scores are masked). */
__device__ __forceinline__ void qsa_load_tile(uint8_t *td, float *tsc, const uint8_t *kv, const uint32_t *toks,
                                              uint32_t n, uint32_t data_off, uint32_t scale_off) {
    const uint32_t t = threadIdx.x >> 1, half = threadIdx.x & 1u;     /* 128 threads: 2 per token */
    uint4 *dst = reinterpret_cast<uint4 *>(td + t * QSA_TPITCH + half * 128u);
    if (t >= n) {
        #pragma unroll
        for (int i = 0; i < 8; i++) dst[i] = make_uint4(0u, 0u, 0u, 0u);
        if (!half) for (int b = 0; b < 8; b++) tsc[t * 8u + b] = 0.f;
        return;
    }
    const uint8_t *rec = kv + (uint64_t)toks[t] * PULSAR_QSA_KV_TOKEN_BYTES;
    const uint4 *src = reinterpret_cast<const uint4 *>(rec + data_off + half * 128u);
    #pragma unroll
    for (int i = 0; i < 8; i++) dst[i] = src[i];
    if (!half) for (int b = 0; b < 8; b++) tsc[t * 8u + b] = exp2f((float)((int)rec[scale_off + b] - 127));
}

template <bool FOLD>
__global__ void __launch_bounds__(128) qsa_attn_kernel(
        const qsa_row *rows, uint32_t row0, const uint32_t *sel, const float *q, const float *qg,
        float *part, float2 *ml, __nv_bfloat16 *out_bf16, __nv_fp8_e4m3 *out, unsigned char *out_scale,
        int kbp, float *tap) {
    __shared__ __align__(16) float qs[QSA_GQA * PULSAR_QSA_HEAD_DIM];
    __shared__ __align__(16) uint8_t td[QSA_SPLIT * QSA_TPITCH];
    __shared__ float tsc[QSA_SPLIT * 8u];
    __shared__ float ss[16 * QSA_SPITCH];          /* scores, then P */
    __shared__ float sml[16 * 2];
    __shared__ uint32_t toks[QSA_SPLIT];
    const uint32_t g = blockIdx.y, rg = blockIdx.z, r = row0 + rg;
    const qsa_row row = rows[r];
    const uint32_t ns = (row.n_list + QSA_SPLIT - 1u) / QSA_SPLIT;
    if (!FOLD && blockIdx.x >= ns) return;
    const uint32_t *rsel = sel + (uint64_t)r * PULSAR_QSA_TOP_BLOCKS;
    const float *qsrc = q + ((uint64_t)r * PULSAR_QSA_N_HEAD + g * QSA_GQA) * PULSAR_QSA_HEAD_DIM;
    for (uint32_t i = threadIdx.x; i < QSA_GQA * PULSAR_QSA_HEAD_DIM / 4u; i += blockDim.x) {
        reinterpret_cast<float4 *>(qs)[i] = reinterpret_cast<const float4 *>(qsrc)[i];
    }
    const uint32_t w = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t gr = lane >> 2, c = lane & 3u;     /* fragment row group, column pair */
    const bool row_hi_live = gr + 8u < QSA_GQA;       /* MMA rows 8..15 hold heads 8..11 then padding */

    /* the running fold (FOLD): this lane's (head, dim) elements -- 2 dim blocks x 4 n-tiles x
     * (2 dims of head gr, 2 of head gr + 8) -- and each head's M, L */
    float FA[2][4][4];
    float FM[2] = {-INFINITY, -INFINITY}, FL[2] = {0.f, 0.f};
    #pragma unroll
    for (int bb = 0; bb < 2; bb++)
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) FA[bb][nt][0] = FA[bb][nt][1] = FA[bb][nt][2] = FA[bb][nt][3] = 0.f;

    for (uint32_t s = FOLD ? 0u : blockIdx.x; s < (FOLD ? ns : blockIdx.x + 1u); s++) {
        const uint32_t j0 = s * QSA_SPLIT;
        const uint32_t n = min(QSA_SPLIT, row.n_list - j0);
        __syncthreads();                          /* the previous split is done with smem */
        if (threadIdx.x < n) toks[threadIdx.x] = qsa_listed_token(row, rsel, j0 + threadIdx.x);
        __syncthreads();
        qsa_load_tile(td, tsc, row.kv, toks, n, qsa_k_data(g), qsa_k_scale(g));
        __syncthreads();

        /* ---- S = Q K^T: warp w owns tokens 16w .. 16w+15 (two n-tiles) */
        {
            float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            for (uint32_t b = 0; b < 8u; b++) {
                float pb[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
                #pragma unroll
                for (uint32_t kk = 0; kk < 2u; kk++) {
                    const uint32_t d0 = b * 32u + kk * 16u;
                    const float *q0 = &qs[gr * PULSAR_QSA_HEAD_DIM + d0 + 2u * c];
                    const float *q1 = &qs[(row_hi_live ? gr + 8u : gr) * PULSAR_QSA_HEAD_DIM + d0 + 2u * c];
                    uint32_t ah[4], al[4];
                    qsa_split2(q0[0], q0[1], ah[0], al[0]);
                    qsa_split2(row_hi_live ? q1[0] : 0.f, row_hi_live ? q1[1] : 0.f, ah[1], al[1]);
                    qsa_split2(q0[8], q0[9], ah[2], al[2]);
                    qsa_split2(row_hi_live ? q1[8] : 0.f, row_hi_live ? q1[9] : 0.f, ah[3], al[3]);
                    #pragma unroll
                    for (uint32_t nt = 0; nt < 2u; nt++) {
                        const uint8_t *kr = td + (w * 16u + nt * 8u + gr) * QSA_TPITCH + d0 + 2u * c;
                        const uint32_t b0 = qsa_e4m3x2(*reinterpret_cast<const uint16_t *>(kr));
                        const uint32_t b1 = qsa_e4m3x2(*reinterpret_cast<const uint16_t *>(kr + 8));
                        qsa_mma(pb[nt], ah[0], ah[1], ah[2], ah[3], b0, b1);
                        qsa_mma(pb[nt], al[0], al[1], al[2], al[3], b0, b1);
                    }
                }
                #pragma unroll
                for (uint32_t nt = 0; nt < 2u; nt++) {
                    const uint32_t t0 = w * 16u + nt * 8u + 2u * c;
                    const float s0 = tsc[t0 * 8u + b], s1 = tsc[(t0 + 1u) * 8u + b];
                    acc[nt][0] = __fmaf_rn(pb[nt][0], s0, acc[nt][0]);
                    acc[nt][1] = __fmaf_rn(pb[nt][1], s1, acc[nt][1]);
                    acc[nt][2] = __fmaf_rn(pb[nt][2], s0, acc[nt][2]);
                    acc[nt][3] = __fmaf_rn(pb[nt][3], s1, acc[nt][3]);
                }
            }
            /* scores in the base-2 domain: (q . k) / sqrt(256) * log2(e); masked past n */
            const float scl = 0.0625f * 1.4426950408889634f;
            #pragma unroll
            for (uint32_t nt = 0; nt < 2u; nt++) {
                const uint32_t t0 = w * 16u + nt * 8u + 2u * c;
                ss[gr * QSA_SPITCH + t0]             = t0 < n ? __fmul_rn(acc[nt][0], scl) : -INFINITY;
                ss[gr * QSA_SPITCH + t0 + 1u]        = t0 + 1u < n ? __fmul_rn(acc[nt][1], scl) : -INFINITY;
                ss[(gr + 8u) * QSA_SPITCH + t0]      = t0 < n ? __fmul_rn(acc[nt][2], scl) : -INFINITY;
                ss[(gr + 8u) * QSA_SPITCH + t0 + 1u] = t0 + 1u < n ? __fmul_rn(acc[nt][3], scl) : -INFINITY;
            }
        }
        __syncthreads();                          /* K is dead: V may land in td */
        qsa_load_tile(td, tsc, row.kv, toks, n, qsa_v_data(g), qsa_v_scale(g));

        /* ---- the split's softmax: head h = tid / 8, 8 tokens per thread, fixed order */
        {
            const uint32_t h = threadIdx.x >> 3, sub = threadIdx.x & 7u;
            float *sr = &ss[h * QSA_SPITCH + sub * 8u];
            float m = -INFINITY;
            #pragma unroll
            for (int i = 0; i < 8; i++) m = fmaxf(m, sr[i]);
            #pragma unroll
            for (int o = 4; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            float l = 0.f;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const float p = exp2f(__fsub_rn(sr[i], m));
                sr[i] = p;
                l = __fadd_rn(l, p);
            }
            #pragma unroll
            for (int o = 4; o > 0; o >>= 1) l = __fadd_rn(l, __shfl_xor_sync(0xffffffffu, l, o));
            if (sub == 0) { sml[h * 2] = m; sml[h * 2 + 1] = l; }
        }
        __syncthreads();

        /* ---- O = P V: warp w owns dim blocks 2w, 2w+1 (4 n-tiles each); the V
         * scale of (token, block) rides in the A operand */
        const uint64_t slot0 = ((uint64_t)rg * PULSAR_QSA_N_HEAD + g * QSA_GQA) * QSA_MAX_SPLITS + s;
        #pragma unroll
        for (uint32_t bb = 0; bb < 2u; bb++) {
            const uint32_t b = 2u * w + bb;
            float o[4][4];
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
            #pragma unroll
            for (uint32_t ks = 0; ks < QSA_SPLIT / 16u; ks++) {
                const uint32_t t0 = ks * 16u + 2u * c;
                const float v00 = tsc[t0 * 8u + b], v01 = tsc[(t0 + 1u) * 8u + b];
                const float v10 = tsc[(t0 + 8u) * 8u + b], v11 = tsc[(t0 + 9u) * 8u + b];
                const float *p0 = &ss[gr * QSA_SPITCH + t0];
                const float *p1 = &ss[(gr + 8u) * QSA_SPITCH + t0];
                uint32_t ah[4], al[4];
                qsa_split2(__fmul_rn(p0[0], v00), __fmul_rn(p0[1], v01), ah[0], al[0]);
                qsa_split2(row_hi_live ? __fmul_rn(p1[0], v00) : 0.f, row_hi_live ? __fmul_rn(p1[1], v01) : 0.f, ah[1], al[1]);
                qsa_split2(__fmul_rn(p0[8], v10), __fmul_rn(p0[9], v11), ah[2], al[2]);
                qsa_split2(row_hi_live ? __fmul_rn(p1[8], v10) : 0.f, row_hi_live ? __fmul_rn(p1[9], v11) : 0.f, ah[3], al[3]);
                #pragma unroll
                for (uint32_t nt = 0; nt < 4u; nt++) {
                    const uint8_t *vc = td + b * 32u + nt * 8u + gr;
                    const uint32_t b0 = qsa_e4m3x2((uint16_t)(vc[t0 * QSA_TPITCH] | (vc[(t0 + 1u) * QSA_TPITCH] << 8)));
                    const uint32_t b1 = qsa_e4m3x2((uint16_t)(vc[(t0 + 8u) * QSA_TPITCH] | (vc[(t0 + 9u) * QSA_TPITCH] << 8)));
                    qsa_mma(o[nt], ah[0], ah[1], ah[2], ah[3], b0, b1);
                    qsa_mma(o[nt], al[0], al[1], al[2], al[3], b0, b1);
                }
            }
            if (FOLD) {
                const float m0 = sml[gr * 2], l0 = sml[gr * 2 + 1];
                const float m1 = sml[(gr + 8u) * 2], l1 = sml[(gr + 8u) * 2 + 1];
                #pragma unroll
                for (uint32_t nt = 0; nt < 4u; nt++) {
                    float M, L;
                    M = FM[0]; L = FL[0]; qsa_fold(M, L, FA[bb][nt][0], m0, l0, o[nt][0]);
                    M = FM[0]; L = FL[0]; qsa_fold(M, L, FA[bb][nt][1], m0, l0, o[nt][1]);
                    M = FM[1]; L = FL[1]; qsa_fold(M, L, FA[bb][nt][2], m1, l1, o[nt][2]);
                    M = FM[1]; L = FL[1]; qsa_fold(M, L, FA[bb][nt][3], m1, l1, o[nt][3]);
                }
            } else {
                #pragma unroll
                for (uint32_t nt = 0; nt < 4u; nt++) {
                    const uint32_t d = b * 32u + nt * 8u + 2u * c;
                    float *o0 = &part[(slot0 + (uint64_t)gr * QSA_MAX_SPLITS) * PULSAR_QSA_HEAD_DIM + d];
                    *reinterpret_cast<float2 *>(o0) = make_float2(o[nt][0], o[nt][1]);
                    if (row_hi_live) {
                        float *o1 = &part[(slot0 + (uint64_t)(gr + 8u) * QSA_MAX_SPLITS) * PULSAR_QSA_HEAD_DIM + d];
                        *reinterpret_cast<float2 *>(o1) = make_float2(o[nt][2], o[nt][3]);
                    }
                }
            }
        }
        if (FOLD) {                               /* the heads' own (M, L) advance once per split */
            float a0 = 0.f, a1 = 0.f;
            qsa_fold(FM[0], FL[0], a0, sml[gr * 2], sml[gr * 2 + 1], 0.f);
            qsa_fold(FM[1], FL[1], a1, sml[(gr + 8u) * 2], sml[(gr + 8u) * 2 + 1], 0.f);
        } else if (threadIdx.x < QSA_GQA) {
            ml[slot0 + (uint64_t)threadIdx.x * QSA_MAX_SPLITS] = make_float2(sml[threadIdx.x * 2], sml[threadIdx.x * 2 + 1]);
        }
    }
    if constexpr (FOLD) {

        /* ---- the epilogue in place: gate, the f32 tap, and the o_proj E4M3 slot.  A
         * 32-dim MX block of one head lives in the 4 lanes of a quad (8 values each). */
        #pragma unroll
        for (uint32_t hh = 0; hh < 2u; hh++) {
            const bool live = hh == 0u || row_hi_live;   /* quad-uniform; dead quads still shuffle */
            const uint32_t head = g * QSA_GQA + gr + 8u * hh;
            const float *gate = qg + (uint64_t)r * PULSAR_QSA_Q_IN + (uint64_t)(live ? head : 0u) * 2u * PULSAR_QSA_HEAD_DIM
                                + PULSAR_QSA_HEAD_DIM;
            #pragma unroll
            for (uint32_t bb = 0; bb < 2u; bb++) {
                const uint32_t b = 2u * w + bb;
                float y[4][2];
                float amax = 0.f;
                #pragma unroll
                for (uint32_t nt = 0; nt < 4u; nt++) {
                    const uint32_t d = b * 32u + nt * 8u + 2u * c;
                    y[nt][0] = live ? qsa_gated(FA[bb][nt][2 * hh], FL[hh], gate[d]) : 0.f;
                    y[nt][1] = live ? qsa_gated(FA[bb][nt][2 * hh + 1], FL[hh], gate[d + 1u]) : 0.f;
                    amax = fmaxf(amax, fmaxf(fabsf(y[nt][0]), fabsf(y[nt][1])));
                }
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                const int se = pulsar_mx_shared_exp(amax);
                if (!live) continue;
                const uint32_t col0 = head * PULSAR_QSA_HEAD_DIM + b * 32u;
                #pragma unroll
                for (uint32_t nt = 0; nt < 4u; nt++) {
                    const uint32_t col = col0 + nt * 8u + 2u * c;
                    if (out) {
                        out[(uint64_t)r * PULSAR_QSA_OUT_DIM + col] = pulsar_mx_encode(y[nt][0], se);
                        out[(uint64_t)r * PULSAR_QSA_OUT_DIM + col + 1u] = pulsar_mx_encode(y[nt][1], se);
                    }
                    /* L251 / ac69748f: bf16 is what the o_proj reads; the E4M3 slot is optional now. */
                    if (out_bf16) {
                        out_bf16[(uint64_t)r * PULSAR_QSA_OUT_DIM + col] = __float2bfloat16(y[nt][0]);
                        out_bf16[(uint64_t)r * PULSAR_QSA_OUT_DIM + col + 1u] = __float2bfloat16(y[nt][1]);
                    }
                    if (tap) *reinterpret_cast<float2 *>(&tap[(uint64_t)r * PULSAR_QSA_OUT_DIM + col]) = make_float2(y[nt][0], y[nt][1]);
                }
                if (c == 0u && out_scale) out_scale[pulsar_mx_sfoff((int)r, (int)(col0 >> 5), kbp)] = pulsar_mx_scale_byte(se);
            }
        }
    }
}

/* ---- fold + gate + o_proj slot for the split schedule ------------------------ *
 * grid (row in group, query head); thread = dim. */
__global__ void __launch_bounds__(PULSAR_QSA_HEAD_DIM) qsa_combine_kernel(
        const qsa_row *rows, uint32_t row0, const float *part, const float2 *ml, const float *qg,
        __nv_bfloat16 *out_bf16, __nv_fp8_e4m3 *out, unsigned char *out_scale, int kbp, float *tap) {
    const uint32_t rg = blockIdx.x, h = blockIdx.y, d = threadIdx.x, r = row0 + rg;
    const qsa_row row = rows[r];
    const uint32_t ns = (row.n_list + QSA_SPLIT - 1u) / QSA_SPLIT;
    const uint64_t base = ((uint64_t)rg * PULSAR_QSA_N_HEAD + h) * QSA_MAX_SPLITS;
    float M = -INFINITY, L = 0.f, A = 0.f;
    for (uint32_t s = 0; s < ns; s++) {
        const float2 e = ml[base + s];
        qsa_fold(M, L, A, e.x, e.y, part[(base + s) * PULSAR_QSA_HEAD_DIM + d]);
    }
    const float y = qsa_gated(A, L, qg[(uint64_t)r * PULSAR_QSA_Q_IN + (uint64_t)h * 2u * PULSAR_QSA_HEAD_DIM
                                       + PULSAR_QSA_HEAD_DIM + d]);
    const uint32_t col = h * PULSAR_QSA_HEAD_DIM + d;
    if (tap) tap[(uint64_t)r * PULSAR_QSA_OUT_DIM + col] = y;
    if (out) pulsar_mx_emit_block(y, col, r, PULSAR_QSA_OUT_DIM, kbp, out, out_scale);
    if (out_bf16) out_bf16[(uint64_t)r * PULSAR_QSA_OUT_DIM + col] = __float2bfloat16(y);
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
    uint32_t *groups;
    float    *q;
    float    *iq;
    uint32_t *sel;
    float    *scores;
    uint32_t *topk;
    float    *part;
    float2   *ml;
    uint64_t  bytes;
    uint64_t  score_cap;    /* bytes of the scores buffer */
    uint32_t  topk_rows;    /* rows the top-k buffer holds */
};

/* Rows per selection pass for blocks-per-row nb: the scores stay inside the
 * budget, and one row always fits. */
static uint32_t qsa_score_rows(uint32_t n_rows, uint32_t nb) {
    return (uint32_t)std::min<uint64_t>(n_rows, std::max<uint64_t>(1u, QSA_SCORE_BUDGET / ((uint64_t)nb * sizeof(float))));
}

static uint64_t qsa_up(uint64_t x) { return (x + 255u) & ~255ull; }

static qsa_ws qsa_ws_layout(uint8_t *base, uint32_t n_rows, uint32_t max_ctx) {
    qsa_ws w{};
    /* Sized for EVERY call with positions < max_ctx: a pass at nb blocks takes
     * qsa_score_rows(nb) rows, so the scores peak at the widest nb and the
     * top-k rows at the narrowest selecting nb (513). */
    const uint32_t nb_max = max_ctx / PULSAR_QSA_BLOCK;
    const bool selects = nb_max > PULSAR_QSA_TOP_BLOCKS;
    w.score_cap = selects ? std::max<uint64_t>(std::min<uint64_t>((uint64_t)n_rows * nb_max * sizeof(float), QSA_SCORE_BUDGET),
                                               (uint64_t)nb_max * sizeof(float)) : 0u;
    w.topk_rows = selects ? qsa_score_rows(n_rows, PULSAR_QSA_TOP_BLOCKS + 1u) : 0u;
    const uint32_t att_rows = n_rows < QSA_FOLD_ROWS ? n_rows : 0u;   /* partials: the split schedule only */
    uint64_t off = 0;
    auto take = [&](uint64_t bytes) { uint8_t *p = base ? base + off : nullptr; off += qsa_up(bytes); return p; };
    w.rows   = (qsa_row *)take((uint64_t)n_rows * sizeof(qsa_row));
    w.list   = (uint32_t *)take((uint64_t)n_rows * sizeof(uint32_t));
    w.groups = (uint32_t *)take((uint64_t)n_rows * sizeof(uint32_t));
    w.q      = (float *)take((uint64_t)n_rows * PULSAR_QSA_N_HEAD * PULSAR_QSA_HEAD_DIM * sizeof(float));
    w.iq     = (float *)take((uint64_t)n_rows * PULSAR_QSA_IDX_HEADS * PULSAR_QSA_IDX_DIM * sizeof(float));
    w.sel    = (uint32_t *)take((uint64_t)n_rows * PULSAR_QSA_TOP_BLOCKS * sizeof(uint32_t));
    w.scores = (float *)take(w.score_cap);
    w.topk   = (uint32_t *)take((uint64_t)w.topk_rows * PULSAR_QSA_TOP_BLOCKS * sizeof(uint32_t));
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
    if (n_rows > 65535u) return qsa_refuse("more than 65535 rows in one call (the grid's row dimension)");
    if (!layer->q_norm || !layer->k_norm || !layer->idx_q_norm || !layer->idx_k_norm) {
        return qsa_refuse("a norm weight is missing or short");
    }
    const uint64_t R = n_rows;
    if (!io->qg || io->qg->bytes < R * PULSAR_QSA_Q_IN * 4u || !io->k || io->k->bytes < R * PULSAR_QSA_KV_IN * 4u ||
        !io->v || io->v->bytes < R * PULSAR_QSA_KV_IN * 4u || !io->idx || io->idx->bytes < R * PULSAR_QSA_IDX_IN * 4u) {
        return qsa_refuse("a projection input is missing or shorter than n_rows rows");
    }
    if (!io->out_bf16 && !io->out_e4m3) return qsa_refuse("no o_proj output (out_bf16 and out_e4m3 both NULL)");
    if (io->out_e4m3 && (!io->out_scale || io->out_sf_pitch != pulsar_mx_kbp(PULSAR_QSA_OUT_DIM))) {
        return qsa_refuse("an o_proj E4M3 slot needs its scale plane and the 6144-wide KBp");
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
                        "splits of %u listed tokens %s\n", n_rows, n_select, PULSAR_QSA_TOP_BLOCKS, QSA_SPLIT,
                n_rows >= QSA_FOLD_ROWS ? "folded in the CTA" : "in parallel + combine");
    }

    if (!cuda_ok(cudaMemcpyAsync(ws.rows, h_rows.data(), R * sizeof(qsa_row), cudaMemcpyHostToDevice),
                 "qsa rows upload")) return 0;
    if (n_select && !cuda_ok(cudaMemcpyAsync(ws.list, h_list.data(), (uint64_t)n_select * 4u,
                                             cudaMemcpyHostToDevice), "qsa list upload")) return 0;
    qsa_rope_tab tab;
    pulsar_qsa_inv_freq(tab.inv);

    qsa_block_keys_kernel<<<(n_rows + 7u) / 8u, 256>>>(ws.rows, n_rows, (const float *)io->idx->ptr,
                                                       layer->idx_k_norm, tab);
    if (!cuda_ok(cudaGetLastError(), "qsa block keys launch")) return 0;
    qsa_prep_kernel<<<n_rows, 1024>>>(ws.rows, (const float *)io->qg->ptr, (const float *)io->k->ptr,
                                      (const float *)io->v->ptr, (const float *)io->idx->ptr,
                                      layer->q_norm, layer->k_norm,
                                      layer->idx_q_norm, tab, ws.q, ws.iq);
    if (!cuda_ok(cudaGetLastError(), "qsa prep launch")) return 0;

    uint32_t nb_call = 0;
    for (uint32_t r : h_list) nb_call = std::max(nb_call, h_rows[r].nb);
    const uint32_t pass_rows = n_select ? qsa_score_rows(n_select, nb_call) : 0u;
    for (uint32_t g0 = 0; g0 < n_select; g0 += pass_rows) {
        const uint32_t gn = std::min(pass_rows, n_select - g0);
        uint32_t nb_max = 0;
        for (uint32_t i = 0; i < gn; i++) nb_max = std::max(nb_max, h_rows[h_list[g0 + i]].nb);
        /* row groups: consecutive listed rows of one sequence, at most QSA_SCORE_ROWS */
        std::vector<uint32_t> grp;
        for (uint32_t i = 0; i < gn; i++) {
            if (!grp.empty()) {
                const uint32_t f = grp.back() >> 8, c = grp.back() & 0xffu;
                if (c < QSA_SCORE_ROWS && row_seq[h_list[g0 + f]] == row_seq[h_list[g0 + i]]) { grp.back()++; continue; }
            }
            grp.push_back((i << 8) | 1u);
        }
        if (!cuda_ok(cudaMemcpyAsync(ws.groups, grp.data(), grp.size() * 4u, cudaMemcpyHostToDevice), "qsa groups upload")) return 0;
        dim3 grid((nb_max + QSA_SCORE_BLK - 1u) / QSA_SCORE_BLK, (uint32_t)grp.size());
        qsa_score_kernel<<<grid, QSA_SCORE_BLK>>>(ws.rows, ws.list + g0, ws.groups, ws.iq, ws.scores, nb_max);
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

    {
        __nv_fp8_e4m3 *o8 = (__nv_fp8_e4m3 *)io->out_e4m3;
        __nv_bfloat16 *obf = (__nv_bfloat16 *)io->out_bf16;
        unsigned char *osc = (unsigned char *)io->out_scale;
        float *tap = io->tap_out_f32 ? (float *)io->tap_out_f32->ptr : nullptr;
        const float *qg = (const float *)io->qg->ptr;
        if (n_rows >= QSA_FOLD_ROWS) {
            qsa_attn_kernel<true><<<dim3(1, PULSAR_QSA_N_KV, n_rows), 128>>>(ws.rows, 0, ws.sel, ws.q, qg, nullptr,
                                                                            nullptr, obf, o8, osc, io->out_sf_pitch, tap);
            if (!cuda_ok(cudaGetLastError(), "qsa attention (fold) launch")) return 0;
        } else {
            uint32_t ns_max = 0;
            for (uint32_t i = 0; i < n_rows; i++) ns_max = std::max(ns_max, (h_rows[i].n_list + QSA_SPLIT - 1u) / QSA_SPLIT);
            qsa_attn_kernel<false><<<dim3(ns_max, PULSAR_QSA_N_KV, n_rows), 128>>>(ws.rows, 0, ws.sel, ws.q, qg, ws.part,
                                                                                  ws.ml, obf, o8, osc, io->out_sf_pitch, tap);
            if (!cuda_ok(cudaGetLastError(), "qsa attention (split) launch")) return 0;
            qsa_combine_kernel<<<dim3(n_rows, PULSAR_QSA_N_HEAD), PULSAR_QSA_HEAD_DIM>>>(
                    ws.rows, 0, ws.part, ws.ml, qg, obf, o8, osc, io->out_sf_pitch, tap);
            if (!cuda_ok(cudaGetLastError(), "qsa combine launch")) return 0;
        }
    }
    if (io->tap_sel) {
        qsa_sel_tap_kernel<<<n_rows, 128>>>(ws.rows, ws.sel, (uint32_t *)io->tap_sel->ptr, n_rows);
        if (!cuda_ok(cudaGetLastError(), "qsa selection tap launch")) return 0;
    }
    return 1;
}

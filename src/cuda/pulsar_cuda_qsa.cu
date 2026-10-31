/** @file
 *  Qwen3.8-Flash-Next full attention + QSA (quantised sparse attention) kernels. */
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
#include "lib/pulsar_yarn.h"
#include "pulsar_cuda_arch.cuh"

#include <algorithm>

namespace {

/// Listed tokens per attention split (one CTA's K/V tile).
constexpr uint32_t QSA_SPLIT       = 64u;    /* listed tokens per attention split */
/// Query heads per KV head (12).
constexpr uint32_t QSA_GQA         = PULSAR_QSA_N_HEAD / PULSAR_QSA_N_KV;
/// Tokens a selecting row attends through its selected blocks: top blocks x block size (2048).
constexpr uint32_t QSA_BUDGET      = PULSAR_QSA_TOP_BLOCKS * PULSAR_QSA_BLOCK;   /* 2048 */
/// Most tokens any row lists: the budget plus a partial open block (2051).
constexpr uint32_t QSA_MAX_LISTED  = QSA_BUDGET + PULSAR_QSA_BLOCK - 1u;         /* 2051 */
/// Most splits any row has (33): the split schedule's partials are carved per this.
constexpr uint32_t QSA_MAX_SPLITS  = (QSA_MAX_LISTED + QSA_SPLIT - 1u) / QSA_SPLIT;  /* 33 */
/// Bytes of block scores one selection pass may use; qsa_score_rows sizes the pass to it.
constexpr uint64_t QSA_SCORE_BUDGET = 64ull << 20;   /* bytes of scores per selection pass */
/// Block keys per score CTA (= its threads).
constexpr uint32_t QSA_SCORE_BLK   = 128u;   /* blocks per score CTA */
/// Shared-memory pitch of one staged block key, in u32 (bf16 pairs), padded by 4 words against bank conflicts.
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

/* One rank's heads (L266 step 7): tensor parallelism splits the attention by KV head, each of TP ranks
 * owning N_KV / TP KV heads and the GQA query heads that read them; its q / gate / k / v inputs, KV
 * records and o_proj input are its own heads', in the full model's order.  The indexer (block keys,
 * scores, the top-k) is whole on every rank, so the ranks select the same blocks.  TP = 1 is the
 * whole layer. */
/** @tparam TP tensor-parallel ranks (1 or 2) */
template <uint32_t TP> struct qsa_shape {
    /** @var N_HEAD
     *  The rank's query heads.
     *  @var N_KV
     *  The rank's KV heads. */
    static constexpr uint32_t N_HEAD = PULSAR_QSA_N_HEAD / TP, N_KV = PULSAR_QSA_N_KV / TP;
    /** @var Q_IN
     *  Width of the rank's q + gate input row (f32), per head [q 256 | gate 256].
     *  @var KV_IN
     *  Width of the rank's k (or v) input row. */
    static constexpr uint32_t Q_IN = N_HEAD * 2u * PULSAR_QSA_HEAD_DIM, KV_IN = N_KV * PULSAR_QSA_HEAD_DIM;
    /// Width of the rank's o_proj input row.
    static constexpr uint32_t OUT_DIM = N_HEAD * PULSAR_QSA_HEAD_DIM;
    /// Bytes of one token's KV record on this rank: K then V E4M3 rows of every KV head, then their E8M0 scales.
    static constexpr uint32_t KV_TOKEN_BYTES = 2u * N_KV * (PULSAR_QSA_HEAD_DIM + PULSAR_QSA_HEAD_DIM / 32u);
    /// Warps of the prep kernel: one per query head, K head, V head and indexer q head.
    static constexpr uint32_t PREP_WARPS = N_HEAD + 2u * N_KV + PULSAR_QSA_IDX_HEADS;
    static_assert(PULSAR_QSA_N_KV % TP == 0, "whole KV heads per rank");
    static_assert(N_HEAD / N_KV == QSA_GQA, "a rank keeps whole GQA groups");
};
static_assert(qsa_shape<1>::Q_IN == PULSAR_QSA_Q_IN && qsa_shape<1>::KV_IN == PULSAR_QSA_KV_IN &&
              qsa_shape<1>::OUT_DIM == PULSAR_QSA_OUT_DIM && qsa_shape<1>::KV_TOKEN_BYTES == PULSAR_QSA_KV_TOKEN_BYTES &&
              qsa_shape<1>::PREP_WARPS == 32u, "the header's whole-layer sizes");

/* Byte offsets inside one token's record (1056 B at TP 1). */
/** @tparam TP tensor-parallel ranks
 *  @param g KV head
 *  @return byte offset of head g's K data in a token record */
template <uint32_t TP> __host__ __device__ constexpr uint32_t qsa_k_data(uint32_t g) { return g * PULSAR_QSA_HEAD_DIM; }
/** @tparam TP tensor-parallel ranks
 *  @param g KV head
 *  @return byte offset of head g's V data */
template <uint32_t TP> __host__ __device__ constexpr uint32_t qsa_v_data(uint32_t g) {
    return (qsa_shape<TP>::N_KV + g) * PULSAR_QSA_HEAD_DIM;
}
/** @tparam TP tensor-parallel ranks
 *  @param g KV head
 *  @return byte offset of head g's 8 K scale bytes */
template <uint32_t TP> __host__ __device__ constexpr uint32_t qsa_k_scale(uint32_t g) {
    return 2u * qsa_shape<TP>::N_KV * PULSAR_QSA_HEAD_DIM + g * (PULSAR_QSA_HEAD_DIM / 32u);
}
/** @tparam TP tensor-parallel ranks
 *  @param g KV head
 *  @return byte offset of head g's 8 V scale bytes */
template <uint32_t TP> __host__ __device__ constexpr uint32_t qsa_v_scale(uint32_t g) {
    return qsa_k_scale<TP>(qsa_shape<TP>::N_KV + g);
}

enum : uint32_t {
    QSA_ROW_COMPLETES = 1u,   /* pos % 4 == 3: this row finishes block pos / 4 */
    QSA_ROW_STAGE     = 2u,   /* the row's block is still open after this call */
    QSA_ROW_SELECT    = 4u,   /* nb > 512: the row attends to a selection */
};

/// One call row's descriptor, built on the host and uploaded to the workspace (the kernels' only per-row input).
struct qsa_row {
    /// The row's sequence KV cache: [ctx][KV_TOKEN_BYTES] token records.
    uint8_t       *kv;
    /// The sequence's block keys, bf16 [ctx / 4][PULSAR_QSA_IDX_DIM].
    __nv_bfloat16 *bkey;
    /// The sequence's raw indexer keys of its open block, f32 [4][PULSAR_QSA_IDX_DIM] (slot pos % 4).
    float         *stage;
    /// The row's token position.
    uint32_t pos;
    /// Rows of the same sequence before this one in the call.
    uint32_t run_off;   /* rows of the same sequence before this one in the call */
    /// QSA_ROW_* bits.
    uint32_t flags;
    /// Complete indexer blocks visible: (pos + 1) / 4.
    uint32_t nb;        /* complete blocks visible: (pos + 1) / 4 */
    /// Tokens the row attends: pos + 1, or QSA_BUDGET + (pos + 1) % 4 when it selects.
    uint32_t n_list;    /* tokens attended: pos + 1, or 2048 + (pos + 1) % 4 */
    /// Zero.
    uint32_t pad;
    /// L268: the row's (T, H, W) rope positions ((pos, pos, pos) for text).
    uint32_t rope[3];   /* L268: the row's (T, H, W) rope positions -- (pos, pos, pos) for text */
    /// The rope positions of the row's indexer block's first token (pos - 3 for text).
    uint32_t brope[3];  /* ... and its indexer block's first token's (pos - 3 for text) */
};

/// The rope table passed by value to the kernels.
struct qsa_rope_tab { float inv[PULSAR_QSA_ROT_DIM / 2]; float mscale; };

/** @var qsa_rope_tab::inv
 *  Inverse frequency of each of the 32 rope pairs (f32, from pulsar_qsa_rope_init).
 *  @var qsa_rope_tab::mscale
 *  The rope's attention scaling (YaRN mscale; 1 natively). */

/** @param v this lane's value
 *  @return the warp's sum, __fadd_rn xor tree (the same value in every lane) */
__device__ __forceinline__ float qsa_warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
/** @param v this lane's value
 *  @return the warp's maximum */
__device__ __forceinline__ float qsa_warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* cos/sin of the rope angle for pair `i` at the position triple `p` (T, H, W).  L268: the interleaved mRoPE
 * axis of pair i -- H when i % 3 == 1, W when i % 3 == 2, else T (HF apply_interleaved_mrope, mrope_section
 * [11, 11, 10] over 32 pairs); a text row's three axes are equal, so its angle is the 1D rope's.  The angle is the
 * f32 product transformers forms (inv_freq f32 times the position as f32); its cos/sin are taken in double so a
 * large angle is not the fast-math __sinf's, rounded to f32, then times the rope's mscale in f32 (transformers'
 * `freqs.cos() * attention_scaling`; L280 YaRN -- natively 1, and x * 1.0f is x, so the native rope's bytes are
 * the ones it had before the factor existed). */
/** @param tab the rope table
 *  @param p   the position triple (T, H, W)
 *  @param i   the rope pair, 0..31
 *  @param c   out: cos * mscale
 *  @param s   out: sin * mscale */
__device__ __forceinline__ void qsa_rope_cs(const qsa_rope_tab &tab, const uint32_t (&p)[3], uint32_t i,
                                            float *c, float *s) {
    const uint32_t pos = p[i % 3u == 1u ? 1 : i % 3u == 2u ? 2 : 0];
    const float ang = __fmul_rn((float)pos, tab.inv[i]);
    double sd, cd;
    sincos((double)ang, &sd, &cd);
    *c = __fmul_rn((float)cd, tab.mscale);
    *s = __fmul_rn((float)sd, tab.mscale);
}

/* One warp, one head of NJ*32 elements: lane l holds x[l + 32 j].  RMSNorm with
 * weight (1 + w), then RoPE on the pair (x[l], x[l + 32]) -- lane l's j = 0, 1. */
/** @tparam NJ elements per lane (head width / 32)
 *  @param x the head row, lane l's x[j] = element l + 32 j; normed and roped in place
 *  @param w bf16 [NJ * 32] norm weight, applied as (1 + w)
 *  @param c this lane's pair cos (qsa_rope_cs)
 *  @param s this lane's pair sin */
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
/** One warp: encode a 256-wide head row as 8 MX32 E4M3 blocks.
 *  @param x      lane l's x[j] = element l + 32 j
 *  @param data   out: 256 E4M3 bytes
 *  @param scales out: 8 E8M0 bytes (lane 0 writes) */
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
/** One warp per row (8 rows a 256-thread block): rows that complete a block mean-pool its 4 raw keys, norm + rope at
 *  the block start, and store the bf16 block key.  Grid = ceil(n_rows / 8), block = 256.
 *  @param rows   the call's row descriptors
 *  @param n_rows rows in the call
 *  @param idx    f32 [n_rows][PULSAR_QSA_IDX_IN] indexer input (q heads then the raw key)
 *  @param kw     bf16 [PULSAR_QSA_IDX_DIM] indexer key norm weight
 *  @param tab    the rope table */
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
    qsa_rope_cs(tab, row.brope, (uint32_t)lane, &c, &s);
    qsa_norm_rope<4>(x, kw, c, s);
    __nv_bfloat16 *dst = row.bkey + (uint64_t)(row.pos / PULSAR_QSA_BLOCK) * PULSAR_QSA_IDX_DIM;
    #pragma unroll
    for (int j = 0; j < 4; j++) dst[lane + 32 * j] = __float2bfloat16_rn(x[j]);
}

/* ---- prep: 32 warps per row ------------------------------------------------- *
 * warps 0..23 query heads, 24..25 K heads, 26..27 V heads, 28..31 indexer q heads
 * (warp 28 also stages the raw indexer key). */
/** One CTA per row, 32 * PREP_WARPS threads (one warp per head; see the comment above).
 *  @tparam TP tensor-parallel ranks
 *  @param rows   the call's row descriptors
 *  @param qg     f32 [rows][Q_IN] q + gate projections
 *  @param kin    f32 [rows][KV_IN] k projection
 *  @param vin    f32 [rows][KV_IN] v projection
 *  @param idx    f32 [rows][PULSAR_QSA_IDX_IN] indexer projection
 *  @param qw     bf16 [256] q norm weight
 *  @param kw     bf16 [256] k norm weight
 *  @param iqw    bf16 [PULSAR_QSA_IDX_DIM] indexer q norm weight
 *  @param tab    the rope table
 *  @param q_out  out, f32 [rows][N_HEAD][256] normed, roped q
 *  @param iq_out out, f32 [rows][PULSAR_QSA_IDX_HEADS][PULSAR_QSA_IDX_DIM] normed, roped indexer q */
template <uint32_t TP>
__global__ void __launch_bounds__(1024) qsa_prep_kernel(
        const qsa_row *rows, const float *qg, const float *kin, const float *vin, const float *idx,
        const uint16_t *qw, const uint16_t *kw, const uint16_t *iqw, qsa_rope_tab tab,
        float *q_out, float *iq_out) {
    using Qs = qsa_shape<TP>;
    const uint32_t r = blockIdx.x;
    const qsa_row row = rows[r];
    const uint32_t w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    float c, s;
    qsa_rope_cs(tab, row.rope, (uint32_t)lane, &c, &s);
    if (w < Qs::N_HEAD) {
        const float *src = qg + (uint64_t)r * Qs::Q_IN + (uint64_t)w * 2u * PULSAR_QSA_HEAD_DIM;
        float x[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) x[j] = src[lane + 32 * j];
        qsa_norm_rope<8>(x, qw, c, s);
        float *dst = q_out + ((uint64_t)r * Qs::N_HEAD + w) * PULSAR_QSA_HEAD_DIM;
        #pragma unroll
        for (int j = 0; j < 8; j++) dst[lane + 32 * j] = x[j];
    } else if (w < Qs::N_HEAD + 2u * Qs::N_KV) {
        const uint32_t g = (w - Qs::N_HEAD) % Qs::N_KV;
        const bool is_k = w < Qs::N_HEAD + Qs::N_KV;
        const float *src = (is_k ? kin : vin) + (uint64_t)r * Qs::KV_IN + (uint64_t)g * PULSAR_QSA_HEAD_DIM;
        float x[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) x[j] = src[lane + 32 * j];
        if (is_k) qsa_norm_rope<8>(x, kw, c, s);
        uint8_t *rec = row.kv + (uint64_t)row.pos * Qs::KV_TOKEN_BYTES;
        qsa_emit_row(x, rec + (is_k ? qsa_k_data<TP>(g) : qsa_v_data<TP>(g)),
                     rec + (is_k ? qsa_k_scale<TP>(g) : qsa_v_scale<TP>(g)));
    } else {
        const uint32_t h = w - (Qs::N_HEAD + 2u * Qs::N_KV);
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
/// Most listed rows of one sequence a score CTA serves (a group).
constexpr uint32_t QSA_SCORE_ROWS = 4u;

/** @param qs the row's indexer q, f32 [PULSAR_QSA_IDX_HEADS][PULSAR_QSA_IDX_DIM] (shared)
 *  @param kr one block key, bf16 pairs (shared)
 *  @return sum over heads in order of relu(q_h . k), each dot a sequential fmaf chain */
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
/** Grid = (ceil(nb_max / QSA_SCORE_BLK), groups), block = QSA_SCORE_BLK (one thread per block key); static smem
 *  the group's q rows plus QSA_SCORE_BLK staged keys.
 *  @param rows   the call's row descriptors
 *  @param list   the pass's selecting rows (indices into rows)
 *  @param groups per group: (first listed index << 8) | row count
 *  @param iq     f32 [rows][PULSAR_QSA_IDX_HEADS][PULSAR_QSA_IDX_DIM] indexer q
 *  @param scores out, f32 [pass rows][nb_max]; -inf past a row's nb
 *  @param nb_max the pass's score row pitch (its widest nb) */
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
/** One CTA (256 threads) per listed row: bitonic-sort its PULSAR_QSA_TOP_BLOCKS selected block indices ascending.
 *  @param list the pass's selecting rows
 *  @param topk uint32 [pass rows][PULSAR_QSA_TOP_BLOCKS] the top-k's picks
 *  @param sel  out, uint32 [rows][PULSAR_QSA_TOP_BLOCKS] (indexed by the row) */
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
/** @param row the row
 *  @param sel the row's sorted selection
 *  @param j   listed position, below row.n_list
 *  @return the token position attended at j */
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
/// Bytes per staged K/V row in smem (256 + 16 padding).
constexpr uint32_t QSA_TPITCH = PULSAR_QSA_HEAD_DIM + 16u;   /* bytes per smem K/V row */
/// Floats per staged score row (64 + 4 padding).
constexpr uint32_t QSA_SPITCH = QSA_SPLIT + 4u;              /* floats per score row */
/// Calls of at least this many rows take the in-CTA fold schedule (no partials).
constexpr uint32_t QSA_FOLD_ROWS = 64u;                      /* calls at least this wide fold in the CTA */
static_assert(QSA_SPLIT == 64u, "the warp layout below tiles 64 tokens: 4 warps x 2 n-tiles (QK), 4 k-steps (PV)");

/** One mma.sync m16n8k16 f16 x f16 -> f32, accumulating into d (sm_80+; traps on older arches).
 *  @param d  the accumulator fragment (in/out)
 *  @param a0 A fragment register 0
 *  @param a1 A fragment register 1
 *  @param a2 A fragment register 2
 *  @param a3 A fragment register 3
 *  @param b0 B fragment register 0
 *  @param b1 B fragment register 1 */
__device__ __forceinline__ void qsa_mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
#if PULSAR_SM80
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}
/* f32 pair -> (hi, lo) packed f16x2 words: hi = f16(x), lo = f16(x - hi). */
/** @param x0 first value
 *  @param x1 second value
 *  @param hi out: f16x2 (f16(x0), f16(x1))
 *  @param lo out: f16x2 of the residuals x - hi */
__device__ __forceinline__ void qsa_split2(float x0, float x1, uint32_t &hi, uint32_t &lo) {
    const __half h0 = __float2half_rn(x0), h1 = __float2half_rn(x1);
    const __half l0 = __float2half_rn(__fsub_rn(x0, __half2float(h0)));
    const __half l1 = __float2half_rn(__fsub_rn(x1, __half2float(h1)));
    const __half2 H = __halves2half2(h0, h1), L = __halves2half2(l0, l1);
    hi = *reinterpret_cast<const uint32_t *>(&H);
    lo = *reinterpret_cast<const uint32_t *>(&L);
}
/* two E4M3 bytes (low byte = first element) -> packed f16x2, exact */
/** @param two two E4M3 codes, the first in the low byte
 *  @return the f16x2 of their values (exact) */
__device__ __forceinline__ uint32_t qsa_e4m3x2(uint16_t two) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)two, __NV_E4M3);
    uint32_t u;
    memcpy(&u, &h, 4);
    return u;
}

/* The fold of one split (m, l, a) into a row's running (M, L, A): THE recurrence. */
/** @param M running max (log2 domain), in/out
 *  @param L running denominator, in/out
 *  @param A running numerator, in/out
 *  @param m the split's max
 *  @param l the split's denominator
 *  @param a the split's numerator */
__device__ __forceinline__ void qsa_fold(float &M, float &L, float &A, float m, float l, float a) {
    const float mn = fmaxf(M, m);
    const float c1 = exp2f(__fsub_rn(M, mn)), c2 = exp2f(__fsub_rn(m, mn));
    L = __fadd_rn(__fmul_rn(L, c1), __fmul_rn(l, c2));
    A = __fadd_rn(__fmul_rn(A, c1), __fmul_rn(a, c2));
    M = mn;
}
/* The epilogue: normalise, times the sigmoid output gate. */
/** @param A the folded numerator
 *  @param L the folded denominator
 *  @param gate the head's gate logit
 *  @return (A / L) * sigmoid(gate) */
__device__ __forceinline__ float qsa_gated(float A, float L, float gate) {
    return __fmul_rn(__fdiv_rn(A, L), __fdiv_rn(1.0f, __fadd_rn(1.0f, __expf(-gate))));
}

/* One K or V tile (64 listed tokens, KV head g) into smem as raw E4M3 plus its
 * 8 scales per token; rows past n are zero (their scores are masked). */
/** 128 threads, 2 per token.
 *  @tparam TP tensor-parallel ranks
 *  @param td        out, smem [64][QSA_TPITCH] E4M3 rows
 *  @param tsc       out, smem [64][8] f32 block scales
 *  @param kv        the sequence's KV records
 *  @param toks      smem [64] token positions of the split
 *  @param n         live tokens in the split
 *  @param data_off  record offset of the head's data (qsa_k_data / qsa_v_data)
 *  @param scale_off record offset of its scales */
template <uint32_t TP>
__device__ __forceinline__ void qsa_load_tile(uint8_t *td, float *tsc, const uint8_t *kv, const uint32_t *toks,
                                              uint32_t n, uint32_t data_off, uint32_t scale_off) {
    using Qs = qsa_shape<TP>;
    const uint32_t t = threadIdx.x >> 1, half = threadIdx.x & 1u;     /* 128 threads: 2 per token */
    uint4 *dst = reinterpret_cast<uint4 *>(td + t * QSA_TPITCH + half * 128u);
    if (t >= n) {
        #pragma unroll
        for (int i = 0; i < 8; i++) dst[i] = make_uint4(0u, 0u, 0u, 0u);
        if (!half) for (int b = 0; b < 8; b++) tsc[t * 8u + b] = 0.f;
        return;
    }
    const uint8_t *rec = kv + (uint64_t)toks[t] * Qs::KV_TOKEN_BYTES;
    const uint4 *src = reinterpret_cast<const uint4 *>(rec + data_off + half * 128u);
    #pragma unroll
    for (int i = 0; i < 8; i++) dst[i] = src[i];
    if (!half) for (int b = 0; b < 8; b++) tsc[t * 8u + b] = exp2f((float)((int)rec[scale_off + b] - 127));
}

/** Block = 128 (4 warps), static smem (q, K/V tile, scales, scores).  Grid (split, KV head, row) without FOLD,
 *  (1, KV head, row) with it; see the schedule comment above.
 *  @tparam FOLD fold the row's splits in the CTA and write the output (true), or write the split's partial
 *  @tparam TP   tensor-parallel ranks
 *  @param rows      the call's row descriptors
 *  @param row0      first row of this launch (blockIdx.z is relative to it)
 *  @param sel       uint32 [rows][PULSAR_QSA_TOP_BLOCKS] sorted selections
 *  @param q         f32 [rows][N_HEAD][256] prepped q
 *  @param qg        f32 [rows][Q_IN] q + gate (the gate half read)
 *  @param part      out (split schedule), f32 [rows][N_HEAD][QSA_MAX_SPLITS][256] unnormalised partials
 *  @param ml        out (split schedule), [rows][N_HEAD][QSA_MAX_SPLITS] (max, denominator)
 *  @param out_bf16  out (FOLD), bf16 [rows][OUT_DIM] o_proj input, or NULL
 *  @param out       out (FOLD), E4M3 [rows][OUT_DIM] o_proj slot, or NULL
 *  @param out_scale out (FOLD), its E8M0 slab, or NULL
 *  @param kbp       pulsar_mx_kbp(OUT_DIM) of out_scale
 *  @param tap       out (FOLD), f32 [rows][OUT_DIM] debug tap, or NULL */
template <bool FOLD, uint32_t TP>
__global__ void __launch_bounds__(128) qsa_attn_kernel(
        const qsa_row *rows, uint32_t row0, const uint32_t *sel, const float *q, const float *qg,
        float *part, float2 *ml, __nv_bfloat16 *out_bf16, __nv_fp8_e4m3 *out, unsigned char *out_scale,
        int kbp, float *tap) {
    using Qs = qsa_shape<TP>;
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
    const float *qsrc = q + ((uint64_t)r * Qs::N_HEAD + g * QSA_GQA) * PULSAR_QSA_HEAD_DIM;
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
        qsa_load_tile<TP>(td, tsc, row.kv, toks, n, qsa_k_data<TP>(g), qsa_k_scale<TP>(g));
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
        qsa_load_tile<TP>(td, tsc, row.kv, toks, n, qsa_v_data<TP>(g), qsa_v_scale<TP>(g));

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
        const uint64_t slot0 = ((uint64_t)rg * Qs::N_HEAD + g * QSA_GQA) * QSA_MAX_SPLITS + s;
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
            const float *gate = qg + (uint64_t)r * Qs::Q_IN + (uint64_t)(live ? head : 0u) * 2u * PULSAR_QSA_HEAD_DIM
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
                        out[(uint64_t)r * Qs::OUT_DIM + col] = pulsar_mx_encode(y[nt][0], se);
                        out[(uint64_t)r * Qs::OUT_DIM + col + 1u] = pulsar_mx_encode(y[nt][1], se);
                    }
                    /* L251 / ac69748f: bf16 is what the o_proj reads; the E4M3 slot is optional now. */
                    if (out_bf16) {
                        out_bf16[(uint64_t)r * Qs::OUT_DIM + col] = __float2bfloat16(y[nt][0]);
                        out_bf16[(uint64_t)r * Qs::OUT_DIM + col + 1u] = __float2bfloat16(y[nt][1]);
                    }
                    if (tap) *reinterpret_cast<float2 *>(&tap[(uint64_t)r * Qs::OUT_DIM + col]) = make_float2(y[nt][0], y[nt][1]);
                }
                if (c == 0u && out_scale) out_scale[pulsar_mx_sfoff((int)r, (int)(col0 >> 5), kbp)] = pulsar_mx_scale_byte(se);
            }
        }
    }
}

/* ---- fold + gate + o_proj slot for the split schedule ------------------------ *
 * grid (row in group, query head); thread = dim. */
/** Grid = (rows, N_HEAD), block = 256 (thread = dim).
 *  @tparam TP tensor-parallel ranks
 *  @param rows      the call's row descriptors
 *  @param row0      first row of this launch
 *  @param part      f32 split partials (qsa_attn_kernel without FOLD)
 *  @param ml        the splits' (max, denominator)
 *  @param qg        f32 [rows][Q_IN] q + gate
 *  @param out_bf16  out, bf16 [rows][OUT_DIM], or NULL
 *  @param out       out, E4M3 [rows][OUT_DIM], or NULL
 *  @param out_scale out, its E8M0 slab
 *  @param kbp       pulsar_mx_kbp(OUT_DIM)
 *  @param tap       out, f32 [rows][OUT_DIM], or NULL */
template <uint32_t TP>
__global__ void __launch_bounds__(PULSAR_QSA_HEAD_DIM) qsa_combine_kernel(
        const qsa_row *rows, uint32_t row0, const float *part, const float2 *ml, const float *qg,
        __nv_bfloat16 *out_bf16, __nv_fp8_e4m3 *out, unsigned char *out_scale, int kbp, float *tap) {
    using Qs = qsa_shape<TP>;
    const uint32_t rg = blockIdx.x, h = blockIdx.y, d = threadIdx.x, r = row0 + rg;
    const qsa_row row = rows[r];
    const uint32_t ns = (row.n_list + QSA_SPLIT - 1u) / QSA_SPLIT;
    const uint64_t base = ((uint64_t)rg * Qs::N_HEAD + h) * QSA_MAX_SPLITS;
    float M = -INFINITY, L = 0.f, A = 0.f;
    for (uint32_t s = 0; s < ns; s++) {
        const float2 e = ml[base + s];
        qsa_fold(M, L, A, e.x, e.y, part[(base + s) * PULSAR_QSA_HEAD_DIM + d]);
    }
    const float y = qsa_gated(A, L, qg[(uint64_t)r * Qs::Q_IN + (uint64_t)h * 2u * PULSAR_QSA_HEAD_DIM
                                       + PULSAR_QSA_HEAD_DIM + d]);
    const uint32_t col = h * PULSAR_QSA_HEAD_DIM + d;
    if (tap) tap[(uint64_t)r * Qs::OUT_DIM + col] = y;
    if (out) pulsar_mx_emit_block(y, col, r, Qs::OUT_DIM, kbp, out, out_scale);
    if (out_bf16) out_bf16[(uint64_t)r * Qs::OUT_DIM + col] = __float2bfloat16(y);
}

/** Copy each row's selection to the tap (all ones for a row that attends everything).  Grid = n_rows, any block.
 *  @param rows   the call's row descriptors
 *  @param sel    uint32 [rows][PULSAR_QSA_TOP_BLOCKS]
 *  @param tap    out, uint32 [rows][PULSAR_QSA_TOP_BLOCKS]
 *  @param n_rows rows */
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
/// The call's carved workspace (qsa_ws_layout).
struct qsa_ws {
    /// [n_rows] row descriptors.
    qsa_row  *rows;
    /// [n_rows] the selecting rows' indices.
    uint32_t *list;
    /// [n_rows] score groups ((first << 8) | count).
    uint32_t *groups;
    /// f32 [n_rows][N_HEAD][256] prepped q.
    float    *q;
    /// f32 [n_rows][IDX_HEADS][IDX_DIM] prepped indexer q.
    float    *iq;
    /// [n_rows][TOP_BLOCKS] sorted selections.
    uint32_t *sel;
    /// f32 block scores of one pass (score_cap bytes).
    float    *scores;
    /// [topk_rows][TOP_BLOCKS] the top-k's picks for one pass.
    uint32_t *topk;
    /// f32 split partials (calls under QSA_FOLD_ROWS rows only).
    float    *part;
    /// The partials' (max, denominator).
    float2   *ml;
    /// Total bytes of the layout.
    uint64_t  bytes;
    /// Bytes of the scores buffer.
    uint64_t  score_cap;    /* bytes of the scores buffer */
    /// Rows the top-k buffer holds.
    uint32_t  topk_rows;    /* rows the top-k buffer holds */
};

/* Rows per selection pass for blocks-per-row nb: the scores stay inside the
 * budget, and one row always fits. */
/** @param n_rows selecting rows
 *  @param nb     blocks per row
 *  @return rows per selection pass */
static uint32_t qsa_score_rows(uint32_t n_rows, uint32_t nb) {
    return (uint32_t)std::min<uint64_t>(n_rows, std::max<uint64_t>(1u, QSA_SCORE_BUDGET / ((uint64_t)nb * sizeof(float))));
}

/** @param x bytes
 *  @return x rounded up to 256 */
static uint64_t qsa_up(uint64_t x) { return (x + 255u) & ~255ull; }

/** Size (base NULL) or carve the workspace for any call of at most n_rows rows below max_ctx.
 *  @param base    workspace start, or nullptr
 *  @param n_rows  rows
 *  @param max_ctx context bound
 *  @return the layout (pointers null when sizing; bytes set) */
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

/** @param why the reason printed
 *  @return 0 (the API's failure value) */
static int qsa_refuse(const char *why) {
    fprintf(stderr, "pulsar: qsa attention refused: %s\n", why);
    return 0;
}

/// Once-per-shape announcement state of the attention launch.
static pulsar_shape_once g_qsa_announce;

}  // namespace

uint32_t pulsar_qsa_kv_token_bytes(int tp_ranks) {
    return tp_ranks == 2 ? qsa_shape<2>::KV_TOKEN_BYTES : qsa_shape<1>::KV_TOKEN_BYTES;
}

/* transformers' rope init (the default rope, or _compute_yarn_parameters) in double, each result rounded ONCE to f32:
 * pos_freqs = theta^(2i/64); extrapolation = 1 / pos_freqs; interpolation = 1 / (factor * pos_freqs); blended by
 * each pair's extrapolation weight; attention factor 1 + 0.1 ln(factor).  Host code of a .cu TU: IEEE (nvcc's
 * --use_fast_math is the device's), so pow and log are libm's double functions. */
int pulsar_qsa_rope_init(pulsar_qsa_rope *out, const pulsar_qsa_yarn *yarn) {
    if (!out) return qsa_refuse("pulsar_qsa_rope_init: no output");
    constexpr uint32_t D = PULSAR_QSA_ROT_DIM;
    double low = 0.0, high = 0.0;
    if (yarn) {
        if (!(yarn->factor > 1.0) || !isfinite(yarn->factor) || yarn->orig_ctx == 0 || !(yarn->beta_slow > 0.0) ||
            !(yarn->beta_fast > yarn->beta_slow) || !isfinite(yarn->beta_fast)) {
            return qsa_refuse("YaRN needs factor > 1, an original context, and beta_fast > beta_slow > 0");
        }
        pulsar_yarn_corr_range<double>(D, yarn->orig_ctx, PULSAR_QSA_ROPE_THETA, yarn->beta_fast, yarn->beta_slow,
                                       &low, &high);
        if (!(high > low)) return qsa_refuse("the YaRN ramp's pair range is empty at this original context");
    }
    for (uint32_t i = 0; i < D / 2; i++) {
        const double pos_freq = pow(PULSAR_QSA_ROPE_THETA, (double)(2u * i) / (double)D);
        const double extrap = 1.0 / pos_freq;
        out->inv_freq[i] = (float)(yarn ? pulsar_yarn_blend<double>(1.0 / (yarn->factor * pos_freq), extrap,
                                                                     pulsar_yarn_extrap_weight<double>(low, high, (double)i))
                                        : extrap);
    }
    out->mscale = yarn ? (float)pulsar_yarn_mscale<double>(yarn->factor) : 1.0f;
    return 1;
}

uint64_t pulsar_gpu_qsa_workspace_bytes(uint32_t n_rows, uint32_t max_ctx) {
    /* Sized for EVERY call of at most n_rows rows, not only an n_rows one: a call of fewer than
     * QSA_FOLD_ROWS rows takes the split schedule and needs its partials, which an n_rows >=
     * QSA_FOLD_ROWS layout does not carve.  (A 96-row session refused its 24-row prompt: 2.76 MB
     * sized, 20.3 MB needed.  Larger contexts hid it -- their score budget covered the partials.) */
    const uint64_t at_max = qsa_ws_layout(nullptr, n_rows, max_ctx).bytes;
    if (n_rows < QSA_FOLD_ROWS) return at_max;
    const uint64_t split = qsa_ws_layout(nullptr, QSA_FOLD_ROWS - 1u, max_ctx).bytes;
    return at_max > split ? at_max : split;
}

int pulsar_gpu_qsa_forward(const pulsar_qsa_layer *layer,
                           const pulsar_qsa_seq *seqs, uint32_t n_seqs,
                           const uint32_t *row_seq, const uint32_t *row_pos, uint32_t n_rows,
                           const pulsar_qsa_io *io, pulsar_gpu_tensor *workspace) {
    PULSAR_REQUIRE_CC(80, "QSA attention (bf16 m16n8k16)", 0);   /* L321: refuse below the lane's arch, before its kernels trap */
    if (!layer || !seqs || !row_seq || !row_pos || !io || !workspace || n_rows == 0 || n_seqs == 0) {
        return qsa_refuse("a NULL argument or zero rows");
    }
    if (n_rows > 65535u) return qsa_refuse("more than 65535 rows in one call (the grid's row dimension)");
    if (!layer->q_norm || !layer->k_norm || !layer->idx_q_norm || !layer->idx_k_norm) {
        return qsa_refuse("a norm weight is missing or short");
    }
    const uint64_t R = n_rows;
    if (io->tp_ranks != 0 && io->tp_ranks != 1 && io->tp_ranks != 2) return qsa_refuse("tp_ranks is not 1 or 2");
    const bool tp2 = io->tp_ranks == 2;
    const uint64_t Q_IN = tp2 ? qsa_shape<2>::Q_IN : qsa_shape<1>::Q_IN, KV_IN = tp2 ? qsa_shape<2>::KV_IN : qsa_shape<1>::KV_IN;
    const uint64_t OUT_DIM = tp2 ? qsa_shape<2>::OUT_DIM : qsa_shape<1>::OUT_DIM;
    const uint64_t KV_TOKEN = tp2 ? qsa_shape<2>::KV_TOKEN_BYTES : qsa_shape<1>::KV_TOKEN_BYTES;
    if (!io->qg || io->qg->bytes < R * Q_IN * 4u || !io->k || io->k->bytes < R * KV_IN * 4u ||
        !io->v || io->v->bytes < R * KV_IN * 4u || !io->idx || io->idx->bytes < R * PULSAR_QSA_IDX_IN * 4u) {
        return qsa_refuse("a projection input is missing or shorter than n_rows rows");
    }
    if (!io->out_bf16 && !io->out_e4m3) return qsa_refuse("no o_proj output (out_bf16 and out_e4m3 both NULL)");
    if (!io->rope) return qsa_refuse("no rope (pulsar_qsa_io::rope; pulsar_qsa_rope_init makes it)");
    if (io->out_e4m3 && (!io->out_scale || io->out_sf_pitch != pulsar_mx_kbp((uint32_t)OUT_DIM))) {
        return qsa_refuse("an o_proj E4M3 slot needs its scale plane and the rank's o_proj-wide KBp");
    }
    if ((io->tap_out_f32 && io->tap_out_f32->bytes < R * OUT_DIM * 4u) ||
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
            S.kv->bytes < (uint64_t)S.cap * KV_TOKEN ||
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
        /* L268: the rope positions -- the caller's mRoPE triples, or the text position on every axis */
        const uint32_t bstart = p >= PULSAR_QSA_BLOCK - 1u ? p - (PULSAR_QSA_BLOCK - 1u) : 0u;
        for (int a = 0; a < 3; a++) {
            d.rope[a] = io->row_rope ? io->row_rope[(uint64_t)r * 6u + (uint64_t)a] : p;
            d.brope[a] = io->row_rope ? io->row_rope[(uint64_t)r * 6u + 3u + (uint64_t)a] : bstart;
        }
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
                        "splits of %u listed tokens %s, rope mscale %.9g\n", n_rows, n_select,
                PULSAR_QSA_TOP_BLOCKS, QSA_SPLIT, n_rows >= QSA_FOLD_ROWS ? "folded in the CTA" : "in parallel + combine",
                (double)io->rope->mscale);
    }

    if (!cuda_ok(cudaMemcpyAsync(ws.rows, h_rows.data(), R * sizeof(qsa_row), cudaMemcpyHostToDevice),
                 "qsa rows upload")) return 0;
    if (n_select && !cuda_ok(cudaMemcpyAsync(ws.list, h_list.data(), (uint64_t)n_select * 4u,
                                             cudaMemcpyHostToDevice), "qsa list upload")) return 0;
    qsa_rope_tab tab;
    memcpy(tab.inv, io->rope->inv_freq, sizeof(tab.inv));
    tab.mscale = io->rope->mscale;

    qsa_block_keys_kernel<<<(n_rows + 7u) / 8u, 256>>>(ws.rows, n_rows, (const float *)io->idx->ptr,
                                                       layer->idx_k_norm, tab);
    if (!cuda_ok(cudaGetLastError(), "qsa block keys launch")) return 0;
    if (io->tp_ranks == 2)
        qsa_prep_kernel<2><<<n_rows, 32u * qsa_shape<2>::PREP_WARPS>>>(ws.rows, (const float *)io->qg->ptr,
                (const float *)io->k->ptr, (const float *)io->v->ptr, (const float *)io->idx->ptr,
                layer->q_norm, layer->k_norm, layer->idx_q_norm, tab, ws.q, ws.iq);
    else
        qsa_prep_kernel<1><<<n_rows, 32u * qsa_shape<1>::PREP_WARPS>>>(ws.rows, (const float *)io->qg->ptr,
                (const float *)io->k->ptr, (const float *)io->v->ptr, (const float *)io->idx->ptr,
                layer->q_norm, layer->k_norm, layer->idx_q_norm, tab, ws.q, ws.iq);
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
        const bool tp2 = io->tp_ranks == 2;
        if (n_rows >= QSA_FOLD_ROWS) {
            if (tp2) qsa_attn_kernel<true, 2><<<dim3(1, qsa_shape<2>::N_KV, n_rows), 128>>>(ws.rows, 0, ws.sel, ws.q, qg,
                         nullptr, nullptr, obf, o8, osc, io->out_sf_pitch, tap);
            else     qsa_attn_kernel<true, 1><<<dim3(1, qsa_shape<1>::N_KV, n_rows), 128>>>(ws.rows, 0, ws.sel, ws.q, qg,
                         nullptr, nullptr, obf, o8, osc, io->out_sf_pitch, tap);
            if (!cuda_ok(cudaGetLastError(), "qsa attention (fold) launch")) return 0;
        } else {
            uint32_t ns_max = 0;
            for (uint32_t i = 0; i < n_rows; i++) ns_max = std::max(ns_max, (h_rows[i].n_list + QSA_SPLIT - 1u) / QSA_SPLIT);
            if (tp2) qsa_attn_kernel<false, 2><<<dim3(ns_max, qsa_shape<2>::N_KV, n_rows), 128>>>(ws.rows, 0, ws.sel, ws.q,
                         qg, ws.part, ws.ml, obf, o8, osc, io->out_sf_pitch, tap);
            else     qsa_attn_kernel<false, 1><<<dim3(ns_max, qsa_shape<1>::N_KV, n_rows), 128>>>(ws.rows, 0, ws.sel, ws.q,
                         qg, ws.part, ws.ml, obf, o8, osc, io->out_sf_pitch, tap);
            if (!cuda_ok(cudaGetLastError(), "qsa attention (split) launch")) return 0;
            if (tp2) qsa_combine_kernel<2><<<dim3(n_rows, qsa_shape<2>::N_HEAD), PULSAR_QSA_HEAD_DIM>>>(
                         ws.rows, 0, ws.part, ws.ml, qg, obf, o8, osc, io->out_sf_pitch, tap);
            else     qsa_combine_kernel<1><<<dim3(n_rows, qsa_shape<1>::N_HEAD), PULSAR_QSA_HEAD_DIM>>>(
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

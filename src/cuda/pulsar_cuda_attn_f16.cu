/* Attention on the tensor cores, fp16 operands, f32 accumulate.
 *
 * WHY fp16 AND NOT SOMETHING SMALLER.  Measured, not assumed
 * (harness tests/attn_precision_fidelity.cc, real activations, f64 reference;
 *  the harness itself was deleted in a71e346 -- these numbers are its record):
 *   format  rate          top-1 attention position preserved
 *   f32     14.5 TMAC/s   100%      <- what the FMA-pipe kernel does today
 *   fp16    62.9 (4.3x)   ~100%     <- this kernel
 *   bf16    62.9 (4.3x)   99.87%    <- same speed, 3-8x the error: dominated
 *   fp8     125.0 (8.6x)  96.8%     <- worst single head off by 100%
 * bf16 is strictly dominated by fp16 -- identical throughput for more error --
 * so the only real question was fp16 vs fp8, and fp8 moves where 2-3% of
 * head-token pairs attend.  Nothing saturates in any format (max|q| 17.6,
 * max|kv| 5.65), so this is a resolution choice, not a range one.
 *
 * WHAT THIS REPLACED.  The f32 online-softmax kernels (the last of them, the
 * decode kernel, deleted in L166 -- this kernel is now the ONE decode/prefill
 * attention kernel for every production shape) ran at pipe_tensor 0%,
 * pipe_fma 40%, pipe_lsu 56%: one warp per head, one KV row at a time, and a
 * 32-lane shuffle reduction to produce ONE score out of 16 FMAs of useful
 * work -- while all 8 warps in the block re-read the same 2 KB latent row out
 * of shared memory.  Both costs are structural, and both disappear when the
 * reduction becomes the MMA's k dimension and the operands are reused across
 * the MMA's m dimension.
 *
 * MLA, so K AND V ARE THE SAME TENSOR.  The launcher passes raw_kv as both, and
 * there is one latent 512-wide vector per position.  The tile is therefore
 * staged ONCE and used in both orientations: as the B operand transposed for
 * the scores, and as the B operand directly for the value sum.
 *
 * DECOMPOSITION.  Block = 16 heads (the MMA's M) x 8 warps.
 *   phase 1  S[16][16] = Q[16][512] . KV[16][512]^T
 *            2 n-tiles x 32 k-steps = 64 MMAs.  With 8 warps that is 2 n-tiles
 *            x a 4-way split of k, so each warp owns 8 k-steps and the four
 *            partials are summed through smem.  Splitting k (not n) is what
 *            lets every warp hold its Q fragments in REGISTERS for the whole
 *            kernel -- 8 k-steps x 4 regs = 32 -- since Q never changes.
 *   phase 2  online softmax over the tile, f32, per head.
 *   phase 3  O[16][512] += P[16][16] . KV[16][512]
 *            warp w owns dims 64w..64w+63: 8 n-tiles x 1 k-step = 8 MMAs, and
 *            its accumulator is 16 heads x 64 dims = 32 regs/lane.
 *
 * The output accumulator is what sets the shape.  O is heads x 512 in f32, so
 * splitting the 512 across the 8 warps is the only way it fits in registers at
 * all: 16x512/(8x32) = 32 per lane.  Everything else follows from that.
 *
 * Fragment layout is the one verified in tests/attn_mma_probe.cu, not the one
 * remembered from the ISA doc -- the block-scaled work turned up two layout
 * details that were silent-wrong-answer traps, and a layout bug here produces
 * plausible attention, not an error.
 */
#include "pulsar_cuda_internal.h"
#include "pulsar_cuda_mx.cuh"
#include "pulsar_cuda_rope.cuh"

#include <cstdio>
#include <cstring>
#include <type_traits>   /* the declaration-vs-definition guard below */
#include <cuda_pipeline.h>
#include <cuda_fp16.h>

#define AF16_MT         2u        /* M-tiles per block */
#define AF16_HEADS     16u        /* the MMA's M */
#define AF16_HPB      (AF16_HEADS * AF16_MT)       /* 32 heads per block */
#define AF16_DIM      512u        /* head_dim this kernel is specialised for */
#define AF16_ROWS      16u        /* KV rows per tile = the MMA's K in phase 3 */
#define AF16_WARPS     16u
#define AF16_THREADS  (AF16_WARPS * 32u)
#define AF16_DPW      (AF16_DIM / AF16_WARPS)      /* 32 output dims per warp */
#define AF16_ROWB     (PULSAR_ATTN_PACK_ROWBYTES(AF16_DIM))
/* dynamic smem for the double-buffered raw KV tile stage (L037 lever 1) */
#define AF16_DYNSMEM_BYTES (2u * AF16_ROWS * AF16_ROWB)
#define AF16_KSTEPS   (AF16_DIM / 16u)             /* 32 k-steps for the scores */
#define AF16_KPW      (AF16_KSTEPS / 4u)           /* 8 k-steps per warp (4-way) */

/* WHY 2 M-TILES.  ncu on the 1-M-tile version: pipe_tensor 6-8%, pipe_lsu 33%,
 * 7.15 GB of L2 traffic in 11.9 ms -- the MMAs idle while the kernel moves KV.
 * Each block stages the whole window (up to 640 rows x 2 KB) and there were 4
 * head-groups per token all staging the SAME rows, so the traffic modelled at
 * 10.7 GB against the 7.15 measured (the window ramps up, so the model is high).
 *
 * With M fixed at 16, the block count is n_tokens*n_head/16 NO MATTER how the
 * 16 is split between tokens and heads -- so batching tokens per block does not
 * help, and adding a token only widens the staged window.  The only lever is
 * more (token, head) PAIRS per block, i.e. more M-tiles, which costs registers:
 * Q fragments and the O accumulator both scale with it.  Two tiles halves the
 * traffic at 32 Q + 32 O registers, which fits; four would need 128 before
 * temporaries and spill. */
/* +8 halves of padding: the phase-1 B fragment walks DOWN a column of the tile
 * (fixed dim, varying row), so an unpadded 512-wide row stride would put every
 * lane in the same bank. */
#define AF16_KVSTRIDE (AF16_DIM + 8u)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define PULSAR_ATTN_F16_MMA 1
#else
#define PULSAR_ATTN_F16_MMA 0
#endif

#if PULSAR_ATTN_F16_MMA
__device__ __forceinline__ static void af16_mma(
        float &d0, float &d1, float &d2, float &d3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

/* Load a pair of ADJACENT stored-Q elements as one packed-f16 fragment word.
 * For __half storage the bytes are already the fragment encoding, so it is a
 * single aligned 4-byte load -- the old path round-tripped half->float->half
 * for nothing (L086 T7).  Alignment: callers pass even element indices off
 * 256-aligned row bases.  (A <float> specialization that converted on load
 * existed for the f32-Q instantiation; that instantiation's only reference was
 * a smem grant, both deleted 2026-09-02.  A future Q width change adds its
 * specialization here and the compiler names the site.) */
template <typename QT>
__device__ __forceinline__ static uint32_t af16_load_pair(const QT *p, uint32_t i);
template <>
__device__ __forceinline__ uint32_t af16_load_pair<__half>(const __half *p, uint32_t i) {
    return *(const uint32_t *)(p + i);
}

/* Not under PULSAR_ATTN_F16_MMA: the fused Q-prep pair builder below (L037)
 * calls af16_pack from outside the MMA-only region. */
__device__ __forceinline__ static uint32_t af16_pack(float lo, float hi) {
    return (uint32_t)__half_as_ushort(__float2half(lo)) |
           ((uint32_t)__half_as_ushort(__float2half(hi)) << 16);
}

/* 512 threads means one block already fills 65536/(512) = 128 registers worth
 * of the SM budget, so asking for 2 resident blocks caps the kernel at 64 and
 * it spills (measured: 192 B stack, 196 B spill traffic).  1 is not a
 * concession here, it is the only setting that lets the accumulators live in
 * registers at all. */
#ifndef AF16_MINBLK
#define AF16_MINBLK 1
#endif
/* Build one packed f16 pair from RAW q: RMS-scale both elements, rope-rotate
 * when the pair sits in the tail. c0 is even and n_nope is even, so a pair
 * never straddles the nope/rope boundary and one c0 >= n_nope test covers
 * both elements. The rotation is the shared core every tail-rope consumer
 * uses (pulsar_cuda_rope.cuh) -- a transcribed copy here is how bit-exactness
 * would die. */
template <typename QT>
__device__ static inline uint32_t af16_pack_qraw(
        const QT *qrow, uint32_t c0, float scale, uint32_t q_nope,
        const pulsar_gpu_q_prep qp, uint32_t rope_pos,
        float corr0, float corr1) {
    float x0 = q_load<QT>(qrow, c0) * scale;
    float x1 = q_load<QT>(qrow, c0 + 1u) * scale;
    if (c0 >= q_nope) {
        float r0, r1;
        rope_pair_rotate_core_dev(x0, x1, c0 - q_nope, qp.n_rot, rope_pos, 0,
                                  qp.freq_base, qp.freq_scale, qp.ext_factor,
                                  qp.attn_factor, corr0, corr1, &r0, &r1);
        x0 = r0;
        x1 = r1;
    }
    return af16_pack(x0, x1);
}

template <typename QT>
__global__ __launch_bounds__(AF16_THREADS, AF16_MINBLK)
static void attn_f16_kernel(
        pulsar_heads_t *__restrict__ heads,   /* [n_tokens][n_head][512], STORED width
                                               * = PULSAR_HEADS_ELT_SIZE (L033) */
        const float *__restrict__ sinks,      /* [n_head] */
        const QT *__restrict__ q,             /* [n_tokens][n_head][512], f32 or __half */
        const pulsar_attn_pack_t *__restrict__ raw_kv,   /* PULSAR_ATTN_PACK rows */
        const pulsar_attn_pack_t *__restrict__ comp_kv,  /* PULSAR_ATTN_PACK rows; may
                                               * alias raw.  These carried a bare
                                               * float* until L092 -- "THE POINTER
                                               * TYPE LIES" was this comment's
                                               * opening line -- and a mis-strided
                                               * read of the 584 B rows once
                                               * produced OOB NaNs from a clean
                                               * compile.  The opaque type makes
                                               * that unwritable now: only the
                                               * pack accessors can read these. */
        const int32_t *__restrict__ topk,     /* NULL = dense window mode */
        uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio,
        uint32_t n_head,
        uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start_in,
        uint32_t top_k,
        /* positions: ring mode -> per-row query position for the row plan and
         * the fused Q rope; dense mode -> fused Q rope only (a banked
         * zero-prefix prefill batch ropes at positions[t], not t). */
        const int32_t *__restrict__ positions, const int32_t *__restrict__ seq_id,
        const void *const *__restrict__ comp_bank_ptrs,
        uint32_t comp_cap, uint32_t n_banks,
        int ring,                         /* ring/descriptor row plan; topk may
                                           * be NULL -> visible comp prefix */
        int non_causal,                   /* ring mode only: a query sees every
                                           * raw row up to the ring's last one,
                                           * not only rows at or before its own
                                           * position (the drafter's raw-window
                                           * forward).  Changes WHICH raw rows
                                           * are visible, never how a visible
                                           * row is folded; comp visibility is
                                           * untouched. */
        /* Grouped E4M3 activation slots for the attn-output "a" projection.
         * NULL = f32 only.  This epilogue owns the NOPE blocks (head dims
         * [0, n_nope)); rope_tail_kernel rewrites [n_nope, AF16_DIM) in place
         * afterwards and owns those blocks -- see pulsar_gpu_mxfp8_gact_slot. */
        __nv_fp8_e4m3 *__restrict__ gact_data,
        unsigned char *__restrict__ gact_scale,
        int gact_kbp, uint32_t gact_slab, uint32_t n_groups, uint32_t n_nope,
        /* The grouped encoding is indexed by BATCH-ABSOLUTE token, but this
         * kernel may be launched over a VIEW covering one token (the per-token
         * indexed prefill loop).  gact_tok0 is this launch's offset into the
         * batch and gact_ntok the batch's total row count -- they are the row
         * and the stride the "a" GEMM will read with, and are NOT the kernel's
         * own n_tokens unless the launch covers the whole batch. */
        uint32_t gact_tok0, uint32_t gact_ntok,
        /* L037 lever 3: when q_raw != 0, `q` holds RAW head projections and
         * this kernel applies the per-head RMS norm and tail rope itself at
         * Q-fragment build, bit-exactly matching head_rms_norm_rope_tail
         * (shared rope core, replicated reduction -- see the prologue). */
        pulsar_gpu_q_prep qp, int q_raw) {
#if !PULSAR_ATTN_F16_MMA
    (void)heads; (void)sinks; (void)q; (void)raw_kv; (void)comp_kv; (void)topk;
    (void)n_tokens; (void)n_comp; (void)window; (void)ratio; (void)n_head;
    (void)pos0; (void)n_raw; (void)raw_cap; (void)raw_start_in; (void)top_k;
    (void)positions; (void)seq_id; (void)comp_bank_ptrs; (void)comp_cap; (void)n_banks;
    (void)ring; (void)non_causal;
    (void)gact_data; (void)gact_scale; (void)gact_kbp; (void)gact_slab;
    (void)n_groups; (void)n_nope;
    (void)gact_tok0; (void)gact_ntok;
    (void)qp; (void)q_raw;
#else
    const uint32_t t = blockIdx.x;
    const uint32_t hbase = blockIdx.y * AF16_HPB;
    if (t >= n_tokens || hbase >= n_head) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t g = lane >> 2u;          /* fragment groupID 0..7 */
    const uint32_t tg = lane & 3u;          /* thread-in-group 0..3 */
    /* ldmatrix addressing for the phase-3 B fragment: lanes 0-7 supply the eight
     * rows of the k0..7 tile, lanes 8-15 the k8..15 tile (16-31 are ignored for
     * .x2 but still compute an address, harmlessly duplicating). Verified
     * bit-identical to the scalar loads it replaces -- see the note at the load. */
    const uint32_t ldm_row = lane & 7u;
    const uint32_t ldm_half = (lane >> 3u) & 1u;

    /* Dead/evicted row: zero the heads and leave together.  This MUST precede
     * every __syncthreads in the kernel -- a partial early return past a
     * barrier hangs the block. */
    if (seq_id && (uint32_t)seq_id[t] >= n_banks) {
        for (uint32_t i = tid; i < AF16_HPB * AF16_DIM; i += AF16_THREADS) {
            const uint32_t hh = hbase + i / AF16_DIM;
            if (hh < n_head) heads_store(heads, ((uint64_t)t * n_head + hh) * AF16_DIM + (i % AF16_DIM), 0.0f);
        }
        return;
    }

    /* ---- row plan ------------------------------------------------------
     * Two modes.  Dense window (topk == NULL, ring == 0) is the raw/mixed
     * prefill launchers' contiguous causal window over the batch's own
     * tokens.  RING mode is the decode-batch and indexed plan: raw rows come
     * out of a ring buffer (hence the modulo and the row table) and the
     * compressed rows are either a top-k SELECTION (topk != NULL) or the
     * visible prefix (topk == NULL -- the decode-batch sweep).  The
     * banked-descriptor variant (positions/seq_id/comp_bank_ptrs) and
     * ATTN_PACK comp rows are both served: the preamble below derives the
     * per-row qpos / raw_base / comp_src from the descriptors, and
     * NULL descriptors collapse every term to the scalar pos0+t path.  Bank
     * isolation is gated by tests/attn_f16_banked_test.cu; pack decode goes
     * through the shared attn_comp_pack_ld.
     *
     * RAW VISIBILITY.  The ring holds positions [first_raw_pos, raw_last_pos]
     * (per row under descriptors: the window ending at positions[t]; scalar:
     * the n_raw rows ending at pos0 + n_tokens - 1).  A row sees
     *   lo = max(first_raw_pos, qpos + 1 - window)      (window 0 = unbounded)
     *   hi = non_causal ? raw_last_pos : min(qpos, raw_last_pos)
     * so causal rows see only ring rows at or before their own position and
     * non-causal rows see the whole ring up to its last row.  Under
     * descriptors raw_last_pos == positions[t], so the drafter's banked
     * launch (positions = each bank's VISIBILITY position, the bank's last
     * draft position, window = raw_cap) derives the same window either way;
     * the scalar launch (pos0 = the first query's position, n_raw = the
     * visible rows) is where the two rules differ. */
    __shared__ uint32_t sRawRows[256];
    __shared__ uint32_t sRawCount, sRawFirst;
    __shared__ uint32_t sVisComp;
    uint32_t raw_count, raw_start = 0u, comp_count = 0u;
    uint32_t comp_base = 0u;
    const pulsar_attn_pack_t *comp_src = comp_kv;
    if (ring) {
        /* Descriptor (banked) preamble; NULL descriptors collapse to the
         * scalar pos0+t path. */
        const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
        uint32_t eff_n_raw = n_raw, eff_raw_start = raw_start_in, first_raw_pos;
        if (positions) {
            eff_n_raw = (window != 0u && qpos + 1u > window) ? window : qpos + 1u;
            if (eff_n_raw > raw_cap) eff_n_raw = raw_cap;
            eff_raw_start = (qpos + 1u - eff_n_raw) % raw_cap;
            first_raw_pos = qpos + 1u - eff_n_raw;
        } else {
            first_raw_pos = pos0 + n_tokens - n_raw;
        }
        const uint32_t sid_b = seq_id ? (uint32_t)seq_id[t] : 0u;
        const uint32_t raw_base = seq_id ? sid_b * raw_cap : 0u;
        comp_src = comp_bank_ptrs ? (const pulsar_attn_pack_t *)comp_bank_ptrs[sid_b] : comp_kv;
        comp_base = comp_bank_ptrs ? 0u : (seq_id ? sid_b * comp_cap : 0u);
        if (tid == 0u) {
            uint32_t rc = 0u, rf = 0u;
            if (eff_n_raw != 0u) {
                const uint32_t raw_last_pos = first_raw_pos + eff_n_raw - 1u;
                if (qpos >= first_raw_pos) {
                    uint32_t lo = first_raw_pos;
                    if (window != 0u && qpos + 1u > window) {
                        const uint32_t wlo = qpos + 1u - window;
                        if (wlo > lo) lo = wlo;
                    }
                    const uint32_t hi = non_causal ? raw_last_pos
                                      : (qpos < raw_last_pos ? qpos : raw_last_pos);
                    if (hi >= lo) { rf = lo - first_raw_pos; rc = hi - lo + 1u; }
                    if (rc > 256u) rc = 256u;
                }
            }
            sRawCount = rc; sRawFirst = rf;
        }
        __syncthreads();
        raw_count = sRawCount;
        for (uint32_t r = tid; r < raw_count; r += AF16_THREADS)
            sRawRows[r] = raw_base + (eff_raw_start + sRawFirst + r) % raw_cap;
        uint32_t visible_comp = n_comp;
        if (ratio != 0u) {
            visible_comp = (qpos + 1u) / ratio;
            if (visible_comp > n_comp) visible_comp = n_comp;
        }
        /* No topk table means this launcher sweeps the visible prefix (the
         * decode-batch path); with a table, top_k caps the selection. */
        comp_count = topk ? (top_k < visible_comp ? top_k : visible_comp) : visible_comp;
        sVisComp = visible_comp;
        __syncthreads();
    } else {
        raw_count = (window != 0u && t + 1u > window) ? window : t + 1u;
        raw_start = t + 1u - raw_count;
        if (n_comp != 0u && ratio != 0u) {
            comp_count = (t + 1u) / ratio;
            if (comp_count > n_comp) comp_count = n_comp;
        }
    }
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)AF16_DIM);

    /* __align__(16) is REQUIRED by the ldmatrix in phase 3: it loads 8 x 16 bits
     * per lane and the address must be 16-byte aligned.  A __half array is only
     * 2-byte aligned by default, and this one follows sRawRows[256] plus three
     * uint32_t, so its natural offset is 1036 -- misaligned by 12. */
    __shared__ __align__(16) __half sKV[AF16_ROWS * AF16_KVSTRIDE];
    __shared__ float  sPart[4][AF16_MT][AF16_HEADS][AF16_ROWS];  /* k-split partials */
    __shared__ float  sS[AF16_HPB][AF16_ROWS];
    __shared__ __half sP[AF16_HPB][AF16_ROWS];
    __shared__ float  sCorr[AF16_HPB];
    /* Per-tile row validity for the top-k selection. An entry the indexer left
     * invalid means "there is no k-th row here", so it must contribute NOTHING;
     * see the mask in phase 2. Written by the d4 == 0 thread of each row during
     * staging, so the existing post-staging __syncthreads covers it and no extra
     * barrier is needed. Rows >= nr are never read (phase 2 masks them anyway),
     * so a stale entry from the previous tile is harmless. */
    __shared__ uint8_t sRowBad[AF16_ROWS];
    __shared__ float  sM[AF16_HPB], sL[AF16_HPB];

    /* ---- Q fragments, once, into registers -----------------------------
     * Warp w owns k-steps [ (w>>1)*KPW, +KPW ) of head-block M=16.  Q is
     * constant for the whole kernel, so this is loaded once and never re-read;
     * that is the entire reason phase 1 splits k rather than n. */
    /* 4 jobs = 2 M-tiles x 2 n-tiles; 4 warps split k within each job. */
    const uint32_t job = warp & 3u;
    const uint32_t mtile = job >> 1u, ntile = job & 1u;
    const uint32_t kgrp = warp >> 2u;                   /* 0..3 */
    /* L037 lever 3 prologue: per-head RMS scales for the block's 32 rows,
     * computed with EXACTLY the standalone kernel's reduction (thread-local
     * strided 2-element square sums, then a 256-wide power-of-two tree --
     * head_rms_norm_rope_tail_kernel runs at blockDim 256), two rows per
     * pass across this block's 512 threads. Bit-exactness of the whole
     * fusion hangs on this loop matching that kernel operation-for-
     * operation; change either together or the prefill gate goes red. */
    __shared__ float sQscale[AF16_HPB];
    float q_corr0 = 0.0f, q_corr1 = 0.0f;
    if (q_raw) {
        __shared__ float sQpart[2][256];
        const uint32_t half = tid >> 8u;     /* which of the pass's two rows */
        const uint32_t vtid = tid & 255u;    /* the standalone kernel's tid */
        for (uint32_t r2 = 0; r2 < AF16_HPB; r2 += 2u) {
            const QT *xr = q + ((uint64_t)t * n_head + hbase + r2 + half) * AF16_DIM;
            float sum = 0.0f;
            for (uint32_t i = vtid; i < AF16_DIM; i += 256u) {
                float v = q_load<QT>(xr, i);
                sum += v * v;
            }
            sQpart[half][vtid] = sum;
            __syncthreads();
            for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
                if (vtid < stride) sQpart[half][vtid] += sQpart[half][vtid + stride];
                __syncthreads();
            }
            if (vtid == 0u)
                sQscale[r2 + half] = rsqrtf(sQpart[half][0] / (float)AF16_DIM + qp.eps);
            __syncthreads();
        }
        if (qp.ext_factor != 0.0f)
            rope_corr_dims_dev(qp.n_rot, qp.n_ctx_orig, qp.freq_base,
                               qp.beta_fast, qp.beta_slow, &q_corr0, &q_corr1);
    }

    uint32_t qf[AF16_KPW][4];
    {
        const uint64_t qbase =
            ((uint64_t)t * n_head + hbase + mtile * AF16_HEADS) * AF16_DIM;
        const uint32_t q_nope = AF16_DIM - qp.n_rot;
        const uint32_t q_rope_pos = positions ? (uint32_t)positions[t] : pos0 + t;
        const float sa = q_raw ? sQscale[mtile * AF16_HEADS + g] : 0.0f;
        const float sb = q_raw ? sQscale[mtile * AF16_HEADS + g + 8u] : 0.0f;
        #pragma unroll
        for (uint32_t s = 0; s < AF16_KPW; s++) {
            const uint32_t k0 = (kgrp * AF16_KPW + s) * 16u;
            const uint32_t hA = g, hB = g + 8u;         /* the fragment's two rows */
            const uint32_t c0 = k0 + tg * 2u, c1 = k0 + tg * 2u + 8u;
            const QT *qa = q + qbase + (uint64_t)hA * AF16_DIM;
            const QT *qb = q + qbase + (uint64_t)hB * AF16_DIM;
            if (q_raw) {
                qf[s][0] = af16_pack_qraw<QT>(qa, c0, sa, q_nope, qp, q_rope_pos, q_corr0, q_corr1);
                qf[s][1] = af16_pack_qraw<QT>(qb, c0, sb, q_nope, qp, q_rope_pos, q_corr0, q_corr1);
                qf[s][2] = af16_pack_qraw<QT>(qa, c1, sa, q_nope, qp, q_rope_pos, q_corr0, q_corr1);
                qf[s][3] = af16_pack_qraw<QT>(qb, c1, sb, q_nope, qp, q_rope_pos, q_corr0, q_corr1);
            } else {
                qf[s][0] = af16_load_pair<QT>(qa, c0);
                qf[s][1] = af16_load_pair<QT>(qb, c0);
                qf[s][2] = af16_load_pair<QT>(qa, c1);
                qf[s][3] = af16_load_pair<QT>(qb, c1);
            }
        }
    }

    /* ---- L037 lever 1: double-buffered RAW-byte tile staging ------------
     * Each ATTN_PACK row is AF16_ROWB bytes = a whole number of aligned
     * 8-byte chunks; the NEXT tile's rows stream into the other buffer via
     * cp.async while the CURRENT tile runs its MMA/softmax phases, which is
     * what finally overlaps the load/store unit with the tensor pipe (the
     * serial version measured pipe_tensor 6-8% vs LSU ~33%). The decode
     * below reads the staged bytes with the SAME arithmetic the global-read
     * version used (attn_comp_pack_ld's math, one decoder for the one
     * format), so this stage is byte movement only -- bit-exact. The
     * resolver mirrors the old per-element source logic one-for-one: raw
     * ring rows (sRawRows/raw_start), then comp rows with the topk bad-row
     * masking; a row outside the tile stages nothing and decodes to zero,
     * exactly as before. */
    /* The double buffer lives in DYNAMIC shared memory: adding its ~18.7 KB
     * to the kernel's static declarations blew the 48 KB static cap
     * (ptxas 0xc540 > 0xc000). The GB10 has 101 KB per SM; the launchers
     * opt in via cudaFuncAttributeMaxDynamicSharedMemorySize and pass
     * AF16_DYNSMEM_BYTES at launch. */
    extern __shared__ __align__(16) uint8_t af16_dynsmem[];
    uint8_t (*sRawB)[AF16_ROWS][AF16_ROWB] =
        (uint8_t (*)[AF16_ROWS][AF16_ROWB])af16_dynsmem;
    __shared__ const uint8_t *sSrc[2][AF16_ROWS];
    __shared__ uint8_t sBadStage[2][AF16_ROWS];
    #define AF16_STAGE_TILE(BUF, ROW0)                                        \
    do {                                                                      \
        const uint32_t _nr = (ROW0) < n_score                                 \
            ? min(AF16_ROWS, n_score - (ROW0)) : 0u;                          \
        if (tid < AF16_ROWS) {                                                \
            const uint32_t _r = tid;                                          \
            const uint32_t _sr = (ROW0) + _r;                                 \
            const uint8_t *_p = NULL;                                         \
            bool _bad = false;                                                \
            if (_r < _nr) {                                                   \
                if (_sr < raw_count) {                                        \
                    const uint32_t _rr = ring ? sRawRows[_sr]                 \
                                              : (raw_start + _sr);            \
                    _p = (const uint8_t *)raw_kv + (uint64_t)_rr * AF16_ROWB; \
                } else {                                                      \
                    uint32_t _ci = _sr - raw_count;                           \
                    if (topk) {                                               \
                        const int32_t _c = topk[(uint64_t)t * top_k + _ci];   \
                        _bad = !(_c >= 0 && (uint32_t)_c < sVisComp);         \
                        _ci = _bad ? 0u : (uint32_t)_c;                       \
                    }                                                         \
                    _p = (const uint8_t *)comp_src +                          \
                         ((uint64_t)comp_base + _ci) * AF16_ROWB;             \
                }                                                             \
            }                                                                 \
            sSrc[BUF][_r] = _p;                                               \
            sBadStage[BUF][_r] = _bad ? 1u : 0u;                              \
        }                                                                     \
        __syncthreads();                                                      \
        for (uint32_t _c = tid; _c < AF16_ROWS * (AF16_ROWB / 8u);            \
             _c += AF16_THREADS) {                                            \
            const uint32_t _r = _c / (AF16_ROWB / 8u);                        \
            const uint32_t _off = (_c % (AF16_ROWB / 8u)) * 8u;               \
            if (sSrc[BUF][_r])                                                \
                __pipeline_memcpy_async(&sRawB[BUF][_r][_off],                \
                                        sSrc[BUF][_r] + _off, 8);             \
        }                                                                     \
        __pipeline_commit();                                                  \
    } while (0)

    /* ---- running softmax state + output accumulator --------------------- */
    if (tid < AF16_HPB) { sM[tid] = -INFINITY; sL[tid] = 0.0f; }
    float acc[AF16_MT][AF16_DPW / 8u][4];
    #pragma unroll
    for (uint32_t m = 0; m < AF16_MT; m++)
        #pragma unroll
        for (uint32_t n = 0; n < AF16_DPW / 8u; n++)
            acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.0f;
    __syncthreads();

    AF16_STAGE_TILE(0u, 0u);
    for (uint32_t row0 = 0, tix = 0; row0 < n_score; row0 += AF16_ROWS, tix++) {
        const uint32_t nr = min(AF16_ROWS, n_score - row0);
        const uint32_t buf = tix & 1u;
        const int have_next = row0 + AF16_ROWS < n_score;
        if (have_next) AF16_STAGE_TILE(buf ^ 1u, row0 + AF16_ROWS);
        /* Leave the just-committed next-tile group in flight; wait only for
         * the group that filled THIS buffer. */
        __pipeline_wait_prior(have_next ? 1 : 0);
        __syncthreads();

        /* ---- decode the staged tile to fp16 -----------------------------
         * Same restructured decode as before (a uint32 of four E4M3 bytes
         * against one scale byte, bf16 rope tail), now reading smem the
         * cp.async stage filled. Raw and comp rows are the SAME format, so
         * the two old source arms collapse into this one decoder. */
        for (uint32_t i = tid; i < AF16_ROWS * AF16_DIM / 4u; i += AF16_THREADS) {
            const uint32_t r = i / (AF16_DIM / 4u);
            const uint32_t d4 = (i % (AF16_DIM / 4u)) * 4u;
            float f0 = 0.f, f1 = 0.f, f2 = 0.f, f3 = 0.f;
            if (r < nr && sSrc[buf][r]) {
                const uint8_t *pr = sRawB[buf][r];
                /* One row format for raw and comp alike (L111 unification):
                 * the row-relative decode is the one attn_comp_row_ld4
                 * (pulsar_cuda_internal.h), reading the smem copy the stage
                 * filled. */
                const float4 v = attn_comp_row_ld4(pr, d4 >> 2, AF16_DIM);
                f0 = v.x; f1 = v.y; f2 = v.z; f3 = v.w;
            }
            if (d4 == 0u) sRowBad[r] = (r < nr) ? sBadStage[buf][r] : 0u;
            __half2 *dst = (__half2 *)&sKV[r * AF16_KVSTRIDE + d4];
            dst[0] = make_half2(__float2half(f0), __float2half(f1));
            dst[1] = make_half2(__float2half(f2), __float2half(f3));
        }
        __syncthreads();

        /* ---- phase 1: S = Q . KV^T ------------------------------------- */
        {
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            const uint32_t rbase = ntile * 8u;
            #pragma unroll
            for (uint32_t s = 0; s < AF16_KPW; s++) {
                const uint32_t k0 = (kgrp * AF16_KPW + s) * 16u;
                /* B[k=dim][n=row] = KV[row][dim]: fixed row per lane group,
                 * two dim pairs.  This is the column walk the padding is for. */
                const __half *kr = &sKV[(rbase + g) * AF16_KVSTRIDE + k0];
                const uint32_t b0 = *(const uint32_t *)&kr[tg * 2u];
                const uint32_t b1 = *(const uint32_t *)&kr[tg * 2u + 8u];
                af16_mma(s0, s1, s2, s3, qf[s][0], qf[s][1], qf[s][2], qf[s][3], b0, b1);
            }
            sPart[kgrp][mtile][g][rbase + tg * 2u]      = s0;
            sPart[kgrp][mtile][g][rbase + tg * 2u + 1u] = s1;
            sPart[kgrp][mtile][g + 8u][rbase + tg * 2u]      = s2;
            sPart[kgrp][mtile][g + 8u][rbase + tg * 2u + 1u] = s3;
        }
        __syncthreads();

        /* ---- phase 2: sum the k-split, then online softmax --------------- */
        for (uint32_t i = tid; i < AF16_HPB * AF16_ROWS; i += AF16_THREADS) {
            const uint32_t h = i / AF16_ROWS, r = i % AF16_ROWS;
            const uint32_t mt = h / AF16_HEADS, hh = h % AF16_HEADS;
            const float v = sPart[0][mt][hh][r] + sPart[1][mt][hh][r] +
                            sPart[2][mt][hh][r] + sPart[3][mt][hh][r];
            sS[h][r] = (r < nr && !sRowBad[r]) ? v * scale : -INFINITY;
        }
        __syncthreads();

        if (tid < AF16_HPB) {
            const uint32_t h = tid;
            float mx = sM[h];
            for (uint32_t r = 0; r < nr; r++) mx = fmaxf(mx, sS[h][r]);
            /* ⚠ MASKING MAKES AN ALL--INF TILE REACHABLE, AND expf(-INF - -INF)
             * IS NaN.  Before the row mask above, every r < nr carried a real
             * score so mx could only be -INF when the tile was empty of rows at
             * all; now a tile whose selections are ALL invalid produces one.
             * Such a tile contributes nothing, so short-circuit p to zero rather
             * than letting a NaN into sP and from there into the output.
             * mx == -INF implies sM[h] == -INF (mx >= sM[h]), so corr is already
             * 0 and l stays 0 -- nothing has been accumulated to rescale. */
            const bool empty = (mx == -INFINITY);
            const float corr = (sM[h] == -INFINITY) ? 0.0f : __expf(sM[h] - mx);
            float l = sL[h] * corr;
            for (uint32_t r = 0; r < nr; r++) {
                const float p = empty ? 0.0f : __expf(sS[h][r] - mx);
                /* Sum the ROUNDED weight, not p: the MMA multiplies sP, so the
                 * normaliser has to be the sum of what it multiplies or the
                 * weights do not sum to one.  The sink term in the epilogue is
                 * deliberately NOT rounded -- it has no V row, it is denominator
                 * mass only. */
                const __half ph = __float2half(p);
                sP[h][r] = ph;
                l += __half2float(ph);
            }
            for (uint32_t r = nr; r < AF16_ROWS; r++) sP[h][r] = __float2half(0.f);
            sM[h] = mx; sL[h] = l; sCorr[h] = corr;
        }
        __syncthreads();

        /* ---- phase 3: O = O*corr + P . KV ------------------------------- */
        #pragma unroll
        for (uint32_t m = 0; m < AF16_MT; m++) {
            /* The accumulator's two head rows for this lane are g and g+8 of
             * THIS m-tile; every warp now sweeps both tiles against the one
             * staged KV tile, which is the whole point of the change. */
            const uint32_t hb0 = m * AF16_HEADS;
            const float ca = sCorr[hb0 + g], cb = sCorr[hb0 + g + 8u];
            const uint32_t pa0 = *(const uint32_t *)&sP[hb0 + g][tg * 2u];
            const uint32_t pa1 = *(const uint32_t *)&sP[hb0 + g + 8u][tg * 2u];
            const uint32_t pa2 = *(const uint32_t *)&sP[hb0 + g][tg * 2u + 8u];
            const uint32_t pa3 = *(const uint32_t *)&sP[hb0 + g + 8u][tg * 2u + 8u];
            #pragma unroll
            for (uint32_t n = 0; n < AF16_DPW / 8u; n++) {
                acc[m][n][0] *= ca; acc[m][n][1] *= ca;
                acc[m][n][2] *= cb; acc[m][n][3] *= cb;
                /* B[k=row][n=dim]: this lane's column is n_base+g, and its four
                 * k values are rows 2t, 2t+1, 2t+8, 2t+9 -- i.e. every lane walks
                 * a COLUMN of a row-major tile, which is what `ldmatrix.trans`
                 * exists for.  One instruction replaces four scalar LDS.u16 plus
                 * the shift/or repacking, and the transpose is free in the
                 * instruction rather than paid in address arithmetic.
                 *
                 * BIT-IDENTICAL, verified rather than assumed: a standalone probe
                 * (pulsar-notes/probes/ldm_probe.cu) filled this exact [16][520]
                 * layout with position-encoding values, built the fragment both
                 * ways and compared the raw registers -- 0 of 512 differed across
                 * all warps and lanes.  Guessing an ldmatrix lane mapping is
                 * precisely how you get a kernel that runs and is quietly wrong,
                 * so the mapping is evidence, not derivation. */
                const uint32_t colbase = warp * AF16_DPW + n * 8u;
                uint32_t b0, b1;
                {
                    const uint32_t saddr = (uint32_t)__cvta_generic_to_shared(
                        &sKV[(ldm_half * 8u + ldm_row) * AF16_KVSTRIDE + colbase]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                                 "{%0,%1}, [%2];\n"
                                 : "=r"(b0), "=r"(b1) : "r"(saddr));
                }
                af16_mma(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3],
                         pa0, pa1, pa2, pa3, b0, b1);
            }
        }
        __syncthreads();
    }

    /* ---- epilogue: fold the sink, normalise, store ---------------------- */
    if (tid < AF16_HPB) {
        const uint32_t h = hbase + tid;
        const float sink = (h < n_head) ? sinks[h] : -INFINITY;
        const float nm = fmaxf(sM[tid], sink);
        const float os = (sM[tid] == -INFINITY) ? 0.0f : __expf(sM[tid] - nm);
        sCorr[tid] = os;
        sL[tid] = sL[tid] * os + __expf(sink - nm);
    }
    __syncthreads();

    {
        /* D layout is d0=D[g][2t], d1=D[g][2t+1], d2=D[g+8][2t], d3=D[g+8][2t+1]
         * with M = head and N = dim.  So this lane holds heads g and g+8 (which
         * is why the corrections above are sCorr[g]/sCorr[g+8]) at dims
         * n_base+2t and n_base+2t+1.  Note N is the DIM axis here and the ROW
         * axis in phase 1 -- the two phases transpose, and mixing them up is
         * the one bug this kernel cannot detect at runtime. */
        #pragma unroll
        for (uint32_t m = 0; m < AF16_MT; m++) {
            const uint32_t hb0 = m * AF16_HEADS;
            const float ia = sCorr[hb0 + g] /
                             (sL[hb0 + g] == 0.0f ? 1.0f : sL[hb0 + g]);
            const float ib = sCorr[hb0 + g + 8u] /
                             (sL[hb0 + g + 8u] == 0.0f ? 1.0f : sL[hb0 + g + 8u]);
            const uint32_t ha = hbase + hb0 + g, hb = hbase + hb0 + g + 8u;
            pulsar_heads_t *oa = heads + ((uint64_t)t * n_head + ha) * AF16_DIM;
            pulsar_heads_t *ob = heads + ((uint64_t)t * n_head + hb) * AF16_DIM;
            /* ⚠ The E4M3 emission below takes the SAME acc[] registers, NOT a
             * read-back of these stores -- so narrowing this buffer does not
             * change what the attn-output GEMM multiplies on this path.  The
             * fidelity effect of the flip arrives through rope_tail, which DOES
             * read heads back, rotates in place, and emits the tail's share. */
            #pragma unroll
            for (uint32_t n = 0; n < AF16_DPW / 8u; n++) {
                const uint32_t nb = warp * AF16_DPW + n * 8u;
                heads_store(oa, nb + tg * 2u,      acc[m][n][0] * ia);
                heads_store(oa, nb + tg * 2u + 1u, acc[m][n][1] * ia);
                heads_store(ob, nb + tg * 2u,      acc[m][n][2] * ib);
                heads_store(ob, nb + tg * 2u + 1u, acc[m][n][3] * ib);
            }
            /* E4M3 for the attn-output "a" GEMM, straight out of the registers
             * the f32 stores above just used.
             *
             * WHY THE REDUCTION IS ONLY 4 LANES WIDE: warp w owns head dims
             * [32w, 32w+32) -- exactly one MX block -- but for 16 heads at once,
             * and any ONE head's 32 values live in the quad of lanes sharing g
             * (lane = g*4 + tg).  So the block amax is fmaxf over this lane's 8
             * registers then two shuffles across tg, CHEAPER than the warp-wide
             * reduction pulsar_mx_emit_block does.  The predicate is warp-
             * uniform, so every lane reaches the shuffles.
             *
             * The GEMM wants the grouped/transposed layout that
             * mxfp8_quant_act_grouped_kernel produces: data[g][tok][group_dim]
             * with a per-group scale slab, where group g holds n_head/n_groups
             * consecutive heads.  group_dim % 32 == 0 and AF16_DIM % 32 == 0, so
             * a head's blocks never straddle a group boundary. */
            if (gact_data && (warp * AF16_DPW) < n_nope) {
                const uint32_t hpg = n_head / n_groups;      /* heads per group */
                const uint32_t gd  = hpg * AF16_DIM;         /* == group_dim */
                float aa = 0.0f, ab = 0.0f;
                #pragma unroll
                for (uint32_t n = 0; n < AF16_DPW / 8u; n++) {
                    aa = fmaxf(aa, fabsf(acc[m][n][0] * ia));
                    aa = fmaxf(aa, fabsf(acc[m][n][1] * ia));
                    ab = fmaxf(ab, fabsf(acc[m][n][2] * ib));
                    ab = fmaxf(ab, fabsf(acc[m][n][3] * ib));
                }
                aa = fmaxf(aa, __shfl_xor_sync(0xffffffffu, aa, 1));
                aa = fmaxf(aa, __shfl_xor_sync(0xffffffffu, aa, 2));
                ab = fmaxf(ab, __shfl_xor_sync(0xffffffffu, ab, 1));
                ab = fmaxf(ab, __shfl_xor_sync(0xffffffffu, ab, 2));
                const int sea = pulsar_mx_shared_exp(aa);
                const int seb = pulsar_mx_shared_exp(ab);
                const uint32_t gra = ha / hpg, hha = ha % hpg;
                const uint32_t grb = hb / hpg, hhb = hb % hpg;
                const uint32_t arow = gact_tok0 + t;   /* batch-absolute token */
                __nv_fp8_e4m3 *da = gact_data + ((size_t)gra * gact_ntok + arow) * gd + hha * AF16_DIM;
                __nv_fp8_e4m3 *db = gact_data + ((size_t)grb * gact_ntok + arow) * gd + hhb * AF16_DIM;
                #pragma unroll
                for (uint32_t n = 0; n < AF16_DPW / 8u; n++) {
                    const uint32_t nb = warp * AF16_DPW + n * 8u;
                    da[nb + tg * 2u]      = pulsar_mx_encode(acc[m][n][0] * ia, sea);
                    da[nb + tg * 2u + 1u] = pulsar_mx_encode(acc[m][n][1] * ia, sea);
                    db[nb + tg * 2u]      = pulsar_mx_encode(acc[m][n][2] * ib, seb);
                    db[nb + tg * 2u + 1u] = pulsar_mx_encode(acc[m][n][3] * ib, seb);
                }
                if (tg == 0u) {
                    const uint32_t kba = hha * (AF16_DIM / 32u) + warp;
                    const uint32_t kbb = hhb * (AF16_DIM / 32u) + warp;
                    gact_scale[(size_t)gra * gact_slab +
                               pulsar_mx_sfoff((int)arow, (int)kba, gact_kbp)] = pulsar_mx_scale_byte(sea);
                    gact_scale[(size_t)grb * gact_slab +
                               pulsar_mx_sfoff((int)arow, (int)kbb, gact_kbp)] = pulsar_mx_scale_byte(seb);
                }
            }
        }
    }
#endif
}

/* ---- launcher ----------------------------------------------------------- */

/* PULSAR_ATTN_F16_MMA is a DEVICE-side guard -- __CUDA_ARCH__ is undefined in
 * the host pass, so testing it here would compile the launcher down to
 * "return 0" and the kernel would look like a permanent shape-gate refusal.
 * The host question is a runtime one: does this device actually have the
 * m16n8k16 tensor cores?  Ask the driver, once. */
static int af16_device_supported(void) {
    static int ok = -1;
    if (ok < 0) {
        int major = 0, minor = 0, dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) { ok = 0; return ok; }
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            ok = 0; return ok;
        }
        ok = (major >= 8) ? 1 : 0;      /* mma.m16n8k16.f16 is sm_80+ */
    }
    return ok;
}

/* Only the fp16 tier consumes packed comp rows on the multi-token
 * single-sequence path; the f32 indexed kernel's own gate refuses them there
 * (descr || !comp_kv_pack || n_tokens == 1 || f16_idx_ok -- the last term IS
 * this function, widening the gate for the tier that reads pack natively).
 * So this answers for the tier that is actually going to run. */
int pulsar_gpu_attention_prefill_reads_packed_comp(void) {
    /* L106 K14: this had a body IDENTICAL to pulsar_gpu_attn_f16_tier_on and
     * the two drifted risk was real (same predicate, two caches).  The two
     * NAMES stay -- call sites ask two different questions ("does prefill
     * consume packed comp rows?" vs "is the f16 tier on?") that happen to have
     * the same answer today; if that ever stops being true, the split point is
     * already named.  One body now. */
    return pulsar_gpu_attn_f16_tier_on();
}

/* One-time dynamic-smem opt-in for attn_f16_kernel: static shared alone is
 * capped at 48 KB, so the tile double-buffer's bytes must be granted here.
 * Fail-loud: a refused grant means the launch would silently get 0 dynamic
 * bytes and fault, so the caller refuses instead. */
static int af16_dynsmem_ok(void) {
    static int state = 0;   /* 0 = untried, 1 = ok, -1 = refused */
    if (state == 0) {
        /* ⚠ THE GRANT IS PER-INSTANTIATION.  attn_f16_kernel is templated on
         * batch_q's element type (L045), and a missing grant does not fail
         * loudly at build time -- the launch gets 0 dynamic bytes and faults,
         * exactly as the comment above warns.  Grant the instantiation that
         * launches, named by the same typedef the launch sites use, so a
         * future width change moves the grant with it.  (A second <float>
         * grant used to sit here as a "safety net"; it was the only reference
         * to that instantiation, i.e. a whole dead kernel kept alive by a
         * comment -- deleted 2026-09-02.) */
        cudaError_t e = cudaFuncSetAttribute(attn_f16_kernel<pulsar_q_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)AF16_DYNSMEM_BYTES);
        state = (e == cudaSuccess) ? 1 : -1;
        if (state < 0)
            fprintf(stderr, "pulsar: attn f16 dynamic smem grant refused: %s\n",
                    cudaGetErrorString(e));
    }
    return state > 0;
}

int pulsar_gpu_attn_f16_tier_on(void) {
    /* The PULSAR_CUDA_ATTN_F16=0 opt-out was retired 2026-09-02 (no caller;
     * the cuBLAS two-GEMM arm it fell back to is gone).  The tier is a
     * property of the device now. */
    return af16_device_supported();
}

int pulsar_gpu_attention_f16_prefill_mx(
        void *heads_v, const float *sinks, const void *q,
        const pulsar_attn_pack_t *raw_kv, const pulsar_attn_pack_t *comp_kv,
        uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim,
        void *gact_data, void *gact_scale, int gact_kbp,
        uint32_t gact_slab, uint32_t n_groups, uint32_t n_nope,
        uint32_t gact_tok0, uint32_t gact_ntok,
        const int *positions, const pulsar_gpu_q_prep *q_prep) {
    pulsar_heads_t *heads = (pulsar_heads_t *)heads_v;
    if (!heads || !sinks || !q || !raw_kv) return 0;
    pulsar_gpu_q_prep qp;
    memset(&qp, 0, sizeof qp);
    if (q_prep) {
        /* The fused Q load owns the whole nope/rope split: refuse shapes the
         * fragment-pair math cannot cover rather than rope the wrong dims. */
        if (q_prep->n_rot == 0u || (q_prep->n_rot & 1u) ||
            q_prep->n_rot > AF16_DIM || ((AF16_DIM - q_prep->n_rot) & 1u)) {
            fprintf(stderr, "pulsar: attn f16 cannot fuse q norm+rope for n_rot=%u\n",
                    q_prep->n_rot);
            return 0;
        }
        qp = *q_prep;
    }

    if (head_dim != AF16_DIM) return 0;
    if (n_head == 0u || (n_head % AF16_HPB) != 0u) return 0;
    if (n_tokens == 0u) return 0;
    if (n_comp != 0u && !comp_kv) return 0;
    if (!af16_device_supported()) return 0;
    /* Same packed-geometry contract the indexed entry enforces: the rope tail
     * must fit and the nope span must divide the scale block, or attn_pack_ld
     * walks rows that are not there. */
    if (head_dim <= PULSAR_ATTN_PACK_NROT ||
        ((head_dim - PULSAR_ATTN_PACK_NROT) % PULSAR_KV4_NV_BLOCK) != 0) return 0;
    /* The epilogue's grouped index math assumes whole heads per group and that
     * the nope/rope split falls on an MX block boundary (so rope_tail can own
     * the tail blocks without either side touching the other's).  Refuse the
     * emission rather than write a partial encoding the GEMM would then read. */
    if (gact_data && (n_groups == 0u || (n_head % n_groups) != 0u ||
                      (n_nope % 32u) != 0u || n_nope > AF16_DIM)) {
        fprintf(stderr, "pulsar: attn f16 cannot emit MX for n_head=%u n_groups=%u "
                        "n_nope=%u\n", n_head, n_groups, n_nope);
        return 0;
    }
    if (!af16_dynsmem_ok()) return 0;
    {
    dim3 grid(n_tokens, n_head / AF16_HPB, 1);
    /* Dense mode: `positions` reaches the kernel for the fused Q rope only
     * (positions[t] instead of t); the row plan stays the dense causal
     * window.  With q_prep == NULL it is unused. */
    attn_f16_kernel<pulsar_q_t><<<grid, AF16_THREADS, AF16_DYNSMEM_BYTES>>>(heads, sinks, (const pulsar_q_t *)q, raw_kv,
                                            comp_kv ? comp_kv : raw_kv, NULL,
                                            n_tokens, n_comp, window, ratio,
                                            n_head, 0u, 0u, 1u, 0u, 0u,
                                            (const int32_t *)positions, NULL, NULL, 0u, 1u, 0, 0,
                                            (__nv_fp8_e4m3 *)gact_data,
                                            (unsigned char *)gact_scale,
                                            gact_kbp, gact_slab, n_groups, n_nope,
                                            gact_tok0, gact_ntok,
                                            qp, q_prep != NULL);
    return cuda_ok(cudaGetLastError(), "attention f16 mma launch");
    }
}

/* Declaration-vs-definition guard for the wrapper below.
 *
 * pulsar_gpu.h reaches this TU through pulsar_cuda_internal.h, so a drifted
 * declaration does NOT produce a redefinition error -- C++ just quietly adds a
 * second overload.  No engine translation unit references this wrapper (only
 * tests/attn_f16_kernel_test.cu does), so the stale overload stays undefined
 * and unreferenced and the whole engine still links clean.  That is exactly
 * how the L033 increment-2 signature change shipped a header the compiler was
 * happy with and the attn gate was not.
 *
 * So this pins the declared type against the definition's signature.  Observed
 * behaviour, not predicted: reverting the header to `float *heads` makes this
 * fail as a plain static assertion (nvcc resolves the decltype rather than
 * rejecting it as ambiguous), reported at this line with the message below.
 * Either way the build stops here -- and a build error beats a link error in a
 * gate an hour later.
 *
 * If you change this function's signature you must edit BOTH the header and
 * the definition; this assert only tells you that you forgot, it cannot tell
 * you which one is right. */
static_assert(std::is_same<
        decltype(pulsar_gpu_attention_f16_prefill),
        int(void *, const float *, const void *,
            const pulsar_attn_pack_t *, const pulsar_attn_pack_t *,
            uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
            const pulsar_gpu_q_prep *)>::value,
    "pulsar_gpu.h's pulsar_gpu_attention_f16_prefill has drifted from the "
    "definition below; update the header -- do not add an overload");

int pulsar_gpu_attention_f16_prefill(
        void *heads_v, const float *sinks, const void *q,
        const pulsar_attn_pack_t *raw_kv, const pulsar_attn_pack_t *comp_kv,
        uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim,
        const pulsar_gpu_q_prep *q_prep) {
    return pulsar_gpu_attention_f16_prefill_mx(heads_v, sinks, q, raw_kv, comp_kv,
                                               n_tokens, n_comp, window, ratio,
                                               n_head, head_dim,
                                               NULL, NULL, 0, 0u, 0u, 0u, 0u, 0u,
                                               NULL, q_prep);
}

/* Indexed variant: raw rows come from a ring buffer; compressed rows are a
 * top-k SELECTION (topk != NULL) or the visible prefix (topk == NULL).
 * Banked descriptors are all-or-nothing (positions+seq_id+comp_bank_ptrs
 * together or none -- a partial set is refused rather than guessed at), and
 * comp rows are ATTN_PACK, always -- the format parameter is gone (see the note
 * on the kernel).  non_causal selects the raw visibility rule (kernel note:
 * RAW VISIBILITY); it is the drafter's raw-window forward and is passed
 * through unchanged.  A 0 here is a real failure, never a silent shape
 * demotion. */
int pulsar_gpu_attention_f16_indexed(
        void *heads_v, const float *sinks, const void *q,
        const pulsar_attn_pack_t *raw_kv, const pulsar_attn_pack_t *comp_kv, const int *topk,
        uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
        uint32_t raw_start, uint32_t n_comp, uint32_t top_k, uint32_t window,
        uint32_t ratio, uint32_t n_head, uint32_t head_dim,
        const int *positions, const int *seq_id, const void *const *comp_bank_ptrs,
        uint32_t comp_cap, uint32_t n_banks, uint32_t non_causal,
        const pulsar_gpu_q_prep *q_prep) {
    /* topk may be NULL: the decode-batch/continued-prefill path sweeps the
     * visible comp prefix rather than a selection. */
    pulsar_heads_t *heads = (pulsar_heads_t *)heads_v;
    if (!heads || !sinks || !q || !raw_kv || !comp_kv) return 0;
    pulsar_gpu_q_prep qp;
    memset(&qp, 0, sizeof qp);
    if (q_prep) {
        /* The fused Q load owns the whole nope/rope split: refuse shapes the
         * fragment-pair math cannot cover rather than rope the wrong dims. */
        if (q_prep->n_rot == 0u || (q_prep->n_rot & 1u) ||
            q_prep->n_rot > AF16_DIM || ((AF16_DIM - q_prep->n_rot) & 1u)) {
            fprintf(stderr, "pulsar: attn f16 cannot fuse q norm+rope for n_rot=%u\n",
                    q_prep->n_rot);
            return 0;
        }
        qp = *q_prep;
    }

    /* Descriptors are all-or-nothing, as in the dispatcher's own check. */
    if ((positions != NULL) != (seq_id != NULL)) return 0;
    /* comp_cap is the PER-BANK comp-row stride, so it is only meaningful when
     * there are compressed rows to address.  Requiring it unconditionally under
     * descriptors rejected the legitimate n_comp == 0 case -- which is what
     * broke mixed-batch continued prefill for three days (L032): the launcher
     * returned 0, the caller correctly refused to fall through, and the whole
     * mixed run died with 'multiseq step_end FAILED'.  A banked batch whose
     * visible comp prefix is empty is normal, not a bug. */
    if (positions && (n_banks == 0u || (n_comp != 0u && comp_cap == 0u))) return 0;
    if (head_dim != AF16_DIM) return 0;
    if (n_head == 0u || (n_head % AF16_HPB) != 0u) return 0;
    if (n_tokens == 0u || raw_cap == 0u) return 0;
    if (topk && top_k == 0u) return 0;      /* a table with no budget is a bug */
    /* No bound on n_raw: it is the whole raw RING, and the per-token raw_count
     * is what sRawRows[256] must hold; the kernel caps it at 256
     * (PULSAR_CUDA_ATTENTION_RAW_SCORE_CAP, which the dispatcher's descriptor
     * check enforces on `window`). */
    if (!af16_device_supported()) return 0;
    if (!af16_dynsmem_ok()) return 0;
    dim3 grid(n_tokens, n_head / AF16_HPB, 1);
    attn_f16_kernel<pulsar_q_t><<<grid, AF16_THREADS, AF16_DYNSMEM_BYTES>>>(heads, sinks, (const pulsar_q_t *)q, raw_kv, comp_kv,
                                            (const int32_t *)topk,
                                            n_tokens, n_comp, window, ratio,
                                            n_head, pos0, n_raw, raw_cap,
                                            raw_start, top_k,
                                            (const int32_t *)positions,
                                            (const int32_t *)seq_id,
                                            comp_bank_ptrs, comp_cap,
                                            positions ? n_banks : 1u, 1,
                                            non_causal != 0u,
                                            NULL, NULL, 0, 0u, 0u, 0u, 0u, 0u,
                                            qp, q_prep != NULL);
    return cuda_ok(cudaGetLastError(), "attention f16 indexed launch");
}

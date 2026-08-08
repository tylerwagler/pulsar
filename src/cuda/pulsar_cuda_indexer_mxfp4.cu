/* Indexer scorer on the SM120 block-scaled tensor cores.
 *
 * Replaces indexer_scores_wmma128_kernel's 64 serial head iterations (two
 * __syncthreads() each, 128 per block, and a full FP4->f32->__half dequant of
 * the compressed rows into a 32 KB fp16 tile) with one block-scaled GEMM that
 * consumes the cache close to its stored form.
 *
 * Design + measurements: docs/indexer-mxfp4-scorer.md
 * Instruction contract:  tests/idx_mxfp4_probe.cu   (every constant below is
 *                        measured on GB10, not read off a spec)
 * Operand choice:        tests/idx_quant_fidelity.cc (E4M3 for Q, measured)
 *
 *   score[t][c] = ( sum_h ReLU(q[t,h,:] . k[c,:]) * w[t,h] ) * scale
 *
 * MEASURED CONTRACT -- the two that bite:
 *   ue8m0 value = 2^(byte - 128).  The OCP spec's 127 bias would scale every
 *     product by 1/4 and present as "the k-sum is a quarter of what it owes",
 *     not as an error.
 *   e2m1 nibble sits at bits [5:2] of its byte container and decodes to 4x the
 *     nominal table.  At bit 0 it decodes as a LINEAR code*0.5 ramp that agrees
 *     with the real table for codes 0..4 and silently diverges at 5,6,7.
 *   Hence, for a cache byte s (our storage is 2^(s-127) * e2m1):
 *     sf_hw = s - 1        (+1 for the bias, -2 for the 4x)
 *
 * DECOMPOSITION.  One block = one token x one 64-wide compressed tile.
 *   M = 64 (the token's heads)      -- the head loop becomes the GEMM's M
 *   N = 64 (compressed rows)
 *   K = 128 (head_dim), four k=32 MMA steps
 * The MX 32-element scale blocks line up exactly with the MMA's k=32, so each
 * step consumes precisely one scale per operand -- no regrouping.
 *
 * Grid is (ceil(n_comp/64), n_tokens): at n_comp=512, n_tokens=512 that is
 * 4096 blocks against the old kernel's 128, which is the occupancy half of the
 * problem.  The sync count goes from 128 per block to 3.
 */
#include "pulsar_cuda_internal.h"

#include <cstdio>

/* The block-scaled asm cannot be emitted into the compute_120 PTX fallback pass
 * -- ptxas rejects .kind::mxf8f6f4 for .target sm_120 -- so it must be guarded
 * exactly the way CUTLASS guards its copy, or the whole TU fails to build at
 * our normal -arch=sm_120f.  See the probe's header. */
/* __CUDA_ARCH__ alone is NOT the discriminator: at -arch=sm_120f nvcc runs two
 * device passes and __CUDA_ARCH__ is 1200 in both, so gating on it emits the asm
 * into the compute_120 PTX (JIT-fallback) pass, where ptxas rejects it and the
 * whole TU fails to build.  __CUDA_ARCH_FAMILY_SPECIFIC__ is defined ONLY in the
 * sm_120f SASS pass (verified: it prints once where __CUDA_ARCH__ prints twice);
 * __CUDA_ARCH_SPECIFIC__ / __CUDA_ARCH_FEAT_SM120_ALL cover the sm_12Xa targets. */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) &&                     \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) ||                               \
     defined(__CUDA_ARCH_SPECIFIC__) ||                                      \
     defined(__CUDA_ARCH_FEAT_SM120_ALL))
#define PULSAR_IDX_MXFP4_MMA 1
#else
#define PULSAR_IDX_MXFP4_MMA 0
#endif

#define IDX_HEADS     64u
#define IDX_HEAD_DIM  128u
#define IDX_NTILE     128u         /* compressed rows per block */
#define IDX_KSLABS    (IDX_HEAD_DIM / 32u)   /* 4 */
/* Shared rows are PADDED off the bank period.  A 128-byte row stride is exactly
 * 32 words = the bank count, so every row starts in bank 0: the eight lanes of
 * an m16n8 group read eight different rows at the same k, and all eight hit the
 * same bank -- an 8-way conflict on every A fetch, and 4-way on B at 64 B.
 * Padding by one word breaks the period without changing any addressing math
 * beyond the stride constant. */
#define IDX_ASTRIDE   (IDX_HEAD_DIM + 4u)        /* 132 B */
#define IDX_BSTRIDE   ((IDX_HEAD_DIM / 2u) + 4u) /*  68 B */
#define IDX_TOKTILE   2u           /* tokens per block: B is shared across them */
#define IDX_THREADS   256u         /* 8 warps: 4 per token */

/* ---- E4M3 encode (host-side rules, device-side implementation) ----------- */

__device__ __forceinline__ static uint8_t idx_f32_to_e4m3(float v) {
    /* Round-to-nearest onto the E4M3 grid: 4-bit exponent (bias 7), 3 mantissa
     * bits, max finite 448.  __nv_cvt_float_to_fp8 exists but pulls in a header
     * this TU does not otherwise need, and this is off the hot path (one
     * conversion per Q element per tile). */
    const float a = fabsf(v);
    const uint32_t sign = (v < 0.0f) ? 0x80u : 0x00u;
    if (!(a > 0.0f)) return (uint8_t)sign;              /* zero or NaN-in -> 0 */
    if (a >= 448.0f) return (uint8_t)(sign | 0x7Eu);    /* saturate to max finite */
    int e;
    frexpf(a, &e);              /* a = m * 2^e, m in [0.5,1) -> exponent is e-1 */
    e -= 1;
    if (e < -6) {                                        /* subnormal */
        const float step = ldexpf(1.0f, -9);
        int m = (int)(a / step + 0.5f);
        if (m > 7) m = 7;
        return (uint8_t)(sign | (uint32_t)m);
    }
    const float step = ldexpf(1.0f, e - 3);
    int m = (int)(a / step + 0.5f) - 8;                  /* strip implicit bit */
    int ee = e + 7;
    if (m > 7) { m = 0; ee += 1; }                       /* rounded up a binade */
    if (ee > 15) return (uint8_t)(sign | 0x7Eu);
    return (uint8_t)(sign | ((uint32_t)ee << 3) | (uint32_t)m);
}

/* ---- the MMA ------------------------------------------------------------ */

__device__ __forceinline__ static void idx_mma_m16n8k32(
        float &d0, float &d1, float &d2, float &d3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1,
        uint32_t sfa, uint32_t sfb) {
#if PULSAR_IDX_MXFP4_MMA
    const uint16_t bid = 0, tid = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {%15,%16}, {%17}, {%18,%19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3),
          "r"(sfa), "h"(bid), "h"(tid),
          "r"(sfb), "h"(bid), "h"(tid));
#else
    /* The PTX-only pass compiles this arm.  Producing ZEROS here would be the
     * exact hazard this codebase keeps finding: a path that silently returns
     * plausible-looking wrong numbers instead of refusing.  Emit NaN so that if
     * this image is ever JITed and run, the scores are visibly broken rather
     * than quietly mis-selecting rows. */
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1; (void)sfa; (void)sfb;
    d0 = d1 = d2 = d3 = __int_as_float(0x7fffffff);
#endif
}

/* ---- Q pre-pack --------------------------------------------------------- */
/* Q is shared by every compressed tile of a token, but the scorer runs one
 * block per (token, 64-comp tile) -- so quantising Q inside the scorer repeats
 * the whole 64x128 amax-scan-and-encode ceil(n_comp/64) times per token.  At
 * n_comp=512 that is 8x redundant scalar work, and it dominated: the first
 * version measured 1.367 ms/launch against the old kernel's 0.550 ms average,
 * at roughly 3 TFLOP/s -- i.e. the tensor cores were idling behind the encode.
 *
 * Hoisting it into its own pass is why Entrpi ships a separate
 * indexer_mxf4_encode_rows_kernel (12.83 ms) next to its scorer (4.94 ms). */
__global__ __launch_bounds__(256, 4)
static void idx_pack_q_kernel(
        uint8_t *__restrict__ qa,            /* [n_tokens][heads][128] e4m3 */
        uint8_t *__restrict__ qsf,           /* [n_tokens][heads][4]   ue8m0 */
        const float *__restrict__ q,
        uint32_t n_tokens) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n_tokens * IDX_HEADS * IDX_KSLABS;
    if (slot >= total) return;
    const uint32_t blk = slot % IDX_KSLABS;
    const uint32_t h   = (slot / IDX_KSLABS) % IDX_HEADS;
    const uint32_t t   = slot / (IDX_KSLABS * IDX_HEADS);

    const float *src = q + ((uint64_t)t * IDX_HEADS + h) * IDX_HEAD_DIM + blk * 32u;
    float amax = 0.0f;
    for (uint32_t i = 0; i < 32u; i++) amax = fmaxf(amax, fabsf(src[i]));
    int se = -127;
    if (amax > 0.0f) { int e; frexpf(amax, &e); se = (e - 1) - 8; }   /* e4m3 emax = 8 */
    int byte = se + 128;                       /* hw applies 2^(byte-128) */
    if (byte < 0) byte = 0;
    if (byte > 255) byte = 255;
    qsf[((uint64_t)t * IDX_HEADS + h) * IDX_KSLABS + blk] = (uint8_t)byte;

    const float inv = ldexpf(1.0f, -se);
    uint8_t *dst = qa + ((uint64_t)t * IDX_HEADS + h) * IDX_HEAD_DIM + blk * 32u;
    for (uint32_t i = 0; i < 32u; i++) dst[i] = idx_f32_to_e4m3(src[i] * inv);
}

/* Two packed bytes (four e2m1 nibbles) -> four byte containers with each nibble
 * at bits [5:2], which is where the measured contract says the MMA reads them. */
__device__ __forceinline__ static uint32_t idx_spread4(uint32_t x) {
    return (((x & 0x000Fu)) | ((x & 0x00F0u) << 4) |
            ((x & 0x0F00u) << 8) | ((x & 0xF000u) << 12)) << 2;
}

/* ---- kernel ------------------------------------------------------------- */

__global__ __launch_bounds__(IDX_THREADS, 2)
static void idx_scores_mxfp4_kernel(
        float *__restrict__ scores,          /* [n_tokens][n_comp] */
        const uint8_t *__restrict__ qa,      /* [n_tokens][heads][128] e4m3 */
        const uint8_t *__restrict__ qsf,     /* [n_tokens][heads][4]   ue8m0 */
        const float *__restrict__ weights,   /* [n_tokens][heads] */
        const uint8_t *__restrict__ comp,    /* [n_comp][68] MXKV-FP4 rows */
        uint32_t n_comp, uint32_t n_tokens, uint32_t pos0,
        uint32_t ratio, float scale, int causal) {
    const uint32_t tile_c = blockIdx.x * IDX_NTILE;
    const uint32_t tok0   = blockIdx.y * IDX_TOKTILE;
    if (tok0 >= n_tokens) return;
    const uint32_t ntok = min(IDX_TOKTILE, n_tokens - tok0);

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t g    = lane >> 2u;        /* groupID  0..7 */
    const uint32_t tig  = lane & 3u;         /* thread-in-group 0..3 */

    /* Two tokens share this block, so the tile is only fully masked when it is
     * masked for the LATER (more permissive) of the two. */
    const uint32_t vis_max = (pos0 + tok0 + ntok - 1u + 1u) / ratio;
    if (causal && tile_c >= vis_max) {
        for (uint32_t i = tid; i < IDX_NTILE * ntok; i += IDX_THREADS) {
            const uint32_t t = tok0 + i / IDX_NTILE;
            const uint32_t comp_i = tile_c + (i % IDX_NTILE);
            if (comp_i < n_comp) scores[(uint64_t)t * n_comp + comp_i] = -INFINITY;
        }
        return;
    }

    /* No S tile.  The first version materialised a [heads x N] f32 tile in
     * shared and swept 64 heads per output column through it -- 16 KB of smem
     * and a strided, bank-conflicting, 64-deep dependent chain per column.  It
     * also capped N at 64, because [64 x 128] f32 would have been 64 KB.
     * Reducing in-register over the warp's own rows and combining warps through
     * a 2 KB partial buffer removes both problems, which is what lets N double. */
    __shared__ uint8_t sA[IDX_TOKTILE * IDX_HEADS * IDX_ASTRIDE];    /* ~17 KB e4m3 */
    __shared__ uint8_t sB[IDX_NTILE * IDX_BSTRIDE];          /* ~8.7 KB packed nibbles */
    __shared__ uint8_t sSFA[IDX_TOKTILE * IDX_HEADS * IDX_KSLABS];   /* 512 B */
    __shared__ uint8_t sSFB[IDX_NTILE * IDX_KSLABS];        /* 512 B */
    __shared__ float   sPart[IDX_THREADS / 32u][IDX_NTILE]; /*  2 KB */

    /* ---- stage Q: straight copy of the pre-packed bytes ------------------ */
    {
        const uint32_t sbytes = ntok * IDX_HEADS * IDX_KSLABS;
        const uint8_t *qsrc = qa  + (uint64_t)tok0 * IDX_HEADS * IDX_HEAD_DIM;
        const uint8_t *ssrc = qsf + (uint64_t)tok0 * IDX_HEADS * IDX_KSLABS;
        /* row-wise, because the shared stride is padded and the global one is not */
        for (uint32_t slot = tid; slot < IDX_TOKTILE * IDX_HEADS * 32u; slot += IDX_THREADS) {
            const uint32_t r = slot / 32u;              /* row = token*heads + head */
            const uint32_t j = slot % 32u;              /* 32 x uint32 = 128 B */
            uint32_t *dst = (uint32_t *)(sA + r * IDX_ASTRIDE);
            dst[j] = (r < ntok * IDX_HEADS)
                   ? *(const uint32_t *)(qsrc + r * IDX_HEAD_DIM + j * 4u)
                   : 0u;
        }
        for (uint32_t i = tid; i < sbytes; i += IDX_THREADS)
            sSFA[i] = ssrc[i];
    }

    /* ---- stage K: raw copy; the nibble spread moves to the MMA load --------
     * Keeping the rows PACKED in shared halves this tile (16 KB -> 8 KB) and
     * halves B's shared-read traffic, and the block's smem drops 37 KB -> 29 KB,
     * which is what limits how many blocks an SM can hold.  Expanding two bytes
     * into four byte-containers is ~7 ALU ops at the point of use, against a
     * shared-memory read we no longer have to make. */
    for (uint32_t slot = tid; slot < IDX_NTILE * 16u; slot += IDX_THREADS) {
        const uint32_t c = slot / 16u;
        const uint32_t j = slot % 16u;                  /* 16 x uint32 = 64 B/row */
        const uint32_t comp_i = tile_c + c;
        uint32_t *dst = (uint32_t *)(sB + c * IDX_BSTRIDE);
        if (comp_i >= n_comp) {
            dst[j] = 0u;
            if (j < IDX_KSLABS) sSFB[c * IDX_KSLABS + j] = 0;
            continue;
        }
        const uint8_t *row = comp + (uint64_t)comp_i * PULSAR_MXKV_FP4_ROWBYTES(128u);
        dst[j] = *(const uint32_t *)(row + j * 4u);
        if (j < IDX_KSLABS) {
            const int sfb = (int)row[64u + j] - 1;   /* +1 bias, -2 for the 4x */
            sSFB[c * IDX_KSLABS + j] = (uint8_t)(sfb < 0 ? 0 : sfb);
        }
    }
    __syncthreads();

    /* ---- GEMM + fused head reduction ------------------------------------ */
    /* warp w owns m-tile w (heads 16w..16w+15) and sweeps every n-tile. */
    const uint32_t wtok   = warp / 4u;                 /* which of the 2 tokens */
    const uint32_t m_local = (warp % 4u) * 16u;         /* head offset within it */
    const uint32_t m_base  = wtok * IDX_HEADS + m_local;
    const uint32_t token_w = tok0 + wtok;
    const float *wrow = weights + (uint64_t)(token_w < n_tokens ? token_w : tok0) * IDX_HEADS;
    const float wg0 = (wtok < ntok) ? wrow[m_local + g]      : 0.0f;
    const float wg1 = (wtok < ntok) ? wrow[m_local + g + 8u] : 0.0f;

    /* Register-blocked over N: the A fragment (16 B/lane) is identical for every
     * n-tile at a given k-slab, so loading it once and firing NB MMAs against NB
     * B fragments (8 B/lane each) cuts shared-memory traffic per MMA from
     * 24 B/lane to (16 + 8*NB)/NB.  At NB=4 that is 12 B/lane, a 2x reduction --
     * and smem traffic, not MMA issue, is what was holding this at ~8 TFLOP/s
     * (393 KB of smem reads per block against 2.1 M MACs). */
    /* NB=4 measured best; NB=8 is 0.336 vs 0.330, i.e. the extra register
     * pressure cancels the traffic it saves.  minBlocksPerMultiprocessor is 2:
     * 4 was unsatisfiable at ~29 KB of smem per block and only distorted
     * register allocation (0.330 at 2, 0.336 at 1, 0.331 at 3). */
    #define IDX_NB 4u
    for (uint32_t nt0 = 0; nt0 < IDX_NTILE / 8u; nt0 += IDX_NB) {
        float d[IDX_NB][4];
        #pragma unroll
        for (uint32_t j = 0; j < IDX_NB; j++) { d[j][0] = d[j][1] = d[j][2] = d[j][3] = 0.f; }

        for (uint32_t s = 0; s < IDX_KSLABS; s++) {
            const uint32_t k0 = s * 32u;
            const uint8_t *ar0 = sA + (m_base + g)      * IDX_ASTRIDE + k0 + tig * 4u;
            const uint8_t *ar1 = sA + (m_base + g + 8u) * IDX_ASTRIDE + k0 + tig * 4u;
            const uint32_t a0 = *(const uint32_t *)ar0;
            const uint32_t a1 = *(const uint32_t *)ar1;
            const uint32_t a2 = *(const uint32_t *)(ar0 + 16u);
            const uint32_t a3 = *(const uint32_t *)(ar1 + 16u);
            const uint32_t m_for_sfa = m_base + ((tig == 1u) ? (g + 8u) : g);
            const uint32_t sfa = sSFA[m_for_sfa * IDX_KSLABS + s];

            #pragma unroll
            for (uint32_t j = 0; j < IDX_NB; j++) {
                const uint32_t n_base = (nt0 + j) * 8u;
                const uint8_t *br = sB + (n_base + g) * IDX_BSTRIDE
                                       + (k0 >> 1) + tig * 2u;
                const uint32_t b0 = idx_spread4(*(const uint16_t *)br);
                const uint32_t b1 = idx_spread4(*(const uint16_t *)(br + 8u));
                const uint32_t sfb = sSFB[(n_base + g) * IDX_KSLABS + s];
                idx_mma_m16n8k32(d[j][0], d[j][1], d[j][2], d[j][3],
                                 a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
        }

        #pragma unroll
        for (uint32_t j = 0; j < IDX_NB; j++) {
            const uint32_t n_base = (nt0 + j) * 8u;
            float c0 = (d[j][0] > 0.f ? d[j][0] * wg0 : 0.f) + (d[j][2] > 0.f ? d[j][2] * wg1 : 0.f);
            float c1 = (d[j][1] > 0.f ? d[j][1] * wg0 : 0.f) + (d[j][3] > 0.f ? d[j][3] * wg1 : 0.f);
            #pragma unroll
            for (int m = 4; m < 32; m <<= 1) {
                c0 += __shfl_xor_sync(0xffffffffu, c0, m);
                c1 += __shfl_xor_sync(0xffffffffu, c1, m);
            }
            if (g == 0u) {
                sPart[warp][n_base + tig * 2u]      = c0;
                sPart[warp][n_base + tig * 2u + 1u] = c1;
            }
        }
    }
    #undef IDX_NB
    __syncthreads();

    /* ---- combine the warps' head-groups, then scale and mask ------------ */
    for (uint32_t i = tid; i < IDX_NTILE * ntok; i += IDX_THREADS) {
        const uint32_t tl = i / IDX_NTILE;              /* 0 or 1 */
        const uint32_t c  = i % IDX_NTILE;
        const uint32_t comp_i = tile_c + c;
        if (comp_i >= n_comp) continue;
        float acc = 0.f;
        #pragma unroll
        for (uint32_t w = 0; w < 4u; w++) acc += sPart[tl * 4u + w][c];
        float out = acc * scale;
        const uint32_t t = tok0 + tl;
        if (causal && comp_i >= ((pos0 + t + 1u) / ratio)) out = -INFINITY;
        scores[(uint64_t)t * n_comp + comp_i] = out;
    }
}

/* ---- launcher ----------------------------------------------------------- */

extern "C" int pulsar_gpu_indexer_scores_mxfp4(
        float *scores, const float *q, const float *weights, const void *comp,
        uint32_t n_comp, uint32_t n_tokens, uint32_t pos0,
        uint32_t n_head, uint32_t head_dim, uint32_t ratio,
        float scale, int causal, int fp4) {
    /* Shape gate, evaluated once per launch -- never per token or per layer. */
    if (!scores || !q || !weights || !comp) return 0;
    if (!fp4) return 0;                       /* packed rows only */
    if (n_head != IDX_HEADS || head_dim != IDX_HEAD_DIM) return 0;
    if (n_comp == 0u || n_tokens == 0u) return 0;

    /* Pre-pack Q once per token; the scorer then only copies bytes. */
    const uint64_t qa_bytes  = (uint64_t)n_tokens * IDX_HEADS * IDX_HEAD_DIM;
    const uint64_t qsf_bytes = (uint64_t)n_tokens * IDX_HEADS * IDX_KSLABS;
    uint8_t *qa = (uint8_t *)cuda_tmp_alloc(qa_bytes + qsf_bytes, "indexer mxfp4 Q pack");
    if (!qa) return 0;
    uint8_t *qsf = qa + qa_bytes;

    const uint32_t pack_total = n_tokens * IDX_HEADS * IDX_KSLABS;
    idx_pack_q_kernel<<<(pack_total + 255u) / 256u, 256>>>(qa, qsf, q, n_tokens);
    if (!cuda_ok(cudaGetLastError(), "indexer mxfp4 Q pack launch")) return 0;

    dim3 grid((n_comp + IDX_NTILE - 1u) / IDX_NTILE,
              (n_tokens + IDX_TOKTILE - 1u) / IDX_TOKTILE, 1);
    idx_scores_mxfp4_kernel<<<grid, IDX_THREADS>>>(
        scores, qa, qsf, weights, (const uint8_t *)comp,
        n_comp, n_tokens, pos0, ratio, scale, causal);
    return cuda_ok(cudaGetLastError(), "indexer scores mxfp4 launch");
}

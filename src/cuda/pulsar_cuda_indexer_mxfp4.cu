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
#define IDX_NTILE     64u          /* compressed rows per block */
#define IDX_KSLABS    (IDX_HEAD_DIM / 32u)   /* 4 */
#define IDX_THREADS   128u         /* 4 warps */

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

/* ---- kernel ------------------------------------------------------------- */

__global__ __launch_bounds__(IDX_THREADS, 4)
static void idx_scores_mxfp4_kernel(
        float *__restrict__ scores,          /* [n_tokens][n_comp] */
        const uint8_t *__restrict__ qa,      /* [n_tokens][heads][128] e4m3 */
        const uint8_t *__restrict__ qsf,     /* [n_tokens][heads][4]   ue8m0 */
        const float *__restrict__ weights,   /* [n_tokens][heads] */
        const uint8_t *__restrict__ comp,    /* [n_comp][68] MXKV-FP4 rows */
        uint32_t n_comp, uint32_t n_tokens, uint32_t pos0,
        uint32_t ratio, float scale, int causal) {
    const uint32_t tile_c = blockIdx.x * IDX_NTILE;
    const uint32_t token  = blockIdx.y;
    if (token >= n_tokens) return;

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t g    = lane >> 2u;        /* groupID  0..7 */
    const uint32_t tig  = lane & 3u;         /* thread-in-group 0..3 */

    /* Causal early-out: identical rule to the kernel being replaced. */
    if (causal) {
        const uint32_t visible = (pos0 + token + 1u) / ratio;
        if (tile_c >= visible) {
            for (uint32_t c = tid; c < IDX_NTILE; c += IDX_THREADS) {
                const uint32_t comp_i = tile_c + c;
                if (comp_i < n_comp) scores[(uint64_t)token * n_comp + comp_i] = -INFINITY;
            }
            return;
        }
    }

    __shared__ uint8_t sA[IDX_HEADS * IDX_HEAD_DIM];        /*  8 KB e4m3      */
    __shared__ uint8_t sB[IDX_NTILE * IDX_HEAD_DIM];        /*  8 KB e2m1<<2   */
    __shared__ uint8_t sSFA[IDX_HEADS * IDX_KSLABS];        /*  256 B          */
    __shared__ uint8_t sSFB[IDX_NTILE * IDX_KSLABS];        /*  256 B          */
    __shared__ float   sS[IDX_HEADS * IDX_NTILE];           /* 16 KB f32       */

    /* ---- stage Q: straight copy of the pre-packed bytes ------------------ */
    {
        const uint8_t *qsrc = qa  + (uint64_t)token * IDX_HEADS * IDX_HEAD_DIM;
        const uint8_t *ssrc = qsf + (uint64_t)token * IDX_HEADS * IDX_KSLABS;
        for (uint32_t i = tid * 4u; i < IDX_HEADS * IDX_HEAD_DIM; i += IDX_THREADS * 4u)
            *(uint32_t *)(sA + i) = *(const uint32_t *)(qsrc + i);
        for (uint32_t i = tid; i < IDX_HEADS * IDX_KSLABS; i += IDX_THREADS)
            sSFA[i] = ssrc[i];
    }

    /* ---- stage K: spread packed nibbles to bits [5:2], rebias scales ----- */
    for (uint32_t slot = tid; slot < IDX_NTILE * 32u; slot += IDX_THREADS) {
        const uint32_t c = slot / 32u;
        const uint32_t j = slot % 32u;                 /* 32 bytes of 64 per pass */
        const uint32_t comp_i = tile_c + c;
        uint8_t *dst = sB + c * IDX_HEAD_DIM;
        if (comp_i >= n_comp) {
            dst[j * 2u] = 0; dst[j * 2u + 1u] = 0;
            dst[j * 2u + 64u] = 0; dst[j * 2u + 65u] = 0;
            if (j < IDX_KSLABS) sSFB[c * IDX_KSLABS + j] = 0;
            continue;
        }
        const uint8_t *row = comp + (uint64_t)comp_i * PULSAR_MXKV_FP4_ROWBYTES(128u);
        const uint8_t b0 = row[j];
        const uint8_t b1 = row[j + 32u];
        dst[j * 2u]        = (uint8_t)(((uint32_t)b0 & 0xFu) << 2);
        dst[j * 2u + 1u]   = (uint8_t)(((uint32_t)b0 >> 4) << 2);
        dst[j * 2u + 64u]  = (uint8_t)(((uint32_t)b1 & 0xFu) << 2);
        dst[j * 2u + 65u]  = (uint8_t)(((uint32_t)b1 >> 4) << 2);
        if (j < IDX_KSLABS) {
            const int s = (int)row[64u + j] - 1;        /* +1 bias, -2 for the 4x */
            sSFB[c * IDX_KSLABS + j] = (uint8_t)(s < 0 ? 0 : s);
        }
    }
    __syncthreads();

    /* ---- GEMM: M=64 heads, N=64 comp, K=128 ----------------------------- */
    /* warp w owns m-tile w (heads 16w..16w+15) and sweeps all 8 n-tiles. */
    const uint32_t m_base = warp * 16u;
    for (uint32_t nt = 0; nt < IDX_NTILE / 8u; nt++) {
        const uint32_t n_base = nt * 8u;
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

        for (uint32_t s = 0; s < IDX_KSLABS; s++) {
            const uint32_t k0 = s * 32u;
            /* A: rows m_base+g and m_base+g+8; k = tig*4 + {0..3} and +16 */
            const uint8_t *ar0 = sA + (m_base + g)      * IDX_HEAD_DIM + k0 + tig * 4u;
            const uint8_t *ar1 = sA + (m_base + g + 8u) * IDX_HEAD_DIM + k0 + tig * 4u;
            const uint32_t a0 = *(const uint32_t *)ar0;
            const uint32_t a1 = *(const uint32_t *)ar1;
            const uint32_t a2 = *(const uint32_t *)(ar0 + 16u);
            const uint32_t a3 = *(const uint32_t *)(ar1 + 16u);
            /* B: col n_base+g; k = tig*4 + {0..3} and +16 */
            const uint8_t *br = sB + (n_base + g) * IDX_HEAD_DIM + k0 + tig * 4u;
            const uint32_t b0 = *(const uint32_t *)br;
            const uint32_t b1 = *(const uint32_t *)(br + 16u);
            /* SFA: row m <- lane 4*(m%8)+(m/8).  Lane t therefore supplies the
             * scale for row (t&3)==1 ? g+8 : g; lanes with (t&3)>=2 are unused
             * but must still pass something defined. */
            const uint32_t m_for_sfa = m_base + ((tig == 1u) ? (g + 8u) : g);
            const uint32_t sfa = sSFA[m_for_sfa * IDX_KSLABS + s];
            /* SFB: col n <- lane 4n, i.e. lane 4g supplies col g. */
            const uint32_t sfb = sSFB[(n_base + g) * IDX_KSLABS + s];
            idx_mma_m16n8k32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb);
        }

        /* C layout: rows g (d0,d1) and g+8 (d2,d3), cols tig*2 + {0,1}. */
        const uint32_t c0 = n_base + tig * 2u;
        sS[(m_base + g)      * IDX_NTILE + c0]      = d0;
        sS[(m_base + g)      * IDX_NTILE + c0 + 1u] = d1;
        sS[(m_base + g + 8u) * IDX_NTILE + c0]      = d2;
        sS[(m_base + g + 8u) * IDX_NTILE + c0 + 1u] = d3;
    }
    __syncthreads();

    /* ---- epilogue: sum_h ReLU(S) * w[t,h], then scale + causal mask ------ */
    const float *wrow = weights + (uint64_t)token * IDX_HEADS;
    for (uint32_t c = tid; c < IDX_NTILE; c += IDX_THREADS) {
        const uint32_t comp_i = tile_c + c;
        if (comp_i >= n_comp) continue;
        float acc = 0.0f;
        for (uint32_t h = 0; h < IDX_HEADS; h++) {
            const float v = sS[h * IDX_NTILE + c];
            if (v > 0.0f) acc += v * wrow[h];
        }
        float out = acc * scale;
        if (causal) {
            const uint32_t visible = (pos0 + token + 1u) / ratio;
            if (comp_i >= visible) out = -INFINITY;
        }
        scores[(uint64_t)token * n_comp + comp_i] = out;
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

    dim3 grid((n_comp + IDX_NTILE - 1u) / IDX_NTILE, n_tokens, 1);
    idx_scores_mxfp4_kernel<<<grid, IDX_THREADS>>>(
        scores, qa, qsf, weights, (const uint8_t *)comp,
        n_comp, n_tokens, pos0, ratio, scale, causal);
    return cuda_ok(cudaGetLastError(), "indexer scores mxfp4 launch");
}

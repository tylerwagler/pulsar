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
 * Operand choice:        e2m1 for BOTH operands since L090.4 -- Q arrives as
 *                        the producer's packed FP4, so staging is a nibble
 *                        spread, not an encode.  (tests/idx_quant_fidelity.cc
 *                        measured E4M3-for-Q against unquantised Q, before the
 *                        producer's FP4 crush made the choice moot.)
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
#include <cuda_fp8.h>
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
/* Shared B is stored LANE-MAJOR, not row-major: 64 B per compressed row, laid
 * out as [tig][slab][half] halfwords.  The MMA's B fragment wants, for lane
 * group position tig, the halfword at row byte (slab*16 + half*8 + tig*2) --
 * eight scattered 2-byte reads per row across the four slabs.  Permuting once
 * at staging time makes those eight halfwords contiguous, so the whole k loop's
 * B operand arrives in ONE 16-byte LDS.  No padding: the warp's 32 lanes read
 * 32 consecutive 16 B chunks, which is already conflict-free. */
#define IDX_BPSTRIDE  64u
/* ONE token staged at a time.  With K resident across the group there is no
 * reuse argument left for staging two, and sA is the biggest smem consumer:
 * dropping it 16.9 KB -> 8.5 KB takes the block from ~31 KB to ~20 KB, i.e.
 * from 3 blocks/SM (50% occupancy) to 5 (83%).  Occupancy is the one limit the
 * device numbers actually confirmed. */
#define IDX_TOKTILE   1u
/* Tokens PER BLOCK.  The compressed tile is identical for every token, so it is
 * staged once and every token in the group is swept through it: B staging and
 * its global traffic drop by IDX_TOKGROUP/IDX_TOKTILE.  Entrpi's scorer launches
 * grid=4x256 block=512 for a whole 4096-token chunk, i.e. one launch per layer
 * against our eight -- amortising the tile setup over far more math is the
 * structural difference their geometry points at. */
#define IDX_TOKGROUP  8u
#define IDX_THREADS   256u         /* 8 warps: 4 per token */

/* (The E4M3 encode helpers idx_f32_to_e4m3 / idx_amax_shift lived here until
 * the e2m1 operand switch: Q staging is a nibble spread now and this TU
 * encodes nothing.) */

/* ---- the MMA ------------------------------------------------------------ */

__device__ __forceinline__ static void idx_mma_m16n8k32(
        float &d0, float &d1, float &d2, float &d3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1,
        uint32_t sfa, uint32_t sfb) {
#if PULSAR_IDX_MXFP4_MMA
    const uint16_t bid = 0, tid = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        /* D and C are the SAME registers: "+f" read-write operands, referenced
         * twice.  Declaring them as separate "=f" outputs and "f" inputs let the
         * compiler allocate two accumulator sets and copy between them on every
         * MMA -- 16 accumulators x 4 k-slabs of pure register churn per n-block. */
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
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

/* ---- Q pre-stage -------------------------------------------------------- */
/* Q is shared by every compressed tile of a token, but the scorer runs one
 * block per (token, 64-comp tile) -- so preparing Q inside the scorer would
 * repeat the work ceil(n_comp/64) times per token; it stays hoisted into its
 * own pass (the same reason Entrpi ships a separate encode-rows kernel).
 * The pass used to be an ENCODE (f32 -> amax scan -> e4m3), and its cost was
 * measured to dominate the launch (intercept 0.270 ms = 82% at n_comp=512,
 * slope 1.23e-4 ms/row = the GEMM).  Since the e2m1 switch it is a nibble
 * SPREAD of the producer's packed rows -- 68 B read, 132 B written per row,
 * no arithmetic -- so the old encode-cost analysis is retired with it. */

/* Two packed bytes (four e2m1 nibbles) -> four byte containers with each nibble
 * at bits [5:2], which is where the measured contract says the MMA reads them. */
__device__ __forceinline__ static uint32_t idx_spread4(uint32_t x) {
    return (((x & 0x000Fu)) | ((x & 0x00F0u) << 4) |
            ((x & 0x0F00u) << 8) | ((x & 0xF000u) << 12)) << 2;
}

/* Q operand staging, native-format (L090.4 follow-up, Tyler's rule: a value
 * that is FP4 travels as FP4).  The producer's packed row already holds the
 * E2M1 codes and E8M0 scales; the MMA takes e2m1 operands in 8-bit containers
 * (nibble at bits [5:2], decoding 4x nominal -- the measured contract above).
 * So "staging" is a nibble spread plus a scale-byte rebias:
 *   sf_hw = s       (no rebias -- see the K-side note),
 * the SAME treatment the K side applies to its identical rows.  Nothing is
 * decoded and nothing is re-encoded: no amax, no rounding, no second scale
 * derivation anywhere in the scorer.
 * (The old e4m3 staging decoded the row to f32 and re-encoded; E4M3-for-Q was
 * measured in tests/idx_quant_fidelity.cc against UNQUANTISED Q, a premise the
 * producer's FP4 crush has since made moot -- on-grid values encode exactly
 * either way, so this change is value-identical and strictly simpler.) */
__global__ __launch_bounds__(256, 4)
static void idx_expand_q_kernel(
        uint8_t *__restrict__ qa,            /* [n_tokens][heads][128] e2m1-in-byte */
        uint8_t *__restrict__ qsf,           /* [n_tokens][heads][4]   ue8m0 (rebiased) */
        const pulsar_mxkv_pack_t *__restrict__ q,   /* packed 68 B rows */
        uint32_t n_rows) {                   /* n_tokens * IDX_HEADS */
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row  = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    if (row >= n_rows) return;
    const uint8_t *r = (const uint8_t *)q + (uint64_t)row * (IDX_HEAD_DIM / 2u + IDX_KSLABS);
    const uint32_t two = *(const uint16_t *)(r + lane * 2u);   /* 4 nibbles, dims 4*lane.. */
    ((uint32_t *)(qa + (uint64_t)row * IDX_HEAD_DIM))[lane] = idx_spread4(two);
    if ((lane & 7u) == 0u) {
        /* NO rebias: sf_hw == the stored byte.  See the K-side note below --
         * measured, the e2m1 x e2m1 container decodes plain and the scale bias
         * is the spec's 127, so the stored ue8m0 byte passes through. */
        qsf[(uint64_t)row * IDX_KSLABS + (lane >> 3u)] =
            r[(IDX_HEAD_DIM / 2u) + (lane >> 3u)];
    }
}

/* The D5 E4M3 round-trip (idx_q_e4m3_roundtrip_kernel + its wrapper) lived
 * here until L090.4.  It forced all three scorers onto one Q operand format
 * by round-tripping the f32 container through this tier's E4M3 quantisation.
 * The producer now emits Q as packed E2M1 rows and every scorer decodes the
 * same bytes, so the unification is structural and the round-trip -- an
 * identity on values already crushed to the FP4 grid -- had nothing left to
 * do. */

/* ---- kernel ------------------------------------------------------------- */

/* Occupancy target.  ncu on the 87-register build: LSU 37%, ALU 20%, tensor
 * 17%, IPC 0.38 of 1.0 -- nothing saturated, so the kernel is latency-bound and
 * wants more warps resident, not fewer instructions.  87 regs x 256 threads =
 * 22272 of the SM's 65536, which caps it at 2 blocks (16 of 48 warps, 33%).
 * Raising minBlocksPerMultiprocessor forces the register budget down: 3 blocks
 * needs <=85 regs, 4 needs <=64.  Both now fit in smem, which 29 KB/block did
 * not -- the B permute took the block to 19.5 KB, so 4 x 19.5 = 78 KB < 100 KB.
 *
 * MEASURED, and the two levers only pay together (ms per compressed row):
 *                     2 blocks/SM   3 blocks/SM
 *   B row-major        1.271e-4      1.3125e-4
 *   B lane-major       1.323e-4      1.198e-4
 * Either change alone is a regression.  Row-major B is LSU-bound, so extra
 * warps only add contention on the pipe that is already the busiest; cutting
 * the B loads is what turns the occupancy into a win.  4 and 5 blocks/SM both
 * lose again -- ptxas spills (16 B and 32 B of stack) and the slope goes to
 * 1.376e-4 and 1.497e-4. */
#ifndef IDX_MINBLK
#define IDX_MINBLK 3
#endif
__global__ __launch_bounds__(IDX_THREADS, IDX_MINBLK)
static void idx_scores_mxfp4_kernel(
        float *__restrict__ scores,          /* [n_tokens][n_comp] */
        const uint8_t *__restrict__ qa,      /* [n_tokens][heads][128] e2m1-in-byte */
        const uint8_t *__restrict__ qsf,     /* [n_tokens][heads][4]   ue8m0 */
        const float *__restrict__ weights,   /* [n_tokens][heads] */
        const uint8_t *__restrict__ comp,    /* [n_comp][68] MXKV-FP4 rows */
        uint32_t n_comp, uint32_t n_tokens, uint32_t pos0,
        uint32_t ratio, float scale, int causal) {
    const uint32_t tile_c = blockIdx.x * IDX_NTILE;
    const uint32_t tok_base = blockIdx.y * IDX_TOKGROUP;
    if (tok_base >= n_tokens) return;
    const uint32_t ngroup = min(IDX_TOKGROUP, n_tokens - tok_base);

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t g    = lane >> 2u;        /* groupID  0..7 */
    const uint32_t tig  = lane & 3u;         /* thread-in-group 0..3 */

    /* Two tokens share this block, so the tile is only fully masked when it is
     * masked for the LATER (more permissive) of the two. */
    const uint32_t vis_max = (pos0 + tok_base + ngroup) / ratio;
    if (causal && tile_c >= vis_max) {
        for (uint32_t i = tid; i < IDX_NTILE * ngroup; i += IDX_THREADS) {
            const uint32_t t = tok_base + i / IDX_NTILE;
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
    __shared__ __align__(16) uint8_t sB[IDX_NTILE * IDX_BPSTRIDE];   /* 8 KB packed nibbles */
    __shared__ uint8_t sSFA[IDX_TOKTILE * IDX_HEADS * IDX_KSLABS];   /* 512 B */
    __shared__ uint8_t sSFB[IDX_NTILE * IDX_KSLABS];        /* 512 B */
    __shared__ float   sPart[4][IDX_NTILE];                /*  2 KB: 4 m-tiles */

    /* ---- stage K ONCE for the whole token group -------------------------- */
    /* ---- stage K: raw copy; the nibble spread moves to the MMA load --------
     * Keeping the rows PACKED in shared halves this tile (16 KB -> 8 KB) and
     * halves B's shared-read traffic, and the block's smem drops 37 KB -> 29 KB,
     * which is what limits how many blocks an SM can hold.  Expanding two bytes
     * into four byte-containers is ~7 ALU ops at the point of use, against a
     * shared-memory read we no longer have to make. */
    /* Halfword-granular so the permute can be applied on the way in.  Source
     * halfword u of a row decomposes as u = slab*8 + half*4 + tig, and lands at
     * lane-major position tig*8 + slab*2 + half. */
    for (uint32_t slot = tid; slot < IDX_NTILE * 32u; slot += IDX_THREADS) {
        const uint32_t c = slot >> 5u;                  /* 32 halfwords = 64 B/row */
        const uint32_t u = slot & 31u;
        const uint32_t sl = u >> 3u, rem = u & 7u, hf = rem >> 2u, tg = rem & 3u;
        uint16_t *dst = (uint16_t *)(sB + c * IDX_BPSTRIDE) + tg * 8u + sl * 2u + hf;
        const uint32_t comp_i = tile_c + c;
        if (comp_i >= n_comp) {
            *dst = 0u;
            if (u < IDX_KSLABS) sSFB[c * IDX_KSLABS + u] = 0;
            continue;
        }
        const uint8_t *row = comp + (uint64_t)comp_i * PULSAR_MXKV_FP4_ROWBYTES(128u);
        *dst = *(const uint16_t *)(row + u * 2u);
        if (u < IDX_KSLABS) {
            /* NO rebias: the stored ue8m0 byte IS the hardware scale factor.
             *
             * This was `- 1`, documented as "+1 for the 128-vs-127 ue8m0 bias,
             * -2 for the nibble<<2 container's 4x".  Both of those are real
             * effects and both were measured -- but in idx_mxfp4_probe's
             * PHASE0e, which used an e4m3 A-side operand.  This kernel is
             * e2m1 x e2m1 at scale_vec::1X, where the bits[5:2] container
             * decodes PLAIN and the bias is the spec's 127.  The two wrong
             * corrections nearly cancelled, leaving every score at exactly
             * storage/4 -- uniform, so it looked like a working kernel, and
             * invisible downstream because these scores feed only a top-k and
             * a uniform positive scale cannot reorder a ranking.
             *
             * Caught 2026-09-01 by repairing this tier's correctness gate,
             * which had not compiled since f9fbafe (L137).  The gate now pins
             * the relationship at 1.0 and mutation-fails if it drifts. */
            sSFB[c * IDX_KSLABS + u] = row[64u + u];
        }
    }
    __syncthreads();

    /* ---- sweep the group's tokens through the resident K tile ------------ */
    for (uint32_t tp = 0; tp < ngroup; tp += IDX_TOKTILE) {
    const uint32_t tok0 = tok_base + tp;
    const uint32_t ntok = min(IDX_TOKTILE, n_tokens - tok0);
    if (tp) __syncthreads();          /* previous iteration's sPart/sA consumed */

    /* ---- stage Q: straight copy of the pre-packed bytes ------------------ */
    {
        const uint32_t sbytes = ntok * IDX_HEADS * IDX_KSLABS;
        const uint8_t *qsrc = qa  + (uint64_t)tok0 * IDX_HEADS * IDX_HEAD_DIM;
        const uint8_t *ssrc = qsf + (uint64_t)tok0 * IDX_HEADS * IDX_KSLABS;
        /* row-wise, because the shared stride is padded and the global one is not */
        for (uint32_t slot = tid; slot < IDX_HEADS * 32u; slot += IDX_THREADS) {
            const uint32_t r = slot / 32u;              /* head */
            const uint32_t j = slot % 32u;              /* 32 x uint32 = 128 B */
            uint32_t *dst = (uint32_t *)(sA + r * IDX_ASTRIDE);
            dst[j] = *(const uint32_t *)(qsrc + r * IDX_HEAD_DIM + j * 4u);
        }
        for (uint32_t i = tid; i < sbytes; i += IDX_THREADS)
            sSFA[i] = ssrc[i];
    }

    __syncthreads();

    /* ---- GEMM + fused head reduction ------------------------------------ */
    /* warp w owns m-tile w (heads 16w..16w+15) and sweeps every n-tile. */
    /* 8 warps over a 64-row M (4 m-tiles) x 128-col N: warp = (n-half, m-tile). */
    const uint32_t warp_m = warp & 3u;                  /* m-tile 0..3 */
    const uint32_t warp_n = warp >> 2u;                 /* which half of N */
    const uint32_t m_base = warp_m * 16u;
    const float *wrow = weights + (uint64_t)tok0 * IDX_HEADS;
    const float wg0 = wrow[m_base + g];
    const float wg1 = wrow[m_base + g + 8u];

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
    #ifndef IDX_NB
    #define IDX_NB 4u
    #endif
    const uint32_t nt_lo = warp_n * (IDX_NTILE / 16u);
    for (uint32_t nt0 = nt_lo; nt0 < nt_lo + IDX_NTILE / 16u; nt0 += IDX_NB) {
        float d[IDX_NB][4];
        #pragma unroll
        for (uint32_t j = 0; j < IDX_NB; j++) { d[j][0] = d[j][1] = d[j][2] = d[j][3] = 0.f; }

        /* B and its scales are now loop-invariant across the k slabs: the
         * lane-major layout puts all four slabs of a row in one 16 B chunk, and
         * the four scale bytes in one uint32.  So B costs 1 LDS per n-tile for
         * the whole k loop instead of 8, and its scales 1 instead of 4 -- the
         * warp goes from 17 LDS per slab per 4 MMAs (4.25 per MMA) to 4+1 for A
         * and 2 per n-tile amortised over all four slabs.  ncu put L1/TEX at 69%
         * with L2 at 10% and named an L1TEX scoreboard stall as 36% of the
         * 12.9 cycles between issues, so this pipe, not the MMA, was the limit. */
        uint4 braw[IDX_NB];
        uint32_t sfb4[IDX_NB];
        #pragma unroll
        for (uint32_t j = 0; j < IDX_NB; j++) {
            const uint32_t n_base = (nt0 + j) * 8u;
            braw[j] = *(const uint4 *)(sB + (n_base + g) * IDX_BPSTRIDE + tig * 16u);
            sfb4[j] = *(const uint32_t *)(sSFB + (n_base + g) * IDX_KSLABS);
        }

        /* Software-pipelined within the slab: issue ALL operand fetches, then
         * all MMAs.  Previously each j fetched its B fragment immediately before
         * its own MMA, so every MMA sat behind a shared-memory read and the warp
         * had one load in flight at a time.  At 48 SMs / 50% occupancy the
         * kernel was measured at ~74 cycles per MMA per SM -- ~1.4% of issue
         * rate -- i.e. latency-bound on exactly this chain, not on operand
         * volume.  Batching the fetches gives the scheduler NB independent loads
         * to overlap against the MMA chain, and unrolling the slab loop lets it
         * hoist across slabs too. */
        #pragma unroll
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

            /* s is a compile-time constant in every unrolled copy, so the word
             * select folds and braw stays in registers. */
            #define IDX_BSEL(v, i) ((i) == 0u ? (v).x : (i) == 1u ? (v).y : \
                                    (i) == 2u ? (v).z : (v).w)
            uint32_t bb0[IDX_NB], bb1[IDX_NB], sfbv[IDX_NB];
            #pragma unroll
            for (uint32_t j = 0; j < IDX_NB; j++) {
                const uint32_t bw = IDX_BSEL(braw[j], s);
                bb0[j]  = idx_spread4(bw & 0xFFFFu);     /* half 0 */
                bb1[j]  = idx_spread4(bw >> 16);         /* half 1 */
                sfbv[j] = (sfb4[j] >> (s * 8u)) & 0xFFu;
            }
            #undef IDX_BSEL
            #pragma unroll
            for (uint32_t j = 0; j < IDX_NB; j++)
                idx_mma_m16n8k32(d[j][0], d[j][1], d[j][2], d[j][3],
                                 a0, a1, a2, a3, bb0[j], bb1[j], sfa, sfbv[j]);
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
                sPart[warp_m][n_base + tig * 2u]      = c0;
                sPart[warp_m][n_base + tig * 2u + 1u] = c1;
            }
        }
    }
    /* left defined: the sweep sets it from the build line */
    __syncthreads();

    /* ---- combine the warps' head-groups, then scale and mask ------------ */
    for (uint32_t c = tid; c < IDX_NTILE; c += IDX_THREADS) {
        const uint32_t comp_i = tile_c + c;
        if (comp_i >= n_comp) continue;
        float acc = 0.f;
        #pragma unroll
        for (uint32_t w = 0; w < 4u; w++) acc += sPart[w][c];
        float out = acc * scale;
        if (causal && comp_i >= ((pos0 + tok0 + 1u) / ratio)) out = -INFINITY;
        scores[(uint64_t)tok0 * n_comp + comp_i] = out;
    }
    }   /* token group */
}

/* ---- launcher ----------------------------------------------------------- */

/* No extern "C": pulsar_gpu.h declares the whole backend seam with plain
 * linkage, and mismatching it here is a hard error once the header is in. */
int pulsar_gpu_indexer_scores_mxfp4(
        float *scores, const pulsar_mxkv_pack_t *q, const float *weights,
        const pulsar_mxkv_pack_t *comp,
        uint32_t n_comp, uint32_t n_tokens, uint32_t pos0,
        uint32_t n_head, uint32_t head_dim, uint32_t ratio,
        float scale, int causal) {
    /* Shape gate, evaluated once per launch -- never per token or per layer. */
    if (!scores || !q || !weights || !comp) return 0;
    if (n_head != IDX_HEADS || head_dim != IDX_HEAD_DIM) return 0;
    if (n_comp == 0u || n_tokens == 0u) return 0;

    /* Pre-pack Q once per token; the scorer then only copies bytes. */
    const uint64_t qa_bytes  = (uint64_t)n_tokens * IDX_HEADS * IDX_HEAD_DIM;
    const uint64_t qsf_bytes = (uint64_t)n_tokens * IDX_HEADS * IDX_KSLABS;
    cuda_arena ar;
    if (!cuda_arena_begin(&ar, ((qa_bytes + 255u) & ~255ull) + qsf_bytes,
                          "indexer mxfp4 Q pack")) return 0;
    uint8_t *qa  = (uint8_t *)cuda_arena_take(&ar, qa_bytes, 256);
    /* qsf used to start at qa + qa_bytes with no padding, so its alignment was
     * whatever the Q slab's byte count happened to leave.  The arena aligns it. */
    uint8_t *qsf = (uint8_t *)cuda_arena_take(&ar, qsf_bytes, 256);
    if (!qsf) return 0;   /* take() latches: one check covers both */

    /* One warp per (token, head) row, 8 warps per block. */
    const uint32_t pack_rows = n_tokens * IDX_HEADS;
    idx_expand_q_kernel<<<(pack_rows + 7u) / 8u, 256>>>(qa, qsf, q, pack_rows);
    if (!cuda_ok(cudaGetLastError(), "indexer mxfp4 Q expand launch")) return 0;

    dim3 grid((n_comp + IDX_NTILE - 1u) / IDX_NTILE,
              (n_tokens + IDX_TOKGROUP - 1u) / IDX_TOKGROUP, 1);
    idx_scores_mxfp4_kernel<<<grid, IDX_THREADS>>>(
        scores, qa, qsf, weights, (const uint8_t *)comp,
        n_comp, n_tokens, pos0, ratio, scale, causal);
    return cuda_ok(cudaGetLastError(), "indexer scores mxfp4 launch");
}

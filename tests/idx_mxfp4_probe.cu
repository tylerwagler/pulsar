/* Block-scaled MMA probe for the indexer scorer (design: docs/indexer-mxfp4-scorer.md).
 *
 * Modelless, standalone, no CUTLASS. Answers the three make-or-break questions
 * BEFORE any kernel gets written, following the precedent of the MoE's
 * standalone self-check against a double-precision oracle:
 *
 *   1. Does mma.sync...kind::mxf8f6f4.block_scale ACTUALLY assemble and run at
 *      sm_120f on this GB10?  The header's #if says SM120 + CUDA>=12.8 enables
 *      it; that is a claim about the header, not about our build.
 *   2. What is the C (accumulator) thread->element mapping?  Validated, not
 *      assumed.
 *   3. What is the SF thread->row/col mapping?  cute says
 *        SFALayout = Layout<Shape<Shape<_2,_2,_8>,_32>>  // effectively 16 threads
 *        SFBLayout = Layout<Shape<Shape<_4,_8>,_32>>     // effectively  8 threads
 *      i.e. threads SHARE scales through stride-0 modes.  Guessing this is the
 *      documented way to "read scrambled scales", so recover it empirically.
 *
 * WHY NO CUTLASS HERE.  The probe sets A and B to all-ones, and with a uniform
 * operand every byte of every A/B register is the same value REGARDLESS of the
 * per-thread fragment layout.  So the packing cannot be got wrong, which is
 * exactly what lets the SF mapping be isolated as the only unknown.  The real
 * kernel will use cute atoms and their MMA_Traits; this probe deliberately does
 * not, so that a cute API mistake cannot be confused for a hardware answer.
 *
 * BUILD ARCH.  Raw block-scaled asm cannot be compiled unguarded at the tree's
 * normal -arch=sm_120f: nvcc also emits it into the compute_120 PTX (JIT
 * fallback) pass, where ptxas rejects .kind::mxf8f6f4.  CUTLASS avoids this by
 * guarding its copy behind CUTE_ARCH_MXF8F6F4_MMA_ENABLED, which is why the
 * instruction DOES ship in pulsar_mxfp4_cutlass.o at sm_120f (256 x
 * QMMA.SF.16832.F32.E4M3.E2M1.E8, confirmed with cuobjdump).  This probe sidesteps
 * the issue by emitting a single arch-specific target.
 *
 * build+run:
 *   nvcc -O3 -gencode arch=compute_121a,code=sm_121a -o /tmp/idx_mxfp4_probe \
 *        tests/idx_mxfp4_probe.cu
 *   /tmp/idx_mxfp4_probe
 *
 * MEASURED CONTRACT (all phases pass; see docs/indexer-mxfp4-scorer.md):
 *   A (e4m3)  : one value per byte at bit 0, decodes to the standard table.
 *   B (e2m1)  : nibble at bits [5:2] of its byte; decodes to 4x the e2m1 table.
 *   ue8m0     : value = 2^(byte - 128)  -- NOT the spec's 127 bias.
 *   our cache : sf_hw = stored_byte - 1 (+1 bias, -2 for the x4).
 *   SFA lanes : row m <- lane 4*(m%8) + (m/8)
 *   SFB lanes : col n <- lane 4*n
 *   C layout  : row = lane/4 (d0,d1) and +8 (d2,d3); col = (lane%4)*2 + {0,1}
 */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#ifndef SF_REPLICATE
#define SF_REPLICATE 0
#endif

/* e4m3: 1.0 = 0 0111 000 = 0x38 (exponent bias 7).
 * e2m1: 1.0 = nibble 0x2 (the {0,.5,1,1.5,2,3,4,6} grid), byte container 0x02.
 * ue8m0: value = 2^(byte-127), so byte 127 == 1.0. */
#define E4M3_ONE_x4  0x38383838u
#define E2M1_ONE_x4  0x02020202u
#define UE8M0_ONE    128u   /* measured, not 127: see PHASE0b */

__global__ static void probe_kernel(float *out, const uint8_t *sfa, const uint8_t *sfb,
                                    uint32_t apat, uint32_t bpat) {
    const uint32_t lane = threadIdx.x;      /* one warp */
    if (lane >= 32u) return;

    const uint32_t a0 = apat, a1 = apat, a2 = apat, a3 = apat;
    const uint32_t b0 = bpat, b1 = bpat;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    const float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

    /* The SF operand is a 32-BIT register, not a byte.  If the hardware reads
     * all four byte lanes as scales, passing a bare byte leaves lanes 1..3 at
     * 0 -> 2^(0-127) ~ 0, which would silently zero three quarters of the
     * k-extent and is exactly the 8-vs-32 signature.  SF_REPLICATE=1 broadcasts
     * the byte to all four lanes to test that. */
#if SF_REPLICATE
    const uint32_t my_sfa = 0x01010101u * (uint32_t)sfa[lane];
    const uint32_t my_sfb = 0x01010101u * (uint32_t)sfb[lane];
#else
    const uint32_t my_sfa = sfa[lane];
    const uint32_t my_sfb = sfb[lane];
#endif
    const uint16_t bid = 0, tid = 0;

    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
           "r"(b0),  "r"(b1),
           "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
           "r"(my_sfa), "h"(bid), "h"(tid),
           "r"(my_sfb), "h"(bid), "h"(tid));

    /* Dump raw per-thread accumulators; the host decides what the mapping is
     * rather than this kernel asserting one. */
    out[lane * 4u + 0u] = d0;
    out[lane * 4u + 1u] = d1;
    out[lane * 4u + 2u] = d2;
    out[lane * 4u + 3u] = d3;
}

static int run(const uint8_t *h_sfa, const uint8_t *h_sfb, float *h_out,
               uint32_t apat = E4M3_ONE_x4, uint32_t bpat = E2M1_ONE_x4) {
    uint8_t *d_sfa = nullptr, *d_sfb = nullptr;
    float *d_out = nullptr;
    if (cudaMalloc(&d_sfa, 32) != cudaSuccess ||
        cudaMalloc(&d_sfb, 32) != cudaSuccess ||
        cudaMalloc(&d_out, 32 * 4 * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n");
        return -1;
    }
    cudaMemcpy(d_sfa, h_sfa, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sfb, h_sfb, 32, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, 32 * 4 * sizeof(float));

    probe_kernel<<<1, 32>>>(d_out, d_sfa, d_sfb, apat, bpat);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        fprintf(stderr, "LAUNCH FAILED: %s\n", cudaGetErrorString(launch));
        return -2;
    }
    const cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        fprintf(stderr, "EXEC FAILED: %s\n", cudaGetErrorString(sync));
        return -3;
    }
    cudaMemcpy(h_out, d_out, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_sfa); cudaFree(d_sfb); cudaFree(d_out);
    return 0;
}

int main(void) {
    int dev = 0, major = 0, minor = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    printf("device cc = %d.%d\n\n", major, minor);

    uint8_t sfa[32], sfb[32];
    float out[32 * 4];

    /* ---- Phase 0: operand ENCODING sweep -------------------------------
     * Phase 1 returned a uniform 8.0 where all-ones x all-ones over k=32 must
     * give 32.0.  Uniform means the C mapping is fine and the arithmetic is
     * self-consistent -- only 8 of 32 products are landing.  Rather than argue
     * about container rules, vary the byte patterns and read the answer off.
     * Expected value if a pattern decodes to v_a and v_b in EVERY slot is
     * 32*v_a*v_b. */
    {
        for (int i = 0; i < 32; i++) { sfa[i] = UE8M0_ONE; sfb[i] = UE8M0_ONE; }
        struct { const char *tag; uint32_t a, b; } cases[] = {
            {"A=e4m3 1.0 (0x38)   B=e2m1 1.0 lo-nib (0x02)", 0x38383838u, 0x02020202u},
            {"A=e4m3 1.0 (0x38)   B=nib pair 1.0/1.0 (0x22)", 0x38383838u, 0x22222222u},
            {"A=e4m3 1.0 (0x38)   B=e2m1 2.0 lo-nib (0x04)", 0x38383838u, 0x04040404u},
            {"A=e4m3 1.0 (0x38)   B=e2m1 1.0 hi-nib (0x20)", 0x38383838u, 0x20202020u},
            {"A=e4m3 2.0 (0x40)   B=e2m1 1.0 lo-nib (0x02)", 0x40404040u, 0x02020202u},
            {"A=nib pair (0x88)   B=e2m1 1.0 lo-nib (0x02)", 0x88888888u, 0x02020202u},
            /* Which BYTE LANES does the instruction actually consume?  All-ones
             * gives 8, not 32, so only a quarter of the k-extent is landing.
             * Alternating patterns say which lanes are live. */
            {"A=1.0 all           B=0x02 in bytes 0,2 only", 0x38383838u, 0x00020002u},
            {"A=1.0 all           B=0x02 in bytes 1,3 only", 0x38383838u, 0x02000200u},
            {"A=1.0 all           B=0x02 in byte 0 only   ", 0x38383838u, 0x00000002u},
            {"A=1.0 bytes 0,2     B=0x02 all             ", 0x00380038u, 0x02020202u},
            {"A=1.0 bytes 1,3     B=0x02 all             ", 0x38003800u, 0x02020202u},
            {"A=1.0 byte 0 only   B=0x02 all             ", 0x00000038u, 0x02020202u},
        };
        printf("PHASE0 encoding sweep (all SF=1.0; C[0] shown, all elements equal):\n");
        for (unsigned ci = 0; ci < sizeof(cases)/sizeof(cases[0]); ci++) {
            if (run(sfa, sfb, out, cases[ci].a, cases[ci].b) != 0) return 1;
            int uniform = 1;
            for (int i = 1; i < 128; i++) if (out[i] != out[0]) { uniform = 0; break; }
            printf("  %-46s -> %8.2f %s\n", cases[ci].tag, out[0],
                   uniform ? "" : "(NON-UNIFORM)");
        }
        printf("\n");
    }

    /* ---- Phase 0b: UE8M0 bias -----------------------------------------
     * All byte lanes are consumed (0b above) and SF replication changes
     * nothing, yet C is 32/4.  A factor of exactly 4 with TWO scales applied is
     * 0.5 x 0.5, i.e. the ue8m0 unit byte is not where assumed.  Sweep it: the
     * byte b that yields 32.00 is the true 2^0. */
    {
        printf("PHASE0b ue8m0 bias sweep (A=B=1.0, expect 32.00 at the unit byte):\n");
        for (int b = 124; b <= 131; b++) {
            for (int i = 0; i < 32; i++) { sfa[i] = (uint8_t)b; sfb[i] = (uint8_t)b; }
            if (run(sfa, sfb, out, 0x38383838u, 0x02020202u) != 0) return 1;
            printf("  sf byte %3d -> %10.4f%s\n", b, out[0],
                   out[0] == 32.0f ? "   <== unit" : "");
        }
        printf("\n");
    }

    /* ---- Phase 0c: absolute operand value tables -----------------------
     * Two readings fit 0b: either the ue8m0 bias is 128 with operands exactly
     * 1.0, or the bias is the spec's 127 and the operand product carries a
     * hidden 0.25.  Pin it by reading the tables off directly at the measured
     * unit byte: with A=1.0 and k=32, C/32 IS the decoded B value. */
    {
        for (int i = 0; i < 32; i++) { sfa[i] = 128; sfb[i] = 128; }
        static const float e2m1_ref[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        printf("PHASE0c e2m1 value table at sf=128 (A=1.0, C/32 == decoded B):\n");
        for (int code = 0; code < 8; code++) {
            const uint32_t bp = 0x01010101u * (uint32_t)code;
            if (run(sfa, sfb, out, 0x38383838u, bp) != 0) return 1;
            const float got = out[0] / 32.0f;
            printf("  code %d -> %6.3f   (our table %6.3f) %s\n",
                   code, got, e2m1_ref[code],
                   got == e2m1_ref[code] ? "match" : "MISMATCH");
        }
        /* And the A side: with B held at code 2, C/(32*B) is the decoded A. */
        printf("PHASE0c e4m3 spot-check at sf=128 (B=code2):\n");
        const struct { uint8_t byte; float ref; } acheck[] = {
            {0x38, 1.0f}, {0x40, 2.0f}, {0x30, 0.5f}, {0x3c, 1.5f},
        };
        for (unsigned i = 0; i < sizeof(acheck)/sizeof(acheck[0]); i++) {
            const uint32_t ap = 0x01010101u * (uint32_t)acheck[i].byte;
            if (run(sfa, sfb, out, ap, 0x02020202u) != 0) return 1;
            printf("  0x%02x -> %6.3f   (e4m3 %6.3f)\n",
                   acheck[i].byte, out[0] / 32.0f, acheck[i].ref);
        }
        printf("\n");
    }

    /* ---- Phase 0d: fp4 container BIT POSITION --------------------------
     * 0c shows the low nibble decoding as a linear code*0.5 ramp, not as e2m1
     * (codes 5,6,7 give 2.5,3.0,3.5 where e2m1 is 3,4,6).  cute ships
     * fp4_shift_A/fp4_shift_B helpers, which exist precisely because the 4-bit
     * value does not sit at bit 0 of its byte container.  Find the shift that
     * reproduces the real e2m1 table. */
    {
        for (int i = 0; i < 32; i++) { sfa[i] = 128; sfb[i] = 128; }
        static const float e2m1_ref[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        printf("PHASE0d fp4 container shift (looking for the e2m1 table):\n");
        for (int sh = 0; sh <= 4; sh++) {
            int all = 1;
            printf("  shift %d: ", sh);
            for (int code = 0; code < 8; code++) {
                const uint32_t bp = 0x01010101u * ((uint32_t)code << sh);
                if (run(sfa, sfb, out, 0x38383838u, bp) != 0) return 1;
                const float got = out[0] / 32.0f;
                printf("%5.2f ", got);
                if (got != e2m1_ref[code]) all = 0;
            }
            printf("  %s\n", all ? "<== E2M1 TABLE" : "");
        }
        printf("\n");
    }

    /* ---- Phase 0e: the full production recipe --------------------------
     * Everything measured above, combined, in the form the kernel will use:
     *   B byte container = e2m1 nibble << 2        (bits [5:2])
     *   decoded value    = 4 * e2m1(nibble)        (constant, fold into SF)
     *   ue8m0 unit byte  = 128                     (value = 2^(byte-128))
     * so a B-side scale byte of 126 cancels the x4 exactly.
     *
     * Our cache stores its own e8 byte with bias 127 (idx_comp_load_dev builds
     * the float as byte<<23, i.e. 2^(byte-127)), so the conversion for a stored
     * byte s is simply  sf_hw = s - 1  (+1 for the 127->128 bias, -2 for the
     * x4).  One subtraction per scale, no arithmetic per element. */
    {
        static const float e2m1_ref[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        for (int i = 0; i < 32; i++) { sfa[i] = 128; sfb[i] = 126; }
        printf("PHASE0e production recipe (A=1.0, B=nib<<2, sfb=126; C/32 == e2m1):\n");
        int all = 1;
        for (int code = 0; code < 8; code++) {
            const uint32_t bp = 0x01010101u * ((uint32_t)code << 2);
            if (run(sfa, sfb, out, 0x38383838u, bp) != 0) return 1;
            const float got = out[0] / 32.0f;
            if (got != e2m1_ref[code]) all = 0;
            printf("  nib %d -> %6.3f  (e2m1 %6.3f) %s\n", code, got, e2m1_ref[code],
                   got == e2m1_ref[code] ? "" : "MISMATCH");
        }
        printf("  RECIPE: %s\n\n", all ? "VERIFIED" : "FAILED");
    }

    /* ---- Phase 1: availability + C layout ------------------------------
     * All scales 1.0, A=B=1.0, K=32  =>  every C element must be exactly 32.
     * This simultaneously proves the instruction runs and that we are reading
     * the accumulator correctly (any element != 32 means the C mapping or the
     * operand packing is wrong, before scales enter the picture). */
    for (int i = 0; i < 32; i++) { sfa[i] = UE8M0_ONE; sfb[i] = UE8M0_ONE; }
    if (run(sfa, sfb, out) != 0) {
        printf("PHASE1: INSTRUCTION UNAVAILABLE OR FAILED\n");
        return 1;
    }
    int bad = 0;
    for (int i = 0; i < 128; i++) if (out[i] != 32.0f) bad++;
    printf("PHASE1 availability+C: %s (%d/128 elements != 32.0; first four: %.1f %.1f %.1f %.1f)\n",
           bad == 0 ? "PASS" : "FAIL", bad, out[0], out[1], out[2], out[3]);
    if (bad) {
        printf("  -> operand packing or C mapping is wrong; SF mapping below is meaningless\n");
        return 1;
    }

    /* ---- Phase 2: SFA thread -> M-row mapping --------------------------
     * Give lane t the scale 2^(t mod 16) and hold SFB at 1.0.  Then
     *   C[m][n] = 32 * sfa_effective[m]
     * so log2(C/32) names WHICH lane's byte the hardware applied to row m.
     * The standard m16n8 accumulator layout is
     *   d0,d1 -> row = lane/4, col = (lane%4)*2 + {0,1}
     *   d2,d3 -> row = lane/4 + 8, same cols
     * Phase 1 having passed means that layout reads a uniform tile correctly;
     * this phase is what actually tests it against a NON-uniform one. */
    for (int i = 0; i < 32; i++) { sfa[i] = (uint8_t)(UE8M0_ONE + i); sfb[i] = UE8M0_ONE; }
    if (run(sfa, sfb, out, 0x38383838u, 0x08080808u) != 0) return 1;   /* B nib 2 << 2 */
    printf("\nPHASE2 SFA: which LANE supplies each M row (distinct 2^lane per lane)\n");
    for (int lane = 0; lane < 32; lane += 4) {
        const int row_lo = lane / 4, row_hi = row_lo + 8;
        /* C = 32 * 1.0 * 4.0 * 2^(sfa-128) * 2^(126-128) ... sfb is unit(128)
         * here and B decodes to 4.0, so C = 128 * 2^e  ->  e = log2(C/128). */
        const int e_lo = (int)(__builtin_log2f(out[lane * 4 + 0] / 128.0f) + 0.5f);
        const int e_hi = (int)(__builtin_log2f(out[lane * 4 + 2] / 128.0f) + 0.5f);
        printf("  row %2d <- lane %2d      row %2d <- lane %2d\n",
               row_lo, e_lo, row_hi, e_hi);
    }

    /* ---- Phase 3: SFB thread -> N-col mapping -------------------------- */
    for (int i = 0; i < 32; i++) { sfa[i] = UE8M0_ONE; sfb[i] = (uint8_t)(UE8M0_ONE + i); }
    if (run(sfa, sfb, out, 0x38383838u, 0x08080808u) != 0) return 1;
    printf("\nPHASE3 SFB: which LANE supplies each N col\n");
    for (int lane = 0; lane < 4; lane++) {
        for (int half = 0; half < 2; half++) {
            const int col = (lane % 4) * 2 + half;
            const int e = (int)(__builtin_log2f(out[lane * 4 + half] / 128.0f) + 0.5f);
            printf("  col %2d <- lane %2d\n", col, e);
        }
    }

    printf("\nPROBE_DONE\n");
    return 0;
}

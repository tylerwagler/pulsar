/* Round-trip test for the four live quant formats (iq2_xxs, q2_K, fp8_e4m3,
 * mxfp4). Guards three properties the offline pipeline depends on:
 *   1. relative RMSE on Gaussian data is sane for each format's bit budget
 *      (a sign or layout bug shows up as rel-RMSE near sqrt(2));
 *   2. signs survive the round-trip (the historical mxfp4 encoder bug wrote
 *      the sign at bit 7 of a 4-bit nibble, silently flipping every negative
 *      weight positive — this test exists because of it);
 *   3. fp8/mxfp4 are value-level idempotent: re-quantizing a dequantized
 *      block reproduces exactly the same reals (byte layouts may differ when
 *      the block scale re-buckets, values may not).
 */
#include "quants.h"
#include "quants_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned long st = 88172645463325252ULL;
static double rnd(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (double)(st >> 11) / 9007199254740992.0; }
static float gauss(void) { double u = rnd(), v = rnd(); return (float)(sqrt(-2 * log(u + 1e-12)) * cos(6.2831853 * v)); }

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static void quant_roundtrip(ds4q_type t, void (*deq)(const void *, float *, int64_t),
                            const float *x, int n, int ncols, float *rc) {
    int nrows = n / ncols;
    size_t rs = ds4q_row_size(t, ncols);
    unsigned char *q = malloc(rs * nrows);
    /* iq2_xxs requires an imatrix; a flat one exercises the same code path. */
    float *iw = malloc((size_t)ncols * sizeof(float));
    for (int i = 0; i < ncols; i++) iw[i] = 1.0f;
    ds4q_quantize_init(t);
    ds4q_quantize_chunk(t, x, q, 0, nrows, ncols, iw);
    for (int r = 0; r < nrows; r++) deq(q + (size_t)r * rs, rc + (size_t)r * ncols, ncols);
    free(iw);
    free(q);
}

static void check_format(const char *name, ds4q_type t,
                         void (*deq)(const void *, float *, int64_t),
                         const float *x, int n, int ncols,
                         double max_rel_rmse, float sign_floor) {
    float *rc = malloc((size_t)n * sizeof(float));
    quant_roundtrip(t, deq, x, n, ncols, rc);

    double se = 0, sx = 0;
    for (int i = 0; i < n; i++) { double e = x[i] - rc[i]; se += e * e; sx += (double)x[i] * x[i]; }
    double rel = sqrt(se / sx);
    printf("  %-9s rel_rmse %.4f (max %.4f)\n", name, rel, max_rel_rmse);
    CHECK(rel <= max_rel_rmse, "%s rel_rmse %.4f exceeds %.4f", name, rel, max_rel_rmse);

    /* Signs must survive for every value large enough that the format cannot
     * round it to zero. sign_floor is per-format: |x| above this fraction of
     * the global amax cannot quantize to 0 in any block. */
    float amax = 0;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    int sign_flips = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(x[i]) < sign_floor * amax) continue;
        if (x[i] * rc[i] < 0) sign_flips++;
    }
    CHECK(sign_flips == 0, "%s flipped the sign of %d large-magnitude values", name, sign_flips);
    free(rc);
}

/* fp8/mxfp4 value-level idempotency: quantize(dequant(q)) must dequantize to
 * exactly the same reals. Catches encoder/decoder disagreement on layout,
 * sign position, or rounding of exactly-representable values. */
static void check_idempotent(const char *name, ds4q_type t,
                             void (*deq)(const void *, float *, int64_t),
                             const float *x, int n, int ncols) {
    float *r1 = malloc((size_t)n * sizeof(float));
    float *r2 = malloc((size_t)n * sizeof(float));
    quant_roundtrip(t, deq, x, n, ncols, r1);
    quant_roundtrip(t, deq, r1, n, ncols, r2);
    int diffs = 0;
    for (int i = 0; i < n; i++) if (r1[i] != r2[i]) diffs++;
    CHECK(diffs == 0, "%s not value-idempotent: %d/%d values changed on requant", name, diffs, n);
    free(r1); free(r2);
}

int main(void) {
    const int NC = 4096, NR = 8, N = NC * NR;
    float *x = malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) x[i] = gauss() * 0.05f;

    printf("quant round-trip (Gaussian, zero mean — half the mass is negative):\n");
    /* Thresholds are ~2x the measured healthy values; the sign bug pushed
     * mxfp4 rel_rmse to ~1.4, far past any of these. */
    check_format("iq2_xxs", DS4Q_TYPE_IQ2_XXS,  ds4q_dequantize_iq2_xxs, x, N, NC, 0.50, 0.30f);
    check_format("q2_K",    DS4Q_TYPE_Q2_K,     ds4q_dequantize_q2_k,    x, N, NC, 0.40, 0.30f);
    check_format("fp8_e4m3",DS4Q_TYPE_FP8_E4M3, ds4q_dequantize_fp8_e4m3,x, N, NC, 0.05, 0.01f);
    check_format("mxfp4",   DS4Q_TYPE_MXFP4,    ds4q_dequantize_mxfp4,   x, N, NC, 0.30, 0.10f);

    check_idempotent("fp8_e4m3", DS4Q_TYPE_FP8_E4M3, ds4q_dequantize_fp8_e4m3, x, N, NC);
    check_idempotent("mxfp4",    DS4Q_TYPE_MXFP4,    ds4q_dequantize_mxfp4,    x, N, NC);

    /* K-quant refinement collapse (upstream antirez/ds4 ed5c1a92, L192).  In
     * exact arithmetic a least-squares fit's intercept never reaches the block
     * maximum, but with near-degenerate weights (zeros and a 1e23 dynamic
     * range: what an imatrix column times sqrt(sigma2 + x^2) can produce) the
     * normal-equation determinant is rounding noise and an accepted fit can
     * set min = 0 >= max.  Every later step then divides by a non-positive
     * range: iscale flips sign, all-negative x code as nmax, and the fit that
     * mis-codes the outlier at index 10 as its neighbours wins on equal
     * error.  The guard stops the loop at the collapse.  Inputs found by a
     * 20M-trial search over the unguarded loop; without the guard L[10]
     * comes back 3 (the same code as the -0.2 values), with it 0. */
    {
        const float xv[16] = {
            -0.199877203f, -0.199877203f, -0.199877203f, -0.199877203f,
            -0.199877203f, -0.199877203f, -0.199877203f, -0.199877203f,
            -0.199877203f, -0.199877203f, -1.19987726f,  -0.199877203f,
            -0.199877203f, -0.199877203f, -0.199877203f, -0.199877203f,
        };
        const float wv[16] = {
            1.61155338e+10f, 355.506195f,    807180288.0f,    4.4922388e-05f,
            993090304.0f,    7.19309619e+09f, 1.27754907e-11f, 3.6475544f,
            3.50451501e-07f, 0.0f,            0.000257659791f, 2.2799686e+11f,
            0.0f,            5.28423694e-11f, 3.49987349e-05f, 15618188.0f,
        };
        uint8_t L[16], Laux[16];
        float mn = 0.0f;
        float sc = ds4q_make_qkx3_quants(16, 3, xv, wv, L, &mn, Laux, -0.9f, 0.05f, 36, false);
        CHECK(isfinite(sc) && isfinite(mn), "qkx3 collapse: non-finite scale %g / min %g", sc, mn);
        CHECK(L[10] != L[0],
              "qkx3 collapse: the -1.2 outlier (L[10]=%d) got the same code as the -0.2 values (L[0]=%d)",
              L[10], L[0]);
        CHECK(L[10] == 0, "qkx3 collapse: L[10]=%d, want 0", L[10]);
    }

    /* Exactly-representable mxfp4 values must round-trip bit-perfectly,
     * including their signs. */
    {
        const float mags[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
        float xv[32], rc[32];
        for (int i = 0; i < 32; i++) {
            float m = mags[(i / 2) % 8];
            xv[i] = (i & 1) ? -m : m;
        }
        quant_roundtrip(DS4Q_TYPE_MXFP4, ds4q_dequantize_mxfp4, xv, 32, 32, rc);
        for (int i = 0; i < 32; i++)
            CHECK(rc[i] == xv[i] || (xv[i] == 0.0f && rc[i] == 0.0f),
                  "mxfp4 exact value %g came back as %g (i=%d)", xv[i], rc[i], i);
    }

    /* CUTLASS_MXFP4 (type 40) pack invariants:
     *   1. sizing matches the real dspark expert blobs (N=2048, K=4096 ->
     *      4456448 B/expert incl. 262144 B of SF; ground truth measured from
     *      gguf/dspark.gguf, produced by the validated CUTLASS packer);
     *   2. the data region is the source E2M1 array byte-verbatim;
     *   3. the SF swizzle is a bijection into the padded tile (every source
     *      scale lands on a unique in-range offset; padding bytes stay 0),
     *      and unswizzling recovers the source scales exactly. */
    {
        CHECK(ds4q_cutlass_mxfp4_sf_bytes(2048, 4096) == 262144,
              "cutlass_mxfp4 sf_bytes(2048,4096) = %zu, want 262144",
              ds4q_cutlass_mxfp4_sf_bytes(2048, 4096));
        CHECK(ds4q_cutlass_mxfp4_bytes(2048, 4096) == 4456448,
              "cutlass_mxfp4 bytes(2048,4096) = %zu, want 4456448",
              ds4q_cutlass_mxfp4_bytes(2048, 4096));

        const int64_t NN = 320, KK = 224;   /* exercises padding: 320 rows -> 384, 7 kblocks -> 8 */
        const size_t db = (size_t)(NN * KK / 2);
        const size_t sb = ds4q_cutlass_mxfp4_sf_bytes(NN, KK);
        const int64_t kbn = KK / 32;
        uint8_t *e2m1 = malloc(db);
        uint8_t *e8m0 = malloc((size_t)(NN * kbn));
        uint8_t *blob = malloc(db + sb);
        for (size_t i = 0; i < db; i++) e2m1[i] = (uint8_t)(rnd() * 256);
        for (int64_t i = 0; i < NN * kbn; i++) e8m0[i] = (uint8_t)(1 + i % 253); /* nonzero */
        ds4q_pack_cutlass_mxfp4(e2m1, e8m0, blob, NN, KK);

        CHECK(memcmp(blob, e2m1, db) == 0, "cutlass_mxfp4 data region is not byte-verbatim");

        /* Unswizzle by scanning the SF region: each nonzero byte must map
         * back from exactly one (row, kb); count and compare values. */
        const uint8_t *sf = blob + db;
        size_t nonzero = 0;
        for (size_t i = 0; i < sb; i++) if (sf[i]) nonzero++;
        CHECK(nonzero == (size_t)(NN * kbn),
              "cutlass_mxfp4 SF nonzero count %zu, want %lld (swizzle not injective or out of range)",
              nonzero, (long long)(NN * kbn));
        int sf_bad = 0;
        const int64_t kbp = (kbn + 3) / 4 * 4;
        for (int64_t r = 0; r < NN && sf_bad < 8; r++) {
            for (int64_t kb = 0; kb < kbn; kb++) {
                size_t off = (size_t)((r / 128) * (kbp / 4) + kb / 4) * 512
                           + (size_t)(r % 32) * 16 + (size_t)((r % 128) / 32) * 4 + (size_t)(kb % 4);
                if (off >= sb || sf[off] != e8m0[r * kbn + kb]) { sf_bad++; break; }
            }
        }
        CHECK(sf_bad == 0, "cutlass_mxfp4 SF unswizzle mismatch on %d row(s)", sf_bad);
        free(e2m1); free(e8m0); free(blob);
    }

    /* G8 guard: the mx_sfoff swizzle is triplicated byte-identically across
     * CUDA (src/cuda/ds4_cuda_matmul.cu), C (ds4q_mx_sfoff), and numpy
     * (build_scale_dest_index in repack_mxfp8_lt.py). Assert the C copy agrees
     * with an INDEPENDENT recomputation of the numpy formula over a grid of
     * (row, kb, KBp), so a future edit to one copy can't silently diverge. */
    {
        int off_bad = 0;
        const int64_t kbps[] = { 4, 8, 12, 32, 132 };
        for (size_t ki = 0; ki < sizeof(kbps) / sizeof(kbps[0]) && off_bad < 8; ki++) {
            const int64_t KBp = kbps[ki];
            for (int64_t o = 0; o < 320 && off_bad < 8; o++) {
                for (int64_t kb = 0; kb < KBp && off_bad < 8; kb++) {
                    /* exact transcription of numpy build_scale_dest_index */
                    size_t want = (size_t)(((o / 128) * (KBp / 4) + (kb / 4)) * 512
                                  + (o % 32) * 16 + ((o % 128) / 32) * 4 + (kb % 4));
                    if (ds4q_mx_sfoff(o, kb, KBp) != want) {
                        off_bad++;
                        printf("FAIL: ds4q_mx_sfoff(%lld,%lld,%lld)=%zu want %zu\n",
                               (long long)o, (long long)kb, (long long)KBp,
                               ds4q_mx_sfoff(o, kb, KBp), want);
                    }
                }
            }
        }
        CHECK(off_bad == 0, "ds4q_mx_sfoff diverged from the numpy formula on %d point(s)", off_bad);
    }

    /* MXFP8_LT (type 41) pack invariants (mirrors the type-40 checks above):
     *   1. sizing: data = out*in, SF = rup(out,128)*rup(KB,4); for 128-aligned
     *      shapes the total equals the type-38 size (out*KB*33);
     *   2. the E4M3 data region is the type-38 blocks de-interleaved to [out,in];
     *   3. the SF swizzle is a bijection into the padded tile and unswizzles
     *      back to the source E8M0 scales exactly. */
    {
        const int64_t OUT = 256, IN = 128;          /* 128-aligned workhorse shape */
        const int64_t KB = IN / 32;                 /* 4 */
        CHECK(ds4q_mxfp8_lt_sf_bytes(OUT, IN) == (size_t)(OUT * KB),
              "mxfp8_lt sf_bytes(256,128)=%zu want %lld",
              ds4q_mxfp8_lt_sf_bytes(OUT, IN), (long long)(OUT * KB));
        CHECK(ds4q_mxfp8_lt_bytes(OUT, IN) == (size_t)(OUT * KB * 33),
              "mxfp8_lt bytes(256,128)=%zu want type-38 size %lld",
              ds4q_mxfp8_lt_bytes(OUT, IN), (long long)(OUT * KB * 33));

        const size_t raw_bytes = (size_t)(OUT * KB * 33);
        const size_t db = (size_t)(OUT * IN);
        uint8_t *raw = malloc(raw_bytes);
        uint8_t *blob = malloc(ds4q_mxfp8_lt_bytes(OUT, IN));
        for (size_t i = 0; i < raw_bytes; i++) raw[i] = (uint8_t)(rnd() * 256);
        for (int64_t b = 0; b < OUT * KB; b++)                 /* E8M0 in [1,254] */
            raw[(size_t)b * 33] = (uint8_t)(1 + (int)(rnd() * 253));
        ds4q_pack_mxfp8_lt(raw, blob, OUT, IN);

        int data_bad = 0;
        for (int64_t r = 0; r < OUT && data_bad < 4; r++)
            for (int64_t kb = 0; kb < KB; kb++)
                if (memcmp(blob + (size_t)(r * IN + kb * 32),
                           raw + ((size_t)(r * KB + kb)) * 33 + 1, 32) != 0) { data_bad++; break; }
        CHECK(data_bad == 0, "mxfp8_lt data region not de-interleaved E4M3 (%d row(s))", data_bad);

        const uint8_t *sf = blob + db;
        const size_t sb = ds4q_mxfp8_lt_sf_bytes(OUT, IN);
        size_t nonzero = 0;
        for (size_t i = 0; i < sb; i++) if (sf[i]) nonzero++;
        CHECK(nonzero == (size_t)(OUT * KB),
              "mxfp8_lt SF nonzero %zu want %lld (swizzle not injective)",
              nonzero, (long long)(OUT * KB));
        int sf_bad = 0;
        const int64_t kbp = (KB + 3) / 4 * 4;
        for (int64_t r = 0; r < OUT && sf_bad < 8; r++)
            for (int64_t kb = 0; kb < KB; kb++)
                if (sf[ds4q_mx_sfoff(r, kb, kbp)] != raw[((size_t)(r * KB + kb)) * 33]) { sf_bad++; break; }
        CHECK(sf_bad == 0, "mxfp8_lt SF unswizzle mismatch on %d row(s)", sf_bad);
        free(raw); free(blob);
    }

    free(x);
    if (failures) { printf("%d FAILURE(S)\n", failures); return 1; }
    printf("OK\n");
    return 0;
}

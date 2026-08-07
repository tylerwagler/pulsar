/* mmq_soa_equiv.cu -- does the aligned-SoA layout change the NUMBERS?
 *
 * The offline repack is only a drop-in if the SoA kernels return bit-identical
 * f32 to the raw-block kernels on the same weights.  If they do, an offline
 * type-43 model must produce the same logits as the raw model, and any
 * end-to-end difference is engine plumbing rather than layout.
 *
 * Runs, on a REAL tensor pulled from the model, at the production MoE shape:
 *   ds4_mmq_iq2_xxs_moe_pair(raw, raw)  vs  ds4_mmq_iq2_xxs_moe_pair_soa(ali, ali)
 *   ds4_mmq_iq2_xxs_moe(raw)            vs  ds4_mmq_iq2_xxs_moe_soa(ali)
 * and compares the outputs BIT-exactly (memcmp on the f32 bytes), not with a
 * tolerance -- a tolerance would hide exactly the reordering we care about.
 *
 * Usage: mmq_soa_equiv RAW.bin ALIGNED.bin M K n_experts n_expert_used n_tokens
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#include "ds4_mmq.h"

extern "C" int ds4_cuda_q8_fold_take_q81(const void *src, uint64_t in_dim, const void **q81) {
    (void)src; (void)in_dim; (void)q81; return 0;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(2);} } while (0)

static unsigned char *slurp(const char *path, uint64_t *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); exit(2); }
    fseeko(f, 0, SEEK_END);
    uint64_t n = (uint64_t)ftello(f);
    fseeko(f, 0, SEEK_SET);
    unsigned char *p = (unsigned char *)malloc(n);
    if (!p) { fprintf(stderr, "alloc failed\n"); exit(2); }
    if (fread(p, 1, n, f) != n) { fprintf(stderr, "short read %s\n", path); exit(2); }
    fclose(f);
    *n_out = n;
    return p;
}

/* Bit-exact f32 comparison, with the worst ULP/abs gap when it fails so a
 * near-miss (reassociation) reads differently from a wrong layout (garbage). */
static int cmp_f32(const char *tag, const float *a, const float *b, uint64_t n) {
    if (memcmp(a, b, n * sizeof(float)) == 0) {
        printf("%s: BIT-IDENTICAL (%llu floats)\n", tag, (unsigned long long)n);
        return 0;
    }
    uint64_t ndiff = 0, first = ~0ull;
    double maxabs = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        if (a[i] != b[i] || (isnan(a[i]) != isnan(b[i]))) {
            if (first == ~0ull) first = i;
            ndiff++;
            double d = fabs((double)a[i] - (double)b[i]);
            if (d > maxabs) maxabs = d;
        }
    }
    fprintf(stderr, "%s: DIFFERS -- %llu of %llu floats (%.4f%%), first at %llu "
            "(%.9g vs %.9g), max |delta| = %.9g\n", tag,
            (unsigned long long)ndiff, (unsigned long long)n,
            100.0 * (double)ndiff / (double)n, (unsigned long long)first,
            (double)a[first], (double)b[first], maxabs);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: %s RAW.bin ALIGNED.bin M K n_experts n_expert_used n_tokens\n", argv[0]);
        return 1;
    }
    const int M = atoi(argv[3]);
    const int K = atoi(argv[4]);
    const int NE = atoi(argv[5]);
    const int NEU = atoi(argv[6]);
    const int NT = atoi(argv[7]);

    uint64_t raw_n = 0, ali_n = 0;
    unsigned char *raw = slurp(argv[1], &raw_n);
    unsigned char *ali = slurp(argv[2], &ali_n);
    printf("M=%d K=%d experts=%d used=%d tokens=%d  raw=%llu aligned=%llu\n",
           M, K, NE, NEU, NT, (unsigned long long)raw_n, (unsigned long long)ali_n);

    int dev = 0; CK(cudaGetDevice(&dev));
    if (ds4_mmq_init(dev) != 0) { fprintf(stderr, "ds4_mmq_init failed\n"); return 2; }

    const uint64_t xn = (uint64_t)NT * K;
    const uint64_t idn = (uint64_t)NT * NEU;
    const uint64_t on = (uint64_t)NT * NEU * M;

    float *hx = (float *)malloc(xn * sizeof(float));
    int32_t *hid = (int32_t *)malloc(idn * sizeof(int32_t));
    /* Deterministic inputs: a fixed LCG, so a rerun compares like for like. */
    uint32_t s = 12345u;
    for (uint64_t i = 0; i < xn; i++) {
        s = s * 1664525u + 1013904223u;
        hx[i] = ((float)(s >> 8) / (float)(1u << 24) - 0.5f) * 2.0f;
    }
    for (uint64_t i = 0; i < idn; i++) {
        s = s * 1664525u + 1013904223u;
        hid[i] = (int32_t)((s >> 8) % (uint32_t)NE);
    }

    unsigned char *d_raw, *d_ali;
    float *d_x, *d_a1, *d_b1, *d_a2, *d_b2;
    int32_t *d_id;
    CK(cudaMalloc(&d_raw, raw_n));
    CK(cudaMalloc(&d_ali, ali_n));
    CK(cudaMalloc(&d_x, xn * sizeof(float)));
    CK(cudaMalloc(&d_id, idn * sizeof(int32_t)));
    CK(cudaMalloc(&d_a1, on * sizeof(float)));
    CK(cudaMalloc(&d_b1, on * sizeof(float)));
    CK(cudaMalloc(&d_a2, on * sizeof(float)));
    CK(cudaMalloc(&d_b2, on * sizeof(float)));
    CK(cudaMemcpy(d_raw, raw, raw_n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ali, ali, ali_n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x, hx, xn * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_id, hid, idn * sizeof(int32_t), cudaMemcpyHostToDevice));

    float *ha = (float *)malloc(on * sizeof(float));
    float *hb = (float *)malloc(on * sizeof(float));
    float *ha2 = (float *)malloc(on * sizeof(float));
    float *hb2 = (float *)malloc(on * sizeof(float));
    int rc = 0;

    /* ---- pair: raw vs soa ---- */
    CK(cudaMemset(d_a1, 0, on * sizeof(float)));
    CK(cudaMemset(d_b1, 0, on * sizeof(float)));
    int r = ds4_mmq_iq2_xxs_moe_pair(d_raw, d_raw, d_x, d_id, d_a1, d_b1,
                                     M, K, NT, NE, NEU, 0);
    CK(cudaDeviceSynchronize());
    if (r != 0) { fprintf(stderr, "moe_pair rc=%d\n", r); return 2; }

    CK(cudaMemset(d_a2, 0, on * sizeof(float)));
    CK(cudaMemset(d_b2, 0, on * sizeof(float)));
    r = ds4_mmq_iq2_xxs_moe_pair_soa(d_ali, d_ali, d_x, d_id, d_a2, d_b2,
                                     M, K, NT, NE, NEU, 0);
    CK(cudaDeviceSynchronize());
    if (r != 0) { fprintf(stderr, "moe_pair_soa rc=%d\n", r); return 2; }

    CK(cudaMemcpy(ha, d_a1, on * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ha2, d_a2, on * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb, d_b1, on * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb2, d_b2, on * sizeof(float), cudaMemcpyDeviceToHost));
    rc |= cmp_f32("pair out_a (moe_pair vs moe_pair_soa)", ha, ha2, on);
    rc |= cmp_f32("pair out_b (moe_pair vs moe_pair_soa)", hb, hb2, on);

    /* ---- single: raw vs soa ---- */
    CK(cudaMemset(d_a1, 0, on * sizeof(float)));
    r = ds4_mmq_iq2_xxs_moe(d_raw, d_x, d_id, d_a1, M, K, NT, NE, NEU, 0);
    CK(cudaDeviceSynchronize());
    if (r != 0) { fprintf(stderr, "moe rc=%d\n", r); return 2; }

    CK(cudaMemset(d_a2, 0, on * sizeof(float)));
    r = ds4_mmq_iq2_xxs_moe_soa(d_ali, d_x, d_id, d_a2, M, K, NT, NE, NEU, 0);
    CK(cudaDeviceSynchronize());
    if (r != 0) { fprintf(stderr, "moe_soa rc=%d\n", r); return 2; }

    float *hs = (float *)malloc(on * sizeof(float));
    float *hs2 = (float *)malloc(on * sizeof(float));
    CK(cudaMemcpy(hs, d_a1, on * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs2, d_a2, on * sizeof(float), cudaMemcpyDeviceToHost));
    rc |= cmp_f32("single out (moe vs moe_soa)", hs, hs2, on);

    /* Cross-checks that separate "the artifact is wrong" from "the two kernels
     * schedule their accumulation differently".  If pair-vs-single already
     * differs on the RAW weights, then any pair-vs-pair_soa gap is a property
     * of the kernels, not of the layout. */
    printf("--- cross-checks (same weights, different kernel) ---\n");
    (void)cmp_f32("raw:     moe_pair out_a vs moe out", ha, hs, on);
    (void)cmp_f32("aligned: moe_pair_soa out_a vs moe_soa out", ha2, hs2, on);

    printf(rc ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return rc;
}

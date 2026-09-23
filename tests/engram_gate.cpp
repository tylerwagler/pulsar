/* ENGRAM DEVICE-PATH gate (L242): rows -> slot -> `wkv` GEMM -> gate -> gated add, on
 * one GPU, against a fixture computed from the checkpoint's own rows and weights.
 *
 * WHAT IT PROVES
 * --------------
 * The fixture (tools/engram/gen_engram_fixture.py) carries, for a fixed window of real
 * token ids: the 24 row ids per token the reference hash names, the 264-byte rows read
 * from the checkpoint at those ids, a seeded bf16 residual `h`, and the reference's
 * result -- `kv`, `dot`, `gate`, `h_out` -- computed in f32/f64 with every operand
 * dequantised (rows: E4M3 x E8M0; `wkv`: E4M3 x its 32x32 block scale).
 *
 * Our path feeds the rows' E4M3 + E8M0 bytes to the MX GEMM directly and `wkv`'s bytes
 * likewise, so every product the GEMM forms is exactly the reference's product; the only
 * differences are the fp32 accumulation order and the bf16 rounding of the stored
 * residual.  The thresholds are therefore TIGHT, not tolerance-of-convenience:
 *
 *   A. kv       max |ours - ref| <= 1e-4 * max |ref|  (accumulation order only), both
 *               GEMM arms (decode rows: the batched GEMV; prefill rows: cuBLASLt)
 *   B. gate     the gate recomputed on the HOST from OUR kv matches the fixture's gate
 *               to 1e-5 (isolates the GEMM from the gate arithmetic)
 *   C. h_out    every element within 1 bf16 ulp of bf16(ref) -- the one rounding
 *   D. identity the whole path run twice is byte-identical
 *   E. dead     a dead (image-span) token's copies are untouched
 *
 * Needs the layer's aux files (`engram-l<L>.wkv` in the pre-stored MXFP8_LT layout and
 * `engram-l<L>.qk`), read from ENGRAM_DIR; the 189 GiB row files are NOT needed.
 *
 * usage: engram_gate FIXTURE.fix ENGRAM_DIR */
#include "pulsar_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 256u
#define N_SCALE 8u
#define ROW_BYTES 264u
#define N_COLS 24u
#define WKV_IN 6144u
#define WKV_OUT 25600u

static int g_fail;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail = 1; printf("  FAIL  "); printf(__VA_ARGS__); printf("\n"); } else { printf("  ok    "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = malloc((size_t)sz);
    if (!p || fread(p, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(p); return NULL; }
    fclose(f);
    *n = (size_t)sz;
    return p;
}
static float bf16f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

typedef struct {
    uint32_t layer, T, n_hc, dim, n_cols, row_bytes; float eps;
    const int32_t *ids; const uint64_t *cols; const uint8_t *rows; const uint16_t *h_bits;
    const float *weight, *kv, *dot, *gate, *h_out;
} fixture;

static int parse_fixture(const uint8_t *p, size_t n, fixture *F) {
    if (n < 8 + 28 || memcmp(p, "PENGFIX1", 8)) return 0;
    memcpy(&F->layer, p + 8, 4); memcpy(&F->T, p + 12, 4); memcpy(&F->n_hc, p + 16, 4);
    memcpy(&F->dim, p + 20, 4); memcpy(&F->n_cols, p + 24, 4); memcpy(&F->row_bytes, p + 28, 4);
    memcpy(&F->eps, p + 32, 4);
    if (F->n_cols != N_COLS || F->row_bytes != ROW_BYTES || F->T == 0 || F->n_hc == 0) return 0;
    const uint8_t *q = p + 36;
    const size_t T = F->T, H = F->n_hc, D = F->dim;
    F->ids = (const int32_t *)q; q += T * 4;
    F->cols = (const uint64_t *)q; q += T * N_COLS * 8;
    F->rows = q; q += T * N_COLS * ROW_BYTES;
    F->h_bits = (const uint16_t *)q; q += T * H * D * 2;
    F->weight = (const float *)q; q += H * D * 4;
    F->kv = (const float *)q; q += T * (H + 1) * D * 4;
    F->dot = (const float *)q; q += T * H * 4;
    F->gate = (const float *)q; q += T * H * 4;
    F->h_out = (const float *)q; q += T * H * D * 4;
    return (size_t)(q - p) == n;
}

/* one run of the whole path; fills kv_out and h_out (host copies) */
static int run_path(const fixture *F, const void *map_key, uint64_t map_size, pulsar_gpu_tensor *rows,
                    pulsar_gpu_tensor *x_key, pulsar_gpu_tensor *kv, pulsar_gpu_tensor *hc,
                    pulsar_gpu_tensor *qk, pulsar_gpu_tensor *dead, int decode,
                    float *kv_host, uint16_t *h_host) {
    const uint32_t T = F->T, H = F->n_hc, D = F->dim;
    pulsar_gpu_tensor_write(hc, 0, F->h_bits, (uint64_t)T * H * D * 2);
    pulsar_gpu_matmul_set_batch_decode_rows(decode ? (int)T : 0);
    int ok = pulsar_gpu_engram_rows_emit(x_key, rows, T);
    if (ok) ok = pulsar_gpu_matmul_mxfp8_tensor(kv, map_key, map_size, 0, WKV_IN, WKV_OUT, x_key, T);
    pulsar_gpu_matmul_set_batch_decode_rows(0);
    pulsar_gpu_act_slot_drop(x_key);
    if (ok) ok = pulsar_gpu_engram_gate_add(hc, kv, qk, dead, T, H, D, F->eps);
    if (ok) ok = pulsar_gpu_tensor_read(kv, 0, kv_host, (uint64_t)T * (H + 1) * D * 4);
    if (ok) ok = pulsar_gpu_tensor_read(hc, 0, h_host, (uint64_t)T * H * D * 2);
    return ok;
}

static void grade(const fixture *F, const float *kv_host, const uint16_t *h_host, const char *regime, int expect_dead0) {
    const uint32_t T = F->T, H = F->n_hc, D = F->dim;
    /* A. kv */
    float kv_max = 0.f, kv_err = 0.f;
    for (size_t i = 0; i < (size_t)T * (H + 1) * D; i++) {
        kv_max = fmaxf(kv_max, fabsf(F->kv[i]));
        kv_err = fmaxf(kv_err, fabsf(kv_host[i] - F->kv[i]));
    }
    CHECK(kv_err <= 1e-4f * kv_max, "%s kv: max |ours-ref| %.3e vs max|ref| %.3f (%.2e relative; bound 1e-4)",
          regime, kv_err, kv_max, kv_err / kv_max);
    /* B. gate from OUR kv, on the host, vs the fixture's gate */
    float gate_err = 0.f;
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t c = 0; c < H; c++) {
            double sh = 0, sk = 0, sd = 0;
            const float *key = kv_host + ((size_t)t * (H + 1) + c) * D;
            for (uint32_t i = 0; i < D; i++) {
                const float h = bf16f(F->h_bits[((size_t)t * H + c) * D + i]);
                sh += (double)h * h; sk += (double)key[i] * key[i]; sd += (double)h * F->weight[(size_t)c * D + i] * key[i];
            }
            const double rstd = 1.0 / sqrt(sh / D + F->eps) / sqrt(sk / D + F->eps);
            const double dot = sd * rstd / sqrt((double)D);
            const double g = 1.0 / (1.0 + exp(-copysign(sqrt(fmax(fabs(dot), 1e-6)), dot)));
            gate_err = fmaxf(gate_err, fabsf((float)g - F->gate[(size_t)t * H + c]));
        }
    }
    CHECK(gate_err <= 1e-5f, "%s gate (host, from our kv) vs reference: max |delta| %.2e (bound 1e-5)", regime, gate_err);
    /* C. h_out: |ours - ref| <= one bf16 ulp of the reference (the one rounding) plus
     * this arm's measured GEMM error (the value's absolute error, gate <= 1 -- near
     * zero the ulp is smaller than that error, so the bound has both terms);
     * E. dead token untouched */
    size_t over = 0, changed_dead = 0, checked = 0;
    float worst = 0.f;
    for (uint32_t t = 0; t < T; t++) {
        for (size_t i = 0; i < (size_t)H * D; i++) {
            const size_t k = (size_t)t * H * D + i;
            if (expect_dead0 && t == 0) { if (h_host[k] != F->h_bits[k]) changed_dead++; continue; }
            const float ref = F->h_out[k], ours = bf16f(h_host[k]);
            const float a = fabsf(ref);
            const float ulp = a > 0.f ? exp2f(floorf(log2f(a)) - 7.0f) : 0.f;
            const float err = fabsf(ours - ref);
            if (err > ulp + kv_err) over++;
            if (err - ulp > worst) worst = err - ulp;
            checked++;
        }
    }
    CHECK(over == 0, "%s h_out: %zu of %zu elements beyond 1 bf16 ulp + the arm's GEMM error %.2e (worst excess %.2e)",
          regime, over, checked, kv_err, worst);
    if (expect_dead0) CHECK(changed_dead == 0, "%s dead token 0: %zu elements changed (must be 0)", regime, changed_dead);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s FIXTURE.fix ENGRAM_DIR\n", argv[0]); return 2; }
    size_t fn = 0;
    uint8_t *fbuf = (uint8_t *)read_file(argv[1], &fn);
    fixture F;
    if (!fbuf || !parse_fixture(fbuf, fn, &F)) { fprintf(stderr, "bad fixture %s\n", argv[1]); return 2; }
    char wkv_p[4096], qk_p[4096];
    snprintf(wkv_p, sizeof wkv_p, "%s/engram-l%u.wkv", argv[2], F.layer);
    snprintf(qk_p, sizeof qk_p, "%s/engram-l%u.qk", argv[2], F.layer);
    size_t wkv_n = 0, qk_n = 0;
    uint8_t *wkv = (uint8_t *)read_file(wkv_p, &wkv_n);
    uint8_t *qk = (uint8_t *)read_file(qk_p, &qk_n);
    const size_t data_bytes = (size_t)WKV_OUT * WKV_IN, KBp = (WKV_IN / 32 + 3) / 4 * 4;
    const size_t scale_bytes = (size_t)((WKV_OUT + 127) / 128 * 128) * KBp;
    if (!wkv || wkv_n != data_bytes + scale_bytes) { fprintf(stderr, "bad wkv file %s (%zu bytes, want %zu)\n", wkv_p, wkv_n, data_bytes + scale_bytes); return 2; }
    if (!qk || qk_n != (size_t)F.n_hc * F.dim * 2 * 2) { fprintf(stderr, "bad qk file %s\n", qk_p); return 2; }
    /* q*k product, f32, from the aux file; must equal the fixture's `weight` bit for bit
     * (both are the same bf16 products, formed in f32) */
    float *weight = (float *)malloc((size_t)F.n_hc * F.dim * 4);
    const uint16_t *qb = (const uint16_t *)qk, *kb = qb + (size_t)F.n_hc * F.dim;
    for (size_t i = 0; i < (size_t)F.n_hc * F.dim; i++) weight[i] = bf16f(qb[i]) * bf16f(kb[i]);
    printf("engram gate: layer %u, %u tokens x %u copies x %u, eps %.1e, rows [%llu..%llu]\n",
           F.layer, F.T, F.n_hc, F.dim, (double)F.eps,
           (unsigned long long)F.cols[0], (unsigned long long)F.cols[(size_t)F.T * N_COLS - 1]);
    CHECK(memcmp(weight, F.weight, (size_t)F.n_hc * F.dim * 4) == 0, "q*k product from the aux file == the fixture's");

    if (!pulsar_gpu_init()) { fprintf(stderr, "no GPU\n"); return 2; }   /* 1 = up, like every pulsar_gpu_* */
    const uint32_t T = F.T, H = F.n_hc, D = F.dim;
    pulsar_gpu_tensor *wdata = pulsar_gpu_tensor_alloc(data_bytes), *wscale = pulsar_gpu_tensor_alloc(scale_bytes);
    pulsar_gpu_tensor *rows = pulsar_gpu_tensor_alloc((uint64_t)T * N_COLS * ROW_BYTES);
    pulsar_gpu_tensor *x_key = pulsar_gpu_tensor_alloc((uint64_t)T * WKV_IN * 4);
    pulsar_gpu_tensor *kv = pulsar_gpu_tensor_alloc((uint64_t)T * WKV_OUT * 4);
    pulsar_gpu_tensor *hc = pulsar_gpu_tensor_alloc((uint64_t)T * H * D * 2);
    pulsar_gpu_tensor *qkw = pulsar_gpu_tensor_alloc((uint64_t)H * D * 4);
    pulsar_gpu_tensor *dead = pulsar_gpu_tensor_alloc(T);
    if (!wdata || !wscale || !rows || !x_key || !kv || !hc || !qkw || !dead) { fprintf(stderr, "alloc\n"); return 2; }
    pulsar_gpu_tensor_write(wdata, 0, wkv, data_bytes);
    pulsar_gpu_tensor_write(wscale, 0, wkv + data_bytes, scale_bytes);
    pulsar_gpu_tensor_write(rows, 0, F.rows, (uint64_t)T * N_COLS * ROW_BYTES);
    pulsar_gpu_tensor_write(qkw, 0, weight, (uint64_t)H * D * 4);
    const void *map_key = wkv;   /* unique to this weight set */
    if (!pulsar_gpu_register_fp8_lt_weight_resident(map_key, 0, WKV_IN, WKV_OUT, wdata, wscale)) {
        fprintf(stderr, "resident wkv registration refused\n"); return 2;
    }
    float *kv_a = (float *)malloc((size_t)T * WKV_OUT * 4), *kv_b = (float *)malloc((size_t)T * WKV_OUT * 4);
    uint16_t *h_a = (uint16_t *)malloc((size_t)T * H * D * 2), *h_b = (uint16_t *)malloc((size_t)T * H * D * 2);

    /* decode arm, twice (D. identity) */
    if (!run_path(&F, map_key, wkv_n, rows, x_key, kv, hc, qkw, NULL, 1, kv_a, h_a)) { CHECK(0, "decode-arm path refused"); }
    else {
        grade(&F, kv_a, h_a, "decode arm", 0);
        if (!run_path(&F, map_key, wkv_n, rows, x_key, kv, hc, qkw, NULL, 1, kv_b, h_b)) CHECK(0, "decode-arm second run refused");
        else CHECK(memcmp(kv_a, kv_b, (size_t)T * WKV_OUT * 4) == 0 && memcmp(h_a, h_b, (size_t)T * H * D * 2) == 0,
                   "decode arm: two runs byte-identical (kv and h_out)");
    }
    /* prefill arm */
    if (!run_path(&F, map_key, wkv_n, rows, x_key, kv, hc, qkw, NULL, 0, kv_a, h_a)) CHECK(0, "prefill-arm path refused");
    else grade(&F, kv_a, h_a, "prefill arm", 0);
    /* dead token: mark token 0 dead; its copies must come back untouched, the rest as before */
    uint8_t *dm = (uint8_t *)calloc(T, 1); dm[0] = 1;
    pulsar_gpu_tensor_write(dead, 0, dm, T);
    if (!run_path(&F, map_key, wkv_n, rows, x_key, kv, hc, qkw, dead, 1, kv_a, h_a)) CHECK(0, "dead-mask path refused");
    else grade(&F, kv_a, h_a, "decode arm + dead token 0", 1);

    printf("ENGRAM GATE: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}

/* lora_rows_gate -- L338: the runtime LoRA primitive (pulsar_rows_lora_launch) graded.  Model-free; seconds.
 *
 *   L1  against the f64 reference: y + scale * B (A x) for Gemma 4's shapes (in 2816 -> out 4096 / 8192 / 2048 /
 *       2112 at the stored pitch 2176, in 2112 at pitch 2176 -> 2816) and odd ones, rank 32 and others: |err| <=
 *       1e-5 * (|y| + scale * sum |B| |A x|-terms) per element (f32 accumulation of <= 2816 + 64 terms)
 *   L2  untouched: y's columns past `out` and x's past `in` (a padded stored width) neither read nor written
 *   L3  row independence: every row of a 37-row call == that row computed alone (1-row call), byte for byte
 *   L4  determinism: two calls on the same inputs, byte for byte
 *   L5  refusals: rank 0, in > ld_x, out > ld_y, NaN scale
 *   L6  a row mask: the marked rows == the full call's, the unmarked rows == the base y, byte for byte
 * Exit 0 = every check passed. */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "cuda/pulsar_cuda_rowops.h"

static int n_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { n_fail++; printf("  FAIL  "); printf(__VA_ARGS__); printf("\n"); } else { printf("  ok    "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint64_t rs = 0x2545F4914F6CDD1Dull;
static float urand() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; }

struct dev { void *p = nullptr; explicit dev(size_t n) { cudaMalloc(&p, n); } ~dev() { cudaFree(p); } };

static uint16_t to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
static float from_bf16(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

/* one case: returns false on a launch failure */
static bool run_case(uint32_t T, uint32_t in, uint64_t ld_x, uint32_t r, uint32_t out, uint64_t ld_y, float scale,
                     const char *what) {
    std::vector<uint16_t> x(T * ld_x);
    std::vector<float> A((size_t)r * in), B((size_t)out * r), y0(T * ld_y);
    for (auto &v : x) v = to_bf16(urand());
    for (auto &v : A) v = urand() * 0.02f;
    for (auto &v : B) v = urand() * 0.02f;
    for (auto &v : y0) v = urand();
    dev dx(x.size() * 2), dA(A.size() * 4), dB(B.size() * 4), dy(y0.size() * 4), dw((size_t)T * r * 4), dy2(y0.size() * 4);
    cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dA.p, A.data(), A.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB.p, B.data(), B.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dy.p, y0.data(), y0.size() * 4, cudaMemcpyHostToDevice);
    if (pulsar_rows_lora_launch((const uint16_t *)dx.p, ld_x, T, in, (float *)dA.p, r, (float *)dB.p, out, scale,
                                (float *)dy.p, ld_y, (float *)dw.p, NULL, 0) != 0 || cudaDeviceSynchronize() != cudaSuccess)
        return false;
    std::vector<float> y(y0.size());
    cudaMemcpy(y.data(), dy.p, y.size() * 4, cudaMemcpyDeviceToHost);
    /* L1 / L2 */
    double worst = 0.0;
    bool untouched = true;
    for (uint32_t t = 0; t < T; t++) {
        std::vector<double> u(r), ua(r);
        for (uint32_t j = 0; j < r; j++)
            for (uint32_t k = 0; k < in; k++) {
                const double p = (double)A[j * in + k] * from_bf16(x[t * ld_x + k]);
                u[j] += p;
                ua[j] += fabs(p);
            }
        for (uint64_t o = 0; o < ld_y; o++) {
            const float got = y[t * ld_y + o];
            if (o >= out) {
                untouched &= memcmp(&got, &y0[t * ld_y + o], 4) == 0;
                continue;
            }
            double e = y0[t * ld_y + o], S = fabs(y0[t * ld_y + o]);
            for (uint32_t j = 0; j < r; j++) {
                e += scale * (double)B[o * r + j] * u[j];
                S += fabs(scale * (double)B[o * r + j]) * ua[j];
            }
            const double rel = S > 0 ? fabs(got - e) / S : 0.0;
            if (rel > worst) worst = rel;
        }
    }
    CHECK(worst <= 1e-5, "L1 %s: %u rows, %u -> rank %u -> %u: worst |err| / (|y| + sum |terms|) %.3g (bound 1e-5)",
          what, T, in, r, out, worst);
    CHECK(untouched, "L2 %s: columns past out (%u of pitch %llu) untouched", what, out, (unsigned long long)ld_y);
    /* L4: the same call again */
    cudaMemcpy(dy2.p, y0.data(), y0.size() * 4, cudaMemcpyHostToDevice);
    pulsar_rows_lora_launch((const uint16_t *)dx.p, ld_x, T, in, (float *)dA.p, r, (float *)dB.p, out, scale,
                            (float *)dy2.p, ld_y, (float *)dw.p, NULL, 0);
    std::vector<float> y2(y0.size());
    cudaMemcpy(y2.data(), dy2.p, y2.size() * 4, cudaMemcpyDeviceToHost);
    CHECK(memcmp(y.data(), y2.data(), y.size() * 4) == 0, "L4 %s: two calls byte-identical", what);
    /* L3: each row alone */
    bool same = true;
    for (uint32_t t = 0; t < T && same; t++) {
        cudaMemcpy(dy2.p, y0.data() + t * ld_y, ld_y * 4, cudaMemcpyHostToDevice);
        pulsar_rows_lora_launch((const uint16_t *)dx.p + t * ld_x, ld_x, 1, in, (float *)dA.p, r, (float *)dB.p, out,
                                scale, (float *)dy2.p, ld_y, (float *)dw.p, NULL, 0);
        std::vector<float> one(ld_y);
        cudaMemcpy(one.data(), dy2.p, ld_y * 4, cudaMemcpyDeviceToHost);
        same = memcmp(one.data(), y.data() + t * ld_y, ld_y * 4) == 0;
    }
    CHECK(same, "L3 %s: every row of the %u-row call == that row alone, byte for byte", what, T);
    /* L6: a row mask -- marked rows == the unmasked call's, unmarked rows == y0, byte for byte */
    std::vector<uint8_t> mask(T);
    for (uint32_t t = 0; t < T; t++) mask[t] = (uint8_t)(t % 3 != 1);
    dev dm(T);
    cudaMemcpy(dm.p, mask.data(), T, cudaMemcpyHostToDevice);
    cudaMemcpy(dy2.p, y0.data(), y0.size() * 4, cudaMemcpyHostToDevice);
    pulsar_rows_lora_launch((const uint16_t *)dx.p, ld_x, T, in, (float *)dA.p, r, (float *)dB.p, out, scale,
                            (float *)dy2.p, ld_y, (float *)dw.p, (const uint8_t *)dm.p, 0);
    cudaMemcpy(y2.data(), dy2.p, y2.size() * 4, cudaMemcpyDeviceToHost);
    bool masked = true;
    for (uint32_t t = 0; t < T; t++)
        masked &= memcmp(y2.data() + t * ld_y, (mask[t] ? y.data() : y0.data()) + t * ld_y, ld_y * 4) == 0;
    CHECK(masked, "L6 %s: a row mask -- marked rows == the full call's, the others == the base, byte for byte", what);
    return true;
}

int main() {
    struct c { uint32_t T, in; uint64_t ld_x; uint32_t r, out; uint64_t ld_y; float s; const char *what; };
    const c cases[] = {
        {37, 2816, 2816, 32, 4096, 4096, 2.0f, "q (window)"},
        {37, 2816, 2816, 32, 8192, 8192, 2.0f, "q (global)"},
        {37, 2816, 2816, 32, 2048, 2048, 2.0f, "k / v (window)"},
        {37, 2816, 2816, 32, 2112, 2176, 2.0f, "MLP gate / up at the stored pitch"},
        {37, 2112, 2176, 32, 2816, 2816, 2.0f, "MLP down from the stored pitch"},
        {37, 8192, 8192, 32, 2816, 2816, 2.0f, "o (global)"},
        {5, 100, 104, 7, 33, 40, 0.5f, "odd widths, rank 7"},
        {1, 64, 64, 64, 64, 64, 1.0f, "rank 64, one row"},
    };
    for (const c &k : cases)
        if (!run_case(k.T, k.in, k.ld_x, k.r, k.out, k.ld_y, k.s, k.what)) { n_fail++; printf("  FAIL  %s: launch\n", k.what); }
    dev d(1 << 20);
    float *f = (float *)d.p;
    const uint16_t *x = (const uint16_t *)d.p;
    CHECK(pulsar_rows_lora_launch(x, 64, 1, 64, f, 0, f, 64, 1.0f, f, 64, f, NULL, 0) == -1, "L5 rank 0 refused");
    CHECK(pulsar_rows_lora_launch(x, 32, 1, 64, f, 4, f, 64, 1.0f, f, 64, f, NULL, 0) == -1, "L5 in > ld_x refused");
    CHECK(pulsar_rows_lora_launch(x, 64, 1, 64, f, 4, f, 65, 1.0f, f, 64, f, NULL, 0) == -1, "L5 out > ld_y refused");
    CHECK(pulsar_rows_lora_launch(x, 64, 1, 64, f, 4, f, 64, NAN, f, 64, f, NULL, 0) == -1, "L5 a NaN scale refused");
    printf("\nlora_rows_gate: %s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}

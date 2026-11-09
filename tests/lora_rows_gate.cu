/* lora_rows_gate -- L338: the runtime LoRA primitive (pulsar_rows_lora_launch, the fp16 tensor cores) graded.
 * Model-free; seconds.  Both row kinds (prompt / decode) unless a check says otherwise.
 *
 *   L1  against the f64 reference y + scale * B (A x) on the f16 pairs as stored, for Gemma 4's shapes (in 2816 -> out
 *       4096 / 8192 / 2048 / 2112 at the stored pitch 2176, in 2112 at pitch 2176 -> 2816, 8192 -> 2816) and others:
 *       |err| <= 2^-10 * (|y| + scale * sum |B| |A x|-terms) per element -- the f16 rounding of the rank-r intermediate
 *       (2^-11 relative) plus f32 accumulation
 *   L2  untouched: y's columns past `out` (a padded stored width)
 *   L3  row independence: every row of a call == that row computed alone (a 1-row call), byte for byte
 *   L4  determinism: two calls on the same inputs, byte for byte
 *   L5  refusals: rank 0, a width not a multiple of 8, in > ld_x, out > ld_y, a NaN scale, a short workspace
 *   L6  a row mask: the marked rows == the full call's, the unmarked rows == the base y, byte for byte
 *   L7  a prompt cut anywhere == the prompt whole (two calls, 20 + 17 rows, == one of 37), byte for byte
 *   L8  more rows than one block (PULSAR_ROWS_LORA_BLOCK): 300 rows == each 100-row third called alone
 * Exit 0 = every check passed. */
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "cuda/pulsar_cuda_f16_rows.h"

static int n_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { n_fail++; printf("  FAIL  "); printf(__VA_ARGS__); printf("\n"); } else { printf("  ok    "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint64_t rs = 0x2545F4914F6CDD1Dull;
static float urand() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; }

struct dev { void *p = nullptr; explicit dev(size_t n) { cudaMalloc(&p, n ? n : 1); } ~dev() { cudaFree(p); } };

static uint16_t to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
static float from_bf16(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t to_f16(float f) { return __half_as_ushort(__float2half_rn(f)); }
static float from_f16(uint16_t u) { return __half2float(__ushort_as_half(u)); }

/* y (device, ld_y) <- y0, then the LoRA over rows [t0, t0 + n) of x on the row kind; false on a refusal */
static bool call(const dev &dx, uint64_t ld_x, uint32_t t0, uint32_t n, const pulsar_rows_lora &l, const dev &dy,
                 uint64_t ld_y, const uint8_t *mask, bool prompt, const dev &ws, size_t ws_bytes) {
    return pulsar_rows_lora_launch((const uint16_t *)dx.p + (size_t)t0 * ld_x, ld_x, (int)n, &l,
                                   (float *)dy.p + (size_t)t0 * ld_y, ld_y, mask ? mask + t0 : NULL, prompt, ws.p,
                                   ws_bytes, 0) == 0;
}

static std::vector<float> get(const dev &d, size_t n) {
    std::vector<float> h(n);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d.p, n * 4, cudaMemcpyDeviceToHost);
    return h;
}

/* one case: returns false on a refusal / launch failure */
static bool run_case(uint32_t T, uint32_t in, uint64_t ld_x, uint32_t r, uint32_t out, uint64_t ld_y, float scale,
                     const char *what) {
    std::vector<uint16_t> x(T * ld_x), A((size_t)r * in), B((size_t)out * r);
    std::vector<float> y0(T * ld_y);
    for (auto &v : x) v = to_bf16(urand() * (urand() > 0.9f ? 40.0f : 1.0f));   /* a few large values per row */
    for (auto &v : A) v = to_f16(urand() * 0.02f);
    for (auto &v : B) v = to_f16(urand() * 0.02f);
    for (auto &v : y0) v = urand();
    dev dx(x.size() * 2), dA(A.size() * 2), dB(B.size() * 2), dy(y0.size() * 4), dy2(y0.size() * 4);
    const size_t wsb = pulsar_rows_lora_workspace_bytes((int)T, (int)in, (int)r, (int)out);
    dev ws(wsb);
    cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dA.p, A.data(), A.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dB.p, B.data(), B.size() * 2, cudaMemcpyHostToDevice);
    pulsar_rows_lora l{dA.p, dB.p, r, in, out, scale};
    for (int pr = 1; pr >= 0; pr--) {
        const bool prompt = pr != 0;
        const char *kind = prompt ? "prompt" : "decode";
        cudaMemcpy(dy.p, y0.data(), y0.size() * 4, cudaMemcpyHostToDevice);
        if (!call(dx, ld_x, 0, T, l, dy, ld_y, NULL, prompt, ws, wsb)) return false;
        const std::vector<float> y = get(dy, y0.size());
        /* L1 / L2 */
        double worst = 0.0;
        bool untouched = true;
        for (uint32_t t = 0; t < T; t++) {
            std::vector<double> u(r), ua(r);
            for (uint32_t j = 0; j < r; j++)
                for (uint32_t k = 0; k < in; k++) {
                    const double p = (double)from_f16(A[j * in + k]) * from_bf16(x[t * ld_x + k]);
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
                    const double b = scale * (double)from_f16(B[o * r + j]);
                    e += b * u[j];
                    S += fabs(b) * ua[j];
                }
                const double rel = S > 0 ? fabs(got - e) / S : 0.0;
                if (rel > worst) worst = rel;
            }
        }
        CHECK(worst <= 1.0 / 1024.0, "L1 %s %s: %u rows, %u -> rank %u -> %u: worst |err| / (|y| + sum |terms|) %.3g "
              "(bound 2^-10)", what, kind, T, in, r, out, worst);
        CHECK(untouched, "L2 %s %s: columns past out (%u of pitch %llu) untouched", what, kind, out,
              (unsigned long long)ld_y);
        /* L4 */
        cudaMemcpy(dy2.p, y0.data(), y0.size() * 4, cudaMemcpyHostToDevice);
        call(dx, ld_x, 0, T, l, dy2, ld_y, NULL, prompt, ws, wsb);
        CHECK(get(dy2, y0.size()) == y, "L4 %s %s: two calls byte-identical", what, kind);
        /* L3 */
        cudaMemcpy(dy2.p, y0.data(), y0.size() * 4, cudaMemcpyHostToDevice);
        for (uint32_t t = 0; t < T; t++) call(dx, ld_x, t, 1, l, dy2, ld_y, NULL, prompt, ws, wsb);
        CHECK(memcmp(get(dy2, y0.size()).data(), y.data(), y.size() * 4) == 0,
              "L3 %s %s: every row of the %u-row call == that row alone, byte for byte", what, kind, T);
        /* L7 */
        if (T > 20) {
            cudaMemcpy(dy2.p, y0.data(), y0.size() * 4, cudaMemcpyHostToDevice);
            call(dx, ld_x, 0, 20, l, dy2, ld_y, NULL, prompt, ws, wsb);
            call(dx, ld_x, 20, T - 20, l, dy2, ld_y, NULL, prompt, ws, wsb);
            CHECK(memcmp(get(dy2, y0.size()).data(), y.data(), y.size() * 4) == 0,
                  "L7 %s %s: 20 + %u rows in two calls == the %u whole, byte for byte", what, kind, T - 20, T);
        }
        /* L6 */
        std::vector<uint8_t> mask(T);
        for (uint32_t t = 0; t < T; t++) mask[t] = (uint8_t)(t % 3 != 1);
        dev dm(T);
        cudaMemcpy(dm.p, mask.data(), T, cudaMemcpyHostToDevice);
        cudaMemcpy(dy2.p, y0.data(), y0.size() * 4, cudaMemcpyHostToDevice);
        call(dx, ld_x, 0, T, l, dy2, ld_y, (const uint8_t *)dm.p, prompt, ws, wsb);
        const std::vector<float> ym = get(dy2, y0.size());
        bool masked = true;
        for (uint32_t t = 0; t < T; t++)
            masked &= memcmp(ym.data() + t * ld_y, (mask[t] ? y.data() : y0.data()) + t * ld_y, ld_y * 4) == 0;
        CHECK(masked, "L6 %s %s: a row mask -- marked rows == the full call's, the others == the base, byte for byte",
              what, kind);
        /* L8 */
        if (T > PULSAR_ROWS_LORA_BLOCK) {
            const uint32_t third = T / 3;
            cudaMemcpy(dy2.p, y0.data(), y0.size() * 4, cudaMemcpyHostToDevice);
            call(dx, ld_x, 0, third, l, dy2, ld_y, NULL, prompt, ws, wsb);
            call(dx, ld_x, third, third, l, dy2, ld_y, NULL, prompt, ws, wsb);
            call(dx, ld_x, 2 * third, T - 2 * third, l, dy2, ld_y, NULL, prompt, ws, wsb);
            CHECK(memcmp(get(dy2, y0.size()).data(), y.data(), y.size() * 4) == 0,
                  "L8 %s %s: %u rows (> one %d-row block) == its thirds called alone, byte for byte", what, kind, T,
                  PULSAR_ROWS_LORA_BLOCK);
        }
    }
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
        {300, 2816, 2816, 32, 2816, 2816, 2.0f, "300 rows (> one block)"},
        {5, 104, 112, 8, 40, 48, 0.5f, "narrow widths, rank 8"},
        {1, 64, 64, 64, 64, 64, 1.0f, "rank 64, one row"},
    };
    for (const c &k : cases)
        if (!run_case(k.T, k.in, k.ld_x, k.r, k.out, k.ld_y, k.s, k.what)) { n_fail++; printf("  FAIL  %s: refused\n", k.what); }
    dev d(1 << 22);
    float *f = (float *)d.p;
    const uint16_t *x = (const uint16_t *)d.p;
    const size_t big = 1 << 22;
    auto L5 = [&](pulsar_rows_lora l, uint64_t ld_x, uint64_t ld_y, size_t ws_bytes) {
        return pulsar_rows_lora_launch(x, ld_x, 1, &l, f, ld_y, NULL, true, d.p, ws_bytes, 0) == -1;
    };
    CHECK(L5({d.p, d.p, 0, 64, 64, 1.0f}, 64, 64, big), "L5 rank 0 refused");
    CHECK(L5({d.p, d.p, 4, 64, 64, 1.0f}, 64, 64, big), "L5 a rank not a multiple of 8 refused");
    CHECK(L5({d.p, d.p, 8, 64, 64, 1.0f}, 32, 64, big), "L5 in > ld_x refused");
    CHECK(L5({d.p, d.p, 8, 64, 72, 1.0f}, 64, 64, big), "L5 out > ld_y refused");
    CHECK(L5({d.p, d.p, 8, 64, 64, NAN}, 64, 64, big), "L5 a NaN scale refused");
    CHECK(L5({d.p, d.p, 8, 64, 64, 1.0f}, 64, 64, 16), "L5 a short workspace refused");
    printf("\nlora_rows_gate: %s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}

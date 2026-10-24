/* POOL GATE (L321): the encoders' mean pool (src/cuda/pulsar_cuda_pool.cu, the production object) -- model-free.
 *
 *   1. bit-exact: out[c] == a host loop in the same arithmetic (f32 adds in row order, one f32 divide) at n = 1, 7,
 *      600 and 16384 rows of 4096 (an encoder's widest input), from heavy-tailed bf16 rows at a padded ld;
 *   2. the canary: every output written (no NaN left from a 0xff fill);
 *   3. a row's place matters only through the order: the same rows in a call at another ld give the same bits;
 *   4. refusals: a null pointer, no rows, a width or ld not a multiple of 8, a misaligned pointer.
 *
 * usage: ./tests/pool_gate */
#include "../src/cuda/pulsar_cuda_pool.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "POOL FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); g_fail = 1; } } while (0)
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", \
                  cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 16); }
static double rndn(void) {
    const double u1 = ((double)rnd() + 0.5) / 4294967296.0, u2 = ((double)rnd() + 0.5) / 4294967296.0;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}
static uint16_t bf16_bits(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    u += 0x7fffu + ((u >> 16) & 1u);   /* round to nearest even */
    return (uint16_t)(u >> 16);
}
static float bf16_val(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

int main(void) {
    const uint32_t W = 4096, LD = W + 8;
    printf("pool-gate: the encoders' mean pool (production object) vs the same arithmetic on the host\n");
    for (uint32_t n : {1u, 7u, 600u, 16384u}) {
        std::vector<uint16_t> h((size_t)n * LD, 0);
        for (uint32_t t = 0; t < n; t++)
            for (uint32_t c = 0; c < W; c++)
                h[(size_t)t * LD + c] = bf16_bits((float)(rndn() * (c % 97 == 3 ? 300.0 : 2.0)));   /* outlier columns */
        std::vector<float> want(W);
        for (uint32_t c = 0; c < W; c++) {
            volatile float acc = 0.0f;
            for (uint32_t t = 0; t < n; t++) acc = acc + bf16_val(h[(size_t)t * LD + c]);
            want[c] = acc / (float)n;
        }
        uint16_t *dx = nullptr, *dx2 = nullptr;
        float *dout = nullptr;
        CK(cudaMalloc(&dx, h.size() * 2));
        CK(cudaMemcpy(dx, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dout, W * 4));
        CK(cudaMemset(dout, 0xff, W * 4));
        int rc = pulsar_rows_mean_f32_launch(dx, LD, n, W, dout, 0);
        CK(cudaDeviceSynchronize());
        std::vector<float> got(W);
        CK(cudaMemcpy(got.data(), dout, W * 4, cudaMemcpyDeviceToHost));
        size_t diff = 0, nan = 0;
        for (uint32_t c = 0; c < W; c++) {
            nan += isnan(got[c]);
            diff += memcmp(&got[c], &want[c], 4) != 0;
        }
        CHECK(rc == 0 && diff == 0 && nan == 0, "n=%u: rc %d, %zu of %u columns differ from the host loop, %zu NaN",
              n, rc, diff, W, nan);
        /* the same rows packed at ld = W: the same bits */
        std::vector<uint16_t> p((size_t)n * W);
        for (uint32_t t = 0; t < n; t++) memcpy(&p[(size_t)t * W], &h[(size_t)t * LD], W * 2);
        CK(cudaMalloc(&dx2, p.size() * 2));
        CK(cudaMemcpy(dx2, p.data(), p.size() * 2, cudaMemcpyHostToDevice));
        rc = pulsar_rows_mean_f32_launch(dx2, W, n, W, dout, 0);
        CK(cudaDeviceSynchronize());
        std::vector<float> got2(W);
        CK(cudaMemcpy(got2.data(), dout, W * 4, cudaMemcpyDeviceToHost));
        CHECK(rc == 0 && memcmp(got2.data(), got.data(), W * 4) == 0, "n=%u: ld %u and ld %u disagree", n, LD, W);
        printf("  n=%-5u x %u: bit-exact to the host loop, every column written, ld-independent\n", n, W);
        cudaFree(dx); cudaFree(dx2); cudaFree(dout);
    }
    uint16_t *dx = nullptr;
    float *dout = nullptr;
    CK(cudaMalloc(&dx, 64 * 2)); CK(cudaMalloc(&dout, 64 * 4));
    CHECK(pulsar_rows_mean_f32_launch(nullptr, 8, 1, 8, dout, 0) == -1, "a null x accepted");
    CHECK(pulsar_rows_mean_f32_launch(dx, 8, 0, 8, dout, 0) == -1, "no rows accepted");
    CHECK(pulsar_rows_mean_f32_launch(dx, 8, 1, 12, dout, 0) == -1, "width 12 accepted");
    CHECK(pulsar_rows_mean_f32_launch(dx, 12, 1, 8, dout, 0) == -1, "ld 12 accepted");
    CHECK(pulsar_rows_mean_f32_launch(dx + 1, 8, 1, 8, dout, 0) == -1, "a misaligned x accepted");
    printf("  refusals: null, no rows, width / ld not a multiple of 8, misaligned -- all refused\n");
    cudaFree(dx); cudaFree(dout);
    printf(g_fail ? "POOL GATE FAIL\n" : "POOL GATE PASS\n");
    return g_fail;
}

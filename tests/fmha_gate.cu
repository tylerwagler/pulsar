/* FMHA GATE (L321, L331): the bidirectional fused attention over f16 and bf16 rows (src/cuda/pulsar_cuda_fmha.cu,
 * CUTLASS's memory-efficient attention) -- the production objects, model-free, at the RADIO ViTs' layouts: q | k | v the
 * three column blocks of a fused qkv row, the output its own rows.  f16: Nemotron Parse's (16 heads of 72); bf16:
 * Nemotron 3.5 Super VL's C-RADIO ViT-H (16 heads of 80).
 *
 *   A1  vs a double host reference, n = 1, 37, 600, 2001 rows: |y - ref| <= bound (|ref| + mean_k |v|) per element --
 *       4e-3 for f16 (f16 operands, output and probabilities), 3.2e-2 for bf16 (eight times: bf16 keeps 3 fewer
 *       mantissa bits in the operands, the probabilities and the output) -- every output written;
 *   A2  a page / a tile: 12298 rows (f16: 10 class tokens + 96 x 128 patches), 13322 (bf16: 10 + the largest tile's
 *       13312 patches), its time (info) and every output finite;
 *   A3  refusals: a head dim not a multiple of 8, one past 128, a misaligned pointer, a leading dimension short of the
 *       heads.
 *
 * usage: ./tests/fmha_gate */
#include "../src/cuda/pulsar_cuda_fmha.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <type_traits>
#include <vector>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FMHA FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); g_fail = 1; } } while (0)
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", \
                  cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 16); }
static double rndn(void) {
    const double u1 = ((double)rnd() + 0.5) / 4294967296.0, u2 = ((double)rnd() + 0.5) / 4294967296.0;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

static const uint32_t H = 16;

/* one dtype's instance: T __half (f16_launch) or __nv_bfloat16 (bf16_launch) */
template <class T> static float tf(T v);
template <> float tf<__half>(__half v) { return __half2float(v); }
template <> float tf<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <class T> static T ft(float v);
template <> __half ft<__half>(float v) { return __float2half(v); }
template <> __nv_bfloat16 ft<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <class T>
static int run(uint32_t n, uint32_t HD, bool grade, const char *what) {
    constexpr bool bf = std::is_same<T, __nv_bfloat16>::value;
    const uint32_t W = H * HD, LD = 3 * W;
    const double bound = bf ? 3.2e-2 : 4e-3;
    std::vector<T> qkv((size_t)n * LD);
    for (auto &x : qkv) x = ft<T>((float)(rndn() * 1.5));
    T *d = nullptr, *dy = nullptr;
    CK(cudaMalloc(&d, qkv.size() * 2));
    CK(cudaMemcpy(d, qkv.data(), qkv.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dy, (size_t)n * W * 2));
    CK(cudaMemset(dy, 0xff, (size_t)n * W * 2));
    const float scale = 1.0f / sqrtf((float)HD);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    const uint16_t *q = (const uint16_t *)d;
    const int rc = (bf ? pulsar_fmha_bf16_launch : pulsar_fmha_f16_launch)(q, LD, q + W, LD, q + 2 * W, LD, n, H, HD,
                                                                            scale, (uint16_t *)dy, W, 0);
    cudaEventRecord(e1);
    CK(cudaDeviceSynchronize());
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    std::vector<T> y((size_t)n * W);
    CK(cudaMemcpy(y.data(), dy, y.size() * 2, cudaMemcpyDeviceToHost));
    size_t bad = 0, nonfinite = 0;
    double worst = 0;
    for (size_t i = 0; i < y.size(); i++) nonfinite += !isfinite(tf<T>(y[i]));
    if (grade) {
        std::vector<double> s(n);
        for (uint32_t r = 0; r < n; r++)
            for (uint32_t h = 0; h < H; h++) {
                double mx = -1e300;
                for (uint32_t k = 0; k < n; k++) {
                    double dot = 0;
                    for (uint32_t c = 0; c < HD; c++)
                        dot += (double)tf<T>(qkv[(size_t)r * LD + h * HD + c]) *
                               (double)tf<T>(qkv[(size_t)k * LD + W + h * HD + c]);
                    s[k] = dot * scale;
                    mx = fmax(mx, s[k]);
                }
                double z = 0;
                for (uint32_t k = 0; k < n; k++) z += (s[k] = exp(s[k] - mx));
                for (uint32_t c = 0; c < HD; c++) {
                    double ref = 0, mv = 0;
                    for (uint32_t k = 0; k < n; k++) {
                        const double vv = tf<T>(qkv[(size_t)k * LD + 2 * W + h * HD + c]);
                        ref += s[k] / z * vv;
                        mv += fabs(vv) / n;
                    }
                    const double err = fabs((double)tf<T>(y[(size_t)r * W + h * HD + c]) - ref) / (fabs(ref) + mv);
                    worst = fmax(worst, err);
                    bad += !(err <= bound);
                }
            }
    }
    CHECK(rc == 0 && bad == 0 && nonfinite == 0, "%s n=%u: rc %d, %zu outputs past the bound (worst %.2e), %zu non-finite",
          what, n, rc, bad, worst, nonfinite);
    if (grade) printf("  %-6s %s hd %u n=%5u: worst |y - ref| / (|ref| + mean|v|) %.2e (bound %.1e), every output "
                      "written\n", what, bf ? "bf16" : "f16", HD, n, worst, bound);
    else printf("  %-6s %s hd %u n=%5u: %.1f ms (%.1f TFLOPS), every output finite\n", what, bf ? "bf16" : "f16", HD, n, ms,
                4.0 * n * (double)n * HD * H / (ms * 1e-3) / 1e12);
    cudaFree(d);
    cudaFree(dy);
    return 0;
}

int main(void) {
    printf("fmha-gate: bidirectional fused attention over f16 rows (16 heads of 72) and bf16 rows (16 heads of 80), the "
           "production objects\n");
    for (uint32_t n : {1u, 37u, 600u, 2001u})
        if (run<__half>(n, 72, true, "A1") || run<__nv_bfloat16>(n, 80, true, "A1")) return 1;
    if (run<__half>(12298, 72, false, "A2") || run<__nv_bfloat16>(13322, 80, false, "A2")) return 1;
    uint16_t *b = nullptr;
    CK(cudaMalloc(&b, 1 << 20));
    CHECK(pulsar_fmha_f16_launch(b, 1024, b, 1024, b, 1024, 4, 8, 76, 0.1f, b, 1024, 0) == -1, "A3 a head dim of 76 accepted");
    CHECK(pulsar_fmha_f16_launch(b, 2048, b, 2048, b, 2048, 4, 8, 136, 0.1f, b, 2048, 0) == -1, "A3 a head dim of 136 accepted");
    CHECK(pulsar_fmha_f16_launch(b + 1, 1024, b, 1024, b, 1024, 4, 8, 64, 0.1f, b, 1024, 0) == -1, "A3 a misaligned q accepted");
    CHECK(pulsar_fmha_f16_launch(b, 256, b, 1024, b, 1024, 4, 8, 64, 0.1f, b, 1024, 0) == -1, "A3 a short ld accepted");
    CHECK(pulsar_fmha_bf16_launch(b, 1024, b, 1024, b, 1024, 4, 8, 76, 0.1f, b, 1024, 0) == -1, "A3 bf16: a head dim of 76 accepted");
    printf("  refusals: head dim 76 / 136, a misaligned pointer, a short leading dimension (bf16: head dim 76)\n");
    cudaFree(b);
    printf(g_fail ? "FMHA GATE FAIL\n" : "FMHA GATE PASS\n");
    return g_fail;
}

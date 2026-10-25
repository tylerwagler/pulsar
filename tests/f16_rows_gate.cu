/* F16 ROWS GATE (L321): the f16 rows arm (src/cuda/pulsar_cuda_f16_rows.cu over pulsar_cuda_lt_fixed.cu) and the
 * row ops it brings (LayerNorm, the biased residual add, the biased GELU emit; pulsar_cuda_rowops.cu) -- the production
 * objects, model-free, at Nemotron Parse's shapes.
 *
 *   L1  the linear vs a double host reference: |y - ref| <= 8 sqrt(K) 2^-24 * sum_k |x_k w_k| per output (the f32
 *       accumulation of exact products: the operand is the bf16 row scaled by a power of two), every output
 *       written, with an outlier column 2^10 above the rest (the prescale's reason);
 *   L2  row independence, bit for bit: each prompt row alone == the same row in a call of all of them, and each
 *       decode row alone == batched (decode rows ride blocks of 16: 37 rows is three blocks);
 *   L3  the refusals (a width not a multiple of 8, a short workspace, a misaligned weight);
 *   R1  layernorm vs a double host LayerNorm: within one bf16 ulp of the reference value (rsqrtf, the f32 sums);
 *   R2  bias add: res + (x + bias) rounded once (bit-exact to the host), then the LayerNorm of the stored row ==
 *       R1's on that row, bit for bit;
 *   R3  bias emit + GELU vs a double host GELU (erf): within one bf16 ulp plus the f32 cancellation of 1 + erf in the
 *       negative tail (torch's CUDA GELU computes the same expression in f32: 2^-22 |v|); no activation: bit-exact.
 *
 * usage: ./tests/f16_rows_gate */
#include "../src/cuda/pulsar_cuda_f16_rows.h"
#include "../src/cuda/pulsar_cuda_rowops.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "F16 ROWS FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); g_fail = 1; } } while (0)
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", \
                  cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

static uint64_t g_rng = 0x2545f4914f6cdd1dull;
static uint32_t rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 16); }
static double rndn(void) {
    const double u1 = ((double)rnd() + 0.5) / 4294967296.0, u2 = ((double)rnd() + 0.5) / 4294967296.0;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}
static uint16_t bf16_bits(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static float bf16_val(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
/* one bf16 ulp at |v| */
static double bf16_ulp(double v) {
    int e;
    frexp(fabs(v) > 1e-30 ? v : 1e-30, &e);
    return ldexp(1.0, e - 8);
}

template <class T>
static T *dev(const std::vector<T> &h) {
    T *d = nullptr;
    if (cudaMalloc(&d, h.size() * sizeof(T) + 16) != cudaSuccess) return nullptr;
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T>
static std::vector<T> host(const T *d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

static int linear_case(int in, int out, int rows) {
    std::vector<__half> w((size_t)out * in);
    std::vector<float> wf((size_t)out * in);
    for (size_t i = 0; i < w.size(); i++) {
        w[i] = __float2half((float)(rndn() * 0.05));
        wf[i] = __half2float(w[i]);
    }
    std::vector<uint16_t> x((size_t)rows * in);
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < in; k++) x[(size_t)r * in + k] = bf16_bits((float)(rndn() * (k == 7 ? 1024.0 : 1.0)));
    __half *dw = dev(w);
    uint16_t *dx = dev(x);
    float *dy = nullptr, *dy1 = nullptr;
    CK(cudaMalloc(&dy, (size_t)rows * out * 4));
    CK(cudaMalloc(&dy1, (size_t)out * 4));
    const size_t wsb = pulsar_f16_rows_workspace_bytes(rows, in);
    void *ws = nullptr;
    CK(cudaMalloc(&ws, wsb));
    for (bool prompt : {true, false}) {
        CK(cudaMemset(dy, 0xff, (size_t)rows * out * 4));
        const int rc = pulsar_f16_rows_linear_launch(dw, dx, rows, in, out, dy, prompt, ws, wsb, 0);
        CK(cudaDeviceSynchronize());
        const std::vector<float> y = host(dy, (size_t)rows * out);
        double worst = 0;
        size_t bad = 0;
        const double tol = 8.0 * sqrt((double)in) * ldexp(1.0, -24);
        for (int r = 0; r < rows; r++)
            for (int n = 0; n < out; n++) {
                double ref = 0, mag = 0;
                for (int k = 0; k < in; k++) {
                    const double p = (double)bf16_val(x[(size_t)r * in + k]) * wf[(size_t)n * in + k];
                    ref += p;
                    mag += fabs(p);
                }
                const double err = fabs((double)y[(size_t)r * out + n] - ref) / (mag > 0 ? mag : 1);
                worst = fmax(worst, err);
                bad += !(err <= tol);
            }
        CHECK(rc == 0 && bad == 0, "L1 %s %d -> %d x %d rows: rc %d, %zu outputs past %.1e x sum|xw| (worst %.2e)",
              prompt ? "prompt" : "decode", in, out, rows, rc, bad, tol, worst);
        bool same = true;
        for (int r = 0; r < rows && same; r++) {
            if (pulsar_f16_rows_linear_launch(dw, dx + (size_t)r * in, 1, in, out, dy1, prompt, ws, wsb, 0) != 0) {
                same = false;
                break;
            }
            CK(cudaDeviceSynchronize());
            const std::vector<float> y1 = host(dy1, (size_t)out);
            same = memcmp(y1.data(), &y[(size_t)r * out], (size_t)out * 4) == 0;
        }
        CHECK(same, "L2 %s %d -> %d: a row alone != the same row in a call of %d", prompt ? "prompt" : "decode", in, out,
              rows);
        printf("  %-6s %5d -> %5d x %3d rows: worst |y - ref| / sum|xw| %.2e (bound %.1e); every row alone == batched\n",
               prompt ? "prompt" : "decode", in, out, rows, worst, tol);
    }
    cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(dy1); cudaFree(ws);
    return 0;
}

static void ln_ref(const float *x, int W, const float *w, const float *b, double eps, double *y) {
    double mean = 0, var = 0;
    for (int c = 0; c < W; c++) mean += x[c];
    mean /= W;
    for (int c = 0; c < W; c++) var += (x[c] - mean) * (x[c] - mean);
    const double rstd = 1.0 / sqrt(var / W + eps);
    for (int c = 0; c < W; c++) y[c] = (x[c] - mean) * rstd * w[c] + b[c];
}

static int rowops_case(int W, int rows) {
    std::vector<uint16_t> res((size_t)rows * W);
    std::vector<float> x((size_t)rows * W), w(W), b(W), bias(W);
    for (auto &v : res) v = bf16_bits((float)(rndn() * 3.0 + 0.5));
    for (auto &v : x) v = (float)(rndn() * 2.0);
    for (int c = 0; c < W; c++) { w[c] = (float)(1.0 + 0.2 * rndn()); b[c] = (float)(0.1 * rndn()); bias[c] = (float)(0.3 * rndn()); }
    uint16_t *dres = dev(res), *dy = nullptr;
    float *dx = dev(x), *dw = dev(w), *db = dev(b), *dbias = dev(bias);
    CK(cudaMalloc(&dy, (size_t)rows * W * 2));
    /* R1 */
    int rc = pulsar_rows_layernorm_launch(dres, W, rows, W, dw, db, 1e-6f, dy, W, 0);
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> y = host(dy, (size_t)rows * W);
    size_t bad = 0;
    std::vector<double> ref(W);
    std::vector<float> rowf(W);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < W; c++) rowf[c] = bf16_val(res[(size_t)r * W + c]);
        ln_ref(rowf.data(), W, w.data(), b.data(), 1e-6, ref.data());
        for (int c = 0; c < W; c++) bad += !(fabs(bf16_val(y[(size_t)r * W + c]) - ref[c]) <= bf16_ulp(ref[c]));
    }
    CHECK(rc == 0 && bad == 0, "R1 layernorm width %d: rc %d, %zu values past one bf16 ulp", W, rc, bad);
    /* R2 */
    rc = pulsar_rows_bias_add_launch(dres, W, dx, W, dbias, rows, W, dw, db, 1e-6f, dy, W, 0);
    CK(cudaDeviceSynchronize());
    const std::vector<uint16_t> res2 = host(dres, (size_t)rows * W), y2 = host(dy, (size_t)rows * W);
    size_t bad_add = 0;
    for (size_t i = 0; i < res.size(); i++) {
        const float d = x[i] + bias[i % W];
        bad_add += res2[i] != bf16_bits(bf16_val(res[i]) + d);
    }
    uint16_t *dy3 = nullptr;
    CK(cudaMalloc(&dy3, (size_t)rows * W * 2));
    rc |= pulsar_rows_layernorm_launch(dres, W, rows, W, dw, db, 1e-6f, dy3, W, 0);
    CK(cudaDeviceSynchronize());
    const std::vector<uint16_t> y3 = host(dy3, (size_t)rows * W);
    CHECK(rc == 0 && bad_add == 0 && y2 == y3, "R2 bias add width %d: rc %d, %zu sums differ from one rounding, fused "
          "norm %s the norm alone", W, rc, bad_add, y2 == y3 ? "==" : "!=");
    /* R3 */
    for (pulsar_rows_act act : {PULSAR_ROWS_ACT_NONE, PULSAR_ROWS_ACT_GELU}) {
        rc = pulsar_rows_bias_emit_launch(dx, W, rows, W, dbias, act, dy, W, 0);
        CK(cudaDeviceSynchronize());
        y = host(dy, (size_t)rows * W);
        size_t b3 = 0;
        for (size_t i = 0; i < x.size(); i++) {
            const float z = x[i] + bias[i % W];
            if (act == PULSAR_ROWS_ACT_NONE) b3 += y[i] != bf16_bits(z);
            else {
                const double g = 0.5 * z * (1.0 + erf(z / sqrt(2.0)));
                b3 += !(fabs(bf16_val(y[i]) - g) <= bf16_ulp(g) + ldexp(fabs((double)z), -22));
            }
        }
        CHECK(rc == 0 && b3 == 0, "R3 bias emit (%s) width %d: rc %d, %zu values off", act ? "gelu" : "none", W, rc, b3);
    }
    printf("  rows width %4d x %d: layernorm within a bf16 ulp, bias add one rounding + fused norm == norm, bias emit "
           "(none bit-exact, gelu within a ulp)\n", W, rows);
    cudaFree(dres); cudaFree(dy); cudaFree(dy3); cudaFree(dx); cudaFree(dw); cudaFree(db); cudaFree(dbias);
    return 0;
}

int main(void) {
    printf("f16-rows-gate: the f16 rows arm + the LayerNorm / bias / GELU row ops (production objects)\n");
    const int shapes[][2] = {{1152, 3456}, {4304, 1152}, {2304, 1024}, {1024, 512}, {4096, 1024}};
    for (auto &s : shapes)
        if (linear_case(s[0], s[1], 37)) return 1;
    /* L3 */
    void *ws = nullptr;
    CK(cudaMalloc(&ws, 1 << 20));
    __half *w = nullptr;
    uint16_t *x = nullptr;
    float *y = nullptr;
    CK(cudaMalloc(&w, 64 * 64 * 2)); CK(cudaMalloc(&x, 64 * 64 * 2)); CK(cudaMalloc(&y, 64 * 64 * 4));
    CHECK(pulsar_f16_rows_linear_launch(w, x, 2, 60, 64, y, true, ws, 1 << 20, 0) == -1, "L3 an input width of 60 accepted");
    CHECK(pulsar_f16_rows_linear_launch(w, x, 2, 64, 64, y, true, ws, 16, 0) == -1, "L3 a 16-byte workspace accepted");
    CHECK(pulsar_f16_rows_linear_launch(w + 1, x, 2, 64, 64, y, true, ws, 1 << 20, 0) == -1, "L3 a misaligned weight accepted");
    cudaFree(ws); cudaFree(w); cudaFree(x); cudaFree(y);
    printf("  refusals: a width not a multiple of 8, a short workspace, a misaligned weight\n");
    for (int W : {1152, 1024})
        if (rowops_case(W, 300)) return 1;
    printf(g_fail ? "F16 ROWS GATE FAIL\n" : "F16 ROWS GATE PASS\n");
    return g_fail;
}

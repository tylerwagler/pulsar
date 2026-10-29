/* F16 ROWS GATE (L321): the f16 rows arm (src/cuda/pulsar_cuda_f16_rows.cu over pulsar_cuda_lt_fixed.cu) and the
 * row ops it brings (the f32-residual LayerNorm and add, the biased GELU emit, the f16 gather; pulsar_cuda_rowops.cu)
 * -- the production objects, model-free, at Nemotron Parse's shapes.
 *
 *   L1  the linear vs a double host reference: |y - ref| <= 8 sqrt(K) 2^-24 * sum_k |x_k w_k| per output (the f32
 *       accumulation of exact products: the operand is the bf16 row scaled by a power of two), every output
 *       written, with an outlier column 2^10 above the rest (the prescale's reason);
 *   L2  row independence, bit for bit: each prompt row alone == the same row in a call of all of them, and each
 *       decode row alone == batched (decode rows ride blocks of 16: 37 rows is three blocks);
 *   L3  the refusals (a width not a multiple of 8, a short workspace, a misaligned weight);
 *   L4  f16 operand rows (pulsar_f16_rows_linear_h_launch): == the bf16 entry on the same values (both are exact
 *       operands of the one GEMM: an f16 row whose values are bf16 values gives the bf16 row's bytes), bit for bit;
 *   R1  layernorm32 vs a double host LayerNorm, from bf16 rows and from f32 rows + a bias: f32 out within 2^-20
 *       relative, bf16 / f16 out within one ulp of their format plus 2^-18 absolute (rsqrtf, the f32 sums);
 *   R2  res32_add: res + (x + bias) in f32 (bit-exact to the host), then its fused LayerNorm == layernorm32 of the
 *       stored row, bit for bit, in both 16-bit formats;
 *   R3  bias emit + GELU vs a double host GELU (erf): within one ulp of the output format plus the f32 cancellation of
 *       1 + erf in the negative tail (torch's CUDA GELU computes the same expression in f32: 2^-22 |v|); no
 *       activation: bit-exact;
 *   R4  the f16 gather: table row * scale as f32, bit-exact (an exact product), a NaN row for an id outside.
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

static void ln_ref(const double *x, int W, const float *w, const float *b, double eps, double *y) {
    double mean = 0, var = 0;
    for (int c = 0; c < W; c++) mean += x[c];
    mean /= W;
    for (int c = 0; c < W; c++) var += (x[c] - mean) * (x[c] - mean);
    const double rstd = 1.0 / sqrt(var / W + eps);
    for (int c = 0; c < W; c++) y[c] = (x[c] - mean) * rstd * w[c] + b[c];
}

static float f16_val(uint16_t h) { return __half2float(__ushort_as_half(h)); }
static double f16_ulp(double v) {
    int e;
    frexp(fabs(v) > 6.2e-5 ? v : 6.2e-5, &e);
    return ldexp(1.0, e - 11);
}

/* the 16-bit value in `fmt` and its ulp */
static double out_val(uint16_t u, pulsar_rows_out fmt) { return fmt == PULSAR_ROWS_OUT_F16 ? f16_val(u) : bf16_val(u); }
static double out_ulp(double v, pulsar_rows_out fmt) { return fmt == PULSAR_ROWS_OUT_F16 ? f16_ulp(v) : bf16_ulp(v); }

static int rowops_case(int W, int rows) {
    std::vector<uint16_t> xb((size_t)rows * W);
    std::vector<float> res((size_t)rows * W), x((size_t)rows * W), w(W), b(W), bias(W);
    for (auto &v : xb) v = bf16_bits((float)(rndn() * 3.0 + 0.5));
    for (auto &v : res) v = (float)(rndn() * 30.0);
    for (auto &v : x) v = (float)(rndn() * 2.0);
    for (int c = 0; c < W; c++) { w[c] = (float)(1.0 + 0.2 * rndn()); b[c] = (float)(0.1 * rndn()); bias[c] = (float)(0.3 * rndn()); }
    uint16_t *dxb = dev(xb), *dy = nullptr, *dy2 = nullptr;
    float *dres = dev(res), *dx = dev(x), *dw = dev(w), *db = dev(b), *dbias = dev(bias), *dyf = nullptr;
    CK(cudaMalloc(&dy, (size_t)rows * W * 2)); CK(cudaMalloc(&dy2, (size_t)rows * W * 2));
    CK(cudaMalloc(&dyf, (size_t)rows * W * 4));
    std::vector<double> in(W), ref(W);
    /* R1: from bf16 rows, and from f32 rows + a bias */
    for (int src = 0; src < 2; src++)
        for (pulsar_rows_out fmt : {PULSAR_ROWS_OUT_BF16, PULSAR_ROWS_OUT_F16}) {
            const int rc = src == 0 ? pulsar_rows_layernorm32_launch(dxb, false, W, NULL, rows, W, dw, db, 1e-6f, dy, W, fmt,
                                                                     dyf, W, 0)
                                    : pulsar_rows_layernorm32_launch(dx, true, W, dbias, rows, W, dw, db, 1e-6f, dy, W,
                                                                     fmt, dyf, W, 0);
            CK(cudaDeviceSynchronize());
            const std::vector<uint16_t> y = host(dy, (size_t)rows * W);
            const std::vector<float> yf = host(dyf, (size_t)rows * W);
            size_t bad = 0;
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < W; c++)
                    in[c] = src == 0 ? (double)bf16_val(xb[(size_t)r * W + c]) : (double)x[(size_t)r * W + c] + bias[c];
                ln_ref(in.data(), W, w.data(), b.data(), 1e-6, ref.data());
                for (int c = 0; c < W; c++) {
                    /* one ulp of the format, plus the f32 arithmetic's own error near zero (2^-18 absolute) */
                    bad += !(fabs(out_val(y[(size_t)r * W + c], fmt) - ref[c]) <= out_ulp(ref[c], fmt) + ldexp(1.0, -18));
                    bad += !(fabs(yf[(size_t)r * W + c] - ref[c]) <= ldexp(fabs(ref[c]) + 1.0, -20));
                }
            }
            CHECK(rc == 0 && bad == 0, "R1 layernorm32 width %d from %s to %s: rc %d, %zu values off", W,
                  src ? "f32 + bias" : "bf16", fmt ? "f16" : "bf16", rc, bad);
        }
    /* R2 */
    for (pulsar_rows_out fmt : {PULSAR_ROWS_OUT_BF16, PULSAR_ROWS_OUT_F16}) {
        float *dr = dev(res);
        int rc = pulsar_rows_res32_add_launch(dr, W, dx, W, dbias, rows, W, dw, db, 1e-6f, dy, W, fmt, 0);
        CK(cudaDeviceSynchronize());
        const std::vector<float> r2 = host(dr, (size_t)rows * W);
        size_t bad_add = 0;
        for (size_t i = 0; i < res.size(); i++) bad_add += r2[i] != res[i] + (x[i] + bias[i % W]);
        rc |= pulsar_rows_layernorm32_launch(dr, true, W, NULL, rows, W, dw, db, 1e-6f, dy2, W, fmt, NULL, 0, 0);
        CK(cudaDeviceSynchronize());
        CHECK(rc == 0 && bad_add == 0 && host(dy, (size_t)rows * W) == host(dy2, (size_t)rows * W),
              "R2 res32 add width %d (%s): rc %d, %zu sums differ from f32, the fused norm %s the norm alone", W,
              fmt ? "f16" : "bf16", rc, bad_add, host(dy, (size_t)rows * W) == host(dy2, (size_t)rows * W) ? "==" : "!=");
        cudaFree(dr);
    }
    /* R3 */
    for (pulsar_rows_out fmt : {PULSAR_ROWS_OUT_BF16, PULSAR_ROWS_OUT_F16})
        for (pulsar_rows_act act : {PULSAR_ROWS_ACT_NONE, PULSAR_ROWS_ACT_GELU, PULSAR_ROWS_ACT_RELU2}) {
            const int rc = pulsar_rows_bias_emit_launch(dx, W, rows, W, dbias, act, dy, W, fmt, 0);
            CK(cudaDeviceSynchronize());
            const std::vector<uint16_t> y = host(dy, (size_t)rows * W);
            size_t b3 = 0;
            for (size_t i = 0; i < x.size(); i++) {
                const float z = x[i] + bias[i % W];
                if (act == PULSAR_ROWS_ACT_NONE || act == PULSAR_ROWS_ACT_RELU2) {
                    /* L331 relu^2: max(z, 0), one f32 square, one rounding -- bit-exact like the plain emit */
                    const float q = act == PULSAR_ROWS_ACT_NONE ? z : (z > 0.0f ? z : 0.0f) * (z > 0.0f ? z : 0.0f);
                    b3 += y[i] != (fmt == PULSAR_ROWS_OUT_F16 ? __half_as_ushort(__float2half_rn(q)) : bf16_bits(q));
                } else {
                    const double g = 0.5 * z * (1.0 + erf(z / sqrt(2.0)));
                    b3 += !(fabs(out_val(y[i], fmt) - g) <= out_ulp(g, fmt) + ldexp(fabs((double)z), -22));
                }
            }
            CHECK(rc == 0 && b3 == 0, "R3 bias emit (%s, %s) width %d: rc %d, %zu values off",
                  act == PULSAR_ROWS_ACT_GELU ? "gelu" : act == PULSAR_ROWS_ACT_RELU2 ? "relu2" : "none",
                  fmt ? "f16" : "bf16", W, rc, b3);
        }
    printf("  rows width %4d x %d: layernorm32 (bf16 / f32 + bias in; bf16 / f16 / f32 out) within an ulp, res32 add "
           "exact + fused norm == norm, bias emit (none / relu2 bit-exact, gelu within an ulp)\n", W, rows);
    cudaFree(dxb); cudaFree(dy); cudaFree(dy2); cudaFree(dyf); cudaFree(dres); cudaFree(dx); cudaFree(dw); cudaFree(db);
    cudaFree(dbias);
    return 0;
}

/* L4: f16 operand rows whose values are bf16 values == the bf16 entry on them, bit for bit */
static int f16_operand_case(int in, int out, int rows) {
    std::vector<__half> w((size_t)out * in);
    for (auto &v : w) v = __float2half((float)(rndn() * 0.05));
    std::vector<uint16_t> xb((size_t)rows * in), xh((size_t)rows * in);
    for (size_t i = 0; i < xb.size(); i++) {
        xb[i] = bf16_bits((float)(rndn() * 4.0));
        xh[i] = __half_as_ushort(__float2half_rn(bf16_val(xb[i])));   /* exact: |x| is small and normal */
    }
    __half *dw = dev(w);
    uint16_t *dxb = dev(xb), *dxh = dev(xh);
    float *y1 = nullptr, *y2 = nullptr;
    CK(cudaMalloc(&y1, (size_t)rows * out * 4)); CK(cudaMalloc(&y2, (size_t)rows * out * 4));
    const size_t wsb = pulsar_f16_rows_workspace_bytes(rows, in);
    void *ws = nullptr;
    CK(cudaMalloc(&ws, wsb));
    bool same = true;
    for (bool prompt : {true, false}) {
        const int a = pulsar_f16_rows_linear_launch(dw, dxb, rows, in, out, y1, prompt, ws, wsb, 0);
        const int b = pulsar_f16_rows_linear_h_launch(dw, dxh, rows, in, out, y2, prompt, 0);
        CK(cudaDeviceSynchronize());
        same = same && a == 0 && b == 0 && host(y1, (size_t)rows * out) == host(y2, (size_t)rows * out);
    }
    CHECK(same, "L4 %d -> %d: the f16 operand entry != the bf16 entry on the same values", in, out);
    printf("  f16 operands %4d -> %4d x %d rows: == the bf16 entry on the same values (prompt and decode)\n", in, out,
           rows);
    cudaFree(dw); cudaFree(dxb); cudaFree(dxh); cudaFree(y1); cudaFree(y2); cudaFree(ws);
    return 0;
}

/* R4 */
static int gather_case(void) {
    const uint32_t V = 300, Wd = 1024, n = 5;
    std::vector<__half> t((size_t)V * Wd);
    for (auto &v : t) v = __float2half((float)(rndn() * 0.1));
    const std::vector<int32_t> ids = {7, 0, 299, 300, 42};
    __half *dt = dev(t);
    int32_t *di = dev(ids);
    float *dy = nullptr;
    CK(cudaMalloc(&dy, (size_t)n * Wd * 4));
    const int rc = pulsar_rows_gather_f16_launch((const uint16_t *)dt, V, Wd, di, n, 32.0f, dy, Wd, 0);
    CK(cudaDeviceSynchronize());
    const std::vector<float> y = host(dy, (size_t)n * Wd);
    size_t bad = 0;
    for (uint32_t r = 0; r < n; r++)
        for (uint32_t c = 0; c < Wd; c++) {
            const float g = y[(size_t)r * Wd + c];
            bad += ids[r] < (int32_t)V ? g != __half2float(t[(size_t)ids[r] * Wd + c]) * 32.0f : !isnan(g);
        }
    CHECK(rc == 0 && bad == 0, "R4 gather: rc %d, %zu values off", rc, bad);
    printf("  gather f16 x 32 -> f32: exact, an id past the table a NaN row\n");
    cudaFree(dt); cudaFree(di); cudaFree(dy);
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
    for (auto &s : shapes)
        if (f16_operand_case(s[0], s[1], 37)) return 1;
    for (int W : {1152, 1024})
        if (rowops_case(W, 300)) return 1;
    if (gather_case()) return 1;
    printf(g_fail ? "F16 ROWS GATE FAIL\n" : "F16 ROWS GATE PASS\n");
    return g_fail;
}

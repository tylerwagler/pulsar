/* NVFP4 ROWS GATE (L298): NVFP4 W4A16 weights against raw bf16 rows (src/cuda/pulsar_cuda_nvfp4.h), through the
 * production objects and the rows front door.
 *
 * Model-free.  Random e2m1 codes (every code, both signs), random E4M3 scales over the format's range (subnormals,
 * zeros and the 448 top included; the scale plane 128x4-tiled), a random global scale; random bf16 rows.
 *
 *   1. graded: y against double arithmetic on the weight dequantized on the host (e2m1 x e4m3 x global): every output
 *      within 2^-18 of its own sum |w x| (the f32 accumulation's envelope -- products are exact, so that is the only
 *      error), and the bf16 output's relative error against the double result beside the bf16 rounding floor of that
 *      result.  Shapes: every K split (1, 2, 4, 8), a span ending inside a super-step, output widths off 128 and off
 *      16, the LiLiCorr drafter's shapes (the MLP and head ones; the wide head at a reduced row count).
 *   2. bytes: the bf16 output == the f32 output rounded to nearest even; a repeated call is bit-identical
 *      (deterministic: no atomics, a fixed summation order).
 *   3. M-independence: M = 1, 3, 8, 9 and 20 (16-row blocks) -- each row the bytes of the same row at M = 16.
 *   4. the front door (pulsar_rows_linear_launch with an nvfp4 reference) == the arm; refusals (in % 64, no output,
 *      misaligned weight, an nvfp4 reference carrying an EXL3 rate).
 *   5. the multi launch (a gate and an up of one input, one launch; pulsar_nvfp4_rows_multi_launch) and the front
 *      door's multi (pulsar_rows_linear_multi_bf16_launch / _multi_launch grouping them) == each Linear alone, bit for
 *      bit.
 *   6. timing (information; `--bench` only): the drafter's shapes at M = 1, 4, 8 with the L2 flushed before every call
 *      (weights read cold), GB/s of the nvfp4 bytes, against the BF16 rows kernel on a bf16 weight of the shape; and
 *      the drafter's gate + up as one multi launch.
 *
 * usage: ./tests/nvfp4_rows_gate [--bench]
 */
#include "cuda/pulsar_cuda_bf16_rows.cuh"
#include "cuda/pulsar_cuda_nvfp4.h"
#include "cuda/pulsar_cuda_qwen.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "NVFP4-ROWS FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); \
                                       g_fail = 1; } } while (0)
#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "NVFP4-ROWS FAIL: %s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static uint64_t g_rng = 0x4E5F4F5F4E5F4F5FULL;
static uint32_t rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 16); }
static double rndu(void) { return ((double)rnd() + 0.5) / 4294967296.0; }

static uint16_t bf16_bits(float v) {
    const __nv_bfloat16 b = __float2bfloat16(v);
    uint16_t u;
    memcpy(&u, &b, 2);
    return u;
}
static float bf16_val(uint16_t u) {
    const uint32_t f = (uint32_t)u << 16;
    float x;
    memcpy(&x, &f, 4);
    return x;
}
static float e4m3_val(uint8_t b) {
    __nv_fp8_e4m3 v;
    memcpy(&v, &b, 1);
    return (float)v;
}
static const float k_e2m1[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6};

template <class T> static T *upload(const std::vector<T> &v) {
    T *d = nullptr;
    CUDA_OK(cudaMalloc((void **)&d, v.size() * sizeof(T) + 16));
    CUDA_OK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> static std::vector<T> download(const T *d, size_t n) {
    std::vector<T> v(n);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

/* One synthetic nvfp4 Linear: the stored slice and its exact bf16 dequantization (no global scale). */
struct fix {
    int in, out;
    float gs;
    std::vector<uint8_t> blob;     /* values | 128x4 scales | global */
    std::vector<uint16_t> deq;     /* [out][in] bf16 = e2m1 * e4m3 */
};
static fix make_fix(int in, int out, bool deq) {
    fix f;
    f.in = in; f.out = out;
    uint64_t vb, sb, st;
    if (!pulsar_nvfp4_layout(in, out, &vb, &sb, &st)) { fprintf(stderr, "bad fixture shape\n"); exit(1); }
    f.blob.assign(st, 0);
    for (uint64_t i = 0; i < vb; i++) f.blob[i] = (uint8_t)rnd();
    const int kb4 = in / 64;
    std::vector<uint8_t> sc((size_t)out * (in / 16));
    for (auto &s : sc) {
        /* the exponent field 0..15, mantissa 0..7, never NaN (0x7f); a zero now and then (ModelOpt's scales are
         * non-negative) */
        uint8_t b = (uint8_t)(rnd() % 127);
        if (rnd() % 97 == 0) b = 0;
        s = b;
    }
    for (int r = 0; r < out; r++)
        for (int kb = 0; kb < in / 16; kb++)
            f.blob[vb + ((size_t)(r / 128) * kb4 + kb / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + kb % 4] =
                sc[(size_t)r * (in / 16) + kb];
    f.gs = (float)(ldexp(1.0 + rndu(), -10));
    memcpy(&f.blob[vb + sb], &f.gs, 4);
    if (deq) {
        f.deq.resize((size_t)out * in);
        for (int r = 0; r < out; r++)
            for (int k = 0; k < in; k++) {
                const uint8_t byte = f.blob[(size_t)r * (in / 2) + k / 2];
                const float v = k_e2m1[k & 1 ? byte >> 4 : byte & 15] * e4m3_val(sc[(size_t)r * (in / 16) + k / 16]);
                const uint16_t b = bf16_bits(v);
                if (bf16_val(b) != v) { fprintf(stderr, "e2m1 x e4m3 not exact in bf16?\n"); exit(1); }
                f.deq[(size_t)r * in + k] = b;
            }
    }
    return f;
}

static std::vector<uint16_t> rnd_rows(int M, int in) {
    std::vector<uint16_t> v((size_t)M * in);
    for (auto &x : v) x = bf16_bits((float)((rndu() + rndu() + rndu() + rndu() - 2.0) * 1.7));
    return v;
}

static void section_exact(void) {
    struct shape { int in, out; const char *what; };
    const shape shapes[] = {
        {4096, 12288, "4096 -> 12288 (gate / up)"}, {12288, 4096, "12288 -> 4096 (down)"},
        {2048, 1024, "2048 -> 1024 (head proj)"},   {1024, 2048, "1024 -> 2048 (head up)"},
        {1024, 1024, "1024 -> 1024 (head down)"},   {512, 1000, "out off 16"},
        {640, 200, "out off 128, in off 128"},      {4096, 40960, "wide, K unsplit"},
        {20480, 4096, "20480 -> 4096 (fc)"}};
    const int MR = 20;
    for (const shape &sh : shapes) {
        const fix f = make_fix(sh.in, sh.out, true);
        const std::vector<uint16_t> X = rnd_rows(MR, f.in);
        uint8_t *dW = upload(f.blob);
        uint16_t *dX = upload(X), *dYb = nullptr;
        float *dY = nullptr;
        CUDA_OK(cudaMalloc((void **)&dY, (size_t)MR * f.out * 4));
        CUDA_OK(cudaMalloc((void **)&dYb, (size_t)MR * f.out * 2));
        auto run = [&](int M, int r0, std::vector<uint16_t> *yb) {
            CUDA_OK(cudaMemset(dY, 0xff, (size_t)M * f.out * 4));
            CUDA_OK(cudaMemset(dYb, 0xff, (size_t)M * f.out * 2));
            const int rc = pulsar_nvfp4_rows_linear_launch(dW, dX + (size_t)r0 * f.in, M, f.in, f.out, dY, dYb, 0);
            CHECK(rc == 0, "%s: M=%d refused (rc=%d)", sh.what, M, rc);
            if (yb) *yb = download(dYb, (size_t)M * f.out);
            return download(dY, (size_t)M * f.out);
        };
        std::vector<uint16_t> yb16;
        const std::vector<float> y16 = run(16, 0, &yb16);
        /* 2. bytes: bf16 = the f32 rounded; a repeat is the same bytes */
        size_t badb = 0;
        for (size_t i = 0; i < y16.size(); i++) badb += yb16[i] != bf16_bits(y16[i]);
        const std::vector<float> again = run(16, 0, nullptr);
        const bool repeat = memcmp(again.data(), y16.data(), y16.size() * 4) == 0;
        CHECK(badb == 0, "%s: %zu bf16 outputs are not the f32 output rounded", sh.what, badb);
        CHECK(repeat, "%s: a repeated call differs (not deterministic)", sh.what);
        /* 1. graded against double */
        double worst = 0, e2 = 0, f2 = 0, r2 = 0;
        size_t over = 0;
        for (int t = 0; t < 16; t++)
            for (int o = 0; o < f.out; o++) {
                double s = 0, a = 0;
                for (int k = 0; k < f.in; k++) {
                    const double p = (double)bf16_val(f.deq[(size_t)o * f.in + k]) * bf16_val(X[(size_t)t * f.in + k]);
                    s += p;
                    a += fabs(p);
                }
                s *= f.gs;
                a *= f.gs;
                const double e = fabs((double)y16[(size_t)t * f.out + o] - s);
                over += !(e <= ldexp(a, -18) + 1e-30);
                worst = fmax(worst, e / fmax(a, 1e-30));
                const double b = bf16_val(yb16[(size_t)t * f.out + o]), bfloor = bf16_val(bf16_bits((float)s));
                e2 += (b - s) * (b - s);
                f2 += (bfloor - s) * (bfloor - s);
                r2 += s * s;
            }
        CHECK(over == 0, "%s: %zu outputs past the f32 envelope (worst err / sum|wx| %.2e)", sh.what, over, worst);
        /* 3. M-independence */
        size_t diff = 0;
        for (int M : {1, 3, 8, 9}) {
            const std::vector<float> y = run(M, 0, nullptr);
            diff += memcmp(y.data(), y16.data(), y.size() * 4) != 0;
        }
        const std::vector<float> y20 = run(MR, 0, nullptr);
        diff += memcmp(y20.data(), y16.data(), y16.size() * 4) != 0;
        for (int r : {16, 19}) {
            const std::vector<float> y1 = run(1, r, nullptr);
            diff += memcmp(y1.data(), y20.data() + (size_t)r * f.out, (size_t)f.out * 4) != 0;
        }
        CHECK(diff == 0, "%s: %zu calls (M = 1, 3, 8, 9, 20 and rows 16, 19 alone) differ from the same rows of M = 16",
              sh.what, diff);
        printf("  %-28s K split %d: worst err/sum|wx| %.2e; bf16 out rel err %.3e (bf16 floor %.3e); bf16 == f32 "
               "rounded %s; repeat %s; M = 1..20 %s\n", sh.what, pulsar_nvfp4_ksplit(f.in, f.out), worst,
               sqrt(e2 / r2), sqrt(f2 / r2), badb ? "NO" : "yes", repeat ? "bit-identical" : "DIFFERS",
               diff ? "DIFFER" : "bit-identical");
        /* 4. the front door */
        if (sh.out == 2048) {
            pulsar_rows_linear l{};
            l.w = dW; l.in = f.in; l.out = f.out; l.nvfp4 = true;
            CUDA_OK(cudaMemset(dY, 0xff, (size_t)16 * f.out * 4));
            CHECK(pulsar_rows_linear_launch(&l, dX, 16, dY, nullptr, 0, 0) == 0, "%s: the front door refused", sh.what);
            const std::vector<float> yf = download(dY, (size_t)16 * f.out);
            CHECK(memcmp(yf.data(), y16.data(), yf.size() * 4) == 0, "%s: the front door != the arm", sh.what);
            l.k2 = 8;
            CHECK(pulsar_rows_linear_launch(&l, dX, 16, dY, nullptr, 0, 0) == -1,
                  "%s: an nvfp4 reference with an EXL3 rate accepted", sh.what);
            printf("  front door (nvfp4 reference) == the arm; one carrying an EXL3 rate refused\n");
        }
        cudaFree(dW); cudaFree(dX); cudaFree(dY); cudaFree(dYb);
    }
    float *y = nullptr;
    CUDA_OK(cudaMalloc((void **)&y, 4096));
    CHECK(pulsar_nvfp4_rows_linear_launch((void *)y, (const uint16_t *)y, 1, 1024 + 32, 16, y, nullptr, 0) == -1,
          "in %% 64 != 0 accepted");
    CHECK(pulsar_nvfp4_rows_linear_launch((void *)y, (const uint16_t *)y, 1, 1024, 16, nullptr, nullptr, 0) == -1,
          "no output accepted");
    CHECK(pulsar_nvfp4_rows_linear_launch((const uint8_t *)y + 4, (const uint16_t *)y, 1, 1024, 16, y, nullptr, 0) == -1,
          "a misaligned weight accepted");
    cudaFree(y);
    printf("  refusals: in %% 64, no output, a misaligned weight -- refused\n");
}

static void section_multi(void) {
    const int in = 1024, out = 2048, M = 8;
    const fix a = make_fix(in, out, false), b = make_fix(in, out, false);
    const std::vector<uint16_t> X = rnd_rows(M, in);
    uint8_t *dA = upload(a.blob), *dB = upload(b.blob);
    uint16_t *dX = upload(X), *h[2] = {nullptr, nullptr};
    float *f[2] = {nullptr, nullptr};
    for (int j = 0; j < 2; j++) {
        CUDA_OK(cudaMalloc((void **)&f[j], (size_t)M * out * 4));
        CUDA_OK(cudaMalloc((void **)&h[j], (size_t)M * out * 2));
    }
    auto alone = [&](const uint8_t *w) {
        CHECK(pulsar_nvfp4_rows_linear_launch(w, dX, M, in, out, f[0], h[0], 0) == 0, "multi: a Linear alone refused");
        return std::make_pair(download(f[0], (size_t)M * out), download(h[0], (size_t)M * out));
    };
    const auto ra = alone(dA), rb = alone(dB);
    auto same = [&](int j, const std::pair<std::vector<float>, std::vector<uint16_t>> &r, bool f32, bool b16) {
        bool ok = true;
        if (f32) ok &= memcmp(download(f[j], (size_t)M * out).data(), r.first.data(), (size_t)M * out * 4) == 0;
        if (b16) ok &= memcmp(download(h[j], (size_t)M * out).data(), r.second.data(), (size_t)M * out * 2) == 0;
        return ok;
    };
    const void *w[2] = {dA, dB};
    for (int j = 0; j < 2; j++) {
        CUDA_OK(cudaMemset(f[j], 0xff, (size_t)M * out * 4));
        CUDA_OK(cudaMemset(h[j], 0xff, (size_t)M * out * 2));
    }
    CHECK(pulsar_nvfp4_rows_multi_launch(w, 2, dX, M, in, out, f, h, 0) == 0, "multi launch refused");
    CHECK(same(0, ra, true, true) && same(1, rb, true, true), "the multi launch != each Linear alone");
    /* the front door's multi: f32 outputs, then bf16 outputs (the kernel's epilogue) */
    pulsar_rows_linear la{}, lb{};
    la.w = dA; lb.w = dB;
    la.in = lb.in = in; la.out = lb.out = out; la.nvfp4 = lb.nvfp4 = true;
    const pulsar_rows_linear *ls[2] = {&la, &lb};
    CUDA_OK(cudaMemset(f[0], 0xff, (size_t)M * out * 4));
    CUDA_OK(cudaMemset(f[1], 0xff, (size_t)M * out * 4));
    CHECK(pulsar_rows_linear_multi_launch(ls, 2, dX, M, f, nullptr, 0, 0) == 0, "front door multi (f32) refused");
    CHECK(same(0, ra, true, false) && same(1, rb, true, false), "front door multi (f32) != each Linear alone");
    CUDA_OK(cudaMemset(h[0], 0xff, (size_t)M * out * 2));
    CUDA_OK(cudaMemset(h[1], 0xff, (size_t)M * out * 2));
    CHECK(pulsar_rows_linear_multi_bf16_launch(ls, 2, dX, M, f, h, nullptr, 0, 0) == 0, "front door multi (bf16) refused");
    CHECK(same(0, ra, false, true) && same(1, rb, false, true), "front door multi (bf16) != each Linear alone");
    CHECK(pulsar_nvfp4_rows_multi_launch(w, PULSAR_NVFP4_MULTI_MAX + 1, dX, M, in, out, f, h, 0) == -1,
          "a multi launch past PULSAR_NVFP4_MULTI_MAX accepted");
    printf("  multi: gate + up in one launch, and the front door's multi (f32 and bf16), == each Linear alone; "
           "%d + 1 Linears refused\n", PULSAR_NVFP4_MULTI_MAX);
    for (int j = 0; j < 2; j++) { cudaFree(f[j]); cudaFree(h[j]); }
    cudaFree(dA); cudaFree(dB); cudaFree(dX);
}

static void section_bench(void) {
    struct shape { int in, out; const char *what; };
    const shape shapes[] = {{4096, 131072, "lm_head"},  {20480, 4096, "fc"},          {4096, 12288, "mlp gate/up"},
                            {12288, 4096, "mlp down"},  {2048, 1024, "head ctx/token"}, {1024, 2048, "head up"},
                            {1024, 1024, "head down"}};
    void *flush = nullptr;
    const size_t flush_b = 256u << 20;
    CUDA_OK(cudaMalloc(&flush, flush_b));
    cudaEvent_t e0, e1;
    CUDA_OK(cudaEventCreate(&e0));
    CUDA_OK(cudaEventCreate(&e1));
    printf("  timing: L2 flushed before every call (weights cold); GB/s of the stored bytes; median of 30\n");
    printf("  %-16s %6s %7s  %3s  %10s %8s   %10s %8s   %6s\n", "shape", "in", "out", "M", "nvfp4 us", "GB/s",
           "bf16 us", "GB/s", "speedup");
    for (const shape &sh : shapes) {
        const fix f = make_fix(sh.in, sh.out, false);
        uint8_t *dW = upload(f.blob);
        uint16_t *dB = nullptr;   /* a bf16 weight of the same shape (contents do not matter for time) */
        CUDA_OK(cudaMalloc((void **)&dB, (size_t)f.out * f.in * 2));
        CUDA_OK(cudaMemset(dB, 0x3c, (size_t)f.out * f.in * 2));
        const std::vector<uint16_t> X = rnd_rows(8, f.in);
        uint16_t *dX = upload(X);
        float *dY = nullptr;
        CUDA_OK(cudaMalloc((void **)&dY, (size_t)8 * f.out * 4));
        for (int M : {1, 4, 8}) {
            auto med = [&](auto &&call) {
                std::vector<float> t;
                for (int it = 0; it < 33; it++) {
                    CUDA_OK(cudaMemsetAsync(flush, it, flush_b, 0));
                    CUDA_OK(cudaEventRecord(e0));
                    call();
                    CUDA_OK(cudaEventRecord(e1));
                    CUDA_OK(cudaEventSynchronize(e1));
                    float ms = 0;
                    CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
                    if (it >= 3) t.push_back(ms * 1000.0f);
                }
                std::sort(t.begin(), t.end());
                return (double)t[t.size() / 2];
            };
            const double tn = med([&] { pulsar_nvfp4_rows_linear_launch(dW, dX, M, f.in, f.out, dY, nullptr, 0); });
            const double tb = med([&] { pulsar_bf16_rows_linear_launch(dB, dX, M, f.in, f.out, dY, false, 0); });
            printf("  %-16s %6d %7d  %3d  %10.1f %8.1f   %10.1f %8.1f   %5.2fx\n", sh.what, f.in, f.out, M, tn,
                   (double)f.blob.size() / tn / 1e3, tb, (double)f.out * f.in * 2 / tb / 1e3, tb / tn);
            if (f.out == 12288) {   /* the drafter's gate + up: one multi launch (the same weight twice: cold anyway) */
                uint8_t *dW2 = upload(f.blob);
                const void *w2[2] = {dW, dW2};
                float *ys[2] = {dY, dY};
                const double tm = med([&] { pulsar_nvfp4_rows_multi_launch(w2, 2, dX, M, f.in, f.out, ys, nullptr, 0); });
                printf("  %-16s %6d %7d  %3d  %10.1f %8.1f   (gate + up, one launch; two launches: %.1f us)\n",
                       "mlp gate+up", f.in, f.out, M, tm, 2.0 * f.blob.size() / tm / 1e3, 2 * tn);
                cudaFree(dW2);
            }
        }
        cudaFree(dW); cudaFree(dB); cudaFree(dX); cudaFree(dY);
    }
    cudaFree(flush);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

int main(int argc, char **argv) {
    const bool bench = argc > 1 && !strcmp(argv[1], "--bench");
    printf("NVFP4 rows gate (L298): NVFP4 W4A16 x raw bf16 rows\n");
    if (!bench) {
        section_exact();
        section_multi();
    }
    else section_bench();
    if (g_fail) {
        printf("NVFP4 rows gate: FAIL\n");
        return 1;
    }
    printf("NVFP4 rows gate: PASS\n");
    return 0;
}

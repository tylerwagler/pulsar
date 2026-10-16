/* Row operations gate (L295, pulsar_cuda_rowops.h) -- model-free, links the PRODUCTION object
 * (src/cuda/pulsar_cuda_rowops.o).
 *
 *   1. rmsnorm   rows of width 4096 (Inkling's) and 1024: BIT-EXACT against a host replica of the kernel's one
 *                arithmetic (each of 256 threads an f32 fma chain over its 8-element chunks in order, the xor-shuffle
 *                fold, warp 0's ordered sum, rsqrtf, one rounding), TOLERANCE-graded against double; strided input
 *                and output.
 *   2. add       res = bf16(res + delta) bit-exact; the fused norm of the updated row bit-exact against the replica on
 *                the ROUNDED residual (vLLM's add_rmsnorm); the add alone leaves y untouched.
 *   3. emit      bf16(x) and bf16(x * s) for an f32 and a bf16 scalar, bit-exact; strided rows; the slack untouched.
 *   4. rows      a row's bytes are the same alone, in a batch of 300, and repeated (no batch dependence).
 * Then refusals. */
#include "cuda/pulsar_cuda_rowops.h"
#include "sconv_ref.h"   /* sref_f2bf / sref_bf2f: the bf16 conversions the block gates share */

#include <cuda_runtime.h>

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static int g_fail = 0;
static void check(bool ok, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void check(bool ok, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    printf("  [%s] ", ok ? "ok" : "FAIL"); vprintf(fmt, ap); printf("\n");
    fflush(stdout);
    va_end(ap);
    if (!ok) g_fail = 1;
}

static uint64_t mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(mix(seed + 0x9e3779b97f4a7c15ull)) {}
    double u() { s += 0x9e3779b97f4a7c15ull; return (double)(mix(s) >> 11) * (1.0 / 9007199254740992.0); }
    double n() { const double a = u() + 1e-12, b = u(); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }
};

/* the kernel's reduction, replayed on the host: thread i's f32 fma chain over chunks i, i + 256, ...; the warps' xor
 * folds (offsets 16, 8, 4, 2, 1 -- every lane ends with the same sum, lane 0's taken); warp 0 adds warp sums 0..7 */
static float replica_rstd(const uint16_t *x, uint32_t width, float eps) {
    float th[256];
    for (int t = 0; t < 256; t++) {
        float ss = 0.0f;
        for (uint32_t c = (uint32_t)t; c < width / 8; c += 256)
            for (int k = 0; k < 8; k++) {
                const float v = sref_bf2f(x[c * 8 + k]);
                ss = fmaf(v, v, ss);
            }
        th[t] = ss;
    }
    float warp[8];
    for (int w = 0; w < 8; w++) {
        float l[32];
        for (int i = 0; i < 32; i++) l[i] = th[w * 32 + i];
        for (int o = 16; o > 0; o >>= 1) {
            float n[32];
            for (int i = 0; i < 32; i++) n[i] = l[i] + l[i ^ o];
            memcpy(l, n, sizeof(l));
        }
        warp[w] = l[0];
    }
    float tot = 0.0f;
    for (int w = 0; w < 8; w++) tot += warp[w];
    return 1.0f / sqrtf(tot / (float)width + eps);   /* rsqrtf: compared at the bf16 output, see the tolerance */
}

/* the replica's bf16 outputs; `rstd` from the device rule above (rsqrtf is approximate -- the host uses the exact
 * reciprocal square root, so a row whose rsqrtf differs in its last bit can move an output by one ulp: counted) */
static void replica_norm(const uint16_t *x, const uint16_t *w, uint32_t width, float eps, uint16_t *y) {
    const float r = replica_rstd(x, width, eps);
    for (uint32_t c = 0; c < width; c++) y[c] = sref_f2bf((sref_bf2f(x[c]) * r) * sref_bf2f(w[c]));
}

static double norm_double(const uint16_t *x, const uint16_t *w, uint32_t width, float eps, std::vector<double> *y) {
    double ss = 0;
    for (uint32_t c = 0; c < width; c++) ss += (double)sref_bf2f(x[c]) * sref_bf2f(x[c]);
    const double r = 1.0 / sqrt(ss / width + eps);
    y->resize(width);
    for (uint32_t c = 0; c < width; c++) (*y)[c] = sref_bf2f(x[c]) * r * sref_bf2f(w[c]);
    return r;
}

static int bf_ulps(uint16_t a, uint16_t b) {
    auto ord = [](uint16_t v) { return (v & 0x8000u) ? -(int)(v & 0x7fffu) : (int)v; };
    return abs(ord(a) - ord(b));
}

template <class T> static T *dev(const std::vector<T> &h) {
    T *d = nullptr;
    CK(cudaMalloc(&d, h.size() * sizeof(T) + 16));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> static std::vector<T> host(const T *d, size_t n) {
    std::vector<T> h(n);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

static std::vector<uint16_t> rand_bf16(Rng &g, size_t n, double scale) {
    std::vector<uint16_t> v(n);
    for (size_t i = 0; i < n; i++) v[i] = sref_f2bf((float)(g.n() * scale));
    return v;
}

static void part_norm(uint32_t width, uint32_t rows) {
    Rng g(width * 31 + rows);
    const uint64_t ldx = width + 24, ldy = width + 40;
    const float eps = 1e-6f;
    std::vector<uint16_t> x((size_t)rows * ldx), w = rand_bf16(g, width, 0.3);
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < width; c++) x[(size_t)r * ldx + c] = sref_f2bf((float)(g.n() * (0.05 + r % 7)));
    std::vector<uint16_t> ypre((size_t)rows * ldy, 0x7777u);
    uint16_t *dx = dev(x), *dw = dev(w), *dy = dev(ypre);
    check(pulsar_rows_rmsnorm_launch(dx, ldx, rows, width, dw, eps, dy, ldy, 0) == 0, "rmsnorm %u rows of %u launched",
          rows, width);
    const std::vector<uint16_t> y = host(dy, (size_t)rows * ldy);
    uint64_t exact = 0, total = 0, off1 = 0, slack_bad = 0;
    double num = 0, den = 0;
    std::vector<uint16_t> rep(width);
    std::vector<double> ref;
    for (uint32_t r = 0; r < rows; r++) {
        replica_norm(&x[(size_t)r * ldx], w.data(), width, eps, rep.data());
        norm_double(&x[(size_t)r * ldx], w.data(), width, eps, &ref);
        for (uint32_t c = 0; c < width; c++) {
            const uint16_t got = y[(size_t)r * ldy + c];
            exact += got == rep[c];
            off1 += bf_ulps(got, rep[c]) > 1;
            total++;
            const double d = sref_bf2f(got) - ref[c];
            num += d * d;
            den += ref[c] * ref[c];
        }
        for (uint64_t c = width; c < ldy; c++) slack_bad += y[(size_t)r * ldy + c] != 0x7777u;
    }
    const double fro = sqrt(num / den);
    check(off1 == 0 && exact * 1000 >= total * 999,
          "rmsnorm %ux%u: %.4f%% == the host replica's bytes, none past 1 ulp (rsqrtf's last bit)", rows, width,
          100.0 * exact / total);
    check(fro < 3e-3, "rmsnorm %ux%u vs double: rel-fro %.3e (bf16's rounding)", rows, width, fro);
    check(slack_bad == 0, "rmsnorm: the output stride's slack untouched");
    /* a row's bytes: alone, and in the batch */
    for (const uint32_t r : {0u, rows / 2, rows - 1}) {
        uint16_t *d1 = nullptr;
        CK(cudaMalloc(&d1, width * 2));
        check(pulsar_rows_rmsnorm_launch(dx + (size_t)r * ldx, ldx, 1, width, dw, eps, d1, width, 0) == 0, "one row");
        const std::vector<uint16_t> one = host(d1, width);
        check(memcmp(one.data(), &y[(size_t)r * ldy], width * 2) == 0, "rmsnorm row %u alone == in the batch of %u",
              r, rows);
        CK(cudaFree(d1));
    }
    CK(cudaFree(dx)); CK(cudaFree(dw)); CK(cudaFree(dy));
}

static void part_add(uint32_t width, uint32_t rows) {
    Rng g(width * 7 + rows);
    const float eps = 1e-6f;
    std::vector<uint16_t> res = rand_bf16(g, (size_t)rows * width, 2.0), d = rand_bf16(g, (size_t)rows * width, 0.5),
                          w = rand_bf16(g, width, 0.3), ypre((size_t)rows * width, 0x1234u);
    uint16_t *dr = dev(res), *dd = dev(d), *dw = dev(w), *dy = dev(ypre);
    check(pulsar_rows_add_launch(dr, width, dd, width, rows, width, dw, eps, dy, width, 0) == 0, "add + norm launched");
    const std::vector<uint16_t> r2 = host(dr, res.size()), y = host(dy, ypre.size());
    uint64_t add_ok = 0, norm_exact = 0, off1 = 0;
    std::vector<uint16_t> rep(width);
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < width; c++) {
            const size_t i = (size_t)r * width + c;
            add_ok += r2[i] == sref_f2bf(sref_bf2f(res[i]) + sref_bf2f(d[i]));
        }
        replica_norm(&r2[(size_t)r * width], w.data(), width, eps, rep.data());
        for (uint32_t c = 0; c < width; c++) {
            norm_exact += y[(size_t)r * width + c] == rep[c];
            off1 += bf_ulps(y[(size_t)r * width + c], rep[c]) > 1;
        }
    }
    const uint64_t n = (uint64_t)rows * width;
    check(add_ok == n, "add %ux%u: res == bf16(res + delta) on every element (%llu of %llu)", rows, width,
          (unsigned long long)add_ok, (unsigned long long)n);
    check(off1 == 0 && norm_exact * 1000 >= n * 999, "add's norm %ux%u: %.4f%% == the replica on the ROUNDED residual, "
          "none past 1 ulp", rows, width, 100.0 * norm_exact / n);
    /* the add alone */
    std::vector<uint16_t> y2pre((size_t)rows * width, 0x4321u);
    uint16_t *dr2 = dev(res), *dy2 = dev(y2pre);
    check(pulsar_rows_add_launch(dr2, width, dd, width, rows, width, nullptr, 0.0f, nullptr, 0, 0) == 0, "add alone");
    const std::vector<uint16_t> r3 = host(dr2, res.size()), y3 = host(dy2, y2pre.size());
    check(memcmp(r3.data(), r2.data(), r3.size() * 2) == 0, "the add alone == the add of the fused call");
    check(memcmp(y3.data(), y2pre.data(), y3.size() * 2) == 0, "the add alone writes no normed rows");
    CK(cudaFree(dr)); CK(cudaFree(dd)); CK(cudaFree(dw)); CK(cudaFree(dy)); CK(cudaFree(dr2)); CK(cudaFree(dy2));
}

static void part_emit() {
    Rng g(99);
    const uint32_t width = 6656, rows = 37;
    const uint64_t ldx = width + 4, ldy = width + 8;
    std::vector<float> x((size_t)rows * ldx);
    for (auto &v : x) v = (float)(g.n() * 3.0);
    const float sf = 0.73f;
    const uint16_t sb = sref_f2bf(1.375f);
    float *dx = dev(x), *dsf = dev(std::vector<float>{sf});
    uint16_t *dsb = dev(std::vector<uint16_t>{sb});
    for (int kind = 0; kind < 3; kind++) {
        std::vector<uint16_t> ypre((size_t)rows * ldy, 0x5555u);
        uint16_t *dy = dev(ypre);
        const void *s = kind == 0 ? nullptr : kind == 1 ? (const void *)dsf : (const void *)dsb;
        check(pulsar_rows_emit_bf16_launch(dx, ldx, rows, width, s, (pulsar_rows_scale)kind, dy, ldy, 0) == 0,
              "emit kind %d launched", kind);
        const std::vector<uint16_t> y = host(dy, ypre.size());
        uint64_t ok = 0, slack = 0;
        const float m = kind == 0 ? 1.0f : kind == 1 ? sf : sref_bf2f(sb);
        for (uint32_t r = 0; r < rows; r++) {
            for (uint32_t c = 0; c < width; c++) {
                const float v = x[(size_t)r * ldx + c];
                ok += y[(size_t)r * ldy + c] == sref_f2bf(kind == 0 ? v : v * m);
            }
            for (uint64_t c = width; c < ldy; c++) slack += y[(size_t)r * ldy + c] != 0x5555u;
        }
        check(ok == (uint64_t)rows * width && slack == 0, "emit %s: every element == bf16(x%s), the slack untouched",
              kind == 0 ? "plain" : kind == 1 ? "x f32 scalar" : "x bf16 scalar", kind == 0 ? "" : " * s");
        CK(cudaFree(dy));
    }
    CK(cudaFree(dx)); CK(cudaFree(dsf)); CK(cudaFree(dsb));
}

static void part_refusals() {
    uint16_t *d = nullptr;
    float *f = nullptr;
    CK(cudaMalloc(&d, 1 << 20));
    CK(cudaMalloc(&f, 1 << 20));
    printf("  (refusals below print their reasons)\n");
    check(pulsar_rows_rmsnorm_launch(d, 100, 1, 100, d, 1e-6f, d + 4096, 100, 0) == -1, "rmsnorm: width not a multiple of 8");
    check(pulsar_rows_rmsnorm_launch(d, 64, 1, 64, nullptr, 1e-6f, d + 4096, 64, 0) == -1, "rmsnorm: no weight");
    check(pulsar_rows_rmsnorm_launch(d + 1, 64, 1, 64, d, 1e-6f, d + 4096, 64, 0) == -1, "rmsnorm: misaligned rows");
    check(pulsar_rows_add_launch(d, 64, nullptr, 64, 1, 64, nullptr, 0.f, nullptr, 0, 0) == -1, "add: no delta");
    check(pulsar_rows_add_launch(d, 64, d + 1024, 64, 1, 64, d, -1.f, d + 4096, 64, 0) == -1, "add: negative eps");
    check(pulsar_rows_emit_bf16_launch(f, 64, 1, 64, f, PULSAR_ROWS_SCALE_NONE, d, 64, 0) == -1,
          "emit: a scalar with kind NONE");
    check(pulsar_rows_emit_bf16_launch(f, 64, 1, 64, nullptr, PULSAR_ROWS_SCALE_F32, d, 64, 0) == -1,
          "emit: kind F32 with no scalar");
    check(pulsar_rows_emit_bf16_launch(f, 66, 1, 64, nullptr, PULSAR_ROWS_SCALE_NONE, d, 64, 0) == -1,
          "emit: ld_x not a multiple of 4");
    CK(cudaFree(d));
    CK(cudaFree(f));
}

int main() {
    printf("rowops gate (L295: the row operations between a layer plan's blocks)\n");
    part_norm(4096, 300);
    part_norm(1024, 17);
    part_add(4096, 129);
    part_emit();
    part_refusals();
    printf(g_fail ? "ROWOPS GATE FAIL\n" : "ROWOPS GATE PASS\n");
    return g_fail;
}

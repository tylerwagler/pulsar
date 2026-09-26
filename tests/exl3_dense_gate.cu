/* EXL3 DENSE GATE (L251): the dense-Linear arm against the host authority.
 *
 * Model-free.  Random trellis words (any 16-bit word is a valid tile), fp16
 * scales in the real checkpoints' distribution (suh ~0.025 with signs, svh
 * ~+-1), activations drawn from a heavy-tailed distribution and ENCODED to the
 * A8 slot the way the producers encode (shared exponent floor(log2 amax) - 7,
 * E4M3 round-to-nearest-even) -- not random E4M3 bytes, which are uniform in
 * exponent and make a degenerate fixture.  The reference is built only from
 * src/engine/exl3_trellis.h (the host dequant that tests/exl3_dequant_gate.cpp
 * holds byte-exact to exllamav3) and double arithmetic:
 *
 *   y = svh * H128( W_hat^T H128(suh * x) )     per row, per 128-block
 *
 * It grades the PRODUCTION object (src/cuda/mmq/ds4_exl3_dense.o, built with
 * the engine's flags), not a re-compile of the TU:
 *   1. every Qwen dense shape the lane needs (2560->10240 DeltaNet in_proj_qkv,
 *      2560->6144 in_proj_z, 6144->2560 out_proj, 2560->1280) and a DeepSeek
 *      dense shape (4096->1024, attn_q_a), at K = 2, 3, 4, 5: the 16-row output
 *      vs the double reference, max and median relative error (tolerance = f32
 *      accumulation order; the median proves the fixture is not degenerate);
 *   2. M = 1, 2, 4, 8 are BIT-IDENTICAL to the same rows of the M = 16 run (a
 *      row's bytes do not depend on the batch width -- the decode-row contract);
 *   3. prefill: M = 40 (three row blocks, the last partial) at one shape and
 *      M = 300 (three 128-row slabs, the last partial) at another -- rows
 *      0..15 bit-identical to M = 16, every other row against the reference;
 *   4. mutations: one flipped trellis bit must move outputs, and a sign-flipped
 *      suh block must push the error far past the tolerance;
 *   5. refusals: K % 128, an uninstantiated rate, a short workspace, M = 0.
 *
 * usage: ./tests/exl3_dense_gate
 */
#include "../src/cuda/mmq/ds4_exl3_dense.cuh"
#include "../src/cuda/pulsar_cuda_mx.cuh"
#include "../src/engine/exl3_trellis.h"
#include "exl3_dense_ref.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "EXL3-DENSE FAIL: " __VA_ARGS__); \
                                       fprintf(stderr, "\n"); g_fail = 1; } } while (0)
#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "EXL3-DENSE FAIL: %s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 16); }
static double rndu(void) { return ((double)rnd() + 0.5) / 4294967296.0; }
static double rndn(void) { return sqrt(-2.0 * log(rndu())) * cos(6.283185307179586 * rndu()); }

/* round-to-nearest-even onto the finite E4M3 grid, saturating (the device's
 * cvt.rn.satfinite): positive codes 0x00..0x7e are increasing in value, and a
 * tie picks the even code, whose mantissa LSB is 0 */
static uint8_t f64_to_e4m3(double v) {
    const uint8_t sign = v < 0 ? 0x80 : 0;
    const double a = fabs(v);
    int lo = 0, hi = 0x7e;
    if (a >= exl3t_e4m3_to_f64(0x7e)) return sign | 0x7e;
    while (hi - lo > 1) { const int mid = (lo + hi) / 2; if (exl3t_e4m3_to_f64((uint8_t)mid) <= a) lo = mid; else hi = mid; }
    const double dl = a - exl3t_e4m3_to_f64((uint8_t)lo), dh = exl3t_e4m3_to_f64((uint8_t)hi) - a;
    const int code = dl < dh ? lo : dh < dl ? hi : ((lo & 1) ? hi : lo);
    return sign | (uint8_t)code;
}

/* The weight: one [trellis | suh | svh] slice. */
struct weight {
    int K, N, k2;
    uint64_t trellis = 0, scales = 0, stride = 0;
    std::vector<uint8_t> bytes;
    uint16_t *words() { return (uint16_t *)bytes.data(); }
    const uint16_t *suh() const { return (const uint16_t *)(bytes.data() + trellis); }
    const uint16_t *svh() const { return suh() + K; }
};
static weight make_weight(int K, int N, int k2) {
    weight w; w.K = K; w.N = N; w.k2 = k2;
    if (!exl3_expert_layout(K, N, k2, &w.trellis, &w.scales, &w.stride)) { fprintf(stderr, "layout refused\n"); exit(1); }
    w.bytes.resize(w.stride);
    uint16_t *t = (uint16_t *)w.bytes.data();
    for (uint64_t i = 0; i < w.trellis / 2; i++) t[i] = (uint16_t)rnd();
    uint16_t *s = (uint16_t *)(w.bytes.data() + w.trellis);
    for (int i = 0; i < K; i++) s[i] = exl3_f32_to_f16((float)((0.5 + rndu()) * 0.025 * ((rnd() & 1) ? 1.0 : -1.0)));
    for (int i = 0; i < N; i++) s[K + i] = exl3_f32_to_f16((float)((0.75 + 0.5 * rndu()) * ((rnd() & 1) ? 1.0 : -1.0)));
    return w;
}
/* The activation slot: E4M3 [rows][K] + the swizzled ue8m0 slab, and x decoded. */
struct slot {
    int rows, K, kbp;
    std::vector<uint8_t> q, sf;
    std::vector<double> x;
};
static slot make_slot(int rows, int K) {
    slot s; s.rows = rows; s.K = K; s.kbp = pulsar_mx_kbp(K);
    s.q.assign((size_t)rows * K, 0);
    s.sf.assign(pulsar_mx_sf_slab_bytes(rows, s.kbp), 0);
    s.x.assign((size_t)rows * K, 0.0);
    std::vector<double> chan(K);
    for (int k = 0; k < K; k++) chan[k] = (rnd() % 64 == 0) ? 12.0 : 1.0;    /* a few outlier channels */
    for (int r = 0; r < rows; r++) {
        std::vector<double> v(K);
        for (int k = 0; k < K; k++) v[k] = rndn() * chan[k] * 0.7;
        for (int g = 0; g < K / 32; g++) {
            double amax = 0;
            for (int j = 0; j < 32; j++) amax = fmax(amax, fabs(v[g * 32 + j]));
            int se = amax > 0 ? (int)floor(log2(amax)) - 7 : -127;    /* pulsar_mx_shared_exp */
            se = std::max(-127, std::min(127, se));
            s.sf[pulsar_mx_sfoff(r, g, s.kbp)] = (uint8_t)(se + 127);
            for (int j = 0; j < 32; j++) {
                const uint8_t b = f64_to_e4m3(v[g * 32 + j] * ldexp(1.0, -se));
                s.q[(size_t)r * K + g * 32 + j] = b;
                s.x[(size_t)r * K + g * 32 + j] = exl3t_e4m3_to_f64(b) * ldexp(1.0, se);
            }
        }
    }
    return s;
}

/* max and median of |got - want| / max|want of the row| over rows */
static void grade(const std::vector<float> &got, const std::vector<double> &want, int rows, int N, int row_off,
                  double *mx, double *med) {
    std::vector<double> e;
    e.reserve((size_t)rows * N);
    *mx = 0;
    for (int r = 0; r < rows; r++) {
        double sc = 0;
        for (int n = 0; n < N; n++) sc = fmax(sc, fabs(want[(size_t)(r + row_off) * N + n]));
        for (int n = 0; n < N; n++) {
            const double d = fabs((double)got[(size_t)r * N + n] - want[(size_t)(r + row_off) * N + n]) / sc;
            e.push_back(d);
            *mx = fmax(*mx, d);
        }
    }
    std::nth_element(e.begin(), e.begin() + e.size() / 2, e.end());
    *med = e[e.size() / 2];
}

struct dev {
    uint8_t *w = nullptr, *q = nullptr, *sf = nullptr;
    float *y = nullptr, *ws = nullptr;
    size_t ws_bytes = 0;
};
static void upload(const weight &w, const slot &s, int max_rows, dev &d) {
    CUDA_OK(cudaMalloc((void **)&d.w, w.stride));
    CUDA_OK(cudaMemcpy(d.w, w.bytes.data(), w.stride, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc((void **)&d.q, s.q.size()));
    CUDA_OK(cudaMemcpy(d.q, s.q.data(), s.q.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc((void **)&d.sf, s.sf.size()));
    CUDA_OK(cudaMemcpy(d.sf, s.sf.data(), s.sf.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc((void **)&d.y, (size_t)max_rows * w.N * sizeof(float)));
    d.ws_bytes = ds4_exl3_dense_workspace_bytes(max_rows, w.K, w.N);
    CUDA_OK(cudaMalloc((void **)&d.ws, d.ws_bytes));
}
static void release(dev &d) { cudaFree(d.w); cudaFree(d.q); cudaFree(d.sf); cudaFree(d.y); cudaFree(d.ws); d = dev(); }

static int run(const dev &d, int k2, int M, int K, int N, std::vector<float> &y) {
    CUDA_OK(cudaMemset(d.y, 0xff, (size_t)M * N * sizeof(float)));      /* NaN canary: every output must be written */
    const int rc = ds4_exl3_dense_launch(d.w, k2, d.q, d.sf, d.y, M, K, N, d.ws, d.ws_bytes, 0);
    if (rc) return rc;
    CUDA_OK(cudaDeviceSynchronize());
    y.resize((size_t)M * N);
    CUDA_OK(cudaMemcpy(y.data(), d.y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

int main(void) {
    struct shape { int K, N; const char *what; };
    const shape shapes[] = {
        {2560, 10240, "Qwen DeltaNet in_proj_qkv"},
        {2560, 6144,  "Qwen DeltaNet in_proj_z"},
        {6144, 2560,  "Qwen DeltaNet out_proj"},
        {2560, 1280,  "Qwen 2560->1280"},
        {4096, 1024,  "DeepSeek attn_q_a"},
    };
    const int rates[] = {4, 6, 8, 10};
    const int widths[] = {1, 2, 4, 8};
    const double tol = 2e-5;
    printf("exl3-dense-gate: the EXL3 dense arm (production object) vs src/engine/exl3_trellis.h + double\n");

    for (const shape &sh : shapes) {
        const int K = sh.K, N = sh.N;
        const int max_rows = (K == 2560 && N == 6144) ? 40 : (K == 2560 && N == 1280) ? 300 : 16;
        const slot s = make_slot(max_rows, K);
        for (int k2 : rates) {
            const weight w = make_weight(K, N, k2);
            std::vector<double> what, yref;
            exl3t_dequant(w.bytes.data(), K, N, k2, what);
            exl3t_reference(w.bytes.data(), what, K, N, k2, s.x.data(), max_rows, yref);
            dev d;
            upload(w, s, max_rows, d);
            std::vector<float> y16, y;
            int rc = run(d, k2, 16, K, N, y16);
            CHECK(rc == 0, "%s K=%d: launch rc=%d", sh.what, k2 / 2, rc);
            if (rc) { release(d); continue; }
            double mx, med;
            grade(y16, yref, 16, N, 0, &mx, &med);
            size_t nonfinite = 0;
            for (float v : y16) nonfinite += !std::isfinite(v);
            CHECK(nonfinite == 0, "%s K=%d: %zu non-finite outputs (an unwritten NaN canary)", sh.what, k2 / 2, nonfinite);
            CHECK(mx < tol, "%s %dx%d K=%d M=16: max rel %.3e (limit %.0e)", sh.what, K, N, k2 / 2, mx, tol);
            CHECK(med > 1e-10, "%s K=%d: median rel %.3e -- a degenerate fixture, not a pass", sh.what, k2 / 2, med);
            size_t diff_m = 0;
            for (int M : widths) {
                rc = run(d, k2, M, K, N, y);
                CHECK(rc == 0, "%s K=%d M=%d: launch rc=%d", sh.what, k2 / 2, M, rc);
                if (!rc) diff_m += memcmp(y.data(), y16.data(), y.size() * sizeof(float)) != 0;
            }
            CHECK(diff_m == 0, "%s K=%d: %zu of 4 widths (M=1,2,4,8) differ from the same rows at M=16", sh.what, k2 / 2, diff_m);
            printf("  %-26s %5d -> %-5d K=%d splits %d: M=16 max rel %.2e median %.2e; M=1,2,4,8 %s\n",
                   sh.what, K, N, k2 / 2, ds4_exl3_dense_splits(K, N), mx, med,
                   diff_m ? "DIFFER" : "bit-identical to M=16's rows");

            if (max_rows > 16 && k2 == 8) {
                /* 3. prefill rows: row blocks of 16 (the last partial), and slabs of 128 */
                rc = run(d, k2, max_rows, K, N, y);
                CHECK(rc == 0, "prefill M=%d launch rc=%d", max_rows, rc);
                if (!rc) {
                    const bool same16 = memcmp(y.data(), y16.data(), y16.size() * sizeof(float)) == 0;
                    std::vector<float> tail(y.begin() + (size_t)16 * N, y.end());
                    grade(tail, yref, max_rows - 16, N, 16, &mx, &med);
                    CHECK(same16, "prefill M=%d: rows 0..15 differ from the M=16 run", max_rows);
                    CHECK(mx < tol, "prefill M=%d: rows 16..%d max rel %.3e", max_rows, max_rows - 1, mx);
                    printf("  prefill M=%d (%s): rows 0..15 %s; rows 16..%d max rel %.2e median %.2e\n", max_rows,
                           max_rows > 128 ? "slabs 128+128+44" : "row blocks 16+16+8",
                           same16 ? "bit-identical to M=16" : "DIFFER", max_rows - 1, mx, med);
                }
            }
            if (sh.N == 10240 && k2 == 8) {
                /* 4. mutations: the gate must be able to fail */
                weight wm = w;
                wm.words()[12345] ^= 0x0100u;
                dev dm; upload(wm, s, 16, dm);
                rc = run(dm, k2, 16, K, N, y);
                size_t moved = 0;
                for (size_t i = 0; i < y.size(); i++) moved += y[i] != y16[i];
                CHECK(rc == 0 && moved > 0, "mutation: a flipped trellis bit moved no output -- the comparison is degenerate");
                release(dm);
                weight ws2 = w;
                uint16_t *su = (uint16_t *)(ws2.bytes.data() + ws2.trellis);
                for (int k = 256; k < 384; k++) su[k] ^= 0x8000u;                /* one suh block sign-flipped */
                upload(ws2, s, 16, dm);
                rc = run(dm, k2, 16, K, N, y);
                double mmx = 0, mmed = 0;
                if (!rc) grade(y, yref, 16, N, 0, &mmx, &mmed);
                CHECK(rc == 0 && mmx > 100 * tol, "mutation: a sign-flipped suh block graded %.3e -- the reference would not catch it", mmx);
                release(dm);
                printf("  mutations: one trellis bit moved %zu outputs; a sign-flipped suh block grades max rel %.2e (%.0fx the limit)\n",
                       moved, mmx, mmx / tol);

                /* 5. refusals */
                void *ws = d.ws;
                CHECK(ds4_exl3_dense_launch(d.w, k2, d.q, d.sf, d.y, 16, K + 64, N, ws, d.ws_bytes, 0) == -1, "K %% 128 accepted");
                CHECK(ds4_exl3_dense_launch(d.w, 5, d.q, d.sf, d.y, 16, K, N, ws, d.ws_bytes, 0) == -1, "rate 2.5 accepted");
                CHECK(ds4_exl3_dense_launch(d.w, k2, d.q, d.sf, d.y, 16, K, N, ws, d.ws_bytes - 4, 0) == -1, "short workspace accepted");
                CHECK(ds4_exl3_dense_launch(d.w, k2, d.q, d.sf, d.y, 0, K, N, ws, d.ws_bytes, 0) == -1, "M = 0 accepted");
                CHECK(ds4_exl3_dense_workspace_bytes(16, K + 64, N) == 0, "workspace sized for a refused shape");
                printf("  refusals: K %% 128, rate 2.5, short workspace, M = 0 -- all refused\n");
            }
            release(d);
        }
    }
    printf(g_fail ? "EXL3-DENSE GATE FAIL\n" : "EXL3-DENSE GATE PASS\n");
    return g_fail;
}

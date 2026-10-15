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
 *      row's bytes do not depend on the batch width -- the decode-row contract); L288 C1: so are DECODE rows past
 *      16 (M = 17, 32 and the shape's widest: 16-row GEMV blocks in turn) -- every 16-row
 *      block bit for bit a launch of those rows alone, and graded against the double reference;
 *   3. prefill: prompt chunks of M = 40 / 97 / 300 at several shapes -- the prefill GEMM (one fp16 operand plane):
 *      every output within the fp16-operand envelope (exl3t_fp16_operand_bound) plus f32 order;
 *   4. mutations: one flipped trellis bit must move outputs, and a sign-flipped
 *      suh block must push the error far past the tolerance;
 *   5. refusals: K % 128, an uninstantiated rate, a short workspace, M = 0.
 *   6. the multi launch (L288 exl3-dense): each output bit for bit its single launch.
 *   7. the heads (L288 qwen-exl3-head): lm_head 2560 -> 248320 and the draft head 2560 -> 82304 at K = 3..5 --
 *      M-independent rows, and windows cut from the head equal its columns bit for bit and grade vs double.
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
/* The activation: BF16 rows [rows][K], and x is the same value read back.
 *
 * L251 / ac69748f: this was an E4M3 slot -- codes plus a swizzled ue8m0 slab -- and both sides of
 * the comparison used the DECODED code.  The dense arm reads bf16 now, so the slot is gone and the
 * reference's activation is the bf16 VALUE.  The rounding goes through __float2bfloat16 rather than
 * a hand-rolled bit trick, so the reference cannot drift from what the kernel does. */
struct slot {
    int rows, K;
    std::vector<uint16_t> q;      /* bf16 bit patterns, [rows][K] */
    std::vector<double> x;
};
static uint16_t bf16_bits(double v) {
    const __nv_bfloat16 b = __float2bfloat16((float)v);
    uint16_t u = 0;
    std::memcpy(&u, &b, sizeof(u));
    return u;
}
static double bf16_back(uint16_t u) {
    __nv_bfloat16 b;
    std::memcpy(&b, &u, sizeof(u));
    return (double)__bfloat162float(b);
}
static slot make_slot(int rows, int K) {
    slot s; s.rows = rows; s.K = K;
    s.q.assign((size_t)rows * K, 0);
    s.x.assign((size_t)rows * K, 0.0);
    std::vector<double> chan(K);
    for (int k = 0; k < K; k++) chan[k] = (rnd() % 64 == 0) ? 12.0 : 1.0;    /* a few outlier channels */
    for (int r = 0; r < rows; r++) {
        for (int k = 0; k < K; k++) {
            const uint16_t b = bf16_bits(rndn() * chan[k] * 0.7);
            s.q[(size_t)r * K + k] = b;
            s.x[(size_t)r * K + k] = bf16_back(b);
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
    uint8_t *w = nullptr;             /* the trellis weights */
    uint8_t *q = nullptr;             /* the bf16 activation rows (2 bytes per element) */
    float *y = nullptr, *ws = nullptr;
    size_t ws_bytes = 0;
};
static void upload(const weight &w, const slot &s, int max_rows, dev &d) {
    CUDA_OK(cudaMalloc((void **)&d.w, w.stride));
    CUDA_OK(cudaMemcpy(d.w, w.bytes.data(), w.stride, cudaMemcpyHostToDevice));
    const size_t qb = s.q.size() * sizeof(uint16_t);
    CUDA_OK(cudaMalloc((void **)&d.q, qb));
    CUDA_OK(cudaMemcpy(d.q, s.q.data(), qb, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc((void **)&d.y, (size_t)max_rows * w.N * sizeof(float)));
    d.ws_bytes = ds4_exl3_dense_workspace_bytes(max_rows, w.K, w.N);
    CUDA_OK(cudaMalloc((void **)&d.ws, d.ws_bytes));
}
static void release(dev &d) { cudaFree(d.w); cudaFree(d.q); cudaFree(d.y); cudaFree(d.ws); d = dev(); }

static int run(const dev &d, int k2, int M, int K, int N, std::vector<float> &y, bool prompt = false, int r0 = 0) {
    CUDA_OK(cudaMemset(d.y, 0xff, (size_t)M * N * sizeof(float)));      /* NaN canary: every output must be written */
    /* r0: the call's rows start at activation row r0 (a block of a wider call, run alone) */
    const int rc = ds4_exl3_dense_launch(d.w, k2, d.q + (size_t)r0 * K * 2, d.y, M, K, N, d.ws, d.ws_bytes, 0, prompt);
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
        {2560, 640,   "Qwen shared gate/up"},
        {640,  2560,  "Qwen shared down"},
        {4096, 1024,  "DeepSeek attn_q_a"},
    };
    const int rates[] = {4, 6, 8, 10, 12, 16};
    const int widths[] = {1, 2, 4, 8};
    const double tol = 2e-5;
    printf("exl3-dense-gate: the EXL3 dense arm (production object) vs src/engine/exl3_trellis.h + double\n");

    for (const shape &sh : shapes) {
        const int K = sh.K, N = sh.N;
        const int max_rows = (K == 2560 && N == 6144) ? 40 : (K == 2560 && N == 1280) ? 300
                           : (N == 640 || K == 640) ? 97 : 16;
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

            if (max_rows > 16) {
                /* 2b. L288 C1: decode rows past 16 -- every 16-row block of the call bit for bit those rows run alone
                 * (rows 0..15 the M = 16 run's), every row graded against the double reference */
                size_t diff_w = 0, blocks = 0;
                double wmx = 0, wmed = 0;
                std::vector<float> yb;
                for (int Mw : {17, 32, max_rows}) {
                    rc = run(d, k2, Mw, K, N, y);
                    CHECK(rc == 0, "%s K=%d decode M=%d: launch rc=%d", sh.what, k2 / 2, Mw, rc);
                    if (rc) continue;
                    grade(y, yref, Mw, N, 0, &mx, &med);
                    wmx = fmax(wmx, mx);
                    wmed = med;
                    CHECK(mx < tol, "%s K=%d decode M=%d: max rel %.3e (limit %.0e)", sh.what, k2 / 2, Mw, mx, tol);
                    const std::vector<float> yw = y;
                    for (int r0 = 0; r0 < Mw; r0 += 16) {
                        const int m = Mw - r0 < 16 ? Mw - r0 : 16;
                        rc = run(d, k2, m, K, N, yb, false, r0);
                        CHECK(rc == 0, "%s K=%d decode rows %d..%d alone: launch rc=%d", sh.what, k2 / 2, r0, r0 + m - 1, rc);
                        if (!rc) diff_w += memcmp(yb.data(), yw.data() + (size_t)r0 * N, yb.size() * sizeof(float)) != 0;
                        blocks++;
                    }
                }
                CHECK(diff_w == 0, "%s K=%d: %zu of %zu 16-row blocks of decode calls past 16 rows differ from those rows "
                      "alone", sh.what, k2 / 2, diff_w, blocks);
                printf("  decode M=17/32/%d: %zu row blocks bit-identical to their rows alone; max rel %.2e median %.2e\n",
                       max_rows, blocks - diff_w, wmx, wmed);
                /* 3. prefill rows: a prompt chunk, row blocks of 16 (the last partial), and slabs of 128 */
                rc = run(d, k2, max_rows, K, N, y, true);
                CHECK(rc == 0, "prefill M=%d launch rc=%d", max_rows, rc);
                if (!rc) {
                    /* a prompt chunk is the PREFILL GEMM (qwen_exl3_dense_prefill.cu, L251): it sums in its own
                     * order, so its rows agree with the decode GEMV to rounding, not to the bit -- every
                     * row is graded against the double reference instead, rows 0..15 included */
                    /* ...its operand is ONE fp16 plane (qwen_exl3_dense_prefill.cu), so each output is graded
                     * against that rounding's exact envelope (exl3t_fp16_operand_bound) plus the f32-order
                     * tolerance of its row -- not the GEMV's f32-class limit */
                    const bool same16 = memcmp(y.data(), y16.data(), y16.size() * sizeof(float)) == 0;
                    std::vector<double> bnd;
                    exl3t_fp16_operand_bound(w.bytes.data(), what, K, N, k2, s.x.data(), max_rows, bnd);
                    size_t over = 0;
                    double worst = 0;
                    for (int r = 0; r < max_rows; r++) {
                        double sc = 0;
                        for (int n = 0; n < N; n++) sc = fmax(sc, fabs(yref[(size_t)r * N + n]));
                        for (int n = 0; n < N; n++) {
                            const size_t i = (size_t)r * N + n;
                            const double err = fabs((double)y[i] - yref[i]), lim = bnd[i] + tol * sc;
                            over += err > lim;
                            worst = fmax(worst, err / lim);
                        }
                    }
                    grade(y, yref, max_rows, N, 0, &mx, &med);
                    CHECK(over == 0, "prefill M=%d: %zu outputs past the fp16-operand envelope (worst err/limit %.3f)",
                          max_rows, over, worst);
                    printf("  prefill GEMM M=%d: every output within its fp16-operand envelope (worst err/limit %.3f); "
                           "max rel %.2e median %.2e (rows 0..15 %s the GEMV's bits)\n", max_rows, worst, mx, med,
                           same16 ? "equal to" : "rounding-close to");
                    /* L266: a PROMPT chunk of any width takes this GEMM, and a row's bytes do not depend on
                     * how many rows share the call (the pinned cuBLASLt algorithm, no split-K) -- what makes
                     * a prompt cut anywhere equal to one prefilled whole */
                    size_t diff_p = 0;
                    std::vector<float> yp;
                    for (int Mp : {1, 3, 16, 17}) {
                        rc = run(d, k2, Mp, K, N, yp, true);
                        CHECK(rc == 0, "prompt M=%d launch rc=%d", Mp, rc);
                        if (!rc) diff_p += memcmp(yp.data(), y.data(), yp.size() * sizeof(float)) != 0;
                    }
                    CHECK(diff_p == 0, "%s K=%d: %zu of 4 prompt widths (M=1,3,16,17) differ from the same rows at M=%d",
                          sh.what, k2 / 2, diff_p, max_rows);
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
                CHECK(ds4_exl3_dense_launch(d.w, k2, d.q, d.y, 16, K + 64, N, ws, d.ws_bytes, 0) == -1, "K %% 128 accepted");
                CHECK(ds4_exl3_dense_launch(d.w, 5, d.q, d.y, 16, K, N, ws, d.ws_bytes, 0) == -1, "rate 2.5 accepted");
                CHECK(ds4_exl3_dense_launch(d.w, k2, d.q, d.y, 16, K, N, ws, d.ws_bytes - 4, 0) == -1, "short workspace accepted");
                CHECK(ds4_exl3_dense_launch(d.w, k2, d.q, d.y, 0, K, N, ws, d.ws_bytes, 0) == -1, "M = 0 accepted");
                CHECK(ds4_exl3_dense_workspace_bytes(16, K + 64, N) == 0, "workspace sized for a refused shape");
                printf("  refusals: K %% 128, rate 2.5, short workspace, M = 0 -- all refused\n");
            }
            release(d);
        }
    }
    /* 6. L288 exl3-dense: the multi launch.  Up to four projections of one input -- mixed widths and rates, the
     * shapes the GDN / QSA blocks and the shared expert give it -- each output bit for bit its single launch's,
     * at every decode width, from one prep launch and one GEMV launch per rate; prompt chunks too. */
    {
        struct set { int K, n; int N[4], k2[4]; const char *what; };
        const set sets[] = {
            {2560, 4, {12288, 512, 512, 640}, {10, 10, 10, 10}, "Qwen QSA q / k / v / index (one rate)"},
            {2560, 2, {10240, 6144, 0, 0}, {10, 10, 0, 0}, "Qwen GDN in_proj_qkv / _z"},
            {2560, 2, {640, 640, 0, 0}, {10, 8, 0, 0}, "a shared gate / up at two rates (two GEMVs)"},
            {2560, 3, {1280, 512, 640, 0}, {8, 10, 8, 0}, "three projections, rates 8 / 10 / 8"},
        };
        for (const set &c : sets) {
            const int K = c.K;
            slot s = make_slot(16, K);
            std::vector<weight> w;
            std::vector<dev> d(c.n);
            for (int j = 0; j < c.n; j++) {
                w.push_back(make_weight(K, c.N[j], c.k2[j]));
                upload(w[j], s, 16, d[j]);
            }
            ds4_exl3_dense_proj p[4] = {};
            size_t yb[4] = {};
            float *ym = nullptr;
            size_t ytot = 0;
            for (int j = 0; j < c.n; j++) { yb[j] = ytot; ytot += (size_t)16 * c.N[j]; }
            CUDA_OK(cudaMalloc((void **)&ym, ytot * sizeof(float)));
            for (int j = 0; j < c.n; j++) p[j] = ds4_exl3_dense_proj{d[j].w, c.k2[j], c.N[j], ym + yb[j]};
            const size_t mws = ds4_exl3_dense_multi_workspace_bytes(p, c.n, 16, K);
            size_t sum = 0;
            for (int j = 0; j < c.n; j++) sum += ds4_exl3_dense_workspace_bytes(16, K, c.N[j]);
            CHECK(mws == sum, "%s: multi workspace %zu != the singles' sum %zu", c.what, mws, sum);
            void *mw = nullptr;
            CUDA_OK(cudaMalloc(&mw, mws));
            size_t diff = 0, checked = 0;
            for (int M : {1, 2, 3, 5, 8, 16}) {
                for (bool prompt : {false, true}) {
                    if (prompt && M != 5 && M != 16) continue;
                    CUDA_OK(cudaMemset(ym, 0xff, ytot * sizeof(float)));
                    const int rc = ds4_exl3_dense_multi_launch(p, c.n, d[0].q, M, K, mw, mws, 0, prompt);
                    CHECK(rc == 0, "%s: multi launch refused at M=%d (rc=%d)", c.what, M, rc);
                    CUDA_OK(cudaDeviceSynchronize());
                    for (int j = 0; j < c.n; j++) {
                        std::vector<float> one, many((size_t)M * c.N[j]);
                        CUDA_OK(cudaMemcpy(many.data(), ym + yb[j], many.size() * sizeof(float), cudaMemcpyDeviceToHost));
                        CHECK(run(d[j], c.k2[j], M, K, c.N[j], one, prompt) == 0, "%s: single launch refused", c.what);
                        diff += std::memcmp(one.data(), many.data(), many.size() * sizeof(float)) != 0;
                        checked++;
                    }
                }
            }
            CHECK(diff == 0, "%s: %zu of %zu (width, projection) outputs differ from the single launch", c.what, diff, checked);
            CHECK(ds4_exl3_dense_multi_launch(p, c.n, d[0].q, 16, K, mw, mws - 16, 0) == -1, "%s: short workspace accepted",
                  c.what);
            printf("  multi %-44s %zu (width, projection) outputs bit-identical to their single launches\n", c.what,
                   checked);
            cudaFree(mw);
            cudaFree(ym);
            for (dev &x : d) release(x);
        }
        ds4_exl3_dense_proj five[5] = {};
        CHECK(ds4_exl3_dense_multi_launch(five, 5, (void *)16, 1, 2560, (void *)16, 1 << 20, 0) == -1, "five projections accepted");
        printf("  multi refusals: a short workspace, five projections -- refused\n");
    }
    /* 7. L288 qwen-exl3-head: the heads through this arm -- Qwen's lm_head 2560 -> 248320 (1940 128-output blocks,
     * past the epilogue's former 1024 counters) and the MTP draft head 2560 -> 82304, at K = 3 / 4 / 5 (and the draft
     * head at K = 3 / 4).  Too wide for the double reference over every output, so: (a) the head's rows at M = 1, 5,
     * 9 are bit for bit its M = 16 rows; (b) three 2560-output windows (the first, one inside, the last) cut from the
     * head's bytes (exl3_expert_cut) and run alone are bit for bit the head's columns -- the split count is K's alone
     * at both widths -- and are graded against the double reference at the section-1 limit. */
    {
        struct head { int N, k2; const char *what; };
        const head heads[] = {{248320, 6, "lm_head"}, {248320, 8, "lm_head"}, {248320, 10, "lm_head"},
                              {82304, 6, "MTP draft head"}, {82304, 8, "MTP draft head"}};
        const int K = 2560, W = 2560;
        const slot s = make_slot(16, K);
        for (const head &hd : heads) {
            const int N = hd.N, k2 = hd.k2;
            CHECK(ds4_exl3_dense_splits(K, N) == ds4_exl3_dense_splits(K, W), "%s: the window's split count is not the "
                  "head's -- the window comparison below would not be bit-exact by construction", hd.what);
            const weight w = make_weight(K, N, k2);
            dev d;
            upload(w, s, 16, d);
            std::vector<float> y16, y;
            int rc = run(d, k2, 16, K, N, y16);
            CHECK(rc == 0, "%s 2560 -> %d K=%d: launch rc=%d", hd.what, N, k2 / 2, rc);
            if (rc) { release(d); continue; }
            size_t nonfinite = 0;
            for (float v : y16) nonfinite += !std::isfinite(v);
            CHECK(nonfinite == 0, "%s K=%d: %zu non-finite outputs (an unwritten NaN canary)", hd.what, k2 / 2, nonfinite);
            size_t diff_m = 0;
            for (int M : {1, 5, 9}) {
                rc = run(d, k2, M, K, N, y);
                CHECK(rc == 0, "%s K=%d M=%d: launch rc=%d", hd.what, k2 / 2, M, rc);
                if (!rc) diff_m += memcmp(y.data(), y16.data(), y.size() * sizeof(float)) != 0;
            }
            CHECK(diff_m == 0, "%s K=%d: %zu of 3 widths (M=1,5,9) differ from the same rows at M=16", hd.what, k2 / 2, diff_m);
            double wmx = 0, wmed = 0;
            size_t wdiff = 0;
            for (const int lo : {0, (N / 2 / 128) * 128, N - W}) {
                exl3_cut c;
                weight sub;
                sub.K = K; sub.N = W; sub.k2 = k2;
                if (!exl3_expert_cut(K, N, k2, false, lo, lo + W, 0, W, &c) ||
                    !exl3_expert_layout(K, W, k2, &sub.trellis, &sub.scales, &sub.stride)) {
                    CHECK(false, "%s K=%d: the window [%d, %d) does not cut", hd.what, k2 / 2, lo, lo + W);
                    continue;
                }
                sub.bytes.resize(sub.stride);
                exl3_expert_cut_apply(&c, w.bytes.data(), sub.bytes.data());
                std::vector<double> what, yref;
                exl3t_dequant(sub.bytes.data(), K, W, k2, what);
                exl3t_reference(sub.bytes.data(), what, K, W, k2, s.x.data(), 16, yref);
                dev ds;
                upload(sub, s, 16, ds);
                std::vector<float> ys;
                rc = run(ds, k2, 16, K, W, ys);
                CHECK(rc == 0, "%s K=%d window %d: launch rc=%d", hd.what, k2 / 2, lo, rc);
                if (!rc) {
                    double mx, med;
                    grade(ys, yref, 16, W, 0, &mx, &med);
                    wmx = fmax(wmx, mx);
                    wmed = fmax(wmed, med);
                    for (int r = 0; r < 16; r++)
                        wdiff += memcmp(ys.data() + (size_t)r * W, y16.data() + (size_t)r * N + lo, (size_t)W * sizeof(float)) != 0;
                }
                release(ds);
            }
            CHECK(wmx < tol, "%s 2560 -> %d K=%d: a window's max rel %.3e (limit %.0e)", hd.what, N, k2 / 2, wmx, tol);
            CHECK(wmed > 1e-10, "%s K=%d: window median rel %.3e -- a degenerate fixture, not a pass", hd.what, k2 / 2, wmed);
            CHECK(wdiff == 0, "%s K=%d: %zu of 48 window rows differ from the head's columns", hd.what, k2 / 2, wdiff);
            printf("  head %-15s 2560 -> %-6d K=%d splits %d (%d blocks): M=1,5,9 %s; 3 windows of %d cut from it == "
                   "its columns %s, vs double max rel %.2e (median <= %.2e)\n", hd.what, N, k2 / 2,
                   ds4_exl3_dense_splits(K, N), N / 128, diff_m ? "DIFFER" : "bit-identical to M=16's rows", W,
                   wdiff ? "DIFFER" : "bit for bit", wmx, wmed);
            /* timed (information, not graded): a call at 1 / 4 / 16 rows, GB/s over the weight's bytes */
            cudaEvent_t e0, e1;
            CUDA_OK(cudaEventCreate(&e0));
            CUDA_OK(cudaEventCreate(&e1));
            printf("    time:");
            for (const int M : {1, 4, 16}) {
                for (int i = 0; i < 3; i++) (void)ds4_exl3_dense_launch(d.w, k2, d.q, d.y, M, K, N, d.ws, d.ws_bytes, 0);
                CUDA_OK(cudaEventRecord(e0));
                for (int i = 0; i < 20; i++) (void)ds4_exl3_dense_launch(d.w, k2, d.q, d.y, M, K, N, d.ws, d.ws_bytes, 0);
                CUDA_OK(cudaEventRecord(e1));
                CUDA_OK(cudaEventSynchronize(e1));
                float ms = 0;
                CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
                printf("  %2d rows %.3f ms (%.0f GB/s)", M, ms / 20, (double)w.stride / (ms / 20 * 1e-3) / 1e9);
            }
            printf("\n");
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            release(d);
        }
    }
    printf(g_fail ? "EXL3-DENSE GATE FAIL\n" : "EXL3-DENSE GATE PASS\n");
    return g_fail;
}

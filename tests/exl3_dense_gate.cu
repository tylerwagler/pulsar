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
 *      16 (M = 17, 32 and the shape's widest: 16-row passes in one launch up to 32 rows, L341, and 32-row launches
 *      in turn past that) -- every 16-row block bit for bit a launch of those rows alone, and graded against the
 *      double reference;
 *   3. prefill: prompt chunks of M = 40 / 97 / 300 at several shapes -- the prefill GEMM (one fp16 operand plane):
 *      every output within the fp16-operand envelope (exl3t_fp16_operand_bound) plus f32 order;
 *   4. mutations: one flipped trellis bit must move outputs, and a sign-flipped
 *      suh block must push the error far past the tolerance;
 *   5. refusals: K % 128, an uninstantiated rate, a short workspace, M = 0.
 *   6. the multi launch (L288 exl3-dense): each output bit for bit its single launch; L296 dec2: the decode
 *      epilogue's bf16 outputs are the f32 outputs rounded to nearest even, byte for byte (a prompt chunk refuses them).
 *   7. the heads (L288 qwen-exl3-head): lm_head 2560 -> 248320 and the draft head 2560 -> 82304 at K = 3..5 --
 *      M-independent rows, and windows cut from the head equal its columns bit for bit and grade vs double.
 *   8. a logical width short of the stored 128 multiple (L295 F7: Inkling's head 4096 -> 201024, stored 201088):
 *      bit for bit the stored call's columns, nothing written past N, graded vs double.
 *   9. the GEMV's two CTA forms (L380: 16 warps, one CTA an SM; 8 warps, two an SM): (a) the selection rule
 *      (ds4_exl3_dense_gemv_arm) on a table of (rows, grid, SMs, device) -- the measured GB10 launches on their side
 *      of it, one wave, five waves, M > 8, two passes and every other architecture on the 16-warp side; (b) each
 *      form FORCED (ds4_exl3_dense_arm_force) on the same launches -- every rate, the narrow and the wide CTA
 *      (shapes sized from the device's SM count so gemv_wide picks each; the description proves which kernel ran),
 *      K of 1, 3, 5 and 20 blocks, M = 1, 2, 3, 4, 8, 16, single and multi launches (one rate and mixed), f32 and
 *      bf16 outputs: the 8-warp form's bytes are the 16-warp form's, every f32 row its M = 16 row, every bf16 row
 *      the f32 row rounded, the single shapes graded vs double; any differing byte fails; (c) a mutated weight
 *      moves the 8-warp output; the forced 8-warp form refuses a two-pass launch.  Run on every device: the forced
 *      8-warp form runs on sm_75 too (the rule never picks it there), so its bytes are held on Turing as well.
 *  10. the prompt arm's two kernels (L381: the fixed-algorithm GEMM on the weight decoded to fp16; the in-place
 *      kernel on small chunks): (a) the selection rule (ds4_exl3_dense_prompt_arm) on a table of (rows, device);
 *      (b) each kernel FORCED (ds4_exl3_dense_prompt_force) on the same launches -- every rate; K of 1, 3 and 20
 *      blocks (K is always whole 128-blocks: the arm refuses any other); N a 128 multiple, short of its stored
 *      multiple, and under one block; single and multi launches (one rate and mixed); the prompt launches of
 *      Qwen3.8-Flash-Next, Nemotron-3.5 Super and Gemma 4; M = 1, 2, 3, 5, 16, 17, 30, 64, 65, 128, the rule's 256
 *      and 257: EVERY output byte of the in-place kernel is the GEMM arm's, nothing is written past the rows, and
 *      every row of the in-place kernel at 256 rows is that row launched alone; (c) the rule's own path: a shape's
 *      first launch is the check (both kernels, compared), its next ones the in-place kernel, 257 rows the GEMM --
 *      the description proves which -- and every one writes the GEMM arm's bytes; (d) the check can fail: with a
 *      fault armed (ds4_exl3_dense_prompt_check_fault) it finds the mismatch, the in-place kernel turns itself off
 *      for every shape, and the launch still returns the GEMM arm's bytes; a flipped trellis bit moves the in-place
 *      output.  The equality is a property of the device's cuBLASLt kernel: (b)-(d) FAIL on any differing byte on
 *      a device the rule names (sm_12x) and are information elsewhere (the rule keeps the GEMM there).
 *
 * usage: ./tests/exl3_dense_gate --plan   section 10's fixtures walked on the host (no GPU): every layout and workspace
 *                                         it will ask for, a refused one failing here
 *        ./tests/exl3_dense_gate     (make exl3-dense-gate CUDA_ARCH=sm_120f on GB10; CUDA_ARCH=sm_75 on Turing)
 */
#include "../src/cuda/mmq/ds4_exl3_dense.cuh"
#include "../src/cuda/pulsar_cuda_arch.cuh"
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
    return __bfloat16_as_ushort(__float2bfloat16((float)v));
}
static double bf16_back(uint16_t u) {
    return (double)__bfloat162float(__ushort_as_bfloat16(u));
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

/* 3. a prompt chunk through the prefill GEMM (qwen_exl3_dense_prefill.cu), graded against the double reference;
 * y16 = the decode GEMV's M = 16 rows to compare rows 0..15 with. */
static void grade_prefill(const dev &d, const weight &w, const slot &s, const std::vector<double> &what,
                          const std::vector<double> &yref, int k2, int max_rows, int K, int N, const char *what_s,
                          double tol, const std::vector<float> &y16) {
    std::vector<float> y;
    double mx, med;
    int rc;
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
              what_s, k2 / 2, diff_p, max_rows);
    }
}

/* 9. L380: the GEMV's two CTA forms.  (a) the rule's table; (b) both forms forced on the same launches, byte for byte;
 * (c) a mutation and the forced form's refusal. */
static void arm_section(const double tol) {
    const int W16 = DS4_EXL3_DENSE_ARM_16W, W8 = DS4_EXL3_DENSE_ARM_8W;
    /* (a) the rule: rows, the launched grid, the device's SMs and compute capability -> the form */
    struct rule { int M, ctas, sms, cc, want; const char *what; };
    const rule rules[] = {
        {1, 512, 48, 121, W8, "GB10 2560->10240+6144 narrow, 11 waves, 1 row (measured 0.85)"},
        {4, 512, 48, 121, W8, "the same at 4 rows (0.83)"},
        {8, 256, 48, 121, W8, "the same launch wide at 8 rows, 6 waves (0.93)"},
        {1, 416, 48, 121, W8, "GB10 2560->12288+2x512 narrow, 9 waves (0.88)"},
        {8, 208, 48, 121, W16, "the same launch wide at 8 rows, 5 waves (measured 1.07: slower)"},
        {1, 320, 48, 121, W8, "GB10 2560->10240 narrow, 7 waves (0.85)"},
        {8, 160, 48, 121, W16, "the same wide at 8 rows, 4 waves (0.97: inside the noise)"},
        {1, 3880, 48, 121, W8, "GB10 lm_head wide, 81 waves, 1 row (0.90)"},
        {8, 3880, 48, 121, W8, "GB10 lm_head wide at 8 rows (0.91)"},
        {1, 40, 48, 121, W16, "GB10 one wave of 40 CTAs, 1 row (1.05-1.50: slower)"},
        {8, 40, 48, 121, W16, "GB10 one wave of 40 CTAs, 8 rows"},
        {1, 240, 48, 121, W16, "5 full waves: the last grid that keeps 16 warps"},
        {1, 241, 48, 121, W8, "the first CTA of a 6th wave"},
        {9, 3880, 48, 121, W16, "9 rows: past the measured widths"},
        {16, 512, 48, 121, W16, "16 rows"},
        {17, 3880, 48, 121, W16, "two passes (17 rows)"},
        {32, 3880, 48, 121, W16, "two passes (32 rows)"},
        {1, 512, 48, 120, W8, "sm_120: the same family as GB10's sm_121"},
        {1, 512, 170, 120, W16, "sm_120 with 170 SMs: the same grid is 4 waves there"},
        {1, 1024, 170, 120, W8, "sm_120 with 170 SMs: 7 waves"},
        {1, 512, 48, 75, W16, "sm_75 (Turing, 48 SMs): not measured"},
        {1, 3880, 72, 75, W16, "sm_75 (Turing, 72 SMs): not measured"},
        {1, 3880, 48, 89, W16, "sm_89: not measured"},
        {1, 3880, 48, 100, W16, "sm_100: not measured"},
        {1, 3880, 48, 0, W16, "a device whose capability is unknown"},
        {0, 3880, 48, 121, W16, "no rows"},
        {1, 3880, 0, 121, W16, "no SM count"},
    };
    size_t rule_bad = 0;
    for (const rule &r : rules) {
        const int got = ds4_exl3_dense_gemv_arm(r.M, r.ctas, r.sms, r.cc);
        CHECK(got == r.want, "arm rule: M=%d grid=%d SMs=%d sm_%d (%s) -> %s, want %s", r.M, r.ctas, r.sms, r.cc, r.what,
              got == W8 ? "8 warps" : "16 warps", r.want == W8 ? "8 warps" : "16 warps");
        rule_bad += got != r.want;
    }
    printf("  arm rule: %zu of %zu (rows, grid, SMs, device) cases on their side (8 warps iff sm_12x, M <= 8, 6+ waves)\n",
           sizeof rules / sizeof *rules - rule_bad, sizeof rules / sizeof *rules);

    ds4_exl3_dense_gemv_desc dv{};
    if (ds4_exl3_dense_gemv_describe(10, 1, 20, &dv)) {
        CHECK(false, "arm forms: the device does not describe a GEMV launch -- section 9 not run");
        return;
    }
    const int sms = dv.sms, cc = pulsar_cuda_device_cc();
    printf("  this device: sm_%d, %d SMs; unforced, a 512-CTA launch runs %d warps at 1 row and %d at 16\n", cc, sms,
           ds4_exl3_dense_gemv_arm(1, 512, sms, cc) == W8 ? 8 : 16, ds4_exl3_dense_gemv_arm(16, 256, sms, cc) == W8 ? 8 : 16);
    CHECK(ds4_exl3_dense_arm_force(7) == -1, "arm forms: an unknown forced form accepted");

    /* (b) the launches.  Narrow: at most one wave of 32-output CTAs (gemv_wide keeps 32 outputs at every M); wide:
     * 4 x SMs of them (two waves of 64-output CTAs for four: gemv_wide takes 64 outputs at every M). */
    struct lcase { const char *what; int K, n, N[4], k2[4], nt; };
    std::vector<lcase> cases;
    const int rates[] = {4, 6, 8, 10, 12, 16};
    const int wide_n = 128 * sms;
    for (int k2 : rates) {
        cases.push_back({"narrow, K of 1 block", 128, 1, {640, 0, 0, 0}, {k2, 0, 0, 0}, 2});
        cases.push_back({"narrow, K of 3 blocks", 384, 1, {640, 0, 0, 0}, {k2, 0, 0, 0}, 2});
        cases.push_back({"narrow, K of 5 blocks", 640, 1, {640, 0, 0, 0}, {k2, 0, 0, 0}, 2});
        cases.push_back({"narrow, K of 20 blocks", 2560, 1, {640, 0, 0, 0}, {k2, 0, 0, 0}, 2});
        cases.push_back({"wide", 2560, 1, {wide_n, 0, 0, 0}, {k2, 0, 0, 0}, 4});
        cases.push_back({"wide, K of 5 blocks", 640, 1, {wide_n, 0, 0, 0}, {k2, 0, 0, 0}, 4});
        cases.push_back({"narrow multi", 2560, 3, {256, 128, 256, 0}, {k2, k2, k2, 0}, 2});
        cases.push_back({"wide multi", 2560, 3, {wide_n - 1024, 512, 512, 0}, {k2, k2, k2, 0}, 4});
    }
    cases.push_back({"multi at rates 8 / 10 / 8 (two GEMVs)", 2560, 3, {256, 512, 128, 0}, {8, 10, 8, 0}, 2});
    cases.push_back({"wide multi at rates 10 / 6 (two GEMVs)", 2560, 2, {wide_n, wide_n, 0, 0}, {10, 6, 0, 0}, 4});

    const int Ms[] = {1, 2, 3, 4, 8, 16};
    size_t n_out = 0, n_diff = 0, n_m16 = 0, n_rn = 0, n_lane = 0, n_graded = 0;
    double g_mx = 0;
    for (const lcase &c : cases) {
        const int K = c.K, n = c.n;
        const slot s = make_slot(16, K);
        std::vector<weight> w;
        uint8_t *dw[4] = {}, *dq = nullptr;
        float *dy[4] = {};
        uint16_t *dyb[4] = {};
        void *ws = nullptr;
        ds4_exl3_dense_proj pf[4] = {}, pb[4] = {};
        CUDA_OK(cudaMalloc((void **)&dq, s.q.size() * sizeof(uint16_t)));
        CUDA_OK(cudaMemcpy(dq, s.q.data(), s.q.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        for (int j = 0; j < n; j++) {
            w.push_back(make_weight(K, c.N[j], c.k2[j]));
            CUDA_OK(cudaMalloc((void **)&dw[j], w[j].stride));
            CUDA_OK(cudaMemcpy(dw[j], w[j].bytes.data(), w[j].stride, cudaMemcpyHostToDevice));
            CUDA_OK(cudaMalloc((void **)&dy[j], (size_t)16 * c.N[j] * sizeof(float)));
            CUDA_OK(cudaMalloc((void **)&dyb[j], (size_t)16 * c.N[j] * sizeof(uint16_t)));
            pf[j] = ds4_exl3_dense_proj{dw[j], c.k2[j], c.N[j], dy[j], nullptr};
            pb[j] = ds4_exl3_dense_proj{dw[j], c.k2[j], c.N[j], nullptr, dyb[j]};
        }
        const size_t wsb = ds4_exl3_dense_multi_workspace_bytes(pf, n, 16, K);
        CUDA_OK(cudaMalloc(&ws, wsb));
        /* one launch under a forced form: each projection's M rows, f32 or bf16, canaries first (every output written) */
        auto launch = [&](int arm, int M, bool bf16, const ds4_exl3_dense_proj *p, std::vector<std::vector<uint8_t>> &out) {
            const size_t el = bf16 ? sizeof(uint16_t) : sizeof(float);
            for (int j = 0; j < n; j++)
                CUDA_OK(cudaMemset(bf16 ? (void *)dyb[j] : (void *)dy[j], 0xff, (size_t)M * c.N[j] * el));
            if (ds4_exl3_dense_arm_force(arm) != 0) return -1;
            const int rc = ds4_exl3_dense_multi_launch(p, n, dq, M, K, ws, wsb, 0, false);
            CUDA_OK(cudaDeviceSynchronize());
            ds4_exl3_dense_arm_force(0);
            if (rc) return rc;
            out.assign(n, {});
            for (int j = 0; j < n; j++) {
                out[j].resize((size_t)M * c.N[j] * el);
                CUDA_OK(cudaMemcpy(out[j].data(), bf16 ? (void *)dyb[j] : (void *)dy[j], out[j].size(), cudaMemcpyDeviceToHost));
            }
            return 0;
        };
        /* the instrument proves the lane: under each forced form, every rate's GEMV is that form's kernel at the
         * CTA width the case was sized for */
        bool lanes = true;
        for (int j = 0; j < n; j++) {
            int c32 = 0;
            for (int i = 0; i < n; i++)
                if (c.k2[i] == c.k2[j]) c32 += (int)(exl3_out_stored((uint64_t)c.N[i]) / 32);
            for (int M : Ms)
                for (int arm : {W16, W8}) {
                    ds4_exl3_dense_gemv_desc d{};
                    ds4_exl3_dense_arm_force(arm);
                    const int rc = ds4_exl3_dense_gemv_describe(c.k2[j], M, c32, &d);
                    ds4_exl3_dense_arm_force(0);
                    const bool ok = rc == 0 && d.arm == arm && d.threads == (arm == W8 ? 256 : 512) && d.nt == c.nt;
                    CHECK(ok, "arm forms: %s K=%d M=%d: forced %d warps describes rc=%d form %d, %d threads, %d n-tiles a "
                          "CTA (want %d) on %d SMs -- the case does not run the kernel it names", c.what, c.k2[j] / 2, M,
                          arm == W8 ? 8 : 16, rc, d.arm, d.threads, d.nt, c.nt, sms);
                    lanes = lanes && ok;
                    n_lane++;
                }
        }
        std::vector<std::vector<uint8_t>> ref16, y, yb16, yb8;
        int rc = launch(W16, 16, false, pf, ref16);
        CHECK(rc == 0, "arm forms: %s K=%d: the 16-warp M=16 launch rc=%d", c.what, c.k2[0] / 2, rc);
        for (int M : Ms) {
            if (rc || !lanes) break;
            for (int arm : {W16, W8}) {
                int r2 = launch(arm, M, false, pf, y);
                CHECK(r2 == 0, "arm forms: %s K=%d M=%d: the %d-warp f32 launch rc=%d", c.what, c.k2[0] / 2, M,
                      arm == W8 ? 8 : 16, r2);
                if (r2) { n_diff++; continue; }
                for (int j = 0; j < n; j++) {
                    const bool same = memcmp(y[j].data(), ref16[j].data(), y[j].size()) == 0;
                    CHECK(same, "arm forms: %s K=%d M=%d projection %d (%d -> %d): the %d-warp CTA's f32 rows are not the "
                          "16-warp CTA's M=16 rows", c.what, c.k2[j] / 2, M, j, K, c.N[j], arm == W8 ? 8 : 16);
                    (arm == W8 ? n_diff : n_m16) += !same;
                    n_out += arm == W8;
                }
                std::vector<std::vector<uint8_t>> &ybx = arm == W8 ? yb8 : yb16;
                r2 = launch(arm, M, true, pb, ybx);
                CHECK(r2 == 0, "arm forms: %s K=%d M=%d: the %d-warp bf16 launch rc=%d", c.what, c.k2[0] / 2, M,
                      arm == W8 ? 8 : 16, r2);
                if (r2) { n_diff++; continue; }
                for (int j = 0; j < n; j++) {   /* the bf16 rows are the f32 rows rounded to nearest even */
                    size_t bad = 0;
                    const uint16_t *b = (const uint16_t *)ybx[j].data();
                    for (size_t i = 0; i < ybx[j].size() / 2; i++) {
                        uint32_t u;
                        memcpy(&u, ref16[j].data() + i * 4, 4);
                        u += 0x7fffu + ((u >> 16) & 1u);
                        bad += (uint16_t)(u >> 16) != b[i];
                    }
                    CHECK(bad == 0, "arm forms: %s K=%d M=%d projection %d: %zu of the %d-warp CTA's bf16 outputs are not "
                          "the f32 outputs rounded", c.what, c.k2[j] / 2, M, j, bad, arm == W8 ? 8 : 16);
                    n_rn += bad != 0;
                }
            }
            for (int j = 0; j < n && yb8.size() == (size_t)n && yb16.size() == (size_t)n; j++) {
                const bool same = yb8[j].size() == yb16[j].size() && memcmp(yb8[j].data(), yb16[j].data(), yb8[j].size()) == 0;
                CHECK(same, "arm forms: %s K=%d M=%d projection %d (%d -> %d): the 8-warp CTA's bf16 rows are not the "
                      "16-warp CTA's", c.what, c.k2[j] / 2, M, j, K, c.N[j]);
                n_diff += !same;
                n_out++;
            }
        }
        if (!rc && lanes && n == 1) {   /* the 8-warp form against the host authority in double */
            std::vector<double> what, yref;
            exl3t_dequant(w[0].bytes.data(), K, c.N[0], c.k2[0], what);
            exl3t_reference(w[0].bytes.data(), what, K, c.N[0], c.k2[0], s.x.data(), 16, yref);
            if (launch(W8, 16, false, pf, y) == 0) {
                std::vector<float> yf((size_t)16 * c.N[0]);
                memcpy(yf.data(), y[0].data(), y[0].size());
                double mx, med;
                grade(yf, yref, 16, c.N[0], 0, &mx, &med);
                CHECK(mx < tol, "arm forms: %s %d -> %d K=%d: the 8-warp CTA's M=16 max rel %.3e (limit %.0e)", c.what, K,
                      c.N[0], c.k2[0] / 2, mx, tol);
                CHECK(med > 1e-10, "arm forms: %s K=%d: median rel %.3e -- a degenerate fixture, not a pass", c.what,
                      c.k2[0] / 2, med);
                g_mx = fmax(g_mx, mx);
                n_graded++;
            }
        }
        if (!rc && lanes && n == 1 && c.nt == 4 && c.k2[0] == 10 && K == 2560) {
            /* (c) the comparison can fail: one flipped trellis bit moves the 8-warp form's output */
            weight wm = w[0];
            wm.words()[12345] ^= 0x0100u;
            CUDA_OK(cudaMemcpy(dw[0], wm.bytes.data(), wm.stride, cudaMemcpyHostToDevice));
            const int r2 = launch(W8, 16, false, pf, y);
            size_t moved = 0;
            if (!r2)
                for (size_t i = 0; i < y[0].size(); i += 4) moved += memcmp(y[0].data() + i, ref16[0].data() + i, 4) != 0;
            CHECK(r2 == 0 && moved > 0, "arm forms: a flipped trellis bit moved no 8-warp output -- the comparison is "
                  "degenerate");
            CUDA_OK(cudaMemcpy(dw[0], w[0].bytes.data(), w[0].stride, cudaMemcpyHostToDevice));
            /* the 8-warp form is one pass: forced on 17 rows it refuses, in the launch and in the description */
            ds4_exl3_dense_gemv_desc d{};
            ds4_exl3_dense_arm_force(W8);
            const int r17 = ds4_exl3_dense_multi_launch(pf, 1, dq, 17, K, ws, wsb, 0, false);
            const int d17 = ds4_exl3_dense_gemv_describe(10, 17, c.N[0] / 32, &d);
            ds4_exl3_dense_arm_force(0);
            CHECK(r17 == -1 && d17 == -1, "arm forms: the forced 8-warp CTA took a two-pass launch (rc=%d, describe %d)", r17,
                  d17);
            printf("  arm forms: a flipped trellis bit moved %zu 8-warp outputs; the forced 8-warp CTA refuses 17 rows\n", moved);
        }
        for (int j = 0; j < n; j++) { cudaFree(dw[j]); cudaFree(dy[j]); cudaFree(dyb[j]); }
        cudaFree(dq);
        cudaFree(ws);
    }
    ds4_exl3_dense_gemv_desc d8{};
    ds4_exl3_dense_arm_force(W8);
    const int rd = ds4_exl3_dense_gemv_describe(10, 8, 4 * sms, &d8);
    ds4_exl3_dense_arm_force(0);
    CHECK(n_out > 0 && n_graded > 0, "arm forms: nothing was compared");
    printf("  arm forms: %zu launches (K = 2, 3, 4, 5, 6, 8; narrow and wide; M = 1, 2, 3, 4, 8, 16; single and multi; f32 "
           "and bf16), %zu (launch, projection) outputs of the 8-warp CTA: %zu differ from the 16-warp CTA's bytes; %zu "
           "16-warp f32 outputs off their M=16 rows; %zu bf16 outputs off the f32 rounded; %zu kernel descriptions "
           "checked; %zu single shapes vs double, max rel %.2e\n", cases.size() * 24, n_out, n_diff, n_m16, n_rn, n_lane,
           n_graded, g_mx);
    if (!rd)
        printf("  arm forms: the 8-warp K=5 wide CTA here: %d registers a thread, %d B shared, %d CTAs an SM at 8 rows "
               "(information)\n", d8.regs, d8.smem, d8.ctas_per_sm);
}

/* 10. the prompt arm's launches: N is each projection's LOGICAL width (stored at exl3_out_stored, a 128 multiple). */
struct pcase { const char *what; int K, n, N[4], k2[4]; bool full; };
static std::vector<pcase> prompt_cases(void) {
    std::vector<pcase> cases;
    for (int k2 : {4, 6, 8, 10, 12, 16}) {
        cases.push_back({"K of 1 block", 128, 1, {640, 0, 0, 0}, {k2, 0, 0, 0}, true});
        cases.push_back({"K of 3 blocks", 384, 1, {640, 0, 0, 0}, {k2, 0, 0, 0}, true});
        cases.push_back({"K of 20 blocks", 2560, 1, {640, 0, 0, 0}, {k2, 0, 0, 0}, true});
        cases.push_back({"N short of its stored 1024", 640, 1, {1000, 0, 0, 0}, {k2, 0, 0, 0}, true});
        cases.push_back({"N under one block (stored 128)", 640, 1, {100, 0, 0, 0}, {k2, 0, 0, 0}, true});
        cases.push_back({"multi", 1280, 3, {256, 1000, 128, 0}, {k2, k2, k2, 0}, true});
    }
    cases.push_back({"multi at rates 4 / 5 / 6 / 2", 1280, 4, {256, 512, 100, 640}, {8, 10, 12, 4}, true});
    /* the families' prompt launches (family_qwen_s4.cpp, nemotron_h_forward.cpp, gemma4_forward.cpp), at 5 rows and
     * the rule's last */
    cases.push_back({"Qwen GDN qkv + z", 2560, 2, {10240, 6144, 0, 0}, {10, 10, 0, 0}, false});
    cases.push_back({"Qwen QSA q + k + v", 2560, 3, {12288, 512, 512, 0}, {10, 10, 10, 0}, false});
    cases.push_back({"Qwen QSA q + k + v + index_qk (K8 / K6 container)", 2560, 4, {12288, 512, 512, 640}, {16, 16, 16, 12}, false});
    cases.push_back({"Qwen GDN out / QSA o", 6144, 1, {2560, 0, 0, 0}, {10, 0, 0, 0}, false});
    cases.push_back({"Qwen shared gate + up", 2560, 2, {640, 640, 0, 0}, {10, 10, 0, 0}, false});
    cases.push_back({"Qwen shared down", 640, 1, {2560, 0, 0, 0}, {10, 0, 0, 0}, false});
    cases.push_back({"Qwen PLE key", 2560, 1, {10240, 0, 0, 0}, {10, 0, 0, 0}, false});
    cases.push_back({"Qwen PLE value", 2560, 1, {2560, 0, 0, 0}, {10, 0, 0, 0}, false});
    cases.push_back({"Nemotron Mamba in_proj", 4096, 1, {18560, 0, 0, 0}, {8, 0, 0, 0}, false});
    cases.push_back({"Nemotron Mamba out_proj / MTP eh_proj", 8192, 1, {4096, 0, 0, 0}, {8, 0, 0, 0}, false});
    cases.push_back({"Nemotron attention q + k + v", 4096, 3, {4096, 256, 256, 0}, {8, 8, 8, 0}, false});
    cases.push_back({"Nemotron latent in", 4096, 1, {1024, 0, 0, 0}, {8, 0, 0, 0}, false});
    cases.push_back({"Nemotron latent out", 1024, 1, {4096, 0, 0, 0}, {8, 0, 0, 0}, false});
    cases.push_back({"Nemotron shared up", 4096, 1, {5376, 0, 0, 0}, {8, 0, 0, 0}, false});
    cases.push_back({"Nemotron shared down", 5376, 1, {4096, 0, 0, 0}, {8, 0, 0, 0}, false});
    cases.push_back({"Gemma window q + k + v", 2816, 3, {4096, 2048, 2048, 0}, {8, 8, 8, 0}, false});
    cases.push_back({"Gemma full q + k", 2816, 2, {8192, 1024, 0, 0}, {8, 8, 0, 0}, false});
    cases.push_back({"Gemma full o_proj", 8192, 1, {2816, 0, 0, 0}, {8, 0, 0, 0}, false});
    cases.push_back({"Gemma MLP gate / up", 2816, 1, {2176, 0, 0, 0}, {8, 0, 0, 0}, false});
    cases.push_back({"Gemma MLP down", 2176, 1, {2816, 0, 0, 0}, {8, 0, 0, 0}, false});

    return cases;
}

/* One projection's weight for a prompt case: the [trellis | suh | svh] slice at the STORED width, svh 0 on the pad
 * columns past the logical N (exllamav3's pad; section 8's construction).  false, said by name, when the layout is
 * refused -- the case fails, the section goes on. */
static bool prompt_weight(const pcase &c, int j, weight &w) {
    const int K = c.K, N = c.N[j], k2 = c.k2[j], Ns = (int)exl3_out_stored((uint64_t)N);
    uint64_t lt = 0, lsc = 0, lst = 0, et = 0, esc = 0, est = 0;
    const bool ok = N > 0 && N % 4 == 0 && exl3_linear_layout(K, N, k2, &lt, &lsc, &lst) &&
                    exl3_expert_layout(K, Ns, k2, &et, &esc, &est) && est == lst && et == lt;
    CHECK(ok, "prompt kernels: %s projection %d: %d -> %d (stored %d) at k2=%d is not an EXL3 dense layout -- the case "
          "is not run", c.what, j, K, N, Ns, k2);
    if (!ok) return false;
    w = make_weight(K, Ns, k2);
    uint16_t *svp = (uint16_t *)(w.bytes.data() + w.trellis) + K;
    for (int n = N; n < Ns; n++) svp[n] = 0;
    return true;
}

/* `--plan`: section 10's fixtures walked on the host, no GPU -- every case's layouts and workspace are what the
 * section will ask for, so a refused fixture shows here (and on any build host) instead of on the device. */
static int prompt_plan(void) {
    const std::vector<pcase> cases = prompt_cases();
    const int R = DS4_EXL3_DENSE_PROMPT_INPLACE_ROWS + 1;
    size_t n_proj = 0, n_pad = 0, n_tail_only = 0;
    for (const pcase &c : cases) {
        ds4_exl3_dense_proj pf[4] = {};
        bool ok = c.n >= 1 && c.n <= 4 && c.K > 0 && c.K % 128 == 0;
        CHECK(ok, "plan: %s: %d projections of K=%d", c.what, c.n, c.K);
        printf("  plan %-52s K=%-5d %s:", c.what, c.K, c.full ? "M 1..257" : "M 5, 256");
        for (int j = 0; j < c.n && ok; j++) {
            const int N = c.N[j], Ns = (int)exl3_out_stored((uint64_t)N);
            weight w;   /* the section's own construction, on the host */
            const bool lay = prompt_weight(c, j, w) && ds4_exl3_dense_rate_supported(c.k2[j]);
            CHECK(lay, "plan: %s projection %d: %d -> %d (stored %d) at k2=%d is refused", c.what, j, c.K, N, Ns, c.k2[j]);
            ok = ok && lay;
            pf[j] = ds4_exl3_dense_proj{nullptr, c.k2[j], N, nullptr, nullptr};
            printf(" %d%s@K%.1f (%.1f MB)", N, N == Ns ? "" : N < 128 ? " [tail only]" : " [padded]", c.k2[j] / 2.0,
                   lay ? w.stride / 1e6 : 0.0);
            n_proj++;
            n_pad += N != Ns;
            n_tail_only += N < 128;
        }
        const size_t wsb = ok ? ds4_exl3_dense_multi_workspace_bytes(pf, c.n, R, c.K) : 0;
        CHECK(!ok || wsb > 0, "plan: %s: the arm refuses the set's workspace", c.what);
        printf("  workspace %zu B%s\n", wsb, ok && wsb ? "" : "  *** REFUSED ***");
    }
    {   /* (d)'s two shapes */
        const weight wa = make_weight(640, 640, 10), wb = make_weight(640, 256, 10);   /* exit(1) if refused */
        CHECK(wa.stride > 0 && wb.stride > 0, "plan: the check leg's weights are empty");
    }
    CHECK(n_pad > 0 && n_tail_only > 0, "plan: no case has a logical width short of its stored one, or none under a block");
    printf("exl3-dense-gate --plan: %zu cases, %zu projections (%zu with a logical width short of the stored one, %zu of "
           "them under one block): %s\n", cases.size(), n_proj, n_pad, n_tail_only,
           g_fail ? "A FIXTURE IS REFUSED" : "every fixture is a layout the arm takes");
    return g_fail;
}

/* 10. L381: the prompt arm's two kernels.  (a) the rule's table; (b) both kernels forced on the same launches, byte
 * for byte, and the in-place kernel's rows one at a time; (c) the rule's path with its check; (d) the check's
 * mismatch path and a mutation. */
static void prompt_arm_section(void) {
    const int GEMM = DS4_EXL3_DENSE_PROMPT_GEMM, INP = DS4_EXL3_DENSE_PROMPT_INPLACE, CHK = DS4_EXL3_DENSE_PROMPT_CHECK;
    const int T = DS4_EXL3_DENSE_PROMPT_INPLACE_ROWS;
    auto arm_name = [&](int a) { return a == GEMM ? "the GEMM" : a == INP ? "in place" : a == CHK ? "the check" : "?"; };
    /* (a) the rule: rows and the device's compute capability -> the kernel */
    struct rule { int M, cc, want; const char *what; };
    const rule rules[] = {
        {1, 121, INP, "GB10, 1 row"},
        {5, 121, INP, "GB10, 5 rows (measured: Qwen 66.6 -> 17.1 ms a chunk, Nemotron 147.1 -> 22.5)"},
        {64, 121, INP, "GB10, 64 rows (64.4 -> 18.9, 141.8 -> 25.8)"},
        {T, 121, INP, "GB10, 256 rows: the last measured width (72.2 -> 33.7, 161.7 -> 67.3)"},
        {T + 1, 121, GEMM, "GB10, 257 rows: past the measured widths"},
        {4096, 121, GEMM, "GB10, a bulk chunk"},
        {1, 120, INP, "sm_120: the same family as GB10's sm_121"},
        {T, 120, INP, "sm_120, 256 rows"},
        {T + 1, 120, GEMM, "sm_120, 257 rows"},
        {5, 75, GEMM, "sm_75 (Turing): the order is not established there"},
        {5, 89, GEMM, "sm_89: not established"},
        {5, 100, GEMM, "sm_100: not established"},
        {5, 0, GEMM, "a device whose capability is unknown"},
        {0, 121, GEMM, "no rows"},
        {-3, 121, GEMM, "negative rows"},
    };
    size_t rule_bad = 0;
    for (const rule &r : rules) {
        const int got = ds4_exl3_dense_prompt_arm(r.M, r.cc);
        CHECK(got == r.want, "prompt rule: M=%d sm_%d (%s) -> %s, want %s", r.M, r.cc, r.what, arm_name(got),
              arm_name(r.want));
        rule_bad += got != r.want;
    }
    printf("  prompt rule: %zu of %zu (rows, device) cases on their side (in place iff sm_12x and 1 <= M <= %d)\n",
           sizeof rules / sizeof *rules - rule_bad, sizeof rules / sizeof *rules, T);
    CHECK(ds4_exl3_dense_prompt_force(7) == -1, "prompt kernels: an unknown forced kernel accepted");

    const int cc = pulsar_cuda_device_cc();
    const bool est = ds4_exl3_dense_prompt_arm(1, cc) == INP;   /* the rule names this device */
    ds4_exl3_dense_prompt_desc d0{};
    if (ds4_exl3_dense_prompt_describe(10, 1, 128, 128, &d0)) {
        CHECK(false, "prompt kernels: the device does not describe a prompt launch -- section 10 not run");
        return;
    }
    CHECK(!est || !d0.off, "prompt kernels: an earlier section's prompt launch FAILED the in-place kernel's check on this "
          "device (the loud line above): the in-place kernel does not write the fixed GEMM's bytes here");
    ds4_exl3_dense_prompt_check_reset();
    if (!est)
        printf("  N/A the in-place prompt kernel on sm_%d: the rule names sm_12x only, every prompt launch here is the "
               "GEMM -- the forced comparison below is information, the check's legs (c), (d) do not exist here\n", cc);

    /* (b), (c) the launches (prompt_cases) */
    const std::vector<pcase> cases = prompt_cases();
    const std::vector<int> Ms_full = {1, 2, 3, 5, 16, 17, 30, 64, 65, 128, T, T + 1}, Ms_fam = {5, T};
    const int R = T + 1;   /* rows of a case's fixture */
    size_t n_cmp = 0, n_out = 0, n_diff = 0, n_nan = 0, n_past = 0, n_row = 0, n_row_bad = 0, n_rule = 0, n_rule_bad = 0,
           n_lane = 0, n_lane_bad = 0, n_fail = 0;
    for (const pcase &c : cases) {
        const int K = c.K, n = c.n;
        std::vector<weight> w(n);
        bool built = true;
        for (int j = 0; j < n; j++) built = prompt_weight(c, j, w[j]) && built;
        if (!built) { n_fail++; continue; }   /* said by name; the other cases still run */
        const slot s = make_slot(R, K);
        uint8_t *dw[4] = {}, *dq = nullptr;
        float *dy[4] = {};
        void *ws = nullptr;
        ds4_exl3_dense_proj pf[4] = {};
        CUDA_OK(cudaMalloc((void **)&dq, s.q.size() * sizeof(uint16_t)));
        CUDA_OK(cudaMemcpy(dq, s.q.data(), s.q.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        for (int j = 0; j < n; j++) {
            CUDA_OK(cudaMalloc((void **)&dw[j], w[j].stride));
            CUDA_OK(cudaMemcpy(dw[j], w[j].bytes.data(), w[j].stride, cudaMemcpyHostToDevice));
            CUDA_OK(cudaMalloc((void **)&dy[j], (size_t)(R + 1) * c.N[j] * sizeof(float)));
            pf[j] = ds4_exl3_dense_proj{dw[j], c.k2[j], c.N[j], dy[j], nullptr};
        }
        const size_t wsb = ds4_exl3_dense_multi_workspace_bytes(pf, n, R, K);
        CHECK(wsb > 0, "prompt kernels: %s: the arm refuses the set's workspace", c.what);
        CUDA_OK(cudaMalloc(&ws, wsb ? wsb : 16));
        /* one prompt launch of rows [r0, r0 + M) under a forced kernel (0: the rule's): each projection's M rows, a
         * NaN canary under them (every output written) and in the row after them (none written past) */
        auto launch = [&](int force, int M, int r0, std::vector<std::vector<float>> &out) {
            for (int j = 0; j < n; j++) CUDA_OK(cudaMemset(dy[j], 0xff, (size_t)(M + 1) * c.N[j] * sizeof(float)));
            if (ds4_exl3_dense_prompt_force(force) != 0) return -1;
            const int rc = ds4_exl3_dense_multi_launch(pf, n, dq + (size_t)r0 * K * 2, M, K, ws, wsb, 0, true);
            CUDA_OK(cudaDeviceSynchronize());
            ds4_exl3_dense_prompt_force(0);
            if (rc) return rc;
            out.assign(n, {});
            for (int j = 0; j < n; j++) {
                std::vector<float> y((size_t)(M + 1) * c.N[j]);
                CUDA_OK(cudaMemcpy(y.data(), dy[j], y.size() * sizeof(float), cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < (size_t)M * c.N[j]; i++) n_nan += !(y[i] == y[i]);
                for (size_t i = (size_t)M * c.N[j]; i < y.size(); i++) {
                    uint32_t u;
                    memcpy(&u, &y[i], 4);
                    n_past += u != 0xffffffffu;
                }
                y.resize((size_t)M * c.N[j]);
                out[j] = std::move(y);
            }
            return 0;
        };
        /* outputs of a (launch, projection) that differ, byte for byte */
        auto differ = [](const std::vector<float> &a, const std::vector<float> &b) {
            size_t d = a.size() != b.size();
            for (size_t i = 0; i < a.size() && i < b.size(); i++) d += memcmp(&a[i], &b[i], sizeof(float)) != 0;
            return d;
        };
        const std::vector<int> &Ms = c.full ? Ms_full : Ms_fam;
        std::vector<std::vector<std::vector<float>>> g(Ms.size());   /* the GEMM arm's outputs, an M */
        std::vector<std::vector<float>> yi, yT;
        bool ok = true;
        size_t case_diff = 0;
        /* (b) each kernel forced, every output */
        for (size_t mi = 0; mi < Ms.size() && ok; mi++) {
            const int M = Ms[mi];
            int rc = launch(GEMM, M, 0, g[mi]);
            CHECK(rc == 0, "prompt kernels: %s K=%d M=%d: the forced GEMM launch rc=%d", c.what, K, M, rc);
            if (!rc) rc = launch(INP, M, 0, yi);
            CHECK(rc == 0, "prompt kernels: %s K=%d M=%d: the forced in-place launch rc=%d", c.what, K, M, rc);
            if (rc) { ok = false; n_fail++; break; }
            for (int j = 0; j < n; j++) {
                const size_t d = differ(yi[j], g[mi][j]);
                CHECK(!est || d == 0, "prompt kernels: %s M=%d projection %d (%d -> %d, K=%.1f bits): %zu of %zu outputs "
                      "of the in-place kernel are not the GEMM arm's bytes", c.what, M, j, K, c.N[j], c.k2[j] / 2.0, d,
                      yi[j].size());
                n_diff += d;
                case_diff += d;
                n_out += yi[j].size();
                n_cmp++;
            }
            if (M == T) yT = yi;
        }
        /* a row's bytes do not depend on the rows beside it: every row of the T-row in-place launch, launched alone */
        if (ok && c.full && yT.size() == (size_t)n) {
            for (int r = 0; r < T; r++) {
                if (launch(INP, 1, r, yi)) { CHECK(false, "prompt kernels: %s: the one-row in-place launch refused", c.what); n_fail++; break; }
                for (int j = 0; j < n; j++) {
                    const bool same = memcmp(yi[j].data(), yT[j].data() + (size_t)r * c.N[j], (size_t)c.N[j] * sizeof(float)) == 0;
                    CHECK(same, "prompt kernels: %s projection %d (%d -> %d, K=%.1f bits): row %d of the in-place kernel at "
                          "%d rows is not the row launched alone", c.what, j, K, c.N[j], c.k2[j] / 2.0, r, T);
                    n_row_bad += !same;
                    n_row++;
                }
            }
        }
        /* (c) the rule's own path: the description says which kernel each launch is, and each writes the GEMM arm's
         * bytes.  A shape's first launch at <= T rows is its check; from then on it runs in place; T + 1 is the GEMM. */
        if (ok) {
            ds4_exl3_dense_prompt_check_reset();
            for (size_t mi = 0; mi < Ms.size(); mi++) {
                const int M = Ms[mi];
                for (int pass = 0; pass < 2; pass++) {   /* before the launch, after it */
                    if (pass == 1) {
                        const int rc = launch(0, M, 0, yi);
                        CHECK(rc == 0, "prompt kernels: %s M=%d: the rule's launch rc=%d", c.what, M, rc);
                        if (rc) { n_fail++; break; }
                        for (int j = 0; j < n; j++) {
                            const size_t d = differ(yi[j], g[mi][j]);
                            CHECK(d == 0, "prompt kernels: %s M=%d projection %d (%d -> %d): %zu outputs of the rule's "
                                  "launch are not the GEMM arm's bytes", c.what, M, j, K, c.N[j], d);
                            n_rule_bad += d != 0;
                            n_rule++;
                        }
                    }
                    for (int j = 0; j < n; j++) {
                        ds4_exl3_dense_prompt_desc d{};
                        const int rd = ds4_exl3_dense_prompt_describe(c.k2[j], M, K, c.N[j], &d);
                        /* a device the rule names with every byte equal: the check before the shape's first small
                         * launch, in place after it; anything else: the GEMM */
                        const bool small = est && M <= T && !d.off;
                        const int want = !small ? GEMM : pass == 0 && mi == 0 ? CHK : INP;
                        const bool good = rd == 0 && d.arm == want && (!est || case_diff != 0 || !d.off) &&
                                          (want != INP || d.checked);
                        CHECK(good, "prompt kernels: %s M=%d projection %d (%d -> %d) %s its launch: described rc=%d as %s "
                              "(checked %d, off %d), want %s", c.what, M, j, K, c.N[j], pass ? "after" : "before", rd,
                              arm_name(d.arm), d.checked, d.off, arm_name(want));
                        n_lane_bad += !good;
                        n_lane++;
                    }
                }
            }
        }
        for (int j = 0; j < n; j++) { cudaFree(dw[j]); cudaFree(dy[j]); }
        cudaFree(dq);
        cudaFree(ws);
    }
    CHECK(n_nan == 0 && n_past == 0, "prompt kernels: %zu outputs left unwritten (or NaN), %zu written past a launch's rows",
          n_nan, n_past);
    CHECK(n_cmp > 0 && n_row > 0 && n_fail == 0, "prompt kernels: nothing was compared, or a launch refused");
    printf("  prompt kernels: %zu launches (K = 2, 3, 4, 5, 6, 8 bits; K of 1, 3, 20 blocks; N whole, short of its stored "
           "multiple and under a block; single and multi; the Qwen, Nemotron and Gemma prompt launches; M = 1 .. %d), %zu "
           "(launch, projection) outputs = %zu f32: %zu of the in-place kernel's differ from the GEMM arm's bytes%s; %zu "
           "rows launched alone, %zu off their row at %d rows; %zu outputs of the rule's own launches, %zu off the GEMM "
           "arm's; %zu descriptions, %zu not the kernel the rule names\n", cases.size(), T + 1, n_cmp, n_out, n_diff,
           est ? "" : " (information on this device)", n_row, n_row_bad, T, n_rule, n_rule_bad, n_lane, n_lane_bad);

    /* (d) the check can fail, and the comparison is not degenerate */
    {
        const int K = 640, k2 = 10, NA = 640, NB = 256, M = 5;
        const slot s = make_slot(M, K);
        const weight wa = make_weight(K, NA, k2), wb = make_weight(K, NB, k2);
        dev da, db;
        upload(wa, s, M, da);
        upload(wb, s, M, db);
        auto one = [&](const dev &d, int N, int force, std::vector<float> &y) {
            ds4_exl3_dense_prompt_force(force);
            const int rc = run(d, k2, M, K, N, y, true);
            ds4_exl3_dense_prompt_force(0);
            return rc;
        };
        std::vector<float> ga, gb, ia, y;
        CHECK(one(da, NA, GEMM, ga) == 0 && one(db, NB, GEMM, gb) == 0 && one(da, NA, INP, ia) == 0,
              "prompt check: a forced launch refused");
        /* a flipped trellis bit moves the in-place kernel's output */
        weight wm = wa;
        wm.words()[321] ^= 0x0100u;
        CUDA_OK(cudaMemcpy(da.w, wm.bytes.data(), wm.stride, cudaMemcpyHostToDevice));
        size_t moved = 0;
        if (one(da, NA, INP, y) == 0)
            for (size_t i = 0; i < y.size() && i < ia.size(); i++) moved += memcmp(&y[i], &ia[i], sizeof(float)) != 0;
        CHECK(moved > 0, "prompt check: a flipped trellis bit moved no in-place output -- the comparison is degenerate");
        CUDA_OK(cudaMemcpy(da.w, wa.bytes.data(), wa.stride, cudaMemcpyHostToDevice));
        if (est) {
            ds4_exl3_dense_prompt_desc d{};
            ds4_exl3_dense_prompt_check_reset();
            ds4_exl3_dense_prompt_check_fault(true);
            printf("  prompt check: a fault is armed -- the loud line that follows is the leg working\n");
            fflush(stdout);
            const int rc = one(da, NA, 0, y);
            ds4_exl3_dense_prompt_check_fault(false);
            CHECK(rc == 0 && y.size() == ga.size() && memcmp(y.data(), ga.data(), ga.size() * sizeof(float)) == 0,
                  "prompt check: the launch whose check failed did not return the GEMM arm's bytes (rc=%d)", rc);
            CHECK(ds4_exl3_dense_prompt_describe(k2, M, K, NA, &d) == 0 && d.off && !d.checked && d.arm == GEMM,
                  "prompt check: after a mismatch the failed shape is described as %s (off %d, checked %d), want the GEMM, "
                  "off", arm_name(d.arm), d.off, d.checked);
            CHECK(ds4_exl3_dense_prompt_describe(k2, M, K, NB, &d) == 0 && d.off && d.arm == GEMM,
                  "prompt check: after a mismatch ANOTHER shape is described as %s (off %d): the in-place kernel must be "
                  "off for every shape", arm_name(d.arm), d.off);
            const int rb = one(db, NB, 0, y);
            CHECK(rb == 0 && y.size() == gb.size() && memcmp(y.data(), gb.data(), gb.size() * sizeof(float)) == 0,
                  "prompt check: another shape's launch after the mismatch is not the GEMM arm's bytes (rc=%d)", rb);
            CHECK(ds4_exl3_dense_prompt_describe(k2, M, K, NB, &d) == 0 && !d.checked && d.arm == GEMM,
                  "prompt check: a shape was checked after the in-place kernel turned itself off");
            /* the fault gone, the check passes again: the mismatch above was the fault's */
            ds4_exl3_dense_prompt_check_reset();
            const int r3 = one(da, NA, 0, y);
            CHECK(r3 == 0 && ds4_exl3_dense_prompt_describe(k2, M, K, NA, &d) == 0 && !d.off && d.checked && d.arm == INP,
                  "prompt check: without the fault the shape did not pass its check (rc=%d, %s, off %d, checked %d)", r3,
                  arm_name(d.arm), d.off, d.checked);
            printf("  prompt check: a faulted check turned the in-place kernel off for every shape and returned the GEMM "
                   "arm's bytes; without the fault the shape passes; a flipped trellis bit moved %zu in-place outputs\n",
                   moved);
        } else {
            printf("  N/A the check's mismatch leg on sm_%d (no launch is checked here); a flipped trellis bit moved %zu "
                   "in-place outputs\n", cc, moved);
        }
        ds4_exl3_dense_prompt_check_reset();
        release(da);
        release(db);
    }
    for (int k2 : {4, 6, 8, 10, 12, 16}) {
        ds4_exl3_dense_prompt_desc d{};
        if (!ds4_exl3_dense_prompt_describe(k2, T, 2560, 10240, &d))
            printf("  prompt kernels: the in-place kernel at K=%.1f bits here: %d registers a thread, %d B shared, %d CTAs "
                   "an SM (information)\n", k2 / 2.0, d.regs, d.smem, d.ctas_per_sm);
    }
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--plan")) return prompt_plan();
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
    /* every section runs on every device the engine targets: the decode GEMV's m16n8k16 MMA and cp.async have sm_75
     * bodies since L338 (pulsar_cuda_arch.cuh; the MMA's split is a graded numerics change on Turing) */

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
            printf("  %-26s %5d -> %-5d K=%d: M=16 max rel %.2e median %.2e; M=1,2,4,8 %s\n",
                   sh.what, K, N, k2 / 2, mx, med,
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
                grade_prefill(d, w, s, what, yref, k2, max_rows, K, N, sh.what, tol, y16);
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
            for (int j = 0; j < c.n; j++) p[j] = ds4_exl3_dense_proj{d[j].w, c.k2[j], c.N[j], ym + yb[j], nullptr};
            /* L296 dec2: the same projections emitting bf16 rows (decode rows) */
            uint16_t *ybm = nullptr;
            CUDA_OK(cudaMalloc((void **)&ybm, ytot * sizeof(uint16_t)));
            ds4_exl3_dense_proj pb[4] = {};
            for (int j = 0; j < c.n; j++) pb[j] = ds4_exl3_dense_proj{d[j].w, c.k2[j], c.N[j], nullptr, ybm + yb[j]};
            size_t bdiff = 0, bchecked = 0;
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
                    if (prompt) {
                        CHECK(ds4_exl3_dense_multi_launch(pb, c.n, d[0].q, M, K, mw, mws, 0, true) == -1,
                              "%s: bf16 outputs on a prompt chunk accepted", c.what);
                        continue;
                    }
                    /* the bf16 rows are the f32 outputs rounded to nearest even, byte for byte */
                    CUDA_OK(cudaMemset(ybm, 0xff, ytot * sizeof(uint16_t)));
                    CHECK(ds4_exl3_dense_multi_launch(pb, c.n, d[0].q, M, K, mw, mws, 0, false) == 0,
                          "%s: bf16 multi launch refused at M=%d", c.what, M);
                    CUDA_OK(cudaDeviceSynchronize());
                    for (int j = 0; j < c.n; j++) {
                        std::vector<float> f((size_t)M * c.N[j]);
                        std::vector<uint16_t> b(f.size());
                        CUDA_OK(cudaMemcpy(f.data(), ym + yb[j], f.size() * sizeof(float), cudaMemcpyDeviceToHost));
                        CUDA_OK(cudaMemcpy(b.data(), ybm + yb[j], b.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
                        for (size_t i = 0; i < f.size(); i++) {
                            uint32_t u;
                            std::memcpy(&u, &f[i], 4);
                            u += 0x7fffu + ((u >> 16) & 1u);   /* round to nearest even (finite outputs) */
                            bdiff += (uint16_t)(u >> 16) != b[i];
                        }
                        bchecked += f.size();
                    }
                }
            }
            CHECK(diff == 0, "%s: %zu of %zu (width, projection) outputs differ from the single launch", c.what, diff, checked);
            CHECK(bdiff == 0 && bchecked > 0, "%s: %zu of %zu bf16 outputs are not the f32 outputs rounded", c.what, bdiff,
                  bchecked);
            pb[0].y = ym;   /* both outputs */
            CHECK(ds4_exl3_dense_multi_launch(pb, c.n, d[0].q, 1, K, mw, mws, 0) == -1, "%s: y and yb both accepted", c.what);
            cudaFree(ybm);
            CHECK(ds4_exl3_dense_multi_launch(p, c.n, d[0].q, 16, K, mw, mws - 16, 0) == -1, "%s: short workspace accepted",
                  c.what);
            printf("  multi %-44s %zu (width, projection) outputs bit-identical to their single launches; %zu bf16 "
                   "outputs == the f32 rounded\n", c.what, checked, bchecked);
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
            printf("  head %-15s 2560 -> %-6d K=%d (%d blocks): M=1,5,9 %s; 3 windows of %d cut from it == "
                   "its columns %s, vs double max rel %.2e (median <= %.2e)\n", hd.what, N, k2 / 2,
                   N / 128, diff_m ? "DIFFER" : "bit-identical to M=16's rows", W,
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
    /* 8. L295 F7: a logical width short of the stored one.  exllamav3 pads every Linear to a 128 multiple
     * (exl3_out_stored: Inkling's head 4096 -> 201024 ships 201088 wide, svh 0 on the 64 pad columns); the arm
     * computes the stored width and writes the logical columns.  Per shape: (a) the logical call is BIT FOR BIT the
     * stored call's first N columns -- decode widths (the GEMV, M = 1, 5, 16) and prompt chunks (the GEMM, M = 1, 40,
     * each GEMM column through the shape's one algorithm); (b) nothing is written past y's N columns (a canary over
     * the stored width's remainder); (c) the stored call's pad outputs are exactly 0 (svh 0); (d) graded against the
     * double reference -- the whole output for the small shapes, the head's last 2560 stored columns (cut out with
     * exl3_expert_cut, the tail block among them) for the head; (e) a logical width off 4 is refused. */
    {
        struct pad { int K, N, k2; const char *what; };
        const pad pads[] = {{4096, 201024, 12, "Inkling head"}, {4096, 201024, 10, "Inkling head"},
                            {512, 1088, 8, "one tail block"}, {512, 200, 12, "n % 16 != 0"},
                            {512, 64, 6, "all tail"}};
        for (const pad &pc : pads) {
            const int K = pc.K, N = pc.N, k2 = pc.k2, Ns = (int)exl3_out_stored((uint64_t)N), MR = 40;
            weight w = make_weight(K, Ns, k2);
            uint16_t *svp = (uint16_t *)(w.bytes.data() + w.trellis) + K;
            for (int n = N; n < Ns; n++) svp[n] = 0;   /* exllamav3's pad */
            uint64_t lt = 0, lsc = 0, lst = 0;
            CHECK(exl3_linear_layout(K, N, k2, &lt, &lsc, &lst) && lst == w.stride && lt == w.trellis,
                  "%s: exl3_linear_layout(%d, %d) is not the stored %d-wide slice", pc.what, K, N, Ns);
            CHECK(ds4_exl3_dense_workspace_bytes(16, K, N) == ds4_exl3_dense_workspace_bytes(16, K, Ns),
                  "%s: the logical width's workspace is not the stored width's", pc.what);
            const slot s = make_slot(MR, K);
            dev d;
            upload(w, s, MR, d);   /* y and the workspace at the stored width: room for the canary */
            /* one call at width n (Ns or N): the whole y region of M x Ns NaN-filled first, y [M][n] returned, and
             * whether anything past M x n was written */
            auto call = [&](int n, int M, bool prompt, std::vector<float> &y, bool *past) {
                CUDA_OK(cudaMemset(d.y, 0xff, (size_t)M * Ns * sizeof(float)));
                const int rc = ds4_exl3_dense_launch(d.w, k2, d.q, d.y, M, K, n, d.ws, d.ws_bytes, 0, prompt);
                if (rc) return rc;
                CUDA_OK(cudaDeviceSynchronize());
                std::vector<float> all((size_t)M * Ns);
                CUDA_OK(cudaMemcpy(all.data(), d.y, all.size() * sizeof(float), cudaMemcpyDeviceToHost));
                *past = false;
                for (size_t i = (size_t)M * n; i < all.size(); i++) {
                    uint32_t u;
                    std::memcpy(&u, &all[i], 4);
                    *past |= u != 0xffffffffu;
                }
                y.assign(all.begin(), all.begin() + (size_t)M * n);
                return 0;
            };
            /* the logical rows [M][N] vs the stored rows [M][Ns]' first N columns */
            auto same_cols = [&](const std::vector<float> &yl, const std::vector<float> &ys, int M) {
                size_t bad = 0;
                for (int r = 0; r < M; r++)
                    bad += memcmp(yl.data() + (size_t)r * N, ys.data() + (size_t)r * Ns, (size_t)N * sizeof(float)) != 0;
                return bad;
            };
            std::vector<float> ys16, yl, ysp, ylp;
            bool past = false, any_past = false;
            int rc = call(Ns, 16, false, ys16, &past);
            CHECK(rc == 0, "%s K=%d: the stored-width call refused (rc=%d)", pc.what, k2 / 2, rc);
            if (rc) { release(d); continue; }
            size_t pad_nonzero = 0;
            for (int r = 0; r < 16; r++)
                for (int n = N; n < Ns; n++) pad_nonzero += ys16[(size_t)r * Ns + n] != 0.0f;
            size_t bad = 0, rows = 0;
            for (int M : {16, 5, 1}) {
                rc = call(N, M, false, yl, &past);
                CHECK(rc == 0, "%s K=%d M=%d: the logical call refused (rc=%d)", pc.what, k2 / 2, M, rc);
                if (rc) continue;
                bad += same_cols(yl, ys16, M);
                rows += M;
                any_past |= past;
            }
            /* the GEMM: prompt chunks at 40 and 1 rows, logical vs stored, and 1 vs 40 (chunk invariance) */
            size_t bad_p = 0, rows_p = 0;
            rc = call(Ns, MR, true, ysp, &past);
            CHECK(rc == 0, "%s K=%d: the stored-width prompt call refused (rc=%d)", pc.what, k2 / 2, rc);
            for (int M : {MR, 1}) {
                if (rc) break;
                const int r2 = call(N, M, true, ylp, &past);
                CHECK(r2 == 0, "%s K=%d prompt M=%d: the logical call refused (rc=%d)", pc.what, k2 / 2, M, r2);
                if (r2) continue;
                bad_p += same_cols(ylp, ysp, M);
                rows_p += M;
                any_past |= past;
            }
            CHECK(bad == 0, "%s K=%d: %zu of %zu decode rows differ from the stored call's first %d columns", pc.what,
                  k2 / 2, bad, rows, N);
            CHECK(bad_p == 0, "%s K=%d: %zu of %zu prompt rows differ from the stored call's first %d columns", pc.what,
                  k2 / 2, bad_p, rows_p, N);
            CHECK(!any_past, "%s K=%d: a logical call wrote past y's %d columns", pc.what, k2 / 2, N);
            CHECK(pad_nonzero == 0, "%s K=%d: %zu pad outputs of the stored call are not 0 (svh 0)", pc.what, k2 / 2,
                  pad_nonzero);
            /* (d) the grade: the whole output (small shapes) or the head's last 2560 stored columns, cut out */
            const int W = K == 4096 ? 2560 : Ns, lo = Ns - W;
            weight sub;
            sub.K = K; sub.N = W; sub.k2 = k2;
            exl3_cut cut;
            const bool cut_ok = exl3_expert_cut(K, Ns, k2, false, lo, Ns, 0, W, &cut) &&
                                exl3_expert_layout(K, W, k2, &sub.trellis, &sub.scales, &sub.stride);
            CHECK(cut_ok, "%s: the window [%d, %d) does not cut", pc.what, lo, Ns);
            double mx = 0, med = 0, pmx = 0, pmed = 0;
            if (cut_ok) {
                sub.bytes.resize(sub.stride);
                exl3_expert_cut_apply(&cut, w.bytes.data(), sub.bytes.data());
                std::vector<double> what, yref;
                exl3t_dequant(sub.bytes.data(), K, W, k2, what);
                exl3t_reference(sub.bytes.data(), what, K, W, k2, s.x.data(), MR, yref);
                /* the logical call's last W - (Ns - N) columns are the window's first ones; grade those of the
                 * decode rows (M = 16, rerun) and of the prompt chunk (M = 40) */
                const int Wl = W - (Ns - N);
                std::vector<float> yd;
                if (call(N, 16, false, yd, &past) == 0) {
                    std::vector<float> got((size_t)16 * Wl);
                    std::vector<double> want((size_t)16 * Wl);
                    for (int r = 0; r < 16; r++)
                        for (int c = 0; c < Wl; c++) {
                            got[(size_t)r * Wl + c] = yd[(size_t)r * N + lo + c];
                            want[(size_t)r * Wl + c] = yref[(size_t)r * W + c];
                        }
                    grade(got, want, 16, Wl, 0, &mx, &med);
                }
                {
                    std::vector<float> got((size_t)MR * Wl);
                    std::vector<double> want((size_t)MR * Wl);
                    std::vector<double> bnd;
                    exl3t_fp16_operand_bound(sub.bytes.data(), what, K, W, k2, s.x.data(), MR, bnd);
                    size_t over = 0;
                    for (int r = 0; r < MR; r++) {
                        double sc = 0;
                        for (int c = 0; c < Wl; c++) sc = fmax(sc, fabs(yref[(size_t)r * W + c]));
                        for (int c = 0; c < Wl; c++) {
                            /* the stored prompt call's columns: the logical one's, bit for bit (checked above) */
                            const double g = ysp.empty() ? 0.0 : (double)ysp[(size_t)r * Ns + lo + c];
                            got[(size_t)r * Wl + c] = (float)g;
                            want[(size_t)r * Wl + c] = yref[(size_t)r * W + c];
                            over += fabs(g - yref[(size_t)r * W + c]) > bnd[(size_t)r * W + c] + tol * sc;
                        }
                    }
                    grade(got, want, MR, Wl, 0, &pmx, &pmed);
                    CHECK(over == 0, "%s K=%d: %zu prompt outputs past the fp16-operand envelope", pc.what, k2 / 2, over);
                }
            }
            CHECK(mx < tol, "%s %d -> %d K=%d: decode max rel %.3e (limit %.0e)", pc.what, K, N, k2 / 2, mx, tol);
            CHECK(med > 1e-10, "%s K=%d: median rel %.3e -- a degenerate fixture, not a pass", pc.what, k2 / 2, med);
            /* (e) a logical width off 4 */
            CHECK(ds4_exl3_dense_launch(d.w, k2, d.q, d.y, 16, K, N - 2, d.ws, d.ws_bytes, 0) == -1 &&
                  ds4_exl3_dense_workspace_bytes(16, K, N - 2) == 0, "%s: a logical width %% 4 != 0 accepted", pc.what);
            printf("  pad %-22s %4d -> %-6d (stored %d) K=%d: decode M=16,5,1 and prompt M=40,1 == the stored call's "
                   "columns %s; nothing past N %s; pad outputs %s; vs double max rel %.2e (median %.2e), prompt %.2e\n",
                   pc.what, K, N, Ns, k2 / 2, bad || bad_p ? "DIFFER" : "bit for bit", any_past ? "WRITTEN" : "written",
                   pad_nonzero ? "NONZERO" : "0", mx, med, pmx);
            if (K == 4096) {   /* timed (information): the head at 1 / 4 / 16 rows */
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
            }
            release(d);
        }
    }
    arm_section(tol);
    prompt_arm_section();
    printf(g_fail ? "EXL3-DENSE GATE FAIL\n" : "EXL3-DENSE GATE PASS\n");
    return g_fail;
}

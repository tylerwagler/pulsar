/* Correctness gate for the fp16 tensor-core attention kernel.
 *
 * #includes the shipped .cu so it drives the REAL kernel, not a copy.
 *
 * The oracle applies the SAME fp16 operand quantisation in f64, so the format's
 * error cancels and what is left is purely whether the fragment layouts, the
 * two transposed phases, the k-split reduction and the online-softmax
 * bookkeeping are right.  A layout bug does not show up as a small error here;
 * it shows up as garbage, which is the point -- what fp16 COSTS was settled
 * separately in tests/attn_precision_fidelity.cc, so this test only asks
 * whether the kernel computes the function it claims to.
 *
 * WHERE THE RESIDUAL COMES FROM.  It settles at ~4e-4 worst-case, not the ~1e-6
 * that pure f32-vs-f64 accumulation would give, and that is explained rather
 * than tolerated: the kernel accumulates scores in f32 where the oracle uses
 * f64, the two therefore differ by ~1e-6 BEFORE the softmax, and P is then
 * rounded to fp16 -- so a probability sitting near a rounding boundary lands on
 * a different fp16 value in the two, a full ULP apart.  One fp16 ULP is 2^-11 =
 * 4.9e-4, which is exactly where the worst case sits.  (First guess was __expf
 * vs expf; that was WRONG -- swapping them changes the number by 0.05%.)
 *
 * THE GATE IS VALIDATED BY MUTATION, not by assertion.  Injecting one layout
 * bug (phase-3 B row index 8 -> 4) moves the worst per-(token,head) rel L2 from
 * 3.8e-4 to 7.8e-1 -- 2000x the threshold.  A layout error is O(1) here, which
 * is what makes a loose tolerance safe.
 *
 * build+run:
 *   nvcc -O3 -arch=sm_120f -Isrc -Isrc/cuda -o /tmp/attnkt tests/attn_f16_kernel_test.cu
 *   /tmp/attnkt [n_tokens] [window] [n_head] [bench]
 * "bench" skips the O(n^2 d) f64 oracle and only times, for shapes where the
 * oracle would take longer than the measurement is worth.
 */
#include "../src/cuda/pulsar_cuda_attn_f16.cu"

int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "cuda error (%s): %s\n", what, cudaGetErrorString(err));
    return 0;
}

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

/* host mirror of the fp16 round trip the kernel applies to its operands */
static double h16(double v) { return (double)__half2float(__float2half((float)v)); }

int main(int argc, char **argv) {
    const uint32_t n_tokens = (argc > 1) ? (uint32_t)atoi(argv[1]) : 40u;
    const uint32_t window   = (argc > 2) ? (uint32_t)atoi(argv[2]) : 24u;
    const uint32_t n_head   = (argc > 3) ? (uint32_t)atoi(argv[3]) : 32u;
    /* n_comp/ratio exercise the COMPRESSED tail, which the mixed-window launcher
     * uses and the raw-window one does not.  Wiring the kernel in against only
     * the n_comp=0 path would have shipped that half untested. */
    const uint32_t n_comp   = (argc > 5) ? (uint32_t)atoi(argv[5]) : 0u;
    const uint32_t ratio    = (argc > 6) ? (uint32_t)atoi(argv[6]) : 4u;
    const uint32_t D = AF16_DIM;

    printf("attn f16 kernel test: n_tokens=%u window=%u n_head=%u head_dim=%u"
           " n_comp=%u ratio=%u\n\n", n_tokens, window, n_head, D, n_comp, ratio);

    std::mt19937_64 rng(20260808);
    std::normal_distribution<double> nd(0.0, 1.0);

    std::vector<float> q((size_t)n_tokens * n_head * D), kv((size_t)n_tokens * D),
                       ckv((size_t)(n_comp ? n_comp : 1u) * D),
                       sinks(n_head), out((size_t)n_tokens * n_head * D, -12345.f);
    for (auto &v : q) v = (float)(nd(rng) * 0.5);
    for (auto &v : kv) v = (float)(nd(rng) * 0.5);
    for (auto &v : ckv) v = (float)(nd(rng) * 0.5);
    for (auto &v : sinks) v = (float)(nd(rng) * 0.25);

    const int bench_only = (argc > 4 && argv[4][0] == 'b');

    /* ---- oracle, f64, same fp16 operands ---------------------------------- */
    std::vector<double> ref(bench_only ? 0 : (size_t)n_tokens * n_head * D, 0.0);
    for (uint32_t t = 0; t < (bench_only ? 0u : n_tokens); t++) {
        const uint32_t cnt = (window != 0u && t + 1u > window) ? window : t + 1u;
        const uint32_t start = t + 1u - cnt;
        uint32_t ccnt = 0u;
        if (n_comp && ratio) { ccnt = (t + 1u) / ratio; if (ccnt > n_comp) ccnt = n_comp; }
        const uint32_t tot = cnt + ccnt;
        /* row r < cnt is a raw row; beyond that it is a compressed one */
        auto kvrow = [&](uint32_t r) -> const float * {
            return (r < cnt) ? &kv[(size_t)(start + r) * D] : &ckv[(size_t)(r - cnt) * D];
        };
        const double scale = 1.0 / std::sqrt((double)D);
        for (uint32_t h = 0; h < n_head; h++) {
            std::vector<double> s(tot);
            double mx = -INFINITY;
            for (uint32_t r = 0; r < tot; r++) {
                double acc = 0.0;
                const float *kr = kvrow(r);
                for (uint32_t d = 0; d < D; d++)
                    acc += h16(q[((size_t)t * n_head + h) * D + d]) * h16(kr[d]);
                s[r] = acc * scale;
                mx = std::max(mx, s[r]);
            }
            const double sink = sinks[h];
            const double nm = std::max(mx, sink);
            double sum = std::exp(sink - nm);
            std::vector<double> p(tot);
            for (uint32_t r = 0; r < tot; r++) { p[r] = std::exp(s[r] - nm); sum += p[r]; }
            for (uint32_t d = 0; d < D; d++) {
                double o = 0.0;
                for (uint32_t r = 0; r < tot; r++) o += h16(p[r]) * h16(kvrow(r)[d]);
                ref[((size_t)t * n_head + h) * D + d] = o / sum;
            }
        }
    }

    /* ---- kernel ----------------------------------------------------------- */
    float *dq, *dkv, *dckv, *ds, *dout;
    cudaMalloc(&dq, q.size() * 4); cudaMalloc(&dkv, kv.size() * 4);
    cudaMalloc(&dckv, ckv.size() * 4);
    cudaMemcpy(dckv, ckv.data(), ckv.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&ds, sinks.size() * 4); cudaMalloc(&dout, out.size() * 4);
    cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dkv, kv.data(), kv.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, sinks.data(), sinks.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dout, out.data(), out.size() * 4, cudaMemcpyHostToDevice);

    const int rc = pulsar_gpu_attention_f16_prefill(dout, ds, dq, dkv,
                                                    n_comp ? dckv : NULL,
                                                    n_tokens, n_comp, window, ratio,
                                                    n_head, D, 0);
    if (!rc) { printf("LAUNCH REFUSED (shape gate)\n"); return 1; }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("EXEC FAILED: %s\n", cudaGetErrorString(cudaGetLastError())); return 1;
    }
    cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost);

    /* ---- timing, at the shape nsys measured the shipping kernel at --------- */
    {
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        const int iters = 20;
        for (int i = 0; i < 3; i++)
            pulsar_gpu_attention_f16_prefill(dout, ds, dq, dkv, n_comp ? dckv : NULL,
                                             n_tokens, n_comp, window, ratio, n_head, D, 0);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int i = 0; i < iters; i++)
            pulsar_gpu_attention_f16_prefill(dout, ds, dq, dkv, n_comp ? dckv : NULL,
                                             n_tokens, n_comp, window, ratio, n_head, D, 0);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
        printf("timing: %.4f ms/launch  (n_tokens=%u window=%u n_head=%u)\n",
               ms / iters, n_tokens, window, n_head);
    }
    if (bench_only) { printf("\nBENCH ONLY (oracle skipped)\n"); return 0; }

    /* ---- compare ---------------------------------------------------------- */
    /* Per (token, head) relative L2, the same scale-aware measure
     * tests/attn_precision_fidelity.cc reports, so the two are comparable.  A
     * plain elementwise max-relative is useless here: an output element that
     * happens to sit near zero divides a normal rounding difference by almost
     * nothing and reports ~1.0 while the vector is perfectly good. */
    double sum_abs = 0, sum_ref = 0, worst_l2 = 0, max_abs = 0;
    size_t nan = 0, untouched = 0;
    for (size_t i = 0; i < out.size(); i++) {
        if (std::isnan((double)out[i])) nan++;
        if (out[i] == -12345.f) untouched++;
    }
    for (uint32_t t = 0; t < n_tokens; t++)
        for (uint32_t h = 0; h < n_head; h++) {
            double num = 0, den = 0;
            for (uint32_t d = 0; d < D; d++) {
                const size_t i = ((size_t)t * n_head + h) * D + d;
                const double e = (double)out[i] - ref[i];
                num += e * e; den += ref[i] * ref[i];
                sum_abs += std::fabs(e); sum_ref += std::fabs(ref[i]);
                max_abs = std::max(max_abs, std::fabs(e));
            }
            if (den > 0) worst_l2 = std::max(worst_l2, std::sqrt(num / den));
        }
    printf("mean |delta| / mean |ref|   = %.3e\n", sum_ref > 0 ? sum_abs / sum_ref : 0.0);
    printf("worst per-(tok,head) rel L2 = %.3e\n", worst_l2);
    printf("max |delta|                 = %.3e\n", max_abs);
    printf("NaN = %zu, never-written = %zu\n", nan, untouched);

    const int pass = (nan == 0 && untouched == 0 && worst_l2 < 1e-3 &&
                      (sum_ref > 0 && sum_abs / sum_ref < 1e-3));
    printf("\nATTN F16 KERNEL TEST: %s\n", pass ? "PASS" : "FAIL");
    if (!pass) {
        printf("  first few (ref vs got):\n");
        for (size_t i = 0; i < 8 && i < out.size(); i++)
            printf("    [%zu] %12.6f  %12.6f\n", i, ref[i], (double)out[i]);
    }
    return pass ? 0 : 1;
}

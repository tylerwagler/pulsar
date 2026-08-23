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
#include <cstring>
#include <vector>
#include <random>

/* host mirror of the fp16 round trip the kernel applies to its operands */
static double h16(double v) { return (double)__half2float(__float2half((float)v)); }


#include "attn_pack_fixture.h"   /* host E4M3 encode/decode: see the note there
 * on why the fixture ENCODES a draw instead of drawing random bytes */

int main(int argc, char **argv) {
    const uint32_t n_tokens = (argc > 1) ? (uint32_t)atoi(argv[1]) : 40u;
    const uint32_t window   = (argc > 2) ? (uint32_t)atoi(argv[2]) : 24u;
    const uint32_t n_head   = (argc > 3) ? (uint32_t)atoi(argv[3]) : 32u;
    /* n_comp/ratio exercise the COMPRESSED tail, which the mixed-window launcher
     * uses and the raw-window one does not.  Wiring the kernel in against only
     * the n_comp=0 path would have shipped that half untested. */
    const uint32_t n_comp   = (argc > 5) ? (uint32_t)atoi(argv[5]) : 0u;
    const uint32_t ratio    = (argc > 6) ? (uint32_t)atoi(argv[6]) : 4u;
    /* top_k > 0 switches to INDEXED mode: compressed rows become a top-k
     * selection and raw rows come from a ring buffer.  Exercised against the
     * same f64 oracle, because the row PLAN is the part most likely to be
     * subtly wrong and it is invisible in the output magnitude. */
    const uint32_t top_k    = (argc > 7) ? (uint32_t)atoi(argv[7]) : 0u;
    const uint32_t raw_cap  = (argc > 8) ? (uint32_t)atoi(argv[8]) : 0u;
    const uint32_t D = AF16_DIM;

    printf("attn f16 kernel test: n_tokens=%u window=%u n_head=%u head_dim=%u"
           " n_comp=%u ratio=%u\n\n", n_tokens, window, n_head, D, n_comp, ratio);

    std::mt19937_64 rng(20260808);
    std::normal_distribution<double> nd(0.0, 1.0);

    std::vector<float> q((size_t)n_tokens * n_head * D), kv((size_t)n_tokens * D),
                       ckv((size_t)(n_comp ? n_comp : 1u) * D),
                       sinks(n_head), out((size_t)n_tokens * n_head * D, -12345.f);
    for (auto &v : q) v = (float)(nd(rng) * 0.5);
    /* Raw KV: build PULSAR_ATTN_PACK rows first, then DECODE them into kv[] so
     * the f64 oracle below and the kernel are looking at the same numbers. */
    const uint32_t n_nope_h = D - PULSAR_ATTN_PACK_NROT;
    const size_t pack_row_h = (size_t)PULSAR_ATTN_PACK_ROWBYTES(D);
    std::vector<uint8_t> rawp((size_t)n_tokens * pack_row_h);
    {
        for (uint32_t r = 0; r < n_tokens; r++) {
            uint8_t *row = &rawp[(size_t)r * pack_row_h];
            /* scales FIRST -- the payload byte is an encode against its own
             * block scale, so the scale has to exist before the value does.
             * Near 2^0 so decoded magnitudes match the f32 fixture this
             * replaced, and so the tolerances below stay meaningful. */
            for (uint32_t sc = 0; sc < PULSAR_ATTN_PACK_SCALES_PAD(D); sc++)
                row[n_nope_h + sc] = (uint8_t)(126 + (int)((r + sc) % 3));
            for (uint32_t d = 0; d < n_nope_h; d++)
                row[d] = host_e4m3_encode((float)(nd(rng) * 0.5),
                                          host_pack_block_scale(row, n_nope_h, d,
                                                                PULSAR_FP8_KV_BLOCK));
            uint16_t *rope = (uint16_t *)(row + n_nope_h + PULSAR_ATTN_PACK_SCALES_PAD(D));
            for (uint32_t d = 0; d < PULSAR_ATTN_PACK_NROT; d++) {
                const float f = (float)(nd(rng) * 0.5);
                uint32_t u; std::memcpy(&u, &f, sizeof u);
                rope[d] = (uint16_t)(u >> 16);          /* truncate to bf16 */
            }
            for (uint32_t d = 0; d < D; d++) {
                if (d < n_nope_h) {
                    kv[(size_t)r * D + d] = host_e4m3_decode(row[d],
                        host_pack_block_scale(row, n_nope_h, d, PULSAR_FP8_KV_BLOCK));
                } else {
                    kv[(size_t)r * D + d] = host_bf16_widen(rope[d - n_nope_h]);
                }
            }
        }
    }
    /* Comp rows are PULSAR_ATTN_PACK too.  They were f32 until 2026-08-18 and the
     * kernel chose between formats on a comp_pack flag; that flag is gone, so a
     * comp row IS a packed row and the fixture has to build one.  Same
     * encode-a-draw discipline as the raw rows above -- see attn_pack_fixture.h. */
    const size_t n_ckv = (size_t)(n_comp ? n_comp : 1u);
    std::vector<uint8_t> ckvp(n_ckv * pack_row_h);
    for (size_t r = 0; r < n_ckv; r++) {
        uint8_t *row = &ckvp[r * pack_row_h];
        for (uint32_t sc = 0; sc < PULSAR_ATTN_PACK_SCALES_PAD(D); sc++)
            row[n_nope_h + sc] = (uint8_t)(126 + (int)((r + sc) % 3));
        for (uint32_t d = 0; d < n_nope_h; d++)
            row[d] = host_e4m3_encode((float)(nd(rng) * 0.5),
                                      host_pack_block_scale(row, n_nope_h, d,
                                                            PULSAR_FP8_KV_BLOCK));
        uint16_t *rope = (uint16_t *)(row + n_nope_h + PULSAR_ATTN_PACK_SCALES_PAD(D));
        for (uint32_t d = 0; d < PULSAR_ATTN_PACK_NROT; d++) {
            const float f = (float)(nd(rng) * 0.5);
            uint32_t u; std::memcpy(&u, &f, sizeof u);
            rope[d] = (uint16_t)(u >> 16);
        }
        for (uint32_t d = 0; d < D; d++)
            ckv[r * D + d] = (d < n_nope_h)
                ? host_e4m3_decode(row[d], host_pack_block_scale(row, n_nope_h, d,
                                                                 PULSAR_FP8_KV_BLOCK))
                : host_bf16_widen(rope[d - n_nope_h]);
    }
    for (auto &v : sinks) v = (float)(nd(rng) * 0.25);

    const int bench_only = (argc > 4 && argv[4][0] == 'b');
    /* top_k > 0: indexed selection.  top_k == 0 with raw_cap set: the
     * DECODE-BATCH shape -- ring raw rows plus a visible-prefix comp sweep
     * with NO topk table.  At pos0=0 and raw_cap >= n_tokens the ring
     * collapses to the dense window, so the dense oracle is the reference. */
    const int indexed = (top_k != 0u) || (raw_cap != 0u);
    const int use_topk = (top_k != 0u);
    const uint32_t rcap = raw_cap ? raw_cap : n_tokens;
    const uint32_t n_raw = indexed ? (n_tokens < rcap ? n_tokens : rcap) : 0u;
    const uint32_t pos0 = 0u;
    /* deterministic pseudo-selection, deliberately unsorted and repeating so a
     * kernel that assumed a sorted prefix would fail */
    std::vector<int32_t> tk((size_t)n_tokens * (top_k ? top_k : 1u), 0);
    if (indexed)
        for (uint32_t t = 0; t < n_tokens; t++)
            for (uint32_t i = 0; i < top_k; i++)
                tk[(size_t)t * top_k + i] = n_comp ? (int32_t)((t * 7u + i * 13u) % n_comp) : 0;

    /* ---- oracle, f64, same fp16 operands ---------------------------------- */
    std::vector<double> ref(bench_only ? 0 : (size_t)n_tokens * n_head * D, 0.0);
    for (uint32_t t = 0; t < (bench_only ? 0u : n_tokens); t++) {
        uint32_t cnt, start = 0u, ccnt = 0u, rfirst = 0u, vis = n_comp;
        if (indexed) {
            const uint32_t qpos = pos0 + t;
            const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
            uint32_t rc = 0u;
            if (n_raw != 0u) {
                const uint32_t last = first_raw_pos + n_raw - 1u;
                if (qpos >= first_raw_pos) {
                    uint32_t lo = first_raw_pos;
                    if (window != 0u && qpos + 1u > window) {
                        const uint32_t wlo = qpos + 1u - window;
                        if (wlo > lo) lo = wlo;
                    }
                    const uint32_t hi = qpos < last ? qpos : last;
                    if (hi >= lo) { rfirst = lo - first_raw_pos; rc = hi - lo + 1u; }
                    if (rc > 256u) rc = 256u;
                }
            }
            cnt = rc;
            if (ratio) { vis = (qpos + 1u) / ratio; if (vis > n_comp) vis = n_comp; }
            ccnt = use_topk ? (top_k < vis ? top_k : vis) : vis;
        } else {
            cnt = (window != 0u && t + 1u > window) ? window : t + 1u;
            start = t + 1u - cnt;
            if (n_comp && ratio) { ccnt = (t + 1u) / ratio; if (ccnt > n_comp) ccnt = n_comp; }
        }
        const uint32_t tot = cnt + ccnt;
        auto kvrow = [&](uint32_t r) -> const float * {
            if (r < cnt) {
                const uint32_t rr = indexed ? ((0u + rfirst + r) % rcap) : (start + r);
                return &kv[(size_t)rr * D];
            }
            uint32_t ci = r - cnt;
            if (indexed && use_topk) {
                const int32_t c = tk[(size_t)t * top_k + ci];
                ci = (c >= 0 && (uint32_t)c < vis) ? (uint32_t)c : 0u;
            }
            return &ckv[(size_t)ci * D];
        };
        /* An out-of-range top-k selection is NO row, not row 0.  The f16 kernel
         * loads row 0 to keep the access in range and the branch uniform, then
         * masks the score to -INF so the row contributes zero; substituting row
         * 0 for real would inject an unselected position AND double-count row 0
         * whenever row 0 was also legitimately selected.  This oracle used to
         * mirror the f32 kernel, which does substitute -- so it encoded the
         * behaviour the f16 kernel was written to fix, and every top_k>0 shape
         * disagreed with it by ~8e-1.  The divergence is confined to
         * out-of-VISIBLE selections, which is why ratio=0 (everything visible)
         * agreed and ratio=4 did not. */
        auto kvbad = [&](uint32_t r) -> bool {
            if (r < cnt || !(indexed && use_topk)) return false;
            const int32_t c = tk[(size_t)t * top_k + (r - cnt)];
            return !(c >= 0 && (uint32_t)c < vis);
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
                s[r] = kvbad(r) ? -INFINITY : acc * scale;
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
    pulsar_q_t *dq;
    float *dkv, *dckv, *ds, *dout;
    dq = fixture_upload_q(q); cudaMalloc(&dkv, rawp.size());
    cudaMalloc(&dckv, ckvp.size());
    cudaMemcpy(dckv, ckvp.data(), ckvp.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&ds, sinks.size() * 4); cudaMalloc(&dout, out.size() * 4);
    cudaMemcpy(dkv, rawp.data(), rawp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, sinks.data(), sinks.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dout, out.data(), out.size() * 4, cudaMemcpyHostToDevice);

    int32_t *dtk = NULL;
    if (use_topk) {
        cudaMalloc(&dtk, tk.size() * 4);
        cudaMemcpy(dtk, tk.data(), tk.size() * 4, cudaMemcpyHostToDevice);
    }
    const int rc = indexed
        ? pulsar_gpu_attention_f16_indexed(dout, ds, dq, (const pulsar_attn_pack_t *)dkv,
                                           (const pulsar_attn_pack_t *)dckv, use_topk ? dtk : NULL,
                                           n_tokens, pos0, n_raw, rcap, 0u,
                                           n_comp, top_k, window, ratio, n_head, D,
                                           NULL, NULL, NULL, 0u, 1u, NULL)
        : pulsar_gpu_attention_f16_prefill(dout, ds, dq, (const pulsar_attn_pack_t *)dkv,
                                           n_comp ? (const pulsar_attn_pack_t *)dckv : NULL,
                                           n_tokens, n_comp, window, ratio,
                                           n_head, D, NULL);
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
            pulsar_gpu_attention_f16_prefill(dout, ds, dq, (const pulsar_attn_pack_t *)dkv, n_comp ? (const pulsar_attn_pack_t *)dckv : NULL,
                                             n_tokens, n_comp, window, ratio, n_head, D, NULL);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int i = 0; i < iters; i++)
            pulsar_gpu_attention_f16_prefill(dout, ds, dq, (const pulsar_attn_pack_t *)dkv, n_comp ? (const pulsar_attn_pack_t *)dckv : NULL,
                                             n_tokens, n_comp, window, ratio, n_head, D, NULL);
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

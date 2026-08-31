/* Correctness gate for the block-scaled indexer scorer.
 *
 * #includes the shipped .cu so it drives the REAL kernel, not a copy (the
 * pattern the since-deleted tests/attn_indexed_bench.cu used; a71e346).
 *
 * The oracle applies the SAME E4M3 quantisation and the SAME cache decode in
 * f64, so quantisation error cancels and what remains is purely whether the
 * fragment layouts, SF lane maps and scale rebias are right.  A layout bug does
 * not produce a small error here -- it produces garbage -- which is the point:
 * the fidelity question was already answered separately
 * (tests/idx_quant_fidelity.cc), so this test is only asking "does the kernel
 * compute the function it claims to".
 *
 * build+run:
 *   nvcc -O3 -arch=sm_120f -Isrc -Isrc/cuda -o /tmp/idx_kt tests/idx_mxfp4_kernel_test.cu
 *   /tmp/idx_kt
 */
#include "../src/cuda/pulsar_cuda_indexer_mxfp4.cu"

/* cuda_ok lives in pulsar_cuda_runtime.cu; this TU links neither the engine nor
 * the rest of the backend, so provide the one symbol the launcher needs. */
/* cuda_tmp_alloc is the backend's scratch arena (pulsar_cuda_runtime.cu).  This
 * TU links neither, so back it with a plain cudaMalloc that is reused across
 * calls -- enough for a correctness/timing harness. */
void *cuda_tmp_alloc(uint64_t bytes, const char *what) {
    static void *buf = nullptr; static uint64_t cap = 0;
    if (bytes > cap) {
        if (buf) cudaFree(buf);
        if (cudaMalloc(&buf, bytes) != cudaSuccess) {
            fprintf(stderr, "tmp alloc failed (%s)\n", what); buf = nullptr; cap = 0; return nullptr;
        }
        cap = bytes;
    }
    return buf;
}

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

static const double kE2M1[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};

/* Host mirror of idx_f32_to_e4m3 + its decode, in double. */
static double host_e4m3_roundtrip(double v) {
    const double a = std::fabs(v);
    const double sgn = (v < 0.0) ? -1.0 : 1.0;
    if (!(a > 0.0)) return 0.0;
    if (a >= 448.0) return sgn * 448.0;
    int e; std::frexp(a, &e); e -= 1;
    if (e < -6) {
        const double step = std::ldexp(1.0, -9);
        int m = (int)(a / step + 0.5);
        if (m > 7) m = 7;
        return sgn * (m * step);
    }
    const double step = std::ldexp(1.0, e - 3);
    int m = (int)(a / step + 0.5) - 8;
    int ee = e + 7;
    if (m > 7) { m = 0; ee += 1; }
    if (ee > 15) return sgn * 448.0;
    return sgn * ((8 + m) * std::ldexp(1.0, ee - 7 - 3));
}

int main(int argc, char **argv) {
    const uint32_t n_comp   = (argc > 1) ? (uint32_t)atoi(argv[1]) : 256u;
    const uint32_t n_tokens = (argc > 2) ? (uint32_t)atoi(argv[2]) : 8u;
    const uint32_t ratio = 4u;
    const uint32_t pos0 = (argc > 3) ? (uint32_t)atoi(argv[3]) : 4096u;
    const float scale = 0.125f;
    const int causal = 1;

    printf("indexer mxfp4 kernel test: n_comp=%u n_tokens=%u heads=%u dim=%u\n\n",
           n_comp, n_tokens, IDX_HEADS, IDX_HEAD_DIM);

    std::mt19937_64 rng(20260808);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_int_distribution<int> nib(0, 15);
    std::uniform_int_distribution<int> sfd(120, 134);

    const size_t rowb = PULSAR_MXKV_FP4_ROWBYTES(128u);
    std::vector<uint8_t> comp((size_t)n_comp * rowb);
    for (auto &b : comp) b = 0;
    for (uint32_t c = 0; c < n_comp; c++) {
        uint8_t *r = &comp[(size_t)c * rowb];
        for (int j = 0; j < 64; j++) r[j] = (uint8_t)(nib(rng) | (nib(rng) << 4));
        for (int j = 0; j < 4; j++)  r[64 + j] = (uint8_t)sfd(rng);
    }

    std::vector<float> q((size_t)n_tokens * IDX_HEADS * IDX_HEAD_DIM);
    for (auto &v : q) v = (float)nd(rng);
    std::vector<float> w((size_t)n_tokens * IDX_HEADS);
    for (auto &v : w) v = (float)std::fabs(nd(rng));

    /* ---- oracle ---------------------------------------------------------- */
    std::vector<double> ref((size_t)n_tokens * n_comp, 0.0);
    for (uint32_t t = 0; t < n_tokens; t++) {
        /* quantise this token's Q exactly as the kernel does */
        std::vector<double> qa((size_t)IDX_HEADS * IDX_HEAD_DIM);
        for (uint32_t h = 0; h < IDX_HEADS; h++) {
            for (uint32_t blk = 0; blk < IDX_KSLABS; blk++) {
                const float *src = &q[((size_t)t * IDX_HEADS + h) * IDX_HEAD_DIM + blk * 32];
                double amax = 0.0;
                for (int i = 0; i < 32; i++) amax = std::max(amax, (double)std::fabs(src[i]));
                int se = -127;
                if (amax > 0.0) { int e; std::frexp(amax, &e); se = (e - 1) - 8; }
                const double s = std::ldexp(1.0, se);
                for (int i = 0; i < 32; i++)
                    qa[(size_t)h * IDX_HEAD_DIM + blk * 32 + i] =
                        host_e4m3_roundtrip((double)src[i] / s) * s;
            }
        }
        const uint32_t visible = (pos0 + t + 1u) / ratio;
        for (uint32_t c = 0; c < n_comp; c++) {
            if (causal && c >= visible) { ref[(size_t)t * n_comp + c] = -INFINITY; continue; }
            const uint8_t *r = &comp[(size_t)c * rowb];
            double kv[IDX_HEAD_DIM];
            for (uint32_t d = 0; d < IDX_HEAD_DIM; d++) {
                const double sc = std::ldexp(1.0, (int)r[64 + (d >> 5)] - 127);
                const uint8_t byte = r[d >> 1];
                const uint8_t n = (d & 1u) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0xF);
                const double mag = kE2M1[n & 7u];
                kv[d] = (n & 8u) ? -mag * sc : mag * sc;
            }
            double acc = 0.0;
            for (uint32_t h = 0; h < IDX_HEADS; h++) {
                double dot = 0.0;
                for (uint32_t d = 0; d < IDX_HEAD_DIM; d++)
                    dot += qa[(size_t)h * IDX_HEAD_DIM + d] * kv[d];
                if (dot > 0.0) acc += dot * (double)w[(size_t)t * IDX_HEADS + h];
            }
            ref[(size_t)t * n_comp + c] = acc * (double)scale;
        }
    }

    /* ---- kernel ---------------------------------------------------------- */
    float *d_scores = nullptr, *d_q = nullptr, *d_w = nullptr;
    uint8_t *d_comp = nullptr;
    cudaMalloc(&d_scores, (size_t)n_tokens * n_comp * sizeof(float));
    cudaMalloc(&d_q, q.size() * sizeof(float));
    cudaMalloc(&d_w, w.size() * sizeof(float));
    cudaMalloc(&d_comp, comp.size());
    cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_comp, comp.data(), comp.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_scores, 0, (size_t)n_tokens * n_comp * sizeof(float));

    const int rc = pulsar_gpu_indexer_scores_mxfp4(
        d_scores, d_q, d_w, d_comp, n_comp, n_tokens, pos0,
        IDX_HEADS, IDX_HEAD_DIM, ratio, scale, causal, /*fp4=*/1);
    if (!rc) { printf("LAUNCH REFUSED (shape gate)\n"); return 1; }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("EXEC FAILED: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    std::vector<float> got((size_t)n_tokens * n_comp);
    cudaMemcpy(got.data(), d_scores, got.size() * sizeof(float), cudaMemcpyDeviceToHost);

    /* ---- compare --------------------------------------------------------- */
    double max_rel = 0.0, sum_abs = 0.0, sum_ref = 0.0;
    size_t nfinite = 0, nbad = 0, nnan = 0;
    for (size_t i = 0; i < got.size(); i++) {
        const double r = ref[i], g = (double)got[i];
        if (std::isinf(r) && std::isinf(g) && (r < 0) == (g < 0)) continue;  /* masked */
        if (std::isnan(g)) { nnan++; continue; }
        if (!std::isfinite(r) || !std::isfinite(g)) { nbad++; continue; }
        nfinite++;
        sum_abs += std::fabs(g - r);
        sum_ref += std::fabs(r);
        const double den = std::fabs(r) > 1e-6 ? std::fabs(r) : 1e-6;
        max_rel = std::max(max_rel, std::fabs(g - r) / den);
    }
    printf("compared %zu finite elements (%zu masked-inf skipped, %zu NaN, %zu mismatch-kind)\n",
           nfinite, got.size() - nfinite - nnan - nbad, nnan, nbad);
    printf("mean |delta| / mean |ref| = %.3e\n", sum_ref > 0 ? sum_abs / sum_ref : 0.0);
    printf("max relative error        = %.3e\n", max_rel);

    /* ---- timing at the engine's real launch shape ------------------------
     * The engine issues 168 launches per 4096-token prefill (four 512-token
     * sub-batches x 42 layers) and the old wmma128 scorer costs 92.38 ms total,
     * i.e. 0.550 ms per launch on average as n_comp ramps 32 -> 512+.  Time the
     * same per-launch work here.  This is a kernel-level number, not an
     * end-to-end one -- it does not include the Q packing the engine would
     * still have to do outside this kernel. */
    /* Pack alone.  The n_comp sweep put 0.270 ms of the 0.329 in a term that
     * does not scale with n_comp; time that term directly rather than keep
     * reading it off a regression intercept. */
    {
        uint8_t *d_qa = nullptr, *d_qsf = nullptr;
        const size_t qab = (size_t)n_tokens * IDX_HEADS * IDX_HEAD_DIM;
        cudaMalloc(&d_qa, qab);
        cudaMalloc(&d_qsf, (size_t)n_tokens * IDX_HEADS * IDX_KSLABS);
        const uint32_t rows = n_tokens * IDX_HEADS;
        cudaEvent_t p0, p1; cudaEventCreate(&p0); cudaEventCreate(&p1);
        const int iters = 50;
        for (int i = 0; i < 5; i++)
            idx_pack_q_kernel<<<(rows + 7u) / 8u, 256>>>(d_qa, d_qsf, d_q, rows);
        cudaDeviceSynchronize();
        cudaEventRecord(p0);
        for (int i = 0; i < iters; i++)
            idx_pack_q_kernel<<<(rows + 7u) / 8u, 256>>>(d_qa, d_qsf, d_q, rows);
        cudaEventRecord(p1);
        cudaEventSynchronize(p1);
        float ms = 0.f; cudaEventElapsedTime(&ms, p0, p1);
        const double bytes = (double)qab * 4.0 + (double)qab;   /* f32 in, e4m3 out */
        /* Cache-warm: the loop re-reads one 16.8 MB Q buffer, so this lands
         * above DRAM bandwidth and is a lower bound on the real cost.  The
         * honest number is the n_comp-sweep intercept, which pays cold misses
         * and both launch overheads. */
        printf("pack alone: %.4f ms/launch  (%.1f GB/s effective, cache-warm)\n",
               ms / iters, bytes / ((ms / iters) * 1e-3) / 1e9);
        cudaFree(d_qa); cudaFree(d_qsf);
    }

    {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        const int iters = 50;
        for (int i = 0; i < 5; i++)   /* warm */
            pulsar_gpu_indexer_scores_mxfp4(d_scores, d_q, d_w, d_comp, n_comp,
                                            n_tokens, pos0, IDX_HEADS, IDX_HEAD_DIM,
                                            ratio, scale, causal, 1);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int i = 0; i < iters; i++)
            pulsar_gpu_indexer_scores_mxfp4(d_scores, d_q, d_w, d_comp, n_comp,
                                            n_tokens, pos0, IDX_HEADS, IDX_HEAD_DIM,
                                            ratio, scale, causal, 1);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
        printf("timing: %.4f ms/launch  (n_comp=%u n_tokens=%u)\n", ms / iters, n_comp, n_tokens);
    }

    const int pass = (nnan == 0 && nbad == 0 && max_rel < 1e-3);
    printf("\nKERNEL TEST: %s\n", pass ? "PASS" : "FAIL");
    if (!pass) {
        printf("  first few (ref vs got):\n");
        for (size_t i = 0; i < 8 && i < got.size(); i++)
            printf("    [%zu] %14.6f  %14.6f\n", i, ref[i], (double)got[i]);
    }
    return pass ? 0 : 1;
}

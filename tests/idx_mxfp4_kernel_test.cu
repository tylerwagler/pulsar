/* Correctness gate for the block-scaled indexer scorer.
 *
 * #includes the shipped .cu so it drives the REAL kernel, not a copy (the
 * pattern the since-deleted tests/attn_indexed_bench.cu used; a71e346).
 *
 * The oracle decodes the SAME PACKED BYTES the kernel is given, in f64, so
 * quantisation error cancels and what remains is purely whether the fragment
 * layouts, SF lane maps and scale rebias are right.  Q and comp are both
 * MXKV-FP4 68-byte rows since f9fbafe ("the indexer Q container IS the
 * quantized encoding now"), and both decode by the sanctioned accessors in
 * pulsar_cuda_indexer.cu: scale = 2^(byte-127), low nibble first, E2M1.
 *
 * The -1 SF rebias the MXFP4 kernel applies to BOTH operands
 * (idx_expand_q_kernel for Q, the sSFB load for comp) re-encodes storage into
 * the MMA's ue8m0 operand.  The oracle deliberately does NOT mirror it: that
 * rebias is under test, so the kernel has to reach the same values its own way.  A layout bug does
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

/* Bump arena over one reservation (pulsar_cuda_runtime.cu).  Mirrors the real
 * implementation including the STICKY failure latch -- the launcher takes its
 * slices and then does a single null check, which is only valid because one
 * refusal poisons every later take.  A stub that always succeeded would hand
 * back overlapping scratch and the gate would still print PASS. */
int cuda_arena_begin(cuda_arena *a, uint64_t bytes, const char *what) {
    if (!a) return 0;
    a->base = nullptr; a->cap = 0; a->used = 0; a->what = what; a->failed = 0;
    if (bytes == 0) return 1;
    void *p = cuda_tmp_alloc(bytes, what);
    if (!p) { a->failed = 1; return 0; }
    a->base = (uint8_t *)p;
    a->cap  = bytes;
    return 1;
}

void *cuda_arena_take(cuda_arena *a, uint64_t bytes, uint64_t align) {
    if (!a || a->failed) return nullptr;
    if (bytes == 0) return nullptr;
    if (align == 0) align = 16;
    const uint64_t off = (a->used + (align - 1)) & ~(align - 1);
    if (off > a->cap || bytes > a->cap - off) { a->failed = 1; return nullptr; }
    a->used = off + bytes;
    return a->base + off;
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

/* Host mirror of the SANCTIONED packed-row decode (idx_comp_load_dev /
 * idx_q_load4 in pulsar_cuda_indexer.cu), in double.  One helper for both
 * operands, because since f9fbafe they are the same 68-byte format. */
static double host_pack_ld(const uint8_t *row, uint32_t d) {
    const double scale = std::ldexp(1.0, (int)row[64 + (d >> 5)] - 127);
    const uint8_t byte = row[d >> 1];
    const uint8_t n = (d & 1u) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0xF);
    const double mag = kE2M1[n & 7u] * scale;
    return (n & 8u) ? -mag : mag;
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

    /* Q is stored in the same packed layout as comp (f9fbafe), so it is
     * generated the same way -- random nibbles and random block scales -- and
     * NOT as f32 that the test then quantises.  There is no f32 Q anywhere on
     * this path any more. */
    const size_t qrows = (size_t)n_tokens * IDX_HEADS;
    std::vector<uint8_t> qp(qrows * rowb, 0);
    for (size_t r = 0; r < qrows; r++) {
        uint8_t *row = &qp[r * rowb];
        for (int j = 0; j < 64; j++) row[j] = (uint8_t)(nib(rng) | (nib(rng) << 4));
        for (int j = 0; j < 4; j++)  row[64 + j] = (uint8_t)sfd(rng);
    }
    std::vector<float> w((size_t)n_tokens * IDX_HEADS);
    for (auto &v : w) v = (float)std::fabs(nd(rng));

    /* ---- oracle ---------------------------------------------------------- */
    std::vector<double> ref((size_t)n_tokens * n_comp, 0.0);
    for (uint32_t t = 0; t < n_tokens; t++) {
        /* decode this token's packed Q rows -- the same bytes the kernel gets */
        std::vector<double> qa((size_t)IDX_HEADS * IDX_HEAD_DIM);
        for (uint32_t h = 0; h < IDX_HEADS; h++) {
            const uint8_t *qrow = &qp[((size_t)t * IDX_HEADS + h) * rowb];
            for (uint32_t d = 0; d < IDX_HEAD_DIM; d++)
                qa[(size_t)h * IDX_HEAD_DIM + d] = host_pack_ld(qrow, d);
        }
        const uint32_t visible = (pos0 + t + 1u) / ratio;
        for (uint32_t c = 0; c < n_comp; c++) {
            if (causal && c >= visible) { ref[(size_t)t * n_comp + c] = -INFINITY; continue; }
            const uint8_t *r = &comp[(size_t)c * rowb];
            double kv[IDX_HEAD_DIM];
            for (uint32_t d = 0; d < IDX_HEAD_DIM; d++) kv[d] = host_pack_ld(r, d);
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
    float *d_scores = nullptr, *d_w = nullptr;
    uint8_t *d_q = nullptr, *d_comp = nullptr;
    cudaMalloc(&d_scores, (size_t)n_tokens * n_comp * sizeof(float));
    cudaMalloc(&d_q, qp.size());
    cudaMalloc(&d_w, w.size() * sizeof(float));
    cudaMalloc(&d_comp, comp.size());
    cudaMemcpy(d_q, qp.data(), qp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_comp, comp.data(), comp.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_scores, 0, (size_t)n_tokens * n_comp * sizeof(float));

    const int rc = pulsar_gpu_indexer_scores_mxfp4(
        d_scores, (const pulsar_mxkv_pack_t *)d_q, d_w,
        (const pulsar_mxkv_pack_t *)d_comp, n_comp, n_tokens, pos0,
        IDX_HEADS, IDX_HEAD_DIM, ratio, scale, causal);
    if (!rc) { printf("LAUNCH REFUSED (shape gate)\n"); return 1; }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("EXEC FAILED: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    std::vector<float> got((size_t)n_tokens * n_comp);
    cudaMemcpy(got.data(), d_scores, got.size() * sizeof(float), cudaMemcpyDeviceToHost);

    /* ---- compare ----------------------------------------------------------
     * `ref` is the STORAGE semantics: the producer writes a plain bias-127
     * byte (pulsar_cuda_norm_kv.cu:746) and the sanctioned accessor reads it
     * back as 2^(byte-127) (idx_comp_load_dev).  Measured, this tier lands at
     * exactly ref/4 -- uniformly, at every shape tried.
     *
     * So the assertion is UNIFORMITY plus the value of the factor, not raw
     * equality.  Rescaling the oracle to match would pin the discrepancy in
     * place forever; asserting the relationship keeps both alarms live:
     *   - a layout / SF-lane-map / per-block rebias bug destroys uniformity
     *     (it produces garbage, never a clean constant ratio)  -> FAIL
     *   - the factor moving off 4 changes the scale relationship             -> FAIL
     * Whether the factor SHOULD be 4 is tracked in the ledger; it is invisible
     * in production because these scores feed only a top-k, and a uniform
     * positive scale cannot reorder a ranking. */
    const double kTierScale = 1.0;   /* L137: the tier reproduces storage semantics exactly */

    double max_rel = 0.0, sum_abs = 0.0, sum_ref = 0.0;
    double ratio_lo = 1e300, ratio_hi = -1e300;
    size_t nfinite = 0, nbad = 0, nnan = 0;
    for (size_t i = 0; i < got.size(); i++) {
        const double r = ref[i], g = (double)got[i];
        if (std::isinf(r) && std::isinf(g) && (r < 0) == (g < 0)) continue;  /* masked */
        if (std::isnan(g)) { nnan++; continue; }
        if (!std::isfinite(r) || !std::isfinite(g)) { nbad++; continue; }
        nfinite++;
        if (std::fabs(r) > 1e-6) {
            const double q = r / g;            /* expected: kTierScale, every element */
            ratio_lo = std::min(ratio_lo, q);
            ratio_hi = std::max(ratio_hi, q);
        }
        const double rs = r / kTierScale;      /* compare against the scaled oracle */
        sum_abs += std::fabs(g - rs);
        sum_ref += std::fabs(rs);
        const double den = std::fabs(rs) > 1e-6 ? std::fabs(rs) : 1e-6;
        max_rel = std::max(max_rel, std::fabs(g - rs) / den);
    }
    printf("compared %zu finite elements (%zu masked-inf skipped, %zu NaN, %zu mismatch-kind)\n",
           nfinite, got.size() - nfinite - nnan - nbad, nnan, nbad);
    printf("mean |delta| / mean |ref| = %.3e   (vs oracle/%.0f)\n",
           sum_ref > 0 ? sum_abs / sum_ref : 0.0, kTierScale);
    printf("max relative error        = %.3e\n", max_rel);
    printf("ref/got ratio spread      = [%.6f, %.6f]  (expect %.1f, uniform)\n",
           ratio_lo, ratio_hi, kTierScale);

    /* ---- timing at the engine's real launch shape ------------------------
     * The engine issues 168 launches per 4096-token prefill (four 512-token
     * sub-batches x 42 layers) and the old wmma128 scorer costs 92.38 ms total,
     * i.e. 0.550 ms per launch on average as n_comp ramps 32 -> 512+.  Time the
     * same per-launch work here.  This is a kernel-level number, not an
     * end-to-end one -- it does not include the Q packing the engine would
     * still have to do outside this kernel. */
    /* The "pack alone" timing block that stood here launched idx_pack_q_kernel
     * over an f32 Q buffer.  Both are gone: the producer emits packed Q, so
     * there is no pack step in this path to time.  The nearest surviving
     * fixed-cost term is idx_expand_q_kernel, which the scorer runs internally
     * and which is therefore already inside the per-launch number below. */

    {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        const int iters = 50;
        for (int i = 0; i < 5; i++)   /* warm */
            pulsar_gpu_indexer_scores_mxfp4(d_scores,
                                            (const pulsar_mxkv_pack_t *)d_q, d_w,
                                            (const pulsar_mxkv_pack_t *)d_comp, n_comp,
                                            n_tokens, pos0, IDX_HEADS, IDX_HEAD_DIM,
                                            ratio, scale, causal);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int i = 0; i < iters; i++)
            pulsar_gpu_indexer_scores_mxfp4(d_scores,
                                            (const pulsar_mxkv_pack_t *)d_q, d_w,
                                            (const pulsar_mxkv_pack_t *)d_comp, n_comp,
                                            n_tokens, pos0, IDX_HEADS, IDX_HEAD_DIM,
                                            ratio, scale, causal);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
        printf("timing: %.4f ms/launch  (n_comp=%u n_tokens=%u)\n", ms / iters, n_comp, n_tokens);
    }

    /* Uniformity is the layout alarm: a fragment or lane-map bug cannot leave
     * the ratio constant.  The band is tight (1e-3) because the ratio is a
     * power of two and should be exact to f32 rounding. */
    const int ratio_ok = (ratio_lo > kTierScale * (1.0 - 1e-3) &&
                          ratio_hi < kTierScale * (1.0 + 1e-3));
    if (!ratio_ok)
        printf("  RATIO NOT UNIFORM AT %.1f: [%.6f, %.6f] -- layout/lane-map suspect\n",
               kTierScale, ratio_lo, ratio_hi);
    const int pass = (nnan == 0 && nbad == 0 && max_rel < 1e-3 && ratio_ok);
    printf("\nKERNEL TEST: %s\n", pass ? "PASS" : "FAIL");
    if (!pass) {
        printf("  first few (ref vs got):\n");
        for (size_t i = 0; i < 8 && i < got.size(); i++)
            printf("    [%zu] %14.6f  %14.6f\n", i, ref[i], (double)got[i]);
    }
    return pass ? 0 : 1;
}

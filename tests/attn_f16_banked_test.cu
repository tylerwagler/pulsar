/* Bank-isolation gate for the fp16 indexed attention kernel.
 *
 * The banked descriptor path is the MULTI-SEQUENCE one: raw rows live at
 * seq_id[t]*raw_cap inside a shared ring, and compressed rows come from a
 * per-sequence base pointer.  Getting the base or the modulo wrong does not
 * crash and does not produce NaN -- it reads ANOTHER SEQUENCE's KV, which is
 * valid memory and plausible attention.  One conversation leaking into another
 * is the failure mode, so it needs a test that cannot be fooled by my own
 * misunderstanding of the index math.
 *
 * An f64 oracle CANNOT provide that.  If I transcribe the descriptor contract
 * wrongly I transcribe it wrongly into the oracle too, both agree, and the test
 * passes while the kernel attends to the wrong sequence.  A test is only as
 * independent as the understanding behind it.
 *
 * SO THE CHECK IS ALGEBRAIC INSTEAD.  Fill bank b -- raw rows AND comp rows --
 * with the CONSTANT vector v_b.  Attention output is a convex combination of
 * the value rows it selects, so if it reads only bank b the result is exactly
 * v_b, whatever the scores, the softmax, the ring offset, or which top-k rows
 * were chosen.  Reading any other bank mixes in v_b' and moves the answer.
 * The assertion therefore tests bank ISOLATION directly and depends on no
 * index arithmetic of mine at all.
 *
 * Sinks are set far negative so the sink term drops out of the denominator;
 * otherwise the output would be v_b scaled by S/(S+sink) and the clean
 * equality would be lost.
 *
 * WHY THE MATCH IS EXACT, AND WHAT THAT MEANS.  A constant bank makes every KV
 * row identical, so every score is identical, so every softmax weight is
 * exp(0) = 1 -- which is exact in fp16.  The result is therefore v_b to the
 * last bit rather than to ~1e-4.  That is a property of the construction, not
 * extra evidence of correctness: this test proves BANK ISOLATION and nothing
 * about varied softmax weights.  Varied weights are what
 * tests/attn_f16_kernel_test.cu covers in indexed mode against the f64 oracle.
 * The two are complementary and neither alone is sufficient.
 *
 * The detector still works under that degeneracy: a wrong-bank read pulls in
 * v_b', whose score differs, so the output becomes a mixture; and a read that
 * is wrong for EVERY row lands on some other constant v_b'' != v_b.
 *
 * Also gated: an evicted row (seq_id[t] >= n_banks) must zero its heads.
 *
 * build+run:
 *   nvcc -O3 -arch=sm_120f -Isrc -Isrc/cuda -o /tmp/attnbk tests/attn_f16_banked_test.cu
 *   /tmp/attnbk
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

int main(void) {
    const uint32_t D = AF16_DIM, n_head = 32u;
    const uint32_t n_banks = 4u, raw_cap = 64u, comp_cap = 32u;
    const uint32_t n_tokens = 16u, top_k = 8u, window = 24u, ratio = 2u;
    const uint32_t n_comp = comp_cap, n_raw = raw_cap;

    printf("attn f16 BANK ISOLATION test: %u banks, raw_cap=%u comp_cap=%u,"
           " %u tokens x %u heads\n\n", n_banks, raw_cap, comp_cap, n_tokens, n_head);

    /* bank b is the constant v_b, everywhere: raw ring slice AND comp slice */
    auto vb = [](uint32_t b) { return 0.25f * (float)(b + 1u); };

    std::vector<float> raw((size_t)n_banks * raw_cap * D);
    std::vector<float> comp((size_t)n_banks * comp_cap * D);
    for (uint32_t b = 0; b < n_banks; b++) {
        for (uint32_t r = 0; r < raw_cap; r++)
            for (uint32_t d = 0; d < D; d++)
                raw[((size_t)b * raw_cap + r) * D + d] = vb(b);
        for (uint32_t r = 0; r < comp_cap; r++)
            for (uint32_t d = 0; d < D; d++)
                comp[((size_t)b * comp_cap + r) * D + d] = vb(b);
    }

    std::mt19937_64 rng(20260808);
    std::normal_distribution<float> nd(0.f, 0.5f);
    std::vector<float> q((size_t)n_tokens * n_head * D), sinks(n_head, -100.0f),
                       out((size_t)n_tokens * n_head * D, -12345.f);
    for (auto &v : q) v = nd(rng);

    /* token -> bank, plus one deliberately EVICTED row */
    const uint32_t evicted = 5u;
    std::vector<int32_t> seq(n_tokens), pos(n_tokens), tk((size_t)n_tokens * top_k);
    for (uint32_t t = 0; t < n_tokens; t++) {
        seq[t] = (t == evicted) ? (int32_t)n_banks : (int32_t)(t % n_banks);
        pos[t] = (int32_t)(20u + t);                 /* varied, > window */
        for (uint32_t i = 0; i < top_k; i++)
            tk[(size_t)t * top_k + i] = (int32_t)((t * 5u + i * 3u) % comp_cap);
    }

    float *dq, *draw, *dcomp, *ds, *dout; int32_t *dtk, *dpos, *dseq;
    cudaMalloc(&dq, q.size()*4); cudaMalloc(&draw, raw.size()*4);
    cudaMalloc(&dcomp, comp.size()*4); cudaMalloc(&ds, sinks.size()*4);
    cudaMalloc(&dout, out.size()*4); cudaMalloc(&dtk, tk.size()*4);
    cudaMalloc(&dpos, pos.size()*4); cudaMalloc(&dseq, seq.size()*4);
    cudaMemcpy(dq, q.data(), q.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(draw, raw.data(), raw.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dcomp, comp.data(), comp.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, sinks.data(), sinks.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dout, out.data(), out.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dtk, tk.data(), tk.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dpos, pos.data(), pos.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dseq, seq.data(), seq.size()*4, cudaMemcpyHostToDevice);

    /* comp_bank_ptrs: one base pointer per bank, the shape the engine passes */
    std::vector<const void *> hbp(n_banks);
    for (uint32_t b = 0; b < n_banks; b++) hbp[b] = dcomp + (size_t)b * comp_cap * D;
    const void **dbp = NULL;
    cudaMalloc(&dbp, n_banks * sizeof(void *));
    cudaMemcpy(dbp, hbp.data(), n_banks * sizeof(void *), cudaMemcpyHostToDevice);

    const int rc = pulsar_gpu_attention_f16_indexed(
        dout, ds, dq, draw, dcomp, dtk, n_tokens, /*pos0*/0u, n_raw, raw_cap,
        /*raw_start*/0u, n_comp, top_k, window, ratio, n_head, D, /*raw_f16*/0,
        dpos, dseq, dbp, comp_cap, n_banks);
    if (!rc) { printf("LAUNCH REFUSED\n"); return 1; }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("EXEC FAILED: %s\n", cudaGetErrorString(cudaGetLastError())); return 1;
    }
    cudaMemcpy(out.data(), dout, out.size()*4, cudaMemcpyDeviceToHost);

    double worst_leak = 0.0; uint32_t leak_tok = 0, leak_head = 0;
    size_t evict_bad = 0, untouched = 0, nan = 0;
    for (uint32_t t = 0; t < n_tokens; t++) {
        const int ev = (t == evicted);
        const float want = ev ? 0.0f : vb((uint32_t)seq[t]);
        for (uint32_t h = 0; h < n_head; h++)
            for (uint32_t d = 0; d < D; d++) {
                const float g = out[((size_t)t * n_head + h) * D + d];
                if (std::isnan(g)) { nan++; continue; }
                if (g == -12345.f) { untouched++; continue; }
                const double e = std::fabs((double)g - want);
                if (ev) { if (e > 0.0) evict_bad++; continue; }
                if (e > worst_leak) { worst_leak = e; leak_tok = t; leak_head = h; }
            }
    }
    printf("worst |out - v_bank| = %.3e  (token %u, head %u)\n",
           worst_leak, leak_tok, leak_head);
    printf("evicted row non-zero elements = %zu\n", evict_bad);
    printf("NaN = %zu, never-written = %zu\n", nan, untouched);
    printf("  bank constants: ");
    for (uint32_t b = 0; b < n_banks; b++) printf("v%u=%.2f ", b, vb(b));
    printf("\n  a cross-bank read moves the answer by >= %.2f\n", 0.25f);

    const int pass = (nan == 0 && untouched == 0 && evict_bad == 0 && worst_leak < 1e-2);
    printf("\nATTN F16 BANK ISOLATION: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

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
#include "attn_pack_fixture.h"

/* Encode a CONSTANT row in ATTN_PACK layout.  0.25/0.5/0.75/1.0 are all exact
 * in E4M3 with a unit block scale, so the packed row decodes to exactly the
 * same constant and the isolation assertion below is unchanged -- it now also
 * exercises packed row ADDRESSING (row stride, scale offset, rope offset).
 * The FORMAT itself is not at risk here: the kernel decodes through the same
 * attn_comp_pack_ld the f32 kernel uses, so there is one implementation. */
static void pack_const_row(uint8_t *dst, float v, uint32_t head_dim) {
    const uint32_t n_nope = head_dim - PULSAR_ATTN_PACK_NROT;
    /* Exact NV construction: row scale = v (f32, exact), every per-16 scale
     * code = e4m3(1.0), every nibble = e2m1(1.0) -> each nope element decodes
     * to exactly 1.0 * 1.0 * v = v.  The rope tail narrows v to bf16, exact
     * whenever v is exact in e4m3 (3 mantissa bits <= bf16's 7). */
    const uint32_t nib = n_nope / 2u, nblk = n_nope / 16u;
    for (uint32_t i = 0; i < nib; i++) dst[i] = (uint8_t)(2u | (2u << 4)); /* e2m1 1.0 pairs */
    for (uint32_t i = 0; i < nblk; i++) dst[nib + i] = (uint8_t)(7u << 3); /* e4m3 1.0 */
    memcpy(dst + nib + nblk, &v, sizeof v);                               /* row scale = v */
    __nv_bfloat16 *rope = (__nv_bfloat16 *)(dst + nib + nblk + 4u);
    for (uint32_t i = 0; i < PULSAR_ATTN_PACK_NROT; i++) rope[i] = __float2bfloat16(v);
}

int main(int argc, char **argv) {
    /* These fixtures and CPU references are E4M3-only; a stray PULSAR_KV4 in the
     * environment would make the launch dispatch decode E4M3 rows as nibbles and
     * fail confusingly.  Pin the format rather than inherit it. */

    /* Comp banks are ATTN_PACK, full stop.  This took a 'p' argument selecting
     * between packed and f32 banks until 2026-08-18, when the format parameter
     * was removed from the kernels -- there is one comp format, so there is one
     * mode.  The argument is still accepted and ignored so old invocations do
     * not fail. */
    (void)argc; (void)argv;
    /* L106 K10: a `packed` flag pinned to 1 kept an f32 comp arm alive that
     * built and uploaded a full slab and would MIS-STRIDE (2048 vs 584 B/row)
     * against the packed reader if ever unpinned.  The arm is gone; this test
     * exercises the shipped ATTN_PACK format only. */
    const uint32_t D = AF16_DIM, n_head = 32u;
    const uint32_t n_banks = 4u, raw_cap = 64u, comp_cap = 32u;
    const uint32_t n_tokens = 16u, top_k = 8u, window = 24u, ratio = 2u;
    const uint32_t n_comp = comp_cap, n_raw = raw_cap;

    printf("attn f16 BANK ISOLATION test: %u banks, raw_cap=%u comp_cap=%u,"
           " %u tokens x %u heads, comp=%s\n\n", n_banks, raw_cap, comp_cap,
           n_tokens, n_head, "ATTN_PACK");

    /* bank b is the constant v_b, everywhere: raw ring slice AND comp slice */
    auto vb = [](uint32_t b) { return 0.25f * (float)(b + 1u); };

    const size_t pack_row_b = (size_t)PULSAR_ATTN_PACK_ROWBYTES(D);
    std::vector<uint8_t> rawp((size_t)n_banks * raw_cap * pack_row_b);
    for (uint32_t b = 0; b < n_banks; b++) {
        for (uint32_t r = 0; r < raw_cap; r++)
            pack_const_row(&rawp[((size_t)b * raw_cap + r) * pack_row_b], vb(b), D);
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

    /* packed comp banks: same constants, ATTN_PACK layout */
    const uint64_t prow = PULSAR_ATTN_PACK_ROWBYTES(D);
    std::vector<uint8_t> pcomp((size_t)n_banks * comp_cap * prow, 0);
        for (uint32_t b = 0; b < n_banks; b++)
            for (uint32_t r = 0; r < comp_cap; r++)
                pack_const_row(&pcomp[((size_t)b * comp_cap + r) * prow], vb(b), D);

    pulsar_q_t *dq;
    float *draw, *ds; int32_t *dtk, *dpos, *dseq;
    /* dout carries the STORED heads type (L033); the sentinel below must
     * round-trip through it, or a bf16 width turns every never-written slot
     * into a false "written garbage" (-12345 is not representable in bf16). */
    pulsar_heads_t *dout;
    std::vector<pulsar_heads_t> out_h(out.size());
    const float kSent = (float)(pulsar_heads_t)(-12345.f);
    dq = fixture_upload_q(q); cudaMalloc(&draw, rawp.size()); cudaMalloc(&ds, sinks.size()*4);
    cudaMalloc(&dout, out.size()*sizeof(pulsar_heads_t)); cudaMalloc(&dtk, tk.size()*4);
    cudaMalloc(&dpos, pos.size()*4); cudaMalloc(&dseq, seq.size()*4);
    cudaMemcpy(draw, rawp.data(), rawp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, sinks.data(), sinks.size()*4, cudaMemcpyHostToDevice);
    for (size_t i = 0; i < out.size(); i++) out_h[i] = (pulsar_heads_t)out[i];
    cudaMemcpy(dout, out_h.data(), out_h.size()*sizeof(pulsar_heads_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dtk, tk.data(), tk.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dpos, pos.data(), pos.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dseq, seq.data(), seq.size()*4, cudaMemcpyHostToDevice);

    /* comp_bank_ptrs: one base pointer per bank, the shape the engine passes */
    uint8_t *dpk = NULL;
    cudaMalloc(&dpk, pcomp.size());
    cudaMemcpy(dpk, pcomp.data(), pcomp.size(), cudaMemcpyHostToDevice);
    std::vector<const void *> hbp(n_banks);
    for (uint32_t b = 0; b < n_banks; b++)
        hbp[b] = (const void *)(dpk + (size_t)b * comp_cap * prow);
    const void **dbp = NULL;
    cudaMalloc(&dbp, n_banks * sizeof(void *));
    cudaMemcpy(dbp, hbp.data(), n_banks * sizeof(void *), cudaMemcpyHostToDevice);

    /* Two selection modes share this kernel and BOTH must be bank-isolated:
     *   topk != NULL  a top-k selection      (the indexed prefill/decode path)
     *   topk == NULL  visible-prefix sweep   (attention_decode_batch_launch's
     *                                         continued-prefill call)
     * Only the first was covered until 2026-08-15; the second is the mode a
     * banked continued-prefill batch actually takes.  Every comp row in bank b
     * holds v_b, so both modes must return v_b for a correctly isolated read. */
    int overall = 1;
    for (int mode = 0; mode < 2; mode++) {
        const int32_t *use_tk = mode ? NULL : dtk;
        const uint32_t use_topk = mode ? 0u : top_k;
        const char *label = mode ? "visible-prefix sweep (topk=NULL)" : "top-k selection";
        printf("---- %s ----\n", label);
        cudaMemset(dout, 0, out.size() * sizeof(pulsar_heads_t));
        std::fill(out.begin(), out.end(), kSent);
        for (size_t i = 0; i < out.size(); i++) out_h[i] = (pulsar_heads_t)out[i];
        cudaMemcpy(dout, out_h.data(), out_h.size() * sizeof(pulsar_heads_t), cudaMemcpyHostToDevice);

    const int rc = pulsar_gpu_attention_f16_indexed(
        dout, ds, dq, (const pulsar_attn_pack_t *)draw,
        (const pulsar_attn_pack_t *)dpk, use_tk, n_tokens,
        /*pos0*/0u, n_raw, raw_cap, /*raw_start*/0u, n_comp, use_topk, window, ratio,
        n_head, D, dpos, dseq, dbp, comp_cap, n_banks, NULL);
    if (!rc) { printf("LAUNCH REFUSED\n"); return 1; }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("EXEC FAILED: %s\n", cudaGetErrorString(cudaGetLastError())); return 1;
    }
    cudaMemcpy(out_h.data(), dout, out_h.size()*sizeof(pulsar_heads_t), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < out.size(); i++) out[i] = (float)out_h[i];

    double worst_leak = 0.0; uint32_t leak_tok = 0, leak_head = 0;
    size_t evict_bad = 0, untouched = 0, nan = 0;
    for (uint32_t t = 0; t < n_tokens; t++) {
        const int ev = (t == evicted);
        const float want = ev ? 0.0f : vb((uint32_t)seq[t]);
        for (uint32_t h = 0; h < n_head; h++)
            for (uint32_t d = 0; d < D; d++) {
                const float g = out[((size_t)t * n_head + h) * D + d];
                if (std::isnan(g)) { nan++; continue; }
                if (g == kSent) { untouched++; continue; }
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
    printf("  a cross-bank read moves the answer by >= %.2f\n", 0.25f);

    const int pass = (nan == 0 && untouched == 0 && evict_bad == 0 && worst_leak < 1e-2);
    printf("  %s: %s\n\n", label, pass ? "PASS" : "FAIL");
    if (!pass) overall = 0;
    }

    printf("ATTN F16 BANK ISOLATION: %s\n", overall ? "PASS" : "FAIL");
    return overall ? 0 : 1;
}

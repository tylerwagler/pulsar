/* Split-KV decode attention gate: the split walk + softmax merge must agree
 * with the single-walk heads8-online kernel on the SAME inputs, across every
 * staging branch (raw f32/f16, comp f32/fp8), both descriptor modes
 * (single_all scalar decode; positions/seq_id banked with an evicted row),
 * and the edge shapes (n_score < n_split, comp_count == 0).
 *
 * WHAT THIS TEST PROVES -- AND WHAT IT DOESN'T.  Golden here is the
 * PRODUCTION single-walk kernel, not an oracle: if my understanding of the
 * kernel's input contract were wrong, both sides would see the same wrongly-
 * built inputs and still agree.  So this test proves the split REFACTOR is
 * value-preserving (up to softmax-merge reassociation), and engine-level
 * teacher-forced KL against pre-change dumps proves the integration reads
 * real caches correctly.  Neither alone is sufficient; together they close.
 *
 * Agreement is to reassociation tolerance, NOT bit-exact: the merge re-folds
 * per-slice (m, l, o) with exp(m_i - m*) weights.  Worst per-(token,head)
 * relative L2 must stay under 1e-5 (f32 epsilon-class); a real indexing bug
 * moves entire value rows and shows up at 1e-1-class, a 10,000x margin.
 *
 * build+run (sparky):
 *   nvcc -O3 -arch=sm_120f -Isrc -Isrc/cuda -o /tmp/attnsplit \
 *       tests/attn_decode_split_test.cu
 *   /tmp/attnsplit
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>

int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "cuda error (%s): %s\n", what, cudaGetErrorString(err));
    return 0;
}
#include "../src/cuda/pulsar_cuda_attention.cu"

/* Link stubs: the include above pulls in every launcher in the attention TU,
 * whose call graph reaches runtime/cublas/f16 symbols this test never
 * executes.  Stubs satisfy the linker; any accidental call fails loudly. */
cublasHandle_t g_cublas;
int g_cublas_ready = 0;
int g_quality_mode = 0;
cublasLtHandle_t g_cublaslt;
void *cuda_tmp_alloc(uint64_t, const char *what) {
    fprintf(stderr, "stub cuda_tmp_alloc hit (%s)\n", what); exit(97);
}
const char *cuda_model_range_ptr(const void *, uint64_t, uint64_t, const char *what) {
    fprintf(stderr, "stub cuda_model_range_ptr hit (%s)\n", what); exit(97);
}
int cublas_ok(cublasStatus_t, const char *what) {
    fprintf(stderr, "stub cublas_ok hit (%s)\n", what); exit(97);
}
int pulsar_gpu_attention_prefill_reads_packed_comp(void) { return 0; }
int pulsar_gpu_attention_f16_indexed(float *, const float *, const float *,
        const float *, const float *, const int *, uint32_t, uint32_t, uint32_t,
        uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
        uint32_t, int, const int *, const int *, const void * const *, uint32_t,
        uint32_t, int) {
    fprintf(stderr, "stub f16_indexed hit\n"); exit(97);
}
int pulsar_gpu_attention_f16_prefill(float *, const float *, const float *,
        const float *, const float *, uint32_t, uint32_t, uint32_t, uint32_t,
        uint32_t, uint32_t, int) {
    fprintf(stderr, "stub f16_prefill hit\n"); exit(97);
}

struct Case {
    const char *name;
    uint32_t n_tokens, n_raw, raw_cap, raw_start, n_comp, window, ratio;
    uint32_t pos_base;      /* positions[t] = pos_base + 3*t (descr mode) */
    int descr;              /* 0: single_all scalar decode; 1: banked */
    int raw_f16, comp_fp8, comp_pack;
    uint32_t n_banks, comp_cap;
};

int main() {
    const uint32_t D = 512u, n_head = 64u;
    std::mt19937_64 rng(20260809);
    std::normal_distribution<float> nd(0.f, 0.8f);

    const Case cases[] = {
        {"single_all f32comp",  1, 128, 4352, 17, 600, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        {"single_all fp8comp",  1, 128, 4352, 90, 512, 0, 0, 0, 0, 0, 1, 0, 1, 0},
        {"single_all raw-f16",  1, 128, 4352,  5, 640, 0, 0, 0, 0, 1, 0, 0, 1, 0},
        {"single_all pack",     1, 128, 4352, 41, 600, 0, 0, 0, 0, 0, 0, 1, 1, 0},
        {"banked 4tok",         4,  64,  64,  0, 200, 24, 4, 850, 1, 0, 0, 0, 2, 256},
        {"banked evict+fp8",    6,  64,  64,  0, 220, 24, 4, 900, 1, 0, 1, 0, 2, 256},
        {"banked pack",         4,  64,  64,  0, 200, 24, 4, 870, 1, 0, 0, 1, 2, 256},
        {"tiny n_score",        1,   3,  64, 10,   1, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        {"comp empty",          2,  40,  64,  0,   0, 48, 4, 100, 1, 0, 0, 0, 1, 64},
    };

    double worst_all = 0.0;
    int fail = 0;
    for (const Case &c : cases) {
        const uint32_t total_raw_rows = (c.descr ? c.n_banks : 1u) * c.raw_cap;
        const uint32_t total_comp_rows = (c.descr ? c.n_banks : 1u) *
                                         (c.descr ? c.comp_cap : c.n_comp);
        std::vector<float> raw((size_t)total_raw_rows * D);
        std::vector<float> comp((size_t)(total_comp_rows ? total_comp_rows : 1) * D);
        std::vector<float> q((size_t)c.n_tokens * n_head * D);
        std::vector<float> sinks(n_head);
        for (auto &v : raw) v = nd(rng);
        for (auto &v : comp) v = nd(rng);
        for (auto &v : q) v = nd(rng) * 0.25f;
        for (auto &v : sinks) v = nd(rng);   /* sinks in-distribution: the merge
                                                applies them once; a double- or
                                                un-applied sink shifts outputs */

        /* fp8 comp: build byte rows + per-64 scales; both kernels run the
         * same dequant, agreement is what's under test. */
        const uint32_t fp8_row = PULSAR_FP8_KV_ROWBYTES(D);
        std::vector<uint8_t> comp8((size_t)(total_comp_rows ? total_comp_rows : 1) * fp8_row);
        if (c.comp_fp8) {
            std::uniform_int_distribution<int> bd(0, 255);
            for (size_t r = 0; r < total_comp_rows; r++) {
                uint8_t *row = &comp8[r * fp8_row];
                for (uint32_t d = 0; d < D; d++) {
                    int b = bd(rng);
                    if ((b & 0x78) == 0x78) b &= ~0x40;      /* no NaN/Inf e4m3 */
                    row[d] = (uint8_t)b;
                }
                float *sc = (float *)(row + D);
                for (uint32_t s = 0; s < D / 64u; s++) sc[s] = 0.5f + 0.25f * (float)(s % 3);
            }
        }

        /* ATTN_PACK rows: 448 e4m3 bytes + 7 E8M0 scales + pad + 64 f32 rope.
         * Random VALID bytes -- attn_pack_e4m3 is total (no NaN encodings in
         * its bit math), so any byte pattern decodes deterministically and
         * golden-vs-split agreement is what's under test. */
        const uint64_t pack_row = PULSAR_ATTN_PACK_ROWBYTES(D);
        std::vector<uint8_t> compp((size_t)(total_comp_rows ? total_comp_rows : 1) * pack_row);
        if (c.comp_pack) {
            std::uniform_int_distribution<int> bd(0, 255);
            const uint32_t n_nope = D - PULSAR_ATTN_PACK_NROT;
            for (size_t r = 0; r < total_comp_rows; r++) {
                uint8_t *row = &compp[r * pack_row];
                for (uint32_t d = 0; d < n_nope; d++) row[d] = (uint8_t)bd(rng);
                for (uint32_t s = 0; s < PULSAR_ATTN_PACK_SCALES_PAD(D); s++)
                    row[n_nope + s] = (uint8_t)(120 + (int)(r + s) % 14);
                float *rope = (float *)(row + n_nope + PULSAR_ATTN_PACK_SCALES_PAD(D));
                for (uint32_t i = 0; i < PULSAR_ATTN_PACK_NROT; i++) rope[i] = nd(rng);
            }
        }

        /* raw f16 rows: kernel reads halves when raw_f16 -- reuse raw[] as
         * the value source, store as __half. */
        std::vector<__half> rawh((size_t)total_raw_rows * D);
        if (c.raw_f16)
            for (size_t i = 0; i < rawh.size(); i++) rawh[i] = __float2half(raw[i]);

        std::vector<int32_t> pos(c.n_tokens), seq(c.n_tokens);
        for (uint32_t t = 0; t < c.n_tokens; t++) {
            pos[t] = (int32_t)(c.pos_base + 3u * t);
            seq[t] = (int32_t)(t % (c.n_banks ? c.n_banks : 1u));
        }
        if (c.name[7] == 'e' && c.n_tokens >= 3) seq[2] = (int32_t)c.n_banks; /* evicted */

        float *draw, *dcomp, *dq, *ds, *dgold, *dsplit;
        int32_t *dpos = NULL, *dseq = NULL;
        const void **dbp = NULL;
        cudaMalloc(&draw, c.raw_f16 ? rawh.size() * 2 : raw.size() * 4);
        cudaMalloc(&dcomp, c.comp_fp8 ? comp8.size()
                         : c.comp_pack ? compp.size() : comp.size() * 4);
        cudaMalloc(&dq, q.size() * 4);
        cudaMalloc(&ds, sinks.size() * 4);
        cudaMalloc(&dgold, q.size() * 4);
        cudaMalloc(&dsplit, q.size() * 4);
        if (c.raw_f16) cudaMemcpy(draw, rawh.data(), rawh.size() * 2, cudaMemcpyHostToDevice);
        else           cudaMemcpy(draw, raw.data(), raw.size() * 4, cudaMemcpyHostToDevice);
        if (c.comp_fp8)       cudaMemcpy(dcomp, comp8.data(), comp8.size(), cudaMemcpyHostToDevice);
        else if (c.comp_pack) cudaMemcpy(dcomp, compp.data(), compp.size(), cudaMemcpyHostToDevice);
        else                  cudaMemcpy(dcomp, comp.data(), comp.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(ds, sinks.data(), sinks.size() * 4, cudaMemcpyHostToDevice);
        cudaMemset(dgold, 0xe5, q.size() * 4);
        cudaMemset(dsplit, 0x5e, q.size() * 4);
        if (c.descr) {
            cudaMalloc(&dpos, pos.size() * 4);
            cudaMalloc(&dseq, seq.size() * 4);
            cudaMemcpy(dpos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dseq, seq.data(), seq.size() * 4, cudaMemcpyHostToDevice);
            std::vector<const void *> hbp(c.n_banks);
            const size_t bank_bytes = c.comp_fp8 ? (size_t)c.comp_cap * fp8_row
                                    : c.comp_pack ? (size_t)c.comp_cap * pack_row
                                                  : (size_t)c.comp_cap * D * 4;
            for (uint32_t b = 0; b < c.n_banks; b++)
                hbp[b] = (const uint8_t *)dcomp + b * bank_bytes;
            cudaMalloc(&dbp, c.n_banks * sizeof(void *));
            cudaMemcpy(dbp, hbp.data(), c.n_banks * sizeof(void *), cudaMemcpyHostToDevice);
        }

        const uint32_t hg = (n_head + 7u) / 8u;
        /* golden: single walk (z=1, no partials) */
        dim3 g1(c.n_tokens, hg, 1);
        attention_decode_mixed_heads8_online_kernel<<<g1, 256>>>(dgold, ds, dq,
                (const float *)draw, (const float *)dcomp, 0, c.comp_fp8, c.comp_pack,
                c.n_tokens, 0, c.n_raw, c.raw_cap, c.raw_start, c.n_comp,
                c.window, c.ratio, n_head, D, c.raw_f16,
                dpos, dseq, (const void * const *)dbp,
                c.comp_cap, c.descr ? c.n_banks : 1u, NULL, NULL);
        /* candidate: split walk + merge through the production scratch */
        float *po, *pml;
        if (!cuda_ok(cudaGetSymbolAddress((void **)&po, g_dec_splitkv_part_o), "sym o") ||
            !cuda_ok(cudaGetSymbolAddress((void **)&pml, g_dec_splitkv_part_ml), "sym ml"))
            return 1;
        dim3 gs(c.n_tokens, hg, PULSAR_DEC_SPLITKV_S);
        attention_decode_mixed_heads8_online_kernel<<<gs, 256>>>(dsplit, ds, dq,
                (const float *)draw, (const float *)dcomp, 0, c.comp_fp8, c.comp_pack,
                c.n_tokens, 0, c.n_raw, c.raw_cap, c.raw_start, c.n_comp,
                c.window, c.ratio, n_head, D, c.raw_f16,
                dpos, dseq, (const void * const *)dbp,
                c.comp_cap, c.descr ? c.n_banks : 1u, po, pml);
        dim3 gm(c.n_tokens, n_head, 1);
        attention_decode_split_merge_kernel<<<gm, 128>>>(dsplit, ds, po, pml,
                PULSAR_DEC_SPLITKV_S, n_head, D);
        if (!cuda_ok(cudaDeviceSynchronize(), c.name)) return 1;

        std::vector<float> gold(q.size()), split(q.size());
        cudaMemcpy(gold.data(), dgold, gold.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(split.data(), dsplit, split.size() * 4, cudaMemcpyDeviceToHost);

        double worst = 0.0; uint32_t wt = 0, wh = 0; size_t nans = 0;
        for (uint32_t t = 0; t < c.n_tokens; t++)
            for (uint32_t h = 0; h < n_head; h++) {
                double e2 = 0.0, n2 = 0.0;
                for (uint32_t d = 0; d < D; d++) {
                    const float a = gold[((size_t)t * n_head + h) * D + d];
                    const float b = split[((size_t)t * n_head + h) * D + d];
                    if (std::isnan(a) || std::isnan(b)) { nans++; continue; }
                    e2 += (double)(a - b) * (a - b);
                    n2 += (double)a * a;
                }
                const double rel = std::sqrt(e2 / (n2 > 1e-30 ? n2 : 1e-30));
                if (rel > worst) { worst = rel; wt = t; wh = h; }
            }
        const int ok = (nans == 0 && worst < 1e-5);
        printf("%-20s worst rel L2 %.3e (tok %u head %u) nan=%zu  %s\n",
               c.name, worst, wt, wh, nans, ok ? "ok" : "FAIL");
        if (!ok) fail = 1;
        if (worst > worst_all) worst_all = worst;

        cudaFree(draw); cudaFree(dcomp); cudaFree(dq); cudaFree(ds);
        cudaFree(dgold); cudaFree(dsplit);
        if (dpos) cudaFree(dpos);
        if (dseq) cudaFree(dseq);
        if (dbp) cudaFree(dbp);
    }
    printf("\nATTN DECODE SPLIT-KV: %s (worst %.3e)\n", fail ? "FAIL" : "PASS", worst_all);
    return fail;
}

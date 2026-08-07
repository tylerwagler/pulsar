/* Prices the ONE remaining input-side gap in the MMQ port (plan 41b).
 *
 * We are 1.43x slower per mul_mat_q call than Palaferri on the SAME kernel,
 * same instantiation, same shapes -- and with BETTER locality, since our REAP25
 * model is [4096,2048,192] against their [4096,2048,256], i.e. 128 tokens per
 * expert vs 96 and 25% less weight data.  So the deficit is input-side.
 *
 * Prime suspect: block_iq2_xxs is 66 B with qs[] at offset 2, so a raw block
 * stream is only 2-byte aligned and nvcc must emit LDG.E.U16 -- two 16-bit
 * loads per 32-bit weight word (our own finding, 34fb67d).  Both competitors
 * feed MMQ an aligned artifact; we feed raw blocks.  MMQ is far more
 * weight-read-bound than our dp4a kernel (tensor cores make the MACs cheap),
 * so the penalty should hurt it much more than the 1.07x we measured on dp4a.
 *
 * A/B varies ONLY the weight layout: ds4_mmq_iq2_xxs_moe_pair (raw blocks) vs
 * ds4_mmq_iq2_xxs_moe_pair_soa (aligned), identical shapes, ids and activations.
 *
 *   nvcc -O3 --use_fast_math -arch=sm_120f -Isrc -Isrc/cuda/mmq \
 *        -o /tmp/mmq_align_bench tests/mmq_align_bench.cu \
 *        src/cuda/mmq/*.o -lcudart -lcublas
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#include "ds4_mmq.h"
#include "ds4_repack.h"

#define CHECK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

/* "no fold available" stub -- the q8-fold registry is a single-token decode
 * path the prefill entries never reach.  Same stub the vendored cuda/mmq/test
 * TUs use, and the same one pulsar_cuda_moe.cu provides in-engine. */
extern "C" int ds4_cuda_q8_fold_take_q81(const void *src, uint64_t in_dim, const void **q81) {
    (void)src; (void)in_dim; (void)q81;
    return 0;
}

/* production v5mx routed-expert shape */
static const int K        = 4096;   /* expert_in_dim  */
static const int M        = 2048;   /* expert_mid_dim */
static const int NEXP     = 192;    /* REAP25-pruned  */
static const int NUSED    = 6;

static double time_ms(void (*fn)(void *), void *ctx, int iters) {
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    fn(ctx); CHECK(cudaDeviceSynchronize());          /* warm */
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) fn(ctx);
    CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
    float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return ms / iters;
}

struct Ctx {
    const void *Wa, *Wb;      /* raw    */
    const void *WaS, *WbS;    /* soa    */
    const float *X; const int32_t *ids;
    float *outA, *outB;
    int n_tokens; int soa;
};
static void run(void *p) {
    Ctx *c = (Ctx *)p;
    if (c->soa)
        ds4_mmq_iq2_xxs_moe_pair_soa(c->WaS, c->WbS, c->X, c->ids, c->outA, c->outB,
                                     M, K, c->n_tokens, NEXP, NUSED, 0);
    else
        ds4_mmq_iq2_xxs_moe_pair(c->Wa, c->Wb, c->X, c->ids, c->outA, c->outB,
                                 M, K, c->n_tokens, NEXP, NUSED, 0);
}

int main(int argc, char **argv) {
    const int n_tokens = argc > 1 ? atoi(argv[1]) : 4096;
    const int iters    = argc > 2 ? atoi(argv[2]) : 5;
    CHECK(cudaSetDevice(0));
    if (ds4_mmq_init(0) != 0) { fprintf(stderr, "ds4_mmq_init failed\n"); return 1; }

    /* IQ2_XXS: 256 weights per 66 B block */
    const uint64_t raw_bytes = (uint64_t)NEXP * M * K / 256ull * 66ull;
    const uint64_t soa_bytes = ds4_mmq_iq2_xxs_aligned_bytes(M, K, NEXP);
    printf("shape K=%d M=%d experts=%d used=%d tokens=%d\n", K, M, NEXP, NUSED, n_tokens);
    printf("raw %.2f GiB/tensor   soa %.2f GiB/tensor\n",
           raw_bytes / 1073741824.0, soa_bytes / 1073741824.0);

    void *Wa, *Wb, *WaS, *WbS; float *X, *outA, *outB; int32_t *ids;
    CHECK(cudaMalloc(&Wa, raw_bytes)); CHECK(cudaMalloc(&Wb, raw_bytes));
    CHECK(cudaMalloc(&WaS, soa_bytes)); CHECK(cudaMalloc(&WbS, soa_bytes));
    CHECK(cudaMalloc(&X, (size_t)n_tokens * K * sizeof(float)));
    CHECK(cudaMalloc(&outA, (size_t)n_tokens * NUSED * M * sizeof(float)));
    CHECK(cudaMalloc(&outB, (size_t)n_tokens * NUSED * M * sizeof(float)));
    CHECK(cudaMalloc(&ids, (size_t)n_tokens * NUSED * sizeof(int32_t)));

    /* random weight bytes: timing depends on layout and routing, not values */
    std::mt19937 rng(1234);
    std::vector<uint8_t> h(1u << 22);
    for (auto &v : h) v = (uint8_t)rng();
    for (uint64_t o = 0; o < raw_bytes; o += h.size()) {
        size_t n = (size_t)std::min<uint64_t>(h.size(), raw_bytes - o);
        CHECK(cudaMemcpy((uint8_t *)Wa + o, h.data(), n, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy((uint8_t *)Wb + o, h.data(), n, cudaMemcpyHostToDevice));
    }
    std::vector<float> hx((size_t)n_tokens * K);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto &v : hx) v = d(rng);
    CHECK(cudaMemcpy(X, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<int32_t> hi((size_t)n_tokens * NUSED);
    std::uniform_int_distribution<int> de(0, NEXP - 1);
    for (auto &v : hi) v = de(rng);
    CHECK(cudaMemcpy(ids, hi.data(), hi.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    /* build the aligned artifacts from the same raw bytes */
    if (!ds4_repack_iq2_aligned_device(WaS, Wa, K, M, NEXP, 0) ||
        !ds4_repack_iq2_aligned_device(WbS, Wb, K, M, NEXP, 0)) {
        fprintf(stderr, "repack failed\n"); return 1;
    }
    CHECK(cudaDeviceSynchronize());

    /* Cost of BUILDING the aligned artifact.  We cannot hold one for every
     * layer (30 IQ2 layers x 3 tensors x 0.39 GiB ~ 35 GB, on top of an 86 GB
     * model in 121 GB), so the practical shapes are a per-layer reusable
     * scratch or a partial cache -- both of which pay this per tensor. */
    {
        cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
        ds4_repack_iq2_aligned_device(WaS, Wa, K, M, NEXP, 0);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(a));
        for (int i = 0; i < iters; i++) ds4_repack_iq2_aligned_device(WaS, Wa, K, M, NEXP, 0);
        CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
        float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, a, b));
        printf("  repack      %8.3f ms/tensor\n", ms / iters);
    }

    Ctx c{Wa, Wb, WaS, WbS, X, ids, outA, outB, n_tokens, 0};
    c.soa = 0; double raw_ms = time_ms(run, &c, iters);
    c.soa = 1; double soa_ms = time_ms(run, &c, iters);

    printf("\n  raw blocks  %8.3f ms/pair-call\n", raw_ms);
    printf("  aligned soa %8.3f ms/pair-call\n", soa_ms);
    printf("  speedup     %8.3fx\n", raw_ms / soa_ms);
    printf("\n  in-engine reference: ours 21.5 ms/call, Palaferri 15.0 ms/call (1.43x)\n");
    return 0;
}

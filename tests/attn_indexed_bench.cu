/* Prefill attention stall-reason harness.
 *
 * Attention is now the largest single prefill cost -- 1652 ms of 5160 at a
 * 4096-token prefill (32%), against Entrpi's 649 ms -- after the MoE work took
 * routed IQ2 from 4055 ms to 783.  nsys hardware counters on the shipped kernel
 * say:
 *     SMs Active 98.8%   SM Issue 41.9%   Tensor Active 0.0%   Warps 32.6%
 * i.e. the SMs are occupied but issuing on fewer than half of cycles, with zero
 * tensor-core use.  That is nearly the pre-D2R MoE signature (99.6/46.4/0.0/32.8),
 * and that kernel went on to yield 5.18x from tensor cores -- but signature is
 * NOT proof.  The same numbers once produced a confident "occupancy is the
 * limiter" read, and the streaming-K fix built on it went flat.
 *
 * gb20b exposes no DRAM counter, so nsys cannot separate memory-bound from
 * exec-dependency.  ncu can, but it fails (rc=9) inside a loaded 86 GB model and
 * works modelless.  Hence this TU.
 *
 * It #includes the shipped .cu so the file-static kernel is the REAL one, not a
 * copy.  (tests/iq2_row32_soa_diff.cu used the same pattern; it went with the
 * IQ2 dp4a kernels it diffed.)
 *
 * CALIBRATION GATE -- the whole point.  A synthetic attention bench is worthless
 * unless it reproduces the engine's own cost; the 2026-07-22 MoE NO-GO came from
 * a synthetic microbench that measured a different regime (it called the stage
 * DRAM-bandwidth-bound at 96% when the real engine had DRAM idle).  In-engine
 * this kernel runs 168 launches averaging 8.05 ms.  This harness prints its own
 * ms/launch next to that number and FAILS if it is not within 2x.  If the gate
 * fails, the stall breakdown taken from it means nothing -- fix the shape first.
 *
 * Shipped v5mx geometry (do not shrink it; the MoE lesson was that toy sizes
 * hide the effect): n_head 64, head_dim 512, top_k 512, window 128,
 * raw_cap 4352, comp_cap 1026, grid (n_tokens, (n_head+15)/16), block 512.
 *
 * Build (this TU replaces pulsar_cuda_attention.o):
 *   OBJS=$(ls src/cuda/*.o src/engine/*.o | grep -v pulsar_cuda_attention.o)
 *   nvcc -O3 --use_fast_math -arch=sm_120f \
 *        -Isrc -Isrc/cuda -Isrc/engine -o /tmp/attnbench \
 *        tests/attn_indexed_bench.cu $OBJS -lm -Xcompiler -pthread \
 *        -L$CUDA_HOME/targets/sbsa-linux/lib -lcudart -lcublas -lcublasLt
 *
 * Then, for the stall reasons this exists to get:
 *   sudo ncu --set full --kernel-name-base mangled \
 *        -k regex:attention_indexed_mixed_heads8 /tmp/attnbench 4096 3
 */
#include "../src/cuda/pulsar_cuda_attention.cu"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

/* In-engine reference, read off the trace rather than assumed:
 *   grid 512x4, i.e. n_tokens = 512 PER LAUNCH (not 4096 -- the chunk is split)
 *   168 launches, 2.730 .. 10.364 ms, avg 8.046, a 3.8x spread
 * The spread is n_comp growing with depth, so the AVERAGE is not a single shape
 * and calibrating to it is meaningless.  The gate instead sweeps n_comp and
 * requires the harness curve to BRACKET the engine's observed range. */
static const double kEngineMin = 2.730;
static const double kEngineMax = 10.364;
/* The engine's spread is a repeating 8-launch cycle -- eight 512-token
 * sub-batches per layer, ramping with CAUSAL DEPTH and plateauing once the
 * compressed rows reach top_k=512.  One observed layer, in order: */
static const double kEngineRamp[8] =
    {2.730, 4.922, 7.232, 8.870, 10.314, 9.965, 10.265, 9.982};

int main(int argc, char **argv) {
    const uint32_t n_tokens = argc > 1 ? (uint32_t)atoi(argv[1]) : 512u;  /* grid.x from the trace */
    const int      iters    = argc > 2 ? atoi(argv[2]) : 5;

    const uint32_t n_head   = 64u;
    const uint32_t head_dim = 512u;
    const uint32_t top_k    = 512u;
    const uint32_t window   = 128u;
    const uint32_t raw_cap  = 4352u;
    const uint32_t comp_cap = 1026u;
    uint32_t n_raw_rt       = raw_cap;
    /* n_comp is NOT comp_cap.  comp_cap is the ALLOCATION; n_comp is how many
     * compressed rows actually exist at this depth, and the engine logs
     * "ATTN-MIXED n_tokens=4096 n_comp=32" for a cold 4096-token prefill.
     * Using comp_cap here made the harness scan 1026 rows instead of 32 and
     * blew the calibration gate by 7.70x -- the gate caught it. */
    uint32_t n_comp_rt      = argc > 3 ? (uint32_t)atoi(argv[3]) : 32u;
    const uint32_t ratio    = 4u;
    /* PRODUCTION IS ALWAYS f16 (the raw ring has no f32 mode; PULSAR_IDX_FP4
     * likewise defaults on).  This was 0 for the first calibration runs, i.e.
     * the harness staged f32 raw KV -- twice production's traffic -- while the
     * stall breakdown was read off it.  argv[5] overrides. */
    const int      raw_f16  = argc > 5 ? atoi(argv[5]) : 1;

    CK(cudaSetDevice(0));

    const size_t q_elems    = (size_t)n_tokens * n_head * head_dim;
    const size_t rawkv_el   = (size_t)raw_cap * head_dim;
    const size_t compkv_el  = (size_t)comp_cap * head_dim;
    const size_t topk_el    = (size_t)n_tokens * top_k;

    float *q = nullptr, *heads = nullptr, *raw_kv = nullptr, *comp_kv = nullptr, *sinks = nullptr;
    int32_t *topk = nullptr, *positions = nullptr;
    CK(cudaMalloc(&q,       q_elems   * sizeof(float)));
    CK(cudaMalloc(&heads,   q_elems   * sizeof(float)));
    CK(cudaMalloc(&raw_kv,  rawkv_el  * sizeof(float)));  /* f32-sized: fits either format */
    CK(cudaMalloc(&comp_kv, compkv_el * sizeof(float)));
    CK(cudaMalloc(&sinks,   n_head    * sizeof(float)));
    CK(cudaMalloc(&topk,    topk_el   * sizeof(int32_t)));
    CK(cudaMalloc(&positions, (size_t)n_tokens * sizeof(int32_t)));

    printf("shape n_tokens=%u n_head=%u head_dim=%u top_k=%u window=%u raw_cap=%u comp_cap=%u\n",
           n_tokens, n_head, head_dim, top_k, window, raw_cap, comp_cap);
    printf("alloc q/out %.2f GiB each, raw_kv %.1f MiB, comp_kv %.1f MiB\n",
           q_elems * sizeof(float) / 1073741824.0,
           rawkv_el * sizeof(float) / 1048576.0,
           compkv_el * sizeof(float) / 1048576.0);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> uf(-1.f, 1.f);
    {
        std::vector<float> h(1u << 20);
        for (auto &v : h) v = uf(rng);
        for (size_t o = 0; o < q_elems; o += h.size()) {
            const size_t n = std::min(h.size(), q_elems - o);
            CK(cudaMemcpy(q + o, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        }
        for (size_t o = 0; o < rawkv_el; o += h.size()) {
            const size_t n = std::min(h.size(), rawkv_el - o);
            CK(cudaMemcpy(raw_kv + o, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        }
        for (size_t o = 0; o < compkv_el; o += h.size()) {
            const size_t n = std::min(h.size(), compkv_el - o);
            CK(cudaMemcpy(comp_kv + o, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        }
        std::vector<float> hs(n_head);
        for (auto &v : hs) v = uf(rng);
        CK(cudaMemcpy(sinks, hs.data(), hs.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    std::vector<int32_t> htopk(topk_el);
    {
        std::vector<int32_t> hp(n_tokens);
        for (uint32_t t = 0; t < n_tokens; t++) hp[t] = (int32_t)t;
        CK(cudaMemcpy(positions, hp.data(), hp.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    /* topk MUST be regenerated per n_comp: entries >= n_comp are skipped by the
     * kernel, so filling from [0,comp_cap) made the sweep flat past n_comp=128
     * and hid the very axis being swept.  The gate caught that too. */
    auto fill_topk = [&](uint32_t nc) {
        std::uniform_int_distribution<int> ui(0, (int)(nc ? nc - 1u : 0u));
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t i = 0; i < top_k; i++) htopk[(size_t)t * top_k + i] = ui(rng);
            std::sort(htopk.begin() + (size_t)t * top_k,
                      htopk.begin() + (size_t)(t + 1) * top_k);
        }
        CK(cudaMemcpy(topk, htopk.data(), htopk.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    };

    const dim3 grid(n_tokens, (n_head + 15u) / 16u, 1);
    /* ROWS_PER_STAGE is the smem staging depth -- the knob that sets how many KV
     * rows are buffered per iteration, i.e. exactly the traffic the dominant
     * short_scoreboard (MIO/shared-memory) stall is waiting on.  Sweep it before
     * contemplating any rewrite. */
    uint32_t rps_rt = argc > 4 ? (uint32_t)atoi(argv[4]) : 8u;
    auto launch = [&]() {
        switch (rps_rt) {
        case 4:
            attention_indexed_mixed_heads8_online_kernel<4, 16><<<grid, 512>>>(
                heads, sinks, q, raw_kv, comp_kv, topk,
                n_tokens, 0u, n_raw_rt, raw_cap, 0u,
                n_comp_rt, top_k, window, ratio, n_head, head_dim, raw_f16,
                0u, positions, nullptr, nullptr, comp_cap, 1u);
            return;
        case 16:
            attention_indexed_mixed_heads8_online_kernel<16, 16><<<grid, 512>>>(
                heads, sinks, q, raw_kv, comp_kv, topk,
                n_tokens, 0u, n_raw_rt, raw_cap, 0u,
                n_comp_rt, top_k, window, ratio, n_head, head_dim, raw_f16,
                0u, positions, nullptr, nullptr, comp_cap, 1u);
            return;
        default: break;
        }
        attention_indexed_mixed_heads8_online_kernel<8, 16><<<grid, 512>>>(
            heads, sinks, q, raw_kv, comp_kv, topk,
            n_tokens, /*pos0=*/0u, n_raw_rt, raw_cap, /*raw_start=*/0u,
            n_comp_rt, top_k, window, ratio, n_head, head_dim, raw_f16,
            /*comp_kv_pack=*/0u, positions, /*seq_id=*/nullptr,
            /*comp_bank_ptrs=*/nullptr, comp_cap, /*n_banks=*/1u);
    };

    launch();
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());

    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

    /* Sweep CAUSAL DEPTH, not n_comp.  The kernel derives the compressed range
     * from qpos = positions[t], so with positions pinned to 0..511 the scan was
     * capped at ~128 rows regardless of the n_comp argument -- which is exactly
     * why the earlier n_comp sweep went flat past 128 and failed the gate.
     * Sub-batch i covers tokens [i*512, (i+1)*512) of a 4096-token chunk. */
    double lo = 1e30, hi = 0.0;
    double worst_rel = 0.0;
    printf("\n  sub-batch  base_pos   harness    engine     rel\n");
    for (uint32_t i = 0; i < 8u; i++) {
        const uint32_t base = i * n_tokens;
        n_comp_rt = (base + n_tokens) / ratio;
        if (n_comp_rt > comp_cap) n_comp_rt = comp_cap;
        n_raw_rt = base + n_tokens;
        if (n_raw_rt > raw_cap) n_raw_rt = raw_cap;
        fill_topk(n_comp_rt);
        {
            std::vector<int32_t> hp(n_tokens);
            for (uint32_t t = 0; t < n_tokens; t++) hp[t] = (int32_t)(base + t);
            CK(cudaMemcpy(positions, hp.data(), hp.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
        }
        launch(); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        double best = 1e30;
        for (int k = 0; k < iters; k++) {
            CK(cudaEventRecord(a)); launch(); CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            float ms = 0.f; CK(cudaEventElapsedTime(&ms, a, b));
            if (ms < best) best = ms;
        }
        const double rel = best / kEngineRamp[i];
        if (fabs(rel - 1.0) > worst_rel) worst_rel = fabs(rel - 1.0);
        printf("  %8u  %8u  %8.3f  %8.3f  %6.2fx\n", i, base, best, kEngineRamp[i], rel);
        if (best < lo) lo = best;
        if (best > hi) hi = best;
    }

    printf("\n  harness range   %6.3f .. %6.3f ms\n", lo, hi);
    printf("  engine  range   %6.3f .. %6.3f ms  (168 launches, grid 512x4)\n",
           kEngineMin, kEngineMax);
    printf("  worst per-point deviation from the engine ramp: %.1f%%\n", worst_rel * 100.0);

    /* Bracketing the range is necessary but weak; require the RAMP to match
     * within 25% at every point, so the harness tracks the same work axis. */
    const bool brackets = (lo <= kEngineMax) && (hi >= kEngineMin) && (worst_rel <= 0.25);
    if (!brackets) {
        printf("\n  CALIBRATION GATE: FAIL -- harness range does not overlap the engine's.\n"
               "  It is measuring a different regime; any stall breakdown taken from\n"
               "  it is meaningless.  Fix the shape before trusting ncu.\n");
        return 1;
    }
    printf("\n  CALIBRATION GATE: PASS -- harness brackets the engine's observed range.\n"
           "  Pick the n_comp whose ms matches the launch you care about, then ncu it.\n");
    return 0;
}

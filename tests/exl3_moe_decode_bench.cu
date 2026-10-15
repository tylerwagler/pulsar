/* EXL3 routed-MoE DECODE / VERIFY bench (task #46): one routed layer through the production front door
 * (exl3_moe_routed_launch: sort, gate / up, fold, down, sum) at decode-row widths, at production shapes:
 *
 *   qwen  Qwen3.8-Flash-Next: 512 experts, top-10, 2560 -> fused gate_up 1280 -> down 640 -> 2560, bf16 rows
 *         (and bf16 mid), the family's activation format
 *   v4    DeepSeek V4: 256 experts, top-6, 4096 -> 2048 (gate + up pair) -> 4096, the E4M3 slot
 *   v41   DeepSeek V4.1: 384 experts, top-6, 5120 -> 2304 -> 5120, the E4M3 slot
 *
 * Rows are S streams x R verify rows: a stream's row r keeps each of row r-1's picks with probability `keep`
 * (consecutive tokens of one stream share experts) and draws the rest uniformly; streams are independent.
 * DRAM-cold: NL distinct copies of the layer's stacks and NROUTE routings are cycled, so an expert's bytes are
 * not L2-resident from the previous rep (production reads 40 other layers in between).  Prints us per layer
 * per width and the union of experts the rows touch.  Not a gate.
 *
 *   qwenp the same with turboderp's split gate / up stacks (td305: K3 experts)
 *
 * usage: [EXL3_BENCH_PROMPT=1] [EXL3_BENCH_SHARED=k2] ./tests/exl3_moe_decode_bench qwen|qwenp|v4|v41 k2_gate_up k2_down
 *        keep S:R [S:R ...]
 * (EXL3_BENCH_PROMPT=1 runs the rows as a prompt chunk: the grouped GEMM, for the A/B against the decode arm;
 *  EXL3_BENCH_SHARED=k2 (Qwen shapes) adds the riding shared expert -- three EXL3 dense slices 2560 -> 640 -> 2560 at
 *  rate k2, sigmoid-scaled -- as the family passes it, L288 exl3-dense)
 */
#include "pulsar_gpu.h"
#include "cuda/pulsar_cuda_mx.cuh"
#include "cuda/mmq/ds4_mmq.h"
#include "cuda/mmq/exl3_moe.cuh"
#include "engine/exl3_trellis.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "bench: %s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static uint32_t g_rng = 0x1234567u;
static uint32_t rnd(void) { uint32_t x = g_rng; x ^= x << 13; x ^= x >> 17; x ^= x << 5; return g_rng = x; }

__device__ __forceinline__ uint32_t hash32(uint64_t i, uint32_t seed) {
    uint32_t x = (uint32_t)i * 0x9E3779B1u ^ (uint32_t)(i >> 32) ^ seed;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
__global__ void fill_bytes(uint8_t *p, uint64_t n, uint32_t seed) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        p[i] = (uint8_t)hash32(i, seed);
}
__global__ void fill_scales(uint8_t *arena, uint64_t stride, uint64_t trellis, int E, int K, int N, uint32_t seed) {
    const int e = blockIdx.y;
    __half *s = reinterpret_cast<__half *>(arena + (uint64_t)e * stride + trellis);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < K + N; i += gridDim.x * blockDim.x) {
        const uint32_t h = hash32((uint64_t)e * 65536 + i, seed);
        const float m = 0.5f + (float)(h & 0xffff) / 65536.0f;
        s[i] = __float2half((h & 0x10000u) ? -m : m) * __float2half(i < K ? 0.025f : 1.0f);
    }
}
__global__ void fill_e4m3(uint8_t *p, uint64_t n, uint32_t seed) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        uint8_t b = (uint8_t)hash32(i, seed);
        if ((b & 0x7f) == 0x7f) b ^= 1;
        p[i] = b;
    }
}
__global__ void fill_bf16(__nv_bfloat16 *p, uint64_t n, uint32_t seed) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        p[i] = __float2bfloat16((float)(int)(hash32(i, seed) & 0xffff) / 32768.0f - 1.0f);
}

struct stack {
    uint8_t *arena = nullptr;
    const void **table = nullptr;
};
static stack make_stack(int K, int N, int k2, int E, uint32_t seed) {
    stack s;
    uint64_t trellis = 0, scales = 0, stride = 0;
    if (!exl3_expert_layout(K, N, k2, &trellis, &scales, &stride)) { fprintf(stderr, "layout\n"); exit(1); }
    CUDA_OK(cudaMalloc(&s.arena, stride * E));
    fill_bytes<<<1024, 256>>>(s.arena, stride * E, seed);
    fill_scales<<<dim3(8, E), 256>>>(s.arena, stride, trellis, E, K, N, seed ^ 0x5555u);
    std::vector<const void *> t(2 * (size_t)E);
    for (int e = 0; e < E; e++) { t[2 * e] = s.arena + e * stride; t[2 * e + 1] = s.arena + e * stride + trellis; }
    CUDA_OK(cudaMalloc(&s.table, t.size() * sizeof(void *)));
    CUDA_OK(cudaMemcpy(s.table, t.data(), t.size() * sizeof(void *), cudaMemcpyHostToDevice));
    CUDA_OK(cudaDeviceSynchronize());
    return s;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 6) {
        fprintf(stderr, "usage: %s qwen|v4|v41 k2_gate_up k2_down keep S:R [S:R ...]\n", argv[0]);
        return 2;
    }
    /* qwen: our fused gate_up quant (U-e4-d5); qwenp: turboderp's split gate / up packs (td305) */
    const bool qwenp = !strcmp(argv[1], "qwenp"), qwen = qwenp || !strcmp(argv[1], "qwen"), v41 = !strcmp(argv[1], "v41");
    const bool fused = qwen && !qwenp;
    const int H = qwen ? 2560 : v41 ? 5120 : 4096, MID = qwen ? 640 : v41 ? 2304 : 2048;
    const int E = qwen ? 512 : v41 ? 384 : 256, TOPK = qwen ? 10 : 6;
    const int k2gu = atoi(argv[2]), k2d = atoi(argv[3]);
    const double keep = atof(argv[4]);
    std::vector<std::pair<int, int>> widths;
    for (int i = 5; i < argc; i++) {
        int s = 1, r = 1;
        if (sscanf(argv[i], "%d:%d", &s, &r) != 2) { fprintf(stderr, "bad S:R %s\n", argv[i]); return 2; }
        widths.push_back({s, r});
    }
    int maxT = 0;
    for (auto &w : widths) maxT = std::max(maxT, w.first * w.second);
    if (!pulsar_gpu_init()) { fprintf(stderr, "bench: no GPU\n"); return 2; }
    int dev = 0;
    CUDA_OK(cudaGetDevice(&dev));
    if (ds4_mmq_init(dev) != 0) { fprintf(stderr, "bench: MMQ init\n"); return 2; }
    constexpr int NL = 3, NROUTE = 16;
    printf("exl3-moe-decode-bench: %s, %d -> %d (%s) -> %d, %d experts top-%d, K gate_up %g down %g, keep %.2f, "
           "%d layer copies x %d routings (DRAM-cold)%s\n", argv[1], H, MID, fused ? "fused gate_up" : "gate + up pair", H,
           E, TOPK, k2gu / 2.0, k2d / 2.0, keep, NL, NROUTE, getenv("EXL3_BENCH_SHARED") ? ", + the shared expert" : "");
    stack GU[NL], G[NL], U[NL], D[NL];
    for (int l = 0; l < NL; l++) {
        if (fused) GU[l] = make_stack(H, 2 * MID, k2gu, E, 11 + 100 * l);
        else { G[l] = make_stack(H, MID, k2gu, E, 11 + 100 * l); U[l] = make_stack(H, MID, k2gu, E, 22 + 100 * l); }
        D[l] = make_stack(MID, H, k2d, E, 33 + 100 * l);
    }
    /* the riding shared expert (Qwen): one [trellis | suh | svh] slice per projection per layer copy */
    const int k2s = getenv("EXL3_BENCH_SHARED") ? atoi(getenv("EXL3_BENCH_SHARED")) : 0;
    if (k2s && !qwen) { fprintf(stderr, "bench: EXL3_BENCH_SHARED is the Qwen shapes' shared expert\n"); return 2; }
    stack SG[NL], SU[NL], SD[NL];
    if (k2s)
        for (int l = 0; l < NL; l++) {
            SG[l] = make_stack(H, MID, k2s, 1, 44 + 100 * l);
            SU[l] = make_stack(H, MID, k2s, 1, 55 + 100 * l);
            SD[l] = make_stack(MID, H, k2s, 1, 66 + 100 * l);
        }
    const int64_t maxP = (int64_t)maxT * TOPK;
    const int kbx = pulsar_mx_kbp(H), kbm = pulsar_mx_kbp(MID);
    void *x = nullptr, *xsf = nullptr, *mid = nullptr, *midsf = nullptr;
    float *gz, *uz, *dz, *out, *wts;
    int32_t *sel;
    uint32_t *nf;
    if (qwen) {
        CUDA_OK(cudaMalloc(&x, (size_t)maxT * H * 2));
        fill_bf16<<<1024, 256>>>((__nv_bfloat16 *)x, (uint64_t)maxT * H, 7);
        CUDA_OK(cudaMalloc(&mid, (size_t)maxP * MID * 2));
    } else {
        CUDA_OK(cudaMalloc(&x, (size_t)maxT * H));
        fill_e4m3<<<1024, 256>>>((uint8_t *)x, (uint64_t)maxT * H, 7);
        CUDA_OK(cudaMalloc(&xsf, pulsar_mx_sf_slab_bytes(maxT, kbx)));
        std::vector<uint8_t> sf(pulsar_mx_sf_slab_bytes(maxT, kbx), 0);
        for (int r = 0; r < maxT; r++)
            for (int g = 0; g < H / 32; g++) sf[pulsar_mx_sfoff(r, g, kbx)] = (uint8_t)(118 + rnd() % 8);
        CUDA_OK(cudaMemcpy(xsf, sf.data(), sf.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMalloc(&mid, (size_t)maxP * MID));
        CUDA_OK(cudaMalloc(&midsf, pulsar_mx_sf_slab_bytes((int)maxP, kbm)));
        CUDA_OK(cudaMemset(midsf, 0, pulsar_mx_sf_slab_bytes((int)maxP, kbm)));
    }
    CUDA_OK(cudaMalloc(&gz, (size_t)maxP * 2 * MID * 4));
    CUDA_OK(cudaMalloc(&uz, (size_t)maxP * MID * 4));
    CUDA_OK(cudaMalloc(&dz, (size_t)maxP * H * 4));
    CUDA_OK(cudaMalloc(&out, (size_t)maxT * H * 4));
    CUDA_OK(cudaMalloc(&wts, (size_t)NROUTE * maxP * 4));
    CUDA_OK(cudaMalloc(&sel, (size_t)NROUTE * maxP * 4));
    CUDA_OK(cudaMalloc(&nf, 4));
    CUDA_OK(cudaMemset(nf, 0, 4));
    float *syg = nullptr, *syu = nullptr, *sys = nullptr, *ssc = nullptr;
    void *sh_h = nullptr, *sws = nullptr;
    const size_t sws_bytes = k2s ? exl3_moe_shared_ws_bytes(maxT, H, MID, H) : 0;
    if (k2s) {
        CUDA_OK(cudaMalloc(&syg, (size_t)maxT * MID * 4));
        CUDA_OK(cudaMalloc(&syu, (size_t)maxT * MID * 4));
        CUDA_OK(cudaMalloc(&sys, (size_t)maxT * H * 4));
        CUDA_OK(cudaMalloc(&ssc, (size_t)maxT * 4));
        CUDA_OK(cudaMemset(ssc, 0, (size_t)maxT * 4));
        CUDA_OK(cudaMalloc(&sh_h, (size_t)maxT * MID * 2));
        CUDA_OK(cudaMalloc(&sws, sws_bytes));
    }
    {
        std::vector<float> w((size_t)NROUTE * maxP);
        for (auto &v : w) v = 0.1f + (float)(rnd() % 1000) / 4000.0f;
        CUDA_OK(cudaMemcpy(wts, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
    }
    CUDA_OK(cudaDeviceSynchronize());
    cudaStream_t st = cudaStreamPerThread;
    const bool as_prompt = getenv("EXL3_BENCH_PROMPT") != nullptr;   /* the same rows as a prompt chunk: the grouped GEMM */
    const exl3_moe_act xa = qwen ? exl3_moe_act{EXL3_MOE_ACT_BF16_ROWS, x, nullptr, 0}
                                 : exl3_moe_act{EXL3_MOE_ACT_E4M3_SLOT, x, xsf, kbx};
    const exl3_moe_act ma = qwen ? exl3_moe_act{EXL3_MOE_ACT_BF16_ROWS, mid, nullptr, 0}
                                 : exl3_moe_act{EXL3_MOE_ACT_E4M3_SLOT, mid, midsf, kbm};
    for (auto &wd : widths) {
        const int S = wd.first, R = wd.second, T = S * R;
        /* NROUTE routings of T rows */
        std::vector<int32_t> s((size_t)NROUTE * maxP);
        double uni = 0;
        for (int q = 0; q < NROUTE; q++) {
            int32_t *sq = s.data() + (size_t)q * maxP;
            std::set<int> all;
            for (int si = 0; si < S; si++)
                for (int r = 0; r < R; r++) {
                    const int t = si * R + r;
                    for (int k = 0; k < TOPK; k++) {
                        int e;
                        bool dup;
                        do {
                            if (r > 0 && (double)(rnd() % 10000) / 10000.0 < keep)
                                e = sq[(size_t)(t - 1) * TOPK + rnd() % TOPK];
                            else
                                e = (int)(rnd() % (uint32_t)E);
                            dup = false;
                            for (int j = 0; j < k; j++) dup |= sq[(size_t)t * TOPK + j] == e;
                        } while (dup);
                        sq[(size_t)t * TOPK + k] = e;
                        all.insert(e);
                    }
                }
            uni += (double)all.size();
        }
        CUDA_OK(cudaMemcpy(sel, s.data(), s.size() * 4, cudaMemcpyHostToDevice));
        auto layer = [&](int i) -> int {
            const int l = i % NL, q = i % NROUTE;
            const exl3_moe_stacks w = fused ? exl3_moe_stacks{GU[l].table, nullptr, nullptr, D[l].table, k2gu, k2d, E, H, MID, H}
                                           : exl3_moe_stacks{nullptr, G[l].table, U[l].table, D[l].table, k2gu, k2d, E, H, MID, H};
            const exl3_moe_bufs b{gz, uz, dz, ma};
            const exl3_moe_shared sh{SG[l].arena, SU[l].arena, SD[l].arena, k2s, k2s, k2s, MID, ssc, syg, syu, sys,
                                     static_cast<__nv_bfloat16 *>(sh_h), sws, sws_bytes};
            if (k2s && exl3_moe_shared_fork(sh, xa, T, H, H, as_prompt, st) != 0) return -1;   /* the family's order */
            return exl3_moe_routed_launch(w, sel + (size_t)q * maxP, wts + (size_t)q * maxP, T, TOPK, 10.0f, xa, b, out,
                                          nf, 1u, as_prompt, st, k2s ? &sh : nullptr, k2s ? 1 : 0);
        };
        for (int i = 0; i < NROUTE * NL; i++)
            if (layer(i) != 0) { fprintf(stderr, "bench: layer T=%d refused\n", T); return 1; }
        CUDA_OK(cudaStreamSynchronize(st));
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        const int reps = NROUTE * NL * 4;
        cudaEventRecord(a, st);
        for (int i = 0; i < reps; i++) layer(i);
        cudaEventRecord(b, st);
        CUDA_OK(cudaEventSynchronize(b));
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        printf("  S=%d R=%-2d T=%-3d  %8.1f us/layer  %7.1f us/row  (union %.1f experts)\n", S, R, T,
               1000.0 * ms / reps, 1000.0 * ms / reps / T, uni / NROUTE);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    return 0;
}

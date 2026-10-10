/* EXL3 MoE prefill bench (L287): the routed experts of ONE DeepSeek layer on a prompt chunk, at production
 * shapes, through the production front door -- ds4_exl3_moe_pair (gate / up), the fold, ds4_exl3_moe_single
 * (down), the sum -- over the producer's E4M3 slot, every expert a distinct random EXL3 slice (DRAM-resident
 * stacks of the real size), uniform top-k routing.  Prints ms per layer and the layer's tokens/s.  Also the
 * D2R-vs-decode-once comparison for one projection: the prefill arm (trellis decoded into the mma B
 * fragments) against decoding each routed expert ONCE into fp16 and a cuBLASLt GEMM per expert (the dense
 * prefill arm, qwen_exl3_dense_prefill_launch, per expert with its rows).  Not a gate.
 *
 * usage: ./tests/exl3_moe_prefill_bench [v4|v41] [k2] [tokens ...]
 */
#include "pulsar_gpu.h"
#include "cuda/pulsar_cuda_mx.cuh"
#include "cuda/mmq/ds4_mmq.h"
#include "cuda/mmq/ds4_exl3_gemv.cuh"
#include "cuda/mmq/exl3_moe_prefill.cuh"
#include "engine/exl3_trellis.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
/* every byte random (any trellis word is a valid tile) */
__global__ void fill_bytes(uint8_t *p, uint64_t n, uint32_t seed) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        p[i] = (uint8_t)hash32(i, seed);
}
/* the scales plane of every slice: suh ~ +-0.025, svh ~ +-1 */
__global__ void fill_scales(uint8_t *arena, uint64_t stride, uint64_t trellis, int E, int K, int N, uint32_t seed) {
    const int e = blockIdx.y;
    __half *s = reinterpret_cast<__half *>(arena + (uint64_t)e * stride + trellis);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < K + N; i += gridDim.x * blockDim.x) {
        const uint32_t h = hash32((uint64_t)e * 65536 + i, seed);
        const float m = 0.5f + (float)(h & 0xffff) / 65536.0f;
        s[i] = __float2half((h & 0x10000u) ? -m : m) * __float2half(i < K ? 0.025f : 1.0f);
    }
}
/* E4M3 values (no NaN) and bf16 rows */
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
    uint64_t stride = 0, trellis = 0;
};
static stack make_stack(int K, int N, int k2, int E, uint32_t seed) {
    stack s;
    uint64_t scales = 0;
    if (!exl3_expert_layout(K, N, k2, &s.trellis, &scales, &s.stride)) { fprintf(stderr, "layout\n"); exit(1); }
    CUDA_OK(cudaMalloc(&s.arena, s.stride * E));
    fill_bytes<<<1024, 256>>>(s.arena, s.stride * E, seed);
    fill_scales<<<dim3(8, E), 256>>>(s.arena, s.stride, s.trellis, E, K, N, seed ^ 0x5555u);
    std::vector<const void *> t(2 * (size_t)E);
    for (int e = 0; e < E; e++) { t[2 * e] = s.arena + e * s.stride; t[2 * e + 1] = s.arena + e * s.stride + s.trellis; }
    CUDA_OK(cudaMalloc(&s.table, t.size() * sizeof(void *)));
    CUDA_OK(cudaMemcpy(s.table, t.data(), t.size() * sizeof(void *), cudaMemcpyHostToDevice));
    CUDA_OK(cudaDeviceSynchronize());
    return s;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const bool v41 = argc > 1 && !strcmp(argv[1], "v41");
    const int H = v41 ? 5120 : 4096, MID = v41 ? 2304 : 2048, E = v41 ? 384 : 256, TOPK = 6;
    const int k2 = argc > 2 ? atoi(argv[2]) : 6;
    std::vector<int> toks;
    for (int i = 3; i < argc; i++) toks.push_back(atoi(argv[i]));
    if (toks.empty()) toks = {512, 4096, 8192};
    int maxT = 0;
    for (int t : toks) maxT = t > maxT ? t : maxT;
    if (!pulsar_gpu_init()) { fprintf(stderr, "bench: no GPU\n"); return 2; }
    int dev = 0;
    CUDA_OK(cudaGetDevice(&dev));
    if (ds4_mmq_init(dev) != 0) { fprintf(stderr, "bench: MMQ init\n"); return 2; }
    printf("exl3-moe-prefill-bench: %s routed experts, %d -> %d x %d experts, top-%d, K=%g, distinct slices\n",
           v41 ? "V4.1" : "V4 Vision-Exp", H, MID, E, TOPK, k2 / 2.0);
    stack G = make_stack(H, MID, k2, E, 11), U = make_stack(H, MID, k2, E, 22), D = make_stack(MID, H, k2, E, 33);
    printf("  stacks: %.2f GB\n", (double)(G.stride + U.stride + D.stride) * E / 1e9);

    const int64_t maxP = (int64_t)maxT * TOPK;
    const int kbx = pulsar_mx_kbp(H), kbm = pulsar_mx_kbp(MID);
    uint8_t *xq, *xsf, *midq, *midsf;
    float *gz, *uz, *dz, *out, *wts;
    int32_t *sel;
    uint32_t *nf;
    CUDA_OK(cudaMalloc(&xq, (size_t)maxT * H));
    CUDA_OK(cudaMalloc(&xsf, pulsar_mx_sf_slab_bytes(maxT, kbx)));
    CUDA_OK(cudaMalloc(&midq, (size_t)maxP * MID));
    CUDA_OK(cudaMalloc(&midsf, pulsar_mx_sf_slab_bytes((int)maxP, kbm)));
    CUDA_OK(cudaMalloc(&gz, (size_t)maxP * MID * 4));
    CUDA_OK(cudaMalloc(&uz, (size_t)maxP * MID * 4));
    CUDA_OK(cudaMalloc(&dz, (size_t)maxP * H * 4));
    CUDA_OK(cudaMalloc(&out, (size_t)maxT * H * 4));
    CUDA_OK(cudaMalloc(&wts, (size_t)maxP * 4));
    CUDA_OK(cudaMalloc(&sel, (size_t)maxP * 4));
    CUDA_OK(cudaMalloc(&nf, 4));
    CUDA_OK(cudaMemset(nf, 0, 4));
    fill_e4m3<<<1024, 256>>>(xq, (uint64_t)maxT * H, 7);
    {
        std::vector<uint8_t> sf(pulsar_mx_sf_slab_bytes(maxT, kbx), 0);
        for (int r = 0; r < maxT; r++)
            for (int g = 0; g < H / 32; g++) sf[pulsar_mx_sfoff(r, g, kbx)] = (uint8_t)(118 + rnd() % 8);
        CUDA_OK(cudaMemcpy(xsf, sf.data(), sf.size(), cudaMemcpyHostToDevice));
    }
    CUDA_OK(cudaMemset(midsf, 0, pulsar_mx_sf_slab_bytes((int)maxP, kbm)));
    {
        std::vector<int32_t> s((size_t)maxP);
        std::vector<float> w((size_t)maxP);
        for (int t = 0; t < maxT; t++)
            for (int k = 0; k < TOPK; k++) {
                int e;
                bool dup;
                do { e = (int)(rnd() % (uint32_t)E); dup = false; for (int q = 0; q < k; q++) dup |= s[(size_t)t * TOPK + q] == e; } while (dup);
                s[(size_t)t * TOPK + k] = e;
                w[(size_t)t * TOPK + k] = 0.1f + (float)(rnd() % 1000) / 4000.0f;
            }
        CUDA_OK(cudaMemcpy(sel, s.data(), s.size() * 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(wts, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
    }
    CUDA_OK(cudaDeviceSynchronize());
    cudaStream_t st = cudaStreamPerThread;
    auto layer = [&](int T) -> int {
        const int64_t P = (int64_t)T * TOPK;
        int rc = ds4_exl3_moe_pair(G.table, U.table, k2, sel, gz, uz, MID, H, T, E, TOPK, st, xq, xsf, kbx, true);
        rc |= ds4_exl3_moe_fold_launch(gz, uz, sel, wts, G.table, U.table, D.table, H, MID, P, 10.0f, midq, midsf, kbm, st);
        rc |= ds4_exl3_moe_single(D.table, k2, sel, dz, H, MID, (int)P, E, 1, st, midq, midsf, kbm, true);
        rc |= ds4_exl3_moe_sum_launch(out, dz, sel, D.table, MID, H, TOPK, T, nf, 1u, st);
        return rc;
    };
    printf("1. one routed layer (pair + fold + down + sum) through the front door, uniform top-%d\n", TOPK);
    for (int T : toks) {
        if (layer(T) != 0) { fprintf(stderr, "bench: layer T=%d refused\n", T); return 1; }
        CUDA_OK(cudaStreamSynchronize(st));
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        const int reps = T >= 4096 ? 3 : 10;
        cudaEventRecord(a, st);
        for (int i = 0; i < reps; i++) layer(T);
        cudaEventRecord(b, st);
        CUDA_OK(cudaEventSynchronize(b));
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        ms /= reps;
        const double flop = 2.0 * T * TOPK * 3.0 * (double)H * MID;
        printf("  T=%-5d  %8.3f ms/layer  %9.0f tok/s (this layer's experts)  %5.1f TFLOP/s\n", T, ms,
               T / (ms * 1e-3), flop / (ms * 1e-3) / 1e12);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    /* 2. one projection (gate, H -> MID, rotating) at a 4096-token chunk: D2R vs decode-once + GEMM */
    {
        const int T = 4096;
        const int64_t P = (int64_t)T * TOPK;
        __nv_bfloat16 *xb;
        CUDA_OK(cudaMalloc(&xb, (size_t)P * H * 2));
        fill_bf16<<<1024, 256>>>(xb, (uint64_t)P * H, 3);
        std::vector<int32_t> s((size_t)P);
        CUDA_OK(cudaMemcpy(s.data(), sel, P * 4, cudaMemcpyDeviceToHost));
        /* the expert-sorted schedule, rows of an expert contiguous in xb (per-pair rows) */
        std::vector<int32_t> ids, bounds(E + 1, 0), cnt(E, 0);
        for (int64_t p = 0; p < P; p++) cnt[s[p]]++;
        for (int e = 0; e < E; e++) bounds[e + 1] = bounds[e] + cnt[e];
        ids.resize(P);
        for (int64_t i = 0; i < P; i++) ids[i] = (int32_t)i;
        int32_t *d_ids, *d_b;
        float *y;
        CUDA_OK(cudaMalloc(&d_ids, P * 4)); CUDA_OK(cudaMalloc(&d_b, (E + 1) * 4));
        CUDA_OK(cudaMalloc(&y, (size_t)P * MID * 4));
        CUDA_OK(cudaMemcpy(d_ids, ids.data(), P * 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_b, bounds.data(), (E + 1) * 4, cudaMemcpyHostToDevice));
        const size_t wsb = exl3_moe_prefill_ws_bytes(P, H, E);
        void *ws;
        CUDA_OK(cudaMalloc(&ws, wsb));
        const exl3_moe_act act{EXL3_MOE_ACT_BF16_ROWS, xb, nullptr, 0};
        auto d2r = [&]() { return exl3_moe_prefill_launch(G.table, k2, true, act, d_ids, d_ids, d_b, y, MID, H, P, E, ws, wsb, st); };
        auto once = [&]() {
            int rc = 0;
            for (int e = 0; e < E; e++)
                if (cnt[e]) rc |= qwen_exl3_dense_prefill_launch(G.arena + e * G.stride, k2, xb + (size_t)bounds[e] * H,
                                                                 y + (size_t)bounds[e] * MID, cnt[e], H, MID, G.trellis, st);
            return rc;
        };
        float ms[2] = {0, 0};
        for (int pass = 0; pass < 2; pass++) {
            if ((pass ? once() : d2r()) != 0) { fprintf(stderr, "bench: arm %d refused\n", pass); return 1; }
            CUDA_OK(cudaStreamSynchronize(st));
            cudaEvent_t a, b;
            cudaEventCreate(&a); cudaEventCreate(&b);
            cudaEventRecord(a, st);
            for (int i = 0; i < 3; i++) (void)(pass ? once() : d2r());
            cudaEventRecord(b, st);
            CUDA_OK(cudaEventSynchronize(b));
            cudaEventElapsedTime(&ms[pass], a, b);
            ms[pass] /= 3;
        }
        printf("2. gate projection, %d tokens (%.0f rows per expert): D2R prefill arm %.3f ms; decode-once fp16 + "
               "cuBLASLt GEMM per expert %.3f ms (%.2fx)\n", T, (double)P / E, ms[0], ms[1], ms[1] / ms[0]);
    }
    return 0;
}

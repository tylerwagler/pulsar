/* GQA attention block microbenchmark (L295 F1) at Inkling-Small's shapes:
 * 32 q / 8 kv / 128, relative bias, global (full, extent 1024, temperature)
 * and window (512, extent 512) layers.  Not a gate.
 *
 *   roofline    a streaming read kernel (achievable DRAM bandwidth) and an
 *               mma.sync m16n8k16 bf16 loop (achievable tensor-core rate),
 *               measured here, beside GB10's 273 GB/s and 125 TFLOPS specs
 *   decode      1 / 4 / 16 sequences x one row, and a 6-row verify, at 4K /
 *               32K / 128K context, DRAM-cold (each sequence its own pool, an
 *               L2 flush between calls): time per call, the KV bytes it must
 *               read, GB/s against the measured read bandwidth
 *   prefill     a 4096-row chunk at position 0 and at 32K: time, attention
 *               FLOPs (4 * n_q * head_dim per (row, key) attended; the P V hi
 *               / lo pair counts once), TFLOPS against the measured mma rate
 * Times are cudaEvent spans over back-to-back calls (the host's descriptor
 * build and upload included, as the lane pays them). */
#include "cuda/pulsar_cuda_attn_gqa.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

__global__ void read_kernel(const uint4 *p, uint64_t n, uint32_t *sink) {
    uint32_t acc = 0;
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) *sink = acc;
}
__global__ void mma_kernel(float *sink, int iters) {
    float d[4][4] = {};
    uint32_t a = threadIdx.x * 0x3c003c00u, b = 0x3c003c00u ^ threadIdx.x;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+f"(d[j][0]), "+f"(d[j][1]), "+f"(d[j][2]), "+f"(d[j][3])
                         : "r"(a), "r"(b), "r"(a), "r"(b), "r"(b), "r"(a));
    }
    float s = 0;
    for (int j = 0; j < 4; j++) s += d[j][0] + d[j][1] + d[j][2] + d[j][3];
    if (s == 1234.5f) *sink = s;
}

static uint8_t *g_flush;
static const uint64_t FLUSH = 256ull << 20;
static void flush_l2() { CK(cudaMemsetAsync(g_flush, 0, FLUSH, 0)); }

struct Layer {
    pulsar_gqa_layer L{};
    uint16_t *qw, *kw, *P;
};
static uint16_t *rand_bf16(uint64_t n, float sd) {
    std::vector<uint16_t> h(n);
    uint64_t s = 0x9e3779b97f4a7c15ull ^ n;
    for (auto &x : h) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const float f = ((float)((s >> 40) & 0xffffff) / 16777216.f - 0.5f) * 3.4f * sd;
        x = __bfloat16_as_ushort(__float2bfloat16(f));
    }
    uint16_t *d;
    CK(cudaMalloc(&d, n * 2));
    CK(cudaMemcpy(d, h.data(), n * 2, cudaMemcpyHostToDevice));
    return d;
}
static Layer make_layer(bool window) {
    Layer y;
    pulsar_gqa_layer &L = y.L;
    L.n_q = 32; L.n_kv = 8; L.head_dim = 128; L.kv_fmt = PULSAR_GQA_KV_BF16;
    L.span = window ? PULSAR_GQA_WINDOW : PULSAR_GQA_FULL; L.window = window ? 512u : 0u;
    L.scale = 1.0f / 128; L.qk_norm = 1; L.norm_eps = 1e-6f;
    y.qw = rand_bf16(128, 1.f); y.kw = rand_bf16(128, 1.f);
    L.q_norm_w = y.qw; L.k_norm_w = y.kw;
    L.rel_extent = window ? 512u : 1024u; L.rel_dim = 16;
    y.P = rand_bf16((uint64_t)16 * L.rel_extent, 0.5f); L.rel_proj = y.P;
    L.temp_floor = window ? 0.f : 128000.f; L.temp_alpha = 0.1f;
    return y;
}

/* time `iters` calls of rows (one run per sequence: seq s at positions pos[s] .. pos[s] + run - 1) */
static double time_calls(const pulsar_gqa_layer &L, std::vector<pulsar_gqa_seq> &seqs, const std::vector<uint32_t> &pos,
                         uint32_t run, int iters, bool cold) {
    const uint32_t ns = (uint32_t)seqs.size(), n = ns * run;
    const uint32_t ld = 32 * 128 + 2 * 8 * 128 + 32 * 16;
    uint32_t max_ctx = 0;
    std::vector<uint32_t> rs(n), rp(n);
    for (uint32_t s = 0; s < ns; s++)
        for (uint32_t i = 0; i < run; i++) { rs[s * run + i] = s; rp[s * run + i] = pos[s] + i; max_ctx = std::max(max_ctx, pos[s] + run); }
    uint16_t *in = rand_bf16((uint64_t)n * ld, 1.f);
    __nv_bfloat16 *out;
    CK(cudaMalloc(&out, (uint64_t)n * 4096 * 2));
    const uint64_t wsb = pulsar_gqa_workspace_bytes(&L, n, max_ctx);
    void *ws;
    CK(cudaMalloc(&ws, wsb));
    pulsar_gqa_io io{};
    io.q = {in, ld, 128}; io.k = {in + 4096, ld, 128}; io.v = {in + 4096 + 1024, ld, 128}; io.r = {in + 4096 + 2048, ld, 16};
    io.out_bf16 = out; io.ld_out = 4096; io.workspace = ws; io.workspace_bytes = wsb;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    double ms_total = 0;
    for (int it = -1; it < iters; it++) {        /* -1: warm-up */
        if (cold) flush_l2();
        CK(cudaEventRecord(e0, 0));
        if (pulsar_gqa_forward(&L, seqs.data(), ns, rs.data(), rp.data(), n, &io, 0) != 0) { fprintf(stderr, "refused\n"); exit(2); }
        CK(cudaEventRecord(e1, 0));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        if (it >= 0) ms_total += ms;
    }
    CK(cudaFree(in)); CK(cudaFree(out)); CK(cudaFree(ws));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    return ms_total / iters;
}

__global__ void fill_kernel(uint16_t *p, uint64_t n, uint64_t seed) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t z = (i + seed) * 0x9e3779b97f4a7c15ull;
        z = (z ^ (z >> 31)) * 0xbf58476d1ce4e5b9ull;
        p[i] = __bfloat16_as_ushort(__float2bfloat16(((float)(z >> 40) / 16777216.f - 0.5f) * 3.4f));
    }
}
static std::vector<pulsar_gqa_seq> make_pools(const pulsar_gqa_layer &L, uint32_t n, uint32_t cap) {
    std::vector<pulsar_gqa_seq> v(n);
    const uint64_t b = (uint64_t)cap * pulsar_gqa_kv_token_bytes(&L);
    for (auto &s : v) {
        CK(cudaMalloc(&s.kv, b)); s.bytes = b; s.cap = cap;
        fill_kernel<<<1024, 256>>>((uint16_t *)s.kv, b / 2, (uint64_t)(&s - v.data()) << 40);
    }
    return v;
}
static void free_pools(std::vector<pulsar_gqa_seq> &v) { for (auto &s : v) CK(cudaFree(s.kv)); }

int main() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("gqa_attn_bench: %s, %d SMs, L2 %d MB\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20);
    CK(cudaMalloc(&g_flush, FLUSH));
    uint32_t *sink;
    CK(cudaMalloc(&sink, 64));

    /* ---- roofline */
    double bw = 0, tf = 0;
    {
        const uint64_t bytes = 2048ull << 20;
        uint4 *buf;
        CK(cudaMalloc(&buf, bytes));
        CK(cudaMemset(buf, 1, bytes));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        read_kernel<<<prop.multiProcessorCount * 8, 512>>>(buf, bytes / 16, sink);
        CK(cudaEventRecord(e0));
        for (int i = 0; i < 5; i++) read_kernel<<<prop.multiProcessorCount * 8, 512>>>(buf, bytes / 16, sink);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        bw = 5.0 * bytes / (ms * 1e-3) / 1e9;
        const int iters = 20000, blocks = prop.multiProcessorCount * 8;
        mma_kernel<<<blocks, 256>>>((float *)sink, 100);
        CK(cudaEventRecord(e0));
        mma_kernel<<<blocks, 256>>>((float *)sink, iters);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(&ms, e0, e1));
        tf = (double)blocks * 8 * iters * 4 * 2.0 * 16 * 8 * 16 / (ms * 1e-3) / 1e12;
        CK(cudaFree(buf));
        printf("roofline: DRAM read %.1f GB/s measured (spec 273); mma.sync bf16 %.1f TFLOPS measured (spec 125 dense)\n\n", bw, tf);
    }

    Layer glob = make_layer(false), win = make_layer(true);
    const uint64_t tb = pulsar_gqa_kv_token_bytes(&glob.L);

    /* ---- decode, DRAM-cold */
    printf("decode (DRAM-cold): layer, ctx, rows -> us/call, KV MB read, GB/s, %% of measured BW\n");
    for (int w = 0; w < 2; w++) {
        const Layer &Ly = w ? win : glob;
        for (uint32_t ctx : {4096u, 32768u, 131072u}) {
            if (w && ctx != 4096u && ctx != 131072u) continue;
            for (uint32_t nseq : {1u, 4u, 16u}) {
                const uint32_t cap = w ? 512u + 15u : ctx + 8u;
                std::vector<pulsar_gqa_seq> pools = make_pools(Ly.L, nseq, cap);
                std::vector<uint32_t> pos(nseq, ctx - 1u);
                const double ms = time_calls(Ly.L, pools, pos, 1, 20, true);
                const double keys = w ? 512.0 : (double)ctx;
                const double mb = nseq * keys * tb / 1e6;
                printf("  %-6s ctx %6u  %2u seq x 1 row : %8.1f us  %7.1f MB  %6.1f GB/s  %5.1f%%\n", w ? "window" : "global", ctx,
                       nseq, ms * 1e3, mb, mb / 1e3 / (ms * 1e-3), 100.0 * mb / 1e3 / (ms * 1e-3) / bw);
                free_pools(pools);
            }
            {   /* a 6-row verify of one sequence */
                const uint32_t cap = w ? 512u + 15u : ctx + 8u;
                std::vector<pulsar_gqa_seq> pools = make_pools(Ly.L, 1, cap);
                std::vector<uint32_t> pos(1, ctx - 6u);
                const double ms = time_calls(Ly.L, pools, pos, 6, 20, true);
                const double keys = w ? 512.0 + 5 : (double)ctx;
                const double mb = keys * tb / 1e6;
                printf("  %-6s ctx %6u   1 seq x 6 rows: %8.1f us  %7.1f MB  %6.1f GB/s  %5.1f%%\n", w ? "window" : "global", ctx,
                       ms * 1e3, mb, mb / 1e3 / (ms * 1e-3), 100.0 * mb / 1e3 / (ms * 1e-3) / bw);
                free_pools(pools);
            }
        }
    }

    /* ---- prefill: a 4096-row chunk */
    printf("\nprefill (one 4096-row chunk): layer, start -> ms, attention TFLOP, TFLOPS, %% of measured mma rate\n");
    for (int w = 0; w < 2; w++) {
        const Layer &Ly = w ? win : glob;
        for (uint32_t start : {0u, 32768u}) {
            const uint32_t n = 4096, cap = w ? 512u + 15u : start + n + 8u;
            std::vector<pulsar_gqa_seq> pools = make_pools(Ly.L, 1, cap);
            std::vector<uint32_t> pos(1, start);
            const double ms = time_calls(Ly.L, pools, pos, n, 5, false);
            double keys = 0;
            for (uint32_t i = 0; i < n; i++) {
                const double p = start + i;
                keys += w ? std::min(p + 1.0, 512.0) : p + 1.0;
            }
            const double tflop = keys * 4.0 * 32 * 128 / 1e12;
            printf("  %-6s start %6u: %8.2f ms  %7.3f TFLOP  %6.1f TFLOPS  %5.1f%%\n", w ? "window" : "global", start, ms, tflop,
                   tflop / (ms * 1e-3), 100.0 * tflop / (ms * 1e-3) / tf);
            free_pools(pools);
        }
    }
    return 0;
}

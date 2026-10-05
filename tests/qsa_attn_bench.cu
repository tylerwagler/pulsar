/* QSA ATTENTION bench (L251 S3): one Qwen attention layer's pulsar_gpu_qsa_forward
 * (everything between the Linears) timed on GB10.
 *
 *   decode   M sequences (M = 1, 4, 8), one row each, at depth D (8K, 64K, 256K):
 *            the caches are filled with valid synthetic E4M3/MX32 rows and bf16
 *            block keys (selection runs at every depth past 2051), and a 256 MB
 *            write between iterations evicts L2 -- DRAM-cold, as a served step
 *            meets a layer's cache.  Reports the median of 30.
 *   prefill  one sequence, a 2048-row chunk at depth D (0, 8K, 32K, 128K).
 *
 * usage: qsa_attn_bench [decode|prefill|all] */
#include "pulsar_gpu.h"
#include "pulsar_cuda_mx.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <vector>

namespace {

__global__ void fill_kv(uint8_t *kv, uint64_t tokens, uint64_t seed) {
    const uint64_t n = tokens * PULSAR_QSA_KV_TOKEN_BYTES;
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t h = (i + seed) * 0x9e3779b97f4a7c15ull;
        h ^= h >> 29;
        const uint32_t b = (uint32_t)(i % PULSAR_QSA_KV_TOKEN_BYTES);
        kv[i] = b < 1024u ? (uint8_t)((h & 0x80u) | (h % 0x77u)) : (uint8_t)(120u + h % 8u);
    }
}
__global__ void fill_bkey(__nv_bfloat16 *k, uint64_t n, uint64_t seed) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t h = (i + seed) * 0x9e3779b97f4a7c15ull;
        h ^= h >> 31;
        k[i] = __float2bfloat16((float)(h % 20001u) / 10000.f - 1.f);
    }
}
__global__ void fill_f32(float *x, uint64_t n, uint64_t seed) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t h = (i + seed) * 0xbf58476d1ce4e5b9ull;
        h ^= h >> 27;
        x[i] = (float)(h % 20001u) / 5000.f - 2.f;
    }
}
__global__ void scrub(uint4 *p, uint64_t n) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        p[i] = make_uint4((uint32_t)i, 1u, 2u, 3u);
    }
}

pulsar_gpu_tensor *alloc(uint64_t bytes) {
    pulsar_gpu_tensor *t = pulsar_gpu_tensor_alloc(bytes);
    if (!t) { fprintf(stderr, "alloc %llu failed\n", (unsigned long long)bytes); exit(2); }
    return t;
}
void *P(pulsar_gpu_tensor *t) { return pulsar_gpu_tensor_device_ptr(t); }

struct Layer {
    pulsar_gpu_tensor *qg, *k, *v, *idx, *slot, *sc, *ws, *norm[4];
    uint32_t rows;
};

Layer make_layer(uint32_t rows, uint32_t max_ctx) {
    Layer L;
    L.rows = rows;
    L.qg = alloc((uint64_t)rows * PULSAR_QSA_Q_IN * 4);
    L.k = alloc((uint64_t)rows * PULSAR_QSA_KV_IN * 4);
    L.v = alloc((uint64_t)rows * PULSAR_QSA_KV_IN * 4);
    L.idx = alloc((uint64_t)rows * PULSAR_QSA_IDX_IN * 4);
    fill_f32<<<256, 256>>>((float *)P(L.qg), (uint64_t)rows * PULSAR_QSA_Q_IN, 1);
    fill_f32<<<256, 256>>>((float *)P(L.k), (uint64_t)rows * PULSAR_QSA_KV_IN, 2);
    fill_f32<<<256, 256>>>((float *)P(L.v), (uint64_t)rows * PULSAR_QSA_KV_IN, 3);
    fill_f32<<<256, 256>>>((float *)P(L.idx), (uint64_t)rows * PULSAR_QSA_IDX_IN, 4);
    L.slot = alloc((uint64_t)rows * PULSAR_QSA_OUT_DIM);
    L.sc = alloc(pulsar_mx_sf_slab_bytes((int)rows, pulsar_mx_kbp(PULSAR_QSA_OUT_DIM)));
    L.ws = alloc(pulsar_gpu_qsa_workspace_bytes(rows, max_ctx));
    const uint32_t w[4] = {PULSAR_QSA_HEAD_DIM, PULSAR_QSA_HEAD_DIM, PULSAR_QSA_IDX_DIM, PULSAR_QSA_IDX_DIM};
    for (int i = 0; i < 4; i++) {
        L.norm[i] = alloc(w[i] * 2);                 /* the container stores the norms bf16 */
        { std::vector<uint16_t> b16(w[i]);
          for (uint32_t j = 0; j < w[i]; j++) b16[j] = (uint16_t)((uint32_t)(float)(10 + i) >> 16);
          pulsar_gpu_tensor_write(L.norm[i], 0, b16.data(), (uint64_t)w[i] * 2); }
    }
    return L;
}

pulsar_qsa_seq make_seq(uint32_t cap, uint64_t seed) {
    pulsar_qsa_seq s{alloc((uint64_t)cap * PULSAR_QSA_KV_TOKEN_BYTES), alloc((uint64_t)(cap / 4) * PULSAR_QSA_BKEY_BYTES),
                     alloc(PULSAR_QSA_STAGE_BYTES), cap};
    fill_kv<<<1024, 256>>>((uint8_t *)P(s.kv), cap, seed);
    fill_bkey<<<1024, 256>>>((__nv_bfloat16 *)P(s.bkey), (uint64_t)(cap / 4) * PULSAR_QSA_IDX_DIM, seed);
    fill_f32<<<4, 128>>>((float *)P(s.stage), PULSAR_QSA_STAGE_BYTES / 4, seed);
    return s;
}
void free_seq(pulsar_qsa_seq &s) {
    pulsar_gpu_tensor_free(s.kv);
    pulsar_gpu_tensor_free(s.bkey);
    pulsar_gpu_tensor_free(s.stage);
}

bool fwd(Layer &L, std::vector<pulsar_qsa_seq> &seqs, const std::vector<uint32_t> &rs, const std::vector<uint32_t> &rp) {
    pulsar_qsa_layer lw{(const uint16_t *)pulsar_gpu_tensor_device_ptr(L.norm[0]), (const uint16_t *)pulsar_gpu_tensor_device_ptr(L.norm[1]),
                        (const uint16_t *)pulsar_gpu_tensor_device_ptr(L.norm[2]), (const uint16_t *)pulsar_gpu_tensor_device_ptr(L.norm[3])};
    pulsar_qsa_io io{};
    io.qg = L.qg; io.k = L.k; io.v = L.v; io.idx = L.idx;
    io.out_e4m3 = P(L.slot);
    io.out_scale = P(L.sc);
    io.out_sf_pitch = pulsar_mx_kbp(PULSAR_QSA_OUT_DIM);
    return pulsar_gpu_qsa_forward(&lw, seqs.data(), (uint32_t)seqs.size(), rs.data(), rp.data(), (uint32_t)rs.size(), &io, L.ws);
}

/* median microseconds of `iters` forward calls, each after an L2 scrub */
double time_calls(Layer &L, std::vector<pulsar_qsa_seq> &seqs, const std::vector<uint32_t> &rs,
                  const std::vector<uint32_t> &rp, pulsar_gpu_tensor *scrub_buf, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::vector<double> us;
    for (int it = -3; it < iters; it++) {
        scrub<<<1024, 256>>>((uint4 *)P(scrub_buf), pulsar_gpu_tensor_bytes(scrub_buf) / 16);
        cudaEventRecord(a, cudaStreamPerThread);
        if (!fwd(L, seqs, rs, rp)) { fprintf(stderr, "forward refused\n"); exit(1); }
        cudaEventRecord(b, cudaStreamPerThread);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        if (it >= 0) us.push_back(ms * 1000.0);
    }
    std::sort(us.begin(), us.end());
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return us[us.size() / 2];
}

}  // namespace

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "all";
    if (!pulsar_gpu_init()) return 2;
    pulsar_gpu_tensor *scrub_buf = alloc(256ull << 20);
    const bool dec = !strcmp(mode, "decode") || !strcmp(mode, "all");
    const bool pre = !strcmp(mode, "prefill") || !strcmp(mode, "all");
    if (dec) {
        printf("DECODE (one Qwen attention layer, DRAM-cold, median of 30): us per call\n");
        printf("  depth      M=1        M=4        M=8     (us per row at M=8)\n");
        const uint32_t depths[3] = {8192, 65536, 262144};
        for (uint32_t D : depths) {
            const uint32_t cap = D + 4;
            std::vector<pulsar_qsa_seq> seqs;
            for (uint32_t i = 0; i < 8; i++) seqs.push_back(make_seq(cap, 1000 + i));
            Layer L = make_layer(8, cap);
            double t[3];
            const uint32_t Ms[3] = {1, 4, 8};
            for (int mi = 0; mi < 3; mi++) {
                std::vector<uint32_t> rs, rp;
                for (uint32_t i = 0; i < Ms[mi]; i++) { rs.push_back(i); rp.push_back(D); }
                t[mi] = time_calls(L, seqs, rs, rp, scrub_buf, 30);
            }
            printf("  %6uK  %8.1f   %8.1f   %8.1f   (%.1f)\n", D / 1024, t[0], t[1], t[2], t[2] / 8);
            for (auto &s : seqs) free_seq(s);
            pulsar_gpu_synchronize();
        }
    }
    if (pre) {
        printf("PREFILL (one layer, one sequence, a 2048-row chunk at depth D, median of 5)\n");
        const uint32_t depths[4] = {0, 8192, 32768, 131072};
        const uint32_t C = 2048;
        Layer L = make_layer(C, 131072 + C + 4);
        for (uint32_t D : depths) {
            std::vector<pulsar_qsa_seq> seqs{make_seq(D + C + 4, 77)};
            std::vector<uint32_t> rs(C, 0), rp(C);
            for (uint32_t i = 0; i < C; i++) rp[i] = D + i;
            const double us = time_calls(L, seqs, rs, rp, scrub_buf, 5);
            printf("  depth %6uK: %9.1f us per chunk = %8.0f tok/s per layer (%.0f tok/s for all 12 layers)\n",
                   D / 1024, us, C / (us * 1e-6), C / (us * 1e-6) / 12.0);
            free_seq(seqs[0]);
        }
    }
    return 0;
}

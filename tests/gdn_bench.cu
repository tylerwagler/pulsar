/* Gated DeltaNet microbenchmark (L251) -- one layer's non-Linear GDN work
 * (pulsar_gdn_forward: conv/gates kernel + recurrence/norm kernel), output as
 * the A8 slot only (the production format; the f32 plane skipped).
 *
 *   decode   M = 1, 4, 8, 16 sequences x one token.  The state slots rotate
 *            over a 64-slot pool (~200 MB) so every step reads its state from
 *            DRAM, as a 36-layer model does; the step's inputs (the in_proj
 *            outputs, 40 KB a row) stay L2-warm, as they would be behind the
 *            GEMV that wrote them.
 *   prefill  one sequence of T = 512, 2048, 8192 tokens.
 *   roofline a float4 streaming copy of 256 MB in the same process.
 *
 * Not a gate: it prints, it does not grade. */
#include "cuda/pulsar_cuda_gdn.h"
#include "cuda/pulsar_cuda_mx.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static const int QKV = PULSAR_GDN_QKV_DIM, VD = PULSAR_GDN_V_DIM, NV = PULSAR_GDN_NV;

__global__ void copy_kernel(const float4 *__restrict__ in, float4 *__restrict__ out, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) out[i] = in[i];
}
__global__ void fill_kernel(float *p, size_t n, float scale, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u ^ seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        p[i] = scale * ((float)(h & 0xffffff) / 8388608.f - 1.f);
    }
}
static void fill(float *p, size_t n, float scale, uint32_t seed) { fill_kernel<<<1024, 256>>>(p, n, scale, seed); CK(cudaGetLastError()); }

int main() {
    const int SLOTS = 64, MAXR = 8192;
    float *conv_w, *A_log, *dt_bias, *norm_w;
    CK(cudaMalloc(&conv_w, (size_t)QKV * 4 * 4)); CK(cudaMalloc(&A_log, NV * 4)); CK(cudaMalloc(&dt_bias, NV * 4)); CK(cudaMalloc(&norm_w, 128 * 4));
    fill(conv_w, (size_t)QKV * 4, 0.5f, 1); fill(A_log, NV, 1.f, 2); fill(dt_bias, NV, 2.f, 3); fill(norm_w, 128, 1.f, 4);
    float *qkv, *z, *a, *b, *cs, *rs;
    CK(cudaMalloc(&qkv, (size_t)MAXR * QKV * 4)); CK(cudaMalloc(&z, (size_t)MAXR * VD * 4));
    CK(cudaMalloc(&a, (size_t)MAXR * NV * 4)); CK(cudaMalloc(&b, (size_t)MAXR * NV * 4));
    fill(qkv, (size_t)MAXR * QKV, 2.f, 5); fill(z, (size_t)MAXR * VD, 2.f, 6); fill(a, (size_t)MAXR * NV, 2.f, 7); fill(b, (size_t)MAXR * NV, 2.f, 8);
    CK(cudaMalloc(&cs, PULSAR_GDN_CONV_STATE_FLOATS * SLOTS * 4)); CK(cudaMalloc(&rs, PULSAR_GDN_REC_STATE_FLOATS * SLOTS * 4));
    fill(cs, PULSAR_GDN_CONV_STATE_FLOATS * SLOTS, 1.f, 9); fill(rs, PULSAR_GDN_REC_STATE_FLOATS * SLOTS, 0.05f, 10);
    const size_t sb = pulsar_gdn_scratch_bytes(MAXR);
    void *scratch; CK(cudaMalloc(&scratch, sb));
    const int kbp = pulsar_mx_kbp(VD);
    void *oq, *os;
    CK(cudaMalloc(&oq, (size_t)MAXR * VD)); CK(cudaMalloc(&os, pulsar_mx_sf_slab_bytes(MAXR, kbp)));
    CK(cudaMemset(os, 0, pulsar_mx_sf_slab_bytes(MAXR, kbp)));
    int32_t *slot;
    CK(cudaMalloc(&slot, MAXR * 4));
    CK(cudaMemset(slot, 0, MAXR * 4));                  /* prefill: every row on slot 0 */
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const pulsar_gdn_weights w = {conv_w, A_log, dt_bias, norm_w};

    /* roofline */
    {
        const size_t n = (size_t)256 << 20 >> 4;   /* 256 MB of float4 */
        float4 *src, *dst; CK(cudaMalloc(&src, n * 16)); CK(cudaMalloc(&dst, n * 16));
        CK(cudaMemset(src, 0, n * 16));
        copy_kernel<<<48 * 16, 256>>>(src, dst, n);
        CK(cudaEventRecord(e0));
        for (int i = 0; i < 10; i++) copy_kernel<<<48 * 16, 256>>>(src, dst, n);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("roofline: streaming copy %.0f GB/s (read + write, 256 MB)\n", 2.0 * n * 16 * 10 / (ms * 1e6));
        CK(cudaFree(src)); CK(cudaFree(dst));
    }

    auto mkcall = [&](int n_seq, int seq_rows) {
        pulsar_gdn_call c{};
        c.n_seq = n_seq; c.seq_rows = seq_rows; c.row_slot = slot;
        c.conv_state = cs; c.rec_state = rs;
        c.qkv = qkv; c.ld_qkv = QKV; c.z = z; c.ld_z = VD; c.a = a; c.ld_a = NV; c.b = b; c.ld_b = NV;
        c.scratch = scratch; c.scratch_bytes = sb;
        c.out_e4m3 = oq; c.out_scale = os; c.out_kbp = kbp;
        return c;
    };

    printf("decode (one layer, DRAM-cold state, A8 output): M, us/step, state MB r+w, all MB, GB/s\n");
    const int Ms[] = {1, 4, 8, 16};
    for (int M : Ms) {
        const int sets = SLOTS / M;
        std::vector<std::vector<int32_t>> sl(sets, std::vector<int32_t>(M));
        for (int s = 0; s < sets; s++) for (int i = 0; i < M; i++) sl[s][i] = s * M + i;
        int32_t *slots_all; CK(cudaMalloc(&slots_all, (size_t)sets * M * 4));
        for (int s = 0; s < sets; s++) CK(cudaMemcpy(slots_all + s * M, sl[s].data(), M * 4, cudaMemcpyHostToDevice));
        pulsar_gdn_call c = mkcall(M, 1);
        const int ITER = 256;
        for (int it = 0; it < 16; it++) { c.row_slot = slots_all + (it % sets) * M; if (pulsar_gdn_forward(&w, &c, 0)) return 1; }
        CK(cudaEventRecord(e0));
        for (int it = 0; it < ITER; it++) { c.row_slot = slots_all + (it % sets) * M; pulsar_gdn_forward(&w, &c, 0); }
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        const double us = ms * 1e3 / ITER;
        const double state = M * 2.0 * (PULSAR_GDN_REC_STATE_FLOATS + PULSAR_GDN_CONV_STATE_FLOATS) * 4;
        const double other = M * (4.0 * (QKV + VD + 2 * NV) + 2.0 * 4 * (QKV + 2 * NV) + VD + VD / 32);
        printf("  M=%2d  %7.1f us  %6.1f MB  %6.1f MB  %5.0f GB/s\n", M, us, state / 1e6, (state + other) / 1e6,
               (state + other) / (us * 1e3));
        CK(cudaFree(slots_all));
    }

    printf("prefill (one layer, one sequence, A8 output): T, ms, tokens/s\n");
    const int Ts[] = {512, 2048, 8192};
    for (int T : Ts) {
        pulsar_gdn_call c = mkcall(1, T);
        if (pulsar_gdn_forward(&w, &c, 0)) return 1;
        const int ITER = T >= 8192 ? 3 : 10;
        CK(cudaEventRecord(e0));
        for (int it = 0; it < ITER; it++) pulsar_gdn_forward(&w, &c, 0);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("  T=%5d  %8.3f ms  %9.0f tok/s\n", T, ms / ITER, T / (ms / ITER * 1e-3));
    }
    CK(cudaDeviceSynchronize());
    return 0;
}

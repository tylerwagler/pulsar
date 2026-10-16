/* Mamba-2 block timings at Nemotron 3.5 Super's shapes (L296 N2).  Not a gate.
 *
 *   roofline   a measured streaming copy (read + write, DRAM-cold) and an f32 FMA loop on every SM;
 *   decode     1 and 16 sequences, one row each, slots rotated past L2 (DRAM-cold state);
 *   verify     one sequence, 8 rows with both captures, then the DELTA commit of all 8;
 *   prefill    one sequence, a 4096-row call (and 16 x 4096 rows, for the multi-sequence rate).
 * Each line names the bytes / flops it counts.  Run it alone on the GPU (gpuq, not --shared). */
#include "cuda/pulsar_cuda_mamba2.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

__global__ void copy_kernel(const float4 *__restrict__ a, float4 *__restrict__ b, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b[i] = a[i];
}
__global__ void fma_kernel(float *out, int iters) {
    float a = threadIdx.x * 1e-3f, b = 1.0001f, c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, c4 = 0.f, c5 = 0.f, c6 = 0.f, c7 = 0.f;
    for (int i = 0; i < iters; ++i) {
        c0 = fmaf(a, b, c0); c1 = fmaf(a, b, c1); c2 = fmaf(a, b, c2); c3 = fmaf(a, b, c3);
        c4 = fmaf(a, b, c4); c5 = fmaf(a, b, c5); c6 = fmaf(a, b, c6); c7 = fmaf(a, b, c7);
    }
    if (c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 == 12345.f) out[0] = 1.f;
}

static float elapsed(cudaEvent_t a, cudaEvent_t b) { float ms = 0; CK(cudaEventElapsedTime(&ms, a, b)); return ms; }

int main() {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    int sms = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));

    /* roofline */
    double copy_gbs, fma_tflops;
    {
        const size_t n = (size_t)256 << 20;   /* 1 GiB of float4 per side / 4 */
        float4 *a, *b;
        CK(cudaMalloc(&a, n * 16 / 4)); CK(cudaMalloc(&b, n * 16 / 4));
        const size_t n4 = n / 4;
        copy_kernel<<<sms * 8, 512>>>(a, b, n4);
        CK(cudaEventRecord(e0));
        for (int i = 0; i < 10; ++i) copy_kernel<<<sms * 8, 512>>>(a, b, n4);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        copy_gbs = 10.0 * 2.0 * (double)n4 * 16 / (elapsed(e0, e1) * 1e-3) / 1e9;
        CK(cudaFree(a)); CK(cudaFree(b));
        float *o; CK(cudaMalloc(&o, 4));
        const int iters = 1 << 16;
        fma_kernel<<<sms * 4, 512>>>(o, iters);
        CK(cudaEventRecord(e0));
        fma_kernel<<<sms * 4, 512>>>(o, iters);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        fma_tflops = (double)sms * 4 * 512 * iters * 8 * 2 / (elapsed(e0, e1) * 1e-3) / 1e12;
        CK(cudaFree(o));
    }
    printf("roofline: streaming copy %.1f GB/s (read + write, 1 GiB each side); f32 FMA %.1f TFLOP/s on %d SMs\n", copy_gbs,
           fma_tflops, sms);

    /* the Nemotron layer, parameters at plausible values */
    const pulsar_mamba2_geom g{128, 64, 128, 8, 4, 1024, 1e-5f};
    const uint32_t H = 128, HP = 8192, CC = pulsar_mamba2_conv_channels(&g), ROW = HP + CC + H;
    const uint32_t LD = (ROW + 7) / 8 * 8;
    const uint64_t rec = pulsar_mamba2_conv_state_elems(&g), ssmf = pulsar_mamba2_ssm_state_floats(&g),
                   updf = pulsar_mamba2_upd_floats(&g);
    std::vector<uint16_t> hp((size_t)CC * 4 + CC + 3 * 128 + HP + 64, 0x3f80);   /* 1.0 */
    for (size_t i = 0; i < (size_t)CC * 4; ++i) hp[i] = 0x3e80;                  /* conv taps 0.25 */
    for (size_t i = (size_t)CC * 5 + 128; i < (size_t)CC * 5 + 256; ++i) hp[i] = 0xc080;   /* dt_bias -4 */
    uint16_t *d_par;
    CK(cudaMalloc(&d_par, hp.size() * 2));
    CK(cudaMemcpy(d_par, hp.data(), hp.size() * 2, cudaMemcpyHostToDevice));
    pulsar_mamba2_layer L{g, d_par, d_par + (size_t)CC * 4, d_par + (size_t)CC * 5, d_par + (size_t)CC * 5 + 128,
                          d_par + (size_t)CC * 5 + 256, d_par + (size_t)CC * 5 + 384};
    const uint32_t kPool = 64, kMax = 16 * 4096;
    uint16_t *d_in, *d_out, *d_conv, *d_ccap;
    float *d_ssm, *d_scap;
    void *d_scr;
    int32_t *d_first, *d_slot;
    CK(cudaMalloc(&d_in, (size_t)4096 * LD * 2));
    std::vector<uint16_t> hin((size_t)4096 * LD);
    uint64_t z = 1;
    for (auto &v : hin) { z = z * 6364136223846793005ull + 1442695040888963407ull; v = (uint16_t)(0x3e00 + ((z >> 33) & 0x1ff)) | (uint16_t)((z >> 20) & 0x8000); }
    CK(cudaMemcpy(d_in, hin.data(), hin.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_out, (size_t)kMax * HP * 2));
    CK(cudaMalloc(&d_conv, kPool * rec * 2)); CK(cudaMemset(d_conv, 0, kPool * rec * 2));
    CK(cudaMalloc(&d_ssm, kPool * ssmf * 4)); CK(cudaMemset(d_ssm, 0, kPool * ssmf * 4));
    CK(cudaMalloc(&d_ccap, (size_t)16 * rec * 2)); CK(cudaMalloc(&d_scap, (size_t)16 * updf * 4));
    const size_t scr = pulsar_mamba2_scratch_bytes(&g, kMax);
    CK(cudaMalloc(&d_scr, scr));
    CK(cudaMalloc(&d_first, (kPool + 1) * 4)); CK(cudaMalloc(&d_slot, kPool * 4 * 64));

    /* S sequences of R rows each (input rows reused across sequences: the bytes moved are the same) */
    auto make_call = [&](uint32_t S, uint32_t R, uint32_t slot0, bool capture, const uint16_t *in, uint64_t ld_in) {
        pulsar_mamba2_call k{};
        k.n_seqs = S; k.n_rows = S * R; k.max_seq_rows = R; k.seq_first = d_first; k.seq_slot = d_slot + slot0;
        k.z = in; k.ld_z = ld_in; k.xbc = in + HP; k.ld_xbc = ld_in; k.dt = in + HP + CC; k.ld_dt = ld_in;
        k.conv_state = d_conv; k.ssm_state = d_ssm;
        k.conv_capture = capture ? d_ccap : nullptr; k.ssm_capture = capture ? d_scap : nullptr;
        k.scratch = d_scr; k.scratch_bytes = scr; k.out = d_out; k.ld_out = HP;
        return k;
    };
    auto set_rows = [&](uint32_t S, uint32_t R) {
        std::vector<int32_t> f(S + 1);
        for (uint32_t s = 0; s <= S; ++s) f[s] = (int32_t)(s * R);
        CK(cudaMemcpy(d_first, f.data(), f.size() * 4, cudaMemcpyHostToDevice));
    };
    /* slot tables: entry i * S + s = slot (i * S + s) % kPool, so successive calls walk the pool */
    {
        std::vector<int32_t> sl((size_t)kPool * 64);
        for (size_t i = 0; i < sl.size(); ++i) sl[i] = (int32_t)(i % kPool);
        CK(cudaMemcpy(d_slot, sl.data(), sl.size() * 4, cudaMemcpyHostToDevice));
    }
    const double state_bytes = (double)ssmf * 4 * 2 + (double)rec * 2 * 2;   /* SSM read + write, conv read + write */
    const double act_bytes_row = (double)LD * 2 + (double)HP * 2;            /* the in_proj row read, the output written */

    for (uint32_t S : {1u, 16u}) {
        set_rows(S, 1);
        const int iters = 200;
        for (int w = 0; w < 4; ++w) {
            auto k = make_call(S, 1, (uint32_t)(w * S) % kPool, false, d_in, LD);
            if (pulsar_mamba2_forward(&L, &k, 0)) return 2;
        }
        CK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) {
            auto k = make_call(S, 1, (uint32_t)(i * S) % kPool, false, d_in, LD);
            if (pulsar_mamba2_forward(&L, &k, 0)) return 2;
        }
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        const double us = elapsed(e0, e1) * 1e3 / iters;
        const double bytes = S * (state_bytes + act_bytes_row);
        printf("decode %2u seq x 1 row: %7.1f us a layer, %.1f MB moved (SSM state %.1f MB r+w), %.1f GB/s = %.0f%% of the copy roofline;"
               " x 40 layers = %.2f ms a step\n", S, us, bytes / 1e6, S * (double)ssmf * 8 / 1e6, bytes / (us * 1e-6) / 1e9,
               100.0 * bytes / (us * 1e-6) / 1e9 / copy_gbs, us * 40 / 1e3);
    }

    /* verify: 8 rows with capture, then the commit */
    {
        set_rows(1, 8);
        const int iters = 100;
        CK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) {
            auto k = make_call(1, 8, (uint32_t)i % kPool, true, d_in, LD);
            if (pulsar_mamba2_forward(&L, &k, 0)) return 2;
            if (pulsar_mamba2_commit(&g, d_ssm, (uint32_t)i % kPool, d_scap, 8, 0)) return 2;
        }
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        const double us = elapsed(e0, e1) * 1e3 / iters;
        printf("verify 1 seq x 8 rows + captures + commit of 8: %7.1f us a layer (capture %.0f KB a row)\n", us,
               ((double)updf * 4 + (double)rec * 2) / 1024);
        CK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) {
            auto k = make_call(1, 8, (uint32_t)i % kPool, false, d_in, LD);
            if (pulsar_mamba2_forward(&L, &k, 0)) return 2;
        }
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        printf("        the same 8 rows uncaptured:            %7.1f us\n", elapsed(e0, e1) * 1e3 / iters);
    }

    /* prefill */
    for (uint32_t S : {1u, 16u}) {
        const uint32_t R = 4096;
        set_rows(S, R);
        /* S > 1: each sequence reads its own copy of the 4096 input rows */
        uint16_t *in = d_in;
        uint16_t *big = nullptr;
        if (S > 1) {
            CK(cudaMalloc(&big, (size_t)S * R * LD * 2));
            for (uint32_t s = 0; s < S; ++s) CK(cudaMemcpy(big + (size_t)s * R * LD, d_in, (size_t)R * LD * 2, cudaMemcpyDeviceToDevice));
            in = big;
        }
        auto k = make_call(S, R, 0, false, in, LD);
        if (pulsar_mamba2_forward(&L, &k, 0)) return 2;
        const int iters = S == 1 ? 10 : 3;
        CK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) if (pulsar_mamba2_forward(&L, &k, 0)) return 2;
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        const double ms = elapsed(e0, e1) / iters;
        const double rows = (double)S * R;
        const double flops = rows * (double)ssmf * 5;   /* per state element: dA*S (1) + fma (2) + the y fma (2) */
        printf("prefill %2u seq x %u rows: %8.2f ms a layer, %8.0f rows/s a layer (x 40 layers: %.0f rows/s), scan %.1f TFLOP/s "
               "= %.0f%% of the FMA loop\n", S, R, ms, rows / (ms * 1e-3), rows / (ms * 1e-3) / 40, flops / (ms * 1e-3) / 1e12,
               100.0 * flops / (ms * 1e-3) / 1e12 / fma_tflops);
        if (big) CK(cudaFree(big));
    }
    return 0;
}

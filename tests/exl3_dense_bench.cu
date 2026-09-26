/* EXL3 dense-Linear microbenchmark (L251): the dense arm against pulsar's
 * MXFP8 decode GEMV and a plain wire-read roofline, at the dense shapes the
 * Qwen lane needs and a DeepSeek one, M = 1, 2, 4, 8, 16.
 *
 *   EXL3 K=2..5   ds4_exl3_dense_launch (split-K GEMV + epilogue), reading the
 *                 A8 slot the MXFP8 call below armed -- the same producer bytes
 *   MXFP8         pulsar_gpu_matmul_mxfp8_tensor with the M rows declared
 *                 DECODE rows (the engine's one-row GEMV at M = 1, the NT
 *                 batched GEMV at 2..16), the A8 slot armed by
 *                 pulsar_gpu_mxfp8_act_cache_encode_f32 -- the production arm;
 *                 the engine prints its own "W8A8 (E4M3 acts)" announce lines
 *   read          a streaming uint4 read of the same bytes: the wire roofline
 *
 * Then PREFILL widths (M = 128, 512, 2048): the same EXL3 arm (every row runs
 * the GEMV -- decode and prefill are one arithmetic) against MXFP8 with the
 * rows declared PREFILL rows, i.e. the cuBLASLt MX tensor-core GEMM a prefill
 * chunk runs.  "x read" is then time over one weight read, not a roofline.
 *
 * DRAM-cold: every format cycles through replicas of its weight totalling at
 * least 96 MB (GB10's L2 is 24 MB), so no timed call finds its weight in L2.
 * The launches are enqueued behind a spin kernel and timed with events, so the
 * number is GPU time back to back, not host enqueue time (a run whose enqueue
 * outlasted the spin says so).  Time = per call; GB/s = the weight's wire
 * bytes (EXL3: trellis + suh + svh; MXFP8: E4M3 data + E8M0 scales) / time.
 *
 *   tests/exl3_dense_bench [iters=120] [shape index, -1 = all] [M, 0 = all]
 */
#include "pulsar_gpu.h"
#include "cuda/pulsar_cuda_mx.cuh"
#include "cuda/mmq/ds4_exl3_dense.cuh"
#include "engine/exl3_trellis.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static uint64_t g_rng = 0x243F6A8885A308D3ull;
static uint32_t rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 16); }

static const double kMinRotateBytes = 96.0e6;

__global__ void spin_kernel(long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
}

__global__ void read_kernel(const uint4 *__restrict__ p, size_t n, unsigned *sink) {
    uint32_t acc = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        const uint4 a = p[i], b = p[i + stride], c = p[i + 2 * stride], d = p[i + 3 * stride];
        acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n; i += stride) { const uint4 a = p[i]; acc ^= a.x ^ a.y ^ a.z ^ a.w; }
    if (acc == 0x9E3779B9u) *sink = acc;          /* keeps the loads alive */
}

static int n_replicas(double bytes) {
    int n = (int)ceil(kMinRotateBytes / bytes);
    return n < 2 ? 2 : n;
}

/* Time `iters` calls of fn(i) back to back behind a spin kernel; us per call. */
template <typename F>
static double time_calls(int iters, F fn) {
    const cudaStream_t st = cudaStreamPerThread;
    for (int i = 0; i < 3; i++) fn(i);                  /* warm the code paths */
    CK(cudaStreamSynchronize(st));
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    spin_kernel<<<1, 1, 0, st>>>(100000000LL);          /* ~40-50 ms on GB10 */
    CK(cudaEventRecord(t0, st));
    const auto h0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) fn(i);
    const double host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count();
    CK(cudaEventRecord(t1, st));
    CK(cudaEventSynchronize(t1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    if (host_ms > 35.0) fprintf(stderr, "  (warning: enqueue took %.1f ms, may exceed the spin -- host-bound timing)\n", host_ms);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms * 1000.0 / iters;
}

int main(int argc, char **argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 120;
    const int only_shape = argc > 2 ? atoi(argv[2]) : -1;
    const int only_m = argc > 3 ? atoi(argv[3]) : 0;
    struct shape { int K, N; const char *what; };
    const shape shapes[] = {
        {2560, 10240, "DeltaNet in_proj_qkv"},
        {2560, 6144,  "DeltaNet in_proj_z"},
        {6144, 2560,  "DeltaNet out_proj"},
        {2560, 1280,  "2560->1280"},
        {4096, 1024,  "DeepSeek attn_q_a"},
    };
    const int widths[] = {1, 2, 4, 8, 16, 128, 512, 2048};
    const int kMaxRows = 2048;
    const int rates[] = {4, 6, 8, 10};

    if (!pulsar_gpu_init()) { fprintf(stderr, "no GPU\n"); return 2; }
    int sms = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    unsigned *d_sink = nullptr;
    CK(cudaMalloc((void **)&d_sink, 4));
    printf("exl3_dense_bench: %d SMs, %d calls per point, weights rotated over >= %.0f MB of replicas (DRAM-cold)\n",
           sms, iters, kMinRotateBytes / 1e6);
    printf("%-22s %-11s %4s | %-9s %8s %7s %6s | read-roofline us\n", "shape", "arm", "M", "bytes MB", "us", "GB/s", "x read");

    for (const shape &sh : shapes) {
        if (only_shape >= 0 && &sh - shapes != only_shape) continue;
        const int K = sh.K, N = sh.N;
        /* the activation: 16 rows of f32 the bench produces, armed as the A8
         * slot by the engine's own producer-side encoder */
        pulsar_gpu_tensor *x = pulsar_gpu_tensor_alloc((uint64_t)kMaxRows * K * 4);
        pulsar_gpu_tensor *out = pulsar_gpu_tensor_alloc((uint64_t)kMaxRows * N * 4);
        {
            std::vector<float> h((size_t)kMaxRows * K);
            for (auto &v : h) v = ((float)(rnd() >> 8) / 16777216.0f - 0.5f) * 4.0f;
            pulsar_gpu_tensor_write(x, 0, h.data(), h.size() * 4);
        }
        float *d_y = nullptr, *d_ws = nullptr;
        const size_t ws_bytes = ds4_exl3_dense_workspace_bytes(kMaxRows, K, N);
        CK(cudaMalloc((void **)&d_y, (size_t)kMaxRows * N * 4));
        CK(cudaMalloc((void **)&d_ws, ws_bytes));

        /* MXFP8 replicas: resident MXFP8_LT weights under one key, one offset each */
        const int KBp = pulsar_mx_kbp(K);
        const uint64_t f8_data = (uint64_t)K * N, f8_scale = (uint64_t)pulsar_mx_rup(N, 128) * KBp;
        const uint64_t f8_bytes = f8_data + f8_scale;
        const uint64_t f8_span = (uint64_t)N * ((K + 31) / 32) * 33;     /* the resolver's offset stride */
        const int n_f8 = n_replicas((double)f8_bytes);
        std::vector<pulsar_gpu_tensor *> f8d(n_f8), f8s(n_f8);
        static char key_storage[8];
        const void *key = key_storage + (&sh - shapes);
        {
            std::vector<uint8_t> hd(f8_data), hs(f8_scale);
            for (int r = 0; r < n_f8; r++) {
                for (auto &b : hd) { b = (uint8_t)rnd(); if ((b & 0x7f) == 0x7f) b ^= 1; }   /* no NaN codes */
                for (auto &b : hs) b = (uint8_t)(120 + rnd() % 8);
                f8d[r] = pulsar_gpu_tensor_alloc(f8_data);
                f8s[r] = pulsar_gpu_tensor_alloc(f8_scale);
                pulsar_gpu_tensor_write(f8d[r], 0, hd.data(), f8_data);
                pulsar_gpu_tensor_write(f8s[r], 0, hs.data(), f8_scale);
                if (!pulsar_gpu_register_fp8_lt_weight_resident(key, (uint64_t)r * f8_span, K, N, f8d[r], f8s[r])) {
                    fprintf(stderr, "resident MXFP8 registration refused\n"); return 2;
                }
            }
        }
        const uint64_t map_size = (uint64_t)n_f8 * f8_span;

        /* EXL3 replicas per rate */
        struct exl3_set { int k2; uint64_t stride; int n; uint8_t *arena; };
        std::vector<exl3_set> sets;
        for (int k2 : rates) {
            uint64_t tb, sb, stride;
            if (!exl3_expert_layout(K, N, k2, &tb, &sb, &stride)) { fprintf(stderr, "layout\n"); return 2; }
            exl3_set s{k2, stride, n_replicas((double)stride), nullptr};
            CK(cudaMalloc((void **)&s.arena, s.stride * s.n));
            std::vector<uint8_t> h(s.stride);
            for (int r = 0; r < s.n; r++) {
                uint16_t *t = (uint16_t *)h.data();
                for (uint64_t i = 0; i < tb / 2; i++) t[i] = (uint16_t)rnd();
                for (uint64_t i = 0; i < (uint64_t)(K + N); i++)
                    t[tb / 2 + i] = exl3_f32_to_f16((float)(0.025 * ((rnd() & 1) ? 1 : -1)));
                CK(cudaMemcpy(s.arena + (size_t)r * s.stride, h.data(), s.stride, cudaMemcpyHostToDevice));
            }
            sets.push_back(s);
        }

        /* the read roofline, per byte size, over the same replica arenas */
        auto read_us = [&](const uint8_t *arena, uint64_t bytes, uint64_t stride, int n) {
            return time_calls(iters, [&](int i) {
                read_kernel<<<sms * 8, 256, 0, cudaStreamPerThread>>>(
                    (const uint4 *)(arena + (size_t)(i % n) * stride), bytes / 16, d_sink);
            });
        };
        std::vector<double> roof(sets.size());
        for (size_t j = 0; j < sets.size(); j++) roof[j] = read_us(sets[j].arena, sets[j].stride, sets[j].stride, sets[j].n);
        /* MXFP8's bytes live in two tensors per replica; the roofline reads one contiguous run
         * of the same size per replica */
        uint8_t *f8_scratch = nullptr;
        const int n_f8r = n_replicas((double)f8_bytes);
        CK(cudaMalloc((void **)&f8_scratch, (size_t)f8_bytes * n_f8r));
        CK(cudaMemset(f8_scratch, 0x11, (size_t)f8_bytes * n_f8r));
        const double roof_f8 = read_us(f8_scratch, f8_bytes, f8_bytes, n_f8r);
        cudaFree(f8_scratch);

        for (int M : widths) {
            if (only_m && M != only_m) continue;
            if (!pulsar_gpu_mxfp8_act_cache_encode_f32(x, M, K)) { fprintf(stderr, "act encode refused\n"); return 2; }
            const void *xq = nullptr, *xs = nullptr;
            int kbp = 0;
            if (!pulsar_gpu_mxfp8_act_cache_get_e4m3(x, M, K, &xq, &xs, &kbp) || kbp != KBp) {
                fprintf(stderr, "no E4M3 slot for (%d, %d) -- refusing\n", M, K); return 2;
            }
            /* MXFP8: decode rows up to the decode cap, prefill rows past it */
            const bool prefill = M > 16;
            const int n_it = prefill ? (iters + 9) / 10 : iters;
            pulsar_gpu_matmul_set_batch_decode_rows(prefill ? 0 : M);
            bool f8_ok = true;
            const double t_f8 = time_calls(n_it, [&](int i) {
                const int r = i % n_f8;
                if (!pulsar_gpu_matmul_mxfp8_tensor(out, key, map_size, (uint64_t)r * f8_span, K, N, x, M)) f8_ok = false;
            });
            pulsar_gpu_matmul_set_batch_decode_rows(0);
            if (!f8_ok) { fprintf(stderr, "MXFP8 GEMV refused at M=%d\n", M); return 2; }
            printf("%-22s %-11s %4d | %9.2f %8.1f %7.1f %6.2f | %.1f\n", sh.what, prefill ? "MXFP8 pf" : "MXFP8", M, f8_bytes / 1e6, t_f8,
                   f8_bytes / (t_f8 * 1e-6) / 1e9, t_f8 / roof_f8, roof_f8);
            /* EXL3 at every rate, from the same slot */
            for (size_t j = 0; j < sets.size(); j++) {
                const exl3_set &s = sets[j];
                bool ok = true;
                const double t = time_calls(n_it, [&](int i) {
                    if (ds4_exl3_dense_launch(s.arena + (size_t)(i % s.n) * s.stride, s.k2, xq, xs, d_y, M, K, N,
                                              d_ws, ws_bytes, cudaStreamPerThread)) ok = false;
                });
                if (!ok) { fprintf(stderr, "EXL3 arm refused\n"); return 2; }
                char arm[32];
                snprintf(arm, sizeof arm, "EXL3 K=%d", s.k2 / 2);
                printf("%-22s %-11s %4d | %9.2f %8.1f %7.1f %6.2f | %.1f\n", sh.what, arm, M, s.stride / 1e6, t,
                       s.stride / (t * 1e-6) / 1e9, t / roof[j], roof[j]);
            }
        }
        printf("%-22s splits %d, EXL3 rotation replicas %d..%d, MXFP8 %d\n", sh.what, ds4_exl3_dense_splits(K, N),
               sets.back().n, sets.front().n, n_f8);
        for (auto &s : sets) cudaFree(s.arena);
        for (int r = 0; r < n_f8; r++) { pulsar_gpu_tensor_free(f8d[r]); pulsar_gpu_tensor_free(f8s[r]); }
        cudaFree(d_y); cudaFree(d_ws);
        pulsar_gpu_act_slot_drop(x);
        pulsar_gpu_tensor_free(x); pulsar_gpu_tensor_free(out);
    }
    return 0;
}

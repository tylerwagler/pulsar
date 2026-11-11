/* EXL3 dense-Linear microbenchmark (L251): the dense arm against pulsar's
 * MXFP8 decode GEMV and a plain wire-read roofline, at the dense shapes the
 * Qwen lane needs and a DeepSeek one, M = 1, 2, 4, 8, 16 and the two-pass widths 17, 24, 32.
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
 *   tests/exl3_dense_bench [iters=120] [shape index, -1 = all] [M, 0 = all] [set: qwen (default) | gemma]
 *
 * The gemma set (L338): Gemma 4 26B-A4B's dense Linears and its K=6 tied head, at the rates its pack stores (4, 6).
 * The MXFP8 arm needs the MX tensor cores (sm_100+); below them it is not run (said once).
 *
 * THE ROUND MODE (L380): the GEMV's CTA forms priced at the launches a model makes a decode round --
 *
 *   tests/exl3_dense_bench round [iters=200] [repeats=3] [M list=1,2,4,8] [set: qwen (default) | nemotron]
 *                                [k2, 0 = the set's rate] [arms: rule (default) | 8w] [shape index, -1 = all]
 *
 * see round_main below.  How the 8-warp CTA's rule (ds4_exl3_dense_gemv_arm) was priced on GB10 (gpuq #702 / #703:
 * `round 200 5 1,2,4,8 qwen`) and how it is re-priced on another device, rate or model.  Build with the served arch
 * (make tests/exl3_dense_bench CUDA_ARCH=sm_120f) and run with the GPU to itself: no pulsar-server on it.
 */
#include "pulsar_gpu.h"
#include "cuda/pulsar_cuda_mx.cuh"
#include "cuda/mmq/ds4_exl3_dense.cuh"
#include "engine/exl3_trellis.h"

#include <cuda_runtime.h>

#include <algorithm>
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

/* THE ROUND MODE (L380).  The EXL3 dense launches a model makes a decode round, each as its family makes it
 * (ds4_exl3_dense_multi_launch: one PREP + one PDL GEMV over the launch's projections, decode rows, f32 outputs), at
 * one rate, synthetic weights, under each CTA form:
 *
 *   16w    the 16-warp CTA forced on every launch (ds4_exl3_dense_arm_force) -- the arm before L380
 *   rule   the shipped selection (ds4_exl3_dense_gemv_arm on this device); the "form" column says which it took
 *   8w     (asked for with `8w`) the 8-warp CTA forced, one-pass launches only (M <= 16)
 *
 * qwen (Qwen3.8-Flash-Next, K = 5): a round's launches and their counts --
 *   GDN in_proj_qkv + in_proj_z   ONE multi launch 2560 -> 10240 + 6144      x36
 *   QSA q + k + v                 ONE multi launch 2560 -> 12288 + 512 + 512 x12
 *   GDN out_proj / QSA o_proj     6144 -> 2560                               x36 + x12
 *   shared gate + up              ONE multi launch 2560 -> 640 + 640         x48
 *   shared down                   640 -> 2560                                x48
 *   PLE key_proj, value_proj      2560 -> 10240, 2560 -> 2560                x1 each
 *   lm_head                       2560 -> 248320                             x1
 * nemotron (Nemotron-3.5 Super's trunk, K = 4): the shapes are the profile's (PULSAR_NEMOTRON_H_SHAPE_SUPER) through
 * the forward's launches (nemotron_h_forward.cpp), the counts the model's 40 Mamba-2 / 8 attention / 40 MoE blocks
 * (nemotron_h.h) --
 *   Mamba-2 in_proj               4096 -> 18560 ([z 8192 | x B C 10240 | dt 128]) x40
 *   Mamba-2 out_proj              8192 -> 4096                               x40
 *   attention q + k + v           ONE multi launch 4096 -> 4096 + 256 + 256  x8
 *   attention o_proj              4096 -> 4096                               x8
 *   latent in / latent out        4096 -> 1024, 1024 -> 4096                 x40 each
 *   shared up / shared down       4096 -> 5376, 5376 -> 4096                 x40 each
 *   lm_head                       4096 -> 131072                             x1
 *   NOT in it: the MTP drafter's module (eh_proj, its attention and MoE: a draft step's, not the trunk's); and a
 *   decode step runs the shared expert beside the routed chain on a second stream (L296 dec2), which a launch timed
 *   alone does not see.  Which of these Linears a served container stores as EXL3 is the container's: check its
 *   layouts before reading the totals as that model's round.
 *
 * Per (launch, M): one launch of each form on the same weights and rows into canary-filled outputs -- every form's
 * bytes must be the 16-warp form's (the forms are one arithmetic; a differing byte fails the run) -- then `repeats`
 * rounds of timings with the forms' order rotated, each `iters` calls back to back behind a spin kernel, DRAM-cold
 * (each call takes the next replica of every projection's weight, >= 96 MB of replicas a launch).  Medians; then the
 * per-round totals at the counts above.  The description (ds4_exl3_dense_gemv_describe) names the kernel each form
 * ran: its warps, registers and CTAs an SM.
 * Returns 0; 2 on a usage, setup or launch failure; 3 when a form's output is not the 16-warp form's. */
static int round_main(int argc, char **argv) {
    const int iters = argc > 2 ? atoi(argv[2]) : 200;
    const int repeats = argc > 3 ? atoi(argv[3]) : 3;
    std::vector<int> Ms;
    {
        const char *l = argc > 4 ? argv[4] : "1,2,4,8";
        for (const char *q = l; *q;) {
            Ms.push_back(atoi(q));
            while (*q && *q != ',') q++;
            if (*q == ',') q++;
        }
    }
    const char *set = argc > 5 ? argv[5] : "qwen";
    const bool nemotron = !strcmp(set, "nemotron");
    const int k2 = argc > 6 && atoi(argv[6]) ? atoi(argv[6]) : nemotron ? 8 : 10;
    const char *arms_s = argc > 7 ? argv[7] : "rule";
    const bool with_8w = !strcmp(arms_s, "8w");
    const int only_shape = argc > 8 ? atoi(argv[8]) : -1;
    int maxM = 0;
    bool usage = iters < 1 || repeats < 1 || Ms.empty() || (!nemotron && strcmp(set, "qwen")) ||
                 (!with_8w && strcmp(arms_s, "rule")) || !ds4_exl3_dense_rate_supported(k2);
    for (int M : Ms) {
        usage = usage || M < 1 || M > 32;   /* one launch: 16-row passes, two at most */
        maxM = std::max(maxM, M);
    }
    if (usage) {
        fprintf(stderr, "usage: exl3_dense_bench round [iters=200] [repeats=3] [M list=1,2,4,8 (1..32)] [qwen|nemotron] "
                        "[k2, 0 = the set's] [rule|8w] [shape]\n");
        return 2;
    }
    struct lshape { const char *what; int K, n, N[DS4_EXL3_DENSE_MAX_PROJ], count; };
    const lshape qwen_shapes[] = {
        {"GDN qkv+z  2560->10240+6144",  2560, 2, {10240, 6144, 0, 0}, 36},
        {"QSA q+k+v  2560->12288+2x512", 2560, 3, {12288, 512, 512, 0}, 12},
        {"GDN out / QSA o 6144->2560",   6144, 1, {2560, 0, 0, 0}, 48},
        {"shared gate+up 2560->2x640",   2560, 2, {640, 640, 0, 0}, 48},
        {"shared down 640->2560",        640,  1, {2560, 0, 0, 0}, 48},
        {"PLE key 2560->10240",          2560, 1, {10240, 0, 0, 0}, 1},
        {"PLE value 2560->2560",         2560, 1, {2560, 0, 0, 0}, 1},
        {"lm_head 2560->248320",         2560, 1, {248320, 0, 0, 0}, 1},
    };
    const lshape nemotron_shapes[] = {
        {"Mamba in_proj 4096->18560",    4096, 1, {18560, 0, 0, 0}, 40},
        {"Mamba out_proj 8192->4096",    8192, 1, {4096, 0, 0, 0}, 40},
        {"attn q+k+v 4096->4096+2x256",  4096, 3, {4096, 256, 256, 0}, 8},
        {"attn o_proj 4096->4096",       4096, 1, {4096, 0, 0, 0}, 8},
        {"latent in 4096->1024",         4096, 1, {1024, 0, 0, 0}, 40},
        {"latent out 1024->4096",        1024, 1, {4096, 0, 0, 0}, 40},
        {"shared up 4096->5376",         4096, 1, {5376, 0, 0, 0}, 40},
        {"shared down 5376->4096",       5376, 1, {4096, 0, 0, 0}, 40},
        {"lm_head 4096->131072",         4096, 1, {131072, 0, 0, 0}, 1},
    };
    const lshape *shapes = nemotron ? nemotron_shapes : qwen_shapes;
    const int n_shapes = nemotron ? (int)(sizeof nemotron_shapes / sizeof *nemotron_shapes)
                                  : (int)(sizeof qwen_shapes / sizeof *qwen_shapes);
    struct form { const char *name; int force; };
    const form forms[] = {{"16w", DS4_EXL3_DENSE_ARM_16W}, {"rule", 0}, {"8w", DS4_EXL3_DENSE_ARM_8W}};
    const int n_forms = with_8w ? 3 : 2;

    if (!pulsar_gpu_init()) { fprintf(stderr, "no GPU\n"); return 2; }
    int sms = 0, cc_major = 0, cc_minor = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    CK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0));
    CK(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, 0));
    size_t gfree = 0, gtotal = 0;
    CK(cudaMemGetInfo(&gfree, &gtotal));
    unsigned *d_sink = nullptr;
    CK(cudaMalloc((void **)&d_sink, 4));
    printf("exl3_dense_bench round: set %s, K=%.1f bits, sm_%d%d, %d SMs, %.1f of %.1f GB free, %d calls x %d repeats a "
           "point, DRAM-cold (>= %.0f MB of replicas a launch)\n", set, k2 / 2.0, cc_major, cc_minor, sms, gfree / 1e9,
           gtotal / 1e9, iters, repeats, kMinRotateBytes / 1e6);
    printf("columns: NT = n-tiles a CTA (2: 32 outputs, 4: 64), grid = GEMV CTAs; form = the warps the rule took; reg, "
           "c/SM = registers a thread and CTAs resident an SM of the 16w / rule%s kernels; us = median per launch (prep + "
           "GEMV); GB/s = weight bytes / us; r/16 = rule us / 16w us\n", with_8w ? " / 8w" : "");
    printf("%-30s %2s %7s %2s %5s %4s %-11s %-7s | %9s %9s %9s | %6s %6s | %6s %6s | %s\n", "launch", "M", "MB", "NT", "grid",
           "form", "reg", "c/SM", "16w us", "rule us", "8w us", "GB/s16", "GB/s r", "r/16", "8/16", "bytes");

    struct result { double bytes, us[3]; bool have, have8; };
    std::vector<std::vector<result>> res(n_shapes, std::vector<result>(Ms.size(), result{0, {0, 0, 0}, false, false}));
    bool launch_failed = false, bytes_failed = false;
    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    };

    for (int si = 0; si < n_shapes; si++) {
        const lshape &sh = shapes[si];
        if (only_shape >= 0 && si != only_shape) continue;
        const int K = sh.K, n = sh.n;
        uint64_t tb[DS4_EXL3_DENSE_MAX_PROJ], stride[DS4_EXL3_DENSE_MAX_PROJ], pitch[DS4_EXL3_DENSE_MAX_PROJ];
        double bytes = 0;
        int c32 = 0;
        for (int j = 0; j < n; j++) {
            uint64_t sb;
            if (!exl3_linear_layout(K, sh.N[j], k2, &tb[j], &sb, &stride[j])) { fprintf(stderr, "round: layout\n"); return 2; }
            pitch[j] = (stride[j] + 255) & ~(uint64_t)255;   /* replicas 16-byte aligned */
            bytes += (double)stride[j];
            c32 += (int)(exl3_out_stored(sh.N[j]) / 32);
        }
        const int nrep = n_replicas(bytes);
        size_t need = (size_t)maxM * K * 2;
        for (int j = 0; j < n; j++) need += pitch[j] * nrep + (size_t)maxM * sh.N[j] * 4;
        CK(cudaMemGetInfo(&gfree, &gtotal));
        if (need + (256u << 20) > gfree) {
            fprintf(stderr, "round: %s needs %.2f GB of device memory, %.2f GB free -- refusing\n", sh.what, need / 1e9,
                    gfree / 1e9);
            return 2;
        }
        /* weights: random trellis words, +-0.025 scales (the bench's fixture), every replica its own bytes */
        uint8_t *arena[DS4_EXL3_DENSE_MAX_PROJ] = {};
        float *d_y[DS4_EXL3_DENSE_MAX_PROJ] = {};
        for (int j = 0; j < n; j++) {
            CK(cudaMalloc((void **)&arena[j], pitch[j] * nrep));
            CK(cudaMalloc((void **)&d_y[j], (size_t)maxM * sh.N[j] * 4));
            std::vector<uint8_t> h(pitch[j]);
            const uint64_t scales = (uint64_t)K + exl3_out_stored(sh.N[j]);
            for (int r = 0; r < nrep; r++) {
                uint16_t *t = (uint16_t *)h.data();
                for (uint64_t i = 0; i < tb[j] / 2; i++) t[i] = (uint16_t)rnd();
                for (uint64_t i = 0; i < scales; i++) t[tb[j] / 2 + i] = exl3_f32_to_f16((float)(0.025 * ((rnd() & 1) ? 1 : -1)));
                CK(cudaMemcpy(arena[j] + (size_t)r * pitch[j], h.data(), pitch[j], cudaMemcpyHostToDevice));
            }
        }
        /* the rows: bf16, the family's block input */
        uint16_t *d_x = nullptr;
        CK(cudaMalloc((void **)&d_x, (size_t)maxM * K * 2));
        {
            std::vector<uint16_t> hb((size_t)maxM * K);
            for (auto &v : hb) {
                const __nv_bfloat16 b = __float2bfloat16(((float)(rnd() >> 8) / 16777216.0f - 0.5f) * 4.0f);
                std::memcpy(&v, &b, sizeof(uint16_t));
            }
            CK(cudaMemcpy(d_x, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice));
        }
        ds4_exl3_dense_proj p0[DS4_EXL3_DENSE_MAX_PROJ];
        for (int j = 0; j < n; j++) p0[j] = ds4_exl3_dense_proj{arena[j], k2, sh.N[j], d_y[j], nullptr};
        const size_t ws_bytes = ds4_exl3_dense_multi_workspace_bytes(p0, n, maxM, K);
        void *d_ws = nullptr;
        if (!ws_bytes) { fprintf(stderr, "round: %s: the arm refuses the shape\n", sh.what); return 2; }
        CK(cudaMalloc(&d_ws, ws_bytes));
        /* one launch as the family makes it: replica i of every projection's weight */
        auto launch = [&](int i, int M) {
            ds4_exl3_dense_proj p[DS4_EXL3_DENSE_MAX_PROJ];
            for (int j = 0; j < n; j++) {
                p[j] = p0[j];
                p[j].w = arena[j] + (size_t)(i % nrep) * pitch[j];
            }
            return ds4_exl3_dense_multi_launch(p, n, d_x, M, K, d_ws, ws_bytes, cudaStreamPerThread, false) == 0;
        };

        for (size_t mi = 0; mi < Ms.size(); mi++) {
            const int M = Ms[mi];
            /* the kernel each form runs here; the forced 8-warp CTA has none past 16 rows */
            ds4_exl3_dense_gemv_desc d[3] = {};
            bool run[3] = {};
            for (int f = 0; f < n_forms; f++) {
                ds4_exl3_dense_arm_force(forms[f].force);
                run[f] = ds4_exl3_dense_gemv_describe(k2, M, c32, &d[f]) == 0;
            }
            ds4_exl3_dense_arm_force(0);
            if (!run[0] || !run[1]) { fprintf(stderr, "round: no launch description for %s M=%d\n", sh.what, M); return 2; }
            /* the bytes: every form on the same weights and rows */
            std::vector<std::vector<float>> y0(n);
            bool ok = true;
            size_t n_diff = 0, n_nan = 0;
            for (int f = 0; f < n_forms && ok; f++) {
                if (!run[f]) continue;
                ds4_exl3_dense_arm_force(forms[f].force);
                for (int j = 0; j < n; j++) CK(cudaMemset(d_y[j], 0xff, (size_t)M * sh.N[j] * 4));   /* NaN */
                ok = launch(0, M);
                CK(cudaStreamSynchronize(cudaStreamPerThread));
                for (int j = 0; j < n && ok; j++) {
                    std::vector<float> y((size_t)M * sh.N[j]);
                    CK(cudaMemcpy(y.data(), d_y[j], y.size() * 4, cudaMemcpyDeviceToHost));
                    for (float v : y) n_nan += !(v == v);
                    if (f == 0) y0[j] = y;
                    else n_diff += memcmp(y.data(), y0[j].data(), y.size() * 4) != 0;
                }
            }
            const bool same = ok && n_diff == 0 && n_nan == 0;
            if (ok && !same) bytes_failed = true;

            /* timing: the forms in turn, the order rotated a repeat */
            std::vector<double> t[3];
            for (int r = 0; r < repeats && ok; r++)
                for (int a = 0; a < n_forms && ok; a++) {
                    const int f = (a + r) % n_forms;
                    if (!run[f]) continue;
                    ds4_exl3_dense_arm_force(forms[f].force);
                    t[f].push_back(time_calls(iters, [&](int i) { if (!launch(i, M)) ok = false; }));
                }
            ds4_exl3_dense_arm_force(0);
            if (!ok) {
                launch_failed = true;
                printf("%-30s %2d  LAUNCH FAILED\n", sh.what, M);
                continue;
            }
            const double u16 = median(t[0]), ur = median(t[1]), u8 = run[2] ? median(t[2]) : 0.0;
            res[si][mi] = result{bytes, {u16, ur, u8}, true, run[2]};
            char regs[24], occ[24], c8[16] = "-", r8[16] = "-";
            if (run[2]) {
                snprintf(regs, sizeof regs, "%d/%d/%d", d[0].regs, d[1].regs, d[2].regs);
                snprintf(occ, sizeof occ, "%d/%d/%d", d[0].ctas_per_sm, d[1].ctas_per_sm, d[2].ctas_per_sm);
                snprintf(c8, sizeof c8, "%.1f", u8);
                snprintf(r8, sizeof r8, "%.3f", u8 / u16);
            } else {
                snprintf(regs, sizeof regs, "%d/%d", d[0].regs, d[1].regs);
                snprintf(occ, sizeof occ, "%d/%d", d[0].ctas_per_sm, d[1].ctas_per_sm);
            }
            printf("%-30s %2d %7.2f %2d %5d %4s %-11s %-7s | %9.1f %9.1f %9s | %6.1f %6.1f | %6.3f %6s | %s\n", sh.what, M,
                   bytes / 1e6, d[1].nt, d[1].ctas, d[1].arm == DS4_EXL3_DENSE_ARM_8W ? "8w" : "16w", regs, occ, u16, ur, c8,
                   bytes / (u16 * 1e-6) / 1e9, bytes / (ur * 1e-6) / 1e9, ur / u16, r8,
                   same ? "identical" : "*** DIFFER (or NaN / unwritten) ***");
            for (int f = 0; f < n_forms; f++) {
                if (!run[f]) continue;
                printf("    %s us:", forms[f].name);
                for (double v : t[f]) printf(" %.1f", v);
            }
            printf("   (%d replicas)\n", nrep);
            fflush(stdout);
        }
        for (int j = 0; j < n; j++) { cudaFree(arena[j]); cudaFree(d_y[j]); }
        cudaFree(d_x);
        cudaFree(d_ws);
    }

    /* the round: every launch at its count */
    printf("\nper-round totals (median us x the launch's count a round; the last launch is the head):\n");
    printf("%2s | %10s %10s %7s | %10s %7s | %10s %10s %7s\n", "M", "16w ms", "rule ms", "r/16", "8w ms", "8/16",
           "no-head 16", "no-head r", "r/16");
    for (size_t mi = 0; mi < Ms.size(); mi++) {
        double ms[3] = {0, 0, 0}, nh[2] = {0, 0};
        bool all = true, all8 = with_8w;
        for (int si = 0; si < n_shapes; si++) {
            const result &r = res[si][mi];
            if (!r.have) { all = false; continue; }
            all8 = all8 && r.have8;
            for (int f = 0; f < 3; f++) ms[f] += r.us[f] * shapes[si].count / 1e3;
            if (si != n_shapes - 1)
                for (int f = 0; f < 2; f++) nh[f] += r.us[f] * shapes[si].count / 1e3;
        }
        char m8[16] = "-", r8[16] = "-";
        if (all8) {
            snprintf(m8, sizeof m8, "%.3f", ms[2]);
            snprintf(r8, sizeof r8, "%.3f", ms[0] > 0 ? ms[2] / ms[0] : 0.0);
        }
        printf("%2d | %10.3f %10.3f %7.3f | %10s %7s | %10.3f %10.3f %7.3f%s\n", Ms[mi], ms[0], ms[1],
               ms[0] > 0 ? ms[1] / ms[0] : 0.0, m8, r8, nh[0], nh[1], nh[0] > 0 ? nh[1] / nh[0] : 0.0,
               all ? "" : "   (PARTIAL: not every launch ran)");
    }
    if (launch_failed) { fprintf(stderr, "round: *** A LAUNCH FAILED ***\n"); return 2; }
    if (bytes_failed) {
        fprintf(stderr, "round: *** A FORM'S OUTPUT IS NOT THE 16-WARP FORM'S (or holds a NaN / an unwritten output): the "
                        "timings above do not compare the same work, and exl3-dense-gate should be failing ***\n");
        return 3;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "round")) return round_main(argc, argv);
    const int iters = argc > 1 ? atoi(argv[1]) : 120;
    const int only_shape = argc > 2 ? atoi(argv[2]) : -1;
    const int only_m = argc > 3 ? atoi(argv[3]) : 0;
    struct shape { int K, N; const char *what; };
    const bool gemma = argc > 4 && !strcmp(argv[4], "gemma");
    const shape gemma_shapes[] = {
        {2816, 4096,   "g4 q_proj (window)"},
        {2816, 8192,   "g4 q_proj (global)"},
        {2816, 2048,   "g4 k/v_proj (window)"},
        {4096, 2816,   "g4 o_proj (window)"},
        {2816, 2112,   "g4 mlp gate/up"},
        {2112, 2816,   "g4 mlp down"},
        {2816, 262144, "g4 head (tied)"},
    };
    const shape qwen_shapes[] = {
        {2560, 10240, "DeltaNet in_proj_qkv"},
        {2560, 6144,  "DeltaNet in_proj_z"},
        {6144, 2560,  "DeltaNet out_proj"},
        {2560, 1280,  "2560->1280"},
        {4096, 1024,  "DeepSeek attn_q_a"},
        {2560, 640,   "Qwen shared gate/up"},
        {640, 2560,   "Qwen shared down"},
    };
    const shape *shapes_p = gemma ? gemma_shapes : qwen_shapes;
    const int n_shapes = gemma ? (int)(sizeof gemma_shapes / sizeof *gemma_shapes) : (int)(sizeof qwen_shapes / sizeof *qwen_shapes);
    const int widths[] = {1, 2, 4, 8, 16, 17, 24, 32, 128, 512, 2048};   /* 17..32: the GEMV's two-pass verify widths (L341 / L349) */
    const int kMaxRows = 2048;
    const std::vector<int> rates = gemma ? std::vector<int>{8, 12} : std::vector<int>{4, 6, 8, 10};

    if (!pulsar_gpu_init()) { fprintf(stderr, "no GPU\n"); return 2; }
    int sms = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    unsigned *d_sink = nullptr;
    CK(cudaMalloc((void **)&d_sink, 4));
    printf("exl3_dense_bench: %d SMs, %d calls per point, weights rotated over >= %.0f MB of replicas (DRAM-cold)\n",
           sms, iters, kMinRotateBytes / 1e6);
    int cc_major = 0;
    CK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0));
    const bool with_f8 = cc_major >= 10;
    if (!with_f8) printf("exl3_dense_bench: sm_%d: no MX tensor cores -- the MXFP8 arm is not run\n", cc_major * 10);
    printf("%-22s %-11s %4s | %-9s %8s %7s %6s | read-roofline us\n", "shape", "arm", "M", "bytes MB", "us", "GB/s", "x read");

    for (int si = 0; si < n_shapes; si++) {
        const shape &sh = shapes_p[si];
        if (only_shape >= 0 && si != only_shape) continue;
        if (gemma && si == n_shapes - 1 && only_shape < 0) continue;   /* the head (2.2 GB of replicas) only by index */
        const int K = sh.K, N = sh.N;
        /* the activation: 16 rows of f32 the bench produces, armed as the A8
         * slot by the engine's own producer-side encoder */
        pulsar_gpu_tensor *x = pulsar_gpu_tensor_alloc((uint64_t)kMaxRows * K * 4);
        /* L251 / ac69748f: the EXL3 dense arm reads BF16 and there is no f32 -> bf16 activation
         * encoder, so the harness carries its own bf16 activation BESIDE the f32 one the MXFP8 arm
         * needs.  Filled from the same values through __float2bfloat16, so both arms still see
         * identical rows -- which is the whole point of this comparison. */
        pulsar_gpu_tensor *xb = pulsar_gpu_tensor_alloc((uint64_t)kMaxRows * K * 2);
        pulsar_gpu_tensor *out = pulsar_gpu_tensor_alloc((uint64_t)kMaxRows * N * 4);
        {
            std::vector<float> h((size_t)kMaxRows * K);
            for (auto &v : h) v = ((float)(rnd() >> 8) / 16777216.0f - 0.5f) * 4.0f;
            pulsar_gpu_tensor_write(x, 0, h.data(), h.size() * 4);
            std::vector<uint16_t> hb(h.size());
            for (size_t i = 0; i < h.size(); i++) {
                const __nv_bfloat16 b = __float2bfloat16(h[i]);
                std::memcpy(&hb[i], &b, sizeof(uint16_t));
            }
            pulsar_gpu_tensor_write(xb, 0, hb.data(), hb.size() * 2);
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
        const int n_f8 = with_f8 ? n_replicas((double)f8_bytes) : 0;
        std::vector<pulsar_gpu_tensor *> f8d(n_f8), f8s(n_f8);
        static char key_storage[8];
        const void *key = key_storage + si;
        if (with_f8) {
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
        double roof_f8 = 0;
        if (with_f8) {
            const int n_f8r = n_replicas((double)f8_bytes);
            CK(cudaMalloc((void **)&f8_scratch, (size_t)f8_bytes * n_f8r));
            CK(cudaMemset(f8_scratch, 0x11, (size_t)f8_bytes * n_f8r));
            roof_f8 = read_us(f8_scratch, f8_bytes, f8_bytes, n_f8r);
            cudaFree(f8_scratch);
        }

        for (int M : widths) {
            if (only_m && M != only_m) continue;
            if (with_f8 && !pulsar_gpu_mxfp8_act_cache_encode_f32(x, M, K)) { fprintf(stderr, "act encode refused\n"); return 2; }
            /* L251 / ac69748f: the EXL3 arm takes the bf16 plane, not an E4M3 slot. */
            void *xq = nullptr;
            if (!pulsar_gpu_bf16_act_slot(xb, M, K, &xq) || !xq) {
                fprintf(stderr, "no bf16 activation for (%d, %d) -- refusing\n", M, K); return 2;
            }
            pulsar_gpu_bf16_act_note(xb, M, K);
            /* MXFP8: decode rows up to the decode cap, prefill rows past it */
            const bool prefill = M > 16;
            const int n_it = M > 32 ? (iters + 9) / 10 : iters;   /* every GEMV width at full count */
            if (with_f8) {
                pulsar_gpu_matmul_set_batch_decode_rows(prefill ? 0 : M);
                bool f8_ok = true;
                const double t_f8 = time_calls(n_it, [&](int i) {
                    const int r = i % n_f8;
                    if (!pulsar_gpu_matmul_mxfp8_tensor(out, key, map_size, (uint64_t)r * f8_span, K, N, x, M)) f8_ok = false;
                });
                pulsar_gpu_matmul_set_batch_decode_rows(0);
                if (!f8_ok) { fprintf(stderr, "MXFP8 GEMV refused at M=%d\n", M); return 2; }
                printf("%-22s %-11s %4d | %9.2f %8.1f %7.1f %6.2f | %.1f\n", sh.what, prefill ? "MXFP8 pf" : "MXFP8", M,
                       f8_bytes / 1e6, t_f8, f8_bytes / (t_f8 * 1e-6) / 1e9, t_f8 / roof_f8, roof_f8);
            }
            /* EXL3 at every rate, from the same rows (bf16) */
            for (size_t j = 0; j < sets.size(); j++) {
                const exl3_set &s = sets[j];
                bool ok = true;
                const double t = time_calls(n_it, [&](int i) {
                    if (ds4_exl3_dense_launch(s.arena + (size_t)(i % s.n) * s.stride, s.k2, xq, d_y, M, K, N,
                                              d_ws, ws_bytes, cudaStreamPerThread)) ok = false;
                });
                if (!ok) { fprintf(stderr, "EXL3 arm refused\n"); return 2; }
                char arm[32];
                snprintf(arm, sizeof arm, "EXL3 K=%d", s.k2 / 2);
                printf("%-22s %-11s %4d | %9.2f %8.1f %7.1f %6.2f | %.1f\n", sh.what, arm, M, s.stride / 1e6, t,
                       s.stride / (t * 1e-6) / 1e9, t / roof[j], roof[j]);
            }
        }
        printf("%-22s EXL3 rotation replicas %d..%d, MXFP8 %d\n", sh.what, sets.back().n, sets.front().n, n_f8);
        for (auto &s : sets) cudaFree(s.arena);
        for (int r = 0; r < n_f8; r++) { pulsar_gpu_tensor_free(f8d[r]); pulsar_gpu_tensor_free(f8s[r]); }
        cudaFree(d_y); cudaFree(d_ws);
        pulsar_gpu_act_slot_drop(x);
        pulsar_gpu_act_slot_drop(xb);
        pulsar_gpu_tensor_free(x); pulsar_gpu_tensor_free(xb); pulsar_gpu_tensor_free(out);
    }
    return 0;
}

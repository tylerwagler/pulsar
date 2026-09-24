/* Routed-expert decode GEMV microbenchmark (L241 4g-2).
 *
 * Times pulsar_cutlass_expert_ffn_gemv_small -- the decode arm of the routed
 * MoE (gate/up + SwiGLU emitting E4M3, then down) -- at the pair's production
 * shape under expert tensor-parallel: every selected expert at HALF the
 * intermediate width (in 4096, mid 1024, out 4096), 6 slots, one row.  Runs on
 * one GPU with synthetic stacks, so ncu can profile it (it cannot inside a
 * live pair: kernel replay collides with the TP proxy's mapped slab).
 *
 * Prints the average time per call, the effective weight bandwidth, and a
 * checksum of the output: a kernel change that claims bit-exactness must
 * leave the checksum unchanged.
 *
 * Consecutive calls cycle through N_SETS independent expert selections, as
 * consecutive layers do in production: the same 6 experts every call would
 * leave part of their bytes in L2 and time a warm cache no decode step sees.
 * The checksum folds every set's output.  mid defaults to the pair's half
 * width; 2048 is the one-box shape.
 *
 *   tests/expert_gemv_bench [n_tokens=1] [iters=200] [mid=1024]
 */
#include "pulsar_gpu.h"
#include "cuda/pulsar_cuda_mx.cuh"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/* The two symbols the CUTLASS object needs from the rest of the backend. */
void pulsar_gpu_seg_note_device_free(void) {}
int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "expert_gemv_bench: %s: %s\n", what, cudaGetErrorString(err));
    return 0;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)g_rng; }

/* cutlass_mxfp4_expert_layout, restated for the bench's own buffers. */
static void layout(uint64_t k, uint64_t n, uint64_t *data, uint64_t *stride) {
    const uint64_t kp = (k + 127) / 128 * 128, np = (n + 127) / 128 * 128;
    *data = n * k / 2;
    *stride = *data + (np / 32) * kp;
}

static uint8_t *make_stack(int n_total, uint64_t k, uint64_t n, uint64_t *stride_out, uint64_t *data_out) {
    uint64_t data = 0, stride = 0;
    layout(k, n, &data, &stride);
    std::vector<uint8_t> h((size_t)n_total * stride);
    for (int e = 0; e < n_total; e++) {
        uint8_t *p = h.data() + (size_t)e * stride;
        for (uint64_t i = 0; i < data; i++) p[i] = (uint8_t)rnd();                   /* E2M1 pairs */
        for (uint64_t i = data; i < stride; i++) p[i] = (uint8_t)(120 + rnd() % 8);  /* E8M0 near 1 */
    }
    uint8_t *d = NULL;
    CK(cudaMalloc(&d, h.size()));
    CK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice));
    *stride_out = stride;
    *data_out = data;
    return d;
}

static const uint8_t *const *make_table(uint8_t *base, uint64_t stride, int n_total) {
    std::vector<const uint8_t *> t(n_total);
    for (int e = 0; e < n_total; e++) t[e] = base + (size_t)e * stride;
    const uint8_t **d = NULL;
    CK(cudaMalloc(&d, n_total * sizeof(void *)));
    CK(cudaMemcpy(d, t.data(), n_total * sizeof(void *), cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char **argv) {
    const int n_tokens = argc > 1 ? atoi(argv[1]) : 1;
    const int iters = argc > 2 ? atoi(argv[2]) : 200;
    const int mid = argc > 3 ? atoi(argv[3]) : 1024;
    const int n_total = 256, n_expert = 6, in = 4096, out = 4096;
    const int n_slots = n_tokens * n_expert;
    enum { N_SETS = 16 };

    uint64_t gs = 0, gd = 0, us = 0, ud = 0, ds = 0, dd = 0;
    uint8_t *g = make_stack(n_total, in, mid, &gs, &gd);
    uint8_t *u = make_stack(n_total, in, mid, &us, &ud);
    uint8_t *dn = make_stack(n_total, mid, out, &ds, &dd);
    const uint8_t *const *gt = make_table(g, gs, n_total);
    const uint8_t *const *ut = make_table(u, us, n_total);
    const uint8_t *const *dt = make_table(dn, ds, n_total);

    /* selected experts: distinct within a row, rows independent; N_SETS sets */
    std::vector<int32_t> sel((size_t)N_SETS * n_slots);
    std::vector<float> rw(n_slots, 1.0f / n_expert);
    for (int t = 0; t < N_SETS * n_tokens; t++)
        for (int s = 0; s < n_expert; s++) {
            int e;
            bool dup;
            do { e = (int)(rnd() % n_total); dup = false; for (int q = 0; q < s; q++) dup |= sel[t * n_expert + q] == e; } while (dup);
            sel[t * n_expert + s] = e;
        }
    int32_t *dsel = NULL; float *drw = NULL, *dout = NULL;
    CK(cudaMalloc(&dsel, sel.size() * 4)); CK(cudaMemcpy(dsel, sel.data(), sel.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&drw, n_slots * 4)); CK(cudaMemcpy(drw, rw.data(), n_slots * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dout, (size_t)n_slots * out * 4));

    /* the activation: E4M3 values + the pulsar_mx_sfoff-swizzled E8M0 plane */
    const int kbp = pulsar_mx_rup(in / 32, 4);
    const size_t sfb = pulsar_mx_sf_slab_bytes(n_tokens, kbp);
    std::vector<uint8_t> xq((size_t)n_tokens * in), xs(sfb, 127);
    for (auto &b : xq) { uint8_t v = (uint8_t)rnd(); if ((v & 0x7f) == 0x7f) v &= 0xfe; b = v; }   /* no NaN */
    uint8_t *dxq = NULL, *dxs = NULL;
    CK(cudaMalloc(&dxq, xq.size())); CK(cudaMemcpy(dxq, xq.data(), xq.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dxs, xs.size())); CK(cudaMemcpy(dxs, xs.data(), xs.size(), cudaMemcpyHostToDevice));

    auto call = [&](int set) {
        const int rc = pulsar_cutlass_expert_ffn_gemv_small(dout, dsel + (size_t)(set % N_SETS) * n_slots, drw, gt, ut, dt, gs, gd, ds, dd,
                                                            7.0f, n_tokens, n_expert, (unsigned)n_total,
                                                            in, mid, out, dxq, dxs, kbp);
        if (rc) { fprintf(stderr, "expert_gemv_bench: the GEMV refused (rc %d)\n", rc); exit(1); }
    };
    for (int i = 0; i < 2 * N_SETS; i++) call(i);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) call(i);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, a, b));
    const double us_call = 1e3 * ms / iters;

    /* distinct experts' bytes a call must read (dedupe across rows), averaged
     * over the sets the timed calls cycled through */
    double distinct_sum = 0.0;
    for (int i = 0; i < iters; i++) {
        std::vector<int> seen(n_total, 0);
        const int32_t *s = sel.data() + (size_t)(i % N_SETS) * n_slots;
        for (int q = 0; q < n_slots; q++) if (!seen[s[q]]++) distinct_sum += 1.0;
    }
    const double distinct = distinct_sum / iters;
    const double bytes = distinct * (gs + us + ds);
    uint64_t sum = 1469598103934665603ull;
    std::vector<float> h((size_t)n_slots * out);
    for (int set = 0; set < N_SETS; set++) {
        call(set);
        CK(cudaMemcpy(h.data(), dout, h.size() * 4, cudaMemcpyDeviceToHost));
        for (float f : h) { uint32_t w; memcpy(&w, &f, 4); sum = (sum ^ w) * 1099511628211ull; }
    }
    printf("expert_gemv_bench: n_tokens=%d mid=%d slots=%d distinct=%.2f  %.2f us/call  %.1f GB/s  checksum %016llx\n",
           n_tokens, mid, n_slots, distinct, us_call, bytes / (us_call * 1e3), (unsigned long long)sum);
    return 0;
}

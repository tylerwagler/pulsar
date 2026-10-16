/* L298: the block drafter's small ops (src/cuda/pulsar_cuda_blockdraft.h) against a host authority, model-free.
 *
 *   C1  the grouped dynamic conv (2 taps, group 16, blocks of 8 rows, several blocks): bit-exact against the host's
 *       f32 transcription of vLLM's Triton kernel (base + delta per tap, fmaf in tap order, one bf16 rounding); the
 *       block boundary (row 0 of a block reads no earlier row) asserted by a mutation
 *   C2  SwiGLU: bit-exact against the host's f32 silu(g) * u rounded once
 *   C3  the row copy into a column window: byte-exact, the rest of the destination untouched
 *   C4  top-k + log-softmax over a 131072-wide bf16 row (ties planted): the ids exactly (descending, ties to the lower
 *       id), the log-probs within 1e-5 of the host's double logsumexp
 *   C5  refusals: rows not whole blocks, k past 16
 *
 *   ./tests/blockdraft_gate */
#include "pulsar_cuda_blockdraft.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <vector>

static int n_fail = 0;
#define CHECK(c, ...)                                                                                                  \
    do {                                                                                                               \
        if (c) { printf("  [ok] "); printf(__VA_ARGS__); printf("\n"); }                                               \
        else { printf("  [FAIL] "); printf(__VA_ARGS__); printf("\n"); n_fail++; }                                     \
    } while (0)

static uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static float bf2f(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static uint32_t g_lcg = 12345u;
static float rnd() {   /* [-1, 1) */
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return (float)((g_lcg >> 8) & 0xffffu) / 32768.0f - 1.0f;
}
template <class T> static T *dev(const std::vector<T> &h) {
    T *d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T> static std::vector<T> host(const T *d, size_t n) {
    std::vector<T> h(n);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

static void conv_case() {
    const uint32_t W = 4096, G = 16, T = 2, B = 8, R = 32, NG = W / G;
    std::vector<uint16_t> x(R * W), delta(R * T * NG), base(T * W);
    for (auto &v : x) v = f2bf(rnd() * 3.0f);
    for (auto &v : delta) v = f2bf(rnd() * 0.5f);
    for (auto &v : base) v = f2bf(rnd());
    uint16_t *dx = dev(x), *dd = dev(delta), *db = dev(base), *dout = nullptr;
    cudaMalloc(&dout, R * W * 2);
    const int rc = pulsar_blockdraft_conv_launch(dx, W, dd, T * NG, db, T, G, B, R, W, dout, W, 0);
    const std::vector<uint16_t> got = host(dout, R * W);
    size_t bad = 0, bad_nomask = 0;
    for (uint32_t r = 0; r < R; r++)
        for (uint32_t c = 0; c < W; c++) {
            const uint32_t g = c / G, p = r % B;
            float acc = (bf2f(base[c]) + bf2f(delta[r * T * NG + g])) * bf2f(x[r * W + c]);
            float acc2 = acc;
            for (uint32_t t = 1; t < T; t++) {
                const float co = bf2f(base[t * W + c]) + bf2f(delta[r * T * NG + t * NG + g]);
                if (p >= t) acc = fmaf(co, bf2f(x[(r - t) * W + c]), acc);
                if (r >= t) acc2 = fmaf(co, bf2f(x[(r - t) * W + c]), acc2);   /* no block boundary: a mutation */
            }
            bad += got[r * W + c] != f2bf(acc);
            bad_nomask += got[r * W + c] != f2bf(acc2);
        }
    CHECK(rc == 0 && bad == 0, "C1 grouped conv, %u rows in blocks of %u, %u channels, group %u: %zu of %u elements "
          "differ from the host's f32 transcription", R, B, W, G, bad, R * W);
    CHECK(bad_nomask > 0, "C1 mutation: a conv reading across the block boundary differs (%zu elements)", bad_nomask);
    cudaFree(dx); cudaFree(dd); cudaFree(db); cudaFree(dout);
}

static void swiglu_case() {
    const uint32_t W = 12288, R = 8;
    std::vector<uint16_t> g(R * W), u(R * W);
    for (auto &v : g) v = f2bf(rnd() * 6.0f);
    for (auto &v : u) v = f2bf(rnd() * 2.0f);
    uint16_t *dg = dev(g), *du = dev(u), *dh = nullptr;
    cudaMalloc(&dh, R * W * 2);
    const int rc = pulsar_blockdraft_swiglu_launch(dg, W, du, W, R, W, dh, W, 0);
    const std::vector<uint16_t> got = host(dh, R * W);
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); i++) {
        const float a = bf2f(g[i]);
        bad += got[i] != f2bf(a / (1.0f + expf(-a)) * bf2f(u[i]));
    }
    /* expf on the device and the host may differ in the last bit; graded as at most a bf16 ulp */
    size_t far = 0;
    for (size_t i = 0; i < got.size(); i++) {
        const float a = bf2f(g[i]), want = a / (1.0f + expf(-a)) * bf2f(u[i]);
        far += fabsf(bf2f(got[i]) - want) > fabsf(want) * 0.008f + 1e-30f;
    }
    CHECK(rc == 0 && far == 0, "C2 SwiGLU %u x %u: %zu elements off the host's by more than a bf16 ulp (%zu not "
          "bit-identical: the libm exp's last bit)", R, W, far, bad);
    cudaFree(dg); cudaFree(du); cudaFree(dh);
}

static void copy_case() {
    const uint32_t R = 37, W = 4096, LD = 5 * W, COL = 2 * W;
    std::vector<uint16_t> src(R * W), dst(R * LD);
    for (auto &v : src) v = f2bf(rnd());
    for (auto &v : dst) v = 0x1234;
    uint16_t *ds = dev(src), *dd = dev(dst);
    const int rc = pulsar_blockdraft_rows_copy_launch(ds, W, R, W, dd + COL, LD, 0);
    const std::vector<uint16_t> got = host(dd, R * LD);
    size_t bad = 0;
    for (uint32_t r = 0; r < R; r++)
        for (uint32_t c = 0; c < LD; c++) {
            const uint16_t want = c >= COL && c < COL + W ? src[r * W + c - COL] : 0x1234;
            bad += got[r * LD + c] != want;
        }
    CHECK(rc == 0 && bad == 0, "C3 row copy of %u rows into columns [%u, %u) of %u-wide rows: %zu bytes wrong", R, COL,
          COL + W, LD, bad);
    cudaFree(ds); cudaFree(dd);
}

static void topk_case() {
    const uint32_t V = 131072, R = 7, K = 8;
    std::vector<uint16_t> x(R * V);
    for (auto &v : x) v = f2bf(rnd() * 8.0f);
    for (uint32_t r = 0; r < R; r++) {   /* planted ties at the top: equal values at ids in reverse order */
        x[r * V + 90000] = f2bf(20.0f + r);
        x[r * V + 50] = f2bf(20.0f + r);
        x[r * V + 70000] = f2bf(19.5f);
    }
    uint16_t *dx = dev(x);
    int32_t *di = nullptr;
    float *dl = nullptr;
    cudaMalloc(&di, R * K * 4);
    cudaMalloc(&dl, R * K * 4);
    const int rc = pulsar_rows_topk_logprob_launch(dx, V, R, V, K, di, dl, 0);
    const std::vector<int32_t> ids = host(di, R * K);
    const std::vector<float> lp = host(dl, R * K);
    size_t bad_id = 0;
    double worst = 0;
    for (uint32_t r = 0; r < R; r++) {
        std::vector<uint32_t> o(V);
        for (uint32_t i = 0; i < V; i++) o[i] = i;
        const uint16_t *row = &x[r * V];
        std::partial_sort(o.begin(), o.begin() + K, o.end(), [&](uint32_t a, uint32_t b) {
            const float va = bf2f(row[a]), vb = bf2f(row[b]);
            return va > vb || (va == vb && a < b);
        });
        double mx = -1e30, s = 0;
        for (uint32_t i = 0; i < V; i++) mx = std::max(mx, (double)bf2f(row[i]));
        for (uint32_t i = 0; i < V; i++) s += exp((double)bf2f(row[i]) - mx);
        for (uint32_t j = 0; j < K; j++) {
            bad_id += (uint32_t)ids[r * K + j] != o[j];
            worst = std::max(worst, fabs((double)lp[r * K + j] - ((double)bf2f(row[o[j]]) - mx - log(s))));
        }
    }
    CHECK(rc == 0 && bad_id == 0 && ids[0] == 50 && ids[1] == 90000, "C4 top-%u of %u rows x %u: %zu ids differ from "
          "the host's order (ties to the lower id: row 0 starts %d, %d)", K, R, V, bad_id, ids[0], ids[1]);
    CHECK(worst < 1e-5, "C4 the log-probs within 1e-5 of the host's double logsumexp (worst %.3g)", worst);
    cudaFree(dx); cudaFree(di); cudaFree(dl);
}

int main() {
    printf("blockdraft gate (src/cuda/pulsar_cuda_blockdraft.h)\n");
    conv_case();
    swiglu_case();
    copy_case();
    topk_case();
    uint16_t *d = nullptr;
    cudaMalloc(&d, 1 << 20);
    CHECK(pulsar_blockdraft_conv_launch(d, 64, d, 8, d, 2, 16, 8, 12, 64, d + 4096, 64, 0) == -1,
          "C5 refused: rows not whole blocks");
    CHECK(pulsar_rows_topk_logprob_launch(d, 64, 1, 64, 17, (int32_t *)d, (float *)d, 0) == -1, "C5 refused: k 17");
    cudaFree(d);
    printf(n_fail ? "blockdraft_gate: FAIL (%d)\n" : "blockdraft_gate: PASS\n", n_fail);
    return n_fail != 0;
}

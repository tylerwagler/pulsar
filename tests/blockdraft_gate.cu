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
 *   C6  (L299) the stream mean (a hyper-connection trunk's tap: 4 bf16 streams -> their mean) into a column window:
 *       bit-exact against the host's bf16(((s0 + s1 + s2 + s3) * 0.25)), the rest of the destination untouched
 *   C7  (L299) top-k over a head whose rows are a vocabulary subset: the ids are the id map's of the C4 rows
 *   C8  (L299) the CandidateSelector chain head (src/cuda/pulsar_cuda_candsel.h) over 3 banks of 7 slots x 16
 *       candidates, rank 256, ties planted: every score within 1e-5 relative of the host's double transcription of
 *       z-lab's score, the walk (first argmax, slot 0 from the anchor) exactly; a bank's bytes alone == beside others
 *   C9  (L335) top-k over rows whose first 4096 columns are -inf (masked): no NaN log-prob, the C4 grade over the rest
 *
 *   ./tests/blockdraft_gate */
#include "pulsar_cuda_blockdraft.h"
#include "pulsar_cuda_candsel.h"

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
    const int rc = pulsar_rows_topk_logprob_launch(dx, V, R, V, K, nullptr, di, dl, 0);
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
    /* C7: the same rows as a head over a vocabulary subset -- column c is id map[c] */
    std::vector<int32_t> map(V);
    for (uint32_t c = 0; c < V; c++) map[c] = (int32_t)(3u * c + 7u);
    int32_t *dm = dev(map);
    const int rc7 = pulsar_rows_topk_logprob_launch(dx, V, R, V, K, dm, di, dl, 0);
    const std::vector<int32_t> mids = host(di, R * K);
    const std::vector<float> mlp = host(dl, R * K);
    size_t bad7 = 0;
    for (size_t i = 0; i < mids.size(); i++) bad7 += mids[i] != map[(uint32_t)ids[i]] || mlp[i] != lp[i];
    CHECK(rc7 == 0 && bad7 == 0, "C7 top-k through an id map: %zu of %u (id, log-prob) pairs differ from C4's mapped",
          bad7, R * K);
    cudaFree(dm);
    /* C9 (L335): masked logits.  The first 4096 columns of every row are -inf, so every thread's first several
     * strided columns are -inf while its running max still is: the online sum took expf(-inf - -inf) = NaN there,
     * and the whole row's log-probs came back NaN.  They must match the host's logsumexp over the finite columns. */
    for (uint32_t r = 0; r < R; r++)
        for (uint32_t i = 0; i < 4096u; i++) x[r * V + i] = 0xff80u;   /* bf16 -inf */
    cudaMemcpy(dx, x.data(), x.size() * 2u, cudaMemcpyHostToDevice);
    const int rc9 = pulsar_rows_topk_logprob_launch(dx, V, R, V, K, nullptr, di, dl, 0);
    const std::vector<int32_t> ids9 = host(di, R * K);
    const std::vector<float> lp9 = host(dl, R * K);
    size_t bad9 = 0, nan9 = 0;
    double worst9 = 0;
    for (uint32_t r = 0; r < R; r++) {
        const uint16_t *row = &x[r * V];
        std::vector<uint32_t> o(V);
        for (uint32_t i = 0; i < V; i++) o[i] = i;
        std::partial_sort(o.begin(), o.begin() + K, o.end(), [&](uint32_t a, uint32_t b) {
            const float va = bf2f(row[a]), vb = bf2f(row[b]);
            return va > vb || (va == vb && a < b);
        });
        double mx = -1e30, s = 0;
        for (uint32_t i = 0; i < V; i++) mx = std::max(mx, (double)bf2f(row[i]));
        for (uint32_t i = 0; i < V; i++) s += exp((double)bf2f(row[i]) - mx);
        for (uint32_t j = 0; j < K; j++) {
            bad9 += (uint32_t)ids9[r * K + j] != o[j];
            nan9 += lp9[r * K + j] != lp9[r * K + j];
            worst9 = std::max(worst9, fabs((double)lp9[r * K + j] - ((double)bf2f(row[o[j]]) - mx - log(s))));
        }
    }
    CHECK(rc9 == 0 && bad9 == 0 && nan9 == 0 && worst9 < 1e-5,
          "C9 rows whose first 4096 columns are -inf: %zu ids differ, %zu NaN log-probs, worst %.3g", bad9, nan9,
          worst9);
    cudaFree(dx); cudaFree(di); cudaFree(dl);
}

static void stream_mean_case() {
    const uint32_t R = 21, NS = 4, W = 2560, LD = 5 * W, COL = 3 * W;
    std::vector<uint16_t> src(R * NS * W), dst(R * LD);
    for (size_t i = 0; i < src.size(); i++) src[i] = f2bf(rnd() * (i % 7 == 0 ? 40.0f : 1.5f));
    for (auto &v : dst) v = 0x4321;
    uint16_t *ds = dev(src), *dd = dev(dst);
    const int rc = pulsar_blockdraft_stream_mean_launch(ds, NS * W, NS, R, W, dd + COL, LD, 0);
    const std::vector<uint16_t> got = host(dd, R * LD);
    size_t bad = 0;
    for (uint32_t r = 0; r < R; r++)
        for (uint32_t c = 0; c < LD; c++) {
            uint16_t want = 0x4321;
            if (c >= COL && c < COL + W) {
                float a = 0.f;
                for (uint32_t st = 0; st < NS; st++) a += bf2f(src[(r * NS + st) * W + c - COL]);
                want = f2bf(a * (1.0f / NS));
            }
            bad += got[r * LD + c] != want;
        }
    CHECK(rc == 0 && bad == 0, "C6 stream mean of %u rows x %u streams x %u into columns [%u, %u): %zu elements wrong",
          R, NS, W, COL, COL + W, bad);
    cudaFree(ds); cudaFree(dd);
}

/* the host transcription of z-lab's CandidateSelector score and walk (double) */
static void candsel_case() {
    const uint32_t NB = 3, SL = 7, K = 16, RK = 256, V = 5000, HB = 8;   /* hp: 8 rows a bank (the block), slot s at 1 + s */
    std::vector<int32_t> cand(NB * SL * K), anchor(NB);
    std::vector<float> unary(NB * SL * K);
    std::vector<uint16_t> hp(NB * HB * RK), pred((size_t)V * RK), succ((size_t)V * RK);
    for (auto &v : pred) v = f2bf(rnd());
    for (auto &v : succ) v = f2bf(rnd());
    for (auto &v : hp) v = f2bf(rnd() * 0.5f);
    for (uint32_t i = 0; i < NB * SL * K; i++) {
        g_lcg = g_lcg * 1664525u + 1013904223u;
        cand[i] = (int32_t)((g_lcg >> 8) % V);
        unary[i] = rnd() * 3.0f - (float)(i % K) * 0.2f;
    }
    for (uint32_t b = 0; b < NB; b++) anchor[b] = (int32_t)(17u + 101u * b);
    /* a planted tie: bank 1 slot 0's candidates 3 and 5 are the same token at the same unary -> equal scores; the
     * walk takes the lower index if they lead */
    cand[(1 * SL + 0) * K + 5] = cand[(1 * SL + 0) * K + 3];
    unary[(1 * SL + 0) * K + 5] = unary[(1 * SL + 0) * K + 3] = 50.0f;
    int32_t *dc = dev(cand), *da = dev(anchor), *dids = nullptr, *drows = nullptr;
    float *du = dev(unary), *dsc = nullptr;
    uint16_t *dh = dev(hp), *dp = dev(pred), *ds = dev(succ);
    cudaMalloc(&dsc, NB * SL * K * K * 4);
    cudaMalloc(&dids, NB * SL * 4);
    cudaMalloc(&drows, NB * SL * 4);
    const int rc = pulsar_candsel_launch(NB, SL, K, RK, V, dc, du, dh + RK, HB, RK, da, dp, ds, dsc, dids, drows, 0);
    const std::vector<float> sc = host(dsc, NB * SL * K * K);
    const std::vector<int32_t> ids = host(dids, NB * SL), rows = host(drows, NB * SL);
    double worst = 0;
    size_t bad_walk = 0;
    for (uint32_t b = 0; b < NB; b++) {
        uint32_t prev = 0;
        for (uint32_t s = 0; s < SL; s++) {
            const uint16_t *h = &hp[(b * HB + 1 + s) * RK];
            std::vector<double> rowv(K * K);
            for (uint32_t i = 0; i < K; i++)
                for (uint32_t c = 0; c < K; c++) {
                    const int32_t p = s == 0 ? anchor[b] : cand[(b * SL + s - 1) * K + i];
                    const int32_t t = cand[(b * SL + s) * K + c];
                    double acc = 0;
                    for (uint32_t r = 0; r < RK; r++)
                        acc += (double)bf2f(succ[(size_t)t * RK + r]) * bf2f(pred[(size_t)p * RK + r]) * bf2f(h[r]);
                    const double want = unary[(b * SL + s) * K + c] + acc;
                    rowv[i * K + c] = want;
                    const double g = sc[((b * SL + s) * K + i) * K + c];
                    worst = std::max(worst, fabs(g - want) / std::max(1.0, fabs(want)));
                }
            /* the walk on the DEVICE's scores (first maximum), against the device's choice */
            const float *dr = &sc[((b * SL + s) * K + prev) * K];
            uint32_t j = 0;
            for (uint32_t c = 1; c < K; c++)
                if (dr[c] > dr[j]) j = c;
            bad_walk += (uint32_t)rows[b * SL + s] != j || ids[b * SL + s] != cand[(b * SL + s) * K + j];
            prev = j;
        }
    }
    CHECK(rc == 0 && worst < 1e-5, "C8 selector scores, %u banks x %u slots x %u x %u, rank %u: worst relative "
          "distance %.3g from the host's double", NB, SL, K, K, RK, worst);
    CHECK(bad_walk == 0 && rows[1 * SL + 0] == 3, "C8 the walk: %zu slot choices differ from the first argmax of the "
          "row the previous slot chose (bank 1's planted tie takes candidate %d, want 3)", bad_walk, rows[1 * SL + 0]);
    /* bank 2 alone == bank 2 beside the others */
    const int rc1 = pulsar_candsel_launch(1, SL, K, RK, V, dc + 2 * SL * K, du + 2 * SL * K, dh + 2 * HB * RK + RK, HB,
                                          RK, da + 2, dp, ds, dsc, dids, drows, 0);
    const std::vector<float> one = host(dsc, SL * K * K);
    CHECK(rc1 == 0 && memcmp(one.data(), &sc[2 * SL * K * K], one.size() * 4) == 0,
          "C8 a bank's scores alone == the same bank beside two others, byte for byte");
    CHECK(pulsar_candsel_launch(1, SL, 17, RK, V, dc, du, dh, HB, RK, da, dp, ds, dsc, dids, drows, 0) == -1,
          "C8 refused: k 17");
    cudaFree(dc); cudaFree(da); cudaFree(dids); cudaFree(drows); cudaFree(du); cudaFree(dsc); cudaFree(dh);
    cudaFree(dp); cudaFree(ds);
}

int main() {
    printf("blockdraft gate (src/cuda/pulsar_cuda_blockdraft.h)\n");
    conv_case();
    swiglu_case();
    copy_case();
    topk_case();
    stream_mean_case();
    candsel_case();
    uint16_t *d = nullptr;
    cudaMalloc(&d, 1 << 20);
    CHECK(pulsar_blockdraft_conv_launch(d, 64, d, 8, d, 2, 16, 8, 12, 64, d + 4096, 64, 0) == -1,
          "C5 refused: rows not whole blocks");
    CHECK(pulsar_rows_topk_logprob_launch(d, 64, 1, 64, 17, nullptr, (int32_t *)d, (float *)d, 0) == -1,
          "C5 refused: k 17");
    cudaFree(d);
    printf(n_fail ? "blockdraft_gate: FAIL (%d)\n" : "blockdraft_gate: PASS\n", n_fail);
    return n_fail != 0;
}

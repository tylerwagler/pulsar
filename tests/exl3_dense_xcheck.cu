/* EXL3 dense-Linear cross-check tool (L251): runs the dense arm (the
 * production object) on a REAL weight that exllamav3's quantize_exl3 produced,
 * with activation rows handed over as the A8 slot's bytes, so a driver script
 * can compare it with exllamav3's own forward on the same activations
 * (pulsar-notes research/l251/exl3-dense/xcheck.py).  Not a gate: it needs
 * the files the driver writes.
 *
 *   tests/exl3_dense_xcheck DIR K N k2 M
 *
 *   DIR/w.bin     the [trellis | suh | svh] slice: exllamav3's `trellis` int16
 *                 (K/16, N/16, 16 K) tensor bytes verbatim, then `suh`, `svh`
 *   DIR/xq.bin    M x K E4M3 bytes; DIR/xs.bin  M x K/32 ue8m0 bytes, row-major
 *   writes DIR/y_pulsar.bin     M x N f32, the arm at width M
 *          DIR/y_pulsar_m1.bin  M x N f32, each row alone (M = 1)
 *          DIR/y_ref.bin        M x N f64, the host authority (tests/exl3_dense_ref.h)
 */
#include "../src/cuda/mmq/ds4_exl3_dense.cuh"
#include "../src/cuda/pulsar_cuda_mx.cuh"
#include "exl3_dense_ref.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static std::vector<uint8_t> slurp(const std::string &path, size_t want) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
    std::vector<uint8_t> b(want);
    const size_t got = fread(b.data(), 1, want, f);
    const int extra = fgetc(f);
    fclose(f);
    if (got != want || extra != EOF) { fprintf(stderr, "%s: expected exactly %zu bytes\n", path.c_str(), want); exit(2); }
    return b;
}
static void spill(const std::string &path, const void *p, size_t n) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f || fwrite(p, 1, n, f) != n) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(2); }
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "usage: %s DIR K N k2 M\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    const int K = atoi(argv[2]), N = atoi(argv[3]), k2 = atoi(argv[4]), M = atoi(argv[5]);
    uint64_t trellis = 0, stride = 0;
    exl3t_layout(K, N, k2, &trellis, &stride);
    const std::vector<uint8_t> w = slurp(dir + "/w.bin", stride);
    const std::vector<uint8_t> xq = slurp(dir + "/xq.bin", (size_t)M * K);
    const std::vector<uint8_t> xs = slurp(dir + "/xs.bin", (size_t)M * K / 32);

    /* the slot: E4M3 rows as given, the scale bytes into the swizzled slab */
    const int kbp = pulsar_mx_kbp(K);
    std::vector<uint8_t> sf(pulsar_mx_sf_slab_bytes(M, kbp), 0);
    std::vector<double> x((size_t)M * K);
    for (int r = 0; r < M; r++)
        for (int g = 0; g < K / 32; g++) {
            const uint8_t b = xs[(size_t)r * (K / 32) + g];
            sf[pulsar_mx_sfoff(r, g, kbp)] = b;
            for (int j = 0; j < 32; j++)
                x[(size_t)r * K + g * 32 + j] = exl3t_e4m3_to_f64(xq[(size_t)r * K + g * 32 + j]) * ldexp(1.0, (int)b - 127);
        }

    uint8_t *d_w = nullptr, *d_q = nullptr, *d_sf = nullptr;
    float *d_y = nullptr, *d_ws = nullptr;
    const size_t ws_bytes = ds4_exl3_dense_workspace_bytes(M, K, N);
    CK(cudaMalloc((void **)&d_w, stride));
    CK(cudaMalloc((void **)&d_q, xq.size()));
    CK(cudaMalloc((void **)&d_sf, sf.size()));
    CK(cudaMalloc((void **)&d_y, (size_t)M * N * 4));
    CK(cudaMalloc((void **)&d_ws, ws_bytes));
    CK(cudaMemcpy(d_w, w.data(), stride, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_q, xq.data(), xq.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sf, sf.data(), sf.size(), cudaMemcpyHostToDevice));

    std::vector<float> y((size_t)M * N), y1((size_t)M * N);
    if (ds4_exl3_dense_launch(d_w, k2, d_q, d_sf, d_y, M, K, N, d_ws, ws_bytes, 0)) return 1;
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost));
    /* each row alone: its own one-row slot (the row's bytes, scale row 0) */
    for (int r = 0; r < M; r++) {
        std::vector<uint8_t> sf1(pulsar_mx_sf_slab_bytes(1, kbp), 0);
        for (int g = 0; g < K / 32; g++) sf1[pulsar_mx_sfoff(0, g, kbp)] = xs[(size_t)r * (K / 32) + g];
        CK(cudaMemcpy(d_q, xq.data() + (size_t)r * K, K, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sf, sf1.data(), sf1.size(), cudaMemcpyHostToDevice));
        if (ds4_exl3_dense_launch(d_w, k2, d_q, d_sf, d_y, 1, K, N, d_ws, ws_bytes, 0)) return 1;
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(y1.data() + (size_t)r * N, d_y, (size_t)N * 4, cudaMemcpyDeviceToHost));
    }

    std::vector<double> what, yref;
    exl3t_dequant(w.data(), K, N, k2, what);
    exl3t_reference(w.data(), what, K, N, k2, x.data(), M, yref);
    double mx = 0, num = 0, den = 0;
    for (int r = 0; r < M; r++) {
        double sc = 0;
        for (int n = 0; n < N; n++) sc = fmax(sc, fabs(yref[(size_t)r * N + n]));
        for (int n = 0; n < N; n++) {
            const double d = (double)y[(size_t)r * N + n] - yref[(size_t)r * N + n];
            mx = fmax(mx, fabs(d) / sc);
            num += d * d; den += yref[(size_t)r * N + n] * yref[(size_t)r * N + n];
        }
    }
    const bool m1_same = memcmp(y.data(), y1.data(), y.size() * 4) == 0;
    printf("exl3_dense_xcheck: %d -> %d K=%g M=%d splits %d: vs host authority max rel %.3e, rel Frobenius %.3e; "
           "rows alone (M=1) %s\n", K, N, k2 / 2.0, M, ds4_exl3_dense_splits(K, N), mx, sqrt(num / den),
           m1_same ? "bit-identical to the M-row run" : "DIFFER from the M-row run");
    spill(dir + "/y_pulsar.bin", y.data(), y.size() * 4);
    spill(dir + "/y_pulsar_m1.bin", y1.data(), y1.size() * 4);
    spill(dir + "/y_ref.bin", yref.data(), yref.size() * 8);
    return (mx < 2e-5 && m1_same) ? 0 : 1;
}

/* The EXL3 dense-Linear host reference (L251), shared by tests/exl3_dense_gate.cu
 * and tests/exl3_dense_xcheck.cu: built only from src/engine/exl3_trellis.h
 * (the host dequant that tests/exl3_dequant_gate.cpp holds byte-exact to
 * exllamav3) and double arithmetic,
 *
 *   y = svh * H128( W_hat^T H128(suh * x) )     per row, per 128-block.
 */
#pragma once

#include "../src/engine/exl3_trellis.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

/** OCP E4M3 (1-4-3, bias 7, subnormals at exp 0) -> double; 0x7f/0xff (NaN) are
 *  never produced by the fixtures. */
static inline double exl3t_e4m3_to_f64(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const double v = e == 0 ? (double)m / 8.0 * ldexp(1.0, -6) : (1.0 + (double)m / 8.0) * ldexp(1.0, e - 7);
    return s ? -v : v;
}

/** The slice's layout, or exit: a fixture with a bad shape is a test bug. */
static inline void exl3t_layout(int K, int N, int k2, uint64_t *trellis, uint64_t *stride) {
    uint64_t scales = 0;
    if (!exl3_expert_layout((uint64_t)K, (uint64_t)N, k2, trellis, &scales, stride)) {
        fprintf(stderr, "exl3t: K=%d N=%d k2=%d is not an EXL3 layout\n", K, N, k2);
        exit(2);
    }
}

/** W_hat (K, N) row-major, double, from a [trellis | suh | svh] slice. */
static inline void exl3t_dequant(const uint8_t *slice, int K, int N, int k2, std::vector<double> &out) {
    const int words = exl3_words_per_tile(k2), ntn = N / 16;
    out.assign((size_t)K * N, 0.0);
    uint16_t tile[256];
    const uint16_t *base = (const uint16_t *)slice;
    for (int kt = 0; kt < K / 16; kt++)
        for (int nt = 0; nt < ntn; nt++) {
            exl3_tile_dequant(base + ((size_t)kt * ntn + nt) * words, k2, tile);
            for (int r = 0; r < 16; r++)
                for (int c = 0; c < 16; c++)
                    out[(size_t)(kt * 16 + r) * N + nt * 16 + c] = exl3_f16_to_f32(tile[r * 16 + c]);
        }
}

/** y [rows][N] for activation rows x [rows][K] (the slot's decoded values). */
static inline void exl3t_reference(const uint8_t *slice, const std::vector<double> &what, int K, int N, int k2,
                                   const double *x, int rows, std::vector<double> &y) {
    uint64_t trellis = 0, stride = 0;
    exl3t_layout(K, N, k2, &trellis, &stride);
    const uint16_t *suh = (const uint16_t *)(slice + trellis), *svh = suh + K;
    y.assign((size_t)rows * N, 0.0);
    std::vector<double> xr(K), z(N);
    for (int r = 0; r < rows; r++) {
        for (int k = 0; k < K; k++) xr[k] = x[(size_t)r * K + k] * exl3_f16_to_f32(suh[k]);
        for (int i = 0; i + 128 <= K; i += 128) exl3_had128(xr.data() + i);
        std::fill(z.begin(), z.end(), 0.0);
        for (int k = 0; k < K; k++) {
            const double xk = xr[k];
            const double *wr = what.data() + (size_t)k * N;
            for (int n = 0; n < N; n++) z[n] += wr[n] * xk;
        }
        for (int i = 0; i + 128 <= N; i += 128) exl3_had128(z.data() + i);
        for (int n = 0; n < N; n++) y[(size_t)r * N + n] = z[n] * exl3_f16_to_f32(svh[n]);
    }
}

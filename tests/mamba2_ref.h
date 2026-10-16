/* Host reference for the Mamba-2 mixer block (L296 N2, src/cuda/pulsar_cuda_mamba2.h): one sequence at a time
 * from a carried state, in DOUBLE, with the block's (vLLM's) bf16 rounding points mirrored -- the conv output,
 * y, and the normalized value before the weight -- so the grade measures the f32 arithmetic, not the declared
 * roundings.  `mut` selects a deliberately wrong block so the gate can prove its grade sees each feature.
 *
 * Inputs per row are bf16-exact floats: z [H*P], xbc [CC], dt [H].  State: conv [(K-1)][CC] (the last K-1
 * inputs, oldest first), ssm [H][P][N]; both advanced. */
#pragma once

#include "sconv_ref.h"

#include <math.h>
#include <stdint.h>
#include <vector>

struct m2ref_layer {
    int H = 0, P = 0, N = 0, G = 0, K = 0, NG = 0;   /* NG: the norm group */
    double eps = 1e-5;
    std::vector<float> conv_w, conv_b;               /* [CC][K], [CC] */
    std::vector<float> A_log, dt_bias, D, norm_w;    /* [H] x 3, [H*P] */
    int CC() const { return H * P + 2 * G * N; }
};

enum m2ref_mut {
    M2_EXACT = 0,
    M2_NO_D,          /* the D skip dropped */
    M2_NO_GATE,       /* the norm's input is y, not y * SiLU(z) */
    M2_GROUP_HALF,    /* the norm groups are half the declared size */
    M2_GROUP_WHOLE,   /* one norm group over all H*P channels */
    M2_GROUP_MAP,     /* head h reads B / C group h % G instead of h / (H / G) */
    M2_NO_CONV_BIAS,  /* the conv's bias dropped */
};

static inline double m2_softplus(double v) { return v > 20.0 ? v : log1p(exp(v)); }

/* T rows: z [T][ldz], xbc [T][ldx], dt [T][ldd] -> out [T][H*P] (double, the weight applied, unrounded) */
static inline void m2ref_seq(const m2ref_layer &L, int T, const float *z, size_t ldz, const float *xbc, size_t ldx,
                             const float *dt_raw, size_t ldd, double *conv, double *ssm, double *out,
                             m2ref_mut mut = M2_EXACT) {
    const int H = L.H, P = L.P, N = L.N, G = L.G, K = L.K, CC = L.CC(), HP = H * P;
    const int NG = mut == M2_GROUP_HALF ? L.NG / 2 : mut == M2_GROUP_WHOLE ? HP : L.NG;
    std::vector<double> xc((size_t)CC), y((size_t)HP), v((size_t)HP);
    for (int t = 0; t < T; ++t) {
        const float *xr = xbc + (size_t)t * ldx;
        for (int c = 0; c < CC; ++c) {
            double acc = mut == M2_NO_CONV_BIAS ? 0.0 : (double)L.conv_b[(size_t)c];
            for (int i = 0; i < K - 1; ++i) acc += (double)L.conv_w[(size_t)c * K + i] * conv[(size_t)i * CC + c];
            acc += (double)L.conv_w[(size_t)c * K + K - 1] * (double)xr[c];
            acc = acc / (1.0 + exp(-acc));
            xc[(size_t)c] = (double)sref_round((float)acc);
        }
        for (int c = 0; c < CC; ++c) {
            for (int i = 0; i + 1 < K - 1; ++i) conv[(size_t)i * CC + c] = conv[(size_t)(i + 1) * CC + c];
            conv[(size_t)(K - 2) * CC + c] = (double)xr[c];
        }
        for (int h = 0; h < H; ++h) {
            const int g = mut == M2_GROUP_MAP ? h % G : h / (H / G);
            const double dt = m2_softplus((double)dt_raw[(size_t)t * ldd + h] + (double)L.dt_bias[(size_t)h]);
            const double dA = exp(dt * -exp((double)L.A_log[(size_t)h]));
            const double *B = &xc[(size_t)HP + (size_t)g * N], *C = &xc[(size_t)HP + (size_t)G * N + (size_t)g * N];
            for (int p = 0; p < P; ++p) {
                const double x = xc[(size_t)h * P + p];
                double *S = ssm + ((size_t)h * P + p) * N;
                double acc = 0.0;
                for (int n = 0; n < N; ++n) {
                    S[n] = dA * S[n] + dt * B[n] * x;
                    acc += S[n] * C[n];
                }
                if (mut != M2_NO_D) acc += (double)L.D[(size_t)h] * x;
                y[(size_t)h * P + p] = (double)sref_round((float)acc);
            }
        }
        const float *zr = z + (size_t)t * ldz;
        for (int c = 0; c < HP; ++c) {
            const double zz = (double)zr[c];
            v[(size_t)c] = mut == M2_NO_GATE ? y[(size_t)c] : y[(size_t)c] * (zz / (1.0 + exp(-zz)));
        }
        for (int g0 = 0; g0 < HP; g0 += NG) {
            double ss = 0.0;
            for (int c = g0; c < g0 + NG; ++c) ss += v[(size_t)c] * v[(size_t)c];
            const double r = 1.0 / sqrt(ss / NG + L.eps);
            for (int c = g0; c < g0 + NG; ++c)
                out[(size_t)t * HP + c] = (double)L.norm_w[(size_t)c] * (double)sref_round((float)(v[(size_t)c] * r));
        }
    }
}

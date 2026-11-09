/* Host authorities for the short convolution block (L295 E3,
 * src/cuda/pulsar_cuda_sconv.h), one sequence at a time from a carried state:
 *
 *   sref_seq_double  the convolution in DOUBLE (the fidelity reference), with
 *                    `mut` selecting a deliberately wrong one so the gate can
 *                    prove its grade sees each feature;
 *   sref_seq_bits    the block's own arithmetic replayed on the host -- the f32
 *                    fma chain from the bias oldest tap first, the f32 residual
 *                    add, one bf16 round-to-nearest-even -- which the kernel
 *                    must equal BYTE for byte (std::fmaf and the device's
 *                    __fmaf_rn are both the correctly rounded fused
 *                    multiply-add).  Not for a SiLU declaration: the device's
 *                    fast-math exp has no host replay, so a SiLU stream is
 *                    graded against the double and held to its own schedules.
 *
 * State: [K-1][C] values, row K-2 the newest input (the slot layout); both
 * functions advance it.  Inputs are bf16-exact floats. */
#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <vector>

static inline uint16_t sref_f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static inline float sref_bf2f(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static inline float sref_round(float f) { return sref_bf2f(sref_f2bf(f)); }

struct sref_conv {
    int C = 0, K = 0;
    bool residual = true;
    bool silu = false;
    std::vector<float> w;   /* [C][K], bf16-exact */
    std::vector<float> b;   /* empty, or [C] bf16-exact */
};

enum sref_mut {
    SREF_EXACT = 0,
    SREF_REVERSED,      /* taps applied newest-first (w[0] on the current row) */
    SREF_NO_RESIDUAL,   /* the declared residual dropped */
    SREF_LAGGED,        /* the window ends one row early: the current row is not a tap */
    SREF_NO_BIAS,       /* the declared bias dropped */
    SREF_NO_SILU,       /* the declared SiLU dropped */
};

/* x [T][ld] -> y [T][C]; state [(K-1) * C] carried. */
static inline void sref_seq_double(const sref_conv &s, int T, const float *x, size_t ld, double *state, double *y,
                            sref_mut mut = SREF_EXACT) {
    const int C = s.C, K = s.K;
    std::vector<double> win((size_t)(K + 1));
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < C; ++c) {
            /* win[0 .. K]: inputs at t-K .. t (win[0] only for the lagged mutation) */
            win[0] = 0.0;   /* t-K is never part of the exact window; the lagged one reads a dropped row as 0 */
            for (int i = 0; i < K - 1; ++i) win[1 + i] = state[(size_t)i * C + c];
            win[K] = x[(size_t)t * ld + c];
            double acc = s.b.empty() || mut == SREF_NO_BIAS ? 0.0 : (double)s.b[(size_t)c];
            for (int i = 0; i < K; ++i) {
                const double wi = s.w[(size_t)c * K + (mut == SREF_REVERSED ? K - 1 - i : i)];
                acc += wi * (mut == SREF_LAGGED ? win[i] : win[1 + i]);
            }
            const bool res = mut == SREF_NO_RESIDUAL ? !s.residual : s.residual;
            if (s.silu && mut != SREF_NO_SILU) acc = acc / (1.0 + exp(-acc));
            y[(size_t)t * C + c] = res ? acc + win[K] : acc;
        }
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i + 1 < K - 1; ++i) state[(size_t)i * C + c] = state[(size_t)(i + 1) * C + c];
            state[(size_t)(K - 2) * C + c] = x[(size_t)t * ld + c];
        }
    }
}

/* the block's bytes: x [T][ld] -> y [T][C] bf16; state [(K-1) * C] floats carried. */
static inline void sref_seq_bits(const sref_conv &s, int T, const float *x, size_t ld, float *state, uint16_t *y) {
    const int C = s.C, K = s.K;
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < C; ++c) {
            const float xv = x[(size_t)t * ld + c];
            float acc = s.b.empty() ? 0.0f : s.b[(size_t)c];
            for (int i = 0; i < K - 1; ++i) acc = fmaf(s.w[(size_t)c * K + i], state[(size_t)i * C + c], acc);
            acc = fmaf(s.w[(size_t)c * K + K - 1], xv, acc);
            y[(size_t)t * C + c] = sref_f2bf(s.residual ? acc + xv : acc);
        }
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i + 1 < K - 1; ++i) state[(size_t)i * C + c] = state[(size_t)(i + 1) * C + c];
            state[(size_t)(K - 2) * C + c] = x[(size_t)t * ld + c];
        }
    }
}

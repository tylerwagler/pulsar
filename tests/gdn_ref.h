/* Host authority for the Gated DeltaNet kernels (L251): transformers'
 * Qwen4ExpTextGatedDeltaNet between its projections, in DOUBLE, token by token
 * in the reference's own order (torch_recurrent_gated_delta_rule):
 *
 *   S *= decay;  kv = S^T k;  delta = (v - kv) * beta;  S += k delta^T;  o = S^T q
 *
 * One sequence per call; both states are carried in double by the caller.
 * `mut` selects a deliberately wrong reference, so the gate can prove it
 * discriminates the property (see tests/gdn_gate.cu). */
#pragma once

#include "cuda/pulsar_cuda_gdn.h"

#include <math.h>
#include <stddef.h>

enum gdn_ref_mut {
    GDN_REF_EXACT = 0,
    GDN_REF_GQA_TILED,   /* V head h reads K head h % 16 (torch .repeat, not repeat_interleave) */
    GDN_REF_NO_QSCALE,   /* q not scaled by 128^-0.5 */
};

struct gdn_ref_weights {
    const float *conv_w;   /* [10240][4] */
    const float *A_log;    /* [48] */
    const float *dt_bias;  /* [48] */
    const float *norm_w;   /* [128] */
};

/* conv [3][10240], rec [48][128][128] in/out; y [T][6144] out. */
static void gdn_ref_seq(const gdn_ref_weights &w, int T,
                        const float *qkv, int ld_qkv, const float *z, int ld_z,
                        const float *a, int ld_a, const float *b, int ld_b,
                        double *conv, double *rec, double *y, gdn_ref_mut mut = GDN_REF_EXACT) {
    const int QKV = PULSAR_GDN_QKV_DIM, NK = PULSAR_GDN_NK, NV = PULSAR_GDN_NV, D = PULSAR_GDN_DK;
    double *x = new double[QKV];
    double kv[128], dl[128], o[128];
    for (int t = 0; t < T; t++) {
        const float *in = qkv + (size_t)t * ld_qkv;
        for (int c = 0; c < QKV; c++) {
            const double acc = (double)w.conv_w[c * 4 + 0] * conv[0 * QKV + c]
                             + (double)w.conv_w[c * 4 + 1] * conv[1 * QKV + c]
                             + (double)w.conv_w[c * 4 + 2] * conv[2 * QKV + c]
                             + (double)w.conv_w[c * 4 + 3] * (double)in[c];
            x[c] = acc / (1.0 + exp(-acc));
            conv[0 * QKV + c] = conv[1 * QKV + c];
            conv[1 * QKV + c] = conv[2 * QKV + c];
            conv[2 * QKV + c] = in[c];
        }
        for (int hk = 0; hk < 2 * NK; hk++) {           /* q heads then k heads */
            double ss = 0;
            for (int i = 0; i < D; i++) ss += x[hk * D + i] * x[hk * D + i];
            double inv = 1.0 / sqrt(ss + 1e-6);
            if (hk < NK && mut != GDN_REF_NO_QSCALE) inv /= sqrt((double)D);
            for (int i = 0; i < D; i++) x[hk * D + i] *= inv;
        }
        for (int h = 0; h < NV; h++) {
            const int kh = mut == GDN_REF_GQA_TILED ? h % NK : h / (NV / NK);
            const double *q = x + kh * D, *k = x + NK * D + kh * D, *v = x + 2 * NK * D + h * D;
            const double sp_in = (double)a[(size_t)t * ld_a + h] + (double)w.dt_bias[h];
            const double sp = sp_in > 20.0 ? sp_in : log1p(exp(sp_in));
            const double decay = exp(-exp((double)w.A_log[h]) * sp);
            const double beta = 1.0 / (1.0 + exp(-(double)b[(size_t)t * ld_b + h]));
            double *S = rec + (size_t)h * D * D;          /* [k][v] */
            for (int i = 0; i < D * D; i++) S[i] *= decay;
            for (int j = 0; j < D; j++) kv[j] = 0;
            for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) kv[j] += S[i * D + j] * k[i];
            for (int j = 0; j < D; j++) dl[j] = (v[j] - kv[j]) * beta;
            for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) S[i * D + j] += k[i] * dl[j];
            for (int j = 0; j < D; j++) o[j] = 0;
            for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) o[j] += S[i * D + j] * q[i];
            double ss = 0;
            for (int j = 0; j < D; j++) ss += o[j] * o[j];
            const double inv = 1.0 / sqrt(ss / D + 1e-6);
            for (int j = 0; j < D; j++) {
                const double zz = (double)z[(size_t)t * ld_z + h * D + j];
                y[(size_t)t * PULSAR_GDN_V_DIM + h * D + j] = (double)w.norm_w[j] * (o[j] * inv) / (1.0 + exp(-zz));
            }
        }
    }
    delete[] x;
}

#include "pulsar_engine_internal.h"


















/* =========================================================================
 * Hyper-Connection Transforms.
 * =========================================================================
 *
 * DeepSeek V4 Flash keeps four hyper-connection streams per token.  Before
 * attention or FFN, a learned small projection chooses how to reduce the HC
 * state into the 4096-wide sublayer input.  After the sublayer, the post and
 * combine weights expand the result back into the four-stream HC state.
 */

/* Decode the HC control projection.  The output contains pre weights, post
 * gates, and a small doubly-normalized combine matrix. */
void hc_split_sinkhorn_one(
        float       * out,
        const float * mix,
        const float * scale,
        const float * base,
        int           n_hc,
        int           iters,
        float         eps) {
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    for (int i = 0; i < n_hc; i++) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }

    for (int i = 0; i < n_hc; i++) {
        const int off = n_hc + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    float c[16 * 16];

    for (int dst = 0; dst < n_hc; dst++) {
        float row_max = PULSAR_NEG_INF;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            const int off = 2 * n_hc + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if (v > row_max) row_max = v;
        }

        float row_sum = 0.0f;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv = 1.0f / row_sum;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }

    for (int src = 0; src < n_hc; src++) {
        float sum = 0.0f;
        for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];

        const float inv = 1.0f / (sum + eps);
        for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
    }

    for (int iter = 1; iter < iters; iter++) {
        for (int dst = 0; dst < n_hc; dst++) {
            float sum = 0.0f;
            for (int src = 0; src < n_hc; src++) sum += c[src + dst * n_hc];

            const float inv = 1.0f / (sum + eps);
            for (int src = 0; src < n_hc; src++) c[src + dst * n_hc] *= inv;
        }

        for (int src = 0; src < n_hc; src++) {
            float sum = 0.0f;
            for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];

            const float inv = 1.0f / (sum + eps);
            for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
        }
    }

    for (int i = 0; i < n_hc * n_hc; i++) out[2 * n_hc + i] = c[i];
}


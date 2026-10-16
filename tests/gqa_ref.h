/* Host authority for the GQA attention block (src/cuda/pulsar_cuda_attn_gqa.h):
 * the declared arithmetic, with the attention itself in double.
 *
 *   q = bf16(bf16(f32(x * rstd) * w_q) * tau)        rstd = f32(1 / sqrt(mean(x^2) + eps))
 *   k = bf16(f32(x * rstd) * w_k),  v = v_in
 *   bias(d) = bf16(bf16(r . P[:, d]) * tau)          0 <= d < extent, else 0
 *   s_x = (q . k_x) * scale + bias(pos - x),  y = softmax(s) V  [* sigmoid(gate)]
 *
 * tau is the module's own f32 formula (gqa_tau): the reference vLLM / transformers
 * compute it in torch f32, which the GPU does not see -- both read this value. */
#ifndef PULSAR_TESTS_GQA_REF_H
#define PULSAR_TESTS_GQA_REF_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <vector>

static inline uint16_t gref_f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static inline float gref_bf2f(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static inline float gref_round(float f) { return gref_bf2f(gref_f2bf(f)); }

struct gref_layer {
    uint32_t n_q, n_kv, hd;
    uint32_t window;            /* 0 = full */
    float scale;
    bool qk_norm;
    float eps;
    std::vector<float> qw, kw;  /* bf16 values */
    uint32_t extent;            /* 0 = no bias */
    std::vector<float> P;       /* [16][extent], bf16 values */
    float floor_, alpha;        /* floor_ 0 = no temperature */
    bool gate;
};

static inline float gref_tau(const gref_layer &L, uint32_t pos) {
    if (L.floor_ <= 0.f) return 1.0f;
    volatile float ratio = (float)(pos + 1u) / L.floor_;
    const float r = ratio < 1.0f ? 1.0f : (float)ratio;
    volatile float prod = L.alpha * (float)log((double)r);
    return 1.0f + prod;
}

/* one head vector x (bf16 values) -> its stored bf16 value (norm with weight w, or as is), as floats */
static inline void gref_norm(const gref_layer &L, const float *x, const std::vector<float> &w, float *out) {
    if (!L.qk_norm) { for (uint32_t i = 0; i < L.hd; i++) out[i] = x[i]; return; }
    double ss = 0;
    for (uint32_t i = 0; i < L.hd; i++) ss += (double)x[i] * x[i];
    const float rstd = (float)(1.0 / sqrt(ss / L.hd + (double)L.eps));
    for (uint32_t i = 0; i < L.hd; i++) {
        volatile float t = x[i] * rstd;
        out[i] = gref_round(t * w[i]);
    }
}

/* The attention of one row.  kh / vh: the sequence's stored K / V for every position
 * <= pos, [pos][n_kv][hd] (the k already normed); qin, r, gate: the row's raw inputs
 * (n_q x hd, n_q x 16, n_q x hd).  out: n_q x hd doubles.  Mutations for the gate's
 * discrimination checks: drop_bias, drop_tau, window_delta. */
struct gref_mut { bool drop_bias = false, drop_tau = false; int window_delta = 0; };

static inline void gref_row(const gref_layer &L, uint32_t pos, const float *kh, const float *vh,
                            const float *qin, const float *r, const float *gate, double *out,
                            const gref_mut &mu = gref_mut()) {
    const uint32_t G = L.n_q / L.n_kv, hd = L.hd;
    const float tau = mu.drop_tau ? 1.0f : gref_tau(L, pos);
    const uint32_t W = L.window ? (uint32_t)((int)L.window + mu.window_delta) : 0u;
    const uint32_t lo = W && pos + 1u > W ? pos + 1u - W : 0u;
    std::vector<float> q(hd);
    std::vector<double> s(pos + 1u - lo);
    for (uint32_t h = 0; h < L.n_q; h++) {
        const uint32_t g = h / G;
        gref_norm(L, qin + (uint64_t)h * hd, L.qw, q.data());
        for (uint32_t i = 0; i < hd; i++) q[i] = gref_round(q[i] * tau);
        double mx = -INFINITY;
        for (uint32_t x = lo; x <= pos; x++) {
            const float *k = kh + ((uint64_t)x * L.n_kv + g) * hd;
            double dot = 0;
            for (uint32_t i = 0; i < hd; i++) dot += (double)q[i] * k[i];
            double b = 0;
            const uint32_t d = pos - x;
            if (L.extent && !mu.drop_bias && d < L.extent) {
                double rp = 0;
                for (uint32_t j = 0; j < 16; j++) rp += (double)r[h * 16u + j] * L.P[(uint64_t)j * L.extent + d];
                b = gref_round(gref_round((float)rp) * tau);
            }
            s[x - lo] = dot * L.scale + b;
            mx = s[x - lo] > mx ? s[x - lo] : mx;
        }
        double den = 0;
        for (auto &e : s) { e = exp(e - mx); den += e; }
        double *o = out + (uint64_t)h * hd;
        for (uint32_t i = 0; i < hd; i++) o[i] = 0;
        for (uint32_t x = lo; x <= pos; x++) {
            const float *v = vh + ((uint64_t)x * L.n_kv + g) * hd;
            const double p = s[x - lo] / den;
            for (uint32_t i = 0; i < hd; i++) o[i] += p * v[i];
        }
        if (L.gate) {
            for (uint32_t i = 0; i < hd; i++) o[i] *= 1.0 / (1.0 + exp(-(double)gate[(uint64_t)h * hd + i]));
        }
    }
}

#endif

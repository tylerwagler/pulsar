/* Host authority for the GQA attention block (src/cuda/pulsar_cuda_attn_gqa.h):
 * the declared arithmetic, with the attention itself in double.
 *
 *   q = bf16(rope(bf16(f32(x * rstd) * w_q)) * tau)  rstd = f32(1 / sqrt(mean(x^2) + eps))
 *   k = rope(bf16(f32(x * rstd) * w_k)),  v = v_in
 *   bias(d) = bf16(bf16(r . P[:, d]) * tau)          0 <= d < extent, else 0
 *   s_x = (q . k_x) * scale + bias(pos - x),  y = softmax(s) V  [* sigmoid(gate)]
 *
 * rope (L298, rope_theta > 0): the NeoX half-split rotation in the HF / vLLM bf16
 * arithmetic, transcribed here from the formula (deliberately NOT the kernel's
 * header): pair p of a d-dim head is (x[p], x[p + d/2]), inv = f32 1 / powf(base,
 * 2p/d), ang = f32(pos) * inv, c = bf16(cosf(ang)), s = bf16(sinf(ang)),
 * r0 = bf16(bf16(x0 c) - bf16(x1 s)), r1 = bf16(bf16(x1 c) + bf16(x0 s)).
 * The keys a row attends are lo .. hi: hi = pos (causal), or a block's run's last
 * position (bidirectional inside the run, no bias).
 *
 * tau is the module's own f32 formula (gqa_tau): the reference vLLM / transformers
 * compute it in torch f32, which the GPU does not see -- both read this value. */
#ifndef PULSAR_TESTS_GQA_REF_H
#define PULSAR_TESTS_GQA_REF_H

#include <math.h>
#include <stdlib.h>
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
    float rope_theta = 0.f;     /* > 0: NeoX rope of q and k */
};

/* how the rope is applied: as declared (NeoX), not at all, or with the interleaved
 * pair layout (x[2p], x[2p+1]) -- the last two are the gate's mutations */
enum gref_rope_mode { GREF_ROPE_NEOX = 0, GREF_ROPE_NONE = 1, GREF_ROPE_INTERLEAVED = 2 };

/* one head vector x (bf16 values, normed) rotated in place at `pos` */
static inline void gref_rope(const gref_layer &L, uint32_t pos, float *x, int mode = GREF_ROPE_NEOX) {
    if (!(L.rope_theta > 0.f) || mode == GREF_ROPE_NONE) return;
    const uint32_t d = L.hd, half = d / 2u;
    for (uint32_t p = 0; p < half; p++) {
        volatile float e = (float)(2u * p) / (float)d;
        volatile float inv = 1.0f / powf(L.rope_theta, e);
        volatile float ang = (float)pos * inv;
        const float c = gref_round(cosf(ang)), s = gref_round(sinf(ang));
        const uint32_t i0 = mode == GREF_ROPE_INTERLEAVED ? 2u * p : p;
        const uint32_t i1 = mode == GREF_ROPE_INTERLEAVED ? 2u * p + 1u : p + half;
        const float x0 = x[i0], x1 = x[i1];
        x[i0] = gref_round(gref_round(x0 * c) - gref_round(x1 * s));
        x[i1] = gref_round(gref_round(x1 * c) + gref_round(x0 * s));
    }
}

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

/* The attention of one row, over the keys lo .. hi (hi = pos: causal; a block row's
 * hi is its run's last position).  kh / vh: the sequence's stored K / V for the
 * positions base .. hi, [x - base][n_kv][hd] (the k already normed and roped);
 * qin, r, gate: the row's raw inputs (n_q x hd, n_q x 16, n_q x hd).  out: n_q x hd
 * doubles.  Mutations for the gate's discrimination checks: drop_bias, drop_tau,
 * window_delta, and the q rope's mode (the matching k comes from a History built
 * with the same mode). */
struct gref_mut { bool drop_bias = false, drop_tau = false; int window_delta = 0; int rope = GREF_ROPE_NEOX; };

static inline void gref_row(const gref_layer &L, uint32_t pos, uint32_t hi, uint32_t base, const float *kh,
                            const float *vh, const float *qin, const float *r, const float *gate, double *out,
                            const gref_mut &mu = gref_mut()) {
    const uint32_t G = L.n_q / L.n_kv, hd = L.hd;
    const float tau = mu.drop_tau ? 1.0f : gref_tau(L, pos);
    const uint32_t W = L.window ? (uint32_t)((int)L.window + mu.window_delta) : 0u;
    const uint32_t lo = W && pos + 1u > W ? pos + 1u - W : 0u;
    if (lo < base || hi < pos || (hi > pos && L.extent)) abort();   /* the history must hold every key; no block bias */
    std::vector<float> q(hd);
    std::vector<double> s(hi + 1u - lo);
    for (uint32_t h = 0; h < L.n_q; h++) {
        const uint32_t g = h / G;
        gref_norm(L, qin + (uint64_t)h * hd, L.qw, q.data());
        gref_rope(L, pos, q.data(), mu.rope);
        for (uint32_t i = 0; i < hd; i++) q[i] = gref_round(q[i] * tau);
        double mx = -INFINITY;
        for (uint32_t x = lo; x <= hi; x++) {
            const float *k = kh + ((uint64_t)(x - base) * L.n_kv + g) * hd;
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
        for (uint32_t x = lo; x <= hi; x++) {
            const float *v = vh + ((uint64_t)(x - base) * L.n_kv + g) * hd;
            const double p = s[x - lo] / den;
            for (uint32_t i = 0; i < hd; i++) o[i] += p * v[i];
        }
        if (L.gate) {
            for (uint32_t i = 0; i < hd; i++) o[i] *= 1.0 / (1.0 + exp(-(double)gate[(uint64_t)h * hd + i]));
        }
    }
}

#endif

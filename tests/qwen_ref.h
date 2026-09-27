/* Host references for the Qwen3.8-Flash-Next S4 pieces (L251): the router, the
 * Gated Residual read / write, the PLE injection and the MoE block, in double,
 * following transformers' modeling_qwen4_exp.py -- with the A8 encodings at
 * exactly the points the device encodes (the producer rule: an activation that
 * feeds an E4M3-weight or trellis GEMV is E4M3 per 32 with the shared exponent
 * floor(log2 amax) - 7) and the bf16 roundings where the device rounds (the
 * streams, the block-input row, the router's logits and weights).  Everything
 * else is double: the grade measures the device's f32 arithmetic against the
 * definition, not against a second f32 implementation.
 *
 * Shared by tests/qwen_s4_gate.cu (model-free) and tests/qwen_xcheck.cu
 * (real weights, driven by tools/qwen/s4_xcheck.py). */
#pragma once

#include "../src/cuda/pulsar_cuda_mx.cuh"
#include "../src/cuda/pulsar_cuda_qwen.h"
#include "exl3_dense_ref.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace qref {

constexpr int H = PULSAR_QWEN_HIDDEN, S = PULSAR_QWEN_HC, HC = PULSAR_QWEN_HC_HIDDEN, R = PULSAR_QWEN_HC_LOWRANK;
constexpr int E = PULSAR_QWEN_N_EXPERT, TOPK = PULSAR_QWEN_TOPK, MID = PULSAR_QWEN_EXPERT_MID;
constexpr double EPS = 1e-6;

inline double bf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
/** round-to-nearest-even to bf16 (from double via float: the double -> float step
 *  cannot straddle a bf16 tie that the direct rounding would not) */
inline uint16_t to_bf(double v) {
    float f = (float)v;
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
inline double bf_round(double v) { return bf(to_bf(v)); }
inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
inline double silu(double x) { return x * sigmoid(x); }

/** The producer encoding of one row of n (n % 32 == 0) into an A8 slot row:
 *  E4M3 bytes + the swizzled scale bytes, and the decoded values. */
inline void mx_encode_row(const double *v, int n, int row, int kbp, uint8_t *q, uint8_t *sf, double *dec) {
    for (int g = 0; g < n / 32; g++) {
        double amax = 0;
        for (int j = 0; j < 32; j++) amax = fmax(amax, fabs(v[g * 32 + j]));
        int se = amax > 0 ? (int)floor(log2(amax)) - 7 : -127;     /* pulsar_mx_shared_exp */
        se = std::max(-127, std::min(127, se));
        if (sf) sf[pulsar_mx_sfoff(row, g, kbp)] = (uint8_t)(se + 127);
        for (int j = 0; j < 32; j++) {
            const uint8_t b = exl3t_f64_to_e4m3(v[g * 32 + j] * ldexp(1.0, -se));
            if (q) q[g * 32 + j] = b;
            if (dec) dec[g * 32 + j] = exl3t_e4m3_to_f64(b) * ldexp(1.0, se);
        }
    }
}
/** decode row `row` of an A8 slot */
inline void mx_decode_row(const uint8_t *q, const uint8_t *sf, int n, int row, int kbp, double *out) {
    for (int k = 0; k < n; k++)
        out[k] = exl3t_e4m3_to_f64(q[(size_t)row * n + k]) * ldexp(1.0, (int)sf[pulsar_mx_sfoff(row, k / 32, kbp)] - 127);
}

/** An MXFP8_LT weight on the host: [out][in] codes + swizzled E8M0, decoded. */
struct mx8 {
    int out = 0, in = 0;
    std::vector<uint8_t> q, sf;
    double at(int r, int k) const {
        return exl3t_e4m3_to_f64(q[(size_t)r * in + k]) *
               ldexp(1.0, (int)sf[pulsar_mx_sfoff(r, k / 32, pulsar_mx_kbp(in))] - 127);
    }
    /** encode a real-valued [out][in] matrix the way the builder does
     *  (tools/container/producers.py _quantize_fp8_e4m3_planes + swizzle) */
    void encode(const std::vector<double> &w, int o, int i) {
        out = o; in = i;
        q.assign((size_t)o * i, 0);
        sf.assign(pulsar_mx_sf_slab_bytes(o, pulsar_mx_kbp(i)), 0);
        for (int r = 0; r < o; r++) mx_encode_row(w.data() + (size_t)r * i, i, r, pulsar_mx_kbp(i), q.data() + (size_t)r * i, sf.data(), nullptr);
    }
};

/* ---- router ------------------------------------------------------------ */

/** logits[e] for e < E (+ the shared gate at E), double */
inline void router_logits(const uint16_t *x, const uint16_t *wr, const uint16_t *wsg, double *lg) {
    for (int e = 0; e <= E; e++) {
        const uint16_t *w = e < E ? wr + (size_t)e * H : wsg;
        double s = 0;
        for (int k = 0; k < H; k++) s += bf(w[k]) * bf(x[k]);
        lg[e] = s;
    }
}
/** the decision from (f32) logits: bf16, softmax, top-k (ties -> lower id),
 *  renormalised, bf16 weights; sgate */
inline void router_select(const float *lg, int32_t *sel, double *w, double *sgate) {
    std::vector<double> p(E);
    double m = -INFINITY;
    for (int e = 0; e < E; e++) { p[e] = bf_round(lg[e]); m = fmax(m, p[e]); }
    double z = 0;
    for (int e = 0; e < E; e++) { p[e] = exp(p[e] - m); z += p[e]; }
    for (int e = 0; e < E; e++) p[e] /= z;
    std::vector<int> idx(E);
    for (int e = 0; e < E; e++) idx[e] = e;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return p[a] > p[b]; });
    double s = 0;
    for (int k = 0; k < TOPK; k++) s += p[idx[k]];
    for (int k = 0; k < TOPK; k++) { sel[k] = idx[k]; w[k] = bf_round(p[idx[k]] / s); }
    *sgate = bf_round(sigmoid(bf_round(lg[E])));
}

/* ---- Gated Residual ---------------------------------------------------- */

struct gr_out {
    std::vector<double> x;     ///< [H] the block input before its bf16 rounding
    double inj[S];
};
/** the read for one token; streams bf16 [S][H] */
inline gr_out gr_read(const uint16_t *streams, const uint16_t *norm_w, const mx8 &down, const mx8 &up,
                      const uint16_t *inject) {
    std::vector<double> xn(HC), xnd(HC);
    for (int s = 0; s < S; s++) {
        double ss = 0;
        for (int c = 0; c < H; c++) { const double v = bf(streams[s * H + c]); ss += v * v; }
        const double r = 1.0 / sqrt(ss / H + EPS);
        for (int c = 0; c < H; c++) xn[s * H + c] = bf(streams[s * H + c]) * r * (1.0 + bf(norm_w[s * H + c]));
    }
    mx_encode_row(xn.data(), HC, 0, pulsar_mx_kbp(HC), nullptr, nullptr, xnd.data());
    gr_out o;
    for (int j = 0; j < S; j++) {
        double z = 0;
        if (inject) for (int k = 0; k < HC; k++) z += bf(inject[(size_t)j * HC + k]) * xn[k];
        o.inj[j] = 2.0 * sigmoid(z / S);
    }
    std::vector<double> a(R), ad(R);
    for (int r = 0; r < R; r++) {
        double d = 0;
        for (int k = 0; k < HC; k++) d += down.at(r, k) * xnd[k];
        a[r] = silu(d / S);
    }
    mx_encode_row(a.data(), R, 0, pulsar_mx_kbp(R), nullptr, nullptr, ad.data());
    o.x.assign(H, 0.0);
    for (int c = 0; c < H; c++) {
        double acc = 0;
        for (int s = 0; s < S; s++) {
            double z = 0;
            for (int k = 0; k < R; k++) z += up.at(s * H + c, k) * ad[k];
            acc += sigmoid(z) * xn[s * H + c];
        }
        o.x[c] = acc / S;
    }
    return o;
}

/* ---- the dense Linear (EXL3) ---------------------------------------------- */

struct linear {
    int in = 0, out = 0, k2 = 0;
    std::vector<uint8_t> bytes;
    std::vector<double> what;
    void dequant() { exl3t_dequant(bytes.data(), in, out, k2, what); }
    /** y [out] from decoded activation x [in] */
    void run(const double *x, double *y) const {
        std::vector<double> yy;
        exl3t_reference(bytes.data(), what, in, out, k2, x, 1, yy);
        std::copy(yy.begin(), yy.end(), y);
    }
};

/* ---- PLE ----------------------------------------------------------------- */

/** one token's injection; hist = this sequence's gvn history (every earlier
 *  token's gvn, oldest first, zeros before the start), appended to */
inline void ple_token(const uint16_t *emb, const linear &key, const linear &value, const uint16_t *nk,
                      const uint16_t *nq, const uint16_t *nc, const uint16_t *conv_w, uint16_t *streams,
                      std::vector<std::vector<double>> &hist) {
    std::vector<double> e(H), ed(H), k(HC), v(H);
    for (int c = 0; c < H; c++) e[c] = bf(emb[c]);
    mx_encode_row(e.data(), H, 0, pulsar_mx_kbp(H), nullptr, nullptr, ed.data());
    key.run(ed.data(), k.data());
    value.run(ed.data(), v.data());
    std::vector<double> gv(HC), gvn(HC);
    for (int s = 0; s < S; s++) {
        double ssk = 0, ssq = 0;
        for (int c = 0; c < H; c++) { ssk += k[s * H + c] * k[s * H + c]; const double q = bf(streams[s * H + c]); ssq += q * q; }
        const double rk = 1.0 / sqrt(ssk / H + EPS), rq = 1.0 / sqrt(ssq / H + EPS);
        double dot = 0;
        for (int c = 0; c < H; c++)
            dot += k[s * H + c] * rk * (1.0 + bf(nk[s * H + c])) * bf(streams[s * H + c]) * rq * (1.0 + bf(nq[s * H + c]));
        const double g = dot / sqrt((double)H);
        const double gs = g > 0 ? sqrt(fmax(g, 1e-6)) : g < 0 ? -sqrt(fmax(-g, 1e-6)) : 0.0;
        const double sg = sigmoid(gs);
        double ssv = 0;
        for (int c = 0; c < H; c++) { gv[s * H + c] = sg * v[c]; ssv += gv[s * H + c] * gv[s * H + c]; }
        const double rc = 1.0 / sqrt(ssv / H + EPS);
        for (int c = 0; c < H; c++) gvn[s * H + c] = gv[s * H + c] * rc * (1.0 + bf(nc[s * H + c]));
    }
    hist.push_back(gvn);
    const int n = (int)hist.size();
    for (int ch = 0; ch < HC; ch++) {
        double y = 0;
        for (int m = 0; m < PULSAR_QWEN_PLE_TAPS; m++) {
            const int back = (PULSAR_QWEN_PLE_TAPS - 1 - m) * PULSAR_QWEN_PLE_DIL;
            const double tap = n - 1 - back >= 0 ? hist[n - 1 - back][ch] : 0.0;
            y += bf(conv_w[ch * PULSAR_QWEN_PLE_TAPS + m]) * tap;
        }
        streams[ch] = to_bf(bf(streams[ch]) + gv[ch] + silu(y));
    }
}

/* ---- MoE ------------------------------------------------------------------ */

struct expert { const linear *gate, *up, *down; };

/** the block for one token given the routing (sel, w, sgate -- the device's);
 *  x = the decoded E4M3 block input */
inline void moe_token(const double *x, const int32_t *sel, const double *w, double sgate,
                      const std::vector<expert> &experts, const linear &sg, const linear &su, const linear &sd,
                      double *out) {
    std::fill(out, out + H, 0.0);
    std::vector<double> yg(MID), yu(MID), t(MID), td(MID), y(H);
    for (int k = 0; k < TOPK; k++) {
        const expert &ex = experts[sel[k]];
        ex.gate->run(x, yg.data());
        ex.up->run(x, yu.data());
        /* the fold: v = silu(g) u w (the router weight folded in before the encode,
         * as the device does), then the down input's rotation t = H(v suh_d), E4M3;
         * the down GEMV takes t as it is (pre-rotated) and the sum applies H + svh */
        uint64_t trellis = 0, stride = 0;
        exl3t_layout(MID, H, ex.down->k2, &trellis, &stride);
        const uint16_t *suh = (const uint16_t *)(ex.down->bytes.data() + trellis);
        for (int n = 0; n < MID; n++) t[n] = silu(yg[n]) * yu[n] * w[k] * exl3_f16_to_f32(suh[n]);
        for (int i = 0; i < MID; i += 128) exl3_had128(t.data() + i);
        mx_encode_row(t.data(), MID, 0, pulsar_mx_kbp(MID), nullptr, nullptr, td.data());
        /* y = svh H(W^T td): the down's reference minus its input rotation */
        std::vector<double> z(H, 0.0);
        for (int kk = 0; kk < MID; kk++)
            for (int o = 0; o < H; o++) z[o] += ex.down->what[(size_t)kk * H + o] * td[kk];
        for (int i = 0; i < H; i += 128) exl3_had128(z.data() + i);
        const uint16_t *svh = suh + MID;
        for (int o = 0; o < H; o++) out[o] += z[o] * exl3_f16_to_f32(svh[o]);
    }
    std::vector<double> hg(sg.out), hu(su.out), h(sg.out), hd(sg.out), ys(H);
    sg.run(x, hg.data());
    su.run(x, hu.data());
    for (int n = 0; n < sg.out; n++) h[n] = silu(hg[n]) * hu[n];
    mx_encode_row(h.data(), sg.out, 0, pulsar_mx_kbp(sg.out), nullptr, nullptr, hd.data());
    sd.run(hd.data(), ys.data());
    for (int o = 0; o < H; o++) out[o] += sgate * ys[o];
}

} // namespace qref

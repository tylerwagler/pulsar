/* What does tensor-core attention actually cost in fidelity?
 *
 * docs/engine-perf-map.md puts attention at 22-25% of prefill with pipe_tensor
 * at 0% on both kernels: the matmuls run on the FP32 FMA pipe with the tensor
 * cores idle.  Moving them onto the tensor cores means picking an operand
 * format, and that is a fidelity decision, not a free win.  This measures the
 * cost of each candidate so the decision comes from numbers.
 *
 * THE ARCHITECTURE MATTERS.  This is MLA: pulsar_gpu_attention_prefill_raw_
 * heads_tensor passes raw_kv as BOTH the key and the value operand, so there is
 * ONE latent 512-dim vector per position, not a separate K and V.  A format has
 * to survive two different jobs at once -- the score dot product and the
 * weighted value sum -- and there is no option to keep V wider than K.
 *
 * WHAT IS QUANTISED.  A tensor-core attention quantises the GEMM OPERANDS and
 * accumulates in f32; the softmax stays f32.  So:
 *   scores:  quant(q) . quant(kv)            accumulated f32
 *   softmax: f32, unchanged
 *   output:  sum_r quant(p_r) * quant(kv_r)  accumulated f32
 * P is a GEMM operand too and is quantised like the rest -- that is the half
 * that gets forgotten, and for the 4-bit formats it is what actually hurts.
 *
 * THE BASELINE IS THE POINT.  The kernel shipping TODAY already computes in f32
 * and already has an error against f64.  Every candidate is reported as a
 * MULTIPLE of that existing error, because "bf16 costs 2x the error we already
 * ship" is a decision and "bf16 has relative error 3.7e-3" is not.
 *
 * All formats run the same two-pass arithmetic, so the comparison isolates
 * operand precision.  The shipping kernel folds its softmax online instead;
 * that difference is a second-order f32 effect and is not what is being
 * measured here.
 *
 * Real activations, dumped from a real prefill, because the question is dynamic
 * range: Q outlier channels and the spread of the latent kv vector are exactly
 * what a Gaussian sample would not reproduce.
 *
 * build+run:
 *   g++ -O2 -o /tmp/attnfid tests/attn_precision_fidelity.cc -lm
 *   PULSAR_DUMP_ATTN=/tmp/attn.bin ./pulsar-bench -m <model> \
 *       --prompt-file <f> --ctx-max 4000 --gen-tokens 0
 *   /tmp/attnfid /tmp/attn.bin [n_tokens_to_score]
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

/* ---- operand formats ---------------------------------------------------- */

/* Round-to-nearest-even onto a binary float grid with man_bits explicit
 * mantissa bits and minimum normal exponent emin.  Saturates to maxfin, like
 * the hardware converts on this path (cvt.rn.satfinite). */
static double quant_float(double v, int man_bits, int emin, double maxfin) {
    if (!std::isfinite(v)) return v;
    const double a = std::fabs(v);
    if (a == 0.0) return v;
    const double sgn = v < 0 ? -1.0 : 1.0;
    if (a >= maxfin) return sgn * maxfin;
    int e; std::frexp(a, &e); e -= 1;         /* a in [2^e, 2^(e+1)) */
    if (e < emin) e = emin;                   /* subnormal: fixed quantum */
    const double step = std::ldexp(1.0, e - man_bits);
    double r = std::nearbyint(a / step) * step;   /* nearbyint: ties-to-even */
    if (r >= maxfin) r = maxfin;
    return sgn * r;
}

static double q_fp16(double v) { return quant_float(v, 10, -14, 65504.0); }
static double q_bf16(double v) { return quant_float(v,  7, -126, 3.3895313892515355e38); }
static double q_e4m3(double v) { return quant_float(v,  3, -6, 448.0); }
static double q_e5m2(double v) { return quant_float(v,  2, -14, 57344.0); }
static double q_f32(double v)  { return (double)(float)v; }

/* e2m1: 4-bit, no subnormal ladder worth modelling -- snap to the value table. */
static const double kE2M1[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
static double q_e2m1(double v) {
    const double a = std::fabs(v), sgn = v < 0 ? -1.0 : 1.0;
    double best = 0.0, bd = 1e300;
    for (int i = 0; i < 8; i++) {
        const double d = std::fabs(a - kE2M1[i]);
        if (d < bd) { bd = d; best = kE2M1[i]; }
    }
    return sgn * best;
}

enum Fmt { F32, FP16, BF16, E4M3, E5M2, MXFP8, MXFP4, NFMT };
static const char *fmt_name(Fmt f) {
    switch (f) {
        case F32:   return "f32  (SHIPPING TODAY)";
        case FP16:  return "fp16";
        case BF16:  return "bf16";
        case E4M3:  return "e4m3 (unscaled)";
        case E5M2:  return "e5m2 (unscaled)";
        case MXFP8: return "MXFP8 e4m3+e8m0/32";
        default:    return "MXFP4 e2m1+e8m0/32";
    }
}

/* Block-scaled formats need the whole vector, since the e8m0 scale is shared
 * across each 32 -- the layout this device consumes natively (see
 * docs/indexer-mxfp4-scorer.md for the measured instruction contract). */
static void quantize_vec(Fmt f, const double *in, double *out, size_t n) {
    if (f == MXFP8 || f == MXFP4) {
        const int emax = (f == MXFP8) ? 8 : 2;   /* e4m3 emax 8, e2m1 emax 2 */
        for (size_t b = 0; b < n; b += 32) {
            const size_t m = std::min((size_t)32, n - b);
            double amax = 0.0;
            for (size_t i = 0; i < m; i++) amax = std::max(amax, std::fabs(in[b + i]));
            int se = -127;
            if (amax > 0.0) { int e; std::frexp(amax, &e); se = (e - 1) - emax; }
            const double s = std::ldexp(1.0, se), inv = std::ldexp(1.0, -se);
            for (size_t i = 0; i < m; i++)
                out[b + i] = (f == MXFP8 ? q_e4m3(in[b + i] * inv)
                                         : q_e2m1(in[b + i] * inv)) * s;
        }
        return;
    }
    double (*q)(double) = q_f32;
    switch (f) {
        case FP16: q = q_fp16; break; case BF16: q = q_bf16; break;
        case E4M3: q = q_e4m3; break; case E5M2: q = q_e5m2; break;
        default:   q = q_f32;  break;
    }
    for (size_t i = 0; i < n; i++) out[i] = q(in[i]);
}

/* ---- dump reader -------------------------------------------------------- */

struct Dump {
    uint32_t n_tokens, n_head, head_dim, window, raw_f16, tok0, ndump;
    std::vector<float>  q;      /* [ndump][n_head][head_dim] */
    std::vector<double> kv;     /* [n_tokens][head_dim] */
    std::vector<float>  sinks;  /* [n_head] */
};

static double half_to_double(uint16_t h) {
    const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    double v;
    if (e == 0)       v = std::ldexp((double)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else              v = std::ldexp((double)(m | 0x400u), (int)e - 25);
    return s ? -v : v;
}

static int load_dump(const char *path, Dump &d) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 0; }
    uint32_t hdr[8];
    if (fread(hdr, sizeof(hdr), 1, f) != 1 || hdr[0] != 0x41545444u) {
        fprintf(stderr, "%s: not an attention dump\n", path); fclose(f); return 0;
    }
    d.n_tokens = hdr[1]; d.n_head = hdr[2]; d.head_dim = hdr[3];
    d.window = hdr[4]; d.raw_f16 = hdr[5]; d.tok0 = hdr[6]; d.ndump = hdr[7];
    const size_t qn = (size_t)d.ndump * d.n_head * d.head_dim;
    const size_t kvn = (size_t)d.n_tokens * d.head_dim;
    d.q.resize(qn); d.kv.resize(kvn); d.sinks.resize(d.n_head);
    int ok = fread(d.q.data(), sizeof(float), qn, f) == qn;
    if (ok) {
        if (d.raw_f16) {
            std::vector<uint16_t> t(kvn);
            ok = fread(t.data(), 2, kvn, f) == kvn;
            for (size_t i = 0; ok && i < kvn; i++) d.kv[i] = half_to_double(t[i]);
        } else {
            std::vector<float> t(kvn);
            ok = fread(t.data(), 4, kvn, f) == kvn;
            for (size_t i = 0; ok && i < kvn; i++) d.kv[i] = (double)t[i];
        }
    }
    if (ok) ok = fread(d.sinks.data(), sizeof(float), d.n_head, f) == d.n_head;
    fclose(f);
    if (!ok) { fprintf(stderr, "%s: short read\n", path); return 0; }
    return 1;
}

/* ---- one (token, head) attention ---------------------------------------- */
/* Mirrors attention_static_mixed_heads8_online_kernel: window over the latent
 * rows, one vector serving as both key and value, plus the per-head sink.
 * kvq is the pre-quantised kv for this format (it does not depend on the head,
 * so quantising it per head would be pure waste).  fmt == NFMT means f64. */
static void attend(const Dump &d, uint32_t ti, uint32_t head, Fmt fmt, int reference,
                   const std::vector<double> &kvq, std::vector<double> &out,
                   std::vector<double> &prob) {
    const uint32_t D = d.head_dim;
    const uint32_t t = d.tok0 + ti;
    const uint32_t cnt = (d.window != 0u && t + 1u > d.window) ? d.window : t + 1u;
    const uint32_t start = t + 1u - cnt;
    const double scale = 1.0 / std::sqrt((double)D);

    std::vector<double> qd(D), qq(D);
    const float *qs = &d.q[((size_t)ti * d.n_head + head) * D];
    for (uint32_t i = 0; i < D; i++) qd[i] = (double)qs[i];
    if (reference) qq = qd; else quantize_vec(fmt, qd.data(), qq.data(), D);

    prob.assign(cnt, 0.0);
    double mx = -INFINITY;
    for (uint32_t r = 0; r < cnt; r++) {
        const double *k = &kvq[(size_t)(start + r) * D];
        double acc = 0.0;
        if (reference) { for (uint32_t i = 0; i < D; i++) acc += qq[i] * k[i]; }
        else { float a = 0.f; for (uint32_t i = 0; i < D; i++) a += (float)(qq[i] * k[i]); acc = a; }
        prob[r] = acc * scale;
        mx = std::max(mx, prob[r]);
    }
    const double sink = (double)d.sinks[head];
    mx = std::max(mx, sink);
    double sum = std::exp(sink - mx);
    for (uint32_t r = 0; r < cnt; r++) { prob[r] = std::exp(prob[r] - mx); sum += prob[r]; }
    const double inv = sum > 0.0 ? 1.0 / sum : 0.0;
    for (uint32_t r = 0; r < cnt; r++) prob[r] *= inv;
    /* Carry the sink as a final bucket so the weights are a PROPER distribution
     * summing to 1.  Without it sum(prob) < 1 and KL(ref||var) can come out
     * negative, which is how the first version of this test reported fp16. */
    prob.push_back(std::exp(sink - mx) * inv);

    /* P is a GEMM operand: quantise it too.  Only the cnt real rows -- the sink
     * bucket appended above is a normalisation term, not an operand. */
    std::vector<double> pq(cnt);
    if (reference) pq.assign(prob.begin(), prob.begin() + cnt);
    else quantize_vec(fmt, prob.data(), pq.data(), cnt);

    out.assign(D, 0.0);
    if (reference) {
        for (uint32_t r = 0; r < cnt; r++) {
            const double *v = &kvq[(size_t)(start + r) * D];
            for (uint32_t i = 0; i < D; i++) out[i] += pq[r] * v[i];
        }
    } else {
        std::vector<float> acc(D, 0.f);
        for (uint32_t r = 0; r < cnt; r++) {
            const double *v = &kvq[(size_t)(start + r) * D];
            for (uint32_t i = 0; i < D; i++) acc[i] += (float)(pq[r] * v[i]);
        }
        for (uint32_t i = 0; i < D; i++) out[i] = acc[i];
    }
}

struct Stat { double mean = 0, p99 = 0, max = 0; };
static Stat summarize(std::vector<double> &v) {
    Stat s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    for (double x : v) s.mean += x;
    s.mean /= (double)v.size();
    s.p99 = v[(size_t)((double)(v.size() - 1) * 0.99)];
    s.max = v.back();
    return s;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <attn dump> [tokens]\n", argv[0]); return 2; }
    Dump d;
    if (!load_dump(argv[1], d)) return 1;
    uint32_t ntok = (argc > 2) ? (uint32_t)atoi(argv[2]) : 24u;
    if (ntok > d.ndump) ntok = d.ndump;

    printf("attention operand-precision fidelity\n");
    printf("  dump: n_tokens=%u n_head=%u head_dim=%u window=%u raw_f16=%u"
           " (scoring %u tokens x %u heads from tok0=%u)\n\n",
           d.n_tokens, d.n_head, d.head_dim, d.window, d.raw_f16, ntok, d.n_head, d.tok0);

    const size_t kvn = (size_t)d.n_tokens * d.head_dim;

    /* f64 reference: kv unquantised. */
    std::vector<double> ref_out, ref_prob, tmp_out, tmp_prob;
    std::vector<std::vector<double>> ref_all(ntok * d.n_head), refp_all(ntok * d.n_head);
    for (uint32_t ti = 0; ti < ntok; ti++)
        for (uint32_t h = 0; h < d.n_head; h++) {
            attend(d, ti, h, NFMT, 1, d.kv, ref_out, ref_prob);
            ref_all[ti * d.n_head + h] = ref_out;
            refp_all[ti * d.n_head + h] = ref_prob;
        }

    /* Operand range, and how much of it each format cannot hold.  fp16 tops out
     * at 65504 and e4m3 at 448, so "does the data even fit" is a separate
     * question from "how finely is it resolved". */
    {
        double qmax = 0.0, kmax = 0.0;
        for (float v : d.q) qmax = std::max(qmax, (double)std::fabs(v));
        for (double v : d.kv) kmax = std::max(kmax, std::fabs(v));
        printf("  operand range: max|q| = %.4g   max|kv| = %.4g\n", qmax, kmax);
        printf("  saturating elements (operand clipped to the format's max finite):\n");
        for (int fi = 0; fi < (int)NFMT; fi++) {
            const Fmt f = (Fmt)fi;
            if (f == MXFP8 || f == MXFP4) continue;      /* block scale removes the question */
            const double lim = (f == FP16) ? 65504.0 : (f == BF16) ? 3.389531e38
                             : (f == E4M3) ? 448.0 : (f == E5M2) ? 57344.0 : 3.4028235e38;
            size_t nq = 0, nk = 0;
            for (float v : d.q) if (std::fabs((double)v) >= lim) nq++;
            for (double v : d.kv) if (std::fabs(v) >= lim) nk++;
            printf("    %-22s q %zu/%zu   kv %zu/%zu\n", fmt_name(f), nq, d.q.size(), nk, d.kv.size());
        }
        printf("\n");
    }

    printf("  %-22s %11s %11s %11s %10s %11s %9s\n",
           "format", "rel L2 mean", "rel L2 p99", "rel L2 max", "vs f32", "attn-wt KL", "top1 same");
    printf("  %-22s %11s %11s %11s %10s %11s %9s\n",
           "----------------------", "-----------", "-----------", "-----------",
           "----------", "-----------", "---------");

    double f32_mean = 0.0;
    for (int fi = 0; fi < (int)NFMT; fi++) {
        const Fmt f = (Fmt)fi;
        std::vector<double> kvq(kvn);
        for (uint32_t r = 0; r < d.n_tokens; r++)
            quantize_vec(f, &d.kv[(size_t)r * d.head_dim], &kvq[(size_t)r * d.head_dim], d.head_dim);

        std::vector<double> rel, kls;
        size_t top1_same = 0, top1_n = 0;
        rel.reserve(ntok * d.n_head); kls.reserve(ntok * d.n_head);
        for (uint32_t ti = 0; ti < ntok; ti++)
            for (uint32_t h = 0; h < d.n_head; h++) {
                attend(d, ti, h, f, 0, kvq, tmp_out, tmp_prob);
                const std::vector<double> &R = ref_all[ti * d.n_head + h];
                double num = 0.0, den = 0.0;
                for (size_t i = 0; i < R.size(); i++) {
                    const double e = tmp_out[i] - R[i];
                    num += e * e; den += R[i] * R[i];
                }
                rel.push_back(den > 0.0 ? std::sqrt(num / den) : 0.0);
                /* KL(ref || variant) over the attention weights: does it still
                 * attend to the same positions, independent of output scale. */
                const std::vector<double> &P = refp_all[ti * d.n_head + h];
                double kl = 0.0;
                for (size_t r = 0; r < P.size() && r < tmp_prob.size(); r++)
                    if (P[r] > 1e-300) {
                        const double qv = tmp_prob[r] > 1e-300 ? tmp_prob[r] : 1e-300;
                        kl += P[r] * std::log(P[r] / qv);
                    }
                kls.push_back(kl);
                /* Does it still attend hardest to the same position?  The most
                 * interpretable form of "did the behaviour change". */
                if (!P.empty() && !tmp_prob.empty()) {
                    const size_t ai = std::max_element(P.begin(), P.end()) - P.begin();
                    const size_t bi = std::max_element(tmp_prob.begin(), tmp_prob.end()) - tmp_prob.begin();
                    top1_n++; if (ai == bi) top1_same++;
                }
            }
        Stat s = summarize(rel), k = summarize(kls);
        if (f == F32) f32_mean = s.mean;
        char mult[32];
        if (f == F32) snprintf(mult, sizeof(mult), "%s", "1.0x (base)");
        else snprintf(mult, sizeof(mult), "%.0fx", f32_mean > 0 ? s.mean / f32_mean : 0.0);
        printf("  %-22s %11.3e %11.3e %11.3e %10s %11.3e %8.2f%%\n",
               fmt_name(f), s.mean, s.p99, s.max, mult, k.mean,
               top1_n ? 100.0 * (double)top1_same / (double)top1_n : 0.0);
        fflush(stdout);
    }

    printf("\n  rel L2 = ||out - ref||/||ref|| per (token, head), f64 reference.\n");
    printf("  'vs f32' is the multiple of the error the SHIPPING kernel already has.\n");
    printf("  attn-wt KL = KL(ref || variant) over the softmax weights, mean.\n");
    return 0;
}

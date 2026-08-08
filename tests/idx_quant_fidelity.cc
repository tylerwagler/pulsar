/* Indexer scorer fork decision: E4M3 vs E2M1 for the Q operand.
 *
 * Host-only, no GPU, no CUDA.  The MMA contract is already settled empirically
 * (tests/idx_mxfp4_probe.cu); what is left is a pure QUANTISATION question, and
 * that does not need tensor cores to answer:
 *
 *   score[t][c] = ( sum_h ReLU(q[t,h,:] . k[c,:]) * w[t,h] ) * scale
 *
 * K is free either way -- the compressed cache rows are QAT-roundtripped onto
 * the E2M1 x E8M0 grid already, so feeding them natively is bit-identical to
 * what the current dequantiser reconstructs.  Q is the ONLY lossy operand, and
 * today it goes f32 -> __half (10 mantissa bits).  The fork is what replaces
 * that: E4M3 (3 mantissa bits + per-32 E8M0) or E2M1 (1 bit + per-32 E8M0).
 *
 * THE METRIC IS NOT SCORE ERROR.  The scorer's only consumer is top-k
 * selection, so what matters is whether the SELECTED SET changes.  A large
 * score error that preserves the ranking costs nothing; a small one that
 * reorders the boundary costs rows.  Both are reported, because the gap
 * between them is the whole argument.
 *
 * Q distribution is swept, because top-k overlap is sensitive to the score
 * distribution and a single synthetic draw could flatter either format.  If
 * the answer is stable across shapes, the conclusion is robust to not having
 * captured real Q yet; if it is not, that says real data is required before
 * committing.
 *
 * build+run:
 *   g++ -O2 -o /tmp/idx_quant_fidelity tests/idx_quant_fidelity.cc -lm
 *   /tmp/idx_quant_fidelity
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>

static const int N_HEAD   = 64;
static const int HEAD_DIM = 128;
static const int BLK      = 32;    /* MX block: 32 elements share one E8M0 scale */

/* ---- element grids ------------------------------------------------------ */

/* E2M1: the 8 magnitudes our cache is roundtripped onto. */
static const double kE2M1[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};

static double quant_e2m1(double v) {
    const double a = std::fabs(v);
    double best = kE2M1[0]; double bd = std::fabs(a - kE2M1[0]);
    for (int i = 1; i < 8; i++) {
        const double d = std::fabs(a - kE2M1[i]);
        if (d < bd) { bd = d; best = kE2M1[i]; }
    }
    return v < 0 ? -best : best;
}

/* fp16: what the CURRENT kernel converts Q to (__float2half).  This, not f64,
 * is the deployed baseline -- the decision is about what CHANGES versus today,
 * so it gets its own column. */
static double quant_fp16(double v) {
    const double a = std::fabs(v);
    if (a == 0.0) return 0.0;
    if (a < 5.9604644775390625e-08 * 0.5) return 0.0;
    int e = (int)std::floor(std::log2(a));
    if (e < -14) e = -14;
    const double step = std::ldexp(1.0, e - 10);   /* 10 mantissa bits */
    double q = std::round(a / step) * step;
    if (q > 65504.0) q = 65504.0;
    return v < 0 ? -q : q;
}

/* E4M3: 4-bit exponent (bias 7), 3-bit mantissa, max finite 448. */
static double quant_e4m3(double v) {
    const double a = std::fabs(v);
    if (a == 0.0) return 0.0;
    const double maxv = 448.0, minsub = 0.001953125 /* 2^-9 */;
    if (a >= maxv) return v < 0 ? -maxv : maxv;
    if (a < minsub * 0.5) return 0.0;
    int e = (int)std::floor(std::log2(a));
    if (e < -6) e = -6;                       /* subnormal region */
    const double step = std::ldexp(1.0, e - 3);   /* 3 mantissa bits */
    double q = std::round(a / step) * step;
    if (q > maxv) q = maxv;
    return v < 0 ? -q : q;
}

/* ---- MX block quantisation (shared power-of-two scale per 32) ----------- */

template <typename QFn>
static void quant_block(const double *in, double *out, int n, double emax_elem, QFn qf) {
    for (int b = 0; b < n; b += BLK) {
        const int len = std::min(BLK, n - b);
        double amax = 0.0;
        for (int i = 0; i < len; i++) amax = std::max(amax, std::fabs(in[b + i]));
        if (amax == 0.0) { for (int i = 0; i < len; i++) out[b + i] = 0.0; continue; }
        /* MX shared scale: align the block's top exponent with the format's. */
        const int se = (int)std::floor(std::log2(amax)) - (int)emax_elem;
        const double s = std::ldexp(1.0, se);
        for (int i = 0; i < len; i++) out[b + i] = qf(in[b + i] / s) * s;
    }
}

/* ---- scoring ------------------------------------------------------------ */

static void score_rows(const std::vector<double> &q,     /* [head][dim] one token */
                       const std::vector<double> &k,     /* [comp][dim] */
                       const std::vector<double> &w,     /* [head] */
                       int n_comp, std::vector<double> &out) {
    out.assign(n_comp, 0.0);
    for (int c = 0; c < n_comp; c++) {
        double acc = 0.0;
        for (int h = 0; h < N_HEAD; h++) {
            const double *qh = &q[(size_t)h * HEAD_DIM];
            const double *kc = &k[(size_t)c * HEAD_DIM];
            double d = 0.0;
            for (int i = 0; i < HEAD_DIM; i++) d += qh[i] * kc[i];
            if (d > 0.0) acc += d * w[h];
        }
        out[c] = acc;
    }
}

static std::vector<int> topk(const std::vector<double> &s, int k) {
    std::vector<int> idx(s.size());
    for (size_t i = 0; i < s.size(); i++) idx[i] = (int)i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return s[a] > s[b]; });
    idx.resize(k);
    std::sort(idx.begin(), idx.end());
    return idx;
}

/* WHERE do the displaced rows sit in the reference ranking?  The whole case for
 * tolerating any churn is that the rows that change are the lowest-scoring of
 * the selected set -- the ones the attention barely weights.  That is an
 * assertion until measured, so measure it: report the reference RANK of every
 * row that the reference selected and the candidate dropped.  Rank 0 = highest
 * scoring.  If the dropped ranks cluster just under top_k, the argument holds;
 * if any land near 0, it does not. */
static void dropped_ranks(const std::vector<double> &sref, const std::vector<int> &tref,
                          const std::vector<int> &tcand, int top_k,
                          double *mean_rank, int *best_rank) {
    std::vector<int> order(sref.size());
    for (size_t i = 0; i < sref.size(); i++) order[i] = (int)i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return sref[a] > sref[b]; });
    std::vector<int> rank(sref.size());
    for (size_t r = 0; r < order.size(); r++) rank[order[r]] = (int)r;

    std::vector<int> dropped;
    std::set_difference(tref.begin(), tref.end(), tcand.begin(), tcand.end(),
                        std::back_inserter(dropped));
    if (dropped.empty()) { *mean_rank = (double)top_k; *best_rank = top_k; return; }
    double sum = 0; int best = (int)sref.size();
    for (int d : dropped) { sum += rank[d]; best = std::min(best, rank[d]); }
    *mean_rank = sum / (double)dropped.size();
    *best_rank = best;
}

static int overlap(const std::vector<int> &a, const std::vector<int> &b) {
    std::vector<int> inter;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(inter));
    return (int)inter.size();
}

int main(int argc, char **argv) {
    const int n_comp = (argc > 1) ? atoi(argv[1]) : 1024;
    const int top_k  = (argc > 2) ? atoi(argv[2]) : 512;
    const int trials = (argc > 3) ? atoi(argv[3]) : 8;

    printf("indexer scorer fork: Q in E4M3 vs E2M1   (K exact on the E2M1 grid)\n");
    printf("n_comp=%d  top_k=%d  heads=%d  dim=%d  trials=%d\n\n",
           n_comp, top_k, N_HEAD, HEAD_DIM, trials);

    struct Dist { const char *name; int kind; double p; };
    const Dist dists[] = {
        {"gaussian  sigma=1",      0, 1.0},
        {"gaussian  sigma=0.25",   0, 0.25},
        {"heavy-tail t3",          1, 3.0},
        {"heavy-tail t1.5 (spiky)",1, 1.5},
        {"uniform   [-1,1]",       2, 1.0},
    };

    printf("keep vs EXACT (how far each is from ideal), then keep vs FP16 (what actually changes)\n\n");
    printf("%-24s | %7s %7s %7s | %-19s | %-19s\n", "Q distribution",
           "fp16", "E4M3", "E2M1", "E4M3 dropped rank", "E2M1 dropped rank");
    printf("%-24s-+-%7s-%7s-%7s-+-%-19s-+-%-19s\n", "------------------------",
           "-------", "-------", "-------", "-------------------", "-------------------");

    for (const Dist &D : dists) {
        double keep4 = 0, keep2 = 0, err4 = 0, err2 = 0;
        double keeph = 0, keep4h = 0, keep2h = 0;
        double mrank4 = 0, mrank2 = 0; int brank4 = 1 << 30, brank2 = 1 << 30;

        for (int t = 0; t < trials; t++) {
            std::mt19937_64 rng(1234567 + t * 7919 + D.kind * 104729 + (int)(D.p * 1000));
            std::normal_distribution<double> nd(0.0, 1.0);
            std::uniform_real_distribution<double> ud(-1.0, 1.0);
            std::chi_squared_distribution<double> cd(D.p);

            /* Q: one token, 64 heads x 128 dims. */
            std::vector<double> q((size_t)N_HEAD * HEAD_DIM);
            for (auto &v : q) {
                if (D.kind == 0)      v = nd(rng) * D.p;
                else if (D.kind == 1) v = nd(rng) / std::sqrt(cd(rng) / D.p);  /* student-t */
                else                  v = ud(rng);
            }

            /* K: already ON the E2M1 x E8M0 grid, which is the real constraint. */
            std::vector<double> k((size_t)n_comp * HEAD_DIM);
            for (int c = 0; c < n_comp; c++) {
                std::vector<double> raw(HEAD_DIM);
                for (auto &v : raw) v = nd(rng);
                quant_block(raw.data(), &k[(size_t)c * HEAD_DIM], HEAD_DIM, 2.0, quant_e2m1);
            }

            /* routing weights: positive, normalised */
            std::vector<double> w(N_HEAD);
            double wsum = 0.0;
            for (auto &v : w) { v = std::fabs(nd(rng)) + 1e-3; wsum += v; }
            for (auto &v : w) v /= wsum;

            /* reference, and the two candidates */
            std::vector<double> qe4(q.size()), qe2(q.size()), qh(q.size());
            for (int h = 0; h < N_HEAD; h++) {
                quant_block(&q[(size_t)h * HEAD_DIM], &qe4[(size_t)h * HEAD_DIM], HEAD_DIM, 8.0, quant_e4m3);
                quant_block(&q[(size_t)h * HEAD_DIM], &qe2[(size_t)h * HEAD_DIM], HEAD_DIM, 2.0, quant_e2m1);
            }
            for (size_t i = 0; i < q.size(); i++) qh[i] = quant_fp16(q[i]);   /* no block scale */

            std::vector<double> sref, s4, s2, sh;
            score_rows(q,   k, w, n_comp, sref);
            score_rows(qe4, k, w, n_comp, s4);
            score_rows(qe2, k, w, n_comp, s2);
            score_rows(qh,  k, w, n_comp, sh);

            const std::vector<int> tref = topk(sref, top_k);
            const std::vector<int> thalf = topk(sh, top_k);
            keep4 += (double)overlap(tref, topk(s4, top_k)) / top_k;
            keep2 += (double)overlap(tref, topk(s2, top_k)) / top_k;
            keeph += (double)overlap(tref, thalf) / top_k;
            /* the number that actually decides: change versus the DEPLOYED path */
            keep4h += (double)overlap(thalf, topk(s4, top_k)) / top_k;
            keep2h += (double)overlap(thalf, topk(s2, top_k)) / top_k;
            double mr; int br;
            dropped_ranks(sref, tref, topk(s4, top_k), top_k, &mr, &br);
            mrank4 += mr; if (br < brank4) brank4 = br;
            dropped_ranks(sref, tref, topk(s2, top_k), top_k, &mr, &br);
            mrank2 += mr; if (br < brank2) brank2 = br;

            double e4 = 0, e2 = 0, den = 0;
            for (int c = 0; c < n_comp; c++) {
                den += std::fabs(sref[c]);
                e4  += std::fabs(s4[c] - sref[c]);
                e2  += std::fabs(s2[c] - sref[c]);
            }
            err4 += e4 / (den > 0 ? den : 1);
            err2 += e2 / (den > 0 ? den : 1);
        }

        printf("%-24s | %6.2f%% %6.2f%% %6.2f%% | mean %6.1f best %4d | mean %6.1f best %4d\n",
               D.name,
               100.0 * keeph / trials, 100.0 * keep4 / trials, 100.0 * keep2 / trials,
               mrank4 / trials, brank4, mrank2 / trials, brank2);
    }

    printf("\nkeep columns  = fraction of the reference top-%d still selected\n", top_k);
    printf("dropped rank  = reference rank of the rows the candidate failed to select\n");
    printf("                (rank 0 = highest scoring; top_k = %d is the selection boundary)\n", top_k);
    printf("                'best' is the HIGHEST-ranked row ever dropped across all trials --\n");
    printf("                the worst case, and the number that decides whether churn is benign\n");
    printf("FIDELITY_DONE\n");
    return 0;
}

/* GQA attention block gate (L295 F1) -- model-free, links the PRODUCTION
 * object (src/cuda/pulsar_cuda_attn_gqa.o, the engine's NVCCFLAGS incl.
 * fast-math).  Per geometry (Inkling's global and window layers, a G = 12
 * gated layer, an MHA head_dim-64 window layer with a bias shorter than its
 * window):
 *
 *   1. fidelity   a one-call prefill (the folded schedule) against the double
 *                 host authority (tests/gqa_ref.h), every row: TOLERANCE-graded
 *                 (relative Frobenius and worst element of the f32 tap; the
 *                 bf16 output's share of elements equal to bf16(reference)).
 *   2. decode     one-row calls (the split schedule) == the prefill, BIT-EXACT:
 *                 bf16 output, f32 tap, the pool.
 *   3. chunking   the prompt cut at 1, 37, 64, 63, 200 rows == one call, bit-exact.
 *   4. verify     runs of 2..16 rows after a prefill == the prefill, bit-exact.
 *   5. rollback   a 6-row verify of junk rows, rolled back to its frontier and
 *                 re-run with the true rows == the prefill, bit-exact (on a
 *                 ring of window - 1 + 16 slots); on a ring of exactly window
 *                 slots the same rollback must NOT match (the slack is load-bearing).
 *   6. batching   16 decode rows of 16 sequences in one call == each alone, and
 *                 a 70-row prefill run + 5 decode rows (the folded schedule) ==
 *                 each alone, bit-exact.
 *   7. mutations  the reference without the bias, without the temperature, or
 *                 with the window one key wider must FAIL against the kernel
 *                 (>= 20x the bar), so the grade sees each feature.
 * Then a LONG POSITION case: Inkling's global layer on a pool of 131000
 * synthetic tokens (the temperature's 128000 floor crossed): a 64-row call ==
 * one-row decodes and a 4-row verify (bit-exact), graded against the
 * reference on three rows.  Then refusals.
 *
 * The fixture is live: an additive bias of ~N(0, 4) over logits of ~N(0, 0.1)
 * (RMS-normed q, k at scale 1/head_dim) gives peaked rows; the log prints the
 * reference's rms and the mean max attention weight. */
#include "cuda/pulsar_cuda_attn_gqa.h"
#include "gqa_ref.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static int g_fail = 0;
static void check(bool ok, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void check(bool ok, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    printf("  [%s] ", ok ? "ok" : "FAIL"); vprintf(fmt, ap); printf("\n");
    fflush(stdout);
    va_end(ap);
    if (!ok) g_fail = 1;
}

/* ---- deterministic inputs: (stream, seq id, position, index) -> N(0,1) ---------- */
static uint64_t mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(mix(seed + 0x9e3779b97f4a7c15ull)) {}
    double u() { s += 0x9e3779b97f4a7c15ull; return (double)(mix(s) >> 11) * (1.0 / 9007199254740992.0); }
    double n() { const double a = u() + 1e-12, b = u(); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }
};
static float bfn(Rng &g, double sd) { return gref_round((float)(g.n() * sd)); }
/* N(0, sd) on the grid k / q, |k| <= lim: r (q = 8, lim 32) and P (q = 16, lim 32) on these grids make
 * every r . P sum exact in f32 whatever the summation order (multiples of 1/128 below 2^7), so the bias's
 * bf16 rounding is the same on the tensor cores and in double: the grade then sees the kernel's arithmetic,
 * not rounding-boundary flips of an intermediate (the mini fixture grades real r and P). */
static float gridn(Rng &g, double sd, double q, double lim) {
    return (float)(std::max(-lim, std::min(lim, nearbyint(g.n() * sd * q))) / q);
}

/* ---- a case: one layer declaration, device weights, its host mirror -------------- */
struct Case {
    const char *name;
    pulsar_gqa_layer L{};
    gref_layer R{};
    uint32_t N = 0;          /* prompt length */
    uint32_t cap = 0;        /* pool slots */
    double in_sd = 1.0;      /* raw q / k scale */
    uint16_t *d_qw = nullptr, *d_kw = nullptr, *d_P = nullptr;
    uint64_t ld = 0, off_k = 0, off_v = 0, off_r = 0, off_g = 0;   /* the fused input row */
};

static void upload_bf(uint16_t **d, const std::vector<float> &h) {
    std::vector<uint16_t> b(h.size());
    for (size_t i = 0; i < h.size(); i++) b[i] = gref_f2bf(h[i]);
    CK(cudaMalloc(d, b.size() * 2));
    CK(cudaMemcpy(*d, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
}

static void case_init(Case &C, const char *name, uint32_t n_q, uint32_t n_kv, uint32_t hd, uint32_t window, uint32_t cap,
                      bool norm, uint32_t extent, float floor_, float alpha, bool gate, float scale, uint32_t N, uint64_t seed) {
    C.name = name; C.N = N; C.cap = cap;
    gref_layer &R = C.R;
    R.n_q = n_q; R.n_kv = n_kv; R.hd = hd; R.window = window; R.scale = scale; R.qk_norm = norm; R.eps = 1e-6f;
    R.extent = extent; R.floor_ = floor_; R.alpha = alpha; R.gate = gate;
    Rng g(seed);
    R.qw.resize(hd); R.kw.resize(hd);
    for (uint32_t i = 0; i < hd; i++) { R.qw[i] = gref_round((float)(1.0 + 0.25 * g.n())); R.kw[i] = gref_round((float)(1.0 + 0.25 * g.n())); }
    R.P.resize((size_t)16 * extent);
    for (auto &p : R.P) p = gridn(g, 0.5, 16, 32);
    upload_bf(&C.d_qw, R.qw); upload_bf(&C.d_kw, R.kw);
    if (extent) upload_bf(&C.d_P, R.P);
    pulsar_gqa_layer &L = C.L;
    L.n_q = n_q; L.n_kv = n_kv; L.head_dim = hd; L.kv_fmt = PULSAR_GQA_KV_BF16;
    L.span = window ? PULSAR_GQA_WINDOW : PULSAR_GQA_FULL; L.window = window;
    L.scale = scale; L.qk_norm = norm; L.norm_eps = 1e-6f;
    L.q_norm_w = norm ? C.d_qw : nullptr; L.k_norm_w = norm ? C.d_kw : nullptr;
    L.rel_extent = extent; L.rel_dim = extent ? 16u : 0u; L.rel_proj = C.d_P;
    L.temp_floor = floor_; L.temp_alpha = alpha; L.out_gate = gate;
    C.off_k = (uint64_t)n_q * hd; C.off_v = C.off_k + (uint64_t)n_kv * hd; C.off_r = C.off_v + (uint64_t)n_kv * hd;
    C.off_g = C.off_r + (uint64_t)n_q * 16u; C.ld = C.off_g + (uint64_t)n_q * hd + 8u;   /* a padded fused row */
}

/* the raw inputs of (seq id, pos): fused row [q | k | v | r | gate], bf16 values */
static void gen_row(const Case &C, uint64_t sid, uint32_t pos, float *row) {
    Rng g(mix(sid * 0x1000193ull + 77) ^ mix((uint64_t)pos * 0x5bd1e995ull + 13));
    for (uint64_t i = 0; i < C.off_r; i++) row[i] = bfn(g, C.in_sd * (i % 37 == 5 ? 6.0 : 1.0));   /* outlier channels */
    for (uint64_t i = C.off_r; i < C.off_g; i++) row[i] = gridn(g, 1.0, 8, 32);
    for (uint64_t i = C.off_g; i < C.ld; i++) row[i] = bfn(g, 2.0);
}

struct Seq { void *kv = nullptr; uint32_t cap = 0; uint64_t sid = 0; uint64_t bytes = 0; };
static Seq seq_new(const Case &C, uint64_t sid) {
    Seq s; s.cap = C.cap; s.sid = sid;
    s.bytes = (uint64_t)C.cap * pulsar_gqa_kv_token_bytes(&C.L);
    CK(cudaMalloc(&s.kv, s.bytes));
    CK(cudaMemset(s.kv, 0xff, s.bytes));   /* NaN bf16: a read of an unwritten slot poisons the output */
    return s;
}
static void seq_free(Seq &s) { CK(cudaFree(s.kv)); s.kv = nullptr; }

struct Out { std::vector<uint16_t> y; std::vector<float> tap; };   /* one row: n_q * hd */

static void *g_ws = nullptr;
static uint64_t g_ws_bytes = 0;
/* one call; `rows` = (index into seqs, pos); `junk` marks rows whose inputs come from a junk stream */
static int run_call(const Case &C, std::vector<Seq *> &seqs, const std::vector<std::pair<uint32_t, uint32_t>> &rows,
                    std::vector<Out> *out, bool junk = false) {
    const uint32_t n = (uint32_t)rows.size(), W = C.R.n_q * C.R.hd;
    std::vector<float> rowf(C.ld);
    std::vector<uint16_t> in((uint64_t)n * C.ld);
    std::vector<uint32_t> rs(n), rp(n);
    uint32_t max_ctx = 0;
    for (uint32_t i = 0; i < n; i++) {
        rs[i] = rows[i].first; rp[i] = rows[i].second;
        gen_row(C, seqs[rs[i]]->sid + (junk ? 0x777777ull : 0ull), rp[i], rowf.data());
        for (uint64_t j = 0; j < C.ld; j++) in[(uint64_t)i * C.ld + j] = gref_f2bf(rowf[j]);
        max_ctx = std::max(max_ctx, rp[i] + 1u);
    }
    const uint64_t need = pulsar_gqa_workspace_bytes(&C.L, n, max_ctx);
    if (need > g_ws_bytes) {
        if (g_ws) CK(cudaFree(g_ws));
        CK(cudaMalloc(&g_ws, need)); g_ws_bytes = need;
    }
    uint16_t *d_in; uint16_t *d_out; float *d_tap;
    CK(cudaMalloc(&d_in, in.size() * 2));
    CK(cudaMemcpy(d_in, in.data(), in.size() * 2, cudaMemcpyHostToDevice));
    const uint64_t ld_out = W + 2u;
    CK(cudaMalloc(&d_out, (uint64_t)n * ld_out * 2));
    CK(cudaMalloc(&d_tap, (uint64_t)n * W * 4));
    std::vector<pulsar_gqa_seq> ss(seqs.size());
    for (size_t i = 0; i < seqs.size(); i++) ss[i] = {seqs[i]->kv, seqs[i]->bytes, seqs[i]->cap};
    pulsar_gqa_io io{};
    io.q = {d_in, C.ld, C.R.hd};
    io.k = {d_in + C.off_k, C.ld, C.R.hd};
    io.v = {d_in + C.off_v, C.ld, C.R.hd};
    io.r = {d_in + C.off_r, C.ld, 16u};
    io.gate = {d_in + C.off_g, C.ld, C.R.hd};
    io.out_bf16 = d_out; io.ld_out = ld_out; io.tap_out_f32 = d_tap;
    io.workspace = g_ws; io.workspace_bytes = g_ws_bytes;
    const int rc = pulsar_gqa_forward(&C.L, ss.data(), (uint32_t)ss.size(), rs.data(), rp.data(), n, &io, 0);
    CK(cudaDeviceSynchronize());
    if (rc == 0 && out) {
        std::vector<uint16_t> yo((uint64_t)n * ld_out);
        out->resize(n);
        CK(cudaMemcpy(yo.data(), d_out, yo.size() * 2, cudaMemcpyDeviceToHost));
        std::vector<float> tp((uint64_t)n * W);
        CK(cudaMemcpy(tp.data(), d_tap, tp.size() * 4, cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < n; i++) {
            (*out)[i].y.assign(yo.begin() + (uint64_t)i * ld_out, yo.begin() + (uint64_t)i * ld_out + W);
            (*out)[i].tap.assign(tp.begin() + (uint64_t)i * W, tp.begin() + (uint64_t)(i + 1) * W);
        }
    }
    CK(cudaFree(d_in)); CK(cudaFree(d_out)); CK(cudaFree(d_tap));
    return rc;
}

/* a run of consecutive rows of one sequence in calls of the given sizes (the last repeats) */
static std::vector<Out> run_schedule(const Case &C, Seq &s, uint32_t p0, uint32_t p1, const std::vector<uint32_t> &sizes) {
    std::vector<Out> all;
    std::vector<Seq *> v{&s};
    uint32_t p = p0, k = 0;
    while (p < p1) {
        const uint32_t n = std::min(sizes[std::min<size_t>(k, sizes.size() - 1)], p1 - p);
        std::vector<std::pair<uint32_t, uint32_t>> rows;
        for (uint32_t i = 0; i < n; i++) rows.push_back({0u, p + i});
        std::vector<Out> o;
        if (run_call(C, v, rows, &o) != 0) { check(false, "%s: a call of %u rows at %u was refused", C.name, n, p); return all; }
        all.insert(all.end(), o.begin(), o.end());
        p += n; k++;
    }
    return all;
}

static bool same(const Out &a, const Out &b) {
    return a.y == b.y && a.tap.size() == b.tap.size() && !memcmp(a.tap.data(), b.tap.data(), a.tap.size() * 4);
}
static uint32_t first_diff(const std::vector<Out> &a, const std::vector<Out> &b, uint32_t off_b = 0) {
    for (uint32_t i = 0; i < b.size(); i++) if (!same(a[off_b + i], b[i])) return i;
    return UINT32_MAX;
}
static bool pools_equal(const Case &C, const Seq &a, const Seq &b, uint32_t upto) {
    const uint64_t tb = pulsar_gqa_kv_token_bytes(&C.L);
    const uint64_t bytes = (C.L.span == PULSAR_GQA_WINDOW ? (uint64_t)C.cap : (uint64_t)upto) * tb;
    std::vector<uint8_t> x(bytes), y(bytes);
    CK(cudaMemcpy(x.data(), a.kv, bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y.data(), b.kv, bytes, cudaMemcpyDeviceToHost));
    return x == y;
}

/* ---- the reference over a sequence ------------------------------------------------ */
struct History {
    std::vector<float> kh, vh;       /* [pos][n_kv][hd]: stored k (normed), v */
    std::vector<float> rows;         /* raw fused rows [pos][ld] */
};
static void history_build(const Case &C, uint64_t sid, uint32_t n, History &H) {
    const uint32_t nkv = C.R.n_kv, hd = C.R.hd;
    H.kh.assign((uint64_t)n * nkv * hd, 0.f); H.vh = H.kh;
    H.rows.assign((uint64_t)n * C.ld, 0.f);
    for (uint32_t p = 0; p < n; p++) {
        float *row = &H.rows[(uint64_t)p * C.ld];
        gen_row(C, sid, p, row);
        for (uint32_t g = 0; g < nkv; g++) {
            gref_norm(C.R, row + C.off_k + (uint64_t)g * hd, C.R.kw, &H.kh[((uint64_t)p * nkv + g) * hd]);
            memcpy(&H.vh[((uint64_t)p * nkv + g) * hd], row + C.off_v + (uint64_t)g * hd, hd * 4);
        }
    }
}
struct Grade { double fro, max, eq, rms; };
/* grade the kernel rows `got` (positions p_list) against the reference; parallel over rows */
static Grade grade(const Case &C, const History &H, const std::vector<uint32_t> &p_list, const std::vector<const Out *> &got,
                   const gref_mut &mu = gref_mut()) {
    const uint32_t W = C.R.n_q * C.R.hd, nr = (uint32_t)p_list.size();
    std::vector<double> ref((uint64_t)nr * W);
    const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; t++) th.emplace_back([&, t] {
        for (uint32_t i = t; i < nr; i += nt) {
            const float *row = &H.rows[(uint64_t)p_list[i] * C.ld];
            gref_row(C.R, p_list[i], H.kh.data(), H.vh.data(), row, row + C.off_r, row + C.off_g, &ref[(uint64_t)i * W], mu);
        }
    });
    for (auto &x : th) x.join();
    double ss = 0, se = 0, am = 0, em = 0, eq = 0;
    for (uint32_t i = 0; i < nr; i++) {
        for (uint32_t j = 0; j < W; j++) {
            const double r = ref[(uint64_t)i * W + j], d = fabs((double)got[i]->tap[j] - r);
            ss += r * r; se += d * d; am = std::max(am, fabs(r)); em = std::max(em, d);
            eq += got[i]->y[j] == gref_f2bf((float)r);
        }
    }
    Grade G;
    G.fro = sqrt(se / ss); G.max = em / am; G.eq = eq / ((double)nr * W); G.rms = sqrt(ss / ((double)nr * W));
    return G;
}

static const double BAR_FRO = 2e-5, BAR_MAX = 2e-4, BAR_EQ = 0.99;

/* the mean over rows / heads of the largest attention weight (fixture liveness) */
static double mean_peak(const Case &C, const History &H, const std::vector<uint32_t> &p_list) {
    double acc = 0; uint32_t cnt = 0;
    const uint32_t G = C.R.n_q / C.R.n_kv, hd = C.R.hd;
    for (size_t i = 0; i < p_list.size(); i += 7) {
        const uint32_t pos = p_list[i];
        const float *row = &H.rows[(uint64_t)pos * C.ld];
        const float tau = gref_tau(C.R, pos);
        const uint32_t lo = C.R.window && pos + 1u > C.R.window ? pos + 1u - C.R.window : 0u;
        std::vector<float> q(hd);
        for (uint32_t h = 0; h < C.R.n_q; h += 5) {
            gref_norm(C.R, row + (uint64_t)h * hd, C.R.qw, q.data());
            for (auto &e : q) e = gref_round(e * tau);
            std::vector<double> s;
            double mx = -INFINITY;
            for (uint32_t x = lo; x <= pos; x++) {
                double dot = 0;
                const float *k = &H.kh[((uint64_t)x * C.R.n_kv + h / G) * hd];
                for (uint32_t d = 0; d < hd; d++) dot += (double)q[d] * k[d];
                double b = 0;
                if (C.R.extent && pos - x < C.R.extent) {
                    double rp = 0;
                    for (int j = 0; j < 16; j++) rp += (double)row[C.off_r + h * 16 + j] * C.R.P[(uint64_t)j * C.R.extent + pos - x];
                    b = gref_round(gref_round((float)rp) * tau);
                }
                s.push_back(dot * C.R.scale + b); mx = std::max(mx, s.back());
            }
            double den = 0;
            for (double e : s) den += exp(e - mx);
            acc += 1.0 / den; cnt++;
        }
    }
    return acc / cnt;
}

static void run_case(Case &C) {
    printf("\n=== %s: %uq/%ukv/%u, %s %u (pool %u), rel bias extent %u, temperature floor %.0f, qk norm %s, gate %s, "
           "%u tokens\n", C.name, C.R.n_q, C.R.n_kv, C.R.hd, C.R.window ? "window" : "full, window", C.R.window, C.cap,
           C.R.extent, C.R.floor_, C.R.qk_norm ? "on" : "off", C.R.gate ? "on" : "off", C.N);
    const uint32_t N = C.N;
    History H;
    history_build(C, 1, N, H);

    /* 1. fidelity of a one-call prefill */
    Seq A = seq_new(C, 1);
    std::vector<Out> pre = run_schedule(C, A, 0, N, {N});
    if (pre.size() != N) { seq_free(A); return; }
    std::vector<uint32_t> all(N);
    std::vector<const Out *> gp(N);
    for (uint32_t p = 0; p < N; p++) { all[p] = p; gp[p] = &pre[p]; }
    const Grade G = grade(C, H, all, gp);
    printf("  reference rms %.4f, mean peak attention weight %.3f\n", G.rms, mean_peak(C, H, all));
    check(G.fro <= BAR_FRO && G.max <= BAR_MAX && G.eq >= BAR_EQ,
          "fidelity (prefill, %u rows, graded): fro %.2e (bar %.0e), max %.2e (bar %.0e), bf16 out == bf16(ref) %.4f (bar %.2f)",
          N, G.fro, BAR_FRO, G.max, BAR_MAX, G.eq, BAR_EQ);

    /* 2. decode */
    {
        Seq B = seq_new(C, 1);
        std::vector<Out> dec = run_schedule(C, B, 0, N, {1});
        const uint32_t d = first_diff(pre, dec);
        check(dec.size() == N && d == UINT32_MAX && pools_equal(C, A, B, N),
              "decode: %u one-row calls == the prefill (out, tap, pool), bit-exact%s", N,
              d == UINT32_MAX ? "" : " -- first difference at a row");
        seq_free(B);
    }
    /* 3. chunking */
    {
        Seq B = seq_new(C, 1);
        std::vector<Out> ch = run_schedule(C, B, 0, N, {1, 37, 64, 63, 200});
        check(ch.size() == N && first_diff(pre, ch) == UINT32_MAX && pools_equal(C, A, B, N),
              "chunking: calls of 1, 37, 64, 63, 200, 200, ... rows == one call, bit-exact");
        seq_free(B);
    }
    /* 4. verify runs */
    {
        Seq B = seq_new(C, 1);
        const uint32_t N0 = N - 2 - 3 - 4 - 5 - 6 - 16 - 1;
        std::vector<Out> v = run_schedule(C, B, 0, N0, {N0});
        bool ok = v.size() == N0;
        uint32_t p = N0;
        for (uint32_t k : {2u, 3u, 4u, 5u, 6u, 16u, 1u}) {
            std::vector<Out> o = run_schedule(C, B, p, p + k, {k});
            ok = ok && o.size() == k && first_diff(pre, o, p) == UINT32_MAX;
            p += k;
        }
        check(ok && pools_equal(C, A, B, N), "verify: runs of 2, 3, 4, 5, 6, 16 rows after a %u-row prefill == the prefill, bit-exact", N0);
        seq_free(B);
    }
    /* 5. rollback of a junk verify */
    if (C.R.window) {
        for (int pass = 0; pass < 2; pass++) {
            const uint32_t cap_save = C.cap;
            if (pass == 1) C.cap = C.R.window;
            Seq B = seq_new(C, 1);
            const uint32_t N0 = N - 40;
            run_schedule(C, B, 0, N0, {N0});
            std::vector<Seq *> v{&B};
            std::vector<std::pair<uint32_t, uint32_t>> rows;
            for (uint32_t i = 0; i < 6; i++) rows.push_back({0u, N0 + i});
            run_call(C, v, rows, nullptr, true);                     /* the junk verify ... */
            std::vector<Out> o = run_schedule(C, B, N0, N, {3, 1}); /* ... rolled back to N0 */
            const bool match = o.size() == N - N0 && first_diff(pre, o, N0) == UINT32_MAX;
            if (pass == 0) check(match, "rollback: a 6-row junk verify, rolled back, == the prefill (ring of %u = window - 1 + 16), bit-exact", C.cap);
            else check(!match, "rollback negative control: on a ring of exactly the window (%u) the same rollback differs", C.cap);
            seq_free(B);
            C.cap = cap_save;
        }
    }
    /* 6. batching: 16 decode rows of 16 sequences, then a 70-row run + 5 decode rows */
    {
        const uint32_t NS = 16;
        std::vector<Seq> X(NS), Y(NS);
        std::vector<uint32_t> len(NS);
        bool ok = true;
        for (uint32_t s = 0; s < NS; s++) {
            len[s] = 1u + (uint32_t)((s * 97u + 13u) % (N - 80u));
            X[s] = seq_new(C, 100 + s); Y[s] = seq_new(C, 100 + s);
            run_schedule(C, X[s], 0, len[s], {len[s]});
            run_schedule(C, Y[s], 0, len[s], {len[s]});
        }
        std::vector<Seq *> xs;
        std::vector<std::pair<uint32_t, uint32_t>> rows;
        for (uint32_t s = 0; s < NS; s++) { xs.push_back(&X[s]); rows.push_back({s, len[s]}); }
        std::vector<Out> bat;
        ok = run_call(C, xs, rows, &bat) == 0;
        for (uint32_t s = 0; s < NS && ok; s++) {
            std::vector<Out> one = run_schedule(C, Y[s], len[s], len[s] + 1, {1});
            ok = one.size() == 1 && same(one[0], bat[s]);
        }
        check(ok, "batching: 16 decode rows of 16 sequences in one call == each alone, bit-exact");
        /* the folded schedule: sequence 0 runs 70 rows, 1..5 one row each */
        rows.clear(); xs.clear();
        for (uint32_t s = 0; s < 6; s++) xs.push_back(&X[s]);
        for (uint32_t i = 0; i < 70; i++) rows.push_back({0u, len[0] + 1u + i});
        for (uint32_t s = 1; s < 6; s++) rows.push_back({s, len[s] + 1u});
        ok = run_call(C, xs, rows, &bat) == 0;
        std::vector<Out> r0 = run_schedule(C, Y[0], len[0] + 1u, len[0] + 71u, {7});
        for (uint32_t i = 0; i < 70 && ok; i++) ok = same(r0[i], bat[i]);
        for (uint32_t s = 1; s < 6 && ok; s++) {
            std::vector<Out> one = run_schedule(C, Y[s], len[s] + 1, len[s] + 2, {1});
            ok = one.size() == 1 && same(one[0], bat[70 + s - 1]);
        }
        check(ok, "batching: a 70-row run + 5 decode rows in one (folded) call == each alone in 7-row / 1-row calls, bit-exact");
        for (uint32_t s = 0; s < NS; s++) { seq_free(X[s]); seq_free(Y[s]); }
    }
    /* 7. mutations: each feature must be visible to the grade */
    {
        struct { const char *what; gref_mut mu; bool applies; } ms[3];
        ms[0] = {"no bias", gref_mut(), C.R.extent > 0}; ms[0].mu.drop_bias = true;
        ms[1] = {"no temperature", gref_mut(), C.R.floor_ > 0 && N > C.R.floor_}; ms[1].mu.drop_tau = true;
        ms[2] = {"window + 1", gref_mut(), C.R.window > 0}; ms[2].mu.window_delta = 1;
        for (auto &m : ms) {
            if (!m.applies) continue;
            const Grade M = grade(C, H, all, gp, m.mu);
            check(M.fro >= 20 * BAR_FRO || M.max >= 20 * BAR_MAX,
                  "mutation '%s' fails the grade: fro %.2e, max %.2e", m.what, M.fro, M.max);
        }
    }
    seq_free(A);
}

/* Inkling's global layer past the temperature floor: a pool of P0 synthetic tokens */
static void run_long(Case &C, uint32_t P0) {
    printf("\n=== %s: %uq/%ukv/%u full, rows at %u.. (temperature floor %.0f crossed)\n", C.name, C.R.n_q, C.R.n_kv, C.R.hd,
           P0, C.R.floor_);
    const uint32_t nkv = C.R.n_kv, hd = C.R.hd, n_new = 64;
    const uint64_t tb = pulsar_gqa_kv_token_bytes(&C.L);
    History H;
    H.kh.resize((uint64_t)(P0 + n_new) * nkv * hd); H.vh.resize(H.kh.size());
    H.rows.assign((uint64_t)(P0 + n_new) * C.ld, 0.f);
    std::vector<uint16_t> hist((uint64_t)P0 * tb / 2);
    {
        const unsigned nt = 16;
        std::vector<std::thread> th;
        for (unsigned t = 0; t < nt; t++) th.emplace_back([&, t] {
            for (uint32_t p = t; p < P0; p += nt) {
                Rng g(mix(0xabcdefull + p));
                for (uint32_t gg = 0; gg < nkv; gg++)
                    for (uint32_t i = 0; i < hd; i++) {
                        const float k = bfn(g, 1.0), v = bfn(g, 1.0);
                        H.kh[((uint64_t)p * nkv + gg) * hd + i] = k; H.vh[((uint64_t)p * nkv + gg) * hd + i] = v;
                        hist[(uint64_t)p * tb / 2 + (uint64_t)gg * hd + i] = gref_f2bf(k);
                        hist[(uint64_t)p * tb / 2 + (uint64_t)(nkv + gg) * hd + i] = gref_f2bf(v);
                    }
            }
        });
        for (auto &x : th) x.join();
    }
    for (uint32_t p = P0; p < P0 + n_new; p++) {
        float *row = &H.rows[(uint64_t)p * C.ld];
        gen_row(C, 1, p, row);
        for (uint32_t g = 0; g < nkv; g++) {
            gref_norm(C.R, row + C.off_k + (uint64_t)g * hd, C.R.kw, &H.kh[((uint64_t)p * nkv + g) * hd]);
            memcpy(&H.vh[((uint64_t)p * nkv + g) * hd], row + C.off_v + (uint64_t)g * hd, hd * 4);
        }
    }
    Seq A = seq_new(C, 1), B = seq_new(C, 1);
    CK(cudaMemcpy(A.kv, hist.data(), hist.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(B.kv, hist.data(), hist.size() * 2, cudaMemcpyHostToDevice));
    std::vector<Out> big = run_schedule(C, A, P0, P0 + n_new, {n_new});
    std::vector<Out> dec = run_schedule(C, B, P0, P0 + 4, {1});
    std::vector<Out> ver = run_schedule(C, B, P0 + 4, P0 + 8, {4});
    check(big.size() == n_new && first_diff(big, dec) == UINT32_MAX && first_diff(big, ver, 4) == UINT32_MAX,
          "long position: one-row decodes and a 4-row verify at %u.. == a 64-row (folded) call, bit-exact", P0);
    std::vector<uint32_t> pl{P0, P0 + 1, P0 + n_new - 1};
    std::vector<const Out *> gp{&big[0], &big[1], &big[n_new - 1]};
    const Grade G = grade(C, H, pl, gp);
    printf("  tau(%u) = %.6f, reference rms %.4f\n", P0 + n_new - 1, gref_tau(C.R, P0 + n_new - 1), G.rms);
    check(G.fro <= BAR_FRO && G.max <= BAR_MAX && G.eq >= BAR_EQ,
          "long position fidelity (3 rows, graded): fro %.2e, max %.2e, bf16 out == bf16(ref) %.4f", G.fro, G.max, G.eq);
    gref_mut mu; mu.drop_tau = true;
    const Grade M = grade(C, H, pl, gp, mu);
    check(M.fro >= 10 * BAR_FRO, "mutation 'no temperature' at long position fails the grade: fro %.2e", M.fro);
    seq_free(A); seq_free(B);
}

static void run_refusals(Case &C) {
    printf("\n=== refusals (each must return -1 and launch nothing)\n");
    Seq A = seq_new(C, 1);
    std::vector<Seq *> v{&A};
    auto expect = [&](const char *what, pulsar_gqa_layer L, std::vector<std::pair<uint32_t, uint32_t>> rows) {
        const pulsar_gqa_layer keep = C.L;
        C.L = L;
        const int rc = run_call(C, v, rows, nullptr);
        C.L = keep;
        check(rc == -1, "refused: %s (rc %d)", what, rc);
    };
    pulsar_gqa_layer L = C.L;
    L.n_kv = 3; expect("n_q not a multiple of n_kv", L, {{0, 0}});
    L = C.L; L.head_dim = 96; expect("head_dim 96 (no built tile)", L, {{0, 0}});
    L = C.L; L.rel_dim = 32; expect("rel_dim 32", L, {{0, 0}});
    L = C.L; L.qk_norm = 1; L.q_norm_w = nullptr; expect("qk_norm without its weight", L, {{0, 0}});
    L = C.L; L.window = 5000; L.span = PULSAR_GQA_WINDOW; expect("a ring smaller than the window", L, {{0, 0}});
    expect("rows not consecutive", C.L, {{0, 0}, {0, 2}});
    expect("a position past a full pool's cap", C.L, {{0, C.cap}});
    seq_free(A);
}

/* ---- --fixture DIR: the Inkling-Small mini checkpoint's attention (tests/gqa_mini_fixture.py) ---- *
 * Graded against transformers' InklingAttention in fp32 (ref32, from the same bf16 inputs) and
 * required to be at least as close to it as transformers' own bf16 run (ref16). */
static std::vector<uint8_t> slurp(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "gqa_attn_gate: cannot open %s\n", path.c_str()); exit(2); }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> b((size_t)n);
    if (fread(b.data(), 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "gqa_attn_gate: short read %s\n", path.c_str()); exit(2); }
    fclose(f);
    return b;
}
template <class T> static T *dup(const std::vector<uint8_t> &b) {
    T *d; CK(cudaMalloc(&d, b.size())); CK(cudaMemcpy(d, b.data(), b.size(), cudaMemcpyHostToDevice)); return d;
}

static void run_fixture_layer(const std::string &dir) {
    uint32_t T = 0, window = 0, extent = 0;
    float floor_ = 0, alpha = 0, eps = 0;
    {
        FILE *m = fopen((dir + "/meta.txt").c_str(), "r");
        if (!m) { check(false, "%s/meta.txt missing", dir.c_str()); return; }
        char key[32]; double val;
        while (fscanf(m, "%31s %lf", key, &val) == 2) {
            const std::string k = key;
            if (k == "tokens") T = (uint32_t)val; else if (k == "window") window = (uint32_t)val;
            else if (k == "extent") extent = (uint32_t)val; else if (k == "floor") floor_ = (float)val;
            else if (k == "alpha") alpha = (float)val; else if (k == "eps") eps = (float)val;
        }
        fclose(m);
    }
    const uint32_t NQ = 32, NKV = 8, HD = 128, W = NQ * HD, LD = NQ * HD + 2 * NKV * HD + NQ * 16;
    const std::vector<uint8_t> in = slurp(dir + "/in.bin"), qn = slurp(dir + "/qn.bin"), kn = slurp(dir + "/kn.bin"),
                               P = slurp(dir + "/P.bin"), r32 = slurp(dir + "/ref32.bin"), r16 = slurp(dir + "/ref16.bin");
    if (in.size() != (size_t)T * LD * 2 || P.size() != (size_t)16 * extent * 2 || r32.size() != (size_t)T * W * 4 ||
        r16.size() != r32.size()) {
        check(false, "%s: file sizes disagree with meta.txt", dir.c_str());
        return;
    }
    pulsar_gqa_layer L{};
    L.n_q = NQ; L.n_kv = NKV; L.head_dim = HD; L.kv_fmt = PULSAR_GQA_KV_BF16;
    L.span = window ? PULSAR_GQA_WINDOW : PULSAR_GQA_FULL; L.window = window; L.scale = 1.0f / HD;
    L.qk_norm = 1; L.norm_eps = eps; L.q_norm_w = dup<uint16_t>(qn); L.k_norm_w = dup<uint16_t>(kn);
    L.rel_extent = extent; L.rel_dim = 16; L.rel_proj = dup<uint16_t>(P);
    L.temp_floor = floor_; L.temp_alpha = alpha;
    uint16_t *d_in = dup<uint16_t>(in);
    const uint32_t cap = window ? window + 15u : T;
    auto run = [&](uint32_t chunk, std::vector<uint16_t> &y, std::vector<float> &tap) {
        void *kv;
        const uint64_t kvb = (uint64_t)cap * pulsar_gqa_kv_token_bytes(&L);
        CK(cudaMalloc(&kv, kvb)); CK(cudaMemset(kv, 0xff, kvb));
        uint16_t *d_out; float *d_tap;
        CK(cudaMalloc(&d_out, (uint64_t)T * W * 2)); CK(cudaMalloc(&d_tap, (uint64_t)T * W * 4));
        const uint64_t wsb = pulsar_gqa_workspace_bytes(&L, chunk, T);
        void *ws; CK(cudaMalloc(&ws, wsb));
        pulsar_gqa_seq sq{kv, kvb, cap};
        int rc = 0;
        for (uint32_t p = 0; p < T && rc == 0; p += chunk) {
            const uint32_t n = std::min(chunk, T - p);
            std::vector<uint32_t> rs(n, 0u), rp(n);
            for (uint32_t i = 0; i < n; i++) rp[i] = p + i;
            pulsar_gqa_io io{};
            const uint16_t *base = d_in + (uint64_t)p * LD;
            io.q = {base, LD, HD}; io.k = {base + NQ * HD, LD, HD}; io.v = {base + NQ * HD + NKV * HD, LD, HD};
            io.r = {base + NQ * HD + 2 * NKV * HD, LD, 16};
            io.out_bf16 = d_out + (uint64_t)p * W; io.ld_out = W; io.tap_out_f32 = d_tap + (uint64_t)p * W;
            io.workspace = ws; io.workspace_bytes = wsb;
            rc = pulsar_gqa_forward(&L, &sq, 1, rs.data(), rp.data(), n, &io, 0);
        }
        CK(cudaDeviceSynchronize());
        y.resize((size_t)T * W); tap.resize((size_t)T * W);
        CK(cudaMemcpy(y.data(), d_out, y.size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(tap.data(), d_tap, tap.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaFree(kv)); CK(cudaFree(d_out)); CK(cudaFree(d_tap)); CK(cudaFree(ws));
        return rc;
    };
    std::vector<uint16_t> y1, y2;
    std::vector<float> t1, t2;
    const int rc1 = run(T, y1, t1), rc2 = run(1, y2, t2);
    check(rc1 == 0 && rc2 == 0 && y1 == y2 && !memcmp(t1.data(), t2.data(), t1.size() * 4),
          "%s: one %u-row call == %u one-row calls, bit-exact", dir.c_str(), T, T);
    const float *a = (const float *)r32.data(), *b = (const float *)r16.data();
    double ss = 0, se = 0, s16 = 0, am = 0, em = 0, eq = 0, eqo = 0;
    for (size_t i = 0; i < (size_t)T * W; i++) {
        const double r = a[i], d = t1[i] - r, d16 = b[i] - r;
        ss += r * r; se += d * d; s16 += d16 * d16; am = std::max(am, fabs(r)); em = std::max(em, fabs(d));
        eq += y1[i] == gref_f2bf(a[i]);
        eqo += gref_f2bf(b[i]) == gref_f2bf(a[i]);
    }
    const double fro = sqrt(se / ss), fro16 = sqrt(s16 / ss);
    printf("  %s: %u tokens, %s, extent %u, temperature floor %.0f; rms(ref32) %.4f\n", dir.c_str(), T,
           window ? "window" : "full", extent, floor_, sqrt(ss / ((double)T * W)));
    check(fro <= fro16 && fro <= 1e-2,
          "%s (graded vs transformers fp32): block fro %.3e, max %.3e, bf16 out == bf16(ref32) %.4f; transformers "
          "bf16 fro %.3e (== %.4f): the block is %.1fx closer",
          dir.c_str(), fro, em / am, eq / ((double)T * W), fro16, eqo / ((double)T * W), fro16 / fro);
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--fixture")) {
        printf("gqa_attn_gate --fixture %s: the Inkling-Small mini checkpoint's attention vs transformers\n", argv[2]);
        run_fixture_layer(std::string(argv[2]) + "/layer0");
        run_fixture_layer(std::string(argv[2]) + "/layer2");
        printf("\ngqa_attn_gate --fixture: %s\n", g_fail ? "FAIL" : "PASS");
        return g_fail ? 1 : 0;
    }
    int dev = 0;
    CK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    printf("gqa_attn_gate: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("bars (graded vs the double host reference): fro %.0e, max %.0e, bf16 out == bf16(ref) >= %.2f\n", BAR_FRO, BAR_MAX, BAR_EQ);

    const uint32_t N = 1300;
    Case g1, g2, g3, g4;
    /* Inkling global: 32q/8kv/128, full, relative bias over 1024, the temperature (128000, 0.1), scale 1/128 */
    case_init(g1, "inkling-global", 32, 8, 128, 0, N + 64, true, 1024, 128000.f, 0.1f, false, 1.0f / 128, N, 11);
    /* Inkling window: 512 keys, bias over 512, no temperature; ring = window - 1 + 16 */
    case_init(g2, "inkling-window", 32, 8, 128, 512, 512 + 15, true, 512, 0.f, 0.f, false, 1.0f / 128, N, 22);
    /* a QSA-shaped group (G = 12) with the sigmoid output gate, no norm, a strong temperature, scale 1/sqrt(hd) */
    case_init(g3, "group12-gated", 24, 2, 128, 0, 400, false, 0, 64.f, 0.3f, true, 1.0f / sqrtf(128.f), 300, 33);
    /* MHA (G = 1), head_dim 64, window 100 with a bias over only 64 distances, a strong temperature */
    case_init(g4, "mha64-window", 4, 4, 64, 100, 100 + 15, true, 64, 32.f, 0.2f, false, 1.0f / 64, 300, 44);
    run_case(g1);
    run_case(g2);
    run_case(g3);
    run_case(g4);

    Case gl;
    const uint32_t P0 = 131000;
    case_init(gl, "inkling-global-long", 32, 8, 128, 0, P0 + 64, true, 1024, 128000.f, 0.1f, false, 1.0f / 128, P0 + 64, 55);
    run_long(gl, P0);

    run_refusals(g1);

    printf("\ngqa_attn_gate: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}

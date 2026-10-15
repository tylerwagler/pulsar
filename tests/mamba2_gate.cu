/* Mamba-2 mixer block gate (L296 N2) -- model-free, links the PRODUCTION objects (src/cuda/pulsar_cuda_mamba2.o
 * and the E3 conv it runs, the engine's NVCCFLAGS incl. fast-math).  Per geometry (Nemotron 3.5 Super's layer,
 * one TP=2 rank of it, and a small odd one):
 *
 *   1. fidelity   a one-call prefill against the DOUBLE reference (tests/mamba2_ref.h, the block's bf16 rounding
 *                 points mirrored): relative Frobenius, the share of outputs equal to bf16(reference), the worst
 *                 bf16 ulp; the f32 SSM state against the double state; the conv slot == the last K-1 inputs.
 *   2. decode     one-row calls == the prefill, BIT-EXACT (out, conv slot, SSM slot).
 *   3. chunking   the prompt cut at 1, 2, 3, 5, 16, 17, 37, 64, 15 rows == one call.
 *   4. verify     runs of 2..16 rows with both captures, each committed whole (the DELTA commit + the SNAPSHOT
 *                 record copy) == the prefill; the verify leaves the SSM slot as it found it; every conv record
 *                 == the state after its row.
 *   5. rollback   a 6-row verify whose rows after j are junk, committed to keep j + 1 rows (j = 0 .. 5) and
 *                 decoded on == the prefill; committing all 6, or skipping the conv record copy, must differ.
 *   6. batching   16 sequences on permuted slots: ragged prefill runs (one of 70 rows), then a mixed decode +
 *                 multi-row call, then a captured verify of every sequence in ONE call committed to a different
 *                 keep each == each sequence alone, bit-exact.
 *   7. mutations  the double reference without D, without the gate, with the norm group halved or whole, with
 *                 the B/C group map wrong, or without the conv bias must FAIL the grade (>= 10x the bar).
 * Then refusals.
 *
 * --fixture DIR grades the block on real Nemotron 3.5 Super layers (tests/mamba2_mini_fixture.py): their
 * in_proj outputs and parameters against transformers' NemotronHMamba2Mixer in fp32 (dt unclamped, as vLLM).  The
 * distance splits into the declared bf16 rounding points (the double reference with them, vs fp32) and the block's
 * own arithmetic (the block vs that double): the first must account for the block's distance to within 10%, the
 * second must meet the synthetic bars; transformers' own bf16 run is printed beside them.  Decode and chunked
 * schedules on the real rows must equal the one-call prefill bit for bit. */
#include "cuda/pulsar_cuda_mamba2.h"
#include "mamba2_ref.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
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

/* The fidelity bars: the block's bf16 output vs the double reference with the same rounding points.  The output's
 * own rounding is ~1e-3 relative; an intermediate bf16 flip (the f32 path landing on the other side of a rounding
 * boundary than the double) moves an element by one ulp. */
static const double kFroBar = 3e-3;
static const double kExactShareBar = 0.99;
static const double kStateBar = 1e-4;    /* the f32 SSM state vs the double state */

static double fro_rel(const std::vector<double> &a, const std::vector<double> &ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double d = a[i] - ref[i]; num += d * d; den += ref[i] * ref[i]; }
    return sqrt(num / std::max(den, 1e-300));
}
static int bf_ulps(uint16_t a, uint16_t b) {
    auto ord = [](uint16_t v) { return (v & 0x8000u) ? -(int)(v & 0x7fffu) : (int)v; };
    return abs(ord(a) - ord(b));
}
struct Grade { double fro = 0, exact_share = 0; int max_ulp = 0; };
static Grade grade(const std::vector<uint16_t> &y, const std::vector<double> &ref) {
    Grade g;
    std::vector<double> yd(y.size());
    size_t eq = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        yd[i] = (double)sref_bf2f(y[i]);
        const uint16_t rb = sref_f2bf((float)ref[i]);
        eq += rb == y[i];
        g.max_ulp = std::max(g.max_ulp, bf_ulps(rb, y[i]));
    }
    g.fro = fro_rel(yd, ref);
    g.exact_share = (double)eq / (double)std::max<size_t>(y.size(), 1);
    return g;
}
static bool same16(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * 2) == 0;
}
static bool same32(const std::vector<float> &a, const std::vector<float> &b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

/* ---- a geometry under test, its sequences' in_proj rows and the device pools ------------------------- */
static const uint32_t kSlots = 32, kSeqs = 16, kMaxRows = 512;

struct Case {
    const char *name;
    m2ref_layer R;
    pulsar_mamba2_layer L{};
    uint32_t Nrows = 0, HP = 0, CC = 0, ROW = 0;  /* ROW: the in_proj row [z | xbc | dt], padded to 8 */
    uint64_t rec = 0, ssmf = 0, updf = 0;
    std::vector<std::vector<uint16_t>> X;         /* per sequence: [Nrows][ROW] bf16 */
    uint16_t *d_par = nullptr;                    /* the parameters, one allocation */
    uint16_t *d_conv = nullptr; float *d_ssm = nullptr;
    uint16_t *d_in = nullptr, *d_out = nullptr, *d_ccap = nullptr;
    float *d_scap = nullptr;
    void *d_scr = nullptr; size_t scr_bytes = 0;
    int32_t *d_first = nullptr, *d_slot = nullptr;
};

static void case_init(Case &c, const char *name, int H, int G, int NG, int K, uint32_t Nrows, uint64_t seed) {
    c.name = name; c.Nrows = Nrows;
    m2ref_layer &R = c.R;
    R.H = H; R.P = 64; R.N = 128; R.G = G; R.K = K; R.NG = NG; R.eps = (double)1e-5f;
    c.HP = (uint32_t)(H * 64); c.CC = (uint32_t)R.CC(); c.ROW = (c.HP + c.CC + (uint32_t)H + 7u) / 8u * 8u;
    Rng g(seed);
    auto bfv = [&](size_t n, double mu, double sd) {
        std::vector<float> v(n);
        for (auto &x : v) x = sref_round((float)(mu + sd * g.n()));
        return v;
    };
    R.conv_w = bfv((size_t)c.CC * K, 0.0, 0.35);
    R.conv_b = bfv(c.CC, 0.0, 0.2);
    R.A_log.resize(H); R.dt_bias.resize(H);
    for (int h = 0; h < H; ++h) {
        R.A_log[(size_t)h] = sref_round((float)(2.5 * g.u()));          /* A in [-12, -1] */
        R.dt_bias[(size_t)h] = sref_round((float)(-5.0 + 4.0 * g.u())); /* dt ~ 0.007 .. 0.3 */
    }
    R.D = bfv(H, 1.0, 0.3);
    R.norm_w = bfv(c.HP, 1.0, 0.2);
    /* one device block: conv_w | conv_b | A_log | dt_bias | D | norm_w, each 16-byte aligned */
    const std::vector<float> *parts[6] = {&R.conv_w, &R.conv_b, &R.A_log, &R.dt_bias, &R.D, &R.norm_w};
    size_t off[6], tot = 0;
    for (int i = 0; i < 6; ++i) { off[i] = tot; tot += (parts[i]->size() + 7) / 8 * 8; }
    std::vector<uint16_t> hp(tot, 0);
    for (int i = 0; i < 6; ++i)
        for (size_t k = 0; k < parts[i]->size(); ++k) hp[off[i] + k] = sref_f2bf((*parts[i])[k]);
    CK(cudaMalloc(&c.d_par, tot * 2));
    CK(cudaMemcpy(c.d_par, hp.data(), tot * 2, cudaMemcpyHostToDevice));
    pulsar_mamba2_layer &L = c.L;
    L.geom = {(uint32_t)H, 64, 128, (uint32_t)G, (uint32_t)K, (uint32_t)NG, 1e-5f};
    L.conv_w = c.d_par + off[0]; L.conv_b = c.d_par + off[1]; L.A_log = c.d_par + off[2];
    L.dt_bias = c.d_par + off[3]; L.D = c.d_par + off[4]; L.norm_w = c.d_par + off[5];
    c.rec = pulsar_mamba2_conv_state_elems(&L.geom);
    c.ssmf = pulsar_mamba2_ssm_state_floats(&L.geom);
    c.updf = pulsar_mamba2_upd_floats(&L.geom);
    /* inputs: z ~ N(0, 1); x scaled per head (0.25 .. 4, so norm groups differ in scale); B, C ~ N(0, 1);
     * dt ~ N(0, 0.7) */
    c.X.resize(kSeqs);
    std::vector<double> hs((size_t)H);
    for (int h = 0; h < H; ++h) hs[(size_t)h] = pow(2.0, -2.0 + 4.0 * (double)((h * 7) % H) / H);
    for (uint32_t q = 0; q < kSeqs; ++q) {
        Rng gq(seed * 131 + q + 1);
        c.X[q].assign((size_t)Nrows * c.ROW, 0);
        for (uint32_t t = 0; t < Nrows; ++t) {
            uint16_t *r = &c.X[q][(size_t)t * c.ROW];
            for (uint32_t k = 0; k < c.HP; ++k) r[k] = sref_f2bf((float)gq.n());
            for (uint32_t k = 0; k < c.CC; ++k)
                r[c.HP + k] = sref_f2bf((float)(gq.n() * (k < c.HP ? hs[k / 64] : 1.0)));
            for (int h = 0; h < H; ++h) r[c.HP + c.CC + h] = sref_f2bf((float)(0.7 * gq.n()));
        }
    }
    CK(cudaMalloc(&c.d_conv, kSlots * c.rec * 2));
    CK(cudaMalloc(&c.d_ssm, kSlots * c.ssmf * 4));
    CK(cudaMalloc(&c.d_in, (size_t)kMaxRows * c.ROW * 2));
    CK(cudaMalloc(&c.d_out, (size_t)kMaxRows * c.HP * 2));
    CK(cudaMalloc(&c.d_ccap, (size_t)kMaxRows * c.rec * 2));
    CK(cudaMalloc(&c.d_scap, (size_t)kMaxRows * c.updf * 4));
    c.scr_bytes = pulsar_mamba2_scratch_bytes(&L.geom, kMaxRows);
    CK(cudaMalloc(&c.d_scr, c.scr_bytes));
    CK(cudaMalloc(&c.d_first, (kMaxRows + 1) * 4));
    CK(cudaMalloc(&c.d_slot, kMaxRows * 4));
}
static void case_free(Case &c) {
    cudaFree(c.d_par); cudaFree(c.d_conv); cudaFree(c.d_ssm); cudaFree(c.d_in); cudaFree(c.d_out);
    cudaFree(c.d_ccap); cudaFree(c.d_scap); cudaFree(c.d_scr); cudaFree(c.d_first); cudaFree(c.d_slot);
}

static void zero_slot(Case &c, uint32_t slot) {
    CK(cudaMemset(c.d_conv + slot * c.rec, 0, c.rec * 2));
    CK(cudaMemset(c.d_ssm + slot * c.ssmf, 0, c.ssmf * 4));
}
static std::vector<uint16_t> get_conv(Case &c, uint32_t slot) {
    std::vector<uint16_t> h(c.rec);
    CK(cudaMemcpy(h.data(), c.d_conv + slot * c.rec, c.rec * 2, cudaMemcpyDeviceToHost));
    return h;
}
static std::vector<float> get_ssm(Case &c, uint32_t slot) {
    std::vector<float> h(c.ssmf);
    CK(cudaMemcpy(h.data(), c.d_ssm + slot * c.ssmf, c.ssmf * 4, cudaMemcpyDeviceToHost));
    return h;
}
struct Slots { std::vector<uint16_t> conv; std::vector<float> ssm; };
static Slots get_slots(Case &c, uint32_t slot) { return {get_conv(c, slot), get_ssm(c, slot)}; }
static bool same_slots(const Slots &a, const Slots &b) { return same16(a.conv, b.conv) && same32(a.ssm, b.ssm); }

/* one run of a call: sequence q's rows [start, start + count) on `slot`; rows from index junk_from of the run take
 * another sequence's inputs instead */
struct Run { uint32_t q, slot, start, count, junk_from = ~0u; };

/* Runs the call; out holds call row r's output ([rows][HP]); captures when asked. */
static int call(Case &c, const std::vector<Run> &runs, std::vector<uint16_t> *out, bool capture) {
    std::vector<int32_t> first(1, 0), slot;
    std::vector<uint16_t> hx;
    uint32_t longest = 0;
    for (const Run &r : runs) {
        for (uint32_t i = 0; i < r.count; ++i) {
            const uint32_t src = i >= r.junk_from ? (r.q + 7) % kSeqs : r.q;
            const uint16_t *row = &c.X[src][(size_t)(r.start + i) * c.ROW];
            hx.insert(hx.end(), row, row + c.ROW);
        }
        first.push_back(first.back() + (int32_t)r.count);
        slot.push_back((int32_t)r.slot);
        longest = std::max(longest, r.count);
    }
    const uint32_t rows = (uint32_t)first.back();
    CK(cudaMemcpy(c.d_in, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c.d_first, first.data(), first.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c.d_slot, slot.data(), slot.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(c.d_out, 0xff, (size_t)rows * c.HP * 2));
    pulsar_mamba2_call k{};
    k.n_seqs = (uint32_t)runs.size(); k.n_rows = rows; k.max_seq_rows = longest;
    k.seq_first = c.d_first; k.seq_slot = c.d_slot;
    k.z = c.d_in; k.ld_z = c.ROW;
    k.xbc = c.d_in + c.HP; k.ld_xbc = c.ROW;
    k.dt = c.d_in + c.HP + c.CC; k.ld_dt = c.ROW;
    k.conv_state = c.d_conv; k.ssm_state = c.d_ssm;
    k.conv_capture = capture ? c.d_ccap : nullptr;
    k.ssm_capture = capture ? c.d_scap : nullptr;
    k.scratch = c.d_scr; k.scratch_bytes = c.scr_bytes;
    k.out = c.d_out; k.ld_out = c.HP;
    const int rc = pulsar_mamba2_forward(&c.L, &k, 0);
    if (rc) return rc;
    CK(cudaDeviceSynchronize());
    out->resize((size_t)rows * c.HP);
    CK(cudaMemcpy(out->data(), c.d_out, out->size() * 2, cudaMemcpyDeviceToHost));
    return 0;
}

/* the commit a speculative step makes for one run of a captured call: the SSM slot advanced by the run's first
 * `keep` rows (DELTA), the conv slot set to the record of its row keep - 1 (SNAPSHOT) */
static bool commit(Case &c, uint32_t slot, uint32_t run_first, uint32_t keep, bool conv_copy = true) {
    if (pulsar_mamba2_commit(&c.L.geom, c.d_ssm, slot, c.d_scap + (size_t)run_first * c.updf, keep, 0) != 0) return false;
    if (conv_copy)
        CK(cudaMemcpy(c.d_conv + slot * c.rec, c.d_ccap + (size_t)(run_first + keep - 1) * c.rec, c.rec * 2,
                      cudaMemcpyDeviceToDevice));
    CK(cudaDeviceSynchronize());
    return true;
}

/* sequence q's outputs for positions [0, Nrows), assembled from a schedule of calls */
struct Seqout {
    std::vector<uint16_t> y;
    void put(const Case &c, uint32_t pos, const uint16_t *row) {
        if (y.empty()) y.assign((size_t)c.Nrows * c.HP, 0);
        memcpy(&y[(size_t)pos * c.HP], row, (size_t)c.HP * 2);
    }
};

/* the conv state after position p (inclusive) of sequence q: its inputs at p - (K-2) .. p */
static std::vector<uint16_t> conv_after(const Case &c, uint32_t q, int64_t p) {
    const int K = c.R.K;
    std::vector<uint16_t> s(c.rec, 0);
    for (int i = 0; i < K - 1; ++i) {
        const int64_t pos = p - (K - 2) + i;
        if (pos >= 0) memcpy(&s[(size_t)i * c.CC], &c.X[q][(size_t)pos * c.ROW + c.HP], (size_t)c.CC * 2);
    }
    return s;
}

static void reference(const Case &c, uint32_t q, uint32_t T, m2ref_mut mut, std::vector<double> *out,
                      std::vector<double> *ssm_out = nullptr) {
    std::vector<float> f((size_t)T * c.ROW);
    for (size_t i = 0; i < f.size(); ++i) f[i] = sref_bf2f(c.X[q][i]);
    std::vector<double> conv(c.rec, 0.0), ssm(c.ssmf, 0.0);
    out->assign((size_t)T * c.HP, 0.0);
    m2ref_seq(c.R, (int)T, f.data(), c.ROW, f.data() + c.HP, c.ROW, f.data() + c.HP + c.CC, c.ROW, conv.data(), ssm.data(),
              out->data(), mut);
    if (ssm_out) *ssm_out = ssm;
}

static void run_case(Case &c) {
    const uint32_t N = c.Nrows, HP = c.HP;
    printf("\n== %s: H %d x P 64, N 128, G %d, conv K %d over %u ch, norm group %d, %u rows ==\n", c.name, c.R.H, c.R.G,
           c.R.K, c.CC, c.R.NG, N);

    /* 1: the one-call prefill vs the double */
    std::vector<double> ref, ref_ssm;
    reference(c, 0, N, M2_EXACT, &ref, &ref_ssm);
    std::vector<uint16_t> pf;
    zero_slot(c, 3);
    check(call(c, {{0, 3, 0, N}}, &pf, false) == 0, "prefill call");
    const Slots pfs = get_slots(c, 3);
    const Grade g = grade(pf, ref);
    double rms = 0;
    for (double v : ref) rms += v * v;
    rms = sqrt(rms / (double)ref.size());
    check(g.fro <= kFroBar && g.exact_share >= kExactShareBar,
          "fidelity vs double: fro %.3e (bar %.0e), == bf16(ref) %.4f%% (bar %.0f%%), worst %d ulp (rms(ref) %.3f)", g.fro,
          kFroBar, 100.0 * g.exact_share, 100.0 * kExactShareBar, g.max_ulp, rms);
    {
        std::vector<double> sd(pfs.ssm.begin(), pfs.ssm.end());
        const double sf = fro_rel(sd, ref_ssm);
        check(sf <= kStateBar, "SSM state after %u rows vs double: fro %.3e (bar %.0e)", N, sf, kStateBar);
    }
    check(same16(pfs.conv, conv_after(c, 0, N - 1)), "conv slot == the last %d inputs", c.R.K - 1);

    /* 2: decode */
    {
        zero_slot(c, 5);
        Seqout so;
        std::vector<uint16_t> o;
        bool ok = true;
        for (uint32_t p = 0; p < N && ok; ++p) {
            ok = call(c, {{0, 5, p, 1}}, &o, false) == 0;
            if (ok) so.put(c, p, o.data());
        }
        check(ok && same16(so.y, pf) && same_slots(get_slots(c, 5), pfs), "decode: %u one-row calls == the prefill (out, conv, ssm)", N);
    }

    /* 3: chunked */
    {
        zero_slot(c, 6);
        const uint32_t cuts[] = {1, 2, 3, 5, 16, 17, 37, 64, 15};
        Seqout so;
        std::vector<uint16_t> o;
        uint32_t p = 0, k = 0;
        bool ok = true;
        while (p < N && ok) {
            const uint32_t n = std::min(N - p, cuts[k++ % 9]);
            ok = call(c, {{0, 6, p, n}}, &o, false) == 0;
            for (uint32_t r = 0; ok && r < n; ++r) so.put(c, p + r, &o[(size_t)r * HP]);
            p += n;
        }
        check(ok && same16(so.y, pf) && same_slots(get_slots(c, 6), pfs), "chunking: cut at 1/2/3/5/16/17/37/64/15 == one call");
    }

    /* 4: verify runs with both captures, committed whole */
    {
        zero_slot(c, 7);
        Seqout so;
        std::vector<uint16_t> o;
        const uint32_t P0 = std::min<uint32_t>(40, N);
        bool ok = call(c, {{0, 7, 0, P0}}, &o, false) == 0;
        for (uint32_t r = 0; ok && r < P0; ++r) so.put(c, r, &o[(size_t)r * HP]);
        uint32_t p = P0, k = 2;
        bool caps = true, untouched = true;
        while (p < N && ok) {
            const uint32_t n = std::min(N - p, k);
            k = k == 16 ? 2 : k + 1;
            const std::vector<float> before = get_ssm(c, 7);
            ok = call(c, {{0, 7, p, n}}, &o, true) == 0;
            untouched = untouched && same32(get_ssm(c, 7), before);
            for (uint32_t r = 0; ok && r < n; ++r) {
                so.put(c, p + r, &o[(size_t)r * HP]);
                std::vector<uint16_t> rec(c.rec);
                CK(cudaMemcpy(rec.data(), c.d_ccap + (size_t)r * c.rec, c.rec * 2, cudaMemcpyDeviceToHost));
                caps = caps && same16(rec, conv_after(c, 0, p + r));
            }
            ok = ok && commit(c, 7, 0, n);
            p += n;
        }
        check(ok && same16(so.y, pf) && same_slots(get_slots(c, 7), pfs),
              "verify: runs of 2..16 rows after a %u-row prefill, each committed whole == the prefill", P0);
        check(ok && untouched, "verify: a captured call leaves the SSM slot as it found it");
        check(ok && caps, "verify: every conv record == the state after its row");
    }

    /* 5: rollback to any accepted row */
    if (N >= 60) {
        const uint32_t P0 = 40;
        auto attempt = [&](uint32_t j, uint32_t keep, bool conv_copy) {
            const uint32_t slot = 8 + (j % 8);
            zero_slot(c, slot);
            Seqout so;
            std::vector<uint16_t> o;
            bool ok = call(c, {{0, slot, 0, P0}}, &o, false) == 0;
            for (uint32_t r = 0; ok && r < P0; ++r) so.put(c, r, &o[(size_t)r * HP]);
            Run v{0, slot, P0, 6, j + 1};   /* rows 0..j true, the rest junk */
            ok = ok && call(c, {v}, &o, true) == 0;
            for (uint32_t r = 0; ok && r <= j; ++r) so.put(c, P0 + r, &o[(size_t)r * HP]);
            ok = ok && commit(c, slot, 0, keep, conv_copy);
            for (uint32_t p = P0 + j + 1; ok && p < N; ++p) {
                ok = call(c, {{0, slot, p, 1}}, &o, false) == 0;
                so.put(c, p, o.data());
            }
            return std::make_pair(ok, ok && same16(so.y, pf) && same_slots(get_slots(c, slot), pfs));
        };
        bool all = true;
        for (uint32_t j = 0; j < 6; ++j) {
            const auto r = attempt(j, j + 1, true);
            all = all && r.first && r.second;
        }
        check(all, "rollback: a 6-row verify with junk after row j, committed to keep j + 1 (j = 0..5), decoded on == the prefill");
        const auto n1 = attempt(2, 6, true);
        check(n1.first && !n1.second, "rollback negative control: committing all 6 rows (junk included) differs");
        const auto n2 = attempt(2, 3, false);
        check(n2.first && !n2.second, "rollback negative control: skipping the conv record copy differs");
    }

    /* 6: batching */
    {
        uint32_t len[kSeqs], slot_of[kSeqs];
        for (uint32_t q = 0; q < kSeqs; ++q) {
            len[q] = q == 9 ? 70u : 1u + (q * 7u) % 20u;
            len[q] = std::min(len[q], N - 40);
            slot_of[q] = 16 + (q * 5u) % 16u;
            zero_slot(c, slot_of[q]);
        }
        std::vector<Run> runs;
        for (uint32_t q = 0; q < kSeqs; ++q) runs.push_back({q, slot_of[q], 0, len[q]});
        std::vector<uint16_t> o;
        std::vector<Seqout> so(kSeqs);
        bool ok = call(c, runs, &o, false) == 0;
        size_t r0 = 0;
        for (uint32_t q = 0; ok && q < kSeqs; ++q)
            for (uint32_t r = 0; r < len[q]; ++r, ++r0) so[q].put(c, r, &o[r0 * HP]);
        /* a mixed call: even sequences decode one row, odd ones run 2..16 rows */
        runs.clear();
        for (uint32_t q = 0; q < kSeqs; ++q) {
            const uint32_t n = q % 2 == 0 ? 1u : std::min(2u + q, 16u);
            runs.push_back({q, slot_of[q], len[q], n});
        }
        ok = ok && call(c, runs, &o, false) == 0;
        r0 = 0;
        for (uint32_t q = 0; ok && q < kSeqs; ++q) {
            for (uint32_t r = 0; r < runs[q].count; ++r, ++r0) so[q].put(c, len[q] + r, &o[r0 * HP]);
            len[q] += runs[q].count;
        }
        /* a captured verify of all 16 in one call (runs of 1..8, junk after row keep - 1), each committed to its keep */
        uint32_t keep[kSeqs], first_row[kSeqs];
        runs.clear();
        uint32_t f = 0;
        for (uint32_t q = 0; q < kSeqs; ++q) {
            const uint32_t n = 1u + (q * 3u) % 8u;
            keep[q] = 1u + (q * 5u) % n;
            runs.push_back({q, slot_of[q], len[q], n, keep[q]});
            first_row[q] = f;
            f += n;
        }
        ok = ok && call(c, runs, &o, true) == 0;
        for (uint32_t q = 0; ok && q < kSeqs; ++q) {
            for (uint32_t r = 0; r < keep[q]; ++r) so[q].put(c, len[q] + r, &o[(size_t)(first_row[q] + r) * HP]);
            ok = commit(c, slot_of[q], first_row[q], keep[q]);
            len[q] += keep[q];
        }
        bool eq = ok;
        for (uint32_t q = 0; eq && q < kSeqs; ++q) {
            std::vector<uint16_t> alone;
            zero_slot(c, 1);
            eq = call(c, {{q, 1, 0, len[q]}}, &alone, false) == 0;
            const std::vector<uint16_t> got(so[q].y.begin(), so[q].y.begin() + (size_t)len[q] * HP);
            eq = eq && same16(got, alone) && same_slots(get_slots(c, slot_of[q]), get_slots(c, 1));
        }
        check(eq, "batching: 16 sequences on permuted slots (ragged prefill incl. 70 rows; decode + runs; one captured "
                  "verify call, a different keep each) == each alone");
    }

    /* 7: mutations */
    for (m2ref_mut m : {M2_NO_D, M2_NO_GATE, M2_GROUP_HALF, M2_GROUP_WHOLE, M2_GROUP_MAP, M2_NO_CONV_BIAS}) {
        if (m == M2_GROUP_WHOLE && c.R.NG == (int)HP) continue;
        if (m == M2_GROUP_MAP && c.R.G == 1) continue;
        std::vector<double> mref;
        reference(c, 0, N, m, &mref);
        const Grade mg = grade(pf, mref);
        const char *nm = m == M2_NO_D ? "D dropped" : m == M2_NO_GATE ? "gate dropped" : m == M2_GROUP_HALF ? "norm group halved"
                       : m == M2_GROUP_WHOLE ? "one norm group" : m == M2_GROUP_MAP ? "B/C group map h % G" : "conv bias dropped";
        check(mg.fro >= 10 * kFroBar, "mutation (%s) fails the grade: fro %.3e (>= %.0e)", nm, mg.fro, 10 * kFroBar);
    }
}

static void refusals(Case &c) {
    printf("\n== refusals ==\n");
    std::vector<uint16_t> o;
    zero_slot(c, 0);
    check(call(c, {{0, 0, 0, 4}}, &o, false) == 0, "the base call is accepted");
    /* a call built by hand so each field can be broken */
    const int32_t hf[2] = {0, 4}, hs[1] = {0};
    CK(cudaMemcpy(c.d_first, hf, 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c.d_slot, hs, 4, cudaMemcpyHostToDevice));
    pulsar_mamba2_call k{};
    k.n_seqs = 1; k.n_rows = 4; k.max_seq_rows = 4; k.seq_first = c.d_first; k.seq_slot = c.d_slot;
    k.z = c.d_in; k.ld_z = c.ROW; k.xbc = c.d_in + c.HP; k.ld_xbc = c.ROW; k.dt = c.d_in + c.HP + c.CC; k.ld_dt = c.ROW;
    k.conv_state = c.d_conv; k.ssm_state = c.d_ssm; k.scratch = c.d_scr; k.scratch_bytes = c.scr_bytes;
    k.out = c.d_out; k.ld_out = c.HP;
    check(pulsar_mamba2_forward(&c.L, &k, 0) == 0, "the hand-built call is accepted");
    CK(cudaDeviceSynchronize());
    auto refused = [&](const char *what, const pulsar_mamba2_layer &L, const pulsar_mamba2_call &kk) {
        check(pulsar_mamba2_forward(&L, &kk, 0) == -1, "refused: %s", what);
    };
    pulsar_mamba2_layer L = c.L;
    pulsar_mamba2_call t = k;
    L.geom.head_dim = 32; refused("head_dim 32 (built: 64 x 128)", L, k);
    L = c.L; L.geom.d_state = 64; refused("d_state 64", L, k);
    L = c.L; L.geom.n_groups = 5; refused("n_heads not a multiple of n_groups", L, k);
    L = c.L; L.geom.n_groups = 0; refused("no groups", L, k);
    L = c.L; L.geom.norm_group = 128; refused("norm group 128 (not a multiple of 256)", L, k);
    L = c.L; L.geom.norm_group = 512; refused("a norm group not dividing H*P (512 into 768)", L, k);
    L = c.L; L.geom.conv_kernel = 1; refused("conv kernel 1", L, k);
    L = c.L; L.geom.conv_kernel = 9; refused("conv kernel 9", L, k);
    L = c.L; L.geom.norm_eps = 0.f; refused("norm eps 0", L, k);
    L = c.L; L.D = nullptr; refused("no D", L, k);
    L = c.L; L.norm_w = nullptr; refused("no norm weight", L, k);
    L = c.L; L.conv_w = c.L.conv_w + 1; refused("a misaligned conv weight", L, k);
    t = k; t.ssm_capture = c.d_scap; refused("an SSM capture without the conv capture", c.L, t);
    t = k; t.conv_capture = c.d_ccap; refused("a conv capture without the SSM capture", c.L, t);
    t = k; t.scratch_bytes = pulsar_mamba2_scratch_bytes(&c.L.geom, 4) - 16; refused("short scratch", c.L, t);
    t = k; t.out = c.d_in + 8; t.ld_out = c.ROW; refused("an output overlapping the inputs", c.L, t);
    t = k; t.ld_xbc = c.CC - 8; refused("xbc stride below CC", c.L, t);
    t = k; t.ld_z = c.ROW + 4; refused("a z stride not a multiple of 8", c.L, t);
    t = k; t.z = c.d_in + 4; refused("a misaligned z", c.L, t);
    t = k; t.ssm_state = c.d_ssm + 1; refused("a misaligned SSM pool", c.L, t);
    t = k; t.n_seqs = 0; refused("no sequences", c.L, t);
    t = k; t.max_seq_rows = 5; refused("a longest run past the rows", c.L, t);
    t = k; t.seq_first = nullptr; refused("no row descriptors", c.L, t);
    check(pulsar_mamba2_commit(&c.L.geom, c.d_ssm, 0, c.d_scap, 0, 0) == -1, "refused: a commit of 0 rows");
    check(pulsar_mamba2_commit(&c.L.geom, c.d_ssm, 0, c.d_scap + 1, 1, 0) == -1, "refused: a misaligned DELTA record");
    L = c.L; L.geom.head_dim = 32;
    check(pulsar_mamba2_commit(&L.geom, c.d_ssm, 0, c.d_scap, 1, 0) == -1, "refused: a commit at an unbuilt geometry");
}

/* ---- real layers (tests/mamba2_mini_fixture.py) ----------------------------------------------------- */
static bool read_file(const std::string &p, std::vector<uint8_t> *out) {
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); return false; }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out->resize((size_t)n);
    const bool ok = fread(out->data(), 1, (size_t)n, f) == (size_t)n;
    fclose(f);
    return ok;
}

/* DIR/layers.txt lists the layer subdirectories; DIR/geom.txt holds "H P N G K NG eps T". */
static int fixture(const char *dir) {
    std::vector<uint8_t> gt, lt;
    if (!read_file(std::string(dir) + "/geom.txt", &gt) || !read_file(std::string(dir) + "/layers.txt", &lt)) return 2;
    gt.push_back(0); lt.push_back(0);
    unsigned H, Pd, Nd, G, K, NG, T;
    float eps;
    if (sscanf((const char *)gt.data(), "%u %u %u %u %u %u %f %u", &H, &Pd, &Nd, &G, &K, &NG, &eps, &T) != 8) return 2;
    printf("== real Mamba-2 layers vs transformers' NemotronHMamba2Mixer (fp32): H %u x P %u, N %u, G %u, K %u, norm "
           "group %u, eps %g, %u rows ==\n", H, Pd, Nd, G, K, NG, eps, T);
    const pulsar_mamba2_geom geom{H, Pd, Nd, G, K, NG, eps};
    const uint32_t HP = H * Pd, CC = pulsar_mamba2_conv_channels(&geom), ROW = HP + CC + H;
    const uint64_t rec = pulsar_mamba2_conv_state_elems(&geom), ssmf = pulsar_mamba2_ssm_state_floats(&geom);
    char name[64];
    const char *lp = (const char *)lt.data();
    int used = 0;
    while (sscanf(lp, "%63s%n", name, &used) == 1) {
        lp += used;
        const std::string base = std::string(dir) + "/" + name + "/";
        std::vector<uint8_t> in, r32, rbf, s32, par[6];
        const char *pn[6] = {"conv_w", "conv_b", "A_log", "dt_bias", "D", "norm_w"};
        bool ok = read_file(base + "in.bin", &in) && read_file(base + "ref32.bin", &r32) && read_file(base + "refbf16.bin", &rbf) &&
                  read_file(base + "state32.bin", &s32);
        for (int i = 0; ok && i < 6; ++i) ok = read_file(base + pn[i] + ".bin", &par[i]);
        if (!ok || in.size() != (size_t)T * ROW * 2 || r32.size() != (size_t)T * HP * 4 || rbf.size() != (size_t)T * HP * 2 ||
            s32.size() != ssmf * 4) {
            fprintf(stderr, "%s: missing or mis-sized fixture files\n", name);
            return 2;
        }
        uint16_t *d_par[6];
        for (int i = 0; i < 6; ++i) {
            CK(cudaMalloc(&d_par[i], par[i].size() + 16));
            CK(cudaMemcpy(d_par[i], par[i].data(), par[i].size(), cudaMemcpyHostToDevice));
        }
        pulsar_mamba2_layer L{geom, d_par[0], d_par[1], d_par[2], d_par[3], d_par[4], d_par[5]};
        /* the in_proj row is [z | xbc | dt] at ROW; pad it to a multiple of 8 on the device */
        const uint32_t LD = (ROW + 7) / 8 * 8;
        std::vector<uint16_t> hin((size_t)T * LD, 0);
        for (uint32_t t = 0; t < T; ++t) memcpy(&hin[(size_t)t * LD], in.data() + (size_t)t * ROW * 2, (size_t)ROW * 2);
        uint16_t *d_in, *d_out, *d_conv;
        float *d_ssm;
        void *d_scr;
        int32_t *d_first, *d_slot;
        const size_t scr = pulsar_mamba2_scratch_bytes(&geom, T);
        CK(cudaMalloc(&d_in, hin.size() * 2)); CK(cudaMemcpy(d_in, hin.data(), hin.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_out, (size_t)3 * T * HP * 2));
        CK(cudaMalloc(&d_conv, 3 * rec * 2)); CK(cudaMemset(d_conv, 0, 3 * rec * 2));
        CK(cudaMalloc(&d_ssm, 3 * ssmf * 4)); CK(cudaMemset(d_ssm, 0, 3 * ssmf * 4));
        CK(cudaMalloc(&d_scr, scr));
        CK(cudaMalloc(&d_first, 8)); CK(cudaMalloc(&d_slot, 4));
        /* schedule s on slot s: one call; T one-row calls; chunks of 128 then 1..37-row cuts */
        auto run = [&](uint32_t slot, uint32_t start, uint32_t n) {
            const int32_t f[2] = {0, (int32_t)n}, s = (int32_t)slot;
            CK(cudaMemcpy(d_first, f, 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_slot, &s, 4, cudaMemcpyHostToDevice));
            pulsar_mamba2_call k{};
            k.n_seqs = 1; k.n_rows = n; k.max_seq_rows = n; k.seq_first = d_first; k.seq_slot = d_slot;
            k.z = d_in + (size_t)start * LD; k.ld_z = LD;
            k.xbc = k.z + HP; k.ld_xbc = LD; k.dt = k.z + HP + CC; k.ld_dt = LD;
            k.conv_state = d_conv; k.ssm_state = d_ssm; k.scratch = d_scr; k.scratch_bytes = scr;
            k.out = d_out + ((size_t)slot * T + start) * HP; k.ld_out = HP;
            return pulsar_mamba2_forward(&L, &k, 0) == 0;
        };
        ok = run(0, 0, T);
        for (uint32_t t = 0; ok && t < T; ++t) ok = run(1, t, 1);
        {
            const uint32_t cuts[] = {128, 1, 37, 300, 5, 64};
            uint32_t p = 0, i = 0;
            while (ok && p < T) { const uint32_t n = std::min(T - p, cuts[i++ % 6]); ok = run(2, p, n); p += n; }
        }
        CK(cudaDeviceSynchronize());
        std::vector<uint16_t> y((size_t)3 * T * HP);
        std::vector<float> st((size_t)3 * ssmf);
        CK(cudaMemcpy(y.data(), d_out, y.size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(st.data(), d_ssm, st.size() * 4, cudaMemcpyDeviceToHost));
        const float *ref = (const float *)r32.data();
        const uint16_t *hbf = (const uint16_t *)rbf.data();
        /* the double reference with the block's (vLLM's) bf16 rounding points on the same inputs and weights: it
         * splits the block's distance to transformers fp32 into the declared roundings (dbl vs fp32) and the
         * block's own f32 arithmetic (block vs dbl) */
        std::vector<double> dref((size_t)T * HP), dssm(ssmf, 0.0), dconv(rec, 0.0);
        {
            m2ref_layer R;
            R.H = (int)H; R.P = (int)Pd; R.N = (int)Nd; R.G = (int)G; R.K = (int)K; R.NG = (int)NG; R.eps = (double)eps;
            auto widen = [&](const std::vector<uint8_t> &b) {
                std::vector<float> v(b.size() / 2);
                for (size_t i = 0; i < v.size(); ++i) v[i] = sref_bf2f(((const uint16_t *)b.data())[i]);
                return v;
            };
            R.conv_w = widen(par[0]); R.conv_b = widen(par[1]); R.A_log = widen(par[2]); R.dt_bias = widen(par[3]);
            R.D = widen(par[4]); R.norm_w = widen(par[5]);
            const std::vector<float> f = widen(in);
            m2ref_seq(R, (int)T, f.data(), ROW, f.data() + HP, ROW, f.data() + HP + CC, ROW, dconv.data(), dssm.data(), dref.data());
        }
        auto stats = [&](auto get, auto want, double *fro, double *share) {
            double num = 0, den = 0;
            size_t eq = 0;
            for (size_t i = 0; i < (size_t)T * HP; ++i) {
                const double v = get(i), r = want(i), d = v - r;
                num += d * d; den += r * r;
                eq += sref_f2bf((float)v) == sref_f2bf((float)r);
            }
            *fro = sqrt(num / den);
            *share = (double)eq / ((double)T * HP);
        };
        auto vs32 = [&](size_t i) { return (double)ref[i]; };
        auto vsdbl = [&](size_t i) { return dref[i]; };
        auto blk = [&](size_t i) { return (double)sref_bf2f(y[i]); };
        double fb, sb, fh, sh, fd, sd, fbd, sbd;
        stats(blk, vs32, &fb, &sb);
        stats([&](size_t i) { return (double)sref_bf2f(hbf[i]); }, vs32, &fh, &sh);
        stats(vsdbl, vs32, &fd, &sd);
        stats(blk, vsdbl, &fbd, &sbd);
        auto sfro_vs = [&](auto want) {
            double num = 0, den = 0;
            for (uint64_t i = 0; i < ssmf; ++i) { const double d = (double)st[i] - want(i); num += d * d; den += want(i) * want(i); }
            return sqrt(num / den);
        };
        const float *sref32 = (const float *)s32.data();
        const double sfro = sfro_vs([&](uint64_t i) { return (double)sref32[i]; });
        const double sfro_d = sfro_vs([&](uint64_t i) { return dssm[i]; });
        double dnum = 0, dden = 0;
        for (uint64_t i = 0; i < ssmf; ++i) { const double d = dssm[i] - sref32[i]; dnum += d * d; dden += (double)sref32[i] * sref32[i]; }
        printf("  %s vs transformers fp32: block fro %.3e (== bf16 %.2f%%) | double with the block's rounding points %.3e (%.2f%%)"
               " | transformers bf16 %.3e (%.2f%%)\n", name, fb, 100.0 * sb, fd, 100.0 * sd, fh, 100.0 * sh);
        printf("  %s block vs the double with its rounding points: fro %.3e, == bf16 %.4f%%;  SSM state: block vs fp32 %.3e, "
               "double vs fp32 %.3e, block vs double %.3e\n", name, fbd, 100.0 * sbd, sfro, sqrt(dnum / dden), sfro_d);
        check(ok && fbd <= 3e-3 && sbd >= 0.99, "%s: block vs the double with its rounding points: fro %.3e (bar 3e-3), == bf16 "
              "%.4f%% (bar 99%%)", name, fbd, 100.0 * sbd);
        check(ok && fb <= 1.1 * fd && fb <= 1e-2, "%s: block vs transformers fp32 %.3e is the declared roundings' own distance "
              "%.3e (within 10%%) and within 1e-2 (transformers bf16: %.3e)", name, fb, fd, fh);
        check(ok && sfro_d <= 1e-4, "%s: the f32 SSM state after %u rows vs the double: fro %.3e (bar 1e-4)", name, T, sfro_d);
        const bool d1 = memcmp(&y[0], &y[(size_t)T * HP], (size_t)T * HP * 2) == 0 && memcmp(&st[0], &st[ssmf], ssmf * 4) == 0;
        const bool d2 = memcmp(&y[0], &y[(size_t)2 * T * HP], (size_t)T * HP * 2) == 0 && memcmp(&st[0], &st[2 * ssmf], ssmf * 4) == 0;
        check(ok && d1, "%s: %u one-row decode calls == one prefill call, bit-exact (out, SSM state)", name, T);
        check(ok && d2, "%s: chunks of 128/1/37/300/5/64 == one prefill call, bit-exact", name);
        for (int i = 0; i < 6; ++i) cudaFree(d_par[i]);
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_conv); cudaFree(d_ssm); cudaFree(d_scr); cudaFree(d_first); cudaFree(d_slot);
    }
    printf("\n%s\n", g_fail ? "MAMBA2 MINI GRADE FAILED" : "MAMBA2 MINI GRADE PASS");
    return g_fail;
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--fixture")) return fixture(argv[2]);
    if (argc != 1) { fprintf(stderr, "usage: %s [--fixture DIR]\n", argv[0]); return 2; }
    printf("mamba2 gate (L296 N2): the family-neutral Mamba-2 mixer block\n");
    Case cases[3];
    case_init(cases[0], "Nemotron 3.5 Super layer", 128, 8, 1024, 4, 160, 21);
    case_init(cases[1], "one TP=2 rank of it", 64, 4, 1024, 4, 120, 22);
    case_init(cases[2], "small: 12 heads, 3 groups, norm group 256, kernel 2", 12, 3, 256, 2, 120, 23);
    for (Case &c : cases) run_case(c);
    refusals(cases[2]);
    for (Case &c : cases) case_free(c);
    printf("\n%s\n", g_fail ? "MAMBA2 GATE FAILED" : "MAMBA2 GATE PASS");
    return g_fail;
}

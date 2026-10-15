/* Short convolution block gate (L295 E3) -- model-free, links the PRODUCTION
 * object (src/cuda/pulsar_cuda_sconv.o, the engine's NVCCFLAGS incl.
 * fast-math).  Per declaration (Inkling's K / V stream and residual stream,
 * plus kernels 2, 3 and 8 with and without the residual):
 *
 *   1. bytes      a one-call prefill == the host replay of the block's own
 *                 arithmetic (tests/sconv_ref.h sref_seq_bits), every byte, and
 *                 the advanced slot == the last K-1 inputs.
 *   2. fidelity   the same call against the DOUBLE reference: TOLERANCE-graded
 *                 (relative Frobenius; the share of outputs equal to
 *                 bf16(reference); the worst distance in bf16 ulps).
 *   3. decode     one-row calls == the prefill, BIT-EXACT (outputs and slot).
 *   4. chunking   the prompt cut at 1, 2, 3, 5, 16, 17, 37, ... rows == one call.
 *   5. verify     runs of 2..16 rows with capture after a 40-row prefill == the
 *                 prefill, and every captured record == the state after its row.
 *   6. rollback   a 6-row verify whose rows after j are junk, restored to its row
 *                 j's capture and decoded on == the prefill (j = 0, 2, 4); the
 *                 same run WITHOUT the restore must differ (the restore is
 *                 load-bearing).
 *   7. batching   16 sequences on permuted slots, ragged prefill runs (one of 70
 *                 rows) then a mixed decode + verify call == each sequence
 *                 alone, bit-exact.
 *   8. mutations  the double reference with the taps reversed, the residual
 *                 flipped, or the window lagged one row must FAIL the grade
 *                 (>= 20x the bar), so the grade sees each feature.
 * Then the multi-stream launch (Inkling's K and V read strided from one fused
 * qkvr row, in one launch == two launches), and refusals.
 *
 * --fixture DIR grades the block on the Inkling-Small mini checkpoint's layer 0
 * (tests/sconv_mini_fixture.py): its four streams' real inputs and weights
 * against transformers' InklingShortConvolution in fp32. */
#include "cuda/pulsar_cuda_sconv.h"
#include "sconv_ref.h"

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

/* The fidelity bar: kernel bf16 vs the double reference.  The output's own bf16 rounding is ~1e-3 relative;
 * the f32 chain adds ~1e-7. */
static const double kFroBar = 3e-3;
static const double kExactShareBar = 0.999;   /* share of outputs equal to bf16(reference) */

/* ---- a declaration under test, its sequences' inputs and the device pool ------------------------------ */
struct Case {
    const char *name;
    sref_conv R;
    pulsar_sconv_decl D{};
    uint16_t *d_w = nullptr;
    uint32_t N = 0;                          /* rows per sequence */
    uint64_t ld = 0;                         /* the call buffers' row stride (> C: strided rows) */
    uint64_t sld = 0, sstride = 0;           /* a slot's row stride (> C: strided records) and the pool's slot stride */
    std::vector<std::vector<float>> X;       /* per sequence: [N][C] bf16-exact inputs */
    uint16_t *d_pool = nullptr;              /* 32 slots */
    uint16_t *d_x = nullptr, *d_y = nullptr, *d_cap = nullptr;   /* call buffers, up to kMaxRows rows */
    int32_t *d_first = nullptr, *d_slot = nullptr;
};
static const uint32_t kSlots = 32, kSeqs = 16, kMaxRows = 4096;

static void case_init(Case &c, const char *name, int C, int K, bool residual, uint32_t N, uint64_t seed) {
    c.name = name; c.N = N;
    c.R.C = C; c.R.K = K; c.R.residual = residual;
    Rng g(seed);
    c.R.w.resize((size_t)C * K);
    for (auto &w : c.R.w) w = sref_round((float)(0.4 * g.n()));
    std::vector<uint16_t> wb(c.R.w.size());
    for (size_t i = 0; i < wb.size(); ++i) wb[i] = sref_f2bf(c.R.w[i]);
    CK(cudaMalloc(&c.d_w, wb.size() * 2));
    CK(cudaMemcpy(c.d_w, wb.data(), wb.size() * 2, cudaMemcpyHostToDevice));
    c.D.channels = (uint32_t)C; c.D.kernel = (uint32_t)K; c.D.residual = residual ? 1 : 0; c.D.weight = c.d_w;
    c.ld = (uint64_t)C + 16;
    c.sld = (uint64_t)C + 8;
    c.sstride = (uint64_t)(K - 1) * c.sld + 8;
    c.X.resize(kSeqs);
    for (uint32_t q = 0; q < kSeqs; ++q) {
        Rng gq(seed * 131 + q + 1);
        c.X[q].resize((size_t)N * C);
        for (auto &x : c.X[q]) x = sref_round((float)gq.n());
    }
    CK(cudaMalloc(&c.d_pool, kSlots * c.sstride * 2));
    CK(cudaMalloc(&c.d_x, kMaxRows * c.ld * 2));
    CK(cudaMalloc(&c.d_y, kMaxRows * c.ld * 2));
    CK(cudaMalloc(&c.d_cap, kMaxRows * c.sstride * 2));
    CK(cudaMalloc(&c.d_first, (kSeqs + 1) * 4));
    CK(cudaMalloc(&c.d_slot, kSeqs * 4));
}
static void case_free(Case &c) {
    cudaFree(c.d_w); cudaFree(c.d_pool); cudaFree(c.d_x); cudaFree(c.d_y); cudaFree(c.d_cap);
    cudaFree(c.d_first); cudaFree(c.d_slot);
}

static void zero_slot(Case &c, uint32_t slot) { CK(cudaMemset(c.d_pool + slot * c.sstride, 0, c.sstride * 2)); }
/* a slot's K-1 rows of C, gathered from its row stride */
static std::vector<uint16_t> get_slot(Case &c, uint32_t slot) {
    std::vector<uint16_t> h(pulsar_sconv_state_elems(&c.D));
    CK(cudaMemcpy2D(h.data(), (size_t)c.R.C * 2, c.d_pool + slot * c.sstride, c.sld * 2, (size_t)c.R.C * 2, c.R.K - 1,
                    cudaMemcpyDeviceToHost));
    return h;
}

/* One run of a call: sequence q's rows [start, start + count) on `slot`; `junk` rows (from index junk_from of
 * the run) take another sequence's inputs instead. */
struct Run { uint32_t q, slot, start, count, junk_from = ~0u; };

/* Runs the call; out[r] is call row r's bf16 output ([rows][C]); captures into `cap` when asked. */
static int call(Case &c, const std::vector<Run> &runs, std::vector<uint16_t> *out, bool capture,
                std::vector<uint16_t> *cap = nullptr) {
    const int C = c.R.C;
    std::vector<int32_t> first(1, 0), slot;
    std::vector<uint16_t> hx;
    uint32_t longest = 0;
    for (const Run &r : runs) {
        for (uint32_t i = 0; i < r.count; ++i) {
            const uint32_t src_q = i >= r.junk_from ? (r.q + 7) % kSeqs : r.q;
            const float *row = c.X[src_q].data() + (size_t)(r.start + i) * C;
            const size_t base = hx.size();
            hx.resize(base + c.ld, 0x7fc1);   /* the stride's padding holds NaN bits: a read past C shows */
            for (int k = 0; k < C; ++k) hx[base + k] = sref_f2bf(row[k]);
        }
        first.push_back(first.back() + (int32_t)r.count);
        slot.push_back((int32_t)r.slot);
        longest = std::max(longest, r.count);
    }
    const uint32_t rows = (uint32_t)first.back();
    CK(cudaMemcpy(c.d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c.d_first, first.data(), first.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c.d_slot, slot.data(), slot.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(c.d_y, 0xff, (size_t)rows * c.ld * 2));
    pulsar_sconv_stream s{};
    s.decl = &c.D; s.x = c.d_x; s.ld_x = c.ld; s.y = c.d_y; s.ld_y = c.ld;
    s.state = c.d_pool; s.state_ld = c.sld; s.state_stride = c.sstride;
    s.capture = capture ? c.d_cap : nullptr; s.capture_ld = c.sld; s.capture_stride = c.sstride;
    pulsar_sconv_rows rw{(uint32_t)runs.size(), rows, longest, c.d_first, c.d_slot};
    const int rc = pulsar_sconv_forward(&s, 1, &rw, 0);
    if (rc) return rc;
    CK(cudaDeviceSynchronize());
    out->resize((size_t)rows * C);
    CK(cudaMemcpy2D(out->data(), (size_t)C * 2, c.d_y, c.ld * 2, (size_t)C * 2, rows, cudaMemcpyDeviceToHost));
    if (capture && cap) {
        const uint64_t se = pulsar_sconv_state_elems(&c.D);
        cap->resize((size_t)rows * se);   /* record r's K-1 rows of C, gathered from the capture's row stride */
        CK(cudaMemcpy2D(cap->data(), (size_t)C * 2, c.d_cap, c.sld * 2, (size_t)C * 2, (size_t)rows * (c.R.K - 1),
                        cudaMemcpyDeviceToHost));
        (void)se;
    }
    return 0;
}

/* sequence q's outputs for positions [0, N) assembled from a schedule of calls; rows land at their positions */
struct Seqout {
    std::vector<uint16_t> y;   /* [N][C] */
    void put(const Case &c, uint32_t pos, const uint16_t *row) {
        if (y.empty()) y.assign((size_t)c.N * c.R.C, 0);
        memcpy(&y[(size_t)pos * c.R.C], row, (size_t)c.R.C * 2);
    }
};

/* the state after position p (inclusive) of sequence q, as the slot holds it */
static std::vector<uint16_t> state_after(const Case &c, uint32_t q, int64_t p) {
    const int C = c.R.C, K = c.R.K;
    std::vector<uint16_t> s((size_t)(K - 1) * C, 0);
    for (int i = 0; i < K - 1; ++i) {
        const int64_t pos = p - (K - 2) + i;
        if (pos < 0) continue;
        for (int k = 0; k < C; ++k) s[(size_t)i * C + k] = sref_f2bf(c.X[q][(size_t)pos * C + k]);
    }
    return s;
}

static bool same(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, size_t *first_diff = nullptr) {
    if (a.size() != b.size()) { if (first_diff) *first_diff = 0; return false; }
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) { if (first_diff) *first_diff = i; return false; }
    return true;
}

struct Grade { double fro = 0, exact_share = 0; int max_ulp = 0; };
static int bf_ulps(uint16_t a, uint16_t b) {
    auto ord = [](uint16_t v) { return (v & 0x8000u) ? -(int)(v & 0x7fffu) : (int)v; };
    return abs(ord(a) - ord(b));
}
static Grade grade(const std::vector<uint16_t> &y, const std::vector<double> &ref) {
    Grade g;
    double num = 0, den = 0;
    size_t eq = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        const double d = (double)sref_bf2f(y[i]) - ref[i];
        num += d * d; den += ref[i] * ref[i];
        const uint16_t rb = sref_f2bf((float)ref[i]);
        if (rb == y[i]) ++eq;
        g.max_ulp = std::max(g.max_ulp, bf_ulps(rb, y[i]));
    }
    g.fro = sqrt(num / std::max(den, 1e-300));
    g.exact_share = (double)eq / (double)std::max<size_t>(y.size(), 1);
    return g;
}

static void run_case(Case &c) {
    const int C = c.R.C, K = c.R.K;
    const uint32_t N = c.N;
    printf("\n== %s: C %d, K %d, residual %d, %u rows ==\n", c.name, C, K, (int)c.R.residual, N);
    const uint64_t se = pulsar_sconv_state_elems(&c.D);

    /* references for sequence 0 */
    std::vector<double> ref((size_t)N * C), st_d(se, 0.0);
    sref_seq_double(c.R, (int)N, c.X[0].data(), (size_t)C, st_d.data(), ref.data());
    std::vector<uint16_t> bits((size_t)N * C);
    {
        std::vector<float> st_f(se, 0.0f);
        sref_seq_bits(c.R, (int)N, c.X[0].data(), (size_t)C, st_f.data(), bits.data());
    }
    double rms = 0;
    for (double v : ref) rms += v * v;
    rms = sqrt(rms / (double)ref.size());

    /* 1 + 2: the one-call prefill */
    std::vector<uint16_t> pf;
    zero_slot(c, 3);
    check(call(c, {{0, 3, 0, N}}, &pf, false) == 0, "prefill call");
    const std::vector<uint16_t> pf_state = get_slot(c, 3);
    size_t fd = 0;
    const bool by = same(pf, bits, &fd);
    check(by, "bytes: prefill == the host replay of the block's arithmetic (%zu outputs)%s", pf.size(),
          by ? "" : (" -- first diff at " + std::to_string(fd)).c_str());
    check(same(pf_state, state_after(c, 0, N - 1)), "bytes: the advanced slot == the last %d inputs", K - 1);
    const Grade g = grade(pf, ref);
    check(g.fro <= kFroBar && g.exact_share >= kExactShareBar && g.max_ulp <= 1,
          "fidelity vs double: fro %.3e (bar %.0e), == bf16(ref) %.4f%%, worst %d ulp (rms(ref) %.3f)", g.fro, kFroBar,
          100.0 * g.exact_share, g.max_ulp, rms);

    /* 3: one-row decode calls */
    {
        zero_slot(c, 5);
        Seqout so;
        std::vector<uint16_t> o;
        bool ok = true;
        for (uint32_t p = 0; p < N && ok; ++p) {
            ok = call(c, {{0, 5, p, 1}}, &o, false) == 0;
            if (ok) so.put(c, p, o.data());
        }
        check(ok && same(so.y, pf) && same(get_slot(c, 5), pf_state), "decode: %u one-row calls == the prefill (out, slot)", N);
    }

    /* 4: chunked */
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
            for (uint32_t r = 0; ok && r < n; ++r) so.put(c, p + r, &o[(size_t)r * C]);
            p += n;
        }
        check(ok && same(so.y, pf) && same(get_slot(c, 6), pf_state), "chunking: cut at 1/2/3/5/16/17/37/64/15 == one call");
    }

    /* 5: verify runs with capture */
    {
        zero_slot(c, 7);
        Seqout so;
        std::vector<uint16_t> o, cap;
        const uint32_t P = std::min<uint32_t>(40, N);
        bool ok = call(c, {{0, 7, 0, P}}, &o, false) == 0;
        for (uint32_t r = 0; ok && r < P; ++r) so.put(c, r, &o[(size_t)r * C]);
        uint32_t p = P, k = 2;
        bool caps = true;
        while (p < N && ok) {
            const uint32_t n = std::min(N - p, k);
            k = k == 16 ? 2 : k + 1;
            ok = call(c, {{0, 7, p, n}}, &o, true, &cap) == 0;
            for (uint32_t r = 0; ok && r < n; ++r) {
                so.put(c, p + r, &o[(size_t)r * C]);
                const std::vector<uint16_t> want = state_after(c, 0, p + r);
                caps = caps && memcmp(&cap[(size_t)r * se], want.data(), se * 2) == 0;
            }
            p += n;
        }
        check(ok && same(so.y, pf) && same(get_slot(c, 7), pf_state), "verify: runs of 2..16 rows after a %u-row prefill == the prefill", P);
        check(ok && caps, "verify: every captured record == the state after its row");
    }

    /* 6: rollback */
    if (N >= 60) {
        for (uint32_t j : {0u, 2u, 4u}) {
            for (int restore = 1; restore >= 0; --restore) {
                const uint32_t slot = 8 + j;
                zero_slot(c, slot);
                Seqout so;
                std::vector<uint16_t> o, cap;
                const uint32_t P = 40;
                bool ok = call(c, {{0, slot, 0, P}}, &o, false) == 0;
                for (uint32_t r = 0; ok && r < P; ++r) so.put(c, r, &o[(size_t)r * C]);
                Run v{0, slot, P, 6, j + 1};   /* rows 0..j true, the rest junk */
                ok = ok && call(c, {v}, &o, true, &cap) == 0;
                for (uint32_t r = 0; ok && r <= j; ++r) so.put(c, P + r, &o[(size_t)r * C]);
                pulsar_sconv_stream s{};
                s.decl = &c.D; s.state = c.d_pool; s.state_ld = c.sld; s.state_stride = c.sstride;
                s.capture = c.d_cap; s.capture_ld = c.sld; s.capture_stride = c.sstride;
                if (restore) ok = ok && pulsar_sconv_restore(&s, slot, j, 0) == 0;
                for (uint32_t p = P + j + 1; ok && p < N; ++p) {
                    ok = call(c, {{0, slot, p, 1}}, &o, false) == 0;
                    so.put(c, p, o.data());
                }
                const bool eq = ok && same(so.y, pf) && same(get_slot(c, slot), pf_state);
                if (restore) check(eq, "rollback: verify of 6 rows, junk after row %u, restored to row %u's capture, decoded on == the prefill", j, j);
                else check(ok && !eq, "rollback negative control: the same run WITHOUT the restore differs (row %u)", j);
            }
        }
    }

    /* 7: batching -- 16 sequences on permuted slots */
    {
        uint32_t len[kSeqs], slot_of[kSeqs];
        for (uint32_t q = 0; q < kSeqs; ++q) {
            len[q] = q == 9 ? 70u : 1u + (q * 7u) % 20u;
            len[q] = std::min(len[q], N - 20);
            slot_of[q] = 16 + (q * 5u) % 16u;
            zero_slot(c, slot_of[q]);
        }
        std::vector<Seqout> so(kSeqs);
        std::vector<Run> runs;
        for (uint32_t q = 0; q < kSeqs; ++q) runs.push_back({q, slot_of[q], 0, len[q]});
        std::vector<uint16_t> o;
        bool ok = call(c, runs, &o, false) == 0;
        size_t r0 = 0;
        for (uint32_t q = 0; ok && q < kSeqs; ++q)
            for (uint32_t r = 0; r < len[q]; ++r, ++r0) so[q].put(c, r, &o[r0 * C]);
        /* a mixed call: even sequences decode one row, odd ones verify 2..16 rows */
        runs.clear();
        uint32_t nxt[kSeqs];
        for (uint32_t q = 0; q < kSeqs; ++q) {
            const uint32_t n = q % 2 == 0 ? 1u : std::min(2u + q, 16u);
            runs.push_back({q, slot_of[q], len[q], n});
            nxt[q] = n;
        }
        ok = ok && call(c, runs, &o, false) == 0;
        r0 = 0;
        for (uint32_t q = 0; ok && q < kSeqs; ++q)
            for (uint32_t r = 0; r < nxt[q]; ++r, ++r0) so[q].put(c, len[q] + r, &o[r0 * C]);
        /* each sequence alone, one call over the same rows */
        bool eq = ok;
        for (uint32_t q = 0; eq && q < kSeqs; ++q) {
            const uint32_t total = len[q] + nxt[q];
            std::vector<uint16_t> alone;
            zero_slot(c, 1);
            eq = call(c, {{q, 1, 0, total}}, &alone, false) == 0;
            const std::vector<uint16_t> got(so[q].y.begin(), so[q].y.begin() + (size_t)total * C);
            eq = eq && same(got, alone) && same(get_slot(c, slot_of[q]), get_slot(c, 1));
        }
        check(eq, "batching: 16 sequences on permuted slots (ragged runs incl. 70 rows, then decode + verify) == each alone");
    }

    /* 8: mutations */
    for (sref_mut m : {SREF_REVERSED, SREF_NO_RESIDUAL, SREF_LAGGED}) {
        std::vector<double> mref((size_t)N * C), ms(se, 0.0);
        sref_seq_double(c.R, (int)N, c.X[0].data(), (size_t)C, ms.data(), mref.data(), m);
        const Grade mg = grade(pf, mref);
        const char *nm = m == SREF_REVERSED ? "taps reversed" : m == SREF_NO_RESIDUAL ? "residual flipped" : "window lagged";
        check(mg.fro >= 20 * kFroBar, "mutation (%s) fails the grade: fro %.3e (>= %.0e)", nm, mg.fro, 20 * kFroBar);
    }
}

/* Inkling's layer: K and V as strided columns of one fused qkvr row plus the two 4096 deltas, their state in ONE
 * packed record a sequence ([3][10240], streams at channels 0 / 1024 / 2048 / 6144); one launch == four launches */
static void multi_stream() {
    printf("\n== multi-stream: K and V from a fused [q 4096 | k 1024 | v 1024 | r 512] row + two 4096 streams, one packed "
           "[3][10240] record ==\n");
    const uint32_t T = 37, LD = 6656;
    Rng g(77);
    std::vector<uint16_t> hrow((size_t)T * LD), hw((size_t)1024 * 4 * 2 + (size_t)4096 * 4 * 2);
    for (auto &v : hrow) v = sref_f2bf((float)g.n());
    for (auto &v : hw) v = sref_f2bf((float)(0.4 * g.n()));
    uint16_t *d_row, *d_w, *d_y, *d_pool, *d_big, *d_bigy;
    CK(cudaMalloc(&d_row, hrow.size() * 2)); CK(cudaMemcpy(d_row, hrow.data(), hrow.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_w, hw.size() * 2)); CK(cudaMemcpy(d_w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_y, (size_t)2 * 2 * T * 4096 * 2));
    CK(cudaMalloc(&d_bigy, (size_t)2 * 2 * T * 4096 * 2));
    std::vector<uint16_t> hbig((size_t)T * 8192);
    for (auto &v : hbig) v = sref_f2bf((float)g.n());
    CK(cudaMalloc(&d_big, hbig.size() * 2));
    CK(cudaMemcpy(d_big, hbig.data(), hbig.size() * 2, cudaMemcpyHostToDevice));
    const uint64_t RW = 10240, REC = 3 * RW;            /* the record's row width and size */
    const uint64_t off[4] = {0, 1024, 2048, 6144};      /* each stream's channels in the record */
    CK(cudaMalloc(&d_pool, 2 * REC * 2));
    CK(cudaMemset(d_pool, 0, 2 * REC * 2));
    int32_t hfirst[2] = {0, (int32_t)T}, hslot[1] = {0};
    int32_t *d_first, *d_slot;
    CK(cudaMalloc(&d_first, 8)); CK(cudaMalloc(&d_slot, 4));
    CK(cudaMemcpy(d_first, hfirst, 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_slot, hslot, 4, cudaMemcpyHostToDevice));
    pulsar_sconv_decl kd{1024, 4, 1, d_w}, vd{1024, 4, 1, d_w + 4096}, ad{4096, 4, 1, d_w + 8192}, md{4096, 4, 1, d_w + 8192 + 16384};
    const pulsar_sconv_rows rows{1, T, T, d_first, d_slot};
    auto stream_of = [&](const pulsar_sconv_decl *d, const uint16_t *x, uint64_t ldx, uint16_t *y, uint32_t i) {
        pulsar_sconv_stream s{};
        s.decl = d; s.x = x; s.ld_x = ldx; s.y = y; s.ld_y = 4096;
        s.state = d_pool + off[i]; s.state_ld = RW; s.state_stride = REC;
        return s;
    };
    /* one launch: K, V, and the two 4096 streams (4 streams, two widths) */
    const pulsar_sconv_stream four[4] = {
        stream_of(&kd, d_row + 4096, LD, d_y, 0), stream_of(&vd, d_row + 5120, LD, d_y + (size_t)T * 4096, 1),
        stream_of(&ad, d_big, 8192, d_y + (size_t)2 * T * 4096, 2), stream_of(&md, d_big + 4096, 8192, d_y + (size_t)3 * T * 4096, 3)};
    check(pulsar_sconv_forward(four, 4, &rows, 0) == 0, "one launch of 4 streams (1024, 1024, 4096, 4096)");
    for (int i = 0; i < 4; ++i) {
        pulsar_sconv_stream s = four[i];
        s.y = d_bigy + (size_t)i * T * 4096;
        s.state = d_pool + REC + off[i];   /* record 1 */
        if (pulsar_sconv_forward(&s, 1, &rows, 0) != 0) check(false, "single-stream launch %d", i);
    }
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> a((size_t)4 * T * 4096), b(a.size()), rec(2 * REC);
    CK(cudaMemcpy(a.data(), d_y, a.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), d_bigy, b.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(rec.data(), d_pool, rec.size() * 2, cudaMemcpyDeviceToHost));
    bool eq = true;
    for (int i = 0; i < 4 && eq; ++i) {
        const uint32_t C = i < 2 ? 1024 : 4096;
        for (uint32_t t = 0; t < T && eq; ++t)
            eq = memcmp(&a[(size_t)i * T * 4096 + (size_t)t * 4096], &b[(size_t)i * T * 4096 + (size_t)t * 4096], C * 2) == 0;
    }
    check(eq && memcmp(rec.data(), rec.data() + REC, REC * 2) == 0,
          "4 streams in one launch == 4 launches (outputs and the whole packed record)");
    /* the record holds each stream's last 3 inputs in its channel range */
    bool laid = true;
    for (int i = 0; i < 4; ++i)
        for (uint64_t r = 0; r < 3; ++r) {
            const uint32_t C = i < 2 ? 1024 : 4096;
            const uint16_t *src = i < 2 ? &hrow[(size_t)(T - 3 + r) * LD + 4096 + 1024 * i] : &hbig[(size_t)(T - 3 + r) * 8192 + 4096 * (i - 2)];
            laid = laid && memcmp(&rec[r * RW + off[i]], src, (size_t)C * 2) == 0;
        }
    check(laid, "the packed record's row r holds every stream's input at position T-3+r in its channel range");
    /* the K stream against the host replay, on its strided input */
    {
        sref_conv R; R.C = 1024; R.K = 4; R.residual = true; R.w.resize(4096);
        for (int i = 0; i < 4096; ++i) R.w[i] = sref_bf2f(hw[i]);
        std::vector<float> x((size_t)T * 1024), st(3 * 1024, 0.f);
        for (uint32_t t = 0; t < T; ++t) for (int k = 0; k < 1024; ++k) x[(size_t)t * 1024 + k] = sref_bf2f(hrow[(size_t)t * LD + 4096 + k]);
        std::vector<uint16_t> y((size_t)T * 1024);
        sref_seq_bits(R, (int)T, x.data(), 1024, st.data(), y.data());
        bool ok = true;
        for (uint32_t t = 0; t < T && ok; ++t) ok = memcmp(&y[(size_t)t * 1024], &a[(size_t)t * 4096], 1024 * 2) == 0;
        check(ok, "the K stream read strided from the fused row == the host replay");
    }
    cudaFree(d_row); cudaFree(d_w); cudaFree(d_y); cudaFree(d_bigy); cudaFree(d_big); cudaFree(d_pool);
    cudaFree(d_first); cudaFree(d_slot);
}

static void refusals() {
    printf("\n== refusals ==\n");
    uint16_t *b; int32_t *d;
    CK(cudaMalloc(&b, 1 << 20)); CK(cudaMalloc(&d, 64));
    const int32_t hf[3] = {0, 4, 0};   /* seq_first {0, 4}, seq_slot {0} */
    CK(cudaMemcpy(d, hf, 12, cudaMemcpyHostToDevice));
    pulsar_sconv_decl D{64, 4, 1, b};
    pulsar_sconv_stream s{};
    s.decl = &D; s.x = b + 4096; s.ld_x = 64; s.y = b + 8192; s.ld_y = 64; s.state = b + 16384; s.state_ld = 64; s.state_stride = 192;
    pulsar_sconv_rows r{1, 4, 4, d, d + 2};
    check(pulsar_sconv_forward(&s, 1, &r, 0) == 0, "the base call is accepted");
    CK(cudaDeviceSynchronize());
    auto refused = [&](const char *what, pulsar_sconv_stream s2, pulsar_sconv_rows r2, uint32_t n = 1) {
        pulsar_sconv_stream two[2] = {s2, s2};
        check(pulsar_sconv_forward(two, n, &r2, 0) == -1, "refused: %s", what);
    };
    pulsar_sconv_decl d2 = D; pulsar_sconv_stream t = s; pulsar_sconv_rows r2 = r;
    d2.channels = 12; t.decl = &d2; refused("channels not a multiple of 8", t, r);
    d2 = D; d2.kernel = 1; refused("kernel 1", t, r);
    d2 = D; d2.kernel = 9; refused("kernel 9", t, r);
    d2 = D; d2.weight = b + 1; refused("a misaligned weight", t, r);
    t = s; t.x = b + 4097; refused("a misaligned input", t, r);
    t = s; t.ld_x = 60; refused("ld_x below the channels", t, r);
    t = s; t.ld_y = 68; refused("ld_y not a multiple of 8", t, r);
    t = s; t.y = b + 4096 + 64; refused("an output overlapping the input rows", t, r);
    t = s; t.state_stride = 184; refused("a slot stride below (K-2) * state_ld + C", t, r);
    t = s; t.state_ld = 56; refused("a slot row stride below the channels", t, r);
    t = s; t.state_ld = 68; t.state_stride = 208; refused("a slot row stride not a multiple of 8", t, r);
    t = s; t.capture = b + 32768; t.capture_ld = 64; t.capture_stride = 64; refused("a capture stride below one record", t, r);
    t = s; refused("no streams", t, r, 0);
    refused("five streams", t, r, 5);
    r2 = r; r2.n_seqs = 0; refused("no sequences", s, r2);
    r2 = r; r2.max_seq_rows = 5; refused("a longest run past the call's rows", s, r2);
    r2 = r; r2.seq_first = nullptr; refused("no row descriptors", s, r2);
    {
        pulsar_sconv_decl k3 = D; k3.kernel = 3;
        pulsar_sconv_stream two[2] = {s, s};
        two[1].decl = &k3; two[1].y = b + 12288; two[1].state = b + 20480;
        check(pulsar_sconv_forward(two, 2, &r, 0) == -1, "refused: streams of one call with different kernels");
    }
    cudaFree(b); cudaFree(d);
}

/* ---- the mini-checkpoint fixture (tests/sconv_mini_fixture.py) ---------------------------------------- */
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

static int fixture(const char *dir) {
    printf("== the Inkling-Small mini, layer 0: the four streams vs transformers' InklingShortConvolution (fp32) ==\n");
    const char *names[4] = {"k", "v", "attn", "mlp"};
    const uint32_t Cs[4] = {1024, 1024, 4096, 4096};
    for (int i = 0; i < 4; ++i) {
        std::vector<uint8_t> xin, w, r32;
        const std::string base = std::string(dir) + "/" + names[i];
        if (!read_file(base + ".in.bin", &xin) || !read_file(base + ".w.bin", &w) || !read_file(base + ".ref32.bin", &r32)) return 2;
        const uint32_t C = Cs[i], T = (uint32_t)(xin.size() / 2 / C);
        if (w.size() != (size_t)C * 4 * 2 || r32.size() != (size_t)T * C * 4) { fprintf(stderr, "%s: sizes\n", names[i]); return 2; }
        uint16_t *d_x, *d_w, *d_y, *d_y1, *d_pool;
        int32_t *d_first, *d_slot;
        CK(cudaMalloc(&d_x, xin.size())); CK(cudaMemcpy(d_x, xin.data(), xin.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_w, w.size())); CK(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_y, xin.size())); CK(cudaMalloc(&d_y1, xin.size()));
        CK(cudaMalloc(&d_pool, (size_t)2 * 3 * C * 2)); CK(cudaMemset(d_pool, 0, (size_t)2 * 3 * C * 2));
        CK(cudaMalloc(&d_first, (T + 1) * 4)); CK(cudaMalloc(&d_slot, T * 4));
        pulsar_sconv_decl D{C, 4, 1, d_w};
        pulsar_sconv_stream s{};
        s.decl = &D; s.x = d_x; s.ld_x = C; s.y = d_y; s.ld_y = C; s.state = d_pool; s.state_ld = C; s.state_stride = (uint64_t)3 * C;
        const int32_t one[2] = {0, (int32_t)T}, zero = 0;
        CK(cudaMemcpy(d_first, one, 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_slot, &zero, 4, cudaMemcpyHostToDevice));
        pulsar_sconv_rows r{1, T, T, d_first, d_slot};
        bool ok = pulsar_sconv_forward(&s, 1, &r, 0) == 0;
        /* the same rows as T one-row decode calls on slot 1 */
        const int32_t first1[2] = {0, 1}, one_slot = 1;
        int32_t *d_f1, *d_s1;
        CK(cudaMalloc(&d_f1, 8)); CK(cudaMalloc(&d_s1, 4));
        CK(cudaMemcpy(d_f1, first1, 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_s1, &one_slot, 4, cudaMemcpyHostToDevice));
        for (uint32_t t = 0; ok && t < T; ++t) {
            pulsar_sconv_stream s1 = s;
            s1.x = d_x + (size_t)t * C; s1.y = d_y1 + (size_t)t * C;
            pulsar_sconv_rows r1{1, 1, 1, d_f1, d_s1};
            ok = pulsar_sconv_forward(&s1, 1, &r1, 0) == 0;
        }
        CK(cudaDeviceSynchronize());
        std::vector<uint16_t> y((size_t)T * C), y1(y.size());
        CK(cudaMemcpy(y.data(), d_y, y.size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(y1.data(), d_y1, y1.size() * 2, cudaMemcpyDeviceToHost));
        const float *ref = (const float *)r32.data();
        double num = 0, den = 0, maxd = 0;
        size_t eq = 0;
        int worst = 0;
        for (size_t k = 0; k < y.size(); ++k) {
            const double d = (double)sref_bf2f(y[k]) - ref[k];
            num += d * d; den += (double)ref[k] * ref[k];
            maxd = std::max(maxd, fabs(d));
            const uint16_t rb = sref_f2bf(ref[k]);
            eq += rb == y[k];
            worst = std::max(worst, bf_ulps(rb, y[k]));
        }
        const double fro = sqrt(num / den), share = (double)eq / (double)y.size();
        check(ok && fro <= kFroBar && worst <= 1,
              "%-4s C %4u, %u rows: fro %.3e vs transformers fp32, max |d| %.3e, == bf16(transformers) %.4f%%, worst %d ulp "
              "(rms %.3f)", names[i], C, T, fro, maxd, 100.0 * share, worst, sqrt(den / (double)y.size()));
        check(ok && y == y1, "%-4s %u one-row decode calls == one prefill call, bit-exact", names[i], T);
        cudaFree(d_x); cudaFree(d_w); cudaFree(d_y); cudaFree(d_y1); cudaFree(d_pool); cudaFree(d_first); cudaFree(d_slot);
        cudaFree(d_f1); cudaFree(d_s1);
    }
    printf("\n%s\n", g_fail ? "SCONV MINI GRADE FAILED" : "SCONV MINI GRADE PASS");
    return g_fail;
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--fixture")) return fixture(argv[2]);
    if (argc != 1) { fprintf(stderr, "usage: %s [--fixture DIR]\n", argv[0]); return 2; }
    printf("sconv gate (L295 E3): the family-neutral short convolution block\n");
    Case cases[5];
    case_init(cases[0], "Inkling K / V stream", 1024, 4, true, 300, 11);
    case_init(cases[1], "Inkling residual stream", 4096, 4, true, 300, 12);
    case_init(cases[2], "kernel 2, plain", 64, 2, false, 120, 13);
    case_init(cases[3], "kernel 8", 256, 8, true, 120, 14);
    case_init(cases[4], "kernel 3, plain, 8 channels", 8, 3, false, 90, 15);
    for (Case &c : cases) run_case(c);
    for (Case &c : cases) case_free(c);
    multi_stream();
    refusals();
    printf("\n%s\n", g_fail ? "SCONV GATE FAILED" : "SCONV GATE PASS");
    return g_fail;
}

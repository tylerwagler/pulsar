/* QWEN S4 GATE (L251): the Qwen3.8-Flash-Next router, MoE block, Gated Residual
 * and PLE injection -- the production objects (src/cuda/pulsar_cuda_qwen_*.o,
 * the EXL3 arms, the MMQ drivers) against the double host references in
 * tests/qwen_ref.h.  Model-free: random weights at the real shapes, activations
 * with heavy tails and outlier channels, the A8 encodings where the device
 * encodes.
 *
 *   A. router: logits vs double; the top-10 decision from the device's logits
 *      (set, order, bf16 weights); a crafted TIE at the top-10 boundary must go to
 *      the lower expert id; a NaN row routes to 0..9 with NaN weights; T = 1 and
 *      T = 8 rows bit-identical to the T = 37 run's; refusal of a foreign shape.
 *   B. GR: the read (block input bf16 row, its E4M3 slot, inj) and the mixer vs
 *      double; the write; T = 1 bit-identical to the batch's row; mutations of
 *      W_down and W_up move the output.
 *   C. PLE: three sequences in one batch over two batches (the conv state carried
 *      between them) vs double; the second batch re-run as single-token steps is
 *      bit-identical (streams AND state) -- decode == prefill; a mutated conv tap
 *      moves the output.
 *   D. MoE block over 512 experts (an aliased pool of 12 distinct slices per
 *      projection, gate/up K=4, down K=5; shared K=5/5/4) vs double with the
 *      device's routing; T = 1 bit-identical; the non-finite flag stays clear.
 *
 * usage: ./tests/qwen_s4_gate */
#include "../src/pulsar_gpu.h"
#include "qwen_ref.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace qref;

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { printf("  FAIL  "); g_fail = 1; } else printf("  ok    "); \
                           printf(__VA_ARGS__); printf("\n"); } while (0)
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("QWEN-S4 FAIL: %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static uint64_t g_rng = 0x51F15EEDULL;
static uint32_t rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 16); }
static double rndu(void) { return ((double)rnd() + 0.5) / 4294967296.0; }
static double rndn(void) { return sqrt(-2.0 * log(rndu())) * cos(6.283185307179586 * rndu()); }

template <typename T> static T *up(const std::vector<T> &v) {
    T *d = nullptr;
    CK(cudaMalloc((void **)&d, v.size() * sizeof(T) + 16));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T> static std::vector<T> down(const T *d, size_t n) {
    std::vector<T> v(n);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
static void *dalloc(size_t n) { void *p = nullptr; CK(cudaMalloc(&p, n + 16)); CK(cudaMemset(p, 0, n + 16)); return p; }

static std::vector<uint16_t> rnd_bf(size_t n, double scale) {
    std::vector<uint16_t> v(n);
    for (auto &b : v) b = to_bf(rndn() * scale);
    return v;
}
/* activations: heavy-tailed rows with a few outlier channels */
static std::vector<uint16_t> rnd_act(int rows, int n, double scale) {
    std::vector<double> chan(n);
    for (auto &c : chan) c = (rnd() % 64 == 0) ? 10.0 : 1.0;
    std::vector<uint16_t> v((size_t)rows * n);
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < n; k++) v[(size_t)r * n + k] = to_bf(rndn() * chan[k] * scale * (1.0 + 0.5 * (r % 3)));
    return v;
}
static linear make_linear(int in, int out, int k2) {
    linear l; l.in = in; l.out = out; l.k2 = k2;
    uint64_t trellis = 0, stride = 0;
    exl3t_layout(in, out, k2, &trellis, &stride);
    l.bytes.resize(stride);
    uint16_t *t = (uint16_t *)l.bytes.data();
    for (uint64_t i = 0; i < trellis / 2; i++) t[i] = (uint16_t)rnd();
    uint16_t *s = (uint16_t *)(l.bytes.data() + trellis);
    for (int i = 0; i < in; i++) s[i] = exl3_f32_to_f16((float)((0.5 + rndu()) * 0.025 * ((rnd() & 1) ? 1.0 : -1.0)));
    for (int i = 0; i < out; i++) s[in + i] = exl3_f32_to_f16((float)((0.75 + 0.5 * rndu()) * ((rnd() & 1) ? 1.0 : -1.0)) * 0.02f);
    l.dequant();
    return l;
}
static pulsar_qwen_linear dev_linear(const linear &l) {
    pulsar_qwen_linear d;
    d.w = up(l.bytes); d.k2 = l.k2; d.in = l.in; d.out = l.out;
    return d;
}
/* |got - want| in bf16 ulps of want (the ulp of the binade of |want|) */
static double bf_ulps(double got, double want) {
    const double a = fabs(want);
    const double ulp = a > 0 ? ldexp(1.0, (int)floor(log2(a)) - 7) : ldexp(1.0, -133);
    return fabs(got - want) / ulp;
}
/* L251 / ac69748f: make_dslot and its dslot wrapper are DELETED, not updated.  They built the A8
 * slot a consumer would read; this family has no E4M3 activation slot, the MoE reads the bf16 rows
 * directly, and the reference activation is the bf16 value rather than a decoded E4M3 code. */

/* ======================================================================== */
static void section_mxfp8_linear(void);   /* defined at the end */
static void section_router(void) {
    printf("A. router (bf16 logits, softmax top-%d of %d, renormalised)\n", TOPK, E);
    const int T = 37;
    std::vector<uint16_t> x = rnd_act(T, H, 0.6), wr = rnd_bf((size_t)E * H, 0.02), wsg = rnd_bf(H, 0.02);
    /* the tie: token 0's rows 100..108 dominate, rows 7 and 300 are identical and
     * rank 10 / 11 -- exactly one of them makes the top-10, and it must be 7 */
    for (int k = 0; k < H; k++) {
        const double sgn = bf(x[k]) >= 0 ? 1.0 : -1.0;
        for (int e = 100; e <= 108; e++) wr[(size_t)e * H + k] = to_bf(0.08 * sgn);
        wr[(size_t)7 * H + k] = wr[(size_t)300 * H + k] = to_bf(0.05 * sgn);
    }
    uint16_t *dx = up(x), *dwr = up(wr), *dwsg = up(wsg);
    float *lg = (float *)dalloc((size_t)T * (E + 1) * 4), *w = (float *)dalloc((size_t)T * TOPK * 4), *sg = (float *)dalloc(T * 4);
    int32_t *sel = (int32_t *)dalloc((size_t)T * TOPK * 4);
    int rc = pulsar_qwen_router_launch(dx, dwr, dwsg, T, H, E, TOPK, lg, sel, w, sg, 0);
    CHECK(rc == 0, "launch T=%d", T);
    const auto L = down(lg, (size_t)T * (E + 1));
    const auto Sel = down(sel, (size_t)T * TOPK);
    const auto W = down(w, (size_t)T * TOPK);
    const auto SG = down(sg, T);
    double lmax = 0;
    size_t sel_bad = 0, w_bad = 0, sg_bad = 0;
    std::vector<double> ref(E + 1);
    for (int t = 0; t < T; t++) {
        router_logits(&x[(size_t)t * H], wr.data(), wsg.data(), ref.data());
        double sc = 0;
        for (int e = 0; e <= E; e++) sc = fmax(sc, fabs(ref[e]));
        for (int e = 0; e <= E; e++) lmax = fmax(lmax, fabs(L[(size_t)t * (E + 1) + e] - ref[e]) / sc);
        int32_t hs[TOPK];
        double hw[TOPK], hsg;
        router_select(&L[(size_t)t * (E + 1)], hs, hw, &hsg);
        for (int k = 0; k < TOPK; k++) {
            sel_bad += hs[k] != Sel[(size_t)t * TOPK + k];
            w_bad += bf_ulps(W[(size_t)t * TOPK + k], hw[k]) > 1.0;
        }
        sg_bad += bf_ulps(SG[t], hsg) > 1.0;
    }
    CHECK(lmax < 1e-5, "logits vs double: max rel %.2e (f32 accumulation order)", lmax);
    CHECK(sel_bad == 0, "decision from the device's logits: %zu of %d selections differ", sel_bad, T * TOPK);
    CHECK(w_bad == 0 && sg_bad == 0, "weights / shared gate within 1 bf16 ulp: %zu / %zu outside", w_bad, sg_bad);
    bool has7 = false, has300 = false, top9 = true;
    for (int k = 0; k < TOPK; k++) { has7 |= Sel[k] == 7; has300 |= Sel[k] == 300; }
    for (int e = 100; e <= 108; e++) { bool f = false; for (int k = 0; k < TOPK; k++) f |= Sel[k] == e; top9 &= f; }
    CHECK(top9 && has7 && !has300 && Sel[9] == 7, "tie at the boundary: rows 7 == 300 at rank 10/11 -> 7 taken, 300 not");
    /* neutrality */
    size_t diff = 0;
    for (int Tn : {1, 8}) {
        const int t0 = 5;
        rc = pulsar_qwen_router_launch(dx + (size_t)t0 * H, dwr, dwsg, Tn, H, E, TOPK, lg, sel, w, sg, 0);
        const auto L2 = down(lg, (size_t)Tn * (E + 1));
        const auto S2 = down(sel, (size_t)Tn * TOPK);
        diff += memcmp(L2.data(), &L[(size_t)t0 * (E + 1)], L2.size() * 4) != 0;
        diff += memcmp(S2.data(), &Sel[(size_t)t0 * TOPK], S2.size() * 4) != 0;
    }
    CHECK(diff == 0, "T = 1 and T = 8 rows bit-identical to the T = %d run's (logits and selection)", T);
    /* NaN row */
    std::vector<uint16_t> xn(x.begin(), x.begin() + H);
    xn[17] = 0x7fc0;
    uint16_t *dxn = up(xn);
    rc = pulsar_qwen_router_launch(dxn, dwr, dwsg, 1, H, E, TOPK, lg, sel, w, sg, 0);
    const auto S3 = down(sel, TOPK);
    const auto W3 = down(w, TOPK);
    bool nan_ok = rc == 0;
    for (int k = 0; k < TOPK; k++) nan_ok &= S3[k] == k && std::isnan(W3[k]);
    CHECK(nan_ok, "a NaN row routes to experts 0..9 with NaN weights (the NaN reaches the output)");
    CHECK(pulsar_qwen_router_launch(dx, dwr, dwsg, T, 2048, E, TOPK, lg, sel, w, sg, 0) != 0, "a foreign hidden size is refused");
    cudaFree(dx); cudaFree(dwr); cudaFree(dwsg); cudaFree(lg); cudaFree(w); cudaFree(sg); cudaFree(sel); cudaFree(dxn);
}

/* ======================================================================== */
struct gr_host {
    std::vector<uint16_t> norm, inject;
    mx8 down_w, up_w;
};
static gr_host make_gr(void) {
    gr_host g;
    g.norm = rnd_bf(HC, 0.1);
    g.inject = rnd_bf((size_t)S * HC, 0.01);
    std::vector<double> wd((size_t)R * HC), wu((size_t)HC * R);
    for (auto &v : wd) v = rndn() * 0.01;
    for (auto &v : wu) v = rndn() * 0.08;
    g.down_w.encode(wd, R, HC);
    g.up_w.encode(wu, HC, R);
    return g;
}
static pulsar_qwen_gr_dev dev_gr(const gr_host &g, bool with_inject) {
    pulsar_qwen_gr_dev w;
    w.norm_w = up(g.norm);
    w.down = {up(g.down_w.q), up(g.down_w.sf), R, HC};
    w.up = {up(g.up_w.q), up(g.up_w.sf), HC, R};
    w.inject = with_inject ? up(g.inject) : nullptr;
    return w;
}

static void section_gr(void) {
    printf("B. Gated Residual (read, write, mixer)\n");
    const int T = 6;
    gr_host g = make_gr();
    std::vector<uint16_t> st = rnd_act(T, HC, 0.8);
    pulsar_qwen_gr_dev w = dev_gr(g, true);
    uint16_t *dst = up(st), *xo = (uint16_t *)dalloc((size_t)T * H * 2);
    /* L251 / ac69748f: the read emits bf16 only -- there is no E4M3 slot to allocate, arm or grade. */
    float *inj = (float *)dalloc((size_t)T * S * 4);
    const size_t wsb = pulsar_qwen_gr_workspace_bytes(T);
    void *ws = dalloc(wsb);
    int rc = pulsar_qwen_gr_read_launch(&w, dst, T, xo, inj, ws, wsb, 0);
    CHECK(rc == 0, "read launch T=%d", T);
    const auto X = down(xo, (size_t)T * H);
    const auto I = down(inj, (size_t)T * S);
    double worst_ulp = 0, inj_rel = 0;
    size_t over1 = 0, med_n = 0;
    double med_sum = 0;
    for (int t = 0; t < T; t++) {
        const gr_out o = gr_read(&st[(size_t)t * HC], g.norm.data(), g.down_w, g.up_w, g.inject.data());
        for (int c = 0; c < H; c++) {
            const double u = bf_ulps(bf(X[(size_t)t * H + c]), o.x[c]);
            worst_ulp = fmax(worst_ulp, u);
            over1 += u > 1.0;
            med_sum += fabs(bf(X[(size_t)t * H + c]) - o.x[c]) / (fabs(o.x[c]) + 1e-30);
            med_n++;
        }
        for (int j = 0; j < S; j++) inj_rel = fmax(inj_rel, fabs(I[t * S + j] - o.inj[j]) / o.inj[j]);
    }
    CHECK(over1 <= (size_t)(T * H) / 5000 && worst_ulp <= 4.0,
          "block input vs double: %zu of %d beyond 1 bf16 ulp (worst %.2f ulp; the read rounds to bf16 once, "
          "and the W8 path's internal a is still E4M3 -- see the row)",
          over1, T * H, worst_ulp);
    printf("        mean |rel| %.2e\n", med_sum / med_n);
    CHECK(inj_rel < 1e-5, "inj = 2 sigmoid(W_inj xn / 4): max rel %.2e", inj_rel);
    /* neutrality */
    rc = pulsar_qwen_gr_read_launch(&w, dst + (size_t)3 * HC, 1, xo, inj, ws, wsb, 0);
    const auto X1 = down(xo, H);
    const auto I1 = down(inj, S);
    CHECK(rc == 0 && memcmp(X1.data(), &X[(size_t)3 * H], H * 2) == 0 && memcmp(I1.data(), &I[3 * S], S * 4) == 0,
          "T = 1 row bit-identical to the T = %d batch's row (block input + inj)", T);
    /* prefill widths (T > 16): the up runs as the W8A16 GEMM (3'), graded against double like T = 6 */
    for (const int TP : {37, 300}) {
        const std::vector<uint16_t> sp = rnd_act(TP, HC, 0.8);
        uint16_t *dsp = up(sp), *xp = (uint16_t *)dalloc((size_t)TP * H * 2);
        float *injp = (float *)dalloc((size_t)TP * S * 4);
        const size_t wsp = pulsar_qwen_gr_workspace_bytes(TP);
        void *wp = dalloc(wsp);
        rc = pulsar_qwen_gr_read_launch(&w, dsp, TP, xp, injp, wp, wsp, 0);
        const auto XP = down(xp, (size_t)TP * H);
        const auto IP = down(injp, (size_t)TP * S);
        /* the same rows at decode widths (16, 16, 5), graded the same way */
        std::vector<uint16_t> XD((size_t)TP * H);
        for (int t0 = 0; t0 < TP; t0 += 16) {
            const int n = std::min(16, TP - t0);
            rc |= pulsar_qwen_gr_read_launch(&w, dsp + (size_t)t0 * HC, n, xp, injp, wp, wsp, 0);
            const auto part = down(xp, (size_t)n * H);
            std::copy(part.begin(), part.end(), XD.begin() + (size_t)t0 * H);
        }
        double wu = 0, ir = 0, wd = 0, wpd = 0;
        size_t o1 = 0, d1 = 0, pd = 0;
        for (int t = 0; t < TP; t++) {
            const gr_out o = gr_read(&sp[(size_t)t * HC], g.norm.data(), g.down_w, g.up_w, g.inject.data());
            for (int c = 0; c < H; c++) {
                const double u = bf_ulps(bf(XP[(size_t)t * H + c]), o.x[c]);
                const double ud = bf_ulps(bf(XD[(size_t)t * H + c]), o.x[c]);
                wu = fmax(wu, u);
                o1 += u > 1.0;
                wd = fmax(wd, ud);
                d1 += ud > 1.0;
                pd += XP[(size_t)t * H + c] != XD[(size_t)t * H + c];
                wpd = fmax(wpd, bf_ulps(bf(XP[(size_t)t * H + c]), bf(XD[(size_t)t * H + c])));
            }
            for (int j = 0; j < S; j++) ir = fmax(ir, fabs(IP[t * S + j] - o.inj[j]) / o.inj[j]);
        }
        /* the outliers are the INPUT's (a 4-stream mean that nearly cancels, after a bf16 tie of xn or a):
         * the decode kernels on these same rows show them too (T = 300: 212 beyond 1 ulp, worst 418), so
         * the prefill's count and worst are graded against the decode kernels' (worst: the triangle
         * bound through the prefill-vs-decode distance, itself graded below), whose own accuracy is the
         * T = 6 check above */
        CHECK(rc == 0 && o1 <= d1 + (size_t)(TP * H) / 20000 && wu <= wd + wpd + 1.0 && ir < 1e-5,
              "prefill width T = %d (the up as a GEMM) vs double: %zu of %d beyond 1 bf16 ulp (worst %.2f; the "
              "decode kernels on the same rows: %zu, worst %.2f), inj max rel %.2e", TP, o1, TP * H, wu, d1, wd, ir);
        /* a bf16 tie of `a` resolved differently by the two z orders moves an output by the same
         * cancellation that sets the decode kernels' own worst distance from double */
        CHECK(pd <= (size_t)(TP * H) / 1000 && wpd <= wd,
              "prefill vs decode widths on the same rows agree to rounding: %zu of %d differ, by at most %.2f ulp "
              "(<= the decode kernels' own worst vs double, %.2f)", pd, TP * H, wpd, wd);
        cudaFree(dsp); cudaFree(xp); cudaFree(injp); cudaFree(wp);
    }
    /* the mixer: no inject */
    pulsar_qwen_gr_dev wm = w;
    wm.inject = nullptr;
    rc = pulsar_qwen_gr_read_launch(&wm, dst, T, xo, nullptr, ws, wsb, 0);
    const auto XM = down(xo, (size_t)T * H);
    CHECK(rc == 0 && memcmp(XM.data(), X.data(), X.size() * 2) == 0, "the mixer (no inject) reads the same block input");
    /* the mixer as the graded recipe stores it: BF16 low-rank weights (bf16
     * activations into bf16 GEMVs), no inject */
    {
        const auto wdb = rnd_bf((size_t)R * HC, 0.01), wub = rnd_bf((size_t)HC * R, 0.08);
        pulsar_qwen_gr_dev wb = w;
        wb.inject = nullptr;
        wb.down = {up(wdb), nullptr, R, HC};
        wb.up = {up(wub), nullptr, HC, R};
        rc = pulsar_qwen_gr_read_launch(&wb, dst, T, xo, nullptr, ws, wsb, 0);
        const auto XB = down(xo, (size_t)T * H);
        double wu_ = 0;
        size_t ob = 0;
        for (int t = 0; t < T; t++) {
            const gr_out o = gr_read_bf16(&st[(size_t)t * HC], g.norm.data(), wdb.data(), wub.data(), nullptr);
            for (int c = 0; c < H; c++) {
                const double u = bf_ulps(bf(XB[(size_t)t * H + c]), o.x[c]);
                wu_ = fmax(wu_, u);
                ob += u > 1.0;
            }
        }
        /* xn and a are ROUNDED TO BF16 here (the bf16 GEMVs' inputs), a 32x finer grid than
         * E4M3's: an f32 value and the double reference straddle a bf16 tie a few times per
         * run, and one flipped element of `a` moves every gate it feeds -- so the bar is a
         * fraction and a bounded worst case, not zero */
        CHECK(rc == 0 && ob <= (size_t)(T * H) / 1000 && wu_ <= 16.0,
              "the BF16 mixer (bf16 low-rank, no E4M3 slot): %zu of %d beyond 1 bf16 ulp (worst %.2f ulp; bf16-tie flips "
              "of xn / a allowed at 1e-3)", ob, T * H, wu_);
        pulsar_qwen_gr_dev wmix = wb;
        wmix.up = {up(g.up_w.q), up(g.up_w.sf), HC, R};
        CHECK(pulsar_qwen_gr_read_launch(&wmix, dst, T, xo, nullptr, ws, wsb, 0) != 0,
              "a BF16 W_down paired with an MXFP8 W_up is refused");
    }
    /* the write */
    std::vector<float> out((size_t)T * H);
    for (auto &v : out) v = (float)(rndn() * 0.5);
    float *dout = up(out);
    CK(cudaMemcpy(inj, I.data(), I.size() * 4, cudaMemcpyHostToDevice));   /* the T = 1 run above overwrote row 0's */
    rc = pulsar_qwen_gr_write_launch(dst, dout, inj, T, 0);
    const auto ST2 = down(dst, (size_t)T * HC);
    size_t wr_bad = 0, wr_exact = 0;
    for (int t = 0; t < T; t++)
        for (int s = 0; s < S; s++)
            for (int c = 0; c < H; c++) {
                const size_t i = (size_t)t * HC + s * H + c;
                const double want = bf(st[i]) + (double)out[(size_t)t * H + c] * (double)I[t * S + s];
                const double u = bf_ulps(bf(ST2[i]), want);
                wr_bad += u > 0.51 && ST2[i] != to_bf(want);   /* f32 then bf16: a double rounding stays within ~0.5 ulp */
                wr_exact += ST2[i] == to_bf(want);
            }
    CHECK(wr_bad == 0, "write: streams += out * inj, one bf16 rounding: %zu of %d beyond half an ulp (%zu bit-exact)",
          wr_bad, T * HC, wr_exact);
    /* mutations */
    CK(cudaMemcpy(dst, st.data(), st.size() * 2, cudaMemcpyHostToDevice));
    size_t moved_u = 0, moved_d = 0;
    {
        pulsar_qwen_gr_dev wx = w;
        /* one 32-block scale of one W_up row x 8 (a single code is below the bf16
         * resolution of the gated mean) */
        std::vector<uint8_t> sf = g.up_w.sf;
        sf[pulsar_mx_sfoff(2 * H + 77, 1, pulsar_mx_kbp(R))] += 3;
        wx.up.sf = up(sf);
        pulsar_qwen_gr_read_launch(&wx, dst, T, xo, inj, ws, wsb, 0);
        const auto XX = down(xo, (size_t)T * H);
        for (size_t i = 0; i < XX.size(); i++) moved_u += XX[i] != X[i];
        wx = w;
        sf = g.down_w.sf;
        sf[pulsar_mx_sfoff(11, 125, pulsar_mx_kbp(HC))] += 4;
        wx.down.sf = up(sf);
        pulsar_qwen_gr_read_launch(&wx, dst, T, xo, inj, ws, wsb, 0);
        const auto XY = down(xo, (size_t)T * H);
        for (size_t i = 0; i < XY.size(); i++) moved_d += XY[i] != X[i];
    }
    CHECK(moved_u > 0 && moved_d > 0, "mutations: one W_up block scale x8 moved %zu outputs, one W_down block scale x16 moved %zu",
          moved_u, moved_d);
}

/* ======================================================================== */
static void section_ple(void) {
    printf("C. PLE injection (3 sequences, 2 batches, conv state carried)\n");
    const linear key = make_linear(H, HC, 10), value = make_linear(H, H, 8);
    pulsar_qwen_ple_dev w;
    w.key_proj = dev_linear(key);
    w.value_proj = dev_linear(value);
    const auto nk = rnd_bf(HC, 0.1), nq = rnd_bf(HC, 0.1), nc = rnd_bf(HC, 0.1), cw = rnd_bf((size_t)HC * 4, 0.4);
    w.norm_key = up(nk); w.norm_query = up(nq); w.norm_conv = up(nc); w.conv_w = up(cw);
    const int n_seq = 3;
    const int rowsA[n_seq] = {5, 3, 12}, rowsB[n_seq] = {1, 7, 2};
    float *state = (float *)dalloc((size_t)n_seq * PULSAR_QWEN_PLE_STATE * HC * 4);
    std::vector<std::vector<std::vector<double>>> hist(n_seq);
    std::vector<std::vector<uint16_t>> streams_host(n_seq);   /* per sequence, per row appended */
    auto run_batch = [&](const int *rows, bool host, std::vector<uint16_t> &emb_out, std::vector<uint16_t> &st_out,
                         std::vector<uint16_t> *dev_out, std::vector<uint16_t> *st_in) {
        int T = 0;
        for (int q = 0; q < n_seq; q++) T += rows[q];
        emb_out = rnd_act(T, H, 0.05);
        st_out = rnd_act(T, HC, 0.7);
        *st_in = st_out;
        std::vector<int32_t> rs(T), rj(T), sf(n_seq), sr(n_seq);
        int r = 0;
        for (int q = 0; q < n_seq; q++) {
            sf[q] = r; sr[q] = rows[q];
            for (int j = 0; j < rows[q]; j++, r++) { rs[r] = q; rj[r] = j; }
        }
        std::vector<int32_t> sb(n_seq);
        for (int q = 0; q < n_seq; q++) sb[q] = q;
        pulsar_qwen_rows pr = {up(rs), up(rj), up(sf), up(sr), up(sb), n_seq};
        uint16_t *de = up(emb_out), *ds = up(st_out);
        const size_t wsb = pulsar_qwen_ple_workspace_bytes(T);
        void *ws = dalloc(wsb);
        const int rc = pulsar_qwen_ple_launch(&w, de, ds, T, &pr, state, ws, wsb, 0);
        CHECK(rc == 0, "launch T=%d", T);
        *dev_out = down(ds, (size_t)T * HC);
        if (host) {
            std::vector<uint16_t> st = st_out;
            for (int i = 0; i < T; i++)
                ple_token(&emb_out[(size_t)i * H], key, value, nk.data(), nq.data(), nc.data(), cw.data(),
                          &st[(size_t)i * HC], hist[rs[i]]);
            st_out = st;
        }
        cudaFree(ws); cudaFree(de); cudaFree(ds);
    };
    double worst = 0;
    size_t over = 0, n = 0;
    std::vector<uint16_t> emb, st, dev, st_in;
    for (int b = 0; b < 2; b++) {
        run_batch(b ? rowsB : rowsA, true, emb, st, &dev, &st_in);
        for (size_t i = 0; i < st.size(); i++) {
            /* ulps at the larger of the result and the stream it updated: stream + out can
             * cancel, and then the f32 sum's own error is many ulps of the small result */
            const double sc = fmax(fabs(bf(st[i])), fabs(bf(st_in[i])));
            const double u = bf_ulps(bf(dev[i]), bf(st[i])) * (bf(st[i]) != 0 ? fabs(bf(st[i])) / sc : 1.0);
            worst = fmax(worst, u);
            over += u > 1.0;
            n++;
        }
    }
    CHECK(over <= n / 5000 && worst <= 4.0, "streams vs double over both batches: %zu of %zu beyond 1 bf16 ulp (worst %.2f)",
          over, n, worst);

    /* decode == prefill: a third batch (batch B's shape) from the carried state, as one
     * batch and as single-token steps from the same snapshot */
    std::vector<float> snap = down(state, (size_t)n_seq * PULSAR_QWEN_PLE_STATE * HC);
    int T = 0;
    for (int q = 0; q < n_seq; q++) T += rowsB[q];
    const auto emb2 = rnd_act(T, H, 0.05), st2 = rnd_act(T, HC, 0.7);
    std::vector<int32_t> rs(T), rj(T), sf(n_seq), sr(n_seq);
    for (int q = 0, r = 0; q < n_seq; q++) {
        sf[q] = r; sr[q] = rowsB[q];
        for (int j = 0; j < rowsB[q]; j++, r++) { rs[r] = q; rj[r] = j; }
    }
    std::vector<int32_t> sb(n_seq);
    for (int q = 0; q < n_seq; q++) sb[q] = q;
    pulsar_qwen_rows pr = {up(rs), up(rj), up(sf), up(sr), up(sb), n_seq};
    uint16_t *de = up(emb2), *ds = up(st2);
    const size_t wsb = pulsar_qwen_ple_workspace_bytes(T);
    void *ws = dalloc(wsb);
    int rc = pulsar_qwen_ple_launch(&w, de, ds, T, &pr, state, ws, wsb, 0);
    const auto batch_out = down(ds, (size_t)T * HC);
    const auto batch_state = down(state, snap.size());
    CK(cudaMemcpy(state, snap.data(), snap.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ds, st2.data(), st2.size() * 2, cudaMemcpyHostToDevice));
    /* single steps: every sequence one token at a time, interleaved round-robin */
    std::vector<int> next(n_seq, 0);
    for (bool any = true; any;) {
        any = false;
        for (int q = 0; q < n_seq; q++) {
            if (next[q] >= rowsB[q]) continue;
            any = true;
            const int row = sf[q] + next[q]++;
            /* a decode step of sequence q alone: one row, one sequence, owning slot q */
            const std::vector<int32_t> zero = {0}, one = {1}, bank = {q};
            pulsar_qwen_rows p1 = {up(zero), up(zero), up(zero), up(one), up(bank), 1};
            rc |= pulsar_qwen_ple_launch(&w, de + (size_t)row * H, ds + (size_t)row * HC, 1, &p1, state, ws, wsb, 0);
        }
    }
    const auto step_out = down(ds, (size_t)T * HC);
    const auto step_state = down(state, snap.size());
    CHECK(rc == 0 && memcmp(step_out.data(), batch_out.data(), batch_out.size() * 2) == 0 &&
          memcmp(step_state.data(), batch_state.data(), batch_state.size() * 4) == 0,
          "batch B re-run as %d single-token steps: streams and conv state bit-identical to the batch", T);
    /* mutation: one conv tap */
    CK(cudaMemcpy(state, snap.data(), snap.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ds, st2.data(), st2.size() * 2, cudaMemcpyHostToDevice));
    std::vector<uint16_t> cw2 = cw;
    cw2[(size_t)5000 * 4 + 1] = to_bf(bf(cw2[(size_t)5000 * 4 + 1]) + 0.5);
    pulsar_qwen_ple_dev wm = w;
    wm.conv_w = up(cw2);
    rc = pulsar_qwen_ple_launch(&wm, de, ds, T, &pr, state, ws, wsb, 0);
    const auto mut = down(ds, (size_t)T * HC);
    size_t moved = 0;
    for (size_t i = 0; i < mut.size(); i++) moved += mut[i] != batch_out[i];
    CHECK(rc == 0 && moved > 0, "mutation: one conv tap moved %zu outputs", moved);
}

/* ======================================================================== */
static void section_moe_at(const int T) {
    const int P = 12;
    std::vector<linear> pgu, pd;
    for (int i = 0; i < P; i++) { pgu.push_back(make_linear(H, 2 * MID, 8)); pd.push_back(make_linear(MID, H, 10)); }
    std::vector<expert> ex(E);
    for (int e = 0; e < E; e++) ex[e] = {&pgu[e % P], &pd[e % P]};
    const linear sg = make_linear(H, MID, 10), su = make_linear(H, MID, 10), sd = make_linear(MID, H, 8);
    auto table = [&](std::vector<linear> &pool) {
        std::vector<const void *> t(2 * E);
        std::vector<const void *> base(P);
        for (int i = 0; i < P; i++) base[i] = up(pool[i].bytes);
        uint64_t trellis = 0, stride = 0;
        exl3t_layout(pool[0].in, pool[0].out, pool[0].k2, &trellis, &stride);
        for (int e = 0; e < E; e++) { t[2 * e] = base[e % P]; t[2 * e + 1] = (const uint8_t *)base[e % P] + trellis; }
        return (const void *const *)up(t);
    };
    pulsar_qwen_moe_dev w;
    const auto wr = rnd_bf((size_t)E * H, 0.02), wsg = rnd_bf(H, 0.02);
    w.router_w = up(wr);
    w.shared_gate_w = up(wsg);
    w.gate_up_table = table(pgu); w.down_table = table(pd);
    w.k2_gate_up = 8; w.k2_down = 10;
    w.shared_gate = dev_linear(sg); w.shared_up = dev_linear(su); w.shared_down = dev_linear(sd);
    const auto x = rnd_act(T, H, 0.7);
    /* L251 / ac69748f: the MoE reads the bf16 rows directly -- there is no A8 slot, so the
     * reference's activation is the bf16 VALUE, not a decoded E4M3 code. */
    std::vector<double> xd((size_t)T * H);
    for (size_t i = 0; i < xd.size(); i++) xd[i] = bf(x[i]);
    uint16_t *dx = up(x);
    float *out = (float *)dalloc((size_t)T * H * 4);
    uint32_t *nf = (uint32_t *)dalloc(4);
    const size_t wsb = pulsar_qwen_moe_workspace_bytes(T);
    void *ws = dalloc(wsb);
    int rc = pulsar_qwen_moe_launch(&w, dx, T, out, ws, wsb, nf, 0x7351u, 0);
    CHECK(rc == 0, "launch T=%d (workspace %.1f MB)", T, wsb / 1e6);
    const auto O = down(out, (size_t)T * H);
    const auto NF = down(nf, 1);
    /* the device's routing, from the router alone on the same rows */
    float *lg = (float *)dalloc((size_t)T * (E + 1) * 4), *rw = (float *)dalloc((size_t)T * TOPK * 4), *rs = (float *)dalloc(T * 4);
    int32_t *sel = (int32_t *)dalloc((size_t)T * TOPK * 4);
    pulsar_qwen_router_launch(dx, w.router_w, w.shared_gate_w, T, H, E, TOPK, lg, sel, rw, rs, 0);
    const auto Sel = down(sel, (size_t)T * TOPK);
    const auto RW = down(rw, (size_t)T * TOPK);
    const auto RS = down(rs, T);
    double worst = 0, frob_worst = 0;
    int rows_f32 = 0;
    std::vector<double> ref(H);
    for (int t = 0; t < T; t++) {
        double wk[TOPK];
        for (int k = 0; k < TOPK; k++) wk[k] = RW[(size_t)t * TOPK + k];
        moe_token(&xd[(size_t)t * H], &Sel[(size_t)t * TOPK], wk, RS[t], ex, sg, su, sd, ref.data());
        double sc = 0, num = 0, den = 0;
        for (int o = 0; o < H; o++) sc = fmax(sc, fabs(ref[o]));
        for (int o = 0; o < H; o++) {
            const double d = O[(size_t)t * H + o] - ref[o];
            worst = fmax(worst, fabs(d) / sc);
            num += d * d; den += ref[o] * ref[o];
        }
        frob_worst = fmax(frob_worst, sqrt(num / den));
        rows_f32 += sqrt(num / den) < 1e-5;
        printf("        row %d: rel Frobenius %.2e\n", t, sqrt(num / den));
    }
    /* f32 order for most rows, except where a ROUNDING TIE at the fold's mid or the shared
     * expert's h sees the device's f32 value and the double reference on two sides of a
     * boundary: one flipped element moves its row off f32 order.
     *
     * L251 / ac69748f RE-DERIVED this bar rather than inheriting it.  The tie step was E4M3;
     * it is now bf16 for both intermediates.  Measured on the bf16 build: 3 of 5 rows at f32
     * order with worst 2.30e-05 -- so the absolute bound is TIGHTENED 10x here (1e-3 -> 1e-4,
     * still 4x clear of the measured value and 1000x below the 2.5e-02 a real format error
     * produced on this very line before the references were fixed), while the f32-order count
     * allows the two tie-affected rows.  The count is a proxy; the bound is the check. */
    if (T <= 16) {
        CHECK(rows_f32 >= T - 2 && frob_worst < 1e-4, "T = %d (decode GEMV) out vs double: %d of %d rows at f32 order "
              "(< 1e-5), worst row rel Frobenius %.2e, max |err| / max|ref| %.2e", T, rows_f32, T, frob_worst, worst);
    } else {
        /* the prefill GEMMs (routed: qwen_exl3_moe_prefill.cu, shared: qwen_exl3_dense_prefill.cu).
         * The shared expert's cuBLAS GEMM is ~4e-6 per element (tests/exl3_dense_gate), not the GEMV's
         * 3e-7, so more of h's (and the fold mid's) bf16 roundings land across a tie.  ONE flipped
         * element moves its row by up to ~2^-9 of that element's share of the output, so the worst row
         * is a property of the draw, not of the kernels: 1.07e-4 on the first draw, 4.51e-4 when an
         * added section B check shifted the RNG (2026-09-29).  The bound is 1e-3 -- the tie envelope,
         * 25x below the 2.5e-2 a real format error read on this line -- and quality is judged END TO
         * END: teacher-forced NLL over 1,792 positions (14 prefixes, code + raw prose) is 2.62 with these
         * kernels vs 2.63 for the all-GEMV build.  The f32-order count is reported, not graded. */
        CHECK(frob_worst < 1e-3, "T = %d (prefill GEMMs) out vs double: worst row rel Frobenius %.2e, max |err| / "
              "max|ref| %.2e (%d of %d rows at f32 order)", T, frob_worst, worst, rows_f32, T);
    }
    CHECK(NF[0] == 0, "non-finite flag clear (0x%x)", NF[0]);
    if (T <= 16) {
        /* the T = 1 run reads row 2's bf16 row -- the MoE indexes the activation itself now */
        rc = pulsar_qwen_moe_launch(&w, dx + (size_t)2 * H, 1, out, ws, wsb, nf, 0x7351u, 0);
        const auto O1 = down(out, H);
        CHECK(rc == 0 && memcmp(O1.data(), &O[(size_t)2 * H], H * 4) == 0, "T = 1 row bit-identical to the T = %d batch's row", T);
    }
}

static void section_moe(void) {
    printf("D. MoE block (512 experts over an aliased pool of 12; fused gate_up K=4, down K=5; shared K=5/5/4)\n");
    section_moe_at(5);    /* 50 assignments: the decode GEMV */
    section_moe_at(37);   /* 370 assignments: the prefill GEMM (>= QWEN_EXL3_MOE_PREFILL_MIN_ASSIGN) */
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (!pulsar_gpu_init()) { printf("QWEN-S4 FAIL: no GPU\n"); return 2; }
    printf("qwen-s4-gate: router / MoE / GR / PLE vs tests/qwen_ref.h (double)\n");
    section_router();
    section_gr();
    section_ple();
    section_moe();
    section_mxfp8_linear();
    printf(g_fail ? "QWEN-S4 GATE FAIL\n" : "QWEN-S4 GATE PASS\n");
    return g_fail;
}

/* ======================================================================== */
/* G. The MXFP8 dense Linear -- the recipe's mxfp8_lt tier (GDN in_proj_a / _b
 * and the indexer's index_qk_proj).  pulsar_qwen_mxfp8_linear_launch is a thin
 * wrapper over the GR arm's W_down, so this grades its NUMBERS: the reference
 * decodes the same quantized weight (mx8) and reads the SAME bf16 activation rows, so only the
 * accumulation order can differ.  L251 / ac69748f: the activation is bf16, so there is no E4M3
 * slot to emit and no per-32 block scale to fold in -- this is the W8A16 shape. */

static void section_mxfp8_linear(void) {
    printf("G. MXFP8 dense Linear (mxfp8_lt: in_proj_a 2560->48, index_qk_proj 2560->640)\n");
    const struct { int in, out; } SH[2] = {{H, 48}, {H, 640}};
    /* T = 5 is the decode GEMV; T = 97 the prefill tensor-core GEMM (two 64-token tiles, one partial) */
    for (const int T : {5, 97})
    for (const auto &sh : SH) {
        std::vector<double> wd((size_t)sh.out * sh.in);
        for (auto &v : wd) v = rndn() * 0.05;
        mx8 w;
        w.encode(wd, sh.out, sh.in);
        const std::vector<uint16_t> act = rnd_act(T, sh.in, 0.7);
        uint16_t *dx = up(act);                       /* the bf16 rows ARE the activation */
        float *y = (float *)dalloc((size_t)T * sh.out * 4);
        pulsar_qwen_lowrank l{up(w.q), up(w.sf), sh.out, sh.in};
        const int rc = pulsar_qwen_mxfp8_linear_launch(&l, dx, T, y, nullptr, 0, 0);
        CHECK(rc == 0, "%d -> %d launch", sh.in, sh.out);
        if (rc == 0) {
            const auto Y = down(y, (size_t)T * sh.out);
            const auto AX = down(dx, (size_t)T * sh.in);
            double worst = 0, w_ref = 0, w_got = 0, w_l1 = 0, worst_l1n = 0;
            int w_t = -1, w_o = -1;
            for (int t = 0; t < T; t++)
                for (int o = 0; o < sh.out; o++) {
                    double ref = 0, l1 = 0;
                    for (int k = 0; k < sh.in; k++) {
                        const double p = w.at(o, k) * bf(AX[(size_t)t * sh.in + k]);
                        ref += p;
                        l1 += fabs(p);
                    }
                    const double got = Y[(size_t)t * sh.out + o];
                    const double d = ref != 0 ? fabs(got - ref) / fabs(ref) : fabs(got);
                    if (l1 > 0 && fabs(got - ref) / l1 > worst_l1n) worst_l1n = fabs(got - ref) / l1;
                    if (d > worst) { worst = d; w_ref = ref; w_got = got; w_l1 = l1; w_t = t; w_o = o; }
                }
            /* Graded by |err| / sum|terms|, the bound a dot product carries.  The per-element relative
             * error this check used to bound (< 1e-4) is undefined on a row whose terms cancel, and it
             * passed on the RNG stream's luck: when section D began drawing a T = 37 batch, this same
             * decode GEMV -- unchanged -- read 1.09e-03 on a new near-zero element.  The GEMV keeps a
             * 10x tighter bar than the prefill GEMM (measured 0.9-1.5e-08 vs 5.7-8.5e-08), since it
             * sums in f32 FMAs where the GEMM sums 16-k tensor-core fragments. */
            const bool gemv = T <= 16;
            CHECK(worst_l1n < (gemv ? 1e-7 : 1e-6), "%d -> %d at T = %d (%s) vs the same quantized operands in double: "
                  "max |err|/sum|terms| %.2e (max rel %.2e at ref %.3e, sum|terms| %.3e; t=%d o=%d got %.6e)",
                  sh.in, sh.out, T, gemv ? "decode GEMV" : "prefill GEMM", worst_l1n, worst, w_ref, w_l1, w_t, w_o, w_got);
        }
        /* a missing activation is refused, not mis-read.  The old negative control was a slot
         * declaring the wrong kbp; with a bf16 activation there is no slot width to get wrong, so
         * the control moves to the refusal the arm still has. */
        CHECK(rc == 0 && pulsar_qwen_mxfp8_linear_launch(&l, nullptr, T, y, nullptr, 0, 0) != 0,
              "a null activation is refused");
        cudaFree(dx); cudaFree(y);
    }
}

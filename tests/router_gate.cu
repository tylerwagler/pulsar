/* L295 F2: the shared MoE router (src/cuda/pulsar_cuda_router.cu) against a double host reference, for every router
 * the engine declares -- Qwen3.8-Flash-Next's (pulsar_qwen_router_cfg), DeepSeek V4's through its tensor binding
 * (pulsar_gpu_router_select_batch_tensor: the correction bias, the vision image-slot bias, the hash-routed token
 * table), Inkling-Small's and Nemotron 3.5 Super's -- plus, when a fixture is given, Inkling's router on REAL router weights against
 * transformers' own InklingTopkRouter (tests/inkling_router_fixture.py).
 *
 *   A  logits: bf16 rows x bf16 weight rows (+ a tail) -> f32, vs double; a row's logits the same bytes at any T
 *   B  select, per router: picks exact vs the host decision on the device's logits (a row whose k-th / (k+1)-th
 *      selection scores sit closer than the device's rounding is reported inconclusive, not graded), weights /
 *      gammas / scores vs double; batch-width neutrality (each row alone == its row in the batch, byte for byte)
 *   C  rule 9: a non-finite score row routes to experts 0..k-1 with NaN weights and NaN gammas
 *   D  refusals: a config the router does not have, a shape without an instance, a table without token ids
 *   (Nemotron 3.5 Super's 512 / top-22 sum-renormalised router rides A-C as a declared case, L296 N4)
 *   E  (fixture) Inkling-Small mini, layers 1 and 2: our logits vs transformers' fp32 logits, our picks / weights /
 *      sink gammas vs InklingTopkRouter's
 *
 * Model-free except E.  Usage: router_gate [inkling-router.bin] */
#include "pulsar_gpu.h"
#include "pulsar_cuda_qwen.h"
#include "pulsar_cuda_router.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { printf("  FAIL  "); g_fail = 1; } else printf("  ok    "); \
                           printf(__VA_ARGS__); printf("\n"); } while (0)
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
                    printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static uint64_t g_rng = 0xF2A0D7E5ULL;
static uint32_t rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 16); }
static double rndu(void) { return ((double)rnd() + 0.5) / 4294967296.0; }
static double rndn(void) { return sqrt(-2.0 * log(rndu())) * cos(6.283185307179586 * rndu()); }

static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) {   /* round to nearest even, as __float2bfloat16 */
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static double bf_round(double v) { return bf2f(f2bf((float)v)); }
static double bf_ulps(double got, double want) {
    const double ulp = std::max(fabs(want), 1e-30) * (1.0 / 128.0);
    return fabs(got - want) / ulp;
}
template <typename T> static T *up(const std::vector<T> &v) {
    T *d = nullptr;
    CK(cudaMalloc(&d, v.size() * sizeof(T) + 16));
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

/* ======================================================================== */
/* the host reference: the router's contract (pulsar_cuda_router.h) in double */

struct host_layer {
    std::vector<float> bias, alt_bias;
    int32_t alt_min = 0;
    std::vector<int32_t> table;
    int32_t table_rows = 0;
    double global_scale = 1.0;
    bool has_global = false;
};
struct host_route {
    std::vector<int32_t> sel;
    std::vector<double> w, gam, probs;
    bool inconclusive = false;   ///< a k-boundary or in-pick near-tie the device's rounding may order either way
    bool nan_route = false;
};
static double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

static host_route host_select(const pulsar_router_cfg &c, const host_layer &l, const float *lg, int32_t tok,
                              bool have_tokens) {
    const int E = c.n_expert, K = c.top_k, S = c.shared_gate == PULSAR_ROUTER_SHARED_NONE ? 0 : c.n_shared;
    host_route r;
    std::vector<double> raw(E), p(E);
    for (int e = 0; e < E; e++) raw[e] = c.logits_bf16 ? bf_round(lg[e]) : (double)lg[e];
    if (c.score == PULSAR_ROUTER_SCORE_SOFTMAX) {
        double m = -INFINITY, z = 0;
        for (int e = 0; e < E; e++) m = std::max(m, raw[e]);
        for (int e = 0; e < E; e++) z += p[e] = exp(raw[e] - m);
        for (int e = 0; e < E; e++) p[e] /= z;
    } else if (c.score == PULSAR_ROUTER_SCORE_SQRT_SOFTPLUS) {
        for (int e = 0; e < E; e++) p[e] = sqrt(raw[e] > 20 ? raw[e] : log1p(exp(raw[e])));
    } else {
        for (int e = 0; e < E; e++) p[e] = sigmoid(raw[e]);
    }
    r.probs = p;
    for (int e = 0; e < E; e++)
        if (!std::isfinite(p[e])) {
            r.nan_route = true;
            for (int k = 0; k < K; k++) { r.sel.push_back(k); r.w.push_back(NAN); }
            r.gam.assign(S, NAN);
            return r;
        }
    const bool alt = !l.alt_bias.empty() && have_tokens && tok >= l.alt_min;
    const bool sinks = c.renorm == PULSAR_ROUTER_RENORM_LOGSIGMOID_SINKS;
    std::vector<double> v;
    if (!l.table.empty() && !alt) {
        const int32_t row = tok < 0 || tok >= l.table_rows ? 0 : tok;
        for (int k = 0; k < K; k++) {
            const int32_t e = l.table[(size_t)row * K + k];
            r.sel.push_back(e);
            v.push_back(e >= 0 && e < E ? p[e] : 0.0);
        }
    } else {
        const std::vector<float> &b = alt ? l.alt_bias : l.bias;
        std::vector<double> s(E), mag(E);
        for (int e = 0; e < E; e++) {
            s[e] = b.empty() ? p[e] : p[e] + (double)b[e];
            mag[e] = b.empty() ? fabs(p[e]) : fabs(p[e]) + fabs((double)b[e]);   /* the device's rounding scale */
        }
        std::vector<int> ord(E);
        std::iota(ord.begin(), ord.end(), 0);
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int bb) { return s[a] > s[bb]; });
        for (int k = 0; k < K; k++) {
            const double gap = s[ord[k]] - s[ord[k + 1]];
            if (gap < 2e-6 * std::max(mag[ord[k]], mag[ord[k + 1]]) && !(gap == 0 && ord[k] < ord[k + 1]))
                r.inconclusive = true;   /* ranks k / k+1 closer than the device's float rounding */
            r.sel.push_back(ord[k]);
            v.push_back(sinks ? raw[ord[k]] : p[ord[k]]);
        }
    }
    if (sinks) {
        for (int j = 0; j < S; j++) v.push_back(c.logits_bf16 ? bf_round(lg[E + j]) : (double)lg[E + j]);
        double m = -INFINITY, z = 0;
        for (double &a : v) { a = std::min(a, 0.0) - log1p(exp(-fabs(a))); m = std::max(m, a); }
        for (double &a : v) z += a = exp(a - m);
        const double scale = (double)c.route_scale * (l.has_global ? l.global_scale : 1.0);
        for (size_t i = 0; i < v.size(); i++) {
            double x = v[i] / z * scale;
            if (c.weights_bf16) x = bf_round(x);
            ((int)i < K ? r.w : r.gam).push_back(x);
        }
        return r;
    }
    double sum = 0;
    for (double a : v) sum += a;
    sum = std::max(sum, (double)c.sum_floor);
    for (double a : v) r.w.push_back(c.weights_bf16 ? bf_round(a / sum * c.route_scale) : a / sum * c.route_scale);
    for (int j = 0; j < S; j++) {
        const double g = c.logits_bf16 ? bf_round(lg[E + j]) : (double)lg[E + j];
        r.gam.push_back(c.weights_bf16 ? bf_round(sigmoid(g)) : sigmoid(g));
    }
    return r;
}

/* ======================================================================== */
/* the routers under test                                                    */

struct router_case {
    const char *name;
    pulsar_router_cfg c;
    host_layer hl;
    bool ds_binding;          ///< route through DeepSeek's tensor binding instead of the launcher
    double logit_scale;
};

/* DeepSeek's declared router, stated here as the contract the binding must meet (the binding is the one place the
 * engine declares it: pulsar_cuda_hc_router.cu ds4_router_cfg). */
static pulsar_router_cfg ds_contract(int E, int K) {
    pulsar_router_cfg c{};
    c.n_expert = E; c.top_k = K; c.n_shared = 0; c.shared_gate = PULSAR_ROUTER_SHARED_NONE;
    c.score = PULSAR_ROUTER_SCORE_SQRT_SOFTPLUS; c.renorm = PULSAR_ROUTER_RENORM_SUM;
    c.logits_bf16 = false; c.weights_bf16 = false; c.sum_floor = 6.103515625e-5f; c.route_scale = 1.5f;
    return c;
}
/* Inkling-Small's router (config.json + the vLLM / transformers references): 256 routed experts, top-6, sigmoid +
 * selection bias, log-sigmoid renormalisation over the 6 routed + 2 sink logits, x route_scale 8 x global_scale,
 * f32 logits.  The family module (L294 E1) will declare this; until then the gate holds the row. */
static pulsar_router_cfg inkling_cfg(void) {
    pulsar_router_cfg c{};
    c.n_expert = 256; c.top_k = 6; c.n_shared = 2; c.shared_gate = PULSAR_ROUTER_SHARED_SINK;
    c.score = PULSAR_ROUTER_SCORE_SIGMOID; c.renorm = PULSAR_ROUTER_RENORM_LOGSIGMOID_SINKS;
    c.logits_bf16 = false; c.weights_bf16 = false; c.sum_floor = 0.0f; c.route_scale = 8.0f;
    return c;
}

/* Nemotron 3.5 Super's router (L296 N4; config.json llm_config + vLLM's NemotronHMoE): 512 routed experts, top-22,
 * sigmoid scores, the e_score_correction_bias for selection only, the picks renormalised by their sum x
 * routed_scaling_factor 5.0, f32 logits (vLLM's gate runs fp32), one UNGATED shared expert (no gate row).  Stated here
 * as the contract the family's declaration must meet (src/families/nemotron_h/nemotron_h_forward.cpp router_cfg). */
static pulsar_router_cfg nemotron_contract(void) {
    pulsar_router_cfg c{};
    c.n_expert = 512; c.top_k = 22; c.n_shared = 1; c.shared_gate = PULSAR_ROUTER_SHARED_NONE;
    c.score = PULSAR_ROUTER_SCORE_SIGMOID; c.renorm = PULSAR_ROUTER_RENORM_SUM;
    c.logits_bf16 = false; c.weights_bf16 = false; c.sum_floor = 0.0f; c.route_scale = 5.0f;
    return c;
}

/* DeepSeek's binding resolves its tensors from a model map; every case gets its own slot in it (the device-copy
 * cache is keyed by (map, offset)). */
static std::vector<uint8_t> g_map(16u << 20);
static uint64_t g_map_used = 0;
static uint64_t map_put(const void *p, size_t n) {
    const uint64_t off = (g_map_used + 255) & ~(uint64_t)255;
    if (off + n > g_map.size()) { printf("router_gate: model map full\n"); exit(2); }
    memcpy(g_map.data() + off, p, n);
    g_map_used = off + n;
    return off;
}
static const uint32_t kVocab = 1000;

struct dev_out {
    std::vector<int32_t> sel;
    std::vector<float> w, gam, probs;
    int rc;
};
/* Route T rows of host logits (row width = the cfg's logit rows) on the device. */
static dev_out route(const router_case &rc, const std::vector<float> &lg, const std::vector<int32_t> &tok, int T) {
    const pulsar_router_cfg &c = rc.c;
    const int E = c.n_expert, K = c.top_k, S = c.shared_gate == PULSAR_ROUTER_SHARED_NONE ? 0 : c.n_shared;
    const int R = pulsar_router_logit_rows(&c);
    dev_out o;
    if (rc.ds_binding) {
        const host_layer &l = rc.hl;
        const uint64_t ob = l.bias.empty() ? 0 : map_put(l.bias.data(), (size_t)E * 4);
        const uint64_t ovl = l.alt_bias.empty() ? 0 : map_put(l.alt_bias.data(), (size_t)E * 4);
        const uint64_t oh = l.table.empty() ? 0 : map_put(l.table.data(), l.table.size() * 4);
        pulsar_gpu_tensor *tl = pulsar_gpu_tensor_alloc((uint64_t)T * E * 4), *ts = pulsar_gpu_tensor_alloc((uint64_t)T * K * 4),
                          *tw = pulsar_gpu_tensor_alloc((uint64_t)T * K * 4), *tp = pulsar_gpu_tensor_alloc((uint64_t)T * E * 4),
                          *tt = pulsar_gpu_tensor_alloc((uint64_t)T * 4);
        pulsar_gpu_tensor_write(tl, 0, lg.data(), (uint64_t)T * E * 4);
        pulsar_gpu_tensor_write(tt, 0, tok.data(), (uint64_t)T * 4);
        const int ok = pulsar_gpu_router_select_batch_tensor(ts, tw, tp, g_map.data(), g_map.size(), ob, !l.bias.empty(),
                                                             oh, l.table.empty() ? 0 : (uint32_t)l.table_rows, tl, tt,
                                                             (uint32_t)E, (uint32_t)K, c.route_scale, (uint32_t)T, ovl,
                                                             kVocab, !l.alt_bias.empty());
        o.rc = ok ? 0 : -1;
        o.sel.resize((size_t)T * K); o.w.resize((size_t)T * K); o.probs.resize((size_t)T * E);
        CK(cudaDeviceSynchronize());
        pulsar_gpu_tensor_read(ts, 0, o.sel.data(), o.sel.size() * 4);
        pulsar_gpu_tensor_read(tw, 0, o.w.data(), o.w.size() * 4);
        pulsar_gpu_tensor_read(tp, 0, o.probs.data(), o.probs.size() * 4);
        pulsar_gpu_tensor_free(tl); pulsar_gpu_tensor_free(ts); pulsar_gpu_tensor_free(tw);
        pulsar_gpu_tensor_free(tp); pulsar_gpu_tensor_free(tt);
        return o;
    }
    float *dlg = up(std::vector<float>(lg.begin(), lg.begin() + (size_t)T * R));
    int32_t *dtok = up(std::vector<int32_t>(tok.begin(), tok.begin() + T));
    int32_t *dsel = (int32_t *)dalloc((size_t)T * K * 4);
    float *dw = (float *)dalloc((size_t)T * K * 4), *dg = S ? (float *)dalloc((size_t)S * T * 4) : nullptr,
          *dp = (float *)dalloc((size_t)T * E * 4);
    const host_layer &h = rc.hl;
    float *db = h.bias.empty() ? nullptr : up(h.bias), *dgs = h.has_global ? up(std::vector<float>{(float)h.global_scale}) : nullptr;
    pulsar_router_layer l{};
    l.bias = db;
    l.global_scale = dgs;
    o.rc = pulsar_router_select_launch(&c, &l, dlg, dtok, T, dsel, dw, dg, dp, 0);
    o.sel = down(dsel, (size_t)T * K);
    o.w = down(dw, (size_t)T * K);
    o.gam = S ? down(dg, (size_t)S * T) : std::vector<float>{};
    o.probs = down(dp, (size_t)T * E);
    cudaFree(dlg); cudaFree(dtok); cudaFree(dsel); cudaFree(dw); cudaFree(dg); cudaFree(dp); cudaFree(db); cudaFree(dgs);
    return o;
}

static void section_logits(void) {
    printf("A. router logits (bf16 x bf16 -> f32, fixed order)\n");
    struct shp { const char *name; int H, n_w, n_tail; };
    for (const shp &s : {shp{"Qwen 2560 x (512 + the shared gate row)", 2560, 512, 1},
                         shp{"Inkling 4096 x 258 (256 routed + 2 sink rows, one tensor)", 4096, 258, 0},
                         shp{"Nemotron 4096 x 512 (no gate rows: an ungated shared expert)", 4096, 512, 0}}) {
        const int T = 37, R = s.n_w + s.n_tail;
        std::vector<uint16_t> x((size_t)T * s.H), w((size_t)R * s.H);
        for (auto &v : x) v = f2bf((float)(0.7 * rndn()));
        for (auto &v : w) v = f2bf((float)(0.02 * rndn()));
        uint16_t *dx = up(x), *dw = up(w);
        float *dl = (float *)dalloc((size_t)T * R * 4);
        const uint16_t *tail = s.n_tail ? dw + (size_t)s.n_w * s.H : nullptr;
        int rc = pulsar_router_logits_launch(dx, T, s.H, dw, s.n_w, tail, s.n_tail, dl, 0);
        const auto L = down(dl, (size_t)T * R);
        double worst = 0;
        for (int t = 0; t < T; t++) {
            std::vector<double> ref(R);
            double mx = 0;
            for (int e = 0; e < R; e++) {
                double a = 0;
                for (int k = 0; k < s.H; k++) a += (double)bf2f(x[(size_t)t * s.H + k]) * bf2f(w[(size_t)e * s.H + k]);
                ref[e] = a;
                mx = std::max(mx, fabs(a));
            }
            for (int e = 0; e < R; e++) worst = std::max(worst, fabs(L[(size_t)t * R + e] - ref[e]) / mx);
        }
        CHECK(rc == 0 && worst < 1e-5, "%s: vs double, max rel %.2e (f32 accumulation order)", s.name, worst);
        size_t diff = 0;
        for (int Tn : {1, 8}) {
            const int t0 = 5;
            rc |= pulsar_router_logits_launch(dx + (size_t)t0 * s.H, Tn, s.H, dw, s.n_w, tail, s.n_tail, dl, 0);
            const auto L2 = down(dl, (size_t)Tn * R);
            diff += memcmp(L2.data(), &L[(size_t)t0 * R], L2.size() * 4) != 0;
        }
        CHECK(rc == 0 && diff == 0, "%s: T = 1 and T = 8 rows bit-identical to the T = %d run's", s.name, T);
        cudaFree(dx); cudaFree(dw); cudaFree(dl);
    }
    CHECK(pulsar_router_logits_launch(nullptr, 1, 2560, nullptr, 512, nullptr, 0, nullptr, 0) != 0,
          "null rows are refused");
}

static void grade(const router_case &rc, int T, const std::vector<float> &lg, const std::vector<int32_t> &tok,
                  const dev_out &o) {
    const pulsar_router_cfg &c = rc.c;
    const int E = c.n_expert, K = c.top_k, S = c.shared_gate == PULSAR_ROUTER_SHARED_NONE ? 0 : c.n_shared;
    const int R = pulsar_router_logit_rows(&c);
    size_t sel_bad = 0, w_bad = 0, g_bad = 0, p_bad = 0;
    int inconclusive = 0;
    double w_worst = 0;
    for (int t = 0; t < T; t++) {
        const host_route h = host_select(c, rc.hl, &lg[(size_t)t * R], tok[t], true);
        for (int e = 0; e < E; e++)
            p_bad += fabs(o.probs[(size_t)t * E + e] - h.probs[e]) > 2e-6 * std::max(1.0, fabs(h.probs[e]));
        if (h.inconclusive) { inconclusive++; continue; }
        for (int k = 0; k < K; k++) {
            sel_bad += o.sel[(size_t)t * K + k] != h.sel[k];
            const double got = o.w[(size_t)t * K + k], want = h.w[k];
            const double err = c.weights_bf16 ? bf_ulps(got, want) / 128.0 : fabs(got - want) / std::max(fabs(want), 1e-30);
            w_worst = std::max(w_worst, err);
            w_bad += c.weights_bf16 ? bf_ulps(got, want) > 1.0 : err > 1e-5;
        }
        for (int j = 0; j < S; j++) {
            const double got = o.gam[(size_t)j * T + t], want = h.gam[j];
            g_bad += c.weights_bf16 ? bf_ulps(got, want) > 1.0 : fabs(got - want) > 1e-5 * fabs(want);
        }
    }
    CHECK(o.rc == 0 && sel_bad == 0, "%s: picks == the host decision on the device's logits (%zu of %d differ; %d of "
          "%d rows inconclusive)", rc.name, sel_bad, (T - inconclusive) * K, inconclusive, T);
    CHECK(w_bad == 0 && g_bad == 0 && p_bad == 0, "%s: weights / gammas / scores vs double: %zu / %zu / %zu outside (%s; "
          "worst weight %.2e)", rc.name, w_bad, g_bad, p_bad, c.weights_bf16 ? "1 bf16 ulp" : "1e-5 rel", w_worst);
}

static void section_select(void) {
    printf("B. select: every declared router vs the double reference; batch-width neutrality\n");
    std::vector<router_case> cases;
    {
        router_case q{"Qwen 512 / top-10, softmax, sigmoid shared gate", pulsar_qwen_router_cfg(512, 10), {}, false, 2.0};
        cases.push_back(q);
    }
    for (int v = 0; v < 6; v++) {
        static const int E[6] = {384, 384, 256, 256, 256, 128}, K[6] = {6, 6, 6, 6, 6, 3};
        static const char *nm[6] = {"DeepSeek 384 / top-6 + bias", "DeepSeek 384 / top-6, no bias",
                                    "DeepSeek 256 / top-6 + bias + image-slot bias",
                                    "DeepSeek 256 / top-6 hash layer + image slots", "DeepSeek 256 / top-6 hash layer",
                                    "DeepSeek 128 / top-3 (DSpark drafter) + bias"};
        router_case d{nm[v], ds_contract(E[v], K[v]), {}, true, 1.0 + v * 0.4};
        if (v != 1 && v != 4) { d.hl.bias.resize(E[v]); for (auto &b : d.hl.bias) b = (float)(0.3 * rndn()); }
        if (v == 2 || v == 3) {
            d.hl.alt_bias.resize(E[v]);
            for (auto &b : d.hl.alt_bias) b = (float)(0.3 * rndn());
            d.hl.alt_min = (int32_t)kVocab;
        }
        if (v == 3 || v == 4) {
            d.hl.table_rows = (int32_t)kVocab;
            d.hl.table.resize((size_t)kVocab * K[v]);
            for (auto &e : d.hl.table) e = (int32_t)(rnd() % E[v]);
            d.hl.table[7] = -1;   /* an out-of-range id: its weight is 0 */
        }
        cases.push_back(d);
    }
    {
        router_case i{"Inkling 256 / top-6, sigmoid + bias, log-sigmoid over 6 + 2 sinks, x 8 x global", inkling_cfg(),
                      {}, false, 2.5};
        i.hl.bias.resize(256);
        for (auto &b : i.hl.bias) b = (float)(0.05 * rndn());
        i.hl.global_scale = 0.0204065;
        i.hl.has_global = true;
        cases.push_back(i);
    }
    {
        router_case n{"Nemotron 512 / top-22, sigmoid + bias, SUM x 5.0, one ungated shared", nemotron_contract(), {},
                      false, 2.0};
        n.hl.bias.resize(512);
        for (auto &b : n.hl.bias) b = (float)(0.05 * rndn());
        cases.push_back(n);
    }
    const int T = 37;
    for (const router_case &rc : cases) {
        const int R = pulsar_router_logit_rows(&rc.c), E = rc.c.n_expert, K = rc.c.top_k;
        const int S = rc.c.shared_gate == PULSAR_ROUTER_SHARED_NONE ? 0 : rc.c.n_shared;
        std::vector<float> lg((size_t)T * R);
        for (auto &v : lg) v = (float)(rc.logit_scale * rndn());
        for (int t = 0; t < T; t++) {
            lg[(size_t)t * R + 9] = lg[(size_t)t * R + 40];       /* an exact tie: the lower id ranks first */
            lg[(size_t)t * R + 11] = -30.0f;                      /* softplus's / sigmoid's far arms */
            lg[(size_t)t * R + 12] = 25.0f;
        }
        std::vector<int32_t> tok(T);
        for (int t = 0; t < T; t++) tok[t] = (t % 5 == 4) ? (int32_t)(kVocab + t % 3) : (int32_t)(rnd() % kVocab);
        tok[3] = -2;   /* a bad id on a hash layer reads row 0 */
        const dev_out o = route(rc, lg, tok, T);
        grade(rc, T, lg, tok, o);
        size_t diff = 0;
        for (int t = 0; t < T; t += 6) {
            const std::vector<float> one(lg.begin() + (size_t)t * R, lg.begin() + (size_t)(t + 1) * R);
            const dev_out o1 = route(rc, one, std::vector<int32_t>{tok[t]}, 1);
            diff += memcmp(o1.sel.data(), &o.sel[(size_t)t * K], (size_t)K * 4) != 0;
            diff += memcmp(o1.w.data(), &o.w[(size_t)t * K], (size_t)K * 4) != 0;
            diff += memcmp(o1.probs.data(), &o.probs[(size_t)t * E], (size_t)E * 4) != 0;
            for (int j = 0; j < S; j++) diff += memcmp(&o1.gam[j], &o.gam[(size_t)j * T + t], 4) != 0;
        }
        CHECK(diff == 0, "%s: each row routed alone == its row of the T = %d batch, byte for byte", rc.name, T);
        if (rc.c.renorm == PULSAR_ROUTER_RENORM_LOGSIGMOID_SINKS) {
            double worst = 0;
            const double total = rc.c.route_scale * rc.hl.global_scale;
            for (int t = 0; t < T; t++) {
                double s = 0;
                for (int k = 0; k < K; k++) s += o.w[(size_t)t * K + k];
                for (int j = 0; j < S; j++) s += o.gam[(size_t)j * T + t];
                worst = std::max(worst, fabs(s - total) / total);
            }
            CHECK(worst < 1e-5, "%s: picks + sinks share route_scale x global_scale = %.6g (worst rel %.2e)", rc.name,
                  total, worst);
        }
    }
}

static void section_nan(void) {
    printf("C. rule 9: a non-finite score row has no routing\n");
    for (const pulsar_router_cfg &c : {pulsar_qwen_router_cfg(512, 10), inkling_cfg()}) {
        const int R = pulsar_router_logit_rows(&c), K = c.top_k, S = c.n_shared;
        router_case rc{"", c, {}, false, 1.0};
        std::vector<float> lg((size_t)2 * R);
        for (auto &v : lg) v = (float)rndn();
        lg[R + 17] = NAN;
        const dev_out o = route(rc, lg, std::vector<int32_t>{0, 0}, 2);
        bool ok = o.rc == 0;
        for (int k = 0; k < K; k++) ok &= o.sel[(size_t)K + k] == k && std::isnan(o.w[(size_t)K + k]);
        for (int j = 0; j < S; j++) ok &= std::isnan(o.gam[(size_t)j * 2 + 1]) && !std::isnan(o.gam[(size_t)j * 2]);
        for (int k = 0; k < K; k++) ok &= !std::isnan(o.w[k]);
        CHECK(ok, "%s router: the NaN row routes to experts 0..%d with NaN weights and gammas; its neighbour is "
              "untouched", c.score == PULSAR_ROUTER_SCORE_SOFTMAX ? "Qwen" : "Inkling", K - 1);
    }
}

static void section_refusals(void) {
    printf("D. refusals\n");
    float *lg = (float *)dalloc(4 * 1024);
    int32_t *sel = (int32_t *)dalloc(1024), *tok = (int32_t *)dalloc(64);
    float *w = (float *)dalloc(1024), *g = (float *)dalloc(64);
    const pulsar_router_layer none{};
    pulsar_router_cfg c = inkling_cfg();
    c.renorm = PULSAR_ROUTER_RENORM_SUM;
    CHECK(pulsar_router_select_launch(&c, &none, lg, nullptr, 1, sel, w, g, nullptr, 0) != 0,
          "a sink gate without the sink renormalisation");
    c = inkling_cfg();
    CHECK(pulsar_router_select_launch(&c, &none, lg, nullptr, 1, sel, w, nullptr, nullptr, 0) != 0,
          "gated shared experts without a gamma output");
    c.n_shared = PULSAR_ROUTER_SHARED_MAX + 1;
    CHECK(pulsar_router_select_launch(&c, &none, lg, nullptr, 1, sel, w, g, nullptr, 0) != 0,
          "more shared experts than %d", PULSAR_ROUTER_SHARED_MAX);
    c = inkling_cfg();
    c.top_k = 7;
    CHECK(pulsar_router_select_launch(&c, &none, lg, nullptr, 1, sel, w, g, nullptr, 0) != 0,
          "256 / top-7: a shape without an instance");
    c = ds_contract(256, 6);
    pulsar_router_layer tab{};
    tab.table = tok;
    tab.table_rows = 4;
    CHECK(pulsar_router_select_launch(&c, &tab, lg, nullptr, 1, sel, w, nullptr, nullptr, 0) != 0,
          "a token-id table without token ids");
    c.score = (pulsar_router_score)0;
    CHECK(pulsar_router_select_launch(&c, &none, lg, nullptr, 1, sel, w, nullptr, nullptr, 0) != 0,
          "a score function the router does not have");
    cudaFree(lg); cudaFree(sel); cudaFree(tok); cudaFree(w); cudaFree(g);
}

/* E: the fixture's records (tests/inkling_router_fixture.py). */
static void section_inkling_mini(const char *path) {
    printf("E. Inkling-Small mini (real router weights) vs transformers' InklingTopkRouter: %s\n", path);
    FILE *f = fopen(path, "rb");
    if (!f) { CHECK(false, "cannot open the fixture %s", path); return; }
    int records = 0;
    for (;;) {
        char magic[8];
        if (fread(magic, 1, 8, f) != 8) break;
        int32_t hdr[7];
        float sc[2];
        if (memcmp(magic, "INKRTR01", 8) || fread(hdr, 4, 7, f) != 7 || fread(sc, 4, 2, f) != 2) {
            CHECK(false, "a malformed record");
            break;
        }
        const int layer = hdr[0], T = hdr[1], H = hdr[2], R = hdr[3], E = hdr[4], K = hdr[5], S = hdr[6];
        std::vector<uint16_t> x((size_t)T * H), W((size_t)R * H);
        std::vector<float> bias(E), rlg((size_t)T * R), rw((size_t)T * K), rg((size_t)T * S);
        std::vector<int32_t> ridx((size_t)T * K);
        bool rd = fread(x.data(), 2, x.size(), f) == x.size() && fread(W.data(), 2, W.size(), f) == W.size() &&
                  fread(bias.data(), 4, E, f) == (size_t)E && fread(rlg.data(), 4, rlg.size(), f) == rlg.size() &&
                  fread(ridx.data(), 4, ridx.size(), f) == ridx.size() && fread(rw.data(), 4, rw.size(), f) == rw.size() &&
                  fread(rg.data(), 4, rg.size(), f) == rg.size();
        if (!rd) { CHECK(false, "a truncated record"); break; }
        records++;
        pulsar_router_cfg c = inkling_cfg();
        const bool shape_ok = E == c.n_expert && K == c.top_k && S == c.n_shared && R == E + S && sc[0] == c.route_scale;
        CHECK(shape_ok, "layer %d: the checkpoint's router is the declared one (%d / top-%d + %d sinks, route_scale %g)",
              layer, E, K, S, sc[0]);
        if (!shape_ok) continue;
        uint16_t *dx = up(x), *dW = up(W);
        float *dl = (float *)dalloc((size_t)T * R * 4), *db = up(bias), *dgs = up(std::vector<float>{sc[1]});
        float *dw = (float *)dalloc((size_t)T * K * 4), *dg = (float *)dalloc((size_t)S * T * 4);
        int32_t *dsel = (int32_t *)dalloc((size_t)T * K * 4);
        pulsar_router_layer l{};
        l.bias = db;
        l.global_scale = dgs;
        int rc = pulsar_router_logits_launch(dx, T, H, dW, R, nullptr, 0, dl, 0);
        if (!rc) rc = pulsar_router_select_launch(&c, &l, dl, nullptr, T, dsel, dw, dg, nullptr, 0);
        const auto L = down(dl, (size_t)T * R);
        const auto Sel = down(dsel, (size_t)T * K);
        const auto Wt = down(dw, (size_t)T * K);
        const auto G = down(dg, (size_t)S * T);
        double lworst = 0;
        for (int t = 0; t < T; t++) {
            double mx = 0;
            for (int e = 0; e < R; e++) mx = std::max(mx, (double)fabsf(rlg[(size_t)t * R + e]));
            for (int e = 0; e < R; e++) lworst = std::max(lworst, fabs(L[(size_t)t * R + e] - rlg[(size_t)t * R + e]) / mx);
        }
        CHECK(rc == 0 && lworst < 1e-5, "layer %d: logits vs transformers' fp32 F.linear, max rel %.2e", layer, lworst);
        int set_bad = 0, inconclusive = 0;
        double wworst = 0, gworst = 0;
        for (int t = 0; t < T; t++) {
            /* the reference's own selection scores on its own logits: a k-boundary closer than float rounding is
             * inconclusive (torch.topk and the warp argmax may order it either way) */
            std::vector<double> s(E);
            for (int e = 0; e < E; e++) s[e] = sigmoid(rlg[(size_t)t * R + e]) + bias[e];
            std::vector<double> ss(s);
            std::sort(ss.begin(), ss.end(), std::greater<double>());
            if (ss[K - 1] - ss[K] < 1e-6) { inconclusive++; continue; }
            for (int k = 0; k < K; k++) {
                const int32_t e = Sel[(size_t)t * K + k];
                int at = -1;
                for (int kk = 0; kk < K; kk++) if (ridx[(size_t)t * K + kk] == e) at = kk;
                if (at < 0) { set_bad++; continue; }
                const double want = rw[(size_t)t * K + at];
                wworst = std::max(wworst, fabs(Wt[(size_t)t * K + k] - want) / fabs(want));
            }
            for (int j = 0; j < S; j++) {
                const double want = rg[(size_t)t * S + j];
                gworst = std::max(gworst, fabs(G[(size_t)j * T + t] - want) / fabs(want));
            }
        }
        CHECK(set_bad == 0, "layer %d: our picks == InklingTopkRouter's (as sets: it takes topk(sorted=False)); %d of %d "
              "differ, %d of %d rows inconclusive", layer, set_bad, (T - inconclusive) * K, inconclusive, T);
        CHECK(wworst < 1e-4 && gworst < 1e-4, "layer %d: routed weights / sink gammas vs the reference: worst rel %.2e / "
              "%.2e (bar 1e-4: f32 logits in another summation order, fast-math exp / log1p)", layer, wworst, gworst);
        cudaFree(dx); cudaFree(dW); cudaFree(dl); cudaFree(db); cudaFree(dgs); cudaFree(dw); cudaFree(dg); cudaFree(dsel);
    }
    fclose(f);
    CHECK(records > 0, "%d fixture records read", records);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (pulsar_gpu_init() == 0) { printf("ROUTER FAIL: no GPU\n"); return 2; }
    printf("router-gate: the shared MoE router (pulsar_cuda_router.cu) vs a double reference\n");
    section_logits();
    section_select();
    section_nan();
    section_refusals();
    if (argc > 1) section_inkling_mini(argv[1]);
    else printf("E. SKIPPED: no Inkling fixture given -- the real-weights check did NOT run\n");
    printf(g_fail ? "ROUTER GATE FAIL\n" : "ROUTER GATE PASS\n");
    return g_fail ? 1 : 0;
}

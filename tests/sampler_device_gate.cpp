/* sampler-device-gate (L302): the device sampler (pulsar_gpu_sample_rows, _logprobs_rows, _spec_walk) against the
 * host sampler, byte for byte.  Needs a GPU, no model.
 *
 * WHAT IS PINNED, per row: the token (or the refusal), the emitted distribution -- its size, every id in order and
 * every probability's bits -- and, for the logprob kernel, the scored token's and the top-20's (id, logit, logprob)
 * bits; per verify run, the walk's commit and carry.  The reference is the host sampler as the engine compiled it
 * (tests/sampler_host_ref.cpp, frozen at L302; while the engine still carried its own copy this gate held all three
 * equal).  The device runs every batch twice and must give the same bytes (no race in its reductions).
 *
 * WHY THESE SHAPES.  Every candidate path the host had: greedy (temperature <= 0 and the GREEDY flag), top-k (k =
 * 1 .. 1024, clamps), the min-p prefilter (full-nucleus arm and the id-order-sum arm), the full sort (min_p tiny),
 * the row kernel's shared store and the slot kernel's global sort (flat and near-flat rows overflow it); knob
 * clamps and non-finite knobs; rows with NaN / +-inf holes, no finite value, -0 / +0 and exact ties (the order and
 * the argmax's lowest id), probabilities in the subnormal range (wide rows at low temperature), two tied leaders;
 * the token mask (dense random, a tool-name-like handful, everything masked); row strides wider than the sampled
 * width (a draft vocabulary inside a wider row).  Every case at each vocabulary a family serves.
 *
 *   ./tests/sampler_device_gate [--quick] [--seed N] [--bench | --bench-only] */
#include "pulsar_gpu.h"
#include "sampler_host_ref.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr uint32_t ROWS = PULSAR_GPU_SAMPLE_ROWS_MAX;
constexpr uint32_t DIST_CAP = 4096;

struct case_row {
    std::vector<float> x;      /* the stored row (stride wide) */
    uint32_t stride = 0;
    std::vector<uint32_t> mask;   /* empty = no mask */
    pulsar_gpu_sample_req req{};
    const char *shape = "";
};

std::mt19937_64 g_rng(0x5eed1302ull);

float gauss() {
    static std::normal_distribution<float> n(0.0f, 1.0f);
    return n(g_rng);
}
uint32_t urand(uint32_t n) { return (uint32_t)(g_rng() % n); }
float frand() { return (float)((g_rng() >> 40) * (1.0 / 16777216.0)); }

const char *k_shapes[] = {"lm", "peaked", "flat", "near-flat", "holes", "no-finite", "signed-zeros", "tied-leaders",
                          "wide", "quantized", "few-finite"};
constexpr uint32_t N_SHAPES = sizeof(k_shapes) / sizeof(k_shapes[0]);

void gen_row(uint32_t shape, uint32_t n, std::vector<float> &x) {
    x.resize(n);
    const float NI = -INFINITY, PI = INFINITY, NN = NAN;
    switch (shape) {
    case 0:   /* an LM-like row: a broad body and a few strong leaders */
        for (uint32_t i = 0; i < n; i++) x[i] = 2.0f * gauss() - 3.0f;
        for (int j = 0; j < 8; j++) x[urand(n)] = 6.0f + 9.0f * frand();
        break;
    case 1:
        for (uint32_t i = 0; i < n; i++) x[i] = gauss();
        x[urand(n)] = 30.0f;
        break;
    case 2: {
        const float c = 1.5f;
        for (uint32_t i = 0; i < n; i++) x[i] = c;
        break;
    }
    case 3:
        for (uint32_t i = 0; i < n; i++) x[i] = 0.01f * gauss();
        break;
    case 4:
        for (uint32_t i = 0; i < n; i++) x[i] = 2.0f * gauss();
        for (uint32_t j = 0; j < n / 20; j++) {
            const uint32_t k = urand(3);
            x[urand(n)] = k == 0 ? NN : k == 1 ? PI : NI;
        }
        break;
    case 5:
        for (uint32_t i = 0; i < n; i++) x[i] = (i & 1) ? NI : NN;
        break;
    case 6: {
        const float v[4] = {0.0f, -0.0f, 1.0f, -1.0f};
        for (uint32_t i = 0; i < n; i++) x[i] = v[urand(4)];
        break;
    }
    case 7: {
        for (uint32_t i = 0; i < n; i++) x[i] = 2.0f * gauss();
        const uint32_t a = urand(n), b = urand(n);
        x[a] = 9.0f;
        x[b] = 9.0f;
        x[urand(n)] = 8.9999990f;
        break;
    }
    case 8:
        for (uint32_t i = 0; i < n; i++) x[i] = 30.0f * gauss();
        break;
    case 9:
        for (uint32_t i = 0; i < n; i++) x[i] = std::round(2.0f * gauss() * 2.0f) / 2.0f;
        break;
    default:
        for (uint32_t i = 0; i < n; i++) x[i] = NI;
        for (int j = 0; j < 10; j++) x[urand(n)] = 3.0f * gauss();
        break;
    }
}

void gen_mask(uint32_t kind, uint32_t n, const std::vector<float> &x, std::vector<uint32_t> &m) {
    const uint32_t words = (n + 31u) / 32u;
    m.assign(words, 0u);
    auto allow = [&](uint32_t i) { m[i >> 5] |= 1u << (i & 31u); };
    if (kind == 0) {   /* dense random: about 3/4 allowed, the argmax among them or not */
        for (uint32_t i = 0; i < n; i++)
            if (urand(4) != 0) allow(i);
    } else if (kind == 1) {   /* a tool name's handful */
        for (int j = 0; j < 40; j++) allow(urand(n));
    } else if (kind == 2) {   /* everything but the row's best few */
        for (uint32_t i = 0; i < n; i++) allow(i);
        uint32_t best = 0;
        for (uint32_t i = 1; i < n; i++)
            if (std::isfinite(x[i]) && (!std::isfinite(x[best]) || x[i] > x[best])) best = i;
        m[best >> 5] &= ~(1u << (best & 31u));
    }   /* kind 3: nothing allowed */
}

void gen_req(pulsar_gpu_sample_req &q) {
    static const float temps[] = {0.0f, -1.0f, 0.05f, 0.1f, 0.3f, 0.6f, 0.7f, 1.0f, 1.0f, 1.3f, 2.0f, 5.0f};
    static const int topks[] = {0, 0, 0, 0, 1, 2, 5, 20, 40, 100, 1024, 2000, -3};
    static const float topps[] = {1.0f, 1.0f, 1.0f, 0.95f, 0.9f, 0.8f, 0.5f, 0.1f, 0.0f, 1.5f, 0.999f};
    static const float minps[] = {0.05f, 0.05f, 0.0f, 0.0f, 1e-31f, 1e-20f, 0.01f, 0.02f, 0.1f, 0.5f, 1.0f, 2.0f,
                                  -0.5f, 0.06f};
    q.temperature = temps[urand(sizeof(temps) / sizeof(temps[0]))];
    q.top_k = topks[urand(sizeof(topks) / sizeof(topks[0]))];
    q.top_p = topps[urand(sizeof(topps) / sizeof(topps[0]))];
    q.min_p = minps[urand(sizeof(minps) / sizeof(minps[0]))];
    if (urand(40) == 0) {   /* a non-finite knob */
        const uint32_t w = urand(3);
        (w == 0 ? q.temperature : w == 1 ? q.top_p : q.min_p) = urand(2) ? NAN : INFINITY;
    }
    q.key = g_rng();
    q.flags = urand(12) == 0 ? PULSAR_GPU_SAMPLE_GREEDY : 0u;
    q.mask_row = PULSAR_GPU_SAMPLE_NO_MASK;
    q.probe = -1;
}

/* the host's verdict on one row: what sample_top_p_min_p (or the walk's argmax for a GREEDY row) gives, and the
 * distribution it drew from */
struct host_out {
    int tok = -1;
    std::vector<int> ids;
    std::vector<float> probs;
};

/* `map` (NULL = identity): a drafter's draft vocabulary -- the dist is built over the row, then its ids mapped, then
 * drawn (family_qwen.cpp's qwen_mtp_dist, as it stood) */
host_out host_sample(const float *row, uint32_t n, const pulsar_gpu_sample_req &q, sref::pulsar_sample_scratch *sc,
                     const int32_t *map = nullptr) {
    host_out h;
    if (q.flags & PULSAR_GPU_SAMPLE_GREEDY) {
        h.tok = sref::sample_argmax(row, n);
        if (h.tok >= 0 && map) h.tok = map[h.tok];
        if (h.tok >= 0) { h.ids = {h.tok}; h.probs = {1.0f}; }
        return h;
    }
    sref::pulsar_sample_dist d;
    if (!sref::pulsar_sample_dist_build(row, n, q.temperature, q.top_k, q.top_p, q.min_p, sc, &d)) return h;
    if (map)
        for (uint32_t i = 0; i < d.n; i++) d.ids[i] = map[d.ids[i]];
    h.tok = q.temperature <= 0.0f ? d.ids[0] : sref::pulsar_sample_dist_draw(&d, q.key);
    h.ids.assign(d.ids, d.ids + d.n);
    h.probs.assign(d.probs, d.probs + d.n);
    sref::pulsar_sample_dist_free(&d);
    return h;
}

uint32_t fbits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

struct gate {
    uint32_t cases = 0, fails = 0, overflow_rows = 0;
    uint32_t by_path[4] = {};
    void fail(const char *what, uint32_t v, const case_row &c, uint32_t r, const char *detail) {
        if (fails++ < 25)
            fprintf(stderr, "  FAIL %s  V=%u shape=%s T=%g k=%d p=%g minp=%g greedy=%u mask=%s row %u: %s\n", what, v,
                    c.shape, c.req.temperature, c.req.top_k, c.req.top_p, c.req.min_p,
                    c.req.flags & PULSAR_GPU_SAMPLE_GREEDY, c.mask.empty() ? "no" : "yes", r, detail);
    }
};

/* One batch of up to ROWS rows on the device: sample (twice), dist, logprobs, against the host. */
bool run_batch(gate &g, uint32_t n_vocab, std::vector<case_row> &rows, pulsar_gpu_tensor *dl, pulsar_gpu_tensor *dm,
               pulsar_gpu_tensor *scratch, pulsar_gpu_tensor *dout, pulsar_gpu_tensor *ddist, pulsar_gpu_tensor *dlp,
               bool check_lp, const std::vector<int32_t> *map, pulsar_gpu_tensor *dmap) {
    const int32_t *hm = map ? map->data() : nullptr;
    if (map && !pulsar_gpu_tensor_write(dmap, 0, map->data(), map->size() * 4)) return false;
    pulsar_gpu_tensor *idm = map ? dmap : nullptr;
    if (map) check_lp = false;   /* logprobs score the row's own ids */
    const uint32_t nr = (uint32_t)rows.size();
    const uint32_t stride = rows[0].stride, words = (n_vocab + 31u) / 32u;
    std::vector<float> host((size_t)nr * stride);
    std::vector<uint32_t> mask((size_t)nr * words, 0u);
    std::vector<pulsar_gpu_sample_req> reqs(nr);
    for (uint32_t r = 0; r < nr; r++) {
        memcpy(&host[(size_t)r * stride], rows[r].x.data(), (size_t)stride * 4);
        reqs[r] = rows[r].req;
        reqs[r].row = r;
        if (!rows[r].mask.empty()) {
            memcpy(&mask[(size_t)r * words], rows[r].mask.data(), (size_t)words * 4);
            reqs[r].mask_row = r;
        }
    }
    if (!pulsar_gpu_tensor_write(dl, 0, host.data(), host.size() * 4) ||
        !pulsar_gpu_tensor_write(dm, 0, mask.data(), mask.size() * 4))
        return false;
    /* each row probes a token whose emitted probability it reports: its argmax, a random id, or none */
    for (uint32_t r = 0; r < nr; r++) {
        const uint32_t w = urand(3);
        int32_t p = w == 0 ? -1 : w == 1 ? (int32_t)urand(n_vocab) : sref::sample_argmax(rows[r].x.data(), n_vocab);
        if (p >= 0 && hm) p = hm[p];
        reqs[r].probe = p;
    }
    std::vector<pulsar_gpu_sample_out> out1(nr), out2(nr);
    std::vector<uint32_t> dist((size_t)nr * (1 + 2 * DIST_CAP));
    if (!pulsar_gpu_sample_rows(dl, 0, stride, n_vocab, nr, nr, reqs.data(), dm, words, idm, scratch, dout, 0, ddist,
                                DIST_CAP) ||
        !pulsar_gpu_synchronize() || !pulsar_gpu_tensor_read(dout, 0, out1.data(), out1.size() * sizeof(out1[0])) ||
        !pulsar_gpu_tensor_read(ddist, 0, dist.data(), dist.size() * 4))
        return false;
    if (!pulsar_gpu_sample_rows(dl, 0, stride, n_vocab, nr, nr, reqs.data(), dm, words, idm, scratch, dout, 0, nullptr, 0) ||
        !pulsar_gpu_synchronize() || !pulsar_gpu_tensor_read(dout, 0, out2.data(), out2.size() * sizeof(out2[0])))
        return false;
    /* logprobs of each row's host token (the masked view when the row is masked) */
    std::vector<pulsar_gpu_logprob_row> lp(nr);
    std::vector<uint32_t> mrows(nr);
    for (uint32_t r = 0; r < nr; r++) mrows[r] = reqs[r].mask_row;
    if (check_lp && (!pulsar_gpu_logprobs_rows(dl, 0, stride, n_vocab, nr, dm, words, mrows.data(), dout, 0, sizeof(pulsar_gpu_sample_out) / 4,
                                               PULSAR_GPU_LOGPROB_TOPK_MAX, dlp) ||
                     !pulsar_gpu_synchronize() || !pulsar_gpu_tensor_read(dlp, 0, lp.data(), lp.size() * sizeof(lp[0]))))
        return false;
    std::vector<host_out> ho(nr);
    std::vector<std::vector<float>> view(nr);
#pragma omp parallel for schedule(dynamic)
    for (uint32_t r = 0; r < nr; r++) {
        sref::pulsar_sample_scratch sc;
        memset(&sc, 0, sizeof(sc));
        view[r].assign(rows[r].x.begin(), rows[r].x.begin() + n_vocab);
        if (!rows[r].mask.empty())
            for (uint32_t i = 0; i < n_vocab; i++)
                if (!((rows[r].mask[i >> 5] >> (i & 31u)) & 1u)) view[r][i] = -INFINITY;
        ho[r] = host_sample(view[r].data(), n_vocab, reqs[r], &sc, hm);
        sref::pulsar_sample_scratch_free(&sc);
    }
    char det[256];
    for (uint32_t r = 0; r < nr; r++) {
        g.cases++;
        const int32_t t1 = out1[r].token, t2 = out2[r].token;
        const uint32_t st = out1[r].status;
        const host_out &h = ho[r];
        if (t1 != t2 || out2[r].status != st || fbits(out1[r].probe_p) != fbits(out2[r].probe_p)) {
            snprintf(det, sizeof(det), "two launches disagree (%d vs %d)", t1, t2);
            g.fail("determinism", n_vocab, rows[r], r, det);
        }
        const bool dev_ok = st == PULSAR_GPU_SAMPLE_OK && t1 >= 0;
        if (dev_ok != (h.tok >= 0) || (dev_ok && t1 != h.tok)) {
            snprintf(det, sizeof(det), "token: device %d (status %u), host %d", t1, st, h.tok);
            g.fail("token", n_vocab, rows[r], r, det);
            continue;
        }
        if (!dev_ok) continue;
        const uint32_t *d = &dist[(size_t)r * (1 + 2 * DIST_CAP)];
        if (d[0] != h.ids.size()) {
            snprintf(det, sizeof(det), "dist size: device %u, host %zu", d[0], h.ids.size());
            g.fail("dist", n_vocab, rows[r], r, det);
            continue;
        }
        for (uint32_t i = 0; i < d[0] && i < DIST_CAP; i++) {
            if ((int)d[1 + i] != h.ids[i] || d[1 + DIST_CAP + i] != fbits(h.probs[i])) {
                snprintf(det, sizeof(det), "dist[%u] of %u: device (%u, %08x) host (%d, %08x)", i, d[0], d[1 + i],
                         d[1 + DIST_CAP + i], h.ids[i], fbits(h.probs[i]));
                g.fail("dist", n_vocab, rows[r], r, det);
                break;
            }
        }
        if (h.ids.size() > 2048) g.overflow_rows++;
        {   /* L301's label: the probe's emitted probability (pulsar_sample_dist_prob: the first match, 0 if absent) */
            float hp = 0.0f;
            for (size_t i = 0; i < h.ids.size(); i++)
                if (h.ids[i] == reqs[r].probe) { hp = h.probs[i]; break; }
            if (fbits(hp) != fbits(out1[r].probe_p)) {
                snprintf(det, sizeof(det), "probe %d p: device %08x host %08x", reqs[r].probe,
                         fbits(out1[r].probe_p), fbits(hp));
                g.fail("probe", n_vocab, rows[r], r, det);
            }
        }
        if (!check_lp) continue;
        sref::pulsar_token_score top[PULSAR_GPU_LOGPROB_TOPK_MAX], tk;
        const int nt = sref::pulsar_logits_top_logprobs(view[r].data(), (int)n_vocab, top,
                                                        (int)PULSAR_GPU_LOGPROB_TOPK_MAX);
        const int ht = sref::pulsar_logits_token_logprob(view[r].data(), (int)n_vocab, h.tok, &tk);
        const pulsar_gpu_logprob_row &L = lp[r];
        if ((uint32_t)nt != L.n_top) {
            snprintf(det, sizeof(det), "top count: device %u, host %d", L.n_top, nt);
            g.fail("logprobs", n_vocab, rows[r], r, det);
            continue;
        }
        for (int i = 0; i < nt; i++) {
            if (L.top[i].id != top[i].id || fbits(L.top[i].logit) != fbits(top[i].logit) ||
                fbits(L.top[i].logprob) != fbits(top[i].logprob)) {
                snprintf(det, sizeof(det), "top[%d]: device (%d, %08x, %08x) host (%d, %08x, %08x)", i, L.top[i].id,
                         fbits(L.top[i].logit), fbits(L.top[i].logprob), top[i].id, fbits(top[i].logit),
                         fbits(top[i].logprob));
                g.fail("logprobs", n_vocab, rows[r], r, det);
                break;
            }
        }
        if ((ht != 0) != (L.have_token != 0) ||
            (ht && (L.token.id != tk.id || fbits(L.token.logit) != fbits(tk.logit) ||
                    fbits(L.token.logprob) != fbits(tk.logprob)))) {
            snprintf(det, sizeof(det), "token logprob: device (%u: %d, %08x) host (%d: %d, %08x)", L.have_token,
                     L.token.id, fbits(L.token.logprob), ht, tk.id, fbits(tk.logprob));
            g.fail("logprobs", n_vocab, rows[r], r, det);
        }
    }
    return true;
}

/* The walk: runs over rows the sampler drew, drafts that agree for a while and then do not. */
bool run_walk(gate &g, uint32_t n_vocab, pulsar_gpu_tensor *dl, pulsar_gpu_tensor *scratch, pulsar_gpu_tensor *dout,
              pulsar_gpu_tensor *dwalk) {
    std::vector<float> host((size_t)ROWS * n_vocab);
    std::vector<pulsar_gpu_sample_req> reqs(ROWS);
    std::vector<uint32_t> run_len;
    for (uint32_t r = 0; r < ROWS; r++) {
        std::vector<float> x;
        gen_row(urand(2) ? 0u : 1u, n_vocab, x);
        memcpy(&host[(size_t)r * n_vocab], x.data(), (size_t)n_vocab * 4);
        gen_req(reqs[r]);
        reqs[r].row = r;
        reqs[r].temperature = urand(3) ? 0.7f : 0.0f;
        if (!std::isfinite(reqs[r].top_p) || !std::isfinite(reqs[r].min_p)) reqs[r].top_p = reqs[r].min_p = 0.05f;
    }
    if (!pulsar_gpu_tensor_write(dl, 0, host.data(), host.size() * 4)) return false;
    /* the host draws first: the drafts are built from its tokens, and each row probes its draft as a verify does */
    std::vector<host_out> ho(ROWS);
    for (uint32_t r = 0; r < ROWS; r++) {
        sref::pulsar_sample_scratch sc;
        memset(&sc, 0, sizeof(sc));
        if (urand(5) == 0) reqs[r].flags |= PULSAR_GPU_SAMPLE_GREEDY;
        ho[r] = host_sample(&host[(size_t)r * n_vocab], n_vocab, reqs[r], &sc);
        sref::pulsar_sample_scratch_free(&sc);
    }
    std::vector<pulsar_gpu_spec_run> runs;
    for (uint32_t r0 = 0; r0 < ROWS;) {
        const uint32_t k = std::min<uint32_t>(urand(9), ROWS - r0 - 1u);
        pulsar_gpu_spec_run run{};
        run.row0 = r0;
        run.k = k;
        const uint32_t agree = urand(k + 1);
        for (uint32_t j = 0; j < k; j++) {
            /* a miss drafts the row's runner-up when it has one, else any other id */
            const host_out &h = ho[r0 + j];
            run.pend[j] = j < agree ? h.tok : h.ids.size() > 1 && h.ids[1] != h.tok ? h.ids[1]
                                                                                      : (h.tok + 1) % (int)n_vocab;
            reqs[r0 + j].probe = run.pend[j];
            const bool greedy = (reqs[r0 + j].flags & PULSAR_GPU_SAMPLE_GREEDY) || reqs[r0 + j].temperature <= 0.0f;
            run.greedy_rows |= greedy ? 1u << j : 0u;
        }
        runs.push_back(run);
        r0 += k + 1;
    }
    if (!pulsar_gpu_sample_rows(dl, 0, n_vocab, n_vocab, ROWS, ROWS, reqs.data(), nullptr, 0, nullptr, scratch, dout, 0, nullptr, 0))
        return false;
    std::vector<int> tok(ROWS);
    for (uint32_t r = 0; r < ROWS; r++) tok[r] = ho[r].tok;
    std::vector<pulsar_gpu_spec_walk_out> wo(runs.size());
    if (!pulsar_gpu_spec_walk(dout, 0, ROWS, runs.data(), (uint32_t)runs.size(), dwalk, 0) ||
        !pulsar_gpu_synchronize() || !pulsar_gpu_tensor_read(dwalk, 0, wo.data(), wo.size() * sizeof(wo[0])))
        return false;
    for (size_t i = 0; i < runs.size(); i++) {
        /* the host walk with L301's labels (7e6392a2: greedy row -> acceptance, sampled row -> p_target(draft)) */
        uint32_t c = 0, nl = 0;
        float label[16];
        uint8_t hit[16];
        while (c < runs[i].k) {
            const host_out &h = ho[runs[i].row0 + c];
            const bool ok = h.tok == runs[i].pend[c];
            float p = 0.0f;
            for (size_t j = 0; j < h.ids.size(); j++)
                if (h.ids[j] == runs[i].pend[c]) { p = h.probs[j]; break; }
            label[c] = (runs[i].greedy_rows >> c) & 1u ? (ok ? 1.0f : 0.0f) : p;
            hit[c] = ok;
            nl = c + 1;
            if (!ok) break;
            c++;
        }
        const int carry = tok[runs[i].row0 + c];
        bool labels_ok = wo[i].n_label == nl;
        for (uint32_t j = 0; labels_ok && j < nl; j++)
            labels_ok = fbits(wo[i].label[j]) == fbits(label[j]) && wo[i].hit[j] == hit[j];
        g.cases++;
        if (wo[i].commit != c || wo[i].carry != carry || wo[i].status != PULSAR_GPU_SAMPLE_OK || !labels_ok) {
            g.fails++;
            fprintf(stderr, "  FAIL walk V=%u run %zu (k %u): device commit %u carry %d status %u labels %u%s, host %u %d %u\n",
                    n_vocab, i, runs[i].k, wo[i].commit, wo[i].carry, wo[i].status, wo[i].n_label,
                    labels_ok ? "" : " (differ)", c, carry, nl);
        }
    }
    return true;
}

void bench(uint32_t n_vocab, pulsar_gpu_tensor *dl, pulsar_gpu_tensor *scratch, pulsar_gpu_tensor *dout) {
    std::vector<float> host((size_t)ROWS * n_vocab);
    for (uint32_t r = 0; r < ROWS; r++) {
        std::vector<float> x;
        gen_row(0, n_vocab, x);
        memcpy(&host[(size_t)r * n_vocab], x.data(), (size_t)n_vocab * 4);
    }
    pulsar_gpu_tensor_write(dl, 0, host.data(), host.size() * 4);
    struct mode { const char *name; float t; int k; float p, m; } modes[] = {
        {"greedy", 0.0f, 0, 1.0f, 0.05f}, {"T0.7 min_p 0.05", 0.7f, 0, 1.0f, 0.05f},
        {"T0.7 top-k 20 top-p 0.95", 0.7f, 20, 0.95f, 0.0f}, {"T0.7 top-p 0.9 min_p 0.05", 0.7f, 0, 0.9f, 0.05f},
        {"T1.0 min_p 0 (full sort)", 1.0f, 0, 1.0f, 0.0f}};
    for (const mode &m : modes) {
        for (uint32_t nr : {1u, 5u, 33u}) {
            std::vector<pulsar_gpu_sample_req> reqs(nr);
            for (uint32_t r = 0; r < nr; r++) {
                reqs[r] = pulsar_gpu_sample_req{m.t, m.p, m.m, m.k, 0x1234ull + r, 0u, PULSAR_GPU_SAMPLE_NO_MASK, -1, r};
            }
            const int it = 20;
            pulsar_gpu_sample_rows(dl, 0, n_vocab, n_vocab, nr, nr, reqs.data(), nullptr, 0, nullptr, scratch, dout, 0, nullptr, 0);
            pulsar_gpu_synchronize();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; i++)
                pulsar_gpu_sample_rows(dl, 0, n_vocab, n_vocab, nr, nr, reqs.data(), nullptr, 0, nullptr, scratch, dout, 0, nullptr,
                                       0);
            pulsar_gpu_synchronize();
            const double us =
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
            printf("  bench V=%u %-28s %2u rows: %8.1f us a launch\n", n_vocab, m.name, nr, us);
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    bool quick = false, do_bench = false, bench_only = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--quick")) quick = true;
        else if (!strcmp(argv[i], "--bench")) do_bench = true;
        else if (!strcmp(argv[i], "--bench-only")) do_bench = bench_only = true;
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) g_rng.seed(strtoull(argv[++i], nullptr, 0));
        else { fprintf(stderr, "usage: %s [--quick] [--seed N] [--bench | --bench-only]\n", argv[0]); return 2; }
    }
    if (!pulsar_gpu_init()) { fprintf(stderr, "sampler-device-gate: no GPU\n"); return 1; }
    /* every vocabulary a family serves, Qwen's draft vocabulary, and a small one */
    static const uint32_t vocabs[] = {248320u, 129280u, 201024u, 82196u, 4099u};
    const uint32_t batches = bench_only ? 0u : quick ? 6u : 40u;
    gate g;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t v : vocabs) {
        const uint32_t max_stride = v + 77u;
        pulsar_gpu_tensor *dl = pulsar_gpu_tensor_alloc((uint64_t)ROWS * max_stride * 4);
        pulsar_gpu_tensor *dm = pulsar_gpu_tensor_alloc((uint64_t)ROWS * ((v + 31u) / 32u) * 4);
        pulsar_gpu_tensor *scratch = pulsar_gpu_tensor_alloc(pulsar_gpu_sample_scratch_bytes(v));
        pulsar_gpu_tensor *dout = pulsar_gpu_tensor_alloc(ROWS * sizeof(pulsar_gpu_sample_out));
        pulsar_gpu_tensor *ddist = pulsar_gpu_tensor_alloc((uint64_t)ROWS * (1 + 2 * DIST_CAP) * 4);
        pulsar_gpu_tensor *dlp = pulsar_gpu_tensor_alloc(ROWS * sizeof(pulsar_gpu_logprob_row));
        pulsar_gpu_tensor *dwalk = pulsar_gpu_tensor_alloc(ROWS * sizeof(pulsar_gpu_spec_walk_out));
        pulsar_gpu_tensor *dmap = pulsar_gpu_tensor_alloc((uint64_t)v * 4u);
        if (!dl || !dm || !scratch || !dout || !ddist || !dlp || !dwalk || !dmap) { fprintf(stderr, "alloc failed\n"); return 1; }
        const uint32_t f0 = g.fails, c0 = g.cases;
        for (uint32_t b = 0; b < batches; b++) {
            const uint32_t stride = b % 5 == 4 ? v + 77u : v;
            std::vector<case_row> rows(ROWS);
            for (uint32_t r = 0; r < ROWS; r++) {
                case_row &c = rows[r];
                const uint32_t shape = (b * ROWS + r) % N_SHAPES;
                c.shape = k_shapes[shape];
                std::vector<float> x;
                gen_row(shape, v, x);
                c.stride = stride;
                c.x.assign(stride, 777.0f);   /* the columns past the sampled width are never read */
                memcpy(c.x.data(), x.data(), (size_t)v * 4);
                gen_req(c.req);
                if (urand(5) == 0) gen_mask(urand(10) == 0 ? 3u : urand(3), v, x, c.mask);
            }
            /* every seventh batch draws through a draft vocabulary: an ascending subset of a vocabulary twice as wide */
            std::vector<int32_t> map;
            if (b % 7 == 6) {
                map.resize(v);
                int32_t at = 0;
                for (uint32_t i = 0; i < v; i++) map[i] = (at += 1 + (int32_t)urand(2));
            }
            if (!run_batch(g, v, rows, dl, dm, scratch, dout, ddist, dlp, b % 2 == 0, map.empty() ? nullptr : &map,
                           dmap)) {
                fprintf(stderr, "sampler-device-gate: a launch failed\n");
                return 1;
            }
            if (!run_walk(g, v, dl, scratch, dout, dwalk)) {
                fprintf(stderr, "sampler-device-gate: a walk launch failed\n");
                return 1;
            }
        }
        printf("  V=%-6u %5u cases, %u failures\n", v, g.cases - c0, g.fails - f0);
        if (do_bench) bench(v, dl, scratch, dout);
        pulsar_gpu_tensor_free(dl); pulsar_gpu_tensor_free(dm); pulsar_gpu_tensor_free(scratch);
        pulsar_gpu_tensor_free(dout); pulsar_gpu_tensor_free(ddist); pulsar_gpu_tensor_free(dlp);
        pulsar_gpu_tensor_free(dwalk);
        pulsar_gpu_tensor_free(dmap);
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("sampler-device-gate: %u cases (%u rows past the shared store), %u failures, %.1f s -- %s\n", g.cases,
           g.overflow_rows, g.fails, s, g.fails ? "FAIL" : "PASS");
    return g.fails ? 1 : 0;
}

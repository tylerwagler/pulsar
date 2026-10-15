/* Head options gate (L295 E5) -- model-free, links the PRODUCTION object
 * (src/cuda/pulsar_cuda_head.o).
 *
 *   1. embed norm   rows of a random table normed against the DOUBLE reference:
 *                   TOLERANCE-graded (relative Frobenius, share equal to
 *                   bf16(reference), worst ulp); a row's bytes are the same
 *                   alone, in a batch and repeated (no batch dependence).
 *   2. embed plain  without a norm the rows are copied as stored, bit-exact.
 *   3. stray id     an id outside the table writes a NaN row, never another
 *                   row, and leaves its batchmates' bytes as they were.
 *   4. logits       Inkling's head (201024 rows, 200058 real, scale 1/16) on
 *                   strided rows: every real logit == logit * scale (f32,
 *                   bit-exact against the host product), every pad -inf, the
 *                   stride's slack untouched; with the pads pre-loaded LARGER
 *                   than any real logit the argmax is still a real id.
 *   5. identity     a head that declares nothing launches nothing.
 * Then refusals.  --fixture DIR grades both against the Inkling-Small mini's
 * transformers fp32 capture (tests/head_mini_fixture.py). */
#include "cuda/pulsar_cuda_head.h"
#include "sconv_ref.h"   /* sref_f2bf / sref_bf2f: the bf16 conversions the block gates share */

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

static int bf_ulps(uint16_t a, uint16_t b) {
    auto ord = [](uint16_t v) { return (v & 0x8000u) ? -(int)(v & 0x7fffu) : (int)v; };
    return abs(ord(a) - ord(b));
}

struct Grade { double fro = 0, share = 0; int worst = 0; };
static Grade grade_bf(const uint16_t *y, const double *ref, size_t n) {
    Grade g;
    double num = 0, den = 0;
    size_t eq = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)sref_bf2f(y[i]) - ref[i];
        num += d * d; den += ref[i] * ref[i];
        const uint16_t rb = sref_f2bf((float)ref[i]);
        eq += rb == y[i];
        g.worst = std::max(g.worst, bf_ulps(rb, y[i]));
    }
    g.fro = sqrt(num / std::max(den, 1e-300));
    g.share = (double)eq / (double)n;
    return g;
}

/* rsqrtf is within 2 ulp, so an output may sit one bf16 ulp off the correctly rounded reference */
static const double kEmbedFroBar = 3e-3, kEmbedShareBar = 0.99;

static void embed_tests() {
    printf("\n== embedding norm ==\n");
    const uint32_t V = 1000, H = 4096, n = 64;
    Rng g(5);
    std::vector<uint16_t> table((size_t)V * H), w(H);
    for (auto &v : table) v = sref_f2bf((float)(0.02 * g.n()));
    for (auto &v : w) v = sref_f2bf((float)(1.0 + 0.25 * g.n()));
    std::vector<int32_t> ids(n);
    for (uint32_t i = 0; i < n; ++i) ids[i] = (int32_t)(mix(i * 31 + 7) % V);
    ids[5] = ids[40] = 17; ids[0] = 0; ids[n - 1] = (int32_t)V - 1;
    uint16_t *d_table, *d_w, *d_out, *d_one;
    int32_t *d_ids;
    const uint64_t ld = H + 8;
    CK(cudaMalloc(&d_table, table.size() * 2)); CK(cudaMemcpy(d_table, table.data(), table.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_w, H * 2)); CK(cudaMemcpy(d_w, w.data(), H * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_ids, n * 4)); CK(cudaMemcpy(d_ids, ids.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_out, n * ld * 2)); CK(cudaMalloc(&d_one, ld * 2));
    pulsar_head_decl h{H, d_w, 1e-6f, 1, 1, 1.0f};
    check(pulsar_head_embed_launch(&h, d_table, V, d_ids, n, d_out, ld, 0) == 0, "a batch of %u ids (repeats, first and last row)", n);
    std::vector<uint16_t> out((size_t)n * ld);
    CK(cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<uint16_t> y((size_t)n * H);
    std::vector<double> ref((size_t)n * H);
    for (uint32_t t = 0; t < n; ++t) {
        const uint16_t *x = &table[(size_t)ids[t] * H];
        double ss = 0;
        for (uint32_t i = 0; i < H; ++i) ss += (double)sref_bf2f(x[i]) * sref_bf2f(x[i]);
        const double rstd = 1.0 / sqrt(ss / H + 1e-6);
        for (uint32_t i = 0; i < H; ++i) {
            ref[(size_t)t * H + i] = (double)sref_bf2f(x[i]) * rstd * sref_bf2f(w[i]);
            y[(size_t)t * H + i] = out[(size_t)t * ld + i];
        }
    }
    const Grade gr = grade_bf(y.data(), ref.data(), y.size());
    check(gr.fro <= kEmbedFroBar && gr.share >= kEmbedShareBar && gr.worst <= 1,
          "fidelity vs double: fro %.3e (bar %.0e), == bf16(ref) %.3f%%, worst %d ulp", gr.fro, kEmbedFroBar, 100 * gr.share, gr.worst);
    check(memcmp(&y[5 * H], &y[40 * H], H * 2) == 0, "a repeated id's rows are identical");
    bool alone = true;
    for (uint32_t t = 0; t < n && alone; t += 7) {
        CK(cudaMemset(d_one, 0, ld * 2));
        alone = pulsar_head_embed_launch(&h, d_table, V, d_ids + t, 1, d_one, ld, 0) == 0;
        std::vector<uint16_t> one(H);
        CK(cudaMemcpy(one.data(), d_one, H * 2, cudaMemcpyDeviceToHost));
        alone = alone && memcmp(one.data(), &y[(size_t)t * H], H * 2) == 0;
    }
    check(alone, "a row embedded alone == the same row in the batch, bit-exact");
    /* mutation: the reference without the norm weight must fail the grade */
    {
        std::vector<double> nw(ref.size());
        for (uint32_t t = 0; t < n; ++t)
            for (uint32_t i = 0; i < H; ++i) nw[(size_t)t * H + i] = ref[(size_t)t * H + i] / sref_bf2f(w[i]);
        const Grade m = grade_bf(y.data(), nw.data(), y.size());
        check(m.fro >= 20 * kEmbedFroBar, "mutation (norm weight dropped) fails the grade: fro %.3e", m.fro);
    }
    /* plain: no norm, copied as stored */
    pulsar_head_decl plain = h;
    plain.embed_norm_w = nullptr;
    check(pulsar_head_embed_launch(&plain, d_table, V, d_ids, n, d_out, ld, 0) == 0, "a head without an embedding norm");
    CK(cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost));
    bool copy = true;
    for (uint32_t t = 0; t < n && copy; ++t) copy = memcmp(&out[(size_t)t * ld], &table[(size_t)ids[t] * H], H * 2) == 0;
    check(copy, "without a norm every row is the table's row, bit-exact");
    /* a stray id */
    std::vector<int32_t> bad(ids.begin(), ids.begin() + 4);
    bad[2] = (int32_t)V;
    CK(cudaMemcpy(d_ids, bad.data(), 16, cudaMemcpyHostToDevice));
    check(pulsar_head_embed_launch(&h, d_table, V, d_ids, 4, d_out, ld, 0) == 0, "a batch with an id past the table");
    CK(cudaMemcpy(out.data(), d_out, (size_t)4 * ld * 2, cudaMemcpyDeviceToHost));
    bool nanrow = true, others = true;
    for (uint32_t i = 0; i < H; ++i) nanrow = nanrow && std::isnan(sref_bf2f(out[2 * ld + i]));
    for (uint32_t t : {0u, 1u, 3u}) others = others && memcmp(&out[(size_t)t * ld], &y[(size_t)t * H], H * 2) == 0;
    check(nanrow && others, "the stray id's row is NaN and its batchmates' bytes are unchanged");
    cudaFree(d_table); cudaFree(d_w); cudaFree(d_out); cudaFree(d_one); cudaFree(d_ids);
}

static void logits_tests() {
    printf("\n== logits finish (Inkling: 201024 rows, 200058 real, scale 1/16) ==\n");
    const uint32_t NL = 201024, NR = 200058, rows = 5;
    const uint64_t ld = NL + 40;
    Rng g(9);
    std::vector<float> in((size_t)rows * ld);
    for (size_t i = 0; i < in.size(); ++i) in[i] = (float)(3.0 * g.n());
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = NR; c < NL; ++c) in[(size_t)r * ld + c] = 1e30f;   /* pads louder than any real logit */
    float *d;
    CK(cudaMalloc(&d, in.size() * 4));
    CK(cudaMemcpy(d, in.data(), in.size() * 4, cudaMemcpyHostToDevice));
    pulsar_head_decl h{4096, nullptr, 0.0f, NL, NR, 1.0f / 16.0f};
    check(pulsar_head_logits_finish_launch(&h, d, rows, ld, 0) == 0, "%u strided rows", rows);
    std::vector<float> out(in.size());
    CK(cudaMemcpy(out.data(), d, out.size() * 4, cudaMemcpyDeviceToHost));
    bool real = true, pads = true, slack = true, top = true;
    for (uint32_t r = 0; r < rows; ++r) {
        const float *a = &in[(size_t)r * ld], *b = &out[(size_t)r * ld];
        int best = -1;
        for (uint32_t c = 0; c < NR; ++c) {
            real = real && b[c] == a[c] * (1.0f / 16.0f);   /* host f32 product: IEEE, no fast-math on the host */
        }
        for (uint32_t c = NR; c < NL; ++c) pads = pads && std::isinf(b[c]) && b[c] < 0;
        for (uint64_t c = NL; c < ld; ++c) slack = slack && memcmp(&b[c], &a[c], 4) == 0;
        float bv = 0;
        for (uint32_t c = 0; c < NL; ++c)
            if (std::isfinite(b[c]) && (best < 0 || b[c] > bv)) { bv = b[c]; best = (int)c; }
        top = top && best >= 0 && (uint32_t)best < NR;
    }
    check(real, "every real logit == logit * 1/16 (the f32 product), bit-exact");
    check(pads, "every pad logit (ids 200058 .. 201023) is -inf");
    check(slack, "the stride's slack past 201024 is untouched");
    check(top, "with the pads pre-loaded at 1e30 the finite argmax is a real id");
    /* identity: nothing declared, nothing launched */
    CK(cudaMemcpy(d, in.data(), in.size() * 4, cudaMemcpyHostToDevice));
    pulsar_head_decl id{4096, nullptr, 0.0f, NL, NL, 1.0f};
    check(pulsar_head_logits_finish_launch(&id, d, rows, ld, 0) == 0, "a head that declares nothing");
    CK(cudaMemcpy(out.data(), d, out.size() * 4, cudaMemcpyDeviceToHost));
    check(memcmp(out.data(), in.data(), in.size() * 4) == 0, "... leaves the logits as the GEMM wrote them");
    cudaFree(d);
}

static void refusals() {
    printf("\n== refusals ==\n");
    float *d; uint16_t *t; int32_t *ids;
    CK(cudaMalloc(&d, 1 << 20)); CK(cudaMalloc(&t, 1 << 20)); CK(cudaMalloc(&ids, 64));
    CK(cudaMemset(ids, 0, 64));
    const pulsar_head_decl base{64, t, 1e-6f, 1000, 900, 0.5f};
    auto lg = [&](const char *what, pulsar_head_decl h, uint32_t n, uint64_t ld) {
        check(pulsar_head_logits_finish_launch(&h, d, n, ld, 0) == -1, "refused: %s", what);
    };
    auto em = [&](const char *what, pulsar_head_decl h, uint64_t ld, const uint16_t *tab) {
        check(pulsar_head_embed_launch(&h, tab, 16, ids, 4, t + 4096, ld, 0) == -1, "refused: %s", what);
    };
    check(pulsar_head_logits_finish_launch(&base, d, 4, 1000, 0) == 0 && pulsar_head_embed_launch(&base, t, 16, ids, 4, t + 4096, 64, 0) == 0,
          "the base calls are accepted");
    pulsar_head_decl h = base;
    h.n_real = 0; lg("no real ids", h, 1, 1000);
    h = base; h.n_real = 1001; lg("more real ids than logits", h, 1, 1000);
    h = base; h.logit_scale = 0.0f; lg("a zero scale", h, 1, 1000);
    h = base; h.logit_scale = INFINITY; lg("an infinite scale", h, 1, 1000);
    lg("a row stride below the logits", base, 1, 999);
    lg("no rows", base, 0, 1000);
    h = base; h.n_embd = 60; em("an embedding width not a multiple of 8", h, 64, t);
    em("an output stride below the width", base, 56, t);
    em("a misaligned table", base, 64, t + 1);
    h = base; h.embed_norm_w = t + 1; em("a misaligned norm weight", h, 64, t);
    h = base; h.embed_norm_eps = -1.0f; em("a negative epsilon", h, 64, t);
    CK(cudaDeviceSynchronize());
    cudaFree(d); cudaFree(t); cudaFree(ids);
}

/* ---- the mini fixture ------------------------------------------------------------------------------- */
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
    const std::string D(dir);
    std::vector<uint8_t> meta;
    if (!read_file(D + "/meta.txt", &meta)) return 2;
    meta.push_back(0);
    unsigned T = 0, R = 0, H = 0, NV = 0, NR = 0;
    double mup = 0, eps = 0;
    if (sscanf((const char *)meta.data(), "tokens %u rows %u n_embd %u n_vocab %u n_real %u mup %lf eps %lf", &T, &R, &H, &NV, &NR,
               &mup, &eps) != 7) { fprintf(stderr, "meta.txt\n"); return 2; }
    printf("== the Inkling-Small mini: embedding norm and logits finish vs transformers fp32 ==\n");
    std::vector<uint8_t> ids, table, nw, eref, raw, lref;
    if (!read_file(D + "/emb.ids.bin", &ids) || !read_file(D + "/emb.table.bin", &table) || !read_file(D + "/emb.norm_w.bin", &nw) ||
        !read_file(D + "/emb.ref32.bin", &eref) || !read_file(D + "/head.raw.bin", &raw) || !read_file(D + "/head.ref.bin", &lref))
        return 2;
    const uint32_t U = (uint32_t)(table.size() / 2 / H);
    uint16_t *d_table, *d_w, *d_out;
    int32_t *d_ids;
    CK(cudaMalloc(&d_table, table.size())); CK(cudaMemcpy(d_table, table.data(), table.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_w, nw.size())); CK(cudaMemcpy(d_w, nw.data(), nw.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_ids, ids.size())); CK(cudaMemcpy(d_ids, ids.data(), ids.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_out, (size_t)T * H * 2));
    const pulsar_head_decl h{H, d_w, (float)eps, NV, NR, (float)(1.0 / mup)};
    bool ok = pulsar_head_embed_launch(&h, d_table, U, d_ids, T, d_out, H, 0) == 0;
    std::vector<uint16_t> y((size_t)T * H);
    CK(cudaMemcpy(y.data(), d_out, y.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<double> ref((size_t)T * H);
    for (size_t i = 0; i < ref.size(); ++i) ref[i] = ((const float *)eref.data())[i];
    const Grade g = grade_bf(y.data(), ref.data(), y.size());
    check(ok && g.fro <= kEmbedFroBar && g.worst <= 1,
          "embedding norm, %u tokens: fro %.3e vs transformers fp32 (bf16 output), == bf16(transformers) %.3f%%, worst %d ulp",
          T, g.fro, 100 * g.share, g.worst);
    float *d_l;
    CK(cudaMalloc(&d_l, raw.size()));
    CK(cudaMemcpy(d_l, raw.data(), raw.size(), cudaMemcpyHostToDevice));
    ok = pulsar_head_logits_finish_launch(&h, d_l, R, NV, 0) == 0;
    std::vector<float> lg((size_t)R * NV);
    CK(cudaMemcpy(lg.data(), d_l, lg.size() * 4, cudaMemcpyDeviceToHost));
    const float *lr = (const float *)lref.data();
    double num = 0, den = 0;
    bool pads = true;
    unsigned top_agree = 0;
    for (unsigned r = 0; r < R; ++r) {
        int bi = -1, ri = 0;
        float bv = 0, rv = lr[(size_t)r * NR];
        for (unsigned c = 0; c < NR; ++c) {
            const double d = (double)lg[(size_t)r * NV + c] - lr[(size_t)r * NR + c];
            num += d * d; den += (double)lr[(size_t)r * NR + c] * lr[(size_t)r * NR + c];
            if (lr[(size_t)r * NR + c] > rv) { rv = lr[(size_t)r * NR + c]; ri = (int)c; }
        }
        for (unsigned c = 0; c < NV; ++c) {
            const float v = lg[(size_t)r * NV + c];
            if (c >= NR) pads = pads && std::isinf(v) && v < 0;
            else if (bi < 0 || v > bv) { bv = v; bi = (int)c; }
        }
        top_agree += bi == ri;
    }
    const double fro = sqrt(num / den);
    check(ok && fro <= 1e-5, "logits, %u rows: (GEMM) x 1/%g vs transformers' (hidden / %g) x unembed: fro %.3e -- the scale is applied "
          "at runtime on unscaled weights", R, mup, mup, fro);
    check(ok && pads, "every pad logit (%u .. %u) is -inf", NR, NV - 1);
    check(ok && top_agree == R, "top-1 agrees with the reference on %u / %u rows", top_agree, R);
    cudaFree(d_table); cudaFree(d_w); cudaFree(d_out); cudaFree(d_ids); cudaFree(d_l);
    printf("\n%s\n", g_fail ? "HEAD MINI GRADE FAILED" : "HEAD MINI GRADE PASS");
    return g_fail;
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--fixture")) return fixture(argv[2]);
    if (argc != 1) { fprintf(stderr, "usage: %s [--fixture DIR]\n", argv[0]); return 2; }
    printf("head gate (L295 E5): the family head options\n");
    embed_tests();
    logits_tests();
    refusals();
    printf("\n%s\n", g_fail ? "HEAD GATE FAILED" : "HEAD GATE PASS");
    return g_fail;
}

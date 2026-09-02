/* L149 gate: the device min-p prefilter (pulsar_gpu_minp_prefilter_rows) must
 * equal its host emulation bit-for-bit. Needs a GPU, no model.
 *
 * WHAT IS PINNED. For every row: the reported count (all candidates, even past
 * the cap), the row max id under the host scan's rule (first finite maximum,
 * lowest id on ties), the max logit bits, and -- when the count fits the cap
 * -- the candidate ids in ascending id order with their logits. The host side
 * is the same emulation pulsar_test --sampler feeds to
 * pulsar_sample_dist_build_prefiltered, so the two gates compose: device ==
 * emulation here, emulation-fed build == full-row build there.
 *
 * WHY THESE SHAPES. Ties exactly AT the threshold exercise the `>=`; NaN, +inf
 * and -inf holes exercise the finite filter (an +inf would otherwise be the
 * max); an all-non-finite row must report count 0 and max id -1; a flat row
 * against a deep floor must overflow the cap and leave the arrays untouched;
 * strided multi-row batches check the row addressing the verify path will use.
 *
 * Also prints the per-launch time for a 9-row batch: the number the L149
 * price (~630 us of host time per draft position) is being traded for. */
#include "pulsar_gpu.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

enum { N_VOCAB = 129280, CAP = 2048, ROW_I32 = 3 + 2 * CAP };

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 11);
}
static float rndf(void) { return (float)rnd() / 4294967296.0f; }

/* Host emulation -- the contract pulsar_sample_dist_build_prefiltered relies on. */
static void emulate(const float *x, uint32_t n, float delta, int32_t *hdr,
                    std::vector<int32_t> &ids, std::vector<float> &vals) {
    float mx = 0.0f;
    int32_t mi = -1;
    for (uint32_t i = 0; i < n; i++) {
        const float v = x[i];
        if (!std::isfinite(v)) continue;
        if (mi < 0 || v > mx) { mx = v; mi = (int32_t)i; }
    }
    ids.clear(); vals.clear();
    if (mi < 0) { hdr[0] = 0; hdr[1] = -1; hdr[2] = 0; return; }
    const float thr = mx + delta;
    for (uint32_t i = 0; i < n; i++) {
        const float v = x[i];
        if (std::isfinite(v) && v >= thr) { ids.push_back((int32_t)i); vals.push_back(v); }
    }
    hdr[0] = (int32_t)ids.size();
    hdr[1] = mi;
    memcpy(&hdr[2], &mx, sizeof(float));
}

/* shapes */
static void fill(float *x, uint32_t n, int shape, float delta) {
    for (uint32_t i = 0; i < n; i++) {
        /* rough gaussian, sd ~3, from 4 uniforms */
        x[i] = 3.0f * ((rndf() + rndf() + rndf() + rndf()) - 2.0f) * 1.7f;
    }
    switch (shape) {
    case 0: /* peaked: one clear max */
        x[rnd() % n] = 40.0f;
        break;
    case 1: /* flat: everything within 0.5 -> a deep floor takes the whole row */
        for (uint32_t i = 0; i < n; i++) x[i] = 0.5f * rndf();
        break;
    case 2: { /* ties at the max (lowest id must win) and ties exactly at the floor */
        const uint32_t a = 5000, b = 90000, c = 120000;
        x[a] = x[b] = x[c] = 25.0f;
        const float thr = 25.0f + delta;         /* the device computes mx + delta in float too */
        for (uint32_t k = 0; k < 64; k++) x[(k * 2017u + 13u) % n] = thr;   /* AT the threshold: kept */
        for (uint32_t k = 0; k < 64; k++) x[(k * 3011u + 29u) % n] = nextafterf(thr, -INFINITY); /* 1 ulp under: cut */
        break;
    }
    case 3: /* holes: NaN, +inf (must NOT be the max), -inf */
        for (uint32_t k = 0; k < 500; k++) x[rnd() % n] = NAN;
        for (uint32_t k = 0; k < 200; k++) x[rnd() % n] = INFINITY;
        for (uint32_t k = 0; k < 200; k++) x[rnd() % n] = -INFINITY;
        x[777] = 30.0f;
        break;
    case 4: /* all non-finite */
        for (uint32_t i = 0; i < n; i++) x[i] = (i & 1) ? NAN : -INFINITY;
        break;
    case 5: /* single finite */
        for (uint32_t i = 0; i < n; i++) x[i] = -INFINITY;
        x[n - 1] = -3.0f;
        break;
    case 6: /* signed zeros: -0.0 == +0.0 to the compare; max among zeros is the lowest id */
        for (uint32_t i = 0; i < n; i++) x[i] = -50.0f;
        x[10] = -0.0f; x[20] = 0.0f; x[30] = -0.0f;
        break;
    default: break;
    }
}

int main(void) {
    if (pulsar_gpu_init() == 0) {
        fprintf(stderr, "minp_prefilter_gate: no GPU\n");
        return 2;
    }
    struct Case { const char *name; uint32_t rows; uint32_t stride; float delta; int shape; };
    const float d05 = (float)(1.0 * (log(0.05) - 1e-3));   /* T=1, min_p 0.05: the production floor */
    const Case cases[] = {
        {"peaked, 1 row",                 1, N_VOCAB,       d05,    0},
        {"peaked, 9 rows strided",        9, N_VOCAB + 64,  d05,    0},
        {"peaked, 16 rows, T=0.3",       16, N_VOCAB,       0.3f * d05, 0},
        {"flat vs deep floor (overflow)", 1, N_VOCAB,       -15.0f, 1},
        {"ties at max and at floor",      3, N_VOCAB + 8,   d05,    2},
        {"nan/inf holes",                 4, N_VOCAB,       d05,    3},
        {"all non-finite",                1, N_VOCAB,       d05,    4},
        {"single finite",                 2, N_VOCAB,       d05,    5},
        {"signed zeros, delta 0",         2, N_VOCAB,       0.0f,   6},
    };
    int failures = 0, checked_rows = 0;
    std::vector<int32_t> eids; std::vector<float> evals;
    for (const Case &c : cases) {
        std::vector<float> host((size_t)c.rows * c.stride, -INFINITY);
        for (uint32_t r = 0; r < c.rows; r++) fill(host.data() + (size_t)r * c.stride, N_VOCAB, c.shape, c.delta);
        std::vector<int32_t> out((size_t)c.rows * ROW_I32, (int32_t)0xDEADBEEF);
        pulsar_gpu_tensor *dx = pulsar_gpu_tensor_alloc((uint64_t)host.size() * sizeof(float));
        pulsar_gpu_tensor *dout = pulsar_gpu_tensor_alloc((uint64_t)out.size() * sizeof(int32_t));
        if (!dx || !dout ||
            !pulsar_gpu_tensor_write(dx, 0, host.data(), (uint64_t)host.size() * sizeof(float)) ||
            !pulsar_gpu_tensor_write(dout, 0, out.data(), (uint64_t)out.size() * sizeof(int32_t)) ||
            !pulsar_gpu_minp_prefilter_rows(dout, dx, 0, c.rows, c.stride, N_VOCAB, c.delta, CAP) ||
            !pulsar_gpu_end_commands() ||
            !pulsar_gpu_tensor_read(dout, 0, out.data(), (uint64_t)out.size() * sizeof(int32_t))) {
            printf("FAIL %-32s: launch/copy failed\n", c.name);
            failures++;
            continue;
        }
        int bad = 0;
        for (uint32_t r = 0; r < c.rows; r++) {
            int32_t hdr[3];
            emulate(host.data() + (size_t)r * c.stride, N_VOCAB, c.delta, hdr, eids, evals);
            const int32_t *o = out.data() + (size_t)r * ROW_I32;
            if (o[0] != hdr[0] || o[1] != hdr[1] || o[2] != hdr[2]) {
                printf("  row %u header: device (%d,%d,0x%08x) host (%d,%d,0x%08x)\n", r,
                       o[0], o[1], (unsigned)o[2], hdr[0], hdr[1], (unsigned)hdr[2]);
                bad++;
                continue;
            }
            if (hdr[0] > 0 && (uint32_t)hdr[0] <= CAP) {
                if (memcmp(o + 3, eids.data(), (size_t)hdr[0] * sizeof(int32_t)) != 0 ||
                    memcmp(o + 3 + CAP, evals.data(), (size_t)hdr[0] * sizeof(float)) != 0) {
                    printf("  row %u: candidate ids/logits differ (count %d)\n", r, hdr[0]);
                    bad++;
                }
            } else if ((uint32_t)hdr[0] > CAP) {
                /* overflow: the arrays must be untouched */
                for (uint32_t k = 0; k < 2 * CAP; k++)
                    if (o[3 + k] != (int32_t)0xDEADBEEF) { bad++; printf("  row %u: overflow wrote arrays\n", r); break; }
            }
            checked_rows++;
        }
        printf("%s %-32s rows=%u count[0]=%d max_id[0]=%d\n", bad ? "FAIL" : "ok  ", c.name, c.rows,
               out[0], out[1]);
        failures += bad;
        /* timing on the 9-row batch */
        if (c.rows == 9) {
            const int reps = 50;
            pulsar_gpu_end_commands();
            auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < reps; k++)
                pulsar_gpu_minp_prefilter_rows(dout, dx, 0, c.rows, c.stride, N_VOCAB, c.delta, CAP);
            pulsar_gpu_end_commands();
            auto t1 = std::chrono::steady_clock::now();
            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
            printf("     9-row prefilter: %.1f us per launch (%.1f us per row)\n", us, us / 9.0);
        }
        pulsar_gpu_tensor_free(dx);
        pulsar_gpu_tensor_free(dout);
    }
    printf("MINP PREFILTER GATE: %s (%d rows checked, %d failures)\n", failures ? "FAIL" : "PASS",
           checked_rows, failures);
    return failures ? 1 : 0;
}

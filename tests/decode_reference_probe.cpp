/* L260 PROBE: decode rows graded against the B300 reference, at each width served.
 *
 * Tyler's decision (2026-10-01): decode rows may take width-dependent arithmetic
 * -- the fastest arm at each width.  Rule 7 says that is graded, not argued, and
 * the reference gate grades PREFILL rows only.  This probe grades DECODE rows on
 * the same blobs: for each reference depth d and width w, bank 0 is prefilled
 * classically to d - w, then ONE decode step carries the w teacher-forced tokens
 * at positions d - w .. d - 1 as a verify-shaped run (decode row kind, every row
 * headed), and its LAST row -- the prediction after d tokens -- is compared with
 * the reference row for d:
 *   KL(ref || ours) over the softmax, top-1 agreement,
 *   and KL / max|d| against the SAME tokens over the SAME prefill decoded as w
 *   one-row steps -- the width effect alone (an earlier version compared with a
 *   w = 1 run whose prefill went to d - 1, i.e. a different KV prefix, and
 *   reported 2-7 logits of "width" effect that was prefix arithmetic).
 *
 *   PULSAR_MSEQ_BANKS=2 ./tests/decode_reference_probe MODEL REF.bin TOKENS.bin [w,w,...]
 * Report only: it prints, it does not enforce (the gate built on it will).
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "gate_entry.h"
#include "gate_fixture.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "DS4PFXG1"
#define MAX_DEPTHS 8u
#define REF_LEN 24u
typedef struct {
    char     magic[8];
    uint32_t version;
    uint32_t n_depths;
    uint32_t width;
    uint32_t reserved;
    uint64_t prompt_fnv;
    uint32_t depths[MAX_DEPTHS];
    char     build_ref[REF_LEN];
} blob_header;

static double kl_of(const float *ref, const float *cur, int n) {
    double mr = -INFINITY, mc = -INFINITY;
    for (int i = 0; i < n; i++) { if (ref[i] > mr) mr = ref[i]; if (cur[i] > mc) mc = cur[i]; }
    double zr = 0, zc = 0;
    for (int i = 0; i < n; i++) { zr += exp((double)ref[i] - mr); zc += exp((double)cur[i] - mc); }
    const double lzr = mr + log(zr), lzc = mc + log(zc);
    double kl = 0;
    for (int i = 0; i < n; i++) {
        const double lp = (double)ref[i] - lzr, lq = (double)cur[i] - lzc;
        kl += exp(lp) * (lp - lq);
    }
    return kl < 0 ? 0 : kl;
}
static int argmax_of(const float *x, int n) {
    int a = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[a]) a = i;
    return a;
}
static double maxabs_of(const float *a, const float *b, int n) {
    double m = 0;
    for (int i = 0; i < n; i++) { const double d = fabs((double)a[i] - b[i]); if (d > m) m = d; }
    return m;
}

/* the last row of the w tokens ending at depth d over a prefill to d - w: one
 * w-wide decode step, or (serial) w one-row steps -- into `out` */
static bool decode_last_row(pulsar_engine *e, const int *toks, uint32_t d, uint32_t w, float *out, int width,
                            bool serial) {
    pulsar_session *s = NULL;
    const int ctx = (int)(((d + 4095u) / 4096u + 1u) * 4096u);
    if (pulsar_session_create(&s, e, ctx) != 0) { fprintf(stderr, "session create failed (ctx %d)\n", ctx); return false; }
    bool ok = gate_pool_fits(s, 2) && gate_populate_bank(s, 0, toks, (int)(d - w), NULL, "decode reference prefix");
    pulsar_multiseq_req rq[16];
    for (uint32_t j = 0; j < w; j++) { rq[j].bank = 0; rq[j].pos = (int32_t)(d - w + j); rq[j].token = toks[d - w + j]; }
    float *lg = (float *)malloc((size_t)w * (size_t)width * sizeof(float));
    uint32_t nr = 0;
    char err[256];
    const uint32_t steps = serial ? w : 1u, per = serial ? 1u : w;
    for (uint32_t k = 0; ok && k < steps; k++) {
        if (pulsar_session_decode_mixed(s, rq + k * per, per, lg, (int)(per * (uint32_t)width), &nr,
                                        PULSAR_MSEQ_HEAD_ALL_ROWS, err, sizeof err) != 0) {
            fprintf(stderr, "decode step (d %u, w %u, %s %u): %s\n", d, w, serial ? "serial" : "wide", k, err);
            ok = false;
        }
        if (ok && nr != per) { fprintf(stderr, "decode step headed %u rows, want %u\n", nr, per); ok = false; }
    }
    if (ok) memcpy(out, lg + (size_t)(per - 1) * (size_t)width, (size_t)width * sizeof(float));
    free(lg);
    pulsar_session_free(s);
    return ok;
}

int GATE_ENTRY(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL REF.bin TOKENS.bin [w,w,...]\n", argv[0]);
        return 2;
    }
    uint32_t widths[8] = {1, 4, 9, 16};
    int n_w = 4;
    if (argc > 4) {
        n_w = 0;
        for (char *p = strtok(argv[4], ","); p && n_w < 8; p = strtok(NULL, ",")) widths[n_w++] = (uint32_t)atoi(p);
    }
    for (int i = 0; i < n_w; i++) {
        if (widths[i] < 1 || widths[i] > 16) { fprintf(stderr, "width %u out of [1, 16]\n", widths[i]); return 2; }
    }
    FILE *fp = fopen(argv[2], "rb");
    blob_header h;
    if (!fp || fread(&h, sizeof h, 1, fp) != 1 || memcmp(h.magic, MAGIC, 8) || h.n_depths == 0 ||
        h.n_depths > MAX_DEPTHS || h.width == 0) {
        fprintf(stderr, "reference %s unreadable or not a prefill reference blob\n", argv[2]);
        if (fp) fclose(fp);
        return 1;
    }
    float *ref = (float *)malloc((size_t)h.n_depths * h.width * sizeof(float));
    if (fread(ref, sizeof(float), (size_t)h.n_depths * h.width, fp) != (size_t)h.n_depths * h.width) {
        fprintf(stderr, "reference %s: short body\n", argv[2]);
        fclose(fp);
        return 1;
    }
    fclose(fp);
    size_t tb = 0;
    char *traw = gate_read_file(argv[3], &tb);
    if (!traw || tb % sizeof(int) || tb / sizeof(int) < h.depths[h.n_depths - 1]) {
        fprintf(stderr, "tokens %s unreadable or too short for depth %u\n", argv[3], h.depths[h.n_depths - 1]);
        return 1;
    }
    const int *toks = (const int *)traw;

    pulsar_engine *e = NULL;
    pulsar_engine_options o;
    memset(&o, 0, sizeof o);
    o.model_path = argv[1];
    o.backend = PULSAR_BACKEND_CUDA;
    o.prefill_chunk = 4096;
    if (gate_engine_open(&e, &o) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    const int width = pulsar_engine_logits_width(e);
    const int ncmp = (int)h.width < width ? (int)h.width : width;
    float *one = (float *)malloc((size_t)width * sizeof(float));
    float *cur = (float *)malloc((size_t)width * sizeof(float));
    printf("decode reference probe: %s (engine '%.*s', %u depths)\n", argv[2], (int)REF_LEN, h.build_ref, h.n_depths);
    printf("  depth  w   KL(ref||ours)  top1(ref/ours)   KL vs serial   max|d| vs serial\n");
    double sum_kl[8] = {0}, sum_kw[8] = {0};
    int flips[8] = {0};
    int rc = 0;
    for (uint32_t i = 0; i < h.n_depths && rc == 0; i++) {
        const uint32_t d = h.depths[i];
        const float *rr = ref + (size_t)i * h.width;
        for (int k = 0; k < n_w && rc == 0; k++) {
            const uint32_t w = widths[k];
            if (d < w + 1) continue;
            if (!decode_last_row(e, toks, d, w, cur, width, false)) { rc = 1; break; }
            if (w > 1 && !decode_last_row(e, toks, d, w, one, width, true)) { rc = 1; break; }
            if (w == 1) memcpy(one, cur, (size_t)width * sizeof(float));
            const float *dst = cur;
            const double kl = kl_of(rr, dst, ncmp);
            const int ar = argmax_of(rr, ncmp), ao = argmax_of(dst, ncmp);
            const double klw = kl_of(one, dst, ncmp);
            const double mw = maxabs_of(one, dst, ncmp);
            printf("  %5u %2u   %.3e      %6d/%-6d %s  %.3e   %.3e\n", d, w, kl, ar, ao, ar == ao ? " " : "*", klw, mw);
            sum_kl[k] += kl;
            sum_kw[k] += klw;
            flips[k] += ar != ao;
        }
    }
    if (rc == 0) {
        printf("  summary (mean over %u depths):\n", h.n_depths);
        for (int k = 0; k < n_w; k++)
            printf("    w=%2u  mean KL(ref||ours) %.3e  top1 misses %d  mean KL vs serial %.3e\n", widths[k],
                   sum_kl[k] / h.n_depths, flips[k], sum_kw[k] / h.n_depths);
    }
    free(one);
    free(cur);
    free(ref);
    free(traw);
    gate_engine_close(e);
    return rc;
}

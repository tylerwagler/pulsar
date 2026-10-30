/* L260 / L262 GATE: decode rows graded against the B300 reference, at each width served.
 *
 * Tyler's decision (2026-10-01): decode rows may take width-dependent arithmetic
 * -- the fastest arm at each width.  Rule 7 says that is graded, not argued, and
 * the reference gate grades PREFILL rows only.  This probe grades DECODE rows on
 * the same blobs: for each reference depth d and width w, bank 0 is prefilled
 * classically to d - w (since 2026-10-14 from the depth's one cold prefill's grid
 * checkpoint, asserted byte-identical to cold: prefix_resumed), then ONE decode step carries the w teacher-forced tokens
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
 * L262: it ENFORCES (cuda-decode-reference-gate-story / -code in the battery):
 * for every depth and width w > 1, KL(serial || wide) <= max(GATE_REF_WIDTH_KL_FRACTION
 * x KL(ref || serial), GATE_REF_WIDTH_KL_FLOOR), and the wide top-1 is the
 * serial's or the reference's.  Widths up to PULSAR_SPEC_LOGITS_ROWS (the
 * spec-on verify slab); widths <= pulsar_gpu_matmul_decode_exact_rows() are byte-identical to the
 * serial run and byte-gated elsewhere, so the default grades 12/16/24/32 only.  --known-flip
 * D[,D] names depths whose top-1 is a near tie (the prefill reference gate's list): KL still
 * enforced, a top-1 flip accepted by name.  This, not GATE 5R's replayed rows, is the fidelity
 * grade for width-dependent decode arithmetic (Tyler 2026-10-03: "go ahead").
 * L284: the serial run decodes all w rows anyway, so EVERY row of the wide step is graded against
 * the same row decoded one token at a time -- the verify's base and draft rows, not only the one the
 * reference holds.  Their KL(serial || wide) is REPORTED per width (no reference row to bound it
 * against -- a bound borrowed from the last row FAILED code depth 512, whose earlier rows are near
 * ties while the reference row is confident); a top-1 flip FAILS when the serial margin is
 * >= GATE_DECISIVE_MARGIN, the one statement rounding-sized arithmetic cannot excuse.
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "gate_entry.h"
#include "gate_fixture.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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

/* Bank 0 of `s` brought to the prefix [0, n) the cheap way (2026-10-14, battery time): the depth's ONE cold
 * prefill (prefix_cold) left a grid checkpoint at the last grid point below its end, and every later run restores
 * the deepest checkpoint at or below n and prefills the rest -- the cold prefill's computation (L264; the
 * chunk-neutrality gate's L / M / N schedules).  Before, every (width, arm) prefilled its prefix from 0: 8
 * prefills per depth, 380k tokens of the story gate's 392 s.  The resume is ANNOUNCED (origin) and the identity
 * is ASSERTED in this gate, not argued: the depth's widest wide run is the one cold run, repeated at the end
 * through this path and compared byte for byte (see the depth loop). */
static bool prefix_cold(pulsar_session *s, const int *toks, uint32_t n) {
    if (!gate_populate_bank(s, 0, toks, (int)n, NULL, "decode reference prefix")) return false;
    pulsar_session_bank_state_save(s, 0);   /* the carry a later run re-installs after its multiseq step */
    return true;
}
static bool prefix_resumed(pulsar_session *s, const int *toks, uint32_t n, int *origin) {
    char err[256];
    /* a multiseq step left the classic view stale (session_multiseq.cpp); the carry saved after the last sync
     * re-establishes it, as the server's bank switch does */
    if (!pulsar_session_bank_state_restore(s, 0)) { fprintf(stderr, "decode reference: bank 0 restore failed\n"); return false; }
    const int G = pulsar_session_checkpoint_best(s, TokPos{(int)n}).raw();
    if (G <= 0) {
        fprintf(stderr, "decode reference: no grid checkpoint at or below %u -- the cold prefill left none\n", n);
        return false;
    }
    if (pulsar_session_restore_checkpoint(s, TokPos{G}, err, sizeof err) != 0) {
        fprintf(stderr, "decode reference: %s\n", err);
        return false;
    }
    pulsar_tokens p = { .v = (int *)toks, .len = (int)n, .cap = (int)n };
    if (pulsar_session_sync(s, &p, err, sizeof err) != 0) {
        fprintf(stderr, "decode reference: resume sync to %u failed: %s\n", n, err);
        return false;
    }
    if (pulsar_session_resume_origin(s).raw() != G) {
        fprintf(stderr, "decode reference: the sync to %u resumed from %d, not the restored checkpoint %d\n", n,
                pulsar_session_resume_origin(s).raw(), G);
        return false;
    }
    pulsar_session_bank_state_save(s, 0);
    *origin = G;
    return true;
}

/* the w rows of the w tokens ending at depth d over bank 0's prefix [0, d - w): one w-wide decode step, or
 * (serial) w one-row steps -- into `out` (w x width) */
static bool decode_rows(pulsar_session *s, const int *toks, uint32_t d, uint32_t w, float *out, int width,
                        bool serial) {
    bool ok = true;
    pulsar_multiseq_req rq[PULSAR_SPEC_LOGITS_ROWS];
    for (uint32_t j = 0; j < w; j++) { rq[j].bank = 0; rq[j].pos = TokPos{(int32_t)(d - w + j)}; rq[j].token = toks[d - w + j]; }
    float *lg = (float *)malloc((size_t)w * (size_t)width * sizeof(float));
    uint32_t nr = 0;
    char err[256];
    const uint32_t steps = serial ? w : 1u, per = serial ? 1u : w;
    for (uint32_t k = 0; ok && k < steps; k++) {
        if (pulsar_session_decode_mixed(s, rq + k * per, per, lg + (size_t)k * per * (size_t)width,
                                        (int)(per * (uint32_t)width), &nr,
                                        PULSAR_MSEQ_HEAD_ALL_ROWS, err, sizeof err) != 0) {
            fprintf(stderr, "decode step (d %u, w %u, %s %u): %s\n", d, w, serial ? "serial" : "wide", k, err);
            ok = false;
        }
        if (ok && nr != per) { fprintf(stderr, "decode step headed %u rows, want %u\n", nr, per); ok = false; }
    }
    if (ok) memcpy(out, lg, (size_t)w * (size_t)width * sizeof(float));
    free(lg);
    return ok;
}

int GATE_ENTRY(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL REF.bin TOKENS.bin [w,w,...] [--known-flip D[,D...]]\n", argv[0]);
        return 2;
    }
    uint32_t widths[8] = {12, 16, 24, 32};
    int n_w = 4;
    uint32_t known_flip[MAX_DEPTHS];
    int n_kf = 0;
    for (int a = 4; a < argc; a++) {
        if (!strcmp(argv[a], "--known-flip") && a + 1 < argc) {
            for (char *p = strtok(argv[++a], ","); p && n_kf < (int)MAX_DEPTHS; p = strtok(NULL, ","))
                known_flip[n_kf++] = (uint32_t)atoi(p);
        } else {
            n_w = 0;
            for (char *p = strtok(argv[a], ","); p && n_w < 8; p = strtok(NULL, ",")) widths[n_w++] = (uint32_t)atoi(p);
        }
    }
    for (int i = 0; i < n_w; i++) {
        if (widths[i] < 1 || widths[i] > PULSAR_SPEC_LOGITS_ROWS) {
            fprintf(stderr, "width %u out of [1, %u]\n", widths[i], (unsigned)PULSAR_SPEC_LOGITS_ROWS);
            return 2;
        }
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
    float *one_all = (float *)malloc((size_t)PULSAR_SPEC_LOGITS_ROWS * (size_t)width * sizeof(float));
    float *cur_all = (float *)malloc((size_t)PULSAR_SPEC_LOGITS_ROWS * (size_t)width * sizeof(float));
    float *cold_all = (float *)malloc((size_t)PULSAR_SPEC_LOGITS_ROWS * (size_t)width * sizeof(float));
    /* L284: every row of the step against the same row of the serial run (rows < w carry no reference row) */
    double rows_kl[8] = {0}, rows_kl_max[8] = {0};
    long rows_n[8] = {0}, rows_flip[8] = {0};
    int rows_violations = 0;
    printf("decode reference gate: %s (engine '%.*s', %u depths)\n", argv[2], (int)REF_LEN, h.build_ref, h.n_depths);
    printf("  depth  w   KL(ref||ours)  top1(ref/ours)   KL vs serial   max|d| vs serial   bound\n");
    int violations = 0;
    double sum_kl[8] = {0}, sum_kw[8] = {0};
    int flips[8] = {0};
    int rc = 0;
    for (uint32_t i = 0; i < h.n_depths && rc == 0; i++) {
        const uint32_t d = h.depths[i];
        const float *rr = ref + (size_t)i * h.width;
        /* one session per depth; its one cold prefill is to the SHORTEST prefix any width needs (d - the widest
         * width), so the grid checkpoint it leaves lies at or below every run's prefix */
        uint32_t wmax = 0;
        for (int k = 0; k < n_w; k++) if (d >= widths[k] + 1 && widths[k] > wmax) wmax = widths[k];
        if (wmax == 0) continue;
        pulsar_session *s = NULL;
        const int ctx = (int)(((d + 4095u) / 4096u + 1u) * 4096u);
        if (pulsar_session_create(&s, e, ctx) != 0) { fprintf(stderr, "session create failed (ctx %d)\n", ctx); rc = 1; break; }
        if (!gate_pool_fits(s, 2) || !prefix_cold(s, toks, d - wmax) ||
            !decode_rows(s, toks, d, wmax, cold_all, width, false)) {
            pulsar_session_free(s);
            rc = 1;
            break;
        }
        int origin_lo = 1 << 30, origin_hi = 0, resumed = 0;
        bool identical = false;
        for (int k = 0; k < n_w && rc == 0; k++) {
            const uint32_t w = widths[k];
            if (d < w + 1) continue;
            int origin = 0;
            if (!prefix_resumed(s, toks, d - w, &origin) || !decode_rows(s, toks, d, w, cur_all, width, false)) { rc = 1; break; }
            origin_lo = origin < origin_lo ? origin : origin_lo;
            origin_hi = origin > origin_hi ? origin : origin_hi;
            resumed++;
            if (w == wmax) {
                /* the resumed prefix's run against the depth's cold run: the premise of every row above */
                identical = memcmp(cur_all, cold_all, (size_t)w * (size_t)width * sizeof(float)) == 0;
                if (!identical) {
                    printf("  depth %u w %u: the run over the resumed prefix DIFFERS from the cold prefill's -- "
                           "FAIL (a resume is not the cold computation)\n", d, w);
                    violations++;
                }
            }
            if (w > 1) {
                if (!prefix_resumed(s, toks, d - w, &origin) || !decode_rows(s, toks, d, w, one_all, width, true)) { rc = 1; break; }
                origin_lo = origin < origin_lo ? origin : origin_lo;
                origin_hi = origin > origin_hi ? origin : origin_hi;
                resumed++;
            }
            if (w == 1) memcpy(one_all, cur_all, (size_t)width * sizeof(float));
            const float *one = one_all + (size_t)(w - 1) * width, *cur = cur_all + (size_t)(w - 1) * width;
            double step_kl = 0;
            for (uint32_t j = 0; w > 1 && j < w; j++) {
                const float *sr = one_all + (size_t)j * width, *wr = cur_all + (size_t)j * width;
                const double kj = kl_of(sr, wr, ncmp);
                step_kl += kj;
                rows_kl[k] += kj;
                rows_kl_max[k] = fmax(rows_kl_max[k], kj);
                rows_n[k]++;
                const int as = argmax_of(sr, ncmp), aw = argmax_of(wr, ncmp);
                if (as != aw) {
                    /* a width change is rounding-sized: it may only flip a near-tie of the serial row */
                    const double margin = (double)sr[as] - (double)sr[aw];
                    const bool decisive = margin >= GATE_DECISIVE_MARGIN;
                    rows_flip[k]++;
                    rows_violations += decisive;
                    printf("    row %u of d %u w %u: top-1 %d (serial) -> %d (wide), serial margin %.3f%s\n", j, d, w, as,
                           aw, margin, decisive ? "  FAIL (decisive flip)" : "");
                }
            }
            const float *dst = cur;
            const double kl = kl_of(rr, dst, ncmp);
            const int ar = argmax_of(rr, ncmp), ao = argmax_of(dst, ncmp);
            const double klw = kl_of(one, dst, ncmp);
            const double mw = maxabs_of(one, dst, ncmp);
            /* the bound is relative to the one-row run's own distance from the reference */
            const double kl_one = kl_of(rr, one, ncmp);
            const double bound = fmax(GATE_REF_WIDTH_KL_FRACTION * kl_one, GATE_REF_WIDTH_KL_FLOOR);
            const int a1 = argmax_of(one, ncmp);
            bool flip_named = false;
            for (int f = 0; f < n_kf; f++) flip_named |= known_flip[f] == d;
            const bool top_ok = ao == a1 || ao == ar || flip_named;
            const bool ok = w == 1 || (klw <= bound && top_ok);
            /* L284: the other rows' KL is REPORTED, not bounded: the reference holds only the last row, and the
             * model's own distance from the source moves row to row (code 512: the reference row is confident,
             * rows 480..511 are near ties), so no bound derived from it transfers.  Only a decisive flip fails. */
            const double rows_mean = w > 1 ? step_kl / w : 0.0;
            printf("  %5u %2u   %.3e      %6d/%-6d %s  %.3e   %.3e        %.3e%s%s   rows mean %.3e\n", d, w, kl, ar,
                   ao, ar == ao ? " " : "*", klw, mw, bound, ok ? "" : top_ok ? "  FAIL (KL)" : "  FAIL (top-1)",
                   flip_named && ao != a1 && ao != ar ? "  (known-flip depth: top-1 accepted by name)" : "", rows_mean);
            violations += !ok;
            sum_kl[k] += kl;
            sum_kw[k] += klw;
            flips[k] += ar != ao;
        }
        pulsar_session_free(s);
        if (rc == 0)
            printf("    [resume] depth %u: one cold prefill to %u; %d runs resumed from grid checkpoint %d..%d; "
                   "w %u over the resumed prefix %s the cold run\n", d, d - wmax, resumed, origin_lo, origin_hi,
                   wmax, identical ? "== (byte-identical)" : "!=");
    }
    if (rc == 0) {
        printf("  summary (mean over %u depths):\n", h.n_depths);
        for (int k = 0; k < n_w; k++)
            printf("    w=%2u  mean KL(ref||ours) %.3e  top1 misses %d  mean KL vs serial %.3e\n", widths[k],
                   sum_kl[k] / h.n_depths, flips[k], sum_kw[k] / h.n_depths);
        printf("  every row vs the serial run (L284; KL reported, FAIL only on a decisive top-1 flip):\n");
        for (int k = 0; k < n_w; k++)
            if (rows_n[k])
                printf("    w=%2u  rows %4ld  KL mean %.3e  max %.3e  top-1 flips %ld\n", widths[k], rows_n[k],
                       rows_kl[k] / (double)rows_n[k], rows_kl_max[k], rows_flip[k]);
        violations += rows_violations;
    }
    free(one_all);
    free(cur_all);
    free(cold_all);
    free(ref);
    free(traw);
    gate_engine_close(e);
    if (rc == 0) {
        printf("DECODE REFERENCE GATE: %s -- %d violation(s) over %u depths x %d widths\n",
               violations ? "FAIL" : "PASS", violations, h.n_depths, n_w);
        if (violations) rc = 1;
    }
    return rc;
}

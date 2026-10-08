/* L284 PROBE: every row of a verify-shaped decode step, graded per width against the
 * same token's one-token decode and, where the B300 capture has the position, against
 * the reference.
 *
 * cuda-decode-reference-gate grades the LAST row of a w-wide step (end-anchored at a
 * reference depth).  This grades EVERY row, start-anchored: bank b is prefilled to
 * P_b = min(d_b - 1, n_tok - 32), then
 *   serial: 32 one-row decode steps of toks[P_b ..] on that bank  (the one-token truth),
 *   wide:   over a fresh prefill to the same P_b, ONE decode step carrying r rows per
 *           bank for every bank of the group (w = NB x r rows in one call),
 * and row k of the wide step is compared with serial row k: KL(serial || wide), top-1
 * agreement, byte identity.  Row d_b - 1 - P_b predicts the token after d_b -- the
 * reference row when d_b is a captured depth -- so each width also reports
 * KL(ref || wide) beside KL(ref || serial): which arm is closer to the source.
 *
 *   PULSAR_MSEQ_BANKS=NB ./tests/verify_width_probe MODEL REF.bin TOKENS.bin
 *        [--widths 1,4,8,10,11,16,24,32] [--banks NB] [--extra D,D,...] [--max-depth D]
 *
 * --banks NB groups the depths NB at a time, one bank each, and spreads every width
 * over them (widths not divisible by NB are skipped) -- the production verify shape
 * (several streams, a few rows each).  --extra adds depths with no reference row (KL
 * vs one-token decode only); --max-depth drops deeper captured depths (prefill cost).
 * Measurement only: it never fails on a number. */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "gate_entry.h"
#include "gate_fixture.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <vector>

#define MAGIC "DS4PFXG1"
#define MAX_DEPTHS 8u
#define REF_LEN 24u
#define SERIAL_ROWS 32u
#define MAX_W 16
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

static int argmax_of(const float *x, int n) {
    int a = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[a]) a = i;
    return a;
}

typedef struct {
    double kl_sum, kl_max, kl0_sum;  /* all rows; row 0 of each bank (the verify's base row) */
    long   n, n0, agree, agree0, ident;
    double ref_wide, ref_serial;     /* KL(ref || .) summed over reference rows */
    int    n_ref, top_wide, top_serial;
} width_stats;

static uint32_t parse_list(char *s, uint32_t *out, uint32_t cap) {
    uint32_t n = 0;
    for (char *p = strtok(s, ","); p && n < cap; p = strtok(NULL, ",")) out[n++] = (uint32_t)atoi(p);
    return n;
}

int GATE_ENTRY(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL REF.bin TOKENS.bin [--widths w,..] [--banks NB] [--extra D,..] "
                        "[--max-depth D]\n", argv[0]);
        return 2;
    }
    uint32_t widths[MAX_W] = {1, 4, 8, 10, 11, 16, 24, 32};
    uint32_t n_w = 8, nb = 1, max_depth = 0xffffffffu, extra[16], n_extra = 0;
    for (int a = 4; a + 1 < argc; a += 2) {
        if (!strcmp(argv[a], "--widths")) n_w = parse_list(argv[a + 1], widths, MAX_W);
        else if (!strcmp(argv[a], "--banks")) nb = (uint32_t)atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "--extra")) n_extra = parse_list(argv[a + 1], extra, 16);
        else if (!strcmp(argv[a], "--max-depth")) max_depth = (uint32_t)atoi(argv[a + 1]);
        else { fprintf(stderr, "unknown option %s\n", argv[a]); return 2; }
    }
    if (nb < 1 || nb > 8) { fprintf(stderr, "--banks %u out of [1, 8]\n", nb); return 2; }
    for (uint32_t i = 0; i < n_w; i++)
        if (widths[i] < 1 || widths[i] > PULSAR_SPEC_LOGITS_ROWS) { fprintf(stderr, "bad width %u\n", widths[i]); return 2; }

    FILE *fp = fopen(argv[2], "rb");
    blob_header h;
    if (!fp || fread(&h, sizeof h, 1, fp) != 1 || memcmp(h.magic, MAGIC, 8) || h.n_depths == 0 ||
        h.n_depths > MAX_DEPTHS || h.width == 0) {
        fprintf(stderr, "reference %s unreadable\n", argv[2]);
        if (fp) fclose(fp);
        return 1;
    }
    std::vector<float> ref((size_t)h.n_depths * h.width);
    const bool body_ok = fread(ref.data(), sizeof(float), ref.size(), fp) == ref.size();
    fclose(fp);
    if (!body_ok) { fprintf(stderr, "reference %s: short body\n", argv[2]); return 1; }
    size_t tb = 0;
    char *traw = gate_read_file(argv[3], &tb);
    if (!traw || tb % sizeof(int)) { fprintf(stderr, "tokens %s unreadable\n", argv[3]); return 1; }
    const int *toks = (const int *)traw;
    const uint32_t n_tok = (uint32_t)(tb / sizeof(int));

    /* the depth list: captured depths (with their reference row) then the extras */
    uint32_t depth[32];
    int ref_of[32];
    uint32_t nd = 0;
    for (uint32_t i = 0; i < h.n_depths && nd < 32; i++)
        if (h.depths[i] <= max_depth && h.depths[i] <= n_tok && h.depths[i] > SERIAL_ROWS) { depth[nd] = h.depths[i]; ref_of[nd++] = (int)i; }
    for (uint32_t i = 0; i < n_extra && nd < 32; i++)
        if (extra[i] + SERIAL_ROWS <= n_tok && extra[i] > SERIAL_ROWS) { depth[nd] = extra[i]; ref_of[nd++] = -1; }

    pulsar_engine_set_bank_pool(nb < 2 ? 2 : nb);   /* a pool, so decode_mixed runs banked */
    pulsar_engine *e = NULL;
    pulsar_engine_options o;
    memset(&o, 0, sizeof o);
    o.model_path = argv[1];
    o.backend = PULSAR_BACKEND_CUDA;
    o.prefill_chunk = 4096;
    if (gate_engine_open(&e, &o) != 0) { fprintf(stderr, "engine open failed\n"); free(traw); return 1; }
    const int V = pulsar_engine_logits_width(e);
    const int ncmp = (int)h.width < V ? (int)h.width : V;
    printf("verify width probe: %s, %u depths, %u bank(s) per step, exact rows <= %d\n", argv[2], nd, nb,
           pulsar_gpu_matmul_decode_exact_rows());

    width_stats st[MAX_W];
    memset(st, 0, sizeof st);
    std::vector<float> serial((size_t)nb * SERIAL_ROWS * V), wide((size_t)SERIAL_ROWS * V);
    char err[256];
    int rc = 0;
    for (uint32_t g0 = 0; rc == 0 && g0 + nb <= nd; g0 += nb) {
        uint32_t P[8], dmax = 0;
        for (uint32_t b = 0; b < nb; b++) {
            const uint32_t d = depth[g0 + b];
            P[b] = d - 1 < n_tok - SERIAL_ROWS ? d - 1 : n_tok - SERIAL_ROWS;
            if (d > dmax) dmax = d;
        }
        const int ctx = (int)(((dmax + SERIAL_ROWS + 4095u) / 4096u + 1u) * 4096u);
        pulsar_session *s = NULL;
        if (pulsar_session_create(&s, e, ctx) != 0) { fprintf(stderr, "session create failed\n"); rc = 1; break; }
        if (!gate_pool_fits(s, nb < 2 ? 2 : nb)) { pulsar_session_free(s); rc = 1; break; }
        /* serial truth: every bank prefilled, then one-row steps bank by bank */
        bool ok = true;
        for (uint32_t b = 0; ok && b < nb; b++) ok = gate_populate_bank(s, b, toks, (int)P[b], NULL, "serial prefix");
        for (uint32_t b = 0; ok && b < nb; b++)
            for (uint32_t k = 0; ok && k < SERIAL_ROWS; k++) {
                pulsar_multiseq_req q = {b, (int32_t)(P[b] + k), toks[P[b] + k]};
                uint32_t got = 0;
                ok = pulsar_session_decode_mixed(s, &q, 1, serial.data() + ((size_t)b * SERIAL_ROWS + k) * V, V, &got,
                                                 PULSAR_MSEQ_HEAD_ALL_ROWS, err, sizeof err) == 0 && got == 1;
                if (!ok) fprintf(stderr, "serial step b%u k%u: %s\n", b, k, err);
            }
        for (uint32_t wi = 0; ok && wi < n_w; wi++) {
            const uint32_t w = widths[wi];
            if (w % nb) continue;
            const uint32_t r = w / nb;
            for (uint32_t b = 0; ok && b < nb; b++) ok = gate_populate_bank(s, b, toks, (int)P[b], NULL, "wide prefix");
            pulsar_multiseq_req q[PULSAR_SPEC_LOGITS_ROWS];
            for (uint32_t b = 0; b < nb; b++)
                for (uint32_t k = 0; k < r; k++) q[b * r + k] = {b, (int32_t)(P[b] + k), toks[P[b] + k]};
            uint32_t got = 0;
            ok = pulsar_session_decode_mixed(s, q, w, wide.data(), (int)w * V, &got, PULSAR_MSEQ_HEAD_ALL_ROWS, err,
                                             sizeof err) == 0 && got == w;
            if (!ok) { fprintf(stderr, "wide step w%u: %s\n", w, err); break; }
            width_stats *x = &st[wi];
            for (uint32_t b = 0; b < nb; b++) {
                double ks = 0, km = 0;
                int ag = 0, id = 0;
                for (uint32_t k = 0; k < r; k++) {
                    const float *sr = serial.data() + ((size_t)b * SERIAL_ROWS + k) * V, *wr = wide.data() + ((size_t)b * r + k) * V;
                    const double kl = gate_row_kl(sr, wr, ncmp);
                    const int a = argmax_of(sr, ncmp) == argmax_of(wr, ncmp);
                    ks += kl; km = kl > km ? kl : km; ag += a;
                    id += memcmp(sr, wr, (size_t)V * sizeof(float)) == 0;
                    if (k == 0) { x->kl0_sum += kl; x->n0++; x->agree0 += a; }
                }
                x->kl_sum += ks; x->kl_max = km > x->kl_max ? km : x->kl_max; x->n += r; x->agree += ag; x->ident += id;
                const uint32_t d = depth[g0 + b];
                const uint32_t rr = d - 1 - P[b];
                char refcol[160] = "";
                if (ref_of[g0 + b] >= 0 && rr < r) {
                    const float *R = ref.data() + (size_t)ref_of[g0 + b] * h.width;
                    const float *sr = serial.data() + ((size_t)b * SERIAL_ROWS + rr) * V, *wr = wide.data() + ((size_t)b * r + rr) * V;
                    const double kw = gate_row_kl(R, wr, ncmp), ks1 = gate_row_kl(R, sr, ncmp);
                    const int ar = argmax_of(R, ncmp), aw = argmax_of(wr, ncmp), as = argmax_of(sr, ncmp);
                    x->ref_wide += kw; x->ref_serial += ks1; x->n_ref++; x->top_wide += aw == ar; x->top_serial += as == ar;
                    snprintf(refcol, sizeof refcol, "  ref row %u: KL(ref||wide) %.3e KL(ref||serial) %.3e top1 %d/%d/%d", rr,
                             kw, ks1, ar, aw, as);
                }
                printf("  d %5u b%u w %2u (r %2u): KL vs serial mean %.3e max %.3e  top1 %d/%u  identical %d/%u%s\n", d, b, w,
                       r, ks / r, km, ag, r, id, r, refcol);
                fflush(stdout);
            }
        }
        pulsar_session_free(s);
        if (!ok) rc = 1;
    }
    if (rc == 0) {
        printf("  summary (%u bank(s) per step):\n", nb);
        printf("     w   rows  KL mean    KL max     row0 KL mean  top1 agree  row0 agree  identical   ref rows  "
               "KL(ref||wide)  KL(ref||serial)  ref top1 wide/serial\n");
        for (uint32_t wi = 0; wi < n_w; wi++) {
            const width_stats *x = &st[wi];
            if (!x->n) continue;
            printf("    %2u  %5ld  %.3e  %.3e  %.3e     %4ld/%-4ld  %4ld/%-4ld  %4ld/%-4ld  %4d       %.3e      %.3e        %d/%d\n",
                   widths[wi], x->n, x->kl_sum / x->n, x->kl_max, x->kl0_sum / x->n0, x->agree, x->n, x->agree0, x->n0,
                   x->ident, x->n, x->n_ref, x->n_ref ? x->ref_wide / x->n_ref : 0.0,
                   x->n_ref ? x->ref_serial / x->n_ref : 0.0, x->top_wide, x->top_serial);
        }
    }
    free(traw);
    gate_engine_close(e);
    return rc;
}

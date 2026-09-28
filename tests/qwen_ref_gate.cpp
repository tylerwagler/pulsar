/* qwen_ref_gate.cpp -- L251 item (4): the Qwen REFERENCE GATE.
 *
 *   ./tests/qwen_ref_gate <container> <anchors-dir> [prompt ...]   (default: code story)
 *
 * The anchors (qwen_anchors.py, the BF16 streamed source) record, per prompt,
 *   <P>.tokens.bin  i32 [n]              the exact prompt tokens
 *   <P>.ref.bin     88-byte DS4PFXG1 header + f32 [n_rows][W]  reference logits, in the json's row order
 *   <P>.ref.json    {"width":W, "rows":[{"depth":d,"argmax_id":i,"entropy_nats":e,"p_top1":p}, ...]}
 * The engine is fed tokens[0:depth] for every recorded depth in ASCENDING order
 * (the family's prefix reuse carries the session forward), and its last row is
 * compared with that row of ref.bin.  ARGMAX MUST MATCH at every depth -- a wrong
 * argmax is a different model, not a rounding difference.  The entropy / p_top1
 * from the json are printed beside the engine's so a near miss is visible.
 *
 * Not part of the battery: it needs a real container and the anchors on disk. */
#include "pulsar.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Bytes before <P>.ref.bin's first logit row: magic "DS4PFXG1"[8], u32 version,
 * u32 n_rows, u64 width, u64 prompt_fnv, the per-depth frames and the build_ref
 * string, zero-padded.  The producer is qwen_anchors.py; this gate checks the
 * magic, the version and the width rather than trusting the offset. */
#define QWEN_REF_HDR 88u

static int g_fail = 0;
/* PULSAR_REF_FRESH=1: grade each depth in a FRESH session (see the depth loop). */
static int g_fresh = 0;
/* PULSAR_REF_DUMP=<dir>: write each engine logits row as f32 [W] there. */
static const char *g_dump = NULL;

static void check(bool ok, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    printf("  [%s] ", ok ? "ok" : "FAIL"); vprintf(fmt, ap); printf("\n");
    va_end(ap);
    if (!ok) g_fail++;
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "qwen-ref gate: out of memory\n"); exit(2); }
    return p;
}

static unsigned char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    const long sz = ftell(f);
    if (sz <= 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    unsigned char *b = (unsigned char *)xmalloc((size_t)sz);
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(b); return NULL; }
    fclose(f);
    *n = (size_t)sz;
    return b;
}

/* One recorded depth.  The anchors' json is a fixed shape (an ordered list of
 * rows with these four keys), so this scans it in order rather than pulling in a
 * JSON parser -- the file is machine-written by qwen_anchors.py. */
struct rec { int depth, argmax; double entropy, p_top1; };

static int parse_recs(const char *path, struct rec **out) {
    size_t n = 0;
    unsigned char *b = slurp(path, &n);
    if (!b) return 0;
    int cap = 32, k = 0;
    struct rec *r = (struct rec *)xmalloc(sizeof(struct rec) * (size_t)cap);
    const char *p = (const char *)b;
    const char *end = p + n;
    while ((p = strstr(p, "\"depth\":")) != NULL && p < end) {
        const char *a = strstr(p, "\"argmax_id\":");
        if (!a) break;
        if (k == cap) { cap *= 2; r = (struct rec *)realloc(r, sizeof(struct rec) * (size_t)cap); }
        r[k].depth = atoi(p + 8);
        r[k].argmax = atoi(a + 12);
        const char *e = strstr(p, "\"entropy_nats\":");
        const char *t = strstr(p, "\"p_top1\":");
        r[k].entropy = e ? atof(e + 15) : 0.0;
        r[k].p_top1 = t ? atof(t + 9) : 0.0;
        k++;
        p += 8;
    }
    free(b);
    *out = r;
    return k;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <container> <anchors-dir> [prompt ...]\n", argv[0]);
        return 2;
    }
    const char *prompts[8];
    int np = 0;
    if (argc > 3) {
        for (int i = 3; i < argc && np < 8; i++) prompts[np++] = argv[i];
    } else {
        prompts[np++] = "code";
        prompts[np++] = "story";
    }
    g_fresh = getenv("PULSAR_REF_FRESH") != NULL;
    g_dump  = getenv("PULSAR_REF_DUMP");
    if (g_fresh) printf("      (PULSAR_REF_FRESH: every depth is graded in a fresh session)\n");
    if (g_dump)  printf("      (PULSAR_REF_DUMP: engine rows -> %s)\n", g_dump);

    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) {
        fprintf(stderr, "qwen-ref gate: %s did not open\n", argv[1]);
        return 2;
    }
    const int W = pulsar_engine_logits_width(e);
    check(W == 248320, "logits width %d (the anchors are 248320)", W);

    for (int i = 0; i < np; i++) {
        char tp[1024], rp[1024], jp[1024];
        snprintf(tp, sizeof(tp), "%s/%s.tokens.bin", argv[2], prompts[i]);
        snprintf(rp, sizeof(rp), "%s/%s.ref.bin", argv[2], prompts[i]);
        snprintf(jp, sizeof(jp), "%s/%s.ref.json", argv[2], prompts[i]);
        size_t tn = 0, rn = 0;
        unsigned char *tb = slurp(tp, &tn), *rb = slurp(rp, &rn);
        struct rec *recs = NULL;
        const int nr = parse_recs(jp, &recs);
        check(tb && rb && nr > 0, "%s: tokens.bin + ref.bin + ref.json present (%d depths)", prompts[i], nr);
        if (!tb || !rb || nr <= 0) { free(tb); free(rb); free(recs); continue; }

        const int32_t *toks = (const int32_t *)tb;
        const int n_tok = (int)(tn / sizeof(int32_t));
        /* ref.bin is NOT a bare f32 array: it opens with an 88-byte DS4PFXG1
         * header -- magic[8], u32 version, u32 n_rows, u64 width, u64 prompt_fnv,
         * then the per-depth frames and the build_ref string, padded.  Reading it
         * as rows from byte 0 shifts every reference row by 22 floats, which made
         * this gate report argmax mismatch against its OWN anchors on nearly every
         * depth (and a 2e+33 max|logit diff| against header bytes).  Parse it, so
         * the instrument proves it is reading the shape it claims. */
        const float *ref = NULL;
        int n_ref = 0;
        if (!rb || rn < QWEN_REF_HDR || memcmp(rb, "DS4PFXG1", 8) != 0) {
            check(false, "%s: ref.bin has no DS4PFXG1 header", prompts[i]);
        } else {
            uint32_t ver = 0, hrows = 0;
            uint64_t hw = 0;
            memcpy(&ver, rb + 8, 4);
            memcpy(&hrows, rb + 12, 4);
            memcpy(&hw, rb + 16, 8);
            n_ref = (int)((rn - QWEN_REF_HDR) / (sizeof(float) * (size_t)W));
            if (ver != 2 || hw != (uint64_t)W || n_ref < (int)hrows) {
                check(false, "%s: ref.bin header says v%u width %llu rows %u; parsing to %d rows of %d",
                      prompts[i], ver, (unsigned long long)hw, hrows, n_ref, W);
                n_ref = 0;
            } else {
                n_ref = (int)hrows;
                ref = (const float *)(rb + QWEN_REF_HDR);
            }
        }
        check(n_ref == nr, "%s: ref.bin holds %d rows == the json's %d", prompts[i], n_ref, nr);
        if (!ref) { free(tb); free(rb); free(recs); continue; }

        int maxd = 0;
        for (int k = 0; k < nr; k++) if (recs[k].depth > maxd) maxd = recs[k].depth;
        pulsar_session *sess = NULL;
        if (pulsar_session_create(&sess, e, maxd + 64) != 0) {
            check(false, "%s: session of %d tokens", prompts[i], maxd + 64);
            free(tb); free(rb); free(recs);
            continue;
        }
        float *row = (float *)xmalloc(sizeof(float) * (size_t)W);
        int am_ok = 0, graded = 0;
        double worst_kl = 0, worst_mx = 0;
        for (int k = 0; k < nr; k++) {
            const int d = recs[k].depth;
            if (d <= 0 || d > n_tok) { check(false, "%s: depth %d out of range (n_tok %d)", prompts[i], d, n_tok); break; }
            /* PULSAR_REF_FRESH=1 grades every depth in a FRESH session, so the same
             * depth is reached by one sync from empty instead of by prefix reuse.
             * A depth that is right fresh and wrong reused (or the reverse) LOCALIZES
             * the fault to the reuse path rather than to the forward -- which is the
             * question when one depth of a run disagrees and its neighbours do not. */
            if (g_fresh && k > 0) {
                pulsar_session_free(sess);
                sess = NULL;
                if (pulsar_session_create(&sess, e, maxd + 64) != 0) {
                    check(false, "%s: fresh session of %d tokens", prompts[i], maxd + 64);
                    break;
                }
            }
            pulsar_tokens t = { (int *)toks, d, d };
            char err[256] = "";
            if (pulsar_session_sync(sess, &t, err, sizeof(err)) != 0) {
                check(false, "%s d=%d sync: %s", prompts[i], d, err); break;
            }
            err[0] = '\0';
            if (pulsar_session_eval(sess, 1, err, sizeof(err)) != 0) {
                check(false, "%s d=%d eval: %s", prompts[i], d, err); break;
            }
            /* copy_logits returns the number of logits WRITTEN and 0 on error -- the
             * opposite convention to set_logits (0 on success).  Checking `!= 0` here
             * flagged a perfectly good 248320-logit copy as a failure. */
            if (pulsar_session_copy_logits(sess, row, W) != W) {
                check(false, "%s d=%d copy_logits", prompts[i], d); break;
            }
            int am = 0;
            for (int j = 1; j < W; j++) if (row[j] > row[am]) am = j;
            /* the reference row for this depth, graded as well: the json's argmax
             * must be ref.bin's (that cross-checks the parse), and the KL /
             * max|logit| against the engine is the gate's graded number. */
            const float *rr = ref + (size_t)k * (size_t)W;
            int ram = 0;
            for (int j = 1; j < W; j++) if (rr[j] > rr[ram]) ram = j;
            if (ram != recs[k].argmax)
                printf("      d=%-6d note: ref.bin argmax %d, the json says %d\n", d, ram, recs[k].argmax);
            double mref = rr[0], meng = row[0];
            for (int j = 0; j < W; j++) {
                if (rr[j] > mref) mref = rr[j];
                if (row[j] > meng) meng = row[j];
            }
            double sref = 0, seng = 0;
            for (int j = 0; j < W; j++) { sref += exp((double)rr[j] - mref); seng += exp((double)row[j] - meng); }
            double kl = 0, mx = 0;
            uint64_t he = 1469598103934665603ull, hr = 1469598103934665603ull;
            for (int j = 0; j < W; j++) {
                const double pr = exp((double)rr[j] - mref) / sref, pe = exp((double)row[j] - meng) / seng;
                if (pr > 0 && pe > 0) kl += pr * log(pr / pe);
                const double dd = fabs((double)rr[j] - (double)row[j]);
                if (dd > mx) mx = dd;
                uint32_t be, br;
                memcpy(&be, &row[j], 4); he = (he ^ be) * 1099511628211ull;
                memcpy(&br, &rr[j], 4);  hr = (hr ^ br) * 1099511628211ull;
            }
            if (kl > worst_kl) worst_kl = kl;
            if (mx > worst_mx) worst_mx = mx;
            graded++;
            /* Least-squares fit row ~ slope*rr + intercept, and the max residual after
             * removing it.  A constant offset is softmax-invariant (harmless, and it is
             * what makes the engine's top logit look "too high" while KL stays ~0); a
             * slope != 1 is a real temperature/scale defect; a large residual is
             * structure.  Fitting says which, instead of reading the raw max diff. */
            double mr = 0, me = 0;
            for (int j = 0; j < W; j++) { mr += rr[j]; me += row[j]; }
            mr /= W; me /= W;
            double sxy = 0, sxx = 0;
            for (int j = 0; j < W; j++) { const double dx = rr[j] - mr; sxy += dx * (row[j] - me); sxx += dx * dx; }
            const double slope = sxx > 0 ? sxy / sxx : 1.0;
            const double icept = me - slope * mr;
            double resid = 0;
            for (int j = 0; j < W; j++) {
                const double e = fabs((double)row[j] - (slope * (double)rr[j] + icept));
                if (e > resid) resid = e;
            }
            /* ALWAYS one line per depth.  Two depths sharing `fnv eng` means the session
             * never advanced -- prefix reuse handed back the same row -- which a bare
             * argmax count hides completely (it reads as a near miss instead of a stall). */
            printf("      d=%-6d eng %7d @ %9.4f | ref %7d @ %9.4f | KL %.3e | fit x%.4f %+.3f resid %.3e | fnv %016llx/%016llx%s\n",
                   d, am, (double)row[am], ram, (double)rr[ram], kl, slope, icept, resid,
                   (unsigned long long)he, (unsigned long long)hr,
                   (am == recs[k].argmax) ? "" : "   <-- ARGMAX MISMATCH");
            /* The top 5 of each row, token by token.  A uniform scale keeps the SAME
             * tokens on top with proportional values; a structural difference puts
             * different tokens there.  The whole-row fit cannot tell those apart because
             * 248k near-zero tails outvote the handful of entries that decide the
             * distribution, which is exactly what the sub-1 slopes above are. */
            int et[5], rt[5];
            for (int q = 0; q < 5; q++) {
                int be = -1, br = -1;
                for (int j = 0; j < W; j++) {
                    bool ue = false, ur = false;
                    for (int z = 0; z < q; z++) { if (et[z] == j) ue = true; if (rt[z] == j) ur = true; }
                    if (!ue && (be < 0 || row[j] > row[be])) be = j;
                    if (!ur && (br < 0 || rr[j] > rr[br])) br = j;
                }
                et[q] = be; rt[q] = br;
            }
            printf("               eng top5");
            for (int q = 0; q < 5; q++) printf(" %d@%.3f", et[q], (double)row[et[q]]);
            printf("\n               ref top5");
            for (int q = 0; q < 5; q++) printf(" %d@%.3f", rt[q], (double)rr[rt[q]]);
            printf("\n");
            /* WHICH reference depth does this engine row actually look like?  The engine
             * predicts <|im_end|> several tokens before the reference does in `story`, so
             * the fault could be a SHIFTED CONTEXT (positions/prefix) rather than a
             * slightly different model -- and those need different fixes.  Grading the
             * engine's row against every recorded reference row separates them: a row
             * that best matches a different depth is a shift, not a numerical drift. */
            int bestk = -1;
            double bestkl = 0;
            for (int k2 = 0; k2 < nr; k2++) {
                const float *r2 = ref + (size_t)k2 * (size_t)W;
                double m2 = r2[0];
                for (int j = 1; j < W; j++) if (r2[j] > m2) m2 = r2[j];
                double s2 = 0;
                for (int j = 0; j < W; j++) s2 += exp((double)r2[j] - m2);
                double q = 0;
                for (int j = 0; j < W; j++) {
                    const double pr = exp((double)r2[j] - m2) / s2, pe = exp((double)row[j] - meng) / seng;
                    if (pr > 0 && pe > 0) q += pr * log(pr / pe);
                }
                if (bestk < 0 || q < bestkl) { bestkl = q; bestk = k2; }
            }
            printf("               eng d=%-6d best matches ref[%d] d=%-6d KL %.3e%s\n",
                   d, bestk, recs[bestk].depth, bestkl,
                   (recs[bestk].depth == d) ? "" : "   <-- SHIFTED CONTEXT");
            /* CONFIDENCE, engine beside the anchor's recorded value.  The engine collapsing
             * to a delta on <|im_end|> while the reference is at p_top1 0.35-0.69 is the
             * defect in one number, and the anchors already carry the reference's side. */
            {
                const double pe_top = exp((double)row[am] - meng) / seng;
                double H = 0;
                for (int j = 0; j < W; j++) {
                    const double p = exp((double)row[j] - meng) / seng;
                    if (p > 0) H -= p * log(p);
                }
                printf("               eng p_top1 %.4f H %.4f  ||  ref p_top1 %.4f H %.4f   (eng_top %.3f vs ref_top %.3f)\n",
                       pe_top, H, recs[k].p_top1, recs[k].entropy_nats,
                       (double)row[am], (double)rr[ram]);
            }
            if (g_dump) {
                char fp2[1200];
                snprintf(fp2, sizeof(fp2), "%s/%s.d%d.eng.f32", g_dump, prompts[i], d);
                FILE *f2 = fopen(fp2, "wb");
                if (f2) { fwrite(row, sizeof(float), (size_t)W, f2); fclose(f2); }
            }
            if (am == recs[k].argmax) am_ok++;
        }
        check(graded == nr && am_ok == nr, "%s: argmax matches at %d / %d depths", prompts[i], am_ok, nr);
        printf("      %s: worst KL(ref||engine) %.3e, worst max|logit diff| %.3e over %d depths\n",
               prompts[i], worst_kl, worst_mx, graded);
        free(row);
        pulsar_session_free(sess);
        free(tb); free(rb); free(recs);
    }

    pulsar_engine_close(e);
    printf("QWEN-REF GATE %s (%d)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

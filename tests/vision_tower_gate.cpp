/* L216 vision-tower gate: the engine's ViT + aligner vs the reference's.
 *
 * Goldens come from tests/vision_tower_goldens.py, which runs the checkpoint's
 * OWN inference/vision.py on deterministic patches and dumps the input plus
 * stage outputs.  This gate runs the engine's forward on the same patches and
 * compares the aligner result.
 *
 * TOLERANCE, not equality.  The reference attention goes through
 * F.scaled_dot_product_attention, whose reduction order a hand-written
 * attention cannot reproduce, and every stage rounds to bf16 (~3 decimal
 * digits); the two implementations are therefore expected to differ by a few
 * bf16 ulps per operation.  The gate reports relative-RMS and max-abs error and
 * fails above TOWER_REL_RMS_MAX.  The layout/pixel/codec gates ARE exact; this
 * one cannot be, and pretending otherwise would either fail honest code or
 * force a meaningless threshold.
 *
 * MODEL-DEPENDENT, GPU-resident (like session_payload_gate).
 *
 * usage: ./tests/vision_tower_gate MODEL GOLDENS
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "pulsar_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* NO absolute threshold: the goldens carry the reference's OWN bf16-vs-fp32
 * gap per case, and the gate requires our bf16 output to be no further from the
 * reference's bf16 than bf16 itself is from fp32.  That is the strongest
 * statement available for a tower whose attention goes through
 * F.scaled_dot_product_attention (which no hand-written attention reproduces bit
 * for bit), and it is self-calibrating -- a hard-coded tolerance would just be a
 * number someone picked.
 *
 * WHY 3x AND NOT 1x: our difference is not a dtype cast, it is a different
 * REDUCTION ORDER in every GEMM and in the attention, which is a strictly larger
 * perturbation than rounding the same sums to bf16 -- so being within a small
 * multiple of the floor is evidence of correctness, not a concession.  What the
 * gate is actually for is bugs of the size this path has already produced: the
 * wqkv-layout bug measured ~150x the floor and the first draft ~40x, so a 3x
 * line catches anything real with two orders of magnitude to spare.
 *
 * MEASURED 2026-09-12 (production-shaped patches): patch_embed 1.4e-06 (exact),
 * block0 3.5e-03, final_norm 3.3e-02 -- a steady per-block accumulation, which
 * is the signature of reduction-order noise rather than a localised defect. */
#define TOWER_FLOOR_SLACK 3.0

static float bf16_to_f32(uint16_t bits) {
    uint32_t u = (uint32_t)bits << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static uint32_t rd_u32(FILE *f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) { fprintf(stderr, "tower gate: short read\n"); exit(2); }
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}
static int rd_i32(FILE *f) { return (int)rd_u32(f); }
static float rd_f32(FILE *f) {
    uint32_t u = rd_u32(f);
    float v; memcpy(&v, &u, sizeof v);
    return v;
}
static void rd(void *dst, size_t n, FILE *f) {
    if (n && fread(dst, 1, n, f) != n) { fprintf(stderr, "tower gate: short read\n"); exit(2); }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL GOLDENS\n", argv[0]);
        return 2;
    }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1];
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "tower gate: engine open failed\n"); return 2; }
    if (!e->vision_ready) {
        fprintf(stderr, "tower gate: the artifact carries no vision tower\n");
        return 2;
    }

    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "tower gate: cannot open %s\n", argv[2]); return 2; }
    char magic[4];
    rd(magic, 4, f);
    if (memcmp(magic, "VTX2", 4)) { fprintf(stderr, "tower gate: bad magic\n"); return 2; }
    const uint32_t n_cases = rd_u32(f);

    int failures = 0;
    for (uint32_t c = 0; c < n_cases; c++) {
        const int n_h = rd_i32(f), n_w = rd_i32(f), patch = rd_i32(f), dim = rd_i32(f);
        const int heads = rd_i32(f), inter = rd_i32(f), ratio = rd_i32(f);
        (void)rd_f32(f);                       /* rope theta, already compiled in */
        const uint32_t n_stage = rd_u32(f);
        const int text_dim = rd_i32(f);
        const double floor_rel = (double)rd_f32(f);   /* the reference's own bf16-vs-fp32 gap */
        const size_t n_tok = (size_t)n_h * n_w;
        const size_t n_llm = (size_t)((n_h + ratio - 1) / ratio) * ((n_w + ratio - 1) / ratio);

        const size_t stage_elems = n_tok * (size_t)dim;
        uint16_t *patches = (uint16_t *)malloc(n_tok * 3 * patch * patch * sizeof(uint16_t));
        uint16_t *want_stage = (uint16_t *)malloc((size_t)(2 + n_stage) * stage_elems * sizeof(uint16_t));
        uint16_t *got_stage = (uint16_t *)malloc((size_t)(2 + n_stage) * stage_elems * sizeof(uint16_t));
        uint16_t *want = (uint16_t *)malloc(n_llm * text_dim * sizeof(uint16_t));
        uint16_t *got = (uint16_t *)malloc(n_llm * text_dim * sizeof(uint16_t));
        if (!patches || !want_stage || !got_stage || !want || !got) {
            fprintf(stderr, "tower gate: oom\n");
            return 2;
        }
        rd(patches, n_tok * 3 * patch * patch * sizeof(uint16_t), f);
        rd(want_stage, (size_t)(2 + n_stage) * stage_elems * sizeof(uint16_t), f);
        rd(want, n_llm * text_dim * sizeof(uint16_t), f);

        int rows = 0;
        const int ok = vision_forward(&e->vision_weights, &e->model, patches, n_h, n_w,
                                      got, (int)(n_llm * text_dim), &rows,
                                      got_stage, n_stage);
        if (ok) {
            /* Stage-by-stage FIRST: a bad patch_embed means the weight plumbing
             * is wrong, which is a different bug from a wrong rope/norm, and
             * without this split the aligner number says only "something". */
            printf("  case %u (%dx%d):\n", c, n_h, n_w);
            for (uint32_t st = 0; st < 2 + n_stage; st++) {
                const uint16_t *g = got_stage + (size_t)st * stage_elems;
                const uint16_t *w = want_stage + (size_t)st * stage_elems;
                double sq = 0.0, rq = 0.0, ma = 0.0;
                for (size_t i = 0; i < stage_elems; i++) {
                    const double a = bf16_to_f32(g[i]), b = bf16_to_f32(w[i]);
                    const double d = a - b;
                    sq += d * d; rq += b * b;
                    if (fabs(d) > ma) ma = fabs(d);
                }
                char nm[32];
                if (st == 0) snprintf(nm, sizeof nm, "patch_embed");
                else if (st == 1 + n_stage) snprintf(nm, sizeof nm, "final_norm");
                else snprintf(nm, sizeof nm, "block%u", st - 1);
                printf("    %-12s rel-RMS %.3e  max-abs %.3e\n",
                       nm, rq > 0.0 ? sqrt(sq / rq) : sqrt(sq), ma);
            }
        }
        if (!ok || rows != (int)n_llm) {
            fprintf(stderr, "  FAIL case %u (%dx%d): forward refused or returned %d rows (want %zu)\n",
                    c, n_h, n_w, rows, n_llm);
            failures++;
        } else {
            double sumsq = 0.0, refsq = 0.0, maxabs = 0.0;
            size_t first = 0;
            int have_first = 0;
            for (size_t i = 0; i < n_llm * (size_t)text_dim; i++) {
                const double a = bf16_to_f32(got[i]), b = bf16_to_f32(want[i]);
                const double d = a - b;
                sumsq += d * d;
                refsq += b * b;
                if (fabs(d) > maxabs) { maxabs = fabs(d); first = i; have_first = 1; }
            }
            const double rel = refsq > 0.0 ? sqrt(sumsq / refsq) : sqrt(sumsq);
            const double limit = floor_rel * TOWER_FLOOR_SLACK;
            const int pass = rel <= limit;
            printf("  %s case %u: %dx%d -> %zu aligner rows, rel-RMS %.3e, max-abs %.3e "
                   "(bf16 floor %.3e, limit %.3e)",
                   pass ? "PASS" : "FAIL", c, n_h, n_w, n_llm, rel, maxabs, floor_rel, limit);
            if (!pass && have_first)
                printf(" (worst at row %zu col %zu: got %.6g want %.6g)",
                       first / (size_t)text_dim, first % (size_t)text_dim,
                       bf16_to_f32(got[first]), bf16_to_f32(want[first]));
            printf("\n");
            if (!pass) failures++;
        }
        free(patches); free(want_stage); free(got_stage); free(want); free(got);
    }
    fclose(f);
    printf("VISION-TOWER GATE: %s (%u cases, %d failures; rel-RMS must stay within "
           "%.2fx the reference's own bf16-vs-fp32 floor)\n",
           failures ? "FAIL" : "PASS", n_cases, failures, (double)TOWER_FLOOR_SLACK);
    return failures ? 1 : 0;
}

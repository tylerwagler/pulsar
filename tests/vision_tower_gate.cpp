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

/* First-pass threshold, to be tightened from the first measurement rather than
 * argued.  bf16 carries ~2^-8 relative precision, so a few ulps per op over 32
 * blocks is expected; anything approaching this is a real disagreement. */
#define TOWER_REL_RMS_MAX 3.0e-2f

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
    if (memcmp(magic, "VTX1", 4)) { fprintf(stderr, "tower gate: bad magic\n"); return 2; }
    const uint32_t n_cases = rd_u32(f);

    int failures = 0;
    for (uint32_t c = 0; c < n_cases; c++) {
        const int n_h = rd_i32(f), n_w = rd_i32(f), patch = rd_i32(f), dim = rd_i32(f);
        const int heads = rd_i32(f), inter = rd_i32(f), ratio = rd_i32(f);
        (void)rd_f32(f);                       /* rope theta, already compiled in */
        const uint32_t n_stage = rd_u32(f);
        const int text_dim = rd_i32(f);
        const size_t n_tok = (size_t)n_h * n_w;
        const size_t n_llm = (size_t)((n_h + ratio - 1) / ratio) * ((n_w + ratio - 1) / ratio);

        uint16_t *patches = (uint16_t *)malloc(n_tok * 3 * patch * patch * sizeof(uint16_t));
        uint16_t *scratch = (uint16_t *)malloc((n_tok * dim + n_llm * text_dim) * sizeof(uint16_t));
        uint16_t *want = (uint16_t *)malloc(n_llm * text_dim * sizeof(uint16_t));
        uint16_t *got = (uint16_t *)malloc(n_llm * text_dim * sizeof(uint16_t));
        if (!patches || !scratch || !want || !got) { fprintf(stderr, "tower gate: oom\n"); return 2; }
        rd(patches, n_tok * 3 * patch * patch * sizeof(uint16_t), f);
        rd(scratch, n_tok * dim * sizeof(uint16_t), f);                    /* patch_embed */
        for (uint32_t s = 0; s < n_stage; s++) rd(scratch, n_tok * dim * sizeof(uint16_t), f);
        rd(scratch, n_tok * dim * sizeof(uint16_t), f);                    /* final norm */
        rd(want, n_llm * text_dim * sizeof(uint16_t), f);

        int rows = 0;
        const int ok = vision_forward(&e->vision_weights, &e->model, patches, n_h, n_w,
                                      got, (int)(n_llm * text_dim), &rows);
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
            const int pass = rel <= TOWER_REL_RMS_MAX;
            printf("  %s case %u: %dx%d -> %zu aligner rows, rel-RMS %.3e, max-abs %.3e",
                   pass ? "PASS" : "FAIL", c, n_h, n_w, n_llm, rel, maxabs);
            if (!pass && have_first)
                printf(" (worst at row %zu col %zu: got %.6g want %.6g)",
                       first / (size_t)text_dim, first % (size_t)text_dim,
                       bf16_to_f32(got[first]), bf16_to_f32(want[first]));
            printf("\n");
            if (!pass) failures++;
        }
        free(patches); free(scratch); free(want); free(got);
    }
    fclose(f);
    printf("VISION-TOWER GATE: %s (%u cases, %d failures; rel-RMS limit %.1e)\n",
           failures ? "FAIL" : "PASS", n_cases, failures, (double)TOWER_REL_RMS_MAX);
    return failures ? 1 : 0;
}

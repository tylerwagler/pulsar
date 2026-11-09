/* L216 vision-merge gate: the engine's span embeddings vs the reference's.
 *
 * `vision_merge_span()` is the whole per-image path as the model consumes it:
 * ViT + aligner over the prepared patches, then the scatter into the span --
 * IMAGE slots take the aligner rows in `perm` order, every other slot takes its
 * type's learned vector (image_start / image_pad / image_newline / image_end).
 * The goldens come from the reference's own `merge_image_embeddings`.
 *
 * TOLERANCE, floor-calibrated: this contains the tower forward, whose attention
 * cannot reproduce F.scaled_dot_product_attention's reduction order.  The
 * goldens carry the reference's own bf16-vs-fp32 gap per case, and the gate
 * requires us to stay within the same 3x used by vision-tower-gate (a different
 * reduction order is a larger perturbation than a dtype cast).
 *
 * MODEL-DEPENDENT, GPU-resident (like vision_tower_gate).
 *
 * usage: ./tests/vision_merge_gate MODEL GOLDENS
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "pulsar_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MERGE_FLOOR_SLACK 3.0

static float bf16_to_f32(uint16_t bits) {
    uint32_t u = (uint32_t)bits << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}
static uint32_t rd_u32(FILE *f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) { fprintf(stderr, "merge gate: short read\n"); exit(2); }
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}
static int rd_i32(FILE *f) { return (int)rd_u32(f); }
static float rd_f32(FILE *f) {
    uint32_t u = rd_u32(f);
    float v; memcpy(&v, &u, sizeof v);
    return v;
}
static void rd(void *dst, size_t n, FILE *f) {
    if (n && fread(dst, 1, n, f) != n) { fprintf(stderr, "merge gate: short read\n"); exit(2); }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL GOLDENS\n", argv[0]); return 2; }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1];
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "merge gate: engine open failed\n"); return 2; }
    if (!e->vision_ready) { fprintf(stderr, "merge gate: no vision tower in this artifact\n"); return 2; }

    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "merge gate: cannot open %s\n", argv[2]); return 2; }
    char magic[4];
    rd(magic, 4, f);
    if (memcmp(magic, "VMG1", 4)) { fprintf(stderr, "merge gate: bad magic\n"); return 2; }
    const uint32_t n_cases = rd_u32(f);
    int failures = 0;

    for (uint32_t c = 0; c < n_cases; c++) {
        const int n_h = rd_i32(f), n_w = rd_i32(f), patch = rd_i32(f), text_dim = rd_i32(f);
        const int start_pos = rd_i32(f);
        const int span_len = rd_i32(f);
        const double floor_rel = (double)rd_f32(f);
        const int n_perm = rd_i32(f);
        (void)start_pos;
        const int r = (int)PULSAR_VISION_DOWNSAMPLE;
        const size_t n_patch = (size_t)n_h * n_w;

        uint16_t *patches = (uint16_t *)malloc(n_patch * 3u * patch * patch * sizeof(uint16_t));
        int32_t *types = (int32_t *)malloc((size_t)span_len * sizeof(int32_t));
        int32_t *perm = (int32_t *)malloc((size_t)n_perm * sizeof(int32_t));
        uint16_t *want = (uint16_t *)malloc((size_t)span_len * text_dim * sizeof(uint16_t));
        uint16_t *got = (uint16_t *)malloc((size_t)span_len * text_dim * sizeof(uint16_t));
        if (!patches || !types || !perm || !want || !got) { fprintf(stderr, "merge gate: oom\n"); return 2; }
        rd(patches, n_patch * 3u * patch * patch * sizeof(uint16_t), f);
        rd(types, (size_t)span_len * sizeof(int32_t), f);
        rd(perm, (size_t)n_perm * sizeof(int32_t), f);
        rd(want, (size_t)span_len * text_dim * sizeof(uint16_t), f);

        pulsar_vision_prepared prep;
        memset(&prep, 0, sizeof prep);
        prep.patches = patches;
        prep.span_types = types;
        prep.perm = perm;
        prep.n_vit_h = n_h;
        prep.n_vit_w = n_w;
        prep.n_llm_h = (n_h + r - 1) / r;
        prep.n_llm_w = (n_w + r - 1) / r;
        prep.span_len = span_len;
        prep.n_perm = n_perm;

        int out_len = 0;
        const int ok = vision_merge_span(&e->vision_weights, &e->model, &prep,
                                         got, span_len * text_dim, &out_len);
        if (!ok || out_len != span_len) {
            fprintf(stderr, "  FAIL case %u (%dx%d): merge refused or returned %d (want %d)\n",
                    c, n_h, n_w, out_len, span_len);
            failures++;
        } else {
            double sumsq = 0.0, refsq = 0.0, maxabs = 0.0;
            for (size_t i = 0; i < (size_t)span_len * text_dim; i++) {
                const double a = bf16_to_f32(got[i]), b = bf16_to_f32(want[i]);
                const double d = a - b;
                sumsq += d * d; refsq += b * b;
                if (fabs(d) > maxabs) maxabs = fabs(d);
            }
            const double rel = refsq > 0.0 ? sqrt(sumsq / refsq) : sqrt(sumsq);
            const double limit = floor_rel * MERGE_FLOOR_SLACK;
            const int pass = rel <= limit;
            printf("  %s case %u: %dx%d -> span %d, rel-RMS %.3e, max-abs %.3e "
                   "(bf16 floor %.3e, limit %.3e)\n",
                   pass ? "PASS" : "FAIL", c, n_h, n_w, span_len, rel, maxabs, floor_rel, limit);
            if (!pass) failures++;
        }
        free(patches); free(types); free(perm); free(want); free(got);
    }
    fclose(f);
    printf("VISION-MERGE GATE: %s (%u cases, %d failures; within %.2fx the reference's own bf16 floor)\n",
           failures ? "FAIL" : "PASS", n_cases, failures, (double)MERGE_FLOOR_SLACK);
    return failures ? 1 : 0;
}

/* L216 vision-span gate: the composed per-image path vs the reference's.
 *
 * `vision_prepare_image()` is decode + preprocess + the sentinel block -- the
 * reference's load_image followed by build_image_block(n_llm_h, n_llm_w,
 * start_pos), which is what prepare_vl_inputs does per image.  Each piece was
 * gated separately (codec, pixel, layout); this grades the COMPOSITION, which is
 * what the server will actually call, and it is exact: every input here is
 * integer or fixed-point, so unlike the tower gate there is no tolerance.
 *
 *   ./tests/vision_span_gate [goldens-file] */
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd_u32(FILE *f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) { fprintf(stderr, "span gate: short read\n"); exit(2); }
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}
static int rd_i32(FILE *f) { return (int)rd_u32(f); }
static float rd_f32(FILE *f) {
    uint32_t u = rd_u32(f);
    float v; memcpy(&v, &u, sizeof v);
    return v;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "tests/test-vectors/vision-span-goldens.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "vision-span gate: cannot open %s\n", path); return 2; }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "VSP1", 4)) {
        fprintf(stderr, "vision-span gate: bad magic\n");
        return 2;
    }
    const uint32_t n = rd_u32(f);
    int failures = 0;
    for (uint32_t c = 0; c < n; c++) {
        const int w = rd_i32(f), h = rd_i32(f);
        pulsar_vision_args a;
        a.patch_size = rd_i32(f);
        a.downsample_ratio = rd_i32(f);
        a.max_n_token = rd_i32(f);
        a.min_pixels = rd_i32(f);
        a.max_wh_ratio = rd_f32(f);
        const int vocab = rd_i32(f);
        const int start_pos = rd_i32(f);
        const int want_vh = rd_i32(f), want_vw = rd_i32(f);
        const int want_lh = rd_i32(f), want_lw = rd_i32(f);
        const int want_span = rd_i32(f), enc_len = rd_i32(f);
        uint8_t *enc = (uint8_t *)malloc((size_t)enc_len);
        int32_t *want_ids = (int32_t *)malloc((size_t)want_span * sizeof(int32_t));
        int32_t *want_types = (int32_t *)malloc((size_t)want_span * sizeof(int32_t));
        const int want_perm_n = want_lh * want_lw;   /* perm indexes aligner rows */
        int32_t *want_perm = (int32_t *)malloc((size_t)want_perm_n * sizeof(int32_t));
        if (!enc || !want_ids || !want_types || !want_perm) { fprintf(stderr, "span gate: oom\n"); return 2; }
        if (fread(enc, 1, (size_t)enc_len, f) != (size_t)enc_len ||
            fread(want_ids, sizeof(int32_t), (size_t)want_span, f) != (size_t)want_span ||
            fread(want_types, sizeof(int32_t), (size_t)want_span, f) != (size_t)want_span ||
            fread(want_perm, sizeof(int32_t), (size_t)want_perm_n, f) != (size_t)want_perm_n) {
            fprintf(stderr, "span gate: short read\n");
            return 2;
        }

        pulsar_vision_prepared got;
        const int ok = vision_prepare_image(enc, (size_t)enc_len, &a, start_pos, vocab, &got);
        int bad = 0;
        if (!ok) {
            fprintf(stderr, "  FAIL case %u (%dx%d start=%d): prepare refused\n", c, w, h, start_pos);
            failures++;
        } else {
            if (got.n_vit_h != want_vh || got.n_vit_w != want_vw ||
                got.n_llm_h != want_lh || got.n_llm_w != want_lw) {
                fprintf(stderr, "  FAIL case %u: grid got vit %dx%d llm %dx%d want vit %dx%d llm %dx%d\n",
                        c, got.n_vit_h, got.n_vit_w, got.n_llm_h, got.n_llm_w,
                        want_vh, want_vw, want_lh, want_lw);
                bad = 1;
            }
            if (got.span_len != want_span) {
                fprintf(stderr, "  FAIL case %u: span len got %d want %d\n", c, got.span_len, want_span);
                bad = 1;
            } else {
                for (int i = 0; i < want_span && !bad; i++) {
                    if (got.span_ids[i] != want_ids[i] || got.span_types[i] != want_types[i]) {
                        fprintf(stderr, "  FAIL case %u (%dx%d start=%d): span[%d] id/type got %d/%d want %d/%d\n",
                                c, w, h, start_pos, i, got.span_ids[i], got.span_types[i],
                                want_ids[i], want_types[i]);
                        bad = 1;
                    }
                }
            }
            if (!bad && got.n_perm != want_perm_n) {
                fprintf(stderr, "  FAIL case %u: perm len got %d want %d\n", c, got.n_perm, want_perm_n);
                bad = 1;
            }
            for (int i = 0; i < want_perm_n && !bad; i++) {
                if (got.perm[i] != want_perm[i]) {
                    fprintf(stderr, "  FAIL case %u: perm[%d] got %d want %d\n", c, i, got.perm[i], want_perm[i]);
                    bad = 1;
                }
            }
            if (bad) failures++;
            vision_prepared_free(&got);
        }
        free(enc); free(want_ids); free(want_types); free(want_perm);
    }
    fclose(f);
    printf("VISION-SPAN GATE: %s (%u cases, %d failures)\n", failures ? "FAIL" : "PASS", n, failures);
    return failures ? 1 : 0;
}

/* L216 vision-pixel gate: grade src/engine/vision.cpp's pixel path against the
 * checkpoint's own image_processor.py.
 *
 * Goldens come from tests/vision_pixel_goldens.py, which RUNS the reference's
 * load_image() and records, per case, the DECODED RGB it resampled plus the ViT
 * patches it produced.  The gate is handed the same RGB, so what is graded here
 * is the resample/contain/pad (Pillow BICUBIC, 127-grey canvas), the
 * (x/255-0.5)/0.5 normalisation, the bf16 rounding and the patchify -- bit for
 * bit.  The codec is out of scope on purpose (PNG is lossless, so the recorded
 * RGB is exactly what any PNG decoder must yield).
 *
 *   ./tests/vision_pixel_gate [goldens-file] */
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd_u32(FILE *f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) { fprintf(stderr, "pixel gate: short read\n"); exit(2); }
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}
static int rd_i32(FILE *f) { return (int)rd_u32(f); }
static float rd_f32(FILE *f) {
    uint32_t u = rd_u32(f);
    float v;
    memcpy(&v, &u, sizeof v);
    return v;
}
static void rd_bytes(FILE *f, void *dst, size_t n) {
    if (n && fread(dst, 1, n, f) != n) { fprintf(stderr, "pixel gate: short read\n"); exit(2); }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "tests/test-vectors/vision-pixel-goldens.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "vision-pixel gate: cannot open %s\n", path); return 2; }
    char magic[4];
    rd_bytes(f, magic, 4);
    if (memcmp(magic, "VPX1", 4)) { fprintf(stderr, "vision-pixel gate: bad magic\n"); return 2; }
    const uint32_t n_cases = rd_u32(f);

    int failures = 0, cases = 0;
    for (uint32_t c = 0; c < n_cases; c++) {
        const int w = rd_i32(f), h = rd_i32(f);
        pulsar_vision_args a;
        a.patch_size = rd_i32(f);
        a.downsample_ratio = rd_i32(f);
        a.max_n_token = rd_i32(f);
        a.min_pixels = rd_i32(f);
        a.max_wh_ratio = rd_f32(f);
        const int want_vh = rd_i32(f), want_vw = rd_i32(f);
        const int want_lh = rd_i32(f), want_lw = rd_i32(f);

        uint8_t *rgb = (uint8_t *)malloc((size_t)w * (size_t)h * 3u);
        if (!rgb || fread(rgb, 1, (size_t)w * (size_t)h * 3u, f) != (size_t)w * (size_t)h * 3u) {
            fprintf(stderr, "pixel gate: short RGB read\n");
            return 2;
        }
        const size_t n_patch = (size_t)want_vh * (size_t)want_vw;
        const size_t per_patch = (size_t)3 * a.patch_size * a.patch_size;
        uint16_t *want = (uint16_t *)malloc(n_patch * per_patch * sizeof(uint16_t));
        uint16_t *got = (uint16_t *)malloc(n_patch * per_patch * sizeof(uint16_t));
        if (!want || !got) { fprintf(stderr, "pixel gate: oom\n"); return 2; }
        rd_bytes(f, want, n_patch * per_patch * sizeof(uint16_t));

        pulsar_vision_image img;
        const int ok = vision_preprocess_rgb(rgb, w, h, &a, got, n_patch * per_patch, &img);
        cases++;
        if (!ok) {
            fprintf(stderr, "  FAIL case %u (%dx%d): vision_preprocess_rgb refused\n", c, w, h);
            failures++;
        } else if (img.n_vit_h != want_vh || img.n_vit_w != want_vw ||
                   img.n_llm_h != want_lh || img.n_llm_w != want_lw) {
            fprintf(stderr, "  FAIL case %u (%dx%d): grid got vit %dx%d llm %dx%d want vit %dx%d llm %dx%d\n",
                    c, w, h, img.n_vit_h, img.n_vit_w, img.n_llm_h, img.n_llm_w,
                    want_vh, want_vw, want_lh, want_lw);
            failures++;
        } else {
            size_t ndiff = 0, first = 0;
            for (size_t i = 0; i < n_patch * per_patch; i++) {
                if (got[i] != want[i]) {
                    if (!ndiff) first = i;
                    ndiff++;
                }
            }
            if (ndiff) {
                fprintf(stderr, "  FAIL case %u (%dx%d): %zu of %zu bf16 patch values differ; "
                        "first at %zu got 0x%04x want 0x%04x (patch %zu, ch %zu, y %zu, x %zu)\n",
                        c, w, h, ndiff, n_patch * per_patch, first,
                        got[first], want[first],
                        first / per_patch,
                        (first % per_patch) / ((size_t)a.patch_size * a.patch_size),
                        (first % ((size_t)a.patch_size * a.patch_size)) / a.patch_size,
                        first % a.patch_size);
                failures++;
            }
        }
        free(rgb); free(want); free(got);
    }
    fclose(f);
    printf("VISION-PIXEL GATE: %s (%d cases, %d failures)\n",
           failures ? "FAIL" : "PASS", cases, failures);
    return failures ? 1 : 0;
}

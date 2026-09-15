/* L216 vision-codec gate: our decoder vs Pillow's, on the same bytes.
 *
 * The reference decodes with Pillow (libpng for PNG, libjpeg-turbo at libjpeg's
 * defaults for JPEG) and the engine links the same two libraries, so the claim
 * "the decoded pixels match" is testable rather than assumed.  Goldens come from
 * tests/vision_codec_goldens.py, which records the encoded file and the RGB
 * Pillow's .convert("RGB") produced from it.
 *
 *   ./tests/vision_codec_gate [goldens-file] */
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd_u32(FILE *f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) { fprintf(stderr, "codec gate: short read\n"); exit(2); }
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}
static int rd_i32(FILE *f) { return (int)rd_u32(f); }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "tests/test-vectors/vision-codec-goldens.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "vision-codec gate: cannot open %s\n", path); return 2; }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "VCX1", 4)) {
        fprintf(stderr, "vision-codec gate: bad magic\n");
        return 2;
    }
    const uint32_t n = rd_u32(f);
    int failures = 0;
    for (uint32_t c = 0; c < n; c++) {
        const uint32_t fmt = rd_u32(f);
        const int w = rd_i32(f), h = rd_i32(f);
        const uint32_t enc_len = rd_u32(f);
        uint8_t *enc = (uint8_t *)malloc(enc_len ? enc_len : 1);
        uint8_t *want = (uint8_t *)malloc((size_t)w * (size_t)h * 3u);
        if (!enc || !want) { fprintf(stderr, "codec gate: oom\n"); return 2; }
        if (fread(enc, 1, enc_len, f) != enc_len ||
            fread(want, 1, (size_t)w * (size_t)h * 3u, f) != (size_t)w * (size_t)h * 3u) {
            fprintf(stderr, "codec gate: short read\n");
            return 2;
        }
        uint8_t *got = NULL;
        int gw = 0, gh = 0;
        const int ok = vision_decode_rgb(enc, enc_len, &got, &gw, &gh);
        const char *kind = fmt == 0 ? "png" : "jpeg";
        if (!ok) {
            fprintf(stderr, "  FAIL case %u (%s %dx%d): vision_decode_rgb refused\n", c, kind, w, h);
            failures++;
        } else if (gw != w || gh != h) {
            fprintf(stderr, "  FAIL case %u (%s): size got %dx%d want %dx%d\n", c, kind, gw, gh, w, h);
            failures++;
        } else {
            const size_t total = (size_t)w * (size_t)h * 3u;
            size_t ndiff = 0, first = 0, maxdelta = 0;
            for (size_t i = 0; i < total; i++) {
                if (got[i] != want[i]) {
                    const int d = (int)got[i] - (int)want[i];
                    if (!ndiff) first = i;
                    ndiff++;
                    if ((size_t)(d < 0 ? -d : d) > maxdelta) maxdelta = (size_t)(d < 0 ? -d : d);
                }
            }
            if (ndiff) {
                fprintf(stderr, "  FAIL case %u (%s %dx%d): %zu of %zu bytes differ "
                        "(max delta %zu); first at byte %zu (px %zu, ch %zu) got %u want %u\n",
                        c, kind, w, h, ndiff, total, maxdelta, first, first / 3, first % 3,
                        got[first], want[first]);
                failures++;
            }
        }
        free(got);
        free(enc);
        free(want);
    }
    fclose(f);
    printf("VISION-CODEC GATE: %s (%u cases, %d failures)\n",
           failures ? "FAIL" : "PASS", n, failures);
    return failures ? 1 : 0;
}

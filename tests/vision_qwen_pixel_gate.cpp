/* L268 gate: Qwen3.8-Flash-Next's image PREPROCESSING against HF's own processor, exactly.
 *
 * The golden (tests/vision_qwen_pixel_goldens.py) ran transformers' Qwen2VLImageProcessorPil on encoded images that
 * cover smart_resize's branches (plain rounding, the min_pixels upscale, a wide aspect, a JPEG).  This gate feeds the
 * SAME encoded bytes to qwen_vision_preprocess -- the shared codec and Pillow resize (vision.cpp), Qwen's
 * smart_resize, rescale / normalise and merge-block patchify (vision_qwen.cpp) -- and requires:
 *   - the resized size and the patch grid, exactly;
 *   - every pixel_values float32, bit for bit (an FNV-1a over the whole array, row-major);
 *   - a spread of sample rows, bit for bit, so a failure says WHERE.
 * No model, no GPU. */
#include "pulsar.h"
#include "pulsar_engine_internal.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static void rd(void *dst, size_t n, FILE *f) {
    if (n && fread(dst, 1, n, f) != n) { fprintf(stderr, "qwen pixel gate: short read\n"); exit(2); }
}
static uint32_t rd_u32(FILE *f) { uint32_t v; rd(&v, 4, f); return v; }

static uint64_t fnv1a(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 0xCBF29CE484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001B3ull; }
    return h;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "tests/test-vectors/vision-qwen-pixel-goldens.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "qwen pixel gate: cannot open %s\n", path); return 2; }
    char magic[4];
    rd(magic, 4, f);
    if (memcmp(magic, "QVP1", 4)) { fprintf(stderr, "qwen pixel gate: bad magic\n"); return 2; }
    const uint32_t n_cases = rd_u32(f);
    int failures = 0;
    for (uint32_t c = 0; c < n_cases; c++) {
        std::vector<uint8_t> enc(rd_u32(f));
        rd(enc.data(), enc.size(), f);
        uint32_t hdr[8];
        rd(hdr, sizeof hdr, f);
        const uint32_t src_w = hdr[0], src_h = hdr[1], want_rh = hdr[2], want_rw = hdr[3], want_gh = hdr[4],
                       want_gw = hdr[5], want_rows = hdr[6], want_cols = hdr[7];
        uint64_t want_fnv;
        rd(&want_fnv, 8, f);
        const uint32_t n_samples = rd_u32(f);
        std::vector<uint32_t> sample_row(n_samples);
        std::vector<float> sample((size_t)n_samples * want_cols);
        for (uint32_t k = 0; k < n_samples; k++) {
            sample_row[k] = rd_u32(f);
            rd(&sample[(size_t)k * want_cols], (size_t)want_cols * 4, f);
        }

        qwen_vision_pixels px;
        char err[256] = "";
        int bad = 0;
        if (!qwen_vision_preprocess(enc.data(), enc.size(), &px, err, sizeof err)) {
            printf("  FAIL case %u (%ux%u): preprocess refused: %s\n", c, src_w, src_h, err);
            failures++;
            continue;
        }
        if ((uint32_t)px.resized_h != want_rh || (uint32_t)px.resized_w != want_rw || (uint32_t)px.grid_h != want_gh ||
            (uint32_t)px.grid_w != want_gw || (uint32_t)px.rows != want_rows || (uint32_t)px.cols != want_cols) {
            printf("  FAIL case %u (%ux%u): resized %dx%d grid %dx%d rows %d cols %d, want %ux%u grid %ux%u rows %u "
                   "cols %u\n", c, src_w, src_h, px.resized_w, px.resized_h, px.grid_w, px.grid_h, px.rows, px.cols,
                   want_rw, want_rh, want_gw, want_gh, want_rows, want_cols);
            bad++;
        }
        for (uint32_t k = 0; !bad && k < n_samples; k++) {
            const float *got = px.values + (size_t)sample_row[k] * px.cols;
            const float *want = &sample[(size_t)k * want_cols];
            for (uint32_t j = 0; j < want_cols; j++) {
                if (memcmp(&got[j], &want[j], 4) != 0) {
                    printf("  FAIL case %u: row %u col %u = %.9g, want %.9g (channel %u frame %u y %u x %u)\n", c,
                           sample_row[k], j, (double)got[j], (double)want[j], j / 512, (j / 256) % 2, (j / 16) % 16,
                           j % 16);
                    bad++;
                    break;
                }
            }
        }
        const uint64_t got_fnv = bad ? 0 : fnv1a(px.values, (size_t)px.rows * px.cols * 4);
        if (!bad && got_fnv != want_fnv) {
            printf("  FAIL case %u: pixel_values digest %016llx, want %016llx (the samples matched)\n", c,
                   (unsigned long long)got_fnv, (unsigned long long)want_fnv);
            bad++;
        }
        printf("%s case %u: %ux%u -> %dx%d, grid %dx%d, %d patches\n", bad ? "FAIL" : "ok  ", c, src_w, src_h,
               px.resized_w, px.resized_h, px.grid_w, px.grid_h, px.rows);
        failures += bad;
        qwen_vision_pixels_free(&px);
    }
    fclose(f);
    printf("VISION QWEN PIXEL GATE: %s (%u cases, %d failures)\n", failures ? "FAIL" : "PASS", n_cases, failures);
    return failures ? 1 : 0;
}

/* L331 nemotron-vision-pixel-gate: Nemotron 3.5 Super VL's image preprocessing and image front against the checkpoint's
 * own processor -- model-free, host only (no GPU, no weights).
 *
 * Goldens: tests/test-vectors/nemotron-vision-goldens.bin (tests/nemotron_vision_goldens.py, the checkpoint's
 * image_processing_nemotron_h_omni.py on torch's CPU).
 *   S  the tile rule (pulsar_nemotron_h_tile) == the processor's _compute_target_patches for every size: patches and
 *      LLM tokens (0 tolerance)
 *   P  the pixels (pulsar_nemotron_h_image_prepare: the engine's codec, torch's antialiased bicubic, the
 *      normalisation): the tile's size; the bf16 values the tower reads bit-exact where the golden carries them all,
 *      the f32 sample bit-exact everywhere (PIX_F32_ULPS / PIX_BF16_FLIPS 0: the resize reproduces the contraction
 *      of torch's build, vision.cpp; a torch built otherwise would move by an ulp, and the gate says so); the patches
 *      in merge-block order == the pixels
 *   C  the CPE position rows (pulsar_nemotron_h_positions) == modeling_radio's own interpolation, on the golden's
 *      synthetic table, every value within POS_REL of the reference (torch's channels-last bilinear kernel sums the
 *      four taps in its vector order: within a few f32 ulps)
 *   F  the image front on a fake engine: a tower-less engine refuses an image by name (the client-visible 400); with
 *      the tower declared, the expansion is the tile's (w h / 4) `<image>` ids, one block (block_extent), the
 *      placeholder literal `<img><image></img>`
 * usage: ./tests/nemotron_vision_pixel_gate GOLDENS */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "families/nemotron_h/nemotron_h.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#define PIX_F32_ULPS 0          /* bit-exact: the resize reproduces torch's contraction (vision.cpp) */
#define PIX_BF16_FLIPS 0.0      /* and so every bf16 value the tower reads */
#define POS_REL 1e-6            /* the positions: relative to the table's scale (0.5) */

static int g_fail = 0, g_checks = 0;
#define CHECK(c, ...) do { g_checks++; if (!(c)) { g_fail++; printf("  FAIL "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static FILE *g_f;
static void rd(void *dst, size_t n) {
    if (n && fread(dst, 1, n, g_f) != n) { fprintf(stderr, "nemotron vision pixel gate: short read\n"); exit(2); }
}
static uint32_t u32() { uint32_t v; rd(&v, 4); return v; }

static uint16_t bf16_of(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static int64_t ulps(float a, float b) {
    int32_t x, y;
    memcpy(&x, &a, 4);
    memcpy(&y, &b, 4);
    if (x < 0) x = (int32_t)0x80000000 - x;
    if (y < 0) y = (int32_t)0x80000000 - y;
    return llabs((int64_t)x - (int64_t)y);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s GOLDENS\n", argv[0]); return 2; }
    g_f = fopen(argv[1], "rb");
    if (!g_f) { fprintf(stderr, "nemotron vision pixel gate: cannot open %s\n", argv[1]); return 2; }
    char magic[4];
    rd(magic, 4);
    if (memcmp(magic, "NVG1", 4)) { fprintf(stderr, "nemotron vision pixel gate: bad magic\n"); return 2; }
    const pulsar_nemotron_h_vision_shape *vs = &PULSAR_NEMOTRON_H_VISION_SHAPE;

    /* S: the tile rule */
    const uint32_t n_sizes = u32(), budget = u32();
    CHECK(budget == vs->max_patches, "the golden's budget %u is not the profile's max_patches %u", budget,
          vs->max_patches);
    int s_bad = 0;
    for (uint32_t i = 0; i < n_sizes; i++) {
        uint32_t r[5];
        rd(r, sizeof r);
        uint32_t wp = 0, hp = 0;
        pulsar_nemotron_h_tile(vs, (int)r[0], (int)r[1], &wp, &hp);
        const bool ok = wp == r[2] && hp == r[3] && wp * hp / 4 == r[4];
        if (!ok) printf("  FAIL size %ux%u: tile %ux%u, the processor's %ux%u (%u tokens)\n", r[0], r[1], wp, hp, r[2],
                        r[3], r[4]);
        g_checks++;
        g_fail += !ok;
        s_bad += !ok;
    }
    printf("S tile rule: %u sizes, %d mismatched\n", n_sizes, s_bad);

    /* P: the pixels */
    const uint32_t n_pix = u32();
    std::vector<uint8_t> first_png;
    uint32_t first_tok = 0;
    for (uint32_t c = 0; c < n_pix; c++) {
        std::vector<uint8_t> png(u32());
        rd(png.data(), png.size());
        const uint32_t H = u32(), W = u32(), ntok = u32(), stride = u32(), full = u32();
        const size_t n = (size_t)3 * H * W;
        std::vector<uint16_t> want_bf16(full ? n : 0);
        if (full) rd(want_bf16.data(), n * 2);
        std::vector<float> sample(u32());
        rd(sample.data(), sample.size() * 4);
        if (c == 0) { first_png = png; first_tok = ntok; }
        pulsar_nemotron_h_image img;
        char err[256] = "";
        const bool ok = pulsar_nemotron_h_image_prepare(vs, png.data(), png.size(), &img, err, sizeof err);
        CHECK(ok, "pixels %u: prepare refused: %s", c, err);
        if (!ok) continue;
        CHECK(img.hp * 16 == H && img.wp * 16 == W && img.wp * img.hp / 4 == ntok,
              "pixels %u: tile %ux%u px, the processor's %ux%u", c, img.wp * 16, img.hp * 16, W, H);
        if (img.pixels.size() != n) continue;
        size_t flips = 0, worse = 0;
        for (size_t i = 0; full && i < n; i++) {
            const uint16_t g = bf16_of(img.pixels[i]);
            if (g != want_bf16[i]) (abs((int)g - (int)want_bf16[i]) == 1 ? flips : worse)++;
        }
        int64_t max_ulp = 0;
        for (size_t k = 0; k < sample.size(); k++) {
            const int64_t u = ulps(img.pixels[k * stride], sample[k]);
            if (u > max_ulp) max_ulp = u;
        }
        CHECK(worse == 0 && (double)flips <= PIX_BF16_FLIPS * (double)n,
              "pixels %u: %zu bf16 values one step off (limit %.0e of %zu), %zu further", c, flips, PIX_BF16_FLIPS, n,
              worse);
        CHECK(max_ulp <= PIX_F32_ULPS, "pixels %u: an f32 value %lld ulps from torch's (limit %d)", c,
              (long long)max_ulp, PIX_F32_ULPS);
        /* the patches: merge-block order over the bf16 pixels */
        bool patch_ok = img.patches.size() == (size_t)img.wp * img.hp * 768;
        for (uint32_t r = 0; patch_ok && r < img.hp; r += 7)
            for (uint32_t col = 0; patch_ok && col < img.wp; col += 5) {
                const uint16_t *p = img.patches.data() + (size_t)pulsar_nemotron_h_patch_slot(r, col, img.wp) * 768;
                for (int ch = 0; ch < 3 && patch_ok; ch++)
                    for (int y = 0; y < 16 && patch_ok; y += 3)
                        for (int x = 0; x < 16 && patch_ok; x += 5)
                            patch_ok = p[(ch * 16 + y) * 16 + x] ==
                                       bf16_of(img.pixels[((size_t)ch * H + r * 16 + y) * W + col * 16 + x]);
            }
        CHECK(patch_ok, "pixels %u: a patch element is not its pixel (merge-block order)", c);
        printf("P pixels %u: %ux%u px, %u tokens; bf16 %s (%zu one step off), f32 sample max %lld ulps\n", c, W, H, ntok,
               full ? "graded" : "not stored", flips, (long long)max_ulp);
    }
    /* C: the CPE positions on the synthetic table */
    {
        const uint32_t C = u32(), n_tiles = u32();
        std::vector<float> field((size_t)vs->pos_grid * vs->pos_grid * C);
        uint64_t x = 331;
        for (float &v : field) {
            x = x * 6364136223846793005ull + 1442695040888963407ull;
            v = (float)((double)((x >> 40) & 0xFFFFu) / 65536.0 - 0.5);
        }
        pulsar_nemotron_h_vision_shape s = *vs;
        s.width = C;
        double worst = 0;
        for (uint32_t t = 0; t < n_tiles; t++) {
            const uint32_t wp = u32(), hp = u32();
            std::vector<float> want((size_t)wp * hp * C), got;
            rd(want.data(), want.size() * 4);
            pulsar_nemotron_h_positions(&s, field, wp, hp, &got);
            double d = 0;
            for (uint32_t r = 0; r < hp; r++)
                for (uint32_t c = 0; c < wp; c++)
                    for (uint32_t k = 0; k < C; k++)
                        d = fmax(d, fabs((double)got[(size_t)pulsar_nemotron_h_patch_slot(r, c, wp) * C + k] -
                                         want[((size_t)r * wp + c) * C + k]));
            worst = fmax(worst, d);
            CHECK(d <= POS_REL * 0.5, "positions %ux%u: max |diff| %.3e (limit %.1e)", wp, hp, d, POS_REL * 0.5);
        }
        printf("C positions: %u tiles, max |diff| %.3e\n", n_tiles, worst);
    }
    fclose(g_f);

    /* F: the image front on a fake engine */
    {
        pulsar_engine e;
        memset(&e, 0, sizeof e);
        e.family = &PULSAR_FAMILY_NEMOTRON_H;
        pulsar_nemotron_h_model *w = new pulsar_nemotron_h_model();
        e.family_state = w;
        pulsar_image_ref img = {first_png.data(), first_png.size(), -1};
        pulsar_tokens prompt = {0}, out = {0};
        pulsar_tokens_push(&prompt, 10);
        pulsar_tokens_push(&prompt, 18);   /* the renderer's <image> */
        pulsar_tokens_push(&prompt, 11);
        char err[256] = "";
        const bool refused = !pulsar_image_expand(&e, &prompt, 0, &img, 1, &out, err, sizeof err);
        CHECK(refused && strstr(err, "no vision tower"), "a tower-less engine did not refuse the image by name: %s",
              err);
        pulsar_tokens_free(&out);
        w->vision.present = true;
        w->vision.shape = *vs;
        w->vision.image_id = 18;
        w->vision.start_id = 19;
        w->vision.end_id = 20;
        e.vision_ready = true;
        err[0] = 0;
        const bool ok = pulsar_image_expand(&e, &prompt, 0, &img, 1, &out, err, sizeof err);
        int run = 0;
        for (int i = 0; i < out.len; i++) run += out.v[i] == 18;
        int len = 0;
        CHECK(ok && img.start_pos == 1 && run == (int)first_tok && out.len == 2 + run &&
                  pulsar_image_block_extent(&e, out.v, out.len, 1, &len) && len == run,
              "the expansion: %s, start %d, %d <image> ids (the processor's %u), block %d", ok ? "ok" : err,
              img.start_pos, run, first_tok, len);
        CHECK(!strcmp(PULSAR_NEMOTRON_H_IMAGE_FRONT.placeholder_text, "<img><image></img>"),
              "the placeholder literal is %s", PULSAR_NEMOTRON_H_IMAGE_FRONT.placeholder_text);
        printf("F image front: refused without a tower (\"%s\"); %d-token block with it\n", "no vision tower bound",
               run);
        pulsar_tokens_free(&prompt);
        pulsar_tokens_free(&out);
        delete w;
    }
    printf("NEMOTRON VISION PIXEL GATE: %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

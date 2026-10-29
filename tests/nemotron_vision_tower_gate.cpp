/* L331 nemotron-vision-tower-gate: Nemotron 3.5 Super VL's C-RADIO tower + projector on the device against the
 * checkpoint's own code run in FP32 on the CPU (tests/nemotron_vision_goldens.py's fixture).
 *
 * Opens the engine on a container that holds the tower (pulsar.vision_present), and for each fixture image runs the
 * engine's whole chain on the same encoded bytes -- the preprocessing nemotron-vision-pixel-gate pins to the
 * processor, the shared ViT encoder (bf16 rows, the bf16 FMHA), vision_final_layernorm, the pixel shuffle, mlp1 -- and
 * grades the 4096-wide rows the language model receives.
 *
 * PASS RULE (L268's, vision-qwen-tower-gate's): an image passes when its relative RMS against the FP32 reference is
 * within FLOOR_SLACK x the reference's OWN bf16-vs-fp32 gap (the checkpoint's code run in bf16, recorded per image in
 * the fixture) -- self-calibrating, no hand-picked number.  Reported beside it: max |diff|, the cosine, and for case 0
 * the distance to the fixture's no-final-norm rows (the arm the checkpoint's code does not take: it must be far, the
 * two arms are 25% apart) and the tower's residual after the last block (vit_out, f32: rel RMS, informational).
 *
 * MODEL-DEPENDENT, GPU.  usage: ./tests/nemotron_vision_tower_gate MODEL GOLDENS */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "families/nemotron_h/nemotron_h.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#define FLOOR_SLACK 3.0   /* vision-qwen-tower-gate's line, for the same reasons */

static FILE *g_f;
static bool rd(void *dst, size_t n) { return !n || fread(dst, 1, n, g_f) == n; }
static uint32_t u32() { uint32_t v = 0; if (!rd(&v, 4)) { fprintf(stderr, "tower gate: short read\n"); exit(2); } return v; }
static void skip(size_t n) { if (fseek(g_f, (long)n, SEEK_CUR) != 0) { fprintf(stderr, "tower gate: short file\n"); exit(2); } }
static float bf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

struct grade { double rel, maxabs, cos; };
static grade compare(const float *a, const float *b, size_t n) {
    double se = 0, sr = 0, sa = 0, dot = 0, mx = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double)a[i] - b[i];
        se += d * d;
        sr += (double)b[i] * b[i];
        sa += (double)a[i] * a[i];
        dot += (double)a[i] * b[i];
        if (fabs(d) > mx) mx = fabs(d);
    }
    return {sr > 0 ? sqrt(se / sr) : sqrt(se), mx, sa > 0 && sr > 0 ? dot / sqrt(sa * sr) : 0.0};
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL GOLDENS\n", argv[0]); return 2; }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0 || !e) { fprintf(stderr, "tower gate: %s did not open\n", argv[1]); return 2; }
    const pulsar_nemotron_h_model *w = pulsar_nemotron_h_of(e);
    if (!w || !w->vision.present || !pulsar_engine_has_vision(e)) {
        fprintf(stderr, "tower gate: %s is not a Nemotron-H container with the vision tower\n", argv[1]);
        pulsar_engine_close(e);
        return 2;
    }
    g_f = fopen(argv[2], "rb");
    if (!g_f) { fprintf(stderr, "tower gate: cannot open %s\n", argv[2]); return 2; }
    char magic[4];
    if (!rd(magic, 4) || memcmp(magic, "NVG1", 4)) { fprintf(stderr, "tower gate: bad magic\n"); return 2; }
    skip((size_t)u32() * 20 + 4);                       /* SIZES (and the budget) */
    for (uint32_t c = 0, n = u32(); c < n; c++) {     /* PIXELS */
        skip(u32());
        const uint32_t H = u32(), W = u32();
        (void)u32(); (void)u32();
        if (u32()) skip((size_t)3 * H * W * 2);
        skip((size_t)u32() * 4);
    }
    {   /* POS */
        const uint32_t C = u32(), n = u32();
        for (uint32_t t = 0; t < n; t++) {
            const uint32_t wp = u32(), hp = u32();
            skip((size_t)wp * hp * C * 4);
        }
    }
    const uint32_t n_cases = u32(), vit_rows = u32();
    int failures = 0;
    for (uint32_t c = 0; c < n_cases; c++) {
        std::vector<uint8_t> png(u32());
        if (!rd(png.data(), png.size())) return 2;
        const uint32_t H = u32(), W = u32(), ntok = u32(), n_arms = u32();
        uint16_t *rows = NULL;
        int n_rows = 0;
        std::vector<float> vit;
        char err[256] = "";
        const bool ok = pulsar_nemotron_h_vision_encode(&e->model, &w->vision, png.data(), png.size(), &rows, &n_rows,
                                                        c == 0 ? &vit : NULL, err, sizeof err);
        if (!ok) printf("  FAIL case %u: the encode refused: %s\n", c, err);
        failures += !ok;
        std::vector<float> got;
        if (ok) {
            got.resize((size_t)n_rows * 4096);
            for (size_t i = 0; i < got.size(); i++) got[i] = bf(rows[i]);
        }
        free(rows);
        for (uint32_t a = 0; a < n_arms; a++) {
            const uint32_t final_norm = u32();
            float floor_rel = 0, store = 0;
            if (!rd(&floor_rel, 4) || !rd(&store, 4)) return 2;
            std::vector<uint16_t> ref16((size_t)ntok * 4096);
            if (!rd(ref16.data(), ref16.size() * 2)) return 2;
            if (!ok) continue;
            if ((uint32_t)n_rows != ntok) {
                printf("  FAIL case %u: %d rows, the reference's %u (%ux%u patches)\n", c, n_rows, ntok, W, H);
                failures++;
                continue;
            }
            std::vector<float> ref(ref16.size());
            for (size_t i = 0; i < ref.size(); i++) ref[i] = bf(ref16[i]);
            const grade g = compare(got.data(), ref.data(), ref.size());
            if (final_norm) {
                const double limit = FLOOR_SLACK * floor_rel;
                const bool pass = g.rel <= limit;
                printf("  %s case %u (%ux%u patches, %u rows): rel_rms %.3e max_abs %.3e cos %.6f vs the FP32 "
                       "reference (its bf16 floor %.3e, limit %.3e; fixture storage %.1e)\n",
                       pass ? "PASS" : "FAIL", c, W, H, ntok, g.rel, g.maxabs, g.cos, floor_rel, limit, store);
                failures += !pass;
            } else {
                printf("  info case %u: rel_rms %.3e to the fixture's NO-final-norm rows (the arm the checkpoint's code "
                       "does not take; its own floor %.3e)\n", c, g.rel, floor_rel);
            }
        }
        if (c == 0) {
            const uint32_t r = u32(), cols = u32();
            std::vector<float> ref((size_t)r * cols);
            if (!rd(ref.data(), ref.size() * 4)) return 2;
            if (ok && vit.size() >= ref.size() && cols == 1280 && r == vit_rows) {
                const grade g = compare(vit.data(), ref.data(), ref.size());
                printf("  info case 0 vit_out (the first %u rows: 10 prefix + patches, f32): rel_rms %.3e max_abs %.3e\n",
                       r, g.rel, g.maxabs);
            }
        }
    }
    fclose(g_f);
    pulsar_engine_close(e);
    printf("NEMOTRON VISION TOWER GATE: %s (%u images, %d failures; within %.1fx the reference's own bf16-vs-fp32 "
           "floor)\n", failures ? "FAIL" : "PASS", n_cases, failures, FLOOR_SLACK);
    return failures ? 1 : 0;
}

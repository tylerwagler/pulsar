/* L268 vision-qwen-tower gate: Qwen3.8-Flash-Next's tower + merger vs HF's own Qwen4ExpVisionModel.
 *
 * Goldens come from tests/vision_qwen_tower_goldens.py: each case's encoded image, and per stage the reference's
 * bf16 rows plus the reference's OWN bf16-vs-fp32 relative RMS.  This gate opens the artifact's mapping (no engine:
 * the tower is the only thing graded), binds `model.visual.*`, and runs the engine's whole image chain on the same
 * bytes -- the preprocessing vision-qwen-pixel-gate pins to HF bit for bit, then the tower and merger
 * (pulsar_cuda_vision.cu, the kernels DeepSeek's tower shares).
 *
 * TOLERANCE, as vision-tower-gate grades DeepSeek's tower and for the same reason: the reference attention goes
 * through scaled_dot_product_attention, whose reduction order no hand-written attention reproduces.  A stage passes
 * when its relative RMS against the reference's bf16 is within TOWER_FLOOR_SLACK x the reference's own
 * bf16-vs-fp32 gap -- self-calibrating, no hand-picked number.
 *
 * MODEL-DEPENDENT, GPU.  usage: ./tests/vision_qwen_tower_gate MODEL GOLDENS */
#include "pulsar.h"
#include "pulsar_engine_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#define TOWER_FLOOR_SLACK 3.0   /* the same line as vision-tower-gate's, for the same reasons */

static float bf16_to_f32(uint16_t bits) {
    uint32_t u = (uint32_t)bits << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static void rd(void *dst, size_t n, FILE *f) {
    if (n && fread(dst, 1, n, f) != n) { fprintf(stderr, "qwen tower gate: short read\n"); exit(2); }
}
static uint32_t rd_u32(FILE *f) { uint32_t v; rd(&v, 4, f); return v; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL GOLDENS\n", argv[0]); return 2; }
    pulsar_model m;
    if (!model_open(&m, argv[1], true)) { fprintf(stderr, "qwen tower gate: %s did not open\n", argv[1]); return 2; }
    pulsar_qwen_vision_weights_dev w;
    memset(&w, 0, sizeof w);
    bool present = false;
    if (!qwen_vision_bind(&m, &w, &present) || !present) {
        fprintf(stderr, "qwen tower gate: %s carries no bindable Qwen vision tower\n", argv[1]);
        model_close(&m);
        return 2;
    }
    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "qwen tower gate: cannot open %s\n", argv[2]); return 2; }
    char magic[4];
    rd(magic, 4, f);
    if (memcmp(magic, "QVT1", 4)) { fprintf(stderr, "qwen tower gate: bad magic\n"); return 2; }
    const uint32_t n_cases = rd_u32(f);
    int failures = 0;
    for (uint32_t c = 0; c < n_cases; c++) {
        std::vector<uint8_t> enc(rd_u32(f));
        rd(enc.data(), enc.size(), f);
        const uint32_t gh = rd_u32(f), gw = rd_u32(f), n_stages = rd_u32(f);
        uint16_t *rows = NULL, *dbg = NULL;
        int n_rows = 0, n_patches = 0;
        char err[256] = "";
        const bool ok = qwen_vision_encode(&w, enc.data(), enc.size(), &rows, &n_rows, &dbg, &n_patches, err,
                                           sizeof err);
        if (!ok) printf("  FAIL case %u: the encode refused: %s\n", c, err);
        failures += !ok;
        for (uint32_t s = 0; s < n_stages; s++) {
            char name[9] = {0};
            rd(name, 8, f);
            const uint32_t r = rd_u32(f), cols = rd_u32(f);
            float floor_rel;
            rd(&floor_rel, 4, f);
            std::vector<uint16_t> ref((size_t)r * cols);
            rd(ref.data(), ref.size() * 2, f);
            if (!ok) continue;
            const uint16_t *got = NULL;
            if (!strcmp(name, "merger")) got = (int)r == n_rows ? rows : NULL;
            else if ((int)r == n_patches) {
                const size_t stride = (size_t)n_patches * PULSAR_QWEN_VISION_DIM;
                got = !strcmp(name, "input") ? dbg : !strcmp(name, "block0") ? dbg + stride
                    : !strcmp(name, "blockN") ? dbg + 2 * stride : NULL;
            }
            if (!got) {
                printf("  FAIL case %u stage %s: %u x %u rows, the engine made %d patches / %d rows\n", c, name, r,
                       cols, n_patches, n_rows);
                failures++;
                continue;
            }
            double se = 0, sr = 0, maxabs = 0;
            for (size_t i = 0; i < ref.size(); i++) {
                const double a = bf16_to_f32(got[i]), b = bf16_to_f32(ref[i]);
                se += (a - b) * (a - b);
                sr += b * b;
                if (fabs(a - b) > maxabs) maxabs = fabs(a - b);
            }
            const double rel = sr > 0 ? sqrt(se / sr) : sqrt(se);
            const double limit = (double)floor_rel * TOWER_FLOOR_SLACK;
            const bool pass = rel <= limit;
            printf("  %s case %u (grid %ux%u) %-7s rel_rms %.3e max_abs %.3e (bf16 floor %.3e, limit %.3e)\n",
                   pass ? "PASS" : "FAIL", c, gh, gw, name, rel, maxabs, (double)floor_rel, limit);
            failures += !pass;
        }
        free(rows);
        free(dbg);
    }
    fclose(f);
    model_close(&m);
    printf("VISION QWEN TOWER GATE: %s (%u cases, %d failures; within %.2fx the reference's own bf16-vs-fp32 floor)\n",
           failures ? "FAIL" : "PASS", n_cases, failures, (double)TOWER_FLOOR_SLACK);
    return failures ? 1 : 0;
}

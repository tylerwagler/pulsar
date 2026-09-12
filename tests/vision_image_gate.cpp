/* L216 gate: the ENGINE's whole per-image lane, end to end, vs the reference.
 *
 * Input is the encoded image FILE; output is the block that lands in the HC
 * carrier.  Everything in between is the engine: its libpng/libjpeg-turbo
 * decode, its Pillow-faithful preprocess, span discovery from the prompt's own
 * sentinel ids, the ViT + aligner, and the scatter into the carrier.
 *
 * The other fixtures each carry half of this.  vision-span-goldens has the bytes
 * and the layout but not the embeddings; vision-merge-goldens has the patches and
 * the embeddings but not the bytes.  Neither can drive gpu_graph_merge_image_spans(),
 * whose input is bytes -- so neither would have noticed a decode or preprocess
 * regression on the lane as the model actually consumes it.  This is that
 * instrument, and it is the one the previous increment owed.
 *
 * It also pins PLACEMENT with a real model: the carrier is pre-filled with a
 * pattern and every row outside the span must survive untouched, and every HC
 * stream of a span row must carry the same merged vector (the reference expands
 * to hc_mult copies only after the merge).
 *
 * TOLERANCE, floor-calibrated: the lane contains the tower forward, whose
 * attention cannot reproduce F.scaled_dot_product_attention's reduction order.
 * Each case carries the reference's own bf16-vs-fp32 gap and the gate requires
 * the same 3x that vision-tower-gate and vision-merge-gate use.
 *
 * MODEL-DEPENDENT (needs a Vision-Exp artifact) and GPU-resident.
 *
 * usage: ./tests/vision_image_gate MODEL GOLDENS
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "pulsar_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define IMAGE_FLOOR_SLACK 3.0

static float bf16_to_f32(uint16_t bits) {
    uint32_t u = (uint32_t)bits << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}
static uint32_t rd_u32(FILE *f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) { fprintf(stderr, "image gate: short read\n"); exit(2); }
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}
static int rd_i32(FILE *f) { return (int)rd_u32(f); }
static float rd_f32(FILE *f) {
    uint32_t u = rd_u32(f);
    float v; memcpy(&v, &u, sizeof v);
    return v;
}
static void rd(void *dst, size_t n, FILE *f) {
    if (n && fread(dst, 1, n, f) != n) { fprintf(stderr, "image gate: short read\n"); exit(2); }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL GOLDENS\n", argv[0]); return 2; }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1];
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "image gate: engine open failed\n"); return 2; }
    if (!e->vision_ready) { fprintf(stderr, "image gate: no vision tower in this artifact\n"); return 2; }
    /* The carrier belongs to a session's graph, which is the tensor the prefill
     * actually seeds -- using the real one is the point. */
    pulsar_session *sess = NULL;
    if (pulsar_session_create(&sess, e, 4096) != 0) {
        fprintf(stderr, "image gate: session create failed\n");
        return 2;
    }

    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "image gate: cannot open %s\n", argv[2]); return 2; }
    char magic[4];
    rd(magic, 4, f);
    if (memcmp(magic, "VIG1", 4)) { fprintf(stderr, "image gate: bad magic\n"); return 2; }
    const uint32_t n_cases = rd_u32(f);
    const uint32_t n_embd = PULSAR_N_EMBD, n_hc = PULSAR_N_HC;
    const size_t per_row = (size_t)n_hc * n_embd * PULSAR_HC_ELT_SIZE;
    int failures = 0;

    for (uint32_t c = 0; c < n_cases; c++) {
        const int patch = rd_i32(f), downsample = rd_i32(f);
        const int max_n_token = rd_i32(f), min_pixels = rd_i32(f);
        const float max_wh_ratio = rd_f32(f);
        const int vocab = rd_i32(f), start_pos = rd_i32(f);
        const int n_vh = rd_i32(f), n_vw = rd_i32(f);
        const int span_len = rd_i32(f);
        const double floor_rel = (double)rd_f32(f);
        const int enc_len = rd_i32(f);
        std::vector<uint8_t> enc((size_t)enc_len);
        std::vector<int32_t> types((size_t)span_len);
        std::vector<uint16_t> want((size_t)span_len * n_embd);
        rd(enc.data(), enc.size(), f);
        rd(types.data(), types.size() * 4, f);
        rd(want.data(), want.size() * 2, f);

        /* The prompt carries the block at start_pos as `vocab + role`, exactly how
         * a renderer emits it, with ordinary text ids on either side.
         *
         * NOTE which position the image NAMES.  build_image_block's types are in
         * final order and the block does not begin with IMAGE_START -- the
         * reference prepends a compressor pad, so the START sentinel lands
         * `start_off` slots in.  pulsar_image_ref names the SENTINEL (that is what
         * vision_span_extent looks for), while the golden's start_pos names the
         * BLOCK (that is what merge_image_embeddings writes at). */
        int start_off = -1;
        for (int i = 0; i < span_len; i++) if (types[i] == 0) { start_off = i; break; }
        if (start_off < 0) {
            printf("  FAIL case %u: the fixture block carries no IMAGE_START\n", c);
            failures++;
            continue;
        }
        const int n_tokens = start_pos + span_len + 3;
        std::vector<int32_t> ids((size_t)n_tokens, 100);
        for (int i = 0; i < span_len; i++) ids[(size_t)start_pos + i] = vocab + types[i];

        const uint64_t carrier_bytes = pulsar_gpu_tensor_bytes(sess->graph.batch_cur_hc);
        if ((uint64_t)n_tokens * per_row > carrier_bytes) {
            fprintf(stderr, "  FAIL case %u: prompt %d exceeds the carrier (%llu bytes)\n",
                    c, n_tokens, (unsigned long long)carrier_bytes);
            failures++;
            continue;
        }

        /* A pattern no merged row can be mistaken for, so "outside the span was
         * left alone" is answerable. */
        std::vector<uint16_t> fill((size_t)n_tokens * n_hc * n_embd, 0xA5A5u);
        std::vector<uint16_t> back(fill.size(), 0);
        pulsar_image_ref img = { enc.data(), enc.size(), start_pos + start_off };
        pulsar_vision_request vreq = { &img, 1, &e->vision_weights };

        int bad = 0;
        if (!pulsar_gpu_tensor_write(sess->graph.batch_cur_hc, 0, fill.data(),
                                     (uint64_t)fill.size() * 2))
            { printf("FAIL case %u: carrier write failed\n", c); failures++; continue; }

        if (!gpu_graph_merge_image_spans(sess->graph.batch_cur_hc, &e->model, ids.data(), n_tokens,
                                         &vreq, 0, (uint32_t)n_tokens)) {
            printf("  FAIL case %u (%dx%d start=%d): merge refused\n", c, n_vh, n_vw, start_pos);
            failures++;
            continue;
        }
        if (!pulsar_gpu_end_commands() ||
            !pulsar_gpu_tensor_read(sess->graph.batch_cur_hc, 0, back.data(),
                                    (uint64_t)fill.size() * 2)) {
            printf("  FAIL case %u: carrier read failed\n", c);
            failures++;
            continue;
        }

        /* every HC stream of every span row equals the reference's block */
        double sumsq = 0.0, refsq = 0.0, maxabs = 0.0;
        for (int r = 0; r < span_len && !bad; r++) {
            for (uint32_t h = 0; h < n_hc && !bad; h++) {
                for (uint32_t d = 0; d < n_embd; d++) {
                    const size_t at = ((size_t)(start_pos + r) * n_hc + h) * n_embd + d;
                    const double got = bf16_to_f32(back[at]);
                    const double ref = bf16_to_f32(want[(size_t)r * n_embd + d]);
                    if (got != ref && h == 0) {
                        const double dd = got - ref;
                        sumsq += dd * dd; refsq += ref * ref;
                        if (fabs(dd) > maxabs) maxabs = fabs(dd);
                    } else if (h != 0 && back[at] != back[((size_t)(start_pos + r) * n_hc) * n_embd + d]) {
                        printf("  FAIL case %u: row %d HC stream %u differs from stream 0\n",
                               c, start_pos + r, h);
                        bad++;
                    }
                }
            }
        }
        /* and nothing outside the span moved */
        for (int t = 0; t < n_tokens && !bad; t++) {
            if (t >= start_pos && t < start_pos + span_len) continue;
            if (back[(size_t)t * n_hc * n_embd] != 0xA5A5u ||
                back[((size_t)t + 1) * n_hc * n_embd - 1] != 0xA5A5u) {
                printf("  FAIL case %u: row %d outside the span was rewritten\n", c, t);
                bad++;
            }
        }
        const double rel = refsq > 0.0 ? sqrt(sumsq / refsq) : sqrt(sumsq);
        const double limit = floor_rel * IMAGE_FLOOR_SLACK;
        const int pass = !bad && rel <= limit;
        printf("  %s case %u: %dx%d start=%-3d vit %dx%d span %-4d rel-RMS %.3e max-abs %.3e "
               "(bf16 floor %.3e, limit %.3e)\n",
               pass ? "PASS" : "FAIL", c, n_vh, n_vw, start_pos, n_vh, n_vw, span_len,
               rel, maxabs, floor_rel, limit);
        if (!pass) failures++;

        (void)patch; (void)downsample; (void)max_n_token; (void)min_pixels; (void)max_wh_ratio;
    }

    /* ---- an image whose span the prompt does not carry is refused ---------- */
    {
        std::vector<uint8_t> enc(1024, 0);          /* contents do not matter: it must refuse first */
        std::vector<int32_t> ids(32, 100);          /* text only: no sentinels at all */
        pulsar_image_ref img = { enc.data(), enc.size(), 4 };
        pulsar_vision_request vreq = { &img, 1, &e->vision_weights };
        const bool got = gpu_graph_merge_image_spans(sess->graph.batch_cur_hc, &e->model,
                                                     ids.data(), (int)ids.size(), &vreq, 0, 32);
        if (got) {
            printf("  FAIL an image with no sentinel span in the prompt was accepted\n");
            failures++;
        } else {
            printf("  PASS an image with no sentinel span in the prompt is refused\n");
        }
    }

    fclose(f);
    pulsar_session_free(sess);
    printf("VISION IMAGE GATE: %s (%u cases, %d failures; within %.2fx the reference's own bf16 floor)\n",
           failures ? "FAIL" : "PASS", n_cases, failures, (double)IMAGE_FLOOR_SLACK);
    return failures ? 1 : 0;
}

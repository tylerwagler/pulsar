/* L216 gate: an image request through the PUBLIC API, end to end.
 *
 * Everything else in the vision suite grades a piece: the tower forward, the
 * span layout, the merge, the placement, the producer, the router's image-slot
 * bias, the visibility counts.  This is the only gate that calls
 * pulsar_session_sync_mm() and makes the ENGINE run the whole thing -- decode,
 * preprocess, span discovery, tower, merge, span-atomic prefill, 43 layers,
 * output head -- on a real Vision-Exp artifact.
 *
 * What it asserts:
 *   - a well-formed image request SYNCS (returns 0) and leaves usable logits;
 *   - a prompt carrying sentinel ids with NO image is REFUSED.  Without that
 *     check the engine would happily prefill rows whose embeddings never
 *     arrived: the embedder zero-masks an out-of-vocab id and only the merge
 *     puts anything there, so the failure would be a fluent wrong answer;
 *   - an image whose sentinel block is not in the prompt is REFUSED;
 *   - a span that would not fit one prefill chunk is REFUSED;
 *   - (L273) an image turn on a bank a ghost rewind left STALE resumes from its
 *     grid checkpoint, and its logits are bit-identical to the cold prefill
 *     with the same chunk boundary.
 *
 * MODEL-DEPENDENT (needs a Vision-Exp artifact) and GPU-resident.
 *
 * usage: ./tests/vision_image_sync_gate MODEL GOLDENS
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "pulsar_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static uint32_t rd_u32(FILE *f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) { fprintf(stderr, "sync gate: short read\n"); exit(2); }
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}
static int rd_i32(FILE *f) { return (int)rd_u32(f); }
static float rd_f32(FILE *f) { uint32_t u = rd_u32(f); float v; memcpy(&v, &u, sizeof v); return v; }
static void rd(void *dst, size_t n, FILE *f) {
    if (n && fread(dst, 1, n, f) != n) { fprintf(stderr, "sync gate: short read\n"); exit(2); }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL GOLDENS\n", argv[0]); return 2; }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1];
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "sync gate: engine open failed\n"); return 2; }
    if (!e->vision_ready) { fprintf(stderr, "sync gate: no vision tower in this artifact\n"); return 2; }
    pulsar_session *sess = NULL;
    if (pulsar_session_create(&sess, e, 4096) != 0) { fprintf(stderr, "sync gate: session failed\n"); return 2; }

    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "sync gate: cannot open %s\n", argv[2]); return 2; }
    char magic[4];
    rd(magic, 4, f);
    if (memcmp(magic, "VIG1", 4)) { fprintf(stderr, "sync gate: bad magic\n"); return 2; }
    const uint32_t n_cases = rd_u32(f);
    int failures = 0;

    std::vector<float> logits((size_t)PULSAR_N_VOCAB);

    for (uint32_t c = 0; c < n_cases; c++) {
        const int patch = rd_i32(f), downsample = rd_i32(f);
        const int max_n_token = rd_i32(f), min_pixels = rd_i32(f);
        const float max_wh_ratio = rd_f32(f);
        const int vocab = rd_i32(f), start_pos = rd_i32(f);
        const int n_vh = rd_i32(f), n_vw = rd_i32(f);
        const int span_len = rd_i32(f);
        (void)rd_f32(f);
        const int enc_len = rd_i32(f);
        std::vector<uint8_t> enc((size_t)enc_len);
        std::vector<int32_t> types((size_t)span_len);
        rd(enc.data(), enc.size(), f);
        rd(types.data(), types.size() * 4, f);
        /* The pad is the golden's own width, NOT this binary's default profile:
         * PULSAR_N_EMBD is 5120 in a V4.1-default build and the file was
         * written at the reference's 4096, which desyncs the reader. */
        std::vector<uint16_t> skip((size_t)span_len * PULSAR_VISION_GOLDEN_N_EMBD);
        rd(skip.data(), skip.size() * 2, f);
        (void)patch; (void)downsample; (void)max_n_token; (void)min_pixels; (void)max_wh_ratio;

        /* The renderer's output: text, the image block, a little more text. */
        pulsar_tokens prompt = {};
        for (int i = 0; i < start_pos; i++) pulsar_tokens_push(&prompt, 100 + i % 7);
        for (int i = 0; i < span_len; i++) pulsar_tokens_push(&prompt, vocab + types[i]);
        for (int i = 0; i < 3; i++) pulsar_tokens_push(&prompt, 200 + i);

        pulsar_image_ref img = { enc.data(), enc.size(), start_pos };
        char err[512] = {0};
        int bad = 0;

        /* ---- a well-formed request syncs ---------------------------------- */
        if (pulsar_session_sync_mm(sess, &prompt, &img, 1, err, sizeof err) != 0) {
            printf("  FAIL case %u (%dx%d): sync_mm refused a valid image request: %s\n",
                   c, n_vh, n_vw, err);
            bad++;
        } else {
            const int am = pulsar_session_argmax(sess);
            if (am < 0 || am >= (int)PULSAR_N_VOCAB) {
                printf("  FAIL case %u: argmax %d out of range after an image prefill\n", c, am);
                bad++;
            }
            /* copy_logits returns the COUNT it copied, not 0 on success. */
            if (pulsar_session_copy_logits(sess, logits.data(), (int)PULSAR_N_VOCAB) !=
                (int)PULSAR_N_VOCAB) {
                printf("  FAIL case %u: logits unreadable after an image prefill\n", c);
                bad++;
            } else {
                int nonfinite = 0;
                for (size_t i = 0; i < logits.size(); i++) if (!isfinite(logits[i])) nonfinite++;
                if (nonfinite) {
                    printf("  FAIL case %u: %d non-finite logits after an image prefill\n", c, nonfinite);
                    bad++;
                }
            }
        }

        /* ---- the same prompt with no image is refused --------------------- */
        {
            char err2[512] = {0};
            pulsar_tokens p2 = {};
            for (int i = 0; i < prompt.len; i++) pulsar_tokens_push(&p2, prompt.v[i]);
            if (pulsar_session_sync(sess, &p2, err2, sizeof err2) == 0) {
                printf("  FAIL case %u: a prompt with sentinels and no images was accepted\n", c);
                bad++;
            } else {
                printf("  ok   case %u: sentinels with no image refused (%s)\n", c, err2);
            }
            pulsar_tokens_free(&p2);
        }

        /* ---- an image whose block is not in the prompt is refused --------- */
        {
            char err3[512] = {0};
            pulsar_tokens p3 = {};
            for (int i = 0; i < 8; i++) pulsar_tokens_push(&p3, 100 + i);
            for (int i = 0; i < 3; i++) pulsar_tokens_push(&p3, 200 + i);
            /* no sentinels at all, so the block start is a plain text token */
            pulsar_image_ref img3 = { enc.data(), enc.size(), 4 };
            if (pulsar_session_sync_mm(sess, &p3, &img3, 1, err3, sizeof err3) == 0) {
                printf("  FAIL case %u: an image with no block in the prompt was accepted\n", c);
                bad++;
            } else {
                printf("  ok   case %u: image with no block refused (%s)\n", c, err3);
            }
            pulsar_tokens_free(&p3);
        }

        /* ---- L273: an image turn on a STALE bank (a reply's ghost rewind leaves it mid-group) resumes from its
         *      grid checkpoint, byte for byte the cold prefill of the same prompt (before fab03497: rebuilt cold) */
        {
            char err4[512] = {0};
            const char *step = "text sync";
            pulsar_tokens text = {}, turn = {};
            for (int i = 0; i < 300; i++) pulsar_tokens_push(&text, 100 + i % 7);
            pulsar_session_invalidate(sess);
            bool ok4 = pulsar_session_sync(sess, &text, err4, sizeof err4) == 0;
            if (ok4) step = "decode";
            for (int k = 0; ok4 && k < 3; k++)
                ok4 = pulsar_session_eval(sess, pulsar_session_argmax(sess), err4, sizeof err4) == 0;
            if (ok4) {
                pulsar_session_rewind(sess, pulsar_session_pos(sess) - 1);   /* the ghost token */
                /* the installed bank's flag (pulsar_session_bank_comp_stale answers for a bank POOL; this session
                 * is classic, one bank) */
                step = "stale after the ghost rewind";
                ok4 = sess->graph->ms_comp_state_stale[gpu_graph_cur_bank(sess->graph)];
            }
            /* the turn the server builds: the live history, the renderer's placeholder, a little text -- expanded from
             * the live length (a block's layout depends on where it starts, so the golden's block is not reused) */
            pulsar_image_ref img4 = { enc.data(), enc.size(), 0 };
            if (ok4) {
                step = "expanding the turn's placeholder";
                const pulsar_tokens *live = pulsar_session_tokens(sess);
                const int live_len = live->len;
                pulsar_tokens raw = {}, ph = {};
                for (int i = 0; i < live_len; i++) pulsar_tokens_push(&raw, live->v[i]);
                pulsar_tokenize_rendered_chat(e, PULSAR_IMAGE_PLACEHOLDER, &ph);
                for (int i = 0; i < ph.len; i++) pulsar_tokens_push(&raw, ph.v[i]);
                for (int i = 0; i < 3; i++) pulsar_tokens_push(&raw, 200 + i);
                ok4 = ph.len == 1 && pulsar_expand_image_placeholders(e, &raw, live_len, &img4, 1, &turn, err4,
                                                                      sizeof err4) == 1 && img4.start_pos == live_len;
                pulsar_tokens_free(&raw);
                pulsar_tokens_free(&ph);
            }
            if (ok4) {
                step = "image turn on the stale bank";
                ok4 = pulsar_session_sync_mm(sess, &turn, &img4, 1, err4, sizeof err4) == 0;
            }
            const int origin = ok4 ? pulsar_session_resume_origin(sess) : -1;
            std::vector<float> resumed((size_t)PULSAR_N_VOCAB);
            if (ok4) {
                step = "the cold prefill of the same prompt";
                ok4 = pulsar_session_copy_logits(sess, resumed.data(), (int)PULSAR_N_VOCAB) == (int)PULSAR_N_VOCAB;
                /* the cold prefill with the resume's chunk boundary: [0, origin) as text, then the rest in one
                 * chunk.  NOT the one-chunk prefill: an image block's rows depend on where its chunk starts (L282,
                 * found here 2026-10-07: relL2 ~0.47 between the two, text chunk-invariant) -- that is graded
                 * against the reference there, not here */
                pulsar_session_invalidate(sess);
                pulsar_tokens head = {};
                for (int i = 0; i < origin && i < turn.len; i++) pulsar_tokens_push(&head, turn.v[i]);
                ok4 = ok4 && origin > 0 && pulsar_session_sync(sess, &head, err4, sizeof err4) == 0 &&
                      pulsar_session_sync_mm(sess, &turn, &img4, 1, err4, sizeof err4) == 0 &&
                      pulsar_session_copy_logits(sess, logits.data(), (int)PULSAR_N_VOCAB) == (int)PULSAR_N_VOCAB;
                pulsar_tokens_free(&head);
            }
            int differ = 0;
            for (size_t i = 0; ok4 && i < logits.size(); i++) differ += memcmp(&resumed[i], &logits[i], 4) != 0;
            if (!ok4) {
                printf("  FAIL case %u: stale-bank image turn: %s failed %s\n", c, step, err4);
                bad++;
            } else if (origin <= 0 || origin > text.len || differ) {
                printf("  FAIL case %u: stale-bank image turn resumed at %d (want a grid checkpoint in (0,%d]), "
                       "%d/%u logits differ from cold\n", c, origin, text.len, differ, (unsigned)PULSAR_N_VOCAB);
                bad++;
            } else {
                printf("  ok   case %u: stale-bank image turn resumed from grid checkpoint %d, logits == cold\n", c,
                       origin);
            }
            pulsar_session_invalidate(sess);
            pulsar_tokens_free(&text);
            pulsar_tokens_free(&turn);
        }

        printf("%s case %u: %dx%d start=%-3d span %-4d\n", bad ? "FAIL" : "ok  ", c, n_vh, n_vw,
               start_pos, span_len);
        failures += bad;
        pulsar_tokens_free(&prompt);
    }

    fclose(f);
    pulsar_session_free(sess);
    pulsar_engine_close(e);
    printf("VISION IMAGE SYNC GATE: %s (%u cases, %d failures)\n",
           failures ? "FAIL" : "PASS", n_cases, failures);
    return failures ? 1 : 0;
}

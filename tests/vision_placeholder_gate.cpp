/* L216 gate: the sentinel-id PRODUCER -- placeholder in, sentinel block out.
 *
 * The engine's whole image path needs someone to put `vocab_size + role` ids into
 * a prompt; the renderer will, via vision_expand_image_placeholders(), which is
 * the reference's prepare_vl_inputs(): walk the tokenized prompt and replace each
 * IMAGE_PLACEHOLDER token with that image's block.
 *
 * This grades that producer against the LAYOUT math rather than against a new
 * golden, because the two are already independently graded: build_image_block is
 * pinned by vision-layout-gate/vision-span-gate and the reader side by
 * vision-visible-gate.  What is new here is only that the ids are WRITTEN, and
 * the claims worth pinning are that:
 *
 *   - the ids written are `vocab_size + <the block's own types>`, in order;
 *   - the reported block start is the prompt length at that moment (not the
 *     IMAGE_START slot, which lands a compressor pad later -- confusing the two
 *     is a real bug that a tolerance-based grade let through once);
 *   - the span READER recovers exactly the block the writer wrote, and
 *     vision_image_visible() sees its spans -- a round trip, so a writer and
 *     reader that drifted apart cannot both pass;
 *   - the text around the placeholder survives, in order;
 *   - a placeholder/image count mismatch is refused, not silently accepted.
 *
 * The images come from vision-image-goldens (encoded bytes + the block's types),
 * so this needs NO model and NO GPU and runs in the battery.
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint32_t rd_u32(FILE *f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) { fprintf(stderr, "placeholder gate: short read\n"); exit(2); }
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}
static int rd_i32(FILE *f) { return (int)rd_u32(f); }
static float rd_f32(FILE *f) { uint32_t u = rd_u32(f); float v; memcpy(&v, &u, sizeof v); return v; }
static void rd(void *dst, size_t n, FILE *f) {
    if (n && fread(dst, 1, n, f) != n) { fprintf(stderr, "placeholder gate: short read\n"); exit(2); }
}

/* The prompt's stand-in for the tokenizer's IMAGE_PLACEHOLDER.  The producer takes
 * the id as an argument precisely so it does not have to know which token the
 * tokenizer assigned; the engine passes the real one. */
enum { PLACEHOLDER = 129264, TEXT = 100 };

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "tests/test-vectors/vision-image-goldens.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "placeholder gate: cannot open %s\n", path); return 2; }
    char magic[4];
    rd(magic, 4, f);
    if (memcmp(magic, "VIG1", 4)) { fprintf(stderr, "placeholder gate: bad magic\n"); return 2; }
    const uint32_t n_cases = rd_u32(f);
    int failures = 0, checked = 0;

    for (uint32_t c = 0; c < n_cases; c++) {
        const int patch = rd_i32(f), downsample = rd_i32(f);
        const int max_n_token = rd_i32(f), min_pixels = rd_i32(f);
        const float max_wh_ratio = rd_f32(f);
        const int vocab = rd_i32(f), start_pos = rd_i32(f);
        const int n_vh = rd_i32(f), n_vw = rd_i32(f);
        const int span_len = rd_i32(f);
        (void)rd_f32(f);                       /* the bf16 floor: not this gate's business */
        const int enc_len = rd_i32(f);
        std::vector<uint8_t> enc((size_t)enc_len);
        std::vector<int32_t> types((size_t)span_len);
        rd(enc.data(), enc.size(), f);
        rd(types.data(), types.size() * 4, f);
        std::vector<uint16_t> skip((size_t)span_len * PULSAR_N_EMBD);
        rd(skip.data(), skip.size() * 2, f);

        pulsar_vision_args args = { patch, downsample, max_n_token, min_pixels, max_wh_ratio };

        /* The prompt the renderer would hand over: `start_pos` text tokens, the
         * placeholder, then a tail.  Placing it at start_pos makes the block land
         * exactly where the golden's block sits, so the types are comparable. */
        pulsar_tokens in = {};
        for (int i = 0; i < start_pos; i++) pulsar_tokens_push(&in, TEXT + i % 7);
        pulsar_tokens_push(&in, PLACEHOLDER);
        pulsar_tokens_push(&in, TEXT + 1);
        pulsar_tokens_push(&in, TEXT + 2);

        pulsar_tokens out = {};
        pulsar_vision_prepared prep;
        memset(&prep, 0, sizeof prep);
        int start = -1;
        pulsar_image_ref img = { enc.data(), enc.size(), 0 };
        int bad = 0;

        if (!vision_expand_image_placeholders(&out, &in, PLACEHOLDER, &img, 1,
                                              &args, vocab, &prep, &start)) {
            printf("  FAIL case %u: expander refused a valid prompt\n", c);
            bad++;                     /* every later check is guarded on !bad */
        }
        if (!bad && start != start_pos) {
            printf("  FAIL case %u: block start %d, want %d\n", c, start, start_pos);
            bad++;
        }
        if (!bad && out.len != in.len - 1 + span_len) {
            printf("  FAIL case %u: %d tokens out, want %d\n", c, out.len, in.len - 1 + span_len);
            bad++;
        }
        for (int i = 0; i < span_len && !bad; i++) {
            if (out.v[start_pos + i] != vocab + types[i]) {
                printf("  FAIL case %u: id[%d] = %d, want vocab+%d = %d\n", c, start_pos + i,
                       out.v[start_pos + i], types[i], vocab + types[i]);
                bad++;
            }
        }
        /* the text around the placeholder survives, in order */
        for (int i = 0; i < start_pos && !bad; i++)
            if (out.v[i] != in.v[i]) { printf("  FAIL case %u: prefix token %d changed\n", c, i); bad++; }
        for (int i = 0; i < 2 && !bad; i++)
            if (out.v[start_pos + span_len + i] != in.v[start_pos + 1 + i]) {
                printf("  FAIL case %u: tail token %d changed\n", c, i);
                bad++;
            }

        /* ROUND TRIP: the reader must recover exactly the block the writer wrote,
         * and the visibility pass must see it as one span. */
        int found = 0;
        if (!bad && !vision_span_extent(out.v, out.len, vocab, start_pos, &found)) {
            printf("  FAIL case %u: vision_span_extent cannot find the block just written\n", c);
            bad++;
        } else if (!bad && found != span_len) {
            printf("  FAIL case %u: vision_span_extent reports %d rows, wrote %d\n", c, found, span_len);
            bad++;
        }
        if (!bad) {
            std::vector<int32_t> left((size_t)out.len), right((size_t)out.len);
            vision_image_visible(out.v, out.len, vocab, max_n_token, left.data(), right.data());
            /* The visibility span is [IMAGE_START, IMAGE_END], NOT the whole
             * block: the reference's `valid` is false at the leading compressor
             * pads, which sit before the START.  So the pads see nothing, and
             * every slot from the START to the END inclusive sees the whole span
             * -- which is the bidirectional reach the tower needs. */
            int start_off = -1;
            for (int i = 0; i < span_len; i++) if (types[i] == 0) { start_off = i; break; }
            const int vs = start_pos + start_off, ve = start_pos + span_len - 1;
            if (start_off < 0) { printf("  FAIL case %u: block has no IMAGE_START\n", c); bad++; }
            for (int i = start_pos; i < vs && !bad; i++) {
                if (left[i] != 0 || right[i] != 0) {
                    printf("  FAIL case %u: leading pad at %d is marked visible (l%d r%d)\n",
                           c, i, left[i], right[i]);
                    bad++;
                }
            }
            for (int i = vs; i <= ve && !bad; i++) {
                if (left[i] != i - vs || right[i] != ve - i) {
                    printf("  FAIL case %u: slot %d sees left %d right %d, want %d/%d\n",
                           c, i, left[i], right[i], i - vs, ve - i);
                    bad++;
                }
            }
            if (!bad) {
                int vis = 0;
                for (int i = 0; i < out.len; i++) if (left[i] || right[i]) vis++;
                if (vis != ve - vs + 1) {
                    printf("  FAIL case %u: %d slots visible, want %d\n", c, vis, ve - vs + 1);
                    bad++;
                }
            }
        }

        /* a placeholder/image count mismatch is refused, not guessed */
        if (!bad) {
            pulsar_tokens o2 = {};
            pulsar_vision_prepared p2; memset(&p2, 0, sizeof p2);
            int s2 = -1;
            if (vision_expand_image_placeholders(&o2, &in, PLACEHOLDER, NULL, 0,
                                                 &args, vocab, &p2, &s2)) {
                printf("  FAIL case %u: a prompt with a placeholder and zero images was accepted\n", c);
                bad++;
            }
            pulsar_tokens_free(&o2);
        }

        printf("%s case %u: %dx%d start=%-3d span %-4d -> %d tokens\n",
               bad ? "FAIL" : "ok  ", c, n_vh, n_vw, start_pos, span_len, out.len);
        failures += bad;
        checked++;
        vision_prepared_free(&prep);
        pulsar_tokens_free(&out);
        pulsar_tokens_free(&in);
    }

    fclose(f);
    printf("VISION PLACEHOLDER GATE: %s (%d cases, %d failures)\n",
           failures ? "FAIL" : "PASS", checked, failures);
    return failures ? 1 : 0;
}

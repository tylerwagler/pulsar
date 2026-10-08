/* VISION CHUNK GATE (L283): an image block's logits do not depend on where the prefill chunk that holds it starts.
 *
 * The same image prompt is prefilled cold -- the block in a chunk that starts at 0 -- and again with that chunk
 * starting elsewhere, on the public session API: a head synced first, then the whole image prompt, which resumes at
 * the head (pulsar_session_resume_origin; asserted).  The last row's logits must be byte-identical in every case --
 * the bar text rows already meet (cuda-chunk-neutrality-gate).  Every prompt ends 3 tokens past the first grid point
 * at or after its block, so both runs make the same final cut there (pulsar_ckpt_final_cut) and differ only in where
 * the block's chunk starts.  Each run prints its chunks, from the one cut rule (pulsar_prefill_plan_next_end) at the
 * start the engine reported, and asserts the chunk that owns the block starts where the case says:
 *
 *   before    block at 300; the resumed run's chunk starts at 256, before it
 *   at        block at 256; the resumed run's chunk starts exactly at it
 *   straddle  block across the 256 grid point; the resumed run's chunk starts at 128
 *   after     block at 40, prefilled by the head [0, 384) (a chunk from 0 in both runs); the resumed run continues
 *             from 384 over the block's cached rows
 *   deep      block at 2200; the resumed run's chunk starts at 2176 -- past 512 compressed rows, so the ratio-4
 *             layers take the indexed (top-k) arm rather than the visible-prefix one
 *
 * Why the chunk from 0 is the right side: the reference merges an image only on its start_pos == 0 pass, where the
 * attention of every row inside an image span reaches the whole span (get_image_visible /
 * get_window_topk_idxs_visible: the one bidirectional region of the window), whatever its absolute position.  A
 * chunk that starts later must give the same rows.  Before L283 the chunk-past-0 attention arms (the raw ring and the
 * visible-prefix mixed arm, attention_decode_batch_launch) passed no span visibility and ran the block causally:
 * relL2 0.21..0.67 on the last row, every logit moved; text rows untouched.
 *
 * Every family that serves images runs it (the runner's NEED_VISION); the heads are on DeepSeek's 128 resume grid, so
 * a split-invariant family resumes at the same points.
 *
 *   ./tests/vision_chunk_gate MODEL [GOLDENS]   (GOLDENS: tests/test-vectors/vision-image-goldens.bin, run from the
 *                                                repo root; only the encoded images are read)
 */
#include "pulsar.h"
#include "gate_entry.h"
#include "pulsar_engine_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define VCG_CTX 4096
#define VCG_GRID 128u   /* where the heads and the final cuts sit: DeepSeek's resume grid */

static bool rd(void *dst, size_t n, FILE *f) { return n == 0 || fread(dst, 1, n, f) == n; }
static bool rd_u32(FILE *f, uint32_t *v) {
    uint8_t b[4];
    if (!rd(b, 4, f)) return false;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return true;
}

/* The encoded images of the vision-image goldens (VIG1); the rest of each case is skipped. */
static bool read_images(const char *path, std::vector<std::vector<uint8_t>> *out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "vision chunk gate: cannot open %s\n", path); return false; }
    char magic[4];
    uint32_t n = 0;
    bool ok = rd(magic, 4, f) && !memcmp(magic, "VIG1", 4) && rd_u32(f, &n);
    for (uint32_t c = 0; ok && c < n; c++) {
        uint32_t hdr[11], enc_len = 0;   /* patch .. span_len (index 9), a float, then enc_len */
        ok = rd(hdr, sizeof hdr, f) && rd_u32(f, &enc_len);
        const uint32_t span_len = hdr[9];
        std::vector<uint8_t> enc(enc_len);
        ok = ok && rd(enc.data(), enc.size(), f) &&
             fseek(f, (long)span_len * 4 + (long)span_len * PULSAR_VISION_GOLDEN_N_EMBD * 2, SEEK_CUR) == 0;
        if (ok) out->push_back(std::move(enc));
    }
    fclose(f);
    if (!ok || out->empty()) fprintf(stderr, "vision chunk gate: %s is not a readable VIG1 file\n", path);
    return ok && !out->empty();
}

/* `pre` text tokens, the image's block where the placeholder expands (img->start_pos), then text to `total` tokens
 * or, at 0, to 3 past the first grid point at or after the block's end.  *block_len = the block's length. */
static bool build_prompt(pulsar_engine *e, int pre, int total, pulsar_image_ref *img, pulsar_tokens *out,
                         int *block_len, char *err, size_t errlen) {
    pulsar_tokens raw = {};
    for (int i = 0; i < pre; i++) pulsar_tokens_push(&raw, 100 + i % 7);
    pulsar_tokens_push(&raw, e->family->vision->placeholder_id(e));
    bool ok = pulsar_expand_image_placeholders(e, &raw, 0, img, 1, out, err, errlen) == 1;
    pulsar_tokens_free(&raw);
    if (ok && !(img->start_pos == pre &&
                e->family->vision->block_extent(e, out->v, out->len, pre, block_len) && pre + *block_len == out->len)) {
        snprintf(err, errlen, "the expansion did not put one block at %d", pre);
        ok = false;
    }
    if (!ok) return false;
    const int end = total > 0 ? total : (int)((out->len + VCG_GRID - 1u) / VCG_GRID * VCG_GRID) + 3;
    for (int i = out->len; i < end; i++) pulsar_tokens_push(out, 200 + i % 5);
    return out->len == end;
}

/* The chunks a sync of `p` from `start` makes -- the prefill loop's own cut rule over the session's shape -- into
 * `desc`; returns the start of the chunk that owns position `bs`. */
static uint32_t plan_chunks(pulsar_session *s, const pulsar_tokens *p, const pulsar_image_ref *img, uint32_t start,
                            uint32_t bs, char *desc, size_t n) {
    pulsar_prefill_shape shape = {};
    s->engine->family->session->sync->prefill_shape(s, &shape);
    pulsar_prefill_plan plan;
    pulsar_prefill_plan_init(&plan, &shape, start, pulsar_session_kv_store(s), s->engine, p, img, 1);
    uint32_t owner = UINT32_MAX;
    size_t w = 0;
    desc[0] = 0;
    for (uint32_t pos = start; pos < (uint32_t)p->len;) {
        const uint32_t end = pulsar_prefill_plan_next_end(&plan, pos, (uint32_t)p->len);
        if (end <= pos) break;
        if (bs >= pos && bs < end) owner = pos;
        if (w < n) w += (size_t)snprintf(desc + w, n - w, " [%u,%u)", pos, end);
        pos = end;
    }
    return owner;
}

/* Prefill `prompt` in a fresh session: its first `head` tokens (0 = none; `head_img`: the head holds the block), then
 * the whole prompt, which must resume at `head`.  The block's chunk -- in the head's sync when the head holds it --
 * must start at `owner`. */
static bool run(pulsar_engine *e, const pulsar_tokens *prompt, const pulsar_image_ref *img, int head, bool head_img,
                uint32_t owner, const char *tag, float *logits, int width, char *err, size_t errlen) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, e, VCG_CTX) != 0) { snprintf(err, errlen, "session create failed"); return false; }
    const uint32_t bs = (uint32_t)img->start_pos;
    char head_desc[256] = "", desc[256] = "";
    uint32_t got_owner = UINT32_MAX;
    pulsar_tokens h = *prompt;
    h.len = head;
    bool ok = true;
    if (head > 0) {
        if (head_img) got_owner = plan_chunks(s, &h, img, 0u, bs, head_desc, sizeof head_desc);
        ok = (head_img ? pulsar_session_sync_mm(s, &h, img, 1, err, errlen)
                       : pulsar_session_sync(s, &h, err, errlen)) == 0;
    }
    if (ok) {
        const uint32_t d = plan_chunks(s, prompt, img, (uint32_t)head, bs, desc, sizeof desc);
        if (!head_img) got_owner = d;
        ok = pulsar_session_sync_mm(s, prompt, img, 1, err, errlen) == 0;
    }
    const int origin = ok ? pulsar_session_resume_origin(s) : -1;
    if (ok && origin != head) {
        snprintf(err, errlen, "%s: the image prompt resumed at %d, the case needs %d", tag, origin, head);
        ok = false;
    }
    if (ok && got_owner != owner) {
        snprintf(err, errlen, "%s: the block's chunk starts at %d, the case needs %u (chunks%s%s)", tag,
                 got_owner == UINT32_MAX ? -1 : (int)got_owner, owner, head_desc, desc);
        ok = false;
    }
    if (ok && pulsar_session_copy_logits(s, logits, width) != width) {
        snprintf(err, errlen, "%s: logits unreadable", tag);
        ok = false;
    }
    if (ok) printf("    %-8s chunks%s%s%s\n", tag, head_desc, head_img ? " then" : "", desc);
    pulsar_session_free(s);
    return ok;
}

int GATE_ENTRY(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [GOLDENS]\n", argv[0]); return 2; }
    std::vector<std::vector<uint8_t>> images;
    if (!read_images(argc > 2 ? argv[2] : "tests/test-vectors/vision-image-goldens.bin", &images)) return 2;
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    opt.prefill_chunk = 4096;   /* cuda-chunk-neutrality-gate's engine: every prompt here is under one chunk */
    opt.dspark_disable = true;
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "vision chunk gate: engine open failed\n"); return 2; }
    if (!pulsar_engine_has_vision(e)) {
        fprintf(stderr, "vision chunk gate: %s serves no images\n", pulsar_engine_family_name(e));
        gate_engine_close(e);
        return 2;
    }
    const int width = pulsar_engine_logits_width(e);
    const char *family = pulsar_engine_family_name(e);
    std::vector<float> cold((size_t)width), cur((size_t)width);
    int failures = 0, checks = 0;

    for (size_t c = 0; c < images.size(); c++) {
        pulsar_image_ref img = { images[c].data(), images[c].size(), 0 };
        /* the straddle case's start: 256 inside the block (its length depends on where it starts only through the
         * compressor pad, so the length at 250 places it) */
        char err[512] = {0};
        pulsar_tokens probe = {};
        int straddle_pre = -1, len = 0;
        if (build_prompt(e, 250, 0, &img, &probe, &len, err, sizeof err)) {
            straddle_pre = 256 - len / 2;
            if (straddle_pre <= 128) straddle_pre = 129;
        }
        pulsar_tokens_free(&probe);
        /* `cross`: a grid point the block must straddle (0 = none) */
        struct vcase { const char *label; int pre, total, head; bool head_img; int cross; } cases[] = {
            {"before: block at 300, its chunk from 256", 300, 0, 256, false, 0},
            {"at: block at 256, its chunk from 256", 256, 0, 256, false, 0},
            {"straddle: block across 256, its chunk from 128", straddle_pre, 0, 128, false, 256},
            {"after: block at 40 in [0,384), resumed at 384", 40, 450, 384, true, 0},
            /* past 512 compressed rows on the ratio-4 layers: the top-k (indexed) arm */
            {"deep: block at 2200, its chunk from 2176", 2200, 0, 2176, false, 0},
        };
        for (const vcase &k : cases) {
            pulsar_tokens prompt = {};
            int block_len = 0;
            err[0] = 0;
            bool ok = k.pre > 0 && build_prompt(e, k.pre, k.total, &img, &prompt, &block_len, err, sizeof err);
            if (!ok && !err[0]) snprintf(err, sizeof err, "no block length to place the straddle case");
            /* the case is the case it says: where the block sits against the head */
            if (ok && k.head > k.pre && k.head < k.pre + block_len) {
                snprintf(err, sizeof err, "head %d is inside the block [%d,%d)", k.head, k.pre, k.pre + block_len);
                ok = false;
            }
            if (ok && k.cross && !(k.pre < k.cross && k.cross < k.pre + block_len)) {
                snprintf(err, sizeof err, "the block [%d,%d) does not straddle %d", k.pre, k.pre + block_len, k.cross);
                ok = false;
            }
            if (ok) printf("  image %zu  %s: block [%d,%d), prompt %d\n", c, k.label, k.pre, k.pre + block_len,
                           prompt.len);
            ok = ok && run(e, &prompt, &img, 0, false, 0u, "cold", cold.data(), width, err, sizeof err) &&
                 run(e, &prompt, &img, k.head, k.head_img, k.head_img ? 0u : (uint32_t)k.head, "resumed",
                     cur.data(), width, err, sizeof err);
            checks++;
            if (!ok) {
                printf("  FAIL image %zu  %s: %s\n", c, k.label, err);
                failures++;
                pulsar_tokens_free(&prompt);
                continue;
            }
            int differ = 0, am_c = 0, am_r = 0;
            double num = 0.0, den = 0.0, worst = 0.0;
            for (int i = 0; i < width; i++) {
                differ += memcmp(&cold[(size_t)i], &cur[(size_t)i], sizeof(float)) != 0;
                const double d = (double)cur[(size_t)i] - (double)cold[(size_t)i];
                num += d * d;
                den += (double)cold[(size_t)i] * cold[(size_t)i];
                if (fabs(d) > worst) worst = fabs(d);
                if (cold[(size_t)i] > cold[(size_t)am_c]) am_c = i;
                if (cur[(size_t)i] > cur[(size_t)am_r]) am_r = i;
            }
            printf("  %s image %zu  %s: %d/%d last-row logits differ from cold (relL2 %.3g, max|d| %.3g, "
                   "argmax %d/%d)\n", differ ? "FAIL" : "ok  ", c, k.label, differ, width,
                   den > 0.0 ? sqrt(num / den) : 0.0, worst, am_c, am_r);
            failures += differ != 0;
            pulsar_tokens_free(&prompt);
        }
    }
    printf("VISION CHUNK GATE [%s]: %s (%d cases, %d failures)\n", family, failures ? "FAIL" : "PASS", checks,
           failures);
    gate_engine_close(e);
    return failures ? 1 : 0;
}

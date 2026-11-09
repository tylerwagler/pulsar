/* L216 gate: image-span VISIBILITY, graded against the checkpoint's own code.
 *
 * An image span is the one place the model is bidirectional: a patch has to see
 * its whole image rather than just the causal sliding window, or each aligner
 * row would be computed from a different prefix of the image.  The reference
 * expresses that as two pure integer functions (inference/model.py):
 *
 *     get_image_visible(input_ids, vocab_size, max_image_tokens)
 *         valid  = (cumsum(is_start) > cumsum(is_end)) | is_end
 *         starts = where(is_start, idx, 0).cummax()
 *         left   = ((idx - starts) * valid).clamp(max=max_image_tokens - 1)
 *         ends   = where(is_end, idx, seqlen).flip().cummin().flip()
 *         right  = ((ends - idx) * valid).clamp(max=max_image_tokens)
 *
 *     get_window_topk_idxs_visible(window_size, seqlen, left, right, max_image_tokens)
 *         width     = min(seqlen, window_size + max_image_tokens)
 *         left_add  = (left - (window_size - 1)).clamp(min=0)
 *         starts    = (idx - (window_size - 1) - left_add).clamp(min=0)
 *         matrix    = starts + arange(width),  -1 where > idx + right
 *
 * which tests/vision_visible_goldens.py dumps by running the reference itself.
 *
 * EXACT, NO TOLERANCE.  This is integer arithmetic; a "close" answer is a wrong
 * answer.  Every id, count and matrix entry is compared for equality.
 *
 * The fixture also carries the reference's PLAIN window matrix
 * (get_window_topk_idxs) for each sequence, which is what the engine uses when
 * there is no image.  The gate asserts across cases that the fixture covers
 * BOTH sides -- at least one sequence where visibility changes the matrix and
 * at least one where it must leave it alone -- so a port that ignored the image
 * span entirely, or one that applied it everywhere, cannot pass.  (The malformed
 * cases matter here: an END with no START is defined by the reference to change
 * nothing, and a port that "cleaned that up" would diverge.)
 *
 * Needs no GPU and no model.
 */
#include "pulsar.h"
#include "pulsar_gpu.h"
#include "pulsar_engine_internal.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

enum { NAME_LEN = 32 };

struct Case {
    char name[NAME_LEN + 1];
    uint32_t n, width;
    std::vector<int32_t> ids, left, right, matrix, plain;
};

static uint32_t rd_u32(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static int cmp_vec(const std::vector<int32_t> &got, const std::vector<int32_t> &want,
                   const char *what, const char *name) {
    for (size_t i = 0; i < want.size(); i++) {
        if (got[i] != want[i]) {
            printf("  %s: %s[%zu] = %d, reference %d\n", name, what, i, got[i], want[i]);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "tests/test-vectors/vision-visible-goldens.bin";
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "vision_visible_gate: cannot open %s\n", path);
        fprintf(stderr, "  (regenerate with: python3 tests/vision_visible_goldens.py <snapshot-dir> "
                        "> tests/test-vectors/vision-visible-goldens.bin)\n");
        return 2;
    }
    std::vector<uint8_t> raw;
    {
        fseek(f, 0, SEEK_END);
        const long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (sz < 24) { fprintf(stderr, "vision_visible_gate: %s is truncated\n", path); fclose(f); return 2; }
        raw.resize((size_t)sz);
        if (fread(raw.data(), 1, (size_t)sz, f) != (size_t)sz) { fclose(f); return 2; }
    }
    fclose(f);

    if (memcmp(raw.data(), "VVS1", 4) != 0) {
        fprintf(stderr, "vision_visible_gate: %s is not a VVS1 fixture\n", path);
        return 2;
    }
    const uint32_t n_cases = rd_u32(&raw[4]);
    const int window = (int)rd_u32(&raw[8]);
    const int n_vocab = (int)rd_u32(&raw[12]);
    const int max_img = (int)rd_u32(&raw[16]);

    size_t off = 24;
    std::vector<Case> cases;
    for (uint32_t ci = 0; ci < n_cases; ci++) {
        if (off + NAME_LEN + 8 > raw.size()) { fprintf(stderr, "truncated\n"); return 2; }
        Case c;
        memcpy(c.name, &raw[off], NAME_LEN);
        c.name[NAME_LEN] = '\0';
        off += NAME_LEN;
        c.n = rd_u32(&raw[off]);
        c.width = rd_u32(&raw[off + 4]);
        off += 8;
        /* ids + left + right are 3n words; matrix + plain are n*width each. */
        const size_t words = (size_t)c.n * 3 + (size_t)c.n * c.width * 2;
        if (off + words * 4 > raw.size()) { fprintf(stderr, "truncated at %s\n", c.name); return 2; }
        const int32_t *p = (const int32_t *)(const void *)&raw[off];
        c.ids.assign(p, p + c.n); p += c.n;
        c.left.assign(p, p + c.n); p += c.n;
        c.right.assign(p, p + c.n); p += c.n;
        c.matrix.assign(p, p + (size_t)c.n * c.width); p += (size_t)c.n * c.width;
        c.plain.assign(p, p + (size_t)c.n * c.width);
        off += words * 4;
        cases.push_back(std::move(c));
    }

    int failures = 0, checked = 0, n_differs = 0, n_same = 0;
    for (const Case &c : cases) {
        int bad = 0;

        /* The engine's own width must agree with the reference's, because the
         * caller sizes the matrix from it. */
        const int w = vision_visible_width((int)c.n, window, max_img);
        if ((uint32_t)w != c.width) {
            printf("  %s: width %d, reference %u\n", c.name, w, c.width);
            bad++;
        }

        std::vector<int32_t> left(c.n, 0), right(c.n, 0);
        vision_image_visible(c.ids.data(), (int)c.n, n_vocab, max_img, left.data(), right.data());
        bad += cmp_vec(left, c.left, "left", c.name);
        bad += cmp_vec(right, c.right, "right", c.name);

        const int differs = (c.matrix != c.plain);
        if (differs) n_differs++; else n_same++;

        std::vector<int32_t> matrix((size_t)c.n * (size_t)w, -7);
        vision_window_topk_visible(window, (int)c.n, c.left.data(), c.right.data(), max_img, matrix.data());
        if (matrix != c.matrix) {
            for (size_t i = 0; i < matrix.size(); i++) {
                if (matrix[i] != c.matrix[i]) {
                    printf("  %s: matrix[%zu] (row %zu, col %zu) = %d, reference %d\n",
                           c.name, i, i / (size_t)w, i % (size_t)w, matrix[i], c.matrix[i]);
                    break;
                }
            }
            bad++;
        }

        printf("%s %-34s n=%-4u width=%-4d visibility %s\n", bad ? "FAIL" : "ok  ", c.name, c.n,
               c.width, differs ? "widens the window" : "leaves the window alone");
        failures += bad;
        checked += (int)c.n;
    }

    /* ---- the fixture must cover BOTH sides -------------------------------- */
    if (n_differs == 0 || n_same == 0) {
        printf("FAIL fixture is one-sided: %d cases change the window, %d leave it alone\n",
               n_differs, n_same);
        failures++;
    } else {
        printf("ok   fixture covers both sides (%d change the window, %d leave it alone)\n",
               n_differs, n_same);
    }

    printf("VISION VISIBLE GATE: %s (%d cases, %d rows checked, %d failures)\n",
           failures ? "FAIL" : "PASS", (int)cases.size(), checked, failures);
    return failures ? 1 : 0;
}

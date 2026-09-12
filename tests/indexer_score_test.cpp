/* HOST oracle for the V4 indexer's SCORING stage.
 *
 * The last host-verifiable piece of the indexer port.  The indexer's own
 * compressor and the Hadamard are device work (rows/L218.md s29), but the
 * scoring -- the combine, the future mask and the top-k sentinel -- is pure
 * arithmetic over rows, and three of its details fail by producing a plausible
 * answer rather than an error:
 *
 *   1. the mask is applied to the SUMMED score, not to the per-head dots.
 *      Masking first would relu(-inf) to 0 and sum to 0, so a future row would
 *      score ZERO and become selectable.
 *   2. the mask is `t >= (s + 1) / ratio` -- a compressed row becomes visible to
 *      query s only once s has reached the end of that row's token group.  Off
 *      by one either way still selects plausible rows.
 *   3. the -1 sentinel: k = min(index_topk, end_pos / ratio), so early in a
 *      sequence k exceeds the number of VISIBLE rows and the extra picks are
 *      -inf entries; every selected index >= (s+1)/ratio is rewritten to -1.
 *
 * This is deliberately a transcription rather than a shared fixture: when the
 * device gate lands it compares the MXFP4 tier's kernels against THIS, and an
 * oracle sharing code with the thing under test can be wrong in the same
 * direction.  Transcribed from /mnt/models/dsv4-flash-0731/inference/model.py
 * class Indexer, forward() lines 408-439.
 *
 * The gate compares the masked scores (tolerance, -inf aware -- the reference and
 * the engine both fp4-quantise the operands, so the scores cannot be exact) and
 * the selected indices as a MULTISET (exact).  A multiset because torch.topk
 * does not define the order of equal values, attention is order-invariant, and
 * grading an unspecified order would be grading nothing -- the generator asserts
 * instead that the selection BOUNDARY never ties, which is the part that would
 * make WHICH rows are chosen unspecified.
 */

#include <math.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>

#include "indexer_score_fixture.h"

static unsigned g_checks, g_failures, g_printed;
#define CHECK_DETAIL_MAX 12

static void check(int ok, const char *fmt, ...) {
    g_checks++;
    if (ok) return;
    g_failures++;
    if (g_printed++ >= CHECK_DETAIL_MAX) {
        if (g_printed == CHECK_DETAIL_MAX + 1) puts("FAIL: ... (further failures counted only)");
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    fputs("FAIL: ", stdout);
    vprintf(fmt, ap);
    putchar('\n');
    va_end(ap);
}

/* fp4 operands on both sides: the scores agree to a tolerance, never exactly. */
#define TOL 2e-3f

static int score_close(float a, float b) {
    if (isinf(a) || isinf(b)) return isinf(a) && isinf(b) && ((a < 0) == (b < 0));
    return fabsf(a - b) <= TOL;
}

static int cmp_i32(const void *a, const void *b) {
    const int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

/* ---- the transcription ------------------------------------------------- */

/* index_score = (relu(q . kv_c) * weights).sum(heads), then the future mask,
 * then top-k with the -1 sentinel.  Out is sorted: see the header. */
static void indexer_score(const float *q,        /* [seqlen][H][D] */
                          const float *kv_c,     /* [n_comp][D] */
                          const float *wproj,    /* [seqlen][H] */
                          uint32_t seqlen, uint32_t n_comp, uint32_t ratio,
                          uint32_t start_pos, uint32_t index_topk, uint32_t offset,
                          float *scores_out,     /* [seqlen][n_comp] (or NULL) */
                          int32_t *out,          /* [seqlen][k] */
                          uint32_t *k_out) {
    const float scale = powf((float)IDX_D, -0.5f) * powf((float)IDX_H, -0.5f);
    const uint32_t end_pos = start_pos + seqlen;
    uint32_t k = end_pos / ratio;
    if (k > index_topk) k = index_topk;
    *k_out = k;

    for (uint32_t s = 0; s < seqlen; s++) {
        for (uint32_t t = 0; t < n_comp; t++) {
            float acc = 0.0f;
            for (uint32_t h = 0; h < IDX_H; h++) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < IDX_D; d++) dot += q[(s * IDX_H + h) * IDX_D + d] * kv_c[t * IDX_D + d];
                if (dot < 0.0f) dot = 0.0f;                     /* relu, BEFORE the weight */
                acc += dot * (wproj[s * IDX_H + h] * scale);    /* weights_proj * softmax_scale * H^-0.5 */
            }
            /* the mask lands on the SUMMED score (detail 1) */
            if (start_pos == 0 && t >= (s + 1u) / ratio) acc = -INFINITY;
            if (scores_out) scores_out[s * n_comp + t] = acc;
        }
    }

    /* top-k over the compressed rows; ties broken by lower index (harmless, and
     * the generator asserts the selection boundary never ties) */
    for (uint32_t s = 0; s < seqlen; s++) {
        /* `chosen` tracks RAW row indices.  Testing the OUTPUT for "already
         * taken" instead is a real trap: the output holds the mapped value
         * (index + offset, or -1), so the comparison never matches and the same
         * row is selected k times. */
        uint8_t chosen[64];
        memset(chosen, 0, sizeof chosen);
        for (uint32_t r = 0; r < k; r++) {
            uint32_t best = 0;
            int have = 0;
            for (uint32_t t = 0; t < n_comp; t++) {
                if (chosen[t]) continue;
                if (!have || scores_out[s * n_comp + t] > scores_out[s * n_comp + best]) { best = t; have = 1; }
            }
            if (!have) { out[s * k + r] = -1; continue; }
            chosen[best] = 1;
            /* detail 3: a selected row that is not yet visible becomes -1 */
            out[s * k + r] = (start_pos == 0 && (int32_t)best >= (int32_t)((s + 1u) / ratio))
                                 ? -1 : (int32_t)best + (int32_t)offset;
        }
        qsort(&out[s * k], k, sizeof(int32_t), cmp_i32);
    }
}

/* ---- the gate ----------------------------------------------------------- */

static void run_case(const char *name, uint32_t ratio, uint32_t start_pos, uint32_t seqlen,
                     uint32_t n_comp, uint32_t index_topk, uint32_t offset,
                     const float *q, const float *kv_c, const float *wproj, const float *weights,
                     const float *want_score, const int32_t *want_idx) {
    /* the fixture carries weights = wproj * scale as its own column, so assert
     * the scale formula on the way in rather than silently trusting either side */
    const float scale = powf((float)IDX_D, -0.5f) * powf((float)IDX_H, -0.5f);
    for (uint32_t s = 0; s < seqlen; s++)
        for (uint32_t h = 0; h < IDX_H; h++)
            check(fabsf(wproj[s * IDX_H + h] * scale - weights[s * IDX_H + h]) < 1e-6f,
                  "%s: fixture weights[%u][%u] is not wproj * softmax_scale * H^-0.5", name, s, h);

    static float scores[64 * 64];
    static int32_t got[64 * 64];
    uint32_t k = 0;
    memset(scores, 0, sizeof scores);
    memset(got, 0, sizeof got);
    indexer_score(q, kv_c, wproj, seqlen, n_comp, ratio, start_pos, index_topk, offset,
                  n_comp ? scores : NULL, got, &k);

    uint32_t want_k = (start_pos + seqlen) / ratio;
    if (want_k > index_topk) want_k = index_topk;
    check(k == want_k, "%s: k is %u, want %u (min(index_topk, end_pos/ratio))", name, k, want_k);

    uint32_t score_mismatch = 0, idx_mismatch = 0;
    if (n_comp) {
        for (uint32_t s = 0; s < seqlen; s++) {
            for (uint32_t t = 0; t < n_comp; t++) {
                const uint32_t i = s * n_comp + t;
                if (!score_close(scores[i], want_score[i])) {
                    if (score_mismatch < 4)
                        check(0, "%s: score[%u][%u] got %.6g want %.6g", name, s, t, scores[i], want_score[i]);
                    score_mismatch++;
                }
            }
        }
    }
    for (uint32_t s = 0; s < seqlen && k; s++) {
        for (uint32_t r = 0; r < k; r++) {
            const int32_t a = got[s * k + r], b = want_idx[s * k + r];
            if (a != b) {
                if (idx_mismatch < 4)
                    check(0, "%s: selected[%u][%u] got %d want %d", name, s, r, a, b);
                idx_mismatch++;
            }
        }
    }
    check(score_mismatch == 0, "%s: %u summed scores wrong", name, score_mismatch);
    check(idx_mismatch == 0, "%s: %u selected indices wrong", name, idx_mismatch);

    printf("--- %-6s ratio %u start_pos %-3u seqlen %-3u n_comp %u  k %u  %u rows selected\n",
           name, ratio, start_pos, seqlen, n_comp, k, seqlen * k);
}

int main(void) {
    printf("indexer score gate: V4 scoring vs the shipped reference\n");
    run_case("pref", PREF_RATIO, PREF_START_POS, PREF_SEQLEN, PREF_NCOMP, 512u, PREF_OFFSET,
             &PREF_Q[0][0][0], &PREF_KV[0][0], &PREF_WPROJ[0][0], &PREF_WEIGHTS[0][0],
             &PREF_SCORE[0][0], &PREF_EXPECT[0][0]);
    run_case("dec", DEC_RATIO, DEC_START_POS, DEC_SEQLEN, DEC_NCOMP, 512u, DEC_OFFSET,
             &DEC_Q[0][0][0], &DEC_KV[0][0], &DEC_WPROJ[0][0], &DEC_WEIGHTS[0][0],
             &DEC_SCORE[0][0], &DEC_EXPECT[0][0]);
    run_case("tiny", TINY_RATIO, TINY_START_POS, TINY_SEQLEN, TINY_NCOMP, 512u, TINY_OFFSET,
             &TINY_Q[0][0][0], &TINY_KV[0][0], &TINY_WPROJ[0][0], &TINY_WEIGHTS[0][0],
             &TINY_SCORE[0][0], &TINY_EXPECT[0][0]);
    /* index_topk 1 against 2 compressed rows: the min() actually bites, which it
     * never does at 512 until long context */
    run_case("cap", CAP_RATIO, CAP_START_POS, CAP_SEQLEN, CAP_NCOMP, 1u, CAP_OFFSET,
             &CAP_Q[0][0][0], &CAP_KV[0][0], &CAP_WPROJ[0][0], &CAP_WEIGHTS[0][0],
             &CAP_SCORE[0][0], &CAP_EXPECT[0][0]);
    printf("indexer score gate: %u checks, %u failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

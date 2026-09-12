/* L216 gate: the MoE router's IMAGE-SLOT bias (bias_vl).
 *
 * WHAT IS PINNED.  The reference's gate keys on an out-of-vocab image mask:
 *
 *     image_mask = (input_ids >= self.vocab_size) if self.bias_vl is not None else None
 *     if self.hash:
 *         if image_mask is None:
 *             indices = self.tid2eid[input_ids]
 *         else:
 *             indices = self.tid2eid[torch.where(image_mask, 0, input_ids)]
 *             vl_indices = (scores + self.bias_vl).topk(self.topk, dim=-1)[1]
 *             indices = torch.where(image_mask.unsqueeze(-1), vl_indices.to(indices.dtype), indices)
 *     else:
 *         if image_mask is None:
 *             scores = scores + self.bias
 *         else:
 *             scores = scores + torch.where(image_mask.unsqueeze(-1), self.bias_vl, self.bias)
 *         indices = scores.topk(self.topk, dim=-1)[1]
 *     weights = original_scores.gather(1, indices)
 *     weights /= weights.sum(dim=-1, keepdim=True)
 *     weights *= self.route_scale
 *
 * so an image slot selects with bias_vl instead of the text bias, and on a hash
 * layer it ESCAPES tid2eid entirely -- that table has one row per real token
 * and no row for a sentinel.  The routing WEIGHTS still come from the UNBIASED
 * scores (`original_scores`), which is the pre-existing contract.
 *
 * THE CONSTRUCTION IS DETERMINISTIC, NOT RANDOM.  A random layer makes this
 * gate flaky in the one place it must not be: whether a case is DISCRIMINATING.
 * With random scores the gap between the 6th and 7th expert is occasionally
 * smaller than the device's fast-math softplus noise, and the case then proves
 * nothing while still reporting "ok".  So the scores are separated by
 * construction:
 *
 *   - the 256 experts get strictly descending softplus scores with a step of
 *     5e-3, and every expert in TEXT_SET / IMAGE_SET is ranked LAST, so no
 *     unbiased leader can outscore a biased expert;
 *   - the text bias  is +2 on TEXT_SET  -> a text slot selects TEXT_SET;
 *   - bias_vl        is +4 on IMAGE_SET -> an image slot selects IMAGE_SET.
 *
 * TEXT_SET, IMAGE_SET, the tid2eid rows and the score leaders are pairwise
 * disjoint, so "which bias was used" is answerable by inspecting WHICH SET came
 * back -- no re-derivation of a topk, and no coincidence can imitate it.  Every
 * decisive gap is >= 0.5, orders of magnitude above the noise, and the gate
 * asserts the observed gap anyway so a later edit to the construction cannot
 * silently make a case vacuous.
 *
 * ALL CASES SHARE ONE BATCH (one logits vector, one bias, one tid2eid).  The
 * cases differ only in hash_mode, has_vl_bias, which region bias_vl points at,
 * and the tokens -- so "a text-only batch is unchanged by the presence of
 * bias_vl" is a controlled comparison of identical inputs, not two draws.
 *
 * THE ORACLE.  Needs a GPU, no model.  The expected result comes from
 * `reference_gate()` below, a line-for-line transcription of the block above.
 * A transcription is a weaker oracle than running the reference: the snapshot's
 * own model.py cannot be imported on this machine today (no torch anywhere, and
 * sparky's one venv is a python3.12 tree under a python3.14 interpreter), so
 * generating these goldens from the reference is still owed.  The GUARD
 * section is what keeps the gate from being vacuous in the meantime: a kernel
 * that ignored bias_vl, applied it to text slots, or took the biased score for
 * the weights FAILS here on set identity alone, whatever the transcription
 * believes.
 *
 * WHY INDICES ARE EXACT AND WEIGHTS ARE NOT.  The kernel is compiled with
 * --use_fast_math (approximate expf/log1pf on the device) while this host side
 * is built -fno-fast-math and uses libm, so the two softplus() implementations
 * differ by an ulp or so.  Expert INDICES are asserted exactly -- the score
 * construction puts a >= 5e-3 gap at the top-6 boundary, and the case asserts
 * it -- while the WEIGHTS are asserted to a documented tolerance and the
 * observed error is printed.
 */
#include "pulsar_gpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

enum {
    N_EXPERT = 256,
    TOPK = 6,
    N_VOCAB = 129280,
    N_SET = 6,
    BIAS_OFF = 0,
    VL_OFF = 4096,
    HASH_OFF = 8192,
    /* Score step between consecutive experts by rank. */
    SCORE_STEP = 5e-3f,
    /* Bias magnitudes.  Each must exceed the whole score span (1.5 - 0.225)
     * plus the other's span, so the intended set wins outright. */
    TEXT_BIAS = 2.0f,
    IMAGE_BIAS = 4.0f,
};

static const int32_t TEXT_SET[N_SET] = {17, 88, 133, 200, 211, 250};
static const int32_t IMAGE_SET[N_SET] = {3, 61, 99, 150, 175, 240};
/* tid2eid rows: one distinct block of experts per token, disjoint from both
 * sets above so a hash row can never be mistaken for a bias-driven selection. */
enum { HASH_BASE = 100 };

static int in_set(const int32_t *set, int32_t e) {
    for (int i = 0; i < N_SET; i++) if (set[i] == e) return 1;
    return 0;
}

/* Softplus exactly as the kernel computes it (softplus_dev, including its
 * saturation branches -- the branch at -20 is load-bearing: it is what keeps an
 * extreme negative logit from underflowing to a different value than
 * log1pf(expf(x)) gives on the device). */
static float softplus_host(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

/* The reference's Gate.forward, transcribed.  `logits` is what the engine is
 * fed (the reference would have computed it as linear(x, weight)); everything
 * from softplus onward is the reference's. */
static void reference_gate(const float *logits, int32_t tok, int32_t hash_mode,
                           int32_t has_vl_bias, const float *bias, const float *bias_vl,
                           const int32_t *tid2eid, int32_t *idx_out, float *w_out,
                           float *margin_out) {
    float p[N_EXPERT];
    for (int e = 0; e < N_EXPERT; e++) p[e] = sqrtf(softplus_host(logits[e]));

    const int is_image = has_vl_bias && tok >= N_VOCAB;
    if (hash_mode && !is_image) {
        int32_t t = tok;
        if (t < 0 || t >= N_VOCAB) t = 0;   /* the kernel's clamp, text ids only */
        const int32_t *row = tid2eid + (size_t)t * TOPK;
        float sum = 0.0f;
        for (int j = 0; j < TOPK; j++) {
            const int32_t e = row[j];
            idx_out[j] = e;
            const float v = (e >= 0 && e < N_EXPERT) ? p[e] : 0.0f;
            w_out[j] = v;
            sum += v;
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        for (int j = 0; j < TOPK; j++) w_out[j] = w_out[j] / sum * 1.5f;
        *margin_out = INFINITY;   /* no boundary to win: the row is fixed */
        return;
    }

    /* scores = scores + where(image_mask, bias_vl, bias) */
    float s[N_EXPERT];
    for (int e = 0; e < N_EXPERT; e++) {
        const float b = is_image ? bias_vl[e] : (hash_mode ? 0.0f : bias[e]);
        s[e] = p[e] + b;
    }
    /* scores.topk(topk)[1] -- descending by score, ties to the lower id, which
     * is the kernel's router_score_better(). */
    float sum = 0.0f, sixth = 0.0f;
    for (int k = 0; k < TOPK; k++) {
        int best = -1;
        for (int e = 0; e < N_EXPERT; e++)
            if (best < 0 || s[e] > s[best]) best = e;
        if (k == TOPK - 1) sixth = s[best];
        idx_out[k] = best;
        w_out[k] = p[best];        /* weights are the UNBIASED scores */
        sum += p[best];
        s[best] = -INFINITY;
    }
    /* How much score the 6th pick won the 7th by: how much noise it would take
     * to change the selection. */
    int nxt = -1;
    for (int e = 0; e < N_EXPERT; e++)
        if (nxt < 0 || s[e] > s[nxt]) nxt = e;
    *margin_out = (nxt >= 0) ? (sixth - s[nxt]) : INFINITY;

    sum = fmaxf(sum, 6.103515625e-5f);
    for (int j = 0; j < TOPK; j++) w_out[j] = w_out[j] / sum * 1.5f;
}

static uint64_t g_rng = 0x243F6A8885A308D3ull;
static uint32_t rnd(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 11);
}

struct Case {
    const char *name;
    int32_t hash_mode;
    int32_t has_vl_bias;
    int32_t vl_is_text_bias;   /* bias_vl points AT the text bias region */
    int32_t n_rows;
    const int32_t *tokens;
};

int main(void) {
    if (pulsar_gpu_init() == 0) {
        fprintf(stderr, "vision_router_gate: no GPU\n");
        return 2;
    }

    /* ---- the synthetic layer, built once and shared by every case ---------- */
    const uint64_t hash_bytes = (uint64_t)N_VOCAB * TOPK * sizeof(int32_t);
    const uint64_t map_size = HASH_OFF + hash_bytes;
    std::vector<uint8_t> map(map_size, 0);
    float *bias = (float *)(void *)(map.data() + BIAS_OFF);
    float *bias_vl = (float *)(void *)(map.data() + VL_OFF);
    int32_t *tid2eid = (int32_t *)(void *)(map.data() + HASH_OFF);

    for (int i = 0; i < N_SET; i++) {
        bias[TEXT_SET[i]] = TEXT_BIAS;
        bias_vl[IMAGE_SET[i]] = IMAGE_BIAS;
    }
    for (int t = 0; t < 64; t++)
        for (int j = 0; j < TOPK; j++) tid2eid[(size_t)t * TOPK + j] = HASH_BASE + t * TOPK + j;

    /* logits: a strictly descending softplus score by rank, with every biased
     * expert ranked last.  p = sqrt(softplus(x)) is inverted by
     * x = log(expm1(p^2)), so the host recovers the target p to rounding and
     * the gaps below are real, not hoped for. */
    std::vector<float> logits(N_EXPERT);
    {
        int32_t order[N_EXPERT];
        int n = 0;
        for (int e = 0; e < N_EXPERT; e++)
            if (!in_set(TEXT_SET, e) && !in_set(IMAGE_SET, e)) order[n++] = e;
        for (int i = n - 1; i > 0; i--) {          /* deterministic shuffle */
            const int j = (int)(rnd() % (uint32_t)(i + 1));
            const int32_t tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }
        for (int e = 0; e < N_EXPERT; e++)
            if (in_set(TEXT_SET, e) || in_set(IMAGE_SET, e)) order[n++] = e;
        for (int r = 0; r < N_EXPERT; r++) {
            const float target = 1.5f - SCORE_STEP * (float)r;
            logits[order[r]] = logf(expm1f(target * target));
        }
    }

    const int32_t text_tokens[] = {7, 1, 40, 63};
    const int32_t mixed_tokens[] = {7, N_VOCAB + 2, 40, N_VOCAB + 4, 1, N_VOCAB + 0};
    const int32_t image_tokens[] = {N_VOCAB + 2, N_VOCAB + 0};

    const Case cases[] = {
        {"learned  text only, no bias_vl",      0, 0, 0, 4, text_tokens},
        {"learned  text only, bias_vl present", 0, 1, 0, 4, text_tokens},
        {"learned  mixed text+image",           0, 1, 0, 6, mixed_tokens},
        {"hash     text only, no bias_vl",      1, 0, 0, 4, text_tokens},
        {"hash     text only, bias_vl present", 1, 1, 0, 4, text_tokens},
        {"hash     mixed text+image",           1, 1, 0, 6, mixed_tokens},
        {"learned  image, bias_vl == bias",     0, 1, 1, 2, image_tokens},
    };
    const int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    int failures = 0, checked = 0;
    std::vector<int32_t> dev_idx_all[8];
    std::vector<float> dev_w_all[8];

    for (int ci = 0; ci < n_cases; ci++) {
        const Case &c = cases[ci];
        const uint64_t logits_bytes = (uint64_t)c.n_rows * N_EXPERT * sizeof(float);
        std::vector<float> batch((size_t)c.n_rows * N_EXPERT);
        for (int r = 0; r < c.n_rows; r++)
            memcpy(&batch[(size_t)r * N_EXPERT], logits.data(), (size_t)N_EXPERT * sizeof(float));

        std::vector<int32_t> idx((size_t)c.n_rows * TOPK), exp_idx((size_t)c.n_rows * TOPK);
        std::vector<float> w((size_t)c.n_rows * TOPK), exp_w((size_t)c.n_rows * TOPK), margins(c.n_rows);
        /* Case 7 points bias_vl at the text bias region, so the transcription
         * must be handed the same region the device is. */
        const float *vl_region = c.vl_is_text_bias ? bias : bias_vl;
        for (int r = 0; r < c.n_rows; r++)
            reference_gate(&batch[(size_t)r * N_EXPERT], c.tokens[r], c.hash_mode,
                           c.has_vl_bias, bias, vl_region, tid2eid,
                           &exp_idx[(size_t)r * TOPK], &exp_w[(size_t)r * TOPK], &margins[r]);

        const uint64_t vl_off = c.vl_is_text_bias ? BIAS_OFF : VL_OFF;
        pulsar_gpu_tensor *dlog = pulsar_gpu_tensor_alloc(logits_bytes);
        pulsar_gpu_tensor *dtok = pulsar_gpu_tensor_alloc((uint64_t)c.n_rows * sizeof(int32_t));
        pulsar_gpu_tensor *dsel = pulsar_gpu_tensor_alloc((uint64_t)c.n_rows * TOPK * sizeof(int32_t));
        pulsar_gpu_tensor *dw = pulsar_gpu_tensor_alloc((uint64_t)c.n_rows * TOPK * sizeof(float));
        int bad = 0;
        const int launched =
            dlog && dtok && dsel && dw &&
            pulsar_gpu_tensor_write(dlog, 0, batch.data(), logits_bytes) &&
            pulsar_gpu_tensor_write(dtok, 0, c.tokens, (uint64_t)c.n_rows * sizeof(int32_t)) &&
            pulsar_gpu_router_select_batch_tensor(dsel, dw, NULL, map.data(), map_size,
                                                  BIAS_OFF, HASH_OFF, N_VOCAB, 0, 0,
                                                  !c.hash_mode, c.hash_mode != 0,
                                                  dlog, dtok, N_EXPERT, TOPK, 1.5f,
                                                  (uint32_t)c.n_rows, vl_off,
                                                  N_VOCAB, c.has_vl_bias != 0) &&
            pulsar_gpu_end_commands() &&
            pulsar_gpu_tensor_read(dsel, 0, idx.data(), idx.size() * sizeof(int32_t)) &&
            pulsar_gpu_tensor_read(dw, 0, w.data(), w.size() * sizeof(float));
        if (!launched) {
            printf("FAIL %-36s: launch/copy failed\n", c.name);
            failures++;
        }

        double min_gap = INFINITY, max_werr = 0.0;
        int idx_bad = 0;
        for (int r = 0; launched && r < c.n_rows; r++) {
            const int32_t *ei = &exp_idx[(size_t)r * TOPK];
            const int32_t *di = &idx[(size_t)r * TOPK];
            if (margins[r] < min_gap) min_gap = (double)margins[r];

            /* ---- GUARD 1: the case must be able to see what it claims ------
             * Every selection boundary needs a gap wider than the device's
             * fast-math noise, and the construction above promises >= 5e-3. */
            if (margins[r] != INFINITY && margins[r] < 1.0e-3f) {
                printf("  row %d: top-6 boundary gap %.3e is too small to be conclusive\n",
                       r, (double)margins[r]);
                bad++;
            }

            /* ---- GUARD 2: the selection must BE the set the bias names -----
             * This is the guard that does not depend on the transcription: a
             * kernel that ignored bias_vl cannot return IMAGE_SET here. */
            const int is_image = c.has_vl_bias && c.tokens[r] >= N_VOCAB;
            const int32_t *want = NULL;
            if (c.hash_mode && !is_image) want = tid2eid + (size_t)(int32_t)c.tokens[r] * TOPK;
            else if (is_image) want = c.vl_is_text_bias ? TEXT_SET : IMAGE_SET;
            else if (!c.hash_mode) want = TEXT_SET;
            if (want) {
                int32_t mine[TOPK], theirs[TOPK];
                memcpy(mine, ei, sizeof(mine));
                memcpy(theirs, want, sizeof(theirs));
                for (int a = 0; a < TOPK; a++)          /* compare as sets */
                    for (int b = a + 1; b < TOPK; b++) {
                        if (mine[a] > mine[b]) { const int32_t t = mine[a]; mine[a] = mine[b]; mine[b] = t; }
                        if (theirs[a] > theirs[b]) { const int32_t t = theirs[a]; theirs[a] = theirs[b]; theirs[b] = t; }
                    }
                if (memcmp(mine, theirs, sizeof(mine)) != 0) {
                    printf("  row %d: reference selected", r);
                    for (int j = 0; j < TOPK; j++) printf(" %d", ei[j]);
                    printf("  but the biased set is");
                    for (int j = 0; j < TOPK; j++) printf(" %d", want[j]);
                    printf("\n");
                    bad++;
                }
            }
        }

        /* ---- device vs the transcription ---- */
        for (int r = 0; launched && r < c.n_rows; r++) {
            if (memcmp(&idx[(size_t)r * TOPK], &exp_idx[(size_t)r * TOPK], TOPK * sizeof(int32_t)) != 0) {
                if (idx_bad++ < 2) {
                    printf("  row %d idx: device", r);
                    for (int j = 0; j < TOPK; j++) printf(" %d", idx[(size_t)r * TOPK + j]);
                    printf("\n           ref   ");
                    for (int j = 0; j < TOPK; j++) printf(" %d", exp_idx[(size_t)r * TOPK + j]);
                    printf("\n");
                }
            }
            for (int j = 0; j < TOPK; j++) {
                const double d = fabs((double)w[(size_t)r * TOPK + j] - (double)exp_w[(size_t)r * TOPK + j]);
                if (d > max_werr) max_werr = d;
            }
        }
        const double WTOL = 1.0e-4;   /* device fast-math vs host libm in softplus */
        if (idx_bad) { printf("  %d row(s) with a different expert SET\n", idx_bad); bad++; }
        if (launched && max_werr > WTOL) {
            printf("  weights differ by %.3e (> %.0e)\n", max_werr, WTOL);
            bad++;
        }

        if (launched) {
            dev_idx_all[ci] = idx;
            dev_w_all[ci] = w;
        }
        printf("%s %-36s rows=%d min gap=%.2e max|dW|=%.2e\n", bad ? "FAIL" : "ok  ", c.name,
               c.n_rows, min_gap, max_werr);
        failures += bad;
        if (launched) checked += c.n_rows;
        pulsar_gpu_tensor_free(dlog);
        pulsar_gpu_tensor_free(dtok);
        pulsar_gpu_tensor_free(dsel);
        pulsar_gpu_tensor_free(dw);
    }

    /* ---- the controlled comparison: the same batch, with and without ------ */
    struct Pair { int a, b; const char *why; };
    const Pair same[] = {
        {1, 0, "bias_vl present must not disturb a text-only batch (learned)"},
        {4, 3, "bias_vl present must not disturb a text-only batch (hash)"},
    };
    for (const Pair &pr : same) {
        if (dev_idx_all[pr.a].empty() || dev_idx_all[pr.b].empty()) continue;
        if (dev_idx_all[pr.a] != dev_idx_all[pr.b] || dev_w_all[pr.a] != dev_w_all[pr.b]) {
            printf("FAIL %s\n", pr.why);
            failures++;
        } else {
            printf("ok   %s\n", pr.why);
        }
    }

    printf("VISION ROUTER GATE: %s (%d rows checked, %d failures)\n",
           failures ? "FAIL" : "PASS", checked, failures);
    return failures ? 1 : 0;
}

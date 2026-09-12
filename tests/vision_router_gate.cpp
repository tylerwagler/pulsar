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
 * so an image slot selects with bias_vl instead of the text bias, and on a
 * hash layer it ESCAPES tid2eid entirely -- that table has one row per real
 * token and no row for a sentinel.  The routing WEIGHTS still come from the
 * UNBIASED scores (`original_scores`), which is the pre-existing contract.
 *
 * THE ORACLE.  Needs a GPU, no model: the bias / bias_vl / tid2eid tensors are
 * synthetic and laid out in one model_map, and the expected result comes from
 * `reference_gate()` below -- a line-for-line transcription of the block above.
 * A transcription is a weaker oracle than running the reference: the snapshot's
 * own model.py cannot be imported on this machine today (no torch, and the one
 * sparky venv is a python3.12 tree under a python3.14 interpreter), so the
 * golden-from-reference step is still owed.  What keeps this gate from being
 * vacuous is the GUARD section: every case asserts its own discriminating
 * power, so a kernel that ignored bias_vl, applied it to text, or took the
 * biased score for weights FAILS here rather than passing quietly.
 *
 * WHY INDICES ARE EXACT AND WEIGHTS ARE NOT.  The kernel is compiled with
 * --use_fast_math (approximate expf/log1pf on the device) while this host side
 * uses libm, so the two softplus() implementations differ in the last ulp or
 * so.  Expert INDICES are therefore asserted exactly -- and each case proves
 * the score gap at the top-6 boundary is large enough (> 1e-3) that no such
 * noise can reorder it -- while the WEIGHTS are asserted to a documented
 * tolerance and the observed error is printed.
 */
#include "pulsar_gpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

enum { N_EXPERT = 256, TOPK = 6, N_VOCAB = 129280, BIAS_OFF = 0, VL_OFF = 4096, HASH_OFF = 8192 };

/* Softplus exactly as the kernel computes it (softplus_dev, including its
 * saturation branches -- the branch at -20 is load-bearing: it is what keeps
 * an extreme negative logit from underflowing to a different value than
 * log1pf(expf(x)) would give on the device). */
static float softplus_host(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

/* The reference's Gate.forward, transcribed.  `scores` here is the engine's
 * `logits` with softplus().sqrt() applied, i.e. `original_scores`. */
static void reference_gate(const float *logits, int32_t tok, int32_t hash_mode,
                           int32_t has_vl_bias, const float *bias, const float *bias_vl,
                           const int32_t *tid2eid, int32_t *idx_out, float *w_out,
                           float *margin_out) {
    float p[N_EXPERT];
    for (int e = 0; e < N_EXPERT; e++) p[e] = sqrtf(softplus_host(logits[e]));

    const int is_image = has_vl_bias && tok >= N_VOCAB;
    if (hash_mode && !is_image) {
        int32_t t = tok;
        if (t < 0 || t >= N_VOCAB) t = 0;   /* the kernel's own clamp, for text ids only */
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
        *margin_out = INFINITY;   /* no topk boundary: the row is fixed */
        return;
    }

    /* scores = scores + where(image_mask, bias_vl, bias) */
    float s[N_EXPERT];
    for (int e = 0; e < N_EXPERT; e++) {
        const float b = is_image ? bias_vl[e] : (hash_mode ? 0.0f : bias[e]);
        s[e] = p[e] + b;
    }
    /* scores.topk(topk)[1] -- descending by score, ties to the lower id, which
     * is what the kernel's router_score_better() does. */
    float sum = 0.0f, margin = INFINITY;
    for (int k = 0; k < TOPK; k++) {
        int best = -1;
        for (int e = 0; e < N_EXPERT; e++)
            if (best < 0 || s[e] > s[best]) best = e;
        idx_out[k] = best;
        w_out[k] = p[best];
        sum += p[best];
        s[best] = -INFINITY;
    }
    /* The gap the 6th pick won over the 7th by: how much noise it would take
     * to change the selection. */
    {
        int nxt = -1;
        for (int e = 0; e < N_EXPERT; e++)
            if (nxt < 0 || s[e] > s[nxt]) nxt = e;
        if (nxt >= 0) margin = logits[idx_out[TOPK - 1]] - logits[nxt];
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (int j = 0; j < TOPK; j++) w_out[j] = w_out[j] / sum * 1.5f;
    *margin_out = margin;
}

static uint64_t g_rng = 0x243F6A8885A308D3ull;
static uint32_t rnd(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 11);
}
static float rndf(float lo, float hi) {
    return lo + (float)rnd() / 4294967296.0f * (hi - lo);
}

/* The experts bias_vl boosts.  Image rows MUST select from this set: a kernel
 * that ignored bias_vl cannot produce them, so this is the guard that makes the
 * whole gate non-vacuous. */
static const int32_t BOOSTED[] = {17, 88, 133, 200, 211, 250};
enum { N_BOOSTED = 6 };

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

    /* ---- the synthetic layer -------------------------------------------------
     * One model_map holding all three tensors at non-overlapping offsets, the
     * way a real artifact holds them.  bias_vl is the text bias with a +5 boost
     * on BOOSTED, so "did the right bias get used" is answerable by inspection
     * rather than by re-deriving a topk. */
    const uint64_t hash_bytes = (uint64_t)N_VOCAB * TOPK * sizeof(int32_t);
    const uint64_t map_size = HASH_OFF + hash_bytes;
    std::vector<uint8_t> map(map_size, 0);
    float *bias = (float *)(void *)(map.data() + BIAS_OFF);
    float *bias_vl = (float *)(void *)(map.data() + VL_OFF);
    int32_t *tid2eid = (int32_t *)(void *)(map.data() + HASH_OFF);

    for (int e = 0; e < N_EXPERT; e++) bias[e] = rndf(-3.0f, 3.0f);
    memcpy(bias_vl, bias, N_EXPERT * sizeof(float));
    for (int i = 0; i < N_BOOSTED; i++) bias_vl[BOOSTED[i]] += 5.0f;

    /* Only the rows a case indexes need to be meaningful; every text id used
     * below gets a distinct, deterministic row.  Row 0 is what the reference
     * gathers for an image slot before discarding it, so it is filled too. */
    for (int t = 0; t < 64; t++) {
        uint32_t h = (uint32_t)t * 2654435761u;
        for (int j = 0; j < TOPK; j++) {
            h = h * 1664525u + 1013904223u;
            tid2eid[(size_t)t * TOPK + j] = (int32_t)((h >> 8) % N_EXPERT);
        }
    }

    const int32_t text_tokens[] = {7, 1, 40, 63};
    const int32_t mixed_tokens[] = {7, N_VOCAB + 2, 40, N_VOCAB + 4, 1, N_VOCAB + 0};
    const int32_t image_tokens[] = {N_VOCAB + 2, N_VOCAB + 0};

    const Case cases[] = {
        {"learned  text only, no bias_vl",     0, 0, 0, 4, text_tokens},
        {"learned  text only, bias_vl present", 0, 1, 0, 4, text_tokens},
        {"learned  mixed text+image",          0, 1, 0, 6, mixed_tokens},
        {"hash     text only, no bias_vl",     1, 0, 0, 4, text_tokens},
        {"hash     text only, bias_vl present", 1, 1, 0, 4, text_tokens},
        {"hash     mixed text+image",          1, 1, 0, 6, mixed_tokens},
        {"learned  image, bias_vl == bias",    0, 1, 1, 2, image_tokens},
    };
    const int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    int failures = 0, checked = 0;
    std::vector<int32_t> dev_idx_all[8];
    std::vector<float> dev_w_all[8];

    for (int ci = 0; ci < n_cases; ci++) {
        const Case &c = cases[ci];
        const uint64_t logits_bytes = (uint64_t)c.n_rows * N_EXPERT * sizeof(float);
        std::vector<float> logits((size_t)c.n_rows * N_EXPERT);
        for (size_t i = 0; i < logits.size(); i++) logits[i] = rndf(-4.0f, 4.0f);

        std::vector<int32_t> idx((size_t)c.n_rows * TOPK), exp_idx((size_t)c.n_rows * TOPK);
        std::vector<float> w((size_t)c.n_rows * TOPK), exp_w((size_t)c.n_rows * TOPK), margins(c.n_rows);
        for (int r = 0; r < c.n_rows; r++)
            reference_gate(logits.data() + (size_t)r * N_EXPERT, c.tokens[r], c.hash_mode,
                           c.has_vl_bias, bias, bias_vl, tid2eid,
                           &exp_idx[(size_t)r * TOPK], &exp_w[(size_t)r * TOPK], &margins[r]);

        const uint64_t vl_off = c.vl_is_text_bias ? BIAS_OFF : VL_OFF;
        pulsar_gpu_tensor *dlog = pulsar_gpu_tensor_alloc(logits_bytes);
        pulsar_gpu_tensor *dtok = pulsar_gpu_tensor_alloc((uint64_t)c.n_rows * sizeof(int32_t));
        pulsar_gpu_tensor *dsel = pulsar_gpu_tensor_alloc((uint64_t)c.n_rows * TOPK * sizeof(int32_t));
        pulsar_gpu_tensor *dw = pulsar_gpu_tensor_alloc((uint64_t)c.n_rows * TOPK * sizeof(float));
        int bad = 0;
        const int launched =
            dlog && dtok && dsel && dw &&
            pulsar_gpu_tensor_write(dlog, 0, logits.data(), logits_bytes) &&
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

        /* ---- GUARD: the case must be able to see the difference it claims ----
         * Every selection boundary needs a wide enough gap that the device's
         * fast-math softplus cannot reorder it. */
        for (int r = 0; launched && r < c.n_rows; r++) {
            if (margins[r] != INFINITY && margins[r] < 1.0e-3f) {
                printf("  row %d: topk boundary gap %.3e is too small to be conclusive\n", r,
                       (double)margins[r]);
                bad++;
            }
        }
        /* Image rows must actually land on the experts bias_vl boosts.  Without
         * this the gate would pass on a kernel that ignored bias_vl whenever
         * the two biases happened to agree. */
        if (launched && c.has_vl_bias && !c.vl_is_text_bias) {
            for (int r = 0; r < c.n_rows; r++) {
                const int is_image = c.tokens[r] >= N_VOCAB;
                if (!is_image) continue;
                int hit = 0;
                for (int j = 0; j < TOPK; j++)
                    for (int i = 0; i < N_BOOSTED; i++)
                        if (exp_idx[(size_t)r * TOPK + j] == BOOSTED[i]) hit++;
                if (hit < N_BOOSTED) {
                    printf("  row %d: image slot selected only %d of the %d boosted experts\n",
                           r, hit, N_BOOSTED);
                    bad++;
                }
                /* The hash escape: an image slot must NOT be tid2eid's row 0,
                 * which is what the reference gathers before discarding it. */
                if (c.hash_mode && memcmp(&exp_idx[(size_t)r * TOPK], tid2eid, TOPK * sizeof(int32_t)) == 0) {
                    printf("  row %d: image slot on a hash layer took the tid2eid row\n", r);
                    bad++;
                }
            }
        }

        /* ---- device vs the transcription ---- */
        int idx_bad = 0;
        double max_werr = 0.0;
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
        if (max_werr > WTOL) {
            printf("  weights differ by %.3e (> %.0e)\n", max_werr, WTOL);
            bad++;
        }

        if (launched) {
            dev_idx_all[ci] = idx;
            dev_w_all[ci] = w;
        }
        printf("%s %-36s rows=%d max|dW|=%.2e\n", bad ? "FAIL" : "ok  ", c.name, c.n_rows, max_werr);
        failures += bad;
        if (launched) checked += c.n_rows;
        pulsar_gpu_tensor_free(dlog);
        pulsar_gpu_tensor_free(dtok);
        pulsar_gpu_tensor_free(dsel);
        pulsar_gpu_tensor_free(dw);
    }

    /* ---- the cross-case invariants ---------------------------------------- */
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

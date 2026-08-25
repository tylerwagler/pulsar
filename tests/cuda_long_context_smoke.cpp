#include "pulsar_gpu.h"

/* PULSAR_ATTN_PACK row bytes, computed locally on purpose.  The engine has
 * this arithmetic too, but its macro goes through the shape GLOBAL and this
 * smoke links only the CUDA layer -- no engine objects, no model.  Must match
 * PULSAR_ATTN_PACK_ROWBYTES in src/cuda/pulsar_cuda_internal.h:
 * [n_nope e4m3][n_nope/64 E8M0, padded to 4][n_rot bf16] = 584 B at 512/64.
 *
 * This used to be a THIRD hand-written copy of the row geometry, next to the
 * engine's and the backend's.  It now asks the backend, which is the same thing
 * gpu_graph_alloc does -- a test that computes the layout itself cannot catch a
 * layout change, it can only disagree with one. */
#define SMOKE_ATTN_NROT 64u
#define SMOKE_ATTN_ROWBYTES(HD) pulsar_gpu_attn_pack_rowbytes(HD)

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static double getenv_seconds(const char *name, double fallback) {
    const char *s = getenv(name);
    if (!s || !s[0]) return fallback;
    char *end = NULL;
    const double v = strtod(s, &end);
    return end != s && v > 0.0 ? v : fallback;
}

static int check_large_topk(void) {
    const uint32_t n_comp = 32768;
    const uint32_t n_tokens = 32;
    const uint32_t top_k = 512;
    const uint64_t score_count = (uint64_t)n_comp * n_tokens;
    float *scores_host = (float *)malloc((size_t)score_count * sizeof(float));
    uint32_t *selected_host = (uint32_t *)malloc((size_t)n_tokens * top_k * sizeof(uint32_t));
    if (!scores_host || !selected_host) return 1;

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < n_comp; i++) {
            scores_host[(uint64_t)t * n_comp + i] = (float)i;
        }
    }

    pulsar_gpu_tensor *scores = pulsar_gpu_tensor_alloc(score_count * sizeof(float));
    pulsar_gpu_tensor *selected = pulsar_gpu_tensor_alloc((uint64_t)n_tokens * top_k * sizeof(uint32_t));
    int rc = 1;
    double elapsed = 0.0;
    if (scores && selected &&
        pulsar_gpu_tensor_write(scores, 0, scores_host, score_count * sizeof(float))) {
        /* Exclude one-time CUDA module/kernel setup from the throughput guard. */
        if (!pulsar_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) ||
            !pulsar_gpu_synchronize()) {
            rc = 1;
            goto cleanup;
        }
        const double t0 = monotonic_seconds();
        if (pulsar_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) &&
            pulsar_gpu_synchronize()) {
            elapsed = monotonic_seconds() - t0;
            rc = pulsar_gpu_tensor_read(selected, 0, selected_host,
                                     (uint64_t)n_tokens * top_k * sizeof(uint32_t)) ? 0 : 1;
        }
    }
    if (rc == 0) {
        for (uint32_t t = 0; t < n_tokens && rc == 0; t++) {
            for (uint32_t i = 0; i < top_k; i++) {
                const uint32_t expected = n_comp - 1u - i;
                const uint32_t got = selected_host[(uint64_t)t * top_k + i];
                if (got != expected) {
                    fprintf(stderr, "top-k mismatch token=%u rank=%u got=%u expected=%u\n",
                            t, i, got, expected);
                    rc = 1;
                    break;
                }
            }
        }
    }
    if (rc == 0) {
        const double max_seconds = getenv_seconds("PULSAR_CUDA_TOPK_REGRESSION_SEC", 2.0);
        fprintf(stderr, "cuda-regression: top-k n_comp=%u n_tokens=%u elapsed=%.3fs\n",
                n_comp, n_tokens, elapsed);
        if (elapsed > max_seconds) {
            fprintf(stderr, "top-k regression: %.3fs exceeds %.3fs\n", elapsed, max_seconds);
            rc = 1;
        }
    }

cleanup:
    pulsar_gpu_tensor_free(selected);
    pulsar_gpu_tensor_free(scores);
    free(selected_host);
    free(scores_host);
    return rc;
}

static int check_decode_attention_overflow_path(void) {
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_raw = 128;
    const uint32_t n_comp = 8100;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)n_raw * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;

    float *sinks = (float *)calloc(n_head, sizeof(float));
    float *q_host = (float *)calloc((size_t)q_count, sizeof(float));
    float *raw_host = (float *)calloc((size_t)raw_count, sizeof(float));
    float *comp_host = (float *)calloc((size_t)comp_count, sizeof(float));
    float *heads_host = (float *)calloc((size_t)q_count, sizeof(float));
    if (!sinks || !q_host || !raw_host || !comp_host || !heads_host) return 1;

    for (uint32_t c = 0; c < n_comp; c++) {
        comp_host[(uint64_t)c * head_dim] = 1.0f;
    }

    /* Stored heads width, not f32 (L033) -- this test was the fifth raw-byte
     * reader of a narrowed buffer found in one day; see heads_store4. */
    pulsar_gpu_tensor *heads = pulsar_gpu_tensor_alloc_elt(q_count, PULSAR_HEADS_ELT_SIZE);
    /* Attention Q: stored element size, not f32 (L045). */
    pulsar_gpu_tensor *q = pulsar_gpu_tensor_alloc_elt(q_count, PULSAR_Q_ELT_SIZE);
    pulsar_gpu_tensor *raw_f32 = pulsar_gpu_tensor_alloc(raw_count * sizeof(float));
    pulsar_gpu_tensor *raw = pulsar_gpu_tensor_alloc((uint64_t)n_raw * SMOKE_ATTN_ROWBYTES(head_dim));
    pulsar_gpu_tensor *comp_f32 = pulsar_gpu_tensor_alloc(comp_count * sizeof(float));
    /* Packed comp rows, via the shipping encoder -- see the note in the
     * multibank case; an f32 comp buffer is read at the packed stride now. */
    pulsar_gpu_tensor *comp = pulsar_gpu_tensor_alloc((uint64_t)n_comp * SMOKE_ATTN_ROWBYTES(head_dim));
    int rc = 1;
    if (heads && q && raw && raw_f32 && comp && comp_f32 &&
        pulsar_gpu_tensor_write_q_f32(q, 0, q_host, q_count) &&
        pulsar_gpu_tensor_write(raw_f32, 0, raw_host, raw_count * sizeof(float)) &&
        pulsar_gpu_store_raw_kv_batch_tensor(raw, raw_f32, n_raw, 0, n_raw,
                                            head_dim, NULL, NULL, 1) &&
        pulsar_gpu_tensor_write(comp_f32, 0, comp_host, comp_count * sizeof(float)) &&
        pulsar_gpu_attn_pack_quantize_store_tensor(comp_f32, comp, 0, n_comp,
                                                   head_dim, SMOKE_ATTN_NROT,
                                                   /*keep_f32=*/true) &&
        pulsar_gpu_attention_decode_heads_tensor(heads,
                                              sinks,
                                              n_head * sizeof(float),
                                              0,
                                              q,
                                              raw,
                                              n_raw,
                                              n_raw,
                                              0,
                                              comp,
                                              n_comp,
                                              n_head,
                                              head_dim) &&
        pulsar_gpu_synchronize() &&
        pulsar_gpu_tensor_read_f32(heads, 0, heads_host, q_count)) {
        rc = 0;
        for (uint32_t h = 0; h < n_head; h++) {
            const float v = heads_host[(uint64_t)h * head_dim];
            if (v < 0.90f) {
                fprintf(stderr, "attention fallback ignored compressed rows for head=%u value=%f\n",
                        h, (double)v);
                rc = 1;
            }
        }
    }

    pulsar_gpu_tensor_free(comp);
    pulsar_gpu_tensor_free(comp_f32);
    pulsar_gpu_tensor_free(raw_f32);
    pulsar_gpu_tensor_free(raw);
    pulsar_gpu_tensor_free(q);
    pulsar_gpu_tensor_free(heads);
    free(heads_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
    free(sinks);
    return rc;
}

static int check_dspark_non_causal_attention(void) {
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_tokens = 5;
    const uint32_t n_raw = 5;
    const uint32_t raw_cap = 5;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)n_raw * head_dim;
    const uint64_t heads_count = (uint64_t)n_tokens * n_head * head_dim;

    float *sinks = (float *)calloc(n_head, sizeof(float));
    float *kvrow = (float *)calloc((size_t)head_dim, sizeof(float));
    float *q_row = (float *)calloc((size_t)n_head * head_dim, sizeof(float));
    float *raw_host = (float *)calloc((size_t)raw_count, sizeof(float));
    float *q_host = (float *)calloc((size_t)q_count * n_tokens, sizeof(float));
    float *heads_causal = (float *)calloc((size_t)heads_count, sizeof(float));
    float *heads_non_causal = (float *)calloc((size_t)heads_count, sizeof(float));
    if (!sinks || !kvrow || !q_row || !raw_host || !q_host || !heads_causal || !heads_non_causal) return 1;

    for (uint32_t d = 0; d < head_dim; d++) {
        kvrow[d] = (float)(d % 16) * 0.1f;
    }
    for (uint32_t d = 0; d < n_head * head_dim; d++) {
        q_row[d] = (float)(d % 8) * 0.2f;
    }
    for (uint32_t t = 0; t < n_raw; t++) {
        memcpy(raw_host + (uint64_t)t * head_dim, kvrow, head_dim * sizeof(float));
    }
    for (uint32_t t = 0; t < n_tokens; t++) {
        memcpy(q_host + (uint64_t)(t * n_head) * head_dim, q_row, (uint64_t)n_head * head_dim * sizeof(float));
    }

    pulsar_gpu_tensor *heads_c = pulsar_gpu_tensor_alloc_elt(heads_count, PULSAR_HEADS_ELT_SIZE);
    pulsar_gpu_tensor *heads_nc = pulsar_gpu_tensor_alloc_elt(heads_count, PULSAR_HEADS_ELT_SIZE);
    /* Attention Q: stored element size, not f32 (L045). */
    pulsar_gpu_tensor *q = pulsar_gpu_tensor_alloc_elt((uint64_t)n_tokens * q_count, PULSAR_Q_ELT_SIZE);
    pulsar_gpu_tensor *raw_f32 = pulsar_gpu_tensor_alloc(raw_count * sizeof(float));
    pulsar_gpu_tensor *raw = pulsar_gpu_tensor_alloc((uint64_t)raw_cap * SMOKE_ATTN_ROWBYTES(head_dim));
    int rc = 1;
    if (heads_c && heads_nc && q && raw && raw_f32 &&
        pulsar_gpu_tensor_write_q_f32(q, 0, q_host, (uint64_t)n_tokens * q_count) &&
        pulsar_gpu_tensor_write(raw_f32, 0, raw_host, raw_count * sizeof(float)) &&
        pulsar_gpu_store_raw_kv_batch_tensor(raw, raw_f32, raw_cap, 0, n_raw,
                                            head_dim, NULL, NULL, 1)) {

        int ok_c = pulsar_gpu_attention_decode_raw_batch_heads_tensor(heads_c,
                                                                    sinks,
                                                                    n_head * sizeof(float),
                                                                    0,
                                                                    q,
                                                                    raw,
                                                                    n_tokens, 0,
                                                                    n_raw, raw_cap, 0,
                                                                    0, n_head, head_dim, 0,
                                                                    NULL, NULL, 0, 1,
                                          NULL /* q pre-normed */);
        int ok_nc = pulsar_gpu_attention_decode_raw_batch_heads_tensor(heads_nc,
                                                                     sinks,
                                                                     n_head * sizeof(float),
                                                                     0,
                                                                     q,
                                                                     raw,
                                                                     n_tokens, 0,
                                                                     n_raw, raw_cap, 0,
                                                                     0, n_head, head_dim, 1,
                                                                     NULL, NULL, 0, 1,
                                          NULL /* q pre-normed */);
        if (ok_c && ok_nc && pulsar_gpu_synchronize() &&
            pulsar_gpu_tensor_read_f32(heads_c, 0, heads_causal, heads_count) &&
            pulsar_gpu_tensor_read_f32(heads_nc, 0, heads_non_causal, heads_count)) {

            int causal_all_equal = 1;
            for (uint32_t t = 1; t < n_tokens; t++) {
                if (memcmp(heads_causal, heads_causal + (uint64_t)t * n_head * head_dim,
                           n_head * head_dim * sizeof(float)) != 0) {
                    causal_all_equal = 0;
                    break;
                }
            }

            int non_causal_all_equal = 1;
            for (uint32_t t = 1; t < n_tokens; t++) {
                if (memcmp(heads_non_causal, heads_non_causal + (uint64_t)t * n_head * head_dim,
                           n_head * head_dim * sizeof(float)) != 0) {
                    non_causal_all_equal = 0;
                    break;
                }
            }

            if (causal_all_equal) {
                fprintf(stderr, "dspark_attn: causal outputs unexpectedly equal\n");
                rc = 1;
            } else if (!non_causal_all_equal) {
                fprintf(stderr, "dspark_attn: non-causal outputs not all equal\n");
                rc = 1;
            } else {
                rc = 0;
            }
        }
    }

    pulsar_gpu_tensor_free(raw_f32);
    pulsar_gpu_tensor_free(raw);
    pulsar_gpu_tensor_free(q);
    pulsar_gpu_tensor_free(heads_nc);
    pulsar_gpu_tensor_free(heads_c);
    free(heads_non_causal);
    free(heads_causal);
    free(q_host);
    free(raw_host);
    free(q_row);
    free(kvrow);
    free(sinks);
    return rc;
}

static int check_dspark_markov_head(void) {
    const uint32_t vocab_size = 4096;
    const uint32_t embed_dim = 256;
    const uint32_t n_draft = 5;
    int32_t prev_tokens[5] = {42, 100, 500, 1200, 3000};

    /* The markov head reads its weights from the model map: stage w1|w2 in one
     * PAGE-ALIGNED buffer so the runtime's per-range cudaHostRegister of the
     * two offsets never overlaps a page (w_bytes is a page multiple here).
     * The buffer is intentionally never freed: its pages stay host-registered
     * until pulsar_gpu_cleanup, and munmap of registered pages is UB. */
    const uint64_t w_bytes = (uint64_t)vocab_size * embed_dim * sizeof(float);
    float *map_host = NULL;
    if (posix_memalign((void **)&map_host, 4096, (size_t)(2 * w_bytes)) != 0) return 1;
    memset(map_host, 0, (size_t)(2 * w_bytes));
    float *base_host = (float *)calloc((size_t)vocab_size, sizeof(float));
    if (!map_host || !base_host) return 1;
    float *w1_host = map_host;
    float *w2_host = map_host + (uint64_t)vocab_size * embed_dim;

    for (uint32_t v = 0; v < vocab_size; v++) {
        for (uint32_t i = 0; i < embed_dim; i++) {
            w1_host[(uint64_t)v * embed_dim + i] = (float)((v * 7 + i * 13) % 100) * 0.01f;
            w2_host[(uint64_t)v * embed_dim + i] = (float)((v * 3 + i * 11) % 50) * 0.02f;
        }
        base_host[v] = (float)(v % 200) * 0.01f;
    }

    pulsar_gpu_tensor *base = pulsar_gpu_tensor_alloc((uint64_t)vocab_size * sizeof(float));
    pulsar_gpu_tensor *ref_logits = pulsar_gpu_tensor_alloc((uint64_t)vocab_size * sizeof(float));
    int rc = 1;
    if (base && ref_logits &&
        pulsar_gpu_tensor_write(base, 0, base_host, (uint64_t)vocab_size * sizeof(float))) {

        int32_t id = prev_tokens[0];
        for (uint32_t step = 0; step < n_draft; step++) {
            int32_t gpu_id = 0;
            if (!pulsar_gpu_dspark_markov_step_model(ref_logits, &gpu_id, NULL, base,
                                                  map_host, 2 * w_bytes,
                                                  0, w_bytes,
                                                  id, vocab_size, embed_dim,
                                                  /*w1_bf16=*/0, /*w2_bf16=*/0)) {
                rc = 1;
                goto cleanup;
            }

            const float *embed = w1_host + (uint64_t)id * embed_dim;
            float cpu_best = -1e30f;
            int32_t cpu_id = 0;
            for (uint32_t v = 0; v < vocab_size; v++) {
                float dot = 0.0f;
                const float *w2r = w2_host + (uint64_t)v * embed_dim;
                for (uint32_t i = 0; i < embed_dim; i++)
                    dot += w2r[i] * embed[i];
                float val = base_host[v] + dot;
                if (val > cpu_best) { cpu_best = val; cpu_id = (int32_t)v; }
            }

            if (gpu_id != cpu_id) {
                fprintf(stderr, "markov step %u: GPU=%d CPU=%d\n", step, gpu_id, cpu_id);
                rc = 1;
                goto cleanup;
            }
            id = gpu_id;
        }
        rc = 0;
    }

cleanup:
    pulsar_gpu_tensor_free(ref_logits);
    pulsar_gpu_tensor_free(base);
    free(base_host);
    /* map_host intentionally leaked (host-registered; see above). */
    return rc;
}

static int check_dspark_confidence_head(void) {
    const uint32_t n_positions = 3;
    const uint32_t hidden_dim = 32;
    const uint32_t embed_dim = 8;
    const uint32_t total_dim = hidden_dim + embed_dim;
    const uint32_t vocab_size = 16;
    const int32_t token_ids_host[3] = {3, 7, 12};

    /* The confidence head gathers its embedding row from markov_w1 by token id
     * and reads w1/proj from the model map.  Page-align the buffer and place
     * proj on its own page so the runtime's per-range cudaHostRegister calls
     * never overlap; intentionally never freed (registered pages, see the
     * markov check). */
    const uint64_t proj_offset = 4096;   /* w1 (512 B) fits below it */
    float *map_host = NULL;
    if (posix_memalign((void **)&map_host, 4096,
                       (size_t)(proj_offset + total_dim * sizeof(float))) != 0) return 1;
    memset(map_host, 0, (size_t)(proj_offset + total_dim * sizeof(float)));
    float *hidden_host = (float *)calloc((size_t)n_positions * hidden_dim, sizeof(float));
    float scores_host[3];
    if (!map_host || !hidden_host) return 1;
    float *w1_host = map_host;
    float *proj_host = (float *)((char *)map_host + proj_offset);

    for (uint32_t v = 0; v < vocab_size; v++)
        for (uint32_t i = 0; i < embed_dim; i++)
            w1_host[(uint64_t)v * embed_dim + i] = (float)((v + i) % 3) * 0.3f;
    for (uint32_t i = 0; i < total_dim; i++)
        proj_host[i] = (float)((i % 7) + 1) * 0.1f;
    for (uint32_t p = 0; p < n_positions; p++) {
        for (uint32_t i = 0; i < hidden_dim; i++)
            hidden_host[(uint64_t)p * hidden_dim + i] = (float)((p + i) % 5) * 0.5f;
    }

    pulsar_gpu_tensor *scores = pulsar_gpu_tensor_alloc((uint64_t)n_positions * sizeof(float));
    pulsar_gpu_tensor *hidden = pulsar_gpu_tensor_alloc((uint64_t)n_positions * hidden_dim * sizeof(float));
    pulsar_gpu_tensor *token_ids = pulsar_gpu_tensor_alloc((uint64_t)n_positions * sizeof(int32_t));
    int rc = 1;
    if (scores && hidden && token_ids &&
        pulsar_gpu_tensor_write(hidden, 0, hidden_host, (uint64_t)n_positions * hidden_dim * sizeof(float)) &&
        pulsar_gpu_tensor_write(token_ids, 0, token_ids_host, (uint64_t)n_positions * sizeof(int32_t)) &&
        pulsar_gpu_dspark_confidence_score_model(scores, hidden, token_ids,
                                              map_host,
                                              proj_offset + total_dim * sizeof(float),
                                              0, proj_offset,
                                              n_positions, hidden_dim, embed_dim,
                                              vocab_size,
                                              /*w1_bf16=*/0, /*proj_bf16=*/0) &&
        pulsar_gpu_synchronize() &&
        pulsar_gpu_tensor_read(scores, 0, scores_host, (uint64_t)n_positions * sizeof(float))) {
        rc = 0;
        for (uint32_t p = 0; p < n_positions; p++) {
            const float *emb = w1_host + (uint64_t)token_ids_host[p] * embed_dim;
            float dot = 0.0f;
            for (uint32_t i = 0; i < hidden_dim; i++)
                dot += hidden_host[(uint64_t)p * hidden_dim + i] * proj_host[i];
            for (uint32_t i = 0; i < embed_dim; i++)
                dot += emb[i] * proj_host[hidden_dim + i];
            float expected = 1.0f / (1.0f + expf(-dot));
            if (fabsf(scores_host[p] - expected) > 1e-5f) {
                fprintf(stderr, "confidence pos %u: GPU=%.6f CPU=%.6f\n",
                        p, (double)scores_host[p], (double)expected);
                rc = 1;
            }
        }
    }

    pulsar_gpu_tensor_free(token_ids);
    pulsar_gpu_tensor_free(hidden);
    pulsar_gpu_tensor_free(scores);
    free(hidden_host);
    /* map_host intentionally leaked (host-registered; see the markov check). */
    return rc;
}


/* ---------------------------------------------------------------------------
 * Multi-bank descriptor attention smoke (Tier-2 step 2).
 *
 * Two banks of a raw-ring + compressed-cache pool are filled with different
 * pseudorandom sequences, then the descriptor-aware decode attention runs
 * rows from both banks (mixed seq_ids, per-row positions incl. a wrapped
 * ring and an EMIT-boundary row, qpos ≡ ratio-1 mod ratio) in ONE launch.
 * Every row must be BYTE-identical to the same computation run
 * single-session (scalar mode against that bank's slab view) with the
 * ENGINE-TRUE scalar n_comp — the engine emits a step's compressed row
 * BEFORE attention, so at position qpos the live engine passes
 * (qpos+1)/ratio, and the banked per-row derivation must reproduce exactly
 * that (a floor(qpos/ratio) derivation fails the emit-boundary rows below).
 * Rows must also be independent of batch composition (row permutation).
 * Attention here is deterministic per row: one block per (row, head), so
 * per-row reduction order cannot depend on batchmates.
 * ------------------------------------------------------------------------- */

static uint32_t mb_rng_state = 0x12345u;
static float mb_rand(void) {
    mb_rng_state = mb_rng_state * 1664525u + 1013904223u;
    return ((float)(mb_rng_state >> 9) / 8388608.0f) - 1.0f;   /* [-1, 1) */
}

typedef struct {
    uint32_t bank;
    uint32_t qpos;
    uint32_t ref_n_comp;   /* scalar n_comp for the single-session reference:
                            * ENGINE-TRUE frontier at qpos, i.e. (qpos+1)/ratio
                            * (emit-before-attention), capped by the bank's
                            * comp capacity.  Exception: the heads8-online case
                            * passes the cross-bank superset to pin kernel
                            * selection (see the comment there). */
} mb_row;

/* One descriptor launch over `rows`, then per-row single-session reference
 * launches against bank views; byte-compare.  indexed != 0 exercises the
 * indexed (top-k) entry instead of the mixed entry. */
/* Pack n_rows x 128 f32 values into MXKV FP4 rows (64 nibble bytes + 4 E8M0
 * scale bytes) -- the indexer Q operand format since L090.4.  The scorer
 * comparisons below are device-vs-device over the SAME packed tensor, so only
 * determinism matters here, not encode tie-breaking. */
static void mb_pack_q_rows(const float *src, uint8_t *dst, uint64_t n_rows) {
    static const float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    for (uint64_t r = 0; r < n_rows; r++) {
        const float *x = src + r * 128u;
        uint8_t *out = dst + r * 68u;
        for (uint32_t blk = 0; blk < 4u; blk++) {
            float amax = 7.052966104933725e-38f;
            for (uint32_t i = 0; i < 32u; i++)
                amax = fmaxf(amax, fabsf(x[blk * 32u + i]));
            int e8 = (int)ceilf(log2f(amax / 6.0f)) + 127;
            e8 = e8 < 0 ? 0 : (e8 > 254 ? 254 : e8);
            const float scale = ldexpf(1.0f, e8 - 127);
            out[64u + blk] = (uint8_t)e8;
            for (uint32_t i = 0; i < 32u; i += 2u) {
                uint8_t n2[2];
                for (uint32_t k = 0; k < 2u; k++) {
                    float v = x[blk * 32u + i + k] / scale;
                    const uint8_t sign = v < 0.f ? 8u : 0u;
                    v = fminf(6.0f, fabsf(v));
                    uint8_t best = 0; float bd = 1e30f;
                    for (uint8_t c = 0; c < 8u; c++) {
                        const float d = fabsf(v - kE2M1[c]);
                        if (d < bd) { bd = d; best = c; }
                    }
                    n2[k] = (uint8_t)(best | sign);
                }
                out[(blk * 32u + i) >> 1u] = (uint8_t)(n2[0] | (n2[1] << 4));
            }
        }
    }
}

static int mb_run_case(const char *label,
                       const mb_row *rows, uint32_t n_rows,
                       pulsar_gpu_tensor *raw_slab, uint32_t raw_cap,
                       pulsar_gpu_tensor *comp_slab, uint32_t comp_cap,
                       uint32_t n_comp_superset, uint32_t window,
                       uint32_t ratio, uint32_t n_banks,
                       const float *sinks, uint32_t n_head, uint32_t head_dim,
                       int indexed, const int32_t *topk_host, uint32_t top_k) {
    const uint64_t row_f32 = (uint64_t)n_head * head_dim;
    const uint64_t q_count = (uint64_t)n_rows * row_f32;
    float *q_host = (float *)malloc(q_count * sizeof(float));
    float *out_batch = (float *)malloc(q_count * sizeof(float));
    float *out_ref = (float *)malloc(row_f32 * sizeof(float));
    int32_t *pos_host = (int32_t *)malloc(n_rows * sizeof(int32_t));
    int32_t *sid_host = (int32_t *)malloc(n_rows * sizeof(int32_t));
    if (!q_host || !out_batch || !out_ref || !pos_host || !sid_host) {
        free(sid_host); free(pos_host); free(out_ref); free(out_batch); free(q_host);
        return 1;
    }
    mb_rng_state = 0xbeef1234u;
    for (uint64_t i = 0; i < q_count; i++) q_host[i] = mb_rand() * 0.25f;
    for (uint32_t r = 0; r < n_rows; r++) {
        pos_host[r] = (int32_t)rows[r].qpos;
        sid_host[r] = (int32_t)rows[r].bank;
    }

    /* Attention Q: stored element size, not f32 (L045).  heads stays f32. */
    pulsar_gpu_tensor *q = pulsar_gpu_tensor_alloc_elt(q_count, PULSAR_Q_ELT_SIZE);
    /* Stored heads width, not f32 (L033) -- this test was the fifth raw-byte
     * reader of a narrowed buffer found in one day; see heads_store4. */
    pulsar_gpu_tensor *heads = pulsar_gpu_tensor_alloc_elt(q_count, PULSAR_HEADS_ELT_SIZE);
    pulsar_gpu_tensor *positions = pulsar_gpu_tensor_alloc(n_rows * sizeof(int32_t));
    pulsar_gpu_tensor *seq_id = pulsar_gpu_tensor_alloc(n_rows * sizeof(int32_t));
    pulsar_gpu_tensor *topk = NULL;
    int rc = 1;
    if (!q || !heads || !positions || !seq_id ||
        !pulsar_gpu_tensor_write_q_f32(q, 0, q_host, q_count) ||
        !pulsar_gpu_tensor_write(positions, 0, pos_host, n_rows * sizeof(int32_t)) ||
        !pulsar_gpu_tensor_write(seq_id, 0, sid_host, n_rows * sizeof(int32_t))) goto done;
    if (indexed) {
        topk = pulsar_gpu_tensor_alloc((uint64_t)n_rows * top_k * sizeof(int32_t));
        if (!topk || !pulsar_gpu_tensor_write(topk, 0, topk_host,
                                           (uint64_t)n_rows * top_k * sizeof(int32_t))) goto done;
    }

    /* Descriptor launch: all rows, mixed banks, one call. */
    {
        int ok;
        if (indexed) {
            ok = pulsar_gpu_attention_indexed_mixed_batch_heads_tensor(
                    heads, sinks, (uint64_t)n_head * sizeof(float), 0,
                    q, raw_slab, comp_slab, topk,
                    n_rows, 0, window, raw_cap, 0,
                    n_comp_superset, top_k, window, ratio, n_head, head_dim,
                    positions, seq_id, NULL, comp_cap, n_banks,
                                          NULL /* q pre-normed */);
        } else {
            ok = pulsar_gpu_attention_decode_mixed_batch_heads_tensor(
                    heads, sinks, (uint64_t)n_head * sizeof(float), 0,
                    q, raw_slab, n_comp_superset ? comp_slab : NULL,
                    n_rows, 0, window, raw_cap, 0,
                    n_comp_superset, window, ratio, n_head, head_dim, 0,
                    positions, seq_id, NULL, comp_cap, n_banks,
                                          NULL /* q pre-normed */);
        }
        if (!ok || !pulsar_gpu_synchronize() ||
            !pulsar_gpu_tensor_read_f32(heads, 0, out_batch, q_count)) {
            fprintf(stderr, "multibank %s: descriptor launch failed\n", label);
            goto done;
        }
    }

    /* Per-row single-session references against the bank's slab view.
     *
     * The reference is a classic scalar-mode BATCH of the 3 consecutive
     * positions ending at the row's qpos (NULL descriptors — the unchanged
     * single-session verify-batch shape), and we compare the last row.  A
     * 1-row reference would be wrong-by-construction: the generic kernel's
     * score pass legitimately switches to a cheaper loop when the LAUNCH is
     * single-token (n_tokens == 1), and that loop's float accumulation order
     * differs.  Within multi-row launches the per-row math depends only on
     * (bank, qpos) — which is exactly the batch-composition independence this
     * smoke asserts byte-for-byte. */
    for (uint32_t r = 0; r < n_rows; r++) {
        const mb_row *row = &rows[r];
        const uint32_t ref_rows = 3;
        /* raw is PULSAR_ATTN_PACK now, comp is still the f32 slab this test hands
         * over with comp_kv_pack=0 -- so the two bank strides are NOT the same
         * arithmetic any more, which is exactly what this line got wrong. */
        const uint64_t raw_bank_bytes = (uint64_t)raw_cap * SMOKE_ATTN_ROWBYTES(head_dim);
        const uint64_t comp_bank_bytes = (uint64_t)comp_cap * SMOKE_ATTN_ROWBYTES(head_dim);
        pulsar_gpu_tensor *raw_view = pulsar_gpu_tensor_view(
                raw_slab, (uint64_t)row->bank * raw_bank_bytes, raw_bank_bytes);
        pulsar_gpu_tensor *comp_view = comp_slab
            ? pulsar_gpu_tensor_view(comp_slab, (uint64_t)row->bank * comp_bank_bytes,
                                  comp_bank_bytes)
            : NULL;
        /* Attention Q, same as the batch side: element size, not f32. */
        pulsar_gpu_tensor *q_ref = pulsar_gpu_tensor_alloc_elt((uint64_t)ref_rows * row_f32, PULSAR_Q_ELT_SIZE);
        pulsar_gpu_tensor *h_ref = pulsar_gpu_tensor_alloc((uint64_t)ref_rows * row_f32 * sizeof(float));
        pulsar_gpu_tensor *tk_ref = indexed
            ? pulsar_gpu_tensor_alloc((uint64_t)ref_rows * top_k * sizeof(int32_t))
            : NULL;
        float *q_ref_host = (float *)malloc((uint64_t)ref_rows * row_f32 * sizeof(float));
        int32_t *tk_ref_host = indexed
            ? (int32_t *)malloc((uint64_t)ref_rows * top_k * sizeof(int32_t))
            : NULL;
        int ok = raw_view && q_ref && h_ref && q_ref_host &&
                 (!comp_slab || comp_view) &&
                 (!indexed || (tk_ref && tk_ref_host));
        if (ok) {
            /* rows 0..1 are position-fillers; row 2 is the row under test. */
            for (uint64_t i = 0; i < (uint64_t)(ref_rows - 1) * row_f32; i++)
                q_ref_host[i] = 0.01f;
            memcpy(q_ref_host + (uint64_t)(ref_rows - 1) * row_f32,
                   q_host + (uint64_t)r * row_f32, row_f32 * sizeof(float));
            ok = pulsar_gpu_tensor_write_q_f32(q_ref, 0, q_ref_host,
                                           (uint64_t)ref_rows * row_f32);
            if (ok && indexed) {
                for (uint32_t rr = 0; rr < ref_rows; rr++)
                    memcpy(tk_ref_host + (uint64_t)rr * top_k,
                           topk_host + (uint64_t)r * top_k, top_k * sizeof(int32_t));
                ok = pulsar_gpu_tensor_write(tk_ref, 0, tk_ref_host,
                                          (uint64_t)ref_rows * top_k * sizeof(int32_t));
            }
        }
        const uint32_t n_raw = (row->qpos + 1u < window) ? row->qpos + 1u : window;
        const uint32_t raw_start = (row->qpos + 1u - n_raw) % raw_cap;
        const uint32_t pos0 = row->qpos - (ref_rows - 1u);
        if (ok) {
            if (indexed) {
                ok = pulsar_gpu_attention_indexed_mixed_batch_heads_tensor(
                        h_ref, sinks, (uint64_t)n_head * sizeof(float), 0,
                        q_ref, raw_view, comp_view, tk_ref,
                        ref_rows, pos0, n_raw, raw_cap, raw_start,
                        row->ref_n_comp, top_k, window, ratio, n_head, head_dim,
                        NULL, NULL, NULL, 0, 1,
                                          NULL /* q pre-normed */);
            } else {
                ok = pulsar_gpu_attention_decode_mixed_batch_heads_tensor(
                        h_ref, sinks, (uint64_t)n_head * sizeof(float), 0,
                        q_ref, raw_view, row->ref_n_comp ? comp_view : NULL,
                        ref_rows, pos0, n_raw, raw_cap, raw_start,
                        row->ref_n_comp, window, ratio, n_head, head_dim, 0,
                        NULL, NULL, NULL, 0, 1,
                                          NULL /* q pre-normed */);
            }
        }
        ok = ok && pulsar_gpu_synchronize() &&
             pulsar_gpu_tensor_read(h_ref, (uint64_t)(ref_rows - 1) * row_f32 * sizeof(float),
                                 out_ref, row_f32 * sizeof(float));
        free(tk_ref_host);
        free(q_ref_host);
        pulsar_gpu_tensor_free(tk_ref);
        pulsar_gpu_tensor_free(h_ref);
        pulsar_gpu_tensor_free(q_ref);
        pulsar_gpu_tensor_free(comp_view);
        pulsar_gpu_tensor_free(raw_view);
        if (!ok) {
            fprintf(stderr, "multibank %s: reference launch failed (row %u)\n", label, r);
            goto done;
        }
        if (memcmp(out_ref, out_batch + (uint64_t)r * row_f32,
                   row_f32 * sizeof(float)) != 0) {
            fprintf(stderr, "multibank %s: row %u (bank %u pos %u) != single-session\n",
                    label, r, row->bank, row->qpos);
            goto done;
        }
    }
    printf("  multibank %s: %u rows across %u banks -> byte-identical\n",
           label, n_rows, n_banks);
    rc = 0;

done:
    pulsar_gpu_tensor_free(topk);
    pulsar_gpu_tensor_free(seq_id);
    pulsar_gpu_tensor_free(positions);
    pulsar_gpu_tensor_free(heads);
    pulsar_gpu_tensor_free(q);
    free(sid_host); free(pos_host); free(out_ref); free(out_batch); free(q_host);
    return rc;
}

static int check_multibank_decode_attention(void) {
    const uint32_t n_head = 8, head_dim = 512;
    const uint32_t raw_cap = 64, window = 32, ratio = 4, n_banks = 2;
    const uint32_t comp_cap = 32;               /* per-bank comp-row stride */
    /* static: the runtime host-registers the model-map page (never freed). */
    static float sinks[8];
    for (uint32_t h = 0; h < n_head; h++) sinks[h] = 0.1f * (float)h;

    const uint64_t raw_count = (uint64_t)n_banks * raw_cap * head_dim;
    const uint64_t comp_count = (uint64_t)n_banks * comp_cap * head_dim;
    float *raw_host = (float *)malloc(raw_count * sizeof(float));
    float *comp_host = (float *)malloc(comp_count * sizeof(float));
    if (!raw_host || !comp_host) {
        free(comp_host); free(raw_host);
        return 1;
    }
    mb_rng_state = 0xace1u;
    for (uint64_t i = 0; i < raw_count; i++) raw_host[i] = mb_rand();
    for (uint64_t i = 0; i < comp_count; i++) comp_host[i] = mb_rand();

    const uint64_t raw_row_b = SMOKE_ATTN_ROWBYTES(head_dim);
    pulsar_gpu_tensor *raw_f32 = pulsar_gpu_tensor_alloc(raw_count * sizeof(float));
    pulsar_gpu_tensor *raw_slab = pulsar_gpu_tensor_alloc((uint64_t)n_banks * raw_cap * raw_row_b);
    /* Comp rows are ATTN_PACK.  They were f32 here until 2026-08-18, when the
     * comp format parameter was removed from the kernels -- an f32 comp slab now
     * gets read at the packed stride.  Written through the SHIPPING encoder
     * rather than hand-packed, so the test cannot disagree with the format. */
    pulsar_gpu_tensor *comp_f32 = pulsar_gpu_tensor_alloc(comp_count * sizeof(float));
    pulsar_gpu_tensor *comp_slab =
        pulsar_gpu_tensor_alloc((uint64_t)n_banks * comp_cap * SMOKE_ATTN_ROWBYTES(head_dim));
    int rc = 1;
    if (!raw_slab || !raw_f32 || !comp_slab || !comp_f32 ||
        !pulsar_gpu_tensor_write(raw_f32, 0, raw_host, raw_count * sizeof(float)) ||
        !pulsar_gpu_tensor_write(comp_f32, 0, comp_host, comp_count * sizeof(float)) ||
        !pulsar_gpu_attn_pack_quantize_store_tensor(comp_f32, comp_slab, 0,
                                                   (uint32_t)(n_banks * comp_cap),
                                                   head_dim, SMOKE_ATTN_NROT,
                                                   /*keep_f32=*/true)) goto done;
    for (uint32_t b = 0; b < n_banks; b++) {
        pulsar_gpu_tensor *bv = pulsar_gpu_tensor_view(raw_slab, (uint64_t)b * raw_cap * raw_row_b,
                                                       (uint64_t)raw_cap * raw_row_b);
        pulsar_gpu_tensor *fv = pulsar_gpu_tensor_view(raw_f32,
                                                       (uint64_t)b * raw_cap * head_dim * sizeof(float),
                                                       (uint64_t)raw_cap * head_dim * sizeof(float));
        int okb = bv && fv && pulsar_gpu_store_raw_kv_batch_tensor(bv, fv, raw_cap, 0, raw_cap,
                                                                  head_dim, NULL, NULL, 1);
        pulsar_gpu_tensor_free(fv); pulsar_gpu_tensor_free(bv);
        if (!okb) goto done;
    }

    {
        /* Generic mixed kernel: bank 0 with a WRAPPED ring (qpos 100 > raw_cap
         * 64), bank 1 at qpos 39 — an EMIT boundary (39 ≡ ratio-1 mod 4) where
         * the engine-true frontier is (39+1)/4 = 10 while a floor(39/4)
         * derivation would see only 9; this row FAILS under the old floor rule
         * (regression teeth for the emit-step divergence bug) — and bank 1
         * again at 37 (non-emit, frontier 9).  References use the engine-true
         * (qpos+1)/4 as scalar n_comp; the batch passes the cross-bank
         * superset 25. */
        const mb_row rows[3] = { {0, 100, 25}, {1, 39, 10}, {1, 37, 9} };
        if (mb_run_case("mixed-generic", rows, 3, raw_slab, raw_cap,
                        comp_slab, comp_cap, 25, window, ratio, n_banks,
                        sinks, n_head, head_dim, 0, NULL, 0) != 0) goto done;
        /* Batch-composition independence: permuted rows, same per-row bytes
         * (mb_run_case re-verifies each row against its reference). */
        const mb_row perm[3] = { {1, 37, 9}, {0, 100, 25}, {1, 39, 10} };
        if (mb_run_case("mixed-generic-permuted", perm, 3, raw_slab, raw_cap,
                        comp_slab, comp_cap, 25, window, ratio, n_banks,
                        sinks, n_head, head_dim, 0, NULL, 0) != 0) goto done;
        /* Raw-only path (n_comp = 0): both banks, one wrapped. */
        const mb_row raw_rows[2] = { {0, 100, 0}, {1, 17, 0} };
        if (mb_run_case("raw-only", raw_rows, 2, raw_slab, raw_cap,
                        NULL, comp_cap, 0, window, ratio, n_banks,
                        sinks, n_head, head_dim, 0, NULL, 0) != 0) goto done;
        /* Indexed (top-k) entry: ids >= the row's visible frontier and -1
         * sentinels must be dropped per row (bank 1 at qpos 39 sees the
         * engine-true 10 visible rows, so ids 12/24 are in-superset but
         * beyond ITS frontier).  Row 1's id 9 is the emit-boundary teeth:
         * exactly the row emitted at step 39 — visible under the engine-true
         * (39+1)/4 = 10 rule, dropped by the old floor(39/4) = 9 rule. */
        const uint32_t top_k = 8;
        const int32_t topk_host[3 * 8] = {
             3, 24,  0, 12, -1,  7, 19,  5,
             8,  2, 12,  9, 24,  0,  6,  4,
             1,  8, -1,  5, 24,  3, 12,  2,
        };
        const mb_row idx_rows[3] = { {0, 100, 25}, {1, 39, 10}, {1, 37, 9} };
        /* Banked mode forces the generic indexed kernel (heads8 variants stay
         * single-bank); pin the scalar reference to the same kernel. */
        setenv("PULSAR_CUDA_NO_INDEXED_HEADS8", "1", 1);
        const int idx_rc = mb_run_case("indexed-generic", idx_rows, 3, raw_slab, raw_cap,
                                       comp_slab, comp_cap, 25, window, ratio, n_banks,
                                       sinks, n_head, head_dim, 1, topk_host, top_k);
        unsetenv("PULSAR_CUDA_NO_INDEXED_HEADS8");
        if (idx_rc != 0) goto done;
    }

    /* heads8-online variant: a cross-bank superset n_comp too large for the
     * shared-memory score buffer forces the online kernel for the batch AND
     * for the single-row references.  Bank 0 sits at qpos 45999, an EMIT
     * boundary (45999 ≡ 3 mod 4): its engine-true frontier is 46000/4 =
     * 11500 == the scalar, while floor(45999/4) = 11499 — teeth for the
     * online kernel's per-row rule.  Bank 1 (qpos 20000, frontier 5000)
     * must ALSO get the 11500 scalar in its reference: scalar n_comp picks
     * the kernel (a 5000 scalar fits the score buffer and would take the
     * generic kernel, whose reduction order differs), while the rows a
     * reference actually reads come from the kernel's own (qpos+1)/ratio
     * derivation, which caps at 5000 regardless of the larger scalar. */
    {
        const uint32_t big_comp_cap = 11504;
        const uint32_t big_n_comp = 11500;
        const uint64_t big_count = (uint64_t)n_banks * big_comp_cap * head_dim;
        float *big_host = (float *)malloc(big_count * sizeof(float));
        pulsar_gpu_tensor *big_slab = pulsar_gpu_tensor_alloc(big_count * sizeof(float));
        int big_rc = 1;
        if (big_host && big_slab) {
            mb_rng_state = 0x5eed5u;
            for (uint64_t i = 0; i < big_count; i++) big_host[i] = mb_rand();
            if (pulsar_gpu_tensor_write(big_slab, 0, big_host, big_count * sizeof(float))) {
                const mb_row rows[2] = { {0, 45999, big_n_comp}, {1, 20000, big_n_comp} };
                big_rc = mb_run_case("mixed-online", rows, 2, raw_slab, raw_cap,
                                     big_slab, big_comp_cap, big_n_comp, window,
                                     ratio, n_banks, sinks, n_head, head_dim,
                                     0, NULL, 0);
            }
        }
        pulsar_gpu_tensor_free(big_slab);
        free(big_host);
        if (big_rc != 0) goto done;
    }

    /* Dead-row guard teeth: a row with an out-of-pool bank id (-1 sentinel)
     * must produce zero head outputs (fail-visible) while its batchmate is
     * still byte-identical to single-session. */
    {
        const uint64_t row_f32 = (uint64_t)n_head * head_dim;
        const int32_t pos_host2[2] = {100, 37};
        const int32_t sid_host2[2] = {-1, 1};
        float *q_host2 = (float *)malloc(2 * row_f32 * sizeof(float));
        float *out2 = (float *)malloc(2 * row_f32 * sizeof(float));
        pulsar_gpu_tensor *q2 = pulsar_gpu_tensor_alloc(2 * row_f32 * sizeof(float));
        pulsar_gpu_tensor *h2 = pulsar_gpu_tensor_alloc(2 * row_f32 * sizeof(float));
        pulsar_gpu_tensor *p2 = pulsar_gpu_tensor_alloc(2 * sizeof(int32_t));
        pulsar_gpu_tensor *s2 = pulsar_gpu_tensor_alloc(2 * sizeof(int32_t));
        int dead_rc = 1;
        if (q_host2 && out2 && q2 && h2 && p2 && s2) {
            mb_rng_state = 0xbeef1234u;
            for (uint64_t i = 0; i < 2 * row_f32; i++) q_host2[i] = mb_rand() * 0.25f;
            if (pulsar_gpu_tensor_write(q2, 0, q_host2, 2 * row_f32 * sizeof(float)) &&
                pulsar_gpu_tensor_write(p2, 0, pos_host2, sizeof(pos_host2)) &&
                pulsar_gpu_tensor_write(s2, 0, sid_host2, sizeof(sid_host2)) &&
                pulsar_gpu_attention_decode_mixed_batch_heads_tensor(
                        h2, sinks, (uint64_t)n_head * sizeof(float), 0,
                        q2, raw_slab, comp_slab,
                        2, 0, window, raw_cap, 0,
                        25, window, ratio, n_head, head_dim, 0,
                        p2, s2, NULL, comp_cap, n_banks,
                                          NULL /* q pre-normed */) &&
                pulsar_gpu_synchronize() &&
                pulsar_gpu_tensor_read(h2, 0, out2, 2 * row_f32 * sizeof(float))) {
                dead_rc = 0;
                for (uint64_t i = 0; i < row_f32; i++) {
                    if (out2[i] != 0.0f) {
                        fprintf(stderr, "multibank dead-row: nonzero output at %llu\n",
                                (unsigned long long)i);
                        dead_rc = 1;
                        break;
                    }
                }
                if (dead_rc == 0)
                    printf("  multibank dead-row: out-of-pool seq_id -> zero heads\n");
            }
        }
        pulsar_gpu_tensor_free(s2); pulsar_gpu_tensor_free(p2);
        pulsar_gpu_tensor_free(h2); pulsar_gpu_tensor_free(q2);
        free(out2); free(q_host2);
        if (dead_rc != 0) goto done;
    }
    rc = 0;

done:
    pulsar_gpu_tensor_free(comp_slab);
    pulsar_gpu_tensor_free(comp_f32);
    pulsar_gpu_tensor_free(raw_f32);
    pulsar_gpu_tensor_free(raw_slab);
    free(comp_host);
    free(raw_host);
    return rc;
}

/* ---------------------------------------------------------------------------
 * Multi-bank descriptor indexer smoke (Tier-2 step 3).
 *
 * The banked indexer scan (generic per-(comp,row) kernel and the n_tokens==1
 * direct-one fast tier) must score each row byte-identically to the same
 * (bank, qpos) scored single-session against that bank's slab view, with the
 * per-row ENGINE-TRUE visible frontier (qpos+1)/ratio — emit-boundary rows
 * (qpos ≡ ratio-1 mod ratio) are the regression teeth for the visibility
 * rule, and the -INF tail beyond a shallow bank's frontier must keep top-k
 * selection inside that bank's own rows.  Dead rows (out-of-pool seq_id)
 * must score -INF everywhere (fail-visible).  Covers both the f32 and the
 * MXFP4-packed (68 B/row) cache formats.
 * ------------------------------------------------------------------------- */

static int mb_idx_run_case(const char *label,
                           const mb_row *rows, uint32_t n_rows,
                           pulsar_gpu_tensor *index_slab, uint32_t comp_cap,
                           uint32_t n_comp_superset, uint32_t ratio,
                           uint32_t n_banks, uint32_t n_head, uint32_t head_dim,
                           uint64_t index_bank_bytes, uint32_t top_k) {
    const uint64_t q_row = (uint64_t)n_head * head_dim;
    const float scale = 0.125f;
    float *q_host = (float *)malloc((uint64_t)n_rows * q_row * sizeof(float));
    float *w_host = (float *)malloc((uint64_t)n_rows * n_head * sizeof(float));
    float *out_batch = (float *)malloc((uint64_t)n_rows * n_comp_superset * sizeof(float));
    float *out_ref = (float *)malloc((uint64_t)n_comp_superset * sizeof(float));
    uint32_t *sel_batch = (uint32_t *)malloc((uint64_t)n_rows * top_k * sizeof(uint32_t));
    uint32_t *sel_ref = (uint32_t *)malloc((uint64_t)top_k * sizeof(uint32_t));
    int32_t *pos_host = (int32_t *)malloc(n_rows * sizeof(int32_t));
    int32_t *sid_host = (int32_t *)malloc(n_rows * sizeof(int32_t));
    const uint64_t qp_row = (uint64_t)n_head * 68u;   /* packed rows per token-row */
    uint8_t *qp_host = (uint8_t *)malloc((uint64_t)n_rows * qp_row);
    pulsar_gpu_tensor *q = pulsar_gpu_tensor_alloc((uint64_t)n_rows * qp_row);
    pulsar_gpu_tensor *w = pulsar_gpu_tensor_alloc((uint64_t)n_rows * n_head * sizeof(float));
    pulsar_gpu_tensor *scores = pulsar_gpu_tensor_alloc((uint64_t)n_rows * n_comp_superset * sizeof(float));
    pulsar_gpu_tensor *scores_ref = pulsar_gpu_tensor_alloc((uint64_t)n_comp_superset * sizeof(float));
    pulsar_gpu_tensor *sel = pulsar_gpu_tensor_alloc((uint64_t)n_rows * top_k * sizeof(uint32_t));
    pulsar_gpu_tensor *sel_ref_t = pulsar_gpu_tensor_alloc((uint64_t)top_k * sizeof(uint32_t));
    pulsar_gpu_tensor *positions = pulsar_gpu_tensor_alloc(n_rows * sizeof(int32_t));
    pulsar_gpu_tensor *seq_id = pulsar_gpu_tensor_alloc(n_rows * sizeof(int32_t));
    int rc = 1;
    if (!q_host || !qp_host || !w_host || !out_batch || !out_ref || !sel_batch || !sel_ref ||
        !pos_host || !sid_host || !q || !w || !scores || !scores_ref || !sel ||
        !sel_ref_t || !positions || !seq_id) goto done;
    mb_rng_state = 0x1d0b5u;
    for (uint64_t i = 0; i < (uint64_t)n_rows * q_row; i++) q_host[i] = mb_rand() * 0.5f;
    for (uint64_t i = 0; i < (uint64_t)n_rows * n_head; i++) w_host[i] = mb_rand() * 0.5f + 0.75f;
    for (uint32_t r = 0; r < n_rows; r++) {
        pos_host[r] = (int32_t)rows[r].qpos;
        sid_host[r] = (int32_t)rows[r].bank;
    }
    mb_pack_q_rows(q_host, qp_host, (uint64_t)n_rows * n_head);
    if (!pulsar_gpu_tensor_write(q, 0, qp_host, (uint64_t)n_rows * qp_row) ||
        !pulsar_gpu_tensor_write(w, 0, w_host, (uint64_t)n_rows * n_head * sizeof(float)) ||
        !pulsar_gpu_tensor_write(positions, 0, pos_host, n_rows * sizeof(int32_t)) ||
        !pulsar_gpu_tensor_write(seq_id, 0, sid_host, n_rows * sizeof(int32_t))) goto done;

    /* Banked launch: all rows, mixed banks, one call. */
    if (!pulsar_gpu_indexer_scores_decode_batch_tensor(scores, q, w, index_slab,
                                                    n_comp_superset, n_rows, 0,
                                                    n_head, head_dim, ratio, scale,
                                                    positions, seq_id, NULL, comp_cap, n_banks) ||
        !pulsar_gpu_indexer_topk_tensor(sel, scores, n_comp_superset, n_rows, top_k) ||
        !pulsar_gpu_synchronize() ||
        !pulsar_gpu_tensor_read(scores, 0, out_batch,
                             (uint64_t)n_rows * n_comp_superset * sizeof(float)) ||
        !pulsar_gpu_tensor_read(sel, 0, sel_batch,
                             (uint64_t)n_rows * top_k * sizeof(uint32_t))) {
        fprintf(stderr, "multibank indexer %s: banked launch failed\n", label);
        goto done;
    }

    /* Per-row single-session references against the bank's slab view: the
     * classic scalar path (NULL descriptors) with pos0 = qpos and the SAME
     * superset n_comp (the scalar-path causal clamp is already per-row
     * (pos0+t+1)/ratio, so the -INF tail pattern must match exactly).  A
     * 1-row launch with these head counts stays on the same generic kernel
     * (the direct-one tier requires n_head 64). */
    for (uint32_t r = 0; r < n_rows; r++) {
        const mb_row *row = &rows[r];
        pulsar_gpu_tensor *bank_view = pulsar_gpu_tensor_view(
                index_slab, (uint64_t)row->bank * index_bank_bytes, index_bank_bytes);
        pulsar_gpu_tensor *q_view = pulsar_gpu_tensor_view(q, (uint64_t)r * qp_row,
                                                     qp_row);
        pulsar_gpu_tensor *w_view = pulsar_gpu_tensor_view(w, (uint64_t)r * n_head * sizeof(float),
                                                     (uint64_t)n_head * sizeof(float));
        int ok = bank_view && q_view && w_view &&
                 pulsar_gpu_indexer_scores_decode_batch_tensor(scores_ref, q_view, w_view,
                                                            bank_view, n_comp_superset, 1,
                                                            row->qpos, n_head, head_dim,
                                                            ratio, scale,
                                                            NULL, NULL, NULL, 0, 1) &&
                 pulsar_gpu_indexer_topk_tensor(sel_ref_t, scores_ref, n_comp_superset, 1, top_k) &&
                 pulsar_gpu_synchronize() &&
                 pulsar_gpu_tensor_read(scores_ref, 0, out_ref,
                                     (uint64_t)n_comp_superset * sizeof(float)) &&
                 pulsar_gpu_tensor_read(sel_ref_t, 0, sel_ref,
                                     (uint64_t)top_k * sizeof(uint32_t));
        pulsar_gpu_tensor_free(w_view);
        pulsar_gpu_tensor_free(q_view);
        pulsar_gpu_tensor_free(bank_view);
        if (!ok) {
            fprintf(stderr, "multibank indexer %s: reference launch failed (row %u)\n", label, r);
            goto done;
        }
        if (memcmp(out_ref, out_batch + (uint64_t)r * n_comp_superset,
                   (uint64_t)n_comp_superset * sizeof(float)) != 0) {
            fprintf(stderr, "multibank indexer %s: scores row %u (bank %u pos %u) != single-session\n",
                    label, r, row->bank, row->qpos);
            goto done;
        }
        if (memcmp(sel_ref, sel_batch + (uint64_t)r * top_k,
                   (uint64_t)top_k * sizeof(uint32_t)) != 0) {
            fprintf(stderr, "multibank indexer %s: top-k row %u (bank %u pos %u) != single-session\n",
                    label, r, row->bank, row->qpos);
            goto done;
        }
        /* Top-k must stay inside the row's own visible frontier whenever it
         * has at least top_k visible rows (shorter rows legitimately carry
         * -INF losers, which the consuming attention kernel drops per row). */
        const uint32_t visible = (row->qpos + 1u) / ratio < n_comp_superset
            ? (row->qpos + 1u) / ratio : n_comp_superset;
        if (visible >= top_k) {
            for (uint32_t k = 0; k < top_k; k++) {
                if (sel_batch[(uint64_t)r * top_k + k] >= visible) {
                    fprintf(stderr, "multibank indexer %s: top-k row %u selected id %u >= visible %u\n",
                            label, r, sel_batch[(uint64_t)r * top_k + k], visible);
                    goto done;
                }
            }
        }
    }
    printf("  multibank indexer %s: %u rows across %u banks -> byte-identical (scores + top-k)\n",
           label, n_rows, n_banks);
    rc = 0;

done:
    pulsar_gpu_tensor_free(seq_id); pulsar_gpu_tensor_free(positions);
    pulsar_gpu_tensor_free(sel_ref_t); pulsar_gpu_tensor_free(sel);
    pulsar_gpu_tensor_free(scores_ref); pulsar_gpu_tensor_free(scores);
    pulsar_gpu_tensor_free(w); pulsar_gpu_tensor_free(q);
    free(sid_host); free(pos_host); free(sel_ref); free(sel_batch);
    free(out_ref); free(out_batch); free(w_host); free(qp_host); free(q_host);
    return rc;
}

static int check_multibank_indexer(void) {
    const uint32_t n_banks = 2, comp_cap = 32, ratio = 4;
    const uint32_t n_comp_sup = 25;   /* cross-bank superset */
    const uint32_t head_dim = 128;
    int rc = 1;

    /* Rows: bank 0 deep (qpos 100 -> visible 25 == superset), bank 1 at the
     * EMIT boundary qpos 39 (visible (39+1)/4 = 10; a floor(39/4) = 9 bug
     * loses score row 9 -> teeth), bank 1 at qpos 37 (visible 9), and bank 1
     * at qpos 17 (visible 4 < top_k 8: the -INF losers case). */
    const mb_row rows[4] = { {0, 100, 0}, {1, 39, 0}, {1, 37, 0}, {1, 17, 0} };

    /* MXFP4-packed cache (production format): synthetic packed rows — 64
     * nibble bytes + 4 E8M0 scale bytes constrained to a sane exponent range
     * (the dequant is well-defined for any codes; references and banked
     * launches read the same bytes). */
    {
        const uint64_t row_bytes = 68;   /* PULSAR_MXKV_FP4_ROWBYTES(128) */
        const uint64_t bank_bytes = comp_cap * row_bytes;
        const uint64_t total = (uint64_t)n_banks * bank_bytes;
        uint8_t *host = (uint8_t *)malloc(total);
        pulsar_gpu_tensor *slab = pulsar_gpu_tensor_alloc(total);
        int case_rc = 1;
        if (host && slab) {
            mb_rng_state = 0xf4f4f4u;
            for (uint64_t i = 0; i < total; i++) {
                const uint64_t in_row = i % row_bytes;
                const uint32_t r8 = (uint32_t)(mb_rng_state >> 16) & 0xffu;
                mb_rng_state = mb_rng_state * 1664525u + 1013904223u;
                host[i] = in_row < 64 ? (uint8_t)r8 : (uint8_t)(115u + (r8 & 15u));
            }
            if (pulsar_gpu_tensor_write(slab, 0, host, total)) {
                case_rc = mb_idx_run_case("fp4", rows, 4, slab, comp_cap,
                                          n_comp_sup, ratio, n_banks, 4, head_dim,
                                          bank_bytes, 8);
                /* fp4 direct-one fast tier (n_tokens == 1, n_head 64): the
                 * n_head=4 case above dispatches the generic kernel, so the
                 * banked fp4 DIRECT-ONE tier needs its own row (same slab;
                 * emit-boundary qpos 39 as visibility teeth).  Reference is
                 * the same scalar-mode dispatch on the bank's view. */
                if (case_rc == 0) {
                    const mb_row one[1] = { {1, 39, 0} };
                    case_rc = mb_idx_run_case("fp4-direct-one", one, 1, slab,
                                              comp_cap, n_comp_sup, ratio,
                                              n_banks, 64, head_dim,
                                              bank_bytes, 8);
                }
            }
        }
        pulsar_gpu_tensor_free(slab);
        free(host);
        if (case_rc != 0) goto done;
    }



    /* Dead-row guard: an out-of-pool seq_id must produce an all -INF score
     * row (fail-visible) while its batchmate row scores normally. */
    {
        const uint32_t n_head = 4;
        const uint64_t q_row = (uint64_t)n_head * head_dim;
        const uint64_t bank_bytes = (uint64_t)comp_cap * head_dim * sizeof(float);
        const uint64_t count = (uint64_t)n_banks * comp_cap * head_dim;
        float *host = (float *)malloc(count * sizeof(float));
        float *q_host = (float *)malloc(2 * q_row * sizeof(float));
        float *w_host = (float *)malloc(2 * n_head * sizeof(float));
        float *out = (float *)malloc(2 * (uint64_t)n_comp_sup * sizeof(float));
        const int32_t pos_host[2] = {100, 37};
        const int32_t sid_host[2] = {-1, 1};
        pulsar_gpu_tensor *slab = pulsar_gpu_tensor_alloc(count * sizeof(float));
        uint8_t qp_host2[2 * 64 * 68];   /* n_head<=64 packed rows x 2 tokens */
        pulsar_gpu_tensor *q = pulsar_gpu_tensor_alloc(2 * (uint64_t)n_head * 68u);
        pulsar_gpu_tensor *w = pulsar_gpu_tensor_alloc(2 * n_head * sizeof(float));
        pulsar_gpu_tensor *scores = pulsar_gpu_tensor_alloc(2 * (uint64_t)n_comp_sup * sizeof(float));
        pulsar_gpu_tensor *p = pulsar_gpu_tensor_alloc(2 * sizeof(int32_t));
        pulsar_gpu_tensor *s = pulsar_gpu_tensor_alloc(2 * sizeof(int32_t));
        int dead_rc = 1;
        (void)bank_bytes;
        if (host && q_host && w_host && out && slab && q && w && scores && p && s) {
            mb_rng_state = 0xdeadf00u;
            for (uint64_t i = 0; i < count; i++) host[i] = mb_rand();
            for (uint64_t i = 0; i < 2 * q_row; i++) q_host[i] = mb_rand() * 0.5f;
            for (uint64_t i = 0; i < 2 * n_head; i++) w_host[i] = 1.0f;
            if (pulsar_gpu_tensor_write(slab, 0, host, count * sizeof(float)) &&
                (mb_pack_q_rows(q_host, qp_host2, 2 * (uint64_t)n_head), 1) &&
                pulsar_gpu_tensor_write(q, 0, qp_host2, 2 * (uint64_t)n_head * 68u) &&
                pulsar_gpu_tensor_write(w, 0, w_host, 2 * n_head * sizeof(float)) &&
                pulsar_gpu_tensor_write(p, 0, pos_host, sizeof(pos_host)) &&
                pulsar_gpu_tensor_write(s, 0, sid_host, sizeof(sid_host)) &&
                pulsar_gpu_indexer_scores_decode_batch_tensor(scores, q, w, slab,
                                                           n_comp_sup, 2, 0,
                                                           n_head, head_dim, ratio, 0.125f,
                                                           p, s, NULL, comp_cap, n_banks) &&
                pulsar_gpu_synchronize() &&
                pulsar_gpu_tensor_read(scores, 0, out, 2 * (uint64_t)n_comp_sup * sizeof(float))) {
                dead_rc = 0;
                for (uint32_t c = 0; c < n_comp_sup; c++) {
                    if (!(out[c] == -INFINITY)) {
                        fprintf(stderr, "multibank indexer dead-row: finite score at %u\n", c);
                        dead_rc = 1;
                        break;
                    }
                }
                if (dead_rc == 0)
                    printf("  multibank indexer dead-row: out-of-pool seq_id -> all -INF scores\n");
            }
        }
        pulsar_gpu_tensor_free(s); pulsar_gpu_tensor_free(p); pulsar_gpu_tensor_free(scores);
        pulsar_gpu_tensor_free(w); pulsar_gpu_tensor_free(q); pulsar_gpu_tensor_free(slab);
        free(out); free(w_host); free(q_host); free(host);
        if (dead_rc != 0) goto done;
    }
    rc = 0;
done:
    return rc;
}

/* ---------------------------------------------------------------------------
 * Multi-bank raw-store scatter smoke: one banked launch storing rows into
 * two banks' rings (incl. a wrapped slot and a dead row that must store
 * NOTHING) must leave the pool byte-identical to per-bank classic stores.
 * Covers the f16 and f32 ring formats.
 * ------------------------------------------------------------------------- */
static int check_multibank_raw_store(void) {
    /* head_dim is the model's, not an arbitrary 32: PULSAR_ATTN_PACK needs
     * head_dim > n_rot with the nope span divisible by the scale block, so the
     * old 32-wide fixture cannot be expressed in the format the ring now holds.
     * The f16/f32 loop is gone with it -- there is one ring format. */
    const uint32_t n_banks = 2, raw_cap = 64, head_dim = 512, n_rows = 4;
    const int32_t pos_host[4] = {100, 5, 6, 7};    /* 100 % 64 = 36 (wrap) */
    const int32_t sid_host[4] = {0, 1, 1, -1};     /* row 3 dead */
    const uint64_t row_bytes = SMOKE_ATTN_ROWBYTES(head_dim);
    const uint64_t bank_bytes = (uint64_t)raw_cap * row_bytes;
    const uint64_t total = (uint64_t)n_banks * bank_bytes;
    const uint64_t kv_count = (uint64_t)n_rows * head_dim;
    int rc = 1;
    uint8_t *fill = (uint8_t *)malloc(total);
    uint8_t *got = (uint8_t *)malloc(total);
    uint8_t *want = (uint8_t *)malloc(total);
    float *kv_host = (float *)malloc(kv_count * sizeof(float));
    pulsar_gpu_tensor *slab = pulsar_gpu_tensor_alloc(total);
    pulsar_gpu_tensor *slab_ref = pulsar_gpu_tensor_alloc(total);
    pulsar_gpu_tensor *kv = pulsar_gpu_tensor_alloc(kv_count * sizeof(float));
    pulsar_gpu_tensor *positions = pulsar_gpu_tensor_alloc(n_rows * sizeof(int32_t));
    pulsar_gpu_tensor *seq_id = pulsar_gpu_tensor_alloc(n_rows * sizeof(int32_t));
    int ok = fill && got && want && kv_host && slab && slab_ref && kv &&
             positions && seq_id;
    if (ok) {
        memset(fill, 0x55, total);   /* poison: untouched slots must match */
        mb_rng_state = 0x5708eu;
        for (uint64_t i = 0; i < kv_count; i++) kv_host[i] = mb_rand();
        ok = pulsar_gpu_tensor_write(slab, 0, fill, total) &&
             pulsar_gpu_tensor_write(slab_ref, 0, fill, total) &&
             pulsar_gpu_tensor_write(kv, 0, kv_host, kv_count * sizeof(float)) &&
             pulsar_gpu_tensor_write(positions, 0, pos_host, sizeof(pos_host)) &&
             pulsar_gpu_tensor_write(seq_id, 0, sid_host, sizeof(sid_host));
    }
    /* Banked scatter: one launch, all rows. */
    if (ok) ok = pulsar_gpu_store_raw_kv_batch_tensor(slab, kv, raw_cap, 0, n_rows,
                                                   head_dim,
                                                   positions, seq_id, n_banks);
    /* Reference: classic per-row stores into the bank views (dead row
     * skipped -- it must store nothing). */
    for (uint32_t r = 0; ok && r < n_rows; r++) {
        if (sid_host[r] < 0) continue;
        pulsar_gpu_tensor *bank_view = pulsar_gpu_tensor_view(
                slab_ref, (uint64_t)sid_host[r] * bank_bytes, bank_bytes);
        pulsar_gpu_tensor *kv_view = pulsar_gpu_tensor_view(
                kv, (uint64_t)r * head_dim * sizeof(float),
                (uint64_t)head_dim * sizeof(float));
        ok = bank_view && kv_view &&
             pulsar_gpu_store_raw_kv_tensor(bank_view, kv_view, raw_cap,
                                         (uint32_t)pos_host[r] % raw_cap,
                                         head_dim);
        pulsar_gpu_tensor_free(kv_view);
        pulsar_gpu_tensor_free(bank_view);
    }
    ok = ok && pulsar_gpu_synchronize() &&
         pulsar_gpu_tensor_read(slab, 0, got, total) &&
         pulsar_gpu_tensor_read(slab_ref, 0, want, total);
    if (ok && memcmp(got, want, total) != 0) {
        fprintf(stderr, "multibank raw-store: banked scatter != per-bank stores\n");
        ok = 0;
    }
    pulsar_gpu_tensor_free(seq_id); pulsar_gpu_tensor_free(positions);
    pulsar_gpu_tensor_free(kv); pulsar_gpu_tensor_free(slab_ref); pulsar_gpu_tensor_free(slab);
    free(kv_host); free(want); free(got); free(fill);
    if (!ok) return 1;
    printf("  multibank raw-store: banked scatter == per-bank stores "
           "(%llu B/row, dead row stored nothing)\n", (unsigned long long)row_bytes);
    rc = 0;
    return rc;
}

int main(void) {
    if (!pulsar_gpu_init()) return 1;
    int rc = check_large_topk();
    if (check_dspark_markov_head() != 0) rc = 1;
    if (check_dspark_confidence_head() != 0) rc = 1;
    if (check_dspark_non_causal_attention() != 0) rc = 1;
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    if (check_multibank_decode_attention() != 0) rc = 1;
    if (check_multibank_indexer() != 0) rc = 1;
    if (check_multibank_raw_store() != 0) rc = 1;
    pulsar_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}

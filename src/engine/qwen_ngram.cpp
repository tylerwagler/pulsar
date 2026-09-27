/* Qwen3.8-Flash-Next PLE n-gram ids (L251 S4) -- see qwen_ngram.h for the
 * definition and why it is exact.  tests/qwen_ngram_test.cpp holds it equal to
 * the source module's `ngram_ids` on the calibration corpus. */

#include "qwen_ngram.h"
#include "pulsar_engine_internal.h"

int pulsar_qwen_ngram_layout_check(const pulsar_qwen_ngram_layout *L) {
    if (!L || L->vocab == 0 || L->eos < 0 || (uint32_t)L->eos >= L->vocab || L->n_rows == 0) {
        fprintf(stderr, "pulsar: qwen n-gram layout: empty vocab or an EOS outside it -- refusing\n");
        return 0;
    }
    const int64_t mult_cap = INT64_MAX / (int64_t)L->vocab;
    for (uint32_t i = 0; i < PULSAR_QWEN_NGRAM_ORDER; i++) {
        if (L->mult[i] <= 0 || (L->mult[i] & 1) == 0 || L->mult[i] >= mult_cap) {
            fprintf(stderr, "pulsar: qwen n-gram layout: multiplier %u = %lld is not odd in (0, %lld) -- refusing\n",
                    i, (long long)L->mult[i], (long long)mult_cap);
            return 0;
        }
    }
    for (uint32_t h = 0; h < PULSAR_QWEN_NGRAM_COLS; h++) {
        if (L->prime[h] == 0 || L->offset[h] + L->prime[h] > L->n_rows ||
            (h > 0 && L->offset[h] != L->offset[h - 1] + L->prime[h - 1]) || (h == 0 && L->offset[0] != 0)) {
            fprintf(stderr, "pulsar: qwen n-gram layout: head %u range [%llu, +%llu) is not the next range of a "
                            "%llu-row table -- refusing\n", h, (unsigned long long)L->offset[h],
                    (unsigned long long)L->prime[h], (unsigned long long)L->n_rows);
            return 0;
        }
    }
    return 1;
}

void pulsar_qwen_ngram_ctx_init(const pulsar_qwen_ngram_layout *L, pulsar_qwen_ngram_ctx *ctx) {
    ctx->prev[0] = ctx->prev[1] = L->eos;
}

void pulsar_qwen_ngram_rows(const pulsar_qwen_ngram_layout *L, pulsar_qwen_ngram_ctx *ctx,
                            const int32_t *ids, uint32_t n, uint64_t *rows) {
    if (!L || !ctx || (n && (!ids || !rows))) pulsar_die("qwen n-gram rows: called with a null layout, context or buffer");
    int32_t p2 = ctx->prev[0], p1 = ctx->prev[1];
    for (uint32_t i = 0; i < n; i++) {
        const int32_t x = ids[i];
        if (x < 0 || (uint32_t)x >= L->vocab) pulsar_die("qwen n-gram rows: token id outside the vocab");
        const int64_t s0 = x, s1 = p1, s2 = (p1 == L->eos) ? L->eos : p2;
        const int64_t m2 = (s0 * L->mult[0]) ^ (s1 * L->mult[1]);
        const int64_t m3 = m2 ^ (s2 * L->mult[2]);
        uint64_t *r = rows + (uint64_t)i * PULSAR_QWEN_NGRAM_COLS;
        for (uint32_t h = 0; h < PULSAR_QWEN_NGRAM_HEADS; h++) {
            r[h] = (uint64_t)m2 % L->prime[h] + L->offset[h];
            r[PULSAR_QWEN_NGRAM_HEADS + h] = (uint64_t)m3 % L->prime[PULSAR_QWEN_NGRAM_HEADS + h] +
                                             L->offset[PULSAR_QWEN_NGRAM_HEADS + h];
        }
        p2 = p1;
        p1 = x;
    }
    ctx->prev[0] = p2;
    ctx->prev[1] = p1;
}

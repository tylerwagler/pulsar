/* Qwen3.8-Flash-Next PLE n-gram ids (L251 S4) -- see qwen_ngram.h for the
 * definition and why it is exact.  tests/qwen_ngram_test.cpp holds it equal to
 * the source module's `ngram_ids` on the calibration corpus. */

#include "qwen_ngram.h"
#include "pulsar_engine_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

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

int pulsar_qwen_ngram_table_open(pulsar_engram_table *t, const char *path, uint32_t layer,
                                 const pulsar_qwen_ngram_layout *L) {
    if (!t || !path || !L) return 0;
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "pulsar: qwen n-gram rows %s: %s\n", path, strerror(errno));
        return 0;
    }
    unsigned char hdr[PULSAR_ENGRAM_HDR_BYTES];
    const bool got = pread(fd, hdr, sizeof hdr, 0) == (ssize_t)sizeof hdr;
    struct stat st;
    const bool sized = fstat(fd, &st) == 0;
    close(fd);
    uint32_t version = 0, file_layer = 0, row_bytes = 0, dim = 0, n_scale = 0, dtype = 0, n_heads = 0;
    uint64_t n_rows = 0;
    if (got) {
        memcpy(&version, hdr + 8, 4); memcpy(&file_layer, hdr + 12, 4); memcpy(&n_rows, hdr + 16, 8);
        memcpy(&row_bytes, hdr + 24, 4); memcpy(&dim, hdr + 28, 4); memcpy(&n_scale, hdr + 32, 4);
        memcpy(&dtype, hdr + 36, 4); memcpy(&n_heads, hdr + 40, 4);   /* 44: rows_per_part, the source's split -- one file here */
    }
    const uint64_t want = PULSAR_ENGRAM_HDR_BYTES + L->n_rows * PULSAR_QWEN_NGRAM_ROW_BYTES;
    if (!got || !sized || memcmp(hdr, "PENGRAM1", 8) != 0 || version != PULSAR_QWEN_NGRAM_FILE_VERSION ||
        file_layer != layer || n_rows != L->n_rows || row_bytes != PULSAR_QWEN_NGRAM_ROW_BYTES ||
        dim * 2u != PULSAR_QWEN_NGRAM_ROW_BYTES || n_scale != 0 || dtype != PULSAR_QWEN_NGRAM_DTYPE_BF16 ||
        n_heads != PULSAR_QWEN_NGRAM_COLS || (uint64_t)st.st_size != want) {
        fprintf(stderr, "pulsar: qwen n-gram rows %s: header says %.8s v%u layer %u rows %llu record %u (dim %u, "
                        "%u scales, dtype %u) heads %u, size %lld; this model wants PENGRAM1 v%u layer %u rows %llu "
                        "record %u bf16 heads %u size %llu -- refusing\n",
                path, got ? (const char *)hdr : "?", version, file_layer, (unsigned long long)n_rows, row_bytes, dim,
                n_scale, dtype, n_heads, sized ? (long long)st.st_size : -1LL, PULSAR_QWEN_NGRAM_FILE_VERSION, layer,
                (unsigned long long)L->n_rows, PULSAR_QWEN_NGRAM_ROW_BYTES, PULSAR_QWEN_NGRAM_COLS,
                (unsigned long long)want);
        return 0;
    }
    const uint64_t base = PULSAR_ENGRAM_HDR_BYTES;
    return pulsar_engram_table_open_parts(t, layer, 1, &path, &base, L->n_rows, PULSAR_QWEN_NGRAM_ROW_BYTES, L->n_rows);
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

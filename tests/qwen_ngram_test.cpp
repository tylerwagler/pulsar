/* QWEN N-GRAM test (L251 S4) -- HOST ONLY: the PLE n-gram ids against the source
 * module, and the row gather against the checkpoint's bytes.
 *
 * The vectors (tools/qwen/gen_ngram_vectors.py) hold the checkpoint's layout buffers,
 * token sequences (the calibration corpus + synthetic EOS cases) with the ids
 * transformers' Qwen4ExpTextNGramEmbedding computed for them, and a sample of table
 * rows read from the checkpoint's shard tensors with the (file, offset) of every part.
 *
 *   1. ids: pulsar_qwen_ngram_rows over each whole sequence == the source's ids, every
 *      one of the 16 per token (bit-exact, no tolerance);
 *   2. chunking: the same sequences fed in random chunks (1-token decode steps
 *      included) through the carried context give the same ids -- the per-sequence
 *      state really is the last two tokens;
 *   3. mutation: a multiplier off by 2 must change the ids (the comparison is live);
 *   4. refusals: an even multiplier and a broken head range are refused by the check;
 *   5. rows: the engine's pread pool returns the checkpoint's bytes for every sampled
 *      row (including the first and last row of every part) -- over the checkpoint's 128
 *      shard parts in place, or with --rowfile over the container builder's PENGRAM1 v2
 *      row file (header asserted against the layout; a v1 header and a wrong layer are
 *      refused); a row past the table fails the gather by name.
 *
 * usage: qwen_ngram_test VECTORS [--no-rows | --rowfile PATH] */
#include "pulsar_engine_internal.h"
#include "qwen_ngram.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

static int g_fail;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail = 1; printf("  FAIL  "); } else { printf("  ok    "); } \
                              printf(__VA_ARGS__); printf("\n"); } while (0)

struct reader {
    std::vector<unsigned char> b;
    size_t p = 0;
    template <typename T> T get() { T v; if (p + sizeof v > b.size()) { fprintf(stderr, "short vectors\n"); exit(2); }
                                    memcpy(&v, b.data() + p, sizeof v); p += sizeof v; return v; }
    const unsigned char *take(size_t n) { if (p + n > b.size()) { fprintf(stderr, "short vectors\n"); exit(2); }
                                          const unsigned char *q = b.data() + p; p += n; return q; }
};

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s VECTORS [--no-rows]\n", argv[0]); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    const bool rows_part = !(argc > 2 && strcmp(argv[2], "--no-rows") == 0);
    const char *rowfile = (argc > 3 && strcmp(argv[2], "--rowfile") == 0) ? argv[3] : NULL;
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    reader r;
    fseek(f, 0, SEEK_END);
    r.b.resize((size_t)ftell(f));
    fseek(f, 0, SEEK_SET);
    if (fread(r.b.data(), 1, r.b.size(), f) != r.b.size()) { fprintf(stderr, "short read\n"); return 2; }
    fclose(f);
    if (memcmp(r.take(8), "QNGRAM01", 8) != 0) { fprintf(stderr, "bad magic\n"); return 2; }

    pulsar_qwen_ngram_layout L;
    memset(&L, 0, sizeof L);
    L.vocab = r.get<uint32_t>();
    L.eos = r.get<int32_t>();
    L.n_rows = r.get<uint64_t>();
    for (auto &m : L.mult) m = r.get<int64_t>();
    for (auto &p : L.prime) p = r.get<uint64_t>();
    for (auto &o : L.offset) o = r.get<uint64_t>();
    const uint64_t rows_per_part = r.get<uint64_t>();
    const uint32_t n_parts = r.get<uint32_t>();
    std::vector<uint64_t> bases(n_parts);
    std::vector<std::string> paths(n_parts);
    for (uint32_t i = 0; i < n_parts; i++) {
        bases[i] = r.get<uint64_t>();
        const uint16_t n = r.get<uint16_t>();
        paths[i].assign((const char *)r.take(n), n);
    }
    printf("qwen-ngram-test: vocab %u eos %d table %llu rows in %u parts of %llu\n", L.vocab, L.eos,
           (unsigned long long)L.n_rows, n_parts, (unsigned long long)rows_per_part);
    CHECK(pulsar_qwen_ngram_layout_check(&L), "the checkpoint's layout passes the invariant check");

    struct seq { std::vector<int32_t> ids; std::vector<uint64_t> rows; };
    std::vector<seq> seqs(r.get<uint32_t>());
    size_t n_tok = 0, n_eos = 0;
    for (auto &s : seqs) {
        const uint32_t n = r.get<uint32_t>();
        s.ids.resize(n);
        s.rows.resize((size_t)n * PULSAR_QWEN_NGRAM_COLS);
        memcpy(s.ids.data(), r.take((size_t)n * 4), (size_t)n * 4);
        memcpy(s.rows.data(), r.take(s.rows.size() * 8), s.rows.size() * 8);
        n_tok += n;
        for (int32_t x : s.ids) n_eos += x == L.eos;
    }

    /* 1. whole sequences */
    size_t bad = 0, bad_tok = 0;
    std::vector<uint64_t> got;
    for (const auto &s : seqs) {
        got.assign(s.rows.size(), 0);
        pulsar_qwen_ngram_ctx ctx;
        pulsar_qwen_ngram_ctx_init(&L, &ctx);
        pulsar_qwen_ngram_rows(&L, &ctx, s.ids.data(), (uint32_t)s.ids.size(), got.data());
        for (size_t t = 0; t < s.ids.size(); t++) {
            size_t b = 0;
            for (uint32_t h = 0; h < PULSAR_QWEN_NGRAM_COLS; h++) b += got[t * 16 + h] != s.rows[t * 16 + h];
            bad += b;
            bad_tok += b != 0;
        }
    }
    CHECK(bad == 0, "ids: %zu sequences, %zu tokens (%zu EOS), %zu ids: %zu differ from the source module (%zu tokens)",
          seqs.size(), n_tok, n_eos, n_tok * 16, bad, bad_tok);

    /* 2. chunked, through the carried context */
    uint32_t rng = 0x9e3779b9u;
    size_t bad_chunk = 0, chunks = 0;
    for (const auto &s : seqs) {
        got.assign(s.rows.size(), 0);
        pulsar_qwen_ngram_ctx ctx;
        pulsar_qwen_ngram_ctx_init(&L, &ctx);
        for (size_t t = 0; t < s.ids.size();) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            const size_t want = (rng % 4 == 0) ? 1 + rng % 37 : 1;    /* mostly decode steps */
            const size_t n = want < s.ids.size() - t ? want : s.ids.size() - t;
            pulsar_qwen_ngram_rows(&L, &ctx, s.ids.data() + t, (uint32_t)n, got.data() + t * 16);
            t += n;
            chunks++;
        }
        for (size_t i = 0; i < got.size(); i++) bad_chunk += got[i] != s.rows[i];
    }
    CHECK(bad_chunk == 0, "chunked: %zu chunks (mostly 1-token steps): %zu ids differ", chunks, bad_chunk);

    /* 3. mutation */
    {
        pulsar_qwen_ngram_layout M = L;
        M.mult[2] += 2;
        const seq &s = seqs[0];
        got.assign(s.rows.size(), 0);
        pulsar_qwen_ngram_ctx ctx;
        pulsar_qwen_ngram_ctx_init(&M, &ctx);
        pulsar_qwen_ngram_rows(&M, &ctx, s.ids.data(), (uint32_t)s.ids.size(), got.data());
        size_t moved = 0;
        for (size_t i = 0; i < got.size(); i++) moved += got[i] != s.rows[i];
        CHECK(moved > 0 && moved <= got.size() / 2, "mutation: trigram multiplier + 2 moves %zu ids (trigram heads only)", moved);
    }
    /* 4. refusals */
    {
        pulsar_qwen_ngram_layout M = L;
        M.mult[1] += 1;
        CHECK(!pulsar_qwen_ngram_layout_check(&M), "an even multiplier is refused (message above expected)");
        M = L;
        M.offset[5] += 1;
        CHECK(!pulsar_qwen_ngram_layout_check(&M), "a head range that is not the next range is refused (message above expected)");
    }

    /* 5. rows */
    const uint32_t n_check = r.get<uint32_t>();
    if (!rows_part) {
        printf("  skip  rows: --no-rows\n");
    } else {
        std::vector<const char *> pp(n_parts);
        for (uint32_t i = 0; i < n_parts; i++) pp[i] = paths[i].c_str();
        pulsar_engram_table t;
        if (rowfile) {
            CHECK(!pulsar_qwen_ngram_table_open(&t, rowfile, 2u, &L), "the row file opened as layer 2 is refused (message above expected)");
            pulsar_qwen_ngram_layout L2 = L;
            L2.n_rows += 128;
            CHECK(!pulsar_qwen_ngram_table_open(&t, rowfile, 1u, &L2), "a layout with another row count is refused (message above expected)");
        }
        if (rowfile ? !pulsar_qwen_ngram_table_open(&t, rowfile, 1u, &L)
                    : !pulsar_engram_table_open_parts(&t, 1u, n_parts, pp.data(), bases.data(), rows_per_part,
                                                      PULSAR_QWEN_NGRAM_ROW_BYTES, L.n_rows)) {
            CHECK(0, "open the table (%s)", rowfile ? rowfile : pp[0]);
        } else {
            printf("        rows via %s\n", rowfile ? "the PENGRAM1 v2 row file" : "the checkpoint's 128 shard parts");
            std::vector<uint64_t> ids(n_check);
            std::vector<unsigned char> want((size_t)n_check * PULSAR_QWEN_NGRAM_ROW_BYTES);
            for (uint32_t i = 0; i < n_check; i++) {
                ids[i] = r.get<uint64_t>();
                memcpy(want.data() + (size_t)i * PULSAR_QWEN_NGRAM_ROW_BYTES, r.take(PULSAR_QWEN_NGRAM_ROW_BYTES),
                       PULSAR_QWEN_NGRAM_ROW_BYTES);
            }
            /* shuffle so the pool's slices cross parts in every order */
            for (uint32_t i = n_check; i > 1; i--) {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                const uint32_t j = rng % i;
                std::swap(ids[i - 1], ids[j]);
                std::vector<unsigned char> tmp(want.begin() + (size_t)(i - 1) * 320, want.begin() + (size_t)i * 320);
                std::copy(want.begin() + (size_t)j * 320, want.begin() + (size_t)(j + 1) * 320, want.begin() + (size_t)(i - 1) * 320);
                std::copy(tmp.begin(), tmp.end(), want.begin() + (size_t)j * 320);
            }
            pulsar_engram_io *io = pulsar_engram_io_create(PULSAR_ENGRAM_IO_THREADS);
            std::vector<unsigned char> dst(want.size(), 0);
            const int ok = pulsar_engram_gather_wait(pulsar_engram_gather_start(io, &t, ids.data(), n_check, dst.data()));
            size_t rb = 0;
            for (uint32_t i = 0; i < n_check; i++)
                rb += memcmp(dst.data() + (size_t)i * 320, want.data() + (size_t)i * 320, 320) != 0;
            CHECK(ok && rb == 0, "rows: %u sampled rows (every source part's first and last included; %u parts in the source): %zu differ",
                  n_check, n_parts, rb);
            const uint64_t past = L.n_rows;
            const int bad_ok = pulsar_engram_gather_wait(pulsar_engram_gather_start(io, &t, &past, 1, dst.data()));
            CHECK(!bad_ok, "a row past the table fails the gather (message above expected)");
            pulsar_engram_io_destroy(io);
            pulsar_engram_table_close(&t);
        }
    }
    printf(g_fail ? "QWEN-NGRAM TEST FAIL\n" : "QWEN-NGRAM TEST PASS\n");
    return g_fail;
}

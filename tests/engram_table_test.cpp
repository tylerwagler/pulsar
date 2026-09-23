/* ENGRAM TABLE test (L242) -- HOST ONLY: the row file's header contract and the gather
 * pool, against the device-path fixture's rows.
 *
 * The fixture (tools/engram/gen_engram_fixture.py) holds, for a fixed token window, the
 * 24 row ids per token and the 264-byte rows read from the CHECKPOINT at those ids.  The
 * row file (tools/engram/engram_rows.c) holds every row of the layer, interleaved.  So a
 * gather of the fixture's ids from the row file must return the fixture's bytes exactly:
 * that closes the loop hash -> row id -> file offset -> record, with the checkpoint as
 * the oracle on the other side.  Also asserted: a wrong-layer open is refused, a row id
 * past the table fails the gather by name, and a 1,000-row gather over the pool lands
 * every row where it belongs (order independence).
 *
 * usage: engram_table_test FIXTURE.fix ENGRAM_DIR      (reads ENGRAM_DIR/engram-l<L>.rows) */
#include "pulsar_engine_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail = 1; printf("  FAIL  "); } else { printf("  ok    "); } printf(__VA_ARGS__); printf("\n"); } while (0)

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s FIXTURE.fix ENGRAM_DIR\n", argv[0]); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);   /* a crash must not swallow the checks that ran */
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    const long fn = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *fx = (unsigned char *)malloc((size_t)fn);
    if (!fx || fread(fx, 1, (size_t)fn, f) != (size_t)fn) { fprintf(stderr, "short fixture\n"); return 2; }
    fclose(f);
    if (memcmp(fx, "PENGFIX1", 8) != 0) { fprintf(stderr, "bad fixture magic\n"); return 2; }
    uint32_t layer = 0, T = 0, n_cols = 0, row_bytes = 0;
    memcpy(&layer, fx + 8, 4); memcpy(&T, fx + 12, 4); memcpy(&n_cols, fx + 24, 4); memcpy(&row_bytes, fx + 28, 4);
    if (n_cols != PULSAR_ENGRAM_N_COLS || row_bytes != PULSAR_ENGRAM_ROW_BYTES) { fprintf(stderr, "fixture geometry\n"); return 2; }
    const uint64_t *cols = (const uint64_t *)(fx + 36 + (size_t)T * 4);
    const unsigned char *rows = fx + 36 + (size_t)T * 4 + (size_t)T * n_cols * 8;
    const uint32_t n = T * n_cols;

    /* the row count comes from the layout, as the engine will pass it; the fixture's
     * ids bound it from below, the file header states it */
    char path[4096];
    snprintf(path, sizeof path, "%s/engram-l%u.rows", argv[2], layer);
    FILE *rf = fopen(path, "rb");
    if (!rf) { perror(path); return 2; }
    unsigned char hdr[PULSAR_ENGRAM_HDR_BYTES];
    if (fread(hdr, 1, sizeof hdr, rf) != sizeof hdr) { fprintf(stderr, "short header\n"); return 2; }
    fclose(rf);
    uint64_t n_rows = 0; memcpy(&n_rows, hdr + 16, 8);
    printf("engram table test: layer %u, %llu rows, %u fixture rows in [%llu, %llu]\n", layer,
           (unsigned long long)n_rows, n, (unsigned long long)cols[0], (unsigned long long)cols[n - 1]);

    pulsar_engram_table t;
    CHECK(!pulsar_engram_table_open(&t, path, layer + 1u, n_rows), "a wrong-layer open is refused (the message above is expected)");
    CHECK(!pulsar_engram_table_open(&t, path, layer, n_rows + 1u), "a wrong-row-count open is refused (the message above is expected)");
    if (!pulsar_engram_table_open(&t, path, layer, n_rows)) { CHECK(0, "open %s", path); return 1; }
    CHECK(t.n_rows == n_rows && t.fd >= 0, "open: %s (%llu rows)", path, (unsigned long long)t.n_rows);

    pulsar_engram_io *io = pulsar_engram_io_create(PULSAR_ENGRAM_IO_THREADS);
    if (!io) { CHECK(0, "io pool"); return 1; }
    unsigned char *dst = (unsigned char *)calloc((size_t)n, PULSAR_ENGRAM_ROW_BYTES);
    pulsar_engram_gather *g = pulsar_engram_gather_start(io, &t, cols, n, dst);
    const int ok = g && pulsar_engram_gather_wait(g);
    CHECK(ok, "gather of the fixture's %u rows completed", n);
    if (ok) {
        uint32_t bad = 0;
        for (uint32_t i = 0; i < n; i++)
            if (memcmp(dst + (size_t)i * PULSAR_ENGRAM_ROW_BYTES, rows + (size_t)i * PULSAR_ENGRAM_ROW_BYTES, PULSAR_ENGRAM_ROW_BYTES) != 0) bad++;
        CHECK(bad == 0, "%u of %u gathered rows differ from the checkpoint's bytes in the fixture", bad, n);
    }
    /* an out-of-range id fails by name, the in-range rows still land */
    uint64_t *ids2 = (uint64_t *)malloc((size_t)n * 8);
    memcpy(ids2, cols, (size_t)n * 8);
    ids2[n / 2] = n_rows;   /* one past the end */
    memset(dst, 0, (size_t)n * PULSAR_ENGRAM_ROW_BYTES);
    g = pulsar_engram_gather_start(io, &t, ids2, n, dst);
    CHECK(g && !pulsar_engram_gather_wait(g), "a row id past the table fails the gather (the message above is expected)");
    CHECK(memcmp(dst, rows, PULSAR_ENGRAM_ROW_BYTES) == 0, "the in-range rows of a failed gather still landed");
    /* a wide gather: the fixture's rows repeated to 1,000 with a shuffled order */
    const uint32_t W = 1000;
    uint64_t *ids3 = (uint64_t *)malloc((size_t)W * 8);
    uint32_t *src = (uint32_t *)malloc((size_t)W * 4);
    uint32_t s = 12345u;
    for (uint32_t i = 0; i < W; i++) { s = s * 1664525u + 1013904223u; src[i] = (s >> 8) % n; ids3[i] = cols[src[i]]; }
    unsigned char *dst3 = (unsigned char *)calloc((size_t)W, PULSAR_ENGRAM_ROW_BYTES);
    g = pulsar_engram_gather_start(io, &t, ids3, W, dst3);
    const int ok3 = g && pulsar_engram_gather_wait(g);
    uint32_t bad3 = 0;
    if (ok3) for (uint32_t i = 0; i < W; i++)
        if (memcmp(dst3 + (size_t)i * PULSAR_ENGRAM_ROW_BYTES, rows + (size_t)src[i] * PULSAR_ENGRAM_ROW_BYTES, PULSAR_ENGRAM_ROW_BYTES) != 0) bad3++;
    CHECK(ok3 && bad3 == 0, "a %u-row shuffled gather over the pool: %u misplaced", W, bad3);

    pulsar_engram_io_destroy(io);
    pulsar_engram_table_close(&t);
    free(dst); free(dst3); free(ids2); free(ids3); free(src); free(fx);
    printf("ENGRAM TABLE TEST: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}

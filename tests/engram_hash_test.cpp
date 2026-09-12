/* HOST gate for the Engram n-gram hash (src/engine/engram.cpp).
 *
 * No model, no device: the hash is integer arithmetic over a fixed layout, so
 * this binary grades the C++ port against vectors generated from the real V4.1
 * layout by pulsar-notes/gate-baseline/l218-v41/gen_engram_fixture.py.
 *
 * What this pins, and why each part is worth a check:
 *
 *   1. the 384 expected columns (2 layers x 8 positions x 24), byte-for-byte.
 *      This is the whole hash: the window, the XOR, the per-head modulus and
 *      the bucket offset.  A wrong multiplier, modulus or offset shows up here.
 *   2. every column lands inside its layer's table.  The offsets are the
 *      exclusive prefix sum of that layer's primes, so this is really a check
 *      that the two agree -- if they drift, the hash still produces plausible
 *      numbers and only silently reads the wrong rows.
 *   3. the pad path.  Positions 0..(max_ngram-2) mix real tokens with pad; the
 *      fixture's first three rows are those positions, so (3) is covered by (1)
 *      rather than by a separate instrument.
 *   4. the refusals.  The hash uses a plain `%` and relies on `rolling` never
 *      being negative, which holds only while every compressed id is in range.
 *      These two death tests are what make that a checked precondition instead
 *      of an assumption: an out-of-range id must die, not hash to a wrong row.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <sys/wait.h>
#include <unistd.h>

#include "pulsar_engine_internal.h"
#include "engram_hash_fixture.h"

static unsigned g_checks, g_failures, g_printed;

/* A broken port fails hundreds of columns at once; show the first handful with
 * the (order, head) breakdown and just count the rest. */
#define CHECK_DETAIL_MAX 10

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

/* Call `fn` in a forked child and report whether it died at all, and with what
 * on stderr.  pulsar_die() exits 1 after one line; anything else is a failure. */
static void expect_dies(void (*fn)(void), const char *what, const char *want_msg) {
    int fds[2];
    if (pipe(fds) != 0) { check(0, "pipe() failed for the %s case", what); return; }

    /* The child inherits stdout, and exit() flushes it -- without this the
     * banner would be printed once more by every death test. */
    fflush(stdout);

    const pid_t pid = fork();
    if (pid < 0) {
        check(0, "fork() failed for the %s case", what);
        close(fds[0]); close(fds[1]);
        return;
    }
    if (pid == 0) {
        /* stderr is unbuffered, so pulsar_die's line lands in the pipe before
         * exit() runs.  _exit avoids flushing the parent's inherited stdout. */
        dup2(fds[1], 2);
        close(fds[0]); close(fds[1]);
        fn();
        _exit(0); /* reaching here means the guard did NOT fire */
    }

    close(fds[1]);
    char buf[512];
    ssize_t n = read(fds[0], buf, sizeof buf - 1);
    close(fds[0]);
    buf[n > 0 ? (size_t)n : 0] = '\0';

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) { check(0, "waitpid() failed for the %s case", what); return; }

    check(WIFEXITED(status) && WEXITSTATUS(status) == 1,
          "%s: expected pulsar_die's exit(1), got %s (stderr: %s)", what,
          WIFEXITED(status) ? "a normal exit" : "a signal", buf);
    check(strstr(buf, want_msg) != NULL,
          "%s: stderr should name the refusal, got \"%s\"", what, buf);
}

/* The layout under test, built entirely from the fixture. */
static pulsar_engram_layout make_layout(void) {
    pulsar_engram_layout L;
    memset(&L, 0, sizeof L);
    L.n_vocab           = ENGRAM_FIX_N_VOCAB;
    L.compressed_vocab  = ENGRAM_FIX_COMPRESSED_VOCAB;
    L.pad_compressed_id = ENGRAM_FIX_PAD_COMPRESSED_ID;
    L.n_layers          = ENGRAM_FIX_N_LAYERS;
    L.token_map         = ENGRAM_FIX_TOKEN_MAP;
    L.multipliers       = &ENGRAM_FIX_MULT[0][0];
    L.primes            = &ENGRAM_FIX_PRIMES[0][0];
    L.offsets           = &ENGRAM_FIX_OFFSETS[0][0];
    L.num_embeddings    = ENGRAM_FIX_NUM_EMBEDDINGS;
    return L;
}

static void case_token_id_out_of_range(void) {
    pulsar_engram_layout L = make_layout();
    int32_t ids[ENGRAM_FIX_N_POS];
    memcpy(ids, ENGRAM_FIX_SEQ, sizeof ids);
    ids[3] = ENGRAM_FIX_N_VOCAB; /* one past the map's end */
    uint32_t cols[ENGRAM_FIX_N_COLS];
    pulsar_engram_hash_pos(&L, 0, ids, ENGRAM_FIX_N_POS, 3, cols);
}

static void case_compressed_id_out_of_range(void) {
    pulsar_engram_layout L = make_layout();
    int32_t map[ENGRAM_FIX_N_VOCAB];
    memcpy(map, ENGRAM_FIX_TOKEN_MAP, sizeof map);
    map[2] = (int32_t)ENGRAM_FIX_COMPRESSED_VOCAB; /* one past the compressed vocab */
    L.token_map = map;
    uint32_t cols[ENGRAM_FIX_N_COLS];
    pulsar_engram_hash_pos(&L, 0, ENGRAM_FIX_SEQ, ENGRAM_FIX_N_POS, 2, cols);
}

int main(void) {
    pulsar_engram_layout L = make_layout();

    printf("engram hash gate: %d layers x %d positions x %d columns from the V4.1 layout\n",
           ENGRAM_FIX_N_LAYERS, ENGRAM_FIX_N_POS, ENGRAM_FIX_N_COLS);

    for (uint32_t layer = 0; layer < ENGRAM_FIX_N_LAYERS; layer++) {
        for (uint32_t pos = 0; pos < ENGRAM_FIX_N_POS; pos++) {
            uint32_t cols[ENGRAM_FIX_N_COLS];
            memset(cols, 0xff, sizeof cols);
            pulsar_engram_hash_pos(&L, layer, ENGRAM_FIX_SEQ, ENGRAM_FIX_N_POS, pos, cols);

            for (uint32_t c = 0; c < ENGRAM_FIX_N_COLS; c++) {
                const uint32_t want = ENGRAM_FIX_EXPECT[layer][pos][c];
                /* A column index maps back to (order, head) unambiguously, so a
                 * failure can name which modulus/offset went wrong. */
                check(cols[c] == want,
                      "layer %u pos %u (token %d, src id %d) col %u (ngram %u, head %u): "
                      "got %u want %u",
                      layer, pos, ENGRAM_FIX_SEQ[pos], ENGRAM_FIX_SOURCE_IDS[pos], c,
                      c / ENGRAM_FIX_N_HEADS + 2, c % ENGRAM_FIX_N_HEADS, cols[c], want);
                check(cols[c] < ENGRAM_FIX_NUM_EMBEDDINGS[layer],
                      "layer %u pos %u col %u: %u is outside the table (%llu rows)",
                      layer, pos, c, cols[c],
                      (unsigned long long)ENGRAM_FIX_NUM_EMBEDDINGS[layer]);
            }
        }
    }

    expect_dies(case_token_id_out_of_range,
                "token id out of range", "token id outside the layout's token map");
    expect_dies(case_compressed_id_out_of_range,
                "compressed id out of range", "compressed id outside the layout's compressed vocab");

    printf("engram hash gate: %u checks, %u failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

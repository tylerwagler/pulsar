/* L183 PROBE DRIVER: prefill exactly N story tokens as ONE chunk, then exit.
 *
 * The chunk-mate census asks which prefill kernels are M-dependent: run this
 * driver twice (N and N + 1) with the engine's per-op activation dumps armed
 * (PULSAR_CUDA_GRAPH_DUMP_PREFIX / _NAME / _LAYER, steering.cpp) and compare
 * the first N rows of every dumped op, layer by layer.  The first op whose
 * INPUT rows are identical and whose OUTPUT rows differ is M-dependent; every
 * op before it is neutral at this shape.  The driver itself does nothing but
 * open, tokenize, prefill N tokens and close -- the instrument is the dump.
 *
 *   ./tests/prefill_chunk_census_probe MODEL N [CHUNK=8192] [FIRST=0] [MODE=sync|classic]
 *
 * MODE (L195): with FIRST > 0, "sync" resumes through pulsar_session_sync (since
 * L183 a cold prefill from the grid, so the second call reproduces the cold
 * chunking and the census sees nothing); "classic" continues from the
 * checkpoint as one chunk [FIRST, N) (gate_prefill_suffix_classic) -- the
 * off-grid chunk whose bytes the census is about.
 */
#include "gate_fixture.h"
#include "pulsar.h"
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1u);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f);
    buf[n] = 0;
    if (len_out) *len_out = (size_t)n;
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL N [CHUNK=8192] [FIRST=0]\n", argv[0]); return 2; }
    const int n = atoi(argv[2]);
    const uint32_t chunk = argc > 3 ? (uint32_t)atoi(argv[3]) : 8192u;
    /* FIRST > 0: sync FIRST tokens first, then N -- the second sync evaluates
     * [FIRST, N) as a chunk starting off position 0, the resume shape; its
     * dumps carry pos0 = FIRST. */
    const int first = argc > 4 ? atoi(argv[4]) : 0;
    const bool classic = argc > 5 && strcmp(argv[5], "classic") == 0;
    if (argc > 5 && !classic && strcmp(argv[5], "sync") != 0) { fprintf(stderr, "MODE is sync or classic\n"); return 2; }
    if (n < 1 || chunk < 16u || first < 0 || first >= n) { fprintf(stderr, "bad N, CHUNK or FIRST\n"); return 2; }

    pulsar_engine_options opt; memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    opt.prefill_chunk = chunk;
    opt.dspark_disable = true;
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }

    int rc = 1;
    pulsar_session *s = NULL;
    pulsar_tokens toks; memset(&toks, 0, sizeof toks);
    {
        size_t text_len = 0;
        char *text = read_file("tests/long_context_story_prompt.txt", &text_len);
        if (!text) { fprintf(stderr, "prompt file read failed (run from the repo root)\n"); goto done; }
        pulsar_tokenize_text(e, text, &toks);
        free(text);
        if (toks.len < n) { fprintf(stderr, "prompt has %d tokens, need %d\n", toks.len, n); goto done; }
        const int ctx = n + 64 > 8192 ? n + 64 : 8192;
        if (pulsar_session_create(&s, e, ctx) != 0) { fprintf(stderr, "session failed\n"); goto done; }
        char err[256];
        pulsar_tokens p = toks;
        if (first > 0) {
            p.len = first;
            if (pulsar_session_sync(s, &p, err, sizeof err) != 0) { fprintf(stderr, "sync(first): %s\n", err); goto done; }
        }
        p.len = n;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (classic && first > 0) {
            if (!gate_prefill_suffix_classic(s, &toks, first, n, err, sizeof err)) { fprintf(stderr, "classic: %s\n", err); goto done; }
        } else if (pulsar_session_sync(s, &p, err, sizeof err) != 0) { fprintf(stderr, "sync: %s\n", err); goto done; }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double secs = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
        printf("CENSUS DRIVER: prefilled %d tokens in chunks of %u (%s%s): %.3f s, %.0f tok/s\n", n, chunk,
               (uint32_t)(n - first) <= chunk ? "one chunk" : "several chunks",
               first > 0 ? (classic ? ", classic continuation after a first sync" : ", after a first sync") : "",
               secs, (double)(n - first) / secs);
        /* L195: the frontier logits, hashed (FNV-1a over the f32 bytes) -- the
         * end-to-end answer to "did this chunking change the bytes" without a
         * per-layer dump.  Two runs that print the same hash agree on every
         * one of the 129280 logits. */
        {
            const int width = pulsar_engine_logits_width(e);
            float *row = (float *)malloc((size_t)width * sizeof(float));
            if (row && pulsar_session_copy_logits(s, row, width) == width) {
                uint64_t h = 1469598103934665603ull;
                const unsigned char *b = (const unsigned char *)row;
                for (size_t i = 0; i < (size_t)width * sizeof(float); i++) { h ^= b[i]; h *= 1099511628211ull; }
                int am = 0; for (int i = 1; i < width; i++) if (row[i] > row[am]) am = i;
                printf("CENSUS LOGITS: frontier row %d, fnv1a %016llx, argmax %d (%.6f)\n", n - 1,
                       (unsigned long long)h, am, (double)row[am]);
            }
            free(row);
        }
        rc = 0;
    }
done:
    pulsar_tokens_free(&toks);
    if (s) pulsar_session_free(s);
    pulsar_engine_close(e);
    return rc;
}

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
 *   ./tests/prefill_chunk_census_probe MODEL N [CHUNK=8192]
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL N [CHUNK=8192]\n", argv[0]); return 2; }
    const int n = atoi(argv[2]);
    const uint32_t chunk = argc > 3 ? (uint32_t)atoi(argv[3]) : 8192u;
    if (n < 1 || chunk < 16u) { fprintf(stderr, "bad N or CHUNK\n"); return 2; }

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
        pulsar_tokens p = toks; p.len = n;
        if (pulsar_session_sync(s, &p, err, sizeof err) != 0) { fprintf(stderr, "sync: %s\n", err); goto done; }
        printf("CENSUS DRIVER: prefilled %d tokens in chunks of %u (%s)\n", n, chunk,
               (uint32_t)n <= chunk ? "one chunk" : "several chunks");
        rc = 0;
    }
done:
    pulsar_tokens_free(&toks);
    if (s) pulsar_session_free(s);
    pulsar_engine_close(e);
    return rc;
}

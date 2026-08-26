/* L112 inc C bisection probe: localize the batch-shape numeric drift.
 *
 * Runs ONE decode_mixed sweep in one of two modes and exits; the caller sets
 * the PULSAR_CUDA_GRAPH_DUMP_* envs and diffs the dumped stage tensors row-wise
 * between modes to find the FIRST kernel whose run-0 rows depend on the batch
 * shape:
 *
 *   solo : single K-row run  (bank 0, story offset 0, positions C0..C0+K)
 *   cob  : two K-row runs    (bank 0 offset 0, bank 1 offset 700)
 *
 * Run 0's rows sit at batch rows [0,K) in BOTH modes, and every dump stamps
 * pos = pos0 = C0, so equal filenames compare directly (solo file holds K rows,
 * cob holds 2K; compare the first K).
 *
 * K = 257 (NOT ratio-aligned): both modes take the per-token compressor path,
 * so the known replay/per-token kernel split cannot alias as shape drift.
 *
 *   usage: PULSAR_MSEQ_BANKS=2 ./tests/cobatch_bisect_probe MODEL solo|cob
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pulsar.h"
#include "pulsar_engine_internal.h"

#define C0 128
#define K4 257
#define OFFB 700

static pulsar_engine *g_e;
static pulsar_tokens g_toks;

static char *read_file(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)s + 1);
    if (!b || fread(b, 1, (size_t)s, f) != (size_t)s) { fclose(f); free(b); return NULL; }
    fclose(f); b[s] = 0; if (n) *n = (size_t)s; return b;
}

static bool populate(pulsar_session *s, uint32_t bank, int off) {
    pulsar_gpu_graph *g = &s->graph; char e[256];
    if (g->banks.n_banks && !gpu_graph_bank_repoint(g, bank)) return false;
    pulsar_session_invalidate(s);
    pulsar_tokens p = { .v = g_toks.v + off, .len = C0, .cap = C0 };
    if (pulsar_session_sync(s, &p, e, sizeof e) != 0) {
        fprintf(stderr, "populate bank %u: %s\n", bank, e); return false;
    }
    gpu_graph_bank_counters_capture(g, bank);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL solo|cob\n", argv[0]); return 2; }
    const bool cob = strcmp(argv[2], "cob") == 0;
    pulsar_engine_options o; memset(&o, 0, sizeof o);
    o.model_path = argv[1]; o.backend = PULSAR_BACKEND_CUDA;
    if (pulsar_engine_open(&g_e, &o) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    size_t tl = 0; char *txt = read_file("tests/long_context_story_prompt.txt", &tl);
    if (!txt) { fprintf(stderr, "prompt read failed\n"); return 1; }
    memset(&g_toks, 0, sizeof g_toks); pulsar_tokenize_text(g_e, txt, &g_toks); free(txt);
    if (OFFB + C0 + K4 > g_toks.len) { fprintf(stderr, "prompt too short\n"); return 1; }

    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) { fprintf(stderr, "session failed\n"); return 1; }
    if (s->graph.banks.n_banks < (cob ? 2u : 1u)) {
        fprintf(stderr, "pool too small (set PULSAR_MSEQ_BANKS=2)\n"); return 1;
    }
    if (!populate(s, 0, 0)) return 1;
    if (cob && !populate(s, 1, OFFB)) return 1;

    const int vocab = (int)PULSAR_N_VOCAB;
    const int nrows = cob ? 2 * K4 : K4;
    pulsar_multiseq_req *rq = (pulsar_multiseq_req *)malloc((size_t)nrows * sizeof(*rq));
    for (int j = 0; j < K4; j++) { rq[j].bank = 0; rq[j].pos = C0 + j; rq[j].token = g_toks.v[C0 + j]; }
    if (cob)
        for (int j = 0; j < K4; j++) {
            rq[K4 + j].bank = 1; rq[K4 + j].pos = C0 + j; rq[K4 + j].token = g_toks.v[OFFB + C0 + j];
        }
    float *lg = (float *)malloc((size_t)(cob ? 2 : 1) * vocab * sizeof(float));
    uint32_t nr = 0; char e[256];
    if (pulsar_session_decode_mixed(s, rq, (uint32_t)nrows, lg, (cob ? 2 : 1) * vocab,
                                 &nr, 0u, e, sizeof e) != 0) {
        fprintf(stderr, "sweep failed: %s\n", e); return 1;
    }
    printf("PROBE %s: sweep ok, n_runs=%u, run0 logit[0]=%.9e\n",
           cob ? "cob" : "solo", nr, (double)lg[0]);
    free(lg); free(rq);
    pulsar_session_free(s);
    pulsar_engine_close(g_e);
    return 0;
}

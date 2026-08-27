/* L120 rewind-frontier gate — position-truth of the compressor frontiers
 * across pulsar_session_rewind.
 *
 * The bug this pins: pulsar_session::rewind trimmed tokens but never
 * reconciled layer_n_comp/layer_n_index_comp, so any rewind crossing a
 * (pos+1)%ratio emit boundary left the pair one row high.  Latent since
 * 6de76e3 (classic re-assigned the counters absolutely before anything
 * validated); first asserted by the unified lane's position-true admission
 * check, first fired by production anthropic live continuations (the one
 * serving path with no counter-rebuilding re-route between the rewind and
 * the check).  See pulsar-notes rows/L120.md.
 *
 * Shape (deterministic — no residue luck, unlike the serving repro):
 *   1. prefill to a frontier past several ratio-4 boundaries;
 *   2. rewind one token at a time across every mod-4 residue, asserting
 *      after each that EVERY compressing layer's counters equal pos/ratio
 *      (ratio-128 layers get clamped across their own boundary too);
 *   3. the incident anatomy in miniature: after a boundary-crossing rewind,
 *      sync a continuation (the incremental prefill path — the one that
 *      preserved the skew in production) and re-assert at the new frontier.
 *
 * MODEL-DEPENDENT, GPU-resident.  Run under the memory discipline.  NOT part
 * of `make test`.
 *
 * usage: ./tests/rewind_frontier_gate MODEL
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "pulsar_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "REWIND FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); g_fail = 1; } } while (0)

static char *read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, fp) != (size_t)n) { fclose(fp); free(buf); return NULL; }
    fclose(fp);
    buf[n] = '\0';
    if (len_out) *len_out = (size_t)n;
    return buf;
}

/* Every compressing layer's frontier pair must equal pos/ratio exactly —
 * position-truth, the same predicate the multiseq admission check enforces. */
static void check_frontiers(pulsar_session *s, int pos, const char *what) {
    pulsar_gpu_graph *g = &s->graph;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t want = (uint32_t)pos / ratio;
        CHECK(g->layer_n_comp[il] == want,
              "%s: layer %u n_comp %u want %u (pos %d ratio %u)",
              what, il, g->layer_n_comp[il], want, pos, ratio);
        if (ratio == 4) {
            CHECK(g->layer_n_index_comp[il] == want,
                  "%s: layer %u n_index_comp %u want %u (pos %d)",
                  what, il, g->layer_n_index_comp[il], want, pos);
        }
    }
}

static bool sync_prefix(pulsar_session *s, pulsar_tokens *toks, int len) {
    pulsar_tokens p;
    memset(&p, 0, sizeof(p));
    p.v = toks->v;
    p.len = p.cap = len;
    char err[256];
    const int rc = pulsar_session_sync(s, &p, err, sizeof(err));
    if (rc != 0) fprintf(stderr, "sync to %d failed: %s\n", len, err);
    return rc == 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }

    pulsar_engine *e = NULL;
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }

    size_t text_len = 0;
    char *text = read_file("tests/long_context_story_prompt.txt", &text_len);
    if (!text) { fprintf(stderr, "prompt file read failed\n"); return 1; }
    pulsar_tokens toks;
    memset(&toks, 0, sizeof(toks));
    pulsar_tokenize_text(e, text, &toks);
    free(text);
    if (toks.len < 300) { fprintf(stderr, "prompt too short\n"); return 1; }

    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, e, 4096) != 0) { fprintf(stderr, "session create failed\n"); return 1; }

    /* 1. Frontier past several ratio-4 boundaries AND one ratio-128 boundary,
     * on a boundary-interior position so the very first rewind crosses. */
    const int top = 130;
    if (!sync_prefix(s, &toks, top)) return 1;
    check_frontiers(s, top, "post-prefill baseline");

    /* 2. Walk the frontier down one token at a time: eight rewinds cover
     * every mod-4 residue twice, and 129->127 crosses the ratio-128
     * boundary.  Unfixed rewind leaves the counters at the old frontier on
     * every boundary cross. */
    for (int pos = top - 1; pos >= top - 8; pos--) {
        pulsar_session_rewind(s, pos);
        CHECK(pulsar_session_pos(s) == pos, "rewind: pos %d want %d",
              pulsar_session_pos(s), pos);
        check_frontiers(s, pos, "post-rewind");
    }

    /* 3. Incident anatomy: from the rewound frontier (122), continue with an
     * incremental prefill — the serving path that carried the stale counter
     * to the admission check — and re-assert position-truth at the new
     * frontier.  On the unfixed binary the walk above already failed, and
     * this leg fails too (stale base + incremental emits). */
    const int cont = 199;
    if (!sync_prefix(s, &toks, cont)) return 1;
    check_frontiers(s, cont, "post-continuation");

    /* One more boundary-crossing rewind after the continuation, then a
     * second continuation: rewind machinery must stay position-true under
     * repeated cycles, not just once from a fresh prefill. */
    pulsar_session_rewind(s, cont - 4);
    check_frontiers(s, cont - 4, "second rewind");
    if (!sync_prefix(s, &toks, 260)) return 1;
    check_frontiers(s, 260, "second continuation");

    pulsar_session_free(s);
    pulsar_engine_close(e);

    if (g_fail) { fprintf(stderr, "REWIND GATE: FAIL\n"); return 1; }
    printf("REWIND GATE: PASS\n");
    return 0;
}

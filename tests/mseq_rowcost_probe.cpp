/* L116 follow-on: row-cost curve for the shared multiseq decode forward.
 *
 * Times pulsar_session_decode_mixed (ALL_ROWS head) over a B x R grid --
 * B banks, R teacher-forced consecutive rows per bank (the spec-round
 * chain shape) -- at fixed per-bank depth. This prices the dynamic row-cap
 * design (L049 increment 2, the cost-table argmax): where the "<=16 rows is
 * near-flat" assumption stops holding, and whether the >16-row crossover
 * into the MMA matmul arm has a cliff.
 *
 * Deliberately NOT timed: spec capture (arm_capture stays 0 -- its buffers
 * are the thing the increment resizes) and host round bookkeeping. Chains
 * ADVANCE across iterations (the gate-proven driver contract); per-cell
 * depth drifts by ITERS*R rows, so min_ms (earliest iterations, base depth)
 * is the drift-free statistic and med_ms bounds it from above.
 *
 * MODEL-DEPENDENT, GB10, manual:
 *   PULSAR_MSEQ_BANKS=4 ./tests/mseq_rowcost_probe MODEL [DEPTH ...]
 * (from the repo root -- reads tests/long_context_story_prompt.txt)
 * Output: one line per cell -- depth, B, R, total rows, median/min ms per
 * sweep, rows-per-ms -- machine-greppable "ROWCOST" prefix.
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PROBE_MAX_B 4
#define PROBE_MAX_R 16
#define PROBE_ITERS 12

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

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

static unsigned long long fnv1a(const void *buf, size_t n) {
    const unsigned char *p2 = (const unsigned char *)buf;
    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= p2[i]; h *= 1099511628211ull; }
    return h;
}

static int cmp_dbl(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL [DEPTH ...]\n", argv[0]);
        return 2;
    }
    int depths[4], n_depths = 0;
    for (int i = 2; i < argc && n_depths < 4; i++) depths[n_depths++] = atoi(argv[i]);
    if (n_depths == 0) { depths[0] = 512; depths[1] = 2048; n_depths = 2; }

    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    char err[256];
    if (pulsar_engine_open(&e, &opt) != 0) {
        fprintf(stderr, "engine open failed\n");
        return 1;
    }

    size_t text_len = 0;
    char *text = read_file("tests/long_context_story_prompt.txt", &text_len);
    if (!text) { fprintf(stderr, "cannot read tests/long_context_story_prompt.txt\n"); return 1; }
    pulsar_tokens toks;
    memset(&toks, 0, sizeof(toks));
    pulsar_tokenize_text(e, text, &toks);
    if (toks.len < 4096) {
        fprintf(stderr, "prompt too short (%d)\n", toks.len);
        return 1;
    }
    free(text);
    const int vocab = (int)PULSAR_N_VOCAB;

    /* PROBE_BS / PROBE_RS: comma-separated grid restriction, for profiling a
     * single cell under nsys (e.g. PROBE_BS=1 PROBE_RS=1). Default = full grid. */
    int Bs[8] = {1, 2, 4}, n_bs = 3;
    int Rs[8] = {1, 2, 4, 8, 16}, n_rs = 5;
    const char *bs_env = getenv("PROBE_BS");
    if (bs_env && *bs_env) {
        n_bs = 0;
        for (const char *p2 = bs_env; *p2 && n_bs < 8;) {
            Bs[n_bs++] = atoi(p2);
            while (*p2 && *p2 != ',') p2++;
            if (*p2 == ',') p2++;
        }
    }
    const char *rs_env = getenv("PROBE_RS");
    if (rs_env && *rs_env) {
        n_rs = 0;
        for (const char *p2 = rs_env; *p2 && n_rs < 8;) {
            Rs[n_rs++] = atoi(p2);
            while (*p2 && *p2 != ',') p2++;
            if (*p2 == ',') p2++;
        }
    }

    for (int di = 0; di < n_depths; di++) {
        const int depth = depths[di];
        if ((PROBE_MAX_B - 1) * 137 + depth > toks.len) {
            fprintf(stderr, "depth %d too deep for prompt (%d)\n", depth, toks.len);
            continue;
        }
        /* One session per depth; ctx must hold depth + the deepest chain. */
        pulsar_session *s = NULL;
        const int ctx = depth + 512 < 4096 ? 4096 : depth + 512;
        if (pulsar_session_create(&s, e, ctx) != 0) {
            fprintf(stderr, "session create failed\n");
            return 1;
        }
        pulsar_gpu_graph *g = &s->graph;
        if ((int)gpu_graph_bank_pool_count(g) < PROBE_MAX_B) {
            fprintf(stderr, "pool too small: %u < %d (set PULSAR_MSEQ_BANKS)\n",
                    gpu_graph_bank_pool_count(g), PROBE_MAX_B);
            return 1;
        }
        int feed_tok[PROBE_MAX_B];
        for (int k = 0; k < PROBE_MAX_B; k++) {
            if (g->banks.n_banks && !gpu_graph_bank_repoint(g, (uint32_t)k)) {
                fprintf(stderr, "repoint %d failed\n", k);
                return 1;
            }
            pulsar_session_invalidate(s);
            pulsar_tokens p;
            memset(&p, 0, sizeof(p));
            p.v = toks.v + k * 137;   /* overlapping slices: content irrelevant */
            p.len = p.cap = depth;
            if (pulsar_session_sync(s, &p, err, sizeof(err)) != 0) {
                fprintf(stderr, "prefill bank %d depth %d failed: %s\n", k, depth, err);
                return 1;
            }
            gpu_graph_bank_counters_capture(g, (uint32_t)k);
            feed_tok[k] = pulsar_session_argmax(s);
        }

        float *logits = (float *)malloc((size_t)PROBE_MAX_B * PROBE_MAX_R * (size_t)vocab * sizeof(float));
        if (!logits) { fprintf(stderr, "oom\n"); return 1; }

        for (int bi = 0; bi < n_bs; bi++) {
            for (int ri = 0; ri < n_rs; ri++) {
                const int B = Bs[bi], R = Rs[ri];
                const uint32_t rows = (uint32_t)(B * R);
                pulsar_multiseq_req reqs[PROBE_MAX_B * PROBE_MAX_R];
                double ms[PROBE_ITERS];
                unsigned long long cell_hash = 0;
                bool cell_ok = true;
                for (int it = 0; it < PROBE_ITERS && cell_ok; it++) {
                    /* Chains advance: iteration it starts every bank at its
                     * committed frontier depth + it*R (rows committed by the
                     * previous iteration), exactly how the spec lane feeds
                     * consecutive rounds. */
                    uint32_t w = 0;
                    for (int k = 0; k < B; k++) {
                        for (int j = 0; j < R; j++) {
                            reqs[w].bank = (uint32_t)k;
                            reqs[w].pos = depth + it * R + j;
                            reqs[w].token = feed_tok[k];
                            w++;
                        }
                    }
                    uint32_t got = 0;
                    const double t0 = now_s();
                    const int rc = pulsar_session_decode_mixed(s, reqs, rows, logits,
                            rows * (uint32_t)vocab, &got,
                            PULSAR_MSEQ_HEAD_ALL_ROWS, err, sizeof(err));
                    const double t1 = now_s();
                    if (rc != 0 || got != rows) {
                        printf("ROWCOST depth=%d B=%d R=%d rows=%u REJECTED it=%d rc=%d got=%u err=%s\n",
                               depth, B, R, rows, it, rc, got, err);
                        cell_ok = false;
                        break;
                    }
                    ms[it] = (t1 - t0) * 1e3;
                    /* Differential witness: fold EVERY iteration's logits into
                     * one running hash (teacher-forced feed is deterministic,
                     * so iteration i's inputs are identical across runs). Two
                     * runs of the same cell must agree bitwise — including
                     * runs where later iterations take a replayed CUDA graph
                     * (L119 segments) while early ones ran eager. */
                    {
                        unsigned long long ih = fnv1a(logits,
                                (size_t)rows * (size_t)vocab * sizeof(float));
                        cell_hash = fnv1a(&ih, sizeof ih) ^ (cell_hash * 1099511628211ull);
                        /* PROBE_ITER_HASH=1: per-iteration hashes, to see at
                         * which iteration a capture/replay arm first diverges
                         * from the eager arm (dev diagnostic). */
                        static int iter_hash = -1;
                        if (iter_hash < 0)
                            iter_hash = getenv("PROBE_ITER_HASH") != NULL;
                        if (iter_hash)
                            printf("ITERHASH depth=%d B=%d R=%d it=%d h=%016llx\n",
                                   depth, B, R, it, ih);
                    }
                }
                /* Reset the touched banks for the next cell: repoint +
                 * re-prefill from scratch keeps every cell's starting frontier
                 * at exactly `depth` without leaning on rewind-after-multiseq
                 * (an engine path the gates do not exercise). */
                for (int k = 0; k < B; k++) {
                    if (g->banks.n_banks && !gpu_graph_bank_repoint(g, (uint32_t)k)) {
                        fprintf(stderr, "reset repoint %d failed\n", k);
                        return 1;
                    }
                    pulsar_session_invalidate(s);
                    pulsar_tokens p;
                    memset(&p, 0, sizeof(p));
                    p.v = toks.v + k * 137;
                    p.len = p.cap = depth;
                    if (pulsar_session_sync(s, &p, err, sizeof(err)) != 0) {
                        fprintf(stderr, "reset prefill bank %d failed: %s\n", k, err);
                        return 1;
                    }
                    gpu_graph_bank_counters_capture(g, (uint32_t)k);
                    feed_tok[k] = pulsar_session_argmax(s);
                }
                if (!cell_ok) continue;
                qsort(ms, PROBE_ITERS, sizeof(double), cmp_dbl);
                const double med = ms[PROBE_ITERS / 2], mn = ms[0];
                printf("ROWCOST depth=%d B=%d R=%d rows=%u med_ms=%.3f min_ms=%.3f rows_per_ms=%.2f hash=%016llx\n",
                       depth, B, R, rows, med, mn, (double)rows / med, cell_hash);
                fflush(stdout);
            }
        }
        free(logits);
        pulsar_session_free(s);
    }
    pulsar_tokens_free(&toks);
    pulsar_engine_close(e);
    return 0;
}

/* Cost of extending a live checkpoint by a short suffix -- the
 * pulsar_session_sync resume path.
 *
 * WHY THIS EXISTS. Until L131 this path had a crossover: suffixes below
 * gpu_graph_resume_prefill_min_tokens() were evaluated by a separate
 * single-token encoder, everything else by batched prefill. That constant was
 * carried over from an M3 Max measurement, was never re-measured on GB10, and
 * NOTHING IN tests/ TOUCHED THE PATH IT GATED. It was changed twice on
 * reasoning alone before anyone built this.
 *
 * The crossover and the second encoder are now gone -- every positive suffix
 * takes batched prefill -- so this is no longer an A/B. It is a regression
 * guard: a resume should cost roughly one batched chunk regardless of K, and
 * must NOT grow linearly in K (linear growth is the signature of a
 * token-at-a-time path having crept back in).
 *
 * Measured on GB10 2026-09-01, medians of 30 reps at base=512:
 *     K=1  57.9 ms    K=2  80.8 ms    K=3  98.0 ms
 *     K=4 115.2 ms    K=6 127.0 ms
 * Sub-linear in K, as batched prefill should be. For contrast the deleted
 * single-token path was 54.4 ms * K exactly -- 326 ms at K=6.
 *
 * Not wired into `make gates`: it wants a model and reports timings rather
 * than a pass/fail, so it is a probe, not a gate.
 */
#include "pulsar.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL [suffix_tokens=1] [reps=40] [base=512]\n", argv[0]);
        return 2;
    }
    const int K    = argc > 2 ? atoi(argv[2]) : 1;
    const int REPS = argc > 3 ? atoi(argv[3]) : 40;
    const int BASE = argc > 4 ? atoi(argv[4]) : 512;

    pulsar_engine *e = NULL;
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    if (pulsar_engine_open(&e, &opt) != 0) {
        fprintf(stderr, "engine open failed\n");
        return 1;
    }

    /* A deterministic token stream: ids that exist in any vocab. Content does
     * not matter here -- the cost of a resume is set by the suffix length and
     * the encoder, not by which tokens they are. */
    const int total = BASE + K * REPS + 8;
    pulsar_tokens toks;
    memset(&toks, 0, sizeof(toks));
    for (int i = 0; i < total; i++) {
        const int id = 1000 + (i * 7919) % 20000;
        pulsar_tokens_push(&toks, id);
    }

    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, e, 8192) != 0) {
        fprintf(stderr, "session create failed\n");
        return 1;
    }

    char err[256];
    pulsar_tokens cur;
    memset(&cur, 0, sizeof(cur));
    for (int i = 0; i < BASE; i++) pulsar_tokens_push(&cur, toks.v[i]);
    if (pulsar_session_sync(s, &cur, err, sizeof(err)) != 0) {
        fprintf(stderr, "base sync failed: %s\n", err);
        return 1;
    }

    printf("resume-suffix probe: K=%d reps=%d base=%d (batched resume, L131)\n",
           K, REPS, BASE);

    double *ms = (double *)calloc((size_t)REPS, sizeof(double));
    int n = 0;
    for (int r = 0; r < REPS; r++) {
        for (int k = 0; k < K; k++) pulsar_tokens_push(&cur, toks.v[cur.len]);
        const double t0 = now_sec();
        const int rc = pulsar_session_sync(s, &cur, err, sizeof(err));
        const double t1 = now_sec();
        if (rc != 0) { fprintf(stderr, "sync failed at rep %d: %s\n", r, err); return 1; }
        ms[n++] = (t1 - t0) * 1000.0;
    }

    qsort(ms, (size_t)n, sizeof(double), cmp_double);
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += ms[i];
    /* Median and the trimmed mean: a resume that happens to cross a chunk or
     * compressor boundary is genuinely more expensive, and one such rep should
     * not decide the comparison. */
    const double med = ms[n / 2];
    double tsum = 0.0; int tn = 0;
    for (int i = n / 10; i < n - n / 10; i++) { tsum += ms[i]; tn++; }
    printf("  n=%d  min=%.3f ms  median=%.3f ms  trimmed_mean=%.3f ms  max=%.3f ms  mean=%.3f ms\n",
           n, ms[0], med, tn ? tsum / tn : 0.0, ms[n - 1], sum / n);
    return 0;
}

/* CHUNK-NEUTRALITY GATE (L183): a prompt's logits do not depend on how it was
 * chunked or resumed.
 *
 * The same 8600 tokens are prefilled under seven schedules on the production
 * chunk grid (prefill_chunk 4096):
 *   A  sync(8600)              -> chunks [0,4096) [4096,8192) [8192,8600)   the cold prompt
 *   B  sync(6),    sync(8600)  -> a 6-row first chunk, then the rest
 *   C  sync(2048), sync(8600)  -> two mid-size chunks
 *   D  sync(4000), sync(8600)  -> a warm-fork resume at an off-grid frontier
 *   E  sync(4500), sync(8600)  -> a resume PAST a grid boundary: the second sync
 *                                 must rewind to 4096 and restore the compressor
 *                                 state the first sync snapshotted when it crossed
 *                                 4096 -- the grid-snapshot path
 *   F  sync(8100), eval x200, sync(N) -> DECODE crosses the 8192 grid, then a resume:
 *                                the latest snapshot is 4096, so [4096, N) is redone
 *                                (the first L183 cut demanded a snapshot at 8192
 *                                and re-prefilled from 0 -- dogfood 2026-09-05, L194)
 *   G  sync(4000), eval x200, sync(N) -> decode crosses 4096 with NO snapshot: cold from 0
 * (sync evaluates only the suffix when the checkpoint is a prefix, so B..E are
 * exactly what a prefix-cache hit or a warm fork does; F and G are a turn after
 * the model generated across a grid boundary.)  After each schedule the
 * frontier logits are read, then ONE classic decode step is taken and its logits
 * read too -- the decode step consumes the ratio-4 compressor state the prefill
 * left behind (L181), so a wrong state shows there.  B..E must be byte-identical
 * to A on both rows.
 *
 * Why it holds: since L183 a resume is a cold prefill from the last grid
 * boundary at or below its checkpoint (pulsar_session_sync), so every chunk
 * boundary, row count and row offset a kernel sees is the cold prefill's.  It
 * has to be that way: a chunk's bytes depend on its row count (the plain-weight
 * GEMM picked split-K by M), on a row's offset within the call (the tall-K HC-mix
 * kernel), and on how the compressed rows a chunk attends were split between the
 * cache and the chunk (the mixed attention tier) -- L183's censuses.  Before the
 * rule every schedule here differed from A from layer 0 on.  This gate is what
 * says "a resume is the same computation as a cold prefill" for the served
 * engine; the byte gate pins one chunking, this pins that the chunking does not
 * matter.
 *
 *   ./tests/chunk_neutrality_gate MODEL
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "gate_entry.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GATE_N 8600
#define GATE_CTX 8704
#define GATE_SCHEDULES 7
#define GATE_EVALS 200   /* decode steps that carry schedules F and G across a grid boundary */

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

/* Run one schedule in a fresh session: sync to `first` tokens (0 = skip), eval
 * the next `evals` prompt tokens one by one (the decode lane carrying the
 * frontier forward, as a generation does), sync to GATE_N, read the frontier
 * row, eval token GATE_N, read that row.  `origin` is where that last sync
 * must have resumed from (pulsar_session_resume_origin): the schedule is only
 * the schedule it claims to be if the resume started there. */
static int run_schedule(pulsar_engine *e, const pulsar_tokens *toks, int first, int evals, int origin,
                        int width, float *frontier, float *decoded, char *err, size_t errlen) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, e, GATE_CTX) != 0) { snprintf(err, errlen, "session create failed"); return 1; }
    int rc = 1;
    pulsar_tokens p = *toks;
    if (first > 0) {
        p.len = first;
        if (pulsar_session_sync(s, &p, err, errlen) != 0) goto done;
    }
    for (int i = 0; i < evals; i++) {
        if (pulsar_session_eval(s, toks->v[first + i], err, errlen) != 0) goto done;
    }
    p.len = GATE_N;
    if (pulsar_session_sync(s, &p, err, errlen) != 0) goto done;
    if (pulsar_session_resume_origin(s) != origin) {
        snprintf(err, errlen, "resume origin %d, schedule expects %d (the resume did not start where this schedule says)",
                 pulsar_session_resume_origin(s), origin);
        goto done;
    }
    if (pulsar_session_copy_logits(s, frontier, width) != width) { snprintf(err, errlen, "frontier logits read failed"); goto done; }
    if (pulsar_session_eval(s, toks->v[GATE_N], err, errlen) != 0) goto done;
    if (pulsar_session_copy_logits(s, decoded, width) != width) { snprintf(err, errlen, "decode logits read failed"); goto done; }
    rc = 0;
done:
    pulsar_session_free(s);
    return rc;
}

static int compare_rows(const char *what, const char *label, const float *ref, const float *cur, int width) {
    uint64_t bad = 0; double worst = 0.0; int worst_i = -1;
    for (int i = 0; i < width; i++) {
        if (memcmp(&ref[i], &cur[i], sizeof(float)) != 0) {
            bad++;
            const double d = fabs((double)ref[i] - (double)cur[i]);
            if (d > worst) { worst = d; worst_i = i; }
        }
    }
    if (bad == 0) {
        printf("  %-50s %-8s %d logits byte-identical to schedule A  OK\n", label, what, width);
        return 0;
    }
    printf("  %-50s %-8s %llu of %d logits DIFFER from schedule A (worst |d| %.3e at %d: A=%.6f cur=%.6f)  FAIL\n",
           label, what, (unsigned long long)bad, width, worst, worst_i, (double)ref[worst_i], (double)cur[worst_i]);
    return 1;
}

int GATE_ENTRY(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }
    pulsar_engine_options opt; memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    opt.prefill_chunk = 4096;   /* the production grid, as the byte gate */
    opt.dspark_disable = true;
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }

    int rc = 1;
    pulsar_tokens toks; memset(&toks, 0, sizeof toks);
    float *rows = NULL;
    {
        size_t text_len = 0;
        char *text = read_file("tests/long_context_story_prompt.txt", &text_len);
        if (!text) { fprintf(stderr, "prompt file read failed (run from the repo root)\n"); goto done; }
        pulsar_tokenize_text(e, text, &toks);
        free(text);
        if (toks.len < GATE_N + 1) { fprintf(stderr, "prompt has %d tokens, need %d\n", toks.len, GATE_N + 1); goto done; }
        const int width = pulsar_engine_logits_width(e);
        /* GATE_SCHEDULES x (frontier, decoded) */
        rows = (float *)malloc((size_t)(2 * GATE_SCHEDULES) * (size_t)width * sizeof(float));
        if (!rows) goto done;
        struct { const char *label; int first; int evals; int origin; } sched[GATE_SCHEDULES] = {
            {"A: cold [0,4096) [4096,8192) [8192,8600)", 0, 0, 0},
            {"B: sync 6, then 8600", 6, 0, 0},
            {"C: sync 2048, then 8600", 2048, 0, 0},
            {"D: resume at 4000, then 8600", 4000, 0, 0},
            {"E: resume at 4500 (past the 4096 grid), then 8600", 4500, 0, 4096},
            {"F: sync 8100, decode 200 across 8192, resume from 4096", 8100, GATE_EVALS, 4096},
            {"G: sync 4000, decode 200 across 4096, no snapshot: cold", 4000, GATE_EVALS, 0},
        };
        char err[256];
        printf("chunk-neutrality gate: %d tokens, prefill chunk %u, %d schedules; frontier row + one decode step each\n",
               GATE_N, opt.prefill_chunk, GATE_SCHEDULES);
        for (int k = 0; k < GATE_SCHEDULES; k++) {
            if (run_schedule(e, &toks, sched[k].first, sched[k].evals, sched[k].origin, width,
                             rows + (size_t)(2 * k) * width,
                             rows + (size_t)(2 * k + 1) * width, err, sizeof err) != 0) {
                fprintf(stderr, "CHUNK-NEUTRALITY GATE: schedule %s failed: %s\n", sched[k].label, err);
                goto done;
            }
        }
        int fails = 0;
        for (int k = 1; k < GATE_SCHEDULES; k++) {
            fails += compare_rows("frontier", sched[k].label, rows, rows + (size_t)(2 * k) * width, width);
            fails += compare_rows("decode+1", sched[k].label, rows + (size_t)width, rows + (size_t)(2 * k + 1) * width, width);
        }
        printf(fails == 0 ? "CHUNK-NEUTRALITY GATE: PASS\n" : "CHUNK-NEUTRALITY GATE: FAIL (%d rows differ)\n", fails);
        rc = fails == 0 ? 0 : 1;
    }
done:
    free(rows);
    pulsar_tokens_free(&toks);
    gate_engine_close(e);
    return rc;
}

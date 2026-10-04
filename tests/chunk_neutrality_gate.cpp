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
 *   F  sync(8100), eval x200, sync(N) -> DECODE crosses 8192; the resume redoes the decoded
 *                                tokens from the prefill's last grid point (8064), because
 *                                decode rows are the decode kernels' and a cold prefill's
 *                                are not (L195)
 *   G  sync(4000), eval x200, sync(N) -> same from 3968
 *   K  sync(4000), eval x4600, cut to 4100, sync(N) -> the reasoning-stripping
 *                                client (L264): decode runs past the raw ring's
 *                                reach (raw_cap 4352), so the raw slots the
 *                                resume's window needs were overwritten by
 *                                decode rows the cut then removed; the resume
 *                                must still equal the cold prefill
 *   L  sync(N), restore the grid checkpoint at 4096, sync(N) -> L264's resume:
 *                                the prefill left a checkpoint at every grid
 *                                point a chunk ended on; restoring one is a
 *                                resume from it
 *   M  K's history, restored to the checkpoint 3968 the first sync left -- the
 *                                reasoning-stripping turn resumed the L264 way
 *   N  sync(N), restore the checkpoint at 8576 -- the split the prefill makes at
 *                                the last grid point below the prompt's end
 *   O  sync(N), persist the session as a chain of disk SEGMENTS in a segment
 *                                store (L264 S4), restart (store reopened, session
 *                                fresh), find the chain by text, load it, sync(N):
 *                                resumes at 8576 -- the disk cache end to end
 *   P  M's history persisted as a segment chain to 3968 and resumed from it --
 *                                the reasoning-stripping turn coming back from disk
 * And one leg outside the table, X: a segment chain to 8576 restored, then
 * sync(8576) EXACTLY -- no new rows, so the sync must re-evaluate the last one
 * (the load moved the KV, not the logits) and match the cold prefill of 8576.
 * The resume grid is 128 (DeepSeek V4's, kv_state_ds4.cpp; L195): a resume rewinds to the grid
 * point below the PREFILL frontier, warms the compressor state up over the 32 tokens
 * before it, and redoes < 128 tokens plus whatever was generated since.  Nothing is
 * saved, so a resume below a cut works the same.
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
#include "pulsar_segstore.h"

#include <dirent.h>
#include <unistd.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GATE_N 8600
#define GATE_CTX 8704
#define GATE_SCHEDULES 16
#define GATE_EVALS 200   /* decode steps that carry schedules F and G across a grid boundary */
#define GATE_EVALS_PAST_RING 4600   /* schedule K: more than raw_cap (4352) decode rows past the prefill */

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

/* L264 S4: persist the session's grid checkpoints up to `limit` as a chain of
 * segments in a real segment STORE -- keyed by the rendered text of their
 * tokens, the way the server's disk KV cache keys them -- close the store and
 * throw the session away (the restart), reopen the store, find the chain by a
 * text lookup of the whole prompt alone, and load it into a fresh session.  Says
 * the chain it moved, so the log shows the segment path ran. */
struct segment_writer { pulsar_session *s; int G_prev, G; };
static int segment_write(FILE *fp, void *ud, char *err, size_t errlen) {
    const segment_writer *w = (const segment_writer *)ud;
    return pulsar_session_save_segment(w->s, fp, w->G_prev, w->G, NULL, err, errlen);
}
static char *render_span(pulsar_engine *e, const pulsar_tokens *t, int from, int to, size_t *len) {
    size_t cap = 4096, n = 0;
    char *out = (char *)malloc(cap);
    for (int i = from; out && i < to; i++) {
        size_t pl = 0;
        char *piece = pulsar_token_text(e, t->v[i], &pl);
        if (n + pl + 1 > cap) { while (n + pl + 1 > cap) cap *= 2; out = (char *)realloc(out, cap); }
        if (out) { memcpy(out + n, piece, pl); n += pl; }
        free(piece);
    }
    if (out) out[n] = 0;
    *len = n;
    return out;
}
static int segment_round_trip(pulsar_engine *e, pulsar_session **sp, const pulsar_tokens *prompt, int limit,
                              char *err, size_t errlen) {
    int Gs[64];
    int n = 0;
    for (int G = pulsar_session_checkpoint_best(*sp, limit); G > 0 && n < 64;
         G = pulsar_session_checkpoint_best(*sp, G - 1)) Gs[n++] = G;
    if (n == 0) { snprintf(err, errlen, "no grid checkpoint at or below %d to persist", limit); return 1; }
    const char *tmp = getenv("TMPDIR");
    char dir[512];
    snprintf(dir, sizeof dir, "%s/chunk_neutrality_segs.XXXXXX", tmp && *tmp ? tmp : "/tmp");
    if (!mkdtemp(dir)) { snprintf(err, errlen, "mkdtemp %.200s failed", dir); return 1; }
    const uint32_t model = pulsar_segstore_identity((uint32_t)pulsar_engine_model_id(e),
                                                   (uint32_t)pulsar_engine_routed_quant_bits(e));
    int rc = 1;
    pulsar_segstore *st = pulsar_segstore_open(dir, 0, model, NULL, NULL);
    if (!st) { snprintf(err, errlen, "segment store open failed"); goto out; }
    printf("  [segments]");
    {
        char parent[41] = "";
        const pulsar_tokens *hist = pulsar_session_tokens(*sp);
        for (int k = n - 1, prev = 0; k >= 0; prev = Gs[k], k--) {
            size_t tl = 0;
            char *text = render_span(e, hist, prev, Gs[k], &tl);
            segment_writer w = { *sp, prev, Gs[k] };
            char key[41];
            const bool ok = text && pulsar_segstore_put(st, parent, (uint32_t)prev, (uint32_t)Gs[k], text, tl,
                                                        pulsar_session_segment_bytes(*sp, prev, Gs[k]), segment_write,
                                                        0, NULL, &w, key, err, errlen);
            free(text);
            if (!ok) goto out;
            memcpy(parent, key, sizeof key);
            printf(" [%d,%d)", prev, Gs[k]);
        }
    }
    printf("\n");
    pulsar_segstore_close(st);
    st = NULL;
    pulsar_session_free(*sp);
    *sp = NULL;
    /* the restart: nothing carries over but the directory */
    if (!(st = pulsar_segstore_open(dir, 0, model, NULL, NULL))) { snprintf(err, errlen, "segment store reopen failed"); goto out; }
    if (pulsar_session_create(sp, e, GATE_CTX) != 0) { snprintf(err, errlen, "session create failed"); goto out; }
    {
        size_t tl = 0;
        char *text = render_span(e, prompt, 0, prompt->len, &tl);
        pulsar_segstore_seg chain[64];
        const int m = text ? pulsar_segstore_lookup(st, text, tl, chain, 64) : 0;
        free(text);
        if (m != n || (int)chain[m - 1].G != Gs[0]) {
            snprintf(err, errlen, "text lookup found a chain of %d to %d, persisted %d to %d", m,
                     m ? (int)chain[m - 1].G : 0, n, Gs[0]);
            goto out;
        }
        for (int i = 0; i < m; i++) {
            uint64_t pb = 0;
            FILE *fp = pulsar_segstore_open_payload(st, chain[i].key, &pb);
            int G = 0;
            const int lrc = fp ? pulsar_session_load_segment(*sp, fp, pb, i == m - 1, &G, chain[i].key, err, errlen) : 1;
            if (fp) fclose(fp);
            if (lrc != 0) { if (!fp) snprintf(err, errlen, "segment %d unreadable", i); goto out; }
        }
    }
    rc = 0;
out:
    if (st) pulsar_segstore_close(st);
    {   /* the store is scratch: remove its files and the directory */
        DIR *d = opendir(dir);
        for (struct dirent *de; d && (de = readdir(d));) {
            if (de->d_name[0] == '.') continue;
            char path[1024];
            snprintf(path, sizeof path, "%s/%s", dir, de->d_name);
            unlink(path);
        }
        if (d) closedir(d);
        rmdir(dir);
    }
    return rc;
}

/* Run one schedule in a fresh session: sync to `first` tokens (0 = skip), eval
 * the next `evals` prompt tokens one by one (the decode lane carrying the
 * frontier forward, as a generation does), sync to GATE_N, read the frontier
 * row, eval token GATE_N, read that row.  `origin` is where that last sync
 * must have resumed from (pulsar_session_resume_origin): the schedule is only
 * the schedule it claims to be if the resume started there.  -1 is a sync
 * that did not resume at all (schedule A: a fresh session rebuilds). */
static int run_schedule(pulsar_engine *e, const pulsar_tokens *toks, int first, int evals, int origin,
                        bool via_snapshot, int cut, int restore, int segments, int width, float *frontier,
                        float *decoded, char *err, size_t errlen) {
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
    if (cut > 0) {
        /* an edited tail (a client that drops what it saw): the bank is rewound BELOW its
         * prefill frontier; the resume must start at the grid point below the
         * cut (the dogfood case of 2026-09-06 12:08 was cold from 0 here) */
        pulsar_session_rewind(s, cut);
    }
    if (restore > 0) {
        /* L264: the grid checkpoint the first sync's prefill captured.  Say which
         * one, and that it was there: a restore that silently found nothing would
         * leave the sync below resuming some other way. */
        const int best = pulsar_session_checkpoint_best(s, restore);
        printf("  [restore] checkpoints <= %d: deepest %d\n", restore, best);
        if (best != restore) {
            snprintf(err, errlen, "no grid checkpoint at %d (deepest at or below it: %d)", restore, best);
            goto done;
        }
        if (pulsar_session_restore_checkpoint(s, restore, err, errlen) != 0) goto done;
    }
    if (segments > 0 && segment_round_trip(e, &s, toks, segments, err, errlen) != 0) goto done;
    if (via_snapshot) {
        /* the disk-cache path: serialise, throw the session away, restore into
         * a fresh one */
        pulsar_session_snapshot snap; memset(&snap, 0, sizeof snap);
        if (pulsar_session_save_snapshot(s, &snap, err, errlen) != 0) goto done;
        pulsar_session_free(s);
        s = NULL;
        if (pulsar_session_create(&s, e, GATE_CTX) != 0) { pulsar_session_snapshot_free(&snap); snprintf(err, errlen, "session create failed"); return 1; }
        const int lrc = pulsar_session_load_snapshot(s, &snap, err, errlen);
        pulsar_session_snapshot_free(&snap);
        if (lrc != 0) goto done;
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
        struct { const char *label; int first; int evals; int origin; bool via_snapshot; int cut; int restore; int segments; } sched[GATE_SCHEDULES] = {
            /* origins are on the 128 resume grid (L195): a prefill leaves its
             * snapshot at the last grid point it reached, a decode saves at
             * every crossing; a prompt under 128 tokens has none (cold) */
            {"A: cold [0,4096) [4096,8192) [8192,8600)", 0, 0, -1, false, 0, 0, 0},   /* a fresh session: the sync is a rebuild, not a resume */
            {"B: sync 6 (under the grid), then 8600: cold", 6, 0, 0, false, 0, 0, 0},
            {"C: sync 2048 (a grid point), then 8600", 2048, 0, 2048, false, 0, 0, 0},
            {"D: resume at 4000 (last grid point 3968), then 8600", 4000, 0, 3968, false, 0, 0, 0},
            {"E: resume at 4500 (last grid point 4480), then 8600", 4500, 0, 4480, false, 0, 0, 0},
            /* decode saves nothing (its rows are the decode kernels'); the
             * resume redoes the generated tokens from the last PREFILL grid
             * point -- the only way it equals the cold prefill */
            {"F: sync 8100 (grid point 8064), decode 200 across 8192, resume from 8064", 8100, GATE_EVALS, 8064, false, 0, 0, 0},
            {"G: sync 4000 (grid point 3968), decode 200 across 4096, resume from 3968", 4000, GATE_EVALS, 3968, false, 0, 0, 0},
            /* the payload carries the prefill frontier and a raw window deep
             * enough for the warm-up: a disk-restored bank resumes too */
            {"H: sync 4500, save+load into a fresh session, resume from 4480", 4500, 0, 4480, true, 0, 0, 0},
            {"I: sync 8192 (a chunk end on the grid), save+load, resume from 8192", 8192, 0, 8192, true, 0, 0, 0},
            /* the cut: prefilled to 4500, rewound to 4300 (an edited tail); the
             * resume starts at the deepest grid CHECKPOINT at or below the cut's
             * grid point 4224 -- the prefill's chunks ended at 4096 and 4480, so
             * 4096 (L264: before it, the projection ring reached 4224) */
            {"J: sync 4500, cut to 4300, resume from 4096", 4500, 0, 4096, false, 4300, 0, 0},
            /* L264: a turn whose generation ran past the raw ring's reach, echoed
             * back without it (the client stripped its reasoning): every raw slot
             * of [3840, 3968) was rewritten by decode rows 8192.. that the cut
             * removes.  The prefill's grid point 3968 is the exact resume. */
            {"K: sync 4000, decode 4600 past the ring, cut to 4100, resume from 3968", 4000, GATE_EVALS_PAST_RING, 3968, false, 4100, 0, 0},
            /* L264: resumes from the grid checkpoints the prefills captured */
            {"L: sync 8600, restore the checkpoint at 4096, resume from 4096", GATE_N, 0, 4096, false, 0, 4096, 0},
            {"M: sync 4000, decode 4600 past the ring, restore 3968, resume from 3968", 4000, GATE_EVALS_PAST_RING, 3968, false, 0, 3968, 0},
            {"N: sync 8600, restore the split checkpoint at 8576, resume from 8576", GATE_N, 0, 8576, false, 0, 8576, 0},
            /* L264 S4: the same resumes through disk segments */
            {"O: sync 8600, segment chain to 8576 through disk, resume from 8576", GATE_N, 0, 8576, false, 0, 0, 8576},
            {"P: sync 4000, decode 4600, segment chain to 3968 through disk, resume from 3968", 4000, GATE_EVALS_PAST_RING, 3968, false, 0, 0, 3968},
        };
        char err[256];
        printf("chunk-neutrality gate: %d tokens, prefill chunk %u, %d schedules; frontier row + one decode step each\n",
               GATE_N, opt.prefill_chunk, GATE_SCHEDULES);
        for (int k = 0; k < GATE_SCHEDULES; k++) {
            if (run_schedule(e, &toks, sched[k].first, sched[k].evals, sched[k].origin, sched[k].via_snapshot, sched[k].cut,
                             sched[k].restore, sched[k].segments, width,
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
        /* L264: a request that ends EXACTLY where a restored chain ends needs no
         * new rows -- but the load moved the KV, not the logits.  The sync must
         * re-evaluate the last row (logits_stale) and come out byte-identical to
         * the cold prefill of the same tokens, not sample whatever the session
         * held before. */
        {
            const int X = 8576;   /* a grid point the cold prefill captures (its final split) */
            pulsar_tokens px = toks;
            px.len = X;
            float *cold = (float *)malloc((size_t)width * sizeof(float));
            float *back = (float *)malloc((size_t)width * sizeof(float));
            pulsar_session *sx = NULL;
            err[0] = '\0';
            bool ok = cold && back && pulsar_session_create(&sx, e, GATE_CTX) == 0 &&
                      pulsar_session_sync(sx, &px, err, sizeof err) == 0 &&
                      pulsar_session_copy_logits(sx, cold, width) == width &&
                      segment_round_trip(e, &sx, &px, X, err, sizeof err) == 0 &&
                      pulsar_session_pos(sx) == X &&
                      pulsar_session_sync(sx, &px, err, sizeof err) == 0 &&
                      pulsar_session_copy_logits(sx, back, width) == width;
            if (!ok) {
                fprintf(stderr, "CHUNK-NEUTRALITY GATE: exact-length restore leg failed: %s\n", err);
                fails++;
            } else {
                printf("  [exact] chain to %d restored, sync(%d) resumed from %d\n", X, X,
                       pulsar_session_resume_origin(sx));
                fails += compare_rows("frontier", "X: segment chain to 8576, sync(8576) exactly", cold, back, width);
            }
            pulsar_session_free(sx);
            free(cold);
            free(back);
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

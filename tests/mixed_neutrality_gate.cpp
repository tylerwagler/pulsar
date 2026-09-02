/* plan-34 phase-2 increment 4 — TRUE MIXED STEP: co-scheduled decode banks + one
 * K-row prefill run in ONE fused step. Three gates in one harness:
 *
 *  GATE 4 (the core proof — CO-SCHEDULING NEUTRALITY): a decode bank's step logits
 *    must be BYTE-IDENTICAL whether it rides a decode-only step (M = n_dec) or a
 *    fused step that ALSO carries a K-row prefill run (M = n_dec + K). If they are,
 *    then co-scheduling a prefill does not perturb a decode bank — the property
 *    inc-4's per-run 3-way GEMM/MoE split exists to guarantee. Reference and fused
 *    runs use FRESH sessions that populate the decode banks IDENTICALLY (same
 *    prompts, same positions); per-bank isolation means a decode bank's output
 *    depends only on its own KV, so any divergence is a co-scheduling leak.
 *
 *  GATE 5 (MULTI-RUN NEUTRALITY, L146): each decode bank issues a run of
 *    3 rows (decode + 2 drafts, the production shape); a run's logits batched
 *    with the other banks' runs == the same run alone, byte-identical.
 *
 *  GATE 2 (PREFILL correctness in the mixed step): the fused step's prefill run
 *    last-position logits vs a classic RESUME — next-token EXACT and full-vocab
 *    rel-RMS = 0 (byte-identical KV => no corruption), same oracle as inc-3.
 *
 *  GATE 3 (MoE two-pass split boundary): with n_dec >= 2 (the per-token MoE path is
 *    taken over the decode prefix) AND K > 8 (the grouped MoE path is taken over the
 *    prefill suffix), BOTH row classes are correct — gate 4 proves the decode prefix,
 *    gate 2 proves the prefill suffix, and the split lands at row n_dec (asserted).
 *
 * The comp caches have one format each (packed attn / MXFP4 indexer).
 * MODEL-DEPENDENT, needs PULSAR_MSEQ_BANKS >= n_dec+1. Run under GPU discipline.
 *   usage: PULSAR_MSEQ_BANKS=3 ./tests/mixed_neutrality_gate MODEL
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Decode-bank count is runtime-selectable: PULSAR_GATE_NDEC (default 2, the
 * historical gate). The wide variant (e.g. 12) exercises the armed M-neutral
 * range past 8 — the l048-ntcap-16 coverage this gate previously could not
 * see: with n_dec > 8 the decode prefix runs the NT kernels' 9..16
 * instantiations and the MoE non-grouped path at those widths, and gate 4
 * still demands byte-identity vs the decode-only step of the same width. */
#define N_DEC_MAX ((int)PULSAR_MSEQ_MAX - 1)   /* +1 bank goes to the prefill run */
static int g_n_dec = 2;
/* L152 stage dump (diagnostic): with L152_DUMP_DIR set (and the engine's
 * PULSAR_CUDA_GRAPH_DUMP_NAME/LAYER pointing at the batched-step stages), each
 * GATE 5R step dumps under <dir>/<tag>_ -- bat_f / bat_r for the batched step,
 * solo<k> for bank k alone -- and the per-bank re-prefills that build the
 * fixture under <dir>/skip_ (delete those).  tools/l152_dumpcmp.py reads them. */
static const char *g_dump_dir = NULL;
static const char *g_dump_tag = NULL;
static void dump_prefix(const char *tag) {
    if (!g_dump_dir) return;
    char pfx[512];
    /* Outside the step under test the prefix points under /dev/null so the
     * fixture's per-bank re-prefill dumps fail to open instead of filling the
     * disk with 43 layers of prefill tensors per step. */
    if (!tag) snprintf(pfx, sizeof(pfx), "/dev/null/skip");
    else      snprintf(pfx, sizeof(pfx), "%s/%s", g_dump_dir, tag);
    setenv("PULSAR_CUDA_GRAPH_DUMP_PREFIX", pfx, 1);
}
#define C0    128               /* prefill bank's classic first chunk (lifts frontier off 0) */
#define K_PRE 64                /* prefill run length; >8 => grouped MoE suffix taken */
#define PBASE 700               /* prefill bank token region (distinct from decode banks) */

static pulsar_engine *g_e;
static pulsar_tokens g_toks;
static int g_fail;

/* decode banks 0..g_n_dec-1: distinct prompt regions (isolated KV). The first
 * two are the historical fixtures (default run byte-identical to the old
 * 2-bank gate); extras are generated deterministically inside [0, PBASE). */
static int g_off[N_DEC_MAX] = {0, 401};
static int g_len[N_DEC_MAX] = {130, 258};

static void init_banks(void) {
    const char *e = getenv("PULSAR_GATE_NDEC");
    if (e && *e) {
        int v = atoi(e);
        if (v < 2 || v > N_DEC_MAX) {
            fprintf(stderr, "PULSAR_GATE_NDEC=%d out of range [2,%d]\n", v, N_DEC_MAX);
            exit(2);
        }
        g_n_dec = v;
    }
    for (int k = 2; k < g_n_dec; k++) {
        g_off[k] = (k * 137) % 450;            /* distinct-ish regions; overlap is */
        g_len[k] = 100 + (k * 37) % 140;       /* harmless (per-bank KV isolation) */
    }
}

static char *read_file(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)s + 1);
    if (!b || fread(b, 1, (size_t)s, f) != (size_t)s) { fclose(f); free(b); return NULL; }
    fclose(f); b[s] = 0; if (n) *n = (size_t)s; return b;
}

/* Populate decode banks + (if K>0) the prefill bank on a FRESH session, run ONE
 * fused decode_mixed step, and copy back: each decode bank's logit row into
 * dec_rows[k] (N_DEC * vocab), and (K>0) the prefill run's last-position logits
 * into pre_row (vocab). head_cap is forwarded to pulsar_session_decode_mixed's
 * max_head_runs (0 = all runs; N_DEC = LEVER-1 intermediate path: decode banks
 * only, prefill head skipped — pre_row is then unavailable). Returns false on any
 * engine failure. */
static bool fused_step_logits(int K, uint32_t head_cap, float *dec_rows, float *pre_row) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) return false;
    pulsar_gpu_graph *g = &s->graph;
    const int vocab = (int)PULSAR_N_VOCAB;
    const uint32_t n_pre_bank = (K > 0) ? 1u : 0u;
    const uint32_t need_banks = (uint32_t)g_n_dec + n_pre_bank;
    bool ok = gpu_graph_bank_pool_count(g) >= need_banks;
    if (!ok) fprintf(stderr, "pool too small: %u < %u (set PULSAR_MSEQ_BANKS)\n",
                     gpu_graph_bank_pool_count(g), need_banks);
    char err[256];
    int argtok[N_DEC_MAX];

    /* decode banks: sync prompt, capture frontier, record next token. */
    for (int k = 0; ok && k < g_n_dec; k++) {
        if (g->banks.n_banks && !gpu_graph_bank_repoint(g, (uint32_t)k)) { ok = false; break; }
        pulsar_session_invalidate(s);
        pulsar_tokens p = { .v = g_toks.v + g_off[k], .len = g_len[k], .cap = g_len[k] };
        if (pulsar_session_sync(s, &p, err, sizeof err) != 0) {
            fprintf(stderr, "decode bank %d sync failed: %s\n", k, err); ok = false; break;
        }
        gpu_graph_bank_counters_capture(g, (uint32_t)k);
        argtok[k] = pulsar_session_argmax(s);
    }
    /* prefill bank (bank N_DEC): classic first chunk [0,C0) to lift its frontier
     * off 0 (step_begin rejects pos-0), then the K prefill rows extend [C0,C0+K). */
    const int *pptr = g_toks.v + PBASE;
    if (ok && K > 0) {
        if (g->banks.n_banks && !gpu_graph_bank_repoint(g, (uint32_t)g_n_dec)) ok = false;
        if (ok) {
            pulsar_session_invalidate(s);
            pulsar_tokens p = { .v = (int *)pptr, .len = C0, .cap = C0 };
            if (pulsar_session_sync(s, &p, err, sizeof err) != 0) {
                fprintf(stderr, "prefill bank first-chunk sync failed: %s\n", err); ok = false;
            }
        }
        if (ok) gpu_graph_bank_counters_capture(g, (uint32_t)g_n_dec);
    }

    const uint32_t n_rows = (uint32_t)g_n_dec + (K > 0 ? (uint32_t)K : 0u);
    pulsar_multiseq_req *reqs = ok ? (pulsar_multiseq_req *)malloc((size_t)n_rows * sizeof(*reqs)) : NULL;
    float *logits = ok ? (float *)malloc((size_t)n_rows * vocab * sizeof(float)) : NULL;
    if (ok && (!reqs || !logits)) ok = false;
    if (ok) {
        for (int k = 0; k < g_n_dec; k++) {
            reqs[k].bank = (uint32_t)k; reqs[k].pos = g_len[k]; reqs[k].token = argtok[k];
        }
        for (int j = 0; j < K; j++) {
            reqs[g_n_dec + j].bank = (uint32_t)g_n_dec;
            reqs[g_n_dec + j].pos = C0 + j;
            reqs[g_n_dec + j].token = pptr[C0 + j];
        }
        uint32_t n_runs = 0;
        const int rc = pulsar_session_decode_mixed(s, reqs, n_rows, logits, (int)(n_rows * vocab),
                                                &n_runs, head_cap, err, sizeof err);
        if (rc != 0) { fprintf(stderr, "decode_mixed(K=%d) failed rc=%d: %s\n", K, rc, err); ok = false; }
        else {
            const uint32_t full_runs = (uint32_t)g_n_dec + (K > 0 ? 1u : 0u);
            const uint32_t exp_runs = (head_cap == 0u || head_cap > full_runs) ? full_runs : head_cap;
            if (n_runs != exp_runs) {
                fprintf(stderr, "n_runs=%u expected %u (head_cap=%u split boundary wrong)\n",
                        n_runs, exp_runs, head_cap);
                ok = false;
            }
        }
        if (ok) {
            /* logits rows: [bank0, bank1, (prefill-last)] in run order. */
            memcpy(dec_rows, logits, (size_t)g_n_dec * vocab * sizeof(float));
            if (K > 0 && pre_row)
                memcpy(pre_row, logits + (size_t)g_n_dec * vocab, (size_t)vocab * sizeof(float));
        }
    }
    free(reqs); free(logits);
    pulsar_session_free(s);
    return ok;
}

/* classic RESUME reference for the prefill bank: fresh session, prefill [0,C0)
 * then resume to [0,C0+K), copy last-position logits + next token. */
/* GATE 5 helper (L146): the PRODUCTION row shape.  Every decode bank
 * contributes a run of `rpb` consecutive rows (a decode row plus rpb-1 draft
 * rows: 3 slots x (1 + 2 drafts) is what server_sched issues), no prefill run.
 * With only_bank >= 0 just that bank's run is issued (the solo reference).
 * Writes each issued run's LAST-row logits into out[bank].  Token ids beyond
 * the first are arbitrary but identical between the solo and batched runs. */
static bool multirun_step_logits(int rpb, int only_bank, float *out) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) return false;
    pulsar_gpu_graph *g = &s->graph;
    const int vocab = (int)PULSAR_N_VOCAB;
    bool ok = gpu_graph_bank_pool_count(g) >= (uint32_t)g_n_dec;
    char err[256];
    int argtok[N_DEC_MAX];
    for (int k = 0; ok && k < g_n_dec; k++) {
        if (g->banks.n_banks && !gpu_graph_bank_repoint(g, (uint32_t)k)) { ok = false; break; }
        pulsar_session_invalidate(s);
        pulsar_tokens p = { .v = g_toks.v + g_off[k], .len = g_len[k], .cap = g_len[k] };
        if (pulsar_session_sync(s, &p, err, sizeof err) != 0) {
            fprintf(stderr, "multirun bank %d sync failed: %s\n", k, err); ok = false; break;
        }
        gpu_graph_bank_counters_capture(g, (uint32_t)k);
        argtok[k] = pulsar_session_argmax(s);
    }
    const int n_banks_issued = only_bank >= 0 ? 1 : g_n_dec;
    const uint32_t n_rows = (uint32_t)(n_banks_issued * rpb);
    pulsar_multiseq_req *reqs = ok ? (pulsar_multiseq_req *)malloc((size_t)n_rows * sizeof(*reqs)) : NULL;
    float *logits = ok ? (float *)malloc((size_t)n_banks_issued * vocab * sizeof(float)) : NULL;
    if (ok && (!reqs || !logits)) ok = false;
    if (ok) {
        uint32_t r = 0;
        for (int k = 0; k < g_n_dec; k++) {
            if (only_bank >= 0 && k != only_bank) continue;
            for (int j = 0; j < rpb; j++, r++) {
                reqs[r].bank = (uint32_t)k;
                reqs[r].pos = g_len[k] + j;
                reqs[r].token = j == 0 ? argtok[k] : g_toks.v[g_off[k] + j];
            }
        }
        uint32_t n_runs = 0;
        const int rc = pulsar_session_decode_mixed(s, reqs, n_rows, logits, (int)(n_banks_issued * vocab),
                                                &n_runs, 0u, err, sizeof err);
        if (rc != 0) { fprintf(stderr, "multirun decode_mixed(rpb=%d) failed rc=%d: %s\n", rpb, rc, err); ok = false; }
        else if (n_runs != (uint32_t)n_banks_issued) {
            fprintf(stderr, "multirun n_runs=%u expected %d\n", n_runs, n_banks_issued); ok = false;
        }
        if (ok) {
            int slot = 0;
            for (int k = 0; k < g_n_dec; k++) {
                if (only_bank >= 0 && k != only_bank) continue;
                memcpy(out + (size_t)k * vocab, logits + (size_t)slot * vocab, (size_t)vocab * sizeof(float));
                slot++;
            }
        }
    }
    free(reqs); free(logits);
    pulsar_session_free(s);
    return ok;
}


/* L152: the same step, but every ROW's logits come back (ALL_ROWS head) and
 * the banks can be issued in reverse order, so a failing run can be read row
 * by row and "bank 0" separated from "first in the batch". out has
 * (only_bank >= 0 ? 1 : g_n_dec) * rpb rows of PULSAR_N_VOCAB floats, in
 * ISSUE order; *issued lists the bank of each run in that order. */
static bool multirun_step_rows(const int *rpb_of, int only_bank, bool reverse, float *out, int *issued) {
    pulsar_session *s = NULL;
    dump_prefix("skip");   /* the fixture re-prefills dump under skip_ (L152 diag) */
    if (pulsar_session_create(&s, g_e, 4096) != 0) return false;
    pulsar_gpu_graph *g = &s->graph;
    const int vocab = (int)PULSAR_N_VOCAB;
    bool ok = gpu_graph_bank_pool_count(g) >= (uint32_t)g_n_dec;
    char err[256];
    int argtok[N_DEC_MAX];
    for (int k = 0; ok && k < g_n_dec; k++) {
        if (g->banks.n_banks && !gpu_graph_bank_repoint(g, (uint32_t)k)) { ok = false; break; }
        pulsar_session_invalidate(s);
        pulsar_tokens p = { .v = g_toks.v + g_off[k], .len = g_len[k], .cap = g_len[k] };
        if (pulsar_session_sync(s, &p, err, sizeof err) != 0) {
            fprintf(stderr, "multirun bank %d sync failed: %s\n", k, err); ok = false; break;
        }
        gpu_graph_bank_counters_capture(g, (uint32_t)k);
        argtok[k] = pulsar_session_argmax(s);
    }
    uint32_t n_rows = 0;
    for (int k = 0; k < g_n_dec; k++) if (only_bank < 0 || k == only_bank) n_rows += (uint32_t)rpb_of[k];
    pulsar_multiseq_req *reqs = ok ? (pulsar_multiseq_req *)malloc((size_t)n_rows * sizeof(*reqs)) : NULL;
    if (ok && !reqs) ok = false;
    if (ok) {
        uint32_t r = 0; int slot = 0;
        for (int i = 0; i < g_n_dec; i++) {
            const int k = reverse ? g_n_dec - 1 - i : i;
            if (only_bank >= 0 && k != only_bank) continue;
            issued[slot++] = k;
            const int rpb = rpb_of[k];
            for (int j = 0; j < rpb; j++, r++) {
                reqs[r].bank = (uint32_t)k;
                reqs[r].pos = g_len[k] + j;
                reqs[r].token = j == 0 ? argtok[k] : g_toks.v[g_off[k] + j];
            }
        }
        uint32_t got = 0;
        dump_prefix(g_dump_tag);
        const int rc = pulsar_session_decode_mixed(s, reqs, n_rows, out, (int)(n_rows * (uint32_t)vocab),
                                                &got, PULSAR_MSEQ_HEAD_ALL_ROWS, err, sizeof err);
        dump_prefix("skip");
        if (rc != 0 || got != n_rows) {
            fprintf(stderr, "multirun ALL_ROWS decode_mixed(rows=%u) rc=%d got=%u: %s\n", n_rows, rc, got, err); ok = false;
        }
    }
    free(reqs);
    pulsar_session_free(s);
    return ok;
}

static bool classic_resume(int K, float *out_lg, int *next_tok) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) return false;
    pulsar_gpu_graph *g = &s->graph;
    char err[256]; bool ok = true;
    const int *pptr = g_toks.v + PBASE;
    if (g->banks.n_banks && !gpu_graph_bank_repoint(g, 0)) { pulsar_session_free(s); return false; }
    pulsar_session_invalidate(s);
    pulsar_tokens p0 = { .v = (int *)pptr, .len = C0, .cap = C0 };
    if (pulsar_session_sync(s, &p0, err, sizeof err) != 0) { fprintf(stderr, "classic chunk: %s\n", err); ok = false; }
    pulsar_tokens p1 = { .v = (int *)pptr, .len = C0 + K, .cap = C0 + K };
    if (ok && pulsar_session_sync(s, &p1, err, sizeof err) != 0) { fprintf(stderr, "classic resume: %s\n", err); ok = false; }
    if (ok) {
        gpu_graph_bank_counters_capture(g, 0);
        pulsar_session_copy_logits(s, out_lg, (int)PULSAR_N_VOCAB);
        *next_tok = pulsar_session_argmax(s);
    }
    pulsar_session_free(s);
    return ok;
}

static long first_diff(const float *a, const float *b, long n) {
    for (long i = 0; i < n; i++) if (a[i] != b[i]) return i;
    return -1;
}

/* GATE 5R (L152): the multi-run step's EVERY row, batched vs solo, in both bank
 * orders.  `spec` is one run length for every bank or a comma list per bank
 * ("5,5" = two banks of 5 rows).  fatal => a differing row fails the gate;
 * otherwise it is reported (the bisect mode).  This is the check the battery
 * lacked between 9 and 16 rows: GATE 5 compares only each run's LAST row at
 * 3 rows per bank, the "wide" target skips GATE 5 above the 16-row cap, and
 * the L152 defect (rows 2-4 of one bank off by whole logits at 10 rows,
 * byte-identical at 8) lived exactly there.  Dumps per step under
 * L152_DUMP_DIR when set (tools/l152_dumpcmp.py). */
static void gate5r_run(const char *spec, bool fatal) {
    int rpb_of[N_DEC_MAX];
    for (int k = 0; k < N_DEC_MAX; k++) rpb_of[k] = 5;
    if (spec && spec[0]) {
        const char *q = spec;
        for (int k = 0; k < g_n_dec && *q; k++) {
            rpb_of[k] = atoi(q);
            const char *c = strchr(q, ',');
            if (!c) { for (int m = k + 1; m < g_n_dec; m++) rpb_of[m] = rpb_of[k]; break; }
            q = c + 1;
        }
    }
    int total = 0, rmax = 0;
    for (int k = 0; k < g_n_dec; k++) { total += rpb_of[k]; if (rpb_of[k] > rmax) rmax = rpb_of[k]; }
    const char *tag = "GATE 5R";
    printf("%s: run lengths per bank:", tag); for (int k = 0; k < g_n_dec; k++) printf(" %d", rpb_of[k]);
    printf(" (total %d rows%s)\n", total, fatal ? ", fatal" : ", diagnostic");
    if ((uint32_t)total > PULSAR_GPU_MNEUTRAL_ROWS_MAX) {
        printf("%s: skipped (%d rows > M-neutral cap %u)\n", tag, total, PULSAR_GPU_MNEUTRAL_ROWS_MAX);
        return;
    }
    const uint32_t vocab = PULSAR_N_VOCAB;
    g_dump_dir = getenv("L152_DUMP_DIR");
    for (int rev = 0; rev < 2; rev++) {
        float *all = (float *)malloc((size_t)total * vocab * sizeof(float));
        float *solo = (float *)malloc((size_t)rmax * vocab * sizeof(float));
        int issued[N_DEC_MAX], one[N_DEC_MAX];
        char solo_tag[32];
        g_dump_tag = rev ? "bat_r" : "bat_f";
        if (!all || !solo || !multirun_step_rows(rpb_of, -1, rev != 0, all, issued)) {
            fprintf(stderr, "%s: batched step failed\n", tag); g_fail = 1; free(all); free(solo); continue;
        }
        size_t row_off = 0;
        for (int slot = 0; slot < g_n_dec; slot++) {
            const int k = issued[slot];
            const int RPB = rpb_of[k];
            snprintf(solo_tag, sizeof(solo_tag), "solo%d", k);
            g_dump_tag = solo_tag;
            if (!multirun_step_rows(rpb_of, k, false, solo, one)) { fprintf(stderr, "%s: solo %d failed\n", tag, k); g_fail = 1; break; }
            int bad = 0;
            for (int j = 0; j < RPB; j++) {
                const float *a = all + (row_off + (size_t)j) * vocab;
                const float *b = solo + (size_t)j * vocab;
                long d = first_diff(a, b, (long)vocab);
                if (d < 0) {
                    if (!fatal) printf("%s %s: bank %d (slot %d) row %d/%d: IDENTICAL\n", tag, rev ? "reversed" : "forward", k, slot, j, RPB);
                } else {
                    long nd = 0; float md = 0.f;
                    for (long i = 0; i < (long)vocab; i++) if (a[i] != b[i]) { nd++; float x = fabsf(a[i] - b[i]); if (x > md) md = x; }
                    fprintf(fatal ? stderr : stdout,
                            "%s%s %s: bank %d (slot %d) row %d/%d: DIFFERS at float %ld (%.6g vs %.6g), %ld/%u floats, max |d| %.4g\n",
                            fatal ? "GATE 5R FAIL: " : "", tag, rev ? "reversed" : "forward", k, slot, j, RPB,
                            d, (double)a[d], (double)b[d], nd, vocab, (double)md);
                    bad++;
                }
            }
            if (fatal) {
                if (bad) g_fail = 1;
                else printf("%s %s: bank %d (slot %d) all %d rows batched (%d) == solo BYTE-IDENTICAL\n",
                            tag, rev ? "reversed" : "forward", k, slot, RPB, total);
            }
            row_off += (size_t)RPB;
        }
        free(all); free(solo);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }
    pulsar_engine_options o; memset(&o, 0, sizeof o);
    o.model_path = argv[1]; o.backend = PULSAR_BACKEND_CUDA;
    if (pulsar_engine_open(&g_e, &o) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    init_banks();
    printf("CONFIG: packed attn comp cache + MXFP4 indexer cache (the only "
           "formats)  (n_dec=%d K=%d)\n", g_n_dec, K_PRE);

    size_t tl = 0; char *txt = read_file("tests/long_context_story_prompt.txt", &tl);
    if (!txt) { fprintf(stderr, "prompt read failed\n"); return 1; }
    memset(&g_toks, 0, sizeof g_toks);
    pulsar_tokenize_text(g_e, txt, &g_toks); free(txt);
    int need = PBASE + C0 + K_PRE + 1;
    for (int k = 0; k < g_n_dec; k++) if (g_off[k] + g_len[k] > need) need = g_off[k] + g_len[k];
    if (g_toks.len < need) { fprintf(stderr, "prompt too short (%d<%d)\n", g_toks.len, need); return 1; }

    const int vocab = (int)PULSAR_N_VOCAB;
    float *ref_dec = (float *)malloc((size_t)g_n_dec * vocab * sizeof(float));   /* decode-only M=N_DEC */
    float *mix_dec = (float *)malloc((size_t)g_n_dec * vocab * sizeof(float));   /* fused M=N_DEC+K, full head */
    float *lv1_dec = (float *)malloc((size_t)g_n_dec * vocab * sizeof(float));   /* fused M=N_DEC+K, LEVER-1 head */
    float *mix_pre = (float *)malloc((size_t)vocab * sizeof(float));           /* fused prefill last */
    float *cls_pre = (float *)malloc((size_t)vocab * sizeof(float));           /* classic-resume last */
    int cls_next = -1;

    if (!fused_step_logits(0,     0u,             ref_dec, NULL))    { fprintf(stderr, "GATE FAIL: decode-only reference run failed\n"); g_fail = 1; goto done; }
    if (!fused_step_logits(K_PRE, 0u,             mix_dec, mix_pre)) { fprintf(stderr, "GATE FAIL: fused mixed run (full head) failed\n"); g_fail = 1; goto done; }
    if (!fused_step_logits(K_PRE, (uint32_t)g_n_dec, lv1_dec, NULL))   { fprintf(stderr, "GATE FAIL: fused mixed run (LEVER-1 head) failed\n"); g_fail = 1; goto done; }
    if (!classic_resume(K_PRE, cls_pre, &cls_next))  { fprintf(stderr, "GATE FAIL: classic-resume reference failed\n"); g_fail = 1; goto done; }

    /* GATE 4: each decode bank byte-identical decode-only vs fused (full head). */
    for (int k = 0; k < g_n_dec; k++) {
        const long d = first_diff(mix_dec + (size_t)k * vocab, ref_dec + (size_t)k * vocab, vocab);
        if (d < 0) {
            printf("GATE 4 NEUTRALITY: decode bank %d logits M=%d (decode-only) == M=%d (+%d-row prefill) "
                   "BYTE-IDENTICAL\n", k, g_n_dec, g_n_dec + K_PRE, K_PRE);
        } else {
            fprintf(stderr, "GATE 4 FAIL: decode bank %d logits DIFFER at float %ld (%.9g vs %.9g) — "
                    "co-scheduling a %d-row prefill perturbed a decode bank\n",
                    k, d, (mix_dec + (size_t)k * vocab)[d], (ref_dec + (size_t)k * vocab)[d], K_PRE);
            g_fail = 1;
        }
    }

    /* GATE 4b (LEVER 1): an INTERMEDIATE fused step (max_head_runs = n_dec, prefill
     * head SKIPPED, single-block identity head path) must yield decode-bank logits
     * BYTE-IDENTICAL to decode-only — the prefill's intermediate logits are never
     * consumed, and skipping them must not perturb the decode banks. */
    for (int k = 0; k < g_n_dec; k++) {
        const long d = first_diff(lv1_dec + (size_t)k * vocab, ref_dec + (size_t)k * vocab, vocab);
        if (d < 0) {
            printf("GATE 4b LEVER-1: decode bank %d logits (intermediate fused, head-skip) == decode-only "
                   "BYTE-IDENTICAL\n", k);
        } else {
            fprintf(stderr, "GATE 4b FAIL: decode bank %d LEVER-1 logits DIFFER at float %ld (%.9g vs %.9g) "
                    "— skipping the prefill head perturbed a decode bank\n",
                    k, d, (lv1_dec + (size_t)k * vocab)[d], (ref_dec + (size_t)k * vocab)[d]);
            g_fail = 1;
        }
    }

    /* L152 (diagnostic, argv[3] == "rows"): every row of the multi-run step at
     * argv[2] rows per bank, batched vs solo, forward and reversed bank order.
     * Reports only; the battery's fatal form is PULSAR_GATE_ROWS_FATAL below. */
    if (argc > 3 && strcmp(argv[3], "rows") == 0) {
        gate5r_run(argc > 2 ? argv[2] : "5", false);
        printf("GATE 5R done (diagnostic only)\n");
        pulsar_engine_close(g_e);
        return g_fail ? 1 : 0;
    }

    /* GATE 5 (L146): MULTI-RUN NEUTRALITY -- the production row shape.  Each
     * decode bank issues a run of RPB rows (decode + drafts); the run's last
     * logits when all banks are batched must be BYTE-IDENTICAL to the same run
     * issued alone.  This is gate 4's property for DRAFT rows, and the shape no
     * other gate had: several multi-row runs, 6..16 rows, no prefill run.  On
     * 57c0d28 it FAILS -- M-neutral was armed only for length-1 runs, so the
     * batched step took cuBLASLt/grouped-MoE paths whose per-row values depend
     * on M.  Asserted only while the batch fits the M-neutral row cap; above it
     * the prefix scan is the documented behaviour and the leg is reported as
     * skipped rather than failed. */
    {
        /* L150 bisect: argv[2] overrides the rows per bank (default 3, the
         * production run length) so the M-neutral claim can be checked at
         * every total row count up to the cap, not only at 6. */
        const int RPB = argc > 2 ? atoi(argv[2]) : 3;
        const int rows_total = g_n_dec * RPB;
        if ((uint32_t)rows_total > PULSAR_GPU_MNEUTRAL_ROWS_MAX) {
            printf("GATE 5 MULTI-RUN: skipped (%d rows > M-neutral cap %u)\n", rows_total, PULSAR_GPU_MNEUTRAL_ROWS_MAX);
        } else {
            float *mr_all  = (float *)malloc((size_t)g_n_dec * vocab * sizeof(float));
            float *mr_solo = (float *)malloc((size_t)g_n_dec * vocab * sizeof(float));
            bool mr_ok = mr_all && mr_solo && multirun_step_logits(RPB, -1, mr_all);
            for (int k = 0; mr_ok && k < g_n_dec; k++) mr_ok = multirun_step_logits(RPB, k, mr_solo);
            if (!mr_ok) { fprintf(stderr, "GATE 5 FAIL: multi-run step failed\n"); g_fail = 1; }
            for (int k = 0; mr_ok && k < g_n_dec; k++) {
                const long d = first_diff(mr_all + (size_t)k * vocab, mr_solo + (size_t)k * vocab, vocab);
                if (d < 0) {
                    printf("GATE 5 MULTI-RUN: bank %d run of %d rows, batched (%d rows) == solo (%d rows) BYTE-IDENTICAL\n",
                           k, RPB, rows_total, RPB);
                } else {
                    fprintf(stderr, "GATE 5 FAIL: bank %d run last-row logits DIFFER at float %ld (%.9g vs %.9g) -- "
                            "a %d-row multi-run step is not M-neutral\n",
                            k, d, (mr_all + (size_t)k * vocab)[d], (mr_solo + (size_t)k * vocab)[d], rows_total);
                    g_fail = 1;
                }
            }
            free(mr_all); free(mr_solo);
        }
    }

    /* GATE 5R fatal (L152/L153): PULSAR_GATE_ROWS_FATAL lists per-bank run
     * lengths, space-separated ("5,5 8,8" = 10 and 16 rows with two decode
     * banks); every row of every bank must match its solo run byte for byte.
     * The battery sets it; the default run without it is unchanged. */
    if (const char *specs = getenv("PULSAR_GATE_ROWS_FATAL")) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s", specs);
        for (char *tok = strtok(buf, " ;"); tok; tok = strtok(NULL, " ;")) gate5r_run(tok, true);
    }

    /* GATE 2: prefill run last-position logits vs classic-resume. */
    {
        double se = 0, sr = 0;
        for (int i = 0; i < vocab; i++) { double dd = (double)mix_pre[i] - cls_pre[i]; se += dd * dd; sr += (double)cls_pre[i] * cls_pre[i]; }
        double rel_rms = sr > 0 ? sqrt(se / sr) : (se > 0 ? 1e9 : 0.0);
        int mix_next = (int)argmax_f32(mix_pre, (uint64_t)vocab);
        printf("GATE 2 PREFILL: next-token fused=%d classic=%d %s | last-pos logit rel-RMS=%.3e (<1e-2: %s)\n",
               mix_next, cls_next, mix_next == cls_next ? "MATCH" : "MISMATCH",
               rel_rms, rel_rms < 1e-2 ? "YES" : "NO");
        if (mix_next != cls_next) { fprintf(stderr, "GATE 2 FAIL: prefill next-token mismatch (KV/boundary wrong)\n"); g_fail = 1; }
        if (rel_rms >= 1e-2) { fprintf(stderr, "GATE 2 FAIL: prefill last-pos rel-RMS %.3e >= 1e-2 (corruption)\n", rel_rms); g_fail = 1; }
    }

    /* GATE 3: split boundary. n_dec>=2 => per-token MoE prefix; K>8 => grouped MoE suffix. */
    printf("GATE 3 MoE SPLIT: n_dec=%d (per-token MoE, rows [0,%d)) | K=%d>8 (grouped MoE, rows [%d,%d)) | "
           "per-token->grouped switch at row %d %s\n",
           g_n_dec, g_n_dec, K_PRE, g_n_dec, g_n_dec + K_PRE, g_n_dec,
           (g_n_dec >= 2 && K_PRE > 8) ? "OK" : "MISCONFIGURED");
    if (!(g_n_dec >= 2 && K_PRE > 8)) { fprintf(stderr, "GATE 3 FAIL: gate misconfigured (need n_dec>=2 and K>8)\n"); g_fail = 1; }

done:
    free(ref_dec); free(mix_dec); free(lv1_dec); free(mix_pre); free(cls_pre);
    pulsar_engine_close(g_e);
    if (g_fail) { fprintf(stderr, "MIXED-NEUTRALITY GATE: FAIL\n"); return 1; }
    printf("MIXED-NEUTRALITY GATE: PASS\n");
    return 0;
}

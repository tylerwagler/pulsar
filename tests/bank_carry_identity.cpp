/* bank_carry_identity -- L260: the server's batched speculative lane, byte-reproducible, for before/after
 * comparison of a bookkeeping change to the bank switch (save/restore).
 *
 *   PULSAR_MSEQ_BANKS=3 ./tests/bank_carry_identity MODEL [TICKS] [TEMP] [bank|batch]
 *
 * Drives three banks exactly as server_sched.cpp's worker_spec_batched_quantum does -- per bank restore ->
 * next_base -> round_begin -> rows -> save; ONE decode_mixed ALL_ROWS forward; per bank restore -> round_end ->
 * save; one batched redraft; per bank restore -> redraft_commit -> save -- with WIDE sampling (temperature TEMP,
 * default 1.0; no top-k, top-p 1, min-p 0), so a drafted q can exceed the compact cap and the verify walk must
 * rebuild it from the draft's full row: the carry path L260 changed.  Prints every bank's accepted tokens per
 * tick and an FNV hash per bank; a bookkeeping change must reproduce the output byte for byte on the same model
 * (run the same source against both trees and diff).  It also counts the pending drafts that took the full-row
 * path at save time -- zero means this run did not exercise that path and proves nothing about it.
 *
 * MODE `batch` drives the same lane through the phase batch calls (pulsar_session_spec_assemble_batch /
 * round_end_batch / redraft_commit_batch, L260 phase 2); its token lines must equal MODE `bank`'s on the same
 * build.  Both modes print the host wall time per tick of each phase.
 *
 * Not a gate (it has no reference of its own); it is the instrument for a before/after identity check. */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "gate_entry.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

enum { NB = 3, ROWS = 32 };
static const int k_prompt_off[NB] = {0, 401, 700};
static const int k_prompt_len[NB] = {130, 258, 160};

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    const long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, fp) != (size_t)n) { fclose(fp); free(buf); return NULL; }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

/* host wall time of every bank save / restore the lane makes (the L260 cost), printed at the end */
static double g_save_us, g_restore_us;
static uint64_t g_saves, g_restores;
static double now_us() {
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count() / 1e3;
}
static void timed_save(pulsar_session *s, uint32_t b) {
    const double t0 = now_us();
    pulsar_session_bank_state_save(s, b);
    g_save_us += now_us() - t0;
    g_saves++;
}
static bool timed_restore(pulsar_session *s, uint32_t b) {
    const double t0 = now_us();
    const bool ok = pulsar_session_bank_state_restore(s, b);
    g_restore_us += now_us() - t0;
    g_restores++;
    return ok;
}

static uint64_t fnv(uint64_t h, int v) {
    for (int i = 0; i < 4; i++) { h ^= (uint64_t)((uint32_t)v >> (8 * i) & 0xffu); h *= 1099511628211ull; }
    return h;
}

/* pending drafts a bank's saved carry holds as full rows (the non-compact q path).  The compact test is spelled out
 * here, not taken from pulsar_spec_q_compact, because this instrument must also build against the tree from
 * BEFORE that helper existed (the "before" half of the identity check). */
static uint32_t full_row_pendings(const pulsar_spec_carry_state &sp) {
    if (!sp.pend_sampled) return 0;
    const uint32_t n = sp.n_pend < 16u ? sp.n_pend : 16u;
    uint32_t k = 0;
    for (uint32_t j = 0; j < n; j++)
        k += !(sp.pend_qn[j] > 0 && sp.pend_qn[j] <= PULSAR_DSPARK_QDIST_CAP);
    return k;
}

int GATE_ENTRY(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [TICKS] [TEMP] [bank|batch]\n", argv[0]); return 2; }
    const int ticks = argc > 2 ? atoi(argv[2]) : 24;
    const float temp = argc > 3 ? (float)atof(argv[3]) : 1.0f;
    const bool batch = argc > 4 && strcmp(argv[4], "batch") == 0;
    if (argc > 4 && !batch && strcmp(argv[4], "bank") != 0) { fprintf(stderr, "MODE is bank or batch\n"); return 2; }
    const int top_k = 0;
    const float top_p = 1.0f, min_p = 0.0f;
    pulsar_engine *e = NULL;
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    if (gate_engine_open(&e, &opt) != 0 || !pulsar_engine_has_spec_rounds(e)) {
        fprintf(stderr, "engine open failed or no drafter\n");
        return 1;
    }
    char *text = read_file("tests/long_context_story_prompt.txt");
    if (!text) { fprintf(stderr, "prompt file read failed\n"); return 1; }
    pulsar_tokens toks;
    memset(&toks, 0, sizeof(toks));
    pulsar_tokenize_text(e, text, &toks);
    free(text);
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, e, 4096) != 0 || pulsar_session_bank_count(s) < NB) {
        fprintf(stderr, "session create failed or fewer than %d banks (PULSAR_MSEQ_BANKS)\n", NB);
        return 1;
    }
    char err[256];
    for (int b = 0; b < NB; b++) {
        pulsar_tokens p;
        memset(&p, 0, sizeof(p));
        p.v = toks.v + k_prompt_off[b];
        p.len = p.cap = k_prompt_len[b];
        if (pulsar_session_bank_repoint(s, (uint32_t)b) != 0) { fprintf(stderr, "repoint %d\n", b); return 1; }
        pulsar_session_invalidate(s);
        if (pulsar_session_sync(s, &p, err, sizeof(err)) != 0) { fprintf(stderr, "prefill %d: %s\n", b, err); return 1; }
        pulsar_session_bank_state_save(s, (uint32_t)b);
    }
    const int vocab = pulsar_engine_logits_width(e);
    const int eos = pulsar_token_eos(e);
    float *logits = (float *)malloc((size_t)ROWS * (size_t)vocab * sizeof(float));
    pulsar_spec_round *r[NB];
    for (int b = 0; b < NB; b++) r[b] = pulsar_spec_round_new();
    uint64_t rngs[NB] = {0x2545F4914F6CDD1Dull, 0x9E3779B97F4A7C15ull, 0xD1B54A32D192ED03ull};
    uint64_t hash[NB] = {1469598103934665603ull, 1469598103934665603ull, 1469598103934665603ull};
    uint64_t full_rows = 0, drafted = 0;
    double ph_us[3] = {0.0, 0.0, 0.0};   /* assemble, round_end, commit */
    pulsar_spec_step steps[NB];
    int acc_b[NB][17];
    for (int t = 0; t < ticks; t++) {
        pulsar_multiseq_req reqs[ROWS];
        int first[NB];
        uint32_t row0[NB], rows = 0;
        double t0 = now_us();
        if (batch) {
            for (int b = 0; b < NB; b++) {
                memset(&steps[b], 0, sizeof(steps[b]));
                steps[b].bank = (uint32_t)b;
                steps[b].round = r[b];
                steps[b].temperature = temp; steps[b].top_k = top_k;
                steps[b].top_p = top_p; steps[b].min_p = min_p;
                steps[b].rng = &rngs[b];
                steps[b].max_tokens = 64;
                steps[b].k_alloc = -1;
                steps[b].accepted_cap = 17;
            }
            if (pulsar_session_spec_assemble_batch(s, steps, NB, eos, ROWS, reqs, &rows) != 0) {
                fprintf(stderr, "tick %d assemble batch refused\n", t);
                return 1;
            }
            for (int b = 0; b < NB; b++) {
                if (steps[b].status != PULSAR_SPEC_STEP_OK) {
                    fprintf(stderr, "tick %d bank %d assemble status %d: %s\n", t, b, steps[b].status, steps[b].err);
                    return 1;
                }
                first[b] = steps[b].first_token;
                row0[b] = steps[b].row0;
            }
        }
        for (int b = 0; !batch && b < NB; b++) {
            if (!timed_restore(s, (uint32_t)b)) { fprintf(stderr, "restore %d\n", b); return 1; }
            first[b] = pulsar_session_spec_next_base(s, temp, top_k, top_p, min_p, &rngs[b]);
            if (first[b] < 0 || pulsar_session_spec_round_begin(s, r[b], first[b], 64, 17, temp, top_k, top_p, min_p,
                                                                err, sizeof(err)) != 0) {
                fprintf(stderr, "tick %d bank %d begin: %s\n", t, b, err);
                return 1;
            }
            row0[b] = rows;
            rows += pulsar_spec_round_fill_reqs(r[b], (uint32_t)b, first[b], reqs + rows);
            timed_save(s, (uint32_t)b);
        }
        ph_us[0] += now_us() - t0;
        pulsar_session_spec_arm_capture(s, rows);
        uint32_t got = 0;
        const int rc = pulsar_session_decode_mixed(s, reqs, rows, logits, (int)(rows * (uint32_t)vocab), &got,
                                                   PULSAR_MSEQ_HEAD_ALL_ROWS, err, sizeof(err));
        pulsar_session_spec_arm_capture(s, 0u);
        if (rc != 0 || got != rows) { fprintf(stderr, "tick %d forward: %s\n", t, err); return 1; }
        int na_b[NB];
        t0 = now_us();
        if (batch) {
            for (int b = 0; b < NB; b++) {
                steps[b].first_token = first[b];
                steps[b].row0 = row0[b];
                steps[b].accepted = acc_b[b];
                steps[b].accepted_cap = 17;
            }
            if (pulsar_session_spec_round_end_batch(s, steps, NB, eos, logits) != 0) {
                fprintf(stderr, "tick %d round_end batch refused\n", t);
                return 1;
            }
            for (int b = 0; b < NB; b++) {
                if (steps[b].status != PULSAR_SPEC_STEP_OK) {
                    fprintf(stderr, "tick %d bank %d end status %d: %s\n", t, b, steps[b].status, steps[b].err);
                    return 1;
                }
                na_b[b] = steps[b].n_accepted;
            }
        }
        for (int b = 0; !batch && b < NB; b++) {
            if (!timed_restore(s, (uint32_t)b)) { fprintf(stderr, "restore %d\n", b); return 1; }
            na_b[b] = pulsar_session_spec_round_end(s, r[b], first[b], eos, temp, top_k, top_p, min_p, &rngs[b],
                                                    logits, row0[b], acc_b[b], 17, err, sizeof(err));
            if (na_b[b] < 0) { fprintf(stderr, "tick %d bank %d end: %s\n", t, b, err); return 1; }
            timed_save(s, (uint32_t)b);
        }
        ph_us[1] += now_us() - t0;
        printf("tick %d:", t);
        for (int b = 0; b < NB; b++) {
            printf(" [b%d", b);
            for (int k = 0; k < na_b[b]; k++) { printf(" %d", acc_b[b][k]); hash[b] = fnv(hash[b], acc_b[b][k]); }
            printf("]");
        }
        printf("\n");
        uint32_t banks[NB] = {0, 1, 2};
        uint64_t *rps[NB] = {&rngs[0], &rngs[1], &rngs[2]};
        if (pulsar_session_spec_redraft_batch(s, r, banks, rps, NB, err, sizeof(err)) != 0)
            fprintf(stderr, "tick %d redraft: %s (banks take a plain step)\n", t, err);
        t0 = now_us();
        if (batch) {
            for (int b = 0; b < NB; b++) {
                memset(&steps[b], 0, sizeof(steps[b]));
                steps[b].bank = (uint32_t)b;
                steps[b].round = r[b];
            }
            if (pulsar_session_spec_redraft_commit_batch(s, steps, NB) != 0) {
                fprintf(stderr, "tick %d commit batch refused\n", t);
                return 1;
            }
            for (int b = 0; b < NB; b++)
                if (steps[b].status != PULSAR_SPEC_STEP_OK) { fprintf(stderr, "tick %d bank %d commit\n", t, b); return 1; }
        }
        for (int b = 0; !batch && b < NB; b++) {
            if (!timed_restore(s, (uint32_t)b)) { fprintf(stderr, "restore %d\n", b); return 1; }
            pulsar_session_spec_redraft_commit(s, r[b]);
            timed_save(s, (uint32_t)b);
        }
        ph_us[2] += now_us() - t0;
        for (int b = 0; b < NB; b++) {
            drafted += s->bank_carry[b].spec.n_pend;
            full_rows += full_row_pendings(s->bank_carry[b].spec);
        }
    }
    printf("hash b0 %016llx b1 %016llx b2 %016llx\n", (unsigned long long)hash[0], (unsigned long long)hash[1],
           (unsigned long long)hash[2]);
    printf("pending drafts %llu, of them full-row (non-compact q) %llu%s\n", (unsigned long long)drafted,
           (unsigned long long)full_rows, full_rows ? "" : "  -- the full-row path was NOT exercised");
    printf("host: %llu saves %.1f us mean, %llu restores %.1f us mean (the timing lines differ run to run; the "
           "token lines above are the identity)\n", (unsigned long long)g_saves,
           g_saves ? g_save_us / (double)g_saves : 0.0, (unsigned long long)g_restores,
           g_restores ? g_restore_us / (double)g_restores : 0.0);
    printf("host per tick (%s): assemble %.1f us, round_end %.1f us, commit %.1f us\n", batch ? "batch" : "bank",
           ph_us[0] / ticks, ph_us[1] / ticks, ph_us[2] / ticks);
    for (int b = 0; b < NB; b++) pulsar_spec_round_free(r[b]);
    free(logits);
    pulsar_session_free(s);
    pulsar_tokens_free(&toks);
    gate_engine_close(e);
    return 0;
}

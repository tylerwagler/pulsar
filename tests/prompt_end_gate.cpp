/* L289: the prompt end -- a sync of the same prompt the bank last prefilled resumes AT its end with the logits
 * that prefill left, evaluating no row -- is the cold prefill's state byte for byte, on real weights, through the
 * public API only (every family with PULSAR_FAMILY_CAP_PAYLOAD: the payload is the byte-level witness).
 *
 * Two sessions run the same history: A keeps its bank, C is invalidated before the repeat, so C's repeat is the
 * cold prefill A's must equal.  For each prompt length P (below every grid point, on it, off it, past a chunk):
 *
 *   1. COLD      the first sync of P prefills from 0 on both.
 *   2. REUSE     after N_PRE speculated tokens (the decode moved every lane, the drafter's state and the pools'
 *                part-filled rows), A's sync of P resumes at P -- no row evaluated -- and C's (invalidated) from 0.
 *   3. STATE     A's logits are C's byte for byte, and so is its session payload (the tokens, the logits, the
 *                frontier walk -- every lane, the drafter's rings, the n-gram context --, every pool's rows to P,
 *                the draft-depth controller) except storage no step reads before writing, which is reported by
 *                range (DeepSeek: DSpark ring rows past its counters, left by the cold side's earlier decode).
 *   4. DECODE    greedy speculation from both commits the same tokens in the same rounds.
 *   5. AGAIN     A's next sync of P (after its own reuse and those rounds) resumes at P again, logits == C's cold.
 *   6. MISS      a prompt of the same length with a different last token is not served by the record: it
 *                prefills, and its logits equal C's cold prefill of it.
 *
 * usage: ./tests/prompt_end_gate MODEL [P] */
#include "pulsar.h"
#include "gate_entry.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_PRE 16    /* greedy tokens speculated before the repeat */
#define N_SPEC 32   /* greedy tokens compared after it */

static int g_fail;

static void say(bool ok, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void say(bool ok, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "  %s ", ok ? "ok  " : "FAIL");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    if (!ok) g_fail++;
}

/* greedy decode to n tokens: speculative rounds where the engine runs them (each round's count recorded), else
 * one argmax eval a token (a round of 1) */
static int greedy_run(pulsar_engine *e, pulsar_session *s, int n, int *toks, int *rounds, int *n_rounds, float *lg,
                      char *err, size_t errlen) {
    uint64_t rng = 7;
    int got = 0;
    *n_rounds = 0;
    const int W = pulsar_engine_logits_width(e);
    while (got < n) {
        if (pulsar_engine_has_spec_rounds(e)) {
            int buf[17];
            const int k = pulsar_session_generate_speculative(s, 0.0f, 0, 1.0f, 0.0f, &rng, n - got, buf, 17, err,
                                                              errlen);
            if (k <= 0) return 1;
            rounds[(*n_rounds)++] = k;
            for (int i = 0; i < k && got < n; i++) toks[got++] = buf[i];
        } else {
            pulsar_session_copy_logits(s, lg, W);
            int best = 0;
            for (int i = 1; i < W; i++) if (lg[i] > lg[best]) best = i;
            if (pulsar_session_eval(s, best, err, errlen) != 0) return 1;
            rounds[(*n_rounds)++] = 1;
            toks[got++] = best;
        }
    }
    return 0;
}

/* the session's payload, in memory (NULL on a failed save) */
static char *payload_of(pulsar_session *s, long *len, char *err, size_t errlen) {
    FILE *fp = tmpfile();
    if (!fp) return NULL;
    char *buf = NULL;
    if (pulsar_session_save_payload(s, fp, err, errlen) == 0) {
        fflush(fp);
        *len = ftell(fp);
        buf = *len > 0 ? (char *)malloc((size_t)*len) : NULL;
        rewind(fp);
        if (buf && fread(buf, 1, (size_t)*len, fp) != (size_t)*len) { free(buf); buf = NULL; }
    }
    fclose(fp);
    return buf;
}

/* the first differing byte, -1 when equal */
static long first_diff(const char *a, long al, const char *b, long bl) {
    const long n = al < bl ? al : bl;
    for (long i = 0; i < n; i++) if (a[i] != b[i]) return i;
    return al == bl ? -1 : n;
}

static void run_length(pulsar_engine *e, pulsar_session *A, pulsar_session *C, const pulsar_tokens *base, int P) {
    char err[320] = "";
    const int W = pulsar_engine_logits_width(e);
    float *la = (float *)malloc((size_t)W * sizeof(float));
    float *lc = (float *)malloc((size_t)W * sizeof(float));
    float *scratch = (float *)malloc((size_t)W * sizeof(float));
    int ta[N_SPEC > N_PRE ? N_SPEC : N_PRE], tc[N_SPEC > N_PRE ? N_SPEC : N_PRE];
    int ra[N_SPEC > N_PRE ? N_SPEC : N_PRE], rc[N_SPEC > N_PRE ? N_SPEC : N_PRE], na = 0, nc = 0;
    char *pa = NULL, *pc = NULL;
    long pal = 0, pcl = 0;
    pulsar_tokens p = { base->v, P, P };
    pulsar_tokens m;
    memset(&m, 0, sizeof m);
    fprintf(stderr, "prompt of %d tokens\n", P);
    {
    /* 1. COLD */
    pulsar_session_invalidate(A);
    pulsar_session_invalidate(C);
    const int s1 = pulsar_session_sync(A, &p, err, sizeof err);
    const int o1a = pulsar_session_resume_origin(A).raw();
    const int s1c = s1 == 0 ? pulsar_session_sync(C, &p, err, sizeof err) : 1;
    say(s1 == 0 && s1c == 0 && o1a == 0 && pulsar_session_resume_origin(C).raw() == 0,
        "1. COLD: both prefill from 0 (origins %d, %d) %s", o1a, pulsar_session_resume_origin(C).raw(), err);
    if (s1 || s1c) goto done;

    /* 2. REUSE: the same speculation on both, then the repeat */
    if (greedy_run(e, A, N_PRE, ta, ra, &na, scratch, err, sizeof err) != 0 ||
        greedy_run(e, C, N_PRE, tc, rc, &nc, scratch, err, sizeof err) != 0) {
        say(false, "2. speculate: %s", err);
        goto done;
    }
    say(!memcmp(ta, tc, sizeof(int) * N_PRE), "2. the two sessions decoded the same %d tokens before the repeat", N_PRE);
    pulsar_session_invalidate(C);
    const int s2 = pulsar_session_sync(A, &p, err, sizeof err);
    const int o2a = pulsar_session_resume_origin(A).raw();
    const int s2c = s2 == 0 ? pulsar_session_sync(C, &p, err, sizeof err) : 1;
    const int o2c = pulsar_session_resume_origin(C).raw();
    say(s2 == 0 && s2c == 0 && o2a == P && o2c == 0 && pulsar_session_pos(A).raw() == P,
        "2. REUSE: A resumes at its prompt end (origin %d of %d, pos %d), C prefills cold (origin %d) %s", o2a, P,
        pulsar_session_pos(A).raw(), o2c, err);
    if (s2 || s2c) goto done;

    /* 3. STATE */
    pulsar_session_copy_logits(A, la, W);
    pulsar_session_copy_logits(C, lc, W);
    pa = payload_of(A, &pal, err, sizeof err);
    pc = pa ? payload_of(C, &pcl, err, sizeof err) : NULL;
    const long d = pa && pc ? first_diff(pa, pal, pc, pcl) : 0;
    say(!memcmp(la, lc, (size_t)W * sizeof(float)) && pa && pc && pal == pcl,
        "3. STATE: logits byte-identical to the cold prefill's; payloads %ld / %ld B %s", pal, pcl, err);
    /* the payload byte for byte, reported range by range: it also carries storage no step reads before writing
     * (DeepSeek: DSpark ring rows past the ring's counters, which the cold side's previous decode wrote and the
     * record's capture did not) -- legs 4 and 5 grade everything a decode reads */
    if (pa && pc && pal == pcl) {
        long ranges = 0, bytes = 0;
        for (long i = 0; i + 8 < pal; i++) {   /* the trailing 8 bytes are the digest */
            if (pa[i] == pc[i]) continue;
            long j = i;
            while (j + 8 < pal && (pa[j] != pc[j] || (j + 64 < pal - 8 && memcmp(pa + j, pc + j, 64)))) j++;
            if (ranges < 8) fprintf(stderr, "         payload differs at [%ld, %ld), %ld B\n", i, j, j - i);
            ranges++;
            bytes += j - i;
            i = j;
        }
        fprintf(stderr, "  %s 3. STATE: payload byte-identical %s (%ld range(s), %ld B differ; logits end at %ld)\n",
                d < 0 ? "ok  " : "info", d < 0 ? "yes" : "no", ranges, bytes, (long)(32 + 4 * P + 4 + 16 + (long)W * 4));
    }

    /* 4. DECODE */
    const bool ok4 = greedy_run(e, A, N_SPEC, ta, ra, &na, scratch, err, sizeof err) == 0 &&
                     greedy_run(e, C, N_SPEC, tc, rc, &nc, scratch, err, sizeof err) == 0;
    int tok_diff = -1, round_diff = -1;
    for (int i = 0; ok4 && i < N_SPEC && tok_diff < 0; i++) if (ta[i] != tc[i]) tok_diff = i;
    for (int i = 0; ok4 && i < (na < nc ? na : nc) && round_diff < 0; i++) if (ra[i] != rc[i]) round_diff = i;
    if (ok4 && round_diff < 0 && na != nc) round_diff = na < nc ? na : nc;
    say(ok4 && tok_diff < 0 && round_diff < 0,
        "4. DECODE: %d greedy tokens in %d / %d rounds, identical (first differing token %d, round %d) %s", N_SPEC, na,
        nc, tok_diff, round_diff, err);

    /* 5. AGAIN */
    const int s5 = pulsar_session_sync(A, &p, err, sizeof err);
    const int o5 = pulsar_session_resume_origin(A).raw();
    pulsar_session_copy_logits(A, la, W);
    say(s5 == 0 && o5 == P && !memcmp(la, lc, (size_t)W * sizeof(float)),
        "5. AGAIN: after its own reuse and %d rounds A resumes at %d again (origin %d), logits == cold %s", na, P, o5,
        err);

    /* 6. MISS */
    for (int i = 0; i < P; i++) pulsar_tokens_push(&m, base->v[i]);
    m.v[P - 1] = base->v[P - 1] != base->v[P] ? base->v[P] : base->v[P + 1];
    const int s6 = pulsar_session_sync(A, &m, err, sizeof err);
    const int o6 = pulsar_session_resume_origin(A).raw();
    pulsar_session_copy_logits(A, la, W);
    pulsar_session_invalidate(C);
    const int s6c = s6 == 0 ? pulsar_session_sync(C, &m, err, sizeof err) : 1;
    pulsar_session_copy_logits(C, lc, W);
    say(s6 == 0 && s6c == 0 && o6 < P && !memcmp(la, lc, (size_t)W * sizeof(float)),
        "6. MISS: a different last token prefills (origin %d < %d), logits == cold %s", o6, P, err);
    }
done:
    pulsar_tokens_free(&m);
    free(pa);
    free(pc);
    free(la);
    free(lc);
    free(scratch);
}

int GATE_ENTRY(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [P]\n", argv[0]); return 2; }
    static const int lengths[] = { 51, 256, 300, 2085 };
    g_fail = 0;
    pulsar_engine *e = NULL;
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    pulsar_session *A = NULL, *C = NULL;
    pulsar_tokens base;
    memset(&base, 0, sizeof base);
    {
    if (!pulsar_engine_has_snapshots(e)) {
        say(false, "the family declares payloads (PULSAR_FAMILY_CAP_PAYLOAD)");
        goto out;
    }
    FILE *tf = fopen("tests/long_context_story_prompt.txt", "rb");
    if (!tf) { say(false, "prompt file"); goto out; }
    fseek(tf, 0, SEEK_END);
    const long tl = ftell(tf);
    rewind(tf);
    char *text = (char *)malloc((size_t)tl + 1);
    const bool rd = text && fread(text, 1, (size_t)tl, tf) == (size_t)tl;
    fclose(tf);
    if (!rd) { free(text); say(false, "prompt read"); goto out; }
    text[tl] = 0;
    pulsar_tokenize_text(e, text, &base);
    free(text);
    const int ctx = 2085 + 1024;
    if (base.len < 2085 + 2) { say(false, "prompt has %d tokens", base.len); goto out; }
    if (pulsar_session_create(&A, e, ctx) != 0 || pulsar_session_create(&C, e, ctx) != 0) {
        say(false, "session create");
        goto out;
    }
    if (argc > 2) run_length(e, A, C, &base, atoi(argv[2]));   /* one length, by hand */
    else
        for (size_t i = 0; i < sizeof(lengths) / sizeof(lengths[0]); i++) run_length(e, A, C, &base, lengths[i]);
    }
out:
    if (A) pulsar_session_free(A);
    if (C) pulsar_session_free(C);
    free(base.v);
    gate_engine_close(e);
    fprintf(stderr, "prompt-end gate: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}

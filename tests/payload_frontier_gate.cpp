/* L284: the session payload's round trip at a live FRONTIER, on real weights, through the public API only --
 * so it runs on any family with PULSAR_FAMILY_CAP_PAYLOAD (its first leg is Qwen's kv-state payload,
 * session_payload.cpp; DeepSeek's graph payload has its own byte-level gate, session_payload_gate).
 *
 * The frontier is off the grid and past the prefill (a prompt of P tokens, then D decoded), which is the
 * state a payload exists for: neither a grid checkpoint nor a segment chain holds it.
 *
 *   1. SIZE      payload_bytes() is what the save writes.
 *   2. FRONTIER  a fresh session that loads it stands at the same position with the same logits, and the
 *                next N decode steps give byte-identical rows (the recurrent lanes, the KV and index pools,
 *                the n-gram context -- everything a decode reads).
 *   3. DRAFTER   (a model with spec rounds) greedy speculation from the two sessions commits the same tokens
 *                in the same rounds: the drafter's state (Qwen: the MTP layer's trailing KV, its pending row
 *                and stage) survived, or the acceptance would differ.
 *   4. IDEMPOTENT a session loaded from the payload saves it again byte for byte (snapshot == file).
 *   5. RESUME    a sync that does not extend the frontier resumes from the carried grid checkpoint, from the
 *                same origin as the live session, and its logits equal a cold prefill's.
 *   6. DIGEST    one flipped data byte is refused as a digest mismatch.
 *
 * usage: ./tests/payload_frontier_gate MODEL [P] */
#include "pulsar.h"
#include "gate_entry.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_DECODE 6     /* decode rows between the prompt and the frontier */
#define N_CHECK 8      /* decode rows compared after the load */
#define N_SPEC 32      /* greedy tokens speculated from both sessions */

static pulsar_session *g_s[3];
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

static char *read_all(FILE *fp, long *len) {
    fflush(fp);
    *len = ftell(fp);
    if (*len <= 0) return NULL;
    char *buf = (char *)malloc((size_t)*len);
    rewind(fp);
    if (!buf || fread(buf, 1, (size_t)*len, fp) != (size_t)*len) { free(buf); return NULL; }
    return buf;
}

static int load_bytes(pulsar_session *s, const char *bytes, long len, char *err, size_t errlen) {
    FILE *fp = fmemopen((void *)bytes, (size_t)len, "rb");
    if (!fp) { snprintf(err, errlen, "fmemopen"); return 1; }
    const int rc = pulsar_session_load_payload(s, fp, (uint64_t)len, err, errlen);
    fclose(fp);
    return rc;
}

/* greedy speculation to n tokens: the tokens, and each round's count (the drafter's acceptance) */
static int spec_run(pulsar_session *s, int n, int *toks, int *rounds, int *n_rounds, char *err, size_t errlen) {
    uint64_t rng = 7;
    int got = 0;
    *n_rounds = 0;
    while (got < n) {
        int buf[17];
        const int k = pulsar_session_generate_speculative(s, 0.0f, 0, 1.0f, 0.0f, &rng, n - got, buf, 17, err,
                                                          errlen);
        if (k <= 0) return 1;
        rounds[(*n_rounds)++] = k;
        for (int i = 0; i < k && got < n; i++) toks[got++] = buf[i];
    }
    return 0;
}

int GATE_ENTRY(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [P]\n", argv[0]); return 2; }
    const int P = argc > 2 ? atoi(argv[2]) : 2085;   /* off the 128 grid: the payload carries the checkpoint below */
    const int ctx = P + 1024;
    memset(g_s, 0, sizeof(g_s));
    g_fail = 0;
    pulsar_engine *e = NULL;
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    char err[320] = "";
    char *f1 = NULL;
    long f1_len = 0;
    float *lg = NULL;
    pulsar_tokens base;
    memset(&base, 0, sizeof base);
    pulsar_session_snapshot snap;
    memset(&snap, 0, sizeof snap);
    FILE *fp = NULL;
    const int W = pulsar_engine_logits_width(e);
    const int T = P + N_DECODE;
    {
    if (!pulsar_engine_has_snapshots(e)) {
        say(false, "the family declares payloads (PULSAR_FAMILY_CAP_PAYLOAD)");
        goto done;
    }
    FILE *tf = fopen("tests/long_context_story_prompt.txt", "rb");
    if (!tf) { say(false, "prompt file"); goto done; }
    fseek(tf, 0, SEEK_END);
    const long tl = ftell(tf);
    rewind(tf);
    char *text = (char *)malloc((size_t)tl + 1);
    const bool rd = text && fread(text, 1, (size_t)tl, tf) == (size_t)tl;
    fclose(tf);
    if (!rd) { free(text); say(false, "prompt read"); goto done; }
    text[tl] = 0;
    pulsar_tokenize_text(e, text, &base);
    free(text);
    if (base.len < T + N_CHECK + 4) { say(false, "prompt has %d tokens, need %d", base.len, T + N_CHECK + 4); goto done; }
    lg = (float *)malloc((size_t)W * (2 * N_CHECK + 4) * sizeof(float));
    float *ref0 = lg, *refA = lg + W, *got = refA + (size_t)W * N_CHECK, *cold = got + W;
    for (int i = 0; i < 3; i++)
        if (pulsar_session_create(&g_s[i], e, ctx) != 0) { say(false, "session %d create", i); goto done; }
    pulsar_session *A = g_s[0], *B = g_s[1], *X = g_s[2];

    /* A: prompt, D decode rows, the frontier saved */
    pulsar_tokens pp = { base.v, P, P };
    if (pulsar_session_sync(A, &pp, err, sizeof err) != 0) { say(false, "A sync: %s", err); goto done; }
    for (int i = P; i < T; i++)
        if (pulsar_session_eval(A, base.v[i], err, sizeof err) != 0) { say(false, "A eval: %s", err); goto done; }
    pulsar_session_copy_logits(A, ref0, W);
    const uint64_t pb = pulsar_session_payload_bytes(A);
    fp = tmpfile();
    const int src = fp ? pulsar_session_save_payload(A, fp, err, sizeof err) : 1;
    f1 = src == 0 ? read_all(fp, &f1_len) : NULL;
    say(src == 0 && f1 && (uint64_t)f1_len == pb, "1. SIZE: saved at %d (prompt %d): payload_bytes %llu, written %ld %s",
        T, P, (unsigned long long)pb, f1_len, err);
    if (!f1) goto done;

    /* A decodes on from its live state -- the reference */
    for (int i = 0; i < N_CHECK; i++) {
        if (pulsar_session_eval(A, base.v[T + i], err, sizeof err) != 0) { say(false, "A eval: %s", err); goto done; }
        pulsar_session_copy_logits(A, refA + (size_t)W * i, W);
    }

    /* B: load, the same frontier and the same next rows */
    err[0] = '\0';
    if (load_bytes(B, f1, f1_len, err, sizeof err) != 0) { say(false, "2. FRONTIER: load: %s", err); goto done; }
    pulsar_session_copy_logits(B, got, W);
    say(pulsar_session_pos(B) == T && !memcmp(got, ref0, (size_t)W * sizeof(float)),
        "2. FRONTIER: loaded at pos %d, logits byte-identical to the saved session's", pulsar_session_pos(B));
    int first_diff = -1;
    for (int i = 0; i < N_CHECK && first_diff < 0; i++) {
        if (pulsar_session_eval(B, base.v[T + i], err, sizeof err) != 0) { say(false, "B eval: %s", err); goto done; }
        pulsar_session_copy_logits(B, got, W);
        if (memcmp(got, refA + (size_t)W * i, (size_t)W * sizeof(float))) first_diff = i;
    }
    say(first_diff < 0, "2. FRONTIER: %d decode rows after the load byte-identical (first differing row %d)", N_CHECK,
        first_diff);

    /* both sessions stand at the same state: greedy speculation from each */
    if (pulsar_engine_has_spec_rounds(e)) {
        int ta[N_SPEC], tb[N_SPEC], ra[N_SPEC], rb[N_SPEC], na = 0, nb = 0;
        const bool ok = spec_run(A, N_SPEC, ta, ra, &na, err, sizeof err) == 0 &&
                        spec_run(B, N_SPEC, tb, rb, &nb, err, sizeof err) == 0;
        say(ok && na == nb && !memcmp(ta, tb, sizeof ta) && !memcmp(ra, rb, (size_t)na * sizeof(int)),
            "3. DRAFTER: %d greedy tokens in %d / %d rounds, tokens and round sizes identical %s", N_SPEC, na, nb, err);
    } else {
        fprintf(stderr, "  --   3. DRAFTER: the model has no spec rounds\n");
    }

    /* B again from the file (a load over a live session), and saved back: the same bytes */
    if (load_bytes(B, f1, f1_len, err, sizeof err) != 0) { say(false, "4. reload: %s", err); goto done; }
    const int ssv = pulsar_session_save_snapshot(B, &snap, err, sizeof err);
    say(ssv == 0 && (long)snap.len == f1_len && !memcmp(snap.ptr, f1, (size_t)f1_len),
        "4. IDEMPOTENT: the loaded session's snapshot is the file byte for byte (%llu B) %s",
        (unsigned long long)snap.len, err);

    /* a sync that diverges past the prefill frontier: B resumes from the carried checkpoint, as A does */
    pulsar_tokens q;
    memset(&q, 0, sizeof q);
    for (int i = 0; i < P + 3; i++) pulsar_tokens_push(&q, base.v[i]);
    q.v[P + 2] = base.v[7] != base.v[P + 2] ? base.v[7] : base.v[8];
    const int sa = pulsar_session_sync(A, &q, err, sizeof err);
    const int oa = pulsar_session_resume_origin(A);
    const int sb = sa == 0 ? pulsar_session_sync(B, &q, err, sizeof err) : 1;
    const int ob = pulsar_session_resume_origin(B);
    pulsar_session_copy_logits(B, got, W);
    const int sx = sb == 0 ? pulsar_session_sync(X, &q, err, sizeof err) : 1;
    pulsar_session_copy_logits(X, cold, W);
    say(sx == 0 && oa == ob && ob > 0 && !memcmp(got, cold, (size_t)W * sizeof(float)),
        "5. RESUME: origins live %d, restored %d; restored logits == cold prefill's %s", oa, ob, err);
    pulsar_tokens_free(&q);

    /* one flipped data byte (the trailing 8 are the digest) */
    {
        char *bad = (char *)malloc((size_t)f1_len);
        memcpy(bad, f1, (size_t)f1_len);
        bad[f1_len - 9] ^= (char)0xFF;
        err[0] = '\0';
        const int crc = load_bytes(X, bad, f1_len, err, sizeof err);
        say(crc != 0 && strstr(err, "digest mismatch"), "6. DIGEST: a flipped byte refused (%s)", err);
        free(bad);
    }
    }
done:
    if (fp) fclose(fp);
    pulsar_session_snapshot_free(&snap);
    for (int i = 0; i < 3; i++) if (g_s[i]) pulsar_session_free(g_s[i]);
    memset(g_s, 0, sizeof(g_s));
    free(f1);
    free(lg);
    free(base.v);
    gate_engine_close(e);
    fprintf(stderr, "payload-frontier gate: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}

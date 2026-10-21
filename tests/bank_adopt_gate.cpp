/* L313: bank-independent checkpoints -- a bank that adopts ANOTHER bank's published state (pulsar_session_bank_adopt:
 * the source's pool rows below T + its grid checkpoint or record at T) and prefills the rest is the cold prefill's
 * state byte for byte, on real weights, through the public API only.  The reuse == cold must-pass, across banks.
 *
 * Bank 0 prefills a chat prompt P (a long system message + a user turn + the generation header) and decodes a few
 * greedy tokens past it -- the main conversation, busy.  Then for each shape Q:
 *   EXT    P plus 211 more tokens (the main conversation's next turn): Qwen / Nemotron adopt P's prompt end, DeepSeek
 *          the grid checkpoint below it;
 *   SIDE   P to its message endpoint (the generation header's start) plus another user turn and its header (Claude
 *          Code's side request): split-invariant families adopt the endpoint record, DeepSeek the grid point below;
 *   SAME   P itself: every family adopts P's prompt end with its logits and evaluates no row;
 * bank 1 adopts bank 0's state for Q and syncs Q; bank 2 syncs Q cold from nothing; their logits after the sync and
 * N greedy tokens after it must be identical.  Last, bank 0 -- which lent its state -- still resumes P at its prompt
 * end with the cold logits.
 *
 * usage: ./tests/bank_adopt_gate MODEL */
#include "pulsar.h"
#include "gate_entry.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_PRE 8
#define N_GEN 16

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

static bool install(pulsar_session *s, int *live, uint32_t b) {
    if (*live >= 0) pulsar_session_bank_state_save(s, (uint32_t)*live);
    if (!pulsar_session_bank_state_restore(s, b)) return false;
    *live = (int)b;
    return true;
}

static bool greedy(pulsar_session *s, int n, int *out, char *err, size_t errlen) {
    for (int i = 0; i < n; i++) {
        const int t = pulsar_session_argmax(s);
        if (t < 0 || pulsar_session_eval(s, t, err, errlen) != 0) return false;
        out[i] = t;
    }
    return true;
}

static void shape(pulsar_engine *e, pulsar_session *s, int *live, const char *name, const pulsar_tokens *Q) {
    char err[320] = "";
    const int W = pulsar_engine_logits_width(e);
    float *la = (float *)malloc((size_t)W * sizeof(float)), *lc = (float *)malloc((size_t)W * sizeof(float));
    int ta[N_GEN], tc[N_GEN];
    /* bank 1 adopts */
    bool ok = install(s, live, 1);
    if (ok) pulsar_session_invalidate(s);
    const int T = ok ? pulsar_session_bank_adopt(s, 0, Q, err, sizeof err) : -1;
    const int rs = T >= 0 ? pulsar_session_sync(s, Q, err, sizeof err) : 1;
    const int origin = pulsar_session_resume_origin(s);
    if (rs == 0) pulsar_session_copy_logits(s, la, W);
    const bool ga = rs == 0 && greedy(s, N_GEN, ta, err, sizeof err);
    say(T > 0 && rs == 0 && ga, "%s: bank 1 adopts bank 0's state at %d of %d and syncs (origin %d) %s", name, T, Q->len,
        origin, err);
    /* bank 2 cold */
    ok = install(s, live, 2);
    if (ok) pulsar_session_invalidate(s);
    const int rc = ok ? pulsar_session_sync(s, Q, err, sizeof err) : 1;
    const int origin_c = pulsar_session_resume_origin(s);
    if (rc == 0) pulsar_session_copy_logits(s, lc, W);
    const bool gc = rc == 0 && greedy(s, N_GEN, tc, err, sizeof err);
    say(rc == 0 && gc && origin_c == 0, "%s: bank 2 prefills it cold (origin %d) %s", name, origin_c, err);
    int diff = -1;
    for (int i = 0; ga && gc && i < N_GEN && diff < 0; i++) if (ta[i] != tc[i]) diff = i;
    say(rs == 0 && rc == 0 && !memcmp(la, lc, (size_t)W * sizeof(float)) && ga && gc && diff < 0,
        "%s: REUSE == COLD -- logits byte-identical, %d greedy tokens identical (first differing %d)", name, N_GEN, diff);
    free(la);
    free(lc);
}

int GATE_ENTRY(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }
    g_fail = 0;
    pulsar_engine *e = NULL;
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    pulsar_session *s = NULL;
    pulsar_tokens P = {0}, side = {0}, ext = {0}, story = {0}, Q = {0};
    char *text = NULL;
    int live = -1;
    {
    FILE *tf = fopen("tests/long_context_story_prompt.txt", "rb");
    if (!tf) { say(false, "prompt file"); goto out; }
    fseek(tf, 0, SEEK_END);
    const long tl = ftell(tf);
    rewind(tf);
    text = (char *)calloc(1, (size_t)tl + 1);
    const bool rd = fread(text, 1, (size_t)tl, tf) == (size_t)tl;
    fclose(tf);
    if (!rd) { say(false, "prompt read"); goto out; }
    text[12000] = '\0';   /* ~3k tokens of system message */
    pulsar_encode_chat_prompt(e, text, "Who is the story about? Answer in one sentence.", PULSAR_THINK_NONE, &P);
    pulsar_encode_chat_prompt(e, text, "Describe your most recent action in 3-5 words.", PULSAR_THINK_NONE, &side);
    pulsar_tokenize_text(e, text + 12001, &story);
    pulsar_turn_markers m;
    if (!pulsar_chat_turn_markers(e, &m)) { say(false, "the tokenizer spells turn markers"); goto out; }
    int E = -1, U = -1;
    for (int i = P.len - 1; i > 0 && E < 0; i--) if (pulsar_turn_marker_at(&m, P.v, P.len, i) != 0) E = i;
    for (int i = side.len - 1; i > 0 && U < 0; i--) if (pulsar_turn_marker_at(&m, side.v, side.len, i) == 1) U = i;
    if (E <= 0 || U <= 0 || story.len < 211) { say(false, "P's endpoint %d, the side turn's marker %d", E, U); goto out; }
    fprintf(stderr, "bank-adopt: P %d tokens, its message endpoint %d\n", P.len, E);
    pulsar_engine_set_bank_pool(4);
    if (pulsar_session_create(&s, e, P.len + 4096) != 0 || pulsar_session_bank_count(s) != 4) {
        say(false, "a 4-bank session");
        goto out;
    }
    /* bank 0: the main conversation, past its prompt */
    char err[320] = "";
    int pre[N_PRE];
    const bool b0 = install(s, &live, 0) && pulsar_session_sync(s, &P, err, sizeof err) == 0 &&
                    greedy(s, N_PRE, pre, err, sizeof err);
    say(b0, "bank 0 prefilled P and decoded %d tokens past it %s", N_PRE, err);
    if (!b0) goto out;
    /* EXT */
    for (int i = 0; i < P.len; i++) pulsar_tokens_push(&ext, P.v[i]);
    for (int i = 0; i < 211; i++) pulsar_tokens_push(&ext, story.v[i]);
    shape(e, s, &live, "EXT", &ext);
    /* SIDE: P to its endpoint, then the side request's own user turn and header */
    for (int i = 0; i < E; i++) pulsar_tokens_push(&Q, P.v[i]);
    for (int i = U; i < side.len; i++) pulsar_tokens_push(&Q, side.v[i]);
    shape(e, s, &live, "SIDE", &Q);
    /* SAME */
    shape(e, s, &live, "SAME", &P);
    /* the lender: bank 0 still resumes P at its prompt end, with the cold logits (bank 2 last synced P cold) */
    const int W = pulsar_engine_logits_width(e);
    float *l2 = (float *)malloc((size_t)W * sizeof(float)), *l0 = (float *)malloc((size_t)W * sizeof(float));
    bool okc = install(s, &live, 2) && pulsar_session_sync(s, &P, err, sizeof err) == 0;   /* resumes at its own end */
    if (okc) pulsar_session_copy_logits(s, l2, W);
    const bool ok0 = okc && install(s, &live, 0) && pulsar_session_sync(s, &P, err, sizeof err) == 0;
    const int o0 = pulsar_session_resume_origin(s);
    if (ok0) pulsar_session_copy_logits(s, l0, W);
    say(ok0 && o0 == P.len && !memcmp(l0, l2, (size_t)W * sizeof(float)),
        "LENDER: bank 0 resumes P at its prompt end (origin %d of %d), logits == cold %s", o0, P.len, err);
    free(l0);
    free(l2);
    }
out:
    if (s) pulsar_session_free(s);
    pulsar_tokens_free(&P);
    pulsar_tokens_free(&side);
    pulsar_tokens_free(&ext);
    pulsar_tokens_free(&story);
    pulsar_tokens_free(&Q);
    free(text);
    gate_engine_close(e);
    fprintf(stderr, "bank-adopt gate: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}

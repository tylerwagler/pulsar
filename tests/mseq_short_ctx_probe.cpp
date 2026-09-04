/* L161 GATE: a 1-row decode step and an N-row decode step are the same numerics.
 *
 * Promoted from the L160 probe that found the split (rows/L161.md).  Exits
 * non-zero when any pair below differs by one bit.
 *
 *
 * The spec-sampling oracle's banked rewrite reproduced the serial gate byte for
 * byte at one row per forward and diverged on ~20% of draws at two or more
 * rows, deterministically per trajectory and independent of which bank or
 * which batchmates -- at a context of ~30 tokens.  The battery's neutrality
 * gates assert row-neutrality at positions 100..240.  This probe asks the
 * question directly at the oracle's own state: the same token decoded on the
 * same prefilled bank by (C) classic eval, (A) a 1-row decode_mixed step, (B)
 * a 2-row step with a second bank, (D) a 2-row step with a different token on
 * the second bank.  Prints per-pair differing logits, max |delta| and argmax.
 *
 *   ./tests/mseq_short_ctx_probe MODEL [filler_tokens] [one|two]
 *
 * With a third argument the probe runs ONLY that step (one bank, 1-row step /
 * two banks, 2-row step) so a per-stage dump (PULSAR_CUDA_GRAPH_DUMP_PREFIX,
 * _POS=<pos>) captures exactly one step per process; tools/dumpcmp_rows.py
 * DIR 1 1 --batched two --solo one then names the first differing stage.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pulsar.h"

static const char *PROMPT =
    "The economic history of the Mediterranean is inseparable from its ports. ";

static int g_fail = 0;
static void cmp(const char *label, const float *a, const float *b, int n) {
    int nd = 0, ia = 0, ib = 0; float md = 0.0f;
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) { nd++; const float d = fabsf(a[i] - b[i]); if (d > md) md = d; }
        if (a[i] > a[ia]) ia = i;
        if (b[i] > b[ib]) ib = i;
    }
    printf("%-34s differing=%6d/%d  max|delta|=%.6g  argmax %d vs %d %s\n",
           label, nd, n, (double)md, ia, ib, nd == 0 ? "IDENTICAL" : "DIFFERS");
    if (nd) g_fail = 1;
}

static int bank_sync(pulsar_session *s, uint32_t b, const pulsar_tokens *p, char *err, size_t errlen) {
    if (pulsar_session_bank_repoint(s, b) != 0) { snprintf(err, errlen, "repoint %u", b); return 1; }
    pulsar_session_invalidate(s);
    if (pulsar_session_sync(s, p, err, errlen) != 0) return 1;
    pulsar_session_bank_state_save(s, b);
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [filler]\n", argv[0]); return 2; }
    const int filler = argc > 2 ? atoi(argv[2]) : 0;
    const char *only = argc > 3 ? argv[3] : NULL;
    pulsar_engine_options opt = { .model_path = argv[1], .backend = PULSAR_BACKEND_CUDA };
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    if (setenv("PULSAR_MSEQ_BANKS", "2", 1) != 0) return 1;
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, e, 4096) != 0) { fprintf(stderr, "session failed\n"); return 1; }
    if (pulsar_session_bank_count(s) < 2) { fprintf(stderr, "pool has %d banks\n", pulsar_session_bank_count(s)); return 1; }

    char *user = NULL;
    if (filler > 0) {
        const size_t cap = (size_t)filler * 8u + strlen(PROMPT) + 64u;
        user = (char *)malloc(cap);
        size_t off = 0;
        for (int i = 0; i < filler && off + 8 < cap; i++)
            off += (size_t)snprintf(user + off, cap - off, "port%d ", i % 997);
        snprintf(user + off, cap - off, "%s", PROMPT);
    }
    pulsar_tokens prompt = {0};
    pulsar_chat_begin(e, &prompt);
    pulsar_chat_append_message(e, &prompt, "user", user ? user : PROMPT);
    pulsar_chat_append_assistant_prefix(e, &prompt, PULSAR_THINK_NONE);

    char err[256];
    const int vw = pulsar_engine_logits_width(e);
    float *C = (float *)malloc((size_t)vw * sizeof(float));
    float *A = (float *)malloc((size_t)vw * sizeof(float));
    float *B = (float *)malloc((size_t)2 * vw * sizeof(float));
    float *D = (float *)malloc((size_t)2 * vw * sizeof(float));

    /* C: classic eval on bank 0 */
    if (bank_sync(s, 0, &prompt, err, sizeof(err))) { fprintf(stderr, "sync: %s\n", err); return 1; }
    const int pos = pulsar_session_pos(s);
    const int tok = pulsar_session_argmax(s);
    int tok2 = 0;
    {
        float *lg = (float *)malloc((size_t)vw * sizeof(float));
        pulsar_session_copy_logits(s, lg, vw);
        float best = -INFINITY;
        for (int i = 0; i < vw; i++) if (i != tok && lg[i] > best) { best = lg[i]; tok2 = i; }
        free(lg);
    }
    printf("ctx pos=%d  tok=%d  tok2=%d  vocab=%d\n", pos, tok, tok2, vw);
    if (only) {
        /* dump mode: exactly one step in this process */
        if (!strcmp(only, "one")) {
            pulsar_multiseq_req r = { 0u, pos, tok };
            uint32_t got = 0;
            if (pulsar_session_decode_mixed(s, &r, 1, A, vw, &got, 0u, err, sizeof(err)) != 0 || got != 1) {
                fprintf(stderr, "1-row step: %s\n", err); return 1;
            }
        } else {
            if (bank_sync(s, 1, &prompt, err, sizeof(err))) return 1;
            pulsar_multiseq_req r[2] = { { 0u, pos, tok }, { 1u, pos, tok } };
            uint32_t got = 0;
            if (pulsar_session_decode_mixed(s, r, 2, B, 2 * vw, &got, 0u, err, sizeof(err)) != 0 || got != 2) {
                fprintf(stderr, "2-row step: %s\n", err); return 1;
            }
        }
        printf("dump step done (%s)\n", only);
        return 0;
    }
    if (pulsar_session_eval(s, tok, err, sizeof(err)) != 0) { fprintf(stderr, "eval: %s\n", err); return 1; }
    pulsar_session_copy_logits(s, C, vw);

    /* A: 1-row decode_mixed on bank 0 */
    if (bank_sync(s, 0, &prompt, err, sizeof(err))) return 1;
    {
        pulsar_multiseq_req r = { 0u, pos, tok };
        uint32_t got = 0;
        if (pulsar_session_decode_mixed(s, &r, 1, A, vw, &got, 0u, err, sizeof(err)) != 0 || got != 1) {
            fprintf(stderr, "1-row step: %s\n", err); return 1;
        }
    }
    /* B: 2-row step, both banks the same token */
    if (bank_sync(s, 0, &prompt, err, sizeof(err))) return 1;
    if (bank_sync(s, 1, &prompt, err, sizeof(err))) return 1;
    {
        pulsar_multiseq_req r[2] = { { 0u, pos, tok }, { 1u, pos, tok } };
        uint32_t got = 0;
        if (pulsar_session_decode_mixed(s, r, 2, B, 2 * vw, &got, 0u, err, sizeof(err)) != 0 || got != 2) {
            fprintf(stderr, "2-row step: %s\n", err); return 1;
        }
    }
    /* D: 2-row step, second bank a different token */
    if (bank_sync(s, 0, &prompt, err, sizeof(err))) return 1;
    if (bank_sync(s, 1, &prompt, err, sizeof(err))) return 1;
    {
        pulsar_multiseq_req r[2] = { { 0u, pos, tok }, { 1u, pos, tok2 } };
        uint32_t got = 0;
        if (pulsar_session_decode_mixed(s, r, 2, D, 2 * vw, &got, 0u, err, sizeof(err)) != 0 || got != 2) {
            fprintf(stderr, "2-row step (D): %s\n", err); return 1;
        }
    }
    cmp("A(1-row) vs C(classic eval)", A, C, vw);
    cmp("B row0 (2-row) vs A(1-row)", B, A, vw);
    cmp("B row1 (bank 1) vs B row0", B + vw, B, vw);
    cmp("D row0 vs B row0 (batchmate diff)", D, B, vw);
    cmp("B row0 vs C(classic)", B, C, vw);
    printf("ROW NEUTRALITY GATE: %s\n", g_fail ? "FAIL" : "PASS");
    free(user); free(C); free(A); free(B); free(D);
    pulsar_session_free(s);
    pulsar_engine_close(e);
    return g_fail;
}

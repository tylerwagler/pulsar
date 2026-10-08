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
#include "gate_entry.h"
#include "gate_util.h"

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

int GATE_ENTRY(int argc, char **argv) {
    g_fail = 0;
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [filler]\n", argv[0]); return 2; }
    const int filler = argc > 2 ? atoi(argv[2]) : 0;
    const char *only = argc > 3 ? argv[3] : NULL;
    pulsar_engine_options opt = { .model_path = argv[1], .backend = PULSAR_BACKEND_CUDA };
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    /* The engine parses PULSAR_MSEQ_BANKS once per process, so after the open
     * this API is the only way to size the pool the next session gets. */
    pulsar_engine_set_bank_pool(2u);
    pulsar_session *s = NULL;
    char *user = NULL;
    pulsar_tokens prompt = {0};
    float *C = NULL, *A = NULL, *B = NULL, *D = NULL;
    int rc = 1;
    {
        if (filler > 0) {
            const size_t cap = (size_t)filler * 8u + strlen(PROMPT) + 64u;
            user = (char *)malloc(cap);
            size_t off = 0;
            for (int i = 0; i < filler && off + 8 < cap; i++)
                off += (size_t)snprintf(user + off, cap - off, "port%d ", i % 997);
            snprintf(user + off, cap - off, "%s", PROMPT);
        }
        /* the family's one-turn render (L278: pulsar_chat_begin is DeepSeek's template and ends a Qwen run) */
        pulsar_encode_chat_prompt(e, NULL, user ? user : PROMPT, PULSAR_THINK_NONE, &prompt);
        /* The context from the rendered prompt (L278: the filler's tokens per word are the family's tokenizer's --
         * DeepSeek 1100 -> ~2217 tokens, Qwen 1100 -> 4205), so the deep entries (L170, L175) fit on every family. */
        int ctx = 4096;
        while (ctx < prompt.len + 512) ctx *= 2;
        if (pulsar_session_create(&s, e, ctx) != 0) { fprintf(stderr, "session failed (ctx %d)\n", ctx); goto done; }
        if (pulsar_session_bank_count(s) < 2) { fprintf(stderr, "pool has %d banks\n", pulsar_session_bank_count(s)); goto done; }

        char err[256];
        const int vw = pulsar_engine_logits_width(e);
        C = (float *)malloc((size_t)vw * sizeof(float));
        A = (float *)malloc((size_t)vw * sizeof(float));
        B = (float *)malloc((size_t)2 * vw * sizeof(float));
        D = (float *)malloc((size_t)2 * vw * sizeof(float));

        /* C: classic eval on bank 0 */
        if (!gate_bank_sync(s, 0, prompt.v, prompt.len, NULL, "row-neutrality")) goto done;
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
                    fprintf(stderr, "1-row step: %s\n", err); goto done;
                }
            } else {
                if (!gate_bank_sync(s, 1, prompt.v, prompt.len, NULL, "row-neutrality")) goto done;
                pulsar_multiseq_req r[2] = { { 0u, pos, tok }, { 1u, pos, tok } };
                uint32_t got = 0;
                if (pulsar_session_decode_mixed(s, r, 2, B, 2 * vw, &got, 0u, err, sizeof(err)) != 0 || got != 2) {
                    fprintf(stderr, "2-row step: %s\n", err); goto done;
                }
            }
            printf("dump step done (%s)\n", only);
            rc = 0;
            goto done;
        }
        if (pulsar_session_eval(s, tok, err, sizeof(err)) != 0) { fprintf(stderr, "eval: %s\n", err); goto done; }
        pulsar_session_copy_logits(s, C, vw);

        /* A: 1-row decode_mixed on bank 0 */
        if (!gate_bank_sync(s, 0, prompt.v, prompt.len, NULL, "row-neutrality")) goto done;
        {
            pulsar_multiseq_req r = { 0u, pos, tok };
            uint32_t got = 0;
            if (pulsar_session_decode_mixed(s, &r, 1, A, vw, &got, 0u, err, sizeof(err)) != 0 || got != 1) {
                fprintf(stderr, "1-row step: %s\n", err); goto done;
            }
        }
        /* B: 2-row step, both banks the same token */
        if (!gate_bank_sync(s, 0, prompt.v, prompt.len, NULL, "row-neutrality")) goto done;
        if (!gate_bank_sync(s, 1, prompt.v, prompt.len, NULL, "row-neutrality")) goto done;
        {
            pulsar_multiseq_req r[2] = { { 0u, pos, tok }, { 1u, pos, tok } };
            uint32_t got = 0;
            if (pulsar_session_decode_mixed(s, r, 2, B, 2 * vw, &got, 0u, err, sizeof(err)) != 0 || got != 2) {
                fprintf(stderr, "2-row step: %s\n", err); goto done;
            }
        }
        /* D: 2-row step, second bank a different token */
        if (!gate_bank_sync(s, 0, prompt.v, prompt.len, NULL, "row-neutrality")) goto done;
        if (!gate_bank_sync(s, 1, prompt.v, prompt.len, NULL, "row-neutrality")) goto done;
        {
            pulsar_multiseq_req r[2] = { { 0u, pos, tok }, { 1u, pos, tok2 } };
            uint32_t got = 0;
            if (pulsar_session_decode_mixed(s, r, 2, D, 2 * vw, &got, 0u, err, sizeof(err)) != 0 || got != 2) {
                fprintf(stderr, "2-row step (D): %s\n", err); goto done;
            }
        }
        cmp("A(1-row) vs C(classic eval)", A, C, vw);
        cmp("B row0 (2-row) vs A(1-row)", B, A, vw);
        cmp("B row1 (bank 1) vs B row0", B + vw, B, vw);
        cmp("D row0 vs B row0 (batchmate diff)", D, B, vw);
        cmp("B row0 vs C(classic)", B, C, vw);
        printf("ROW NEUTRALITY GATE: %s\n", g_fail ? "FAIL" : "PASS");
        rc = g_fail;
    }
done:
    free(user); free(C); free(A); free(B); free(D);
    pulsar_tokens_free(&prompt);
    pulsar_session_free(s);
    gate_engine_close(e);
    return rc;
}

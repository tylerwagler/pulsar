/* L251: end-to-end generation for the Qwen4-exp lane -- token ids in, token ids out.
 *
 * The reference gate proves FORWARDS: one row per recorded depth, graded against anchors.  This
 * proves the lane PRODUCES TOKENS, walking the same session lane the gate walks -- sync() to
 * extend the prefix by one (which leaves that prefix's next-token logits), copy_logits(), argmax.
 *
 * It exists because the family has no tokenizer or renderer yet (S5), so the CLI's text path
 * refuses by design ("only --inspect runs").  Feeding ids and printing ids keeps S5 out of the
 * measurement; `qwen_generate_decode.py` then decodes the emitted ids with the reference
 * tokenizer, which is the only way to ask whether the generation is coherent.
 *
 *   ./tests/qwen_generate <container> <tokens.bin> <n_predict> [out.bin]
 *
 * out.bin is the PROMPT'S ids followed by the generated ones, as int32 LE, so a caller can
 * decode the whole thing in one pass.  Not part of the battery: it needs a real container and
 * a tokens.bin on disk.
 *
 * NOTE ON WHAT THIS MEASURES: the container's weights are EXL3 (and the QSA KV cache is MXFP8),
 * so a greedy stream here is NOT expected to equal the BF16 source's token for token.
 */
#include "pulsar.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "qwen-generate: out of memory\n"); exit(2); }
    return p;
}

static unsigned char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    const long sz = ftell(f);
    if (sz <= 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    unsigned char *b = (unsigned char *)xmalloc((size_t)sz);
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(b); return NULL; }
    fclose(f);
    *n = (size_t)sz;
    return b;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <container> <tokens.bin> <n_predict> [out.bin]\n", argv[0]);
        return 2;
    }
    const int n_predict = atoi(argv[3]);
    if (n_predict <= 0) { fprintf(stderr, "qwen-generate: n_predict must be > 0\n"); return 2; }

    size_t tn = 0;
    unsigned char *tb = slurp(argv[2], &tn);
    if (!tb || tn % sizeof(int32_t) != 0) {
        fprintf(stderr, "qwen-generate: %s is not a non-empty int32 array\n", argv[2]);
        return 2;
    }
    const int n_tok = (int)(tn / sizeof(int32_t));

    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) {
        fprintf(stderr, "qwen-generate: %s did not open\n", argv[1]);
        return 2;
    }
    const int W = pulsar_engine_logits_width(e);

    /* One slot per prompt token, one per generated token, a little headroom for the
     * allocator's own rounding -- the gate sizes its session the same way. */
    const int cap = n_tok + n_predict + 8;
    pulsar_session *sess = NULL;
    if (pulsar_session_create(&sess, e, cap) != 0) {
        fprintf(stderr, "qwen-generate: session of %d tokens refused\n", cap);
        return 2;
    }

    int32_t *ids = (int32_t *)xmalloc(sizeof(int32_t) * (size_t)cap);
    memcpy(ids, tb, tn);
    free(tb);
    float *row = (float *)xmalloc(sizeof(float) * (size_t)W);

    printf("qwen-generate: %s, prompt %d tokens, width %d, greedy %d tokens\n",
           argv[1], n_tok, W, n_predict);
    printf("tokens:");
    fflush(stdout);

    /* NLL MODE (QWEN_NLL_PREFIX=P): prefill tokens[0:P] with sync(), then TEACHER-FORCE the rest of
     * tokens.bin through eval() -- at each step the logprob of the true next token and whether the
     * argmax is it.  The decode steps are the same code in every build, so two builds that differ
     * only in their PREFILL kernels differ here only through the state the prefill left: the
     * measurement for "did the prefill path cost quality", over hundreds of positions instead of a
     * handful of argmaxes.  n_predict caps the teacher-forced steps. */
    const char *nll_env = getenv("QWEN_NLL_PREFIX");
    if (nll_env && nll_env[0]) {
        const int P = atoi(nll_env);
        const int N = n_tok - P < n_predict ? n_tok - P : n_predict;
        if (P <= 0 || N <= 0) { fprintf(stderr, "qwen-generate: QWEN_NLL_PREFIX=%d leaves no tokens to score\n", P); return 2; }
        char err[256] = "";
        const double t0 = now_s();
        pulsar_tokens t = { ids, P, P };
        if (pulsar_session_sync(sess, &t, err, sizeof(err)) != 0) { fprintf(stderr, "qwen-generate: sync: %s\n", err); return 1; }
        double nll = 0.0, mass_end = 0.0, mass_pad = 0.0;
        int top1 = 0, scored = 0;
        /* QWEN_NLL_MODE=sync scores every position through the PREFILL path instead (sync() of the
         * prefix extended by one -- a one-row prefill step), so eval-vs-sync separates the decode
         * step from the state the prompt left. */
        const char *mode = getenv("QWEN_NLL_MODE");
        const bool via_sync = mode && !strcmp(mode, "sync");
        for (int i = 0; i < N; i++) {
            if (i > 0 && via_sync) {
                pulsar_tokens ext = { ids, P + i, P + i };
                if (pulsar_session_sync(sess, &ext, err, sizeof(err)) != 0) {
                    fprintf(stderr, "qwen-generate: sync at %d: %s\n", P + i, err);
                    break;
                }
            } else if (i > 0 && pulsar_session_eval(sess, ids[P + i - 1], err, sizeof(err)) != 0) {
                fprintf(stderr, "qwen-generate: eval at %d: %s\n", P + i, err);
                break;
            }
            if (pulsar_session_copy_logits(sess, row, W) != W) { fprintf(stderr, "qwen-generate: copy_logits\n"); break; }
            double mx = row[0];
            int am = 0;
            for (int j = 1; j < W; j++) if (row[j] > mx) { mx = row[j]; am = j; }
            double se = 0.0;
            for (int j = 0; j < W; j++) se += exp((double)row[j] - mx);
            const int truth = ids[P + i];
            nll += -((double)row[truth] - mx - log(se));
            top1 += am == truth;
            scored++;
            /* where the mass goes: <|im_end|>, the text-less padded ids past the tokenizer's 248077,
             * and -- when the argmax misses -- what it picked */
            double p_end = exp((double)row[248046] - mx) / se, p_pad = 0.0;
            for (int j = 248077; j < W; j++) p_pad += exp((double)row[j] - mx) / se;
            mass_end += p_end;
            mass_pad += p_pad;
            if (am != truth && i < 24)
                printf("  pos %d: truth %d (p %.4f)  argmax %d (p %.4f)  p<|im_end|> %.4f  p_pad %.2e\n", P + i, truth,
                       exp((double)row[truth] - mx) / se, am, 1.0 / se, p_end, p_pad);
        }
        printf("qwen-generate: mean p(<|im_end|>) %.4f, mean p(padded ids >= 248077) %.3e\n", mass_end / scored,
               mass_pad / scored);
        printf("qwen-generate: NLL prefix %d, %d teacher-forced positions: mean NLL %.5f nats (ppl %.4f), top-1 %.4f "
               "(%d/%d), %.1f s\n", P, scored, nll / scored, exp(nll / scored), (double)top1 / scored, top1, scored,
               now_s() - t0);
        pulsar_session_free(sess);
        free(ids);
        free(row);
        return 0;
    }

    int T = n_tok, generated = 0;
    double t_prefill = 0.0, t_decode = 0.0;
    for (int i = 0; i < n_predict; i++) {
        /* Step 0 prefills the prompt with sync(), which leaves the prompt's next-token row in the
         * logits.  Every later step DECODES the previous argmax with eval(s, token) -- the
         * session's one-token decode path, the one a server stream runs.  (The old eval(sess, 1)
         * decoded id 1 in front of every prediction: the "degenerate repetition" this tool
         * reported before the fix.)  Each step is timed through copy_logits, i.e. end to end. */
        char err[256] = "";
        const double t0 = now_s();
        if (i == 0) {
            pulsar_tokens t = { ids, T, T };
            if (pulsar_session_sync(sess, &t, err, sizeof(err)) != 0) {
                fprintf(stderr, "\nqwen-generate: sync at %d tokens: %s\n", T, err);
                break;
            }
        } else if (pulsar_session_eval(sess, ids[T - 1], err, sizeof(err)) != 0) {
            fprintf(stderr, "\nqwen-generate: eval at %d tokens: %s\n", T, err);
            break;
        }
        /* copy_logits returns the count WRITTEN (0 on error), the opposite of set_logits. */
        if (pulsar_session_copy_logits(sess, row, W) != W) {
            fprintf(stderr, "\nqwen-generate: copy_logits at %d tokens\n", T);
            break;
        }
        int am = 0;
        for (int j = 1; j < W; j++) if (row[j] > row[am]) am = j;
        if (i == 0) t_prefill = now_s() - t0; else t_decode += now_s() - t0;
        /* L251: the top-5 and the MARGIN per step.  A greedy run that locks onto a repeat token is
         * a precision question -- the reference's token is either a hair behind (knife-edge, so a
         * few-percent fidelity deficit explains it) or nowhere near (a real error).  Without this
         * the loop's cause cannot be told apart. */
        {
            int best[5];
            for (int k = 0; k < 5; k++) best[k] = -1;
            for (int j = 0; j < W; j++) {
                for (int k = 0; k < 5; k++) {
                    if (best[k] < 0 || row[j] > row[best[k]]) {
                        for (int m = 4; m > k; m--) best[m] = best[m - 1];
                        best[k] = j;
                        break;
                    }
                }
            }
            printf("\n  step %2d |", i);
            for (int k = 0; k < 5; k++) printf(" %d@%.4f", best[k], (double)row[best[k]]);
            printf(" | margin %.4f", (double)(row[best[0]] - row[best[1]]));
            fflush(stdout);
        }
        /* L251: STOP on the model's own turn terminator.  Without this the test samples PAST
         * <|im_end|> and what looks like degenerate repetition is the harness refusing to let the
         * model stop -- the model's natural continuation of a finished turn IS another terminator. */
        const bool stop = (am == 248046 /* <|im_end|> */ || am == 248044 /* <|endoftext|> */);
        ids[T++] = am;
        generated++;
        printf(" %d", am);
        fflush(stdout);
        if (stop) { printf(" <stop:%d>", am); fflush(stdout); break; }
    }
    printf("\n");
    /* SPEED: prefill is the prompt's one sync(); decode is every eval() after it.  Both include
     * the host argmax and the top-5 scan (negligible beside a step), not the printing. */
    if (generated > 0)
        printf("qwen-generate: SPEED prefill %d tokens in %.3f s = %.1f tok/s | decode %d tokens in %.3f s = %.2f tok/s (%.1f ms/token)\n",
               n_tok, t_prefill, n_tok / t_prefill, generated - 1, t_decode,
               generated > 1 ? (generated - 1) / t_decode : 0.0,
               generated > 1 ? 1e3 * t_decode / (generated - 1) : 0.0);

    if (argc > 4) {
        FILE *f = fopen(argv[4], "wb");
        if (!f) { fprintf(stderr, "qwen-generate: cannot write %s\n", argv[4]); return 1; }
        const size_t n = (size_t)T * sizeof(int32_t);
        if (fwrite(ids, 1, n, f) != n) { fclose(f); fprintf(stderr, "qwen-generate: short write\n"); return 1; }
        fclose(f);
        printf("qwen-generate: %d ids (prompt %d + generated %d) -> %s\n",
               T, n_tok, generated, argv[4]);
    }

    pulsar_session_free(sess);
    free(ids);
    free(row);
    return 0;
}

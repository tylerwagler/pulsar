/* L251: the Qwen bank pool (family_qwen_banks.cpp) on the REAL container -- one conversation per
 * bank, none disturbing another, graded bit for bit against single-bank sessions.
 *
 *   ./tests/qwen_banks_gate <container>
 *
 *   B1  bank 0 prefills A, bank 1 prefills B, bank 0 is restored and decodes a token: its logits
 *       equal a fresh one-bank session's A + that token, byte for byte (B's prefill left A alone)
 *   B2  a batched decode of both banks equals each bank's own single-row decode, byte for byte
 *   B3  an invalidate while bank 1 is live does not touch bank 0: restored, bank 0 decodes on
 *   B4  the router's view: bank_pos / bank_tokens per bank, live and carried
 *   B5  what a recurrent pool cannot do refuses: forks (permanently infeasible), KV spill, residency
 * Not part of the battery: it needs the real container. */
#include "pulsar.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static int n_fail = 0;
#define CHECK(c, ...) do { printf("  %s  ", (c) ? "ok  " : "FAIL"); printf(__VA_ARGS__); printf("\n"); if (!(c)) n_fail++; } while (0)

static std::vector<float> logits_of(pulsar_session *s, int W) {
    std::vector<float> v((size_t)W);
    if (pulsar_session_copy_logits(s, v.data(), W) != W) v.clear();
    return v;
}

static bool same(const std::vector<float> &a, const std::vector<float> &b, double *maxd) {
    *maxd = 0;
    if (a.size() != b.size() || a.empty()) { *maxd = 1e30; return false; }
    for (size_t i = 0; i < a.size(); i++) {
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > *maxd) *maxd = d;
    }
    return memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static pulsar_tokens text_tokens(pulsar_engine *e, const char *text) {
    pulsar_tokens t = {0};
    pulsar_tokenize_text(e, text, &t);
    return t;
}

/* A fresh one-bank session: sync(prompt), then eval each of `extra`; its final logits. */
static std::vector<float> reference(pulsar_engine *e, const pulsar_tokens *prompt, const int *extra, int n_extra,
                                    int W) {
    pulsar_engine_set_bank_pool(1);
    pulsar_session *s = NULL;
    std::vector<float> out;
    char err[256] = "";
    if (pulsar_session_create(&s, e, 4096) != 0) return out;
    if (pulsar_session_sync(s, prompt, err, sizeof(err)) == 0) {
        bool ok = true;
        for (int i = 0; ok && i < n_extra; i++) ok = pulsar_session_eval(s, extra[i], err, sizeof(err)) == 0;
        if (ok) out = logits_of(s, W);
    }
    pulsar_session_free(s);
    return out;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <container>\n", argv[0]); return 2; }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "qwen-banks-gate: %s did not open\n", argv[1]); return 2; }
    const int W = pulsar_engine_logits_width(e);
    pulsar_tokens A = text_tokens(e, "The lighthouse keeper counted the ships every evening, writing each name in a small");
    pulsar_tokens B = text_tokens(e, "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return");
    const int ta = 3409, tb = 308;   /* one decode token each (any ids: the gate compares engines, not text) */
    printf("qwen-banks-gate: A %d tokens, B %d tokens\n", A.len, B.len);

    /* references first, each in its own one-bank session */
    const std::vector<float> refA1 = reference(e, &A, &ta, 1, W);
    const std::vector<float> refB1 = reference(e, &B, &tb, 1, W);
    const int ta2[2] = {ta, 11};
    const std::vector<float> refA2 = reference(e, &A, ta2, 2, W);
    CHECK(!refA1.empty() && !refB1.empty() && !refA2.empty(), "one-bank references ran");

    pulsar_engine_set_bank_pool(2);
    pulsar_session *s = NULL;
    char err[256] = "";
    CHECK(pulsar_session_create(&s, e, 4096) == 0 && s, "a 2-bank session");
    if (!s) return 1;
    CHECK(pulsar_session_bank_count(s) == 2, "bank_count %d", pulsar_session_bank_count(s));

    /* B1 */
    CHECK(pulsar_session_bank_state_restore(s, 0), "restore bank 0 (fresh)");
    CHECK(pulsar_session_sync(s, &A, err, sizeof(err)) == 0, "bank 0 prefills A: %s", err);
    pulsar_session_bank_state_save(s, 0);
    CHECK(pulsar_session_bank_state_restore(s, 1), "restore bank 1 (fresh)");
    CHECK(pulsar_session_sync(s, &B, err, sizeof(err)) == 0, "bank 1 prefills B: %s", err);
    pulsar_session_bank_state_save(s, 1);
    CHECK(pulsar_session_bank_state_restore(s, 0), "restore bank 0");
    CHECK(pulsar_session_eval(s, ta, err, sizeof(err)) == 0, "bank 0 decodes: %s", err);
    double d = 0;
    CHECK(same(logits_of(s, W), refA1, &d), "B1 bank 0 after bank 1's prefill == one-bank A + token (max |diff| %.3g)", d);
    pulsar_session_bank_state_save(s, 0);

    /* B4: the router's view */
    CHECK(pulsar_session_bank_pos(s, 0) == A.len + 1 && pulsar_session_bank_pos(s, 1) == B.len,
          "B4 bank_pos: bank 0 %d (want %d), bank 1 %d (want %d)", pulsar_session_bank_pos(s, 0), A.len + 1,
          pulsar_session_bank_pos(s, 1), B.len);

    /* B2: bank 0 at A + ta (next token 11), bank 1 at B (next token tb) -- one batched step */
    {
        pulsar_multiseq_req rows[2] = { {0u, A.len + 1, 11}, {1u, B.len, tb} };
        std::vector<float> out((size_t)2 * W);
        uint32_t n_out = 0;
        err[0] = '\0';
        const int rc = pulsar_session_decode_mixed(s, rows, 2, out.data(), 2 * W, &n_out, 0, err, sizeof(err));
        CHECK(rc == 0 && n_out == 2, "B2 batched decode of both banks (rc %d, %u rows): %s", rc, n_out, err);
        std::vector<float> r0(out.begin(), out.begin() + W), r1(out.begin() + W, out.end());
        CHECK(same(r0, refA2, &d), "B2 bank 0's batched row == one-bank A + ta + 11 (max |diff| %.3g)", d);
        CHECK(same(r1, refB1, &d), "B2 bank 1's batched row == one-bank B + tb (max |diff| %.3g)", d);
        /* the lane's bookkeeping: note what bank 0 was fed, via its restore */
        CHECK(pulsar_session_bank_state_restore(s, 0), "restore bank 0 after the lane");
        const int fed = 11;
        pulsar_session_note_committed_tokens(s, &fed, 1);
        CHECK(pulsar_session_bank_pos(s, 0) == A.len + 2, "the committed token recorded (bank_pos %d)",
              pulsar_session_bank_pos(s, 0));
        pulsar_session_bank_state_save(s, 0);
        CHECK(pulsar_session_bank_state_restore(s, 1), "restore bank 1 after the lane");
        pulsar_session_note_committed_tokens(s, &tb, 1);
        pulsar_session_bank_state_save(s, 1);
    }

    /* B3: invalidate bank 1's host view; bank 0 must decode on as if nothing happened */
    {
        CHECK(pulsar_session_bank_state_restore(s, 1), "restore bank 1");
        pulsar_session_invalidate(s);
        CHECK(pulsar_session_bank_state_restore(s, 0), "restore bank 0 after bank 1's invalidate");
        pulsar_tokens A3 = {0};
        pulsar_tokens_copy(&A3, &A);
        pulsar_tokens_push(&A3, ta);
        pulsar_tokens_push(&A3, 11);
        pulsar_tokens_push(&A3, 13);
        /* sync of an EXTENSION of bank 0's history continues it (one new token) */
        err[0] = '\0';
        CHECK(pulsar_session_sync(s, &A3, err, sizeof(err)) == 0, "bank 0 continues by sync: %s", err);
        const int t3[3] = {ta, 11, 13};
        const std::vector<float> refA3 = reference(e, &A, t3, 3, W);
        pulsar_engine_set_bank_pool(2);
        CHECK(same(logits_of(s, W), refA3, &d), "B3 bank 0 after bank 1's invalidate == one-bank A + 3 tokens "
              "(max |diff| %.3g)", d);
        pulsar_tokens_free(&A3);
    }

    /* B5 */
    int t1[1] = {1};
    CHECK(pulsar_session_bank_fork(s, 0, 1, t1, 1, 0) != 0, "B5 bank fork refused");
    CHECK(pulsar_session_bank_fork_partial_feasible(s, 0, 4) == PULSAR_FORK_RING_SCROLLED,
          "B5 partial fork permanently infeasible");
    CHECK(pulsar_session_bank_touched_kv_bytes(s, 0) == 0 && !pulsar_session_bank_free_physical(s, 1),
          "B5 no per-bank KV accounting / residency");

    pulsar_session_free(s);
    pulsar_tokens_free(&A);
    pulsar_tokens_free(&B);
    pulsar_engine_close(e);
    printf(n_fail ? "QWEN-BANKS GATE FAIL (%d)\n" : "QWEN-BANKS GATE PASS\n", n_fail);
    return n_fail ? 1 : 0;
}

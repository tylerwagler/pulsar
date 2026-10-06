/* L272 P5: the session contract every family meets, on the REAL model -- what the core's session
 * infrastructure (sync driver, prefill walk, grid checkpoints, bank carry, demand-paged accounting) promises,
 * graded bit for bit against cold single-bank sessions through the public API only.  One engine load runs
 * both parts; a family that lacks a capability fails here by name rather than being skipped silently.
 *
 *   ./tests/session_contract_gate <model> [prefill_chunk]
 *
 * Chunks (L266 step 5, L272 P2; DeepSeek's own planner schedules are chunk_neutrality_gate's):
 *   C1  sync(P[0:k]) then sync(P[0:N]) == sync(P[0:N]) cold: the logits, byte for byte, k on the grid and off it
 *   C2  ... and four decode steps after each
 *   C3  the checkpoint resume: sync(P) captures the state at P's last grid point; a DIVERGENT sync(Q) restores
 *       it -- logits + 4 decode steps == Q cold.  Also after decoding past P.
 *   C4  disk segments: the chain [0, 512) + [512, G) saved, loaded into a FRESH session, then sync(Q) == Q cold
 *   C5  the sync is interruptible: at 256-row chunks the cancel hook stops sync(P) at a chunk boundary
 *       (PULSAR_SESSION_SYNC_INTERRUPTED), the progress hook heard every chunk in order, and the next sync(P)
 *       finishes it -- logits + 4 decode steps == the cold prefill at 256-row chunks
 * Banks (L251, L270, L272 P2's one carry):
 *   B1  bank 0 prefills A, bank 1 prefills B, bank 0 restored decodes a token == a one-bank A + that token
 *   B2  a batched decode of both banks == each bank's own single-row decode
 *   B3  an invalidate while bank 1 is live does not touch bank 0
 *   B4  the router's view: bank_pos / bank_tokens per bank, live and carried
 *   B5  the demand-paged accounting: each bank's touched KV is priced and sums within the session's, a decode
 *       quantum is priced; a per-bank physical eviction either refuses or leaves the bank holding nothing
 * In the battery for every family (`make session-contract-gate-qwen` / `-ds`).  Replaces L266's
 * qwen_chunk_neutrality_gate and qwen_banks_gate. */
#include "pulsar.h"

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

static int n_fail = 0;
#define CHECK(c, ...) do { const bool ok_ = (c); printf("  %s  ", ok_ ? "ok  " : "FAIL"); printf(__VA_ARGS__); printf("\n"); if (!ok_) n_fail++; } while (0)

static std::vector<float> logits_of(pulsar_session *s, int W) {
    std::vector<float> v((size_t)W);
    if (pulsar_session_copy_logits(s, v.data(), W) != W) v.clear();
    return v;
}

static double maxdiff(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size() || a.empty()) return 1e30;
    double m = 0;
    for (size_t i = 0; i < a.size(); i++) m = fmax(m, fabs((double)a[i] - (double)b[i]));
    return m;
}

static const int kDecode[4] = {13, 220, 1576, 3409};   /* any ids: the gate compares engines, not text */

/* sync each prefix in `cuts` in turn (the last is the whole prompt), then decode kDecode; the logits
 * after the final sync and after each decode step */
static std::vector<std::vector<float>> run(pulsar_engine *e, const pulsar_tokens *P, const std::vector<int> &cuts,
                                           int W) {
    std::vector<std::vector<float>> out;
    pulsar_engine_set_bank_pool(1);
    pulsar_session *s = NULL;
    char err[256] = "";
    if (pulsar_session_create(&s, e, 8192) != 0) return out;
    bool ok = true;
    for (int k : cuts) {
        pulsar_tokens pre = *P;
        pre.len = k;
        ok = ok && pulsar_session_sync(s, &pre, err, sizeof(err)) == 0;
    }
    if (ok) out.push_back(logits_of(s, W));
    for (int i = 0; ok && i < 4; i++) {
        ok = pulsar_session_eval(s, kDecode[i], err, sizeof(err)) == 0;
        if (ok) out.push_back(logits_of(s, W));
    }
    if (!ok) { fprintf(stderr, "qwen-chunk-neutrality: %s\n", err); out.clear(); }
    pulsar_session_free(s);
    return out;
}

/* C5's hooks: the progress events heard, and a cancel hook that stops at its `stop_at`-th poll */
struct c5_hooks {
    std::vector<int> chunk_events;
    int polls = 0;
    int stop_at = 0;
};
static void c5_progress(void *ud, const char *event, int current, int) {
    if (!strcmp(event, "prefill_chunk")) ((c5_hooks *)ud)->chunk_events.push_back(current);
}
static bool c5_cancel(void *ud) {
    c5_hooks *h = (c5_hooks *)ud;
    return ++h->polls == h->stop_at;
}

#define TRACE(s, what) printf("    [%s] pos %d | bank_pos 0:%d 1:%d\n", what, pulsar_session_pos(s), pulsar_session_bank_pos(s, 0), pulsar_session_bank_pos(s, 1))
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

/* The same, then sync(`ext`) -- a prompt extension, the path B3's bank takes.  Since L266 a prompt chunk
 * runs the prefill arms at every width (so a resume equals a cold prefill), and a one-token extension by
 * sync is no longer the same arithmetic as eval of that token: the reference takes the bank's own path. */
static std::vector<float> reference_ext(pulsar_engine *e, const pulsar_tokens *prompt, const int *extra, int n_extra,
                                        const pulsar_tokens *ext, int W) {
    pulsar_engine_set_bank_pool(1);
    pulsar_session *s = NULL;
    std::vector<float> out;
    char err[256] = "";
    if (pulsar_session_create(&s, e, 4096) != 0) return out;
    bool ok = pulsar_session_sync(s, prompt, err, sizeof(err)) == 0;
    for (int i = 0; ok && i < n_extra; i++) ok = pulsar_session_eval(s, extra[i], err, sizeof(err)) == 0;
    if (ok && pulsar_session_sync(s, ext, err, sizeof(err)) == 0) out = logits_of(s, W);
    pulsar_session_free(s);
    return out;
}

/* C1-C5 on one engine */
static void part_chunks(pulsar_engine *e, int W, bool explicit_chunk) {

    /* a prompt of mixed prose and code, long enough for several grid points and prefill chunks */
    std::string text;
    for (int i = 0; i < 24; i++) {
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "Entry %d. The lighthouse keeper counted %d ships this evening and wrote each name in a small "
                 "ledger.\ndef ships_%d(n):\n    return [name for name in ledger(n) if name.startswith('%c')]\n",
                 i, 3 + 7 * i, i, 'A' + i % 26);
        text += buf;
    }
    uint32_t grid = 0;
    {
        pulsar_engine_set_bank_pool(1);
        pulsar_session *g = NULL;
        if (pulsar_session_create(&g, e, 8192) == 0) { grid = pulsar_session_resume_grid(g); pulsar_session_free(g); }
    }
    CHECK(grid > 0 && 512 % grid == 0, "the family's resume grid is %u (C4's 512 is on it)", grid);
    if (!grid || 512 % grid) return;
    pulsar_tokens P = {0};
    pulsar_tokenize_text(e, text.c_str(), &P);
    const int N = P.len;
    printf("session-contract: chunks: %d tokens, prefill_chunk %s, grid %u\n", N, explicit_chunk ? "explicit" : "default", grid);

    const std::vector<std::vector<float>> cold = run(e, &P, {N}, W);
    CHECK(cold.size() == 5, "cold prefill + 4 decode steps ran");
    if (cold.size() != 5) { pulsar_tokens_free(&P); return; }
    for (int k : {128, 256, 512, 1024, 100, 333, 777}) {
        if (k >= N) continue;
        const std::vector<std::vector<float>> cut = run(e, &P, {k, N}, W);
        if (cut.size() != 5) { CHECK(false, "cut at %d ran", k); continue; }
        CHECK(memcmp(cut[0].data(), cold[0].data(), (size_t)W * sizeof(float)) == 0,
              "C1 cut at %4d (%s grid): logits == cold, max |diff| %.3e", k, k % (int)grid ? "off" : "on ", maxdiff(cut[0], cold[0]));
        double worst = 0;
        bool same = true;
        for (int i = 1; i < 5; i++) {
            same = same && memcmp(cut[i].data(), cold[i].data(), (size_t)W * sizeof(float)) == 0;
            worst = fmax(worst, maxdiff(cut[i], cold[i]));
        }
        CHECK(same, "C2 cut at %4d: 4 decode steps == cold, max |diff| %.3e", k, worst);
    }
    /* C3: the resume */
    for (int decoded : {0, 3}) {
        const int k = N - 37;                                   /* Q shares P[0:k], then its own tail */
        pulsar_tokens Q = {0};
        for (int i = 0; i < k; i++) pulsar_tokens_push(&Q, P.v[i]);
        for (int i = 0; i < 60; i++) pulsar_tokens_push(&Q, P.v[(i * 7) % 200]);
        const std::vector<std::vector<float>> qcold = run(e, &Q, {Q.len}, W);
        pulsar_engine_set_bank_pool(1);
        pulsar_session *s = NULL;
        char err[256] = "";
        bool ok = pulsar_session_create(&s, e, 8192) == 0 && pulsar_session_sync(s, &P, err, sizeof(err)) == 0;
        for (int i = 0; ok && i < decoded; i++) ok = pulsar_session_eval(s, kDecode[i], err, sizeof(err)) == 0;
        const int G = ok ? pulsar_session_checkpoint_best(s, k) : 0;
        CHECK(ok && G == N / (int)grid * (int)grid && G > 0, "C3 (%d decoded) sync(P) left a checkpoint at %d (want %d): %s",
              decoded, G, N / (int)grid * (int)grid, err);
        std::vector<std::vector<float>> got;
        if (ok && pulsar_session_sync(s, &Q, err, sizeof(err)) == 0) {
            got.push_back(logits_of(s, W));
            for (int i = 0; i < 4 && pulsar_session_eval(s, kDecode[i], err, sizeof(err)) == 0; i++) got.push_back(logits_of(s, W));
        }
        bool same = got.size() == 5 && qcold.size() == 5;
        double worst = 0;
        for (size_t i = 0; same && i < 5; i++) {
            same = memcmp(got[i].data(), qcold[i].data(), (size_t)W * sizeof(float)) == 0;
            worst = fmax(worst, maxdiff(got[i], qcold[i]));
        }
        CHECK(same, "C3 (%d decoded) divergent sync resumed at %d: logits + 4 decode steps == Q cold, max |diff| %.3e %s",
              decoded, G, worst, err);
        pulsar_session_free(s);
        pulsar_tokens_free(&Q);
    }
    /* C4: the chain through files */
    {
        const int Gend = N / (int)grid * (int)grid, k = N - 37;
        pulsar_tokens P600 = P;
        P600.len = 600;
        pulsar_tokens Q = {0};
        for (int i = 0; i < k; i++) pulsar_tokens_push(&Q, P.v[i]);
        for (int i = 0; i < 60; i++) pulsar_tokens_push(&Q, P.v[(i * 7) % 200]);
        const std::vector<std::vector<float>> qcold = run(e, &Q, {Q.len}, W);
        char err[256] = "";
        pulsar_engine_set_bank_pool(1);
        pulsar_session *a = NULL, *b = NULL;
        bool ok = pulsar_session_create(&a, e, 8192) == 0 && pulsar_session_sync(a, &P600, err, sizeof(err)) == 0 &&
                  pulsar_session_sync(a, &P, err, sizeof(err)) == 0;
        CHECK(ok && pulsar_session_checkpoint_best(a, 512) == 512 && pulsar_session_checkpoint_best(a, N) == Gend,
              "C4 checkpoints at 512 and %d after two syncs: %s", Gend, err);
        FILE *f1 = tmpfile(), *f2 = tmpfile();
        const uint64_t b1 = pulsar_session_segment_bytes(a, 0, 512), b2 = pulsar_session_segment_bytes(a, 512, Gend);
        ok = ok && f1 && f2 && pulsar_session_save_segment(a, f1, 0, 512, NULL, err, sizeof(err)) == 0 &&
             pulsar_session_save_segment(a, f2, 512, Gend, NULL, err, sizeof(err)) == 0;
        CHECK(ok, "C4 saved [0, 512) %llu B and [512, %d) %llu B: %s", (unsigned long long)b1, Gend, (unsigned long long)b2, err);
        pulsar_session_free(a);
        int g1 = 0, g2 = 0;
        if (ok) {
            rewind(f1);
            rewind(f2);
            ok = pulsar_session_create(&b, e, 8192) == 0 &&
                 pulsar_session_load_segment(b, f1, b1, false, &g1, NULL, err, sizeof(err)) == 0 &&
                 pulsar_session_load_segment(b, f2, b2, true, &g2, NULL, err, sizeof(err)) == 0;
        }
        CHECK(ok && g1 == 512 && g2 == Gend, "C4 a fresh session loaded the chain to %d: %s", g2, err);
        std::vector<std::vector<float>> got;
        if (ok && pulsar_session_sync(b, &Q, err, sizeof(err)) == 0) {
            got.push_back(logits_of(b, W));
            for (int i = 0; i < 4 && pulsar_session_eval(b, kDecode[i], err, sizeof(err)) == 0; i++) got.push_back(logits_of(b, W));
        }
        bool same = got.size() == 5 && qcold.size() == 5;
        double worst = 0;
        for (size_t i = 0; same && i < 5; i++) {
            same = memcmp(got[i].data(), qcold[i].data(), (size_t)W * sizeof(float)) == 0;
            worst = fmax(worst, maxdiff(got[i], qcold[i]));
        }
        CHECK(same, "C4 sync(Q) on the loaded chain: logits + 4 decode steps == Q cold, max |diff| %.3e %s", worst, err);
        if (b) pulsar_session_free(b);
        if (f1) fclose(f1);
        if (f2) fclose(f2);
        pulsar_tokens_free(&Q);
    }
    /* C5: an interrupted sync resumes exactly (L272 P2) */
    if (explicit_chunk) {
        printf("  skip  C5 (an explicit prefill_chunk pins the session's chunk; C5 sets its own)\n");
    } else if (N <= 512) {
        CHECK(false, "C5 needs a prompt of more than two 256-row chunks (have %d)", N);
    } else {
        setenv("PULSAR_CUDA_PREFILL_CHUNK", "256", 1);
        /* the reference is the cold prefill at the SAME chunk: a family need not be chunk-SIZE neutral
         * (DeepSeek's chunk bytes depend on the row count, L183), only cut neutral */
        const std::vector<std::vector<float>> cold256 = run(e, &P, {N}, W);
        pulsar_engine_set_bank_pool(1);
        pulsar_session *s = NULL;
        char err[256] = "";
        const bool made = pulsar_session_create(&s, e, 8192) == 0;
        unsetenv("PULSAR_CUDA_PREFILL_CHUNK");
        CHECK(made, "C5 session at 256-row chunks: %s", err);
        if (made) {
            CHECK(pulsar_session_prefill_quantum_min_suffix(s) == 1, "C5 the server may interrupt this family's prefill");
            c5_hooks h;
            h.stop_at = 2;   /* poll 1 is before the first chunk, poll 2 after it: stop at 256 */
            pulsar_session_set_progress(s, c5_progress, &h);
            pulsar_session_set_cancel(s, c5_cancel, &h);
            const int rc1 = pulsar_session_sync(s, &P, err, sizeof(err));
            CHECK(rc1 == PULSAR_SESSION_SYNC_INTERRUPTED && pulsar_session_pos(s) == 256,
                  "C5 the cancel hook stopped sync(P) at a chunk boundary: rc %d, pos %d (want %d, 256) %s", rc1,
                  pulsar_session_pos(s), PULSAR_SESSION_SYNC_INTERRUPTED, err);
            pulsar_session_set_cancel(s, NULL, NULL);
            const int rc2 = pulsar_session_sync(s, &P, err, sizeof(err));
            bool monotone = !h.chunk_events.empty() && h.chunk_events.back() == N;
            for (size_t i = 1; i < h.chunk_events.size(); i++) monotone &= h.chunk_events[i] >= h.chunk_events[i - 1];
            const bool heard_256 = std::find(h.chunk_events.begin(), h.chunk_events.end(), 256) != h.chunk_events.end();
            CHECK(rc2 == 0 && pulsar_session_pos(s) == N, "C5 the next sync(P) finished the prompt: rc %d, pos %d %s", rc2,
                  pulsar_session_pos(s), err);
            CHECK(monotone && heard_256, "C5 progress heard every chunk in order (%zu events, last %d)",
                  h.chunk_events.size(), h.chunk_events.empty() ? -1 : h.chunk_events.back());
            std::vector<std::vector<float>> got;
            if (rc2 == 0) {
                got.push_back(logits_of(s, W));
                for (int i = 0; i < 4 && pulsar_session_eval(s, kDecode[i], err, sizeof(err)) == 0; i++)
                    got.push_back(logits_of(s, W));
            }
            bool same = got.size() == 5;
            double worst = 0;
            for (size_t i = 0; same && i < 5; i++) {
                same = cold256.size() == 5 && memcmp(got[i].data(), cold256[i].data(), (size_t)W * sizeof(float)) == 0;
                worst = fmax(worst, cold256.size() == 5 ? maxdiff(got[i], cold256[i]) : 1e30);
            }
            CHECK(same, "C5 interrupted + resumed at 256-row chunks: logits + 4 decode steps == cold at 256, max |diff| %.3e %s",
                  worst, err);
            pulsar_session_free(s);
        }
    }
    pulsar_tokens_free(&P);
}

/* B1-B5 on the same engine */
static void part_banks(pulsar_engine *e, int W) {
    pulsar_tokens A = text_tokens(e, "The lighthouse keeper counted the ships every evening, writing each name in a small");
    pulsar_tokens B = text_tokens(e, "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return");
    const int ta = 3409, tb = 308;   /* one decode token each (any ids: the gate compares engines, not text) */
    printf("session-contract: banks: A %d tokens, B %d tokens\n", A.len, B.len);

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
    if (!s) { pulsar_tokens_free(&A); pulsar_tokens_free(&B); return; }
    CHECK(pulsar_session_bank_count(s) == 2, "bank_count %d", pulsar_session_bank_count(s));

    /* B1 */
    CHECK(pulsar_session_bank_state_restore(s, 0), "restore bank 0 (fresh)");
    CHECK(pulsar_session_sync(s, &A, err, sizeof(err)) == 0, "bank 0 prefills A: %s", err);
    TRACE(s, "after sync A");
    pulsar_session_bank_state_save(s, 0);
    TRACE(s, "after save 0");
    CHECK(pulsar_session_bank_state_restore(s, 1), "restore bank 1 (fresh)");
    TRACE(s, "after restore 1");
    CHECK(pulsar_session_sync(s, &B, err, sizeof(err)) == 0, "bank 1 prefills B: %s", err);
    TRACE(s, "after sync B");
    pulsar_session_bank_state_save(s, 1);
    CHECK(pulsar_session_bank_state_restore(s, 0), "restore bank 0");
    TRACE(s, "after restore 0");
    CHECK(pulsar_session_eval(s, ta, err, sizeof(err)) == 0, "bank 0 decodes: %s", err);
    TRACE(s, "after eval ta");
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
        const int t2[2] = {ta, 11};
        const std::vector<float> refA3 = reference_ext(e, &A, t2, 2, &A3, W);
        pulsar_engine_set_bank_pool(2);
        CHECK(same(logits_of(s, W), refA3, &d), "B3 bank 0 after bank 1's invalidate == one-bank A + 2 decoded + "
              "sync to A3 (max |diff| %.3g)", d);
        pulsar_tokens_free(&A3);
    }

    /* B5 (L270; the fork legs went with the bank fork, L264/L265) */
    {
        const uint64_t t0 = pulsar_session_bank_touched_kv_bytes(s, 0), t1 = pulsar_session_bank_touched_kv_bytes(s, 1);
        const uint64_t all = pulsar_session_touched_kv_bytes(s), q = pulsar_session_quantum_growth_bytes_per_bank(s, 8);
        CHECK(t0 > 0 && t1 > 0 && all >= t0 + t1 && q > 0,
              "B5 touched KV per bank %llu / %llu B, session %llu B, an 8-token quantum %llu B",
              (unsigned long long)t0, (unsigned long long)t1, (unsigned long long)all, (unsigned long long)q);
        /* the eviction contract: a family whose banks share tensors refuses (Qwen); one that can release a
         * bank's pages does, and that bank then holds none (DeepSeek) */
        const bool freed = pulsar_session_bank_free_physical(s, 1);
        const uint64_t t1_after = pulsar_session_bank_touched_kv_bytes(s, 1);
        CHECK(!freed || t1_after == 0, "B5 a per-bank physical eviction %s (bank 1 touched after: %llu B)",
              freed ? "released the bank" : "refused (shared tensors)", (unsigned long long)t1_after);
    }

    pulsar_session_free(s);
    pulsar_tokens_free(&A);
    pulsar_tokens_free(&B);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model> [prefill_chunk]\n", argv[0]); return 2; }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    if (argc > 2) opt.prefill_chunk = (uint32_t)atoi(argv[2]);
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "session-contract: %s did not open\n", argv[1]); return 2; }
    const int W = pulsar_engine_logits_width(e);
    part_chunks(e, W, argc > 2);
    part_banks(e, W);
    pulsar_engine_close(e);
    printf(n_fail ? "SESSION-CONTRACT GATE FAIL (%d)\n" : "SESSION-CONTRACT GATE PASS\n", n_fail);
    return n_fail != 0;
}

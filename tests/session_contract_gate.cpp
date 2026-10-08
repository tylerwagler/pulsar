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
 *       quantum is priced; a per-bank physical eviction of an idle bank leaves it holding nothing
 *   B6  a verify step over two, three and four banks' runs (the speculation lane's, up to ten rows) gives each
 *       bank its run's rows verified alone, byte for byte (L272 P1 S4, L284)
 * The fused step (L284 #2; pulsar_session_decode_fused: a verify's rows and a prompt chunk in one forward), bank 0
 * verifying 3 rows in front of bank 1's chunk of 62 and of 70 rows (either side of the attention's 64-row fold):
 *   F1  the verify rows' logits == the same rows verified alone, byte for byte (B6's oracle)
 *   F2  the chunk's headed last row == a sync of the same chunk on that bank, byte for byte (C1's guarantee).
 *       DeepSeek: skipped -- a fused prompt row there is graded in a tolerance band (fused_step_gate (2))
 *   F3  four decode steps on the chunk's bank after the fused step == the classic run (sync, sync, 4 x eval).
 *       DeepSeek: skipped with F2 (its prompt KV is F2's band, not the classic bytes)
 *   F4  the verify rows do not move the prompt rows: the same chunk as a fused step with no verify rows gives the
 *       same headed row, byte for byte.  DeepSeek: skipped -- its fused step carries decode rows by contract;
 *       fused_step_gate (5) grades the same fact against a plain mixed step
 * Several prompt runs (L284 #2 increments 4 and 5, Qwen; DeepSeek skips them with F2): F5-F8 at part_fused_runs --
 * a 130-row continuation and a dirty bank's 2048 rows from 0 behind 3 verify rows in both bank orders, the decode
 * after, the runs without the verify, and the core's record of each chunk (note_prefilled: history, next-token
 * logits, the grid checkpoint a divergent sync then resumes from == cold).
 * Speculation (L284):
 *   S1  a request boundary re-arms it: a bank whose yield quench latched (forced by its field -- the one reach past
 *       the public API) carries the latch across a bank switch, and the next request's sync on that bank speculates
 *       again; an invalidate drops the lookahead and re-arms too
 * In the battery for every family as a runner gate (L278: tests/gates_runner.cpp hosts each family's model); the
 * standalone `make session-contract-gate-qwen` / `-ds` remain for iterating.  Replaces L266's
 * qwen_chunk_neutrality_gate and qwen_banks_gate. */
#include "pulsar.h"
#include "gate_entry.h"
#include "gate_util.h"
#include "pulsar_engine_internal.h"   /* S1 forces the quench latch by its field */

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

static int n_fail = 0;
#define CHECK(c, ...) GATE_CHECK(n_fail, c, __VA_ARGS__)

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

/* C5's hooks: the progress events heard, and a cancel hook that stops at the first poll after a chunk has
 * landed -- family-neutral (a family's sync may poll more than its walk does, e.g. once before any work) */
struct c5_hooks {
    std::vector<int> chunk_events;
};
static void c5_progress(void *ud, const char *event, int current, int) {
    if (!strcmp(event, "prefill_chunk")) ((c5_hooks *)ud)->chunk_events.push_back(current);
}
static bool c5_cancel(void *ud) {
    const c5_hooks *h = (const c5_hooks *)ud;
    for (int e : h->chunk_events) if (e > 0) return true;   /* a chunk landed: stop at this boundary */
    return false;
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
static std::vector<float> verify_alone(pulsar_engine *e, const pulsar_tokens *P, const int *run, int nr, int W);

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
        /* the eviction contract (L284 #3: every family's): an idle bank's pages are released, and it then holds
         * none (bank_residency_gate grades the way back) */
        const bool freed = pulsar_session_bank_free_physical(s, 1);
        const uint64_t t1_after = pulsar_session_bank_touched_kv_bytes(s, 1);
        CHECK(freed && t1_after == 0 && pulsar_session_bank_is_evicted(s, 1),
              "B5 a per-bank physical eviction released idle bank 1 (touched after: %llu B)",
              (unsigned long long)t1_after);
    }

    pulsar_session_free(s);

    /* B6 (L272 P1 S4; L284 to four banks): a verify step over several banks' runs -- the speculation lane's step,
     * every row headed -- gives each bank the rows its run gives verified alone, byte for byte.  Two, three and
     * four banks: bank 0's run [ta, 11, 12], bank 1's [tb, 13], bank 2's (A again) [ta, 14], bank 3's (B again)
     * [tb, 17, 18] -- ten rows at four banks, the widest step whose rows keep their one-token bytes in both
     * families (DeepSeek's decode GEMMs leave the GEMV for cuBLASLt / the E4M3 MMA from 11 rows; Qwen's at 17).
     * Wider verify steps are graded, not byte-gated (L284). */
    if (!pulsar_engine_has_spec_rounds(e)) {
        printf("  skip  B6 (no drafter on this model: a verify step needs one)\n");
    } else {
        struct b6_run { const pulsar_tokens *P; int run[3]; int nr; };
        const b6_run runs[4] = {{&A, {ta, 11, 12}, 3}, {&B, {tb, 13}, 2}, {&A, {ta, 14}, 2}, {&B, {tb, 17, 18}, 3}};
        std::vector<float> solo[4];
        for (int b = 0; b < 4; b++) solo[b] = verify_alone(e, runs[b].P, runs[b].run, runs[b].nr, W);
        for (int nb = 2; nb <= 4; nb++) {
            pulsar_engine_set_bank_pool((uint32_t)nb);
            pulsar_session *x = NULL;
            std::vector<pulsar_multiseq_req> q;
            bool ran = pulsar_session_create(&x, e, 4096) == 0;
            for (int b = 0; ran && b < nb; b++) {
                ran = pulsar_session_bank_state_restore(x, (uint32_t)b) &&
                      pulsar_session_sync(x, runs[b].P, err, sizeof(err)) == 0;
                if (ran) pulsar_session_bank_state_save(x, (uint32_t)b);
                for (int i = 0; i < runs[b].nr; i++) q.push_back({(uint32_t)b, runs[b].P->len + i, runs[b].run[i]});
            }
            const uint32_t nr = (uint32_t)q.size();
            CHECK(nr <= pulsar_engine_fused_heads_max(e), "B6 %d banks' runs (%u rows) fit the verify width %u", nb,
                  nr, pulsar_engine_fused_heads_max(e));
            std::vector<float> all((size_t)nr * W);
            uint32_t got = 0;
            ran = ran && pulsar_session_decode_mixed(x, q.data(), nr, all.data(), (int)nr * W, &got,
                                                     PULSAR_MSEQ_HEAD_ALL_ROWS, err, sizeof(err)) == 0 && got == nr;
            std::string verdict;
            bool same_all = ran;
            size_t at = 0;
            for (int b = 0; b < nb; b++) {
                const size_t n = (size_t)runs[b].nr * W;
                const bool same_b = ran && solo[b].size() == n && memcmp(all.data() + at, solo[b].data(), n * 4) == 0;
                same_all = same_all && same_b;
                verdict += " bank " + std::to_string(b) + (same_b ? " identical" : " DIFFERS");
                at += n;
            }
            CHECK(same_all, "B6 a verify of %d banks' runs (%u rows) == each run verified alone:%s %s", nb, nr,
                  verdict.c_str(), ran ? "" : err);
            if (x) pulsar_session_free(x);
        }
    }
    pulsar_tokens_free(&A);
    pulsar_tokens_free(&B);
}

/* F1-F4 (L284 #2): the fused step on the same engine */
static std::vector<float> verify_alone(pulsar_engine *e, const pulsar_tokens *P, const int *run, int nr, int W) {
    std::vector<float> rows;
    pulsar_engine_set_bank_pool(1);
    pulsar_session *x = NULL;
    char err[256] = "";
    if (pulsar_session_create(&x, e, 4096) != 0) return rows;
    if (pulsar_session_sync(x, P, err, sizeof(err)) == 0) {
        std::vector<pulsar_multiseq_req> q((size_t)nr);
        for (int i = 0; i < nr; i++) q[(size_t)i] = {0u, P->len + i, run[i]};
        rows.resize((size_t)nr * W);
        uint32_t got = 0;
        if (pulsar_session_decode_mixed(x, q.data(), (uint32_t)nr, rows.data(), nr * W, &got, PULSAR_MSEQ_HEAD_ALL_ROWS,
                                        err, sizeof(err)) != 0 || got != (uint32_t)nr) {
            fprintf(stderr, "session-contract: verify alone: %s\n", err);
            rows.clear();
        }
    }
    pulsar_session_free(x);
    return rows;
}

/* a 2-bank session holding A on bank 0 (when `a`) and `pre` on bank 1, then ONE fused step: bank 0's verify rows
 * `run` (nr of them, 0 = none) and bank 1's chunk chunk[0, m) at pre->len, its last row headed.  The step's
 * logits rows; the session stays open in *keep (NULL = freed). */
static std::vector<float> fused_step(pulsar_engine *e, const pulsar_tokens *A, const int *run, int nr,
                                     const pulsar_tokens *pre, const int *chunk, int m, int W, pulsar_session **keep) {
    std::vector<float> out;
    pulsar_engine_set_bank_pool(2);
    pulsar_session *x = NULL;
    char err[256] = "";
    /* the prompt's bank first, so a verifying bank is the installed one (the server's shape) */
    bool ok = pulsar_session_create(&x, e, 4096) == 0;
    ok = ok && pulsar_session_bank_state_restore(x, 1) && pulsar_session_sync(x, pre, err, sizeof(err)) == 0;
    if (ok) pulsar_session_bank_state_save(x, 1);
    if (ok && nr > 0) {
        ok = pulsar_session_bank_state_restore(x, 0) && pulsar_session_sync(x, A, err, sizeof(err)) == 0;
        if (ok) pulsar_session_bank_state_save(x, 0);
    }
    if (ok) {
        std::vector<pulsar_multiseq_req> q;
        for (int i = 0; i < nr; i++) q.push_back({0u, A->len + i, run[i]});
        for (int j = 0; j < m; j++) q.push_back({1u, pre->len + j, chunk[j]});
        pulsar_fused_shape sh;
        memset(&sh, 0, sizeof sh);
        sh.n_dec = (uint32_t)nr;
        sh.n_pf = 1;
        sh.head_last[0] = 1;
        out.resize((size_t)(nr + 1) * W);
        uint32_t got = 0;
        ok = pulsar_session_decode_fused(x, q.data(), (uint32_t)q.size(), &sh, out.data(), (nr + 1) * W, &got, err,
                                         sizeof(err)) == 0;
        if (ok && got != (uint32_t)nr + 1u) snprintf(err, sizeof(err), "headed %u rows, want %d", got, nr + 1);
        ok = ok && got == (uint32_t)nr + 1u;
    }
    if (!ok) {
        fprintf(stderr, "session-contract: fused step: %s\n", err);
        out.clear();
    }
    if (keep && ok) *keep = x;
    else if (x) pulsar_session_free(x);
    return out;
}

static void part_fused(pulsar_engine *e, int W) {
    if (!pulsar_engine_has_fused_step(e)) {
        printf("  skip  F1-F4 (%s has no fused step)\n", pulsar_engine_family_name(e));
        return;
    }
    if (!pulsar_engine_has_spec_rounds(e)) {
        printf("  skip  F1-F4 (no drafter on this model: the fused step's verify rows need one)\n");
        return;
    }
    const bool classic_bytes = pulsar_engine_family(e) != PULSAR_FAMILY_ID_DEEPSEEK4;
    if (!classic_bytes)
        printf("  skip  F2-F4 (DeepSeek: a fused prompt row is graded in a tolerance band by fused_step_gate (2), and "
               "its fused step carries decode rows by contract -- (5) grades F4's fact)\n");
    pulsar_tokens A = text_tokens(e, "The lighthouse keeper counted the ships every evening, writing each name in a small");
    std::string text;
    for (int i = 0; i < 12; i++) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Ledger %d: the tide came in at %d past the hour and the keeper lit lamp %c.\n", i,
                 7 + 3 * i, 'A' + i);
        text += buf;
    }
    pulsar_tokens L = text_tokens(e, text.c_str());
    const int runA[3] = {3409, 11, 12};
    const std::vector<float> soloA = verify_alone(e, &A, runA, 3, W);
    CHECK(soloA.size() == (size_t)3 * W, "F the verify alone ran");
    for (int m : {62, 70}) {
        const int pre_len = 40;
        if (L.len < pre_len + m) { CHECK(false, "F the ledger text has %d tokens (want %d)", L.len, pre_len + m); break; }
        pulsar_tokens pre = L, whole = L;
        pre.len = pre_len;
        whole.len = pre_len + m;
        const int *chunk = L.v + pre_len;
        pulsar_session *x = NULL;
        const std::vector<float> got = fused_step(e, &A, runA, 3, &pre, chunk, m, W, &x);
        CHECK(got.size() == (size_t)4 * W, "F fused step: 3 verify rows + a %d-row chunk, 4 rows headed", m);
        if (got.size() != (size_t)4 * W) { if (x) pulsar_session_free(x); continue; }
        CHECK(soloA.size() == (size_t)3 * W && memcmp(got.data(), soloA.data(), soloA.size() * 4) == 0,
              "F1 (%d-row chunk) the verify rows == verified alone, byte for byte", m);
        if (!classic_bytes) { pulsar_session_free(x); continue; }
        /* the classic run: sync(pre), sync(whole), 4 x eval */
        std::vector<std::vector<float>> cl;
        {
            pulsar_engine_set_bank_pool(1);
            pulsar_session *c = NULL;
            char err[256] = "";
            bool ok = pulsar_session_create(&c, e, 4096) == 0 && pulsar_session_sync(c, &pre, err, sizeof(err)) == 0 &&
                      pulsar_session_sync(c, &whole, err, sizeof(err)) == 0;
            if (ok) cl.push_back(logits_of(c, W));
            for (int i = 0; ok && i < 4; i++) {
                ok = pulsar_session_eval(c, kDecode[i], err, sizeof(err)) == 0;
                if (ok) cl.push_back(logits_of(c, W));
            }
            if (!ok) fprintf(stderr, "session-contract: F classic run: %s\n", err);
            if (c) pulsar_session_free(c);
        }
        const std::vector<float> row(got.begin() + (size_t)3 * W, got.end());
        double d = 0;
        CHECK(cl.size() == 5 && same(row, cl[0], &d), "F2 (%d-row chunk) the fused chunk's headed row == its sync, "
              "byte for byte (max |diff| %.3g)", m, d);
        /* F3: bank 1 decodes on in the fused session, one row a step */
        bool steps = cl.size() == 5;
        double worst = 0;
        for (int i = 0; steps && i < 4; i++) {
            const pulsar_multiseq_req r1 = {1u, whole.len + i, kDecode[i]};
            std::vector<float> lg((size_t)W);
            uint32_t n_out = 0;
            char err[256] = "";
            steps = pulsar_session_decode_mixed(x, &r1, 1, lg.data(), W, &n_out, 0, err, sizeof(err)) == 0 && n_out == 1;
            if (!steps) { fprintf(stderr, "session-contract: F3 decode: %s\n", err); break; }
            double di = 0;
            steps = same(lg, cl[(size_t)i + 1], &di);
            worst = fmax(worst, di);
        }
        CHECK(steps, "F3 (%d-row chunk) 4 decode steps after the fused step == the classic run (max |diff| %.3g)", m,
              worst);
        pulsar_session_free(x);
        /* F4: the same chunk with no verify rows in front */
        const std::vector<float> bare = fused_step(e, &A, runA, 0, &pre, chunk, m, W, NULL);
        CHECK(bare.size() == (size_t)W && same(row, bare, &d), "F4 (%d-row chunk) the chunk's row without the verify "
              "rows == with them, byte for byte (max |diff| %.3g)", m, d);
    }
    pulsar_tokens_free(&A);
    pulsar_tokens_free(&L);
}

/* A one-bank 4096-token session (the fused sessions' size): sync each prefix in `cuts`, then eval kDecode; the
 * logits after the last sync and after each eval. */
static std::vector<std::vector<float>> classic(pulsar_engine *e, const pulsar_tokens *P, const std::vector<int> &cuts,
                                               int W) {
    std::vector<std::vector<float>> out;
    pulsar_engine_set_bank_pool(1);
    pulsar_session *c = NULL;
    char err[256] = "";
    bool ok = pulsar_session_create(&c, e, 4096) == 0;
    for (int k : cuts) {
        pulsar_tokens pre = *P;
        pre.len = k;
        ok = ok && pulsar_session_sync(c, &pre, err, sizeof(err)) == 0;
    }
    if (ok) out.push_back(logits_of(c, W));
    for (int i = 0; ok && i < 4; i++) {
        ok = pulsar_session_eval(c, kDecode[i], err, sizeof(err)) == 0;
        if (ok) out.push_back(logits_of(c, W));
    }
    if (!ok) { fprintf(stderr, "session-contract: classic run: %s\n", err); out.clear(); }
    if (c) pulsar_session_free(c);
    return out;
}

static std::string ledger_text(const char *what, int lines) {
    std::string text;
    for (int i = 0; i < lines; i++) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s %d: the tide came in at %d past the hour and the keeper lit lamp %c.\n", what, i,
                 (7 + 3 * i) % 60, 'A' + i % 26);
        text += buf;
    }
    return text;
}

/* F5-F8 (L284 #2 increments 4 and 5): several prompt runs in one fused step, on a 3-bank session.  Bank 0 verifies
 * 3 rows; bank 1 continues its 40-token prefix by a 130-row chunk (past the verify width and the attention's 64-row
 * fold); bank 2 -- DIRTY: it held another conversation, decoded on it, and was invalidated, as a reused server bank
 * is -- prefills a 2048-row chunk from 0 (a grid point).  In either bank order:
 *   F5  the verify rows == verified alone, and each run's headed row == its classic sync, byte for byte
 *   F6  four decode steps on each prompt bank after the step == its classic run
 *   F7  the same two runs with no verify rows in front give the same headed rows
 *   F8  the record (pulsar_session_note_prefilled, the core's for every family): each chunk joins its bank's
 *       history with its headed row as the next-token logits; bank 2's chunk, which ends on the grid, leaves a
 *       checkpoint there; and a DIVERGENT sync of bank 2 resumes from it == the divergent prompt cold */
static void part_fused_runs(pulsar_engine *e, int W) {
    if (!pulsar_engine_has_fused_step(e) || !pulsar_engine_has_spec_rounds(e) ||
        pulsar_engine_family(e) == PULSAR_FAMILY_ID_DEEPSEEK4) {
        printf("  skip  F5-F8 (%s: no fused step with classic-byte prompt rows -- see F2)\n",
               pulsar_engine_family_name(e));
        return;
    }
    pulsar_tokens A = text_tokens(e, "The lighthouse keeper counted the ships every evening, writing each name in a small");
    pulsar_tokens L = text_tokens(e, ledger_text("Ledger", 12).c_str());
    pulsar_tokens L2 = text_tokens(e, ledger_text("Logbook", 160).c_str());
    pulsar_tokens D = text_tokens(e, ledger_text("Manifest", 40).c_str());
    const int pre_len = 40, m1 = 130, m2 = 2048;
    if (L.len < pre_len + m1 || L2.len < m2 + 64 || D.len < 300) {
        CHECK(false, "F5 texts too short (%d, %d, %d tokens)", L.len, L2.len, D.len);
        return;
    }
    printf("session-contract: fused runs: bank 1 %d + %d rows, bank 2 dirty (%d tokens) then %d rows from 0\n", pre_len,
           m1, D.len, m2);
    const int runA[3] = {3409, 11, 12};
    const std::vector<float> soloA = verify_alone(e, &A, runA, 3, W);
    pulsar_tokens pre = L, c2 = L2;
    pre.len = pre_len;
    c2.len = m2;
    const std::vector<std::vector<float>> cl1 = classic(e, &L, {pre_len, pre_len + m1}, W);
    const std::vector<std::vector<float>> cl2 = classic(e, &L2, {m2}, W);
    CHECK(soloA.size() == (size_t)3 * W && cl1.size() == 5 && cl2.size() == 5, "F5 the references ran");
    if (cl1.size() != 5 || cl2.size() != 5 || soloA.size() != (size_t)3 * W) return;

    /* the 3-bank session, before the step: bank 0 holds A, bank 1 the prefix, bank 2 a dead conversation */
    auto setup = [&](bool verify) -> pulsar_session * {
        pulsar_engine_set_bank_pool(3);
        pulsar_session *x = NULL;
        char err[256] = "";
        bool ok = pulsar_session_create(&x, e, 4096) == 0;
        ok = ok && pulsar_session_bank_state_restore(x, 2) && pulsar_session_sync(x, &D, err, sizeof(err)) == 0 &&
             pulsar_session_eval(x, kDecode[0], err, sizeof(err)) == 0 && pulsar_session_eval(x, kDecode[1], err, sizeof(err)) == 0;
        if (ok) { pulsar_session_invalidate(x); pulsar_session_bank_state_save(x, 2); }
        ok = ok && pulsar_session_bank_state_restore(x, 1) && pulsar_session_sync(x, &pre, err, sizeof(err)) == 0;
        if (ok) pulsar_session_bank_state_save(x, 1);
        if (ok && verify) {
            ok = pulsar_session_bank_state_restore(x, 0) && pulsar_session_sync(x, &A, err, sizeof(err)) == 0;
            if (ok) pulsar_session_bank_state_save(x, 0);
        }
        if (!ok) { fprintf(stderr, "session-contract: F5 setup: %s\n", err); if (x) pulsar_session_free(x); x = NULL; }
        return x;
    };
    /* ONE fused step: the verify (when nr), then the runs in `order` (1 = bank 1's chunk, 2 = bank 2's), each headed.
     * The step's logits rows, verify first, then the runs in order. */
    auto step = [&](pulsar_session *x, int nr, const int order[2]) {
        std::vector<pulsar_multiseq_req> q;
        for (int i = 0; i < nr; i++) q.push_back({0u, A.len + i, runA[i]});
        for (int r = 0; r < 2; r++) {
            if (order[r] == 1) for (int j = 0; j < m1; j++) q.push_back({1u, pre_len + j, L.v[pre_len + j]});
            else for (int j = 0; j < m2; j++) q.push_back({2u, j, L2.v[j]});
        }
        pulsar_fused_shape sh;
        memset(&sh, 0, sizeof sh);
        sh.n_dec = (uint32_t)nr;
        sh.n_pf = 2;
        sh.head_last[0] = sh.head_last[1] = 1;
        std::vector<float> out((size_t)(nr + 2) * W);
        uint32_t got = 0;
        char err[256] = "";
        const int rc = pulsar_session_decode_fused(x, q.data(), (uint32_t)q.size(), &sh, out.data(), (nr + 2) * W, &got,
                                                   err, sizeof(err));
        if (rc != 0 || got != (uint32_t)nr + 2u) {
            fprintf(stderr, "session-contract: F5 fused step: rc %d, %u rows headed: %s\n", rc, got, err);
            out.clear();
        }
        return out;
    };
    auto row = [&](const std::vector<float> &v, int i) {
        return std::vector<float>(v.begin() + (size_t)i * W, v.begin() + (size_t)(i + 1) * W);
    };
    const int orders[2][2] = {{1, 2}, {2, 1}};
    std::vector<float> head1[2], head2[2];
    for (int o = 0; o < 2; o++) {
        const int *ord = orders[o];
        pulsar_session *x = setup(true);
        if (!x) { CHECK(false, "F5 (order %d,%d) the 3-bank session", ord[0], ord[1]); continue; }
        const std::vector<float> got = step(x, 3, ord);
        CHECK(!got.empty(), "F5 (order %d,%d) one fused step: 3 verify rows + %d + %d prompt rows, 5 rows headed",
              ord[0], ord[1], ord[0] == 1 ? m1 : m2, ord[0] == 1 ? m2 : m1);
        if (got.empty()) { pulsar_session_free(x); continue; }
        const int i1 = ord[0] == 1 ? 3 : 4, i2 = ord[0] == 1 ? 4 : 3;
        head1[o] = row(got, i1);
        head2[o] = row(got, i2);
        double d1 = 0, d2 = 0;
        CHECK(memcmp(got.data(), soloA.data(), soloA.size() * 4) == 0,
              "F5 (order %d,%d) the verify rows == verified alone, byte for byte", ord[0], ord[1]);
        CHECK(same(head1[o], cl1[0], &d1), "F5 (order %d,%d) bank 1's %d-row continuation == its sync, byte for byte "
              "(max |diff| %.3g)", ord[0], ord[1], m1, d1);
        CHECK(same(head2[o], cl2[0], &d2), "F5 (order %d,%d) dirty bank 2's %d rows from 0 == a cold sync, byte for "
              "byte (max |diff| %.3g)", ord[0], ord[1], m2, d2);
        if (o == 0) {
            /* F6: each prompt bank decodes on, one row a step */
            for (int b = 1; b <= 2; b++) {
                const std::vector<std::vector<float>> &cl = b == 1 ? cl1 : cl2;
                const int T = b == 1 ? pre_len + m1 : m2;
                bool steps = true;
                double worst = 0;
                for (int i = 0; steps && i < 4; i++) {
                    const pulsar_multiseq_req r = {(uint32_t)b, T + i, kDecode[i]};
                    std::vector<float> lg((size_t)W);
                    uint32_t n_out = 0;
                    char err[256] = "";
                    steps = pulsar_session_decode_mixed(x, &r, 1, lg.data(), W, &n_out, 0, err, sizeof(err)) == 0 &&
                            n_out == 1;
                    if (!steps) { fprintf(stderr, "session-contract: F6 decode: %s\n", err); break; }
                    double di = 0;
                    steps = same(lg, cl[(size_t)i + 1], &di);
                    worst = fmax(worst, di);
                }
                CHECK(steps, "F6 bank %d: 4 decode steps after the fused step == its classic run (max |diff| %.3g)", b,
                      worst);
            }
        } else {
            /* F8: the record, then a divergent sync of bank 2 from the checkpoint the record left */
            char err[256] = "";
            bool ok = pulsar_session_bank_state_restore(x, 2) &&
                      pulsar_session_note_prefilled(x, L2.v, m2, ord[0] == 2 ? 0 : 1) == 0;
            double d = 0;
            CHECK(ok && pulsar_session_pos(x) == m2 && same(logits_of(x, W), cl2[0], &d),
                  "F8 bank 2's record: %d tokens, its headed row as the next-token logits (max |diff| %.3g)",
                  pulsar_session_pos(x), d);
            CHECK(ok && pulsar_session_checkpoint_best(x, m2) == m2,
                  "F8 bank 2's chunk ended on the grid: a checkpoint at %d (best %d)", m2,
                  pulsar_session_checkpoint_best(x, m2));
            /* the conversation decodes on past the record (the classic eval) -- so the divergent sync below cannot
             * continue the history and must resume from the checkpoint */
            for (int i = 0; ok && i < 2; i++) ok = pulsar_session_eval(x, kDecode[i], err, sizeof(err)) == 0;
            CHECK(ok && same(logits_of(x, W), cl2[2], &d), "F8 bank 2 evaluates on from the record == its classic run "
                  "(max |diff| %.3g) %s", d, ok ? "" : err);
            if (ok) pulsar_session_bank_state_save(x, 2);
            ok = pulsar_session_bank_state_restore(x, 1) &&
                 pulsar_session_note_prefilled(x, L.v + pre_len, m1, ord[0] == 1 ? 0 : 1) == 0;
            CHECK(ok && pulsar_session_pos(x) == pre_len + m1 && same(logits_of(x, W), cl1[0], &d),
                  "F8 bank 1's record: %d tokens, its headed row as the next-token logits (max |diff| %.3g)",
                  pulsar_session_pos(x), d);
            if (ok) pulsar_session_bank_state_save(x, 1);
            /* bank 1 continues by sync (an extension of a history it prefilled whole) */
            pulsar_tokens w1 = L;
            w1.len = pre_len + m1 + 6;
            ok = pulsar_session_bank_state_restore(x, 1) && pulsar_session_sync(x, &w1, err, sizeof(err)) == 0;
            const std::vector<float> ref1 = reference(e, &w1, NULL, 0, W);
            pulsar_engine_set_bank_pool(3);
            CHECK(ok && same(logits_of(x, W), ref1, &d), "F8 bank 1's sync past the record == its prompt cold (max "
                  "|diff| %.3g) %s", d, ok ? "" : err);
            if (ok) pulsar_session_bank_state_save(x, 1);
            /* bank 2: the chunk's tokens then a different tail -- resumes from the checkpoint at m2 */
            pulsar_tokens Q = {0};
            pulsar_tokens_copy(&Q, &c2);
            for (int i = 0; i < 40; i++) pulsar_tokens_push(&Q, D.v[100 + i]);
            ok = pulsar_session_bank_state_restore(x, 2) && pulsar_session_sync(x, &Q, err, sizeof(err)) == 0;
            std::vector<float> got_q = ok ? logits_of(x, W) : std::vector<float>();
            const int kq[2] = {kDecode[2], kDecode[3]};
            for (int i = 0; ok && i < 2; i++) ok = pulsar_session_eval(x, kq[i], err, sizeof(err)) == 0;
            std::vector<float> got_q2 = ok ? logits_of(x, W) : std::vector<float>();
            const std::vector<float> refQ = reference(e, &Q, NULL, 0, W), refQ2 = reference(e, &Q, kq, 2, W);
            pulsar_engine_set_bank_pool(3);
            double dq = 0, dq2 = 0;
            CHECK(ok && same(got_q, refQ, &dq) && same(got_q2, refQ2, &dq2),
                  "F8 a divergent sync of bank 2 resumes from the fused chunk's checkpoint == cold: logits + 2 decode "
                  "steps (max |diff| %.3g, %.3g) %s", dq, dq2, ok ? "" : err);
            pulsar_tokens_free(&Q);
        }
        pulsar_session_free(x);
    }
    CHECK(!head1[0].empty() && head1[0] == head1[1] && head2[0] == head2[1], "F5 the bank order moves no byte");
    /* F7: no verify rows in front */
    pulsar_session *x = setup(false);
    const std::vector<float> bare = x ? step(x, 0, orders[0]) : std::vector<float>();
    if (x) pulsar_session_free(x);
    double d1 = 0, d2 = 0;
    CHECK(bare.size() == (size_t)2 * W && same(row(bare, 0), head1[0], &d1) && same(row(bare, 1), head2[0], &d2),
          "F7 the two runs without the verify rows == with them, byte for byte (max |diff| %.3g, %.3g)", d1, d2);
    pulsar_tokens_free(&A);
    pulsar_tokens_free(&L);
    pulsar_tokens_free(&L2);
    pulsar_tokens_free(&D);
}

/* S1 (L284): a request boundary is the core's, for every family -- a sync (and an invalidate) drops the speculative
 * lookahead and re-arms the yield quench.  Before, only DeepSeek's sync did: a Qwen bank whose quench latched
 * kept the latch in its shadow (the bank carry saves the whole speculative state) and served every later request
 * plain.  The latch is forced by its field -- the controller's own trigger needs a stream the drafter loses on --
 * then carried across a bank switch, and the next request on that bank must speculate again. */
static uint64_t spec_drafted(pulsar_engine *e) {
    pulsar_spec_metrics m;
    memset(&m, 0, sizeof m);
    pulsar_engine_spec_metrics(e, &m);
    return m.draft_tokens;
}

static bool spec_generate(pulsar_session *s, int n, char *err, size_t errlen) {
    uint64_t rng = 7;
    for (int got = 0; got < n;) {
        int buf[17];
        const int k = pulsar_session_generate_speculative(s, 0.0f, 0, 1.0f, 0.0f, &rng, n - got, buf, 17, err,
                                                          errlen);
        if (k <= 0) return k == 0;
        got += k;
    }
    return true;
}

static void part_spec_lookahead(pulsar_engine *e) {
    CHECK(pulsar_engine_has_spec_rounds(e), "S1 the family speculates (a drafter behind the round API)");
    if (!pulsar_engine_has_spec_rounds(e)) return;
    pulsar_tokens A = text_tokens(e, "The lighthouse keeper counted the ships every evening, writing each name in a small");
    pulsar_tokens B = text_tokens(e, "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return");
    pulsar_tokens more = text_tokens(e, " book. One night a ship came in without a name, and he wrote");
    pulsar_engine_set_bank_pool(2);
    pulsar_session *s = NULL;
    char err[256] = "";
    CHECK(pulsar_session_create(&s, e, 4096) == 0 && s, "S1 a 2-bank session");
    if (s) {
        CHECK(pulsar_session_bank_state_restore(s, 0) && pulsar_session_sync(s, &A, err, sizeof(err)) == 0,
              "S1 bank 0 prefills A: %s", err);
        uint64_t d0 = spec_drafted(e);
        CHECK(spec_generate(s, 24, err, sizeof(err)) && spec_drafted(e) > d0,
              "S1 bank 0 speculates (%llu draft tokens): %s", (unsigned long long)(spec_drafted(e) - d0), err);
        s->spec.spec_quenched = true;   /* the latch, as the yield quench sets it */
        d0 = spec_drafted(e);
        CHECK(spec_generate(s, 8, err, sizeof(err)) && spec_drafted(e) == d0,
              "S1 a latched bank decodes plain (%llu draft tokens): %s", (unsigned long long)(spec_drafted(e) - d0),
              err);
        pulsar_session_bank_state_save(s, 0);
        CHECK(pulsar_session_bank_state_restore(s, 1) && pulsar_session_sync(s, &B, err, sizeof(err)) == 0,
              "S1 bank 1 prefills B: %s", err);
        pulsar_session_bank_state_save(s, 1);
        CHECK(pulsar_session_bank_state_restore(s, 0) && s->spec.spec_quenched,
              "S1 bank 0's shadow carries the latch across the switch");
        /* the next request on bank 0: its history and a new turn */
        pulsar_tokens next = {0};
        pulsar_tokens_copy(&next, pulsar_session_tokens(s));
        for (int i = 0; i < more.len; i++) pulsar_tokens_push(&next, more.v[i]);
        err[0] = '\0';
        CHECK(pulsar_session_sync(s, &next, err, sizeof(err)) == 0 && !s->spec.spec_quenched,
              "S1 the next request's sync re-arms the quench: %s", err);
        d0 = spec_drafted(e);
        CHECK(spec_generate(s, 24, err, sizeof(err)) && spec_drafted(e) > d0,
              "S1 the next request on the latched bank speculates again (%llu draft tokens): %s",
              (unsigned long long)(spec_drafted(e) - d0), err);
        s->spec.spec_quenched = true;
        pulsar_session_invalidate(s);
        CHECK(!s->spec.spec_quenched && !s->spec.spec_carry_valid && s->spec.n_pend == 0,
              "S1 an invalidate drops the lookahead and re-arms the quench");
        pulsar_tokens_free(&next);
        pulsar_session_free(s);
    }
    pulsar_tokens_free(&A);
    pulsar_tokens_free(&B);
    pulsar_tokens_free(&more);
}

int GATE_ENTRY(int argc, char **argv) {
    n_fail = 0;
    if (argc < 2) { fprintf(stderr, "usage: %s <model> [prefill_chunk]\n", argv[0]); return 2; }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    if (argc > 2) opt.prefill_chunk = (uint32_t)atoi(argv[2]);
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "session-contract: %s did not open\n", argv[1]); return 2; }
    const int W = pulsar_engine_logits_width(e);
    part_chunks(e, W, argc > 2);
    part_banks(e, W);
    part_fused(e, W);
    part_fused_runs(e, W);
    part_spec_lookahead(e);
    gate_engine_close(e);
    printf(n_fail ? "SESSION-CONTRACT GATE FAIL (%d)\n" : "SESSION-CONTRACT GATE PASS\n", n_fail);
    return n_fail != 0;
}

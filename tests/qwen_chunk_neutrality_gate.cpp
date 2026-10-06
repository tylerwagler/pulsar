/* L266 step 5: is a Qwen prefill CUT ANYWHERE byte-identical to one cold prefill?  The question a
 * checkpoint resume rests on (a resume restores the state at G and prefills [G, N) -- a cut at G).
 * The DeepSeek analogue is tests/chunk_neutrality_gate.cpp.  Qwen's per-token kernels (GDN, QSA)
 * are split-invariant by construction (gdn_gate, qsa_attn_gate); its prefill GEMMs are not proven
 * to be: a cut changes which rows share a GEMM.  On the REAL container:
 *
 *   ./tests/qwen_chunk_neutrality_gate <container> [prefill_chunk]
 *
 *   C1  sync(P[0:k]) then sync(P[0:N]) == sync(P[0:N]) cold: the logits, byte for byte, for k on the
 *       128 grid and off it
 *   C2  ... and four decode steps after each, byte for byte (a state difference the last row hides
 *       shows up as soon as the state is read again)
 *   C3  L266 step 5, the checkpoint resume: sync(P) captures the state at P's last grid point; a
 *       DIVERGENT sync(Q) (Q shares P's first k tokens, k past that point) restores it and prefills the
 *       rest -- its logits and four decode steps == Q prefilled cold, byte for byte.  Also after decoding
 *       past P (the checkpoint sits in the prefill-only history, below the decoded rows).
 *   C4  disk segments: sync(P[0:600]) then sync(P) leave checkpoints at 512 and P's last grid point; the
 *       chain [0, 512) + [512, G) saved, loaded into a FRESH session, then sync(Q) == Q cold, byte for
 *       byte (the pools' rows and the slot travel through the file).
 *   C5  L272 P2: the sync is interruptible -- at 256-row chunks the cancel hook stops sync(P) at a chunk
 *       boundary (PULSAR_SESSION_SYNC_INTERRUPTED, the view standing there), the progress hook heard every
 *       chunk in order, and the next sync(P) finishes the prompt: logits + 4 decode steps == cold (at the
 *       default chunk), byte for byte.  The server yields a long prefill and honours a disconnect this way.
 * In the battery since L272 P2 (`make qwen-chunk-neutrality-gate-device`, QWEN_GATE_MODEL = the real container). */
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

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <container> [prefill_chunk]\n", argv[0]); return 2; }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    if (argc > 2) opt.prefill_chunk = (uint32_t)atoi(argv[2]);
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "qwen-chunk-neutrality: %s did not open\n", argv[1]); return 2; }
    const int W = pulsar_engine_logits_width(e);

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
    pulsar_tokens P = {0};
    pulsar_tokenize_text(e, text.c_str(), &P);
    const int N = P.len;
    printf("qwen-chunk-neutrality: %d tokens, prefill_chunk %s\n", N, argc > 2 ? argv[2] : "default");

    const std::vector<std::vector<float>> cold = run(e, &P, {N}, W);
    CHECK(cold.size() == 5, "cold prefill + 4 decode steps ran");
    if (cold.size() != 5) return 1;
    for (int k : {128, 256, 512, 1024, 100, 333, 777}) {
        if (k >= N) continue;
        const std::vector<std::vector<float>> cut = run(e, &P, {k, N}, W);
        if (cut.size() != 5) { CHECK(false, "cut at %d ran", k); continue; }
        CHECK(memcmp(cut[0].data(), cold[0].data(), (size_t)W * sizeof(float)) == 0,
              "C1 cut at %4d (%s grid): logits == cold, max |diff| %.3e", k, k % 128 ? "off" : "on ", maxdiff(cut[0], cold[0]));
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
        CHECK(ok && G == N / 128 * 128 && G > 0, "C3 (%d decoded) sync(P) left a checkpoint at %d (want %d): %s",
              decoded, G, N / 128 * 128, err);
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
        const int Gend = N / 128 * 128, k = N - 37;
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
    if (argc > 2) {
        printf("  skip  C5 (an explicit prefill_chunk pins the session's chunk; C5 sets its own)\n");
    } else if (N <= 512) {
        CHECK(false, "C5 needs a prompt of more than two 256-row chunks (have %d)", N);
    } else {
        setenv("PULSAR_CUDA_PREFILL_CHUNK", "256", 1);
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
                same = memcmp(got[i].data(), cold[i].data(), (size_t)W * sizeof(float)) == 0;
                worst = fmax(worst, maxdiff(got[i], cold[i]));
            }
            CHECK(same, "C5 interrupted + resumed at 256-row chunks: logits + 4 decode steps == cold, max |diff| %.3e %s",
                  worst, err);
            pulsar_session_free(s);
        }
    }
    printf(n_fail ? "QWEN-CHUNK-NEUTRALITY GATE FAIL (%d)\n" : "QWEN-CHUNK-NEUTRALITY GATE PASS\n", n_fail);
    pulsar_tokens_free(&P);
    pulsar_engine_close(e);
    return n_fail != 0;
}

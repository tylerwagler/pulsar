/* L294 / L295: the Inkling forward on the core row-step driver, on a real container, through the public API only.
 *
 *   ./tests/inkling_fwd_gate capture <container> <ids.bin> <out_dir>
 *   ./tests/inkling_fwd_gate exact   <container> <ids.bin | synthetic>
 *
 * ids.bin is the reference capture's prompt (int32 token ids; tests/inkling_mini_grade.py writes it from the
 * transformers fp32 capture).  `synthetic` is 1300 ids of a fixed LCG over the real vocabulary -- the byte invariants
 * need no reference (make inkling-fwd-gate).
 *
 * capture -- what the layer grade reads (tests/inkling_mini_grade.py grades it against the reference):
 *   the PREFILL path: the whole prompt in ONE chunk; with PULSAR_CUDA_GRAPH_DUMP_PREFIX set by the caller, the
 *   driver's dump tap writes the residual stream after every layer for every row ("inkling_h"); the last row's
 *   logits to <out_dir>/prefill_last.f32.
 *   the DECODE path: the first token by a sync, then every next token by pulsar_session_eval (one row a step, the
 *   decode arms); every position's logits (the full row, padding included) to <out_dir>/decode_logits.f32.
 *
 * exact -- the invariants the blocks promise, at the model:
 *   X1  a prompt cut anywhere is the cold prefill: sync(P[0:k]) then sync(P) == sync(P) whole -- the logits and
 *       four decode steps after, byte for byte, k on the 128 grid and off it, under and over F1's 64-row fold
 *   X2  a batched decode is each bank's own: four banks (two prompts twice, different next tokens) decoded in one
 *       pulsar_session_decode_multiseq step == each row decoded alone in a one-bank session, byte for byte; and two
 *       steps on (the state each batched row left is the lone row's)
 *   X3  decode vs prefill (GRADED, not bytes): the decode path's logits at a prompt's last position against the
 *       prefill's -- the EXL3 linears and the routed MoE take their decode arms on a decode row and their prompt
 *       arms on a prompt row (L284 #2), so the two are one model's two arms, graded by max |diff| and the argmax
 *   X4  prefix reuse: sync(P) leaves grid checkpoints; a sync of Q (P's first 950 tokens, then another tail)
 *       resumes from the checkpoint at P's last grid point, 896 (a one-chunk prefill's capture) -- its logits and
 *       four decode steps == Q cold, byte for byte
 *   X5  the payload round trip: a session at P + two decoded tokens saved to a payload, loaded into a FRESH session,
 *       both decode four more tokens -- byte for byte; and the snapshot round trip the same way
 *   X6  the bank carry: bank 0 prefills A, bank 1 prefills B, bank 0 restored decodes == a one-bank A + that token */
#include "pulsar.h"
#include "gate_entry.h"
#include "gate_util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

static int n_fail = 0;
#define CHECK(c, ...) GATE_CHECK(n_fail, c, __VA_ARGS__)

static std::vector<int32_t> read_ids(const char *path) {
    std::vector<int32_t> v;
    FILE *fp = fopen(path, "rb");
    if (!fp) return v;
    int32_t x;
    while (fread(&x, sizeof(x), 1, fp) == 1) v.push_back(x);
    fclose(fp);
    return v;
}

static pulsar_tokens toks(const std::vector<int32_t> &ids, size_t lo, size_t hi) {
    pulsar_tokens t;
    memset(&t, 0, sizeof(t));
    for (size_t i = lo; i < hi && i < ids.size(); i++) pulsar_tokens_push(&t, ids[i]);
    return t;
}

static std::vector<float> logits_of(pulsar_session *s, int W) {
    std::vector<float> v((size_t)W);
    if (pulsar_session_copy_logits(s, v.data(), W) != W) v.clear();
    return v;
}

static double maxdiff(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size() || a.empty()) return 1e30;
    double m = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (isinf(a[i]) && isinf(b[i]) && a[i] == b[i]) continue;   /* the padding's -inf */
        m = fmax(m, fabs((double)a[i] - (double)b[i]));
    }
    return m;
}

static bool same(const std::vector<float> &a, const std::vector<float> &b, double *d) {
    *d = maxdiff(a, b);
    return !a.empty() && a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static int argmax(const std::vector<float> &v) {
    int a = 0;
    for (size_t i = 1; i < v.size(); i++) if (v[i] > v[(size_t)a]) a = (int)i;
    return a;
}

/* the next tokens X1 / X4 / X5 decode after a prompt (any ids: the gate compares engines, not text) */
static const int kDecode[4] = {1000, 2024, 13, 200};

/* sync(P) (after sync(P[0:k]) when k > 0) and the logits + four decode steps after, in a fresh one-bank session */
static std::vector<std::vector<float>> run(pulsar_engine *e, const std::vector<int32_t> &ids, size_t n, size_t k,
                                           int W, int ctx) {
    std::vector<std::vector<float>> out;
    pulsar_engine_set_bank_pool(1);
    pulsar_session *s = NULL;
    char err[256] = "";
    if (pulsar_session_create(&s, e, ctx) != 0 || !s) { fprintf(stderr, "session create failed\n"); return out; }
    bool ok = true;
    if (k > 0) {
        pulsar_tokens a = toks(ids, 0, k);
        ok = pulsar_session_sync(s, &a, err, sizeof(err)) == 0;
        pulsar_tokens_free(&a);
    }
    pulsar_tokens P = toks(ids, 0, n);
    if (ok) ok = pulsar_session_sync(s, &P, err, sizeof(err)) == 0;
    pulsar_tokens_free(&P);
    if (ok) {
        out.push_back(logits_of(s, W));
        for (int i = 0; i < 4 && ok; i++) {
            ok = pulsar_session_eval(s, kDecode[i], err, sizeof(err)) == 0;
            if (ok) out.push_back(logits_of(s, W));
        }
    }
    if (!ok) { fprintf(stderr, "run(k=%zu): %s\n", k, err); out.clear(); }
    pulsar_session_free(s);
    return out;
}

static bool same_run(const std::vector<std::vector<float>> &a, const std::vector<std::vector<float>> &b, double *d) {
    *d = 0;
    if (a.size() != 5 || b.size() != 5) { *d = 1e30; return false; }
    bool ok = true;
    for (size_t i = 0; i < 5; i++) {
        double di = 0;
        ok = same(a[i], b[i], &di) && ok;
        *d = fmax(*d, di);
    }
    return ok;
}

static int capture(pulsar_engine *e, const std::vector<int32_t> &ids, const char *out_dir, int W) {
    const int T = (int)ids.size();
    char path[1024], err[256] = "";
    pulsar_engine_set_bank_pool(1);
    /* the prefill path: the whole prompt in one chunk (the engine's prefill chunk is the caller's, >= T) */
    {
        pulsar_session *s = NULL;
        CHECK(pulsar_session_create(&s, e, T + 64) == 0 && s, "capture: a one-bank session of %d tokens", T + 64);
        if (!s) return 1;
        CHECK(pulsar_session_prefill_cap(s) >= T, "capture: the prefill chunk %d holds the prompt (%d) whole",
              pulsar_session_prefill_cap(s), T);
        pulsar_tokens P = toks(ids, 0, ids.size());
        const double t0 = gate_now_s();
        const int rc = pulsar_session_sync(s, &P, err, sizeof(err));
        CHECK(rc == 0, "capture: the prefill of %d tokens (%.2f s): %s", T, gate_now_s() - t0, err);
        pulsar_tokens_free(&P);
        const std::vector<float> l = logits_of(s, W);
        snprintf(path, sizeof(path), "%s/prefill_last.f32", out_dir);
        FILE *fp = fopen(path, "wb");
        CHECK(fp && !l.empty() && fwrite(l.data(), sizeof(float), l.size(), fp) == l.size(), "capture: wrote %s", path);
        if (fp) fclose(fp);
        pulsar_session_free(s);
    }
    /* the decode path: one row a step */
    {
        pulsar_session *s = NULL;
        CHECK(pulsar_session_create(&s, e, T + 64) == 0 && s, "capture: a second session");
        if (!s) return 1;
        snprintf(path, sizeof(path), "%s/decode_logits.f32", out_dir);
        FILE *fp = fopen(path, "wb");
        pulsar_tokens P = toks(ids, 0, 1);
        bool ok = fp && pulsar_session_sync(s, &P, err, sizeof(err)) == 0;
        pulsar_tokens_free(&P);
        const double t0 = gate_now_s();
        for (int i = 0; ok && i < T; i++) {
            if (i > 0) ok = pulsar_session_eval(s, ids[(size_t)i], err, sizeof(err)) == 0;
            const std::vector<float> l = ok ? logits_of(s, W) : std::vector<float>();
            ok = ok && !l.empty() && fwrite(l.data(), sizeof(float), l.size(), fp) == l.size();
        }
        const double dt = gate_now_s() - t0;
        CHECK(ok, "capture: %d positions decoded one row a step (%.2f s, %.1f ms a step): %s", T, dt, 1e3 * dt / T, err);
        if (fp) fclose(fp);
        pulsar_session_free(s);
    }
    return 0;
}

static void exact(pulsar_engine *e, const std::vector<int32_t> &ids, int W) {
    const int ctx = 2048;
    const size_t N = ids.size() < 1000 ? ids.size() : 1000;
    /* X1 */
    {
        const std::vector<std::vector<float>> cold = run(e, ids, N, 0, W, ctx);
        CHECK(cold.size() == 5, "X1 the cold prefill of %zu tokens + 4 decode steps ran", N);
        for (const size_t k : {(size_t)1, (size_t)37, (size_t)63, (size_t)64, (size_t)65, (size_t)128, (size_t)200,
                               (size_t)511, (size_t)512, (size_t)513, (size_t)640, (size_t)999}) {
            if (k >= N) continue;
            double d = 0;
            const std::vector<std::vector<float>> cut = run(e, ids, N, k, W, ctx);
            CHECK(same_run(cut, cold, &d), "X1 sync(P[0:%zu]) then sync(P): logits + 4 decode steps == cold (max |diff| "
                  "%.3g)", k, d);
        }
    }
    /* X2: four banks (A, B, A, B -- prompts of 300 and 257 tokens) */
    {
        const size_t la = 300, lb0 = 400, lb = 257;
        const int next[4] = {1000, 2024, 13, 200};
        std::vector<std::vector<float>> alone[4];
        for (int b = 0; b < 4; b++) {
            pulsar_engine_set_bank_pool(1);
            pulsar_session *s = NULL;
            char err[256] = "";
            if (pulsar_session_create(&s, e, ctx) != 0 || !s) break;
            pulsar_tokens P = b % 2 ? toks(ids, lb0, lb0 + lb) : toks(ids, 0, la);
            bool ok = pulsar_session_sync(s, &P, err, sizeof(err)) == 0;
            for (int step = 0; ok && step < 2; step++) {
                ok = pulsar_session_eval(s, next[(b + step) % 4], err, sizeof(err)) == 0;
                if (ok) alone[b].push_back(logits_of(s, W));
            }
            pulsar_tokens_free(&P);
            pulsar_session_free(s);
        }
        pulsar_engine_set_bank_pool(4);
        pulsar_session *s = NULL;
        char err[256] = "";
        CHECK(pulsar_session_create(&s, e, ctx) == 0 && s && pulsar_session_bank_count(s) == 4, "X2 a 4-bank session");
        if (s) {
            int pos[4];
            for (int b = 0; b < 4; b++) {
                pulsar_tokens P = b % 2 ? toks(ids, lb0, lb0 + lb) : toks(ids, 0, la);
                pos[b] = P.len;
                CHECK(pulsar_session_bank_state_restore(s, (uint32_t)b) && pulsar_session_sync(s, &P, err, sizeof(err)) == 0,
                      "X2 bank %d prefills %d tokens: %s", b, P.len, err);
                pulsar_session_bank_state_save(s, (uint32_t)b);
                pulsar_tokens_free(&P);
            }
            for (int step = 0; step < 2; step++) {
                pulsar_multiseq_req rows[4];
                for (int b = 0; b < 4; b++) rows[b] = {(uint32_t)b, pos[b] + step, next[(b + step) % 4]};
                std::vector<float> out((size_t)4 * W);
                err[0] = '\0';
                const int rc = pulsar_session_decode_multiseq(s, rows, 4, out.data(), 4 * W, err, sizeof(err));
                CHECK(rc == 0, "X2 step %d: one batched decode of four banks: %s", step, err);
                for (int b = 0; rc == 0 && b < 4; b++) {
                    double d = 0;
                    const std::vector<float> r(out.begin() + (size_t)b * W, out.begin() + (size_t)(b + 1) * W);
                    CHECK(alone[b].size() == 2 && same(r, alone[b][(size_t)step], &d),
                          "X2 step %d bank %d's batched row == the row decoded alone (max |diff| %.3g)", step, b, d);
                }
            }
            pulsar_session_free(s);
        }
    }
    /* X3: the decode path's last row vs the prefill's, graded */
    {
        const size_t n = 300;
        pulsar_engine_set_bank_pool(1);
        pulsar_session *s = NULL;
        char err[256] = "";
        std::vector<float> pre, dec;
        if (pulsar_session_create(&s, e, ctx) == 0 && s) {
            pulsar_tokens P = toks(ids, 0, n);
            if (pulsar_session_sync(s, &P, err, sizeof(err)) == 0) pre = logits_of(s, W);
            pulsar_tokens_free(&P);
            pulsar_session_free(s);
        }
        s = NULL;
        if (pulsar_session_create(&s, e, ctx) == 0 && s) {
            pulsar_tokens P = toks(ids, 0, 1);
            bool ok = pulsar_session_sync(s, &P, err, sizeof(err)) == 0;
            for (size_t i = 1; ok && i < n; i++) ok = pulsar_session_eval(s, ids[i], err, sizeof(err)) == 0;
            if (ok) dec = logits_of(s, W);
            pulsar_tokens_free(&P);
            pulsar_session_free(s);
        }
        double d = maxdiff(pre, dec), num = 0, den = 0;
        for (size_t i = 0; i < pre.size() && i < dec.size(); i++)
            if (isfinite(pre[i]) && isfinite(dec[i])) { num += (pre[i] - dec[i]) * (double)(pre[i] - dec[i]); den += (double)pre[i] * pre[i]; }
        CHECK(!pre.empty() && !dec.empty() && argmax(pre) == argmax(dec),
              "X3 (graded) %zu tokens: the decode path's last logits vs the prefill's -- argmax %d / %d, max |diff| %.3g, "
              "rel-fro %.3e (the decode vs prompt arms of the EXL3 linears and the MoE)", n,
              pre.empty() ? -1 : argmax(pre), dec.empty() ? -1 : argmax(dec), d, den > 0 ? sqrt(num / den) : 0.0);
    }
    /* X4: a divergent sync resumes from the checkpoint at P's last grid point, 896 */
    {
        std::vector<int32_t> q(ids.begin(), ids.begin() + 950);
        for (size_t i = 400; i < 500; i++) q.push_back(ids[i]);
        const std::vector<std::vector<float>> cold = run(e, q, q.size(), 0, W, ctx);
        pulsar_engine_set_bank_pool(1);
        pulsar_session *s = NULL;
        char err[256] = "";
        std::vector<std::vector<float>> got;
        if (pulsar_session_create(&s, e, ctx) == 0 && s) {
            pulsar_tokens P = toks(ids, 0, N), Q = toks(q, 0, q.size());
            bool ok = pulsar_session_sync(s, &P, err, sizeof(err)) == 0 && pulsar_session_sync(s, &Q, err, sizeof(err)) == 0;
            const int origin = ok ? pulsar_session_resume_origin(s) : -1;
            CHECK(ok && origin == 896, "X4 sync(P) then sync(Q) diverging at 950 resumed from %d (want 896): %s", origin, err);
            if (ok) {
                got.push_back(logits_of(s, W));
                for (int i = 0; i < 4 && ok; i++) {
                    ok = pulsar_session_eval(s, kDecode[i], err, sizeof(err)) == 0;
                    if (ok) got.push_back(logits_of(s, W));
                }
            }
            pulsar_tokens_free(&P);
            pulsar_tokens_free(&Q);
            pulsar_session_free(s);
        }
        double d = 0;
        CHECK(same_run(got, cold, &d), "X4 the resumed Q: logits + 4 decode steps == Q cold (max |diff| %.3g)", d);
    }
    /* X5: payload and snapshot round trips */
    {
        pulsar_engine_set_bank_pool(1);
        pulsar_session *a = NULL, *b = NULL, *c = NULL;
        char err[256] = "";
        const bool made = pulsar_session_create(&a, e, ctx) == 0 && pulsar_session_create(&b, e, ctx) == 0 &&
                          pulsar_session_create(&c, e, ctx) == 0;
        CHECK(made, "X5 three sessions");
        if (made) {
            pulsar_tokens P = toks(ids, 0, 777);
            bool ok = pulsar_session_sync(a, &P, err, sizeof(err)) == 0 && pulsar_session_eval(a, 1000, err, sizeof(err)) == 0 &&
                      pulsar_session_eval(a, 2024, err, sizeof(err)) == 0;
            CHECK(ok, "X5 session A at 777 + 2 decoded: %s", err);
            pulsar_tokens_free(&P);
            /* the payload */
            const std::string path = std::string(getenv("TMPDIR") ? getenv("TMPDIR") : ".") + "/inkling_fwd_payload.bin";
            FILE *fp = fopen(path.c_str(), "w+b");
            const uint64_t bytes = ok ? pulsar_session_payload_bytes(a) : 0;
            ok = ok && fp && pulsar_session_save_payload(a, fp, err, sizeof(err)) == 0;
            CHECK(ok, "X5 saved a payload of %llu B: %s", (unsigned long long)bytes, err);
            if (ok) {
                fflush(fp);
                rewind(fp);
                ok = pulsar_session_load_payload(b, fp, bytes, err, sizeof(err)) == 0;
                CHECK(ok, "X5 loaded it into a fresh session: %s", err);
            }
            if (fp) { fclose(fp); remove(path.c_str()); }
            /* the snapshot */
            pulsar_session_snapshot snap;
            memset(&snap, 0, sizeof(snap));
            bool sn = pulsar_session_save_snapshot(a, &snap, err, sizeof(err)) == 0 &&
                      pulsar_session_load_snapshot(c, &snap, err, sizeof(err)) == 0;
            CHECK(sn, "X5 a snapshot saved and loaded into a fresh session: %s", err);
            pulsar_session_snapshot_free(&snap);
            for (int i = 0; ok && i < 4; i++) {
                const bool ea = pulsar_session_eval(a, kDecode[i], err, sizeof(err)) == 0;
                const bool eb = pulsar_session_eval(b, kDecode[i], err, sizeof(err)) == 0;
                const bool ec = sn && pulsar_session_eval(c, kDecode[i], err, sizeof(err)) == 0;
                double d1 = 0, d2 = 0;
                const std::vector<float> la = logits_of(a, W);
                CHECK(ea && eb && same(la, logits_of(b, W), &d1), "X5 decode %d: payload-restored == original (max |diff| "
                      "%.3g)", i, d1);
                if (sn) CHECK(ec && same(la, logits_of(c, W), &d2), "X5 decode %d: snapshot-restored == original (max "
                              "|diff| %.3g)", i, d2);
            }
        }
        if (a) pulsar_session_free(a);
        if (b) pulsar_session_free(b);
        if (c) pulsar_session_free(c);
    }
    /* X6: the bank carry */
    {
        const int ta = 1000;
        pulsar_engine_set_bank_pool(1);
        std::vector<float> ref;
        pulsar_session *s = NULL;
        char err[256] = "";
        if (pulsar_session_create(&s, e, ctx) == 0 && s) {
            pulsar_tokens A = toks(ids, 0, 500);
            if (pulsar_session_sync(s, &A, err, sizeof(err)) == 0 && pulsar_session_eval(s, ta, err, sizeof(err)) == 0)
                ref = logits_of(s, W);
            pulsar_tokens_free(&A);
            pulsar_session_free(s);
        }
        pulsar_engine_set_bank_pool(2);
        s = NULL;
        if (pulsar_session_create(&s, e, ctx) == 0 && s) {
            pulsar_tokens A = toks(ids, 0, 500), B = toks(ids, 600, 900);
            bool ok = pulsar_session_bank_state_restore(s, 0) && pulsar_session_sync(s, &A, err, sizeof(err)) == 0;
            pulsar_session_bank_state_save(s, 0);
            ok = ok && pulsar_session_bank_state_restore(s, 1) && pulsar_session_sync(s, &B, err, sizeof(err)) == 0;
            pulsar_session_bank_state_save(s, 1);
            ok = ok && pulsar_session_bank_state_restore(s, 0) && pulsar_session_eval(s, ta, err, sizeof(err)) == 0;
            double d = 0;
            CHECK(ok && same(logits_of(s, W), ref, &d), "X6 bank 0 after bank 1's prefill == one-bank A + token (max "
                  "|diff| %.3g): %s", d, err);
            pulsar_tokens_free(&A);
            pulsar_tokens_free(&B);
            pulsar_session_free(s);
        }
    }
}

int GATE_ENTRY(int argc, char **argv) {
    n_fail = 0;
    if (argc < 4 || (strcmp(argv[1], "capture") && strcmp(argv[1], "exact")) ||
        (!strcmp(argv[1], "capture") && argc < 5)) {
        fprintf(stderr, "usage: %s capture <container> <ids.bin> <out_dir> | exact <container> <ids.bin>\n", argv[0]);
        return 2;
    }
    std::vector<int32_t> ids;
    if (!strcmp(argv[3], "synthetic")) {
        uint32_t x = 2463534242u;
        for (int i = 0; i < 1300; i++) {
            x = x * 1664525u + 1013904223u;
            ids.push_back((int32_t)((x >> 8) % 199000u));   /* ordinary ids, below the special tokens at the table's top */
        }
    } else {
        ids = read_ids(argv[3]);
    }
    if (ids.size() < 64) { fprintf(stderr, "inkling-fwd: %s holds %zu ids (want >= 64)\n", argv[3], ids.size()); return 2; }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[2];
    opt.backend = PULSAR_BACKEND_CUDA;
    opt.prefill_chunk = 2048;   /* the capture's one chunk; X1's cuts are the sync's own */
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "inkling-fwd: %s did not open\n", argv[2]); return 2; }
    const int W = pulsar_engine_logits_width(e);
    printf("inkling-fwd: %s, %zu prompt ids, logits width %d\n", argv[2], ids.size(), W);
    if (!strcmp(argv[1], "capture")) capture(e, ids, argv[4], W);
    else exact(e, ids, W);
    gate_engine_close(e);
    printf(n_fail ? "INKLING-FWD GATE FAIL (%d)\n" : "INKLING-FWD GATE PASS\n", n_fail);
    return n_fail != 0;
}

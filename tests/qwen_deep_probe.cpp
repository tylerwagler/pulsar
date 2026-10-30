/* L280: the session mechanisms past the native 262,144 positions, on the real model, through the public API only.
 *
 * L298's lesson: a mechanism nobody ran at the new length is a mechanism nobody knows works there.  This runs the
 * session contract's must-pass facts (session_contract_gate C1 / C2 / C4, payload_frontier_gate's FRONTIER, the
 * spec == plain contract) at a depth D past 262,144, with the rope the option names (--rope-yarn F; the YaRN lane
 * that serves past the native context), and grades each byte for byte against ONE cold prefill:
 *
 *   COLD     sync(P[0:D]) in one call: the logits, then 4 decode steps of the document's next tokens; its time and
 *            the bank's touched / resident KV bytes (the demand-paged accounting at depth)
 *   C1/C2    resume inside a live session: sync(P[0:k]) with k off the grid and past 262,144, then sync(P[0:D]):
 *            the logits + 4 decode steps == COLD
 *   C4       the disk chain: that session's checkpoints at grid(k) and grid(D) saved as segments [0, Gk) and
 *            [Gk, GD) (each one's bytes reported), loaded into a FRESH session, then sync(P[0:D]) == COLD
 *   SPEC     from C4's session after its decode steps, greedy speculation (the MTP drafter) for N tokens == the
 *            plain greedy decode of COLD's session for N tokens, token for token; tokens / round reported
 *   PAYLOAD  C4's session's frontier saved as a payload, loaded into a fresh session: the next 8 decode steps ==
 *            the live session's, byte for byte, and the payload's bytes == payload_bytes()
 *
 *   qwen_deep_probe <container> <tokens.bin> <D> <scratch dir> [--rope-yarn F] [--spec N]
 *
 * tokens.bin is int32 token ids, at least D + 4 + N.  Scratch holds the segments and the payload (D x ~13.4 KB
 * each); they are deleted at exit.  Exit 0 = every check passed. */
#include "pulsar.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <string>
#include <vector>

static int g_fail;
#define CHECK(c, ...) do { const bool ok_ = (c); if (!ok_) g_fail++; printf("  %s ", ok_ ? "ok  " : "FAIL"); \
                           printf(__VA_ARGS__); printf("\n"); fflush(stdout); } while (0)

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

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

/* the logits after the sync, then after each of the 4 decode steps */
static bool after_sync(pulsar_session *s, const int32_t *next, int W, std::vector<std::vector<float>> *out, char *err,
                       size_t errlen) {
    out->clear();
    out->push_back(logits_of(s, W));
    for (int i = 0; i < 4; i++) {
        if (pulsar_session_eval(s, next[i], err, errlen) != 0) return false;
        out->push_back(logits_of(s, W));
    }
    return true;
}

static bool same_rows(const std::vector<std::vector<float>> &a, const std::vector<std::vector<float>> &b, double *worst) {
    *worst = 0;
    bool same = a.size() == 5 && b.size() == 5;
    for (size_t i = 0; same && i < 5; i++) {
        same = a[i].size() == b[i].size() && memcmp(a[i].data(), b[i].data(), a[i].size() * sizeof(float)) == 0;
        *worst = fmax(*worst, maxdiff(a[i], b[i]));
    }
    return same;
}

int main(int argc, char **argv) {
    double yarn = 0.0;
    int n_spec = 64;
    bool payload_only = false;   /* only PAYLOAD, on one cold prefill (no C1 / C4 / SPEC comparison) */
    std::vector<char *> pos;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--rope-yarn") && i + 1 < argc) yarn = atof(argv[++i]);
        else if (!strcmp(argv[i], "--spec") && i + 1 < argc) n_spec = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--payload-only")) payload_only = true;
        else pos.push_back(argv[i]);
    }
    if (pos.size() != 4) {
        fprintf(stderr, "usage: %s <container> <tokens.bin> <D> <scratch dir> [--rope-yarn F] [--spec N]\n", argv[0]);
        return 2;
    }
    const int D = atoi(pos[2]);
    const std::string dir = pos[3];
    std::vector<int32_t> ids;
    {
        FILE *f = fopen(pos[1], "rb");
        if (!f) { fprintf(stderr, "qwen-deep-probe: cannot open %s\n", pos[1]); return 2; }
        fseek(f, 0, SEEK_END);
        const long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        ids.resize((size_t)(n > 0 ? n : 0) / 4);
        const bool rd = fread(ids.data(), 4, ids.size(), f) == ids.size();
        fclose(f);
        if (!rd || (int)ids.size() < D + 4 + n_spec) {
            fprintf(stderr, "qwen-deep-probe: %s holds %zu tokens, need %d\n", pos[1], ids.size(), D + 4 + n_spec);
            return 2;
        }
    }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = pos[0];
    opt.backend = PULSAR_BACKEND_CUDA;
    opt.tp_rank = -1;
    opt.rope_yarn_factor = yarn;
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "qwen-deep-probe: %s did not open\n", pos[0]); return 2; }
    const int W = pulsar_engine_logits_width(e);
    const int ctx = D + 4 + n_spec + 1024;
    const int32_t *next = &ids[(size_t)D];
    char err[512] = "";
    printf("qwen-deep-probe: D %d, ctx %d, rope-yarn %g, spec %d tokens\n", D, ctx, yarn, n_spec);

    if (payload_only) {
        /* the frontier PAYLOAD grades, from one cold prefill: 4 decode steps, n_spec speculated tokens, one plain
         * step -- then the same save / load / 8-step comparison as below */
        pulsar_engine_set_bank_pool(1);
        pulsar_session *s = NULL, *x = NULL;
        pulsar_tokens P = {(int *)ids.data(), D, D};
        const double t0 = now_s();
        bool ok = pulsar_session_create(&s, e, ctx) == 0 && pulsar_session_sync(s, &P, err, sizeof err) == 0;
        CHECK(ok, "COLD sync(P[0:%d]): %.1f s %s", D, now_s() - t0, err);
        std::vector<std::vector<float>> c;
        ok = ok && after_sync(s, next, W, &c, err, sizeof err);
        uint64_t rng = 7;
        int got = 0, rounds = 0;
        while (ok && got < n_spec) {
            int buf[17];
            const int n = pulsar_session_generate_speculative(s, 0.0f, 0, 1.0f, 0.0f, &rng, n_spec - got, buf, 17, err,
                                                              sizeof err);
            if (n <= 0) { ok = false; break; }
            got += n;
            rounds++;
        }
        for (int i = 0; ok && i < 6; i++) ok = pulsar_session_eval(s, pulsar_session_argmax(s), err, sizeof err) == 0;
        CHECK(ok, "4 decode steps, %d tokens speculated in %d rounds, 6 plain steps: frontier %d %s", got, rounds,
              pulsar_session_pos(s), err);
        const std::string fp_ = dir + "/l280-payload.bin";
        FILE *pf = ok ? fopen(fp_.c_str(), "wb") : NULL;
        const uint64_t pb = ok ? pulsar_session_payload_bytes(s) : 0;
        const double t1 = now_s();
        ok = ok && pf && pulsar_session_save_payload(s, pf, err, sizeof err) == 0;
        const long wrote = pf ? ftell(pf) : 0;
        if (pf) fclose(pf);
        CHECK(ok && (uint64_t)wrote == pb, "PAYLOAD saved at %d: %.2f GiB (payload_bytes %.2f GiB) in %.1f s %s",
              pulsar_session_pos(s), wrote / 1073741824.0, pb / 1073741824.0, now_s() - t1, err);
        pf = ok ? fopen(fp_.c_str(), "rb") : NULL;
        const double t2 = now_s();
        ok = ok && pf && pulsar_session_create(&x, e, ctx) == 0 &&
             pulsar_session_load_payload(x, pf, pb, err, sizeof err) == 0 && pulsar_session_pos(x) == pulsar_session_pos(s);
        if (pf) fclose(pf);
        CHECK(ok, "PAYLOAD loaded into a fresh session at %d in %.1f s %s", x ? pulsar_session_pos(x) : -1,
              now_s() - t2, err);
        int diff_rows = 0;
        for (int i = 0; ok && i < 8; i++) {
            const std::vector<float> ls = logits_of(s, W), lx = logits_of(x, W);
            diff_rows += ls.size() != lx.size() || memcmp(ls.data(), lx.data(), ls.size() * sizeof(float)) != 0;
            const int t = pulsar_session_argmax(s);
            ok = pulsar_session_eval(s, t, err, sizeof err) == 0 && pulsar_session_eval(x, t, err, sizeof err) == 0;
        }
        CHECK(ok && diff_rows == 0, "PAYLOAD the loaded frontier's logits and next 8 decode steps == the live session's "
              "(%d of 8 rows differ) %s", diff_rows, err);
        if (x) pulsar_session_free(x);
        if (s) pulsar_session_free(s);
        remove(fp_.c_str());
        pulsar_engine_close(e);
        printf("\nqwen-deep-probe (payload only): %s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
        return g_fail ? 1 : 0;
    }

    /* COLD, then the plain greedy continuation SPEC compares against */
    std::vector<std::vector<float>> cold;
    std::vector<int> plain;
    {
        pulsar_engine_set_bank_pool(1);
        pulsar_session *s = NULL;
        pulsar_tokens P = {(int *)ids.data(), D, D};
        const double t0 = now_s();
        bool ok = pulsar_session_create(&s, e, ctx) == 0 && pulsar_session_sync(s, &P, err, sizeof err) == 0;
        const double t1 = now_s();
        const uint64_t touched = ok ? pulsar_session_touched_kv_bytes(s).raw() : 0;
        const uint64_t resident = ok ? pulsar_session_resident_bytes(s) : 0;
        CHECK(ok, "COLD sync(P[0:%d]) in one call: %.1f s (%.0f tok/s); touched KV %.2f GiB (%.0f B/token), resident %.2f GiB %s",
              D, t1 - t0, D / (t1 - t0), touched / 1073741824.0, (double)touched / D, resident / 1073741824.0, err);
        ok = ok && after_sync(s, next, W, &cold, err, sizeof err);
        CHECK(ok && cold.size() == 5, "COLD logits + 4 decode steps %s", err);
        for (int i = 0; ok && i < n_spec; i++) {
            const int t = pulsar_session_argmax(s);
            plain.push_back(t);
            ok = pulsar_session_eval(s, t, err, sizeof err) == 0;
        }
        CHECK(ok && (int)plain.size() == n_spec, "COLD plain greedy %d tokens %s", n_spec, err);
        if (s) pulsar_session_free(s);
        if (!ok) { pulsar_engine_close(e); return 1; }
    }

    /* C1/C2, then save the chain (C4) */
    const uint32_t grid = 128;
    const int k = 262144 + 1000 + 37;
    const std::string f1 = dir + "/l280-seg1.bin", f2 = dir + "/l280-seg2.bin", fp_ = dir + "/l280-payload.bin";
    int Gk = 0, GD = 0;
    uint64_t b1 = 0, b2 = 0;
    {
        pulsar_engine_set_bank_pool(1);
        pulsar_session *s = NULL;
        pulsar_tokens Pk = {(int *)ids.data(), k, k}, P = {(int *)ids.data(), D, D};
        const bool grid_ok = pulsar_session_create(&s, e, ctx) == 0 && pulsar_session_resume_grid(s) == grid;
        CHECK(grid_ok, "the resume grid is %u", grid);
        const double t0 = now_s();
        bool ok = grid_ok && k < D && pulsar_session_sync(s, &Pk, err, sizeof err) == 0 &&
                  pulsar_session_sync(s, &P, err, sizeof err) == 0;
        const double t1 = now_s();
        Gk = ok ? pulsar_session_checkpoint_best(s, k) : 0;
        GD = ok ? pulsar_session_checkpoint_best(s, D) : 0;
        /* which checkpoints survive is the ladder's call (checkpoint.cpp: Qwen keeps 2 slots), so the chain is cut
         * at whichever grid checkpoint below k the session kept -- the last one must be D's grid point */
        CHECK(ok && Gk > 0 && Gk % (int)grid == 0 && GD == D / (int)grid * (int)grid,
              "C1 sync(P[0:%d]) then sync(P[0:%d]): %.1f s; checkpoints kept at %d and %d %s", k, D, t1 - t0, Gk, GD,
              err);
        FILE *a = ok ? fopen(f1.c_str(), "wb") : NULL, *b = ok ? fopen(f2.c_str(), "wb") : NULL;
        b1 = ok ? pulsar_session_segment_bytes(s, 0, Gk, NULL) : 0;
        b2 = ok ? pulsar_session_segment_bytes(s, Gk, GD, NULL) : 0;
        const double t2 = now_s();
        ok = ok && a && b && pulsar_session_save_segment(s, a, 0, Gk, NULL, err, sizeof err) == 0 &&
             pulsar_session_save_segment(s, b, Gk, GD, NULL, err, sizeof err) == 0;
        if (a) fclose(a);
        if (b) fclose(b);
        CHECK(ok, "C4 saved [0, %d) %.2f GiB and [%d, %d) %.2f GiB in %.1f s %s", Gk, b1 / 1073741824.0, Gk, GD,
              b2 / 1073741824.0, now_s() - t2, err);
        std::vector<std::vector<float>> got;
        ok = ok && after_sync(s, next, W, &got, err, sizeof err);
        double worst = 0;
        CHECK(ok && same_rows(got, cold, &worst), "C1/C2 resumed at %d: logits + 4 decode steps == COLD, max |diff| %.3e %s",
              k, worst, err);
        if (s) pulsar_session_free(s);
    }

    /* C4: the chain into a fresh session; SPEC; PAYLOAD */
    {
        pulsar_engine_set_bank_pool(1);
        pulsar_session *s = NULL, *x = NULL;
        pulsar_tokens P = {(int *)ids.data(), D, D};
        FILE *a = fopen(f1.c_str(), "rb"), *b = fopen(f2.c_str(), "rb");
        int g1 = 0, g2 = 0;
        const double t0 = now_s();
        bool ok = a && b && pulsar_session_create(&s, e, ctx) == 0 &&
                  pulsar_session_load_segment(s, a, b1, false, &g1, NULL, err, sizeof err) == 0 &&
                  pulsar_session_load_segment(s, b, b2, true, &g2, NULL, err, sizeof err) == 0;
        if (a) fclose(a);
        if (b) fclose(b);
        CHECK(ok && g1 == Gk && g2 == GD, "C4 a fresh session loaded the chain to %d in %.1f s %s", g2, now_s() - t0, err);
        std::vector<std::vector<float>> got;
        ok = ok && pulsar_session_sync(s, &P, err, sizeof err) == 0 && after_sync(s, next, W, &got, err, sizeof err);
        double worst = 0;
        CHECK(ok && same_rows(got, cold, &worst), "C4 sync(P[0:%d]) on the loaded chain: logits + 4 decode steps == COLD, "
              "max |diff| %.3e %s", D, worst, err);

        pulsar_spec_metrics m0, m1;
        memset(&m0, 0, sizeof m0);
        memset(&m1, 0, sizeof m1);
        pulsar_engine_spec_metrics(e, &m0);
        std::vector<int> spec;
        int rounds = 0;
        uint64_t rng = 7;
        const bool has_spec = pulsar_engine_has_spec_rounds(e);
        while (ok && has_spec && (int)spec.size() < n_spec) {
            int buf[17];
            const int n = pulsar_session_generate_speculative(s, 0.0f, 0, 1.0f, 0.0f, &rng, n_spec - (int)spec.size(), buf,
                                                              17, err, sizeof err);
            if (n <= 0) { ok = false; break; }
            rounds++;
            for (int i = 0; i < n && (int)spec.size() < n_spec; i++) spec.push_back(buf[i]);
        }
        pulsar_engine_spec_metrics(e, &m1);
        int first_diff = -1;
        for (int i = 0; i < n_spec && i < (int)spec.size(); i++) if (spec[i] != plain[i]) { first_diff = i; break; }
        CHECK(has_spec && ok && (int)spec.size() == n_spec && first_diff < 0,
              "SPEC greedy speculation == plain greedy for %d tokens at depth %d (first difference %d): %d rounds, "
              "%.2f tokens / round, %llu drafted %s", n_spec, D, first_diff, rounds, (double)n_spec / (rounds ? rounds : 1),
              (unsigned long long)(m1.draft_tokens - m0.draft_tokens), err);

        /* PAYLOAD: the frontier (off the grid, past the prefill, the drafter's state moved) -- after 6 plain steps,
         * which leave the MTP lane at a savable frontier (its index stage is a 4-token block, rewritten only by
         * whole blocks; payload_frontier_gate's order) */
        for (int i = 0; ok && i < 6; i++) ok = pulsar_session_eval(s, pulsar_session_argmax(s), err, sizeof err) == 0;
        FILE *pf = ok ? fopen(fp_.c_str(), "wb") : NULL;
        const uint64_t pb = ok ? pulsar_session_payload_bytes(s) : 0;
        const double t1 = now_s();
        ok = ok && pf && pulsar_session_save_payload(s, pf, err, sizeof err) == 0;
        const long wrote = pf ? ftell(pf) : 0;
        if (pf) fclose(pf);
        CHECK(ok && (uint64_t)wrote == pb, "PAYLOAD saved at %d: %.2f GiB (payload_bytes %.2f GiB) in %.1f s %s",
              pulsar_session_pos(s), wrote / 1073741824.0, pb / 1073741824.0, now_s() - t1, err);
        pf = ok ? fopen(fp_.c_str(), "rb") : NULL;
        ok = ok && pf && pulsar_session_create(&x, e, ctx) == 0 &&
             pulsar_session_load_payload(x, pf, pb, err, sizeof err) == 0 && pulsar_session_pos(x) == pulsar_session_pos(s);
        if (pf) fclose(pf);
        CHECK(ok, "PAYLOAD loaded into a fresh session at %d %s", x ? pulsar_session_pos(x) : -1, err);
        int diff_rows = 0;
        for (int i = 0; ok && i < 8; i++) {
            const std::vector<float> ls = logits_of(s, W), lx = logits_of(x, W);
            diff_rows += ls.size() != lx.size() || memcmp(ls.data(), lx.data(), ls.size() * sizeof(float)) != 0;
            const int t = pulsar_session_argmax(s);
            ok = pulsar_session_eval(s, t, err, sizeof err) == 0 && pulsar_session_eval(x, t, err, sizeof err) == 0;
        }
        CHECK(ok && diff_rows == 0, "PAYLOAD the loaded frontier's logits and next 8 decode steps == the live session's "
              "(%d of 8 rows differ) %s", diff_rows, err);
        if (x) pulsar_session_free(x);
        if (s) pulsar_session_free(s);
    }
    remove(f1.c_str());
    remove(f2.c_str());
    remove(fp_.c_str());
    pulsar_engine_close(e);
    printf("\nqwen-deep-probe: %s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

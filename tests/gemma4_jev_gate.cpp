/* gemma4_jev_gate -- L338 P2: the runtime adapter (Gemma 4's JEV LoRA) through the public API, on a real container
 * that carries it (tools/container --adapter jev).  The capture the decision grade reads (tests/gemma4/jev_grade.py):
 *
 *   gemma4_jev_gate <container> <cases.bin> <out.bin>
 *
 * cases.bin: u32 n, then per case u32 len + i32 ids[len] (jev_grade.py prep writes them from decision_ref.pt);
 * out.bin:   per case, the final-norm hidden of its last prompt token, f32 [hidden] (pulsar_session_copy_hidden).
 *
 * What it checks itself:
 *   J1  the engine carries the adapter; a bank holding tokens refuses the switch; an empty one takes it
 *   J2  every case synced on the adapter bank, one after another (each reuses the bank's common prefix with the case
 *       before: the shared policy), its hidden read back
 *   J3  prefix reuse == cold: the first three cases again, each in a FRESH session (adapter on, nothing reused), byte
 *       for byte the same hidden
 *   J4  the adapter is on the bank only: the first case on a plain bank of the same engine gives a DIFFERENT hidden
 *       (the LoRA ran), and that plain hidden is the same bytes on a second plain session
 * Public API only.  Exit 0 = every check passed (the decision grade is jev_grade.py's). */
#include "pulsar.h"
#include "gate_entry.h"
#include "gate_util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

static int n_fail = 0;
#define CHECK(c, ...) GATE_CHECK(n_fail, c, __VA_ARGS__)

static std::vector<std::vector<int32_t>> read_cases(const char *path) {
    std::vector<std::vector<int32_t>> v;
    FILE *fp = fopen(path, "rb");
    uint32_t n = 0;
    if (!fp || fread(&n, 4, 1, fp) != 1) { if (fp) fclose(fp); return v; }
    for (uint32_t i = 0; i < n; i++) {
        uint32_t len = 0;
        if (fread(&len, 4, 1, fp) != 1) break;
        std::vector<int32_t> ids(len);
        if (fread(ids.data(), 4, len, fp) != len) break;
        v.push_back(std::move(ids));
    }
    fclose(fp);
    return v;
}

static pulsar_tokens toks(const std::vector<int32_t> &ids) {
    pulsar_tokens t;
    memset(&t, 0, sizeof(t));
    for (int32_t x : ids) pulsar_tokens_push(&t, x);
    return t;
}

/* sync `ids` on bank 0 of `s` and read the last token's final-norm hidden; empty on failure */
static std::vector<float> run(pulsar_session *s, const std::vector<int32_t> &ids, int E) {
    char err[256] = "";
    pulsar_tokens P = toks(ids);
    const int rc = pulsar_session_sync(s, &P, err, sizeof(err));
    pulsar_tokens_free(&P);
    std::vector<float> h((size_t)E);
    if (rc != 0) { fprintf(stderr, "sync: %s\n", err); h.clear(); return h; }
    if (pulsar_session_copy_hidden(s, h.data(), E) != E) h.clear();
    return h;
}

/* a fresh one-bank session, adapter on (or off) before any token */
static pulsar_session *fresh(pulsar_engine *e, int ctx, bool adapter) {
    pulsar_engine_set_bank_pool(1);
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, e, ctx) != 0 || !s) return NULL;
    if (adapter && !pulsar_session_set_bank_adapter(s, 0, true)) { pulsar_session_free(s); return NULL; }
    return s;
}

int GATE_ENTRY(int argc, char **argv) {
    n_fail = 0;
    if (argc < 4) {
        fprintf(stderr, "usage: %s <container> <cases.bin> <out.bin>\n", argv[0]);
        return 2;
    }
    const std::vector<std::vector<int32_t>> cases = read_cases(argv[2]);
    if (cases.empty()) { fprintf(stderr, "gemma4-jev: %s holds no cases\n", argv[2]); return 2; }
    size_t longest = 0;
    for (const auto &c : cases) longest = c.size() > longest ? c.size() : longest;
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    opt.prefill_chunk = 2048;
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "gemma4-jev: %s did not open\n", argv[1]); return 2; }
    const int ctx = (int)longest + 64;
    int E = 0;
    {   /* the hidden width: the readout of a one-token sync */
        pulsar_session *s = fresh(e, ctx, false);
        std::vector<float> probe(16384);
        char err[128];
        pulsar_tokens P = toks({cases[0][0]});
        if (s && pulsar_session_sync(s, &P, err, sizeof(err)) == 0) E = pulsar_session_copy_hidden(s, probe.data(), 16384);
        pulsar_tokens_free(&P);
        if (s) pulsar_session_free(s);
    }
    printf("gemma4-jev: %s, %zu cases (longest %zu), hidden %d\n", argv[1], cases.size(), longest, E);
    CHECK(pulsar_engine_has_adapter(e) && E > 0, "J1 the engine carries the runtime adapter; the hidden is %d wide", E);
    if (E <= 0) { pulsar_engine_close(e); return 1; }
    /* J1: the switch moves an empty bank only */
    {
        pulsar_session *s = fresh(e, ctx, false);
        std::vector<float> h = s ? run(s, cases[0], E) : std::vector<float>();
        CHECK(s && !h.empty() && !pulsar_session_set_bank_adapter(s, 0, true) && !pulsar_session_bank_adapter(s, 0),
              "J1 a bank holding %zu tokens refuses the switch", cases[0].size());
        if (s) pulsar_session_free(s);
    }
    /* J2: every case on one adapter bank, in order */
    std::vector<std::vector<float>> hid;
    {
        pulsar_session *s = fresh(e, ctx, true);
        CHECK(s && pulsar_session_bank_adapter(s, 0), "J1 an empty bank takes the switch");
        const double t0 = gate_now_s();
        for (size_t i = 0; s && i < cases.size(); i++) hid.push_back(run(s, cases[i], E));
        const double dt = gate_now_s() - t0;
        size_t ok = 0;
        for (const auto &h : hid) ok += !h.empty();
        CHECK(ok == cases.size(), "J2 %zu / %zu cases synced on the adapter bank, each reusing the one before "
                                  "(%.2f s, %.0f ms a case)", ok, cases.size(), dt, 1e3 * dt / (double)cases.size());
        if (s) pulsar_session_free(s);
    }
    if (hid.size() != cases.size()) { pulsar_engine_close(e); printf("GEMMA4-JEV GATE FAIL\n"); return 1; }
    /* J3: prefix reuse == cold */
    for (size_t i = 0; i < 3 && i < cases.size(); i++) {
        pulsar_session *s = fresh(e, ctx, true);
        const std::vector<float> h = s ? run(s, cases[i], E) : std::vector<float>();
        CHECK(!h.empty() && hid[i].size() == h.size() && !memcmp(h.data(), hid[i].data(), h.size() * 4),
              "J3 case %zu cold (a fresh adapter session) == after the bank's reuse, byte for byte", i);
        if (s) pulsar_session_free(s);
    }
    /* J4: the plain bank */
    {
        pulsar_session *a = fresh(e, ctx, false), *b = fresh(e, ctx, false);
        const std::vector<float> pa = a ? run(a, cases[0], E) : std::vector<float>();
        if (a) pulsar_session_free(a);
        const std::vector<float> pb = b ? run(b, cases[0], E) : std::vector<float>();
        if (b) pulsar_session_free(b);
        double diff = 0;
        for (size_t k = 0; k < pa.size() && k < hid[0].size(); k++) diff = fmax(diff, fabs((double)pa[k] - hid[0][k]));
        CHECK(!pa.empty() && pa.size() == pb.size() && !memcmp(pa.data(), pb.data(), pa.size() * 4) && diff > 0,
              "J4 a plain bank: the same bytes twice, and not the adapter's (max |diff| %.4g)", diff);
    }
    FILE *fp = fopen(argv[3], "wb");
    bool wrote = fp != NULL;
    for (const auto &h : hid) wrote = wrote && !h.empty() && fwrite(h.data(), 4, h.size(), fp) == h.size();
    if (fp) fclose(fp);
    CHECK(wrote, "wrote %zu hidden rows to %s", hid.size(), argv[3]);
    pulsar_engine_close(e);
    printf("GEMMA4-JEV GATE %s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}

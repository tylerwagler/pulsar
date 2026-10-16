/* L296 N9: the chained-MTP drafter (src/engine/spec_mtp.h) on a real container -- any family whose drafter is the MTP
 * lane (Nemotron-H's mini first) -- and (L298) the block drafter (src/engine/spec_block.h: LiLiCorr), whose M0 checks
 * its lane and whose M3 asserts its context rings and anchor equal the plain twin's.  Speculation never changes the
 * output: every check here is the plain decode's bytes.
 *
 *   ./tests/mtp_spec_gate <model> [tokens] [sections]
 *   ./tests/mtp_spec_gate <model> --probe <ids.bin> <out_dir> [positions]
 *
 * Also a runner gate (tests/gates_runner.cpp `mtp-spec-gate`, on every hosted model whose drafter is the MTP lane).
 *
 * exact (`tokens` per stream, default 256; `sections` the digits of the M sections to run, default all):
 *   M0  the lane runs: the engine's drafter is the MTP drafter, a session builds its lane, a prefill parks every bank's
 *       row at its last position
 *   M1  greedy spec == plain: four prompts, prefill + `tokens` tokens, at each fixed depth 1..5 with no stop rule and
 *       at the drafter's own numbers (the adaptive controller from the card's start, the stop rule) -- the emitted
 *       tokens byte for byte; tokens a round reported
 *   M2  seeded T 0.7 spec == plain: the same stream's draws (L288 sdet: a draft sits at the key its token's draw
 *       reads) -- the emitted tokens byte for byte at depth 3 and at the drafter's numbers
 *   M3  the forced-reject stress: before every round the pendings are replaced by the plain run's own continuation
 *       with a corrupted token at a drawn place (from the first draft -- nothing kept -- to none -- the whole chain
 *       kept), up to 15 deep, so each round's commit rolls the verify back to a planned row: the Mamba SSM state
 *       (DELTA), its conv tail (SNAPSHOT), the attention K|V (POOL, position-addressed) and the drafter's slot (its
 *       K|V and the parked row) must be the plain twin's byte for byte at the end -- every trunk slot's resident
 *       state and every pool row the stream wrote, then the drafter's (asserted when the family declares its module
 *       row-exact, pulsar_mtp_ops::row_exact; reported otherwise) -- and four decode steps after == the twin's
 *   M4  the batched lane (the server's round API): four banks' streams in shared verify steps (assemble, one
 *       decode_mixed with every row headed, round_end, ONE redraft pass, commit) == each bank's plain stream, greedy
 *   M5  the batched lane at seeded T 0.7 with each position's mode its own (L296 toolsample: the server's tool-call
 *       override, a call's structure greedy and its payload sampled, flips inside a round): every other run of 3
 *       positions greedy, the plain stream drawing each token at its position's mode and the rounds through
 *       pulsar_spec_step::greedy_rows -- == each bank's plain stream, rounds that cross between the modes counted
 *
 * --probe (the draft grade's capture, tests/nemotron_mtp_grade.py): the first draft step's logits at every position of
 * a reference prompt, teacher-forced by the trunk (the lane's probe: each lockstep module row headed) -- the DECODE
 * path (a sync of the first token, then one eval a token: one row a step, the decode arms) to
 * <out_dir>/mtp_decode.f32, and the PROMPT path (syncs growing by 31 tokens: each lockstep step 31 rows, the prompt
 * arms) to <out_dir>/mtp_prefill.f32; row p is the module's prediction of token p + 2 from (the trunk at p, token
 * p + 1), `positions` rows (default all but the last). */
#include "pulsar.h"
#include "gate_entry.h"
#include "gate_util.h"
#include "pulsar_engine_internal.h"   /* the drafter's numbers, the pendings, the declared state, the lane's probe */
#include "spec_mtp.h"
#include "spec_block.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

static int n_fail = 0;
#define CHECK(c, ...) GATE_CHECK(n_fail, c, __VA_ARGS__)

static const pulsar_drafter_ops *g_base;   /* the engine's own drafter */
static pulsar_drafter_ops g_fixed;         /* a copy at a fixed depth */
static bool g_unpriced;                    /* a fixed depth: no priced depth either (every chain the whole depth) */
/* the engine's own settings, put back when the gate ends (a runner gate borrows the engine) */
static float g_tau0;
static int g_depth0;
static pulsar_lane_cost_fit g_cost0;

/* the drafter's numbers for a run: depth 0 = its own (the controller, the stop rule, the priced depth); else that
 * depth fixed, no stop rule, no price -- every chain drafts the whole depth, so the module runs recursively that deep */
static void drafter_at_depth(pulsar_engine *e, uint32_t depth) {
    g_unpriced = depth != 0;
    if (depth == 0) {
        e->drafter_ops = g_base;
        e->spec_tau = g_tau0;
        e->spec_depth = g_depth0;
        return;
    }
    g_fixed = *g_base;
    g_fixed.depth = depth;
    g_fixed.adapt = NULL;
    e->drafter_ops = &g_fixed;
    e->spec_tau = -1.0f;   /* no stop: every chain drafts the whole depth */
    e->spec_depth = 0;
}

static pulsar_tokens toks(const std::vector<int32_t> &v, size_t lo, size_t hi) {
    pulsar_tokens t;
    memset(&t, 0, sizeof(t));
    for (size_t i = lo; i < hi && i < v.size(); i++) pulsar_tokens_push(&t, v[i]);
    return t;
}

static std::vector<int32_t> vec(const pulsar_tokens &t) { return std::vector<int32_t>(t.v, t.v + t.len); }

static std::vector<float> logits_of(pulsar_session *s, int W) {
    std::vector<float> v((size_t)W);
    if (pulsar_session_copy_logits(s, v.data(), W) != W) v.clear();
    return v;
}

static uint64_t drafted(pulsar_engine *e, uint64_t *rounds, uint64_t *accepted) {
    pulsar_spec_metrics m;
    memset(&m, 0, sizeof m);
    pulsar_engine_spec_metrics(e, &m);
    if (rounds) *rounds = m.num_drafts;
    if (accepted) *accepted = m.accepted_tokens;
    return m.draft_tokens;
}

/* The pendings a forced round verifies: the plain stream's continuation after the carry, corrupted at `bad` (K = none)
 * -- set on the installed bank's shadow as the drafter would have stamped them. */
static void force_pendings(pulsar_session *s, const std::vector<int32_t> &ref, size_t at, uint32_t K, uint32_t bad) {
    pulsar_spec_carry_state &sp = s->spec;
    if (!sp.spec_carry_valid || at >= ref.size()) return;
    uint32_t k = 0;
    for (; k < K && at + 1 + k < ref.size(); k++) {
        int32_t t = ref[at + 1 + k];
        if (k == bad) t = t == 1000 ? 1001 : 1000;   /* an ordinary id the stream does not have here */
        sp.pend[k] = t;
        sp.pend_conf[k] = 1.0f;
    }
    sp.dspark_chain_unharvested = false;
    sp.n_pend = k;
    sp.pend_copy = false;
    sp.pend_base = sp.spec_carry_token;
    sp.pend_pos = (int32_t)s->checkpoint.len;
    sp.pend_temp = 0.0f;
    sp.pend_top_k = 0;
    sp.pend_top_p = 1.0f;
    sp.pend_min_p = 0.0f;
}

struct stream {
    std::vector<int32_t> out;   /* the emitted tokens */
    uint64_t rounds = 0, drafts = 0, accepted = 0;
    bool ok = false;
};

/* M5's per-position sampling: the stream's token `pos` (0 = the first generated) draws greedy in every other run of
 * `greedy_run` positions (0 = never) -- a stand-in for the server's tool-call override (structure greedy, payload
 * sampled), which flips inside a speculative round. */
static bool pos_greedy(int greedy_run, size_t pos) { return greedy_run > 0 && (pos / (size_t)greedy_run) % 2u == 1u; }

/* One stream on a fresh one-bank session: sync(prompt), then `n` tokens -- plain (quenched: the plain step the round
 * API takes), or speculative; `force` (a reference stream) replaces every round's pendings (M3); `keep` leaves the
 * session open in *keep_s; `greedy_run` (plain only) draws each token at its position's mode (pos_greedy). */
static stream run(pulsar_engine *e, const pulsar_tokens &prompt, int n, bool spec, float temp, uint64_t seed,
                  const std::vector<int32_t> *force = NULL, pulsar_session **keep_s = NULL, int greedy_run = 0) {
    stream r;
    pulsar_engine_set_bank_pool(1);
    pulsar_session *s = NULL;
    char err[256] = "";
    if (pulsar_session_create(&s, e, 4096) != 0 || !s) { fprintf(stderr, "session create failed\n"); return r; }
    bool ok = pulsar_session_sync(s, &prompt, err, sizeof(err)) == 0;
    uint64_t r0 = 0, a0 = 0;
    const uint64_t d0 = drafted(e, &r0, &a0);
    uint64_t rng = seed;
    uint32_t lcg = 12345u;
    while (ok && (int)r.out.size() < n) {
        /* plain: the quench's own route.  spec: the yield quench disarmed -- it latches on a drafter that loses (the
         * mini's), and a latched round drafts nothing for the next */
        s->spec.spec_quenched = !spec;
        s->spec.spec_quench_steps = 0;
        s->spec.spec_quench_debt = 0.0f;
        s->spec.spec_quench_ewma = 0.0f;
        /* unpriced: the spec lane's measured cost forgotten before every round, so no chain is priced out (L288: the
         * price needs the fit) */
        if (g_unpriced) memset(&e->lane_cost[PULSAR_LANE_SPEC], 0, sizeof(e->lane_cost[PULSAR_LANE_SPEC]));
        if (force && spec) {
            lcg = lcg * 1664525u + 1013904223u;
            const uint32_t K = 1u + (lcg >> 8) % PULSAR_MTP_DRAFT_MAX;
            const uint32_t bad = (lcg >> 20) % (K + 1u);   /* K: the whole chain is the stream's */
            force_pendings(s, *force, r.out.size(), K, bad);
        }
        int buf[17];
        const float t = !spec && pos_greedy(greedy_run, r.out.size()) ? 0.0f : temp;   /* a plain step: one token */
        const int k = pulsar_session_generate_speculative(s, t, 0, 1.0f, 0.0f, &rng, n - (int)r.out.size(), buf, 17,
                                                          err, sizeof(err));
        if (k <= 0) {
            ok = k == 0;
            break;
        }
        for (int i = 0; i < k; i++) r.out.push_back(buf[i]);
        if (pulsar_token_is_stop(e, buf[k - 1])) break;
    }
    if (!ok) fprintf(stderr, "stream (%s): %s\n", spec ? "spec" : "plain", err);
    r.drafts = drafted(e, &r.rounds, &r.accepted) - d0;
    r.rounds -= r0;
    r.accepted -= a0;
    r.ok = ok;
    if (keep_s) *keep_s = s;
    else pulsar_session_free(s);
    return r;
}

static int first_diff(const std::vector<int32_t> &a, const std::vector<int32_t> &b) {
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; i++) if (a[i] != b[i]) return (int)i;
    return a.size() == b.size() ? -1 : (int)n;
}

/* A bank's declared state, read off the device as the frontier `rows` describes it: slots [k0, k1), each resident
 * decl's bank bytes (FIXED, RING whole; STAGED: the open block's rows below the frontier -- a rejected verify row past
 * it is no state) and the whole rows below the frontier of each pool (a drafter slot trails the trunk by one row: its
 * caller passes one row fewer).  One piece per (slot, decl), named, so a difference says where it is. */
struct state_piece {
    std::string name;
    std::vector<uint8_t> bytes;
};
static std::vector<state_piece> bank_state(pulsar_session *s, uint32_t b, uint32_t k0, uint32_t k1, uint32_t rows) {
    std::vector<state_piece> out;
    const pulsar_state_set *ss = s->states;
    const pulsar_state_plan *p = ss->plan;
    bool ok = true;
    for (uint32_t k = k0; ok && k < k1; k++) {
        for (uint32_t id = 0; ok && id < p->n_decl; id++) {
            const pulsar_state_decl *d = &p->decl[id];
            if (!(p->slot[k] >> id & 1u)) continue;
            const pulsar_gpu_tensor *t;
            uint64_t off, n;
            if (d->kind == PULSAR_STATE_POOL) {
                n = (uint64_t)rows / d->tokens * d->bytes;
                t = pulsar_state_pool(ss, k, id, b);
                off = 0;
            } else {
                n = d->kind == PULSAR_STATE_STAGED ? (uint64_t)(rows % d->tokens) * d->bytes
                                                    : pulsar_state_decl_bank_bytes(d);
                t = pulsar_state_tensor(ss, k, id);
                off = (uint64_t)b * pulsar_state_decl_bank_bytes(d);
            }
            state_piece pc;
            pc.name = std::string(d->name) + "@" + std::to_string(k);
            pc.bytes.resize(n);
            ok = t && (n == 0 || pulsar_gpu_tensor_read(t, off, pc.bytes.data(), n) != 0);
            out.push_back(std::move(pc));
        }
    }
    if (!ok) out.clear();
    return out;
}

/* the pieces' bytes and the first piece that differs ("" = none) */
static std::string state_diff(const std::vector<state_piece> &a, const std::vector<state_piece> &b, size_t *bytes) {
    *bytes = 0;
    for (const state_piece &p : a) *bytes += p.bytes.size();
    if (a.empty() || a.size() != b.size()) return "the state did not read";
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].name != b[i].name || a[i].bytes.size() != b[i].bytes.size()) return a[i].name;
        size_t at = 0, last = 0, n = 0;
        for (size_t x = 0; x < a[i].bytes.size(); x++)
            if (a[i].bytes[x] != b[i].bytes[x]) {
                if (n++ == 0) at = x;
                last = x;
            }
        if (n) return a[i].name + " (" + std::to_string(n) + " bytes from offset " + std::to_string(at) + " to " +
                      std::to_string(last) + " of " + std::to_string(a[i].bytes.size()) + ")";
    }
    return "";
}

/* four plain decode steps after the stream: the session's logits each step */
static std::vector<std::vector<float>> decode4(pulsar_session *s, int W) {
    static const int kDecode[4] = {1000, 2024, 13, 200};
    std::vector<std::vector<float>> out;
    char err[256] = "";
    for (int i = 0; i < 4; i++) {
        if (pulsar_session_eval(s, kDecode[i], err, sizeof(err)) != 0) { out.clear(); break; }
        out.push_back(logits_of(s, W));
    }
    return out;
}

static std::vector<pulsar_tokens> prompts(pulsar_engine *e) {
    std::vector<pulsar_tokens> P;
    pulsar_tokens story;
    if (!gate_load_story(e, &story, 1200)) return P;
    const std::vector<int32_t> v = vec(story);
    P.push_back(toks(v, 0, 96));
    P.push_back(toks(v, 300, 500));
    P.push_back(toks(v, 700, 1033));
    pulsar_tokens_free(&story);
    pulsar_tokens code;
    memset(&code, 0, sizeof(code));
    pulsar_tokenize_text(e, "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n"
                            "    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n", &code);
    P.push_back(code);
    return P;
}

/* M4 / M5: the batched lane (the server's round API) -- four banks' streams in shared verify steps (assemble, one
 * decode_mixed with every row headed, round_end, ONE redraft pass, commit), each == its plain stream `ref`.  At
 * `temp` > 0 with `greedy_run` set, every position draws at its own mode as the server's per-position override does
 * (L296 toolsample): the assemble's base at the base's, the round's rows at theirs through greedy_rows. */
static void batched_lane(pulsar_engine *e, const std::vector<pulsar_tokens> &P, int n, int W,
                         const std::vector<stream> &ref, const char *tag, float temp, uint64_t seed0, int greedy_run) {
    const int NB = 4;
    pulsar_engine_set_bank_pool(NB);
    pulsar_session *s = NULL;
    char err[256] = "";
    bool ok = pulsar_session_create(&s, e, 4096) == 0 && s;
    for (int b = 0; ok && b < NB; b++) {
        ok = pulsar_session_bank_state_restore(s, (uint32_t)b) && pulsar_session_sync(s, &P[b], err, sizeof(err)) == 0;
        if (ok) pulsar_session_bank_state_save(s, (uint32_t)b);
    }
    CHECK(ok, "%s four banks prefilled: %s", tag, err);
    std::vector<std::vector<int32_t>> got(NB);
    pulsar_spec_round *rd[NB];
    uint64_t rngs[NB];
    bool done[NB] = {};
    for (int b = 0; b < NB; b++) { rd[b] = pulsar_spec_round_new(); rngs[b] = temp > 0.0f ? seed0 + (uint64_t)b : 0u; }
    const uint64_t d0 = drafted(e, NULL, NULL);
    const uint32_t width = pulsar_engine_fused_heads_max(e);
    std::vector<pulsar_multiseq_req> reqs(width);
    static int acc[NB][PULSAR_SPEC_LOGITS_ROWS + 1];
    int ticks = 0, crossing = 0;
    while (ok && ticks++ < 4 * n) {
        pulsar_spec_step st[NB];
        uint32_t bank_of[NB];
        int m = 0;
        for (int b = 0; b < NB; b++) {
            if (done[b]) continue;
            /* the yield quench disarmed, as in run() */
            ok = pulsar_session_bank_state_restore(s, (uint32_t)b);
            s->spec.spec_quenched = false;
            s->spec.spec_quench_steps = 0;
            s->spec.spec_quench_debt = 0.0f;
            s->spec.spec_quench_ewma = 0.0f;
            pulsar_session_bank_state_save(s, (uint32_t)b);
            if (!ok) break;
            memset(&st[m], 0, sizeof st[m]);
            st[m].bank = (uint32_t)b;
            st[m].round = rd[b];
            st[m].temperature = pos_greedy(greedy_run, got[b].size()) ? 0.0f : temp;   /* the base's position */
            st[m].top_p = 1.0f;
            st[m].rng = &rngs[b];
            st[m].max_tokens = n - (int)got[b].size();
            st[m].k_alloc = -1;
            st[m].accepted_cap = (int)(PULSAR_SPEC_LOGITS_ROWS + 1);
            bank_of[m++] = (uint32_t)b;
        }
        if (m == 0) break;
        uint32_t rows = 0;
        (void)pulsar_session_spec_assemble_batch(s, st, m, width, reqs.data(), &rows);
        /* as the server's lane: a bank whose base is a stop token emits it and ends (never evaluated); a bank
         * over the row budget sits the tick out untouched; the rest verify (a deep chain -- the drafter's cap 7,
         * a copy chain -- can fill the width) */
        int live = 0;
        for (int i = 0; ok && i < m; i++) {
            if (st[i].status == PULSAR_SPEC_STEP_EOS) {
                got[bank_of[i]].push_back(st[i].first_token);
                done[bank_of[i]] = true;
            } else if (st[i].status == PULSAR_SPEC_STEP_OK) {
                st[live] = st[i];
                bank_of[live++] = bank_of[i];
            } else if (st[i].status != PULSAR_SPEC_STEP_SKIPPED) {
                ok = false;
                snprintf(err, sizeof err, "assemble bank %u: status %d %s", st[i].bank, (int)st[i].status, st[i].err);
            }
        }
        if (!ok) break;
        m = live;
        if (m == 0 || rows == 0) continue;
        std::vector<float> logits((size_t)rows * W);
        uint32_t gotr = 0;
        pulsar_session_spec_arm_capture(s, rows);
        ok = pulsar_session_decode_mixed(s, reqs.data(), rows, logits.data(), (int)rows * W, &gotr,
                                         PULSAR_MSEQ_HEAD_ALL_ROWS, err, sizeof(err)) == 0 && gotr == rows;
        pulsar_session_spec_arm_capture(s, 0u);
        if (!ok) break;
        for (int i = 0; i < m; i++) {
            st[i].accepted = acc[i];
            st[i].accepted_cap = (int)(PULSAR_SPEC_LOGITS_ROWS + 1);
            /* row j: the token after the base (position got.size()) and j drafts */
            st[i].temperature = temp;
            st[i].greedy_rows = 0u;
            const size_t base_pos = got[bank_of[i]].size();
            for (uint32_t j = 0; j < st[i].n_rows; j++)
                if (pos_greedy(greedy_run, base_pos + 1u + j)) st[i].greedy_rows |= 1u << j;
            const uint32_t all = st[i].n_rows >= 32u ? ~0u : (1u << st[i].n_rows) - 1u;
            crossing += st[i].n_rows > 1u && st[i].greedy_rows != 0u && st[i].greedy_rows != all;
        }
        ok = pulsar_session_spec_round_end_batch(s, st, m, logits.data()) == 0;
        for (int i = 0; ok && i < m; i++) {
            ok = st[i].status == PULSAR_SPEC_STEP_OK;
            if (!ok) { snprintf(err, sizeof err, "round_end bank %u: %s", st[i].bank, st[i].err); break; }
            const uint32_t b = bank_of[i];
            for (int k = 0; k < st[i].n_accepted; k++) got[b].push_back(acc[i][k]);
            if ((int)got[b].size() >= n || (st[i].n_accepted && pulsar_token_is_stop(e, acc[i][st[i].n_accepted - 1])))
                done[b] = true;
        }
        if (!ok) break;
        pulsar_spec_round *rr[NB];
        uint64_t *rp[NB];
        for (int i = 0; i < m; i++) { rr[i] = rd[bank_of[i]]; rp[i] = &rngs[bank_of[i]]; }
        ok = pulsar_session_spec_redraft_batch(s, rr, bank_of, rp, m, err, sizeof(err)) == 0 &&
             pulsar_session_spec_redraft_commit_batch(s, st, m) == 0;
    }
    const uint64_t dd = drafted(e, NULL, NULL) - d0;
    CHECK(ok && dd > 0, "%s the batched lane ran %d ticks, %llu drafts: %s", tag, ticks, (unsigned long long)dd, err);
    if (greedy_run > 0)
        CHECK(crossing > 0, "%s %d verified rounds crossed between greedy and sampled positions", tag, crossing);
    for (int b = 0; b < NB; b++) {
        const int d = first_diff(got[b], ref[b].out);
        CHECK(ok && d == -1, "%s bank %d: %zu tokens in shared verify steps == its plain stream (first difference "
              "at %d)", tag, b, got[b].size(), d);
    }
    for (int b = 0; b < NB; b++) pulsar_spec_round_free(rd[b]);
    if (s) pulsar_session_free(s);
}

static int exact(pulsar_engine *e, int n, const char *only) {
    auto on = [only](char c) { return !only || strchr(only, c) != NULL; };   /* the sections to run (M0 always) */
    const int W = pulsar_engine_logits_width(e);
    g_base = e->drafter_ops;
    g_tau0 = e->spec_tau;
    g_depth0 = e->spec_depth;
    g_cost0 = e->lane_cost[PULSAR_LANE_SPEC];
    /* M0 */
    const bool block = pulsar_engine_drafter(e) == PULSAR_DRAFTER_BLOCK;
    CHECK(pulsar_engine_has_spec_rounds(e) && g_base &&
          (block ? g_base == &PULSAR_BLOCK_DRAFTER_OPS && e->block_drafter
                 : pulsar_engine_drafter(e) == PULSAR_DRAFTER_MTP && g_base->absorb_banked == pulsar_mtp_absorb_banked &&
                       g_base->draft == pulsar_mtp_draft),
          "M0 the engine's drafter is a lane drafter: the %s (%s: depth %u, %u max, tau %.2f, adaptive %s)",
          block ? "block drafter" : "chained-MTP drafter", g_base ? g_base->name : "none", g_base ? g_base->depth : 0u,
          g_base ? g_base->depth_max : 0u, g_base ? (double)g_base->tau : 0.0, g_base && g_base->adapt ? "on" : "off");
    if (!g_base) return 1;
    std::vector<pulsar_tokens> P = prompts(e);
    CHECK(P.size() == 4, "M0 four prompts (%zu)", P.size());
    if (P.size() != 4) return 1;
    {
        pulsar_engine_set_bank_pool(1);
        pulsar_session *s = NULL;
        char err[256] = "";
        const bool made = pulsar_session_create(&s, e, 4096) == 0 && s;
        const bool synced = made && pulsar_session_sync(s, &P[0], err, sizeof(err)) == 0;
        if (block) {
            CHECK(synced && s->driver->block && !s->driver->mtp,
                  "M0 a session builds the block drafter's lane (no MTP lane) and the prefill absorbs its context: %s", err);
        } else {
            const pulsar_mtp_lane *l = made && s->driver ? s->driver->mtp : NULL;
            CHECK(synced && l && l->modules >= 1 && l->pend_pos[0] == (uint32_t)P[0].len - 1u,
                  "M0 a session builds the lane (%u module(s)) and the prefill parks bank 0's row at %d (%d): %s",
                  l ? l->modules : 0u, P[0].len - 1, l ? (int)l->pend_pos[0] : -2, err);
        }
        if (s) pulsar_session_free(s);
    }
    /* M1, M2: plain references */
    std::vector<stream> plain(P.size()), plain_t(P.size());
    for (size_t i = 0; (on('1') || on('2') || on('4')) && i < P.size(); i++) {
        plain[i] = run(e, P[i], n, false, 0.0f, 0);
        plain_t[i] = on('2') ? run(e, P[i], n, false, 0.7f, 0x5eed + i) : plain[i];
        CHECK(plain[i].ok && plain_t[i].ok && !plain[i].out.empty(), "M1 prompt %zu (%d tokens): plain greedy %zu tokens, "
              "plain T0.7 %zu tokens", i, P[i].len, plain[i].out.size(), plain_t[i].out.size());
    }
    /* rule 5: a spec stream's trunk steps are its verifies (the module steps are not the driver's), so the segment
     * cache's movement across M1's spec streams is the graphed verify lane's (L296 fullspec) */
    uint64_t rep0 = 0, cap0 = 0;
    pulsar_gpu_seg_counts(&rep0, &cap0);
    for (uint32_t depth = 0; on('1') && depth <= 5; depth++) {
        if (depth == 1) {
            uint64_t rep1 = 0, cap1 = 0;
            pulsar_gpu_seg_counts(&rep1, &cap1);
            printf("  info  M1 the verify steps' graphs: %llu replayed, %llu captured (the drafter's numbers)\n",
                   (unsigned long long)(rep1 - rep0), (unsigned long long)(cap1 - cap0));
        }
        drafter_at_depth(e, depth);
        uint64_t rounds = 0, drafts = 0, acc = 0, emitted = 0;
        for (size_t i = 0; i < P.size(); i++) {
            const stream sp = run(e, P[i], n, true, 0.0f, 0);
            const int d = first_diff(sp.out, plain[i].out);
            CHECK(sp.ok && d == -1, "M1 depth %s prompt %zu: greedy spec == plain, %zu tokens (first difference at %d; "
                  "%llu rounds, %llu drafts, %llu accepted)", depth ? std::to_string(depth).c_str() : "(drafter's)", i,
                  sp.out.size(), d, (unsigned long long)sp.rounds, (unsigned long long)sp.drafts,
                  (unsigned long long)sp.accepted);
            rounds += sp.rounds;
            drafts += sp.drafts;
            acc += sp.accepted;
            emitted += sp.out.size();
        }
        CHECK(drafts > 0 && (depth == 0 || (double)drafts >= 0.95 * depth * (double)rounds),
              "M1 depth %s drafted: %llu rounds with drafts, %llu drafts, %llu accepted (%.1f%%), %.3f "
              "tokens a drafted round", depth ? std::to_string(depth).c_str() : "(drafter's)",
              (unsigned long long)rounds, (unsigned long long)drafts, (unsigned long long)acc,
              drafts ? 100.0 * (double)acc / (double)drafts : 0.0, rounds ? 1.0 + (double)acc / (double)rounds : 0.0);
        (void)emitted;
    }
    for (uint32_t depth : {3u, 0u}) {
        drafter_at_depth(e, depth);
        for (size_t i = 0; on('2') && i < P.size(); i++) {
            const stream sp = run(e, P[i], n, true, 0.7f, 0x5eed + i);
            const int d = first_diff(sp.out, plain_t[i].out);
            CHECK(sp.ok && d == -1, "M2 depth %s prompt %zu: seeded T0.7 spec == plain, %zu tokens (first difference at "
                  "%d; %llu drafts, %llu accepted)", depth ? std::to_string(depth).c_str() : "(drafter's)", i,
                  sp.out.size(), d, (unsigned long long)sp.drafts, (unsigned long long)sp.accepted);
        }
    }
    /* M3 the forced-reject stress */
    drafter_at_depth(e, 0);
    for (size_t i = 0; on('3') && i < 2; i++) {
        pulsar_session *tw = NULL, *sp_s = NULL;
        const stream twin = run(e, P[i], n, false, 0.0f, 0, NULL, &tw);
        const stream sp = run(e, P[i], n, true, 0.0f, 0, &twin.out, &sp_s);
        const int d = first_diff(sp.out, twin.out);
        CHECK(sp.ok && twin.ok && d == -1, "M3 prompt %zu: %zu tokens through %llu forced rounds (%llu drafts, %llu "
              "accepted) == plain (first difference at %d)", i, sp.out.size(), (unsigned long long)sp.rounds,
              (unsigned long long)sp.drafts, (unsigned long long)sp.accepted, d);
        if (tw && sp_s && sp.ok && twin.ok && d == -1) {
            const pulsar_state_plan *p = sp_s->states->plan;
            uint32_t len = (uint32_t)P[i].len + (uint32_t)sp.out.size();
            /* a stream that ends on a stop token: the plain step emits it without feeding it, a round that verified
             * it as a kept draft has its row -- the twin feeds it too, so both stand at the same frontier (a full
             * model's stream can stop inside `tokens`; the mini's never did) */
            const bool stopped = !sp.out.empty() && pulsar_token_is_stop(e, sp.out.back());
            if (stopped && tw->states->bank_pos[0] == len - 1u && sp_s->states->bank_pos[0] == len) {
                char err[256] = "";
                CHECK(pulsar_session_eval(tw, sp.out.back(), err, sizeof(err)) == 0,
                      "M3 prompt %zu: the twin feeds the stop token the spec round verified: %s", i, err);
            }
            if (stopped && tw->states->bank_pos[0] == len - 1u && sp_s->states->bank_pos[0] == len - 1u)
                len--;   /* neither fed it (the round's own token, not a kept draft) */
            CHECK(sp_s->states->bank_pos[0] == len && tw->states->bank_pos[0] == len,
                  "M3 prompt %zu: both at %u tokens (%u, %u)", i, len, sp_s->states->bank_pos[0], tw->states->bank_pos[0]);
            size_t nb = 0, nd = 0;
            const std::string tdiff = state_diff(bank_state(sp_s, 0, 0, p->n_trunk, len), bank_state(tw, 0, 0, p->n_trunk, len), &nb);
            CHECK(tdiff.empty(), "M3 prompt %zu: the trunk's state after the rollbacks == the plain twin's, byte for byte "
                  "(%zu bytes: every slot's recurrent / conv state and pool rows%s%s)", i, nb,
                  tdiff.empty() ? "" : "; differs: ", tdiff.c_str());
            const std::string ddiff = state_diff(bank_state(sp_s, 0, p->n_trunk, p->n_slots, len - 1u),
                                                 bank_state(tw, 0, p->n_trunk, p->n_slots, len - 1u), &nd);
            if (block)   /* L298: the context rings and the anchor -- a rejected row never enters them, every context
                          * row is the decode arms' (M-independent) bytes: the plain twin's exactly */
                CHECK(ddiff.empty(), "M3 prompt %zu: the block drafter's context (its K|V rings and anchor) == the plain "
                      "twin's, byte for byte (%zu bytes%s%s)", i, nd, ddiff.empty() ? "" : "; differs: ", ddiff.c_str());
            else if (sp_s->driver->mtp->ops->row_exact)
                CHECK(ddiff.empty(), "M3 prompt %zu: the drafter's slot (its pool rows and the parked row) == the plain "
                      "twin's, byte for byte (%zu bytes%s%s)", i, nd, ddiff.empty() ? "" : "; differs: ", ddiff.c_str());
            else   /* the family declares its module rows row-count dependent (pulsar_mtp_ops::row_exact): graded */
                printf("  info  M3 prompt %zu: the drafter's slot vs the plain twin's (%zu bytes): %s (the module is not "
                       "row-exact by declaration)\n", i, nd, ddiff.empty() ? "byte-identical" : ddiff.c_str());
            const std::vector<std::vector<float>> ga = decode4(sp_s, W), gb = decode4(tw, W);
            CHECK(ga.size() == 4 && ga == gb, "M3 prompt %zu: four decode steps after == the twin's, byte for byte", i);
        }
        if (tw) pulsar_session_free(tw);
        if (sp_s) pulsar_session_free(sp_s);
    }
    /* M4 the batched lane: four banks, one shared verify a tick */
    drafter_at_depth(e, 0);
    if (on('4')) batched_lane(e, P, n, W, plain, "M4", 0.0f, 0u, 0);
    /* M5 the batched lane at seeded T0.7, each position's mode flipping inside the rounds (L296 toolsample) */
    if (on('5')) {
        const int RUN = 3;
        std::vector<stream> plain_m(P.size());
        for (size_t i = 0; i < P.size(); i++) {
            plain_m[i] = run(e, P[i], n, false, 0.7f, 0x70015ULL + i, NULL, NULL, RUN);
            CHECK(plain_m[i].ok && !plain_m[i].out.empty(), "M5 prompt %zu: plain T0.7 with every other run of %d "
                  "positions greedy, %zu tokens", i, RUN, plain_m[i].out.size());
        }
        batched_lane(e, P, n, W, plain_m, "M5", 0.7f, 0x70015ULL, RUN);
    }
    drafter_at_depth(e, 0);
    e->lane_cost[PULSAR_LANE_SPEC] = g_cost0;
    for (auto &t : P) pulsar_tokens_free(&t);
    return 0;
}

static std::vector<int32_t> read_ids(const char *path) {
    std::vector<int32_t> v;
    FILE *fp = fopen(path, "rb");
    if (!fp) return v;
    int32_t x;
    while (fread(&x, sizeof(x), 1, fp) == 1) v.push_back(x);
    fclose(fp);
    return v;
}

static bool write_rows(const std::string &path, const pulsar_mtp_probe &pr, uint32_t width, uint32_t positions) {
    /* rows in position order (each position appears once) */
    std::vector<int> at(positions, -1);
    for (uint32_t r = 0; r < pr.n; r++)
        if (pr.pos[r] >= 0 && (uint32_t)pr.pos[r] < positions) at[(uint32_t)pr.pos[r]] = (int)r;
    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp) return false;
    bool ok = true;
    for (uint32_t p = 0; ok && p < positions; p++)
        ok = at[p] >= 0 && fwrite(pr.logits + (size_t)at[p] * width, sizeof(float), width, fp) == width;
    fclose(fp);
    return ok;
}

static int probe(pulsar_engine *e, const char *ids_path, const char *out_dir, int positions) {
    const std::vector<int32_t> ids = read_ids(ids_path);
    const int T = (int)ids.size();
    if (positions <= 0 || positions > T - 1) positions = T - 1;
    CHECK(T > 32, "probe: %d reference ids from %s", T, ids_path);
    if (T <= 32) return 1;
    const uint32_t V = (uint32_t)pulsar_engine_logits_width(e);
    pulsar_mtp_probe pr{};
    pr.cap = (uint32_t)T;
    pr.logits = (float *)malloc((size_t)pr.cap * V * sizeof(float));
    pr.pos = (int32_t *)malloc((size_t)pr.cap * sizeof(int32_t));
    char err[256] = "";
    for (int path = 0; path < 2 && pr.logits && pr.pos; path++) {
        pulsar_engine_set_bank_pool(1);
        pulsar_session *s = NULL;
        bool ok = pulsar_session_create(&s, e, 4096) == 0 && s && s->driver && s->driver->mtp;
        pr.n = 0;
        if (ok) s->driver->mtp->probe = &pr;
        if (ok && path == 0) {   /* DECODE: one row a step */
            pulsar_tokens t = toks(ids, 0, 1);
            ok = pulsar_session_sync(s, &t, err, sizeof(err)) == 0;
            pulsar_tokens_free(&t);
            for (int i = 1; ok && i <= positions; i++) ok = pulsar_session_eval(s, ids[(size_t)i], err, sizeof(err)) == 0;
        }
        for (int c = 31; ok && path == 1; c += 31) {   /* PROMPT: lockstep steps of 31 rows */
            const int end = c < positions + 1 ? c : positions + 1;
            pulsar_tokens t = toks(ids, 0, (size_t)end);
            ok = pulsar_session_sync(s, &t, err, sizeof(err)) == 0;
            pulsar_tokens_free(&t);
            if (end == positions + 1) break;
        }
        if (s && s->driver && s->driver->mtp) s->driver->mtp->probe = NULL;
        const std::string name = std::string(out_dir) + (path == 0 ? "/mtp_decode.f32" : "/mtp_prefill.f32");
        ok = ok && write_rows(name, pr, V, (uint32_t)positions);
        CHECK(ok, "probe %s: %u module rows headed, %d positions -> %s: %s", path == 0 ? "DECODE" : "PROMPT", pr.n,
              positions, name.c_str(), err);
        if (s) pulsar_session_free(s);
    }
    free(pr.logits);
    free(pr.pos);
    return 0;
}

int GATE_ENTRY(int argc, char **argv) {
    n_fail = 0;
    const bool is_probe = argc > 2 && !strcmp(argv[2], "--probe");
    if (argc < 2 || (is_probe && argc < 5)) {
        fprintf(stderr, "usage: %s <model> [tokens] [sections, e.g. 134] | <model> --probe <ids.bin> <out_dir> "
                        "[positions]\n", argv[0]);
        return 2;
    }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "mtp-spec: %s did not open\n", argv[1]); return 2; }
    if (is_probe) (void)probe(e, argv[3], argv[4], argc > 5 ? atoi(argv[5]) : 0);
    else (void)exact(e, argc > 2 ? atoi(argv[2]) : 256, argc > 3 ? argv[3] : NULL);
    gate_engine_close(e);
    printf(n_fail ? "MTP-SPEC GATE FAIL (%d)\n" : "MTP-SPEC GATE PASS\n", n_fail);
    return n_fail != 0;
}

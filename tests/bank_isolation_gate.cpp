/* L357: bank isolation on the batched decode lane, for every family on the core step driver (Qwen first) -- the
 * intent of the DeepSeek-only tests/multiseq_frontier_gate.cpp, through the public bank and step API, graded on the
 * declared state (state_kinds.h) instead of DeepSeek's compressor caches.
 *
 *   ./tests/bank_isolation_gate <model> [steps]
 *
 * One session, three banks: each prefills its own stretch of the story prompt, of its own length, through the public
 * bank API (restore, sync, save).  Bank 2 is then INSTALLED (the live bank: a step that resolved a row's bank through
 * the live binding instead of the row's own id would land there) and `steps` (default 8) batched decode steps
 * (pulsar_session_decode_mixed, one row a bank, every token teacher-forced from the story) advance banks 0 and 1.
 *   I1  the idle bank: every declared state byte of bank 2 -- each slot's FIXED / RING / STAGED bytes and each pool
 *       WHOLE, the rows past its frontier too, where a batchmate's position would land a misrouted write -- is
 *       byte-identical before and after the steps (checked in every run below)
 *   I2  batch order: the same steps with the rows in the order (bank 1, bank 0), in a fresh session -- each bank's
 *       logits at every step and its whole declared state after the steps are byte-identical to the (0, 1) order's
 *   I3  a batchmate's position: bank 0 prefilled MOVE tokens longer (its rows at other positions; the step keeps its
 *       width), bank 1 as before -- bank 1's logits at every step and its whole state after are byte-identical to the
 *       first run's: a row's position reaches only its own row
 * DeepSeek's compressor-ratio scenarios (a ratio group closing on one bank only) left with its frontier gate (L357); this
 * gate reads the declared state, so it refuses a family without one by name.  A runner gate (tests/gates_runner.cpp
 * `bank-isolation-gate`). */
#include "pulsar.h"
#include "gate_entry.h"
#include "gate_fixture.h"   /* gate_bank_state: the declared state */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

static int n_fail = 0;
#define CHECK(c, ...) GATE_CHECK(n_fail, c, __VA_ARGS__)

#define N_BANKS   3u
#define IDLE_BANK 2u
#define MOVE      37   /* I3: bank 0's prompt this much longer */

/* each bank's stretch of the story and its prompt length */
static const int bank_off[N_BANKS] = {0, 1200, 2400};
static const int bank_len[N_BANKS] = {300, 170, 230};

/** One run of the script: the stepped banks' logits and every bank's whole state after the steps. */
struct iso_run {
    bool ok = false;
    std::vector<float> lg[2];                         ///< banks 0, 1: [steps][W]
    std::vector<gate_state_piece> idle_before;        ///< the idle bank's whole state before the steps
    std::vector<gate_state_piece> after[N_BANKS];     ///< every bank's whole state after the steps
};

/* Prefill the three banks (bank 0 `len0` tokens), install the idle bank, and step banks 0 and 1 `steps` times in the
 * row order (0, 1), or (1, 0) when `swap`. */
static iso_run run_script(pulsar_engine *e, const pulsar_tokens *story, int len0, bool swap, int steps, int W,
                          const char *what) {
    iso_run r;
    pulsar_engine_set_bank_pool(N_BANKS);
    pulsar_session *s = NULL;
    char err[256] = "";
    if (pulsar_session_create(&s, e, 4096) != 0 || !s) {
        fprintf(stderr, "%s: session create failed\n", what);
        return r;
    }
    const int len[N_BANKS] = {len0, bank_len[1], bank_len[2]};
    bool ok = gate_bank_pool_fits(s, (int)N_BANKS);
    for (uint32_t b = 0; ok && b < N_BANKS; b++) ok = gate_bank_sync(s, b, story->v + bank_off[b], len[b], NULL, what);
    ok = ok && pulsar_session_bank_state_restore(s, IDLE_BANK);
    const uint32_t slots = gate_state_slots(s);
    if (ok && slots == 0) {
        fprintf(stderr, "%s: %s has no declared state (state_kinds.h) -- this gate grades it\n", what,
                pulsar_engine_family_name(e));
        ok = false;
    }
    if (ok) r.idle_before = gate_bank_state(s, IDLE_BANK, 0, slots, GATE_STATE_WHOLE);
    for (int b = 0; b < 2; b++) r.lg[b].resize((size_t)steps * W);
    std::vector<float> out((size_t)2 * W);
    for (int t = 0; ok && t < steps; t++) {
        pulsar_multiseq_req rq[2];
        for (int k = 0; k < 2; k++) {
            const uint32_t b = (uint32_t)(swap ? 1 - k : k);
            rq[k] = {b, TokPos{len[b] + t}, story->v[bank_off[b] + len[b] + t]};
        }
        uint32_t n_out = 0;
        err[0] = '\0';
        const int rc = pulsar_session_decode_mixed(s, rq, 2, out.data(), 2 * W, &n_out, 0, err, sizeof(err));
        if (rc != 0 || n_out != 2) {
            fprintf(stderr, "%s: step %d (rc %d, %u rows): %s\n", what, t, rc, n_out, err);
            ok = false;
            break;
        }
        for (int k = 0; k < 2; k++)   /* logits row k is the step's row k */
            memcpy(r.lg[rq[k].bank].data() + (size_t)t * W, out.data() + (size_t)k * W, (size_t)W * sizeof(float));
    }
    for (uint32_t b = 0; ok && b < N_BANKS; b++) r.after[b] = gate_bank_state(s, b, 0, slots, GATE_STATE_WHOLE);
    r.ok = ok;
    pulsar_session_free(s);
    return r;
}

/* the first step whose row differs, or -1 when every step's row is byte-identical */
static int first_step_diff(const std::vector<float> &a, const std::vector<float> &b, int steps, int W) {
    if (a.size() != b.size() || a.size() != (size_t)steps * W) return 0;
    for (int t = 0; t < steps; t++)
        if (gate_first_diff(a.data() + (size_t)t * W, b.data() + (size_t)t * W, W) >= 0) return t;
    return -1;
}

/* I1 on one run: the idle bank's whole state unchanged by the steps */
static void check_idle(const iso_run &r, const char *what) {
    size_t n = 0;
    const std::string d = gate_state_diff(r.idle_before, r.after[IDLE_BANK], &n);
    CHECK(r.ok && d.empty(), "I1 %s: idle bank %u (installed) is byte-identical after the steps (%zu bytes%s%s)", what,
          IDLE_BANK, n, d.empty() ? "" : "; differs: ", d.c_str());
}

int GATE_ENTRY(int argc, char **argv) {
    n_fail = 0;
    if (argc < 2) { fprintf(stderr, "usage: %s <model> [steps]\n", argv[0]); return 2; }
    const int steps = argc > 2 ? atoi(argv[2]) : 8;
    if (steps < 1) { fprintf(stderr, "bank-isolation: steps must be >= 1\n"); return 2; }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "bank-isolation: %s did not open\n", argv[1]); return 2; }
    const int W = pulsar_engine_logits_width(e);
    pulsar_tokens story;
    int need = 0;
    for (uint32_t b = 0; b < N_BANKS; b++)
        need = need > bank_off[b] + bank_len[b] + MOVE + steps ? need : bank_off[b] + bank_len[b] + MOVE + steps;
    if (!gate_load_story(e, &story, need)) { gate_engine_close(e); return 2; }
    printf("bank-isolation: %s, banks 0/1/2 at %d/%d/%d tokens, bank %u idle and installed, %d batched steps\n",
           pulsar_engine_family_name(e), bank_len[0], bank_len[1], bank_len[2], IDLE_BANK, steps);

    const iso_run a = run_script(e, &story, bank_len[0], false, steps, W, "order (0, 1)");
    CHECK(a.ok, "order (0, 1): the three banks prefilled and stepped");
    check_idle(a, "order (0, 1)");

    /* I2 */
    const iso_run sw = run_script(e, &story, bank_len[0], true, steps, W, "order (1, 0)");
    CHECK(sw.ok, "order (1, 0): the same banks stepped in the swapped row order");
    check_idle(sw, "order (1, 0)");
    for (uint32_t b = 0; b < 2; b++) {
        const int t = a.ok && sw.ok ? first_step_diff(a.lg[b], sw.lg[b], steps, W) : 0;
        CHECK(a.ok && sw.ok && t < 0, "I2 bank %u's logits at every step are byte-identical in either row order "
              "(first differing step %d)", b, t);
        size_t n = 0;
        const std::string d = gate_state_diff(a.after[b], sw.after[b], &n);
        CHECK(a.ok && sw.ok && d.empty(), "I2 bank %u's state after the steps is byte-identical in either row order "
              "(%zu bytes%s%s)", b, n, d.empty() ? "" : "; differs: ", d.c_str());
    }

    /* I3 */
    const iso_run mv = run_script(e, &story, bank_len[0] + MOVE, false, steps, W, "bank 0 moved");
    CHECK(mv.ok, "bank 0 moved: bank 0 prefilled %d tokens (%d more), its rows %d positions later", bank_len[0] + MOVE,
          MOVE, MOVE);
    check_idle(mv, "bank 0 moved");
    {
        const int t = a.ok && mv.ok ? first_step_diff(a.lg[1], mv.lg[1], steps, W) : 0;
        CHECK(a.ok && mv.ok && t < 0, "I3 bank 1's logits at every step are byte-identical with its batchmate moved "
              "(first differing step %d)", t);
        size_t n = 0;
        const std::string d = gate_state_diff(a.after[1], mv.after[1], &n);
        CHECK(a.ok && mv.ok && d.empty(), "I3 bank 1's state after the steps is byte-identical with its batchmate "
              "moved (%zu bytes%s%s)", n, d.empty() ? "" : "; differs: ", d.c_str());
    }

    pulsar_tokens_free(&story);
    gate_engine_close(e);
    printf(n_fail ? "BANK-ISOLATION GATE FAIL (%d)\n" : "BANK-ISOLATION GATE PASS\n", n_fail);
    return n_fail != 0;
}

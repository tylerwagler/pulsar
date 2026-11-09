/* L306 GATE: a refused block-drafter pass fails the single-bank lane's request by name -- never a plain round.
 *
 * The single lane (pulsar_session_generate_speculative: CLI, eval, the server's warmup) drafts the next chain at the
 * end of each round.  Before L306 a refused draft printed "(the round takes a plain step next)" and the stream went on
 * without speculation (ENGINEERING-RULES 1: a fallback).  Now the draft op returns PULSAR_DRAFT_REFUSED and the round
 * refuses, as the batched lane's draft_batch already did.
 *
 * The refusal is forced through the drafter's own path: for one call the engine's draft op is a wrapper that detaches
 * the session's block lane around the block drafter's own draft (draft_chains then refuses: "speculative decoding
 * needs the block drafter's lane") -- only the draft: the round's absorb before it runs on the lane.  PASS: a
 * speculative stream runs before; the forced call returns < 0 with an err naming the drafter's refusal.
 *   [PULSAR_GATE_DRAFTER=dflash2] ./tests/draft_refusal_gate MODEL   (a container whose drafter is a block drafter) */
#include "pulsar.h"
#include "gate_entry.h"
#include "gate_util.h"
#include "pulsar_engine_internal.h"
#include "step_driver.h"

#include <stdio.h>
#include <string.h>

static const pulsar_drafter_ops *g_block;   /* the engine's block drafter */

/* the block drafter's draft with the session's lane detached: its own refusal path */
static uint32_t refusing_draft(pulsar_session *s, int next_base, bool features_ready, float temperature, int top_k,
                               float top_p, float min_p, uint64_t *rng) {
    pulsar_block_lane *lane = s->driver->block;
    s->driver->block = NULL;
    const uint32_t k = g_block->draft(s, next_base, features_ready, temperature, top_k, top_p, min_p, rng);
    s->driver->block = lane;
    return k;
}

int GATE_ENTRY(int argc, char **argv) {
    int fails = 0;
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL\n", argv[0]);
        return 2;
    }
    pulsar_engine_options o;
    memset(&o, 0, sizeof o);
    o.model_path = argv[1];
    o.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &o) != 0) {
        fprintf(stderr, "engine open failed\n");
        return 1;
    }
    pulsar_engine_set_bank_pool(1);
    pulsar_session *s = NULL;
    pulsar_tokens story;
    memset(&story, 0, sizeof story);
    char err[256] = "";
    bool ok = pulsar_engine_drafter(e) == PULSAR_DRAFTER_BLOCK && gate_load_story(e, &story, 400) &&
              pulsar_session_create(&s, e, 4096) == 0 && s && s->driver && s->driver->block;
    GATE_CHECK(fails, ok, "a session on a block drafter's container, its lane built");
    if (story.len > 400) story.len = 400;   /* the story's first 400 tokens: the prompt */
    if (ok) ok = pulsar_session_sync(s, &story, err, sizeof err) == 0;
    uint64_t rng = 0;
    int toks[17], emitted = 0;
    for (int i = 0; ok && i < 4; i++) {
        s->spec.spec_quenched = false;
        const int n = pulsar_session_generate_speculative(s, 0.0f, 0, 1.0f, 0.0f, &rng, 16, toks, 17, err, sizeof err);
        ok = n > 0;
        emitted += n > 0 ? n : 0;
    }
    GATE_CHECK(fails, ok && emitted > 4, "the speculative stream runs before the refusal (%d tokens in 4 rounds): %s",
               emitted, err);
    if (ok) {
        g_block = e->drafter_ops;
        pulsar_drafter_ops forced = *g_block;
        forced.draft = refusing_draft;
        e->drafter_ops = &forced;
        err[0] = '\0';
        s->spec.spec_quenched = false;
        const int n = pulsar_session_generate_speculative(s, 0.0f, 0, 1.0f, 0.0f, &rng, 16, toks, 17, err, sizeof err);
        e->drafter_ops = g_block;
        GATE_CHECK(fails, n < 0 && strstr(err, "drafter refused") != NULL,
                   "a refused draft pass fails the request by name, no plain round (returned %d: \"%s\")", n, err);
    }
    if (s) pulsar_session_free(s);
    pulsar_tokens_free(&story);
    gate_engine_close(e);
    printf("DRAFT-REFUSAL GATE: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

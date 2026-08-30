/* L120 rewind-frontier gate — position-truth of the compressor frontiers
 * across pulsar_session_rewind.
 *
 * The bug this pins: pulsar_session::rewind trimmed tokens but never
 * reconciled layer_n_comp/layer_n_index_comp, so any rewind crossing a
 * (pos+1)%ratio emit boundary left the pair one row high.  Latent since
 * 6de76e3 (classic re-assigned the counters absolutely before anything
 * validated); first asserted by the unified lane's position-true admission
 * check, first fired by production anthropic live continuations (the one
 * serving path with no counter-rebuilding re-route between the rewind and
 * the check).  See pulsar-notes rows/L120.md.
 *
 * Shape (deterministic — no residue luck, unlike the serving repro):
 *   1. prefill to a frontier past several ratio-4 boundaries;
 *   2. rewind one token at a time across every mod-4 residue, asserting
 *      after each that EVERY compressing layer's counters equal pos/ratio
 *      (ratio-128 layers get clamped across their own boundary too);
 *   3. the incident anatomy in miniature: after a boundary-crossing rewind,
 *      sync a continuation (the incremental prefill path — the one that
 *      preserved the skew in production) and re-assert at the new frontier;
 *   4. VALUE leg (L120 value-half): a ghost rewind with a DIVERGENT ghost
 *      branch, then byte-compare the re-emitted attn+index comp rows of
 *      every ratio-4 layer against a never-ghosted control.  The scenario
 *      deliberately avoids crossing a 128-emit boundary so it asserts the
 *      ratio-4 claim in isolation;
 *   5. RATIO-128 leg (L124): the same divergent-ghost shape SHIFTED so the
 *      ghost span crosses the first 128-emit boundary (ghosts 126..129,
 *      rewind to 126).  Ghost stores past the boundary alias the slots of
 *      committed positions g-128, and the re-emit of comp row 0 fires at
 *      re-decode of position 127 -- BEFORE re-decode reaches the aliased
 *      owners -- pooling wrong-POSITION values.  The undo log restores the
 *      aliased slots byte-exactly on rewind; this leg byte-compares the
 *      re-emitted ratio-128 comp row 0 across every ratio-128 layer against
 *      a never-ghosted control.
 *
 * MODEL-DEPENDENT, GPU-resident.  Run under the memory discipline.  NOT part
 * of `make test`.
 *
 * usage: ./tests/rewind_frontier_gate MODEL
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "pulsar_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "REWIND FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); g_fail = 1; } } while (0)

static char *read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, fp) != (size_t)n) { fclose(fp); free(buf); return NULL; }
    fclose(fp);
    buf[n] = '\0';
    if (len_out) *len_out = (size_t)n;
    return buf;
}

/* Every compressing layer's frontier pair must equal pos/ratio exactly —
 * position-truth, the same predicate the multiseq admission check enforces. */
static void check_frontiers(pulsar_session *s, int pos, const char *what) {
    pulsar_gpu_graph *g = &s->graph;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t want = (uint32_t)pos / ratio;
        CHECK(g->layer_n_comp[il] == want,
              "%s: layer %u n_comp %u want %u (pos %d ratio %u)",
              what, il, g->layer_n_comp[il], want, pos, ratio);
        if (ratio == 4) {
            CHECK(g->layer_n_index_comp[il] == want,
                  "%s: layer %u n_index_comp %u want %u (pos %d)",
                  what, il, g->layer_n_index_comp[il], want, pos);
        }
    }
}

static bool sync_prefix(pulsar_session *s, pulsar_tokens *toks, int len) {
    pulsar_tokens p;
    memset(&p, 0, sizeof(p));
    p.v = toks->v;
    p.len = p.cap = len;
    char err[256];
    const int rc = pulsar_session_sync(s, &p, err, sizeof(err));
    if (rc != 0) fprintf(stderr, "sync to %d failed: %s\n", len, err);
    return rc == 0;
}

/* Extend the session by exactly one token via sync — pos0 is unaligned so
 * this takes the per-row store/shift path, never the aligned prefill (whose
 * ratio-4 window refresh would heal the contamination this leg exists to
 * detect). */
static bool extend_one(pulsar_session *s, pulsar_tokens *toks, int upto) {
    pulsar_tokens p;
    memset(&p, 0, sizeof(p));
    p.v = toks->v;
    p.len = p.cap = upto;
    char err[256];
    const int rc = pulsar_session_sync(s, &p, err, sizeof(err));
    if (rc != 0) fprintf(stderr, "extend to %d failed: %s\n", upto, err);
    return rc == 0;
}

/* FNV-1a over attn comp rows 34..36 and (ratio 4) index comp rows 34..36
 * of every compressing layer, read raw D2H off the classic single-session
 * caches (row 34 = pre-rewind sanity, 35 = the re-emitted group, 36 = the
 * first post-heal group). */
static uint64_t comp_rows_hash(pulsar_session *s) {
    pulsar_gpu_graph *g = &s->graph;
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = PULSAR_ENGINE_IDXFP4_ROWBYTES;
    uint64_t h = 1469598103934665603ull;
    uint8_t buf[8192];
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio != 4) continue;
        for (uint32_t row = 34; row <= 36; row++) {
            if (row >= g->layer_n_comp[il]) continue;
            if (pulsar_gpu_tensor_read(g->layer_attn_comp_cache[il],
                                      (uint64_t)row * attn_row, buf, attn_row) == 0)
                return 0;
            for (uint64_t i = 0; i < attn_row; i++) { h ^= buf[i]; h *= 1099511628211ull; }
            if (row >= g->layer_n_index_comp[il]) continue;
            if (pulsar_gpu_tensor_read(g->layer_index_comp_cache[il],
                                      (uint64_t)row * idx_row, buf, idx_row) == 0)
                return 0;
            for (uint64_t i = 0; i < idx_row; i++) { h ^= buf[i]; h *= 1099511628211ull; }
        }
    }
    return h;
}

/* One value-leg session: prefill 0..126; if ghost_branch is set, extend
 * one-at-a-time through the DIVERGENT tokens to 130 and rewind to 126;
 * then extend one-at-a-time through the true tokens to 134; hash the
 * re-emitted comp rows. */
static uint64_t value_leg_hash(pulsar_engine *e, pulsar_tokens *toks,
                               const pulsar_tokens *ghost_branch) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, e, 4096) != 0) { fprintf(stderr, "value leg: session create failed\n"); return 0; }
    uint64_t h = 0;
    pulsar_tokens work;
    memset(&work, 0, sizeof(work));
    work.v = (int *)malloc(sizeof(int) * 256);
    work.cap = 256;
    memcpy(work.v, toks->v, sizeof(int) * 150);
    work.len = 150;
    bool ok = true;
    {
        /* Base prefill stops at 112 and the last 14 positions are decoded
         * as singles: the ring at the rewind point is then built by pure
         * store/shift evolution in BOTH sessions, the same mechanism the
         * replay reproduces.  (A prefill boundary inside the replay span
         * would compare store-path bytes against the aligned path's
         * refresh-seeded ring — a low-bit projection-path duality the
         * codebase documents separately; the replay's contract is
         * store-path value truth.) */
        pulsar_tokens p;
        memset(&p, 0, sizeof(p));
        p.v = work.v;
        p.len = p.cap = 128;
        char err[256];
        ok = pulsar_session_sync(s, &p, err, sizeof(err)) == 0;
        if (!ok) fprintf(stderr, "value leg: base prefill failed: %s\n", err);
    }
    for (int upto = 129; ok && upto <= 142; upto++)
        ok = extend_one(s, &work, upto);
    if (ok && ghost_branch) {
        int saved[4];
        memcpy(saved, work.v + 142, sizeof(saved));
        memcpy(work.v + 142, ghost_branch->v, sizeof(int) * 4);
        for (int upto = 143; ok && upto <= 146; upto++)
            ok = extend_one(s, &work, upto);
        if (ok) {
            pulsar_session_rewind(s, 142);
            ok = pulsar_session_pos(s) == 142;
            if (!ok) fprintf(stderr, "value leg: rewind landed at %d\n", pulsar_session_pos(s));
        }
        memcpy(work.v + 142, saved, sizeof(saved));
    }
    for (int upto = 143; ok && upto <= 150; upto++)
        ok = extend_one(s, &work, upto);
    if (ok) h = comp_rows_hash(s);
    free(work.v);
    pulsar_session_free(s);
    return h;
}

/* FNV-1a over ratio-128 comp row 0 of every ratio-128 layer -- the row the
 * L124 aliasing contaminates. */
static uint64_t r128_row0_hash(pulsar_session *s) {
    pulsar_gpu_graph *g = &s->graph;
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    uint64_t h = 1469598103934665603ull;
    uint8_t buf[8192];
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (pulsar_layer_compress_ratio(il) != 128u) continue;
        if (g->layer_n_comp[il] < 1u) continue;
        if (pulsar_gpu_tensor_read(g->layer_attn_comp_cache[il], 0, buf, attn_row) == 0)
            return 0;
        for (uint64_t i = 0; i < attn_row; i++) { h ^= buf[i]; h *= 1099511628211ull; }
    }
    return h;
}

/* One ratio-128-leg session: sync 0..112, singles to 126; if ghost_branch,
 * DIVERGENT singles 126..129 (crossing the b=128 emit boundary: ghosts 128
 * and 129 alias the slots of committed positions 0 and 1), rewind to 126;
 * then true singles to 134 -- the 128-emit at position 127 re-pools the
 * full window, aliased slots included.  Hash ratio-128 comp row 0. */
static uint64_t r128_leg_hash(pulsar_engine *e, pulsar_tokens *toks,
                              const pulsar_tokens *ghost_branch) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, e, 4096) != 0) { fprintf(stderr, "r128 leg: session create failed\n"); return 0; }
    uint64_t h = 0;
    pulsar_tokens work;
    memset(&work, 0, sizeof(work));
    work.v = (int *)malloc(sizeof(int) * 256);
    work.cap = 256;
    memcpy(work.v, toks->v, sizeof(int) * 200);
    work.len = 200;
    bool ok = true;
    {
        pulsar_tokens p;
        memset(&p, 0, sizeof(p));
        p.v = work.v;
        p.len = p.cap = 112;
        char err[256];
        ok = pulsar_session_sync(s, &p, err, sizeof(err)) == 0;
        if (!ok) fprintf(stderr, "r128 leg: base prefill failed: %s\n", err);
    }
    for (int upto = 113; ok && upto <= 126; upto++)
        ok = extend_one(s, &work, upto);
    if (ok && ghost_branch) {
        int saved[4];
        memcpy(saved, work.v + 126, sizeof(saved));
        memcpy(work.v + 126, ghost_branch->v, sizeof(int) * 4);
        for (int upto = 127; ok && upto <= 130; upto++)
            ok = extend_one(s, &work, upto);
        if (ok) {
            pulsar_session_rewind(s, 126);
            ok = pulsar_session_pos(s) == 126;
            if (!ok) fprintf(stderr, "r128 leg: rewind landed at %d\n", pulsar_session_pos(s));
        }
        memcpy(work.v + 126, saved, sizeof(saved));
    }
    for (int upto = 127; ok && upto <= 134; upto++)
        ok = extend_one(s, &work, upto);
    if (ok) h = r128_row0_hash(s);
    free(work.v);
    pulsar_session_free(s);
    return h;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }

    pulsar_engine *e = NULL;
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }

    size_t text_len = 0;
    char *text = read_file("tests/long_context_story_prompt.txt", &text_len);
    if (!text) { fprintf(stderr, "prompt file read failed\n"); return 1; }
    pulsar_tokens toks;
    memset(&toks, 0, sizeof(toks));
    pulsar_tokenize_text(e, text, &toks);
    free(text);
    if (toks.len < 300) { fprintf(stderr, "prompt too short\n"); return 1; }

    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, e, 4096) != 0) { fprintf(stderr, "session create failed\n"); return 1; }

    /* 1. Frontier past several ratio-4 boundaries AND one ratio-128 boundary,
     * on a boundary-interior position so the very first rewind crosses. */
    const int top = 130;
    if (!sync_prefix(s, &toks, top)) return 1;
    check_frontiers(s, top, "post-prefill baseline");

    /* 2. Walk the frontier down one token at a time: eight rewinds cover
     * every mod-4 residue twice, and 129->127 crosses the ratio-128
     * boundary.  Unfixed rewind leaves the counters at the old frontier on
     * every boundary cross. */
    for (int pos = top - 1; pos >= top - 8; pos--) {
        pulsar_session_rewind(s, pos);
        CHECK(pulsar_session_pos(s) == pos, "rewind: pos %d want %d",
              pulsar_session_pos(s), pos);
        check_frontiers(s, pos, "post-rewind");
    }

    /* 3. Incident anatomy: from the rewound frontier (122), continue with an
     * incremental prefill — the serving path that carried the stale counter
     * to the admission check — and re-assert position-truth at the new
     * frontier.  On the unfixed binary the walk above already failed, and
     * this leg fails too (stale base + incremental emits). */
    const int cont = 199;
    if (!sync_prefix(s, &toks, cont)) return 1;
    check_frontiers(s, cont, "post-continuation");

    /* One more boundary-crossing rewind after the continuation, then a
     * second continuation: rewind machinery must stay position-true under
     * repeated cycles, not just once from a fresh prefill. */
    pulsar_session_rewind(s, cont - 4);
    check_frontiers(s, cont - 4, "second rewind");
    if (!sync_prefix(s, &toks, 260)) return 1;
    check_frontiers(s, 260, "second continuation");

    pulsar_session_free(s);

    /* ---- 4. VALUE leg (L120 value-half): the comp rows a ghost rewind
     * re-emits must be byte-identical to a session that never saw the
     * ghosts.  Ghosts are decoded via single-token syncs (the per-row
     * store/shift path — aligned prefill would heal the window via its
     * refresh and mask the bug), with a DIVERGENT ghost branch (identical
     * tokens would re-store identical projections and hide it).
     * Control:  prefill 0..126, then true tokens one at a time to 134.
     * Victim:   prefill 0..126, DIVERGENT tokens one at a time to 130
     *           (ghost branch), rewind to 126, true tokens to 134.
     * Compare attn + index comp rows 30..32 across every ratio-4 layer:
     * row 30 is a pre-rewind sanity row, row 31 is the re-emitted group
     * [124..127] (the window-contamination target), row 32 is the first
     * post-heal group. */
    const uint64_t ctl = value_leg_hash(e, &toks, NULL);
    pulsar_tokens ghost_branch;
    memset(&ghost_branch, 0, sizeof(ghost_branch));
    ghost_branch.v = (int *)malloc(sizeof(int) * 4);
    ghost_branch.len = ghost_branch.cap = 4;
    for (int i = 0; i < 4; i++) ghost_branch.v[i] = toks.v[500 + i];
    const uint64_t vic = value_leg_hash(e, &toks, &ghost_branch);
    free(ghost_branch.v);
    CHECK(ctl != 0 && vic != 0, "value leg: a session failed (ctl=%llx vic=%llx)",
          (unsigned long long)ctl, (unsigned long long)vic);
    CHECK(ctl == vic,
          "value leg: ghost rewind contaminated re-emitted comp rows "
          "(ctl=%llx vic=%llx)", (unsigned long long)ctl, (unsigned long long)vic);

    /* ---- 5. RATIO-128 leg (L124): the crossing shape the value leg
     * deliberately avoided.  Divergent ghosts 126..129 cross the b=128
     * boundary; ghost stores at 128/129 alias committed slots 0/1; the
     * re-emit of ratio-128 comp row 0 at position 127 pools them.  With the
     * undo log the rewind restores the aliased slots byte-exactly. */
    const uint64_t rctl = r128_leg_hash(e, &toks, NULL);
    pulsar_tokens rghost;
    memset(&rghost, 0, sizeof(rghost));
    rghost.v = (int *)malloc(sizeof(int) * 4);
    rghost.len = rghost.cap = 4;
    for (int i = 0; i < 4; i++) rghost.v[i] = toks.v[600 + i];
    const uint64_t rvic = r128_leg_hash(e, &toks, &rghost);
    free(rghost.v);
    CHECK(rctl != 0 && rvic != 0, "r128 leg: a session failed (ctl=%llx vic=%llx)",
          (unsigned long long)rctl, (unsigned long long)rvic);
    CHECK(rctl == rvic,
          "r128 leg: boundary-crossing ghost rewind aliased committed slots "
          "into comp row 0 (ctl=%llx vic=%llx)",
          (unsigned long long)rctl, (unsigned long long)rvic);

    /* ---- 6. SERVED-SHAPE leg (2026-08-30): the shape production actually
     * takes, which legs 4 and 5 do NOT.
     *
     * Legs 4/5 build their ghosts through sync/classic decode. Those paths
     * DEPOSIT into the projection ring (gpu_prefill.cpp:1312 and the note at
     * :2818, both guarded `!mseq && !spec_comp_save_n`), so the rewind replay
     * is covered and restores values. The server never decodes that way:
     * gpu_graph_multiseq_step_begin sets batch_multiseq=true for EVERY
     * multiseq step, so no deposit happens and the replay's coverage test
     * fails by construction. Measured on a served workload 2026-08-30:
     * 0 replay TAKEN / 2 skipped, spans [18,22) and [39,43) against a rewind
     * needing 12..17 (plans/ONE-STATE-MODEL-STAGE0B.md).
     *
     * So this leg pins what IS live on that path and what production depends
     * on: the counter clamp. It deliberately does NOT assert value
     * restoration, because on this path there is none -- that is a documented
     * limit, not a bug, and legs 4/5 already pin the value claim for the
     * paths that do deposit.
     *
     * If someone later extends the ring to cover decode (option (b) in
     * plans/ONE-STATE-MODEL.md), this leg keeps passing and a NEW value
     * assertion belongs here beside it. */
    {
        pulsar_session *ms = NULL;
        const int ms_rc = pulsar_session_create(&ms, e, 4096);
        CHECK(ms_rc == 0 && ms != NULL, "served leg: session create failed (rc=%d)", ms_rc);
        if (ms_rc == 0 && ms) {
            char err[256];
            /* prefill to a frontier well past several ratio-4 boundaries */
            if (sync_prefix(ms, &toks, 600)) {
                /* decode through the SERVED entry: 1 row on bank 0. This sets
                 * batch_multiseq even with no pool allocated (bank 0 aliases
                 * the classic tensors), which is exactly the production shape. */
                static float *ms_logits = (float *)malloc(sizeof(float) * (size_t)PULSAR_N_VOCAB);
                bool ok = true;
                for (int k = 0; ok && k < 6; k++) {
                    pulsar_multiseq_req req;
                    req.bank  = 0u;
                    req.pos   = (int32_t)pulsar_session_pos(ms);
                    req.token = toks.v[600 + k];
                    uint32_t out_rows = 0;
                    const int rc = pulsar_session_decode_mixed(
                            ms, &req, 1u, ms_logits, (int)PULSAR_N_VOCAB,
                            &out_rows, 0u, err, sizeof(err));
                    if (rc != 0) {
                        fprintf(stderr, "served leg: decode_mixed rc=%d %s\n", rc, err);
                        ok = false;
                        break;
                    }
                    pulsar_session_note_committed_tokens(ms, &toks.v[600 + k], 1);
                }
                if (ok) {
                    /* ghost rewind across a ratio-4 boundary, exactly as the
                     * server does when emission stops mid accepted-batch */
                    const int target = pulsar_session_pos(ms) - 3;
                    pulsar_session_rewind(ms, target);
                    check_frontiers(ms, target, "served leg (mseq ghost rewind)");
                }
            }
            pulsar_session_free(ms);
        }
    }

    pulsar_engine_close(e);

    if (g_fail) { fprintf(stderr, "REWIND GATE: FAIL\n"); return 1; }
    printf("REWIND GATE: PASS\n");
    return 0;
}

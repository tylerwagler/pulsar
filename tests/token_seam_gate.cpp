/* L115 token-seam gate — sampled-vs-canonical boundary drift must not cost
 * the live KV.
 *
 * The bug this pins: generated text freezes SAMPLED token boundaries into
 * the live checkpoint; the client's echo of the same bytes re-tokenizes
 * CANONICALLY (live `))`+`**` vs echoed `))**`), the id-exact common-prefix
 * declared divergence at the earliest seam, and the conversation re-paid its
 * whole history as a cold prefill on every later turn (production: two
 * ~200k colds in one morning, rows/L115.md).
 *
 * Legs:
 *   1. WALK: pulsar_tokens_prefix_match crosses a synthetic seam and
 *      matches to the shared boundary on both sides (id-common stops at it).
 *   2. RESCUE: a session built with the sampled (split) boundaries, synced
 *      against the canonical (fused) render plus a suffix, must (a) succeed,
 *      (b) end at the stitched length, and (c) RETAIN the live split ids at
 *      the seam — retention proves the live KV survived (a rebuild would
 *      hold the canonical fused id instead).
 *   3. SHORTER ECHO (the production shape): live additionally carries a
 *      tail the echo does not — the client strips generated reasoning — so
 *      the echo both re-tokenizes AND ends before the live frontier.
 *      SCOPE, measured 2026-08-28: this leg is a BEHAVIOUR PIN, not a
 *      discriminator for the increment-2 sync generalization — it passes
 *      with the old seam-only rescue condition too, because its shared
 *      prefix contains a seam and that alone fires the old rescue.  It
 *      cannot be made discriminating at this level: with no seam, a
 *      rewind-and-stitch and a rebuild produce identical ids, identical
 *      length and identical KV, so they differ only in COST.  The change
 *      that altered production outcomes is the server resolver gate, and
 *      its witness is the e2e shorter-echo probe (in-place extension vs
 *      `live kv cache miss` + disk load), not this gate.
 *
 * The seam pair is discovered from the model's own vocabulary at runtime: a
 * multi-byte token whose text re-tokenizes as 2+ non-empty-text tokens with
 * identical bytes.  MODEL-DEPENDENT, GPU-resident.  NOT part of `make test`.
 *
 * usage: ./tests/token_seam_gate MODEL
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "pulsar_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "SEAM FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); g_fail = 1; } } while (0)

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

/* Do these tokens' texts concatenate to exactly (text, len), with every
 * piece non-empty (control tokens must not hide inside a seam)? */
static bool pieces_match_bytes(pulsar_engine *e, const pulsar_tokens *p,
                               const char *text, size_t len) {
    size_t off = 0;
    for (int i = 0; i < p->len; i++) {
        size_t pl = 0;
        const char *pt = pulsar_token_text(e, p->v[i], &pl);
        if (!pt || pl == 0) return false;
        if (off + pl > len || memcmp(text + off, pt, pl) != 0) return false;
        off += pl;
    }
    return off == len;
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
    pulsar_tokens canon;
    memset(&canon, 0, sizeof(canon));
    pulsar_tokenize_text(e, text, &canon);
    free(text);
    if (canon.len < 800) { fprintf(stderr, "prompt too short\n"); return 1; }

    /* ---- discover a seam pair: canon[k] whose text re-tokenizes split ---- */
    int k = -1;
    pulsar_tokens split;
    memset(&split, 0, sizeof(split));
    for (int i = 64; i < 190 && k < 0; i++) {
        size_t tl = 0;
        const char *tt = pulsar_token_text(e, canon.v[i], &tl);
        if (!tt || tl < 2 || tl > 63) continue;
        char first[64];
        memcpy(first, tt, 1);
        first[1] = '\0';
        char rest[64];
        memcpy(rest, tt + 1, tl - 1);
        rest[tl - 1] = '\0';
        pulsar_tokens pa, pb;
        memset(&pa, 0, sizeof(pa));
        memset(&pb, 0, sizeof(pb));
        pulsar_tokenize_text(e, first, &pa);
        pulsar_tokenize_text(e, rest, &pb);
        pulsar_tokens joined;
        memset(&joined, 0, sizeof(joined));
        joined.v = (int *)malloc(sizeof(int) * (size_t)(pa.len + pb.len));
        joined.cap = pa.len + pb.len;
        for (int m = 0; m < pa.len; m++) joined.v[joined.len++] = pa.v[m];
        for (int m = 0; m < pb.len; m++) joined.v[joined.len++] = pb.v[m];
        if (joined.len >= 2 && pieces_match_bytes(e, &joined, tt, tl) &&
            !(joined.len == 1 && joined.v[0] == canon.v[i])) {
            k = i;
            split = joined;    /* take ownership */
        } else {
            free(joined.v);
        }
        pulsar_tokens_free(&pa);
        pulsar_tokens_free(&pb);
    }
    CHECK(k >= 0, "no discoverable seam pair in vocab (scan window 64..190)");
    if (k < 0) { fprintf(stderr, "SEAM GATE: FAIL\n"); return 1; }

    /* live variant = canon with canon[k] replaced by its split pieces */
    const int live_len = canon.len - 1 + split.len;
    int *live = (int *)malloc(sizeof(int) * (size_t)live_len);
    memcpy(live, canon.v, sizeof(int) * (size_t)k);
    memcpy(live + k, split.v, sizeof(int) * (size_t)split.len);
    memcpy(live + k + split.len, canon.v + k + 1,
           sizeof(int) * (size_t)(canon.len - k - 1));

    /* ---- leg 1: the byte walk crosses the seam ---- */
    {
        pulsar_prefix_match m;
        pulsar_tokens_prefix_match(e, live, live_len, canon.v, canon.len, &m);
        const int a_n = m.live_cut, b_n = m.prompt_cut;
        CHECK(m.seamed, "walk did not report the synthetic seam");
        CHECK(a_n == live_len && b_n == canon.len,
              "walk stopped at (%d,%d), want (%d,%d) [seam at %d]",
              a_n, b_n, live_len, canon.len, k);
        int i = 0;
        while (i < live_len && i < canon.len && live[i] == canon.v[i]) i++;
        CHECK(i == k, "id-common %d expected to stop at the seam %d", i, k);
    }

    /* ---- leg 2: sync seam-rescue keeps the live KV ---- */
    {
        pulsar_session *s = NULL;
        if (pulsar_session_create(&s, e, 4096) != 0) { fprintf(stderr, "session create failed\n"); return 1; }
        char err[256];
        /* Build the session with the SAMPLED boundaries: live[0..k+split+8) */
        const int l1 = k + split.len + 8;
        pulsar_tokens plive;
        memset(&plive, 0, sizeof(plive));
        plive.v = live;
        plive.len = plive.cap = l1;
        CHECK(pulsar_session_sync(s, &plive, err, sizeof(err)) == 0,
              "live-boundary sync failed: %s", err);
        /* Echo turn: canonical render of the same bytes plus 24 new tokens. */
        const int c1 = k + 1 + 8;      /* same bytes as live[0..l1) */
        const int c2 = c1 + 24;
        pulsar_tokens pcanon;
        memset(&pcanon, 0, sizeof(pcanon));
        pcanon.v = canon.v;
        pcanon.len = pcanon.cap = c2;
        CHECK(pulsar_session_sync(s, &pcanon, err, sizeof(err)) == 0,
              "canonical sync failed: %s", err);
        const int want_len = l1 + 24;
        CHECK(pulsar_session_pos(s) == want_len,
              "stitched length %d want %d", pulsar_session_pos(s), want_len);
        const pulsar_tokens *toks = pulsar_session_tokens(s);
        CHECK(toks && toks->len == want_len, "session tokens len %d want %d",
              toks ? toks->len : -1, want_len);
        if (toks && toks->len == want_len) {
            CHECK(toks->v[k] == split.v[0] && toks->v[k + 1] == split.v[1],
                  "seam ids replaced (%d,%d) want (%d,%d) — live KV was rebuilt",
                  toks->v[k], toks->v[k + 1], split.v[0], split.v[1]);
            CHECK(toks->v[want_len - 1] == canon.v[c2 - 1],
                  "suffix tail %d want %d", toks->v[want_len - 1], canon.v[c2 - 1]);
        }
        pulsar_session_free(s);
    }

    /* ---- leg 3: SHORTER echo (the production shape) ----------------------
     * Live carries a tail the client's echo does not — it strips generated
     * reasoning, so the echo both re-tokenizes across the seam AND ends
     * before the live frontier (measured 2026-08-28: live 390,258 vs echo
     * 390,018).  The old resolver gate demanded the echo consume the whole
     * live history, so this shape could never be served warm and fell to a
     * disk snapshot that REPLACED the live session.  Same retention
     * discriminator as leg 2: split ids surviving proves the live KV was
     * rewound and stitched rather than rebuilt. */
    {
        pulsar_session *s = NULL;
        if (pulsar_session_create(&s, e, 4096) != 0) { fprintf(stderr, "session create failed\n"); return 1; }
        char err[256];
        const int l1 = k + split.len + 8;      /* shared region, live boundaries */
        const int tail = 30;                   /* "generated reasoning", stripped */
        pulsar_tokens plive;
        memset(&plive, 0, sizeof(plive));
        plive.v = (int *)malloc(sizeof(int) * (size_t)(l1 + tail));
        plive.cap = l1 + tail;
        memcpy(plive.v, live, sizeof(int) * (size_t)l1);
        memcpy(plive.v + l1, canon.v + 400, sizeof(int) * (size_t)tail);
        plive.len = l1 + tail;
        CHECK(pulsar_session_sync(s, &plive, err, sizeof(err)) == 0,
              "leg3 live sync failed: %s", err);

        /* Echo: canonical render of the SHARED bytes only, plus a new turn. */
        const int c1 = k + 1 + 8;
        const int newn = 12;
        pulsar_tokens pecho;
        memset(&pecho, 0, sizeof(pecho));
        pecho.v = (int *)malloc(sizeof(int) * (size_t)(c1 + newn));
        pecho.cap = c1 + newn;
        memcpy(pecho.v, canon.v, sizeof(int) * (size_t)c1);
        memcpy(pecho.v + c1, canon.v + 700, sizeof(int) * (size_t)newn);
        pecho.len = c1 + newn;

        /* The walk must stop at the shared boundary, not inside either tail —
         * assert it so the leg cannot silently test a different shape. */
        pulsar_prefix_match m3;
        pulsar_tokens_prefix_match(e, plive.v, plive.len, pecho.v, pecho.len, &m3);
        const int a_n = m3.live_cut, b_n = m3.prompt_cut;
        CHECK(a_n == l1 && b_n == c1,
              "leg3 walk (%d,%d) want (%d,%d) — tails may share a leading byte",
              a_n, b_n, l1, c1);

        CHECK(pulsar_session_sync(s, &pecho, err, sizeof(err)) == 0,
              "leg3 echo sync failed: %s", err);
        const int want_len = l1 + newn;
        CHECK(pulsar_session_pos(s) == want_len,
              "leg3 stitched length %d want %d (rebuild would give %d)",
              pulsar_session_pos(s), want_len, c1 + newn);
        const pulsar_tokens *toks = pulsar_session_tokens(s);
        if (toks && toks->len == want_len) {
            CHECK(toks->v[k] == split.v[0] && toks->v[k + 1] == split.v[1],
                  "leg3 seam ids replaced (%d,%d) want (%d,%d) — live KV was rebuilt",
                  toks->v[k], toks->v[k + 1], split.v[0], split.v[1]);
            CHECK(toks->v[want_len - 1] == canon.v[700 + newn - 1],
                  "leg3 suffix tail %d want %d",
                  toks->v[want_len - 1], canon.v[700 + newn - 1]);
        }
        free(pecho.v);
        free(plive.v);
        pulsar_session_free(s);
    }

    /* ---- leg 4: a FORK across seams (the routing-side holdout) -----------
     * Routing validated forks by token id, so a bank whose history carries
     * sampled boundaries could not be forked onto its own canonical echo:
     * `warm-advance-in-place refused (token-mismatch)`, measured in
     * production 2026-08-28 at 328k and 390k.  Unlike leg 3 this leg IS a
     * discriminator — the id compare fails outright on the old code.  The
     * destination's stamped history must also carry the LIVE (split) ids,
     * because the checkpoint labels the KV that was cloned. */
    {
        pulsar_session *s = NULL;
        if (pulsar_session_create(&s, e, 4096) != 0) { fprintf(stderr, "session create failed\n"); return 1; }
        if (pulsar_session_bank_count(s) < 2) {
            fprintf(stderr, "leg4 SKIPPED: needs PULSAR_MSEQ_BANKS>=2\n");
        } else {
            char err[256];
            const int l1 = k + split.len + 8;
            pulsar_tokens plive;
            memset(&plive, 0, sizeof(plive));
            plive.v = live;
            plive.len = plive.cap = l1;
            CHECK(pulsar_session_sync(s, &plive, err, sizeof(err)) == 0,
                  "leg4 live sync failed: %s", err);
            /* The echo: canonical ids for the same bytes, one turn longer. */
            const int c1 = k + 1 + 8;
            const int c2 = c1 + 16;
            const int rc = pulsar_session_bank_fork(s, 0, 1, canon.v, c2, l1);
            CHECK(rc == 0, "leg4 fork refused (rc=%d) — id-exact validation", rc);
            if (rc == 0) {
                CHECK(pulsar_session_bank_state_restore(s, 1),
                      "leg4 restore of forked bank failed");
                const pulsar_tokens *dt = pulsar_session_tokens(s);
                CHECK(dt && dt->len == l1,
                      "leg4 forked frontier %d want %d", dt ? dt->len : -1, l1);
                if (dt && dt->len == l1) {
                    CHECK(dt->v[k] == split.v[0] && dt->v[k + 1] == split.v[1],
                          "leg4 forked history stamped with CANONICAL ids (%d,%d) "
                          "want live (%d,%d) — the label would misdescribe the KV",
                          dt->v[k], dt->v[k + 1], split.v[0], split.v[1]);
                }
            }
        }
        pulsar_session_free(s);
    }

    free(live);
    free(split.v);
    pulsar_tokens_free(&canon);
    pulsar_engine_close(e);

    if (g_fail) { fprintf(stderr, "SEAM GATE: FAIL\n"); return 1; }
    printf("SEAM GATE: PASS\n");
    return 0;
}

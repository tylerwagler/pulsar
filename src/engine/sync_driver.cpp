/* sync_driver.cpp -- L272 P2: the core's sync, the one every family runs (L284) by supplying its sync ops.
 *
 * A sync makes the live bank hold exactly `prompt`.  For a family whose state is restored from grid
 * checkpoints (kv_state.h) and prefilled by the core loop (prefill_loop.cpp), the decision is the same
 * every time, so it lives here once:
 *   - the view must agree with the bank's state to be continued (the family says whether it does);
 *   - the same prompt with fresh logits is a no-op;
 *   - a prompt that extends a view the bank prefilled whole continues it (L284: decode rows are not a
 *     prefill's bytes, so a view holding any resumes as below);
 *   - otherwise resume from the deepest grid checkpoint the prompt still shares (the one resume rule,
 *     pulsar_session_resume_point -- one token short of the prompt, so the stale-logits case re-evaluates
 *     its last row), else reset the bank and prefill from 0;
 *   - the prefill runs on the core loop: interruptible, with progress, cut by the one rule over the family's
 *     prefill shape (prefill_loop.cpp).
 * Qwen (L266, L272 P2) and DeepSeek (L284; its L148 self-heal is its state_agrees: a stale compressor or a
 * frontier ahead of the view is a view the state does not agree with) both run on it.
 *
 * L115/L284 token seam, the core's for every family: a client re-sends sampled turns in canonical spelling
 * (Claude Code after tool continuations), so the ids part at the first seam while every byte still agrees.
 * pulsar_session_seam_stitch keeps the live history up to the deepest shared byte boundary and the prompt
 * after it; the driver syncs the stitched prompt (it extends the view, or resumes from the deepest grid
 * checkpoint at or below the boundary through the one resume rule).  The stitch is read BEFORE the agreement
 * check, so a bank whose state moved past its view (a stop's rewind left it stale) keeps the rescue.
 *
 * L268 images, the core's for every family (image_front.cpp, image_identity.cpp):
 *   - a request's blocks must each fit one chunk; a prompt with no images carries no sentinel;
 *   - the live KV is kept only under the licence: the prompt extends it and the held images' records are
 *     exactly the live ones (else the bank resets);
 *   - a resume from a grid checkpoint keeps the KV below it only when the records below it are the request's,
 *     and never from inside a block (else cold);
 *   - the images are borrowed on the session for the prefill (the family's chunk merges the blocks it owns),
 *     and the records follow the view: what landed, interrupted or not. */
#include "pulsar_engine_internal.h"

/* The records of `images` whose blocks end at or below `limit` -- the KV's image identity after a prefill
 * stood at `limit`.  A request without images keeps the records below the start (a text extension keeps its
 * prefix's blocks). */
static void sync_note_images(pulsar_session *s, const pulsar_tokens *prompt, const pulsar_image_ref *images,
                             int n_images, uint32_t limit) {
    if (n_images <= 0) {
        pulsar_image_identity_trim(&s->live_images, limit);
        return;
    }
    if (!pulsar_image_identity_build(s->engine->family->vision, s->engine, prompt->v, prompt->len, images, n_images,
                                     limit, &s->live_images))
        s->live_images.n = 0;
}

/* Does the KV below `G` hold exactly the request's images there -- and is G outside every block? */
static bool sync_images_hold_below(const pulsar_session *s, const pulsar_tokens *prompt,
                                   const pulsar_image_ref *images, int n_images, uint32_t G) {
    for (int i = 0; i < n_images; i++) {
        int len = 0;
        (void)pulsar_image_block_extent(s->engine, prompt->v, prompt->len, images[i].start_pos, &len);
        if ((uint32_t)images[i].start_pos < G && (uint32_t)(images[i].start_pos + len) > G) return false;
    }
    pulsar_image_identity want, held = s->live_images;
    if (!pulsar_image_identity_build(s->engine->family->vision, s->engine, prompt->v, prompt->len, images, n_images,
                                     G, &want))
        return false;
    pulsar_image_identity_trim(&held, G);
    return pulsar_image_identity_equal(&held, &want);
}

bool pulsar_session_seam_stitch(pulsar_session *s, const pulsar_tokens *prompt, const pulsar_image_ref *images,
                                int n_images, int past, pulsar_seam_stitch *out) {
    pulsar_prefix_match m;
    s->prefix_match(prompt, &m);
    if (m.live_cut <= past) return false;
    const int live_n = m.live_cut, prompt_n = m.prompt_cut;
    pulsar_tokens *st = &out->tokens;
    free(st->v);
    st->cap = live_n + (prompt->len - prompt_n);
    st->v = (int *)xmalloc((size_t)st->cap * sizeof(int));
    memcpy(st->v, s->checkpoint.v, (size_t)live_n * sizeof(int));
    memcpy(st->v + live_n, prompt->v + prompt_n, (size_t)(prompt->len - prompt_n) * sizeof(int));
    st->len = st->cap;
    out->live_cut = live_n;
    out->prompt_cut = prompt_n;
    /* L226/L273: a block below the seam sits at its live position (which the prompt's canonical ids may have
     * shifted), a block above it moves with the suffix, and the images arrive in prompt order, so the stitched
     * prompt's sentinel blocks, walked in order, are their positions.  A stitch that cuts a block (the walk
     * finds a malformed span) or leaves a block count the request's images do not match declines. */
    if (n_images <= 0) return true;
    bool placed_ok = n_images <= pulsar_seam_stitch::IMAGES_MAX;
    if (placed_ok) {
        int starts[pulsar_seam_stitch::IMAGES_MAX];
        const int nb = pulsar_image_block_starts(s->engine, st, st->len, starts, pulsar_seam_stitch::IMAGES_MAX);
        placed_ok = nb == n_images;
        for (int i = 0; placed_ok && i < n_images; i++) {
            out->placed[i] = images[i];
            out->placed[i].start_pos = starts[i];
        }
        if (!placed_ok)
            fprintf(stderr, "pulsar: image request: the stitched prompt (live %d + suffix from %d) carries "
                            "%d image block(s) for %d image(s) -- rebuilding cold\n",
                    live_n, prompt_n, nb, n_images);
    }
    return placed_ok;
}

int pulsar_session_sync_default(pulsar_session *s, const pulsar_tokens *prompt, const pulsar_image_ref *images,
                                int n_images, const pulsar_sync_ops *ops, char *err, size_t errlen) {
    /* a prompt that fills the context leaves no row for the eval that follows it (L272 B4) */
    if (!prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        if (err) snprintf(err, errlen, "%s: prompt length %d outside [1, %d)", ops->name, prompt ? prompt->len : -1,
                          s->ctx_size);
        return 1;
    }
    s->resume_origin = -1;
    pulsar_engine *e = s->engine;
    if (n_images > 0) {
        if (!e->vision_ready) {
            if (err) snprintf(err, errlen, "this model has no vision tower bound; it cannot accept images");
            return 1;
        }
        if (!pulsar_image_spans_fit(e, prompt->v, prompt->len, images, n_images, s->prefill_cap, NULL, err, errlen))
            return 1;
    } else if (!pulsar_image_refuse_orphans(e, prompt, err, errlen)) {
        return 1;
    }
    const uint32_t live = pulsar_session_live_bank(s);
    /* `common` and the seam are read before the agreement check: a view the state has moved past can no longer
     * be continued, but its prefix (and the live bytes it re-spells) still say which checkpoint the prompt shares */
    int common = s->checkpoint_valid ? pulsar_tokens_common_prefix(&s->checkpoint, prompt) : 0;
    /* L284 token seam: a prompt that leaves the view by ids may still re-spell it by bytes past `common`; the
     * stitched prompt keeps those live tokens, and everything below runs on it (its images re-placed) */
    pulsar_seam_stitch seam;
    if (s->checkpoint_valid && common < s->checkpoint.len &&
        pulsar_session_seam_stitch(s, prompt, images, n_images, common, &seam)) {
        if (seam.tokens.len >= s->ctx_size) {
            if (err) snprintf(err, errlen, "%s: prompt length %d outside [1, %d)", ops->name, seam.tokens.len,
                              s->ctx_size);
            return 1;
        }
        fprintf(stderr, "pulsar: %s: token seam -- keeping %d live tokens for the prompt's first %d (shared by "
                        "id %d)\n", ops->name, seam.live_cut, seam.prompt_cut, common);
        prompt = &seam.tokens;
        if (n_images > 0) images = seam.placed;
        common = pulsar_tokens_common_prefix(&s->checkpoint, prompt);
    }
    /* the bank's state is the authority; the view must agree with it to be continued (a batched step on the live
     * bank moves the state, not the view; DeepSeek: a stale compressor or a frontier ahead of the view, L148) */
    if (s->checkpoint_valid && !ops->state_agrees(s)) s->checkpoint_valid = false;
    /* L268: an image request extends the live KV only under the licence */
    if (n_images > 0 && s->checkpoint_valid) {
        pulsar_image_licence lic;
        pulsar_image_licence_decide(s, prompt, images, n_images, &lic);
        if (lic.extends_live && !lic.keep) {
            fprintf(stderr, "pulsar: %s: image request: the live history's images are not this prompt's (%s) -- "
                            "not continuing it\n", ops->name,
                    lic.straddles ? "a block straddles the common prefix" : "different images or blocks");
            s->checkpoint_valid = false;
        }
    }
    /* L284: only a view the bank PREFILLED whole is continued (pulsar_session_bank_continues, the rule DeepSeek's
     * sync reads too).  Decode rows are the decode arms' (a step's
     * rows are not a prefill chunk's bytes, kv_state_qwen.cpp), so a view with any is resumed instead, from
     * the deepest checkpoint at or below the prefill frontier -- the tokens generated since are prefilled
     * again and the prompt is the cold prefill's bytes, as DeepSeek's sync does it (L195) */
    const bool extends = s->checkpoint_valid && common == s->checkpoint.len && common < prompt->len &&
                         pulsar_session_bank_continues(s, live, common);
    /* the same prompt is a no-op only while the logits are its next-token row; when they are stale the
     * resume below stops one token short, so the last row is evaluated again */
    if (s->checkpoint_valid && common == s->checkpoint.len && common == prompt->len && !s->logits_stale) return 0;
    uint32_t start = 0;
    if (extends) {
        start = (uint32_t)common;
    } else {
        uint32_t G = pulsar_session_resume_point(s, live, common, prompt->len);
        /* L268: the KV below G is kept only when its image records are the request's there, and G is outside
         * every block (its merged rows cannot be re-evaluated) */
        if (G && !sync_images_hold_below(s, prompt, images, n_images, G)) {
            fprintf(stderr, "pulsar: %s: bank %u's checkpoint at %u holds other images (or cuts a block) -- "
                            "prefilling from 0\n", ops->name, live, G);
            G = 0;
        }
        if (G) {
            if (!pulsar_ckpt_restore(pulsar_session_kv_store(s), live, G)) {
                if (err) snprintf(err, errlen, "%s: restoring bank %u's checkpoint at %u failed", ops->name, live, G);
                return 1;
            }
            s->logits_stale = true;   /* a restore moves the state, not the logits */
            start = G;
            pulsar_image_identity_trim(&s->live_images, G);
            fprintf(stderr, "pulsar: %s: bank %u resumes from its checkpoint at %u (prompt %d, shared %d)\n",
                    ops->name, live, G, prompt->len, common);
        } else if (!ops->reset_bank(s)) {
            if (err) snprintf(err, errlen, "%s: could not clear the session's state", ops->name);
            return 1;
        } else {
            s->live_images.n = 0;   /* a cold bank holds no image */
        }
    }
    s->resume_origin = (int)start;
    /* the loop owns the view from here: it stands at each chunk's end as the chunk lands, so an
     * interrupted sync leaves a valid prefix the next sync extends.  The images are borrowed for it (the
     * family's chunk merges the blocks it owns) and released on every exit. */
    if (n_images > 0 && !pulsar_image_identity_build(e->family->vision, e, prompt->v, prompt->len, images, n_images,
                                                     (uint32_t)prompt->len, &s->sync_identity)) {
        if (err) snprintf(err, errlen, "%s: the request's image blocks do not read", ops->name);
        return 1;
    }
    s->sync_images = n_images > 0 ? images : NULL;
    s->sync_n_images = n_images > 0 ? n_images : 0;
    s->sync_prompt = n_images > 0 ? prompt : NULL;
    const int rc = ops->prefill(s, prompt, start);
    s->sync_images = NULL;
    s->sync_n_images = 0;
    s->sync_prompt = NULL;
    if (rc != 1) sync_note_images(s, prompt, images, n_images, (uint32_t)s->checkpoint.len);
    else s->live_images.n = 0;
    if (rc == 1 && err) snprintf(err, errlen, "%s: prefill refused (see the log for the op)", ops->name);
    return rc;
}

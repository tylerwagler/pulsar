/* prefill_loop.cpp -- L272 P2: the family-neutral prefill loop.
 *
 * A sync's prefill is the same walk for every family: ONE cut rule (pulsar_prefill_plan_next_end), a grid
 * checkpoint wherever a chunk lands on the grid in a prefill's state (P13: pulsar_ckpt_landed, the one capture
 * rule), the session's view advanced as each chunk lands, the progress hooks told, and the cancel hook polled at
 * every chunk boundary where a resume is exact -- which is how the server yields a long prefill to other slots and
 * honours a client that hung up (gen_prefill_cancel_cb).  The family supplies only the chunk's forward and its
 * prefill shape (pulsar_sync_ops::prefill_shape).  Before L272 Qwen's prefill had its own loop with no progress and
 * no cancel; before L284 DeepSeek's sync had its own planner (imatrix.cpp) with the same rules in another order.
 *
 * The walk (pulsar_prefill_walk_run) is shared with DeepSeek's session-less range prefill (the imatrix collector,
 * the gates' classic suffix): the cut is the plan's, the effects are each caller's hooks.  The events are
 * prefill_chunk / prefill_display at the start and after each chunk. */
#include "pulsar_engine_internal.h"

void pulsar_prefill_plan_init(pulsar_prefill_plan *p, const pulsar_prefill_shape *shape, uint32_t start,
                              const pulsar_ckpt_store *store, const pulsar_engine *e, const pulsar_tokens *prompt,
                              const pulsar_image_ref *images, int n_images) {
    p->grid_cap = shape->cap;
    p->chunk_cap = start != 0u && shape->resumed_cap < shape->cap ? shape->resumed_cap : shape->cap;
    p->align = shape->align;
    p->store = store;
    p->e = e;
    p->prompt = prompt;
    p->images = n_images > 0 ? images : NULL;
    p->n_images = n_images > 0 ? n_images : 0;
}

/* L268: the image block a cut at `cut` would split -- [bs, be) with bs < cut < be -- or false.  The FIRST such
 * block in request order (the cut rule adjusts block by block in that order). */
static bool prefill_plan_block_across(const pulsar_prefill_plan *p, uint32_t cut, uint32_t *bs, uint32_t *be) {
    for (int i = 0; i < p->n_images; i++) {
        int len = 0;
        const int st = p->images[i].start_pos;
        if (!pulsar_image_block_extent(p->e, p->prompt->v, p->prompt->len, st, &len)) continue;
        if ((uint32_t)st < cut && (uint32_t)(st + len) > cut) {
            *bs = (uint32_t)st;
            *be = (uint32_t)(st + len);
            return true;
        }
    }
    return false;
}

uint32_t pulsar_prefill_plan_next_end(const pulsar_prefill_plan *p, uint32_t pos0, uint32_t end) {
    const uint32_t remaining = end - pos0;
    /* snap to the absolute grid of the cap after any unaligned start: a resume (start != 0) lands on the cold
     * prefill's boundaries, and so does the chunk after an image cut below -- on the cold pass too, which is what
     * keeps a resumed image prefill the cold one's chunk for chunk */
    uint32_t local_cap = p->chunk_cap;
    const uint32_t to_boundary = pulsar_prefill_to_boundary(pos0, p->grid_cap);
    if (to_boundary < local_cap) local_cap = to_boundary;
    uint32_t chunk = remaining < local_cap ? remaining : local_cap;
    /* every NON-final chunk end on the alignment (DeepSeek: the compress ratios' LCM -- one unaligned boundary makes
     * every later chunk's start unaligned and each takes the per-token compressor path; 1 = none).  The final chunk
     * keeps its exact remainder; an unaligned START pays for its first chunk only, which still ends aligned. */
    if (chunk < remaining && p->align > 1u) {
        const uint32_t aligned_end = (pos0 + chunk) / p->align * p->align;
        if (aligned_end > pos0) chunk = aligned_end - pos0;
    }
    /* L268: an image block is merged whole by the chunk that owns it, so a cut never splits one: it moves to the
     * block's start, or -- the block starts this chunk (it fits one: pulsar_image_spans_fit) -- past its end.
     * Positions only, so the cold pass and any resume over the same prompt cut alike. */
    for (int i = 0; i < p->n_images; i++) {
        int len = 0;
        const int st = p->images[i].start_pos;
        if (!pulsar_image_block_extent(p->e, p->prompt->v, p->prompt->len, st, &len)) continue;
        const uint32_t bs = (uint32_t)st, be = (uint32_t)(st + len), ce = pos0 + chunk;
        if (bs < ce && ce < be) chunk = bs > pos0 ? bs - pos0 : be - pos0;
    }
    /* L264 / P13: the final chunk stops at the last grid point inside it (pulsar_ckpt_final_cut), so the prefill
     * leaves the checkpoint the next turn resumes from -- not when that point falls inside an image block */
    if (pos0 + chunk == end) {
        const uint32_t cut = pulsar_ckpt_final_cut(p->store, pos0, end);
        uint32_t bs = 0, be = 0;
        if (!prefill_plan_block_across(p, cut, &bs, &be)) chunk = cut - pos0;
    }
    return pos0 + chunk;
}

int pulsar_prefill_walk_run(const pulsar_prefill_walk *w, uint32_t start, uint32_t end) {
    for (uint32_t pos0 = start; pos0 < end;) {
        if (w->stop && w->stop(w->ud)) {
            (void)pulsar_gpu_synchronize();   /* L188: drain before handing the stream to whatever runs next */
            return PULSAR_SESSION_SYNC_INTERRUPTED;
        }
        const uint32_t chunk_end = pulsar_prefill_plan_next_end(w->plan, pos0, end);
        if (chunk_end <= pos0 || chunk_end > end) {
            fprintf(stderr, "pulsar: prefill walk: a chunk from %u cut at %u (prompt ends at %u)\n", pos0, chunk_end, end);
            return 1;
        }
        if (!w->chunk(w->ud, pos0, chunk_end - pos0, chunk_end == end)) return 1;
        if (w->landed && !w->landed(w->ud, chunk_end)) return 1;
        pos0 = chunk_end;
        if (pos0 < end && w->stop && w->stop(w->ud)) {
            (void)pulsar_gpu_synchronize();
            return PULSAR_SESSION_SYNC_INTERRUPTED;
        }
    }
    return 0;
}

static void prefill_loop_progress(pulsar_session *s, int current, int total) {
    if (s->progress) s->progress(s->progress_ud, "prefill_chunk", current, total);
    if (s->display_progress) s->display_progress(s->display_progress_ud, "prefill_display", current, total);
}

/* The session loop's hooks over the walk: the view advanced and the shared capture rule applied as each chunk
 * lands, the session's cancel hook as the stop -- polled only where a resume is exact. */
struct prefill_loop_ctx {
    pulsar_session *s;
    const pulsar_tokens *prompt;
    uint32_t bank;
    pulsar_ckpt_store *ckpt;
    pulsar_prefill_chunk_fn chunk;
    void *ud;
    /** L281 (b): the walk stands where a resume is exact -- its start, or a chunk end the model's prefill
     *  reproduces (anywhere when split_invariant, else on the resume grid; a chunk never ends inside an image
     *  block).  A stop is honoured only there: an interrupt at an image block's off-grid end would resume from the
     *  grid point below it, inside the block, i.e. from 0. */
    bool at_resume_point;
};

static bool prefill_loop_chunk(void *ud, uint32_t pos0, uint32_t rows, bool last) {
    const prefill_loop_ctx *c = (const prefill_loop_ctx *)ud;
    return c->chunk(c->s, c->prompt, pos0, rows, last, c->ud);
}

static bool prefill_loop_landed(void *ud, uint32_t chunk_end) {
    prefill_loop_ctx *c = (prefill_loop_ctx *)ud;
    pulsar_session *s = c->s;
    for (int i = s->checkpoint.len; i < (int)chunk_end; i++) token_vec_push(&s->checkpoint, c->prompt->v[i]);
    s->checkpoint_valid = true;
    if (!pulsar_ckpt_landed(c->ckpt, c->bank, chunk_end)) return false;
    c->at_resume_point = c->ckpt->ops->split_invariant || chunk_end % c->ckpt->ops->resume_grid == 0u;
    prefill_loop_progress(s, (int)chunk_end, c->prompt->len);
    return true;
}

static bool prefill_loop_stop(void *ud) {
    const prefill_loop_ctx *c = (const prefill_loop_ctx *)ud;
    return c->at_resume_point && pulsar_session_cancelled(c->s);
}

int pulsar_prefill_loop(pulsar_session *s, const pulsar_tokens *prompt, uint32_t start, pulsar_prefill_chunk_fn chunk,
                        void *ud) {
    const uint32_t end = (uint32_t)prompt->len;
    pulsar_prefill_shape shape = {};
    s->engine->family->session->sync->prefill_shape(s, &shape);
    pulsar_ckpt_store *ckpt = pulsar_session_kv_store(s);
    if (shape.cap == 0u || shape.align == 0u || start >= end || !ckpt || !ckpt->ops) {
        fprintf(stderr, "pulsar: prefill loop: nothing to prefill (start %u, end %u, cap %u)\n", start, end, shape.cap);
        return 1;
    }
    pulsar_prefill_plan plan;
    pulsar_prefill_plan_init(&plan, &shape, start, ckpt, s->engine, prompt, s->sync_images, s->sync_n_images);
    if (plan.n_images > 0) {
        /* every block must fit one chunk of THIS walk (a resume's chunks may be narrower than the cold pass's) */
        char verr[384];
        if (!pulsar_image_spans_fit(s->engine, prompt->v, prompt->len, plan.images, plan.n_images, plan.chunk_cap,
                                    NULL, verr, sizeof(verr))) {
            fprintf(stderr, "pulsar: %s\n", verr);
            return 1;
        }
        /* L268: a start inside an image block would re-evaluate its merged rows without their image */
        uint32_t bs = 0, be = 0;
        if (start > 0 && prefill_plan_block_across(&plan, start, &bs, &be)) {
            fprintf(stderr, "pulsar: prefill loop: start %u is inside image block [%u, %u) -- refusing\n", start, bs, be);
            return 1;
        }
    }
    /* the view below `start` is the session's already; it grows with the prompt from here */
    s->checkpoint.len = 0;
    for (uint32_t i = 0; i < start; i++) token_vec_push(&s->checkpoint, prompt->v[i]);
    s->checkpoint_valid = start > 0u;
    prefill_loop_progress(s, (int)start, prompt->len);
    prefill_loop_ctx c = { s, prompt, pulsar_session_live_bank(s), ckpt, chunk, ud, true };
    const pulsar_prefill_walk w = { &plan, prefill_loop_chunk, prefill_loop_landed, prefill_loop_stop, &c };
    const int rc = pulsar_prefill_walk_run(&w, start, end);
    /* the logits are the prompt's next-token row only when the final chunk ran (a stop leaves the view
     * at a boundary, a failure leaves it at the last chunk that landed) */
    s->logits_stale = rc != 0;
    if (rc == 1) s->checkpoint_valid = false;
    return rc;
}

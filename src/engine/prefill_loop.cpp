/* prefill_loop.cpp -- L272 P2: the family-neutral prefill loop.
 *
 * A sync's prefill is the same walk for every family: chunks of the session's prefill cap from the
 * resume point, one chunk cut at the grid point the family captures, the session's view advanced as
 * each chunk lands, the progress hooks told, and the cancel hook polled at every chunk boundary --
 * which is how the server yields a long prefill to other slots and honours a client that hung up
 * (gen_prefill_cancel_cb).  The family supplies only the chunk's forward.  Before L272 Qwen's prefill
 * had its own loop with no progress and no cancel: a disconnected Claude Code request prefilled to the
 * end and held every other slot behind it.
 *
 * Both this loop and DeepSeek's planner (imatrix.cpp gpu_graph_prefill_chunked_range: absolute cap
 * snaps, compress-ratio alignment, image blocks, a capture at every grid chunk end) run ONE walk,
 * pulsar_prefill_walk_run: the order is the walk's, the planning and the effects are each caller's
 * hooks.  The events are the planner's: prefill_chunk / prefill_display at the start and after each
 * chunk. */
#include "pulsar_engine_internal.h"

int pulsar_prefill_walk_run(const pulsar_prefill_walk *w, uint32_t start, uint32_t end) {
    for (uint32_t pos0 = start; pos0 < end;) {
        if (w->stop && w->stop(w->ud)) {
            (void)pulsar_gpu_synchronize();   /* L188: drain before handing the stream to whatever runs next */
            return PULSAR_SESSION_SYNC_INTERRUPTED;
        }
        const uint32_t chunk_end = w->next_end(w->ud, pos0, end);
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

/* The session loop's hooks over the walk: chunks of `cap`, one cut at `capture_at`, the view advanced and
 * the state captured as each chunk lands, the session's cancel hook as the stop. */
struct prefill_loop_ctx {
    pulsar_session *s;
    const pulsar_tokens *prompt;
    uint32_t cap, capture_at, bank;
    pulsar_ckpt_store *ckpt;
    pulsar_prefill_chunk_fn chunk;
    void *ud;
};

static uint32_t prefill_loop_next_end(void *ud, uint32_t pos0, uint32_t end) {
    const prefill_loop_ctx *c = (const prefill_loop_ctx *)ud;
    uint32_t rows = end - pos0 < c->cap ? end - pos0 : c->cap;
    if (c->capture_at && pos0 < c->capture_at && pos0 + rows > c->capture_at) rows = c->capture_at - pos0;
    return pos0 + rows;
}

static bool prefill_loop_chunk(void *ud, uint32_t pos0, uint32_t rows, bool last) {
    const prefill_loop_ctx *c = (const prefill_loop_ctx *)ud;
    return c->chunk(c->s, c->prompt, pos0, rows, last, c->ud);
}

static bool prefill_loop_landed(void *ud, uint32_t chunk_end) {
    const prefill_loop_ctx *c = (const prefill_loop_ctx *)ud;
    pulsar_session *s = c->s;
    for (int i = s->checkpoint.len; i < (int)chunk_end; i++) token_vec_push(&s->checkpoint, c->prompt->v[i]);
    s->checkpoint_valid = true;
    if (c->capture_at && chunk_end == c->capture_at && !pulsar_ckpt_capture(c->ckpt, c->bank, c->capture_at))
        return false;
    prefill_loop_progress(s, (int)chunk_end, c->prompt->len);
    return true;
}

static bool prefill_loop_stop(void *ud) { return pulsar_session_cancelled(((const prefill_loop_ctx *)ud)->s); }

int pulsar_prefill_loop(pulsar_session *s, const pulsar_tokens *prompt, uint32_t start, uint32_t cap,
                        uint32_t capture_at, pulsar_ckpt_store *ckpt, uint32_t bank,
                        pulsar_prefill_chunk_fn chunk, void *ud) {
    const uint32_t end = (uint32_t)prompt->len;
    if (cap == 0u || start >= end) {
        fprintf(stderr, "pulsar: prefill loop: nothing to prefill (start %u, end %u, cap %u)\n", start, end, cap);
        return 1;
    }
    /* the view below `start` is the session's already; it grows with the prompt from here */
    s->checkpoint.len = 0;
    for (uint32_t i = 0; i < start; i++) token_vec_push(&s->checkpoint, prompt->v[i]);
    s->checkpoint_valid = start > 0u;
    prefill_loop_progress(s, (int)start, prompt->len);
    prefill_loop_ctx c = { s, prompt, cap, capture_at, bank, ckpt, chunk, ud };
    const pulsar_prefill_walk w = { prefill_loop_next_end, prefill_loop_chunk, prefill_loop_landed, prefill_loop_stop, &c };
    const int rc = pulsar_prefill_walk_run(&w, start, end);
    /* the logits are the prompt's next-token row only when the final chunk ran (a stop leaves the view
     * at a boundary, a failure leaves it at the last chunk that landed) */
    s->logits_stale = rc != 0;
    if (rc == 1) s->checkpoint_valid = false;
    return rc;
}

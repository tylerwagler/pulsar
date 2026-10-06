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
 * DeepSeek's planner (imatrix.cpp gpu_graph_prefill_chunked_range) still runs its own walk -- absolute
 * cap snaps, compress-ratio alignment, image blocks, a capture at every grid chunk end -- and moves
 * onto this loop in a later step under chunk_neutrality_gate.  The events and the poll points here are
 * that planner's: prefill_chunk / prefill_display at the start and after each chunk, the hook polled
 * before and after each chunk. */
#include "pulsar_engine_internal.h"

static void prefill_loop_progress(pulsar_session *s, int current, int total) {
    if (s->progress) s->progress(s->progress_ud, "prefill_chunk", current, total);
    if (s->display_progress) s->display_progress(s->display_progress_ud, "prefill_display", current, total);
}

/* A stop at a chunk boundary: the view stands at the last chunk's end (set as it landed), and the
 * logits are not that position's next-token row (only the final chunk heads its last row). */
static int prefill_loop_interrupted(pulsar_session *s) {
    s->logits_stale = true;
    (void)pulsar_gpu_synchronize();   /* L188: drain before handing the stream to whatever runs next */
    return PULSAR_SESSION_SYNC_INTERRUPTED;
}

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
    for (uint32_t pos0 = start; pos0 < end;) {
        if (pulsar_session_cancelled(s)) return prefill_loop_interrupted(s);
        uint32_t rows = end - pos0 < cap ? end - pos0 : cap;
        if (capture_at && pos0 < capture_at && pos0 + rows > capture_at) rows = capture_at - pos0;
        const uint32_t chunk_end = pos0 + rows;
        if (!chunk(s, prompt, pos0, rows, chunk_end == end, ud)) {
            s->checkpoint_valid = false;
            s->logits_stale = true;
            return 1;
        }
        for (uint32_t i = pos0; i < chunk_end; i++) token_vec_push(&s->checkpoint, prompt->v[i]);
        s->checkpoint_valid = true;
        if (capture_at && chunk_end == capture_at && !pulsar_ckpt_capture(ckpt, bank, capture_at)) {
            s->logits_stale = true;
            return 1;
        }
        prefill_loop_progress(s, (int)chunk_end, prompt->len);
        pos0 = chunk_end;
        if (pos0 < end && pulsar_session_cancelled(s)) return prefill_loop_interrupted(s);
    }
    s->logits_stale = false;
    return 0;
}

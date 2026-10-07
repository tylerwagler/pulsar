/* sync_driver.cpp -- L272 P2: the core's default sync, the one a family gets by supplying three ops.
 *
 * A sync makes the live bank hold exactly `prompt`.  For a family whose state is restored from grid
 * checkpoints (kv_state.h) and prefilled by the core loop (prefill_loop.cpp), the decision is the same
 * every time, so it lives here once:
 *   - the view must agree with the bank's state to be continued (the family says whether it does);
 *   - the same prompt with fresh logits is a no-op;
 *   - a prompt that extends the view continues it;
 *   - otherwise resume from the deepest grid checkpoint the prompt still shares (the one resume rule,
 *     pulsar_session_resume_point -- one token short of the prompt, so the stale-logits case re-evaluates
 *     its last row), else reset the bank and prefill from 0;
 *   - the prefill runs on the core loop: interruptible, with progress, at the session's chunk cap.
 * Qwen (L266, L272 P2) runs on it.  DeepSeek keeps its own sync for what only it has -- image licensing,
 * the compressor self-heal (L148) and the token-seam rescue (L115) -- over the same shared pieces: the
 * resume rule, the prefill walk and the logits flag. */
#include "pulsar_engine_internal.h"

int pulsar_session_sync_default(pulsar_session *s, const pulsar_tokens *prompt, const pulsar_sync_ops *ops,
                                char *err, size_t errlen) {
    /* a prompt that fills the context leaves no row for the eval that follows it (L272 B4) */
    if (!prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        if (err) snprintf(err, errlen, "%s: prompt length %d outside [1, %d)", ops->name, prompt ? prompt->len : -1,
                          s->ctx_size);
        return 1;
    }
    s->resume_origin = -1;
    const uint32_t live = pulsar_session_live_bank(s);
    /* `common` is read before the agreement check: a view the state has moved past can no longer be
     * continued, but its prefix still says which checkpoint the prompt shares */
    const int common = s->checkpoint_valid ? pulsar_tokens_common_prefix(&s->checkpoint, prompt) : 0;
    /* the bank's state is the authority; the view must agree with it to be continued (a batched step on
     * the live bank moves the state, not the view) */
    if (s->checkpoint_valid && !ops->state_agrees(s)) s->checkpoint_valid = false;
    const bool extends = s->checkpoint_valid && common == s->checkpoint.len && common < prompt->len;
    /* the same prompt is a no-op only while the logits are its next-token row; when they are stale the
     * resume below stops one token short, so the last row is evaluated again */
    if (s->checkpoint_valid && common == s->checkpoint.len && common == prompt->len && !s->logits_stale) return 0;
    uint32_t start = 0;
    if (extends) {
        start = (uint32_t)common;
    } else {
        const uint32_t G = pulsar_session_resume_point(s, live, common, prompt->len);
        if (G) {
            if (!pulsar_ckpt_restore(pulsar_session_kv_store(s), live, G)) {
                if (err) snprintf(err, errlen, "%s: restoring bank %u's checkpoint at %u failed", ops->name, live, G);
                return 1;
            }
            s->logits_stale = true;   /* a restore moves the state, not the logits */
            start = G;
            fprintf(stderr, "pulsar: %s: bank %u resumes from its checkpoint at %u (prompt %d, shared %d)\n",
                    ops->name, live, G, prompt->len, common);
        } else if (!ops->reset_bank(s)) {
            if (err) snprintf(err, errlen, "%s: could not clear the session's state", ops->name);
            return 1;
        }
    }
    s->resume_origin = (int)start;
    /* the loop owns the view from here: it stands at each chunk's end as the chunk lands, so an
     * interrupted sync leaves a valid prefix the next sync extends */
    const int rc = ops->prefill(s, prompt, start);
    if (rc == 1 && err) snprintf(err, errlen, "%s: prefill refused (see the log for the op)", ops->name);
    return rc;
}

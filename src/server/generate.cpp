#include "pulsar_server_internal.h"



/* ---- OpenAI logprobs ledger ---------------------------------------------
 *
 * The contract these helpers exist to keep: a reported logprob is the TARGET
 * model's distribution at the position the token was drawn at.  That is only
 * true if the value is captured at the DRAW, from the very row the sampler
 * read, so every decode lane calls one of the two capture entries below
 * immediately after choosing its token and gen_emit_token consumes the capture
 * when that token is committed to the response.
 *
 * DSpark speculative decode is the one lane that cannot honour it: the target
 * rows that verified the accepted drafts live inside the fused verify batch and
 * are overwritten by the next step, and after
 * pulsar_session_generate_speculative returns the session's logits describe
 * only the position AFTER the last committed token.  Reading the drafter's
 * distribution instead would be a silently wrong number, so a request that asks
 * for logprobs decodes with speculation OFF (see gen_decode_init) and pays the
 * throughput instead. */
void logprob_ledger_reset(logprob_ledger *lg) {
    if (!lg) return;
    for (int i = 0; i < lg->len; i++) {
        free(lg->v[i].tok.piece);
        for (int t = 0; t < lg->v[i].n_top; t++) free(lg->v[i].top[t].piece);
        free(lg->v[i].top);
    }
    lg->len = 0;
    lg->streamed = 0;
    lg->pending_valid = false;
}



void logprob_ledger_free(logprob_ledger *lg) {
    if (!lg) return;
    logprob_ledger_reset(lg);
    free(lg->v);
    lg->v = NULL;
    lg->cap = 0;
}



/* pulsar_*_top_logprobs pads its output to k and marks the unused tail with
 * id < 0 (a vocabulary smaller than k, or an all-non-finite row).  Only the
 * filled prefix is meaningful. */
static int logprob_trim(const pulsar_token_score *top, int n) {
    int i = 0;
    while (i < n && top[i].id >= 0) i++;
    return i;
}



/* Capture from the session's own logits row — the classic single-slot lane,
 * where pulsar_session_sample() drew from exactly this distribution and the
 * session has not been advanced yet. */
void logprob_capture_session(logprob_ledger *lg, pulsar_session *sess, int token) {
    if (!lg || !lg->enabled) return;
    pulsar_token_score chosen;
    lg->pending_logprob = pulsar_session_token_logprob(sess, token, &chosen)
                              ? chosen.logprob : -INFINITY;
    lg->pending_n_top = lg->top_k > 0
                            ? logprob_trim(lg->pending_top,
                                           pulsar_session_top_logprobs(sess, lg->pending_top,
                                                                    lg->top_k))
                            : 0;
    lg->pending_valid = true;
}



/* Capture from ONE row of a batched decode step.  Those entries return a row
 * per bank and leave the session's logits untouched by contract, so the row the
 * host sampler read is the only correct source here. */
void logprob_capture_row(logprob_ledger *lg, const float *logits, int n_vocab, int token) {
    if (!lg || !lg->enabled) return;
    pulsar_token_score chosen;
    lg->pending_logprob = pulsar_logits_token_logprob(logits, n_vocab, token, &chosen)
                              ? chosen.logprob : -INFINITY;
    lg->pending_n_top = lg->top_k > 0
                            ? logprob_trim(lg->pending_top,
                                           pulsar_logits_top_logprobs(logits, n_vocab,
                                                                   lg->pending_top,
                                                                   lg->top_k))
                            : 0;
    lg->pending_valid = true;
}



/* Bind the pending capture to the token now being emitted, materializing the
 * alternatives' text here (the capture sites hold ids only; detokenizing 20 ids
 * per position is worth doing once, off the sampler).
 *
 * A missing capture means an emit path reached here without a matching draw
 * site.  Report nothing rather than a distribution belonging to some other
 * position: the whole request's logprobs are dropped, because a ledger with a
 * hole in it is not something a client can align to its tokens.  Returns false
 * exactly then, so the caller can say so in the log. */
bool logprob_commit(logprob_ledger *lg, pulsar_engine *engine, int token,
                    const char *piece, size_t piece_len, size_t end_off) {
    if (!lg || !lg->enabled) return true;
    if (!lg->pending_valid) {
        logprob_ledger_reset(lg);
        lg->enabled = false;
        return false;
    }
    if (lg->len == lg->cap) {
        lg->cap = lg->cap ? lg->cap * 2 : 64;
        lg->v = (logprob_entry *)server_xrealloc(lg->v, (size_t)lg->cap * sizeof(lg->v[0]));
    }
    logprob_entry *e = &lg->v[lg->len++];
    e->tok.token = token;
    e->tok.piece = xstrndup(piece ? piece : "", piece_len);
    e->tok.piece_len = piece_len;
    e->tok.logprob = lg->pending_logprob;
    e->end_off = end_off;
    e->n_top = lg->pending_n_top;
    e->top = NULL;
    if (e->n_top > 0) {
        e->top = (logprob_token *)server_xmalloc((size_t)e->n_top * sizeof(e->top[0]));
        for (int i = 0; i < e->n_top; i++) {
            size_t n = 0;
            char *text = pulsar_token_text(engine, lg->pending_top[i].id, &n);
            e->top[i].token = lg->pending_top[i].id;
            e->top[i].piece = text ? text : xstrndup("", 0);
            e->top[i].piece_len = text ? n : 0;
            e->top[i].logprob = lg->pending_top[i].logprob;
        }
    }
    lg->pending_valid = false;
    return true;
}



/* How many not-yet-streamed entries are fully covered by a release watermark:
 * the count of leading unstreamed entries whose piece ends at or before `upto`.
 * SIZE_MAX is the final flush (everything still held). */
int logprob_stream_ready(const logprob_ledger *lg, size_t upto) {
    if (!lg || !lg->enabled) return 0;
    int n = lg->streamed;
    while (n < lg->len && lg->v[n].end_off <= upto) n++;
    return n;
}



bool thinking_state::tail_ends_with(const char *s) const {
    const auto *st = this;
    int n = (int)strlen(s);
    return st->tail_len >= n && !memcmp(st->tail + st->tail_len - n, s, (size_t)n);
}



void thinking_state::feed(const char *p, size_t len) {
    auto *st = this;
    if (!st || !p) return;
    for (size_t i = 0; i < len; i++) {
        if (st->tail_len == (int)sizeof(st->tail)) {
            memmove(st->tail, st->tail + 1, sizeof(st->tail) - 1);
            st->tail_len--;
        }
        st->tail[st->tail_len++] = p[i];
        if (st->tail_ends_with("<think>")) st->inside = true;
        else if (st->tail_ends_with("</think>")) st->inside = false;
    }
}



thinking_state thinking_state_from_prompt(const request *r) {
    thinking_state st = {0};
    if (r && r->prompt_text) {
        st.feed(r->prompt_text, strlen(r->prompt_text));
    } else if (r && pulsar_think_mode_enabled(r->think_mode)) {
        st.inside = true;
    }
    return st;
}



static char *rendered_chat_system_region(const char *prompt_text) {
    if (!prompt_text) return xstrdup("");
    const char *p = prompt_text;
    const char *bos = PULSAR_SERVER_RENDER_BOS;
    const size_t bos_len = strlen(bos);
    if (!strncmp(p, bos, bos_len)) p += bos_len;
    const pulsar_think_mode prefixed_modes[] = {PULSAR_THINK_MAX, PULSAR_THINK_HIGH};
    for (size_t i = 0; i < sizeof(prefixed_modes) / sizeof(prefixed_modes[0]); i++) {
        const char *effort_prefix = pulsar_think_effort_prefix(prefixed_modes[i]);
        const size_t effort_prefix_len = strlen(effort_prefix);
        if (effort_prefix_len && !strncmp(p, effort_prefix, effort_prefix_len)) {
            p += effort_prefix_len;
            break;
        }
    }
    while (*p && isspace((unsigned char)*p)) p++;

    const char *user = strstr(p, PULSAR_RENDER_USER);
    const char *assistant = strstr(p, PULSAR_RENDER_ASSISTANT);
    const char *end = NULL;
    if (user && assistant) end = user < assistant ? user : assistant;
    else end = user ? user : assistant;
    if (!end) end = p + strlen(p);
    while (end > p && isspace((unsigned char)end[-1])) end--;
    return xstrndup(p, (size_t)(end - p));
}



/* A server-side tool result appended to the live session mid-turn: close an
 * open think block, then exactly what the replay path renders for ONE tool
 * message after the assistant's call turn -- EOS, the tool_result, the
 * generation prefix (render_live_tool_tail) -- or next-turn prefix reuse dies. */
static char *build_live_tool_result_suffix(const request *r,
                                           const thinking_state *thinking,
                                           const char *result_text) {
    const pulsar_think_mode mode = r ? r->think_mode : PULSAR_THINK_NONE;
    buf suffix = {0};
    if (pulsar_think_mode_enabled(mode) && thinking && thinking->inside) {
        buf_puts(&suffix, "</think>");
    }
    chat_msgs msgs = {0};
    chat_msg result = {0};
    result.role = xstrdup("tool");
    result.content = xstrdup(result_text ? result_text : "");
    chat_msgs_push(&msgs, result);
    char *tail = render_live_tool_tail(&msgs, 0, r && r->has_tools, mode);
    buf_puts(&suffix, tail);
    free(tail);
    chat_msgs_free(&msgs);
    return buf_take(&suffix);
}

char *build_invalid_dsml_tool_error_suffix(const request *r,
                                                  const thinking_state *thinking,
                                                  const char *detail) {
    char *system = rendered_chat_system_region(r ? r->prompt_text : NULL);
    buf tool_error = {0};
    buf_puts(&tool_error, "Tool error: invalid DSML tool call");
    if (detail && detail[0]) {
        buf_puts(&tool_error, ": ");
        buf_puts(&tool_error, detail);
    }
    buf_puts(&tool_error,
             "\nThe previous assistant output was not executed because the DSML syntax was malformed. "
             "Emit a new valid DSML tool call, or answer normally if no tool is needed.");
    if (system && system[0]) {
        buf_puts(&tool_error, "\n\nSystem prompt reminder:\n");
        buf_puts(&tool_error, system);
    }

    char *suffix = build_live_tool_result_suffix(r, thinking, tool_error.ptr ? tool_error.ptr : "");

    free(system);
    buf_free(&tool_error);
    return suffix;
}



bool should_remember_thinking_checkpoint(const request *r,
                                                const thinking_state *thinking,
                                                const char *finish) {
    if (!r || r->kind != REQ_CHAT) return false;
    /* has_tools is NOT a disqualifier: a client can advertise tools and still
     * strip reasoning on replay (openwebui).  prompt_preserves_reasoning now
     * reflects the client's ACTUAL replay behavior, so it is the sole gate;
     * remember_thinking_checkpoint renders the tool-context vs toolless form. */
    if (r->prompt_preserves_reasoning) return false;
    if (!pulsar_think_mode_enabled(r->think_mode)) return false;
    if (finish && (!strcmp(finish, "error") || !strcmp(finish, "length"))) return false;
    if (thinking && thinking->inside) return false;
    return true;
}



/* The assistant turn after prompt_text's generation prefix ("<｜Assistant｜>"
 * + "<think>" or "</think>") as the model SAMPLED it: reasoning, think closed,
 * content, DSML, and the EOS only when the turn sampled one (a tool-call turn
 * stops at the closing tool_calls tag; the replay's EOS there belongs to the
 * tail).  One primitive with the renderer (append_assistant_turn_sampled next
 * to append_assistant_turn_close), so the key byte-matches both the live KV
 * and the replay's prefix by construction (L196). */
char *build_tool_checkpoint_suffix(const request *r, const char *content,
                                          const char *reasoning, const tool_calls *calls) {
    const bool think = pulsar_think_mode_enabled(r->think_mode);
    buf suffix = {0};
    append_assistant_turn_sampled(&suffix, think, think ? (reasoning ? reasoning : "") : NULL,
                                  content, calls);
    return buf_take(&suffix);
}



char *build_responses_visible_assistant_suffix(const request *r,
                                                      const char *content,
                                                      const char *reasoning,
                                                      const tool_calls *calls) {
    buf suffix = {0};
    /* This suffix mirrors what a Responses client can replay, not necessarily
     * every token in KV.  Hidden reasoning stays live in the session unless the
     * next client replay is expected to include it.  In practice, pi replays
     * reasoning summaries for tool-call turns, but not for final assistant
     * answers; Codex currently requests no summaries at all.  So only include
     * reasoning in the remembered visible prefix when this assistant turn ended
     * in tool calls.  A client that does replay final-answer reasoning will not
     * match this visible shortcut and can still use exact token-prefix replay. */
    const bool think = pulsar_think_mode_enabled(r->think_mode);
    const bool replay = think && r->reasoning_summary_emit && calls && calls->len > 0;
    append_assistant_turn_sampled(&suffix, think, replay ? (reasoning ? reasoning : "") : NULL,
                                  content, calls);
    return buf_take(&suffix);
}



/* In thinking mode without tools, old assistant reasoning is intentionally not
 * rendered back into later prompts.  The sampled live graph still contains the
 * reasoning bytes, so the next request would miss the session cache even though
 * the visible conversation prefix is logically the same.
 *
 *   prompt-without-final-<think> + </think> + visible-content + eos
 *
 * is exactly the visible prefix that render_chat_prompt_text() will produce on
 * the next turn.  Do not rebuild the KV cache to erase hidden reasoning here:
 * that caused long post-answer pauses and threw away useful sampled state.
 * Instead, remember the visible bytes as a key for the current sampled frontier.
 * The next request can then continue from live KV while tokenizing only the new
 * visible suffix. */
char *build_toolless_thinking_visible_text(const request *r,
                                                  const char *content) {
    if (!r || !r->prompt_text) return NULL;
    if (!pulsar_think_mode_enabled(r->think_mode)) return NULL;

    size_t pt_len = strlen(r->prompt_text);
    const char *think_tag = "<think>";
    size_t tag_len = strlen(think_tag);
    if (pt_len < tag_len ||
        memcmp(r->prompt_text + pt_len - tag_len, think_tag, tag_len) != 0) {
        return NULL;
    }

    buf visible = {0};
    buf_append(&visible, r->prompt_text, pt_len - tag_len);
    /* the stripped turn: no opener, the think close, the content, EOS */
    append_assistant_turn_close(&visible, true, NULL, content, NULL);
    return buf_take(&visible);
}



bool server::should_canonicalize_tool_checkpoint(const tool_calls *calls) const {
    if (!calls || calls->len == 0) return false;
    return !(calls->raw_dsml && calls->raw_dsml[0]);
}


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
    /* V4.1 leads a thinking conversation with the System token and the effort
     * line; neither belongs to the client's system region. */
    const size_t sys_len = strlen(PULSAR_RENDER_SYSTEM);
    if (!strncmp(p, PULSAR_RENDER_SYSTEM, sys_len)) p += sys_len;
    p += pulsar_think_effort_prefix_len(p);
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
static char *build_live_tool_result_suffix_spans(const request *r,
                                                 const thinking_state *thinking,
                                                 const char *result_text,
                                                 chat_text_span **spans_out,
                                                 uint32_t *n_spans_out) {
    if (spans_out) *spans_out = NULL;
    if (n_spans_out) *n_spans_out = 0;
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
    /* The tool body is CLIENT data (L223): the tail marks it, and the ranges ride
     * along so the caller tokenises those bytes as plain text. */
    chat_text_span *tail_spans = NULL;
    uint32_t tail_n = 0;
    char *tail = render_live_tool_tail_spans(&msgs, 0, r && r->has_tools, mode, r ? r->family->v41 : true,
                                             &tail_spans, &tail_n);
    buf_puts_spanned(&suffix, tail, tail_spans, tail_n);
    free(tail);
    free(tail_spans);
    chat_msgs_free(&msgs);
    if (spans_out) {
        *spans_out = suffix.spans;
        *n_spans_out = suffix.n_spans;
        suffix.spans = NULL;
        suffix.n_spans = suffix.cap_spans = 0;
    }
    return buf_take(&suffix);
}

char *build_invalid_dsml_tool_error_suffix_spans(const request *r,
                                                 const thinking_state *thinking,
                                                 const char *detail,
                                                 chat_text_span **spans_out,
                                                 uint32_t *n_spans_out) {
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

    char *suffix = build_live_tool_result_suffix_spans(r, thinking,
                                                       tool_error.ptr ? tool_error.ptr : "",
                                                       spans_out, n_spans_out);

    free(system);
    buf_free(&tool_error);
    return suffix;
}

/* L284 P4: the rule every family's parser applies to a turn whose tool text is not a valid call --
 * malformed, unterminated (the cap cut it; no tag is ever repaired), or naming an undeclared tool.  Its
 * valid calls are kept and the broken ones dropped.  With none kept it loops ONCE through the
 * model-visible tool error -- non-streaming (a stream already carried the text), a chat with tools,
 * forced or not (the retry re-opens a forced call: continue_after_invalid_dsml) -- otherwise the turn
 * is TEXT (turn_as_text) with a truthful finish (turn_finish).  parser_finish_turn applies it. */
bool turn_tool_retry_allowed(const gen_state *g) {
    const request *r = &g->j->req;
    return r->kind == REQ_CHAT && r->has_tools && !r->stream && !g->recovery_attempted && strcmp(g->finish, "error");
}

static bool turn_tool_retry(server *s, session_slot *sl, gen_state *g, const char *detail, server_turn *out) {
    int recovery_tokens = 0;
    char recovery_err[160] = {0};
    server_log(PULSAR_LOG_WARNING, "pulsar-server: chat ctx=%s%s%s continuing with model-visible tool error",
               g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags);
    s->trace_event(g->trace_id, "continuing with model-visible tool error");
    tool_calls_free(&out->calls);
    if (!s->continue_after_invalid_dsml(sl, &g->j->req, &g->thinking, detail, &recovery_tokens, recovery_err,
                                        sizeof(recovery_err))) {
        g->finish = "error";
        out->finish = g->finish;
        snprintf(g->err, sizeof(g->err), "invalid tool call recovery failed: %s",
                 recovery_err[0] ? recovery_err : "unknown error");
        return false;
    }
    server_log(PULSAR_LOG_GENERATION, "pulsar-server: chat ctx=%s%s%s tool-error continuation appended %d tokens",
               g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags, recovery_tokens);
    s->trace_event(g->trace_id, "tool-error continuation appended %d tokens", recovery_tokens);
    out->retry = true;
    return true;
}

static const char *turn_finish(const gen_state *g, int n_calls) {
    if (!strcmp(g->finish, "error") || !strcmp(g->finish, "length")) return g->finish;
    return n_calls ? "tool_calls" : "stop";
}

static void turn_as_text(const gen_state *g, server_turn *out) {
    const char *t = g->text.ptr ? g->text.ptr : "";
    const char *close = pulsar_think_mode_enabled(g->j->req.think_mode) ? strstr(t, "</think>") : NULL;
    size_t answer = 0;
    if (close) {
        const size_t lo = strncmp(t, "<think>", 7) ? 0 : 7;
        if ((size_t)(close - t) > lo) out->reasoning = xstrndup(t + lo, (size_t)(close - t) - lo);
        answer = (size_t)(close - t) + strlen("</think>");
    }
    out->content = xstrndup(t + answer, g->text.len - answer);
    out->calls.len = 0;
    out->parse_failed = true;
    out->finish = turn_finish(g, 0);
}

bool parser_finish_turn(server *s, session_slot *sl, gen_state *g, server_turn *out, bool broken,
                        const char *why, bool logged, size_t tail_from, bool ok) {
    const request *r = &g->j->req;
    const char *detail = why;
    char dropped[512];
    if (out->calls.len) {
        /* L272: a call to an undeclared tool is not executable: dropped (after the family put the stream's
         * ids on the calls: a live projection stopped at the first such call, so every call before it kept
         * its index) */
        int kept = 0;
        for (int i = 0; i < out->calls.len; i++) {
            char d[512];
            if (tool_call_declared(r, out->calls.v[i].name, d, sizeof(d))) {
                out->calls.v[kept++] = out->calls.v[i];
                continue;
            }
            if (!broken) {
                snprintf(dropped, sizeof dropped, "%s", d);
                detail = dropped;
                logged = false;
            }
            broken = true;
            free(out->calls.v[i].id);
            free(out->calls.v[i].name);
            free(out->calls.v[i].arguments);
        }
        if (kept < out->calls.len) {
            /* the sampled bytes hold the dropped call: no tool memory for this turn (a prefix miss) */
            free(out->calls.raw_dsml);
            out->calls.raw_dsml = NULL;
        }
        out->calls.len = kept;
    }
    if (broken && !logged) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: chat ctx=%s%s%s %s", g->ctx_span, g->req_flags[0] ? " " : "",
                   g->req_flags, detail);
        s->trace_event(g->trace_id, "%s", detail);
    }
    if (broken && out->calls.len == 0) {
        free(out->content);
        free(out->reasoning);
        out->content = out->reasoning = NULL;
        if (turn_tool_retry_allowed(g)) return turn_tool_retry(s, sl, g, detail, out);
        turn_as_text(g, out);
        if (tail_from < g->text.len) {
            /* the stream stopped at the call; the rest goes out as text at the finish */
            out->tail = g->text.ptr + tail_from;
            out->tail_len = g->text.len - tail_from;
        }
        return ok;
    }
    if (out->calls.len) {
        s->assign_tool_call_ids(&out->calls, r->api);
        s->tool_memory_remember(&out->calls);   /* a no-op without the sampled bytes */
    }
    out->finish = turn_finish(g, out->calls.len);
    return ok;
}

char *build_invalid_dsml_tool_error_suffix(const request *r,
                                                  const thinking_state *thinking,
                                                  const char *detail) {
    return build_invalid_dsml_tool_error_suffix_spans(r, thinking, detail, NULL, NULL);
}



/* The assistant turn after prompt_text's generation prefix ("<｜Assistant｜>"
 * + "<think>" or "</think>") as the model SAMPLED it: reasoning, think closed,
 * content, DSML, and the EOS only when the turn sampled one (a tool-call turn
 * stops at the closing tool_calls tag; the replay's EOS there belongs to the
 * tail).  One primitive with the renderer (append_assistant_turn_sampled next
 * to append_assistant_turn_close), so the key byte-matches both the live KV
 * and the replay's prefix by construction (L196). */
char *build_tool_checkpoint_suffix_spans(const request *r, const char *content,
                                         const char *reasoning, const tool_calls *calls,
                                         chat_text_span **spans_out, uint32_t *n_spans_out) {
    if (spans_out) *spans_out = NULL;
    if (n_spans_out) *n_spans_out = 0;
    if (!r->family->assistant_turn_sampled) return NULL;   /* the family has no sampled-turn render */
    const bool think = pulsar_think_mode_enabled(r->think_mode);
    return r->family->assistant_turn_sampled(r, think, think ? (reasoning ? reasoning : "") : NULL, content, calls,
                                             spans_out, n_spans_out);
}

char *build_tool_checkpoint_suffix(const request *r, const char *content,
                                          const char *reasoning, const tool_calls *calls) {
    return build_tool_checkpoint_suffix_spans(r, content, reasoning, calls, NULL, NULL);
}



char *build_responses_visible_assistant_suffix_spans(const request *r,
                                                     const char *content,
                                                     const char *reasoning,
                                                     const tool_calls *calls,
                                                     chat_text_span **spans_out,
                                                     uint32_t *n_spans_out) {
    if (spans_out) *spans_out = NULL;
    if (n_spans_out) *n_spans_out = 0;
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
    if (!r->family->assistant_turn_sampled) return NULL;   /* the family has no sampled-turn render */
    return r->family->assistant_turn_sampled(r, think, replay ? (reasoning ? reasoning : "") : NULL, content, calls,
                                             spans_out, n_spans_out);
}

char *build_responses_visible_assistant_suffix(const request *r,
                                                      const char *content,
                                                      const char *reasoning,
                                                      const tool_calls *calls) {
    return build_responses_visible_assistant_suffix_spans(r, content, reasoning, calls,
                                                          NULL, NULL);
}



bool server::should_canonicalize_tool_checkpoint(const tool_calls *calls) const {
    if (!calls || calls->len == 0) return false;
    return !(calls->raw_dsml && calls->raw_dsml[0]);
}


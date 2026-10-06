/* parser_deepseek.cpp -- DeepSeek's output parser behind server_output_parser_ops (L272 P3 steps 2-3).
 *
 * The generated text is "<think>reasoning</think>answer" with DSML tool-call blocks; this is every
 * DeepSeek-specific reading of it the server used to do inline in server_jobs.cpp: the decode-time
 * DSML tracker (the greedy region of a tool call), the marker scan that ends a turn at a closed tool
 * block (with its think gate: a block inside reasoning is not a call unless it is complete), the live
 * projection into the protocol sink (deepseek_stream.cpp's one walk), and the finish -- the tag repair
 * of a truncated block, the final DSML parse, the model-visible tool-error retry when the block is
 * malformed, the stream's ids onto the parsed calls, tool memory.  Nothing here is reached except
 * through k_parser_deepseek. */
#include "pulsar_server_internal.h"

static void *ds_parser_create(server *, gen_state *g, char *, size_t) {
    job *j = g->j;
    deepseek_parser *ps = (deepseek_parser *)server_xmalloc(sizeof *ps);
    memset(ps, 0, sizeof *ps);
    /* the live projection into the sink, for a streamed chat on any protocol */
    ps->walk_on = g->sink.text != NULL;
    if (ps->walk_on) deepseek_stream_walk_init(&ps->walk, &j->req);
    dsml_decode_tracker_init(&ps->tracker);
    ps->next_tool_progress = 128;
    /* Tool markers inside a reasoning block are NOT tool calls -- the model is thinking about calling
     * something -- so marker scanning waits for the block to close. */
    ps->thinking_gates_tool_markers = pulsar_think_mode_enabled(j->req.think_mode);
    ps->tool_scan_waiting_for_think_close = ps->thinking_gates_tool_markers && g->thinking.inside;
    return ps;
}

static void ds_parser_destroy(void *st) {
    deepseek_parser *ps = (deepseek_parser *)st;
    if (!ps) return;
    deepseek_stream_walk_free(&ps->walk);
    free(ps);
}

/* tool_choice="required": the prompt was prefilled into an open DSML tool_calls block and g->text
 * seeded with that exact prefix (including the closing </think>, so the thinking-mode parser sees
 * reasoning end and a complete tool block); the trackers start "inside tool call" -- the model now
 * generates only the invoke body. */
static void ds_parser_seed(void *st, gen_state *g) {
    deepseek_parser *ps = (deepseek_parser *)st;
    ps->saw_tool_start = true;
    ps->tool_scan_waiting_for_think_close = false;
    dsml_decode_tracker_update(&ps->tracker, g->text.ptr, g->text.len);
    ps->tool_scan_from = g->text.len;
}

static bool ds_parser_in_tool_call(const void *st, const gen_state *) {
    const deepseek_parser *ps = (const deepseek_parser *)st;
    const dsml_decode_state d = ps->tracker.decode;
    return dsml_decode_state_is_tool(d) && !dsml_decode_state_uses_payload_sampling(d);
}

/* A closed DSML block ends the turn (finish "tool_calls"). */
static bool ds_parser_turn_complete(const void *st, const gen_state *) {
    return ((const deepseek_parser *)st)->saw_tool_end;
}

static void ds_parser_tool_progress(const void *st, bool *opened, bool *closed) {
    const deepseek_parser *ps = (const deepseek_parser *)st;
    *opened = ps->saw_tool_start;
    *closed = ps->saw_tool_end;
}

/* Per emitted token (upto = the bytes of g->text the stop-string scan released): the decode tracker,
 * the live projection, then the marker scan.  `final` is the stream's last flush at the finish. */
static bool ds_parser_feed(void *st, server *s, gen_state *g, size_t upto, bool final) {
    deepseek_parser *ps = (deepseek_parser *)st;
    job *j = g->j;
    if (final) {
        return !ps->walk_on || deepseek_stream_update(&ps->walk, &g->sink, g->text.ptr ? g->text.ptr : "",
                                                       g->text.len, true);
    }
    if (j->req.kind == REQ_CHAT && j->req.has_tools)
        dsml_decode_tracker_update(&ps->tracker, g->text.ptr, g->text.len);
    /* DeepSeek's one walk over its raw text (L267) */
    if (ps->walk_on && !deepseek_stream_update(&ps->walk, &g->sink, g->text.ptr, upto, false)) return false;

    if (j->req.kind != REQ_CHAT || !j->req.has_tools) return true;
    if (ps->thinking_gates_tool_markers && g->thinking.inside) {
        /* A DSML block inside reasoning is not executable, and an opening
         * marker alone can be quoted protocol text. A COMPLETE block is
         * unambiguous enough to recover: stop with finish=tool_calls and
         * let the parse-side recovery return the call structurally
         * (upstream ds4 51a1c14). */
        if (complete_tool_call_inside_thinking(g->text.ptr, g->text.len, &ps->think_recovery_scan_from)) {
            ps->saw_tool_start = true;
            ps->saw_tool_end = true;
            server_log(PULSAR_LOG_WARNING,
                       "pulsar-server: chat ctx=%s%s%s recovered a complete tool call "
                       "from unclosed reasoning after %d generated tokens",
                       g->ctx_span,
                       g->req_flags[0] ? " " : "",
                       g->req_flags,
                       g->completion);
            s->trace_event(g->trace_id,
                        "recovered complete tool call from unclosed reasoning after %d generated tokens",
                        g->completion);
        } else {
            ps->tool_scan_waiting_for_think_close = true;
            ps->tool_scan_from = g->text.len;
        }
        return true;
    }
    if (ps->tool_scan_waiting_for_think_close) {
        const char *think_end = find_last_substr(g->text.ptr, "</think>");
        ps->tool_scan_from = think_end ? (size_t)((think_end + 8) - g->text.ptr) : g->text.len;
        if (ps->tool_scan_from > g->text.len) ps->tool_scan_from = g->text.len;
        ps->tool_scan_waiting_for_think_close = false;
    }
    if (ps->tool_scan_from > g->text.len) ps->tool_scan_from = g->text.len;
    const char *tool_scan = g->text.ptr ? g->text.ptr + ps->tool_scan_from : "";
    bool orphan_end = false;
    bool old_start = ps->saw_tool_start;
    bool old_end = ps->saw_tool_end;
    observe_tool_markers(tool_scan, &ps->saw_tool_start, &ps->saw_tool_end, &orphan_end);
    if (orphan_end && !ps->saw_orphan_tool_end) {
        ps->saw_orphan_tool_end = true;
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: chat ctx=%s%s%s ignored orphan tool-call end marker after %d generated tokens",
                   g->ctx_span,
                   g->req_flags[0] ? " " : "",
                   g->req_flags,
                   g->completion);
        s->trace_event(g->trace_id,
                    "ignored orphan tool-call end marker after %d generated tokens",
                    g->completion);
    }
    if (ps->saw_tool_start && !old_start) {
        s->trace_event(g->trace_id, "entered tool-call block after %d generated tokens", g->completion);
    }
    if (ps->saw_tool_end && !old_end) {
        s->trace_event(g->trace_id, "closed tool-call block after %d generated tokens", g->completion);
    }
    const size_t marker_hold = 80;
    size_t hold_from = g->text.len > marker_hold ? g->text.len - marker_hold : 0;
    if (hold_from > ps->tool_scan_from) ps->tool_scan_from = hold_from;
    if (s->trace && g->completion >= ps->next_tool_progress) {
        s->trace_event(g->trace_id,
                    "progress gen=%d dsml_start=%d dsml_end=%d",
                    g->completion, ps->saw_tool_start ? 1 : 0, ps->saw_tool_end ? 1 : 0);
        ps->next_tool_progress += 128;
    }
    return true;
}

/* The model-visible retry (non-streaming, once per request): a tool error plus the prompt reminder
 * appended to the live session, and the generation loops (out->retry).  false = it could not run;
 * the caller's finish is an error then. */
static bool ds_parser_retry(deepseek_parser *, server *s, session_slot *sl, gen_state *g, const char *detail,
                            const char *why, server_turn *out) {
    job *j = g->j;
    int recovery_tokens = 0;
    char recovery_err[160] = {0};
    server_log(PULSAR_LOG_WARNING, "pulsar-server: chat ctx=%s%s%s %s; continuing with model-visible tool error",
               g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags, why);
    s->trace_event(g->trace_id, "%s; continuing with model-visible tool error", why);
    if (!s->continue_after_invalid_dsml(sl, &j->req, &g->thinking, detail, &recovery_tokens, recovery_err,
                                        sizeof(recovery_err))) {
        g->finish = "error";
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

/* The turn's final reading: the tag repair of a truncated block, the final DSML parse (a malformed
 * block promotes the raw text to content, or loops the generation through the retry), the stream's ids
 * onto the parsed calls, tool memory, the finish label. */
static bool ds_parser_finish(void *st, server *s, session_slot *sl, gen_state *g, server_turn *out) {
    deepseek_parser *ps = (deepseek_parser *)st;
    job *j = g->j;
    const bool retry_allowed = !j->req.stream && !g->recovery_attempted;

    /* L077 (tool-call truncation): when the tag repair below completes a
     * LENGTH-CAPPED call, the emitted arguments are well-formed JSON with a
     * silently cut-off value -- the finish reason is the client's ONLY signal
     * that the turn was cut (openai_stream.cpp's finalize comment already
     * states this contract; the unconditional "tool_calls" relabel broke it). */
    bool truncated_tool_repair = false;
    if (j->req.kind == REQ_CHAT && j->req.has_tools &&
        ps->saw_tool_start && !ps->saw_tool_end && strcmp(g->finish, "error") != 0)
    {
        /* Deterministically complete a simple truncation.  Anything more than
         * missing closing tags stays model-owned: for non-streaming requests,
         * append a tool error plus prompt reminder to the live session and let
         * the model issue a fresh call. */
        bool completed_truncation = false;
        buf repaired = {0};
        if (try_repair_dsml(g->text.ptr, g->text.len, &repaired)) {
            /* Parse repaired text to verify it produces valid tool calls */
            tool_calls test_calls = {0};
            char *test_content = NULL;
            char *test_reasoning = NULL;
            bool repair_ok = parse_generated_message_ex(repaired.ptr, false, &test_content, &test_reasoning, &test_calls);
            free(test_content);
            free(test_reasoning);
            if (repair_ok && test_calls.len > 0) {
                /* Repair succeeded - replace text with repaired version */
                free(g->text.ptr);
                g->text.ptr = buf_take(&repaired);
                g->text.len = strlen(g->text.ptr);
                g->text.cap = g->text.len ? g->text.len + 1 : 0;
                ps->saw_tool_end = true;
                completed_truncation = true;
                if (strcmp(g->finish, "length") == 0) truncated_tool_repair = true;
                server_log(PULSAR_LOG_WARNING,
                           "pulsar-server: chat ctx=%s%s%s repaired unterminated tool call (%d calls recovered)",
                           g->ctx_span,
                           g->req_flags[0] ? " " : "",
                           g->req_flags,
                           test_calls.len);
                s->trace_event(g->trace_id, "repaired unterminated tool call (%d calls recovered)", test_calls.len);
            }
            tool_calls_free(&test_calls);
        }
        buf_free(&repaired);
        if (!completed_truncation) {
            if (retry_allowed) {
                if (ds_parser_retry(ps, s, sl, g, "unterminated tool call", "unterminated tool call", out)) return true;
            } else {
                g->finish = "error";
                snprintf(g->err, sizeof(g->err), "unterminated tool call");
            }
        }
    }

    out->finish = g->finish;
    if (j->req.kind != REQ_CHAT) return true;
    bool parse_failed = false;
    bool parsed_ok = parse_generated_message_for_response(
        g->text.ptr ? g->text.ptr : "",
        j->req.has_tools,
        ps->saw_tool_start,
        pulsar_think_mode_enabled(j->req.think_mode),
        &out->finish,
        g->err,
        sizeof(g->err),
        &out->content,
        &out->reasoning,
        &out->calls,
        &parse_failed);
    if (!parsed_ok && parse_failed && j->req.has_tools && ps->saw_tool_start) {
        /* parse_generated_message failed even though DSML was present.
         * Semantic repair is intentionally avoided: if the parser cannot
         * execute the block, feed the model a tool error and the protocol
         * reminder so it owns the corrected next action. */
        if (retry_allowed) {
            const char *detail = g->err[0] ? g->err : "invalid tool call";
            if (ds_parser_retry(ps, s, sl, g, detail, "invalid tool call", out)) {
                free(out->content);
                free(out->reasoning);
                out->content = out->reasoning = NULL;
                tool_calls_free(&out->calls);
                return true;
            }
            out->finish = "error";
        }
        if (!parsed_ok) {
            /* Print raw DSML snippet for debugging */
            size_t dsml_snippet_len = 0;
            const char *dsml_start = NULL;
            const char *p;
            /* g->text.len - 20 underflows (size_t) when the text is under
             * 20 bytes -- a bare 19-byte short tool-call marker with no
             * body reaches here -- making the bound ~2^64 and walking the
             * strncmp off the heap buffer. Scan every valid start offset
             * instead; g->text is a NUL-terminated buf, so each strncmp is
             * self-bounded at the terminator. */
            for (p = g->text.ptr; p && (size_t)(p - g->text.ptr) < g->text.len; p++) {
                if ((strncmp(p, PULSAR_TOOL_CALLS_START, strlen(PULSAR_TOOL_CALLS_START)) == 0) ||
                    (strncmp(p, PULSAR_TOOL_CALLS_START_SHORT, strlen(PULSAR_TOOL_CALLS_START_SHORT)) == 0) ||
                    (strncmp(p, "<tool_calls>", 12) == 0)) {
                    dsml_start = p;
                    break;
                }
            }
            if (dsml_start) {
                dsml_snippet_len = g->text.len - (dsml_start - g->text.ptr);
                if (dsml_snippet_len > 500) dsml_snippet_len = 500;
            }
            /* Also log a snippet of the full text to see what the model output */
            size_t text_snippet_len = g->text.len > 300 ? 300 : g->text.len;
            server_log(PULSAR_LOG_WARNING,
                       "pulsar-server: chat ctx=%s%s%s invalid tool call returned as assistant text finish=%s [text_len=%zu saw_start=%d saw_end=%d text_snippet: %.*s]",
                       g->ctx_span,
                       g->req_flags[0] ? " " : "",
                       g->req_flags,
                       out->finish,
                       g->text.len,
                       ps->saw_tool_start,
                       ps->saw_tool_end,
                       (int)text_snippet_len,
                       g->text.ptr ? g->text.ptr : "(null)");
            server_log(PULSAR_LOG_WARNING,
                       "pulsar-server: chat ctx=%s%s%s invalid tool call dsml_snippet: %.*s",
                       g->ctx_span,
                       g->req_flags[0] ? " " : "",
                       g->req_flags,
                       (int)dsml_snippet_len,
                       dsml_start ? dsml_start : "(none)");
            s->trace_event(g->trace_id,
                        "invalid tool call returned as assistant text finish=%s",
                        out->finish);
        }
    }
    out->parse_failed = parse_failed;
    if (parse_failed && ps->walk_on && ps->walk.emit_pos < g->text.len) {
        /* the walk stopped at the block's marker; the final reading returned the rest to content */
        out->tail = g->text.ptr + ps->walk.emit_pos;
        out->tail_len = g->text.len - ps->walk.emit_pos;
    }
    if (out->calls.len) {
        if (ps->walk_on) apply_stream_tool_ids(&out->calls, &ps->walk.tool);
        s->assign_tool_call_ids(&out->calls, j->req.api);
        s->tool_memory_remember(&out->calls);
        /* L077: a length-capped, tag-repaired call reports "length" -- the
         * repaired calls are still emitted (replayed transcripts stay
         * parseable), but the label must not claim a complete call. */
        out->finish = truncated_tool_repair ? "length" : "tool_calls";
    }
    return true;
}

const server_output_parser_ops k_parser_deepseek = {
    /* .create        = */ ds_parser_create,
    /* .destroy       = */ ds_parser_destroy,
    /* .seed          = */ ds_parser_seed,
    /* .feed          = */ ds_parser_feed,
    /* .in_tool_call  = */ ds_parser_in_tool_call,
    /* .turn_complete = */ ds_parser_turn_complete,
    /* .tool_progress = */ ds_parser_tool_progress,
    /* .finish        = */ ds_parser_finish,
};

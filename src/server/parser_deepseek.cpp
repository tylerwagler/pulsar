/* parser_deepseek.cpp -- DeepSeek's output parser behind server_output_parser_ops (L272 P3 steps 2-3).
 *
 * The generated text is "<think>reasoning</think>answer" with DSML tool-call blocks; this is every
 * DeepSeek-specific reading of it the server used to do inline in server_jobs.cpp: the decode-time
 * DSML tracker (the greedy region of a tool call), the marker scan that ends a turn at a closed tool
 * block (with its think gate: a block inside reasoning is not a call unless it is complete), the live
 * projection into the protocol sink (deepseek_stream.cpp's one walk), and the finish -- the final DSML
 * parse, the shared rule for a broken call (L284 P4: retry or text, never a repaired tag), the stream's
 * ids onto the parsed calls, tool memory.  Nothing here is reached except through k_parser_deepseek. */
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

/* The turn's final reading (L284 P4, the rule every family shares -- see turn_tool_retry_allowed): the
 * DSML parse of the block (an unterminated one -- the cap cut it -- is not parsed and never repaired), the
 * stream's ids onto the parsed calls; then the finish every family shares (parser_finish_turn: the
 * undeclared-call drop, the retry or text, tool memory, the finish label). */
static bool ds_parser_finish(void *st, server *s, session_slot *sl, gen_state *g, server_turn *out) {
    deepseek_parser *ps = (deepseek_parser *)st;
    job *j = g->j;
    out->finish = g->finish;
    if (j->req.kind != REQ_CHAT) return true;
    const bool block = j->req.has_tools && ps->saw_tool_start;
    const bool unterminated = block && !ps->saw_tool_end;
    const char *why = "";
    bool broken = unterminated;
    if (unterminated) {
        why = "unterminated tool call (the turn ended inside the tool_calls block)";
    } else if (!parse_generated_message_ex(g->text.ptr ? g->text.ptr : "", pulsar_think_mode_enabled(j->req.think_mode),
                                           &out->content, &out->reasoning, &out->calls)) {
        free(out->content);
        free(out->reasoning);
        out->content = out->reasoning = NULL;
        tool_calls_free(&out->calls);
        broken = block;
        why = "invalid tool call";
    }
    if (out->calls.len && ps->walk_on) apply_stream_tool_ids(&out->calls, &ps->walk.tool);
    /* as text, the walk stopped at the block's marker: the rest goes out at the finish */
    const size_t tail_from = ps->walk_on ? ps->walk.emit_pos : SIZE_MAX;
    return parser_finish_turn(s, sl, g, out, broken, why, /*logged=*/false, tail_from, true);
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

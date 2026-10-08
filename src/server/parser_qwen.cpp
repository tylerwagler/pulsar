/* parser_qwen.cpp -- Qwen's output parser behind server_output_parser_ops (L272 P3 steps 2-3; the
 * reading itself is src/lib/qwen_output.cpp, L251).  ONE qwen_output_parser per decode attempt is
 * fed g->text's bytes as the stop-string scan releases them, and its events become the response:
 * reasoning and content deltas into the protocol sink, each completed call given its id and handed to
 * the sink whole through the generic tool events (a protocol without live events sends it with the
 * finish).  A call reaches the client only when complete (TOOL_END): a call the parser later rejects
 * (ERROR) was never announced, so a stream never carries a call the final message lacks.  A Qwen turn
 * ends at the family's stop token, so it never ends at a closed call.  A malformed call is the model's
 * output, not a server fault: logged and dropped by the parser -- and, when the turn had no valid call
 * and the request is not streaming, retried once with a model-visible tool error (step 4: the family's
 * tool_error_suffix).  The turn's calls as sampled (the parser's raw span of g->text) are the tool
 * memory's key, replayed verbatim by the renderer so the next prompt byte-matches the live KV. */
#include "pulsar_server_internal.h"

qwen_gen::~qwen_gen() { tool_calls_free(&calls); }

/* The forced-call seed (g->text's first bytes) read before the model's first token: the think close
 * and the open <tool_call>, text-free -- no event but the call's announcement can come of it. */
static void qwen_parser_seed(void *st, gen_state *g) {
    qwen_gen *q = (qwen_gen *)st;
    q->ev.clear();
    q->parser.feed(g->text.ptr, g->text.len, &q->ev);
    q->fed = g->text.len;
    for (const qwen_out_event &e : q->ev)
        if (e.kind != qwen_out_event::TOOL_BEGIN) pulsar_die("qwen parser: the forced-call seed is not an open call");
}

static void *qwen_parser_create(server *, gen_state *g, char *err, size_t errlen) {
    job *j = g->j;
    qwen_gen *q = new qwen_gen();
    /* its tools are the ones the prompt rendered (they type the arguments); it starts in reasoning
     * exactly when the generation prompt opened "<think>\n" */
    char perr[200];
    if (!q->parser.init(pulsar_think_mode_enabled(j->req.think_mode), j->req.qwen_tools_json, perr, sizeof perr)) {
        snprintf(err, errlen, "qwen output parser: %s", perr);
        delete q;
        return NULL;
    }
    return q;
}

static void qwen_parser_destroy(void *st) { delete (qwen_gen *)st; }

static bool qwen_parser_in_tool_call(const void *st, const gen_state *) {
    const qwen_output_parser &p = ((const qwen_gen *)st)->parser;
    return p.in_tool_call() && !p.in_payload();
}

static bool qwen_parser_turn_complete(const void *, const gen_state *) { return false; }

static void qwen_parser_tool_progress(const void *st, bool *opened, bool *closed) {
    const qwen_gen *q = (const qwen_gen *)st;
    *opened = q->calls.len > 0 || q->parser.in_tool_call();
    *closed = q->calls.len > 0;
}

/* Feed the parser g->text[fed, upto) and, when `final`, end the turn; then turn its events into the
 * response.  Reasoning and content go out as deltas when streaming (the parser keeps both for the
 * final message either way).  false = a client write failed. */
static bool qwen_parser_feed(void *st, server *s, gen_state *g, size_t upto, bool final) {
    qwen_gen *q = (qwen_gen *)st;
    job *j = g->j;
    if (q->finished) return q->stream_ok;   /* the finish fed the end of the turn already */
    q->ev.clear();
    if (upto > q->fed) {
        q->parser.feed(g->text.ptr + q->fed, upto - q->fed, &q->ev);
        q->fed = upto;
    }
    if (final) {
        q->parser.finish(&q->ev);
        q->finished = true;
    }
    const bool stream = j->req.stream;
    for (const qwen_out_event &e : q->ev) {
        switch (e.kind) {
        case qwen_out_event::REASONING:
        case qwen_out_event::CONTENT:
            if (stream && g->sink.text &&
                !g->sink.text(&g->sink, e.kind == qwen_out_event::REASONING, e.text.data(), e.text.size(), q->fed))
                return q->stream_ok = false;
            break;
        case qwen_out_event::TOOL_BEGIN:
            break;   /* a call goes out whole, at TOOL_END */
        case qwen_out_event::TOOL_END: {
            char undeclared[512];
            if (!tool_call_declared(&j->req, e.name.c_str(), undeclared, sizeof(undeclared))) {
                /* a malformed call (L272): never announced, so the stream and the final message agree */
                server_log(PULSAR_LOG_WARNING, "pulsar-server: chat ctx=%s%s%s qwen output: %s",
                           g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags, undeclared);
                s->trace_event(g->trace_id, "qwen output: %s", undeclared);
                q->last_error = undeclared;
                q->undeclared++;
                break;
            }
            tool_call tc = {0};
            tc.name = xstrdup(e.name.c_str());
            tc.arguments = xstrdup(e.arguments.c_str());
            tool_calls_push(&q->calls, tc);
            s->assign_tool_call_ids(&q->calls, j->req.api);
            /* a call handed over whole, as the protocol's live tool events: the open section ends, then
             * the call's header, its whole argument object and its end (OpenAI: the start delta and one
             * arguments delta; Anthropic: one tool_use block) -- a protocol without live events sends
             * it with the finish */
            if (stream && g->sink.tool_ops) {
                chat_sink *k = &g->sink;
                const int idx = q->calls.len - 1;
                const tool_call *c = &q->calls.v[idx];
                const char *a = c->arguments ? c->arguments : "";
                if (!k->end(k, false) || !k->tool_ops->begin(k, idx, c->id, c->name) ||
                    !k->tool_ops->args(k, idx, a, strlen(a)) || !k->tool_ops->end(k, idx))
                    return q->stream_ok = false;
            }
            break;
        }
        case qwen_out_event::ERROR:
            server_log(PULSAR_LOG_WARNING, "pulsar-server: chat ctx=%s%s%s qwen output: %s",
                       g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags, e.text.c_str());
            s->trace_event(g->trace_id, "qwen output: %s", e.text.c_str());
            q->last_error = e.text;
            break;
        }
    }
    return true;
}

/* The turn's final reading: the rest of the text (a stop string's held tail never reaches it: g->text
 * was cut at the match) and the end of the turn; the parser holds reasoning, content and the completed
 * calls in the template's normal form, and the calls' sampled bytes become their tool memory.  A turn
 * whose every call was malformed loops the generation once through the model-visible tool error
 * (non-streaming: a stream already carried the text; not after a forced call: its seed would be
 * re-read as text the continued KV does not hold). */
static bool qwen_parser_finish(void *st, server *s, session_slot *sl, gen_state *g, server_turn *out) {
    qwen_gen *q = (qwen_gen *)st;
    job *j = g->j;
    out->finish = g->finish;
    bool ok = true;
    if (strcmp(g->finish, "error") != 0) ok = qwen_parser_feed(q, s, g, g->text.len, true);
    if ((q->parser.errors() > 0 || q->undeclared > 0) && q->calls.len == 0 && strcmp(g->finish, "error") != 0 && !j->req.stream &&
        !g->recovery_attempted && !j->req.force_tool_call && j->req.has_tools)
    {
        int recovery_tokens = 0;
        char recovery_err[160] = {0};
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: chat ctx=%s%s%s malformed tool call; continuing with model-visible tool error",
                   g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags);
        s->trace_event(g->trace_id, "malformed tool call; continuing with model-visible tool error");
        if (!s->continue_after_invalid_dsml(sl, &j->req, &g->thinking, q->last_error.c_str(), &recovery_tokens,
                                            recovery_err, sizeof(recovery_err))) {
            g->finish = "error";
            out->finish = g->finish;
            snprintf(g->err, sizeof(g->err), "malformed tool call recovery failed: %s",
                     recovery_err[0] ? recovery_err : "unknown error");
            return false;
        }
        server_log(PULSAR_LOG_GENERATION,
                   "pulsar-server: chat ctx=%s%s%s tool-error continuation appended %d tokens",
                   g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags, recovery_tokens);
        s->trace_event(g->trace_id, "tool-error continuation appended %d tokens", recovery_tokens);
        out->retry = true;
        return true;
    }
    out->content = xstrdup(q->parser.content().c_str());
    out->reasoning = q->parser.reasoning().empty() ? NULL : xstrdup(q->parser.reasoning().c_str());
    out->calls = q->calls;
    memset(&q->calls, 0, sizeof(q->calls));
    size_t raw_lo = 0, raw_hi = 0;
    /* the sampled bytes are the tool memory's key only when they are exactly the kept calls: a dropped
     * undeclared call is in them, so that turn re-renders canonically (a prefix miss, never a wrong prompt) */
    if (q->undeclared == 0 && q->parser.raw_span(&raw_lo, &raw_hi)) {
        if (raw_hi > g->text.len || raw_lo >= raw_hi) pulsar_die("qwen parser: raw span outside the turn's text");
        out->calls.raw_dsml = xstrndup(g->text.ptr + raw_lo, raw_hi - raw_lo);
        s->tool_memory_remember(&out->calls);
    }
    if (out->calls.len && strcmp(out->finish, "error") != 0) out->finish = "tool_calls";
    return ok;
}

const server_output_parser_ops k_parser_qwen = {
    /* .create        = */ qwen_parser_create,
    /* .destroy       = */ qwen_parser_destroy,
    /* .seed          = */ qwen_parser_seed,
    /* .feed          = */ qwen_parser_feed,
    /* .in_tool_call  = */ qwen_parser_in_tool_call,
    /* .turn_complete = */ qwen_parser_turn_complete,
    /* .tool_progress = */ qwen_parser_tool_progress,
    /* .finish        = */ qwen_parser_finish,
};

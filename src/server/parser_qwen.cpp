/* parser_qwen.cpp -- Qwen's output parser behind server_output_parser_ops (L272 P3 steps 2-3; the
 * reading itself is src/lib/qwen_output.cpp, L251).  ONE qwen_output_parser per decode attempt is
 * fed g->text's bytes as the stop-string scan releases them, and its events become the response:
 * reasoning and content deltas into the protocol sink, and each call as it is read (L284 P5) through
 * the generic tool events -- announced when its name is complete and declared, each parameter's
 * `"key": value` when the parameter closes, the object's close at the call's close (a protocol without
 * live events sends the calls with the finish).  A Qwen turn ends at the family's stop token, so it
 * never ends at a closed call.  A broken call (malformed, cut by the cap, or naming an undeclared tool)
 * takes the rule every family shares (L284 P4, turn_tool_retry_allowed): an announced one is closed on
 * the stream as sent, the finish drops it, and a turn left with no valid call retries through the
 * model-visible tool error when allowed, else it is text.  The turn's calls as sampled (the parser's
 * raw span of g->text) are the tool memory's key, replayed verbatim by the renderer so the next prompt
 * byte-matches the live KV. */
#include "pulsar_server_internal.h"

qwen_gen::~qwen_gen() { tool_calls_free(&calls); }

/* The forced-call seed (g->text's first bytes) read before the model's first token: the think close
 * and the open <tool_call>, text-free -- a named seed announces its call, which the first feed sends. */
static void qwen_parser_seed(void *st, gen_state *g) {
    qwen_gen *q = (qwen_gen *)st;
    q->ev.clear();
    q->parser.feed(g->text.ptr, g->text.len, &q->ev);
    q->fed = g->text.len;
    for (const qwen_out_event &e : q->ev)
        if (e.kind != qwen_out_event::TOOL_BEGIN) pulsar_die("qwen parser: the forced-call seed is not an open call");
    q->pending = q->ev;
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

/* the greedy region (L284 P6): a call's structure; its payload is sampled */
static bool qwen_parser_in_tool_call(const void *st, const gen_state *) {
    const qwen_output_parser &p = ((const qwen_gen *)st)->parser;
    return p.in_tool_call() && !p.in_payload();
}

static bool qwen_parser_turn_complete(const void *, const gen_state *) { return false; }

static void qwen_parser_tool_progress(const void *st, bool *opened, bool *closed) {
    const qwen_gen *q = (const qwen_gen *)st;
    *opened = q->calls.len > 0 || q->parser.in_tool_call();
    *closed = q->calls.len > (q->open ? 1 : 0);
}

/* The announced open call's close on the stream: the argument object as far as it was sent, then the
 * protocol's end of the call. */
static bool qwen_close_open_call(qwen_gen *q, gen_state *g) {
    chat_sink *k = &g->sink;
    return k->tool_ops->args(k, q->wire_open, q->open_args ? "}" : "{}", q->open_args ? 1 : 2) &&
           k->tool_ops->end(k, q->wire_open);
}

/* Feed the parser g->text[fed, upto) and, when `final`, end the turn; then turn its events into the
 * response.  Reasoning and content go out as deltas when streaming (the parser keeps both for the
 * final message either way).  false = a client write failed. */
static bool qwen_parser_feed(void *st, server *s, gen_state *g, size_t upto, bool final) {
    qwen_gen *q = (qwen_gen *)st;
    job *j = g->j;
    if (q->finished) return q->stream_ok;   /* the finish fed the end of the turn already */
    q->ev.swap(q->pending);                 /* a seed's announcement goes out with the first feed */
    q->pending.clear();
    if (upto > q->fed) {
        q->parser.feed(g->text.ptr + q->fed, upto - q->fed, &q->ev);
        q->fed = upto;
    }
    if (final) {
        q->parser.finish(&q->ev);
        q->finished = true;
    }
    const bool live = j->req.stream && g->sink.tool_ops;
    chat_sink *k = &g->sink;
    for (const qwen_out_event &e : q->ev) {
        switch (e.kind) {
        case qwen_out_event::REASONING:
        case qwen_out_event::CONTENT:
            if (j->req.stream && k->text &&
                !k->text(k, e.kind == qwen_out_event::REASONING, e.text.data(), e.text.size(), q->fed))
                return q->stream_ok = false;
            break;
        case qwen_out_event::TOOL_BEGIN: {
            char undeclared[512];
            if (!tool_call_declared(&j->req, e.name.c_str(), undeclared, sizeof(undeclared))) {
                /* a broken call (L272): never announced; TOOL_END or ERROR counts it */
                q->last_error = undeclared;
                break;
            }
            tool_call tc = {0};
            tc.name = xstrdup(e.name.c_str());
            tool_calls_push(&q->calls, tc);
            s->assign_tool_call_ids(&q->calls, j->req.api);   /* the id the stream shows is the call's */
            q->open = true;
            q->open_args = false;
            q->wire_open = q->wire_n++;   /* the stream's index: a broken call keeps its own */
            if (live) {
                const tool_call *c = &q->calls.v[q->calls.len - 1];
                if (!k->end(k, false) || !k->tool_ops->begin(k, q->wire_open, c->id, c->name))
                    return q->stream_ok = false;
            }
            break;
        }
        case qwen_out_event::TOOL_ARGS:
            if (!q->open) break;
            if (live && !k->tool_ops->args(k, q->wire_open, e.text.data(), e.text.size()))
                return q->stream_ok = false;
            q->open_args = true;
            break;
        case qwen_out_event::TOOL_END:
            if (!q->open) {
                server_log(PULSAR_LOG_WARNING, "pulsar-server: chat ctx=%s%s%s qwen output: %s",
                           g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags, q->last_error.c_str());
                s->trace_event(g->trace_id, "qwen output: %s", q->last_error.c_str());
                q->undeclared++;
                break;
            }
            q->calls.v[q->calls.len - 1].arguments = xstrdup(e.arguments.c_str());
            if (live && !qwen_close_open_call(q, g)) return q->stream_ok = false;
            q->open = false;
            break;
        case qwen_out_event::ERROR:
            server_log(PULSAR_LOG_WARNING, "pulsar-server: chat ctx=%s%s%s qwen output: %s",
                       g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags, e.text.c_str());
            s->trace_event(g->trace_id, "qwen output: %s", e.text.c_str());
            q->last_error = e.text;
            if (q->open) {
                /* announced, then broken: closed on the stream as sent, and not a call */
                if (live && !qwen_close_open_call(q, g)) return q->stream_ok = false;
                tool_call *c = &q->calls.v[--q->calls.len];
                free(c->id);
                free(c->name);
                free(c->arguments);
                memset(c, 0, sizeof(*c));
                q->open = false;
            }
            break;
        }
    }
    return true;
}

/* The turn's final reading: the rest of the text (a stop string's held tail never reaches it: g->text
 * was cut at the match) and the end of the turn; the parser holds reasoning, content and the completed
 * calls in the template's normal form, and the calls' sampled bytes become their tool memory.  A turn
 * with a broken call and no valid one takes the shared rule (turn_tool_retry_allowed). */
static bool qwen_parser_finish(void *st, server *s, session_slot *sl, gen_state *g, server_turn *out) {
    qwen_gen *q = (qwen_gen *)st;
    job *j = g->j;
    out->finish = g->finish;
    if (j->req.kind != REQ_CHAT) return true;
    bool ok = true;
    if (strcmp(g->finish, "error") != 0) ok = qwen_parser_feed(q, s, g, g->text.len, true);
    if ((q->parser.errors() > 0 || q->undeclared > 0) && q->calls.len == 0) {
        if (turn_tool_retry_allowed(g)) return turn_tool_retry(s, sl, g, q->last_error.c_str(), out);
        turn_as_text(g, out);
        const char *t = g->text.ptr ? g->text.ptr : "";
        const char *call = strstr(t, "<tool_call>");
        if (call) {
            /* the stream stopped at the call; the rest goes out as text at the finish */
            out->tail = call;
            out->tail_len = g->text.len - (size_t)(call - t);
        }
        return ok;
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
    out->finish = turn_finish(g, out->calls.len);
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

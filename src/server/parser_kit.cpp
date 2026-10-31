/* parser_kit.cpp -- the output-parser kit behind server_output_parser_ops (L272 P3 steps 2-3; the kit
 * since L295 F5): every hook but `create` is shared by the families whose parser is a configuration of
 * the kit (src/lib/out_parse_kit.h -- Qwen today).  A family's create builds its configured out_parser
 * and hands it to parser_kit_start.
 *
 * ONE out_parser per decode attempt is fed g->text's bytes as the stop-string scan releases them, and its
 * events become the response: reasoning and content deltas into the protocol sink, and each call as it is
 * read (L284 P5) through the generic tool events -- announced when its name is complete and declared, its
 * argument fragments as the parser releases them, the object's close at the call's close (a protocol
 * without live events sends the calls with the finish).  A kit turn ends at the family's stop token, so it
 * never ends at a closed call.  A broken call (malformed, cut by the cap, or naming an undeclared tool)
 * takes the rule every family shares (L284 P4, turn_tool_retry_allowed): an announced one is closed on the
 * stream as sent, the finish drops it, and a turn left with no valid call retries through the
 * model-visible tool error when allowed, else it is text.  The turn's calls as sampled (the parser's raw
 * span of g->text) are the tool memory's key, replayed verbatim by the renderer so the next prompt
 * byte-matches the live KV. */
/** @file
 *  The output-parser kit behind server_output_parser_ops: the hooks shared by every kit-configured family.
 */
#include "pulsar_server_internal.h"

kit_gen::~kit_gen() {
    tool_calls_free(&calls);
    delete parser;
}

void *parser_kit_start(out_parser *p) {
    kit_gen *q = new kit_gen();
    q->parser = p;
    return q;
}

/* The forced-call seed (g->text's first bytes) read before the model's first token: the family's "thinking
 * skipped, a call opened", text-free -- a named seed announces its call, which the first feed sends. */
/// server_output_parser_ops::seed: read the forced-call seed; dies if it is anything but an open call.
/// @param st  the kit state (a kit_gen, from parser_kit_start)
/// @param g   the attempt (g->text holds the seed)
void parser_kit_seed(void *st, gen_state *g) {
    kit_gen *q = (kit_gen *)st;
    q->ev.clear();
    q->parser->feed(g->text.ptr, g->text.len, &q->ev);
    q->fed = g->text.len;
    for (const out_event &e : q->ev)
        if (e.kind != out_event::TOOL_BEGIN) pulsar_die("output parser: the forced-call seed is not an open call");
    q->pending = q->ev;
}

/// server_output_parser_ops::destroy: free the kit state.
/// @param st  the kit state (a kit_gen, from parser_kit_start)
void parser_kit_destroy(void *st) { delete (kit_gen *)st; }

/* the greedy region (L284 P6): a call's structure; its payload is sampled */
/// server_output_parser_ops::in_tool_call: inside a call's structure and not its payload.
/// @param st  the kit state, or a probe (parser_kit_probe_new)
/// @return true when the next draw is greedy
bool parser_kit_in_tool_call(const void *st, const gen_state *) {
    const out_parser &p = *((const kit_gen *)st)->parser;
    return p.in_tool_call() && !p.in_payload();
}

/* L296 toolsample: the probe is a kit_gen around a copy of the parser's reading, at the same fed offset; it is fed
 * what parser_kit_feed would feed it and read only by parser_kit_in_tool_call (no sink, no calls, events dropped). */
/// server_output_parser_ops::probe_new.
/// @param st  the kit state (a kit_gen, from parser_kit_start)
/// @return a new probe (owned by the caller; parser_kit_probe_free)
void *parser_kit_probe_new(const void *st) {
    const kit_gen *q = (const kit_gen *)st;
    kit_gen *p = new kit_gen();
    p->parser = q->parser->clone();
    p->fed = q->fed;
    return p;
}

/// server_output_parser_ops::probe_feed: feed the probe text[fed, upto).
/// @param probe  the probe
/// @param text   the generated text with the round's tokens so far
/// @param upto   bytes the stop-string scan released (nothing is fed at or below what the probe already read)
void parser_kit_probe_feed(void *probe, const gen_state *, const char *text, size_t, size_t upto) {
    kit_gen *p = (kit_gen *)probe;
    if (upto <= p->fed) return;
    p->ev.clear();
    p->parser->feed(text + p->fed, upto - p->fed, &p->ev);
    p->fed = upto;
}

/// server_output_parser_ops::probe_free.
/// @param probe  the probe (parser_kit_probe_new)
void parser_kit_probe_free(void *probe) { delete (kit_gen *)probe; }

/// server_output_parser_ops::turn_complete: a kit turn ends only at the stop token.
/// @return false, always
bool parser_kit_turn_complete(const void *, const gen_state *) { return false; }

/// server_output_parser_ops::tool_progress (diagnostics).
/// @param st  the kit state (a kit_gen, from parser_kit_start)
/// @param opened  set when a call has been read or one is open
/// @param closed  set when a call has closed (one beyond the open one)
void parser_kit_tool_progress(const void *st, bool *opened, bool *closed) {
    const kit_gen *q = (const kit_gen *)st;
    *opened = q->calls.len > 0 || q->parser->in_tool_call();
    *closed = q->calls.len > (q->open ? 1 : 0);
}

/* The announced open call's close on the stream: the argument object as far as it was sent, then the
 * protocol's end of the call. */
/// @param q  the kit state with an announced open call (wire_open)
/// @param g  the attempt (its sink with tool_ops)
/// @return false when a client write failed
static bool kit_close_open_call(kit_gen *q, gen_state *g) {
    chat_sink *k = &g->sink;
    return k->tool_ops->args(k, q->wire_open, q->open_args ? "}" : "{}", q->open_args ? 1 : 2) &&
           k->tool_ops->end(k, q->wire_open);
}

/* Feed the parser g->text[fed, upto) and, when `final`, end the turn; then turn its events into the
 * response.  Reasoning and content go out as deltas when streaming (the parser keeps both for the
 * final message either way).  false = a client write failed. */
/// server_output_parser_ops::feed.
/// @param st  the kit state (a kit_gen, from parser_kit_start)
/// @param s      the server (call ids, the trace)
/// @param g      the attempt
/// @param upto   bytes of g->text released by the stop-string scan
/// @param final  end the turn after feeding
/// @return false when a client write failed (sticky: later calls after the finish return it)
bool parser_kit_feed(void *st, server *s, gen_state *g, size_t upto, bool final) {
    kit_gen *q = (kit_gen *)st;
    job *j = g->j;
    if (q->finished) return q->stream_ok;   /* the finish fed the end of the turn already */
    q->ev.swap(q->pending);                 /* a seed's announcement goes out with the first feed */
    q->pending.clear();
    if (upto > q->fed) {
        q->parser->feed(g->text.ptr + q->fed, upto - q->fed, &q->ev);
        q->fed = upto;
    }
    if (final) {
        q->parser->finish(&q->ev);
        q->finished = true;
    }
    const char *family = q->parser->grammar().name;
    const bool live = j->req.stream && g->sink.tool_ops;
    chat_sink *k = &g->sink;
    for (const out_event &e : q->ev) {
        switch (e.kind) {
        case out_event::REASONING:
        case out_event::CONTENT:
            if (j->req.stream && k->text &&
                !k->text(k, e.kind == out_event::REASONING, e.text.data(), e.text.size(), q->fed))
                return q->stream_ok = false;
            break;
        case out_event::TOOL_BEGIN: {
            char undeclared[512];
            if (!tool_call_declared(&j->req, e.name.c_str(), undeclared, sizeof(undeclared))) {
                /* a broken call (L272): never announced; TOOL_END or ERROR counts it */
                q->last_error = undeclared;
                break;
            }
            tool_call tc = {0};
            tc.name = server_xstrdup(e.name.c_str());
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
        case out_event::TOOL_ARGS:
            if (!q->open) break;
            if (live && !k->tool_ops->args(k, q->wire_open, e.text.data(), e.text.size()))
                return q->stream_ok = false;
            q->open_args = true;
            break;
        case out_event::TOOL_END:
            if (!q->open) {
                server_log(PULSAR_LOG_WARNING, "pulsar-server: chat ctx=%s%s%s %s output: %s",
                           g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags, family, q->last_error.c_str());
                s->trace_event(g->trace_id, "%s output: %s", family, q->last_error.c_str());
                q->undeclared++;
                break;
            }
            q->calls.v[q->calls.len - 1].arguments = server_xstrdup(e.arguments.c_str());
            if (live && !kit_close_open_call(q, g)) return q->stream_ok = false;
            q->open = false;
            break;
        case out_event::ERROR:
            server_log(PULSAR_LOG_WARNING, "pulsar-server: chat ctx=%s%s%s %s output: %s",
                       g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags, family, e.text.c_str());
            s->trace_event(g->trace_id, "%s output: %s", family, e.text.c_str());
            q->last_error = e.text;
            if (q->open) {
                /* announced, then broken: closed on the stream as sent, and not a call */
                if (live && !kit_close_open_call(q, g)) return q->stream_ok = false;
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
 * calls in the family's normal form, and the calls' sampled bytes become their tool memory; then the
 * finish every family shares (parser_finish_turn), the broken calls already logged as they streamed. */
/// server_output_parser_ops::finish.
/// @param st  the kit state (a kit_gen, from parser_kit_start)
/// @param s    the server
/// @param sl   the slot (the retry appends to its live session)
/// @param g    the attempt
/// @param out  receives the turn's reading (content, reasoning, calls owned by the caller)
/// @return false when a client write failed
bool parser_kit_finish(void *st, server *s, session_slot *sl, gen_state *g, server_turn *out) {
    kit_gen *q = (kit_gen *)st;
    job *j = g->j;
    out->finish = g->finish;
    if (j->req.kind != REQ_CHAT) return true;
    bool ok = true;
    if (strcmp(g->finish, "error") != 0) ok = parser_kit_feed(q, s, g, g->text.len, true);
    const out_parser &p = *q->parser;
    out->content = server_xstrdup(p.content().c_str());
    out->reasoning = p.reasoning().empty() ? NULL : server_xstrdup(p.reasoning().c_str());
    out->calls = q->calls;
    memset(&q->calls, 0, sizeof(q->calls));
    size_t raw_lo = 0, raw_hi = 0;
    /* the sampled bytes are the tool memory's key only when they are exactly the kept calls: a dropped
     * undeclared call is in them, so that turn re-renders canonically (a prefix miss, never a wrong prompt) */
    if (out->calls.len && q->undeclared == 0 && p.raw_span(&raw_lo, &raw_hi)) {
        if (raw_hi > g->text.len || raw_lo >= raw_hi) pulsar_die("output parser: raw span outside the turn's text");
        out->calls.raw_calls = server_xstrndup(g->text.ptr + raw_lo, raw_hi - raw_lo);
    }
    /* as text, the stream stopped at the call: the rest goes out at the finish */
    const char *t = g->text.ptr ? g->text.ptr : "";
    const char *call = strstr(t, p.grammar().call_open);
    return parser_finish_turn(s, sl, g, out, p.errors() > 0 || q->undeclared > 0, q->last_error.c_str(),
                              /*logged=*/true, call ? (size_t)(call - t) : SIZE_MAX, ok);
}

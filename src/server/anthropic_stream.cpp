/** @file
 *  Anthropic Messages responses: the final JSON body, the SSE event stream and its protocol sink (chat_sink).
 */
#include "pulsar_server_internal.h"



/* A matched client stop sequence is reported as such, with the sequence; tool
 * calls and the length cap take precedence, as in the Messages API. */
/// The Messages API stop_reason for a finish.
/// @param finish         the finish label ("tool_calls", "length", "stop", ...; may be NULL)
/// @param stop_sequence  the client stop sequence that matched, or NULL
/// @return "tool_use", "max_tokens", "stop_sequence" or "end_turn" (static strings)
static const char *anthropic_stop_reason(const char *finish, const char *stop_sequence) {
    if (finish && !strcmp(finish, "tool_calls")) return "tool_use";
    if (finish && !strcmp(finish, "length")) return "max_tokens";
    if (stop_sequence) return "stop_sequence";
    return "end_turn";
}

/// Append the "stop_reason" and "stop_sequence" members (no braces) for a finish.
/// @param b              the JSON being built
/// @param finish         the finish label (may be NULL)
/// @param stop_sequence  the matched stop sequence, or NULL (sent only when it is the reason)
static void append_anthropic_stop(buf *b, const char *finish, const char *stop_sequence) {
    const char *reason = anthropic_stop_reason(finish, stop_sequence);
    buf_puts(b, "\"stop_reason\":");
    json_escape(b, reason);
    buf_puts(b, ",\"stop_sequence\":");
    if (!strcmp(reason, "stop_sequence")) json_escape(b, stop_sequence);
    else buf_puts(b, "null");
}



/// Append one tool_use content block for a final (non-streamed) message.
/// @param b          the JSON being built
/// @param tc         the call; its id is used when set, else "toolu_<id_prefix>_<i>"
/// @param id_prefix  the message id, for the fallback tool id
/// @param i          the call's index, for the fallback tool id
static void append_anthropic_tool_use(buf *b, const tool_call *tc, const char *id_prefix, int i) {
    char idbuf[128];
    snprintf(idbuf, sizeof(idbuf), "toolu_%s_%d", id_prefix, i);
    buf_puts(b, "{\"type\":\"tool_use\",\"id\":");
    json_escape(b, tc->id && tc->id[0] ? tc->id : idbuf);
    buf_puts(b, ",\"name\":");
    json_escape(b, tc->name ? tc->name : "");
    buf_puts(b, ",\"input\":");
    append_json_object_or_empty(b, tc->arguments);
    buf_putc(b, '}');
}



/// Append one thinking content block.
/// @param b          the JSON being built
/// @param reasoning  the reasoning text (NULL: empty)
/// @param signature  the block's signature (the message id; NULL: empty)
static void append_anthropic_thinking(buf *b, const char *reasoning, const char *signature) {
    buf_puts(b, "{\"type\":\"thinking\",\"thinking\":");
    json_escape(b, reasoning ? reasoning : "");
    buf_puts(b, ",\"signature\":");
    json_escape(b, signature ? signature : "");
    buf_putc(b, '}');
}



/// Append a final message's "content" array: thinking, text, then the tool_use blocks; an empty text block when
/// nothing else was written or only thinking was (a message must end with content).
/// @param b          the JSON being built
/// @param text       the answer text (NULL or empty: none)
/// @param reasoning  the reasoning (NULL or empty: no thinking block)
/// @param calls      the turn's calls (may be NULL)
/// @param id_prefix  the message id: the thinking signature and the fallback tool ids
void append_anthropic_content(buf *b, const char *text, const char *reasoning,
                                     const tool_calls *calls, const char *id_prefix) {
    buf_putc(b, '[');
    bool wrote = false;
    bool wrote_after_thinking = false;
    if (reasoning && reasoning[0]) {
        append_anthropic_thinking(b, reasoning, id_prefix);
        wrote = true;
    }
    if (text && text[0]) {
        if (wrote) buf_putc(b, ',');
        buf_puts(b, "{\"type\":\"text\",\"text\":");
        json_escape(b, text);
        buf_putc(b, '}');
        wrote = true;
        wrote_after_thinking = true;
    }
    if (calls) {
        for (int i = 0; i < calls->len; i++) {
            if (wrote) buf_putc(b, ',');
            append_anthropic_tool_use(b, &calls->v[i], id_prefix, i);
            wrote = true;
            wrote_after_thinking = true;
        }
    }
    if (!wrote || ((reasoning && reasoning[0]) && !wrote_after_thinking)) {
        if (wrote) buf_putc(b, ',');
        buf_puts(b, "{\"type\":\"text\",\"text\":\"\"}");
    }
    buf_putc(b, ']');
}



/// Append the "usage" object: input tokens net of the cache split, output tokens, cache read / creation.
/// @param b                  the JSON being built
/// @param r                  the request (its cache_read_tokens / cache_write_tokens; NULL: none)
/// @param prompt_tokens      the prompt's tokens
/// @param completion_tokens  the tokens generated
static void append_anthropic_usage_json(buf *b, const request *r,
                                        int prompt_tokens, int completion_tokens) {
    int cache_read_tokens = r ? r->cache_read_tokens : 0;
    int cache_write_tokens = r ? r->cache_write_tokens : 0;
    resolve_cache_split(&cache_read_tokens, &cache_write_tokens, prompt_tokens);
    int input_tokens = prompt_tokens - cache_read_tokens - cache_write_tokens;
    if (input_tokens < 0) input_tokens = 0;
    buf_printf(b,
               "{\"input_tokens\":%d,\"output_tokens\":%d,"
               "\"cache_read_input_tokens\":%d,\"cache_creation_input_tokens\":%d}",
               input_tokens, completion_tokens, cache_read_tokens, cache_write_tokens);
}



/// Send a non-streamed Messages response (one 200 application/json body).
/// @param fd                 the client socket
/// @param r                  the request (its model)
/// @param id                 the message id
/// @param text               the answer text (may be NULL)
/// @param reasoning          the reasoning (may be NULL)
/// @param calls              the turn's calls (may be NULL)
/// @param finish             the finish label
/// @param stop_sequence      the matched stop sequence, or NULL
/// @param prompt_tokens      the prompt's tokens
/// @param completion_tokens  the tokens generated
/// @return false when the write failed
bool anthropic_final_response(int fd,
                                     const request *r, const char *id, const char *text,
                                     const char *reasoning, const tool_calls *calls, const char *finish,
                                     const char *stop_sequence,
                                     int prompt_tokens, int completion_tokens) {
    buf b = {0};
    buf_printf(&b, "{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":", id);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"content\":");
    append_anthropic_content(&b, text, reasoning, calls, id);
    buf_putc(&b, ',');
    append_anthropic_stop(&b, finish, stop_sequence);
    buf_puts(&b, ",\"usage\":");
    append_anthropic_usage_json(&b, r, prompt_tokens, completion_tokens);
    buf_puts(&b, "}\n");
    bool ok = http_response(fd, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}



/// Write one named SSE event ("event: ...\ndata: ...\n\n").
/// @param fd     the client socket
/// @param event  the event name
/// @param data   the event's JSON data
/// @return false when the write failed
static bool sse_event(int fd, const char *event, const char *data) {
    buf b = {0};
    buf_puts(&b, "event: ");
    buf_puts(&b, event);
    buf_puts(&b, "\ndata: ");
    buf_puts(&b, data);
    buf_puts(&b, "\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}



/// Send message_start and reset the stream state (st->active = whether it went out).
/// @param fd             the client socket
/// @param r              the request (its model, cache split)
/// @param id             the message id
/// @param prompt_tokens  the prompt's tokens, for the opening usage
/// @param st             the stream state, zeroed here
/// @return false when the write failed
bool anthropic_sse_start_live(int fd, const request *r, const char *id,
                                     int prompt_tokens, anthropic_stream *st) {
    buf b = {0};
    json_escape(&b, r->model);
    char *model_json = buf_take(&b);

    buf_printf(&b,
        "{\"type\":\"message_start\",\"message\":{\"id\":\"%s\",\"type\":\"message\","
        "\"role\":\"assistant\",\"model\":%s,\"content\":[],\"stop_reason\":null,"
        "\"stop_sequence\":null,\"usage\":",
        id, model_json);
    append_anthropic_usage_json(&b, r, prompt_tokens, 0);
    buf_puts(&b, "}}");
    bool ok = sse_event(fd, "message_start", b.ptr);
    buf_free(&b);
    free(model_json);

    memset(st, 0, sizeof(*st));
    st->active = ok;
    return ok;
}



/* Text and thinking blocks have fixed JSON shapes.  Tool blocks are opened by
 * name later, after the DSML invoke tag is complete, so they use a dedicated
 * opener instead of this helper. */
/// Open a text or thinking content block at st->next_index (a no-op when that kind is already open).
/// @param fd    the client socket
/// @param st    the stream state
/// @param type  ANTH_BLOCK_TEXT or ANTH_BLOCK_THINKING
/// @return false when another block is open or the write failed
static bool anthropic_sse_open_block(int fd, anthropic_stream *st,
                                     anthropic_block_type type) {
    if (st->open_block == type) return true;
    if (st->open_block != ANTH_BLOCK_NONE) return false;

    buf b = {0};
    if (type == ANTH_BLOCK_THINKING) {
        buf_printf(&b,
                   "{\"type\":\"content_block_start\",\"index\":%d,"
                   "\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\","
                   "\"signature\":\"\"}}",
                   st->next_index);
    } else {
        buf_printf(&b,
                   "{\"type\":\"content_block_start\",\"index\":%d,"
                   "\"content_block\":{\"type\":\"text\",\"text\":\"\"}}",
                   st->next_index);
    }
    bool ok = sse_event(fd, "content_block_start", b.ptr);
    buf_free(&b);
    if (ok) st->open_block = type;
    return ok;
}



/// Open a tool_use content block at st->next_index (a no-op when one is already open).
/// @param fd       the client socket
/// @param st       the stream state
/// @param tool_id  the call id the client sees (NULL: empty)
/// @param name     the tool's name (NULL: empty)
/// @return false when another kind of block is open or the write failed
static bool anthropic_sse_open_tool_block(int fd, anthropic_stream *st,
                                          const char *tool_id,
                                          const char *name) {
    if (st->open_block == ANTH_BLOCK_TOOL) return true;
    if (st->open_block != ANTH_BLOCK_NONE) return false;

    buf b = {0};
    buf_printf(&b,
               "{\"type\":\"content_block_start\",\"index\":%d,"
               "\"content_block\":{\"type\":\"tool_use\",\"id\":",
               st->next_index);
    json_escape(&b, tool_id ? tool_id : "");
    buf_puts(&b, ",\"name\":");
    json_escape(&b, name ? name : "");
    buf_puts(&b, ",\"input\":{}}}");
    bool ok = sse_event(fd, "content_block_start", b.ptr);
    buf_free(&b);
    if (ok) st->open_block = ANTH_BLOCK_TOOL;
    return ok;
}



/// Send a text_delta or thinking_delta for the open block.
/// @param fd    the client socket
/// @param st    the stream state (its next_index is the open block's)
/// @param type  which delta (ANTH_BLOCK_THINKING: thinking_delta, else text_delta)
/// @param text  the bytes
/// @param len   bytes at `text` (0: nothing sent)
/// @return false when the write failed
static bool anthropic_sse_delta_live(int fd, const anthropic_stream *st,
                                     anthropic_block_type type,
                                     const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    if (type == ANTH_BLOCK_THINKING) {
        buf_printf(&b,
                   "{\"type\":\"content_block_delta\",\"index\":%d,"
                   "\"delta\":{\"type\":\"thinking_delta\",\"thinking\":",
                   st->next_index);
        json_escape_n(&b, text, len);
        buf_puts(&b, "}}");
    } else {
        buf_printf(&b,
                   "{\"type\":\"content_block_delta\",\"index\":%d,"
                   "\"delta\":{\"type\":\"text_delta\",\"text\":",
                   st->next_index);
        json_escape_n(&b, text, len);
        buf_puts(&b, "}}");
    }
    bool ok = sse_event(fd, "content_block_delta", b.ptr);
    buf_free(&b);
    return ok;
}



/* Anthropic's input_json_delta carries a fragment of a JSON object, encoded as
 * a JSON string.  We stream exactly the same object that the final DSML parser
 * will build: an opening "{", quoted keys, raw JSON values or escaped string
 * contents, and the closing "}". */
/// Send an input_json_delta for the open tool_use block.
/// @param fd    the client socket
/// @param st    the stream state
/// @param text  a fragment of the arguments object's JSON text
/// @param len   bytes at `text` (0: nothing sent)
/// @return false when the write failed
static bool anthropic_sse_tool_delta_live(int fd, const anthropic_stream *st,
                                          const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    buf_printf(&b,
               "{\"type\":\"content_block_delta\",\"index\":%d,"
               "\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":",
               st->next_index);
    json_escape_n(&b, text, len);
    buf_puts(&b, "}}");
    bool ok = sse_event(fd, "content_block_delta", b.ptr);
    buf_free(&b);
    return ok;
}



/// Close the open block (a thinking block first gets its signature_delta) and advance next_index.
/// @param fd  the client socket
/// @param id  the message id, the thinking block's signature
/// @param st  the stream state (no open block: a no-op)
/// @return false when a write failed
static bool anthropic_sse_close_block_live(int fd, const char *id,
                                           anthropic_stream *st) {
    if (st->open_block == ANTH_BLOCK_NONE) return true;

    buf b = {0};
    bool ok = true;
    if (st->open_block == ANTH_BLOCK_THINKING) {
        buf_printf(&b,
                   "{\"type\":\"content_block_delta\",\"index\":%d,"
                   "\"delta\":{\"type\":\"signature_delta\",\"signature\":",
                   st->next_index);
        json_escape(&b, id);
        buf_puts(&b, "}}");
        ok = sse_event(fd, "content_block_delta", b.ptr);
        buf_free(&b);
    }
    if (ok) {
        buf_printf(&b, "{\"type\":\"content_block_stop\",\"index\":%d}",
                   st->next_index);
        ok = sse_event(fd, "content_block_stop", b.ptr);
        buf_free(&b);
    }
    if (ok) {
        st->open_block = ANTH_BLOCK_NONE;
        st->next_index++;
    }
    return ok;
}



/* The Anthropic side of a family's live tool-call events: a tool_use content
 * block per call, its arguments as input_json_delta, stopped when the call
 * closes.  The id is the one the client sees from here on: a DeepSeek parse
 * copies it into the parsed tool_call (apply_stream_tool_ids) before
 * tool_memory_remember(), so the next tool_result can continue from the live
 * KV state instead of re-rendering canonical JSON. */
/// sink_tool_ops::begin: open the call's tool_use block.
/// @param k     the Anthropic sink
/// @param id    the call id the client sees
/// @param name  the tool's name
/// @return false when the write failed
static bool anthropic_tool_begin(chat_sink *k, int, const char *id, const char *name) {
    return anthropic_sse_open_tool_block(k->fd, (anthropic_stream *)k->st, id, name);
}

/// sink_tool_ops::args: stream an arguments fragment as input_json_delta.
/// @param k     the Anthropic sink
/// @param text  the fragment
/// @param len   bytes at `text`
/// @return false when the write failed
static bool anthropic_tool_args(chat_sink *k, int, const char *text, size_t len) {
    return anthropic_sse_tool_delta_live(k->fd, (anthropic_stream *)k->st, text, len);
}

/// sink_tool_ops::end: close the tool_use block and count it as streamed (anthropic_stream::tools_streamed).
/// @param k  the Anthropic sink
/// @return false when the write failed
static bool anthropic_tool_end(chat_sink *k, int) {
    anthropic_stream *st = (anthropic_stream *)k->st;
    if (!anthropic_sse_close_block_live(k->fd, k->id, st)) return false;
    st->tools_streamed++;
    return true;
}

/// The Anthropic sink's live tool-call events (tool_use blocks).
static const sink_tool_ops anthropic_tool_ops = {anthropic_tool_begin, anthropic_tool_args, anthropic_tool_end};






/// Send, whole, every call the live events did not stream (calls past st->tools_streamed): start, one
/// input_json_delta with the arguments, stop.
/// @param fd     the client socket
/// @param r      the request (unused)
/// @param id     the message id, for fallback tool ids
/// @param st     the stream state
/// @param calls  the turn's calls (NULL: none)
/// @return false when a write failed
static bool anthropic_sse_tool_blocks_live(int fd, const request *r, const char *id,
                                           anthropic_stream *st,
                                           const tool_calls *calls) {
    (void)r;
    if (!calls) return true;

    buf b = {0};
    /* Tool calls the live DSML projection completed have already produced
     * start/delta/stop events.  Only emit the tail calls that were not seen by
     * it, for example if the first DSML bytes only become available during the
     * final flush (or the family hands calls over whole). */
    int already_streamed = st->tools_streamed;
    if (already_streamed > calls->len) already_streamed = calls->len;
    for (int i = already_streamed; i < calls->len; i++, st->next_index++) {
        const tool_call *tc = &calls->v[i];
        char idbuf[128];
        snprintf(idbuf, sizeof(idbuf), "toolu_%s_%d", id, i);
        buf_printf(&b,
                   "{\"type\":\"content_block_start\",\"index\":%d,"
                   "\"content_block\":{\"type\":\"tool_use\",\"id\":",
                   st->next_index);
        json_escape(&b, tc->id && tc->id[0] ? tc->id : idbuf);
        buf_puts(&b, ",\"name\":");
        json_escape(&b, tc->name ? tc->name : "");
        buf_puts(&b, ",\"input\":{}}}");
        bool ok = sse_event(fd, "content_block_start", b.ptr);
        buf_free(&b);
        if (!ok) return false;

        buf_printf(&b,
                   "{\"type\":\"content_block_delta\",\"index\":%d,"
                   "\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":",
                   st->next_index);
        append_json_object_string(&b, tc->arguments);
        buf_puts(&b, "}}");
        ok = sse_event(fd, "content_block_delta", b.ptr);
        buf_free(&b);
        if (!ok) return false;

        buf_printf(&b, "{\"type\":\"content_block_stop\",\"index\":%d}",
                   st->next_index);
        ok = sse_event(fd, "content_block_stop", b.ptr);
        buf_free(&b);
        if (!ok) return false;
    }
    return true;
}



/// Send message_delta (stop reason, output tokens) and message_stop.
/// @param fd                 the client socket
/// @param finish             the finish label
/// @param stop_sequence      the matched stop sequence, or NULL
/// @param completion_tokens  the tokens generated
/// @return false when a write failed
static bool anthropic_sse_stop_live(int fd, const char *finish, const char *stop_sequence,
                                    int completion_tokens) {
    buf b = {0};
    buf_puts(&b, "{\"type\":\"message_delta\",\"delta\":{");
    append_anthropic_stop(&b, finish, stop_sequence);
    buf_puts(&b, "},\"usage\":{\"output_tokens\":");
    buf_printf(&b, "%d}}", completion_tokens);
    bool ok = sse_event(fd, "message_delta", b.ptr);
    buf_free(&b);
    if (ok) ok = sse_event(fd, "message_stop", "{\"type\":\"message_stop\"}");
    return ok;
}



/* The end of a streamed message, every block closed: an empty text block when
 * only thinking was sent (a message must end with content), the tool_use blocks
 * not yet streamed, then message_delta + message_stop. */
/// End a streamed message with every block closed: an empty text block when only thinking went out, unstreamed tool_use blocks, message_delta and message_stop.
/// @param fd                 the client socket
/// @param r                  the request
/// @param id                 the message id
/// @param st                 the stream state (no block open)
/// @param calls              the turn's calls (may be NULL)
/// @param finish             the finish label
/// @param stop_sequence      the matched stop sequence, or NULL
/// @param completion_tokens  the tokens generated
/// @return false when a write failed
static bool anthropic_sse_finish_tail(int fd, const request *r, const char *id, anthropic_stream *st,
                                      const tool_calls *calls, const char *finish,
                                      const char *stop_sequence, int completion_tokens) {
    if (st->sent_thinking && !st->sent_text && (!calls || calls->len == 0)) {
        if (!anthropic_sse_open_block(fd, st, ANTH_BLOCK_TEXT)) return false;
        if (!anthropic_sse_close_block_live(fd, id, st)) return false;
    }

    if (!anthropic_sse_tool_blocks_live(fd, r, id, st, calls)) return false;
    return anthropic_sse_stop_live(fd, finish, stop_sequence, completion_tokens);
}

/* ---- L267: the Anthropic sink (chat_sink) ------------------------------------ */

/* Reasoning into a thinking block, content into a text block: the open block
 * closes when the other kind arrives (DeepSeek's walk closes it at the section
 * end first). */
/// chat_sink::text for Anthropic.
/// @param k          the sink
/// @param reasoning  the text is reasoning (a thinking block), else answer (a text block)
/// @param text       the bytes
/// @param len        bytes at `text` (0, or an inactive stream: nothing sent)
/// @return false when a write failed
static bool anthropic_sink_text_cb(chat_sink *k, bool reasoning, const char *text, size_t len, size_t) {
    anthropic_stream *st = (anthropic_stream *)k->st;
    if (!st->active || len == 0) return true;
    const anthropic_block_type type = reasoning ? ANTH_BLOCK_THINKING : ANTH_BLOCK_TEXT;
    if (st->open_block != type && !anthropic_sse_close_block_live(k->fd, k->id, st)) return false;
    if (!anthropic_sse_open_block(k->fd, st, type)) return false;
    if (!anthropic_sse_delta_live(k->fd, st, type, text, len)) return false;
    if (reasoning) st->sent_thinking = true;
    else st->sent_text = true;
    return true;
}

/// chat_sink::end for Anthropic: close the open block.
/// @param k  the sink
/// @return false when a write failed
static bool anthropic_sink_end_cb(chat_sink *k, bool) {
    return anthropic_sse_close_block_live(k->fd, k->id, (anthropic_stream *)k->st);
}

/// Build the Anthropic protocol sink over `st` (live tool events on; worker thread).
/// @param k   the sink to fill
/// @param fd  the client socket
/// @param s   the server (NULL in unit tests)
/// @param r   the request
/// @param id  the message id (borrowed for the sink's life)
/// @param st  the stream state (started by anthropic_sse_start_live)
void anthropic_sink_init(chat_sink *k, int fd, server *s, const request *r, const char *id,
                         anthropic_stream *st) {
    /* a block first seen in the final flush goes out with the finish */
    *k = {fd, s, r, id, st, anthropic_sink_text_cb, anthropic_sink_end_cb, &anthropic_tool_ops, false};
}

/* The end of the message: the open block closed, then the tail. */
/// End an Anthropic stream: close the open block, then anthropic_sse_finish_tail.
/// @param k                  the Anthropic sink
/// @param calls              the turn's calls (may be NULL)
/// @param finish             the finish label
/// @param stop_sequence      the matched stop sequence, or NULL
/// @param completion_tokens  the tokens generated
/// @return false when a write failed
bool anthropic_sse_finish(chat_sink *k, const tool_calls *calls, const char *finish,
                          const char *stop_sequence, int completion_tokens) {
    anthropic_stream *st = (anthropic_stream *)k->st;
    if (!anthropic_sse_close_block_live(k->fd, k->id, st)) return false;
    return anthropic_sse_finish_tail(k->fd, k->r, k->id, st, calls, finish, stop_sequence, completion_tokens);
}

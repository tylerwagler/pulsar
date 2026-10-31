/** @file
 *  OpenAI Responses API responses: the final JSON body, the SSE event stream and its protocol sink.
 */
#include "pulsar_server_internal.h"



/* Codex' Responses API uses 24-hex suffixes for response/item ids
 * (random_prefixed_id with 12 bytes).  The prefix controls the variant
 * (resp_, rs_, msg_, fc_) so each event references a stable identifier across
 * output_item.added / .done. */


/// Reset a Responses stream state: fresh resp_ / rs_ / msg_ ids (24 hex), no items open (indexes -1).
/// @param r   the request (unused)
/// @param st  the state, zeroed
void responses_stream_init(const request *r, responses_stream *st) {
    (void)r;
    memset(st, 0, sizeof(*st));
    random_prefixed_id(st->response_id, sizeof(st->response_id), "resp_", 12);
    random_prefixed_id(st->reasoning_id, sizeof(st->reasoning_id), "rs_", 12);
    random_prefixed_id(st->message_id, sizeof(st->message_id), "msg_", 12);
    st->reasoning_index = -1;
    st->message_index = -1;
}



/// Free the accumulated reasoning and message text (the state itself is the caller's).
/// @param st  the state (NULL: nothing)
void responses_stream_free(responses_stream *st) {
    if (!st) return;
    buf_free(&st->reasoning_text);
    buf_free(&st->message_text);
}



/* Codex parses an explicit sequence_number on every Responses event for
 * ordering and reconnect resilience. We inject it after the `{"type":...` head
 * so emitters can stay readable while still producing the wire shape Codex
 * expects. */
/// Send one Responses SSE event with sequence_number spliced in after its "type" (st->sequence advances).
/// @param fd    the client socket
/// @param st    the stream state
/// @param body  the event's JSON object, starting with {"type":"..."
/// @return false when the write failed
static bool responses_sse_emit_event(int fd, responses_stream *st, const char *body) {
    buf b = {0};
    buf_puts(&b, "data: ");
    /* body always starts with `{"type":"..."`. We splice in sequence_number
     * after the closing quote of that string so every event has it as the
     * second field. */
    const char *type_close = NULL;
    if (body[0] == '{') {
        const char *p = body + 1;
        /* Skip the literal `"type":` then the value string. */
        if (!strncmp(p, "\"type\":\"", 8)) {
            const char *q = p + 8;
            while (*q && *q != '"') {
                if (*q == '\\' && q[1]) q += 2;
                else q++;
            }
            if (*q == '"') type_close = q + 1;
        }
    }
    if (type_close) {
        size_t head_len = (size_t)(type_close - body);
        buf_append(&b, body, head_len);
        buf_printf(&b, ",\"sequence_number\":%d", st->sequence++);
        buf_puts(&b, type_close);
    } else {
        buf_puts(&b, body);
    }
    buf_puts(&b, "\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}



/// Send response.created (status in_progress, empty output).
/// @param fd          the client socket
/// @param r           the request (its model)
/// @param st          the stream state (its response_id)
/// @param created_at  the response's creation time, Unix seconds
/// @return false when the write failed
bool responses_sse_created(int fd, const request *r, responses_stream *st,
                                  long created_at) {
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.created\",\"response\":{\"id\":\"%s\","
        "\"object\":\"response\",\"created_at\":%ld,\"status\":\"in_progress\","
        "\"model\":", st->response_id, created_at);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"output\":[]}}");
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}



/// Send output_item.added for the reasoning item at st->reasoning_index.
/// @param fd  the client socket
/// @param st  the stream state
/// @return false when the write failed
static bool responses_sse_reasoning_added(int fd, responses_stream *st) {
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.output_item.added\",\"output_index\":%d,"
        "\"item\":{\"id\":\"%s\",\"type\":\"reasoning\",\"status\":\"in_progress\","
        "\"summary\":[]}}",
        st->reasoning_index, st->reasoning_id);
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}



/// Send reasoning_summary_part.added (summary index 0) for the reasoning item.
/// @param fd  the client socket
/// @param st  the stream state
/// @return false when the write failed
static bool responses_sse_reasoning_summary_part_added(int fd, responses_stream *st) {
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.reasoning_summary_part.added\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"summary_index\":0,"
        "\"part\":{\"type\":\"summary_text\",\"text\":\"\"}}",
        st->reasoning_id, st->reasoning_index);
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}



/// Send a reasoning_summary_text.delta.
/// @param fd    the client socket
/// @param st    the stream state
/// @param text  the bytes
/// @param len   bytes at `text` (0: nothing sent)
/// @return false when the write failed
static bool responses_sse_reasoning_delta(int fd, responses_stream *st,
                                          const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.reasoning_summary_text.delta\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"summary_index\":0,\"delta\":",
        st->reasoning_id, st->reasoning_index);
    json_escape_n(&b, text, len);
    buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}



/// An output item's status for a finish.
/// @param finish  the finish label (may be NULL)
/// @return "incomplete" for length / error, else "completed"
static const char *responses_item_status_for_finish(const char *finish) {
    if (finish && (!strcmp(finish, "length") || !strcmp(finish, "error"))) return "incomplete";
    return "completed";
}



/// Close the reasoning item: summary_text.done, summary_part.done (when one was opened), output_item.done; the
/// item is completed only when the model closed its reasoning itself.
/// @param fd      the client socket
/// @param st      the stream state
/// @param finish  the finish label (unused: the status follows reasoning_closed_naturally)
/// @return false when a write failed
static bool responses_sse_reasoning_done(int fd, responses_stream *st,
                                         const char *finish) {
    /* If the stream terminates before `</think>` was actually observed the
     * reasoning item is partial — regardless of why generation stopped (EOS,
     * stop sequence, tool_calls, length, error). Force the item to incomplete
     * so a client replay rejects it instead of feeding unfinished hidden state
     * back as completed history. */
    (void)finish;
    const char *item_status =
        st->reasoning_closed_naturally ? "completed" : "incomplete";
    /* Mirror the message-item close sequence: emit summary_text.done +
     * summary_part.done before the output_item.done so clients that key off
     * part lifecycle don't see a dangling open summary part. */
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.reasoning_summary_text.done\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"summary_index\":0,\"text\":",
        st->reasoning_id, st->reasoning_index);
    json_escape_n(&b, st->reasoning_text.ptr ? st->reasoning_text.ptr : "",
                  st->reasoning_text.len);
    buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    if (!ok) {
        buf_free(&b);
        return false;
    }

    if (st->reasoning_summary_started) {
        buf_free(&b);
        buf_printf(&b,
            "{\"type\":\"response.reasoning_summary_part.done\","
            "\"item_id\":\"%s\",\"output_index\":%d,\"summary_index\":0,"
            "\"part\":{\"type\":\"summary_text\",\"text\":",
            st->reasoning_id, st->reasoning_index);
        json_escape_n(&b, st->reasoning_text.ptr ? st->reasoning_text.ptr : "",
                      st->reasoning_text.len);
        buf_puts(&b, "}}");
        ok = responses_sse_emit_event(fd, st, b.ptr);
        if (!ok) {
            buf_free(&b);
            return false;
        }
    }

    buf_free(&b);
    buf_printf(&b,
        "{\"type\":\"response.output_item.done\",\"output_index\":%d,"
        "\"item\":{\"id\":\"%s\",\"type\":\"reasoning\",\"status\":\"%s\",\"summary\":[",
        st->reasoning_index, st->reasoning_id, item_status);
    if (st->reasoning_text.len) {
        buf_puts(&b, "{\"type\":\"summary_text\",\"text\":");
        json_escape_n(&b, st->reasoning_text.ptr, st->reasoning_text.len);
        buf_putc(&b, '}');
    }
    buf_puts(&b, "]}}");
    ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}



/// Send output_item.added for the assistant message item at st->message_index.
/// @param fd  the client socket
/// @param st  the stream state
/// @return false when the write failed
static bool responses_sse_message_added(int fd, responses_stream *st) {
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.output_item.added\",\"output_index\":%d,"
        "\"item\":{\"id\":\"%s\",\"type\":\"message\",\"status\":\"in_progress\","
        "\"role\":\"assistant\",\"content\":[]}}",
        st->message_index, st->message_id);
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}



/// Send content_part.added (an empty output_text part) for the message item.
/// @param fd  the client socket
/// @param st  the stream state
/// @return false when the write failed
static bool responses_sse_message_text_part_added(int fd, responses_stream *st) {
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.content_part.added\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,"
        "\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}",
        st->message_id, st->message_index);
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}



/// Send an output_text.delta for the message item.
/// @param fd    the client socket
/// @param st    the stream state
/// @param text  the bytes
/// @param len   bytes at `text` (0: nothing sent)
/// @return false when the write failed
static bool responses_sse_output_text_delta(int fd, responses_stream *st,
                                            const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.output_text.delta\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,\"delta\":",
        st->message_id, st->message_index);
    json_escape_n(&b, text, len);
    buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}



/// Close the message item: output_text.done, content_part.done, output_item.done with the accumulated text.
/// @param fd      the client socket
/// @param st      the stream state
/// @param finish  the finish label (the item's status)
/// @return false when a write failed
static bool responses_sse_message_done(int fd, responses_stream *st,
                                       const char *finish) {
    const char *item_status = responses_item_status_for_finish(finish);
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.output_text.done\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,\"text\":",
        st->message_id, st->message_index);
    json_escape_n(&b, st->message_text.ptr ? st->message_text.ptr : "",
                  st->message_text.len);
    buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    if (!ok) {
        buf_free(&b);
        return false;
    }

    buf_free(&b);
    buf_printf(&b,
        "{\"type\":\"response.content_part.done\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,"
        "\"part\":{\"type\":\"output_text\",\"text\":",
        st->message_id, st->message_index);
    json_escape_n(&b, st->message_text.ptr ? st->message_text.ptr : "",
                  st->message_text.len);
    buf_puts(&b, ",\"annotations\":[]}}");
    ok = responses_sse_emit_event(fd, st, b.ptr);
    if (!ok) {
        buf_free(&b);
        return false;
    }

    buf_free(&b);
    buf_printf(&b,
        "{\"type\":\"response.output_item.done\",\"output_index\":%d,"
        "\"item\":{\"id\":\"%s\",\"type\":\"message\",\"status\":\"%s\","
        "\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":",
        st->message_index, st->message_id, item_status);
    json_escape_n(&b, st->message_text.ptr ? st->message_text.ptr : "",
                  st->message_text.len);
    buf_puts(&b, ",\"annotations\":[]}]}}");
    ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}



/// Whether a call is the client-executed tool_search (sent as a tool_search_call item).
/// @param tc     the call (NULL: false)
/// @param order  the tool's schema order (NULL: the name alone decides)
/// @return true for a tool_search call the request declared as Responses tool search
static bool responses_tool_call_is_tool_search(const tool_call *tc,
                                               const tool_schema_order *order) {
    return tc && tc->name && !strcmp(tc->name, "tool_search") &&
           (!order || order->responses_tool_search);
}



/* The internal tool_call doesn't track whether it came from a function_call or
 * a custom_tool_call (or what tool kind is registered). For round-trip
 * correctness with the rare custom_tool_call clients, we preserve any provided
 * call_id verbatim and pre-assign a stable fc_id; the discriminator currently
 * defaults to function_call because Codex CLI registers all its tools as
 * function tools. */
/// Build the per-call Responses item identities: a fresh fc_ id, the call id (or a fresh call_ id), the output
/// index.
/// @param out                    receives the malloc'd array (calls->len entries; NULL for no calls); caller frees
/// @param calls                  the turn's calls (may be NULL)
/// @param starting_output_index  the output index of the first call
static void responses_tool_items_build(responses_tool_item **out,
                                       const tool_calls *calls,
                                       int starting_output_index) {
    *out = NULL;
    if (!calls || calls->len == 0) return;
    responses_tool_item *items = (responses_tool_item *)server_xmalloc((size_t)calls->len * sizeof(*items));
    for (int i = 0; i < calls->len; i++) {
        memset(&items[i], 0, sizeof(items[i]));
        random_prefixed_id(items[i].fc_id, sizeof(items[i].fc_id), "fc_", 12);
        if (calls->v[i].id && calls->v[i].id[0]) {
            snprintf(items[i].call_id, sizeof(items[i].call_id), "%s", calls->v[i].id);
        } else {
            random_prefixed_id(items[i].call_id, sizeof(items[i].call_id), "call_", 12);
        }
        items[i].is_custom = false;
        items[i].output_index = starting_output_index + i;
    }
    *out = items;
}



/// Append one function_call (or custom_tool_call / tool_search_call) output item.
/// @param b            the JSON being built
/// @param tc           the call
/// @param item         its Responses identity (responses_tool_items_build)
/// @param item_status  the item's status string
/// @param with_args    include the arguments (false: an empty arguments string, for output_item.added)
/// @param orders       the request's tool schema orders (wire names, namespaces)
void responses_append_function_call_item(buf *b, const tool_call *tc,
                                                const responses_tool_item *item,
                                                const char *item_status,
                                                bool with_args,
                                                const tool_schema_orders *orders) {
    const tool_schema_order *order = tool_schema_orders_find(orders, tc->name);
    if (responses_tool_call_is_tool_search(tc, order)) {
        buf_printf(b,
            "{\"id\":\"%s\",\"type\":\"tool_search_call\",\"status\":\"%s\","
            "\"call_id\":\"%s\",\"execution\":\"client\",\"arguments\":",
            item->fc_id, item_status, item->call_id);
        if (with_args) append_json_object_or_empty(b, tc->arguments);
        else buf_puts(b, "{}");
        buf_putc(b, '}');
        return;
    }

    const char *item_type = item->is_custom ? "custom_tool_call" : "function_call";
    const char *body_field = item->is_custom ? "input" : "arguments";
    buf_printf(b,
        "{\"id\":\"%s\",\"type\":\"%s\",\"status\":\"%s\",\"name\":",
        item->fc_id, item_type, item_status);
    json_escape(b, order && order->wire_name ? order->wire_name :
                   (tc->name ? tc->name : ""));
    if (order && order->tool_namespace) {
        buf_puts(b, ",\"namespace\":");
        json_escape(b, order->tool_namespace);
    }
    buf_puts(b, ",\"call_id\":");
    json_escape(b, item->call_id);
    buf_printf(b, ",\"%s\":", body_field);
    if (!with_args) {
        buf_puts(b, "\"\"");
    } else if (item->is_custom) {
        json_escape(b, tc->arguments ? tc->arguments : "");
    } else {
        append_json_object_string(b, tc->arguments);
    }
    buf_putc(b, '}');
}



/// Send output_item.added (in_progress, no arguments) or output_item.done for one call.
/// @param fd      the client socket
/// @param st      the stream state
/// @param tc      the call
/// @param item    its Responses identity
/// @param orders  the request's tool schema orders
/// @param finish  the finish label (the done item's status)
/// @param done    send .done, else .added
/// @return false when the write failed
static bool responses_sse_function_call_event(int fd, responses_stream *st,
                                              const tool_call *tc,
                                              const responses_tool_item *item,
                                              const tool_schema_orders *orders,
                                              const char *finish,
                                              bool done) {
    /* The added event marks a tool call as in_progress per the Responses
     * lifecycle; only output_item.done (and the terminal response output)
     * carry the final completed / incomplete status. The added item ships with
     * an empty arguments string so clients that accumulate via
     * function_call_arguments.delta + .done don't end up with doubled JSON. */
    const char *item_status = done ? responses_item_status_for_finish(finish) : "in_progress";
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.output_item.%s\",\"output_index\":%d,\"item\":",
        done ? "done" : "added", item->output_index);
    responses_append_function_call_item(&b, tc, item, item_status, done, orders);
    buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}



/* Stream function-call arguments as a single delta + done, since DS4 generates
 * the whole DSML invoke as one unit before the worker decides which tool was
 * called. Clients that follow the OpenAI Responses lifecycle expect both
 * events between output_item.added (in_progress) and output_item.done. */
/// Send a function call's arguments as one function_call_arguments.delta and .done.
/// @param fd      the client socket
/// @param st      the stream state
/// @param tc      the call
/// @param item    its Responses identity (custom calls and tool_search send nothing)
/// @param orders  the request's tool schema orders
/// @return false when a write failed
static bool responses_sse_function_call_arguments_done(int fd, responses_stream *st,
                                                       const tool_call *tc,
                                                       const responses_tool_item *item,
                                                       const tool_schema_orders *orders) {
    const tool_schema_order *order = tool_schema_orders_find(orders, tc->name);
    if (item->is_custom || responses_tool_call_is_tool_search(tc, order)) return true;
    buf args = {0};
    append_json_object_string(&args, tc->arguments);
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.function_call_arguments.delta\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"delta\":",
        item->fc_id, item->output_index);
    buf_append(&b, args.ptr ? args.ptr : "\"\"", args.ptr ? args.len : 2);
    buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    if (!ok) {
        buf_free(&b);
        buf_free(&args);
        return false;
    }

    buf_free(&b);
    buf_printf(&b,
        "{\"type\":\"response.function_call_arguments.done\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"name\":",
        item->fc_id, item->output_index);
    json_escape(&b, order && order->wire_name ? order->wire_name :
                    (tc->name ? tc->name : ""));
    if (order && order->tool_namespace) {
        buf_puts(&b, ",\"namespace\":");
        json_escape(&b, order->tool_namespace);
    }
    buf_puts(&b, ",\"arguments\":");
    buf_append(&b, args.ptr ? args.ptr : "\"\"", args.ptr ? args.len : 2);
    buf_putc(&b, '}');
    ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    buf_free(&args);
    return ok;
}



/// The response's status for a finish.
/// @param finish  the finish label (may be NULL)
/// @return "incomplete" (length), "failed" (error) or "completed"
static const char *responses_status_for_finish(const char *finish) {
    if (finish && !strcmp(finish, "length")) return "incomplete";
    if (finish && !strcmp(finish, "error")) return "failed";
    return "completed";
}



/// Append the Responses "usage" object (input with the cache split, output, total).
/// @param b              the JSON being built
/// @param r              the request (its cache split; NULL: none)
/// @param input_tokens   the prompt's tokens
/// @param output_tokens  the tokens generated
static void append_responses_usage_json(buf *b, const request *r,
                                        int input_tokens, int output_tokens) {
    int cached_tokens = r ? r->cache_read_tokens : 0;
    int cache_write_tokens = r ? r->cache_write_tokens : 0;
    resolve_cache_split(&cached_tokens, &cache_write_tokens, input_tokens);
    buf_printf(b,
        "{\"input_tokens\":%d,\"input_tokens_details\":{\"cached_tokens\":%d,\"cache_write_tokens\":%d},"
        "\"output_tokens\":%d,\"output_tokens_details\":{\"reasoning_tokens\":0},"
        "\"total_tokens\":%d}",
        input_tokens, cached_tokens, cache_write_tokens,
        output_tokens, input_tokens + output_tokens);
}



/// Send the terminal event (response.completed, .incomplete or .failed) with the whole output: the reasoning item,
/// the message item and the calls.
/// @param fd                 the client socket
/// @param r                  the request
/// @param st                 the stream state
/// @param calls              the turn's calls (may be NULL)
/// @param tool_items         their Responses identities (NULL: calls not listed)
/// @param finish             the finish label
/// @param prompt_tokens      the prompt's tokens
/// @param completion_tokens  the tokens generated
/// @param created_at         the response's creation time, Unix seconds
/// @return false when the write failed
bool responses_sse_completed(int fd, const request *r,
                                    responses_stream *st,
                                    const tool_calls *calls,
                                    const responses_tool_item *tool_items,
                                    const char *finish,
                                    int prompt_tokens, int completion_tokens,
                                    long created_at) {
    /* Codex routes terminal behaviour off the event type, not response.status.
     * Decide here so clients see response.failed / response.incomplete instead
     * of a "completed" wrapper marked failed in a sub-field. */
    const char *event_type = "response.completed";
    if (finish && !strcmp(finish, "error")) event_type = "response.failed";
    else if (finish && !strcmp(finish, "length")) event_type = "response.incomplete";
    const char *status = responses_status_for_finish(finish);

    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"%s\",\"response\":{\"id\":\"%s\","
        "\"object\":\"response\",\"created_at\":%ld,\"status\":\"%s\",\"model\":",
        event_type, st->response_id, created_at, status);
    json_escape(&b, r->model);
    if (!strcmp(event_type, "response.failed")) {
        buf_puts(&b, ",\"error\":{\"code\":\"server_error\","
                     "\"message\":\"generation failed\"}");
    } else if (!strcmp(event_type, "response.incomplete")) {
        buf_puts(&b, ",\"incomplete_details\":{\"reason\":\"max_tokens\"}");
    }
    const char *item_status = responses_item_status_for_finish(finish);
    buf_puts(&b, ",\"output\":[");
    bool wrote = false;
    if (st->reasoning_emitted_any) {
        /* Match responses_sse_reasoning_done: if the stream stopped before
         * </think>, the reasoning item is partial regardless of the
         * response-level finish status, so replay must reject it. */
        const char *reasoning_status =
            st->reasoning_closed_naturally ? "completed" : "incomplete";
        buf_printf(&b,
            "{\"id\":\"%s\",\"type\":\"reasoning\",\"status\":\"%s\",\"summary\":[",
            st->reasoning_id, reasoning_status);
        if (st->reasoning_text.len) {
            buf_puts(&b, "{\"type\":\"summary_text\",\"text\":");
            json_escape_n(&b, st->reasoning_text.ptr, st->reasoning_text.len);
            buf_putc(&b, '}');
        }
        buf_puts(&b, "]}");
        wrote = true;
    }
    if (st->message_emitted_any) {
        if (wrote) buf_putc(&b, ',');
        buf_printf(&b,
            "{\"id\":\"%s\",\"type\":\"message\",\"status\":\"%s\","
            "\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":",
            st->message_id, item_status);
        json_escape_n(&b, st->message_text.ptr ? st->message_text.ptr : "",
                      st->message_text.len);
        buf_puts(&b, ",\"annotations\":[]}]}");
        wrote = true;
    }
    if (calls && tool_items) {
        for (int i = 0; i < calls->len; i++) {
            if (wrote) buf_putc(&b, ',');
            responses_append_function_call_item(&b, &calls->v[i], &tool_items[i],
                                                item_status, true,
                                                &r->tool_orders);
            wrote = true;
        }
    }
    buf_putc(&b, ']');
    buf_puts(&b, ",\"usage\":");
    append_responses_usage_json(&b, r, prompt_tokens, completion_tokens);
    buf_puts(&b, "}}");
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}



/* Responses streaming consumes the same raw token text the OpenAI live stream
 * consumes: <think>...</think> is reasoning, anything before the tool-call
 * marker is output text. Tool-call argument deltas are not surfaced because
 * Codex' SSE parser only ingests function_call items via output_item.done. */
/* One reasoning delta into the reasoning item (opened, with its summary part, on
 * the first) and one output_text delta into the message item: the two shapes
 * every producer streams -- DeepSeek's text projection below and a family's
 * output parser (responses_sink_text). */
/// Stream reasoning into the reasoning item, opening it and its summary part on the first delta.
/// @param fd    the client socket
/// @param st    the stream state (the text is also accumulated in st->reasoning_text)
/// @param text  the bytes
/// @param len   bytes at `text`
/// @return false when a write failed
static bool responses_emit_reasoning(int fd, responses_stream *st, const char *text, size_t len) {
    if (!st->reasoning_item_opened) {
        st->reasoning_index = st->next_output_index++;
        if (!responses_sse_reasoning_added(fd, st)) return false;
        st->reasoning_item_opened = true;
    }
    if (!st->reasoning_summary_started) {
        if (!responses_sse_reasoning_summary_part_added(fd, st)) return false;
        st->reasoning_summary_started = true;
    }
    if (!responses_sse_reasoning_delta(fd, st, text, len)) return false;
    buf_append(&st->reasoning_text, text, len);
    st->reasoning_emitted_any = true;
    return true;
}

/// Stream answer text into the message item, opening it and its text part on the first delta.
/// @param fd    the client socket
/// @param st    the stream state (the text is also accumulated in st->message_text)
/// @param text  the bytes
/// @param len   bytes at `text`
/// @return false when a write failed
static bool responses_emit_text(int fd, responses_stream *st, const char *text, size_t len) {
    if (!st->message_item_opened) {
        st->message_index = st->next_output_index++;
        if (!responses_sse_message_added(fd, st)) return false;
        st->message_item_opened = true;
    }
    if (!st->message_text_part_open) {
        if (!responses_sse_message_text_part_added(fd, st)) return false;
        st->message_text_part_open = true;
    }
    if (!responses_sse_output_text_delta(fd, st, text, len)) return false;
    buf_append(&st->message_text, text, len);
    st->message_emitted_any = true;
    return true;
}



/* ---- L267: the Responses sink (chat_sink) ------------------------------------ */

/* Reasoning reaches the client only when it opted into a summary
 * (reasoning.summary); tool calls go out with the response's completion
 * (responses_sse_finish), not as they decode. */
/// chat_sink::text for Responses.
/// @param k          the sink
/// @param reasoning  the text is reasoning (sent only with reasoning_summary_emit)
/// @param text       the bytes
/// @param len        bytes at `text` (0, or an inactive stream: nothing sent)
/// @return false when a write failed
static bool responses_sink_text_cb(chat_sink *k, bool reasoning, const char *text, size_t len, size_t) {
    responses_stream *st = (responses_stream *)k->st;
    if (!st->active || len == 0) return true;
    if (reasoning) return !k->r->reasoning_summary_emit || responses_emit_reasoning(k->fd, st, text, len);
    return responses_emit_text(k->fd, st, text, len);
}

/* A reasoning section that ended at the model's own </think> is a complete
 * reasoning item (not one a stop or an error cut short). */
/// chat_sink::end for Responses.
/// @param k             the sink
/// @param think_closed  the section ended at the model's own reasoning close
/// @return true
static bool responses_sink_end_cb(chat_sink *k, bool think_closed) {
    if (think_closed) ((responses_stream *)k->st)->reasoning_closed_naturally = true;
    return true;
}

/// Build the Responses protocol sink over `st` (no live tool events: calls go out with the finish).
/// @param k   the sink to fill
/// @param fd  the client socket
/// @param s   the server (NULL in unit tests)
/// @param r   the request
/// @param id  the response id (borrowed for the sink's life)
/// @param st  the stream state (responses_stream_init)
void responses_sink_init(chat_sink *k, int fd, server *s, const request *r, const char *id,
                         responses_stream *st) {
    *k = {fd, s, r, id, st, responses_sink_text_cb, responses_sink_end_cb, NULL, false};
}



bool responses_sse_finish(int fd, const request *r, responses_stream *st, const char *recovered_tail,
                          size_t tail_len, const tool_calls *calls, const char *finish, int prompt_tokens,
                          int completion_tokens, long created_at) {
    /* Close any half-open reasoning summary so the TUI knows the part ended
     * before we slot in any tool calls or completion. */
    if (st->reasoning_item_opened && !st->reasoning_item_closed) {
        if (!responses_sse_reasoning_done(fd, st, finish)) return false;
        st->reasoning_item_closed = true;
    }
    /* Recovery path: when DSML tool parsing fails the worker promotes the entire
     * generation to assistant text.  Streaming had already stopped at the tool
     * marker, so the bytes after the walk's position never made it to the
     * client: they go out as additional output_text deltas so what the client
     * accumulates matches output_item.done and the terminal response. */
    if (recovered_tail && tail_len && !responses_emit_text(fd, st, recovered_tail, tail_len)) return false;
    if (st->message_item_opened && !st->message_item_closed) {
        if (!responses_sse_message_done(fd, st, finish)) return false;
        st->message_item_closed = true;
    }
    responses_tool_item *items = NULL;
    responses_tool_items_build(&items, calls, st->next_output_index);
    if (items && calls) st->next_output_index += calls->len;
    bool ok = true;
    if (items && calls) {
        for (int i = 0; i < calls->len && ok; i++) {
            ok = responses_sse_function_call_event(fd, st, &calls->v[i], &items[i],
                                                   &r->tool_orders, finish, false);
            if (ok) ok = responses_sse_function_call_arguments_done(fd, st, &calls->v[i],
                                                                    &items[i],
                                                                    &r->tool_orders);
            if (ok) ok = responses_sse_function_call_event(fd, st, &calls->v[i], &items[i],
                                                           &r->tool_orders, finish, true);
        }
    }
    if (ok) ok = responses_sse_completed(fd, r, st, calls, items, finish,
                                         prompt_tokens, completion_tokens, created_at);
    free(items);
    return ok;
}



/// Send a non-streamed Responses body (fresh ids; the reasoning item only with reasoning_summary_emit).
/// @param fd                 the client socket
/// @param r                  the request
/// @param id                 the request's response id (unused: the body's ids are drawn here)
/// @param text               the answer text (may be NULL)
/// @param reasoning          the reasoning (may be NULL)
/// @param calls              the turn's calls (may be NULL)
/// @param finish             the finish label
/// @param prompt_tokens      the prompt's tokens
/// @param completion_tokens  the tokens generated
/// @return false when the write failed
bool responses_final_response(int fd,
                                     const request *r, const char *id,
                                     const char *text, const char *reasoning,
                                     const tool_calls *calls, const char *finish,
                                     int prompt_tokens, int completion_tokens) {
    (void)id;
    char response_id[40], reasoning_id[40], message_id[40];
    random_prefixed_id(response_id, sizeof(response_id), "resp_", 12);
    random_prefixed_id(reasoning_id, sizeof(reasoning_id), "rs_", 12);
    random_prefixed_id(message_id, sizeof(message_id), "msg_", 12);

    responses_tool_item *items = NULL;
    responses_tool_items_build(&items, calls, 0);

    long now = (long)time(NULL);
    const char *status = responses_status_for_finish(finish);
    const char *item_status = responses_item_status_for_finish(finish);
    buf b = {0};
    buf_printf(&b,
        "{\"id\":\"%s\",\"object\":\"response\",\"created_at\":%ld,\"status\":\"%s\","
        "\"model\":",
        response_id, now, status);
    json_escape(&b, r->model);
    if (finish && !strcmp(finish, "error")) {
        buf_puts(&b, ",\"error\":{\"code\":\"server_error\","
                     "\"message\":\"generation failed\"}");
    } else if (finish && !strcmp(finish, "length")) {
        buf_puts(&b, ",\"incomplete_details\":{\"reason\":\"max_tokens\"}");
    }
    buf_puts(&b, ",\"output\":[");
    bool wrote = false;
    if (reasoning && reasoning[0] && r->reasoning_summary_emit) {
        /* Non-streaming path runs after the worker has post-processed the
         * generation, so any reasoning here came from a parsed assistant turn
         * where </think> was observed (otherwise the reasoning text would be
         * empty). Tag it with the response-level item_status which still flips
         * to incomplete/failed when finish is length/error. */
        buf_printf(&b,
            "{\"id\":\"%s\",\"type\":\"reasoning\",\"status\":\"%s\","
            "\"summary\":[{\"type\":\"summary_text\",\"text\":",
            reasoning_id, item_status);
        json_escape(&b, reasoning);
        buf_puts(&b, "}]}");
        wrote = true;
    }
    if (text && text[0]) {
        if (wrote) buf_putc(&b, ',');
        buf_printf(&b,
            "{\"id\":\"%s\",\"type\":\"message\",\"status\":\"%s\","
            "\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":",
            message_id, item_status);
        json_escape(&b, text);
        buf_puts(&b, ",\"annotations\":[]}]}");
        wrote = true;
    }
    if (calls && items) {
        for (int i = 0; i < calls->len; i++) {
            if (wrote) buf_putc(&b, ',');
            responses_append_function_call_item(&b, &calls->v[i], &items[i],
                                                item_status, true,
                                                &r->tool_orders);
            wrote = true;
        }
    }
    buf_putc(&b, ']');
    buf_puts(&b, ",\"usage\":");
    append_responses_usage_json(&b, r, prompt_tokens, completion_tokens);
    buf_putc(&b, '}');
    bool ok = http_response(fd, 200, "application/json", b.ptr);
    buf_free(&b);
    free(items);
    return ok;
}



/// Send a non-streamed OpenAI chat.completion or text_completion body (usage and timings appended).
/// @param fd                 the client socket
/// @param r                  the request (kind, model, tool orders)
/// @param id                 the response id
/// @param text               the answer text
/// @param reasoning          the reasoning (chat: reasoning_content; may be NULL)
/// @param calls              the turn's calls (chat only; may be NULL)
/// @param finish             the finish_reason
/// @param prompt_tokens      the prompt's tokens
/// @param completion_tokens  the tokens generated
/// @param lp                 the logprob ledger (chat; NULL or disabled: none)
/// @return false when the write failed
bool final_response(int fd,
                           const request *r, const char *id, const char *text,
                           const char *reasoning, const tool_calls *calls, const char *finish,
                           int prompt_tokens, int completion_tokens,
                           const logprob_ledger *lp) {
    buf b = {0};
    long now = (long)time(NULL);
    if (r->kind == REQ_CHAT) {
        buf_printf(&b, "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        json_escape(&b, text ? text : "");
        if (reasoning && reasoning[0]) {
            buf_puts(&b, ",\"reasoning_content\":");
            json_escape(&b, reasoning);
        }
        if (calls && calls->len) {
            buf_puts(&b, ",\"tool_calls\":");
            append_tool_calls_json(&b, calls, id);
        }
        buf_putc(&b, '}');
        /* Sibling of message/finish_reason inside the choice, as in the OpenAI
         * schema; absent unless the client asked for it. */
        append_openai_logprobs_json(&b, lp, 0, lp ? lp->len : 0);
        buf_puts(&b, ",\"finish_reason\":");
        json_escape(&b, finish);
        buf_puts(&b, "}],\"usage\":");
    } else {
        buf_printf(&b, "{\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"text\":");
        json_escape(&b, text);
        buf_puts(&b, ",\"index\":0,\"finish_reason\":");
        json_escape(&b, finish);
        buf_puts(&b, "}],\"usage\":");
    }
    append_openai_usage_json(&b, r, prompt_tokens, completion_tokens);
    append_openai_timings_json(&b, r);
    buf_puts(&b, "}\n");
    bool ok = http_response(fd, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}


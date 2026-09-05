#include "pulsar_server_internal.h"



static long long wall_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}



/* Thread-local: only the GPU worker installs a writer (around a job), so
 * client threads always see NULL here and keep the classic bounded-blocking
 * send_all behavior with zero contention. */
static __thread slot_writer *g_slot_writer;



void slot_writer_init(slot_writer *w, int fd) {
    memset(w, 0, sizeof(*w));
    w->fd = fd;
}



void slot_writer_install(slot_writer *w) {
    g_slot_writer = w;
}



void slot_writer_free(slot_writer *w) {
    if (!w) return;
    if (g_slot_writer == w) g_slot_writer = NULL;
    buf_free(&w->pending);
    w->off = 0;
    w->stall_deadline_ms = 0;
}



/* Push queued bytes without ever sleeping. Returns false once the writer has
 * failed (hard socket error, stall timeout with bytes still queued, or pending
 * overflow); bytes may legitimately remain queued on a true return. */
bool slot_writer_flush(slot_writer *w) {
    if (w->failed) return false;
    while (w->off < w->pending.len) {
        ssize_t r = send(w->fd, w->pending.ptr + w->off, w->pending.len - w->off, 0);
        if (r > 0) {
            w->off += (size_t)r;
            w->stall_deadline_ms = wall_ms() + PULSAR_SERVER_SEND_STALL_TIMEOUT_MS;
            continue;
        }
        if (r < 0 && errno == EINTR) continue;
        if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
        w->failed = true;
        return false;
    }
    if (w->off == w->pending.len) {
        w->pending.len = 0;
        w->off = 0;
        w->stall_deadline_ms = 0;
        return true;
    }
    if (w->off >= 65536) {
        /* Compact the consumed prefix so a long slow stream cannot grow the
         * queue by its full transferred size. */
        memmove(w->pending.ptr, w->pending.ptr + w->off, w->pending.len - w->off);
        w->pending.len -= w->off;
        w->off = 0;
    }
    if (wall_ms() >= w->stall_deadline_ms) {
        w->failed = true;
        return false;
    }
    return true;
}



static bool slot_writer_send(slot_writer *w, const void *p, size_t n) {
    if (w->failed) return false;
    if (g_stop_requested) {
        /* Match blocking send_all: any write during shutdown fails. */
        w->failed = true;
        return false;
    }
    /* Stamp BEFORE the fast path below, which can return early once the bytes
     * go straight to the wire — stamping at the buf_append would miss exactly
     * the case where the client is keeping up (the common one). */
    w->last_write_ms = wall_ms();
    const char *s = (const char *)p;
    if (w->off == w->pending.len) {
        /* Nothing queued: try the wire directly so hard errors (EPIPE from a
         * closed peer) surface on this call, exactly like the blocking path. */
        while (n) {
            ssize_t r = send(w->fd, s, n, 0);
            if (r > 0) {
                s += r;
                n -= (size_t)r;
                continue;
            }
            if (r < 0 && errno == EINTR) continue;
            if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
            w->failed = true;
            return false;
        }
        if (!n) return true;
        w->pending.len = 0;
        w->off = 0;
    }
    buf_append(&w->pending, s, n);
    if (w->pending.len - w->off > PULSAR_SERVER_WRITER_MAX_PENDING_BYTES) {
        w->failed = true;
        return false;
    }
    if (!w->stall_deadline_ms) {
        w->stall_deadline_ms = wall_ms() + PULSAR_SERVER_SEND_STALL_TIMEOUT_MS;
    }
    return slot_writer_flush(w);
}



bool slot_writer_idle_for(const slot_writer *w, long long now_ms, long long interval_ms) {
    if (!w || w->failed || w->fd < 0) return false;
    if (!w->last_write_ms) return false;   /* nothing sent yet: no clock to judge */
    return now_ms - w->last_write_ms >= interval_ms;
}



/* Surface-appropriate keepalive for a slot that has gone quiet mid-request.
 *
 * Anthropic gets a REAL `ping` event: it is documented, and clients are told to
 * ignore unknown/ping events, so it costs nothing.
 *
 * OpenAI-chat and Responses get an SSE COMMENT (": ping"). A comment is
 * discarded by every conformant SSE parser before it reaches application code,
 * so it cannot perturb the delta stream. NOTE this DIVERGES from upstream
 * Entrpi 40ca8d5, which sends a real `response.in_progress` on the Responses
 * surface. Deliberate: `response.in_progress` is a lifecycle event carrying a
 * `sequence_number`, and our responses_stream hands out monotonic indices
 * (st->next_output_index) — repeating it would either duplicate or advance the
 * sequence for a non-event, which strict clients may reject. A comment keeps
 * the socket warm with zero protocol surface. */
bool gen_stream_heartbeat(gen_state *g) {
    if (!g) return false;
    if (!slot_writer_idle_for(&g->writer, wall_ms(), PULSAR_SERVER_HEARTBEAT_MS)) return false;

    const char *beat = NULL;
    if (g->anthropic_live.active) {
        beat = "event: ping\ndata: {\"type\": \"ping\"}\n\n";
    } else if (g->responses_live.active || g->openai_live.active) {
        beat = ": ping\n\n";
    }
    if (!beat) return false;   /* non-streaming request: nothing to keep alive */

    /* Write through the slot's OWN writer, not send_all: send_all only becomes
     * non-blocking when that writer is INSTALLED thread-locally, which it is
     * not during the worker's sweep — it would block the single GPU worker on
     * a slow client. slot_writer_send is non-blocking and re-stamps
     * last_write_ms, so the next beat is one full interval away. */
    return slot_writer_send(&g->writer, beat, strlen(beat));
}



/* Blocking completion of all queued bytes, used once per job after the final
 * response: same poll cadence and stall timeout as blocking send_all. */
bool slot_writer_drain(slot_writer *w) {
    while (!w->failed && w->off < w->pending.len) {
        if (g_stop_requested) {
            w->failed = true;
            break;
        }
        if (!slot_writer_flush(w)) break;
        if (w->off >= w->pending.len) break;
        long long remaining = w->stall_deadline_ms - wall_ms();
        if (remaining <= 0) {
            w->failed = true;
            break;
        }
        struct pollfd pfd = {.fd = w->fd, .events = POLLOUT};
        int timeout = remaining > 50 ? 50 : (int)remaining;
        int rc;
        do {
            rc = poll(&pfd, 1, timeout);
        } while (rc < 0 && errno == EINTR);
        if (rc < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) {
            w->failed = true;
            break;
        }
    }
    return !w->failed;
}



bool send_all(int fd, const void *p, size_t n) {
    if (g_slot_writer && g_slot_writer->fd == fd) {
        return slot_writer_send(g_slot_writer, p, n);
    }
    const char *s = (const char *)p;
    long long deadline = wall_ms() + PULSAR_SERVER_SEND_STALL_TIMEOUT_MS;
    while (n) {
        if (g_stop_requested) return false;
        ssize_t w = send(fd, s, n, 0);
        if (w < 0 && errno == EINTR) continue;
        if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            long long remaining = deadline - wall_ms();
            if (remaining <= 0) return false;
            struct pollfd pfd = {.fd = fd, .events = POLLOUT};
            int timeout = remaining > 50 ? 50 : (int)remaining;
            int rc;
            do {
                rc = poll(&pfd, 1, timeout);
            } while (rc < 0 && errno == EINTR);
            if (rc < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) return false;
            continue;
        }
        if (w <= 0) return false;
        s += w;
        n -= (size_t)w;
        deadline = wall_ms() + PULSAR_SERVER_SEND_STALL_TIMEOUT_MS;
    }
    return true;
}



void json_escape(buf *b, const char *s) {
    buf_putc(b, '"');
    json_escape_fragment_n(b, s, strlen(s));
    buf_putc(b, '"');
}



void json_escape_n(buf *b, const char *s, size_t n) {
    char *tmp = xstrndup(s ? s : "", n);
    json_escape(b, tmp);
    free(tmp);
}



void json_escape_fragment_n(buf *b, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            buf_putc(b, '\\');
            buf_putc(b, (char)c);
        } else if (c == '\n') {
            buf_puts(b, "\\n");
        } else if (c == '\r') {
            buf_puts(b, "\\r");
        } else if (c == '\t') {
            buf_puts(b, "\\t");
        } else if (c < 0x20) {
            buf_printf(b, "\\u%04x", (unsigned)c);
        } else {
            buf_putc(b, (char)c);
        }
    }
}


/* Trim a TRAILING partial DSML closing tag of ANY style from [start, len).
 * A truncated generation can end mid-closing-tag, and the model sometimes
 * mixes styles (opens short "<DSML|...>", closes long "</|DSML|...>"), so
 * the active-style hold in the streamers cannot recognize the tail.  Any
 * suffix beginning at a '<' that is a prefix of ANY known closing form is
 * tag debris, never value content.  Returns len unchanged when the tail is
 * not tag-shaped. */
size_t trim_truncated_dsml_close_tail(const char *raw, size_t start, size_t len) {
    /* Every closing literal of every syntax in the table. */
    const char *ends[PULSAR_DSML_SYNTAXES * 3];
    size_t max_tag = 0;
    for (size_t i = 0; i < PULSAR_DSML_SYNTAXES; i++) {
        ends[i * 3 + 0] = pulsar_dsml_syntaxes[i].param_end;
        ends[i * 3 + 1] = pulsar_dsml_syntaxes[i].invoke_end;
        ends[i * 3 + 2] = pulsar_dsml_syntaxes[i].tool_calls_end;
    }
    for (size_t i = 0; i < sizeof(ends) / sizeof(ends[0]); i++) {
        const size_t l = strlen(ends[i]);
        if (l > max_tag) max_tag = l;
    }
    const size_t scan = len - start > max_tag ? len - max_tag : start;
    for (size_t i = len; i > scan; i--) {
        if (raw[i - 1] != '<') continue;
        const size_t marker = i - 1;
        const size_t tail = len - marker;
        for (size_t e = 0; e < sizeof(ends) / sizeof(ends[0]); e++) {
            if (tail < strlen(ends[e]) && !memcmp(raw + marker, ends[e], tail))
                return marker;
        }
        break;
    }
    return len;
}

/* The EARLIEST occurrence of any syntax's tool_calls opener (closer) in s. */
const char *find_any_tool_start(const char *s) {
    const char *best = NULL;
    for (size_t i = 0; i < PULSAR_DSML_SYNTAXES; i++) {
        const char *p = strstr(s, pulsar_dsml_syntaxes[i].tool_calls_start);
        if (p && (!best || p < best)) best = p;
    }
    return best;
}



const char *find_any_tool_end(const char *s) {
    const char *best = NULL;
    for (size_t i = 0; i < PULSAR_DSML_SYNTAXES; i++) {
        const char *p = strstr(s, pulsar_dsml_syntaxes[i].tool_calls_end);
        if (p && (!best || p < best)) best = p;
    }
    return best;
}



void observe_tool_markers(const char *scan, bool *saw_start,
                                 bool *saw_end, bool *orphan_end) {
    if (!scan) return;
    bool had_start = *saw_start;
    const char *start = find_any_tool_start(scan);
    if (start) *saw_start = true;

    const char *end_scan = had_start ? scan : (start ? start : NULL);
    const char *end = end_scan ? find_any_tool_end(end_scan) : NULL;
    if (end) {
        *saw_end = true;
    } else if (!had_start && !start && find_any_tool_end(scan)) {
        if (orphan_end) *orphan_end = true;
    }
}



size_t trim_tool_separator_ws(const char *raw, size_t start, size_t limit) {
    while (limit > start && isspace((unsigned char)raw[limit - 1])) limit--;
    return limit;
}



static const char *skip_ascii_ws(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}



const char *find_last_substr(const char *s, const char *needle) {
    if (!s || !needle || !needle[0]) return NULL;
    const char *last = NULL;
    const char *p = s;
    while ((p = strstr(p, needle)) != NULL) {
        last = p;
        p++;
    }
    return last;
}



static void tool_call_json_args_add(buf *args, const char *name, const char *value, const char *is_string) {
    if (args->len) buf_puts(args, ", ");
    json_escape(args, name ? name : "");
    buf_puts(args, ": ");
    if (is_string && !strcmp(is_string, "true")) {
        json_escape(args, value ? value : "");
    } else {
        char *min = json_minify_raw_value(value ? value : "null");
        buf_puts(args, min && min[0] ? min : "null");
        free(min);
    }
}



/* DSML produced by the model is usually a flat list of typed parameters:
 *
 *   <parameter name="path" string="true">/tmp/x</parameter>
 *   <parameter name="timeout" string="false">10</parameter>
 *
 * Long generations sometimes drift into a looser XML-ish shape, omitting the
 * outer string attribute and putting child parameters inside it.  The server
 * does not know client tool schemas, so it cannot make that semantically
 * perfect.  Still, returning a structured JSON value lets the client/tool layer
 * reject or repair the call, which is much better than aborting the assistant
 * turn and losing the whole sampled continuation.
 */
static bool dsml_parse_leaf_param_json(const char **p_in, const char *param_start,
                                       const char *param_end, buf *out) {
    const char *p = *p_in;
    if (strncmp(p, param_start, strlen(param_start)) != 0) return false;
    const char *tag_end = strchr(p, '>');
    if (!tag_end) return false;

    char *tag = xstrndup(p, (size_t)(tag_end - p + 1));
    char *name = pulsar_dsml_attr(tag, "name");
    char *is_string = pulsar_dsml_attr(tag, "string");
    free(tag);
    if (!name) {
        free(is_string);
        return false;
    }

    const char *value_start = tag_end + 1;
    const char *value_end = strstr(value_start, param_end);
    if (!value_end) {
        free(name);
        free(is_string);
        return false;
    }

    char *raw_value = xstrndup(value_start, (size_t)(value_end - value_start));
    const char *type = is_string ? is_string : "true";
    char *value = !strcmp(type, "true") ?
        pulsar_dsml_unescape(raw_value) : xstrdup(raw_value);
    tool_call_json_args_add(out, name, value, type);

    free(name);
    free(is_string);
    free(raw_value);
    free(value);
    *p_in = value_end + strlen(param_end);
    return true;
}



static bool dsml_parse_nested_params_object(const char **p_in,
                                            const char *param_start,
                                            const char *param_end,
                                            buf *out) {
    const char *p = *p_in;
    buf members = {0};
    bool any = false;

    for (;;) {
        p = skip_ascii_ws(p);
        if (strncmp(p, param_start, strlen(param_start)) != 0) break;
        if (!dsml_parse_leaf_param_json(&p, param_start, param_end, &members)) {
            buf_free(&members);
            return false;
        }
        any = true;
    }

    if (!any) {
        buf_free(&members);
        return false;
    }
    buf_putc(out, '{');
    buf_puts(out, members.ptr ? members.ptr : "");
    buf_putc(out, '}');
    buf_free(&members);
    *p_in = p;
    return true;
}



static void split_reasoning_content(const char *text, size_t n, char **content_out, char **reasoning_out) {
    char *s = xstrndup(text ? text : "", n);
    char *body = s;
    const bool opened_think = !strncmp(body, "<think>", 7);
    if (opened_think) body += 7;

    char *think_end = strstr(body, "</think>");
    if (think_end) {
        *think_end = '\0';
        *reasoning_out = xstrdup(body);
        *content_out = xstrdup(think_end + 8);
    } else if (opened_think) {
        /* Generation ended inside the think block (token cap / stop). The
         * partial reasoning is still reasoning: surfacing it as content
         * hands raw chain-of-thought to clients that score or display the
         * answer channel (the streaming path already promises reasoning is
         * never visible assistant text). */
        *reasoning_out = xstrdup(body);
        *content_out = xstrdup("");
    } else {
        *reasoning_out = NULL;
        *content_out = xstrdup(s);
    }
    free(s);
}



/* The prefix before a tool call recovered from UNCLOSED reasoning is
 * reasoning, not content (upstream ds4 51a1c14). */
static void unterminated_reasoning_before_tool(const char *text,
                                               size_t prefix_len,
                                               char **content_out,
                                               char **reasoning_out) {
    const char *body = text ? text : "";
    if (prefix_len > strlen(body)) prefix_len = strlen(body);
    if (prefix_len >= 7 && !strncmp(body, "<think>", 7)) {
        body += 7;
        prefix_len -= 7;
    }
    *reasoning_out = xstrndup(body, prefix_len);
    *content_out = xstrdup("");
}

bool parse_generated_message_ex(const char *text, bool require_thinking_closed,
                                       char **content_out, char **reasoning_out,
                                       tool_calls *calls) {
    text = text ? text : "";
    const char *tool_search = text;
    bool recovered_unclosed_tool = false;

    /* When thinking mode is enabled the model is expected to close
     * </think> before it enters the executable assistant surface.  DSML inside
     * reasoning is just model text: it may be a mistaken attempt, a quotation,
     * or an explanation of the protocol.  Treating it as a real tool call
     * duplicates it into both reasoning and structured tool_calls, and can make
     * clients execute something the assistant had not actually emitted as its
     * post-thinking action. */
    if (require_thinking_closed) {
        const char *think_end = find_last_substr(text, "</think>");
        if (!think_end) {
            /* Model did not close thinking. A COMPLETE tool block in the
             * buffer is unambiguous enough to recover structurally (upstream
             * ds4 51a1c14); anything less — truncation, a quoted opening —
             * stays reasoning, even when the <think> opener lives in the
             * prompt rather than the generation, and off the content
             * channel. */
            const char *candidate = find_any_tool_start(text);
            if (!candidate || !find_any_tool_end(candidate)) {
                server_log(PULSAR_LOG_TOOL,
                           "thinking not closed, ignoring incomplete DSML in reasoning");
                const char *body = !strncmp(text, "<think>", 7) ? text + 7 : text;
                *reasoning_out = xstrdup(body);
                *content_out = xstrdup("");
                return true;
            }
            tool_search = candidate;
            recovered_unclosed_tool = true;
        } else {
            tool_search = think_end + 8;
        }
    }

    /* Which syntax opens the block: table order is the preference (canonical
     * first), and within one syntax the "\n\n"-separated opener is preferred
     * over a bare one -- the same ranking the pre-L184 if/else chain had. */
    const char *start = NULL;
    const pulsar_dsml_syntax *syn = NULL;
    for (size_t i = 0; i < PULSAR_DSML_SYNTAXES && !start; i++) {
        const pulsar_dsml_syntax *cand = &pulsar_dsml_syntaxes[i];
        char sep[80];
        snprintf(sep, sizeof sep, "\n\n%s", cand->tool_calls_start);
        start = strstr(tool_search, sep);
        if (!start) start = strstr(tool_search, cand->tool_calls_start);
        if (start) syn = cand;
    }
    if (!start) {
        split_reasoning_content(text, strlen(text), content_out, reasoning_out);
        return true;
    }

    size_t content_len = trim_tool_separator_ws(text, 0, (size_t)(start - text));
    const char *raw_block_start = start;
    const char *tool_calls_start = syn->tool_calls_start;
    const char *tool_calls_end = syn->tool_calls_end;
    const char *invoke_start = syn->invoke_start;
    const char *invoke_end = syn->invoke_end;
    const char *param_start = syn->param_start;
    const char *param_end = syn->param_end;

    const char *p = strstr(start, tool_calls_start);
    if (!p) return false;
    p += strlen(tool_calls_start);

    for (;;) {
        p = skip_ascii_ws(p);
        if (!strncmp(p, tool_calls_end, strlen(tool_calls_end))) {
            const char *raw_block_end = p + strlen(tool_calls_end);
            free(calls->raw_dsml);
            calls->raw_dsml = xstrndup(raw_block_start, (size_t)(raw_block_end - raw_block_start));
            if (recovered_unclosed_tool) {
                unterminated_reasoning_before_tool(text, content_len,
                                                   content_out, reasoning_out);
            } else {
                split_reasoning_content(text, content_len, content_out, reasoning_out);
            }
            return true;
        }
        if (strncmp(p, invoke_start, strlen(invoke_start)) != 0) return false;
        const char *tag_end = strchr(p, '>');
        if (!tag_end) return false;
        char *tag = xstrndup(p, (size_t)(tag_end - p + 1));
        char *name = pulsar_dsml_attr(tag, "name");
        free(tag);
        if (!name) return false;
        p = tag_end + 1;

        buf args = {0};
        while (true) {
            p = skip_ascii_ws(p);
            if (!strncmp(p, invoke_end, strlen(invoke_end))) {
                p += strlen(invoke_end);
                break;
            }
            if (strncmp(p, param_start, strlen(param_start)) != 0) {
                free(name);
                buf_free(&args);
                return false;
            }
            tag_end = strchr(p, '>');
            if (!tag_end) {
                free(name);
                buf_free(&args);
                return false;
            }
            tag = xstrndup(p, (size_t)(tag_end - p + 1));
            char *param_name = pulsar_dsml_attr(tag, "name");
            char *param_is_string = pulsar_dsml_attr(tag, "string");
            free(tag);
            if (!param_name) {
                free(name);
                free(param_name);
                free(param_is_string);
                buf_free(&args);
                return false;
            }
            const char *value_start = tag_end + 1;
            if (!param_is_string &&
                !strncmp(skip_ascii_ws(value_start), param_start, strlen(param_start)))
            {
                buf nested = {0};
                const char *nested_p = value_start;
                if (!dsml_parse_nested_params_object(&nested_p, param_start,
                                                     param_end, &nested)) {
                    free(name);
                    free(param_name);
                    buf_free(&nested);
                    buf_free(&args);
                    return false;
                }
                tool_call_json_args_add(&args, param_name,
                                        nested.ptr ? nested.ptr : "{}",
                                        "false");
                buf_free(&nested);
                p = skip_ascii_ws(nested_p);
                if (!strncmp(p, param_end, strlen(param_end))) {
                    p += strlen(param_end);
                }
                free(param_name);
                continue;
            }
            const char *value_end = strstr(value_start, param_end);
            if (!value_end) {
                free(name);
                free(param_name);
                free(param_is_string);
                buf_free(&args);
                return false;
            }
            char *raw_value = xstrndup(value_start, (size_t)(value_end - value_start));
            const char *type = param_is_string ? param_is_string : "true";
            char *value = !strcmp(type, "true") ?
                pulsar_dsml_unescape(raw_value) : xstrdup(raw_value);
            tool_call_json_args_add(&args, param_name, value, type);
            free(param_name);
            free(param_is_string);
            free(raw_value);
            free(value);
            p = value_end + strlen(param_end);
        }

        tool_call tc = {0};
        tc.name = name;
        buf wrapped = {0};
        buf_putc(&wrapped, '{');
        buf_puts(&wrapped, args.ptr ? args.ptr : "");
        buf_putc(&wrapped, '}');
        tc.arguments = buf_take(&wrapped);
        tool_calls_push(calls, tc);
        buf_free(&args);
    }
}



/* Try to repair a truncated DSML block.
 *
 * DSML nesting order is: tool_calls > invoke > parameter.
 * Single-pass scan: count opens vs closes, then append missing closing tags.
 *
 * Returns true if repair was applied, false if the text had no recognizable DSML
 * or was already balanced.  This deliberately does not rewrite malformed but
 * balanced DSML into assistant text; semantic recovery belongs to the model. */
bool try_repair_dsml(const char *s, size_t len, buf *out) {
    if (!s || !len) return false;

    /* Only scan DSML tags after the last </think>.  DSML mentioned inside
     * reasoning is not executable — it inflates tag counts and causes false
     * positive repairs.  If no </think> is found, scan from the start
     * (thinking mode is not active or thinking was never opened). */
    const char *think_end = find_last_substr(s, "</think>");
    const char *scan_start = think_end ? (think_end + 8) : s;
    size_t scan_len = (size_t)((s + len) - scan_start);

    /* Detect style from first <tool_calls> tag */
    const char *ts, *te, *is, *ie, *ps, *pe;
    if (strstr(scan_start, PULSAR_TOOL_CALLS_START)) {
        ts = PULSAR_TOOL_CALLS_START;  te = PULSAR_TOOL_CALLS_END;
        is = PULSAR_INVOKE_START;      ie = PULSAR_INVOKE_END;
        ps = PULSAR_PARAM_START;       pe = PULSAR_PARAM_END;
    } else if (strstr(scan_start, PULSAR_TOOL_CALLS_START_SHORT)) {
        ts = PULSAR_TOOL_CALLS_START_SHORT;  te = PULSAR_TOOL_CALLS_END_SHORT;
        is = PULSAR_INVOKE_START_SHORT;      ie = PULSAR_INVOKE_END_SHORT;
        ps = PULSAR_PARAM_START_SHORT;       pe = PULSAR_PARAM_END_SHORT;
    } else if (strstr(scan_start, "<tool_calls>")) {
        ts = "<tool_calls>";   te = "</tool_calls>";
        is = "<invoke";        ie = "</invoke>";
        ps = "<parameter";     pe = "</parameter>";
    } else {
        return false; /* No recognizable DSML start tag */
    }

    /* Single-pass: count all 6 tag types in one scan */
    size_t tos = 0, toe = 0, ios = 0, ioe = 0, pos = 0, poe = 0;
    const char *e = scan_start + scan_len;
    for (const char *p = scan_start; p < e; ) {
        size_t d;
        if ((d = strlen(ts)) && !strncmp(p, ts, d)) { tos++; p += d; }
        else if ((d = strlen(te)) && !strncmp(p, te, d)) { toe++; p += d; }
        else if ((d = strlen(is)) && !strncmp(p, is, d)) { ios++; p += d; }
        else if ((d = strlen(ie)) && !strncmp(p, ie, d)) { ioe++; p += d; }
        else if ((d = strlen(ps)) && !strncmp(p, ps, d)) { pos++; p += d; }
        else if ((d = strlen(pe)) && !strncmp(p, pe, d)) { poe++; p += d; }
        else p++;
    }
    if (tos == toe && ios == ioe && pos == poe) return false;
    if (toe > tos || ioe > ios || poe > pos) {
        /* Extra closing tags are not a truncation pattern.  Refuse repair so the
         * unsigned differences below cannot wrap and append a huge suffix. */
        return false;
    }
    /* Repair: copy original text and append missing closing tags in reverse
     * order.  First TRIM a trailing partial closing tag: the model was cut
     * mid-"</...>", and appending a fresh close after the fragment would bake
     * tag debris into the parsed parameter VALUE (seen live: a truncated bash
     * call carrying "</｜DSML｜" inside command).  The fragment counts no tag
     * above, so the appended deficit is unchanged by the trim. */
    size_t keep = trim_truncated_dsml_close_tail(s, (size_t)(scan_start - s), len);
    buf_append(out, s, keep);
    for (size_t i = 0; i < pos - poe; i++) buf_puts(out, pe);
    for (size_t i = 0; i < ios - ioe; i++) buf_puts(out, ie);
    for (size_t i = 0; i < tos - toe; i++) buf_puts(out, te);
    return true;
}



static const char *tool_parse_failure_recovery_finish(const char *finish) {
    /* Once DSML failed to parse there is no executable tool call to report.
     * Preserve a true length stop, because callers can distinguish truncation
     * from a completed turn.  Every other non-error tool-parse failure becomes
     * a normal assistant stop with the raw model text returned as content. */
    if (finish && !strcmp(finish, "length")) return "length";
    return "stop";
}



bool parse_generated_message_for_response(const char *text,
                                                 bool has_tools,
                                                 bool saw_tool_start,
                                                 bool require_thinking_closed,
                                                 const char **finish_io,
                                                 char *err,
                                                 size_t errlen,
                                                 char **content_out,
                                                 char **reasoning_out,
                                                 tool_calls *calls,
                                                 bool *recovered_out) {
    if (recovered_out) *recovered_out = false;

    bool parsed_ok = parse_generated_message_ex(text ? text : "",
                                                require_thinking_closed,
                                                content_out, reasoning_out,
                                                calls);
    if (parsed_ok) return true;

    free(*content_out);
    free(*reasoning_out);
    *content_out = xstrdup(text ? text : "");
    *reasoning_out = NULL;
    tool_calls_free(calls);

    /* A malformed tool block is model output, not a server failure.  The
     * generation worker may hide this turn from the client, append a tool error
     * plus protocol reminder to the live session, and let the model try again.
     * If that continuation is unavailable, parsed_content keeps the raw text as
     * a last-resort assistant fallback instead of crashing the request. */
    const char *finish = finish_io && *finish_io ? *finish_io : "stop";
    if (has_tools && saw_tool_start && strcmp(finish, "error") != 0) {
        if (finish_io) *finish_io = tool_parse_failure_recovery_finish(finish);
        if (err && errlen) snprintf(err, errlen, "invalid tool call");
        if (recovered_out) *recovered_out = true;
    }
    return false;
}



void append_json_object_string(buf *b, const char *json) {
    buf tmp = {0};
    append_json_object_or_empty(&tmp, json);
    json_escape(b, tmp.ptr ? tmp.ptr : "{}");
    buf_free(&tmp);
}



void append_tool_calls_json(buf *b, const tool_calls *calls, const char *id_prefix,
                                   const tool_schema_orders *orders) {
    (void)orders;
    buf_putc(b, '[');
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        if (i) buf_putc(b, ',');
        char idbuf[128];
        snprintf(idbuf, sizeof(idbuf), "%s_tool_%d", id_prefix, i);
        buf_puts(b, "{\"id\":");
        json_escape(b, tc->id ? tc->id : idbuf);
        buf_puts(b, ",\"type\":\"function\",\"function\":{\"name\":");
        json_escape(b, tc->name ? tc->name : "");
        buf_puts(b, ",\"arguments\":");
        append_json_object_string(b, tc->arguments);
        buf_puts(b, "}}");
    }
    buf_putc(b, ']');
}



void append_tool_call_deltas_json(buf *b, const tool_calls *calls, const char *id_prefix,
                                         const tool_schema_orders *orders) {
    (void)orders;
    buf_putc(b, '[');
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        if (i) buf_putc(b, ',');
        char idbuf[128];
        snprintf(idbuf, sizeof(idbuf), "%s_tool_%d", id_prefix, i);
        buf_puts(b, "{\"index\":");
        buf_printf(b, "%d", i);
        buf_puts(b, ",\"id\":");
        json_escape(b, tc->id ? tc->id : idbuf);
        buf_puts(b, ",\"type\":\"function\",\"function\":{\"name\":");
        json_escape(b, tc->name ? tc->name : "");
        buf_puts(b, ",\"arguments\":");
        append_json_object_string(b, tc->arguments);
        buf_puts(b, "}}");
    }
    buf_putc(b, ']');
}




/* =========================================================================
 * The DSML tool-stream projection, shared by the OpenAI and Anthropic
 * streamers (L184).  See ::dsml_tool_stream in the header for the contract.
 * ========================================================================= */

void dsml_tool_stream_free(dsml_tool_stream *ts) {
    if (!ts) return;
    for (int i = 0; i < ts->ids_cap; i++) free(ts->ids[i]);
    free(ts->ids);
    ts->ids = NULL;
    ts->ids_cap = 0;
}



/* Which syntax opens the block at `pos` decides every literal the projection
 * matches from here on. */
bool dsml_tool_stream_init(dsml_tool_stream *ts, const char *raw, size_t raw_len, size_t pos) {
    dsml_tool_stream_free(ts);
    memset(ts, 0, sizeof(*ts));
    ts->active = true;
    ts->state = DSML_TOOL_BETWEEN_INVOKES;
    for (size_t i = 0; i < PULSAR_DSML_SYNTAXES; i++) {
        const pulsar_dsml_syntax *syn = &pulsar_dsml_syntaxes[i];
        if (raw_full_lit(raw, raw_len, pos, syn->tool_calls_start)) {
            ts->syn = syn;
            ts->parse_pos = pos + strlen(syn->tool_calls_start);
            return true;
        }
    }
    ts->active = false;
    ts->state = DSML_TOOL_ERROR;
    return false;
}



static bool dsml_tool_stream_has_id(const dsml_tool_stream *ts, const char *id, int upto) {
    if (!ts || !id || !id[0]) return false;
    if (upto > ts->ids_cap) upto = ts->ids_cap;
    for (int i = 0; i < upto; i++) {
        if (ts->ids[i] && !strcmp(ts->ids[i], id)) return true;
    }
    return false;
}



/* Free function (NOT a server:: method): legitimately called with a null
 * server (the unit tests project without a bound server).  As a member the
 * `!s` guard would be elided under -O3 (this assumed non-null). */
const char *dsml_tool_stream_id(server *s, dsml_tool_stream *ts, int index, api_style api) {
    if (!ts || index < 0) return "";
    if (index >= ts->ids_cap) {
        int old = ts->ids_cap;
        int cap = old ? old : 4;
        while (cap <= index) cap *= 2;
        ts->ids = (char **)server_xrealloc(ts->ids, (size_t)cap * sizeof(ts->ids[0]));
        memset(ts->ids + old, 0, (size_t)(cap - old) * sizeof(ts->ids[0]));
        ts->ids_cap = cap;
    }
    if (!ts->ids[index]) {
        char id[64];
        for (;;) {
            random_tool_id(id, sizeof(id), api);
            if (!dsml_tool_stream_has_id(ts, id, index) &&
                (!s || !s->tool_memory_has_id(id))) break;  /* null s (no bound server) => no dedup */
        }
        ts->ids[index] = xstrdup(id);
    }
    return ts->ids[index];
}



static bool dsml_tool_stream_fail(dsml_tool_stream *ts) {
    ts->active = false;
    ts->state = DSML_TOOL_ERROR;
    return true;
}



/* A string parameter's value bytes: DSML entities undone, then JSON-string
 * escaped, as a fragment inside the already-open quotes. */
static bool dsml_tool_emit_string_value(dsml_tool_stream *ts, const dsml_tool_stream_ops *ops,
                                        void *ctx, const char *text, size_t len) {
    if (len == 0) return true;
    char *raw = xstrndup(text, len);
    char *unescaped = pulsar_dsml_unescape(raw);
    buf frag = {0};
    json_escape_fragment_n(&frag, unescaped, strlen(unescaped));
    bool ok = ops->args_fragment(ctx, ts, frag.ptr ? frag.ptr : "", frag.len);
    buf_free(&frag);
    free(unescaped);
    free(raw);
    return ok;
}



static bool dsml_tool_emit_param_prefix(dsml_tool_stream *ts, const dsml_tool_stream_ops *ops,
                                        void *ctx, const char *name, bool is_string) {
    buf frag = {0};
    if (ts->first_param) ts->first_param = false;
    else buf_putc(&frag, ',');
    json_escape(&frag, name ? name : "");
    buf_putc(&frag, ':');
    if (is_string) buf_putc(&frag, '"');
    bool ok = ops->args_fragment(ctx, ts, frag.ptr ? frag.ptr : "", frag.len);
    buf_free(&frag);
    return ok;
}



static bool dsml_tool_start_invoke(dsml_tool_stream *ts, const dsml_tool_stream_ops *ops,
                                   void *ctx, const char *raw, size_t raw_len) {
    const char *tag_end = (const char *)memchr(raw + ts->parse_pos, '>', raw_len - ts->parse_pos);
    if (!tag_end) return true;
    char *tag = xstrndup(raw + ts->parse_pos, (size_t)(tag_end - (raw + ts->parse_pos) + 1));
    char *name = pulsar_dsml_attr(tag, "name");
    free(tag);
    if (!name) return dsml_tool_stream_fail(ts);

    bool ok = ops->begin_invoke(ctx, ts, name) &&
              ops->args_fragment(ctx, ts, "{", 1);
    free(name);
    if (!ok) return false;

    ts->emitted_any = true;
    ts->args_open = true;
    ts->first_param = true;
    ts->parse_pos = (size_t)(tag_end - raw) + 1;
    ts->state = DSML_TOOL_BETWEEN_PARAMS;
    return true;
}



static bool dsml_tool_start_param(dsml_tool_stream *ts, const dsml_tool_stream_ops *ops,
                                  void *ctx, const char *raw, size_t raw_len) {
    const char *tag_end = (const char *)memchr(raw + ts->parse_pos, '>', raw_len - ts->parse_pos);
    if (!tag_end) return true;
    char *tag = xstrndup(raw + ts->parse_pos, (size_t)(tag_end - (raw + ts->parse_pos) + 1));
    char *name = pulsar_dsml_attr(tag, "name");
    char *is_string = pulsar_dsml_attr(tag, "string");
    free(tag);
    if (!name || !is_string) {
        free(name);
        free(is_string);
        return dsml_tool_stream_fail(ts);
    }
    bool string_value = !strcmp(is_string, "true");
    bool ok = dsml_tool_emit_param_prefix(ts, ops, ctx, name, string_value);
    free(name);
    free(is_string);
    if (!ok) return false;

    ts->param_is_string = string_value;
    ts->parse_pos = (size_t)(tag_end - raw) + 1;
    ts->state = DSML_TOOL_PARAM_VALUE;
    return true;
}



static bool dsml_tool_emit_value(dsml_tool_stream *ts, const dsml_tool_stream_ops *ops,
                                 void *ctx, const char *raw, size_t value_end) {
    if (value_end <= ts->parse_pos) return true;
    return ts->param_is_string ?
        dsml_tool_emit_string_value(ts, ops, ctx, raw + ts->parse_pos, value_end - ts->parse_pos) :
        ops->args_fragment(ctx, ts, raw + ts->parse_pos, value_end - ts->parse_pos);
}



static bool dsml_tool_finish_param(dsml_tool_stream *ts, const dsml_tool_stream_ops *ops,
                                   void *ctx, const char *raw, size_t value_end) {
    if (!dsml_tool_emit_value(ts, ops, ctx, raw, value_end)) return false;
    if (ts->param_is_string && !ops->args_fragment(ctx, ts, "\"", 1)) return false;
    ts->parse_pos = value_end + strlen(ts->syn->param_end);
    ts->state = DSML_TOOL_BETWEEN_PARAMS;
    return true;
}



/* The invocation's "}" and the protocol's block stop, then the next index. */
static bool dsml_tool_close_invoke(dsml_tool_stream *ts, const dsml_tool_stream_ops *ops, void *ctx) {
    if (ts->args_open && !ops->args_fragment(ctx, ts, "}", 1)) return false;
    ts->args_open = false;
    if (!ops->end_invoke(ctx, ts)) return false;
    ts->index++;
    return true;
}



/* Generation ended (final) while a streamed tool call is still open: the
 * client has already received the call header and a prefix of the argument
 * deltas, so the truncation cannot be reclassified as text (the non-stream
 * path's try_repair_dsml equivalent).  What CAN be guaranteed is well-formed
 * wire JSON: flush the un-emitted value bytes, close an open string value,
 * close the args object, stop the block.  The argument VALUE stays truncated
 * -- exactly what the repaired non-stream parse of the same bytes yields --
 * and the finish reason (length) still tells the client the turn was cut. */
bool dsml_tool_stream_finalize(dsml_tool_stream *ts, const dsml_tool_stream_ops *ops, void *ctx,
                               const char *raw, size_t raw_len) {
    if (!ts->active) return true;
    if (ts->state == DSML_TOOL_PARAM_VALUE) {
        /* Flush only up to the stream-safe limit: the held-back tail is a
         * partial closing tag (the model was mid-"</...parameter>") or a
         * split UTF-8 sequence -- tag debris and mojibake, never value
         * content.  Dropping it beats the non-stream repair here, which
         * keeps the fragment in the value. */
        size_t limit = tool_param_value_stream_safe_len(
                raw, ts->parse_pos, raw_len, ts->syn->param_end, ts->param_is_string);
        limit = trim_truncated_dsml_close_tail(raw, ts->parse_pos, limit);
        if (!dsml_tool_emit_value(ts, ops, ctx, raw, limit)) return false;
        if (limit > ts->parse_pos) ts->parse_pos = limit;
        if (ts->param_is_string && !ops->args_fragment(ctx, ts, "\"", 1)) return false;
        ts->state = DSML_TOOL_BETWEEN_PARAMS;
    }
    if (ts->args_open && !dsml_tool_close_invoke(ts, ops, ctx)) return false;
    ts->active = false;
    ts->state = DSML_TOOL_DONE;
    return true;
}



bool dsml_tool_stream_update(dsml_tool_stream *ts, const dsml_tool_stream_ops *ops, void *ctx,
                             const char *raw, size_t raw_len) {
    const pulsar_dsml_syntax *syn = ts->syn;
    while (ts->active && ts->parse_pos < raw_len) {
        if (ts->state == DSML_TOOL_BETWEEN_INVOKES) {
            while (ts->parse_pos < raw_len && isspace((unsigned char)raw[ts->parse_pos])) ts->parse_pos++;
            if (ts->parse_pos >= raw_len) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, syn->tool_calls_end)) {
                ts->parse_pos += strlen(syn->tool_calls_end);
                ts->active = false;
                ts->state = DSML_TOOL_DONE;
                return true;
            }
            if (raw_partial_any(raw, raw_len, ts->parse_pos, syn->tool_calls_end, syn->invoke_start)) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, syn->invoke_start)) {
                size_t before_pos = ts->parse_pos;
                dsml_tool_stream_state before_state = ts->state;
                if (!dsml_tool_start_invoke(ts, ops, ctx, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            return dsml_tool_stream_fail(ts);
        }

        if (ts->state == DSML_TOOL_BETWEEN_PARAMS) {
            while (ts->parse_pos < raw_len && isspace((unsigned char)raw[ts->parse_pos])) ts->parse_pos++;
            if (ts->parse_pos >= raw_len) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, syn->invoke_end)) {
                if (!dsml_tool_close_invoke(ts, ops, ctx)) return false;
                ts->parse_pos += strlen(syn->invoke_end);
                ts->state = DSML_TOOL_BETWEEN_INVOKES;
                continue;
            }
            if (raw_partial_any(raw, raw_len, ts->parse_pos, syn->invoke_end, syn->param_start)) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, syn->param_start)) {
                size_t before_pos = ts->parse_pos;
                dsml_tool_stream_state before_state = ts->state;
                if (!dsml_tool_start_param(ts, ops, ctx, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            return dsml_tool_stream_fail(ts);
        }

        if (ts->state == DSML_TOOL_PARAM_VALUE) {
            const char *end = find_lit_bounded(raw + ts->parse_pos, raw_len - ts->parse_pos,
                                               syn->param_end);
            if (end) {
                if (!dsml_tool_finish_param(ts, ops, ctx, raw, (size_t)(end - raw))) return false;
                continue;
            }
            size_t limit = tool_param_value_stream_safe_len(raw, ts->parse_pos, raw_len,
                                                            syn->param_end, ts->param_is_string);
            if (!dsml_tool_emit_value(ts, ops, ctx, raw, limit)) return false;
            if (limit > ts->parse_pos) ts->parse_pos = limit;
            return true;
        }

        return true;
    }
    return true;
}

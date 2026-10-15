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

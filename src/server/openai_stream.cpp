#include "pulsar_server_internal.h"



bool http_response(int fd, int code, const char *type, const char *body) {
    const char *reason = code == 200 ? "OK" :
                         code == 204 ? "No Content" :
                         code == 400 ? "Bad Request" :
                         code == 404 ? "Not Found" :
                         code == 409 ? "Conflict" :
                         code == 500 ? "Internal Server Error" : "Error";
    const size_t body_len = body ? strlen(body) : 0;
    buf h = {0};
    buf_printf(&h,
        "HTTP/1.1 %d %s\r\n"
        "Content-Length: %zu\r\n",
        code, reason, body_len);
    if (type && type[0]) {
        buf_puts(&h, "Content-Type: ");
        buf_puts(&h, type);
        buf_puts(&h, "\r\n");
    }
    buf_puts(&h, "Connection: close\r\n\r\n");
    bool ok = send_all(fd, h.ptr, h.len);
    if (ok && body_len) ok = send_all(fd, body, body_len);
    buf_free(&h);
    return ok;
}



bool http_error(int fd, int code, const char *msg) {
    buf b = {0};
    buf_puts(&b, "{\"error\":{\"message\":");
    json_escape(&b, msg);
    buf_puts(&b, ",\"type\":\"invalid_request_error\"}}\n");
    bool ok = http_response(fd, code, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

/* Anthropic error envelope: the Messages API wraps the error object in a
 * top-level {"type":"error", ...}, and the Anthropic SDK keys on that
 * discriminator -- an OpenAI-shaped {"error":{...}} surfaces as a generic
 * unknown APIError instead of a typed InvalidRequestError. The
 * context-length path already emits this shape; every other 4xx on the
 * /v1/messages surface must too. */
bool http_error_anthropic(int fd, int code, const char *msg) {
    buf b = {0};
    buf_puts(&b, "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":");
    json_escape(&b, msg);
    buf_puts(&b, "}}\n");
    bool ok = http_response(fd, code, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}



static const char *context_length_error_param(const request *r) {
    if (!r) return "prompt";
    if (r->api == API_RESPONSES) return "input";
    return r->kind == REQ_COMPLETION ? "prompt" : "messages";
}



bool request_exceeds_context(const request *r, int ctx_size) {
    /* pulsar_session_sync() rejects prompt->len >= ctx_size because generation
     * needs at least one free context slot.  Catch the same boundary here so
     * clients get a normal protocol error instead of a later backend failure. */
    return r && r->prompt.len >= ctx_size;
}



bool http_error_context_length_exceeded(int fd,
                                               const request *r,
                                               int n_prompt_tokens,
                                               int ctx_size) {
    buf b = {0};
    char msg[160];
    snprintf(msg, sizeof(msg),
             "Prompt has %d tokens, but the configured context size is %d tokens",
             n_prompt_tokens, ctx_size);

    if (r && r->api == API_ANTHROPIC) {
        buf_puts(&b, "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":");
        json_escape(&b, msg);
        buf_puts(&b, ",\"n_prompt_tokens\":");
        buf_printf(&b, "%d", n_prompt_tokens);
        buf_puts(&b, ",\"n_ctx\":");
        buf_printf(&b, "%d", ctx_size);
        buf_puts(&b, "}}\n");
    } else {
        buf_puts(&b, "{\"error\":{\"message\":");
        json_escape(&b, msg);
        buf_puts(&b, ",\"type\":\"invalid_request_error\",\"param\":");
        json_escape(&b, context_length_error_param(r));
        buf_puts(&b, ",\"code\":\"context_length_exceeded\",\"n_prompt_tokens\":");
        buf_printf(&b, "%d", n_prompt_tokens);
        buf_puts(&b, ",\"n_ctx\":");
        buf_printf(&b, "%d", ctx_size);
        buf_puts(&b, "}}\n");
    }
    bool ok = http_response(fd, 400, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}



/* Streaming is a translation state machine over the raw DS4 text.  The model
 * may produce <think> and DSML tool blocks; clients should receive those as
 * protocol-native reasoning/tool deltas, never as visible assistant text. */
bool sse_headers(int fd) {
    buf h = {0};
    buf_puts(&h,
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n");
    buf_puts(&h, "Connection: close\r\n\r\n");
    bool ok = send_all(fd, h.ptr, h.len);
    buf_free(&h);
    return ok;
}



bool sse_error_event(int fd, const request *r, const char *msg) {
    const char *message = msg && msg[0] ? msg : "internal server error";
    buf b = {0};
    if (r && r->api == API_ANTHROPIC) {
        buf_puts(&b, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":");
        json_escape(&b, message);
        buf_puts(&b, "}}\n\n");
    } else {
        buf_puts(&b, "event: error\ndata: {\"error\":{\"message\":");
        json_escape(&b, message);
        buf_puts(&b, ",\"type\":\"server_error\"}}\n\n");
    }
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}



bool sse_chunk(int fd, const request *r, const char *id, const char *text, const char *finish) {
    buf b = {0};
    long now = (long)time(NULL);
    if (r->kind == REQ_CHAT) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":");
        if (text) {
            buf_puts(&b, "{\"content\":");
            json_escape(&b, text);
            buf_putc(&b, '}');
        } else {
            buf_puts(&b, finish ? "{}" : "{\"role\":\"assistant\"}");
        }
        buf_puts(&b, ",\"finish_reason\":");
        if (finish) json_escape(&b, finish); else buf_puts(&b, "null");
        buf_puts(&b, "}]}\n\n");
    } else {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"text\":");
        json_escape(&b, text ? text : "");
        buf_puts(&b, ",\"index\":0,\"finish_reason\":");
        if (finish) json_escape(&b, finish); else buf_puts(&b, "null");
        buf_puts(&b, "}]}\n\n");
    }
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}



int clamp_usage_tokens(int value, int max) {
    if (value < 0) return 0;
    if (max >= 0 && value > max) return max;
    return value;
}

/* The cache read/write split reported to clients: read is capped at the
 * prompt size, write at what read leaves. Single-sourced so all three
 * protocol usage emitters (OpenAI/Anthropic/Responses) apply the same policy
 * -- the order (read before write) is a contract they must not drift on. */
void resolve_cache_split(int *cache_read, int *cache_write, int total) {
    *cache_read = clamp_usage_tokens(*cache_read, total);
    *cache_write = clamp_usage_tokens(*cache_write, total - *cache_read);
}



void append_openai_usage_json(buf *b, const request *r,
                                     int prompt_tokens, int completion_tokens) {
    int cached_tokens = r ? r->cache_read_tokens : 0;
    int cache_write_tokens = r ? r->cache_write_tokens : 0;
    resolve_cache_split(&cached_tokens, &cache_write_tokens, prompt_tokens);
    /* OpenAI defines cached_tokens as prompt tokens retrieved from cache.
     * Newly-prefilled tokens are useful to expose, but they are a DS4 extension
     * and must stay separate so OpenAI-compatible clients do not over-count
     * cache hits. */
    buf_printf(b,
               "{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,"
               "\"prompt_tokens_details\":{\"cached_tokens\":%d,\"cache_write_tokens\":%d}}",
               prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
               cached_tokens, cache_write_tokens);
}



/* A token piece can be a PARTIAL UTF-8 sequence — the tokenizer splits
 * multibyte characters across tokens — while the wire must stay valid UTF-8.
 * "bytes" carries the exact bytes and is the authoritative form; "token"
 * substitutes U+FFFD for anything that is not a well-formed sequence, which is
 * what the OpenAI API does with the same problem. */
static void append_logprob_text_json(buf *b, const char *piece, size_t len) {
    buf clean = {0};
    size_t i = 0;
    while (i < len) {
        const unsigned char c = (unsigned char)piece[i];
        size_t need = c < 0x80                 ? 1
                    : (c >= 0xc2 && c <= 0xdf) ? 2
                    : (c >= 0xe0 && c <= 0xef) ? 3
                    : (c >= 0xf0 && c <= 0xf4) ? 4
                                               : 0;
        bool ok = need > 0 && len - i >= need;
        for (size_t k = 1; ok && k < need; k++)
            ok = ((unsigned char)piece[i + k] & 0xc0) == 0x80;
        /* 10xxxxxx continuations alone still admit overlongs, UTF-16
         * surrogates and > U+10FFFF; the second byte carries the extra
         * constraint (Unicode 15 Table 3-7).  Without this, ED A0 80 walks
         * through as "well-formed" and a strict client rejects the JSON. */
        if (ok && need >= 3) {
            const unsigned char c1 = (unsigned char)piece[i + 1];
            ok = c == 0xe0 ? c1 >= 0xa0
               : c == 0xed ? c1 <= 0x9f
               : c == 0xf0 ? c1 >= 0x90
               : c == 0xf4 ? c1 <= 0x8f
                           : true;
        }
        if (ok) {
            buf_append(&clean, piece + i, need);
            i += need;
        } else {
            buf_puts(&clean, "\xef\xbf\xbd");
            i++;
        }
    }
    buf_putc(b, '"');
    json_escape_fragment_n(b, clean.ptr ? clean.ptr : "", clean.len);
    buf_putc(b, '"');
    buf_free(&clean);
}



/* JSON has no -Infinity.  A degenerate row (every logit non-finite) is the only
 * way to produce one, and `null` is how the engine's --dump-logits reports the
 * same condition — an unmistakable hole beats a fabricated finite number. */
static void append_logprob_number_json(buf *b, float v) {
    if (isfinite(v)) buf_printf(b, "%.9g", (double)v);
    else buf_puts(b, "null");
}



/* {"token":...,"logprob":...,"bytes":[...]  — the whole of an alternative's
 * entry and the head of a chosen token's, whose "top_logprobs" array follows.
 * Left OPEN for that reason; both callers close the object. */
static void append_logprob_token_open_json(buf *b, const logprob_token *t) {
    buf_puts(b, "{\"token\":");
    append_logprob_text_json(b, t->piece, t->piece_len);
    buf_puts(b, ",\"logprob\":");
    append_logprob_number_json(b, t->logprob);
    buf_puts(b, ",\"bytes\":[");
    for (size_t k = 0; k < t->piece_len; k++) {
        if (k) buf_putc(b, ',');
        buf_printf(b, "%u", (unsigned)(unsigned char)t->piece[k]);
    }
    buf_putc(b, ']');
}



/* The OpenAI logprobs object for entries [from, to) of a request's ledger,
 * emitted with a leading comma so callers append it inside a choice object
 * (append_openai_timings_json's convention).  Nothing is written unless the
 * client asked for logprobs.
 *
 * The array is named `content` because that is the field name in the OpenAI
 * schema; see logprob_ledger for what it actually spans (every generated token,
 * not only the visible content substring). */
void append_openai_logprobs_json(buf *b, const logprob_ledger *lg, int from, int to) {
    if (!lg || !lg->enabled) return;
    if (from < 0) from = 0;
    if (to > lg->len) to = lg->len;
    buf_puts(b, ",\"logprobs\":{\"content\":[");
    for (int i = from; i < to; i++) {
        const logprob_entry *e = &lg->v[i];
        if (i > from) buf_putc(b, ',');
        append_logprob_token_open_json(b, &e->tok);
        buf_puts(b, ",\"top_logprobs\":[");
        for (int t = 0; t < e->n_top; t++) {
            if (t) buf_putc(b, ',');
            append_logprob_token_open_json(b, &e->top[t]);
            buf_putc(b, '}');
        }
        buf_puts(b, "]}");
    }
    buf_puts(b, "]}");
}



/* Streaming form: attach the entries whose token bytes this chunk RELEASES
 * (piece ending at or before the watermark; SIZE_MAX on the final chunk) and
 * mark them streamed, so the entries concatenated over a stream reproduce the
 * non-streaming array exactly once. */
static void append_openai_logprobs_delta(buf *b, logprob_ledger *lp, size_t release_upto) {
    if (!lp || !lp->enabled) return;
    const int to = logprob_stream_ready(lp, release_upto);
    append_openai_logprobs_json(b, lp, lp->streamed, to);
    lp->streamed = to;
}



/* Additive per-response timing block (llama.cpp-ish field names for client
 * familiarity, plus DS4 cache-split and DSpark extensions). Emits a leading
 * ",\"timings\":{...}" so callers append it directly after the usage object.
 * Pure metadata: every value comes from counters the worker already kept, and
 * the rates are derived here with guarded divisions (a zero denominator omits
 * the rate rather than emitting a non-finite number). Never affects sampling. */
void append_openai_timings_json(buf *b, const request *r) {
    const req_timings *t = r ? &r->timings : NULL;
    if (!t || !t->valid) return;

    int prefill_computed = t->prompt_n - t->cached_n;
    if (prefill_computed < 0) prefill_computed = 0;

    buf_puts(b, ",\"timings\":{");
    buf_printf(b, "\"ttft_s\":%.6f", t->ttft_s);
    buf_printf(b, ",\"prompt_n\":%d,\"cached_n\":%d,\"prefill_n\":%d",
               t->prompt_n, t->cached_n, prefill_computed);
    /* Prefill throughput is over the tokens actually computed (cache hits cost
     * ~no kernel work); omit when nothing was computed or the interval is empty. */
    if (prefill_computed > 0 && t->prefill_s > 0.0) {
        buf_printf(b, ",\"prefill_per_second\":%.2f",
                   (double)prefill_computed / t->prefill_s);
    }
    buf_printf(b, ",\"predicted_n\":%d", t->decode_n);
    if (t->decode_n > 0 && t->decode_s > 0.0) {
        buf_printf(b, ",\"predicted_per_second\":%.2f",
                   (double)t->decode_n / t->decode_s);
    }
    /* DSpark speculative decode: accept-rate alpha and mean committed tokens per
     * verify step, both scoped to THIS request (per-session counter deltas). */
    if (t->spec_active && t->spec_draft > 0) {
        buf_printf(b, ",\"spec_accept_rate\":%.4f",
                   (double)t->spec_accepted / (double)t->spec_draft);
    }
    if (t->spec_active && t->spec_drafts > 0) {
        buf_printf(b, ",\"spec_tokens_per_step\":%.4f",
                   (double)t->spec_gen / (double)t->spec_drafts);
    }
    buf_putc(b, '}');
}



static bool sse_usage_chunk(int fd, const request *r, const char *id,
                            int prompt_tokens, int completion_tokens) {
    if (!r->stream_include_usage) return true;

    buf b = {0};
    long now = (long)time(NULL);
    if (r->kind == REQ_CHAT) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[],\"usage\":");
    } else {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[],\"usage\":");
    }
    append_openai_usage_json(&b, r, prompt_tokens, completion_tokens);
    append_openai_timings_json(&b, r);
    buf_puts(&b, "}\n\n");

    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}



bool sse_done(int fd, const request *r, const char *id,
                     int prompt_tokens, int completion_tokens) {
    return sse_usage_chunk(fd, r, id, prompt_tokens, completion_tokens) &&
           send_all(fd, "data: [DONE]\n\n", 14);
}



bool sse_chat_finish(int fd, const request *r, const char *id, const char *content,
                            const char *reasoning, const tool_calls *calls, const char *finish,
                            int prompt_tokens, int completion_tokens) {
    if (!sse_chunk(fd, r, id, NULL, NULL)) return false;

    buf b = {0};
    long now = (long)time(NULL);
    if (reasoning && reasoning[0]) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":");
        json_escape(&b, reasoning);
        buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    }
    if (content && content[0]) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"content\":");
        json_escape(&b, content);
        buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    }
    if (calls && calls->len) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":");
        append_tool_call_deltas_json(&b, calls, id, &r->tool_orders);
        buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    }
    buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":");
    json_escape(&b, finish);
    buf_puts(&b, "}]}\n\n");

    bool ok = send_all(fd, b.ptr, b.len) &&
              sse_done(fd, r, id, prompt_tokens, completion_tokens);
    buf_free(&b);
    return ok;
}



void openai_stream_start(const request *r, openai_stream *st) {
    memset(st, 0, sizeof(*st));
    st->active = true;
    st->mode = pulsar_think_mode_enabled(r->think_mode) ? OPENAI_STREAM_THINKING : OPENAI_STREAM_TEXT;
    st->guard_second_reasoning =
        pulsar_think_mode_enabled(r->think_mode) && r->has_tools;
}



static void openai_tool_stream_free(openai_tool_stream *ts) {
    if (!ts) return;
    for (int i = 0; i < ts->ids_cap; i++) free(ts->ids[i]);
    free(ts->ids);
    ts->ids = NULL;
    ts->ids_cap = 0;
}



void openai_stream_free(openai_stream *st) {
    if (!st) return;
    openai_tool_stream_free(&st->tool);
}



static bool openai_tool_stream_has_id(const openai_tool_stream *ts,
                                      const char *id, int upto) {
    if (!ts || !id || !id[0]) return false;
    if (upto > ts->ids_cap) upto = ts->ids_cap;
    for (int i = 0; i < upto; i++) {
        if (ts->ids[i] && !strcmp(ts->ids[i], id)) return true;
    }
    return false;
}



/* Free function (NOT a server:: method): legitimately called with a null server.
 * As a member the `!s` guard would be elided under -O3 (this assumed non-null);
 * as a free function taking server* the guard holds. */
static const char *openai_tool_stream_id(server *s, openai_tool_stream *ts,
                                         int index) {
    if (!ts || index < 0) return "";
    if (index >= ts->ids_cap) {
        int old = ts->ids_cap;
        int cap = old ? old : 4;
        while (cap <= index) cap *= 2;
        ts->ids = (char* *)server_xrealloc(ts->ids, (size_t)cap * sizeof(ts->ids[0]));
        memset(ts->ids + old, 0, (size_t)(cap - old) * sizeof(ts->ids[0]));
        ts->ids_cap = cap;
    }
    if (!ts->ids[index]) {
        char id[64];
        for (;;) {
            random_tool_id(id, sizeof(id), API_OPENAI);
            if (!openai_tool_stream_has_id(ts, id, index) &&
                (!s || !s->tool_memory_has_id(id))) break;  /* null s (no bound server) => no dedup, as the predecessor free fn did */
        }
        ts->ids[index] = xstrdup(id);
    }
    return ts->ids[index];
}



size_t text_stream_safe_limit(const char *raw, size_t start,
                                     size_t raw_len, bool has_tools,
                                     bool final);



/* `release_upto` is the absolute offset in the raw completion that this delta
 * carries the client up to; the logprob entries it covers ride along. */
static bool sse_chat_delta_n(int fd, const request *r, const char *id,
                             const char *field, const char *text, size_t len,
                             logprob_ledger *lp, size_t release_upto) {
    if (len == 0) return true;
    buf b = {0};
    long now = (long)time(NULL);
    buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{");
    json_escape(&b, field);
    buf_putc(&b, ':');
    json_escape_n(&b, text, len);
    buf_putc(&b, '}');
    append_openai_logprobs_delta(&b, lp, release_upto);
    buf_puts(&b, ",\"finish_reason\":null}]}\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}



/* OpenAI clients can consume function.arguments as a stream of JSON text
 * fragments.  DS4 generates XML-ish DSML instead, so this parser switches to a
 * hidden tool mode at <...tool_calls>, emits the tool header once the invoke tag
 * is complete, then translates each parameter body into argument deltas while
 * holding only tiny tails for partial closing tags, UTF-8, and DSML entities. */
static bool sse_chat_tool_call_start_delta(int fd, const request *r, const char *id,
                                           int index, const char *tool_id,
                                           const char *name) {
    buf b = {0};
    long now = (long)time(NULL);
    buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":");
    buf_printf(&b, "%d", index);
    buf_puts(&b, ",\"id\":");
    json_escape(&b, tool_id ? tool_id : "");
    buf_puts(&b, ",\"type\":\"function\",\"function\":{\"name\":");
    json_escape(&b, name ? name : "");
    buf_puts(&b, ",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}



static bool sse_chat_tool_call_args_delta_n(int fd, const request *r, const char *id,
                                            int index, const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    long now = (long)time(NULL);
    buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":");
    buf_printf(&b, "%d", index);
    buf_puts(&b, ",\"function\":{\"arguments\":");
    json_escape_n(&b, text, len);
    buf_puts(&b, "}}]},\"finish_reason\":null}]}\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}



bool raw_full_lit(const char *raw, size_t raw_len, size_t pos, const char *lit) {
    size_t n = strlen(lit);
    return pos <= raw_len && raw_len - pos >= n && !memcmp(raw + pos, lit, n);
}



static bool raw_partial_lit(const char *raw, size_t raw_len, size_t pos, const char *lit) {
    size_t n = strlen(lit);
    if (pos > raw_len || raw_len - pos >= n) return false;
    return !memcmp(raw + pos, lit, raw_len - pos);
}



bool raw_partial_any(const char *raw, size_t raw_len, size_t pos,
                            const char *a, const char *b) {
    return raw_partial_lit(raw, raw_len, pos, a) || raw_partial_lit(raw, raw_len, pos, b);
}



const char *find_lit_bounded(const char *s, size_t n, const char *lit) {
    size_t m = strlen(lit);
    if (m == 0) return s;
    if (n < m) return NULL;
    for (size_t i = 0; i <= n - m; i++) {
        if (!memcmp(s + i, lit, m)) return s + i;
    }
    return NULL;
}



const dsml_syntax dsml_syntaxes[3] = {
    {
        PULSAR_TOOL_CALLS_START, PULSAR_TOOL_CALLS_END,
        PULSAR_INVOKE_START, PULSAR_INVOKE_END,
        PULSAR_PARAM_START, PULSAR_PARAM_END,
    },
    {
        PULSAR_TOOL_CALLS_START_SHORT, PULSAR_TOOL_CALLS_END_SHORT,
        PULSAR_INVOKE_START_SHORT, PULSAR_INVOKE_END_SHORT,
        PULSAR_PARAM_START_SHORT, PULSAR_PARAM_END_SHORT,
    },
    {
        "<tool_calls>", "</tool_calls>",
        "<invoke", "</invoke>",
        "<parameter", "</parameter>",
    },
};



static bool raw_partial_lit_min(const char *raw, size_t raw_len, size_t pos,
                                const char *lit, size_t min_len) {
    size_t lit_len = strlen(lit);
    if (!raw || pos > raw_len || raw_len - pos >= lit_len) return false;
    size_t avail = raw_len - pos;
    return avail >= min_len && !memcmp(raw + pos, lit, avail);
}



static size_t dsml_max_tool_start_len(void) {
    size_t max = 0;
    for (size_t i = 0; i < sizeof(dsml_syntaxes) / sizeof(dsml_syntaxes[0]); i++) {
        size_t n = strlen(dsml_syntaxes[i].tool_calls_start);
        if (n > max) max = n;
    }
    return max;
}



static bool dsml_find_tool_start(const char *raw, size_t raw_len,
                                 size_t *pos_out,
                                 const dsml_syntax **syn_out) {
    const char *best = NULL;
    const dsml_syntax *best_syn = NULL;
    for (size_t i = 0; i < sizeof(dsml_syntaxes) / sizeof(dsml_syntaxes[0]); i++) {
        const char *p = find_lit_bounded(raw, raw_len, dsml_syntaxes[i].tool_calls_start);
        if (p && (!best || p < best)) {
            best = p;
            best_syn = &dsml_syntaxes[i];
        }
    }
    if (!best) return false;
    *pos_out = (size_t)(best - raw) + strlen(best_syn->tool_calls_start);
    *syn_out = best_syn;
    return true;
}



static bool dsml_find_tool_start_from(const char *raw, size_t raw_len,
                                      size_t start,
                                      size_t *pos_out,
                                      const dsml_syntax **syn_out) {
    if (start > raw_len) return false;
    size_t rel = 0;
    if (!dsml_find_tool_start(raw + start, raw_len - start, &rel, syn_out)) {
        return false;
    }
    *pos_out = start + rel;
    return true;
}



static bool dsml_attr_is_string_true(const char *raw, size_t raw_len,
                                     size_t tag_start, size_t tag_end) {
    if (tag_end <= tag_start || tag_end > raw_len) return false;
    char *tag = xstrndup(raw + tag_start, tag_end - tag_start);
    char *is_string = dsml_attr(tag, "string");
    bool result = is_string && !strcmp(is_string, "true");
    free(is_string);
    free(tag);
    return result;
}



#ifdef PULSAR_SERVER_TEST

static bool raw_suffix_partial_lit(const char *raw, size_t raw_len,
                                   const char *lit, size_t min_len) {
    size_t lit_len = strlen(lit);
    if (!raw || raw_len == 0 || lit_len == 0) return false;
    size_t max = raw_len < lit_len ? raw_len : lit_len - 1;
    for (size_t n = min_len; n <= max; n++) {
        if (!memcmp(raw + raw_len - n, lit, n)) return true;
    }
    return false;
}



static dsml_decode_state dsml_decode_scan_json_param(const char *raw,
                                                     size_t raw_len,
                                                     size_t pos,
                                                     const dsml_syntax *syn) {
    bool in_string = false;
    bool escaped = false;
    while (pos < raw_len) {
        if (!in_string && raw_full_lit(raw, raw_len, pos, syn->param_end)) {
            return DSML_DECODE_STRUCTURAL;
        }
        unsigned char c = (unsigned char)raw[pos++];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
        } else if (c == '"') {
            in_string = true;
        }
    }
    if (!in_string && raw_suffix_partial_lit(raw, raw_len, syn->param_end, 2)) {
        return DSML_DECODE_STRUCTURAL;
    }
    return in_string ? DSML_DECODE_JSON_STRING : DSML_DECODE_JSON_STRUCTURAL;
}



/* Slow reference recognizer used by tests. */
dsml_decode_state dsml_decode_state_for_text(const char *raw, size_t raw_len) {
    if (!raw || raw_len == 0) return DSML_DECODE_OUTSIDE;

    size_t pos = 0;
    const dsml_syntax *syn = NULL;
    if (!dsml_find_tool_start(raw, raw_len, &pos, &syn)) {
        return DSML_DECODE_OUTSIDE;
    }

    for (;;) {
        while (pos < raw_len && isspace((unsigned char)raw[pos])) pos++;
        if (pos >= raw_len) return DSML_DECODE_STRUCTURAL;

        if (raw_full_lit(raw, raw_len, pos, syn->tool_calls_end)) {
            return DSML_DECODE_OUTSIDE;
        }
        if (raw_full_lit(raw, raw_len, pos, syn->invoke_end)) {
            pos += strlen(syn->invoke_end);
            continue;
        }
        if (raw_full_lit(raw, raw_len, pos, syn->invoke_start)) {
            const char *tag_end = (const char *)memchr(raw + pos, '>', raw_len - pos);
            if (!tag_end) return DSML_DECODE_STRUCTURAL;
            pos = (size_t)(tag_end - raw) + 1;
            continue;
        }
        if (raw_full_lit(raw, raw_len, pos, syn->param_start)) {
            size_t tag_start = pos;
            const char *tag_end_ptr = (const char *)memchr(raw + pos, '>', raw_len - pos);
            if (!tag_end_ptr) return DSML_DECODE_STRUCTURAL;
            size_t tag_end = (size_t)(tag_end_ptr - raw) + 1;
            bool string_value = dsml_attr_is_string_true(raw, raw_len, tag_start, tag_end);
            pos = tag_end;

            if (string_value) {
                const char *end = find_lit_bounded(raw + pos, raw_len - pos, syn->param_end);
                if (!end) {
                    if (raw_suffix_partial_lit(raw, raw_len, syn->param_end, 2)) {
                        return DSML_DECODE_STRUCTURAL;
                    }
                    return DSML_DECODE_STRING_BODY;
                }
                pos = (size_t)(end - raw) + strlen(syn->param_end);
                continue;
            }

            dsml_decode_state json_state =
                dsml_decode_scan_json_param(raw, raw_len, pos, syn);
            if (json_state == DSML_DECODE_STRUCTURAL) {
                const char *end = find_lit_bounded(raw + pos, raw_len - pos, syn->param_end);
                if (!end) return DSML_DECODE_STRUCTURAL;
                pos = (size_t)(end - raw) + strlen(syn->param_end);
                continue;
            }
            return json_state;
        }

        for (size_t i = 0; i < sizeof(dsml_syntaxes) / sizeof(dsml_syntaxes[0]); i++) {
            if (raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].tool_calls_end) ||
                raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].invoke_start) ||
                raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].invoke_end) ||
                raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].param_start) ||
                raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].param_end))
            {
                return DSML_DECODE_STRUCTURAL;
            }
        }
        return DSML_DECODE_STRUCTURAL;
    }
}


#endif


bool dsml_decode_state_is_tool(dsml_decode_state state) {
    return state != DSML_DECODE_OUTSIDE;
}



bool dsml_decode_state_uses_payload_sampling(dsml_decode_state state) {
    return state == DSML_DECODE_STRING_BODY || state == DSML_DECODE_JSON_STRING;
}



void dsml_decode_tracker_init(dsml_decode_tracker *dt) {
    memset(dt, 0, sizeof(*dt));
    dt->mode = DSML_TRACK_SEARCH;
    dt->decode = DSML_DECODE_OUTSIDE;
}



/* Track where generation is inside a DSML tool call.  This is intentionally a
 * forgiving recognizer, not a validator: malformed DSML still gets parsed later
 * by the normal tool-call parser.  Here we only need enough state to decide
 * whether the next token belongs to protocol syntax or arbitrary payload. */
void dsml_decode_tracker_update(dsml_decode_tracker *dt,
                                       const char *raw, size_t raw_len) {
    if (!dt || !raw) return;

    for (;;) {
        if (dt->mode == DSML_TRACK_DONE) {
            dt->decode = DSML_DECODE_OUTSIDE;
            return;
        }

        if (dt->mode == DSML_TRACK_SEARCH) {
            size_t pos = 0;
            const dsml_syntax *syn = NULL;
            if (!dsml_find_tool_start_from(raw, raw_len, dt->pos, &pos, &syn)) {
                size_t hold = dsml_max_tool_start_len();
                dt->pos = raw_len > hold ? raw_len - hold : 0;
                dt->decode = DSML_DECODE_OUTSIDE;
                return;
            }
            dt->syn = syn;
            dt->pos = pos;
            dt->mode = DSML_TRACK_STRUCTURAL;
            dt->decode = DSML_DECODE_STRUCTURAL;
        }

        if (dt->mode == DSML_TRACK_STRING_BODY) {
            while (dt->pos < raw_len) {
                if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->param_end)) {
                    dt->pos += strlen(dt->syn->param_end);
                    dt->mode = DSML_TRACK_STRUCTURAL;
                    dt->decode = DSML_DECODE_STRUCTURAL;
                    goto structural;
                }
                if (raw_partial_lit_min(raw, raw_len, dt->pos, dt->syn->param_end, 2)) {
                    dt->decode = DSML_DECODE_STRUCTURAL;
                    return;
                }
                dt->pos++;
            }
            dt->decode = DSML_DECODE_STRING_BODY;
            return;
        }

        if (dt->mode == DSML_TRACK_JSON_PARAM) {
            while (dt->pos < raw_len) {
                if (!dt->json_in_string) {
                    if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->param_end)) {
                        dt->pos += strlen(dt->syn->param_end);
                        dt->mode = DSML_TRACK_STRUCTURAL;
                        dt->decode = DSML_DECODE_STRUCTURAL;
                        goto structural;
                    }
                    if (raw_partial_lit_min(raw, raw_len, dt->pos, dt->syn->param_end, 2)) {
                        dt->decode = DSML_DECODE_STRUCTURAL;
                        return;
                    }
                }

                unsigned char c = (unsigned char)raw[dt->pos++];
                if (dt->json_in_string) {
                    if (dt->json_escaped) {
                        dt->json_escaped = false;
                    } else if (c == '\\') {
                        dt->json_escaped = true;
                    } else if (c == '"') {
                        dt->json_in_string = false;
                    }
                } else if (c == '"') {
                    dt->json_in_string = true;
                }
            }
            dt->decode = dt->json_in_string ?
                DSML_DECODE_JSON_STRING : DSML_DECODE_JSON_STRUCTURAL;
            return;
        }

structural:
        while (dt->mode == DSML_TRACK_STRUCTURAL) {
            while (dt->pos < raw_len && isspace((unsigned char)raw[dt->pos])) dt->pos++;
            if (dt->pos >= raw_len) {
                dt->decode = DSML_DECODE_STRUCTURAL;
                return;
            }

            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->tool_calls_end)) {
                dt->mode = DSML_TRACK_DONE;
                dt->pos += strlen(dt->syn->tool_calls_end);
                dt->decode = DSML_DECODE_OUTSIDE;
                return;
            }
            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->invoke_end)) {
                dt->pos += strlen(dt->syn->invoke_end);
                continue;
            }
            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->invoke_start)) {
                const char *tag_end = (const char *)memchr(raw + dt->pos, '>', raw_len - dt->pos);
                if (!tag_end) {
                    dt->decode = DSML_DECODE_STRUCTURAL;
                    return;
                }
                dt->pos = (size_t)(tag_end - raw) + 1;
                continue;
            }
            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->param_start)) {
                size_t tag_start = dt->pos;
                const char *tag_end = (const char *)memchr(raw + dt->pos, '>', raw_len - dt->pos);
                if (!tag_end) {
                    dt->decode = DSML_DECODE_STRUCTURAL;
                    return;
                }
                size_t tag_after = (size_t)(tag_end - raw) + 1;
                bool string_value = dsml_attr_is_string_true(raw, raw_len, tag_start, tag_after);
                dt->pos = tag_after;
                if (string_value) {
                    dt->mode = DSML_TRACK_STRING_BODY;
                    dt->decode = DSML_DECODE_STRING_BODY;
                } else {
                    dt->mode = DSML_TRACK_JSON_PARAM;
                    dt->json_in_string = false;
                    dt->json_escaped = false;
                    dt->decode = DSML_DECODE_JSON_STRUCTURAL;
                }
                break;
            }

            if (raw_partial_lit(raw, raw_len, dt->pos, dt->syn->tool_calls_end) ||
                raw_partial_lit(raw, raw_len, dt->pos, dt->syn->invoke_start) ||
                raw_partial_lit(raw, raw_len, dt->pos, dt->syn->invoke_end) ||
                raw_partial_lit(raw, raw_len, dt->pos, dt->syn->param_start) ||
                raw_partial_lit(raw, raw_len, dt->pos, dt->syn->param_end))
            {
                dt->decode = DSML_DECODE_STRUCTURAL;
                return;
            }

            dt->decode = DSML_DECODE_STRUCTURAL;
            return;
        }
    }
}



static size_t dsml_entity_stream_safe_len(const char *raw, size_t start, size_t limit) {
    static const char *ents[] = {"&amp;", "&lt;", "&gt;", "&quot;", "&apos;"};
    const size_t max_ent = 6;
    size_t scan = limit > start + max_ent ? limit - max_ent : start;
    for (size_t i = limit; i > scan; i--) {
        if (raw[i - 1] != '&') continue;
        size_t amp = i - 1;
        size_t tail = limit - amp;
        for (size_t ei = 0; ei < sizeof(ents) / sizeof(ents[0]); ei++) {
            size_t elen = strlen(ents[ei]);
            if (tail < elen && !memcmp(raw + amp, ents[ei], tail)) return amp;
        }
        break;
    }
    return limit;
}



size_t tool_param_value_stream_safe_len(const char *raw, size_t start,
                                               size_t raw_len, const char *param_end,
                                               bool is_string) {
    /* Hold a trailing partial closing tag of ANY DSML style, not only the
     * active one: the model sometimes mixes styles (opens short, closes
     * long), and a mismatched partial close streamed as value bytes is tag
     * debris in the argument (seen live: a truncated call carrying
     * "</｜DSML｜" inside command).  A held tail self-resolves on the next
     * update: either the prefix breaks (real value text, released) or the
     * tag completes (parsed or trimmed at finalize). */
    (void)param_end;
    size_t limit = trim_truncated_dsml_close_tail(raw, start, raw_len);
    if (is_string) limit = dsml_entity_stream_safe_len(raw, start, limit);
    return utf8_stream_safe_len(raw, start, limit, false);
}



static bool openai_tool_emit_args_fragment(int fd, const request *r, const char *id,
                                           openai_tool_stream *ts,
                                           const char *text, size_t len) {
    return sse_chat_tool_call_args_delta_n(fd, r, id, ts->index, text, len);
}



static bool openai_tool_emit_string_value(int fd, const request *r, const char *id,
                                          openai_tool_stream *ts,
                                          const char *text, size_t len) {
    if (len == 0) return true;
    char *raw = xstrndup(text, len);
    char *unescaped = dsml_unescape_text(raw);
    buf frag = {0};
    json_escape_fragment_n(&frag, unescaped, strlen(unescaped));
    bool ok = openai_tool_emit_args_fragment(fd, r, id, ts, frag.ptr ? frag.ptr : "", frag.len);
    buf_free(&frag);
    free(unescaped);
    free(raw);
    return ok;
}



static bool openai_tool_emit_param_prefix(int fd, const request *r, const char *id,
                                          openai_tool_stream *ts,
                                          const char *name, bool is_string) {
    buf frag = {0};
    if (ts->first_param) ts->first_param = false;
    else buf_putc(&frag, ',');
    json_escape(&frag, name ? name : "");
    buf_putc(&frag, ':');
    if (is_string) buf_putc(&frag, '"');
    bool ok = openai_tool_emit_args_fragment(fd, r, id, ts, frag.ptr ? frag.ptr : "", frag.len);
    buf_free(&frag);
    return ok;
}



static bool openai_tool_stream_init(openai_tool_stream *ts, const char *raw,
                                    size_t raw_len, size_t pos) {
    openai_tool_stream_free(ts);
    memset(ts, 0, sizeof(*ts));
    ts->active = true;
    ts->state = DSML_TOOL_BETWEEN_INVOKES;
    ts->parse_pos = pos;
    if (raw_full_lit(raw, raw_len, pos, PULSAR_TOOL_CALLS_START)) {
        ts->parse_pos += strlen(PULSAR_TOOL_CALLS_START);
        ts->tool_calls_end = PULSAR_TOOL_CALLS_END;
        ts->invoke_start = PULSAR_INVOKE_START;
        ts->invoke_end = PULSAR_INVOKE_END;
        ts->param_start = PULSAR_PARAM_START;
        ts->param_end = PULSAR_PARAM_END;
    } else if (raw_full_lit(raw, raw_len, pos, PULSAR_TOOL_CALLS_START_SHORT)) {
        ts->parse_pos += strlen(PULSAR_TOOL_CALLS_START_SHORT);
        ts->tool_calls_end = PULSAR_TOOL_CALLS_END_SHORT;
        ts->invoke_start = PULSAR_INVOKE_START_SHORT;
        ts->invoke_end = PULSAR_INVOKE_END_SHORT;
        ts->param_start = PULSAR_PARAM_START_SHORT;
        ts->param_end = PULSAR_PARAM_END_SHORT;
    } else if (raw_full_lit(raw, raw_len, pos, "<tool_calls>")) {
        ts->parse_pos += strlen("<tool_calls>");
        ts->tool_calls_end = "</tool_calls>";
        ts->invoke_start = "<invoke";
        ts->invoke_end = "</invoke>";
        ts->param_start = "<parameter";
        ts->param_end = "</parameter>";
    } else {
        ts->active = false;
        ts->state = DSML_TOOL_ERROR;
        return false;
    }
    return true;
}



static bool openai_tool_stream_fail(openai_tool_stream *ts) {
    ts->active = false;
    ts->state = DSML_TOOL_ERROR;
    return true;
}



static bool openai_tool_start_invoke(int fd, server *s, const request *r, const char *id,
                                     openai_tool_stream *ts,
                                     const char *raw, size_t raw_len) {
    const char *tag_end = (const char *)memchr(raw + ts->parse_pos, '>', raw_len - ts->parse_pos);
    if (!tag_end) return true;
    char *tag = xstrndup(raw + ts->parse_pos, (size_t)(tag_end - (raw + ts->parse_pos) + 1));
    char *name = dsml_attr(tag, "name");
    free(tag);
    if (!name) return openai_tool_stream_fail(ts);

    const char *tool_id = openai_tool_stream_id(s, ts, ts->index);
    bool ok = sse_chat_tool_call_start_delta(fd, r, id, ts->index, tool_id, name) &&
              openai_tool_emit_args_fragment(fd, r, id, ts, "{", 1);
    free(name);
    if (!ok) return false;

    ts->emitted_any = true;
    ts->args_open = true;
    ts->first_param = true;
    ts->parse_pos = (size_t)(tag_end - raw) + 1;
    ts->state = DSML_TOOL_BETWEEN_PARAMS;
    return true;
}



static bool openai_tool_start_param(int fd, const request *r, const char *id,
                                    openai_tool_stream *ts,
                                    const char *raw, size_t raw_len) {
    const char *tag_end = (const char *)memchr(raw + ts->parse_pos, '>', raw_len - ts->parse_pos);
    if (!tag_end) return true;
    char *tag = xstrndup(raw + ts->parse_pos, (size_t)(tag_end - (raw + ts->parse_pos) + 1));
    char *name = dsml_attr(tag, "name");
    char *is_string = dsml_attr(tag, "string");
    free(tag);
    if (!name || !is_string) {
        free(name);
        free(is_string);
        return openai_tool_stream_fail(ts);
    }
    bool string_value = !strcmp(is_string, "true");
    bool ok = openai_tool_emit_param_prefix(fd, r, id, ts, name, string_value);
    free(name);
    free(is_string);
    if (!ok) return false;

    ts->param_is_string = string_value;
    ts->parse_pos = (size_t)(tag_end - raw) + 1;
    ts->state = DSML_TOOL_PARAM_VALUE;
    return true;
}



static bool openai_tool_finish_param(int fd, const request *r, const char *id,
                                     openai_tool_stream *ts,
                                     const char *raw, size_t value_end) {
    if (value_end > ts->parse_pos) {
        bool ok = ts->param_is_string ?
            openai_tool_emit_string_value(fd, r, id, ts, raw + ts->parse_pos,
                                          value_end - ts->parse_pos) :
            openai_tool_emit_args_fragment(fd, r, id, ts, raw + ts->parse_pos,
                                           value_end - ts->parse_pos);
        if (!ok) return false;
    }
    if (ts->param_is_string &&
        !openai_tool_emit_args_fragment(fd, r, id, ts, "\"", 1)) return false;
    ts->parse_pos = value_end + strlen(ts->param_end);
    ts->state = DSML_TOOL_BETWEEN_PARAMS;
    return true;
}



/* Generation ended (final) while a streamed tool call is still open: the
 * client has already received the call header and a prefix of the argument
 * deltas, so the truncation cannot be reclassified as text (the non-stream
 * path's try_repair_dsml equivalent).  What CAN be guaranteed is well-formed
 * wire JSON: flush the un-emitted value bytes, close an open string value,
 * close the args object.  The argument VALUE stays truncated — exactly what
 * the repaired non-stream parse of the same bytes yields — and the finish
 * reason (length) still tells the client the turn was cut. */
static bool openai_tool_stream_finalize(int fd, const request *r, const char *id,
                                        openai_tool_stream *ts,
                                        const char *raw, size_t raw_len) {
    if (!ts->active) return true;
    if (ts->state == DSML_TOOL_PARAM_VALUE) {
        /* Flush only up to the stream-safe limit: the held-back tail is a
         * partial closing tag (the model was mid-"</...parameter>") or a
         * split UTF-8 sequence -- tag debris and mojibake, never value
         * content.  Dropping it beats the non-stream repair here, which
         * keeps the fragment in the value. */
        size_t limit = tool_param_value_stream_safe_len(
                raw, ts->parse_pos, raw_len, ts->param_end, ts->param_is_string);
        limit = trim_truncated_dsml_close_tail(raw, ts->parse_pos, limit);
        if (limit > ts->parse_pos) {
            bool ok = ts->param_is_string ?
                openai_tool_emit_string_value(fd, r, id, ts, raw + ts->parse_pos,
                                              limit - ts->parse_pos) :
                openai_tool_emit_args_fragment(fd, r, id, ts, raw + ts->parse_pos,
                                               limit - ts->parse_pos);
            if (!ok) return false;
            ts->parse_pos = limit;
        }
        if (ts->param_is_string &&
            !openai_tool_emit_args_fragment(fd, r, id, ts, "\"", 1)) return false;
        ts->state = DSML_TOOL_BETWEEN_PARAMS;
    }
    if (ts->args_open) {
        if (!openai_tool_emit_args_fragment(fd, r, id, ts, "}", 1)) return false;
        ts->args_open = false;
    }
    ts->active = false;
    ts->state = DSML_TOOL_DONE;
    return true;
}

static bool openai_tool_stream_update(int fd, server *s, const request *r, const char *id,
                                      openai_tool_stream *ts,
                                      const char *raw, size_t raw_len) {
    while (ts->active && ts->parse_pos < raw_len) {
        if (ts->state == DSML_TOOL_BETWEEN_INVOKES) {
            while (ts->parse_pos < raw_len && isspace((unsigned char)raw[ts->parse_pos])) ts->parse_pos++;
            if (ts->parse_pos >= raw_len) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->tool_calls_end)) {
                ts->parse_pos += strlen(ts->tool_calls_end);
                ts->active = false;
                ts->state = DSML_TOOL_DONE;
                return true;
            }
            if (raw_partial_any(raw, raw_len, ts->parse_pos, ts->tool_calls_end, ts->invoke_start)) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->invoke_start)) {
                size_t before_pos = ts->parse_pos;
                dsml_tool_stream_state before_state = ts->state;
                if (!openai_tool_start_invoke(fd, s, r, id, ts, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            return openai_tool_stream_fail(ts);
        }

        if (ts->state == DSML_TOOL_BETWEEN_PARAMS) {
            while (ts->parse_pos < raw_len && isspace((unsigned char)raw[ts->parse_pos])) ts->parse_pos++;
            if (ts->parse_pos >= raw_len) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->invoke_end)) {
                if (ts->args_open &&
                    !openai_tool_emit_args_fragment(fd, r, id, ts, "}", 1)) return false;
                ts->args_open = false;
                ts->parse_pos += strlen(ts->invoke_end);
                ts->index++;
                ts->state = DSML_TOOL_BETWEEN_INVOKES;
                continue;
            }
            if (raw_partial_any(raw, raw_len, ts->parse_pos, ts->invoke_end, ts->param_start)) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->param_start)) {
                size_t before_pos = ts->parse_pos;
                dsml_tool_stream_state before_state = ts->state;
                if (!openai_tool_start_param(fd, r, id, ts, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            return openai_tool_stream_fail(ts);
        }

        if (ts->state == DSML_TOOL_PARAM_VALUE) {
            const char *end = find_lit_bounded(raw + ts->parse_pos,
                                               raw_len - ts->parse_pos,
                                               ts->param_end);
            if (end) {
                if (!openai_tool_finish_param(fd, r, id, ts, raw,
                                              (size_t)(end - raw))) return false;
                continue;
            }
            size_t limit = tool_param_value_stream_safe_len(raw, ts->parse_pos,
                                                            raw_len, ts->param_end,
                                                            ts->param_is_string);
            if (limit > ts->parse_pos) {
                bool ok = ts->param_is_string ?
                    openai_tool_emit_string_value(fd, r, id, ts, raw + ts->parse_pos,
                                                  limit - ts->parse_pos) :
                    openai_tool_emit_args_fragment(fd, r, id, ts, raw + ts->parse_pos,
                                                   limit - ts->parse_pos);
                if (!ok) return false;
                ts->parse_pos = limit;
            }
            return true;
        }

        return true;
    }
    return true;
}



bool openai_sse_stream_update(int fd, server *s, const request *r, const char *id,
                                     openai_stream *st,
                                     const char *raw, size_t raw_len,
                                     bool final) {
    if (!st->active || !raw) return true;

    if (st->mode == OPENAI_STREAM_THINKING) {
        if (!st->checked_think_prefix) {
            const char *open = "<think>";
            const size_t open_len = strlen(open);
            if (raw_len < open_len && !strncmp(raw, open, raw_len) && !final) {
                return true;
            }
            if (raw_len >= open_len && !strncmp(raw, open, open_len)) {
                st->emit_pos = open_len;
            }
            st->checked_think_prefix = true;
        }

        const char *close = strstr(raw + st->emit_pos, "</think>");
        /* A tool call starting before any </think> is the unclosed-reasoning
         * recovery case (upstream ds4 51a1c14): stream only the prose before
         * the marker as reasoning, hold until the block completes, and keep
         * the protocol bytes off every channel. */
        const char *tool = r->has_tools ?
            find_any_tool_start(raw + st->emit_pos) : NULL;
        const bool tool_before_close = tool && (!close || tool < close);
        /* The END must also land before </think>: a block whose end falls past
         * the close straddles the reasoning boundary and is NOT an executable
         * call (upstream ds4 0ead8a8). */
        const char *tool_end = tool_before_close ? find_any_tool_end(tool) : NULL;
        const bool complete_tool = tool_end && (!close || tool_end < close);
        size_t limit;
        if (complete_tool) {
            limit = trim_tool_separator_ws(raw, st->emit_pos,
                                           (size_t)(tool - raw));
        } else if (close) {
            /* An incomplete marker that remains inside a closed think block is
             * reasoning text, not an executable call. */
            limit = (size_t)(close - raw);
        } else if (final) {
            /* Match non-stream parsing: flush incomplete DSML as reasoning. */
            limit = raw_len;
        } else if (tool_before_close) {
            limit = trim_tool_separator_ws(raw, st->emit_pos,
                                           (size_t)(tool - raw));
        } else {
            const size_t hold = strlen("</think>") - 1;
            limit = raw_len > hold ? raw_len - hold : st->emit_pos;
            limit = utf8_stream_safe_len(raw, st->emit_pos, limit, false);
        }

        if (limit > st->emit_pos) {
            if (!sse_chat_delta_n(fd, r, id, "reasoning_content",
                                  raw + st->emit_pos,
                                  limit - st->emit_pos, st->lp, limit)) return false;
            st->sent_reasoning = true;
            st->emit_pos = limit;
        }

        if (complete_tool) {
            st->emit_pos = (size_t)(tool - raw);
            st->mode = OPENAI_STREAM_SUPPRESS;
            return true;
        }

        if (close) {
            st->emit_pos = (size_t)(close - raw) + strlen("</think>");
            st->mode = OPENAI_STREAM_TEXT;
        } else if (final) {
            st->mode = OPENAI_STREAM_SUPPRESS;
            return true;
        } else {
            return true;
        }
    }

    if (st->mode == OPENAI_STREAM_TEXT) {
        if (st->guard_second_reasoning) {
            /* A second </think> before any tool marker means the held text
             * was another reasoning pass — reroute it. A tool marker or the
             * final flush releases the hold as genuine answer text. */
            const char *close = strstr(raw + st->emit_pos, "</think>");
            const char *tool2 = r->has_tools ?
                find_any_tool_start(raw + st->emit_pos) : NULL;
            if (close && (!tool2 || close < tool2)) {
                const size_t limit = (size_t)(close - raw);
                if (limit > st->emit_pos &&
                    !sse_chat_delta_n(fd, r, id, "reasoning_content",
                                      raw + st->emit_pos,
                                      limit - st->emit_pos, st->lp, limit)) return false;
                if (limit > st->emit_pos) st->sent_reasoning = true;
                st->emit_pos = limit + strlen("</think>");
                st->guard_second_reasoning = false;
            } else if (!tool2 && !final) {
                return true;
            } else {
                st->guard_second_reasoning = false;
            }
        }

        const char *tool = r->has_tools ? find_any_tool_start(raw + st->emit_pos) : NULL;
        size_t limit = text_stream_safe_limit(raw, st->emit_pos, raw_len,
                                              r->has_tools, final);

        if (limit > st->emit_pos) {
            if (!sse_chat_delta_n(fd, r, id, "content",
                                  raw + st->emit_pos,
                                  limit - st->emit_pos, st->lp, limit)) return false;
            st->emit_pos = limit;
        }

        if (tool) {
            st->emit_pos = (size_t)(tool - raw);
            if (openai_tool_stream_init(&st->tool, raw, raw_len, st->emit_pos)) {
                st->mode = OPENAI_STREAM_TOOL;
            } else {
                st->mode = OPENAI_STREAM_SUPPRESS;
            }
        } else if (final) {
            st->mode = OPENAI_STREAM_SUPPRESS;
        }
    }

    if (st->mode == OPENAI_STREAM_TOOL) {
        if (!openai_tool_stream_update(fd, s, r, id, &st->tool, raw, raw_len)) return false;
        if (final && st->tool.active &&
            !openai_tool_stream_finalize(fd, r, id, &st->tool, raw, raw_len)) return false;
        if (!st->tool.active) st->mode = OPENAI_STREAM_SUPPRESS;
    }
    return true;
}



bool openai_sse_finish_live(int fd, server *s, const request *r, const char *id,
                                   openai_stream *st, const char *raw,
                                   size_t raw_len, const tool_calls *calls,
                                   const char *finish, int prompt_tokens,
                                   int completion_tokens) {
    if (!openai_sse_stream_update(fd, s, r, id, st, raw, raw_len, true)) return false;

    buf b = {0};
    long now = (long)time(NULL);
    if (calls && calls->len && !st->tool.emitted_any) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":");
        append_tool_call_deltas_json(&b, calls, id, &r->tool_orders);
        buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    }
    buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{}");
    /* Entries whose bytes never reached a content/reasoning delta (tool-call
     * bytes, suppressed protocol markers, a held tail) are still this
     * completion's tokens: flush the remainder here so the streamed logprobs
     * concatenate to the non-streaming array. */
    append_openai_logprobs_delta(&b, st->lp, SIZE_MAX);
    buf_puts(&b, ",\"finish_reason\":");
    json_escape(&b, finish);
    buf_puts(&b, "}]}\n\n");

    bool ok = send_all(fd, b.ptr, b.len) &&
              sse_done(fd, r, id, prompt_tokens, completion_tokens);
    buf_free(&b);
    return ok;
}



bool request_uses_openai_live_stream(const request *r) {
    return r->stream && r->api == API_OPENAI && r->kind == REQ_CHAT;
}



bool request_uses_responses_live_stream(const request *r) {
    return r->stream && r->api == API_RESPONSES && r->kind == REQ_CHAT;
}



bool request_uses_structured_stream(const request *r) {
    return r->stream && (r->api == API_ANTHROPIC ||
                         r->api == API_RESPONSES ||
                         request_uses_openai_live_stream(r));
}


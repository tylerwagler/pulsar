#include "pulsar_server_internal.h"



/* Writes one complete response.  `extra` is NULL or pre-formatted
 * "Name: value\r\n" lines, placed before Connection. */
static bool http_write(int fd, int code, const char *type, const char *body,
                       const char *extra) {
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
    if (extra) buf_puts(&h, extra);
    buf_puts(&h, "Connection: close\r\n\r\n");
    bool ok = send_all(fd, h.ptr, h.len);
    if (ok && body_len) ok = send_all(fd, body, body_len);
    buf_free(&h);
    return ok;
}

bool http_response(int fd, int code, const char *type, const char *body) {
    return http_write(fd, code, type, body, NULL);
}

/* A retryable failure tells the client when to come back: Retry-After in
 * whole seconds (never an HTTP date; Claude Code reads integers and stops
 * retrying above 60) and X-Should-Retry: true.  Permanent failures carry
 * neither, so a client never backs off to retry something that cannot work. */
bool http_response_retry(int fd, int code, const char *type, const char *body,
                         int retry_after_s) {
    char extra[96];
    snprintf(extra, sizeof extra, "Retry-After: %d\r\nX-Should-Retry: true\r\n",
             retry_after_s);
    return http_write(fd, code, type, body, extra);
}

static void openai_error_body(buf *b, const char *msg) {
    buf_puts(b, "{\"error\":{\"message\":");
    json_escape(b, msg);
    buf_puts(b, ",\"type\":\"invalid_request_error\"}}\n");
}

bool http_error(int fd, int code, const char *msg) {
    buf b = {0};
    openai_error_body(&b, msg);
    bool ok = http_response(fd, code, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

bool http_error_retry(int fd, int code, const char *msg, int retry_after_s) {
    buf b = {0};
    openai_error_body(&b, msg);
    bool ok = http_response_retry(fd, code, "application/json", b.ptr, retry_after_s);
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
        /* Table 3-7 well-formedness is the lib's one rule (utf8_seq_ok,
         * L187): ED A0 80 and the other overlong / surrogate / > U+10FFFF
         * forms are rejected there, so a strict client never sees them. */
        const int need = utf8_seq_ok((const unsigned char *)piece + i, len - i);
        if (need > 0) {
            buf_append(&clean, piece + i, (size_t)need);
            i += (size_t)need;
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






void openai_stream_start(const request *r, openai_stream *st) {
    (void)r;
    memset(st, 0, sizeof(*st));
    st->active = true;
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



static bool raw_partial_lit_min(const char *raw, size_t raw_len, size_t pos,
                                const char *lit, size_t min_len) {
    size_t lit_len = strlen(lit);
    if (!raw || pos > raw_len || raw_len - pos >= lit_len) return false;
    size_t avail = raw_len - pos;
    return avail >= min_len && !memcmp(raw + pos, lit, avail);
}



static size_t dsml_max_tool_start_len(void) {
    size_t max = 0;
    for (size_t i = 0; i < PULSAR_DSML_SYNTAXES; i++) {
        size_t n = strlen(pulsar_dsml_syntaxes[i].tool_calls_start);
        if (n > max) max = n;
    }
    return max;
}



static bool dsml_find_tool_start(const char *raw, size_t raw_len,
                                 size_t *pos_out,
                                 const pulsar_dsml_syntax **syn_out) {
    const char *best = NULL;
    const pulsar_dsml_syntax *best_syn = NULL;
    for (size_t i = 0; i < PULSAR_DSML_SYNTAXES; i++) {
        const char *p = find_lit_bounded(raw, raw_len, pulsar_dsml_syntaxes[i].tool_calls_start);
        if (p && (!best || p < best)) {
            best = p;
            best_syn = &pulsar_dsml_syntaxes[i];
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
                                      const pulsar_dsml_syntax **syn_out) {
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
    char *is_string = pulsar_dsml_attr(tag, "string");
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
                                                     const pulsar_dsml_syntax *syn) {
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
    const pulsar_dsml_syntax *syn = NULL;
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

        for (size_t i = 0; i < PULSAR_DSML_SYNTAXES; i++) {
            if (raw_partial_lit(raw, raw_len, pos, pulsar_dsml_syntaxes[i].tool_calls_end) ||
                raw_partial_lit(raw, raw_len, pos, pulsar_dsml_syntaxes[i].invoke_start) ||
                raw_partial_lit(raw, raw_len, pos, pulsar_dsml_syntaxes[i].invoke_end) ||
                raw_partial_lit(raw, raw_len, pos, pulsar_dsml_syntaxes[i].param_start) ||
                raw_partial_lit(raw, raw_len, pos, pulsar_dsml_syntaxes[i].param_end))
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
            const pulsar_dsml_syntax *syn = NULL;
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



/* The OpenAI side of the shared DSML tool-stream projection (genmsg.cpp
 * dsml_tool_stream_update): tool_call deltas keyed by the invocation index;
 * nothing to send when an invocation closes. */
/* ---- L267: the OpenAI sink (chat_sink) --------------------------------------- */

static bool openai_sink_text_cb(chat_sink *k, bool reasoning, const char *text, size_t len, size_t release_upto) {
    openai_stream *st = (openai_stream *)k->st;
    if (!st->active) return true;
    if (!sse_chat_delta_n(k->fd, k->r, k->id, reasoning ? "reasoning_content" : "content", text, len, st->lp,
                          release_upto)) return false;
    if (reasoning) st->sent_reasoning = true;
    return true;
}

static bool openai_sink_end_cb(chat_sink *, bool) { return true; }

/* A DSML invocation as it decodes: the tool_call start delta (id + name), then
 * its argument object's JSON in fragments. */
static bool openai_tool_begin_invoke(void *vctx, dsml_tool_stream *ts, const char *name) {
    chat_sink *k = (chat_sink *)vctx;
    const char *tool_id = dsml_tool_stream_id(k->s, ts, ts->index, API_OPENAI);
    ((openai_stream *)k->st)->tools_streamed++;
    return sse_chat_tool_call_start_delta(k->fd, k->r, k->id, ts->index, tool_id, name);
}

static bool openai_tool_args_fragment(void *vctx, dsml_tool_stream *ts, const char *text, size_t len) {
    chat_sink *k = (chat_sink *)vctx;
    return sse_chat_tool_call_args_delta_n(k->fd, k->r, k->id, ts->index, text, len);
}

static bool openai_tool_end_invoke(void *, dsml_tool_stream *) { return true; }

static const dsml_tool_stream_ops openai_tool_ops = {
    openai_tool_begin_invoke, openai_tool_args_fragment, openai_tool_end_invoke,
};

void openai_sink_init(chat_sink *k, int fd, server *s, const request *r, const char *id, openai_stream *st) {
    *k = {fd, s, r, id, st, openai_sink_text_cb, openai_sink_end_cb, &openai_tool_ops, true};
}

/* A call handed over whole (Qwen): the start delta (index, id, name, empty
 * arguments) and one arguments delta carrying the whole JSON object -- the
 * OpenAI streaming shape, so a client that concatenates argument fragments gets
 * the object. */
bool openai_sink_tool_call(chat_sink *k, int index, const tool_call *tc) {
    const char *args = tc->arguments ? tc->arguments : "";
    ((openai_stream *)k->st)->tools_streamed++;
    return sse_chat_tool_call_start_delta(k->fd, k->r, k->id, index, tc->id, tc->name) &&
           sse_chat_tool_call_args_delta_n(k->fd, k->r, k->id, index, args, strlen(args));
}



/* Append the closing chat chunk (empty delta, the remaining logprob entries,
 * finish_reason) to `b`, send `b`, then the usage chunk and [DONE]; frees `b`. */
static bool openai_sse_send_final(int fd, const request *r, const char *id, long now, buf *b,
                                  logprob_ledger *lp, const char *finish,
                                  int prompt_tokens, int completion_tokens) {
    buf_printf(b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(b, r->model);
    buf_puts(b, ",\"choices\":[{\"index\":0,\"delta\":{}");
    /* Entries whose bytes never reached a content/reasoning delta (tool-call
     * bytes, suppressed protocol markers, a held tail) are still this
     * completion's tokens: flush the remainder here so the streamed logprobs
     * concatenate to the non-streaming array. */
    append_openai_logprobs_delta(b, lp, SIZE_MAX);
    buf_puts(b, ",\"finish_reason\":");
    json_escape(b, finish);
    buf_puts(b, "}]}\n\n");

    bool ok = send_all(fd, b->ptr, b->len) &&
              sse_done(fd, r, id, prompt_tokens, completion_tokens);
    buf_free(b);
    return ok;
}



/* The end of the stream: the calls no delta carried yet (a block the final
 * parse recovered), then the closing chunk, usage and [DONE]. */
bool openai_sse_finish(chat_sink *k, const tool_calls *calls, const char *finish, int prompt_tokens,
                       int completion_tokens) {
    const openai_stream *st = (const openai_stream *)k->st;
    buf b = {0};
    long now = (long)time(NULL);
    if (calls && calls->len && !st->tools_streamed) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", k->id, now);
        json_escape(&b, k->r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":");
        append_tool_call_deltas_json(&b, calls, k->id, &k->r->tool_orders);
        buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    }
    return openai_sse_send_final(k->fd, k->r, k->id, now, &b, st->lp, finish, prompt_tokens, completion_tokens);
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


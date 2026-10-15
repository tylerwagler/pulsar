/* deepseek_dsml.cpp -- DeepSeek's DSML machinery (L295 F5: moved out of the shared server files, byte for byte).
 *
 * The model samples "<think>reasoning</think>answer" with DSML tool-call blocks.  Everything that READS that text
 * lives here: the decode-time tracker (the greedy region of a call, parser_deepseek.cpp's in_tool_call), the
 * live tool-stream projection into a protocol sink (deepseek_stream.cpp's walk), the text hold before a tool
 * marker, the final parse, the stream's call ids onto the parsed calls, the call-block finder tool memory keys
 * on (server_family_ops::find_call_block) and the model-visible tool error (server_family_ops::tool_error_suffix).
 * Until L295 these bodies sat in openai_stream.cpp, genmsg.cpp, anthropic_stream.cpp, kv_cache.cpp,
 * tool_memory.cpp, server_jobs.cpp and generate.cpp. */
#include "deepseek_dsml.h"

/* ---- the raw-text helpers and the decode-time tracker ------------------------------------------------ */

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



/* ---- the final parse: markers, recovery and the DSML parameters -------------------------------------- */

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
            free(calls->raw_calls);
            calls->raw_calls = xstrndup(raw_block_start, (size_t)(raw_block_end - raw_block_start));
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



/* ---- a completed tool block inside unclosed reasoning ------------------------------------------------ */

/* A completed tool block inside unclosed reasoning can be recovered without
 * predicting what the model will emit after an injected close marker
 * (upstream ds4 51a1c14; replaces the forced-</think> injection recovery,
 * which sometimes made the model read the call as already issued and end the
 * turn without it). Keep a short overlap until the opening appears, then wait
 * for its matching end — a lone "<" or partial marker keeps decoding
 * untouched. */
bool complete_tool_call_inside_thinking(const char *text, size_t len,
                                        size_t *scan_from) {
    if (!text || !scan_from) return false;
    if (*scan_from > len) *scan_from = len;
    const char *start = find_any_tool_start(text + *scan_from);
    if (!start) {
        const size_t hold = 80; /* > longest stanza opening */
        *scan_from = len > hold ? len - hold : 0;
        return false;
    }
    *scan_from = (size_t)(start - text);
    return find_any_tool_end(start) != NULL;
}



/* ---- the text a stream may release before a tool marker ---------------------------------------------- */

size_t text_stream_safe_limit(const char *raw, size_t start,
                                     size_t raw_len, bool has_tools,
                                     bool final) {
    if (raw_len <= start) return raw_len;

    size_t limit = raw_len;
    if (has_tools) {
        const char *tool = find_any_tool_start(raw + start);
        if (tool) {
            limit = trim_tool_separator_ws(raw, start, (size_t)(tool - raw));
            return utf8_stream_safe_len(raw, start, limit, true);
        }

        if (!final) {
            /* Tool calls are hidden from the API client and returned as
             * structured tool_use/tool_calls blocks.  The whitespace just
             * before the DSML marker is syntax too: if we stream it as
             * assistant text, the next client request sends it back and our
             * renderer adds the canonical "\n\n" separator again.  Hold
             * trailing whitespace until a following non-whitespace byte proves
             * it is ordinary text, or until a tool marker proves it should be
             * dropped. */
            while (limit > start && isspace((unsigned char)raw[limit - 1])) limit--;

            /* Also hold a partial '<...tool_calls...' marker that may be split
             * across generated tokens. */
            const size_t max_marker = 80;
            size_t scan = raw_len - start > max_marker ? raw_len - max_marker : start;
            for (size_t i = raw_len; i > scan; i--) {
                if (raw[i - 1] == '<') {
                    size_t marker = i - 1;
                    if (marker < limit) limit = marker;
                    break;
                }
            }
            limit = trim_tool_separator_ws(raw, start, limit);
        }
    }
    return utf8_stream_safe_len(raw, start, limit, final);
}



/* ---- the live tool-stream projection ----------------------------------------------------------------- */

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



static bool dsml_tool_close_invoke(dsml_tool_stream *ts, chat_sink *k);

/* The block is not a call (the finish reads it as text, L284 P4): the projection stops, and a call it
 * already announced is closed on the stream as sent -- the rule every family's stream keeps. */
static bool dsml_tool_stream_fail(dsml_tool_stream *ts, chat_sink *k) {
    const bool ok = !ts->args_open || dsml_tool_close_invoke(ts, k);
    ts->active = false;
    ts->state = DSML_TOOL_ERROR;
    return ok;
}



/* A string parameter's value bytes: DSML entities undone, then JSON-string
 * escaped, as a fragment inside the already-open quotes. */
static bool dsml_tool_emit_string_value(dsml_tool_stream *ts, chat_sink *k, const char *text, size_t len) {
    if (len == 0) return true;
    char *raw = xstrndup(text, len);
    char *unescaped = pulsar_dsml_unescape(raw);
    buf frag = {0};
    json_escape_fragment_n(&frag, unescaped, strlen(unescaped));
    bool ok = k->tool_ops->args(k, ts->index, frag.ptr ? frag.ptr : "", frag.len);
    buf_free(&frag);
    free(unescaped);
    free(raw);
    return ok;
}



static bool dsml_tool_emit_param_prefix(dsml_tool_stream *ts, chat_sink *k, const char *name, bool is_string) {
    buf frag = {0};
    if (ts->first_param) ts->first_param = false;
    else buf_putc(&frag, ',');
    json_escape(&frag, name ? name : "");
    buf_putc(&frag, ':');
    if (is_string) buf_putc(&frag, '"');
    bool ok = k->tool_ops->args(k, ts->index, frag.ptr ? frag.ptr : "", frag.len);
    buf_free(&frag);
    return ok;
}



static bool dsml_tool_start_invoke(dsml_tool_stream *ts, chat_sink *k, const char *raw, size_t raw_len) {
    const char *tag_end = (const char *)memchr(raw + ts->parse_pos, '>', raw_len - ts->parse_pos);
    if (!tag_end) return true;
    char *tag = xstrndup(raw + ts->parse_pos, (size_t)(tag_end - (raw + ts->parse_pos) + 1));
    char *name = pulsar_dsml_attr(tag, "name");
    free(tag);
    if (!name) return dsml_tool_stream_fail(ts, k);
    if (!tool_call_declared(k->r, name, NULL, 0)) {
        /* L272: a call to an undeclared tool is a malformed block -- the live projection stops before
         * announcing it, and the finish drops it (or retries) */
        free(name);
        return dsml_tool_stream_fail(ts, k);
    }

    /* the id the client sees, fixed per index once created (apply_stream_tool_ids copies it into the
     * parsed call at the finish) */
    const char *id = dsml_tool_stream_id(k->s, ts, ts->index, k->r->api);
    bool ok = k->tool_ops->begin(k, ts->index, id, name) &&
              k->tool_ops->args(k, ts->index, "{", 1);
    free(name);
    if (!ok) return false;

    ts->emitted_any = true;
    ts->args_open = true;
    ts->first_param = true;
    ts->parse_pos = (size_t)(tag_end - raw) + 1;
    ts->state = DSML_TOOL_BETWEEN_PARAMS;
    return true;
}



static bool dsml_tool_start_param(dsml_tool_stream *ts, chat_sink *k, const char *raw, size_t raw_len) {
    const char *tag_end = (const char *)memchr(raw + ts->parse_pos, '>', raw_len - ts->parse_pos);
    if (!tag_end) return true;
    char *tag = xstrndup(raw + ts->parse_pos, (size_t)(tag_end - (raw + ts->parse_pos) + 1));
    char *name = pulsar_dsml_attr(tag, "name");
    char *is_string = pulsar_dsml_attr(tag, "string");
    free(tag);
    if (!name || !is_string) {
        free(name);
        free(is_string);
        return dsml_tool_stream_fail(ts, k);
    }
    bool string_value = !strcmp(is_string, "true");
    bool ok = dsml_tool_emit_param_prefix(ts, k, name, string_value);
    free(name);
    free(is_string);
    if (!ok) return false;

    ts->param_is_string = string_value;
    ts->parse_pos = (size_t)(tag_end - raw) + 1;
    ts->state = DSML_TOOL_PARAM_VALUE;
    return true;
}



static bool dsml_tool_emit_value(dsml_tool_stream *ts, chat_sink *k, const char *raw, size_t value_end) {
    if (value_end <= ts->parse_pos) return true;
    return ts->param_is_string ?
        dsml_tool_emit_string_value(ts, k, raw + ts->parse_pos, value_end - ts->parse_pos) :
        k->tool_ops->args(k, ts->index, raw + ts->parse_pos, value_end - ts->parse_pos);
}



static bool dsml_tool_finish_param(dsml_tool_stream *ts, chat_sink *k, const char *raw, size_t value_end) {
    if (!dsml_tool_emit_value(ts, k, raw, value_end)) return false;
    if (ts->param_is_string && !k->tool_ops->args(k, ts->index, "\"", 1)) return false;
    ts->parse_pos = value_end + strlen(ts->syn->param_end);
    ts->state = DSML_TOOL_BETWEEN_PARAMS;
    return true;
}



/* The invocation's "}" and the protocol's block stop, then the next index. */
static bool dsml_tool_close_invoke(dsml_tool_stream *ts, chat_sink *k) {
    if (ts->args_open && !k->tool_ops->args(k, ts->index, "}", 1)) return false;
    ts->args_open = false;
    if (!k->tool_ops->end(k, ts->index)) return false;
    ts->index++;
    return true;
}



/* Generation ended (final) while a streamed tool call is still open: the
 * client has already received the call header and a prefix of the argument
 * deltas, which a stream cannot take back.  The rule every family's stream
 * keeps (L284 P4): an announced call is closed AS SENT -- flush the
 * un-emitted value bytes, close an open string value, close the args object,
 * stop the block -- so the wire JSON is well-formed; the finish reason
 * (length) says the turn was cut, and the final reading returns no call
 * (the turn is text: no tag repair). */
bool dsml_tool_stream_finalize(dsml_tool_stream *ts, chat_sink *k,
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
        if (!dsml_tool_emit_value(ts, k, raw, limit)) return false;
        if (limit > ts->parse_pos) ts->parse_pos = limit;
        if (ts->param_is_string && !k->tool_ops->args(k, ts->index, "\"", 1)) return false;
        ts->state = DSML_TOOL_BETWEEN_PARAMS;
    }
    if (ts->args_open && !dsml_tool_close_invoke(ts, k)) return false;
    ts->active = false;
    ts->state = DSML_TOOL_DONE;
    return true;
}



bool dsml_tool_stream_update(dsml_tool_stream *ts, chat_sink *k,
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
                if (!dsml_tool_start_invoke(ts, k, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            return dsml_tool_stream_fail(ts, k);
        }

        if (ts->state == DSML_TOOL_BETWEEN_PARAMS) {
            while (ts->parse_pos < raw_len && isspace((unsigned char)raw[ts->parse_pos])) ts->parse_pos++;
            if (ts->parse_pos >= raw_len) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, syn->invoke_end)) {
                if (!dsml_tool_close_invoke(ts, k)) return false;
                ts->parse_pos += strlen(syn->invoke_end);
                ts->state = DSML_TOOL_BETWEEN_INVOKES;
                continue;
            }
            if (raw_partial_any(raw, raw_len, ts->parse_pos, syn->invoke_end, syn->param_start)) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, syn->param_start)) {
                size_t before_pos = ts->parse_pos;
                dsml_tool_stream_state before_state = ts->state;
                if (!dsml_tool_start_param(ts, k, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            return dsml_tool_stream_fail(ts, k);
        }

        if (ts->state == DSML_TOOL_PARAM_VALUE) {
            const char *end = find_lit_bounded(raw + ts->parse_pos, raw_len - ts->parse_pos,
                                               syn->param_end);
            if (end) {
                if (!dsml_tool_finish_param(ts, k, raw, (size_t)(end - raw))) return false;
                continue;
            }
            size_t limit = tool_param_value_stream_safe_len(raw, ts->parse_pos, raw_len,
                                                            syn->param_end, ts->param_is_string);
            if (!dsml_tool_emit_value(ts, k, raw, limit)) return false;
            if (limit > ts->parse_pos) ts->parse_pos = limit;
            return true;
        }

        return true;
    }
    return true;
}



/* ---- the stream's call ids onto the parsed calls ----------------------------------------------------- */

void apply_stream_tool_ids(tool_calls *calls, const dsml_tool_stream *ts) {
    if (!calls || !ts) return;
    /* The SSE stream may have exposed tool ids before final DSML parsing.  The
     * parsed calls must inherit those ids before assign_tool_call_ids() and
     * tool_memory_remember(), otherwise the client returns a tool_result for an
     * id that the continuation fast path does not know. */
    int n = calls->len < ts->ids_cap ? calls->len : ts->ids_cap;
    for (int i = 0; i < n; i++) {
        if (calls->v[i].id && calls->v[i].id[0]) continue;
        if (ts->ids[i] && ts->ids[i][0]) calls->v[i].id = xstrdup(ts->ids[i]);
    }
}



/* ---- the call-block finder (tool memory's key) ------------------------------------------------------- */

const char *find_next_dsml_tool_block(const char *p, const char **end_out) {
    struct block_form {
        const char *start;
        const char *end;
    } forms[] = {
        /* Every spelling either family renders, so the scanner finds the last
         * tool block whatever produced the transcript (the shipped 0731
         * artifacts sample V4's unspaced tags, V4.1 the spaced ones). */
        {"\n\n" PULSAR_TOOL_CALLS_START, PULSAR_TOOL_CALLS_END},
        {PULSAR_TOOL_CALLS_START, PULSAR_TOOL_CALLS_END},
        {"\n\n" PULSAR_TOOL_CALLS_START_SHORT, PULSAR_TOOL_CALLS_END_SHORT},
        {PULSAR_TOOL_CALLS_START_SHORT, PULSAR_TOOL_CALLS_END_SHORT},
        {"\n\n" PULSAR_TOOL_CALLS_START_V4, PULSAR_TOOL_CALLS_END_V4},
        {PULSAR_TOOL_CALLS_START_V4, PULSAR_TOOL_CALLS_END_V4},
        {"\n\n" PULSAR_TOOL_CALLS_START_SHORT_V4, PULSAR_TOOL_CALLS_END_SHORT_V4},
        {PULSAR_TOOL_CALLS_START_SHORT_V4, PULSAR_TOOL_CALLS_END_SHORT_V4},
        {"\n\n<tool_calls>", "</tool_calls>"},
        {"<tool_calls>", "</tool_calls>"},
    };

    const char *best = NULL;
    const char *best_end = NULL;
    for (size_t i = 0; i < sizeof(forms) / sizeof(forms[0]); i++) {
        const char *s = strstr(p, forms[i].start);
        if (!s || (best && s >= best)) continue;
        const char *e = strstr(s, forms[i].end);
        if (!e) continue;
        best = s;
        best_end = e + strlen(forms[i].end);
    }
    if (end_out) *end_out = best_end;
    return best;
}



/* ---- the model-visible tool error -------------------------------------------------------------------- */

static char *rendered_chat_system_region(const char *prompt_text) {
    if (!prompt_text) return xstrdup("");
    const char *p = prompt_text;
    const char *bos = PULSAR_SERVER_RENDER_BOS;
    const size_t bos_len = strlen(bos);
    if (!strncmp(p, bos, bos_len)) p += bos_len;
    /* V4.1 leads a thinking conversation with the System token and the effort
     * line; neither belongs to the client's system region. */
    const size_t sys_len = strlen(PULSAR_RENDER_SYSTEM);
    if (!strncmp(p, PULSAR_RENDER_SYSTEM, sys_len)) p += sys_len;
    p += pulsar_think_effort_prefix_len(p);
    while (*p && isspace((unsigned char)*p)) p++;

    const char *user = strstr(p, PULSAR_RENDER_USER);
    const char *assistant = strstr(p, PULSAR_RENDER_ASSISTANT);
    const char *end = NULL;
    if (user && assistant) end = user < assistant ? user : assistant;
    else end = user ? user : assistant;
    if (!end) end = p + strlen(p);
    while (end > p && isspace((unsigned char)end[-1])) end--;
    return xstrndup(p, (size_t)(end - p));
}



/* A server-side tool result appended to the live session mid-turn: close an
 * open think block, then exactly what the replay path renders for ONE tool
 * message after the assistant's call turn -- EOS, the tool_result, the
 * generation prefix (render_live_tool_tail) -- or next-turn prefix reuse dies. */
static char *build_live_tool_result_suffix_spans(const request *r,
                                                 const thinking_state *thinking,
                                                 const char *result_text,
                                                 chat_text_span **spans_out,
                                                 uint32_t *n_spans_out) {
    if (spans_out) *spans_out = NULL;
    if (n_spans_out) *n_spans_out = 0;
    const pulsar_think_mode mode = r ? r->think_mode : PULSAR_THINK_NONE;
    buf suffix = {0};
    if (pulsar_think_mode_enabled(mode) && thinking && thinking->inside) {
        buf_puts(&suffix, "</think>");
    }
    chat_msgs msgs = {0};
    chat_msg result = {0};
    result.role = xstrdup("tool");
    result.content = xstrdup(result_text ? result_text : "");
    chat_msgs_push(&msgs, result);
    /* The tool body is CLIENT data (L223): the tail marks it, and the ranges ride
     * along so the caller tokenises those bytes as plain text. */
    chat_text_span *tail_spans = NULL;
    uint32_t tail_n = 0;
    char *tail = render_live_tool_tail_spans(&msgs, 0, r && r->has_tools, mode, r ? r->family->v41 : true,
                                             &tail_spans, &tail_n);
    buf_puts_spanned(&suffix, tail, tail_spans, tail_n);
    free(tail);
    free(tail_spans);
    chat_msgs_free(&msgs);
    if (spans_out) {
        *spans_out = suffix.spans;
        *n_spans_out = suffix.n_spans;
        suffix.spans = NULL;
        suffix.n_spans = suffix.cap_spans = 0;
    }
    return buf_take(&suffix);
}



char *build_invalid_dsml_tool_error_suffix_spans(const request *r,
                                                 const thinking_state *thinking,
                                                 const char *detail,
                                                 chat_text_span **spans_out,
                                                 uint32_t *n_spans_out) {
    char *system = rendered_chat_system_region(r ? r->prompt_text : NULL);
    buf tool_error = {0};
    buf_puts(&tool_error, "Tool error: invalid DSML tool call");
    if (detail && detail[0]) {
        buf_puts(&tool_error, ": ");
        buf_puts(&tool_error, detail);
    }
    buf_puts(&tool_error,
             "\nThe previous assistant output was not executed because the DSML syntax was malformed. "
             "Emit a new valid DSML tool call, or answer normally if no tool is needed.");
    if (system && system[0]) {
        buf_puts(&tool_error, "\n\nSystem prompt reminder:\n");
        buf_puts(&tool_error, system);
    }

    char *suffix = build_live_tool_result_suffix_spans(r, thinking,
                                                       tool_error.ptr ? tool_error.ptr : "",
                                                       spans_out, n_spans_out);

    free(system);
    buf_free(&tool_error);
    return suffix;
}



char *build_invalid_dsml_tool_error_suffix(const request *r,
                                                  const thinking_state *thinking,
                                                  const char *detail) {
    return build_invalid_dsml_tool_error_suffix_spans(r, thinking, detail, NULL, NULL);
}

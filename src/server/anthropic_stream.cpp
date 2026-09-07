#include "pulsar_server_internal.h"



static const char *anthropic_stop_reason(const char *finish) {
    if (finish && !strcmp(finish, "tool_calls")) return "tool_use";
    if (finish && !strcmp(finish, "length")) return "max_tokens";
    return "end_turn";
}



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



static void append_anthropic_thinking(buf *b, const char *reasoning, const char *signature) {
    buf_puts(b, "{\"type\":\"thinking\",\"thinking\":");
    json_escape(b, reasoning ? reasoning : "");
    buf_puts(b, ",\"signature\":");
    json_escape(b, signature ? signature : "");
    buf_putc(b, '}');
}



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



bool anthropic_final_response(int fd,
                                     const request *r, const char *id, const char *text,
                                     const char *reasoning, const tool_calls *calls, const char *finish,
                                     int prompt_tokens, int completion_tokens) {
    buf b = {0};
    buf_printf(&b, "{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":", id);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"content\":");
    append_anthropic_content(&b, text, reasoning, calls, id);
    buf_puts(&b, ",\"stop_reason\":");
    json_escape(&b, anthropic_stop_reason(finish));
    buf_puts(&b, ",\"stop_sequence\":null,\"usage\":");
    append_anthropic_usage_json(&b, r, prompt_tokens, completion_tokens);
    buf_puts(&b, "}\n");
    bool ok = http_response(fd, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}



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
    st->mode = pulsar_think_mode_enabled(r->think_mode) ? ANTH_STREAM_THINKING : ANTH_STREAM_TEXT;
    st->guard_second_reasoning =
        pulsar_think_mode_enabled(r->think_mode) && r->has_tools;
    return ok;
}



void anthropic_stream_free(anthropic_stream *st) {
    if (!st) return;
    dsml_tool_stream_free(&st->tool);
}



/* Text and thinking blocks have fixed JSON shapes.  Tool blocks are opened by
 * name later, after the DSML invoke tag is complete, so they use a dedicated
 * opener instead of this helper. */
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



/* The Anthropic side of the shared DSML tool-stream projection (genmsg.cpp
 * dsml_tool_stream_update): a tool_use content block per invocation, its
 * arguments as input_json_delta, stopped when the invocation closes.  The
 * block/stop lifecycle stays here, in the protocol; the DSML walk is shared. */
typedef struct {
    int fd;
    server *s;
    const char *id;
    anthropic_stream *st;
} anthropic_tool_ctx;

static bool anthropic_tool_begin_invoke(void *vctx, dsml_tool_stream *ts, const char *name) {
    anthropic_tool_ctx *c = (anthropic_tool_ctx *)vctx;
    /* This id is already visible to the client.  After final parsing,
     * apply_anthropic_stream_tool_ids() copies it into the parsed tool_call
     * before tool_memory_remember(), so the next tool_result can continue from
     * the live KV state instead of re-rendering canonical JSON. */
    const char *tool_id = dsml_tool_stream_id(c->s, ts, ts->index, API_ANTHROPIC);
    return anthropic_sse_open_tool_block(c->fd, c->st, tool_id, name);
}

static bool anthropic_tool_args_fragment(void *vctx, dsml_tool_stream *ts, const char *text, size_t len) {
    (void)ts;
    anthropic_tool_ctx *c = (anthropic_tool_ctx *)vctx;
    return anthropic_sse_tool_delta_live(c->fd, c->st, text, len);
}

static bool anthropic_tool_end_invoke(void *vctx, dsml_tool_stream *ts) {
    (void)ts;
    anthropic_tool_ctx *c = (anthropic_tool_ctx *)vctx;
    return anthropic_sse_close_block_live(c->fd, c->id, c->st);
}

static const dsml_tool_stream_ops anthropic_tool_ops = {
    anthropic_tool_begin_invoke, anthropic_tool_args_fragment, anthropic_tool_end_invoke,
};



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



bool anthropic_sse_stream_update(int fd, server *s, const request *r, const char *id,
                                        anthropic_stream *st,
                                        const char *raw, size_t raw_len,
                                        bool final) {
    if (!st->active || !raw) return true;

    if (st->mode == ANTH_STREAM_THINKING) {
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
        /* Unclosed-reasoning recovery (upstream ds4 51a1c14): see the OpenAI
         * stream twin. */
        const char *tool = r->has_tools ?
            find_any_tool_start(raw + st->emit_pos) : NULL;
        const bool tool_before_close = tool && (!close || tool < close);
        /* The END must also land before </think> (upstream ds4 0ead8a8): see
         * the OpenAI stream twin. */
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
            if (!anthropic_sse_open_block(fd, st, ANTH_BLOCK_THINKING)) return false;
            if (!anthropic_sse_delta_live(fd, st, ANTH_BLOCK_THINKING,
                                          raw + st->emit_pos,
                                          limit - st->emit_pos)) return false;
            st->sent_thinking = true;
            st->emit_pos = limit;
        }

        if (complete_tool) {
            if (!anthropic_sse_close_block_live(fd, id, st)) return false;
            st->emit_pos = (size_t)(tool - raw);
            st->mode = ANTH_STREAM_SUPPRESS;
            return true;
        }

        if (close || final) {
            if (!anthropic_sse_close_block_live(fd, id, st)) return false;
            if (close) {
                st->emit_pos = (size_t)(close - raw) + strlen("</think>");
                st->mode = ANTH_STREAM_TEXT;
            } else {
                st->mode = ANTH_STREAM_SUPPRESS;
                return true;
            }
        } else {
            return true;
        }
    }

    if (st->mode == ANTH_STREAM_TEXT) {
        if (st->guard_second_reasoning) {
            /* Second </think> before any tool marker: the held text was
             * another reasoning pass — emit it as a thinking block. */
            const char *close = strstr(raw + st->emit_pos, "</think>");
            const char *tool2 = r->has_tools ?
                find_any_tool_start(raw + st->emit_pos) : NULL;
            if (close && (!tool2 || close < tool2)) {
                const size_t limit = (size_t)(close - raw);
                if (limit > st->emit_pos) {
                    if (!anthropic_sse_open_block(fd, st, ANTH_BLOCK_THINKING)) return false;
                    if (!anthropic_sse_delta_live(fd, st, ANTH_BLOCK_THINKING,
                                                  raw + st->emit_pos,
                                                  limit - st->emit_pos)) return false;
                    st->sent_thinking = true;
                }
                if (!anthropic_sse_close_block_live(fd, id, st)) return false;
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
            if (!anthropic_sse_open_block(fd, st, ANTH_BLOCK_TEXT)) return false;
            if (!anthropic_sse_delta_live(fd, st, ANTH_BLOCK_TEXT,
                                          raw + st->emit_pos,
                                          limit - st->emit_pos)) return false;
            st->sent_text = true;
            st->emit_pos = limit;
        }

        if (tool) {
            if (!anthropic_sse_close_block_live(fd, id, st)) return false;
            st->emit_pos = (size_t)(tool - raw);
            /* On normal token-by-token updates, switch from hidden text to a
             * live tool_use projection as soon as the DSML block starts.  On
             * final catch-up from plain text, leave the block for the existing
             * final emitter so old non-incremental behavior stays unchanged. */
            if (!final &&
                dsml_tool_stream_init(&st->tool, raw, raw_len, st->emit_pos)) {
                st->mode = ANTH_STREAM_TOOL;
            } else {
                st->mode = ANTH_STREAM_SUPPRESS;
            }
        } else if (final) {
            if (!anthropic_sse_close_block_live(fd, id, st)) return false;
            st->mode = ANTH_STREAM_SUPPRESS;
        }
    }

    if (st->mode == ANTH_STREAM_TOOL) {
        anthropic_tool_ctx ctx = {fd, s, id, st};
        if (!dsml_tool_stream_update(&st->tool, &anthropic_tool_ops, &ctx, raw, raw_len)) return false;
        if (final && st->tool.active &&
            !dsml_tool_stream_finalize(&st->tool, &anthropic_tool_ops, &ctx, raw, raw_len)) return false;
        if (!st->tool.active) st->mode = ANTH_STREAM_SUPPRESS;
    }
    return true;
}



static bool anthropic_sse_tool_blocks_live(int fd, const request *r, const char *id,
                                           anthropic_stream *st,
                                           const tool_calls *calls) {
    (void)r;
    if (!calls) return true;

    buf b = {0};
    /* Tool calls completed by anthropic_tool_stream_update() have already
     * produced start/delta/stop events.  Only emit the tail calls that were not
     * seen by the live projection, for example if the first DSML bytes only
     * become available during final flush. */
    int already_streamed = st->tool.emitted_any ? st->tool.index : 0;
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



static bool anthropic_sse_stop_live(int fd, const char *finish,
                                    int completion_tokens) {
    buf b = {0};
    buf_puts(&b, "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":");
    json_escape(&b, anthropic_stop_reason(finish));
    buf_puts(&b, ",\"stop_sequence\":null},\"usage\":{\"output_tokens\":");
    buf_printf(&b, "%d}}", completion_tokens);
    bool ok = sse_event(fd, "message_delta", b.ptr);
    buf_free(&b);
    if (ok) ok = sse_event(fd, "message_stop", "{\"type\":\"message_stop\"}");
    return ok;
}



bool anthropic_sse_finish_live(int fd, server *s, const request *r, const char *id,
                                      anthropic_stream *st, const char *raw,
                                      size_t raw_len, const tool_calls *calls,
                                      const char *finish, int completion_tokens) {
    if (!anthropic_sse_stream_update(fd, s, r, id, st, raw, raw_len, true)) return false;

    if (st->sent_thinking && !st->sent_text && (!calls || calls->len == 0)) {
        if (!anthropic_sse_open_block(fd, st, ANTH_BLOCK_TEXT)) return false;
        if (!anthropic_sse_close_block_live(fd, id, st)) return false;
    }

    if (!anthropic_sse_tool_blocks_live(fd, r, id, st, calls)) return false;
    return anthropic_sse_stop_live(fd, finish, completion_tokens);
}




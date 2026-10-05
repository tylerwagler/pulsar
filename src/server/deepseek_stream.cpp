/* L267: DeepSeek's live stream projection -- ONE walk over the raw generated text for every protocol.
 *
 * The text is "<think>reasoning</think>answer" with DSML tool-call blocks in the answer (or, recovered,
 * in unclosed reasoning).  The walk decides what each released byte is -- reasoning, answer, a tool
 * call, or held protocol syntax -- and drives the request's protocol sink (chat_sink): OpenAI deltas,
 * Anthropic content blocks or Responses items.  Before L267 each protocol carried its own copy of this
 * walk; tests/sse_golden pins the bytes they sent.  The final DSML parse is genmsg.cpp's, and the
 * finish is the protocol's. */
#include "pulsar_server_internal.h"

void deepseek_stream_walk_init(deepseek_stream_walk *w, const request *r) {
    memset(w, 0, sizeof(*w));
    const bool think = pulsar_think_mode_enabled(r->think_mode);
    w->mode = think ? DS_WALK_THINKING : DS_WALK_TEXT;
    w->guard_second_reasoning = think && r->has_tools;
}

void deepseek_stream_walk_free(deepseek_stream_walk *w) {
    if (w) dsml_tool_stream_free(&w->tool);
}

bool deepseek_stream_update(deepseek_stream_walk *w, chat_sink *k, const char *raw, size_t raw_len, bool final) {
    if (!raw) return true;
    const request *r = k->r;

    if (w->mode == DS_WALK_THINKING) {
        if (!w->checked_think_prefix) {
            /* The chat template ends the prompt with the literal `<think>`, so
             * generation usually starts mid-reasoning; a repeated open tag is
             * skipped.  The mode changes to TEXT only when `</think>` is
             * observed (a no-prefix shortcut leaked reasoning as answer). */
            const char *open = "<think>";
            const size_t open_len = strlen(open);
            if (raw_len < open_len && !strncmp(raw, open, raw_len) && !final) {
                return true;
            }
            if (raw_len >= open_len && !strncmp(raw, open, open_len)) {
                w->emit_pos = open_len;
            }
            w->checked_think_prefix = true;
        }

        const char *close = strstr(raw + w->emit_pos, "</think>");
        /* A tool call starting before any </think> is the unclosed-reasoning
         * recovery case (upstream ds4 51a1c14): stream only the prose before
         * the marker as reasoning, hold until the block completes, and keep
         * the protocol bytes off every channel. */
        const char *tool = r->has_tools ?
            find_any_tool_start(raw + w->emit_pos) : NULL;
        const bool tool_before_close = tool && (!close || tool < close);
        /* The END must also land before </think>: a block whose end falls past
         * the close straddles the reasoning boundary and is NOT an executable
         * call (upstream ds4 0ead8a8). */
        const char *tool_end = tool_before_close ? find_any_tool_end(tool) : NULL;
        const bool complete_tool = tool_end && (!close || tool_end < close);
        size_t limit;
        if (complete_tool) {
            limit = trim_tool_separator_ws(raw, w->emit_pos,
                                           (size_t)(tool - raw));
        } else if (close) {
            /* An incomplete marker that remains inside a closed think block is
             * reasoning text, not an executable call. */
            limit = (size_t)(close - raw);
        } else if (final) {
            /* Match non-stream parsing: flush incomplete DSML as reasoning. */
            limit = raw_len;
        } else if (tool_before_close) {
            limit = trim_tool_separator_ws(raw, w->emit_pos,
                                           (size_t)(tool - raw));
        } else {
            const size_t hold = strlen("</think>") - 1;
            limit = raw_len > hold ? raw_len - hold : w->emit_pos;
            limit = utf8_stream_safe_len(raw, w->emit_pos, limit, false);
        }

        if (limit > w->emit_pos) {
            if (!k->text(k, true, raw + w->emit_pos, limit - w->emit_pos, limit)) return false;
            w->emit_pos = limit;
        }

        if (complete_tool) {
            if (!k->end(k, false)) return false;
            w->emit_pos = (size_t)(tool - raw);
            w->mode = DS_WALK_SUPPRESS;
            return true;
        }

        if (close || final) {
            if (!k->end(k, close != NULL)) return false;
            if (!close) {
                w->mode = DS_WALK_SUPPRESS;
                return true;
            }
            w->emit_pos = (size_t)(close - raw) + strlen("</think>");
            w->mode = DS_WALK_TEXT;
        } else {
            return true;
        }
    }

    if (w->mode == DS_WALK_TEXT) {
        if (w->guard_second_reasoning) {
            /* A second </think> before any tool marker means the held text
             * was another reasoning pass -- reroute it. A tool marker or the
             * final flush releases the hold as genuine answer text. */
            const char *close = strstr(raw + w->emit_pos, "</think>");
            const char *tool2 = r->has_tools ?
                find_any_tool_start(raw + w->emit_pos) : NULL;
            if (close && (!tool2 || close < tool2)) {
                const size_t limit = (size_t)(close - raw);
                if (limit > w->emit_pos &&
                    !k->text(k, true, raw + w->emit_pos, limit - w->emit_pos, limit)) return false;
                if (!k->end(k, false)) return false;
                w->emit_pos = limit + strlen("</think>");
                w->guard_second_reasoning = false;
            } else if (!tool2 && !final) {
                return true;
            } else {
                w->guard_second_reasoning = false;
            }
        }

        const char *tool = r->has_tools ? find_any_tool_start(raw + w->emit_pos) : NULL;
        size_t limit = text_stream_safe_limit(raw, w->emit_pos, raw_len,
                                              r->has_tools, final);

        if (limit > w->emit_pos) {
            if (!k->text(k, false, raw + w->emit_pos, limit - w->emit_pos, limit)) return false;
            w->emit_pos = limit;
        }

        if (tool) {
            if (!k->end(k, false)) return false;
            w->emit_pos = (size_t)(tool - raw);
            /* Switch to the live tool-call projection as soon as the DSML block
             * starts -- on a protocol that streams calls; one first seen in the
             * final flush goes to the finish unless the protocol streams it then
             * too. */
            if (k->tool_ops && (!final || k->tools_on_final) &&
                dsml_tool_stream_init(&w->tool, raw, raw_len, w->emit_pos)) {
                w->mode = DS_WALK_TOOL;
            } else {
                w->mode = DS_WALK_SUPPRESS;
            }
        } else if (final) {
            if (!k->end(k, false)) return false;
            w->mode = DS_WALK_SUPPRESS;
        }
    }

    if (w->mode == DS_WALK_TOOL) {
        if (!dsml_tool_stream_update(&w->tool, k->tool_ops, k, raw, raw_len)) return false;
        if (final && w->tool.active &&
            !dsml_tool_stream_finalize(&w->tool, k->tool_ops, k, raw, raw_len)) return false;
        if (!w->tool.active) w->mode = DS_WALK_SUPPRESS;
    }
    return true;
}

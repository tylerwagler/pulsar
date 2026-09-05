#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* The ONE <think> / </think> tag scanner for the streaming display tools (CLI,
 * eval TUI, agent).  Model text arrives one token piece at a time and a tag
 * can straddle pieces, so the scanner withholds a '<'-led prefix that could
 * still become a tag and replays it with the next piece.  Before L187 this
 * loop existed verbatim in pulsar_cli.cpp, pulsar_eval.cpp, agent/markdown.cpp
 * (dead: never fed text) and agent/tool_viz.cpp (with DSML interleaving).
 *
 * The server's three detectors (generate.cpp rolling tail, genmsg.cpp,
 * anthropic_stream.cpp) are a different shape (they split a finished string,
 * not a byte stream) and are not served by this header. */

/** True when `p[0..n)` starts with the whole of `prefix`. */
static inline bool bytes_has_prefix(const char *p, size_t n, const char *prefix) {
    const size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

/** True when `p[0..n)` is a proper (shorter) prefix of `prefix`: more bytes
 * could still complete it. */
static inline bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    const size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}

/** Scanner state.  Zero-initialise, or set `in_think` when generation starts
 * inside a reasoning block (the assistant prefix already emitted <think>). */
struct pulsar_think_scanner {
    bool in_think;      ///< between `<think>` and `</think>`
    char pending[8];    ///< '<'-led bytes withheld while a tag may still be forming
    size_t pending_len; ///< bytes held in `pending`
};

/** Feed `len` bytes of model text to the scanner.
 *
 * Sink contract (duck-typed, resolved at compile time):
 *   bool tags_enabled()   -- tags are recognised at this byte (false while a
 *                            DSML block is being parsed: its bytes are not prose)
 *   void think_open_tag() -- a complete <think> was consumed; fires BEFORE
 *                            in_think flips so a held prefix flushes in the old state
 *   void think_close_tag()-- a complete </think> was consumed; fires BEFORE
 *                            in_think flips (reset styling here)
 *   bool at_line_start()  -- the last byte the sink wrote was '\n' (or nothing yet)
 *   void newline()        -- write one raw '\n' outside any text formatting
 *   void text(char c)     -- one byte of prose or thinking text; read
 *                            scanner.in_think for styling
 *
 * After </think> the scanner brings the output to a line start and then
 * writes `blank_lines_after_close` more newlines (CLI: 0, agent: 1).
 * `finish` (end of generation) releases a held prefix as text: an unfinished
 * "<thi" at the very end is prose. */
template <class Sink>
static inline void pulsar_think_scan(pulsar_think_scanner *s, const char *text, size_t len,
                                     bool finish, int blank_lines_after_close, Sink &sink) {
    static const char open_tag[] = "<think>";
    static const char close_tag[] = "</think>";
    static_assert(sizeof(close_tag) - 1 <= sizeof(s->pending),
                  "pending must hold every proper prefix of the longest tag");

    assert(s->pending_len <= sizeof(s->pending));
    const size_t total = s->pending_len + len;
    char stack_buf[256];
    static_assert(sizeof(stack_buf) >= sizeof(s->pending), "the small buffer holds a whole pending array");
    char *buf = total <= sizeof(stack_buf) ? stack_buf : (char *)malloc(total);
    if (!buf) abort();
    /* Copy the whole pending array (fixed size, so no length the compiler has
     * to range-prove): bytes past pending_len are overwritten by `text` or lie
     * beyond `total` and are never read. */
    memcpy(buf, s->pending, sizeof(s->pending));
    if (len) memcpy(buf + s->pending_len, text, len);
    s->pending_len = 0;

    size_t i = 0;
    while (i < total) {
        const char *cur = buf + i;
        const size_t rem = total - i;
        if (sink.tags_enabled()) {
            if (bytes_has_prefix(cur, rem, open_tag)) {
                sink.think_open_tag();
                s->in_think = true;
                i += sizeof(open_tag) - 1;
                continue;
            }
            if (bytes_has_prefix(cur, rem, close_tag)) {
                sink.think_close_tag();
                s->in_think = false;
                if (!sink.at_line_start()) sink.newline();
                for (int k = 0; k < blank_lines_after_close; k++) sink.newline();
                i += sizeof(close_tag) - 1;
                continue;
            }
            if (!finish && cur[0] == '<' &&
                (bytes_is_partial_prefix(cur, rem, open_tag) ||
                 bytes_is_partial_prefix(cur, rem, close_tag)))
            {
                /* rem < strlen(close_tag) here, so it fits (static_assert above) */
                memcpy(s->pending, cur, rem);
                s->pending_len = rem;
                break;
            }
        }
        sink.text(cur[0]);
        i++;
    }
    if (buf != stack_buf) free(buf);
}

#pragma once

#include <stddef.h>

/* The ONE UTF-8 well-formedness rule for the engine, CLI, eval TUI and agent.
 *
 * Strict per Unicode 15 Table 3-7: a lead byte is 0xC2-0xDF (2 bytes),
 * 0xE0-0xEF (3) or 0xF0-0xF4 (4); 0xC0/0xC1 (overlong 2-byte forms) and
 * 0xF5-0xFF are NOT leads; the second byte is range-restricted after 0xE0
 * (>= 0xA0, no overlong 3-byte), 0xED (< 0xA0, no surrogates), 0xF0 (>= 0x90,
 * no overlong 4-byte) and 0xF4 (< 0x90, nothing above U+10FFFF).
 *
 * Before L187 this rule lived six times in two families: a loose bitmask in
 * the tokenizer that took 0xC0/0xC1 and 0xF5-0xF7 as leads and never looked at
 * the continuation bytes, and lead-range/Table 3-7 copies in the agent, the
 * CLI and the server.  The server's `utf8_stream_safe_len` (request.cpp) is
 * the streaming hold-back built on the same rule and is meant to call
 * `utf8_seq_len` below. */

/** Bytes in the UTF-8 sequence that `lead` begins: 1 for ASCII, 2-4 for a
 * valid multibyte lead, 0 for a byte that cannot start a sequence (a
 * continuation byte, 0xC0, 0xC1 or 0xF5-0xFF). */
static inline int utf8_seq_len(unsigned char lead) {
    if (lead < 0x80) return 1;
    if (lead >= 0xc2 && lead <= 0xdf) return 2;
    if (lead >= 0xe0 && lead <= 0xef) return 3;
    if (lead >= 0xf0 && lead <= 0xf4) return 4;
    return 0;
}

/** Length of the well-formed UTF-8 sequence at the start of `p[0..n)`, or 0
 * when the bytes there do not form one (bad lead, truncated by `n`, a
 * continuation byte outside 0x80-0xBF, or a Table 3-7 second-byte
 * violation).  A caller advancing through arbitrary bytes treats 0 as "this
 * one byte is its own unit". */
static inline int utf8_seq_ok(const unsigned char *p, size_t n) {
    if (n == 0) return 0;
    const int len = utf8_seq_len(p[0]);
    if (len == 0 || (size_t)len > n) return 0;
    if (len == 1) return 1;
    const unsigned char c1 = p[1];
    if (p[0] == 0xe0 && c1 < 0xa0) return 0;
    if (p[0] == 0xed && c1 >= 0xa0) return 0;
    if (p[0] == 0xf0 && c1 < 0x90) return 0;
    if (p[0] == 0xf4 && c1 >= 0x90) return 0;
    for (int i = 1; i < len; i++) {
        if ((p[i] & 0xc0) != 0x80) return 0;
    }
    return len;
}

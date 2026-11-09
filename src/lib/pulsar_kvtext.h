/** @file
 *  Rendered-text helpers for KV reuse shared by pulsar-server and pulsar-agent, plus small byte / path primitives. */
#ifndef PULSAR_KVTEXT_H
#define PULSAR_KVTEXT_H

/* Rendered-text helpers for KV reuse, shared by pulsar-server's disk KV cache
 * and pulsar-agent's sessions (both persist KV as segment chains keyed by the
 * rendered text of their tokens, pulsar_kvchain.h), plus the small byte/path
 * primitives their own files use. */

#include "pulsar.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/** The bytes `tokens` render to: pulsar_history_text (each id's text; an image block once, as its placeholder);
 *  caller frees; NULL when an image block is malformed. */
/// @param engine  the engine whose vocabulary renders the ids
/// @param tokens  the token ids
/// @param out_len receives the text's length in bytes
/// @return the malloc'd text, or NULL when an image block is malformed
char *pulsar_kvtext_render_tokens_text(pulsar_engine *engine,
                                     const pulsar_tokens *tokens,
                                     size_t *out_len);
/** Is `prefix` a byte prefix of `text`?
 *  @param text       the text
 *  @param text_len   its length in bytes
 *  @param prefix     the candidate prefix
 *  @param prefix_len its length in bytes
 *  @return true when prefix_len <= text_len and the bytes agree */
bool pulsar_kvtext_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len);
/** L196: does `text` end where `tokens` end?  A checkpoint's text names the
 * bytes of its tokens, so it ends with the EOS piece exactly when the last
 * token is the EOS.  A tool-call turn stops at the closing tool_calls tag
 * without sampling one; a key that carried the EOS anyway made every consumer
 * tokenise the request bytes AFTER it, so the EOS never entered the bank and
 * the live history byte-diverged from every later render at that point.
 * Store and load both refuse a disagreeing pair. */
/// @param engine   the engine (its EOS id and text)
/// @param text     the rendered text
/// @param text_len its length in bytes
/// @param tokens   the live tokens, or NULL
/// @return true when the text ends with the EOS piece exactly when the tokens end with the EOS
bool pulsar_kvtext_text_ends_with_live(pulsar_engine *engine, const char *text,
                                     size_t text_len, const pulsar_tokens *tokens);
/** Replace `dst` with the first `n` tokens of `src` (all of them when it is shorter).
 *  @param dst the destination (emptied first)
 *  @param src the source, or NULL (dst left empty)
 *  @param n   tokens to copy */
void pulsar_kvtext_tokens_copy_prefix(pulsar_tokens *dst, const pulsar_tokens *src, int n);
/** `exact_prefix`'s tokens, then `suffix_text` tokenised as RENDERED chat --
 * except the bytes inside `spans`/`n_spans` (the suffix's CLIENT-DATA ranges),
 * which are plain text (L223), so a client spelling in the suffix cannot become
 * a control token.  `spans` may be NULL when the suffix is entirely the
 * server's own text. */
/// @param engine       the engine whose tokenizer renders the suffix
/// @param exact_prefix the tokens kept as they are
/// @param suffix_text  the rendered text that follows them
/// @param spans        the suffix's client-data byte ranges, tokenised as plain text, or NULL
/// @param n_spans      entries in `spans`
/// @param out          receives the prompt's tokens
void pulsar_kvtext_build_prompt_from_exact_prefix_and_text_suffix(
        pulsar_engine *engine,
        const pulsar_tokens *exact_prefix,
        const char *suffix_text,
        const pulsar_text_span *spans,
        uint32_t n_spans,
        pulsar_tokens *out);

/** The SHA-1 of a byte range as 40 lowercase hex chars.
 *  @param ptr the bytes
 *  @param len their count
 *  @param out receives the hex digest and a NUL */
void pulsar_kvtext_sha1_bytes_hex(const void *ptr, size_t len, char out[41]);
/** `dir` + "/" + `name` (no slash doubled when dir already ends in one).
 *  @param dir  the directory
 *  @param name the entry name
 *  @return the malloc'd path (caller frees) */
char *pulsar_kvtext_path_join(const char *dir, const char *name);
/** Store `v` little-endian in 4 bytes.
 *  @param p receives the 4 bytes
 *  @param v the value */
void pulsar_kvtext_le_put32(uint8_t *p, uint32_t v);
/** Read a 4-byte little-endian value.
 *  @param p the 4 bytes
 *  @return the value */
uint32_t pulsar_kvtext_le_get32(const uint8_t *p);

#endif

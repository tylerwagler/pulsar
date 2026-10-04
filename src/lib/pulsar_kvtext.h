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

/** The bytes `tokens` render to (each id's text, concatenated); caller frees. */
char *pulsar_kvtext_render_tokens_text(pulsar_engine *engine,
                                     const pulsar_tokens *tokens,
                                     size_t *out_len);
bool pulsar_kvtext_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len);
/** L196: does `text` end where `tokens` end?  A checkpoint's text names the
 * bytes of its tokens, so it ends with the EOS piece exactly when the last
 * token is the EOS.  A tool-call turn stops at the closing tool_calls tag
 * without sampling one; a key that carried the EOS anyway made every consumer
 * tokenise the request bytes AFTER it, so the EOS never entered the bank and
 * the live history byte-diverged from every later render at that point.
 * Store and load both refuse a disagreeing pair. */
bool pulsar_kvtext_text_ends_with_live(pulsar_engine *engine, const char *text,
                                     size_t text_len, const pulsar_tokens *tokens);
void pulsar_kvtext_tokens_copy_prefix(pulsar_tokens *dst, const pulsar_tokens *src, int n);
/** `exact_prefix`'s tokens, then `suffix_text` tokenised as RENDERED chat --
 * except the bytes inside `spans`/`n_spans` (the suffix's CLIENT-DATA ranges),
 * which are plain text (L223), so a client spelling in the suffix cannot become
 * a control token.  `spans` may be NULL when the suffix is entirely the
 * server's own text. */
void pulsar_kvtext_build_prompt_from_exact_prefix_and_text_suffix(
        pulsar_engine *engine,
        const pulsar_tokens *exact_prefix,
        const char *suffix_text,
        const pulsar_text_span *spans,
        uint32_t n_spans,
        pulsar_tokens *out);

void pulsar_kvtext_sha1_bytes_hex(const void *ptr, size_t len, char out[41]);
char *pulsar_kvtext_path_join(const char *dir, const char *name);
void pulsar_kvtext_le_put32(uint8_t *p, uint32_t v);
uint32_t pulsar_kvtext_le_get32(const uint8_t *p);

#endif

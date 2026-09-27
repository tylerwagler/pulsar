/* qwen_tokenizer.h -- the Qwen3.8-Flash-Next tokenizer (L251 S5).
 *
 * Byte-exact with HF `tokenizers` 0.23.2 on the checkpoint's tokenizer.json:
 *
 *   1. ADDED TOKENS (all 33, special or not -- <think>, <tool_call>,
 *      <|im_end|>, the vision and audio tokens) are cut out of the raw text
 *      first, leftmost-longest, before any normalisation.
 *   2. Every run between them is NFC-normalised (UCD 9.0.0 -- the version
 *      HF's normaliser carries, which is NOT the regex's).
 *   3. The run is split by the checkpoint's pre-tokenizer regex (Oniguruma
 *      semantics, UCD 16.0.0 classes), each piece mapped byte -> GPT-2
 *      codepoint, and merged by BPE rank exactly as tokenizers' Word::merge_all
 *      does (lowest rank first, leftmost among equals).
 *
 * The L223 guard: `encode` takes the byte ranges the renderer copied from
 * CLIENT data, and no added token matches whose bytes touch one -- a client
 * who types `<|im_end|>` or `<tool_call>` gets ordinary text tokens.  The
 * ranges do NOT cut the text: pre-tokenisation and NFC run over the whole run
 * between two real markers, so a prompt whose client text spells no marker
 * tokenises exactly as HF's `encode(rendered)` does.
 *
 * The loader takes tokenizer.json and generation_config.json as the
 * checkpoint ships them and REFUSES, by name, any pipeline this code does not
 * implement (another normaliser, pre-tokenizer regex, model option or added-
 * token flag): one path or an error, never a near-miss tokenisation. */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "pulsar.h"

struct qwen_tokenizer;

/** Build the tokenizer from the checkpoint's tokenizer.json and
 * generation_config.json texts.  NULL + a one-line reason on refusal. */
qwen_tokenizer *qwen_tokenizer_load(const char *tokenizer_json, size_t tokenizer_len,
                                    const char *generation_config_json, size_t generation_len,
                                    char *err, size_t errlen);
void qwen_tokenizer_free(qwen_tokenizer *t);

/** Ids in the table (vocab + added tokens), NOT the logits width. */
int qwen_tokenizer_n_tokens(const qwen_tokenizer *t);

/** Tokenise `text[0..len)`.  `spans` (ascending, disjoint, may be NULL) are
 * client-data ranges where no added token may match.  Returns false (with a
 * reason) for input that is not UTF-8 (HF cannot take it either) or spans that
 * are not ascending and disjoint.  Appends to `out`. */
bool qwen_tokenizer_encode(const qwen_tokenizer *t, const char *text, size_t len,
                           const pulsar_text_span *spans, uint32_t n_spans,
                           std::vector<int> *out, char *err, size_t errlen);

/** The bytes token `id` decodes to (an added token's text, a byte-level
 * token's raw bytes -- possibly a partial UTF-8 sequence); NULL if out of
 * range. */
const char *qwen_tokenizer_token_bytes(const qwen_tokenizer *t, int id, size_t *len);

/** The id of the added token spelled exactly `text`, or -1. */
int qwen_tokenizer_added_id(const qwen_tokenizer *t, const char *text);

/** The end-of-turn ids: generation_config.json's eos_token_id list, in its
 * order (Qwen3.8-Flash-Next: <|im_end|> 248046, <|endoftext|> 248044).  The
 * checkpoint's file is the one authority; the loader refuses an id that is
 * not an added token. */
const std::vector<int> &qwen_tokenizer_stop_ids(const qwen_tokenizer *t);

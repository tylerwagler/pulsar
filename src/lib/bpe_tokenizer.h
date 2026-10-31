/** @file
 *  The shared byte-level BPE tokenizer engine; each family's tokenizer is a configuration of it. */
/* bpe_tokenizer.h -- the ONE byte-level BPE tokenizer engine (L251 S5; shared since L295 F5).
 *
 * A family's tokenizer is a CONFIGURATION of this engine, never a copy of it: the vocabulary and merge
 * ranks (from the checkpoint's own file), its added tokens, its stop ids, whether runs are NFC-normalised,
 * and the pre-tokenizer pattern -- chosen by name from a registry (bpe_pretok_*), never a regex compiled
 * at runtime.  Byte-exact with HF `tokenizers` 0.23.2 on every configuration it loads:
 *
 *   1. ADDED TOKENS (special or not) are cut out of the raw text first, leftmost-longest, before any
 *      normalisation.
 *   2. Every run between them is NFC-normalised (UCD 9.0.0 -- the version HF's normaliser carries, which is
 *      NOT the regex's) when the configuration says so.
 *   3. The run is split by the configuration's pre-tokenizer pattern (Oniguruma semantics, UCD 16.0.0
 *      classes), each piece mapped byte -> GPT-2 codepoint, and merged:
 *      - HF merges (tokenizer.json): by merge rank exactly as tokenizers' Word::merge_all does (lowest rank
 *        first, leftmost among equals) -- after, with ignore_merges, the whole-piece lookup below;
 *      - tiktoken ranks: a piece that IS a token is that token (tokenizers' ignore_merges, tiktoken's
 *        encoder lookup); otherwise the adjacent pair whose concatenation has the lowest rank merges
 *        first, leftmost among equals (tiktoken's byte_pair_merge).
 *
 * The L223 guard: `encode` takes the byte ranges the renderer copied from CLIENT data, and no added token
 * matches whose bytes touch one -- a client who types a marker gets ordinary text tokens.  The ranges do
 * NOT cut the text: pre-tokenisation and NFC run over the whole run between two real markers, so a prompt
 * whose client text spells no marker tokenises exactly as HF's `encode(rendered)` does.
 *
 * The loaders take the checkpoint's files as shipped and REFUSE, by name, any pipeline this code does not
 * implement (another normaliser, an unregistered pre-tokenizer pattern, a model option or an added-token
 * flag): one path or an error, never a near-miss tokenisation. */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "pulsar.h"

struct bpe_tokenizer;

/** One pre-tokenizer pattern the engine implements: its name, the regex as the checkpoint's file spells it
 * (JSON-decoded), and the hand-written matcher that IS that regex under Oniguruma (graded over every scalar
 * value against tokenizers' own Split).  The registry is the one place a pattern lives. */
struct bpe_pretok_pattern {
    const char *name;   ///< "qwen35" (Qwen3.5 / 3.8), "o200k" (Inkling; tiktoken's o200k_base split), "tekken" (Nemotron-H)
    const char *regex;  ///< the spelling tokenizer.json carries
    /** The end of the first alternative that matches at `i` of the codepoints `s` (classes `cl`, from the
     *  engine's Unicode table), `n` long. */
    size_t (*match_at)(const uint32_t *s, const uint32_t *cl, size_t n, size_t i);
};
/** The registered pattern named `name`, or NULL. */
/// @param name the pattern's registry name ("qwen35", "o200k", "tekken")
/// @return the registry entry (static storage), or NULL when no pattern has that name
const bpe_pretok_pattern *bpe_pretok_by_name(const char *name);
/** The registered pattern whose regex is spelled exactly `regex`, or NULL. */
/// @param regex the regex as tokenizer.json spells it (JSON-decoded)
/// @return the registry entry (static storage), or NULL when no pattern is spelled that way
const bpe_pretok_pattern *bpe_pretok_by_regex(const char *regex);

/** HF tokenizer.json (BPE model, Split(registered regex) + ByteLevel pre-tokenizer, ByteLevel decoder and
 * post-processor) in one of the pipelines the registered pattern names -- Qwen's (NFC normaliser, every merge
 * applied) or Nemotron-H's (no normaliser, ignore_merges: a piece found whole in the vocab is that token) --
 * plus the config file whose `eos_token_id` (an int or a list, in its order) lists the stop ids (Qwen,
 * Nemotron-H: generation_config.json).  NULL + a one-line reason on refusal. */
/// @param tokenizer_json  tokenizer.json's bytes (need not be NUL-terminated)
/// @param tokenizer_len   their length
/// @param eos_config_json the config file carrying `eos_token_id` (generation_config.json)
/// @param eos_len         its length
/// @param err             receives the refusal reason (may be NULL)
/// @param errlen          size of `err`
/// @return a new tokenizer (free with bpe_tokenizer_free), or NULL on refusal
bpe_tokenizer *bpe_tokenizer_load_hf(const char *tokenizer_json, size_t tokenizer_len, const char *eos_config_json,
                                     size_t eos_len, char *err, size_t errlen);

/** A tiktoken rank file (one "base64(bytes) rank" line per token, rank = id, every id in [0, n) once), the
 * registered pattern `pattern` it is split by (no normaliser: tiktoken has none), the special tokens -- every
 * entry of tokenizer_config.json's `added_tokens_decoder`, whose content must be the rank file's bytes at
 * that id -- and the config file whose `eos_token_id` lists the stop ids (Inkling: config.json).  NULL + a
 * one-line reason on refusal. */
/// @param ranks                 the tiktoken rank file's bytes
/// @param ranks_len             their length
/// @param pattern               the registered pre-tokenizer pattern's name (bpe_pretok_by_name)
/// @param tokenizer_config_json tokenizer_config.json (its `added_tokens_decoder` gives the special tokens)
/// @param config_len            its length
/// @param eos_config_json       the config file carrying `eos_token_id` (Inkling: config.json)
/// @param eos_len               its length
/// @param err                   receives the refusal reason (may be NULL)
/// @param errlen                size of `err`
/// @return a new tokenizer (free with bpe_tokenizer_free), or NULL on refusal
bpe_tokenizer *bpe_tokenizer_load_tiktoken(const char *ranks, size_t ranks_len, const char *pattern,
                                           const char *tokenizer_config_json, size_t config_len,
                                           const char *eos_config_json, size_t eos_len, char *err, size_t errlen);
/// Free a tokenizer from either loader.
/// @param t the tokenizer; NULL is a no-op
void bpe_tokenizer_free(bpe_tokenizer *t);

/** The configuration's pre-tokenizer pattern. */
/// @param t the tokenizer
/// @return its registry entry (static storage, never NULL for a loaded tokenizer)
const bpe_pretok_pattern *bpe_tokenizer_pattern(const bpe_tokenizer *t);

/** Ids in the table (vocab + added tokens), NOT the logits width. */
/// @param t the tokenizer
/// @return the id count: vocab plus added tokens
int bpe_tokenizer_n_tokens(const bpe_tokenizer *t);

/** Tokenise `text[0..len)`.  `spans` (ascending, disjoint, may be NULL) are client-data ranges where no added
 * token may match.  Returns false (with a reason) for input that is not UTF-8 (HF cannot take it either) or
 * spans that are not ascending and disjoint.  Appends to `out`. */
/// @param t       the tokenizer
/// @param text    the text (UTF-8, need not be NUL-terminated)
/// @param len     its length in bytes
/// @param spans   client-data byte ranges where no added token may match (ascending, disjoint; may be NULL)
/// @param n_spans entries in `spans`
/// @param out     the ids are appended here
/// @param err     receives the reason on failure (may be NULL)
/// @param errlen  size of `err`
/// @return true on success, false for non-UTF-8 input or malformed spans
bool bpe_tokenizer_encode(const bpe_tokenizer *t, const char *text, size_t len, const pulsar_text_span *spans,
                          uint32_t n_spans, std::vector<int> *out, char *err, size_t errlen);

/** The bytes token `id` decodes to (an added token's text, a byte-level token's raw bytes -- possibly a
 * partial UTF-8 sequence); NULL if out of range. */
/// @param t   the tokenizer
/// @param id  the token id
/// @param len receives the byte count (may be NULL)
/// @return the token's bytes (owned by the tokenizer; use `*len`, the bytes may hold NUL), or NULL if out of range
const char *bpe_tokenizer_token_bytes(const bpe_tokenizer *t, int id, size_t *len);

/** The id of the added token spelled exactly `text`, or -1. */
/// @param t    the tokenizer
/// @param text the added token's exact spelling (NUL-terminated)
/// @return its id, or -1 when no added token is spelled `text`
int bpe_tokenizer_added_id(const bpe_tokenizer *t, const char *text);

/** The end-of-turn ids: the eos config's eos_token_id list, in its order (Qwen3.8-Flash-Next: <|im_end|>
 * 248046, <|endoftext|> 248044; Inkling: <|content_model_end_sampling|> 200006).  The checkpoint's file is
 * the one authority; the loaders refuse an id that is not an added token. */
/// @param t the tokenizer
/// @return the stop ids, in the eos config's order (owned by the tokenizer)
const std::vector<int> &bpe_tokenizer_stop_ids(const bpe_tokenizer *t);

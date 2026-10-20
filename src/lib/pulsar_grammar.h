#ifndef PULSAR_GRAMMAR_H
#define PULSAR_GRAMMAR_H

/* L303: grammar-constrained decoding -- the ONE door to the vendored xgrammar core (src/vendor/xgrammar).
 *
 * A grammar is an xgrammar STRUCTURAL TAG (JSON): a tree of formats -- const strings, JSON schemas (in JSON or a
 * family's tool-parameter style: qwen_xml, deepseek_xml, deepseek_v4_1_xml), regexes, EBNF grammars, sequences,
 * alternatives, repeats.  Every structured output pulsar serves is one (constrain.cpp builds them), so there is
 * one compile path.  It is compiled once against the model's vocabulary into an immutable pulsar_grammar; each constrained request runs its own
 * pulsar_grammar_matcher over it: accept a token, roll back k tokens, fill the next token's mask.
 *
 * THE MASK.  One row is pulsar_grammar_mask_words(n_vocab) uint32 words; bit (t & 31) of word (t >> 5) set = token t
 * may be drawn.  The same [rows][words] layout is the engine's per-row mask (pulsar_token_mask_apply) and the device
 * sampler's (L302): a masked token's logit becomes -inf before every sampling transform, identically in plain and
 * verify rows.  Bits past the tokenizer's vocabulary (a padded logits width) are clear.
 *
 * EXCEPTIONS.  xgrammar reports every error by throwing.  pulsar_grammar.cpp is the only translation unit in pulsar
 * that is built with -fexceptions besides the vendored ones, and it is the boundary: every entry below catches
 * everything and turns it into a return value plus a message, so no exception ever crosses into the
 * -fno-exceptions engine and server.  A grammar that fails to compile fails its request; a matcher that fails
 * fails only its request (rule 9: said once, by the request's error).
 *
 * COMPILE.  pulsar_grammar_compile runs off the decode thread (the request's HTTP thread) on a worker thread with
 * a wall-clock deadline: xgrammar has none of its own, and a pathological schema can take seconds
 * (CVE-2025-32381's class; its recursion guard is xgrammar's).  A compile that misses its deadline fails the
 * request; the worker finishes in the background and its result lands in the compiler's cache, which is capped in
 * bytes.  At most PULSAR_GRAMMAR_COMPILES_MAX compiles run at once; another is refused by name. */

#include <stddef.h>
#include <stdint.h>

/** The compile workers that may run at once (late ones included); a compile past it is refused. */
#define PULSAR_GRAMMAR_COMPILES_MAX 4

typedef struct pulsar_grammar_compiler pulsar_grammar_compiler;
typedef struct pulsar_grammar pulsar_grammar;
typedef struct pulsar_grammar_matcher pulsar_grammar_matcher;


/** Words in one mask row of an `n_vocab`-wide logits row. */
static inline uint32_t pulsar_grammar_mask_words(uint32_t n_vocab) { return (n_vocab + 31u) / 32u; }

/** A compiler over the model's vocabulary: token t's DECODED bytes are tok[t] (tok_len[t] bytes; an empty token
 *  is never allowed in a grammar region), `n_vocab` >= n_tokens is the logits width, `stop` the ids that end a
 *  turn (allowed exactly where the grammar is complete).  `cache_bytes` caps the compiled-grammar cache.
 *  NULL + err on failure. */
pulsar_grammar_compiler *pulsar_grammar_compiler_new(const char *const *tok, const size_t *tok_len, int n_tokens,
                                                     int n_vocab, const int32_t *stop, int n_stop,
                                                     int64_t cache_bytes, char *err, size_t errlen);
void pulsar_grammar_compiler_free(pulsar_grammar_compiler *c);
/** The logits width the compiler's masks cover. */
int pulsar_grammar_compiler_vocab(const pulsar_grammar_compiler *c);
/** Token `t` is one of the compiler's stop ids. */
bool pulsar_grammar_compiler_is_stop(const pulsar_grammar_compiler *c, int t);

/** Compile the structural tag `src` (JSON: {"type":"structural_tag","format":{...}}), waiting at most
 *  `deadline_s` seconds.  NULL + err: the grammar is invalid, the deadline
 *  passed, or the compile workers are all busy.  The result is the caller's (pulsar_grammar_free). */
pulsar_grammar *pulsar_grammar_compile(pulsar_grammar_compiler *c, const char *src,
                                       double deadline_s, char *err, size_t errlen);
void pulsar_grammar_free(pulsar_grammar *g);

/** A matcher at the grammar's start (it keeps its whole history, so any accepted token can be rolled back).
 *  With `end_at_complete` the matcher terminates as soon as the grammar is complete (a region inside a turn: a tool call); without, it
 *  terminates on a stop token (the whole rest of the turn: a response format).  NULL + err on failure. */
pulsar_grammar_matcher *pulsar_grammar_matcher_new(const pulsar_grammar *g, bool end_at_complete,
                                                   char *err, size_t errlen);
void pulsar_grammar_matcher_free(pulsar_grammar_matcher *m);
/** Advance by one token.  false = the grammar does not allow it here (the matcher is unchanged) or the matcher
 *  failed (err says which). */
bool pulsar_grammar_matcher_accept(pulsar_grammar_matcher *m, int32_t token, char *err, size_t errlen);
/** Advance by raw bytes (a token's tail past a trigger).  false = not allowed here or the matcher failed. */
bool pulsar_grammar_matcher_accept_bytes(pulsar_grammar_matcher *m, const char *bytes, size_t n, char *err,
                                         size_t errlen);
/** Undo the last `n` accepts (a byte run counts as one).  false + err on failure. */
bool pulsar_grammar_matcher_rollback(pulsar_grammar_matcher *m, int n, char *err, size_t errlen);
/** Write the next token's mask row (pulsar_grammar_mask_words(vocab) words).  false + err on failure. */
bool pulsar_grammar_matcher_fill(pulsar_grammar_matcher *m, uint32_t *row, char *err, size_t errlen);
/** The grammar has ended (end_at_complete: complete; otherwise a stop token was accepted). */
bool pulsar_grammar_matcher_terminated(const pulsar_grammar_matcher *m);

#endif

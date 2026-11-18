/** @file
 *  Grammar-constrained decoding (L303): the one door to the vendored xgrammar core, and the token-mask layout. */
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
 * may be drawn.  The same [rows][words] layout is the device sampler's mask (L302: pulsar_row_draw.mask, the
 * speculative walk's per-row masks): a masked token's logit becomes -inf before every sampling transform,
 * identically in plain and verify rows.  Bits past the tokenizer's vocabulary (a padded logits width) are clear.
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
 * bytes.
 *
 * ADMISSION (L427; pulsar_grammar_admit.h is the mechanism).  At most PULSAR_GRAMMAR_COMPILES_MAX compiles run at
 * once (late ones included).  A request that finds them all busy WAITS for a slot, in arrival order, inside its own
 * deadline: the wait and the compile share the one deadline.  Requests for the same structural tag on the same
 * compiler -- xgrammar's cache key -- share ONE compile and hold no slot of their own; each waits inside its own
 * deadline, and a compile nobody waits for any more still finishes into the cache.  At most
 * PULSAR_GRAMMAR_WAITERS_MAX requests wait at once (each holds its connection thread); one more is refused at once.
 * pulsar_grammar_compile_info says which of these a call met. */

#include <stddef.h>
#include <stdint.h>

/** The compile workers that may run at once (late ones included); a compile past it waits for a slot. */
#define PULSAR_GRAMMAR_COMPILES_MAX 4
/** The requests that may wait in pulsar_grammar_compile at once -- for a slot, on a shared compile or on their own
 *  (each holds its HTTP connection thread and nothing else); one more is refused at once. */
#define PULSAR_GRAMMAR_WAITERS_MAX 32

/** How a pulsar_grammar_compile call ended. */
typedef enum {
    PULSAR_GRAMMAR_COMPILED = 0,   ///< a grammar came back
    PULSAR_GRAMMAR_INVALID,        ///< the compile ran and failed: the grammar's own error (the request's fault)
    PULSAR_GRAMMAR_LATE,           ///< the compile was still running at the deadline (it finishes into the cache)
    PULSAR_GRAMMAR_NO_SLOT,        ///< the deadline passed while the compile still waited for a slot (server busy)
    PULSAR_GRAMMAR_CROWDED,        ///< PULSAR_GRAMMAR_WAITERS_MAX requests already waited (server busy)
    PULSAR_GRAMMAR_NO_THREAD,      ///< the worker thread could not be started (server busy)
    PULSAR_GRAMMAR_ENDS,           ///< array bound, not an end
} pulsar_grammar_compile_end;

/** What one pulsar_grammar_compile call met on its way (L427), for the caller's status, log and counters. */
typedef struct {
    pulsar_grammar_compile_end end;   ///< how the call ended
    bool shared;                      ///< it joined a compile another request had already asked for
    double slot_wait_s;               ///< seconds its compile waited for a slot while this call waited; 0 = none
} pulsar_grammar_compile_info;

/** The process's compile admission counters (L427), as /metrics exports them. */
typedef struct {
    int running;                          ///< compile workers alive now (<= PULSAR_GRAMMAR_COMPILES_MAX)
    int waiting;                          ///< requests waiting in pulsar_grammar_compile now
    uint64_t slot_waits;                  ///< calls whose compile had to wait for a slot
    double slot_wait_s;                   ///< ... and the seconds they waited, summed
    uint64_t shared;                      ///< calls that joined another request's compile
    uint64_t ends[PULSAR_GRAMMAR_ENDS];   ///< calls by pulsar_grammar_compile_end
} pulsar_grammar_compile_stats;

/// A grammar compiler bound to one model's vocabulary, with its compiled-grammar cache (opaque).
typedef struct pulsar_grammar_compiler pulsar_grammar_compiler;
/// One compiled, immutable grammar; shared by every matcher over it (opaque).
typedef struct pulsar_grammar pulsar_grammar;
/// One request's matcher state over a pulsar_grammar (opaque).
typedef struct pulsar_grammar_matcher pulsar_grammar_matcher;


/** Words in one mask row of an `n_vocab`-wide logits row. */
/// @param n_vocab the logits row width in tokens
/// @return uint32 words per mask row: ceil(n_vocab / 32)
static inline uint32_t pulsar_grammar_mask_words(uint32_t n_vocab) { return (n_vocab + 31u) / 32u; }

/** A compiler over the model's vocabulary: token t's DECODED bytes are tok[t] (tok_len[t] bytes; an empty token
 *  is never allowed in a grammar region), `n_vocab` >= n_tokens is the logits width, `stop` the ids that end a
 *  turn (allowed exactly where the grammar is complete).  `cache_bytes` caps the compiled-grammar cache.
 *  NULL + err on failure. */
/// @param tok         decoded bytes of each token
/// @param tok_len     byte length of each token
/// @param n_tokens    tokens in the tokenizer's vocabulary
/// @param n_vocab     the logits width (>= n_tokens)
/// @param stop        the ids that end a turn
/// @param n_stop      entries in `stop`
/// @param cache_bytes byte cap of the compiled-grammar cache
/// @param err         receives the reason on failure
/// @param errlen      size of `err`
/// @return the compiler, or NULL on failure
pulsar_grammar_compiler *pulsar_grammar_compiler_new(const char *const *tok, const size_t *tok_len, int n_tokens,
                                                     int n_vocab, const int32_t *stop, int n_stop,
                                                     int64_t cache_bytes, char *err, size_t errlen);
/** Free a compiler (pulsar_grammar_compiler_new).
 *  @param c the compiler, or NULL */
void pulsar_grammar_compiler_free(pulsar_grammar_compiler *c);
/** The logits width the compiler's masks cover. */
/// @param c the compiler
/// @return the logits width in tokens
int pulsar_grammar_compiler_vocab(const pulsar_grammar_compiler *c);
/** Token `t` is one of the compiler's stop ids. */
/// @param c the compiler
/// @param t a token id
/// @return true when `t` is a stop id
bool pulsar_grammar_compiler_is_stop(const pulsar_grammar_compiler *c, int t);

/** Compile the structural tag `src` (JSON: {"type":"structural_tag","format":{...}}), waiting at most
 *  `deadline_s` seconds in all: for a slot, then for the compile (ADMISSION above).  NULL + err: the grammar is
 *  invalid, the deadline passed, or too many requests wait; `info` says which.  The result is the caller's
 *  (pulsar_grammar_free).  Holds no lock while it waits; call it with none that the decode loop needs. */
/// @param c          the compiler
/// @param src        the structural tag, JSON text
/// @param deadline_s wall-clock seconds to wait for a slot and the compile
/// @param info       receives what the call met, or NULL
/// @param err        receives the reason on failure
/// @param errlen     size of `err`
/// @return the compiled grammar (caller frees), or NULL
pulsar_grammar *pulsar_grammar_compile(pulsar_grammar_compiler *c, const char *src, double deadline_s,
                                       pulsar_grammar_compile_info *info, char *err, size_t errlen);
/** The process's compile admission counters.
 *  @return a snapshot (taken under the admission lock) */
pulsar_grammar_compile_stats pulsar_grammar_compile_stats_get(void);
/** Free a compiled grammar (pulsar_grammar_compile).
 *  @param g the grammar, or NULL */
void pulsar_grammar_free(pulsar_grammar *g);

/** A matcher at the grammar's start (it keeps its whole history, so any accepted token can be rolled back).
 *  With `end_at_complete` the matcher terminates as soon as the grammar is complete (a region inside a turn: a tool call); without, it
 *  terminates on a stop token (the whole rest of the turn: a response format).  NULL + err on failure. */
/// @param g               the compiled grammar
/// @param end_at_complete terminate once the grammar is complete instead of on a stop token
/// @param err             receives the reason on failure
/// @param errlen          size of `err`
/// @return the matcher (caller frees), or NULL
pulsar_grammar_matcher *pulsar_grammar_matcher_new(const pulsar_grammar *g, bool end_at_complete,
                                                   char *err, size_t errlen);
/** Free a matcher (pulsar_grammar_matcher_new).
 *  @param m the matcher, or NULL */
void pulsar_grammar_matcher_free(pulsar_grammar_matcher *m);
/** Advance by one token.  false = the grammar does not allow it here (the matcher is unchanged) or the matcher
 *  failed (err says which). */
/// @param m      the matcher
/// @param token  the token drawn
/// @param err    receives the reason when the matcher failed
/// @param errlen size of `err`
/// @return true when accepted
bool pulsar_grammar_matcher_accept(pulsar_grammar_matcher *m, int32_t token, char *err, size_t errlen);
/** Advance by raw bytes (a token's tail past a trigger).  false = not allowed here or the matcher failed. */
/// @param m      the matcher
/// @param bytes  the raw bytes
/// @param n      their count
/// @param err    receives the reason when the matcher failed
/// @param errlen size of `err`
/// @return true when accepted
bool pulsar_grammar_matcher_accept_bytes(pulsar_grammar_matcher *m, const char *bytes, size_t n, char *err,
                                         size_t errlen);
/** Undo the last `n` accepts (a byte run counts as one).  false + err on failure. */
/// @param m      the matcher
/// @param n      accepts to undo
/// @param err    receives the reason on failure
/// @param errlen size of `err`
/// @return true on success
bool pulsar_grammar_matcher_rollback(pulsar_grammar_matcher *m, int n, char *err, size_t errlen);
/** Write the next token's mask row (pulsar_grammar_mask_words(vocab) words).  false + err on failure. */
/// @param m      the matcher
/// @param row    receives the mask row
/// @param err    receives the reason on failure
/// @param errlen size of `err`
/// @return true on success
bool pulsar_grammar_matcher_fill(pulsar_grammar_matcher *m, uint32_t *row, char *err, size_t errlen);
/** The grammar has ended (end_at_complete: complete; otherwise a stop token was accepted). */
/// @param m the matcher
/// @return true once the grammar has ended
bool pulsar_grammar_matcher_terminated(const pulsar_grammar_matcher *m);

#endif

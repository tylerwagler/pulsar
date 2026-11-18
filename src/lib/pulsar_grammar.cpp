/** @file
 *  The exception boundary between pulsar and the vendored xgrammar core (built with -fexceptions). */
/* L303: the exception boundary between pulsar and the vendored xgrammar core (pulsar_grammar.h says why it exists
 * and what crosses it).  Built with -fexceptions; every entry catches everything. */
#include "pulsar_grammar.h"
#include "pulsar_grammar_admit.h"

#include <xgrammar/xgrammar.h>

#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>

/* L427: every compile of the process goes through this one admission (pulsar_grammar_admit.h has the rules).
 *
 * THE KEY is (the xgrammar compiler, the structural tag's text), and that is exactly what a cache entry is keyed
 * on: the only compile entry called here is CompileStructuralTag, whose cache key is StructuralTagKey{the JSON
 * text} inside ONE GrammarCompiler (vendor/xgrammar/cpp/grammar_compiler.cc).  Everything else that shapes the
 * result is either inside that text -- constrain.cpp writes strictness, whitespace caps, the tool-parameter style
 * and every schema into the tag -- or fixed when the compiler was built (the vocabulary, the stop ids, RAW
 * decoding), which the compiler's identity covers.  Two requests share a compile only when they would have shared
 * a cache entry.
 *
 * WHAT XGRAMMAR ALREADY DOES.  Its cache stores a shared_future per key: a second thread asking for a key that is
 * being computed blocks on the first thread's future instead of compiling again, and a compile that threw stays in
 * the cache as that exception (ThreadSafeLRUCache: nothing removes a failed entry but LRU eviction), so an
 * identical invalid tag is refused again without a compile.  That dedupes the WORK, not the waiting: before L427
 * every identical request still started a worker that held one of the four slots while it slept on the future, so
 * the fifth request of a fan-out was refused.  The pending set removes those workers; the future and the cached
 * failure are xgrammar's and stay as they were (a job leaves the pending set when it returns, so a retry of a
 * failed tag asks xgrammar again and gets its cached answer). */
/// The process's compile admission: PULSAR_GRAMMAR_COMPILES_MAX slots, PULSAR_GRAMMAR_WAITERS_MAX waiting callers.
static pulsar_grammar_admit<xgrammar::CompiledGrammar> g_admit(PULSAR_GRAMMAR_COMPILES_MAX,
                                                               PULSAR_GRAMMAR_WAITERS_MAX);

/** Opaque compiler handle: one xgrammar compiler over one model vocabulary. */
struct pulsar_grammar_compiler {
    /// The xgrammar compiler; shared so a late compile worker keeps it alive past destroy
    std::shared_ptr<xgrammar::GrammarCompiler> xc;
    /// Vocabulary size the compiler was built for (token ids 0..n_vocab-1)
    int n_vocab = 0;
    /// Per token id: true for the stop tokens given at creation
    std::vector<bool> stop;
};

/** Opaque compiled grammar, ready to build matchers from. */
struct pulsar_grammar {
    /// The xgrammar compiled form
    xgrammar::CompiledGrammar cg;
};

/** Opaque per-request matcher state over a compiled grammar. */
struct pulsar_grammar_matcher {
    /// The xgrammar matcher (accepted-token state)
    xgrammar::GrammarMatcher m;
    /// Vocabulary size of the grammar's tokenizer info; sizes the token bitmask
    int n_vocab;
    /// Terminate as soon as the grammar is complete (see pulsar_grammar_matcher_new)
    bool end_at_complete;
};

/** Write `what` and the exception's message, trimmed of xgrammar's log prefix, as one line into err.
 *  @param err    the caller's error buffer, or NULL (nothing written)
 *  @param errlen size of `err`
 *  @param what   the operation that failed
 *  @param e      the caught exception, or NULL for a non-std one */
static void say(char *err, size_t errlen, const char *what, const std::exception *e) {
    if (!err || !errlen) return;
    if (!e) {
        snprintf(err, errlen, "%s: unknown failure", what);
        return;
    }
    /* xgrammar's messages lead with "[hh:mm:ss] file:line: "; the request only needs what failed */
    const char *msg = e->what();
    const char *body = strstr(msg, ": Check failed: ");
    const char *is_false = body ? strstr(body, " is false: ") : NULL;
    if (is_false) msg = is_false + 11;
    snprintf(err, errlen, "%s: %s", what, msg);
    /* one line: the message's own trailing newline is dropped */
    for (char *p = err; *p; p++)
        if (*p == '\n') *p = ' ';
}

/** The catch clauses every entry ends its try block with: any exception becomes `say` + `return fail`.
 *  @param what the operation, for the message
 *  @param fail the entry's failure return value */
#define GRAMMAR_TRY(what, fail)                                                                                        \
    catch (const std::exception &e) {                                                                                  \
        say(err, errlen, what, &e);                                                                                    \
        return fail;                                                                                                   \
    }                                                                                                                  \
    catch (...) {                                                                                                      \
        say(err, errlen, what, NULL);                                                                                  \
        return fail;                                                                                                   \
    }

pulsar_grammar_compiler *pulsar_grammar_compiler_new(const char *const *tok, const size_t *tok_len, int n_tokens,
                                                     int n_vocab, const int32_t *stop, int n_stop,
                                                     int64_t cache_bytes, char *err, size_t errlen) {
    try {
        std::vector<std::string> vocab((size_t)n_tokens);
        for (int t = 0; t < n_tokens; t++) vocab[(size_t)t].assign(tok[t], tok_len[t]);
        std::vector<int32_t> stops(stop, stop + n_stop);
        /* RAW: the strings ARE the bytes a token decodes to (the engine's pulsar_token_text) */
        xgrammar::TokenizerInfo info(vocab, xgrammar::VocabType::RAW, n_vocab, stops, false);
        auto *c = new pulsar_grammar_compiler();
        c->xc = std::make_shared<xgrammar::GrammarCompiler>(info, 2, true, cache_bytes);
        c->n_vocab = n_vocab;
        c->stop.assign((size_t)n_vocab, false);
        for (int i = 0; i < n_stop; i++)
            if (stop[i] >= 0 && stop[i] < n_vocab) c->stop[(size_t)stop[i]] = true;
        return c;
    }
    GRAMMAR_TRY("grammar vocabulary", NULL)
}

void pulsar_grammar_compiler_free(pulsar_grammar_compiler *c) { delete c; }

int pulsar_grammar_compiler_vocab(const pulsar_grammar_compiler *c) { return c->n_vocab; }

bool pulsar_grammar_compiler_is_stop(const pulsar_grammar_compiler *c, int t) {
    return t >= 0 && t < c->n_vocab && c->stop[(size_t)t];
}

pulsar_grammar *pulsar_grammar_compile(pulsar_grammar_compiler *c, const char *src, double deadline_s,
                                       pulsar_grammar_compile_info *info, char *err, size_t errlen) {
    pulsar_grammar_compile_info met = {PULSAR_GRAMMAR_INVALID, false, 0.0};
    if (info) *info = met;
    try {
        /* the worker's copies: it may outlive this call and the compiler handle (a late compile) */
        std::shared_ptr<xgrammar::GrammarCompiler> xc = c->xc;
        std::string text = src ? src : "";
        auto work = [xc, text](std::string *e) -> std::optional<xgrammar::CompiledGrammar> {
            try {
                return xc->CompileStructuralTag(text);
            } catch (const std::exception &ex) {
                char b[400];
                say(b, sizeof b, "grammar compile", &ex);
                *e = b;
            } catch (...) {
                *e = "grammar compile: unknown failure";
            }
            return std::nullopt;
        };
        std::string reason;
        std::optional<xgrammar::CompiledGrammar> out = g_admit.run(xc.get(), text, work, deadline_s, &met, &reason);
        if (info) *info = met;
        switch (met.end) {
        case PULSAR_GRAMMAR_COMPILED: return new pulsar_grammar{std::move(*out)};
        case PULSAR_GRAMMAR_INVALID: snprintf(err, errlen, "%s", reason.c_str()); break;
        case PULSAR_GRAMMAR_LATE:
            if (met.slot_wait_s > 0.0)
                snprintf(err, errlen, "grammar compile: still running after %.1f s, %.1f s of it waiting for a slot "
                         "(refused; the result is cached)", deadline_s, met.slot_wait_s);
            else
                snprintf(err, errlen, "grammar compile: still running after %.1f s (refused; the result is cached)",
                         deadline_s);
            break;
        case PULSAR_GRAMMAR_NO_SLOT:
            snprintf(err, errlen, "grammar compile: waited %.1f s for one of %d compile slots (refused; retry)",
                     deadline_s, PULSAR_GRAMMAR_COMPILES_MAX);
            break;
        case PULSAR_GRAMMAR_CROWDED:
            snprintf(err, errlen, "grammar compile: %d requests already wait for a compile (refused; retry)",
                     PULSAR_GRAMMAR_WAITERS_MAX);
            break;
        case PULSAR_GRAMMAR_NO_THREAD:
            snprintf(err, errlen, "grammar compile: no worker thread: %s (refused; retry)", reason.c_str());
            break;
        case PULSAR_GRAMMAR_ENDS: break;
        }
        return NULL;
    }
    GRAMMAR_TRY("grammar compile", NULL)
}

pulsar_grammar_compile_stats pulsar_grammar_compile_stats_get(void) { return g_admit.stats(); }

void pulsar_grammar_free(pulsar_grammar *g) { delete g; }

pulsar_grammar_matcher *pulsar_grammar_matcher_new(const pulsar_grammar *g, bool end_at_complete, char *err,
                                                   size_t errlen) {
    try {
        return new pulsar_grammar_matcher{xgrammar::GrammarMatcher(g->cg, std::nullopt, end_at_complete),
                                          g->cg.GetTokenizerInfo().GetVocabSize(), end_at_complete};
    }
    GRAMMAR_TRY("grammar matcher", NULL)
}

void pulsar_grammar_matcher_free(pulsar_grammar_matcher *m) { delete m; }

bool pulsar_grammar_matcher_accept(pulsar_grammar_matcher *m, int32_t token, char *err, size_t errlen) {
    try {
        return m->m.AcceptToken(token);
    }
    GRAMMAR_TRY("grammar accept", false)
}

bool pulsar_grammar_matcher_accept_bytes(pulsar_grammar_matcher *m, const char *bytes, size_t n, char *err,
                                         size_t errlen) {
    try {
        return m->m.AcceptString(std::string(bytes, n));
    }
    GRAMMAR_TRY("grammar accept", false)
}

bool pulsar_grammar_matcher_rollback(pulsar_grammar_matcher *m, int n, char *err, size_t errlen) {
    try {
        m->m.Rollback(n);
        return true;
    }
    GRAMMAR_TRY("grammar rollback", false)
}

bool pulsar_grammar_matcher_fill(pulsar_grammar_matcher *m, uint32_t *row, char *err, size_t errlen) {
    try {
        int64_t shape[2] = {1, (int64_t)pulsar_grammar_mask_words((uint32_t)m->n_vocab)};
        DLTensor t{};
        t.data = row;
        t.device = DLDevice{kDLCPU, 0};
        t.ndim = 2;
        t.dtype = xgrammar::GetBitmaskDLType();
        t.shape = shape;
        m->m.FillNextTokenBitmask(&t, 0);
        return true;
    }
    GRAMMAR_TRY("grammar mask", false)
}

bool pulsar_grammar_matcher_terminated(const pulsar_grammar_matcher *m) {
    /* IsTerminated reads two flags: it cannot throw */
    return m->m.IsTerminated();
}

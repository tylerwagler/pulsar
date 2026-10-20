/* L303: the exception boundary between pulsar and the vendored xgrammar core (pulsar_grammar.h says why it exists
 * and what crosses it).  Built with -fexceptions; every entry catches everything. */
#include "pulsar_grammar.h"

#include <xgrammar/xgrammar.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

/* Workers alive (on time or late): the PULSAR_GRAMMAR_COMPILES_MAX cap. */
static std::atomic<int> g_compiles{0};

struct pulsar_grammar_compiler {
    std::shared_ptr<xgrammar::GrammarCompiler> xc;
    int n_vocab = 0;
    std::vector<bool> stop;
};

struct pulsar_grammar {
    xgrammar::CompiledGrammar cg;
};

struct pulsar_grammar_matcher {
    xgrammar::GrammarMatcher m;
    int n_vocab;
    bool end_at_complete;
};

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

namespace {
/* One compile, shared by the waiting request and the worker (which may outlive the wait). */
struct compile_job {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    std::optional<xgrammar::CompiledGrammar> out;
    std::string err;
};
}  // namespace

pulsar_grammar *pulsar_grammar_compile(pulsar_grammar_compiler *c, const char *src,
                                       double deadline_s, char *err, size_t errlen) {
    try {
        if (g_compiles.fetch_add(1) >= PULSAR_GRAMMAR_COMPILES_MAX) {
            g_compiles.fetch_sub(1);
            snprintf(err, errlen, "grammar compile: %d compiles already running", PULSAR_GRAMMAR_COMPILES_MAX);
            return NULL;
        }
        auto job = std::make_shared<compile_job>();
        std::shared_ptr<xgrammar::GrammarCompiler> xc = c->xc;
        std::string text = src ? src : "";
        std::thread([job, xc, text]() {
            std::optional<xgrammar::CompiledGrammar> out;
            std::string e;
            try {
                out = xc->CompileStructuralTag(text);
            } catch (const std::exception &ex) {
                char b[400];
                say(b, sizeof b, "grammar compile", &ex);
                e = b;
            } catch (...) {
                e = "grammar compile: unknown failure";
            }
            {
                std::lock_guard<std::mutex> lk(job->mu);
                job->out = std::move(out);
                job->err = std::move(e);
                job->done = true;
            }
            job->cv.notify_all();
            g_compiles.fetch_sub(1);
        }).detach();
        std::unique_lock<std::mutex> lk(job->mu);
        if (!job->cv.wait_for(lk, std::chrono::duration<double>(deadline_s), [&] { return job->done; })) {
            snprintf(err, errlen, "grammar compile: still running after %.1f s (refused; the result is cached)",
                     deadline_s);
            return NULL;
        }
        if (!job->out) {
            snprintf(err, errlen, "%s", job->err.c_str());
            return NULL;
        }
        return new pulsar_grammar{std::move(*job->out)};
    }
    GRAMMAR_TRY("grammar compile", NULL)
}

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

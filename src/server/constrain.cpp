/* constrain.cpp -- L303: structured outputs.  A request's response format and its strict or forced tool calls
 * become grammars (xgrammar structural tags, compiled through pulsar_grammar.h), and a per-attempt GATE follows
 * the emitted tokens and masks every draw inside a grammar region.
 *
 * THE GATE.  Unconstrained text costs nothing: until a trigger's bytes go by, no matcher runs and no draw is
 * masked -- the lanes see "not live" and take today's path.  Triggers are byte strings matched with KMP over the
 * tokens' bytes, so they fire whatever tokenization the model chose (DSML's opener is several tokens, Qwen's one;
 * a token that ends past a trigger hands its tail to the region's matcher).  Phases:
 *   THINK  -- inside reasoning (the turn started in it: thinking_state, the stop-scan's authority); watches the
 *             think close, then opens the answer region (a response format) or goes FREE.
 *   FREE   -- watches the call trigger (when calls are armed) and, for a family whose answer has its own
 *             opener (Inkling), the answer's opener.
 *   REGION -- a matcher is live: every draw is masked, every emitted token advances it.  A call region ends
 *             when its grammar is complete (back to FREE: another call may follow); the answer ends on a stop.
 * Every advance is recorded, so a speculative round can walk the gate through its base and drafts to build
 * each verify row's mask and roll it back (gen_constrain_round): row i's mask is exactly the mask a plain lane
 * would apply after the same tokens, which is what keeps spec == plain.
 *
 * L272's tool-name mask is this mechanism's first case: a forced call's seed is fed to the gate as if sampled,
 * so "required" over several tools constrains the name AND its arguments, in every lane (the name mask held
 * speculation off; the gate does not). */
#include "pulsar_server_internal.h"
#include "../lib/pulsar_grammar.h"

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

/* A compile may take this long before the request is refused (the worker finishes into the cache). */
static constexpr double k_compile_deadline_s = 10.0;
/* The compiled-grammar cache's cap. */
static constexpr int64_t k_grammar_cache_bytes = 256LL << 20;
/* A run of whitespace inside JSON is capped: a masked model that keeps choosing a newline would otherwise never
 * leave it (xgrammar's max_whitespace_cnt).  Pretty-printed JSON indents well inside it. */
static constexpr int k_json_ws_max = 24;

/* ==== response formats ==================================================================================== */

void response_format_free(response_format_spec *f) {
    free(f->src);
    f->src = NULL;
    f->kind = RESPONSE_FORMAT_NONE;
}

/* Member `key` of `obj` as a string; NULL when absent or not a string. */
static char *member_string(const char *obj, const char *key) {
    char *raw = json_object_member_raw(obj, key);
    if (!raw) return NULL;
    const char *p = raw;
    char *out = NULL;
    if (!json_string(&p, &out)) out = NULL;
    free(raw);
    return out;
}

bool response_format_parse(const char *raw, response_format_spec *out, char *err, size_t errlen) {
    response_format_free(out);
    const char *p = raw;
    json_ws(&p);
    if (json_lit(&p, "null")) return true;
    if (*p != '{') {
        snprintf(err, errlen, "response_format: not an object");
        return false;
    }
    char *type = member_string(raw, "type");
    if (!type) {
        snprintf(err, errlen, "response_format: no type");
        return false;
    }
    bool ok = true;
    if (!strcmp(type, "text")) {
        out->kind = RESPONSE_FORMAT_NONE;
    } else if (!strcmp(type, "json_object")) {
        out->kind = RESPONSE_FORMAT_JSON_OBJECT;
    } else if (!strcmp(type, "json_schema")) {
        /* chat: {"json_schema": {"name", "schema", "strict"}}; Responses and Anthropic: "schema" alongside type */
        char *js = json_object_member_raw(raw, "json_schema");
        char *schema = json_object_member_raw(js ? js : raw, "schema");
        free(js);
        if (!schema) {
            snprintf(err, errlen, "response_format json_schema: no schema");
            ok = false;
        } else {
            out->kind = RESPONSE_FORMAT_JSON_SCHEMA;
            out->src = schema;
        }
    } else if (!strcmp(type, "regex")) {
        out->src = member_string(raw, "pattern");
        out->kind = RESPONSE_FORMAT_REGEX;
        if (!out->src || !out->src[0]) {
            snprintf(err, errlen, "response_format regex: no pattern");
            ok = false;
        }
    } else if (!strcmp(type, "grammar")) {
        out->src = member_string(raw, "grammar");
        out->kind = RESPONSE_FORMAT_GRAMMAR;
        if (!out->src || !out->src[0]) {
            snprintf(err, errlen, "response_format grammar: no grammar");
            ok = false;
        }
    } else if (!strcmp(type, "structural_tag")) {
        out->src = json_object_member_raw(raw, "format");
        out->kind = RESPONSE_FORMAT_STRUCTURAL_TAG;
        if (!out->src) {
            snprintf(err, errlen, "response_format structural_tag: no format");
            ok = false;
        }
    } else {
        snprintf(err, errlen, "response_format: type \"%.40s\" is not one of text, json_object, json_schema, regex, "
                 "grammar, structural_tag", type);
        ok = false;
    }
    free(type);
    if (!ok) response_format_free(out);
    return ok;
}

/* ==== the plan: the request's grammars ==================================================================== */

struct constrain_plan {
    pulsar_grammar *answer = nullptr;   /* the answer region (a response format): ends on a stop token */
    pulsar_grammar *calls = nullptr;    /* one call region, opened by the call trigger in free text: ends when complete */
    const server_grammar_shape *shape = nullptr;
};

void constrain_plan_free(constrain_plan *p) {
    if (!p) return;
    pulsar_grammar_free(p->answer);
    pulsar_grammar_free(p->calls);
    delete p;
}

/* The grammar compiler over the loaded vocabulary, built once (the first constrained request pays for it). */
static pulsar_grammar_compiler *constrain_compiler(server *s, pulsar_engine *e, char *err, size_t errlen) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    if (s->grammar) return s->grammar;
    const auto t0 = std::chrono::steady_clock::now();
    const int width = pulsar_engine_logits_width(e);
    auto *bytes = new std::vector<std::string>((size_t)width);
    std::vector<const char *> ptrs((size_t)width);
    std::vector<size_t> lens((size_t)width);
    std::vector<int32_t> stops;
    for (int t = 0; t < width; t++) {
        size_t n = 0;
        char *b = pulsar_token_text(e, t, &n);
        if (b) (*bytes)[(size_t)t].assign(b, n);
        free(b);
        ptrs[(size_t)t] = (*bytes)[(size_t)t].data();
        lens[(size_t)t] = (*bytes)[(size_t)t].size();
        if (pulsar_token_is_stop(e, t)) stops.push_back(t);
    }
    pulsar_grammar_compiler *c = pulsar_grammar_compiler_new(ptrs.data(), lens.data(), width, width, stops.data(),
                                                             (int)stops.size(), k_grammar_cache_bytes, err, errlen);
    if (!c) {
        delete bytes;
        return NULL;
    }
    s->token_bytes = bytes;
    s->grammar = c;
    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: grammar vocabulary: %d tokens, %zu stop, built in %.0f ms (L303)",
               width, stops.size(),
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return c;
}

/* One declared tool: its name and its arguments' schema (a strict tool's declared one, another's `true`). */
struct constrain_tool {
    std::string name;
    std::string schema;
};

/* The conversation's tools (conversation_tool_schema_lines: one function schema a line).  `any_strict`: some
 * tool says "strict": true. */
static bool constrain_tools(const chat_conversation *c, std::vector<constrain_tool> *out, bool *any_strict,
                            char *err, size_t errlen) {
    buf lines = {0};
    conversation_tool_schema_lines(c, &lines);
    *any_strict = false;
    bool ok = true;
    for (size_t at = 0; ok && at < lines.len;) {
        const char *nl = (const char *)memchr(lines.ptr + at, '\n', lines.len - at);
        const size_t end = nl ? (size_t)(nl - lines.ptr) : lines.len;
        std::string line(lines.ptr + at, end - at);
        at = end + 1;
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        char *name = member_string(line.c_str(), "name");
        if (!name) {
            snprintf(err, errlen, "tools: a tool without a name cannot be constrained");
            ok = false;
            break;
        }
        char *params = json_object_member_raw(line.c_str(), "parameters");
        if (!params) params = json_object_member_raw(line.c_str(), "input_schema");
        char *strict_raw = json_object_member_raw(line.c_str(), "strict");
        const char *sp = strict_raw;
        bool strict = false;
        if (sp && !json_bool(&sp, &strict)) strict = false;
        if (strict) *any_strict = true;
        out->push_back({name, strict && params ? params : "true"});
        free(name);
        free(params);
        free(strict_raw);
    }
    buf_free(&lines);
    return ok;
}

static void j_const(buf *b, const char *s) {
    buf_puts(b, "{\"type\":\"const_string\",\"value\":");
    json_escape(b, s);
    buf_putc(b, '}');
}

/* One call region: call_trigger block_head (name_open NAME name_close ARGS call_close)[+] block_tail. */
static void j_call_region(buf *b, const server_grammar_shape *sh, const std::vector<constrain_tool> &tools) {
    buf_puts(b, "{\"type\":\"sequence\",\"elements\":[");
    {
        std::string head = std::string(sh->call_trigger) + sh->block_head;
        j_const(b, head.c_str());
        buf_putc(b, ',');
    }
    if (sh->repeat) buf_puts(b, "{\"type\":\"plus\",\"content\":");
    if (tools.size() > 1) buf_puts(b, "{\"type\":\"or\",\"elements\":[");
    for (size_t i = 0; i < tools.size(); i++) {
        if (i) buf_putc(b, ',');
        buf_puts(b, "{\"type\":\"sequence\",\"elements\":[");
        std::string open = std::string(sh->name_open) + tools[i].name + sh->name_close;
        j_const(b, open.c_str());
        buf_puts(b, ",{\"type\":\"json_schema\",\"json_schema\":");
        buf_puts(b, tools[i].schema.c_str());
        buf_puts(b, ",\"style\":");
        json_escape(b, sh->style);
        buf_printf(b, ",\"max_whitespace_cnt\":%d},", k_json_ws_max);
        j_const(b, sh->call_close);
        buf_puts(b, "]}");
    }
    if (tools.size() > 1) buf_puts(b, "]}");
    if (sh->repeat) buf_putc(b, '}');
    if (sh->block_tail[0]) {
        buf_putc(b, ',');
        j_const(b, sh->block_tail);
    }
    buf_puts(b, "]}");
}

/* The response format as one format object. */
static void j_answer_format(buf *b, const response_format_spec *f) {
    switch (f->kind) {
    case RESPONSE_FORMAT_JSON_OBJECT:
        buf_printf(b, "{\"type\":\"json_schema\",\"json_schema\":{\"type\":\"object\",\"additionalProperties\":true},"
                      "\"max_whitespace_cnt\":%d}", k_json_ws_max);
        return;
    case RESPONSE_FORMAT_JSON_SCHEMA:
        buf_puts(b, "{\"type\":\"json_schema\",\"json_schema\":");
        buf_puts(b, f->src);
        buf_printf(b, ",\"max_whitespace_cnt\":%d}", k_json_ws_max);
        return;
    case RESPONSE_FORMAT_REGEX:
        buf_puts(b, "{\"type\":\"regex\",\"pattern\":");
        json_escape(b, f->src);
        buf_putc(b, '}');
        return;
    case RESPONSE_FORMAT_GRAMMAR:
        buf_puts(b, "{\"type\":\"grammar\",\"grammar\":");
        json_escape(b, f->src);
        buf_putc(b, '}');
        return;
    case RESPONSE_FORMAT_STRUCTURAL_TAG:
        buf_puts(b, f->src);
        return;
    case RESPONSE_FORMAT_NONE:
        break;
    }
    pulsar_die("j_answer_format: no format");
}

/* Leading whitespace before an answer (Qwen writes "\n\n" after </think>). */
static void j_ws(buf *b) { buf_puts(b, "{\"type\":\"regex\",\"pattern\":\"[ \\\\t\\\\r\\\\n]*\"}"); }

static pulsar_grammar *constrain_compile(pulsar_grammar_compiler *c, buf *format, const char *what, char *err,
                                         size_t errlen) {
    buf tag = {0};
    buf_puts(&tag, "{\"type\":\"structural_tag\",\"format\":");
    buf_append(&tag, format->ptr, format->len);
    buf_putc(&tag, '}');
    char cerr[400];
    pulsar_grammar *g = pulsar_grammar_compile(c, tag.ptr, k_compile_deadline_s, cerr, sizeof cerr);
    buf_free(&tag);
    if (!g) snprintf(err, errlen, "%s: %s", what, cerr);
    return g;
}

bool constrain_plan_build(pulsar_engine *e, server *s, const chat_conversation *c, request *r, char *err,
                          size_t errlen) {
    constrain_plan_free(r->constrain);
    r->constrain = NULL;
    if (r->kind != REQ_CHAT) return true;
    std::vector<constrain_tool> tools;
    bool any_strict = false;
    if (r->has_tools && !constrain_tools(c, &tools, &any_strict, err, errlen)) return false;
    const bool forced = r->force_tool_call && !tools.empty();
    const bool want_calls = !tools.empty() && (forced || any_strict);
    /* a forced turn is its call: a response format would describe a turn that cannot happen */
    const bool want_answer = c->response_format.kind != RESPONSE_FORMAT_NONE && !forced;
    if (!want_calls && !want_answer) return true;
    const server_grammar_shape *sh = r->family->grammar;
    if (!sh) {
        snprintf(err, errlen, "%s: the %s family declares no grammar shape (L303)",
                 want_answer ? "response_format" : "strict tools", r->family->name);
        return false;
    }
    if (e && pulsar_engine_is_tp(e)) {
        snprintf(err, errlen, "%s: constrained decoding does not cross a tensor-parallel pair (L303)",
                 want_answer ? "response_format" : "strict tools");
        return false;
    }
    /* the parse-without-engine shape (the renderer gate) compiles nothing; a server with a compiler but no
     * engine is the unit tests' (a toy vocabulary) */
    if (!s || (!e && !s->grammar)) return true;
    pulsar_grammar_compiler *comp = s->grammar ? s->grammar : constrain_compiler(s, e, err, errlen);
    if (!comp) return false;
    auto *plan = new constrain_plan();
    plan->shape = sh;
    if (want_calls) {
        buf f = {0};
        j_call_region(&f, sh, tools);
        plan->calls = constrain_compile(comp, &f, "tool calls", err, errlen);
        buf_free(&f);
        if (!plan->calls) {
            constrain_plan_free(plan);
            return false;
        }
    }
    if (want_answer) {
        /* whitespace, then the format and the family's answer close -- or, with tools on offer, the turn's calls
         * instead (a turn may answer with calls; each may follow whitespace) */
        buf f = {0};
        buf_puts(&f, "{\"type\":\"sequence\",\"elements\":[");
        j_ws(&f);
        buf_puts(&f, ",");
        const bool or_calls = !tools.empty() && c->tool_choice != CHAT_TOOL_CHOICE_NONE;
        if (or_calls) buf_puts(&f, "{\"type\":\"or\",\"elements\":[");
        buf_puts(&f, "{\"type\":\"sequence\",\"elements\":[");
        j_answer_format(&f, &c->response_format);
        if (sh->answer_close[0]) {
            buf_putc(&f, ',');
            j_const(&f, sh->answer_close);
        }
        buf_puts(&f, "]}");
        if (or_calls) {
            buf_puts(&f, ",{\"type\":\"plus\",\"content\":{\"type\":\"sequence\",\"elements\":[");
            j_ws(&f);
            buf_putc(&f, ',');
            j_call_region(&f, sh, tools);
            buf_puts(&f, "]}}]}");
        }
        buf_puts(&f, "]}");
        plan->answer = constrain_compile(comp, &f, "response_format", err, errlen);
        buf_free(&f);
        if (!plan->answer) {
            constrain_plan_free(plan);
            return false;
        }
    }
    r->constrain = plan;
    return true;
}

/* ==== the gate ============================================================================================ */

/* One byte pattern under KMP. */
struct constrain_kmp {
    std::string pat;
    std::vector<int> fail;
    void init(const char *p) {
        pat = p ? p : "";
        fail.assign(pat.size(), 0);
        for (size_t i = 1, k = 0; i < pat.size(); i++) {
            while (k && pat[i] != pat[k]) k = (size_t)fail[k - 1];
            if (pat[i] == pat[k]) k++;
            fail[i] = (int)k;
        }
    }
    /* Advance state `st` by byte `c`; true = the pattern just completed (st is reset). */
    bool step(int &st, char c) const {
        if (pat.empty()) return false;
        while (st && pat[(size_t)st] != c) st = fail[(size_t)st - 1];
        if (pat[(size_t)st] == c) st++;
        if ((size_t)st == pat.size()) {
            st = 0;
            return true;
        }
        return false;
    }
};

enum constrain_phase : uint8_t { GATE_THINK, GATE_FREE, GATE_REGION };

/* What one advance changed, to undo it. */
struct constrain_rec {
    constrain_phase phase;
    int kmp_think, kmp_call, kmp_answer;
    int cur;          /* the live matcher's index before */
    int ops;          /* accepts made into matcher `cur_after` (an existing one) that a rollback undoes */
    int cur_after;
    bool opened;      /* this advance created a matcher (a rollback frees it; its accepts go with it) */
};

struct constrain_gate {
    const constrain_plan *plan;
    const std::vector<std::string> *bytes;
    uint32_t words;
    constrain_phase phase = GATE_FREE;
    constrain_kmp think, call, answer;   /* the reasoning close, the call trigger, a separate answer opener */
    int kmp_think = 0, kmp_call = 0, kmp_answer = 0;
    std::vector<pulsar_grammar_matcher *> matchers;   /* every region's, in order; the live one is `cur` */
    int cur = -1;
    std::vector<constrain_rec> hist;
    std::vector<uint32_t> rows;   /* a round's masks */
    std::vector<uint32_t> one;    /* a single draw's mask */
    std::vector<float> scratch;   /* a plain draw's masked copy of its row */
    char err[200];
    ~constrain_gate() {
        for (auto *m : matchers) pulsar_grammar_matcher_free(m);
    }
};

static bool gate_open(constrain_gate *gt, const pulsar_grammar *gr, bool end_at_complete, const char *bytes,
                      size_t n, constrain_rec *rec) {
    pulsar_grammar_matcher *m = pulsar_grammar_matcher_new(gr, end_at_complete, gt->err, sizeof gt->err);
    if (!m) return false;
    gt->matchers.push_back(m);
    gt->cur = (int)gt->matchers.size() - 1;
    gt->phase = GATE_REGION;
    rec->opened = true;
    if (n && !pulsar_grammar_matcher_accept_bytes(m, bytes, n, gt->err, sizeof gt->err)) {
        if (!gt->err[0]) snprintf(gt->err, sizeof gt->err, "grammar: the region's opening bytes are refused");
        return false;
    }
    return true;
}

/* A region whose matcher terminated hands the turn back to free text: only a call region can (it ends when
 * complete; the answer ends on a stop token, which never reaches the gate). */
static void gate_region_done(constrain_gate *gt) {
    if (gt->phase == GATE_REGION && pulsar_grammar_matcher_terminated(gt->matchers[(size_t)gt->cur])) {
        gt->phase = GATE_FREE;
        gt->kmp_call = gt->kmp_answer = 0;
    }
}

/* Advance by one token (`token` >= 0) or by raw bytes (the seed, token -1).  false = gt->err. */
static bool gate_feed(constrain_gate *gt, int token, const char *b, size_t n) {
    constrain_rec rec = {gt->phase, gt->kmp_think, gt->kmp_call, gt->kmp_answer, gt->cur, 0, -1, false};
    const constrain_plan *p = gt->plan;
    gt->err[0] = '\0';
    bool ok = true;
    for (size_t i = 0; ok && i < n;) {
        if (gt->phase == GATE_REGION) {
            pulsar_grammar_matcher *m = gt->matchers[(size_t)gt->cur];
            const bool whole = token >= 0 && i == 0;
            const bool acc = whole ? pulsar_grammar_matcher_accept(m, token, gt->err, sizeof gt->err)
                                   : pulsar_grammar_matcher_accept_bytes(m, b + i, n - i, gt->err, sizeof gt->err);
            if (!acc) {
                if (!gt->err[0])
                    snprintf(gt->err, sizeof gt->err, "grammar: token %d (\"%.*s\") is outside the grammar", token,
                             (int)(n < 40 ? n : 40), b);
                ok = false;
                break;
            }
            rec.ops++;
            rec.cur_after = gt->cur;
            gate_region_done(gt);
            break;
        }
        const char c = b[i++];
        if (gt->phase == GATE_THINK) {
            if (!gt->think.step(gt->kmp_think, c)) continue;
            /* reasoning closed: the answer opens on the rest of the token, or the turn is free text */
            if (!p->answer) {
                gt->phase = GATE_FREE;
                continue;
            }
            ok = gate_open(gt, p->answer, false, b + i, n - i, &rec);
            break;
        }
        /* FREE */
        if (p->calls && gt->call.step(gt->kmp_call, c)) {
            std::string open = gt->call.pat;
            open.append(b + i, n - i);
            ok = gate_open(gt, p->calls, true, open.data(), open.size(), &rec);
            if (ok) gate_region_done(gt);
            break;
        }
        if (p->answer && !p->shape->answer_at_think_close && gt->answer.step(gt->kmp_answer, c)) {
            ok = gate_open(gt, p->answer, false, b + i, n - i, &rec);
            break;
        }
    }
    if (!ok) {
        /* a refused advance leaves the gate as it was: a refused accept changed nothing, and a region this
         * advance opened goes */
        if (rec.opened) {
            pulsar_grammar_matcher_free(gt->matchers.back());
            gt->matchers.pop_back();
        }
        gt->phase = rec.phase;
        gt->kmp_think = rec.kmp_think;
        gt->kmp_call = rec.kmp_call;
        gt->kmp_answer = rec.kmp_answer;
        gt->cur = rec.cur;
        return false;
    }
    gt->hist.push_back(rec);
    return true;
}

static bool gate_rollback(constrain_gate *gt, int n) {
    for (; n > 0; n--) {
        const constrain_rec rec = gt->hist.back();
        gt->hist.pop_back();
        if (rec.opened) {
            pulsar_grammar_matcher_free(gt->matchers.back());
            gt->matchers.pop_back();
        } else if (rec.ops &&
                   !pulsar_grammar_matcher_rollback(gt->matchers[(size_t)rec.cur_after], rec.ops, gt->err,
                                                    sizeof gt->err)) {
            return false;
        }
        gt->phase = rec.phase;
        gt->kmp_think = rec.kmp_think;
        gt->kmp_call = rec.kmp_call;
        gt->kmp_answer = rec.kmp_answer;
        gt->cur = rec.cur;
    }
    return true;
}

static bool gate_fill(constrain_gate *gt, uint32_t *row) {
    return pulsar_grammar_matcher_fill(gt->matchers[(size_t)gt->cur], row, gt->err, sizeof gt->err);
}

static void gate_fail(gen_state *g, const char *what) {
    snprintf(g->err, sizeof g->err, "%s: %s", what, g->gate->err);
}

bool gen_constrain_begin(server *s, gen_state *g) {
    gen_constrain_end(g);
    const request *r = &g->j->req;
    if (!r->constrain) return true;
    auto *gt = new constrain_gate();
    gt->plan = r->constrain;
    gt->bytes = s->token_bytes;
    gt->words = gen_constrain_words(s);
    const server_grammar_shape *sh = gt->plan->shape;
    gt->think.init(sh->answer_at_think_close ? sh->answer_open : "</think>");
    gt->call.init(sh->call_trigger);
    gt->answer.init(sh->answer_at_think_close ? "" : sh->answer_open);
    g->gate = gt;
    /* where the turn starts: inside reasoning (the prompt left a think block open: the stop scan's authority),
     * in the answer when it opens at the think close, else free */
    if (g->thinking.inside) {
        gt->phase = GATE_THINK;
    } else if (gt->plan->answer && sh->answer_at_think_close) {
        constrain_rec rec = {};
        if (!gate_open(gt, gt->plan->answer, false, NULL, 0, &rec)) {
            gate_fail(g, "response_format");
            return false;
        }
    } else {
        gt->phase = GATE_FREE;
    }
    /* a forced call's seed (g->text so far) goes through as if sampled; the turn's own history starts after */
    if (g->text.len && !gate_feed(gt, -1, g->text.ptr, g->text.len)) {
        gate_fail(g, "tool call grammar");
        return false;
    }
    gt->hist.clear();
    return true;
}

void gen_constrain_end(gen_state *g) {
    delete g->gate;
    g->gate = NULL;
}

bool gen_constrain_live(const gen_state *g) { return g->gate && g->gate->phase == GATE_REGION; }

uint32_t gen_constrain_words(const server *s) {
    return pulsar_grammar_mask_words((uint32_t)pulsar_grammar_compiler_vocab(s->grammar));
}

bool gen_constrain_mask(server *, gen_state *g, uint32_t *row) {
    if (!gate_fill(g->gate, row)) {
        gate_fail(g, "grammar mask");
        return false;
    }
    return true;
}

int gen_constrain_draw(server *s, gen_state *g, const float *logits, int width, float temp, int top_k, float top_p,
                       float min_p) {
    const float *row = logits;
    if (gen_constrain_live(g)) {
        constrain_gate *gt = g->gate;
        gt->one.resize(gt->words);
        if (!gen_constrain_mask(s, g, gt->one.data())) return -1;
        gt->scratch.assign(logits, logits + width);
        pulsar_token_mask_apply(gt->scratch.data(), (uint32_t)width, gt->one.data());
        row = gt->scratch.data();
    }
    const int tok = pulsar_sample_logits(row, width, temp, top_k, top_p, min_p, &g->rng);
    if (tok < 0) snprintf(g->err, sizeof g->err, "sampler refused a degenerate logits row (L188)");
    return tok;
}

bool gen_constrain_accept(server *s, gen_state *g, int token) {
    constrain_gate *gt = g->gate;
    if (!gt) return true;
    const std::string &b = (*s->token_bytes)[(size_t)token];
    if (!gate_feed(gt, token, b.data(), b.size())) {
        gate_fail(g, "grammar");
        return false;
    }
    /* an emitted token is never rolled back: only a round's walk is (gen_constrain_round) */
    gt->hist.clear();
    return true;
}

const uint32_t *gen_constrain_base_mask(server *s, gen_state *g) {
    if (!gen_constrain_live(g)) return NULL;
    constrain_gate *gt = g->gate;
    gt->one.resize(gt->words);
    return gen_constrain_mask(s, g, gt->one.data()) ? gt->one.data() : NULL;
}

static bool mask_bit(const uint32_t *row, int t) { return (row[(uint32_t)t >> 5] >> ((uint32_t)t & 31u)) & 1u; }

int gen_constrain_round(server *s, gen_state *g, const int *toks, int n, bool base_emitted, const uint32_t **masks) {
    constrain_gate *gt = g->gate;
    *masks = NULL;
    if (!gt) return 0;
    const int off = base_emitted ? 0 : 1;
    const int rows = n - off + 1;
    const uint32_t W = gt->words;
    gt->rows.resize((size_t)rows * W);
    int fed = 0;
    bool any_live = false, ok = true;
    if (!base_emitted) {
        const std::string &b = (*s->token_bytes)[(size_t)toks[0]];
        ok = gate_feed(gt, toks[0], b.data(), b.size());
        fed += ok;
    }
    int i = 0;
    for (; ok && i < rows; i++) {
        uint32_t *row = gt->rows.data() + (size_t)i * W;
        const bool live = gt->phase == GATE_REGION;
        if (live) {
            ok = gate_fill(gt, row);
            any_live = true;
        } else {
            memset(row, 0xff, (size_t)W * sizeof(uint32_t));
        }
        if (!ok || i == rows - 1) break;
        const int t = toks[off + i];
        /* the walk never reads past a draft its row refuses, nor past a stop (the round clamps there) */
        if ((live && !mask_bit(row, t)) || pulsar_grammar_compiler_is_stop(s->grammar, t)) break;
        const std::string &b = (*s->token_bytes)[(size_t)t];
        ok = gate_feed(gt, t, b.data(), b.size());
        fed += ok;
    }
    for (int k = i + 1; ok && k < rows; k++) memset(gt->rows.data() + (size_t)k * W, 0xff, (size_t)W * sizeof(uint32_t));
    if (!gate_rollback(gt, fed) || !ok) {
        gate_fail(g, "grammar round");
        return -1;
    }
    if (!any_live) return 0;
    *masks = gt->rows.data();
    return rows;
}

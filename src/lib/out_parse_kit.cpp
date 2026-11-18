/** @file
 *  The output-parser kit's implementation and the shared JSON {"name", "args"} call body. */
/* out_parse_kit.cpp -- the output-parser kit.  See out_parse_kit.h. */
#include "out_parse_kit.h"

#include <stdio.h>
#include <string.h>

#include <algorithm>

#include "pulsar_utf8.h"
#include "pyjson.h"
#include "xml_call_body.h"

namespace {

/* Length of the longest suffix of s that is a proper prefix of `marker`: the
 * bytes that must wait for the next chunk before they can be called text. */
/// @param s      the held bytes
/// @param marker a marker the current state could take
/// @return the length of the longest suffix of `s` that is a proper prefix of `marker` (0 if none)
size_t marker_tail(const std::string &s, const std::string &marker) {
    const size_t m = marker.size();
    for (size_t k = std::min(s.size(), m - 1); k > 0; k--)
        if (!s.compare(s.size() - k, k, marker, 0, k)) return k;
    return 0;
}

/* The end of the whole UTF-8 sequences in b[lo, hi): hi, or the start of a sequence hi cuts. */
/// @param b  the bytes
/// @param lo the start of the range (a lead byte is not looked for before it)
/// @param hi one past the range's end
/// @return `hi`, or the offset of the incomplete sequence `hi` cuts
size_t utf8_whole_end(const std::string &b, size_t lo, size_t hi) {
    size_t k = hi;
    while (k > lo && ((unsigned char)b[k - 1] & 0xc0) == 0x80) k--;
    if (k == lo) return hi;   /* no lead byte in reach: not ours to judge */
    const int want = utf8_seq_len((unsigned char)b[k - 1]);
    return want > 1 && (k - 1) + (size_t)want > hi ? k - 1 : hi;
}

}  // namespace

/* ---- the parser ---------------------------------------------------------------------------- */

void out_parser::init(const out_grammar *g, out_call_body *body, out_state initial) {
    g_ = g;
    body_ = body;
    state_ = initial;
    hold_.clear();
    fed_ = 0;
    header_.clear();
    header_at_ = 0;
    block_seen_ = false;
    raw_lo_ = raw_hi_ = call_at_ = 0;
    code_.reset();
    sec_[0] = sec_[1] = section();
    block_.clear();
    index_ = errors_ = 0;
    reasoning_.clear();
    content_.clear();
    calls_.clear();
    prefix_held_.clear();
    prefix_done_ = !g->reasoning_prefix;
}

void out_parser::send(bool reasoning, std::string &&text, std::vector<out_event> *ev) {
    if (text.empty()) return;
    if (reasoning && !prefix_done_) {   /* L338: the grammar's reasoning label, dropped once */
        prefix_held_ += text;
        const std::string pre = g_->reasoning_prefix;
        if (prefix_held_.size() < pre.size() && !pre.compare(0, prefix_held_.size(), prefix_held_)) return;
        prefix_done_ = true;
        text = !prefix_held_.compare(0, pre.size(), pre) ? prefix_held_.substr(pre.size()) : std::move(prefix_held_);
        prefix_held_.clear();
        if (text.empty()) return;
    }
    (reasoning ? reasoning_ : content_) += text;
    out_event e;
    e.kind = reasoning ? out_event::REASONING : out_event::CONTENT;
    e.text = std::move(text);
    ev->push_back(std::move(e));
}

/* A text state's bytes under its section's policy.  Works on whole codepoints; a split sequence waits; an
 * invalid byte is text.  HEADER buffers. */
void out_parser::emit_text(out_state st, const char *p, size_t n, std::vector<out_event> *ev) {
    if (st == OUT_HEADER) {
        header_.append(p, n);
        return;
    }
    const bool reasoning = st == OUT_REASONING;
    const out_text_policy policy = reasoning ? g_->reasoning : g_->content;
    section &s = sec_[reasoning ? 1 : 0];
    if (st == state_) code_.feed(p, n);   /* L419: the section's bytes as sampled (not a header's text sent as content) */
    std::string buf = s.partial + std::string(p, n);
    s.partial.clear();
    std::string out;
    const unsigned char *u = (const unsigned char *)buf.data();
    for (size_t i = 0; i < buf.size();) {
        int k = utf8_seq_ok(u + i, buf.size() - i);
        bool space = k && py_isspace(utf8_cp_at(u + i, k));
        if (!k) {
            /* an incomplete sequence at the end waits; an invalid byte is text */
            const int want = utf8_seq_len(u[i]);
            if (want > 1 && i + (size_t)want > buf.size()) {
                bool prefix_ok = true;
                for (size_t j = i + 1; j < buf.size(); j++) prefix_ok &= (u[j] & 0xc0) == 0x80;
                if (prefix_ok) { s.partial.assign(buf, i, std::string::npos); break; }
            }
            k = 1;
            space = false;
        }
        switch (policy) {
        case OUT_TEXT_STRIP:
            if (space) {
                if (s.started) s.ws.append(buf, i, (size_t)k);
            } else {
                out += s.ws;
                s.ws.clear();
                out.append(buf, i, (size_t)k);
                s.started = true;
            }
            break;
        case OUT_TEXT_RSTRIP:
            if (space) {
                s.ws.append(buf, i, (size_t)k);
            } else {
                out += s.ws;
                s.ws.clear();
                out.append(buf, i, (size_t)k);
            }
            break;
        case OUT_TEXT_KEEP_UNLESS_BLANK:
            if (s.started) {
                out.append(buf, i, (size_t)k);
            } else {
                s.ws.append(buf, i, (size_t)k);
                if (!space) {
                    s.started = true;
                    out += s.ws;
                    s.ws.clear();
                }
            }
            break;
        case OUT_TEXT_TRIM_AT_CALLS:
            if (space) {
                if (s.started || !block_seen_) s.ws.append(buf, i, (size_t)k);   /* after a call: leading dropped */
            } else {
                out += s.ws;
                s.ws.clear();
                out.append(buf, i, (size_t)k);
                s.started = true;
            }
            break;
        }
        i += (size_t)k;
    }
    send(reasoning, std::move(out), ev);
}

/* A section ends (its state is left for another).  A dangling partial sequence is not text either side can
 * own.  STRIP starts the next section afresh; the other policies run over the whole turn. */
void out_parser::end_section(out_state st) {
    if (st != OUT_CONTENT && st != OUT_REASONING) return;
    const bool reasoning = st == OUT_REASONING;
    section &s = sec_[reasoning ? 1 : 0];
    s.partial.clear();
    if ((reasoning ? g_->reasoning : g_->content) == OUT_TEXT_STRIP) {
        s.ws.clear();
        s.started = false;
    }
}

/* The turn ended: what a policy still holds. */
void out_parser::finish_section(out_state st, std::vector<out_event> *ev) {
    const bool reasoning = st == OUT_REASONING;
    section &s = sec_[reasoning ? 1 : 0];
    s.partial.clear();
    const out_text_policy policy = reasoning ? g_->reasoning : g_->content;
    if (policy == OUT_TEXT_KEEP_UNLESS_BLANK && !s.started && calls_.empty()) send(reasoning, std::move(s.ws), ev);
    if (policy == OUT_TEXT_TRIM_AT_CALLS && !block_seen_) send(reasoning, std::move(s.ws), ev);
    s.ws.clear();
}

size_t out_parser::find_marker(const out_transition **tr, size_t *len, size_t *keep) const {
    size_t best = std::string::npos;
    *tr = nullptr;
    *len = 0;
    *keep = 0;
    auto consider = [&](const std::string &m, const out_transition *t) {
        const size_t at = hold_.find(m);
        if (at != std::string::npos && (at < best || (at == best && m.size() > *len))) {
            best = at;
            *tr = t;
            *len = m.size();
        }
        *keep = std::max(*keep, marker_tail(hold_, m));
    };
    for (const out_transition &t : g_->transitions)
        if (t.from == state_) consider(g_->markers[(size_t)t.marker], &t);
    for (const std::string &m : g_->dropped) consider(m, nullptr);
    return best;
}

bool out_parser::quoted(const out_transition &tr) const {
    if (state_ != OUT_CONTENT && state_ != OUT_REASONING) return false;
    return (tr.to == OUT_CALL || (tr.act & OUT_ACT_CODE_TEXT)) && code_.in_code();
}

void out_parser::close_call(std::vector<out_event> *ev) {
    out_call call;
    const char *why = "";
    size_t at = 0;
    if (body_->close(block_, index_, ev, &call, &why, &at)) {
        out_event e;
        e.kind = out_event::TOOL_END;
        e.index = index_;
        e.name = call.name;
        e.arguments = call.arguments;
        ev->push_back(std::move(e));
        calls_.push_back(std::move(call));
    } else {
        out_event e;
        e.kind = out_event::ERROR;
        char msg[200];
        snprintf(msg, sizeof msg, "malformed tool call %d: %s at byte %zu of the block", index_, why, at);
        e.text = msg;
        e.index = index_;
        ev->push_back(std::move(e));
        errors_++;
    }
    index_++;
}

void out_parser::feed(const char *p, size_t n, std::vector<out_event> *ev) {
    hold_.append(p, n);
    fed_ += n;
    for (;;) {
        const out_transition *tr = nullptr;
        size_t len = 0, keep = 0;
        const size_t at = find_marker(&tr, &len, &keep);
        if (at != std::string::npos && !tr) {   /* a dropped marker: the bytes around it join */
            if (state_ == OUT_CALL) block_.append(hold_, 0, at);
            else emit_text(state_, hold_.data(), at, ev);
            if (state_ == OUT_CONTENT || state_ == OUT_REASONING) code_.feed(hold_.data() + at, len);
            hold_.erase(0, at + len);
            continue;
        }
        if (state_ == OUT_CALL) {
            const size_t take = at == std::string::npos ? hold_.size() - keep : at;
            block_.append(hold_, 0, take);
            if (at == std::string::npos) {
                hold_.erase(0, take);
                body_->grow(block_, hold_.size(), false, index_, ev);
                return;
            }
            if (body_->holds_closer(block_)) {   /* L410: the marker is a value's text, the call is still open */
                block_.append(hold_, at, len);
                hold_.erase(0, at + len);
                continue;
            }
            raw_hi_ = fed_ - hold_.size() + at + len;
            const size_t marker_at = fed_ - hold_.size() + at;
            hold_.erase(0, at + len);
            body_->grow(block_, 0, true, index_, ev);
            close_call(ev);
            if (tr->to == OUT_HEADER) header_at_ = marker_at + len;
            state_ = tr->to;
            continue;
        }
        if (at == std::string::npos) {
            emit_text(state_, hold_.data(), hold_.size() - keep, ev);
            hold_.erase(0, hold_.size() - keep);
            return;
        }
        emit_text(state_, hold_.data(), at, ev);
        if (quoted(*tr)) {   /* L419: quoted in Markdown code, the marker is the section's text */
            emit_text(state_, hold_.data() + at, len, ev);
            hold_.erase(0, at + len);
            continue;
        }
        const size_t marker_at = fed_ - hold_.size() + at;
        /* the scanner follows the section's sampled bytes: a marker that keeps the state is among them (it reads no
         * backtick, but it is on the line); one that leaves the state starts the next section afresh */
        if (tr->to == state_) code_.feed(hold_.data() + at, len);
        else code_.reset();
        if (state_ == OUT_HEADER) {
            /* a header is a call's name (dropped) or, closed with no content kind, text */
            if (tr->act & OUT_ACT_HEADER_TEXT) emit_text(OUT_CONTENT, header_.data(), header_.size(), ev);
            header_.clear();
        } else if (tr->to != state_) {
            end_section(state_);
        }
        if (tr->to == OUT_CALL) {
            section &c = sec_[0];
            if (g_->content == OUT_TEXT_KEEP_UNLESS_BLANK && !c.started) c.ws.clear();   /* blank before a call */
            if (g_->content == OUT_TEXT_TRIM_AT_CALLS) {   /* trailing whitespace before a call; strip afresh after */
                c.ws.clear();
                c.started = false;
            }
            if (!block_seen_) {
                raw_lo_ = state_ == OUT_HEADER ? header_at_ : marker_at;
                call_at_ = marker_at;
                block_seen_ = true;
            }
            block_.clear();
            body_->open();
        }
        if (tr->to == OUT_HEADER) header_at_ = marker_at + len;
        hold_.erase(0, at + len);
        state_ = tr->to;
    }
}

void out_parser::finish(std::vector<out_event> *ev) {
    if (state_ == OUT_CALL) {
        block_ += hold_;
        hold_.clear();
        out_event e;
        e.kind = out_event::ERROR;
        char msg[160];
        snprintf(msg, sizeof msg, "unterminated tool call %d (the turn ended inside %s)", index_, g_->call_open);
        e.text = msg;
        e.index = index_;
        ev->push_back(std::move(e));
        errors_++;
        index_++;
    } else {
        /* a held marker prefix at the end was text after all; a header the turn ended in is text */
        emit_text(state_, hold_.data(), hold_.size(), ev);
        hold_.clear();
        if (state_ == OUT_HEADER) {
            emit_text(OUT_CONTENT, header_.data(), header_.size(), ev);
            header_.clear();
        }
        end_section(state_);
    }
    finish_section(OUT_REASONING, ev);
    finish_section(OUT_CONTENT, ev);
}

bool out_parser::open_reasoning(std::string *as_content) const {
    if (state_ != OUT_REASONING) return false;
    size_t lo = 0, hi = reasoning_.size();
    if (g_->content == OUT_TEXT_STRIP || g_->content == OUT_TEXT_RSTRIP) {
        py_strip_bounds(reasoning_.data(), reasoning_.size(), &lo, &hi);
        if (g_->content == OUT_TEXT_RSTRIP) lo = 0;
    }
    as_content->assign(reasoning_, lo, hi - lo);
    return true;
}

bool out_parser::raw_span(size_t *lo, size_t *hi) const {
    if (calls_.empty()) return false;
    *lo = raw_lo_;
    *hi = raw_hi_;
    return true;
}

/* ---- the declared type of a tool parameter ----------------------------------------------------- */

namespace {

/// References followed in one chain: deeper declares nothing (a real schema nests two or three).
constexpr int kParamRefDepth = 8;
/// Schema nodes one parameter's reading may visit: a schema past it declares nothing (a `$ref` diamond
/// would otherwise visit 2^depth nodes for every parameter of every call).
constexpr int kParamNodeBudget = 64;

/* Whether schema node `s` declares string among its types (out_tool_param_type's rule).  `root` is the
 * function's parameters object (the base of its local references); `chain[0..n_chain)` the reference
 * targets on the way here; `budget` the nodes left to visit. */
/// @param root    the function's `parameters` object
/// @param s       the schema node
/// @param chain   the `$ref` targets followed to reach `s` (room for kParamRefDepth)
/// @param n_chain how many
/// @param budget  in/out: nodes left to visit
/// @return true when `s` declares string (or a union that includes it)
bool schema_declares_string(const pyjson_value &root, const pyjson_value &s, const pyjson_value **chain, int n_chain,
                            int *budget) {
    if (--*budget < 0 || s.kind != pyjson_value::OBJ) return false;
    if (const pyjson_value *type = s.get("type")) {   /* a declared type decides alone, as it always did */
        if (type->kind == pyjson_value::STR) return type->s == "string";
        if (type->kind == pyjson_value::ARR)
            for (const pyjson_value &t : type->a)
                if (t.kind == pyjson_value::STR && t.s == "string") return true;
        return false;
    }
    const pyjson_value *ref = s.get("$ref");
    if (ref && ref->kind == pyjson_value::STR) {
        static const char *const kLocal[][2] = {{"#/$defs/", "$defs"}, {"#/definitions/", "definitions"}};
        for (const auto &l : kLocal) {
            const size_t n = strlen(l[0]);
            if (ref->s.compare(0, n, l[0])) continue;
            const pyjson_value *defs = root.get(l[1]);
            const pyjson_value *to = defs ? defs->get(ref->s.c_str() + n) : NULL;
            bool stop = !to || n_chain == kParamRefDepth;   /* unresolved, too deep, or a cycle */
            for (int i = 0; i < n_chain && !stop; i++) stop = chain[i] == to;
            if (stop) break;
            chain[n_chain] = to;
            if (schema_declares_string(root, *to, chain, n_chain + 1, budget)) return true;
            break;
        }
    }
    for (const char *k : {"anyOf", "oneOf"})
        if (const pyjson_value *u = s.get(k); u && u->kind == pyjson_value::ARR)
            for (const pyjson_value &b : u->a)
                if (schema_declares_string(root, b, chain, n_chain, budget)) return true;
    const pyjson_value *all = s.get("allOf");
    if (all && all->kind == pyjson_value::ARR && all->a.size() == 1 &&
        schema_declares_string(root, all->a[0], chain, n_chain, budget))
        return true;
    const pyjson_value *en = s.get("enum");
    if (en && en->kind == pyjson_value::ARR && !en->a.empty() &&
        std::all_of(en->a.begin(), en->a.end(), [](const pyjson_value &v) { return v.kind == pyjson_value::STR; }))
        return true;
    const pyjson_value *c = s.get("const");
    return c && c->kind == pyjson_value::STR;
}

}  // namespace

const char *out_tool_param_type(const pyjson_value &tools, const std::string &fn, const std::string &key) {
    for (const pyjson_value &tool : tools.a) {
        const pyjson_value *f = tool.get("function");
        if (!f) f = &tool;
        const pyjson_value *name = f->get("name");
        if (!name || name->kind != pyjson_value::STR || name->s != fn) continue;
        const pyjson_value *params = f->get("parameters");
        const pyjson_value *props = params ? params->get("properties") : NULL;
        const pyjson_value *prop = props ? props->get(key.c_str()) : NULL;
        if (!prop) return NULL;
        const pyjson_value *chain[kParamRefDepth];
        int budget = kParamNodeBudget;
        return schema_declares_string(*params, *prop, chain, 0, &budget) ? "string" : "other";
    }
    return NULL;
}

/* ---- the shared JSON call body ----------------------------------------------------------------- */

void out_json_call_body::open() {
    pos_ = 0;
    depth_ = 0;
    in_str_ = esc_ = false;
    str_lo_ = 0;
    key_.clear();
    want_value_ = false;
    value_key_.clear();
    name_done_ = false;
    name_.clear();
    args_lo_ = args_hi_ = std::string::npos;
    args_sent_ = 0;
    began_ = payload_ = held_ = false;
}

/* Advance over the body's new bytes: the depth-1 members' keys, "name"'s string value, where the "args"
 * object starts and ends. */
void out_json_call_body::scan(const std::string &b) {
    for (; pos_ < b.size(); pos_++) {
        const char c = b[pos_];
        if (in_str_) {
            if (esc_) {
                esc_ = false;
            } else if (c == '\\') {
                esc_ = true;
            } else if (c == '"') {
                in_str_ = false;
                if (depth_ == 1 && want_value_) {
                    if (value_key_ == "name" && !name_done_) {
                        pyjson_value v;
                        char jerr[8];
                        if (pyjson_parse(b.data() + str_lo_, pos_ + 1 - str_lo_, &v, jerr, sizeof jerr)) {
                            name_ = v.s;
                            name_done_ = true;
                        }
                    }
                    want_value_ = false;
                } else if (depth_ == 1) {
                    key_.assign(b, str_lo_ + 1, pos_ - str_lo_ - 1);
                }
            }
            continue;
        }
        switch (c) {
        case '"':
            in_str_ = true;
            str_lo_ = pos_;
            break;
        case ':':
            if (depth_ == 1) {
                want_value_ = true;
                value_key_ = key_;
            }
            break;
        case ',':
            if (depth_ == 1) {
                want_value_ = false;
                key_.clear();
            }
            break;
        case '{':
        case '[':
            if (depth_ == 1 && want_value_) {
                if (c == '{' && value_key_ == "args" && args_lo_ == std::string::npos) args_lo_ = pos_;
                want_value_ = false;
            }
            depth_++;
            break;
        case '}':
        case ']':
            depth_--;
            if (depth_ == 1 && args_lo_ != std::string::npos && args_hi_ == std::string::npos && pos_ > args_lo_)
                args_hi_ = pos_ + 1;   /* the first container closed back to depth 1 after "args" opened is it */
            break;
        default:
            break;
        }
    }
}

/* The args object's bytes so far, whole UTF-8 sequences, its final "}" held back. */
void out_json_call_body::stream_args(const std::string &b, bool closing, int index, std::vector<out_event> *ev) {
    if (!began_ || args_lo_ == std::string::npos) return;
    const size_t from = args_lo_ + args_sent_;
    size_t end = args_hi_ != std::string::npos ? args_hi_ - 1 : b.size();
    if (!closing && args_hi_ == std::string::npos) end = utf8_whole_end(b, from, end);
    if (end <= from) return;
    out_event e;
    e.kind = out_event::TOOL_ARGS;
    e.index = index;
    e.text.assign(b, from, end - from);
    ev->push_back(std::move(e));
    args_sent_ = end - args_lo_;
}

void out_json_call_body::grow(const std::string &b, size_t held, bool closing, int index,
                              std::vector<out_event> *ev) {
    scan(b);
    held_ = held > 0;
    if (!began_ && name_done_ && !name_.empty()) {
        out_event e;
        e.kind = out_event::TOOL_BEGIN;
        e.index = index;
        e.name = name_;
        ev->push_back(std::move(e));
        began_ = true;
    }
    stream_args(b, closing, index, ev);
    payload_ = !closing && !held_ && in_str_ && depth_ >= 2;
}

bool out_json_call_body::close(const std::string &b, int index, std::vector<out_event> *ev, out_call *call,
                               const char **why, size_t *at) {
    pyjson_value v;
    char jerr[160];
    *at = 0;
    if (!pyjson_parse(b.data(), b.size(), &v, jerr, sizeof jerr) || v.kind != pyjson_value::OBJ) {
        *why = "the body is not one JSON object";
        return false;
    }
    const pyjson_value *name = v.get("name");
    if (!name || name->kind != pyjson_value::STR || name->s.empty()) {
        *why = "no \"name\" string";
        return false;
    }
    const pyjson_value *args = v.get("args");
    if (args && args->kind != pyjson_value::OBJ) {
        *why = "\"args\" is not an object";
        *at = args_lo_ == std::string::npos ? 0 : args_lo_;
        return false;
    }
    if (!began_) {
        out_event e;
        e.kind = out_event::TOOL_BEGIN;
        e.index = index;
        e.name = name->s;
        ev->push_back(std::move(e));
    }
    call->name = name->s;
    call->arguments = args && args_hi_ != std::string::npos ? b.substr(args_lo_, args_hi_ - args_lo_) : "{}";
    return true;
}

/* ---- L295 F8 step 2: the chat-format registry of family parsers ---- */

/// The registry's list (zero-initialised before any constructor runs, so registration order does not matter).
static out_parser_family *g_out_parser_families = nullptr;

void out_parser_family_register(out_parser_family *f) {
    f->next = g_out_parser_families;
    g_out_parser_families = f;
}

const out_parser_family *out_parser_family_find(int chat_format) {
    for (const out_parser_family *f = g_out_parser_families; f; f = f->next)
        if (f->chat_format == chat_format) return f;
    return nullptr;
}

/* ---- the XML tool-call body (moved from qwen_output.cpp, L295 F8 step 1) ---- */

void xml_call_body::open() {
    began_ = false;
    head_ws_ = head_seen_ = 0;
    fn_.clear();
    pay_ = closer_ = value_scan();
}

/* One "<parameter=KEY>\nVALUE\n</parameter>" of function `fn` at b[*i]: 1 = read (*key, the typed *v,
 * *i past it), 0 = not complete yet, -1 = not a parameter; *why names what is missing or wrong.  The one
 * reading close and the stream share.  L429: the stream reads the parameter it waits on at every feed, so it
 * passes `r`, where a 0 leaves how far the key's end and then the value's closer were looked for; the next
 * reading starts there (a 0 said nothing before it ends the key or starts a whole closer, and the body only
 * grows).  The answers are those of a reading from the parameter's start. */
int xml_call_body::read_param(const std::string &b, size_t *i, const std::string &fn, std::string *key,
                              pyjson_value *v, const char **why, param_resume *r) const {
    static const char kParam[] = "<parameter=", kParamEnd[] = "</parameter>";
    const size_t n = strlen(kParam), rem = b.size() - *i;
    if (b.compare(*i, std::min(rem, n), kParam, std::min(rem, n))) {
        *why = "expected <parameter= or </function>";
        return -1;
    }
    *why = "unterminated <parameter=KEY>";
    if (rem < n) return 0;
    const size_t at = *i + n;
    size_t key_end = r ? r->key_end : 0;
    if (!key_end) {
        /* the key ends at its '>'; a newline before it (or with no '>' yet) is not a key */
        key_end = b.find_first_of(">\n", r ? std::max(at, r->seen) : at);
        if (key_end == std::string::npos) {
            if (r) r->seen = b.size();
            return 0;
        }
        if (b[key_end] == '\n') return -1;
        if (r) *r = {key_end, key_end + 1};
    }
    *why = "unterminated parameter value";
    const size_t close = b.find(kParamEnd, r ? r->seen : key_end + 1);
    if (close == std::string::npos) {
        /* a closer that starts in the last strlen - 1 bytes may still complete */
        if (r) r->seen = std::max(r->seen, b.size() - std::min(b.size(), strlen(kParamEnd) - 1));
        return 0;
    }
    *key = b.substr(at, key_end - at);
    size_t vlo = key_end + 1, vhi = close;
    if (vlo < vhi && b[vlo] == '\n') vlo++;
    if (vhi > vlo && b[vhi - 1] == '\n') vhi--;
    const std::string text = b.substr(vlo, vhi - vlo);
    *i = close + strlen(kParamEnd);
    /* A value is JSON only when the schema does not declare it a string
     * AND the text is exactly what tojson would have written for it: the
     * template writes every non-string with tojson, so text in any other
     * spelling can only have been a raw string. */
    *v = pyjson_value();
    const char *type = out_tool_param_type(tools, fn, *key);
    char jerr[8];
    bool as_json = !(type && !strcmp(type, "string")) && pyjson_parse(text.data(), text.size(), v, jerr, sizeof jerr);
    if (as_json) {
        std::string canon;
        pyjson_dump(*v, &canon);
        as_json = v->kind != pyjson_value::STR && canon == text;   /* strings are written raw, never tojson */
        /* a template that spells scalars with str() wrote no "true" / "false" / "null" */
        if (python_scalars && (v->kind == pyjson_value::BOOL || v->kind == pyjson_value::NUL)) as_json = false;
    }
    if (!as_json && python_scalars && !(type && !strcmp(type, "string")) &&
        (text == "True" || text == "False" || text == "None")) {
        *v = pyjson_value();
        v->kind = text == "None" ? pyjson_value::NUL : pyjson_value::BOOL;
        v->b = text == "True";
        return 1;
    }
    if (!as_json) {
        *v = pyjson_value();
        v->kind = pyjson_value::STR;
        v->s = text;
    }
    return 1;
}

/* The open call's parameters that closed since the last call, as TOOL_ARGS fragments.  L429: the parameter
 * still open is read from where its last reading stopped (arg_resume_), not from its start. */
void xml_call_body::stream_params(const std::string &b, int index, std::vector<out_event> *ev) {
    static const char kFnEnd[] = "</function>";
    while (!arg_stop_) {
        size_t i = arg_pos_;
        while (i < b.size() && (b[i] == ' ' || b[i] == '\n' || b[i] == '\t' || b[i] == '\r')) i++;
        arg_pos_ = i;   /* the whitespace between parameters is skipped once */
        const size_t rem = b.size() - i;
        if (!rem || !b.compare(i, std::min(rem, strlen(kFnEnd)), kFnEnd, std::min(rem, strlen(kFnEnd))))
            return;   /* the call's close is close()'s */
        std::string key;
        pyjson_value v;
        const char *why = NULL;
        const int r = read_param(b, &i, fn_, &key, &v, &why, &arg_resume_);
        if (r == 0) return;
        if (r < 0) {
            arg_stop_ = true;   /* close() names the fault */
            return;
        }
        pyjson_value one;
        one.kind = pyjson_value::OBJ;
        one.o.emplace_back(key, std::move(v));
        std::string d;
        pyjson_dump(one, &d);   /* {"key": value} */
        out_event e;
        e.kind = out_event::TOOL_ARGS;
        e.index = index;
        e.text = arg_n_ ? ", " + d.substr(1, d.size() - 2) : d.substr(0, d.size() - 1);
        ev->push_back(std::move(e));
        arg_n_++;
        arg_pos_ = i;
        arg_resume_ = param_resume();
    }
}

/* Advance a value scan over the body's new bytes.  A parameter value starts past "<parameter=KEY>" and ends at
 * the first "</parameter>" -- close()'s reading.  Outside a value `pos` waits at an open tag (or where one may
 * still start); inside one it waits at a partial closer. */
void xml_call_body::scan_values(value_scan *vs, const std::string &b) const {
    static const char kParam[] = "<parameter=", kParamEnd[] = "</parameter>";
    value_scan &ps = *vs;
    for (;;) {
        if (!ps.value) {
            const size_t at = b.find(kParam, ps.pos);
            if (at == std::string::npos) {
                ps.pos = b.size() > strlen(kParam) ? std::max(ps.pos, b.size() - strlen(kParam)) : ps.pos;
                break;
            }
            /* L429: a tag still open is looked through once (key_seen is at most the '>' of the last tag read,
             * so it never hides a later tag's) */
            const size_t gt = b.find('>', std::max(at, ps.key_seen));
            if (gt == std::string::npos) {
                ps.pos = at;
                ps.key_seen = b.size();
                break;
            }
            const char *type = out_tool_param_type(tools, fn_, b.substr(at + strlen(kParam), gt - at - strlen(kParam)));
            ps.value = true;
            ps.is_string = type && !strcmp(type, "string");
            ps.json_str = ps.esc = false;
            ps.pos = ps.vstart = gt + 1;
            continue;
        }
        bool closed = false;
        while (ps.pos < b.size()) {
            if (b[ps.pos] == '<') {
                if (!b.compare(ps.pos, strlen(kParamEnd), kParamEnd)) {
                    ps.pos += strlen(kParamEnd);
                    ps.value = false;
                    closed = true;
                    break;
                }
                if (b.size() - ps.pos < strlen(kParamEnd) &&
                    !b.compare(ps.pos, b.size() - ps.pos, kParamEnd, b.size() - ps.pos))
                    break;   /* a partial closer waits for its next bytes */
            }
            const char c = b[ps.pos++];
            if (ps.is_string) continue;
            if (ps.json_str) {
                if (ps.esc) ps.esc = false;
                else if (c == '\\') ps.esc = true;
                else if (c == '"') ps.json_str = false;
            } else if (c == '"') {
                ps.json_str = true;
            }
        }
        if (!closed) break;
    }
}

/* Advance pay_ over the body's new bytes (the kit holds a partial "</tool_call>" out of it: `held`) and say
 * whether the next byte is payload. */
void xml_call_body::scan_payload(const std::string &b, size_t held) {
    value_scan &ps = pay_;
    scan_values(&ps, b);
    /* the payload: inside a string value or a JSON string, past the template's leading newline, not on
     * "</" of a closer (this block's or the held "</tool_call>") */
    const size_t pending = b.size() - std::min(ps.pos, b.size());
    ps.payload = ps.value && (ps.is_string || ps.json_str) && b.size() > ps.vstart && pending < 2 && held < 2;
}

/* L410: a "</tool_call>" found after `b` is a value's text when `b` ends inside a parameter value -- past a
 * "<parameter=KEY>" with no "</parameter>" after it (read_param's reading: a value ends at the first one).
 * One exception keeps a malformed call's error where it was: a value never closed but followed by
 * "</function>" is a call that ended, not a value that goes on (the model wrote the call's close).
 * L429: the values are followed by a scan of its own (closer_), advanced over what `b` gained since the last
 * marker; a value that spells the marker many times is not read from the body's start each time. */
bool xml_call_body::holds_closer(const std::string &b) {
    static const char kFnEnd[] = "</function>";
    scan_values(&closer_, b);
    if (!closer_.value) return false;
    size_t e = b.size();
    while (e && (b[e - 1] == ' ' || b[e - 1] == '\n' || b[e - 1] == '\t' || b[e - 1] == '\r')) e--;
    return !(e >= strlen(kFnEnd) && !b.compare(e - strlen(kFnEnd), strlen(kFnEnd), kFnEnd));
}

void xml_call_body::grow(const std::string &b, size_t held, bool closing, int index,
                         std::vector<out_event> *ev) {
    if (!began_) {
        /* announce the call as soon as its name is complete (L429: looking from where the last feed stopped) */
        size_t i = head_ws_;
        while (i < b.size() && (b[i] == ' ' || b[i] == '\n' || b[i] == '\t' || b[i] == '\r')) i++;
        head_ws_ = i;
        if (!b.compare(i, 10, "<function=")) {
            /* the name ends at its '>'; a newline first is no name (close() names the fault) */
            const size_t e = b.find_first_of(">\n", std::max(i + 10, head_seen_));
            head_seen_ = e == std::string::npos ? b.size() : e;
            if (e != std::string::npos && b[e] == '>') {
                out_event ev_begin;
                ev_begin.kind = out_event::TOOL_BEGIN;
                ev_begin.index = index;
                ev_begin.name = b.substr(i + 10, e - i - 10);
                fn_ = ev_begin.name;
                ev->push_back(std::move(ev_begin));
                began_ = true;
                arg_pos_ = e + 1;
                arg_resume_ = param_resume();
                arg_n_ = 0;
                arg_stop_ = false;
            }
        }
    }
    if (!began_) return;
    if (!closing) scan_payload(b, held);
    stream_params(b, index, ev);   /* closing: a last parameter closed in this piece goes out before the end */
}

/* The body of one <tool_call> ... </tool_call>. */
bool xml_call_body::close(const std::string &b, int index, std::vector<out_event> *ev, out_call *call,
                          const char **why, size_t *at) {
    size_t i = 0;
    auto skip_ws = [&]() {
        while (i < b.size() && (b[i] == ' ' || b[i] == '\n' || b[i] == '\t' || b[i] == '\r')) i++;
    };
    auto fail = [&](const char *w) {
        *why = w;
        *at = i;
        return false;
    };
    skip_ws();
    if (b.compare(i, 10, "<function=")) return fail("expected <function=");
    i += 10;
    const size_t name_end = b.find('>', i);
    if (name_end == std::string::npos || b.find('\n', i) < name_end) return fail("unterminated <function=NAME>");
    const std::string name = b.substr(i, name_end - i);
    i = name_end + 1;
    pyjson_value args;
    args.kind = pyjson_value::OBJ;
    for (;;) {
        skip_ws();
        if (!b.compare(i, 11, "</function>")) {
            i += 11;
            break;
        }
        std::string key;
        pyjson_value v;
        const char *w = NULL;
        if (read_param(b, &i, name, &key, &v, &w, NULL) != 1) return fail(w);
        bool replaced = false;   /* a repeated key: first position, last value */
        for (auto &kv : args.o)
            if (kv.first == key) { kv.second = v; replaced = true; }
        if (!replaced) args.o.emplace_back(key, std::move(v));
    }
    skip_ws();
    if (i != b.size()) return fail("text after </function>");
    call->name = name;
    pyjson_dump(args, &call->arguments);
    if (!began_) {
        out_event e;
        e.kind = out_event::TOOL_BEGIN;
        e.index = index;
        e.name = name;
        ev->push_back(std::move(e));
    }
    return true;
}

/** @file
 *  The XML tool-call body as it read before L429, kept for the differential test and the bench. */
/* xml_call_body_ref.h -- src/lib's xml_call_body (xml_call_body.h, out_parse_kit.cpp) as it stood at 718c5848, the
 * commit before L429 made its per-feed readings resume: every grow() read the open parameter from its start
 * (read_param's finds), and holds_closer read every parameter from the body's start.  Its code is that commit's,
 * byte for byte, but for the class name, `inline`, and holds_closer's const (the kit's signature changed).  (L430: the
 * live body now shares its tools with its copies; the harness below gives each body its tools in its own form.)
 *
 * It is the REFERENCE of two instruments and is not to be edited with the live body:
 *   tests/qwen_chat_gate.cpp  P3: the live body must produce this one's events, in_payload() answers and calls at
 *                             every feed of every chunking;
 *   tests/xml_call_bench.cpp  what the rescans cost.
 * Delete it with those two when the live body's syntax changes on purpose (the reference would then be wrong). */
#pragma once

#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

#include <memory>
#include <utility>

#include "out_parse_kit.h"
#include "pyjson.h"
#include "xml_call_body.h"

/** xml_call_body before L429 (see xml_call_body.h for the format and the rules). */
class xml_call_body_ref : public out_call_body {
public:
    /** The request's tools (the schema that types arguments). */
    pyjson_value tools;
    /** How the template spelt a non-string SCALAR: false = tojson (Qwen: true / false / null), true = Python's str()
     *  (L296 Nemotron-H's `args_value | string`: True / False / None; numbers spell the same either way; mappings and
     *  lists are tojson in both).  A value is typed only from its template's own spelling. */
    bool python_scalars = false;

    void open() override;
    /// L437 plumbing: the kit's typed hook; the reference's own reading (grow_ref) takes the bare count it was written on.
    void grow(const std::string &b, GenBytes held, bool closing, int index, std::vector<out_event> *ev) override {
        grow_ref(b, held.raw(), closing, index, ev);
    }
    /// out_call_body::grow as it read at the reference's commit.
    /// @param b       the call's body so far
    /// @param held    bytes the kit holds back as a possible closing marker
    /// @param closing the closing marker was found
    /// @param index   the call's index
    /// @param ev      receives the events
    void grow_ref(const std::string &b, size_t held, bool closing, int index, std::vector<out_event> *ev);
    bool in_payload() const override { return pay_.value && pay_.payload; }
    bool holds_closer(const std::string &b) override;   /* the kit's L429 signature; the reading is const */
    /// L437 plumbing: the kit's typed hook; the reference's own reading (close_ref) reports a bare offset.
    bool close(const std::string &b, int index, std::vector<out_event> *ev, out_call *call, const char **why,
               BodyOff *at) override {
        size_t bare = 0;
        const bool ok = close_ref(b, index, ev, call, why, &bare);
        *at = BodyOff{bare};
        return ok;
    }
    /// out_call_body::close as it read at the reference's commit.
    /// @param b     the call's complete body
    /// @param index the call's index
    /// @param ev    receives TOOL_BEGIN if grow never sent it
    /// @param call  receives the call on success
    /// @param why   on failure, what is missing or wrong
    /// @param at    on failure, the byte of the body where reading stopped
    /// @return true when the body is a well-formed call
    bool close_ref(const std::string &b, int index, std::vector<out_event> *ev, out_call *call, const char **why,
                   size_t *at);

private:
    /// Advance pay_ over the body's new bytes and set pay_.payload for the next byte.
    /// @param b    the call's body so far
    /// @param held bytes the kit holds back as a possible "</tool_call>"
    void scan_payload(const std::string &b, size_t held);
    /// Read one "<parameter=KEY>\nVALUE\n</parameter>" of function `fn` at b[*i] (the reading close() and the
    /// stream share).
    /// @param b   the call's body
    /// @param i   in: where the parameter starts; out: past it (on 1)
    /// @param fn  the call's function name (its schema types the value)
    /// @param key receives the parameter's name
    /// @param v   receives the typed value
    /// @param why receives what is missing or wrong (static string)
    /// @return 1 = read, 0 = not complete yet, -1 = not a parameter
    int read_param(const std::string &b, size_t *i, const std::string &fn, std::string *key, pyjson_value *v,
                   const char **why) const;
    /// Emit a TOOL_ARGS fragment for each parameter that closed since the last call.
    /// @param b     the call's body so far
    /// @param index the call's index, for the events
    /// @param ev    receives the fragments
    void stream_params(const std::string &b, int index, std::vector<out_event> *ev);

    /** @var pay_
    *  Where the open block's decode is (in_payload), scanned incrementally. */
    bool began_ = false;       ///< TOOL_BEGIN sent for the open call
    std::string fn_;           ///< the open call's function name, once complete
    size_t arg_pos_ = 0;       ///< body offset past the last parameter streamed (TOOL_ARGS)
    int arg_n_ = 0;            ///< parameters streamed for the open call
    bool arg_stop_ = false;    ///< the open call stopped streaming: its next bytes are not a parameter
    /// The incremental payload scanner's state for the open call.
    struct payload_scan {
        size_t pos = 0;        ///< bytes of the body classified
        bool value = false;    ///< inside a parameter value
        size_t vstart = 0;     ///< where its text starts (past the tag)
        bool is_string = false;///< the schema declares it a string
        bool json_str = false; ///< inside a JSON string literal of a non-string value
        bool esc = false;      ///< the previous byte was a backslash in that literal
        bool payload = false;  ///< the result for the next byte
    } pay_;
};

inline void xml_call_body_ref::open() {
    began_ = false;
    fn_.clear();
    pay_ = payload_scan();
}

/* One "<parameter=KEY>\nVALUE\n</parameter>" of function `fn` at b[*i]: 1 = read (*key, the typed *v,
 * *i past it), 0 = not complete yet, -1 = not a parameter; *why names what is missing or wrong.  The one
 * reading close and the stream share. */
inline int xml_call_body_ref::read_param(const std::string &b, size_t *i, const std::string &fn, std::string *key,
                              pyjson_value *v, const char **why) const {
    static const char kParam[] = "<parameter=";
    const size_t n = strlen(kParam), rem = b.size() - *i;
    if (b.compare(*i, std::min(rem, n), kParam, std::min(rem, n))) {
        *why = "expected <parameter= or </function>";
        return -1;
    }
    *why = "unterminated <parameter=KEY>";
    if (rem < n) return 0;
    const size_t at = *i + n;
    const size_t key_end = b.find('>', at);
    if (b.find('\n', at) < key_end) return -1;
    if (key_end == std::string::npos) return 0;
    *why = "unterminated parameter value";
    const size_t close = b.find("</parameter>", key_end + 1);
    if (close == std::string::npos) return 0;
    *key = b.substr(at, key_end - at);
    size_t vlo = key_end + 1, vhi = close;
    if (vlo < vhi && b[vlo] == '\n') vlo++;
    if (vhi > vlo && b[vhi - 1] == '\n') vhi--;
    const std::string text = b.substr(vlo, vhi - vlo);
    *i = close + 12;
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

/* The open call's parameters that closed since the last call, as TOOL_ARGS fragments. */
inline void xml_call_body_ref::stream_params(const std::string &b, int index, std::vector<out_event> *ev) {
    static const char kFnEnd[] = "</function>";
    while (!arg_stop_) {
        size_t i = arg_pos_;
        while (i < b.size() && (b[i] == ' ' || b[i] == '\n' || b[i] == '\t' || b[i] == '\r')) i++;
        const size_t rem = b.size() - i;
        if (!rem || !b.compare(i, std::min(rem, strlen(kFnEnd)), kFnEnd, std::min(rem, strlen(kFnEnd))))
            return;   /* the call's close is close()'s */
        std::string key;
        pyjson_value v;
        const char *why = NULL;
        const int r = read_param(b, &i, fn_, &key, &v, &why);
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
    }
}

/* Advance pay_ over the body's new bytes (the kit holds a partial "</tool_call>" out of it: `held`).  A
 * parameter value starts past "<parameter=KEY>" and ends at the first "</parameter>" -- close()'s
 * reading. */
inline void xml_call_body_ref::scan_payload(const std::string &b, size_t held) {
    static const char kParam[] = "<parameter=", kParamEnd[] = "</parameter>";
    payload_scan &ps = pay_;
    for (;;) {
        if (!ps.value) {
            const size_t at = b.find(kParam, ps.pos);
            if (at == std::string::npos) {
                ps.pos = b.size() > strlen(kParam) ? std::max(ps.pos, b.size() - strlen(kParam)) : ps.pos;
                break;
            }
            const size_t gt = b.find('>', at);
            if (gt == std::string::npos) {
                ps.pos = at;
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
            if (!b.compare(ps.pos, strlen(kParamEnd), kParamEnd)) {
                ps.pos += strlen(kParamEnd);
                ps.value = false;
                closed = true;
                break;
            }
            if (b[ps.pos] == '<' && b.size() - ps.pos < strlen(kParamEnd) &&
                !b.compare(ps.pos, b.size() - ps.pos, kParamEnd, b.size() - ps.pos))
                break;   /* a partial closer waits for its next bytes */
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
    /* the payload: inside a string value or a JSON string, past the template's leading newline, not on
     * "</" of a closer (this block's or the held "</tool_call>") */
    const size_t pending = b.size() - std::min(ps.pos, b.size());
    ps.payload = ps.value && (ps.is_string || ps.json_str) && b.size() > ps.vstart && pending < 2 && held < 2;
}

/* L410: a "</tool_call>" found after `b` is a value's text when `b` ends inside a parameter value -- past a
 * "<parameter=KEY>" with no "</parameter>" after it (read_param's reading: a value ends at the first one).
 * One exception keeps a malformed call's error where it was: a value never closed but followed by
 * "</function>" is a call that ended, not a value that goes on (the model wrote the call's close). */
inline bool xml_call_body_ref::holds_closer(const std::string &b) {
    static const char kParam[] = "<parameter=", kParamEnd[] = "</parameter>", kFnEnd[] = "</function>";
    size_t pos = 0;
    for (;;) {
        const size_t at = b.find(kParam, pos);
        if (at == std::string::npos) return false;
        const size_t gt = b.find('>', at);
        if (gt == std::string::npos) return false;
        const size_t end = b.find(kParamEnd, gt + 1);
        if (end == std::string::npos) break;
        pos = end + strlen(kParamEnd);
    }
    size_t e = b.size();
    while (e && (b[e - 1] == ' ' || b[e - 1] == '\n' || b[e - 1] == '\t' || b[e - 1] == '\r')) e--;
    return !(e >= strlen(kFnEnd) && !b.compare(e - strlen(kFnEnd), strlen(kFnEnd), kFnEnd));
}

inline void xml_call_body_ref::grow_ref(const std::string &b, size_t held, bool closing, int index,
                         std::vector<out_event> *ev) {
    if (!began_) {
        /* announce the call as soon as its name is complete */
        size_t i = 0;
        while (i < b.size() && (b[i] == ' ' || b[i] == '\n' || b[i] == '\t' || b[i] == '\r')) i++;
        if (!b.compare(i, 10, "<function=")) {
            const size_t e = b.find('>', i + 10);
            const size_t nl = b.find('\n', i + 10);
            if (e != std::string::npos && nl > e) {
                out_event ev_begin;
                ev_begin.kind = out_event::TOOL_BEGIN;
                ev_begin.index = index;
                ev_begin.name = b.substr(i + 10, e - i - 10);
                fn_ = ev_begin.name;
                ev->push_back(std::move(ev_begin));
                began_ = true;
                arg_pos_ = e + 1;
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
inline bool xml_call_body_ref::close_ref(const std::string &b, int index, std::vector<out_event> *ev, out_call *call,
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
        if (read_param(b, &i, name, &key, &v, &w) != 1) return fail(w);
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

/// Give a body its request's tools: the reference keeps its own copy, as it did.
/// @param b     the reference body
/// @param tools the parsed tools array
inline void call_body_set_tools(xml_call_body_ref *b, pyjson_value &&tools) { b->tools = std::move(tools); }
/// Give a body its request's tools: the live body shares them with its copies (L430).
/// @param b     the live body
/// @param tools the parsed tools array
inline void call_body_set_tools(xml_call_body *b, pyjson_value &&tools) {
    b->tools = std::make_shared<const pyjson_value>(std::move(tools));
}

/** A kit parser around one call body `Body` (xml_call_body or xml_call_body_ref): the same kit, the same grammar,
 *  so two of them differ only in the body's reading. */
template <class Body> class call_body_parser : public out_parser {
public:
    Body body;   ///< the parser's own body
    /// Bind the grammar and the body for one turn.
    /// @param g              the grammar (outlives the parser)
    /// @param tools_json     the request's tools array as JSON text, or NULL
    /// @param python_scalars the body's scalar spelling (xml_call_body::python_scalars)
    /// @param initial        the state the turn starts in
    /// @return false when `tools_json` is not a JSON array
    bool start(const out_grammar &g, const char *tools_json, bool python_scalars, out_state initial) {
        pyjson_value tools;
        char jerr[160];
        if (tools_json && (!pyjson_parse(tools_json, strlen(tools_json), &tools, jerr, sizeof jerr) ||
                           tools.kind != pyjson_value::ARR))
            return false;
        call_body_set_tools(&body, std::move(tools));
        body.python_scalars = python_scalars;
        init(&g, &body, initial);
        return true;
    }
    out_parser *clone() const override {
        call_body_parser *p = new call_body_parser(*this);
        p->rebind_body(&p->body);
        return p;
    }
};

/* qwen_output.cpp -- the Qwen3.8-Flash-Next output parser: the output-parser kit's Qwen configuration (its
 * grammar) and Qwen's XML tool-call body.  See qwen_chat.h. */
#include "qwen_chat.h"

#include <stdio.h>
#include <string.h>

const out_grammar &qwen_output_grammar(void) {
    enum { THINK_END, CALL_OPEN, CALL_CLOSE };
    static const out_grammar g = {
        /* .name        = */ "qwen",
        /* .markers     = */ {"</think>", "<tool_call>", "</tool_call>"},
        /* .transitions = */ {
            {OUT_REASONING, THINK_END, OUT_CONTENT, 0},
            {OUT_CONTENT, CALL_OPEN, OUT_CALL, 0},
            {OUT_CALL, CALL_CLOSE, OUT_CONTENT, 0},
        },
        /* .reasoning   = */ OUT_TEXT_STRIP,
        /* .content     = */ OUT_TEXT_STRIP,
        /* .call_open   = */ "<tool_call>",
        /* .dropped     = */ {},
    };
    return g;
}

bool qwen_output_parser::init(bool thinking, const char *tools_json, char *err, size_t errlen) {
    body_.tools = pyjson_value();
    if (tools_json) {
        char jerr[160] = "not a JSON array";
        if (!pyjson_parse(tools_json, strlen(tools_json), &body_.tools, jerr, sizeof jerr) ||
            body_.tools.kind != pyjson_value::ARR) {
            if (err && errlen) snprintf(err, errlen, "tools_not_array: %s", jerr);
            return false;
        }
    }
    out_parser::init(&qwen_output_grammar(), &body_, thinking ? OUT_REASONING : OUT_CONTENT);
    return true;
}

void qwen_xml_call_body::open() {
    began_ = false;
    fn_.clear();
    pay_ = payload_scan();
}

/* The type the schema declares for `key` of function `fn`: "string" when the
 * declared type is or includes string, "other" for any other declaration,
 * NULL when the tool or the property is not declared. */
const char *qwen_xml_call_body::param_type(const std::string &fn, const std::string &key) const {
    for (const pyjson_value &tool : tools.a) {
        const pyjson_value *f = tool.get("function");
        if (!f) f = &tool;
        const pyjson_value *name = f->get("name");
        if (!name || name->kind != pyjson_value::STR || name->s != fn) continue;
        const pyjson_value *params = f->get("parameters");
        const pyjson_value *props = params ? params->get("properties") : NULL;
        const pyjson_value *prop = props ? props->get(key.c_str()) : NULL;
        if (!prop) return NULL;
        const pyjson_value *type = prop->get("type");
        if (!type) return "other";
        if (type->kind == pyjson_value::STR) return type->s == "string" ? "string" : "other";
        if (type->kind == pyjson_value::ARR)
            for (const pyjson_value &t : type->a)
                if (t.kind == pyjson_value::STR && t.s == "string") return "string";
        return "other";
    }
    return NULL;
}

/* One "<parameter=KEY>\nVALUE\n</parameter>" of function `fn` at b[*i]: 1 = read (*key, the typed *v,
 * *i past it), 0 = not complete yet, -1 = not a parameter; *why names what is missing or wrong.  The one
 * reading close and the stream share. */
int qwen_xml_call_body::read_param(const std::string &b, size_t *i, const std::string &fn, std::string *key,
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
    const char *type = param_type(fn, *key);
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
void qwen_xml_call_body::stream_params(const std::string &b, int index, std::vector<out_event> *ev) {
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
void qwen_xml_call_body::scan_payload(const std::string &b, size_t held) {
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
            const char *type = param_type(fn_, b.substr(at + strlen(kParam), gt - at - strlen(kParam)));
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

void qwen_xml_call_body::grow(const std::string &b, size_t held, bool closing, int index,
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
bool qwen_xml_call_body::close(const std::string &b, int index, std::vector<out_event> *ev, out_call *call,
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

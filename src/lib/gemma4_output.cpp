/** @file
 *  The Gemma 4 output parser: the output-parser kit's Gemma grammar and Gemma's call body. */
/* gemma4_output.cpp -- the Gemma 4 output parser: the kit's Gemma configuration (vLLM's gemma4 reading:
 * vllm/parser/gemma4.py) and the call body "call:NAME{ARGS}", ARGS read back from the template's own format_argument
 * spelling.  See gemma4_chat.h. */
#include "gemma4_chat.h"

#include <stdio.h>
#include <string.h>

#include <functional>

namespace {

/// The template's string quote (an added token: one token, its text never split across a feed).
const std::string kQ = "<|\"|>";

/// @param c a byte
/// @return true for vLLM's argument separators' whitespace (space, newline, tab)
bool ws(char c) { return c == ' ' || c == '\n' || c == '\t'; }

/// @param s text
/// @return Python's str.strip() of it
std::string strip(const std::string &s) { return py_strip(s); }

/// @param s a bare value's text
/// @return true when it is a JSON number's spelling (what the template writes for a number: Python's int / float)
bool json_number(const std::string &s) {
    size_t i = 0, n = s.size();
    if (i < n && s[i] == '-') i++;
    if (i >= n) return false;
    if (s[i] == '0') i++;
    else if (s[i] >= '1' && s[i] <= '9') while (i < n && s[i] >= '0' && s[i] <= '9') i++;
    else return false;
    if (i < n && s[i] == '.') {
        if (++i >= n || s[i] < '0' || s[i] > '9') return false;
        while (i < n && s[i] >= '0' && s[i] <= '9') i++;
    }
    if (i < n && (s[i] == 'e' || s[i] == 'E')) {
        if (++i < n && (s[i] == '+' || s[i] == '-')) i++;
        if (i >= n || s[i] < '0' || s[i] > '9') return false;
        while (i < n && s[i] >= '0' && s[i] <= '9') i++;
    }
    return i == n;
}

/// @param raw a bare value's text (stripped)
/// @param as_string the schema declares the property a string
/// @return the value its spelling names: true / false / null, a number, else the text as a string
pyjson_value bare(const std::string &raw, bool as_string) {
    pyjson_value v;
    if (!as_string) {
        if (raw == "true" || raw == "false") {
            v.kind = pyjson_value::BOOL;
            v.b = raw == "true";
            return v;
        }
        if (raw == "null") return v;   /* NUL */
        char jerr[8];
        if (json_number(raw) && pyjson_parse(raw.data(), raw.size(), &v, jerr, sizeof jerr)) return v;
    }
    v = pyjson_value();
    v.kind = pyjson_value::STR;
    v.s = raw;
    return v;
}

/* The end of the bracketed value opening at s[i] (`open` / `close`, <|"|> strings skipped): one past its closer, or
 * s.size() when it never closes (then *closed is false). */
/// @param s      the text
/// @param i      the opener's index
/// @param open   the opening byte
/// @param close  the closing byte
/// @param closed receives whether the closer was found
/// @return one past the closer, or s.size()
size_t match(const std::string &s, size_t i, char open, char close, bool *closed) {
    int depth = 1;
    size_t j = i + 1;
    while (j < s.size() && depth > 0) {
        if (!s.compare(j, kQ.size(), kQ)) {
            const size_t nd = s.find(kQ, j + kQ.size());
            j = nd == std::string::npos ? s.size() : nd + kQ.size();
            continue;
        }
        if (s[j] == open) depth++;
        else if (s[j] == close) depth--;
        j++;
    }
    *closed = depth == 0;
    return j;
}

/// Whether a top-level argument key's property is declared a string (NULL: none is -- a nested object).
using string_keys = std::function<bool(const std::string &)>;

pyjson_value parse_array(const std::string &s);
pyjson_value parse_fields(const std::string &s, const string_keys *string_key);

/* the value starting at s[*i] (whitespace skipped by the caller), *i past it; `stops` end a bare value */
/// @param s         the text
/// @param i         in: the value's first byte; out: past the value
/// @param stops     the bytes that end a bare value
/// @param as_string a bare value stays the raw text (its property is declared a string)
/// @return the value
pyjson_value value_at(const std::string &s, size_t *i, const char *stops, bool as_string) {
    if (!s.compare(*i, kQ.size(), kQ)) {
        const size_t lo = *i + kQ.size(), end = s.find(kQ, lo);
        pyjson_value v;
        v.kind = pyjson_value::STR;
        v.s = s.substr(lo, end == std::string::npos ? std::string::npos : end - lo);
        *i = end == std::string::npos ? s.size() : end + kQ.size();
        return v;
    }
    if (s[*i] == '{' || s[*i] == '[') {
        const bool obj = s[*i] == '{';
        bool closed = false;
        const size_t j = match(s, *i, obj ? '{' : '[', obj ? '}' : ']', &closed);
        const std::string inner = s.substr(*i + 1, (closed ? j - 1 : j) - (*i + 1));
        *i = j;
        return obj ? parse_fields(inner, NULL) : parse_array(inner);
    }
    const size_t lo = *i;
    while (*i < s.size() && !strchr(stops, s[*i])) (*i)++;
    return bare(strip(s.substr(lo, *i - lo)), as_string);
}

/* "k:v,k:v" -> an object (vLLM's _parse_gemma4_fields, typed by the spelling) */
/// @param s          the fields' text (no outer braces)
/// @param string_key the top level's declared-string test, or NULL
/// @return the object
pyjson_value parse_fields(const std::string &s, const string_keys *string_key) {
    pyjson_value obj;
    obj.kind = pyjson_value::OBJ;
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        while (i < n && (ws(s[i]) || s[i] == ',')) i++;
        if (i >= n) break;
        const size_t ks = i;
        while (i < n && s[i] != ':') i++;
        if (i >= n) break;
        std::string key = strip(s.substr(ks, i - ks));
        if (key.size() >= 2 * kQ.size() && !key.compare(0, kQ.size(), kQ) && !key.compare(key.size() - kQ.size(), kQ.size(), kQ))
            key = key.substr(kQ.size(), key.size() - 2 * kQ.size());
        i++;
        while (i < n && ws(s[i])) i++;
        pyjson_value v;
        if (i >= n) {
            v.kind = pyjson_value::STR;   /* "key:" at the end: an empty string (vLLM) */
        } else {
            const bool as_string = string_key && (*string_key)(key);
            const size_t before = i;
            v = value_at(s, &i, ",}]", as_string);
            if (i == before) break;   /* no progress: malformed (vLLM aborts the same way) */
        }
        bool replaced = false;
        for (auto &kv : obj.o)
            if (kv.first == key) { kv.second = std::move(v); replaced = true; break; }
        if (!replaced) obj.o.emplace_back(std::move(key), std::move(v));
    }
    return obj;
}

/* "[a,b]"'s inside -> an array */
/// @param s the array's text (no brackets)
/// @return the array
pyjson_value parse_array(const std::string &s) {
    pyjson_value arr;
    arr.kind = pyjson_value::ARR;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (ws(s[i]) || s[i] == ',')) i++;
        if (i >= s.size()) break;
        const size_t before = i;
        arr.a.push_back(value_at(s, &i, ",]", false));
        if (i == before) break;
    }
    return arr;
}

/* where the name starts in a call body: past "call:" or the first ":" (vLLM's TOOL_PREAMBLE: the earliest of its
 * terminals "call:" and ":"), npos when neither has arrived */
/// @param b the body
/// @return the name's first byte, or npos
size_t name_start(const std::string &b) {
    const size_t c = b.find(':');
    return c == std::string::npos ? c : c + 1;
}

}  // namespace

const out_grammar &gemma4_output_grammar(void) {
    enum { CH_OPEN, CH_CLOSE, CALL_OPEN, CALL_CLOSE };
    static const out_grammar g = {
        /* .name        = */ "gemma4",
        /* .markers     = */ {"<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>"},
        /* .transitions = */ {
            {OUT_CONTENT, CH_OPEN, OUT_REASONING, 0},
            {OUT_REASONING, CH_OPEN, OUT_REASONING, 0},    /* a repeated <|channel>: dropped */
            {OUT_REASONING, CH_CLOSE, OUT_CONTENT, 0},
            {OUT_CONTENT, CH_CLOSE, OUT_CONTENT, 0},       /* a stray <channel|>: dropped */
            {OUT_REASONING, CALL_OPEN, OUT_CALL, 0},       /* a call ends the reasoning */
            {OUT_CONTENT, CALL_OPEN, OUT_CALL, 0},
            {OUT_CALL, CALL_CLOSE, OUT_CONTENT, 0},
            {OUT_CONTENT, CALL_CLOSE, OUT_CONTENT, 0},     /* back-to-back calls' stray closer: dropped */
        },
        /* .reasoning   = */ OUT_TEXT_RSTRIP,
        /* .content     = */ OUT_TEXT_TRIM_AT_CALLS,
        /* .call_open   = */ "<|tool_call>",
        /* .dropped     = */ {"<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<|tool>", "<tool|>", "<|tool_response>",
                              "<tool_response|>", "<|think|>", "<|turn>", "<turn|>", "<|image>", "<|audio>",
                              "<|image|>", "<|audio|>", "<image|>", "<audio|>", "<|video|>"},
        /* .reasoning_prefix = */ "thought\n",
    };
    return g;
}

bool gemma4_call_body::declared_string(const std::string &fn, const std::string &key) const {
    for (const pyjson_value &tool : tools.a) {
        const pyjson_value *f = tool.get("function");
        if (!f) f = &tool;
        const pyjson_value *name = f->get("name");
        if (!name || name->kind != pyjson_value::STR || name->s != fn) continue;
        const pyjson_value *params = f->get("parameters");
        const pyjson_value *props = params ? params->get("properties") : NULL;
        const pyjson_value *prop = props ? props->get(key.c_str()) : NULL;
        const pyjson_value *type = prop ? prop->get("type") : NULL;
        if (!type) return false;
        if (type->kind == pyjson_value::STR) return type->s == "string";
        if (type->kind == pyjson_value::ARR)
            for (const pyjson_value &t : type->a)
                if (t.kind == pyjson_value::STR && t.s == "string") return true;
        return false;
    }
    return false;
}

void gemma4_call_body::open() {
    began_ = false;
    args_lo_ = scan_ = 0;
    in_str_ = false;
}

void gemma4_call_body::grow(const std::string &b, size_t, bool, int index, std::vector<out_event> *ev) {
    if (!began_) {
        const size_t ns = name_start(b);
        if (ns == std::string::npos) return;
        const size_t brace = b.find_first_of("{(", ns);
        if (brace == std::string::npos) return;
        out_event e;
        e.kind = out_event::TOOL_BEGIN;
        e.index = index;
        e.name = strip(b.substr(ns, brace - ns));
        ev->push_back(std::move(e));
        began_ = true;
        args_lo_ = scan_ = brace + 1;
    }
    /* the payload: inside a <|"|> string of the arguments (a quote is one token, so never split; a tail that may
     * still become one waits) */
    while (scan_ < b.size()) {
        const size_t q = b.find(kQ, scan_);
        if (q == std::string::npos) {
            size_t keep = 0;
            for (size_t k = std::min(kQ.size() - 1, b.size() - scan_); k > 0; k--)
                if (!kQ.compare(0, k, b, b.size() - k, k)) { keep = k; break; }
            scan_ = b.size() - keep;
            break;
        }
        in_str_ = !in_str_;
        scan_ = q + kQ.size();
    }
}

bool gemma4_call_body::close(const std::string &b, int index, std::vector<out_event> *ev, out_call *call,
                             const char **why, size_t *at) {
    const size_t ns = name_start(b);
    *at = 0;
    if (ns == std::string::npos) {
        *why = "expected call:NAME";
        return false;
    }
    const size_t brace = b.find_first_of("{(", ns);
    const std::string name = strip(b.substr(ns, brace == std::string::npos ? std::string::npos : brace - ns));
    if (name.empty()) {
        *why = "no function name";
        *at = ns;
        return false;
    }
    pyjson_value args;
    args.kind = pyjson_value::OBJ;
    if (brace != std::string::npos) {
        /* vLLM's _gemma4_arg_converter: strip, drop the call's closer ("}" -- or ")" when ")" outnumber "("), parse */
        std::string text = strip(b.substr(brace + 1));
        size_t opens = 0, closes = 0;
        for (char c : text) opens += c == '(', closes += c == ')';
        if (!text.empty() && (text.back() == '}' || (text.back() == ')' && opens < closes))) text.pop_back();
        const string_keys declared = [this, &name](const std::string &key) { return declared_string(name, key); };
        args = parse_fields(text, &declared);
    }
    if (!began_) {
        out_event e;
        e.kind = out_event::TOOL_BEGIN;
        e.index = index;
        e.name = name;
        ev->push_back(std::move(e));
        began_ = true;
    }
    std::string json;
    pyjson_dump(args, &json);
    out_event e;
    e.kind = out_event::TOOL_ARGS;
    e.index = index;
    e.text = json.substr(0, json.size() - 1);   /* the object without its final "}" */
    ev->push_back(std::move(e));
    call->name = name;
    call->arguments = std::move(json);
    return true;
}

bool gemma4_output_parser::init(bool reasoning, const char *tools_json, char *err, size_t errlen) {
    body_.tools = pyjson_value();
    if (tools_json) {
        char jerr[160] = "not a JSON array";
        if (!pyjson_parse(tools_json, strlen(tools_json), &body_.tools, jerr, sizeof jerr) ||
            body_.tools.kind != pyjson_value::ARR) {
            if (err && errlen) snprintf(err, errlen, "tools_not_array: %s", jerr);
            return false;
        }
    }
    out_parser::init(&gemma4_output_grammar(), &body_, reasoning ? OUT_REASONING : OUT_CONTENT);
    return true;
}

out_parser *gemma4_output_parser::clone() const {
    gemma4_output_parser *p = new gemma4_output_parser(*this);
    p->rebind_body(&p->body_);
    return p;
}

bool gemma4_prompt_opens_reasoning(const char *prompt) {
    if (!prompt) return false;
    const std::string s = prompt;
    const size_t open = s.rfind("<|channel>");
    if (open == std::string::npos) return false;
    for (const char *m : {"<channel|>", "<|tool_call>", "<|turn>", "<|tool_response>"}) {
        const size_t at = s.rfind(m);
        if (at != std::string::npos && at > open) return false;
    }
    return true;
}

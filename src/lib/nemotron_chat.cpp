/* nemotron_chat.cpp -- the Nemotron-H renderer, effort authority and output-parser configuration.  See
 * nemotron_chat.h.  Every literal below is the checkpoint's chat_template.jinja (Nemotron 3.5 Super VL, sha256
 * in tests/test-vectors/nemotron_h/meta.json), in the template's order. */
#include "nemotron_chat.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "pulsar_utf8.h"

/* ---- effort ---------------------------------------------------------------- */

nemotron_effort nemotron_effort_default(void) { return NEMOTRON_EFFORT_HIGH; }

bool nemotron_effort_parse(const char *name, nemotron_effort *out) {
    static const struct { const char *name; nemotron_effort e; } names[] = {
        {"none", NEMOTRON_EFFORT_NONE}, {"medium", NEMOTRON_EFFORT_MEDIUM}, {"high", NEMOTRON_EFFORT_HIGH}};
    for (const auto &n : names)
        if (name && !strcmp(name, n.name)) {
            *out = n.e;
            return true;
        }
    return false;
}

const char *nemotron_effort_name(nemotron_effort e) {
    switch (e) {
    case NEMOTRON_EFFORT_NONE: return "none";
    case NEMOTRON_EFFORT_MEDIUM: return "medium";
    case NEMOTRON_EFFORT_HIGH: return "high";
    }
    return "?";
}

const char *nemotron_generation_prompt(nemotron_effort effort) {
    return effort == NEMOTRON_EFFORT_NONE ? "<|im_start|>assistant\n<think></think>" : "<|im_start|>assistant\n<think>\n";
}

namespace {

const char kMediumSuffix[] = "\n\n{reasoning effort: efficient}";
const char kToolsHead[] = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const char kToolsTail[] =
    "\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested "
    "within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, "
    "but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and "
    "do not tell the user about function calls\n"
    "</IMPORTANT>";

bool refuse(char *err, size_t errlen, const char *key, const char *fmt, ...) {
    if (err && errlen) {
        char msg[300];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(msg, sizeof msg, fmt, ap);
        va_end(ap);
        snprintf(err, errlen, "%s: %s", key, msg);
    }
    return false;
}

uint32_t cp_at(const unsigned char *p, int n) {
    switch (n) {
    case 1: return p[0];
    case 2: return ((uint32_t)(p[0] & 0x1f) << 6) | (p[1] & 0x3f);
    case 3: return ((uint32_t)(p[0] & 0x0f) << 12) | ((uint32_t)(p[1] & 0x3f) << 6) | (p[2] & 0x3f);
    default: return ((uint32_t)(p[0] & 0x07) << 18) | ((uint32_t)(p[1] & 0x3f) << 12) |
                    ((uint32_t)(p[2] & 0x3f) << 6) | (p[3] & 0x3f);
    }
}

/* Jinja |trim (Python str.strip) on UTF-8 bytes: the kept range [*lo, *hi) of s[0, n). */
void strip_bounds(const char *s, size_t n, size_t *lo_out, size_t *hi_out) {
    const unsigned char *u = (const unsigned char *)s;
    size_t lo = 0;
    while (lo < n) {
        const int k = utf8_seq_ok(u + lo, n - lo);
        if (!k || !py_isspace(cp_at(u + lo, k))) break;
        lo += (size_t)k;
    }
    size_t hi = n;
    while (hi > lo) {
        size_t b = hi - 1;
        while (b > lo && (u[b] & 0xc0) == 0x80 && hi - b < 4) b--;
        const int k = utf8_seq_ok(u + b, hi - b);
        if (!k || b + (size_t)k != hi || !py_isspace(cp_at(u + b, k))) break;
        hi = b;
    }
    *lo_out = lo;
    *hi_out = hi;
}

std::string py_strip(const std::string &s) {
    size_t lo = 0, hi = 0;
    strip_bounds(s.data(), s.size(), &lo, &hi);
    return s.substr(lo, hi - lo);
}

/* A piece of text with the client-data ranges inside it (offsets into `s`): what the template's string operations
 * (concatenation, split, trim) act on, carrying the spans along. */
struct piece {
    std::string s;
    std::vector<pulsar_text_span> spans;

    void lit(const std::string &t) { s += t; }
    void client(const std::string &t) {
        if (t.empty()) return;
        spans.push_back({(uint32_t)s.size(), (uint32_t)(s.size() + t.size())});
        s += t;
    }
    void add(const piece &p) {
        const uint32_t base = (uint32_t)s.size();
        for (const pulsar_text_span &sp : p.spans) spans.push_back({base + sp.lo, base + sp.hi});
        s += p.s;
    }
    /* [lo, hi) of this piece, its spans clipped */
    piece sub(size_t lo, size_t hi) const {
        piece out;
        out.s = s.substr(lo, hi - lo);
        for (const pulsar_text_span &sp : spans) {
            const size_t a = sp.lo > lo ? sp.lo : lo, b = sp.hi < hi ? sp.hi : hi;
            if (a < b) out.spans.push_back({(uint32_t)(a - lo), (uint32_t)(b - lo)});
        }
        return out;
    }
    piece trimmed() const {
        size_t lo = 0, hi = 0;
        strip_bounds(s.data(), s.size(), &lo, &hi);
        return sub(lo, hi);
    }
    bool contains(const char *t) const { return s.find(t) != std::string::npos; }
};

struct writer {
    nemotron_render_out *out;
    void lit(const char *t) { out->text += t; }
    void lit(const std::string &t) { out->text += t; }
    void client(const std::string &t) {
        if (t.empty()) return;
        const uint32_t lo = (uint32_t)out->text.size();
        out->text += t;
        out->spans.push_back({lo, (uint32_t)out->text.size()});
    }
    /* a piece, its spans merged with an abutting previous one (the spans stay ascending and disjoint) */
    void add(const piece &p) {
        const uint32_t base = (uint32_t)out->text.size();
        for (const pulsar_text_span &sp : p.spans) {
            const pulsar_text_span v = {base + sp.lo, base + sp.hi};
            if (!out->spans.empty() && out->spans.back().hi == v.lo) out->spans.back().hi = v.hi;
            else out->spans.push_back(v);
        }
        out->text += p.s;
    }
};

bool finish(const nemotron_render_out *out, char *err, size_t errlen) {
    if (out->text.size() > UINT32_MAX)
        return refuse(err, errlen, "prompt_too_large", "%zu bytes do not fit the 32-bit span map", out->text.size());
    return true;
}

/* ---- Python's str() / repr() of what json.loads produced: the template's `| string` ------------------------ */

void py_repr(const pyjson_value &v, std::string *out);

void py_repr_str(const std::string &s, std::string *out) {
    const bool dq = s.find('\'') != std::string::npos && s.find('"') == std::string::npos;
    const char q = dq ? '"' : '\'';
    out->push_back(q);
    const unsigned char *u = (const unsigned char *)s.data();
    for (size_t i = 0; i < s.size();) {
        const int k = utf8_seq_ok(u + i, s.size() - i);
        const uint32_t cp = k ? cp_at(u + i, k) : u[i];
        char esc[16] = "";
        if (cp == '\\') snprintf(esc, sizeof esc, "\\\\");
        else if (cp == (uint32_t)q) snprintf(esc, sizeof esc, "\\%c", q);
        else if (cp == '\n') snprintf(esc, sizeof esc, "\\n");
        else if (cp == '\r') snprintf(esc, sizeof esc, "\\r");
        else if (cp == '\t') snprintf(esc, sizeof esc, "\\t");
        else if (cp < 0x20 || cp == 0x7f || (cp >= 0x80 && cp < 0xa0)) snprintf(esc, sizeof esc, "\\x%02x", cp);
        if (esc[0]) out->append(esc);
        else out->append(s, i, k ? (size_t)k : 1);
        i += k ? (size_t)k : 1;
    }
    out->push_back(q);
}

void py_float(double f, std::string *out) {
    if (isinf(f)) {
        out->append(f > 0 ? "inf" : "-inf");
        return;
    }
    pyjson_value v;
    v.kind = pyjson_value::FLOAT;
    v.f = f;
    pyjson_dump(v, out);   /* repr(): the shortest round-trip digits, Python's notation */
}

void py_repr(const pyjson_value &v, std::string *out) {
    switch (v.kind) {
    case pyjson_value::NUL: out->append("None"); return;
    case pyjson_value::BOOL: out->append(v.b ? "True" : "False"); return;
    case pyjson_value::INT: out->append(v.num); return;
    case pyjson_value::FLOAT: py_float(v.f, out); return;
    case pyjson_value::STR: py_repr_str(v.s, out); return;
    case pyjson_value::ARR:
        out->push_back('[');
        for (size_t i = 0; i < v.a.size(); i++) {
            if (i) out->append(", ");
            py_repr(v.a[i], out);
        }
        out->push_back(']');
        return;
    case pyjson_value::OBJ:
        out->push_back('{');
        for (size_t i = 0; i < v.o.size(); i++) {
            if (i) out->append(", ");
            py_repr_str(v.o[i].first, out);
            out->append(": ");
            py_repr(v.o[i].second, out);
        }
        out->push_back('}');
        return;
    }
}

/* `x | string`: a string as itself, anything else its repr (str() of a number, bool, None, list, dict). */
std::string py_str(const pyjson_value &v) {
    if (v.kind == pyjson_value::STR) return v.s;
    std::string out;
    py_repr(v, &out);
    return out;
}

std::string tojson(const pyjson_value &v) {
    std::string out;
    pyjson_dump(v, &out);
    return out;
}

/* the template's render_extra_keys: every key of a mapping not in `handled`, as \n<key>value</key> -- a mapping or
 * a list as tojson, anything else as `| string`.  Keys and values are the client's. */
void extra_keys(writer &w, const pyjson_value *d, std::initializer_list<const char *> handled) {
    if (!d || d->kind != pyjson_value::OBJ) return;
    for (const auto &kv : d->o) {
        bool skip = false;
        for (const char *h : handled) skip |= kv.first == h;
        if (skip) continue;
        w.lit("\n<");
        w.client(kv.first);
        w.lit(">");
        const bool structured = kv.second.kind == pyjson_value::OBJ || kv.second.kind == pyjson_value::ARR;
        w.client(structured ? tojson(kv.second) : py_str(kv.second));
        w.lit("</");
        w.client(kv.first);
        w.lit(">");
    }
}

/* The tools block (one function per tool, the template's XML), its tool data client text throughout. */
void write_tools(writer &w, const pyjson_value &tools) {
    w.lit(kToolsHead);
    for (const pyjson_value &t : tools.a) {
        const pyjson_value *fn = t.get("function");
        const pyjson_value *tool = fn ? fn : &t;
        const pyjson_value *name = tool->get("name");
        w.lit("\n<function>\n<name>");
        if (name) w.client(py_str(*name));
        w.lit("</name>");
        if (const pyjson_value *d = tool->get("description")) {
            w.lit("\n<description>");
            w.client(py_strip(py_str(*d)));
            w.lit("</description>");
        }
        w.lit("\n<parameters>");
        const pyjson_value *params = tool->get("parameters");
        const pyjson_value *props = params ? params->get("properties") : NULL;
        if (props && props->kind == pyjson_value::OBJ) {
            for (const auto &p : props->o) {
                w.lit("\n<parameter>\n<name>");
                w.client(p.first);
                w.lit("</name>");
                const pyjson_value &f = p.second;
                if (const pyjson_value *ty = f.get("type")) {
                    w.lit("\n<type>");
                    w.client(py_str(*ty));
                    w.lit("</type>");
                }
                if (const pyjson_value *d = f.get("description")) {
                    w.lit("\n<description>");
                    w.client(py_strip(py_str(*d)));
                    w.lit("</description>");
                }
                if (const pyjson_value *en = f.get("enum")) {
                    w.lit("\n<enum>");
                    w.client(tojson(*en));
                    w.lit("</enum>");
                }
                extra_keys(w, &f, {"name", "type", "description", "enum"});
                w.lit("\n</parameter>");
            }
        }
        extra_keys(w, params, {"type", "properties", "required"});
        if (const pyjson_value *req = params && params->kind == pyjson_value::OBJ ? params->get("required") : NULL) {
            w.lit("\n<required>");
            w.client(tojson(*req));
            w.lit("</required>");
        }
        w.lit("\n</parameters>");
        extra_keys(w, tool, {"type", "name", "description", "parameters"});
        w.lit("\n</function>");
    }
    w.lit(kToolsTail);
}

/* One call: <tool_call>\n<function=NAME>\n, each argument <parameter=KEY>\nVALUE\n</parameter>\n (a string as itself,
 * a mapping or a list as tojson, anything else as `| string`), </function>\n</tool_call>\n. */
bool write_calls(writer &w, const nemotron_msg_in &m, int i, char *err, size_t errlen) {
    for (int c = 0; c < m.n_calls; c++) {
        const nemotron_tool_call_in &tc = m.calls[c];
        if (!tc.name || !tc.name[0])
            return refuse(err, errlen, "tool_name_missing", "message %d call %d has no function name", i, c);
        w.lit("<tool_call>\n<function=");
        w.client(tc.name);
        w.lit(">\n");
        if (tc.arguments && tc.arguments[0]) {
            pyjson_value args;
            char jerr[160];
            if (!pyjson_parse(tc.arguments, strlen(tc.arguments), &args, jerr, sizeof jerr))
                return refuse(err, errlen, "tool_arguments_not_json", "message %d call %d: arguments: %s", i, c, jerr);
            if (args.kind != pyjson_value::OBJ)
                return refuse(err, errlen, "tool_arguments_not_object",
                              "message %d call %d: arguments decode to a JSON %s, not an object (the template "
                              "renders an object's items)", i, c,
                              args.kind == pyjson_value::STR ? "string" : args.kind == pyjson_value::ARR ? "array"
                                                                                                       : "scalar");
            for (const auto &kv : args.o) {
                w.lit("<parameter=");
                w.client(kv.first);
                w.lit(">\n");
                const bool structured = kv.second.kind == pyjson_value::OBJ || kv.second.kind == pyjson_value::ARR;
                w.client(structured ? tojson(kv.second) : py_str(kv.second));
                w.lit("\n</parameter>\n");
            }
        }
        w.lit("</function>\n</tool_call>\n");
    }
    return true;
}

/* The assistant's content as the template builds it: "<think>\n" + reasoning + "</think>" + content when the
 * reasoning has non-space text, else "<think></think>" + content unless the content spells a think tag itself. */
piece assistant_content(const nemotron_msg_in &m) {
    piece c;
    const std::string reasoning = m.reasoning ? m.reasoning : "";
    const std::string content = m.content ? m.content : "";
    if (!py_strip(reasoning).empty()) {
        c.lit("<think>\n");
        c.client(reasoning);
        c.lit("</think>");
        c.client(content);
    } else {
        if (content.find("<think>") == std::string::npos && content.find("</think>") == std::string::npos)
            c.lit("<think></think>");
        c.client(content);
    }
    return c;
}

/* An assistant turn.  `truncate`: the turn is before the last user message (truncate_history_thinking). */
bool write_assistant(writer &w, const nemotron_msg_in &m, bool truncate, int i, char *err, size_t errlen) {
    const piece content = assistant_content(m);
    if (m.n_calls > 0 || (m.raw_calls && m.raw_calls[0])) {
        if (m.n_calls <= 0)
            return refuse(err, errlen, "raw_calls_without_calls", "message %d carries sampled call bytes but no call", i);
        w.lit("<|im_start|>assistant\n");
        if (!truncate) {
            w.add(content.trimmed());   /* never empty: it opens with a think tag or holds non-space text */
            w.lit("\n");
        } else {
            piece c = content;
            const size_t close = c.s.rfind("</think>");
            if (close != std::string::npos) {
                c = c.sub(close + 8, c.s.size());
            } else {
                const size_t open = c.s.find("<think>");
                if (open != std::string::npos) c = c.sub(0, open);
            }
            piece t;
            t.lit("<think></think>");
            t.add(c);
            w.add(t);
            w.lit("\n");
        }
        if (m.raw_calls && m.raw_calls[0]) {
            w.lit(m.raw_calls);   /* the model's own bytes, never client data */
            w.lit("\n");
        } else if (!write_calls(w, m, i, err, errlen)) {
            return false;
        }
        w.lit("<|im_end|>\n");
        return true;
    }
    if (!truncate) {
        w.lit("<|im_start|>assistant\n");
        w.add(content.trimmed());
        w.lit("<|im_end|>\n");
        return true;
    }
    piece c = content;
    if (c.contains("<think>") && c.contains("</think>")) {
        piece t;
        t.lit("<think></think>");
        t.add(c.sub(c.s.rfind("</think>") + 8, c.s.size()));
        c = t;
    }
    c = c.trimmed();
    w.lit("<|im_start|>assistant\n");
    w.add(c);
    w.lit("<|im_end|>\n");
    return true;
}

/* The loop body over `msgs`: `prev` = the role before msgs[0] (NULL: the conversation's first), `last_user` the
 * index in msgs of the last user message (-1: none), `after` the role after the last one (NULL: the conversation
 * ends there). */
bool write_messages(writer &w, const nemotron_msg_in *msgs, int n, const char *prev, int last_user,
                    nemotron_effort effort, char *err, size_t errlen) {
    for (int i = 0; i < n; i++) {
        const nemotron_msg_in &m = msgs[i];
        const char *role = m.role ? m.role : "";
        const std::string content = m.content ? m.content : "";
        if (!strcmp(role, "assistant")) {
            if (!write_assistant(w, m, i < last_user, i, err, errlen)) return false;
        } else if (!strcmp(role, "user") || !strcmp(role, "system")) {
            w.lit("<|im_start|>");
            w.lit(role);
            w.lit("\n");
            w.client(content);
            if (!strcmp(role, "user") && i == last_user && effort == NEMOTRON_EFFORT_MEDIUM) w.lit(kMediumSuffix);
            w.lit("<|im_end|>\n");
        } else if (!strcmp(role, "tool")) {
            const char *before = i ? msgs[i - 1].role : prev;
            if (before && strcmp(before, "tool")) w.lit("<|im_start|>user\n");
            w.lit("<tool_response>\n");
            w.client(content);
            w.lit("\n</tool_response>\n");
            const bool last = i + 1 == n;
            if (last || strcmp(msgs[i + 1].role ? msgs[i + 1].role : "", "tool")) w.lit("<|im_end|>\n");
        } else {
            w.lit("<|im_start|>");
            w.lit(role);   /* the API's role word: the template writes it, not client text */
            w.lit("\n");
            w.client(content);
            w.lit("<|im_end|>\n");
        }
    }
    return true;
}

int last_user_of(const nemotron_msg_in *msgs, int n) {
    int last = -1;
    for (int i = 0; i < n; i++)
        if (msgs[i].role && !strcmp(msgs[i].role, "user")) last = i;
    return last;
}

}  // namespace

bool nemotron_chat_render(const nemotron_render_in &in, nemotron_render_out *out, char *err, size_t errlen) {
    out->text.clear();
    out->spans.clear();
    writer w{out};
    if (in.n_msgs <= 0) return refuse(err, errlen, "no_messages", "No messages provided.");
    pyjson_value tools;
    if (in.tools_json) {
        char jerr[160];
        if (!pyjson_parse(in.tools_json, strlen(in.tools_json), &tools, jerr, sizeof jerr))
            return refuse(err, errlen, "tools_not_array", "tools: %s", jerr);
        if (tools.kind != pyjson_value::ARR) return refuse(err, errlen, "tools_not_array", "tools is not a JSON array");
    }
    const bool first_system = in.msgs[0].role && !strcmp(in.msgs[0].role, "system");
    const std::string system = first_system && in.msgs[0].content ? in.msgs[0].content : "";
    const nemotron_msg_in *loop = in.msgs + (first_system ? 1 : 0);
    const int n = in.n_msgs - (first_system ? 1 : 0);
    w.lit("<|im_start|>system\n");
    w.client(system);
    if (in.tools_json && !tools.a.empty()) {
        if (!system.empty()) w.lit("\n\n");
        write_tools(w, tools);
    }
    w.lit("<|im_end|>\n");
    if (!write_messages(w, loop, n, NULL, last_user_of(loop, n), in.effort, err, errlen)) return false;
    if (in.add_generation_prompt) w.lit(nemotron_generation_prompt(in.effort));
    return finish(out, err, errlen);
}

bool nemotron_chat_render_assistant_turn(const nemotron_msg_in &m, bool thinking, nemotron_render_out *out, char *err,
                                         size_t errlen) {
    out->text.clear();
    out->spans.clear();
    writer w{out};
    const std::string reasoning = m.reasoning ? m.reasoning : "";
    const bool has_reasoning = !py_strip(reasoning).empty();
    if (!thinking && has_reasoning)
        return refuse(err, errlen, "reasoning_without_thinking",
                      "a thinking-off turn cannot carry reasoning (the generation prompt closed its think block empty)");
    if (thinking && !has_reasoning)
        return refuse(err, errlen, "reasoning_missing",
                      "a thinking-on turn without reasoning replays as <think></think>, not as a continuation of the "
                      "generation prompt's <think>\\n");
    const std::string content = m.content ? m.content : "";
    if (!thinking && (content.find("<think>") != std::string::npos || content.find("</think>") != std::string::npos))
        return refuse(err, errlen, "think_tag_in_content",
                      "a thinking-off turn whose content spells a think tag replays without the empty think block");
    /* the full render's trimmed "<think>\n" R "</think>" C (or "<think></think>" C) minus the prompt's opening */
    const piece full = assistant_content(m).trimmed();
    const size_t skip = thinking ? 8 /* "<think>\n" */ : 15 /* "<think></think>" */;
    w.add(full.sub(skip, full.s.size()));
    if (m.n_calls > 0 || (m.raw_calls && m.raw_calls[0])) {
        if (m.n_calls <= 0)
            return refuse(err, errlen, "raw_calls_without_calls", "the turn carries sampled call bytes but no call");
        w.lit("\n");
        if (m.raw_calls && m.raw_calls[0]) {
            w.lit(m.raw_calls);
            w.lit("\n");
        } else if (!write_calls(w, m, 0, err, errlen)) {
            return false;
        }
    } else {
        w.lit("<|im_end|>\n");
    }
    return finish(out, err, errlen);
}

bool nemotron_chat_render_tail(const nemotron_msg_in *msgs, int n, nemotron_effort effort, nemotron_render_out *out,
                               char *err, size_t errlen) {
    out->text.clear();
    out->spans.clear();
    writer w{out};
    if (n <= 0) return refuse(err, errlen, "no_messages", "No messages provided.");
    w.lit("<|im_end|>\n");
    if (!write_messages(w, msgs, n, "assistant", last_user_of(msgs, n), effort, err, errlen)) return false;
    w.lit(nemotron_generation_prompt(effort));
    return finish(out, err, errlen);
}

bool nemotron_chat_render_head(const char *system, nemotron_render_out *out, char *err, size_t errlen) {
    out->text.clear();
    out->spans.clear();
    writer w{out};
    w.lit("<|im_start|>system\n");
    w.client(system ? system : "");
    w.lit("<|im_end|>\n");
    return finish(out, err, errlen);
}

bool nemotron_chat_render_turn(const nemotron_msg_in *msgs, int n, nemotron_effort effort, bool generation_prompt,
                               nemotron_render_out *out, char *err, size_t errlen) {
    out->text.clear();
    out->spans.clear();
    writer w{out};
    if (n <= 0 && !generation_prompt) return refuse(err, errlen, "no_messages", "No messages provided.");
    if (!write_messages(w, msgs, n, "assistant", last_user_of(msgs, n), effort, err, errlen)) return false;
    if (generation_prompt) w.lit(nemotron_generation_prompt(effort));
    return finish(out, err, errlen);
}

const char *nemotron_find_call_block(const char *p, const char **end) {
    static const char open[] = "<tool_call>", close[] = "</tool_call>";
    const char *start = strstr(p, open);
    if (!start) return NULL;
    const char *e = start;
    for (;;) {
        const char *c = strstr(e, close);
        if (!c) return NULL;
        e = c + (sizeof close - 1);
        const char *q = e;
        while (*q == '\n' || *q == ' ' || *q == '\t' || *q == '\r') q++;
        if (strncmp(q, open, sizeof open - 1)) break;
        e = q;
    }
    *end = e;
    return start;
}

std::string nemotron_rendered_system_text(const char *prompt) {
    static const char open[] = "<|im_start|>system\n";
    if (!prompt || strncmp(prompt, open, sizeof open - 1)) return std::string();
    const char *p = prompt + (sizeof open - 1);
    const char *end = strstr(p, "<|im_end|>");
    return end ? py_strip(std::string(p, (size_t)(end - p))) : std::string();
}

/* ---- the output parser: the kit in vLLM's nemotron_v3 configuration ---------------------------------------- */

const out_grammar &nemotron_output_grammar(void) {
    enum { THINK_OPEN, THINK_END, CALL_OPEN, CALL_CLOSE };
    static const out_grammar g = {
        /* .name        = */ "nemotron_h",
        /* .markers     = */ {"<think>", "</think>", "<tool_call>", "</tool_call>"},
        /* .transitions = */ {
            {OUT_REASONING, THINK_OPEN, OUT_REASONING, 0},   /* a repeated <think>: dropped */
            {OUT_REASONING, THINK_END, OUT_CONTENT, 0},
            {OUT_CONTENT, THINK_END, OUT_CONTENT, 0},        /* a stray </think> after the reasoning: dropped */
            {OUT_REASONING, CALL_OPEN, OUT_CALL, 0},         /* a call ends the reasoning */
            {OUT_CONTENT, CALL_OPEN, OUT_CALL, 0},
            {OUT_CALL, CALL_CLOSE, OUT_CONTENT, 0},
        },
        /* .reasoning   = */ OUT_TEXT_RSTRIP,
        /* .content     = */ OUT_TEXT_TRIM_AT_CALLS,
        /* .call_open   = */ "<tool_call>",
        /* .dropped     = */ {},
    };
    return g;
}

bool nemotron_output_parser::init(bool thinking, const char *tools_json, char *err, size_t errlen) {
    body_.tools = pyjson_value();
    body_.python_scalars = true;   /* the template writes `args_value | string`: True / False / None */
    if (tools_json) {
        char jerr[160] = "not a JSON array";
        if (!pyjson_parse(tools_json, strlen(tools_json), &body_.tools, jerr, sizeof jerr) ||
            body_.tools.kind != pyjson_value::ARR) {
            if (err && errlen) snprintf(err, errlen, "tools_not_array: %s", jerr);
            return false;
        }
    }
    out_parser::init(&nemotron_output_grammar(), &body_, thinking ? OUT_REASONING : OUT_CONTENT);
    return true;
}

out_parser *nemotron_output_parser::clone() const {
    nemotron_output_parser *p = new nemotron_output_parser(*this);
    p->rebind_body(&p->body_);
    return p;
}

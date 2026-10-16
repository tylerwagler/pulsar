/* inkling_chat.cpp -- the Inkling-Small renderer, effort authority and output grammar.  See inkling_chat.h.
 * Every literal and every rule below is vLLM's native Inkling renderer (vllm/renderers/inkling.py and
 * inkling_encoding.py) or its parser's transition table (vllm/parser/inkling.py), in vLLM's order, so a
 * refusal names what vLLM would have raised first. */
#include "inkling_chat.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <map>

#include "pulsar_utf8.h"
#include "pyjson.h"

/* ---- effort --------------------------------------------------------------------------------------- */

int inkling_effort_default(void) { return 90; }

int inkling_effort_named(const char *name) {
    static const struct { const char *name; int centi; } names[] = {
        {"none", 0}, {"minimal", 10}, {"low", 20}, {"medium", 70}, {"high", 90}, {"xhigh", 99}, {"max", 99},
    };
    for (const auto &n : names)
        if (name && !strcmp(name, n.name)) return n.centi;
    return -1;
}

/* f"{v:.2f}".rstrip("0").rstrip("."), and "0" -> "0.0" */
bool inkling_effort_text(int centi, std::string *out) {
    if (centi < 0 || centi > 99) return false;
    char b[8];
    snprintf(b, sizeof b, "0.%02d", centi);
    std::string t = b;
    while (t.back() == '0') t.pop_back();
    if (t.back() == '.') t.pop_back();
    *out = t == "0" ? "0.0" : t;
    return true;
}

/* ---- the renderer --------------------------------------------------------------------------------- */

namespace {

const char kMessageUser[] = "<|message_user|>";
const char kMessageModel[] = "<|message_model|>";
const char kMessageSystem[] = "<|message_system|>";
const char kMessageTool[] = "<|message_tool|>";
const char kContentText[] = "<|content_text|>";
const char kContentImage[] = "<|content_image|>";
const char kContentModelEndSampling[] = "<|content_model_end_sampling|>";
const char kContentThinking[] = "<|content_thinking|>";
const char kContentXml[] = "<|content_xml|>";
const char kContentInvokeToolJson[] = "<|content_invoke_tool_json|>";
const char kContentInvokeToolText[] = "<|content_invoke_tool_text|>";
const char kContentToolError[] = "<|content_tool_error|>";
const char kEndMessage[] = "<|end_message|>";

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

/* role -> its message marker (developer folds into system); NULL for a role vLLM does not take */
const char *role_marker(const char *role) {
    if (!role) return NULL;
    if (!strcmp(role, "user")) return kMessageUser;
    if (!strcmp(role, "assistant")) return kMessageModel;
    if (!strcmp(role, "system") || !strcmp(role, "developer")) return kMessageSystem;
    if (!strcmp(role, "tool")) return kMessageTool;
    return NULL;
}

struct writer {
    inkling_render_out *out;
    void put(const char *s) { out->text += s; }
    /* client bytes: recorded as a span */
    void client(const std::string &s) {
        if (s.empty()) return;
        const uint32_t lo = (uint32_t)out->text.size();
        out->text += s;
        const uint32_t hi = (uint32_t)out->text.size();
        if (!out->spans.empty() && out->spans.back().hi == lo) out->spans.back().hi = hi;
        else out->spans.push_back({lo, hi});
    }
    /* `marker` [author] `kind` body `<|end_message|>` -- _append_message; the author and body are client
     * text unless `trusted` */
    void block(const char *marker, const std::string &author, const char *kind, const std::string &body,
               bool trusted = false) {
        put(marker);
        if (trusted) out->text += author;
        else client(author);
        put(kind);
        if (trusted) out->text += body;
        else client(body);
        put(kEndMessage);
    }
};

/* str.strip() would leave nothing: empty or Python whitespace only (invalid UTF-8 is not blank) */
bool py_blank(const char *s) {
    const unsigned char *u = (const unsigned char *)s;
    const size_t n = strlen(s);
    for (size_t i = 0; i < n;) {
        const int k = utf8_seq_ok(u + i, n - i);
        if (!k) return false;
        uint32_t cp = k == 1 ? u[i] : (uint32_t)(u[i] & (0x7f >> k));
        for (int j = 1; j < k; j++) cp = (cp << 6) | (u[i + (size_t)j] & 0x3f);
        if (!py_isspace(cp)) return false;
        i += (size_t)k;
    }
    return true;
}

/* The tool_declare body: [{description, name, parameters, type}] per tool, canonical. */
bool tool_declare_json(const char *tools_json, std::string *out, char *err, size_t errlen) {
    pyjson_value tools;
    char jerr[160];
    if (!pyjson_parse(tools_json, strlen(tools_json), &tools, jerr, sizeof jerr) || tools.kind != pyjson_value::ARR)
        return refuse(err, errlen, "tools_not_array", "%s", jerr);
    pyjson_value specs;
    specs.kind = pyjson_value::ARR;
    for (size_t i = 0; i < tools.a.size(); i++) {
        const pyjson_value &tool = tools.a[i];
        if (tool.kind != pyjson_value::OBJ) return refuse(err, errlen, "tool_spec_invalid", "tool %zu is not an object", i);
        const pyjson_value *fn = tool.get("function");
        static const pyjson_value empty_obj = [] {
            pyjson_value v;
            v.kind = pyjson_value::OBJ;
            return v;
        }();
        const pyjson_value &f = fn ? *fn : empty_obj;
        if (f.kind != pyjson_value::OBJ) return refuse(err, errlen, "tool_spec_invalid", "tool %zu: function is not an object", i);
        const pyjson_value *name = f.get("name");
        if (!name) return refuse(err, errlen, "tool_spec_invalid", "tool %zu has no function.name", i);
        pyjson_value spec;
        spec.kind = pyjson_value::OBJ;
        const pyjson_value *desc = f.get("description");
        pyjson_value empty_str;
        empty_str.kind = pyjson_value::STR;
        spec.o.emplace_back("description", desc && pyjson_truthy(*desc) ? *desc : empty_str);
        spec.o.emplace_back("name", *name);
        const pyjson_value *params = f.get("parameters");
        spec.o.emplace_back("parameters", params && pyjson_truthy(*params) ? *params : empty_obj);
        const pyjson_value *type = tool.get("type");
        pyjson_value function_str;
        function_str.kind = pyjson_value::STR;
        function_str.s = "function";
        spec.o.emplace_back("type", type ? *type : function_str);   /* tool.get("type", "function") */
        specs.a.push_back(std::move(spec));
    }
    if (!pyjson_dump_canonical(specs, out)) return refuse(err, errlen, "json_not_finite", "a tool holds NaN or Infinity");
    return true;
}

/* A call's body: {"name":NAME,"args":CANONICAL} -- _tool_call_json */
bool tool_call_json(const inkling_tool_call_in &c, std::string *out, char *err, size_t errlen) {
    if (!c.name) return refuse(err, errlen, "tool_name_missing", "a tool call's function name is not a string");
    pyjson_value args;
    const char *raw = c.arguments ? c.arguments : "";
    if (py_blank(raw)) {   /* `json.loads(raw_args) if raw_args.strip() else {}` */
        args.kind = pyjson_value::OBJ;
    } else {
        char jerr[160];
        if (!pyjson_parse(raw, strlen(raw), &args, jerr, sizeof jerr))
            return refuse(err, errlen, "tool_arguments_not_json", "%s", jerr);
        if (args.kind != pyjson_value::OBJ)
            return refuse(err, errlen, "tool_arguments_not_object", "a call's arguments decode to a non-object");
    }
    out->append("{\"name\":");
    pyjson_dump_string(c.name, strlen(c.name), out);
    out->append(",\"args\":");
    if (!pyjson_dump_canonical(args, out)) return refuse(err, errlen, "json_not_finite", "a call's arguments hold NaN or Infinity");
    out->push_back('}');
    return true;
}

/* One message as render_inkling_messages writes it (its loop body after the effort block), with the
 * tool_call_id -> name map the earlier calls filled.  `end_sampling`: an assistant message closes with
 * <|content_model_end_sampling|> (false: a sampled tool-call turn, whose close is its tail's first bytes). */
struct message_writer {
    writer w;
    std::map<std::string, std::string> call_name_of_id;   /* tool_call_id -> name, filled as calls render */

    /* the ids only: a message before a turn, whose calls the turn's tool results may name */
    void note_calls(const inkling_msg_in &m) {
        if (!m.role || strcmp(m.role, "assistant")) return;
        for (int k = 0; k < m.n_calls; k++)
            if (m.calls[k].id && m.calls[k].id[0] && m.calls[k].name) call_name_of_id[m.calls[k].id] = m.calls[k].name;
    }

    bool message(const inkling_msg_in &m, int i, bool end_sampling, char *err, size_t errlen) {
        const char *marker = role_marker(m.role);
        if (!strcmp(m.role, "tool")) {
            std::string name = m.name && m.name[0] ? m.name : "";
            if (name.empty()) {
                auto it = call_name_of_id.find(m.tool_call_id ? m.tool_call_id : "");
                if (it != call_name_of_id.end()) name = it->second;
            }
            std::string text = m.content ? m.content : "";
            if (m.parts) {
                text.clear();
                for (int k = 0; k < m.n_parts; k++) {
                    if (m.parts[k].image)
                        return refuse(err, errlen, "tool_result_not_text", "a tool result holds an image");
                    text += m.parts[k].text ? m.parts[k].text : "";
                }
            }
            w.block(kMessageTool, name, kContentText, text);
            return true;
        }
        const bool assistant = !strcmp(m.role, "assistant");
        if (assistant && m.reasoning && m.reasoning[0]) w.block(kMessageModel, "", kContentThinking, m.reasoning);
        if (m.parts) {
            for (int k = 0; k < m.n_parts; k++) {
                if (m.parts[k].image) {
                    w.put(marker);
                    w.put(kContentImage);
                    w.put(kEndMessage);
                } else {
                    w.block(marker, "", kContentText, m.parts[k].text ? m.parts[k].text : "");
                }
            }
        } else if (m.content && m.content[0]) {
            w.block(marker, "", kContentText, m.content);
        }
        if (!assistant) return true;
        /* every call is read as vLLM reads it (its refusals, in its order) whether or not its sampled bytes
         * replay */
        std::vector<std::string> bodies((size_t)(m.n_calls > 0 ? m.n_calls : 0));
        for (int k = 0; k < m.n_calls; k++) {
            const inkling_tool_call_in &c = m.calls[k];
            if (!tool_call_json(c, &bodies[(size_t)k], err, errlen)) return false;
            if (c.id && c.id[0]) call_name_of_id[c.id] = c.name;
        }
        const size_t raw_len = m.raw_calls ? strlen(m.raw_calls) : 0;
        if (raw_len && m.n_calls <= 0)
            return refuse(err, errlen, "raw_calls_without_calls", "message %d carries sampled call bytes but no call", i);
        const char *end = NULL;
        if (raw_len && inkling_find_call_block(m.raw_calls, &end) == m.raw_calls && end == m.raw_calls + raw_len) {
            /* the calls as sampled (the model's own bytes, never client data), after the role marker the
             * generation prompt -- or, in a replay, the renderer -- wrote before the first one */
            w.put(kMessageModel);
            w.out->text.append(m.raw_calls, raw_len);
        } else {
            /* canonical -- also for sampled bytes that are not ONE run of call blocks (text sampled between two
             * calls): replayed, that text would follow the content a second time.  The turn re-renders, a
             * prefix miss, never a wrong prompt (parser_kit_finish's rule for a dropped call). */
            for (int k = 0; k < m.n_calls; k++) w.block(kMessageModel, m.calls[k].name, kContentInvokeToolJson, bodies[(size_t)k]);
        }
        if (end_sampling) w.put(kContentModelEndSampling);
        return true;
    }
};

bool check_role(const inkling_msg_in &m, int i, char *err, size_t errlen) {
    if (role_marker(m.role)) return true;
    return refuse(err, errlen, "unexpected_role", "message %d has role %s", i, m.role ? m.role : "(none)");
}

bool fits(const inkling_render_out *out, char *err, size_t errlen) {
    if (out->text.size() > UINT32_MAX) return refuse(err, errlen, "prompt_too_large", "the render passes 4 GiB");
    return true;
}

/* where one call block that starts at `q` (just past its role marker) ends: the author (no marker inside), the
 * call kind, the body up to the first <|end_message|> -- the parser's reading of it; NULL if `q` starts none */
const char *call_block_end(const char *q) {
    const char *kind = strstr(q, kContentInvokeToolJson);
    if (!kind) return NULL;
    for (const char *a = q; a < kind; a++)
        if (a[0] == '<' && a[1] == '|') return NULL;
    const char *close = strstr(kind + (sizeof kContentInvokeToolJson - 1), kEndMessage);
    return close ? close + (sizeof kEndMessage - 1) : NULL;
}

}  // namespace

const char *inkling_find_call_block(const char *p, const char **end) {
    static const size_t lm = sizeof kMessageModel - 1, lk = sizeof kContentInvokeToolJson - 1;
    for (const char *k = strstr(p, kContentInvokeToolJson); k; k = strstr(k + lk, kContentInvokeToolJson)) {
        /* the header this kind marker closes starts past the last marker before it (or at p); a call's header
         * follows <|message_model|> -- or starts p itself: a turn's sampled bytes begin there, right after the
         * generation prompt's marker */
        const char *start = k;
        while (start > p && !(start - p >= 2 && start[-2] == '|' && start[-1] == '>')) start--;
        if (start > p && !(start - p >= (ptrdiff_t)lm && !strncmp(start - lm, kMessageModel, lm))) continue;
        const char *e = call_block_end(start);
        if (!e) return NULL;   /* an unclosed block is not a key */
        while (!strncmp(e, kMessageModel, lm)) {
            const char *n = call_block_end(e + lm);
            if (!n) break;
            e = n;
        }
        *end = e;
        return start;
    }
    return NULL;
}

bool inkling_chat_render(const inkling_render_in &in, inkling_render_out *out, char *err, size_t errlen) {
    out->text.clear();
    out->spans.clear();
    message_writer mw{writer{out}, {}};
    writer &w = mw.w;
    std::string effort_text;
    if (!inkling_effort_text(in.effort, &effort_text))
        return refuse(err, errlen, "effort_out_of_range", "reasoning_effort must be in [0.0, 0.99]");
    bool effort_pending = true;

    if (in.tools_json) {
        std::string declare;
        if (!tool_declare_json(in.tools_json, &declare, err, errlen)) return false;
        if (declare != "[]") {   /* `if all_tools:` -- an empty list declares nothing */
            w.put(kMessageSystem);
            w.out->text += "tool_declare";
            w.put(kContentXml);
            w.client(declare);
            w.put(kEndMessage);
        }
    }
    for (int i = 0; i < in.n_msgs; i++) {
        const inkling_msg_in &m = in.msgs[i];
        if (!check_role(m, i, err, errlen)) return false;
        const bool system = !strcmp(m.role, "system") || !strcmp(m.role, "developer");
        if (!system && effort_pending) {
            w.block(kMessageSystem, "", kContentText, "Thinking effort level: " + effort_text, true);
            effort_pending = false;
        }
        if (!mw.message(m, i, true, err, errlen)) return false;
    }
    if (effort_pending) w.block(kMessageSystem, "", kContentText, "Thinking effort level: " + effort_text, true);
    if (in.add_generation_prompt) w.put(kMessageModel);
    return fits(out, err, errlen);
}

bool inkling_chat_render_turn(const inkling_msg_in *msgs, int n, int from, bool close_turn, bool add_generation_prompt,
                              inkling_render_out *out, char *err, size_t errlen) {
    out->text.clear();
    out->spans.clear();
    if (from < 0 || from > n || (from == n && !close_turn && !add_generation_prompt))
        return refuse(err, errlen, "no_messages", "the turn starts at message %d of %d", from, n);
    message_writer mw{writer{out}, {}};
    for (int i = 0; i < from; i++) mw.note_calls(msgs[i]);
    if (close_turn) mw.w.put(kContentModelEndSampling);
    for (int i = from; i < n; i++)
        if (!check_role(msgs[i], i, err, errlen) || !mw.message(msgs[i], i, true, err, errlen)) return false;
    if (add_generation_prompt) mw.w.put(kMessageModel);
    return fits(out, err, errlen);
}

std::string inkling_rendered_system_text(const char *prompt) {
    static const size_t ls = sizeof kMessageSystem - 1, le = sizeof kEndMessage - 1;
    static const char effort[] = "Thinking effort level: ";
    std::string out;
    const char *p = prompt ? prompt : "";
    while (!strncmp(p, kMessageSystem, ls)) {
        const char *close = strstr(p, kEndMessage);
        if (!close) break;
        const char *body = NULL;
        if (!strncmp(p + ls, kContentText, sizeof kContentText - 1)) body = p + ls + sizeof kContentText - 1;
        else if (!strncmp(p + ls, "tool_declare", 12) && !strncmp(p + ls + 12, kContentXml, sizeof kContentXml - 1))
            body = p + ls + 12 + sizeof kContentXml - 1;
        if (body && body <= close) {
            const std::string b(body, (size_t)(close - body));
            const bool effort_line = !b.compare(0, sizeof effort - 1, effort);
            if (!effort_line && !b.empty()) out += (out.empty() ? "" : "\n\n") + b;
        }
        p = close + le;
    }
    return out;
}

bool inkling_chat_render_assistant_turn(const inkling_msg_in &m, inkling_render_out *out, char *err, size_t errlen) {
    out->text.clear();
    out->spans.clear();
    if (!m.role || strcmp(m.role, "assistant"))
        return refuse(err, errlen, "unexpected_role", "a sampled turn is the assistant's");
    message_writer mw{writer{out}, {}};
    if (!mw.message(m, 0, m.n_calls <= 0, err, errlen)) return false;
    /* the full render's first block opens with the role marker the generation prompt already wrote */
    static const size_t lm = sizeof kMessageModel - 1;
    if (out->text.compare(0, lm, kMessageModel) != 0)
        return refuse(err, errlen, "empty_turn", "a turn with no reasoning, content or call renders no block, so "
                      "the generation prompt's <|message_model|> has no replay");
    out->text.erase(0, lm);
    for (pulsar_text_span &s : out->spans) {
        s.lo -= (uint32_t)lm;
        s.hi -= (uint32_t)lm;
    }
    return fits(out, err, errlen);
}

/* ---- the output grammar --------------------------------------------------------------------------- */

const out_grammar &inkling_output_grammar(void) {
    enum { MSG_MODEL, TEXT_START, THINK_START, END_MESSAGE, END_SAMPLING, TOOL_START, TOOL_TEXT, TOOL_ERROR };
    static const out_grammar g = {
        /* .name        = */ "inkling",
        /* .markers     = */ {kMessageModel, kContentText, kContentThinking, kEndMessage, kContentModelEndSampling,
                              kContentInvokeToolJson, kContentInvokeToolText, kContentToolError},
        /* .transitions = */ {
            /* between blocks / inside a text block */
            {OUT_CONTENT, MSG_MODEL, OUT_HEADER, 0},
            {OUT_CONTENT, TEXT_START, OUT_CONTENT, 0},
            {OUT_CONTENT, THINK_START, OUT_REASONING, 0},
            {OUT_CONTENT, TOOL_START, OUT_CALL, 0},
            {OUT_CONTENT, TOOL_TEXT, OUT_CONTENT, 0},     /* raw / error tool blocks render as visible text */
            {OUT_CONTENT, TOOL_ERROR, OUT_CONTENT, 0},
            {OUT_CONTENT, END_MESSAGE, OUT_CONTENT, 0},
            {OUT_CONTENT, END_SAMPLING, OUT_CONTENT, 0},
            /* the header: the optional name between the model-role and content-kind markers is metadata */
            {OUT_HEADER, MSG_MODEL, OUT_HEADER, 0},
            {OUT_HEADER, TEXT_START, OUT_CONTENT, 0},
            {OUT_HEADER, THINK_START, OUT_REASONING, 0},
            {OUT_HEADER, TOOL_START, OUT_CALL, 0},
            {OUT_HEADER, TOOL_TEXT, OUT_CONTENT, 0},
            {OUT_HEADER, TOOL_ERROR, OUT_CONTENT, 0},
            {OUT_HEADER, END_MESSAGE, OUT_CONTENT, OUT_ACT_HEADER_TEXT},   /* a block with no kind: text */
            {OUT_HEADER, END_SAMPLING, OUT_CONTENT, OUT_ACT_HEADER_TEXT},
            /* inside a thinking block */
            {OUT_REASONING, THINK_START, OUT_REASONING, 0},
            {OUT_REASONING, TOOL_START, OUT_CALL, 0},     /* a call opening while thinking is still open */
            {OUT_REASONING, END_MESSAGE, OUT_CONTENT, 0},
            {OUT_REASONING, END_SAMPLING, OUT_CONTENT, 0},
            /* a closed call returns to CONTENT: blocks of any kind may follow */
            {OUT_CALL, END_MESSAGE, OUT_CONTENT, 0},
            {OUT_CALL, END_SAMPLING, OUT_CONTENT, 0},
        },
        /* .reasoning   = */ OUT_TEXT_RSTRIP,
        /* .content     = */ OUT_TEXT_KEEP_UNLESS_BLANK,
        /* .call_open   = */ kContentInvokeToolJson,
        /* .dropped     = */ {"<|endoftext|>", kMessageUser, kMessageSystem, kMessageTool, kContentImage,
                              "<|content_audio_input|>", "<|audio|>", kContentXml, "<|begin_of_text|>",
                              "<|audio_end|>"},
    };
    return g;
}

void inkling_output_parser::init(out_state initial) { out_parser::init(&inkling_output_grammar(), &body_, initial); }

out_parser *inkling_output_parser::clone() const {
    inkling_output_parser *p = new inkling_output_parser(*this);
    p->rebind_body(&p->body_);
    return p;
}

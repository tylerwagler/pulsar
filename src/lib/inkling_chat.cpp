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

}  // namespace

bool inkling_chat_render(const inkling_render_in &in, inkling_render_out *out, char *err, size_t errlen) {
    out->text.clear();
    out->spans.clear();
    writer w{out};
    std::string effort_text;
    if (!inkling_effort_text(in.effort, &effort_text))
        return refuse(err, errlen, "effort_out_of_range", "reasoning_effort must be in [0.0, 0.99]");
    bool effort_pending = true;
    std::map<std::string, std::string> call_name_of_id;   /* tool_call_id -> name, filled as calls render */

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
        const char *marker = role_marker(m.role);
        if (!marker) return refuse(err, errlen, "unexpected_role", "message %d has role %s", i, m.role ? m.role : "(none)");
        const bool system = !strcmp(m.role, "system") || !strcmp(m.role, "developer");
        if (!system && effort_pending) {
            w.block(kMessageSystem, "", kContentText, "Thinking effort level: " + effort_text, true);
            effort_pending = false;
        }
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
            continue;
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
        if (assistant) {
            for (int k = 0; k < m.n_calls; k++) {
                const inkling_tool_call_in &c = m.calls[k];
                std::string body;
                if (!tool_call_json(c, &body, err, errlen)) return false;
                if (c.id && c.id[0]) call_name_of_id[c.id] = c.name;
                w.block(kMessageModel, c.name, kContentInvokeToolJson, body);
            }
            w.put(kContentModelEndSampling);
        }
    }
    if (effort_pending) w.block(kMessageSystem, "", kContentText, "Thinking effort level: " + effort_text, true);
    if (in.add_generation_prompt) w.put(kMessageModel);
    if (out->text.size() > UINT32_MAX) return refuse(err, errlen, "prompt_too_large", "the render passes 4 GiB");
    return true;
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

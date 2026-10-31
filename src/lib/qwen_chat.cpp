/** @file
 *  The Qwen3.8-Flash-Next renderer and reasoning-effort authority (the checkpoint's chat_template.jinja). */
/* qwen_chat.cpp -- the Qwen3.8-Flash-Next renderer and effort authority.
 * See qwen_chat.h.  Every literal below is the checkpoint's
 * chat_template.jinja (sha256 c3cf9e34...), and the order of the checks is the
 * template's, so a refusal names the exception HF would have raised first. */
#include "qwen_chat.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "pulsar_utf8.h"

/* ---- effort ---------------------------------------------------------------- */

qwen_effort qwen_effort_default(void) { return QWEN_EFFORT_XHIGH; }

bool qwen_effort_parse(const char *name, qwen_effort *out) {
    static const struct { const char *name; qwen_effort e; } names[] = {
        {"none", QWEN_EFFORT_NONE}, {"low", QWEN_EFFORT_LOW},
        {"medium", QWEN_EFFORT_MEDIUM}, {"xhigh", QWEN_EFFORT_XHIGH},
    };
    for (const auto &n : names)
        if (name && !strcmp(name, n.name)) { *out = n.e; return true; }
    return false;
}

const char *qwen_effort_name(qwen_effort e) {
    switch (e) {
    case QWEN_EFFORT_NONE: return "none";
    case QWEN_EFFORT_LOW: return "low";
    case QWEN_EFFORT_MEDIUM: return "medium";
    case QWEN_EFFORT_XHIGH: return "xhigh";
    }
    return "?";
}

bool qwen_effort_resolve(const char *name, int thinking, qwen_effort *out, char *err, size_t errlen) {
    if (thinking == 0) {
        *out = QWEN_EFFORT_NONE;
        return true;
    }
    if (!name) {
        *out = qwen_effort_default();
        return true;
    }
    if (qwen_effort_parse(name, out) && !(thinking == 1 && *out == QWEN_EFFORT_NONE)) return true;
    if (err && errlen)
        snprintf(err, errlen, "reasoning_effort: this model (Qwen3.8-Flash-Next) takes low, medium or xhigh "
                              "(default), or none for no thinking; got %.40s%s", name,
                 thinking == 1 ? " with thinking requested" : "");
    return false;
}

/* The template's reasoning_instructions for each effort ("" = none). */
/// @param e the effort
/// @return the instruction line the system block carries for `e` (static), "" for none
static const char *effort_line(qwen_effort e) {
    switch (e) {
    case QWEN_EFFORT_XHIGH:
        return "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
               "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
               "clarity in the final answer.";
    case QWEN_EFFORT_LOW:
        return "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
               "the conclusion without unnecessary elaboration.";
    case QWEN_EFFORT_MEDIUM:
    case QWEN_EFFORT_NONE:
        break;
    }
    return "";
}

/// The tools block's head, up to its "<tools>" (the template's literal).
static const char kToolsHead[] = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
/// The tools block's tail from "\n</tools>": the call-format instructions (the template's literal).
static const char kToolsTail[] =
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

namespace {


/// Write "<key>: <fmt...>" into `err` (when given) and fail.
/// @param err    the caller's error buffer, or NULL
/// @param errlen its size in bytes
/// @param key    the refusal's name
/// @param fmt    printf format of the detail
/// @return false, always
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

/// Appends to a render's output, recording client-data spans.
struct writer {
    /// The output being written (not owned).
    qwen_render_out *out;
    /// Append template text.
    /// @param s the literal
    void lit(const char *s) { out->text += s; }
    /// Append template text.
    /// @param s the literal
    void lit(const std::string &s) { out->text += s; }
    /* client data: recorded so the tokenizer keeps markers out of it */
    /// @param s the client's text (nothing is recorded when empty)
    void client(const std::string &s) {
        if (s.empty()) return;
        const uint32_t lo = (uint32_t)out->text.size();
        out->text += s;
        out->spans.push_back({lo, (uint32_t)out->text.size()});
    }
};

/// @param s the text
/// @param p a NUL-terminated prefix
/// @return true when `s` begins with `p`
bool starts_with(const std::string &s, const char *p) { return s.compare(0, strlen(p), p) == 0; }
/// @param s the text
/// @param p a NUL-terminated suffix
/// @return true when `s` ends with `p`
bool ends_with(const std::string &s, const char *p) {
    const size_t n = strlen(p);
    return s.size() >= n && s.compare(s.size() - n, n, p) == 0;
}

/// @param effort the turn's effort
/// @return the generation prompt: an empty closed think block for QWEN_EFFORT_NONE, an open one otherwise
const char *generation_prompt_text(qwen_effort effort) {
    return effort == QWEN_EFFORT_NONE ? "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                                      : "<|im_start|>assistant\n<think>\n";
}

/* The text fits the 32-bit span map, or the render is refused. */
/// @param out    the finished render
/// @param err    the error buffer
/// @param errlen its size
/// @return false (prompt_too_large) when the text passes UINT32_MAX bytes
bool finish(const qwen_render_out *out, char *err, size_t errlen) {
    if (out->text.size() > UINT32_MAX)
        return refuse(err, errlen, "prompt_too_large", "%zu bytes do not fit the 32-bit span map", out->text.size());
    return true;
}

/* An assistant turn's calls after its content: the template's separator before the first ("\n\n",
 * nothing when the content is empty), "\n" between them -- or, with raw_calls, the sampled bytes
 * verbatim after that separator (they are the model's own text, never client data). */
/// @param w             the output
/// @param m             the assistant message
/// @param content_empty its stripped content is empty (no separator before the first call)
/// @param i             the message's index, for refusals
/// @param err           the error buffer
/// @param errlen        its size
/// @return false on raw_calls_without_calls, tool_name_missing or tool_arguments_*
bool render_calls(writer &w, const qwen_msg_in &m, bool content_empty, int i, char *err, size_t errlen) {
    if (m.raw_calls && m.raw_calls[0]) {
        if (m.n_calls <= 0)
            return refuse(err, errlen, "raw_calls_without_calls", "message %d carries sampled call bytes but no call", i);
        if (!content_empty) w.lit("\n\n");
        w.lit(m.raw_calls);
        return true;
    }
    for (int c = 0; c < m.n_calls; c++) {
        const qwen_tool_call_in &tc = m.calls[c];
        if (c == 0) w.lit(content_empty ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function=");
        else w.lit("\n<tool_call>\n<function=");
        if (!tc.name || !tc.name[0])
            return refuse(err, errlen, "tool_name_missing", "message %d call %d has no function name", i, c);
        w.client(tc.name);
        w.lit(">\n");
        if (tc.arguments && tc.arguments[0]) {
            pyjson_value args;
            char jerr[160];
            if (!pyjson_parse(tc.arguments, strlen(tc.arguments), &args, jerr, sizeof jerr))
                return refuse(err, errlen, "tool_arguments_not_json",
                              "message %d call %d: arguments: %s", i, c, jerr);
            /* the template tests `arguments != ''` on the DECODED value */
            const bool empty_string = args.kind == pyjson_value::STR && args.s.empty();
            if (!empty_string) {
                if (args.kind != pyjson_value::OBJ)
                    return refuse(err, errlen, "tool_arguments_not_object",
                                  "message %d call %d: arguments decode to a JSON %s, not an object "
                                  "(the template renders an object's items)", i, c,
                                  args.kind == pyjson_value::STR ? "string" :
                                  args.kind == pyjson_value::ARR ? "array" : "scalar");
                for (const auto &kv : args.o) {
                    w.lit("<parameter=");
                    w.client(kv.first);
                    w.lit(">\n");
                    if (kv.second.kind == pyjson_value::STR) {
                        w.client(kv.second.s);
                    } else {
                        std::string dumped;
                        pyjson_dump(kv.second, &dumped);
                        w.client(dumped);
                    }
                    w.lit("\n</parameter>\n");
                }
            }
        }
        w.lit("</function>\n</tool_call>");
    }
    return true;
}

/* L268: a message's content as the template's render_content writes it -- the text between images client data,
 * each image the template's vision literal (no `Picture N: ` prefix: add_vision_id is off) -- then |trim over the
 * whole rendered content, as the template trims it.  The literal has no whitespace at either end, so the trim can
 * only reach text. */
/// The bytes one image renders as (template text, not client data).
const char kVisionLiteral[] = "<|vision_start|><|image_pad|><|vision_end|>";

/// @param w the output
/// @param m the message (content plus image_at offsets)
void content_with_images(writer &w, const qwen_msg_in &m) {
    const char *c = m.content ? m.content : "";
    const size_t n = strlen(c);
    std::string r;
    std::vector<std::pair<size_t, size_t>> lits;   /* the literals' ranges in r */
    size_t at = 0;
    for (int k = 0; k < m.n_images; k++) {
        const size_t off = m.image_at[k] < n ? m.image_at[k] : n;
        r.append(c + at, off - at);
        lits.push_back({r.size(), r.size() + sizeof kVisionLiteral - 1});
        r += kVisionLiteral;
        at = off;
    }
    r.append(c + at, n - at);
    size_t lo = 0, hi = 0;
    py_strip_bounds(r.data(), r.size(), &lo, &hi);
    size_t p = lo;
    for (const auto &l : lits) {
        if (l.first > p) w.client(r.substr(p, l.first - p));
        w.lit(r.substr(l.first, l.second - l.first));
        p = l.second;
    }
    if (hi > p) w.client(r.substr(p, hi - p));
}

/* One message's turn as the template's loop body writes it.  `prev` / `next` are the neighbouring
 * roles (NULL at either end): a tool result opens the user turn after a non-tool message and closes
 * it before one.  `first`: the conversation's first message -- the only place a system message may
 * stand.  `i` names the message in a refusal. */
/// @param w      the output
/// @param m      the message
/// @param prev   the role before it, or NULL
/// @param next   the role after it, or NULL
/// @param first  it is the conversation's first message
/// @param i      its index, for refusals
/// @param err    the error buffer
/// @param errlen its size
/// @return false on image_position, system_not_first, unexpected_role or a call refusal
bool render_msg(writer &w, const qwen_msg_in &m, const char *prev, const char *next, bool first, int i, char *err,
                size_t errlen) {
    const char *role = m.role ? m.role : "";
    const std::string content = py_strip(m.content);
    if (m.n_images > 0 && strcmp(role, "user") && strcmp(role, "tool"))
        return refuse(err, errlen, "image_position", "Message %d (%s) cannot contain images.", i, role);
    if (!strcmp(role, "system")) {
        if (!first) return refuse(err, errlen, "system_not_first", "System message must be at the beginning.");
    } else if (!strcmp(role, "user")) {
        w.lit("<|im_start|>user\n");
        if (m.n_images > 0) content_with_images(w, m);
        else w.client(content);
        w.lit("<|im_end|>\n");
    } else if (!strcmp(role, "assistant")) {
        /* preserve_thinking is left at the template's default (true):
         * every assistant turn carries its think block */
        w.lit("<|im_start|>assistant\n<think>\n");
        w.client(py_strip(m.reasoning));
        w.lit("\n</think>\n\n");
        w.client(content);
        if (!render_calls(w, m, content.empty(), i, err, errlen)) return false;
        w.lit("<|im_end|>\n");
    } else if (!strcmp(role, "tool")) {
        if (prev && strcmp(prev, "tool")) w.lit("<|im_start|>user");
        w.lit("\n<tool_response>\n");
        if (m.n_images > 0) content_with_images(w, m);
        else w.client(content);
        w.lit("\n</tool_response>");
        if (!next || strcmp(next, "tool")) w.lit("<|im_end|>\n");
    } else {
        return refuse(err, errlen, "unexpected_role", "Unexpected message role (%s).", role);
    }
    return true;
}

/* The head: the system block the template writes before the first message -- the effort line, the tools
 * block (`tools` NULL = none) and the system message's content (`system` NULL = none), or nothing at all when
 * there is no effort line, no tools and no system text. */
/* The tools block's entries: each tool on its own line, tojson -- the one encoding of a tool definition, in the head
 * and (L325) in place where a deferred tool is loaded. */
/// @param w     the output
/// @param tools the OpenAI tools array
void write_tool_entries(writer &w, const pyjson_value &tools) {
    for (const pyjson_value &tool : tools.a) {
        w.lit("\n");
        std::string dumped;
        pyjson_dump(tool, &dumped);
        w.client(dumped);
    }
}

/// Write the head (see the comment above write_tool_entries).
/// @param w      the output
/// @param tools  the tools array, or NULL
/// @param system the system message's content, or NULL
/// @param effort the effort (its effort line, if any)
void write_head(writer &w, const pyjson_value *tools, const char *system, qwen_effort effort) {
    const std::string instructions = effort_line(effort);
    const std::string content = py_strip(system);
    if (tools) {
        w.lit("<|im_start|>system\n");
        if (!instructions.empty()) w.lit(instructions + "\n\n");
        w.lit(kToolsHead);
        write_tool_entries(w, *tools);
        w.lit(kToolsTail);
        if (!content.empty()) {
            w.lit("\n\n");
            w.client(content);
        }
        w.lit("<|im_end|>\n");
    } else if (!content.empty()) {
        w.lit("<|im_start|>system\n");
        if (!instructions.empty()) w.lit(instructions + "\n\n");
        w.client(content);
        w.lit("<|im_end|>\n");
    } else if (!instructions.empty()) {
        w.lit("<|im_start|>system\n" + instructions + "<|im_end|>\n");
    }
}

/* `n` messages that follow a message of role `prev` (never the conversation's first), then the generation
 * prompt for `effort` when asked -- the full render's loop body, so a turn rendered here is the bytes the full
 * render writes for it. */
/// @param w                 the output
/// @param msgs              the messages
/// @param n                 their count
/// @param prev              the role of the message before msgs[0]
/// @param effort            the effort, for the generation prompt
/// @param generation_prompt end with the generation prompt
/// @param err               the error buffer
/// @param errlen            its size
/// @return false on a refusal from render_msg
bool write_turn(writer &w, const qwen_msg_in *msgs, int n, const char *prev, qwen_effort effort,
                bool generation_prompt, char *err, size_t errlen) {
    for (int k = 0; k < n; k++)
        if (!render_msg(w, msgs[k], k ? msgs[k - 1].role : prev, k + 1 < n ? msgs[k + 1].role : NULL, false, k, err,
                        errlen))
            return false;
    if (generation_prompt) w.lit(generation_prompt_text(effort));
    return true;
}

}  // namespace

bool qwen_chat_render(const qwen_render_in &in, qwen_render_out *out, char *err, size_t errlen) {
    out->text.clear();
    out->spans.clear();
    writer w{out};
    if (in.n_msgs <= 0) return refuse(err, errlen, "no_messages", "No messages provided.");

    pyjson_value tools;
    if (in.tools_json) {
        char jerr[160];
        if (!pyjson_parse(in.tools_json, strlen(in.tools_json), &tools, jerr, sizeof jerr))
            return refuse(err, errlen, "tools_not_array", "tools: %s", jerr);
        if (tools.kind != pyjson_value::ARR)
            return refuse(err, errlen, "tools_not_array", "tools is not a JSON array");
    }
    const qwen_msg_in &first = in.msgs[0];
    const bool first_system = first.role && !strcmp(first.role, "system");
    if (first_system && first.n_images > 0)
        return refuse(err, errlen, "image_position", "System message cannot contain images.");
    write_head(w, in.tools_json && !tools.a.empty() ? &tools : NULL, first_system ? first.content : NULL, in.effort);

    /* the template's multi_step_tool scan: some user message must be a real
     * query, not a bare <tool_response> wrapper */
    bool found_query = false;
    for (int i = in.n_msgs - 1; i >= 0 && !found_query; i--) {
        const qwen_msg_in &m = in.msgs[i];
        if (!m.role || strcmp(m.role, "user")) continue;
        const std::string content = py_strip(m.content);
        if (!(starts_with(content, "<tool_response>") && ends_with(content, "</tool_response>")))
            found_query = true;
    }
    if (!found_query) return refuse(err, errlen, "no_user_query", "No user query found in messages.");

    for (int i = 0; i < in.n_msgs; i++)
        if (!render_msg(w, in.msgs[i], i ? in.msgs[i - 1].role : NULL, i + 1 < in.n_msgs ? in.msgs[i + 1].role : NULL,
                        i == 0, i, err, errlen))
            return false;
    if (in.add_generation_prompt) w.lit(generation_prompt_text(in.effort));
    return finish(out, err, errlen);
}

bool qwen_chat_tool_defs(const char *tools_json, qwen_render_out *out, char *err, size_t errlen) {
    out->text.clear();
    out->spans.clear();
    writer w{out};
    pyjson_value tools;
    char jerr[160];
    if (!tools_json || !pyjson_parse(tools_json, strlen(tools_json), &tools, jerr, sizeof jerr))
        return refuse(err, errlen, "tools_not_array", "tools: %s", tools_json ? jerr : "none");
    if (tools.kind != pyjson_value::ARR) return refuse(err, errlen, "tools_not_array", "tools is not a JSON array");
    w.lit("<tools>");
    write_tool_entries(w, tools);
    w.lit("\n</tools>");
    return finish(out, err, errlen);
}

bool qwen_chat_render_assistant_turn(const qwen_msg_in &m, bool thinking, qwen_render_out *out, char *err,
                                     size_t errlen) {
    out->text.clear();
    out->spans.clear();
    writer w{out};
    const std::string reasoning = py_strip(m.reasoning);
    if (thinking) {
        w.client(reasoning);
        w.lit("\n</think>\n\n");
    } else if (!reasoning.empty()) {
        return refuse(err, errlen, "reasoning_without_thinking",
                      "a thinking-off turn cannot carry reasoning (the generation prompt closed its think block empty)");
    }
    const std::string content = py_strip(m.content);
    w.client(content);
    if (!render_calls(w, m, content.empty(), 0, err, errlen)) return false;
    if (m.n_calls <= 0) w.lit("<|im_end|>\n");
    return finish(out, err, errlen);
}

bool qwen_chat_render_tail(const qwen_msg_in *msgs, int n, qwen_effort effort, qwen_render_out *out, char *err,
                           size_t errlen) {
    out->text.clear();
    out->spans.clear();
    writer w{out};
    if (n <= 0) return refuse(err, errlen, "no_messages", "No messages provided.");
    w.lit("<|im_end|>\n");
    if (!write_turn(w, msgs, n, "assistant", effort, true, err, errlen)) return false;
    return finish(out, err, errlen);
}

bool qwen_chat_render_head(const char *system, qwen_effort effort, qwen_render_out *out, char *err, size_t errlen) {
    out->text.clear();
    out->spans.clear();
    writer w{out};
    write_head(w, NULL, system, effort);
    return finish(out, err, errlen);
}

bool qwen_chat_render_turn(const qwen_msg_in *msgs, int n, qwen_effort effort, bool generation_prompt,
                           qwen_render_out *out, char *err, size_t errlen) {
    out->text.clear();
    out->spans.clear();
    writer w{out};
    if (n <= 0 && !generation_prompt) return refuse(err, errlen, "no_messages", "No messages provided.");
    if (!write_turn(w, msgs, n, "assistant", effort, generation_prompt, err, errlen)) return false;
    return finish(out, err, errlen);
}

std::string qwen_system_reminder(const char *content) {
    return std::string("<system-reminder>\n") + (content ? content : "") + "\n</system-reminder>";
}

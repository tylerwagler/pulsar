/* qwen_chat.h -- the Qwen3.8-Flash-Next chat format (L251 S5): the one
 * renderer, the reasoning-effort authority and the output parser.
 *
 * RENDERER.  `qwen_chat_render` writes the text HF's
 * `apply_chat_template(messages, tools=..., reasoning_effort=...,
 * enable_thinking=..., add_generation_prompt=...)` writes with the
 * checkpoint's chat_template.jinja, byte for byte, and records the byte
 * ranges it copied from CLIENT data (the L223 span map; the tokenizer keeps
 * markers out of them).  It is the ONE rendering authority for this family
 * (L185): the server, the CLI and any continuation suffix go through it.
 * Every exception the template raises is a refusal here, by name, and so is
 * a request the template cannot express:
 *
 *   no_messages, system_not_first, unexpected_role, no_user_query -- the
 *       template's own raise_exception calls;
 *   tool_arguments_not_json -- a call's `arguments` (the API's JSON TEXT) does
 *       not parse;
 *   tool_arguments_not_object -- it parses to something the template cannot
 *       iterate with `|items` (a string -- a DOUBLE-encoded object, 89 calls
 *       in the L216 corpus -- or an array).  HF raises "Can only get item
 *       pairs from a mapping"; decoding twice would be a guess;
 *   tools_not_array -- the tools value is not a JSON array (a caller maps an
 *       absent or JSON-null tools field to NULL; HF skips those too);
 *   tool_name_missing -- a call without a function name (HF raises on it);
 *   prompt_too_large -- the text does not fit the 32-bit span map.
 *
 * The API carries a call's arguments as JSON text and HF's template wants the
 * object, so the text is decoded ONCE (what vLLM does before rendering; the
 * 1,471 L216 corpus cases HF "failed to render" were exactly this).
 *
 * EFFORT.  The template knows three efforts and thinking-off:
 * xhigh (its default; a long instruction line), medium (thinking on, NO
 * line), low (a short line), and enable_thinking=false (no line, and the
 * generation prompt closes an empty think block).  `qwen_effort_default` is
 * the family's one default -- the template's own `default('xhigh')` -- and
 * the only way an unspecified effort resolves.
 *
 * OUTPUT PARSER.  `qwen_output_parser` splits sampled text into reasoning,
 * content and tool calls, incrementally (any chunking gives the same events)
 * and in the template's normal form -- reasoning and content stripped the
 * way the template strips them on replay -- so a turn the parser read
 * re-renders to the bytes the model sampled.  See the class. */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "pulsar.h"
#include "pyjson.h"

enum qwen_effort {
    QWEN_EFFORT_NONE,    ///< enable_thinking=false
    QWEN_EFFORT_LOW,
    QWEN_EFFORT_MEDIUM,
    QWEN_EFFORT_XHIGH,
};

/** The family's default effort: the template's `reasoning_effort|default('xhigh')`. */
qwen_effort qwen_effort_default(void);
/** "none", "low", "medium", "xhigh" -> the effort; false for anything else
 * (the template raises on any other name; "none" is thinking-off). */
bool qwen_effort_parse(const char *name, qwen_effort *out);
const char *qwen_effort_name(qwen_effort e);
/** THE resolution every front end uses for a Qwen request.  `name` is the
 * client's reasoning_effort (NULL = not sent); `thinking` its thinking switch
 * (enable_thinking / think / thinking: -1 not sent, 0 off, 1 on).  Thinking
 * off is QWEN_EFFORT_NONE whatever the name (the template ignores the effort
 * then); otherwise an unsent name is qwen_effort_default() and a sent one must
 * be a name the template knows -- "high", "max", "minimal" or a number are
 * refused with the list, never mapped to a neighbour. */
bool qwen_effort_resolve(const char *name, int thinking, qwen_effort *out, char *err, size_t errlen);

/** One tool call of an assistant message, as the API carries it. */
struct qwen_tool_call_in {
    const char *name;       ///< function name
    const char *arguments;  ///< the arguments object as JSON text; NULL or "" = none
};

/** One message.  `content` NULL renders as empty (the template's `none`). */
struct qwen_msg_in {
    const char *role;       ///< system | user | assistant | tool
    const char *content;
    const char *reasoning;  ///< assistant reasoning_content; NULL = none
    const qwen_tool_call_in *calls;
    int n_calls;
};

struct qwen_render_in {
    const qwen_msg_in *msgs;
    int n_msgs;
    const char *tools_json;   ///< the request's tools array as JSON text; NULL = none
    qwen_effort effort;
    bool add_generation_prompt;
};

struct qwen_render_out {
    std::string text;
    std::vector<pulsar_text_span> spans;   ///< client-data ranges, ascending, disjoint
};

/** Render.  false + "<refusal key>: detail" on refusal. */
bool qwen_chat_render(const qwen_render_in &in, qwen_render_out *out, char *err, size_t errlen);

/** Python's str.strip() whitespace (what Jinja's |trim removes): the one
 * definition the renderer and the output parser share. */
bool qwen_py_isspace(uint32_t cp);

/** One thing the output parser recognised. */
struct qwen_out_event {
    enum kind_t {
        REASONING,   ///< `text` is a reasoning delta
        CONTENT,     ///< `text` is a content delta
        TOOL_BEGIN,  ///< call `index` named `name` opened
        TOOL_END,    ///< call `index` complete: `name`, `arguments` (a JSON object's text)
        ERROR,       ///< `text` says what was malformed; the call it concerns is dropped
    } kind;
    std::string text;
    int index = -1;
    std::string name;
    std::string arguments;
};

/** One complete tool call the parser read. */
struct qwen_out_call {
    std::string name;
    std::string arguments;   ///< a JSON object, Python-dumps spelling
};

/** The sampled-text parser for one assistant turn.
 *
 * Format (the template's, and the tools prompt it gives the model):
 * @verbatim
   [reasoning] </think> [content] <tool_call>\n<function=NAME>\n
   <parameter=KEY>\nVALUE\n</parameter>\n ... </function>\n</tool_call> ...
   @endverbatim
 * With thinking on the generation prompt already opened `<think>\n`, so the
 * turn starts in reasoning; with thinking off it closed an empty block and the
 * turn starts in content.  Inside reasoning nothing but `</think>` is
 * structure.  A parameter VALUE is the text between the tags minus one
 * leading and one trailing newline (the template writes exactly those).
 *
 * Arguments are rebuilt as JSON.  The template wrote strings raw and every
 * other value with tojson, so a value is JSON exactly when (a) the tool schema
 * does not declare the property a string (type "string" or a type list that
 * includes it) and (b) the text parses AND is the tojson spelling of what it
 * parses to; anything else is the text as a string.  Either way the value
 * re-renders to the bytes the model sampled.
 *
 * Reasoning and content are reported STRIPPED (the template strips both on
 * replay); whitespace is held back until the next non-space proves it
 * interior.  Text after a tool call is content again (the prompt forbids it;
 * the client still sees it).  A malformed or unterminated call is an ERROR
 * event naming the fault, and is not reported as a call. */
class qwen_output_parser {
public:
    /** `thinking`: the generation prompt opened a think block.  `tools_json`:
     * the request's tools (NULL for none) -- the schema that types arguments.
     * false + reason if tools_json is not a JSON array. */
    bool init(bool thinking, const char *tools_json, char *err, size_t errlen);
    void feed(const char *p, size_t n, std::vector<qwen_out_event> *ev);
    /** End of the turn (a stop token, or the length cap). */
    void finish(std::vector<qwen_out_event> *ev);

    const std::string &reasoning() const { return reasoning_; }
    const std::string &content() const { return content_; }
    const std::vector<qwen_out_call> &calls() const { return calls_; }
    int errors() const { return errors_; }
    /** The parser sits inside an open <tool_call> block: its arguments decode greedily (L272 B8). */
    bool in_tool_call() const { return mode_ == M_TOOL; }

private:
    enum mode_t { M_REASONING, M_CONTENT, M_TOOL, M_AFTER_TOOL };
    struct section {           /* a stripped text stream */
        bool started = false;  ///< a non-space codepoint went out
        std::string ws;        ///< interior whitespace held back
        std::string partial;   ///< an incomplete UTF-8 sequence
    };
    void emit_text(bool reasoning, const char *p, size_t n, std::vector<qwen_out_event> *ev);
    void end_section(bool reasoning);
    void close_call(std::vector<qwen_out_event> *ev);
    const char *param_type(const std::string &fn, const std::string &key) const;

    mode_t mode_ = M_CONTENT;
    std::string hold_;         ///< bytes not yet classified
    section sec_[2];           ///< [0] content, [1] reasoning
    std::string block_;        ///< the open tool-call block body
    bool began_ = false;       ///< TOOL_BEGIN sent for the open call
    int index_ = 0;
    int errors_ = 0;
    std::string reasoning_, content_;
    std::vector<qwen_out_call> calls_;
    pyjson_value tools_;       ///< the request's tools, for typing arguments
};

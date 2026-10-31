/** @file
 *  The Qwen3.8-Flash-Next chat format: renderer, reasoning-effort authority and output parser. */
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
 *   prompt_too_large -- the text does not fit the 32-bit span map;
 *   raw_calls_without_calls -- a message carries sampled call bytes
 *       (qwen_msg_in::raw_calls) but no call;
 *   reasoning_without_thinking -- qwen_chat_render_assistant_turn only: a
 *       thinking-off turn with reasoning.
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
 * OUTPUT PARSER.  `qwen_output_parser` is the output-parser kit (L295 F5,
 * out_parse_kit.h) in Qwen's configuration -- its grammar and its XML call
 * body -- and splits sampled text into reasoning, content and tool calls,
 * incrementally (any chunking gives the same events) and in the template's
 * normal form -- reasoning and content stripped the way the template strips
 * them on replay -- so a turn the parser read re-renders to the bytes the
 * model sampled. */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "out_parse_kit.h"
#include "pulsar.h"
#include "pyjson.h"

/// The reasoning-effort levels the Qwen template can express (its reasoning_effort names; NONE = thinking off).
enum qwen_effort {
    QWEN_EFFORT_NONE,    ///< enable_thinking=false
    /// reasoning_effort=low (the template's short-thinking instruction line)
    QWEN_EFFORT_LOW,
    /// reasoning_effort=medium (no instruction line)
    QWEN_EFFORT_MEDIUM,
    /// reasoning_effort=xhigh, the template's default (the careful-thinking instruction line)
    QWEN_EFFORT_XHIGH,
};

/** The family's default effort: the template's `reasoning_effort|default('xhigh')`. */
/// @return QWEN_EFFORT_XHIGH
qwen_effort qwen_effort_default(void);
/** "none", "low", "medium", "xhigh" -> the effort; false for anything else
 * (the template raises on any other name; "none" is thinking-off). */
/// @param name the effort's name (may be NULL)
/// @param out  receives the effort when `name` is known
/// @return true when `name` is "none", "low", "medium" or "xhigh"
bool qwen_effort_parse(const char *name, qwen_effort *out);
/// The effort's name, as qwen_effort_parse reads it.
/// @param e the effort
/// @return "none", "low", "medium" or "xhigh" ("?" for an out-of-range value)
const char *qwen_effort_name(qwen_effort e);
/** The template's own variables resolved (the CLI tools and the engine's default prompt; the server maps
 * every protocol's effort names onto these first -- chat_family.cpp's one effort table).  `name` is a
 * template name (NULL = not sent); `thinking` the switch (-1 not sent, 0 off, 1 on).  Thinking off is
 * QWEN_EFFORT_NONE whatever the name; otherwise an unsent name is qwen_effort_default() and a sent one
 * must be one the template knows; thinking on with "none" is refused. */
/// @param name     the template effort name, or NULL when not sent
/// @param thinking the thinking switch: -1 not sent, 0 off, 1 on
/// @param out      receives the resolved effort
/// @param err      receives the refusal reason (may be NULL)
/// @param errlen   size of `err`
/// @return true on success, false on refusal
bool qwen_effort_resolve(const char *name, int thinking, qwen_effort *out, char *err, size_t errlen);

/** One tool call of an assistant message, as the API carries it. */
struct qwen_tool_call_in {
    const char *name;       ///< function name
    const char *arguments;  ///< the arguments object as JSON text; NULL or "" = none
};

/** One message.  `content` NULL renders as empty (the template's `none`). */
struct qwen_msg_in {
    const char *role;       ///< system | user | assistant | tool
    /// The message text (client data); NULL renders as empty.  Stripped (Jinja |trim) when rendered.
    const char *content;
    const char *reasoning;  ///< assistant reasoning_content; NULL = none
    /// The assistant message's tool calls (n_calls of them); NULL when none.
    const qwen_tool_call_in *calls;
    /// Number of entries in `calls`.
    int n_calls;
    /** The calls' SAMPLED bytes, from the first "<tool_call>" to the last "</tool_call>" (the server's
     * tool memory, L272 P3): written verbatim after the template's separator in place of rendering
     * `calls`, so a replayed turn byte-matches the live KV.  NULL = render `calls`. */
    const char *raw_calls;
    /** L268: the byte offsets in `content` (ascending) where the message's images sit -- the template's
     *  render_content writes each as <|vision_start|><|image_pad|><|vision_end|> there, outside client data.
     *  NULL / 0 = none.  An image in a system or assistant message is refused, as the template refuses it. */
    const uint32_t *image_at;
    /// Number of entries in `image_at`.
    int n_images;
};

/// A full render's request: the conversation, its tools and the effort.
struct qwen_render_in {
    /// The conversation's messages, in order.
    const qwen_msg_in *msgs;
    /// Number of entries in `msgs` (0 is refused as no_messages).
    int n_msgs;
    const char *tools_json;   ///< the request's tools array as JSON text; NULL = none
    /// The thinking level (the effort line and the generation prompt follow it).
    qwen_effort effort;
    /// End with the generation prompt for `effort`.
    bool add_generation_prompt;
};

/// A render's result: the prompt text and the client-data span map.
struct qwen_render_out {
    /// The rendered text (at most UINT32_MAX bytes, else prompt_too_large).
    std::string text;
    std::vector<pulsar_text_span> spans;   ///< client-data ranges, ascending, disjoint
};

/** Render.  false + "<refusal key>: detail" on refusal. */
/// @param in     the conversation, tools and effort
/// @param out    receives the text and its client-span map
/// @param err    receives "<refusal key>: detail" on refusal
/// @param errlen size of `err`
/// @return true on success, false on refusal
bool qwen_chat_render(const qwen_render_in &in, qwen_render_out *out, char *err, size_t errlen);

/** L325: tool definitions in the tools block's encoding -- "<tools>", each tool on its own line as tojson,
 *  "\n</tools>" -- for a tool loaded mid-conversation, rendered in place inside the turn that loaded it (the
 *  server writes it into that tool result's text).  `tools_json` is an OpenAI tools array.  Refusal:
 *  tools_not_array. */
/// @param tools_json the OpenAI tools array as JSON text
/// @param out        receives the tools block's text and its client-span map
/// @param err        receives "<refusal key>: detail" on refusal
/// @param errlen     size of `err`
/// @return true on success, false on refusal
bool qwen_chat_tool_defs(const char *tools_json, qwen_render_out *out, char *err, size_t errlen);

/** The assistant turn `m` as the model SAMPLES it: the bytes the full render writes for the turn past
 * the generation prompt -- past "<|im_start|>assistant\n<think>\n" with `thinking`, past the closed
 * empty think block without it -- so prefix + this = the full render of the history with the turn
 * appended (L272 P3; the server's KV key and Responses' visible memory).  With thinking: the stripped
 * reasoning, "\n</think>\n\n"; then the stripped content, the calls (or `raw_calls`), and
 * "<|im_end|>\n" when the turn has no calls -- a tool-call turn's close belongs to the tail that
 * follows it (the live KV ends before the stop token either way; see qwen_chat_render_tail).
 * Refusals: reasoning_without_thinking (a thinking-off turn cannot carry reasoning: its block was
 * closed empty by the prompt), and render_calls' (tool_name_missing, tool_arguments_*). */
/// @param m        the assistant message
/// @param thinking the generation prompt opened a think block
/// @param out      receives the turn's bytes and their client-span map
/// @param err      receives "<refusal key>: detail" on refusal
/// @param errlen   size of `err`
/// @return true on success, false on refusal
bool qwen_chat_render_assistant_turn(const qwen_msg_in &m, bool thinking, qwen_render_out *out, char *err,
                                     size_t errlen);

/** A continuation tail appended to a live KV that ends where the model stopped (before the stop
 * token): the turn's close "<|im_end|>\n", then `msgs` as the full render places them after an
 * assistant turn (a tool result opens the user turn; consecutive results share it), then the
 * generation prompt for `effort`.  full(history) + sampled turn + this = full(history + turn + msgs)
 * with add_generation_prompt.  Refusals: no_messages, system_not_first (no tail message is the
 * conversation's first), unexpected_role, and render_calls'. */
/// @param msgs   the messages after the sampled turn
/// @param n      entries in `msgs`
/// @param effort the effort the generation prompt follows
/// @param out    receives the tail's text and its client-span map
/// @param err    receives "<refusal key>: detail" on refusal
/// @param errlen size of `err`
/// @return true on success, false on refusal
bool qwen_chat_render_tail(const qwen_msg_in *msgs, int n, qwen_effort effort, qwen_render_out *out, char *err,
                           size_t errlen);

/** A conversation built turn by turn (L284 P14: the CLI REPL and the agent, through the family's chat front):
 * the HEAD is the system block the full render writes before the first message -- the effort line and the
 * system message's `system` content (NULL = none; no tools) -- and is empty when it has neither.
 * full(history) = head + each turn; a turn after an assistant turn is preceded by that turn's close
 * "<|im_end|>\n" (the sampled turn ends before its stop token, as qwen_chat_render_tail's live KV does). */
/// @param system the system message's content (client data), or NULL for none
/// @param effort the effort whose instruction line the head carries
/// @param out    receives the head's text and its client-span map
/// @param err    receives "<refusal key>: detail" on refusal
/// @param errlen size of `err`
/// @return true on success, false on refusal
bool qwen_chat_render_head(const char *system, qwen_effort effort, qwen_render_out *out, char *err, size_t errlen);

/** One TURN of a conversation built turn by turn: `msgs` as the full render writes them after an assistant
 * turn (a tool result opens the user turn; consecutive results share it), then the generation prompt for
 * `effort` when `generation_prompt`.  Refusals: no_messages (no messages and no generation prompt),
 * system_not_first (a system message here is not the conversation's first -- a front end writes a mid-
 * conversation note as a user turn of qwen_system_reminder), unexpected_role, and render_calls'. */
/// @param msgs              the turn's messages
/// @param n                 entries in `msgs`
/// @param effort            the effort the generation prompt follows
/// @param generation_prompt end with the generation prompt for `effort`
/// @param out               receives the turn's text and its client-span map
/// @param err               receives "<refusal key>: detail" on refusal
/// @param errlen            size of `err`
/// @return true on success, false on refusal
bool qwen_chat_render_turn(const qwen_msg_in *msgs, int n, qwen_effort effort, bool generation_prompt,
                           qwen_render_out *out, char *err, size_t errlen);

/** A system message that is not the conversation's first, as the content of the user turn that carries it
 * (the template has no in-place system turn): the ONE spelling the server's renderer and the turn-by-turn
 * front ends use. */
/// @param content the system message's text (NULL reads as "")
/// @return "<system-reminder>\n" + content + "\n</system-reminder>"
std::string qwen_system_reminder(const char *content);

/** Qwen's tool-call body for the output-parser kit (out_parse_kit.h): XML parameters inside one
 * \<tool_call\> block.
 * @verbatim
   <function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n ... </function>
   @endverbatim
 * A parameter VALUE is the text between the tags minus one leading and one trailing newline (the template
 * writes exactly those).  Arguments are rebuilt as JSON.  The template wrote strings raw and every other
 * value with tojson, so a value is JSON exactly when (a) the tool schema does not declare the property a
 * string (type "string" or a type list that includes it) and (b) the text parses AND is the tojson spelling
 * of what it parses to; anything else is the text as a string.  Either way the value re-renders to the
 * bytes the model sampled.
 *
 * L284 P5: a call streams as it is read -- TOOL_BEGIN when its name is complete, TOOL_ARGS as each
 * parameter CLOSES (a value's JSON spelling is known only then: the schema and the whole text decide it),
 * TOOL_END (or ERROR) at its close.  The fragments plus "}" are the arguments; a repeated key streams twice
 * (a JSON reader keeps the last, as `arguments` does).
 *
 * L284 P6: the payload is inside a parameter value the schema declares a string, or inside a JSON string
 * literal of any other value -- past the value's leading newline (the template's) and not on a partial
 * closer ("</" onward). */
class qwen_xml_call_body : public out_call_body {
public:
    /** The request's tools (the schema that types arguments). */
    pyjson_value tools;
    /** How the template spelt a non-string SCALAR: false = tojson (Qwen: true / false / null), true = Python's str()
     *  (L296 Nemotron-H's `args_value | string`: True / False / None; numbers spell the same either way; mappings and
     *  lists are tojson in both).  A value is typed only from its template's own spelling. */
    bool python_scalars = false;

    void open() override;
    void grow(const std::string &b, size_t held, bool closing, int index, std::vector<out_event> *ev) override;
    bool in_payload() const override { return pay_.value && pay_.payload; }
    bool close(const std::string &b, int index, std::vector<out_event> *ev, out_call *call, const char **why,
               size_t *at) override;

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
    /// @param fn  a function name
    /// @param key one of its parameters
    /// @return "string" when the schema declares the property a string (or a type list holding string), "other" for
    ///         any other declaration, NULL when the tool or the property is not declared
    const char *param_type(const std::string &fn, const std::string &key) const;

    /** @var pay_
    *  Where the open block's decode is (in_payload), scanned incrementally. */
    bool began_ = false;       ///< TOOL_BEGIN sent for the open call
    std::string fn_;           ///< the open call's function name, once complete
    size_t arg_pos_ = 0;       ///< body offset past the last parameter streamed (TOOL_ARGS)
    int arg_n_ = 0;            ///< parameters streamed for the open call
    bool arg_stop_ = false;    ///< the open call stopped streaming: its next bytes are not a parameter
    /// The incremental payload scanner's state for the open call.
    struct payload_scan {      ///< where the open block's decode is (in_payload), scanned incrementally
        size_t pos = 0;        ///< bytes of the body classified
        bool value = false;    ///< inside a parameter value
        size_t vstart = 0;     ///< where its text starts (past the tag)
        bool is_string = false;///< the schema declares it a string
        bool json_str = false; ///< inside a JSON string literal of a non-string value
        bool esc = false;      ///< the previous byte was a backslash in that literal
        bool payload = false;  ///< the result for the next byte
    } pay_;
};

/** Qwen's output grammar for the kit: `</think>` ends the reasoning the generation prompt opened (thinking
 * on); `<tool_call>` ... `</tool_call>` is a call; text after a call is content again (the prompt forbids
 * it; the client still sees it).  Reasoning and content are reported STRIPPED (the template strips both on
 * replay), so a turn the parser read re-renders to the bytes the model sampled. */
/// @return the family's grammar (static storage)
const out_grammar &qwen_output_grammar(void);

/** The sampled-text parser for one Qwen assistant turn: the kit's parser in Qwen's configuration. */
class qwen_output_parser : public out_parser {
public:
    /** `thinking`: the generation prompt opened a think block (the turn starts in reasoning).  `tools_json`:
     * the request's tools (NULL for none) -- the schema that types arguments.  false + reason if tools_json
     * is not a JSON array. */
    /// @param thinking   the turn starts in OUT_REASONING
    /// @param tools_json the request's tools array as JSON text, or NULL
    /// @param err        receives "tools_not_array: <detail>" on failure
    /// @param errlen     its size
    /// @return true when the parser is ready for the turn
    bool init(bool thinking, const char *tools_json, char *err, size_t errlen);
    out_parser *clone() const override;

private:
    /// The call-body reader (Qwen's XML format), schema-typed by the request's tools.
    qwen_xml_call_body body_;
};

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
#include "xml_call_body.h"

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
 * system message's `system` content (NULL = none) and the tools block of `tools_json` (NULL or [] = none; L365:
 * the agent's tools) -- and is empty when it has none of them.
 * full(history) = head + each turn; a turn after an assistant turn is preceded by that turn's close
 * "<|im_end|>\n" (the sampled turn ends before its stop token, as qwen_chat_render_tail's live KV does). */
/// @param system     the system message's content (client data), or NULL for none
/// @param tools_json an OpenAI tools array as JSON text, or NULL for none
/// @param effort     the effort whose instruction line the head carries
/// @param out        receives the head's text and its client-span map
/// @param err        receives "<refusal key>: detail" on refusal
/// @param errlen     size of `err`
/// @return true on success, false on refusal (tools_not_array)
bool qwen_chat_render_head(const char *system, const char *tools_json, qwen_effort effort, qwen_render_out *out,
                           char *err, size_t errlen);

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

/** Qwen's output grammar for the kit: `</think>` ends the reasoning the generation prompt opened (thinking
 * on); `<tool_call>` ... `</tool_call>` is a call; text after a call is content again (the prompt forbids
 * it; the client still sees it).  Reasoning and content are reported STRIPPED (the template strips both on
 * replay), so a turn the parser read re-renders to the bytes the model sampled.
 * L344: with thinking ON, a stray `</think>` in the content after the reasoning closed is dropped (as Nemotron's grammar
 * drops it): a forced close (the thinking budget) leaves a model still reasoning to write its own close in the answer.
 * With thinking OFF no reasoning was opened, so a literal `</think>` in the answer is the answer's text. */
/// @param thinking  the turn opened in reasoning (the generation prompt's `<think>`)
/// @return the family's grammar for that turn (static storage)
const out_grammar &qwen_output_grammar(bool thinking);

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
    xml_call_body body_;
};

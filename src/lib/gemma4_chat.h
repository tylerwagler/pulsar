/** @file
 *  The Gemma 4 chat format: the renderer of the checkpoint's chat_template.jinja (L338). */
/* gemma4_chat.h -- the Gemma 4 chat format (L338): the renderer, as the shared chat kit (chat_render_kit.h) configured
 * for Gemma 4 26B-A4B's chat_template.jinja ("Google Gemma 4 Canonical Chat Template", published 2026-07-09).
 *
 * RENDERER.  `gemma4_chat_render` writes the text HF's `apply_chat_template(messages, tools=..., enable_thinking=...,
 * add_generation_prompt=...)` writes with that template, byte for byte, and records the byte ranges it copied from
 * CLIENT data (the L223 span map).  It is the ONE rendering authority for this family.  What the template does that a
 * reader might not expect, all of it rendered:
 *   - the text opens with <bos>; a system turn is written only when thinking is on, tools are given, or the first
 *     message is a system / developer message (its content trimmed); thinking on puts <|think|> at its top, the tools
 *     follow the system text as <|tool>declaration:...<tool|> in the template's own quoted syntax (<|"|> quotes,
 *     keys in Jinja's case-insensitive dictsort order, types upper-cased);
 *   - an assistant message is a "model" turn; consecutive assistant messages (tool messages between them aside)
 *     continue ONE turn; a model turn's content passes strip_thinking (every <|channel>...<channel|> span dropped,
 *     the rest trimmed), any other content is trimmed;
 *   - reasoning (reasoning_content) renders as <|channel>thought\n...\n<channel|> only after the last user message
 *     (preserve_thinking: off, the template's default; no request sets it);
 *   - a call renders as <|tool_call>call:NAME{key:value,...}<tool_call|> (the arguments object in dictsort order, a
 *     string value quoted <|"|>...<|"|>); the tool messages right after the calling message render inside its turn as
 *     <|tool_response>response:NAME{value:<|"|>text<|"|>}<tool_response|>, NAME the call whose id is the message's
 *     tool_call_id (else the message's name, else "unknown"); a tool message anywhere else renders nothing;
 *   - a turn whose calls got no results ends "<|tool_response>" (the model's response turn is the next thing written);
 *     the generation prompt is "<|turn>model\n" (+ "<|channel>thought\n<channel|>" with thinking off) unless the last
 *     message ended in calls or results: after results it is "<|channel>thought\n" with thinking on and nothing with it
 *     off, after unanswered calls nothing.
 * OUTPUT PARSER (gemma4_output.cpp).  The output-parser kit in Gemma's configuration -- vLLM's gemma4 reading
 * (vllm/parser/gemma4.py): <|channel> opens reasoning (a repeated one and a stray <channel|> or <tool_call|> in
 * content are dropped), <channel|> ends it, <|tool_call> opens a call from either (a call ends the reasoning),
 * <tool_call|> closes it; the reasoning's "thought\n" label is dropped; reasoning right-stripped, content trimmed
 * around calls (vLLM's defaults).  A call body is "call:NAME{ARGS}" (or ":NAME{" / "(", as vLLM reads it) with ARGS
 * the template's own format_argument spelling read back -- <|"|>string<|"|>, bare true / false / null, a bare JSON
 * number, {k:v,...}, [..] -- so a call the server replays renders the bytes the model sampled; a bare value whose
 * property the schema declares a string stays the raw string.  Departure from vLLM, by design: vLLM leaves every
 * bare value a string unless the schema coerces it (count:42 with no schema reads "42" there, 42 here).
 * Refusals, by name: no_messages; content_parts (a content array: not in v1); tools_not_array; tool_function_missing
 * (a tool with no "function" object: the template's lookup raises); tool_name_missing; tool_arguments_missing,
 * tool_arguments_not_json, tool_arguments_not_object (the template raises unless the arguments are an object or
 * null); prompt_too_large. */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "chat_render_kit.h"
#include "out_parse_kit.h"
#include "pulsar.h"
#include "pyjson.h"

/** One tool call of an assistant message, as the API carries it. */
struct gemma4_tool_call_in {
    const char *id;         ///< the call's id (matches a tool message's tool_call_id); NULL = none
    const char *name;       ///< function name
    const char *arguments;  ///< the arguments object as JSON text ("null" renders no arguments)
};

/** One message.  `content` NULL renders as empty. */
struct gemma4_msg_in {
    const char *role;          ///< system | developer | user | assistant | tool (any other role renders as its turn)
    /// The message text (client data); NULL renders as empty.
    const char *content;
    const char *reasoning;     ///< assistant reasoning_content; NULL = none
    /// The assistant message's tool calls (n_calls of them); NULL when none.
    const gemma4_tool_call_in *calls;
    /// Number of entries in `calls`.
    int n_calls;
    const char *tool_call_id;  ///< a tool message's tool_call_id; NULL = none
    const char *name;          ///< a tool message's name (the fallback for its call's name); NULL = none
};

/// A full render's request: the conversation, its tools, thinking and the generation prompt.
struct gemma4_render_in {
    /// The conversation's messages, in order.
    const gemma4_msg_in *msgs;
    /// Number of entries in `msgs` (0 is refused as no_messages).
    int n_msgs;
    const char *tools_json;   ///< the request's tools array as JSON text; NULL = none
    bool thinking;            ///< enable_thinking
    bool add_generation_prompt;   ///< add_generation_prompt: end with the template's opener of the model's reply
};

/// A render's result: the prompt text and the client-data span map.
struct gemma4_render_out : chat_render_out {};

/** Render.  false + "<refusal key>: detail" on refusal. */
/// @param in     the conversation, tools and thinking
/// @param out    receives the text and its client-span map
/// @param err    receives "<refusal key>: detail" on refusal
/// @param errlen size of `err`
/// @return true on success, false on refusal
bool gemma4_chat_render(const gemma4_render_in &in, gemma4_render_out *out, char *err, size_t errlen);

/** The turn-by-turn front's head (pulsar_chat_open): `<bos>`, then the system turn when `thinking` or `system` is
 *  non-empty -- what the full render writes before the messages for a conversation that opens with that system
 *  message (none when `system` is NULL / empty) and the tools of `tools_json` (NULL or [] = none; L365). */
/// @param system     the system text (client data; NULL = none)
/// @param tools_json an OpenAI tools array as JSON text, or NULL for none
/// @param thinking   enable_thinking
/// @param out        receives the head's text and its client-span map
/// @param err        receives "<refusal key>: detail" on refusal
/// @param errlen     size of `err`
/// @return true on success, false on refusal (tools_not_array, a declaration's refusal)
bool gemma4_chat_render_head(const char *system, const char *tools_json, bool thinking, gemma4_render_out *out,
                             char *err, size_t errlen);

/** L365: the tool responses of a sampled model turn that made calls (the agent's tool loop).  Gemma answers calls
 *  INSIDE the model turn: the turn sampled its calls and stopped at "<|tool_response>" (not committed), so this
 *  writes each leading `tool` message's response block (its `name`, the call's function; NULL = "unknown") where the
 *  full render writes it, then -- when only tool messages follow -- the generation prompt that continues the same
 *  turn ("<|channel>thought\n" with thinking, else nothing); when other messages follow, the turn's close and those
 *  messages as gemma4_chat_render_turn writes them.  head + turns + sampled turn (without its stop) + this == the
 *  full render of the same conversation. */
/// @param msgs              the messages: one or more `tool` messages first
/// @param n                 their count
/// @param thinking          enable_thinking
/// @param generation_prompt end with the generation prompt
/// @param out               receives the text and its client-span map
/// @param err               receives "<refusal key>: detail" on refusal
/// @param errlen            size of `err`
/// @return true on success, false on refusal (no_tool_response, a later message's refusal)
bool gemma4_chat_render_tool_turn(const gemma4_msg_in *msgs, int n, bool thinking, bool generation_prompt,
                                  gemma4_render_out *out, char *err, size_t errlen);

/** The turn-by-turn front's turn (pulsar_chat_append_turn): `msgs` (no system message in the first place: the head
 *  holds it) as the full render writes them after the head, then the generation prompt when asked. */
/// @param msgs              the turn's messages
/// @param n                 entries in `msgs`
/// @param thinking          enable_thinking (the generation prompt's form)
/// @param generation_prompt append the model's opening
/// @param out               receives the turn's text and its client-span map
/// @param err               receives "<refusal key>: detail" on refusal
/// @param errlen            size of `err`
/// @return true on success, false on refusal
bool gemma4_chat_render_turn(const gemma4_msg_in *msgs, int n, bool thinking, bool generation_prompt,
                             gemma4_render_out *out, char *err, size_t errlen);

/// The close of a sampled model turn: "<turn|>\n" (the stop the model sampled, then the template's newline).
/// @return the close's text
const char *gemma4_chat_assistant_close(void);

/* ---- the output parser (gemma4_output.cpp) ------------------------------------------------------------------- */

/** Gemma's output grammar (the kit's configuration). */
/// @return the grammar (static storage)
const out_grammar &gemma4_output_grammar(void);

/** Gemma's call body: "call:NAME{ARGS}<tool_call|>" -- the name announced at its "{", the arguments read back from
 *  the template's spelling at the close (one TOOL_ARGS fragment), the payload any <|"|>-quoted string in them. */
class gemma4_call_body : public out_call_body {
public:
    /** The request's tools (a bare value whose top-level property is declared a string stays the raw text). */
    pyjson_value tools;
    void open() override;
    void grow(const std::string &b, size_t held, bool closing, int index, std::vector<out_event> *ev) override;
    bool in_payload() const override { return in_str_; }
    bool close(const std::string &b, int index, std::vector<out_event> *ev, out_call *call, const char **why,
               size_t *at) override;

private:
    /// @param fn  the call's function name
    /// @param key a top-level argument key
    /// @return true when the tools declare that property a string (its type is or includes "string")
    bool declared_string(const std::string &fn, const std::string &key) const;
    bool began_ = false;       ///< TOOL_BEGIN sent
    size_t args_lo_ = 0;       ///< where the arguments start (past the "{" / "("); 0 = not yet
    size_t scan_ = 0;          ///< bytes of the arguments scanned for quotes
    bool in_str_ = false;      ///< inside a <|"|> string: the next byte is payload
};

/** The parser: one per assistant turn. */
class gemma4_output_parser : public out_parser {
public:
    /** `reasoning`: the prompt left the turn inside an open <|channel> (the generation prompt after tool results
     *  with thinking on).  `tools_json`: the request's tools (NULL for none).  false + reason if tools_json is not
     *  a JSON array. */
    /// @param reasoning  the turn starts in OUT_REASONING
    /// @param tools_json the request's tools array as JSON text, or NULL
    /// @param err        receives "tools_not_array: <detail>" on failure
    /// @param errlen     its size
    /// @return true when the parser is ready for the turn
    bool init(bool reasoning, const char *tools_json, char *err, size_t errlen);
    out_parser *clone() const override;

private:
    /// The call-body reader.
    gemma4_call_body body_;
};

/** Whether a rendered prompt leaves the model's turn inside an open thinking channel (vLLM's
 *  _prompt_ends_in_open_reasoning: a <|channel> after the last <channel|>, <|tool_call>, <|turn> or
 *  <|tool_response>). */
/// @param prompt the rendered prompt text
/// @return true when the turn starts in reasoning
bool gemma4_prompt_opens_reasoning(const char *prompt);

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
#include "pulsar.h"

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
    bool add_generation_prompt;
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

/** The turn-by-turn front's head (pulsar_chat_open): <bos>, then the system turn when `thinking` or `system` is
 *  non-empty -- what the full render writes before the messages for a conversation that opens with that system
 *  message (none when `system` is NULL / empty) and no tools. */
/// @param system   the system text (client data; NULL = none)
/// @param thinking enable_thinking
/// @param out      receives the head's text and its client-span map
/// @param err      receives "<refusal key>: detail" on refusal
/// @param errlen   size of `err`
/// @return true on success, false on refusal
bool gemma4_chat_render_head(const char *system, bool thinking, gemma4_render_out *out, char *err, size_t errlen);

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

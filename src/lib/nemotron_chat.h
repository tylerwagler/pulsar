/* nemotron_chat.h -- the Nemotron-H chat format (L296 N8): the renderer, the effort authority and the output
 * parser, as F5's frontend kit configured for Nemotron 3.5 Super's checkpoint.
 *
 * RENDERER.  `nemotron_chat_render` writes the text HF's `apply_chat_template(messages, tools=...,
 * enable_thinking=..., medium_effort=..., add_generation_prompt=...)` writes with the checkpoint's
 * chat_template.jinja (ChatML + <think>), byte for byte, and records the byte ranges it copied from CLIENT data
 * (the L223 span map).  It is the ONE rendering authority for this family: the server, the CLI and every
 * continuation suffix go through it.  What the template does that a reader might not expect, all of it rendered:
 *   - a system turn is always written, empty when the conversation has no system message;
 *   - message content is written as sent (no trim), except an assistant turn, which is trimmed whole;
 *   - truncate_history_thinking (on: the template's default; no request sets it): an assistant turn before the
 *     last user message loses its reasoning ("<think></think>" + what follows the last "</think>");
 *   - medium effort appends "\n\n{reasoning effort: efficient}" to the LAST user message;
 *   - a system message after the first renders in place, as a system turn; any other role renders as its own turn;
 *   - a tool result opens a user turn after a non-tool message and closes it before one -- a tool result that is
 *     the conversation's first message opens nothing (the template's loop.previtem test).
 * Refusals, by name: no_messages; tools_not_array; tool_name_missing; tool_arguments_not_json (a call's arguments
 * text does not parse); tool_arguments_not_object (it parses to a non-object: the template's |items raises);
 * raw_calls_without_calls; prompt_too_large; reasoning_without_thinking (the sampled-turn writer only).
 *
 * EFFORT.  The template knows three levels: thinking off (enable_thinking=false: the generation prompt closes an
 * empty think block), medium (medium_effort: the last user message's suffix) and high (its default).
 *
 * OUTPUT PARSER.  The output-parser kit (out_parse_kit.h) in Nemotron's configuration: vLLM's nemotron_v3 reading
 * (qwen3's grammar -- <think> / </think>, <tool_call> blocks with qwen3_coder's XML body, a call straight from
 * reasoning ends it -- with trailing reasoning whitespace stripped) on qwen_chat.h's XML call body: one tool-call
 * FORMAT, one reader (kernels -- and readers -- keyed by format, not family). */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "out_parse_kit.h"
#include "pulsar.h"
#include "pyjson.h"
#include "qwen_chat.h"

enum nemotron_effort {
    NEMOTRON_EFFORT_NONE,     ///< enable_thinking=false
    NEMOTRON_EFFORT_MEDIUM,   ///< medium_effort=true
    NEMOTRON_EFFORT_HIGH,     ///< the template's default
};

/** The family's default effort: the template's own (thinking on, no medium suffix). */
nemotron_effort nemotron_effort_default(void);
/** "none", "medium", "high" -> the effort; false for anything else. */
bool nemotron_effort_parse(const char *name, nemotron_effort *out);
const char *nemotron_effort_name(nemotron_effort e);

/** One tool call of an assistant message, as the API carries it. */
struct nemotron_tool_call_in {
    const char *name;       ///< function name
    const char *arguments;  ///< the arguments object as JSON text; NULL or "" = none
};

/** One message.  `content` NULL renders as empty. */
struct nemotron_msg_in {
    const char *role;       ///< system | user | assistant | tool (any other role renders as its own turn)
    /// The message text (client data); NULL renders as empty.
    const char *content;
    const char *reasoning;  ///< assistant reasoning_content; NULL = none
    /// The assistant message's tool calls (n_calls of them); NULL when none.
    const nemotron_tool_call_in *calls;
    /// Number of entries in `calls`.
    int n_calls;
    /** The calls' SAMPLED bytes, from the first "<tool_call>" to the end of the last "</tool_call>" (the server's
     *  tool memory): written in place of rendering `calls`, so a replayed turn byte-matches the live KV.  NULL =
     *  render `calls`. */
    const char *raw_calls;
};

/// A full render's request: the conversation, its tools and the effort.
struct nemotron_render_in {
    /// The conversation's messages, in order.
    const nemotron_msg_in *msgs;
    /// Number of entries in `msgs` (0 is refused as no_messages).
    int n_msgs;
    const char *tools_json;   ///< the request's tools array as JSON text; NULL = none
    /// The thinking level the generation prompt and the medium suffix follow.
    nemotron_effort effort;
    /// End with nemotron_generation_prompt(effort).
    bool add_generation_prompt;
};

/// A render's result: the prompt text and the client-data span map.
struct nemotron_render_out {
    /// The rendered text (UTF-8; at most UINT32_MAX bytes, else prompt_too_large).
    std::string text;
    std::vector<pulsar_text_span> spans;   ///< client-data ranges, ascending, disjoint
};

/** Render.  false + "<refusal key>: detail" on refusal. */
bool nemotron_chat_render(const nemotron_render_in &in, nemotron_render_out *out, char *err, size_t errlen);

/** L325: tool definitions in the tools block's encoding -- "<tools>", one function element per tool, "\n</tools>"
 *  (the block without its head and call-format tail) -- for a tool loaded mid-conversation, rendered in place
 *  inside the turn that loaded it.  `tools_json` is an OpenAI tools array.  Refusal: tools_not_array. */
bool nemotron_chat_tool_defs(const char *tools_json, nemotron_render_out *out, char *err, size_t errlen);

/** The assistant turn `m` as the model SAMPLES it after the generation prompt -- the bytes the full render writes
 *  for the turn (it is after the last user message: nothing is truncated) past "<|im_start|>assistant\n<think>\n"
 *  with `thinking`, past "<|im_start|>assistant\n<think></think>" without: reasoning, "</think>", content (the turn
 *  trimmed at its end), then "\n" and the calls (or `raw_calls`) and "<|im_end|>\n" -- or, with no call,
 *  "<|im_end|>\n" alone.  A turn with no reasoning replays as an empty one ("</think>" right after the prompt's
 *  "<think>\n" is NOT the template's "<think></think>": a thinking-on turn without reasoning does not compose, and
 *  is refused as reasoning_missing).  Refusals: reasoning_without_thinking, reasoning_missing, and the calls'. */
bool nemotron_chat_render_assistant_turn(const nemotron_msg_in &m, bool thinking, nemotron_render_out *out, char *err,
                                         size_t errlen);

/** A continuation tail after a sampled turn (nemotron_chat_render_assistant_turn's bytes): `msgs` as the full
 *  render places them after an assistant turn (a tool result opens the user turn; consecutive results share it),
 *  then the generation prompt for `effort`.  full(history) + sampled turn + this = full(history + turn + msgs) when
 *  `msgs` holds no user message (a user message moves last_user_idx: the template then truncates the sampled turn's
 *  reasoning and moves the medium suffix -- the prefix does not compose, by the template's design). */
bool nemotron_chat_render_tail(const nemotron_msg_in *msgs, int n, nemotron_effort effort, nemotron_render_out *out,
                               char *err, size_t errlen);

/** The head of a conversation built turn by turn (the CLI REPL and the agent): the system turn with `system`
 *  (NULL = empty), no tools. */
bool nemotron_chat_render_head(const char *system, nemotron_render_out *out, char *err, size_t errlen);

/** One TURN of a conversation built turn by turn: `msgs` as the full render writes them after the head or an
 *  assistant turn's close (the medium suffix on the turn's last user message), then the generation prompt for
 *  `effort` when `generation_prompt`. */
bool nemotron_chat_render_turn(const nemotron_msg_in *msgs, int n, nemotron_effort effort, bool generation_prompt,
                               nemotron_render_out *out, char *err, size_t errlen);

/** The generation prompt for `effort` (the template's last bytes with add_generation_prompt). */
const char *nemotron_generation_prompt(nemotron_effort effort);

/** Tool memory: the run of consecutive tool-call blocks (whitespace between them) at or after `p` -- one turn's
 *  calls; `*end` = one past it.  NULL = none (an unclosed block is not a key). */
const char *nemotron_find_call_block(const char *p, const char **end);

/** The text of the system turn a rendered prompt opens with (the tools block and the client's system text), or ""
 *  when the prompt has none: what a tool error reminds the model of. */
std::string nemotron_rendered_system_text(const char *prompt);

/** Nemotron's output grammar for the kit (vLLM's nemotron_v3: qwen3_config + strip_trailing_reasoning_whitespace):
 *  `</think>` ends the reasoning the generation prompt opened (a stray one in content is dropped, a repeated
 *  `<think>` in reasoning too); `<tool_call>` opens a call from reasoning or content; `</tool_call>` closes it.
 *  Reasoning is reported with its trailing whitespace dropped; content verbatim until a call opens, then trimmed. */
const out_grammar &nemotron_output_grammar(void);

/** The sampled-text parser for one Nemotron assistant turn: the kit in Nemotron's configuration. */
class nemotron_output_parser : public out_parser {
public:
    /** `thinking`: the generation prompt opened a think block (the turn starts in reasoning).  `tools_json`: the
     *  request's tools (NULL for none) -- the schema that types arguments.  false + reason if tools_json is not a
     *  JSON array. */
    /// @param thinking   the turn starts in OUT_REASONING
    /// @param tools_json the request's tools array as JSON text, or NULL
    /// @param err        receives "tools_not_array: <detail>" on failure
    /// @param errlen     its size
    /// @return true when the parser is ready for the turn
    bool init(bool thinking, const char *tools_json, char *err, size_t errlen);
    out_parser *clone() const override;

private:
    /// The call-body reader (Qwen's XML format, Python scalar spelling), schema-typed by the request's tools.
    qwen_xml_call_body body_;
};

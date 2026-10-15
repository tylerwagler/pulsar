/* inkling_chat.h -- the Inkling-Small chat format (L294 E6, built on the L295 F5 frontend kit): the renderer,
 * the effort authority and the output parser's configuration.
 *
 * RENDERER.  `inkling_chat_render` writes the text whose tokenisation (the shared BPE engine in its tiktoken
 * configuration, with the client-span guard) is the token ids vLLM's native Inkling renderer emits
 * (vllm/renderers/inkling_encoding.py render_inkling_messages, through renderers/inkling.py), byte for byte.
 * vLLM renders to ids, not text, because every text segment is encoded on its own; here each segment sits
 * between two special tokens, which is the same cut, so the text form tokenises to the same ids.  The one
 * deliberate difference is the L223 guard: vLLM's encode_text matches a special token spelled inside client
 * text; this renderer records every client byte range, so a client who types `<|end_message|>` gets text.
 *
 * Every block is `<|message_ROLE|>[author]<|content_KIND|>body<|end_message|>`:
 *   - the tools, when any: ONE system block authored "tool_declare", kind xml, body the canonical JSON
 *     (keys sorted, compact, non-ASCII raw) of [{description, name, parameters, type}] per tool;
 *   - the effort: a system text block "Thinking effort level: X" before the first message that is not
 *     system / developer (after all of them when there is none);
 *   - a message's content: one text block per part (a string content is one part; an empty string is
 *     none), an image part a bare `<|content_image|>` block (the image front expands it);
 *   - an assistant message: its reasoning as a thinking block first, its content, one invoke_tool_json block
 *     per call (authored by the call's name, body `{"name":NAME,"args":CANONICAL_ARGS}`), then
 *     `<|content_model_end_sampling|>`;
 *   - a tool result: one text block authored by the message's name, else the name of the earlier call
 *     whose id it answers;
 *   - the generation prompt: `<|message_model|>`.
 * developer renders as system.
 *
 * Refusals, by key (vLLM raises on each): unexpected_role, tool_name_missing (a call's name is not a
 * string), tool_arguments_not_json, tool_arguments_not_object, tool_result_not_text (an image in a tool
 * result), tools_not_array, tool_spec_invalid (a tool without function.name, or not an object),
 * json_not_finite (allow_nan=False), effort_out_of_range, prompt_too_large.
 *
 * EFFORT.  Inkling's effort is a number in [0, 0.99], written in hundredths here (vLLM's renderer formats
 * `f"{v:.2f}"` with trailing zeros dropped, and "0.0" for zero).  The names any protocol sends map to
 * the renderer's own table: none 0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh / max 0.99; nothing
 * sent is 0.9.
 *
 * OUTPUT.  `inkling_output_grammar` is the output-parser kit's configuration for Inkling's typed content
 * blocks (vllm/parser/inkling.py's transition table), read with the kit's shared JSON call body
 * (out_json_call_body: `{"name":...,"args":{...}}`, the arguments the args object as sampled): see
 * out_parse_kit.h.  Reasoning is the turn's thinking text with its trailing whitespace dropped; content is
 * verbatim unless it is only whitespace and a call opened or was read (vLLM's
 * strip_trailing_reasoning_whitespace and drop_whitespace_only_content_before_tools).  The one departure
 * from vLLM's reading: a call the turn ENDS inside is broken (an ERROR, the rule every family shares), where
 * vLLM carves arguments out of the unterminated wrapper. */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "out_parse_kit.h"
#include "pulsar.h"

/** The effort nothing sent resolves to (vLLM's _DEFAULT_REASONING_EFFORT), in hundredths. */
int inkling_effort_default(void);
/** The effort a name means (vLLM's _NAMED_REASONING_EFFORT), in hundredths; -1 for a name it lacks. */
int inkling_effort_named(const char *name);
/** The effort line's number as the renderer writes it ("0.0", "0.1", "0.05", "0.99"); false when `centi`
 * is outside [0, 99]. */
bool inkling_effort_text(int centi, std::string *out);

/** One tool call of an assistant message, as the API carries it. */
struct inkling_tool_call_in {
    const char *id;         ///< the call's id (NULL = none): a later tool result may name it
    const char *name;       ///< NULL = missing (refused)
    const char *arguments;  ///< the arguments object as JSON text; NULL, "" or blank = {}
};

/** One content part: text, or an image. */
struct inkling_part_in {
    bool image;
    const char *text;       ///< a text part's text (NULL = "")
};

/** One message.  Its content is `content` (a string; NULL = none) or, when `parts` is set, the parts. */
struct inkling_msg_in {
    const char *role;       ///< system | developer | user | assistant | tool
    const char *content;
    const inkling_part_in *parts;
    int n_parts;
    const char *reasoning;  ///< assistant reasoning; NULL or "" = none
    const inkling_tool_call_in *calls;
    int n_calls;
    const char *name;           ///< a tool result's tool name (NULL = look it up by tool_call_id)
    const char *tool_call_id;   ///< a tool result's call id
    /** An assistant message's calls as SAMPLED (the tool memory's bytes, inkling_find_call_block's key: from the
     *  first call's author to the last call's <|end_message|>); NULL = none.  Written verbatim in place of the
     *  canonical call blocks when they are one run of call blocks (each call is still read, and refused, as
     *  vLLM reads it); otherwise the calls render canonically. */
    const char *raw_calls;
};

struct inkling_render_in {
    const inkling_msg_in *msgs;
    int n_msgs;
    const char *tools_json;   ///< the request's tools array as JSON text; NULL = none
    int effort;               ///< hundredths, [0, 99]
    bool add_generation_prompt;
};

struct inkling_render_out {
    std::string text;
    std::vector<pulsar_text_span> spans;   ///< client-data ranges, ascending, disjoint
};

/** Render.  false + "<refusal key>: detail" on refusal. */
bool inkling_chat_render(const inkling_render_in &in, inkling_render_out *out, char *err, size_t errlen);

/* ---- the pieces a live conversation appends (the server's suffix hooks, the engine's chat front).  Each is
 * the full render's own writer, so the prompt so far + a piece == the full render of the longer conversation,
 * byte for byte (inkling-chat-gate S1). */

/** msgs[from, n) as the full render writes them after its effort block (no tool declaration, no effort line:
 * the conversation's start wrote both); the messages before `from` only name the calls a tool result answers.
 * `close_turn`: first the <|content_model_end_sampling|> that ends the sampled turn the piece follows (the
 * transcript stopped before it); `add_generation_prompt`: last, <|message_model|>.  A piece with none of the
 * three is refused (no_messages). */
bool inkling_chat_render_turn(const inkling_msg_in *msgs, int n, int from, bool close_turn, bool add_generation_prompt,
                              inkling_render_out *out, char *err, size_t errlen);

/** One assistant turn as SAMPLED after the generation prompt: the message's full render without its first
 * <|message_model|> (the generation prompt wrote it), closed by <|content_model_end_sampling|> only when it
 * made no call (a tool-call turn's close is its tail's: inkling_chat_render_turn's close_turn).  A turn with
 * no reasoning, content or call has no such render (empty_turn). */
bool inkling_chat_render_assistant_turn(const inkling_msg_in &m, inkling_render_out *out, char *err, size_t errlen);

/** The tool memory's key in a transcript's text: the earliest complete run of call blocks at or after `p`,
 * from a call's author (just past its <|message_model|>, or `p` itself when the author starts it) through the
 * <|end_message|> of the last call that follows block after block; `*end` = one past it.  NULL = none (an
 * unclosed block is none).  The output parser's raw span of a turn's calls has exactly these bounds. */
const char *inkling_find_call_block(const char *p, const char **end);

/** The system turn of a rendered prompt, for the model-visible tool error's reminder: the bodies of its leading
 * system blocks -- the tool declaration, the system messages -- without the effort line, joined by a blank line;
 * empty when it has none. */
std::string inkling_rendered_system_text(const char *prompt);

/** Inkling's output grammar for the kit: its eight content-block markers, the transition table, and the
 * tokenizer's other special tokens as dropped markers (vLLM drops a special token that is not a terminal;
 * inkling-chat-gate asserts the two lists are special_tokens_map.json's).  A turn after the generation
 * prompt's `<|message_model|>` starts in OUT_HEADER. */
const out_grammar &inkling_output_grammar(void);

/** The sampled-text parser for one Inkling assistant turn: the kit's parser in Inkling's configuration (its
 * grammar and the shared JSON call body).  The family's server `create` builds one per decode attempt and
 * hands it to parser_kit_start. */
class inkling_output_parser : public out_parser {
public:
    /** `initial`: where the prompt left the turn -- OUT_HEADER after the generation prompt's
     *  `<|message_model|>`. */
    void init(out_state initial);

private:
    out_json_call_body body_;
};

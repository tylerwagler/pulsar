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
 * blocks (vllm/parser/inkling.py's transition table): see out_parse_kit.h. */
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

/** Inkling's output grammar for the kit: its eight content-block markers and the transition table.  The
 * parser starts in the message header the generation prompt opened. */
const out_grammar &inkling_output_grammar(void);

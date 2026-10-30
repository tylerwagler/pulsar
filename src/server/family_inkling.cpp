/* family_inkling.cpp -- the Inkling family's server table (L294 E6; server_family_ops, chat_family.cpp's
 * switch names it).
 *
 * Everything here is F5's frontend (src/lib/inkling_chat.h): the render is vLLM's native Inkling renderer
 * (vllm/renderers/inkling_encoding.py) byte for byte, the effort map its _NAMED_REASONING_EFFORT and
 * _DEFAULT_REASONING_EFFORT, the output parser the kit in Inkling's configuration (vllm/parser/inkling.py), and
 * every suffix hook one of the renderer's own pieces -- so the prompt so far + a hook's bytes == the full render of
 * the longer conversation (inkling-chat-gate S1 over the corpus; test_inkling_hooks_compose_to_the_full_render).
 *
 * What vLLM's renderer cannot express is refused by name, never rendered in another family's spelling:
 *   - a forced tool_choice (required / a named function): vLLM's Inkling path has no forced call, so there is
 *     no seed and no prefill (forced_call_* are NULL);
 *   - images: the template's <|content_image|> block needs the family's image front (E9), not wired.
 * Departures from vLLM, each the rule every family shares: client text spelling a marker stays text (L223); a
 * call the turn ends inside is broken (L272); a message's text parts arrive joined by the protocol parsers, so
 * they render as one text block where vLLM writes one block a part. */
/** @file
 *  The Inkling family's server table (server_family_ops) over src/lib/inkling_chat.
 */
#include "pulsar_server_internal.h"
#include "../lib/inkling_chat.h"

#include <memory>
#include <string>
#include <vector>

namespace {

/* ---- the effort map: the renderer's hundredths (inkling_chat.h, its one authority) ------------------------ */

/* An integer effort 1..100 (pulsar's axis) is that many hundredths, capped at the renderer's 0.99 */
/** server_effort_map::integer for Inkling.
 *  @param value the integer effort, 1..100
 *  @return the renderer's effort in hundredths, capped at 99 */
int inkling_effort_integer(int value) { return value < 99 ? value : 99; }
/** server_effort_map::default_level for Inkling: the renderer's default effort (inkling_effort_default).
 *  @return the level in hundredths */
int inkling_effort_default_level(pulsar_engine *) { return inkling_effort_default(); }

/** Inkling's effort map (server_family_ops::effort): the named levels are the renderer's own, in hundredths. */
const server_effort_map k_effort_inkling = [] {
    server_effort_map m = {};
    for (int i = 0; i < CHAT_EFFORT_NAMES; i++) {
        m.named[i] = inkling_effort_named(k_effort_names[i]);
        if (m.named[i] < 0) pulsar_die("inkling: an effort name the renderer's table lacks");
    }
    m.integer = inkling_effort_integer;
    m.default_level = inkling_effort_default_level;
    m.think_mode_level = false;   /* r->family_effort carries the hundredths; think_mode the on / off marker */
    return m;
}();

/* ---- the render ---------------------------------------------------------------------------------------- */

/* The parsed messages as the renderer's, borrowing `msgs`: the request's system FIELD first (the protocol parsers
 * append it to the array), then the messages in order -- or, for a TAIL from `start`, the messages before it (they
 * only name the calls a tool result answers) and those from it, the system field not among them.  `*from` = where
 * the rendered piece starts.  false + err: an image (E9's front is not wired). */
/** The renderer's message list built by inkling_messages; its strings borrow the chat_msgs it was built from. */
struct inkling_msgs {
    /// The messages in render order.
    std::vector<inkling_msg_in> m;
    /// Per SOURCE message index: its calls (`m` points in).
    std::vector<std::vector<inkling_tool_call_in>> calls;
    /// Index into `m` where the rendered piece starts.
    int from = 0;
};

/** Build `out` from `msgs` (the rule above).
 *  @param msgs the parsed messages (borrowed: must outlive `out`)
 *  @param start a tail's first message index (read only when `tail`)
 *  @param tail build a tail from `start` instead of the whole conversation
 *  @param out the result
 *  @param err the refusal (an image)
 *  @param errlen bytes at `err`
 *  @return false + err when a message carries an image */
bool inkling_messages(const chat_msgs *msgs, int start, bool tail, inkling_msgs *out, char *err, size_t errlen) {
    out->m.clear();
    out->calls.assign((size_t)msgs->len, {});
    auto add = [&](int i) -> bool {
        const chat_msg *c = &msgs->v[i];
        if (c->images_len > 0) {
            snprintf(err, errlen, "inkling: a message carries an image -- the family's image front (E9) is not wired; "
                                  "this engine renders text only");
            return false;
        }
        std::vector<inkling_tool_call_in> &k = out->calls[(size_t)i];
        for (int j = 0; j < c->calls.len; j++)
            k.push_back({c->calls.v[j].id, c->calls.v[j].name, c->calls.v[j].arguments});
        inkling_msg_in q = {};
        q.role = c->role;
        q.content = c->content;
        q.reasoning = c->reasoning;
        q.calls = k.empty() ? NULL : k.data();
        q.n_calls = (int)k.size();
        q.tool_call_id = c->tool_call_id;
        q.raw_calls = c->calls.raw_calls;
        out->m.push_back(q);
        return true;
    };
    if (tail) {
        for (int i = 0; i < start; i++)
            if (!add(i)) return false;
        out->from = (int)out->m.size();
        for (int i = start; i < msgs->len; i++)
            if (!msgs->v[i].system_field && !add(i)) return false;
        return true;
    }
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < msgs->len; i++)
            if (msgs->v[i].system_field == (pass == 0) && !add(i)) return false;
    out->from = 0;
    return true;
}

/* A render's text and client-data ranges as the request owns them (malloc'd; the ranges NULL when none). */
/** @param out the renderer's output
 *  @param spans_out set to a malloc'd copy of the client-data spans (NULL when none); NULL = not wanted
 *  @param n_spans_out set to the span count (read only with `spans_out`)
 *  @return the rendered text, malloc'd */
char *inkling_take_render(const inkling_render_out &out, chat_text_span **spans_out, uint32_t *n_spans_out) {
    if (spans_out) {
        *spans_out = NULL;
        *n_spans_out = (uint32_t)out.spans.size();
        if (!out.spans.empty()) {
            *spans_out = (chat_text_span *)server_xmalloc(out.spans.size() * sizeof(chat_text_span));
            memcpy(*spans_out, out.spans.data(), out.spans.size() * sizeof(chat_text_span));
        }
    }
    return xstrndup(out.text.data(), out.text.size());
}

/* vLLM's render_inkling_messages over the request: the tools (OpenAI's array) declared in the leading system block,
 * the effort line (r->family_effort, the hundredths the effort map resolved), the generation prompt
 * <|message_model|>; tokenised with the client spans (L223). */
/** server_family_ops::render for Inkling.
 *  @param e the engine (NULL = render only: no tokenisation, no images)
 *  @param s the server
 *  @param c the parsed conversation
 *  @param r the request: prompt_text / prompt_spans / prompt, has_tools and tools_json written
 *  @param err the refusal
 *  @param errlen bytes at `err`
 *  @return false + err for a forced tool_choice, an image, or a render the renderer refuses */
bool inkling_render(pulsar_engine *e, server *s, chat_conversation *c, request *r, char *err, size_t errlen) {
    if (c->tool_choice == CHAT_TOOL_CHOICE_ANY || c->tool_choice == CHAT_TOOL_CHOICE_NAMED) {
        snprintf(err, errlen, "tool_choice %s: Inkling's renderer has no forced tool call (vLLM's native Inkling "
                              "renderer takes none) -- send auto or none",
                 c->tool_choice == CHAT_TOOL_CHOICE_ANY ? "required" : "naming a function");
        return false;
    }
    char *head_raw = NULL;
    if (!chat_openai_tools_json(c, r, &head_raw, err, errlen)) return false;
    const std::unique_ptr<char, void (*)(void *)> head(head_raw, free);
    r->has_tools = r->tools_json != NULL;
    render_prelude(s, c, r);
    inkling_msgs im;
    if (!inkling_messages(&c->msgs, 0, false, &im, err, errlen)) return false;
    inkling_render_in in = {};
    in.msgs = im.m.data();
    in.n_msgs = (int)im.m.size();
    in.tools_json = head.get();   /* the tools block; r->tools_json (every declared tool) types the parser (L325) */
    in.effort = r->family_effort;
    in.add_generation_prompt = true;
    inkling_render_out out;
    if (!inkling_chat_render(in, &out, err, errlen)) return false;
    free(r->prompt_spans);
    r->prompt_text = inkling_take_render(out, &r->prompt_spans, &r->prompt_n_spans);
    if (e) {
        pulsar_tokenize_rendered_chat_spans(e, r->prompt_text, r->prompt_spans, r->prompt_n_spans, &r->prompt);
        if (!request_prepare_images(e, &c->msgs, r, err, errlen)) return false;
    }
    return true;
}

/* L272 B6: the longest prefix two UNRELATED prompts share by template construction.  With no tools and no system
 * message every Inkling prompt opens with the effort block and a user text block's opener; the longest of those over
 * the named efforts (the client's first byte is where the shared part ends). */
/** server_family_ops::trivial_header_tokens for Inkling.
 *  @param e the engine (its tokenizer)
 *  @return the header's length in tokens (the longest over the named efforts) */
int inkling_trivial_header_tokens(pulsar_engine *e) {
    int hdr_len = 0;
    inkling_msg_in user = {};
    user.role = "user";
    user.content = "x";
    for (int i = 0; i < CHAT_EFFORT_NAMES; i++) {
        inkling_render_in in = {};
        in.msgs = &user;
        in.n_msgs = 1;
        in.effort = k_effort_inkling.named[i];
        inkling_render_out out;
        char err[200];
        if (!inkling_chat_render(in, &out, err, sizeof err) || out.spans.empty())
            pulsar_die("inkling: the trivial header's render refused");
        const std::string hdr = out.text.substr(0, out.spans[0].lo);
        pulsar_tokens t = {0};
        pulsar_tokenize_rendered_chat(e, hdr.c_str(), &t);
        if (t.len > hdr_len) hdr_len = t.len;
        pulsar_tokens_free(&t);
    }
    return hdr_len;
}

/* ---- the output parser: the kit in Inkling's configuration, from the generation prompt's header ----------- */

/** server_output_parser_ops::create for Inkling: the parser kit started in a block header.  Never fails.
 *  @return the parser state (parser_kit_destroy frees it) */
void *inkling_parser_create(server *, gen_state *, char *, size_t) {
    inkling_output_parser *p = new inkling_output_parser();
    p->init(OUT_HEADER);   /* the prompt ends with <|message_model|>: the turn starts in its first block's header */
    return parser_kit_start(p);
}

/** Inkling's output parser table: the parser kit's hooks. */
const server_output_parser_ops k_parser_inkling = {
    /* .create        = */ inkling_parser_create,
    /* .destroy       = */ parser_kit_destroy,
    /* .seed          = */ parser_kit_seed,
    /* .feed          = */ parser_kit_feed,
    /* .in_tool_call  = */ parser_kit_in_tool_call,
    /* .turn_complete = */ parser_kit_turn_complete,
    /* .tool_progress = */ parser_kit_tool_progress,
    /* .finish        = */ parser_kit_finish,
    /* .probe_new     = */ parser_kit_probe_new,
    /* .probe_feed    = */ parser_kit_probe_feed,
    /* .probe_free    = */ parser_kit_probe_free,
};

/* ---- step 4: the suffix hooks, each one of the renderer's pieces ------------------------------------------ */

/* The turn as sampled after the generation prompt: reasoning (NULL = not replayed), content, the calls -- their
 * sampled bytes when the tool memory has them -- and the stop token only when no call was made.  Inkling's turn
 * has no think-mode spelling: `think` changes nothing the template writes. */
/** server_family_ops::assistant_turn_sampled for Inkling.
 *  @param reasoning the reasoning to replay; NULL = none
 *  @param content the content text
 *  @param calls the turn's calls; NULL = none
 *  @param spans_out the client-data spans (malloc'd; NULL when none); NULL = not wanted
 *  @param n_spans_out the span count
 *  @return the turn's text (malloc'd), or NULL when the renderer refused it (logged) */
char *inkling_assistant_turn_sampled(const request *, bool, const char *reasoning, const char *content,
                                     const tool_calls *calls, chat_text_span **spans_out, uint32_t *n_spans_out) {
    std::vector<inkling_tool_call_in> k;
    for (int j = 0; calls && j < calls->len; j++) k.push_back({calls->v[j].id, calls->v[j].name, calls->v[j].arguments});
    inkling_msg_in m = {};
    m.role = "assistant";
    m.content = content;
    m.reasoning = reasoning;
    m.calls = k.empty() ? NULL : k.data();
    m.n_calls = (int)k.size();
    m.raw_calls = calls ? calls->raw_calls : NULL;
    inkling_render_out out;
    char err[200];
    if (!inkling_chat_render_assistant_turn(m, &out, err, sizeof err)) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: Inkling sampled-turn render refused: %s", err);
        if (spans_out) { *spans_out = NULL; *n_spans_out = 0; }
        return NULL;
    }
    return inkling_take_render(out, spans_out, n_spans_out);
}

/* After a tool-call turn: its stop token, the messages from `start` (tool results authored by the calls they
 * answer, a mid-loop system message as a system block), the generation prompt. */
/** server_family_ops::tool_result_tail for Inkling.
 *  @param msgs the conversation's messages
 *  @param start the first new user-side message
 *  @param spans_out the client-data spans (malloc'd; NULL when none); NULL = not wanted
 *  @param n_spans_out the span count
 *  @return the tail's text (malloc'd), or NULL when refused (logged) */
char *inkling_tool_result_tail(const request *, const chat_msgs *msgs, int start, chat_text_span **spans_out,
                               uint32_t *n_spans_out) {
    if (spans_out) { *spans_out = NULL; *n_spans_out = 0; }
    inkling_msgs im;
    inkling_render_out out;
    char err[200];
    if (!inkling_messages(msgs, start, true, &im, err, sizeof err) ||
        !inkling_chat_render_turn(im.m.data(), (int)im.m.size(), im.from, true, true, &out, err, sizeof err)) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: Inkling tool-result tail refused: %s", err);
        return NULL;
    }
    return inkling_take_render(out, spans_out, n_spans_out);
}

/* The model-visible retry, as on every family: the malformed turn ended at its stop token, and the error -- with the
 * system turn reminded (the tool declaration and the system text, Qwen's reminder) -- is ONE tool result in the
 * template's own spelling, a <|message_tool|> text block with no author (no call was read to name it), then the
 * generation prompt.  vLLM's renderer writes no other error form (<|content_tool_error|> is never rendered). */
/** server_family_ops::tool_error_suffix for Inkling.
 *  @param r the request (its prompt_text supplies the system reminder)
 *  @param detail what was malformed; NULL or empty = no detail
 *  @param spans_out the client-data spans (malloc'd; NULL when none); NULL = not wanted
 *  @param n_spans_out the span count
 *  @return the suffix's text (malloc'd), or NULL when refused */
char *inkling_tool_error_suffix(const request *r, const thinking_state *, const char *detail, chat_text_span **spans_out,
                                uint32_t *n_spans_out) {
    std::string text = "Tool error: malformed tool call";
    if (detail && detail[0]) text += std::string(": ") + detail;
    text += "\nThe previous assistant output was not executed because its tool call was malformed. Emit a new valid "
            "tool call, or answer normally if no tool is needed.";
    const std::string system = inkling_rendered_system_text(r->prompt_text);
    if (!system.empty()) text += "\n\nSystem prompt reminder:\n" + system;
    chat_msgs msgs = {0};
    chat_msg result = {0};
    result.role = xstrdup("tool");
    result.content = xstrdup(text.c_str());
    chat_msgs_push(&msgs, result);
    char *suffix = inkling_tool_result_tail(r, &msgs, 0, spans_out, n_spans_out);
    chat_msgs_free(&msgs);
    return suffix;
}

/* L303: the answer is a text block's body (<|content_text|> ... <|end_message|>); a call is an invoke_tool_json
 * block's body, {"name":NAME,"args":ARGS} in the renderer's compact JSON (tool_call_json), then its end. */
/** Inkling's grammar shape (server_family_ops::grammar). */
const server_grammar_shape k_grammar_inkling = {
    /* .answer_open           = */ "<|content_text|>",
    /* .answer_at_think_close = */ false,
    /* .answer_close          = */ "<|end_message|>",
    /* .call_trigger          = */ "<|content_invoke_tool_json|>",
    /* .block_head            = */ "",
    /* .name_open             = */ "{\"name\":\"",
    /* .name_close            = */ "\",\"args\":",
    /* .style                 = */ "json",
    /* .call_close            = */ "}<|end_message|>",
    /* .repeat                = */ false,
    /* .block_tail            = */ "",
};

}  // namespace

/* L325: a tool loaded mid-conversation, in place: the tool_declare array (inkling_chat_tool_defs). */
static char *inkling_tool_defs_text(const char *lines, size_t len, char *err, size_t errlen) {
    char *json = tool_schema_lines_openai_tools(lines, len, err, errlen);
    if (!json) return NULL;
    inkling_render_out out;
    const bool ok = inkling_chat_tool_defs(json, &out, err, errlen);
    free(json);
    return ok ? xstrndup(out.text.data(), out.text.size()) : NULL;
}

const server_family_ops k_family_inkling = {
    /* .name                  = */ "Inkling",
    /* .format                = */ PULSAR_CHAT_INKLING,
    /* .v41                   = */ false,
    /* .effort                = */ &k_effort_inkling,
    /* .render                = */ inkling_render,
    /* .trivial_header_tokens = */ inkling_trivial_header_tokens,
    /* .output                = */ &k_parser_inkling,
    /* .assistant_turn_sampled = */ inkling_assistant_turn_sampled,
    /* .tool_result_tail      = */ inkling_tool_result_tail,
    /* .tool_error_suffix     = */ inkling_tool_error_suffix,
    /* .forced_call_seed      = */ NULL,   /* the render refuses a forced tool_choice (vLLM has none for Inkling) */
    /* .forced_call_prefill   = */ NULL,
    /* .grammar               = */ &k_grammar_inkling,
    /* .find_call_block       = */ inkling_find_call_block,
    /* .tool_defs_text        = */ inkling_tool_defs_text,
};

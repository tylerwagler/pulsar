/* family_gemma4.cpp -- the Gemma 4 family's server table (L338; server_family_ops, chat_family.cpp's switch names it).
 *
 * The frontend in Gemma's configuration (src/lib/gemma4_chat.h): the render is the checkpoint's chat_template.jinja
 * byte for byte (gemma4-chat-gate R1 / R2), the effort map its two levels (thinking off / on: the template has no
 * effort level), the output parser the kit in vLLM's gemma4 reading (gemma4-chat-gate P1 / P2).
 *
 * What is not wired yet, each refused by name or left to the server's own refusal of a NULL hook (L338 T3b-2):
 *   - strict tools (the grammar shape has no call markup: Gemma's call arguments, `{key:<|"|>value<|"|>}`, are no
 *     xgrammar style; a forced tool_choice is seeded with the call's opener, its arguments unconstrained), a tool
 *     loaded mid-conversation (no tool_defs_text);
 *   - the suffix hooks (assistant_turn_sampled, tool_result_tail, tool_error_suffix) and tool memory
 *     (find_call_block): every turn is a full render, which the prefix cache matches as far as the bytes agree;
 *   - images (the family's vision front is not in the container), a tool message answering several calls at once.
 * Departures from vLLM: the parser's (gemma4_chat.h); client text spelling a marker stays text (L223). */
/** @file
 *  The Gemma 4 family's server table (server_family_ops) over src/lib/gemma4_chat. */
#include "pulsar_server_internal.h"
#include "../lib/gemma4_chat.h"

#include <memory>
#include <string>
#include <vector>

namespace {

/* ---- the effort map: off or on (the template's enable_thinking) --------------------------------------------- */

/// The levels a Gemma request's family_effort carries.
enum gemma4_level { GEMMA4_THINK_OFF = 0, GEMMA4_THINK_ON = 1 };

/** server_effort_map::default_level for Gemma 4: thinking on (the server turns it on unless asked not to).
 *  @return GEMMA4_THINK_ON */
int gemma4_effort_default_level(pulsar_engine *) { return GEMMA4_THINK_ON; }

/* none -> off; every named effort -> on (the template has one thinking level) */
/** Gemma 4's effort map (server_family_ops::effort). */
const server_effort_map k_effort_gemma4 = {
    {GEMMA4_THINK_OFF, GEMMA4_THINK_ON, GEMMA4_THINK_ON, GEMMA4_THINK_ON, GEMMA4_THINK_ON, GEMMA4_THINK_ON,
     GEMMA4_THINK_ON},
    NULL, gemma4_effort_default_level, /*think_mode_level=*/false};

/* ---- the render ---------------------------------------------------------------------------------------------- */

/** The renderer's message list; its strings borrow the chat_msgs it was built from. */
struct gemma4_msgs {
    /// The messages in render order.
    std::vector<gemma4_msg_in> m;
    /// Per SOURCE message index: its calls (`m` points in).
    std::vector<std::vector<gemma4_tool_call_in>> calls;
};

/* The parsed messages as the renderer's: the request's system FIELD first (the protocol parsers append it to the
 * array), then the messages in order. */
/** @param msgs the parsed messages (borrowed: must outlive `out`)
 *  @param out the result
 *  @param err the refusal
 *  @param errlen bytes at `err`
 *  @return false + err for an image or a tool message answering several calls */
bool gemma4_messages(const chat_msgs *msgs, gemma4_msgs *out, char *err, size_t errlen) {
    out->m.clear();
    out->calls.assign((size_t)msgs->len, {});
    auto add = [&](int i) -> bool {
        const chat_msg *c = &msgs->v[i];
        if (c->images_len > 0) {
            snprintf(err, errlen, "gemma4: a message carries an image -- the container holds no vision tower; this "
                                  "engine renders text only");
            return false;
        }
        if (c->tool_call_ids_len > 1) {
            snprintf(err, errlen, "gemma4: a tool message answers %d calls at once -- the template renders one result "
                                  "a message", c->tool_call_ids_len);
            return false;
        }
        std::vector<gemma4_tool_call_in> &k = out->calls[(size_t)i];
        for (int j = 0; j < c->calls.len; j++)
            k.push_back({c->calls.v[j].id, c->calls.v[j].name, c->calls.v[j].arguments});
        gemma4_msg_in q = {};
        q.role = c->role;
        q.content = c->content;
        q.reasoning = c->reasoning;
        q.calls = k.empty() ? NULL : k.data();
        q.n_calls = (int)k.size();
        q.tool_call_id = c->tool_call_id ? c->tool_call_id : c->tool_call_ids_len == 1 ? c->tool_call_ids[0] : NULL;
        out->m.push_back(q);
        return true;
    };
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < msgs->len; i++)
            if (msgs->v[i].system_field == (pass == 0) && !add(i)) return false;
    return true;
}

/** server_family_ops::render for Gemma 4.
 *  @param e the engine (NULL = render only: no tokenisation)
 *  @param s the server
 *  @param c the parsed conversation
 *  @param r the request: prompt_text / prompt_spans / prompt, has_tools written
 *  @param err the refusal
 *  @param errlen bytes at `err`
 *  @return false + err for an image, a forced tool choice, or a render the renderer refuses */
bool gemma4_render(pulsar_engine *e, server *s, chat_conversation *c, request *r, char *err, size_t errlen) {
    char *head_raw = NULL;
    if (!chat_openai_tools_json(c, r, &head_raw, err, errlen)) return false;
    const std::unique_ptr<char, void (*)(void *)> head(head_raw, free);
    r->has_tools = r->tools_json != NULL;
    render_prelude(s, c, r);
    gemma4_msgs gm;
    if (!gemma4_messages(&c->msgs, &gm, err, errlen)) return false;
    gemma4_render_in in = {};
    in.msgs = gm.m.data();
    in.n_msgs = (int)gm.m.size();
    in.tools_json = head.get();
    in.thinking = r->family_effort == GEMMA4_THINK_ON;
    in.add_generation_prompt = true;
    gemma4_render_out out;
    if (!gemma4_chat_render(in, &out, err, errlen)) return false;
    free(r->prompt_spans);
    r->prompt_spans = NULL;
    r->prompt_n_spans = (uint32_t)out.spans.size();
    if (!out.spans.empty()) {
        r->prompt_spans = (chat_text_span *)server_xmalloc(out.spans.size() * sizeof(chat_text_span));
        memcpy(r->prompt_spans, out.spans.data(), out.spans.size() * sizeof(chat_text_span));
    }
    r->prompt_text = server_xstrndup(out.text.data(), out.text.size());
    /* L338 P4: a required call (OpenAI "required" or a named function, Anthropic "any" or "tool"): the turn is
     * prefilled into an open call (gemma4_forced_call_seed) and the generation seeds its output with the same bytes */
    if ((c->tool_choice == CHAT_TOOL_CHOICE_ANY || c->tool_choice == CHAT_TOOL_CHOICE_NAMED) && r->has_tools) {
        r->force_tool_call = true;
        if (!request_apply_forced_tool_prefill(r, err, errlen)) return false;
    }
    if (e) pulsar_tokenize_rendered_chat_spans(e, r->prompt_text, r->prompt_spans, r->prompt_n_spans, &r->prompt);
    return true;
}

/* L338 P4: the forced call's opener -- "<|tool_call>call:" (vLLM's reading of Gemma's call body), and for a named
 * tool its name and the arguments' "{".  A call opens from the answer or from reasoning alike (it ends the
 * reasoning: gemma4_chat.h), so the seed is the same with thinking on or off.  The arguments are the model's (the
 * shape has no call markup to constrain them with). */
/** server_family_ops::forced_call_seed for Gemma 4.
 *  @param r the request (forced_tool_name)
 *  @param out appended to: the seed bytes */
void gemma4_forced_call_seed(const request *r, buf *out) {
    buf_puts(out, "<|tool_call>call:");
    if (r->forced_tool_name && r->forced_tool_name[0]) {
        buf_puts(out, r->forced_tool_name);
        buf_puts(out, "{");
    }
}

/** server_family_ops::forced_call_prefill for Gemma 4: the prompt kept whole, the seed appended.
 *  @param r the request
 *  @param append receives the seed */
void gemma4_forced_call_prefill(const request *r, const char *, size_t *, buf *append) {
    gemma4_forced_call_seed(r, append);
}

/* the longest prefix two UNRELATED prompts share by template construction: <bos> (a system turn is not always
 * written) */
/** server_family_ops::trivial_header_tokens for Gemma 4.
 *  @param e the engine (its tokenizer)
 *  @return the tokens of "<bos>" */
int gemma4_trivial_header_tokens(pulsar_engine *e) {
    pulsar_tokens t = {0};
    pulsar_tokenize_rendered_chat(e, "<bos>", &t);
    const int n = t.len;
    pulsar_tokens_free(&t);
    return n;
}

/* ---- the output parser: the kit in Gemma's configuration, typed by the request's tools, starting in reasoning
 *      exactly when the prompt leaves the turn inside an open <|channel> ------------------------------------------ */

/** server_output_parser_ops::create for Gemma 4.
 *  @param g the decode attempt (its job's request)
 *  @param err the failure
 *  @param errlen bytes at `err`
 *  @return the parser state (parser_kit_destroy frees it), or NULL + err */
void *gemma4_parser_create(server *, gen_state *g, char *err, size_t errlen) {
    gemma4_output_parser *p = new gemma4_output_parser();
    char perr[200];
    if (!p->init(gemma4_prompt_opens_reasoning(g->j->req.prompt_text), g->j->req.tools_json, perr, sizeof perr)) {
        snprintf(err, errlen, "gemma4 output parser: %s", perr);
        delete p;
        return NULL;
    }
    return parser_kit_start(p);
}

/** Gemma 4's output parser table: the parser kit's hooks. */
const server_output_parser_ops k_parser_gemma4 = {
    /* .create        = */ gemma4_parser_create,
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

}  // namespace

/* L338 P4: the response formats' shape -- the answer opens where the thought channel closes (thinking off, the
 * prompt already closed it: "<|channel>thought\n<channel|>"), Qwen's arrangement around "</think>"; no call markup
 * (strict tools refused by name, constrain_plan_build). */
/** Gemma 4's grammar shape (server_family_ops::grammar): response formats only. */
static const server_grammar_shape k_grammar_gemma4 = {
    /* .answer_open           = */ "<channel|>",
    /* .answer_at_think_close = */ true,
    /* .answer_close          = */ "",
    /* .call_trigger          = */ NULL,   /* no call markup: strict tools refused */
    /* .block_head            = */ "",
    /* .name_open             = */ "",
    /* .name_close            = */ "",
    /* .style                 = */ NULL,
    /* .call_close            = */ "",
    /* .repeat                = */ false,
    /* .block_tail            = */ "",
};

const server_family_ops k_family_gemma4 = {
    /* .name                  = */ "Gemma 4",
    /* .format                = */ PULSAR_CHAT_GEMMA4,
    /* .v41                   = */ false,
    /* .effort                = */ &k_effort_gemma4,
    /* .render                = */ gemma4_render,
    /* .trivial_header_tokens = */ gemma4_trivial_header_tokens,
    /* .output                = */ &k_parser_gemma4,
    /* .assistant_turn_sampled = */ NULL,   /* T3b-2: every turn is a full render */
    /* .tool_result_tail      = */ NULL,
    /* .tool_error_suffix     = */ NULL,
    /* .forced_call_seed      = */ gemma4_forced_call_seed,      /* L338 P4 */
    /* .forced_call_prefill   = */ gemma4_forced_call_prefill,
    /* .grammar               = */ &k_grammar_gemma4,   /* response formats; strict tools refused by name */
    /* .find_call_block       = */ NULL,
    /* .tool_defs_text        = */ NULL,
};

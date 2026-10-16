/* family_nemotron_h.cpp -- the Nemotron-H family's server table (L296 N8; server_family_ops, chat_family.cpp's switch
 * names it).
 *
 * Everything here is F5's frontend in Nemotron's configuration (src/lib/nemotron_chat.h): the render is the
 * checkpoint's chat_template.jinja byte for byte (nemotron-chat-gate), the effort map its three levels, the output
 * parser the kit in vLLM's nemotron_v3 configuration (qwen3_coder's XML calls, R1-style think split), and every
 * suffix hook one of the renderer's own pieces -- so the prompt so far + a hook's bytes == the full render of the
 * longer conversation wherever the template composes (nemotron-chat-gate S1).
 *
 * What is refused by name, never rendered in another spelling:
 *   - a forced tool_choice (required / a named function): the seed and prefill that open a call past the template's
 *     "<think></think>" are not built for this family yet (forced_call_* and forced_name_* NULL);
 *   - images: the template's <image> placeholders need the family's vision front (N11), not wired;
 *   - the template variables beyond the shared ones (medium_effort, truncate_history_thinking, force_nonempty_content):
 *     the server reads enable_thinking and reasoning_effort only; "medium" is the medium_effort level.
 * Departures from vLLM: client text spelling a marker stays text (L223); a call the turn ends inside is broken
 * (L272); a message's text parts arrive joined. */
#include "pulsar_server_internal.h"
#include "../lib/nemotron_chat.h"

#include <string>
#include <vector>

namespace {

/* ---- the effort map: the template's three levels ------------------------------------------------------------ */

int nemotron_effort_default_level(pulsar_engine *) { return (int)nemotron_effort_default(); }

/* none -> thinking off; minimal, low, medium -> medium_effort; high, xhigh, max -> the template's default */
const server_effort_map k_effort_nemotron_h = {
    {NEMOTRON_EFFORT_NONE, NEMOTRON_EFFORT_MEDIUM, NEMOTRON_EFFORT_MEDIUM, NEMOTRON_EFFORT_MEDIUM, NEMOTRON_EFFORT_HIGH,
     NEMOTRON_EFFORT_HIGH, NEMOTRON_EFFORT_HIGH},
    NULL, nemotron_effort_default_level, /*think_mode_level=*/false};

/* ---- the render ---------------------------------------------------------------------------------------------- */

/* The parsed messages as the renderer's, borrowing `msgs`: the request's system FIELD first (the protocol parsers
 * append it to the array), then the messages in order -- or, for a TAIL, the messages from `start`, the system field
 * not among them.  false + err: an image (N11's front is not wired). */
struct nemotron_msgs {
    std::vector<nemotron_msg_in> m;
    std::vector<std::vector<nemotron_tool_call_in>> calls;
};

bool nemotron_messages(const chat_msgs *msgs, int start, bool tail, nemotron_msgs *out, char *err, size_t errlen) {
    out->m.clear();
    out->calls.assign((size_t)msgs->len, {});
    auto add = [&](int i) -> bool {
        const chat_msg *c = &msgs->v[i];
        if (c->images_len > 0) {
            snprintf(err, errlen, "nemotron_h: a message carries an image -- the family's vision front (N11) is not "
                                  "wired; this engine renders text only");
            return false;
        }
        std::vector<nemotron_tool_call_in> &k = out->calls[(size_t)i];
        for (int j = 0; j < c->calls.len; j++) k.push_back({c->calls.v[j].name, c->calls.v[j].arguments});
        nemotron_msg_in q = {};
        q.role = c->role;
        q.content = c->content;
        q.reasoning = c->reasoning;
        q.calls = k.empty() ? NULL : k.data();
        q.n_calls = (int)k.size();
        q.raw_calls = c->calls.raw_calls;
        out->m.push_back(q);
        return true;
    };
    if (tail) {
        for (int i = start; i < msgs->len; i++)
            if (!msgs->v[i].system_field && !add(i)) return false;
        return true;
    }
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < msgs->len; i++)
            if (msgs->v[i].system_field == (pass == 0) && !add(i)) return false;
    return true;
}

/* A render's text and client-data ranges as the request owns them (malloc'd; the ranges NULL when none). */
char *nemotron_take_render(const nemotron_render_out &out, chat_text_span **spans_out, uint32_t *n_spans_out) {
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

bool nemotron_render(pulsar_engine *e, server *s, chat_conversation *c, request *r, char *err, size_t errlen) {
    if (c->tool_choice == CHAT_TOOL_CHOICE_ANY || c->tool_choice == CHAT_TOOL_CHOICE_NAMED) {
        snprintf(err, errlen, "tool_choice %s: Nemotron-H's forced tool call is not built yet -- send auto or none",
                 c->tool_choice == CHAT_TOOL_CHOICE_ANY ? "required" : "naming a function");
        return false;
    }
    if (!chat_openai_tools_json(c, r, err, errlen)) return false;
    r->has_tools = r->tools_json != NULL;
    render_prelude(s, c, r);
    nemotron_msgs nm;
    if (!nemotron_messages(&c->msgs, 0, false, &nm, err, errlen)) return false;
    nemotron_render_in in = {};
    in.msgs = nm.m.data();
    in.n_msgs = (int)nm.m.size();
    in.tools_json = r->tools_json;
    in.effort = (nemotron_effort)r->family_effort;
    in.add_generation_prompt = true;
    nemotron_render_out out;
    if (!nemotron_chat_render(in, &out, err, errlen)) return false;
    free(r->prompt_spans);
    r->prompt_text = nemotron_take_render(out, &r->prompt_spans, &r->prompt_n_spans);
    if (e) {
        pulsar_tokenize_rendered_chat_spans(e, r->prompt_text, r->prompt_spans, r->prompt_n_spans, &r->prompt);
        if (!request_prepare_images(e, &c->msgs, r, err, errlen)) return false;
    }
    return true;
}

/* L272 B6: the longest prefix two UNRELATED prompts share by template construction: the system turn's opener (every
 * Nemotron prompt opens with it, empty or not). */
int nemotron_trivial_header_tokens(pulsar_engine *e) {
    pulsar_tokens t = {0};
    pulsar_tokenize_rendered_chat(e, "<|im_start|>system\n", &t);
    const int n = t.len;
    pulsar_tokens_free(&t);
    return n;
}

/* ---- the output parser: the kit in Nemotron's configuration, typed by the tools the prompt rendered, starting in
 *      reasoning exactly when the generation prompt opened "<think>\n" -------------------------------------------- */

void *nemotron_parser_create(server *, gen_state *g, char *err, size_t errlen) {
    nemotron_output_parser *p = new nemotron_output_parser();
    char perr[200];
    if (!p->init(pulsar_think_mode_enabled(g->j->req.think_mode), g->j->req.tools_json, perr, sizeof perr)) {
        snprintf(err, errlen, "nemotron_h output parser: %s", perr);
        delete p;
        return NULL;
    }
    return parser_kit_start(p);
}

const server_output_parser_ops k_parser_nemotron_h = {
    /* .create        = */ nemotron_parser_create,
    /* .destroy       = */ parser_kit_destroy,
    /* .seed          = */ parser_kit_seed,
    /* .feed          = */ parser_kit_feed,
    /* .in_tool_call  = */ parser_kit_in_tool_call,
    /* .turn_complete = */ parser_kit_turn_complete,
    /* .tool_progress = */ parser_kit_tool_progress,
    /* .finish        = */ parser_kit_finish,
};

/* ---- step 4: the suffix hooks, each one of the renderer's pieces --------------------------------------------- */

char *nemotron_assistant_turn_sampled(const request *, bool think, const char *reasoning, const char *content,
                                      const tool_calls *calls, chat_text_span **spans_out, uint32_t *n_spans_out) {
    std::vector<nemotron_tool_call_in> k;
    for (int j = 0; calls && j < calls->len; j++) k.push_back({calls->v[j].name, calls->v[j].arguments});
    nemotron_msg_in m = {};
    m.role = "assistant";
    m.content = content;
    m.reasoning = reasoning;
    m.calls = k.empty() ? NULL : k.data();
    m.n_calls = (int)k.size();
    m.raw_calls = calls ? calls->raw_calls : NULL;
    nemotron_render_out out;
    char err[200];
    if (!nemotron_chat_render_assistant_turn(m, think, &out, err, sizeof err)) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: Nemotron-H sampled-turn render refused: %s", err);
        if (spans_out) { *spans_out = NULL; *n_spans_out = 0; }
        return NULL;
    }
    return nemotron_take_render(out, spans_out, n_spans_out);
}

char *nemotron_tool_result_tail(const request *r, const chat_msgs *msgs, int start, chat_text_span **spans_out,
                                uint32_t *n_spans_out) {
    if (spans_out) { *spans_out = NULL; *n_spans_out = 0; }
    nemotron_msgs nm;
    nemotron_render_out out;
    char err[200];
    if (!nemotron_messages(msgs, start, true, &nm, err, sizeof err) ||
        !nemotron_chat_render_tail(nm.m.data(), (int)nm.m.size(), (nemotron_effort)r->family_effort, &out, err,
                                   sizeof err)) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: Nemotron-H tool-result tail refused: %s", err);
        return NULL;
    }
    return nemotron_take_render(out, spans_out, n_spans_out);
}

/* The model-visible retry, as on every family: the malformed turn ended at its stop token, and the error -- with the
 * system turn reminded (the tools block and the system text) -- is ONE tool result, then the generation prompt. */
char *nemotron_tool_error_suffix(const request *r, const thinking_state *, const char *detail, chat_text_span **spans_out,
                                 uint32_t *n_spans_out) {
    std::string text = "Tool error: malformed tool call";
    if (detail && detail[0]) text += std::string(": ") + detail;
    text += "\nThe previous assistant output was not executed because its <tool_call> block was malformed. Emit a new "
            "valid <tool_call>, or answer normally if no tool is needed.";
    const std::string system = nemotron_rendered_system_text(r->prompt_text);
    if (!system.empty()) text += "\n\nSystem prompt reminder:\n" + system;
    chat_msgs msgs = {0};
    chat_msg result = {0};
    result.role = xstrdup("tool");
    result.content = xstrdup(text.c_str());
    chat_msgs_push(&msgs, result);
    char *suffix = nemotron_tool_result_tail(r, &msgs, 0, spans_out, n_spans_out);
    chat_msgs_free(&msgs);
    return suffix;
}

}  // namespace

const server_family_ops k_family_nemotron_h = {
    /* .name                  = */ "Nemotron-H",
    /* .format                = */ PULSAR_CHAT_NEMOTRON_H,
    /* .v41                   = */ false,
    /* .effort                = */ &k_effort_nemotron_h,
    /* .render                = */ nemotron_render,
    /* .trivial_header_tokens = */ nemotron_trivial_header_tokens,
    /* .output                = */ &k_parser_nemotron_h,
    /* .assistant_turn_sampled = */ nemotron_assistant_turn_sampled,
    /* .tool_result_tail      = */ nemotron_tool_result_tail,
    /* .tool_error_suffix     = */ nemotron_tool_error_suffix,
    /* .forced_call_seed      = */ NULL,   /* the render refuses a forced tool_choice (not built for this family) */
    /* .forced_call_prefill   = */ NULL,
    /* .forced_name_open      = */ NULL,
    /* .forced_name_close     = */ NULL,
    /* .find_call_block       = */ nemotron_find_call_block,
};

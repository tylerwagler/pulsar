/** @file
 *  The Qwen 3.8 family's server table (server_family_ops) over its chat frontend (qwen_chat.h). */
/* qwen_server.cpp -- Qwen's part of the server's family code, moved out of src/server/chat_family.cpp (L295 F8 Q4,
 * byte for byte): its effort map, the conversion of the parsed conversation into the renderer's messages, the render,
 * the suffix hooks of one sampled turn, the output parser's table, the trivial-match header and the family table,
 * registered by chat format (SERVER_FAMILY).  The shared helpers it calls (chat_strip_image_markers,
 * chat_openai_tools_json, chat_expand_tool_refs, k_grammar_qwen_xml -- Nemotron-H writes the same markup) stay in
 * chat_family.cpp. */
#include "server/pulsar_server_internal.h"
#include "qwen_chat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <memory>
#include <string>
#include <vector>

/* ---- Qwen ----------------------------------------------------------------------------------------- */

/* Qwen's efforts are its template's reasoning_effort names (low, medium, xhigh; "none" is
 * enable_thinking=false); an integer is the nearest preset's; nothing sent is the template's default (xhigh).
 * Downstream reads only whether a think block is open, so think_mode carries PULSAR_THINK_DEFAULT (the
 * enabled marker, not a DeepSeek effort); the render reads family_effort. */
/// server_effort_map::default_level for Qwen.
/// @return the template's default qwen_effort
static int qwen_effort_default_level(pulsar_engine *) { return (int)qwen_effort_default(); }
/** Qwen's effort map (levels are qwen_effort values). */
static const server_effort_map k_effort_qwen = {
    {QWEN_EFFORT_NONE, QWEN_EFFORT_LOW, QWEN_EFFORT_LOW, QWEN_EFFORT_MEDIUM, QWEN_EFFORT_XHIGH, QWEN_EFFORT_XHIGH,
     QWEN_EFFORT_XHIGH},
    NULL, qwen_effort_default_level, /*think_mode_level=*/false};

/* chat_msgs[start..) as the renderer's messages; the pointers borrow `msgs` and `notes`.  The system
 * FIELD renders first (the parser appends it to the array); a system message that is not the
 * conversation's first turns into a user <system-reminder> turn (the template has no in-place system
 * turn; `notes` owns that text); a replayed call carries its sampled bytes when tool memory found them.
 * `tail`: a continuation tail -- no entry is the conversation's first, and the system field is not
 * part of one.  false + err: an image whose parser marker is not at its offset. */
/// @param msgs   the conversation
/// @param start  the first message to convert
/// @param tail   convert a continuation tail
/// @param qm     receives the renderer's messages
/// @param qc     per message, the calls the renderer's messages point into (owned here)
/// @param notes  per message, the system-reminder text a message points into
/// @param imgs   per message, the stripped content and image offsets a message points into
/// @param err     receives the reason on false
/// @param errlen  size of `err`
/// @return false + err on a misplaced image marker
static bool qwen_messages(const chat_msgs *msgs, int start, bool tail, std::vector<qwen_msg_in> *qm,
                          std::vector<std::vector<qwen_tool_call_in>> *qc, std::vector<std::string> *notes,
                          std::vector<chat_msg_images> *imgs, char *err, size_t errlen) {
    qc->assign((size_t)msgs->len, {});
    notes->assign((size_t)msgs->len, std::string());
    imgs->assign((size_t)msgs->len, chat_msg_images());
    for (int pass = 0; pass < 2; pass++) {
        for (int i = start; i < msgs->len; i++) {
            const chat_msg *m = &msgs->v[i];
            if (m->system_field != (pass == 0)) continue;
            if (tail && m->system_field) continue;
            /* L268: images render as the template's vision literal here and in a live tail alike; the server places a
             * tail's new images on the live history (image_continuation_place, every family) */
            if (m->images_len > 0 && !chat_strip_image_markers(m, &(*imgs)[(size_t)i], err, errlen)) return false;
            for (int k = 0; k < m->calls.len; k++)
                (*qc)[(size_t)i].push_back({m->calls.v[k].name, m->calls.v[k].arguments});
            if (!strcmp(m->role, "system") && (tail || !qm->empty())) {
                (*notes)[(size_t)i] = qwen_system_reminder(m->content);
                qm->push_back({"user", (*notes)[(size_t)i].c_str(), NULL, NULL, 0, NULL});
                continue;
            }
            const chat_msg_images &mi = (*imgs)[(size_t)i];
            qm->push_back({m->role, m->images_len > 0 ? mi.content.c_str() : m->content, m->reasoning,
                           (*qc)[(size_t)i].empty() ? NULL : (*qc)[(size_t)i].data(), m->calls.len,
                           m->calls.raw_calls, mi.at.empty() ? NULL : mi.at.data(), (int)mi.at.size()});
        }
    }
    return true;
}

/* A render's text and client-data ranges as the request owns them (malloc'd; the ranges NULL when
 * none). */
/// @param out          the render (its text and spans are copied)
/// @param spans_out    receives the ranges (malloc'd, NULL when none); NULL: not wanted
/// @param n_spans_out  receives how many (written only with spans_out)
/// @return the text (malloc'd)
static char *qwen_take_render(qwen_render_out *out, chat_text_span **spans_out, uint32_t *n_spans_out) {
    if (spans_out) {
        *spans_out = NULL;
        *n_spans_out = (uint32_t)out->spans.size();
        if (!out->spans.empty()) {
            *spans_out = (chat_text_span *)server_xmalloc(out->spans.size() * sizeof(chat_text_span));
            memcpy(*spans_out, out->spans.data(), out->spans.size() * sizeof(chat_text_span));
        }
    }
    return server_xstrndup(out->text.data(), out->text.size());
}

/* Qwen's: the tools block's entries inside <tools></tools> (qwen_chat_tool_defs). */
/// server_family_ops::tool_defs_text for Qwen.
/// @param lines   one function schema a line (parse_tools_value)
/// @param len     bytes at `lines`
/// @param err     receives the reason on NULL
/// @param errlen  size of `err`
/// @return the definitions (malloc'd), or NULL + err
static char *qwen_tool_defs_text(const char *lines, size_t len, char *err, size_t errlen) {
    char *json = tool_schema_lines_openai_tools(lines, len, err, errlen);
    if (!json) return NULL;
    qwen_render_out out;
    const bool ok = qwen_chat_tool_defs(json, &out, err, errlen);
    free(json);
    return ok ? server_xstrndup(out.text.data(), out.text.size()) : NULL;
}

/* HF's apply_chat_template byte for byte (qwen_chat_render, with the L223 client-span map), tokenised
 * with the family's markers; the tools in the template's shape, typed for the output parser.  A forced
 * tool_choice prefills the turn into an open <tool_call> (qwen_forced_call_prefill); a tool-result-only
 * request continues the live KV with the family's tail (render_prelude).  Images are the template's vision
 * literal in place (L268), expanded by the core's walk.  The tools are the conversation's schema lines -- every
 * protocol's kinds and tool_search's loads, as for DeepSeek (L284) -- in OpenAI's shape; an OpenAI request's
 * array goes to the template as sent (HF's input). */
/// server_family_ops::render for Qwen.
/// @param e       the engine (NULL: render the text only, no tokens -- the renderer gate)
/// @param s       the server (NULL: no tool memory, no live continuations)
/// @param c       the parsed conversation (its messages are folded / annotated in place)
/// @param r       the request: receives prompt_text, prompt_spans, prompt and the forced-call flags
/// @param err     receives the reason on false
/// @param errlen  size of `err`
/// @return false + err when the conversation cannot be rendered
static bool qwen_render(pulsar_engine *e, server *s, chat_conversation *c, request *r, char *err, size_t errlen) {
    const qwen_effort qe = (qwen_effort)r->family_effort;
    char *head_raw = NULL;
    if (!chat_openai_tools_json(c, r, &head_raw, err, errlen)) return false;
    const std::unique_ptr<char, void (*)(void *)> head(head_raw, free);
    r->has_tools = r->tools_json != NULL;
    render_prelude(s, c, r);
    std::vector<qwen_msg_in> qm;
    std::vector<std::vector<qwen_tool_call_in>> qc;
    std::vector<std::string> notes;
    std::vector<chat_msg_images> imgs;
    if (!qwen_messages(&c->msgs, 0, false, &qm, &qc, &notes, &imgs, err, errlen)) return false;
    qwen_render_out out;
    if (!qwen_chat_render({qm.data(), (int)qm.size(), head.get(), qe, true}, &out, err, errlen)) return false;
    free(r->prompt_spans);
    r->prompt_text = qwen_take_render(&out, &r->prompt_spans, &r->prompt_n_spans);
    /* A required call (OpenAI "required" or a named function, Anthropic "any" or "tool"): the turn is
     * prefilled past thinking into an open <tool_call> (a named <function= for a named tool), and the
     * generation seeds its output with the same bytes. */
    const bool forced = c->tool_choice == CHAT_TOOL_CHOICE_ANY || c->tool_choice == CHAT_TOOL_CHOICE_NAMED;
    if (forced && r->has_tools) {
        r->force_tool_call = true;
        if (!request_apply_forced_tool_prefill(r, err, errlen)) return false;
    }
    /* With an engine: tokenise the rendered text, then the core's expansion puts each image's block where its
     * placeholder token sits (request_prepare_images, DeepSeek's path too) */
    if (e) {
        if (!request_tokenize_rendered(e, r, err, errlen)) return false;
        if (!request_prepare_images(e, &c->msgs, r, err, errlen)) return false;
    }
    return true;
}

/* ---- step 4: Qwen's suffix hooks -- the template's bytes for one sampled turn, one tail, one tool
 *      error, one forced call, and where a run of <tool_call> blocks sits in a transcript.  Each is
 *      qwen_chat_render's own writer (src/families/qwen/qwen_chat.cpp), so prefix + hook = the full render by
 *      construction (test_qwen_hooks_compose_to_the_full_render). ---------------------------------- */

/// server_family_ops::assistant_turn_sampled for Qwen.
/// @param think      the turn had thinking on
/// @param reasoning  the replayed reasoning (NULL: none)
/// @param content    the turn's content
/// @param calls      its calls (may be NULL)
/// @param spans_out    receives the suffix's client-data ranges (malloc'd, NULL when none); may be NULL
/// @param n_spans_out  receives how many
/// @return the bytes (malloc'd), or NULL when the template refuses the turn (logged)
static char *qwen_assistant_turn_sampled(const request *, bool think, const char *reasoning, const char *content,
                                         const tool_calls *calls, chat_text_span **spans_out, uint32_t *n_spans_out) {
    std::vector<qwen_tool_call_in> qc;
    for (int k = 0; calls && k < calls->len; k++) qc.push_back({calls->v[k].name, calls->v[k].arguments});
    const qwen_msg_in m = {"assistant", content, reasoning, qc.empty() ? NULL : qc.data(), calls ? calls->len : 0,
                           calls ? calls->raw_calls : NULL};
    qwen_render_out out;
    char err[200];
    if (!qwen_chat_render_assistant_turn(m, think, &out, err, sizeof err)) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: Qwen sampled-turn render refused: %s", err);
        if (spans_out) { *spans_out = NULL; *n_spans_out = 0; }
        return NULL;
    }
    return qwen_take_render(&out, spans_out, n_spans_out);
}

/// server_family_ops::tool_result_tail for Qwen.
/// @param r      the request (its effort)
/// @param msgs   the conversation
/// @param start  the first new user-side message
/// @param spans_out    receives the suffix's client-data ranges (malloc'd, NULL when none); may be NULL
/// @param n_spans_out  receives how many
/// @return the tail (malloc'd), or NULL when it cannot be rendered
static char *qwen_tool_result_tail(const request *r, const chat_msgs *msgs, int start, chat_text_span **spans_out,
                                   uint32_t *n_spans_out) {
    if (spans_out) { *spans_out = NULL; *n_spans_out = 0; }
    std::vector<qwen_msg_in> qm;
    std::vector<std::vector<qwen_tool_call_in>> qc;
    std::vector<std::string> notes;
    std::vector<chat_msg_images> imgs;
    char err[200];
    qwen_render_out out;
    if (!qwen_messages(msgs, start, true, &qm, &qc, &notes, &imgs, err, sizeof err) ||
        !qwen_chat_render_tail(qm.data(), (int)qm.size(), (qwen_effort)r->family_effort, &out, err, sizeof err))
        return NULL;
    return qwen_take_render(&out, spans_out, n_spans_out);
}

/* The system turn's body in a rendered Qwen prompt (the tools block and the client's system text), or
 * NULL when the prompt has none: the reminder a tool error carries. */
/// @param prompt  the rendered prompt (NULL: none)
/// @return the region (malloc'd, trailing whitespace trimmed), or NULL
static char *qwen_rendered_system_region(const char *prompt) {
    static const char open[] = "<|im_start|>system\n";
    if (!prompt || strncmp(prompt, open, sizeof open - 1)) return NULL;
    const char *p = prompt + (sizeof open - 1);
    const char *end = strstr(p, "<|im_end|>");
    if (!end) return NULL;
    while (end > p && isspace((unsigned char)end[-1])) end--;
    return server_xstrndup(p, (size_t)(end - p));
}

/// server_family_ops::tool_error_suffix for Qwen: one tool result naming the error, with the system region reminded.
/// @param r       the request (its prompt_text and effort)
/// @param detail  what was wrong (may be NULL or empty)
/// @param spans_out    receives the suffix's client-data ranges (malloc'd, NULL when none); may be NULL
/// @param n_spans_out  receives how many
/// @return the suffix (malloc'd), or NULL when it cannot be rendered
static char *qwen_tool_error_suffix(const request *r, const thinking_state *, const char *detail,
                                    chat_text_span **spans_out, uint32_t *n_spans_out) {
    /* the turn ended at the stop token whatever its think state: the tail closes it */
    char *system = qwen_rendered_system_region(r->prompt_text);
    buf text = {0};
    buf_puts(&text, "Tool error: malformed tool call");
    if (detail && detail[0]) {
        buf_puts(&text, ": ");
        buf_puts(&text, detail);
    }
    buf_puts(&text, "\nThe previous assistant output was not executed because its <tool_call> block was malformed. "
                    "Emit a new valid <tool_call>, or answer normally if no tool is needed.");
    if (system && system[0]) {
        buf_puts(&text, "\n\nSystem prompt reminder:\n");
        buf_puts(&text, system);
    }
    chat_msgs msgs = {0};
    chat_msg result = {0};
    result.role = server_xstrdup("tool");
    result.content = buf_take(&text);
    chat_msgs_push(&msgs, result);
    char *suffix = qwen_tool_result_tail(r, &msgs, 0, spans_out, n_spans_out);
    chat_msgs_free(&msgs);
    free(system);
    return suffix;
}

/* "Thinking skipped, a call opened" in the template's spelling: with thinking on the generation prompt
 * opened "<think>\n", so the seed closes the empty block the way the template renders one and opens the
 * call; with thinking off the prompt already closed it. */
/// @param r    the request (its effort and forced_tool_name)
/// @param out  the seed is appended here
static void qwen_forced_call_seed(const request *r, buf *out) {
    if ((qwen_effort)r->family_effort != QWEN_EFFORT_NONE) buf_puts(out, "\n</think>\n\n");
    /* unnamed: stop before the name's opener -- the model samples "=name>" under the call grammar (L303), as
     * the tokenizer joins them ("=get"); named: the whole tag, tokenized with the prompt as the model saw it */
    buf_puts(out, "<tool_call>\n<function");
    if (r->forced_tool_name && r->forced_tool_name[0]) {
        buf_puts(out, "=");
        buf_puts(out, r->forced_tool_name);
        buf_puts(out, ">\n");
    }
}

/// server_family_ops::forced_call_prefill for Qwen: the prompt is kept whole and the seed appended.
/// @param r       the request
/// @param append  receives the seed
static void qwen_forced_call_prefill(const request *r, const char *, size_t *, buf *append) {
    qwen_forced_call_seed(r, append);   /* the prompt keeps its generation prompt whole */
}

/* The run of consecutive <tool_call> blocks (whitespace between them) at or after `p`: one turn's calls,
 * the bytes the parser recorded as the turn's raw_calls. */
/// server_family_ops::find_call_block for Qwen.
/// @param p    where to search
/// @param end  receives one past the run
/// @return the run's start, or NULL (none, or an unclosed block)
static const char *qwen_find_call_block(const char *p, const char **end) {
    static const char open[] = "<tool_call>", close[] = "</tool_call>";
    const char *start = strstr(p, open);
    if (!start) return NULL;
    const char *e = start;
    for (;;) {
        const char *c = strstr(e, close);
        if (!c) return NULL;   /* an unclosed block is not a key */
        e = c + (sizeof close - 1);
        const char *q = e;
        while (*q == '\n' || *q == ' ' || *q == '\t' || *q == '\r') q++;
        if (strncmp(q, open, sizeof open - 1)) break;
        e = q;
    }
    *end = e;
    return start;
}

/* Qwen's output parser: the kit in Qwen's configuration (qwen_output_parser), typed by the tools the prompt
 * rendered, starting in reasoning exactly when the generation prompt opened "<think>\n"; every other hook
 * is the kit's (parser_kit.cpp). */
/// @param g       the attempt (its request's think mode and tools_json)
/// @param err     receives the reason on NULL
/// @param errlen  size of `err`
/// @return the kit state (parser_kit_start), or NULL + err when the parser cannot be configured
static void *qwen_parser_create(server *, gen_state *g, char *err, size_t errlen) {
    qwen_output_parser *p = new qwen_output_parser();
    char perr[200];
    if (!p->init(pulsar_think_mode_enabled(g->j->req.think_mode), g->j->req.tools_json, perr, sizeof perr)) {
        snprintf(err, errlen, "qwen output parser: %s", perr);
        delete p;
        return NULL;
    }
    return parser_kit_start(p);
}

/** Qwen's output parser table: qwen_parser_create and the kit's shared hooks. */
static const server_output_parser_ops k_parser_qwen = {
    /* .create        = */ qwen_parser_create,
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

/** L272 B6: Qwen's trivial-match header (server::slot_trivial_common_tokens): the system turn's opener -- no BOS,
 *  no effort line. */
/// @param e  the engine (tokenizer)
/// @return tokens in the system turn's opener
static int qwen_trivial_header_tokens(pulsar_engine *e) {
    pulsar_tokens t = {0};
    if (!pulsar_tokenize_rendered_chat(e, "<|im_start|>system\n", &t))
        pulsar_die("qwen: the tokenizer refused the trivial header");
    const int hdr_len = t.len;
    pulsar_tokens_free(&t);
    return hdr_len;
}

/** Qwen's family table. */
static const server_family_ops k_family_qwen = {
    /* .name                  = */ "Qwen",
    /* .format                = */ PULSAR_CHAT_QWEN,
    /* .v41                   = */ false,
    /* .effort                = */ &k_effort_qwen,
    /* .render                = */ qwen_render,
    /* .trivial_header_tokens = */ qwen_trivial_header_tokens,
    /* .output                = */ &k_parser_qwen,
    /* .assistant_turn_sampled = */ qwen_assistant_turn_sampled,
    /* .tool_result_tail      = */ qwen_tool_result_tail,
    /* .tool_error_suffix     = */ qwen_tool_error_suffix,
    /* .forced_call_seed      = */ qwen_forced_call_seed,
    /* .forced_call_prefill   = */ qwen_forced_call_prefill,
    /* .grammar               = */ &k_grammar_qwen_xml,
    /* .find_call_block       = */ qwen_find_call_block,
    /* .tool_defs_text        = */ qwen_tool_defs_text,
    /* .reasoning             = */ NULL,   /* <think> / </think> */
    /* .call_tail             = */ "",   /* </tool_call> ends a call (the "\n" between two opens the next) */
};
SERVER_FAMILY(k_family_qwen)

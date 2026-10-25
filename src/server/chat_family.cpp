/* L267: the model-family half of a chat request.
 *
 * A protocol parser (api_parse.cpp: OpenAI chat, Anthropic Messages, Responses) reads its wire format
 * into a chat_conversation -- the messages, the tools as sent, the tool_choice, and the thinking / effort
 * controls exactly as sent.  render_chat_conversation() is everything after that, and it is the loaded
 * family's: how the controls resolve to a thinking mode, the protocol's tool-result checks (they read
 * the resolved mode), then the prompt.  DeepSeek renders its own template (prompt_render.cpp) with its
 * tool memory, live-continuation suffixes and forced-call prefill; Qwen renders HF's template
 * (src/lib/qwen_chat) with the same memory, continuations and prefill behind its own hooks (L272 P3
 * step 4), and refuses by name what that template cannot express.  A new family adds one table -- in its own
 * file (Inkling: family_inkling.cpp, over F5's src/lib/inkling_chat) -- and one case in
 * server_family_for_format; a new protocol adds one parser and no arm. */
#include "pulsar_server_internal.h"
#include "deepseek_dsml.h"
#include "../lib/qwen_chat.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

void chat_conversation_free(chat_conversation *c) {
    if (!c) return;
    chat_msgs_free(&c->msgs);
    free(c->tools_raw);
    free(c->tool_schemas);
    free(c->deferred_tool_schemas);
    buf_free(&c->loaded_tool_schemas);
    free(c->tool_choice_wire);
    for (int i = 0; i < c->n_controls; i++) free(c->controls[i].raw);
    free(c->controls);
    response_format_free(&c->response_format);
    memset(c, 0, sizeof(*c));
}

void chat_conversation_control(chat_conversation *c, const char *key, char *raw) {
    c->controls = (chat_control *)server_xrealloc(c->controls, (size_t)(c->n_controls + 1) * sizeof(c->controls[0]));
    c->controls[c->n_controls++] = {key, raw};
}



/* The renderer's half of prepare_vl_inputs(): the parsed messages carry their
 * inline images; this gathers them in message/content order, copies the encoded
 * bytes into the request (the messages are freed when parsing returns), and
 * lets the engine replace each rendered PULSAR_IMAGE_PLACEHOLDER with that
 * image's sentinel block, filling each start_pos.  A text-only request is left
 * untouched -- its sync path is pulsar_session_sync(). */
bool request_prepare_images(pulsar_engine *e, const chat_msgs *msgs,
                            request *r, char *err, size_t errlen) {
    int n = 0;
    for (int i = 0; i < msgs->len; i++) n += msgs->v[i].images_len;
    /* A request with no images and no placeholder text is the plain text path:
     * leave r->prompt exactly as tokenized.  A placeholder with no image is a
     * malformed image request, not text, and must be refused -- it is checked
     * by the expander below (and by the engine's own text-path scan). */
    const bool placeholder = r->prompt_text &&
                             strstr(r->prompt_text, PULSAR_IMAGE_PLACEHOLDER) != NULL;
    if (n == 0 && !placeholder) return true;
    if (n > 0) {
        r->images = (pulsar_image_ref *)server_xmalloc((size_t)n * sizeof(r->images[0]));
        memset(r->images, 0, (size_t)n * sizeof(r->images[0]));
        r->n_images = n;
        int k = 0;
        for (int i = 0; i < msgs->len; i++) {
            for (int j = 0; j < msgs->v[i].images_len; j++, k++) {
                const size_t len = msgs->v[i].images[j].len;
                uint8_t *bytes = (uint8_t *)server_xmalloc(len);
                memcpy(bytes, msgs->v[i].images[j].bytes, len);
                r->images[k].bytes = bytes;
                r->images[k].len = len;
                r->images[k].start_pos = -1;
            }
        }
    }
    pulsar_tokens expanded = {0};
    /* The per-request image cost that KV reuse does NOT remove: the expander
     * decodes and preprocesses every image to learn its span geometry, and that
     * happens again on every turn even when the KV rows are reused.  Timed here
     * because it is the thing a prepared-span cache would remove (L226). */
    const double vp_t0 = server_now_sec();
    if (!pulsar_expand_image_placeholders(e, &r->prompt, 0, r->images, r->n_images,
                                          &expanded, err, errlen)) {
        return false;
    }
    server_log(PULSAR_LOG_PREFILL, "pulsar-server: image prepare: %d image(s) decoded+preprocessed in %.1f ms",
               r->n_images, (server_now_sec() - vp_t0) * 1000.0);
    pulsar_tokens_free(&r->prompt);
    r->prompt = expanded;
    return true;
}



/* What every family's render does between its own folding of the messages and its template: the tool
 * memory (the sampled call bytes onto the replayed calls, from RAM or a KV file's trailer; the family's
 * find_call_block and raw replay carry them) and the protocols' live continuations (their tails are the
 * family's tool_result_tail).  Parse-without-server (the renderer gate, the golden) has no memory. */
void render_prelude(server *s, chat_conversation *c, request *r) {
    if (s) {
        s->kv_cache_restore_tool_memory_for_messages(&c->msgs);
        s->tool_memory_attach_to_messages(&c->msgs, &r->tool_replay);
    }
    if (r->api == API_ANTHROPIC) anthropic_prepare_live_continuation(r, &c->msgs);
    if (r->api == API_RESPONSES) responses_prepare_live_continuation(r, &c->msgs);
}



/* ---- the thinking controls: ONE reading for every family (L284 P3) ------------------------------- */

/* Every effort NAME any protocol sends -- OpenAI reasoning_effort, Responses reasoning.effort, Anthropic
 * output_config.effort, chat_template_kwargs.reasoning_effort.  What each means is the loaded family's
 * effort map (server_effort_map, one row per family, declared with its other hooks below); "none" is
 * thinking off on every family. */
extern const char *const k_effort_names[CHAT_EFFORT_NAMES] = {"none", "minimal", "low", "medium", "high", "xhigh",
                                                                "max"};
static_assert(sizeof k_effort_names / sizeof k_effort_names[0] == CHAT_EFFORT_NAMES,
              "server_effort_map::named has one column per effort name");
/* the rows the integer axis and the thinking rule name */
enum { k_effort_row_none = 0, k_effort_row_low = 2, k_effort_row_high = 4, k_effort_row_max = 6 };

static int effort_row(const char *name) {
    for (int i = 0; i < CHAT_EFFORT_NAMES; i++)
        if (!strcmp(name, k_effort_names[i])) return i;
    return -1;
}

/* An integer effort on a template without integer levels: the row of the nearest reference preset
 * (low 50, high 75, max 100). */
static int effort_row_nearest(int value) {
    return value < 63 ? k_effort_row_low : value < 88 ? k_effort_row_high : k_effort_row_max;
}

/* One effort value: null (not sent), a name of k_effort_names, or an integer 1..100. */
static bool effort_value(const char **p, const char *key, chat_effort_ask *a, char *err, size_t errlen) {
    json_ws(p);
    if (json_lit(p, "null")) {
        a->level = -1;
        a->value = 0;
        return true;
    }
    if (**p == '"') {
        char *name = NULL;
        if (!json_string(p, &name)) return false;
        const int row = effort_row(name);
        if (row < 0)
            snprintf(err, errlen, "%s: \"%.40s\" is not an effort -- send none, minimal, low, medium, high, xhigh, "
                                  "max or an integer 1..100", key, name);
        free(name);
        if (row < 0) return false;
        a->level = row;
        a->value = 0;
        return true;
    }
    double v = 0.0;
    if (!json_number(p, &v) || v != (double)(int)v || (int)v < PULSAR_THINK_EFFORT_MIN ||
        (int)v > PULSAR_THINK_EFFORT_MAX) {
        snprintf(err, errlen, "%s: an effort is a name or an integer 1..100", key);
        return false;
    }
    a->level = -1;
    a->value = (int)v;
    return true;
}

static bool thinking_bool(const char **p, chat_effort_ask *a) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    bool on = true;
    if (!json_bool(p, &on)) return false;
    a->thinking = on ? 1 : 0;
    return true;
}

bool chat_effort_ask_read(const chat_control *controls, int n, chat_effort_ask *a, char *err, size_t errlen) {
    a->thinking = -1;
    a->level = -1;
    a->value = 0;
    for (int i = 0; i < n; i++) {
        const char *k = controls[i].key;
        const char *q = controls[i].raw;
        bool ok = true;
        json_ws(&q);
        if (!strcmp(k, "thinking")) {
            /* Anthropic's {"type": "enabled" | "disabled"} or a bool; null is not sent */
            if (!json_lit(&q, "null")) {
                bool on = a->thinking != 0;
                ok = parse_thinking_control_value(&q, &on);
                a->thinking = on ? 1 : 0;
            }
        } else if (!strcmp(k, "think") || !strcmp(k, "enable_thinking")) {
            ok = thinking_bool(&q, a);
        } else if (!strcmp(k, "reasoning_effort") || !strcmp(k, "reasoning.effort")) {
            ok = effort_value(&q, k, a, err, errlen);
        } else if (!strcmp(k, "output_config")) {
            char *raw = *q == '{' ? json_object_member_raw(q, "effort") : NULL;
            const char *ep = raw;
            ok = !raw || effort_value(&ep, "output_config.effort", a, err, errlen);
            free(raw);
        } else if (!strcmp(k, "chat_template_kwargs")) {
            /* vLLM's spelling of the template variables: the two every family's template takes */
            if (!json_lit(&q, "null")) {
                ok = *q == '{';
                if (ok) q++;
                json_ws(&q);
                while (ok && *q && *q != '}') {
                    char *kk = NULL;
                    ok = json_string(&q, &kk);
                    json_ws(&q);
                    ok = ok && *q == ':';
                    if (ok) q++;
                    if (ok && !strcmp(kk, "enable_thinking")) {
                        ok = thinking_bool(&q, a);
                    } else if (ok && !strcmp(kk, "reasoning_effort")) {
                        ok = effort_value(&q, "chat_template_kwargs.reasoning_effort", a, err, errlen);
                    } else if (ok) {
                        snprintf(err, errlen, "chat_template_kwargs.%.40s is not a template variable this server "
                                              "renders (enable_thinking, reasoning_effort)", kk);
                        ok = false;
                    }
                    free(kk);
                    json_ws(&q);
                    if (*q == ',') q++;
                    json_ws(&q);
                }
                ok = ok && *q == '}';
            }
        }
        if (!ok) return false;
    }
    return true;
}

bool chat_effort_ask_thinking(const chat_effort_ask *a, const char *model, bool *on, char *err, size_t errlen) {
    const bool none = a->level == k_effort_row_none;
    if (a->thinking == 1 && none) {
        snprintf(err, errlen, "reasoning_effort: none is thinking off, and thinking was requested on");
        return false;
    }
    if (a->thinking >= 0) *on = a->thinking == 1;
    else if (none) *on = false;
    /* the DeepSeek API's non-thinking model name, on any family: a default, never over a control */
    else *on = !(model && !strcmp(model, "deepseek-chat"));
    return true;
}

/* Thinking by the one rule, then the level the loaded family's effort map gives: a name is its column, an
 * integer the map's integer reading (or the nearest preset's column), nothing sent the template's default;
 * thinking off is the "none" column. */
static bool chat_resolve_thinking(pulsar_engine *e, const chat_conversation *c, request *r, char *err,
                                  size_t errlen) {
    chat_effort_ask a;
    bool on = false;
    if (!chat_effort_ask_read(c->controls, c->n_controls, &a, err, errlen) ||
        !chat_effort_ask_thinking(&a, r->model, &on, err, errlen))
        return false;
    const server_effort_map *m = r->family->effort;
    r->think_mode = PULSAR_THINK_NONE;
    r->family_effort = m->named[k_effort_row_none];
    if (!on) return true;
    const int level = a.level >= 0 ? m->named[a.level]
                      : a.value   ? (m->integer ? m->integer(a.value) : m->named[effort_row_nearest(a.value)])
                                  : m->default_level(e);
    r->family_effort = level;
    r->think_mode = m->think_mode_level ? (pulsar_think_mode)level : PULSAR_THINK_DEFAULT;
    return true;
}

/* ---- DeepSeek ------------------------------------------------------------------------------------- */

/* DeepSeek's efforts are the reference encoder's presets on its 1..100 axis (low 50, high 75, max 100) --
 * think modes; an integer is itself on V4.1 (every 1..100 renders), the nearest preset on V4 (0731: low,
 * high, max); nothing sent is the loaded encoder's default (V4.1 high; 0731 low, L239). */
static int deepseek_effort_integer_v41(int value) { return value; }
static int deepseek_effort_default(pulsar_engine *e) { return (int)pulsar_engine_think_default(e); }
#define DEEPSEEK_EFFORT_NAMED                                                                       \
    {PULSAR_THINK_NONE, PULSAR_THINK_LOW, PULSAR_THINK_LOW, PULSAR_THINK_LOW, PULSAR_THINK_HIGH,    \
     PULSAR_THINK_MAX, PULSAR_THINK_MAX}
static const server_effort_map k_effort_deepseek_v41 = {
    DEEPSEEK_EFFORT_NAMED, deepseek_effort_integer_v41, deepseek_effort_default, /*think_mode_level=*/true};
static const server_effort_map k_effort_deepseek_v4 = {
    DEEPSEEK_EFFORT_NAMED, NULL, deepseek_effort_default, /*think_mode_level=*/true};
#undef DEEPSEEK_EFFORT_NAMED

static bool deepseek_render(pulsar_engine *e, server *s, chat_conversation *c, request *r, char *err,
                            size_t errlen) {
    /* Responses can also carry schemas in its input items (tool_search output). */
    buf schemas = {0};
    conversation_tool_schema_lines(c, &schemas);
    r->has_tools = c->tool_choice != CHAT_TOOL_CHOICE_NONE && schemas.len;
    if (r->api == API_ANTHROPIC) anthropic_fold_tool_results(&c->msgs);
    render_prelude(s, c, r);
    /* L223: keep the client-data ranges; the tokeniser turns a spelling inside
     * client text into ordinary tokens instead of a control token. */
    free(r->prompt_spans);
    r->prompt_spans = NULL;
    r->prompt_n_spans = 0;
    r->prompt_text = render_chat_prompt_text_spans(&c->msgs, r->has_tools ? schemas.ptr : NULL,
                                                   &r->tool_orders, r->think_mode, r->family->v41,
                                                   &r->prompt_spans, &r->prompt_n_spans);
    buf_free(&schemas);
    /* A required call (OpenAI "required" or a named function, Anthropic "any" or
     * "tool"): prefill the assistant turn into an open DSML tool_calls block --
     * the render ends "<｜Assistant｜><think>" (or "</think>"); this rewrites it to
     * skip thinking and open the block (a named invoke for a named tool), and
     * generate_job seeds the output with the same opener so the parser sees a
     * full block. */
    const bool forced = c->tool_choice == CHAT_TOOL_CHOICE_ANY || c->tool_choice == CHAT_TOOL_CHOICE_NAMED;
    if (forced && r->has_tools && r->prompt_text) {
        r->force_tool_call = true;
        if (!request_apply_forced_tool_prefill(r, err, errlen)) return false;
    }
    /* With an engine: tokenise the rendered TEXT, then replace every image
     * placeholder with that image's sentinel block -- the reference's order, and
     * the decoded pixels live in the messages.  Without one the TEXT is the whole
     * contract (the renderer gate compares it with the reference encoder). */
    if (e) {
        pulsar_tokenize_rendered_chat_spans(e, r->prompt_text, r->prompt_spans, r->prompt_n_spans, &r->prompt);
        if (!request_prepare_images(e, &c->msgs, r, err, errlen)) return false;
    }
    return true;
}



/* ---- Qwen ----------------------------------------------------------------------------------------- */

/* Qwen's efforts are its template's reasoning_effort names (low, medium, xhigh; "none" is
 * enable_thinking=false); an integer is the nearest preset's; nothing sent is the template's default (xhigh).
 * Downstream reads only whether a think block is open, so think_mode carries PULSAR_THINK_DEFAULT (the
 * enabled marker, not a DeepSeek effort); the render reads family_effort. */
static int qwen_effort_default_level(pulsar_engine *) { return (int)qwen_effort_default(); }
static const server_effort_map k_effort_qwen = {
    {QWEN_EFFORT_NONE, QWEN_EFFORT_LOW, QWEN_EFFORT_LOW, QWEN_EFFORT_MEDIUM, QWEN_EFFORT_XHIGH, QWEN_EFFORT_XHIGH,
     QWEN_EFFORT_XHIGH},
    NULL, qwen_effort_default_level, /*think_mode_level=*/false};


/* L268: a message's content without the parser's image markers (PULSAR_IMAGE_PLACEHOLDER at image_ph_off -- the
 * offsets, not the spelling, are the authority) and where each image sits in what is left. */
struct qwen_msg_images {
    std::string content;
    std::vector<uint32_t> at;
};

static bool qwen_strip_image_markers(const chat_msg *m, qwen_msg_images *out, char *err, size_t errlen) {
    const char *c = m->content ? m->content : "";
    const size_t n = strlen(c), ph = strlen(PULSAR_IMAGE_PLACEHOLDER);
    size_t at = 0;
    for (int k = 0; k < m->images_len; k++) {
        const size_t off = m->image_ph_off ? m->image_ph_off[k] : SIZE_MAX;
        if (off < at || off > n || n - off < ph || memcmp(c + off, PULSAR_IMAGE_PLACEHOLDER, ph)) {
            snprintf(err, errlen, "image %d of a message has no placeholder at its offset", k);
            return false;
        }
        out->content.append(c + at, off - at);
        out->at.push_back((uint32_t)out->content.size());
        at = off + ph;
    }
    out->content.append(c + at, n - at);
    return true;
}

/* chat_msgs[start..) as the renderer's messages; the pointers borrow `msgs` and `notes`.  The system
 * FIELD renders first (the parser appends it to the array); a system message that is not the
 * conversation's first turns into a user <system-reminder> turn (the template has no in-place system
 * turn; `notes` owns that text); a replayed call carries its sampled bytes when tool memory found them.
 * `tail`: a continuation tail -- no entry is the conversation's first, and the system field is not
 * part of one.  false + err: an image whose parser marker is not at its offset. */
static bool qwen_messages(const chat_msgs *msgs, int start, bool tail, std::vector<qwen_msg_in> *qm,
                          std::vector<std::vector<qwen_tool_call_in>> *qc, std::vector<std::string> *notes,
                          std::vector<qwen_msg_images> *imgs, char *err, size_t errlen) {
    qc->assign((size_t)msgs->len, {});
    notes->assign((size_t)msgs->len, std::string());
    imgs->assign((size_t)msgs->len, qwen_msg_images());
    for (int pass = 0; pass < 2; pass++) {
        for (int i = start; i < msgs->len; i++) {
            const chat_msg *m = &msgs->v[i];
            if (m->system_field != (pass == 0)) continue;
            if (tail && m->system_field) continue;
            /* L268: images render as the template's vision literal here and in a live tail alike; the server places a
             * tail's new images on the live history (image_continuation_place, every family) */
            if (m->images_len > 0 && !qwen_strip_image_markers(m, &(*imgs)[(size_t)i], err, errlen)) return false;
            for (int k = 0; k < m->calls.len; k++)
                (*qc)[(size_t)i].push_back({m->calls.v[k].name, m->calls.v[k].arguments});
            if (!strcmp(m->role, "system") && (tail || !qm->empty())) {
                (*notes)[(size_t)i] = qwen_system_reminder(m->content);
                qm->push_back({"user", (*notes)[(size_t)i].c_str(), NULL, NULL, 0, NULL});
                continue;
            }
            const qwen_msg_images &mi = (*imgs)[(size_t)i];
            qm->push_back({m->role, m->images_len > 0 ? mi.content.c_str() : m->content, m->reasoning,
                           (*qc)[(size_t)i].empty() ? NULL : (*qc)[(size_t)i].data(), m->calls.len,
                           m->calls.raw_calls, mi.at.empty() ? NULL : mi.at.data(), (int)mi.at.size()});
        }
    }
    return true;
}

/* A render's text and client-data ranges as the request owns them (malloc'd; the ranges NULL when
 * none). */
static char *qwen_take_render(qwen_render_out *out, chat_text_span **spans_out, uint32_t *n_spans_out) {
    if (spans_out) {
        *spans_out = NULL;
        *n_spans_out = (uint32_t)out->spans.size();
        if (!out->spans.empty()) {
            *spans_out = (chat_text_span *)server_xmalloc(out->spans.size() * sizeof(chat_text_span));
            memcpy(*spans_out, out->spans.data(), out->spans.size() * sizeof(chat_text_span));
        }
    }
    return xstrndup(out->text.data(), out->text.size());
}

/* The tools of a family whose template takes OpenAI's tools array (Qwen, Inkling): none with tool_choice none; an
 * OpenAI request's array as sent (HF's / vLLM's input); else the conversation's schema lines -- every protocol's
 * kinds and tool_search's loads, as for DeepSeek (L284) -- in OpenAI's shape.  r->tools_json stays NULL when there
 * are none; false + err when the lines do not convert. */
bool chat_openai_tools_json(const chat_conversation *c, request *r, char **head_json, char *err, size_t errlen) {
    *head_json = NULL;
    if (c->tool_choice == CHAT_TOOL_CHOICE_NONE) return true;
    const bool deferred = c->deferred_tool_schemas && c->deferred_tool_schemas[0];
    /* HF's input as sent -- unless some tool is deferred (L325), which only the schema lines separate */
    if (r->api == API_OPENAI && c->tools_raw && !deferred) {
        r->tools_json = xstrdup(c->tools_raw);
        *head_json = xstrdup(c->tools_raw);
        return true;
    }
    buf head = {0}, all = {0};
    conversation_tool_schema_lines(c, &head);
    conversation_declared_tool_schema_lines(c, &all);
    if (all.len) r->tools_json = tool_schema_lines_openai_tools(all.ptr, all.len, err, errlen);
    if (head.len && r->tools_json) *head_json = tool_schema_lines_openai_tools(head.ptr, head.len, err, errlen);
    const bool failed = (all.len && !r->tools_json) || (head.len && !*head_json);
    buf_free(&head);
    buf_free(&all);
    return !failed;
}

/* ---- L325: a tool loaded mid-conversation renders where it was loaded ------------------------------------------- */

/* DeepSeek's tools block lists the schema lines themselves (append_tools_prompt_text, "### Available Tool Schemas"):
 * a loaded tool is its line. */
static char *deepseek_tool_defs_text(const char *lines, size_t len, char *, size_t) { return xstrndup(lines, len); }

/* Qwen's: the tools block's entries inside <tools></tools> (qwen_chat_tool_defs). */
static char *qwen_tool_defs_text(const char *lines, size_t len, char *err, size_t errlen) {
    char *json = tool_schema_lines_openai_tools(lines, len, err, errlen);
    if (!json) return NULL;
    qwen_render_out out;
    const bool ok = qwen_chat_tool_defs(json, &out, err, errlen);
    free(json);
    return ok ? xstrndup(out.text.data(), out.text.size()) : NULL;
}

/* The deferred tool `name`'s schema line (malloc'd), or NULL when the request defers no tool of that name. */
static char *deferred_tool_line(const chat_conversation *c, const char *name) {
    for (const char *at = c->deferred_tool_schemas ? c->deferred_tool_schemas : ""; *at;) {
        const char *nl = strchr(at, '\n');
        const size_t len = nl ? (size_t)(nl - at) : strlen(at);
        const std::string line(at, len);
        at += len + (nl ? 1 : 0);
        char *raw = json_object_member_raw(line.c_str(), "name");
        const char *rp = raw;
        char *got = NULL;
        const bool hit = rp && json_string(&rp, &got) && !strcmp(got, name);
        free(got);
        free(raw);
        if (hit) return xstrdup(line.c_str());
    }
    return NULL;
}

/* The pending definitions `lines` written to `out` in the family's encoding (nothing when there are none). */
static bool flush_tool_defs(buf *out, buf *lines, const request *r, char *err, size_t errlen) {
    if (!lines->len) return true;
    char *text = r->family->tool_defs_text(lines->ptr, lines->len, err, errlen);
    buf_free(lines);
    if (!text) return false;
    buf_puts(out, text);
    free(text);
    return true;
}

bool chat_expand_tool_refs(chat_conversation *c, const request *r, char *err, size_t errlen) {
    for (int i = 0; i < c->msgs.len; i++) {
        chat_msg *m = &c->msgs.v[i];
        if (m->tool_refs_len == 0) continue;
        const char *src = m->content ? m->content : "";
        const size_t n = strlen(src);
        buf out = {0};
        size_t at = 0;
        std::vector<size_t> moved(m->image_ph_off, m->image_ph_off + m->images_len);
        for (int k = 0; k < m->tool_refs_len;) {
            const size_t off = std::min(m->tool_ref_off[k], n);
            buf_append(&out, src + at, off - at);
            at = off;
            const size_t before = out.len;
            /* the run of references at this offset, in order; consecutive deferred ones are ONE block of definitions */
            buf lines = {0};
            for (; k < m->tool_refs_len && std::min(m->tool_ref_off[k], n) == off; k++) {
                char *line = deferred_tool_line(c, m->tool_ref_name[k]);
                if (line) {
                    if (lines.len) buf_putc(&lines, '\n');
                    buf_puts(&lines, line);
                    free(line);
                    continue;
                }
                if (!flush_tool_defs(&out, &lines, r, err, errlen)) {
                    buf_free(&out);
                    return false;
                }
                /* not deferred: the tools block already renders it (or nothing declares it) -- only the fact */
                buf_printf(&out, "[tool available: %s]", m->tool_ref_name[k]);
            }
            if (!flush_tool_defs(&out, &lines, r, err, errlen)) {
                buf_free(&out);
                return false;
            }
            /* every image placeholder at or after the run moves by what it wrote */
            for (int q = 0; q < m->images_len; q++)
                if (m->image_ph_off[q] >= off) moved[(size_t)q] += out.len - before;
        }
        buf_append(&out, src + at, n - at);
        for (int q = 0; q < m->images_len; q++) m->image_ph_off[q] = moved[(size_t)q];
        free(m->content);
        m->content = buf_take(&out);
        if (!m->content) m->content = xstrdup("");
        chat_msg_clear_tool_refs(m);
    }
    return true;
}

/* HF's apply_chat_template byte for byte (qwen_chat_render, with the L223 client-span map), tokenised
 * with the family's markers; the tools in the template's shape, typed for the output parser.  A forced
 * tool_choice prefills the turn into an open <tool_call> (qwen_forced_call_prefill); a tool-result-only
 * request continues the live KV with the family's tail (render_prelude).  Images are the template's vision
 * literal in place (L268), expanded by the core's walk.  The tools are the conversation's schema lines -- every
 * protocol's kinds and tool_search's loads, as for DeepSeek (L284) -- in OpenAI's shape; an OpenAI request's
 * array goes to the template as sent (HF's input). */
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
    std::vector<qwen_msg_images> imgs;
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
        pulsar_tokenize_rendered_chat_spans(e, r->prompt_text, r->prompt_spans, r->prompt_n_spans, &r->prompt);
        if (!request_prepare_images(e, &c->msgs, r, err, errlen)) return false;
    }
    return true;
}

/* ---- step 4: Qwen's suffix hooks -- the template's bytes for one sampled turn, one tail, one tool
 *      error, one forced call, and where a run of <tool_call> blocks sits in a transcript.  Each is
 *      qwen_chat_render's own writer (src/lib/qwen_chat.cpp), so prefix + hook = the full render by
 *      construction (test_qwen_hooks_compose_to_the_full_render). ---------------------------------- */

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

static char *qwen_tool_result_tail(const request *r, const chat_msgs *msgs, int start, chat_text_span **spans_out,
                                   uint32_t *n_spans_out) {
    if (spans_out) { *spans_out = NULL; *n_spans_out = 0; }
    std::vector<qwen_msg_in> qm;
    std::vector<std::vector<qwen_tool_call_in>> qc;
    std::vector<std::string> notes;
    std::vector<qwen_msg_images> imgs;
    char err[200];
    qwen_render_out out;
    if (!qwen_messages(msgs, start, true, &qm, &qc, &notes, &imgs, err, sizeof err) ||
        !qwen_chat_render_tail(qm.data(), (int)qm.size(), (qwen_effort)r->family_effort, &out, err, sizeof err))
        return NULL;
    return qwen_take_render(&out, spans_out, n_spans_out);
}

/* The system turn's body in a rendered Qwen prompt (the tools block and the client's system text), or
 * NULL when the prompt has none: the reminder a tool error carries. */
static char *qwen_rendered_system_region(const char *prompt) {
    static const char open[] = "<|im_start|>system\n";
    if (!prompt || strncmp(prompt, open, sizeof open - 1)) return NULL;
    const char *p = prompt + (sizeof open - 1);
    const char *end = strstr(p, "<|im_end|>");
    if (!end) return NULL;
    while (end > p && isspace((unsigned char)end[-1])) end--;
    return xstrndup(p, (size_t)(end - p));
}

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
    result.role = xstrdup("tool");
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

static void qwen_forced_call_prefill(const request *r, const char *, size_t *, buf *append) {
    qwen_forced_call_seed(r, append);   /* the prompt keeps its generation prompt whole */
}

/* The run of consecutive <tool_call> blocks (whitespace between them) at or after `p`: one turn's calls,
 * the bytes the parser recorded as the turn's raw_calls. */
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

/* L272 B6: the longest rendered prefix two UNRELATED prompts share by template construction -- the
 * slot router's trivial-match header (server::slot_trivial_common_tokens).  DeepSeek: the BOS plus
 * the longest effort preamble the loaded encoder renders (V4.1 renders the numeric line, 0731 its
 * high / max texts; measured before this lived here, the header was built from DeepSeek's BOS on a
 * Qwen engine too).  Qwen: the system turn's opener -- no BOS, no effort line. */
static int qwen_trivial_header_tokens(pulsar_engine *e) {
    pulsar_tokens t = {0};
    pulsar_tokenize_rendered_chat(e, "<|im_start|>system\n", &t);
    const int hdr_len = t.len;
    pulsar_tokens_free(&t);
    return hdr_len;
}

static int deepseek_trivial_header_tokens(pulsar_engine *e) {
    int hdr_len = 0;
    const pulsar_think_mode prefixed_modes[] = {PULSAR_THINK_HIGH, PULSAR_THINK_MAX};
    for (size_t i = 0; i < sizeof(prefixed_modes) / sizeof(prefixed_modes[0]); i++) {
        buf hdr = {0};
        buf_puts(&hdr, PULSAR_SERVER_RENDER_BOS);
        buf_puts(&hdr, pulsar_think_effort_prefix_family(prefixed_modes[i], pulsar_engine_chat_v41(e)));
        pulsar_tokens hdr_tokens = {0};
        pulsar_tokenize_rendered_chat(e, hdr.ptr, &hdr_tokens);
        if (hdr_tokens.len > hdr_len) hdr_len = hdr_tokens.len;
        pulsar_tokens_free(&hdr_tokens);
        buf_free(&hdr);
    }
    return hdr_len;
}

/* ---- step 4: DeepSeek's suffix hooks -- its template's bytes for one turn, one tail, one tool error,
 *      one forced call, and where a DSML block sits in a transcript ------------------------------------ */

static char *deepseek_assistant_turn_sampled(const request *r, bool think, const char *reasoning, const char *content,
                                             const tool_calls *calls, chat_text_span **spans_out,
                                             uint32_t *n_spans_out) {
    buf suffix = {0};
    append_assistant_turn_sampled(&suffix, think, reasoning, content, calls, r->family->v41);
    if (spans_out) {
        *spans_out = suffix.spans;
        *n_spans_out = suffix.n_spans;
        suffix.spans = NULL;
        suffix.n_spans = suffix.cap_spans = 0;
    }
    return buf_take(&suffix);
}

static char *deepseek_tool_result_tail(const request *r, const chat_msgs *msgs, int start, chat_text_span **spans_out,
                                       uint32_t *n_spans_out) {
    /* the loaded template's rules (a 0731 model merges tool results its own way); this used to be
     * hard-coded to V4.1's at every live-tail site */
    return render_live_tool_tail_spans(msgs, start, r->has_tools, r->think_mode, r->family->v41, spans_out,
                                       n_spans_out);
}

static char *deepseek_tool_error_suffix(const request *r, const thinking_state *thinking, const char *detail,
                                        chat_text_span **spans_out, uint32_t *n_spans_out) {
    return build_invalid_dsml_tool_error_suffix_spans(r, thinking, detail, spans_out, n_spans_out);
}

/* The exact bytes a forced tool call is seeded with: close thinking, open the tool_calls block and the
 * invoke tag (named when a specific tool was requested, else up to its name).  The prompt rewrite drops the render's
 * trailing "<think>" opener and skips the seed's close when the render already ended with one. */
static void deepseek_forced_call_seed(const request *r, buf *out) {
    const pulsar_dsml_syntax *d = pulsar_dsml_canonical(r->family->v41);
    buf_puts(out, "</think>\n\n");
    buf_puts(out, d->tool_calls_start);
    buf_puts(out, "\n");
    buf_puts(out, d->invoke_start);
    /* unnamed: stop before the name's opener -- the model samples ` name="NAME">` under the call grammar
     * (L303, the same constraint as Qwen's); named: the whole tag */
    if (r->forced_tool_name && r->forced_tool_name[0]) {
        buf_puts(out, " name=\"");
        buf_puts(out, r->forced_tool_name);
        buf_puts(out, "\">\n");
    }
}

static void deepseek_forced_call_prefill(const request *r, const char *prompt, size_t *keep, buf *append) {
    size_t blen = *keep;
    if (blen >= 7 && !memcmp(prompt + blen - 7, "<think>", 7)) blen -= 7;
    *keep = blen;
    buf seed = {0};
    deepseek_forced_call_seed(r, &seed);
    const bool closed = blen >= 8 && !memcmp(prompt + blen - 8, "</think>", 8);
    buf_puts(append, seed.ptr + (closed ? 8 : 0));   /* already closed: skip the seed's close */
    buf_free(&seed);
}

/* ---- L303: the grammar shapes -- what a constrained turn's bytes look like in each template ----------- */

/* DeepSeek: after reasoning the answer; the calls are ONE DSML block of invokes (append_dsml_tool_calls_text's
 * bytes: the block tag and a newline, each invoke's tag line, its parameters, its close and a newline, the block
 * close), the parameters in xgrammar's DSML style for the template's spelling. */
static const server_grammar_shape k_grammar_deepseek_v41 = {
    /* .answer_open           = */ "</think>",
    /* .answer_at_think_close = */ true,
    /* .answer_close          = */ "",
    /* .call_trigger          = */ PULSAR_TOOL_CALLS_START,
    /* .block_head            = */ "\n",
    /* .name_open             = */ PULSAR_INVOKE_START " name=\"",
    /* .name_close            = */ "\">\n",
    /* .style                 = */ "deepseek_v4_1_xml",
    /* .call_close            = */ PULSAR_INVOKE_END "\n",
    /* .repeat                = */ true,
    /* .block_tail            = */ PULSAR_TOOL_CALLS_END,
};
static const server_grammar_shape k_grammar_deepseek_v4 = {
    /* .answer_open           = */ "</think>",
    /* .answer_at_think_close = */ true,
    /* .answer_close          = */ "",
    /* .call_trigger          = */ PULSAR_TOOL_CALLS_START_V4,
    /* .block_head            = */ "\n",
    /* .name_open             = */ PULSAR_INVOKE_START_V4 " name=\"",
    /* .name_close            = */ "\">\n",
    /* .style                 = */ "deepseek_xml",
    /* .call_close            = */ PULSAR_INVOKE_END_V4 "\n",
    /* .repeat                = */ true,
    /* .block_tail            = */ PULSAR_TOOL_CALLS_END_V4,
};
/* Qwen and Nemotron-H write the same markup (qwen_chat.cpp, nemotron_chat.cpp): one <tool_call> block a call,
 * <function=NAME> and its <parameter=KEY> lines -- xgrammar's qwen_xml style. */
const server_grammar_shape k_grammar_qwen_xml = {
    /* .answer_open           = */ "</think>",
    /* .answer_at_think_close = */ true,
    /* .answer_close          = */ "",
    /* .call_trigger          = */ "<tool_call>",
    /* .block_head            = */ "",
    /* .name_open             = */ "\n<function=",
    /* .name_close            = */ ">\n",
    /* .style                 = */ "qwen_xml",
    /* .call_close            = */ "</function>\n</tool_call>",
    /* .repeat                = */ false,
    /* .block_tail            = */ "",
};

/* ---- the tables and the dispatch ------------------------------------------------------------------- */

static const server_family_ops k_family_deepseek_v41 = {
    /* .name                  = */ "DeepSeek V4.1",
    /* .format                = */ PULSAR_CHAT_DS4_V41,
    /* .v41                   = */ true,
    /* .effort                = */ &k_effort_deepseek_v41,
    /* .render                = */ deepseek_render,
    /* .trivial_header_tokens = */ deepseek_trivial_header_tokens,
    /* .output                = */ &k_parser_deepseek,
    /* .assistant_turn_sampled = */ deepseek_assistant_turn_sampled,
    /* .tool_result_tail      = */ deepseek_tool_result_tail,
    /* .tool_error_suffix     = */ deepseek_tool_error_suffix,
    /* .forced_call_seed      = */ deepseek_forced_call_seed,
    /* .forced_call_prefill   = */ deepseek_forced_call_prefill,
    /* .grammar               = */ &k_grammar_deepseek_v41,
    /* .find_call_block       = */ find_next_dsml_tool_block,
    /* .tool_defs_text        = */ deepseek_tool_defs_text,
};
static const server_family_ops k_family_deepseek_v4 = {
    /* .name                  = */ "DeepSeek V4 (0731)",
    /* .format                = */ PULSAR_CHAT_DS4_V4,
    /* .v41                   = */ false,
    /* .effort                = */ &k_effort_deepseek_v4,
    /* .render                = */ deepseek_render,
    /* .trivial_header_tokens = */ deepseek_trivial_header_tokens,
    /* .output                = */ &k_parser_deepseek,
    /* .assistant_turn_sampled = */ deepseek_assistant_turn_sampled,
    /* .tool_result_tail      = */ deepseek_tool_result_tail,
    /* .tool_error_suffix     = */ deepseek_tool_error_suffix,
    /* .forced_call_seed      = */ deepseek_forced_call_seed,
    /* .forced_call_prefill   = */ deepseek_forced_call_prefill,
    /* .grammar               = */ &k_grammar_deepseek_v4,
    /* .find_call_block       = */ find_next_dsml_tool_block,
    /* .tool_defs_text        = */ deepseek_tool_defs_text,
};
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
};

const server_family_ops *server_family_for_format(pulsar_chat_format fmt) {
    switch (fmt) {
    case PULSAR_CHAT_DS4_V4:  return &k_family_deepseek_v4;
    case PULSAR_CHAT_DS4_V41: return &k_family_deepseek_v41;
    case PULSAR_CHAT_QWEN:    return &k_family_qwen;
    case PULSAR_CHAT_INKLING: return &k_family_inkling;
    case PULSAR_CHAT_NEMOTRON_H: return &k_family_nemotron_h;
    }
    pulsar_die("server_family_for_format: a chat format without a server family table");
    return NULL;   /* unreachable: pulsar_die exits */
}

const server_family_ops *server_family_for_engine(const pulsar_engine *e) {
    return server_family_for_format(pulsar_engine_chat_format(e));
}

/* The loaded family's resolve and render around the protocol's tool-result checks (which read the
 * resolved mode).  The family follows the LOADED model (L218 s123): the renderer, the forced-prefill and
 * the KV-key suffix builders all read it from the request, so a 0731 artifact cannot be primed with
 * V4.1's template by one of them and V4's by another. */
bool render_chat_conversation(pulsar_engine *e, pulsar_chat_format fmt, server *s, chat_conversation *c,
                              request *r, char *err, size_t errlen) {
    r->family = server_family_for_format(fmt);
    if (!chat_resolve_thinking(e, c, r, err, errlen)) return false;
    /* The protocol's tool-result checks: every result's call is in this request's
     * history or bound to the live frontier (parse-without-server skips them). */
    if (s && r->api == API_ANTHROPIC &&
        !s->anthropic_validate_tool_results(&c->msgs, &r->anthropic_requires_live_tool_state, err, errlen))
        return false;
    if (s && r->api == API_RESPONSES &&
        !s->responses_validate_tool_outputs(&c->msgs, r->think_mode, &r->responses_requires_live_tool_state,
                                            &r->responses_requires_live_reasoning, err, errlen))
        return false;
    /* L325: tool_reference blocks become their tools' definitions (or notices) before any render reads the messages */
    if (!chat_expand_tool_refs(c, r, err, errlen)) return false;
    return r->family->render(e, s, c, r, err, errlen) && constrain_plan_build(e, s, c, r, err, errlen);
}

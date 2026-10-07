/* L267: the model-family half of a chat request.
 *
 * A protocol parser (api_parse.cpp: OpenAI chat, Anthropic Messages, Responses) reads its wire format
 * into a chat_conversation -- the messages, the tools as sent, the tool_choice, and the thinking / effort
 * controls exactly as sent.  render_chat_conversation() is everything after that, and it is the loaded
 * family's: how the controls resolve to a thinking mode, the protocol's tool-result checks (they read
 * the resolved mode), then the prompt.  DeepSeek renders its own template (prompt_render.cpp) with its
 * tool memory, live-continuation suffixes and forced-call prefill; Qwen renders HF's template
 * (src/lib/qwen_chat) with the same memory, continuations and prefill behind its own hooks (L272 P3
 * step 4), and refuses by name what that template cannot express.  A new family adds one arm here; a
 * new protocol adds one parser and no arm. */
#include "pulsar_server_internal.h"
#include "../lib/qwen_chat.h"

#include <string>
#include <vector>

void chat_conversation_free(chat_conversation *c) {
    if (!c) return;
    chat_msgs_free(&c->msgs);
    free(c->tools_raw);
    free(c->tool_schemas);
    buf_free(&c->loaded_tool_schemas);
    free(c->tool_choice_wire);
    for (int i = 0; i < c->n_controls; i++) free(c->controls[i].raw);
    free(c->controls);
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
static bool request_prepare_images(pulsar_engine *e, const chat_msgs *msgs,
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
    if (!pulsar_expand_image_placeholders(e, &r->prompt, r->images, r->n_images,
                                          &expanded, err, errlen)) {
        return false;
    }
    server_log(PULSAR_LOG_PREFILL, "pulsar-server: image prepare: %d image(s) decoded+preprocessed in %.1f ms",
               r->n_images, (server_now_sec() - vp_t0) * 1000.0);
    pulsar_tokens_free(&r->prompt);
    r->prompt = expanded;
    return true;
}



/* ---- DeepSeek ------------------------------------------------------------------------------------- */

/* The controls on DeepSeek's effort scale, in arrival order.  The default effort is the loaded
 * family's: V4.1 defaults to high (the reference's default); the V4 (0731) encoder's default is low,
 * which renders no effort line at all (L239).  DeepSeek's template takes no chat_template_kwargs. */
/* What every family's render does between its own folding of the messages and its template: the tool
 * memory (the sampled call bytes onto the replayed calls, from RAM or a KV file's trailer; the family's
 * find_call_block and raw replay carry them) and the protocols' live continuations (their tails are the
 * family's tool_result_tail).  Parse-without-server (the renderer gate, the golden) has no memory. */
static void render_prelude(server *s, chat_conversation *c, request *r) {
    if (s) {
        s->kv_cache_restore_tool_memory_for_messages(&c->msgs);
        s->tool_memory_attach_to_messages(&c->msgs, &r->tool_replay);
    }
    if (r->api == API_ANTHROPIC) anthropic_prepare_live_continuation(r, &c->msgs);
    if (r->api == API_RESPONSES) responses_prepare_live_continuation(r, &c->msgs);
}

static bool deepseek_resolve(pulsar_engine *e, const chat_conversation *c, request *r, char *err, size_t errlen) {
    pulsar_think_mode effort = pulsar_engine_think_default(e);
    bool enabled = true, got_thinking = false;
    for (int i = 0; i < c->n_controls; i++) {
        const char *k = c->controls[i].key;
        const char *q = c->controls[i].raw;
        bool ok = true;
        if (!strcmp(k, "thinking")) {
            ok = parse_thinking_control_value(&q, &enabled);
            got_thinking = true;
        } else if (!strcmp(k, "think") || !strcmp(k, "enable_thinking")) {
            /* enable_thinking is the Qwen/vLLM spelling of our `think` */
            ok = json_bool(&q, &enabled);
            got_thinking = true;
        } else if (!strcmp(k, "reasoning_effort")) {
            ok = parse_reasoning_effort_value(&q, &effort);
        } else if (!strcmp(k, "output_config")) {
            ok = parse_output_config_effort(&q, &effort);
        } else if (!strcmp(k, "reasoning.effort")) {
            /* Only an explicit effort counts as the client opting into thinking control; Responses'
             * "minimal" / "none" effort is thinking off. */
            ok = parse_reasoning_effort_value(&q, &effort);
            got_thinking = true;
            if (effort == PULSAR_THINK_NONE) enabled = false;
        }
        if (!ok) return false;
    }
    if (!got_thinking && model_alias_disables_thinking(r->model)) enabled = false;
    if (!got_thinking && model_alias_enables_thinking(r->model)) enabled = true;
    /* the loaded encoder's efforts (the one rule the CLI and eval apply too, L272 B10) */
    char why[160];
    if (enabled && !pulsar_engine_think_mode_supported(e, effort, why, sizeof why)) {
        if (err && errlen) snprintf(err, errlen, "reasoning_effort: %s", why);
        return false;
    }
    r->think_mode = think_mode_from_enabled(enabled, effort);
    return true;
}

static bool deepseek_render(pulsar_engine *e, server *s, chat_conversation *c, request *r, char *err,
                            size_t errlen) {
    /* Responses can also carry schemas in its input items (tool_search output). */
    buf schemas = {0};
    if (c->tool_schemas && c->tool_schemas[0]) buf_puts(&schemas, c->tool_schemas);
    if (c->loaded_tool_schemas.len) {
        if (schemas.len) buf_putc(&schemas, '\n');
        buf_append(&schemas, c->loaded_tool_schemas.ptr, c->loaded_tool_schemas.len);
    }
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

/* Anthropic's effort levels on the Qwen template's (Tyler 2026-10-05): each reaches the nearest level
 * the template has.  OpenAI's and Responses' reasoning_effort carry the template's own names and are
 * not mapped (qwen_effort_resolve refuses any other). */
static const char *qwen_effort_from_anthropic(const char *level) {
    if (!strcmp(level, "low")) return "low";
    if (!strcmp(level, "medium")) return "medium";
    if (!strcmp(level, "high") || !strcmp(level, "xhigh") || !strcmp(level, "max")) return "xhigh";
    return NULL;
}

/* The controls on the template's variables, in arrival order: enable_thinking (thinking / think /
 * enable_thinking, or chat_template_kwargs) and reasoning_effort.  qwen_effort_resolve is the one
 * resolution; Downstream reads only whether a think block is open, so the request carries
 * PULSAR_THINK_DEFAULT (the enabled marker, not a DeepSeek effort) or NONE. */
static bool qwen_resolve(pulsar_engine *, const chat_conversation *c, request *r, char *err, size_t errlen) {
    qwen_effort qe_v = QWEN_EFFORT_NONE;
    qwen_effort *qe = &qe_v;
    int thinking = -1;          /* -1 not sent, 0 off, 1 on */
    std::string effort;
    bool has_effort = false;
    for (int i = 0; i < c->n_controls; i++) {
        const char *k = c->controls[i].key;
        const char *q = c->controls[i].raw;
        bool ok = true;
        json_ws(&q);
        if (!strcmp(k, "thinking")) {
            if (!json_lit(&q, "null")) {
                bool on = true;
                ok = parse_thinking_control_value(&q, &on);
                thinking = on ? 1 : 0;
            }
        } else if (!strcmp(k, "think") || !strcmp(k, "enable_thinking")) {
            bool on = true;
            ok = json_bool(&q, &on);
            thinking = on ? 1 : 0;
        } else if (!strcmp(k, "reasoning_effort") || !strcmp(k, "reasoning.effort")) {
            has_effort = false;
            if (!json_lit(&q, "null")) {
                char *name = NULL;
                ok = *q == '"' && json_string(&q, &name);
                if (ok) {
                    effort = name;
                    has_effort = true;
                } else {
                    snprintf(err, errlen, "%s: this model (Qwen3.8-Flash-Next) takes a name -- low, medium, xhigh or none", k);
                }
                free(name);
            }
        } else if (!strcmp(k, "output_config")) {
            char *raw = json_object_member_raw(q, "effort");
            const char *ep = raw;
            if (raw && (json_ws(&ep), !json_lit(&ep, "null"))) {
                char *level = NULL;
                const char *mapped = NULL;
                ok = *ep == '"' && json_string(&ep, &level) && (mapped = qwen_effort_from_anthropic(level)) != NULL;
                if (ok) {
                    effort = mapped;
                    has_effort = true;
                } else {
                    snprintf(err, errlen, "output_config.effort: %s%s%s has no Qwen3.8-Flash-Next level -- send low, "
                                          "medium, high, xhigh or max", level ? "\"" : "", level ? level : "a non-name",
                             level ? "\"" : "");
                }
                free(level);
            }
            free(raw);
        } else if (!strcmp(k, "chat_template_kwargs")) {
            /* vLLM's spelling of the template variables.  Only the two this renderer takes are accepted; any
             * other would be silently unrendered. */
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
                        bool on = true;
                        ok = json_bool(&q, &on);
                        thinking = on ? 1 : 0;
                    } else if (ok && !strcmp(kk, "reasoning_effort")) {
                        json_ws(&q);
                        has_effort = false;
                        if (!json_lit(&q, "null")) {
                            char *name = NULL;
                            ok = *q == '"' && json_string(&q, &name);
                            if (ok) {
                                effort = name;
                                has_effort = true;
                            }
                            free(name);
                        }
                    } else if (ok) {
                        snprintf(err, errlen, "chat_template_kwargs.%.40s is not a template variable this server "
                                              "renders for the Qwen family (enable_thinking, reasoning_effort)", kk);
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
    if (!qwen_effort_resolve(has_effort ? effort.c_str() : NULL, thinking, qe, err, errlen)) return false;
    r->think_mode = *qe == QWEN_EFFORT_NONE ? PULSAR_THINK_NONE : PULSAR_THINK_DEFAULT;
    r->family_effort = (int)*qe;   /* the render reads it */
    return true;
}

/* An Anthropic or Responses tools array in the shape the template renders (OpenAI's:
 * {"type":"function","function":{"name","description","parameters"}}), each value's JSON as the client
 * wrote it.  Anthropic's server tools (web search) are dropped as for every family; a Responses tool
 * that is not a function (namespace, tool_search, ...) is refused by name. */
static char *qwen_tools_openai_shape(const chat_conversation *c, api_style api, char *err, size_t errlen) {
    const char *p = c->tools_raw;
    json_ws(&p);
    if (*p != '[') {
        snprintf(err, errlen, "tools: an array is required");
        return NULL;
    }
    p++;
    buf out = {0};
    buf_putc(&out, '[');
    int n = 0;
    for (int i = 0;; i++) {
        json_ws(&p);
        if (*p == ']') break;
        char *raw = NULL;
        if (!json_raw_value(&p, &raw)) {
            buf_free(&out);
            return NULL;
        }
        json_ws(&p);
        if (*p == ',') p++;
        if (api == API_ANTHROPIC && anthropic_server_tool_entry(raw)) {
            free(raw);
            continue;
        }
        if (api == API_RESPONSES) {
            char *type = json_object_member_raw(raw, "type");
            const bool function = type && !strcmp(type, "\"function\"");
            if (!function) {
                snprintf(err, errlen, "tools.%d: a %s tool is not served for the Qwen family (function tools only)", i,
                         type ? type : "typeless");
                free(type);
                free(raw);
                buf_free(&out);
                return NULL;
            }
            free(type);
        }
        char *name = json_object_member_raw(raw, "name");
        char *desc = json_object_member_raw(raw, "description");
        char *params = json_object_member_raw(raw, api == API_ANTHROPIC ? "input_schema" : "parameters");
        if (!name) {
            snprintf(err, errlen, "tools.%d: a tool needs a name", i);
            free(raw);
            free(desc);
            free(params);
            buf_free(&out);
            return NULL;
        }
        buf_puts(&out, n++ ? ", " : "");
        buf_puts(&out, "{\"type\": \"function\", \"function\": {\"name\": ");
        buf_puts(&out, name);
        if (desc) {
            buf_puts(&out, ", \"description\": ");
            buf_puts(&out, desc);
        }
        buf_puts(&out, ", \"parameters\": ");
        buf_puts(&out, params ? params : "{}");
        buf_puts(&out, "}}");
        free(raw);
        free(name);
        free(desc);
        free(params);
    }
    buf_putc(&out, ']');
    return buf_take(&out);
}

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
                (*notes)[(size_t)i] = std::string("<system-reminder>\n") + (m->content ? m->content : "") +
                                      "\n</system-reminder>";
                qm->push_back({"user", (*notes)[(size_t)i].c_str(), NULL, NULL, 0, NULL});
                continue;
            }
            const qwen_msg_images &mi = (*imgs)[(size_t)i];
            qm->push_back({m->role, m->images_len > 0 ? mi.content.c_str() : m->content, m->reasoning,
                           (*qc)[(size_t)i].empty() ? NULL : (*qc)[(size_t)i].data(), m->calls.len,
                           m->calls.raw_dsml, mi.at.empty() ? NULL : mi.at.data(), (int)mi.at.size()});
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

/* HF's apply_chat_template byte for byte (qwen_chat_render, with the L223 client-span map), tokenised
 * with the family's markers; the tools in the template's shape, typed for the output parser.  A forced
 * tool_choice prefills the turn into an open <tool_call> (qwen_forced_call_prefill); a tool-result-only
 * request continues the live KV with the family's tail (render_prelude).  Images are the template's vision
 * literal in place (L268), expanded by the core's walk.  Refused by name: tools loaded by tool_search (the
 * template renders one tools array). */
static bool qwen_render(pulsar_engine *e, server *s, chat_conversation *c, request *r, char *err, size_t errlen) {
    const qwen_effort qe = (qwen_effort)r->family_effort;
    if (c->loaded_tool_schemas.len) {
        snprintf(err, errlen, "tools loaded by tool_search are not served for the Qwen family");
        return false;
    }
    if (c->tools_raw && c->tool_choice != CHAT_TOOL_CHOICE_NONE) {
        r->qwen_tools_json = r->api == API_OPENAI ? xstrdup(c->tools_raw)
                                                  : qwen_tools_openai_shape(c, r->api, err, errlen);
        if (!r->qwen_tools_json) return false;
    }
    r->has_tools = r->qwen_tools_json != NULL;
    render_prelude(s, c, r);
    std::vector<qwen_msg_in> qm;
    std::vector<std::vector<qwen_tool_call_in>> qc;
    std::vector<std::string> notes;
    std::vector<qwen_msg_images> imgs;
    if (!qwen_messages(&c->msgs, 0, false, &qm, &qc, &notes, &imgs, err, errlen)) return false;
    qwen_render_out out;
    if (!qwen_chat_render({qm.data(), (int)qm.size(), r->qwen_tools_json, qe, true}, &out, err, errlen)) return false;
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
                           calls ? calls->raw_dsml : NULL};
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
    /* unnamed: stop before the name's opener -- the model samples "=name>" under the declared-name mask, as
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

/* The exact bytes a forced tool call is seeded with: close thinking, open the tool_calls block, and
 * (when a specific tool was requested) open the named invoke.  The prompt rewrite drops the render's
 * trailing "<think>" opener and skips the seed's close when the render already ended with one. */
static void deepseek_forced_call_seed(const request *r, buf *out) {
    const pulsar_dsml_syntax *d = pulsar_dsml_canonical(r->family->v41);
    buf_puts(out, "</think>\n\n");
    buf_puts(out, d->tool_calls_start);
    buf_puts(out, "\n");
    if (r->forced_tool_name && r->forced_tool_name[0]) {
        buf_puts(out, d->invoke_start);
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

/* ---- the tables and the dispatch ------------------------------------------------------------------- */

static const server_family_ops k_family_deepseek_v41 = {
    /* .name                  = */ "DeepSeek V4.1",
    /* .format                = */ PULSAR_CHAT_DS4_V41,
    /* .v41                   = */ true,
    /* .parser                = */ SERVER_PARSER_DSML,
    /* .resolve               = */ deepseek_resolve,
    /* .render                = */ deepseek_render,
    /* .trivial_header_tokens = */ deepseek_trivial_header_tokens,
    /* .output                = */ &k_parser_deepseek,
    /* .assistant_turn_sampled = */ deepseek_assistant_turn_sampled,
    /* .tool_result_tail      = */ deepseek_tool_result_tail,
    /* .tool_error_suffix     = */ deepseek_tool_error_suffix,
    /* .forced_call_seed      = */ deepseek_forced_call_seed,
    /* .forced_call_prefill   = */ deepseek_forced_call_prefill,
    /* .forced_name_open      = */ NULL,
    /* .forced_name_close     = */ NULL,   /* an unnamed DSML seed opens the block, not the name */
    /* .find_call_block       = */ find_next_dsml_tool_block,
};
static const server_family_ops k_family_deepseek_v4 = {
    /* .name                  = */ "DeepSeek V4 (0731)",
    /* .format                = */ PULSAR_CHAT_DS4_V4,
    /* .v41                   = */ false,
    /* .parser                = */ SERVER_PARSER_DSML,
    /* .resolve               = */ deepseek_resolve,
    /* .render                = */ deepseek_render,
    /* .trivial_header_tokens = */ deepseek_trivial_header_tokens,
    /* .output                = */ &k_parser_deepseek,
    /* .assistant_turn_sampled = */ deepseek_assistant_turn_sampled,
    /* .tool_result_tail      = */ deepseek_tool_result_tail,
    /* .tool_error_suffix     = */ deepseek_tool_error_suffix,
    /* .forced_call_seed      = */ deepseek_forced_call_seed,
    /* .forced_call_prefill   = */ deepseek_forced_call_prefill,
    /* .forced_name_open      = */ NULL,
    /* .forced_name_close     = */ NULL,   /* an unnamed DSML seed opens the block, not the name */
    /* .find_call_block       = */ find_next_dsml_tool_block,
};
static const server_family_ops k_family_qwen = {
    /* .name                  = */ "Qwen",
    /* .format                = */ PULSAR_CHAT_QWEN,
    /* .v41                   = */ false,
    /* .parser                = */ SERVER_PARSER_QWEN,
    /* .resolve               = */ qwen_resolve,
    /* .render                = */ qwen_render,
    /* .trivial_header_tokens = */ qwen_trivial_header_tokens,
    /* .output                = */ &k_parser_qwen,
    /* .assistant_turn_sampled = */ qwen_assistant_turn_sampled,
    /* .tool_result_tail      = */ qwen_tool_result_tail,
    /* .tool_error_suffix     = */ qwen_tool_error_suffix,
    /* .forced_call_seed      = */ qwen_forced_call_seed,
    /* .forced_call_prefill   = */ qwen_forced_call_prefill,
    /* .forced_name_open      = */ "=",   /* an unnamed seed ends at <function: "=get" is one token */
    /* .forced_name_close     = */ ">",
    /* .find_call_block       = */ qwen_find_call_block,
};

const server_family_ops *server_family_for_format(pulsar_chat_format fmt) {
    switch (fmt) {
    case PULSAR_CHAT_DS4_V4:  return &k_family_deepseek_v4;
    case PULSAR_CHAT_DS4_V41: return &k_family_deepseek_v41;
    case PULSAR_CHAT_QWEN:    return &k_family_qwen;
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
    if (!r->family->resolve(e, c, r, err, errlen)) return false;
    /* The protocol's tool-result checks: every result's call is in this request's
     * history or bound to the live frontier (parse-without-server skips them). */
    if (s && r->api == API_ANTHROPIC &&
        !s->anthropic_validate_tool_results(&c->msgs, &r->anthropic_requires_live_tool_state, err, errlen))
        return false;
    if (s && r->api == API_RESPONSES &&
        !s->responses_validate_tool_outputs(&c->msgs, r->think_mode, &r->responses_requires_live_tool_state,
                                            &r->responses_requires_live_reasoning, err, errlen))
        return false;
    return r->family->render(e, s, c, r, err, errlen);
}

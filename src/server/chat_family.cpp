/* L267: the model-family half of a chat request.
 *
 * A protocol parser (api_parse.cpp: OpenAI chat, Anthropic Messages, Responses) reads its wire format
 * into a chat_conversation -- the messages, the tools as sent, the tool_choice, and the thinking / effort
 * controls exactly as sent.  render_chat_conversation() is everything after that, and it is the loaded
 * family's: how the controls resolve to a thinking mode, the protocol's tool-result checks (they read
 * the resolved mode), then the prompt.  DeepSeek renders its own template (prompt_render.cpp) with its
 * tool memory, live-continuation suffixes and forced-call prefill; Qwen renders HF's template
 * (src/lib/qwen_chat) and refuses by name what that template cannot express.  A new family adds one arm
 * here; a new protocol adds one parser and no arm. */
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
    if (!r->chat_v41 && enabled && !pulsar_think_effort_v4_valid(effort)) {
        if (err && errlen) snprintf(err, errlen, "reasoning_effort: the V4 (0731) encoder has three levels -- low, high, max");
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
    /* parse-without-server (the renderer gate, the golden) has no tool memory */
    if (s) {
        s->kv_cache_restore_tool_memory_for_messages(&c->msgs);
        s->tool_memory_attach_to_messages(&c->msgs, &r->tool_replay);
    }
    if (r->api == API_ANTHROPIC) anthropic_prepare_live_continuation(r, &c->msgs);
    if (r->api == API_RESPONSES) responses_prepare_live_continuation(r, &c->msgs);
    /* L223: keep the client-data ranges; the tokeniser turns a spelling inside
     * client text into ordinary tokens instead of a control token. */
    free(r->prompt_spans);
    r->prompt_spans = NULL;
    r->prompt_n_spans = 0;
    r->prompt_text = render_chat_prompt_text_spans(&c->msgs, r->has_tools ? schemas.ptr : NULL,
                                                   &r->tool_orders, r->think_mode, r->chat_v41,
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
        request_apply_forced_tool_prefill(r);
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
static bool qwen_resolve(const chat_conversation *c, request *r, qwen_effort *qe, char *err, size_t errlen) {
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

/* HF's apply_chat_template byte for byte (qwen_chat_render, with the L223 client-span map), tokenised
 * through the engine's tokenizer entry.  Refused by name, because the template cannot express them: a
 * required or named tool call, image content, tool results whose call is not in the history (the
 * family keeps no live tool state), and tools loaded by Responses' tool_search.  tool_choice "none"
 * renders the conversation without the tools.  The request's top-level system / instructions field
 * renders first, wherever its protocol put it. */
static bool qwen_render(pulsar_engine *e, chat_conversation *c, request *r, qwen_effort qe, char *err,
                        size_t errlen) {
    if (c->tool_choice == CHAT_TOOL_CHOICE_ANY || c->tool_choice == CHAT_TOOL_CHOICE_NAMED) {
        if (c->tool_choice_wire)
            snprintf(err, errlen, "tool_choice: \"%.40s\" is not served for the Qwen family "
                                  "(its chat template cannot force a call); use \"auto\" or \"none\"",
                     c->tool_choice_wire);
        else
            snprintf(err, errlen, "tool_choice: a named function is not served for the Qwen family "
                                  "(its chat template cannot force a call); use \"auto\" or \"none\"");
        return false;
    }
    if (r->anthropic_requires_live_tool_state || r->responses_requires_live_tool_state) {
        snprintf(err, errlen, "a tool result answers a call that is not in this request's history; the Qwen "
                              "family keeps no live tool state -- replay the full history");
        return false;
    }
    if (c->loaded_tool_schemas.len) {
        snprintf(err, errlen, "tools loaded by tool_search are not served for the Qwen family");
        return false;
    }
    char *tools = NULL;
    if (c->tools_raw && c->tool_choice != CHAT_TOOL_CHOICE_NONE) {
        tools = r->api == API_OPENAI ? xstrdup(c->tools_raw) : qwen_tools_openai_shape(c, r->api, err, errlen);
        if (!tools) return false;
    }
    /* chat_msgs -> the renderer's messages; the pointers borrow `c->msgs`. */
    std::vector<qwen_msg_in> qm;
    std::vector<std::vector<qwen_tool_call_in>> qc((size_t)c->msgs.len);
    for (int pass = 0; pass < 2; pass++) {
        for (int i = 0; i < c->msgs.len; i++) {
            const chat_msg *m = &c->msgs.v[i];
            if (m->system_field != (pass == 0)) continue;
            if (m->images_len > 0) {
                snprintf(err, errlen, "message %d: image content is not served for the Qwen family", i);
                free(tools);
                return false;
            }
            for (int k = 0; k < m->calls.len; k++)
                qc[(size_t)i].push_back({m->calls.v[k].name, m->calls.v[k].arguments});
            qm.push_back({m->role, m->content, m->reasoning, qc[(size_t)i].empty() ? NULL : qc[(size_t)i].data(),
                          m->calls.len});
        }
    }
    qwen_render_out out;
    if (!qwen_chat_render({qm.data(), (int)qm.size(), tools, qe, true}, &out, err, errlen)) {
        free(tools);
        return false;
    }
    r->prompt_text = xstrndup(out.text.data(), out.text.size());
    free(r->prompt_spans);
    r->prompt_spans = NULL;
    r->prompt_n_spans = (uint32_t)out.spans.size();
    if (r->prompt_n_spans) {
        r->prompt_spans = (pulsar_text_span *)server_xmalloc(out.spans.size() * sizeof(pulsar_text_span));
        memcpy(r->prompt_spans, out.spans.data(), out.spans.size() * sizeof(pulsar_text_span));
    }
    r->has_tools = tools != NULL;
    r->qwen_tools_json = tools;
    if (e) pulsar_tokenize_rendered_chat_spans(e, r->prompt_text, r->prompt_spans, r->prompt_n_spans, &r->prompt);
    return true;
}



/* ---- the dispatch --------------------------------------------------------------------------------- */

bool render_chat_conversation(pulsar_engine *e, pulsar_chat_format fmt, server *s, chat_conversation *c,
                              request *r, char *err, size_t errlen) {
    const bool qwen = fmt == PULSAR_CHAT_QWEN;
    qwen_effort qe = QWEN_EFFORT_NONE;
    /* The chat template family follows the LOADED model (L218 s123): the renderer,
     * the forced-prefill and the KV-key suffix builders all read it from the
     * request, so a 0731 artifact cannot be primed with V4.1's template by one of
     * them and V4's by another. */
    r->chat_qwen = qwen;
    if (!qwen) r->chat_v41 = fmt == PULSAR_CHAT_DS4_V41;
    if (qwen ? !qwen_resolve(c, r, &qe, err, errlen) : !deepseek_resolve(e, c, r, err, errlen)) return false;
    /* The protocol's tool-result checks: every result's call is in this request's
     * history or bound to the live frontier (parse-without-server skips them). */
    if (s && r->api == API_ANTHROPIC &&
        !s->anthropic_validate_tool_results(&c->msgs, &r->anthropic_requires_live_tool_state, err, errlen))
        return false;
    if (s && r->api == API_RESPONSES &&
        !s->responses_validate_tool_outputs(&c->msgs, r->think_mode, &r->responses_requires_live_tool_state,
                                            &r->responses_requires_live_reasoning, err, errlen))
        return false;
    return qwen ? qwen_render(e, c, r, qe, err, errlen) : deepseek_render(e, s, c, r, err, errlen);
}

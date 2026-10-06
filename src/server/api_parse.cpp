#include "pulsar_server_internal.h"



/* Shared sampling-knob parsing for every request surface. Returns 1 if `key`
 * was a sampling knob and its value was consumed, 0 if it is not a sampling
 * knob (caller keeps matching), -1 on a malformed value (caller frees key and
 * goes to its error label). One definition so a knob is never silently dropped
 * by whichever arm a given endpoint's copy-paste happened to include -- every
 * surface accepts temperature/top_p/min_p/top_k/seed identically. logprobs
 * stays endpoint-local (chat only). */
int parse_sampling_key(const char *key, const char **p, request *r) {
    if (!strcmp(key, "temperature")) {
        double v = 0.0;
        /* strtod also accepts nan/inf lexemes.  A NaN fails every range test
         * below (and the engine's `temperature <= 0` greedy test), so it would
         * reach the sampler as a non-finite distribution (review B2).  Refuse
         * non-finite knobs at the surface. */
        if (!json_number(p, &v) || !isfinite(v)) return -1;
        r->temperature = (float)v;
        r->has_temperature = true;
    } else if (!strcmp(key, "top_p")) {
        double v = 0.0;
        if (!json_number(p, &v) || !isfinite(v)) return -1;
        /* The engine maps anything outside (0,1] to 1.0 (pulsar_sample_dist_build).
         * Clamp here too so the stored request matches what will run, and say so
         * once: top_p=0 is a common near-greedy idiom that clients expect to
         * narrow the nucleus, not widen it to the full vocab. */
        if (!(v > 0.0 && v <= 1.0)) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                server_log(PULSAR_LOG_WARNING,
                           "top_p=%g is outside (0,1]; using 1.0 (full nucleus)", v);
            }
            v = 1.0;
        }
        r->top_p = (float)v;
        r->has_top_p = true;
    } else if (!strcmp(key, "min_p")) {
        double v = 0.0;
        if (!json_number(p, &v) || !isfinite(v)) return -1;
        /* out-of-range disables the filter, matching the engine sampler
         * (sample_top_p_min_p); an unvalidated min_p>1 collapses to greedy. */
        if (v < 0.0 || v > 1.0) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                server_log(PULSAR_LOG_WARNING,
                           "min_p=%g is outside [0,1]; filter disabled", v);
            }
            v = 0.0;
        }
        r->min_p = (float)v;
        r->has_min_p = true;
    } else if (!strcmp(key, "top_k")) {
        if (!json_int(p, &r->top_k)) return -1;
        /* Same range the engine enforces (>1024 -> 1024, <=0 -> 0). */
        if (r->top_k < 0 || r->top_k > 1024) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                server_log(PULSAR_LOG_WARNING,
                           "top_k=%d is outside [0,1024]; using %d",
                           r->top_k, r->top_k < 0 ? 0 : 1024);
            }
            r->top_k = r->top_k < 0 ? 0 : 1024;
        }
        r->has_top_k = true;
    } else if (!strcmp(key, "seed")) {
        double v = 0.0;
        if (!json_number(p, &v)) return -1;
        /* NaN fails v>0; +inf or >=2^64 would be UB in the cast. */
        r->seed = (v > 0.0 && v < 18446744073709551616.0) ? (uint64_t)v
                : v > 0.0                                 ? UINT64_MAX
                                                          : 0;
    } else {
        return 0;
    }
    return 1;
}


/* The API parsers are intentionally selective JSON parsers: they keep only
 * fields that affect model semantics, rendering, streaming, or cache keys, and
 * skip extension fields.  Each reads ITS protocol into a chat_conversation and
 * the request's protocol fields; render_chat_conversation (chat_family.cpp) is
 * the loaded family's half -- thinking, the prompt, its tokens (L267). */

/* One thinking / effort control, kept as sent for the family (chat_control). */
static bool take_control(const char **p, chat_conversation *c, const char *key) {
    char *raw = NULL;
    if (!json_raw_value(p, &raw)) return false;
    chat_conversation_control(c, key, raw);
    return true;
}

/* The controls a chat protocol may carry, by request key; the static spelling is the control's key. */
static const char *chat_control_key(const char *key) {
    static const char *const keys[] = {"thinking", "think", "enable_thinking", "reasoning_effort",
                                       "output_config", "chat_template_kwargs"};
    for (const char *k : keys)
        if (!strcmp(key, k)) return k;
    return NULL;
}

/* The tools array: as sent (NULL for null), and one function schema a line with
 * their declared order (parse_tools_value, every protocol's dialect).  Anthropic's
 * server-tool entries are checked first, with the Messages API's own wording. */
static bool take_tools(const char **p, chat_conversation *c, request *r, bool anthropic, char *err,
                       size_t errlen) {
    char *raw = NULL;
    if (!json_raw_value(p, &raw)) return false;
    if (anthropic && !anthropic_tools_supported(raw, err, errlen)) {
        free(raw);
        return false;
    }
    free(c->tool_schemas);
    c->tool_schemas = NULL;
    const char *tp = raw;
    if (!parse_tools_value(&tp, &c->tool_schemas, &r->tool_orders)) {
        free(raw);
        return false;
    }
    free(c->tools_raw);
    c->tools_raw = NULL;
    const char *q = raw;
    json_ws(&q);
    if (json_lit(&q, "null")) free(raw);
    else c->tools_raw = raw;
    return true;
}

/* OpenAI's tool_choice: "none" | "auto" | "required", or a named function
 * {"type":"function","function":{"name":...}}. */
static bool parse_openai_tool_choice(const char **p, chat_conversation *c, request *r, char *err,
                                     size_t errlen) {
    json_ws(p);
    if (**p == '"') {
        char *choice = NULL;
        if (!json_string(p, &choice)) return false;
        free(c->tool_choice_wire);
        c->tool_choice_wire = choice;
        if (!strcmp(choice, "none")) c->tool_choice = CHAT_TOOL_CHOICE_NONE;
        else if (!strcmp(choice, "auto")) c->tool_choice = CHAT_TOOL_CHOICE_AUTO;
        else if (!strcmp(choice, "required")) c->tool_choice = CHAT_TOOL_CHOICE_ANY;
        else {
            snprintf(err, errlen, "tool_choice: \"%.40s\" is not one of \"none\", \"auto\" or \"required\"", choice);
            return false;
        }
        return true;
    }
    if (**p == '{') {
        char *raw = NULL;
        if (!json_raw_value(p, &raw)) return false;
        char *function = json_object_member_raw(raw, "function");
        char *name_raw = function ? json_object_member_raw(function, "name") : NULL;
        const char *np = name_raw;
        char *name = NULL;
        const bool ok = np && json_string(&np, &name) && name[0];
        free(raw);
        free(function);
        free(name_raw);
        if (!ok) {
            free(name);
            snprintf(err, errlen, "tool_choice: a function object needs function.name");
            return false;
        }
        free(r->forced_tool_name);
        r->forced_tool_name = name;
        free(c->tool_choice_wire);
        c->tool_choice_wire = NULL;
        c->tool_choice = CHAT_TOOL_CHOICE_NAMED;
        return true;
    }
    c->tool_choice = CHAT_TOOL_CHOICE_AUTO;
    return json_skip_value(p);
}

bool parse_chat_conversation_openai(const char *body, chat_conversation *c, request *r, char *err,
                                    size_t errlen) {
    if (err && errlen) err[0] = '\0';
    const char *p = body;
    bool got_messages = false;
    bool got_top_logprobs = false;
    int skr = 0;
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) return false;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            return false;
        }
        p++;
        bool ok = true;
        const char *control = chat_control_key(key);
        if (!strcmp(key, "messages")) {
            chat_msgs_free(&c->msgs);
            ok = parse_messages(&p, &c->msgs, err, errlen);
            got_messages = true;
        } else if (!strcmp(key, "tools")) {
            ok = take_tools(&p, c, r, false, err, errlen);
        } else if (!strcmp(key, "tool_choice")) {
            ok = parse_openai_tool_choice(&p, c, r, err, errlen);
        } else if (!strcmp(key, "model")) {
            free(r->model);
            ok = json_string(&p, &r->model);
            r->model_from_request = true;
        } else if (!strcmp(key, "max_tokens") || !strcmp(key, "max_completion_tokens")) {
            ok = json_int(&p, &r->max_tokens);
        } else if ((skr = parse_sampling_key(key, &p, r)) != 0) {
            ok = skr > 0;
        } else if (!strcmp(key, "logprobs")) {
            /* OpenAI SDKs send an explicit null for "not set" on both logprobs
             * fields, so accept it as absent rather than as malformed JSON. */
            json_ws(&p);
            ok = json_lit(&p, "null") || json_bool(&p, &r->logprobs);
        } else if (!strcmp(key, "top_logprobs")) {
            /* Range and the logprobs:true dependency are checked after the loop
             * (the two keys can arrive in either order).  Not parsed with
             * json_int: that folds negatives to 0 and truncates fractions, so
             * the post-loop rejection could never fire for exactly the inputs
             * it exists to reject. */
            json_ws(&p);
            if (!json_lit(&p, "null")) {
                double v = 0.0;
                ok = json_number(&p, &v);
                r->top_logprobs = (v >= 0 && v <= PULSAR_SERVER_MAX_TOP_LOGPROBS
                                   && v == (double)(int)v) ? (int)v : -1;
                got_top_logprobs = true;
            }
        } else if (!strcmp(key, "stream")) {
            ok = json_bool(&p, &r->stream);
        } else if (!strcmp(key, "stream_options")) {
            ok = parse_stream_options(&p, &r->stream_include_usage);
        } else if (control && strcmp(control, "output_config")) {
            ok = take_control(&p, c, control);
        } else if (!strcmp(key, "stop")) {
            ok = parse_stop(&p, &r->stops);
        } else {
            ok = json_skip_value(&p);
        }
        free(key);
        if (!ok) return false;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (*p != '}') return false;
    if (!got_messages) {
        snprintf(err, errlen, "missing messages");
        return false;
    }
    /* OpenAI's two logprobs rules, both 400s there: top_logprobs is meaningless
     * without logprobs:true, and the per-position alternative count is capped.
     * Rejecting is the whole point -- a silently clamped k would hand back a
     * distribution the client did not ask for and cannot detect. */
    if (got_top_logprobs && !r->logprobs) {
        snprintf(err, errlen, "top_logprobs requires logprobs to be true");
        return false;
    }
    if (r->top_logprobs < 0 || r->top_logprobs > PULSAR_SERVER_MAX_TOP_LOGPROBS) {
        snprintf(err, errlen, "top_logprobs must be an integer between 0 and %d",
                 PULSAR_SERVER_MAX_TOP_LOGPROBS);
        return false;
    }
    if (!r->logprobs) r->top_logprobs = 0;
    return true;
}

/* A protocol half, then the loaded family's: the one shape every chat endpoint
 * takes.  The engine is optional (the renderer gate renders with no model). */
static bool parse_and_render(pulsar_engine *e, server *s, const char *body, request *r, char *err,
                             size_t errlen,
                             bool (*protocol)(const char *, chat_conversation *, request *, char *, size_t)) {
    chat_conversation c;
    memset(&c, 0, sizeof(c));
    const bool ok = protocol(body, &c, r, err, errlen) &&
                    render_chat_conversation(e, pulsar_engine_chat_format(e), s, &c, r, err, errlen);
    chat_conversation_free(&c);
    if (!ok) {
        /* A refusal that named its reason keeps it; only a plain shape error
         * falls back to the generic. */
        if (err && errlen && !err[0]) snprintf(err, errlen, "invalid JSON request");
        request_free(r);
    }
    return ok;
}

bool parse_chat_request_render(pulsar_engine *e, server *s, const char *body, int def_tokens,
                               request *r, char *err, size_t errlen) {
    request_init(r, REQ_CHAT, def_tokens);
    return parse_and_render(e, s, body, r, err, errlen, parse_chat_conversation_openai);
}



bool parse_chat_request(pulsar_engine *e, server *s, const char *body, int def_tokens,
                        request *r, char *err, size_t errlen) {
    return parse_chat_request_render(e, s, body, def_tokens, r, err, errlen);
}



bool parse_chat_conversation_anthropic(const char *body, chat_conversation *c, request *r, char *err,
                                       size_t errlen) {
    r->api = API_ANTHROPIC;
    if (err && errlen) err[0] = '\0';
    const char *p = body;
    bool got_messages = false;
    int skr = 0;
    char *system = NULL;

    json_ws(&p);
    if (*p != '{') return false;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto bad;
        }
        p++;
        bool ok;
        ok = true;
        if (!strcmp(key, "messages")) {
            chat_msgs_free(&c->msgs);
            ok = parse_anthropic_messages(&p, &c->msgs, err, errlen);
            got_messages = true;
        } else if (!strcmp(key, "system")) {
            free(system);
            ok = parse_anthropic_system(&p, &system);
        } else if (!strcmp(key, "tools")) {
            ok = take_tools(&p, c, r, true, err, errlen);
        } else if (!strcmp(key, "tool_choice")) {
            /* {"type":"auto"|"any"|"none"|"tool", "name":...}: "any" and "tool"
             * require a call (a named one for "tool"). */
            json_ws(&p);
            if (*p == '{') {
                p++;
                json_ws(&p);
                while (ok && *p && *p != '}') {
                    char *ckey = NULL;
                    ok = json_string(&p, &ckey);
                    json_ws(&p);
                    ok = ok && *p == ':';
                    if (ok) p++;
                    if (ok && !strcmp(ckey, "type")) {
                        char *choice = NULL;
                        ok = json_string(&p, &choice);
                        if (ok) {
                            c->tool_choice = !strcmp(choice, "none") ? CHAT_TOOL_CHOICE_NONE
                                           : !strcmp(choice, "any")  ? CHAT_TOOL_CHOICE_ANY
                                           : !strcmp(choice, "tool") ? CHAT_TOOL_CHOICE_NAMED
                                                                     : CHAT_TOOL_CHOICE_AUTO;
                            free(c->tool_choice_wire);
                            c->tool_choice_wire = choice;
                        }
                    } else if (ok && !strcmp(ckey, "name")) {
                        free(r->forced_tool_name);
                        r->forced_tool_name = NULL;
                        ok = json_string(&p, &r->forced_tool_name);
                    } else if (ok) {
                        ok = json_skip_value(&p);
                    }
                    free(ckey);
                    json_ws(&p);
                    if (*p == ',') p++;
                    json_ws(&p);
                }
                ok = ok && *p == '}';
                if (ok) p++;
            } else {
                ok = json_skip_value(&p);
            }
        } else if (!strcmp(key, "model")) {
            free(r->model);
            ok = json_string(&p, &r->model);
            r->model_from_request = true;
        } else if (!strcmp(key, "max_tokens")) {
            ok = json_int(&p, &r->max_tokens);
        } else if ((skr = parse_sampling_key(key, &p, r)) != 0) {
            ok = skr > 0;
        } else if (!strcmp(key, "stream")) {
            ok = json_bool(&p, &r->stream);
        } else if (!strcmp(key, "stop_sequences")) {
            ok = parse_stop(&p, &r->stops);
        } else if (!strcmp(key, "thinking") || !strcmp(key, "output_config") || !strcmp(key, "reasoning_effort")) {
            ok = take_control(&p, c, chat_control_key(key));
        } else {
            ok = json_skip_value(&p);
        }
        free(key);
        if (!ok) goto bad;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (*p != '}') goto bad;
    if (!got_messages) {
        snprintf(err, errlen, "missing messages");
        goto bad;
    }
    /* The system FIELD renders in the system region (chat_msg::system_field). */
    if (system && system[0]) {
        chat_msg msg = {0};
        msg.role = xstrdup("system");
        msg.content = system;
        msg.system_field = true;
        system = NULL;
        chat_msgs_push(&c->msgs, msg);
    }
    free(system);
    return true;
bad:
    free(system);
    return false;
}

bool parse_anthropic_request(pulsar_engine *e, server *s, const char *body, int def_tokens,
                                    request *r, char *err, size_t errlen) {
    request_init(r, REQ_CHAT, def_tokens);
    return parse_and_render(e, s, body, r, err, errlen, parse_chat_conversation_anthropic);
}



/* Responses API: convert a content-array item (input_text/output_text/text/
 * input_image) into a concatenated string. Strict shape check: bare string,
 * null, or an array of recognized blocks. Numbers / objects / arrays-of-
 * primitives at the top level all reject so the client sees a 400 instead of an
 * answer built on silently dropped context.
 *
 * An `input_image` block decodes exactly like a chat `image_url` block (one
 * authority: server_add_image_block): the encoded file is attached to `msg` and
 * a PULSAR_IMAGE_PLACEHOLDER takes the block's position in the text, so the
 * renderer's image expander sees the same shape it sees from the other two
 * endpoints.  `msg` is NULL in the two positions where an image is not a thing
 * (a tool output's content, a reasoning summary) and an image there fails
 * closed. */
static bool parse_responses_content_array(const char **p, char **out, chat_msg *msg,
                                          char *err, size_t errlen) {
    *out = NULL;  /* reparse-in-place double-free guard, see json_string_n */
    json_ws(p);
    if (**p == '"') return json_string(p, out);
    if (json_lit(p, "null")) {
        *out = xstrdup("");
        return true;
    }
    if (**p != '[') {
        return false;
    }
    (*p)++;
    buf b = {0};
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *s = NULL;
            if (!json_string(p, &s)) goto fail;
            buf_puts(&b, s);
            free(s);
        } else if (**p == '{') {
            (*p)++;
            char *type = NULL;
            char *text = NULL;
            char *image_url = NULL;
            char *file_id = NULL;
            json_ws(p);
            while (**p && **p != '}') {
                char *key = NULL;
                if (!json_string(p, &key)) {
                    free(type);
                    free(text);
                    free(image_url);
                    free(file_id);
                    goto fail;
                }
                json_ws(p);
                if (**p != ':') {
                    free(key);
                    free(type);
                    free(text);
                    free(image_url);
                    free(file_id);
                    goto fail;
                }
                (*p)++;
                if (!strcmp(key, "type")) {
                    free(type);
                    if (!json_string(p, &type)) {
                        free(key);
                        free(text);
                        free(image_url);
                        free(file_id);
                        goto fail;
                    }
                } else if (!strcmp(key, "text")) {
                    free(text);
                    /* The text field of a typed content block is a plain JSON
                     * string. Accept null as the empty string for parity with
                     * upstream serializers that emit null for empty blocks. */
                    json_ws(p);
                    if (json_lit(p, "null")) {
                        text = xstrdup("");
                    } else if (!json_string(p, &text)) {
                        free(key);
                        free(type);
                        free(image_url);
                        free(file_id);
                        goto fail;
                    }
                } else if (!strcmp(key, "image_url")) {
                    /* The Responses schema spells this field as a bare string;
                     * tolerate OpenAI's {"url": ...} wrapper too, exactly as the
                     * chat reader does, so a payload ported between the two
                     * surfaces lands instead of dying on a shape difference. */
                    free(image_url);
                    json_ws(p);
                    if (**p == '{') {
                        (*p)++;
                        json_ws(p);
                        while (**p && **p != '}') {
                            char *ik = NULL;
                            if (!json_string(p, &ik)) {
                                free(key);
                                free(type);
                                free(text);
                                free(file_id);
                                goto fail;
                            }
                            json_ws(p);
                            if (**p != ':') {
                                free(ik);
                                free(key);
                                free(type);
                                free(text);
                                free(file_id);
                                goto fail;
                            }
                            (*p)++;
                            if (!strcmp(ik, "url")) {
                                free(image_url);
                                if (!json_string(p, &image_url)) {
                                    free(ik);
                                    free(key);
                                    free(type);
                                    free(text);
                                    free(file_id);
                                    goto fail;
                                }
                            } else if (!json_skip_value(p)) {
                                free(ik);
                                free(key);
                                free(type);
                                free(text);
                                free(file_id);
                                goto fail;
                            }
                            free(ik);
                            json_ws(p);
                            if (**p == ',') (*p)++;
                            json_ws(p);
                        }
                        if (**p != '}') {
                            free(key);
                            free(type);
                            free(text);
                            free(file_id);
                            goto fail;
                        }
                        (*p)++;
                    } else if (!json_string(p, &image_url)) {
                        free(key);
                        free(type);
                        free(text);
                        free(file_id);
                        goto fail;
                    }
                } else if (!strcmp(key, "file_id")) {
                    /* An uploaded-file reference. This server does not fetch
                     * remote content, so the value is captured only to name it
                     * in the refusal below rather than being dropped silently. */
                    free(file_id);
                    if (!json_string(p, &file_id)) {
                        free(key);
                        free(type);
                        free(text);
                        free(image_url);
                        goto fail;
                    }
                } else if (!json_skip_value(p)) {
                    free(key);
                    free(type);
                    free(text);
                    free(image_url);
                    free(file_id);
                    goto fail;
                }
                free(key);
                json_ws(p);
                if (**p == ',') (*p)++;
                json_ws(p);
            }
            if (**p != '}') {
                free(type);
                free(text);
                free(image_url);
                free(file_id);
                goto fail;
            }
            (*p)++;
            /* Fail closed: a content object must carry a known type AND the
             * field that type needs. Anything else -- missing type, missing
             * text, file/audio types, future schema-drift -- is rejected so the
             * client gets a 400 instead of an answer built on context the
             * server discarded silently. `image_url` is accepted as an alias of
             * `input_image` for the same reason the text branch accepts five
             * spellings. */
            const bool is_text_block = type && (
                !strcmp(type, "input_text") ||
                !strcmp(type, "output_text") ||
                !strcmp(type, "text") ||
                !strcmp(type, "summary_text") ||
                !strcmp(type, "reasoning_text"));
            const bool is_image_block = type && (
                !strcmp(type, "input_image") ||
                !strcmp(type, "image_url"));
            if (is_image_block) {
                if (!image_url || !image_url[0]) {
                    if (err && errlen) {
                        snprintf(err, errlen,
                                 "input_image needs an inline \"image_url\" base64 data: URL%s",
                                 file_id ? " (a file_id is not fetched by this server)" : "");
                    }
                    free(type);
                    free(text);
                    free(image_url);
                    free(file_id);
                    goto fail;
                }
                if (!server_add_image_block(msg, image_url, &b, err, errlen)) {
                    free(type);
                    free(text);
                    free(image_url);
                    free(file_id);
                    goto fail;
                }
            } else if (!is_text_block || !text) {
                if (!is_text_block && err && errlen) {
                    snprintf(err, errlen,
                             "unsupported content block type \"%s\"; this server accepts text, "
                             "input_image and image_url blocks", type ? type : "(missing)");
                }
                free(type);
                free(text);
                free(image_url);
                free(file_id);
                goto fail;
            } else {
                buf_puts(&b, text);
            }
            free(type);
            free(text);
            free(image_url);
            free(file_id);
        } else {
            /* Reject primitives, arrays-of-arrays, nulls: a content array
             * element must be either a string or a typed text object. */
            goto fail;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto fail;
    (*p)++;
    *out = buf_take(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}



/* Codex /v1/responses input items have a `type` discriminator (message,
 * function_call, function_call_output, reasoning, custom_tool_call,
 * custom_tool_call_output, ...). We collapse them into chat_msgs the same way
 * the chat completion / Anthropic parsers do, so the rest of the engine sees a
 * single conversation history shape.
 *
 * Protocol contract for stateless replay:
 *   - The client must replay response.output items before tool outputs.
 *   - For reasoning models, the replay must also include reasoning state.  DS4
 *     can render plain reasoning summaries/content, but it cannot decrypt
 *     reasoning.encrypted_content.  If live state is unavailable and the replay
 *     only contains visible messages/tool calls, later validation marks it as a
 *     lower-fidelity replay; generate_job() logs that and continues from the
 *     visible transcript rather than killing a recoverable agent session.
 *
 * Reasoning items are merged into the next assistant message so
 * render_chat_prompt_text can wrap them in <think>. */
bool parse_responses_input(const char **p, chat_msgs *msgs,
                                  buf *loaded_tool_schemas,
                                  tool_schema_orders *orders,
                                  char *err, size_t errlen) {
    char local_err[192];
    if (!err || errlen == 0) {
        err = local_err;
        errlen = sizeof local_err;
    }
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;

    buf pending_reasoning = {0};
    /* An item's content is parsed before the item's type is known (JSON keys
     * arrive in any order), so images land here first and are adopted by the
     * message the item turns out to be -- or dropped when it is not a
     * message. */
    chat_msg item_images = {0};

    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') goto fail;
        (*p)++;
        char *type = NULL;
        char *role = NULL;
        char *content = NULL;
        char *name = NULL;
        char *tool_namespace = NULL;
        char *call_id = NULL;
        char *item_id = NULL;
        char *arguments = NULL;
        char *output = NULL;
        char *input_str = NULL;
        char *summary = NULL;
        char *action = NULL;
        char *result = NULL;
        char *tools_json = NULL;
        char *status_str = NULL;
        json_ws(p);
        while (**p && **p != '}') {
            char *key = NULL;
            if (!json_string(p, &key)) goto item_fail;
            json_ws(p);
            if (**p != ':') {
                free(key);
                goto item_fail;
            }
            (*p)++;
            if (!strcmp(key, "type")) {
                free(type);
                if (!json_string(p, &type)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "role")) {
                free(role);
                if (!json_string(p, &role)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "content")) {
                free(content);
                if (!parse_responses_content_array(p, &content, &item_images, err, errlen)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "name")) {
                free(name);
                if (!json_string(p, &name)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "namespace")) {
                free(tool_namespace);
                if (!json_string(p, &tool_namespace)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "call_id")) {
                free(call_id);
                if (!json_string(p, &call_id)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "id")) {
                free(item_id);
                if (!json_string(p, &item_id)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "arguments")) {
                free(arguments);
                json_ws(p);
                if (**p == '"') {
                    if (!json_string(p, &arguments)) {
                        free(key);
                        goto item_fail;
                    }
                } else if (!json_raw_value(p, &arguments)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "output")) {
                free(output);
                json_ws(p);
                if (**p == '[') {
                    if (!parse_responses_content_array(p, &output, NULL, err, errlen)) {
                        free(key);
                        goto item_fail;
                    }
                } else if (**p == '"') {
                    if (!json_string(p, &output)) {
                        free(key);
                        goto item_fail;
                    }
                } else if (!json_raw_value(p, &output)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "input")) {
                free(input_str);
                json_ws(p);
                if (**p == '"') {
                    if (!json_string(p, &input_str)) {
                        free(key);
                        goto item_fail;
                    }
                } else if (!json_raw_value(p, &input_str)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "summary")) {
                free(summary);
                if (!parse_responses_content_array(p, &summary, NULL, err, errlen)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "action")) {
                free(action);
                if (!json_raw_value(p, &action)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "result")) {
                free(result);
                json_ws(p);
                if (**p == '"') {
                    if (!json_string(p, &result)) {
                        free(key);
                        goto item_fail;
                    }
                } else if (!json_raw_value(p, &result)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "status")) {
                free(status_str);
                if (!json_string(p, &status_str)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "tools")) {
                /* tool_search_output items carry their discovered tool list
                 * here instead of in `output` / `result`. Keep it separate
                 * from the human-visible result body so malformed tool lists
                 * never get mistaken for normal tool output. */
                free(tools_json);
                if (!json_raw_value(p, &tools_json)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!json_skip_value(p)) {
                free(key);
                goto item_fail;
            }
            free(key);
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
            continue;
item_fail:
            free(type);
            free(role);
            free(content);
            free(name);
            free(tool_namespace);
            free(call_id);
            free(item_id);
            free(arguments);
            free(output);
            free(input_str);
            free(summary);
            free(action);
            free(result);
            free(tools_json);
            free(status_str);
            chat_msg_clear_images(&item_images);
            buf_free(&pending_reasoning);
            return false;
        }
        if (**p != '}') {
            free(type);
            free(role);
            free(content);
            free(name);
            free(tool_namespace);
            free(call_id);
            free(item_id);
            free(arguments);
            free(output);
            free(input_str);
            free(summary);
            free(action);
            free(result);
            free(tools_json);
            free(status_str);
            goto fail;
        }
        (*p)++;

        const char *t = type ? type : "message";
        /* Replayed items must be in a terminal "completed" state. in_progress,
         * incomplete, and failed all represent partial model state the client
         * never confirmed — feeding them back as history would let DS4 continue
         * from a tool action that never finished. Reject explicitly. */
        if (status_str && status_str[0] &&
            strcmp(status_str, "completed") != 0)
        {
            free(type);
            free(role);
            free(content);
            free(name);
            free(tool_namespace);
            free(call_id);
            free(item_id);
            free(arguments);
            free(output);
            free(input_str);
            free(summary);
            free(action);
            free(result);
            free(tools_json);
            free(status_str);
            chat_msg_clear_images(&item_images);
            buf_free(&pending_reasoning);
            return false;
        }
        /* Three classes of items:
         *   1. consumes_reasoning: assistant message / function_call / hosted-tool
         *      call. Attaches pending reasoning to its own assistant message.
         *   2. is_bookkeeping: compaction / context_compaction etc. Semantically
         *      transparent — passes through without touching pending_reasoning.
         *   3. everything else (user message, tool output): forces pending
         *      reasoning to flush in-position as an empty assistant message so it
         *      stays before this item in the rendered history. */
        bool consumes_reasoning =
            (!strcmp(t, "message") && role && !strcmp(role, "assistant")) ||
            !strcmp(t, "function_call") || !strcmp(t, "custom_tool_call") ||
            !strcmp(t, "local_shell_call") || !strcmp(t, "web_search_call") ||
            !strcmp(t, "tool_search_call") || !strcmp(t, "image_generation_call");
        bool is_bookkeeping =
            !strcmp(t, "compaction") || !strcmp(t, "context_compaction");
        if (!consumes_reasoning && !is_bookkeeping && pending_reasoning.len) {
            chat_msg flush_msg = {0};
            flush_msg.role = xstrdup("assistant");
            flush_msg.content = xstrdup("");
            flush_msg.reasoning = buf_take(&pending_reasoning);
            chat_msgs_push(msgs, flush_msg);
        }
        if (!strcmp(t, "message")) {
            chat_msg msg = {0};
            msg.role = xstrdup(role ? role : "user");
            msg.content = content ? content : xstrdup("");
            content = NULL;
            /* Adopt the item's images: they belong to this message, and each
             * block's placeholder already sits in `content` at its position.
             * The placeholder offsets move with them -- one fact, one owner:
             * without them the renderer cannot tell the placeholder the parser
             * wrote from text the client typed, and the image is lost. */
            msg.images = item_images.images;
            msg.images_len = item_images.images_len;
            msg.images_cap = item_images.images_cap;
            msg.image_ph_off = item_images.image_ph_off;
            item_images.images = NULL;
            item_images.image_ph_off = NULL;
            item_images.images_len = 0;
            item_images.images_cap = 0;
            if (!strcmp(msg.role, "assistant") && pending_reasoning.len) {
                msg.reasoning = buf_take(&pending_reasoning);
            }
            chat_msgs_push(msgs, msg);
        } else if (!strcmp(t, "function_call") || !strcmp(t, "custom_tool_call")) {
            tool_call tc = {0};
            tc.id = xstrdup(call_id ? call_id : item_id ? item_id : "");
            /* function_call uses `arguments` (JSON string); custom_tool_call uses
             * `input` (free text). Treat both as the same on-wire argument blob —
             * append_dsml_arguments_from_json will fall back to a single text param
             * if the value isn't a JSON object. */
            const char *args_src = arguments ? arguments :
                                   input_str ? input_str : "{}";
            tc.arguments = xstrdup(args_src);
            if (strcmp(t, "custom_tool_call") && tool_namespace && tool_namespace[0] &&
                name && name[0])
            {
                buf qualified = {0};
                buf_puts(&qualified, tool_namespace);
                buf_puts(&qualified, name);
                tc.name = buf_take(&qualified);
            } else {
                tc.name = xstrdup(name ? name : "");
            }
            /* A Responses turn that has both message text and tool calls splits
             * them across separate output items; the chat template renders the
             * second assistant record without an `<|Assistant|>` prefix, leaving
             * the tool call bare. Merge into the previous assistant message
             * when nothing user-like / tool-output-like came between them. */
            chat_msg *last = msgs->len ? &msgs->v[msgs->len - 1] : NULL;
            if (last && !strcmp(last->role, "assistant")) {
                if (pending_reasoning.len && (!last->reasoning || !last->reasoning[0])) {
                    free(last->reasoning);
                    last->reasoning = buf_take(&pending_reasoning);
                }
                tool_calls_push(&last->calls, tc);
            } else {
                chat_msg msg = {0};
                msg.role = xstrdup("assistant");
                msg.content = xstrdup("");
                if (pending_reasoning.len) msg.reasoning = buf_take(&pending_reasoning);
                tool_calls_push(&msg.calls, tc);
                chat_msgs_push(msgs, msg);
            }
        } else if (!strcmp(t, "function_call_output") || !strcmp(t, "custom_tool_call_output")) {
            chat_msg msg = {0};
            msg.role = xstrdup("tool");
            msg.content = output ? output : xstrdup("");
            output = NULL;
            if (call_id || item_id) {
                chat_msg_add_tool_call_id(&msg, call_id ? call_id : item_id);
            }
            chat_msgs_push(msgs, msg);
        } else if (!strcmp(t, "reasoning")) {
            /* Stash so it merges into the next assistant message. summary is the
             * short-form list, content is the verbose chain. Either can be empty. */
            if (summary && summary[0]) {
                if (pending_reasoning.len) buf_putc(&pending_reasoning, '\n');
                buf_puts(&pending_reasoning, summary);
            }
            if (content && content[0]) {
                if (pending_reasoning.len) buf_putc(&pending_reasoning, '\n');
                buf_puts(&pending_reasoning, content);
            }
        } else if (!strcmp(t, "local_shell_call") || !strcmp(t, "web_search_call") ||
                   !strcmp(t, "tool_search_call") || !strcmp(t, "image_generation_call"))
        {
            /* Hosted-tool history isn't natively supported (DS4 doesn't register
             * these tools), but a Codex client may still replay them when the
             * model used them in a prior turn. Surface them as function_call
             * shaped history so the next prompt retains the action that ran. */
            tool_call tc = {0};
            tc.id = xstrdup(call_id ? call_id : item_id ? item_id : "");
            if (!strcmp(t, "tool_search_call")) {
                tc.name = xstrdup("tool_search");
            } else if (!strcmp(t, "local_shell_call")) {
                tc.name = xstrdup("local_shell");
            } else {
                tc.name = xstrdup(t);
            }
            const char *args_src = action ? action :
                                   arguments ? arguments :
                                   input_str ? input_str : "{}";
            tc.arguments = xstrdup(args_src);
            chat_msg *last = msgs->len ? &msgs->v[msgs->len - 1] : NULL;
            if (last && !strcmp(last->role, "assistant")) {
                if (pending_reasoning.len && (!last->reasoning || !last->reasoning[0])) {
                    free(last->reasoning);
                    last->reasoning = buf_take(&pending_reasoning);
                }
                tool_calls_push(&last->calls, tc);
            } else {
                chat_msg msg = {0};
                msg.role = xstrdup("assistant");
                msg.content = xstrdup("");
                if (pending_reasoning.len) msg.reasoning = buf_take(&pending_reasoning);
                tool_calls_push(&msg.calls, tc);
                chat_msgs_push(msgs, msg);
            }
        } else if (!strcmp(t, "local_shell_call_output") ||
                   !strcmp(t, "web_search_call_output") ||
                   !strcmp(t, "tool_search_output") ||
                   !strcmp(t, "tool_search_call_output") ||
                   !strcmp(t, "image_generation_call_output"))
        {
            if (!strcmp(t, "tool_search_output") && tools_json &&
                loaded_tool_schemas && orders)
            {
                const char *tools_p = tools_json;
                char *schemas = NULL;
                if (!parse_tools_value(&tools_p, &schemas, orders)) {
                    free(schemas);
                    free(type);
                    free(role);
                    free(content);
                    free(name);
                    free(tool_namespace);
                    free(call_id);
                    free(item_id);
                    free(arguments);
                    free(output);
                    free(input_str);
                    free(summary);
                    free(action);
                    free(result);
                    free(tools_json);
                    free(status_str);
                    buf_free(&pending_reasoning);
                    return false;
                }
                if (schemas && schemas[0]) {
                    if (loaded_tool_schemas->len) buf_putc(loaded_tool_schemas, '\n');
                    buf_puts(loaded_tool_schemas, schemas);
                }
                free(schemas);
            }
            chat_msg msg = {0};
            msg.role = xstrdup("tool");
            const char *body = output ? output :
                               result ? result :
                               tools_json ? tools_json : "";
            msg.content = xstrdup(body);
            if (call_id || item_id) {
                chat_msg_add_tool_call_id(&msg, call_id ? call_id : item_id);
            }
            chat_msgs_push(msgs, msg);
        } else if (!is_bookkeeping) {
            /* Anything we don't have an explicit branch for would silently
             * drop replay context. Fail the parse instead so the client sees
             * the limitation rather than ending up with stale generation
             * built on an incomplete history. Only compaction/context_compaction
             * (true Codex bookkeeping) are allowed to pass through silently. */
            free(type);
            free(role);
            free(content);
            free(name);
            free(tool_namespace);
            free(call_id);
            free(item_id);
            free(arguments);
            free(output);
            free(input_str);
            free(summary);
            free(action);
            free(result);
            free(tools_json);
            free(status_str);
            chat_msg_clear_images(&item_images);
            buf_free(&pending_reasoning);
            return false;
        }

        free(type);
        free(role);
        free(content);
        free(name);
        free(tool_namespace);
        free(call_id);
        free(item_id);
        free(arguments);
        free(output);
        free(input_str);
        free(summary);
        free(action);
        free(result);
        free(tools_json);
        free(status_str);
        /* Anything the item collected but did not adopt (a non-message item
         * that carried a content array, or a message that failed to push). */
        chat_msg_clear_images(&item_images);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto fail;
    (*p)++;
    /* Trailing reasoning with no following message/tool item: attach it to an
     * empty assistant message so the next turn still renders a <think>...</think>
     * block. Dropping it loses model state when a previous response ended with
     * a reasoning-only incomplete turn and the client replays the history. */
    if (pending_reasoning.len) {
        chat_msg msg = {0};
        msg.role = xstrdup("assistant");
        msg.content = xstrdup("");
        msg.reasoning = buf_take(&pending_reasoning);
        chat_msgs_push(msgs, msg);
    }
    buf_free(&pending_reasoning);
    return true;
fail:
    chat_msg_clear_images(&item_images);
    buf_free(&pending_reasoning);
    return false;
}



/* Responses API has `reasoning: {"effort": "...", "summary": "..."}`. effort
 * controls thinking depth -- kept as sent, for the family (the
 * "reasoning.effort" control; null is the same as omitting it); summary mode
 * (auto/concise/detailed) controls whether the wire emits summary deltas at
 * all -- per the spec, no reasoning summary is surfaced unless the client opts
 * in. */
static bool parse_responses_reasoning(const char **p, chat_conversation *c, bool *summary_opted_in) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p != '{') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') {
            free(key);
            return false;
        }
        (*p)++;
        bool ok = true;
        if (!strcmp(key, "effort")) {
            json_ws(p);
            if (!json_lit(p, "null")) ok = take_control(p, c, "reasoning.effort");
        } else if (!strcmp(key, "summary")) {
            json_ws(p);
            if (json_lit(p, "null")) {
                /* explicit null disables summary */
            } else if (**p == '"') {
                char *mode = NULL;
                ok = json_string(p, &mode);
                if (ok && (!strcmp(mode, "auto") || !strcmp(mode, "concise") || !strcmp(mode, "detailed")))
                    *summary_opted_in = true;
                free(mode);
            } else {
                ok = json_skip_value(p);
            }
        } else {
            ok = json_skip_value(p);
        }
        free(key);
        if (!ok) return false;
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}



bool parse_chat_conversation_responses(const char *body, chat_conversation *c, request *r, char *err,
                                       size_t errlen) {
    r->api = API_RESPONSES;
    if (err && errlen) err[0] = '\0';
    const char *p = body;
    bool got_input = false;
    int skr = 0;
    char *instructions = NULL;

    json_ws(&p);
    if (*p != '{') return false;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto bad;
        }
        p++;
        bool ok;
        ok = true;
        if (!strcmp(key, "input")) {
            chat_msgs_free(&c->msgs);
            json_ws(&p);
            /* Codex CLI always sends `input` as an array; tolerate bare strings
             * for parity with other Responses-API callers. */
            if (*p == '"') {
                char *plain = NULL;
                ok = json_string(&p, &plain);
                if (ok) {
                    chat_msg msg = {0};
                    msg.role = xstrdup("user");
                    msg.content = plain;
                    chat_msgs_push(&c->msgs, msg);
                }
            } else {
                ok = parse_responses_input(&p, &c->msgs, &c->loaded_tool_schemas, &r->tool_orders, err, errlen);
            }
            got_input = true;
        } else if (!strcmp(key, "instructions")) {
            free(instructions);
            instructions = NULL;
            json_ws(&p);
            if (json_lit(&p, "null")) instructions = xstrdup("");
            else ok = json_string(&p, &instructions);
        } else if (!strcmp(key, "tools")) {
            ok = take_tools(&p, c, r, false, err, errlen);
        } else if (!strcmp(key, "tool_choice")) {
            json_ws(&p);
            if (*p == '"') {
                char *choice = NULL;
                ok = json_string(&p, &choice);
                /* "none" (disable tools) and "auto" (model decides).  "required"
                 * and explicit function targets need constrained decoding we
                 * don't implement -- refused so clients see the limitation
                 * instead of silently downgrading to auto. */
                if (ok && !strcmp(choice, "none")) {
                    c->tool_choice = CHAT_TOOL_CHOICE_NONE;
                } else if (ok && strcmp(choice, "auto") != 0) {
                    snprintf(err, errlen, "tool_choice=%s not supported", choice);
                    ok = false;
                }
                free(choice);
            } else if (*p == '{') {
                snprintf(err, errlen, "forced tool_choice not supported");
                ok = false;
            } else {
                ok = json_skip_value(&p);
            }
        } else if (!strcmp(key, "model")) {
            free(r->model);
            ok = json_string(&p, &r->model);
            r->model_from_request = true;
        } else if (!strcmp(key, "max_output_tokens") || !strcmp(key, "max_tokens")) {
            ok = json_int(&p, &r->max_tokens);
        } else if ((skr = parse_sampling_key(key, &p, r)) != 0) {
            ok = skr > 0;
        } else if (!strcmp(key, "stream")) {
            ok = json_bool(&p, &r->stream);
        } else if (!strcmp(key, "reasoning")) {
            ok = parse_responses_reasoning(&p, c, &r->reasoning_summary_emit);
        } else if (!strcmp(key, "previous_response_id") || !strcmp(key, "conversation")) {
            /* Official Responses state can be durable:
             *   previous_response_id chains to a stored prior response, and
             *   conversation points at a persistent Conversations object.
             *
             * DS4 does not yet implement that durable store.  The supported
             * modes are either (a) a live in-memory continuation checked by
             * visible transcript / tool call ids, or (b) stateless replay of
             * the full input items.  Accepting a non-null durable reference
             * without loading the referenced items would silently truncate the
             * prompt, so reject it explicitly. */
            json_ws(&p);
            if (!json_lit(&p, "null")) {
                snprintf(err, errlen, "%s is not supported; replay full input instead", key);
                ok = false;
            }
        } else {
            ok = json_skip_value(&p);
        }
        free(key);
        if (!ok) goto bad;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (*p != '}') goto bad;
    if (!got_input) {
        snprintf(err, errlen, "missing input");
        goto bad;
    }
    /* instructions in the Responses API replaces any system message -- for Codex
     * it carries the full agent system prompt.  It goes first, as the system
     * FIELD, so render produces a standard system+chat layout. */
    if (instructions && instructions[0]) {
        chat_msg msg = {0};
        msg.role = xstrdup("system");
        msg.content = instructions;
        msg.system_field = true;
        instructions = NULL;
        chat_msgs_push(&c->msgs, msg);
        chat_msg tmp = c->msgs.v[c->msgs.len - 1];
        for (int i = c->msgs.len - 1; i > 0; i--) c->msgs.v[i] = c->msgs.v[i - 1];
        c->msgs.v[0] = tmp;
    }
    free(instructions);
    return true;
bad:
    free(instructions);
    return false;
}

bool parse_responses_request(pulsar_engine *e, server *s, const char *body, int def_tokens,
                                    request *r, char *err, size_t errlen) {
    request_init(r, REQ_CHAT, def_tokens);
    return parse_and_render(e, s, body, r, err, errlen, parse_chat_conversation_responses);
}



static bool parse_prompt(const char **p, char **out) {
    /* Build into a local and publish to *out only on success, so every
     * failure return leaves *out == NULL (the double-free contract, see
     * json_string_n) rather than a partially-parsed heap value. */
    *out = NULL;
    json_ws(p);
    if (**p == '"') return json_string(p, out);
    if (**p != '[') {
        if (!json_skip_value(p)) return false;
        *out = xstrdup("");
        return true;
    }
    (*p)++;
    json_ws(p);
    char *s = NULL;
    if (**p == '"') {
        if (!json_string(p, &s)) return false;
    } else {
        s = xstrdup("");
        if (**p && **p != ']' && !json_skip_value(p)) { free(s); return false; }
    }
    while (**p && **p != ']') {
        json_ws(p);
        if (**p == ',') {
            (*p)++;
            if (!json_skip_value(p)) { free(s); return false; }
        } else {
            break;
        }
    }
    if (**p != ']') { free(s); return false; }
    (*p)++;
    *out = s;
    return true;
}



bool parse_completion_request(pulsar_engine *e, const char *body, int def_tokens,
                                     request *r, char *err, size_t errlen) {
    request_init(r, REQ_COMPLETION, def_tokens);
    r->family = server_family_for_engine(e);
    const char *p = body;
    char *prompt = NULL;
    bool got_thinking = false;
    bool thinking_enabled = true;
    char why[160];
    int skr = 0;
    /* The default effort is the loaded family's: V4.1 defaults to high (the
     * reference's default); the V4 (0731) encoder's default is low, which
     * renders no effort line at all (L239). */
    pulsar_think_mode reasoning_effort = pulsar_engine_think_default(e);

    json_ws(&p);
    if (*p != '{') goto bad;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto bad;
        }
        p++;
        if (!strcmp(key, "prompt")) {
            free(prompt);
            if (!parse_prompt(&p, &prompt)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "model")) {
            free(r->model);
            if (!json_string(&p, &r->model)) {
                free(key);
                goto bad;
            }
            r->model_from_request = true;
        } else if (!strcmp(key, "max_tokens")) {
            if (!json_int(&p, &r->max_tokens)) {
                free(key);
                goto bad;
            }
        } else if ((skr = parse_sampling_key(key, &p, r)) != 0) {
            if (skr < 0) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "stream")) {
            if (!json_bool(&p, &r->stream)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "stream_options")) {
            if (!parse_stream_options(&p, &r->stream_include_usage)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "thinking")) {
            if (!parse_thinking_control_value(&p, &thinking_enabled)) {
                free(key);
                goto bad;
            }
            got_thinking = true;
        } else if (!strcmp(key, "reasoning_effort")) {
            if (!parse_reasoning_effort_value(&p, &reasoning_effort)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "think") || !strcmp(key, "enable_thinking")) {
            /* enable_thinking is the Qwen/vLLM spelling; accept it as a bool
             * alias for our existing `think` field. Both remain additive to the
             * Anthropic-style `thinking` object handled above. */
            if (!json_bool(&p, &thinking_enabled)) {
                free(key);
                goto bad;
            }
            got_thinking = true;
        } else if (!strcmp(key, "stop")) {
            if (!parse_stop(&p, &r->stops)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "logprobs") || !strcmp(key, "top_logprobs")) {
            /* This surface has NO logprobs path: the ledger append and the
             * response field are chat/Responses only (parse_chat_request owns
             * the key there).  The catch-all below used to SKIP it, so a client
             * that asked for distributions got HTTP 200, no payload, and
             * speculation still enabled -- a silent fail-open.  Refuse loudly;
             * an explicit null is "not set" (the OpenAI SDKs send it on both
             * logprobs fields) and stays accepted, as it is on the chat
             * surface. */
            json_ws(&p);
            if (!json_lit(&p, "null")) {
                snprintf(err, errlen,
                         "%s is not supported on /v1/completions; use "
                         "/v1/chat/completions with logprobs:true", key);
                free(key);
                free(prompt);
                request_free(r);
                return false;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto bad;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (*p != '}') goto bad;
    if (!prompt) {
        snprintf(err, errlen, "missing prompt");
        request_free(r);
        return false;
    }
    if (pulsar_engine_chat_format(e) == PULSAR_CHAT_QWEN) {
        /* L251: a Qwen completion is a RAW continuation (vLLM's /v1/completions): no template and
         * no thinking block; the whole prompt is client text, so no added token matches in it */
        r->think_mode = think_mode_from_enabled(false, reasoning_effort);
        free(r->prompt_spans);
        r->prompt_n_spans = 0;
        r->prompt_spans = NULL;
        r->prompt_text = prompt;
        prompt = NULL;
        const size_t plen = strlen(r->prompt_text);
        if (plen) {
            r->prompt_spans = (pulsar_text_span *)malloc(sizeof(pulsar_text_span));
            if (!r->prompt_spans) { if (err && errlen) snprintf(err, errlen, "out of memory"); return false; }
            r->prompt_spans[0].lo = 0;
            r->prompt_spans[0].hi = (uint32_t)plen;
            r->prompt_n_spans = 1;
        }
        pulsar_tokenize_rendered_chat_spans(e, r->prompt_text, r->prompt_spans, r->prompt_n_spans, &r->prompt);
        return true;
    }
    if (!got_thinking && model_alias_disables_thinking(r->model)) thinking_enabled = false;
    if (!got_thinking && model_alias_enables_thinking(r->model)) thinking_enabled = true;
    if (thinking_enabled && !pulsar_engine_think_mode_supported(e, reasoning_effort, why, sizeof why)) {
        if (err && errlen) snprintf(err, errlen, "reasoning_effort: %s", why);
        goto bad;
    }
    r->think_mode = think_mode_from_enabled(thinking_enabled, reasoning_effort);
    free(r->prompt_spans);
    r->prompt_spans = NULL;
    r->prompt_n_spans = 0;
    r->prompt_text = render_completion_prompt_text_spans(prompt, r->think_mode, r->family->v41,
                                                         &r->prompt_spans, &r->prompt_n_spans);
    pulsar_tokenize_rendered_chat_spans(e, r->prompt_text, r->prompt_spans,
                                        r->prompt_n_spans, &r->prompt);
    free(prompt);
    return true;
bad:
    free(prompt);
    snprintf(err, errlen, "invalid JSON request");
    request_free(r);
    return false;
}


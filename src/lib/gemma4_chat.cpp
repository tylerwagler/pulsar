/** @file
 *  The Gemma 4 renderer (the checkpoint's chat_template.jinja). */
/* gemma4_chat.cpp -- the Gemma 4 renderer.  See gemma4_chat.h.  Every literal below is the checkpoint's
 * chat_template.jinja (Gemma 4 26B-A4B, sha256 in tests/test-vectors/gemma4/meta.json), in the template's order; the
 * macros keep their names (format_argument, format_parameters, format_function_declaration, strip_thinking,
 * format_tool_response_block). */
#include "gemma4_chat.h"

#include <string.h>

#include <algorithm>

#include "chat_refuse.h"
#include "pyjson.h"

namespace {

/// The template's string quote (an added token).
const char kQ[] = "<|\"|>";

/// @param v a value, or NULL (Jinja's Undefined)
/// @param k a key
/// @return v[k] when v is a mapping holding k, else NULL
const pyjson_value *get(const pyjson_value *v, const char *k) {
    return v && v->kind == pyjson_value::OBJ ? v->get(k) : NULL;
}
/// @param v a value, or NULL
/// @return Jinja's truthiness (Undefined is false)
bool truthy(const pyjson_value *v) { return v && pyjson_truthy(*v); }

/// @param s a string
/// @return Python's str.upper() of an ASCII string (the template upper-cases schema type names)
std::string upper(std::string s) {
    for (char &c : s)
        if (c >= 'a' && c <= 'z') c = (char)(c - 32);
    return s;
}
/// @param v a value, or NULL
/// @return the template's `v | upper`: '' for Undefined, else str(v).upper()
std::string upper_of(const pyjson_value *v) { return v ? upper(py_str(*v)) : std::string(); }

/// @param s a key
/// @return its ASCII-lower-cased copy (Jinja's dictsort compares keys case-insensitively)
std::string lower(std::string s) {
    for (char &c : s)
        if (c >= 'A' && c <= 'Z') c = (char)(c + 32);
    return s;
}
/* Jinja's `| dictsort`: the mapping's items by key, case-insensitively, stable (Python's sorted) */
/// @param o a mapping
/// @return its items in dictsort order
std::vector<const std::pair<std::string, pyjson_value> *> dictsort(const pyjson_value &o) {
    std::vector<const std::pair<std::string, pyjson_value> *> v;
    for (const auto &kv : o.o) v.push_back(&kv);
    std::stable_sort(v.begin(), v.end(), [](const auto *a, const auto *b) { return lower(a->first) < lower(b->first); });
    return v;
}

/* format_argument: null, a quoted string, true / false, {k:v,...} in dictsort order (keys quoted when escape_keys),
 * [..], a number as Python writes it.  Strings and keys are the client's; the scalars are the template's spelling. */
/// @param p           the output
/// @param v           the value
/// @param escape_keys quote a mapping's keys
void format_argument(chat_piece &p, const pyjson_value &v, bool escape_keys) {
    switch (v.kind) {
    case pyjson_value::NUL: p.lit("null"); return;
    case pyjson_value::STR: p.lit(kQ); p.client(v.s); p.lit(kQ); return;
    case pyjson_value::BOOL: p.lit(v.b ? "true" : "false"); return;
    case pyjson_value::OBJ: {
        p.lit("{");
        bool first = true;
        for (const auto *kv : dictsort(v)) {
            if (!first) p.lit(",");
            first = false;
            if (escape_keys) { p.lit(kQ); p.client(kv->first); p.lit(kQ); }
            else p.client(kv->first);
            p.lit(":");
            format_argument(p, kv->second, escape_keys);
        }
        p.lit("}");
        return;
    }
    case pyjson_value::ARR:
        p.lit("[");
        for (size_t i = 0; i < v.a.size(); i++) {
            format_argument(p, v.a[i], escape_keys);
            if (i + 1 < v.a.size()) p.lit(",");
        }
        p.lit("]");
        return;
    default: p.lit(py_str(v)); return;   /* a number: str() */
    }
}

/* a `required` list: <|"|>item<|"|> joined by ',' (a string iterates its characters) */
/// @param p  the output
/// @param rq the list
void quoted_items(chat_piece &p, const pyjson_value &rq) {
    std::vector<std::string> items;
    if (rq.kind == pyjson_value::ARR)
        for (const pyjson_value &x : rq.a) items.push_back(py_str(x));
    else if (rq.kind == pyjson_value::STR)
        for (size_t i = 0; i < rq.s.size();) {
            size_t k = 1;
            while (i + k < rq.s.size() && ((unsigned char)rq.s[i + k] & 0xC0) == 0x80) k++;
            items.push_back(rq.s.substr(i, k));
            i += k;
        }
    for (size_t i = 0; i < items.size(); i++) {
        p.lit(kQ);
        p.client(items[i]);
        p.lit(kQ);
        if (i + 1 < items.size()) p.lit(",");
    }
}

/// The keys format_parameters skips with filter_keys.
const char *const kStandardKeys[] = {"description", "type", "properties", "required", "nullable"};

/* format_parameters(properties, required, filter_keys): every property, in dictsort order, as KEY:{...} */
/// @param p           the output
/// @param properties  the mapping
/// @param filter_keys skip the standard schema keys
void format_parameters(chat_piece &p, const pyjson_value &properties, bool filter_keys) {
    bool found_first = false;
    for (const auto *kv : dictsort(properties)) {
        const std::string &key = kv->first;
        const pyjson_value *value = &kv->second;
        if (filter_keys) {
            bool standard = false;
            for (const char *s : kStandardKeys) standard |= key == s;
            if (standard) continue;
        }
        bool add_comma = false;
        auto comma = [&] {
            if (add_comma) p.lit(",");
            else add_comma = true;
        };
        if (found_first) p.lit(",");
        found_first = true;
        p.client(key);
        p.lit(":{");
        const pyjson_value *desc = get(value, "description");
        if (truthy(desc)) {
            p.lit("description:");
            p.lit(kQ);
            p.client(py_str(*desc));
            p.lit(kQ);
            add_comma = true;
        }
        const std::string type = upper_of(get(value, "type"));
        if (type == "STRING") {
            const pyjson_value *en = get(value, "enum");
            if (truthy(en)) {
                comma();
                p.lit("enum:");
                format_argument(p, *en, true);
            }
        } else if (type == "ARRAY") {
            const pyjson_value *items = get(value, "items");
            if (items && items->kind == pyjson_value::OBJ && !items->o.empty()) {
                comma();
                p.lit("items:{");
                bool ff = false;
                for (const auto *ikv : dictsort(*items)) {
                    const pyjson_value &iv = ikv->second;
                    if (iv.kind == pyjson_value::NUL) continue;
                    if (ff) p.lit(",");
                    ff = true;
                    if (ikv->first == "properties") {
                        p.lit("properties:{");
                        if (iv.kind == pyjson_value::OBJ) format_parameters(p, iv, false);
                        p.lit("}");
                    } else if (ikv->first == "required") {
                        p.lit("required:[");
                        quoted_items(p, iv);
                        p.lit("]");
                    } else if (ikv->first == "type") {
                        p.lit("type:");
                        if (iv.kind == pyjson_value::STR) {
                            p.lit(kQ);
                            p.client(upper(iv.s));
                            p.lit(kQ);
                        } else {   /* item_value | map('upper') | list, then format_argument */
                            p.lit("[");
                            for (size_t i = 0; i < iv.a.size(); i++) {
                                p.lit(kQ);
                                p.client(upper(py_str(iv.a[i])));
                                p.lit(kQ);
                                if (i + 1 < iv.a.size()) p.lit(",");
                            }
                            p.lit("]");
                        }
                    } else {
                        p.client(ikv->first);
                        p.lit(":");
                        format_argument(p, iv, true);
                    }
                }
                p.lit("}");
            }
        }
        if (truthy(get(value, "nullable"))) {
            comma();
            p.lit("nullable:true");
        }
        if (type == "OBJECT") {
            const pyjson_value *props = get(value, "properties");
            if (props && props->kind == pyjson_value::OBJ) {
                comma();
                p.lit("properties:{");
                format_parameters(p, *props, false);
                p.lit("}");
            } else if (value->kind == pyjson_value::OBJ) {
                comma();
                p.lit("properties:{");
                format_parameters(p, *value, true);
                p.lit("}");
            }
            const pyjson_value *rq = get(value, "required");
            if (truthy(rq)) {
                comma();
                p.lit("required:[");
                quoted_items(p, *rq);
                p.lit("]");
            }
        }
        comma();
        p.lit("type:");
        p.lit(kQ);
        p.client(type);
        p.lit(kQ);
        p.lit("}");
    }
}

/* format_function_declaration(tool): declaration:NAME{description:..,parameters:{..},response:{..}} */
/// @param p      the output
/// @param tool   one entry of the tools array
/// @param err    the error buffer
/// @param errlen its size
/// @return false (refused) for a tool with no function object, or a parameters.properties that is not a mapping
bool format_function_declaration(chat_piece &p, const pyjson_value &tool, char *err, size_t errlen) {
    const pyjson_value *fn = get(&tool, "function");
    if (!fn || fn->kind != pyjson_value::OBJ)
        return chat_refuse(err, errlen, "tool_function_missing", "a tool has no \"function\" object");
    const pyjson_value *name = get(fn, "name"), *desc = get(fn, "description");
    p.lit("declaration:");
    if (name) p.client(py_str(*name));
    p.lit("{description:");
    p.lit(kQ);
    if (desc) p.client(py_str(*desc));
    p.lit(kQ);
    const pyjson_value *params = get(fn, "parameters");
    if (truthy(params)) {
        p.lit(",parameters:{");
        const pyjson_value *props = get(params, "properties");
        if (truthy(props)) {
            if (props->kind != pyjson_value::OBJ)
                return chat_refuse(err, errlen, "tool_schema_invalid", "parameters.properties is not a mapping");
            p.lit("properties:{");
            format_parameters(p, *props, false);
            p.lit("},");
        }
        const pyjson_value *rq = get(params, "required");
        if (truthy(rq)) {
            p.lit("required:[");
            quoted_items(p, *rq);
            p.lit("],");
        }
        const pyjson_value *type = get(params, "type");
        if (truthy(type)) {
            p.lit("type:");
            p.lit(kQ);
            p.client(upper_of(type));
            p.lit(kQ);
            p.lit("}");
        }
    }
    if (fn->get("response")) {
        const pyjson_value *rd = fn->get("response");
        p.lit(",response:{");
        const pyjson_value *rdesc = get(rd, "description");
        if (truthy(rdesc)) {
            p.lit("description:");
            p.lit(kQ);
            p.client(py_str(*rdesc));
            p.lit(kQ);
            p.lit(",");
        }
        if (upper_of(get(rd, "type")) == "OBJECT") {
            p.lit("type:");
            p.lit(kQ);
            p.client("OBJECT");
            p.lit(kQ);
            p.lit("}");
        }
    }
    p.lit("}");
    return true;
}

/* strip_thinking(text): every part between "<channel|>"s keeps only what precedes its first "<|channel>"; trimmed */
/// @param c the content
/// @return the content without its thinking channels, trimmed
chat_piece strip_thinking(const chat_piece &c) {
    static const std::string kClose = "<channel|>", kOpen = "<|channel>";
    chat_piece out;
    size_t lo = 0;
    for (;;) {
        const size_t hi = c.s.find(kClose, lo);
        const size_t end = hi == std::string::npos ? c.s.size() : hi;
        const size_t open = c.s.find(kOpen, lo);
        out.add(c.sub(lo, open != std::string::npos && open < end ? open : end));
        if (hi == std::string::npos) break;
        lo = hi + kClose.size();
    }
    return out.trimmed();
}

/// @param s a client string, or NULL
/// @return it as a piece, all of it client data
chat_piece client_piece(const char *s) {
    chat_piece p;
    if (s) p.client(s);
    return p;
}

/* format_tool_response_block(name, response) for the API's string (or null) body */
/// @param w    the output
/// @param name the call's name (client), or NULL for the literal "unknown"
/// @param body the tool message's content, or NULL (renders null)
void tool_response_block(chat_writer &w, const char *name, const char *body) {
    w.lit("<|tool_response>response:");
    if (name) w.client(name);
    else w.lit("unknown");
    w.lit("{value:");
    if (body) {
        w.lit(kQ);
        w.client(body);
        w.lit(kQ);
    } else {
        w.lit("null");
    }
    w.lit("}<tool_response|>");
}

/// @param a a nullable string
/// @param b a nullable string
/// @return Python's a == b over str-or-None
bool same(const char *a, const char *b) { return (!a && !b) || (a && b && !strcmp(a, b)); }

/// The message-loop's state the generation prompt reads (ns.prev_message_type).
enum prev_type { PREV_NONE, PREV_THINK, PREV_TOOL, PREV_TOOL_CALL, PREV_TOOL_RESPONSE };

/* The template's message loop over `msgs` (the loop_messages: the first system message already taken), then the
 * generation prompt.  `prev` is the state the setup left. */
/// @param w      the output
/// @param msgs   the messages
/// @param n      their count
/// @param thinking enable_thinking
/// @param gen    add_generation_prompt
/// @param prev   ns.prev_message_type after the setup
/// @param err    the error buffer
/// @param errlen its size
/// @return false on refusal
bool write_messages(chat_writer &w, const gemma4_msg_in *msgs, int n, bool thinking, bool gen, prev_type prev,
                    char *err, size_t errlen) {
    int last_user = -1;
    for (int i = 0; i < n; i++)
        if (msgs[i].role && !strcmp(msgs[i].role, "user")) last_user = i;
    const char *prev_non_tool_role = NULL;
    for (int i = 0; i < n; i++) {
        const gemma4_msg_in &m = msgs[i];
        const char *raw_role = m.role ? m.role : "user";
        if (!strcmp(raw_role, "tool")) continue;
        prev = PREV_NONE;
        const bool model = !strcmp(raw_role, "assistant");
        const char *role = model ? "model" : raw_role;
        const bool continue_same = model && prev_non_tool_role && !strcmp(prev_non_tool_role, "assistant");
        if (!continue_same) {
            w.lit("<|turn>");
            w.lit(role);
            w.lit("\n");
        }
        if (m.reasoning && m.reasoning[0] && i > last_user) {
            w.lit("<|channel>thought\n");
            w.client(m.reasoning);
            w.lit("\n<channel|>");
        }
        if (m.n_calls > 0) {
            for (int c = 0; c < m.n_calls; c++) {
                const gemma4_tool_call_in &tc = m.calls[c];
                if (!tc.name || !tc.name[0])
                    return chat_refuse(err, errlen, "tool_name_missing", "message %d call %d has no name", i, c);
                if (!tc.arguments || !tc.arguments[0])
                    return chat_refuse(err, errlen, "tool_arguments_missing", "message %d call %d has no arguments", i, c);
                pyjson_value args;
                char jerr[200];
                if (!pyjson_parse(tc.arguments, strlen(tc.arguments), &args, jerr, sizeof jerr))
                    return chat_refuse(err, errlen, "tool_arguments_not_json", "message %d call %d: %s", i, c, jerr);
                if (args.kind != pyjson_value::OBJ && args.kind != pyjson_value::NUL)
                    return chat_refuse(err, errlen, "tool_arguments_not_object", "message %d call %d", i, c);
                chat_piece p;
                p.lit("<|tool_call>call:");
                p.client(tc.name);
                p.lit("{");
                if (args.kind == pyjson_value::OBJ) {
                    bool first = true;
                    for (const auto *kv : dictsort(args)) {
                        if (!first) p.lit(",");
                        first = false;
                        p.client(kv->first);
                        p.lit(":");
                        format_argument(p, kv->second, false);
                    }
                }
                p.lit("}<tool_call|>");
                w.add(p);
            }
            prev = PREV_TOOL_CALL;
        }
        bool tr_out = false;
        if (m.n_calls > 0)   /* the tool messages right after: each in this turn */
            for (int k = i + 1; k < n && msgs[k].role && !strcmp(msgs[k].role, "tool"); k++) {
                const gemma4_msg_in &f = msgs[k];
                const char *name = f.name && f.name[0] ? f.name : NULL;
                for (int c = 0; c < m.n_calls; c++)
                    if (same(m.calls[c].id, f.tool_call_id)) name = m.calls[c].name;
                tool_response_block(w, name, f.content);
                tr_out = true;
                prev = PREV_TOOL_RESPONSE;
            }
        const chat_piece content = model ? strip_thinking(client_piece(m.content)) : client_piece(m.content).trimmed();
        w.add(content);
        const bool has_content = !content.trimmed().s.empty();
        const char *next_role = NULL;
        for (int j = i + 1; j < n && !next_role; j++)
            if (!msgs[j].role || strcmp(msgs[j].role, "tool")) next_role = msgs[j].role ? msgs[j].role : "user";
        const bool continues = model && next_role && !strcmp(next_role, "assistant") && (m.n_calls == 0 || tr_out);
        if (prev == PREV_TOOL_CALL && !tr_out) w.lit("<|tool_response>");
        else if (continues) {
        } else if (!(tr_out && !has_content && !next_role)) w.lit("<turn|>\n");
        prev_non_tool_role = raw_role;
    }
    if (gen) {
        if (prev != PREV_TOOL_RESPONSE && prev != PREV_TOOL_CALL) {
            w.lit("<|turn>model\n");
            if (!thinking) w.lit("<|channel>thought\n<channel|>");
        } else if (prev == PREV_TOOL_RESPONSE && thinking) {
            w.lit("<|channel>thought\n");
        }
    }
    return true;
}

/* The setup: <bos>, the system turn (thinking, the first system / developer message, the tools); returns the state
 * and how many leading messages it took. */
/// @param w        the output
/// @param msgs     the messages
/// @param n        their count
/// @param tools    the tools array, or NULL
/// @param thinking enable_thinking
/// @param prev     receives ns.prev_message_type
/// @param taken    receives 1 when the first message was the system turn's, else 0
/// @param err      the error buffer
/// @param errlen   its size
/// @return false on refusal
bool write_setup(chat_writer &w, const gemma4_msg_in *msgs, int n, const pyjson_value *tools, bool thinking,
                 prev_type *prev, int *taken, char *err, size_t errlen) {
    *prev = PREV_NONE;
    *taken = 0;
    w.lit("<bos>");
    const bool sys_first = n > 0 && msgs[0].role && (!strcmp(msgs[0].role, "system") || !strcmp(msgs[0].role, "developer"));
    const bool has_tools = tools && !tools->a.empty();
    if (!(thinking || has_tools || sys_first)) return true;
    w.lit("<|turn>system\n");
    if (thinking) {
        w.lit("<|think|>\n");
        *prev = PREV_THINK;
    }
    if (sys_first) {
        w.add(client_piece(msgs[0].content).trimmed());
        *taken = 1;
    }
    if (has_tools) {
        for (const pyjson_value &t : tools->a) {
            chat_piece d;
            if (!format_function_declaration(d, t, err, errlen)) return false;
            w.lit("<|tool>");
            w.add(d.trimmed());
            w.lit("<tool|>");
        }
        *prev = PREV_TOOL;
    }
    w.lit("<turn|>\n");
    return true;
}

}  // namespace

bool gemma4_chat_render(const gemma4_render_in &in, gemma4_render_out *out, char *err, size_t errlen) {
    out->text.clear();
    out->spans.clear();
    if (in.n_msgs <= 0 || !in.msgs) return chat_refuse(err, errlen, "no_messages", "the conversation is empty");
    pyjson_value tools;
    const bool have_tools = in.tools_json && in.tools_json[0];
    if (have_tools) {
        char jerr[200];
        if (!pyjson_parse(in.tools_json, strlen(in.tools_json), &tools, jerr, sizeof jerr) ||
            tools.kind != pyjson_value::ARR)
            return chat_refuse(err, errlen, "tools_not_array", "the tools are not a JSON array");
    }
    chat_writer w{out};
    prev_type prev;
    int taken;
    if (!write_setup(w, in.msgs, in.n_msgs, have_tools ? &tools : NULL, in.thinking, &prev, &taken, err, errlen) ||
        !write_messages(w, in.msgs + taken, in.n_msgs - taken, in.thinking, in.add_generation_prompt, prev, err, errlen))
        return false;
    return chat_render_finish(out, err, errlen);
}

bool gemma4_chat_render_head(const char *system, bool thinking, gemma4_render_out *out, char *err, size_t errlen) {
    out->text.clear();
    out->spans.clear();
    gemma4_msg_in sys = {};
    sys.role = "system";
    sys.content = system;
    const bool has_system = system && system[0];
    chat_writer w{out};
    prev_type prev;
    int taken;
    if (!write_setup(w, has_system ? &sys : NULL, has_system ? 1 : 0, NULL, thinking, &prev, &taken, err, errlen))
        return false;
    return chat_render_finish(out, err, errlen);
}

bool gemma4_chat_render_turn(const gemma4_msg_in *msgs, int n, bool thinking, bool generation_prompt,
                             gemma4_render_out *out, char *err, size_t errlen) {
    out->text.clear();
    out->spans.clear();
    chat_writer w{out};
    if (!write_messages(w, msgs, n, thinking, generation_prompt, PREV_NONE, err, errlen)) return false;
    return chat_render_finish(out, err, errlen);
}

const char *gemma4_chat_assistant_close(void) { return "<turn|>\n"; }

/** @file
 *  L337: Anthropic context editing (context_management.edits) -- the request's edits read, planned on the
 *  parsed conversation and applied before the loaded family renders it.
 *
 *  The edits are server-side: the client keeps sending its full history, and the API removes old tool results
 *  (clear_tool_uses_20250919) and the reasoning of older assistant turns (clear_thinking_20251015) from the
 *  prompt it builds.  Here that prompt is the family's render, so an edit is applied to the chat_conversation
 *  and the render that follows IS the edited prompt: the prefix match against a bank, the disk chain lookup and
 *  the token count all key on the edited bytes, with no second notion of "the prompt".  A cleared result renders
 *  as a short notice in its place, so the conversation's structure (every tool_use still answered) is unchanged.
 *
 *  What an edit removed is measured, never estimated: the request is rendered without it and with it, and the
 *  difference in prompt tokens is its cleared_input_tokens (what clear_at_least reads; an input_tokens trigger reads
 *  the unedited render's size).  A request whose edits clear nothing renders once, exactly as it would without
 *  context_management.
 *
 *  The one thing the edited prompt cannot share with the unedited one is a slot's live tool binding: that KV is the
 *  history as the model saw it.  An edited request carries no live continuation, and the turn it generates records
 *  none (server_jobs.cpp), so every resume of either goes through the prefix match on the bytes it renders.
 */
#include "pulsar_server_internal.h"

#include <cmath>
#include <string>
#include <vector>

/// The text a cleared tool result renders as (the API documents "placeholder text", not its spelling).
static const char k_ctx_cleared_result[] = "[tool result cleared to save context]";

/* ---- parse ---------------------------------------------------------------------------------------------------- */

/// Walk one JSON object's members: `f(key, &p)` reads each value (and must advance past it).
/// @param p  the cursor, at the object (advanced past it)
/// @param f  the member reader: false stops the walk with false
/// @return false on malformed JSON or when `f` refused
template <class F>
static bool ctx_each_member(const char **p, F f) {
    json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        bool ok = **p == ':';
        if (ok) {
            (*p)++;
            ok = f(key, p);
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

/// Read a {"type": <want...>, "value": N} object (a trigger, keep or clear_at_least).
/// @param p        the cursor, at the object (advanced past it)
/// @param where    the member's path, for the error
/// @param types    the accepted type names, NULL-terminated
/// @param type_at  receives the index of the type named
/// @param value    receives the value
/// @param min      the least value accepted
/// @param err      receives the client error on false
/// @param errlen   size of `err`
/// @return false on a malformed object, an unknown type or a value under `min`
static bool ctx_typed_value(const char **p, const char *where, const char *const *types, int *type_at, long *value,
                            long min, char *err, size_t errlen) {
    int t = -1;
    double v = -1.0;
    bool have_v = false;
    const bool shape = ctx_each_member(p, [&](const char *k, const char **q) {
        if (!strcmp(k, "type")) {
            char *name = NULL;
            if (!json_string(q, &name)) return false;
            for (int i = 0; types[i]; i++)
                if (!strcmp(name, types[i])) t = i;
            if (t < 0) snprintf(err, errlen, "%s.type: Input should be '%s'%s", where, types[0], types[1] ? ", ..." : "");
            free(name);
            return t >= 0;
        }
        if (!strcmp(k, "value")) return have_v = json_number(q, &v);
        snprintf(err, errlen, "%s.%.40s: Extra inputs are not permitted", where, k);
        return false;
    });
    if (!shape) {
        if (!err[0]) snprintf(err, errlen, "%s: a malformed object", where);
        return false;
    }
    if (t < 0) {
        snprintf(err, errlen, "%s.type: Field required", where);
        return false;
    }
    if (!have_v || v != std::floor(v) || v < (double)min) {
        snprintf(err, errlen, "%s.value: an integer >= %ld is required", where, min);
        return false;
    }
    *type_at = t;
    *value = v > (double)INT32_MAX ? (long)INT32_MAX : (long)v;   /* a larger count means the same as "never" */
    return true;
}

/// Read an array of strings into one-name-a-line text (null: none).
/// @param p      the cursor (advanced past the value)
/// @param out    receives the malloc'd lines (NULL for null or an empty array)
/// @param where  the member's path, for the error
/// @param err    receives the client error on false
/// @param errlen size of `err`
/// @return false unless the value is null or an array of strings
static bool ctx_name_list(const char **p, char **out, const char *where, char *err, size_t errlen) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p != '[') {
        snprintf(err, errlen, "%s: Input should be a valid list", where);
        return false;
    }
    (*p)++;
    buf b = {0};
    json_ws(p);
    while (**p && **p != ']') {
        char *name = NULL;
        if (!json_string(p, &name)) {
            buf_free(&b);
            snprintf(err, errlen, "%s: Input should be a list of strings", where);
            return false;
        }
        buf_puts(&b, name);
        buf_putc(&b, '\n');
        free(name);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') {
        buf_free(&b);
        return false;
    }
    (*p)++;
    free(*out);
    *out = buf_take(&b);
    return true;
}

/// Whether `name` is one of the lines of `lines`.
/// @param lines  one name a line (NULL: none)
/// @param name   the tool name
/// @return true when listed
static bool ctx_name_listed(const char *lines, const char *name) {
    if (!lines || !name) return false;
    const size_t n = strlen(name);
    for (const char *l = lines; *l;) {
        const char *e = strchr(l, '\n');
        const size_t len = e ? (size_t)(e - l) : strlen(l);
        if (len == n && !memcmp(l, name, n)) return true;
        if (!e) break;
        l = e + 1;
    }
    return false;
}

/// Read one clear_tool_uses_20250919 edit's members (its "type" already known).
/// @param raw    the edit object's JSON text
/// @param where  "context_management.edits.N"
/// @param e      the edit (defaults filled by the caller)
/// @param err    receives the client error on false
/// @param errlen size of `err`
/// @return false on a malformed or unknown member
static bool ctx_read_clear_tool_uses(const char *raw, const char *where, ctx_edit *e, char *err, size_t errlen) {
    static const char *const trigger_types[] = {"input_tokens", "tool_uses", NULL};
    static const char *const keep_types[] = {"tool_uses", NULL};
    static const char *const least_types[] = {"input_tokens", NULL};
    char at[96];
    const char *p = raw;
    return ctx_each_member(&p, [&](const char *k, const char **q) {
        snprintf(at, sizeof at, "%s.%.40s", where, k);
        int t = 0;
        long v = 0;
        if (!strcmp(k, "type")) return json_skip_value(q);
        json_ws(q);
        if (!strcmp(k, "trigger")) {
            if (json_lit(q, "null")) return true;
            if (!ctx_typed_value(q, at, trigger_types, &t, &v, 1, err, errlen)) return false;
            e->trigger_tool_uses = t == 1;
            e->trigger = v;
            return true;
        }
        if (!strcmp(k, "keep")) {
            if (json_lit(q, "null")) return true;
            if (!ctx_typed_value(q, at, keep_types, &t, &v, 0, err, errlen)) return false;
            e->keep = v > INT32_MAX ? INT32_MAX : (int)v;
            return true;
        }
        if (!strcmp(k, "clear_at_least")) {
            if (json_lit(q, "null")) return true;
            if (!ctx_typed_value(q, at, least_types, &t, &v, 0, err, errlen)) return false;
            e->clear_at_least = v;
            return true;
        }
        if (!strcmp(k, "exclude_tools")) return ctx_name_list(q, &e->exclude, at, err, errlen);
        if (!strcmp(k, "clear_tool_inputs")) {
            /* a bool (every cleared call), or the names whose calls' inputs are cleared */
            if (json_lit(q, "null")) return true;
            if (**q == '[') {
                if (!ctx_name_list(q, &e->clear_inputs_only, at, err, errlen)) return false;
                e->clear_inputs = e->clear_inputs_only != NULL;
                return true;
            }
            if (json_bool(q, &e->clear_inputs)) return true;
            snprintf(err, errlen, "%s: Input should be a valid boolean or a list of strings", at);
            return false;
        }
        snprintf(err, errlen, "%s: Extra inputs are not permitted", at);
        return false;
    });
}

/// Read one clear_thinking_20251015 edit's members.
/// @param raw    the edit object's JSON text
/// @param where  "context_management.edits.N"
/// @param e      the edit (keep defaults to all)
/// @param err    receives the client error on false
/// @param errlen size of `err`
/// @return false on a malformed or unknown member
static bool ctx_read_clear_thinking(const char *raw, const char *where, ctx_edit *e, char *err, size_t errlen) {
    static const char *const keep_types[] = {"thinking_turns", "all", NULL};
    char at[96];
    const char *p = raw;
    return ctx_each_member(&p, [&](const char *k, const char **q) {
        snprintf(at, sizeof at, "%s.%.40s", where, k);
        if (!strcmp(k, "type")) return json_skip_value(q);
        if (strcmp(k, "keep")) {
            snprintf(err, errlen, "%s: Extra inputs are not permitted", at);
            return false;
        }
        json_ws(q);
        if (json_lit(q, "null")) return true;
        if (**q == '"') {
            char *s = NULL;
            const bool all = json_string(q, &s) && !strcmp(s, "all");
            free(s);
            if (!all) snprintf(err, errlen, "%s: Input should be 'all' or a thinking_turns object", at);
            e->keep = -1;
            return all;
        }
        /* {"type":"all"} or {"type":"thinking_turns","value":N >= 1} */
        char *type = json_object_member_raw(*q, "type");
        const bool all = type && !strcmp(type, "\"all\"");
        free(type);
        if (all) {
            e->keep = -1;
            return json_skip_value(q);
        }
        int t = 0;
        long v = 0;
        if (!ctx_typed_value(q, at, keep_types, &t, &v, 1, err, errlen)) return false;
        e->keep = v > INT32_MAX ? INT32_MAX : (int)v;
        return true;
    });
}

bool ctx_management_parse(const char **p, chat_conversation *c, request *r, char *err, size_t errlen) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    r->ctx_mgmt = true;
    char *edits = NULL;
    bool ok = ctx_each_member(p, [&](const char *k, const char **q) {
        if (!strcmp(k, "edits")) {
            free(edits);
            edits = NULL;
            return json_raw_value(q, &edits);
        }
        snprintf(err, errlen, "context_management.%.40s: Extra inputs are not permitted", k);
        return false;
    });
    if (!ok) {
        if (!err[0]) snprintf(err, errlen, "context_management: Input should be a valid object");
        free(edits);
        return false;
    }
    const char *q = edits ? edits : "null";
    json_ws(&q);
    if (json_lit(&q, "null")) {
        free(edits);
        return true;
    }
    if (*q != '[') {
        snprintf(err, errlen, "context_management.edits: Input should be a valid list");
        free(edits);
        return false;
    }
    q++;
    json_ws(&q);
    for (int i = 0; ok && *q && *q != ']'; i++) {
        char where[48];
        snprintf(where, sizeof where, "context_management.edits.%d", i);
        char *raw = NULL;
        ok = json_raw_value(&q, &raw);
        char *type_raw = ok ? json_object_member_raw(raw, "type") : NULL;
        const char *tp = type_raw;
        char *type = NULL;
        if (ok && !(tp && json_string(&tp, &type))) {
            snprintf(err, errlen, "%s.type: Field required", where);
            ok = false;
        }
        ctx_edit e = {};
        if (ok && !strcmp(type, "clear_tool_uses_20250919")) {
            e.kind = CTX_EDIT_CLEAR_TOOL_USES;
            e.trigger = 100000;   /* the API's defaults: 100,000 input tokens, keep 3 tool uses */
            e.keep = 3;
            ok = ctx_read_clear_tool_uses(raw, where, &e, err, errlen);
        } else if (ok && !strcmp(type, "clear_thinking_20251015")) {
            /* the default keep is per model; every model the API names as current keeps all turns */
            e.kind = CTX_EDIT_CLEAR_THINKING;
            e.keep = -1;
            ok = ctx_read_clear_thinking(raw, where, &e, err, errlen);
        } else if (ok && !strcmp(type, "compact_20260112")) {
            snprintf(err, errlen, "%s: compact_20260112 (server-side compaction) is not supported by this server; "
                                  "send clear_tool_uses_20250919 or clear_thinking_20251015", where);
            ok = false;
        } else if (ok) {
            snprintf(err, errlen, "%s: Input tag '%.60s' found using 'type' does not match any of the expected tags: "
                                  "'clear_tool_uses_20250919', 'clear_thinking_20251015'", where, type);
            ok = false;
        }
        for (int k = 0; ok && k < c->n_ctx_edits; k++) {
            if (c->ctx_edits[k].kind == e.kind) {
                snprintf(err, errlen, "%s: each edit type may appear once in context_management.edits", where);
                ok = false;
            }
        }
        if (ok && e.kind == CTX_EDIT_CLEAR_THINKING && c->n_ctx_edits > 0) {
            snprintf(err, errlen, "%s: clear_thinking_20251015 must be the first edit in context_management.edits",
                     where);
            ok = false;
        }
        if (ok) {
            c->ctx_edits[c->n_ctx_edits++] = e;
        } else {
            free(e.exclude);
            free(e.clear_inputs_only);
        }
        free(type);
        free(type_raw);
        free(raw);
        json_ws(&q);
        if (*q == ',') q++;
        json_ws(&q);
    }
    if (ok && *q != ']') ok = false;
    free(edits);
    return ok;
}

void ctx_edits_free(chat_conversation *c) {
    for (int i = 0; i < c->n_ctx_edits; i++) {
        free(c->ctx_edits[i].exclude);
        free(c->ctx_edits[i].clear_inputs_only);
    }
    c->n_ctx_edits = 0;
}

/* ---- plan and apply ------------------------------------------------------------------------------------------- */

/** What one edit clears in a conversation, by position: the same body parses to the same parts, so a plan made on
 *  one parse applies to another. */
struct ctx_edit_plan {
    int edit = -1;                              ///< its index in chat_conversation::ctx_edits
    std::vector<int> results;                   ///< tool-result parts whose content is cleared
    std::vector<std::pair<int, int>> inputs;    ///< (part, call) whose arguments are cleared
    std::vector<int> reasoning;                 ///< assistant parts whose reasoning is cleared
    int cleared = 0;                            ///< tool uses / thinking turns cleared
};

/// Plan a clear_tool_uses edit: the oldest clearable tool uses past `keep`, once the tool-use trigger is passed.
/// @param c   the conversation (the Anthropic parse: tool results are "tool" parts)
/// @param e   the edit
/// @param pl  the plan (left empty when nothing is to be cleared)
static void ctx_plan_tool_uses(const chat_conversation *c, const ctx_edit *e, ctx_edit_plan *pl) {
    struct use { int part, call; const char *id, *name; int result; };
    std::vector<use> uses;
    const chat_msgs *m = &c->msgs;
    for (int i = 0; i < m->len; i++)
        for (int k = 0; k < m->v[i].calls.len; k++)
            uses.push_back({i, k, m->v[i].calls.v[k].id, m->v[i].calls.v[k].name, -1});
    if (e->trigger_tool_uses && (long)uses.size() <= e->trigger) return;
    for (use &u : uses) {
        for (int i = u.part + 1; u.id && u.result < 0 && i < m->len; i++)
            if (m->v[i].role && !strcmp(m->v[i].role, "tool") && m->v[i].tool_call_id &&
                !strcmp(m->v[i].tool_call_id, u.id))
                u.result = i;
    }
    /* clearable: answered, not excluded, and not a tool-search result -- its tool_reference blocks are where the
     * deferred tools' definitions render (L325); clearing it would take tools away, not context */
    std::vector<const use *> clearable;
    for (const use &u : uses)
        if (u.result >= 0 && !ctx_name_listed(e->exclude, u.name) && m->v[u.result].tool_refs_len == 0)
            clearable.push_back(&u);
    const int n = (int)clearable.size() - e->keep;
    for (int i = 0; i < n; i++) {
        const use *u = clearable[i];
        pl->results.push_back(u->result);
        if (e->clear_inputs && (!e->clear_inputs_only || ctx_name_listed(e->clear_inputs_only, u->name)))
            pl->inputs.push_back({u->part, u->call});
        pl->cleared++;
    }
}

/// Plan a clear_thinking edit: the reasoning of every assistant turn with thinking older than the last `keep`.
/// @param c   the conversation
/// @param e   the edit (keep -1: all kept, nothing planned)
/// @param pl  the plan
static void ctx_plan_thinking(const chat_conversation *c, const ctx_edit *e, ctx_edit_plan *pl) {
    if (e->keep < 0) return;
    /* an assistant turn is one Anthropic message: a part that does not continue the previous one starts it */
    std::vector<std::vector<int>> turns;   /* the parts with reasoning, per assistant turn that has any */
    const chat_msgs *m = &c->msgs;
    for (int i = 0; i < m->len; i++) {
        const chat_msg *x = &m->v[i];
        if (!x->role || strcmp(x->role, "assistant")) continue;
        const bool starts = !x->anthropic_continues || i == 0 || !m->v[i - 1].role ||
                            strcmp(m->v[i - 1].role, "assistant");
        if (starts) turns.emplace_back();
        if (x->reasoning && x->reasoning[0]) turns.back().push_back(i);
    }
    std::vector<std::vector<int>> thinking;
    for (auto &t : turns)
        if (!t.empty()) thinking.push_back(t);
    const int n = (int)thinking.size() - e->keep;
    for (int i = 0; i < n; i++) {
        pl->reasoning.insert(pl->reasoning.end(), thinking[i].begin(), thinking[i].end());
        pl->cleared++;
    }
}

/// Apply a plan to a conversation (one parsed from the same body as the plan's).
/// @param c   the conversation
/// @param pl  the plan
static void ctx_plan_apply(chat_conversation *c, const ctx_edit_plan *pl) {
    for (int i : pl->results) {
        chat_msg *x = &c->msgs.v[i];
        free(x->content);
        x->content = server_xstrdup(k_ctx_cleared_result);
        for (int k = 0; k < x->images_len; k++) free(x->images[k].bytes);
        free(x->images);
        free(x->image_ph_off);
        x->images = NULL;
        x->image_ph_off = NULL;
        x->images_len = x->images_cap = 0;
    }
    for (const auto &pc : pl->inputs) {
        tool_calls *calls = &c->msgs.v[pc.first].calls;
        free(calls->v[pc.second].arguments);
        calls->v[pc.second].arguments = server_xstrdup("{}");
        /* the sampled bytes tool memory would replay carry the original input */
        calls->inputs_cleared = true;
    }
    for (int i : pl->reasoning) {
        free(c->msgs.v[i].reasoning);
        c->msgs.v[i].reasoning = NULL;
    }
}

/// A rendered request's prompt size: its tokens, or with no engine (the renderer tests) its bytes.
/// @param e  the engine (may be NULL)
/// @param r  the rendered request
/// @return the size
static int ctx_prompt_size(const pulsar_engine *e, const request *r) {
    if (e) return r->prompt.len;
    return r->prompt_text ? (int)strlen(r->prompt_text) : 0;
}

/// Parse the body afresh, apply the accepted plans and `extra`, and render: one trial of the edits.
/// @param e           the engine (may be NULL)
/// @param fmt         the chat format
/// @param s           the server (may be NULL)
/// @param body        the request body
/// @param def_tokens  the generation cap when the request sets none
/// @param accepted    the plans already kept, in order
/// @param extra       the plan under trial
/// @param rk          receives the rendered request (initialised here; the caller frees it)
/// @param err         receives the reason on false
/// @param errlen      size of `err`
/// @return false when the parse or the render refused
static bool ctx_render_with(pulsar_engine *e, pulsar_chat_format fmt, server *s, const char *body, int def_tokens,
                            const std::vector<const ctx_edit_plan *> &accepted, const ctx_edit_plan *extra,
                            request *rk, char *err, size_t errlen) {
    chat_conversation ck = {};
    request_init(rk, REQ_CHAT, def_tokens);
    bool ok = parse_chat_conversation_anthropic(body, &ck, rk, err, errlen);
    if (ok) {
        for (const ctx_edit_plan *a : accepted) ctx_plan_apply(&ck, a);
        ctx_plan_apply(&ck, extra);
        ok = render_chat_conversation(e, fmt, s, &ck, rk, err, errlen);
    }
    chat_conversation_free(&ck);
    return ok;
}

bool parse_anthropic_request_render(pulsar_engine *e, pulsar_chat_format fmt, server *s, const char *body,
                                    int def_tokens, request *r, char *err, size_t errlen) {
    chat_conversation c = {};
    request_init(r, REQ_CHAT, def_tokens);
    bool ok = parse_chat_conversation_anthropic(body, &c, r, err, errlen);
    /* the plans read the conversation as parsed: the render below rewrites parts of it (tool references, the
     * folded tool results) */
    std::vector<ctx_edit_plan> plans;
    for (int i = 0; ok && i < c.n_ctx_edits; i++) {
        ctx_edit_plan pl;
        pl.edit = i;
        if (c.ctx_edits[i].kind == CTX_EDIT_CLEAR_TOOL_USES) ctx_plan_tool_uses(&c, &c.ctx_edits[i], &pl);
        else ctx_plan_thinking(&c, &c.ctx_edits[i], &pl);
        if (pl.cleared > 0) plans.push_back(std::move(pl));
    }
    /* the render as sent: every request renders it, and with nothing to clear it IS the prompt (keep "all", a
     * trigger not reached: one parse, one render, byte-identical to a body without context_management) */
    ok = ok && render_chat_conversation(e, fmt, s, &c, r, err, errlen);
    if (!ok) {
        chat_conversation_free(&c);
        if (err && errlen && !err[0]) snprintf(err, errlen, "invalid JSON request");
        request_free(r);
        return false;
    }
    const int original = ctx_prompt_size(e, r);
    if (r->ctx_mgmt) r->ctx_original_tokens = original;
    /* a request that continues live tool state (its results' calls are not in the history it sent) is served from
     * the bank's KV, which the history cannot edit: it is served as sent */
    if (plans.empty() || r->anthropic_requires_live_tool_state) {
        chat_conversation_free(&c);
        return true;
    }
    std::vector<const ctx_edit_plan *> accepted;
    ctx_edit_applied applied[PULSAR_CTX_EDITS_MAX];
    int n_applied = 0;
    int before = original;
    request kept;                        /* the last render kept (an edited one); empty until one is */
    memset(&kept, 0, sizeof kept);
    bool have_kept = false;
    for (const ctx_edit_plan &pl : plans) {
        const ctx_edit *ed = &c.ctx_edits[pl.edit];
        if (ed->kind == CTX_EDIT_CLEAR_TOOL_USES && !ed->trigger_tool_uses && original <= ed->trigger) continue;
        request rk;
        ok = ctx_render_with(e, fmt, s, body, def_tokens, accepted, &pl, &rk, err, errlen);
        if (!ok) {
            request_free(&rk);
            break;
        }
        const int after = ctx_prompt_size(e, &rk);
        const int cleared = before > after ? before - after : 0;
        if (ed->clear_at_least > 0 && cleared < ed->clear_at_least) {
            request_free(&rk);   /* not worth what it costs the cached prefix: not applied */
            continue;
        }
        if (have_kept) request_free(&kept);
        kept = rk;
        have_kept = true;
        before = after;
        accepted.push_back(&pl);
        applied[n_applied++] = {ed->kind, pl.cleared, cleared};
    }
    chat_conversation_free(&c);
    if (!ok) {
        /* the unedited render succeeded, so an edited one refusing is this server's fault: say which */
        char why[160];
        snprintf(why, sizeof why, "%s", err && err[0] ? err : "a malformed render");
        if (err && errlen) snprintf(err, errlen, "context_management: the edited conversation did not render: %s", why);
        if (have_kept) request_free(&kept);
        request_free(r);
        return false;
    }
    if (have_kept) {
        /* an edit whose parts the family never renders (the reasoning of turns it drops anyway) clears turns but
         * leaves the prompt as it was: then the render as sent stays, live binding and all */
        const bool same = r->prompt.len == kept.prompt.len &&
                          (r->prompt.len == 0 || !memcmp(r->prompt.v, kept.prompt.v, sizeof(r->prompt.v[0]) *
                                                                                           (size_t)r->prompt.len)) &&
                          !strcmp(r->prompt_text ? r->prompt_text : "", kept.prompt_text ? kept.prompt_text : "");
        if (same) {
            request_free(&kept);
        } else {
            request_free(r);
            *r = kept;
            r->ctx_edited = true;
            /* the slot's live tool binding sits on KV of the UNEDITED history: continuing it with only the
             * tool-result tail would serve the model the cleared content again -- the prefix match on the edited
             * prompt decides instead (and server_jobs records no binding after this request) */
            stop_list_clear(&r->anthropic_live_call_ids);
            free(r->anthropic_live_suffix_text);
            free(r->anthropic_live_suffix_spans);
            r->anthropic_live_suffix_text = NULL;
            r->anthropic_live_suffix_spans = NULL;
            r->anthropic_live_suffix_n_spans = 0;
        }
    }
    r->ctx_mgmt = true;
    r->ctx_original_tokens = original;
    r->n_ctx_applied = n_applied;
    for (int i = 0; i < n_applied; i++) r->ctx_applied[i] = applied[i];
    return true;
}

void append_ctx_management_json(buf *b, const request *r) {
    if (!r || !r->ctx_mgmt) return;
    buf_puts(b, ",\"context_management\":{\"applied_edits\":[");
    for (int i = 0; i < r->n_ctx_applied; i++) {
        const ctx_edit_applied *a = &r->ctx_applied[i];
        if (i) buf_putc(b, ',');
        if (a->kind == CTX_EDIT_CLEAR_TOOL_USES)
            buf_printf(b, "{\"type\":\"clear_tool_uses_20250919\",\"cleared_tool_uses\":%d,\"cleared_input_tokens\":%d}",
                       a->cleared, a->cleared_input_tokens);
        else
            buf_printf(b, "{\"type\":\"clear_thinking_20251015\",\"cleared_thinking_turns\":%d,"
                          "\"cleared_input_tokens\":%d}", a->cleared, a->cleared_input_tokens);
    }
    buf_puts(b, "]}");
}

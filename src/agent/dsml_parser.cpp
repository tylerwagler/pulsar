#include "pulsar_agent_internal.h"



/* ============================================================================
 * DSML Tool-Call Parser
 * ============================================================================
 *
 * The model streams raw text tokens.  This parser recognizes completed DSML
 * tool stanzas and keeps a copy of the raw stanza for diagnostics.  It is
 * deliberately strict after the opening marker: typo recovery belongs to the
 * streaming detector so the actual tool parser stays small and predictable.
 *
 * Every literal it matches comes from pulsar_dsml_syntaxes (src/lib/
 * pulsar_dsml.h), the same table the server renders and parses with; the
 * three spellings the model samples (canonical, first-bar-omitted, plain XML)
 * are all recognised, for every tag, and attribute values and string
 * parameter values are decoded with the server's entity decoder (L184).
 */

static void agent_tool_call_free(agent_tool_call *c) {
    if (!c) return;
    free(c->name);
    for (int i = 0; i < c->argc; i++) {
        free(c->args[i].name);
        free(c->args[i].value);
    }
    free(c->args);
    memset(c, 0, sizeof(*c));
}



static void agent_tool_calls_free(agent_tool_calls *calls) {
    if (!calls) return;
    for (int i = 0; i < calls->len; i++) agent_tool_call_free(&calls->v[i]);
    free(calls->v);
    memset(calls, 0, sizeof(*calls));
}



/* A string parameter's bytes carry DSML entities (the model is told to write
 * the closing tag inside a value as "&lt;/｜DSML｜parameter>"); decode them
 * with the ONE decoder the server's parser uses.  JSON-typed values are
 * taken verbatim, as the server does. */
static void agent_tool_call_add_arg(agent_tool_call *c, const char *name,
                                    const char *value, size_t value_len,
                                    bool is_string) {
    if (c->argc == c->argcap) {
        c->argcap = c->argcap ? c->argcap * 2 : 4;
        c->args = (agent_tool_arg *)agent_xrealloc(c->args, (size_t)c->argcap * sizeof(c->args[0]));
    }
    char *raw = xstrndup(value, value_len);
    char *decoded = is_string ? pulsar_dsml_unescape(raw) : raw;
    if (decoded != raw) free(raw);
    c->args[c->argc++] = (agent_tool_arg){
        .name = xstrdup(name),
        .value = decoded,
        .is_string = is_string,
    };
}



static void agent_tool_calls_push(agent_tool_calls *calls, agent_tool_call *call) {
    if (!call->name) return;
    if (calls->len == calls->cap) {
        calls->cap = calls->cap ? calls->cap * 2 : 2;
        calls->v = (agent_tool_call *)agent_xrealloc(calls->v, (size_t)calls->cap * sizeof(calls->v[0]));
    }
    calls->v[calls->len++] = *call;
    memset(call, 0, sizeof(*call));
}



const char *agent_tool_arg_value(const agent_tool_call *call, const char *name) {
    for (int i = 0; i < call->argc; i++) {
        if (call->args[i].name && !strcmp(call->args[i].name, name))
            return call->args[i].value ? call->args[i].value : "";
    }
    return NULL;
}



void agent_dsml_parser_free(agent_dsml_parser *p) {
    if (!p) return;
    free(p->raw);
    agent_tool_call_free(&p->current);
    free(p->param_name);
    agent_tool_calls_free(&p->calls);
    memset(p, 0, sizeof(*p));
}



void agent_dsml_parser_reset(agent_dsml_parser *p) {
    agent_dsml_parser_free(p);
    p->state = AGENT_DSML_SEARCH;
}



static void agent_dsml_raw_append(agent_dsml_parser *p, const char *s, size_t n) {
    if (!n) return;
    if (p->raw_len + n + 1 > p->raw_cap) {
        size_t cap = p->raw_cap ? p->raw_cap * 2 : 512;
        while (cap < p->raw_len + n + 1) cap *= 2;
        p->raw = (char *)agent_xrealloc(p->raw, cap);
        p->raw_cap = cap;
    }
    memcpy(p->raw + p->raw_len, s, n);
    p->raw_len += n;
    p->raw[p->raw_len] = '\0';
}



void agent_dsml_set_error(agent_dsml_parser *p, const char *msg) {
    p->state = AGENT_DSML_ERROR;
    snprintf(p->error, sizeof(p->error), "%s", msg);
}



/* The tags the parser recognises; each names a column pair of the table. */
typedef enum {
    AGENT_DSML_TAG_TOOL_CALLS,
    AGENT_DSML_TAG_INVOKE,
    AGENT_DSML_TAG_PARAMETER,
} agent_dsml_tag;

static const char *agent_dsml_open_lit(const pulsar_dsml_syntax *syn, agent_dsml_tag tag) {
    switch (tag) {
    case AGENT_DSML_TAG_TOOL_CALLS: return syn->tool_calls_start;
    case AGENT_DSML_TAG_INVOKE:     return syn->invoke_start;
    default:                        return syn->param_start;
    }
}

static const char *agent_dsml_close_lit(const pulsar_dsml_syntax *syn, agent_dsml_tag tag) {
    switch (tag) {
    case AGENT_DSML_TAG_TOOL_CALLS: return syn->tool_calls_end;
    case AGENT_DSML_TAG_INVOKE:     return syn->invoke_end;
    default:                        return syn->param_end;
    }
}



static bool agent_dsml_is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}



/* An opening invoke/parameter tag in ANY syntax: the table's literal, then
 * '>' or the whitespace before its attributes. */
static bool agent_dsml_open_tag_is(const char *tag, agent_dsml_tag kind) {
    for (size_t i = 0; i < PULSAR_DSML_SYNTAXES; i++) {
        const char *lit = agent_dsml_open_lit(&pulsar_dsml_syntaxes[i], kind);
        const size_t n = strlen(lit);
        if (strncmp(tag, lit, n) != 0) continue;
        const char c = tag[n];
        if (c == '>' || agent_dsml_is_ws(c)) return true;
    }
    return false;
}



/* The fullwidth bar of the DSML marker, which the model sometimes repeats
 * before a closing tag's '>' ("</｜DSML｜parameter｜>"). */
static const char agent_dsml_bar[] = "｜";

/* A closing tag of `kind` at s, in ANY syntax, accepting the few harmless
 * closing-tag variants the model has been observed to emit: whitespace, a
 * stray bar, whitespace, then '>'.  Opening tags stay strict so accidental
 * prose does not become a tool call. */
static bool agent_dsml_close_tag_at(const char *s, agent_dsml_tag kind, size_t *tag_len) {
    for (size_t i = 0; i < PULSAR_DSML_SYNTAXES; i++) {
        const char *lit = agent_dsml_close_lit(&pulsar_dsml_syntaxes[i], kind);
        const size_t prefix_len = strlen(lit) - 1;   /* every closing literal ends in '>' */
        if (strncmp(s, lit, prefix_len) != 0) continue;
        const char *p = s + prefix_len;
        while (agent_dsml_is_ws(*p)) p++;
        if (strncmp(p, agent_dsml_bar, strlen(agent_dsml_bar)) == 0) p += strlen(agent_dsml_bar);
        while (agent_dsml_is_ws(*p)) p++;
        if (*p != '>') continue;
        if (tag_len) *tag_len = (size_t)(p - s) + 1;
        return true;
    }
    return false;
}



/* Recognize a streamed parameter close tag prefix, in ANY syntax.  Full close
 * detection is handled by agent_dsml_close_tag_at(); this helper exists for
 * online behavior: terminal rendering must hide partial close tags without
 * waiting for the whole parameter to finish. */
bool agent_dsml_parameter_close_tail(const char *tail, size_t len,
                                            bool *complete) {
    const size_t bar_len = sizeof(agent_dsml_bar) - 1;
    *complete = false;
    for (size_t si = 0; si < PULSAR_DSML_SYNTAXES; si++) {
        const char *lit = pulsar_dsml_syntaxes[si].param_end;
        const size_t prefix_len = strlen(lit) - 1;   /* without the '>' */
        if (len <= prefix_len) {
            if (memcmp(lit, tail, len) == 0) return true;
            continue;
        }
        if (memcmp(lit, tail, prefix_len) != 0) continue;
        size_t i = prefix_len;
        while (i < len && agent_dsml_is_ws(tail[i])) i++;
        if (i < len && len - i <= bar_len) {
            if (memcmp(agent_dsml_bar, tail + i, len - i) == 0) return true;
        }
        if (i + bar_len <= len && memcmp(tail + i, agent_dsml_bar, bar_len) == 0)
            i += bar_len;
        bool matched = true;
        for (; i < len; i++) {
            if (tail[i] == '>') {
                *complete = i == len - 1;
                return *complete;
            }
            if (!agent_dsml_is_ws(tail[i])) { matched = false; break; }
        }
        if (matched) return true;
    }
    return false;
}



/* Bytes a partial closing tag must have before the sampler is forced to
 * argmax (param_close_prefix): the "</" plus the shortest DSML marker of any
 * marker-bearing syntax -- a bare "</" inside a value must force nothing.
 * Derived from the table; asserts its shape ("</" marker "parameter>"). */
static size_t agent_dsml_close_force_min(void) {
    static const char name_gt[] = "parameter>";
    size_t min = (size_t)-1;
    for (size_t i = 0; i < PULSAR_DSML_SYNTAXES; i++) {
        const char *lit = pulsar_dsml_syntaxes[i].param_end;
        const size_t n = strlen(lit);
        if (n <= sizeof(name_gt) - 1 || strcmp(lit + n - (sizeof(name_gt) - 1), name_gt) != 0) {
            fprintf(stderr, "pulsar-agent: DSML table row %zu param_end is not \"</\"marker\"parameter>\": %s\n",
                    i, lit);
            abort();
        }
        const size_t marker = n - (sizeof(name_gt) - 1);   /* "</" + marker */
        if (marker > 2 && marker < min) min = marker;
    }
    return min;
}



static void agent_dsml_update_param_close_prefix(agent_dsml_parser *p) {
    p->param_close_prefix = false;
    if (p->state != AGENT_DSML_PARAM_VALUE || p->raw_len <= p->param_value_start)
        return;

    const char *value = p->raw + p->param_value_start;
    const char *end = p->raw + p->raw_len;
    const char *lt = end;
    while (lt > value) {
        lt--;
        if (*lt == '<') break;
    }
    if (lt < value || *lt != '<') return;

    size_t tail_len = (size_t)(end - lt);
    if (tail_len > 64) return;
    bool complete = false;
    p->param_close_prefix =
        tail_len >= agent_dsml_close_force_min() &&
        agent_dsml_parameter_close_tail(lt, tail_len, &complete) &&
        !complete;
}



/* Find a DSML closing tag of `kind` (any syntax, lenient tail). */
static char *agent_dsml_find_close_tag(const char *s, agent_dsml_tag kind, size_t *tag_len) {
    const char *p = s;
    while ((p = strstr(p, "</")) != NULL) {
        if (agent_dsml_close_tag_at(p, kind, tag_len)) return (char *)p;
        p++;
    }
    return NULL;
}



/* Parse as much of the accumulated DSML buffer as possible.  The parser can be
 * called after every streamed byte: incomplete input leaves state unchanged
 * until enough bytes arrive, while malformed completed input switches to
 * AGENT_DSML_ERROR so the model gets a retryable tool error. */
static void agent_dsml_parse(agent_dsml_parser *p) {
    while (p->state == AGENT_DSML_STRUCTURAL || p->state == AGENT_DSML_PARAM_VALUE) {
        if (p->state == AGENT_DSML_PARAM_VALUE) {
            size_t end_tag_len = 0;
            char *end = agent_dsml_find_close_tag(p->raw + p->param_value_start,
                                                  AGENT_DSML_TAG_PARAMETER, &end_tag_len);
            if (!end) return;
            agent_tool_call_add_arg(&p->current, p->param_name ? p->param_name : "",
                                    p->raw + p->param_value_start,
                                    (size_t)(end - (p->raw + p->param_value_start)),
                                    p->param_is_string);
            p->param_close_prefix = false;
            free(p->param_name);
            p->param_name = NULL;
            p->parse_pos = (size_t)(end - p->raw) + end_tag_len;
            p->state = AGENT_DSML_STRUCTURAL;
            continue;
        }

        while (p->parse_pos < p->raw_len && agent_dsml_is_ws(p->raw[p->parse_pos]))
            p->parse_pos++;
        if (p->parse_pos >= p->raw_len) return;

        size_t close_len = 0;
        if (agent_dsml_close_tag_at(p->raw + p->parse_pos, AGENT_DSML_TAG_TOOL_CALLS, &close_len)) {
            agent_tool_calls_push(&p->calls, &p->current);
            p->parse_pos += close_len;
            p->state = AGENT_DSML_DONE;
            return;
        }
        if (agent_dsml_close_tag_at(p->raw + p->parse_pos, AGENT_DSML_TAG_INVOKE, &close_len)) {
            agent_tool_calls_push(&p->calls, &p->current);
            p->parse_pos += close_len;
            continue;
        }

        char *tag_end = strchr(p->raw + p->parse_pos, '>');
        if (!tag_end) return;
        size_t tag_len = (size_t)(tag_end - (p->raw + p->parse_pos)) + 1;
        char *tag = xstrndup(p->raw + p->parse_pos, tag_len);

        if (agent_dsml_open_tag_is(tag, AGENT_DSML_TAG_INVOKE)) {
            agent_tool_call_free(&p->current);
            p->current.name = pulsar_dsml_attr(tag, "name");
            if (!p->current.name) {
                free(tag);
                agent_dsml_set_error(p, "tool invoke without name");
                return;
            }
            p->parse_pos += tag_len;
        } else if (agent_dsml_open_tag_is(tag, AGENT_DSML_TAG_PARAMETER)) {
            free(p->param_name);
            p->param_name = pulsar_dsml_attr(tag, "name");
            char *is_string = pulsar_dsml_attr(tag, "string");
            p->param_is_string = is_string && !strcmp(is_string, "true");
            free(is_string);
            if (!p->param_name) {
                free(tag);
                agent_dsml_set_error(p, "tool parameter without name");
                return;
            }
            p->parse_pos += tag_len;
            p->param_value_start = p->parse_pos;
            p->param_close_prefix = false;
            p->state = AGENT_DSML_PARAM_VALUE;
        } else {
            snprintf(p->error, sizeof(p->error), "unexpected DSML tag: %.*s",
                     (int)(tag_len > 80 ? 80 : tag_len), tag);
            free(tag);
            p->state = AGENT_DSML_ERROR;
            return;
        }
        free(tag);
    }
}



/* The block is seeded with the CANONICAL opener whatever spelling the
 * detector accepted (tool_viz.cpp); the body tags are matched against every
 * spelling, so a block that opens short and closes long still parses. */
void agent_dsml_start(agent_dsml_parser *p) {
    const char *start = PULSAR_DSML_CANONICAL->tool_calls_start;
    p->state = AGENT_DSML_STRUCTURAL;
    p->search_len = 0;
    agent_dsml_raw_append(p, start, strlen(start));
    p->parse_pos = strlen(start);
}



void agent_dsml_feed(agent_dsml_parser *p, const char *s, size_t n) {
    if (p->state == AGENT_DSML_DONE || p->state == AGENT_DSML_ERROR) return;

    for (size_t i = 0; i < n; i++) {
        char c = s[i];
        if (p->state == AGENT_DSML_SEARCH) {
            if (p->search_len == sizeof(p->search_tail)) {
                memmove(p->search_tail, p->search_tail + 1, --p->search_len);
            }
            p->search_tail[p->search_len++] = c;
            for (size_t si = 0; si < PULSAR_DSML_SYNTAXES; si++) {
                const char *start = pulsar_dsml_syntaxes[si].tool_calls_start;
                const size_t start_len = strlen(start);
                if (p->search_len >= start_len &&
                    memcmp(p->search_tail + p->search_len - start_len, start, start_len) == 0) {
                    agent_dsml_start(p);
                    break;
                }
            }
            continue;
        }

        agent_dsml_raw_append(p, &c, 1);
        agent_dsml_parse(p);
        if (p->state == AGENT_DSML_PARAM_VALUE)
            agent_dsml_update_param_close_prefix(p);
        else
            p->param_close_prefix = false;
    }
}

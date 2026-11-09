/** @file
 *  pulsar-agent: tool calling in the loaded family's own syntax (L365).
 *
 *  DeepSeek has no tools template: the agent spells DSML out in its prompt and reads calls with its own parser
 *  (dsml_parser.cpp).  Every other family declares tools in its chat template and has an output parser -- the
 *  server's path (src/lib qwen_chat / nemotron_chat): the head declares agent_build_tools_json()
 *  (pulsar_chat_open_tools), the family's out_parser reads the sampled stream, and its calls run as the agent's
 *  tool calls.  Gemma 4 answers a call INSIDE its model turn: the worker leaves that turn open
 *  (pulsar_chat_calls_continue_turn) and the next turn opens with the named responses.  Inkling is parked: refused
 *  by name. */
#include "pulsar_agent_internal.h"
#include "out_parse_kit.h"
#include "pyjson.h"
#include "chat_render_kit.h"
#include "qwen_chat.h"
#include "nemotron_chat.h"
#include "gemma4_chat.h"



bool agent_engine_dsml(pulsar_engine *e) {
    const pulsar_chat_format f = pulsar_engine_chat_format(e);
    return f == PULSAR_CHAT_DS4_V4 || f == PULSAR_CHAT_DS4_V41;
}

bool agent_worker_dsml(const agent_worker *w) { return agent_engine_dsml(w->engine); }



bool agent_family_tools_supported(pulsar_engine *e, char *err, size_t errlen) {
    switch (pulsar_engine_chat_format(e)) {
    case PULSAR_CHAT_DS4_V4:
    case PULSAR_CHAT_DS4_V41:
    case PULSAR_CHAT_QWEN:
    case PULSAR_CHAT_NEMOTRON_H:
    case PULSAR_CHAT_GEMMA4:
        return true;
    case PULSAR_CHAT_INKLING:
        snprintf(err, errlen, "Inkling's tool calling is not wired into pulsar-agent (the family is parked)");
        return false;
    default:
        snprintf(err, errlen, "this model has no chat format pulsar-agent can call tools in");
        return false;
    }
}



out_parser *agent_family_parser_new(pulsar_engine *e, bool thinking, char *err, size_t errlen) {
    char *tools = agent_build_tools_json();
    char perr[200] = {0};
    out_parser *out = NULL;
    switch (pulsar_engine_chat_format(e)) {
    case PULSAR_CHAT_QWEN: {
        qwen_output_parser *p = new qwen_output_parser();
        if (p->init(thinking, tools, perr, sizeof perr)) out = p;
        else delete p;
        break;
    }
    case PULSAR_CHAT_NEMOTRON_H: {
        nemotron_output_parser *p = new nemotron_output_parser();
        if (p->init(thinking, tools, perr, sizeof perr)) out = p;
        else delete p;
        break;
    }
    case PULSAR_CHAT_GEMMA4: {
        gemma4_output_parser *p = new gemma4_output_parser();
        if (p->init(thinking, tools, perr, sizeof perr)) out = p;
        else delete p;
        break;
    }
    default:   /* DeepSeek reads DSML with the agent's own parser; the rest are refused by name */
        if (agent_family_tools_supported(e, perr, sizeof perr))
            snprintf(perr, sizeof perr, "DeepSeek's calls are DSML, read by the agent's own parser");
        break;
    }
    free(tools);
    if (!out) snprintf(err, errlen, "tool-call parser: %s", perr[0] ? perr : "unavailable");
    return out;
}

void agent_family_parser_free(out_parser *p) { delete p; }

bool agent_family_parser_wants_greedy(const out_parser *p) {
    /* the server's decode rule (L284 P6): a call's structure is argmax, its string payload sampled */
    return p && p->in_tool_call() && !p->in_payload();
}



/** @brief Append one call to `calls` from the family parser's name and arguments object; each member becomes an
 *  argument: a JSON string as its raw text (is_string), any other value as its JSON text.
 *  @param calls     the list, appended to
 *  @param name      the function name
 *  @param arguments the arguments, a JSON object's text
 *  @param err       receives why the arguments are not an object
 *  @param errlen    size of `err`
 *  @return false (nothing appended) when the arguments are not a JSON object */
static bool agent_family_call_append(agent_tool_calls *calls, const std::string &name, const std::string &arguments,
                                     char *err, size_t errlen) {
    pyjson_value v;
    char jerr[160];
    if (!pyjson_parse(arguments.data(), arguments.size(), &v, jerr, sizeof jerr)) {
        snprintf(err, errlen, "%s: arguments are not JSON (%s)", name.c_str(), jerr);
        return false;
    }
    if (v.kind != pyjson_value::OBJ) {
        snprintf(err, errlen, "%s: arguments are not a JSON object", name.c_str());
        return false;
    }
    if (calls->len == calls->cap) {
        calls->cap = calls->cap ? calls->cap * 2 : 4;
        calls->v = (agent_tool_call *)realloc(calls->v, (size_t)calls->cap * sizeof(*calls->v));
        if (!calls->v) abort();
    }
    agent_tool_call *c = &calls->v[calls->len++];
    memset(c, 0, sizeof(*c));
    c->name = strdup(name.c_str());
    c->argcap = (int)v.o.size();
    c->args = c->argcap ? (agent_tool_arg *)calloc((size_t)c->argcap, sizeof(*c->args)) : NULL;
    if (!c->name || (c->argcap && !c->args)) abort();
    for (const auto &m : v.o) {
        agent_tool_arg *a = &c->args[c->argc++];
        a->name = strdup(m.first.c_str());
        a->is_string = m.second.kind == pyjson_value::STR;
        a->value = strdup(a->is_string ? m.second.s.c_str() : py_tojson(m.second).c_str());
        if (!a->name || !a->value) abort();
    }
    return true;
}

bool agent_family_calls_take(const out_parser *p, agent_tool_calls *calls, char *err, size_t errlen) {
    for (const out_call &c : p->calls())
        if (!agent_family_call_append(calls, c.name, c.arguments, err, errlen)) return false;
    return true;
}



#ifdef PULSAR_AGENT_TEST
/** The tools array is valid JSON, one function object per schema; the conversion keeps strings raw and other
 *  values as JSON text; DSML families take no family parser. */
static void test_agent_family_tools(void) {
    char *tools = agent_build_tools_json();
    pyjson_value v;
    char jerr[160];
    AGENT_TEST_ASSERT(pyjson_parse(tools, strlen(tools), &v, jerr, sizeof jerr));
    AGENT_TEST_ASSERT(v.kind == pyjson_value::ARR && v.a.size() == 9);
    for (const pyjson_value &t : v.a) {
        const pyjson_value *fn = t.get("function");
        AGENT_TEST_ASSERT(fn && fn->get("name") && fn->get("parameters"));
    }
    free(tools);

    char *dsml = agent_build_tools_prompt(true);
    char *family = agent_build_tools_prompt(false);
    AGENT_TEST_ASSERT(strstr(dsml, PULSAR_TOOL_CALLS_START) && strstr(dsml, "### Available Tool Schemas"));
    AGENT_TEST_ASSERT(!strstr(family, PULSAR_DSML_SHORT) && !strstr(family, "DSML") &&
                      !strstr(family, "### Available Tool Schemas") && strstr(family, "[upto]"));
    free(dsml);
    free(family);

    agent_tool_calls calls = {0};
    char err[200] = {0};
    AGENT_TEST_ASSERT(agent_family_call_append(&calls, "read",
                                               "{\"path\": \"a \\\"b\\\".c\", \"start_line\": 5, \"whole\": true}",
                                               err, sizeof err));
    AGENT_TEST_ASSERT(calls.len == 1 && !strcmp(calls.v[0].name, "read") && calls.v[0].argc == 3);
    AGENT_TEST_ASSERT(!strcmp(calls.v[0].args[0].value, "a \"b\".c") && calls.v[0].args[0].is_string);
    AGENT_TEST_ASSERT(!strcmp(calls.v[0].args[1].value, "5") && !calls.v[0].args[1].is_string);
    AGENT_TEST_ASSERT(!strcmp(calls.v[0].args[2].value, "true") && !calls.v[0].args[2].is_string);
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&calls.v[0], "path"), "a \"b\".c"));
    AGENT_TEST_ASSERT(!agent_family_call_append(&calls, "bash", "[1]", err, sizeof err) && calls.len == 1);
    AGENT_TEST_ASSERT(!agent_family_call_append(&calls, "bash", "{\"command\":", err, sizeof err) && calls.len == 1);
    agent_tool_calls_free(&calls);

    /* a whole Qwen turn through the family parser: thinking, prose, one call */
    qwen_output_parser q;
    char *qtools = agent_build_tools_json();
    AGENT_TEST_ASSERT(q.init(true, qtools, err, sizeof err));
    free(qtools);
    std::vector<out_event> ev;
    const char *turn = "let me look\n</think>\n\nReading it.\n<tool_call>\n<function=read>\n<parameter=path>\n"
                       "/tmp/x.txt\n</parameter>\n</function>\n</tool_call>";
    for (const char *s = turn; *s; s++) q.feed(s, 1, &ev);
    q.finish(&ev);
    AGENT_TEST_ASSERT(q.errors() == 0 && q.calls().size() == 1);
    AGENT_TEST_ASSERT(agent_family_calls_take(&q, &calls, err, sizeof err) && calls.len == 1);
    AGENT_TEST_ASSERT(!strcmp(calls.v[0].name, "read") &&
                      !strcmp(agent_tool_arg_value(&calls.v[0], "path"), "/tmp/x.txt"));
    agent_tool_calls_free(&calls);

    /* Gemma 4: a thought channel, then two calls in its own syntax; the turn stops at <|tool_response> */
    gemma4_output_parser g;
    char *gtools = agent_build_tools_json();
    AGENT_TEST_ASSERT(g.init(false, gtools, err, sizeof err));
    free(gtools);
    ev.clear();
    const char *gturn = "<|channel>thought\nread then write<channel|>"
                        "<|tool_call>call:read{path:<|\"|>notes/a b.txt<|\"|>,start_line:3}<tool_call|>"
                        "<|tool_call>call:write{content:<|\"|>x\ny<|\"|>,path:<|\"|>o.txt<|\"|>}<tool_call|>";
    for (const char *s = gturn; *s; s++) g.feed(s, 1, &ev);
    g.finish(&ev);
    AGENT_TEST_ASSERT(g.errors() == 0 && g.calls().size() == 2);
    AGENT_TEST_ASSERT(agent_family_calls_take(&g, &calls, err, sizeof err) && calls.len == 2);
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&calls.v[0], "path"), "notes/a b.txt"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&calls.v[0], "start_line"), "3"));
    AGENT_TEST_ASSERT(!strcmp(calls.v[1].name, "write") && !strcmp(agent_tool_arg_value(&calls.v[1], "content"), "x\ny"));
    agent_tool_calls_free(&calls);
}
#endif

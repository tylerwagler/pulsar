/* L267: what the API parsers make of a request, dumped field by field -- the check that a refactor of
 * the parse/render layers changes no request.  Host-only: no engine (the parsers render the prompt TEXT
 * and its client spans; tokenising is the engine's and does not move) and no server (tool memory and
 * the live-state checks are server state, exercised by the unit suite).
 *
 *   ./tests/api_golden tests/api-golden/corpus.jsonl > out.txt
 *   diff tests/api-golden/golden.txt out.txt
 *
 * The corpus is JSONL, one case a line: {"name":..., "endpoint":"chat|anthropic|responses",
 * "family":"deepseek|qwen", "body":{...}}.  The golden was recorded from the parsers before L267 moved
 * them; any difference is a change in what a request means. */
#define PULSAR_SERVER_TEST
#define PULSAR_SERVER_TEST_NO_MAIN
#include "../src/server/util.cpp"
#include "../src/server/request.cpp"
#include "../src/server/prompt_render.cpp"
#include "../src/server/api_parse.cpp"
#include "../src/server/chat_family.cpp"
#include "../src/server/genmsg.cpp"
#include "../src/server/deepseek_stream.cpp"
#include "../src/server/parser_deepseek.cpp"
#include "../src/server/parser_qwen.cpp"
#include "../src/server/openai_stream.cpp"
#include "../src/server/responses_stream.cpp"
#include "../src/server/anthropic_stream.cpp"
#include "../src/server/tool_memory.cpp"
#include "../src/server/kv_cache.cpp"
#include "../src/server/trace.cpp"
#include "../src/server/generate.cpp"
#include "../src/server/server_jobs.cpp"
#include "../src/server/server_sched.cpp"
#include "../src/server/http_server.cpp"
#include "../src/server/cli_main.cpp"

#include <string>

static void put_str(const char *key, const char *v) {
    buf b = {0};
    if (v) json_escape(&b, v);
    printf("  %s: %s\n", key, v ? b.ptr : "null");
    buf_free(&b);
}

static void put_spans(const char *key, const pulsar_text_span *v, uint32_t n) {
    printf("  %s: [", key);
    for (uint32_t i = 0; i < n; i++) printf("%s%zu-%zu", i ? " " : "", (size_t)v[i].lo, (size_t)v[i].hi);
    printf("]\n");
}

static void put_list(const char *key, const stop_list *l) {
    printf("  %s: [", key);
    for (int i = 0; i < l->len; i++) {
        buf b = {0};
        json_escape(&b, l->v[i]);
        printf("%s%s", i ? " " : "", b.ptr);
        buf_free(&b);
    }
    printf("]\n");
}

static void dump(const request *r) {
    /* the golden's labels predate the family table (L272 P3): the same two facts, read from it */
    printf("  kind %d api %d chat_v41 %d chat_qwen %d\n", (int)r->kind, (int)r->api, (int)r->family->v41,
           (int)(r->family->parser == SERVER_PARSER_QWEN));
    put_str("model", r->model);
    printf("  model_from_request %d max_tokens %d\n", r->model_from_request, r->max_tokens);
    printf("  sampling t %.9g/%d top_p %.9g/%d top_k %d/%d min_p %.9g/%d seed %llu\n", r->temperature,
           r->has_temperature, r->top_p, r->has_top_p, r->top_k, r->has_top_k, r->min_p, r->has_min_p,
           (unsigned long long)r->seed);
    printf("  logprobs %d top %d stream %d usage %d\n", r->logprobs, r->top_logprobs, r->stream,
           r->stream_include_usage);
    printf("  think_mode %d has_tools %d force_tool_call %d summary %d\n", (int)r->think_mode, r->has_tools,
           r->force_tool_call, r->reasoning_summary_emit);
    put_str("forced_tool_name", r->forced_tool_name);
    put_list("stops", &r->stops);
    printf("  tool_orders %d:", r->tool_orders.len);
    for (int i = 0; i < r->tool_orders.len; i++) {
        const tool_schema_order *o = &r->tool_orders.v[i];
        printf(" {%s|%s|%s|%d|", o->name ? o->name : "-", o->wire_name ? o->wire_name : "-",
               o->tool_namespace ? o->tool_namespace : "-", o->responses_tool_search);
        for (int k = 0; k < o->len; k++) printf("%s%s", k ? "," : "", o->prop[k]);
        printf("}");
    }
    printf("\n");
    put_str("qwen_tools_json", r->qwen_tools_json);
    put_str("prompt_text", r->prompt_text);
    put_spans("prompt_spans", r->prompt_spans, r->prompt_n_spans);
    printf("  images %d\n", r->n_images);
    printf("  responses live tool %d reasoning %d\n", r->responses_requires_live_tool_state,
           r->responses_requires_live_reasoning);
    put_list("responses_live_call_ids", &r->responses_live_call_ids);
    put_str("responses_live_suffix", r->responses_live_suffix_text);
    put_spans("responses_live_suffix_spans", r->responses_live_suffix_spans, r->responses_live_suffix_n_spans);
    printf("  anthropic live tool %d\n", r->anthropic_requires_live_tool_state);
    put_list("anthropic_live_call_ids", &r->anthropic_live_call_ids);
    put_str("anthropic_live_suffix", r->anthropic_live_suffix_text);
    put_spans("anthropic_live_suffix_spans", r->anthropic_live_suffix_spans, r->anthropic_live_suffix_n_spans);
}

/* one case: parse it the way the server's router would for that endpoint and family */
static bool golden_parse(const char *endpoint, const char *family, const char *body, request *r, char *err,
                         size_t errlen) {
    const bool qwen = !strcmp(family, "qwen");
    bool (*protocol)(const char *, chat_conversation *, request *, char *, size_t) =
        !strcmp(endpoint, "chat")      ? parse_chat_conversation_openai
      : !strcmp(endpoint, "anthropic") ? parse_chat_conversation_anthropic
      : !strcmp(endpoint, "responses") ? parse_chat_conversation_responses
                                       : NULL;
    if (!protocol) {
        snprintf(err, errlen, "(unknown endpoint %s)", endpoint);
        return false;
    }
    /* the server's parse_and_render, with the family named instead of read off an engine */
    request_init(r, REQ_CHAT, 256);
    chat_conversation c;
    memset(&c, 0, sizeof(c));
    const bool ok = protocol(body, &c, r, err, errlen) &&
                    render_chat_conversation(NULL, qwen ? PULSAR_CHAT_QWEN : PULSAR_CHAT_DS4_V41, NULL, &c, r,
                                             err, errlen);
    chat_conversation_free(&c);
    if (!ok) {
        if (!err[0]) snprintf(err, errlen, "invalid JSON request");
        request_free(r);
    }
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s corpus.jsonl\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "r");
    if (!f) { fprintf(stderr, "api_golden: cannot open %s\n", argv[1]); return 2; }
    char *line = NULL;
    size_t cap = 0;
    int n = 0;
    while (getline(&line, &cap, f) > 0) {
        const char *p = line;
        json_ws(&p);
        if (*p != '{') continue;
        p++;
        char *name = NULL, *endpoint = NULL, *family = NULL, *body = NULL;
        bool ok = true;
        while (ok && *p && *p != '}') {
            char *key = NULL;
            json_ws(&p);
            ok = json_string(&p, &key);
            json_ws(&p);
            ok = ok && *p == ':';
            if (ok) p++;
            json_ws(&p);
            if (ok && !strcmp(key, "name")) ok = json_string(&p, &name);
            else if (ok && !strcmp(key, "endpoint")) ok = json_string(&p, &endpoint);
            else if (ok && !strcmp(key, "family")) ok = json_string(&p, &family);
            else if (ok && !strcmp(key, "body")) ok = json_raw_value(&p, &body);
            else if (ok) ok = json_skip_value(&p);
            free(key);
            json_ws(&p);
            if (*p == ',') p++;
        }
        if (!ok || !name || !endpoint || !family || !body) {
            fprintf(stderr, "api_golden: bad corpus line %d\n", n + 1);
            return 2;
        }
        request r;
        memset(&r, 0, sizeof(r));
        char err[512] = "";
        const bool parsed = golden_parse(endpoint, family, body, &r, err, sizeof(err));
        printf("== %s [%s/%s] %s\n", name, endpoint, family, parsed ? "OK" : "REFUSED");
        if (parsed) {
            dump(&r);
            request_free(&r);
        } else {
            put_str("error", err);
        }
        free(name);
        free(endpoint);
        free(family);
        free(body);
        n++;
    }
    free(line);
    fclose(f);
    fprintf(stderr, "api_golden: %d cases\n", n);
    return 0;
}

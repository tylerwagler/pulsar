/* In-file unit tests extracted move-only from cli_main.cpp (the old
 * `#else` branch of its PULSAR_SERVER_TEST guard). Compiled ONLY inside
 * the tests/pulsar_test.cpp harness TU, which #includes every server .c
 * with PULSAR_SERVER_TEST defined and includes cli_main.cpp BEFORE this
 * file, so cli_main.cpp file-statics the tests poke (parse_options,
 * server_kv_budget_bytes, server_default_kv_disk_dir,
 * server_resolve_kv_disk_dir) remain reachable exactly as before.
 * A standalone compile (no PULSAR_SERVER_TEST) yields an empty object
 * for the pulsar-server link. */
#include "pulsar_server_internal.h"

#ifdef PULSAR_SERVER_TEST

static int test_failures = 0;



static void test_assert(bool cond, const char *file, int line, const char *expr) {
    if (cond) return;
    fprintf(stderr, "%s:%d: assertion failed: %s\n", file, line, expr);
    test_failures++;
}



#define TEST_ASSERT(expr) test_assert((expr), __FILE__, __LINE__, #expr)


static void test_tool_schema_order_from_anthropic_schema(void) {
    tool_schema_orders orders = {0};
    tool_schema_orders_add_json(&orders,
        "{\"name\":\"bash\",\"input_schema\":{\"type\":\"object\",\"properties\":{"
        "\"command\":{\"type\":\"string\"},"
        "\"description\":{\"type\":\"string\"}}}}");
    const tool_schema_order *order = tool_schema_orders_find(&orders, "bash");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && order->len == 2);
    TEST_ASSERT(order && !strcmp(order->prop[0], "command"));
    TEST_ASSERT(order && !strcmp(order->prop[1], "description"));
    tool_schema_orders_free(&orders);
}



static void test_tool_schema_order_from_openai_tools(void) {
    const char *json =
        "[{\"type\":\"function\",\"function\":{\"name\":\"edit\",\"parameters\":{"
        "\"type\":\"object\",\"properties\":{"
        "\"filePath\":{\"type\":\"string\"},"
        "\"oldString\":{\"type\":\"string\"},"
        "\"newString\":{\"type\":\"string\"}}}}}]";
    const char *p = json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&p, &schemas, &orders, false, NULL));
    TEST_ASSERT(schemas && strstr(schemas, "\"name\": \"edit\""));
    const tool_schema_order *order = tool_schema_orders_find(&orders, "edit");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && order->len == 3);
    TEST_ASSERT(order && !strcmp(order->prop[0], "filePath"));
    TEST_ASSERT(order && !strcmp(order->prop[1], "oldString"));
    TEST_ASSERT(order && !strcmp(order->prop[2], "newString"));
    free(schemas);
    tool_schema_orders_free(&orders);
}



/* Ported from upstream ds4 3196149: two spellings of the SAME schema —
 * compact-with-\u-escapes and whitespace-padded-with-raw-UTF-8 — must render
 * identical canonical prompt bytes (Python separators, decoded UTF-8,
 * preserved key order), so a client's JSON serializer can no longer change
 * how the toolset tokenizes (or which warm-bank prefixes it can hit). */
static void test_openai_tool_schema_json_spelling_is_canonical(void) {
    const char *compact =
        "[{\"type\":\"function\",\"function\":{\"name\":\"bash\","
        "\"description\":\"Run \\u2014 now\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"command\":{\"type\":\"string\","
        "\"description\":\"line\\nrocket \\ud83d\\ude80\"}},"
        "\"required\":[\"command\"],\"additionalProperties\":false}}}]";
    const char *spaced =
        "[ { \"type\" : \"function\", \"function\" : { \"name\" : \"bash\", "
        "\"description\" : \"Run \xe2\x80\x94 now\", \"parameters\" : { \"type\" : \"object\", "
        "\"properties\" : { \"command\" : { \"type\" : \"string\", "
        "\"description\" : \"line\\nrocket \xf0\x9f\x9a\x80\" } }, \"required\" : [ \"command\" ], "
        "\"additionalProperties\" : false } } } ]";
    char *a = NULL, *b = NULL;
    tool_schema_orders oa = {0}, ob = {0};
    const char *pa = compact, *pb = spaced;
    TEST_ASSERT(parse_tools_value(&pa, &a, &oa, false, NULL));
    TEST_ASSERT(parse_tools_value(&pb, &b, &ob, false, NULL));
    TEST_ASSERT(a && b && !strcmp(a, b));
    TEST_ASSERT(strstr(a, "\"name\": \"bash\""));
    TEST_ASSERT(strstr(a, "rocket \xf0\x9f\x9a\x80"));   /* decoded UTF-8, not \u */
    TEST_ASSERT(strstr(a, "line\\nrocket"));             /* control escape kept */
    TEST_ASSERT(!strstr(a, "\\u2014"));                  /* em-dash decoded */
    free(a); free(b);
    tool_schema_orders_free(&oa);
    tool_schema_orders_free(&ob);
}



/* The canonical-spelling invariant must hold for ANTHROPIC-shaped tools too
 * ({name, description, input_schema} — no "function" wrapper), because that is
 * the surface Claude Code drives.  Two spellings of one schema must render
 * identical prompt bytes, or a client's serializer still fragments warm-bank
 * prefixes and drifts off the reference tokenization. */
static void test_anthropic_tool_schema_json_spelling_is_canonical(void) {
    const char *compact =
        "[{\"name\":\"Bash\",\"description\":\"Run \\u2014 now\","
        "\"input_schema\":{\"type\":\"object\",\"properties\":{"
        "\"command\":{\"type\":\"string\"}},\"required\":[\"command\"]}}]";
    const char *spaced =
        "[ { \"name\" : \"Bash\", \"description\" : \"Run \xe2\x80\x94 now\", "
        "\"input_schema\" : { \"type\" : \"object\", \"properties\" : { "
        "\"command\" : { \"type\" : \"string\" } }, \"required\" : [ \"command\" ] } } ]";
    char *a = NULL, *b = NULL;
    tool_schema_orders oa = {0}, ob = {0};
    const char *pa = compact, *pb = spaced;
    TEST_ASSERT(parse_tools_value(&pa, &a, &oa, false, NULL));
    TEST_ASSERT(parse_tools_value(&pb, &b, &ob, false, NULL));
    TEST_ASSERT(a && b && !strcmp(a, b));
    TEST_ASSERT(strstr(a, "\"name\": \"Bash\""));
    TEST_ASSERT(!strstr(a, "\\u2014"));                  /* em-dash decoded */
    free(a); free(b);
    tool_schema_orders_free(&oa);
    tool_schema_orders_free(&ob);
}



static void test_tool_schema_order_from_responses_tool_search(void) {
    const char *json =
        "[{\"type\":\"tool_search\",\"execution\":\"client\","
        "\"description\":\"Search deferred tools\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"},"
        "\"limit\":{\"type\":\"number\"}},\"required\":[\"query\"]}}]";
    const char *p = json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&p, &schemas, &orders, false, NULL));
    TEST_ASSERT(schemas && strstr(schemas, "\"name\":\"tool_search\""));
    TEST_ASSERT(schemas && strstr(schemas, "\"description\":\"Search deferred tools\""));
    const tool_schema_order *order = tool_schema_orders_find(&orders, "tool_search");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && order->responses_tool_search);
    TEST_ASSERT(order && order->len == 2);
    TEST_ASSERT(order && !strcmp(order->prop[0], "query"));
    TEST_ASSERT(order && !strcmp(order->prop[1], "limit"));
    free(schemas);
    tool_schema_orders_free(&orders);
}



static void test_responses_function_named_tool_search_stays_function_call(void) {
    const char *json =
        "[{\"type\":\"function\",\"function\":{\"name\":\"tool_search\","
        "\"description\":\"A normal user function that happens to use a reserved name\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"}}}}}]";
    const char *p = json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&p, &schemas, &orders, false, NULL));
    const tool_schema_order *order = tool_schema_orders_find(&orders, "tool_search");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && !order->responses_tool_search);

    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_user_tool_search");
    tc.name = xstrdup("tool_search");
    tc.arguments = xstrdup("{\"query\":\"plain function\"}");
    tool_calls_push(&calls, tc);
    responses_tool_item item = {
        .fc_id = "fc_user_tool_search",
        .call_id = "call_user_tool_search",
        .is_custom = false,
        .output_index = 0,
    };

    buf out = {0};
    responses_append_function_call_item(&out, &calls.v[0], &item,
                                        "completed", true, &orders);
    TEST_ASSERT(strstr(out.ptr, "\"type\":\"function_call\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"type\":\"tool_search_call\"") == NULL);

    buf_free(&out);
    tool_calls_free(&calls);
    free(schemas);
    tool_schema_orders_free(&orders);
}



static void test_responses_namespace_tool_schemas_restore_wire_namespace(void) {
    const char *json =
        "[{\"type\":\"namespace\",\"name\":\"mcp__perplexity__\","
        "\"description\":\"Perplexity tools\","
        "\"tools\":[{\"type\":\"function\",\"name\":\"perplexity_search\","
        "\"description\":\"Search the web\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"},"
        "\"recency\":{\"type\":\"number\"}}}}]}]";
    const char *p = json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&p, &schemas, &orders, false, NULL));
    TEST_ASSERT(schemas && strstr(schemas, "\"name\":\"mcp__perplexity__perplexity_search\""));
    TEST_ASSERT(schemas && strstr(schemas, "\"name\":\"perplexity_search\"") == NULL);

    const tool_schema_order *order =
        tool_schema_orders_find(&orders, "mcp__perplexity__perplexity_search");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && order->tool_namespace && !strcmp(order->tool_namespace, "mcp__perplexity__"));
    TEST_ASSERT(order && order->wire_name && !strcmp(order->wire_name, "perplexity_search"));
    TEST_ASSERT(order && order->len == 2);

    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_ns");
    tc.name = xstrdup("mcp__perplexity__perplexity_search");
    tc.arguments = xstrdup("{\"query\":\"deepseek\",\"recency\":7}");
    tool_calls_push(&calls, tc);
    responses_tool_item item = {
        .fc_id = "fc_ns",
        .call_id = "call_ns",
        .is_custom = false,
        .output_index = 0,
    };
    buf out = {0};
    responses_append_function_call_item(&out, &calls.v[0], &item,
                                        "completed", true, &orders);
    TEST_ASSERT(strstr(out.ptr, "\"name\":\"perplexity_search\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"namespace\":\"mcp__perplexity__\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "mcp__perplexity__perplexity_search") == NULL);

    buf_free(&out);
    tool_calls_free(&calls);
    free(schemas);
    tool_schema_orders_free(&orders);
}



static void test_responses_input_tool_search_output_loads_tools(void) {
    const char *json =
        "["
        "{\"type\":\"tool_search_call\",\"call_id\":\"call_search\","
        "\"execution\":\"client\",\"arguments\":{\"query\":\"perplexity\"}},"
        "{\"type\":\"tool_search_output\",\"call_id\":\"call_search\","
        "\"status\":\"completed\",\"execution\":\"client\",\"tools\":["
        "{\"type\":\"namespace\",\"name\":\"mcp__perplexity__\","
        "\"description\":\"Perplexity tools\","
        "\"tools\":[{\"type\":\"function\",\"name\":\"perplexity_search\","
        "\"description\":\"Search with Perplexity\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"}}}}]}]}"
        "]";
    const char *p = json;
    chat_msgs msgs = {0};
    buf loaded = {0};
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_responses_input(&p, &msgs, &loaded, &orders));
    TEST_ASSERT(loaded.ptr && strstr(loaded.ptr, "\"name\":\"mcp__perplexity__perplexity_search\""));
    const tool_schema_order *order =
        tool_schema_orders_find(&orders, "mcp__perplexity__perplexity_search");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && order->tool_namespace && !strcmp(order->tool_namespace, "mcp__perplexity__"));
    TEST_ASSERT(order && order->wire_name && !strcmp(order->wire_name, "perplexity_search"));
    TEST_ASSERT(msgs.len == 2);
    TEST_ASSERT(msgs.v[0].calls.len == 1);
    TEST_ASSERT(!strcmp(msgs.v[0].calls.v[0].name, "tool_search"));
    TEST_ASSERT(strstr(msgs.v[1].content, "mcp__perplexity__") != NULL);

    buf_free(&loaded);
    tool_schema_orders_free(&orders);
    chat_msgs_free(&msgs);
}



static void test_responses_input_tool_search_output_rejects_bad_tools(void) {
    const char *json =
        "[{\"type\":\"tool_search_output\",\"call_id\":\"call_search\","
        "\"status\":\"completed\",\"tools\":{\"not\":\"a tool array\"}}]";
    const char *p = json;
    chat_msgs msgs = {0};
    buf loaded = {0};
    tool_schema_orders orders = {0};
    TEST_ASSERT(!parse_responses_input(&p, &msgs, &loaded, &orders));
    buf_free(&loaded);
    tool_schema_orders_free(&orders);
    chat_msgs_free(&msgs);
}



static void test_responses_input_function_call_namespace_round_trips_to_dsml(void) {
    const char *tools_json =
        "[{\"type\":\"namespace\",\"name\":\"mcp__perplexity__\","
        "\"tools\":[{\"type\":\"function\",\"name\":\"perplexity_search\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"}}}}]}]";
    const char *tools_p = tools_json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&tools_p, &schemas, &orders, false, NULL));

    const char *input_json =
        "[{\"type\":\"function_call\",\"call_id\":\"call_ns\","
        "\"name\":\"perplexity_search\",\"namespace\":\"mcp__perplexity__\","
        "\"arguments\":{\"query\":\"deepseek\"}}]";
    const char *input_p = input_json;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_responses_input(&input_p, &msgs, NULL, NULL));
    TEST_ASSERT(msgs.len == 1);
    TEST_ASSERT(msgs.v[0].calls.len == 1);
    TEST_ASSERT(!strcmp(msgs.v[0].calls.v[0].name,
                        "mcp__perplexity__perplexity_search"));

    char *prompt = render_chat_prompt_text(&msgs, schemas, &orders, PULSAR_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt,
        "<｜DSML｜invoke name=\"mcp__perplexity__perplexity_search\">") != NULL);
    TEST_ASSERT(strstr(prompt, "<｜DSML｜invoke name=\"perplexity_search\">") == NULL);

    free(prompt);
    chat_msgs_free(&msgs);
    free(schemas);
    tool_schema_orders_free(&orders);
}



static void test_responses_output_sends_tool_search_call_item(void) {
    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_search");
    tc.name = xstrdup("tool_search");
    tc.arguments = xstrdup("{\"limit\":3,\"query\":\"perplexity\"}");
    tool_calls_push(&calls, tc);
    const char *tools_json =
        "[{\"type\":\"tool_search\",\"execution\":\"client\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"},\"limit\":{\"type\":\"number\"}}}}]";
    const char *tools_p = tools_json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&tools_p, &schemas, &orders, false, NULL));
    responses_tool_item item = {
        .fc_id = "fc_search",
        .call_id = "call_search",
        .is_custom = false,
        .output_index = 0,
    };

    buf out = {0};
    responses_append_function_call_item(&out, &calls.v[0], &item,
                                        "completed", true, &orders);
    TEST_ASSERT(strstr(out.ptr, "\"type\":\"tool_search_call\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"execution\":\"client\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"status\":\"completed\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"arguments\":{\"limit\":3,\"query\":\"perplexity\"}") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"type\":\"function_call\"") == NULL);

    buf_free(&out);
    free(schemas);
    tool_schema_orders_free(&orders);
    tool_calls_free(&calls);
}



static tool_calls make_swapped_bash_call(void) {
    tool_calls calls = {0};
    tool_call tc = {0};
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"description\":\"list files\",\"command\":\"ls -la\",\"timeout\":10}");
    tool_calls_push(&calls, tc);
    return calls;
}



static tool_schema_orders make_bash_order(void) {
    tool_schema_orders orders = {0};
    tool_schema_orders_add_json(&orders,
        "{\"name\":\"bash\",\"input_schema\":{\"type\":\"object\",\"properties\":{"
        "\"command\":{\"type\":\"string\"},"
        "\"description\":{\"type\":\"string\"}}}}");
    return orders;
}



static char *read_socket_text(int fd) {
    buf b = {0};
    char tmp[1024];
    ssize_t n;
    while ((n = read(fd, tmp, sizeof(tmp))) > 0) {
        buf_append(&b, tmp, (size_t)n);
    }
    return buf_take(&b);
}



static void test_context_length_error_uses_protocol_standard_shape(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.prompt.len = 16;
    TEST_ASSERT(request_exceeds_context(&r, 16));
    TEST_ASSERT(!request_exceeds_context(&r, 17));

    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] >= 0 && sv[1] >= 0) {
        TEST_ASSERT(http_error_context_length_exceeded(sv[0], &r, 16, 16));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 400") != NULL);
        TEST_ASSERT(strstr(out, "\"type\":\"invalid_request_error\"") != NULL);
        TEST_ASSERT(strstr(out, "\"code\":\"context_length_exceeded\"") != NULL);
        TEST_ASSERT(strstr(out, "\"param\":\"messages\"") != NULL);
        TEST_ASSERT(strstr(out, "\"n_prompt_tokens\":16") != NULL);
        TEST_ASSERT(strstr(out, "\"n_ctx\":16") != NULL);
        free(out);
        close(sv[0]);
        close(sv[1]);
    }
    request_free(&r);

    request a;
    request_init(&a, REQ_CHAT, 128);
    a.api = API_ANTHROPIC;

    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] >= 0 && sv[1] >= 0) {
        TEST_ASSERT(http_error_context_length_exceeded(sv[0], &a, 20, 20));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        TEST_ASSERT(strstr(out, "{\"type\":\"error\",\"error\"") != NULL);
        TEST_ASSERT(strstr(out, "\"type\":\"invalid_request_error\"") != NULL);
        TEST_ASSERT(strstr(out, "\"n_prompt_tokens\":20") != NULL);
        free(out);
        close(sv[0]);
        close(sv[1]);
    }
    request_free(&a);
}



/* logprob_stream_ready is the byte-watermark math that decides which ledger
 * entries each SSE chunk releases -- the core of the stream-vs-non-stream
 * concatenation invariant. It advances from lg->streamed while an entry's
 * end_off is at/below the watermark, so entries release in order, each once,
 * and never before the chunk that carries their token bytes. A bug here
 * double-emits or drops a token's logprobs across a chunk boundary. */
static void test_logprob_stream_ready_watermark(void) {
    logprob_entry v[4] = {0};
    v[0].end_off = 3;
    v[1].end_off = 7;
    v[2].end_off = 7;   /* two entries share a watermark (a multi-token piece) */
    v[3].end_off = 12;
    logprob_ledger lg = {0};
    lg.enabled = true;
    lg.v = v;
    lg.len = 4;
    lg.streamed = 0;

    /* disabled ledger releases nothing */
    lg.enabled = false;
    TEST_ASSERT(logprob_stream_ready(&lg, SIZE_MAX) == 0);
    lg.enabled = true;

    /* watermark below the first entry: nothing ready */
    TEST_ASSERT(logprob_stream_ready(&lg, 2) == 0);
    /* exactly at an end_off releases that entry (<=, not <) */
    TEST_ASSERT(logprob_stream_ready(&lg, 3) == 1);
    /* between entries: only those fully covered */
    TEST_ASSERT(logprob_stream_ready(&lg, 6) == 1);
    /* both entries sharing end_off=7 release together, never split */
    TEST_ASSERT(logprob_stream_ready(&lg, 7) == 3);
    TEST_ASSERT(logprob_stream_ready(&lg, 11) == 3);
    /* the terminal SIZE_MAX chunk releases the remainder */
    TEST_ASSERT(logprob_stream_ready(&lg, SIZE_MAX) == 4);

    /* advancing streamed makes it resume, not re-release: entries [0,streamed)
     * are already on the wire and must not appear again. */
    lg.streamed = 3;
    TEST_ASSERT(logprob_stream_ready(&lg, 7) == 3);      /* nothing new at 7 */
    TEST_ASSERT(logprob_stream_ready(&lg, SIZE_MAX) == 4);
    lg.streamed = 4;
    TEST_ASSERT(logprob_stream_ready(&lg, SIZE_MAX) == 4); /* all streamed, none left */
}



/* Generic 4xx on the Anthropic surface must use the {"type":"error", ...}
 * envelope (the SDK's discriminator), not the OpenAI {"error":{...}} shape --
 * every /v1/messages parse failure, not just context-length. */
static void test_error_envelope_shape_per_protocol(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] >= 0 && sv[1] >= 0) {
        TEST_ASSERT(http_error_anthropic(sv[0], 400, "bad field"));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 400") != NULL);
        TEST_ASSERT(strstr(out, "{\"type\":\"error\",\"error\":") != NULL);
        TEST_ASSERT(strstr(out, "\"type\":\"invalid_request_error\"") != NULL);
        TEST_ASSERT(strstr(out, "\"message\":\"bad field\"") != NULL);
        free(out);
        close(sv[0]);
        close(sv[1]);
    }

    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] >= 0 && sv[1] >= 0) {
        TEST_ASSERT(http_error(sv[0], 400, "bad field"));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        /* OpenAI shape: NO top-level "type":"error" wrapper */
        TEST_ASSERT(strstr(out, "{\"error\":{\"message\":") != NULL);
        TEST_ASSERT(strstr(out, "{\"type\":\"error\"") == NULL);
        free(out);
        close(sv[0]);
        close(sv[1]);
    }
}



static void test_anthropic_live_stream_sends_incremental_blocks(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_ANTHROPIC;
    r.stream = true;
    r.think_mode = PULSAR_THINK_HIGH;
    r.has_tools = true;
    r.tool_orders = make_bash_order();

    anthropic_stream st;
    TEST_ASSERT(anthropic_sse_start_live(sv[0], &r, "msg_test", 10, &st));
    const char *raw1 = "need a tool</think>Hello.\n\n";
    TEST_ASSERT(anthropic_sse_stream_update(sv[0], NULL, &r, "msg_test", &st,
                                            raw1, strlen(raw1), false));

    const char *raw =
        "need a tool</think>Hello.\n\n"
        PULSAR_TOOL_CALLS_START "\n";
    TEST_ASSERT(anthropic_sse_stream_update(sv[0], NULL, &r, "msg_test", &st,
                                            raw, strlen(raw), false));

    tool_calls calls = make_swapped_bash_call();
    TEST_ASSERT(anthropic_sse_finish_live(sv[0], NULL, &r, "msg_test", &st,
                                          raw, strlen(raw), &calls,
                                          "tool_calls", 8));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *msg_start = strstr(out, "event: message_start");
    const char *thinking = strstr(out, "\"thinking\":\"need a tool\"");
    const char *signature = strstr(out, "\"type\":\"signature_delta\"");
    const char *text = strstr(out, "\"text\":\"Hello.\"");
    const char *tool = strstr(out, "\"type\":\"tool_use\"");
    const char *stop = strstr(out, "event: message_stop");
    TEST_ASSERT(msg_start != NULL);
    TEST_ASSERT(thinking != NULL);
    TEST_ASSERT(signature != NULL);
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(stop != NULL);
    TEST_ASSERT(msg_start < thinking);
    TEST_ASSERT(thinking < signature);
    TEST_ASSERT(signature < text);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(tool < stop);
    TEST_ASSERT(strstr(out, PULSAR_TOOL_CALLS_START) == NULL);

    free(out);
    tool_calls_free(&calls);
    anthropic_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_anthropic_tool_stream_sends_live_tool_use(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_ANTHROPIC;
    r.stream = true;
    r.think_mode = PULSAR_THINK_NONE;
    r.has_tools = true;
    r.tool_orders = make_bash_order();

    anthropic_stream st;
    TEST_ASSERT(anthropic_sse_start_live(sv[0], &r, "msg_tool", 7, &st));

    const char *raw =
        "Before.\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">echo partial";
    TEST_ASSERT(anthropic_sse_stream_update(sv[0], NULL, &r, "msg_tool", &st,
                                            raw, strlen(raw), false));

    const char *raw_complete =
        "Before.\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">echo partial done" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END;
    TEST_ASSERT(anthropic_sse_stream_update(sv[0], NULL, &r, "msg_tool", &st,
                                            raw_complete, strlen(raw_complete), false));

    char *parsed_content = NULL;
    char *parsed_reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(raw_complete, false, &parsed_content,
                                           &parsed_reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    apply_anthropic_stream_tool_ids(&calls, &st);
    TEST_ASSERT(calls.v[0].id != NULL);
    TEST_ASSERT(!strncmp(calls.v[0].id, "toolu_", 6));
    TEST_ASSERT(anthropic_sse_finish_live(sv[0], NULL, &r, "msg_tool", &st,
                                          raw_complete, strlen(raw_complete),
                                          &calls, "tool_calls", 5));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *text = strstr(out, "\"text\":\"Before.\"");
    const char *tool = strstr(out, "\"type\":\"tool_use\"");
    const char *key = strstr(out, "\\\"command\\\":\\\"");
    const char *partial = strstr(out, "\"partial_json\":\"echo partial\"");
    const char *rest = strstr(out, "\"partial_json\":\" done\"");
    const char *stop = strstr(out, "event: message_stop");
    int tool_use_count = 0;
    for (const char *p = out; (p = strstr(p, "\"type\":\"tool_use\"")) != NULL; p++) {
        tool_use_count++;
    }
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(key != NULL);
    TEST_ASSERT(partial != NULL);
    TEST_ASSERT(rest != NULL);
    TEST_ASSERT(stop != NULL);
    TEST_ASSERT(strstr(out, calls.v[0].id) != NULL);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(tool < key);
    TEST_ASSERT(key < partial);
    TEST_ASSERT(partial < rest);
    TEST_ASSERT(rest < stop);
    TEST_ASSERT(tool_use_count == 1);
    TEST_ASSERT(strstr(out, PULSAR_TOOL_CALLS_START) == NULL);
    TEST_ASSERT(strstr(out, PULSAR_PARAM_START) == NULL);

    free(out);
    free(parsed_content);
    free(parsed_reasoning);
    tool_calls_free(&calls);
    anthropic_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_anthropic_usage_reports_cache_details(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_ANTHROPIC;
    r.cache_read_tokens = 7;
    r.cache_write_tokens = 3;

    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) {
        request_free(&r);
        return;
    }

    TEST_ASSERT(anthropic_final_response(sv[0], &r, "msg_usage", "OK", NULL, NULL, "stop", 10, 2, NULL));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"usage\":{\"input_tokens\":0") != NULL);
    TEST_ASSERT(strstr(out, "\"output_tokens\":2") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_read_input_tokens\":7") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_creation_input_tokens\":3") != NULL);

    free(out);
    close(sv[0]);
    close(sv[1]);

    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) {
        request_free(&r);
        return;
    }

    anthropic_stream st;
    TEST_ASSERT(anthropic_sse_start_live(sv[0], &r, "msg_usage_stream", 10, &st));
    shutdown(sv[0], SHUT_WR);
    out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "event: message_start") != NULL);
    TEST_ASSERT(strstr(out, "\"usage\":{\"input_tokens\":0") != NULL);
    TEST_ASSERT(strstr(out, "\"output_tokens\":0") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_read_input_tokens\":7") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_creation_input_tokens\":3") != NULL);

    free(out);
    close(sv[0]);
    close(sv[1]);
    request_free(&r);
}



static void test_openai_tool_stream_sends_incremental_text(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_HIGH;
    r.has_tools = true;
    r.tool_orders = make_bash_order();

    TEST_ASSERT(sse_chunk(sv[0], &r, "chatcmpl_test", NULL, NULL));

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw1 = "<think>need a tool</think>Hello.\n\n";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_test", &st,
                                         raw1, strlen(raw1), false));

    const char *raw =
        "<think>need a tool</think>Hello.\n\n"
        PULSAR_TOOL_CALLS_START "\n";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_test", &st,
                                         raw, strlen(raw), false));

    tool_calls calls = make_swapped_bash_call();
    TEST_ASSERT(openai_sse_finish_live(sv[0], NULL, &r, "chatcmpl_test", &st,
                                       raw, strlen(raw), &calls,
                                       "tool_calls", 10, 8));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *role = strstr(out, "\"role\":\"assistant\"");
    const char *thinking = strstr(out, "\"reasoning_content\":\"need a tool\"");
    const char *text = strstr(out, "\"content\":\"Hello.\"");
    const char *tool = strstr(out, "\"tool_calls\"");
    const char *done = strstr(out, "data: [DONE]");
    TEST_ASSERT(role != NULL);
    TEST_ASSERT(thinking != NULL);
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(done != NULL);
    TEST_ASSERT(role < thinking);
    TEST_ASSERT(thinking < text);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(tool < done);
    TEST_ASSERT(strstr(out, PULSAR_TOOL_CALLS_START) == NULL);
    TEST_ASSERT(strstr(out, "<think>") == NULL);

    free(out);
    tool_calls_free(&calls);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



/* A truncated generation often ends MID-closing-tag.  Repair used to append
 * fresh closing tags after the fragment, baking "</｜DSML｜"-style debris into
 * the parsed parameter value (and into the streamed args, since the final
 * flush parses repaired text).  The repair must trim the partial tag first. */
static void test_repair_dsml_trims_partial_closing_tag(void) {
    buf fixed = {0};
    buf in = {0};
    buf_puts(&in, "<think>go</think>" PULSAR_TOOL_CALLS_START "\n");
    buf_puts(&in, PULSAR_INVOKE_START " name=\"bash\">\n");
    buf_puts(&in, PULSAR_PARAM_START " name=\"command\" string=\"true\">ls -l /var/log</｜DSML｜");
    TEST_ASSERT(try_repair_dsml(in.ptr, in.len, &fixed));
    TEST_ASSERT(fixed.ptr != NULL);
    /* value ends at the real content; the partial tag is gone and exactly one
     * full closing sequence follows */
    TEST_ASSERT(strstr(fixed.ptr, "/var/log" PULSAR_PARAM_END) != NULL);
    TEST_ASSERT(strstr(fixed.ptr, "</｜DSML｜" PULSAR_PARAM_END) == NULL);
    buf_free(&in);
    buf_free(&fixed);
}

/* A generation cut mid-argument (finish=length) used to leave the streamed
 * tool call's arguments as UNTERMINATED JSON on the wire: the header and a
 * string-value prefix had been emitted, then nothing.  The finalize path
 * (upstream ds4 0ead8a8's problem, solved pulsar-shaped: our non-stream side
 * already repairs via try_repair_dsml; the stream now closes the open string
 * and args object so the wire JSON is well-formed and byte-consistent with
 * that repair).  The value stays visibly truncated; finish_reason=length
 * still marks the cut. */
static void test_openai_tool_stream_truncated_call_closes_args(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_HIGH;
    r.has_tools = true;
    r.tool_orders = make_bash_order();

    openai_stream st;
    openai_stream_start(&r, &st);
    buf raw = {0};
    buf_puts(&raw, "<think>go</think>Running.\n\n" PULSAR_TOOL_CALLS_START "\n");
    buf_puts(&raw, PULSAR_INVOKE_START " name=\"bash\">\n");
    buf_puts(&raw, PULSAR_PARAM_START " name=\"command\" string=\"true\">ls -la /tmp/prof");
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_test", &st,
                                         raw.ptr, raw.len, false));
    /* generation ends here: no </parameter>, no </invoke>, no closing tag */
    TEST_ASSERT(openai_sse_finish_live(sv[0], NULL, &r, "chatcmpl_test", &st,
                                       raw.ptr, raw.len, NULL,
                                       "length", 10, 8));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"name\":\"bash\"") != NULL);
    /* the args stream must END well-formed: a closing quote fragment and a
     * closing brace fragment after the value prefix */
    const char *val = strstr(out, "ls -la /tmp/prof");
    const char *closequote = val ? strstr(val, "\"arguments\":\"\\\"\"") : NULL;
    const char *closebrace = closequote ? strstr(closequote, "\"arguments\":\"}\"") : NULL;
    TEST_ASSERT(val != NULL);
    TEST_ASSERT(closequote != NULL);
    TEST_ASSERT(closebrace != NULL);
    TEST_ASSERT(strstr(out, "\"finish_reason\":\"length\"") != NULL);
    TEST_ASSERT(strstr(out, "data: [DONE]") != NULL);

    free(out);
    buf_free(&raw);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_openai_stream_usage_reports_cache_details(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.stream_include_usage = true;
    r.cache_read_tokens = 7;
    r.cache_write_tokens = 3;

    TEST_ASSERT(sse_done(sv[0], &r, "chatcmpl_usage", 10, 2));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"usage\":{\"prompt_tokens\":10") != NULL);
    TEST_ASSERT(strstr(out, "\"completion_tokens\":2") != NULL);
    TEST_ASSERT(strstr(out, "\"total_tokens\":12") != NULL);
    TEST_ASSERT(strstr(out, "\"prompt_tokens_details\":{") != NULL);
    TEST_ASSERT(strstr(out, "\"cached_tokens\":7") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_write_tokens\":3") != NULL);
    TEST_ASSERT(strstr(out, "data: [DONE]") != NULL);

    free(out);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_responses_usage_reports_cache_details(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_RESPONSES;
    r.cache_read_tokens = 7;
    r.cache_write_tokens = 3;

    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) {
        request_free(&r);
        return;
    }

    TEST_ASSERT(responses_final_response(sv[0], &r, "resp_usage", "OK", NULL, NULL,
                                         "stop", 10, 2));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"usage\":{\"input_tokens\":10") != NULL);
    TEST_ASSERT(strstr(out, "\"input_tokens_details\":{") != NULL);
    TEST_ASSERT(strstr(out, "\"cached_tokens\":7") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_write_tokens\":3") != NULL);
    TEST_ASSERT(strstr(out, "\"output_tokens\":2") != NULL);
    TEST_ASSERT(strstr(out, "\"total_tokens\":12") != NULL);

    free(out);
    close(sv[0]);
    close(sv[1]);

    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) {
        request_free(&r);
        return;
    }

    responses_stream st;
    responses_stream_init(&r, &st);
    TEST_ASSERT(responses_sse_completed(sv[0], &r, &st, NULL, NULL,
                                        "stop", 10, 2, 1234));
    shutdown(sv[0], SHUT_WR);
    out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"type\":\"response.completed\"") != NULL);
    TEST_ASSERT(strstr(out, "\"usage\":{\"input_tokens\":10") != NULL);
    TEST_ASSERT(strstr(out, "\"input_tokens_details\":{") != NULL);
    TEST_ASSERT(strstr(out, "\"cached_tokens\":7") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_write_tokens\":3") != NULL);
    TEST_ASSERT(strstr(out, "\"output_tokens\":2") != NULL);
    TEST_ASSERT(strstr(out, "\"total_tokens\":12") != NULL);

    free(out);
    responses_stream_free(&st);
    close(sv[0]);
    close(sv[1]);
    request_free(&r);
}



static void test_openai_chat_stream_splits_reasoning_without_tools(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_HIGH;
    r.has_tools = false;

    TEST_ASSERT(request_uses_structured_stream(&r));
    TEST_ASSERT(request_uses_openai_live_stream(&r));
    TEST_ASSERT(sse_chunk(sv[0], &r, "chatcmpl_title", NULL, NULL));

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw1 = "We need to generate a title";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_title", &st,
                                         raw1, strlen(raw1), false));

    const char *raw2 =
        "We need to generate a title</think>Free disk space check";
    TEST_ASSERT(openai_sse_finish_live(sv[0], NULL, &r, "chatcmpl_title", &st,
                                       raw2, strlen(raw2), NULL,
                                       "stop", 12, 8));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *role = strstr(out, "\"role\":\"assistant\"");
    const char *reasoning1 = strstr(out, "\"reasoning_content\":\"We need to generate \"");
    const char *reasoning2 = strstr(out, "\"reasoning_content\":\"a title\"");
    const char *content = strstr(out, "\"content\":\"Free disk space check\"");
    const char *done = strstr(out, "data: [DONE]");
    TEST_ASSERT(role != NULL);
    TEST_ASSERT(reasoning1 != NULL);
    TEST_ASSERT(reasoning2 != NULL);
    TEST_ASSERT(content != NULL);
    TEST_ASSERT(done != NULL);
    TEST_ASSERT(role < reasoning1);
    TEST_ASSERT(reasoning1 < reasoning2);
    TEST_ASSERT(reasoning2 < content);
    TEST_ASSERT(content < done);
    TEST_ASSERT(strstr(out, "\"content\":\"We need to generate a title") == NULL);
    TEST_ASSERT(strstr(out, "</think>") == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_openai_tool_stream_sends_partial_arguments(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_NONE;
    r.has_tools = true;
    r.tool_orders = make_bash_order();

    TEST_ASSERT(sse_chunk(sv[0], &r, "chatcmpl_partial_tool", NULL, NULL));

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw =
        "Before.\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">echo partial";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_partial_tool", &st,
                                         raw, strlen(raw), false));

    const char *raw_complete =
        "Before.\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">echo partial done" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_partial_tool", &st,
                                         raw_complete, strlen(raw_complete), false));

    char *parsed_content = NULL;
    char *parsed_reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(raw_complete, false, &parsed_content, &parsed_reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    apply_openai_stream_tool_ids(&calls, &st);
    TEST_ASSERT(calls.v[0].id != NULL);
    TEST_ASSERT(!strncmp(calls.v[0].id, "call_", 5));
    TEST_ASSERT(openai_sse_finish_live(sv[0], NULL, &r, "chatcmpl_partial_tool", &st,
                                       raw_complete, strlen(raw_complete), &calls,
                                       "tool_calls", 10, 4));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *text = strstr(out, "\"content\":\"Before.\"");
    const char *tool = strstr(out, "\"tool_calls\"");
    const char *key = strstr(out, "\\\"command\\\":\\\"");
    const char *partial = strstr(out, "\"arguments\":\"echo partial\"");
    const char *rest = strstr(out, "\"arguments\":\" done\"");
    int tool_id_count = 0;
    for (const char *p = out; (p = strstr(p, "\"id\":\"call_")) != NULL; p++) tool_id_count++;
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(key != NULL);
    TEST_ASSERT(partial != NULL);
    TEST_ASSERT(rest != NULL);
    TEST_ASSERT(strstr(out, calls.v[0].id) != NULL);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(tool < partial);
    TEST_ASSERT(partial < rest);
    TEST_ASSERT(tool_id_count == 1);
    TEST_ASSERT(strstr(out, PULSAR_TOOL_CALLS_START) == NULL);
    TEST_ASSERT(strstr(out, PULSAR_PARAM_START) == NULL);

    free(out);
    free(parsed_content);
    free(parsed_reasoning);
    tool_calls_free(&calls);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



/* A DSML block whose START is inside <think> but whose END lands AFTER
 * </think> straddles the reasoning boundary and is NOT an executable call
 * (upstream ds4 0ead8a8).  Classifying it as a complete tool suppressed the
 * stream from the marker onward, so the post-thinking answer never reached the
 * client.  The boundary text itself is malformed either way; what must not
 * happen is losing the content after </think>. */
static void test_openai_stream_keeps_text_when_tool_straddles_think_close(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_HIGH;
    r.has_tools = true;
    r.tool_orders = make_bash_order();

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw =
        "<think>consider " PULSAR_TOOL_CALLS_START "</think>Answer."
        PULSAR_TOOL_CALLS_END;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_straddle", &st,
                                         raw, strlen(raw), false));
    /* The regression discriminator: the straddling block used to be classified
     * as a complete call, latching the stream into SUPPRESS from the marker
     * onward.  It must instead close reasoning at </think> and continue. */
    TEST_ASSERT(st.mode != OPENAI_STREAM_SUPPRESS);
    TEST_ASSERT(st.mode == OPENAI_STREAM_TEXT);

    /* Flush: TEXT mode holds the tail back until final. */
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_straddle", &st,
                                         raw, strlen(raw), true));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    /* The user-visible regression: content after </think> was swallowed. */
    TEST_ASSERT(strstr(out, "Answer.") != NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



/* L006: a streaming slot that has gone silent (long prefill, or starved behind
 * another job) gets a surface-appropriate keepalive, and only then. */
static void test_stream_heartbeat_only_fires_when_silent(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    gen_state g;
    memset(&g, 0, sizeof(g));
    slot_writer_init(&g.writer, sv[0]);
    g.anthropic_live.active = true;

    /* Clock unarmed (nothing ever sent): no beat — we do not know the client
     * is idle rather than simply not started. */
    TEST_ASSERT(!gen_stream_heartbeat(&g));

    /* Arm the clock far in the past => silent => beat. */
    g.writer.last_write_ms = 1;
    TEST_ASSERT(gen_stream_heartbeat(&g));
    /* The beat re-stamps the clock, so it must NOT fire again immediately. */
    TEST_ASSERT(!gen_stream_heartbeat(&g));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "event: ping") != NULL);          /* real Anthropic event */
    TEST_ASSERT(strstr(out, "\"type\": \"ping\"") != NULL);
    free(out);
    slot_writer_free(&g.writer);
    close(sv[0]);
    close(sv[1]);
}



/* OpenAI/Responses get an SSE COMMENT, which every conformant parser drops
 * before it reaches application code — it cannot perturb the delta stream. */
static void test_stream_heartbeat_openai_uses_sse_comment(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    gen_state g;
    memset(&g, 0, sizeof(g));
    slot_writer_init(&g.writer, sv[0]);
    g.writer.last_write_ms = 1;

    /* No stream projection active (non-streaming request): never beat. */
    TEST_ASSERT(!gen_stream_heartbeat(&g));

    g.openai_live.active = true;
    TEST_ASSERT(gen_stream_heartbeat(&g));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, ": ping") != NULL);
    TEST_ASSERT(strstr(out, "event:") == NULL);   /* comment only, no protocol event */
    TEST_ASSERT(strstr(out, "data:") == NULL);
    free(out);
    slot_writer_free(&g.writer);
    close(sv[0]);
    close(sv[1]);
}



static void test_openai_tool_stream_waits_for_incomplete_tool_tags(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw_invoke = PULSAR_TOOL_CALLS_START "\n" PULSAR_INVOKE_START;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_incomplete_tool", &st,
                                         raw_invoke, strlen(raw_invoke), false));
    TEST_ASSERT(st.mode == OPENAI_STREAM_TOOL);
    TEST_ASSERT(st.tool.state == DSML_TOOL_BETWEEN_INVOKES);

    const char *raw_param =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_incomplete_tool", &st,
                                         raw_param, strlen(raw_param), false));
    TEST_ASSERT(st.mode == OPENAI_STREAM_TOOL);
    TEST_ASSERT(st.tool.state == DSML_TOOL_BETWEEN_PARAMS);

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "\"name\":\"bash\"") != NULL);
    TEST_ASSERT(strstr(out, PULSAR_PARAM_START) == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_openai_tool_stream_sends_partial_raw_arguments(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"edit\">\n"
        PULSAR_PARAM_START " name=\"edits\" string=\"false\">[1,2,3";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_raw_tool", &st,
                                         raw, strlen(raw), false));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"name\":\"edit\"") != NULL);
    TEST_ASSERT(strstr(out, "\\\"edits\\\":") != NULL);
    TEST_ASSERT(strstr(out, "\"arguments\":\"[1,2,3\"") != NULL);
    TEST_ASSERT(strstr(out, PULSAR_TOOL_CALLS_START) == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_openai_tool_stream_holds_partial_dsml_entities(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw_partial =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">echo &amp";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_entity_tool", &st,
                                         raw_partial, strlen(raw_partial), false));

    const char *raw_complete =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">echo &amp; done" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_entity_tool", &st,
                                         raw_complete, strlen(raw_complete), false));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"arguments\":\"echo \"") != NULL);
    TEST_ASSERT(strstr(out, "\"arguments\":\"& done\"") != NULL);
    TEST_ASSERT(strstr(out, "&amp") == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_openai_tool_stream_holds_partial_utf8_arguments(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char prefix[] =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"write\">\n"
        PULSAR_PARAM_START " name=\"content\" string=\"true\">flag ";
    const char suffix[] =
        " done" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END;
    const char flag_utf8[] = {(char)0xf0, (char)0x9f, (char)0x9a, (char)0xa9, 0};
    const char replacement[] = {(char)0xef, (char)0xbf, (char)0xbd, 0};

    buf partial = {0};
    buf_append(&partial, prefix, strlen(prefix));
    buf_putc(&partial, (char)0xf0);
    buf_putc(&partial, (char)0x9f);
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_utf8_tool", &st,
                                         partial.ptr, partial.len, false));

    buf complete = {0};
    buf_append(&complete, prefix, strlen(prefix));
    buf_append(&complete, flag_utf8, 4);
    buf_append(&complete, suffix, strlen(suffix));
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_utf8_tool", &st,
                                         complete.ptr, complete.len, false));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"arguments\":\"flag \"") != NULL);
    TEST_ASSERT(strstr(out, flag_utf8) != NULL);
    TEST_ASSERT(strstr(out, replacement) == NULL);

    free(out);
    buf_free(&partial);
    buf_free(&complete);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_openai_tool_stream_handles_multiple_calls(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"read\">\n"
        PULSAR_PARAM_START " name=\"path\" string=\"true\">a.c" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">wc -l a.c" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_multi_tool", &st,
                                         raw, strlen(raw), false));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    int tool_id_count = 0;
    for (const char *p = out; (p = strstr(p, "\"id\":\"call_")) != NULL; p++) tool_id_count++;
    TEST_ASSERT(tool_id_count == 2);
    TEST_ASSERT(strstr(out, "\"name\":\"read\"") != NULL);
    TEST_ASSERT(strstr(out, "\"name\":\"bash\"") != NULL);
    TEST_ASSERT(strstr(out, "\\\"path\\\":") != NULL);
    TEST_ASSERT(strstr(out, "\\\"command\\\":") != NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_streaming_holds_partial_utf8(void) {
    const char partial[] = {'A', ' ', (char)0xf0, (char)0x9f, 0};
    const char complete[] = {'A', ' ', (char)0xf0, (char)0x9f,
                             (char)0x9a, (char)0xa9, ' ', 'd', 'o', 'n', 'e', 0};
    const char flag_done[] = {(char)0xf0, (char)0x9f,
                              (char)0x9a, (char)0xa9, ' ', 'd', 'o', 'n', 'e', 0};
    const char replacement[] = {(char)0xef, (char)0xbf, (char)0xbd, 0};

    TEST_ASSERT(utf8_stream_safe_len(partial, 0, strlen(partial), false) == 2);
    TEST_ASSERT(utf8_stream_safe_len(complete, 0, strlen(complete), false) == strlen(complete));
    /* L187: byte-fallback tokens can emit bytes that are not UTF-8 at all.
     * The hold-back is lenient and these pin its boundaries so a strict
     * rewrite (which would release E0 80 early) is caught. */
    {
        const char e0_80[] = {'a', (char)0xe0, (char)0x80};          /* truncated 3-byte lead: HELD */
        const char c0_80[] = {'a', (char)0xc0, (char)0x80};          /* 0xC0 is not a lead: released */
        const char f5[]    = {'a', (char)0xf5, (char)0x80, (char)0x80}; /* 0xF5 is not a lead: released */
        const char lone[]  = {(char)0x80};                             /* lone continuation at start: held */
        const char c2[]    = {'a', (char)0xc2};                        /* bare 2-byte lead at the end: held */
        TEST_ASSERT(utf8_stream_safe_len(e0_80, 0, sizeof(e0_80), false) == 1);
        TEST_ASSERT(utf8_stream_safe_len(c0_80, 0, sizeof(c0_80), false) == sizeof(c0_80));
        TEST_ASSERT(utf8_stream_safe_len(f5, 0, sizeof(f5), false) == sizeof(f5));
        TEST_ASSERT(utf8_stream_safe_len(lone, 0, sizeof(lone), false) == 0);
        TEST_ASSERT(utf8_stream_safe_len(c2, 0, sizeof(c2), false) == 1);
        TEST_ASSERT(utf8_stream_safe_len(e0_80, 0, sizeof(e0_80), true) == sizeof(e0_80)); /* final flush releases */
    }

    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_NONE;

    openai_stream st;
    openai_stream_start(&r, &st);
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_utf8", &st,
                                         partial, strlen(partial), false));
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_utf8", &st,
                                         complete, strlen(complete), false));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"content\":\"A \"") != NULL);
    TEST_ASSERT(strstr(out, flag_done) != NULL);
    TEST_ASSERT(strstr(out, replacement) == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_request_defaults_use_min_p_filtering(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    TEST_ASSERT(r.think_mode == PULSAR_THINK_LOW);
    TEST_ASSERT(r.temperature == PULSAR_DEFAULT_TEMPERATURE);
    TEST_ASSERT(r.top_p == PULSAR_DEFAULT_TOP_P);
    TEST_ASSERT(r.top_k == 0);
    TEST_ASSERT(r.min_p == PULSAR_DEFAULT_MIN_P);
    TEST_ASSERT(!r.has_temperature);
    TEST_ASSERT(!r.has_top_k);
    TEST_ASSERT(!r.has_top_p);
    TEST_ASSERT(!r.has_min_p);
    request_free(&r);
}



static void check_resolved_sampling(const request *r, float want_temp,
                                    int want_top_k, float want_top_p,
                                    float want_min_p) {
    float temperature = -1.0f, top_p = -1.0f, min_p = -1.0f;
    int top_k = -1;
    gen_resolve_sampling(r, &temperature, &top_k, &top_p, &min_p);
    TEST_ASSERT(temperature == want_temp);
    TEST_ASSERT(top_k == want_top_k);
    TEST_ASSERT(top_p == want_top_p);
    TEST_ASSERT(min_p == want_min_p);
}

/* The sampling contract: engine defaults apply only to parameters the request
 * left absent; anything the client sent explicitly reaches the sampler as-is,
 * including values that happen to equal the defaults. Exercises
 * gen_resolve_sampling (generate.cpp) over the full matrix of
 * {absent, explicit-nondefault, explicit-equal-to-default} per parameter,
 * with thinking on and off. "Absent" is simulated exactly as the parser
 * leaves it: request_init's default value with the has_ flag false. */
static void test_think_sampling_respects_explicit_params(void) {
    const pulsar_think_mode modes[2] = {PULSAR_THINK_HIGH, PULSAR_THINK_NONE};
    for (int m = 0; m < 2; m++) {
        for (int p = 0; p < 4; p++) {      /* param under test */
            for (int st = 0; st < 3; st++) { /* 0 absent, 1 explicit-nondefault,
                                              * 2 explicit-equal-to-default */
                request r;
                request_init(&r, REQ_CHAT, 128);
                r.think_mode = modes[m];
                float want_temp = PULSAR_DEFAULT_TEMPERATURE;
                int want_top_k = 0;
                float want_top_p = PULSAR_DEFAULT_TOP_P;
                float want_min_p = PULSAR_DEFAULT_MIN_P;
                if (st != 0) {
                    switch (p) {
                    case 0:
                        r.has_temperature = true;
                        r.temperature = st == 1 ? 0.35f : PULSAR_DEFAULT_TEMPERATURE;
                        want_temp = r.temperature;
                        break;
                    case 1:
                        r.has_top_k = true;
                        r.top_k = st == 1 ? 40 : 0;
                        want_top_k = r.top_k;
                        break;
                    case 2:
                        r.has_top_p = true;
                        r.top_p = st == 1 ? 0.9f : PULSAR_DEFAULT_TOP_P;
                        want_top_p = r.top_p;
                        break;
                    case 3:
                        r.has_min_p = true;
                        r.min_p = st == 1 ? 0.0f : PULSAR_DEFAULT_MIN_P;
                        want_min_p = r.min_p;
                        break;
                    }
                }
                check_resolved_sampling(&r, want_temp, want_top_k,
                                        want_top_p, want_min_p);
                request_free(&r);
            }
        }

        /* Explicit temperature EQUAL to the default alongside explicit
         * non-default knobs: the case the old value-only check could not
         * express — with thinking on it clobbered all four; the explicit
         * knobs must survive. */
        request r;
        request_init(&r, REQ_CHAT, 128);
        r.think_mode = modes[m];
        r.temperature = PULSAR_DEFAULT_TEMPERATURE;
        r.has_temperature = true;
        r.top_k = 40;
        r.has_top_k = true;
        r.top_p = 0.9f;
        r.has_top_p = true;
        r.min_p = 0.0f;
        r.has_min_p = true;
        check_resolved_sampling(&r, PULSAR_DEFAULT_TEMPERATURE, 40, 0.9f, 0.0f);

        /* Explicit temperature==0 (others absent): greedy decode must reach
         * the sampler so DSpark speculative decode can engage. */
        request greedy;
        request_init(&greedy, REQ_CHAT, 128);
        greedy.think_mode = modes[m];
        greedy.temperature = 0.0f;
        greedy.has_temperature = true;
        check_resolved_sampling(&greedy, 0.0f, 0, PULSAR_DEFAULT_TOP_P,
                                PULSAR_DEFAULT_MIN_P);
        request_free(&greedy);
        request_free(&r);
    }
}



/* L116: gen_resolve_sampling_decode is the ONE sampling authority for every
 * decode lane (classic, plain-batched, spec-batched, mixed), and the tool
 * admission to the batched lanes rests on it forcing temperature=0 exactly in
 * tool-call structural regions — and nowhere else. Mutation check: dropping
 * the override in the helper fails the STRUCTURAL/JSON_STRUCTURAL rows;
 * over-forcing fails the payload-sampling and no-tools rows. */
static void test_decode_sampling_tool_payload_forcing(void) {
    job j;
    memset(&j, 0, sizeof j);
    request_init(&j.req, REQ_CHAT, 128);
    j.req.think_mode = PULSAR_THINK_NONE;
    j.req.temperature = 0.8f;
    j.req.has_temperature = true;
    gen_state g;
    memset(&g, 0, sizeof g);
    g.j = &j;
    dsml_decode_tracker_init(&g.dsml_tracker);
    static const struct { dsml_decode_state st; bool tools; float want; } cases[] = {
        {DSML_DECODE_OUTSIDE,         true,  0.8f},
        {DSML_DECODE_STRUCTURAL,      true,  0.0f},
        {DSML_DECODE_JSON_STRUCTURAL, true,  0.0f},
        {DSML_DECODE_STRING_BODY,     true,  0.8f}, /* payload sampling */
        {DSML_DECODE_JSON_STRING,     true,  0.8f}, /* payload sampling */
        {DSML_DECODE_STRUCTURAL,      false, 0.8f}, /* no tools: tracker ignored */
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        j.req.has_tools = cases[i].tools;
        g.dsml_tracker.decode = cases[i].st;
        float temperature = -1.0f, top_p = -1.0f, min_p = -1.0f;
        int top_k = -1;
        gen_resolve_sampling_decode(&g, &temperature, &top_k, &top_p, &min_p);
        TEST_ASSERT(temperature == cases[i].want);
    }
    request_free(&j.req);
}



static void test_web_search_tool_recognition(void) {
    int uses = 0;
    TEST_ASSERT(web_search_tool_entry(
        "{\"type\":\"web_search_20250305\",\"name\":\"web_search\",\"max_uses\":3}",
        &uses));
    TEST_ASSERT(uses == 3);
    uses = 0;
    TEST_ASSERT(web_search_tool_entry("{\"type\":\"web_search_20250305\"}", &uses));
    TEST_ASSERT(uses == WEB_SEARCH_DEFAULT_MAX_USES);
    TEST_ASSERT(!web_search_tool_entry(
        "{\"name\":\"web_search\",\"input_schema\":{\"type\":\"object\"}}", &uses));
    TEST_ASSERT(!web_search_tool_entry(
        "{\"type\":\"custom\",\"name\":\"web_search\"}", &uses));

    /* enabled: synthesized internal schema, order flagged server-executed */
    const char *tools = "[{\"type\":\"web_search_20250305\",\"name\":\"web_search\",\"max_uses\":5},"
                        "{\"name\":\"get_weather\",\"input_schema\":{\"type\":\"object\","
                        "\"properties\":{\"city\":{\"type\":\"string\"}}}}]";
    const char *tp = tools;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    int max_uses = 0;
    TEST_ASSERT(parse_tools_value(&tp, &schemas, &orders, true, &max_uses));
    TEST_ASSERT(max_uses == 5);
    TEST_ASSERT(strstr(schemas, "\"name\":\"web_search\"") != NULL);
    TEST_ASSERT(strstr(schemas, "\"query\"") != NULL);
    const tool_schema_order *ord = tool_schema_orders_find(&orders, "web_search");
    TEST_ASSERT(ord && ord->server_web_search);
    ord = tool_schema_orders_find(&orders, "get_weather");
    TEST_ASSERT(ord && !ord->server_web_search);
    free(schemas);
    tool_schema_orders_free(&orders);

    /* disabled: the entry is dropped entirely */
    tp = tools;
    schemas = NULL;
    tool_schema_orders orders2 = {0};
    max_uses = 0;
    TEST_ASSERT(parse_tools_value(&tp, &schemas, &orders2, false, &max_uses));
    TEST_ASSERT(max_uses == 0);
    TEST_ASSERT(strstr(schemas, "web_search") == NULL);
    TEST_ASSERT(tool_schema_orders_find(&orders2, "get_weather") != NULL);
    free(schemas);
    tool_schema_orders_free(&orders2);

    /* query extraction: schema key, plus lenient fallback */
    char *q = web_search_query_from_arguments("{\"query\":\"pulsar engine\"}");
    TEST_ASSERT(q && !strcmp(q, "pulsar engine"));
    free(q);
    q = web_search_query_from_arguments("{\"search_query\":\"fallback\"}");
    TEST_ASSERT(q && !strcmp(q, "fallback"));
    free(q);
}



static void test_web_search_result_replay_rebuild(void) {
    /* The chunk text rides in encrypted_content; replay joins the echoed
     * chunks with a blank line — byte-identical to the live tool_result. */
    const char *content =
        "[{\"type\":\"web_search_result\",\"url\":\"https://a\",\"title\":\"A\","
        "\"encrypted_content\":\"Result:\\nTitle: A\\nURL: https://a\\nSnippet: alpha\","
        "\"page_age\":null},"
        "{\"type\":\"web_search_result\",\"url\":\"https://b\",\"title\":\"B\","
        "\"encrypted_content\":\"Result:\\nTitle: B\\nURL: https://b\\nSnippet: beta\","
        "\"page_age\":null}]";
    char *text = web_search_rebuild_result_text(content);
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(!strcmp(text,
        "Result:\nTitle: A\nURL: https://a\nSnippet: alpha\n\n"
        "Result:\nTitle: B\nURL: https://b\nSnippet: beta"));
    free(text);

    text = web_search_rebuild_result_text("[]");
    TEST_ASSERT(text && !strcmp(text, "No search results found."));
    free(text);

    text = web_search_rebuild_result_text(
        "{\"type\":\"web_search_tool_result_error\",\"error_code\":\"max_uses_exceeded\"}");
    TEST_ASSERT(text && !strcmp(text,
        "Web search failed: max_uses_exceeded. Answer from available information."));
    free(text);
}



static void test_web_search_replay_message_split(void) {
    /* One echoed assistant message holding a completed search round maps back
     * to three template turns: call, tool_result, continuation. */
    const char *messages =
        "[{\"role\":\"user\",\"content\":\"look it up\"},"
        "{\"role\":\"assistant\",\"content\":["
        "{\"type\":\"thinking\",\"thinking\":\"I should search.\",\"signature\":\"sig\"},"
        "{\"type\":\"server_tool_use\",\"id\":\"toolu_ws1\",\"name\":\"web_search\","
        "\"input\":{\"query\":\"pulsar\"}},"
        "{\"type\":\"web_search_tool_result\",\"tool_use_id\":\"toolu_ws1\",\"content\":"
        "[{\"type\":\"web_search_result\",\"url\":\"https://a\",\"title\":\"A\","
        "\"encrypted_content\":\"Result:\\nTitle: A\\nURL: https://a\\nSnippet: alpha\"}]},"
        "{\"type\":\"text\",\"text\":\"Found it.\"}]}]";
    const char *p = messages;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs));
    TEST_ASSERT(msgs.len == 4);
    TEST_ASSERT(!strcmp(msgs.v[1].role, "assistant"));
    TEST_ASSERT(msgs.v[1].calls.len == 1);
    TEST_ASSERT(!strcmp(msgs.v[1].calls.v[0].name, "web_search"));
    TEST_ASSERT(!strcmp(msgs.v[1].calls.v[0].id, "toolu_ws1"));
    TEST_ASSERT(msgs.v[1].reasoning && !strcmp(msgs.v[1].reasoning, "I should search."));
    TEST_ASSERT(!strcmp(msgs.v[2].role, "user"));
    TEST_ASSERT(msgs.v[2].tool_call_id && !strcmp(msgs.v[2].tool_call_id, "toolu_ws1"));
    TEST_ASSERT(strstr(msgs.v[2].content,
        "<tool_result>Result:\nTitle: A\nURL: https://a\nSnippet: alpha</tool_result>") != NULL);
    TEST_ASSERT(!strcmp(msgs.v[3].role, "assistant"));
    TEST_ASSERT(msgs.v[3].content && !strcmp(msgs.v[3].content, "Found it."));

    /* The rendered replay must reproduce the live continuation shape that
     * build_web_search_result_suffix appended (EOS + user tool_result + fresh
     * assistant think turn). */
    char *prompt = render_chat_prompt_text(&msgs, "{}", NULL, PULSAR_THINK_LOW);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt,
        "<｜end▁of▁sentence｜><｜User｜><tool_result>Result:\nTitle: A\nURL: https://a"
        "\nSnippet: alpha</tool_result><｜Assistant｜><think>") != NULL);
    request req;
    request_init(&req, REQ_CHAT, 128);
    req.think_mode = PULSAR_THINK_LOW;
    thinking_state th = {0};
    char *suffix = build_web_search_result_suffix(&req, &th,
        "Result:\nTitle: A\nURL: https://a\nSnippet: alpha");
    TEST_ASSERT(strstr(prompt, suffix) != NULL);
    free(suffix);
    request_free(&req);
    free(prompt);
    chat_msgs_free(&msgs);
}



static void test_reasoning_effort_mapping(void) {
    pulsar_think_mode mode = PULSAR_THINK_NONE;
    TEST_ASSERT(parse_reasoning_effort_name("minimal", &mode) && mode == PULSAR_THINK_LOW);
    TEST_ASSERT(parse_reasoning_effort_name("low", &mode) && mode == PULSAR_THINK_LOW);
    TEST_ASSERT(parse_reasoning_effort_name("medium", &mode) && mode == PULSAR_THINK_LOW);
    TEST_ASSERT(parse_reasoning_effort_name("high", &mode) && mode == PULSAR_THINK_HIGH);
    TEST_ASSERT(parse_reasoning_effort_name("xhigh", &mode) && mode == PULSAR_THINK_MAX);
    TEST_ASSERT(parse_reasoning_effort_name("max", &mode) && mode == PULSAR_THINK_MAX);
    TEST_ASSERT(!parse_reasoning_effort_name("banana", &mode));
    TEST_ASSERT(pulsar_think_mode_for_context(PULSAR_THINK_MAX, 32768) == PULSAR_THINK_LOW);
    TEST_ASSERT(pulsar_think_mode_for_context(PULSAR_THINK_HIGH, 32768) == PULSAR_THINK_LOW);
    TEST_ASSERT(pulsar_think_mode_for_context(PULSAR_THINK_LOW, 32768) == PULSAR_THINK_LOW);
    TEST_ASSERT(pulsar_think_mode_for_context(PULSAR_THINK_MAX,
                                           (int)pulsar_think_max_min_context()) == PULSAR_THINK_MAX);
    TEST_ASSERT(pulsar_think_mode_for_context(PULSAR_THINK_HIGH,
                                           (int)pulsar_think_max_min_context()) == PULSAR_THINK_HIGH);
    /* The three prefixes are distinct: low empty, high/max non-empty and
     * different texts (0731 restructure). */
    TEST_ASSERT(!pulsar_think_effort_prefix(PULSAR_THINK_LOW)[0]);
    TEST_ASSERT(pulsar_think_effort_prefix(PULSAR_THINK_HIGH)[0]);
    TEST_ASSERT(pulsar_think_effort_prefix(PULSAR_THINK_MAX)[0]);
    TEST_ASSERT(strcmp(pulsar_think_effort_prefix(PULSAR_THINK_HIGH),
                       pulsar_think_effort_prefix(PULSAR_THINK_MAX)) != 0);
}



static void test_api_thinking_controls_parse(void) {
    bool enabled = true;
    const char *thinking = "{\"type\":\"disabled\",\"budget_tokens\":1024}";
    TEST_ASSERT(parse_thinking_control_value(&thinking, &enabled));
    TEST_ASSERT(!enabled);
    thinking = "true";
    TEST_ASSERT(parse_thinking_control_value(&thinking, &enabled));
    TEST_ASSERT(enabled);

    pulsar_think_mode mode = PULSAR_THINK_HIGH;
    const char *anth_effort = "{\"effort\":\"max\",\"other\":true}";
    TEST_ASSERT(parse_output_config_effort(&anth_effort, &mode));
    TEST_ASSERT(mode == PULSAR_THINK_MAX);

    const char *openai_effort = "\"xhigh\"";
    mode = PULSAR_THINK_HIGH;
    TEST_ASSERT(parse_reasoning_effort_value(&openai_effort, &mode));
    TEST_ASSERT(mode == PULSAR_THINK_MAX);

    const char *low_effort = "\"low\"";
    mode = PULSAR_THINK_HIGH;
    TEST_ASSERT(parse_reasoning_effort_value(&low_effort, &mode));
    TEST_ASSERT(mode == PULSAR_THINK_LOW);
}



static void test_render_think_max_prompt_prefix(void) {
    chat_msgs msgs = {0};
    chat_msg sys = {0};
    sys.role = xstrdup("system");
    sys.content = xstrdup("You are terse.");
    chat_msgs_push(&msgs, sys);
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("Hello");
    chat_msgs_push(&msgs, user);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_MAX);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(!strncmp(prompt, "<｜begin▁of▁sentence｜>", strlen("<｜begin▁of▁sentence｜>")));
    TEST_ASSERT(strstr(prompt, pulsar_think_max_prefix()) != NULL);
    TEST_ASSERT(strstr(prompt, "You are terse.<｜User｜>Hello<｜Assistant｜><think>") != NULL);
    TEST_ASSERT(strstr(prompt, "</think>") == NULL);

    free(prompt);
    chat_msgs_free(&msgs);
}



/* vLLM PR #44283's bug class (inline role:system accepted) + L113 placement
 * (2026-08-25): the LEADING run of system messages joins the system region;
 * a system message arriving MID-conversation renders IN PLACE as a
 * <system-reminder> environment note. Consolidating mid-stream system
 * messages into the region is what capped every warm-fork at the
 * scaffolding: agent clients append one system-role nudge per turn, and
 * teleporting it to the top shifted the whole rendered prefix. */
static void test_inline_system_message_placement(void) {
    const char *messages =
        "[{\"role\":\"system\",\"content\":\"You are terse.\"},"
        "{\"role\":\"user\",\"content\":\"Hello\"},"
        "{\"role\":\"system\",\"content\":\"Prefer bullet lists.\"}]";
    const char *p = messages;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs));
    TEST_ASSERT(msgs.len == 3);
    TEST_ASSERT(!strcmp(msgs.v[0].role, "system"));
    TEST_ASSERT(!strcmp(msgs.v[2].role, "system"));
    TEST_ASSERT(!msgs.v[0].system_field && !msgs.v[2].system_field);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_LOW);
    TEST_ASSERT(prompt != NULL);
    /* Leading system text is in the region BEFORE the first user turn; the
     * mid-conversation one renders after that turn, in place, wrapped. */
    const char *sys_lead = strstr(prompt, "You are terse.");
    const char *user_turn = strstr(prompt, "<｜User｜>Hello");
    const char *in_place = strstr(prompt,
        "<｜User｜><system-reminder>\nPrefer bullet lists.\n</system-reminder>");
    TEST_ASSERT(sys_lead != NULL);
    TEST_ASSERT(user_turn != NULL);
    TEST_ASSERT(in_place != NULL);
    TEST_ASSERT(sys_lead < user_turn);
    TEST_ASSERT(user_turn < in_place);
    /* Nothing from the nudge leaks into the region. */
    TEST_ASSERT(strstr(prompt, "Prefer bullet lists.") == in_place + strlen("<｜User｜><system-reminder>\n"));
    free(prompt);

    /* The top-level system FIELD is appended to the array by the parser --
     * trailing position, but system_field=true keeps it in the region. */
    chat_msg field = {0};
    field.role = xstrdup("system");
    field.content = xstrdup("Field prompt.");
    field.system_field = true;
    chat_msgs_push(&msgs, field);
    prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_LOW);
    TEST_ASSERT(prompt != NULL);
    const char *field_at = strstr(prompt, "Field prompt.");
    TEST_ASSERT(field_at != NULL);
    TEST_ASSERT(field_at < strstr(prompt, "<｜User｜>Hello"));
    free(prompt);
    chat_msgs_free(&msgs);
}


/* L113 regression pin: appending a system-role nudge plus a new user turn
 * must EXTEND the previous render, never rewrite it. render(N) minus its
 * dangling assistant prefix must be a byte prefix of render(N+1). This is
 * the property whose absence cost every warm-fork past the scaffolding. */
static void test_appended_system_message_keeps_prefix(void) {
    chat_msgs msgs = {0};
    chat_msg m0 = {0};
    m0.role = xstrdup("user");
    m0.content = xstrdup("First question");
    chat_msgs_push(&msgs, m0);
    chat_msg m1 = {0};
    m1.role = xstrdup("assistant");
    m1.content = xstrdup("First answer");
    chat_msgs_push(&msgs, m1);

    /* Tool context, like the agent clients this pins: without it the
     * renderer collapses OLDER assistant think blocks relative to
     * last_user_idx, which is its own (pre-existing, checkpoint-matched)
     * prefix instability and not what this test is about. */
    const char *schemas = "{\"name\":\"noop\"}";
    char *before = render_chat_prompt_text(&msgs, schemas, NULL, PULSAR_THINK_LOW);
    TEST_ASSERT(before != NULL);

    chat_msg nudge = {0};
    nudge.role = xstrdup("system");
    nudge.content = xstrdup("The task tools haven't been used recently.");
    chat_msgs_push(&msgs, nudge);
    chat_msg m2 = {0};
    m2.role = xstrdup("user");
    m2.content = xstrdup("Second question");
    chat_msgs_push(&msgs, m2);

    char *after = render_chat_prompt_text(&msgs, schemas, NULL, PULSAR_THINK_LOW);
    TEST_ASSERT(after != NULL);

    /* Strip render(N)'s dangling assistant prefix if present, then require
     * byte-prefix containment. (This history ends in a completed assistant
     * turn, so there is no dangling prefix -- assert containment directly,
     * and keep the strip logic exercised via the length check.) */
    size_t blen = strlen(before);
    TEST_ASSERT(strlen(after) > blen);
    TEST_ASSERT(strncmp(before, after, blen) == 0);
    /* And the nudge itself sits in place, after the first answer. */
    const char *in_place = strstr(after, "<｜User｜><system-reminder>\n"
                                  "The task tools haven't been used recently.");
    TEST_ASSERT(in_place != NULL);
    TEST_ASSERT(in_place >= after + blen - strlen("<｜Assistant｜>"));
    free(before);
    free(after);
    chat_msgs_free(&msgs);
}


/* 0731 effort levels: HIGH renders its own (distinct) prefix, LOW renders
 * none — a LOW prompt is byte-identical to the pre-0731 default rendering. */
static void test_render_think_effort_prefixes(void) {
    chat_msgs msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("Hello");
    chat_msgs_push(&msgs, user);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, pulsar_think_effort_prefix(PULSAR_THINK_HIGH)) != NULL);
    TEST_ASSERT(strstr(prompt, pulsar_think_max_prefix()) == NULL);
    TEST_ASSERT(strstr(prompt, "<｜User｜>Hello<｜Assistant｜><think>") != NULL);
    free(prompt);

    prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_LOW);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "Reasoning Effort:") == NULL);
    TEST_ASSERT(strstr(prompt, "<｜User｜>Hello<｜Assistant｜><think>") != NULL);
    free(prompt);
    chat_msgs_free(&msgs);
}



static void test_render_non_thinking_prompt_closes_think(void) {
    chat_msgs msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("Hello");
    chat_msgs_push(&msgs, user);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_NONE);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, pulsar_think_max_prefix()) == NULL);
    TEST_ASSERT(strstr(prompt, "<｜User｜>Hello<｜Assistant｜></think>") != NULL);
    free(prompt);
    chat_msgs_free(&msgs);
}



static void test_render_drops_old_reasoning_without_tools(void) {
    chat_msgs msgs = {0};
    chat_msg user1 = {0};
    user1.role = xstrdup("user");
    user1.content = xstrdup("first");
    chat_msgs_push(&msgs, user1);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup("old hidden reasoning");
    assistant.content = xstrdup("first answer");
    chat_msgs_push(&msgs, assistant);
    chat_msg user2 = {0};
    user2.role = xstrdup("user");
    user2.content = xstrdup("second");
    chat_msgs_push(&msgs, user2);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "old hidden reasoning") == NULL);
    TEST_ASSERT(strstr(prompt, "<｜Assistant｜></think>first answer") != NULL);
    TEST_ASSERT(strstr(prompt, "<｜User｜>second<｜Assistant｜><think>") != NULL);

    free(prompt);
    chat_msgs_free(&msgs);
}



static void test_render_preserves_reasoning_with_tools(void) {
    chat_msgs msgs = {0};
    chat_msg user1 = {0};
    user1.role = xstrdup("user");
    user1.content = xstrdup("first");
    chat_msgs_push(&msgs, user1);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup("tool reasoning");
    assistant.content = xstrdup("");
    tool_call tc = {0};
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);
    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.content = xstrdup("/tmp");
    chat_msgs_push(&msgs, tool);

    char *prompt = render_chat_prompt_text(&msgs, "{}", NULL, PULSAR_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "<think>tool reasoning</think>") != NULL);
    TEST_ASSERT(strstr(prompt, "<tool_result>/tmp</tool_result>") != NULL);
    free(prompt);

    prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "<think>tool reasoning</think>") != NULL);
    TEST_ASSERT(strstr(prompt, "<tool_result>/tmp</tool_result>") != NULL);

    free(prompt);
    chat_msgs_free(&msgs);
}



static void test_render_chat_prompt_text_renders_tools_before_system(void) {
    /* The tool-schema block must sit at the head of the system region so the
     * client's system content stays at the tail, right before <｜User｜>.
     * That keeps a per-request dynamic tail (e.g. a timestamp) out of the
     * cached prefix without losing the tool schemas to the trim. */
    chat_msgs msgs = {0};
    chat_msg sys = {0};
    sys.role = xstrdup("system");
    sys.content = xstrdup("CLIENT_SYSTEM_MARKER");
    chat_msgs_push(&msgs, sys);
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("hello");
    chat_msgs_push(&msgs, user);

    char *prompt = render_chat_prompt_text(&msgs, "TOOL_SCHEMA_MARKER", NULL,
                                           PULSAR_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    const char *tools  = strstr(prompt, "## Tools");
    const char *client = strstr(prompt, "CLIENT_SYSTEM_MARKER");
    const char *user_m = strstr(prompt, "<｜User｜>");
    TEST_ASSERT(tools && client && user_m);
    TEST_ASSERT(tools  < client);
    TEST_ASSERT(client < user_m);
    free(prompt);
    chat_msgs_free(&msgs);
}



static void test_dsml_tool_args_preserve_call_order(void) {
    tool_calls calls = make_swapped_bash_call();
    buf b = {0};
    append_dsml_tool_calls_text(&b, &calls);
    const char *command = strstr(b.ptr, "name=\"command\"");
    const char *description = strstr(b.ptr, "name=\"description\"");
    const char *timeout = strstr(b.ptr, "name=\"timeout\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(timeout != NULL);
    TEST_ASSERT(description < command);
    TEST_ASSERT(command < timeout);
    buf_free(&b);
    tool_calls_free(&calls);
}



static void test_openai_tool_args_preserve_call_order(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.tool_orders = make_bash_order();
    tool_calls calls = make_swapped_bash_call();
    buf b = {0};
    append_tool_calls_json(&b, &calls, "test", &r.tool_orders);
    const char *command = strstr(b.ptr, "\\\"command\\\"");
    const char *description = strstr(b.ptr, "\\\"description\\\"");
    const char *timeout = strstr(b.ptr, "\\\"timeout\\\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(timeout != NULL);
    TEST_ASSERT(description < command);
    TEST_ASSERT(command < timeout);
    buf_free(&b);
    tool_calls_free(&calls);
    request_free(&r);
}



static void test_anthropic_thinking_and_tool_args_preserve_call_order(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.tool_orders = make_bash_order();
    tool_calls calls = make_swapped_bash_call();
    buf b = {0};
    append_anthropic_content(&b, "done", "thinking text", &calls, "msg_1", &r.tool_orders, NULL);
    const char *thinking = strstr(b.ptr, "\"type\":\"thinking\"");
    const char *text = strstr(b.ptr, "\"type\":\"text\"");
    const char *tool = strstr(b.ptr, "\"type\":\"tool_use\"");
    const char *command = strstr(b.ptr, "\"command\"");
    const char *description = strstr(b.ptr, "\"description\"");
    TEST_ASSERT(thinking != NULL);
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(thinking < text);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(description < command);
    buf_free(&b);
    tool_calls_free(&calls);
    request_free(&r);
}



/* L196: a checkpoint key for a tool-call turn is the replay minus the EOS the
 * replay renders after the tool_calls block (the sampled tokens stop there). */
static void assert_replay_is_key_plus_eos(const char *key, const char *replay) {
    const size_t klen = strlen(key);
    TEST_ASSERT(strncmp(replay, key, klen) == 0);
    TEST_ASSERT(!strcmp(replay + klen, "<｜end▁of▁sentence｜>"));
}



static void test_checkpoint_key_ends_where_sampled_tokens_end(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = PULSAR_THINK_HIGH;
    r.tool_orders = make_bash_order();
    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_1");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"ls\"}");
    tool_calls_push(&calls, tc);
    /* a tool-call turn: no EOS in the key, the replay has one */
    char *key = build_tool_checkpoint_suffix(&r, "", "need ls", &calls);
    buf replay = {0};
    append_assistant_turn_close(&replay, true, "need ls", "", &calls);
    assert_replay_is_key_plus_eos(key, replay.ptr);
    free(key);
    buf_free(&replay);
    /* a stop turn sampled its EOS: the key carries it and equals the replay */
    key = build_tool_checkpoint_suffix(&r, "done", "", NULL);
    append_assistant_turn_close(&replay, true, "", "done", NULL);
    TEST_ASSERT(!strcmp(key, replay.ptr));
    TEST_ASSERT(strstr(key, "<｜end▁of▁sentence｜>") != NULL);
    free(key);
    buf_free(&replay);
    tool_calls_free(&calls);
    request_free(&r);
}



static void test_parse_short_dsml_and_canonical_suffix(void) {
    const char *generated =
        "<think>need a tool</think>"
        "<DSML｜tool_calls>\n"
        "<DSML｜invoke name=\"bash\">\n"
        "<DSML｜parameter name=\"description\" string=\"true\">list files</DSML｜parameter>\n"
        "<DSML｜parameter name=\"command\" string=\"true\">ls -la</DSML｜parameter>\n"
        "</DSML｜invoke>\n"
        "</DSML｜tool_calls>";
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(reasoning && !strcmp(reasoning, "need a tool"));
    TEST_ASSERT(content && content[0] == '\0');
    TEST_ASSERT(calls.len == 1);

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = PULSAR_THINK_HIGH;
    r.tool_orders = make_bash_order();
    char *suffix = build_tool_checkpoint_suffix(&r, content, reasoning, &calls);
    const char *command = strstr(suffix, "name=\"command\"");
    const char *description = strstr(suffix, "name=\"description\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(description < command);
    TEST_ASSERT(strstr(suffix, "</think>") != NULL);
    /* L196: the turn stopped at the closing tool_calls tag; no EOS was sampled,
     * so the key carries none (the tail renders it). */
    TEST_ASSERT(strstr(suffix, "<｜end▁of▁sentence｜>") == NULL);
    TEST_ASSERT(!strcmp(suffix + strlen(suffix) - strlen("tool_calls>"), "tool_calls>"));

    free(suffix);
    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    request_free(&r);
}



static void test_dsml_parser_recovers_loose_nested_parameters(void) {
    const char *generated =
        "review done\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"edit\">\n"
        PULSAR_PARAM_START " name=\"path\">/private/tmp/tetris.c" PULSAR_PARAM_END "\n"
        PULSAR_PARAM_START " name=\"edits\">\n"
        PULSAR_PARAM_START " name=\"oldText\" string=\"true\">old &lt;text&gt;" PULSAR_PARAM_END "\n"
        PULSAR_PARAM_START " name=\"newText\" string=\"true\">new text" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END;

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(content && !strcmp(content, "review done"));
    TEST_ASSERT(calls.len == 1);
    TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "edit"));
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"path\": \"/private/tmp/tetris.c\"") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"edits\": {") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"oldText\":\"old <text>\"") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"newText\":\"new text\"") != NULL);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}



/* Verify that try_repair_dsml + parse_generated_message produces structurally
   valid tool calls for all three DSML styles and multiple truncation scenarios.
   Balanced but malformed DSML is not repaired: the model must retry it.
   This tests repair ACCURACY, not just that it doesn't crash. */
static void test_dsml_repair_produces_parseable_calls(void) {
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    buf repaired = {0};

    /* === TEST 1: Full DSML - missing </tool_calls> === */
    {
        const char *broken =
            "thinking done\n\n"
            PULSAR_TOOL_CALLS_START "\n"
            PULSAR_INVOKE_START " name=\"bash\">\n"
            PULSAR_PARAM_START " name=\"command\" string=\"true\">ls -la" PULSAR_PARAM_END "\n"
            PULSAR_INVOKE_END "\n";
        /* Missing: PULSAR_TOOL_CALLS_END */

        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken, strlen(broken), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, false, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "bash"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"command\": \"ls -la\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    /* === TEST 2: Full DSML - missing </invoke> and </tool_calls> === */
    {
        const char *broken =
            "\n\n"
            PULSAR_TOOL_CALLS_START "\n"
            PULSAR_INVOKE_START " name=\"edit\">\n"
            PULSAR_PARAM_START " name=\"path\" string=\"true\">/tmp/test.c" PULSAR_PARAM_END "\n";
        /* Missing: PULSAR_INVOKE_END, PULSAR_TOOL_CALLS_END */

        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken, strlen(broken), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, false, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "edit"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"path\": \"/tmp/test.c\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    /* === TEST 3: Full DSML - missing </parameter> === */
    {
        const char *broken =
            "\n\n"
            PULSAR_TOOL_CALLS_START "\n"
            PULSAR_INVOKE_START " name=\"bash\">\n"
            PULSAR_PARAM_START " name=\"command\" string=\"true\">echo hello";
        /* Missing: PULSAR_PARAM_END, PULSAR_INVOKE_END, PULSAR_TOOL_CALLS_END */

        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken, strlen(broken), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, false, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "bash"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"command\": \"echo hello\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    /* === TEST 4: Short DSML - missing closing tags === */
    {
        const char *broken =
            "\n\n"
            PULSAR_TOOL_CALLS_START_SHORT "\n"
            PULSAR_INVOKE_START_SHORT " name=\"write_file\">\n"
            PULSAR_PARAM_START_SHORT " name=\"path\" string=\"true\">/tmp/out.txt" PULSAR_PARAM_END_SHORT "\n"
            PULSAR_PARAM_START_SHORT " name=\"content\" string=\"true\">hello world" PULSAR_PARAM_END_SHORT "\n"
            PULSAR_INVOKE_END_SHORT "\n";
        /* Missing: PULSAR_TOOL_CALLS_END_SHORT */

        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken, strlen(broken), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, false, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "write_file"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"path\": \"/tmp/out.txt\"") != NULL);
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"content\": \"hello world\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    /* === TEST 5: Plain XML - missing closing tags === */
    {
        const char *broken =
            "\n\n"
            "<tool_calls>\n"
            "<invoke name=\"execute_command\">\n"
            "<parameter name=\"command\" string=\"true\">pwd</parameter>\n"
            "</invoke>\n";
        /* Missing: </tool_calls> */

        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken, strlen(broken), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, false, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "execute_command"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"command\": \"pwd\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    /* === TEST 6: Balanced text should NOT be modified === */
    {
        const char *balanced =
            "\n\n"
            PULSAR_TOOL_CALLS_START "\n"
            PULSAR_INVOKE_START " name=\"bash\">\n"
            PULSAR_PARAM_START " name=\"command\" string=\"true\">ls" PULSAR_PARAM_END "\n"
            PULSAR_INVOKE_END "\n"
            PULSAR_TOOL_CALLS_END;

        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(balanced, strlen(balanced), &repaired));
        /* No repair needed */
    }

    /* === TEST 7: No DSML tags should return false === */
    {
        const char *no_dsml = "just plain text, no tools";
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(no_dsml, strlen(no_dsml), &repaired));
    }

    /* === TEST 8: Balanced DSML with no invoke is not repaired === */
    {
        const char *balanced_no_invoke =
            "Let me analyze this.\n\n"
            PULSAR_TOOL_CALLS_START
            "The write tool truncates this too, at what looks like the same content location."
            PULSAR_TOOL_CALLS_END;
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(balanced_no_invoke, strlen(balanced_no_invoke), &repaired));
    }

    /* === TEST 9: Balanced short DSML with no invoke is not repaired === */
    {
        const char *balanced_short_no_invoke =
            "thinking...\n\n"
            PULSAR_TOOL_CALLS_START_SHORT
            "some content here"
            PULSAR_TOOL_CALLS_END_SHORT;
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(balanced_short_no_invoke, strlen(balanced_short_no_invoke), &repaired));
    }

    /* === TEST 10: Balanced plain XML DSML with no invoke is not repaired === */
    {
        const char *balanced_xml_no_invoke =
            "Let me think.\n\n"
            "<tool_calls>"
            "I need to use a tool but I don't know which one."
            "</tool_calls>";
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(balanced_xml_no_invoke, strlen(balanced_xml_no_invoke), &repaired));
    }

    /* === TEST 11: DSML mentioned inside thinking is not repaired === */
    {
        const char *thinking_quote =
            "<think>The protocol uses "
            PULSAR_TOOL_CALLS_START
            "some explanatory text"
            PULSAR_TOOL_CALLS_END
            ", but this is only a quote.</think>\nFinal answer.";
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(thinking_quote, strlen(thinking_quote), &repaired));
    }

    /* === TEST 12: Extra closing tags are unrecoverable, not truncation === */
    {
        const char *orphan_close =
            "done\n\n"
            PULSAR_TOOL_CALLS_START
            PULSAR_TOOL_CALLS_END
            PULSAR_TOOL_CALLS_END;
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(orphan_close, strlen(orphan_close), &repaired));
    }

    /* === TEST 13: Real DSML after thinking still repairs normally === */
    {
        const char *broken_after_think =
            "<think>"
            PULSAR_TOOL_CALLS_START
            "quoted DSML, not executable"
            PULSAR_TOOL_CALLS_END
            "</think>\n\n"
            PULSAR_TOOL_CALLS_START "\n"
            PULSAR_INVOKE_START " name=\"bash\">\n"
            PULSAR_PARAM_START " name=\"command\" string=\"true\">date" PULSAR_PARAM_END "\n"
            PULSAR_INVOKE_END "\n";
        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken_after_think, strlen(broken_after_think), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, true, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "bash"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"command\": \"date\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    buf_free(&repaired);
}



static void test_tool_parse_failure_returns_recoverable_finish(void) {
    const char *generated =
        "trying a tool\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START ">\n"
        PULSAR_TOOL_CALLS_END;

    char err[128] = {0};
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    const char *finish = "tool_calls";
    bool recovered = false;

    TEST_ASSERT(!parse_generated_message_for_response(generated,
                                                       true,
                                                       true,
                                                       false,
                                                       &finish,
                                                       err,
                                                       sizeof(err),
                                                       &content,
                                                       &reasoning,
                                                       &calls,
                                                       &recovered));
    TEST_ASSERT(recovered);
    TEST_ASSERT(!strcmp(finish, "stop"));
    TEST_ASSERT(!strcmp(err, "invalid tool call"));
    TEST_ASSERT(content && strstr(content, PULSAR_TOOL_CALLS_START) != NULL);
    TEST_ASSERT(reasoning == NULL);
    TEST_ASSERT(calls.len == 0);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}



static void test_invalid_dsml_tool_error_suffix_includes_system_prompt(void) {
    request r = {};
    r.think_mode = PULSAR_THINK_HIGH;
    r.prompt_text = xstrdup(
        "<｜begin▁of▁sentence｜>"
        "## Tools\nschema\n\nSystem rule\n\n"
        "<｜User｜>Hi<｜Assistant｜><think>");
    thinking_state st = {.inside = true};

    char *suffix = build_invalid_dsml_tool_error_suffix(&r, &st, "missing invoke name");
    TEST_ASSERT(suffix != NULL);
    TEST_ASSERT(strstr(suffix, "</think><｜end▁of▁sentence｜><｜User｜><tool_result>") == suffix);
    TEST_ASSERT(strstr(suffix, "Tool error: invalid DSML tool call: missing invoke name") != NULL);
    TEST_ASSERT(strstr(suffix, "The previous assistant output was not executed") != NULL);
    TEST_ASSERT(strstr(suffix, "System prompt reminder:\n## Tools\nschema\n\nSystem rule") != NULL);
    TEST_ASSERT(strstr(suffix, "<｜User｜>Hi") == NULL);
    TEST_ASSERT(strstr(suffix, "</tool_result><｜Assistant｜><think>") != NULL);

    free(suffix);
    free(r.prompt_text);
}



static void test_thinking_dsml_is_not_executable_before_think_close(void) {
    const char *generated =
        "<think>I might mention a malformed or tentative tool call here:\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">true" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END
        "\nBut it is still reasoning, not an assistant action.</think>Final answer.";

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, true,
                                           &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 0);
    TEST_ASSERT(reasoning && strstr(reasoning, PULSAR_TOOL_CALLS_START) != NULL);
    TEST_ASSERT(content && !strcmp(content, "Final answer."));

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}



static void test_thinking_dsml_after_think_close_is_executable(void) {
    const char *generated =
        "<think>need a shell check</think>\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">pwd" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END;

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, true,
                                           &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    TEST_ASSERT(reasoning && !strcmp(reasoning, "need a shell check"));
    TEST_ASSERT(content && content[0] == '\0');
    TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "bash"));
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"command\": \"pwd\"") != NULL);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}



static void test_tool_checkpoint_suffix_is_future_prompt_canonical(void) {
    tool_schema_orders orders = make_bash_order();
    const char *tool_schemas =
        "{\"name\":\"bash\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"command\":{},\"description\":{},\"timeout\":{}}}}";

    chat_msgs prefix_msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("inspect");
    chat_msgs_push(&prefix_msgs, user);
    char *prompt_text = render_chat_prompt_text(&prefix_msgs, tool_schemas,
                                                &orders, PULSAR_THINK_HIGH);

    const char *generated =
        "need a tool</think>\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">cd /tmp && git diff 2>/dev/null</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"timeout\" string=\"false\">10</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    TEST_ASSERT(strstr(calls.v[0].arguments, "cd /tmp && git diff 2>/dev/null") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "&amp;&amp;") == NULL);

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = PULSAR_THINK_HIGH;
    r.tool_orders = orders;
    memset(&orders, 0, sizeof(orders));
    char *suffix = build_tool_checkpoint_suffix(&r, content, reasoning, &calls);
    TEST_ASSERT(strstr(suffix, "cd /tmp && git diff 2>/dev/null") != NULL);
    TEST_ASSERT(strstr(suffix, "&amp;&amp;") == NULL);
    TEST_ASSERT(strstr(suffix, "2&gt;/dev/null") == NULL);
    buf canonical = {0};
    buf_puts(&canonical, prompt_text);
    buf_puts(&canonical, suffix);

    chat_msgs history_msgs = {0};
    chat_msg user2 = {0};
    user2.role = xstrdup("user");
    user2.content = xstrdup("inspect");
    chat_msgs_push(&history_msgs, user2);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup(reasoning ? reasoning : "");
    assistant.content = xstrdup(content ? content : "");
    assistant.calls = calls;
    memset(&calls, 0, sizeof(calls));
    chat_msgs_push(&history_msgs, assistant);
    char *future_prompt = render_chat_prompt_text(&history_msgs, tool_schemas,
                                                  &r.tool_orders, PULSAR_THINK_HIGH);

    assert_replay_is_key_plus_eos(canonical.ptr, future_prompt);   /* L196 */

    free(future_prompt);
    buf_free(&canonical);
    free(suffix);
    free(prompt_text);
    free(content);
    free(reasoning);
    chat_msgs_free(&history_msgs);
    chat_msgs_free(&prefix_msgs);
    tool_calls_free(&calls);
    request_free(&r);
    tool_schema_orders_free(&orders);
}



static void test_tool_checkpoint_minifies_json_parameters(void) {
    tool_schema_orders orders = {0};
    tool_schema_orders_add_json(&orders,
        "{\"name\":\"edit\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"path\":{},\"edits\":{}}}}");
    const char *tool_schemas =
        "{\"name\":\"edit\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"path\":{},\"edits\":{}}}}";

    chat_msgs prefix_msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("edit");
    chat_msgs_push(&prefix_msgs, user);
    char *prompt_text = render_chat_prompt_text(&prefix_msgs, tool_schemas,
                                                &orders, PULSAR_THINK_HIGH);

    const char *generated =
        "need edit</think>\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"edit\">\n"
        "<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/file</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"edits\" string=\"false\">"
        "[{\"oldText\": \"status=created\", \"newText\": \"status=created\\nstatus2=resumed\"}]"
        "</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 1);

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = PULSAR_THINK_HIGH;
    r.tool_orders = orders;
    memset(&orders, 0, sizeof(orders));
    char *suffix = build_tool_checkpoint_suffix(&r, content, reasoning, &calls);
    buf canonical = {0};
    buf_puts(&canonical, prompt_text);
    buf_puts(&canonical, suffix);

    chat_msgs history_msgs = {0};
    chat_msg user2 = {0};
    user2.role = xstrdup("user");
    user2.content = xstrdup("edit");
    chat_msgs_push(&history_msgs, user2);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup(reasoning ? reasoning : "");
    assistant.content = xstrdup(content ? content : "");
    assistant.calls = calls;
    memset(&calls, 0, sizeof(calls));
    chat_msgs_push(&history_msgs, assistant);
    char *future_prompt = render_chat_prompt_text(&history_msgs, tool_schemas,
                                                  &r.tool_orders, PULSAR_THINK_HIGH);

    assert_replay_is_key_plus_eos(canonical.ptr, future_prompt);   /* L196 */

    free(future_prompt);
    buf_free(&canonical);
    free(suffix);
    free(prompt_text);
    free(content);
    free(reasoning);
    chat_msgs_free(&history_msgs);
    chat_msgs_free(&prefix_msgs);
    tool_calls_free(&calls);
    request_free(&r);
    tool_schema_orders_free(&orders);
}



static void test_tool_memory_replays_sampled_dsml(void) {
    const char *generated =
        "<think>need shell</think>\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"timeout\" string=\"false\">10</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"description\" string=\"true\">list files</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls sampled = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &sampled));
    TEST_ASSERT(sampled.len == 1);

    server s;
    memset(&s, 0, sizeof(s));
    pthread_mutex_init(&s.tool_mu, NULL);
    s.assign_tool_call_ids(&sampled, API_OPENAI);
    TEST_ASSERT(sampled.v[0].id != NULL);
    TEST_ASSERT(!strncmp(sampled.v[0].id, "call_", 5));
    s.tool_memory_remember(&sampled);

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup(reasoning ? reasoning : "");
    assistant.content = xstrdup(content ? content : "");
    tool_call tc = {0};
    tc.id = xstrdup(sampled.v[0].id);
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"description\":\"list files\",\"command\":\"ls -la\",\"timeout\":10}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    tool_replay_stats stats = {0};
    s.tool_memory_attach_to_messages(&msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml != NULL);
    TEST_ASSERT(stats.mem == 1);
    TEST_ASSERT(stats.disk == 0);
    TEST_ASSERT(stats.canonical == 0);
    TEST_ASSERT(stats.missing_ids == 0);
    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_HIGH);
    const char *command = strstr(prompt, "name=\"command\"");
    const char *timeout = strstr(prompt, "name=\"timeout\"");
    const char *description = strstr(prompt, "name=\"description\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(timeout != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(command < timeout);
    TEST_ASSERT(timeout < description);

    free(prompt);
    chat_msgs_free(&msgs);
    free(content);
    free(reasoning);
    tool_calls_free(&sampled);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}



static void test_anthropic_tool_memory_replays_sampled_dsml(void) {
    const char *sampled_dsml =
        "\n\n" PULSAR_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"Bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"description\" string=\"true\">list files</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        PULSAR_TOOL_CALLS_END;

    server s;
    memset(&s, 0, sizeof(s));
    pthread_mutex_init(&s.tool_mu, NULL);
    s.tool_memory_put("toolu_exact", sampled_dsml);

    const char *json =
        "["
        "{\"role\":\"assistant\",\"content\":["
        "{\"type\":\"tool_use\",\"id\":\"toolu_exact\",\"name\":\"Bash\","
        "\"input\":{\"description\":\"list files\",\"command\":\"ls -la\"}}"
        "]},"
        "{\"role\":\"user\",\"content\":["
        "{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_exact\",\"content\":\"ok\"}"
        "]}"
        "]";
    const char *p = json;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs));
    TEST_ASSERT(msgs.len == 2);
    TEST_ASSERT(msgs.v[1].tool_call_id && !strcmp(msgs.v[1].tool_call_id, "toolu_exact"));

    stop_list ids = {0};
    collect_tool_call_ids(&msgs, &ids);
    TEST_ASSERT(id_list_contains(&ids, "toolu_exact"));
    id_list_free(&ids);

    tool_replay_stats stats = {0};
    s.tool_memory_attach_to_messages(&msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml != NULL);
    TEST_ASSERT(stats.mem == 1);
    TEST_ASSERT(stats.canonical == 0);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_HIGH);
    const char *command = strstr(prompt, "name=\"command\"");
    const char *description = strstr(prompt, "name=\"description\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(command < description);

    free(prompt);
    chat_msgs_free(&msgs);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}



static void test_anthropic_live_tail_renders_tool_results_only(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_ANTHROPIC;
    r.think_mode = PULSAR_THINK_HIGH;

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("toolu_live");
    tc.name = xstrdup("Bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("<tool_result>/tmp</tool_result>");
    chat_msg_add_tool_call_id(&user, "toolu_live");
    chat_msgs_push(&msgs, user);

    /* Anthropic system text is parsed separately and appended to chat_msgs for
     * rendering.  The live-tail finder must ignore it when locating the final
     * tool_result run. */
    chat_msg system = {0};
    system.role = xstrdup("system");
    system.content = xstrdup("You are terse.");
    chat_msgs_push(&msgs, system);

    anthropic_prepare_live_continuation(&r, &msgs);
    TEST_ASSERT(r.anthropic_live_call_ids.len == 1);
    TEST_ASSERT(!strcmp(r.anthropic_live_call_ids.v[0], "toolu_live"));
    TEST_ASSERT(r.anthropic_live_suffix_text != NULL);
    TEST_ASSERT(!strncmp(r.anthropic_live_suffix_text,
                         "<｜end▁of▁sentence｜><｜User｜><tool_result>",
                         strlen("<｜end▁of▁sentence｜><｜User｜><tool_result>")));
    TEST_ASSERT(strstr(r.anthropic_live_suffix_text, "/tmp</tool_result>") != NULL);
    TEST_ASSERT(strstr(r.anthropic_live_suffix_text, "<｜Assistant｜><think>") != NULL);
    TEST_ASSERT(strstr(r.anthropic_live_suffix_text, "Bash") == NULL);

    chat_msgs_free(&msgs);
    request_free(&r);
}



static void test_anthropic_tool_result_id_validation(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("<tool_result>out</tool_result>");
    chat_msg_add_tool_call_id(&user, "toolu_missing");
    chat_msgs_push(&msgs, user);

    char err[160] = {0};
    TEST_ASSERT(!s.anthropic_validate_tool_results(&msgs, NULL,
                                                 err, sizeof(err)));
    TEST_ASSERT(strstr(err, "Anthropic continuation state is not available") != NULL);

    pthread_mutex_lock(&s.tool_mu);
    s.n_slots = 1; /* live bindings are per-slot; has_call_id scans slots */
    s.slots[0].anthropic_live.valid = true;
    s.slots[0].anthropic_live.live_tokens = 10;
    id_list_push_unique(&s.slots[0].anthropic_live.call_ids, "toolu_missing");
    pthread_mutex_unlock(&s.tool_mu);
    bool needs_live_tool_state = false;
    err[0] = '\0';
    TEST_ASSERT(s.anthropic_validate_tool_results(&msgs,
                                                &needs_live_tool_state,
                                                err, sizeof(err)));
    TEST_ASSERT(needs_live_tool_state);

    chat_msgs_free(&msgs);
    live_tool_state_free(&s.slots[0].anthropic_live);
    pthread_mutex_destroy(&s.tool_mu);
}



static void test_anthropic_full_replay_allows_unknown_live_id(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("toolu_replay");
    tc.name = xstrdup("Bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("<tool_result>/tmp</tool_result>");
    chat_msg_add_tool_call_id(&user, "toolu_replay");
    chat_msgs_push(&msgs, user);

    bool needs_live_tool_state = false;
    char err[160] = {0};
    TEST_ASSERT(s.anthropic_validate_tool_results(&msgs,
                                                &needs_live_tool_state,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);

    chat_msgs_free(&msgs);
    pthread_mutex_destroy(&s.tool_mu);
}



static void test_anthropic_tool_use_parses_before_role(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    /* GitHub #127 regression: Crush can replay full Anthropic history with
     * message objects serialized as {"content": ..., "role": ...}.  The parser
     * must still remember prior assistant tool_use ids, otherwise old
     * tool_result blocks are mistaken for live-only continuations and rejected
     * once the live frontier has moved on to newer tool calls. */
    pthread_mutex_lock(&s.tool_mu);
    s.n_slots = 1;
    s.slots[0].anthropic_live.valid = true;
    s.slots[0].anthropic_live.live_tokens = 100;
    id_list_push_unique(&s.slots[0].anthropic_live.call_ids, "toolu_current");
    pthread_mutex_unlock(&s.tool_mu);

    const char *json =
        "["
        "{\"content\":["
        "{\"type\":\"tool_use\",\"id\":\"toolu_old\",\"name\":\"Bash\","
        "\"input\":{\"command\":\"ls\"}}"
        "],\"role\":\"assistant\"},"
        "{\"role\":\"user\",\"content\":["
        "{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_old\",\"content\":\"ok\"}"
        "]},"
        "{\"role\":\"user\",\"content\":\"continue\"}"
        "]";
    const char *p = json;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs));
    TEST_ASSERT(msgs.len == 3);
    TEST_ASSERT(msgs.v[0].calls.len == 1);
    TEST_ASSERT(msgs.v[0].calls.v[0].id &&
                !strcmp(msgs.v[0].calls.v[0].id, "toolu_old"));

    bool needs_live_tool_state = false;
    char err[160] = {0};
    TEST_ASSERT(s.anthropic_validate_tool_results(&msgs,
                                                &needs_live_tool_state,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);

    chat_msgs_free(&msgs);
    live_tool_state_free(&s.slots[0].anthropic_live);
    pthread_mutex_destroy(&s.tool_mu);
}



static void test_tool_checkpoint_canonicalization_gate_exact_replay(void) {
    server s;
    memset(&s, 0, sizeof(s));

    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_exact");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{}");
    tool_calls_push(&calls, tc);
    calls.raw_dsml = xstrdup(
        "\n\n" PULSAR_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "</｜DSML｜invoke>\n"
        PULSAR_TOOL_CALLS_END);

    TEST_ASSERT(!s.should_canonicalize_tool_checkpoint(&calls));

    free(calls.raw_dsml);
    calls.raw_dsml = NULL;
    TEST_ASSERT(s.should_canonicalize_tool_checkpoint(&calls));

    tool_calls_free(&calls);
}



static void test_responses_live_tail_renders_tool_outputs_only(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_RESPONSES;
    r.think_mode = PULSAR_THINK_HIGH;

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_live");
    tc.name = xstrdup("exec_command");
    tc.arguments = xstrdup("{\"cmd\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.tool_call_id = xstrdup("call_live");
    tool.content = xstrdup("/tmp");
    chat_msgs_push(&msgs, tool);

    responses_prepare_live_continuation(&r, &msgs);
    TEST_ASSERT(r.responses_live_call_ids.len == 1);
    TEST_ASSERT(!strcmp(r.responses_live_call_ids.v[0], "call_live"));
    TEST_ASSERT(r.responses_live_suffix_text != NULL);
    TEST_ASSERT(!strncmp(r.responses_live_suffix_text,
                         "<｜end▁of▁sentence｜><｜User｜><tool_result>",
                         strlen("<｜end▁of▁sentence｜><｜User｜><tool_result>")));
    TEST_ASSERT(strstr(r.responses_live_suffix_text, "/tmp</tool_result>") != NULL);
    TEST_ASSERT(strstr(r.responses_live_suffix_text, "<｜Assistant｜><think>") != NULL);
    TEST_ASSERT(strstr(r.responses_live_suffix_text, "exec_command") == NULL);

    chat_msgs_free(&msgs);
    request_free(&r);
}



static void test_responses_tool_output_id_validation(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.tool_call_id = xstrdup("call_missing");
    tool.content = xstrdup("out");
    chat_msgs_push(&msgs, tool);

    char err[160] = {0};
    TEST_ASSERT(!s.responses_validate_tool_outputs(&msgs, PULSAR_THINK_HIGH, NULL, NULL,
                                                 err, sizeof(err)));
    TEST_ASSERT(strstr(err, "Responses continuation state is not available") != NULL);

    pthread_mutex_lock(&s.tool_mu);
    s.n_slots = 1;
    s.slots[0].responses_live.valid = true;
    s.slots[0].responses_live.live_tokens = 10;
    id_list_push_unique(&s.slots[0].responses_live.call_ids, "call_missing");
    pthread_mutex_unlock(&s.tool_mu);
    err[0] = '\0';
    bool needs_live_tool_state = false;
    TEST_ASSERT(s.responses_validate_tool_outputs(&msgs, PULSAR_THINK_HIGH,
                                                &needs_live_tool_state, NULL,
                                                err, sizeof(err)));
    TEST_ASSERT(needs_live_tool_state);

    chat_msgs_free(&msgs);
    live_tool_state_free(&s.slots[0].responses_live);
    pthread_mutex_destroy(&s.tool_mu);
}



static void test_responses_stateless_tool_replay_requires_reasoning(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_replay");
    tc.name = xstrdup("exec_command");
    tc.arguments = xstrdup("{\"cmd\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.tool_call_id = xstrdup("call_replay");
    tool.content = xstrdup("/tmp");
    chat_msgs_push(&msgs, tool);

    char err[160] = {0};
    bool needs_live_reasoning = false;
    bool needs_live_tool_state = false;
    TEST_ASSERT(s.responses_validate_tool_outputs(&msgs, PULSAR_THINK_HIGH,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(needs_live_reasoning);

    pthread_mutex_lock(&s.tool_mu);
    s.n_slots = 1;
    s.slots[0].responses_live.valid = true;
    s.slots[0].responses_live.live_tokens = 123;
    id_list_push_unique(&s.slots[0].responses_live.call_ids, "call_replay");
    pthread_mutex_unlock(&s.tool_mu);
    err[0] = '\0';
    needs_live_reasoning = false;
    needs_live_tool_state = false;
    TEST_ASSERT(s.responses_validate_tool_outputs(&msgs, PULSAR_THINK_HIGH,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(needs_live_reasoning);

    free(msgs.v[0].reasoning);
    msgs.v[0].reasoning = xstrdup("replayed hidden reasoning");
    err[0] = '\0';
    needs_live_reasoning = false;
    needs_live_tool_state = false;
    TEST_ASSERT(s.responses_validate_tool_outputs(&msgs, PULSAR_THINK_HIGH,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(!needs_live_reasoning);

    free(msgs.v[0].reasoning);
    msgs.v[0].reasoning = NULL;
    err[0] = '\0';
    needs_live_reasoning = false;
    needs_live_tool_state = false;
    TEST_ASSERT(s.responses_validate_tool_outputs(&msgs, PULSAR_THINK_NONE,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(!needs_live_reasoning);

    chat_msgs_free(&msgs);
    live_tool_state_free(&s.slots[0].responses_live);
    pthread_mutex_destroy(&s.tool_mu);
}



static void test_responses_visible_suffix_matches_client_replay(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_RESPONSES;
    r.think_mode = PULSAR_THINK_HIGH;
    r.reasoning_summary_emit = true;

    char *suffix = build_responses_visible_assistant_suffix(&r, "5",
                                                            "hidden summary",
                                                            NULL);
    TEST_ASSERT(strstr(suffix, "hidden summary") == NULL);
    TEST_ASSERT(strstr(suffix, "</think>5") != NULL);
    free(suffix);

    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_live");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&calls, tc);

    suffix = build_responses_visible_assistant_suffix(&r, "",
                                                      "tool summary",
                                                      &calls);
    TEST_ASSERT(strstr(suffix, "tool summary</think>") != NULL);
    TEST_ASSERT(strstr(suffix, "<｜DSML｜tool_calls>") != NULL);
    free(suffix);

    tool_calls_free(&calls);
    request_free(&r);
}



static void test_dsml_decode_state_separates_structure_and_payload(void) {
    dsml_decode_tracker tracker;
    dsml_decode_tracker_init(&tracker);

    const char *prefix =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"edit\">\n";
    TEST_ASSERT(dsml_decode_state_for_text(prefix, strlen(prefix)) ==
                DSML_DECODE_STRUCTURAL);
    dsml_decode_tracker_update(&tracker, prefix, strlen(prefix));
    TEST_ASSERT(tracker.decode == DSML_DECODE_STRUCTURAL);

    const char *path_param =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"edit\">\n"
        PULSAR_PARAM_START " name=\"path\" string=\"true\">/tmp/a.py";
    TEST_ASSERT(dsml_decode_state_for_text(path_param, strlen(path_param)) ==
                DSML_DECODE_STRING_BODY);
    dsml_decode_tracker_update(&tracker, path_param, strlen(path_param));
    TEST_ASSERT(tracker.decode == DSML_DECODE_STRING_BODY);

    const char *path_closing =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"edit\">\n"
        PULSAR_PARAM_START " name=\"path\" string=\"true\">/tmp/a.py</";
    TEST_ASSERT(dsml_decode_state_for_text(path_closing, strlen(path_closing)) ==
                DSML_DECODE_STRUCTURAL);
    dsml_decode_tracker_update(&tracker, path_closing, strlen(path_closing));
    TEST_ASSERT(tracker.decode == DSML_DECODE_STRUCTURAL);

    const char *json_struct =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"edit\">\n"
        PULSAR_PARAM_START " name=\"edits\" string=\"false\">[{";
    TEST_ASSERT(dsml_decode_state_for_text(json_struct, strlen(json_struct)) ==
                DSML_DECODE_JSON_STRUCTURAL);
    dsml_decode_tracker_init(&tracker);
    dsml_decode_tracker_update(&tracker, json_struct, strlen(json_struct));
    TEST_ASSERT(tracker.decode == DSML_DECODE_JSON_STRUCTURAL);

    const char *json_string =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"edit\">\n"
        PULSAR_PARAM_START " name=\"edits\" string=\"false\">[{\"newText\":\"for i in";
    TEST_ASSERT(dsml_decode_state_for_text(json_string, strlen(json_string)) ==
                DSML_DECODE_JSON_STRING);
    dsml_decode_tracker_update(&tracker, json_string, strlen(json_string));
    TEST_ASSERT(tracker.decode == DSML_DECODE_JSON_STRING);

    const char *done =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"edit\">\n"
        PULSAR_PARAM_START " name=\"edits\" string=\"false\">[]"
        PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END;
    TEST_ASSERT(dsml_decode_state_for_text(done, strlen(done)) ==
                DSML_DECODE_OUTSIDE);
    dsml_decode_tracker_init(&tracker);
    dsml_decode_tracker_update(&tracker, done, strlen(done));
    TEST_ASSERT(tracker.decode == DSML_DECODE_OUTSIDE);
}



static void test_tool_memory_max_ids_prunes_oldest(void) {
    const char *a_dsml = "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">a</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const char *b_dsml = "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">b</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const char *c_dsml = "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">c</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";

    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);
    s.tool_mem.max_entries = 2;
    s.tool_memory_put("call_a", a_dsml);
    s.tool_memory_put("call_b", b_dsml);
    s.tool_memory_put("call_c", c_dsml);

    chat_msgs msgs = {0};
    chat_msg a = {0};
    a.role = xstrdup("assistant");
    tool_call tc = {.id = xstrdup("call_a"), .name = xstrdup("bash"), .arguments = xstrdup("{}")};
    tool_calls_push(&a.calls, tc);
    chat_msgs_push(&msgs, a);

    tool_replay_stats stats = {0};
    s.tool_memory_attach_to_messages(&msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml == NULL);
    TEST_ASSERT(stats.canonical == 1);
    TEST_ASSERT(stats.missing_ids == 1);

    chat_msgs_free(&msgs);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}



static void test_tool_separator_whitespace_is_not_content(void) {
    const char *generated =
        "<think>need a tool</think>"
        "I will inspect the files.\n\n\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"description\" string=\"true\">list files</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(reasoning && !strcmp(reasoning, "need a tool"));
    TEST_ASSERT(content && !strcmp(content, "I will inspect the files."));
    TEST_ASSERT(calls.len == 1);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}



static void test_dsml_prompt_escapes_tool_supplied_text(void) {
    tool_calls calls = {0};
    tool_call tc = {0};
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"echo 2>&1 && echo </｜DSML｜tool_calls>\",\"count\":1}");
    tool_calls_push(&calls, tc);

    buf b = {0};
    append_dsml_tool_calls_text(&b, &calls);
    TEST_ASSERT(strstr(b.ptr, "echo 2>&1 && echo </｜DSML｜tool_calls>") != NULL);
    TEST_ASSERT(strstr(b.ptr, "2&gt;&amp;1") == NULL);
    TEST_ASSERT(strstr(b.ptr, "&amp;&amp;") == NULL);
    buf_free(&b);
    tool_calls_free(&calls);

    memset(&calls, 0, sizeof(calls));
    memset(&tc, 0, sizeof(tc));
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"echo </｜DSML｜parameter>\",\"count\":1}");
    tool_calls_push(&calls, tc);

    append_dsml_tool_calls_text(&b, &calls);
    TEST_ASSERT(strstr(b.ptr, "echo &lt;/｜DSML｜parameter>") != NULL);
    TEST_ASSERT(strstr(b.ptr, "echo </｜DSML｜parameter>") == NULL);
    buf_free(&b);
    tool_calls_free(&calls);

    chat_msgs msgs = {0};
    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.content = xstrdup("console.log('<<< < > >>>');\n</tool_result>\n<｜DSML｜tool_calls>not a real tool call");
    chat_msgs_push(&msgs, tool);
    char *prompt = render_chat_prompt_text(&msgs, "{}", NULL, PULSAR_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "console.log('<<< < > >>>');") != NULL);
    TEST_ASSERT(strstr(prompt, "console.log('&lt;") == NULL);
    TEST_ASSERT(strstr(prompt, "&lt;/tool_result>\n<｜DSML｜tool_calls>not a real tool call") != NULL);
    TEST_ASSERT(strstr(prompt, "<tool_result>console.log('<<< < > >>>');\n</tool_result>\n") == NULL);
    free(prompt);
    chat_msgs_free(&msgs);
}



static void test_stop_list_parses_all_sequences(void) {
    stop_list stops = {0};
    const char *json = "[\"END\",\"STOP\"]";
    TEST_ASSERT(parse_stop(&json, &stops));
    TEST_ASSERT(stops.len == 2);
    TEST_ASSERT(stops.max_len == 4);

    size_t pos = 0, len = 0;
    TEST_ASSERT(stop_list_find_from(&stops, "hello STOP tail END", 0, &pos, &len));
    TEST_ASSERT(pos == strlen("hello "));
    TEST_ASSERT(len == strlen("STOP"));
    TEST_ASSERT(stop_list_stream_safe_len(&stops, strlen("abcdef")) == 3);
    stop_list_clear(&stops);
    free(stops.v);
}



static void test_stop_list_streaming_holds_and_trims_stop_text(void) {
    stop_list stops = {0};
    const char *json = "[\"</END>\",\"STOP\"]";
    TEST_ASSERT(parse_stop(&json, &stops));

    size_t safe = stop_list_stream_safe_len(&stops, strlen("hello </"));
    TEST_ASSERT(safe == strlen("hel"));

    size_t pos = 0, len = 0;
    TEST_ASSERT(stop_list_find_from(&stops, "answer STOP hidden", 0, &pos, &len));
    TEST_ASSERT(pos == strlen("answer "));
    TEST_ASSERT(len == strlen("STOP"));

    stop_list_clear(&stops);
    free(stops.v);
}



static char *test_nested_json_array(int depth) {
    buf b = {0};
    for (int i = 0; i < depth; i++) buf_putc(&b, '[');
    buf_putc(&b, '0');
    for (int i = 0; i < depth; i++) buf_putc(&b, ']');
    return buf_take(&b);
}



static void test_json_skip_has_nesting_limit(void) {
    char *ok = test_nested_json_array(JSON_MAX_NESTING);
    const char *p = ok;
    TEST_ASSERT(json_skip_value(&p));
    TEST_ASSERT(*p == '\0');
    free(ok);

    char *bad = test_nested_json_array(JSON_MAX_NESTING + 1);
    p = bad;
    TEST_ASSERT(!json_skip_value(&p));
    free(bad);
}



/* Pin the shared sampling-knob parser every protocol surface routes through.
 * The bug class this guards: a surface silently dropping a knob (the audit
 * found /responses dropping seed before the parsers were consolidated onto
 * this one helper).  Covers the seed edge cases the comment in the parser
 * calls out: NaN and non-positive -> 0, >= 2^64 -> UINT64_MAX (not UB). */
static void test_parse_sampling_key_contract(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    const char *p;

    p = "42,";
    TEST_ASSERT(parse_sampling_key("seed", &p, &r) == 1);
    TEST_ASSERT(r.seed == 42u);
    p = "0,";
    TEST_ASSERT(parse_sampling_key("seed", &p, &r) == 1);
    TEST_ASSERT(r.seed == 0u);
    p = "-3,";
    TEST_ASSERT(parse_sampling_key("seed", &p, &r) == 1);
    TEST_ASSERT(r.seed == 0u);
    p = "18446744073709551616,"; /* 2^64: too big for the cast, clamps */
    TEST_ASSERT(parse_sampling_key("seed", &p, &r) == 1);
    TEST_ASSERT(r.seed == UINT64_MAX);
    p = "\"not-a-number\",";
    TEST_ASSERT(parse_sampling_key("seed", &p, &r) == -1);

    p = "0.7,";
    TEST_ASSERT(parse_sampling_key("temperature", &p, &r) == 1);
    TEST_ASSERT(r.temperature > 0.69f && r.temperature < 0.71f);
    p = "1.5,"; /* out-of-range min_p disables the filter, never greedy-collapses */
    TEST_ASSERT(parse_sampling_key("min_p", &p, &r) == 1);
    TEST_ASSERT(r.min_p == 0.0f && r.has_min_p);
    p = "12,";
    TEST_ASSERT(parse_sampling_key("top_k", &p, &r) == 1);
    TEST_ASSERT(r.top_k == 12 && r.has_top_k);

    p = "1,"; /* unknown keys are the caller's problem: 0, untouched pointer */
    TEST_ASSERT(parse_sampling_key("logprobs", &p, &r) == 0);

    request_free(&r);
}

/* The string-valued JSON helpers must null *out on FAILURE, so the parsers'
 * duplicate-key idiom `free(x); if (!helper(&p, &x)) goto fail;` cannot
 * double-free x (the fail label frees it again).  A malformed second value on
 * a repeated request key is attacker-controlled, so a regression here is a
 * remote single-process-server abort.  json_string_n has always done this;
 * json_content / json_raw_value / parse_prompt / parse_responses_content_array
 * / parse_anthropic_system are the ones that had drifted.  We drive the exact
 * idiom: seed the out-pointer with a heap value (as a first key would), free
 * it, reparse a malformed value, and assert the helper nulled the pointer so
 * the trailing free is a free(NULL) no-op. */
static void test_json_value_helpers_null_out_on_failure(void) {
    struct { const char *name; bool (*fn)(const char **, char **); const char *bad; } cases[] = {
        {"json_content",       json_content,       "[}"},
        {"json_raw_value",     json_raw_value,     "tru"},
        {"parse_prompt",       parse_prompt,       "[42, "},
        {"parse_responses_content_array", parse_responses_content_array, "[{"},
        {"parse_anthropic_system",        parse_anthropic_system,        "[{\"type\":}"},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        char *out = xstrdup("value from the first occurrence of the key");
        free(out);                       /* what the reparse idiom does */
        const char *p = cases[i].bad;
        bool ok = cases[i].fn(&p, &out);
        TEST_ASSERT(!ok);                /* the malformed value must fail */
        TEST_ASSERT(out == NULL);        /* ... and must have nulled *out */
        free(out);                       /* the fail-label free: no double-free */
    }
}



static void append_tool_heavy_schema(buf *b, int idx) {
    if (idx) buf_putc(b, ',');
    buf_puts(b, "{\"type\":\"function\",\"function\":{\"name\":");
    char name[64];
    snprintf(name, sizeof(name), "opencode_tool_%02d", idx);
    json_escape(b, name);
    buf_puts(b, ",\"description\":");
    json_escape(b, "Tool schema with many properties and escaped text.");
    buf_puts(b, ",\"parameters\":{\"type\":\"object\",\"properties\":{");
    for (int j = 0; j < 12; j++) {
        if (j) buf_putc(b, ',');
        char prop[64];
        snprintf(prop, sizeof(prop), "arg_%02d_%02d", idx, j);
        json_escape(b, prop);
        buf_puts(b, ":{\"type\":\"string\",\"description\":");
        json_escape(b, "argument description with \\\\ escapes, quotes, and unicode \\ud83d\\ude80");
        buf_putc(b, '}');
    }
    buf_puts(b, "},\"required\":[");
    for (int j = 0; j < 4; j++) {
        if (j) buf_putc(b, ',');
        char prop[64];
        snprintf(prop, sizeof(prop), "arg_%02d_%02d", idx, j);
        json_escape(b, prop);
    }
    buf_puts(b, "]}}}");
}



static void append_tool_heavy_messages(buf *b) {
    buf_putc(b, '[');
    buf_puts(b, "{\"role\":\"system\",\"content\":");
    json_escape(b, "You are running OpenCode with many local tools.");
    buf_puts(b, "},{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":");
    json_escape(b, "Please inspect the repository, edit files, run tests, and report briefly.");
    buf_puts(b, "}]}");

    for (int turn = 0; turn < 24; turn++) {
        buf_puts(b, ",{\"role\":\"assistant\",\"reasoning_content\":");
        json_escape(b, "I need to inspect files, use tools, and keep track of changes.");
        buf_puts(b, ",\"content\":");
        json_escape(b, "I will use the available tools.");
        buf_puts(b, ",\"tool_calls\":[");
        for (int call = 0; call < 3; call++) {
            if (call) buf_putc(b, ',');
            char id[64], name[64];
            snprintf(id, sizeof(id), "call_%02d_%02d", turn, call);
            snprintf(name, sizeof(name), "opencode_tool_%02d", call);

            buf args = {0};
            buf_puts(&args, "{\"path\":\"/tmp/opencode/project/file.c\",");
            buf_printf(&args, "\"range\":\"%d:%d\",", 10 + turn, 14 + turn);
            buf_puts(&args, "\"old\":\"line one\\\\nline two with quotes \\\" and backslash \\\\\\\\ plus rocket ");
            buf_puts(&args, "\\ud83d\\ude80\",");
            buf_puts(&args, "\"new\":\"replacement text\\\\nwith several lines\\\\nand symbols <>&\"}");

            buf_puts(b, "{\"id\":");
            json_escape(b, id);
            buf_puts(b, ",\"type\":\"function\",\"function\":{\"name\":");
            json_escape(b, name);
            buf_puts(b, ",\"arguments\":");
            json_escape(b, args.ptr ? args.ptr : "");
            buf_puts(b, "}}");
            buf_free(&args);
        }
        buf_puts(b, "]}");

        for (int call = 0; call < 3; call++) {
            char id[64];
            snprintf(id, sizeof(id), "call_%02d_%02d", turn, call);
            buf_puts(b, ",{\"role\":\"tool\",\"tool_call_id\":");
            json_escape(b, id);
            buf_puts(b, ",\"content\":[{\"type\":\"text\",\"text\":");
            json_escape(b, "tool output first line\nsecond line with escaped JSON-looking text {\"ok\":true}");
            buf_puts(b, "}]}");
        }
    }
    buf_putc(b, ']');
}



static void test_json_parser_handles_tool_heavy_requests(void) {
    buf tools = {0};
    buf_putc(&tools, '[');
    for (int i = 0; i < 32; i++) append_tool_heavy_schema(&tools, i);
    buf_putc(&tools, ']');

    buf messages = {0};
    append_tool_heavy_messages(&messages);

    for (int i = 0; i < 32; i++) {
        const char *tp = tools.ptr;
        char *schemas = NULL;
        tool_schema_orders orders = {0};
        TEST_ASSERT(parse_tools_value(&tp, &schemas, &orders, false, NULL));
        json_ws(&tp);
        TEST_ASSERT(*tp == '\0');
        /* The heavy schema goes through the canonical re-render (Python-style
         * spacing, request.cpp json_prompt_value) — assert the spaced form. */
        TEST_ASSERT(schemas && strstr(schemas, "\"name\": \"opencode_tool_00\""));
        TEST_ASSERT(tool_schema_orders_find(&orders, "opencode_tool_00") != NULL);
        free(schemas);
        tool_schema_orders_free(&orders);

        const char *mp = messages.ptr;
        chat_msgs msgs = {0};
        TEST_ASSERT(parse_messages(&mp, &msgs));
        json_ws(&mp);
        TEST_ASSERT(*mp == '\0');
        TEST_ASSERT(msgs.len == 98);
        TEST_ASSERT(msgs.v[2].calls.len == 3);
        TEST_ASSERT(msgs.v[2].calls.v[0].arguments != NULL);
        TEST_ASSERT(strstr(msgs.v[2].calls.v[0].arguments, "replacement text") != NULL);
        chat_msgs_free(&msgs);
    }

    buf_free(&messages);
    buf_free(&tools);
}



static void test_json_string_handles_surrogates(void) {
    const char *p = "\"paired \\ud83d\\ude80 lone \\ud83d text badlow \\ud83d\\u0041\"";
    char *s = NULL;
    TEST_ASSERT(json_string(&p, &s));
    TEST_ASSERT(s != NULL);
    TEST_ASSERT(strstr(s, "paired \xf0\x9f\x9a\x80") != NULL);
    TEST_ASSERT(strstr(s, "lone \xef\xbf\xbd text") != NULL);
    TEST_ASSERT(strstr(s, "badlow \xef\xbf\xbd" "A") != NULL);
    TEST_ASSERT(*p == '\0');
    free(s);
}



static void test_model_metadata_clamps_completion_to_context(void) {
    buf b = {0};
    append_model_json_values(&b, "deepseek-v4-flash", "DeepSeek V4 Flash",
                             32768, 393216);
    TEST_ASSERT(strstr(b.ptr, "\"id\":\"deepseek-v4-flash\"") != NULL);
    TEST_ASSERT(strstr(b.ptr, "\"name\":\"DeepSeek V4 Flash\"") != NULL);
    TEST_ASSERT(strstr(b.ptr, "\"context_length\":32768") != NULL);
    TEST_ASSERT(strstr(b.ptr, "\"max_completion_tokens\":32768") != NULL);
    buf_free(&b);

    append_model_json_values(&b, "deepseek-v4-pro", "DeepSeek V4 Pro",
                             100000, 4096);
    TEST_ASSERT(strstr(b.ptr, "\"id\":\"deepseek-v4-pro\"") != NULL);
    TEST_ASSERT(strstr(b.ptr, "\"name\":\"DeepSeek V4 Pro\"") != NULL);
    TEST_ASSERT(strstr(b.ptr, "\"context_length\":100000") != NULL);
    TEST_ASSERT(strstr(b.ptr, "\"max_completion_tokens\":4096") != NULL);
    buf_free(&b);
}



static void test_client_socket_nonblocking_flag(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;
    set_client_socket_nonblocking(sv[0]);
    int flags = fcntl(sv[0], F_GETFL, 0);
    TEST_ASSERT(flags >= 0);
    TEST_ASSERT((flags & O_NONBLOCK) != 0);
    close(sv[0]);
    close(sv[1]);
}



static void test_thinking_state_tracks_prompt_and_generated_tags(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = PULSAR_THINK_HIGH;
    r.prompt_text = xstrdup("<｜Assistant｜><think>");
    thinking_state st = thinking_state_from_prompt(&r);
    TEST_ASSERT(st.inside == true);
    st.feed("reasoning body", strlen("reasoning body"));
    TEST_ASSERT(st.inside == true);
    st.feed("</thi", strlen("</thi"));
    TEST_ASSERT(st.inside == true);
    st.feed("nk>answer", strlen("nk>answer"));
    TEST_ASSERT(st.inside == false);
    st.feed("<thi", strlen("<thi"));
    TEST_ASSERT(st.inside == false);
    st.feed("nk>more", strlen("nk>more"));
    TEST_ASSERT(st.inside == true);
    request_free(&r);

    request_init(&r, REQ_CHAT, 128);
    r.think_mode = PULSAR_THINK_NONE;
    r.prompt_text = xstrdup("<｜Assistant｜></think>");
    st = thinking_state_from_prompt(&r);
    TEST_ASSERT(st.inside == false);
    request_free(&r);
}



static void test_thinking_checkpoint_remember_gate(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = PULSAR_THINK_HIGH;
    thinking_state st = {.inside = true};

    TEST_ASSERT(!should_remember_thinking_checkpoint(&r, &st, "length"));
    TEST_ASSERT(!should_remember_thinking_checkpoint(&r, &st, "stop"));

    st.inside = false;
    TEST_ASSERT(!should_remember_thinking_checkpoint(&r, &st, "length"));
    TEST_ASSERT(should_remember_thinking_checkpoint(&r, &st, "stop"));

    r.prompt_preserves_reasoning = true;
    TEST_ASSERT(!should_remember_thinking_checkpoint(&r, &st, "stop"));
    r.prompt_preserves_reasoning = false;
    r.has_tools = true;
    /* has_tools is NOT a disqualifier since the openwebui/opencode replay
     * fixes: a client can advertise tools and still strip reasoning on
     * replay, so prompt_preserves_reasoning is the sole gate (see
     * should_remember_thinking_checkpoint). The old expectation here was
     * stale from before that change. */
    TEST_ASSERT(should_remember_thinking_checkpoint(&r, &st, "stop"));
    r.has_tools = false;
    r.think_mode = PULSAR_THINK_NONE;
    TEST_ASSERT(!should_remember_thinking_checkpoint(&r, &st, "stop"));

    request_free(&r);
}



static void test_tool_marker_state_ignores_orphan_end(void) {
    bool saw_start = false;
    bool saw_end = false;
    bool orphan_end = false;

    observe_tool_markers("reasoning\n" PULSAR_PARAM_END "\n" PULSAR_INVOKE_END "\n" PULSAR_TOOL_CALLS_END,
                         &saw_start, &saw_end, &orphan_end);
    TEST_ASSERT(!saw_start);
    TEST_ASSERT(!saw_end);
    TEST_ASSERT(orphan_end);

    orphan_end = false;
    observe_tool_markers(PULSAR_TOOL_CALLS_START "\n" PULSAR_INVOKE_START " name=\"bash\">",
                         &saw_start, &saw_end, &orphan_end);
    TEST_ASSERT(saw_start);
    TEST_ASSERT(!saw_end);
    TEST_ASSERT(!orphan_end);

    observe_tool_markers(PULSAR_INVOKE_END "\n" PULSAR_TOOL_CALLS_END,
                         &saw_start, &saw_end, &orphan_end);
    TEST_ASSERT(saw_start);
    TEST_ASSERT(saw_end);
}



static void test_canonical_rewrite_rebuilds_when_live_tail_changes(void) {
    /* Regression for the first canonical-KV rewrite attempt: replacing a small
     * live suffix looks tempting because the raw SWA ring may still contain the
     * needed rows, but compressed KV counters and compressor/indexer frontiers
     * are already past the shared prefix.  Until those graph frontiers can be
     * restored exactly, every rewrite behind the live end must rebuild or load a
     * disk checkpoint. */
    TEST_ASSERT(pulsar_session_rewrite_requires_rebuild(19296, 19290, 19081));
    TEST_ASSERT(pulsar_session_rewrite_requires_rebuild(1024, 1030, 1000));
    TEST_ASSERT(pulsar_session_rewrite_requires_rebuild(1024, 900, 900));

    TEST_ASSERT(!pulsar_session_rewrite_requires_rebuild(1024, 1024, 1024));
    TEST_ASSERT(!pulsar_session_rewrite_requires_rebuild(1024, 1100, 1024));
}



static void test_kv_cache_store_len_uses_configured_boundary(void) {
    kv_disk_cache kc = {0};
    kc.opt = kv_cache_default_options();
    TEST_ASSERT(kv_cache_store_len(&kc, 11011) == 10240);
    TEST_ASSERT(kv_cache_store_len(&kc, 1695) == 1695);

    kc.opt.boundary_trim_tokens = 0;
    kc.opt.boundary_align_tokens = 1000;
    TEST_ASSERT(kv_cache_store_len(&kc, 3500) == 3000);

    kc.opt.boundary_align_tokens = 0;
    TEST_ASSERT(kv_cache_store_len(&kc, 3500) == 3500);
}



static void test_kv_cache_chat_anchor_uses_last_user_before_assistant(void) {
    const int user = 9001;
    const int assistant = 9002;
    kv_disk_cache kc = {0};
    kc.opt = kv_cache_default_options();
    kc.opt.min_tokens = 4;

    pulsar_tokens codex = {0};
    pulsar_tokens_push(&codex, 1);     /* BOS / system */
    pulsar_tokens_push(&codex, 2);
    pulsar_tokens_push(&codex, user);  /* environment_context item */
    pulsar_tokens_push(&codex, 3);
    pulsar_tokens_push(&codex, 4);
    pulsar_tokens_push(&codex, user);  /* actual task starts here */
    pulsar_tokens_push(&codex, 5);
    pulsar_tokens_push(&codex, assistant);
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &codex, user, assistant) == 5);

    pulsar_tokens claude = {0};
    pulsar_tokens_push(&claude, 1);
    pulsar_tokens_push(&claude, 2);
    pulsar_tokens_push(&claude, 3);
    pulsar_tokens_push(&claude, 4);
    pulsar_tokens_push(&claude, user); /* system reminder and task share a turn */
    pulsar_tokens_push(&claude, 5);
    pulsar_tokens_push(&claude, assistant);
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &claude, user, assistant) == 4);

    pulsar_tokens_free(&codex);
    pulsar_tokens_free(&claude);
}



/* The anchor sits above harness-injected preamble jitter, so the cut must back
 * off below it. Numbers are the live Claude Code session measured 2026-08-11:
 * anchor 21,950, replay agreement wandering over 20,393..21,886. */
static void test_kv_cache_sys_prefix_cut_clears_preamble_jitter(void) {
    kv_disk_cache kc = {0};
    kc.opt = kv_cache_default_options();

    const int anchor = 21950;
    const int cut = kv_cache_sys_prefix_cut(&kc, anchor);
    TEST_ASSERT(cut == 18432);
    TEST_ASSERT(cut % kc.opt.boundary_align_tokens == 0);
    /* Must sit below every observed replay-agreement point, or the checkpoint
     * is stored and evicted forever without ever being a valid byte-prefix. */
    TEST_ASSERT(cut < 20393);

    /* Degenerate inputs yield "no checkpoint", never a negative length. */
    TEST_ASSERT(kv_cache_sys_prefix_cut(&kc, 0) == 0);
    TEST_ASSERT(kv_cache_sys_prefix_cut(&kc, kc.opt.min_tokens - 1) == 0);
    TEST_ASSERT(kv_cache_sys_prefix_cut(&kc, kc.opt.sys_prefix_margin_tokens) == 0);
}

static void test_kv_cache_chat_anchor_ignores_multiturn_tail(void) {
    const int user = 9001;
    const int assistant = 9002;
    kv_disk_cache kc = {0};
    kc.opt = kv_cache_default_options();
    kc.opt.min_tokens = 2;

    pulsar_tokens prompt = {0};
    pulsar_tokens_push(&prompt, 1);
    pulsar_tokens_push(&prompt, 2);
    pulsar_tokens_push(&prompt, user);      /* first task */
    pulsar_tokens_push(&prompt, 3);
    pulsar_tokens_push(&prompt, assistant); /* stop scanning here */
    pulsar_tokens_push(&prompt, 4);
    pulsar_tokens_push(&prompt, user);      /* later turn: not a cold anchor */
    pulsar_tokens_push(&prompt, 5);
    pulsar_tokens_push(&prompt, assistant);
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &prompt, user, assistant) == 2);

    kc.opt.min_tokens = 3;
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &prompt, user, assistant) == -1);
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &prompt, -1, assistant) == -1);
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &prompt, user, -1) == -1);

    pulsar_tokens_free(&prompt);
}



static void test_kv_cache_continued_uses_aligned_frontiers(void) {
    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.opt = kv_cache_default_options();

    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10239) == 0);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10240) == 10240);

    kc.continued_last_store_tokens = 4096;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10240) == 10240);

    kc.continued_last_store_tokens = 24576;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 30720) == 30720);

    kc.continued_last_store_tokens = 10240;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 18432) == 0);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 20480) == 20480);

    kc.opt.boundary_align_tokens = 0;
    kc.continued_last_store_tokens = 20480;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 29999) == 0);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 30000) == 30000);

    /* L122: decode advances in multi-token spec rounds and lands PAST the
     * boundary, almost never on it.  A crossed boundary fires once, as an
     * aligned prefix target, and does not refire until the next crossing. */
    kc.opt = kv_cache_default_options();
    kc.continued_last_store_tokens = 10240;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 20483) == 20480);
    kc.continued_last_store_tokens = 20480;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 20487) == 0);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 30723) == 30720);
}



static void test_kv_cache_cold_store_suppresses_duplicate_continued_boundary(void) {
    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.opt = kv_cache_default_options();

    int old = kv_cache_suppress_continued_store(&kc, 10240);
    TEST_ASSERT(old == 0);
    TEST_ASSERT(kc.continued_last_store_tokens == 10240);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10240) == 0);

    kv_cache_restore_suppressed_continued(&kc, old, 10240);
    TEST_ASSERT(kc.continued_last_store_tokens == 0);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10240) == 10240);
}



static void test_kv_cache_file_size_must_fit_budget(void) {
    kv_disk_cache kc = {0};
    kc.budget_bytes = 1100;

    TEST_ASSERT(kv_cache_file_size_fits(&kc, 100, 930, 0, NULL, NULL));
    TEST_ASSERT(!kv_cache_file_size_fits(&kc, 100, 938, 0, NULL, NULL));
    TEST_ASSERT(!kv_cache_file_size_fits(&kc, 100, 900, 40, NULL, NULL));
    TEST_ASSERT(!kv_cache_file_size_fits(&kc, UINT64_MAX, 1, 0, NULL, NULL));

    kc.budget_bytes = 0;
    TEST_ASSERT(kv_cache_file_size_fits(&kc, 100, 900, 40, NULL, NULL));
    TEST_ASSERT(!kv_cache_file_size_fits(&kc, UINT64_MAX, 1, 0, NULL, NULL));
}



static void test_sha1_bytes_hex_matches_known_vector(void) {
    char sha[41];
    sha1_bytes_hex("abc", 3, sha);
    TEST_ASSERT(!strcmp(sha, "a9993e364706816aba3e25717850c26c9cd0d89d"));
}



static void test_kv_stub_file(const char *dir, const char *sha,
                              uint8_t reason, uint32_t tokens, uint32_t hits,
                              uint64_t last_used, uint64_t payload_bytes) {
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    if (!fp) {
        free(path);
        return;
    }

    uint8_t h[KV_CACHE_FIXED_HEADER];
    kv_fill_header(h, 2, reason, 0, tokens, hits, 32768, 100, last_used, payload_bytes);
    uint8_t text_len[4] = {0};
    TEST_ASSERT(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
    TEST_ASSERT(fwrite(text_len, 1, sizeof(text_len), fp) == sizeof(text_len));
    for (uint64_t i = 0; i < payload_bytes; i++) {
        TEST_ASSERT(fputc(0, fp) != EOF);
    }
    TEST_ASSERT(fclose(fp) == 0);
    free(path);
}



static void test_kv_text_stub_file_model(const char *dir, const char *text,
                                         uint8_t model_id, uint8_t reason,
                                         uint32_t tokens,
                                         uint64_t payload_bytes) {
    char sha[41];
    sha1_bytes_hex(text, strlen(text), sha);
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    if (!fp) {
        free(path);
        return;
    }

    uint8_t h[KV_CACHE_FIXED_HEADER];
    pulsar_kvstore_fill_header(h, model_id, 2, reason, 0, tokens, 0,
                            32768, 100, 100, payload_bytes);
    uint8_t text_len[4];
    le_put32(text_len, (uint32_t)strlen(text));
    TEST_ASSERT(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
    TEST_ASSERT(fwrite(text_len, 1, sizeof(text_len), fp) == sizeof(text_len));
    TEST_ASSERT(fwrite(text, 1, strlen(text), fp) == strlen(text));
    for (uint64_t i = 0; i < payload_bytes; i++) {
        TEST_ASSERT(fputc(0, fp) != EOF);
    }
    TEST_ASSERT(fclose(fp) == 0);
    free(path);
}



static void test_kv_text_stub_file(const char *dir, const char *text,
                                   uint8_t reason,
                                   uint32_t tokens, uint64_t payload_bytes) {
    test_kv_text_stub_file_model(dir, text, 0, reason, tokens, payload_bytes);
}



static void test_kv_cache_lookup_uses_longest_text_prefix(void) {
    char tmpl[] = "/tmp/ds4-kv-text-prefix-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *short_text = "transcript prefix";
    const char *long_text = "transcript prefix with sampled token bytes";
    test_kv_text_stub_file(dir, short_text, KV_REASON_COLD, 512, 0);
    test_kv_text_stub_file(dir, long_text, KV_REASON_COLD, 768, 0);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();

    int idx = kv_cache_find_text_prefix(&kc,
        "transcript prefix with sampled token bytes and suffix",
        2, 32768);
    TEST_ASSERT(idx >= 0);
    TEST_ASSERT(idx >= 0 && kc.entry[idx].tokens == 768);
    TEST_ASSERT(idx >= 0 && kc.entry[idx].text_bytes == strlen(long_text));
    TEST_ASSERT(kv_cache_find_text_prefix(&kc, "transcript prefiX", 2, 32768) < 0);

    kv_cache_close(&kc);
    char short_sha[41], long_sha[41];
    sha1_bytes_hex(short_text, strlen(short_text), short_sha);
    sha1_bytes_hex(long_text, strlen(long_text), long_sha);
    char short_name[44], long_name[44];
    snprintf(short_name, sizeof(short_name), "%.40s.kv", short_sha);
    snprintf(long_name, sizeof(long_name), "%.40s.kv", long_sha);
    char *short_path = path_join(dir, short_name);
    char *long_path = path_join(dir, long_name);
    unlink(short_path);
    unlink(long_path);
    free(short_path);
    free(long_path);
    rmdir(dir);
}



static void test_kv_cache_lookup_rejects_wrong_model(void) {
    char tmpl[] = "/tmp/ds4-kv-model-id-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *text = "shared rendered prefix";
    test_kv_text_stub_file_model(dir, text, 1, KV_REASON_COLD, 512, 0);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();

    TEST_ASSERT(pulsar_kvstore_find_text_prefix(&kc, "shared rendered prefix and tail",
                                             0, 2, 32768) < 0);
    int idx = pulsar_kvstore_find_text_prefix(&kc, "shared rendered prefix and tail",
                                           1, 2, 32768);
    TEST_ASSERT(idx >= 0);
    TEST_ASSERT(idx >= 0 && kc.entry[idx].model_id == 1);

    kv_cache_close(&kc);
    char sha[41];
    sha1_bytes_hex(text, strlen(text), sha);
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    unlink(path);
    free(path);
    rmdir(dir);
}



static void test_kv_cache_lookup_rejects_stale_payload_abi(void) {
    char tmpl[] = "/tmp/ds4-kv-stale-abi-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *text = "stale rendered prefix";
    char sha[41];
    sha1_bytes_hex(text, strlen(text), sha);
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);

    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    if (fp) {
        uint8_t h[KV_CACHE_FIXED_HEADER];
        kv_fill_header(h, 2, KV_REASON_COLD, 0, 512, 0, 32768, 100, 100, 0);
        h[20] = 0; /* pre-ABI-guard files used this byte as reserved zero. */
        uint8_t text_len[4];
        le_put32(text_len, (uint32_t)strlen(text));
        TEST_ASSERT(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
        TEST_ASSERT(fwrite(text_len, 1, sizeof(text_len), fp) == sizeof(text_len));
        TEST_ASSERT(fwrite(text, 1, strlen(text), fp) == strlen(text));
        TEST_ASSERT(fclose(fp) == 0);
    }

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();

    TEST_ASSERT(pulsar_kvstore_find_text_prefix(&kc, "stale rendered prefix and tail",
                                             0, 2, 32768) < 0);

    kv_cache_close(&kc);
    unlink(path);
    free(path);
    rmdir(dir);
}



static void test_kv_tool_map_filters_by_dsml_text(void) {
    const char *dsml_keep =
        "\n\n<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    const char *dsml_drop =
        "\n\n<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">zzzz</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";

    server src = {0}, dst = {0};
    pthread_mutex_init(&src.tool_mu, NULL);
    pthread_mutex_init(&dst.tool_mu, NULL);
    src.tool_memory_put("call_keep", dsml_keep);
    src.tool_memory_put("call_drop", dsml_drop);

    FILE *fp = tmpfile();
    TEST_ASSERT(fp != NULL);
    uint64_t estimated_bytes = 0;
    TEST_ASSERT(src.kv_tool_map_serialized_size(dsml_keep, &estimated_bytes));
    uint64_t bytes = 0;
    TEST_ASSERT(src.kv_tool_map_write(fp, dsml_keep, &bytes));
    TEST_ASSERT(bytes > 0);
    TEST_ASSERT(estimated_bytes == bytes);
    rewind(fp);
    TEST_ASSERT(dst.kv_tool_map_load_from_pos(fp, NULL) == 1);

    chat_msgs msgs = {0};
    chat_msg a = {0};
    a.role = xstrdup("assistant");
    tool_call keep = {.id = xstrdup("call_keep"), .name = xstrdup("bash"), .arguments = xstrdup("{}")};
    tool_calls_push(&a.calls, keep);
    chat_msgs_push(&msgs, a);
    chat_msg b = {0};
    b.role = xstrdup("assistant");
    tool_call drop = {.id = xstrdup("call_drop"), .name = xstrdup("bash"), .arguments = xstrdup("{}")};
    tool_calls_push(&b.calls, drop);
    chat_msgs_push(&msgs, b);
    tool_replay_stats stats = {0};
    dst.tool_memory_attach_to_messages(&msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml != NULL);
    TEST_ASSERT(msgs.v[1].calls.raw_dsml == NULL);
    TEST_ASSERT(stats.disk == 1);
    TEST_ASSERT(stats.canonical == 1);
    TEST_ASSERT(stats.missing_ids == 1);
    TEST_ASSERT(strstr(msgs.v[0].calls.raw_dsml, "pwd") != NULL);
    TEST_ASSERT(strstr(msgs.v[0].calls.raw_dsml, "zzzz") == NULL);

    chat_msgs_free(&msgs);
    if (fp) fclose(fp);
    tool_memory_free(&src.tool_mem);
    tool_memory_free(&dst.tool_mem);
    pthread_mutex_destroy(&src.tool_mu);
    pthread_mutex_destroy(&dst.tool_mu);
}



static void test_kv_tool_map_restores_before_prompt_render(void) {
    char tmpl[] = "/tmp/ds4-kv-tool-map-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *sha = "3333333333333333333333333333333333333333";
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    const char *dsml =
        "\n\n<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">echo exact</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    const char *text = dsml;

    server src = {0};
    pthread_mutex_init(&src.tool_mu, NULL);
    src.tool_memory_put("call_disk", dsml);

    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    if (fp) {
        uint8_t h[KV_CACHE_FIXED_HEADER];
        kv_fill_header(h, 2, KV_REASON_CONTINUED, KV_EXT_TOOL_MAP, 512, 0, 32768, 100, 100, 0);
        uint8_t text_len[4];
        le_put32(text_len, (uint32_t)strlen(text));
        TEST_ASSERT(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
        TEST_ASSERT(fwrite(text_len, 1, sizeof(text_len), fp) == sizeof(text_len));
        TEST_ASSERT(fwrite(text, 1, strlen(text), fp) == strlen(text));
        uint64_t ignored = 0;
        TEST_ASSERT(src.kv_tool_map_write(fp, dsml, &ignored));
        TEST_ASSERT(fclose(fp) == 0);
    }

    server dst = {0};
    pthread_mutex_init(&dst.tool_mu, NULL);
    dst.kv.enabled = true;
    dst.kv.dir = xstrdup(dir);
    dst.kv.opt = kv_cache_default_options();

    chat_msgs msgs = {0};
    chat_msg a = {0};
    a.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_disk");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"echo canonical\"}");
    tool_calls_push(&a.calls, tc);
    chat_msgs_push(&msgs, a);

    dst.kv_cache_restore_tool_memory_for_messages(&msgs);
    tool_replay_stats stats = {0};
    dst.tool_memory_attach_to_messages(&msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml != NULL);
    TEST_ASSERT(stats.disk == 1);
    TEST_ASSERT(stats.canonical == 0);
    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_HIGH);
    TEST_ASSERT(strstr(prompt, "echo exact") != NULL);
    TEST_ASSERT(strstr(prompt, "echo canonical") == NULL);

    free(prompt);
    chat_msgs_free(&msgs);
    kv_cache_close(&dst.kv);
    tool_memory_free(&src.tool_mem);
    tool_memory_free(&dst.tool_mem);
    pthread_mutex_destroy(&src.tool_mu);
    pthread_mutex_destroy(&dst.tool_mu);
    unlink(path);
    free(path);
    rmdir(dir);
}



static void test_kv_cache_eviction_values_fresh_snapshots(void) {
    char tmpl[] = "/tmp/ds4-kv-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *old_sha = "1111111111111111111111111111111111111111";
    const char *new_sha = "2222222222222222222222222222222222222222";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, old_sha, KV_REASON_UNKNOWN, 512, 0, now, 4096);
    test_kv_stub_file(dir, new_sha, KV_REASON_UNKNOWN, 2048, 0, now, 2048);

    char old_name[44], new_name[44];
    snprintf(old_name, sizeof(old_name), "%.40s.kv", old_sha);
    snprintf(new_name, sizeof(new_name), "%.40s.kv", new_sha);
    char *old_path = path_join(dir, old_name);
    char *new_path = path_join(dir, new_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL, 0, NULL);

    TEST_ASSERT(access(old_path, F_OK) != 0);
    TEST_ASSERT(access(new_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(old_path);
    unlink(new_path);
    free(old_path);
    free(new_path);
    rmdir(dir);
}



static void test_kv_cache_eviction_prefers_anchor_reason(void) {
    char tmpl[] = "/tmp/ds4-kv-anchor-reason-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *anchor_sha = "1111111111111111111111111111111111111111";
    const char *continued_sha = "2222222222222222222222222222222222222222";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, anchor_sha, KV_REASON_COLD, 2048, 0, now, 2048);
    test_kv_stub_file(dir, continued_sha, KV_REASON_CONTINUED, 2048, 0, now, 2048);

    char anchor_name[44], continued_name[44];
    snprintf(anchor_name, sizeof(anchor_name), "%.40s.kv", anchor_sha);
    snprintf(continued_name, sizeof(continued_name), "%.40s.kv", continued_sha);
    char *anchor_path = path_join(dir, anchor_name);
    char *continued_path = path_join(dir, continued_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL, 0, NULL);

    TEST_ASSERT(access(anchor_path, F_OK) == 0);
    TEST_ASSERT(access(continued_path, F_OK) != 0);

    kv_cache_close(&kc);
    unlink(anchor_path);
    unlink(continued_path);
    free(anchor_path);
    free(continued_path);
    rmdir(dir);
}



/* A sys-prefix checkpoint (truncated shared preamble) must outlive ordinary
 * cold anchors under budget pressure: it is the one file every NEW
 * conversation with the same system prompt can text-prefix restore from, and
 * it sits at hits=0 until the first restart needs it. */
static void test_kv_cache_eviction_prefers_sys_prefix_over_cold(void) {
    char tmpl[] = "/tmp/ds4-kv-sys-prefix-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *sys_sha  = "3333333333333333333333333333333333333333";
    const char *cold_sha = "4444444444444444444444444444444444444444";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, sys_sha, KV_REASON_SYS_PREFIX, 2048, 0, now, 2048);
    test_kv_stub_file(dir, cold_sha, KV_REASON_COLD, 2048, 0, now, 2048);

    char sys_name[44], cold_name[44];
    snprintf(sys_name, sizeof(sys_name), "%.40s.kv", sys_sha);
    snprintf(cold_name, sizeof(cold_name), "%.40s.kv", cold_sha);
    char *sys_path = path_join(dir, sys_name);
    char *cold_path = path_join(dir, cold_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL, 0, NULL);

    TEST_ASSERT(access(sys_path, F_OK) == 0);
    TEST_ASSERT(access(cold_path, F_OK) != 0);

    kv_cache_close(&kc);
    unlink(sys_path);
    unlink(cold_path);
    free(sys_path);
    free(cold_path);
    rmdir(dir);

    TEST_ASSERT(pulsar_kvstore_reason_code("sys-prefix") ==
                PULSAR_KVSTORE_REASON_SYS_PREFIX);
}



static void test_kv_cache_eviction_makes_room_before_store(void) {
    char tmpl[] = "/tmp/ds4-kv-pre-store-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *old_sha = "1111111111111111111111111111111111111111";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, old_sha, KV_REASON_COLD, 4096, 0, now, 2048);

    char old_name[44];
    snprintf(old_name, sizeof(old_name), "%.40s.kv", old_sha);
    char *old_path = path_join(dir, old_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 4096u) + 16u;
    kv_cache_evict(&kc, NULL, KV_CACHE_FIXED_HEADER + 4u + 4096u, NULL);

    TEST_ASSERT(access(old_path, F_OK) != 0);

    kv_cache_close(&kc);
    unlink(old_path);
    free(old_path);
    rmdir(dir);
}



static void test_kv_cache_eviction_ignores_oversize_incoming(void) {
    char tmpl[] = "/tmp/ds4-kv-oversize-store-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *old_sha = "1111111111111111111111111111111111111111";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, old_sha, KV_REASON_COLD, 4096, 0, now, 1024);

    char old_name[44];
    snprintf(old_name, sizeof(old_name), "%.40s.kv", old_sha);
    char *old_path = path_join(dir, old_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 1024u) + 16u;
    kv_cache_evict(&kc, NULL, kc.budget_bytes + 1, NULL);

    TEST_ASSERT(access(old_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(old_path);
    free(old_path);
    rmdir(dir);
}



static void test_kv_cache_eviction_prefers_superseded_continued_prefix(void) {
    char tmpl[] = "/tmp/ds4-kv-prefix-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *continued_text = "system: hello world";
    const char *cold_text = "different stable prefix";
    const char *incoming_text = "system: hello world\nuser: prompt";
    test_kv_text_stub_file(dir, continued_text, KV_REASON_CONTINUED, 4096, 2048);
    test_kv_text_stub_file(dir, cold_text, KV_REASON_COLD, 1024, 2048);

    char continued_sha[41], cold_sha[41];
    sha1_bytes_hex(continued_text, strlen(continued_text), continued_sha);
    sha1_bytes_hex(cold_text, strlen(cold_text), cold_sha);
    char continued_name[44], cold_name[44];
    snprintf(continued_name, sizeof(continued_name), "%.40s.kv", continued_sha);
    snprintf(cold_name, sizeof(cold_name), "%.40s.kv", cold_sha);
    char *continued_path = path_join(dir, continued_name);
    char *cold_path = path_join(dir, cold_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    uint64_t incoming_bytes =
        KV_CACHE_FIXED_HEADER + 4u + strlen(incoming_text) + 2048u;
    kc.budget_bytes =
        incoming_bytes + KV_CACHE_FIXED_HEADER + 4u + strlen(cold_text) + 2048u;
    pulsar_kvstore_eviction_context incoming = {
        .text = incoming_text,
        .text_len = strlen(incoming_text),
        .model_id = 0,
        .quant_bits = 2,
        .ctx_size = 32768,
        .reject_different_quant = false,
    };
    kv_cache_evict(&kc, NULL, incoming_bytes, &incoming);

    TEST_ASSERT(access(continued_path, F_OK) != 0);
    TEST_ASSERT(access(cold_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(continued_path);
    unlink(cold_path);
    free(continued_path);
    free(cold_path);
    rmdir(dir);
}



static void test_kv_cache_eviction_keeps_smaller_context_prefix(void) {
    char tmpl[] = "/tmp/ds4-kv-prefix-ctx-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *continued_text = "system: hello world";
    const char *cold_text = "different stable prefix";
    const char *incoming_text = "system: hello world\nuser: prompt";
    test_kv_text_stub_file(dir, continued_text, KV_REASON_CONTINUED, 4096, 2048);
    test_kv_text_stub_file(dir, cold_text, KV_REASON_COLD, 1024, 2048);

    char continued_sha[41], cold_sha[41];
    sha1_bytes_hex(continued_text, strlen(continued_text), continued_sha);
    sha1_bytes_hex(cold_text, strlen(cold_text), cold_sha);
    char continued_name[44], cold_name[44];
    snprintf(continued_name, sizeof(continued_name), "%.40s.kv", continued_sha);
    snprintf(cold_name, sizeof(cold_name), "%.40s.kv", cold_sha);
    char *continued_path = path_join(dir, continued_name);
    char *cold_path = path_join(dir, cold_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    uint64_t incoming_bytes =
        KV_CACHE_FIXED_HEADER + 4u + strlen(incoming_text) + 2048u;
    kc.budget_bytes =
        incoming_bytes + KV_CACHE_FIXED_HEADER + 4u + strlen(continued_text) + 2048u;
    pulsar_kvstore_eviction_context incoming = {
        .text = incoming_text,
        .text_len = strlen(incoming_text),
        .model_id = 0,
        .quant_bits = 2,
        .ctx_size = 65536,
        .reject_different_quant = false,
    };
    kv_cache_evict(&kc, NULL, incoming_bytes, &incoming);

    TEST_ASSERT(access(continued_path, F_OK) == 0);
    TEST_ASSERT(access(cold_path, F_OK) != 0);

    kv_cache_close(&kc);
    unlink(continued_path);
    unlink(cold_path);
    free(continued_path);
    free(cold_path);
    rmdir(dir);
}



static void test_kv_cache_eviction_score_decays_stale_hits(void) {
    /* stale: lower tokens-per-byte (e.g. tool-heavy prompt) but boosted by
     * 10 hits well in the past.  fresh: higher tokens-per-byte and zero hits,
     * just stored.  The stale hit bonus decays by inactivity, so fresh wins on
     * its better baseline even though stale once had more successful hits. */
    const uint64_t now = 1000u + 14u * KV_CACHE_HIT_HALF_LIFE_SECONDS;
    kv_entry stale = {.tokens = 1024, .hits = 10, .last_used = 1000, .file_size = 4096};
    kv_entry fresh = {.tokens = 2048, .hits = 0,  .last_used = now, .file_size = 4096};

    double s_on = kv_entry_eviction_score(&stale, NULL, now, NULL);
    double f_on = kv_entry_eviction_score(&fresh, NULL, now, NULL);
    TEST_ASSERT(s_on < f_on);

    /* A fresh entry's score never decays below its (0+1) * tokens/size floor,
     * regardless of how old another entry's hit history is. */
    TEST_ASSERT(f_on == 1.0 * (double)fresh.tokens / (double)fresh.file_size);
}

/* The supersedes-continued demotion is the branch A6 refactored: evict() now
 * decides it ONCE per entry and passes the result in, instead of re-deriving it
 * (and re-SHA1ing a prefix of the incoming text) on every scoring pass. Pin the
 * behaviour so the split cannot silently drop the demotion. */
static void test_kv_cache_eviction_score_demotes_superseded_continued(void) {
    const char *text = "the quick brown fox jumps over the lazy dog";
    const size_t text_len = strlen(text);
    const uint32_t prefix_bytes = 19;          /* "the quick brown fox" */

    char prefix_sha[41];
    sha1_bytes_hex(text, prefix_bytes, prefix_sha);

    kv_entry cont = {.quant_bits = 2, .model_id = 1,
                     .reason = KV_REASON_CONTINUED,
                     .tokens = 1024, .hits = 0, .ctx_size = 4096,
                     .last_used = 500, .text_bytes = prefix_bytes,
                     .file_size = 4096};
    memcpy(cont.sha, prefix_sha, sizeof(cont.sha));

    pulsar_kvstore_eviction_context incoming = {};
    incoming.text = text;
    incoming.text_len = text_len;
    incoming.model_id = 1;
    incoming.quant_bits = 2;
    incoming.ctx_size = 4096;
    incoming.reject_different_quant = true;

    const uint64_t now = 500;
    const double plain = kv_entry_eviction_score(&cont, NULL, now, NULL);
    const double demoted = kv_entry_eviction_score(&cont, NULL, now, &incoming);
    /* The incoming checkpoint contains this one's whole text as a prefix, so
     * this entry is redundant and must score strictly lower. */
    TEST_ASSERT(demoted < plain);

    /* A DIFFERENT model must not be treated as superseding, even byte-for-byte. */
    pulsar_kvstore_eviction_context other = incoming;
    other.model_id = 2;
    TEST_ASSERT(kv_entry_eviction_score(&cont, NULL, now, &other) == plain);

    /* Nor may a narrower context supersede a wider one. */
    pulsar_kvstore_eviction_context narrower = incoming;
    narrower.ctx_size = 8192;
    TEST_ASSERT(kv_entry_eviction_score(&cont, NULL, now, &narrower) == plain);
}



static void test_kv_cache_eviction_decayed_hits_tie_break_by_age(void) {
    char tmpl[] = "/tmp/ds4-kv-stale-hit-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *old_sha = "1111111111111111111111111111111111111111";
    const char *new_sha = "2222222222222222222222222222222222222222";
    uint64_t now = (uint64_t)time(NULL);
    uint64_t stale = now > KV_CACHE_HIT_HALF_LIFE_SECONDS * 14ull
        ? now - KV_CACHE_HIT_HALF_LIFE_SECONDS * 14ull
        : 1;
    test_kv_stub_file(dir, old_sha, KV_REASON_COLD, 2048, 15, stale, 2048);
    test_kv_stub_file(dir, new_sha, KV_REASON_COLD, 2048, 0, now, 2048);

    char old_name[44], new_name[44];
    snprintf(old_name, sizeof(old_name), "%.40s.kv", old_sha);
    snprintf(new_name, sizeof(new_name), "%.40s.kv", new_sha);
    char *old_path = path_join(dir, old_name);
    char *new_path = path_join(dir, new_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL, 0, NULL);

    TEST_ASSERT(access(old_path, F_OK) != 0);
    TEST_ASSERT(access(new_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(old_path);
    unlink(new_path);
    free(old_path);
    free(new_path);
    rmdir(dir);
}



static void test_kv_cache_eviction_keeps_aligned_continued_frontiers(void) {
    char tmpl[] = "/tmp/ds4-kv-live-prefix-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *cold_sha = "1111111111111111111111111111111111111111";
    const char *continued_sha = "2222222222222222222222222222222222222222";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, cold_sha, KV_REASON_COLD, 512, 0, now, 2048);
    test_kv_stub_file(dir, continued_sha, KV_REASON_CONTINUED, 2048, 0, now, 2048);

    char cold_name[44], continued_name[44];
    snprintf(cold_name, sizeof(cold_name), "%.40s.kv", cold_sha);
    snprintf(continued_name, sizeof(continued_name), "%.40s.kv", continued_sha);
    char *cold_path = path_join(dir, cold_name);
    char *continued_path = path_join(dir, continued_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL, 0, NULL);

    TEST_ASSERT(access(cold_path, F_OK) != 0);
    TEST_ASSERT(access(continued_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(cold_path);
    unlink(continued_path);
    free(cold_path);
    free(continued_path);
    rmdir(dir);
}



static void test_thinking_checkpoint_canonical_matches_future_prompt(void) {
    /* Simulate: user sends a single message, thinking mode on, no tools.
     * Model generates reasoning + content.  The next request will drop the
     * reasoning from this turn.  Verify that:
     *   prompt_text[:-len("<think>")] + "</think>" + content + "<|eos|>"
     * equals what render_chat_prompt_text produces for the history. */

    chat_msgs prefix_msgs = {0};
    chat_msg user1 = {0};
    user1.role = xstrdup("user");
    user1.content = xstrdup("What is 2+2?");
    chat_msgs_push(&prefix_msgs, user1);

    /* This is what prompt_text looks like for the first generation */
    char *prompt_text = render_chat_prompt_text(&prefix_msgs, NULL, NULL, PULSAR_THINK_HIGH);
    /* prompt_text should end with <think> */
    size_t pt_len = strlen(prompt_text);
    TEST_ASSERT(pt_len >= 7);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 7, "<think>", 7));

    /* The model generates: reasoning + </think> + content */
    const char *reasoning = "Let me think... 2+2 = 4";
    const char *content = "The answer is 4.";

    /* Build the canonical checkpoint text (what we'd produce after canonicalization) */
    buf canonical = {0};
    buf_append(&canonical, prompt_text, pt_len - 7);  /* strip <think> */
    buf_puts(&canonical, "</think>");
    buf_puts(&canonical, content);
    buf_puts(&canonical, "<" "\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c" ">");

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = PULSAR_THINK_HIGH;
    r.prompt_text = xstrdup(prompt_text);
    char *visible = build_toolless_thinking_visible_text(&r, content);
    TEST_ASSERT(visible != NULL);
    TEST_ASSERT(!strcmp(visible, canonical.ptr));
    free(visible);
    request_free(&r);

    /* Now build what the NEXT request would render: history includes this
     * assistant message, plus a new user message.  Extract just the prefix
     * up to and including the eos of the assistant turn. */
    chat_msgs history_msgs = {0};
    chat_msg h_user1 = {0};
    h_user1.role = xstrdup("user");
    h_user1.content = xstrdup("What is 2+2?");
    chat_msgs_push(&history_msgs, h_user1);
    chat_msg h_asst = {0};
    h_asst.role = xstrdup("assistant");
    h_asst.reasoning = xstrdup(reasoning);
    h_asst.content = xstrdup(content);
    chat_msgs_push(&history_msgs, h_asst);
    chat_msg h_user2 = {0};
    h_user2.role = xstrdup("user");
    h_user2.content = xstrdup("Thanks!");
    chat_msgs_push(&history_msgs, h_user2);

    char *future_prompt = render_chat_prompt_text(&history_msgs, NULL, NULL, PULSAR_THINK_HIGH);

    /* The future prompt should START with our canonical text */
    size_t clen = canonical.len;
    TEST_ASSERT(strlen(future_prompt) > clen);
    TEST_ASSERT(!memcmp(future_prompt, canonical.ptr, clen));

    /* And what comes after is the new user turn + assistant prefix */
    const char *rest = future_prompt + clen;
    TEST_ASSERT(strstr(rest, "Thanks!") != NULL);
    TEST_ASSERT(strstr(rest, "<think>") != NULL);  /* new turn starts thinking */

    /* Verify reasoning is NOT in the future prompt for this turn */
    const char *asst_turn = strstr(future_prompt, "<" "\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c" ">");
    TEST_ASSERT(asst_turn != NULL);
    TEST_ASSERT(strstr(future_prompt, reasoning) == NULL);  /* reasoning dropped */

    free(future_prompt);
    buf_free(&canonical);
    free(prompt_text);
    chat_msgs_free(&prefix_msgs);
    chat_msgs_free(&history_msgs);
}



static void test_thinking_canonical_empty_content(void) {
    /* Edge case: model thinks but produces empty content (e.g. tool-less
     * thinking where answer is entirely in reasoning).  Canonical should
     * still be valid: prompt_text[:-7] + "</think><|eos|>" */
    chat_msgs msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("Think about life");
    chat_msgs_push(&msgs, user);

    char *prompt_text = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_HIGH);
    size_t pt_len = strlen(prompt_text);

    /* Build canonical with empty content */
    buf canonical = {0};
    buf_append(&canonical, prompt_text, pt_len - 7);
    buf_puts(&canonical, "</think>");
    /* empty content */
    buf_puts(&canonical, "<" "\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c" ">");

    /* Future prompt with empty content assistant message */
    chat_msgs history = {0};
    chat_msg h_u = {0};
    h_u.role = xstrdup("user");
    h_u.content = xstrdup("Think about life");
    chat_msgs_push(&history, h_u);
    chat_msg h_a = {0};
    h_a.role = xstrdup("assistant");
    h_a.reasoning = xstrdup("Deep thoughts about existence...");
    h_a.content = xstrdup("");
    chat_msgs_push(&history, h_a);
    chat_msg h_u2 = {0};
    h_u2.role = xstrdup("user");
    h_u2.content = xstrdup("Continue");
    chat_msgs_push(&history, h_u2);

    char *future = render_chat_prompt_text(&history, NULL, NULL, PULSAR_THINK_HIGH);
    TEST_ASSERT(strlen(future) > canonical.len);
    TEST_ASSERT(!memcmp(future, canonical.ptr, canonical.len));
    /* reasoning dropped */
    TEST_ASSERT(strstr(future, "Deep thoughts") == NULL);

    free(future);
    buf_free(&canonical);
    free(prompt_text);
    chat_msgs_free(&msgs);
    chat_msgs_free(&history);
}



static void test_thinking_canonical_multi_turn(void) {
    /* Multi-turn: 3 user messages, 2 assistant responses with reasoning.
     * Both prior assistant turns should have reasoning dropped.
     * The canonical after the SECOND generation should produce text that
     * matches the start of a 3rd-turn future prompt. */
    chat_msgs turn2_prefix = {0};
    chat_msg u1 = {0};
    u1.role = xstrdup("user");
    u1.content = xstrdup("Hello");
    chat_msgs_push(&turn2_prefix, u1);
    chat_msg a1 = {0};
    a1.role = xstrdup("assistant");
    a1.reasoning = xstrdup("first reasoning");
    a1.content = xstrdup("Hi there");
    chat_msgs_push(&turn2_prefix, a1);
    chat_msg u2 = {0};
    u2.role = xstrdup("user");
    u2.content = xstrdup("How are you?");
    chat_msgs_push(&turn2_prefix, u2);

    /* prompt_text for the 2nd generation (includes 1st assistant turn) */
    char *prompt_text = render_chat_prompt_text(&turn2_prefix, NULL, NULL, PULSAR_THINK_HIGH);
    size_t pt_len = strlen(prompt_text);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 7, "<think>", 7));

    /* 1st turn reasoning is already dropped in this prompt_text */
    TEST_ASSERT(strstr(prompt_text, "first reasoning") == NULL);
    TEST_ASSERT(strstr(prompt_text, "Hi there") != NULL);

    /* After 2nd generation: canonical drops 2nd reasoning too */
    const char *content2 = "I'm doing well";
    buf canonical = {0};
    buf_append(&canonical, prompt_text, pt_len - 7);
    buf_puts(&canonical, "</think>");
    buf_puts(&canonical, content2);
    buf_puts(&canonical, "<" "\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c" ">");

    /* Future: 3rd user message arrives */
    chat_msgs future_msgs = {0};
    chat_msg fu1 = {0}; fu1.role = xstrdup("user"); fu1.content = xstrdup("Hello");
    chat_msgs_push(&future_msgs, fu1);
    chat_msg fa1 = {0}; fa1.role = xstrdup("assistant");
    fa1.reasoning = xstrdup("first reasoning");
    fa1.content = xstrdup("Hi there");
    chat_msgs_push(&future_msgs, fa1);
    chat_msg fu2 = {0}; fu2.role = xstrdup("user"); fu2.content = xstrdup("How are you?");
    chat_msgs_push(&future_msgs, fu2);
    chat_msg fa2 = {0}; fa2.role = xstrdup("assistant");
    fa2.reasoning = xstrdup("second reasoning");
    fa2.content = xstrdup(content2);
    chat_msgs_push(&future_msgs, fa2);
    chat_msg fu3 = {0}; fu3.role = xstrdup("user"); fu3.content = xstrdup("Great");
    chat_msgs_push(&future_msgs, fu3);

    char *future = render_chat_prompt_text(&future_msgs, NULL, NULL, PULSAR_THINK_HIGH);
    /* Both reasonings dropped */
    TEST_ASSERT(strstr(future, "first reasoning") == NULL);
    TEST_ASSERT(strstr(future, "second reasoning") == NULL);
    /* Canonical is a prefix of future */
    TEST_ASSERT(strlen(future) > canonical.len);
    TEST_ASSERT(!memcmp(future, canonical.ptr, canonical.len));

    free(future);
    buf_free(&canonical);
    free(prompt_text);
    chat_msgs_free(&turn2_prefix);
    chat_msgs_free(&future_msgs);
}



static void test_thinking_canonical_with_tools_preserves_reasoning(void) {
    /* When tools ARE present, reasoning is preserved in re-render.
     * The toolless thinking live binding should NOT fire (has_tools gate),
     * and the tool-call replay path handles it.  Verify the template
     * preserves reasoning when tool_context is true. */
    const char *tool_schemas = "{\"name\":\"bash\"}";

    chat_msgs msgs = {0};
    chat_msg u = {0};
    u.role = xstrdup("user");
    u.content = xstrdup("run ls");
    chat_msgs_push(&msgs, u);

    char *prompt_text = render_chat_prompt_text(&msgs, tool_schemas, NULL, PULSAR_THINK_HIGH);
    size_t pt_len = strlen(prompt_text);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 7, "<think>", 7));

    /* With tools, next render KEEPS reasoning */
    chat_msgs history = {0};
    chat_msg hu = {0}; hu.role = xstrdup("user"); hu.content = xstrdup("run ls");
    chat_msgs_push(&history, hu);
    chat_msg ha = {0}; ha.role = xstrdup("assistant");
    ha.reasoning = xstrdup("I should run bash");
    ha.content = xstrdup("Here you go");
    chat_msgs_push(&history, ha);
    chat_msg hu2 = {0}; hu2.role = xstrdup("user"); hu2.content = xstrdup("thanks");
    chat_msgs_push(&history, hu2);

    char *future = render_chat_prompt_text(&history, tool_schemas, NULL, PULSAR_THINK_HIGH);
    /* Reasoning IS preserved when tools present */
    TEST_ASSERT(strstr(future, "I should run bash") != NULL);
    TEST_ASSERT(strstr(future, "<think>I should run bash</think>") != NULL);

    free(future);
    free(prompt_text);
    chat_msgs_free(&msgs);
    chat_msgs_free(&history);
}



static void test_thinking_canonical_non_thinking_mode_noop(void) {
    /* When thinking is disabled (deepseek-chat), prompt_text ends with
     * </think> not <think>.  The toolless thinking live binding is a no-op
     * (early return on memcmp check). */
    chat_msgs msgs = {0};
    chat_msg u = {0};
    u.role = xstrdup("user");
    u.content = xstrdup("Hello");
    chat_msgs_push(&msgs, u);

    char *prompt_text = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_NONE);
    size_t pt_len = strlen(prompt_text);
    /* Should end with </think>, not <think> */
    TEST_ASSERT(pt_len >= 8);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 8, "</think>", 8));
    /* Does NOT end with <think> */
    TEST_ASSERT(memcmp(prompt_text + pt_len - 7, "<think>", 7) != 0);

    free(prompt_text);
    chat_msgs_free(&msgs);
}



static void test_unterminated_think_stays_off_content(void) {
    /* Generation that ends inside the think block (token cap / stop) must
     * surface as reasoning_content, never as visible content — clients that
     * score or display the answer channel would otherwise receive raw
     * chain-of-thought (the tool-eval-bench MMLU/IFEval artifact). Covers
     * both the generated "<think>" opener and the prompt-pre-opened form. */
    const char *cases[] = {
        "<think>We need to compute the index of the subgroup",
        "We need to compute the index of the subgroup",
    };
    for (size_t i = 0; i < 2; i++) {
        char *content = NULL, *reasoning = NULL;
        tool_calls calls = {0};
        TEST_ASSERT(parse_generated_message_ex(cases[i], true, &content,
                                               &reasoning, &calls));
        TEST_ASSERT(content && content[0] == '\0');
        TEST_ASSERT(reasoning &&
                    !strcmp(reasoning, "We need to compute the index of the subgroup"));
        TEST_ASSERT(calls.len == 0);
        free(content);
        free(reasoning);
        tool_calls_free(&calls);
    }
}



static void test_kv_admission_budget_math(void) {
    const uint64_t GiB = 1024ull * 1024ull * 1024ull;

    /* Budget = usable - weights - overhead - free floor, clamped at 0 (the
     * floor term is the 2026-07-13 lockup fix: a fully committed budget must
     * still leave PULSAR_SERVER_MEM_FLOOR_BYTES of the machine free). */
    TEST_ASSERT(server_kv_budget_bytes(91ull * GiB) ==
                PULSAR_SERVER_USABLE_BYTES - 91ull * GiB -
                PULSAR_SERVER_PROCESS_OVERHEAD_BYTES - PULSAR_SERVER_MEM_FLOOR_BYTES);
    TEST_ASSERT(server_kv_budget_bytes(200ull * GiB) == 0);  /* weights > usable: no underflow */
    /* Reserves alone (no weights) must also clamp, not underflow. */
    TEST_ASSERT(server_kv_budget_bytes(PULSAR_SERVER_USABLE_BYTES) == 0);

    /* Admission: committed + incoming <= budget, with overflow-safe compare. */
    TEST_ASSERT(server_kv_admits(26ull * GiB, 0, 20ull * GiB));
    TEST_ASSERT(server_kv_admits(26ull * GiB, 20ull * GiB, 6ull * GiB));   /* exact fit */
    TEST_ASSERT(!server_kv_admits(26ull * GiB, 20ull * GiB, 7ull * GiB));  /* over by 1 GiB */
    TEST_ASSERT(!server_kv_admits(26ull * GiB, 0, 27ull * GiB));           /* lone over-budget */

    /* GB10 production shape (2026-07-15 re-measure: 18 GiB steady-state
     * process overhead, 4 GiB kernel-breathing-room floor):
     * usable 121 GiB − weights ~85.4 GiB − overhead 18 GiB − floor 4 GiB
     * ⇒ budget ~13.6 GiB. Slot 0 at ctx=65536/pc=4096 costs ~4.6 GiB
     * (measured) and must admit at startup; a doubled ~9.2 GiB session (the
     * ctx=131072 upper bound) must also admit alone; the THIRD 4.6 GiB slot
     * (13.8 GiB committed) must be refused — the 2026-07-13 incident shape
     * admitted three. */
    const uint64_t MiB = 1024ull * 1024ull;
    const uint64_t gb10_weights = 87450ull * MiB;              /* ~85.4 GiB */
    const uint64_t gb10_budget = server_kv_budget_bytes(gb10_weights);
    TEST_ASSERT(gb10_budget == PULSAR_SERVER_USABLE_BYTES - gb10_weights -
                PULSAR_SERVER_PROCESS_OVERHEAD_BYTES - PULSAR_SERVER_MEM_FLOOR_BYTES);
    TEST_ASSERT(gb10_budget > 13ull * GiB && gb10_budget < 14ull * GiB);   /* ~13.6 GiB */
    const uint64_t slot64k = 4710ull * MiB;                    /* ~4.6 GiB @ ctx 64k */
    TEST_ASSERT(server_kv_admits(gb10_budget, 0, slot64k));               /* slot 0, 64k */
    TEST_ASSERT(server_kv_admits(gb10_budget, 0, 2ull * slot64k));        /* slot 0, 128k bound */
    TEST_ASSERT(server_kv_admits(gb10_budget, slot64k, slot64k));         /* second slot */
    TEST_ASSERT(!server_kv_admits(gb10_budget, 2ull * slot64k, slot64k)); /* third refused */
}



/* MemAvailable floor backstop (2026-07-15 re-measure). The floor is kernel
 * breathing room ONLY: process-fixed costs are the ledger's 18 GiB overhead
 * reserve, and the ~8.7 GiB lazy first-request CUDA allocations erode
 * MemAvailable INSIDE that already-subtracted reserve. The old 6 GiB floor
 * double-counted that caution and vetoed sessions the ledger legally
 * admitted. */
static void test_mem_floor_admits_warmed_box_shape(void) {
    const uint64_t MiB = 1024ull * 1024ull;
    const uint64_t GiB = 1024ull * MiB;

    /* The 2026-07-14 Tier-1 exit-gate incident: warmed box, two sessions
     * live, third 2.5 GiB session sees MemAvailable 8.39 GiB. The 6 GiB
     * floor demanded 8.50 and refused (a 0.11 GiB miss with ~5.9 GiB truly
     * free at full commit); the 4 GiB floor must admit — post-admission
     * MemAvailable stays >= ~5.89 GiB, well above the backstop. */
    const uint64_t est = 2560ull * MiB;                     /* 2.5 GiB session */
    TEST_ASSERT(server_mem_floor_admits(8590ull * MiB, est));  /* 8.39 GiB avail */

    /* Warmed-box full-commit steady state measured 2026-07-15: 5.96 GiB
     * avail with three live sessions. A further session would leave ~3.4,
     * below the backstop: refuse. */
    TEST_ASSERT(!server_mem_floor_admits(6103ull * MiB, est)); /* 5.96 GiB avail */

    /* A genuinely tight box must always refuse. */
    TEST_ASSERT(!server_mem_floor_admits(4ull * GiB, est));
    /* Exact boundary: est + floor. */
    TEST_ASSERT(server_mem_floor_admits(est + PULSAR_SERVER_MEM_FLOOR_BYTES, est));
    TEST_ASSERT(!server_mem_floor_admits(est + PULSAR_SERVER_MEM_FLOOR_BYTES - 1, est));
    /* Unreadable /proc/meminfo (avail == 0) fails closed. */
    TEST_ASSERT(!server_mem_floor_admits(0, est));
    /* Overflow guard: absurd estimate must refuse, not wrap. */
    TEST_ASSERT(!server_mem_floor_admits(UINT64_MAX, UINT64_MAX - 1ull * GiB));
}



/* Multi-session increment 4: eviction is the first runtime session-free path,
 * so the ledger must balance EXACTLY across provision→evict→provision cycles
 * — each provisioning commits the session's ACTUAL allocator bytes and each
 * eviction releases that same stored value. */
static void test_session_eviction_ledger_math(void) {
    const uint64_t GiB = 1024ull * 1024ull * 1024ull;
    const uint64_t MiB = 1024ull * 1024ull;
    const uint64_t budget = server_kv_budget_bytes(87450ull * MiB); /* ~13.6 GiB */
    const uint64_t slot0 = 4710ull * MiB;   /* startup slot, ctx 64k measured */
    const uint64_t a = 2560ull * MiB;       /* lazy 64k slot (pc 2048 shape) */
    const uint64_t b = 2571ull * MiB;       /* same shape, distinct actual */

    /* provision slot0 + a + b, filling most of the budget */
    uint64_t committed = slot0;
    TEST_ASSERT(server_kv_admits(budget, committed, a));
    committed += a;
    TEST_ASSERT(server_kv_admits(budget, committed, b));
    committed += b;
    /* budget full for another slot0-sized session: admission refuses
     * (9841 + 4710 = 14551 MiB > ~13.6 GiB budget) */
    TEST_ASSERT(!server_kv_admits(budget, committed, slot0));

    /* evict a: the exact committed value comes back, and the freed budget
     * admits an equal-shape provisioning again */
    committed = server_ledger_release(committed, a);
    TEST_ASSERT(committed == slot0 + b);
    TEST_ASSERT(server_kv_admits(budget, committed, a));
    committed += a;
    TEST_ASSERT(committed == slot0 + b + a);

    /* evict everything back down to slot 0: balance is exact, not approximate */
    committed = server_ledger_release(committed, b);
    committed = server_ledger_release(committed, a);
    TEST_ASSERT(committed == slot0);
    committed = server_ledger_release(committed, slot0);
    TEST_ASSERT(committed == 0);

    /* releasing more than is committed means the pairing broke: clamp to 0
     * (warns loudly; the MemAvailable floor backstops the over-admission) */
    TEST_ASSERT(server_ledger_release(1ull * GiB, 2ull * GiB) == 0);
    TEST_ASSERT(server_ledger_release(0, 1) == 0);
}



/* Victim selection: LRU over IDLE provisioned slots, slot 0 pinned, active
 * and protected slots skipped, ties broken by smallest committed bytes
 * (cheapest to bring back). Pure host-field selection — no session is ever
 * touched. */
static void test_session_eviction_victim_selection(void) {
    const uint64_t GiB = 1024ull * 1024ull * 1024ull;
    session_slot slots[PULSAR_SESSION_POOL_CAP];
    memset(slots, 0, sizeof(slots));
    for (int i = 0; i < PULSAR_SESSION_POOL_CAP; i++) slots[i].provisioned = true;
    slots[0].last_serviced_us = 1;              /* oldest of all — but pinned */
    slots[0].est_cost_bytes = 9ull * GiB;
    slots[1].last_serviced_us = 100;
    slots[1].est_cost_bytes = 3ull * GiB;
    slots[2].last_serviced_us = 50;
    slots[2].est_cost_bytes = 2ull * GiB;
    slots[3].last_serviced_us = 50;             /* LRU tie with slot 2 */
    slots[3].est_cost_bytes = 1ull * GiB;       /* ...but cheaper to restore */

    TEST_ASSERT(server_evict_pick_victim(slots, 4, NULL) == 3); /* LRU tie-break */
    bool protect[PULSAR_SESSION_POOL_CAP] = {0};
    protect[3] = true;
    TEST_ASSERT(server_evict_pick_victim(slots, 4, protect) == 2); /* protected skipped */
    slots[2].active_job = (struct job *)&slots;                    /* busy skipped */
    TEST_ASSERT(server_evict_pick_victim(slots, 4, protect) == 1);
    slots[1].provisioned = false;                                  /* hole skipped */
    TEST_ASSERT(server_evict_pick_victim(slots, 4, protect) == -1);
    protect[3] = false;
    TEST_ASSERT(server_evict_pick_victim(slots, 4, protect) == 3);
    /* n_slots bounds the scan: slot 3 invisible when only 3 are published */
    TEST_ASSERT(server_evict_pick_victim(slots, 3, protect) == -1);
    /* slot 0 alone is never a victim */
    TEST_ASSERT(server_evict_pick_victim(slots, 1, NULL) == -1);
}



/* Routing decision (task #30): the choose-vs-provision gate. Through v0.2.0
 * the gate was best_common == 0, unreachable for rendered chat traffic
 * (every rendered prompt shares the template header — measured common = 4-9
 * tokens across DISTINCT conversations in the task-#24 bounce repro), so
 * sequential conversations always clobbered slot 0 and the pool provisioned
 * zero slots. The classifier reclassifies header-deep matches as no-match
 * for the routing decision only. T mirrors the startup-derived threshold
 * (template header tokens + PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS);
 * the decision must hold for any plausible derivation, so the matrix uses
 * the allowance floor. Owner routing (live tool-state continuations) sits
 * UPSTREAM of this classifier in choose_slot_for_job and is untouched;
 * its slot lookup needs a live session frontier, so it is exercised by the
 * e2e gates rather than here. */
static void test_slot_route_trivial_match_decision(void) {
    const int T = PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS;
    /* Legacy single-threshold behavior: share_ceiling == protect_floor == T. */
    /* zero common vs a warm 5.2k-token conversation: provision (the one
     * case the v0.2.0 gate did handle — behavior kept) */
    TEST_ASSERT(server_slot_match_is_trivial(0, 5200, T, T));
    /* trivial common (template header only, the measured bounce shape):
     * THE FIX — a different conversation must not clobber a warm slot */
    TEST_ASSERT(server_slot_match_is_trivial(4, 5200, T, T));
    TEST_ASSERT(server_slot_match_is_trivial(9, 5200, T, T));
    TEST_ASSERT(server_slot_match_is_trivial(T - 1, 5200, T, T));
    /* real common (long shared prefix: a client resending a longer version
     * of the same prompt, or the documented stateless-continuation
     * pattern): reuse the warm slot, never provision away from it */
    TEST_ASSERT(!server_slot_match_is_trivial(T, 5200, T, T));
    TEST_ASSERT(!server_slot_match_is_trivial(5175, 5900, T, T));
    /* empty slot: nothing to protect, reuse it (never "clobbers") */
    TEST_ASSERT(!server_slot_match_is_trivial(0, 0, T, T));
    /* short same-conversation continuation: common covers nearly the whole
     * slot state — stay on the warm slot even though common < T */
    TEST_ASSERT(!server_slot_match_is_trivial(38, 40, T, T));
    /* sub-threshold warm tail past the match: clobbering costs a sub-second
     * re-prefill, a fresh provisioning costs seconds — reuse (deliberate
     * semantic change from v0.2.0, which provisioned for pos in (0, T)) */
    TEST_ASSERT(!server_slot_match_is_trivial(0, T - 1, T, T));
    /* boundary: destroyed tail exactly at the threshold provisions */
    TEST_ASSERT(server_slot_match_is_trivial(0, T, T, T));
    TEST_ASSERT(server_slot_match_is_trivial(T - 1, 2 * T - 1, T, T));
    TEST_ASSERT(!server_slot_match_is_trivial(T - 1, 2 * T - 2, T, T));

    /* Tools-client bounce: two unrelated Claude Code conversations share a
     * large fixed tool/system prefix. The caller lifts share_ceiling to this
     * job's anchor (say ~1800 tokens) while protect_floor stays T. A match at
     * the shared-prefix depth must now read as TRIVIAL and provision fresh,
     * where the old single-T classifier called it "real" and clobbered. */
    const int CEIL = 1800;
    TEST_ASSERT(server_slot_match_is_trivial(1500, 5200, CEIL, T));   /* the fix */
    TEST_ASSERT(server_slot_match_is_trivial(CEIL - 1, 5200, CEIL, T));
    /* genuine continuation past the shared scaffolding: not trivial, reuse */
    TEST_ASSERT(!server_slot_match_is_trivial(CEIL, 5200, CEIL, T));
    TEST_ASSERT(!server_slot_match_is_trivial(3000, 5200, CEIL, T));
    /* raised ceiling must NOT change the protect side: a slot with only a
     * sub-floor tail past a shared-prefix match is still reused, not protected */
    TEST_ASSERT(!server_slot_match_is_trivial(1500, 1500 + T - 1, CEIL, T));
}



/* Routing probe for thinking chats (task #30): a slot whose live thinking
 * binding byte-matches the request's visible transcript is that
 * conversation's warm continuation, even when the token common prefix is
 * header-short (the client replays visible content; the slot's sampled
 * frontier holds the hidden reasoning). choose_slot_for_job must route such
 * a request back to its slot instead of provisioning a "fresh conversation"
 * slot for it. Host fields only — the session is never touched (the caller
 * passes the live position). */
static void test_thinking_binding_routes_visible_continuation(void) {
    server s;
    memset(&s, 0, sizeof(s));
    pthread_mutex_init(&s.tool_mu, NULL);

    char vis[] = "<BOS>sys<U>hi<A></think>hello<EOS>";
    session_slot sl;
    memset(&sl, 0, sizeof(sl));
    sl.thinking_live.valid = true;
    sl.thinking_live.visible_text = vis;
    sl.thinking_live.visible_len = strlen(vis);
    sl.thinking_live.live_tokens = 40;

    char prompt[] = "<BOS>sys<U>hi<A></think>hello<EOS><U>again";
    request req;
    memset(&req, 0, sizeof(req));
    req.kind = REQ_CHAT;
    req.api = API_OPENAI;
    req.prompt_text = prompt;

    /* the continuation matches its slot at the remembered frontier */
    TEST_ASSERT(s.thinking_live_binds_prompt(&sl, &req, 40) == strlen(vis));
    /* frontier moved (slot served someone else meanwhile): stale, no match */
    TEST_ASSERT(s.thinking_live_binds_prompt(&sl, &req, 41) == 0);
    /* a different conversation sharing only the header must not match —
     * longer than the binding key so the BYTE COMPARE is what rejects it,
     * not the visible_len < prompt_len guard (2026-07-16 review) */
    char other[] = "<BOS>sys<U>completely different much longer conversation";
    req.prompt_text = other;
    TEST_ASSERT(s.thinking_live_binds_prompt(&sl, &req, 40) == 0);
    /* an exact replay (no new suffix) is not a continuation */
    char exact[] = "<BOS>sys<U>hi<A></think>hello<EOS>";
    req.prompt_text = exact;
    TEST_ASSERT(s.thinking_live_binds_prompt(&sl, &req, 40) == 0);
    /* owner-routed protocols resolve via live call ids upstream; the probe
     * must not claim them */
    req.prompt_text = prompt;
    req.api = API_RESPONSES;
    TEST_ASSERT(s.thinking_live_binds_prompt(&sl, &req, 40) == 0);
    req.api = API_OPENAI;
    req.kind = REQ_COMPLETION;
    TEST_ASSERT(s.thinking_live_binds_prompt(&sl, &req, 40) == 0);
    /* invalidated binding (clobbered/evicted slot) never matches */
    req.kind = REQ_CHAT;
    sl.thinking_live.valid = false;
    TEST_ASSERT(s.thinking_live_binds_prompt(&sl, &req, 40) == 0);

    pthread_mutex_destroy(&s.tool_mu);
}



/* Multi-session increment 2: while a job is bound to a slot, the worker's
 * send_all() routes through a slot_writer — writes that do not fit the socket
 * buffer defer instead of blocking, and flushes deliver every byte in order.
 * The wire stream must be identical to the blocking path. */
static void test_slot_writer_defers_and_preserves_order(void) {
    signal(SIGPIPE, SIG_IGN);
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    int small = 4096;
    setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &small, sizeof(small));
    setsockopt(sv[1], SOL_SOCKET, SO_RCVBUF, &small, sizeof(small));
    set_client_socket_nonblocking(sv[0]);
    set_client_socket_nonblocking(sv[1]);

    slot_writer w;
    slot_writer_init(&w, sv[0]);
    slot_writer_install(&w);

    const size_t total = 512 * 1024;
    char *pattern = (char *)server_xmalloc(total);
    for (size_t i = 0; i < total; i++) {
        pattern[i] = (char)((i * 31u + (i >> 8)) & 0xff);
    }
    char *received = (char *)server_xmalloc(total);
    size_t sent = 0, got = 0;
    bool deferred = false;

    /* The peer reads nothing during the sends, so the tiny kernel buffers fill
     * and everything past them must defer into the writer queue. */
    while (sent < total) {
        size_t nchunk = 700 + (sent % 900); /* odd sizes straddle buffers */
        if (nchunk > total - sent) nchunk = total - sent;
        TEST_ASSERT(send_all(sv[0], pattern + sent, nchunk)); /* defers, never fails */
        sent += nchunk;
        if (w.pending.len > w.off) deferred = true;
    }
    TEST_ASSERT(deferred);

    /* Drain: alternate the worker-side flush with peer reads. Bounded so a
     * writer regression that stops delivering (without setting failed) shows
     * up as an assertion instead of a hung test suite. */
    int stagnant = 0;
    while (got < total) {
        TEST_ASSERT(slot_writer_flush(&w));
        char tmp[8192];
        ssize_t r = recv(sv[1], tmp, sizeof(tmp), MSG_DONTWAIT);
        if (r > 0) {
            TEST_ASSERT(got + (size_t)r <= total);
            memcpy(received + got, tmp, (size_t)r);
            got += (size_t)r;
            stagnant = 0;
        } else if (++stagnant >= 100000) {
            TEST_ASSERT(!"slot_writer drain made no progress");
            break;
        }
    }
    TEST_ASSERT(!w.failed);
    TEST_ASSERT(w.pending.len == w.off); /* everything reached the wire */
    TEST_ASSERT(memcmp(pattern, received, total) == 0);

    slot_writer_free(&w); /* also uninstalls */
    free(pattern);
    free(received);
    close(sv[0]);
    close(sv[1]);
}



/* A peer that accepts no bytes past the stall deadline fails the stream, and
 * every later write on the failed writer reports failure — matching the old
 * blocking send_all semantics that the generation loop depends on. */
static void test_slot_writer_stall_times_out(void) {
    signal(SIGPIPE, SIG_IGN);
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    int small = 4096;
    setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &small, sizeof(small));
    setsockopt(sv[1], SOL_SOCKET, SO_RCVBUF, &small, sizeof(small));
    set_client_socket_nonblocking(sv[0]);

    slot_writer w;
    slot_writer_init(&w, sv[0]);
    slot_writer_install(&w);

    char blob[8192];
    memset(blob, 'q', sizeof(blob));
    for (int i = 0; i < 64; i++) {
        TEST_ASSERT(send_all(sv[0], blob, sizeof(blob))); /* peer never reads: defer */
    }
    TEST_ASSERT(w.pending.len > w.off);
    /* Force the deadline instead of sleeping PULSAR_SERVER_SEND_STALL_TIMEOUT_MS. */
    w.stall_deadline_ms = 1;
    TEST_ASSERT(!slot_writer_flush(&w));
    TEST_ASSERT(w.failed);
    TEST_ASSERT(!send_all(sv[0], "x", 1));
    TEST_ASSERT(!slot_writer_drain(&w));

    slot_writer_free(&w);
    close(sv[0]);
    close(sv[1]);
}



/* ---- disk-KV default-on resolution matrix (task #31) ---- */

static char *test_env_save(const char *name) {
    const char *v = getenv(name);
    return v ? xstrdup(v) : NULL;
}

static void test_env_restore(const char *name, char *saved) {
    if (saved) {
        setenv(name, saved, 1);
        free(saved);
    } else {
        unsetenv(name);
    }
}



static void test_kv_disk_default_dir_resolution(void) {
    char *old_xdg = test_env_save("XDG_CACHE_HOME");
    char *old_home = test_env_save("HOME");

    /* XDG_CACHE_HOME wins; nonexistent model path falls back to its raw
     * basename; the .gguf extension is stripped case-insensitively. */
    setenv("XDG_CACHE_HOME", "/tmp/ds4-kvtest-xdg", 1);
    char *d = server_default_kv_disk_dir("/no/such/dir/ds4flash-test.gguf");
    TEST_ASSERT(d && !strcmp(d, "/tmp/ds4-kvtest-xdg/ds4/kv-ds4flash-test"));
    free(d);

    /* The gguf/model.gguf ACTIVE-POINTER symlink must key by the versioned
     * artifact it points at, not by the pointer name. */
    char tmpl[] = "/tmp/ds4-kvtest-XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (dir) {
        char artifact[PATH_MAX], pointer[PATH_MAX];
        snprintf(artifact, sizeof(artifact), "%s/ds4flash-v9-test.gguf", dir);
        snprintf(pointer, sizeof(pointer), "%s/model.gguf", dir);
        FILE *fp = fopen(artifact, "w");
        TEST_ASSERT(fp != NULL);
        if (fp) fclose(fp);
        TEST_ASSERT(symlink(artifact, pointer) == 0);
        d = server_default_kv_disk_dir(pointer);
        TEST_ASSERT(d && !strcmp(d, "/tmp/ds4-kvtest-xdg/ds4/kv-ds4flash-v9-test"));
        free(d);
        unlink(pointer);
        unlink(artifact);
        rmdir(dir);
    }

    /* Without XDG_CACHE_HOME, fall back to ~/.cache; shell-hostile bytes in
     * the model name are sanitized. */
    unsetenv("XDG_CACHE_HOME");
    setenv("HOME", "/tmp/ds4-kvtest-home", 1);
    d = server_default_kv_disk_dir("weird name!.GGUF");
    TEST_ASSERT(d && !strcmp(d, "/tmp/ds4-kvtest-home/.cache/ds4/kv-weird_name_"));
    free(d);

    /* A relative XDG_CACHE_HOME is ignored per the XDG spec (it would key
     * the cache off the current working directory). */
    setenv("XDG_CACHE_HOME", "relative-cache", 1);
    d = server_default_kv_disk_dir("x.gguf");
    TEST_ASSERT(d && !strcmp(d, "/tmp/ds4-kvtest-home/.cache/ds4/kv-x"));
    free(d);
    unsetenv("XDG_CACHE_HOME");

    /* No cache home at all: resolution reports NULL (server then runs with
     * the disk cache disabled instead of guessing a path). */
    unsetenv("HOME");
    TEST_ASSERT(server_default_kv_disk_dir("x.gguf") == NULL);

    test_env_restore("XDG_CACHE_HOME", old_xdg);
    test_env_restore("HOME", old_home);
}



static void test_kv_disk_flag_matrix(void) {
    char *old_xdg = test_env_save("XDG_CACHE_HOME");
    setenv("XDG_CACHE_HOME", "/tmp/ds4-kvtest-xdg", 1);

    /* Unset: the default directory is resolved (default-on). */
    {
        char *argv[] = {(char *)"pulsar-server"};
        server_config c = parse_options(1, argv);
        server_resolve_kv_disk_dir(&c);
        TEST_ASSERT(c.kv_disk_dir != NULL);
        TEST_ASSERT(c.kv_disk_dir &&
                    !strncmp(c.kv_disk_dir, "/tmp/ds4-kvtest-xdg/ds4/kv-",
                             strlen("/tmp/ds4-kvtest-xdg/ds4/kv-")));
        free((char *)c.kv_disk_dir);
    }

    /* Explicit path: used verbatim, exactly as before. */
    {
        char *argv[] = {(char *)"pulsar-server",
                        (char *)"--kv-disk-dir", (char *)"/tmp/explicit-kv"};
        server_config c = parse_options(3, argv);
        server_resolve_kv_disk_dir(&c);
        TEST_ASSERT(c.kv_disk_dir && !strcmp(c.kv_disk_dir, "/tmp/explicit-kv"));
    }

    /* Empty value: opt-out. */
    {
        char *argv[] = {(char *)"pulsar-server",
                        (char *)"--kv-disk-dir", (char *)""};
        server_config c = parse_options(3, argv);
        server_resolve_kv_disk_dir(&c);
        TEST_ASSERT(c.kv_disk_dir == NULL);
    }

    /* --no-kv-disk: opt-out. */
    {
        char *argv[] = {(char *)"pulsar-server", (char *)"--no-kv-disk"};
        server_config c = parse_options(2, argv);
        server_resolve_kv_disk_dir(&c);
        TEST_ASSERT(c.kv_disk_dir == NULL);
    }

    /* Last kv-disk flag wins: opt-out then explicit path re-enables. */
    {
        char *argv[] = {(char *)"pulsar-server", (char *)"--no-kv-disk",
                        (char *)"--kv-disk-dir", (char *)"/tmp/explicit-kv"};
        server_config c = parse_options(4, argv);
        server_resolve_kv_disk_dir(&c);
        TEST_ASSERT(c.kv_disk_dir && !strcmp(c.kv_disk_dir, "/tmp/explicit-kv"));
    }

    test_env_restore("XDG_CACHE_HOME", old_xdg);
}



static void test_kv_cache_open_unusable_dir_disables(void) {
    /* Uncreatable path: open fails, cache stays disabled, no crash. */
    kv_disk_cache kc = {0};
    TEST_ASSERT(!kv_cache_open(&kc, "/proc/ds4-kvtest-nope/kv", 64, false,
                               kv_cache_default_options()));
    TEST_ASSERT(!kc.enabled);
    kv_cache_close(&kc);

    /* Pre-existing read-only directory: refused up front (writability probe)
     * instead of failing store-by-store later.  Mode bits do not bind root. */
    char tmpl[] = "/tmp/ds4-kvtest-ro-XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (dir) {
        TEST_ASSERT(chmod(dir, 0500) == 0);
        kv_disk_cache ro = {0};
        bool opened = kv_cache_open(&ro, dir, 64, false,
                                    kv_cache_default_options());
        if (geteuid() == 0) {
            TEST_ASSERT(opened);
        } else {
            TEST_ASSERT(!opened);
            TEST_ASSERT(!ro.enabled);
        }
        kv_cache_close(&ro);
        chmod(dir, 0700);
        rmdir(dir);
    }
}



/* append_logprob_text_json substitutes U+FFFD for every ill-formed UTF-8
 * sequence in a logprob entry's "token" string — including the ones a bare
 * 10xxxxxx continuation check admits: overlongs, UTF-16 surrogates, and
 * > U+10FFFF.  "bytes" carries the exact bytes; this guard is about keeping
 * the JSON well-formed for strict clients, so the boundary rows of Unicode
 * Table 3-7 are the teeth: the largest legal value on one side of each
 * second-byte constraint and the smallest illegal value on the other. */
/* L192 item 7: every id the server mints is prefix + hex from the OS RNG;
 * two calls never agree and the chat/completion prefixes are the wire's. */
static void test_random_prefixed_id_format(void) {
    char a[96], b[96];
    random_prefixed_id(a, sizeof(a), "chatcmpl-", 12);
    random_prefixed_id(b, sizeof(b), "chatcmpl-", 12);
    TEST_ASSERT(strncmp(a, "chatcmpl-", 9) == 0 && strlen(a) == 9 + 24);
    TEST_ASSERT(strspn(a + 9, "0123456789abcdef") == 24);
    TEST_ASSERT(strcmp(a, b) != 0);
    random_prefixed_id(a, sizeof(a), "cmpl-", 12);
    TEST_ASSERT(strncmp(a, "cmpl-", 5) == 0 && strlen(a) == 5 + 24);
    char small[8];
    random_prefixed_id(small, sizeof(small), "resp_", 12);   /* truncates, never overruns */
    TEST_ASSERT(strlen(small) < sizeof(small) && strncmp(small, "resp_", 5) == 0);
}

static void test_logprob_token_json_sanitizes_ill_formed_utf8(void) {
    static const struct { const char *in; size_t n; const char *out; } cases[] = {
        {"hi",               2, "\"hi\""},
        {"\xe4\xb8\xad",     3, "\"\xe4\xb8\xad\""},          /* U+4E2D */
        {"\xf0\x9f\x9a\x80", 4, "\"\xf0\x9f\x9a\x80\""},      /* U+1F680 */
        {"\xe0\xa0\x80",     3, "\"\xe0\xa0\x80\""},          /* smallest legal E0 */
        {"\xed\x9f\xbf",     3, "\"\xed\x9f\xbf\""},          /* last before surrogates */
        {"\xf0\x90\x80\x80", 4, "\"\xf0\x90\x80\x80\""},      /* U+10000 */
        {"\xf4\x8f\xbf\xbf", 4, "\"\xf4\x8f\xbf\xbf\""},      /* U+10FFFF */
        {"\xed\xa0\x80",     3,                               /* UTF-16 surrogate */
         "\"\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\""},
        {"\xe0\x80\x80",     3,                               /* overlong NUL */
         "\"\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\""},
        {"\xf0\x8f\xbf\xbf", 4,                               /* overlong U+FFFF */
         "\"\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\""},
        {"\xf4\x90\x80\x80", 4,                               /* > U+10FFFF */
         "\"\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\""},
        {"\xe4\xb8",         2,                               /* truncated tail */
         "\"\xef\xbf\xbd\xef\xbf\xbd\""},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        buf b = {0};
        append_logprob_text_json(&b, cases[i].in, cases[i].n);
        const size_t want = strlen(cases[i].out);
        TEST_ASSERT(b.len == want && b.ptr && !memcmp(b.ptr, cases[i].out, want));
        buf_free(&b);
    }
}


/* L179 branch 8 -- L116 tool-call admission to the batched decode lane.
 * Invariant: a slot is admitted iff it is bound (active_job) AND has gen state
 * in GEN_DECODE. The request's has_tools does NOT appear in the decision (a
 * tool-call request rides the batched lane like any other; its forced-greedy
 * payload span is the sampler's business), so the true and false rows of the
 * table must be identical. */
static void test_l179_tool_admission_is_bound_decode_only(void) {
    job j;
    memset(&j, 0, sizeof j);
    gen_state g;
    memset(&g, 0, sizeof g);
    g.j = &j;
    session_slot sl;
    memset(&sl, 0, sizeof sl);
    static const gen_phase phases[] = {GEN_DECODE, GEN_PREFILL_MAIN, GEN_FINISH};
    for (int bound = 0; bound < 2; bound++)
    for (int has_gen = 0; has_gen < 2; has_gen++)
    for (size_t pi = 0; pi < sizeof phases / sizeof phases[0]; pi++) {
        sl.active_job = bound ? &j : NULL;
        sl.gen = has_gen ? &g : NULL;
        g.phase = phases[pi];
        const bool want = bound && has_gen && phases[pi] == GEN_DECODE;
        j.req.has_tools = false;
        const bool no_tools = slot_is_batchable_decode(&sl);
        j.req.has_tools = true;
        const bool tools = slot_is_batchable_decode(&sl);
        TEST_ASSERT(no_tools == want);
        TEST_ASSERT(tools == no_tools);
    }
}

/* L179 branch 12 -- fused-prefill deep-concurrency guard
 * (worker_find_fuse_prefill). Invariant: the fuse is refused iff at least two
 * provisioned, bound, SLOT_DECODING slots sum committed depth STRICTLY above
 * guard_rows; one decoder never trips it however deep, and guard_rows == 0 is
 * the guard off (the inline code skipped the whole block at 0). */
static void test_l179_deep_guard_blocks_two_deep_decoders(void) {
    session_slot slots[4];
    memset(slots, 0, sizeof slots);
    const int rows = 4096;
    for (int i = 0; i < 4; i++) {
        slots[i].provisioned = true;
        slots[i].active_job = (struct job *)&slots;
        slots[i].state = SLOT_DECODING;
        slots[i].bank = (uint32_t)i;
    }
    int n_dec = -1;
    long deep = -1;
    /* two decoders at exactly rows: admits (strict >) */
    slots[0].committed_pos = rows / 2;
    slots[1].committed_pos = rows / 2;
    TEST_ASSERT(!mixed_deep_guard_blocks(slots, 2, rows, &n_dec, &deep));
    TEST_ASSERT(n_dec == 2 && deep == rows);
    /* one row over: refuses */
    slots[1].committed_pos = rows / 2 + 1;
    TEST_ASSERT(mixed_deep_guard_blocks(slots, 2, rows, &n_dec, &deep));
    TEST_ASSERT(deep == rows + 1);
    /* a lone decoder at 10x rows admits */
    slots[0].committed_pos = 10 * rows;
    TEST_ASSERT(!mixed_deep_guard_blocks(slots, 1, rows, NULL, NULL));
    /* a second slot that is NOT decoding (prefilling / unbound /
     * unprovisioned) is not a decoder */
    slots[1].committed_pos = 10 * rows;
    slots[1].state = SLOT_PREFILLING;
    TEST_ASSERT(!mixed_deep_guard_blocks(slots, 2, rows, &n_dec, NULL));
    TEST_ASSERT(n_dec == 1);
    slots[1].state = SLOT_DECODING;
    slots[1].active_job = NULL;
    TEST_ASSERT(!mixed_deep_guard_blocks(slots, 2, rows, NULL, NULL));
    slots[1].active_job = (struct job *)&slots;
    slots[1].provisioned = false;
    TEST_ASSERT(!mixed_deep_guard_blocks(slots, 2, rows, NULL, NULL));
    slots[1].provisioned = true;
    /* restored: two decoders 20x over the guard refuse... */
    TEST_ASSERT(mixed_deep_guard_blocks(slots, 2, rows, NULL, NULL));
    /* ...and guard_rows == 0 never blocks (guard off) */
    TEST_ASSERT(!mixed_deep_guard_blocks(slots, 2, 0, NULL, NULL));
    /* n_slots bounds the scan */
    TEST_ASSERT(!mixed_deep_guard_blocks(slots, 1, rows, NULL, NULL));
    TEST_ASSERT(mixed_deep_guard_blocks(slots, 4, rows, &n_dec, NULL));
    TEST_ASSERT(n_dec == 4);
}

/* L179 branch 14 -- provision_bank's MemAvailable floor. Invariant: the FIRST
 * bank (n_provisioned == 0) is never floor-refused, at any gauge reading
 * including an unreadable one (a first-bank refusal is a worker hard-spin);
 * from the second bank on, avail == 0 fails closed and the box must hold
 * marginal + PULSAR_SERVER_MEM_FLOOR_BYTES (same boundary as
 * server_mem_floor_admits). */
static void test_l179_bank_floor_exempts_first_bank(void) {
    const uint64_t MiB = 1024ull * 1024ull;
    const uint64_t GiB = 1024ull * MiB;
    const uint64_t marginal = 2560ull * MiB;                 /* 2.5 GiB bank */
    const uint64_t floor = marginal + PULSAR_SERVER_MEM_FLOOR_BYTES;
    /* first bank: exempt everywhere */
    TEST_ASSERT(!server_bank_floor_refuses(0, 0, marginal));
    TEST_ASSERT(!server_bank_floor_refuses(0, floor - 1, marginal));
    TEST_ASSERT(!server_bank_floor_refuses(0, floor, marginal));
    TEST_ASSERT(!server_bank_floor_refuses(0, 1ull * GiB, marginal));
    /* second bank onward: gauge and floor both bind */
    TEST_ASSERT(server_bank_floor_refuses(1, 0, marginal));
    TEST_ASSERT(server_bank_floor_refuses(1, floor - 1, marginal));
    TEST_ASSERT(!server_bank_floor_refuses(1, floor, marginal));
    TEST_ASSERT(server_bank_floor_refuses(3, 0, marginal));
    TEST_ASSERT(server_bank_floor_refuses(3, floor - 1, marginal));
    TEST_ASSERT(!server_bank_floor_refuses(3, floor, marginal));
    TEST_ASSERT(!server_bank_floor_refuses(3, 100ull * GiB, marginal));
}

/* L179 branch 4 -- park_live_bank before a batched quantum. Invariant: the
 * live bank's checkpoint is saved iff the bank is real (0 <= live < pool)
 * and belongs to NEITHER the decode set NOR the fused prefill slot; a bank
 * the quantum is about to drive reconciles itself in the entry loop, and a
 * no-live-bank (-1) or out-of-pool id has nothing to park. */
static void test_l179_park_live_bank_only_when_not_in_quantum(void) {
    session_slot slots[4];
    memset(slots, 0, sizeof slots);
    for (int i = 0; i < 4; i++) slots[i].bank = (uint32_t)i;
    session_slot *dec[2] = {&slots[0], &slots[1]};
    const session_slot *pf = &slots[2];
    const int pool = 4;
    /* live bank is a decoder: no park */
    TEST_ASSERT(!park_live_bank_needed(0, pool, dec, 2, NULL));
    TEST_ASSERT(!park_live_bank_needed(1, pool, dec, 2, pf));
    /* live bank is the fused prefill slot: no park */
    TEST_ASSERT(!park_live_bank_needed(2, pool, dec, 2, pf));
    /* live bank is in neither: park */
    TEST_ASSERT(park_live_bank_needed(2, pool, dec, 2, NULL));
    TEST_ASSERT(park_live_bank_needed(3, pool, dec, 2, pf));
    TEST_ASSERT(park_live_bank_needed(0, pool, dec, 0, NULL));
    /* no live bank / out of pool: nothing to park */
    TEST_ASSERT(!park_live_bank_needed(-1, pool, dec, 2, pf));
    TEST_ASSERT(!park_live_bank_needed(pool, pool, dec, 2, pf));
    TEST_ASSERT(!park_live_bank_needed(3, 0, dec, 0, NULL));
    /* a NULL entry in dec is skipped, not dereferenced */
    session_slot *holey[2] = {NULL, &slots[3]};
    TEST_ASSERT(park_live_bank_needed(2, pool, holey, 2, NULL));
    TEST_ASSERT(!park_live_bank_needed(3, pool, holey, 2, NULL));
}

/* L179 branch 2 -- worker_main's lane select (w_decode_lane). Invariant:
 * lane 0 with no decoders; lane 3 (spec-batched) iff the drafter is loaded,
 * no decoder has joined a plain batch (n_batched == 0) and EVERY decoder has
 * spec enabled; lane 2 (plain batched) otherwise, including the L118 batch of
 * one; a solo spec decoder is lane 3. Lane 1 is the retired classic lane,
 * reachable only in the gather loop's impossible no-pool/decoders shape --
 * asserted as what the code computes, not as a feature. */
static void test_l179_lane_select_spec_needs_every_decoder(void) {
    gen_state g[4];
    memset(g, 0, sizeof g);
    session_slot slots[4];
    memset(slots, 0, sizeof slots);
    session_slot *dec[4];
    for (int i = 0; i < 4; i++) {
        slots[i].gen = &g[i];
        g[i].dspark_spec_enabled = true;
        dec[i] = &slots[i];
    }
    const int pool = 4;
    /* nothing to decode: idle */
    TEST_ASSERT(server_pick_decode_lane(pool, true, dec, 0, 0) == 0);
    TEST_ASSERT(server_pick_decode_lane(pool, false, dec, 0, 0) == 0);
    /* four spec decoders: spec lane */
    TEST_ASSERT(server_pick_decode_lane(pool, true, dec, 4, 0) == 3);
    /* one non-spec slot among four drags the group to plain */
    g[2].dspark_spec_enabled = false;
    TEST_ASSERT(server_pick_decode_lane(pool, true, dec, 4, 0) == 2);
    g[2].dspark_spec_enabled = true;
    /* a slot with no gen state likewise */
    slots[3].gen = NULL;
    TEST_ASSERT(server_pick_decode_lane(pool, true, dec, 4, 0) == 2);
    slots[3].gen = &g[3];
    /* a plain batch in flight locks the lane even when all spec */
    TEST_ASSERT(server_pick_decode_lane(pool, true, dec, 4, 1) == 2);
    /* ...and a decoder that has joined the plain lane says so itself */
    g[1].batch_active = true;
    TEST_ASSERT(server_pick_decode_lane(pool, true, dec, 4, 0) == 2);
    g[1].batch_active = false;
    /* no drafter: plain */
    TEST_ASSERT(server_pick_decode_lane(pool, false, dec, 4, 0) == 2);
    /* L118 batch of one: a solo spec decoder is lane 3, solo plain lane 2 */
    TEST_ASSERT(server_pick_decode_lane(pool, true, dec, 1, 0) == 3);
    TEST_ASSERT(server_pick_decode_lane(pool, false, dec, 1, 0) == 2);
    TEST_ASSERT(server_pick_decode_lane(pool, true, dec, 1, 1) == 2);
    /* no pool: idle with no decoders, else the retired classic code 1 */
    TEST_ASSERT(server_pick_decode_lane(0, true, dec, 0, 0) == 0);
    TEST_ASSERT(server_pick_decode_lane(0, true, dec, 1, 0) == 1);
    TEST_ASSERT(server_pick_decode_lane(0, false, dec, 4, 0) == 1);
}

/* Geometric survival for one bank: np pendings at per-position confidence c,
 * surv[j] = c^(j+1) -- the cumprod spec_alloc_rows' caller derives from the
 * drafter carry. */
static void l179_fill_surv(float surv[][16], uint32_t *npend, int i, uint32_t np, float c) {
    float p = 1.0f;
    npend[i] = np;
    for (uint32_t j = 0; j < np; j++) {
        p *= c;
        surv[i][j] = p;
    }
}

/* L179 branch 1 -- the L117 cross-bank K allocator (spec_alloc_rows).
 * Invariants: (a) ISOLATION -- while base rows + every pending fit
 * PULSAR_SPEC_ROW_BUDGET the allocator returns 0 and admits every bank whole
 * (k_alloc[i] == npend[i]) at ANY threshold, so a stale partner carry can
 * never shape this bank's round; (b) OVERFLOW -- it returns 1, each bank
 * gets a prefix (k_alloc[i] <= npend[i]), the base rows plus the admitted
 * rows spend the budget exactly, and the admitted set is the global best:
 * no admitted candidate scores below any unadmitted one; (c) the COST-TABLE
 * cut -- once the best remaining candidate is below thr admission stops,
 * *thr_cut_rows counts what it left, and the budget may go unspent. */
static void test_l179_spec_alloc_rows_isolation_and_ranked_overflow(void) {
    const int B = (int)PULSAR_SPEC_ROW_BUDGET;
    float surv[PULSAR_SESSION_POOL_CAP][16];
    uint32_t npend[PULSAR_SESSION_POOL_CAP];
    int k_alloc[PULSAR_SESSION_POOL_CAP];
    int cut = -1;
    memset(surv, 0, sizeof surv);
    memset(npend, 0, sizeof npend);
    const float thr_fallback = 6.0f / 45.0f;   /* spec_ms_per_tok_ema unset */
    const float thr_live = 6.0f / 30.0f;       /* a live EMA of 30 ms/tok */

    /* (a) demand 3 + 12 = 15 < 16, one bank with hopeless confidence, a
     * fourth bank not decoding (npend 0): everything admitted, no cut. */
    l179_fill_surv(surv, npend, 0, 4, 0.95f);
    l179_fill_surv(surv, npend, 1, 5, 0.01f);
    l179_fill_surv(surv, npend, 2, 3, 0.80f);
    npend[3] = 0;
    TEST_ASSERT(spec_alloc_rows(surv, npend, 4, 3, thr_live, k_alloc, &cut) == 0);
    for (int i = 0; i < 4; i++) TEST_ASSERT(k_alloc[i] == (int)npend[i]);
    TEST_ASSERT(cut == 0);
    /* demand exactly the budget (3 + 13 = 16) still fits */
    l179_fill_surv(surv, npend, 1, 6, 0.01f);
    TEST_ASSERT(spec_alloc_rows(surv, npend, 4, 3, 0.99f, k_alloc, &cut) == 0);
    for (int i = 0; i < 4; i++) TEST_ASSERT(k_alloc[i] == (int)npend[i]);
    TEST_ASSERT(cut == 0);

    /* (b) demand 3 + 17 = 20 > 16 with every survival above thr: ranked. The
     * 13 rows go to the 13 highest survivals: bank 0 (0.95^k) all 6,
     * bank 1 (0.9^k) 5, bank 2 (0.8^k) 2. */
    l179_fill_surv(surv, npend, 0, 6, 0.95f);
    l179_fill_surv(surv, npend, 1, 6, 0.90f);
    l179_fill_surv(surv, npend, 2, 5, 0.80f);
    TEST_ASSERT(spec_alloc_rows(surv, npend, 4, 3, thr_fallback, k_alloc, &cut) == 1);
    TEST_ASSERT(cut == 0);
    int admitted = 0;
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT(k_alloc[i] >= 0 && k_alloc[i] <= (int)npend[i]);
        admitted += k_alloc[i];
    }
    TEST_ASSERT(3 + admitted == B);
    TEST_ASSERT(k_alloc[0] == 6 && k_alloc[1] == 5 && k_alloc[2] == 2 && k_alloc[3] == 0);
    /* global best: the weakest admitted row beats the strongest unadmitted */
    float min_admitted = 2.0f, max_unadmitted = -1.0f;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < k_alloc[i]; j++)
            if (surv[i][j] < min_admitted) min_admitted = surv[i][j];
        if ((uint32_t)k_alloc[i] < npend[i] && surv[i][k_alloc[i]] > max_unadmitted)
            max_unadmitted = surv[i][k_alloc[i]];
    }
    TEST_ASSERT(min_admitted >= max_unadmitted);

    /* (c) thr above every survival: nothing admitted, all 17 rows cut */
    TEST_ASSERT(spec_alloc_rows(surv, npend, 4, 3, 0.99f, k_alloc, &cut) == 1);
    for (int i = 0; i < 4; i++) TEST_ASSERT(k_alloc[i] == 0);
    TEST_ASSERT(cut == 17);
    /* partial cut at the live threshold: demand 3 + 15 = 18 > 16, budget 13,
     * but only 5 + 5 + 2 rows survive above 0.2 (bank 2 at 0.5^k: 0.5, 0.25,
     * then 0.125) -- the cut fires with budget left, bank 2's other 3 rows
     * are the cut count, and the budget goes unspent. */
    l179_fill_surv(surv, npend, 0, 5, 0.95f);
    l179_fill_surv(surv, npend, 1, 5, 0.90f);
    l179_fill_surv(surv, npend, 2, 5, 0.50f);
    TEST_ASSERT(spec_alloc_rows(surv, npend, 4, 3, thr_live, k_alloc, &cut) == 1);
    TEST_ASSERT(k_alloc[0] == 5 && k_alloc[1] == 5 && k_alloc[2] == 2 && k_alloc[3] == 0);
    TEST_ASSERT(cut == 3);
    TEST_ASSERT(3 + 12 < B);
}

/* L179 branch 13 -- the per-quantum client-disconnect poll shared by the
 * three batched lanes (lane_should_abandon). Invariant: a slot is abandoned
 * iff its gen state is GEN_DECODE, its client fd reports a hang-up
 * (gen_client_disconnected), and -- when the lane requires it (plain, mixed)
 * -- batch_feed_valid is set; the spec lane passes require=false. A live
 * peer never abandons in any phase; GEN_PREFILL_MAIN / GEN_FINISH never do;
 * a NULL gen or a negative fd never does. */
static void test_l179_lane_abandon_needs_decode_and_hangup(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;
    gen_state g;
    memset(&g, 0, sizeof g);
    static const gen_phase phases[] = {GEN_DECODE, GEN_PREFILL_MAIN, GEN_FINISH};
    /* live peer: never, whatever the phase / flags */
    for (size_t pi = 0; pi < sizeof phases / sizeof phases[0]; pi++)
    for (int require = 0; require < 2; require++)
    for (int valid = 0; valid < 2; valid++) {
        g.phase = phases[pi];
        g.batch_feed_valid = valid != 0;
        TEST_ASSERT(!lane_should_abandon(&g, require != 0, sv[0]));
    }
    /* peer hangs up */
    close(sv[1]);
    for (size_t pi = 0; pi < sizeof phases / sizeof phases[0]; pi++)
    for (int require = 0; require < 2; require++)
    for (int valid = 0; valid < 2; valid++) {
        g.phase = phases[pi];
        g.batch_feed_valid = valid != 0;
        const bool want = phases[pi] == GEN_DECODE && (!require || valid);
        TEST_ASSERT(lane_should_abandon(&g, require != 0, sv[0]) == want);
    }
    /* the spec lane (require=false) abandons a dead decoder with no feed;
     * the plain/mixed lanes (require=true) do not */
    g.phase = GEN_DECODE;
    g.batch_feed_valid = false;
    TEST_ASSERT(lane_should_abandon(&g, false, sv[0]));
    TEST_ASSERT(!lane_should_abandon(&g, true, sv[0]));
    /* no gen state / no fd: never */
    TEST_ASSERT(!lane_should_abandon(NULL, false, sv[0]));
    TEST_ASSERT(!lane_should_abandon(&g, false, -1));
    close(sv[0]);

    /* L190 C3: a writer that has FAILED (EPIPE, stall, overflow, shutdown) is
     * a gone client too, socket state notwithstanding -- with a LIVE peer and
     * no fd at all, the same phase/feed rule applies. */
    int live[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, live) == 0);
    if (live[0] < 0 || live[1] < 0) return;
    memset(&g, 0, sizeof g);
    g.writer.failed = true;
    for (size_t pi = 0; pi < sizeof phases / sizeof phases[0]; pi++)
    for (int require = 0; require < 2; require++)
    for (int valid = 0; valid < 2; valid++) {
        g.phase = phases[pi];
        g.batch_feed_valid = valid != 0;
        const bool want = phases[pi] == GEN_DECODE && (!require || valid);
        TEST_ASSERT(lane_should_abandon(&g, require != 0, live[0]) == want);
        TEST_ASSERT(lane_should_abandon(&g, require != 0, -1) == want);
    }
    g.writer.failed = false;
    g.phase = GEN_DECODE;
    g.batch_feed_valid = true;
    TEST_ASSERT(!lane_should_abandon(&g, true, live[0]));
    close(live[0]);
    close(live[1]);
}

/* L184: the syntax table is the ONE authority.  Every consumer -- the final
 * parser, the stream projection's opener, the marker finders, the truncated
 * close-tag trimmer -- must recognise every row, not just the canonical one
 * (each used to carry its own 3-element copy; a row added to one and not
 * another would make "stream": true and the final parse disagree). */
static void test_l184_every_consumer_loops_the_syntax_table(void) {
    TEST_ASSERT(PULSAR_DSML_SYNTAXES == 3);
    for (size_t i = 0; i < PULSAR_DSML_SYNTAXES; i++) {
        const pulsar_dsml_syntax *syn = &pulsar_dsml_syntaxes[i];
        buf text = {0};
        buf_puts(&text, "ok\n\n");
        buf_puts(&text, syn->tool_calls_start);
        buf_puts(&text, "\n");
        buf_puts(&text, syn->invoke_start);
        buf_puts(&text, " name=\"bash\">\n");
        buf_puts(&text, syn->param_start);
        buf_puts(&text, " name=\"command\" string=\"true\">ls &amp;&amp; pwd");
        buf_puts(&text, syn->param_end);
        buf_puts(&text, "\n");
        buf_puts(&text, syn->param_start);
        buf_puts(&text, " name=\"timeout\" string=\"false\">10");
        buf_puts(&text, syn->param_end);
        buf_puts(&text, "\n");
        buf_puts(&text, syn->invoke_end);
        buf_puts(&text, "\n");
        buf_puts(&text, syn->tool_calls_end);

        /* the final parser */
        char *content = NULL, *reasoning = NULL;
        tool_calls calls = {0};
        TEST_ASSERT(parse_generated_message_ex(text.ptr, false, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.len == 1 && !strcmp(calls.v[0].name, "bash"));
        TEST_ASSERT(calls.len == 1 &&
                    !strcmp(calls.v[0].arguments, "{\"command\": \"ls && pwd\", \"timeout\": 10}"));
        TEST_ASSERT(content && !strcmp(content, "ok"));

        /* the marker finders */
        const char *start = find_any_tool_start(text.ptr);
        TEST_ASSERT(start == text.ptr + 4);
        TEST_ASSERT(find_any_tool_end(text.ptr) ==
                    text.ptr + text.len - strlen(syn->tool_calls_end));

        /* the stream projection opens on this row and binds to it */
        dsml_tool_stream ts;
        memset(&ts, 0, sizeof ts);
        TEST_ASSERT(dsml_tool_stream_init(&ts, text.ptr, text.len, 4));
        TEST_ASSERT(ts.syn == syn);
        TEST_ASSERT(ts.parse_pos == 4 + strlen(syn->tool_calls_start));
        dsml_tool_stream_free(&ts);

        /* a truncated closing tag of this row is trimmed as tag debris */
        buf cut = {0};
        buf_puts(&cut, "value");
        buf_append(&cut, syn->param_end, strlen(syn->param_end) - 1);
        TEST_ASSERT(trim_truncated_dsml_close_tail(cut.ptr, 0, cut.len) == 5);
        buf_free(&cut);

        free(content);
        free(reasoning);
        tool_calls_free(&calls);
        buf_free(&text);
    }
}

/* L184: the DSML tool-stream machine is ONE function driven through a
 * protocol's emitters.  Drive it through capturing emitters, byte by byte,
 * and pin what the protocol layer receives: one header per invocation, the
 * argument object as fragments that concatenate to canonical JSON, one
 * close per invocation, the index advancing past each. */
typedef struct {
    buf events;   /* "B<name>" per begin, "E" per end */
    buf args;     /* every args fragment, concatenated */
    int begins;
    int ends;
} capture_tool_ctx;

static bool capture_begin_invoke(void *vctx, dsml_tool_stream *ts, const char *name) {
    capture_tool_ctx *c = (capture_tool_ctx *)vctx;
    (void)ts;
    buf_printf(&c->events, "B%s;", name);
    c->begins++;
    return true;
}
static bool capture_args_fragment(void *vctx, dsml_tool_stream *ts, const char *text, size_t len) {
    capture_tool_ctx *c = (capture_tool_ctx *)vctx;
    (void)ts;
    buf_append(&c->args, text, len);
    return true;
}
static bool capture_end_invoke(void *vctx, dsml_tool_stream *ts) {
    capture_tool_ctx *c = (capture_tool_ctx *)vctx;
    (void)ts;
    buf_puts(&c->events, "E;");
    c->ends++;
    return true;
}
static const dsml_tool_stream_ops capture_tool_ops = {
    capture_begin_invoke, capture_args_fragment, capture_end_invoke,
};

static void test_l184_shared_tool_stream_drives_protocol_emitters(void) {
    const char *raw =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">echo \"a\" &lt;b" PULSAR_PARAM_END "\n"
        PULSAR_PARAM_START " name=\"timeout\" string=\"false\">10" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_INVOKE_START " name=\"read\">\n"
        PULSAR_PARAM_START " name=\"path\" string=\"true\">/tmp/x" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END;
    const size_t n = strlen(raw);

    capture_tool_ctx c;
    memset(&c, 0, sizeof c);
    dsml_tool_stream ts;
    memset(&ts, 0, sizeof ts);
    TEST_ASSERT(dsml_tool_stream_init(&ts, raw, n, 0));
    for (size_t len = 1; len <= n; len++) {
        TEST_ASSERT(dsml_tool_stream_update(&ts, &capture_tool_ops, &c, raw, len));
    }
    TEST_ASSERT(!ts.active);
    TEST_ASSERT(ts.state == DSML_TOOL_DONE);
    TEST_ASSERT(ts.index == 2);
    TEST_ASSERT(c.begins == 2 && c.ends == 2);
    TEST_ASSERT(c.events.ptr && !strcmp(c.events.ptr, "Bbash;E;Bread;E;"));
    /* entities undone, JSON-escaped, raw JSON values verbatim, one object per call */
    TEST_ASSERT(c.args.ptr &&
                !strcmp(c.args.ptr, "{\"command\":\"echo \\\"a\\\" <b\",\"timeout\":10}{\"path\":\"/tmp/x\"}"));
    dsml_tool_stream_free(&ts);
    buf_free(&c.events);
    buf_free(&c.args);

    /* truncated mid-value: finalize closes the string and the object and the
     * invocation, so the protocol's wire JSON stays well-formed */
    const char *cut =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">ls -la</";
    memset(&c, 0, sizeof c);
    memset(&ts, 0, sizeof ts);
    TEST_ASSERT(dsml_tool_stream_init(&ts, cut, strlen(cut), 0));
    TEST_ASSERT(dsml_tool_stream_update(&ts, &capture_tool_ops, &c, cut, strlen(cut)));
    TEST_ASSERT(ts.active && ts.state == DSML_TOOL_PARAM_VALUE);
    TEST_ASSERT(dsml_tool_stream_finalize(&ts, &capture_tool_ops, &c, cut, strlen(cut)));
    TEST_ASSERT(!ts.active && ts.state == DSML_TOOL_DONE);
    TEST_ASSERT(ts.index == 1 && c.ends == 1);
    TEST_ASSERT(c.args.ptr && !strcmp(c.args.ptr, "{\"command\":\"ls -la\"}"));
    dsml_tool_stream_free(&ts);
    buf_free(&c.events);
    buf_free(&c.args);
}

/* L184: one entity encode/decode pair (src/lib/pulsar_dsml).  The renderer's
 * attribute encoding round-trips through the parser's decoder; the decoder
 * also knows &apos;, which the model writes and the renderer never does. */
static void test_l184_dsml_entity_pair_round_trips(void) {
    const char *specials = "a&b <c> \"d\" 'e' &amp;literal";
    char *enc = pulsar_dsml_escape_attr(specials);
    TEST_ASSERT(!strcmp(enc, "a&amp;b &lt;c&gt; &quot;d&quot; 'e' &amp;amp;literal"));
    char *dec = pulsar_dsml_unescape(enc);
    TEST_ASSERT(!strcmp(dec, specials));
    free(dec);
    free(enc);
    dec = pulsar_dsml_unescape("&apos;x&apos; & &unknown; &lt;");
    TEST_ASSERT(!strcmp(dec, "'x' & &unknown; <"));
    free(dec);
    char *attr = pulsar_dsml_attr("<x name=\"a&amp;b\" string=\"true\">", "name");
    TEST_ASSERT(attr && !strcmp(attr, "a&b"));
    free(attr);
    TEST_ASSERT(pulsar_dsml_attr("<x name=\"unterminated>", "name") == NULL);
    TEST_ASSERT(pulsar_dsml_attr("<x string=\"true\">", "name") == NULL);
}

static chat_msg l185_msg(const char *role, const char *content, const char *reasoning) {
    chat_msg m = {0};
    m.role = xstrdup(role);
    m.content = content ? xstrdup(content) : NULL;
    m.reasoning = reasoning ? xstrdup(reasoning) : NULL;
    return m;
}

static void l185_add_call(chat_msg *m, const char *id, const char *name, const char *args) {
    tool_call tc = {0};
    tc.id = xstrdup(id);
    tc.name = xstrdup(name);
    tc.arguments = xstrdup(args);
    tool_calls_push(&m->calls, tc);
}

/* render(msgs[0..n)) with its trailing EOS removed, malloc'd */
static char *l185_render_prefix_sans_eos(const chat_msgs *msgs, int n, const char *schemas) {
    chat_msgs view = *msgs;   /* shallow: never freed */
    view.len = n;
    char *text = render_chat_prompt_text(&view, schemas, NULL, PULSAR_THINK_HIGH);
    size_t len = strlen(text), eos = strlen(PULSAR_RENDER_EOS);
    TEST_ASSERT(len >= eos && !strcmp(text + len - eos, PULSAR_RENDER_EOS));
    if (len >= eos) text[len - eos] = '\0';
    return text;
}

/* L185: how a chat turn renders is ONE function (append_chat_msg and its two
 * primitives).  A production-shaped transcript -- system, user/assistant
 * turns with reasoning, a tool call, its result, another turn -- goes through
 * every entry that produces turn bytes: the full replay (pinned), the live
 * tool tail on both protocol shapes, both checkpoint suffix builders, the
 * toolless visible key, the server-tool result suffix and the legacy
 * /v1/completions template.  Shared parts must be byte-equal. */
static void test_l185_every_renderer_produces_the_authority_bytes(void) {
    const char *schemas = "{\"name\":\"bash\"}";
    chat_msgs msgs = {0};
    chat_msgs_push(&msgs, l185_msg("system", "You are terse.", NULL));      /* 0 */
    chat_msgs_push(&msgs, l185_msg("user", "hi", NULL));                     /* 1 */
    chat_msgs_push(&msgs, l185_msg("assistant", "hello", "greet"));          /* 2 */
    chat_msgs_push(&msgs, l185_msg("user", "list /tmp", NULL));              /* 3 */
    chat_msg call = l185_msg("assistant", "", "need ls");                    /* 4 */
    l185_add_call(&call, "call_1", "bash", "{\"command\":\"ls /tmp\"}");
    chat_msgs_push(&msgs, call);
    chat_msg result = l185_msg("tool", "a.txt\nb.txt", NULL);                /* 5 */
    result.tool_call_id = xstrdup("call_1");
    chat_msgs_push(&msgs, result);
    chat_msgs_push(&msgs, l185_msg("assistant", "two files", "read it"));    /* 6 */
    chat_msgs_push(&msgs, l185_msg("user", "thanks", NULL));                 /* 7 */

    /* 1. the authority, pinned from the first history byte */
    char *full = render_chat_prompt_text(&msgs, schemas, NULL, PULSAR_THINK_HIGH);
    const char *hist = strstr(full, PULSAR_RENDER_USER "hi");
    TEST_ASSERT(hist != NULL);
    const char *want_hist =
        "<｜User｜>hi<｜Assistant｜><think>greet</think>hello<｜end▁of▁sentence｜>"
        "<｜User｜>list /tmp<｜Assistant｜><think>need ls</think>"
        "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls /tmp</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>"
        "<｜User｜><tool_result>a.txt\nb.txt</tool_result>"
        "<｜Assistant｜><think>read it</think>two files<｜end▁of▁sentence｜>"
        "<｜User｜>thanks<｜Assistant｜><think>";
    TEST_ASSERT(hist && !strcmp(hist, want_hist));

    /* 2. the live tool tail (Responses shape: tool-role result) == the full
     *    render of the same messages minus the already-live prefix */
    {
        chat_msgs cont = msgs;   /* shallow view through the tool result */
        cont.len = 6;
        char *full_cont = render_chat_prompt_text(&cont, schemas, NULL, PULSAR_THINK_HIGH);
        char *prefix = l185_render_prefix_sans_eos(&msgs, 5, schemas);
        char *tail = render_live_tool_tail(&cont, 5, true, PULSAR_THINK_HIGH);
        TEST_ASSERT(!strcmp(tail, "<｜end▁of▁sentence｜><｜User｜><tool_result>a.txt\nb.txt"
                                  "</tool_result><｜Assistant｜><think>"));
        buf glued = {0};
        buf_puts(&glued, prefix);
        buf_puts(&glued, tail);
        TEST_ASSERT(!strcmp(glued.ptr, full_cont));
        /* ...and the production entry hands out the same bytes */
        request r;
        request_init(&r, REQ_CHAT, 128);
        r.api = API_RESPONSES;
        r.think_mode = PULSAR_THINK_HIGH;
        r.has_tools = true;
        responses_prepare_live_continuation(&r, &cont);
        TEST_ASSERT(r.responses_live_suffix_text && !strcmp(r.responses_live_suffix_text, tail));
        request_free(&r);
        buf_free(&glued);
        free(tail);
        free(prefix);
        free(full_cont);
    }

    /* 3. the live tool tail, Anthropic shape (user-role result carrying the
     *    wrapper), including a trailing system message rendered in place */
    {
        chat_msgs anth = {0};
        for (int i = 0; i < 5; i++) {
            chat_msg m = l185_msg(msgs.v[i].role, msgs.v[i].content, msgs.v[i].reasoning);
            if (msgs.v[i].calls.len) l185_add_call(&m, "call_1", "bash", "{\"command\":\"ls /tmp\"}");
            chat_msgs_push(&anth, m);
        }
        chat_msg ur = l185_msg("user", "<tool_result>a.txt\nb.txt</tool_result>", NULL);
        chat_msg_add_tool_call_id(&ur, "call_1");
        chat_msgs_push(&anth, ur);
        chat_msgs_push(&anth, l185_msg("system", "Be brief.", NULL));
        char *full_anth = render_chat_prompt_text(&anth, schemas, NULL, PULSAR_THINK_HIGH);
        char *prefix = l185_render_prefix_sans_eos(&anth, 5, schemas);
        request r;
        request_init(&r, REQ_CHAT, 128);
        r.api = API_ANTHROPIC;
        r.think_mode = PULSAR_THINK_HIGH;
        r.has_tools = true;
        anthropic_prepare_live_continuation(&r, &anth);
        TEST_ASSERT(r.anthropic_live_suffix_text != NULL);
        buf glued = {0};
        buf_puts(&glued, prefix);
        buf_puts(&glued, r.anthropic_live_suffix_text ? r.anthropic_live_suffix_text : "");
        TEST_ASSERT(!strcmp(glued.ptr, full_anth));
        TEST_ASSERT(strstr(glued.ptr, "<｜User｜><system-reminder>\nBe brief.\n</system-reminder><｜Assistant｜><think>") != NULL);
        request_free(&r);
        buf_free(&glued);
        free(prefix);
        free(full_anth);
        chat_msgs_free(&anth);
    }

    /* 4. the tool checkpoint suffix: prompt_text (ends with the generation
     *    prefix) + suffix == the replay of the finished call turn */
    {
        chat_msgs pre = msgs;
        pre.len = 4;
        char *prompt_text = render_chat_prompt_text(&pre, schemas, NULL, PULSAR_THINK_HIGH);
        chat_msgs with_call = msgs;
        with_call.len = 5;
        char *replay = render_chat_prompt_text(&with_call, schemas, NULL, PULSAR_THINK_HIGH);
        request r;
        request_init(&r, REQ_CHAT, 128);
        r.think_mode = PULSAR_THINK_HIGH;
        r.has_tools = true;
        r.reasoning_summary_emit = true;
        char *suffix = build_tool_checkpoint_suffix(&r, "", "need ls", &msgs.v[4].calls);
        buf key = {0};
        buf_puts(&key, prompt_text);
        buf_puts(&key, suffix);
        assert_replay_is_key_plus_eos(key.ptr, replay);   /* L196: the sampled turn has no EOS */
        /* 5. the Responses visible suffix, with calls: the same bytes */
        char *visible = build_responses_visible_assistant_suffix(&r, "", "need ls", &msgs.v[4].calls);
        TEST_ASSERT(!strcmp(visible, suffix));
        free(visible);
        /* ...without calls it strips the reasoning: the replay of the same
         *    turn with EMPTY reasoning (tool context keeps the block) */
        chat_msgs pre2 = msgs;
        pre2.len = 2;
        char *prompt2 = render_chat_prompt_text(&pre2, schemas, NULL, PULSAR_THINK_HIGH);
        chat_msgs stripped = {0};
        chat_msgs_push(&stripped, l185_msg("system", "You are terse.", NULL));
        chat_msgs_push(&stripped, l185_msg("user", "hi", NULL));
        chat_msgs_push(&stripped, l185_msg("assistant", "hello", ""));
        char *replay2 = render_chat_prompt_text(&stripped, schemas, NULL, PULSAR_THINK_HIGH);
        visible = build_responses_visible_assistant_suffix(&r, "hello", "greet", NULL);
        buf key2 = {0};
        buf_puts(&key2, prompt2);
        buf_puts(&key2, visible);
        TEST_ASSERT(!strcmp(key2.ptr, replay2));
        buf_free(&key2);
        free(visible);
        free(replay2);
        free(prompt2);
        chat_msgs_free(&stripped);
        request_free(&r);
        buf_free(&key);
        free(suffix);
        free(replay);
        free(prompt_text);
    }

    /* 6. toolless: historical reasoning is stripped, and the toolless visible
     *    key is a prefix of the next turn's replay */
    {
        chat_msgs tl = {0};
        chat_msgs_push(&tl, l185_msg("user", "hi", NULL));
        char *prompt_text = render_chat_prompt_text(&tl, NULL, NULL, PULSAR_THINK_HIGH);
        request r;
        request_init(&r, REQ_CHAT, 128);
        r.think_mode = PULSAR_THINK_HIGH;
        r.prompt_text = xstrdup(prompt_text);
        char *visible = build_toolless_thinking_visible_text(&r, "hello");
        chat_msgs_push(&tl, l185_msg("assistant", "hello", "greet"));
        chat_msgs_push(&tl, l185_msg("user", "thanks", NULL));
        char *future = render_chat_prompt_text(&tl, NULL, NULL, PULSAR_THINK_HIGH);
        TEST_ASSERT(visible && !strncmp(future, visible, strlen(visible)));
        TEST_ASSERT(strstr(future, "<｜User｜>hi<｜Assistant｜></think>hello<｜end▁of▁sentence｜>"
                                   "<｜User｜>thanks<｜Assistant｜><think>") != NULL);
        /* an assistant turn AFTER the last user message replays its reasoning */
        tl.len = 2;
        char *prefill = render_chat_prompt_text(&tl, NULL, NULL, PULSAR_THINK_HIGH);
        TEST_ASSERT(strstr(prefill, "<｜User｜>hi<｜Assistant｜><think>greet</think>hello<｜end▁of▁sentence｜>") != NULL);
        tl.len = 3;
        free(prefill);
        free(future);
        free(visible);
        request_free(&r);
        free(prompt_text);
        chat_msgs_free(&tl);
    }

    /* 7. a server-executed tool's result suffix == the live tail of one tool message */
    {
        request r;
        request_init(&r, REQ_CHAT, 128);
        r.think_mode = PULSAR_THINK_HIGH;
        r.has_tools = true;
        thinking_state th;
        memset(&th, 0, sizeof th);
        th.inside = true;
        char *ws = build_web_search_result_suffix(&r, &th, "results </tool_result> x");
        chat_msgs one = {0};
        chat_msgs_push(&one, l185_msg("tool", "results </tool_result> x", NULL));
        char *tail = render_live_tool_tail(&one, 0, true, PULSAR_THINK_HIGH);
        TEST_ASSERT(!strncmp(ws, "</think>", 8) && !strcmp(ws + 8, tail));
        TEST_ASSERT(!strcmp(tail, "<｜end▁of▁sentence｜><｜User｜><tool_result>results &lt;/tool_result> x"
                                  "</tool_result><｜Assistant｜><think>"));
        free(tail);
        free(ws);
        chat_msgs_free(&one);
        request_free(&r);
    }

    /* 8. the legacy /v1/completions template, pinned and through the renderer */
    {
        char *legacy = render_completion_prompt_text("hi", PULSAR_THINK_HIGH);
        buf want = {0};
        buf_puts(&want, PULSAR_SERVER_RENDER_BOS);
        buf_puts(&want, pulsar_think_effort_prefix(PULSAR_THINK_HIGH));
        buf_puts(&want, "You are a helpful assistant<｜User｜>hi<｜Assistant｜><think>");
        TEST_ASSERT(!strcmp(legacy, want.ptr));
        buf_free(&want);
        free(legacy);
        legacy = render_completion_prompt_text("hi", PULSAR_THINK_NONE);
        TEST_ASSERT(!strcmp(legacy, PULSAR_SERVER_RENDER_BOS "You are a helpful assistant<｜User｜>hi<｜Assistant｜></think>"));
        free(legacy);
    }

    free(full);
    chat_msgs_free(&msgs);
}

/* L192 item 4 (upstream a169cffa): tool-history validation is linear -- a
 * call_id -> nearest-preceding-assistant map built while scanning forward
 * replaces a per-id backward rescan.  The semantics it must keep: a repeated
 * id resolves to the LATER declaration (so its reasoning state is read from
 * the right turn), and an id declared only AFTER the tool message is not a
 * prior at all. */
static void test_l192_tool_history_validation_is_nearest_preceding(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg a0 = l185_msg("assistant", "", "thought once");
    l185_add_call(&a0, "call_a", "bash", "{}");
    chat_msgs_push(&msgs, a0);                                   /* 0: declares call_a WITH reasoning */
    chat_msg t1 = l185_msg("tool", "out", NULL);
    t1.tool_call_id = xstrdup("call_a");
    chat_msgs_push(&msgs, t1);                                   /* 1 */
    chat_msg a2 = l185_msg("assistant", "", NULL);
    l185_add_call(&a2, "call_a", "bash", "{}");
    chat_msgs_push(&msgs, a2);                                   /* 2: re-declares call_a WITHOUT reasoning */
    chat_msg t3 = l185_msg("tool", "out2", NULL);
    t3.tool_call_id = xstrdup("call_a");
    chat_msgs_push(&msgs, t3);                                   /* 3: nearest preceding is 2 */
    chat_msg t4 = l185_msg("tool", "out3", NULL);
    t4.tool_call_id = xstrdup("call_b");
    chat_msgs_push(&msgs, t4);                                   /* 4: call_b is declared only later */
    chat_msg a5 = l185_msg("assistant", "", NULL);
    l185_add_call(&a5, "call_b", "bash", "{}");
    chat_msgs_push(&msgs, a5);                                   /* 5 */

    char err[200] = {0};
    bool live_state = true, live_reasoning = false;
    chat_msgs head = msgs;   /* shallow view */
    head.len = 2;
    TEST_ASSERT(s.responses_validate_tool_outputs(&head, PULSAR_THINK_HIGH, &live_state,
                                                &live_reasoning, err, sizeof err));
    TEST_ASSERT(!live_state && !live_reasoning);               /* prior 0 has reasoning */
    head.len = 4;
    live_reasoning = false;
    TEST_ASSERT(s.responses_validate_tool_outputs(&head, PULSAR_THINK_HIGH, &live_state,
                                                &live_reasoning, err, sizeof err));
    TEST_ASSERT(live_reasoning);                                /* prior of 3 is 2, reasoning-less */
    TEST_ASSERT(!s.responses_validate_tool_outputs(&msgs, PULSAR_THINK_HIGH, &live_state,
                                                 &live_reasoning, err, sizeof err));
    TEST_ASSERT(strstr(err, "call_b") != NULL);                 /* declared after the output */

    /* the Anthropic validator shares the map: tool results are user messages */
    chat_msgs anth = {0};
    chat_msg b0 = l185_msg("assistant", "", NULL);
    l185_add_call(&b0, "toolu_a", "Bash", "{}");
    chat_msgs_push(&anth, b0);
    chat_msg u1 = l185_msg("user", "<tool_result>x</tool_result>", NULL);
    chat_msg_add_tool_call_id(&u1, "toolu_a");
    chat_msgs_push(&anth, u1);
    chat_msg u2 = l185_msg("user", "<tool_result>y</tool_result>", NULL);
    chat_msg_add_tool_call_id(&u2, "toolu_b");
    chat_msgs_push(&anth, u2);
    chat_msg b3 = l185_msg("assistant", "", NULL);
    l185_add_call(&b3, "toolu_b", "Bash", "{}");
    chat_msgs_push(&anth, b3);
    err[0] = '\0';
    chat_msgs ahead = anth;
    ahead.len = 2;
    TEST_ASSERT(s.anthropic_validate_tool_results(&ahead, &live_state, err, sizeof err));
    TEST_ASSERT(!live_state);
    TEST_ASSERT(!s.anthropic_validate_tool_results(&anth, &live_state, err, sizeof err));
    TEST_ASSERT(strstr(err, "toolu_b") != NULL);

    chat_msgs_free(&anth);
    chat_msgs_free(&msgs);
    pthread_mutex_destroy(&s.tool_mu);
}

/* L190 C1: the MemAvailable-floor refusal is a per-request condition; its
 * warning prints once per period and carries the count the period swallowed,
 * instead of once per process. */
static void test_l190_mem_floor_warn_is_rate_limited(void) {
    warn_limiter w = {0};
    unsigned skipped = 99;
    TEST_ASSERT(warn_limiter_due(&w, 100.0, 10.0, &skipped));   /* first: prints */
    TEST_ASSERT(skipped == 0);
    TEST_ASSERT(!warn_limiter_due(&w, 101.0, 10.0, &skipped));  /* inside the period */
    TEST_ASSERT(!warn_limiter_due(&w, 109.9, 10.0, &skipped));
    TEST_ASSERT(skipped == 0);                                   /* untouched while suppressed */
    TEST_ASSERT(w.suppressed == 2);
    TEST_ASSERT(warn_limiter_due(&w, 110.0, 10.0, &skipped));   /* period elapsed: prints */
    TEST_ASSERT(skipped == 2);                                   /* ...and reports the two */
    TEST_ASSERT(w.suppressed == 0);
    TEST_ASSERT(w.last_sec == 110.0);
    TEST_ASSERT(!warn_limiter_due(&w, 115.0, 10.0, &skipped));
    TEST_ASSERT(warn_limiter_due(&w, 130.0, 10.0, &skipped));
    TEST_ASSERT(skipped == 1);
}

/* L179 branch 6 (i) -- fresh_make_room's LRU-superseded victim scan
 * (superseded_pick_core). Invariant: slot a is picked only if it is
 * eligible, unprotected, has history (hist_len > 0) and some OTHER slot k is
 * STRICTLY longer (frontier[k] > hist_len[a]) with a's whole history as its
 * prefix (common[a][k] >= hist_len[a]); among such slots the smallest
 * last_us wins (first index on a tie); a plain-LRU idle slot that nothing
 * supersedes is never picked over a superseded one; no supersession is -1. */
static void test_l179_superseded_pick_prefers_redundant_history(void) {
    enum { N = 4 };
    bool protect[N] = {false, false, false, false};
    bool eligible[N] = {true, true, true, true};
    int hist_len[N] = {100, 300, 50, 0};
    int frontier[N] = {100, 300, 50, 0};
    uint64_t last_us[N] = {10, 20, 5, 1};   /* slot 2 is the LRU with history */
    int rows[N][N];
    const int *common[N];
    for (int i = 0; i < N; i++) {
        common[i] = rows[i];
        for (int k = 0; k < N; k++) rows[i][k] = -1;
    }
    /* slot 1's history extends slot 0's whole 100 tokens; slot 2 shares only
     * 30 with either: the superseded slot 0 is picked over the LRU slot 2 */
    rows[0][1] = 100;
    rows[2][0] = 30;
    rows[2][1] = 30;
    TEST_ASSERT(superseded_pick_core(N, protect, eligible, hist_len, frontier, common, last_us) == 0);
    TEST_ASSERT(superseded_pick_core(N, NULL, eligible, hist_len, frontier, common, last_us) == 0);
    /* protected or ineligible: never picked, and nothing else qualifies */
    protect[0] = true;
    TEST_ASSERT(superseded_pick_core(N, protect, eligible, hist_len, frontier, common, last_us) == -1);
    protect[0] = false;
    eligible[0] = false;
    TEST_ASSERT(superseded_pick_core(N, protect, eligible, hist_len, frontier, common, last_us) == -1);
    eligible[0] = true;
    /* the superseder's own eligibility is irrelevant -- only its frontier */
    eligible[1] = false;
    TEST_ASSERT(superseded_pick_core(N, protect, eligible, hist_len, frontier, common, last_us) == 0);
    eligible[1] = true;
    /* two superseded slots: LRU wins (slot 2 at 5 us over slot 0 at 10 us) */
    rows[2][0] = 50;
    TEST_ASSERT(superseded_pick_core(N, protect, eligible, hist_len, frontier, common, last_us) == 2);
    rows[2][0] = 30;
    /* strictly longer: a superseder at EXACTLY a's length does not count */
    frontier[1] = 100;
    TEST_ASSERT(superseded_pick_core(N, protect, eligible, hist_len, frontier, common, last_us) == -1);
    frontier[1] = 300;
    /* a's whole history must be the prefix: 99 of 100 is not */
    rows[0][1] = 99;
    TEST_ASSERT(superseded_pick_core(N, protect, eligible, hist_len, frontier, common, last_us) == -1);
    rows[0][1] = 100;
    /* an empty bank is plain LRU's business even though common >= 0 holds */
    rows[3][1] = 0;
    TEST_ASSERT(superseded_pick_core(N, protect, eligible, hist_len, frontier, common, last_us) == 0);
    /* no supersession anywhere: -1 */
    rows[0][1] = -1;
    TEST_ASSERT(superseded_pick_core(N, protect, eligible, hist_len, frontier, common, last_us) == -1);
    /* a slot never supersedes itself (the diagonal is skipped) */
    rows[0][0] = 100;
    frontier[0] = 200;
    TEST_ASSERT(superseded_pick_core(N, protect, eligible, hist_len, frontier, common, last_us) == -1);
}

/* L179 branch 6 (ii) -- the eviction overlay's usability rule
 * (warm_match_usable, worker_protect_queued_warm_matches). Invariant: a
 * queued job's best match protects its bank iff best_common >=
 * warm_partial_min AND (best_common == frontier -- a full fork, no ring probe
 * -- OR the partial cut is ring-feasible, feasible_rc == PULSAR_FORK_OK). A
 * ring-scrolled or otherwise infeasible cut is dead warmth and stays
 * evictable; below the minimum nothing is protected. */
static void test_l179_warm_match_usable_rule(void) {
    const int min = 64;
    /* partial cut at / above the minimum, ring feasible: usable */
    TEST_ASSERT(warm_match_usable(min, min, 500, PULSAR_FORK_OK));
    TEST_ASSERT(warm_match_usable(200, min, 500, PULSAR_FORK_OK));
    /* below the minimum: not, even when feasible */
    TEST_ASSERT(!warm_match_usable(min - 1, min, 500, PULSAR_FORK_OK));
    TEST_ASSERT(!warm_match_usable(0, min, 500, PULSAR_FORK_OK));
    /* ring-scrolled (or any other refusal): dead warmth */
    TEST_ASSERT(!warm_match_usable(200, min, 500, PULSAR_FORK_RING_SCROLLED));
    TEST_ASSERT(!warm_match_usable(200, min, 500, PULSAR_FORK_EVICTED));
    TEST_ASSERT(!warm_match_usable(200, min, 500, PULSAR_FORK_SHALLOW));
    /* full fork at the frontier: usable regardless of the probe value */
    TEST_ASSERT(warm_match_usable(500, min, 500, PULSAR_FORK_OK));
    TEST_ASSERT(warm_match_usable(500, min, 500, PULSAR_FORK_RING_SCROLLED));
    /* ...but a full match below the minimum is still not */
    TEST_ASSERT(!warm_match_usable(min - 1, min, min - 1, PULSAR_FORK_OK));
}

/* L179 branch 7 (i) -- guard_pick_victim on a host-only server (sess NULL:
 * pulsar_session_bank_fork_pinned reads false, bank_touched_kv_bytes 0).
 * Invariant: bank 0 is never a victim; nothing in the live decode set is;
 * unprovisioned, spilled and bound slots are skipped; among the rest the
 * smallest last_serviced_us wins (touched-bytes tie-break, then first
 * index); no candidate is -1. */
static void test_l179_guard_victim_skips_pinned_live_spilled(void) {
    server s;
    memset(&s, 0, sizeof s);
    s.n_slots = 5;
    static const uint64_t us[5] = {1, 40, 30, 20, 10};   /* bank 0 the oldest */
    for (int i = 0; i < 5; i++) {
        s.slots[i].provisioned = true;
        s.slots[i].bank = (uint32_t)i;
        s.slots[i].last_serviced_us = us[i];
    }
    session_slot *dec[2] = {&s.slots[4], &s.slots[3]};
    /* bank 0 (LRU) is pinned, 3 and 4 are live: LRU of {1, 2} is 2 */
    TEST_ASSERT(s.guard_pick_victim(dec, 2) == 2);
    /* only 2 live: LRU of {1, 3, 4} is 4 */
    dec[0] = &s.slots[2];
    TEST_ASSERT(s.guard_pick_victim(dec, 1) == 4);
    /* a spilled bank is skipped: {1, 3} -> 3 */
    s.slots[4].spilled = true;
    TEST_ASSERT(s.guard_pick_victim(dec, 1) == 3);
    /* an unprovisioned one too: {1} -> 1 */
    s.slots[3].provisioned = false;
    TEST_ASSERT(s.guard_pick_victim(dec, 1) == 1);
    /* a bound one too: nothing left -> -1 */
    s.slots[1].active_job = (struct job *)&s;
    TEST_ASSERT(s.guard_pick_victim(dec, 1) == -1);
    /* with no decode set slot 2 is idle again and is the only candidate;
     * bank 0 (still the oldest) is still never picked */
    TEST_ASSERT(s.guard_pick_victim(dec, 0) == 2);
    s.slots[1].active_job = NULL;
    /* tie on last_serviced_us across 1, 2, 3 (touched is 0 for every bank
     * here): first index */
    s.slots[3].provisioned = true;
    s.slots[2].last_serviced_us = us[1];
    s.slots[3].last_serviced_us = us[1];
    TEST_ASSERT(s.guard_pick_victim(dec, 0) == 1);
    /* a one-slot pool has no victim */
    s.n_slots = 1;
    TEST_ASSERT(s.guard_pick_victim(dec, 0) == -1);
}

/* L179 branch 7 (ii) -- guard_maybe_evict's control law (guard_spill_plan).
 * Invariant: 0 spills when touched + delta fits the bound; otherwise the
 * MINIMUM number of LRU victims whose touched bytes bring the projection
 * back under the bound (finding 2: never the whole idle set); with the
 * victims exhausted and the breach still standing it returns n_victims --
 * every spill it can do -- and with no victim at all 0, both of which the
 * caller follows with back-pressure. A drop larger than the running total
 * saturates at zero. */
static void test_l179_guard_spill_plan_is_minimum(void) {
    const uint64_t GiB = 1024ull * 1024ull * 1024ull;
    const uint64_t drops[3] = {GiB, GiB, GiB};
    /* fits (exactly at the bound): no spill */
    TEST_ASSERT(guard_spill_plan(10 * GiB, GiB, 11 * GiB, drops, 3) == 0);
    /* breach by one byte, three victims available: exactly one spill */
    TEST_ASSERT(guard_spill_plan(10 * GiB + 1, GiB, 11 * GiB, drops, 3) == 1);
    /* breach that one victim cannot clear: two */
    TEST_ASSERT(guard_spill_plan(12 * GiB, GiB, 11 * GiB, drops, 3) == 2);
    /* breach with no victim: nothing to spill (caller back-pressures) */
    TEST_ASSERT(guard_spill_plan(12 * GiB, GiB, 11 * GiB, NULL, 0) == 0);
    /* victims exhausted while still breaching: all three (caller back-pressures) */
    TEST_ASSERT(guard_spill_plan(20 * GiB, GiB, 11 * GiB, drops, 3) == 3);
    /* a hollow victim (nothing resident) does not clear the breach by itself */
    const uint64_t hollow[2] = {0, GiB};
    TEST_ASSERT(guard_spill_plan(10 * GiB + 1, GiB, 11 * GiB, hollow, 2) == 2);
    /* a drop above the running total saturates instead of wrapping */
    const uint64_t huge[1] = {100 * GiB};
    TEST_ASSERT(guard_spill_plan(12 * GiB, GiB, 11 * GiB, huge, 1) == 1);
    /* growth alone can breach an empty pool: nothing resident to drop, so
     * every victim is "spilled" to no effect and the caller back-pressures */
    TEST_ASSERT(guard_spill_plan(0, 2 * GiB, GiB, drops, 3) == 3);
}

/* L179 branch 3 -- choose_slot_for_job's cross-wire divergent route
 * (divergent_route_decision). Invariant: no best, or best_common >=
 * frontier (a linear continuation), is NOT_DIVERGENT; a divergent match
 * (best_common < frontier) routes to a FRESH bank when one was provisioned,
 * else QUEUEs while any job is active (its finish frees a bank, and there is
 * a live reader to protect), else continues IN_PLACE (no bank will ever
 * free and nothing live can be corrupted -- the deadlock-avoidance
 * fallthrough). */
static void test_l179_divergent_route_four_ways(void) {
    /* linear continuation / past the frontier: not divergent, whatever else */
    TEST_ASSERT(divergent_route_decision(true, 500, 500, false, true) == ROUTE_NOT_DIVERGENT);
    TEST_ASSERT(divergent_route_decision(true, 500, 500, true, false) == ROUTE_NOT_DIVERGENT);
    TEST_ASSERT(divergent_route_decision(true, 600, 500, true, true) == ROUTE_NOT_DIVERGENT);
    /* no best: not divergent even with a would-be divergent geometry */
    TEST_ASSERT(divergent_route_decision(false, 0, 0, true, true) == ROUTE_NOT_DIVERGENT);
    TEST_ASSERT(divergent_route_decision(false, 5, 500, false, true) == ROUTE_NOT_DIVERGENT);
    /* divergent with a fresh bank: FRESH, active or not */
    TEST_ASSERT(divergent_route_decision(true, 7, 500, true, true) == ROUTE_FRESH);
    TEST_ASSERT(divergent_route_decision(true, 7, 500, true, false) == ROUTE_FRESH);
    /* divergent, pool full, a live job: QUEUE */
    TEST_ASSERT(divergent_route_decision(true, 7, 500, false, true) == ROUTE_QUEUE);
    TEST_ASSERT(divergent_route_decision(true, 499, 500, false, true) == ROUTE_QUEUE);
    /* divergent, pool full, nothing running: IN_PLACE */
    TEST_ASSERT(divergent_route_decision(true, 7, 500, false, false) == ROUTE_IN_PLACE);
    TEST_ASSERT(divergent_route_decision(true, 0, 1, false, false) == ROUTE_IN_PLACE);
    /* the caller's first probe (no fresh, no active) is a divergence test:
     * it must never read NOT_DIVERGENT for a divergent match */
    TEST_ASSERT(divergent_route_decision(true, 7, 500, false, false) != ROUTE_NOT_DIVERGENT);
}

/* L179 branch 5 -- the commit after choose_slot_for_job's warm-advance-in-place
 * (since 2026-09-06 the route for every divergent match on an idle bank, not
 * only at a full pool). The commit moves exactly two fields: committed_pos to
 * the engine's resume position and the continued-store watermark to 0 (the cut
 * moved the frontier backward; a stale watermark would refuse every continued
 * checkpoint). */
static void test_l179_warm_inplace_commit_moves_two_fields(void) {
    session_slot sl;
    memset(&sl, 0, sizeof sl);
    sl.provisioned = true;
    sl.bank = 3;
    sl.committed_pos = 165045;               /* the old frontier */
    sl.continued_last_store_tokens = 160000; /* a watermark ABOVE the cut */
    sl.state = SLOT_IDLE;
    sl.ctx_size = 262144;
    sl.est_cost_bytes = 7;
    sl.tokens_emitted = 11;
    sl.prefill_counted = 13;
    sl.last_serviced_us = 17;
    warm_inplace_commit(&sl, 164800);        /* the R-aligned cut of common 164812 */
    TEST_ASSERT(sl.committed_pos == 164800);
    TEST_ASSERT(sl.continued_last_store_tokens == 0);
    /* nothing else on the slot moved */
    TEST_ASSERT(sl.provisioned && sl.bank == 3 && sl.state == SLOT_IDLE);
    TEST_ASSERT(sl.ctx_size == 262144 && sl.est_cost_bytes == 7 && sl.tokens_emitted == 11);
    TEST_ASSERT(sl.prefill_counted == 13 && sl.last_serviced_us == 17);
}

/* L179 branch 11 -- worker_evict_one's slot reset (evict_reset_slot_fields).
 * Invariant: the evicted slot is a reusable hole -- unprovisioned,
 * SLOT_EVICTED, no gen, no job, ctx 0, ledger cost 0, no scheduler
 * bookkeeping (tokens_emitted, prefill_counted, last_serviced_us), no
 * continued-store watermark -- and the return value is the ctx it was
 * admitted for (the log line's). The bank id and `spilled` are NOT the
 * reset's to touch: slot i -> bank i is fixed, and the caller reconciles the
 * spill file / physical against `spilled` right after. */
static void test_l179_evict_reset_leaves_a_reusable_hole(void) {
    session_slot sl;
    memset(&sl, 0, sizeof sl);
    job fake_job;
    gen_state fake_gen;
    memset(&fake_job, 0, sizeof fake_job);
    memset(&fake_gen, 0, sizeof fake_gen);
    sl.provisioned = true;
    sl.bank = 5;
    sl.committed_pos = 4096;
    sl.active_job = &fake_job;
    sl.gen = &fake_gen;
    sl.state = SLOT_DECODING;
    sl.ctx_size = 131072;
    sl.est_cost_bytes = 9ull << 30;
    sl.tokens_emitted = 777;
    sl.prefill_counted = 4000;
    sl.last_serviced_us = 123456789ull;
    sl.continued_last_store_tokens = 3072;
    sl.spilled = true;
    TEST_ASSERT(evict_reset_slot_fields(&sl) == 131072);
    TEST_ASSERT(!sl.provisioned);
    TEST_ASSERT(sl.gen == NULL);
    TEST_ASSERT(sl.active_job == NULL);
    TEST_ASSERT(sl.state == SLOT_EVICTED);
    TEST_ASSERT(sl.ctx_size == 0);
    TEST_ASSERT(sl.est_cost_bytes == 0);
    TEST_ASSERT(sl.tokens_emitted == 0);
    TEST_ASSERT(sl.prefill_counted == 0);
    TEST_ASSERT(sl.last_serviced_us == 0);
    TEST_ASSERT(sl.continued_last_store_tokens == 0);
    /* the caller's facts survive the reset */
    TEST_ASSERT(sl.bank == 5);
    TEST_ASSERT(sl.spilled);
    /* an already-empty slot resets to the same hole and reports ctx 0 */
    session_slot empty;
    memset(&empty, 0, sizeof empty);
    empty.bank = 2;
    TEST_ASSERT(evict_reset_slot_fields(&empty) == 0);
    TEST_ASSERT(!empty.provisioned && empty.state == SLOT_EVICTED && empty.bank == 2);
}

/* L179 branch 10 -- the fused mixed quantum's head cap (mixed_head_cap).
 * Invariant: the cap is m (decode runs only) iff the step folds prefill rows
 * (kthis > 0) that do NOT reach len (pos_now + kthis < len) and there are
 * decode banks to head (m > 0); the FINAL sub-chunk, a pure-decode step and a
 * prefill-only step all pass 0 = every run, so the prefill head that IS
 * consumed is never dropped. */
static void test_l179_mixed_head_cap_drops_only_intermediate_prefill_head(void) {
    /* 3 decoders + a 16-row sub-chunk from 100 of a 1000-token prompt: cap 3 */
    TEST_ASSERT(mixed_head_cap(16, 3, 100, 1000) == 3u);
    /* the same shape one row short of the end: still intermediate */
    TEST_ASSERT(mixed_head_cap(16, 3, 983, 1000) == 3u);
    /* the FINAL sub-chunk lands exactly on len: every run */
    TEST_ASSERT(mixed_head_cap(16, 3, 984, 1000) == 0u);
    /* a short final tail (kthis clipped to len - pos_now): every run */
    TEST_ASSERT(mixed_head_cap(7, 3, 993, 1000) == 0u);
    /* pure-decode step (the prefill gave up or is done): every run */
    TEST_ASSERT(mixed_head_cap(0, 3, 100, 1000) == 0u);
    /* prefill-only step (no decoder had a valid feed): every run */
    TEST_ASSERT(mixed_head_cap(16, 0, 100, 1000) == 0u);
    /* one decoder, one prefill row, deep inside the prompt: cap 1 */
    TEST_ASSERT(mixed_head_cap(1, 1, 1, 1000) == 1u);
}

/* L179 branch 10 -- the fused step's give-up verdict (mixed_prefill_giveup).
 * Invariant: only the engine's RECOVERABLE reject (rc == 1) on a step that
 * folded prefill rows (kthis > 0) gives the prefill up; with decode banks in
 * the step (m > 0) they RETRY decode-only, with none the quantum STOPS. A
 * clean step, a hard failure (rc < 0 or any other nonzero), and a recoverable
 * reject on a pure-decode step (nothing to charge the prefill with) all
 * PROCEED to the caller's normal rc handling. */
static void test_l179_mixed_giveup_only_on_recoverable_prefill_reject(void) {
    TEST_ASSERT(mixed_prefill_giveup(0, 16, 3) == MIXED_PROCEED);
    TEST_ASSERT(mixed_prefill_giveup(0, 0, 3) == MIXED_PROCEED);
    TEST_ASSERT(mixed_prefill_giveup(-1, 16, 3) == MIXED_PROCEED);
    TEST_ASSERT(mixed_prefill_giveup(2, 16, 3) == MIXED_PROCEED);
    /* a recoverable reject with no prefill rows in the step is the decoders' */
    TEST_ASSERT(mixed_prefill_giveup(1, 0, 3) == MIXED_PROCEED);
    /* the prefill is charged: decoders retry alone */
    TEST_ASSERT(mixed_prefill_giveup(1, 16, 3) == MIXED_GIVEUP_RETRY_DECODE);
    TEST_ASSERT(mixed_prefill_giveup(1, 1, 1) == MIXED_GIVEUP_RETRY_DECODE);
    /* the prefill was alone in the step: nothing to retry, stop */
    TEST_ASSERT(mixed_prefill_giveup(1, 16, 0) == MIXED_GIVEUP_STOP);
}



static void pulsar_server_unit_tests_run(void) {
    test_logprob_token_json_sanitizes_ill_formed_utf8();
    test_random_prefixed_id_format();
    test_kv_disk_default_dir_resolution();
    test_kv_disk_flag_matrix();
    test_kv_cache_open_unusable_dir_disables();
    test_kv_admission_budget_math();
    test_mem_floor_admits_warmed_box_shape();
    test_session_eviction_ledger_math();
    test_session_eviction_victim_selection();
    test_slot_route_trivial_match_decision();
    test_thinking_binding_routes_visible_continuation();
    test_slot_writer_defers_and_preserves_order();
    test_slot_writer_stall_times_out();
    test_unterminated_think_stays_off_content();
    test_request_defaults_use_min_p_filtering();
    test_think_sampling_respects_explicit_params();
    test_decode_sampling_tool_payload_forcing();
    test_web_search_tool_recognition();
    test_web_search_result_replay_rebuild();
    test_web_search_replay_message_split();
    test_reasoning_effort_mapping();
    test_api_thinking_controls_parse();
    test_render_think_max_prompt_prefix();
    test_render_think_effort_prefixes();
    test_inline_system_message_placement();
    test_appended_system_message_keeps_prefix();
    test_render_non_thinking_prompt_closes_think();
    test_render_drops_old_reasoning_without_tools();
    test_render_preserves_reasoning_with_tools();
    test_render_chat_prompt_text_renders_tools_before_system();
    test_tool_schema_order_from_anthropic_schema();
    test_tool_schema_order_from_openai_tools();
    test_openai_tool_schema_json_spelling_is_canonical();
    test_anthropic_tool_schema_json_spelling_is_canonical();
    test_tool_schema_order_from_responses_tool_search();
    test_responses_function_named_tool_search_stays_function_call();
    test_responses_namespace_tool_schemas_restore_wire_namespace();
    test_responses_input_tool_search_output_loads_tools();
    test_responses_input_tool_search_output_rejects_bad_tools();
    test_responses_input_function_call_namespace_round_trips_to_dsml();
    test_responses_output_sends_tool_search_call_item();
    test_dsml_tool_args_preserve_call_order();
    test_openai_tool_args_preserve_call_order();
    test_anthropic_thinking_and_tool_args_preserve_call_order();
    test_context_length_error_uses_protocol_standard_shape();
    test_error_envelope_shape_per_protocol();
    test_logprob_stream_ready_watermark();
    test_anthropic_live_stream_sends_incremental_blocks();
    test_anthropic_usage_reports_cache_details();
    test_anthropic_tool_stream_sends_live_tool_use();
    test_openai_tool_stream_sends_incremental_text();
    test_openai_tool_stream_truncated_call_closes_args();
    test_repair_dsml_trims_partial_closing_tag();
    test_openai_stream_usage_reports_cache_details();
    test_responses_usage_reports_cache_details();
    test_openai_chat_stream_splits_reasoning_without_tools();
    test_openai_tool_stream_sends_partial_arguments();
    test_openai_tool_stream_waits_for_incomplete_tool_tags();
    test_openai_stream_keeps_text_when_tool_straddles_think_close();
    test_stream_heartbeat_only_fires_when_silent();
    test_stream_heartbeat_openai_uses_sse_comment();
    test_openai_tool_stream_sends_partial_raw_arguments();
    test_openai_tool_stream_holds_partial_dsml_entities();
    test_openai_tool_stream_holds_partial_utf8_arguments();
    test_openai_tool_stream_handles_multiple_calls();
    test_streaming_holds_partial_utf8();
    test_checkpoint_key_ends_where_sampled_tokens_end();
    test_parse_short_dsml_and_canonical_suffix();
    test_dsml_parser_recovers_loose_nested_parameters();
    test_dsml_repair_produces_parseable_calls();
    test_tool_parse_failure_returns_recoverable_finish();
    test_invalid_dsml_tool_error_suffix_includes_system_prompt();
    test_thinking_dsml_is_not_executable_before_think_close();
    test_thinking_dsml_after_think_close_is_executable();
    test_tool_checkpoint_suffix_is_future_prompt_canonical();
    test_tool_checkpoint_minifies_json_parameters();
    test_tool_memory_replays_sampled_dsml();
    test_anthropic_tool_memory_replays_sampled_dsml();
    test_anthropic_live_tail_renders_tool_results_only();
    test_anthropic_tool_result_id_validation();
    test_anthropic_full_replay_allows_unknown_live_id();
    test_anthropic_tool_use_parses_before_role();
    test_tool_checkpoint_canonicalization_gate_exact_replay();
    test_responses_live_tail_renders_tool_outputs_only();
    test_responses_tool_output_id_validation();
    test_responses_stateless_tool_replay_requires_reasoning();
    test_responses_visible_suffix_matches_client_replay();
    test_dsml_decode_state_separates_structure_and_payload();
    test_tool_memory_max_ids_prunes_oldest();
    test_kv_tool_map_filters_by_dsml_text();
    test_kv_tool_map_restores_before_prompt_render();
    test_thinking_checkpoint_canonical_matches_future_prompt();
    test_thinking_canonical_empty_content();
    test_thinking_canonical_multi_turn();
    test_thinking_canonical_with_tools_preserves_reasoning();
    test_thinking_canonical_non_thinking_mode_noop();
    test_tool_separator_whitespace_is_not_content();
    test_dsml_prompt_escapes_tool_supplied_text();
    test_stop_list_parses_all_sequences();
    test_stop_list_streaming_holds_and_trims_stop_text();
    test_json_skip_has_nesting_limit();
    test_json_value_helpers_null_out_on_failure();
    test_parse_sampling_key_contract();
    test_json_parser_handles_tool_heavy_requests();
    test_json_string_handles_surrogates();
    test_model_metadata_clamps_completion_to_context();
    test_client_socket_nonblocking_flag();
    test_thinking_state_tracks_prompt_and_generated_tags();
    test_thinking_checkpoint_remember_gate();
    test_tool_marker_state_ignores_orphan_end();
    test_canonical_rewrite_rebuilds_when_live_tail_changes();
    test_kv_cache_store_len_uses_configured_boundary();
    test_kv_cache_chat_anchor_uses_last_user_before_assistant();
    test_kv_cache_chat_anchor_ignores_multiturn_tail();
    test_kv_cache_sys_prefix_cut_clears_preamble_jitter();
    test_kv_cache_continued_uses_aligned_frontiers();
    test_kv_cache_cold_store_suppresses_duplicate_continued_boundary();
    test_kv_cache_file_size_must_fit_budget();
    test_sha1_bytes_hex_matches_known_vector();
    test_kv_cache_lookup_uses_longest_text_prefix();
    test_kv_cache_lookup_rejects_wrong_model();
    test_kv_cache_lookup_rejects_stale_payload_abi();
    test_kv_cache_eviction_values_fresh_snapshots();
    test_kv_cache_eviction_prefers_anchor_reason();
    test_kv_cache_eviction_prefers_sys_prefix_over_cold();
    test_kv_cache_eviction_makes_room_before_store();
    test_kv_cache_eviction_ignores_oversize_incoming();
    test_kv_cache_eviction_prefers_superseded_continued_prefix();
    test_kv_cache_eviction_keeps_smaller_context_prefix();
    test_kv_cache_eviction_score_decays_stale_hits();
    test_kv_cache_eviction_score_demotes_superseded_continued();
    test_kv_cache_eviction_decayed_hits_tie_break_by_age();
    test_kv_cache_eviction_keeps_aligned_continued_frontiers();
    test_l179_tool_admission_is_bound_decode_only();
    test_l179_deep_guard_blocks_two_deep_decoders();
    test_l179_bank_floor_exempts_first_bank();
    test_l179_park_live_bank_only_when_not_in_quantum();
    test_l179_lane_select_spec_needs_every_decoder();
    test_l179_spec_alloc_rows_isolation_and_ranked_overflow();
    test_l179_lane_abandon_needs_decode_and_hangup();
    test_l190_mem_floor_warn_is_rate_limited();
    test_l184_every_consumer_loops_the_syntax_table();
    test_l184_shared_tool_stream_drives_protocol_emitters();
    test_l184_dsml_entity_pair_round_trips();
    test_l185_every_renderer_produces_the_authority_bytes();
    test_l192_tool_history_validation_is_nearest_preceding();
    test_l179_superseded_pick_prefers_redundant_history();
    test_l179_warm_match_usable_rule();
    test_l179_guard_victim_skips_pinned_live_spilled();
    test_l179_guard_spill_plan_is_minimum();
    test_l179_divergent_route_four_ways();
    test_l179_warm_inplace_commit_moves_two_fields();
    test_l179_evict_reset_leaves_a_reusable_hole();
    test_l179_mixed_head_cap_drops_only_intermediate_prefill_head();
    test_l179_mixed_giveup_only_on_recoverable_prefill_reject();
}



#ifndef PULSAR_SERVER_TEST_NO_MAIN

int main(void) {
    pulsar_server_unit_tests_run();
    if (test_failures) {
        fprintf(stderr, "pulsar-server tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("pulsar-server tests: ok");
    return 0;
}


#endif


#endif /* PULSAR_SERVER_TEST */

/* L267: what the DeepSeek stream projections send, byte for byte -- the check that moving the
 * <think>/DSML walk out of the three protocol streamers (OpenAI chat, Anthropic Messages, Responses)
 * into one producer changes no event.  Host-only: each case is a raw DeepSeek generation fed to the
 * protocol's live projection piece by piece (1, 3, 7 bytes, or whole), then finished the way the server
 * finishes it (the final DSML parse, the protocol's finish).  Ids and timestamps are normalised.
 *
 *   ./tests/sse_golden > out.txt; diff tests/sse-golden/golden.txt out.txt
 *
 * The golden was recorded from the streamers before L267 moved the walk. */
#define PULSAR_SERVER_TEST
#define PULSAR_SERVER_TEST_NO_MAIN
#include "../src/server/util.cpp"
#include "../src/server/request.cpp"
#include "../src/server/prompt_render.cpp"
#include "../src/server/api_parse.cpp"
#include "../src/server/chat_family.cpp"
#include "../src/server/genmsg.cpp"
#include "../src/server/deepseek_stream.cpp"
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

#include <fcntl.h>
#include <regex>
#include <string>
#include <sys/socket.h>
#include <vector>

#define CALL_BASH(cmd) \
    PULSAR_TOOL_CALLS_START "\n" PULSAR_INVOKE_START " name=\"bash\">\n" PULSAR_PARAM_START \
    " name=\"command\" string=\"true\">" cmd PULSAR_PARAM_END "\n" PULSAR_INVOKE_END "\n"
#define CALLS_END PULSAR_TOOL_CALLS_END

struct gcase { const char *name; bool think; bool tools; const char *raw; const char *finish; };

static const gcase CASES[] = {
    {"think-answer", true, false, "Let me compute 17*23.</think>The answer is 391.", "stop"},
    {"think-prefix-repeated", true, false, "<think>hmm</think>ok", "stop"},
    {"think-unclosed", true, false, "thinking on and on, never closing", "length"},
    {"think-empty", true, false, "</think>Direct.", "stop"},
    {"no-think", false, false, "Plain answer with <angle> and </think> text.", "stop"},
    {"utf8", true, false, "naïve 日本語</think>🌞 ok — done", "stop"},
    {"partial-lt", true, true, "a</think>x < y and <b>bold</b> and <｜not a marker", "stop"},
    {"tool-after-think", true, true,
     "plan the call</think>\n\n" CALL_BASH("ls -l /var/log") CALLS_END, "stop"},
    {"text-then-tool", true, true,
     "plan</think>Let me look.\n\n" CALL_BASH("pwd") CALLS_END, "stop"},
    {"two-invokes", true, true,
     "x</think>" PULSAR_TOOL_CALLS_START "\n" PULSAR_INVOKE_START " name=\"bash\">\n" PULSAR_PARAM_START
     " name=\"command\" string=\"true\">a" PULSAR_PARAM_END "\n" PULSAR_INVOKE_END "\n" PULSAR_INVOKE_START
     " name=\"bash\">\n" PULSAR_PARAM_START " name=\"command\" string=\"true\">b &amp; c" PULSAR_PARAM_END "\n"
     PULSAR_PARAM_START " name=\"description\" string=\"true\">two" PULSAR_PARAM_END "\n" PULSAR_INVOKE_END
     "\n" CALLS_END, "stop"},
    {"tool-inside-think", true, true, "plan " CALL_BASH("date") CALLS_END, "stop"},
    {"tool-straddles-close", true, true,
     "consider " PULSAR_TOOL_CALLS_START "</think>Answer." CALLS_END, "stop"},
    {"second-reasoning", true, true, "first</think>second pass</think>final answer", "stop"},
    {"second-reasoning-then-tool", true, true, "a</think>b</think>\n" CALL_BASH("id") CALLS_END, "stop"},
    {"tool-unclosed", true, true, "go</think>" CALL_BASH("sleep 1"), "length"},
    {"no-think-tool", false, true, "Sure.\n\n" CALL_BASH("uname -a") CALLS_END, "stop"},
    {"tool-json-param", true, true,
     "j</think>" PULSAR_TOOL_CALLS_START "\n" PULSAR_INVOKE_START " name=\"bash\">\n" PULSAR_PARAM_START
     " name=\"command\" string=\"false\">{\"k\": [1, 2]}" PULSAR_PARAM_END "\n" PULSAR_INVOKE_END "\n" CALLS_END,
     "stop"},
};

enum proto { P_OPENAI, P_ANTHROPIC, P_RESPONSES, P_RESPONSES_SUMMARY };
static const char *const PROTO_NAME[] = {"openai", "anthropic", "responses", "responses+summary"};

struct streams {
    openai_stream oa;
    anthropic_stream an;
    responses_stream rs;
    chat_sink k;               /* L267: the protocol's sink ... */
    deepseek_stream_walk w;    /* ... and DeepSeek's walk into it */
};

/* The one place the harness touches a protocol's projection: the server's per-piece update. */
static bool golden_update(proto, int, request *, streams *st, const char *raw, size_t len, bool final) {
    return deepseek_stream_update(&st->w, &st->k, raw, len, final);
}

/* ... and its finish, after the server's final parse: the walk's flush, then the protocol's finish */
static bool golden_finish(proto p, int fd, request *r, streams *st, const char *raw, size_t len, tool_calls *calls,
                          const char *finish) {
    if (!deepseek_stream_update(&st->w, &st->k, raw, len, true)) return false;
    switch (p) {
    case P_OPENAI: return openai_sse_finish(&st->k, calls, finish, 10, 20);
    case P_ANTHROPIC: return anthropic_sse_finish(&st->k, calls, finish, NULL, 20);
    default: return responses_sse_finish(fd, r, &st->rs, NULL, 0, calls, finish, 10, 20, 1700000000L);
    }
}

static void drain(int fd, std::string *out) {
    char tmp[4096];
    ssize_t n;
    while ((n = read(fd, tmp, sizeof(tmp))) > 0) out->append(tmp, (size_t)n);
}

static std::string normalise(std::string s) {
    static const std::regex ids("(call|toolu|resp|rs|msg|fc|ws|ts|item)_[A-Za-z0-9_]{6,}");
    static const std::regex created("\"(created|created_at)\":[0-9]+");
    s = std::regex_replace(s, ids, "$1_ID");
    s = std::regex_replace(s, created, "\"$1\":T");
    return s;
}

static std::string run(const gcase &c, proto p, size_t piece) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return "(socketpair failed)\n";
    fcntl(sv[1], F_SETFL, O_NONBLOCK);
    int big = 1 << 22;
    setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &big, sizeof(big));
    request r;
    request_init(&r, REQ_CHAT, 256);
    free(r.model);
    r.model = xstrdup("m");
    r.stream = true;
    r.api = p == P_OPENAI ? API_OPENAI : p == P_ANTHROPIC ? API_ANTHROPIC : API_RESPONSES;
    r.think_mode = c.think ? PULSAR_THINK_HIGH : PULSAR_THINK_NONE;
    r.has_tools = c.tools;
    r.reasoning_summary_emit = p == P_RESPONSES_SUMMARY;
    if (c.tools)
        tool_schema_orders_add_json(&r.tool_orders,
            "{\"name\":\"bash\",\"input_schema\":{\"type\":\"object\",\"properties\":{"
            "\"command\":{\"type\":\"string\"},\"description\":{\"type\":\"string\"}}}}");
    streams st;
    memset(&st, 0, sizeof(st));
    std::string out;
    bool ok = true;
    if (p == P_OPENAI) {
        openai_stream_start(&r, &st.oa);
        openai_sink_init(&st.k, sv[0], NULL, &r, "chatcmpl-X", &st.oa);
    } else if (p == P_ANTHROPIC) {
        ok = anthropic_sse_start_live(sv[0], &r, "msg_X", 10, &st.an);
        anthropic_sink_init(&st.k, sv[0], NULL, &r, "msg_X", &st.an);
    } else {
        responses_stream_init(&r, &st.rs);
        st.rs.active = true;
        ok = responses_sse_created(sv[0], &r, &st.rs, 1700000000L);
        responses_sink_init(&st.k, sv[0], NULL, &r, "resp_X", &st.rs);
    }
    deepseek_stream_walk_init(&st.w, &r);
    const size_t n = strlen(c.raw);
    for (size_t at = 0; ok && at < n;) {
        at = piece ? std::min(n, at + piece) : n;
        ok = golden_update(p, sv[0], &r, &st, c.raw, at, false);
        drain(sv[1], &out);
    }
    const char *finish = c.finish;
    char err[256] = "", *content = NULL, *reasoning = NULL;
    tool_calls calls = {0};
    bool recovered = false;
    const bool saw_tool = c.tools && find_any_tool_start(c.raw) != NULL;
    parse_generated_message_for_response(c.raw, c.tools, saw_tool, c.think, &finish, err, sizeof(err), &content,
                                         &reasoning, &calls, &recovered);
    if (calls.len) finish = "tool_calls";
    if (ok) ok = golden_finish(p, sv[0], &r, &st, c.raw, n, &calls, finish);
    drain(sv[1], &out);
    if (!ok) out += "(a write failed)\n";
    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    deepseek_stream_walk_free(&st.w);
    responses_stream_free(&st.rs);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
    return normalise(out);
}

int main(void) {
    int n = 0;
    for (const gcase &c : CASES) {
        for (int p = P_OPENAI; p <= P_RESPONSES_SUMMARY; p++) {
            for (size_t piece : {(size_t)1, (size_t)3, (size_t)7, (size_t)0}) {
                printf("== %s [%s] piece %zu\n", c.name, PROTO_NAME[p], piece);
                fputs(run(c, (proto)p, piece).c_str(), stdout);
                n++;
            }
        }
    }
    fprintf(stderr, "sse_golden: %d runs\n", n);
    return 0;
}

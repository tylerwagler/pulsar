/* L267: what the DeepSeek stream projections send, byte for byte -- the check that moving the
 * <think>/DSML walk out of the three protocol streamers (OpenAI chat, Anthropic Messages, Responses)
 * into one producer changes no event.  Host-only: each case is a raw DeepSeek generation fed to the
 * protocol's live projection piece by piece (1, 3, 7 bytes, or whole), then finished the way the server
 * finishes it (the final DSML parse, the protocol's finish).  Ids and timestamps are normalised.
 *
 *   ./tests/sse_golden > out.txt; diff tests/sse-golden/golden.txt out.txt
 *
 * The golden was recorded from the streamers before L267 moved the walk.
 *
 * L278 (contract 6): a QWEN leg after it -- raw Qwen generations driven through the server's own sequence for any
 * family (gen_state + the request's family output parser: create, feed per piece, finish, the final feed, then the
 * protocol's finish with the turn's calls; server_jobs.cpp gen_finish), on a request the Qwen renderer prepared
 * (its tools JSON types the arguments).
 *
 * L284: the DeepSeek leg runs that same sequence (it used to emulate the finish with its own final parse), so both
 * families' finishes -- the broken-call rule, the Responses tail -- are what the goldens pin.
 *
 * L294 E6: an INKLING leg after the Qwen one, the same sequence on a request Inkling's renderer prepared (the kit in
 * Inkling's configuration: typed content blocks, the shared JSON call body).  Appended: the earlier legs' bytes are
 * unchanged.
 *
 * L417: a USAGE leg after the Inkling one -- the final usage object with its reasoning tokens, on the two OpenAI
 * surfaces (chat completions with stream_options.include_usage, Responses).  Each fed piece stands for one generated
 * token and goes through the server's own count (reasoning_usage_note after the tracker, as gen_emit_token), and the
 * completion tokens reported are the pieces fed, so reasoning_tokens is a subset of them as on the server.  It adds a
 * Gemma 4 request (the model opens its thought channel itself, twice in one case).  Appended: the earlier legs never
 * count (their usage keeps reasoning_tokens 0, and the chat stream sends no usage), so their bytes are unchanged. */
#define PULSAR_SERVER_TEST
#define PULSAR_SERVER_TEST_NO_MAIN
#include "../src/server/util.cpp"
#include "../src/server/request.cpp"
#include "../src/server/prompt_render.cpp"
#include "../src/server/api_parse.cpp"
#include "../src/server/chat_family.cpp"
#include "../src/server/context_edit.cpp"
#include "family_servers.inc"   /* every module's server table (generated: Makefile FAMILY_SERVERS_INC) */
#include "../src/server/constrain.cpp"
#include "../src/server/genmsg.cpp"
#include "../src/server/deepseek_dsml.cpp"
#include "../src/server/deepseek_stream.cpp"
#include "../src/server/parser_deepseek.cpp"
#include "../src/server/parser_kit.cpp"
#include "../src/server/openai_stream.cpp"
#include "../src/server/responses_stream.cpp"
#include "../src/server/anthropic_stream.cpp"
#include "../src/server/tool_memory.cpp"
#include "../src/server/kv_cache.cpp"
#include "../src/server/trace.cpp"
#include "../src/server/generate.cpp"
#include "../src/server/server_jobs.cpp"
#include "../src/server/sched_value.cpp"
#include "../src/server/server_sched.cpp"
#include "../src/server/http_server.cpp"
#include "../src/server/systemone.cpp"
#include "../src/server/watchdog.cpp"
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
};

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

/* ---- L278: the Qwen leg ---------------------------------------------------------------------------- */

#define QCALL(name, params) "<tool_call>\n<function=" name ">\n" params "</function>\n</tool_call>"
#define QPARAM(k, v) "<parameter=" k ">\n" v "\n</parameter>\n"

/* A Qwen generation is the text after the rendered prompt: with thinking on the prompt ended in "<think>\n", so
 * the turn opens in reasoning. */
static const gcase QCASES[] = {
    {"q-think-answer", true, false, "Let me compute 17*23.\n</think>\n\nThe answer is 391.", "stop"},
    {"q-think-unclosed", true, false, "thinking on and on, never closing", "length"},
    {"q-no-think", false, false, "Plain answer with <angle> and </think> text.", "stop"},
    {"q-utf8", true, false, "naïve 日本語\n</think>\n\n🌞 ok — done", "stop"},
    {"q-partial-lt", true, true, "a\n</think>\n\nx < y and <b>bold</b> and <tool_ca not a call", "stop"},
    {"q-tool-after-think", true, true, "plan the call\n</think>\n\n" QCALL("bash", QPARAM("command", "ls -l /var/log")),
     "stop"},
    {"q-text-then-tool", true, true, "plan\n</think>\n\nLet me look.\n\n" QCALL("bash", QPARAM("command", "pwd")),
     "stop"},
    {"q-two-calls", true, true,
     "x\n</think>\n\n" QCALL("bash", QPARAM("command", "a")) "\n"
     QCALL("bash", QPARAM("command", "b & c") QPARAM("description", "two")), "stop"},
    {"q-no-think-tool", false, true, "Sure.\n\n" QCALL("bash", QPARAM("command", "uname -a")), "stop"},
    {"q-tool-unclosed", true, true, "go\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nsleep", "length"},
    {"q-tool-undeclared", true, true, "hm\n</think>\n\n" QCALL("rm_rf", QPARAM("path", "/")), "stop"},
};

/* ---- L294 E6: the Inkling leg -------------------------------------------------------------------------- */

/* An Inkling generation is the text after the rendered prompt's generation prompt <|message_model|>, up to (not
 * including) the stop token <|content_model_end_sampling|>: the turn opens in its first block's header. */
#define IBLOCK(kind, body) "<|message_model|><|content_" kind "|>" body "<|end_message|>"
#define ICALL(name, args) "<|message_model|>" name "<|content_invoke_tool_json|>{\"name\":\"" name "\",\"args\":" args \
                          "}<|end_message|>"
static const gcase ICASES[] = {
    {"i-think-answer", true, false,
     "<|content_thinking|>Let me compute 17*23.<|end_message|>" IBLOCK("text", "The answer is 391."), "stop"},
    {"i-think-unclosed", true, false, "<|content_thinking|>thinking on and on, never closing", "length"},
    {"i-no-think", false, false, "<|content_text|>Plain answer with <angle> and </think> text.<|end_message|>", "stop"},
    {"i-utf8", true, false, "<|content_thinking|>naïve 日本語  <|end_message|>" IBLOCK("text", "🌞 ok — done"), "stop"},
    {"i-partial-marker", true, true,
     "<|content_text|>x < y and <b>bold</b> and <|content_invoke not a call<|end_message|>", "stop"},
    {"i-tool-after-think", true, true,
     "<|content_thinking|>plan the call<|end_message|>" ICALL("bash", "{\"command\":\"ls -l /var/log\"}"), "stop"},
    {"i-text-then-tool", true, true,
     "<|content_thinking|>plan<|end_message|>" IBLOCK("text", "Let me look.") ICALL("bash", "{\"command\":\"pwd\"}"),
     "stop"},
    {"i-two-calls", true, true,
     "<|content_thinking|>x<|end_message|>" ICALL("bash", "{\"command\":\"a\"}")
     ICALL("bash", "{\"command\":\"b & c\",\"description\":\"two\"}"), "stop"},
    {"i-tool-first", false, true,
     "bash<|content_invoke_tool_json|>{\"name\":\"bash\",\"args\":{\"command\":\"uname -a\"}}<|end_message|>",
     "stop"},
    {"i-tool-unclosed", true, true,
     "<|content_thinking|>go<|end_message|><|message_model|>bash<|content_invoke_tool_json|>{\"name\":\"bash\","
     "\"args\":{\"command\":\"sleep", "length"},
    {"i-tool-undeclared", true, true,
     "<|content_thinking|>hm<|end_message|>" ICALL("rm_rf", "{\"path\":\"/\"}"), "stop"},
    {"i-tool-malformed", true, true,
     "<|content_thinking|>j<|end_message|><|message_model|>bash<|content_invoke_tool_json|>{\"name\":\"bash\","
     "\"args\":[1, 2]}<|end_message|>", "stop"},
};

/* ---- L417: the usage leg ------------------------------------------------------------------------------ */

/// One usage-leg case: a generation and the family whose request it answers.
struct ucase { gcase c; pulsar_chat_format fmt; };
static const ucase UCASES[] = {
    {{"u-ds-think-answer", true, false, "Let me compute 17*23.</think>The answer is 391.", "stop"}, PULSAR_CHAT_DS4_V41},
    {{"u-ds-no-think", false, false, "Plain answer with <think> not reasoning </think> text.", "stop"},
     PULSAR_CHAT_DS4_V41},
    {{"u-q-think-answer", true, false, "Let me compute 17*23.\n</think>\n\nThe answer is 391.", "stop"},
     PULSAR_CHAT_QWEN},
    {{"u-q-think-unclosed", true, false, "thinking on and on, never closing", "length"}, PULSAR_CHAT_QWEN},
    {{"u-q-no-think", false, false, "Plain answer with <think> not reasoning </think> text.", "stop"}, PULSAR_CHAT_QWEN},
    {{"u-q-literal-think-after-close", true, false, "a\n</think>\n\nThe tag <think> opens nothing here.", "stop"},
     PULSAR_CHAT_QWEN},
    {{"u-g-channel-answer", true, false, "<|channel>thought\nLet me compute 17*23.<channel|>The answer is 391.", "stop"},
     PULSAR_CHAT_GEMMA4},
    {{"u-g-two-channels", true, false, "<|channel>thought\nfirst<channel|>Partly. <|channel>thought\nsecond<channel|>Done.",
     "stop"}, PULSAR_CHAT_GEMMA4},
    {{"u-g-no-think", false, false, "Plain answer.", "stop"}, PULSAR_CHAT_GEMMA4},
};

/* the one tool, in each protocol's declaration shape (the renderer reads the request's) */
#define QSCHEMA "{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"},\"description\":{\"type\":\"string\"}}}"
static const char *qtools_for(proto p) {
    switch (p) {
    case P_OPENAI: return "[{\"type\":\"function\",\"function\":{\"name\":\"bash\",\"parameters\":" QSCHEMA "}}]";
    case P_ANTHROPIC: return "[{\"name\":\"bash\",\"input_schema\":" QSCHEMA "}]";
    default: return "[{\"type\":\"function\",\"name\":\"bash\",\"parameters\":" QSCHEMA "}]";
    }
}

/// One run: the generation `c.raw` fed `piece` bytes at a time to protocol `p`'s live projection of a request family
/// `fmt` rendered, then finished as the server finishes it.
/// @param c      the case
/// @param p      the protocol
/// @param piece  bytes a feed (0: the whole text at once)
/// @param fmt    the family's chat format
/// @param usage  L417: the usage leg -- each piece is counted as one generated token (the server's reasoning count)
///               and the chat stream is asked for its usage chunk
/// @return the stream's bytes, ids and timestamps normalised
static std::string run_family(const gcase &c, proto p, size_t piece, pulsar_chat_format fmt, bool usage = false) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return "(socketpair failed)\n";
    fcntl(sv[1], F_SETFL, O_NONBLOCK);
    int big = 1 << 22;
    setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &big, sizeof(big));
    job j;
    memset(&j, 0, sizeof j);
    j.fd = sv[0];
    request *r = &j.req;
    request_init(r, REQ_CHAT, 256);
    free(r->model);
    r->model = server_xstrdup("m");
    r->api = p == P_OPENAI ? API_OPENAI : p == P_ANTHROPIC ? API_ANTHROPIC : API_RESPONSES;
    r->reasoning_summary_emit = p == P_RESPONSES_SUMMARY;
    const bool deepseek = fmt == PULSAR_CHAT_DS4_V4 || fmt == PULSAR_CHAT_DS4_V41;
    if (deepseek) {
        /* DeepSeek: the request as its renderer leaves it -- the family, the think mode, the declared tools */
        r->family = server_family_for_format(fmt);
        r->think_mode = c.think ? PULSAR_THINK_HIGH : PULSAR_THINK_NONE;
        r->has_tools = c.tools;
        if (c.tools)
            tool_schema_orders_add_json(&r->tool_orders,
                "{\"name\":\"bash\",\"input_schema\":{\"type\":\"object\",\"properties\":{"
                "\"command\":{\"type\":\"string\"},\"description\":{\"type\":\"string\"}}}}");
    }
    /* Qwen, Inkling: the request as the server has it after rendering: the family's renderer sets the family, the
     * think mode, the tools JSON (Qwen's parser types arguments with it), and the declared tools a call is checked
     * against */
    chat_conversation conv = {};
    chat_msg u = {0};
    u.role = server_xstrdup("user");
    u.content = server_xstrdup("Go.");
    chat_msgs_push(&conv.msgs, u);
    if (!deepseek && c.tools) {
        conv.tools_raw = server_xstrdup(qtools_for(p));
        tool_schema_orders_add_json(&r->tool_orders,
            "{\"name\":\"bash\",\"parameters\":{\"type\":\"object\",\"properties\":{"
            "\"command\":{\"type\":\"string\"},\"description\":{\"type\":\"string\"}}}}");
    }
    if (!c.think) chat_conversation_control(&conv, "enable_thinking", server_xstrdup("false"));
    char err[256] = "";
    if (!deepseek && !render_chat_conversation(NULL, fmt, NULL, &conv, r, err, sizeof err)) {
        chat_conversation_free(&conv);
        request_free(r);
        close(sv[0]);
        close(sv[1]);
        return std::string("(render failed: ") + err + ")\n";
    }
    chat_conversation_free(&conv);
    r->stream = true;
    r->stream_include_usage = usage;

    server srv;
    memset(&srv, 0, sizeof srv);
    pthread_mutex_init(&srv.tool_mu, NULL);
    gen_state g;
    memset(&g, 0, sizeof g);
    g.j = &j;
    g.thinking = thinking_state_from_prompt(r);   /* the server's gen_decode_init */
    g.reasoning_block = g.thinking.inside;
    int fed = 0;   /* L417: pieces fed = the usage leg's generated tokens */
    streams st;
    memset(&st, 0, sizeof(st));
    std::string out;
    bool ok = true;
    if (p == P_OPENAI) {
        openai_stream_start(r, &st.oa);
        openai_sink_init(&g.sink, sv[0], NULL, r, "chatcmpl-X", &st.oa);
    } else if (p == P_ANTHROPIC) {
        ok = anthropic_sse_start_live(sv[0], r, "msg_X", 10, &st.an);
        anthropic_sink_init(&g.sink, sv[0], NULL, r, "msg_X", &st.an);
    } else {
        responses_stream_init(r, &st.rs);
        st.rs.active = true;
        ok = responses_sse_created(sv[0], r, &st.rs, 1700000000L);
        responses_sink_init(&g.sink, sv[0], NULL, r, "resp_X", &st.rs);
    }
    const server_output_parser_ops *ops = r->family->output;
    void *ps = ops->create(&srv, &g, err, sizeof err);
    if (!ps) {
        out = std::string("(parser create failed: ") + err + ")\n";
        ok = false;
    }
    const size_t n = strlen(c.raw);
    for (size_t at = 0; ok && at < n;) {
        const size_t to = piece ? std::min(n, at + piece) : n;
        buf_append(&g.text, c.raw + at, to - at);
        const bool was_inside = g.thinking.inside;
        g.thinking.feed(c.raw + at, to - at);
        if (usage) reasoning_usage_note(&g, was_inside);   /* gen_emit_token's count, a piece a token */
        fed++;
        at = to;
        ok = ops->feed(ps, &srv, &g, gen_text_end(g.text.len), false);
        drain(sv[1], &out);
    }
    server_turn turn;
    memset(&turn, 0, sizeof turn);
    if (ps) {
        g.finish = c.finish;
        turn.finish = g.finish;
        const bool stream_ok = ops->finish(ps, &srv, NULL, &g, &turn);
        if (ok) ok = stream_ok && ops->feed(ps, &srv, &g, gen_text_end(g.text.len), true);
        if (ok) {
            switch (p) {
            case P_OPENAI: ok = openai_sse_finish(&g.sink, &turn.calls, turn.finish, 10, usage ? fed : 20); break;
            case P_ANTHROPIC: ok = anthropic_sse_finish(&g.sink, &turn.calls, turn.finish, NULL, 20); break;
            default:
                ok = responses_sse_finish(sv[0], r, &st.rs, turn.tail, turn.tail_len, &turn.calls, turn.finish, 10,
                                          usage ? fed : 20, 1700000000L);
            }
        }
        ops->destroy(ps);
    }
    drain(sv[1], &out);
    if (!ok) out += "(a write failed)\n";
    free(turn.content);
    free(turn.reasoning);
    tool_calls_free(&turn.calls);
    buf_free(&g.text);
    responses_stream_free(&st.rs);
    request_free(r);
    pthread_mutex_destroy(&srv.tool_mu);
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
                fputs(run_family(c, (proto)p, piece, PULSAR_CHAT_DS4_V41).c_str(), stdout);
                n++;
            }
        }
    }
    for (const gcase &c : QCASES) {
        for (int p = P_OPENAI; p <= P_RESPONSES_SUMMARY; p++) {
            for (size_t piece : {(size_t)1, (size_t)3, (size_t)7, (size_t)0}) {
                printf("== %s [%s] piece %zu\n", c.name, PROTO_NAME[p], piece);
                fputs(run_family(c, (proto)p, piece, PULSAR_CHAT_QWEN).c_str(), stdout);
                n++;
            }
        }
    }
    for (const gcase &c : ICASES) {
        for (int p = P_OPENAI; p <= P_RESPONSES_SUMMARY; p++) {
            for (size_t piece : {(size_t)1, (size_t)3, (size_t)7, (size_t)0}) {
                printf("== %s [%s] piece %zu\n", c.name, PROTO_NAME[p], piece);
                fputs(run_family(c, (proto)p, piece, PULSAR_CHAT_INKLING).c_str(), stdout);
                n++;
            }
        }
    }
    for (const ucase &u : UCASES) {
        for (proto p : {P_OPENAI, P_RESPONSES}) {
            for (size_t piece : {(size_t)1, (size_t)7}) {
                printf("== %s [%s] piece %zu\n", u.c.name, PROTO_NAME[p], piece);
                fputs(run_family(u.c, p, piece, u.fmt, true).c_str(), stdout);
                n++;
            }
        }
    }
    fprintf(stderr, "sse_golden: %d runs\n", n);
    return 0;
}

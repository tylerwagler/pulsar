/** @file
 *  What the generation path's host work costs by the amount of text generated so far, before and after L430 / L431. */
/* gen_host_bench.cpp -- L430, L431: host work on the scheduler's worker thread that grew with the turn's text.
 *
 * Tokens are emitted on the one thread that launches every model forward, so host time per token or per speculative
 * round delays the next token of every live session.  This prints, for a sampled chat with tools at a point where
 * 10 KB .. 4 MB of text have been generated:
 *
 *   a speculative round's look ahead (L430) -- which of the round's rows are a call's structure
 *   (gen_spec_greedy_rows_text, every round) and where a single-call turn's pending chain ends
 *   (gen_turn_draft_prefix_text, every round of such a turn) -- for a round of 5 tokens, inside an open tool call
 *   whose string value is that long and after plain content of that length, with a two-tool schema and with a
 *   30-tool one; the walks as they were (tests/gen_host_ref.h: a copy of the text and a clone of the parser a
 *   round) against the live ones (read ahead in place).
 *
 *   the stop-string scan (L431) -- gen_release, every token: the total over a turn of 10 KB .. 1 MB that no stop
 *   string ends (4-byte tokens, one line of words), for the stop lists ["\n"], ["\n", "."], ["STOP"] and none; the
 *   release as it was (the resume point of a one-byte list never moved) against the live one.
 *
 * "before" is the reference header's code run in THIS tree.  One of the costs L430 removed is not in it: the parser's
 * clone copied the request's parsed tools, which the bodies now share, so the reference's clone is cheaper here than
 * it was.  The same source builds in a tree at the commit before L430 (copy this file and tests/gen_host_ref.h into
 * it with the Makefile's rule); its "after" column is then the old code as it really ran.
 *
 * It prints; it does not grade.  It does check, and die on, the one thing that would make the numbers meaningless:
 * the two readings giving different answers. */
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

#include <malloc.h>
#include <time.h>

#include <string>
#include <vector>

#include "gen_host_ref.h"

namespace {

/// @return this process's CPU time in seconds
double cpu_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/// Say what went wrong and stop: the numbers would mean nothing.
/// @param what the failure
[[noreturn]] void bench_die(const char *what) {
    fprintf(stderr, "gen_host_bench: %s\n", what);
    exit(1);
}

/// @param n       the length wanted
/// @param breaks  the text has line breaks and full stops (else it is one line of words)
/// @return `n` bytes of text
std::string some_text(size_t n, bool breaks) {
    static const char *const lines[] = {
        "    for (size_t i = 0; i < rows.size(); i++) {\n",
        "        if (rows[i].width < limit && rows[i].name != \"\") out << \"<td>\" << rows[i].name << \"</td>\";\n",
        "        else out << \"<td class='wide'>\" << escape(rows[i].name) << \"</td>\\n\";\n",
        "    }\n",
        "    return std::vector<std::pair<int, std::string>>(table.begin(), table.end());\n",
        "// the next section: see <https://example.org/a/b?c=d&e=f> for the layout\n",
    };
    static const char *const words[] = {"the ", "table ", "of ", "rows ", "and ", "their ", "widths ", "is ", "read ", "once "};
    std::string t;
    for (size_t k = 0; t.size() < n; k++) t += breaks ? lines[k % 6] : words[(k * 7 + k / 10) % 10];
    t.resize(n);
    return t;
}

/// @param tools  how many tools
/// @return an OpenAI tools array: `write` (path, content: strings) and tools - 1 others of eight described parameters
std::string tools_json(int tools) {
    std::string t = "[{\"type\":\"function\",\"function\":{\"name\":\"write\",\"description\":\"Write a file.\","
                    "\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},"
                    "\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}}}";
    for (int i = 1; i < tools; i++) {
        t += ",{\"type\":\"function\",\"function\":{\"name\":\"tool_" + std::to_string(i) +
             "\",\"description\":\"Does one of the things an agent's tools do, described at the length such "
             "descriptions run to, with its caveats and an example of when not to use it.\",\"parameters\":{\"type\":"
             "\"object\",\"properties\":{";
        for (int k = 0; k < 8; k++)
            t += std::string(k ? "," : "") + "\"arg_" + std::to_string(k) +
                 "\":{\"type\":" + (k % 3 ? "\"string\"" : "[\"integer\",\"null\"]") +
                 ",\"description\":\"What this argument selects, its default and its unit.\"}";
        t += "},\"required\":[\"arg_0\"]}}}";
    }
    return t + "]";
}

/// One decode attempt of a Qwen chat with tools, as far as the emit path took it.
struct attempt {
    server srv;        ///< a server with no engine: the emit path's host side only
    job j;             ///< the job (its request)
    gen_state *g;      ///< the attempt

    /// @param tools   the request's tools array
    /// @param single  the turn may make one call (L428: the chain walk runs, the parser keeps a read-ahead copy)
    /// @param stops   the request's stop strings
    attempt(const std::string &tools, bool single, const std::vector<std::string> &stops) {
        memset(&srv, 0, sizeof srv);
        pthread_mutex_init(&srv.tool_mu, NULL);
        memset(&j, 0, sizeof j);
        j.fd = -1;
        request *r = &j.req;
        request_init(r, REQ_CHAT, 1 << 30);
        r->family = server_family_for_format(PULSAR_CHAT_QWEN);
        r->model = server_xstrdup("bench");
        r->api = API_OPENAI;
        const std::string body = "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"temperature\":0.7,"
                                 "\"enable_thinking\":false,\"tools\":" + tools +
                                 (single ? ",\"parallel_tool_calls\":false}" : "}");
        chat_conversation c;
        memset(&c, 0, sizeof c);
        char err[300] = "";
        if (!parse_chat_conversation_openai(body.c_str(), &c, r, err, sizeof err) ||
            !render_chat_conversation(NULL, PULSAR_CHAT_QWEN, &srv, &c, r, err, sizeof err))
            bench_die(err);
        chat_conversation_free(&c);
        for (const std::string &s : stops) stop_list_push(&r->stops, server_xstrdup(s.c_str()));
        g = new gen_state();
        g->j = &j;
        g->finish = "length";
        g->next_decode_log = TokCount{1 << 30};
        g->thinking = thinking_state_from_prompt(r);
        g->parser = r->family->output;
        g->parser_st = g->parser->create(&srv, g, err, sizeof err);
        if (!g->parser_st) bench_die(err);
    }
    attempt(const attempt &) = delete;
    attempt &operator=(const attempt &) = delete;
    ~attempt() {
        g->parser->destroy(g->parser_st);
        free(g->stop_sequence);
        buf_free(&g->text);
        delete g;
        request_free(&j.req);
        pthread_mutex_destroy(&srv.tool_mu);
    }
    /// Emit `text` through the emit path, 64 bytes a token.
    /// @param text  the generated text
    void emit(const std::string &text) {
        for (size_t at = 0; at < text.size(); at += 64)
            if (srv.gen_emit_piece(g, TokId{0}, text.data() + at, std::min<size_t>(64, text.size() - at)))
                bench_die("the emit path ended the turn before the measured point");
    }
};

/// The round the walks read ahead of: five tokens that go on with the text.
const char *const k_round[5] = {"    ", "for ", "(size", "_t i", " = 0"};

/// Time one walk, the reference and the live one, and print the row's two cells.
/// @param a      the attempt, at the measured point
/// @param chain  time the chain walk (gen_turn_draft_prefix_text); else the greedy rows
void walk_cells(attempt *a, bool chain) {
    size_t lens[5];
    for (int i = 0; i < 5; i++) lens[i] = strlen(k_round[i]);
    const size_t size = a->g->text.len;
    const long ref_n = size >= ((size_t)1 << 20) ? 300 : size >= ((size_t)64 << 10) ? 3000 : 20000, live_n = 20000;
    long sink_ref = 0, sink_live = 0;
    double t0 = cpu_s();
    for (long i = 0; i < ref_n; i++)
        sink_ref += chain ? gen_turn_draft_prefix_text_ref(a->g, k_round, lens, 4)
                          : (long)gen_spec_greedy_rows_text_ref(a->g, k_round, lens, 5, false);
    const double ref = (cpu_s() - t0) / (double)ref_n * 1e6;
    t0 = cpu_s();
    for (long i = 0; i < live_n; i++)
        sink_live += chain ? gen_turn_draft_prefix_text(a->g, k_round, lens, TokCount{4}).raw()
                           : (long)gen_spec_greedy_rows_text(a->g, k_round, lens, TokCount{5}, false);
    const double live = (cpu_s() - t0) / (double)live_n * 1e6;
    if (sink_ref / ref_n != sink_live / live_n || a->g->text.len != size)
        bench_die("the reference and the live walk read the round differently");
    printf(" %10.2f %10.2f", ref, live);
}

/// One row of the look-ahead table: a point in a turn, both walks.
/// @param what     the row's label
/// @param tools    the request's tools
/// @param text     the text generated up to the measured point
void walk_row(const char *what, const std::string &tools, const std::string &text) {
    printf("%-46s", what);
    {
        attempt a(tools, false, {});
        a.emit(text);
        walk_cells(&a, false);
    }
    {
        attempt a(tools, true, {});   /* a single-call turn: both walks run every round, and there is a read-ahead */
        a.emit(text);
        walk_cells(&a, false);
        walk_cells(&a, true);
    }
    printf("\n");
    fflush(stdout);
}

/// One row of the stop-scan table: a turn's releases under one stop list, the release as it was and the live one.
/// @param what   the row's label
/// @param stops  the stop strings
/// @param n      the turn's length, bytes
void stop_row(const char *what, const std::vector<std::string> &stops, size_t n) {
    const std::string text = some_text(n, false);
    stop_list list;
    memset(&list, 0, sizeof list);
    for (const std::string &s : stops) stop_list_push(&list, server_xstrdup(s.c_str()));
    double total[2], last[2];
    size_t released[2];
    for (int live = 0; live < 2; live++) {
        thinking_state thinking;
        memset(&thinking, 0, sizeof thinking);
        size_t from = 0, sent = 0, pos = 0, len = 0, up = 0;
        bool hit = false;
        buf b;
        memset(&b, 0, sizeof b);
        double in_release = 0, t_last = 0;
        for (size_t at = 0; at < text.size(); at += 4) {
            const size_t k = std::min<size_t>(4, text.size() - at);
            buf_append(&b, text.data() + at, k);
            const double t0 = cpu_s();
            up = live ? gen_release_bare(&list, b.ptr, b.len, text.data() + at, k, sent, &thinking, &from, &hit, &pos, &len)
                      : gen_release_ref(&list, b.ptr, b.len, text.data() + at, k, sent, &thinking, &from, &hit, &pos, &len);
            t_last = cpu_s() - t0;
            in_release += t_last;
            if (hit) bench_die("a stop string ended the turn that was to be timed whole");
        }
        total[live] = in_release;
        last[live] = t_last;
        released[live] = up;
        buf_free(&b);
    }
    if (released[0] != released[1]) bench_die("the release as it was and the live one released different text");
    printf("%-36s %9zu %12.2f %12.2f %14.2f %12.2f\n", what, (n + 3) / 4, total[0] * 1e3, total[1] * 1e3, last[0] * 1e6,
           last[1] * 1e6);
    fflush(stdout);
    stop_list_clear(&list);
    free(list.v);
}

}  // namespace

int main(void) {
    /* The reference's megabyte copies are timed as copies: with glibc's defaults a large block is mapped afresh and
     * returned to the kernel at each free, and the page faults then cost several times the copy (4-6x at 1 MB on
     * the VM this was written on) by an amount that depends on what the process freed before.  A server pays some of
     * that; the bench leaves it out, so "before" is the floor of what a round cost. */
    mallopt(M_MMAP_THRESHOLD, 1 << 30);
    mallopt(M_TRIM_THRESHOLD, 1 << 30);
    const std::string small = tools_json(2), large = tools_json(30);
    const std::string open_call = "<tool_call>\n<function=write>\n<parameter=path>\nsrc/table.cpp\n</parameter>\n"
                                  "<parameter=content>\n";
    printf("gen_host_bench: host work per speculative round, by the text generated so far (CPU microseconds; "
           "before = tests/gen_host_ref.h, after = the live code)\n\n");
    printf("a round's look ahead, 5 tokens (L430).  tools: %zu-byte schema (2 tools) / %zu-byte schema (30 tools)\n",
           small.size(), large.size());
    printf("%-46s %21s %21s %21s\n", "", "greedy rows", "greedy rows, 1-call", "chain walk, 1-call");
    printf("%-46s %10s %10s %10s %10s %10s %10s\n", "the turn so far", "before", "after", "before", "after", "before",
           "after");
    for (size_t n : {(size_t)10 << 10, (size_t)100 << 10, (size_t)1 << 20, (size_t)4 << 20})
        walk_row(("open call, value of " + std::to_string(n >> 10) + " KB, 2 tools").c_str(), small,
                 open_call + some_text(n, true));
    for (size_t n : {(size_t)10 << 10, (size_t)100 << 10, (size_t)1 << 20, (size_t)4 << 20})
        walk_row(("plain content of " + std::to_string(n >> 10) + " KB, 2 tools").c_str(), small, some_text(n, true));
    for (size_t n : {(size_t)10 << 10, (size_t)1 << 20})
        walk_row(("open call, value of " + std::to_string(n >> 10) + " KB, 30 tools").c_str(), large,
                 open_call + some_text(n, true));

    printf("\nthe stop-string scan over a whole turn that no stop string ends, 4-byte tokens (L431).  The clock is read "
           "around each release:\nboth columns carry that, about 0.3 us a token (the rows with no stop string are it)\n");
    printf("%-36s %9s %12s %12s %14s %12s\n", "", "", "turn, ms", "turn, ms", "last token,", "last token,");
    printf("%-36s %9s %12s %12s %14s %12s\n", "stop strings, the turn's length", "tokens", "before", "after", "us before",
           "us after");
    const struct {
        const char *name;
        std::vector<std::string> stops;
    } lists[] = {{"[\"\\n\"]", {"\n"}}, {"[\"\\n\", \".\"]", {"\n", "."}}, {"[\"STOP\"]", {"STOP"}}, {"none", {}}};
    for (const auto &l : lists)
        for (size_t n : {(size_t)10 << 10, (size_t)100 << 10, (size_t)1 << 20})
            stop_row((std::string(l.name) + ", " + std::to_string(n >> 10) + " KB").c_str(), l.stops, n);
    return 0;
}

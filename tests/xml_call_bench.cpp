/* xml_call_bench.cpp -- L429: what the XML tool-call body's parse costs while a call streams, before and after.
 *
 * A call in the XML format (Qwen, Nemotron-H) is read as it is generated: every released piece of the turn is one
 * out_parser::feed.  Before L429 each feed read the open parameter from its start again, so one long value (a Write
 * tool's file content) cost its length squared over the whole generation.  This prints the parse's CPU time for one
 * call fed in small pieces, through the kit around the live body (src/lib/xml_call_body.h) and around the body as it
 * was (tests/xml_call_body_ref.h):
 *
 *   one string parameter of 10 KB .. 4 MB, in 4-byte and in 64-byte feeds
 *   a value that spells "</tool_call>" on every line (L410: each one asks holds_closer), in 64-byte feeds
 *   2,000 small parameters, in 4-byte and in 64-byte feeds
 *
 * (What a speculative round's look ahead costs at such a point of a call is tests/gen_host_bench.cpp's, L430.)
 *
 * It prints; it does not grade.  It does check, and die on, the one thing that would make the numbers meaningless:
 * the two bodies reading different calls.
 *
 *   ./tests/xml_call_bench [REF_MAX_BYTES]     the largest value the old body is timed on (default 4194304; its
 *                                               4-byte row at 4 MB takes minutes) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <string>
#include <vector>

#include "../src/families/qwen/qwen_chat.h"
#include "xml_call_body_ref.h"

namespace {

/// The tools of the bench's calls: `content` is a declared string (the payload), the small parameters are not declared.
const char kTools[] =
    R"([{"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": )"
    R"({"path": {"type": "string"}, "content": {"type": "string"}}}}}])";

/// @return this process's CPU time in seconds
double cpu_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/// @param n      the length wanted
/// @param closer every line ends in a spelt "</tool_call>" (L410's case)
/// @return `n` bytes of source-like text (angle brackets, quotes, newlines; no "</parameter>")
std::string file_text(size_t n, bool closer) {
    static const char *const lines[] = {
        "    for (size_t i = 0; i < rows.size(); i++) {\n",
        "        if (rows[i].width < limit && rows[i].name != \"\") out << \"<td>\" << rows[i].name << \"</td>\";\n",
        "        else out << \"<td class='wide'>\" << escape(rows[i].name) << \"</td>\\n\";\n",
        "    }\n",
        "    return std::vector<std::pair<int, std::string>>(table.begin(), table.end());\n",
        "// the next section: see <https://example.org/a/b?c=d&e=f> for the layout\n",
    };
    std::string t;
    for (size_t k = 0; t.size() < n; k++) {
        t += lines[k % (sizeof lines / sizeof lines[0])];
        if (closer) t += "</tool_call>\n";
    }
    t.resize(n);
    return t;
}

/// One timed reading.
struct reading {
    double s = 0;            ///< CPU seconds in feed() and finish()
    size_t feeds = 0;        ///< feeds
    size_t fragments = 0;    ///< TOOL_ARGS events
    std::string arguments;   ///< the call's arguments (empty: no call was read)
};

/// @param turn  the turn's bytes
/// @param piece bytes a feed
/// @return the reading of `turn` through the kit around `Body`
template <class Body> reading read_turn(const std::string &turn, size_t piece) {
    call_body_parser<Body> p;
    if (!p.start(qwen_output_grammar(false), kTools, false, OUT_CONTENT)) {
        fprintf(stderr, "xml_call_bench: the tools did not parse\n");
        exit(1);
    }
    reading r;
    std::vector<out_event> ev;
    const double t0 = cpu_s();
    for (size_t at = 0; at < turn.size(); at += piece) {
        ev.clear();
        p.feed(turn.data() + at, std::min(piece, turn.size() - at), &ev);
        r.feeds++;
        for (const out_event &e : ev) r.fragments += e.kind == out_event::TOOL_ARGS;
    }
    ev.clear();
    p.finish(&ev);
    r.s = cpu_s() - t0;
    if (p.calls().size() == 1) r.arguments = p.calls()[0].arguments;
    return r;
}

/// Time one turn through both bodies and print the row.
/// @param what    the row's label
/// @param turn    the turn's bytes
/// @param piece   bytes a feed
/// @param ref     time the old body too
void row(const char *what, const std::string &turn, size_t piece, bool ref) {
    const reading now = read_turn<xml_call_body>(turn, piece);
    if (now.arguments.empty()) {
        fprintf(stderr, "xml_call_bench: %s: no call was read\n", what);
        exit(1);
    }
    if (!ref) {
        printf("%-34s %3zu B %9zu feeds %12s %10.4f s %10s   (%zu fragments)\n", what, piece, now.feeds, "-", now.s, "-",
               now.fragments);
        return;
    }
    const reading old = read_turn<xml_call_body_ref>(turn, piece);
    if (old.arguments != now.arguments || old.fragments != now.fragments) {
        fprintf(stderr, "xml_call_bench: %s: the two bodies read different calls\n", what);
        exit(1);
    }
    printf("%-34s %3zu B %9zu feeds %10.4f s %10.4f s %9.1fx   (%zu fragments)\n", what, piece, now.feeds, old.s, now.s,
           now.s > 0 ? old.s / now.s : 0.0, now.fragments);
    fflush(stdout);
}

/// @param value the `content` parameter's text
/// @return a turn with one call writing it
std::string one_value_turn(const std::string &value) {
    return "<tool_call>\n<function=write>\n<parameter=path>\nsrc/table.cpp\n</parameter>\n<parameter=content>\n" + value +
           "\n</parameter>\n</function>\n</tool_call>";
}

}  // namespace

int main(int argc, char **argv) {
    const size_t ref_max = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : (size_t)4 << 20;
    printf("xml_call_bench: one call's parse while it streams -- CPU seconds in out_parser::feed, the body before "
           "L429 (xml_call_body_ref.h) and now\n\n");
    printf("%-34s %5s %15s %12s %12s %10s\n", "case", "feed", "", "before", "after", "speedup");
    for (size_t piece : {(size_t)4, (size_t)64})
        for (size_t n : {(size_t)10 << 10, (size_t)100 << 10, (size_t)1 << 20, (size_t)4 << 20})
            row(("one string value, " + std::to_string(n >> 10) + " KB").c_str(), one_value_turn(file_text(n, false)), piece,
                n <= ref_max);
    printf("\n");
    for (size_t n : {(size_t)10 << 10, (size_t)100 << 10, (size_t)1 << 20})
        row(("value spelling the closer, " + std::to_string(n >> 10) + " KB").c_str(), one_value_turn(file_text(n, true)), 64,
            n <= ref_max);
    printf("\n");
    std::string many = "<tool_call>\n<function=write>\n";
    for (int i = 0; i < 2000; i++)
        many += "<parameter=k" + std::to_string(i) + ">\n" + (i % 3 ? "value " + std::to_string(i) : std::to_string(i)) +
                "\n</parameter>\n";
    many += "</function>\n</tool_call>";
    for (size_t piece : {(size_t)4, (size_t)64}) row("2000 small parameters", many, piece, true);
    return 0;
}

/* L303: the cost of constrained decoding on the host, at a family's real vocabulary -- what a constrained request
 * adds to a draw.  No engine and no GPU: the checkpoint's tokenizer (src/lib/bpe_tokenizer) supplies the token
 * bytes and the stop ids, the family's render builds the plan, and the gate walks the template's own bytes for a
 * call and for an answer, as the model would emit them (the tokenizer's encoding of them):
 *
 *   ./tests/grammar_bench --tokenizer DIR --family qwen|nemotron|deepseek --width LOGITS_WIDTH
 *   ./tests/grammar_bench --vocab-tsv FILE --stop ID --family deepseek --width LOGITS_WIDTH
 *
 * DIR holds tokenizer.json and generation_config.json; FILE is a vocabulary dump (for a tokenizer whose
 * pre-tokenizer the bpe loader does not register), encoded greedily -- an approximate tokenization.  Per case it prints the plan's compile (cold, then cached),
 * then per token of the walk: the mask fill (the plain lane's cost per constrained draw), the accept, and a
 * speculative round's masks over the next 1 + K tokens (gen_constrain_round, K = 8: the verify rows' cost).  A
 * token the mask refuses is a finding: the template's bytes, tokenized, leave the grammar there. */
#define PULSAR_SERVER_TEST
#define PULSAR_SERVER_TEST_NO_MAIN
#include "../src/server/util.cpp"
#include "../src/server/request.cpp"
#include "../src/server/prompt_render.cpp"
#include "../src/server/api_parse.cpp"
#include "../src/server/chat_family.cpp"
#include "../src/server/family_inkling.cpp"
#include "../src/server/family_nemotron_h.cpp"
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
#include "../src/server/server_sched.cpp"
#include "../src/server/http_server.cpp"
#include "../src/server/cli_main.cpp"
#include "../src/lib/bpe_tokenizer.h"
#include "../src/lib/pulsar_grammar.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool slurp(const std::string &path, std::string *out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char b[65536];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) out->append(b, n);
    fclose(f);
    return true;
}

struct stats {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    void print(const char *what) {
        if (v.empty()) {
            printf("    %-14s n=0\n", what);
            return;
        }
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        double sum = 0;
        for (double x : s) sum += x;
        printf("    %-14s n=%-4zu mean=%8.1f us  p50=%8.1f  p90=%8.1f  p99=%8.1f  max=%8.1f\n", what, s.size(),
               sum / (double)s.size(), s[s.size() / 2], s[s.size() * 9 / 10], s[s.size() * 99 / 100], s.back());
    }
};

#define TOOLS                                                                                                         \
    "\"tools\":["                                                                                                     \
    "{\"type\":\"function\",\"function\":{\"name\":\"search_docs\",\"strict\":true,\"parameters\":{\"type\":"        \
    "\"object\",\"properties\":{\"query\":{\"type\":\"string\"},\"limit\":{\"type\":\"integer\",\"minimum\":1,"      \
    "\"maximum\":50}},\"required\":[\"query\",\"limit\"],\"additionalProperties\":false}}},"                         \
    "{\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"strict\":true,\"parameters\":{\"type\":"          \
    "\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"offset\":{\"type\":\"integer\"}},\"required\":"     \
    "[\"path\"],\"additionalProperties\":false}}},"                                                                  \
    "{\"type\":\"function\",\"function\":{\"name\":\"run_command\",\"strict\":true,\"parameters\":{\"type\":"        \
    "\"object\",\"properties\":{\"command\":{\"type\":\"string\"},\"timeout_s\":{\"type\":\"number\"},\"env\":"      \
    "{\"type\":\"object\",\"additionalProperties\":{\"type\":\"string\"}}},\"required\":[\"command\"],"              \
    "\"additionalProperties\":false}}},"                                                                             \
    "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"strict\":true,\"parameters\":{\"type\":"        \
    "\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"unit\":{\"type\":\"string\",\"enum\":[\"c\","       \
    "\"f\"]}},\"required\":[\"city\",\"unit\"],\"additionalProperties\":false}}}]"

#define RF                                                                                                            \
    "\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"doc\",\"strict\":true,\"schema\":"    \
    "{\"type\":\"object\",\"properties\":{\"title\":{\"type\":\"string\"},\"tags\":{\"type\":\"array\",\"items\":"    \
    "{\"type\":\"string\"}},\"score\":{\"type\":\"number\"},\"author\":{\"type\":\"object\",\"properties\":"         \
    "{\"name\":{\"type\":\"string\"},\"email\":{\"type\":\"string\"}},\"required\":[\"name\",\"email\"],"            \
    "\"additionalProperties\":false},\"status\":{\"type\":\"string\",\"enum\":[\"draft\",\"final\"]}},"              \
    "\"required\":[\"title\",\"tags\",\"score\",\"author\",\"status\"],\"additionalProperties\":false}}}"

static const char *k_answer =
    "{\"title\": \"Configuring the switchyard gate\", \"tags\": [\"gateway\", \"routing\", \"ops\"], \"score\": 0.93, "
    "\"author\": {\"name\": \"Tyler\", \"email\": \"tyler@example.com\"}, \"status\": \"draft\"}";

int main(int argc, char **argv) {
    std::string dir, family, vocab_tsv;
    std::vector<int> stop_ids;
    int width = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--tokenizer")) dir = argv[i + 1];
        else if (!strcmp(argv[i], "--vocab-tsv")) vocab_tsv = argv[i + 1];
        else if (!strcmp(argv[i], "--stop")) stop_ids.push_back(atoi(argv[i + 1]));
        else if (!strcmp(argv[i], "--family")) family = argv[i + 1];
        else if (!strcmp(argv[i], "--width")) width = atoi(argv[i + 1]);
    }
    pulsar_chat_format fmt;
    std::string call_text, think_close;
    if (family == "qwen" || family == "nemotron") {
        fmt = family == "qwen" ? PULSAR_CHAT_QWEN : PULSAR_CHAT_NEMOTRON_H;
        call_text = "<tool_call>\n<function=search_docs>\n<parameter=query>\nhow to configure the switchyard gate\n"
                    "</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>";
        think_close = family == "qwen" ? "I should search the docs.\n</think>\n\n" : "I should search the docs.\n</think>\n";
    } else if (family == "deepseek") {
        fmt = PULSAR_CHAT_DS4_V41;
        call_text = "\n\n" PULSAR_TOOL_CALLS_START "\n" PULSAR_INVOKE_START " name=\"search_docs\">\n" PULSAR_PARAM_START
                    " name=\"query\" string=\"true\">how to configure the switchyard gate" PULSAR_PARAM_END "\n"
                    PULSAR_PARAM_START " name=\"limit\" string=\"false\">5" PULSAR_PARAM_END "\n" PULSAR_INVOKE_END "\n"
                    PULSAR_TOOL_CALLS_END;
        think_close = "I should search the docs.</think>";
    } else {
        fprintf(stderr, "usage: %s --tokenizer DIR --family qwen|nemotron|deepseek --width N\n", argv[0]);
        return 2;
    }
    /* the token table: the checkpoint's tokenizer through src/lib/bpe_tokenizer, or -- for one whose pre-tokenizer
     * that loader does not register (DeepSeek's) -- a dump of its vocabulary ("id<TAB>hexbytes" a line, with
     * --stop ids), encoded greedily by the longest token: an APPROXIMATE tokenization, labelled as such */
    char err[400] = "";
    bpe_tokenizer *tok = NULL;
    std::vector<std::string> bytes;
    std::vector<int> stops;
    int n_tok = 0;
    if (!vocab_tsv.empty()) {
        FILE *f = fopen(vocab_tsv.c_str(), "r");
        if (!f) {
            fprintf(stderr, "grammar_bench: cannot read %s\n", vocab_tsv.c_str());
            return 1;
        }
        char line[4096];
        while (fgets(line, sizeof line, f)) {
            char *tab = strchr(line, '\t');
            if (!tab) continue;
            const int id = atoi(line);
            std::string b;
            for (const char *h = tab + 1; h[0] && h[1] && h[0] != '\n'; h += 2) {
                unsigned v = 0;
                sscanf(h, "%2x", &v);
                b.push_back((char)v);
            }
            if ((size_t)id >= bytes.size()) bytes.resize((size_t)id + 1);
            bytes[(size_t)id] = b;
        }
        fclose(f);
        n_tok = (int)bytes.size();
        stops = stop_ids;
        printf("(approximate tokenization: greedy longest match over %s)\n", vocab_tsv.c_str());
    } else {
        std::string tj, gj;
        if (!slurp(dir + "/tokenizer.json", &tj) || !slurp(dir + "/generation_config.json", &gj)) {
            fprintf(stderr, "grammar_bench: no tokenizer.json + generation_config.json in %s\n", dir.c_str());
            return 1;
        }
        tok = bpe_tokenizer_load_hf(tj.data(), tj.size(), gj.data(), gj.size(), err, sizeof err);
        if (!tok) {
            fprintf(stderr, "grammar_bench: tokenizer: %s\n", err);
            return 1;
        }
        n_tok = bpe_tokenizer_n_tokens(tok);
        bytes.resize((size_t)n_tok);
        for (int t = 0; t < n_tok; t++) {
            size_t n = 0;
            const char *b = bpe_tokenizer_token_bytes(tok, t, &n);
            if (b) bytes[(size_t)t].assign(b, n);
        }
        stops = bpe_tokenizer_stop_ids(tok);
    }
    if (width < n_tok) width = n_tok;
    bytes.resize((size_t)width);
    std::vector<const char *> ptrs((size_t)width);
    std::vector<size_t> lens((size_t)width);
    for (int t = 0; t < width; t++) {
        ptrs[(size_t)t] = bytes[(size_t)t].data();
        lens[(size_t)t] = bytes[(size_t)t].size();
    }
    /* greedy longest-match encoder for the dump mode */
    std::unordered_map<std::string, int> by_bytes;
    size_t longest = 0;
    for (int t = 0; t < n_tok; t++)
        if (!bytes[(size_t)t].empty() && !by_bytes.count(bytes[(size_t)t])) {
            by_bytes[bytes[(size_t)t]] = t;
            longest = std::max(longest, bytes[(size_t)t].size());
        }
    auto encode = [&](const std::string &text, std::vector<int> *ids) -> bool {
        if (tok) return bpe_tokenizer_encode(tok, text.data(), text.size(), NULL, 0, ids, err, sizeof err);
        for (size_t i = 0; i < text.size();) {
            size_t n = std::min(longest, text.size() - i);
            for (; n > 0; n--) {
                auto it = by_bytes.find(text.substr(i, n));
                if (it != by_bytes.end()) {
                    ids->push_back(it->second);
                    break;
                }
            }
            if (!n) {
                snprintf(err, sizeof err, "no token for byte %zu", i);
                return false;
            }
            i += n;
        }
        return true;
    };
    server srv;
    memset(&srv, 0, sizeof srv);
    pthread_mutex_init(&srv.tool_mu, NULL);
    double t0 = now_us();
    srv.grammar = pulsar_grammar_compiler_new(ptrs.data(), lens.data(), width, width, stops.data(), (int)stops.size(),
                                              256LL << 20, err, sizeof err);
    if (!srv.grammar) {
        fprintf(stderr, "grammar_bench: compiler: %s\n", err);
        return 1;
    }
    srv.token_bytes = new std::vector<std::string>(bytes);
    printf("%s: %d tokens, logits width %d, %zu stop ids; vocabulary built in %.0f ms\n", family.c_str(), n_tok, width,
           stops.size(), (now_us() - t0) / 1e3);

    struct bcase {
        const char *name;
        std::string body;
        std::string text;   /* the turn's bytes after the prompt */
    };
    const std::string user = "{\"messages\":[{\"role\":\"user\",\"content\":\"find the gate docs\"}],";
    std::vector<bcase> cases = {
        {"strict tool call", user + TOOLS + "}", think_close + call_text},
        {"forced (required)", user + TOOLS + ",\"tool_choice\":\"required\"}", ""},
        {"response_format", user + RF + "}", think_close + k_answer},
    };
    int findings = 0;
    for (bcase &bc : cases) {
        for (int pass = 0; pass < 2; pass++) {
            job j;
            memset(&j, 0, sizeof j);
            request *r = &j.req;
            request_init(r, REQ_CHAT, 4096);
            chat_conversation c;
            memset(&c, 0, sizeof c);
            t0 = now_us();
            const bool ok = parse_chat_conversation_openai(bc.body.c_str(), &c, r, err, sizeof err) &&
                            render_chat_conversation(NULL, fmt, &srv, &c, r, err, sizeof err);
            const double compile_ms = (now_us() - t0) / 1e3;
            chat_conversation_free(&c);
            if (!ok || !r->constrain) {
                printf("  %s: refused: %s\n", bc.name, err);
                findings++;
                request_free(r);
                break;
            }
            if (pass == 0) {
                printf("  %s: plan (render + compile) cold %.1f ms", bc.name, compile_ms);
                request_free(r);
                continue;
            }
            printf(", cached %.1f ms\n", compile_ms);
            gen_state g;
            memset(&g, 0, sizeof g);
            g.j = &j;
            g.thinking = thinking_state_from_prompt(r);
            std::string text = bc.text;
            if (r->force_tool_call) {
                r->family->forced_call_seed(r, &g.text);
                /* the seed's call, finished: the template's bytes past the seed */
                const std::string seed(g.text.ptr, g.text.len);
                const size_t at = call_text.find(fmt == PULSAR_CHAT_DS4_V41 ? PULSAR_INVOKE_START : "<function");
                const size_t sat = seed.find(fmt == PULSAR_CHAT_DS4_V41 ? PULSAR_INVOKE_START : "<function");
                text = call_text.substr(at + (seed.size() - sat));
            }
            if (!gen_constrain_begin(&srv, &g)) {
                printf("    begin failed: %s\n", g.err);
                findings++;
                request_free(r);
                continue;
            }
            std::vector<int> ids;
            if (!encode(text, &ids)) {
                printf("    encode failed: %s\n", err);
                findings++;
            }
            const uint32_t W = gen_constrain_words(&srv);
            std::vector<uint32_t> row(W);
            stats fill, accept, round, allowed_frac;
            size_t live_tokens = 0;
            for (size_t i = 0; i < ids.size(); i++) {
                const int t = ids[i];
                if (gen_constrain_live(&g)) {
                    live_tokens++;
                    double a = now_us();
                    if (!gen_constrain_mask(&srv, &g, row.data())) {
                        printf("    mask failed: %s\n", g.err);
                        findings++;
                        break;
                    }
                    fill.add(now_us() - a);
                    size_t on = 0;
                    for (uint32_t w = 0; w < W; w++) on += (size_t)__builtin_popcount(row[w]);
                    allowed_frac.add(100.0 * (double)on / (double)width);
                    if (!((row[(uint32_t)t >> 5] >> ((uint32_t)t & 31u)) & 1u)) {
                        printf("    FINDING: token %zu (%d \"%s\") is refused by the mask\n", i, t, bytes[(size_t)t].c_str());
                        findings++;
                        break;
                    }
                    /* a speculative round from here: this token as the base, the next K as drafts */
                    const int K = 8;
                    const int nt = (int)std::min(ids.size() - i, (size_t)K + 1);
                    const uint32_t *masks = NULL;
                    a = now_us();
                    const int rows = gen_constrain_round(&srv, &g, ids.data() + i, nt, false, &masks);
                    round.add(now_us() - a);
                    if (rows < 0) {
                        printf("    round failed: %s\n", g.err);
                        findings++;
                        break;
                    }
                }
                const double a = now_us();
                if (!gen_constrain_accept(&srv, &g, t)) {
                    printf("    accept failed at token %zu: %s\n", i, g.err);
                    findings++;
                    break;
                }
                accept.add(now_us() - a);
            }
            printf("    %zu tokens, %zu in a live region%s\n", ids.size(), live_tokens,
                   gen_constrain_live(&g) ? " (still live at the end)" : "");
            fill.print("mask fill");
            accept.print("accept");
            round.print("round K=8");
            if (!allowed_frac.v.empty()) {
                std::sort(allowed_frac.v.begin(), allowed_frac.v.end());
                printf("    allowed tokens per live row: p50 %.3f%%, max %.3f%%\n",
                       allowed_frac.v[allowed_frac.v.size() / 2], allowed_frac.v.back());
            }
            gen_constrain_end(&g);
            buf_free(&g.text);
            request_free(r);
        }
    }
    pulsar_grammar_compiler_free(srv.grammar);
    delete srv.token_bytes;
    if (tok) bpe_tokenizer_free(tok);
    printf("%s: %d finding(s)\n", family.c_str(), findings);
    return findings ? 1 : 0;
}

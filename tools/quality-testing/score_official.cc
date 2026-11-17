/** @file
 *  The official-continuation scorer: teacher-force a hosted model's continuations through a local container and
 *  report how much probability the container gives them, on any generative family. */
/* score_official -- for each case of a suite: render the user prompt with the loaded family's own chat renderer
 * (thinking off, no system message), prefill it, then feed the official continuation's tokens one at a time and read
 * the next-token row before each.  Nothing is sampled, so sampling parameters play no part.
 *
 * Like for like with antirez/ds4's gguf-tools/quality-testing/score_official.c (README.md here names the commit),
 * whose case directories are read as they are:
 *
 *   mean NLL            sum over EVERY continuation token of -log softmax(row)[token], divided by the token count
 *                       (token-weighted over the suite; no case is dropped).
 *   top-token agreement over the positions of API-ALIGNED cases only: the official top-1 is the top_logprobs entry
 *                       with the highest logprob, mapped to a token id by its bytes; a position counts when that
 *                       entry maps, and agrees when it is the local argmax.  Tied official maxima count once and
 *                       agree if any of them is the local argmax (ds4 since d6b516a); the rule before it -- the
 *                       FIRST LISTED entry, the one ds4's published Qwen baselines were measured with -- is
 *                       reported beside it.
 *   API-aligned         a case whose response parses, has as many logprob positions as the local tokenizer makes of
 *                       the continuation text, and whose every position's bytes are the local token's bytes.  A
 *                       case that is not aligned keeps its NLL and leaves every API figure (said per case).
 *   bytes -> token id   the one vocabulary token with those bytes that plain-text tokenization also yields alone
 *                       (so never a control token); bytes holding U+FFFD or NUL, or matching no token, do not map.
 *
 * One session serves every case: each case starts with pulsar_session_invalidate, so its prompt is a cold prefill
 * through pulsar_session_sync (the normal chunked path) and no row of an earlier case is reused; the position is
 * checked after the prefill and after the last continuation token.
 *
 * usage: score_official MODEL [--out OUT.tsv] [--ctx N] [--limit N] [--dry-run] [--dump-ids] SUITE...
 *        SUITE = a directory holding manifest.tsv (its rows are read inside that directory, wherever the manifest
 *                says they were collected), or a manifest file (its paths are opened as written).
 *        --dry-run opens the container inspect-only (no GPU), renders and tokenizes the cases and prints their token
 *                counts and API alignment; a family whose vocabulary an inspect-only open does not load (DeepSeek)
 *                is refused by name. */
#include "pulsar.h"
#include "pyjson.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include <string>
#include <unordered_map>
#include <vector>

/** One alternative of a position's top_logprobs. */
struct api_alt {
    std::string bytes;   ///< the token's bytes; empty when the API gave none
    double logprob;      ///< the API's logprob
    int tok;             ///< the local token id the bytes map to, or -1
};

/** One position of the official continuation's logprobs. */
struct api_pos {
    std::string bytes;           ///< the emitted token's bytes; empty when the API gave none
    double logprob;              ///< its logprob
    std::vector<api_alt> alts;   ///< top_logprobs, in the API's order
};

/** One case, loaded and tokenized. */
struct score_case {
    std::string id;             ///< the manifest's id
    pulsar_tokens prompt;       ///< the rendered prompt
    pulsar_tokens target;       ///< the official continuation
    std::vector<api_pos> ref;   ///< the API's positions (empty: none parsed)
    long api_prompt_tokens;     ///< usage.prompt_tokens, or -1
    bool have_api;              ///< the response parsed
    bool aligned;               ///< ref lines up with target, token for token
};

/** A suite's (or one case's) totals. */
struct score_totals {
    long cases;              ///< cases scored
    long tokens;             ///< continuation tokens scored
    long first_match;        ///< cases whose first continuation token is the local argmax
    long lcp;                ///< sum of the greedy longest common prefixes
    long target_top1;        ///< positions whose continuation token is the local argmax
    double nll;              ///< sum of -log p(continuation token)
    long aligned_cases;      ///< API-aligned cases
    long aligned_tokens;     ///< their continuation tokens
    long top1_count;         ///< positions whose official top-1 maps (the tied-maxima rule)
    long top1_match;         ///< of those, the local argmax
    long listed_count;       ///< positions whose first listed alternative maps (the first-listed rule)
    long listed_match;       ///< of those, the local argmax
    long topk_count;         ///< positions with alternatives
    long topk_hit;           ///< of those, the local argmax is one of the alternatives
    double p_api_top1;       ///< sum over top1_count of the local probability of the official top-1
    long target_lp_count;    ///< positions with an API logprob for the emitted token
    double target_abs_delta; ///< sum of |local logprob - API logprob| of the emitted token
};

static void die(const char *fmt, ...) __attribute__((format(printf, 1, 2), noreturn));
static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "score_official: ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

static std::string read_file(const std::string &path) {
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp) die("open %s: %s", path.c_str(), strerror(errno));
    std::string out;
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, fp)) > 0) out.append(buf, n);
    if (ferror(fp)) die("read %s failed", path.c_str());
    fclose(fp);
    return out;
}

static bool is_dir(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

/* "a/b/prompts/case_000.txt" -> "prompts/case_000.txt": where a collector's row sits inside its suite directory */
static std::string last_two(const std::string &path) {
    const size_t a = path.rfind('/');
    if (a == std::string::npos || a == 0) return path;
    const size_t b = path.rfind('/', a - 1);
    return b == std::string::npos ? path : path.substr(b + 1);
}

/* the last component of a path, trailing slashes aside */
static std::string base_name(const std::string &path) {
    size_t end = path.size();
    while (end > 1 && path[end - 1] == '/') end--;
    const size_t a = end ? path.rfind('/', end - 1) : std::string::npos;
    return a == std::string::npos ? path.substr(0, end) : path.substr(a + 1, end - a - 1);
}

static bool json_number(const pyjson_value *v, double *out) {
    if (!v) return false;
    if (v->kind == pyjson_value::FLOAT) *out = v->f;
    else if (v->kind == pyjson_value::INT) *out = strtod(v->num.c_str(), NULL);
    else return false;
    return true;
}

/* "bytes": [72, 101, ...] -> the bytes; left empty when absent, null or not bytes */
static void json_bytes(const pyjson_value *v, std::string *out) {
    out->clear();
    if (!v || v->kind != pyjson_value::ARR) return;
    for (const pyjson_value &b : v->a) {
        double x;
        if (!json_number(&b, &x) || x < 0 || x > 255) return out->clear();
        out->push_back((char)(unsigned char)x);
    }
}

/* choices[0].logprobs.content[] of a chat-completions response; false = no usable logprobs */
static bool parse_response(const std::string &json, std::vector<api_pos> *ref, long *prompt_tokens) {
    pyjson_value doc;
    char err[200];
    *prompt_tokens = -1;
    ref->clear();
    if (!pyjson_parse(json.data(), json.size(), &doc, err, sizeof err)) return false;
    double n;
    if (const pyjson_value *usage = doc.get("usage"))
        if (json_number(usage->get("prompt_tokens"), &n)) *prompt_tokens = (long)n;
    const pyjson_value *choices = doc.get("choices");
    if (!choices || choices->kind != pyjson_value::ARR || choices->a.empty()) return false;
    const pyjson_value *lp = choices->a[0].get("logprobs");
    const pyjson_value *content = lp ? lp->get("content") : NULL;
    if (!content || content->kind != pyjson_value::ARR || content->a.empty()) return false;
    for (const pyjson_value &p : content->a) {
        api_pos pos;
        if (!json_number(p.get("logprob"), &pos.logprob) || !isfinite(pos.logprob)) return ref->clear(), false;
        json_bytes(p.get("bytes"), &pos.bytes);
        const pyjson_value *top = p.get("top_logprobs");
        if (top && top->kind == pyjson_value::ARR) {
            for (const pyjson_value &a : top->a) {
                api_alt alt;
                alt.tok = -1;
                if (!json_number(a.get("logprob"), &alt.logprob) || !isfinite(alt.logprob))
                    return ref->clear(), false;
                json_bytes(a.get("bytes"), &alt.bytes);
                pos.alts.push_back(alt);
            }
        }
        ref->push_back(pos);
    }
    return true;
}

static std::string token_bytes(pulsar_engine *e, TokId id) {
    size_t len = 0;
    char *text = pulsar_token_text(e, id, &len);
    if (!text) die("the loaded family has no tokenizer");
    std::string out(text, len);
    free(text);
    return out;
}

static bool valid_utf8(const std::string &s) {
    static const uint32_t lowest[4] = {0, 0x80, 0x800, 0x10000};
    const unsigned char *p = (const unsigned char *)s.data();
    const size_t n = s.size();
    for (size_t i = 0; i < n;) {
        const unsigned char c = p[i];
        const size_t more = c < 0x80 ? 0 : (c & 0xe0) == 0xc0 ? 1 : (c & 0xf0) == 0xe0 ? 2 : (c & 0xf8) == 0xf0 ? 3 : 4;
        if (more == 4 || i + more >= n) return false;
        uint32_t cp = more == 0 ? c : more == 1 ? c & 0x1f : more == 2 ? c & 0x0f : c & 0x07;
        for (size_t k = 1; k <= more; k++) {
            if ((p[i + k] & 0xc0) != 0x80) return false;
            cp = (cp << 6) | (p[i + k] & 0x3f);
        }
        if (cp < lowest[more] || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) return false;
        i += more + 1;
    }
    return true;
}

/** API bytes -> the local token id (the header's rule), remembered per byte string. */
struct byte_map {
    pulsar_engine *e;                               ///< the engine whose vocabulary this is
    std::unordered_map<std::string, int> vocab;     ///< token bytes -> id; -1 when two tokens share the bytes
    std::unordered_map<std::string, int> resolved;  ///< answers given so far

    /** Read the vocabulary's bytes once. @param engine the engine */
    explicit byte_map(pulsar_engine *engine) : e(engine) {
        const int n = pulsar_engine_vocab_size(e);
        vocab.reserve((size_t)n * 2);
        for (int i = 0; i < n; i++) {
            const std::string b = token_bytes(e, TokId(i));
            if (b.empty()) continue;
            auto ins = vocab.emplace(b, i);
            if (!ins.second) ins.first->second = -1;
        }
    }

    /** @param bytes one API token's bytes  @return the id, or -1 when they do not name one plain token */
    int id_of(const std::string &bytes) {
        auto hit = resolved.find(bytes);
        if (hit != resolved.end()) return hit->second;
        int id = -1;
        const bool damaged = bytes.empty() || bytes.find('\0') != std::string::npos ||
                             bytes.find("\xef\xbf\xbd") != std::string::npos;
        if (!damaged) {
            auto v = vocab.find(bytes);
            if (v != vocab.end()) id = v->second;
            /* a control token's spelling is valid UTF-8 and plain text never tokenizes to it: only the token the
             * text tokenizer yields alone stands.  Bytes that are not UTF-8 (half a character) can only be a plain
             * byte-level token, which the tokenizer entry would refuse to take as text. */
            if (id >= 0 && valid_utf8(bytes)) {
                pulsar_tokens tv = {};
                if (!pulsar_tokenize_text(e, bytes.c_str(), &tv) || tv.len != 1 || tv.v[0] != TokId(id)) id = -1;
                pulsar_tokens_free(&tv);
            }
        }
        resolved.emplace(bytes, id);
        return id;
    }
};

/* the manifest's rows -> cases, rendered and tokenized; API positions mapped and checked against the target */
static void load_suite(pulsar_engine *e, byte_map *map, const std::string &arg, int limit, std::string *name,
                       std::vector<score_case> *cases) {
    const bool rooted = is_dir(arg);
    const std::string manifest_path = rooted ? arg + "/manifest.tsv" : arg;
    const size_t slash = manifest_path.rfind('/');
    *name = rooted ? base_name(arg) : slash == std::string::npos ? "." : base_name(manifest_path.substr(0, slash));
    const std::string manifest = read_file(manifest_path);
    for (size_t at = 0; at < manifest.size() && (limit == 0 || (int)cases->size() < limit);) {
        size_t eol = manifest.find('\n', at);
        if (eol == std::string::npos) eol = manifest.size();
        std::string line = manifest.substr(at, eol - at);
        at = eol + 1;
        while (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> col;
        for (size_t c = 0; c <= line.size();) {
            size_t tab = line.find('\t', c);
            if (tab == std::string::npos) tab = line.size();
            col.push_back(line.substr(c, tab - c));
            c = tab + 1;
        }
        if (col.size() < 3) die("%s: a row needs id, prompt file, continuation file", manifest_path.c_str());
        auto path = [&](const std::string &p) { return rooted ? arg + "/" + last_two(p) : p; };

        score_case c = {};
        c.id = col[0];
        c.api_prompt_tokens = -1;
        const std::string prompt_text = read_file(path(col[1]));
        const std::string target_text = read_file(path(col[2]));
        if (!pulsar_encode_chat_prompt(e, NULL, prompt_text.c_str(), PULSAR_THINK_NONE, &c.prompt))
            die("%s: the family's chat renderer refused the prompt", c.id.c_str());
        if (!pulsar_tokenize_text(e, target_text.c_str(), &c.target))
            die("%s: the tokenizer refused the continuation", c.id.c_str());
        if (c.target.len == 0) die("%s: empty continuation", c.id.c_str());

        if (col.size() > 3 && !col[3].empty()) {
            c.have_api = parse_response(read_file(path(col[3])), &c.ref, &c.api_prompt_tokens);
            if (!c.have_api) {
                fprintf(stderr, "%s: no API logprobs parsed; API agreement skipped\n", c.id.c_str());
            } else if ((int)c.ref.size() != c.target.len) {
                fprintf(stderr, "%s: API token count %zu != local continuation tokens %d; API agreement skipped\n",
                        c.id.c_str(), c.ref.size(), c.target.len);
            } else {
                c.aligned = true;
                for (int i = 0; i < c.target.len && c.aligned; i++)
                    c.aligned = !c.ref[i].bytes.empty() && c.ref[i].bytes == token_bytes(e, c.target.v[i]);
                if (!c.aligned)
                    fprintf(stderr, "%s: API token bytes do not match the local token boundaries; API agreement "
                                    "skipped\n", c.id.c_str());
            }
            if (c.aligned)
                for (api_pos &p : c.ref)
                    for (api_alt &a : p.alts) a.tok = map->id_of(a.bytes);
        }
        cases->push_back(c);
    }
    if (cases->empty()) die("%s: no cases", manifest_path.c_str());
}

static void dump_ids(pulsar_engine *e, const score_case &c) {
    fprintf(stderr, "%s prompt ids (%d):", c.id.c_str(), c.prompt.len);
    for (int i = 0; i < c.prompt.len; i++) fprintf(stderr, " %d", c.prompt.v[i].raw());
    fprintf(stderr, "\n%s target ids (%d):", c.id.c_str(), c.target.len);
    for (int i = 0; i < c.target.len; i++) fprintf(stderr, " %d", c.target.v[i].raw());
    std::string text;
    for (int i = 0; i < c.prompt.len; i++) text += token_bytes(e, c.prompt.v[i]);
    fprintf(stderr, "\n%s prompt text: [[%s]]\n", c.id.c_str(), text.c_str());
}

/* log sum exp of the row and its argmax (the lowest id wins a tie); false when the row is unusable */
static bool row_stats(const float *row, int width, double *logsum, int *argmax) {
    int best = -1;
    for (int i = 0; i < width; i++) {
        if (isnan(row[i]) || row[i] == INFINITY) return false;
        if (row[i] != -INFINITY && (best < 0 || row[i] > row[best])) best = i;
    }
    if (best < 0) return false;
    double sum = 0.0;
    for (int i = 0; i < width; i++) sum += exp((double)row[i] - (double)row[best]);   /* exp(-inf) = 0 */
    *logsum = (double)row[best] + log(sum);
    *argmax = best;
    return true;
}

static double row_logprob(const float *row, int width, int tok, double logsum) {
    return tok >= 0 && tok < width ? (double)row[tok] - logsum : -INFINITY;
}

/* one position's API figures, given the local row */
static void score_api_pos(const api_pos &ap, const float *row, int width, double logsum, int greedy, double target_lp,
                          score_totals *t) {
    t->target_lp_count++;
    t->target_abs_delta += fabs(target_lp - ap.logprob);
    if (ap.alts.empty()) return;

    double best = -INFINITY;
    for (const api_alt &a : ap.alts) best = a.logprob > best ? a.logprob : best;
    int ref = -1;
    bool in_topk = false;
    for (const api_alt &a : ap.alts) {
        if (a.tok == greedy) in_topk = true;
        if (a.logprob == best && a.tok >= 0 && (ref < 0 || a.tok == greedy)) ref = a.tok;
    }
    if (ref >= 0) {
        t->top1_count++;
        t->top1_match += ref == greedy;
        t->p_api_top1 += exp(row_logprob(row, width, ref, logsum));
    }
    if (ap.alts[0].tok >= 0) {
        t->listed_count++;
        t->listed_match += ap.alts[0].tok == greedy;
    }
    t->topk_count++;
    t->topk_hit += in_topk;
}

static void add_totals(score_totals *d, const score_totals &s) {
    d->cases += s.cases, d->tokens += s.tokens, d->first_match += s.first_match, d->lcp += s.lcp;
    d->target_top1 += s.target_top1, d->nll += s.nll;
    d->aligned_cases += s.aligned_cases, d->aligned_tokens += s.aligned_tokens;
    d->top1_count += s.top1_count, d->top1_match += s.top1_match;
    d->listed_count += s.listed_count, d->listed_match += s.listed_match;
    d->topk_count += s.topk_count, d->topk_hit += s.topk_hit;
    d->p_api_top1 += s.p_api_top1;
    d->target_lp_count += s.target_lp_count, d->target_abs_delta += s.target_abs_delta;
}

static double ratio(double num, double den) { return den > 0 ? num / den : 0.0; }

/* prefill the case cold, then teacher-force its continuation; the case's figures in *t */
static void score_one(pulsar_session *s, const score_case &c, float *row, int width, score_totals *t) {
    char err[320] = "";
    pulsar_session_invalidate(s);
    if (pulsar_session_sync(s, &c.prompt, err, sizeof err) != 0) die("%s: sync failed: %s", c.id.c_str(), err);
    if (pulsar_session_pos(s) != TokPos(c.prompt.len)) die("%s: the prefill did not end at the prompt", c.id.c_str());

    bool matching = true;
    for (int i = 0; i < c.target.len; i++) {
        const int tok = c.target.v[i].raw();
        double logsum = 0.0;
        int greedy = -1;
        if (pulsar_session_copy_logits(s, row, width) != width || !row_stats(row, width, &logsum, &greedy))
            die("%s: no usable logits row at continuation token %d", c.id.c_str(), i);
        const double lp = row_logprob(row, width, tok, logsum);
        if (!isfinite(lp)) die("%s: continuation token %d (id %d) has no probability", c.id.c_str(), i, tok);
        t->nll += -lp;
        t->target_top1 += greedy == tok;
        if (i == 0) t->first_match += greedy == tok;
        if (matching && greedy == tok) t->lcp++;
        else matching = false;
        if (c.aligned) score_api_pos(c.ref[i], row, width, logsum, greedy, lp, t);
        if (pulsar_session_eval(s, c.target.v[i], err, sizeof err) != 0)
            die("%s: eval failed at continuation token %d: %s", c.id.c_str(), i, err);
    }
    if (pulsar_session_pos(s) != TokPos(c.prompt.len + c.target.len))
        die("%s: the session did not end at prompt + continuation", c.id.c_str());
    t->cases = 1;
    t->tokens = c.target.len;
    t->aligned_cases = c.aligned;
    t->aligned_tokens = c.aligned ? c.target.len : 0;
}

static void print_summary(const std::string &name, const score_totals &t, double seconds) {
    printf("suite %s: cases=%ld tokens=%ld mean_nll=%.6f greedy_is_continuation=%.2f%% first_match=%ld "
           "mean_greedy_lcp=%.2f seconds=%.1f\n",
           name.c_str(), t.cases, t.tokens, ratio(t.nll, t.tokens), 100.0 * ratio(t.target_top1, t.tokens),
           t.first_match, ratio(t.lcp, t.cases), seconds);
    if (!t.aligned_cases) {
        printf("suite %s: no API-aligned case: no top-token agreement\n", name.c_str());
        return;
    }
    printf("suite %s: api_aligned_cases=%ld api_aligned_tokens=%ld top_token_agreement=%.2f%% (%ld/%ld) "
           "first_listed_rule=%.2f%% (%ld/%ld) api_target_mae=%.6f\n",
           name.c_str(), t.aligned_cases, t.aligned_tokens, 100.0 * ratio(t.top1_match, t.top1_count), t.top1_match,
           t.top1_count, 100.0 * ratio(t.listed_match, t.listed_count), t.listed_match, t.listed_count,
           ratio(t.target_abs_delta, t.target_lp_count));
    if (t.topk_count)
        printf("suite %s: mean_p_on_api_top1=%.4f greedy_in_api_topk=%.2f%% (%ld/%ld)\n", name.c_str(),
               ratio(t.p_api_top1, t.top1_count), 100.0 * ratio(t.topk_hit, t.topk_count), t.topk_hit,
               t.topk_count);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int positive_int(const char *s, const char *opt) {
    char *end = NULL;
    const long v = strtol(s, &end, 10);
    if (end == s || *end || v <= 0 || v > (1L << 30)) die("%s takes a positive integer", opt);
    return (int)v;
}

int main(int argc, char **argv) {
    const char *model = NULL, *out_path = NULL;
    int ctx = 0, limit = 0;
    bool dry_run = false, dump = false, bad = false;
    std::vector<std::string> suite_args;
    for (int i = 1; i < argc && !bad; i++) {
        const char *a = argv[i];
        const bool has_value = i + 1 < argc;
        if (!strcmp(a, "--dry-run")) dry_run = true;
        else if (!strcmp(a, "--dump-ids")) dump = true;
        else if (!strcmp(a, "--out") && has_value) out_path = argv[++i];
        else if (!strcmp(a, "--ctx") && has_value) ctx = positive_int(argv[++i], a);
        else if (!strcmp(a, "--limit") && has_value) limit = positive_int(argv[++i], a);
        else if (a[0] == '-') bad = true;
        else if (!model) model = a;
        else suite_args.push_back(a);
    }
    if (bad || !model || suite_args.empty()) {
        fprintf(stderr, "usage: %s MODEL [--out OUT.tsv] [--ctx N] [--limit N] [--dry-run] [--dump-ids] SUITE...\n"
                        "  SUITE      a directory holding manifest.tsv, or a manifest file\n"
                        "  --limit N  the first N cases of each suite\n"
                        "  --dry-run  load, render and tokenize the cases and print their token counts: no GPU\n"
                        "  --dump-ids each case's prompt and continuation ids, and the prompt's text, on stderr\n",
                argv[0]);
        return 2;
    }

    pulsar_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = model;
    opt.backend = PULSAR_BACKEND_CUDA;
    opt.inspect_only = dry_run;
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0 || !e) die("failed to open %s", model);
    char why[200] = "";
    if (!pulsar_engine_has_tokenizer(e)) die("%s has no tokenizer in this build", pulsar_engine_family_name(e));
    if (pulsar_engine_vocab_size(e) <= 0)
        die("%s: no vocabulary is loaded%s", pulsar_engine_family_name(e),
            dry_run ? " by an inspect-only open (--dry-run): this family tokenizes only on a full open" : "");
    if (!pulsar_engine_think_mode_supported(e, PULSAR_THINK_NONE, why, sizeof why))
        die("%s cannot render a prompt with thinking off: %s", pulsar_engine_family_name(e), why);

    byte_map map(e);
    std::vector<std::string> names(suite_args.size());
    std::vector<std::vector<score_case>> suites(suite_args.size());
    int longest = 0;
    for (size_t k = 0; k < suite_args.size(); k++) {
        load_suite(e, &map, suite_args[k], limit, &names[k], &suites[k]);
        long prompt_tokens = 0, target_tokens = 0, aligned = 0, aligned_tokens = 0, api_counts = 0, api_same = 0;
        for (const score_case &c : suites[k]) {
            if (dump) dump_ids(e, c);
            if (dry_run)
                printf("%s/%s\tprompt=%d\ttarget=%d\tapi_prompt=%ld\tapi_positions=%zu\taligned=%d\n",
                       names[k].c_str(), c.id.c_str(), c.prompt.len, c.target.len, c.api_prompt_tokens, c.ref.size(),
                       c.aligned ? 1 : 0);
            prompt_tokens += c.prompt.len, target_tokens += c.target.len;
            aligned += c.aligned, aligned_tokens += c.aligned ? c.target.len : 0;
            api_counts += c.api_prompt_tokens >= 0, api_same += c.api_prompt_tokens == c.prompt.len;
            if (c.prompt.len + c.target.len > longest) longest = c.prompt.len + c.target.len;
        }
        printf("loaded %s: family=%s cases=%zu prompt_tokens=%ld target_tokens=%ld api_aligned_cases=%ld "
               "api_aligned_tokens=%ld prompt_count_is_api=%ld/%ld\n",
               names[k].c_str(), pulsar_engine_family_name(e), suites[k].size(), prompt_tokens, target_tokens,
               aligned, aligned_tokens, api_same, api_counts);
    }
    fflush(stdout);

    int rc = 0;
    if (!dry_run) {
        /* one session for every case, as deep as the longest; never shallower than the old default */
        const int need = longest + 2;
        if (ctx == 0) ctx = need > 4096 ? (need + 1023) / 1024 * 1024 : 4096;
        if (need > ctx) die("the longest case needs --ctx %d", need);
        pulsar_session *s = NULL;
        if (pulsar_session_create(&s, e, ctx) != 0) die("failed to create a %d-token session", ctx);
        const int width = pulsar_engine_logits_width(e);
        std::vector<float> row((size_t)width);
        FILE *out = out_path ? fopen(out_path, "wb") : NULL;
        if (out_path && !out) die("open %s: %s", out_path, strerror(errno));
        if (out)
            fprintf(out, "id\tprompt_tokens\ttarget_tokens\tnll\tavg_nll\tfirst_match\tgreedy_lcp\ttarget_top1"
                         "\tapi_aligned\tapi_top1_count\tapi_top1_match\tapi_listed_count\tapi_listed_match"
                         "\tapi_topk_count\tapi_topk_hit\tp_api_top1_sum\tapi_target_mae\n");
        std::vector<score_totals> totals(suites.size());
        std::vector<double> seconds(suites.size());
        for (size_t k = 0; k < suites.size(); k++) {
            const double t0 = now_s();
            totals[k] = {};
            for (const score_case &c : suites[k]) {
                score_totals t = {};
                score_one(s, c, row.data(), width, &t);
                add_totals(&totals[k], t);
                fprintf(stderr, "%s/%s prompt=%d target=%d avg_nll=%.6f lcp=%ld api_top1=%ld/%ld\n",
                        names[k].c_str(), c.id.c_str(), c.prompt.len, c.target.len, ratio(t.nll, t.tokens), t.lcp,
                        t.top1_match, t.top1_count);
                if (out) {
                    fprintf(out, "%s/%s\t%d\t%d\t%.9f\t%.9f\t%ld\t%ld\t%ld\t%d\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%.9f"
                                 "\t%.9f\n",
                            names[k].c_str(), c.id.c_str(), c.prompt.len, c.target.len, t.nll,
                            ratio(t.nll, t.tokens), t.first_match, t.lcp, t.target_top1, c.aligned ? 1 : 0,
                            t.top1_count, t.top1_match, t.listed_count, t.listed_match, t.topk_count, t.topk_hit,
                            t.p_api_top1, ratio(t.target_abs_delta, t.target_lp_count));
                    fflush(out);
                }
            }
            seconds[k] = now_s() - t0;
        }
        for (size_t k = 0; k < suites.size(); k++) print_summary(names[k], totals[k], seconds[k]);
        if (out && fclose(out) != 0) rc = 1;
        pulsar_session_free(s);
    }
    for (std::vector<score_case> &cases : suites)
        for (score_case &c : cases) {
            pulsar_tokens_free(&c.prompt);
            pulsar_tokens_free(&c.target);
        }
    pulsar_engine_close(e);
    return rc;
}

/* inkling_tokenizer.cpp -- the Inkling family's tokenizer table (family.h pulsar_family_tokenizer; L294 E6's first
 * piece, on L295 F5's frontend kit).
 *
 * The shared BPE engine (src/lib/bpe_tokenizer.h) in its tiktoken configuration -- the checkpoint's rank file, the
 * registered "o200k" pattern, tokenizer_config.json's added tokens, config.json's eos_token_id -- graded byte-exact
 * against HF tokenizers by inkling-chat-gate (F5).  The container carries the three files beside its shards, as a
 * Qwen container carries tokenizer.json (the builder writes no side files): tokenizer.model is the checkpoint's
 * tiktoken/tokenizer.model.  A container without them does not load.
 *
 * The chat front (chat_open / chat_turn / chat_end_assistant, the CLI's turn-by-turn render) is E6's and is not
 * wired: the family declares no PULSAR_FAMILY_CAP_CHAT, so tokenizer.cpp refuses those entries by name before they
 * reach this table, and the front ends refuse the engine at startup.  What the table serves is token ids and text
 * both ways -- pulsar_tokenize_text, pulsar_tokenize_rendered_chat_spans over F5's inkling_chat_render, the stop ids,
 * a token's bytes. */
#include "inkling.h"
#include "lib/bpe_tokenizer.h"

#include <string.h>
#include <sys/stat.h>

#include <string>
#include <vector>

namespace {

const char *const ARCH = PULSAR_INKLING_ARCH;

bpe_tokenizer *TOK(const pulsar_engine *e) { return ((const pulsar_inkling_model *)e->family_state)->tok; }

bool read_text(const std::string &path, std::string *out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char buf[1 << 16];
    size_t n;
    out->clear();
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out->append(buf, n);
    const bool ok = !ferror(f);
    fclose(f);
    return ok;
}

/* `spans` are the client-data ranges where no added token may match.  The entries have no error return; an encode
 * refusal (text that is not UTF-8, or spans out of order -- a renderer defect) is said by name and yields no tokens,
 * never a different tokenization. */
void encode_into(const pulsar_engine *e, const char *text, const pulsar_text_span *spans, uint32_t n_spans,
                 pulsar_tokens *out) {
    std::vector<int> ids;
    char err[256] = "";
    const size_t len = text ? strlen(text) : 0;
    if (!bpe_tokenizer_encode(TOK(e), text ? text : "", len, spans, n_spans, &ids, err, sizeof(err))) {
        fprintf(stderr, "pulsar: %s tokenizer refused %zu bytes: %s\n", ARCH, len, err);
        return;
    }
    for (int id : ids) pulsar_tokens_push(out, id);
}

/* raw text: all of it is client data, so no added token matches */
void encode_text(pulsar_engine *e, const char *text, pulsar_tokens *out) {
    const pulsar_text_span all = {0u, (uint32_t)(text ? strlen(text) : 0)};
    encode_into(e, text, all.hi ? &all : NULL, all.hi ? 1u : 0u, out);
}

void encode_rendered(pulsar_engine *e, const char *text, const pulsar_text_span *spans, uint32_t n_spans,
                     pulsar_tokens *out) {
    encode_into(e, text, spans, n_spans, out);
}

/* the chat front: tokenizer.cpp's chat_require refuses before these are reached (no CAP_CHAT); a caller that gets
 * here went around it */
[[noreturn]] void no_chat_front(const char *op) {
    char msg[256];
    snprintf(msg, sizeof(msg), "inkling: %s: the turn-by-turn chat front is not wired (E6); render with "
             "inkling_chat_render and tokenize with pulsar_tokenize_rendered_chat_spans", op);
    pulsar_die(msg);
    exit(1);   /* unreachable: pulsar_die exits */
}
void chat_open(pulsar_engine *, pulsar_tokens *, const char *, const char *, pulsar_think_mode) {
    no_chat_front("chat_open");
}
void chat_turn(pulsar_engine *, pulsar_tokens *, const pulsar_chat_message *, int, bool, pulsar_think_mode) {
    no_chat_front("chat_turn");
}
void chat_end_assistant(pulsar_engine *, pulsar_tokens *) { no_chat_front("chat_end_assistant"); }

bool is_stop(pulsar_engine *e, int token) {
    for (int id : bpe_tokenizer_stop_ids(TOK(e))) if (id == token) return true;
    return false;
}
/* config.json's eos_token_id (Inkling-Small: <|content_model_end_sampling|>, the end of a sampled turn) */
int eos(pulsar_engine *e) { return bpe_tokenizer_stop_ids(TOK(e)).front(); }

char *token_text(pulsar_engine *e, int token, size_t *len) {
    size_t n = 0;
    const char *b = bpe_tokenizer_token_bytes(TOK(e), token, &n);
    if (!b) n = 0;   /* out of range, e.g. the padded logits rows past the table: no text */
    char *out = (char *)xmalloc(n + 1);
    if (n) memcpy(out, b, n);
    out[n] = '\0';
    if (len) *len = n;
    return out;
}

/* Inkling's turns have no think-close token: reasoning is a typed content block (F5's output grammar) */
int think_close(pulsar_engine *) { return -1; }

/* the server's prefix anchors: E6 with the server table; unknown until then, said once */
bool turn_markers(pulsar_engine *e, pulsar_turn_markers *) {
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "pulsar: %s: the chat turn markers are E6's (the server table): unknown\n", e->family->name);
    }
    return false;
}

void dump(pulsar_engine *e, FILE *fp, const pulsar_tokens *tokens) {
    fprintf(fp, "[");
    for (int i = 0; i < tokens->len; i++) fprintf(fp, "%s%d", i ? ", " : "", tokens->v[i]);
    fprintf(fp, "]\n");
    for (int i = 0; i < tokens->len; i++) {
        size_t n = 0;
        char *piece = token_text(e, tokens->v[i], &n);
        fprintf(fp, "%6d  ", tokens->v[i]);
        pulsar_dump_piece_quoted(fp, piece, n);
        fputc('\n', fp);
        free(piece);
    }
}

int n_tokens(pulsar_engine *e) { return bpe_tokenizer_n_tokens(TOK(e)); }

}  // namespace

const pulsar_family_tokenizer PULSAR_INKLING_TOKENIZER = {
    /* .encode_text              = */ encode_text,
    /* .encode_rendered          = */ encode_rendered,
    /* .chat_open                = */ chat_open,
    /* .chat_turn                = */ chat_turn,
    /* .chat_end_assistant       = */ chat_end_assistant,
    /* .is_stop                  = */ is_stop,
    /* .eos                      = */ eos,
    /* .token_text               = */ token_text,
    /* .think_close              = */ think_close,
    /* .turn_markers             = */ turn_markers,
    /* .dump                     = */ dump,
    /* .n_tokens                 = */ n_tokens,
};

bool pulsar_inkling_load_tokenizer(pulsar_inkling_model *w, const char *model_path) {
    std::string dir = model_path ? model_path : "";
    struct stat st;
    if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        const size_t slash = dir.find_last_of('/');
        dir = slash == std::string::npos ? "." : dir.substr(0, slash);
    }
    std::string ranks, tcfg, cfg;
    const std::string rp = dir + "/tokenizer.model", tp = dir + "/tokenizer_config.json", cp = dir + "/config.json";
    if (!read_text(rp, &ranks) || !read_text(tp, &tcfg) || !read_text(cp, &cfg)) {
        fprintf(stderr, "pulsar: %s: the container carries no %s / %s / %s -- copy the checkpoint's "
                        "tiktoken/tokenizer.model (as tokenizer.model), tokenizer_config.json and config.json beside the "
                        "shards; refusing\n", ARCH, rp.c_str(), tp.c_str(), cp.c_str());
        return false;
    }
    char err[256] = "";
    w->tok = bpe_tokenizer_load_tiktoken(ranks.data(), ranks.size(), "o200k", tcfg.data(), tcfg.size(), cfg.data(),
                                         cfg.size(), err, sizeof(err));
    if (!w->tok) {
        fprintf(stderr, "pulsar: %s: tokenizer refused: %s\n", ARCH, err);
        return false;
    }
    const std::vector<int> &stop = bpe_tokenizer_stop_ids(w->tok);
    if (stop.empty() || (uint32_t)stop.front() != w->shape.eos_id) {
        fprintf(stderr, "pulsar: %s: the tokenizer's first stop id %d is not the container's eos %u -- refusing\n", ARCH,
                stop.empty() ? -1 : stop.front(), w->shape.eos_id);
        return false;
    }
    if ((uint32_t)bpe_tokenizer_n_tokens(w->tok) > w->shape.n_vocab_real) {
        fprintf(stderr, "pulsar: %s: the tokenizer's %d ids exceed the head's %u real rows -- refusing\n", ARCH,
                bpe_tokenizer_n_tokens(w->tok), w->shape.n_vocab_real);
        return false;
    }
    fprintf(stderr, "pulsar: %s: tokenizer %d ids (tiktoken o200k) from %s; stop ids", ARCH,
            bpe_tokenizer_n_tokens(w->tok), rp.c_str());
    for (int id : stop) fprintf(stderr, " %d", id);
    fprintf(stderr, "\n");
    return true;
}

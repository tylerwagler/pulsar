/* inkling_tokenizer.cpp -- the Inkling family's tokenizer table and chat front (family.h pulsar_family_tokenizer;
 * L294 E6, on L295 F5's frontend kit).
 *
 * The shared BPE engine (src/lib/bpe_tokenizer.h) in its tiktoken configuration -- the checkpoint's rank file, the
 * registered "o200k" pattern, tokenizer_config.json's added tokens, config.json's eos_token_id -- graded byte-exact
 * against HF tokenizers by inkling-chat-gate (F5).  The container carries the three files beside its shards
 * (tools/container/inkling.py writes them): tokenizer.model is the checkpoint's tiktoken/tokenizer.model.  A
 * container without them does not load.
 *
 * The chat front (pulsar_chat_open / _append_turn / _end_assistant: the CLI REPL and the agent) is F5's renderer,
 * src/lib/inkling_chat -- the head is the full render of the system message plus the effort line, a turn the
 * renderer's turn piece, the close its <|content_model_end_sampling|> -- so head + turns + sampled turns is the
 * conversation's full render (inkling-chat-gate S1 grades the pieces).  The server renders with the same library
 * behind its family table (src/server/family_inkling.cpp). */
/** @file inkling_tokenizer.cpp
 *  @brief Inkling's tokenizer table and chat front: the shared BPE engine (tiktoken o200k) and F5's renderer. */
#include "inkling.h"
#include "lib/bpe_tokenizer.h"
#include "lib/inkling_chat.h"

#include <string.h>
#include <sys/stat.h>

#include <string>
#include <vector>

namespace {

/// The architecture name refusals are prefixed with.
const char *const ARCH = PULSAR_INKLING_ARCH;

/** The loaded tokenizer of an Inkling engine (no family check: only this table's hooks call it).
 *  @param e  an engine this family loaded
 *  @return the BPE tokenizer pulsar_inkling_load_tokenizer built */
bpe_tokenizer *TOK(const pulsar_engine *e) { return ((const pulsar_inkling_model *)e->family_state)->tok; }

/** Read a whole file.
 *  @param path  the file
 *  @param out   receives its bytes (cleared first)
 *  @return false when the file cannot be opened or a read fails */
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
/** @param e        the engine
 *  @param text     the UTF-8 text (NULL = empty)
 *  @param len      text's length in bytes
 *  @param spans    the client-data byte ranges, ascending (NULL with n_spans 0 = none)
 *  @param n_spans  the number of spans
 *  @param out      the ids are appended here */
void encode_into(const pulsar_engine *e, const char *text, size_t len, const pulsar_text_span *spans, uint32_t n_spans,
                 pulsar_tokens *out) {
    std::vector<int> ids;
    char err[256] = "";
    if (!bpe_tokenizer_encode(TOK(e), text ? text : "", len, spans, n_spans, &ids, err, sizeof(err))) {
        fprintf(stderr, "pulsar: %s tokenizer refused %zu bytes: %s\n", ARCH, len, err);
        return;
    }
    for (int id : ids) pulsar_tokens_push(out, id);
}

/* raw text: all of it is client data, so no added token matches */
/** pulsar_family_tokenizer::encode_text.
 *  @param e     the engine
 *  @param text  NUL-terminated UTF-8 (NULL = empty)
 *  @param out   the ids are appended here */
void encode_text(pulsar_engine *e, const char *text, pulsar_tokens *out) {
    const pulsar_text_span all = {0u, (uint32_t)(text ? strlen(text) : 0)};
    encode_into(e, text, all.hi, all.hi ? &all : NULL, all.hi ? 1u : 0u, out);
}

/** pulsar_family_tokenizer::encode_rendered: added tokens match outside `spans`.
 *  @param e        the engine
 *  @param text     the rendered chat, NUL-terminated (NULL = empty)
 *  @param spans    the client-data byte ranges
 *  @param n_spans  the number of spans
 *  @param out      the ids are appended here */
void encode_rendered(pulsar_engine *e, const char *text, const pulsar_text_span *spans, uint32_t n_spans,
                     pulsar_tokens *out) {
    encode_into(e, text, text ? strlen(text) : 0, spans, n_spans, out);
}

/* ---- the chat front, turn by turn (pulsar.h pulsar_chat_open): inkling_chat's pieces, tokenized with their client
 * spans.  The entries have no error return, so a refusal (a think mode the template has no level for, a role it
 * cannot place) ends the process by name. */

/** End the process for a chat-front refusal (the entries have no error return).
 *  @param op   the public entry that refused
 *  @param err  the renderer's or the think-mode check's reason */
[[noreturn]] void chat_refuse(const char *op, const char *err) {
    fprintf(stderr, "pulsar: %s: %s: %s -- exiting\n", ARCH, op, err);
    exit(1);
}

/** Tokenize one rendered piece with its client spans and append it.
 *  @param e       the engine
 *  @param r       the renderer's text and spans
 *  @param tokens  the ids are appended here */
void chat_append(const pulsar_engine *e, const inkling_render_out &r, pulsar_tokens *tokens) {
    encode_into(e, r.text.data(), r.text.size(), r.spans.data(), (uint32_t)r.spans.size(), tokens);
}

/* The head: the system message (the trusted text -- another family's tool prompt, text here -- then the client's
 * system text) and the effort line, which the full render writes after the leading system messages: the template's
 * default effort with thinking on, 0 with it off (the renderer's own effort authority, inkling_chat.h). */
/** pulsar_family_tokenizer::chat_open.
 *  @param e           the engine
 *  @param tokens      the head's ids are appended here
 *  @param trusted     the trusted system prefix (NULL = none)
 *  @param system      the client's system text (NULL = none)
 *  @param think_mode  picks the effort line (the default effort, or none); an unsupported mode ends the process */
void chat_open(pulsar_engine *e, pulsar_tokens *tokens, const char *trusted, const char *system,
               pulsar_think_mode think_mode) {
    char err[256];
    if (!pulsar_engine_think_mode_supported(e, think_mode, err, sizeof err)) chat_refuse("pulsar_chat_open", err);
    const std::string content = std::string(trusted ? trusted : "") + (system ? system : "");
    inkling_msg_in sys = {};
    sys.role = "system";
    sys.content = content.c_str();
    inkling_render_in in = {};
    in.msgs = content.empty() ? NULL : &sys;
    in.n_msgs = content.empty() ? 0 : 1;
    in.effort = pulsar_think_mode_enabled(think_mode) ? inkling_effort_default() : inkling_effort_named("none");
    in.add_generation_prompt = false;
    inkling_render_out r;
    if (!inkling_chat_render(in, &r, err, sizeof err)) chat_refuse("pulsar_chat_open", err);
    chat_append(e, r, tokens);
}

/* One turn: the messages as the full render writes them after the head (a system note is a system block, as the
 * template places one anywhere), then the generation prompt.  A tool result has no call to name it here. */
/** pulsar_family_tokenizer::chat_turn.
 *  @param e                  the engine
 *  @param tokens             the turn's ids are appended here
 *  @param msgs               the turn's messages (a NULL role reads as "user")
 *  @param n                  the number of messages
 *  @param generation_prompt  append the assistant's opening after the messages
 *  @param think_mode         checked against the engine (an unsupported mode ends the process) */
void chat_turn(pulsar_engine *e, pulsar_tokens *tokens, const pulsar_chat_message *msgs, int n, bool generation_prompt,
               pulsar_think_mode think_mode) {
    char err[256];
    if (!pulsar_engine_think_mode_supported(e, think_mode, err, sizeof err))
        chat_refuse("pulsar_chat_append_turn", err);
    std::vector<inkling_msg_in> m((size_t)(n > 0 ? n : 0));
    for (int i = 0; i < n; i++) {
        m[(size_t)i] = {};
        m[(size_t)i].role = msgs[i].role ? msgs[i].role : "user";
        m[(size_t)i].content = msgs[i].content;
    }
    inkling_render_out r;
    if (!inkling_chat_render_turn(m.data(), n, 0, false, generation_prompt, &r, err, sizeof err))
        chat_refuse("pulsar_chat_append_turn", err);
    chat_append(e, r, tokens);
}

/* the sampled turn's close: its stop token, <|content_model_end_sampling|> (the renderer's close_turn piece) */
/** pulsar_family_tokenizer::chat_end_assistant.
 *  @param e       the engine
 *  @param tokens  the close's ids are appended here */
void chat_end_assistant(pulsar_engine *e, pulsar_tokens *tokens) {
    inkling_render_out r;
    char err[256];
    if (!inkling_chat_render_turn(NULL, 0, 0, true, false, &r, err, sizeof err))
        chat_refuse("pulsar_chat_end_assistant", err);
    chat_append(e, r, tokens);
}

/** pulsar_family_tokenizer::is_stop.
 *  @param e      the engine
 *  @param token  a token id
 *  @return true when it is one of the tokenizer's stop ids */
bool is_stop(pulsar_engine *e, int token) {
    for (int id : bpe_tokenizer_stop_ids(TOK(e))) if (id == token) return true;
    return false;
}
/* config.json's eos_token_id (Inkling-Small: <|content_model_end_sampling|>, the end of a sampled turn) */
/** pulsar_family_tokenizer::eos (load checked it equals the shape's eos_id).
 *  @param e  the engine
 *  @return the tokenizer's first stop id */
int eos(pulsar_engine *e) { return bpe_tokenizer_stop_ids(TOK(e)).front(); }

/** pulsar_family_tokenizer::token_text.
 *  @param e      the engine
 *  @param token  a token id
 *  @param len    receives the byte length (NULL = not wanted)
 *  @return the token's bytes, malloc'd and NUL-terminated (the caller frees); empty outside the table */
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
/** pulsar_family_tokenizer::think_close.
 *  @return -1: no think-close token */
int think_close(pulsar_engine *) { return -1; }

/* The renderer opens every user block with <|message_user|> and every assistant block with <|message_model|>, one
 * added token each; the anchors' rules read the FIRST assistant marker and the LAST user one, which a multi-block
 * turn does not move. */
/** pulsar_family_tokenizer::turn_markers.
 *  @param e    the engine
 *  @param out  receives one user and one assistant marker id
 *  @return false when either added token is missing (said once per process) */
bool turn_markers(pulsar_engine *e, pulsar_turn_markers *out) {
    const int user = bpe_tokenizer_added_id(TOK(e), "<|message_user|>");
    const int model = bpe_tokenizer_added_id(TOK(e), "<|message_model|>");
    if (user < 0 || model < 0) {
        static bool said = false;
        if (!said) {
            said = true;
            fprintf(stderr, "pulsar: %s: the tokenizer has no <|message_user|> (%d) / <|message_model|> (%d) added "
                            "token: the chat turn markers are unknown\n", e->family->name, user, model);
        }
        return false;
    }
    out->user[0] = user;
    out->n_user = 1;
    out->assistant[0] = model;
    out->n_assistant = 1;
    return true;
}

/** pulsar_family_tokenizer::dump (--dump-tokens).
 *  @param e       the engine
 *  @param fp      the stream written
 *  @param tokens  the ids to print */
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

/** pulsar_family_tokenizer::n_tokens.
 *  @param e  the engine
 *  @return the ids in the tokenizer's table (at most n_vocab_real; the head is wider) */
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

/** Load the tokenizer from tokenizer.model, tokenizer_config.json and config.json beside the shards, and check its
 *  first stop id is the shape's eos and its table fits the head's real rows.
 *  @param w           the family state: its shape is read; tok is written
 *  @param model_path  the container's directory, or a shard path inside it
 *  @return true = loaded; false = refused (said) */
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
        fprintf(stderr, "pulsar: %s: the container carries no %s / %s / %s -- the builder (tools/container/inkling.py) "
                        "writes the checkpoint's tiktoken/tokenizer.model (as tokenizer.model), tokenizer_config.json "
                        "and config.json beside the shards; refusing\n", ARCH, rp.c_str(), tp.c_str(), cp.c_str());
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

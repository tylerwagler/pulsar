/* nemotron_h_tokenizer.cpp -- the Nemotron-H family's tokenizer table and chat front (family.h
 * pulsar_family_tokenizer; L296 N8, on L295 F5's frontend kit).
 *
 * The shared BPE engine (src/lib/bpe_tokenizer.h) in its HF configuration -- the checkpoint's tokenizer.json (the
 * "tekken" pre-tokenizer, no normaliser, ignore_merges) and generation_config.json's eos_token_id list (<|im_end|>,
 * </s>) -- graded byte-exact against HF tokenizers by nemotron-chat-gate.  The container carries both files beside
 * its shards (tools/container/nemotron_h.py side_files).  A container without them does not load.
 *
 * The chat front (pulsar_chat_open / _append_turn / _end_assistant: the CLI REPL and the agent) is the renderer,
 * src/lib/nemotron_chat: the head is the system turn, a turn the renderer's turn piece, the close the "<|im_end|>\n"
 * a sampled turn's tail opens with.  The server renders with the same library behind its family table
 * (src/server/family_nemotron_h.cpp). */
/** @file nemotron_h_tokenizer.cpp
 *  @brief Nemotron-H's tokenizer table and ChatML chat front: the shared BPE engine (tekken) and src/lib/nemotron_chat. */
#include "nemotron_h.h"
#include "lib/bpe_tokenizer.h"
#include "lib/nemotron_chat.h"

#include <string.h>
#include <sys/stat.h>

#include <string>
#include <vector>

namespace {

/// The architecture name refusals are prefixed with.
const char *const ARCH = PULSAR_NEMOTRON_H_ARCH;

/** The loaded tokenizer of a Nemotron-H engine (no family check: only this table's hooks call it).
 *  @param e  an engine this family loaded
 *  @return the BPE tokenizer pulsar_nemotron_h_load_tokenizer built */
bpe_tokenizer *TOK(const pulsar_engine *e) { return ((const pulsar_nemotron_h_model *)e->family_state)->tok; }

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

/* `spans` are the client-data ranges where no added token may match.  An encode refusal (text that is not UTF-8, or
 * spans out of order -- a renderer defect) is said by name and yields no tokens, never a different tokenization. */
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

/* ---- the chat front, turn by turn: nemotron_chat's pieces, tokenized with their client spans.  The entries have
 * no error return, so a refusal (a think mode the template has no level for) ends the process by name. */

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
void chat_append(const pulsar_engine *e, const nemotron_render_out &r, pulsar_tokens *tokens) {
    encode_into(e, r.text.data(), r.text.size(), r.spans.data(), (uint32_t)r.spans.size(), tokens);
}

/* thinking on (the template's default effort) or off -- no other level reaches the turn-by-turn front */
/** @param e           the engine
 *  @param think_mode  the caller's mode; one the engine does not support ends the process
 *  @param op          the public entry, for the refusal
 *  @return the template's default effort with thinking on, NEMOTRON_EFFORT_NONE with it off */
nemotron_effort chat_effort(pulsar_engine *e, pulsar_think_mode think_mode, const char *op) {
    char err[256];
    if (!pulsar_engine_think_mode_supported(e, think_mode, err, sizeof err)) chat_refuse(op, err);
    return pulsar_think_mode_enabled(think_mode) ? nemotron_effort_default() : NEMOTRON_EFFORT_NONE;
}

/* The head: the system turn, its content the trusted text (another family's tool prompt: text here) and the
 * client's system text -- client data both. */
/** pulsar_family_tokenizer::chat_open.
 *  @param e           the engine
 *  @param tokens      the head's ids are appended here
 *  @param trusted     the trusted system prefix (NULL = none)
 *  @param system      the client's system text (NULL = none)
 *  @param think_mode  checked against the engine (an unsupported mode ends the process) */
void chat_open(pulsar_engine *e, pulsar_tokens *tokens, const char *trusted, const char *system,
               pulsar_think_mode think_mode) {
    chat_effort(e, think_mode, "pulsar_chat_open");
    const std::string content = std::string(trusted ? trusted : "") + (system ? system : "");
    nemotron_render_out r;
    char err[256];
    if (!nemotron_chat_render_head(content.c_str(), &r, err, sizeof err)) chat_refuse("pulsar_chat_open", err);
    chat_append(e, r, tokens);
}

/* One turn: the messages as the full render writes them after the head or a sampled turn's close (a system note is
 * a system turn: the template renders one anywhere), then the generation prompt. */
/** pulsar_family_tokenizer::chat_turn.
 *  @param e                  the engine
 *  @param tokens             the turn's ids are appended here
 *  @param msgs               the turn's messages (a NULL role reads as "user")
 *  @param n                  the number of messages
 *  @param generation_prompt  append the assistant's opening after the messages
 *  @param think_mode         the effort the generation prompt opens with (an unsupported mode ends the process) */
void chat_turn(pulsar_engine *e, pulsar_tokens *tokens, const pulsar_chat_message *msgs, int n, bool generation_prompt,
               pulsar_think_mode think_mode) {
    const nemotron_effort effort = chat_effort(e, think_mode, "pulsar_chat_append_turn");
    std::vector<nemotron_msg_in> m((size_t)(n > 0 ? n : 0));
    for (int i = 0; i < n; i++) {
        m[(size_t)i] = {};
        m[(size_t)i].role = msgs[i].role ? msgs[i].role : "user";
        m[(size_t)i].content = msgs[i].content;
    }
    nemotron_render_out r;
    char err[256];
    if (!nemotron_chat_render_turn(m.data(), n, effort, generation_prompt, &r, err, sizeof err))
        chat_refuse("pulsar_chat_append_turn", err);
    chat_append(e, r, tokens);
}

/* the sampled turn's close: the "<|im_end|>\n" a tail opens with (nemotron_chat_render_tail) */
/** pulsar_family_tokenizer::chat_end_assistant.
 *  @param e       the engine
 *  @param tokens  the close's ids ("<|im_end|>\n") are appended here */
void chat_end_assistant(pulsar_engine *e, pulsar_tokens *tokens) { encode_into(e, "<|im_end|>\n", 11, NULL, 0u, tokens); }

/** pulsar_family_tokenizer::is_stop.
 *  @param e      the engine
 *  @param token  a token id
 *  @return true when it is one of generation_config's stop ids */
bool is_stop(pulsar_engine *e, int token) {
    for (int id : bpe_tokenizer_stop_ids(TOK(e))) if (id == token) return true;
    return false;
}
/* the config's eos_token_id (<|im_end|> 11; the load checked it is one of generation_config's stop ids, which list
 * </s> first); a caller that ends generation tests them all */
/** pulsar_family_tokenizer::eos.
 *  @param e  the engine
 *  @return the shape's eos_id */
int eos(pulsar_engine *e) { return (int)((const pulsar_nemotron_h_model *)e->family_state)->shape.eos_id; }

/** pulsar_family_tokenizer::token_text.
 *  @param e      the engine
 *  @param token  a token id
 *  @param len    receives the byte length (NULL = not wanted)
 *  @return the token's bytes, malloc'd and NUL-terminated (the caller frees); empty outside the table */
char *token_text(pulsar_engine *e, int token, size_t *len) {
    size_t n = 0;
    const char *b = bpe_tokenizer_token_bytes(TOK(e), token, &n);
    if (!b) n = 0;
    char *out = (char *)xmalloc(n + 1);
    if (n) memcpy(out, b, n);
    out[n] = '\0';
    if (len) *len = n;
    return out;
}

/** pulsar_family_tokenizer::think_close.
 *  @param e  the engine
 *  @return the id of the added token `</think>`, or -1 when the tokenizer has none */
int think_close(pulsar_engine *e) { return bpe_tokenizer_added_id(TOK(e), "</think>"); }

/* ChatML: a turn is <|im_start|> plus the role word.  The checkpoint's BPE holds "user" as one token and splits
 * "assistant" ("ass" "istant"), so a marker is <|im_start|> plus the role word's FIRST token -- which names the role
 * as long as no other role the template writes (user, system, tool) opens with the same token: checked here, and a
 * tokenizer where it does leaves the anchors off, said once. */
/** pulsar_family_tokenizer::turn_markers.
 *  @param e    the engine
 *  @param out  receives the two-id user and assistant markers
 *  @return false when the role words do not tell the turns apart (said once per process) */
bool turn_markers(pulsar_engine *e, pulsar_turn_markers *out) {
    const int start = bpe_tokenizer_added_id(TOK(e), "<|im_start|>");
    pulsar_tokens user = {0}, assistant = {0}, system = {0}, tool = {0};
    encode_text(e, "user", &user);
    encode_text(e, "assistant", &assistant);
    encode_text(e, "system", &system);
    encode_text(e, "tool", &tool);
    const bool ok = start >= 0 && user.len >= 1 && assistant.len >= 1 && system.len >= 1 && tool.len >= 1 &&
                    assistant.v[0] != user.v[0] && assistant.v[0] != system.v[0] && assistant.v[0] != tool.v[0] &&
                    user.v[0] != system.v[0] && user.v[0] != tool.v[0];
    pulsar_tokens_free(&system);
    pulsar_tokens_free(&tool);
    if (ok) {
        out->user[0] = start;
        out->user[1] = user.v[0];
        out->n_user = 2;
        out->assistant[0] = start;
        out->assistant[1] = assistant.v[0];
        out->n_assistant = 2;
    } else {
        static bool said = false;
        if (!said) {
            said = true;
            fprintf(stderr, "pulsar: %s: the role words' first tokens do not tell the turns apart after <|im_start|> "
                            "(%d): the chat turn markers are unknown\n", e->family->name, start);
        }
    }
    pulsar_tokens_free(&user);
    pulsar_tokens_free(&assistant);
    return ok;
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
 *  @return the ids in the tokenizer's table */
int n_tokens(pulsar_engine *e) { return bpe_tokenizer_n_tokens(TOK(e)); }

}  // namespace

const pulsar_family_tokenizer PULSAR_NEMOTRON_H_TOKENIZER = {
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

/** Load the tokenizer from tokenizer.json and generation_config.json beside the shards, and check it against the shape.
 *  @param w           the family state: its shape is read; tok is written
 *  @param model_path  the container's directory, or a shard path inside it
 *  @return true = loaded; false = refused (said) */
bool pulsar_nemotron_h_load_tokenizer(pulsar_nemotron_h_model *w, const char *model_path) {
    std::string dir = model_path ? model_path : "";
    struct stat st;
    if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        const size_t slash = dir.find_last_of('/');
        dir = slash == std::string::npos ? "." : dir.substr(0, slash);
    }
    std::string tok, gen;
    const std::string tp = dir + "/tokenizer.json", gp = dir + "/generation_config.json";
    if (!read_text(tp, &tok) || !read_text(gp, &gen)) {
        fprintf(stderr, "pulsar: %s: the container carries no %s / %s -- the builder (tools/container/nemotron_h.py) "
                        "writes the checkpoint's tokenizer.json and generation_config.json beside the shards; "
                        "refusing\n", ARCH, tp.c_str(), gp.c_str());
        return false;
    }
    char err[256] = "";
    w->tok = bpe_tokenizer_load_hf(tok.data(), tok.size(), gen.data(), gen.size(), err, sizeof(err));
    if (!w->tok) {
        fprintf(stderr, "pulsar: %s: tokenizer refused: %s\n", ARCH, err);
        return false;
    }
    if (strcmp(bpe_tokenizer_pattern(w->tok)->name, "tekken") != 0) {
        fprintf(stderr, "pulsar: %s: the tokenizer's pre-tokenizer is \"%s\", not this family's \"tekken\" -- refusing\n",
                ARCH, bpe_tokenizer_pattern(w->tok)->name);
        return false;
    }
    const std::vector<int> &stop = bpe_tokenizer_stop_ids(w->tok);
    bool has_eos = false;
    for (int id : stop) has_eos |= (uint32_t)id == w->shape.eos_id;
    if (!has_eos) {
        fprintf(stderr, "pulsar: %s: generation_config's stop ids do not hold the container's eos %u -- refusing\n", ARCH,
                w->shape.eos_id);
        return false;
    }
    if ((uint32_t)bpe_tokenizer_n_tokens(w->tok) != w->shape.n_vocab) {
        fprintf(stderr, "pulsar: %s: the tokenizer's %d ids are not the head's %u rows -- refusing\n", ARCH,
                bpe_tokenizer_n_tokens(w->tok), w->shape.n_vocab);
        return false;
    }
    fprintf(stderr, "pulsar: %s: tokenizer %d ids (HF BPE, tekken, ignore_merges) from %s; stop ids", ARCH,
            bpe_tokenizer_n_tokens(w->tok), tp.c_str());
    for (int id : stop) fprintf(stderr, " %d", id);
    fprintf(stderr, "\n");
    return true;
}

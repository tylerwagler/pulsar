/* tokenizer_qwen.cpp -- the Qwen4-exp family's tokenizer table (family.h pulsar_family_tokenizer, L272 P2).
 *
 * A Qwen engine tokenizes with src/lib/qwen_tokenizer (byte-exact with HF on the checkpoint's own
 * tokenizer.json, make qwen-chat-gate) and renders its chat with src/lib/qwen_chat (HF's template, byte for
 * byte) -- whole for the server, head and turn by turn for the CLI REPL and the agent (L284 P14).  Until L272
 * these bodies were branches on e->qwen_tok inside every public entry of tokenizer.cpp. */
#include "pulsar_engine_internal.h"
#include "lib/qwen_chat.h"
#include "lib/qwen_tokenizer.h"

#include <string.h>
#include <vector>

/* `spans` are the client-data ranges where no added token may match.  The entries have no error return;
 * an encode refusal (text that is not UTF-8 -- HF cannot take it either -- or spans out of order, a
 * renderer defect) is reported by name and yields no tokens, never a different tokenization. */
static void qwen_encode_into(const pulsar_engine *e, const char *text, const pulsar_text_span *spans,
                             uint32_t n_spans, pulsar_tokens *out) {
    std::vector<int> ids;
    char err[256] = "";
    const size_t len = text ? strlen(text) : 0;
    if (!qwen_tokenizer_encode(e->qwen_tok, text ? text : "", len, spans, n_spans, &ids, err, sizeof(err))) {
        fprintf(stderr, "pulsar: qwen tokenizer refused %zu bytes: %s\n", len, err);
        return;
    }
    for (int id : ids) pulsar_tokens_push(out, id);
}

/* raw text: all of it is client data, so no added token matches */
static void qwen_tok_encode_text(pulsar_engine *e, const char *text, pulsar_tokens *out) {
    const pulsar_text_span all = {0u, (uint32_t)(text ? strlen(text) : 0)};
    qwen_encode_into(e, text, all.hi ? &all : NULL, all.hi ? 1u : 0u, out);
}

static void qwen_tok_encode_rendered(pulsar_engine *e, const char *text, const pulsar_text_span *spans,
                                     uint32_t n_spans, pulsar_tokens *out) {
    qwen_encode_into(e, text, spans, n_spans, out);
}

/* ---- the chat front, turn by turn (pulsar.h pulsar_chat_open, L284 P14): qwen_chat's head and turn, the
 * pieces the full render is made of, tokenized with their client spans.  The entries have no error return, so
 * a refusal (an effort the template has no level for, a role it cannot place) ends the process by name. */

static void qwen_chat_refuse(const char *op, const char *err) {
    fprintf(stderr, "pulsar: %s: %s -- exiting\n", op, err);
    exit(1);
}

/* the template's effort for a think mode: on (its default effort) or off -- no other level */
static qwen_effort qwen_chat_effort(const pulsar_engine *e, pulsar_think_mode think_mode, const char *op) {
    char err[256];
    qwen_effort qe = QWEN_EFFORT_NONE;
    if (!pulsar_engine_think_mode_supported(e, think_mode, err, sizeof err) ||
        !qwen_effort_resolve(NULL, pulsar_think_mode_enabled(think_mode) ? 1 : 0, &qe, err, sizeof err))
        qwen_chat_refuse(op, err);
    return qe;
}

static void qwen_chat_append(const pulsar_engine *e, const qwen_render_out &r, pulsar_tokens *tokens) {
    qwen_encode_into(e, r.text.c_str(), r.spans.data(), (uint32_t)r.spans.size(), tokens);
}

/* The system block: trusted text (the agent's DSML spells no marker of this family's) and the system text, as
 * the system message's content -- client data both. */
static void qwen_chat_open(pulsar_engine *e, pulsar_tokens *tokens, const char *trusted, const char *system,
                           pulsar_think_mode think_mode) {
    const qwen_effort qe = qwen_chat_effort(e, think_mode, "pulsar_chat_open");
    const std::string content = std::string(trusted ? trusted : "") + (system ? system : "");
    qwen_render_out r;
    char err[256];
    if (!qwen_chat_render_head(content.c_str(), qe, &r, err, sizeof err)) qwen_chat_refuse("pulsar_chat_open", err);
    qwen_chat_append(e, r, tokens);
}

/* A mid-conversation system message is the user turn of its reminder (qwen_system_reminder), as the server
 * renders it. */
static void qwen_chat_turn(pulsar_engine *e, pulsar_tokens *tokens, const pulsar_chat_message *msgs, int n,
                           bool generation_prompt, pulsar_think_mode think_mode) {
    const qwen_effort qe = qwen_chat_effort(e, think_mode, "pulsar_chat_append_turn");
    std::vector<std::string> notes((size_t)(n > 0 ? n : 0));
    std::vector<qwen_msg_in> qm;
    for (int i = 0; i < n; i++) {
        const char *role = msgs[i].role ? msgs[i].role : "user";
        if (!strcmp(role, "system")) {
            notes[(size_t)i] = qwen_system_reminder(msgs[i].content);
            qm.push_back({"user", notes[(size_t)i].c_str(), NULL, NULL, 0, NULL, NULL, 0});
        } else {
            qm.push_back({role, msgs[i].content, NULL, NULL, 0, NULL, NULL, 0});
        }
    }
    qwen_render_out r;
    char err[256];
    if (!qwen_chat_render_turn(qm.data(), (int)qm.size(), qe, generation_prompt, &r, err, sizeof err))
        qwen_chat_refuse("pulsar_chat_append_turn", err);
    qwen_chat_append(e, r, tokens);
}

/* the turn's end the template writes after a sampled turn (qwen_chat_render_tail's first bytes) */
static void qwen_chat_end_assistant(pulsar_engine *e, pulsar_tokens *tokens) {
    qwen_encode_into(e, "<|im_end|>\n", NULL, 0u, tokens);
}

static bool qwen_tok_is_stop(pulsar_engine *e, int token) {
    for (int id : qwen_tokenizer_stop_ids(e->qwen_tok)) if (id == token) return true;
    return false;
}

/* the FIRST of generation_config's stop ids (<|im_end|>); a caller that ends generation must test the
 * whole set with pulsar_token_is_stop */
static int qwen_tok_eos(pulsar_engine *e) { return qwen_tokenizer_stop_ids(e->qwen_tok).front(); }

static char *qwen_tok_token_text(pulsar_engine *e, int token, size_t *len) {
    size_t n = 0;
    const char *b = qwen_tokenizer_token_bytes(e->qwen_tok, token, &n);
    if (!b) n = 0;   /* out of range, e.g. the padded logits rows past the table: no text */
    char *out = (char *)xmalloc(n + 1);
    if (n) memcpy(out, b, n);
    out[n] = '\0';
    if (len) *len = n;
    return out;
}

static int qwen_tok_think_close(pulsar_engine *e) { return qwen_tokenizer_added_id(e->qwen_tok, "</think>"); }

/* L272 B5: Qwen's template spells a turn as <|im_start|> plus the role word, which the checkpoint's BPE
 * holds as one token (Qwen3.8-Flash-Next: "user", "assistant").  A tokenizer that splits either word
 * leaves the anchors off, said once -- never a marker that matches at the wrong place. */
static bool qwen_tok_turn_markers(pulsar_engine *e, pulsar_turn_markers *out) {
    const int start = qwen_tokenizer_added_id(e->qwen_tok, "<|im_start|>");
    pulsar_tokens user = {0}, assistant = {0};
    qwen_tok_encode_text(e, "user", &user);
    qwen_tok_encode_text(e, "assistant", &assistant);
    const bool ok = start >= 0 && user.len == 1 && assistant.len == 1;
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
            fprintf(stderr, "pulsar: %s: the tokenizer does not spell a turn as <|im_start|> (%d) plus one role "
                            "token (\"user\" %d, \"assistant\" %d tokens): the chat turn markers are unknown\n",
                    e->family->name, start, user.len, assistant.len);
        }
    }
    pulsar_tokens_free(&user);
    pulsar_tokens_free(&assistant);
    return ok;
}

/* --dump-tokens: the ids, then one line per token with its bytes (the HF tokenizer has no separate raw
 * vocab string to print beside them) */
static void qwen_tok_dump(pulsar_engine *e, FILE *fp, const pulsar_tokens *tokens) {
    fprintf(fp, "[");
    for (int i = 0; i < tokens->len; i++) fprintf(fp, "%s%d", i ? ", " : "", tokens->v[i]);
    fprintf(fp, "]\n");
    for (int i = 0; i < tokens->len; i++) {
        size_t n = 0;
        char *piece = qwen_tok_token_text(e, tokens->v[i], &n);
        fprintf(fp, "%6d  ", tokens->v[i]);
        pulsar_dump_piece_quoted(fp, piece, n);
        fputc('\n', fp);
        free(piece);
    }
}

const pulsar_family_tokenizer k_qwen_tokenizer = {
    /* .encode_text              = */ qwen_tok_encode_text,
    /* .encode_rendered          = */ qwen_tok_encode_rendered,
    /* .chat_open                = */ qwen_chat_open,
    /* .chat_turn                = */ qwen_chat_turn,
    /* .chat_end_assistant       = */ qwen_chat_end_assistant,
    /* .is_stop                  = */ qwen_tok_is_stop,
    /* .eos                      = */ qwen_tok_eos,
    /* .token_text               = */ qwen_tok_token_text,
    /* .think_close              = */ qwen_tok_think_close,
    /* .turn_markers             = */ qwen_tok_turn_markers,
    /* .dump                     = */ qwen_tok_dump,
};

/* tokenizer_qwen.cpp -- the Qwen4-exp family's tokenizer table (family.h pulsar_family_tokenizer, L272 P2).
 *
 * A Qwen engine tokenizes with src/lib/qwen_tokenizer (byte-exact with HF on the checkpoint's own
 * tokenizer.json, make qwen-chat-gate) and renders its chat whole with src/lib/qwen_chat (HF's template,
 * byte for byte), so it has no incremental marker template.  Until L272 these bodies were branches on
 * e->qwen_tok inside every public entry of tokenizer.cpp. */
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

/* the server's path: render (HF's template, byte for byte), tokenize with its spans */
static void qwen_tok_encode_chat_prompt(pulsar_engine *e, const char *system, const char *prompt,
                                        pulsar_think_mode think_mode, pulsar_tokens *out) {
    qwen_effort qe = QWEN_EFFORT_NONE;
    char err[256] = "";
    const qwen_msg_in msgs[2] = {{"system", system, NULL, NULL, 0}, {"user", prompt ? prompt : "", NULL, NULL, 0}};
    const bool has_system = system && system[0];
    qwen_render_out r;
    bool ok = think_mode == PULSAR_THINK_NONE || think_mode == PULSAR_THINK_DEFAULT;
    if (!ok) snprintf(err, sizeof(err), "thinking effort %d has no Qwen effort (on or off only)", think_mode);
    ok = ok && qwen_effort_resolve(NULL, pulsar_think_mode_enabled(think_mode) ? 1 : 0, &qe, err, sizeof(err)) &&
         qwen_chat_render({has_system ? msgs : msgs + 1, has_system ? 2 : 1, NULL, qe, true}, &r, err, sizeof(err));
    if (!ok) {
        fprintf(stderr, "pulsar: pulsar_encode_chat_prompt: %s -- exiting\n", err);
        exit(1);
    }
    qwen_encode_into(e, r.text.c_str(), r.spans.data(), (uint32_t)r.spans.size(), out);
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
    /* .encode_chat_prompt       = */ qwen_tok_encode_chat_prompt,
    /* .is_stop                  = */ qwen_tok_is_stop,
    /* .eos                      = */ qwen_tok_eos,
    /* .token_text               = */ qwen_tok_token_text,
    /* .think_close              = */ qwen_tok_think_close,
    /* .turn_markers             = */ qwen_tok_turn_markers,
    /* .dump                     = */ qwen_tok_dump,
    /* .incremental_ds4_template = */ false,
};

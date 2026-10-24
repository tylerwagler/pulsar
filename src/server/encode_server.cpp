/* encode_server.cpp -- L321: pulsar-server for an ENCODER engine (pulsar.h pulsar_engine_encode*).
 *
 * An encoder has no session, no chat and no decode schedule, so it serves its own small surface beside none of the
 * decoder's machinery: one thread per connection, each request rendered, tokenized and encoded in one engine call
 * (the engine serialises calls; a request's inputs ride one call, packed into its forwards).
 *
 *   GET  /health, /healthz, /ping, /v1/models
 *   POST /v1/embeddings   (an embedder; OpenAI's shape)
 *        {"input": "text" | ["text", ...], "input_type": "query" | "passage" (NIM's field; default "passage"),
 *         "encoding_format": "float" | "base64", "truncate": "NONE" | "END" (NIM's; default NONE), "model": any}
 *        -> {"object": "list", "data": [{"object": "embedding", "index": i, "embedding": [...] | "<base64 f32 LE>"}],
 *            "model": id, "usage": {"prompt_tokens": n, "total_tokens": n}}
 *        The family's template writes the prefix ("query: " / "passage: "): a client must not prefix the text itself.
 *   POST /v1/rerank, /rerank   (a reranker; the Cohere / Jina / Infinity shape)
 *        {"query": "text", "documents": ["text" | {"text": "text"}, ...], "top_n": k, "return_documents": bool,
 *         "raw_scores": bool, "truncate": "NONE" | "END", "model": any}
 *        -> {"model": id, "results": [{"index": i, "relevance_score": s [, "document": {"text": ..}]}, ...],
 *            "usage": {"total_tokens": n}}   results best first; s = sigmoid(the head's logit) -- a score in (0, 1)
 *            as the shape's clients threshold it -- or the logit itself with raw_scores.
 * An input longer than the model's cap (pulsar_engine_encode_max_tokens) is refused (400) unless truncate is END,
 * which keeps its first cap tokens. */
#include "pulsar_server_internal.h"
#include "lib/pyjson.h"

#include <math.h>

#include <algorithm>
#include <string>
#include <vector>

namespace {

/** Inputs one request may carry. */
constexpr size_t ENC_INPUTS_MAX = 2048;

struct enc_ctx {
    pulsar_engine *e;
    pulsar_encoder_kind kind;
    uint32_t dim, cap;
    const char *id;
};

struct enc_arg {
    const enc_ctx *c;
    int fd;
};

bool send_json(int fd, buf *b) {
    const bool ok = http_response(fd, 200, "application/json", b->ptr ? b->ptr : "{}");
    buf_free(b);
    return ok;
}

void append_base64(buf *b, const uint8_t *p, size_t n) {
    static const char A[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v = (uint32_t)p[i] << 16 | (i + 1 < n ? (uint32_t)p[i + 1] << 8 : 0u) |
                           (i + 2 < n ? (uint32_t)p[i + 2] : 0u);
        buf_putc(b, A[v >> 18 & 63]);
        buf_putc(b, A[v >> 12 & 63]);
        buf_putc(b, i + 1 < n ? A[v >> 6 & 63] : '=');
        buf_putc(b, i + 2 < n ? A[v & 63] : '=');
    }
}

/* the optional string member `key` of `o` (NULL when absent); false when present and not a string */
bool opt_str(const pyjson_value &o, const char *key, const std::string **out) {
    const pyjson_value *v = o.get(key);
    *out = v ? &v->s : NULL;
    return !v || v->kind == pyjson_value::STR;
}

/* the request's truncate: NONE (refuse an over-long input) or END (keep its first cap tokens) */
bool read_truncate(const pyjson_value &o, bool *cut, std::string *err) {
    const std::string *t = NULL;
    if (!opt_str(o, "truncate", &t) || (t && *t != "NONE" && *t != "END")) {
        *err = "\"truncate\" must be \"NONE\" or \"END\"";
        return false;
    }
    *cut = t && *t == "END";
    return true;
}

/* render one input; an over-long one cut (truncate END) or refused */
bool render_one(const enc_ctx *c, pulsar_encode_role role, const char *text, const char *doc, bool cut,
                pulsar_tokens *t, size_t index, std::string *err) {
    char why[384];
    if (!pulsar_engine_encode_render(c->e, role, text, doc, t, why, sizeof why)) {
        *err = "input " + std::to_string(index) + ": " + why;
        return false;
    }
    if ((uint32_t)t->len > c->cap) {
        if (!cut) {
            *err = "input " + std::to_string(index) + " has " + std::to_string(t->len) + " tokens; the model takes " +
                   std::to_string(c->cap) + " (send \"truncate\": \"END\" to keep the first " + std::to_string(c->cap) +
                   ")";
            return false;
        }
        t->len = (int)c->cap;
    }
    return true;
}

void free_all(std::vector<pulsar_tokens> *v) {
    for (pulsar_tokens &t : *v) pulsar_tokens_free(&t);
}

void embeddings(const enc_ctx *c, int fd, const pyjson_value &o) {
    std::string err;
    const pyjson_value *in = o.get("input");
    std::vector<const std::string *> texts;
    if (in && in->kind == pyjson_value::STR) texts.push_back(&in->s);
    else if (in && in->kind == pyjson_value::ARR)
        for (const pyjson_value &x : in->a) texts.push_back(x.kind == pyjson_value::STR ? &x.s : NULL);
    if (texts.empty() || texts.size() > ENC_INPUTS_MAX ||
        std::find(texts.begin(), texts.end(), nullptr) != texts.end()) {
        http_error(fd, 400, "\"input\" must be a string or an array of 1 .. 2048 strings (token arrays are not taken)");
        return;
    }
    const std::string *type = NULL, *fmt = NULL;
    bool cut = false;
    if (!opt_str(o, "input_type", &type) || (type && *type != "query" && *type != "passage")) {
        http_error(fd, 400, "\"input_type\" must be \"query\" or \"passage\"");
        return;
    }
    if (!opt_str(o, "encoding_format", &fmt) || (fmt && *fmt != "float" && *fmt != "base64")) {
        http_error(fd, 400, "\"encoding_format\" must be \"float\" or \"base64\"");
        return;
    }
    if (const pyjson_value *d = o.get("dimensions"); d && !(d->kind == pyjson_value::INT && d->num == std::to_string(c->dim))) {
        http_error(fd, 400, ("\"dimensions\": this model returns " + std::to_string(c->dim) + " only").c_str());
        return;
    }
    if (!read_truncate(o, &cut, &err)) {
        http_error(fd, 400, err.c_str());
        return;
    }
    const pulsar_encode_role role = type && *type == "query" ? PULSAR_ENCODE_QUERY : PULSAR_ENCODE_PASSAGE;
    std::vector<pulsar_tokens> toks(texts.size(), pulsar_tokens{});
    uint64_t n_tok = 0;
    for (size_t i = 0; i < texts.size(); i++) {
        if (!render_one(c, role, texts[i]->c_str(), NULL, cut, &toks[i], i, &err)) {
            free_all(&toks);
            http_error(fd, 400, err.c_str());
            return;
        }
        n_tok += (uint64_t)toks[i].len;
    }
    std::vector<float> out(texts.size() * c->dim);
    char why[384];
    const int rc = pulsar_engine_encode(c->e, toks.data(), (uint32_t)toks.size(), out.data(), why, sizeof why);
    free_all(&toks);
    if (rc != 0) {
        http_error(fd, 500, why);
        return;
    }
    const bool b64 = fmt && *fmt == "base64";
    buf b = {0};
    buf_puts(&b, "{\"object\":\"list\",\"data\":[");
    for (size_t i = 0; i < texts.size(); i++) {
        buf_printf(&b, "%s{\"object\":\"embedding\",\"index\":%zu,\"embedding\":", i ? "," : "", i);
        const float *v = &out[i * c->dim];
        if (b64) {
            buf_putc(&b, '"');
            append_base64(&b, (const uint8_t *)v, (size_t)c->dim * sizeof(float));   /* f32 little-endian */
            buf_putc(&b, '"');
        } else {
            buf_putc(&b, '[');
            for (uint32_t k = 0; k < c->dim; k++) buf_printf(&b, k ? ",%.9g" : "%.9g", (double)v[k]);
            buf_putc(&b, ']');
        }
        buf_putc(&b, '}');
    }
    buf_puts(&b, "],\"model\":");
    json_escape(&b, c->id);
    buf_printf(&b, ",\"usage\":{\"prompt_tokens\":%llu,\"total_tokens\":%llu}}\n", (unsigned long long)n_tok,
               (unsigned long long)n_tok);
    send_json(fd, &b);
}

void rerank(const enc_ctx *c, int fd, const pyjson_value &o) {
    std::string err;
    const pyjson_value *q = o.get("query"), *docs = o.get("documents");
    if (!q || q->kind != pyjson_value::STR) {
        http_error(fd, 400, "\"query\" must be a string");
        return;
    }
    std::vector<const std::string *> texts;
    if (docs && docs->kind == pyjson_value::ARR)
        for (const pyjson_value &x : docs->a) {
            const pyjson_value *t = x.kind == pyjson_value::OBJ ? x.get("text") : &x;
            texts.push_back(t && t->kind == pyjson_value::STR ? &t->s : NULL);
        }
    if (texts.empty() || texts.size() > ENC_INPUTS_MAX ||
        std::find(texts.begin(), texts.end(), nullptr) != texts.end()) {
        http_error(fd, 400, "\"documents\" must be an array of 1 .. 2048 strings or {\"text\": string} objects");
        return;
    }
    size_t top_n = texts.size();
    if (const pyjson_value *k = o.get("top_n")) {
        if (k->kind != pyjson_value::INT || k->num[0] == '-' || k->num == "0") {
            http_error(fd, 400, "\"top_n\" must be a positive integer");
            return;
        }
        const unsigned long long v = strtoull(k->num.c_str(), NULL, 10);
        if (v < top_n) top_n = (size_t)v;
    }
    bool return_docs = false, raw = false, cut = false;
    for (auto [key, dst] : {std::pair<const char *, bool *>{"return_documents", &return_docs}, {"raw_scores", &raw}})
        if (const pyjson_value *v = o.get(key)) {
            if (v->kind != pyjson_value::BOOL) {
                http_error(fd, 400, (std::string("\"") + key + "\" must be a boolean").c_str());
                return;
            }
            *dst = v->b;
        }
    if (!read_truncate(o, &cut, &err)) {
        http_error(fd, 400, err.c_str());
        return;
    }
    std::vector<pulsar_tokens> toks(texts.size(), pulsar_tokens{});
    uint64_t n_tok = 0;
    for (size_t i = 0; i < texts.size(); i++) {
        if (!render_one(c, PULSAR_ENCODE_PAIR, q->s.c_str(), texts[i]->c_str(), cut, &toks[i], i, &err)) {
            free_all(&toks);
            http_error(fd, 400, err.c_str());
            return;
        }
        n_tok += (uint64_t)toks[i].len;
    }
    std::vector<float> logit(texts.size());
    char why[384];
    const int rc = pulsar_engine_encode(c->e, toks.data(), (uint32_t)toks.size(), logit.data(), why, sizeof why);
    free_all(&toks);
    if (rc != 0) {
        http_error(fd, 500, why);
        return;
    }
    std::vector<size_t> order(texts.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return logit[a] > logit[b]; });
    buf b = {0};
    buf_puts(&b, "{\"model\":");
    json_escape(&b, c->id);
    buf_puts(&b, ",\"results\":[");
    for (size_t r = 0; r < top_n; r++) {
        const size_t i = order[r];
        const double s = raw ? (double)logit[i] : 1.0 / (1.0 + exp(-(double)logit[i]));
        buf_printf(&b, "%s{\"index\":%zu,\"relevance_score\":%.9g", r ? "," : "", i, s);
        if (return_docs) {
            buf_puts(&b, ",\"document\":{\"text\":");
            json_escape(&b, texts[i]->c_str());
            buf_putc(&b, '}');
        }
        buf_putc(&b, '}');
    }
    buf_printf(&b, "],\"usage\":{\"total_tokens\":%llu}}\n", (unsigned long long)n_tok);
    send_json(fd, &b);
}

void health(const enc_ctx *c, int fd) {
    buf b = {0};
    buf_puts(&b, "{\"status\":\"ok\",\"model\":");
    json_escape(&b, c->id);
    buf_printf(&b, ",\"encoder\":\"%s\",\"dim\":%u,\"max_tokens\":%u,\"endpoints\":[\"/health\",\"/v1/models\",\"%s\"]}\n",
               c->kind == PULSAR_ENCODER_EMBED ? "embed" : "rerank", c->dim, c->cap,
               c->kind == PULSAR_ENCODER_EMBED ? "/v1/embeddings" : "/v1/rerank");
    send_json(fd, &b);
}

void models(const enc_ctx *c, int fd) {
    buf b = {0};
    buf_puts(&b, "{\"object\":\"list\",\"data\":[{\"id\":");
    json_escape(&b, c->id);
    buf_puts(&b, ",\"object\":\"model\",\"owned_by\":\"pulsar\"}]}\n");
    send_json(fd, &b);
}

void *client(void *arg) {
    enc_arg *a = (enc_arg *)arg;
    const enc_ctx *c = a->c;
    const int fd = a->fd;
    free(a);
    http_request hr = {};
    if (!read_http_request(fd, &hr)) {
        http_error(fd, 400, "bad HTTP request");
    } else if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/health")) {
        health(c, fd);
    } else if (!strcmp(hr.method, "GET") && (!strcmp(hr.path, "/healthz") || !strcmp(hr.path, "/ping"))) {
        http_response(fd, 200, "text/plain", "ok\n");
    } else if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/v1/models")) {
        models(c, fd);
    } else if (!strcmp(hr.method, "POST") &&
               (c->kind == PULSAR_ENCODER_EMBED ? !strcmp(hr.path, "/v1/embeddings")
                                                : !strcmp(hr.path, "/v1/rerank") || !strcmp(hr.path, "/rerank"))) {
        pyjson_value o;
        char err[256];
        if (!pyjson_parse(hr.body ? hr.body : "", hr.body_len, &o, err, sizeof err) || o.kind != pyjson_value::OBJ) {
            http_error(fd, 400, "the body is not a JSON object");
        } else {
            const double t0 = server_now_sec();
            if (c->kind == PULSAR_ENCODER_EMBED) embeddings(c, fd, o);
            else rerank(c, fd, o);
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: %s %.0f ms", hr.path, (server_now_sec() - t0) * 1e3);
        }
    } else {
        http_error(fd, 404, c->kind == PULSAR_ENCODER_EMBED
                                ? "an embedder serves POST /v1/embeddings, GET /v1/models and /health"
                                : "a reranker serves POST /v1/rerank, GET /v1/models and /health");
    }
    http_request_free(&hr);
    close(fd);
    return NULL;
}

}  // namespace

int encode_serve(pulsar_engine *e, const server_config *cfg) {
    enc_ctx c = {e, pulsar_engine_encoder(e), pulsar_engine_encode_dim(e), pulsar_engine_encode_max_tokens(e),
                 pulsar_engine_served_model_id(e)};
    const int lfd = listen_on(cfg->host, cfg->port);
    if (lfd < 0) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: failed to listen on %s:%d: %s", cfg->host, cfg->port,
                   strerror(errno));
        return 1;
    }
    g_listen_fd = lfd;
    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: %s (%s, %u floats an input, inputs up to %u tokens) listening on "
               "http://%s:%d (pulsar %s)", c.id, c.kind == PULSAR_ENCODER_EMBED ? "embedder: POST /v1/embeddings"
                                                                               : "reranker: POST /v1/rerank",
               c.dim, c.cap, cfg->host, cfg->port, PULSAR_VERSION_STR);
    while (!g_stop_requested) {
        const int fd = accept(lfd, NULL, NULL);
        if (fd < 0) {
            if (g_stop_requested) break;
            if (errno == EINTR) continue;
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: accept failed: %s", strerror(errno));
            continue;
        }
        configure_client_socket(fd);
        enc_arg *a = (enc_arg *)server_xmalloc(sizeof(*a));
        a->c = &c;
        a->fd = fd;
        pthread_t th;
        if (pthread_create(&th, NULL, client, a) != 0) {
            free(a);
            http_error_retry(fd, 503, "too many connections", HTTP_RETRY_BUSY_S);
            close(fd);
            continue;
        }
        pthread_detach(th);
    }
    if (g_listen_fd >= 0) {
        close(lfd);
        g_listen_fd = -1;
    }
    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: shutdown requested");
    return 0;
}

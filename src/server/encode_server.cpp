/* encode_server.cpp -- L321: pulsar-server for the SINGLE-CALL engines: encoders (pulsar.h pulsar_engine_encode*)
 * and parsers (pulsar_engine_parse_page).
 *
 * Neither has a session, a chat or a decode schedule, so they serve their own small surface beside none of the
 * decoder's machinery: one thread per connection for the HTTP side, each request one engine call (a request's inputs
 * ride one encode call, packed into its forwards; a page is one parse call).  One process may serve several (-m
 * repeated: pulsar_engine_open_group, one GPU, the engines taking turns a GPU phase at a time): a request goes to the
 * engine of its endpoint's kind -- an embedder for /v1/embeddings, a reranker for /v1/rerank, a parser for /v1/parse
 * -- and, when two engines share a kind, the one its "model" names (with one, "model" is not read).
 *
 * L374: the engine call itself runs on the engine's ONE long-lived GPU thread (enc_executor), never on the
 * connection thread.  The CUDA side keeps its plans, descriptors, scratch arena and counter slices per submitting
 * thread (the chat server's rule: one long-lived worker), so a forward on a thread that lives one connection rebuilt
 * every plan, announced every shape again and left its descriptors and scratch behind at each request.  A connection
 * thread parses, validates and renders, queues the call on its engine's executor (FIFO) and waits for it, then
 * writes the response.  One executor an ENGINE, not one a process: engines still interleave a GPU phase at a time
 * (pulsar_group_gpu_phase), so an embedding does not wait behind a whole page of another engine's parse.
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
 *   POST /v1/parse   (a parser: one page image in, the model's text out)
 *        {"image": "data:image/png;base64,..." (PNG or JPEG, inline), "prompt": the task's added tokens (default the
 *         model's own), "max_tokens": n (default the model's), "return_ids": bool, "model": any}
 *        -> {"object": "parse", "model": id, "text": the decode of the prompt and the generated tokens (the model's
 *            markup, special tokens kept: the checkpoint's postprocessing reads it), "finish_reason": "stop" | "length",
 *            ["ids": [...]], "image": {"width": w, "height": h}, "usage": {"prompt_tokens", "completion_tokens",
 *            "total_tokens"}, "timings": {"vision_ms", "prefill_ms", "decode_ms"}}
 * An input longer than the model's cap (pulsar_engine_encode_max_tokens) is refused (400) unless truncate is END,
 * which keeps its first cap tokens. */
/** @file
 *  pulsar-server for the single-call engines (L321): /v1/embeddings, /v1/rerank and /v1/parse over encoders
 *  and parsers, one HTTP thread per connection and one long-lived GPU thread per engine (L374).
 */
#include "pulsar_server_internal.h"
#include "lib/pyjson.h"
#include "pulsar_lock.hpp"

#include <math.h>

#include <algorithm>
#include <deque>
#include <string>
#include <vector>

namespace {

/** Inputs one request may carry. */
constexpr size_t ENC_INPUTS_MAX = 2048;

/** What an engine serves: the encoder kinds' values (pulsar_encoder_kind) and the parser. */
enum srv_kind { K_EMBED = PULSAR_ENCODER_EMBED, K_RERANK = PULSAR_ENCODER_RERANK, K_PARSE, K_COUNT };

/** One engine call queued on an enc_executor.  It lives on its connection thread's stack: that thread waits in
 *  enc_executor_run until the executor has marked it `finished`, so the executor never holds a job whose owner left
 *  and a job is never freed while it runs. */
struct enc_job {
    /// Calls `work` (enc_executor_run's trampoline over the caller's callable).
    void (*call)(const void *work);
    /// The caller's callable, on the caller's stack.
    const void *work;
    /// The client socket: a job whose client has gone by its turn is not run (gen_client_disconnected).
    int fd;
    /// Set by the executor, under enc_executor::mu, once the job was run or dropped; it does not touch the job after.
    bool finished;
    /// Whether `call` ran (false: dropped, the client had gone).  Valid once `finished`.
    bool ran;
};

/** An engine's one long-lived GPU thread (L374) and its FIFO queue: every pulsar_engine_encode /
 *  pulsar_engine_parse_page of that engine runs here, so the CUDA side's per-thread caches (cuBLASLt plans and
 *  descriptors, the fixed GEMM's shapes, the scratch arena, the EXL3 counter slices) are built once and live as long
 *  as the server.  Started by enc_executor_start, ended by enc_executor_stop before the engine closes. */
struct enc_executor {
    /// Guards `q`, `stop`, `waiters` and every queued job's `finished` / `ran`.
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    /// Signalled when a job is queued or `stop` is set.
    pthread_cond_t wake = PTHREAD_COND_INITIALIZER;
    /// Broadcast when a job finishes (its waiter checks its own `finished`) and when `waiters` drops to 0.
    pthread_cond_t done = PTHREAD_COND_INITIALIZER;
    /// The jobs waiting their turn, oldest first.
    std::deque<enc_job *> q;
    /// Set by enc_executor_stop: the thread runs what is queued, then exits.
    bool stop = false;
    /// Threads inside enc_executor_run: enc_executor_stop frees the executor only once they have all left `mu`.
    int waiters = 0;
    /// The GPU thread.
    pthread_t th;
};

/** An enc_executor's thread: take the oldest job, run it unless its client has gone, mark it finished; exit once
 *  `stop` is set and the queue is empty.
 *  @param arg the enc_executor
 *  @return NULL */
void *enc_executor_main(void *arg) {
    enc_executor *x = (enc_executor *)arg;
    pthread_mutex_lock(&x->mu);
    for (;;) {
        while (x->q.empty() && !x->stop) pthread_cond_wait(&x->wake, &x->mu);
        if (x->q.empty()) break;
        enc_job *j = x->q.front();
        x->q.pop_front();
        pthread_mutex_unlock(&x->mu);
        const bool run = !gen_client_disconnected(j->fd);
        if (run) j->call(j->work);
        pthread_mutex_lock(&x->mu);
        j->ran = run;
        j->finished = true;   /* its waiter may free it as soon as `mu` goes: not read again */
        pthread_cond_broadcast(&x->done);
    }
    pthread_mutex_unlock(&x->mu);
    return NULL;
}

/** Start an executor's GPU thread.
 *  @return the executor (ended and freed by enc_executor_stop), or NULL when the thread could not start */
enc_executor *enc_executor_start(void) {
    enc_executor *x = new enc_executor();
    if (pthread_create(&x->th, NULL, enc_executor_main, x) != 0) {
        delete x;
        return NULL;
    }
    return x;
}

/** End an executor: its thread runs every job still queued, then exits; joined here, and the executor freed once
 *  every waiter of those jobs has left enc_executor_run.  No job may be queued after this is called (encode_serve
 *  calls it once every connection thread has finished).
 *  @param x the executor (freed) */
void enc_executor_stop(enc_executor *x) {
    pthread_mutex_lock(&x->mu);
    x->stop = true;
    pthread_cond_broadcast(&x->wake);
    pthread_mutex_unlock(&x->mu);
    pthread_join(x->th, NULL);
    pthread_mutex_lock(&x->mu);
    while (x->waiters > 0) pthread_cond_wait(&x->done, &x->mu);
    pthread_mutex_unlock(&x->mu);
    delete x;
}

/** Run `work()` on the executor's GPU thread, in FIFO order behind the jobs already queued, and wait for it.
 *  @tparam F a callable taking no arguments; its captures carry the call's inputs and results
 *  @param x the executor
 *  @param fd the client socket (< 0: none); checked at the job's turn
 *  @param work the engine call
 *  @return true when `work` ran; false when the client had gone by the job's turn and it was dropped unrun */
template <class F> bool enc_executor_run(enc_executor *x, int fd, const F &work) {
    enc_job j = {[](const void *w) { (*(const F *)w)(); }, &work, fd, false, false};
    pulsar::ScopedLock lk(&x->mu);
    x->waiters++;
    x->q.push_back(&j);
    pthread_cond_signal(&x->wake);
    while (!j.finished) pthread_cond_wait(&x->done, &x->mu);
    if (--x->waiters == 0) pthread_cond_broadcast(&x->done);
    return j.ran;
}

/** One served engine.  Built once in encode_serve before the accept loop; read-only after, by every client thread. */
struct enc_ctx {
    /// The engine (owned by encode_serve's caller).
    pulsar_engine *e;
    /// The engine's GPU thread: every forward of `e` runs there (started and stopped by encode_serve).
    enc_executor *gpu;
    /// The endpoint it serves.
    srv_kind kind;
    /** dim: floats an embedding (a reranker's 1); cap: an input's token limit (pulsar_engine_encode_max_tokens).
     *  Unused for a parser. */
    uint32_t dim, cap;   ///< an encoder's
    /// The served model id the "model" field names (pulsar_engine_served_model_id).
    const char *id;
};

/** Every engine the process serves, and the connections in flight (shutdown waits for them, then ends the
 *  engines' GPU threads: the engines close after encode_serve returns). */
struct enc_server {
    /// The engines, in -m order; fixed before the first accept.
    std::vector<enc_ctx> m;
    /// Guards `clients`.
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    /// Signalled when `clients` drops to 0.
    pthread_cond_t idle = PTHREAD_COND_INITIALIZER;
    /// Connection threads in flight, under `mu`.
    int clients = 0;
};

/** A connection thread's argument: malloc'd by the accept loop, freed by client(). */
struct enc_arg {
    /// The server (outlives every client thread: encode_serve waits for them).
    const enc_server *srv;
    /// The accepted socket; the client thread closes it.
    int fd;
};

/** Send `b` as a 200 application/json response and free it.
 *  @param fd the client socket
 *  @param b the body (an empty buffer sends "{}"); freed
 *  @return false when the write failed */
bool send_json(int fd, buf *b) {
    const bool ok = http_response(fd, 200, "application/json", b->ptr ? b->ptr : "{}");
    buf_free(b);
    return ok;
}

/** Append the standard base64 (with '=' padding) of `n` bytes.
 *  @param b the buffer appended to
 *  @param p the bytes
 *  @param n their count */
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
/** @param o a JSON object
 *  @param key the member's name
 *  @param out set to the member's string (pointing into `o`), NULL when absent
 *  @return false when the member is present and not a string */
bool opt_str(const pyjson_value &o, const char *key, const std::string **out) {
    const pyjson_value *v = o.get(key);
    *out = v ? &v->s : NULL;
    return !v || v->kind == pyjson_value::STR;
}

/* L335: the request's "dimensions", compared as a NUMBER: 1024 and 1024.0 are the same request (a JSON serializer
 * may spell an integral float either way); a non-integral or non-numeric value is refused as such, and an integral
 * one other than the model's width as that. */
/** @param d   the "dimensions" member (NULL: absent, accepted)
 *  @param dim the model's embedding width
 *  @param err the 400 message on failure
 *  @return false when present and not the integer `dim` */
bool read_dimensions(const pyjson_value *d, uint32_t dim, std::string *err) {
    if (!d) return true;
    const bool integral = d->kind == pyjson_value::INT ||
                          (d->kind == pyjson_value::FLOAT && isfinite(d->f) && d->f == floor(d->f));
    if (!integral) {
        *err = "\"dimensions\" must be an integer";
        return false;
    }
    const bool same = d->kind == pyjson_value::INT ? d->num == std::to_string(dim) : d->f == (double)dim;
    if (!same) {
        *err = "\"dimensions\": this model returns " + std::to_string(dim) + " only";
        return false;
    }
    return true;
}

/* the request's truncate: NONE (refuse an over-long input) or END (keep its first cap tokens) */
/** @param o the request object
 *  @param cut set true for END
 *  @param err the 400 message on failure
 *  @return false when "truncate" is present and neither "NONE" nor "END" */
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
/** @param c the engine
 *  @param role query, passage, or (rerank) a query-document pair
 *  @param text the input (the query for a pair)
 *  @param doc the document of a pair; NULL otherwise
 *  @param cut truncate END: keep the first `c->cap` tokens instead of refusing
 *  @param t the rendered tokens (the caller frees them, also on failure)
 *  @param index the input's index, for the message
 *  @param err the 400 message on failure
 *  @return false when the template refused the input or it is over `c->cap` tokens without `cut` */
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

/** Free every token list in `v` (the vector itself stays).
 *  @param v the rendered inputs */
void free_all(std::vector<pulsar_tokens> *v) {
    for (pulsar_tokens &t : *v) pulsar_tokens_free(&t);
}

/** Run a request's engine call on its engine's GPU thread (enc_executor_run) and wait for it; a call dropped
 *  because the client left while it was queued is logged.
 *  @tparam F a callable taking no arguments
 *  @param c the engine
 *  @param fd the client socket
 *  @param work the engine call
 *  @return false when the call was dropped unrun: the client is gone and no response is owed */
template <class F> bool on_gpu_thread(const enc_ctx *c, int fd, const F &work) {
    if (enc_executor_run(c->gpu, fd, work)) return true;
    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: %s: the client left while its request was queued; not run", c->id);
    return false;
}

/** Serve POST /v1/embeddings (the shape in the file comment): validate, render every input, one encode call, reply.
 *  @param c the embedder
 *  @param fd the client socket (a response or an error is sent unless the client left while the call was queued)
 *  @param o the request body */
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
    if (!read_dimensions(o.get("dimensions"), c->dim, &err)) {
        http_error(fd, 400, err.c_str());
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
    int rc = 0;
    const bool ran = on_gpu_thread(c, fd, [&] {
        rc = pulsar_engine_encode(c->e, toks.data(), (uint32_t)toks.size(), out.data(), why, sizeof why);
    });
    free_all(&toks);
    if (!ran) return;
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

/** Serve POST /v1/rerank: one (query, document) pair an input, one encode call; results sorted by logit, best
 *  first (stable: ties keep document order), cut to top_n.
 *  @param c the reranker
 *  @param fd the client socket (a response or an error is sent unless the client left while the call was queued)
 *  @param o the request body */
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
    int rc = 0;
    const bool ran = on_gpu_thread(c, fd, [&] {
        rc = pulsar_engine_encode(c->e, toks.data(), (uint32_t)toks.size(), logit.data(), why, sizeof why);
    });
    free_all(&toks);
    if (!ran) return;
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

/** Serve POST /v1/parse: decode the data: URL image, one pulsar_engine_parse_page call, reply with its text.
 *  An engine rc of -2 (the input) answers 400, any other failure 500.
 *  @param c the parser
 *  @param fd the client socket (a response or an error is sent unless the client left while the call was queued)
 *  @param o the request body */
void parse(const enc_ctx *c, int fd, const pyjson_value &o) {
    const pyjson_value *im = o.get("image");
    if (!im || im->kind != pyjson_value::STR) {
        http_error(fd, 400, "\"image\" must be an inline base64 data: URL (data:image/png;base64,...)");
        return;
    }
    const std::string *prompt = NULL;
    if (!opt_str(o, "prompt", &prompt)) {
        http_error(fd, 400, "\"prompt\" must be a string (the model's added tokens, back to back)");
        return;
    }
    uint32_t max_new = 0;
    if (const pyjson_value *k = o.get("max_tokens")) {
        if (k->kind != pyjson_value::INT || k->num[0] == '-' || k->num == "0" || k->num.size() > 9) {
            http_error(fd, 400, "\"max_tokens\" must be a positive integer");
            return;
        }
        max_new = (uint32_t)strtoul(k->num.c_str(), NULL, 10);
    }
    bool return_ids = false;
    if (const pyjson_value *v = o.get("return_ids")) {
        if (v->kind != pyjson_value::BOOL) {
            http_error(fd, 400, "\"return_ids\" must be a boolean");
            return;
        }
        return_ids = v->b;
    }
    char why[512];
    size_t n = 0;
    uint8_t *bytes = server_decode_image_data_url(im->s.c_str(), &n, why, sizeof why);
    if (!bytes) {
        http_error(fd, 400, why);
        return;
    }
    pulsar_parse_options po = {prompt ? prompt->c_str() : NULL, max_new};
    pulsar_tokens ids = {};
    char *text = NULL;
    pulsar_parse_stats st = {};
    int rc = 0;
    const bool ran = on_gpu_thread(c, fd, [&] {
        rc = pulsar_engine_parse_page(c->e, bytes, n, &po, &ids, &text, &st, why, sizeof why);
    });
    free(bytes);
    if (!ran) return;
    if (rc != 0) {
        pulsar_tokens_free(&ids);
        http_error(fd, rc == -2 ? 400 : 500, why);
        return;
    }
    buf b = {0};
    buf_puts(&b, "{\"object\":\"parse\",\"model\":");
    json_escape(&b, c->id);
    buf_puts(&b, ",\"text\":");
    json_escape(&b, text);
    buf_printf(&b, ",\"finish_reason\":\"%s\"", st.finished ? "stop" : "length");
    if (return_ids) {
        buf_puts(&b, ",\"ids\":[");
        for (int i = 0; i < ids.len; i++) buf_printf(&b, i ? ",%d" : "%d", ids.v[i]);
        buf_putc(&b, ']');
    }
    buf_printf(&b, ",\"image\":{\"width\":%u,\"height\":%u},\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,"
                   "\"total_tokens\":%u},\"timings\":{\"vision_ms\":%.1f,\"prefill_ms\":%.1f,\"decode_ms\":%.1f}}\n",
               st.image_w, st.image_h, st.prompt_tokens, st.generated_tokens, st.prompt_tokens + st.generated_tokens,
               st.vision_ms, st.prefill_ms, st.decode_ms);
    free(text);
    pulsar_tokens_free(&ids);
    send_json(fd, &b);
}

/** @param k an endpoint kind
 *  @return its short name ("embed", "rerank", "parse") */
const char *kind_name(srv_kind k) { return k == K_EMBED ? "embed" : k == K_RERANK ? "rerank" : "parse"; }
/** @param k an endpoint kind
 *  @return its canonical POST path */
const char *kind_path(srv_kind k) { return k == K_EMBED ? "/v1/embeddings" : k == K_RERANK ? "/v1/rerank" : "/v1/parse"; }
/* the endpoint a POST path names, K_COUNT for none */
/** @param p the request path
 *  @return the endpoint kind it names; K_COUNT for none */
srv_kind path_kind(const char *p) {
    return !strcmp(p, "/v1/embeddings") ? K_EMBED
           : !strcmp(p, "/v1/rerank") || !strcmp(p, "/rerank") ? K_RERANK
           : !strcmp(p, "/v1/parse") ? K_PARSE
                                      : K_COUNT;
}

/** Serve GET /health: every served model with its kind, dimension, token cap and endpoint.
 *  @param srv the server
 *  @param fd the client socket */
void health(const enc_server *srv, int fd) {
    buf b = {0};
    buf_puts(&b, "{\"status\":\"ok\",\"models\":[");
    for (size_t i = 0; i < srv->m.size(); i++) {
        const enc_ctx &c = srv->m[i];
        buf_puts(&b, i ? ",{\"model\":" : "{\"model\":");
        json_escape(&b, c.id);
        if (c.kind == K_PARSE) buf_printf(&b, ",\"parser\":true,\"endpoint\":\"%s\"}", kind_path(c.kind));
        else
            buf_printf(&b, ",\"encoder\":\"%s\",\"dim\":%u,\"max_tokens\":%u,\"endpoint\":\"%s\"}", kind_name(c.kind),
                       c.dim, c.cap, kind_path(c.kind));
    }
    buf_puts(&b, "]}\n");
    send_json(fd, &b);
}

/** Serve GET /v1/models (OpenAI's list shape).
 *  @param srv the server
 *  @param fd the client socket */
void models(const enc_server *srv, int fd) {
    buf b = {0};
    buf_puts(&b, "{\"object\":\"list\",\"data\":[");
    for (size_t i = 0; i < srv->m.size(); i++) {
        buf_puts(&b, i ? ",{\"id\":" : "{\"id\":");
        json_escape(&b, srv->m[i].id);
        buf_puts(&b, ",\"object\":\"model\",\"owned_by\":\"pulsar\"}");
    }
    buf_puts(&b, "]}\n");
    send_json(fd, &b);
}

/* the engine a request for `kind` goes to: the only one of that kind, else the one its "model" names; NULL + `err` */
/** @param srv the server
 *  @param kind the endpoint's kind (the caller checked one engine serves it)
 *  @param o the request body (its "model" is read only when several engines share the kind)
 *  @param err the 400 message when no engine matches
 *  @return the engine, or NULL */
const enc_ctx *route(const enc_server *srv, srv_kind kind, const pyjson_value &o, std::string *err) {
    std::vector<const enc_ctx *> of;
    for (const enc_ctx &c : srv->m)
        if (c.kind == kind) of.push_back(&c);
    if (of.size() == 1) return of[0];
    const pyjson_value *mv = o.get("model");
    for (const enc_ctx *c : of)
        if (mv && mv->kind == pyjson_value::STR && mv->s == c->id) return c;
    std::string ids;
    for (const enc_ctx *c : of) ids += std::string(ids.empty() ? "" : ", ") + c->id;
    *err = "\"model\" must name one of this server's " + std::string(kind_name(kind)) + " models: " + ids;
    return NULL;
}

/** A connection thread: read one HTTP request, serve it, close the socket, and drop enc_server::clients (waking
 *  encode_serve's drain at 0).
 *  @param arg an enc_arg (freed here)
 *  @return NULL */
void *client(void *arg) {
    enc_arg *a = (enc_arg *)arg;
    const enc_server *srv = a->srv;
    const int fd = a->fd;
    free(a);
    http_request hr = {};
    bool has[K_COUNT] = {};
    for (const enc_ctx &c : srv->m) has[c.kind] = true;
    if (!read_http_request(fd, &hr)) {
        http_error(fd, 400, "bad HTTP request");
    } else if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/health")) {
        health(srv, fd);
    } else if (!strcmp(hr.method, "GET") && (!strcmp(hr.path, "/healthz") || !strcmp(hr.path, "/ping"))) {
        http_response(fd, 200, "text/plain", "ok\n");
    } else if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/v1/models")) {
        models(srv, fd);
    } else if (!strcmp(hr.method, "POST") && path_kind(hr.path) != K_COUNT && has[path_kind(hr.path)]) {
        const srv_kind kind = path_kind(hr.path);
        pyjson_value o;
        char err[256];
        std::string why;
        const enc_ctx *c = NULL;
        if (!pyjson_parse(hr.body ? hr.body : "", hr.body_len, &o, err, sizeof err) || o.kind != pyjson_value::OBJ) {
            http_error(fd, 400, "the body is not a JSON object");
        } else if (!(c = route(srv, kind, o, &why))) {
            http_error(fd, 400, why.c_str());
        } else {
            const double t0 = server_now_sec();
            if (kind == K_EMBED) embeddings(c, fd, o);
            else if (kind == K_RERANK) rerank(c, fd, o);
            else parse(c, fd, o);
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: %s %s %.0f ms", hr.path, c->id, (server_now_sec() - t0) * 1e3);
        }
    } else {
        std::string eps;
        for (int k = 0; k < K_COUNT; k++)
            if (has[k]) eps += std::string("POST ") + kind_path((srv_kind)k) + ", ";
        http_error(fd, 404, ("this server serves " + eps + "GET /v1/models and /health").c_str());
    }
    http_request_free(&hr);
    close(fd);
    enc_server *ms = const_cast<enc_server *>(srv);
    pthread_mutex_lock(&ms->mu);
    if (--ms->clients == 0) pthread_cond_broadcast(&ms->idle);
    pthread_mutex_unlock(&ms->mu);
    return NULL;
}

/** End every engine's GPU thread (enc_executor_stop).  Called with no connection thread left.
 *  @param srv the server; each started engine's `gpu` is freed and cleared */
void stop_executors(enc_server *srv) {
    for (enc_ctx &c : srv->m)
        if (c.gpu) {
            enc_executor_stop(c.gpu);
            c.gpu = NULL;
        }
}

}  // namespace

int encode_serve(pulsar_engine *const *engines, int n, const server_config *cfg) {
    enc_server srv;
    for (int i = 0; i < n; i++) {
        pulsar_engine *e = engines[i];
        srv.m.push_back({e, NULL, pulsar_engine_is_parser(e) ? K_PARSE : (srv_kind)pulsar_engine_encoder(e),
                         pulsar_engine_encode_dim(e), pulsar_engine_encode_max_tokens(e), pulsar_engine_served_model_id(e)});
        for (int j = 0; j < i; j++)
            if (!strcmp(srv.m[j].id, srv.m[i].id)) {
                server_log(PULSAR_LOG_DEFAULT, "pulsar-server: two models serve as %s (-m %s and -m %s): a request "
                           "could not name one -- refusing", srv.m[i].id, cfg->models[j], cfg->models[i]);
                return 1;
            }
    }
    for (enc_ctx &c : srv.m)
        if (!(c.gpu = enc_executor_start())) {
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: the GPU thread of %s could not start", c.id);
            stop_executors(&srv);
            return 1;
        }
    const int lfd = listen_on(cfg->host, cfg->port);
    if (lfd < 0) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: failed to listen on %s:%d: %s", cfg->host, cfg->port,
                   strerror(errno));
        stop_executors(&srv);
        return 1;
    }
    g_listen_fd = lfd;
    for (const enc_ctx &c : srv.m)
        if (c.kind == K_PARSE) server_log(PULSAR_LOG_DEFAULT, "pulsar-server: %s: parser, POST %s", c.id, kind_path(c.kind));
        else
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: %s: %s, POST %s (%u floats an input, inputs up to %u tokens)",
                       c.id, c.kind == K_EMBED ? "embedder" : "reranker", kind_path(c.kind), c.dim, c.cap);
    server_warn_if_exposed(cfg->host);
    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: listening on http://%s:%d (pulsar %s)", cfg->host, cfg->port,
               PULSAR_VERSION_STR);
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
        a->srv = &srv;
        a->fd = fd;
        pthread_mutex_lock(&srv.mu);
        const bool at_cap = srv.clients >= PULSAR_SERVER_MAX_CLIENTS;   /* L354: the chat server's gate */
        if (!at_cap) srv.clients++;
        pthread_mutex_unlock(&srv.mu);
        if (at_cap) {
            free(a);
            http_error_retry(fd, 503, "too many connections", HTTP_RETRY_BUSY_S);
            close(fd);
            continue;
        }
        pthread_t th;
        if (pthread_create(&th, NULL, client, a) != 0) {
            pthread_mutex_lock(&srv.mu);
            srv.clients--;
            pthread_mutex_unlock(&srv.mu);
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
    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: shutdown requested, draining requests");
    pthread_mutex_lock(&srv.mu);
    while (srv.clients > 0) pthread_cond_wait(&srv.idle, &srv.mu);
    pthread_mutex_unlock(&srv.mu);
    stop_executors(&srv);   /* the GPU threads end here, before the caller closes the engines */
    return 0;
}

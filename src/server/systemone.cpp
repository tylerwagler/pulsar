/** @file
 *  L338 P2.4: /v1/systemone -- the runtime adapter's System 1 decisions (Gemma 4's JEV judge) on a reserved bank. */
/* systemone.cpp -- the JEV System 1 judge's endpoint (L338 P2.4).
 *
 * WHAT IT ANSWERS.  A request carries one `state` and some `questions`; each question is one prefill of the bare-v1
 * template the adapter was trained on (judge_config.json "template_version"):
 *
 *     <bos>[kind] K\n[state] S\n[question] Q\n[options]\nO_1\n...\nO_n\n[decision]:
 *
 * (a choice question's options lettered "A) ...").  The final-norm hidden of the last row (pulsar_session_copy_hidden)
 * meets the container's decision head on the host: z = softcap * tanh((W h + b) / softcap) over every slot, the
 * kind's slot range [lo, lo + n) taken, p = softmax(z / T_kind) with calibration_gold.json's per-kind temperature --
 * the reference's arithmetic (temp/jev-gemma4/ref/make_refs.py decide()), in double.  noul's options default to
 * false/true (p[1] = P(block)), score's to 0..5 (its expectation reported).  `state` as a string is used verbatim;
 * as an object it is serialized as the training data was, json.dumps(state, indent=1) (ASCII-escaped).
 *
 * WHERE IT RUNS (Tyler 2026-11-02: a reserved judge bank).  The bank pool's last bank is the judge's: the server's
 * chat pool (pool_banks) ends before it, so no chat route, eviction, adoption or disk spill ever sees it, and its
 * K / V are the adapter's alone (pulsar_session_set_bank_adapter at startup, on the empty bank).  The worker serves
 * the judge's queue between its decode quanta: each turn installs the judge bank (the chat bank installed before it
 * saved, as bank_switch saves), prefills at most one chunk of the current prompt, saves the bank and leaves
 * live_bank at the "nothing installed" sentinel, so the next bank_switch installs its target without a save.  A
 * prompt the bank already holds whole (an exact repeat) is cut one token short first: the head reruns on its last
 * row, never reading a stale hidden.
 *
 * The prefix the bank keeps is the previous prompt's: the bare-v1 template opens with the kind and the state, so
 * judgments share little beyond "<bos>[kind] ..." -- the reserved bank is about isolation, not reuse. */
#include "pulsar_server_internal.h"

#include "lib/pyjson.h"
#include "pulsar_lock.hpp"

#include <algorithm>

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

/// Tokens a worker turn prefills of a judge prompt (decode quanta run between).
static constexpr int JUDGE_CHUNK_TOKENS = 1024;

/// One decision kind: its head slots and calibration.
struct judge_kind {
    std::string name;   ///< "noul", "choice", "score"
    uint32_t lo = 0;    ///< first slot
    uint32_t hi = 0;    ///< one past the last
    double T = 1.0;     ///< calibration temperature (calibration_gold.json per_kind)
};

/// One question of a request.
struct judge_q {
    int kind = -1;                      ///< index into server_judge::kinds
    std::vector<std::string> options;   ///< the options as rendered (choice: unlettered)
};

/// One /v1/systemone request in the judge's queue (client thread waits on cv).
struct judge_job {
    std::vector<judge_q> qs;              ///< its questions
    std::vector<pulsar_tokens> prompts;   ///< their bare-v1 prompts, tokenized
    std::vector<std::vector<double>> probs;   ///< per question: the options' probabilities
    std::vector<int> cached;              ///< per question: tokens the bank already held
    std::string err;                      ///< a refusal, empty when it succeeded
    size_t at = 0;                        ///< the question in progress
    int pos = 0;                          ///< tokens of prompts[at] the bank holds
    bool started = false;                 ///< prompts[at]'s prefill has begun
    bool done = false;                    ///< the worker has finished with it
    int fd = -1;                          ///< the client socket (the worker's disconnect poll, L372)
    pthread_mutex_t mu;                   ///< guards done
    pthread_cond_t cv;                    ///< signalled when done
    judge_job *next = nullptr;            ///< next in the queue
};

/// The judge: its bank, head, calibration and queue (server::judge; NULL = no /v1/systemone).
struct server_judge {
    int bank = -1;                        ///< the reserved bank
    uint32_t slots = 0;                   ///< the head's slots
    uint32_t width = 0;                   ///< the hidden width
    const float *w = nullptr;             ///< head weight [slots][width] (the container's mapping)
    const float *b = nullptr;             ///< head bias [slots]
    double softcap = 30.0;                ///< judge_config.json softcap
    std::vector<judge_kind> kinds;        ///< the kinds the head answers
    judge_job *head = nullptr;            ///< queue head (under server::mu)
    judge_job *tail = nullptr;            ///< queue tail (under server::mu)
    uint64_t served = 0;                  ///< questions answered (worker)
};

namespace {

/// Read a whole file.
/// @param path the file
/// @param out receives its bytes
/// @return false when it cannot be read
bool judge_slurp(const std::string &path, std::string *out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out->append(buf, n);
    const bool ok = !ferror(f);
    fclose(f);
    return ok;
}

/// A JSON number (int or float) as a double.
/// @param v the value
/// @param out receives it
/// @return false when v is not a number
bool judge_num(const pyjson_value *v, double *out) {
    if (!v) return false;
    if (v->kind == pyjson_value::INT) { *out = strtod(v->num.c_str(), NULL); return true; }
    if (v->kind == pyjson_value::FLOAT) { *out = v->f; return true; }
    return false;
}

/// Append a JSON string as Python's json.dumps writes it with ensure_ascii (the default): the short escapes, other
/// controls and every non-ASCII codepoint as \\uXXXX (UTF-16, surrogate pairs past the BMP).
/// @param s UTF-8 text
/// @param out the literal is appended here
void judge_dump_ascii(const std::string &s, std::string *out) {
    static const char hx[] = "0123456789abcdef";
    auto u4 = [&](uint32_t c) {
        out->append("\\u");
        for (int k = 12; k >= 0; k -= 4) out->push_back(hx[(c >> k) & 15u]);
    };
    out->push_back('"');
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        uint32_t cp = c;
        size_t n = 1;
        if (c >= 0xf0) { cp = c & 7u; n = 4; }
        else if (c >= 0xe0) { cp = c & 15u; n = 3; }
        else if (c >= 0xc0) { cp = c & 31u; n = 2; }
        for (size_t k = 1; k < n && i + k < s.size(); k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 63u);
        i += n;
        if (cp == '"') out->append("\\\"");
        else if (cp == '\\') out->append("\\\\");
        else if (cp == '\n') out->append("\\n");
        else if (cp == '\r') out->append("\\r");
        else if (cp == '\t') out->append("\\t");
        else if (cp == '\b') out->append("\\b");
        else if (cp == '\f') out->append("\\f");
        else if (cp < 0x20 || (cp > 0x7f && cp < 0x10000)) u4(cp);   /* DEL (0x7f) is ASCII: kept */
        else if (cp >= 0x10000) {
            const uint32_t v = cp - 0x10000;
            u4(0xd800u + (v >> 10));
            u4(0xdc00u + (v & 0x3ffu));
        } else out->push_back((char)cp);
    }
    out->push_back('"');
}

/// Append json.dumps(v, indent=1) (Python's defaults otherwise: ensure_ascii, ", " -> "," + newline, ": ").
/// @param v the value
/// @param depth the nesting depth (the indent in spaces)
/// @param out the text is appended here
void judge_dump_indent1(const pyjson_value &v, int depth, std::string *out) {
    switch (v.kind) {
    case pyjson_value::NUL: out->append("null"); return;
    case pyjson_value::BOOL: out->append(v.b ? "true" : "false"); return;
    case pyjson_value::INT: case pyjson_value::FLOAT: pyjson_dump(v, out); return;
    case pyjson_value::STR: judge_dump_ascii(v.s, out); return;
    case pyjson_value::ARR:
        if (v.a.empty()) { out->append("[]"); return; }
        out->append("[");
        for (size_t i = 0; i < v.a.size(); i++) {
            out->append(i ? ",\n" : "\n");
            out->append((size_t)depth + 1u, ' ');
            judge_dump_indent1(v.a[i], depth + 1, out);
        }
        out->push_back('\n');
        out->append((size_t)depth, ' ');
        out->push_back(']');
        return;
    case pyjson_value::OBJ:
        if (v.o.empty()) { out->append("{}"); return; }
        out->append("{");
        for (size_t i = 0; i < v.o.size(); i++) {
            out->append(i ? ",\n" : "\n");
            out->append((size_t)depth + 1u, ' ');
            judge_dump_ascii(v.o[i].first, out);
            out->append(": ");
            judge_dump_indent1(v.o[i].second, depth + 1, out);
        }
        out->push_back('\n');
        out->append((size_t)depth, ' ');
        out->push_back('}');
        return;
    }
}

/// The kind's index, or -1.
/// @param J the judge
/// @param name the kind's name
/// @return its index
int judge_kind_index(const server_judge *J, const std::string &name) {
    for (size_t i = 0; i < J->kinds.size(); i++)
        if (J->kinds[i].name == name) return (int)i;
    return -1;
}

/// Fail a job and wake its client.
/// @param jb the job
/// @param why the reason
void judge_finish(judge_job *jb, const char *why) {
    if (why) jb->err = why;
    pthread_mutex_lock(&jb->mu);
    jb->done = true;
    pthread_cond_signal(&jb->cv);
    pthread_mutex_unlock(&jb->mu);
}

}  // namespace

bool server_judge_open(server *s, const char *container_dir, int bank, char *err, size_t errlen) {
    server_judge *J = new server_judge();
    J->bank = bank;
    J->slots = pulsar_engine_decision_head(s->engine, &J->w, &J->b, &J->width);
    if (!J->slots) {
        snprintf(err, errlen, "the container carries no decision head");
        delete J;
        return false;
    }
    std::string dir = container_dir ? container_dir : "", jc_text, cal_text;
    pyjson_value jc, cal;
    char perr[160];
    if (!judge_slurp(dir + "/judge_config.json", &jc_text) || !pyjson_parse(jc_text.data(), jc_text.size(), &jc, perr,
                                                                       sizeof perr)) {
        snprintf(err, errlen, "%s/judge_config.json is missing or not JSON", dir.c_str());
        delete J;
        return false;
    }
    if (!judge_slurp(dir + "/calibration_gold.json", &cal_text) ||
        !pyjson_parse(cal_text.data(), cal_text.size(), &cal, perr, sizeof perr)) {
        snprintf(err, errlen, "%s/calibration_gold.json is missing or not JSON", dir.c_str());
        delete J;
        return false;
    }
    const pyjson_value *sl = jc.get("slots");
    const pyjson_value *tv = sl && sl->get("template_version") ? sl->get("template_version") : jc.get("template_version");
    if (!tv || tv->kind != pyjson_value::STR || tv->s != "bare-v1") {
        snprintf(err, errlen, "judge_config.json's template_version is not bare-v1 (the one this endpoint renders)");
        delete J;
        return false;
    }
    double sc;
    if (judge_num(jc.get("softcap"), &sc)) J->softcap = sc;
    const pyjson_value *ranges = sl ? sl->get("ranges") : NULL;
    const pyjson_value *per = cal.get("per_kind");
    if (!ranges || ranges->kind != pyjson_value::OBJ || !per || per->kind != pyjson_value::OBJ) {
        snprintf(err, errlen, "judge_config.json slots.ranges or calibration_gold.json per_kind is missing");
        delete J;
        return false;
    }
    for (const auto &kv : ranges->o) {
        double lo, hi, T;
        if (kv.second.kind != pyjson_value::ARR || kv.second.a.size() != 2 || !judge_num(&kv.second.a[0], &lo) ||
            !judge_num(&kv.second.a[1], &hi) || lo < 0 || hi <= lo || hi > (double)J->slots ||
            !judge_num(per->get(kv.first.c_str()), &T) || !(T > 0.0)) {
            snprintf(err, errlen, "kind %s: its slot range or its calibration temperature is malformed",
                     kv.first.c_str());
            delete J;
            return false;
        }
        J->kinds.push_back({kv.first, (uint32_t)lo, (uint32_t)hi, T});
    }
    /* the bank: empty, the adapter on, saved -- nothing installed after (the worker installs what it serves) */
    pulsar_session *pool = s->sess;
    if (!pulsar_session_bank_state_restore(pool, (uint32_t)bank)) {
        snprintf(err, errlen, "bank %d could not be installed", bank);
        delete J;
        return false;
    }
    pulsar_session_invalidate(pool);
    if (!pulsar_session_set_bank_adapter(pool, (uint32_t)bank, true)) {
        snprintf(err, errlen, "bank %d refused the adapter", bank);
        delete J;
        return false;
    }
    pulsar_session_bank_state_save(pool, (uint32_t)bank);
    s->live_bank = -1;
    s->judge = J;
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: /v1/systemone on bank %d (reserved, the adapter on): decision head %u slots over %u, "
               "soft-cap %.0f, %zu kinds, bare-v1 (L338 P2.4)",
               bank, J->slots, J->width, J->softcap, J->kinds.size());
    return true;
}

void server_judge_close(server *s) {
    if (!s->judge) return;
    delete s->judge;
    s->judge = NULL;
}

/// A judge job waits (read under s->mu).
/// @param s the server
/// @return true when the worker has judge work
bool server_judge_queued(const server *s) { return s->judge && s->judge->head; }

bool server_judge_step(server *s) {
    server_judge *J = s->judge;
    if (!J) return false;
    judge_job *jb;
    {
        pulsar::ScopedLock lk(&s->mu);
        jb = J->head;
    }
    if (!jb) return false;
    auto pop = [&]() {
        pulsar::ScopedLock lk(&s->mu);
        J->head = jb->next;
        if (!J->head) J->tail = NULL;
    };
    if (s->stopping) {
        pop();
        judge_finish(jb, "the server is shutting down");
        return true;
    }
    /* L372: a client that is gone gets no prefill -- polled before each chunk, as a chat prefill's is between its own
     * (gen_prefill_cancel_cb).  Nothing is installed yet, and the bank keeps what the earlier chunks left: a prefix
     * its next request matches against like any other history. */
    if (gen_client_disconnected(jb->fd)) {
        pop();
        judge_finish(jb, "the client disconnected");
        return true;
    }
    pulsar_session *pool = s->sess;
    pulsar_stall_mark("worker: systemone");
    /* install the judge bank: the chat bank installed saved first (bank_switch's rule) */
    if (s->live_bank >= 0 && s->live_bank != J->bank) {
        s->slots[s->live_bank].committed_pos = pulsar_session_pos(pool);
        pulsar_session_bank_state_save(pool, (uint32_t)s->live_bank);
    }
    s->live_bank = -1;
    if (!pulsar_session_bank_state_restore(pool, (uint32_t)J->bank)) {
        pop();
        judge_finish(jb, "the judge bank could not be installed");
        return true;
    }
    char err[256] = "";
    const pulsar_tokens &P = jb->prompts[jb->at];
    bool ok = true;
    if (!jb->started) {
        jb->started = true;
        pulsar_prefix_match m;
        pulsar_session_prefix_match(pool, &P, &m);
        int have = m.prompt_cut.raw();
        jb->cached[jb->at] = have;
        if (have >= P.len) {   /* an exact repeat: one token short first, so the head reruns on the last row */
            pulsar_tokens cut = {P.v, P.len - 1, P.len - 1};
            ok = pulsar_session_sync(pool, &cut, err, sizeof err) == 0;
            have = P.len - 1;
        }
        jb->pos = have;
    }
    const int target = std::min(P.len, jb->pos + JUDGE_CHUNK_TOKENS);
    if (ok) {
        pulsar_tokens view = {P.v, target, target};
        ok = pulsar_session_sync(pool, &view, err, sizeof err) == 0;
    }
    bool finished = false;
    if (ok && target == P.len) {
        std::vector<float> h(J->width);
        ok = pulsar_session_copy_hidden(pool, h.data(), (int)J->width) == (int)J->width;
        if (!ok) snprintf(err, sizeof err, "the hidden could not be read");
        if (ok) {
            const judge_q &q = jb->qs[jb->at];
            const judge_kind &k = J->kinds[(size_t)q.kind];
            const size_t n = q.options.size();
            std::vector<double> z(n);
            double mx = -INFINITY;
            for (size_t o = 0; o < n; o++) {
                const uint32_t slot = k.lo + (uint32_t)o;
                const float *wr = J->w + (size_t)slot * J->width;
                double d = (double)J->b[slot];
                for (uint32_t c = 0; c < J->width; c++) d += (double)wr[c] * (double)h[c];
                z[o] = J->softcap * tanh(d / J->softcap) / k.T;
                mx = std::max(mx, z[o]);
            }
            double sum = 0.0;
            for (size_t o = 0; o < n; o++) sum += (z[o] = exp(z[o] - mx));
            for (size_t o = 0; o < n; o++) z[o] /= sum;
            jb->probs[jb->at] = std::move(z);
            J->served++;
            jb->at++;
            jb->started = false;
            finished = jb->at == jb->prompts.size();
        }
    } else if (ok) {
        jb->pos = target;
    }
    pulsar_session_bank_state_save(pool, (uint32_t)J->bank);   /* nothing installed after: live_bank stays -1 */
    if (!ok) {
        /* a failed sync leaves the bank's history unknown: empty it, the adapter kept */
        if (pulsar_session_bank_state_restore(pool, (uint32_t)J->bank)) {
            pulsar_session_invalidate(pool);
            pulsar_session_bank_state_save(pool, (uint32_t)J->bank);
        }
        pop();
        char why[320];
        snprintf(why, sizeof why, "the judge's prefill failed: %s", err[0] ? err : "unknown");
        judge_finish(jb, why);
        return true;
    }
    if (finished) {
        pop();
        judge_finish(jb, NULL);
    }
    return true;
}

/** L354: the questions one request may ask the judge.  Every question renders the whole state, so without a bound a body
 *  of small questions under one large state holds questions x state tokens and queues as many prefills. */
#define PULSAR_JUDGE_MAX_QUESTIONS 64
/// @param n_questions the questions asked
/// @return true when a request may ask that many
static bool judge_question_count_ok(size_t n_questions) { return n_questions <= PULSAR_JUDGE_MAX_QUESTIONS; }
/// @param total_tokens the request's rendered tokens so far
/// @param ctx          one bank's context
/// @return true when the request is within four contexts' worth
static bool judge_total_tokens_ok(uint64_t total_tokens, int ctx) {
    return total_tokens <= 4u * (uint64_t)(ctx > 0 ? ctx : 0);
}

void server_judge_http(server *s, int fd, const char *body, size_t len) {
    server_judge *J = s->judge;
    if (!J) {
        http_error(fd, 404, "/v1/systemone: this server has no judge (no adapter in the container, or --judge-banks 0)");
        return;
    }
    pyjson_value req;
    char perr[160];
    if (!pyjson_parse(body, len, &req, perr, sizeof perr) || req.kind != pyjson_value::OBJ) {
        http_error(fd, 400, "/v1/systemone: the body is not a JSON object");
        return;
    }
    const pyjson_value *st = req.get("state"), *qs = req.get("questions");
    if (!st || (st->kind != pyjson_value::STR && st->kind != pyjson_value::OBJ) || !qs ||
        qs->kind != pyjson_value::ARR || qs->a.empty()) {
        http_error(fd, 400, "/v1/systemone: wants \"state\" (a string, or an object serialized as json.dumps(state, "
                            "indent=1)) and a non-empty \"questions\" array");
        return;
    }
    if (!judge_question_count_ok(qs->a.size())) {
        char e[128];
        snprintf(e, sizeof e, "/v1/systemone: %zu questions; a request asks at most %d", qs->a.size(),
                 PULSAR_JUDGE_MAX_QUESTIONS);
        http_error(fd, 400, e);
        return;
    }
    std::string state;
    if (st->kind == pyjson_value::STR) state = st->s;
    else judge_dump_indent1(*st, 0, &state);
    judge_job *jb = new judge_job();
    jb->fd = fd;
    pthread_mutex_init(&jb->mu, NULL);
    pthread_cond_init(&jb->cv, NULL);
    auto free_job = [&]() {
        for (auto &p : jb->prompts) pulsar_tokens_free(&p);
        pthread_mutex_destroy(&jb->mu);
        pthread_cond_destroy(&jb->cv);
        delete jb;
    };
    uint64_t total_tokens = 0;
    for (size_t i = 0; i < qs->a.size(); i++) {
        const pyjson_value &qv = qs->a[i];
        const pyjson_value *kv = qv.get("kind"), *qq = qv.get("question"), *ov = qv.get("options");
        char e[256];
        judge_q q;
        if (!kv || kv->kind != pyjson_value::STR || (q.kind = judge_kind_index(J, kv->s)) < 0 || !qq ||
            qq->kind != pyjson_value::STR) {
            snprintf(e, sizeof e, "/v1/systemone: question %zu wants a \"kind\" the head answers and a \"question\" "
                                  "string", i);
            http_error(fd, 400, e);
            free_job();
            return;
        }
        const judge_kind &k = J->kinds[(size_t)q.kind];
        if (ov) {
            if (ov->kind != pyjson_value::ARR) { http_error(fd, 400, "/v1/systemone: \"options\" is not an array"); free_job(); return; }
            for (const auto &o : ov->a) {
                if (o.kind != pyjson_value::STR) { http_error(fd, 400, "/v1/systemone: an option is not a string"); free_job(); return; }
                q.options.push_back(o.s);
            }
        } else if (k.name == "noul") {
            q.options = {"false", "true"};
        } else if (k.name == "score") {
            for (uint32_t o = 0; o < k.hi - k.lo; o++) q.options.push_back(std::to_string(o));
        }
        if (q.options.size() < 2 || q.options.size() > k.hi - k.lo) {
            snprintf(e, sizeof e, "/v1/systemone: question %zu (%s) has %zu options; the head answers 2 .. %u", i,
                     k.name.c_str(), q.options.size(), k.hi - k.lo);
            http_error(fd, 400, e);
            free_job();
            return;
        }
        std::string text = "[kind] " + k.name + "\n[state] " + state + "\n[question] " + qq->s + "\n[options]\n";
        for (size_t o = 0; o < q.options.size(); o++) {
            if (o) text += "\n";
            if (k.name == "choice") text += std::string(1, (char)('A' + o)) + ") ";
            text += q.options[o];
        }
        text += "\n[decision]:";
        pulsar_tokens t = {0};
        if (!pulsar_tokenize_rendered_chat(s->engine, "<bos>", &t) || t.len != 1 ||
            !pulsar_tokenize_text(s->engine, text.c_str(), &t)) {
            pulsar_tokens_free(&t);
            http_error(fd, 500, "/v1/systemone: the prompt could not be tokenized");
            free_job();
            return;
        }
        if (t.len >= s->pool_ctx_size - 1) {
            snprintf(e, sizeof e, "/v1/systemone: question %zu is %d tokens; the context is %d", i, t.len,
                     s->pool_ctx_size);
            pulsar_tokens_free(&t);
            http_error(fd, 400, e);
            free_job();
            return;
        }
        total_tokens += (uint64_t)t.len;
        if (!judge_total_tokens_ok(total_tokens, s->pool_ctx_size)) {
            snprintf(e, sizeof e, "/v1/systemone: the questions render %llu tokens; a request holds at most %llu",
                     (unsigned long long)total_tokens, 4ull * (unsigned long long)s->pool_ctx_size);
            pulsar_tokens_free(&t);
            http_error(fd, 400, e);
            free_job();
            return;
        }
        jb->qs.push_back(std::move(q));
        jb->prompts.push_back(t);
    }
    jb->probs.resize(jb->prompts.size());
    jb->cached.assign(jb->prompts.size(), 0);
    const double t0 = server_now_sec();
    {
        pulsar::ScopedLock lk(&s->mu);
        if (J->tail) J->tail->next = jb;
        else J->head = jb;
        J->tail = jb;
        pthread_cond_signal(&s->cv);
    }
    pthread_mutex_lock(&jb->mu);
    while (!jb->done) pthread_cond_wait(&jb->cv, &jb->mu);
    pthread_mutex_unlock(&jb->mu);
    if (!jb->err.empty()) {
        http_error(fd, 503, jb->err.c_str());
        free_job();
        return;
    }
    std::string out = "{\"object\":\"systemone.decisions\",\"decisions\":[";
    int prompt_tokens = 0, cached = 0;
    for (size_t i = 0; i < jb->qs.size(); i++) {
        const judge_q &q = jb->qs[i];
        const judge_kind &k = J->kinds[(size_t)q.kind];
        const std::vector<double> &p = jb->probs[i];
        size_t best = 0;
        for (size_t o = 1; o < p.size(); o++)
            if (p[o] > p[best]) best = o;
        char num[64];
        out += i ? ",{" : "{";
        out += "\"kind\":";
        pyjson_dump_string(k.name.data(), k.name.size(), &out);
        out += ",\"options\":[";
        for (size_t o = 0; o < q.options.size(); o++) {
            if (o) out += ",";
            pyjson_dump_string(q.options[o].data(), q.options[o].size(), &out);
        }
        out += "],\"probs\":[";
        for (size_t o = 0; o < p.size(); o++) {
            snprintf(num, sizeof num, "%s%.6g", o ? "," : "", p[o]);
            out += num;
        }
        out += "],\"choice\":";
        pyjson_dump_string(q.options[best].data(), q.options[best].size(), &out);
        if (k.name == "noul") {
            snprintf(num, sizeof num, ",\"p_block\":%.6g", p[1]);
            out += num;
        } else if (k.name == "score") {
            double e = 0.0;
            for (size_t o = 0; o < p.size(); o++) e += (double)o * p[o];
            snprintf(num, sizeof num, ",\"score\":%.6g", e);
            out += num;
        }
        out += "}";
        prompt_tokens += jb->prompts[i].len;
        cached += jb->cached[i];
    }
    char tail[160];
    snprintf(tail, sizeof tail, "],\"usage\":{\"prompt_tokens\":%d,\"cached_tokens\":%d},\"ms\":%.1f}\n", prompt_tokens,
             cached, (server_now_sec() - t0) * 1e3);
    out += tail;
    http_response(fd, 200, "application/json", out.c_str());
    free_job();
}

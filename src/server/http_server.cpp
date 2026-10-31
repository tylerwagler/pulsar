/** @file
 *  The HTTP front door: request reading, the client threads' routing, /health, /version, /v1/models,
 *  /metrics and /metrics/stream.
 */
#include "pulsar_server_internal.h"
#include <time.h>
#include "pulsar_lock.hpp"
#include "pulsar_gpu.h"   /* tensor census for the reconciliation gauge */
#include "tp/pulsar_tp.h" /* the /health tp block reads the NODE records */



/// Free a request's body and zero it.
/// @param r  the request read by read_http_request
void http_request_free(http_request *r) {
    free(r->body);
    memset(r, 0, sizeof(*r));
}



/// Find the end of the request head: CRLFCRLF, else a bare LF LF.
/// @param p  the bytes read so far
/// @param n  bytes at `p`
/// @return the offset just past the blank line, or -1 when the head is not complete
static ssize_t header_end(const char *p, size_t n) {
    for (size_t i = 3; i < n; i++) {
        if (p[i - 3] == '\r' && p[i - 2] == '\n' && p[i - 1] == '\r' && p[i] == '\n') return (ssize_t)(i + 1);
    }
    for (size_t i = 1; i < n; i++) {
        if (p[i - 1] == '\n' && p[i] == '\n') return (ssize_t)(i + 1);
    }
    return -1;
}



/* Body length from the request headers. Returns -1 (reject) on a duplicate
 * Content-Length, a non-numeric value, or any Transfer-Encoding header: each
 * connection is Connection: close today so smuggling is only latent, but the
 * framing must stay unambiguous if keep-alive is ever added. */
/// The body length from the request head; rejects ambiguous framing.
/// @param h  the request head
/// @param n  bytes at `h`
/// @return the body length in bytes (0 when no Content-Length), or -1 to reject
static long content_length(const char *h, size_t n) {
    const char *p = h, *end = h + n;
    long clen = 0;
    int seen = 0;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r') len--;
        if (len >= 18 && strncasecmp(line, "Transfer-Encoding:", 18) == 0) {
            return -1;
        }
        if (len >= 15 && strncasecmp(line, "Content-Length:", 15) == 0) {
            const char *v = line + 15;
            while (v < line + len && isspace((unsigned char)*v)) v++;
            char *vend = NULL;
            long parsed = strtol(v, &vend, 10);
            if (vend == v || parsed < 0) return -1;
            if (seen && parsed != clen) return -1;
            seen = 1;
            clen = parsed;
        }
        if (p < end) p++;
    }
    return clen;
}



bool read_http_request(int fd, http_request *r) {
    buf b = {0};
    ssize_t hend = -1;
    const size_t max_header = 64 * 1024;
    const size_t max_body = 64 * 1024 * 1024;
    /* Whole-request read deadline: SO_RCVTIMEO only bounds a single recv, so
     * a client trickling one byte per interval could hold this thread
     * forever. The deadline bounds total arrival time for headers + body. */
    const time_t deadline = time(NULL) + PULSAR_SERVER_REQUEST_READ_DEADLINE_SEC;

    while (hend < 0 && b.len < max_header) {
        char tmp[4096];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        if (time(NULL) > deadline) goto fail;
        buf_append(&b, tmp, (size_t)n);
        hend = header_end(b.ptr, b.len);
    }
    if (hend < 0) goto fail;

    char line[512];
    size_t i;
    i = 0;
    while (i < b.len && b.ptr[i] != '\n' && i + 1 < sizeof(line)) {
        line[i] = b.ptr[i];
        i++;
    }
    line[i] = '\0';
    if (sscanf(line, "%7s %255s", r->method, r->path) != 2) goto fail;
    char *q;
    q = strchr(r->path, '?');
    if (q) *q = '\0';

    long clen;
    clen = content_length(b.ptr, (size_t)hend);
    if (clen < 0 || (size_t)clen > max_body) goto fail;
    while (b.len < (size_t)hend + (size_t)clen) {
        char tmp[8192];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        if (time(NULL) > deadline) goto fail;
        buf_append(&b, tmp, (size_t)n);
    }

    r->body_len = (size_t)clen;
    r->body = (char *)server_xmalloc(r->body_len + 1);
    memcpy(r->body, b.ptr + hend, r->body_len);
    r->body[r->body_len] = '\0';
    buf_free(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}



/// Append one /v1/models item (vLLM / OpenRouter shape: id, root, name, context, top_provider, parameters).
/// @param b               the JSON being built
/// @param id              the model id (also "root")
/// @param name            the display name
/// @param ctx             the context length, tokens
/// @param default_tokens  the generation cap when a request sets none; max_completion_tokens = min(it, ctx)
void append_model_json_values(buf *b, const char *id, const char *name,
                                     int ctx, int default_tokens) {
    const int max_completion = default_tokens < ctx ? default_tokens : ctx;
    buf_printf(b,
        "{\"id\":");
    json_escape(b, id);
    /* vLLM convention: "root" is the model path HF-tooling resolves (e.g.
     * llama-benchy uses it/the id as the tokenizer id). id == root here. */
    buf_puts(b, ",\"root\":");
    json_escape(b, id);
    buf_puts(b,
        ",\"object\":\"model\","
        "\"created\":1767225600,"
        "\"owned_by\":\"pulsar\","
        "\"name\":");
    json_escape(b, name);
    buf_printf(b,
        ","
        "\"context_length\":%d,"
        "\"top_provider\":{"
            "\"context_length\":%d,"
            "\"max_completion_tokens\":%d,"
            "\"is_moderated\":false},"
        "\"supported_parameters\":["
            "\"tools\","
            "\"tool_choice\","
            "\"max_tokens\","
            "\"temperature\","
            "\"top_p\","
            "\"top_k\","
            "\"min_p\","
            "\"stop\","
            "\"seed\","
            "\"stream\","
            "\"reasoning_effort\"]}",
        ctx,
        ctx,
        max_completion);
}



/// append_model_json_values for the loaded model (its served name, the session's context, default_tokens).
/// @param b   the JSON being built
/// @param s   the server
/// @param id  the model id to advertise
static void append_model_json(buf *b, const server *s, const char *id) {
    append_model_json_values(b,
                             id,
                             server_served_model_name(s),
                             pulsar_session_ctx(s->sess).raw(),
                             s->default_tokens);
}



bool server::send_model(int fd, const char *id) {
    auto *s = this;
    buf b = {0};
    append_model_json(&b, s, id);
    buf_putc(&b, '\n');
    bool ok = http_response(fd, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}



bool server::send_models(int fd) {
    auto *s = this;
    /* Advertise only the model actually loaded (its shape id), not both
     * flash/pro aliases — the server serves one GGUF at a time. */
    buf b = {0};
    buf_puts(&b, "{\"object\":\"list\",\"data\":[");
    append_model_json(&b, s, server_served_model_id(s));
    buf_puts(&b, "]}\n");
    bool ok = http_response(fd, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

/* Liveness probe (/healthz, /ping): is the process alive at all? Always 200
 * while the process runs — deliberately independent of readiness/drain state,
 * so a k8s liveness probe never restarts a server that is merely draining.
 * Lock-free, engine-free (safe on a client thread). */
bool server::send_liveness(int fd) {
    return http_response(fd, 200, "application/json",
                         "{\"status\":\"ok\"}\n");
}

/* One rank's fixed /health fields, left open for the live `connected`. */
/// One rank's fixed /health tp fields as a JSON object left open for the live `connected`.
/// @param n  the rank's node record
/// @return the fragment (malloc'd, without its closing brace), or NULL on allocation failure
static char *tp_rank_fragment(const pulsar_tp_node *n) {
    buf b = {0};
    buf_printf(&b, "{\"rank\":%d,\"host\":", n->rank);
    json_escape(&b, n->host);
    /* Empty means "this rank did not say", which is absent, not "". */
    const struct { const char *key; const char *val; } strs[] = {
        { "addr", n->addr }, { "version", n->build }, { "rdma_device", n->rdma_device },
    };
    for (const auto &kv : strs) {
        buf_printf(&b, ",\"%s\":", kv.key);
        if (kv.val[0]) json_escape(&b, kv.val);
        else buf_puts(&b, "null");
    }
    if (n->rdma_port > 0) buf_printf(&b, ",\"rdma_port\":%d", n->rdma_port);
    else buf_puts(&b, ",\"rdma_port\":null");
    return b.ptr;
}

/* The /health tp block's fixed part.  Called once, single-threaded, after the
 * engine is open and before any client thread exists (cli_main).  A server that
 * is not tensor-parallel still publishes itself as a one-rank group, so a
 * dashboard shows its GPU without a separate code path.  A rank the transport
 * has no NODE record for is left out rather than invented; nranks still says
 * how many there should be. */
void server::build_tp_health() {
    tp = pulsar_engine_tp(engine);
    const int n = tp ? (int)pulsar_tp_n_ranks(tp) : 1;
    tp_nranks = n;
    tp_self_rank = tp ? pulsar_tp_rank(tp) : 0;
    tp_transport = tp ? (pulsar_tp_is_rdma(tp) ? "rdma" : "tcp") : NULL;
    tp_rank_json = (char **)calloc((size_t)n, sizeof(char *));
    tp_rank_ids = (int *)calloc((size_t)n, sizeof(int));
    tp_rank_json_n = 0;
    if (!tp_rank_json || !tp_rank_ids) return;
    for (int r = 0; r < n; r++) {
        pulsar_tp_node node;
        if (tp) {
            if (!pulsar_tp_node_info(tp, r, &node)) continue;
        } else {
            memset(&node, 0, sizeof(node));
            if (gethostname(node.host, sizeof(node.host)) != 0) node.host[0] = '\0';
            node.host[sizeof(node.host) - 1] = '\0';
            snprintf(node.build, sizeof(node.build), "%s", PULSAR_VERSION_STR);
        }
        char *frag = tp_rank_fragment(&node);
        if (!frag) continue;
        tp_rank_ids[tp_rank_json_n] = r;
        tp_rank_json[tp_rank_json_n++] = frag;
    }
}

/* Readiness + status (/health): is the server ready to accept work, and what
 * is it doing right now? 200 {"status":"ok",...} when serving; 503
 * {"status":"draining",...} once shutdown has been requested so a load
 * balancer stops routing to it. Reads only the worker-published snapshot under
 * mu (same discipline as /metrics — no engine calls on the client thread). */
bool server::send_health(int fd) {
    auto *s = this;
    const char *model = server_served_model_id(s);
    bool draining;
    int n_slots, running, waiting, capacity;
    time_t started;
    double kv = 0.0; /* max KV utilization across provisioned slots */
    pthread_mutex_lock(&s->mu);
    draining = s->stopping;
    n_slots  = s->n_slots;
    /* The pool the run was sized for, not the number of banks provisioned so
     * far. Banks 1..pool_banks-1 are provisioned lazily as conversations
     * arrive, so `total` ramps from 1 while this does not move — and a reader
     * that wants to reserve space for the pool (pulsar-gui draws one row per
     * slot) needs the number that will not change under it. Classic mode is a
     * pool of one. */
    capacity = s->pool_banks > 0 ? s->pool_banks : 1;
    running  = s->n_generating;
    waiting  = s->n_queued;
    started  = s->started;
    for (int i = 0; i < n_slots; i++) {
        const int pos = s->m_slot_pos[i];
        const int ctx = s->m_slot_ctx[i];
        double u = (ctx > 0 && pos > 0) ? (double)pos / (double)ctx : 0.0;
        if (u > 1.0) u = 1.0;
        if (u > kv) kv = u;
    }
    pthread_mutex_unlock(&s->mu);
    if (running < 0) running = 0;
    if (waiting < 0) waiting = 0;
    const long uptime = started ? (long)(time(NULL) - started) : 0;
    /* the stall watchdog's verdict (watchdog.cpp): a stalled worker is not serving, whatever the queue says */
    const long long stalled = (long long)__atomic_load_n(&s->stall_s, __ATOMIC_RELAXED);

    buf b = {0};
    buf_printf(&b,
        "{\"status\":\"%s\",\"version\":\"%s\",\"model\":\"%s\","
        "\"uptime_s\":%ld,\"stalled\":%lld,\"slots\":{\"total\":%d,\"running\":%d,\"waiting\":%d,"
        "\"capacity\":%d},"
        "\"kv_cache_usage\":%.6f,",
        draining ? "draining" : stalled > 0 ? "stalled" : "ok", PULSAR_VERSION_STR, model,
        uptime, stalled, n_slots, running, waiting, capacity, kv);
    /* The tp block (pulsar-gui docs/contract.md, "Upstream: the /health tp
     * block").  Only `failed` and `connected` are live, and both come from one
     * atomic load; everything else was fixed at bring-up.  Every rank but this
     * one is connected exactly while the transport has not failed: the
     * transport marks itself failed on any peer loss (keepalive included), and
     * it has no per-peer notion finer than that. */
    const bool failed = tp && pulsar_tp_failed(tp);
    buf_printf(&b, "\"tp\":{\"nranks\":%d,\"transport\":", tp_nranks);
    if (tp_transport) buf_printf(&b, "\"%s\"", tp_transport);
    else buf_puts(&b, "null");
    buf_printf(&b, ",\"failed\":%s,\"ranks\":[", failed ? "true" : "false");
    for (int i = 0; i < tp_rank_json_n; i++) {
        if (i) buf_putc(&b, ',');
        buf_puts(&b, tp_rank_json[i]);
        /* tp_rank_ids, not i: a rank with no record is skipped, so the i-th
         * fragment is not necessarily rank i. */
        if (tp && tp_rank_ids[i] != tp_self_rank)
            buf_printf(&b, ",\"connected\":%s", failed ? "false" : "true");
        buf_putc(&b, '}');
    }
    buf_puts(&b, "]}}\n");
    bool ok = draining || stalled > 0
        ? http_response_retry(fd, 503, "application/json", b.ptr, HTTP_RETRY_GOING_AWAY_S)
        : http_response(fd, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

/* Version + build identity (/version), vLLM/OpenAI convention. Version is the
 * git-describe string baked in at build time (see Makefile). */
bool server::send_version(int fd) {
    auto *s = this;
    buf b = {0};
    buf_printf(&b,
        "{\"version\":\"%s\",\"engine\":\"pulsar\",\"cuda_arch\":\"sm_120f\","
        "\"model\":\"%s\",\"model_name\":\"%s\",\"context\":%d}\n",
        PULSAR_VERSION_STR,
        server_served_model_id(s),
        server_served_model_name(s),
        pulsar_session_ctx(s->sess));
    bool ok = http_response(fd, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

/* Root banner so a bare GET / (browsers, uptime probes) gets a 200 with the
 * version and a pointer to the real endpoints instead of a 404. */
bool server::send_root(int fd) {
    buf b = {0};
    buf_printf(&b,
        "{\"service\":\"pulsar-server\",\"version\":\"%s\",\"status\":\"ok\","
        "\"endpoints\":[\"/health\",\"/version\",\"/v1/models\","
        "\"/v1/chat/completions\",\"/v1/completions\",\"/v1/messages\","
        "\"/v1/messages/count_tokens\",\"/v1/responses\",\"/metrics\","
        "\"/metrics/stream\"]}\n",
        PULSAR_VERSION_STR);
    bool ok = http_response(fd, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

/* The phase label set for pulsar:slot_phase. GEN_DECODE_INIT folds into
 * "decode" (it is the transient re-init of a decode attempt, not a state an
 * operator cares to see) and GEN_FINISH/GEN_DONE fold into "finish". Every
 * slot reports all six series with a 0/1 value — the Prometheus state-set
 * convention — so the series set is stable and label values never churn. */
/// The /metrics slot-phase label set, indexed by slot_phase_index().
static const char *const slot_phase_names[] = {
    "idle", "prefill_cold", "prefill_main", "decode", "finish", "restore",
};
enum { SLOT_PHASE_COUNT = 6 };

/* Map a published m_slot_phase (gen_phase + 1; 0 = no bound job) onto that
 * label set. */
/// Map a published m_slot_phase (gen_phase + 1; 0 = no job) to its label index.
/// @param phase_plus_one  a published server::m_slot_phase value
/// @return an index into slot_phase_names
static int slot_phase_index(int phase_plus_one) {
    switch (phase_plus_one - 1) {
    case GEN_RESTORE:      return 5;   /* L330: a disk chain restoring, a slice per worker turn */
    case GEN_PREFILL_COLD: return 1;
    case GEN_PREFILL_MAIN: return 2;
    case GEN_DECODE_INIT:
    case GEN_DECODE:       return 3;
    case GEN_FINISH:
    case GEN_DONE:         return 4;
    default:               return 0; /* 0, or anything unrecognised, is idle */
    }
}

/* Histogram bucket bounds. Declared in pulsar_server_internal.h and defined
 * here, beside the emitter that prints them as le= labels, so the bounds the
 * observer uses and the ones the scrape advertises cannot drift apart.
 * The `extern` is load-bearing: a bare `const` array at namespace scope has
 * internal linkage in C++, which would give every TU its own copy. */
/// Latency histogram bucket upper bounds, seconds (TTFT, TPOT, end-to-end).
extern const double pulsar_hist_seconds_bounds[PULSAR_HIST_BUCKETS] = {
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0,
};
/// Token-count histogram bucket upper bounds, tokens.
extern const double pulsar_hist_tokens_bounds[PULSAR_HIST_BUCKETS] = {
    16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
};

/* Emit one histogram in Prometheus text form. Buckets are stored
 * non-cumulatively, so they are accumulated on the way out; +Inf is the total
 * count, which also covers observations above the last bound. */
/// Emit one histogram in Prometheus text form (cumulative buckets, +Inf, sum, count).
/// @param b       the exposition being built
/// @param name    the metric's name
/// @param help    its HELP text
/// @param h       the histogram (non-cumulative buckets)
/// @param bounds  its bucket upper bounds (PULSAR_HIST_BUCKETS of them)
static void emit_histogram(buf *b, const char *name, const char *help,
                           const pulsar_hist *h, const double *bounds) {
    buf_printf(b, "# HELP %s %s\n", name, help);
    buf_printf(b, "# TYPE %s histogram\n", name);
    uint64_t cum = 0;
    for (int i = 0; i < PULSAR_HIST_BUCKETS; i++) {
        cum += h->bucket[i];
        buf_printf(b, "%s_bucket{le=\"%g\"} %llu\n", name, bounds[i],
                   (unsigned long long)cum);
    }
    buf_printf(b, "%s_bucket{le=\"+Inf\"} %llu\n", name, (unsigned long long)h->count);
    buf_printf(b, "%s_sum %.6f\n", name, h->sum);
    buf_printf(b, "%s_count %llu\n", name, (unsigned long long)h->count);
}

/* Label for each provisioning refusal, indexed by provision_refusal. Only the
 * first two are relieved by eviction; mem_floor means the box itself is tight,
 * which is a different operator action entirely. */
/// The /metrics label per provision_refusal.
static const char *const refusal_names[PROVISION_REFUSAL_COUNT] = {
    "none", "pool_full", "ledger_full", "mem_floor", "create_fail", "kv_touched",
};

/* Prometheus /metrics — DSpark speculative-decode counters in vLLM naming, so
 * tool-eval-bench --spec-live (and any vLLM-oriented scraper) reads acceptance
 * rate, acceptance length, and the per-position waterfall unchanged. All
 * counters are cumulative since engine open; gauges are point-in-time.
 *
 * Series in the pulsar: namespace are additions with no vLLM equivalent; a
 * vLLM-oriented scraper ignores them. */
bool server::send_metrics(int fd) {
    auto *s = this;
    /* This runs on a client thread: it must not call into the engine
     * (CUDA-state audit, pulsar_server_internal.h). Everything below reads the
     * snapshots the worker publishes under mu (m_spec/m_slot_pos/m_slot_ctx/
     * m_slot_phase/m_slot_prefill_*, server_publish_metrics_snapshot —
     * refreshed at bind time and once per quantum, so gauges lag live state by
     * at most one quantum). */
    const char *model = server_served_model_id(s);
    pulsar_spec_metrics m;
    int n_slots;
    double slot_kv[PULSAR_SESSION_POOL_CAP];
    int slot_pos[PULSAR_SESSION_POOL_CAP];
    int slot_ctx[PULSAR_SESSION_POOL_CAP];
    int slot_phase[PULSAR_SESSION_POOL_CAP];
    int slot_depth[PULSAR_SESSION_POOL_CAP];
    int slot_pf_done[PULSAR_SESSION_POOL_CAP];
    int slot_pf_total[PULSAR_SESSION_POOL_CAP];
    pthread_mutex_lock(&s->mu);
    m = s->m_spec;
    const pulsar_verify_curve_view vc = s->m_verify_curve;
    const pulsar_spec_calibration cal = s->m_spec_calib;
    const char *drafter = s->m_drafter_name ? s->m_drafter_name : "";
    n_slots = s->n_slots;
    for (int i = 0; i < n_slots; i++) {
        const int pos = s->m_slot_pos[i];
        const int ctx = s->m_slot_ctx[i];
        double kv = (ctx > 0 && pos > 0) ? (double)pos / (double)ctx : 0.0;
        slot_kv[i] = kv > 1.0 ? 1.0 : kv;
        slot_pos[i] = pos > 0 ? pos : 0;
        slot_ctx[i] = ctx > 0 ? ctx : 0;
        slot_phase[i] = s->m_slot_phase[i];
        slot_depth[i] = s->m_slot_depth[i];
        /* prefill_last_current is -1 until the first progress callback; report
         * 0 tokens done rather than leaking the sentinel. */
        slot_pf_done[i] = s->m_slot_prefill_done[i] > 0 ? s->m_slot_prefill_done[i] : 0;
        slot_pf_total[i] = s->m_slot_prefill_total[i] > 0 ? s->m_slot_prefill_total[i] : 0;
    }
    int running = s->n_generating;
    int waiting = s->n_queued;
    unsigned long long prompt_toks = (unsigned long long)s->m_prompt_tokens;
    unsigned long long pfx_queries = (unsigned long long)s->m_prefix_queries;
    unsigned long long pfx_hits = (unsigned long long)s->m_prefix_hits;
    const pulsar_hist h_ttft = s->m_h_ttft;
    const pulsar_hist h_tpot = s->m_h_tpot;
    const pulsar_hist h_e2e = s->m_h_e2e;
    const pulsar_hist h_prompt_tok = s->m_h_prompt_tok;
    const pulsar_hist h_gen_tok = s->m_h_gen_tok;
    const unsigned long long reqs_finished = (unsigned long long)s->m_requests_finished;
    const unsigned long long evictions = (unsigned long long)s->m_evictions;
    const unsigned long long spills = (unsigned long long)s->m_spills;
    const unsigned long long restores = (unsigned long long)s->m_restores;
    const unsigned long long restore_failures = (unsigned long long)s->m_restore_failures;
    const unsigned long long restore_slices = (unsigned long long)s->m_restore_slices;    /* L330 */
    const double restore_stall_max_s = s->m_restore_stall_max_ms / 1000.0;
    const unsigned long long kv_touched = (unsigned long long)s->m_kv_touched.raw();      /* L328 */
    const unsigned long long kv_reserved = (unsigned long long)s->m_kv_reserved.raw();
    const unsigned long long kv_budget_live = (unsigned long long)s->m_kv_budget.raw();   /* 2026-10-29 */
    const unsigned long long persist_skipped = (unsigned long long)s->m_persist_skipped;
    const unsigned long long prefill_mem_refusals = (unsigned long long)s->m_prefill_mem_refusals;
    const kv_restore_meter disk_meter = s->kv.meter;   /* L310 */
    const unsigned long long preemptions = (unsigned long long)s->m_preemptions;    /* L314 */
    const unsigned long long readmissions = (unsigned long long)s->m_readmissions;
    const kv_dup_census dup = s->m_dup;                                              /* L324 */
    const unsigned long long dup_censuses = (unsigned long long)s->m_dup_censuses;
    const unsigned long long dup_bytes_peak = (unsigned long long)s->m_dup_bytes_peak.raw();
    unsigned long long prefix_waits[PREFIX_WAIT_OUTCOMES];                          /* L329 (a) */
    for (int i = 0; i < PREFIX_WAIT_OUTCOMES; i++) prefix_waits[i] = (unsigned long long)s->m_prefix_waits[i];
    const unsigned long long prefix_wait_gain = (unsigned long long)s->m_prefix_wait_gain;
    uint64_t refusals[PROVISION_REFUSAL_COUNT];
    for (int i = 0; i < PROVISION_REFUSAL_COUNT; i++) refusals[i] = s->m_refusals[i];
    const int block_reason = s->m_queue_block_reason;
    const unsigned long long ledger_committed = (unsigned long long)s->kv_committed_bytes;
    const unsigned long long ledger_budget = (unsigned long long)s->kv_budget_bytes;
    const unsigned long long gen_tokens = (unsigned long long)s->m_gen_tokens;
    const unsigned long long prefill_chunk_tokens =
        (unsigned long long)s->m_prefill_chunk_tokens;
    const int decode_lane = s->m_decode_lane;
    double phase_ms[SERVER_ROUND_PHASE_COUNT];
    memcpy(phase_ms, s->m_spec_phase_ms, sizeof phase_ms);
    const unsigned long long phase_rounds = (unsigned long long)s->m_spec_phase_rounds;
    uint64_t round_syncs[4];
    memcpy(round_syncs, s->m_spec_syncs, sizeof round_syncs);
    pthread_mutex_unlock(&s->mu);
    if (running < 0) running = 0;
    if (waiting < 0) waiting = 0;
    double kv = 0.0; /* unlabeled gauge: max across provisioned slots */
    for (int i = 0; i < n_slots; i++) {
        if (slot_kv[i] > kv) kv = slot_kv[i];
    }

    buf b = {0};
    /* Spec-decode counters (the core of --spec-live). model_name label lets the
     * scraper detect the drafter identity and the spec_decode method. */
    buf_puts(&b, "# HELP vllm:spec_decode_num_draft_tokens_total Cumulative draft tokens proposed.\n");
    buf_puts(&b, "# TYPE vllm:spec_decode_num_draft_tokens_total counter\n");
    buf_printf(&b, "vllm:spec_decode_num_draft_tokens_total{model_name=\"%s\"} %llu\n",
               model, (unsigned long long)m.draft_tokens);
    buf_puts(&b, "# HELP vllm:spec_decode_num_accepted_tokens_total Cumulative draft tokens accepted.\n");
    buf_puts(&b, "# TYPE vllm:spec_decode_num_accepted_tokens_total counter\n");
    buf_printf(&b, "vllm:spec_decode_num_accepted_tokens_total{model_name=\"%s\"} %llu\n",
               model, (unsigned long long)m.accepted_tokens);
    buf_puts(&b, "# HELP vllm:spec_decode_num_drafts_total Cumulative draft rounds.\n");
    buf_puts(&b, "# TYPE vllm:spec_decode_num_drafts_total counter\n");
    buf_printf(&b, "vllm:spec_decode_num_drafts_total{model_name=\"%s\"} %llu\n",
               model, (unsigned long long)m.num_drafts);
    /* Per-position accepted counters -> the scraper derives per-position
     * acceptance = count/num_drafts (the waterfall). Emit 0..max_draft-1 so the
     * chart has a full row even before every position has fired.
     *
     * ⚠ count/num_drafts is the vLLM convention and stays for compatibility,
     * but it is only a per-position RATE at a fixed draft depth. We run the
     * L107 adaptive depth, so a position is absent from rounds that never
     * reached it and the quotient is a joint prefix probability the schedule
     * moves. pulsar:spec_decode_num_verified_tokens_per_pos_total below is the
     * attempt count; the true per-position rate is the ratio of the two, and a
     * non-monotone vllm waterfall cannot be read as drafter quality on its own
     * (pulsar.h's pulsar_spec_metrics comment has the derivation). */
    if (m.max_draft > 0) {
        int np = m.max_draft > 16 ? 16 : m.max_draft;
        buf_puts(&b, "# HELP vllm:spec_decode_num_accepted_tokens_per_pos_total Accepted count per draft position.\n");
        buf_puts(&b, "# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_total counter\n");
        for (int i = 0; i < np; i++)
            buf_printf(&b, "vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name=\"%s\",position=\"%d\"} %llu\n",
                       model, i, (unsigned long long)m.accepted_per_pos[i]);
        buf_puts(&b, "# HELP pulsar:spec_decode_num_verified_tokens_per_pos_total Rounds that actually verified draft position i.\n");
        buf_puts(&b, "# TYPE pulsar:spec_decode_num_verified_tokens_per_pos_total counter\n");
        for (int i = 0; i < np; i++)
            buf_printf(&b, "pulsar:spec_decode_num_verified_tokens_per_pos_total{model_name=\"%s\",position=\"%d\"} %llu\n",
                       model, i, (unsigned long long)m.verified_per_pos[i]);
    }
    /* Token-throughput counters (the scraper derives prompt/gen t/s from deltas). */
    buf_puts(&b, "# HELP vllm:prompt_tokens_total Cumulative prompt tokens prefilled.\n");
    buf_puts(&b, "# TYPE vllm:prompt_tokens_total counter\n");
    buf_printf(&b, "vllm:prompt_tokens_total %llu\n", prompt_toks);
    /* Counted at gen_emit_token, NOT from the engine's spec_gen_tokens: the
     * latter only advances inside the DSpark fused verify loop, and this
     * metric read zero throughput on a busy server before the emit-path
     * counter existed. */
    buf_puts(&b, "# HELP vllm:generation_tokens_total Cumulative tokens emitted, all decode lanes.\n");
    buf_puts(&b, "# TYPE vllm:generation_tokens_total counter\n");
    buf_printf(&b, "vllm:generation_tokens_total %llu\n", gen_tokens);
    /* L114: chunk-granular prefill counter — advances per ~4s prefill chunk
     * (computed rows only), unlike vllm:prompt_tokens_total which advances at
     * request finish. Sample this to build prefill rate curves. */
    buf_puts(&b, "# HELP pulsar:prefill_chunk_tokens_total Computed prefill rows, accumulated per chunk.\n");
    buf_puts(&b, "# TYPE pulsar:prefill_chunk_tokens_total counter\n");
    buf_printf(&b, "pulsar:prefill_chunk_tokens_total %llu\n", prefill_chunk_tokens);
    buf_puts(&b, "# HELP pulsar:spec_decode_gen_tokens_total Tokens emitted by the fused spec loop only.\n");
    buf_puts(&b, "# TYPE pulsar:spec_decode_gen_tokens_total counter\n");
    buf_printf(&b, "pulsar:spec_decode_gen_tokens_total %llu\n", (unsigned long long)m.gen_tokens);
    /* L308: a quenched copy lane's probes (its estimate went stale; session_spec.cpp pulsar_spec_copy_price) */
    buf_puts(&b, "# HELP pulsar:spec_copy_probes_total Copy chains drafted to re-probe a stale copy estimate under tau.\n");
    buf_puts(&b, "# TYPE pulsar:spec_copy_probes_total counter\n");
    buf_printf(&b, "pulsar:spec_copy_probes_total %llu\n", (unsigned long long)m.copy_probes);
    /* L136 / L301: what the verify pick decides.  overflow_rounds = rounds whose row demand exceeded the
     * budget; pick_cut_rows = pending drafts the pick left unverified. */
    buf_puts(&b, "# HELP pulsar:spec_overflow_rounds_total Spec sweeps whose row demand exceeded the shared budget.\n");
    buf_puts(&b, "# TYPE pulsar:spec_overflow_rounds_total counter\n");
    buf_printf(&b, "pulsar:spec_overflow_rounds_total %llu\n",
               (unsigned long long)s->m_spec_overflow_rounds);
    buf_puts(&b, "# HELP pulsar:spec_pick_cut_rows_total Pending drafts the verify pick left unverified.\n");
    buf_puts(&b, "# TYPE pulsar:spec_pick_cut_rows_total counter\n");
    buf_printf(&b, "pulsar:spec_pick_cut_rows_total %llu\n",
               (unsigned long long)s->m_spec_pick_cut_rows);
    buf_puts(&b, "# HELP pulsar:spec_grammar_cut_rows_total Pending drafts cut at the grammar's first refusal before "
                 "the pick (L309).\n");
    buf_puts(&b, "# TYPE pulsar:spec_grammar_cut_rows_total counter\n");
    buf_printf(&b, "pulsar:spec_grammar_cut_rows_total %llu\n",
               (unsigned long long)s->m_spec_grammar_cut_rows);
    /* L301: the verify curve the pick prices with (verify_curve.cpp): the profiled forward per row count, the
     * online correction per row, and the round's measured wall outside the forward. */
    if (vc.valid) {
        buf_puts(&b, "# HELP pulsar:spec_verify_curve_ms Profiled verify forward wall at this many rows, ms.\n");
        buf_puts(&b, "# TYPE pulsar:spec_verify_curve_ms gauge\n");
        for (uint32_t r = 1; r <= vc.rows_max && r < 33u; r++)
            buf_printf(&b, "pulsar:spec_verify_curve_ms{rows=\"%u\",source=\"%s\"} %.4f\n", r,
                       vc.cached ? "cache" : "profile", (double)vc.base_ms[r]);
        buf_puts(&b, "# HELP pulsar:spec_verify_slope_ms The forward's online residual per row past the profile, ms.\n");
        buf_puts(&b, "# TYPE pulsar:spec_verify_slope_ms gauge\n");
        if (vc.slope_valid) buf_printf(&b, "pulsar:spec_verify_slope_ms %.4f\n", (double)vc.slope_ms);
        buf_puts(&b, "# HELP pulsar:spec_verify_host_ms A round's recent wall outside the forward and drafting, ms.\n");
        buf_puts(&b, "# TYPE pulsar:spec_verify_host_ms gauge\n");
        buf_printf(&b, "pulsar:spec_verify_host_ms %.4f\n", (double)vc.host_ms);
        buf_puts(&b, "# HELP pulsar:spec_verify_draft_ms A round's recent drafting wall, ms.\n");
        buf_puts(&b, "# TYPE pulsar:spec_verify_draft_ms gauge\n");
        buf_printf(&b, "pulsar:spec_verify_draft_ms %.4f\n", (double)vc.draft_ms);
    }
    /* L301: the drafters' confidence calibration per draft position: the confidence each verified-or-rejected
     * draft carried against its exact acceptance label (greedy: the row's argmax is the draft; sampled: the
     * target's probability of the draft).  Bias = conf_sum - label_sum, Brier = sq_err_sum / count, mean absolute
     * error = abs_err_sum / count; the bins give the ECE.  drafter = the engine's drafter, or "copy". */
    {
        static const char *const metric[6] = {"conf", "label", "hit", "abs_err", "sq_err", "count"};
        for (int k = 0; k < 6; k++) {
            buf_printf(&b, "# HELP pulsar:spec_calib_%s_sum Draft confidence calibration per position (%s).\n",
                       metric[k], metric[k]);
            buf_printf(&b, "# TYPE pulsar:spec_calib_%s_sum counter\n", metric[k]);
            for (int d = 0; d < 2; d++)
                for (int i = 0; i < 16; i++) {
                    if (cal.n[d][i] == 0) continue;
                    const double v = k == 0 ? cal.conf[d][i] : k == 1 ? cal.label[d][i] : k == 2 ? cal.hit[d][i]
                                   : k == 3 ? cal.abs_err[d][i] : k == 4 ? cal.sq_err[d][i] : (double)cal.n[d][i];
                    buf_printf(&b, "pulsar:spec_calib_%s_sum{drafter=\"%s\",pos=\"%d\"} %.6f\n", metric[k],
                               d ? "copy" : drafter, i, v);
                }
        }
        static const char *const bmetric[3] = {"count", "conf", "label"};
        for (int k = 0; k < 3; k++) {
            buf_printf(&b, "# HELP pulsar:spec_calib_bin_%s_sum Draft confidence calibration per confidence decile "
                           "(%s).\n", bmetric[k], bmetric[k]);
            buf_printf(&b, "# TYPE pulsar:spec_calib_bin_%s_sum counter\n", bmetric[k]);
            for (int d = 0; d < 2; d++)
                for (int i = 0; i < 10; i++) {
                    if (cal.bin_n[d][i] == 0) continue;
                    const double v = k == 0 ? (double)cal.bin_n[d][i] : k == 1 ? cal.bin_conf[d][i] : cal.bin_label[d][i];
                    buf_printf(&b, "pulsar:spec_calib_bin_%s_sum{drafter=\"%s\",bin=\"%d\"} %.6f\n", bmetric[k],
                               d ? "copy" : drafter, i, v);
                }
        }
    }
    /* L288: the spec lane's decode-only rounds by phase, host wall (server_round_phase) -- a client's delta over
     * a window over the rounds' delta is ms/round per phase, measured untraced. */
    buf_puts(&b, "# HELP pulsar:spec_round_phase_ms_total Decode-only spec rounds' host wall by phase, ms.\n");
    buf_puts(&b, "# TYPE pulsar:spec_round_phase_ms_total counter\n");
    for (int p = 0; p < SERVER_ROUND_PHASE_COUNT; p++)
        buf_printf(&b, "pulsar:spec_round_phase_ms_total{phase=\"%s\"} %.3f\n", server_round_phase_name[p], phase_ms[p]);
    buf_puts(&b, "# HELP pulsar:spec_round_phase_rounds_total Decode-only spec rounds timed by phase.\n");
    buf_puts(&b, "# TYPE pulsar:spec_round_phase_rounds_total counter\n");
    buf_printf(&b, "pulsar:spec_round_phase_rounds_total %llu\n", phase_rounds);
    /* L302: the same rounds' host<->device synchronisations, by kind (delta over a window / rounds = per round) */
    buf_puts(&b, "# HELP pulsar:spec_round_syncs_total Decode-only spec rounds' host<->device synchronisations.\n");
    buf_puts(&b, "# TYPE pulsar:spec_round_syncs_total counter\n");
    buf_printf(&b, "pulsar:spec_round_syncs_total{kind=\"stream_drain\"} %llu\n", (unsigned long long)round_syncs[0]);
    buf_printf(&b, "pulsar:spec_round_syncs_total{kind=\"d2h_copy\"} %llu\n", (unsigned long long)round_syncs[1]);
    buf_printf(&b, "pulsar:spec_round_syncs_total{kind=\"d2h_bytes\"} %llu\n", (unsigned long long)round_syncs[2]);
    buf_printf(&b, "pulsar:spec_round_syncs_total{kind=\"h2d_copy\"} %llu\n", (unsigned long long)round_syncs[3]);
    /* Prefix-cache hit rate (scraper computes hits/queries). */
    buf_puts(&b, "# HELP vllm:prefix_cache_queries_total Cumulative prompt tokens looked up in the prefix cache.\n");
    buf_puts(&b, "# TYPE vllm:prefix_cache_queries_total counter\n");
    buf_printf(&b, "vllm:prefix_cache_queries_total %llu\n", pfx_queries);
    buf_puts(&b, "# HELP vllm:prefix_cache_hits_total Cumulative prompt tokens served from the prefix cache.\n");
    buf_puts(&b, "# TYPE vllm:prefix_cache_hits_total counter\n");
    buf_printf(&b, "vllm:prefix_cache_hits_total %llu\n", pfx_hits);
    /* Scheduler + KV gauges. The unlabeled kv_cache_usage_perc series is the
     * MAX across provisioned slots (single-gauge scrapers keep working and
     * see the most-loaded session); the slot-labeled series break it out
     * per session. */
    buf_puts(&b, "# HELP vllm:kv_cache_usage_perc KV cache utilization (0-1); unlabeled = max across slots.\n");
    buf_puts(&b, "# TYPE vllm:kv_cache_usage_perc gauge\n");
    buf_printf(&b, "vllm:kv_cache_usage_perc %.6f\n", kv);
    for (int i = 0; i < n_slots; i++) {
        buf_printf(&b, "vllm:kv_cache_usage_perc{slot=\"%d\"} %.6f\n", i, slot_kv[i]);
    }
    buf_puts(&b, "# HELP vllm:num_requests_running Requests currently generating.\n");
    buf_puts(&b, "# TYPE vllm:num_requests_running gauge\n");
    buf_printf(&b, "vllm:num_requests_running %d\n", running);
    buf_puts(&b, "# HELP vllm:num_requests_waiting Requests queued and not yet started.\n");
    buf_puts(&b, "# TYPE vllm:num_requests_waiting gauge\n");
    buf_printf(&b, "vllm:num_requests_waiting %d\n", waiting);

    /* Per-slot phase. A scraper watching only kv_cache_usage_perc sees a slot's
     * position advance identically whether it is prefilling a prompt or
     * decoding tokens; this is what separates the two. */
    int prefilling = 0;
    for (int i = 0; i < n_slots; i++) {
        const int idx = slot_phase_index(slot_phase[i]);
        if (idx == 1 || idx == 2 || idx == 5) prefilling++;   /* L330: a restoring prompt is ingesting too */
    }
    /* The pool this run was sized for. Publishes the same number /health
     * reports as slots.capacity, so a scraper can compute "how much of the pool
     * is in use" without polling /health — which matters because banks are
     * provisioned lazily, so the set of pulsar:slot_* series grows from one
     * slot to the full pool over the first minutes of a session. */
    buf_puts(&b, "# HELP pulsar:slot_pool_capacity Decode banks this run was sized for.\n");
    buf_puts(&b, "# TYPE pulsar:slot_pool_capacity gauge\n");
    buf_printf(&b, "pulsar:slot_pool_capacity %d\n", s->pool_banks > 0 ? s->pool_banks : 1);
    buf_puts(&b, "# HELP pulsar:spec_draft_depth Adaptive draft depth per slot (0 = no drafter/idle).\n");
    buf_puts(&b, "# TYPE pulsar:spec_draft_depth gauge\n");
    for (int i = 0; i < n_slots; i++)
        buf_printf(&b, "pulsar:spec_draft_depth{slot=\"%d\"} %d\n", i, slot_depth[i]);
    buf_puts(&b, "# HELP pulsar:slot_phase Generation phase per slot (state set; 1 = current).\n");
    buf_puts(&b, "# TYPE pulsar:slot_phase gauge\n");
    for (int i = 0; i < n_slots; i++) {
        const int idx = slot_phase_index(slot_phase[i]);
        for (int p = 0; p < SLOT_PHASE_COUNT; p++)
            buf_printf(&b, "pulsar:slot_phase{slot=\"%d\",phase=\"%s\"} %d\n",
                       i, slot_phase_names[p], p == idx ? 1 : 0);
    }
    /* Absolute token positions. kv_cache_usage_perc is the ratio only; these
     * let a dashboard show "14203 / 32768" and size the remaining context. */
    buf_puts(&b, "# HELP pulsar:slot_position_tokens KV frontier position per slot, in tokens.\n");
    buf_puts(&b, "# TYPE pulsar:slot_position_tokens gauge\n");
    for (int i = 0; i < n_slots; i++)
        buf_printf(&b, "pulsar:slot_position_tokens{slot=\"%d\"} %d\n", i, slot_pos[i]);
    buf_puts(&b, "# HELP pulsar:slot_context_tokens Context size the slot was admitted for.\n");
    buf_puts(&b, "# TYPE pulsar:slot_context_tokens gauge\n");
    for (int i = 0; i < n_slots; i++)
        buf_printf(&b, "pulsar:slot_context_tokens{slot=\"%d\"} %d\n", i, slot_ctx[i]);
    /* Prefill progress, so a long cold prompt shows a moving bar instead of a
     * slot that merely looks busy. Both read 0 outside a prefill phase. */
    buf_puts(&b, "# HELP pulsar:slot_prefill_done_tokens Prompt tokens synced so far on this slot.\n");
    buf_puts(&b, "# TYPE pulsar:slot_prefill_done_tokens gauge\n");
    for (int i = 0; i < n_slots; i++)
        buf_printf(&b, "pulsar:slot_prefill_done_tokens{slot=\"%d\"} %d\n", i, slot_pf_done[i]);
    buf_puts(&b, "# HELP pulsar:slot_prefill_total_tokens Prefill target for this slot's prompt.\n");
    buf_puts(&b, "# TYPE pulsar:slot_prefill_total_tokens gauge\n");
    for (int i = 0; i < n_slots; i++)
        buf_printf(&b, "pulsar:slot_prefill_total_tokens{slot=\"%d\"} %d\n", i, slot_pf_total[i]);
    buf_puts(&b, "# HELP pulsar:num_requests_prefilling Requests currently in a prefill phase (or restoring a disk KV chain, L330).\n");
    buf_puts(&b, "# TYPE pulsar:num_requests_prefilling gauge\n");
    buf_printf(&b, "pulsar:num_requests_prefilling %d\n", prefilling);

    /* Decode lane. L118: everything is a batch — decode always runs through
     * the batched quanta (lane 3 spec-batched, or lane 2 plain-batched when a
     * slot cannot speculate). The classic "spec" lane (1) is retired; its
     * label is kept so scrapers' label sets stay stable, it just never reads
     * 1. spec_decode_* counters advance on "spec-batched". L251 lane 4
     * "family-spec": one greedy decoder on a family with its own speculative
     * generate (Qwen MTP). The pulsar:spec_max_live gauge is deleted with its
     * scheduler knob. */
    static const char *const lane_names[] = { "idle", "spec", "batched", "spec-batched",
                                              "family-spec" };
    const int n_lanes = (int)(sizeof lane_names / sizeof lane_names[0]);
    const int lane = (decode_lane >= 0 && decode_lane < n_lanes) ? decode_lane : 0;
    buf_puts(&b, "# HELP pulsar:decode_lane Active decode lane (state set; spec_decode_* advance on \"spec-batched\").\n");
    buf_puts(&b, "# TYPE pulsar:decode_lane gauge\n");
    for (int i = 0; i < n_lanes; i++)
        buf_printf(&b, "pulsar:decode_lane{lane=\"%s\"} %d\n", lane_names[i], i == lane ? 1 : 0);

    /* Request latency. These are the per-request numbers the response body
     * already reports (req_timings); as histograms they answer "is the tail
     * prefill or decode" without scraping every response. */
    emit_histogram(&b, "vllm:time_to_first_token_seconds",
                   "Time from request start to first emitted token.",
                   &h_ttft, pulsar_hist_seconds_bounds);
    emit_histogram(&b, "vllm:time_per_output_token_seconds",
                   "Decode wall time divided by completion tokens.",
                   &h_tpot, pulsar_hist_seconds_bounds);
    emit_histogram(&b, "vllm:e2e_request_latency_seconds",
                   "Wall time from request start to finish.",
                   &h_e2e, pulsar_hist_seconds_bounds);
    emit_histogram(&b, "vllm:request_prompt_tokens", "Prompt tokens per request.",
                   &h_prompt_tok, pulsar_hist_tokens_bounds);
    emit_histogram(&b, "vllm:request_generation_tokens", "Completion tokens per request.",
                   &h_gen_tok, pulsar_hist_tokens_bounds);
    buf_puts(&b, "# HELP pulsar:requests_finished_total Requests that reached the finish phase.\n");
    buf_puts(&b, "# TYPE pulsar:requests_finished_total counter\n");
    buf_printf(&b, "pulsar:requests_finished_total %llu\n", reqs_finished);

    /* Slot-pool churn. An eviction forces that conversation to replay from a
     * checkpoint on its next turn, so these counters are the explanation for
     * an otherwise mysterious latency spike. */
    buf_puts(&b, "# HELP pulsar:slot_evictions_total Slots evicted to make room for another conversation.\n");
    buf_puts(&b, "# TYPE pulsar:slot_evictions_total counter\n");
    buf_printf(&b, "pulsar:slot_evictions_total %llu\n", evictions);
    buf_puts(&b, "# HELP pulsar:bank_spills_total Banks spilled to disk by the proactive-eviction guard.\n");
    buf_puts(&b, "# TYPE pulsar:bank_spills_total counter\n");
    buf_printf(&b, "pulsar:bank_spills_total %llu\n", spills);
    buf_puts(&b, "# HELP pulsar:bank_restores_total Spilled banks brought back for a job (re-backed; the job restores the chain, L330).\n");
    buf_puts(&b, "# TYPE pulsar:bank_restores_total counter\n");
    buf_printf(&b, "pulsar:bank_restores_total %llu\n", restores);
    buf_puts(&b, "# HELP pulsar:bank_restore_failures_total Spilled banks that could not be re-backed.\n");
    buf_puts(&b, "# TYPE pulsar:bank_restore_failures_total counter\n");
    buf_printf(&b, "pulsar:bank_restore_failures_total %llu\n", restore_failures);
    /* L330: a disk chain loads a slice of segments per worker turn; the longest slice is the longest the worker -- and
     * every stream beside the restore -- stood still for one */
    buf_puts(&b, "# HELP pulsar:kv_restore_slices_total Disk KV chain slices read (a restore reads one per worker turn; a one-shot load is one).\n");
    buf_puts(&b, "# TYPE pulsar:kv_restore_slices_total counter\n");
    buf_printf(&b, "pulsar:kv_restore_slices_total %llu\n", restore_slices);
    buf_puts(&b, "# HELP pulsar:kv_restore_stall_max_seconds The longest worker stall a disk KV restore caused: its longest slice.\n");
    buf_puts(&b, "# TYPE pulsar:kv_restore_stall_max_seconds gauge\n");
    buf_printf(&b, "pulsar:kv_restore_stall_max_seconds %.6f\n", restore_stall_max_s);

    /* L314: running decoders preempted through their terminal checkpoint for a prompt that did not fit, and back */
    buf_puts(&b, "# HELP pulsar:decoder_preemptions_total Running decoders preempted (terminal checkpoint to disk) to admit a prompt.\n");
    buf_puts(&b, "# TYPE pulsar:decoder_preemptions_total counter\n");
    buf_printf(&b, "pulsar:decoder_preemptions_total %llu\n", preemptions);
    buf_puts(&b, "# HELP pulsar:decoder_readmissions_total Preempted decoders reloaded and decoding again.\n");
    buf_puts(&b, "# TYPE pulsar:decoder_readmissions_total counter\n");
    buf_printf(&b, "pulsar:decoder_readmissions_total %llu\n", readmissions);

    /* L324: the duplicate-prefix census of the resident banks (server::kv_dup_census_update, at each request's
     * end): whole grid pages held by more than one bank, what the extra copies cost, and how deep pairs share --
     * the evidence for (or against) sharing prefix pages in GPU memory. */
    buf_puts(&b, "# HELP pulsar:kv_dup_page_tokens Tokens per page of the duplicate census (the disk grid, L323).\n");
    buf_puts(&b, "# TYPE pulsar:kv_dup_page_tokens gauge\n");
    buf_printf(&b, "pulsar:kv_dup_page_tokens %d\n", dup.grid);
    buf_puts(&b, "# HELP pulsar:kv_dup_page_bytes GPU KV bytes one bank holds for one page.\n");
    buf_puts(&b, "# TYPE pulsar:kv_dup_page_bytes gauge\n");
    buf_printf(&b, "pulsar:kv_dup_page_bytes %llu\n", (unsigned long long)dup.page_bytes.raw());
    buf_puts(&b, "# HELP pulsar:kv_dup_banks Resident banks holding at least one whole page.\n");
    buf_puts(&b, "# TYPE pulsar:kv_dup_banks gauge\n");
    buf_printf(&b, "pulsar:kv_dup_banks %d\n", dup.banks);
    buf_puts(&b, "# HELP pulsar:kv_dup_pages_resident Whole pages held, summed over resident banks.\n");
    buf_puts(&b, "# TYPE pulsar:kv_dup_pages_resident gauge\n");
    buf_printf(&b, "pulsar:kv_dup_pages_resident %llu\n", (unsigned long long)dup.pages_resident);
    buf_puts(&b, "# HELP pulsar:kv_dup_pages_distinct Distinct pages among them (a page two banks share by token prefix counts once).\n");
    buf_puts(&b, "# TYPE pulsar:kv_dup_pages_distinct gauge\n");
    buf_printf(&b, "pulsar:kv_dup_pages_distinct %llu\n", (unsigned long long)dup.pages_distinct);
    buf_puts(&b, "# HELP pulsar:kv_dup_pages_shared Distinct pages held by two or more resident banks.\n");
    buf_puts(&b, "# TYPE pulsar:kv_dup_pages_shared gauge\n");
    buf_printf(&b, "pulsar:kv_dup_pages_shared %llu\n", (unsigned long long)dup.pages_shared);
    buf_puts(&b, "# HELP pulsar:kv_dup_pages_duplicate Extra page copies (resident minus distinct).\n");
    buf_puts(&b, "# TYPE pulsar:kv_dup_pages_duplicate gauge\n");
    buf_printf(&b, "pulsar:kv_dup_pages_duplicate %llu\n", (unsigned long long)dup.pages_duplicate);
    buf_puts(&b, "# HELP pulsar:kv_dup_bytes GPU KV bytes the extra page copies cost.\n");
    buf_puts(&b, "# TYPE pulsar:kv_dup_bytes gauge\n");
    buf_printf(&b, "pulsar:kv_dup_bytes %llu\n", (unsigned long long)dup.dup_bytes.raw());
    buf_puts(&b, "# HELP pulsar:kv_dup_bytes_peak The largest kv_dup_bytes any census saw since start.\n");
    buf_puts(&b, "# TYPE pulsar:kv_dup_bytes_peak gauge\n");
    buf_printf(&b, "pulsar:kv_dup_bytes_peak %llu\n", dup_bytes_peak);
    buf_printf(&b, "# HELP pulsar:kv_dup_pairs Resident bank pairs sharing at least one whole page, by depth class (preamble: under %d shared tokens).\n", PULSAR_KV_DUP_DEEP_TOKENS);
    buf_puts(&b, "# TYPE pulsar:kv_dup_pairs gauge\n");
    buf_printf(&b, "pulsar:kv_dup_pairs{class=\"preamble\"} %d\n", dup.pairs[0]);
    buf_printf(&b, "pulsar:kv_dup_pairs{class=\"deep\"} %d\n", dup.pairs[1]);
    buf_puts(&b, "# HELP pulsar:kv_dup_longest_shared_tokens The longest shared token prefix of a resident bank pair, by depth class.\n");
    buf_puts(&b, "# TYPE pulsar:kv_dup_longest_shared_tokens gauge\n");
    buf_printf(&b, "pulsar:kv_dup_longest_shared_tokens{class=\"preamble\"} %d\n", dup.longest[0]);
    buf_printf(&b, "pulsar:kv_dup_longest_shared_tokens{class=\"deep\"} %d\n", dup.longest[1]);
    buf_puts(&b, "# HELP pulsar:kv_dup_censuses_total Duplicate censuses taken (one at each request's end).\n");
    buf_puts(&b, "# TYPE pulsar:kv_dup_censuses_total counter\n");
    buf_printf(&b, "pulsar:kv_dup_censuses_total %llu\n", dup_censuses);

    /* L310: the disk tier's restore-vs-recompute -- what each hit decided, the measured read rates it priced the
     * read with (L327: from the disk and from the page cache), the last decision's two prices, and the tokens each
     * side carried. */
    buf_puts(&b, "# HELP pulsar:kv_disk_restore_decisions_total Disk chain hits by restore-vs-recompute verdict.\n");
    buf_puts(&b, "# TYPE pulsar:kv_disk_restore_decisions_total counter\n");
    for (int i = 0; i < KV_RESTORE_VERDICTS; i++)
        buf_printf(&b, "pulsar:kv_disk_restore_decisions_total{verdict=\"%s\"} %llu\n", kv_restore_verdict_names[i],
                   (unsigned long long)disk_meter.verdicts[i]);
    buf_puts(&b, "# HELP pulsar:kv_disk_read_bytes_per_second Measured disk chain load rate, of the bytes read from "
                 "the disk and from the page cache (0 until the first load).\n");
    buf_puts(&b, "# TYPE pulsar:kv_disk_read_bytes_per_second gauge\n");
    {
        const double per_gib[2] = {disk_meter.read.cold_ms_per_gib, disk_meter.read.cached_ms_per_gib};
        const char *from[2] = {"disk", "page_cache"};
        for (int i = 0; i < 2; i++)
            buf_printf(&b, "pulsar:kv_disk_read_bytes_per_second{from=\"%s\"} %.0f\n", from[i],
                       per_gib[i] > 0.0 ? 1024.0 * 1024.0 * 1024.0 * 1000.0 / per_gib[i] : 0.0);
    }
    buf_puts(&b, "# HELP pulsar:kv_disk_last_transfer_ms The last disk hit's estimated chain read time.\n");
    buf_puts(&b, "# TYPE pulsar:kv_disk_last_transfer_ms gauge\n");
    buf_printf(&b, "pulsar:kv_disk_last_transfer_ms %.3f\n", disk_meter.last_transfer_ms);
    buf_puts(&b, "# HELP pulsar:kv_disk_last_prefill_ms The last disk hit's priced prefill of the tokens it adds.\n");
    buf_puts(&b, "# TYPE pulsar:kv_disk_last_prefill_ms gauge\n");
    buf_printf(&b, "pulsar:kv_disk_last_prefill_ms %.3f\n", disk_meter.last_prefill_ms);
    buf_puts(&b, "# HELP pulsar:kv_disk_restored_tokens_total Tokens loaded disk chains added past the banks' own resumes.\n");
    buf_puts(&b, "# TYPE pulsar:kv_disk_restored_tokens_total counter\n");
    buf_printf(&b, "pulsar:kv_disk_restored_tokens_total %llu\n", (unsigned long long)disk_meter.restored_tokens);
    buf_puts(&b, "# HELP pulsar:kv_disk_recomputed_tokens_total Tokens a recompute verdict left to the prefill.\n");
    buf_puts(&b, "# TYPE pulsar:kv_disk_recomputed_tokens_total counter\n");
    buf_printf(&b, "pulsar:kv_disk_recomputed_tokens_total %llu\n", (unsigned long long)disk_meter.recomputed_tokens);
    /* L318: where each request's sync resumed -- the deepest of the bank's own state, an adopt, the disk chain */
    buf_puts(&b, "# HELP pulsar:resume_decisions_total Resume decisions by the state picked (own bank, adopt, disk).\n");
    buf_puts(&b, "# TYPE pulsar:resume_decisions_total counter\n");
    for (int i = 0; i < RESUME_PICKS; i++)
        buf_printf(&b, "pulsar:resume_decisions_total{source=\"%s\"} %llu\n", resume_pick_names[i],
                   (unsigned long long)disk_meter.picks[i]);
    /* L318: prefills a disconnect abandoned, and the computed prefix each persisted for the re-send */
    buf_puts(&b, "# HELP pulsar:prefill_abandon_persisted_tokens_total Tokens an abandoned prefill left on disk as segments.\n");
    buf_puts(&b, "# TYPE pulsar:prefill_abandon_persisted_tokens_total counter\n");
    buf_printf(&b, "pulsar:prefill_abandon_persisted_tokens_total %llu\n",
               (unsigned long long)disk_meter.abandon_persisted_tokens);
    /* L329 (a): queued requests that waited for a cut another prompt's prefill was filling, and how each wait ended */
    buf_puts(&b, "# HELP pulsar:prefix_waits_total Queued requests' waits for a cut another prompt's prefill was "
                 "filling: started, and ended by outcome (reached, left, timeout).\n");
    buf_puts(&b, "# TYPE pulsar:prefix_waits_total counter\n");
    for (int i = 0; i < PREFIX_WAIT_OUTCOMES; i++)
        buf_printf(&b, "pulsar:prefix_waits_total{outcome=\"%s\"} %llu\n", prefix_wait_names[i], prefix_waits[i]);
    buf_puts(&b, "# HELP pulsar:prefix_wait_gain_tokens_total Tokens the reached waits moved their requests' resume "
                 "past what served them when they began waiting.\n");
    buf_puts(&b, "# TYPE pulsar:prefix_wait_gain_tokens_total counter\n");
    buf_printf(&b, "pulsar:prefix_wait_gain_tokens_total %llu\n", prefix_wait_gain);
    /* L329 (b2): the next-chunk PLE read-ahead (the disk row table's background readers; the pool keeps them) */
    pulsar_read_ahead_stats ra;
    if (pulsar_engine_read_ahead_stats(s->engine, &ra)) {
        buf_puts(&b, "# HELP pulsar:ple_read_ahead_rows_total Prompt rows the next-chunk read-ahead queued, read into the "
                     "page cache, failed, or dropped before a reader took them (cancelled), or not queued at the pool's cap of unread rows (refused).\n");
        buf_puts(&b, "# TYPE pulsar:ple_read_ahead_rows_total counter\n");
        buf_printf(&b, "pulsar:ple_read_ahead_rows_total{outcome=\"queued\"} %llu\n", (unsigned long long)ra.rows_queued);
        buf_printf(&b, "pulsar:ple_read_ahead_rows_total{outcome=\"read\"} %llu\n", (unsigned long long)ra.rows_read);
        buf_printf(&b, "pulsar:ple_read_ahead_rows_total{outcome=\"failed\"} %llu\n", (unsigned long long)ra.rows_failed);
        buf_printf(&b, "pulsar:ple_read_ahead_rows_total{outcome=\"cancelled\"} %llu\n",
                   (unsigned long long)ra.rows_cancelled);
        buf_printf(&b, "pulsar:ple_read_ahead_rows_total{outcome=\"refused\"} %llu\n", (unsigned long long)ra.rows_refused);
        buf_puts(&b, "# HELP pulsar:ple_read_ahead_skipped_total Read-aheads not queued: MemAvailable under the "
                     "read-ahead's headroom (memory pressure).\n");
        buf_puts(&b, "# TYPE pulsar:ple_read_ahead_skipped_total counter\n");
        buf_printf(&b, "pulsar:ple_read_ahead_skipped_total %llu\n", (unsigned long long)ra.skipped);
        buf_puts(&b, "# HELP pulsar:ple_read_ahead_bytes_total Bytes the read-ahead read.\n");
        buf_puts(&b, "# TYPE pulsar:ple_read_ahead_bytes_total counter\n");
        buf_printf(&b, "pulsar:ple_read_ahead_bytes_total %llu\n", (unsigned long long)ra.bytes_read);
        buf_puts(&b, "# HELP pulsar:ple_read_ahead_seconds_total Reader time on read-ahead rows, summed over the readers.\n");
        buf_puts(&b, "# TYPE pulsar:ple_read_ahead_seconds_total counter\n");
        buf_printf(&b, "pulsar:ple_read_ahead_seconds_total %.6f\n", (double)ra.read_ns * 1e-9);
        buf_puts(&b, "# HELP pulsar:ple_read_ahead_issued_total Read-aheads queued (one per prompt chunk ahead).\n");
        buf_puts(&b, "# TYPE pulsar:ple_read_ahead_issued_total counter\n");
        buf_printf(&b, "pulsar:ple_read_ahead_issued_total %llu\n", (unsigned long long)ra.issued);
        buf_puts(&b, "# HELP pulsar:ple_read_ahead_pending_rows Read-ahead rows queued now, not yet read.\n");
        buf_puts(&b, "# TYPE pulsar:ple_read_ahead_pending_rows gauge\n");
        buf_printf(&b, "pulsar:ple_read_ahead_pending_rows %llu\n",
                   (unsigned long long)(ra.rows_pending + ra.in_flight));
    }

    /* Why requests queue. num_requests_waiting says that they do; this says
     * whether evicting something would help or the box is simply out of room. */
    buf_puts(&b, "# HELP pulsar:admission_refusals_total Jobs blocked from binding, by reason.\n");
    buf_puts(&b, "# TYPE pulsar:admission_refusals_total counter\n");
    for (int i = PROVISION_OK + 1; i < PROVISION_REFUSAL_COUNT; i++)
        buf_printf(&b, "pulsar:admission_refusals_total{reason=\"%s\"} %llu\n",
                   refusal_names[i], (unsigned long long)refusals[i]);
    /* m_queue_block_reason holds provision_refusal + 1, so 0 means nothing is
     * blocking and the "none" series carries the 1. */
    const int blocked_by = block_reason > 0 ? block_reason - 1 : (int)PROVISION_OK;
    buf_puts(&b, "# HELP pulsar:queue_blocked_reason Why the head job cannot bind right now (state set).\n");
    buf_puts(&b, "# TYPE pulsar:queue_blocked_reason gauge\n");
    for (int i = 0; i < PROVISION_REFUSAL_COUNT; i++)
        buf_printf(&b, "pulsar:queue_blocked_reason{reason=\"%s\"} %d\n",
                   refusal_names[i], i == blocked_by ? 1 : 0);

    /* Admission ledger. Distinct from kv_cache_usage_perc: a slot early in its
     * context still holds its whole ledgered session cost, so the pool can be
     * full while every slot reads near-empty. */
    buf_puts(&b, "# HELP pulsar:kv_ledger_committed_bytes Session cost committed by the admission ledger.\n");
    buf_puts(&b, "# TYPE pulsar:kv_ledger_committed_bytes gauge\n");
    buf_printf(&b, "pulsar:kv_ledger_committed_bytes %llu\n", ledger_committed);
    buf_puts(&b, "# HELP pulsar:kv_bank_bytes KV bytes one bank holds at the configured context, compressed rows plus index. Demand-paged under overcommit: reserved as VA, resident on touch.\n");
    buf_puts(&b, "# TYPE pulsar:kv_bank_bytes gauge\n");
    buf_printf(&b, "pulsar:kv_bank_bytes %llu\n", (unsigned long long)s->kv_bank_bytes.raw());
    buf_puts(&b, "# HELP pulsar:kv_ledger_budget_bytes Admission ceiling computed at startup.\n");
    buf_puts(&b, "# TYPE pulsar:kv_ledger_budget_bytes gauge\n");
    buf_printf(&b, "pulsar:kv_ledger_budget_bytes %llu\n", ledger_budget);
    /* The part of that budget token KV can actually grow into: the ledger also
     * carries every bank's eager floor, so budget / kv_bank_bytes overstates
     * the tokens the box holds before the 2b guard spills (6.0 of 13.1 GiB on
     * 16 banks at 1M).  0 when the guard is off (no overcommit). */
    buf_puts(&b, "# HELP pulsar:kv_touched_budget_bytes Demand-paged KV the eviction guard allows resident before spilling idle banks, live: touched plus MemAvailable minus the memory floor now, never above the startup ceiling (L319, 2026-10-29; 0 = guard off).\n");
    buf_puts(&b, "# TYPE pulsar:kv_touched_budget_bytes gauge\n");
    buf_printf(&b, "pulsar:kv_touched_budget_bytes %llu\n", s->guard_enabled ? kv_budget_live : 0ULL);
    buf_puts(&b, "# HELP pulsar:kv_touched_ceiling_bytes The startup ceiling the live touched budget never exceeds (min of the ledger share and the post-warmup headroom, L319; 0 = guard off).\n");
    buf_puts(&b, "# TYPE pulsar:kv_touched_ceiling_bytes gauge\n");
    buf_printf(&b, "pulsar:kv_touched_ceiling_bytes %llu\n",
               s->guard_enabled ? (unsigned long long)s->guard_touched_ceiling.raw() : 0ULL);
    buf_puts(&b, "# HELP pulsar:mem_floor_bytes MemAvailable every admission, guard, prefill and persist check keeps free (OS margin + kernel high watermarks + one step's demand).\n");
    buf_puts(&b, "# TYPE pulsar:mem_floor_bytes gauge\n");
    buf_printf(&b, "pulsar:mem_floor_bytes %llu\n", (unsigned long long)server_mem_floor().raw());
    buf_puts(&b, "# HELP pulsar:kv_persist_skipped_total Disk-KV persists skipped because MemAvailable could not cover them over the memory floor.\n");
    buf_puts(&b, "# TYPE pulsar:kv_persist_skipped_total counter\n");
    buf_printf(&b, "pulsar:kv_persist_skipped_total %llu\n", persist_skipped);
    buf_puts(&b, "# HELP pulsar:prefill_mem_refusals_total Prefills stopped mid-prompt because MemAvailable was under the memory floor before a chunk.\n");
    buf_puts(&b, "# TYPE pulsar:prefill_mem_refusals_total counter\n");
    buf_printf(&b, "pulsar:prefill_mem_refusals_total %llu\n", prefill_mem_refusals);
    buf_puts(&b, "# HELP pulsar:worker_stalled_seconds Seconds the worker has made no progress while jobs wait (the stall watchdog; 0 = progressing or off).\n");
    buf_puts(&b, "# TYPE pulsar:worker_stalled_seconds gauge\n");
    buf_printf(&b, "pulsar:worker_stalled_seconds %lld\n", (long long)__atomic_load_n(&s->stall_s, __ATOMIC_RELAXED));
    /* L326: the eager floor that min() subtracts, as the server holds it (guard_eager_bytes) -- the touched budget
     * alone does not give it back once the headroom term binds (L319), so tests read it here, not from a log line. */
    buf_puts(&b, "# HELP pulsar:kv_eager_floor_bytes Eager KV floor already resident that the eviction guard's ledger share subtracts (kv budget minus this, L319/L326; 0 = guard off).\n");
    buf_puts(&b, "# TYPE pulsar:kv_eager_floor_bytes gauge\n");
    buf_printf(&b, "pulsar:kv_eager_floor_bytes %llu\n",
               s->guard_enabled ? (unsigned long long)s->guard_eager_bytes.raw() : 0ULL);
    /* L328: the resident KV every guard and admission decision reads, and the base they read it from (touched plus
     * what the admitted prompts and restoring chains still add) */
    buf_puts(&b, "# HELP pulsar:kv_touched_bytes Demand-paged KV resident now, the pool's high-water rows (0 = guard off).\n");
    buf_puts(&b, "# TYPE pulsar:kv_touched_bytes gauge\n");
    buf_printf(&b, "pulsar:kv_touched_bytes %llu\n", kv_touched);
    buf_puts(&b, "# HELP pulsar:kv_reserved_bytes The base every KV admission decision reads: touched plus what admitted prompts and restoring chains still add (L326/L330; 0 = guard off).\n");
    buf_puts(&b, "# TYPE pulsar:kv_reserved_bytes gauge\n");
    buf_printf(&b, "pulsar:kv_reserved_bytes %llu\n", kv_reserved);

    /* plan-31b phase A: whole-box reconciliation -- the signed gap between
     * what the BOX says this process consumed since a post-boot baseline and
     * what the engine's own tensor census explains. A growing residual is a
     * leak or a phantom the per-allocation est==actual assert cannot see
     * (it checks each allocation, not the sum of everything else). Pure
     * reporting: no admission decision reads it. Page-cache drift from the
     * file-backed model IS included -- interpret trends, not absolutes.
     * Baseline self-captures once, >=60s after the listener came up, so
     * load-time churn stays out of the window. */
    {
        static uint64_t base_avail_kib = 0;
        static uint64_t base_census = 0;
        static bool base_set = false;
        static const char *const akey[] = {"MemAvailable:"};
        uint64_t avail_kib[1] = {0};
        FILE *mf = fopen("/proc/meminfo", "r");
        if (mf) {
            char line[256];
            while (fgets(line, sizeof(line), mf)) {
                if (!strncmp(line, akey[0], strlen(akey[0]))) {
                    unsigned long long v = 0;
                    if (sscanf(line + strlen(akey[0]), " %llu kB", &v) == 1)
                        avail_kib[0] = v;
                }
            }
            fclose(mf);
        }
        const uint64_t census = pulsar_gpu_tensor_alloc_bytes_current();
        if (!base_set && s->started && time(NULL) - s->started >= 60 &&
            avail_kib[0] > 0) {
            base_avail_kib = avail_kib[0];
            base_census = census;
            base_set = true;
        }
        if (base_set && avail_kib[0] > 0) {
            const long long box_delta =
                ((long long)base_avail_kib - (long long)avail_kib[0]) * 1024ll;
            const long long census_delta =
                (long long)census - (long long)base_census;
            buf_puts(&b, "# HELP pulsar:mem_reconcile_residual_bytes Box consumption since baseline minus census-explained bytes (signed; page-cache drift included).\n");
            buf_puts(&b, "# TYPE pulsar:mem_reconcile_residual_bytes gauge\n");
            buf_printf(&b, "pulsar:mem_reconcile_residual_bytes %lld\n",
                       box_delta - census_delta);
        }
    }

    bool ok = http_response(fd, 200, "text/plain; version=0.0.4", b.ptr);
    buf_free(&b);
    return ok;
}



/* ── /metrics/stream ───────────────────────────────────────────────────────
 *
 * Polling, inverted: instead of a client asking /metrics twenty times a second
 * whether anything changed, it opens one connection and is told.
 *
 * Polling is unusually expensive here. Every response closes its connection
 * (http_response hard-codes `Connection: close`) and the accept loop spawns a
 * thread per connection against a client cap shared with real requests, so a
 * 20 Hz scraper costs twenty handshakes and twenty thread creations per second
 * on the machine it is measuring. One stream subscriber costs one thread for
 * the life of the subscription and nothing at all in between.
 *
 * The event carries the generation and nothing else. Shipping the exposition
 * body on every publish would be ~12 KiB per event, worse than the polling it
 * replaces, so the client fetches /metrics itself when the generation moves —
 * once per publish instead of twenty times per publish. */

/// The /metrics/stream loop: one "event: metrics" frame per new generation, an SSE keepalive comment after
/// `keepalive_ms` without one.  Client thread; waits on `cv` under `mu`, writes with `mu` released.
/// @param fd            the client socket
/// @param mu            the mutex guarding `generation` and `stop` (server::mu)
/// @param cv            broadcast on every publish and at shutdown (server::stream_cv)
/// @param generation    the publish counter (server::metrics_generation)
/// @param stop          set at shutdown (server::stopping)
/// @param keepalive_ms  the idle interval before a keepalive, ms
/// @return true when `stop` ended the stream, false when a write failed (the client went away)
bool metrics_stream_pump(int fd, pthread_mutex_t *mu, pthread_cond_t *cv,
                         const unsigned long long *generation, const bool *stop,
                         int keepalive_ms) {
    unsigned long long last = 0;
    bool sent = false;

    for (;;) {
        bool keepalive = false;
        unsigned long long gen;

        pthread_mutex_lock(mu);
        if (*stop) {
            pthread_mutex_unlock(mu);
            return true;
        }
        gen = *generation;

        if (sent && gen == last) {
            /* Nothing published since the last frame. Wait for the worker to
             * publish, or for long enough that the connection has to be
             * proved still alive — whichever comes first. */
            struct timespec deadline;
            clock_gettime(CLOCK_REALTIME, &deadline);
            deadline.tv_sec += keepalive_ms / 1000;
            deadline.tv_nsec += (long)(keepalive_ms % 1000) * 1000000L;
            if (deadline.tv_nsec >= 1000000000L) {
                deadline.tv_sec += 1;
                deadline.tv_nsec -= 1000000000L;
            }
            pthread_cond_timedwait(cv, mu, &deadline);

            if (*stop) {
                pthread_mutex_unlock(mu);
                return true;
            }
            gen = *generation;
            keepalive = (gen == last);
        }
        pthread_mutex_unlock(mu);

        /* The write happens with mu released, and that is the whole of the
         * no-backpressure argument: send_all bounds itself with its own
         * deadline and returns false for a wedged reader, and while it is
         * waiting no lock is held, so the worker publishing the next snapshot
         * never queues behind a slow client. */
        buf b = {0};
        if (keepalive) {
            /* An SSE comment: every parser skips it, and it keeps the socket
             * warm through proxy idle timeouts. */
            buf_puts(&b, ": keepalive\n\n");
        } else {
            struct timespec now;
            clock_gettime(CLOCK_REALTIME, &now);
            buf_puts(&b, "event: metrics\ndata: {\"generation\":");
            buf_printf(&b, "%llu", gen);
            buf_puts(&b, ",\"t\":");
            buf_printf(&b, "%.3f", (double)now.tv_sec + (double)now.tv_nsec / 1e9);
            buf_puts(&b, "}\n\n");
            last = gen;
            sent = true;
        }
        bool ok = send_all(fd, b.ptr, b.len);
        buf_free(&b);
        if (!ok) return false;   /* client gone, or shutdown: send_all checks */
    }
}



bool server::send_metrics_stream(int fd) {
    auto *s = this;

    pthread_mutex_lock(&s->mu);
    const bool at_cap = s->stream_clients >= PULSAR_SERVER_MAX_STREAMS;
    if (!at_cap) s->stream_clients++;
    pthread_mutex_unlock(&s->mu);

    if (at_cap) {
        /* Refused rather than queued: a scraper that cannot stream should fall
         * back to polling, which this server still serves. Blocking here would
         * just move the queueing somewhere less visible. */
        http_error_retry(fd, 503, "too many metric streams", HTTP_RETRY_BUSY_S);
        return false;
    }

    bool ok = sse_headers(fd);
    if (ok) {
        ok = metrics_stream_pump(fd, &s->mu, &s->stream_cv,
                                 &s->metrics_generation, &s->stopping,
                                 PULSAR_METRICS_STREAM_KEEPALIVE_MS);
    }

    pthread_mutex_lock(&s->mu);
    if (s->stream_clients > 0) s->stream_clients--;
    pthread_mutex_unlock(&s->mu);
    return ok;
}



void server::client_done() {
    auto *s = this;
    pthread_mutex_lock(&s->mu);
    if (s->clients > 0) s->clients--;
    pthread_cond_broadcast(&s->clients_cv);
    pthread_mutex_unlock(&s->mu);
}






/// A client connection thread: read one request, route it (health / metrics / models answered here; generation
/// requests parsed, enqueued for the worker and waited on), close the socket, client_done().
/// @param arg  a malloc'd client_arg (freed here)
/// @return NULL
void *client_main(void *arg) {
    client_arg *ca = (client_arg *)arg;
    server *s = ca->srv;
    int fd = ca->fd;
    free(ca);

    http_request hr = {0};
    if (!read_http_request(fd, &hr)) {
        http_error(fd, 400, "bad HTTP request");
        goto done;
    }

    if (!strcmp(hr.method, "OPTIONS")) {
        http_response(fd, 204, NULL, "");
        http_request_free(&hr);
        goto done;
    }

    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/health")) {
        s->send_health(fd);
        http_request_free(&hr);
        goto done;
    }
    if (!strcmp(hr.method, "GET") &&
        (!strcmp(hr.path, "/healthz") || !strcmp(hr.path, "/ping"))) {
        s->send_liveness(fd);
        http_request_free(&hr);
        goto done;
    }
    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/version")) {
        s->send_version(fd);
        http_request_free(&hr);
        goto done;
    }
    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/")) {
        s->send_root(fd);
        http_request_free(&hr);
        goto done;
    }
    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/v1/models")) {
        s->send_models(fd);
        http_request_free(&hr);
        goto done;
    }
    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/metrics")) {
        s->send_metrics(fd);
        http_request_free(&hr);
        goto done;
    }
    /* Blocks for the life of the subscription, which is why it gets its own
     * cap and its own thread rather than sharing the request path. */
    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/metrics/stream")) {
        s->send_metrics_stream(fd);
        http_request_free(&hr);
        goto done;
    }
    const char *model_path_prefix;
    model_path_prefix = "/v1/models/";
    size_t model_path_prefix_len;
    model_path_prefix_len = strlen(model_path_prefix);
    if (!strcmp(hr.method, "GET") &&
        !strncmp(hr.path, model_path_prefix, model_path_prefix_len) &&
        !strcmp(hr.path + model_path_prefix_len,
                server_served_model_id(s)))
    {
        s->send_model(fd, hr.path + model_path_prefix_len);
        http_request_free(&hr);
        goto done;
    }

    /* Anthropic token counting (POST /v1/messages/count_tokens): render and
     * tokenize exactly as /v1/messages would — same template, tool schemas,
     * tool-memory replay, think mode — and return the count without creating
     * a job or touching the session banks. Deliberately no context-length
     * rejection: clients (Claude Code) call this to decide whether a prompt
     * fits, so an over-window prompt must still count. */
    if (!strcmp(hr.method, "POST") &&
        !strcmp(hr.path, "/v1/messages/count_tokens"))
    {
        request creq;
        char cerr[160];
        if (!parse_anthropic_request(s->engine, s, hr.body, s->default_tokens,
                                     &creq, cerr, sizeof(cerr)))
        {
            http_error_anthropic(fd, 400, cerr);
            http_request_free(&hr);
            goto done;
        }
        buf cb = {0};
        buf_printf(&cb, "{\"input_tokens\":%d}\n", creq.prompt.len);
        http_response(fd, 200, "application/json", cb.ptr);
        buf_free(&cb);
        request_free(&creq);
        http_request_free(&hr);
        goto done;
    }

    request req;
    char err[160];
    bool ok;
    ok = false;
    bool anthropic_surface;
    anthropic_surface = false;
    if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/messages")) {
        anthropic_surface = true;
        ok = parse_anthropic_request(s->engine, s, hr.body, s->default_tokens,
                                     &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/chat/completions")) {
        ok = parse_chat_request(s->engine, s, hr.body, s->default_tokens,
                                &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/responses")) {
        ok = parse_responses_request(s->engine, s, hr.body, s->default_tokens,
                                     &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/completions")) {
        ok = parse_completion_request(s->engine, hr.body, s->default_tokens,
                                      &req, err, sizeof(err));
    } else {
        http_error(fd, 404, "unknown endpoint");
        http_request_free(&hr);
        goto done;
    }
    if (ok) req.raw_body = xstrndup(hr.body, hr.body_len);
    if (ok && s->capture) {
        /* L216 --capture-requests: the accepted body verbatim, one JSON line. */
        pthread_mutex_lock(&s->capture_mu);
        fprintf(s->capture, "{\"ts\":%lld,\"path\":\"%s\",\"body\":", (long long)time(NULL), hr.path);
        fwrite(hr.body, 1, hr.body_len, s->capture);
        fputs("}\n", s->capture);
        pthread_mutex_unlock(&s->capture_mu);
    }
    http_request_free(&hr);
    if (!ok) {
        if (anthropic_surface) http_error_anthropic(fd, 400, err);
        else http_error(fd, 400, err);
        goto done;
    }
    if (!req.model_from_request) {
        free(req.model);
        req.model = xstrdup(server_served_model_id(s));
    }
    int ctx_size;
    ctx_size = pulsar_session_ctx(s->sess).raw();
    if (request_exceeds_context(&req, ctx_size)) {
        http_error_context_length_exceeded(fd, &req, req.prompt.len, ctx_size);
        request_free(&req);
        goto done;
    }

    set_client_socket_nonblocking(fd);
    job j;
    memset(&j, 0, sizeof(j));
    j.fd = fd;
    j.req = req;
    pthread_mutex_init(&j.mu, NULL);
    pthread_cond_init(&j.cv, NULL);

    bool enqueued;
    {
        /* Hold j.mu across enqueue + the wait for the worker's completion
         * signal. The ScopedLock MUST release before the destroy below (you
         * cannot pthread_mutex_destroy a held mutex), so the lock lives in this
         * inner scope and the single shared teardown runs after it unlocks. */
        pulsar::ScopedLock lk(&j.mu);
        enqueued = s->enqueue(&j);
        if (enqueued) {
            /* Wake periodically to notice a client that hung up while the
             * job sits in the queue: flag it (the worker reaps pre-bind,
             * skipping eviction/provisioning for a dead fd) and KEEP
             * waiting for done — the client thread never unlinks or frees,
             * so the worker-only pop invariant holds (upstream ds4 e9ded97,
             * reshaped for our stack-owned jobs). */
            while (!j.done) {
                struct timespec ts;
                clock_gettime(CLOCK_REALTIME, &ts);
                ts.tv_nsec += 250 * 1000000L;
                if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
                pthread_cond_timedwait(&j.cv, &j.mu, &ts);
                if (!j.done && !j.cancelled && gen_client_disconnected(j.fd))
                    j.cancelled = true;
            }
        }
    }
    if (!enqueued) http_error_retry(fd, 503, "server shutting down", HTTP_RETRY_GOING_AWAY_S);
    pthread_cond_destroy(&j.cv);
    pthread_mutex_destroy(&j.mu);
    request_free(&j.req);
done:
    close(fd);
    s->client_done();
    return NULL;
}



/// Open the listening socket (IPv4; "localhost" means 127.0.0.1; SO_REUSEADDR, backlog 128).
/// @param host  a dotted IPv4 address or "localhost"
/// @param port  the TCP port
/// @return the socket, or -1 with errno set
int listen_on(const char *host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    if (!strcmp(host, "localhost")) host = "127.0.0.1";
    if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) {
        close(fd);
        errno = EINVAL;
        return -1;
    }
    if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 128) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}



/// Bound a client socket's blocking reads and writes by PULSAR_SERVER_IO_TIMEOUT_SEC.
/// @param fd  the client socket
void configure_client_socket(int fd) {
    struct timeval tv;
    tv.tv_sec = PULSAR_SERVER_IO_TIMEOUT_SEC;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}



/// Make a queued request's socket non-blocking, so the worker's writes never block on a slow client.
/// @param fd  the client socket
void set_client_socket_nonblocking(int fd) {
    /* The inference worker writes streaming responses itself.  Once a request is
     * queued, a blocked socket would block every other request too, so slow
     * clients are failed instead of back-pressuring the model session. */
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) (void)fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}


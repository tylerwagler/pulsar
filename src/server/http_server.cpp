#include "pulsar_server_internal.h"
#include "pulsar_lock.hpp"



static void http_request_free(http_request *r) {
    free(r->body);
    memset(r, 0, sizeof(*r));
}



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



static bool read_http_request(int fd, http_request *r) {
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



static void append_model_json(buf *b, const server *s, const char *id) {
    append_model_json_values(b,
                             id,
                             server_served_model_name(s),
                             pulsar_session_ctx(s->sess),
                             s->default_tokens);
}



bool server::send_model(int fd, const char *id) {
    auto *s = this;
    buf b = {0};
    append_model_json(&b, s, id);
    buf_putc(&b, '\n');
    bool ok = http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
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
    bool ok = http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

/* Liveness probe (/healthz, /ping): is the process alive at all? Always 200
 * while the process runs — deliberately independent of readiness/drain state,
 * so a k8s liveness probe never restarts a server that is merely draining.
 * Lock-free, engine-free (safe on a client thread). */
bool server::send_liveness(int fd) {
    auto *s = this;
    return http_response(fd, s->enable_cors, 200, "application/json",
                         "{\"status\":\"ok\"}\n");
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
    int n_slots, running, waiting;
    time_t started;
    double kv = 0.0; /* max KV utilization across provisioned slots */
    pthread_mutex_lock(&s->mu);
    draining = s->stopping;
    n_slots  = s->n_slots;
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

    buf b = {0};
    buf_printf(&b,
        "{\"status\":\"%s\",\"version\":\"%s\",\"model\":\"%s\","
        "\"uptime_s\":%ld,\"slots\":{\"total\":%d,\"running\":%d,\"waiting\":%d},"
        "\"kv_cache_usage\":%.6f}\n",
        draining ? "draining" : "ok", PULSAR_VERSION_STR, model,
        uptime, n_slots, running, waiting, kv);
    bool ok = http_response(fd, s->enable_cors, draining ? 503 : 200,
                            "application/json", b.ptr);
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
    bool ok = http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

/* Root banner so a bare GET / (browsers, uptime probes) gets a 200 with the
 * version and a pointer to the real endpoints instead of a 404. */
bool server::send_root(int fd) {
    auto *s = this;
    buf b = {0};
    buf_printf(&b,
        "{\"service\":\"pulsar-server\",\"version\":\"%s\",\"status\":\"ok\","
        "\"endpoints\":[\"/health\",\"/version\",\"/v1/models\","
        "\"/v1/chat/completions\",\"/v1/completions\",\"/v1/messages\","
        "\"/v1/messages/count_tokens\",\"/v1/responses\",\"/metrics\"]}\n",
        PULSAR_VERSION_STR);
    bool ok = http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

/* The phase label set for pulsar:slot_phase. GEN_DECODE_INIT folds into
 * "decode" (it is the transient re-init of a decode attempt, not a state an
 * operator cares to see) and GEN_FINISH/GEN_DONE fold into "finish". Every
 * slot reports all five series with a 0/1 value — the Prometheus state-set
 * convention — so the series set is stable and label values never churn. */
static const char *const slot_phase_names[] = {
    "idle", "prefill_cold", "prefill_main", "decode", "finish",
};
enum { SLOT_PHASE_COUNT = 5 };

/* Map a published m_slot_phase (gen_phase + 1; 0 = no bound job) onto that
 * label set. */
static int slot_phase_index(int phase_plus_one) {
    switch (phase_plus_one - 1) {
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
extern const double pulsar_hist_seconds_bounds[PULSAR_HIST_BUCKETS] = {
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0,
};
extern const double pulsar_hist_tokens_bounds[PULSAR_HIST_BUCKETS] = {
    16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
};

/* Emit one histogram in Prometheus text form. Buckets are stored
 * non-cumulatively, so they are accumulated on the way out; +Inf is the total
 * count, which also covers observations above the last bound. */
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
static const char *const refusal_names[PROVISION_REFUSAL_COUNT] = {
    "none", "pool_full", "ledger_full", "mem_floor", "create_fail",
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
    int slot_pf_done[PULSAR_SESSION_POOL_CAP];
    int slot_pf_total[PULSAR_SESSION_POOL_CAP];
    pthread_mutex_lock(&s->mu);
    m = s->m_spec;
    n_slots = s->n_slots;
    for (int i = 0; i < n_slots; i++) {
        const int pos = s->m_slot_pos[i];
        const int ctx = s->m_slot_ctx[i];
        double kv = (ctx > 0 && pos > 0) ? (double)pos / (double)ctx : 0.0;
        slot_kv[i] = kv > 1.0 ? 1.0 : kv;
        slot_pos[i] = pos > 0 ? pos : 0;
        slot_ctx[i] = ctx > 0 ? ctx : 0;
        slot_phase[i] = s->m_slot_phase[i];
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
    uint64_t refusals[PROVISION_REFUSAL_COUNT];
    for (int i = 0; i < PROVISION_REFUSAL_COUNT; i++) refusals[i] = s->m_refusals[i];
    const int block_reason = s->m_queue_block_reason;
    const unsigned long long ledger_committed = (unsigned long long)s->kv_committed_bytes;
    const unsigned long long ledger_budget = (unsigned long long)s->kv_budget_bytes;
    const unsigned long long gen_tokens = (unsigned long long)s->m_gen_tokens;
    const int decode_lane = s->m_decode_lane;
    const int spec_max_live = s->spec_max_live;
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
     * chart has a full row even before every position has fired. */
    if (m.max_draft > 0) {
        int np = m.max_draft > 16 ? 16 : m.max_draft;
        buf_puts(&b, "# HELP vllm:spec_decode_num_accepted_tokens_per_pos_total Accepted count per draft position.\n");
        buf_puts(&b, "# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_total counter\n");
        for (int i = 0; i < np; i++)
            buf_printf(&b, "vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name=\"%s\",position=\"%d\"} %llu\n",
                       model, i, (unsigned long long)m.accepted_per_pos[i]);
    }
    /* Token-throughput counters (the scraper derives prompt/gen t/s from deltas). */
    buf_puts(&b, "# HELP vllm:prompt_tokens_total Cumulative prompt tokens prefilled.\n");
    buf_puts(&b, "# TYPE vllm:prompt_tokens_total counter\n");
    buf_printf(&b, "vllm:prompt_tokens_total %llu\n", prompt_toks);
    /* Counted at gen_emit_token, NOT from the engine's spec_gen_tokens: the
     * latter only advances inside the DSpark fused verify loop, so it stops
     * entirely once the scheduler batches (more than spec_max_live decode
     * banks) and this metric read zero throughput on a busy server. */
    buf_puts(&b, "# HELP vllm:generation_tokens_total Cumulative tokens emitted, all decode lanes.\n");
    buf_puts(&b, "# TYPE vllm:generation_tokens_total counter\n");
    buf_printf(&b, "vllm:generation_tokens_total %llu\n", gen_tokens);
    buf_puts(&b, "# HELP pulsar:spec_decode_gen_tokens_total Tokens emitted by the fused spec loop only.\n");
    buf_puts(&b, "# TYPE pulsar:spec_decode_gen_tokens_total counter\n");
    buf_printf(&b, "pulsar:spec_decode_gen_tokens_total %llu\n", (unsigned long long)m.gen_tokens);
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
        if (idx == 1 || idx == 2) prefilling++;
    }
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
    buf_puts(&b, "# HELP pulsar:num_requests_prefilling Requests currently in a prefill phase.\n");
    buf_puts(&b, "# TYPE pulsar:num_requests_prefilling gauge\n");
    buf_printf(&b, "pulsar:num_requests_prefilling %d\n", prefilling);

    /* Decode lane. Speculative decoding only runs while at most spec_max_live
     * decode banks are live (default 1 — batching wins above that), and only
     * the spec lane feeds the spec_decode_* counters. Without this a scraper
     * cannot tell a genuine acceptance rate from a stale one left over from
     * the last single-request stretch. */
    static const char *const lane_names[] = { "idle", "spec", "batched" };
    const int lane = (decode_lane >= 0 && decode_lane <= 2) ? decode_lane : 0;
    buf_puts(&b, "# HELP pulsar:decode_lane Active decode lane (state set; spec_decode_* only advance on \"spec\").\n");
    buf_puts(&b, "# TYPE pulsar:decode_lane gauge\n");
    for (int i = 0; i < 3; i++)
        buf_printf(&b, "pulsar:decode_lane{lane=\"%s\"} %d\n", lane_names[i], i == lane ? 1 : 0);
    buf_puts(&b, "# HELP pulsar:spec_max_live Decode-bank count at or below which the spec lane runs.\n");
    buf_puts(&b, "# TYPE pulsar:spec_max_live gauge\n");
    buf_printf(&b, "pulsar:spec_max_live %d\n", spec_max_live);

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
    buf_puts(&b, "# HELP pulsar:bank_restores_total Spilled banks reloaded from disk.\n");
    buf_puts(&b, "# TYPE pulsar:bank_restores_total counter\n");
    buf_printf(&b, "pulsar:bank_restores_total %llu\n", restores);
    buf_puts(&b, "# HELP pulsar:bank_restore_failures_total Spilled banks that could not be reloaded.\n");
    buf_puts(&b, "# TYPE pulsar:bank_restore_failures_total counter\n");
    buf_printf(&b, "pulsar:bank_restore_failures_total %llu\n", restore_failures);

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
    buf_puts(&b, "# HELP pulsar:kv_ledger_budget_bytes Admission ceiling computed at startup.\n");
    buf_puts(&b, "# TYPE pulsar:kv_ledger_budget_bytes gauge\n");
    buf_printf(&b, "pulsar:kv_ledger_budget_bytes %llu\n", ledger_budget);

    bool ok = http_response(fd, s->enable_cors, 200, "text/plain; version=0.0.4", b.ptr);
    buf_free(&b);
    return ok;
}



void server::client_done() {
    auto *s = this;
    pthread_mutex_lock(&s->mu);
    if (s->clients > 0) s->clients--;
    pthread_cond_broadcast(&s->clients_cv);
    pthread_mutex_unlock(&s->mu);
}



void set_client_socket_nonblocking(int fd);



void *client_main(void *arg) {
    client_arg *ca = (client_arg *)arg;
    server *s = ca->srv;
    int fd = ca->fd;
    free(ca);

    http_request hr = {0};
    if (!read_http_request(fd, &hr)) {
        http_error(fd, s->enable_cors, 400, "bad HTTP request");
        goto done;
    }

    if (!strcmp(hr.method, "OPTIONS")) {
        http_response(fd, s->enable_cors, 204, NULL, "");
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
                                     pulsar_session_ctx(s->sess), &creq,
                                     cerr, sizeof(cerr)))
        {
            http_error(fd, s->enable_cors, 400, cerr);
            http_request_free(&hr);
            goto done;
        }
        buf cb = {0};
        buf_printf(&cb, "{\"input_tokens\":%d}\n", creq.prompt.len);
        http_response(fd, s->enable_cors, 200, "application/json", cb.ptr);
        buf_free(&cb);
        request_free(&creq);
        http_request_free(&hr);
        goto done;
    }

    request req;
    char err[160];
    bool ok;
    ok = false;
    int ctx_size;
    ctx_size = pulsar_session_ctx(s->sess);
    if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/messages")) {
        ok = parse_anthropic_request(s->engine, s, hr.body, s->default_tokens,
                                     ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/chat/completions")) {
        ok = parse_chat_request(s->engine, s, hr.body, s->default_tokens,
                                ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/responses")) {
        ok = parse_responses_request(s->engine, s, hr.body, s->default_tokens,
                                     ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/completions")) {
        ok = parse_completion_request(s->engine, hr.body, s->default_tokens,
                                      ctx_size, &req, err, sizeof(err));
    } else {
        http_error(fd, s->enable_cors, 404, "unknown endpoint");
        http_request_free(&hr);
        goto done;
    }
    if (ok) req.raw_body = xstrndup(hr.body, hr.body_len);
    http_request_free(&hr);
    if (!ok) {
        http_error(fd, s->enable_cors, 400, err);
        goto done;
    }
    if (!req.model_from_request) {
        free(req.model);
        req.model = xstrdup(server_served_model_id(s));
    }
    if (request_exceeds_context(&req, ctx_size)) {
        http_error_context_length_exceeded(fd, s->enable_cors, &req, req.prompt.len, ctx_size);
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
            while (!j.done) pthread_cond_wait(&j.cv, &j.mu);
        }
    }
    if (!enqueued) http_error(fd, s->enable_cors, 503, "server shutting down");
    pthread_cond_destroy(&j.cv);
    pthread_mutex_destroy(&j.mu);
    request_free(&j.req);
done:
    close(fd);
    s->client_done();
    return NULL;
}



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



void configure_client_socket(int fd) {
    struct timeval tv;
    tv.tv_sec = PULSAR_SERVER_IO_TIMEOUT_SEC;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}



void set_client_socket_nonblocking(int fd) {
    /* The inference worker writes streaming responses itself.  Once a request is
     * queued, a blocked socket would block every other request too, so slow
     * clients are failed instead of back-pressuring the model session. */
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) (void)fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}


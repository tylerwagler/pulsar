#include "pulsar_agent_internal.h"



/* ============================================================================
 * Agent KV Store And Session Persistence
 * ============================================================================
 */

char *agent_session_title_from_text(const char *text, size_t text_len,
                                           size_t max_bytes);



/* L264: the agent persists KV exactly as pulsar-server does -- as SEGMENT CHAINS
 * in a content-addressed store (<cache>/segments, pulsar_kvchain.h).  A save
 * writes only the segments past what the store already holds, and every
 * conversation that starts with the same system prompt shares that prompt's
 * segments; the system prompt itself is just the shortest such chain.
 *
 * A SAVED SESSION is the small file <cache>/<sha>.session: its title, creation
 * and last-use times, the model id, and the exact token ids of the transcript.
 * Its name, SHA1(title || created_at_le64), stays fixed across saves.  The KV is
 * not in it: switching to a session restores the deepest chain the store holds
 * for its rendered text and prefills the rest from the exact ids -- so a session
 * whose segments were evicted or stripped comes back by prefill, never by
 * re-tokenising text (L223). */

#define AGENT_SESSION_MAGIC   0x31534150u   /* "PAS1" */
#define AGENT_SESSION_VERSION 1u

static void agent_put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void agent_put64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static uint32_t agent_get32(const uint8_t *p) { uint32_t v = 0; for (int i = 3; i >= 0; i--) v = (v << 8) | p[i]; return v; }
static uint64_t agent_get64(const uint8_t *p) { uint64_t v = 0; for (int i = 7; i >= 0; i--) v = (v << 8) | p[i]; return v; }
static uint64_t agent_fnv(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

/* Fixed header: magic, version, model id, title bytes, token count (u32 each),
 * created_at, last_used (u64 each) -> 36 bytes; then the title, the ids (u32
 * each), and an FNV-1a digest of everything before it. */
#define AGENT_SESSION_HEADER 36u

void agent_session_file_free(agent_session_file *f) {
    free(f->title);
    pulsar_tokens_free(&f->tokens);
    memset(f, 0, sizeof(*f));
}

bool agent_session_file_write(const char *path, const agent_session_file *f, char *err, size_t err_len) {
    const size_t title_len = f->title ? strlen(f->title) : 0;
    const uint32_t n = f->tokens.len > 0 ? (uint32_t)f->tokens.len : 0u;
    if (title_len > UINT32_MAX) {
        snprintf(err, err_len, "agent session title is too large");
        return false;
    }
    const size_t bytes = AGENT_SESSION_HEADER + title_len + (size_t)n * 4u;
    uint8_t *buf = (uint8_t *)agent_xmalloc(bytes + 8u);
    agent_put32(buf, AGENT_SESSION_MAGIC);
    agent_put32(buf + 4, AGENT_SESSION_VERSION);
    agent_put32(buf + 8, f->model_id);
    agent_put32(buf + 12, (uint32_t)title_len);
    agent_put32(buf + 16, n);
    agent_put64(buf + 20, f->created_at);
    agent_put64(buf + 28, f->last_used);
    if (title_len) memcpy(buf + AGENT_SESSION_HEADER, f->title, title_len);
    for (uint32_t i = 0; i < n; i++)
        agent_put32(buf + AGENT_SESSION_HEADER + title_len + (size_t)i * 4u, (uint32_t)f->tokens.v[i]);
    agent_put64(buf + bytes, agent_fnv(1469598103934665603ull, buf, bytes));

    agent_buf tmpl = {0};
    agent_buf_puts(&tmpl, path);
    agent_buf_puts(&tmpl, ".tmp.XXXXXX");
    char *tmp = agent_buf_take(&tmpl);
    const int fd = mkstemp(tmp);
    bool ok = fd >= 0;
    int saved_errno = ok ? 0 : errno;
    if (ok) {
        size_t off = 0;
        while (ok && off < bytes + 8u) {
            const ssize_t w = write(fd, buf + off, bytes + 8u - off);
            if (w <= 0) { saved_errno = errno; ok = false; } else off += (size_t)w;
        }
        if (ok && fsync(fd) != 0) { saved_errno = errno; ok = false; }
        if (close(fd) != 0 && ok) { saved_errno = errno; ok = false; }
        if (ok && rename(tmp, path) != 0) { saved_errno = errno; ok = false; }
        if (!ok) unlink(tmp);
    }
    if (!ok) snprintf(err, err_len, "%s", saved_errno ? strerror(saved_errno) : "failed to write session file");
    free(tmp);
    free(buf);
    return ok;
}

bool agent_session_file_read(const char *path, agent_session_file *f, char *err, size_t err_len) {
    memset(f, 0, sizeof(*f));
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        snprintf(err, err_len, "%s", strerror(errno));
        return false;
    }
    uint8_t h[AGENT_SESSION_HEADER];
    bool ok = fread(h, 1, sizeof(h), fp) == sizeof(h) && agent_get32(h) == AGENT_SESSION_MAGIC &&
              agent_get32(h + 4) == AGENT_SESSION_VERSION;
    const uint32_t title_len = ok ? agent_get32(h + 12) : 0u;
    const uint32_t n = ok ? agent_get32(h + 16) : 0u;
    /* Sizes are checked against the file before anything is allocated: one
     * corrupt header must not drive a huge allocation on every listing. */
    struct stat st;
    const uint64_t want = (uint64_t)AGENT_SESSION_HEADER + title_len + (uint64_t)n * 4u + 8u;
    ok = ok && fstat(fileno(fp), &st) == 0 && (uint64_t)st.st_size == want && n <= (uint32_t)INT_MAX;
    uint8_t *rest = NULL;
    if (ok) {
        rest = (uint8_t *)agent_xmalloc((size_t)(want - AGENT_SESSION_HEADER));
        ok = fread(rest, 1, (size_t)(want - AGENT_SESSION_HEADER), fp) == (size_t)(want - AGENT_SESSION_HEADER);
    }
    fclose(fp);
    if (ok) {
        const size_t body = (size_t)(want - AGENT_SESSION_HEADER - 8u);
        const uint64_t dg = agent_fnv(agent_fnv(1469598103934665603ull, h, sizeof(h)), rest, body);
        ok = dg == agent_get64(rest + body);
    }
    if (!ok) {
        snprintf(err, err_len, "not a readable agent session file");
        free(rest);
        return false;
    }
    f->model_id = agent_get32(h + 8);
    f->created_at = agent_get64(h + 20);
    f->last_used = agent_get64(h + 28);
    f->title = (char *)agent_xmalloc((size_t)title_len + 1u);
    memcpy(f->title, rest, title_len);
    f->title[title_len] = '\0';
    if (n) {
        f->tokens.v = (int *)agent_xmalloc((size_t)n * sizeof(int));
        for (uint32_t i = 0; i < n; i++) f->tokens.v[i] = (int)agent_get32(rest + title_len + (size_t)i * 4u);
        f->tokens.len = f->tokens.cap = (int)n;
    }
    free(rest);
    return true;
}



bool agent_kv_open(agent_worker *w) {
    char *dir = pulsar_kvtext_path_join(w->cache_dir, "segments");
    w->kv = pulsar_segstore_open(dir, (uint64_t)AGENT_KV_BUDGET_MB << 20,
                                 pulsar_segstore_identity((uint32_t)pulsar_engine_model_id(w->engine),
                                                          (uint32_t)pulsar_engine_routed_quant_bits(w->engine)),
                                 NULL, NULL);
    if (!w->kv)
        fprintf(stderr, "pulsar-agent: KV cache %s unusable; sessions save without cached KV\n", dir);
    free(dir);
    return w->kv != NULL;
}



/* Load the deepest stored chain for `tokens`' rendered text into the live
 * session; returns the tokens it covers (0: none).  The caller then syncs the
 * exact ids on top: the loaded stretch is a prefix of them when the render
 * matches, and sync rebuilds where it does not. */
int agent_kv_load(agent_worker *w, const pulsar_tokens *tokens) {
    if (!w->kv || tokens->len <= 0) return 0;
    size_t text_len = 0;
    char *text = pulsar_kvtext_render_tokens_text(w->engine, tokens, &text_len);
    if (!text) return 0;
    pulsar_segstore_seg chain[PULSAR_KVCHAIN_MAX];
    int n = 0;
    char err[384];
    const int cached = pulsar_kvchain_restore(w->kv, w->engine, w->session, text, text_len, NULL, 0, 0, chain,
                                              PULSAR_KVCHAIN_MAX, &n, err, sizeof(err));
    if (!cached && err[0]) agent_trace(w, "kv restore refused: %s", err);
    free(text);
    return cached;
}



/* Extend the store's chain for the live session's history (pulsar_kvchain).  A
 * failure is reported, not fatal: the session's tokens are what a save keeps. */
void agent_kv_persist(agent_worker *w, const char *what) {
    if (!w->kv) return;
    pulsar_kvchain_persist_result r;
    pulsar_kvchain_persist(w->kv, w->engine, w->session, 0, NULL, &r);
    if (r.err[0]) {
        agent_buf b = {0};
        agent_buf_puts(&b, "pulsar-agent: ");
        agent_buf_puts(&b, what);
        agent_buf_puts(&b, " KV not fully cached: ");
        agent_buf_puts(&b, r.err);
        char *msg = agent_buf_take(&b);
        if (w->cfg->non_interactive) {
            fprintf(stderr, "%s\n", msg);
        } else {
            agent_publish(w, "\n", 1);
            agent_publish(w, msg, strlen(msg));
            agent_publish(w, "\n", 1);
        }
        free(msg);
    }
    if (r.written)
        agent_trace(w, "%s kv persisted chain to %d (%d new segments, %.1f MiB)", what, r.end, r.written,
                    (double)r.bytes / (1024.0 * 1024.0));
}



/* The system prompt's stored chain tip ("" when none): the stretch every session
 * shares, which releasing one session's KV must never take. */
void agent_kv_system_tip(agent_worker *w, char tip[41]) {
    tip[0] = '\0';
    if (!w->kv) return;
    pulsar_tokens sys = {0};
    agent_worker_build_system_tokens(w, &sys);
    size_t text_len = 0;
    char *text = pulsar_kvtext_render_tokens_text(w->engine, &sys, &text_len);
    pulsar_segstore_seg chain[PULSAR_KVCHAIN_MAX];
    const int n = text ? pulsar_segstore_lookup(w->kv, text, text_len, chain, PULSAR_KVCHAIN_MAX) : 0;
    if (n > 0) memcpy(tip, chain[n - 1].key, 41);
    free(text);
    pulsar_tokens_free(&sys);
}



void agent_worker_build_system_tokens(agent_worker *w, pulsar_tokens *out) {
    pulsar_chat_begin(w->engine, out);
    pulsar_chat_append_lead_in(w->engine, out, w->cfg->gen.system && w->cfg->gen.system[0],
                               w->cfg->gen.think_mode);
    agent_append_system_prompt(w->engine, out, w->cfg->gen.system);
}



void agent_publish_system_status(agent_worker *w, const char *msg) {
    if (w->cfg->non_interactive) return;
    if (isatty(STDOUT_FILENO)) {
        static const char marker[] = "\x1b[33m✦ \x1b[38;5;218m";
        agent_publish(w, marker, sizeof(marker) - 1);
        agent_publish(w, msg, strlen(msg));
        agent_publish(w, "\x1b[0m\n", strlen("\x1b[0m\n"));
    } else {
        agent_publish(w, "✦ ", strlen("✦ "));
        agent_publish(w, msg, strlen(msg));
        agent_publish(w, "\n", 1);
    }
}









/* When a model turn finishes with a tool call, queued user messages should not
 * preempt that tool.  The worker asks the UI thread for the queue contents only
 * after the tool result is appended, so the next model input can contain both
 * the tool observation and the user's pending correction. */
char *worker_request_queued_user_drain(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->queued_user_drain_pending = true;
    w->queued_user_drain_answered = false;
    free(w->queued_user_drain_text);
    w->queued_user_drain_text = NULL;
    agent_wake_locked(w);
    pthread_cond_signal(&w->cond);
    while (!w->stop && !w->queued_user_drain_answered)
        pthread_cond_wait(&w->cond, &w->mu);
    char *text = w->queued_user_drain_text;
    w->queued_user_drain_text = NULL;
    w->queued_user_drain_pending = false;
    w->queued_user_drain_answered = false;
    pthread_mutex_unlock(&w->mu);
    return text;
}



bool worker_take_queued_user_drain_request(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool pending = w->queued_user_drain_pending;
    if (pending) w->queued_user_drain_pending = false;
    pthread_mutex_unlock(&w->mu);
    return pending;
}



void worker_answer_queued_user_drain(agent_worker *w, char *text) {
    pthread_mutex_lock(&w->mu);
    free(w->queued_user_drain_text);
    w->queued_user_drain_text = text;
    w->queued_user_drain_answered = true;
    pthread_cond_signal(&w->cond);
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}



/* Synchronize the live DS4 session to a transcript.  This is the agent's main
 * cache-saving operation: if the requested transcript extends the live session,
 * only the suffix is prefetched; otherwise the DS4 session rebuilds from the
 * longest common prefix it can retain. */
int agent_worker_sync_tokens(agent_worker *w, const pulsar_tokens *tokens,
                                    bool publish_progress,
                                    char *err, size_t err_len) {
    int old_pos = pulsar_session_pos(w->session);
    int common = pulsar_session_common_prefix(w->session, tokens);
    int cached = common == old_pos && tokens->len >= old_pos ? common : 0;
    int suffix = tokens->len - cached;
    if (suffix < 0) suffix = tokens->len;

    if (publish_progress) {
        pthread_mutex_lock(&w->mu);
        unsigned prefill_label = w->status.state == AGENT_WORKER_PREFILL ?
            w->status.prefill_label : agent_next_prefill_label();
        w->status.state = AGENT_WORKER_PREFILL;
        w->progress_base = cached;
        w->progress_started_at = agent_now_sec();
        w->status.prefill_done = 0;
        w->status.prefill_total = suffix;
        w->status.prefill_label = prefill_label;
        w->status.prefill_tps = 0.0;
        w->status.generated = 0;
        w->status.gen_tps = 0.0;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
    }

    pulsar_session_set_progress(w->session, publish_progress ? worker_progress_cb : NULL,
                             publish_progress ? w : NULL);
    pulsar_session_set_display_progress(w->session,
                                     publish_progress ? worker_progress_cb : NULL,
                                     publish_progress ? w : NULL);
    pulsar_session_set_cancel(w->session, worker_cancel_session_cb, w);
    int rc = pulsar_session_sync(w->session, tokens, err, err_len);
    pulsar_session_set_cancel(w->session, NULL, NULL);
    pulsar_session_set_progress(w->session, NULL, NULL);
    pulsar_session_set_display_progress(w->session, NULL, NULL);
    return rc;
}



/* Start a new session at the system/tool prompt.  Its KV is the shortest chain
 * in the store: restored when the rendered prompt is unchanged, prefilled and
 * stored when not.  The store is per model and routed format, so switching
 * model families never restores incompatible KV. */
bool agent_worker_reset_to_sysprompt(agent_worker *w, char *err, size_t err_len) {
    pulsar_tokens sys = {0};
    agent_worker_build_system_tokens(w, &sys);
    pulsar_tokens_free(&w->transcript);
    pulsar_tokens_copy(&w->transcript, &sys);
    pulsar_tokens_free(&sys);
    const int cached = agent_kv_load(w, &w->transcript);
    agent_trace(w, "sysprompt kv restored=%d of %d tokens", cached, w->transcript.len);
    if (w->kv && cached == 0) agent_publish_system_status(w, "Updating system prompt cache...");
    if (agent_worker_sync_tokens(w, &w->transcript, true, err, err_len) != 0) return false;
    if (cached < w->transcript.len) agent_kv_persist(w, "system prompt");

    agent_worker_note_system_prompt_seen(w);
    pthread_mutex_lock(&w->mu);
    w->user_activity = false;
    w->session_dirty = false;
    w->status.state = AGENT_WORKER_IDLE;
    w->status.prefill_done = 0;
    w->status.prefill_total = 0;
    w->status.prefill_tps = 0.0;
    w->status.generated = 0;
    w->status.gen_tps = 0.0;
    w->status.greedy_sampling = false;
    w->status.error[0] = '\0';
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
    w->datetime_context_injected = false;
    agent_worker_clear_session_identity(w);
    return true;
}



bool agent_worker_has_user_session(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool yes = w->user_activity;
    pthread_mutex_unlock(&w->mu);
    return yes;
}



bool agent_worker_needs_save(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool yes = w->user_activity && w->session_dirty;
    pthread_mutex_unlock(&w->mu);
    return yes;
}



/* Save the current session under its stable agent identity.  The worker owns
 * the live KV, so busy /save requests are deferred until a stable append-only
 * point and then executed by the worker thread. */
bool agent_worker_save_session_now(agent_worker *w, char sha_out[41],
                                          int *tokens_out,
                                          char *err, size_t err_len) {
    if (!agent_worker_has_user_session(w)) {
        snprintf(err, err_len, "nothing to save");
        return false;
    }

    if (agent_worker_sync_tokens(w, &w->transcript, false, err, err_len) != 0)
        return false;
    if (!w->session_title) {
        size_t text_len = 0;
        char *text = pulsar_kvtext_render_tokens_text(w->engine, &w->transcript, &text_len);
        w->session_title = agent_session_title_from_text(text ? text : "", text ? text_len : 0, 0);
        free(text);
    }
    if (w->session_created_at == 0)
        w->session_created_at = (uint64_t)time(NULL);

    agent_kv_persist(w, "session");
    char sha[41];
    agent_session_identity_sha(w->session_title, w->session_created_at, sha);
    char *path = agent_session_path_for_sha(w->cache_dir, sha);
    agent_session_file f;
    memset(&f, 0, sizeof(f));
    f.title = w->session_title;
    f.created_at = w->session_created_at;
    f.last_used = (uint64_t)time(NULL);
    f.model_id = (uint32_t)pulsar_engine_model_id(w->engine);
    f.tokens = w->transcript;
    const bool ok = agent_session_file_write(path, &f, err, err_len);
    if (ok) {
        memcpy(w->session_sha, sha, sizeof(w->session_sha));
        if (sha_out) memcpy(sha_out, sha, 41);
        pthread_mutex_lock(&w->mu);
        w->session_dirty = false;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
        if (tokens_out) *tokens_out = w->transcript.len;
    }
    free(path);
    return ok;
}



bool agent_worker_save_session(agent_worker *w, char *err, size_t err_len) {
    if (!worker_is_idle(w)) {
        snprintf(err, err_len, "model is busy");
        return false;
    }
    char sha[41];
    int tokens = 0;
    bool ok = agent_worker_save_session_now(w, sha, &tokens, err, err_len);
    if (ok) printf("saved session %.8s (%d tokens)\n", sha, tokens);
    return ok;
}

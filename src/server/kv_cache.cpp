#include "pulsar_server_internal.h"
#include "pulsar_lock.hpp"
#include "lib/pulsar_segstore.h"
#include "lib/pulsar_kvchain.h"

#include <string>




kv_cache_options kv_cache_default_options(void) {
    kv_cache_options o = {};
    o.min_tokens = 512;
    o.sys_prefix_align = 2048;
    o.sys_prefix_margin = 2048;
    return o;
}



void le_put32(uint8_t *p, uint32_t v) {
    pulsar_kvtext_le_put32(p, v);
}




static uint32_t le_get32(const uint8_t *p) {
    return pulsar_kvtext_le_get32(p);
}




#ifdef PULSAR_SERVER_TEST

void sha1_bytes_hex(const void *ptr, size_t len, char out[41]) {
    pulsar_kvtext_sha1_bytes_hex(ptr, len, out);
}


#endif


bool id_list_contains(const stop_list *ids, const char *id) {
    if (!ids || !id || !id[0]) return false;
    for (int i = 0; i < ids->len; i++) {
        if (ids->v[i] && !strcmp(ids->v[i], id)) return true;
    }
    return false;
}



void id_list_push_unique(stop_list *ids, const char *id) {
    if (!ids || !id || !id[0] || id_list_contains(ids, id)) return;
    stop_list_push(ids, xstrdup(id));
}



void id_list_free(stop_list *ids) {
    stop_list_clear(ids);
    free(ids->v);
    memset(ids, 0, sizeof(*ids));
}



void collect_tool_call_ids(const chat_msgs *msgs, stop_list *ids) {
    if (!msgs || !ids) return;
    for (int i = 0; i < msgs->len; i++) {
        id_list_push_unique(ids, msgs->v[i].tool_call_id);
        for (int j = 0; j < msgs->v[i].tool_call_ids_len; j++) {
            id_list_push_unique(ids, msgs->v[i].tool_call_ids[j]);
        }
        const tool_calls *calls = &msgs->v[i].calls;
        for (int j = 0; j < calls->len; j++) {
            id_list_push_unique(ids, calls->v[j].id);
        }
    }
}



char *path_join(const char *dir, const char *name) {
    return pulsar_kvtext_path_join(dir, name);
}








const char *find_next_dsml_tool_block(const char *p, const char **end_out) {
    struct block_form {
        const char *start;
        const char *end;
    } forms[] = {
        /* Every spelling either family renders, so the scanner finds the last
         * tool block whatever produced the transcript (the shipped 0731
         * artifacts sample V4's unspaced tags, V4.1 the spaced ones). */
        {"\n\n" PULSAR_TOOL_CALLS_START, PULSAR_TOOL_CALLS_END},
        {PULSAR_TOOL_CALLS_START, PULSAR_TOOL_CALLS_END},
        {"\n\n" PULSAR_TOOL_CALLS_START_SHORT, PULSAR_TOOL_CALLS_END_SHORT},
        {PULSAR_TOOL_CALLS_START_SHORT, PULSAR_TOOL_CALLS_END_SHORT},
        {"\n\n" PULSAR_TOOL_CALLS_START_V4, PULSAR_TOOL_CALLS_END_V4},
        {PULSAR_TOOL_CALLS_START_V4, PULSAR_TOOL_CALLS_END_V4},
        {"\n\n" PULSAR_TOOL_CALLS_START_SHORT_V4, PULSAR_TOOL_CALLS_END_SHORT_V4},
        {PULSAR_TOOL_CALLS_START_SHORT_V4, PULSAR_TOOL_CALLS_END_SHORT_V4},
        {"\n\n<tool_calls>", "</tool_calls>"},
        {"<tool_calls>", "</tool_calls>"},
    };

    const char *best = NULL;
    const char *best_end = NULL;
    for (size_t i = 0; i < sizeof(forms) / sizeof(forms[0]); i++) {
        const char *s = strstr(p, forms[i].start);
        if (!s || (best && s >= best)) continue;
        const char *e = strstr(s, forms[i].end);
        if (!e) continue;
        best = s;
        best_end = e + strlen(forms[i].end);
    }
    if (end_out) *end_out = best_end;
    return best;
}




bool server::kv_tool_map_measure_locked(const char *text,
                                       uint32_t *count_out,
                                       uint64_t *bytes_out) {
    auto *s = this;
    uint32_t count = 0;
    uint64_t bytes = KV_TOOL_MAP_HEADER;
    uint64_t scan = ++s->tool_mem.scan_clock;
    const char *p = text;
    for (;;) {
        const char *end = NULL;
        const char *start = server_family_for_engine(s->engine)->find_call_block(p, &end);
        if (!start || !end) break;
        tool_memory_block *b =
            tool_memory_find_block_locked(&s->tool_mem, start, (size_t)(end - start));
        if (b && b->seen != scan) {
            b->seen = scan;
            for (tool_memory_entry *e = b->entries; e; e = e->block_next) {
                size_t id_len = strlen(e->id);
                size_t dsml_len = b->len;
                if (id_len > UINT32_MAX || dsml_len > UINT32_MAX) continue;
                if (count == UINT32_MAX) return false;
                if (UINT64_MAX - bytes < 8u ||
                    UINT64_MAX - bytes - 8u < (uint64_t)id_len ||
                    UINT64_MAX - bytes - 8u - (uint64_t)id_len < (uint64_t)dsml_len)
                    return false;
                count++;
                bytes += 8u + (uint64_t)id_len + (uint64_t)dsml_len;
            }
        }
        p = end;
    }
    if (count == 0) bytes = 0;
    if (count_out) *count_out = count;
    if (bytes_out) *bytes_out = bytes;
    return true;
}



bool server::kv_tool_map_serialized_size(const char *text,
                                        uint64_t *bytes_out) {
    auto *s = this;
    if (bytes_out) *bytes_out = 0;
    if (!s || !text || !text[0]) return true;

    pthread_mutex_lock(&s->tool_mu);
    bool ok = s->kv_tool_map_measure_locked(text, NULL, bytes_out);
    pthread_mutex_unlock(&s->tool_mu);
    return ok;
}



bool server::kv_tool_map_write(FILE *fp, const char *text,
                              uint64_t *written_bytes) {
    auto *s = this;
    if (written_bytes) *written_bytes = 0;
    if (!s || !fp || !text || !text[0]) return true;

    pulsar::ScopedLock lk(&s->tool_mu);
    uint32_t count = 0;
    uint64_t bytes = 0;
    bool ok = s->kv_tool_map_measure_locked(text, &count, &bytes);
    if (!ok) return false;
    if (count == 0) return true;

    uint8_t h[KV_TOOL_MAP_HEADER];
    h[0] = KV_TOOL_MAP_MAGIC0;
    h[1] = KV_TOOL_MAP_MAGIC1;
    h[2] = KV_TOOL_MAP_MAGIC2;
    h[3] = KV_TOOL_MAP_VERSION;
    le_put32(h + 4, count);
    ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h);

    uint64_t scan = ++s->tool_mem.scan_clock;
    const char *p = text;
    for (;;) {
        const char *end = NULL;
        const char *start = server_family_for_engine(s->engine)->find_call_block(p, &end);
        if (!start || !end || !ok) break;
        tool_memory_block *b =
            tool_memory_find_block_locked(&s->tool_mem, start, (size_t)(end - start));
        if (b && b->seen != scan) {
            b->seen = scan;
            for (tool_memory_entry *e = b->entries; ok && e; e = e->block_next) {
                size_t id_len = strlen(e->id);
                size_t dsml_len = b->len;
                if (id_len > UINT32_MAX || dsml_len > UINT32_MAX) continue;
                uint8_t lens[8];
                le_put32(lens, (uint32_t)id_len);
                le_put32(lens + 4, (uint32_t)dsml_len);
                ok = fwrite(lens, 1, sizeof(lens), fp) == sizeof(lens) &&
                     fwrite(e->id, 1, id_len, fp) == id_len &&
                     fwrite(b->dsml, 1, dsml_len, fp) == dsml_len;
            }
        }
        p = end;
    }

    if (ok && written_bytes) *written_bytes = bytes;
    return ok;
}



int server::kv_tool_map_load_from_pos(FILE *fp, const stop_list *wanted) {
    auto *s = this;
    if (!s || !fp) return 0;
    uint8_t h[KV_TOOL_MAP_HEADER];
    size_t n = fread(h, 1, sizeof(h), fp);
    if (n == 0 && feof(fp)) return 0;
    if (n != sizeof(h)) return 0;
    if (h[0] != KV_TOOL_MAP_MAGIC0 || h[1] != KV_TOOL_MAP_MAGIC1 ||
        h[2] != KV_TOOL_MAP_MAGIC2 || h[3] != KV_TOOL_MAP_VERSION) return 0;

    uint32_t count = le_get32(h + 4);
    if ((uint64_t)count > (uint64_t)tool_memory_max_entries(&s->tool_mem) * 4u) return 0;
    int loaded = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint8_t lens[8];
        if (fread(lens, 1, sizeof(lens), fp) != sizeof(lens)) return loaded;
        uint32_t id_len = le_get32(lens);
        uint32_t dsml_len = le_get32(lens + 4);
        if (id_len == 0 || id_len > 256 || dsml_len == 0 ||
            dsml_len > PULSAR_TOOL_MEMORY_MAX_BYTES) return loaded;
        char *id = (char *)server_xmalloc((size_t)id_len + 1);
        char *dsml = (char *)server_xmalloc((size_t)dsml_len + 1);
        bool ok = fread(id, 1, id_len, fp) == id_len &&
                  fread(dsml, 1, dsml_len, fp) == dsml_len;
        id[id_len] = '\0';
        dsml[dsml_len] = '\0';
        if (ok && (!wanted || id_list_contains(wanted, id))) {
            s->tool_memory_put_source(id, dsml, TOOL_MEMORY_DISK);
            loaded++;
        }
        free(id);
        free(dsml);
        if (!ok) return loaded;
    }
    return loaded;
}



/* ===== L264 S4: the disk KV cache is the segment store ====================
 * A conversation persists as a chain of segments, one per grid checkpoint its
 * bank holds (pulsar_segstore.h): each new prefill adds only the segments past
 * the chain's last stored one, so a write costs the new tokens, and the chain's
 * earlier segments stay -- a client that rewrites history further back finds
 * the stretch it still shares.  A restore loads the deepest chain the request's
 * bytes reproduce, ending live at its last checkpoint. */

static void kv_seg_log(void *ud, const char *msg) {
    (void)ud;
    server_log(PULSAR_LOG_KVCACHE, "pulsar-server: kv cache %s", msg);
}

bool kv_cache_open(kv_disk_cache *kc, const char *dir, uint64_t budget_mb, uint32_t model_id,
                   kv_cache_options opt) {
    memset(kc, 0, sizeof(*kc));
    kc->opt = opt;
    kc->st = pulsar_segstore_open(dir, budget_mb << 20, model_id, kv_seg_log, NULL);
    if (!kc->st) return false;
    kc->dir = xstrdup(dir);
    kc->enabled = true;
    return true;
}

void kv_cache_close(kv_disk_cache *kc) {
    pulsar_segstore_close(kc->st);
    free(kc->dir);
    memset(kc, 0, sizeof(*kc));
}

char *render_tokens_text(pulsar_engine *engine, const pulsar_tokens *tokens, size_t *out_len) {
    return pulsar_kvtext_render_tokens_text(engine, tokens, out_len);
}

static bool byte_prefix_match(const char *text, size_t text_len,
                              const char *prefix, size_t prefix_len) {
    return pulsar_kvtext_byte_prefix_match(text, text_len, prefix, prefix_len);
}

void tokens_copy_prefix(pulsar_tokens *dst, const pulsar_tokens *src, int n) {
    pulsar_kvtext_tokens_copy_prefix(dst, src, n);
}

void build_prompt_from_exact_prefix_and_text_suffix(
        pulsar_engine *engine,
        const pulsar_tokens *exact_prefix,
        const char *suffix_text,
        const pulsar_text_span *spans,
        uint32_t n_spans,
        pulsar_tokens *out)
{
    pulsar_kvtext_build_prompt_from_exact_prefix_and_text_suffix(
        engine, exact_prefix, suffix_text, spans, n_spans, out);
}

/* The stable rendered chat prefix is everything before the user message that
 * asks this specific task.  Some clients put stable user-role scaffolding
 * first, so the anchor is the last user marker before the first assistant. */
int kv_cache_chat_anchor_pos(const kv_disk_cache *kc, const pulsar_tokens *prompt, const pulsar_turn_markers *m) {
    if (!prompt || !m || m->n_user <= 0 || m->n_assistant <= 0) return -1;
    int last_user = -1;
    for (int i = 0; i < prompt->len; i++) {
        const int role = pulsar_turn_marker_at(m, prompt->v, prompt->len, i);
        if (role == 2) break;
        if (role == 1) last_user = i;
    }
    return last_user >= kc->opt.min_tokens ? last_user : -1;
}

/* Back the anchor cut off below harness-injected preamble jitter and land it on
 * an alignment boundary (a multiple of the resume grid, so the cold prefill that
 * ends there leaves a grid checkpoint).  0 when nothing useful survives. */
int kv_cache_sys_prefix_cut(const kv_disk_cache *kc, int anchor) {
    if (anchor < kc->opt.min_tokens) return 0;
    int cut = anchor - kc->opt.sys_prefix_margin;
    if (kc->opt.sys_prefix_align > 0) cut -= cut % kc->opt.sys_prefix_align;
    return cut >= kc->opt.min_tokens ? cut : 0;
}

namespace {
bool seg_trailer_size(void *ud, const char *text, uint64_t *bytes) {
    return ((server *)ud)->kv_tool_map_serialized_size(text, bytes);
}
int seg_trailer_write(FILE *fp, void *ud, const char *text, char *err, size_t errlen) {
    uint64_t written = 0;
    if (((server *)ud)->kv_tool_map_write(fp, text, &written)) return 0;
    snprintf(err, errlen, "tool map write failed");
    return 1;
}
}  // namespace

int server::kv_cache_persist(session_slot *sl, const char *reason) {
    auto *s = this;
    (void)sl;   /* the caller installed the slot's bank */
    if (!s->kv.enabled) return 0;
    /* A "continued" store fires from the prefill's progress callback; on a TP
     * group that is inside the mirrored sync, where the workers read only
     * chunk verdicts.  The prompt-end store after it persists the same
     * checkpoints. */
    if (pulsar_session_in_mirrored_sync(s->sess)) return 0;
    const double t0 = server_now_sec();
    const pulsar_kvchain_trailer tool_maps = { s, seg_trailer_size, seg_trailer_write };
    pulsar_kvchain_persist_result r;
    pulsar_kvchain_persist(s->kv.st, s->engine, s->sess, s->kv.opt.min_tokens, &tool_maps, &r);
    if (r.err[0])
        server_log(PULSAR_LOG_WARNING, "pulsar-server: kv cache chain not fully stored (reason=%s): %s", reason, r.err);
    if (r.written)
        server_log(PULSAR_LOG_KVCACHE,
                   "pulsar-server: kv cache persisted reason=%s chain to %d (%d new segments, %.1f MiB) %.1f ms",
                   reason, r.end, r.written, (double)r.bytes / (1024.0 * 1024.0), (server_now_sec() - t0) * 1000.0);
    return r.written;
}

int server::kv_cache_try_load_text(session_slot *sl, const char *prompt_text,
                                  const pulsar_text_span *prompt_spans,
                                  uint32_t prompt_n_spans,
                                  pulsar_tokens *effective_prompt,
                                  char **loaded_key_out,
                                  bool responses_protocol) {
    auto *s = this;
    (void)sl;   /* the caller installed the slot's bank */
    if (loaded_key_out) *loaded_key_out = NULL;
    if (!s->kv.enabled || !prompt_text) return 0;
    const size_t prompt_bytes = strlen(prompt_text);
    const double t0 = server_now_sec();
    pulsar_segstore_seg chain[PULSAR_KVCHAIN_MAX];
    int n = 0;
    char err[384];
    const int G = pulsar_kvchain_restore(s->kv.st, s->engine, s->sess, prompt_text, prompt_bytes,
                                         s->kv.opt.min_tokens, chain, PULSAR_KVCHAIN_MAX, &n, err, sizeof(err));
    if (G == 0) {
        if (err[0]) server_log(PULSAR_LOG_WARNING, "pulsar-server: kv cache %s", err);
        return 0;
    }
    const pulsar_tokens *loaded = pulsar_session_tokens(s->sess);
    const uint64_t text_end = chain[n - 1].text_end;
    /* The chain's tool maps: the exact DSML of every tool call inside it. */
    for (int i = 0; i < n; i++) {
        FILE *fp = pulsar_segstore_open_trailer(s->kv.st, chain[i].key, NULL);
        if (!fp) continue;
        s->kv_tool_map_load_from_pos(fp, NULL);
        fclose(fp);
    }
    if (effective_prompt) {
        /* The lookup was by bytes; the graph holds the exact token history.
         * Build the prompt from that history and tokenize only the text after
         * it -- carrying the client-data ranges into the suffix's coordinates
         * so client bytes stay plain text there too (L223). */
        uint32_t tail_n = 0;
        pulsar_text_span *tail = pulsar_text_spans_slice(prompt_spans, prompt_n_spans, (uint32_t)text_end,
                                                         (uint32_t)(prompt_bytes - text_end), &tail_n);
        build_prompt_from_exact_prefix_and_text_suffix(s->engine, loaded, prompt_text + text_end,
                                                       tail, tail_n, effective_prompt);
        free(tail);
    }
    if (loaded_key_out) *loaded_key_out = xstrdup(chain[n - 1].key);
    server_log(PULSAR_LOG_KVCACHE,
               "pulsar-server: kv cache hit%s chain of %d to %u (text %llu of %zu bytes) load=%.1f ms",
               responses_protocol ? " [responses]" : "", n, chain[n - 1].G,
               (unsigned long long)text_end, prompt_bytes, (server_now_sec() - t0) * 1000.0);
    return (int)chain[n - 1].G;
}

int server::kv_cache_try_load(session_slot *sl, const request *req,
                             pulsar_tokens *effective_prompt,
                             char **loaded_key_out) {
    auto *s = this;
    return s->kv_cache_try_load_text(sl, req ? req->prompt_text : NULL,
                                  req ? req->prompt_spans : NULL,
                                  req ? req->prompt_n_spans : 0,
                                  effective_prompt, loaded_key_out,
                                  req && req->api == API_RESPONSES);
}

void server::kv_cache_discard_failed_chain(const char *key) {
    auto *s = this;
    if (!key || !s->kv.enabled) return;
    pulsar_segstore_drop(s->kv.st, key);
    server_log(PULSAR_LOG_KVCACHE, "pulsar-server: kv cache discarded reason=prefill-failed segment=%s", key);
}

void server::kv_cache_restore_tool_memory_for_messages(const chat_msgs *msgs) {
    auto *s = this;
    if (!s || !s->kv.enabled || !msgs) return;
    stop_list wanted = {0};
    collect_tool_call_ids(msgs, &wanted);
    /* Only ids MISSING from the in-memory tool map justify touching disk: a
     * live conversation's ids were remembered at generation time. */
    int keep = 0;
    for (int i = 0; i < wanted.len; i++) {
        if (!s->tool_memory_has_id(wanted.v[i])) wanted.v[keep++] = wanted.v[i];
        else free(wanted.v[i]);
    }
    wanted.len = keep;
    if (wanted.len > 0) {
        struct walk { server *s; stop_list *wanted; } w = { s, &wanted };
        pulsar_segstore_foreach_trailer(s->kv.st, [](FILE *fp, uint64_t, const char *, size_t, void *ud) {
            walk *wk = (walk *)ud;
            wk->s->kv_tool_map_load_from_pos(fp, wk->wanted);
            for (int i = 0; i < wk->wanted->len; i++)
                if (!wk->s->tool_memory_has_id(wk->wanted->v[i])) return true;
            return false;   /* every missing id is back */
        }, &w);
    }
    id_list_free(&wanted);
}



/* Tool-output-only Responses continuation.
 *
 * Some clients send just the new tool outputs after a tool call.  There is no
 * long visible prefix to match in that shape; the call_id itself is the
 * protocol binding to the previous live assistant output.  Use it only when the
 * remembered live frontier and call-id set match exactly. */
int server::responses_live_continuation_prompt(session_slot *sl,
                                              const request *req,
                                              int live_pos,
                                              pulsar_tokens *effective_prompt,
                                              int *matched_ids) {
    auto *s = this;
    if (!s || !req || !effective_prompt) return 0;
    if (req->api != API_RESPONSES || !req->responses_live_suffix_text) return 0;
    if (req->responses_live_call_ids.len == 0) return 0;
    if (!s->responses_live_matches_request(sl, &req->responses_live_call_ids,
                                        live_pos)) return 0;

    const pulsar_tokens *live_tokens = pulsar_session_tokens(s->sess);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->responses_live_suffix_text,
        req->responses_live_suffix_spans, req->responses_live_suffix_n_spans,
        effective_prompt);
    if (matched_ids) *matched_ids = req->responses_live_call_ids.len;
    return live_tokens->len;
}



/* Tool-result Anthropic continuation.
 *
 * /v1/messages has no server-side response object like the OpenAI Responses
 * API, but its tool_use_id is still a precise continuation handle inside a live
 * local agent loop.  When the IDs and live token frontier match, continue from
 * the sampled DSML state and append only the user tool_result suffix. */
int server::anthropic_live_continuation_prompt(session_slot *sl,
                                              const request *req,
                                              int live_pos,
                                              pulsar_tokens *effective_prompt,
                                              int *matched_ids) {
    auto *s = this;
    if (!s || !req || !effective_prompt) return 0;
    if (req->api != API_ANTHROPIC || !req->anthropic_live_suffix_text) return 0;
    if (req->anthropic_live_call_ids.len == 0) return 0;
    if (!s->anthropic_live_matches_request(sl, &req->anthropic_live_call_ids,
                                        live_pos)) return 0;

    const pulsar_tokens *live_tokens = pulsar_session_tokens(s->sess);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->anthropic_live_suffix_text,
        req->anthropic_live_suffix_spans, req->anthropic_live_suffix_n_spans,
        effective_prompt);
    if (matched_ids) *matched_ids = req->anthropic_live_call_ids.len;
    return live_tokens->len;
}



/* Visible-replay Responses continuation.
 *
 * Other clients send the full visible transcript on every turn even though the
 * API semantics still make the request a continuation.  For Responses, exact
 * token-prefix matching is the wrong first question: hidden reasoning may be
 * live in KV but absent from the replay by design.  Instead, verify that the
 * request's rendered text begins with the visible transcript remembered at the
 * live frontier.  If it does, continue from the live token prefix and tokenize
 * only the bytes after that visible boundary.
 *
 * If this check fails, DS4 has no special Responses state to trust.  The caller
 * then uses normal token/text/disk matching, which is the correct fallback for
 * cold starts, edits, restarts, or cross-client replays. */
int server::responses_live_visible_prefix_prompt(session_slot *sl,
                                                const request *req,
                                                int live_pos,
                                                pulsar_tokens *effective_prompt) {
    auto *s = this;
    if (!s || !req || !req->prompt_text || !effective_prompt) return 0;
    if (req->api != API_RESPONSES) return 0;

    const size_t prompt_len = strlen(req->prompt_text);
    size_t visible_len = 0;
    pthread_mutex_lock(&s->tool_mu);
    bool ok = sl->responses_live.valid &&
              sl->responses_live.live_tokens == live_pos &&
              sl->responses_live.visible_text &&
              sl->responses_live.visible_len < prompt_len &&
              byte_prefix_match(req->prompt_text, prompt_len,
                                sl->responses_live.visible_text,
                                sl->responses_live.visible_len);
    if (ok) visible_len = sl->responses_live.visible_len;
    pthread_mutex_unlock(&s->tool_mu);
    if (!ok) return 0;

    const pulsar_tokens *live_tokens = pulsar_session_tokens(s->sess);
    if (!live_tokens || live_tokens->len != live_pos) return 0;
    if (!pulsar_kvtext_text_ends_with_live(s->engine, req->prompt_text, visible_len, live_tokens)) {
        server_log(PULSAR_LOG_ERROR,
                   "pulsar-server: visible key REFUSED (live=%d): the key's EOS disagrees "
                   "with the last live token (L196)", live_pos);
        return 0;
    }

    uint32_t tail_n = 0;
    pulsar_text_span *tail_spans =
        pulsar_text_spans_slice(req->prompt_spans, req->prompt_n_spans, visible_len,
                                prompt_len - visible_len, &tail_n);
    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->prompt_text + visible_len,
        tail_spans, tail_n, effective_prompt);
    free(tail_spans);
    return live_tokens->len;
}




/** @file
 *  The disk KV cache (L264): segment-chain persist / find / restore with the restore-vs-recompute price, the
 *  tool-map trailer, and the protocols' live-continuation prompts.
 */
#include "pulsar_server_internal.h"
#include "pulsar_lock.hpp"
#include "lib/pulsar_segstore.h"
#include "lib/pulsar_kvchain.h"

#include <string>




/// @brief The default checkpoint policy: min_tokens 512, sys-prefix align 2048 and margin 2048 (tokens).
/// @return the options
kv_cache_options kv_cache_default_options(void) {
    kv_cache_options o = {};
    o.min_tokens = 512;
    o.sys_prefix_align = 2048;
    o.sys_prefix_margin = 2048;
    return o;
}



/// @brief Store `v` little-endian at `p` (4 bytes).
/// @param p  the destination
/// @param v  the value
void le_put32(uint8_t *p, uint32_t v) {
    pulsar_kvtext_le_put32(p, v);
}




/// @brief Load a little-endian 32-bit value.
/// @param p  the 4 bytes
/// @return the value
static uint32_t le_get32(const uint8_t *p) {
    return pulsar_kvtext_le_get32(p);
}




#ifdef PULSAR_SERVER_TEST

/// @brief Test hook: the SHA-1 of `len` bytes as 40 lowercase hex digits + NUL.
/// @param ptr  the bytes
/// @param len  bytes at `ptr`
/// @param out  receives the digest text
void sha1_bytes_hex(const void *ptr, size_t len, char out[41]) {
    pulsar_kvtext_sha1_bytes_hex(ptr, len, out);
}


#endif


/// @brief Whether `ids` holds `id` (exact string match).
/// @param ids  the id list (NULL: false)
/// @param id   the id (NULL or empty: false)
/// @return true when present
bool id_list_contains(const stop_list *ids, const char *id) {
    if (!ids || !id || !id[0]) return false;
    for (int i = 0; i < ids->len; i++) {
        if (ids->v[i] && !strcmp(ids->v[i], id)) return true;
    }
    return false;
}



/// @brief Append a copy of `id` unless it is empty or already present.
/// @param ids  the id list (NULL: nothing)
/// @param id   the id
void id_list_push_unique(stop_list *ids, const char *id) {
    if (!ids || !id || !id[0] || id_list_contains(ids, id)) return;
    stop_list_push(ids, server_xstrdup(id));
}



/// @brief Free every id and the array, and zero the list.
/// @param ids  the id list
void id_list_free(stop_list *ids) {
    stop_list_clear(ids);
    free(ids->v);
    memset(ids, 0, sizeof(*ids));
}



/// @brief Every tool-call id the conversation names (tool results' ids and assistant calls' ids), each once.
/// @param msgs  the conversation (NULL: nothing)
/// @param ids   receives the ids (appended, owned)
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



/// @brief `dir` + "/" + `name`.
/// @param dir  the directory
/// @param name  the file name
/// @return malloc'd path
char *path_join(const char *dir, const char *name) {
    return pulsar_kvtext_path_join(dir, name);
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
        const char *start = server_tool_family(s)->find_call_block(p, &end);
        if (!start || !end) break;
        tool_memory_block *b =
            tool_memory_find_block_locked(&s->tool_mem, start, (size_t)(end - start));
        if (b && b->seen != scan) {
            b->seen = scan;
            for (tool_memory_entry *e = b->entries; e; e = e->block_next) {
                size_t id_len = strlen(e->id);
                size_t raw_len = b->len;
                if (id_len > UINT32_MAX || raw_len > UINT32_MAX) continue;
                if (count == UINT32_MAX) return false;
                if (UINT64_MAX - bytes < 8u ||
                    UINT64_MAX - bytes - 8u < (uint64_t)id_len ||
                    UINT64_MAX - bytes - 8u - (uint64_t)id_len < (uint64_t)raw_len)
                    return false;
                count++;
                bytes += 8u + (uint64_t)id_len + (uint64_t)raw_len;
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
        const char *start = server_tool_family(s)->find_call_block(p, &end);
        if (!start || !end || !ok) break;
        tool_memory_block *b =
            tool_memory_find_block_locked(&s->tool_mem, start, (size_t)(end - start));
        if (b && b->seen != scan) {
            b->seen = scan;
            for (tool_memory_entry *e = b->entries; ok && e; e = e->block_next) {
                size_t id_len = strlen(e->id);
                size_t raw_len = b->len;
                if (id_len > UINT32_MAX || raw_len > UINT32_MAX) continue;
                uint8_t lens[8];
                le_put32(lens, (uint32_t)id_len);
                le_put32(lens + 4, (uint32_t)raw_len);
                ok = fwrite(lens, 1, sizeof(lens), fp) == sizeof(lens) &&
                     fwrite(e->id, 1, id_len, fp) == id_len &&
                     fwrite(b->raw, 1, raw_len, fp) == raw_len;
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
        uint32_t raw_len = le_get32(lens + 4);
        if (id_len == 0 || id_len > 256 || raw_len == 0 ||
            raw_len > PULSAR_TOOL_MEMORY_MAX_BYTES) return loaded;
        char *id = (char *)server_xmalloc((size_t)id_len + 1);
        char *raw = (char *)server_xmalloc((size_t)raw_len + 1);
        bool ok = fread(id, 1, id_len, fp) == id_len &&
                  fread(raw, 1, raw_len, fp) == raw_len;
        id[id_len] = '\0';
        raw[raw_len] = '\0';
        if (ok && (!wanted || id_list_contains(wanted, id))) {
            s->tool_memory_put_source(id, raw, TOOL_MEMORY_DISK);
            loaded++;
        }
        free(id);
        free(raw);
        if (!ok) return loaded;
    }
    return loaded;
}



/* ===== L264 S4: the disk KV cache is the segment store ====================
 * A conversation persists as a chain of segments, each ending at a grid
 * checkpoint its bank holds, at least a prefill chunk apart (pulsar_kvchain.h,
 * pulsar_segstore.h): each new prefill adds only the segments past
 * the chain's last stored one, so a write costs the new tokens, and the chain's
 * earlier segments stay -- a client that rewrites history further back finds
 * the stretch it still shares.  A restore loads the deepest chain the request's
 * bytes reproduce, ending live at its last checkpoint. */

/// @brief The segment store's log callback: one KVCACHE line.
/// @param ud   unused
/// @param msg  the message
static void kv_seg_log(void *ud, const char *msg) {
    (void)ud;
    server_log(PULSAR_LOG_KVCACHE, "pulsar-server: kv cache %s", msg);
}

/// @brief Open the segment store at `dir` and enable the disk cache.
/// @param kc         the cache (zeroed first)
/// @param dir        the store directory (copied)
/// @param budget_mb  the store's byte budget, MiB
/// @param identity   the model identity segments are keyed to (a different model's segments do not load)
/// @param opt        the checkpoint policy
/// @return false when the store would not open (the cache stays disabled)
bool kv_cache_open(kv_disk_cache *kc, const char *dir, uint64_t budget_mb, uint32_t identity,
                   kv_cache_options opt) {
    memset(kc, 0, sizeof(*kc));
    kc->opt = opt;
    kc->st = pulsar_segstore_open(dir, budget_mb << 20, identity, kv_seg_log, NULL);
    if (!kc->st) return false;
    kc->dir = server_xstrdup(dir);
    kc->enabled = true;
    return true;
}

/// @brief Close the store, free the directory and zero the cache.
/// @param kc  the cache
void kv_cache_close(kv_disk_cache *kc) {
    pulsar_segstore_close(kc->st);
    free(kc->dir);
    memset(kc, 0, sizeof(*kc));
}

/// @brief The tokens' text, concatenated (the chain keys on it).
/// @param engine   the engine
/// @param tokens   the tokens
/// @param out_len  receives the text's length in bytes
/// @return malloc'd text
char *render_tokens_text(pulsar_engine *engine, const pulsar_tokens *tokens, size_t *out_len) {
    return pulsar_kvtext_render_tokens_text(engine, tokens, out_len);
}

/// @brief Whether `prefix` is a byte prefix of `text`.
/// @param text  the text
/// @param text_len  bytes at `text`
/// @param prefix  the prefix
/// @param prefix_len  bytes at `prefix`
/// @return true on a match
static bool byte_prefix_match(const char *text, size_t text_len,
                              const char *prefix, size_t prefix_len) {
    return pulsar_kvtext_byte_prefix_match(text, text_len, prefix, prefix_len);
}

/// @brief Replace `dst` with the first `n` tokens of `src`.
/// @param dst  the destination
/// @param src  the source
/// @param n  tokens to copy
void tokens_copy_prefix(pulsar_tokens *dst, const pulsar_tokens *src, int n) {
    pulsar_kvtext_tokens_copy_prefix(dst, src, n);
}

/// @brief The prompt to sync: `exact_prefix`'s tokens as stored, then `suffix_text` tokenized on its own (never
///        re-tokenized across the boundary).
/// @param engine        the engine (its tokenizer)
/// @param exact_prefix  the restored tokens
/// @param suffix_text   the text past the prefix
/// @param spans         the suffix's text spans
/// @param n_spans       spans in `spans`
/// @param out           receives the prompt
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

/* L290: where a family tail joins the live history.  Every family's tail (tool_result_tail, tool_error_suffix)
 * opens with its template's turn close -- Qwen "<|im_end|>\n", DeepSeek EOS -- because the full render writes
 * that close after the sampled turn (test_qwen_hooks_compose_to_the_full_render).  Whether the KV already holds
 * the sampled stop is the decode lane's: the batched lanes evaluate the token they emit and a speculative round
 * commits an accepted draft stop, while a round whose base token is the stop never evaluates it.  So the tail
 * joins the history WITHOUT a trailing stop row, and the continued bank is the full render's tokens whichever
 * lane sampled the turn.  Joined after a committed stop, a Qwen tool loop's history read
 * "</tool_call><|im_end|><|im_end|>\n" -- a history no render spells -- and the next request that was not a
 * tool-result continuation (a text-only user turn) resumed from the grid checkpoint before the first doubled
 * close (a 6.4k-token re-prefill in the 2026-10-12 production trace).  A stop the tail re-spells (<|im_end|>)
 * leaves the history untouched; any other stop row (<|endoftext|>) is rewound past by the sync. */
void join_live_history_and_tail(const pulsar_tokens *live, bool live_ends_at_stop, const pulsar_tokens *tail,
                                pulsar_tokens *out) {
    tokens_copy_prefix(out, live, live->len - (live_ends_at_stop && live->len > 0 ? 1 : 0));
    for (int i = 0; i < tail->len; i++) pulsar_tokens_push(out, tail->v[i]);
}

int build_live_continuation_prompt(pulsar_engine *engine, const pulsar_tokens *live, const char *tail_text,
                                   const pulsar_text_span *spans, uint32_t n_spans, pulsar_tokens *out) {
    pulsar_tokens tail = {};
    /* the tail opens with chat markers: the rendered-chat tokenizer, its client ranges plain text (L223) */
    pulsar_tokenize_rendered_chat_spans(engine, tail_text ? tail_text : "", spans, n_spans, &tail);
    join_live_history_and_tail(live, live->len > 0 && pulsar_token_is_stop(engine, live->v[live->len - 1]), &tail,
                               out);
    pulsar_tokens_free(&tail);
    int common = 0;
    while (common < live->len && common < out->len && live->v[common] == out->v[common]) common++;
    return common;
}

/* The stable rendered chat prefix is everything before the user message that
 * asks this specific task.  Some clients put stable user-role scaffolding
 * first, so the anchor is the last user marker before the first assistant. */
/// @param kc      the cache (its min_tokens)
/// @param prompt  the prompt tokens
/// @param m       the template's turn markers (none: -1)
/// @return the position of the last user marker before the first assistant marker, -1 when there is none or it
///         sits below min_tokens
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
/// @param kc      the cache (its min_tokens, sys_prefix_margin, sys_prefix_align)
/// @param anchor  the chat anchor (kv_cache_chat_anchor_pos), tokens
/// @return the cut position in tokens, 0 for none
int kv_cache_sys_prefix_cut(const kv_disk_cache *kc, int anchor) {
    if (anchor < kc->opt.min_tokens) return 0;
    int cut = anchor - kc->opt.sys_prefix_margin;
    if (kc->opt.sys_prefix_align > 0) cut -= cut % kc->opt.sys_prefix_align;
    return cut >= kc->opt.min_tokens ? cut : 0;
}

namespace {
/** pulsar_kvchain_trailer::size: the tool-map trailer a segment of `text` carries.
 *  @param ud the server
 *  @param text the segment's prefix text
 *  @param bytes the trailer's size in bytes (0 = none)
 *  @return false when it cannot be measured */
bool seg_trailer_size(void *ud, const char *text, uint64_t *bytes) {
    return ((server *)ud)->kv_tool_map_serialized_size(text, bytes);
}
/** pulsar_kvchain_trailer::write: write that trailer (server::kv_tool_map_write).
 *  @param fp the segment file, at the trailer's offset
 *  @param ud the server
 *  @param text the segment's prefix text
 *  @param err the failure
 *  @param errlen bytes at `err`
 *  @return 0 on success, 1 + err on failure */
int seg_trailer_write(FILE *fp, void *ud, const char *text, char *err, size_t errlen) {
    uint64_t written = 0;
    if (((server *)ud)->kv_tool_map_write(fp, text, &written)) return 0;
    snprintf(err, errlen, "tool map write failed");
    return 1;
}
}  // namespace

int server::kv_cache_persist(session_slot *sl, const char *reason, bool close, int *chain_end) {
    auto *s = this;
    (void)sl;   /* the caller installed the slot's bank */
    if (chain_end) *chain_end = 0;
    if (!s->kv.enabled) return 0;
    /* A "continued" store fires from the prefill's progress callback; on a TP
     * group that is inside the mirrored sync, where the workers read only
     * chunk verdicts.  The prompt-end store after it persists the same
     * checkpoints. */
    if (pulsar_session_in_mirrored_sync(s->sess)) return 0;
    /* 2026-10-29 (a production hang under memory pressure: a "continued" persist at ~188k under the kernel's min watermark): a persist runs
     * only when MemAvailable covers what it can take -- its dirty-page window and the history's text renders (the
     * device reads stream through the session's pinned buffer, which allocates nothing) -- over the floor.  Else
     * it is SKIPPED, counted and said: the history stays live in its bank, only its disk copy is not written now
     * (a later persist of the same bank writes every segment the store lacks).  A documented outcome, not a
     * fallback: no other write path runs. */
    {
        const pulsar_tokens *toks = pulsar_session_tokens(s->sess);
        const uint64_t need = server_persist_host_bytes(toks ? toks->len : 0);
        const uint64_t avail = server_mem_available_bytes();
        if (!server_headroom_admits(KvBytes{avail}, KvBytes{need}, server_mem_floor())) {
            s->count_metric(&s->m_persist_skipped);
            unsigned skipped = 0;
            if (warn_limiter_due(&s->persist_skip_warn, server_now_sec(), PULSAR_SERVER_MEM_FLOOR_WARN_SEC, &skipped)) {
                const double gib = 1024.0 * 1024.0 * 1024.0;
                server_log(PULSAR_LOG_WARNING,
                           "pulsar-server: kv cache persist SKIPPED reason=%s: MemAvailable %.2f GiB does not cover "
                           "the %.2f GiB it may take over the %.2f GiB floor%s (%u more skipped since the last line)",
                           reason, (double)avail / gib, (double)need / gib, (double)server_mem_floor().raw() / gib,
                           avail ? "" : " (MemAvailable unreadable)", skipped);
            }
            return 0;
        }
    }
    pulsar_stall_mark("kv cache persist");
    const double t0 = server_now_sec();
    const pulsar_kvchain_trailer tool_maps = { s, seg_trailer_size, seg_trailer_write };
    pulsar_kvchain_persist_result r;
    pulsar_kvchain_persist(s->kv.st, s->engine, s->sess, s->kv.opt.min_tokens, close, &tool_maps, &r);
    if (r.err[0])
        server_log(PULSAR_LOG_WARNING, "pulsar-server: kv cache chain not fully stored (reason=%s): %s", reason, r.err);
    if (r.written)
        server_log(PULSAR_LOG_KVCACHE,
                   "pulsar-server: kv cache persisted reason=%s chain to %d (%d new segments, %.1f MiB) %.1f ms",
                   reason, r.end, r.written, (double)r.bytes / (1024.0 * 1024.0), (server_now_sec() - t0) * 1000.0);
    if (chain_end) *chain_end = r.end;
    return r.written;
}

/* ===== L310: restore-vs-recompute ========================================== */

const char *const kv_restore_verdict_names[KV_RESTORE_VERDICTS] = { "load", "recompute", "unmeasured", "unpriced" };

kv_restore_verdict kv_restore_decide(uint32_t gained, bool measured, double transfer_ms, bool priced,
                                     double prefill_ms) {
    if (gained == 0u) return KV_RESTORE_RECOMPUTE;
    if (!measured) return KV_RESTORE_UNMEASURED;
    if (!priced) return KV_RESTORE_UNPRICED;
    return transfer_ms < 0.5 * prefill_ms ? KV_RESTORE_LOAD : KV_RESTORE_RECOMPUTE;
}

/// kv_read_cost's per-load decay: each load weighs 1/8.
#define KV_READ_DECAY (1.0 - 1.0 / 8.0)
/// Bytes per GiB, the unit of kv_read_cost's sums.
#define KV_GIB (1024.0 * 1024.0 * 1024.0)

/// @brief Fold one chain load into the read cost and re-solve both rates (ridge least squares, no intercept).
/// @param c       the cost (kv_restore_meter::read; caller holds server::mu)
/// @param bytes   bytes the load read (0: ignored)
/// @param cached  of which in the page cache (clamped to `bytes`)
/// @param ms      the load's wall time, ms (<= 0: ignored)
void kv_read_cost_observe(kv_read_cost *c, uint64_t bytes, uint64_t cached, double ms) {
    if (!(ms > 0.0) || bytes == 0u) return;
    if (cached > bytes) cached = bytes;
    const double x = (double)(bytes - cached) / KV_GIB, z = (double)cached / KV_GIB;
    c->scc = c->scc * KV_READ_DECAY + x * x;
    c->scw = c->scw * KV_READ_DECAY + x * z;
    c->sww = c->sww * KV_READ_DECAY + z * z;
    c->scy = c->scy * KV_READ_DECAY + x * ms;
    c->swy = c->swy * KV_READ_DECAY + z * ms;
    c->loads++;
    /* the normal equations with the ridge lambda (u - w)^2: [scc + l, scw - l; scw - l, sww + l] [u; w] = [scy; swy];
     * positive definite for any loads (the ridge alone spans u - w, a load spans the rest) */
    const double l = KV_READ_RIDGE, a = c->scc + l, b = c->scw - l, d = c->sww + l;
    const double det = a * d - b * b;
    if (!(det > 0.0)) return;
    const double u = (c->scy * d - b * c->swy) / det, w = (a * c->swy - b * c->scy) / det;
    c->cold_ms_per_gib = u > 0.0 ? u : 0.0;
    c->cached_ms_per_gib = w > 0.0 ? w : 0.0;
}

bool kv_read_cost_ms(const kv_read_cost *c, uint64_t bytes, uint64_t cached, double *ms) {
    *ms = 0.0;
    if (c->loads == 0u) return false;
    if (cached > bytes) cached = bytes;
    *ms = (double)(bytes - cached) / KV_GIB * c->cold_ms_per_gib + (double)cached / KV_GIB * c->cached_ms_per_gib;
    return true;
}

int server::kv_cache_try_load_text(session_slot *sl, const char *prompt_text,
                                  const pulsar_text_span *prompt_spans,
                                  uint32_t prompt_n_spans,
                                  const uint64_t *image_hashes, int n_images, int base,
                                  pulsar_tokens *effective_prompt,
                                  char **loaded_key_out,
                                  bool responses_protocol) {
    (void)sl;   /* the caller installed the slot's bank */
    if (loaded_key_out) *loaded_key_out = NULL;
    kv_chain_hit hit;
    const int G = kv_cache_find(prompt_text, image_hashes, n_images, &hit);
    if (G <= base) return 0;   /* nothing past what the bank serves: no read is priced */
    if (kv_cache_price(&hit, base) == KV_RESTORE_RECOMPUTE) return 0;
    return kv_cache_load(&hit, prompt_text, prompt_spans, prompt_n_spans, base, effective_prompt, loaded_key_out,
                         responses_protocol);
}

/* ===== L318: one resume decision, the deepest valid state ================== */

const char *const resume_pick_names[RESUME_PICKS] = { "own", "adopt", "disk" };

int server_resume_memory_depth(int own, int adopt) {
    return adopt > own + PULSAR_SERVER_ADOPT_MIN_GAIN ? adopt : own;
}

resume_pick server_resume_pick(int own, int adopt, int disk, kv_restore_verdict disk_verdict) {
    const int mem = server_resume_memory_depth(own, adopt);
    if (disk > mem && disk_verdict != KV_RESTORE_RECOMPUTE) return RESUME_DISK;
    return mem > own ? RESUME_ADOPT : RESUME_OWN;
}

int server::kv_cache_find(const char *prompt_text, const uint64_t *image_hashes, int n_images, kv_chain_hit *hit) {
    auto *s = this;
    hit->n = 0;
    hit->bytes = 0;
    hit->cached = 0;
    if (!s->kv.enabled || !prompt_text) return 0;
    char err[384];
    hit->n = pulsar_kvchain_find(s->kv.st, prompt_text, strlen(prompt_text), image_hashes, n_images,
                                 s->kv.opt.min_tokens, hit->chain, PULSAR_KVCHAIN_MAX, &hit->bytes, err, sizeof(err));
    if (err[0]) server_log(hit->n ? PULSAR_LOG_KVCACHE : PULSAR_LOG_WARNING, "pulsar-server: kv cache %s", err);
    /* L327: how much of that read the page cache holds -- the read is priced, and measured, by the two apart */
    if (hit->n > 0) hit->cached = pulsar_kvchain_cached_bytes(s->kv.st, hit->chain, hit->n);
    return hit->n > 0 ? (int)hit->chain[hit->n - 1].G : 0;
}

int server::kv_cache_find_req(const request *req, kv_chain_hit *hit) {
    /* L281: the chain loads past an image block only when this request brings the same image there */
    enum { HASHES_MAX = 64 };
    uint64_t hashes[HASHES_MAX];
    const int n_images = req->n_images < HASHES_MAX ? req->n_images : HASHES_MAX;
    for (int i = 0; i < n_images; i++) hashes[i] = pulsar_image_hash(&req->images[i]);
    return kv_cache_find(req->prompt_text, hashes, n_images, hit);
}

kv_restore_verdict server::kv_cache_price(const kv_chain_hit *hit, int base) {
    auto *s = this;
    /* L310: price the read against the prefill of what the chain adds past `base` */
    const uint32_t chain_G = hit->chain[hit->n - 1].G;
    const uint64_t bytes = hit->bytes;
    const int n = hit->n;
    if (base < 0) base = 0;
    const uint32_t gained = chain_G > (uint32_t)base ? chain_G - (uint32_t)base : 0u;
    double prefill_ms = 0.0, transfer_ms = 0.0;
    const int cap = s->sess ? pulsar_session_prefill_cap(s->sess).raw() : 0;
    const bool priced = pulsar_engine_prefill_ms(s->engine, gained, (uint32_t)base, cap > 0 ? (uint32_t)cap : 0u,
                                                 &prefill_ms);
    pthread_mutex_lock(&s->mu);
    const bool measured = kv_read_cost_ms(&s->kv.meter.read, bytes, hit->cached, &transfer_ms);
    pthread_mutex_unlock(&s->mu);
    const kv_restore_verdict v = kv_restore_decide(gained, measured, transfer_ms, priced, prefill_ms);
    pthread_mutex_lock(&s->mu);
    s->kv.meter.verdicts[v]++;
    s->kv.meter.last_transfer_ms = transfer_ms;
    s->kv.meter.last_prefill_ms = prefill_ms;
    if (v == KV_RESTORE_RECOMPUTE) s->kv.meter.recomputed_tokens += gained;
    pthread_mutex_unlock(&s->mu);
    server_log(PULSAR_LOG_KVCACHE,
               "pulsar-server: kv cache restore-vs-recompute: chain to %u (%d segments, %.1f MiB, %.1f MiB cached) adds "
               "%u tokens past %d: read %.1f ms%s vs prefill %.1f ms%s -> %s",
               chain_G, n, (double)bytes / (1024.0 * 1024.0), (double)hit->cached / (1024.0 * 1024.0), gained, base,
               transfer_ms, measured ? "" : " (unmeasured)", prefill_ms, priced ? "" : " (no fit)",
               kv_restore_verdict_names[v]);
    return v;
}

/* ===== L330: a chain load in slices ======================================
 *
 * A restore used to read its whole chain inside one worker turn: production chains reach 34 GiB / 24 s (the archived
 * 10-22..10-24 log, a 192k Claude Code conversation), and every other stream stood still for the whole read.  Now a
 * request's restore reads a SLICE of segments per worker turn, its job in GEN_RESTORE between decode rounds, the way a
 * long prompt prefills a chunk per turn: segments straight from the store into the installed bank (no whole-chain host
 * buffer -- GB10's memory is the GPU's, ~12 GiB available during those restores), the same segments in the same order
 * as the one-shot load (pulsar_kvchain_load_slice; the one-shot load is ONE slice of the whole chain, here and in the
 * library), so the bank ends byte-for-byte where it would have. */

/* How many segments the next slice reads, pure: from `next`, every segment while their predicted reads (pred_ms) sum
 * within `target_ms` -- one prefill chunk's priced time at the depth the restore stands at, so a restoring job holds
 * the worker no longer than a prefilling one does -- and at least one.  Exactly one while either is unknown: the read
 * meter has measured no load (pred < 0) or the prefill fit has no price (target <= 0). */
/// @param pred_ms    each segment's predicted read, ms (< 0: unmeasured)
/// @param next       the first segment not yet read
/// @param n          segments in the chain
/// @param target_ms  the slice's time budget, ms (<= 0: unpriced)
/// @return segments to read (0 when none remain)
static int server_restore_slice_segments(const double *pred_ms, int next, int n, double target_ms) {
    if (next >= n) return 0;
    if (!(target_ms > 0.0) || !(pred_ms[next] >= 0.0)) return 1;
    double t = pred_ms[next];
    int k = 1;
    while (next + k < n && pred_ms[next + k] >= 0.0 && t + pred_ms[next + k] <= target_ms) t += pred_ms[next + k++];
    return k;
}

void server::kv_restore_begin(kv_restore *r, const kv_chain_hit *hit, int base, bool sliced) {
    auto *s = this;
    r->hit = hit;
    r->next = 0;
    r->base = base < 0 ? 0 : base;
    r->read_ms = 0.0;
    r->slice_max_ms = 0.0;
    r->slices = 0;
    if (!sliced) return;
    /* the read each segment costs, disk and page cache apart (L327), as the meter prices the whole chain */
    for (int i = 0; i < hit->n; i++) {
        const pulsar_segstore_seg *g = &hit->chain[i];
        const uint64_t bytes = pulsar_kvchain_segment_load_bytes(hit->chain, hit->n, i);
        const uint64_t cached = pulsar_segstore_cached_bytes(s->kv.st, g->key, bytes - g->trailer_bytes, true);
        double ms = 0.0;
        pthread_mutex_lock(&s->mu);
        const bool measured = kv_read_cost_ms(&s->kv.meter.read, bytes, cached, &ms);
        pthread_mutex_unlock(&s->mu);
        r->pred_ms[i] = measured ? ms : -1.0;
    }
}

int server::kv_restore_slice(kv_restore *r, int k) {
    auto *s = this;
    const double t0 = server_now_sec();
    char err[384];
    const int rc = pulsar_kvchain_load_slice(s->kv.st, s->sess, r->hit->chain, r->hit->n, &r->next, k, err, sizeof err);
    const double ms = (server_now_sec() - t0) * 1000.0;
    r->read_ms += ms;
    r->slices++;
    if (ms > r->slice_max_ms) r->slice_max_ms = ms;
    pthread_mutex_lock(&s->mu);
    s->m_restore_slices++;
    if (ms > s->m_restore_stall_max_ms) s->m_restore_stall_max_ms = ms;
    pthread_mutex_unlock(&s->mu);
    if (rc != 0) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: kv cache %s", err);
        return -1;
    }
    return r->next >= r->hit->n ? 1 : 0;
}

int server_restore_slice_parse(const char *text) {
    if (!text || !*text) return 0;
    long v = 0;
    for (const char *p = text; *p; p++) {
        if (*p < '0' || *p > '9') return -1;
        v = v * 10 + (*p - '0');
        if (v > 600000) return -1;
    }
    return v > 0 ? (int)v : -1;
}

int server::kv_restore_next(kv_restore *r) {
    auto *s = this;
    /* the slice: one prefill chunk's priced time at the depth the restore stands at (the test hook's, when set) */
    double target_ms = (double)s->restore_slice_ms;
    if (s->restore_slice_ms <= 0) {
        const int cap = pulsar_session_prefill_cap(s->sess).raw();
        const uint32_t depth = r->hit->chain[r->next].G_prev;
        if (cap <= 0 || !pulsar_engine_prefill_ms(s->engine, (uint32_t)cap, depth, (uint32_t)cap, &target_ms))
            target_ms = 0.0;
    }
    return kv_restore_slice(r, server_restore_slice_segments(r->pred_ms, r->next, r->hit->n, target_ms));
}

int server::kv_restore_finish(kv_restore *r, const char *text, const pulsar_text_span *spans, uint32_t n_spans,
                              pulsar_tokens *effective_prompt, char **loaded_key_out, bool responses_protocol) {
    auto *s = this;
    if (loaded_key_out) *loaded_key_out = NULL;
    const pulsar_segstore_seg *chain = r->hit->chain;
    const int n = r->hit->n;
    char err[384];
    const int G = pulsar_kvchain_load_check(s->kv.st, s->engine, s->sess, text, chain, n, err, sizeof err);
    if (err[0]) server_log(PULSAR_LOG_WARNING, "pulsar-server: kv cache %s", err);
    if (G == 0) return 0;
    const size_t text_bytes = strlen(text);
    pthread_mutex_lock(&s->mu);
    kv_read_cost_observe(&s->kv.meter.read, r->hit->bytes, r->hit->cached, r->read_ms);
    s->kv.meter.restored_tokens += G > r->base ? (uint64_t)(G - r->base) : 0u;
    pthread_mutex_unlock(&s->mu);
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
        pulsar_text_span *tail = pulsar_text_spans_slice(spans, n_spans, (uint32_t)text_end,
                                                         (uint32_t)(text_bytes - text_end), &tail_n);
        build_prompt_from_exact_prefix_and_text_suffix(s->engine, loaded, text + text_end, tail, tail_n,
                                                       effective_prompt);
        free(tail);
    }
    if (loaded_key_out) *loaded_key_out = server_xstrdup(chain[n - 1].key);
    server_log(PULSAR_LOG_KVCACHE,
               "pulsar-server: kv cache hit%s chain of %d to %u (text %llu of %zu bytes) load=%.1f ms in %d slice%s "
               "(longest %.1f ms)",
               responses_protocol ? " [responses]" : "", n, chain[n - 1].G, (unsigned long long)text_end, text_bytes,
               r->read_ms, r->slices, r->slices == 1 ? "" : "s", r->slice_max_ms);
    return G;
}

int server::kv_cache_load(const kv_chain_hit *hit, const char *prompt_text, const pulsar_text_span *prompt_spans,
                          uint32_t prompt_n_spans, int base, pulsar_tokens *effective_prompt, char **loaded_key_out,
                          bool responses_protocol) {
    if (loaded_key_out) *loaded_key_out = NULL;
    /* the one-shot load: the whole chain as one slice (canonicalize_tool_checkpoint's rebuild, the one caller, runs to
     * completion inside its quantum like the sync it serves -- and reads only what L310 prices under half the
     * prefill it saves, so the read never stalls longer than that sync would) */
    kv_restore r;
    kv_restore_begin(&r, hit, base, false);
    if (kv_restore_slice(&r, hit->n) != 1) return 0;
    return kv_restore_finish(&r, prompt_text, prompt_spans, prompt_n_spans, effective_prompt, loaded_key_out,
                             responses_protocol);
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

    const int common = build_live_continuation_prompt(
        s->engine, live_tokens, req->responses_live_suffix_text,
        req->responses_live_suffix_spans, req->responses_live_suffix_n_spans,
        effective_prompt);
    if (matched_ids) *matched_ids = req->responses_live_call_ids.len;
    return common;
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

    const int common = build_live_continuation_prompt(
        s->engine, live_tokens, req->anthropic_live_suffix_text,
        req->anthropic_live_suffix_spans, req->anthropic_live_suffix_n_spans,
        effective_prompt);
    if (matched_ids) *matched_ids = req->anthropic_live_call_ids.len;
    return common;
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




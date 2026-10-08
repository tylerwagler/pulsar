/* Job lifecycle around ONE request (split move-only from generate.cpp):
 * the resumable gen_state machine (prefill -> decode -> finish), its
 * prefill-progress/keepalive plumbing, tool-checkpoint canonicalization
 * and thinking-checkpoint remembering, protocol stream emission glue,
 * and the generate_job_* driver stepped by the scheduler
 * (server_sched.cpp). gen_state/gen_phase live in pulsar_server_internal.h
 * because the scheduler TU steps jobs by phase and drives the batched
 * lanes through the batch_* fields. */
#include "pulsar_server_internal.h"



/* A completed tool block inside unclosed reasoning can be recovered without
 * predicting what the model will emit after an injected close marker
 * (upstream ds4 51a1c14; replaces the forced-</think> injection recovery,
 * which sometimes made the model read the call as already issued and end the
 * turn without it). Keep a short overlap until the opening appears, then wait
 * for its matching end — a lone "<" or partial marker keeps decoding
 * untouched. */
bool complete_tool_call_inside_thinking(const char *text, size_t len,
                                        size_t *scan_from) {
    if (!text || !scan_from) return false;
    if (*scan_from > len) *scan_from = len;
    const char *start = find_any_tool_start(text + *scan_from);
    if (!start) {
        const size_t hold = 80; /* > longest stanza opening */
        *scan_from = len > hold ? len - hold : 0;
        return false;
    }
    *scan_from = (size_t)(start - text);
    return find_any_tool_end(start) != NULL;
}



bool server::append_rendered_suffix_to_live_session(session_slot *sl,
                                                   const char *suffix,
                                                   const pulsar_text_span *spans,
                                                   uint32_t n_spans,
                                                   int *tokens_appended,
                                                   char *err, size_t errlen) {
    auto *s = this;
    (void)sl; /* slot is a pure bank descriptor; the session is s->sess */
    if (tokens_appended) *tokens_appended = 0;
    if (!s || !suffix || !suffix[0]) return true;
    const pulsar_tokens *live = pulsar_session_tokens(s->sess);
    if (!live) {
        if (err && errlen) snprintf(err, errlen, "live session is unavailable");
        return false;
    }

    pulsar_tokens target = {0};
    build_prompt_from_exact_prefix_and_text_suffix(s->engine, live, suffix, spans, n_spans, &target);
    const int before = pulsar_session_pos(s->sess);
    bool ok = pulsar_session_sync(s->sess, &target, err, errlen) == 0;
    if (ok && tokens_appended) {
        int delta = pulsar_session_pos(s->sess) - before;
        *tokens_appended = delta > 0 ? delta : 0;
    }
    pulsar_tokens_free(&target);
    return ok;
}



bool server::continue_after_invalid_dsml(session_slot *sl,
                                        const request *r,
                                        const thinking_state *thinking,
                                        const char *detail,
                                        int *tokens_appended,
                                        char *err, size_t errlen) {
    auto *s = this;
    chat_text_span *spans = NULL;
    uint32_t n_spans = 0;
    if (!r->family->tool_error_suffix) {
        snprintf(err, errlen, "%s has no tool-error suffix: no model-visible retry", r->family->name);
        return false;
    }
    char *suffix = r->family->tool_error_suffix(r, thinking, detail, &spans, &n_spans);
    if (r->force_tool_call && suffix) {
        /* L284 P4: a forced call's retry is forced too -- the tail's generation prompt re-opens the call the
         * way the prompt did (the family's forced_call_prefill), so the next attempt's seed is what the KV holds */
        size_t keep = strlen(suffix);
        buf add = {0};
        r->family->forced_call_prefill(r, suffix, &keep, &add);
        buf re = {0};
        buf_append(&re, suffix, keep);
        if (add.len) buf_append(&re, add.ptr, add.len);
        buf_free(&add);
        uint32_t k = 0;
        for (uint32_t i = 0; i < n_spans; i++) {
            if (spans[i].lo >= keep) continue;
            spans[k] = spans[i];
            if (spans[k].hi > keep) spans[k].hi = (uint32_t)keep;
            k++;
        }
        n_spans = k;
        free(suffix);
        suffix = buf_take(&re);
    }
    bool ok = s->append_rendered_suffix_to_live_session(sl, suffix,
                                                     spans, n_spans,
                                                     tokens_appended,
                                                     err, errlen);
    free(suffix);
    free(spans);
    return ok;
}



static void log_tool_calls_summary(const char *ctx, const tool_calls *calls,
                                   bool responses_protocol) {
    if (!calls || calls->len == 0) return;
    buf names = {0};
    buf ids = {0};
    for (int i = 0; i < calls->len; i++) {
        if (i) buf_putc(&names, ',');
        if (i) buf_putc(&ids, ',');
        buf_puts(&names, calls->v[i].name ? calls->v[i].name : "?");
        buf_puts(&ids, calls->v[i].id ? calls->v[i].id : "?");
    }
    char flags[32];
    log_flags(flags, sizeof(flags), responses_protocol, false, false, false, false);
    server_log(PULSAR_LOG_TOOL,
               "pulsar-server: tool calls ctx=%s%s%s n=%d raw_dsml=%d ids=[%s] names=[%s]",
               ctx,
               flags[0] ? " " : "",
               flags,
               calls->len,
               calls->raw_dsml && calls->raw_dsml[0] ? 1 : 0,
               ids.ptr ? ids.ptr : "",
               names.ptr ? names.ptr : "");
    buf_free(&ids);
    buf_free(&names);
}



static void server_progress_cb(void *ud, const char *event, int current, int total) {
    server_prefill_progress *p = (server_prefill_progress *)ud;
    if (!p || !event) return;
    const bool is_chunk = strcmp(event, "prefill_chunk") == 0;
    const bool is_display = strcmp(event, "prefill_display") == 0;
    if (!is_chunk && !is_display) return;

    double now = server_now_sec();
    /* Keep the HTTP/SSE connection alive while prefill runs.  We write the SSE
     * response headers the first time the callback fires and then emit a
     * comment line (`:` prefix, ignored by SSE clients) every few seconds.
     * Best-effort: if the client has already gone away, the writes fail
     * silently and the outer code will discover the closed socket the next
     * time it tries to stream a real event. */
    if (p->stream && p->fd >= 0 && !p->stream_failed) {
        if (!p->headers_sent) {
            p->headers_sent = true;
            if (sse_headers(p->fd)) {
                p->last_keepalive = now;
            } else {
                p->stream_failed = true;
            }
        } else if (now - p->last_keepalive >= 5.0) {
            static const char ka[] = ": prefill\n\n";
            if (send_all(p->fd, ka, sizeof(ka) - 1)) {
                p->last_keepalive = now;
            } else {
                p->stream_failed = true;
            }
        }
    }
    if (is_display) return;
    double elapsed = now - p->t0;
    if (p->seen && current == p->last_current) {
        if (p->srv && p->slot && current > p->cached_tokens) {
            p->srv->kv_cache_persist(p->slot, "continued");
        }
        return;
    }
    int display_start = p->cached_tokens;
    if (display_start < 0 || display_start > p->prompt_tokens) display_start = 0;
    int display_total = p->prompt_tokens - display_start;
    if (display_total <= 0) {
        display_start = 0;
        display_total = p->prompt_tokens > total ? p->prompt_tokens : total;
    }
    int display_current = current - display_start;
    if (display_current < 0) display_current = 0;
    if (display_current > display_total) display_current = display_total;
    double pct = display_total > 0 ? 100.0 * (double)display_current / (double)display_total : 100.0;
    double avg_tps = elapsed > 0.0 ? (double)display_current / elapsed : 0.0;
    /* First callback fires AFTER the first chunk completes, so the chunk has
     * a real rate: its tokens over the elapsed-since-start. The old !seen arm
     * zeroed the interval instead, printing "chunk=0.00 t/s" on the first
     * chunk of every prefill -- which read as a stall on any monitor
     * (pulsar-tui showed alternating dead samples through the 08-25 perf
     * run). Subsequent chunks are unchanged: delta tokens over delta time. */
    int interval_tokens = p->seen ? current - p->last_current : display_current;
    if (interval_tokens < 0) interval_tokens = 0;
    /* L114: chunk-granular prefill accounting via the per-slot watermark
     * (see session_slot.prefill_counted) — NOT interval_tokens, because the
     * mixed lane's fused sub-chunks also advance the watermark and an
     * interval here would recount them. Computed rows only. */
    if (p->srv && p->slot && current > p->slot->prefill_counted) {
        p->srv->w_prefill_chunk_tokens +=
            (uint64_t)(current - p->slot->prefill_counted);
        p->slot->prefill_counted = current;
    }
    double interval_s = p->seen ? now - p->last_t : elapsed;
    double chunk_tps = interval_s > 0.0 ? (double)interval_tokens / interval_s : 0.0;
    p->last_current = current;
    p->last_t = now;
    p->seen = true;
    char flags[64];
    log_flags(flags, sizeof(flags), p->responses_protocol,
              p->has_tools, false, false, false);
    const char *phase = p->phase ? p->phase : "prefill";
    server_log(PULSAR_LOG_PREFILL,
               "pulsar-server: %s ctx=%s%s%s %s chunk %d/%d (%.1f%%) chunk=%.2f t/s avg=%.2f t/s %.3fs",
               p->kind == REQ_CHAT ? "chat" : "completion",
               p->ctx,
               flags[0] ? " " : "",
               flags,
               phase,
               display_current,
               display_total,
               pct,
               chunk_tps,
               avg_tps,
               elapsed);
    if (p->srv && p->slot && current > p->cached_tokens) {
        p->srv->kv_cache_persist(p->slot, "continued");
    }
}



void server::send_prefill_failure_response(const job *j,
                                          const server_prefill_progress *progress,
                                          const char *ctx, const char *flags,
                                          const char *err) {
    const char *kind = j->req.kind == REQ_CHAT ? "chat" : "completion";
    if (j->req.stream && progress && progress->headers_sent) {
        if (progress->stream_failed) {
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: %s ctx=%s%s%s prefill failed after stream closed: %s",
                       kind, ctx, flags && flags[0] ? " " : "",
                       flags && flags[0] ? flags : "", err);
            return;
        }
        if (!sse_error_event(j->fd, &j->req, err)) {
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: %s ctx=%s%s%s prefill SSE error failed: %s",
                       kind, ctx, flags && flags[0] ? " " : "",
                       flags && flags[0] ? flags : "", err);
        }
        return;
    }
    /* An image request that reaches the worker has already had its data: URL,
     * tower and block count validated by the parser, so a prefill refusal here
     * is about the client's image context (an image the tower cannot accept, a
     * span that will not fit a chunk) -- a 400, not a server fault. */
    http_error(j->fd, j->req.n_images > 0 ? 400 : 500, err);
}



/* After a successful tool-call finish, make the live checkpoint match what the
 * next request will render.  Usually that is just the exact DSML remembered by
 * tool id.  If a client sends a tool call without an id we know, the fallback
 * renderer still builds valid DSML from JSON, and this function either rewrites
 * the short suffix in place or reloads an older disk checkpoint before replay. */
void server::canonicalize_tool_checkpoint(session_slot *sl,
                                         const job *j, const char *ctx,
                                         uint64_t trace_id, const char *content,
                                         const char *reasoning, const tool_calls *calls) {
    auto *s = this;
    if (!calls || calls->len == 0 || !j->req.prompt_text) return;

    chat_text_span *suffix_spans = NULL;
    uint32_t suffix_n_spans = 0;
    char *suffix_text = build_tool_checkpoint_suffix_spans(&j->req, content, reasoning, calls,
                                                          &suffix_spans, &suffix_n_spans);

    /* L223: BOTH halves carry client-replayed bytes -- the request prompt and
     * the sampled assistant turn -- so tokenising either with the plain matcher
     * would put a control token back into the LIVE session for a spelling a
     * client wrote.  The suffix's DSML framing is the server's own text and
     * stays control text (it is not inside any range). */
    buf rendered = {0};
    buf_puts_spanned(&rendered, j->req.prompt_text, j->req.prompt_spans,
                     j->req.prompt_n_spans);
    buf_puts_spanned(&rendered, suffix_text, suffix_spans, suffix_n_spans);
    free(suffix_spans);

    pulsar_tokens canonical = {0};
    pulsar_tokenize_rendered_chat_spans(s->engine, rendered.ptr ? rendered.ptr : "",
                                        rendered.spans, rendered.n_spans, &canonical);
    const int live_len = pulsar_session_pos(s->sess);
    const int common = pulsar_session_common_prefix(s->sess, &canonical);
    if (common == live_len && canonical.len == live_len) goto done;

    size_t live_text_len;
    live_text_len = 0;
    char *live_text;
    live_text = render_tokens_text(s->engine, pulsar_session_tokens(s->sess), &live_text_len);
    if (live_text_len == rendered.len &&
        (live_text_len == 0 || memcmp(live_text, rendered.ptr, live_text_len) == 0))
    {
        /* The graph already represents the bytes the next request will render.
         * Token-level canonicalization would only replace a valid sampled
         * history with a different BPE spelling of the same transcript. */
        free(live_text);
        goto done;
    }
    free(live_text);

    if (common < j->req.prompt.len) {
        s->trace_event(trace_id,
                    "tool checkpoint canonicalization skipped: common=%d prompt=%d live=%d canonical=%d",
                    common, j->req.prompt.len, live_len, canonical.len);
        goto done;
    }

    /* L284: the canonical rewrite is a SYNC of the canonical prompt -- every family's one path.  The sync keeps
     * the live tokens up to the deepest byte the canonical spelling shares (the token-seam stitch) and resumes
     * from the grid checkpoint the resume rule picks, so the generated tail is evaluated again in its canonical
     * spelling.  Before, a rewrite behind the live end needed REWIND (Qwen refused it and logged
     * "canonicalization failed" every thinking-off tool turn) and DeepSeek rebuilt it from a disk checkpoint or
     * from 0 even with a grid checkpoint below the cut.  A sync that would start from 0 tries a disk checkpoint
     * first, as a request's does. */
    {
        const int resume = pulsar_session_bank_resume_at(s->sess, (uint32_t)sl->bank, &canonical);
        char *path = NULL;
        pulsar_tokens effective = {0};
        const int loaded = resume > 0 ? 0
            : s->kv_cache_try_load_text(sl, rendered.ptr ? rendered.ptr : "", rendered.spans, rendered.n_spans,
                                        NULL, 0, &effective, &path, false);
        const pulsar_tokens *sync_prompt = loaded > 0 ? &effective : &canonical;
        const int cached = loaded > 0 ? loaded : resume;
        const char *source = loaded > 0 ? "disk" : cached > 0 ? "memory" : "full";
        char sync_ctx[48];
        request_ctx_span(sync_ctx, sizeof(sync_ctx), cached, sync_prompt->len);
        server_log(PULSAR_LOG_KVCACHE,
                   "pulsar-server: tool checkpoint canonicalization syncs %d tokens ctx=%s request_ctx=%s common=%d "
                   "live=%d target=%d cached=%d source=%s%s%s",
                   sync_prompt->len - cached, sync_ctx, ctx, common, live_len, canonical.len, cached, source,
                   path ? " file=" : "", path ? path : "");
        server_prefill_progress progress = {
            .srv = s,
            .slot = sl,
            .kind = j->req.kind,
            .prompt_tokens = sync_prompt->len,
            .cached_tokens = cached,
            .phase = "tool checkpoint canonicalization",
            .has_tools = j->req.has_tools,
            .t0 = server_now_sec(),
            .fd = j->fd,
            .stream = j->req.stream,
            /* this runs after the response stream is in flight, so the SSE headers were sent long ago:
             * the progress callback emits keepalive comments only */
            .headers_sent = true,
        };
        snprintf(progress.ctx, sizeof(progress.ctx), "%s", sync_ctx);
        pulsar_session_set_progress(s->sess, server_progress_cb, &progress);
        pulsar_session_set_display_progress(s->sess, server_progress_cb, &progress);
        char sync_err[160] = {0};
        const int rc = pulsar_session_sync(s->sess, sync_prompt, sync_err, sizeof(sync_err));
        pulsar_session_set_progress(s->sess, NULL, NULL);
        pulsar_session_set_display_progress(s->sess, NULL, NULL);
        if (rc == 0) {
            server_log(PULSAR_LOG_KVCACHE,
                       "pulsar-server: tool checkpoint canonicalized ctx=%s source=%s cached=%d target=%d %.3fs",
                       ctx, source, cached, canonical.len, server_now_sec() - progress.t0);
            s->trace_event(trace_id, "tool checkpoint canonicalized: source=%s common=%d live=%d canonical=%d cached=%d",
                           source, common, live_len, canonical.len, cached);
        } else {
            server_log(PULSAR_LOG_KVCACHE,
                       "pulsar-server: tool checkpoint canonicalization failed ctx=%s source=%s cached=%d target=%d "
                       "error=\"%s\"", ctx, source, cached, canonical.len, sync_err);
            s->trace_event(trace_id, "tool checkpoint canonicalization failed: %s", sync_err);
        }
        pulsar_tokens_free(&effective);
        free(path);
    }

done:
    pulsar_tokens_free(&canonical);
    buf_free(&rendered);
    free(suffix_text);
}



/* =========================================================================
 * Resumable per-slot generation (multi-session increment 2).
 * =========================================================================
 *
 * The old run-to-completion generate_job() is restructured into a state
 * machine the worker steps in bounded quanta:
 *
 *   GEN_PREFILL_COLD -> GEN_PREFILL_MAIN -> GEN_DECODE_INIT -> GEN_DECODE
 *        (one engine chunk per quantum)         (K tokens per quantum)
 *                                                       |
 *                              GEN_DONE <- GEN_FINISH <-+
 *                                              |
 *                (tool-error recovery, the old goto decode_again)
 *                                              v
 *                                       GEN_DECODE_INIT
 *
 * Everything that must survive across quanta lives in gen_state, hung off the
 * slot. A prefill quantum uses the engine's own chunk boundaries: a cancel
 * callback interrupts pulsar_session_sync() after one completed chunk (only when
 * pulsar_session_prefill_quantum_min_suffix() says resumption is bit-exact) and
 * the next quantum re-issues the sync, which resumes from the checkpoint.
 * A decode quantum runs the sampling loop for at most
 * PULSAR_SERVER_DECODE_QUANTUM_TOKENS tokens. Between quanta the session is not
 * touched, so the spec-decode carry (spec_carry_* in session.cpp) and sampling
 * rng stay valid; with one slot the quanta run back-to-back and the output is
 * byte-identical to the old function.
 *
 * Bounded exceptions that INTENTIONALLY stay run-to-completion inside one
 * quantum (kept in increment 3's scheduler by design): the tool-error
 * recovery syncs (short model-visible suffix append) and
 * canonicalize_tool_checkpoint's rebuild in GEN_FINISH. Both are rare repair
 * paths that must observe a consistent session frontier; a co-scheduled slot
 * simply waits out the (bounded) repair. The final logits-writing prefill
 * chunk likewise always completes within its quantum — the engine's cancel
 * check only interrupts when enough suffix remains for a bit-exact resume
 * (pulsar_session_prefill_quantum_min_suffix).
 *
 * The largest quantum overshoot in the system is none of the above: it is
 * lazy slot provisioning (provision_slot in the scheduler below), whose
 * pulsar_session_create is a multi-GiB allocation that can take SECONDS and
 * stalls every bound slot for its duration — larger than the DSpark fused
 * step's ≤17-token burst. Deliberate: all GPU work stays on this one thread
 * (CUDA-state audit, pulsar_server_internal.h).
 */



/* Chunk-note wrapper around server_progress_cb: counts completed prefill
 * chunks in the CURRENT pulsar_session_sync call so the cancel callback can
 * interrupt after exactly one chunk. Counters are reset before each sync.
 * The fused lane calls it too, once per recorded chunk (server_sched.cpp), so a
 * prompt riding fused rounds reports progress, keeps its SSE stream alive and
 * logs its rate exactly like a classic sync. */
void gen_prefill_progress_cb(void *ud, const char *event, int current, int total) {
    gen_state *g = (gen_state *)ud;
    if (event && strcmp(event, "prefill_chunk") == 0) {
        if (g->prefill_last_current >= 0 && current > g->prefill_last_current) {
            g->prefill_chunks_done++;
        }
        if (current > g->prefill_last_current) g->prefill_last_current = current;
        g->prefill_total = total;
        /* L281: publish the snapshot here as well as per quantum.  A prefill that cannot yield
         * -- an image prompt's mm sync runs to completion (gen_step_prefill arms no cancel hook
         * for it) -- kept the worker inside one sync for its whole length, and the per-slot
         * gauges (/metrics slot_prefill_*, slot_position; the TUI's bar) stood still for minutes
         * while the chunk lines logged.  One mutex + copy per chunk (~3 s apart) is host noise. */
        if (g->progress.srv) g->progress.srv->publish_metrics_snapshot();
    }
    server_progress_cb(&g->progress, event, current, total);
}



/* One prefill chunk per quantum. Only interrupt when the engine guarantees
 * bit-exact resumption AND enough suffix remains that the resumed sync takes
 * the batched chunk path rather than the single-token tail path (see
 * pulsar_session_prefill_quantum_min_suffix). */
/* Has the client gone away?  A request whose peer has disconnected is pure
 * waste: on this box a single stream is ~16 t/s, so an abandoned long generation
 * burns minutes of EXCLUSIVE GPU that a live request could have used.  Streaming
 * requests already notice via a failed write (send_all), but a NON-streaming
 * request writes nothing until the very end, so nothing detected the disconnect
 * at all — it ran to max_tokens for a socket no one was reading.
 *
 * Polled non-blocking, and only at coarse boundaries (once per decode quantum,
 * ~1/s, and once before a queued job starts) — never per token.
 *
 * POLLRDHUP is the signal that matters: a client that times out or is Ctrl-C'd
 * sends FIN, which shows up as POLLRDHUP and NOT as POLLHUP (POLLHUP needs a
 * full teardown/RST).  The deliberate trade: a client that half-closes its write
 * side while still reading the response would be treated as gone.  HTTP clients
 * do not do that while awaiting a response, and the same assumption is standard
 * in production servers — but PULSAR_ABORT_ON_DISCONNECT=0 restores the old
 * run-to-completion behavior without a rebuild if some client ever misbehaves.
 * The env is read ONCE (project rule: no per-token getenv). */
bool gen_client_disconnected(int fd) {
    if (fd < 0) return false;
    struct pollfd p;
    p.fd = fd;
    p.events = POLLRDHUP;
    p.revents = 0;
    if (poll(&p, 1, 0) <= 0) return false;   /* 0 = quiet = still connected */
    return (p.revents & (POLLERR | POLLHUP | POLLNVAL | POLLRDHUP)) != 0;
}


static bool gen_prefill_cancel_cb(void *ud) {
    const gen_state *g = (const gen_state *)ud;
    /* A client that cancelled or hung up during a long prefill must stop the
     * engine — otherwise opencode's deep-context prefills run to completion after
     * a cancel, burning the GPU and a bank. Polled between chunks (not per token);
     * gen_step_prefill abandons on the resulting interrupt. */
    if (gen_client_disconnected(g->j->fd)) return true;
    if (g->prefill_min_suffix == 0) return false;
    if (g->prefill_chunks_done < 1) return false;
    if (g->prefill_last_current < 0 || g->prefill_total <= g->prefill_last_current) return false;
    return (uint32_t)(g->prefill_total - g->prefill_last_current) >= g->prefill_min_suffix;
}

/* Bind THIS slot's prefill callbacks to the shared pool session.  Every slot
 * shares s->sess and the callbacks are per-session, so any other slot's
 * begin (which binds its own) or stream begin (which clears them) replaces
 * them: armed once per job, a long prefill lost its progress callback to the
 * next request, never counted a chunk, and the cancel callback never yielded
 * -- a 202k prompt held another slot's decode for 214 s (L260).  Called
 * before every sync this slot issues. */
static void gen_arm_prefill_callbacks(pulsar_session *sess, gen_state *g) {
    pulsar_session_set_progress(sess, gen_prefill_progress_cb, g);
    pulsar_session_set_display_progress(sess, server_progress_cb, &g->progress);
    pulsar_session_set_cancel(sess, gen_prefill_cancel_cb, g);
}



/* Shared failure epilogue for both prefill phases (the old duplicated blocks
 * after each pulsar_session_sync failure). Token vectors and the disk path are
 * freed centrally by gen_state_free. */
void server::gen_prefill_fail(session_slot *sl, bool discard_loaded_entry) {
    auto *s = this;
    gen_state *g = sl->gen;
    pulsar_session_set_cancel(s->sess, NULL, NULL);
    pulsar_session_set_progress(s->sess, NULL, NULL);
    pulsar_session_set_display_progress(s->sess, NULL, NULL);
    if (discard_loaded_entry) {
        s->kv_cache_discard_failed_chain(g->disk_cache_path);
    } else if (g->disk_cache_path) {
        server_log(PULSAR_LOG_KVCACHE, "pulsar-server: kv cache kept segment=%s (prefill ended: %s)",
                   g->disk_cache_path, g->err);
    }
    /* pulsar_session_invalidate acts on the LIVE bank.  A fused round abandons a
     * rider while another conversation's bank is live (L261 2026-10-02: a client
     * that disconnected during a fused prefill wiped a decoding bank's KV, its next
     * step failed on both ranks and the pair was marked failed), so install the
     * slot's own bank first.  A bank that cannot be installed is left alone, and so
     * is every other bank. */
    if (s->bank_switch(sl->bank))
        pulsar_session_invalidate(s->sess);
    else
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: bank %d not installed after a failed prefill; left as is "
                   "(no other bank is invalidated)", sl->bank);
    s->trace_event(g->trace_id, "prefill failed: %s", g->err);
    s->send_prefill_failure_response(g->j, &g->progress, g->ctx_span,
                                  g->req_flags, g->err);
    g->phase = GEN_DONE;
}



/* L261: place an image request's images on a live-continuation effective prompt
 * (the live tokens [0, live_len) + freshly tokenized request text).  The images
 * arrive in prompt order; the first ones are HELD -- the live history already
 * carries their blocks, so their start_pos is where those blocks begin -- and
 * the rest are NEW -- their placeholders sit in the suffix and the one producer
 * of sentinel blocks expands them at their positions in the effective prompt
 * (a block's length depends on its position).  Any disagreement (a live block
 * that does not parse, more blocks than images, a placeholder count that does
 * not match) is refused by name; the engine's fingerprint check then decides
 * whether the held images are really the live ones. */
static bool image_continuation_place(pulsar_engine *e, pulsar_tokens *eff, int live_len,
                                     pulsar_image_ref *images, int n_images,
                                     char *why, size_t whylen) {
    enum { HELD_MAX = 64 };
    int starts[HELD_MAX];
    const int held = pulsar_image_block_starts(e, eff, live_len, starts, HELD_MAX);
    if (held < 0 || held > HELD_MAX || held > n_images) {
        snprintf(why, whylen, "the live history carries %d image block(s) for %d image(s)", held, n_images);
        return false;
    }
    for (int i = 0; i < held; i++) images[i].start_pos = starts[i];
    pulsar_tokens out = {0};
    if (!pulsar_expand_image_placeholders(e, eff, live_len, images + held, n_images - held, &out, why, whylen))
        return false;
    pulsar_tokens_free(eff);
    *eff = out;
    return true;
}



int server_image_cold_cut(int cut, const pulsar_image_ref *images, int n_images, int *inside) {
    *inside = -1;
    for (int i = 0; cut > 0 && i < n_images; i++)
        if (images[i].start_pos < cut) {
            *inside = i;
            return 0;
        }
    return cut;
}

/* Resolve the prompt against every cache layer and decide the prefill plan.
 *
 * Clients resend full prompts as text.  The worker first tries the old exact
 * token-prefix hit, then a rendered-text prefix hit for the live checkpoint,
 * then disk text-prefix restart snapshots, then a cold prefill.  On text-prefix
 * hits we build a fresh effective prompt from the checkpoint's exact token
 * history plus a newly tokenized string suffix; the canonical full-prompt
 * tokens are not sliced because BPE may merge across the byte boundary.  Cold
 * prompt caching is handled before generation: if the stable checkpoint is
 * shorter than the full prompt, we prefill to that boundary, store it, and
 * immediately continue to the real prompt.  The live graph therefore always
 * moves forward. */
void server::gen_begin(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    job *j = g->j;
    /* An image request rides the LIVE and DISK resolvers like any other (L261,
     * L273): a live continuation keeps the bank's history, a chain restores the
     * text prefix below the conversation's first image block (a chain never holds
     * an image row: pulsar_kvchain_persist ends at the first block), and in both
     * cases image_continuation_place re-derives every image's start_pos on the
     * effective prompt -- the held ones at their live blocks, the rest expanded
     * from their placeholders in the suffix -- so the engine merges them where
     * their blocks fall.  Before L273 the disk resolver was off for it: the pair
     * 2026-10-06 20:21, an image turn routed to a fresh bank prefilled 150k
     * tokens cold past the sys-prefix cut where a text turn loads the chain. */
    const bool image_request = j->req.n_images > 0;
    /* Tier-2: install this slot's bank before ANY s->sess touch below (all the
     * pos/common-prefix/tokens reads and the prefill sync run against the live
     * bank). No-op in classic mode / when already live. Finding 1: a failed spill
     * restore (KV unrecoverable) must fail the request, not run against another
     * bank's KV. */
    if (!s->bank_switch(sl->bank)) {
        snprintf(g->err, sizeof g->err,
                 "bank %u state restore failed (evicted KV unrecoverable)", (unsigned)sl->bank);
        s->gen_prefill_fail(sl, false);
        return;
    }
    const int old_pos = pulsar_session_pos(s->sess);
    /* EVAL PIN: report no live common prefix, so every request re-prefills
     * from position 0 regardless of what this bank served before.  The
     * choke-point twins live in slot_common_prefix (routing) and the live
     * resolvers below. */
    /* L115: one authority, both currencies (see pulsar_prefix_match). */
    pulsar_prefix_match pm;
    s->slot_prefix_match(sl, &j->req.prompt, &pm);
    const int common = pm.live_cut;          /* live/bank side: KV rows */
    trace_cache_diag cache_diag = {0};
    trace_cache_capture(&cache_diag, pulsar_session_tokens(s->sess),
                        &j->req.prompt, old_pos, common);
    pulsar_tokens effective_prompt = {0};
    const pulsar_tokens *prompt_for_sync = &j->req.prompt;
    const bool responses_protocol = j->req.api == API_RESPONSES;
    bool responses_live_continuation = false;
    bool anthropic_live_continuation = false;
    const char *responses_live_match = NULL;
    int responses_live_match_ids = 0;
    int anthropic_live_match_ids = 0;
    int cached = 0;
    const char *cache_source = "none";
    int disk_cached = 0;
    if (image_request) {
        server_log(PULSAR_LOG_PREFILL,
                   "pulsar-server: image request (%d image%s): live and disk resolvers apply, "
                   "cold phase covers the sys-prefix only", j->req.n_images, j->req.n_images == 1 ? "" : "s");
    }
    /* Responses gets the first chance to continue from live state.  This is
     * the whole point of the API shape: a request that is bound to prior live
     * output by visible transcript or tool call ids does not need to prove an
     * exact token-prefix match.  Exact token/text/disk matching remains the
     * fallback when the live state is absent or no longer describes the
     * request. */
    cached = s->responses_live_visible_prefix_prompt(sl, &j->req, old_pos,
                                                      &effective_prompt);
    cache_source = cached > 0 ? "responses-visible" : "none";
    if (cached > 0) {
        responses_live_match = "visible-prefix";
        if (s->responses_live_matches_request(sl, &j->req.responses_live_call_ids,
                                           old_pos))
        {
            responses_live_match_ids = j->req.responses_live_call_ids.len;
        }
    }
    if (cached == 0) {
        cached = s->responses_live_continuation_prompt(sl, &j->req, old_pos,
                                                    &effective_prompt,
                                                    &responses_live_match_ids);
        cache_source = cached > 0 ? "responses-tool-output" : "none";
        if (cached > 0) responses_live_match = "tool-output-ids";
    }
    if (cached > 0) {
        responses_live_continuation = true;
        prompt_for_sync = &effective_prompt;
    } else {
        cached = s->anthropic_live_continuation_prompt(sl, &j->req, old_pos,
                                                    &effective_prompt,
                                                    &anthropic_live_match_ids);
        if (cached > 0) {
            anthropic_live_continuation = true;
            cache_source = "anthropic-tool-output";
            prompt_for_sync = &effective_prompt;
        }
    }
    if (cached == 0 && responses_protocol &&
        j->req.responses_requires_live_tool_state)
    {
        /* The parser saw a valid live call_id, but by worker execution time the
         * live frontier no longer matches.  Since the request did not replay
         * the prior assistant call, there is no stateless prefix to match and
         * no disk key to search by. */
        pulsar_tokens_free(&effective_prompt);
        http_error(j->fd, 409,
                   "Responses continuation state is not available; retry by replaying the full input history");
        g->phase = GEN_DONE;
        return;
    } else if (cached == 0 && j->req.api == API_ANTHROPIC &&
               j->req.anthropic_requires_live_tool_state)
    {
        pulsar_tokens_free(&effective_prompt);
        http_error(j->fd, 409,
                   "Anthropic continuation state is not available; retry by replaying the full messages history");
        g->phase = GEN_DONE;
        return;
    } else if (cached == 0) {
        /* L115 increment 2: the live KV serves any prompt whose BYTES are a
         * prefix of the live history — not only an exact extension.  The
         * shape that kept recurring is a SHORTER echo: the client strips
         * generated reasoning, so live carries a tail the prompt does not
         * (measured 2026-08-28: live 390,258 vs echo 390,018, plus seams),
         * and the old `common == old_pos` gate could never pass it.
         * Falling through is expensive, not merely slower: the disk path
         * below REPLACES the live session with a snapshot — that day a 390k
         * live history was discarded for a 281k one and 108,360 tokens were
         * re-prefilled.  Serving it is safe because sync rewinds to the
         * byte-matched live token and stitches, and rewind is position-true
         * and value-true since L120/L124.
         * The half-of-prompt guard keeps a genuinely different conversation
         * (sharing only a system preamble) on its own disk snapshot. */
        /* Reuse the live KV whenever the prompt's BYTES are a prefix of it —
         * exact extension, seam-shifted echo, shorter echo (stripped
         * reasoning), or a client rollback.  The one case to decline is a
         * match so shallow it is just the shared header, where a fresh
         * slot or a disk snapshot serves better: that is exactly
         * server_slot_match_is_trivial, the SAME predicate routing uses to
         * protect warm state.  Reusing it rather than inventing a second
         * threshold keeps one authority for "is this match worth keeping".
         * Accounting takes the PROMPT-side count. */
        const bool trivial =
            server_slot_match_is_trivial(common, old_pos,
                                         s->slot_trivial_common_tokens,
                                         s->slot_trivial_common_tokens);
        /* L266/L284: the byte match is what the bank's tokens re-spell; the count is where the sync actually
         * starts -- the match itself when the sync continues the bank where it stands, else the grid
         * checkpoint it resumes from (every family: decode rows past the prefill are recomputed) */
        const int resume = pulsar_session_bank_resume_at(s->sess, sl->bank, &j->req.prompt);
        cached = (pm.prompt_cut > 0 && !trivial) ? resume : 0;
        cache_source = cached <= 0 ? "none" : cached == pm.prompt_cut ? "memory-token" : "memory-checkpoint";
    }
    if (cached == 0 && old_pos > 0) {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: live kv cache miss%s live=%d prompt=%d common=%d reason=%s",
                   responses_protocol ? " RESPPROTO" : "",
                   old_pos, j->req.prompt.len, common,
                   trace_cache_miss_reason(&cache_diag));
    }
    if (s->kv.enabled && cached == 0 && old_pos >= s->kv.opt.min_tokens) {
        /* Loading a chain replaces the live bank.  Persist its history first
         * (only the segments the store lacks), so the newer conversation state
         * outlives the restore. */
        s->kv_cache_persist(sl, "evict");
    }
    if (cached == 0) {
        disk_cached = s->kv_cache_try_load(sl, &j->req, &effective_prompt,
                                        &g->disk_cache_path);
        if (disk_cached > 0) {
            cached = disk_cached;
            cache_source = "disk-text";
            prompt_for_sync = &effective_prompt;
        }
    }
    /* An effective prompt is the live (or chain-restored) tokens plus freshly
     * tokenized request text: place the images on it -- the held ones where
     * that history already carries their blocks, the new ones expanded from
     * their placeholders at their positions there.  A request whose images do
     * not line up with the history prefills the rendered prompt instead, said
     * by name. */
    if (image_request && cached > 0 && prompt_for_sync == &effective_prompt) {
        char why[256];
        if (!image_continuation_place(s->engine, &effective_prompt, cached, j->req.images,
                                      j->req.n_images, why, sizeof(why))) {
            server_log(PULSAR_LOG_WARNING,
                       "pulsar-server: image request: the %s continuation cannot place its images "
                       "(%s) -- prefilling the rendered prompt", cache_source, why);
            pulsar_tokens_free(&effective_prompt);
            prompt_for_sync = &j->req.prompt;
            cached = 0;
            disk_cached = 0;
            cache_source = "none";
            responses_live_continuation = false;
            anthropic_live_continuation = false;
        }
    }
    const bool responses_reasoning_state_preserved =
        cached > 0 &&
        (!strcmp(cache_source, "responses-visible") ||
         !strcmp(cache_source, "responses-tool-output"));
    const bool responses_visible_replay_without_reasoning =
        responses_protocol &&
        j->req.responses_requires_live_reasoning &&
        !responses_reasoning_state_preserved;
    const int prompt_tokens = prompt_for_sync->len;
    /* OpenAI usage details: the reusable prefix is a cache read, while the
     * effective prompt suffix evaluated by pulsar_session_sync() is written into
     * the live KV cache and can be reused by the next request. */
    j->req.cache_read_tokens = cached;
    j->req.cache_write_tokens = prompt_tokens > cached ? prompt_tokens - cached : 0;

    /* Prometheus /metrics: prompt-throughput + prefix-cache-hit counters. */
    pthread_mutex_lock(&s->mu);
    s->m_prompt_tokens += (uint64_t)(prompt_tokens > 0 ? prompt_tokens : 0);
    s->m_prefix_queries += (uint64_t)(prompt_tokens > 0 ? prompt_tokens : 0);
    s->m_prefix_hits += (uint64_t)(cached > 0 ? cached : 0);
    pthread_mutex_unlock(&s->mu);

    g->prompt_tokens = prompt_tokens;
    g->t0 = server_now_sec();
    g->trace_id = s->trace_begin(j, cached, prompt_tokens, &cache_diag,
                              cache_source, disk_cached, g->disk_cache_path);
    request_ctx_span(g->ctx_span, sizeof(g->ctx_span), cached, prompt_tokens);
    sl->prefill_counted = cached;   /* L114 counter watermark: computed rows start here */
    g->progress = (server_prefill_progress){
        .srv = s,
        .slot = sl,
        .kind = j->req.kind,
        .prompt_tokens = prompt_tokens,
        .cached_tokens = cached,
        .has_tools = j->req.has_tools,
        .responses_protocol = responses_protocol,
        .t0 = g->t0,
        .fd = j->fd,
        .stream = j->req.stream,
    };
    snprintf(g->progress.ctx, sizeof(g->progress.ctx), "%s", g->ctx_span);
    log_flags(g->req_flags, sizeof(g->req_flags), responses_protocol,
              j->req.has_tools, false, false, false);
    if (responses_live_continuation) {
        server_log(PULSAR_LOG_PREFILL,
                   "pulsar-server: responses live continuation RESPPROTO match=%s ids=%d cached=%d prompt=%d",
                   responses_live_match ? responses_live_match : "unknown",
                   responses_live_match_ids,
                   cached,
                   prompt_tokens);
    } else if (anthropic_live_continuation) {
        server_log(PULSAR_LOG_PREFILL,
                   "pulsar-server: anthropic live continuation match=tool-output-ids ids=%d cached=%d prompt=%d",
                   anthropic_live_match_ids,
                   cached,
                   prompt_tokens);
    }
    if (responses_visible_replay_without_reasoning) {
        /* The request replays a prior tool-call turn but omits the hidden
         * reasoning that originally led to it.  A live Responses checkpoint, or
         * a responses-visible disk checkpoint, would preserve that hidden KV.
         * If neither is available, continue from the visible transcript instead
         * of surfacing a hard error to the user.  This is lower fidelity, but it
         * lets old / restarted agent sessions recover and is exactly what the
         * client asked us to prefill. */
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: responses replay RESPPROTO missing reasoning state; continuing from visible history source=%s cached=%d prompt=%d",
                   cache_source,
                   cached,
                   prompt_tokens);
        s->trace_event(g->trace_id,
                    "responses replay missing reasoning state; continuing from visible history source=%s cached=%d",
                    cache_source, cached);
    }
    server_log(PULSAR_LOG_PREFILL,
               "pulsar-server: %s ctx=%s%s%s prompt start",
               j->req.kind == REQ_CHAT ? "chat" : "completion",
               g->ctx_span,
               g->req_flags[0] ? " " : "",
               g->req_flags);
    pulsar_session_set_progress(s->sess, gen_prefill_progress_cb, g);
    pulsar_session_set_display_progress(s->sess, server_progress_cb, &g->progress);

    /* The one cold checkpoint is the shared preamble (system prompt + tools,
     * before the task message): worth a separate prefill phase however long
     * the conversation has grown behind it, because every new conversation
     * can text-prefix restore from it.  The whole-prompt cold cut is gone
     * (L260): it split every fresh prompt into an aligned prefix and a tail
     * forward -- on the pair ~300 ms for a 61-token tail (the grouped MoE at
     * small M reads ~every expert) plus ~167 ms of staging, ~24% of a 2K
     * prompt -- for entries L261's baseline shows rarely hit.  A long
     * conversation still checkpoints through the continued store. */
    int cold_store_len = 0;
    if (cached == 0 &&
        s->kv.enabled &&
        prompt_for_sync->len >= s->kv.opt.min_tokens)
    {
        const int anchor = kv_cache_chat_anchor_pos(&s->kv, prompt_for_sync, &s->turn_markers);
        cold_store_len = kv_cache_sys_prefix_cut(&s->kv, anchor);
    }
    /* An image request's cold phase covers the shared preamble only, and the
     * cut sits a margin below the first user turn, so no image block begins
     * inside it: that phase is the PLAIN text sync (gen_step_prefill), and the
     * main sync then merges every image on top of the checkpoint it leaves --
     * L261 reuse with 0 held images and all of them new, the text-turn-then-
     * image-turn shape.  A block that does begin inside the cut would be split
     * from its merge, so such a cut is dropped and the main pass merges from
     * token 0.  (l264t, 2026-10-05: with the !image_request gate gone, the cut
     * itself was handed to sync_mm with every start_pos past its end -> HTTP
     * 400 "image 0 at N is not a sentinel block" for any prompt whose first
     * user turn sat past ~4k tokens, i.e. every Claude Code request.) */
    if (image_request) {
        int inside = -1;
        const int cut = server_image_cold_cut(cold_store_len, j->req.images, j->req.n_images, &inside);
        if (inside >= 0)
            server_log(PULSAR_LOG_PREFILL,
                       "pulsar-server: image request: image %d at %d begins inside the "
                       "sys-prefix cut %d -- no cold phase, the main pass merges it",
                       inside, j->req.images[inside].start_pos, cold_store_len);
        cold_store_len = cut;
    }
    g->cold_store_len = cold_store_len;
    /* Transfer prompt ownership into the slot state; the prefill phases run in
     * later quanta. */
    g->effective_prompt = effective_prompt;
    g->prompt_for_sync = prompt_for_sync == &effective_prompt ?
                         &g->effective_prompt : prompt_for_sync;
    g->responses_protocol = responses_protocol;
    g->responses_live_continuation = responses_live_continuation;
    g->anthropic_live_continuation = anthropic_live_continuation;

    /* Prefill quantum policy: interrupt the engine's chunk loop only when
     * resumption is bit-exact for this session (see gen_prefill_cancel_cb). The
     * cancel callback itself is armed per-sync in gen_step_prefill, NOT here: in
     * pool mode every slot shares the one pool session (s->sess), so a once-per-job set would be
     * clobbered by the next job the worker binds before prefilling this one. */
    g->prefill_min_suffix = pulsar_session_prefill_quantum_min_suffix(s->sess);

    if (s->kv.enabled &&
        g->cold_store_len >= s->kv.opt.min_tokens &&
        g->cold_store_len < g->prompt_for_sync->len)
    {
        tokens_copy_prefix(&g->cold_prefix, g->prompt_for_sync, g->cold_store_len);
        g->phase = GEN_PREFILL_COLD;
    } else {
        g->phase = GEN_PREFILL_MAIN;
    }
}



/* One prefill quantum: (re-)issue the sync toward the phase's target; the
 * cancel callback stops it after one completed chunk and the checkpoint
 * carries the progress to the next quantum. */
void server::gen_step_prefill(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    const bool cold = g->phase == GEN_PREFILL_COLD;
    const pulsar_tokens *target = cold ? &g->cold_prefix : g->prompt_for_sync;

    /* Arm the callbacks on THIS slot's gen_state right before the sync.
     * In pool mode every slot shares the one pool session (s->sess), and the worker binds several
     * jobs (each of which would set the callback) before prefilling any of them,
     * so a once-per-job set in gen_begin leaves the LAST-bound slot's callback on
     * the shared session. An earlier slot's prefill would then yield on ANOTHER
     * slot's progress and, interrupted before its own first chunk, be misread as a
     * fatal error ("interrupted", HTTP 500). The worker prefills serially, so
     * setting it here binds the correct callback for this exact sync.
     *
     * L281 (b): an IMAGE prefill is interruptible too.  Every quantum re-enters
     * through sync_mm with the same images (the job owns them), the engine records
     * the blocks an interrupted quantum completed (pulsar_session::live_images), and
     * the planner honours a stop only at a grid point outside every block -- where
     * the resume is exact.  So a client that hangs up stops the prefill within a
     * chunk (it ran 379 s to the end on the pair, 2026-10-07), and another slot can
     * interleave with a long image prompt as with a text one. */
    /* The cold phase's target is the shared preamble, which carries no image
     * block (gen_begin drops a cut that any block begins inside), so it is the
     * plain text sync -- interruptible like any text prefill -- and the
     * checkpoint it leaves is one the main image sync extends. */
    const int n_images = cold ? 0 : g->j->req.n_images;
    gen_arm_prefill_callbacks(s->sess, g);

    g->prefill_chunks_done = 0;
    g->prefill_last_current = -1;
    g->prefill_total = 0;
    const int rc = n_images > 0
        ? pulsar_session_sync_mm(s->sess, target, g->j->req.images, n_images,
                                 g->err, sizeof(g->err))
        : pulsar_session_sync(s->sess, target, g->err, sizeof(g->err));
    if (rc == PULSAR_SESSION_SYNC_INTERRUPTED) {
        if (gen_client_disconnected(g->j->fd)) {
            /* Client cancelled mid-prefill: abandon rather than resume or fail, so
             * a long deep-context prefill stops promptly on disconnect instead of
             * running to completion (the decode loop already abandons this way). */
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: client disconnected during prefill, abandoning");
            snprintf(g->err, sizeof(g->err), "client disconnected");
            s->gen_prefill_fail(sl, false);
            return;
        }
        if (g->prefill_chunks_done > 0) return; /* voluntary yield; resume next quantum */
        /* Interrupted without progress cannot be our cancel callback; fail
         * rather than risk a live-lock re-issuing the same sync forever. */
        s->gen_prefill_fail(sl, false);
        return;
    }
    if (rc != 0) {
        s->gen_prefill_fail(sl, true);
        return;
    }

    if (cold) {
        /* The cold prefill ended on the sys-prefix cut, a grid point, so the
         * shared preamble is a checkpoint now: persist it as its own chain. */
        s->kv_cache_persist(sl, "sys-prefix");
        pulsar_tokens_free(&g->cold_prefix);
        g->phase = GEN_PREFILL_MAIN;
        return; /* the cold store is a quantum boundary of its own */
    }

    pulsar_session_set_cancel(s->sess, NULL, NULL);
    s->gen_stream_begin(sl);
}

/* Runs once, in the same quantum that completed the main prefill: clear stale
 * live bindings, persist checkpoints, emit response identity, and start the
 * protocol stream projections that persist across all decode quanta. */
void server::gen_stream_begin(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    job *j = g->j;
    free(g->disk_cache_path);
    g->disk_cache_path = NULL;
    /* Once a non-live request wins, old protocol live bindings are stale. Keep
     * a binding only when this request explicitly continued from it. */
    if (!g->responses_live_continuation) s->responses_live_clear(sl);
    if (!g->anthropic_live_continuation) s->anthropic_live_clear(sl);
    pulsar_session_set_progress(s->sess, NULL, NULL);
    pulsar_session_set_display_progress(s->sess, NULL, NULL);
    s->kv_cache_persist(sl, "prompt");
    server_log(PULSAR_LOG_PREFILL,
               "pulsar-server: %s ctx=%s%s%s prompt done %.3fs",
               j->req.kind == REQ_CHAT ? "chat" : "completion",
               g->ctx_span,
               g->req_flags[0] ? " " : "",
               g->req_flags,
               server_now_sec() - g->t0);
    /* Random ids, like the tool-call and Responses ids (L192 item 7): a
     * counter leaked request ordering and made the unseeded sampler seed
     * below guessable. */
    random_prefixed_id(g->id, sizeof(g->id), j->req.kind == REQ_CHAT ? "chatcmpl-" : "cmpl-", 12);

    g->structured_stream = request_uses_structured_stream(&j->req);
    g->openai_live_chat = request_uses_openai_live_stream(&j->req);
    g->responses_live_chat = request_uses_responses_live_stream(&j->req);
    g->responses_created_at = (long)time(NULL);
    if (j->req.stream) {
        if (g->progress.stream_failed) {
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: %s ctx=%s%s%s stream closed during prefill",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       g->ctx_span,
                       g->req_flags[0] ? " " : "",
                       g->req_flags);
            g->phase = GEN_DONE;
            return;
        }
        /* The prefill progress callback may have already sent the SSE headers
         * to keep the connection alive during a long prefill. Only emit them
         * here when prefill never fired (e.g. fully cached prompt). */
        if (!g->progress.headers_sent && !sse_headers(j->fd)) {
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: %s ctx=%s%s%s sse headers failed",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       g->ctx_span,
                       g->req_flags[0] ? " " : "",
                       g->req_flags);
            g->phase = GEN_DONE;
            return;
        }
        g->progress.headers_sent = true;
        if (j->req.api == API_ANTHROPIC &&
            !anthropic_sse_start_live(j->fd, &j->req, g->id,
                                      g->prompt_tokens, &g->anthropic_live)) {
            server_log(PULSAR_LOG_GENERATION, "pulsar-server: chat ctx=%s anthropic stream start failed", g->ctx_span);
            g->phase = GEN_DONE;
            return;
        }
        if (j->req.api == API_OPENAI && j->req.kind == REQ_CHAT &&
            !sse_chunk(j->fd, &j->req, g->id, NULL, NULL)) {
            server_log(PULSAR_LOG_GENERATION, "pulsar-server: chat ctx=%s openai role chunk failed", g->ctx_span);
            g->phase = GEN_DONE;
            return;
        }
        if (g->openai_live_chat) {
            openai_stream_start(&j->req, &g->openai_live);
            /* Borrowed for the job's lifetime: the delta emitters attach the
             * entries their bytes release (see append_openai_logprobs_delta). */
            g->openai_live.lp = &g->logprobs;
        }
        if (g->responses_live_chat) {
            responses_stream_init(&j->req, &g->responses_live);
            g->responses_live.active = true;
            if (!responses_sse_created(j->fd, &j->req, &g->responses_live, g->responses_created_at)) {
                server_log(PULSAR_LOG_GENERATION,
                           "pulsar-server: chat ctx=%s%s%s responses created event failed",
                           g->ctx_span,
                           g->req_flags[0] ? " " : "",
                           g->req_flags);
                g->phase = GEN_DONE;
                return;
            }
        }
        /* L267: the protocol sink this response streams into; the family's output
         * parser drives it (gen_emit_token). */
        if (j->req.api == API_ANTHROPIC)
            anthropic_sink_init(&g->sink, j->fd, s, &j->req, g->id, &g->anthropic_live);
        else if (g->openai_live_chat)
            openai_sink_init(&g->sink, j->fd, s, &j->req, g->id, &g->openai_live);
        else if (g->responses_live_chat)
            responses_sink_init(&g->sink, j->fd, s, &j->req, g->id, &g->responses_live);
    }

    g->recovery_attempted = false;
    if (j->req.seed) {
        g->rng = j->req.seed;
    } else {
        uint64_t r = 0;
        if (!random_bytes(&r, sizeof(r)) || r == 0)
            pulsar_die("random_bytes failed; cannot seed an unseeded request");
        g->rng = r;
    }
    g->phase = GEN_DECODE_INIT;
}



/* (Re)initialize a decode attempt: the body of the old decode_again label.
 * Runs both for a fresh request and after a tool-error recovery appended a
 * model-visible correction to the live session. */
void server::gen_decode_init(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    job *j = g->j;
    buf_free(&g->text);
    g->plain_stream_pos = 0;
    g->stop_scan_from = 0;
    g->finish = "length";
    g->completion = 0;
    /* Continued attempts (tool-error recovery) spend the request's ONE
     * max_tokens budget: completion_total holds what earlier attempts already
     * generated. */
    g->max_tokens = j->req.max_tokens - g->completion_total;
    int room = pulsar_session_ctx(s->sess) - pulsar_session_pos(s->sess);
    g->next_decode_log = 50;
    if (g->max_tokens < 0) g->max_tokens = 0;
    if (g->max_tokens > room) g->max_tokens = room;
    s->trace_event(g->trace_id, "prefill done; decode_max=%d ctx_room=%d", g->max_tokens, room);
    g->decode_t0 = server_now_sec();
    /* L119: reset the request-scoped DSpark accumulators (decode_again also
     * lands here, keeping them consistent with the reset completion/decode_t0
     * for the attempt that actually finishes). The spec-batched lane fills
     * them per round. */
    g->req_spec_draft = 0;
    g->req_spec_accepted = 0;
    g->req_spec_rounds = 0;
    g->req_spec_gen = 0;
    g->last_decode_log_t = g->decode_t0;
    g->last_decode_log_completion = 0;
    g->thinking = thinking_state_from_prompt(&j->req);
    /* A logprobs request decodes WITHOUT speculation.  The fused DSpark step
     * verifies K drafts in one batch and keeps each position's target row only
     * inside that batch; by the time pulsar_session_generate_speculative
     * returns, the session's logits describe the position AFTER the last
     * committed token and the per-draft rows are gone.  There is no correct
     * distribution left to report for an accepted draft token, and the
     * drafter's own is a different model — so the choice is fewer tokens per
     * second, never a number from the wrong distribution. */
    g->spec_enabled = !j->req.logprobs;
    /* Entries from a superseded attempt (tool-error recovery) go with the
     * text they described: gen_decode_init discards g->text, so the ledger
     * restarts with it. */
    logprob_ledger_reset(&g->logprobs);
    g->logprobs.enabled = j->req.logprobs;
    g->logprobs.top_k = j->req.top_logprobs;

    /* L272 P3: the family's output parser for this attempt (a retry attempt starts a fresh one) */
    g->parser = j->req.family->output;
    if (g->parser_st) g->parser->destroy(g->parser_st);
    g->parser_st = g->parser->create(s, g, g->err, sizeof(g->err));
    if (!g->parser_st) {
        g->finish = "error";
        g->phase = GEN_FINISH;
        return;
    }
    /* tool_choice="required": the prompt was prefilled into an open tool-call block (thinking
     * skipped).  Seed the output with that exact prefix and let the parser read it as emitted; the
     * model now generates only the call's body. */
    if (j->req.kind == REQ_CHAT && j->req.force_tool_call && j->req.family->forced_call_seed) {
        j->req.family->forced_call_seed(&j->req, &g->text);
        if (g->parser->seed) g->parser->seed(g->parser_st, g);
        g->plain_stream_pos = g->text.len;
        /* L272: an unnamed forced call (required with several declared tools) samples its name under
         * the declared-name mask (gen_mask_tool_name); every family's seed ends at the name's opener */
        g->tool_name_constrained = (!j->req.forced_tool_name || !j->req.forced_tool_name[0]) &&
                                   j->req.tool_orders.len > 0;
        g->tool_name_from = g->text.len;
    }
    g->phase = GEN_DECODE;
}



bool gen_tool_name_open(const gen_state *g) {
    if (!g || !g->tool_name_constrained || g->tool_name_from > g->text.len) return false;
    /* the closer is looked for past the opener: DeepSeek's opener ` name="` holds the quote its closer starts with */
    const size_t no = strlen(g->j->req.family->forced_name_open);
    const size_t n = g->text.len - g->tool_name_from;
    if (n <= no) return true;
    return !strstr(g->text.ptr + g->tool_name_from + no, g->j->req.family->forced_name_close);
}

/* L272: mask `row` (the logits a constrained slot draws its next token from) to the tokens that keep the
 * function name a prefix of the family's opener + a declared tool's name + its closer.  The token bytes are built once
 * per server.  false = no token is allowed (cannot happen while a declared name is a strict extension of
 * the text; said once if it does), and the row is left as it was. */
bool gen_mask_tool_name(server *s, gen_state *g, float *row, int width) {
    if (!s->token_bytes) {
        s->token_bytes = new std::vector<std::string>((size_t)width);
        for (int t = 0; t < width; t++) {
            size_t n = 0;
            char *b = pulsar_token_text(s->engine, t, &n);
            (*s->token_bytes)[(size_t)t].assign(b, n);
            free(b);
        }
    }
    const std::vector<std::string> &bytes = *s->token_bytes;
    const request *r = &g->j->req;
    const char *so_far = g->text.ptr ? g->text.ptr + g->tool_name_from : "";
    const size_t n_so_far = g->text.len - g->tool_name_from;
    std::vector<int> keep;
    for (int t = 0; t < width && (size_t)t < bytes.size(); t++) {
        const std::string &b = bytes[(size_t)t];
        if (tool_name_token_allowed(so_far, n_so_far, b.data(), b.size(), r->family->forced_name_open,
                                    &r->tool_orders, r->family->forced_name_close))
            keep.push_back(t);
    }
    if (keep.empty()) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: chat ctx=%s%s%s forced tool name: no token continues "
                   "\"%.*s\" toward a declared tool -- sampling unmasked", g->ctx_span, g->req_flags[0] ? " " : "",
                   g->req_flags, (int)n_so_far, so_far);
        return false;
    }
    std::vector<float> kept(keep.size());
    for (size_t k = 0; k < keep.size(); k++) kept[k] = row[keep[k]];
    for (int t = 0; t < width; t++) row[t] = -INFINITY;
    for (size_t k = 0; k < keep.size(); k++) row[keep[k]] = kept[k];
    return true;
}

/* Sampling contract: request_init() pre-fills the engine defaults, so the
 * request values are already correct for non-thinking requests. In thinking
 * mode the engine defaults are re-asserted, but ONLY for parameters the
 * client left absent (per-param has_* flags set in api_parse.cpp); anything the
 * client sent explicitly is respected as-is. That includes an explicit
 * temperature==0, which selects greedy decode -- and nothing more: it does NOT
 * decide whether DSpark runs. Speculative decode is gated on one condition,
 * `spec_enabled = !req.logprobs` above, and `spec_accept_walk`
 * (session_spec.cpp) carries BOTH acceptance rules -- greedy argmax match at
 * temperature 0, sampled p/q otherwise, with the L149 compact prefilter
 * (`spec_compact_dist`) serving the sparse min-p contract that the engine
 * defaults put a thinking request in. Tool-call payload forcing (temperature=0 while
 * decoding structured tool output, in gen_resolve_sampling_decode) is a separate,
 * deliberate override applied on top of this. */
void gen_resolve_sampling(const request *req, float *temperature,
                                 int *top_k, float *top_p, float *min_p) {
    *temperature = req->temperature;
    *top_k = req->top_k;
    *top_p = req->top_p;
    *min_p = req->min_p;
    if (pulsar_think_mode_enabled(req->think_mode)) {
        if (!req->has_temperature) *temperature = PULSAR_DEFAULT_TEMPERATURE;
        if (!req->has_top_k) *top_k = 0;
        if (!req->has_top_p) *top_p = PULSAR_DEFAULT_TOP_P;
        if (!req->has_min_p) *min_p = PULSAR_DEFAULT_MIN_P;
    }
}



/* Decode-lane sampling resolution: gen_resolve_sampling plus the tool-call
 * greedy override (temperature=0 while the decode sits on a call's structure;
 * its payload -- a string-typed value or a JSON string inside another value --
 * samples normally: the DSML tracker's payload spans for DeepSeek, the Qwen
 * parser's in_payload for Qwen -- L284 P6, one rule). L116: ONE authority for
 * every decode lane — classic, plain-batched, spec-batched, mixed — so a tool
 * request samples the same wherever the scheduler routes it. Granularity is
 * one resolution per spec block / batched round in every lane (the classic
 * lane always worked this way: the override can lag a mid-block tool-marker
 * crossing by up to one block). */
void gen_resolve_sampling_decode(const gen_state *g, float *temperature,
                                 int *top_k, float *top_p, float *min_p) {
    const request *req = &g->j->req;
    gen_resolve_sampling(req, temperature, top_k, top_p, min_p);
    if (req->kind != REQ_CHAT || !req->has_tools) return;
    if (g->parser_st && g->parser->in_tool_call(g->parser_st, g)) *temperature = 0.0f;
}



/* Emit one already-decoded token into the response stream: append it to the
 * accumulated text, feed the thinking/DSML trackers, run stop-string and
 * tool-marker detection, and drive every active protocol stream projection
 * (plain SSE / OpenAI / Anthropic / Responses). Returns true when the decode
 * loop must STOP after this token (EOS, a stop string, a completed tool_calls
 * block, or a client write error), with g->finish (and g->err on error) set.
 *
 * Factored out of the classic decode loop's per-token inner loop (Tier-2 Step 5)
 * so every driver shares ONE emit path: the classic spec/plain decode loop
 * below AND the batched multi-session lanes (which sample each live bank's
 * row on the host, then call this to stream that bank's slot). It touches
 * ONLY host state hung off sl->gen + j->req + the client fd — no engine/CUDA
 * call except pulsar_token_text. That host-only property is what makes the
 * L116 tool admission to the batched lanes sound: all tool-marker tracking,
 * thinking-recovery, and stop handling here runs identically in every lane.
 * Behavior for the single-session path is byte-identical to the
 * pre-factoring inner loop. */
bool server::gen_emit_token(session_slot *sl, int token) {
    auto *s = this;
    gen_state *g = sl->gen;
    job *j = g->j;
    if (pulsar_token_is_stop(s->engine, token)) {
        g->finish = "stop";
        return true;
    }

    size_t piece_len = 0;
    char *piece = pulsar_token_text(s->engine, token, &piece_len);
    g->completion++;
    /* Lane-independent generation counter: every decode lane funnels its
     * accepted tokens through here, so this is the one place that sees them
     * all. EOS returned above, so this counts emitted tokens only. */
    s->w_gen_tokens++;

    s->trace_piece(g->trace_id, piece, piece_len);
    buf_append(&g->text, piece, piece_len);
    if (!logprob_commit(&g->logprobs, s->engine, token, piece, piece_len, g->text.len)) {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: chat ctx=%s emitted a token with no captured "
                   "distribution after %d tokens; logprobs dropped for this request",
                   g->ctx_span, g->completion);
    }
    const bool was_thinking = g->thinking.inside;
    g->thinking.feed(piece, piece_len);
    /* L077 (stop-in-reasoning): client stop sequences describe the VISIBLE
     * answer, not the reasoning stream -- matching them inside <think>
     * truncates mid-block and, with no </think> ever emitted, the finish
     * parse routes everything to reasoning_content and the client gets an
     * empty answer. Suppress matching while inside, and on the close jump
     * the scan to the start of the closing piece (the tracker reports no
     * byte offset, so at most the think-tail bytes of that one piece stay
     * scannable -- the conservative side of the boundary). */
    if (was_thinking && !g->thinking.inside) {
        const size_t close_base = g->text.len - piece_len;
        if (g->stop_scan_from < close_base) g->stop_scan_from = close_base;
    }

    size_t stop_pos = 0, stop_len = 0;
    bool hit_stop = !g->thinking.inside &&
                    stop_list_find_from(&j->req.stops, g->text.ptr,
                                        g->stop_scan_from,
                                        &stop_pos, &stop_len);
    size_t stream_len = hit_stop ?
        stop_pos : stop_list_stream_safe_len(&j->req.stops, g->text.len);
    if (stream_len > g->text.len) stream_len = g->text.len;
    stream_len = utf8_stream_safe_len(g->text.ptr, g->plain_stream_pos,
                                      stream_len, hit_stop);
    if (!hit_stop && j->req.stops.max_len > 1) {
        const size_t hold = j->req.stops.max_len - 1;
        g->stop_scan_from = g->text.len > hold ? g->text.len - hold : 0;
    }

    if (j->req.stream && !g->structured_stream && stream_len > g->plain_stream_pos) {
        char *delta = xstrndup(g->text.ptr + g->plain_stream_pos, stream_len - g->plain_stream_pos);
        bool ok = sse_chunk(j->fd, &j->req, g->id, delta, NULL);
        free(delta);
        if (!ok) {
            g->finish = "error";
            snprintf(g->err, sizeof(g->err), "client stream write failed");
            free(piece);
            return true;
        }
        g->plain_stream_pos = stream_len;
    }
    /* the family's output parser: its decode-time tracking, and the released bytes into the sink */
    if (!g->parser->feed(g->parser_st, s, g, stream_len, false)) {
        g->finish = "error";
        snprintf(g->err, sizeof(g->err), "client stream write failed");
        free(piece);
        return true;
    }
    free(piece);

    if (g->completion >= g->next_decode_log) {
        bool tool_open = false, tool_closed = false;
        if (g->parser->tool_progress) g->parser->tool_progress(g->parser_st, &tool_open, &tool_closed);
        log_decode_progress(j->req.kind, g->prompt_tokens, g->completion,
                            g->responses_protocol,
                            j->req.has_tools,
                            g->thinking.inside,
                            tool_open,
                            tool_closed,
                            g->decode_t0,
                            &g->last_decode_log_t,
                            &g->last_decode_log_completion);
        g->next_decode_log += 50;
    }

    if (hit_stop) {
        free(g->stop_sequence);
        g->stop_sequence = xstrndup(g->text.ptr + stop_pos, stop_len);
        g->finish = "stop";
        g->text.len = stop_pos;
        g->text.ptr[g->text.len] = '\0';
        pulsar_session_invalidate(s->sess);
        return true;
    }

    /* the family's turn ends before the stop token (DeepSeek: a closed DSML block) */
    if (j->req.kind == REQ_CHAT && j->req.has_tools && g->parser->turn_complete(g->parser_st, g)) {
        g->finish = "tool_calls";
        return true;
    }
    return false;
}



/* L118 tombstone (2026-08-26): the classic per-slot decode loop
 * (gen_step_decode) lived here. Deleted: every GEN_DECODE slot is serviced
 * by the batched quanta at n >= 1 (worker_spec_batched_quantum /
 * worker_batched_decode_quantum / worker_mixed_batch_quantum), a solo
 * session being a batch of one. Parity evidence: rows/L118.md (solo 6x6
 * A/B unified median 19.32 vs classic 18.95 t/s; decode-floor/sse/coherence
 * green on the unified lane; disconnect + continued-store + accounting
 * ported). The engine-level classic API (pulsar_session_eval /
 * generate_speculative) is unaffected — pulsar-bench/eval/agent and the
 * gate fixtures still drive it. */



/* Post-decode epilogue: tool repair/recovery, final parse, protocol live
 * state, checkpoints, the final response, and logging. Recovery paths loop
 * back to GEN_DECODE_INIT (the old goto decode_again). */
void server::gen_step_finish(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    job *j = g->j;

    if (g_stop_requested && strcmp(g->finish, "error") != 0) {
        g->finish = "error";
        snprintf(g->err, sizeof(g->err), "shutdown requested");
    }

    /* L272 P3: the family's final reading of the turn -- its parse, the shared broken-call rule (L284 P4),
     * its model-visible retry (the generation loops to a fresh attempt), the stream's ids onto the
     * calls, its memory, the finish label. */
    server_turn turn;
    memset(&turn, 0, sizeof turn);
    turn.finish = g->finish;
    const bool stream_ok = g->parser->finish(g->parser_st, s, sl, g, &turn);
    if (turn.retry) {
        g->recovery_attempted = true;
        g->completion_total += g->completion;
        buf_free(&g->text);
        free(turn.content);
        free(turn.reasoning);
        tool_calls_free(&turn.calls);
        g->phase = GEN_DECODE_INIT; /* the old goto decode_again */
        return;
    }

    if (g->completion > g->last_decode_log_completion) {
        bool tool_open = false, tool_closed = false;
        if (g->parser->tool_progress) g->parser->tool_progress(g->parser_st, &tool_open, &tool_closed);
        log_decode_progress(j->req.kind, g->prompt_tokens, g->completion,
                            g->responses_protocol,
                            j->req.has_tools,
                            g->thinking.inside,
                            tool_open,
                            tool_closed,
                            g->decode_t0,
                            &g->last_decode_log_t,
                            &g->last_decode_log_completion);
    }

    if (j->req.stream && !g->structured_stream && g->text.len > g->plain_stream_pos) {
        char *tail = xstrndup(g->text.ptr + g->plain_stream_pos, g->text.len - g->plain_stream_pos);
        if (!sse_chunk(j->fd, &j->req, g->id, tail, NULL)) g->finish = "error";
        free(tail);
    }

    tool_calls parsed_calls = turn.calls;
    char *parsed_content = turn.content;
    char *parsed_reasoning = turn.reasoning;
    const char *final_finish = !strcmp(g->finish, "error") ? g->finish : turn.finish;
    if (j->req.kind == REQ_CHAT && !parsed_calls.len && j->req.api == API_RESPONSES) s->responses_live_clear(sl);
    log_tool_calls_summary(g->ctx_span, &parsed_calls,
                           g->responses_protocol);

    /* Populate the additive per-response "timings" block from counters the
     * worker already kept. Pure metadata assembled once at finish, off any hot
     * path; the emitter derives the rates. */
    {
        const double finish_t = server_now_sec();
        req_timings *t = &j->req.timings;
        t->ttft_s = g->first_token_t > 0.0 ? g->first_token_t - g->t0 : 0.0;
        t->prefill_s = g->decode_t0 > g->t0 ? g->decode_t0 - g->t0 : 0.0;
        t->decode_s = finish_t > g->decode_t0 ? finish_t - g->decode_t0 : 0.0;
        t->prompt_n = g->prompt_tokens;
        t->cached_n = j->req.cache_read_tokens;
        t->decode_n = g->completion_total + g->completion;
        /* L119: request-scoped accumulators from the spec-batched lane — NOT
         * shared-session counter deltas, which bank save/restore rolls and
         * concurrent banks mix (the response reported impossible values). */
        t->spec_gen = g->req_spec_gen;
        t->spec_accepted = g->req_spec_accepted;
        t->spec_draft = g->req_spec_draft;
        t->spec_drafts = g->req_spec_rounds;
        t->spec_active = t->spec_gen > 0; /* the spec lane ran this request */
        t->valid = true;
        /* Same numbers the response body already carries, folded into the
         * /metrics histograms so TTFT and per-token latency are observable
         * without scraping every response. */
        s->observe_request_timings(t, finish_t - g->t0);
    }

    bool tool_open = false, tool_closed = false;
    if (g->parser->tool_progress) g->parser->tool_progress(g->parser_st, &tool_open, &tool_closed);
    s->trace_finish(g->trace_id, &j->req, final_finish, g->completion,
                 tool_open, tool_closed,
                 parsed_content ? parsed_content : (g->text.ptr ? g->text.ptr : ""),
                 parsed_reasoning, &parsed_calls, server_now_sec() - g->t0);

    if (j->req.api == API_RESPONSES) {
        /* the visible suffix is the family's sampled-turn render; a family without one keeps no
         * visible memory (its renderer refuses a request that needs the live state) */
        if (j->req.family->assistant_turn_sampled && strcmp(final_finish, "error") && strcmp(final_finish, "length")) {
            /* Store the post-turn visible transcript plus the live token
             * frontier.  The next Responses request may replay only this
             * visible surface, while the real session also contains hidden
             * reasoning and exact sampled tool-call bytes. */
            char *visible_suffix =
                build_responses_visible_assistant_suffix(&j->req,
                    parsed_content ? parsed_content : "",
                    parsed_reasoning,
                    &parsed_calls);
            buf visible = {0};
            buf_puts(&visible, j->req.prompt_text ? j->req.prompt_text : "");
            buf_puts(&visible, visible_suffix ? visible_suffix : "");
            s->responses_live_remember(sl, visible.ptr ? visible.ptr : "",
                                    parsed_calls.len ? &parsed_calls : NULL);
            buf_free(&visible);
            free(visible_suffix);
        } else {
            s->responses_live_clear(sl);
        }
    }
    if (j->req.api == API_ANTHROPIC) {
        /* a family without a tool-result tail keeps no live tool state (its renderer refuses a
         * continuation that would need it), so nothing is remembered for it */
        if (parsed_calls.len && j->req.family->tool_result_tail && strcmp(final_finish, "error") &&
            strcmp(final_finish, "length"))
        {
            s->anthropic_live_remember(sl, &parsed_calls);
        } else {
            s->anthropic_live_clear(sl);
        }
    }

    if (!j->req.family->assistant_turn_sampled) {
        /* the family has no sampled-turn render, so no canonical rewrite: the next request resolves
         * by exact token / rendered-text prefix only (L264 retired the binding there was to clear) */
    } else if (j->req.kind == REQ_CHAT && parsed_calls.len &&
        j->req.api != API_RESPONSES &&
        pulsar_think_mode_enabled(j->req.think_mode) &&
        !j->req.force_tool_call)
    {
        /* Tool call with thinking on: nothing to remember or rewrite (L264).  A
         * client that replays the reasoning byte-matches the live KV through it;
         * one that strips it resumes from this turn's prompt-end grid checkpoint.
         * Either way the next request's sync finds its own resume point, so the
         * canonical rewrite below -- which would re-render hidden reasoning --
         * stays out of this turn. */
    } else if (j->req.kind == REQ_CHAT && parsed_calls.len &&
        j->req.api != API_RESPONSES &&
        s->should_canonicalize_tool_checkpoint(&parsed_calls))
    {
        /* Chat/completions has no protocol object that binds the next request
         * to this live KV state.  Canonicalize only the fallback tool-call
         * path where we lack exact sampled DSML replay; when raw DSML is known,
         * replaying those bytes keeps future prompts aligned without rebuilding
         * hidden reasoning.  Responses deliberately skips this path because its
         * previous_response_id contract binds the next turn to live state. */
        s->canonicalize_tool_checkpoint(sl, j, g->ctx_span, g->trace_id,
                                     parsed_content ? parsed_content : "",
                                     parsed_reasoning, &parsed_calls);
    }

    if (!strcmp(final_finish, "error")) {
        /* Internal generation failure (decode / bank-restore / etc.). Do NOT
         * put finish_reason:"error" on the wire -- it is not a valid enum
         * value and strict SDKs reject the chunk -- nor return HTTP 200 with
         * empty content (which reads as a blank successful answer). Streaming:
         * the 200 event-stream headers are already sent, so emit an SSE error
         * event, matching the pre-generation failure path. Non-streaming: a
         * real 500 with the protocol's error envelope. */
        const char *emsg = g->err[0] ? g->err : "internal generation error";
        if (j->req.stream) {
            sse_error_event(j->fd, &j->req, emsg);
        } else if (j->req.api == API_ANTHROPIC) {
            http_error_anthropic(j->fd, 500, emsg);
        } else {
            http_error(j->fd, 500, emsg);
        }
    } else if (j->req.stream) {
        bool response_ok = true;
        if (g->sink.text) {
            /* L267: the family's last text into the sink -- DeepSeek's walk
             * flushes what it held back; a Qwen turn's parser already did -- then
             * the protocol's finish, which sends any calls not yet streamed. */
            const int completion = g->completion_total + g->completion;
            response_ok = stream_ok && g->parser->feed(g->parser_st, s, g, g->text.len, true);
            if (response_ok && j->req.api == API_ANTHROPIC) {
                response_ok = anthropic_sse_finish(&g->sink, &parsed_calls, final_finish, g->stop_sequence, completion);
            } else if (response_ok && j->req.api == API_RESPONSES) {
                /* A malformed tool call the final reading turned back into text:
                 * the stream stopped at its marker, so the rest goes out now. */
                response_ok = responses_sse_finish(j->fd, &j->req, &g->responses_live,
                                                   turn.tail, turn.tail_len, &parsed_calls,
                                                   final_finish, g->prompt_tokens, completion,
                                                   g->responses_created_at);
            } else if (response_ok) {
                response_ok = openai_sse_finish(&g->sink, &parsed_calls, final_finish, g->prompt_tokens, completion);
            }
        } else {
            response_ok = sse_chunk(j->fd, &j->req, g->id, NULL, final_finish) &&
                          sse_done(j->fd, &j->req, g->id, g->prompt_tokens,
                                   g->completion_total + g->completion);
        }
        if (!response_ok) {
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: %s ctx=%s%s%s final stream failed",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       g->ctx_span,
                       g->req_flags[0] ? " " : "",
                       g->req_flags);
        }
    } else if (j->req.api == API_ANTHROPIC) {
        anthropic_final_response(j->fd, &j->req, g->id,
                                 parsed_content ? parsed_content : (g->text.ptr ? g->text.ptr : ""),
                                 parsed_reasoning,
                                 &parsed_calls, final_finish, g->stop_sequence,
                                 g->prompt_tokens,
                                 g->completion_total + g->completion);
    } else if (j->req.api == API_RESPONSES) {
        responses_final_response(j->fd, &j->req, g->id,
                                 parsed_content ? parsed_content : (g->text.ptr ? g->text.ptr : ""),
                                 parsed_reasoning,
                                 &parsed_calls, final_finish,
                                 g->prompt_tokens,
                                 g->completion_total + g->completion);
    } else {
        final_response(j->fd, &j->req, g->id,
                       parsed_content ? parsed_content : (g->text.ptr ? g->text.ptr : ""),
                       parsed_reasoning,
                       &parsed_calls, final_finish,
                       g->prompt_tokens,
                       g->completion_total + g->completion,
                       &g->logprobs);
    }
    if (j->req.kind == REQ_CHAT && j->req.has_tools) {
        char flags[80];
        log_flags(flags, sizeof(flags),
                  g->responses_protocol,
                  true,
                  g->thinking.inside,
                  tool_open,
                  tool_closed);
        if (!strcmp(final_finish, "error") && g->err[0]) {
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: chat ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs",
                       g->ctx_span,
                       g->completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       g->err,
                       server_now_sec() - g->t0);
        } else {
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: chat ctx=%s gen=%d%s%s finish=%s %.3fs",
                       g->ctx_span,
                       g->completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       server_now_sec() - g->t0);
        }
    } else {
        char flags[80];
        log_flags(flags, sizeof(flags),
                  g->responses_protocol,
                  j->req.has_tools,
                  g->thinking.inside,
                  false,
                  false);
        if (!strcmp(final_finish, "error") && g->err[0]) {
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: %s ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       g->ctx_span,
                       g->completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       g->err,
                       server_now_sec() - g->t0);
        } else {
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: %s ctx=%s gen=%d%s%s finish=%s %.3fs",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       g->ctx_span,
                       g->completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       server_now_sec() - g->t0);
        }
    }
    free(parsed_content);
    free(parsed_reasoning);
    tool_calls_free(&parsed_calls);
    g->phase = GEN_DONE;
}



/* ---- state-machine driver: bind, step, unbind ---- */

void server::gen_state_free(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    if (!g) return;
    (void)s;
    /* Callback safety: no gen_state pointer may remain installed anywhere. */
    pulsar_session_set_cancel(s->sess, NULL, NULL);
    pulsar_session_set_progress(s->sess, NULL, NULL);
    pulsar_session_set_display_progress(s->sess, NULL, NULL);
    if (g->parser_st) g->parser->destroy(g->parser_st);
    responses_stream_free(&g->responses_live);
    buf_free(&g->text);
    logprob_ledger_free(&g->logprobs);
    pulsar_tokens_free(&g->effective_prompt);
    pulsar_tokens_free(&g->cold_prefix);
    pulsar_tokens_free(&g->batch_pending);
    free(g->disk_cache_path);
    free(g->stop_sequence);
    slot_writer_free(&g->writer);
    free(g);
    sl->gen = NULL;
}



/* Bind a dequeued job to the slot and resolve its prompt (the first quantum). */
void server::generate_job_begin(session_slot *sl, job *j) {
    auto *s = this;
    gen_state *g = (gen_state *)server_xmalloc(sizeof(*g));
    memset(g, 0, sizeof(*g));
    g->j = j;
    g->prompt_for_sync = &j->req.prompt;
    g->finish = "length";
    sl->gen = g;
    sl->active_job = j;
    sl->state = SLOT_PREFILLING;
    /* All client writes for this job (worker thread only) become non-blocking
     * and deferred; drained in generate_job_end. */
    slot_writer_init(&g->writer, j->fd);
    slot_writer_install(&g->writer);
    s->gen_begin(sl);
}



/* Advance the job by one quantum. */
void server::generate_job_step(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    /* Tier-2: install this slot's bank before any engine work this quantum.
     * After another slot (or a fresh bind) was serviced in between, the pool's
     * live bank may be someone else's; switch back to ours. No-op in classic
     * mode / when already live. Finding 1: fail the request on a failed spill
     * restore rather than run engine work against the wrong bank's KV. */
    if (!s->bank_switch(sl->bank)) {
        /* The FIRST error wins: a slot already finishing on an error (a
         * refused forward, a failed tensor-parallel step -- after which every
         * bank switch refuses too) must report that cause, not this echo of
         * it.  Overwriting it here hid the pair's real refusal behind "bank 0
         * state restore failed" (L241). */
        if (!g->err[0])
            snprintf(g->err, sizeof g->err,
                     "bank %u state restore failed (evicted KV unrecoverable)", (unsigned)sl->bank);
        g->finish = "error";
        if (g->phase == GEN_FINISH) {
            /* Already finishing and the restore STILL fails: run the finish
             * step anyway — the error epilogue answers the client without
             * engine work. Returning here re-steps the slot forever (finish
             * never runs, GEN_DONE never set, slot never freed, client never
             * answered). */
            s->gen_step_finish(sl);
            return;
        }
        g->phase = GEN_FINISH;
        return;
    }
    /* Tier-2: a slot leaving the batched lane (now at GEN_FINISH) has its bank
     * installed above (bank_state_restore cleared the multiseq poison and
     * installed the driver-maintained device counters); catch the host
     * checkpoint up to the tokens multiseq committed, so gen_step_finish's
     * store/continuation see the true frontier. */
    if (g->batch_active) {
        if (g->batch_pending.len > 0)
            pulsar_session_note_committed_tokens(s->sess, g->batch_pending.v,
                                              g->batch_pending.len);
        pulsar_tokens_free(&g->batch_pending);
        g->batch_active = false;
        g->batch_feed_valid = false;
    }
    /* The installed slot writer is worker-thread-local and shared across
     * slots; re-install this slot's writer so send_all() routes through the
     * right deferral queue after another slot (or a fresh bind) was serviced
     * in between. */
    slot_writer_install(&g->writer);
    /* Push any bytes a slow client deferred before spending GPU time. */
    slot_writer_flush(&g->writer);
    switch (g->phase) {
    case GEN_PREFILL_COLD:
    case GEN_PREFILL_MAIN:
        sl->state = SLOT_PREFILLING;
        s->gen_step_prefill(sl);
        break;
    case GEN_DECODE_INIT:
        sl->state = SLOT_DECODING;
        s->gen_decode_init(sl);
        /* L118: GEN_DECODE is serviced exclusively by the batched quanta —
         * the worker's next pass gathers this slot into the batch (a solo
         * session is a batch of one). The classic per-slot decode loop and
         * its first-quantum fall-through are deleted; plan 118. */
        break;
    case GEN_DECODE:
        sl->state = SLOT_DECODING;
        break;
    case GEN_FINISH:
        s->gen_step_finish(sl);
        break;
    case GEN_DONE:
        break;
    }
}



/* Unbind: drain deferred client bytes, free the resumable state. */
void server::generate_job_end(session_slot *sl) {
    auto *s = this;
    if (sl->gen) {
        /* A writer that failed EARLIER already ended the job (lane_should_abandon
         * / the emit path) and logged why; a drain that fails HERE is the one
         * failure nothing else reports -- the final response never reached the
         * client (L190 C3). */
        slot_writer *w = &sl->gen->writer;
        const bool failed_before = w->failed;
        if (!slot_writer_drain(w) && !failed_before)
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: ctx=%s%s%s client stream failed at drain: "
                       "%zu final bytes undelivered",
                       sl->gen->ctx_span,
                       sl->gen->req_flags[0] ? " " : "", sl->gen->req_flags,
                       w->pending.len - w->off);
    }
    s->gen_state_free(sl);
    sl->active_job = NULL;
    sl->state = SLOT_IDLE;
    sl->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
}


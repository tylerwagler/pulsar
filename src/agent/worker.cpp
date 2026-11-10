/** @file
 *  pulsar-agent: the model worker thread (prefill, generation, tool turns). */
#include "pulsar_agent_internal.h"
#include "out_parse_kit.h"



/* ============================================================================
 * Model Worker Thread
 * ============================================================================
 */

/* DSML structure is a machine-readable grammar, so once the model has clearly
 * started a tool stanza we decode grammar bytes greedily.  Parameter values are
 * different: they can be shell commands, code, file contents, or edit bodies,
 * and should keep the configured sampling behavior.  The only exception is a
 * parameter closing tag once it is clearly DSML syntax, not ordinary text such
 * as HTML/XML/code containing "</".
 *
 * This helper is intentionally derived only from the current streaming parser
 * state.  The state object is local to one assistant round, so malformed output,
 * EOS, Ctrl+C, or the next turn cannot accidentally leave sampling greedy. */
/// @brief Whether the next token should be drawn greedily (see above).
/// @param sr the round's stream renderer; NULL = no
/// @return true inside DSML structure, a held DSML-shaped opening, or a possible parameter close
static bool agent_stream_wants_greedy_sampling(const agent_stream_renderer *sr) {
    if (sr && sr->family) return agent_family_parser_wants_greedy(sr->family);   /* L365 */
    if (!sr || !sr->parser) return false;
    if (sr->parser->state == AGENT_DSML_ERROR ||
        sr->parser->state == AGENT_DSML_DONE)
        return false;

    /* A possible opening marker is being held back by the start detector.  A
     * single '<' is too common in prose/code to justify forcing argmax; after
     * the second byte, the buffered prefix still matching here is specifically
     * DSML-shaped ("<｜..." or the tolerated "<D..." typo). */
    if (sr->dsml_start_len > 1) return true;
    if (!sr->dsml_active) return false;

    if (sr->parser->state == AGENT_DSML_STRUCTURAL)
        return true;
    if (sr->parser->state != AGENT_DSML_PARAM_VALUE)
        return false;

    return sr->parser->param_close_prefix;
}



/** @brief Draw the next token from the session: the configured temperature / top-p / min-p, or greedy.
 *  @param w      the worker (its session)
 *  @param cfg    the configuration (sampling knobs)
 *  @param greedy draw the argmax instead
 *  @param rng    the request's sampler stream, stepped
 *  @return the token, or PULSAR_SAMPLE_REFUSED */
static int worker_sample_with_mode(agent_worker *w, const agent_config *cfg,
                                   bool greedy, uint64_t *rng) {
    return pulsar_session_sample(w->session,
                              greedy ? 0.0f : cfg->gen.temperature,
                              0,
                              greedy ? 1.0f : cfg->gen.top_p,
                              greedy ? 0.0f : cfg->gen.min_p,
                              rng);
}



/** @brief Publish whether sampling is currently greedy (status.greedy_sampling), waking the UI on a change.
 *  @param w      the worker; takes w->mu
 *  @param greedy the new state */
static void worker_set_greedy_sampling(agent_worker *w, bool greedy) {
    pthread_mutex_lock(&w->mu);
    if (w->status.greedy_sampling != greedy) {
        w->status.greedy_sampling = greedy;
        agent_wake_locked(w);
    }
    pthread_mutex_unlock(&w->mu);
}



/** @brief End a turn whose compaction did not complete: idle after an interrupt, else the error.
 *  @param w   the worker
 *  @param err the compaction's reason ("interrupted", or why it failed; empty = a generic message)
 *  @return worker_run_turn's status: 0 after an interrupt, 1 on an error */
static int worker_compaction_stopped(agent_worker *w, const char *err) {
    if (agent_err_is_interrupted(err)) {
        worker_clear_interrupt(w);
        agent_set_status(w, AGENT_WORKER_IDLE);
        return 0;
    }
    agent_set_error(w, err[0] ? err : "context compaction failed");
    return 1;
}



/** @brief L373: make the turn being built fit the context before it is rendered, when it carries tool results:
 *  compact (the pending turn set aside, so the summary request does not carry it), and if the results still do not
 *  fit, answer each call with a short error instead.  Tool results are in the pending turn from the moment their
 *  calls ran, so a compaction that fails or is interrupted leaves them there for the next turn -- never dropped,
 *  and (Gemma) the open model turn still gets its responses.
 *  @param w       the worker
 *  @param err     receives why the turn cannot go on ("interrupted", a compaction failure, context full)
 *  @param err_len its capacity
 *  @return true when the turn may be rendered */
static bool worker_fit_tool_turn(agent_worker *w, char *err, size_t err_len) {
    bool has_tool = false;
    for (int i = 0; i < w->turn_len; i++) has_tool = has_tool || !strcmp(w->turn[i].role, "tool");
    int projected = 0;
    if (!has_tool || agent_turn_fits_context(w, AGENT_TOOL_RESULT_RESERVE_TOKENS, &projected)) return true;

    std::vector<pulsar_chat_message> pending(w->turn, w->turn + w->turn_len);
    w->turn_len = 0;
    const bool compacted = agent_worker_compact(w, "tool result would exceed context", err, err_len);
    for (const pulsar_chat_message &m : pending) agent_turn_push(w, m);   /* after a bash update the compaction added */
    if (!compacted) return false;
    if (agent_turn_fits_context(w, AGENT_TOOL_RESULT_RESERVE_TOKENS, &projected)) return true;

    char msg[256];
    snprintf(msg, sizeof(msg),
             "Tool error: tool result still does not fit after context compaction "
             "(projected_prompt=%d tokens, ctx=%d, reserve=%d). "
             "Retry with a smaller read/search/bash output.\n",
             projected, w->cfg->gen.ctx_size, AGENT_TOOL_RESULT_RESERVE_TOKENS);
    for (int i = 0; i < w->turn_len; i++) {
        if (strcmp(w->turn[i].role, "tool")) continue;
        free((char *)w->turn[i].content);
        w->turn[i].content = xstrdup(msg);
    }
    if (agent_turn_fits_context(w, 16, NULL)) return true;
    agent_turn_clear(w);   /* nothing of it can be rendered: said, not carried into every later turn */
    snprintf(err, err_len, "context full after compaction; the pending turn was dropped");
    return false;
}



/* Run one user turn until the assistant stops or returns a tool call.  Tool
 * results are appended to the transcript and the loop continues, which gives
 * the model native DSML tool iteration without a client/server protocol. */
/// @brief Run one user turn, tool iterations included (see above).  Worker thread.
/// @param w         the worker
/// @param user_text the user's message
/// @return 0 (done, or interrupted), or 1 on an error (set with agent_set_error)
static int worker_run_turn(agent_worker *w, const char *user_text) {
    agent_config *cfg = w->cfg;
    pulsar_think_mode think_mode = cfg->gen.think_mode;
    pthread_mutex_lock(&w->mu);
    w->interrupt = false;
    w->status.error[0] = '\0';
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    char compact_err[160] = {0};
    if (!agent_worker_compact_if_needed(w, "soft limit before user turn",
                                        compact_err, sizeof(compact_err)))
        return worker_compaction_stopped(w, compact_err);
    agent_worker_maybe_append_datetime_context(w);
    agent_trace_text(w, "user", user_text ? user_text : "",
                     user_text ? strlen(user_text) : 0);
    if (!w->session_title) {
        w->session_title = agent_session_title_from_prompt(user_text, 0);
        w->session_created_at = (uint64_t)time(NULL);
        agent_session_identity_sha(w->session_title, w->session_created_at,
                                   w->session_sha);
    }
    agent_worker_close_open_turn(w, &w->transcript);   /* L365: a turn left open by an earlier failure */
    agent_turn_add(w, "user", user_text, false);

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    pthread_mutex_lock(&w->mu);
    w->user_activity = true;
    w->session_dirty = true;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    /* A user turn may contain any number of assistant/tool/assistant rounds.
     * Coding agents naturally perform long read/edit/test loops, so there is
     * deliberately no artificial "too many tool calls" ceiling here: context
     * pressure, compaction, user Ctrl+C, and the model's final answer are the
     * real stopping conditions.  The transcript is the single source of truth:
     * after a DSML stanza completes we terminate that assistant message, add
     * the tool result to the next turn as a tool message, then ask the model to
     * continue -- the turn rendered whole by the family's chat front. */
    for (int tool_round = 0; ; tool_round++) {
        if (tool_round > 0 &&
            !agent_worker_compact_if_needed(w, "soft limit before tool continuation",
                                            compact_err, sizeof(compact_err)))
            return worker_compaction_stopped(w, compact_err);
        agent_bash_jobs_poll(w);   /* L373: once a round, whether or not the model asks */
        agent_worker_maybe_append_system_prompt_reminder(w);
        if (!worker_fit_tool_turn(w, compact_err, sizeof(compact_err)))
            return worker_compaction_stopped(w, compact_err);
        if (!agent_turn_flush(w, think_mode)) {   /* L373: never sample after a turn that is not there */
            agent_set_error(w, "the tokenizer refused the turn (its reason is on stderr); nothing was appended");
            return 1;
        }

        const pulsar_tokens *prompt_for_sync = &w->transcript;
        const TokPos old_pos = pulsar_session_pos(w->session);
        const TokCount common = pulsar_session_common_prefix(w->session, &w->transcript);
        int cached = TokPos::after(common) == old_pos && TokPos{w->transcript.len} >= old_pos ? common.raw() : 0;

        int suffix = prompt_for_sync->len - cached;
        agent_trace(w, "prefill tool_round=%d transcript=%d prompt=%d cached=%d suffix=%d think=%s",
                    tool_round, w->transcript.len, prompt_for_sync->len,
                    cached, suffix, pulsar_think_mode_name(think_mode));
        agent_trace_tokens(w, "prefill_suffix", prompt_for_sync, cached);

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
        w->status.greedy_sampling = false;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);

        char err[160];
        pulsar_session_set_progress(w->session, worker_progress_cb, w);
        pulsar_session_set_display_progress(w->session, worker_progress_cb, w);
        pulsar_session_set_cancel(w->session, worker_cancel_session_cb, w);
        int sync_rc = pulsar_session_sync(w->session, prompt_for_sync, err, sizeof(err));
        pulsar_session_set_cancel(w->session, NULL, NULL);
        pulsar_session_set_progress(w->session, NULL, NULL);
        pulsar_session_set_display_progress(w->session, NULL, NULL);
        if (sync_rc == PULSAR_SESSION_SYNC_INTERRUPTED) {
            agent_publish_system_status(
                w, "Model reading interrupted; the model may only be aware of the prefix processed so far.");
            pulsar_chat_end_assistant(w->engine, &w->transcript);
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
        if (sync_rc != 0) {
            agent_set_error(w, err);
            return 1;
        }

        int max_tokens = cfg->gen.n_predict;
        int room = (pulsar_session_ctx(w->session) - pulsar_session_pos(w->session).before()).raw();
        if (room <= 1) max_tokens = 0;
        else if (max_tokens > room - 1) max_tokens = room - 1;

        bool use_color = isatty(STDOUT_FILENO) != 0;
        /* L365: where the prompt leaves the turn -- the generation prompt's thinking, or (Gemma) a thought channel
         * reopened after the tool responses */
        const bool reasoning_open = pulsar_chat_opens_reasoning(w->engine, &w->transcript, think_mode);
        agent_token_renderer renderer = {
            .engine = w->engine,
            .worker = w,
            .format_markdown = use_color,
            .think = {.in_think = reasoning_open},
            .use_color = use_color,
            .last_output_newline = true,
        };
        agent_dsml_parser dsml = {.state = AGENT_DSML_SEARCH};
        agent_stream_renderer stream = {
            .renderer = &renderer,
            .parser = &dsml,
        };
        /* L365: every family but DeepSeek calls tools in its own syntax, read by its output parser; the guard
         * frees it on every exit from the round */
        struct family_guard {
            out_parser *p;
            ~family_guard() { agent_family_parser_free(p); }
        } family = {NULL};
        if (!agent_worker_dsml(w)) {
            family.p = agent_family_parser_new(w->engine, reasoning_open, err, sizeof(err));
            if (!family.p) {
                agent_set_error(w, err);
                return 1;
            }
            stream.family = family.p;
        }
        agent_edit_upto_forcer upto_forcer = {0};
        bool got_tool = false;
        bool malformed_tool = false;
        bool early_tool_error = false;
        int generated = 0;
        double t0 = agent_now_sec();

        pthread_mutex_lock(&w->mu);
        w->status.state = AGENT_WORKER_GENERATING;
        w->status.prefill_tps = 0.0;
        w->status.greedy_sampling = false;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);

        bool status_greedy_sampling = false;
        while (generated < max_tokens && !worker_should_interrupt(w)) {
            bool greedy_sampling = agent_stream_wants_greedy_sampling(&stream);
            if (greedy_sampling != status_greedy_sampling) {
                worker_set_greedy_sampling(w, greedy_sampling);
                status_greedy_sampling = greedy_sampling;
            }
            int token = worker_sample_with_mode(w, cfg, greedy_sampling, &rng);
            if (pulsar_token_is_stop(w->engine, token)) break;

            size_t text_len = 0;
            char *text = pulsar_token_text(w->engine, token, &text_len);
            if (agent_edit_upto_forcer_should_replace(&upto_forcer, &dsml,
                                                       text, text_len))
            {
                agent_trace(w, "edit old auto-upto replaced token=%d text=%.*s",
                            token, (int)(text_len > 80 ? 80 : text_len), text);
                free(text);
                if (worker_force_generated_text(w, "[upto]\n", max_tokens,
                                                &generated, t0, &stream,
                                                err, sizeof(err)) != 0) {
                    agent_dsml_parser_free(&dsml);
                    agent_set_error(w, err);
                    return 1;
                }
            } else {
                free(text);
                if (worker_accept_generated_token(w, token, &generated, t0,
                                                  &stream, err, sizeof(err)) != 0) {
                    agent_dsml_parser_free(&dsml);
                    agent_set_error(w, err);
                    return 1;
                }
            }

            greedy_sampling = agent_stream_wants_greedy_sampling(&stream);
            if (greedy_sampling != status_greedy_sampling) {
                worker_set_greedy_sampling(w, greedy_sampling);
                status_greedy_sampling = greedy_sampling;
            }

            if (dsml.state == AGENT_DSML_DONE) {
                got_tool = true;
                break;
            }
            if (stream.tool_preflight_error) {
                early_tool_error = true;
                break;
            }
            if (dsml.state == AGENT_DSML_ERROR) {
                malformed_tool = true;
                break;
            }
            if (stream.dsml_in_think) {
                malformed_tool = true;
                break;
            }
        }

        bool interrupted = worker_should_interrupt(w);
        agent_stream_text(&stream, NULL, 0, true);
        renderer_finish(&renderer);
        worker_set_greedy_sampling(w, false);
        if (interrupted) {
            pulsar_chat_end_assistant(w->engine, &w->transcript);
            agent_dsml_parser_free(&dsml);
            agent_publish_system_status(w, "Stopped by user");
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
        if (stream.dsml_in_think) {
            got_tool = false;
            malformed_tool = true;
            early_tool_error = false;
            snprintf(dsml.error, sizeof(dsml.error),
                     "tool calling is not allowed inside <think></think>");
        } else if (!malformed_tool && dsml.state == AGENT_DSML_ERROR) {
            malformed_tool = true;
        } else if (!got_tool && !malformed_tool && !early_tool_error &&
                   !interrupted &&
                   (dsml.state == AGENT_DSML_STRUCTURAL ||
                    dsml.state == AGENT_DSML_PARAM_VALUE))
        {
            malformed_tool = true;
            snprintf(dsml.error, sizeof(dsml.error),
                     "incomplete DSML tool call");
        }

        if (family.p) {   /* the parser has read the whole turn (agent_stream_text's finish above) */
            if (stream.family_error[0]) {
                malformed_tool = true;
                snprintf(dsml.error, sizeof(dsml.error), "%s", stream.family_error);
            } else if (!family.p->calls().empty()) {
                if (agent_family_calls_take(family.p, &dsml.calls, dsml.error, sizeof(dsml.error))) got_tool = true;
                else malformed_tool = true;
            }
        }

        /* L365: where calls continue the turn (Gemma), a turn that made calls stays open for their responses
         * (agent_worker_turn_open reads that off the transcript) */
        if (!(family.p && (got_tool || malformed_tool) && pulsar_chat_calls_continue_turn(w->engine)))
            pulsar_chat_end_assistant(w->engine, &w->transcript);

        if (!got_tool && !malformed_tool && !early_tool_error) {
            agent_dsml_parser_free(&dsml);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }

        /* The results join the turn being built as soon as their calls have run: whatever ends this turn early
         * (a compaction that fails or is interrupted, worker_fit_tool_turn at the top of the next round), they
         * are rendered with the next one. */
        if (early_tool_error) {
            agent_buf b = {0};
            agent_buf_puts(&b, "Tool error: ");
            agent_buf_puts(&b, stream.tool_preflight_error_msg[0] ?
                           stream.tool_preflight_error_msg :
                           "edit old selector failed before new was generated");
            agent_buf_puts(&b, "\n");
            agent_turn_add_tool(w, NULL, b.ptr);
            free(b.ptr);
        } else if (malformed_tool) {
            agent_buf b = {0};
            agent_buf_puts(&b, family.p ? "Tool error: invalid tool call: " : "Tool error: invalid DSML tool call: ");
            agent_buf_puts(&b, dsml.error[0] ? dsml.error : "parse error");
            agent_buf_puts(&b, "\n");
            if (!family.p) agent_buf_puts(&b, agent_dsml_syntax_reminder);
            agent_turn_add_tool(w, family.p && dsml.calls.len > 0 ? dsml.calls.v[0].name : NULL, b.ptr);
            free(b.ptr);
        } else if (family.p) {
            /* L365: one result per call, each its own tool message naming its function (the templates answer calls
             * one by one) */
            for (int i = 0; i < dsml.calls.len; i++) {
                char *res = agent_execute_tool_call_one(w, &dsml.calls.v[i]);
                agent_turn_add_tool(w, dsml.calls.v[i].name, res);
                free(res);
            }
        } else {
            char *res = agent_execute_tool_calls(w, &dsml.calls);
            agent_turn_add_tool(w, NULL, res);
            free(res);
        }
        agent_dsml_parser_free(&dsml);

        char *queued_user = worker_request_queued_user_drain(w);
        if (queued_user && queued_user[0]) {
            agent_trace_text(w, "queued_user", queued_user, strlen(queued_user));
            agent_turn_add(w, "user", queued_user, false);
            pthread_mutex_lock(&w->mu);
            w->user_activity = true;
            w->session_dirty = true;
            agent_wake_locked(w);
            pthread_mutex_unlock(&w->mu);
        }
        free(queued_user);
    }
}



void worker_request_save(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->save_requested = true;
    pthread_cond_signal(&w->cond);
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}



void worker_request_compact(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->compact_requested = true;
    pthread_cond_signal(&w->cond);
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}



/** @brief Take (read and clear, under w->mu) the UI's save request.
 *  @param w the worker
 *  @return true when a save was requested */
static bool worker_take_save_requested(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool requested = w->save_requested;
    w->save_requested = false;
    pthread_mutex_unlock(&w->mu);
    return requested;
}



/** @brief Take (read and clear, under w->mu) the UI's compaction request.
 *  @param w the worker
 *  @return true when a compaction was requested */
static bool worker_take_compact_requested(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool requested = w->compact_requested;
    w->compact_requested = false;
    pthread_mutex_unlock(&w->mu);
    return requested;
}



/** @brief Run a requested session save on the worker thread and report it; nothing when none was requested.
 *  @param w the worker */
static void worker_run_deferred_save(agent_worker *w) {
    if (!worker_take_save_requested(w)) return;
    agent_set_status(w, AGENT_WORKER_SAVING);
    char err[160] = {0};
    char sha[41];
    int tokens = 0;
    if (agent_worker_save_session_now(w, sha, &tokens, err, sizeof(err)))
        agent_publishf(w, "\nsaved session %.8s (%d tokens)\n", sha, tokens);
    else
        agent_publishf(w, "\nsave failed: %s\n", err[0] ? err : "unknown error");
    agent_set_status(w, AGENT_WORKER_IDLE);
}



/** @brief Run a requested compaction on the worker thread and report it; nothing when none was requested.
 *  @param w the worker */
static void worker_run_deferred_compact(agent_worker *w) {
    if (!worker_take_compact_requested(w)) return;
    if (!agent_worker_has_user_session(w)) {
        agent_publishf(w, "\ncompact skipped: nothing to compact\n");
        return;
    }

    int before = w->transcript.len;
    char err[160] = {0};
    if (agent_worker_compact(w, "user requested compaction", err, sizeof(err))) {
        if (w->transcript.len != before) {
            pthread_mutex_lock(&w->mu);
            w->session_dirty = true;
            agent_wake_locked(w);
            pthread_mutex_unlock(&w->mu);
        } else {
            agent_publishf(w, "\ncompact skipped: nothing to compact\n");
        }
        agent_set_status(w, AGENT_WORKER_IDLE);
    } else {
        if (agent_err_is_interrupted(err)) {
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return;
        }
        agent_set_error(w, err[0] ? err : "context compaction failed");
    }
}



/* Worker thread entry point.  The UI thread submits plain user text; this
 * thread owns all DS4 session mutation, tool execution, and compaction. */
void *worker_main(void *arg) {
    agent_worker *w = (agent_worker *)arg;
    agent_trace(w, "agent worker start ctx=%d backend=%s model=%s trace=%s",
                w->cfg->gen.ctx_size,
                pulsar_backend_name(w->cfg->engine.backend),
                w->cfg->engine.model_path ? w->cfg->engine.model_path : "",
                w->cfg->gen.trace_path ? w->cfg->gen.trace_path : "");
    char init_err[160] = {0};
    if (!agent_worker_reset_to_sysprompt(w, init_err, sizeof(init_err))) {
        agent_set_error(w, init_err[0] ? init_err : "failed to initialize system prompt");
    }
    agent_trace_tokens(w, "initial_system_prompt", &w->transcript, 0);
    pthread_mutex_lock(&w->mu);
    w->initialized = true;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    while (true) {
        pthread_mutex_lock(&w->mu);
        bool jobs_running = w->bash_jobs != NULL;   /* the list is this thread's: read without w->mu too */
        while (!w->stop && !w->cmd_text && !w->save_requested &&
               !w->compact_requested)
        {
            if (!jobs_running) {
                pthread_cond_wait(&w->cond, &w->mu);
                continue;
            }
            /* L373: an idle worker with a bash job running wakes once a second to drain it and apply its timeout */
            struct timespec until;
            clock_gettime(CLOCK_REALTIME, &until);
            until.tv_sec += 1;
            if (pthread_cond_timedwait(&w->cond, &w->mu, &until) != ETIMEDOUT) continue;
            pthread_mutex_unlock(&w->mu);   /* a job that ends takes w->mu itself */
            jobs_running = agent_bash_jobs_poll(w);
            pthread_mutex_lock(&w->mu);
        }
        if (w->stop) {
            pthread_mutex_unlock(&w->mu);
            break;
        }
        if (!w->cmd_text && w->save_requested) {
            pthread_mutex_unlock(&w->mu);
            worker_run_deferred_save(w);
            continue;
        }
        if (!w->cmd_text && w->compact_requested) {
            pthread_mutex_unlock(&w->mu);
            worker_run_deferred_compact(w);
            continue;
        }
        char *cmd = w->cmd_text;
        w->cmd_text = NULL;
        pthread_mutex_unlock(&w->mu);

        worker_run_turn(w, cmd);
        free(cmd);
        worker_run_deferred_compact(w);
        worker_run_deferred_save(w);
    }

    agent_set_status(w, AGENT_WORKER_STOPPED);
    return NULL;
}


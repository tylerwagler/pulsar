#include "pulsar_agent_internal.h"



/* ============================================================================
 * Session Listing, History Rendering, And Completion
 * ============================================================================
 */

static void agent_format_age(uint64_t when, char *buf, size_t len) {
    uint64_t now = (uint64_t)time(NULL);
    uint64_t age = when && now > when ? now - when : 0;
    if (age < 60) snprintf(buf, len, "%llus ago", (unsigned long long)age);
    else if (age < 3600) snprintf(buf, len, "%llum ago", (unsigned long long)(age / 60));
    else if (age < 86400) snprintf(buf, len, "%lluh ago", (unsigned long long)(age / 3600));
    else snprintf(buf, len, "%llud ago", (unsigned long long)(age / 86400));
}



static char *agent_session_title_from_span(const char *p, const char *end,
                                           size_t max_bytes,
                                           const char *empty_title) {
    bool limited = max_bytes != 0;
    if (limited && max_bytes < 4) max_bytes = 4;
    while (p < end && isspace((unsigned char)*p)) p++;
    while (end > p && isspace((unsigned char)end[-1])) end--;

    agent_buf b = {0};
    bool space = false;
    bool truncated = false;
    for (const char *s = p; s < end; s++) {
        unsigned char c = (unsigned char)*s;
        if (isspace(c)) {
            space = b.len != 0;
            continue;
        }
        if (space && (!limited || b.len + 4 < max_bytes)) {
            agent_buf_puts(&b, " ");
            space = false;
        }
        if (limited && b.len + 4 > max_bytes) {
            truncated = true;
            break;
        }
        agent_buf_append(&b, s, 1);
    }
    if (truncated) agent_buf_puts(&b, "...");
    if (!b.ptr || !b.len) {
        free(b.ptr);
        return xstrdup(empty_title);
    }
    return agent_buf_take(&b);
}



char *agent_session_title_from_prompt(const char *prompt,
                                             size_t max_bytes) {
    const char *p = prompt ? prompt : "";
    return agent_session_title_from_span(p, p + strlen(p), max_bytes,
                                         "(empty user prompt)");
}



/* Extract a human-readable title from the first user turn stored in the
 * rendered transcript.  max_bytes==0 means "full normalized title"; callers
 * that render to the terminal pass an explicit display budget. */
char *agent_session_title_from_text(const char *text, size_t text_len,
                                           size_t max_bytes) {
    static const char user_mark[] = "<｜User｜>";
    static const char assistant_mark[] = "<｜Assistant｜>";
    const char *p = text ? strstr(text, user_mark) : NULL;
    if (!p) return xstrdup("(no user prompt)");
    p += strlen(user_mark);
    const char *end = text + text_len;
    const char *assistant = strstr(p, assistant_mark);
    const char *next_user = strstr(p, user_mark);
    if (assistant && assistant < end) end = assistant;
    if (next_user && next_user < end) end = next_user;
    return agent_session_title_from_span(p, end, max_bytes,
                                         "(empty user prompt)");
}



static char *agent_session_title_clip(const char *title, size_t max_bytes) {
    if (!title) return xstrdup("(no user prompt)");
    size_t len = strlen(title);
    if (max_bytes == 0 || len <= max_bytes) return xstrdup(title);
    if (max_bytes < 4) max_bytes = 4;
    agent_buf b = {0};
    agent_buf_append(&b, title, max_bytes - 3);
    agent_buf_puts(&b, "...");
    return agent_buf_take(&b);
}



static void agent_history_ptrs_push(agent_history_ptrs *p, const char *s,
                                    agent_history_mark mark) {
    if (p->len == p->cap) {
        p->cap = p->cap ? p->cap * 2 : 16;
        p->v = (const char* *)agent_xrealloc(p->v, (size_t)p->cap * sizeof(p->v[0]));
        p->mark = (agent_history_mark *)agent_xrealloc(p->mark, (size_t)p->cap * sizeof(p->mark[0]));
    }
    p->v[p->len] = s;
    p->mark[p->len] = mark;
    p->len++;
}



static const char *agent_memmem(const char *hay, size_t hay_len,
                                const char *needle, size_t needle_len) {
    if (!needle_len) return hay;
    if (needle_len > hay_len) return NULL;
    const char first = needle[0];
    const char *end = hay + hay_len - needle_len + 1;
    for (const char *p = hay; p < end; p++) {
        if (*p == first && memcmp(p, needle, needle_len) == 0) return p;
    }
    return NULL;
}



static const char *agent_history_next_marker(const char *p, const char *end,
                                             agent_history_mark *mark,
                                             size_t *mark_len) {
    static const char user_mark[] = "<｜User｜>";
    static const char assistant_mark[] = "<｜Assistant｜>";
    static const char eos_mark[] = "<｜end▁of▁sentence｜>";
    const char *u = agent_memmem(p, (size_t)(end - p),
                                 user_mark, sizeof(user_mark) - 1);
    const char *a = agent_memmem(p, (size_t)(end - p),
                                 assistant_mark, sizeof(assistant_mark) - 1);
    const char *e = agent_memmem(p, (size_t)(end - p),
                                 eos_mark, sizeof(eos_mark) - 1);
    if (!u && !a && !e) return NULL;
    if (u && (!a || u < a) && (!e || u < e)) {
        if (mark) *mark = AGENT_HISTORY_MARK_USER;
        if (mark_len) *mark_len = sizeof(user_mark) - 1;
        return u;
    }
    if (a && (!e || a < e)) {
        if (mark) *mark = AGENT_HISTORY_MARK_ASSISTANT;
        if (mark_len) *mark_len = sizeof(assistant_mark) - 1;
        return a;
    }
    if (mark) *mark = AGENT_HISTORY_MARK_EOS;
    if (mark_len) *mark_len = sizeof(eos_mark) - 1;
    return e;
}



static void agent_history_trim(const char **p, const char **end) {
    while (*p < *end && isspace((unsigned char)**p)) (*p)++;
    while (*end > *p && isspace((unsigned char)(*end)[-1])) (*end)--;
}



static bool agent_history_has_prefix(const char *p, const char *end,
                                     const char *prefix) {
    size_t n = strlen(prefix);
    return (size_t)(end - p) >= n && memcmp(p, prefix, n) == 0;
}



/* Tool messages are rendered as user turns in the transcript.  Return the
 * inner payload for the current <tool_result> wrapper so /history skips these
 * pseudo-user turns and displays their content without leaking the wrapper. */
static bool agent_history_tool_result_payload(const char **p, const char **end) {
    const char *s = *p, *e = *end;
    agent_history_trim(&s, &e);

    const char *open = "<tool_result>";
    const char *close = "</tool_result>";
    const size_t open_len = strlen(open);
    const size_t close_len = strlen(close);
    if (!agent_history_has_prefix(s, e, open)) return false;

    s += open_len;
    if ((size_t)(e - s) >= close_len &&
        memcmp(e - close_len, close, close_len) == 0)
    {
        e -= close_len;
    }
    *p = s;
    *end = e;
    return true;
}



static bool agent_history_is_tool_user(const char *p, const char *end) {
    agent_history_trim(&p, &end);
    return agent_history_tool_result_payload(&p, &end) ||
           agent_history_has_prefix(p, end, "Tool:") ||
           agent_history_has_prefix(p, end, "Tool result");
}



static void agent_history_ptrs_free(agent_history_ptrs *p) {
    free(p->v);
    free(p->mark);
    memset(p, 0, sizeof(*p));
}



/* Find the oldest rendered-chat marker needed to show the last N user turns.
 * Tool-result pseudo-user turns are skipped while human turns exist, so
 * /history stays centered on the human conversation.  Compacted sessions can
 * legitimately have a tail made only of tool result turns; in that case we
 * fall back to recent tool/assistant events instead of showing an empty
 * history. */
static const char *agent_history_start_for_turns(const char *text, size_t len,
                                                 int user_turns,
                                                 bool *tool_only) {
    const char *end = text + len;
    agent_history_ptrs marks = {0};
    agent_history_ptrs users = {0};
    agent_history_ptrs all_users = {0};
    const char *p = text;
    while (p < end) {
        agent_history_mark mark = AGENT_HISTORY_MARK_NONE;
        size_t mark_len = 0;
        const char *m = agent_history_next_marker(p, end, &mark, &mark_len);
        if (!m) break;
        agent_history_ptrs_push(&marks, m, mark);
        const char *content = m + mark_len;
        agent_history_mark next_mark = AGENT_HISTORY_MARK_NONE;
        size_t next_len = 0;
        const char *next = agent_history_next_marker(content, end,
                                                     &next_mark, &next_len);
        const char *content_end = next ? next : end;
        if (mark == AGENT_HISTORY_MARK_USER) {
            agent_history_ptrs_push(&all_users, m, mark);
            if (!agent_history_is_tool_user(content, content_end))
                agent_history_ptrs_push(&users, m, mark);
        }
        p = content_end;
    }

    const char *start = end;
    if (tool_only) *tool_only = false;
    if (users.len > 0) {
        int idx = users.len - user_turns;
        if (idx < 0) idx = 0;
        start = users.v[idx];
    } else if (all_users.len > 0) {
        int idx = all_users.len - user_turns;
        if (idx < 0) idx = 0;
        start = all_users.v[idx];
        if (tool_only) *tool_only = true;

        /* Tool result messages are stored as user-role turns after the
         * assistant DSML stanza that produced them.  Include that preceding
         * assistant marker when it is still in the retained tail, otherwise
         * replay shows the result but hides the call that caused it. */
        for (int i = marks.len - 1; i >= 0; i--) {
            if (marks.v[i] >= start) continue;
            if (marks.mark[i] == AGENT_HISTORY_MARK_USER) break;
            if (marks.mark[i] == AGENT_HISTORY_MARK_ASSISTANT) {
                start = marks.v[i];
                break;
            }
        }
    }
    agent_history_ptrs_free(&marks);
    agent_history_ptrs_free(&users);
    agent_history_ptrs_free(&all_users);
    return start;
}



static bool agent_history_latest_compaction_summary(const char *text,
                                                    size_t len,
                                                    const char **sum_start,
                                                    const char **sum_end) {
    static const char start_mark[] =
        "[pulsar-agent compacted earlier conversation. Durable task-state summary follows.]";
    static const char end_mark[] =
        "[End compacted summary. Recent conversation continues verbatim below.]";
    const char *end = text + len;
    const char *scan = text;
    const char *best_start = NULL;
    const char *best_end = NULL;
    while (scan < end) {
        const char *s = agent_memmem(scan, (size_t)(end - scan),
                                     start_mark, sizeof(start_mark) - 1);
        if (!s) break;
        const char *content = s + sizeof(start_mark) - 1;
        const char *e = agent_memmem(content, (size_t)(end - content),
                                     end_mark, sizeof(end_mark) - 1);
        if (!e) break;
        best_start = content;
        best_end = e;
        scan = e + sizeof(end_mark) - 1;
    }
    if (!best_start || !best_end) return false;
    agent_history_trim(&best_start, &best_end);
    if (best_start >= best_end) return false;
    if (sum_start) *sum_start = best_start;
    if (sum_end) *sum_end = best_end;
    return true;
}



static void agent_history_publish_limited(agent_worker *w, const char *p,
                                          const char *end, int max_lines,
                                          size_t max_bytes);



static void agent_history_render_compaction_summary(agent_worker *w,
                                                    const char *text,
                                                    size_t len) {
    const char *p = NULL, *end = NULL;
    if (!agent_history_latest_compaction_summary(text, len, &p, &end)) return;
    bool color = isatty(STDOUT_FILENO) != 0;
    if (color) {
        const char *s = "\n\x1b[1;95mCompacted Summary:\x1b[0m\n";
        agent_publish(w, s, strlen(s));
    } else {
        agent_publish(w, "\nCompacted Summary:\n",
                      strlen("\nCompacted Summary:\n"));
    }
    agent_history_publish_limited(w, p, end, 80, 12000);
}



static const char *agent_history_skip_utf8_continuation(const char *p,
                                                        const char *end) {
    while (p < end && (((unsigned char)*p) & 0xc0) == 0x80) p++;
    return p;
}



static const char *agent_history_tail_start(const char *p, const char *end,
                                            int max_lines, size_t max_bytes,
                                            bool *truncated) {
    *truncated = false;
    if (p >= end) return p;

    const char *start = p;
    size_t len = (size_t)(end - p);
    if (max_bytes && len > max_bytes) {
        start = end - max_bytes;
        *truncated = true;
    }

    if (max_lines > 0) {
        const char *scan = end;
        if (scan > p && scan[-1] == '\n') scan--;
        const char *line_start = p;
        int lines = 0;
        while (scan > p) {
            scan--;
            if (*scan == '\n' && ++lines == max_lines) {
                line_start = scan + 1;
                break;
            }
        }
        if (line_start > p) *truncated = true;
        if (line_start > start) start = line_start;
    }

    return agent_history_skip_utf8_continuation(start, end);
}



static void agent_history_publish_limited(agent_worker *w, const char *p,
                                          const char *end, int max_lines,
                                          size_t max_bytes) {
    bool truncated = false;
    const char *start = agent_history_tail_start(p, end, max_lines, max_bytes,
                                                 &truncated);
    if (truncated)
        agent_publish(w, "\n... earlier history truncated; showing tail ...\n",
                      strlen("\n... earlier history truncated; showing tail ...\n"));
    agent_publish(w, start, (size_t)(end - start));
    if (end > start && end[-1] != '\n') agent_publish(w, "\n", 1);
}



static void agent_history_render_assistant(agent_worker *w,
                                           const char *p, const char *end) {
    agent_history_trim(&p, &end);
    if (p >= end) return;
    bool source_truncated = false;
    (void)agent_history_tail_start(p, end,
                                   AGENT_HISTORY_ASSISTANT_MAX_LINES,
                                   AGENT_HISTORY_ASSISTANT_MAX_BYTES,
                                   &source_truncated);
    bool use_color = isatty(STDOUT_FILENO) != 0;
    agent_tail_capture tail = {
        .cap = source_truncated ? (size_t)AGENT_HISTORY_ASSISTANT_MAX_BYTES : (size_t)0,
    };
    agent_token_renderer renderer = {
        .engine = w->engine,
        .worker = w,
        /* History replay should look like the original live output: the user is
         * switching back to a session, not reading a different transcript
         * format.  Tool calls are still dry-rendered below, so replay never
         * executes tools or mutates transcript state. */
        .format_markdown = true,
        .use_color = use_color && !source_truncated,
        .last_output_newline = true,
        .capture = source_truncated ? &tail : NULL,
    };
    agent_dsml_parser dsml = {.state = AGENT_DSML_SEARCH};
    agent_stream_renderer stream = {
        .renderer = &renderer,
        .parser = &dsml,
        .replay = true,
    };

    /* Dry-run replay: the same streaming projection hides DSML and renders
     * semantic tool lines, but no tool is executed and no transcript state is
     * changed.  The saved KV payload remains the only authority for resume. */
    agent_stream_text(&stream, p, (size_t)(end - p), true);
    renderer_finish(&renderer);
    agent_dsml_parser_free(&dsml);

    if (source_truncated) {
        size_t tail_len = 0;
        char *tail_text = tail.take(&tail_len);
        bool rendered_truncated = tail.total > tail_len;
        bool line_truncated = false;
        const char *tail_start =
            agent_history_tail_start(tail_text, tail_text + tail_len,
                                     AGENT_HISTORY_ASSISTANT_MAX_LINES,
                                     AGENT_HISTORY_ASSISTANT_MAX_BYTES,
                                     &line_truncated);
        if (use_color) agent_publish(w, "\x1b[90m", 5);
        agent_publish(w,
                      "\n... earlier assistant history truncated; showing tail ...\n",
                      strlen("\n... earlier assistant history truncated; showing tail ...\n"));
        (void)rendered_truncated;
        agent_publish(w, tail_start, (size_t)(tail_text + tail_len - tail_start));
        if (tail_len && tail_text[tail_len - 1] != '\n') agent_publish(w, "\n", 1);
        if (use_color) agent_publish(w, "\x1b[0m", 4);
        free(tail_text);
    }
}



/* Re-render saved transcript text for /history and /switch.  It intentionally
 * uses the same assistant/token renderer as live output, so restored history
 * looks like the original terminal stream instead of raw rendered-chat text. */
static void agent_history_render_text(agent_worker *w, const char *text,
                                      size_t len, int user_turns) {
    if (user_turns <= 0) return;
    if (user_turns > AGENT_HISTORY_MAX_TURNS)
        user_turns = AGENT_HISTORY_MAX_TURNS;

    const char *end = text + len;
    agent_history_render_compaction_summary(w, text, len);

    bool tool_only = false;
    const char *p = agent_history_start_for_turns(text, len, user_turns,
                                                  &tool_only);
    if (p >= end) {
        agent_publish(w, "\n(no user history)\n", strlen("\n(no user history)\n"));
        return;
    }

    bool color = isatty(STDOUT_FILENO) != 0;
    if (color) agent_publish(w, "\n\x1b[90m", strlen("\n\x1b[90m"));
    else agent_publish(w, "\n", 1);
    if (tool_only) {
        agent_publishf(w, "--- session history: recent tool/assistant events ---\n");
    } else {
        agent_publishf(w, "--- session history: last %d user turn%s ---\n",
                       user_turns, user_turns == 1 ? "" : "s");
    }
    if (color) agent_publish(w, "\x1b[0m", 4);

    while (p < end) {
        agent_history_mark mark = AGENT_HISTORY_MARK_NONE;
        size_t mark_len = 0;
        const char *m = agent_history_next_marker(p, end, &mark, &mark_len);
        if (!m) break;
        const char *content = m + mark_len;
        agent_history_mark next_mark = AGENT_HISTORY_MARK_NONE;
        size_t next_len = 0;
        const char *next = agent_history_next_marker(content, end,
                                                     &next_mark, &next_len);
        const char *content_end = next ? next : end;
        const char *tp = content, *te = content_end;
        agent_history_trim(&tp, &te);

        if (mark == AGENT_HISTORY_MARK_USER) {
            if (agent_history_is_tool_user(tp, te)) {
                const char *payload_start = tp;
                const char *payload_end = te;
                (void)agent_history_tool_result_payload(&payload_start,
                                                        &payload_end);
                if (color) {
                    const char *s = "\x1b[90mTool result:\n";
                    agent_publish(w, s, strlen(s));
                } else {
                    agent_publish(w, "Tool result:\n", strlen("Tool result:\n"));
                }
                agent_history_publish_limited(w, payload_start, payload_end,
                                              12, 3000);
                if (color) agent_publish(w, "\x1b[0m", 4);
            } else {
                if (color) {
                    const char *s = "\x1b[1;32mUser:\x1b[0m\n";
                    agent_publish(w, s, strlen(s));
                } else {
                    agent_publish(w, "User:\n", strlen("User:\n"));
                }
                agent_history_publish_limited(w, tp, te, 24, 6000);
            }
        } else if (mark == AGENT_HISTORY_MARK_ASSISTANT) {
            if (color) {
                const char *s = "\x1b[1;37mAssistant:\x1b[0m\n";
                agent_publish(w, s, strlen(s));
            } else {
                agent_publish(w, "Assistant:\n", strlen("Assistant:\n"));
            }
            agent_history_render_assistant(w, tp, te);
        }
        p = content_end;
    }

    if (color) {
        const char *s = "\x1b[90m--- end history ---\x1b[0m\n";
        agent_publish(w, s, strlen(s));
    } else {
        agent_publish(w, "--- end history ---\n", strlen("--- end history ---\n"));
    }
}



/* Render recent saved transcript text without mutating the live session. */
bool agent_worker_show_history(agent_worker *w, int user_turns,
                                      char *err, size_t err_len) {
    if (!worker_is_idle(w)) {
        snprintf(err, err_len, "model is busy");
        return false;
    }
    size_t text_len = 0;
    char *text = pulsar_kvtext_render_tokens_text(w->engine, &w->transcript,
                                                &text_len);
    if (!text) {
        snprintf(err, err_len, "failed to render session text");
        return false;
    }
    agent_history_render_text(w, text, text_len, user_turns);
    free(text);
    return true;
}



static int agent_session_list_cmp_recent(const void *a, const void *b) {
    const agent_session_list_item *sa = (const agent_session_list_item *)a;
    const agent_session_list_item *sb = (const agent_session_list_item *)b;
    uint64_t ta = sa->f.last_used ? sa->f.last_used : sa->f.created_at;
    uint64_t tb = sb->f.last_used ? sb->f.last_used : sb->f.created_at;
    if (ta < tb) return 1;
    if (ta > tb) return -1;
    return strcmp(sa->sha, sb->sha);
}



/* <40 hex>.session -> the sha. */
static bool agent_session_name_sha(const char *name, char sha[41]) {
    if (strlen(name) != 48 || strcmp(name + 40, ".session") != 0) return false;
    for (int i = 0; i < 40; i++)
        if (!isxdigit((unsigned char)name[i])) return false;
    memcpy(sha, name, 40);
    sha[40] = '\0';
    return true;
}



/* Every saved session of this model whose sha starts with `prefix` ("" for
 * all), read and verified; the caller frees with agent_session_list_free. */
static int agent_session_scan(agent_worker *w, const char *prefix, agent_session_list_item **out) {
    *out = NULL;
    DIR *d = opendir(w->cache_dir);
    if (!d) return 0;
    const size_t plen = strlen(prefix);
    const uint32_t model_id = (uint32_t)pulsar_engine_model_id(w->engine);
    int len = 0, cap = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!agent_session_name_sha(de->d_name, sha)) continue;
        if (plen && strncasecmp(sha, prefix, plen) != 0) continue;
        char *path = pulsar_kvtext_path_join(w->cache_dir, de->d_name);
        agent_session_file f;
        char err[96];
        const bool ok = agent_session_file_read(path, &f, err, sizeof(err));
        free(path);
        if (!ok) continue;
        if (f.model_id != model_id) {
            agent_session_file_free(&f);
            continue;
        }
        if (len == cap) {
            cap = cap ? cap * 2 : 16;
            *out = (agent_session_list_item *)agent_xrealloc(*out, (size_t)cap * sizeof((*out)[0]));
        }
        memcpy((*out)[len].sha, sha, 41);
        (*out)[len].f = f;
        len++;
    }
    closedir(d);
    return len;
}



static void agent_session_list_free(agent_session_list_item *v, int n) {
    for (int i = 0; i < n; i++) agent_session_file_free(&v[i].f);
    free(v);
}



/* Print resumable sessions from <cache>/<sha>.session. */
void agent_worker_list_sessions(agent_worker *w) {
    int cols = renderer_terminal_cols();
    size_t title_budget = cols > 16 ? (size_t)(cols - 12) : 20;
    if (title_budget > 160) title_budget = 160;

    agent_session_list_item *sessions = NULL;
    const int sessions_len = agent_session_scan(w, "", &sessions);
    if (!sessions_len) {
        printf("no saved sessions\n");
        return;
    }
    qsort(sessions, (size_t)sessions_len, sizeof(sessions[0]), agent_session_list_cmp_recent);

    bool color = isatty(STDOUT_FILENO) != 0;
    const char *sha_on = color ? "\x1b[1;96m" : "";
    const char *title_on = color ? "\x1b[1;97m" : "";
    const char *help_on = color ? "\x1b[97m" : "";
    const char *dim = color ? "\x1b[90m" : "";
    const char *reset = color ? "\x1b[0m" : "";

    for (int i = 0; i < sessions_len; i++) {
        const agent_session_file *f = &sessions[i].f;
        char age[32];
        agent_format_age(f->last_used ? f->last_used : f->created_at, age, sizeof(age));
        char *title = agent_session_title_clip(f->title, title_budget);
        printf("%s%.8s%s %s>%s %s%s%s\n",
               sha_on, sessions[i].sha, reset, dim, reset,
               title_on, title, reset);
        printf("         %s> %s, %d tokens%s\n\n", dim, age, f->tokens.len, reset);
        free(title);
    }
    printf("%sUse /switch <id> to select a session, /del <id> to remove, "
           "/strip <id> to release its cached KV.%s\n",
           help_on, reset);
    agent_session_list_free(sessions, sessions_len);
}



/* Tab completion for /switch.  Suggestions are sorted by recent use and accept
 * either an empty prefix or any unambiguous hex prefix. */
void agent_switch_completion_callback(const char *buf,
                                             linenoiseCompletions *lc) {
    agent_worker *w = agent_completion_worker;
    static const char cmd[] = "/switch";
    const size_t cmd_len = sizeof(cmd) - 1;
    if (!w || !buf || strncmp(buf, cmd, cmd_len) != 0) return;

    const char *p = buf + cmd_len;
    if (*p && *p != ' ' && *p != '\t') return;
    while (*p == ' ' || *p == '\t') p++;

    const char *prefix = p;
    size_t prefix_len = strlen(prefix);
    for (size_t i = 0; i < prefix_len; i++) {
        if (!isxdigit((unsigned char)prefix[i])) return;
    }
    if (prefix_len > 40) return;

    agent_session_list_item *sessions = NULL;
    const int n = agent_session_scan(w, prefix, &sessions);
    qsort(sessions, (size_t)n, sizeof(sessions[0]), agent_session_list_cmp_recent);
    for (int i = 0; i < n; i++) {
        char line[64];
        int sha_chars = prefix_len > 8 ? 40 : 8;
        snprintf(line, sizeof(line), "/switch %.*s", sha_chars, sessions[i].sha);
        linenoiseAddCompletion(lc, line);
    }
    agent_session_list_free(sessions, n);
}



/* Resolve a user-provided SHA prefix to exactly one saved session. */
static bool agent_worker_find_session(agent_worker *w, const char *prefix,
                                      char sha_out[41], agent_session_file *f_out,
                                      char *err, size_t err_len) {
    size_t plen = strlen(prefix);
    if (plen == 0 || plen > 40) {
        snprintf(err, err_len, "invalid session SHA prefix");
        return false;
    }
    for (size_t i = 0; i < plen; i++) {
        if (!isxdigit((unsigned char)prefix[i])) {
            snprintf(err, err_len, "invalid session SHA prefix");
            return false;
        }
    }
    agent_session_list_item *sessions = NULL;
    const int n = agent_session_scan(w, prefix, &sessions);
    if (n != 1) {
        if (n == 0) snprintf(err, err_len, "no saved session matches %.40s", prefix);
        else snprintf(err, err_len, "session prefix %.40s is ambiguous", prefix);
        agent_session_list_free(sessions, n);
        return false;
    }
    memcpy(sha_out, sessions[0].sha, 41);
    *f_out = sessions[0].f;
    memset(&sessions[0].f, 0, sizeof(sessions[0].f));
    agent_session_list_free(sessions, n);
    return true;
}



/* Release the KV only this session's chain holds: the segments past the last
 * one any other chain shares, never the system prompt's.  The session keeps its
 * exact ids, so switching to it later prefills what was released. */
static uint64_t agent_session_release_kv(agent_worker *w, const agent_session_file *f) {
    if (!w->kv || f->tokens.len <= 0) return 0;
    size_t text_len = 0;
    char *text = pulsar_kvtext_render_tokens_text(w->engine, &f->tokens, &text_len);
    pulsar_segstore_seg chain[PULSAR_KVCHAIN_MAX];
    const int n = text ? pulsar_segstore_lookup(w->kv, text, text_len, chain, PULSAR_KVCHAIN_MAX) : 0;
    free(text);
    if (n == 0) return 0;
    char sys_tip[41];
    agent_kv_system_tip(w, sys_tip);
    return pulsar_segstore_release(w->kv, chain[n - 1].key, sys_tip[0] ? sys_tip : NULL);
}



bool agent_worker_delete_session(agent_worker *w, const char *prefix,
                                        char sha_out[41],
                                        char *err, size_t err_len) {
    if (!worker_is_idle(w)) {
        snprintf(err, err_len, "model is busy");
        return false;
    }
    char sha[41];
    agent_session_file f;
    if (!agent_worker_find_session(w, prefix, sha, &f, err, err_len)) return false;
    char *path = agent_session_path_for_sha(w->cache_dir, sha);
    const bool ok = unlink(path) == 0;
    if (!ok) snprintf(err, err_len, "%s", strerror(errno));
    else (void)agent_session_release_kv(w, &f);
    if (ok && sha_out) memcpy(sha_out, sha, 41);
    agent_session_file_free(&f);
    free(path);
    return ok;
}



/* Release a saved session's cached KV while keeping the session (L264: its KV
 * lives in the shared segment store; only the stretch no other chain uses
 * goes).  *bytes_out: what was freed. */
bool agent_worker_strip_session(agent_worker *w, const char *prefix,
                                       char sha_out[41],
                                       uint64_t *bytes_out,
                                       char *err, size_t err_len) {
    if (!worker_is_idle(w)) {
        snprintf(err, err_len, "model is busy");
        return false;
    }
    char sha[41];
    agent_session_file f;
    if (!agent_worker_find_session(w, prefix, sha, &f, err, err_len)) return false;
    const uint64_t freed = agent_session_release_kv(w, &f);
    if (sha_out) memcpy(sha_out, sha, 41);
    if (bytes_out) *bytes_out = freed;
    agent_session_file_free(&f);
    return true;
}



/* Load a saved session into the live transcript -- the deepest chain the store
 * holds for it, then the rest by prefill from its exact ids -- and optionally
 * replay recent history for the human. */
bool agent_worker_switch_session(agent_worker *w, const char *prefix,
                                        int history_turns,
                                        char *err, size_t err_len) {
    if (!worker_is_idle(w)) {
        snprintf(err, err_len, "model is busy");
        return false;
    }
    char sha[41];
    agent_session_file f;
    if (!agent_worker_find_session(w, prefix, sha, &f, err, err_len)) return false;
    if (f.tokens.len <= 0) {
        snprintf(err, err_len, "saved session %.8s holds no tokens", sha);
        agent_session_file_free(&f);
        return false;
    }
    const int cached = agent_kv_load(w, &f.tokens);
    if (cached < f.tokens.len) {
        printf("restoring session %.8s: %d of %d tokens cached, prefilling the rest...\n",
               sha, cached, f.tokens.len);
        fflush(stdout);
    }
    const bool ok = agent_worker_sync_tokens(w, &f.tokens, true, err, err_len) == 0;
    if (ok) {
        pulsar_tokens_free(&w->transcript);
        w->transcript = f.tokens;
        agent_turn_clear(w);
        memset(&f.tokens, 0, sizeof(f.tokens));
        free(w->session_title);
        w->session_title = f.title && f.title[0] ? xstrdup(f.title) : xstrdup("(no user prompt)");
        w->session_created_at = f.created_at ? f.created_at : (uint64_t)time(NULL);
        memcpy(w->session_sha, sha, sizeof(w->session_sha));
        agent_worker_note_system_prompt_seen(w);
        w->datetime_context_injected = true;
        pthread_mutex_lock(&w->mu);
        w->user_activity = true;
        w->session_dirty = false;
        w->status.state = AGENT_WORKER_IDLE;
        w->status.ctx_used = w->transcript.len;
        w->status.ctx_size = w->cfg->gen.ctx_size;
        w->status.prefill_tps = 0.0;
        w->status.greedy_sampling = false;
        w->status.error[0] = '\0';
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
        printf("switched to session %.8s (%d tokens)\n", sha, w->transcript.len);
        if (history_turns > 0)
            (void)agent_worker_show_history(w, history_turns, err, err_len);
    }
    agent_session_file_free(&f);
    return ok;
}

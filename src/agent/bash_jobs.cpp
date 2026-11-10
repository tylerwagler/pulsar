/** @file
 *  pulsar-agent: the bash tool's background jobs (spawn, poll, output capture, observation). */
#include "pulsar_agent_internal.h"
#include "pulsar_utf8.h"



/** L373: one job's entry in the exit teardown's table: plain words a signal handler can read.  `path` is written
 *  before `used` is set and never changes while it is; `used` is set before the job's fork, `pgid` before its
 *  command starts (agent_bash_start). */
struct agent_bash_live_slot {
    volatile sig_atomic_t used;   ///< the slot holds a job's output file
    volatile sig_atomic_t pgid;   ///< the job's process group while it runs; 0 once reaped
    char path[sizeof(((agent_bash_job *)0)->path)];   ///< the job's output file
};

/** L373: the live jobs' process groups and output files, for the ONE teardown every way out of the process reaches
 *  (agent_bash_teardown): a fixed table, not the worker's job list, because a signal handler walks it. */
static agent_bash_live_slot g_bash_live[AGENT_BASH_JOBS_MAX];

void agent_bash_teardown(void) {
    for (int i = 0; i < AGENT_BASH_JOBS_MAX; i++) {
        const pid_t pgid = (pid_t)g_bash_live[i].pgid;
        if (pgid > 0) kill(-pgid, SIGKILL);
        if (g_bash_live[i].used) unlink(g_bash_live[i].path);
    }
}

/** @brief The signal handler: the teardown, then the signal's default action (SA_RESETHAND restored it; the raise is
 *  delivered when the handler returns).  Only async-signal-safe calls: kill, unlink, raise.
 *  @param sig the signal */
static void agent_bash_teardown_signal(int sig) {
    agent_bash_teardown();
    raise(sig);
}

void agent_bash_teardown_install(bool with_sigint) {
    atexit(agent_bash_teardown);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = agent_bash_teardown_signal;
    sa.sa_flags = SA_RESETHAND;
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGHUP, &sa, NULL);
    if (with_sigint) sigaction(SIGINT, &sa, NULL);
}

/** @brief A free slot of the live table.
 *  @return its index, or -1 when AGENT_BASH_JOBS_MAX jobs are tracked */
static int agent_bash_live_free_slot(void) {
    for (int i = 0; i < AGENT_BASH_JOBS_MAX; i++)
        if (!g_bash_live[i].used) return i;
    return -1;
}



int agent_bash_job::display_lines() const {
    const auto *job = this;
    if (!job || job->bytes == 0) return 0;
    return job->newline_count + (job->last_byte != '\n');
}



void agent_bash_job::note_output(const char *s, size_t n) {
    auto *job = this;
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\n') job->newline_count++;
    }
    if (n) job->last_byte = s[n - 1];
    job->bytes += n;
}



void agent_bash_job::job_free() {
    auto *job = this;
    if (!job) return;
    if (job->running && job->pid > 0) {
        kill(-job->pid, SIGKILL);
        kill(job->pid, SIGKILL);
        waitpid(job->pid, NULL, 0);
    }
    g_bash_live[job->live_slot].pgid = 0;
    if (job->pipe_fd >= 0) close(job->pipe_fd);
    if (job->tmp_fd >= 0) close(job->tmp_fd);
    /* Remove the mkstemp output file.  Only the three spawn-time error paths
     * unlinked it before, so every COMPLETED bash tool call left a
     * /tmp/ds4_agent_output_* behind for the life of the box — a long agent
     * session orphaned one per command, sized by that command's full output. */
    if (job->path[0]) unlink(job->path);
    g_bash_live[job->live_slot].used = 0;
    free(job->cmd);
    free(job);
}



void agent_bash_jobs_free(agent_worker *w) {
    agent_bash_teardown();   /* the same kill and unlink every other way out runs; job_free then reaps */
    agent_bash_job *job = w->bash_jobs;
    while (job) {
        agent_bash_job *next = job->next;
        job->job_free();
        job = next;
    }
    w->bash_jobs = NULL;
}



bool agent_bash_jobs_poll(agent_worker *w) {
    bool running = false;
    for (agent_bash_job *job = w->bash_jobs; job; job = job->next) {
        job->poll();
        running = running || job->running;
    }
    return running;
}



agent_bash_job *agent_bash_find_job(agent_worker *w, int id, pid_t pid) {
    for (agent_bash_job *job = w->bash_jobs; job; job = job->next) {
        if ((id > 0 && job->id == id) || (id <= 0 && pid > 0 && job->pid == pid))
            return job;
    }
    return NULL;
}



void agent_bash_remove_job(agent_worker *w, agent_bash_job *target) {
    agent_bash_job **link = &w->bash_jobs;
    while (*link) {
        if (*link == target) {
            *link = target->next;
            target->next = NULL;
            target->job_free();
            return;
        }
        link = &(*link)->next;
    }
}



void agent_bash_job::drain() {
    auto *job = this;
    if (!job || job->pipe_fd < 0) return;
    char tmp[4096];
    for (;;) {
        ssize_t n = read(job->pipe_fd, tmp, sizeof(tmp));
        if (n > 0) {
            job->note_output(tmp, (size_t)n);
            if (job->tmp_fd >= 0) write_all(job->tmp_fd, tmp, (size_t)n);
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        break;
    }
}



/** @brief Flag (under w->mu) that a child may have changed the terminal mode, so the UI thread re-checks raw mode
 *  at its next safe point.
 *  @param w the worker; NULL is a no-op */
static void agent_worker_note_terminal_mode_may_have_changed(agent_worker *w) {
    if (!w) return;
    pthread_mutex_lock(&w->mu);
    w->raw_mode_needs_restore = true;
    pthread_mutex_unlock(&w->mu);
}



void agent_bash_job::finalize(int status) {
    auto *job = this;
    job->drain();
    if (job->pipe_fd >= 0) {
        close(job->pipe_fd);
        job->pipe_fd = -1;
    }
    if (job->tmp_fd >= 0) {
        close(job->tmp_fd);
        job->tmp_fd = -1;
    }
    if (WIFEXITED(status)) job->exit_status = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) job->exit_status = 128 + WTERMSIG(status);
    else job->exit_status = -1;
    job->running = false;
    g_bash_live[job->live_slot].pgid = 0;   /* reaped: the id may be reused, the teardown no longer kills it */
    /* A child can still open /dev/tty directly and alter terminal state even
     * though its stdin is /dev/null.  Ask the UI thread to verify raw mode at
     * a safe point instead of touching linenoise from the worker path. */
    agent_worker_note_terminal_mode_may_have_changed(job->worker);
}



/* Drain available output, notice process exit, and enforce timeout.  Called by
 * status/wait/compaction and, for every job, once a tool round and once a second
 * while the worker idles (agent_bash_jobs_poll) instead of a background reaper
 * thread, keeping all bash job state owned by the agent worker. */
void agent_bash_job::poll() {
    auto *job = this;
    if (!job || !job->running) return;
    job->drain();

    int status = 0;
    pid_t rc = waitpid(job->pid, &status, WNOHANG);
    if (rc == job->pid) {
        job->finalize(status);
        return;
    }
    if (rc < 0 && errno != EINTR) {
        job->exit_status = -1;
        job->running = false;
        g_bash_live[job->live_slot].pgid = 0;
        if (job->pipe_fd >= 0) {
            close(job->pipe_fd);
            job->pipe_fd = -1;
        }
        if (job->tmp_fd >= 0) {
            close(job->tmp_fd);
            job->tmp_fd = -1;
        }
        agent_worker_note_terminal_mode_may_have_changed(job->worker);
        return;
    }
    if (agent_now_sec() - job->start_time >= job->timeout_sec) {
        job->timed_out = true;
        kill(-job->pid, SIGKILL);
        kill(job->pid, SIGKILL);
        while (waitpid(job->pid, &status, 0) < 0 && errno == EINTR) {}
        job->finalize(status);
    }
}



/* Spawn a shell command into its own process group so bash_stop/timeout can
 * kill grandchildren created by the shell, not just the /bin/sh wrapper. */
agent_bash_job *agent_bash_start(agent_worker *w, const char *cmd,
                                        int timeout_sec, char *err, size_t err_len) {
    const int slot = agent_bash_live_free_slot();
    if (slot < 0) {
        snprintf(err, err_len, "%d bash jobs are already tracked; stop one with bash_stop", AGENT_BASH_JOBS_MAX);
        return NULL;
    }
    char tmp_path[] = "/tmp/ds4_agent_output_XXXXXX";
    int tmpfd = mkstemp(tmp_path);
    if (tmpfd < 0) {
        snprintf(err, err_len, "failed to create temporary output file: %s", strerror(errno));
        return NULL;
    }

    /* L373: no child runs a command the teardown cannot see.  The output file is in the table before the fork; the
     * child waits on `gate` until the parent has written its process group there, and exits without running anything
     * when the parent is gone first (a signal's teardown on any thread between the fork and that write). */
    int pipefd[2], gate[2];
    if (pipe(pipefd) != 0) {
        snprintf(err, err_len, "failed to create pipe: %s", strerror(errno));
        close(tmpfd);
        unlink(tmp_path);
        return NULL;
    }
    if (pipe2(gate, O_CLOEXEC) != 0) {
        snprintf(err, err_len, "failed to create pipe: %s", strerror(errno));
        close(pipefd[0]);
        close(pipefd[1]);
        close(tmpfd);
        unlink(tmp_path);
        return NULL;
    }
    snprintf(g_bash_live[slot].path, sizeof(g_bash_live[slot].path), "%s", tmp_path);
    g_bash_live[slot].pgid = 0;
    g_bash_live[slot].used = 1;
    pid_t pid = fork();
    if (pid < 0) {
        snprintf(err, err_len, "failed to fork: %s", strerror(errno));
        close(pipefd[0]);
        close(pipefd[1]);
        close(gate[0]);
        close(gate[1]);
        close(tmpfd);
        unlink(tmp_path);
        g_bash_live[slot].used = 0;
        return NULL;
    }
    if (pid == 0) {
        setpgid(0, 0);
        close(tmpfd);
        close(gate[1]);
        char go;
        ssize_t got;
        while ((got = read(gate[0], &go, 1)) < 0 && errno == EINTR) {}
        if (got != 1) _exit(127);   /* the parent ended before it could track this job */
        /* The bash tool is not interactive.  Give the shell /dev/null as
         * stdin so it does not inherit the live linenoise terminal and reset
         * it from raw mode to cooked mode behind the agent's back. */
        int null_fd = open("/dev/null", O_RDONLY);
        if (null_fd >= 0) {
            if (dup2(null_fd, STDIN_FILENO) < 0)
                close(STDIN_FILENO);
            if (null_fd != STDIN_FILENO)
                close(null_fd);
        } else {
            close(STDIN_FILENO);
        }
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        execl("/bin/sh", "sh", "-c", cmd ? cmd : "", (char *)NULL);
        _exit(127);
    }

    close(pipefd[1]);
    close(gate[0]);
    setpgid(pid, pid);
    g_bash_live[slot].pgid = pid;
    write_all(gate[1], "x", 1);
    close(gate[1]);
    int old_flags;
    set_nonblock(pipefd[0], true, &old_flags);

    agent_bash_job *job = (agent_bash_job *)agent_xmalloc(sizeof(*job));
    memset(job, 0, sizeof(*job));
    if (w->next_bash_job_id <= 0) w->next_bash_job_id = 1;
    job->id = w->next_bash_job_id++;
    job->pid = pid;
    job->live_slot = slot;
    job->pipe_fd = pipefd[0];
    job->tmp_fd = tmpfd;
    snprintf(job->path, sizeof(job->path), "%s", tmp_path);
    job->cmd = xstrdup(cmd);
    job->start_time = agent_now_sec();
    job->timeout_sec = timeout_sec;
    job->exit_status = -1;
    job->running = true;
    job->worker = w;
    job->next = w->bash_jobs;
    w->bash_jobs = job;
    return job;
}



/** @brief Append bytes to `b`, then drop from the front so at most `max` bytes remain (keeps the tail).
 *  @param b   the buffer (NUL-terminated after the call)
 *  @param s   bytes to append
 *  @param n   byte count; 0 is a no-op
 *  @param max the byte cap kept */
static void agent_tail_append(agent_buf *b, const char *s, size_t n, size_t max) {
    if (!n) return;
    agent_buf_append(b, s, n);
    if (b->len > max) {
        size_t drop = b->len - max;
        memmove(b->ptr, b->ptr + drop, b->len - drop + 1);
        b->len -= drop;
    }
}



/* Read the first max_lines from the output file, with a byte cap to avoid a
 * pathological single long line flooding the next model turn. */
char *agent_bash_job::read_head(int max_lines,
                                size_t max_bytes, int *lines_read,
                                bool *byte_limited) const {
    const auto *job = this;
    if (lines_read) *lines_read = 0;
    if (byte_limited) *byte_limited = false;
    if (!job || !job->path[0] || job->bytes == 0) return xstrdup("");
    FILE *fp = fopen(job->path, "rb");
    if (!fp) return xstrdup("<failed to reopen output file>\n");

    agent_buf out = {0};
    int lines = 0;
    while (lines < max_lines && out.len < max_bytes) {
        int c = fgetc(fp);
        if (c == EOF) {
            if (ferror(fp) && errno == EINTR) {
                clearerr(fp);
                continue;
            }
            break;
        }
        char ch = (char)c;
        agent_buf_append(&out, &ch, 1);
        if (ch == '\n') lines++;
    }
    if (out.len >= max_bytes && !feof(fp) && byte_limited) *byte_limited = true;
    fclose(fp);
    if (out.len >= max_bytes) {   /* L373: the byte cap cuts on a character boundary */
        out.len = agent_utf8_complete_len(out.ptr, out.len);
        out.ptr[out.len] = '\0';
    }
    if (lines_read) *lines_read = lines + (out.len && out.ptr[out.len - 1] != '\n');
    if (!out.ptr) return xstrdup("");
    return agent_buf_take(&out);
}



/* Read the last max_lines from the full output file.  The model-visible label
 * says "tail -N <file>" so it is clear this is not the complete output. */
char *agent_bash_job::read_tail_lines(int max_lines) const {
    const auto *job = this;
    if (!job || !job->path[0] || job->bytes == 0) return xstrdup("");
    FILE *fp = fopen(job->path, "rb");
    if (!fp) return xstrdup("<failed to reopen output file>\n");

    agent_buf tail = {0};
    char tmp[2048];
    size_t total = 0;
    for (;;) {
        size_t n = fread(tmp, 1, sizeof(tmp), fp);
        total += n;
        if (n) agent_tail_append(&tail, tmp, n, AGENT_BASH_TAIL_BYTES);
        if (n < sizeof(tmp)) {
            if (ferror(fp) && errno == EINTR) {
                clearerr(fp);
                continue;
            }
            break;
        }
    }
    fclose(fp);
    if (!tail.ptr) return xstrdup("");

    /* L373: a tail whose front the byte cap dropped starts on a character boundary */
    char *start = total > AGENT_BASH_TAIL_BYTES ?
        (char *)agent_utf8_skip_continuation(tail.ptr, tail.ptr + tail.len) : tail.ptr;
    int newlines = 0;
    for (char *p = tail.ptr + tail.len; p > tail.ptr; p--) {
        if (p[-1] == '\n' && ++newlines > max_lines) {
            start = p;
            break;
        }
    }
    char *out = xstrdup(start);
    free(tail.ptr);
    return out;
}



/* Build the tool result for a bash job.  The first observation shows the HEAD
 * of the output, every later one the TAIL; mark_observed is what flips it (a
 * repeated bash_status re-shows the current tail: there is no "new bytes only"
 * cursor). */
char *agent_bash_job::observation(bool mark_observed) {
    auto *job = this;
    job->poll();
    bool first_observation = !job->observed_once;
    int display_lines = job->display_lines();
    double elapsed = agent_now_sec() - job->start_time;

    agent_buf out = {0};
    char line[PATH_MAX + 256];
    if (job->running) {
        snprintf(line, sizeof(line),
            "bash job=%d pid=%ld status=running elapsed_sec=%.1f timeout_sec=%.0f\n",
            job->id, (long)job->pid, elapsed, job->timeout_sec);
    } else {
        snprintf(line, sizeof(line),
            "bash job=%d pid=%ld status=done elapsed_sec=%.1f timed_out=%d\n",
            job->id, (long)job->pid, elapsed, job->timed_out ? 1 : 0);
    }
    agent_buf_puts(&out, line);
    if (!job->running) {
        snprintf(line, sizeof(line), "exit_status=%d\n", job->exit_status);
        agent_buf_puts(&out, line);
    }

    if (job->bytes == 0) {
        agent_buf_puts(&out, "<output>\n</output>\n");
    } else if (first_observation) {
        int shown_lines = 0;
        bool byte_limited = false;
        char *head = job->read_head(AGENT_BASH_HEAD_LINES,
                                          AGENT_BASH_HEAD_BYTES,
                                          &shown_lines, &byte_limited);
        bool truncated = byte_limited || display_lines > shown_lines;
        if (!job->running && !truncated) {
            agent_buf_puts(&out, "<output>\n");
            agent_buf_puts(&out, head);
            if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
            agent_buf_puts(&out, "</output>\n");
        } else {
            snprintf(line, sizeof(line),
                     "output_path=%s (%zu bytes, %d lines)\n",
                     job->path[0] ? job->path : "<unavailable>",
                     job->bytes, display_lines);
            agent_buf_puts(&out, line);
            snprintf(line, sizeof(line), "<head -%d %s>\n",
                     AGENT_BASH_HEAD_LINES, job->path);
            agent_buf_puts(&out, line);
            agent_buf_puts(&out, head);
            if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
            agent_buf_puts(&out, "</head>\n");
        }
        free(head);
    } else {
        int tail_lines = job->running ? AGENT_BASH_PROGRESS_TAIL_LINES :
                                        AGENT_BASH_FINAL_TAIL_LINES;
        char *tail = job->read_tail_lines(tail_lines);
        snprintf(line, sizeof(line),
                 "output_path=%s (%zu bytes, %d lines)\n",
                 job->path[0] ? job->path : "<unavailable>",
                 job->bytes, display_lines);
        agent_buf_puts(&out, line);
        snprintf(line, sizeof(line), "<tail -%d %s>\n", tail_lines, job->path);
        agent_buf_puts(&out, line);
        agent_buf_puts(&out, tail);
        if (tail[0] && tail[strlen(tail) - 1] != '\n') agent_buf_puts(&out, "\n");
        snprintf(line, sizeof(line), "</tail>\n");
        agent_buf_puts(&out, line);
        free(tail);
    }
    if (job->running) {
        snprintf(line, sizeof(line),
            "\nUse bash_status job=%d to get info before refresh time; use bash_stop job=%d to stop execution\n",
            job->id, job->id);
        agent_buf_puts(&out, line);
    }

    if (mark_observed) job->observed_once = true;
    return agent_buf_take(&out);
}



/** @brief Show a bash observation's output body to the user: a grey "[showing first/last output lines]" note for a
 *  head/tail section, the body itself (orange when the job finished with a nonzero exit status).
 *  @param w   the worker whose UI receives it
 *  @param obs the observation text (agent_bash_job::observation); NULL or empty shows nothing */
static void agent_bash_publish_observation(agent_worker *w, const char *obs) {
    if (!obs || !obs[0]) return;
    const char *body = NULL;
    const char *label = strstr(obs, "\n<head ");
    const char *close = NULL;
    if (label) {
        close = "</head>";
    } else {
        label = strstr(obs, "\n<tail ");
        if (label) close = "</tail>";
    }
    if (label) {
        const char *tag_end = strstr(label, ">\n");
        if (tag_end) {
            agent_publish(w, "\x1b[90m", 5);
            if (strstr(label, "\n<head ") == label)
                agent_publish(w, "[showing first output lines]\n",
                              strlen("[showing first output lines]\n"));
            else
                agent_publish(w, "[showing last output lines]\n",
                              strlen("[showing last output lines]\n"));
            agent_publish(w, "\x1b[0m", 4);
            body = tag_end + 2;
        }
    } else {
        label = strstr(obs, "\n<output>\n");
        if (label) {
            body = label + strlen("\n<output>\n");
            close = "</output>";
        }
    }
    if (!body || !body[0]) return;
    const char *end = close ? strstr(body, close) : NULL;
    size_t n = end ? (size_t)(end - body) : strlen(body);
    if (n) {
        bool failed = strstr(obs, "status=done") && !strstr(obs, "exit_status=0\n");
        if (failed) agent_publish(w, "\x1b[38;5;208m", 11);
        agent_publish_untrusted(w, body, n);
        if (body[n - 1] != '\n') agent_publish(w, "\n", 1);
        if (failed) agent_publish(w, "\x1b[0m", 4);
    }
}



/** @brief Poll a running job for up to `refresh_sec` seconds (100 ms waits on its pipe), stopping early when it
 *  exits or the worker is interrupted; polls once more at the end.
 *  @param w           the worker (for the interrupt check)
 *  @param job         the job to poll
 *  @param refresh_sec the wait budget, seconds */
static void agent_bash_refresh_for(agent_worker *w, agent_bash_job *job,
                                   int refresh_sec) {
    double start = agent_now_sec();
    while (job->running && agent_now_sec() - start < refresh_sec) {
        if (worker_should_interrupt(w)) break;
        job->poll();
        if (!job->running) break;
        struct pollfd pfd = {.fd = job->pipe_fd, .events = POLLIN};
        poll(&pfd, 1, 100);
    }
    job->poll();
}



/* Common implementation for bash, bash_status, and bash_stop. */
char *agent_bash_job_tool_result(agent_worker *w, agent_bash_job *job,
                                        bool wait, int refresh_sec,
                                        bool stop, bool remove_if_done) {
    if (stop && job->running) {
        kill(-job->pid, SIGTERM);
        kill(job->pid, SIGTERM);
        double start = agent_now_sec();
        while (job->running && agent_now_sec() - start < 1.0) {
            job->poll();
            if (!job->running) break;
            usleep(20000);
        }
        if (job->running) {
            kill(-job->pid, SIGKILL);
            kill(job->pid, SIGKILL);
        }
    }
    if (wait || stop) agent_bash_refresh_for(w, job, refresh_sec);
    else job->poll();

    char *obs = job->observation(true);
    agent_bash_publish_observation(w, obs);
    if (remove_if_done && !job->running) agent_bash_remove_job(w, job);
    return obs;
}



#ifdef PULSAR_AGENT_TEST
/** @brief Whether the text is well-formed UTF-8 (pulsar_utf8.h's rule).
 *  @param s the text
 *  @return true when every byte is part of a well-formed sequence */
static bool test_utf8_well_formed(const char *s) {
    const size_t len = strlen(s);
    for (size_t i = 0; i < len;) {
        const int k = utf8_seq_ok((const unsigned char *)s + i, len - i);
        if (!k) return false;
        i += (size_t)k;
    }
    return true;
}

/** L373: what a tool hands the transcript is well-formed UTF-8 -- a lone lead byte and a character cut short become
 *  U+FFFD at the chat door's repair (agent_chat_messages_repair, what agent_chat_append_turn renders), and the bash
 *  head and tail byte caps cut on a character boundary, so they add no U+FFFD of their own. */
static void test_agent_tool_results_are_utf8(void) {
    AGENT_TEST_ASSERT(agent_utf8_repaired("plain \xe2\x80\x9cquoted\xe2\x80\x9d") == NULL);
    pulsar_chat_message msgs[3] = {
        {"tool", "gcc: \xe2 error", false, "ba\xffsh"},              /* a lone 0xE2, a bad byte in the name */
        {"tool", "warning: \xe2\x80\x98x\xe2\x80", false, NULL},     /* a closing quote cut after two bytes */
        {"user", "caf\xe9 \xc0\xaf \xed\xa0\x80 ok", false, NULL},    /* Latin-1, an overlong form, a surrogate */
    };
    std::vector<char *> owned;
    agent_chat_messages_repair(msgs, 3, &owned);
    AGENT_TEST_ASSERT(owned.size() == 4);
    AGENT_TEST_ASSERT(!strcmp(msgs[0].content, "gcc: \xef\xbf\xbd error") && !strcmp(msgs[0].name, "ba\xef\xbf\xbdsh"));
    AGENT_TEST_ASSERT(!strcmp(msgs[1].content, "warning: \xe2\x80\x98x\xef\xbf\xbd\xef\xbf\xbd"));
    for (const pulsar_chat_message &m : msgs)
        AGENT_TEST_ASSERT(test_utf8_well_formed(m.content) && strstr(m.content, "\xef\xbf\xbd") != NULL);
    AGENT_TEST_ASSERT(strstr(msgs[2].content, " ok") != NULL);
    for (char *p : owned) free(p);

    AGENT_TEST_ASSERT(agent_utf8_complete_len("ab\xe2\x80\x9c", 5) == 5);
    AGENT_TEST_ASSERT(agent_utf8_complete_len("ab\xe2\x80\x9c", 4) == 2);
    AGENT_TEST_ASSERT(agent_utf8_complete_len("ab\xe2\x80\x9c", 3) == 2);
    AGENT_TEST_ASSERT(agent_utf8_complete_len("ab\xf0\x9f\x9b", 5) == 2);
    AGENT_TEST_ASSERT(agent_utf8_complete_len("ab", 2) == 2 && agent_utf8_complete_len("", 0) == 0);
    AGENT_TEST_ASSERT(agent_utf8_complete_len("\x80\x80\x80\x80\x80", 5) == 5);   /* no lead: not this cut's to fix */

    /* one 36000-byte line of curly quotes: the 8 KiB head cap and the 32 KiB tail cap both land inside a character */
    agent_worker w;
    memset(&w, 0, sizeof(w));
    pthread_mutex_init(&w.mu, NULL);
    w.wake_pending = true;   /* no wake pipe */
    char err[160] = {0};
    agent_bash_job *job = agent_bash_start(
        &w, "i=0; while [ $i -lt 12000 ]; do printf '\\342\\200\\234'; i=$((i+1)); done", 60, err, sizeof err);
    AGENT_TEST_ASSERT(job != NULL);
    if (job) {
        while (job->running) {
            job->poll();
            usleep(2000);
        }
        AGENT_TEST_ASSERT(job->bytes == 36000);
        for (int leg = 0; leg < 2; leg++) {   /* the first observation shows the head, the second the tail */
            char *obs = job->observation(true);
            const char *body = strstr(obs, leg ? "<tail " : "<head ");
            AGENT_TEST_ASSERT(body != NULL && strstr(body, "\xe2\x80\x9c") != NULL);
            AGENT_TEST_ASSERT(test_utf8_well_formed(obs));
            const char *first = body ? strstr(body, ">\n") : NULL;
            const char *last = first ? strstr(first, leg ? "\n</tail>" : "\n</head>") : NULL;
            AGENT_TEST_ASSERT(first && last && (size_t)(last - first - 2) ==
                              (size_t)(leg ? AGENT_BASH_TAIL_BYTES : AGENT_BASH_HEAD_BYTES) / 3 * 3);
            free(obs);
        }
        agent_bash_remove_job(&w, job);
    }
    pthread_mutex_destroy(&w.mu);
}

/** @brief Whether a process is gone: no such pid, or a zombie nobody has reaped yet.
 *  @param pid the process
 *  @return true when it no longer runs */
static bool test_pid_dead(pid_t pid) {
    if (kill(pid, 0) != 0) return true;
    char name[64], stat[512];
    snprintf(name, sizeof name, "/proc/%ld/stat", (long)pid);
    FILE *fp = fopen(name, "rb");
    if (!fp) return true;
    const size_t n = fread(stat, 1, sizeof stat - 1, fp);
    fclose(fp);
    stat[n] = '\0';
    const char *state = strrchr(stat, ')');   /* "pid (comm) S ..." */
    return state && state[1] == ' ' && state[2] == 'Z';
}

/** L373: no way out of the process leaves a bash job running or its output file behind -- the teardown itself, the
 *  table's cap, and a child process that leaves by exit() and by each handled signal. */
static void test_agent_bash_teardown(void) {
    agent_worker w;
    memset(&w, 0, sizeof(w));
    pthread_mutex_init(&w.mu, NULL);
    w.wake_pending = true;   /* no wake pipe */
    char err[160] = {0};

    agent_bash_job *job = agent_bash_start(&w, "sleep 300 & sleep 300", 600, err, sizeof err);
    AGENT_TEST_ASSERT(job != NULL);
    if (job) {
        char path[sizeof job->path];
        snprintf(path, sizeof path, "%s", job->path);
        AGENT_TEST_ASSERT(access(path, F_OK) == 0 && !test_pid_dead(job->pid));
        agent_bash_teardown();
        for (int i = 0; job->running && i < 2500; i++) {
            job->poll();
            usleep(2000);
        }
        AGENT_TEST_ASSERT(!job->running && job->exit_status == 128 + SIGKILL && !job->timed_out);
        AGENT_TEST_ASSERT(access(path, F_OK) != 0);
        agent_bash_remove_job(&w, job);
    }

    /* agent_bash_jobs_poll, what the worker runs every tool round and idle second: with nobody observing them, a job
     * that writes four pipes' worth runs to its end, and one past its timeout_sec is killed */
    agent_bash_job *writer = agent_bash_start(&w, "head -c 262144 /dev/zero | tr '\\0' x; echo end", 600, err, sizeof err);
    agent_bash_job *sleeper = agent_bash_start(&w, "sleep 300", 1, err, sizeof err);
    AGENT_TEST_ASSERT(writer != NULL && sleeper != NULL);
    if (writer && sleeper) {
        for (int i = 0; i < 5000 && agent_bash_jobs_poll(&w); i++) usleep(2000);
        AGENT_TEST_ASSERT(!writer->running && writer->exit_status == 0 && writer->bytes == 262144 + 4);
        AGENT_TEST_ASSERT(!sleeper->running && sleeper->timed_out);
        AGENT_TEST_ASSERT(!agent_bash_jobs_poll(&w));
    }
    agent_bash_jobs_free(&w);

    /* a sync's prefill chunk polls the jobs on the worker thread only (worker_progress_cb): the list is that
     * thread's, and a sync the UI thread runs (a session switch) leaves it to the worker's idle tick */
    agent_bash_job *timed = agent_bash_start(&w, "sleep 300", 1, err, sizeof err);
    AGENT_TEST_ASSERT(timed != NULL);
    if (timed) {
        usleep(1100 * 1000);
        AGENT_TEST_ASSERT(!agent_on_worker_thread(&w));           /* no worker thread: this one is "the UI's" */
        worker_progress_cb(&w, "prefill_chunk", 10, 100);
        AGENT_TEST_ASSERT(timed->running && !timed->timed_out);   /* not polled */
        w.worker_tid = pthread_self();
        w.worker_running = true;
        AGENT_TEST_ASSERT(agent_on_worker_thread(&w));
        worker_progress_cb(&w, "prefill_display", 10, 100);       /* the display event polls nothing */
        AGENT_TEST_ASSERT(timed->running && !timed->timed_out);
        for (int i = 0; timed->running && i < 2500; i++) {
            worker_progress_cb(&w, "prefill_chunk", 10, 100);
            usleep(2000);
        }
        AGENT_TEST_ASSERT(!timed->running && timed->timed_out);
        w.worker_running = false;
    }
    agent_bash_jobs_free(&w);

    for (int i = 0; i < AGENT_BASH_JOBS_MAX; i++) AGENT_TEST_ASSERT(agent_bash_start(&w, "true", 600, err, sizeof err) != NULL);
    AGENT_TEST_ASSERT(agent_bash_start(&w, "true", 600, err, sizeof err) == NULL && strstr(err, "already tracked") != NULL);
    agent_bash_jobs_free(&w);
    AGENT_TEST_ASSERT(w.bash_jobs == NULL && agent_bash_live_free_slot() == 0);

    {   /* the fork-to-table window: a job's child whose parent is gone before it is told to go runs nothing */
        char mark[64];
        snprintf(mark, sizeof mark, "/tmp/pulsar-agent-gate-%ld", (long)getpid());
        char cmd[128];
        snprintf(cmd, sizeof cmd, "touch %s", mark);
        fflush(NULL);
        const pid_t child = fork();
        if (child == 0) {
            /* the handler's teardown at the worst moment: the instant the job's child exists */
            pthread_atfork(NULL, [] { agent_bash_teardown(); _exit(0); }, NULL);
            (void)agent_bash_start(&w, cmd, 600, err, sizeof err);
            _exit(97);   /* not reached: the fork's parent side exits */
        }
        int status = 0;
        AGENT_TEST_ASSERT(child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                          WEXITSTATUS(status) == 0);
        usleep(200000);   /* the orphaned job child reads EOF on its gate and exits */
        AGENT_TEST_ASSERT(access(mark, F_OK) != 0);
        unlink(mark);
    }

    static const int ways[] = {0, SIGTERM, SIGHUP, SIGINT};   /* 0: exit(), the atexit path */
    for (int sig : ways) {
        int fds[2], go[2];   /* the child reports its job on fds, and leaves by exit() when told on go */
        AGENT_TEST_ASSERT(pipe(fds) == 0 && pipe(go) == 0);
        fflush(NULL);
        const pid_t child = fork();
        if (child == 0) {
            close(fds[0]);
            close(go[1]);
            agent_bash_teardown_install(true);
            agent_bash_job *j = agent_bash_start(&w, "sleep 300", 600, err, sizeof err);
            if (!j) _exit(99);
            if (write(fds[1], &j->pid, sizeof j->pid) != (ssize_t)sizeof j->pid ||
                write(fds[1], j->path, sizeof j->path) != (ssize_t)sizeof j->path)
                _exit(98);
            char told;
            if (!sig && read(go[0], &told, 1) == 1) exit(0);
            for (;;) pause();
        }
        close(fds[1]);
        close(go[0]);
        pid_t job_pid = 0;
        char path[sizeof job->path] = {0};
        AGENT_TEST_ASSERT(child > 0 && read(fds[0], &job_pid, sizeof job_pid) == (ssize_t)sizeof job_pid &&
                          read(fds[0], path, sizeof path) == (ssize_t)sizeof path);
        close(fds[0]);
        if (child > 0 && job_pid > 0) AGENT_TEST_ASSERT(access(path, F_OK) == 0 && !test_pid_dead(job_pid));
        if (child > 0 && sig) AGENT_TEST_ASSERT(kill(child, sig) == 0);
        if (child > 0 && !sig) AGENT_TEST_ASSERT(write(go[1], "x", 1) == 1);
        close(go[1]);
        if (child <= 0 || job_pid <= 0) continue;
        int status = 0;
        AGENT_TEST_ASSERT(waitpid(child, &status, 0) == child);
        if (sig) AGENT_TEST_ASSERT(WIFSIGNALED(status) && WTERMSIG(status) == sig);   /* still dies by the signal */
        else AGENT_TEST_ASSERT(WIFEXITED(status) && WEXITSTATUS(status) == 0);
        bool dead = false;
        for (int i = 0; !dead && i < 2500; i++) {
            dead = test_pid_dead(job_pid);
            if (!dead) usleep(2000);
        }
        AGENT_TEST_ASSERT(dead);
        AGENT_TEST_ASSERT(access(path, F_OK) != 0);
    }
    pthread_mutex_destroy(&w.mu);
}
#endif

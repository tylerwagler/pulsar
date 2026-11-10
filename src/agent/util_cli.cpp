/** @file
 *  pulsar-agent: small utilities and command-line parsing. */
#include "pulsar_agent_internal.h"
#include "pulsar_utf8.h"
#include "pulsar_argparse.h"



/* ============================================================================
 * Small Utilities And Command-Line Parsing
 * ============================================================================
 */

void agent_sigint_handler(int sig) {
    (void)sig;
    agent_sigint = 1;
}



void *agent_xmalloc(size_t n) {
    void *p = (void *)malloc(n ? n : 1);
    if (!p) {
        perror("pulsar-agent: malloc");
        exit(1);
    }
    return p;
}



char *xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *p = (char *)agent_xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}



char *xstrndup(const char *s, size_t n) {
    char *p = (char *)agent_xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}



void *agent_xrealloc(void *ptr, size_t n) {
    void *p = (void *)realloc(ptr, n ? n : 1);
    if (!p) {
        perror("pulsar-agent: realloc");
        exit(1);
    }
    return p;
}



const char *agent_utf8_skip_continuation(const char *p, const char *end) {
    while (p < end && (((unsigned char)*p) & 0xc0) == 0x80) p++;
    return p;
}



size_t agent_utf8_complete_len(const char *s, size_t n) {
    size_t lead = n;   /* the last byte that is not a continuation byte, looking back one sequence */
    while (lead > 0 && n - lead < 4) {
        lead--;
        if ((((unsigned char)s[lead]) & 0xc0) != 0x80) {
            return (size_t)utf8_seq_len((unsigned char)s[lead]) > n - lead ? lead : n;
        }
    }
    return n;
}



bool agent_write_file_atomic(const char *path, const void *data, size_t len, mode_t create_mode,
                             char *err, size_t err_len) {
    char real[PATH_MAX];
    const char *target = realpath(path, real) ? real : path;   /* a link that resolves is written through */
    struct stat st;
    const bool exists = stat(target, &st) == 0;
    static unsigned counter;
    char *tmp = NULL;
    int fd = -1;
    for (int attempt = 0; fd < 0 && attempt < 16; attempt++) {
        free(tmp);
        agent_buf name = {0};
        char suffix[64];
        snprintf(suffix, sizeof suffix, ".tmp.%ld.%u", (long)getpid(), counter++);
        agent_buf_puts(&name, target);
        agent_buf_puts(&name, suffix);
        tmp = agent_buf_take(&name);
        /* O_EXCL | O_NOFOLLOW: the temp is a new regular file here, never something planted under its name */
        fd = open(tmp, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, create_mode);
        if (fd < 0 && errno != EEXIST) break;
    }
    if (fd < 0) {
        snprintf(err, err_len, "create a temporary file beside %s: %s", path, strerror(errno));
        free(tmp);
        return false;
    }
    const char *step = NULL;
    if (exists && fchmod(fd, st.st_mode & 07777) != 0) step = "chmod";
    for (size_t off = 0; !step && off < len;) {
        const ssize_t w = write(fd, (const char *)data + off, len - off);
        if (w > 0) off += (size_t)w;
        else if (w == 0 || errno != EINTR) step = "write";
    }
    if (!step && fsync(fd) != 0) step = "fsync";
    int saved_errno = errno;
    if (close(fd) != 0 && !step) {
        step = "close";
        saved_errno = errno;
    }
    if (!step && rename(tmp, target) != 0) {
        step = "rename over";
        saved_errno = errno;
    }
    if (step) {
        unlink(tmp);
        snprintf(err, err_len, "%s %s: %s", step, path, strerror(saved_errno));
    }
    free(tmp);
    return !step;
}



void write_all(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t wr = write(fd, p, n);
        if (wr < 0) {
            if (errno == EINTR) continue;
            return;
        }
        p += wr;
        n -= (size_t)wr;
    }
}



void agent_input_buf_append(agent_input_buf *b, const char *s, size_t n) {
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        b->ptr = (char *)agent_xrealloc(b->ptr, cap);
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}



char *agent_input_buf_take(agent_input_buf *b) {
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}



void agent_input_buf_free(agent_input_buf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}



/** @brief Whether `cmd` is the slash command `name`, alone or followed by whitespace and arguments.
 *  @param cmd  the input line
 *  @param name the command, e.g. "/switch"
 *  @return true on a match */
static bool agent_slash_command_with_args(const char *cmd, const char *name) {
    size_t len = strlen(name);
    return !strncmp(cmd, name, len) &&
           (cmd[len] == '\0' || isspace((unsigned char)cmd[len]));
}



bool agent_slash_command_known(const char *cmd) {
    return !strcmp(cmd, "/help") ||
           !strcmp(cmd, "/save") ||
           !strcmp(cmd, "/compact") ||
           !strcmp(cmd, "/list") ||
           !strcmp(cmd, "/quit") ||
           !strcmp(cmd, "/exit") ||
           !strcmp(cmd, "/new") ||
           agent_slash_command_with_args(cmd, "/switch") ||
           agent_slash_command_with_args(cmd, "/del") ||
           agent_slash_command_with_args(cmd, "/strip") ||
           agent_slash_command_with_args(cmd, "/history");
}



/** @brief The backend the agent opens when none is named.
 *  @return PULSAR_BACKEND_CUDA */
static pulsar_backend default_backend(void) {
    return PULSAR_BACKEND_CUDA;
}



double agent_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}



/// @brief Print pulsar-agent's help (PULSAR_HELP_AGENT), or one topic of it.
/// @param fp    the stream (stdout for --help, stderr after a bad option)
/// @param topic the topic asked (NULL: the overview)
static void usage(FILE *fp, const char *topic) {
    pulsar_help_print(fp, PULSAR_HELP_AGENT, topic);
}



agent_config parse_options(int argc, char **argv) {
    agent_config c = {
        .engine = {
            .model_path = "model",
            /* agent's per-token DSML/forcing loop can't consume speculative
             * chunks yet; disable the merged drafter until that is wired */
            .dspark_disable = true,
            .backend = default_backend(),
        },
        .gen = {
            .system = "You are a helpful coding assistant running inside pulsar-agent.",
            .n_predict = 50000,
            .ctx_size = 100000,
            .temperature = PULSAR_DEFAULT_TEMPERATURE,
            .top_p = PULSAR_DEFAULT_TOP_P,
            .min_p = PULSAR_DEFAULT_MIN_P,
            /* the loaded family's default effort, resolved after open (as pulsar's) */
            .think_mode = PULSAR_THINK_DEFAULT,
        },
    };

    bool steering_scale_set = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        if (!strcmp(arg, "-p") || !strcmp(arg, "--prompt")) {
            c.gen.prompt = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--non-interactive")) {
            c.non_interactive = true;
        } else if (!strcmp(arg, "--allow-outside-workspace")) {
            c.allow_outside_workspace = true;
        } else if (!strcmp(arg, "--allow-bash")) {
            c.allow_bash = true;
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.gen.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--trace")) {
            c.gen.trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.engine.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.gen.ctx_size = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.gen.n_predict = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--temp")) {
            c.gen.temperature = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
        } else if (!strcmp(arg, "--top-p")) {
            c.gen.top_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--min-p")) {
            c.gen.min_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--seed")) {
            c.gen.seed = parse_u64(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--think")) {
            /* thinking on at the LOADED family's default effort, resolved after open */
            c.gen.think_mode = PULSAR_THINK_DEFAULT;
            c.gen.think_mode_set = false;
        } else if (!strcmp(arg, "--think-low")) {
            c.gen.think_mode = PULSAR_THINK_LOW;
            c.gen.think_mode_set = true;
        } else if (!strcmp(arg, "--think-effort")) {
            c.gen.think_mode = parse_int_range(need_arg(&i, argc, argv, arg), arg,
                                               PULSAR_THINK_EFFORT_MIN, PULSAR_THINK_EFFORT_MAX);
            c.gen.think_mode_set = true;
        } else if (!strcmp(arg, "--think-high")) {
            c.gen.think_mode = PULSAR_THINK_HIGH;
            c.gen.think_mode_set = true;
        } else if (!strcmp(arg, "--think-max")) {
            c.gen.think_mode = PULSAR_THINK_MAX;
            c.gen.think_mode_set = true;
        } else if (!strcmp(arg, "--nothink")) {
            c.gen.think_mode = PULSAR_THINK_NONE;
            c.gen.think_mode_set = true;
        } else if (!strcmp(arg, "--chdir")) {
            c.chdir_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prefill-chunk")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "pulsar-agent: --prefill-chunk must be positive\n");
                exit(2);
            }
            c.engine.prefill_chunk = (uint32_t)v;
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            steering_scale_set = true;
        } else {
            fprintf(stderr, "pulsar-agent: unknown option: %s\n", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }

    if (c.engine.directional_steering_file && !steering_scale_set)
        c.engine.directional_steering_ffn = 1.0f;
    return c;
}





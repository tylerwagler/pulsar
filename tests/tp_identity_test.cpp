/*
 * TP identity / hello-robustness test (branch tensor_parallel).
 *
 * Host-only: no CUDA, no RDMA -- the pair rides the transport's full-duplex
 * TCP fallback over 127.0.0.1.  Every check runs in its own forked child, and
 * every child arms alarm() before create() while the parent waits with a
 * bounded poll, so a regression that turns a clean failure into a SILENT HANG
 * (the leader's accept() has no timeout) is reported as a FAILURE instead of
 * blocking the suite forever.
 *
 * Three groups:
 *
 *  A. Real-rank identity mismatch matrix.  One side carries the base DS
 *     identity; the other mutates exactly one compared field (gguf_bytes,
 *     model_id, n_layer, n_embd, n_vocab, quant_bits, gate_slot_start,
 *     gate_slot_step, gates_per_token).  Both ranks must FAIL create() loudly,
 *     each with err containing "mismatch", and neither may hang.  Runs every
 *     field with the mutation on the leader and again on the worker.
 *
 *  B. ctx_size is deliberately NOT compared (upstream behavior: the engine
 *     reads the peer's ctx rather than requiring equality).  Two ranks with
 *     different ctx must CONNECT successfully.  Guards against someone
 *     "tightening" the identity check into a mismatch later.
 *
 *  C. Raw-peer robustness on the leader.  The wire hello struct
 *     (pulsar_tp_hello_fixed) is public, so a hand-written peer can reach the
 *     checks no two API-created ranks ever can: flipped magic, bumped
 *     version, a same-role collision, a model field, plus truncated / empty /
 *     garbage bodies.  The leader must fail each one cleanly, never block in
 *     accept/hello.  Plus one worker dial-timeout check against a dead port
 *     (PULSAR_TP_TIMEOUT_SEC=2) proving the 300s default never turns a wrong
 *     address into a session-long wait.
 */
#include <arpa/inet.h>
#include <cerrno>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "tp/pulsar_tp.h"

static int g_failures = 0;
#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            g_failures++;                                                    \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);        \
            std::fprintf(stderr, __VA_ARGS__);                               \
            std::fprintf(stderr, "\n");                                      \
        }                                                                    \
    } while (0)

#define BASE_GGUF 87000000000ull
#define BASE_MODEL 3u
#define BASE_LAYERS 43u
#define BASE_EMBD 4096u
#define BASE_VOCAB 129280u
#define BASE_QBITS 2u
#define BASE_CTX 1048576u

static double tp_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ------------------------------------------------------------------------
 * Mutation table (group A): the nine fields compared by the hello check.
 * ---------------------------------------------------------------------- */

typedef struct {
    const char *name;
    void (*apply)(pulsar_tp_identity *);
} mutation;

static void mut_gguf(pulsar_tp_identity *i) { i->gguf_bytes += 1; }
static void mut_model(pulsar_tp_identity *i) { i->model_id += 1; }
static void mut_layers(pulsar_tp_identity *i) { i->n_layer += 1; }
static void mut_embd(pulsar_tp_identity *i) { i->n_embd += 1; }
static void mut_vocab(pulsar_tp_identity *i) { i->n_vocab += 1; }
static void mut_qbits(pulsar_tp_identity *i) { i->quant_bits += 1; }
static void mut_gss(pulsar_tp_identity *i) { i->gate_slot_start += 1; }
static void mut_gstep(pulsar_tp_identity *i) { i->gate_slot_step += 1; }
static void mut_gpt(pulsar_tp_identity *i) { i->gates_per_token += 1; }

static const mutation MUTATIONS[] = {
    { "gguf_bytes", mut_gguf },       { "model_id", mut_model },
    { "n_layer", mut_layers },        { "n_embd", mut_embd },
    { "n_vocab", mut_vocab },         { "quant_bits", mut_qbits },
    { "gate_slot_start", mut_gss },   { "gate_slot_step", mut_gstep },
    { "gates_per_token", mut_gpt },
};
#define N_MUT ((int)(sizeof(MUTATIONS) / sizeof(MUTATIONS[0])))

/* ------------------------------------------------------------------------
 * Child side.  run_child() never returns: it writes a marker to stdout and
 * _exit()s 0 on the expected outcome or 1 on any deviation.  The parent's
 * assertions are: exit status 0 AND the marker contains the expected token.
 * marker output is written with write() so _exit() cannot lose it.
 * ---------------------------------------------------------------------- */

static void marker(bool fail, const char *fmt, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) n = 0;
    if ((size_t)n >= sizeof(buf)) n = (int)sizeof(buf) - 1;
    if (write(STDOUT_FILENO, buf, (size_t)n) < 0) { /* best effort */ }
    if (write(STDOUT_FILENO, "\n", 1) < 0) { /* best effort */ }
    _exit(fail ? 1 : 0);
}

static void child_run(int rank, int port) {
    alarm(20); /* any hang past 20s kills the child = failure, not a block */
    char err[256] = "";

    pulsar_tp_identity id;
    pulsar_tp_identity_init_defaults(&id, BASE_GGUF, BASE_MODEL, BASE_LAYERS,
                                     BASE_EMBD, BASE_VOCAB, BASE_QBITS,
                                     BASE_CTX);

    const char *muts = getenv("TP_ID_TEST_MUT");      /* "-1" or mutation index */
    const char *side = getenv("TP_ID_TEST_MUT_SIDE"); /* "0"/"1" */
    const char *ctxdiff = getenv("TP_ID_TEST_CTXDIFF");
    const char *rawmode = getenv("TP_ID_TEST_RAW");
    const char *dialdead = getenv("TP_ID_TEST_DIALDEAD");

    if (muts && muts[0] != '-') {
        char *end = NULL;
        const long idx = strtol(muts, &end, 10);
        const int myside = side ? atoi(side) : 0;
        if (end != muts && idx >= 0 && idx < N_MUT && myside == rank)
            MUTATIONS[idx].apply(&id);
    }
    if (ctxdiff && ctxdiff[0] == '1')
        id.ctx_size = rank == 0 ? BASE_CTX : BASE_CTX - 1; /* unequal on purpose */

    pulsar_tp_options opt = {};
    opt.role = rank == 0 ? PULSAR_TP_ROLE_LEADER : PULSAR_TP_ROLE_WORKER;
    opt.peer = rank == 0 ? NULL : "127.0.0.1";
    opt.port = port;

    pulsar_tp *tp = NULL;
    const int rc = pulsar_tp_create(&tp, &opt, &id, err, sizeof(err));

    if (dialdead && dialdead[0] == '1') {
        if (rc == 0 && strstr(err, "connect"))
            marker(0, "DIAL_OK err=%s", err);
        marker(1, "DIAL_ACCEPTED rc=%d err=%s", rc, err);
    } else if (ctxdiff && ctxdiff[0] == '1') {
        if (rc == 1 && tp) {
            pulsar_tp_free(tp);
            marker(0, "CONNECT_OK");
        }
        marker(1, "CONNECT_REJECTED rc=%d err=%s", rc, err);
    } else if (rawmode && rawmode[0] == '1') {
        if (rc == 0 && err[0])
            marker(0, "ABORT_OK err=%s", err);
        marker(1, "ABORT_ACCEPTED rc=%d err=%s", rc, err);
    } else {
        if (rc == 0 && strstr(err, "mismatch"))
            marker(0, "MISMATCH_OK err=%s", err);
        marker(1, "MISMATCH_ACCEPTED rc=%d err=%s", rc, err);
    }
}

/* ------------------------------------------------------------------------
 * Parent helpers: spawn a forked rank child, reap it with a hard bound,
 * capture its stdout marker.
 * ---------------------------------------------------------------------- */

typedef struct {
    pid_t pid;
    int pipe_rd;
} child_handle;

static bool spawn_rank(int rank, int port, child_handle *h) {
    int pfd[2];
    if (pipe(pfd) != 0) {
        CHECK(0, "pipe: %s", std::strerror(errno));
        return false;
    }
    h->pid = fork();
    if (h->pid < 0) {
        close(pfd[0]);
        close(pfd[1]);
        CHECK(0, "fork: %s", std::strerror(errno));
        return false;
    }
    if (h->pid == 0) {
        close(pfd[0]);
        dup2(pfd[1], STDOUT_FILENO);
        close(pfd[1]);
        child_run(rank, port); /* never returns */
        _exit(127);
    }
    close(pfd[1]);
    h->pipe_rd = pfd[0];
    return true;
}

/* Reap `h`, fill `out` with the child's stdout marker, and return the exit
 * status (or a negative value: -1 waitpid error, -2 hung / killed by parent,
 * -3 died by signal).  Bounded: a child that never reaps is SIGKILLed so a
 * hang regression cannot block the suite. */
static int reap_child(child_handle *h, char *out, size_t outlen) {
    out[0] = '\0';
    const double deadline = tp_now_sec() + 45.0;
    int status = 0;
    for (;;) {
        const pid_t r = waitpid(h->pid, &status, WNOHANG);
        if (r == h->pid) break;
        if (r < 0) {
            snprintf(out, outlen, "(waitpid: %s)", std::strerror(errno));
            close(h->pipe_rd);
            return -1;
        }
        if (tp_now_sec() > deadline) {
            kill(h->pid, SIGKILL);
            waitpid(h->pid, &status, 0);
            close(h->pipe_rd);
            snprintf(out, outlen, "(HANG: killed after 45s)");
            return -2;
        }
        usleep(5 * 1000);
    }
    size_t n = 0;
    for (;;) {
        const ssize_t r = read(h->pipe_rd, out + n, outlen - 1 - n);
        if (r <= 0) break;
        n += (size_t)r;
    }
    if (n < outlen - 1) out[n] = '\0';
    close(h->pipe_rd);
    if (WIFSIGNALED(status)) {
        snprintf(out + strnlen(out, outlen - 1), outlen - strnlen(out, outlen - 1),
                 " (signal %d)", WTERMSIG(status));
        return -3;
    }
    return WEXITSTATUS(status);
}

/* Assert one forked child exited 0 and its marker contains `needle`. */
static void expect_child(child_handle *h, const char *what, const char *needle) {
    char out[8192];
    const int st = reap_child(h, out, sizeof(out));
    const bool ok = st == 0 && (needle == NULL || strstr(out, needle) != NULL);
    if (!ok)
        CHECK(0, "%s: child rc=%d (want 0, marker contains \"%s\"); out: %s",
              what, st, needle ? needle : "", out);
}

static int free_port(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in a = {};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(fd, reinterpret_cast<struct sockaddr *>(&a), sizeof(a)) != 0) {
        close(fd);
        return -1;
    }
    socklen_t len = sizeof(a);
    if (getsockname(fd, reinterpret_cast<struct sockaddr *>(&a), &len) != 0) {
        close(fd);
        return -1;
    }
    const int port = ntohs(a.sin_port);
    close(fd);
    return port;
}

/* Shared per-scenario env for the two children of a pair. */
static void pair_env(int port, int mutation, int mut_side, int ctxdiff) {
    char pb[16], mb[16], sb[8];
    snprintf(pb, sizeof(pb), "%d", port);
    snprintf(mb, sizeof(mb), "%d", mutation);
    snprintf(sb, sizeof(sb), "%d", mut_side);
    setenv("TP_ID_TEST_PORT", pb, 1);
    setenv("TP_ID_TEST_MUT", mb, 1);
    setenv("TP_ID_TEST_MUT_SIDE", sb, 1);
    setenv("TP_ID_TEST_CTXDIFF", ctxdiff ? "1" : "0", 1);
    setenv("TP_ID_TEST_RAW", "0", 1);
    setenv("TP_ID_TEST_DIALDEAD", "0", 1);
}

/* ------------------------------------------------------------------------
 * Group A: the mismatch matrix, one forked pair per (mutation, mutating side).
 * ---------------------------------------------------------------------- */

static void run_mismatch_pair(int mutation, int mut_side) {
    const int port = free_port();
    CHECK(port > 0, "free_port failed");
    if (port <= 0) return;
    pair_env(port, mutation, mut_side, 0);

    child_handle cl, cw;
    if (!spawn_rank(0, port, &cl)) return;
    if (!spawn_rank(1, port, &cw)) {
        kill(cl.pid, SIGKILL);
        waitpid(cl.pid, NULL, 0);
        close(cl.pipe_rd);
        return;
    }
    char what[128];
    snprintf(what, sizeof(what), "mismatch field=%s mutated-on-rank=%d leader",
             MUTATIONS[mutation].name, mut_side);
    expect_child(&cl, what, "mismatch");
    snprintf(what, sizeof(what), "mismatch field=%s mutated-on-rank=%d worker",
             MUTATIONS[mutation].name, mut_side);
    expect_child(&cw, what, "mismatch");
}

/* ------------------------------------------------------------------------
 * Group B: ctx_size not compared -> differing ctx must still connect.
 * ---------------------------------------------------------------------- */

static void run_ctx_pair(void) {
    const int port = free_port();
    CHECK(port > 0, "free_port failed (ctx pair)");
    if (port <= 0) return;
    pair_env(port, -1, 0, 1);

    child_handle cl, cw;
    if (!spawn_rank(0, port, &cl)) return;
    if (!spawn_rank(1, port, &cw)) {
        kill(cl.pid, SIGKILL);
        waitpid(cl.pid, NULL, 0);
        close(cl.pipe_rd);
        return;
    }
    expect_child(&cl, "ctx-diff leader (must connect)", "CONNECT_OK");
    expect_child(&cw, "ctx-diff worker (must connect)", "CONNECT_OK");
}

/* ------------------------------------------------------------------------
 * Group C: raw-peer robustness on the leader, then the worker dial-timeout.
 *   The hand-written hello uses the real wire struct, so it can flip magic /
 *   version / role / model fields or send partial / empty / garbage bodies.
 * ---------------------------------------------------------------------- */

enum { RAW_CORRUPT_MAGIC = 1, RAW_CORRUPT_VERSION = 2,
       RAW_CORRUPT_ROLE = 3, RAW_CORRUPT_MODEL = 4 };
enum { RAW_SEND_FULL = 0, RAW_SEND_TRUNC, RAW_SEND_NONE, RAW_SEND_GARBAGE };

typedef struct {
    const char *name;
    const char *needle;
    int corrupt;
    int send_mode;
} raw_case;

static const raw_case RAW_CASES[] = {
    { "bad-magic",   "magic",                 RAW_CORRUPT_MAGIC,   RAW_SEND_FULL },
    { "bad-version", "version",               RAW_CORRUPT_VERSION, RAW_SEND_FULL },
    { "role-collide","role",                  RAW_CORRUPT_ROLE,    RAW_SEND_FULL },
    { "model-field", "mismatch",              RAW_CORRUPT_MODEL,   RAW_SEND_FULL },
    { "truncated",   "hello",                 RAW_CORRUPT_MODEL,   RAW_SEND_TRUNC },
    { "eof-peer",    "hello",                 RAW_CORRUPT_MODEL,   RAW_SEND_NONE },
    { "garbage",     "magic",                 RAW_CORRUPT_MODEL,   RAW_SEND_GARBAGE },
};
#define N_RAW ((int)(sizeof(RAW_CASES) / sizeof(RAW_CASES[0])))

static void craft_hello(pulsar_tp_hello_fixed *h, int corrupt) {
    std::memset(h, 0, sizeof(*h));
    h->magic = PULSAR_TP_MAGIC;
    h->version = PULSAR_TP_PROTOCOL_VERSION;
    h->role = PULSAR_TP_ROLE_WORKER; /* differs from the real leader's role */
    h->rdma_ok = 0;
    h->gguf_bytes = BASE_GGUF;
    h->model_id = BASE_MODEL;
    h->n_layer = BASE_LAYERS;
    h->n_embd = BASE_EMBD;
    h->n_vocab = BASE_VOCAB;
    h->quant_bits = BASE_QBITS;
    h->ctx_size = BASE_CTX;
    switch (corrupt) {
        case RAW_CORRUPT_MAGIC:   h->magic ^= UINT32_C(0x5a5a5a5a); break;
        case RAW_CORRUPT_VERSION: h->version = PULSAR_TP_PROTOCOL_VERSION + 1; break;
        case RAW_CORRUPT_ROLE:    h->role = PULSAR_TP_ROLE_LEADER; break;
        case RAW_CORRUPT_MODEL:   h->n_embd = BASE_EMBD + 1; break;
        default: break;
    }
}

static int raw_write(int fd, const void *buf, size_t len) {
    const char *p = static_cast<const char *>(buf);
    while (len) {
#ifdef MSG_NOSIGNAL
        const ssize_t w = send(fd, p, len, MSG_NOSIGNAL);
#else
        const ssize_t w = send(fd, p, len, 0);
#endif
        if (w < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (w == 0) return 0;
        p += w;
        len -= (size_t)w;
    }
    return 1;
}

static void run_raw_case(const raw_case *c) {
    const int port = free_port();
    CHECK(port > 0, "free_port failed (raw %s)", c->name);
    if (port <= 0) return;
    pair_env(port, -1, 0, 0);
    setenv("TP_ID_TEST_RAW", "1", 1);

    child_handle l;
    if (!spawn_rank(0, port, &l)) return;

    /* Retry-connect: the leader binds then accept()s; our connection is what
     * it accept()s.  ECONNREFUSED until the listener is up. */
    int fd = -1;
    const double deadline = tp_now_sec() + 5.0;
    while (fd < 0 && tp_now_sec() < deadline) {
        fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) break;
        struct sockaddr_in a = {};
        a.sin_family = AF_INET;
        a.sin_port = htons((uint16_t)port);
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (connect(fd, reinterpret_cast<struct sockaddr *>(&a), sizeof(a)) != 0) {
            close(fd);
            fd = -1;
            usleep(5 * 1000);
        }
    }
    if (fd < 0) {
        expect_child(&l, c->name, "ABORT_OK"); /* consume the child */
        CHECK(0, "raw %s: could not connect to leader", c->name);
        return;
    }

    pulsar_tp_hello_fixed h;
    craft_hello(&h, c->corrupt);
    switch (c->send_mode) {
        case RAW_SEND_FULL:
            CHECK(raw_write(fd, &h, sizeof(h)), "raw %s: write failed", c->name);
            usleep(200 * 1000); /* let the leader read the full frame */
            break;
        case RAW_SEND_TRUNC:
            CHECK(raw_write(fd, &h, sizeof(h) / 2 + 3),
                  "raw %s: truncated write failed", c->name);
            break;
        case RAW_SEND_GARBAGE: {
            uint8_t junk[sizeof(h)];
            std::memset(junk, 0xab, sizeof(junk));
            CHECK(raw_write(fd, junk, sizeof(junk)), "raw %s: garbage write failed",
                  c->name);
            usleep(200 * 1000);
            break;
        }
        case RAW_SEND_NONE:
        default:
            break; /* connect then close = EOF */
    }
    close(fd);

    char what[128];
    snprintf(what, sizeof(what), "raw-peer %s", c->name);
    /* Assert the SPECIFIC check that fired (magic/version/role/mismatch/
     * read-failure), so a garbage client aborting for the wrong reason still
     * fails the test.  "ABORT_OK err=..." carries the reason in the marker. */
    expect_child(&l, what, c->needle);
}

/* Worker dial against a bound-but-not-listening socket: must give up within
 * PULSAR_TP_TIMEOUT_SEC (set to 2 by main), failing with a connect error. */
static void run_dial_dead(void) {
    const int port = free_port();
    CHECK(port > 0, "free_port failed (dial-dead)");
    if (port <= 0) return;

    /* Hold the port bound-but-not-listening so connect() gets ECONNREFUSED
     * deterministically for the whole 2s dial window. */
    const int hold = socket(AF_INET, SOCK_STREAM, 0);
    if (hold < 0) {
        CHECK(0, "dial-dead: socket: %s", std::strerror(errno));
        return;
    }
    struct sockaddr_in a = {};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons((uint16_t)port);
    if (bind(hold, reinterpret_cast<struct sockaddr *>(&a), sizeof(a)) != 0) {
        CHECK(0, "dial-dead: bind: %s", std::strerror(errno));
        close(hold);
        return;
    }

    pair_env(port, -1, 0, 0);
    setenv("TP_ID_TEST_DIALDEAD", "1", 1);

    child_handle w;
    if (!spawn_rank(1, port, &w)) {
        close(hold);
        return;
    }
    expect_child(&w, "dial-dead port", "DIAL_OK");
    close(hold);
}

int main(void) {
    setenv("PULSAR_TP_TIMEOUT_SEC", "2", 1); /* every dial is bounded *now* */
    alarm(600); /* absolute backstop against an orchestrator bug */

    for (int side = 0; side < 2; side++)
        for (int m = 0; m < N_MUT; m++) run_mismatch_pair(m, side);
    run_ctx_pair();
    for (int r = 0; r < N_RAW; r++) run_raw_case(&RAW_CASES[r]);
    run_dial_dead();

    alarm(0);
    if (g_failures) {
        std::fprintf(stderr, "tp_identity_test: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf(
        "tp_identity_test: ok (%d mismatch fields x both sides, ctx-diff connect, "
        "%d raw peers, dial-timeout; no hangs)\n",
        N_MUT * 2, N_RAW);
    return 0;
}

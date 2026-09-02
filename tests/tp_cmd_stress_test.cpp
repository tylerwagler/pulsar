/*
 * TP control-plane stress test (branch tensor_parallel).
 *
 * Host-only: no CUDA, no RDMA -- TCP loopback, forked leader/worker.
 *
 * transport_test's frames_phase runs every control frame exactly once.  This
 * test hammers the control plane with thousands of randomized, interleaved
 * session commands and validates the two things a real engine session depends
 * on:
 *
 *   1. The worker's command dispatcher (recv_command -> ack) stays in sync
 *      under sustained varied load: every op references only live sessions,
 *      destroy/create churn keeps the pool ledger consistent on BOTH ranks,
 *      and per-command payloads (tokens, batches, mixed promps, drafts) are
 *      well-formed and correctly sized.
 *   2. The ack convention is uniform -- every ack'd frame round-trips through
 *      wait_command_ack with a session id both sides derive from the frame the
 *      same way (EVAL_BATCH/MIXED_BATCH ack on items[0].session_id, since the
 *      wire leaves command->session_id unset for those).
 *
 * The worker mirrors the session ledger from the received commands themselves
 * (create adds a slot, destroy removes one), so any dropped/reordered/foreign
 * command fails the CHECK -- it can't silently diverge.
 */
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
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

#define POOL_SZ 8
#define BASE_CTX 1048576u
#define OP_STEPS 3000
#define ID_BASE 0x100000ULL

static double tp_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* xorshift64* — tiny, deterministic, good enough to shape the test. */
typedef struct { uint64_t s; } gen;
static uint64_t gen_next(gen *g) {
    g->s ^= g->s >> 12;
    g->s ^= g->s << 25;
    g->s ^= g->s >> 27;
    return g->s * 0x2545F4914F6CDD1DULL;
}

/* Shared session pool: slot i owns session id ID_BASE + i.  The worker drives
 * its ledger purely from received commands, so it cannot silently diverge. */
static bool g_live[POOL_SZ];
static uint64_t g_seq[POOL_SZ]; /* per-session monotonic seq (leader side) */

static int pool_live_count(void);

static bool pool_is_live(uint64_t sid) {
    const uint64_t slot = sid - ID_BASE;
    return slot < POOL_SZ && g_live[slot];
}
static int pool_pick_live(gen *g) {
    int live[POOL_SZ], n = 0;
    for (int i = 0; i < POOL_SZ; i++)
        if (g_live[i]) live[n++] = i;
    if (n == 0) return -1;
    return live[(unsigned)(gen_next(g) % (uint64_t)n)];
}
static int pool_pick_dead(gen *g) {
    int dead[POOL_SZ], n = 0;
    for (int i = 0; i < POOL_SZ; i++)
        if (!g_live[i]) dead[n++] = i;
    if (n == 0) return -1; /* pool full */
    return dead[(unsigned)(gen_next(g) % (uint64_t)n)];
}

static int create_rank(int rank, int port, pulsar_tp **out, char *err,
                       size_t errlen) {
    pulsar_tp_options opt = {};
    opt.role = rank == 0 ? PULSAR_TP_ROLE_LEADER : PULSAR_TP_ROLE_WORKER;
    opt.peer = rank == 0 ? NULL : "127.0.0.1";
    opt.port = port;
    pulsar_tp_identity id = {};
    id.gguf_bytes = 87000000000ull;
    id.model_id = 3u;
    id.n_layer = 43u;
    id.n_embd = 4096u;
    id.n_vocab = 129280u;
    id.quant_bits = 2u;
    id.ctx_size = BASE_CTX;
    return pulsar_tp_create(out, &opt, &id, err, errlen);
}

/* ------------------------------------------------------------------------
 * Worker: reactive dispatcher with a ledger mirror.
 * ---------------------------------------------------------------------- */

static int worker_ack(pulsar_tp *tp, const pulsar_tp_command *cmd, uint64_t sid) {
    if (!pulsar_tp_send_command_ack(tp, sid, 0)) {
        CHECK(0, "worker ack failed (frame %d, sid %llu)", (int)cmd->type,
              (unsigned long long)sid);
        return 0;
    }
    return 1;
}

static void worker_run(pulsar_tp *tp) {
    std::memset(g_live, 0, sizeof(g_live));
    char err[256];
    int commands = 0;
    for (;;) {
        pulsar_tp_command cmd;
        CHECK(pulsar_tp_recv_command(tp, &cmd, err, sizeof(err)) == 1,
              "worker recv #%d: %s", commands, err);
        if (cmd.type == PULSAR_TP_FRAME_STOP) {
            pulsar_tp_command_free(&cmd);
            break;
        }
        switch (cmd.type) {
        case PULSAR_TP_FRAME_SESSION_CREATE: {
            CHECK(pool_is_live(cmd.session_id) == false &&
                      cmd.value == BASE_CTX,
                  "worker create dup/ctx sid=%llu v=%d",
                  (unsigned long long)cmd.session_id, cmd.value);
            g_live[cmd.session_id - ID_BASE] = true;
            worker_ack(tp, &cmd, cmd.session_id);
            break;
        }
        case PULSAR_TP_FRAME_SESSION_DESTROY:
        case PULSAR_TP_FRAME_INVALIDATE:
            CHECK(pool_is_live(cmd.session_id),
                  "worker %s on dead session %llu",
                  cmd.type == PULSAR_TP_FRAME_SESSION_DESTROY ? "destroy"
                                                              : "invalidate",
                  (unsigned long long)cmd.session_id);
            if (cmd.type == PULSAR_TP_FRAME_SESSION_DESTROY)
                g_live[cmd.session_id - ID_BASE] = false;
            worker_ack(tp, &cmd, cmd.session_id);
            break;
        case PULSAR_TP_FRAME_REWIND:
            CHECK(pool_is_live(cmd.session_id) && cmd.value > 0,
                  "worker rewind sid=%llu pos=%d",
                  (unsigned long long)cmd.session_id, cmd.value);
            worker_ack(tp, &cmd, cmd.session_id);
            break;
        case PULSAR_TP_FRAME_EVAL:
            CHECK(pool_is_live(cmd.session_id) && cmd.seq > 0 ,
                  "worker eval sid=%llu seq=%llu",
                  (unsigned long long)cmd.session_id,
                  (unsigned long long)cmd.seq);
            worker_ack(tp, &cmd, cmd.session_id);
            break;
        case PULSAR_TP_FRAME_SYNC:
            CHECK(pool_is_live(cmd.session_id) && cmd.tokens &&
                      cmd.n_tokens >= 1 && cmd.n_tokens <= 16,
                  "worker sync sid=%llu n=%u",
                  (unsigned long long)cmd.session_id, cmd.n_tokens);
            worker_ack(tp, &cmd, cmd.session_id);
            break;
        case PULSAR_TP_FRAME_VERIFY: {
            CHECK(pool_is_live(cmd.session_id) && cmd.tokens &&
                      cmd.n_tokens >= 1 && cmd.n_tokens <= 8,
                  "worker verify sid=%llu n=%u",
                  (unsigned long long)cmd.session_id, cmd.n_tokens);
            worker_ack(tp, &cmd, cmd.session_id);
            /* VERIFY_COMMIT is a one-way in-band frame recv_command does NOT
             * decode -- the worker must pull it here, right after the ack,
             * or the next recv_command sees "invalid frame type 12". */
            int32_t full = -1, replay = -1;
            CHECK(pulsar_tp_recv_verify_commit(tp, &full, &replay) == 1,
                  "worker verify_commit recv");
            CHECK(full == 0 || full == 1,
                  "worker verify_commit full=%d", full);
            CHECK(replay >= 0 && replay < 5,
                  "worker verify_commit replay=%d", replay);
            break;
        }
        case PULSAR_TP_FRAME_EVAL_BATCH: {
            CHECK(cmd.items && cmd.n_items >= 1 && cmd.n_items <= 8,
                  "worker eval_batch n=%u", cmd.n_items);
            int all_live = 1;
            for (uint32_t i = 0; i < cmd.n_items; i++)
                if (!pool_is_live(cmd.items[i].session_id)) all_live = 0;
            CHECK(all_live, "worker eval_batch referenced dead session");
            worker_ack(tp, &cmd, cmd.items ? cmd.items[0].session_id : 0u);
            break;
        }
        case PULSAR_TP_FRAME_MIXED_BATCH: {
            CHECK(cmd.tokens && cmd.n_tokens >= 1 && cmd.n_tokens <= 16 &&
                      cmd.items && cmd.n_items >= 1 && cmd.n_items <= 4,
                  "worker mixed prompt=%u items=%u", cmd.n_tokens,
                  cmd.n_items);
            int all_live = pool_is_live(cmd.session_id);
            for (uint32_t i = 0; i < cmd.n_items; i++)
                if (!pool_is_live(cmd.items[i].session_id)) all_live = 0;
            CHECK(all_live, "worker mixed referenced dead session");
            worker_ack(tp, &cmd, cmd.items ? cmd.items[0].session_id : 0u);
            break;
        }
        default:
            CHECK(0, "worker unexpected frame type %d", (int)cmd.type);
            break;
        }
        pulsar_tp_command_free(&cmd);
        commands++;
    }
    std::printf("worker: %d commands, pool live=%d\n", commands,
                (int)pool_live_count());
    std::fflush(stdout);
}

static int pool_live_count(void) {
    int n = 0;
    for (int i = 0; i < POOL_SZ; i++)
        if (g_live[i]) n++;
    return n;
}

/* ------------------------------------------------------------------------
 * Leader: randomized session-sequence driver.
 * ---------------------------------------------------------------------- */

static int leader_step(pulsar_tp *tp, gen *g, char *err, size_t errlen) {
    unsigned op = (unsigned)(gen_next(g) % 12);
    /* The pool is a finite birth-death process: never strand it at 0 (every
     * live-requiring op would mis-fire) and never kill the last live session.
     * A full-pool create is a harmless no-op, not an error. */
    const int nlive = pool_live_count();
    if (nlive == 0) op = 10;
    if (op == 11 && nlive <= 1) op = 10;

    if (op <= 2) { /* sync */
        const int s = pool_pick_live(g);
        if (s < 0) return 0;
        const uint32_t n = 1u + (uint32_t)(gen_next(g) % 8);
        int toks[16];
        for (uint32_t i = 0; i < n; i++) toks[i] = 1000 + s * 16 + (int)i;
        if (!pulsar_tp_send_sync(tp, ID_BASE + s, toks, n)) return 0;
        return pulsar_tp_wait_command_ack(tp, ID_BASE + s, "sync", err, errlen);
    }
    if (op <= 4) { /* eval */
        const int s = pool_pick_live(g);
        if (s < 0) return 0;
        const uint64_t seq = ++g_seq[s];
        if (!pulsar_tp_send_eval(tp, ID_BASE + s, seq, 2000 + s)) return 0;
        return pulsar_tp_wait_command_ack(tp, ID_BASE + s, "eval", err, errlen);
    }
    if (op == 5) { /* rewind */
        const int s = pool_pick_live(g);
        if (s < 0) return 0;
        if (!pulsar_tp_send_rewind(tp, ID_BASE + s, 1 + (int)(gen_next(g) % 100)))
            return 0;
        return pulsar_tp_wait_command_ack(tp, ID_BASE + s, "rewind", err, errlen);
    }
    if (op == 6) { /* invalidate */
        const int s = pool_pick_live(g);
        if (s < 0) return 0;
        if (!pulsar_tp_send_invalidate(tp, ID_BASE + s)) return 0;
        return pulsar_tp_wait_command_ack(tp, ID_BASE + s, "invalidate", err, errlen);
    }
    if (op == 7) { /* verify + commit */
        const int s = pool_pick_live(g);
        if (s < 0) return 0;
        const uint32_t n = 1u + (uint32_t)(gen_next(g) % 8);
        int drafts[8];
        for (uint32_t i = 0; i < n; i++) drafts[i] = 3000 + s * 8 + (int)i;
        if (!pulsar_tp_send_verify(tp, ID_BASE + s, drafts, n)) return 0;
        if (!pulsar_tp_wait_command_ack(tp, ID_BASE + s, "verify", err, errlen))
            return 0;
        return pulsar_tp_send_verify_commit(tp, (int32_t)(gen_next(g) % 2),
                                            (int32_t)(gen_next(g) % 5));
    }
    if (op == 8) { /* eval_batch */
        const uint32_t nb = 1u + (uint32_t)(gen_next(g) % 4);
        pulsar_tp_batch_item items[4];
        for (uint32_t i = 0; i < nb; i++) {
            const int s = pool_pick_live(g);
            if (s < 0) return 0;
            items[i].session_id = ID_BASE + s;
            items[i].token = 4000 + s * 16 + (int)i;
            items[i].reserved = 0;
        }
        if (!pulsar_tp_send_eval_batch(tp, items, nb)) return 0;
        return pulsar_tp_wait_command_ack(tp, items[0].session_id, "eval_batch",
                                          err, errlen);
    }
    if (op == 9) { /* mixed_batch */
        const uint32_t np = 1u + (uint32_t)(gen_next(g) % 12);
        const uint32_t ni = 1u + (uint32_t)(gen_next(g) % 4);
        int prompt[16];
        for (uint32_t i = 0; i < np; i++) prompt[i] = 5000 + (int)i;
        pulsar_tp_batch_item items[4];
        for (uint32_t i = 0; i < ni; i++) {
            const int s = pool_pick_live(g);
            if (s < 0) return 0;
            items[i].session_id = ID_BASE + s;
            items[i].token = 6000 + s * 16 + (int)i;
            items[i].reserved = 0;
        }
        if (!pulsar_tp_send_mixed_batch(tp, items[0].session_id, prompt, np,
                                        items, ni))
            return 0;
        return pulsar_tp_wait_command_ack(tp, items[0].session_id, "mixed",
                                          err, errlen);
    }
    if (op == 10) { /* session_create */
        const int s = pool_pick_dead(g);
        if (s < 0) return 1; /* pool full — no-op, not an error */
        if (!pulsar_tp_send_session_create(tp, ID_BASE + s, BASE_CTX)) return 0;
        const int ok = pulsar_tp_wait_command_ack(tp, ID_BASE + s, "create",
                                                  err, errlen);
        if (ok) g_live[s] = true;
        return ok;
    }
    /* op == 11: session_destroy */
    const int s = pool_pick_live(g);
    if (s < 0) return 0;
    if (!pulsar_tp_send_session_destroy(tp, ID_BASE + s)) return 0;
    const int ok = pulsar_tp_wait_command_ack(tp, ID_BASE + s, "destroy", err,
                                              errlen);
    if (ok) g_live[s] = false;
    return ok;
}

static int leader_run(pulsar_tp *tp, int steps) {
    std::memset(g_live, 0, sizeof(g_live));
    std::memset(g_seq, 0, sizeof(g_seq));
    gen g = { 0x5EEDULL };
    char err[256];
    /* Seed the pool so the very first random op has live sessions to target
     * (the ledger is primed through the same wire path the worker mirrors). */
    for (int s = 0; s < 3; s++) {
        if (!pulsar_tp_send_session_create(tp, ID_BASE + s, BASE_CTX)) {
            CHECK(0, "leader seed create %d failed", s);
            return 1;
        }
        if (!pulsar_tp_wait_command_ack(tp, ID_BASE + s, "seed create",
                                        err, sizeof(err))) {
            CHECK(0, "leader seed create %d ack: %s", s, err);
            return 1;
        }
        g_live[s] = true;
    }
    const double t0 = tp_now_sec();
    for (int i = 0; i < steps; i++) {
        CHECK(leader_step(tp, &g, err, sizeof(err)) == 1,
              "leader step %d: %s", i, err);
    }
    if (!pulsar_tp_send_stop(tp)) {
        CHECK(0, "leader send_stop failed");
        return 1;
    }
    std::printf("leader: %d steps in %.3fs\n", steps, tp_now_sec() - t0);
    std::fflush(stdout);
    return 0;
}

/* Worker entry: create the transport, then pump the dispatcher until STOP. */
static int worker_entry(int port) {
    char err[256];
    pulsar_tp *tp = NULL;
    if (!create_rank(1, port, &tp, err, sizeof(err))) {
        fprintf(stderr, "worker create: %s\n", err);
        return 1;
    }
    alarm(120);
    worker_run(tp);
    pulsar_tp_free(tp);
    return g_failures != 0;
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

int main(void) {
    const char *role = getenv("TP_CMD_STRESS_ROLE");
    if (role && std::strcmp(role, "worker") == 0) {
        const char *ps = getenv("TP_CMD_STRESS_PORT");
        const int port = ps ? atoi(ps) : 0;
        _exit(worker_entry(port) != 0);
    }

    setenv("PULSAR_TP_TIMEOUT_SEC", "10", 1);
    const int port = free_port();
    if (port <= 0) {
        fprintf(stderr, "no free loopback port\n");
        return 1;
    }
    char pbuf[16];
    snprintf(pbuf, sizeof(pbuf), "%d", port);
    setenv("TP_CMD_STRESS_PORT", pbuf, 1);
    setenv("TP_CMD_STRESS_ROLE", "worker", 1);
    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "fork: %s\n", strerror(errno));
        return 1;
    }
    if (pid == 0) _exit(worker_entry(port) != 0);

    alarm(120);
    char err[256];
    pulsar_tp *tp = NULL;
    int lrc = 1;
    if (!create_rank(0, port, &tp, err, sizeof(err))) {
        CHECK(0, "leader create: %s", err);
    } else {
        lrc = leader_run(tp, OP_STEPS);
        pulsar_tp_free(tp);
    }
    alarm(0);

    int wstatus = 0;
    if (waitpid(pid, &wstatus, 0) < 0) {
        CHECK(0, "waitpid: %s", std::strerror(errno));
        return 1;
    }
    const int child_fail = !WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0;
    if (child_fail) {
        CHECK(0, "worker side failed (exit %d)", WEXITSTATUS(wstatus));
        return 1;
    }
    if (lrc != 0 || g_failures) {
        fprintf(stderr, "tp_cmd_stress_test: FAILED\n");
        return 1;
    }
    std::printf("tp_cmd_stress_test: ok (%d randomized control ops, ledger in sync, clean STOP)\n",
                OP_STEPS);
    return 0;
}

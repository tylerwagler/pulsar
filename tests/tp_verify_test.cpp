/*
 * TP verify/commit reference-grading test (branch tensor_parallel; the host
 * half of engine-slice 4e).  Host-only: no CUDA, no RDMA -- TCP loopback,
 * forked leader/worker.
 *
 * Two parts:
 *
 *  A. Decision rule (pure): pulsar_tp_verify_decide must map per-position
 *     match vectors to the batched-verify (full_accept, replay_n) convention
 *     -- all-match = accept everything, first mismatch at k = stop before k.
 *
 *  B. Leader round-trip (pair): the leader drives
 *     pulsar_tp_verify_leader_round with a deterministic reference hook
 *     ("draft matches when its token is even"), and the worker -- which has
 *     the drafts from the VERIFY frame -- runs the SAME deterministic grade
 *     on its side and must see the leader's VERIFY_COMMIT agree exactly.
 *     This pins the full grading+commit lockstep the real engine pair will
 *     run, at a few hundred verifies.
 */
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include "tp/pulsar_tp.h"
#include "tp/pulsar_tp_verify.h"

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

#define BASE_CTX 1048576u
#define ROUNDS 400
#define VERIFY_SID 9999u

static uint64_t gen_state = 0x5EEDULL;
static uint32_t gen_next(void) {
    gen_state ^= gen_state >> 12;
    gen_state ^= gen_state << 25;
    gen_state ^= gen_state >> 27;
    return (uint32_t)(gen_state * 0x2545F4914F6CDD1DULL);
}

/* Deterministic "reference": a draft row is graded a match iff its token is
 * even.  Both ranks run the same rule from the drafts alone, so the worker's
 * mirror of the leader's decision is exact. */
static void grade_even(void *ud, const int *drafts, uint32_t n, bool *match) {
    (void)ud;
    for (uint32_t i = 0; i < n; i++) match[i] = (drafts[i] & 1) == 0;
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

static void unit_decide(void) {
    bool all[5] = { 1, 1, 1, 1, 1 };
    bool first[5] = { 0, 1, 1, 1, 1 };
    bool mid[5] = { 1, 1, 0, 1, 1 };
    bool one_t[1] = { 1 };
    bool one_f[1] = { 0 };

    pulsar_tp_verify_decision d;

    pulsar_tp_verify_decide(all, 5, &d);
    CHECK(d.full_accept == 1 && d.replay_n == 5, "all-match: %d/%u",
          d.full_accept, d.replay_n);
    pulsar_tp_verify_decide(first, 5, &d);
    CHECK(d.full_accept == 0 && d.replay_n == 0, "first-mismatch: %d/%u",
          d.full_accept, d.replay_n);
    pulsar_tp_verify_decide(mid, 5, &d);
    CHECK(d.full_accept == 0 && d.replay_n == 2, "mid-mismatch: %d/%u",
          d.full_accept, d.replay_n);
    pulsar_tp_verify_decide(one_t, 1, &d);
    CHECK(d.full_accept == 1 && d.replay_n == 1, "single match: %d/%u",
          d.full_accept, d.replay_n);
    pulsar_tp_verify_decide(one_f, 1, &d);
    CHECK(d.full_accept == 0 && d.replay_n == 0, "single mismatch: %d/%u",
          d.full_accept, d.replay_n);
    pulsar_tp_verify_decide(all, 0, &d);
    CHECK(d.full_accept == 0 && d.replay_n == 0, "empty: %d/%u",
          d.full_accept, d.replay_n);
    pulsar_tp_verify_decide(NULL, 5, &d);
    CHECK(d.full_accept == 0 && d.replay_n == 0, "null: %d/%u",
          d.full_accept, d.replay_n);

    /* Boundary: full accept only when every row matched; a run-with-a-
     * different-even count still stops exactly before the first odd draft. */
    bool pat[6] = { 1, 1, 1, 0, 1, 1 };
    pulsar_tp_verify_decide(pat, 6, &d);
    CHECK(d.full_accept == 0 && d.replay_n == 3, "run boundary: %d/%u",
          d.full_accept, d.replay_n);
    std::printf("verify-decide: ok (all/none/first/mid/single/empty/null)\n");
}

static void worker_run(pulsar_tp *tp) {
    char err[256];
    int rounds = 0;
    for (;;) {
        pulsar_tp_command cmd;
        CHECK(pulsar_tp_recv_command(tp, &cmd, err, sizeof(err)) == 1,
              "worker recv: %s", err);
        if (cmd.type == PULSAR_TP_FRAME_STOP) {
            pulsar_tp_command_free(&cmd);
            break;
        }
        if (cmd.type != PULSAR_TP_FRAME_VERIFY) {
            CHECK(0, "worker expected VERIFY, got %d", (int)cmd.type);
            pulsar_tp_command_free(&cmd);
            continue;
        }
        CHECK(cmd.session_id == VERIFY_SID && cmd.tokens &&
                  cmd.n_tokens >= 1 && cmd.n_tokens <= 16,
              "worker verify header sid=%llu n=%u", (unsigned long long)cmd.session_id,
              cmd.n_tokens);
        /* Mirror the leader's grade + decision from the drafts themselves. */
        bool match[16];
        grade_even(NULL, cmd.tokens, cmd.n_tokens, match);
        pulsar_tp_verify_decision want;
        pulsar_tp_verify_decide(match, cmd.n_tokens, &want);
        if (!pulsar_tp_send_command_ack(tp, cmd.session_id, 0)) {
            CHECK(0, "worker verify ack failed");
            pulsar_tp_command_free(&cmd);
            continue;
        }
        pulsar_tp_command_free(&cmd);
        int32_t full = -1, replay = -1;
        CHECK(pulsar_tp_recv_verify_commit(tp, &full, &replay) == 1,
              "worker commit recv");
        CHECK(full == want.full_accept && (uint32_t)replay == want.replay_n,
              "worker commit %d/%d != mirror %d/%u at round %d", full, replay,
              want.full_accept, want.replay_n, rounds);
        rounds++;
    }
    std::printf("worker: %d verify rounds, commits confirmed\n", rounds);
    std::fflush(stdout);
}

static int leader_run(pulsar_tp *tp) {
    pulsar_tp_verify_hooks hooks = {};
    hooks.grade_matches = grade_even;
    char err[256];
    for (int i = 0; i < ROUNDS; i++) {
        const uint32_t n = 1u + (gen_next() % 16);
        int drafts[16];
        for (uint32_t k = 0; k < n; k++) drafts[k] = (int)(gen_next() % 1000);
        CHECK(pulsar_tp_verify_leader_round(tp, VERIFY_SID, drafts, n, &hooks,
                                            err, sizeof(err)) == 1,
              "leader round %d: %s", i, err);
    }
    if (!pulsar_tp_send_stop(tp)) {
        CHECK(0, "leader send_stop failed");
        return 1;
    }
    std::printf("leader: %d verify + commit rounds\n", ROUNDS);
    std::fflush(stdout);
    return 0;
}

static int worker_entry(int port) {
    char err[256];
    pulsar_tp *tp = NULL;
    if (!create_rank(1, port, &tp, err, sizeof(err))) {
        fprintf(stderr, "worker create: %s\n", err);
        return 1;
    }
    alarm(60);
    worker_run(tp);
    pulsar_tp_free(tp);
    std::fflush(stdout);
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
    unit_decide();

    const char *role = getenv("TP_VERIFY_TEST_ROLE");
    if (role && std::strcmp(role, "worker") == 0) {
        const char *ps = getenv("TP_VERIFY_TEST_PORT");
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
    setenv("TP_VERIFY_TEST_PORT", pbuf, 1);
    setenv("TP_VERIFY_TEST_ROLE", "worker", 1);
    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "fork: %s\n", strerror(errno));
        return 1;
    }
    if (pid == 0) _exit(worker_entry(port) != 0);

    alarm(60);
    char err[256];
    pulsar_tp *tp = NULL;
    int lrc = 1;
    if (!create_rank(0, port, &tp, err, sizeof(err))) {
        CHECK(0, "leader create: %s", err);
    } else {
        lrc = leader_run(tp);
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
        fprintf(stderr, "tp_verify_test: FAILED\n");
        return 1;
    }
    std::printf("tp_verify_test: ok (%d verify+commit rounds, worker mirror agrees)\n",
                ROUNDS);
    return 0;
}

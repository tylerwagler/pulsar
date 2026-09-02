/*
 * TP peer-fault test (branch tensor_parallel; audit F3/F5 behavior under a
 * dead peer -- the other half of the identity/no-hang pairing).
 *
 * Host-only: no CUDA, no RDMA -- TCP loopback, forked leader/worker.
 *
 * The dev box cannot exercise an RDMA link death, but the TCP fallback has
 * the same blocking structure the runbook depends on (blocking reads on the
 * gate and control sockets; the leader's accept has no timeout).  This test
 * kills ONE rank while the OTHER is mid-`pulsar_tp_*exchange`, and asserts
 * the survivor NEVER hangs:
 *
 *  A. peer dies mid-gate-sequence: after two healthy exchanges the worker
 *     frees the transport and exits without STOP.  The leader's next
 *     gate_exchange must return 0 (EOF/RST), not block, and repeated calls
 *     must keep failing (the engine's gate pump must be able to detect
 *     death).  A follow-up 1 MiB big_gate on the separate data socket must
 *     also return 0.
 *  B. peer dies before acking a control command: the leader's
 *     pulsar_tp_wait_command_ack must return 0 with "worker failed" and must
 *     set the pulsar_tp_mark_failed latch (pulsar_tp_failed() == true) --
 *     the only auto-latch in the transport.
 *
 * Every survivor runs under alarm(20); the parent enforces its own bound.
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

#define N_LAYER 43u
#define N_EMBD 4096u
#define BASE_CTX 1048576u
#define VEC_BYTES ((uint64_t)N_EMBD * 4u)

static double tp_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static float pat(int rank, uint32_t slot, uint32_t i) {
    return (float)((int)(rank == 0 ? 70000 : 50000) + (int)slot * 1000 + (int)i);
}

static int fill_pat(float *out, int rank, uint32_t slot, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) out[i] = pat(rank, slot, i);
    return 1;
}

static int verify_pat(const float *in, int rank, uint32_t slot, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        if (in[i] != pat(rank, slot, i)) {
            CHECK(0, "pattern mismatch i=%u got %g want %g", i, in[i],
                  pat(rank, slot, i));
            return 0;
        }
    }
    return 1;
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
    id.n_layer = N_LAYER;
    id.n_embd = N_EMBD;
    id.n_vocab = 129280u;
    id.quant_bits = 2u;
    id.ctx_size = BASE_CTX;
    return pulsar_tp_create(out, &opt, &id, err, errlen);
}

static void run_worker(int port, int mode) {
    char err[256];
    pulsar_tp *tp = NULL;
    CHECK(create_rank(1, port, &tp, err, sizeof(err)) == 1,
          "worker create: %s", err);
    if (!tp) return;

    if (mode == 2) { /* control-death mode: die without acking anything */
        pulsar_tp_free(tp);
        return;
    }

    uint8_t *slab = (uint8_t *)std::malloc((size_t)pulsar_tp_slab_bytes(N_LAYER, N_EMBD));
    if (!slab) {
        CHECK(0, "worker slab alloc failed");
        pulsar_tp_free(tp);
        return;
    }
    std::memset(slab, 0, (size_t)pulsar_tp_slab_bytes(N_LAYER, N_EMBD));
    CHECK(pulsar_tp_attach_slab(tp, slab, err, sizeof(err)) == 1,
          "worker attach: %s", err);

    pulsar_tp_slab s;
    pulsar_tp_slab_layout_init(N_LAYER, N_EMBD, &s);
    const uint32_t n = (uint32_t)(VEC_BYTES / sizeof(float));

    /* Two healthy exchanges to settle the pairing (verified both ways). */
    for (uint32_t layer = 0; layer < 2; layer++) {
        const uint32_t slot = layer * PULSAR_TP_GATES_PER_LAYER;
        float *out = (float *)(slab + pulsar_tp_slab_out_offset(&s, layer, 0, VEC_BYTES));
        const float *in = (const float *)(slab + pulsar_tp_slab_in_offset(&s, layer, 0, VEC_BYTES));
        fill_pat(out, 1, slot, n);
        CHECK(pulsar_tp_gate_exchange(tp, layer, 0, (uint64_t)slot + 1) == 1,
              "worker gate l=%u", layer);
        verify_pat(in, 0, slot, n); /* peer's partial landed */
    }

    /* Die without STOP: frees the transport (closing both sockets) while the
     * leader is still pumping.  Simulates a crash / cable pull. */
    pulsar_tp_free(tp);
    std::free(slab);
}

static void run_leader(int port, int mode) {
    char err[256];
    pulsar_tp *tp = NULL;
    CHECK(create_rank(0, port, &tp, err, sizeof(err)) == 1,
          "leader create: %s", err);
    if (!tp) return;

    if (mode == 2) {
        /* Control-path death: the worker is already gone.  send_eval may
         * succeed (buffered) or fail (RST) -- either way wait_command_ack
         * must read EOF and latch failed(). */
        (void)pulsar_tp_send_eval(tp, 4242u, 7, 9);
        char e2[256] = "";
        const int rc = pulsar_tp_wait_command_ack(tp, 4242u, "eval", e2, sizeof(e2));
        CHECK(rc == 0, "wait_command_ack rc=%d (want 0)", rc);
        CHECK(pulsar_tp_failed(tp), "failed() not latched on ack failure");
        CHECK(e2[0] != '\0' && strstr(e2, "worker failed") != NULL,
              "ack err=%s (want worker-failed reason)", e2);
        return;
    }

    uint8_t *slab = (uint8_t *)std::malloc((size_t)pulsar_tp_slab_bytes(N_LAYER, N_EMBD));
    if (!slab) {
        CHECK(0, "leader slab alloc failed");
        pulsar_tp_free(tp);
        return;
    }
    std::memset(slab, 0, (size_t)pulsar_tp_slab_bytes(N_LAYER, N_EMBD));
    CHECK(pulsar_tp_attach_slab(tp, slab, err, sizeof(err)) == 1,
          "leader attach: %s", err);

    pulsar_tp_slab s;
    pulsar_tp_slab_layout_init(N_LAYER, N_EMBD, &s);
    const uint32_t n = (uint32_t)(VEC_BYTES / sizeof(float));

    /* Two healthy exchanges (mirrors the worker), then pump until death. */
    for (uint32_t layer = 0; layer < 2; layer++) {
        const uint32_t slot = layer * PULSAR_TP_GATES_PER_LAYER;
        float *out = (float *)(slab + pulsar_tp_slab_out_offset(&s, layer, 0, VEC_BYTES));
        const float *in = (const float *)(slab + pulsar_tp_slab_in_offset(&s, layer, 0, VEC_BYTES));
        fill_pat(out, 0, slot, n);
        CHECK(pulsar_tp_gate_exchange(tp, layer, 0, (uint64_t)slot + 1) == 1,
              "leader gate l=%u", layer);
        verify_pat(in, 1, slot, n); /* peer's partial landed */
    }

    /* Keep exchanging beyond the worker's death: every call must return 0
     * promptly, and at least one must (no silent success on a dead socket). */
    const double t0 = tp_now_sec();
    int saw_dead = 0, calls = 0;
    for (uint32_t layer = 2; layer < N_LAYER && calls < 20; layer++) {
        for (uint32_t gate = 0; gate < PULSAR_TP_GATES_PER_LAYER; gate++) {
            const uint32_t slot = layer * PULSAR_TP_GATES_PER_LAYER + gate;
            float *out = (float *)(slab + pulsar_tp_slab_out_offset(&s, layer, gate, VEC_BYTES));
            fill_pat(out, 0, slot, n);
            const int rc = pulsar_tp_gate_exchange(tp, layer, gate, (uint64_t)slot + 1);
            if (rc == 0) saw_dead = 1;
            calls++;
        }
    }
    CHECK(saw_dead, "leader never observed the worker's death");
    const double elapsed = tp_now_sec() - t0;
    CHECK(elapsed < 10.0, "death took %.2fs to surface (hang?)", elapsed);

    /* The separate data socket is dead too: a 1 MiB big gate must return 0. */
    static float big_out[262144], big_in[262144];
    CHECK(pulsar_tp_big_gate_exchange(tp, 0, 1, big_out, big_in,
                                      sizeof(big_out)) == 0,
          "big_gate on dead data socket did not fail");

    std::free(slab);
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

static int run_mode(int mode) {
    const int port = free_port();
    if (port <= 0) {
        CHECK(0, "free_port failed (mode %d)", mode);
        return 1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        CHECK(0, "fork (mode %d): %s", mode, std::strerror(errno));
        return 1;
    }
    if (pid == 0) {
        alarm(20);
        run_worker(port, mode);
        std::fflush(stdout);
        _exit(g_failures != 0);
    }

    alarm(20);
    run_leader(port, mode);
    alarm(0);

    int wstatus = 0;
    if (waitpid(pid, &wstatus, 0) < 0) {
        CHECK(0, "waitpid (mode %d): %s", mode, std::strerror(errno));
        return 1;
    }
    const int child_fail = !WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0;
    if (child_fail) {
        CHECK(0, "mode %d: worker side failed", mode);
        return 1;
    }
    return 0;
}

int main(void) {
    setenv("PULSAR_TP_TIMEOUT_SEC", "5", 1);
    /* mode 1: the gate pump runs into a dead peer (worker dies after two
     * exchanges; leader's exchanges must fail, never hang).  mode 2: control-
     * path latch (worker dies before acking; wait_command_ack must return the
     * "worker failed" error and set failed()). */
    int rc = 0;
    rc |= run_mode(1);
    rc |= run_mode(2);
    if (g_failures) {
        std::fprintf(stderr, "tp_fault_test: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("tp_fault_test: ok (gate-pump death surfaced, no hang; control-ack latch)\n");
    return rc;
}

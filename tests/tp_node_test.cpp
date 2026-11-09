/*
 * TP NODE frame test (protocol v13).
 *
 * Host-only: the ranks ride the TCP fallback over 127.0.0.1, one forked child
 * per rank, each with alarm() armed so a hang fails instead of blocking.
 *
 * After bring-up every rank must know every rank's record -- its own and each
 * peer's -- through pulsar_tp_node_info, over both bring-up paths:
 *
 *  A. the legacy pair (pulsar_tp_create, leader/worker roles), and
 *  B. a three-rank full mesh (pulsar_tp_create_mesh).
 *
 * Each rank stamps a distinct build ("build-r<rank>"), so a record delivered to
 * the wrong slot, or a rank reporting its own build for a peer, fails by name.
 * Every child runs on this host, so every record's host is this hostname. Under
 * TCP the RDMA device must be empty: naming an HCA the transport does not use
 * would send an operator to the wrong port counters.
 */
#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
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

static pulsar_tp_identity node_identity(void) {
    pulsar_tp_identity id;
    pulsar_tp_identity_init_defaults(&id, 87000000000ull, 3u, 4u, 256u, 1000u, 2u, 4096u);
    return id;
}

static int free_port(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in a;
    std::memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
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

/* Every rank's record, checked from inside one rank.  Returns the failures. */
static int check_records(const pulsar_tp *tp, int me, int n, bool mesh, const int *ports) {
    const int before = g_failures;
    char host[PULSAR_TP_NODE_STR] = "";
    gethostname(host, sizeof(host));
    host[sizeof(host) - 1] = '\0';

    for (int r = 0; r < n; r++) {
        pulsar_tp_node node;
        std::memset(&node, 0xAB, sizeof(node));
        if (!pulsar_tp_node_info(tp, r, &node)) {
            CHECK(0, "rank %d has no record for rank %d", me, r);
            continue;
        }
        char want_build[32];
        std::snprintf(want_build, sizeof(want_build), "build-r%d", r);
        CHECK(node.rank == r, "rank %d: record for %d says rank %d", me, r, node.rank);
        CHECK(std::strcmp(node.build, want_build) == 0,
              "rank %d: record for %d has build '%s', want '%s'", me, r, node.build, want_build);
        CHECK(std::strcmp(node.host, host) == 0,
              "rank %d: record for %d has host '%s', want '%s'", me, r, node.host, host);
        CHECK(node.rdma_device[0] == '\0' && node.rdma_port == 0,
              "rank %d: record for %d names RDMA device '%s' port %d under TCP",
              me, r, node.rdma_device, node.rdma_port);
        if (mesh) {
            char want_addr[64];
            std::snprintf(want_addr, sizeof(want_addr), "127.0.0.1:%d", ports[r]);
            CHECK(std::strcmp(node.addr, want_addr) == 0,
                  "rank %d: record for %d has addr '%s', want '%s'", me, r, node.addr, want_addr);
        } else if (r == 0) {
            /* The pair's leader knows its listen port; the worker has none. */
            char want_addr[64];
            std::snprintf(want_addr, sizeof(want_addr), "127.0.0.1:%d", ports[0]);
            CHECK(std::strcmp(node.addr, want_addr) == 0,
                  "rank %d: leader record has addr '%s', want '%s'", me, node.addr, want_addr);
        }
    }
    pulsar_tp_node none;
    CHECK(!pulsar_tp_node_info(tp, n, &none), "rank %d: a record exists for rank %d of %d", me, n, n);
    return g_failures - before;
}

static int run_pair_rank(int rank, int port) {
    alarm(20);
    char err[512];
    char build[32];
    std::snprintf(build, sizeof(build), "build-r%d", rank);
    pulsar_tp_options opt;
    std::memset(&opt, 0, sizeof(opt));
    opt.role = rank == 0 ? PULSAR_TP_ROLE_LEADER : PULSAR_TP_ROLE_WORKER;
    opt.peer = "127.0.0.1";   /* the leader binds it, the worker dials it */
    opt.port = port;
    opt.build = build;
    pulsar_tp_identity id = node_identity();
    pulsar_tp *tp = nullptr;
    if (!pulsar_tp_create(&tp, &opt, &id, err, sizeof(err))) {
        std::fprintf(stderr, "FAIL pair rank %d create: %s\n", rank, err);
        return 1;
    }
    const int ports[1] = { port };
    const int bad = check_records(tp, rank, 2, false, ports);
    pulsar_tp_free(tp);
    return bad ? 1 : 0;
}

static int run_mesh_rank(int rank, int n, const int *ports) {
    alarm(20);
    char err[512];
    char peers[512] = "";
    for (int k = 0; k < n; k++) {
        char one[64];
        std::snprintf(one, sizeof(one), "%s127.0.0.1:%d", k ? "," : "", ports[k]);
        std::strncat(peers, one, sizeof(peers) - std::strlen(peers) - 1);
    }
    char build[32];
    std::snprintf(build, sizeof(build), "build-r%d", rank);
    pulsar_tp_options opt;
    std::memset(&opt, 0, sizeof(opt));
    opt.role = PULSAR_TP_ROLE_LEADER;
    opt.rank = rank;
    opt.n_ranks = n;
    opt.peers = peers;
    opt.port = ports[rank];
    opt.build = build;
    pulsar_tp_identity id = node_identity();
    pulsar_tp *tp = nullptr;
    if (!pulsar_tp_create_mesh(&tp, &opt, &id, err, sizeof(err))) {
        std::fprintf(stderr, "FAIL mesh rank %d create: %s\n", rank, err);
        return 1;
    }
    const int bad = check_records(tp, rank, n, true, ports);
    pulsar_tp_free(tp);
    return bad ? 1 : 0;
}

/* Fork one child per rank; the group passes only if every child exits 0. */
template <typename F>
static void run_group(const char *name, int n, F body) {
    pid_t pids[8];
    for (int r = 0; r < n; r++) {
        pids[r] = fork();
        if (pids[r] == 0) _exit(body(r));
        /* The pair's worker dials with retry, and the mesh orders its own
         * connects, so no start-up sleep is needed between ranks. */
    }
    for (int r = 0; r < n; r++) {
        int status = 0;
        waitpid(pids[r], &status, 0);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "%s: rank %d failed (status %d)", name, r, status);
    }
}

int main(void) {
    setenv("PULSAR_TP_TIMEOUT_SEC", "10", 0);
    /* Keep verbs out of it: this test is about the record, and a box with an
     * HCA would otherwise legitimately name it. */
    setenv("PULSAR_TP_RDMA_DEV", "pulsar-tp-node-test-no-such-hca", 1);

    const int pair_port = free_port();
    CHECK(pair_port > 0, "no free port for the pair");
    run_group("pair", 2, [&](int r) { return run_pair_rank(r, pair_port); });

    int ports[3];
    for (int k = 0; k < 3; k++) ports[k] = free_port();
    run_group("mesh", 3, [&](int r) { return run_mesh_rank(r, 3, ports); });

    if (g_failures) {
        std::fprintf(stderr, "tp_node_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("tp_node_test: ok (pair + 3-rank mesh: every rank knows every rank's host, build and addr; no RDMA device named under TCP)\n");
    return 0;
}

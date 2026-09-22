/*
 * TP n-way full-mesh test (docs/tensor-parallel-port.md slice n-way).
 * Host-only: no CUDA, no RDMA — the mesh rides its full-duplex TCP fallback
 * over 127.0.0.1.  Forks n child processes (ranks 1..n-1) plus the parent as
 * rank 0; every rank calls pulsar_tp_create_mesh with the same ordered peer
 * list, attaches a plain slab, and drives pulsar_tp_allreduce_sum.  Each rank
 * starts from a distinct, exact (integer-valued f32) partial; the all-reduce
 * must return the exact sum of every rank's partial, identical on all ranks.
 *
 * Run for n=2 and n=3.  Exit 0 on success, 1 on any mismatch.
 */
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/socket.h>
#include <sys/wait.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>

#include "tp/pulsar_tp.h"

#define MESH_N_LAYER 2u
#define MESH_N_EMBD 64u

static int g_failures = 0;
#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            g_failures++;                                                    \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);        \
            std::fprintf(stderr, __VA_ARGS__);                               \
            std::fprintf(stderr, "\n");                                      \
            std::fflush(stderr);                                             \
        }                                                                    \
    } while (0)

static pulsar_tp_identity mesh_identity(void) {
    pulsar_tp_identity id;
    std::memset(&id, 0, sizeof(id));
    id.gguf_bytes = 123456789ull;
    id.model_id = 1u;
    id.n_layer = MESH_N_LAYER;
    id.n_embd = MESH_N_EMBD;
    id.n_vocab = 1000u;
    id.quant_bits = 2u;
    id.ctx_size = 0u;
    return id;
}

static int free_port(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in a;
    std::memset(&a, 0, sizeof(a));
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

static int run_mesh_rank(int rank, int n, const int *ports) {
    char err[512];
    char peers[1024] = "";
    for (int k = 0; k < n; k++) {
        char one[128];
        std::snprintf(one, sizeof(one), "%s127.0.0.1:%d", k ? "," : "", ports[k]);
        std::strncat(peers, one, sizeof(peers) - std::strlen(peers) - 1);
    }
    pulsar_tp_options opt;
    std::memset(&opt, 0, sizeof(opt));
    opt.role = PULSAR_TP_ROLE_LEADER;   /* irrelevant for n>2 */
    opt.rank = rank;
    opt.n_ranks = n;
    opt.peers = peers;
    opt.port = ports[rank];

    pulsar_tp *tp = nullptr;
    pulsar_tp_identity id = mesh_identity();
    if (!pulsar_tp_create_mesh(&tp, &opt, &id, err, sizeof(err))) {
        CHECK(0, "rank %d mesh create failed: %s", rank, err);
        return 1;
    }
    CHECK(pulsar_tp_rank(tp) == rank, "rank %d reports rank %d", rank,
          pulsar_tp_rank(tp));
    CHECK(pulsar_tp_n_ranks(tp) == (uint32_t)n, "rank %d reports n_ranks %u",
          rank, pulsar_tp_n_ranks(tp));

    const uint64_t slab_bytes = pulsar_tp_slab_bytes(MESH_N_LAYER, MESH_N_EMBD);
    void *slab = std::malloc((size_t)slab_bytes);
    CHECK(slab != nullptr, "rank %d slab alloc failed", rank);
    if (!slab || !pulsar_tp_attach_slab(tp, slab, err, sizeof(err))) {
        CHECK(0, "rank %d slab attach failed: %s", rank, err);
        pulsar_tp_free(tp);
        std::free(slab);
        return 1;
    }

    const uint64_t bytes = (uint64_t)MESH_N_EMBD * sizeof(float);
    float *out = static_cast<float *>(std::malloc((size_t)bytes));
    float *in = static_cast<float *>(std::malloc((size_t)bytes));
    const uint64_t nelt = bytes / sizeof(float);

    /* Distinct exact (integer-valued) partial per rank so the sum is exact. */
    for (uint64_t i = 0; i < nelt; i++)
        out[i] = (float)(1000 * rank + (int)(i % 7));

    if (!pulsar_tp_allreduce_sum(tp, 0, 1, out, in, bytes)) {
        CHECK(0, "rank %d allreduce failed", rank);
        pulsar_tp_free(tp);
        std::free(slab);
        std::free(out);
        std::free(in);
        return 1;
    }

    /* Expected: sum over all ranks of (1000*k + i%7).  Exact in f32. */
    const float expect_base = 1000.0f * (float)(n * (n - 1)) / 2.0f;
    int bad = 0;
    for (uint64_t i = 0; i < nelt; i++) {
        const float want = expect_base + (float)n * (float)(int)(i % 7);
        if (out[i] != want) {
            if (bad == 0)
                std::fprintf(stderr,
                             "tp_mesh_test: rank %d allreduce mismatch at %llu: "
                             "got %f want %f\n",
                             rank, (unsigned long long)i, out[i], want);
            bad++;
        }
    }
    CHECK(bad == 0, "rank %d allreduce had %d mismatched elements", rank, bad);

    pulsar_tp_free(tp);
    std::free(slab);
    std::free(out);
    std::free(in);
    return bad == 0 ? 0 : 1;
}

static int run_mesh(int n) {
    int ports[8];
    for (int k = 0; k < n; k++) {
        ports[k] = free_port();
        if (ports[k] <= 0) {
            std::fprintf(stderr, "tp_mesh_test: no free loopback port (rank %d)\n", k);
            return 1;
        }
    }
    pid_t pids[8];
    for (int k = 1; k < n; k++) {
        pid_t pid = fork();
        if (pid < 0) {
            std::fprintf(stderr, "tp_mesh_test: fork failed\n");
            return 1;
        }
        if (pid == 0) {
            /* Child rank k.  Flush before _exit so parent aggregates output. */
            const int rc = run_mesh_rank(k, n, ports);
            std::fflush(stdout);
            std::fflush(stderr);
            _exit(rc != 0);
        }
        pids[k] = pid;
    }
    int rc = run_mesh_rank(0, n, ports);
    for (int k = 1; k < n; k++) {
        int wstatus = 0;
        if (waitpid(pids[k], &wstatus, 0) < 0) continue;
        if (!WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0) rc = 1;
    }
    return rc;
}

int main(void) {
    int rc = 0;
    for (int n = 2; n <= 3; n++) {
        std::printf("tp_mesh_test: n=%d\n", n);
        std::fflush(stdout);
        if (run_mesh(n) != 0) rc = 1;
        std::fflush(stdout);
        std::fflush(stderr);
    }
    if (rc == 0)
        std::printf("tp_mesh_test: ok (n=2 and n=3 mesh + all-reduce, exact)\n");
    else
        std::printf("tp_mesh_test: FAILED\n");
    return rc;
}

/*
 * TP gate-scheduler test (branch tensor_parallel, docs/tensor-parallel-*
 * slice 4b).  Host-only — no CUDA, no RDMA — the scheduler runs over the
 * transport's full-duplex TCP fallback on 127.0.0.1.
 *
 * Forks on the leader side: parent = LEADER (binds a free loopback port), child
 * = WORKER (dials it).  Both create the transport, attach a host slab, install
 * hooks that publish an f32 pattern into the out-slot and verify the peer's
 * pattern landed in the in-slot, then drive two full decode tokens through
 * pulsar_tp_sched_decode_token (86 exchanges each) and one prefill chunk
 * through pulsar_tp_sched_prefill_chunk.  Any in-slot that doesn't carry the
 * peer's bytes fails the rank it's on.
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
#include "tp/pulsar_tp_sched.h"

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
#define TEST_CTX_SIZE 1048576u
#define VEC_BYTES ((uint64_t)N_EMBD * 4u)

static pulsar_tp_identity test_identity(void) {
    pulsar_tp_identity id = {};
    id.gguf_bytes = 87000000000ull;
    id.model_id = 3u;
    id.n_layer = N_LAYER;
    id.n_embd = N_EMBD;
    id.n_vocab = 129280u;
    id.quant_bits = 2u;
    id.ctx_size = TEST_CTX_SIZE;
    return id;   /* zeroed gate schedule = DS identity slot mapping */
}

/* Distinct, exactly-representable pattern per (rank, slot-index, element). */
static float pat(int rank, uint32_t slot, uint32_t i) {
    return (float)((int)(rank == 0 ? 70000 : 50000) + (int)slot * 1000 + (int)i);
}

typedef struct {
    int rank;
    uint8_t *slab;
    pulsar_tp_slab layout;
} hook_ctx;

/* Normalize a per-exchange counter to the slot it must land in, so the same
 * pattern function is usable over the identity schedule. */
static uint32_t e_slot(uint64_t e) {
    return (uint32_t)((e - 1) % (uint64_t)(N_LAYER * PULSAR_TP_GATES_PER_LAYER));
}

static int hook_write(void *ud_, uint32_t layer, uint32_t gate, uint64_t e) {
    hook_ctx *c = (hook_ctx *)ud_;
    const uint32_t slot = (uint32_t)(layer * PULSAR_TP_GATES_PER_LAYER + gate);
    float *out = (float *)(c->slab + pulsar_tp_slab_out_offset(&c->layout, layer, gate, VEC_BYTES));
    CHECK(slot == e_slot(e), "rank %d hook_write slot %u != e_slot(%llu) %u",
          c->rank, slot, (unsigned long long)e, e_slot(e));
    const uint32_t n = (uint32_t)(VEC_BYTES / sizeof(float));
    for (uint32_t i = 0; i < n; i++) out[i] = pat(c->rank, slot, i);
    return 1;
}

static int hook_read(void *ud_, uint32_t layer, uint32_t gate, uint64_t e) {
    hook_ctx *c = (hook_ctx *)ud_;
    const uint32_t slot = (uint32_t)(layer * PULSAR_TP_GATES_PER_LAYER + gate);
    const float *in = (const float *)(c->slab + pulsar_tp_slab_in_offset(&c->layout, layer, gate, VEC_BYTES));
    const uint32_t n = (uint32_t)(VEC_BYTES / sizeof(float));
    for (uint32_t i = 0; i < n; i++) {
        const float want = pat(1 - c->rank, slot, i);
        if (in[i] != want) {
            CHECK(0, "rank %d read l=%u g=%u e=%llu i=%u: in %g want %g",
                  c->rank, layer, gate, (unsigned long long)e, i, in[i], want);
            break;
        }
    }
    return 1;
}

static int run_rank(pulsar_tp *tp, int rank) {
    uint8_t *slab = (uint8_t *)std::malloc((size_t)pulsar_tp_slab_bytes(N_LAYER, N_EMBD));
    if (!slab) {
        CHECK(0, "rank %d slab alloc failed", rank);
        return 1;
    }
    std::memset(slab, 0, (size_t)pulsar_tp_slab_bytes(N_LAYER, N_EMBD));
    char err[256];
    CHECK(pulsar_tp_attach_slab(tp, slab, err, sizeof(err)) == 1,
          "rank %d attach_slab: %s", rank, err);

    hook_ctx ctx = { rank, slab, {} };
    pulsar_tp_slab_layout_init(N_LAYER, N_EMBD, &ctx.layout);
    pulsar_tp_sched_hooks hooks = { };
    hooks.name = rank == 0 ? "leader-hooks" : "worker-hooks";
    hooks.ud = &ctx;
    hooks.write_partial = hook_write;
    hooks.read_partial = hook_read;
    pulsar_tp_sched *sch = pulsar_tp_sched_new(tp, &hooks, err, sizeof(err));
    CHECK(sch != NULL, "rank %d sched_new: %s", rank, err);

    /* Two full decode tokens: 86 exchanges each, both directions implied by
     * hook symmetry (each rank verifies the OTHER's pattern in its in-slot). */
    CHECK(pulsar_tp_sched_decode_token(sch, err, sizeof(err)) == 1,
          "rank %d decode token 1: %s", rank, err);
    CHECK(pulsar_tp_sched_decode_token(sch, err, sizeof(err)) == 1,
          "rank %d decode token 2: %s", rank, err);

    /* One prefill chunk (big gates, one per layer). */
    uint8_t *cout = (uint8_t *)std::calloc((size_t)(N_LAYER * VEC_BYTES), 1);
    uint8_t *cin = (uint8_t *)std::calloc((size_t)(N_LAYER * VEC_BYTES), 1);
    CHECK(cout && cin, "rank %d prefill alloc failed", rank);
    if (cout && cin) {
        for (uint32_t layer = 0; layer < N_LAYER; layer++) {
            float *o = (float *)(cout + (uint64_t)layer * VEC_BYTES);
            const uint32_t n = (uint32_t)(VEC_BYTES / sizeof(float));
            for (uint32_t i = 0; i < n; i++) o[i] = pat(rank, 900 + layer, i);
        }
        CHECK(pulsar_tp_sched_prefill_chunk(sch, 3, cout, cin, err, sizeof(err)) == 1,
              "rank %d prefill_chunk: %s", rank, err);
        for (uint32_t layer = 0; layer < N_LAYER; layer++) {
            const float *r = (const float *)(cin + (uint64_t)layer * VEC_BYTES);
            const uint32_t n = (uint32_t)(VEC_BYTES / sizeof(float));
            for (uint32_t i = 0; i < n; i++) {
                const float want = pat(1 - rank, 900 + layer, i);
                if (r[i] != want) {
                    CHECK(0, "rank %d prefill l=%u i=%u: got %g want %g",
                          rank, layer, i, r[i], want);
                    break;
                }
            }
        }
    }
    std::free(cout);
    std::free(cin);

    pulsar_tp_sched_free(sch);
    std::free(slab);
    return g_failures != 0;
}

static int create_rank(int rank, int port, pulsar_tp **out, char *err, size_t errlen) {
    pulsar_tp_options opt;
    std::memset(&opt, 0, sizeof(opt));
    opt.role = rank == 0 ? PULSAR_TP_ROLE_LEADER : PULSAR_TP_ROLE_WORKER;
    opt.peer = rank == 0 ? NULL : "127.0.0.1";
    opt.port = port;
    pulsar_tp_identity id = test_identity();
    return pulsar_tp_create(out, &opt, &id, err, errlen);
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

static int run_worker(int port) {
    char err[256];
    pulsar_tp *tp = NULL;
    if (!create_rank(1, port, &tp, err, sizeof(err))) {
        CHECK(0, "worker create: %s", err);
        return 1;
    }
    const int rc = run_rank(tp, 1);
    pulsar_tp_free(tp);
    return rc;
}

static int run_leader(int port) {
    char err[256];
    pulsar_tp *tp = NULL;
    if (!create_rank(0, port, &tp, err, sizeof(err))) {
        CHECK(0, "leader create: %s", err);
        return 1;
    }
    const int rc = run_rank(tp, 0);
    pulsar_tp_free(tp);
    return rc;
}

int main(void) {
    const char *role = getenv("TP_SCHED_TEST_ROLE");
    if (role && std::strcmp(role, "worker") == 0) {
        const char *ps = getenv("TP_SCHED_TEST_PORT");
        const int port = ps ? atoi(ps) : 0;
        const int rc = run_worker(port);
        std::fflush(stdout);
        _exit(rc != 0);
    }

    const int port = free_port();
    if (port <= 0) {
        std::fprintf(stderr, "tp_sched_test: no free loopback port\n");
        return 1;
    }
    char pbuf[16];
    std::snprintf(pbuf, sizeof(pbuf), "%d", port);
    setenv("TP_SCHED_TEST_PORT", pbuf, 1);
    setenv("TP_SCHED_TEST_ROLE", "worker", 1);
    pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr, "tp_sched_test: fork failed\n");
        return 1;
    }
    if (pid == 0) {
        const int rc = run_worker(port);
        std::fflush(stdout);
        _exit(rc != 0);
    }

    const int lrc = run_leader(port);
    int wstatus = 0;
    if (waitpid(pid, &wstatus, 0) < 0) {
        std::fprintf(stderr, "tp_sched_test: waitpid failed: %s\n", std::strerror(errno));
        return 1;
    }
    const int child_fail = !WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0;
    if (lrc != 0 || child_fail) {
        std::fprintf(stderr, "tp_sched_test: FAILED (%s)\n",
                     child_fail ? "worker side" : "leader side");
        return 1;
    }
    std::printf("tp_sched_test: ok (2 decode tokens x 86 gates + prefill chunk, symmetric both ranks)\n");
    return 0;
}

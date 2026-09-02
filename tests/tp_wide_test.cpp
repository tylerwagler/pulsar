/*
 * TP wide-embd / soak test (branch tensor_parallel; audit F4 + F8 host half).
 *
 * Host-only: no CUDA, no RDMA -- the pair rides the transport's TCP fallback
 * over 127.0.0.1 (forked leader/worker, same shape as tp_sched_test).
 *
 * What this covers that the fixed-n_embd=4096 tests cannot:
 *
 *  F4 (chunked-gate boundary).  At n_embd 4096 a gate partial is exactly one
 *     16 KiB message, so the RDMA path's "2 chunked messages, final-chunk-
 *     carries-seq" branch never fires and nothing > 16 KiB ever flows.  Here
 *     the pair also runs at n_embd 8192 (vec_bytes == 2*MSG: the exact chunked
 *     branch boundary) and n_embd 16384 (vec_bytes > 2*MSG: the size RDMA
 *     registration REJECTS -- the host slab plumbing must still round-trip
 *     both widths cleanly, because the scheduler, slab offsets and gate
 *     exchanges are exactly what a wider-embd model will run on).  These
 *     widths exist only to exercise the plumbing; the DS model is 4096.
 *
 *  F8 (slot-reuse soak, host half).  N full decode tokens (env
 *     PULSAR_TP_SOAK_TOKENS, default 8) x 86 gate exchanges each, reusing the
 *     same 86 slots, both directions, at every width: sustained reuse must
 *     keep every partial in its correct slot.  (The RDMA recv-watermark side
 *     of F8 is pair-confirm; the host side is slot placement under load.)
 *
 * Also pins the slab-layout arithmetic at each width (gate out/in offsets
 * stride by vec, batch regions stride by 8 slots) -- the numbers the RDMA
 * registration path will consume.
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
#define N_SLOTS (N_LAYER * PULSAR_TP_GATES_PER_LAYER)
#define BASE_CTX 1048576u
#define SOAK_DEFAULT 8

/* n_embd 4096 = the DS model (one 16 KiB gate message); 8192 = vec 2*MSG, the
 * RDMA chunked-branch boundary; 16384 = vec > 2*MSG (RDMA would reject the
 * registration -- the host path must still be correct). */
static const uint32_t WIDTHS[] = { 4096u, 8192u, 16384u };
#define N_WIDTHS ((int)(sizeof(WIDTHS) / sizeof(WIDTHS[0])))

static double tp_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Distinct, exactly-representable pattern per (rank, slot-index, element). */
static float pat(int rank, uint32_t slot, uint32_t i) {
    return (float)((int)(rank == 0 ? 70000 : 50000) + (int)slot * 1000 + (int)i);
}

typedef struct {
    int rank;
    uint8_t *slab;
    pulsar_tp_slab layout;
    uint64_t vec;      /* n_embd * 4 */
    uint32_t n_embd;
} hook_ctx;

static uint32_t e_slot(uint64_t e) {
    return (uint32_t)((e - 1) % (uint64_t)N_SLOTS);
}

static int hook_write(void *ud_, uint32_t layer, uint32_t gate, uint64_t e) {
    hook_ctx *c = (hook_ctx *)ud_;
    const uint32_t slot = layer * PULSAR_TP_GATES_PER_LAYER + gate;
    float *out = (float *)(c->slab +
        pulsar_tp_slab_out_offset(&c->layout, layer, gate, c->vec));
    CHECK(slot == e_slot(e), "rank %d write slot %u != e_slot(%llu) %u",
          c->rank, slot, (unsigned long long)e, e_slot(e));
    const uint32_t n = (uint32_t)(c->vec / sizeof(float));
    for (uint32_t i = 0; i < n; i++) out[i] = pat(c->rank, slot, i);
    return 1;
}

static int hook_read(void *ud_, uint32_t layer, uint32_t gate, uint64_t e) {
    hook_ctx *c = (hook_ctx *)ud_;
    const uint32_t slot = layer * PULSAR_TP_GATES_PER_LAYER + gate;
    const float *in = (const float *)(c->slab +
        pulsar_tp_slab_in_offset(&c->layout, layer, gate, c->vec));
    const uint32_t n = (uint32_t)(c->vec / sizeof(float));
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

/* One 8-row verify-block batch gate at the given width (full batch region
 * stride): the transport's batch exchange must round-trip 8 x vec bytes. */
static void batch_phase(pulsar_tp *tp, hook_ctx *c) {
    const uint32_t layer = 5u, rows = PULSAR_TP_BATCH_MAX_ROWS;
    const uint64_t bytes = (uint64_t)rows * c->vec;
    const uint32_t n = (uint32_t)(bytes / sizeof(float));
    float *out = (float *)(c->slab +
        pulsar_tp_slab_batch_out_offset(&c->layout, layer, c->vec));
    float *in = (float *)(c->slab +
        pulsar_tp_slab_batch_in_offset(&c->layout, layer, c->vec));
    for (uint32_t i = 0; i < n; i++) out[i] = pat(c->rank, 10000u, i);
    CHECK(pulsar_tp_batch_gate_exchange(tp, layer, rows, 1) == 1,
          "rank %d batch l=%u rows=%u failed", c->rank, layer, rows);
    for (uint32_t i = 0; i < n; i++) {
        const float want = pat(1 - c->rank, 10000u, i);
        if (in[i] != want) {
            CHECK(0, "rank %d batch l=%u i=%u: got %g want %g",
                  c->rank, layer, i, in[i], want);
            break;
        }
    }
}

/* One out-of-slab 1 MiB big gate: exercises the TCP fallback's multi-round
 * write/read split at the given width. */
static void big_phase(pulsar_tp *tp, hook_ctx *c) {
    const uint64_t bytes = 1024u * 1024u;
    const uint32_t n = (uint32_t)(bytes / sizeof(float));
    float *out = (float *)std::calloc((size_t)n, sizeof(float));
    float *in = (float *)std::calloc((size_t)n, sizeof(float));
    CHECK(out && in, "rank %d big alloc failed", c->rank);
    if (!out || !in) return;
    for (uint32_t i = 0; i < n; i++) out[i] = pat(c->rank, 777u, i);
    CHECK(pulsar_tp_big_gate_exchange(tp, 0, 1, out, in, bytes) == 1,
          "rank %d big 1MiB failed", c->rank);
    for (uint32_t i = 0; i < n; i++) {
        if (i % 4096 == 0 && in[i] != pat(1 - c->rank, 777u, i)) {
            CHECK(0, "rank %d big in[%u]=%g want %g", c->rank, i, in[i],
                  pat(1 - c->rank, 777u, i));
            break;
        }
    }
    std::free(out);
    std::free(in);
    (void)c;
}

static int run_rank(pulsar_tp *tp, int rank, uint32_t n_embd, int soak) {
    const uint64_t vec = (uint64_t)n_embd * sizeof(float);
    CHECK(pulsar_tp_vec_bytes(tp) == vec,
          "rank %d vec_bytes=%llu want %llu", rank,
          (unsigned long long)pulsar_tp_vec_bytes(tp), (unsigned long long)vec);

    uint8_t *slab = (uint8_t *)std::malloc(
        (size_t)pulsar_tp_slab_bytes(N_LAYER, n_embd));
    if (!slab) {
        CHECK(0, "rank %d slab alloc %llu failed", rank,
              (unsigned long long)pulsar_tp_slab_bytes(N_LAYER, n_embd));
        return 1;
    }
    std::memset(slab, 0, (size_t)pulsar_tp_slab_bytes(N_LAYER, n_embd));
    char err[256];
    CHECK(pulsar_tp_attach_slab(tp, slab, err, sizeof(err)) == 1,
          "rank %d attach_slab: %s", rank, err);

    hook_ctx ctx = { rank, slab, {}, vec, n_embd };
    pulsar_tp_slab_layout_init(N_LAYER, n_embd, &ctx.layout);
    pulsar_tp_sched_hooks hooks = {};
    hooks.name = rank == 0 ? "wide-leader" : "wide-worker";
    hooks.ud = &ctx;
    hooks.write_partial = hook_write;
    hooks.read_partial = hook_read;
    pulsar_tp_sched *sch = pulsar_tp_sched_new(tp, &hooks, err, sizeof(err));
    CHECK(sch != NULL, "rank %d sched_new: %s", rank, err);

    /* F8 host half: soak many full decode tokens (86 exchanges each) reusing
     * the same 86 slots, both directions, at this width. */
    const double t0 = tp_now_sec();
    int tok;
    for (tok = 0; tok < soak; tok++) {
        CHECK(pulsar_tp_sched_decode_token(sch, err, sizeof(err)) == 1,
              "rank %d decode token %d: %s", rank, tok, err);
    }
    const double dt = tp_now_sec() - t0;

    /* Prefill chunks: one big gate per layer. */
    uint8_t *cout = (uint8_t *)std::calloc((size_t)(N_LAYER * vec), 1);
    uint8_t *cin = (uint8_t *)std::calloc((size_t)(N_LAYER * vec), 1);
    CHECK(cout && cin, "rank %d prefill alloc failed", rank);
    if (cout && cin) {
        for (int ch = 0; ch < 2; ch++) {
            for (uint32_t layer = 0; layer < N_LAYER; layer++) {
                float *o = (float *)(cout + (uint64_t)layer * vec);
                const uint32_t n = (uint32_t)(vec / sizeof(float));
                for (uint32_t i = 0; i < n; i++) o[i] = pat(rank, 900 + layer, i);
            }
            CHECK(pulsar_tp_sched_prefill_chunk(sch, 3 + ch, cout, cin,
                                                err, sizeof(err)) == 1,
                  "rank %d prefill %d: %s", rank, ch, err);
            for (uint32_t layer = 0; layer < N_LAYER; layer++) {
                const float *r = (const float *)(cin + (uint64_t)layer * vec);
                const uint32_t n = (uint32_t)(vec / sizeof(float));
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
    }
    std::free(cout);
    std::free(cin);

    batch_phase(tp, &ctx);
    big_phase(tp, &ctx);

    pulsar_tp_sched_free(sch);
    std::free(slab);
    const double mbps = (double)soak * 86.0 * (double)vec * 2.0 / 1048576.0 / dt;
    std::printf("  embd=%-5u soak=%d ok (%d gates, %.1f MiB exchanged, %.0f MiB/s)\n",
                n_embd, soak, soak * 86, mbps * dt, mbps);
    return g_failures != 0;
}

static int create_rank(int rank, int port, uint32_t n_embd, pulsar_tp **out,
                       char *err, size_t errlen) {
    pulsar_tp_options opt = {};
    opt.role = rank == 0 ? PULSAR_TP_ROLE_LEADER : PULSAR_TP_ROLE_WORKER;
    opt.peer = rank == 0 ? NULL : "127.0.0.1";
    opt.port = port;
    pulsar_tp_identity id = {};
    id.gguf_bytes = 87000000000ull;
    id.model_id = 3u;
    id.n_layer = N_LAYER;
    id.n_embd = n_embd;
    id.n_vocab = 129280u;
    id.quant_bits = 2u;
    id.ctx_size = BASE_CTX;
    return pulsar_tp_create(out, &opt, &id, err, errlen);
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

/* Slab-layout arithmetic pins at this width: within a layer the two gate
 * out/in regions are adjacent vec-sized blocks; across layers the batch
 * regions stride by 8 slots.  These are the offsets the RDMA registration
 * consumes, so pin them before the pair runs. */
static void layout_check(uint32_t n_embd) {
    pulsar_tp_slab s;
    pulsar_tp_slab_layout_init(N_LAYER, n_embd, &s);
    const uint64_t vec = (uint64_t)n_embd * sizeof(float);
    for (uint32_t layer = 0; layer < N_LAYER; layer++) {
        CHECK(pulsar_tp_slab_out_offset(&s, layer, 1, vec) ==
                  pulsar_tp_slab_out_offset(&s, layer, 0, vec) + vec,
              "embd=%u out stride l=%u", n_embd, layer);
        CHECK(pulsar_tp_slab_in_offset(&s, layer, 1, vec) ==
                  pulsar_tp_slab_in_offset(&s, layer, 0, vec) + vec,
              "embd=%u in stride l=%u", n_embd, layer);
    }
    for (uint32_t layer = 0; layer + 1 < N_LAYER; layer++) {
        CHECK(pulsar_tp_slab_batch_out_offset(&s, layer + 1, vec) ==
                  pulsar_tp_slab_batch_out_offset(&s, layer, vec) +
                      (uint64_t)PULSAR_TP_BATCH_MAX_ROWS * vec,
              "embd=%u batch stride l=%u", n_embd, layer);
        CHECK(pulsar_tp_slab_batch_in_offset(&s, layer + 1, vec) ==
                  pulsar_tp_slab_batch_in_offset(&s, layer, vec) +
                      (uint64_t)PULSAR_TP_BATCH_MAX_ROWS * vec,
              "embd=%u batch_in stride l=%u", n_embd, layer);
    }
    CHECK(pulsar_tp_slab_bytes(N_LAYER, n_embd) > 0, "embd=%u slab_bytes", n_embd);
    std::printf("  layout embd=%-5u ok (slab %llu KiB)\n", n_embd,
                (unsigned long long)(pulsar_tp_slab_bytes(N_LAYER, n_embd) / 1024));
}

static int run_worker_child(uint32_t n_embd, int soak) {
    const char *ps = getenv("TP_WIDE_TEST_PORT");
    const int port = ps ? atoi(ps) : 0;
    char err[256];
    pulsar_tp *tp = NULL;
    if (!create_rank(1, port, n_embd, &tp, err, sizeof(err))) {
        CHECK(0, "worker create (embd %u): %s", n_embd, err);
        return 1;
    }
    const int rc = run_rank(tp, 1, n_embd, soak);
    pulsar_tp_free(tp);
    return rc;
}

static int run_wide_width(uint32_t n_embd, int soak) {
    layout_check(n_embd);

    const int port = free_port();
    if (port <= 0) {
        CHECK(0, "free_port failed (embd %u)", n_embd);
        return 1;
    }
    char pbuf[16], ebuf[16], sbuf[16];
    snprintf(pbuf, sizeof(pbuf), "%d", port);
    snprintf(ebuf, sizeof(ebuf), "%u", n_embd);
    snprintf(sbuf, sizeof(sbuf), "%d", soak);
    setenv("TP_WIDE_TEST_PORT", pbuf, 1);
    setenv("TP_WIDE_TEST_EMBD", ebuf, 1);
    setenv("TP_WIDE_TEST_SOAK", sbuf, 1);
    setenv("TP_WIDE_TEST_ROLE", "worker", 1);
    pid_t pid = fork();
    if (pid < 0) {
        CHECK(0, "fork (embd %u): %s", n_embd, std::strerror(errno));
        return 1;
    }
    if (pid == 0) {
        const int rc = run_worker_child(n_embd, soak);
        std::fflush(stdout);
        _exit(rc != 0);
    }

    char err[256];
    pulsar_tp *tp = NULL;
    int lrc = 1;
    if (!create_rank(0, port, n_embd, &tp, err, sizeof(err))) {
        CHECK(0, "leader create (embd %u): %s", n_embd, err);
    } else {
        lrc = run_rank(tp, 0, n_embd, soak);
        pulsar_tp_free(tp);
    }
    int wstatus = 0;
    if (waitpid(pid, &wstatus, 0) < 0) {
        CHECK(0, "waitpid (embd %u): %s", n_embd, std::strerror(errno));
        return 1;
    }
    const int child_fail = !WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0;
    if (lrc != 0 || child_fail) {
        CHECK(0, "width embd=%u FAILED (%s)", n_embd,
              child_fail ? "worker side" : "leader side");
        return 1;
    }
    std::printf("  wide embd=%-5u pair ok\n", n_embd);
    return 0;
}

int main(void) {
    setenv("PULSAR_TP_TIMEOUT_SEC", "10", 1);
    const char *soak_s = getenv("PULSAR_TP_SOAK_TOKENS");
    const int soak = soak_s ? atoi(soak_s) : SOAK_DEFAULT;
    std::printf("tp_wide_test: widths {4096,8192,16384}, soak=%d tokens\n", soak);
    for (int w = 0; w < N_WIDTHS; w++) run_wide_width(WIDTHS[w], soak);
    if (g_failures) {
        std::fprintf(stderr, "tp_wide_test: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("tp_wide_test: ok (wide-embd gate/batch/big + soak at all widths)\n");
    return 0;
}

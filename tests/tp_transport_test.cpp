/*
 * TP transport loopback test (branch tensor_parallel, docs/tensor-parallel-
 * port.md slice 3).  Host-only: no CUDA, no RDMA — the transport rides its
 * full-duplex TCP fallback over 127.0.0.1 (libibverbs is absent on this box,
 * so rdma_ok=0 and is_rdma() must be false).
 *
 * The test forks on the leader side: parent = LEADER (binds a free loopback
 * port, learned via getsockname) and child = WORKER (dials it, port + role
 * passed through the environment before fork).  Both sides build the same
 * pulsar_tp_identity (43 layers / 4096 embd, the slab-layout test shape),
 * hello must succeed, and every exchange is driven symmetrically by the same
 * code path on both ranks: the leader writes a known f32 pattern into its
 * gate-out slot, the pattern must land in the worker's in-slot (and vice
 * versa), batch and big gates round-trip the same way, and one lockstep pair
 * runs send_eval -> recv_command -> send_command_ack -> wait_command_ack.
 *
 * Exit: 0 on success, 1 on any mismatch; mismatches are counted in a global
 * and printed per side before _exit so the parent can aggregate.
 */
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <arpa/inet.h>
#include <netinet/in.h>
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
#define TEST_CTX_SIZE 1048576u

static pulsar_tp_identity test_identity(void) {
    pulsar_tp_identity id = {};
    id.gguf_bytes = 87000000000ull;
    id.model_id = 3u;
    id.n_layer = N_LAYER;
    id.n_embd = N_EMBD;
    id.n_vocab = 129280u;
    id.quant_bits = 2u;
    id.ctx_size = TEST_CTX_SIZE;
    id.gate_slot_start = 0u;
    id.gate_slot_step = 0u;
    id.gates_per_token = 0u;   /* identity slot mapping: slot = (seq-1) % slots */
    return id;
}

/* Deterministic f32 that is exactly representable and distinct per rank. */
static float pat(int rank, uint32_t slot, uint32_t i) {
    return (float)((int)(rank == 0 ? 100000 : 70000) + (int)slot * 1000 + (int)i);
}

static float *slot_at(uint8_t *base, const pulsar_tp_slab *s, uint32_t layer,
                      uint32_t gate, uint64_t vec, int in) {
    const uint64_t off = in ? pulsar_tp_slab_in_offset(s, layer, gate, vec)
                            : pulsar_tp_slab_out_offset(s, layer, gate, vec);
    return reinterpret_cast<float *>(base + off);
}

static void gate_phase(pulsar_tp *tp, uint8_t *base, int rank, const pulsar_tp_slab *s,
                       uint64_t vec, uint32_t layer_lo, uint32_t layer_hi,
                       uint32_t sender) {
    for (uint32_t layer = layer_lo; layer < layer_hi; layer++) {
        for (uint32_t gate = 0; gate < PULSAR_TP_GATES_PER_LAYER; gate++) {
            const uint64_t slot = (uint64_t)layer * PULSAR_TP_GATES_PER_LAYER + gate;
            const uint64_t seq = slot + 1;   /* identity-schedule slot */
            const uint32_t n = (uint32_t)(vec / sizeof(float));
            float *out = slot_at(base, s, layer, gate, vec, 0);
            float *in = slot_at(base, s, layer, gate, vec, 1);
            if (rank == (int)sender)
                for (uint32_t i = 0; i < n; i++) out[i] = pat(sender, (uint32_t)slot, i);
            else
                std::memset(out, 0, vec);
            if (!pulsar_tp_gate_exchange(tp, layer, gate, seq)) {
                CHECK(0, "rank %d gate_exchange l=%u g=%u seq=%llu failed", rank,
                      layer, gate, (unsigned long long)seq);
                return;
            }
            for (uint32_t i = 0; i < n; i++) {
                const float expected = (rank == (int)sender) ? 0.0f
                    : pat(sender, (uint32_t)slot, i);
                if (in[i] != expected) {
                    CHECK(0, "rank %d gate l=%u g=%u i=%u: in %g want %g", rank,
                          layer, gate, i, in[i], expected);
                    return;
                }
            }
        }
    }
}

static void batch_phase(pulsar_tp *tp, uint8_t *base, int rank, const pulsar_tp_slab *s,
                        uint64_t vec, uint32_t layer, uint32_t rows, uint64_t seq,
                        uint32_t sender) {
    const uint64_t bytes = (uint64_t)rows * vec;
    const uint32_t n = (uint32_t)(bytes / sizeof(float));
    float *out = reinterpret_cast<float *>(base + pulsar_tp_slab_batch_out_offset(s, layer, vec));
    float *in = reinterpret_cast<float *>(base + pulsar_tp_slab_batch_in_offset(s, layer, vec));
    if (rank == (int)sender)
        for (uint32_t i = 0; i < n; i++) out[i] = pat(sender, layer, i);
    else
        std::memset(out, 0, bytes);
    if (!pulsar_tp_batch_gate_exchange(tp, layer, rows, seq)) {
        CHECK(0, "rank %d batch_gate_exchange l=%u rows=%u failed", rank, layer, rows);
        return;
    }
    for (uint32_t i = 0; i < n; i++) {
        const float expected = (rank == (int)sender) ? 0.0f : pat(sender, layer, i);
        if (in[i] != expected) {
            CHECK(0, "rank %d batch l=%u i=%u: in %g want %g", rank, layer, i,
                  in[i], expected);
            return;
        }
    }
}

static void big_phase(pulsar_tp *tp, int rank, uint32_t layer, uint64_t seq,
                      uint64_t bytes, uint32_t sender) {
    const uint32_t n = (uint32_t)(bytes / sizeof(float));
    float *out = static_cast<float *>(std::calloc(n ? n : 1u, sizeof(float)));
    float *in = static_cast<float *>(std::calloc(n ? n : 1u, sizeof(float)));
    if (!out || !in) {
        CHECK(out && in, "rank %d big_phase alloc %llu failed", rank,
              (unsigned long long)bytes);
        std::free(out);
        std::free(in);
        return;
    }
    if (rank == (int)sender)
        for (uint32_t i = 0; i < n; i++) out[i] = pat(sender, 999, i);
    if (!pulsar_tp_big_gate_exchange(tp, layer, seq, out, in, bytes)) {
        CHECK(0, "rank %d big_gate_exchange l=%u bytes=%llu failed", rank, layer,
              (unsigned long long)bytes);
    } else {
        for (uint32_t i = 0; i < n; i++) {
            const float expected = (rank == (int)sender) ? 0.0f : pat(sender, 999, i);
            if (in[i] != expected) {
                CHECK(0, "rank %d big l=%u i=%u: in %g want %g", rank, layer, i,
                      in[i], expected);
                break;
            }
        }
    }
    std::free(out);
    std::free(in);
}

static void lockstep_phase(pulsar_tp *tp, int rank) {
    char err[256];
    const uint64_t sid = 4242u;
    if (rank == 0) {
        if (!pulsar_tp_send_eval(tp, sid, 77, 1234)) {
            CHECK(0, "leader send_eval failed");
            return;
        }
        if (!pulsar_tp_wait_command_ack(tp, sid, "eval", err, sizeof(err)))
            CHECK(0, "leader wait_command_ack failed: %s", err);
    } else {
        pulsar_tp_command cmd;
        std::memset(&cmd, 0, sizeof(cmd));
        if (!pulsar_tp_recv_command(tp, &cmd, err, sizeof(err))) {
            CHECK(0, "worker recv_command failed: %s", err);
            return;
        }
        CHECK(cmd.type == PULSAR_TP_FRAME_EVAL, "worker recv type %d want EVAL",
              (int)cmd.type);
        CHECK(cmd.session_id == sid, "worker recv session %llu want %llu",
              (unsigned long long)cmd.session_id, (unsigned long long)sid);
        CHECK(cmd.seq == 77, "worker recv seq %llu want 77",
              (unsigned long long)cmd.seq);
        CHECK(cmd.value == 1234, "worker recv token %d want 1234", cmd.value);
        pulsar_tp_command_free(&cmd);
        if (!pulsar_tp_send_command_ack(tp, sid, 0))
            CHECK(0, "worker send_command_ack failed");
    }
}

/* Round-trip every remaining control-plane frame type over the control
 * socket, strictly ordered so neither side blocks ahead.  Leader sends,
 * worker recv_commands + acks, except verify_commit and logits-half which
 * are one-way. */
static void frames_phase(pulsar_tp *tp, int rank) {
    char err[256];
    const uint64_t sid = 5150u;
    if (rank == 0) {
        const int toks[4] = { 1, 2, 3, 4 };
        CHECK(pulsar_tp_send_sync(tp, sid, toks, 4) == 1, "leader send_sync");
        CHECK(pulsar_tp_wait_command_ack(tp, sid, "sync", err, sizeof(err)),
              "leader sync ack: %s", err);

        pulsar_tp_batch_item items[2] = { {1001, 61, 0}, {1002, 62, 0} };
        CHECK(pulsar_tp_send_eval_batch(tp, items, 2) == 1,
              "leader send_eval_batch");
        CHECK(pulsar_tp_wait_command_ack(tp, sid, "eval_batch", err, sizeof(err)),
              "leader eval_batch ack: %s", err);

        const int prompt[2] = { 11, 12 };
        pulsar_tp_batch_item m = { 2001, 71, 0 };
        CHECK(pulsar_tp_send_mixed_batch(tp, 9999, prompt, 2, &m, 1) == 1,
              "leader send_mixed_batch");
        CHECK(pulsar_tp_wait_command_ack(tp, sid, "mixed_batch", err, sizeof(err)),
              "leader mixed_batch ack: %s", err);

        const int drafts[3] = { 7, 8, 9 };
        CHECK(pulsar_tp_send_verify(tp, sid, drafts, 3) == 1,
              "leader send_verify");
        CHECK(pulsar_tp_wait_command_ack(tp, sid, "verify", err, sizeof(err)),
              "leader verify ack: %s", err);

        CHECK(pulsar_tp_send_verify_commit(tp, 1, 0) == 1,
              "leader send_verify_commit");

        const float half[4] = { 0.5f, 1.5f, 2.5f, 3.5f };
        CHECK(pulsar_tp_send_logits_half(tp, half, 4) == 1,
              "leader send_logits_half");
    } else {
        pulsar_tp_command cmd;

        CHECK(pulsar_tp_recv_command(tp, &cmd, err, sizeof(err)),
              "worker recv sync: %s", err);
        CHECK(cmd.type == PULSAR_TP_FRAME_SYNC, "worker sync frame type %d",
              (int)cmd.type);
        CHECK(cmd.session_id == sid && cmd.n_tokens == 4, "worker sync header");
        CHECK(cmd.tokens && cmd.tokens[0] == 1 && cmd.tokens[3] == 4,
              "worker sync tokens");
        pulsar_tp_command_free(&cmd);
        CHECK(pulsar_tp_send_command_ack(tp, sid, 0) == 1, "worker sync ack");

        CHECK(pulsar_tp_recv_command(tp, &cmd, err, sizeof(err)),
              "worker recv eval_batch: %s", err);
        CHECK(cmd.type == PULSAR_TP_FRAME_EVAL_BATCH && cmd.n_items == 2,
              "worker eval_batch type/items");
        CHECK(cmd.items && cmd.items[0].session_id == 1001 &&
              cmd.items[0].token == 61 && cmd.items[1].token == 62,
              "worker eval_batch items");
        pulsar_tp_command_free(&cmd);
        CHECK(pulsar_tp_send_command_ack(tp, sid, 0) == 1,
              "worker eval_batch ack");

        CHECK(pulsar_tp_recv_command(tp, &cmd, err, sizeof(err)),
              "worker recv mixed_batch: %s", err);
        CHECK(cmd.type == PULSAR_TP_FRAME_MIXED_BATCH,
              "worker mixed_batch type %d", (int)cmd.type);
        CHECK(cmd.session_id == 9999 && cmd.n_tokens == 2 && cmd.n_items == 1,
              "worker mixed_batch header");
        CHECK(cmd.tokens[0] == 11 && cmd.tokens[1] == 12 &&
              cmd.items[0].token == 71, "worker mixed_batch payload");
        pulsar_tp_command_free(&cmd);
        CHECK(pulsar_tp_send_command_ack(tp, sid, 0) == 1,
              "worker mixed_batch ack");

        CHECK(pulsar_tp_recv_command(tp, &cmd, err, sizeof(err)),
              "worker recv verify: %s", err);
        CHECK(cmd.type == PULSAR_TP_FRAME_VERIFY && cmd.n_tokens == 3,
              "worker verify type/tokens");
        CHECK(cmd.tokens[0] == 7 && cmd.tokens[2] == 9, "worker verify drafts");
        pulsar_tp_command_free(&cmd);
        CHECK(pulsar_tp_send_command_ack(tp, sid, 0) == 1, "worker verify ack");

        int32_t full = 0, replay = -1;
        CHECK(pulsar_tp_recv_verify_commit(tp, &full, &replay) == 1,
              "worker recv_verify_commit");
        CHECK(full == 1 && replay == 0, "worker verify_commit values");

        float half[4];
        CHECK(pulsar_tp_recv_logits_half(tp, half, 4) == 1,
              "worker recv_logits_half");
        CHECK(half[0] == 0.5f && half[3] == 3.5f, "worker logits values");
    }
}

static int run_rank(pulsar_tp *tp, int rank) {
    static const char *rname[2] = { "leader", "worker" };
    CHECK(pulsar_tp_rank(tp) == rank, "rank %d: pulsar_tp_rank()=%d", rank,
          pulsar_tp_rank(tp));
    CHECK(!pulsar_tp_is_rdma(tp), "rank %d: expected TCP transport (no verbs device)",
          rank);
    CHECK(pulsar_tp_peer_ctx(tp) == TEST_CTX_SIZE, "rank %d: peer_ctx=%u want %u",
          rank, pulsar_tp_peer_ctx(tp), TEST_CTX_SIZE);
    CHECK(!pulsar_tp_failed(tp), "rank %d: failed() set at startup", rank);

    const uint64_t vec = (uint64_t)N_EMBD * sizeof(float);
    pulsar_tp_slab s;
    pulsar_tp_slab_layout_init(N_LAYER, N_EMBD, &s);
    uint8_t *base = static_cast<uint8_t *>(std::malloc((size_t)s.slab_bytes));
    if (!base) {
        CHECK(0, "rank %d: slab alloc %llu failed", rank,
              (unsigned long long)s.slab_bytes);
        return 1;
    }
    std::memset(base, 0, (size_t)s.slab_bytes);
    char err[256];
    if (!pulsar_tp_attach_slab(tp, base, err, sizeof(err))) {
        CHECK(0, "rank %d: attach_slab failed: %s", rank, err);
        std::free(base);
        return 1;
    }

    /* Gate exchanges over a handful of (layer,gate) pairs, one phase per
     * direction: leader writes first, worker writes second. */
    gate_phase(tp, base, rank, &s, vec, 0, 3, 0);   /* slots 0..5,  seq 1..6  L->W */
    gate_phase(tp, base, rank, &s, vec, 3, 6, 1);   /* slots 6..11, seq 7..12 W->L */

    batch_phase(tp, base, rank, &s, vec, 5, 3, 1, 0);   /* 3 rows, L->W */
    batch_phase(tp, base, rank, &s, vec, 9, 4, 1, 1);   /* 4 rows, W->L */
    batch_phase(tp, base, rank, &s, vec, 12, 8, 2, 0);  /* max 8 rows, L->W */

    big_phase(tp, rank, 0, 1, 262144ull, 0);   /* 256 KiB  L->W */
    big_phase(tp, rank, 1, 2, 3145728ull, 1);  /* 3 MiB   W->L (2 MiB round split) */

    lockstep_phase(tp, rank);
    frames_phase(tp, rank);

    /* Clean shutdown: leader stops, worker sees the STOP frame. */
    if (rank == 0) {
        if (!pulsar_tp_send_stop(tp)) CHECK(0, "rank %d: send_stop failed", rank);
    } else {
        pulsar_tp_command cmd;
        std::memset(&cmd, 0, sizeof(cmd));
        if (!pulsar_tp_recv_command(tp, &cmd, err, sizeof(err))) {
            CHECK(0, "rank %d: recv_command(STOP) failed: %s", rank, err);
        } else {
            CHECK(cmd.type == PULSAR_TP_FRAME_STOP, "rank %d: recv frame %d want STOP",
                  rank, (int)cmd.type);
            pulsar_tp_command_free(&cmd);
        }
    }

    std::free(base);
    if (g_failures)
        std::fprintf(stderr, "tp_transport_test: %s: %d FAILURE(S)\n", rname[rank],
                     g_failures);
    else
        std::fprintf(stdout, "tp_transport_test: %s ok\n", rname[rank]);
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
        CHECK(0, "worker create failed: %s", err);
        std::fprintf(stderr, "tp_transport_test: worker: create failed\n");
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
        CHECK(0, "leader create failed: %s", err);
        std::fprintf(stderr, "tp_transport_test: leader: create failed\n");
        return 1;
    }
    const int rc = run_rank(tp, 0);
    pulsar_tp_free(tp);
    return rc;
}

int main(void) {
    const char *role = getenv("TP_TRANSPORT_TEST_ROLE");

    /* Direct worker invocation (or the forked child): dial the given port. */
    if (role && std::strcmp(role, "worker") == 0) {
        const char *ps = getenv("TP_TRANSPORT_TEST_PORT");
        const int port = ps ? atoi(ps) : 0;
        const int rc = run_worker(port);
        std::fflush(stdout);
        _exit(rc != 0);
    }

    /* Leader side: pick a free loopback port, hand it to the forked worker. */
    const int port = free_port();
    if (port <= 0) {
        std::fprintf(stderr, "tp_transport_test: no free loopback port\n");
        return 1;
    }
    char pbuf[16];
    std::snprintf(pbuf, sizeof(pbuf), "%d", port);
    setenv("TP_TRANSPORT_TEST_PORT", pbuf, 1);
    setenv("TP_TRANSPORT_TEST_ROLE", "worker", 1);
    pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr, "tp_transport_test: fork failed\n");
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
        std::fprintf(stderr, "tp_transport_test: waitpid failed: %s\n",
                     std::strerror(errno));
        return 1;
    }
    const int child_fail = !WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0;
    if (lrc != 0 || child_fail) {
        std::fprintf(stderr, "tp_transport_test: FAILED (%s)\n",
                     child_fail ? "worker side" : "leader side");
        return 1;
    }
    std::printf("tp_transport_test: ok "
                "(loopback TCP: hello, attach, gate/batch/big exchange, "
                "lockstep + all control-plane frames)\n");
    return 0;
}

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

    /* Slice 4d: the vocab all-gather must CONCATENATE the ranks' ranges, not sum
     * them.  V = 1000 over n=3 gives 334/333/333, so the padded uniform stride
     * (the group exchange carries one byte count per round) is exercised, and a
     * row count > 1 exercises the per-row placement.  A mistaken allreduce here
     * would read 1000*sum(ranks) instead of 1000*owner, so the pattern makes the
     * two impossible to confuse. */
    {
        /* Two totals: 1000 (uneven for n=3/4/5, exercising the padded stride) and
         * 129280 -- the REAL n_vocab, which is uneven for every n>2 the group can
         * have.  A fixed seq per case keeps the desync guard meaningful. */
        const uint32_t totals[2] = { 1000u, 129280u };
        for (int ci = 0; ci < 2; ci++) {
        const uint32_t V = totals[ci];
        const uint32_t rows = 2u;
        const uint32_t stride = (V + (uint32_t)n - 1u) / (uint32_t)n;
        float *full = (float *)std::calloc((size_t)rows * V, sizeof(float));
        float *own = (float *)std::calloc((size_t)rows * stride, sizeof(float));
        float *scr = (float *)std::calloc((size_t)rows * stride, sizeof(float));
        uint32_t lo = 0, hi = 0;
        pulsar_tp_owned_range(rank, (uint32_t)n, V, &lo, &hi);
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t i = lo; i < hi; i++)
                own[(size_t)r * stride + (i - lo)] =
                    (float)(1000 * rank + (int)(i % 7) + 10000 * (int)r);
        if (!pulsar_tp_allgather_vocab(tp, 0, 2 + (uint64_t)ci, full, own, scr, rows, V)) {
            CHECK(0, "rank %d vocab all-gather failed (V=%u)", rank, V);
        } else {
            int vbad = 0;
            for (uint32_t r = 0; r < rows; r++) {
                for (uint32_t i = 0; i < V; i++) {
                    uint32_t owner = 0;
                    for (uint32_t k = 0; k < (uint32_t)n; k++) {
                        uint32_t a = 0, b = 0;
                        pulsar_tp_owned_range((int)k, (uint32_t)n, V, &a, &b);
                        if (i >= a && i < b) { owner = k; break; }
                    }
                    const float want =
                        (float)(1000 * (int)owner + (int)(i % 7) + 10000 * (int)r);
                    if (full[(size_t)r * V + i] != want) {
                        if (vbad == 0)
                            std::fprintf(stderr,
                                         "tp_mesh_test: rank %d vocab gather r=%u i=%u "
                                         "got %f want %f\n",
                                         rank, r, i, full[(size_t)r * V + i], want);
                        vbad++;
                    }
                }
            }
            CHECK(vbad == 0, "rank %d vocab all-gather had %d wrong elements", rank, vbad);
        }
        std::free(full);
        std::free(own);
        std::free(scr);
        }   /* totals */
    }

    /* Slice 4b-CUDA: the engine stages a <= PULSAR_TP_BATCH_MAX_ROWS gate
     * payload in the slab's OWN batch region so the RDMA big gate rides DIRECT
     * (no copy through the staging regions); a bigger prefill chunk keeps its
     * own buffer and is staged instead.  WHICH branch is taken is pure pointer
     * arithmetic, so it is assertable here on one box even though engaging the
     * QP needs a pair -- this is the half of the claim hardware cannot hide
     * behind.
     *
     * Note the predicate is SLAB-wide, not region-exact: it answers "is this
     * registered", so a payload longer than one layer's region would still read
     * as direct-able while overrunning into its neighbour's.  The engine's
     * `n_tokens <= PULSAR_TP_BATCH_MAX_ROWS` guard is what keeps that safe, so
     * the size the engine may pass is pinned here too. */
    {
        const uint64_t vec = pulsar_tp_vec_bytes(tp);
        const uint64_t slab_sz =
            pulsar_tp_slab_bytes(pulsar_tp_n_layer(tp), (uint32_t)(vec / sizeof(float)));
        int missing = 0, outside = 0;
        for (uint32_t il = 0; il < pulsar_tp_n_layer(tp); il++) {
            const void *bo = pulsar_tp_slab_batch_out(tp, il);
            const void *bi = pulsar_tp_slab_batch_in(tp, il);
            if (!bo || !bi) { missing++; continue; }
            if (!pulsar_tp_in_slab(tp, bo, PULSAR_TP_BATCH_MAX_ROWS * vec) ||
                !pulsar_tp_in_slab(tp, bi, PULSAR_TP_BATCH_MAX_ROWS * vec))
                outside++;
        }
        CHECK(missing == 0, "rank %d: slab batch region missing for %d layer(s)", rank, missing);
        CHECK(outside == 0,
              "rank %d: %d layer(s) stage OUTSIDE the slab -- they would NOT ride direct",
              rank, outside);
        /* A heap payload is what a prefill chunk passes: it must NOT read as
         * in-slab, or the transport would hand the NIC an unregistered address. */
        CHECK(!pulsar_tp_in_slab(tp, out, bytes),
              "rank %d: a heap payload read as in-slab (prefill must stage)", rank);
        CHECK(!pulsar_tp_in_slab(tp, (const void *)(uintptr_t)0x1000, 64),
              "rank %d: an unrelated address read as in-slab", rank);
        CHECK(!pulsar_tp_in_slab(tp, NULL, 0), "rank %d: NULL read as in-slab", rank);
        /* Longer than the whole slab: the bound must hold, not just the base. */
        CHECK(!pulsar_tp_in_slab(tp, pulsar_tp_slab_batch_out(tp, 0), slab_sz + 1),
              "rank %d: a payload longer than the slab read as in-slab", rank);
        /* A layer past n_layer has no region at all. */
        CHECK(pulsar_tp_slab_batch_out(tp, pulsar_tp_n_layer(tp)) == NULL &&
              pulsar_tp_slab_batch_in(tp, pulsar_tp_n_layer(tp)) == NULL,
              "rank %d: a layer past n_layer returned a batch region", rank);
    }

    /* n-way per-layer gate (n>=3): `out` keeps THIS rank's partial and `in`
     * accumulates the SUM of every peer's -- the pair's contract generalised
     * from "the peer's partial" to "every peer's".  Integer partials so the sum
     * is exact.  Skipped at n=2, where the pair takes its RDMA path and needs an
     * armed decode window; that path is covered by tp_sched_test. */
    if (n >= 3) {
        pulsar_tp_slab lay;
        pulsar_tp_slab_layout_init(MESH_N_LAYER, MESH_N_EMBD, &lay);
        const uint64_t vec = pulsar_tp_vec_bytes(tp);
        const uint64_t nelt = vec / sizeof(float);
        float *op = (float *)((uint8_t *)slab + pulsar_tp_slab_out_offset(&lay, 0, 0, vec));
        float *ip = (float *)((uint8_t *)slab + pulsar_tp_slab_in_offset(&lay, 0, 0, vec));
        for (uint64_t k = 0; k < nelt; k++) op[k] = (float)(1000 * rank + (int)(k % 7));
        memset(ip, 0, (size_t)vec);
        if (!pulsar_tp_gate_exchange(tp, 0, 0, 7)) {
            CHECK(0, "rank %d n-way gate_exchange failed", rank);
        } else {
            const float base = 1000.0f * (float)(n * (n - 1) / 2) - 1000.0f * (float)rank;
            int gbad = 0;
            for (uint64_t k = 0; k < nelt; k++) {
                if (ip[k] != base + (float)(n - 1) * (float)(int)(k % 7)) gbad++;
                if (op[k] != (float)(1000 * rank + (int)(k % 7))) gbad++;   /* out stays local */
            }
            CHECK(gbad == 0,
                  "rank %d n-way gate: %d wrong elements (in must be the peers' sum, "
                  "out must still be local)", rank, gbad);
        }
    }

    /* Same contract for the VERIFY-BATCH gate (rows<=8): every peer's ROWS
     * accumulate into the batch-in region while batch-out keeps this rank's. */
    if (n >= 3) {
        pulsar_tp_slab lay;
        pulsar_tp_slab_layout_init(MESH_N_LAYER, MESH_N_EMBD, &lay);
        const uint64_t vec = pulsar_tp_vec_bytes(tp);
        const uint32_t rws = 2u;
        const uint64_t nelt = (uint64_t)rws * vec / sizeof(float);
        float *bo = (float *)((uint8_t *)slab + pulsar_tp_slab_batch_out_offset(&lay, 0, vec));
        float *bi = (float *)((uint8_t *)slab + pulsar_tp_slab_batch_in_offset(&lay, 0, vec));
        for (uint64_t k = 0; k < nelt; k++) bo[k] = (float)(1000 * rank + (int)(k % 5));
        memset(bi, 0, (size_t)((uint64_t)rws * vec));
        if (!pulsar_tp_batch_gate_exchange(tp, 0, rws, 11)) {
            CHECK(0, "rank %d n-way batch gate failed", rank);
        } else {
            const float base = 1000.0f * (float)(n * (n - 1) / 2) - 1000.0f * (float)rank;
            int bbad = 0;
            for (uint64_t k = 0; k < nelt; k++) {
                if (bi[k] != base + (float)(n - 1) * (float)(int)(k % 5)) bbad++;
                if (bo[k] != (float)(1000 * rank + (int)(k % 5))) bbad++;
            }
            CHECK(bbad == 0, "rank %d n-way batch gate: %d wrong elements", rank, bbad);
        }
    }

    /* The n-way BIG gate (the prefill-chunk shape) over CALLER buffers, same
     * contract: `out` keeps this rank's partial, `in` accumulates the peers'. */
    if (n >= 3) {
        float *bo = (float *)std::malloc((size_t)bytes);
        float *bi = (float *)std::malloc((size_t)bytes);
        const uint64_t nelt_bg = bytes / sizeof(float);
        for (uint64_t k = 0; k < nelt_bg; k++) bo[k] = (float)(1000 * rank + (int)(k % 3));
        memset(bi, 0, (size_t)bytes);
        if (!pulsar_tp_big_gate_exchange(tp, 0, 13, bo, bi, bytes)) {
            CHECK(0, "rank %d n-way big gate failed", rank);
        } else {
            const float base = 1000.0f * (float)(n * (n - 1) / 2) - 1000.0f * (float)rank;
            int gb2 = 0;
            for (uint64_t k = 0; k < nelt_bg; k++) {
                if (bi[k] != base + (float)(n - 1) * (float)(int)(k % 3)) gb2++;
                if (bo[k] != (float)(1000 * rank + (int)(k % 3))) gb2++;
            }
            CHECK(gb2 == 0, "rank %d n-way big gate: %d wrong elements", rank, gb2);
        }
        std::free(bo);
        std::free(bi);
    }

    /* The VERIFY family across the mesh: rank 0 BROADCASTS the drafts, every
     * worker receives them and commits its verdict, and rank 0 collects one
     * commit PER PEER and requires them to agree.  All workers commit the same
     * verdict here, which is what a correct group does (they verify against the
     * same assembled logits). */
    if (n >= 3) {
        const int drafts[3] = { 11, 22, 33 };
        char verr[256];
        verr[0] = 0;
        if (rank == 0) {
            if (!pulsar_tp_send_verify(tp, 0xFEED1234ULL, drafts, 3)) {
                CHECK(0, "rank 0 send_verify (broadcast) failed");
            } else {
                int32_t fa = -1, rn = -1;
                if (!pulsar_tp_recv_verify_commit(tp, &fa, &rn)) {
                    CHECK(0, "rank 0 recv_verify_commit over %d peers failed: %s",
                          n - 1, verr);
                } else {
                    CHECK(fa == 0 && rn == 3,
                          "rank 0 collected commit (%d,%d), expected (0,3)",
                          (int)fa, (int)rn);
                }
            }
        } else {
            pulsar_tp_command cmd;
            if (!pulsar_tp_recv_command(tp, &cmd, verr, sizeof(verr))) {
                CHECK(0, "rank %d verify recv_command: %s", rank, verr);
            } else {
                CHECK(cmd.type == PULSAR_TP_FRAME_VERIFY,
                      "rank %d received frame %d, expected VERIFY (%d)",
                      rank, (int)cmd.type, (int)PULSAR_TP_FRAME_VERIFY);
                pulsar_tp_command_free(&cmd);
                CHECK(pulsar_tp_send_verify_commit(tp, 0, 3),
                      "rank %d send_verify_commit failed", rank);
            }
        }
    }

    /* Session-command plane across the mesh (n>2): rank 0 broadcasts ONE
     * command and collects one ack PER PEER; every other rank receives it and
     * acks.  This is the control path an n-rank session mirror rides, so it
     * only works if both the broadcast and the per-peer collect are n-general
     * -- which is precisely what the pair-only version could not do. */
    for (int round = 0; round < 2; round++) {
        /* TWO rounds with distinct session ids: a collect that read only the
         * FIRST peer's ack would leave the other peers' acks queued, and round
         * two would then read a stale ack whose session id does not match --
         * so this catches a partial collect, which a single round would not. */
        const uint64_t sid = 0xC0DE1234ULL + (uint64_t)round;
        char cerr[256];
        cerr[0] = 0;
        if (rank == 0) {
            if (!pulsar_tp_send_session_create(tp, sid, 4096)) {
                CHECK(0, "rank 0 command broadcast failed");
            } else {
                CHECK(pulsar_tp_wait_command_ack(tp, sid, "session create",
                                                 cerr, sizeof(cerr)),
                      "rank 0 wait_command_ack over %d peers: %s", n - 1, cerr);
            }
        } else {
            pulsar_tp_command cmd;
            if (!pulsar_tp_recv_command(tp, &cmd, cerr, sizeof(cerr))) {
                CHECK(0, "rank %d recv_command: %s", rank, cerr);
            } else {
                CHECK(cmd.type == PULSAR_TP_FRAME_SESSION_CREATE,
                      "rank %d received frame type %d, expected SESSION_CREATE (%d)",
                      rank, (int)cmd.type, (int)PULSAR_TP_FRAME_SESSION_CREATE);
                CHECK(cmd.session_id == sid,
                      "rank %d received session %llu, expected %llu", rank,
                      (unsigned long long)cmd.session_id, (unsigned long long)sid);
                CHECK(pulsar_tp_send_command_ack(tp, sid, 0),
                      "rank %d ack failed", rank);
                pulsar_tp_command_free(&cmd);
            }
        }
    }

    pulsar_tp_free(tp);
    std::free(slab);
    std::free(out);
    std::free(in);
    /* CHECK failures must fail the rank, not just print: the vocab all-gather
     * block below asserts through CHECK, and returning only the all-reduce
     * counter made every CHECK in this file reportable but not gating (caught
     * by mutation: forcing the gather's placement offset to 0 printed 2000
     * wrong elements per rank and still exited 0). */
    return (bad == 0 && g_failures == 0) ? 0 : 1;
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
    for (int n = 2; n <= 5; n++) {
        std::printf("tp_mesh_test: n=%d\n", n);
        std::fflush(stdout);
        if (run_mesh(n) != 0) rc = 1;
        std::fflush(stdout);
        std::fflush(stderr);
    }
    if (rc == 0)
        std::printf("tp_mesh_test: ok (n=2..5 mesh + all-reduce + vocab all-gather, exact)\n");
    else
        std::printf("tp_mesh_test: FAILED\n");
    return rc;
}

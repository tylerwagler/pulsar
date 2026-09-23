/* 4e ENGINE-MIRROR test (L238): the worker RECEIVE LOOP and the leader wrappers.
 *
 * WHY THIS EXISTS
 * ---------------
 * `tp_mesh_test` proves the TRANSPORT and the single-Spark gates prove the
 * NON-TP path; neither enters the engine layer between them -- the leader
 * wrappers that ship frames and the worker loop that applies them.  That gap
 * once shipped an inverted return convention at seven call sites, and later a
 * batch frame that never carried its session id (a correct batched decode
 * would have been refused in production).  A two-process engine pair cannot
 * cover this on one box (earlyoom kills two 86 GB engines during load), so
 * this test takes the other route.
 *
 * HOW
 * ---
 * The worker rank runs the REAL loop, `pulsar_tp_worker_run`, on a fabricated
 * engine: a zeroed pulsar_engine carrying a real mesh transport and an empty
 * session registry.  Every frame the leader sends names a session the worker
 * never created, so every path asserted here returns BEFORE the loop touches
 * a graph -- no model, no weights, no GPU work -- and the assertions are the
 * plumbing that matters:
 *
 *   A. sync / eval / batched decode / mixed step for an unknown session
 *      -> refused by name, and the refusal ACKED so the leader reads it at once;
 *      the verdict frames (bank restore, fork) answer a NEGATIVE status that
 *      the verdict collector reads as a refusal, never as a result
 *   B. a void frame (rewind) for an unknown session -> the worker marks the
 *      pair failed with NO ack; the NEXT acked frame carries the refusal back
 *      as a failed ack, not as a deadline ("did not answer")
 *   C. STOP ends the loop cleanly (rc 0) even on a failed pair, and a worker
 *      that has stopped but stays alive makes the leader's collect time out
 *   D. the leader wrapper on a dead transport refuses before sending
 *   E. a worker rank calling a leader wrapper is refused by name (the
 *      same-driver model this slice retired)
 *
 * The target sets PULSAR_TP_TIMEOUT_SEC=1 and runs under `timeout`: a missing
 * ack would otherwise HANG the leader, and a hang is not a test result. */

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "pulsar.h"
#include "engine/pulsar_engine_internal.h"
#include "tp/pulsar_tp.h"

#define MIRROR_N_LAYER 2u
#define MIRROR_N_EMBD 64u

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

/** An engine the graph never sees: a real transport, an empty registry. */
static pulsar_engine *fabricate_engine(pulsar_tp *tp) {
    pulsar_engine *e = (pulsar_engine *)std::calloc(1, sizeof(*e));
    if (!e) {
        std::fprintf(stderr, "tp_mirror_test: out of memory\n");
        std::exit(1);
    }
    e->tp = tp;
    return e;
}

/** A session the graph never sees, on the LEADER: its wrappers must refuse
 * before touching it. */
static pulsar_session *fabricate_session(pulsar_engine *e, uint64_t mirror_id) {
    pulsar_session *s = (pulsar_session *)std::calloc(1, sizeof(*s));
    if (!s) {
        std::fprintf(stderr, "tp_mirror_test: out of memory\n");
        std::exit(1);
    }
    s->engine = e;
    s->tp_session_id = mirror_id;
    return s;
}

static void fill_prompt(pulsar_tokens *p, int *v, int n) {
    std::memset(p, 0, sizeof(*p));
    for (int i = 0; i < n; i++) v[i] = 100 + i;
    p->v = v;
    p->len = n;
    p->cap = n;
}

#define SID 7ull   /* a session the worker never created */

static int run_leader(pulsar_tp *tp) {
    char err[512];
    int toks[16];
    pulsar_tokens prompt;
    fill_prompt(&prompt, toks, 16);

    /* A. Four acked operations for an unknown session: each must come back as
     * a failed ACK naming the operation.  send_* reporting nonzero is itself
     * the convention check. */
    CHECK(pulsar_tp_send_sync(tp, SID, prompt.v, (uint32_t)prompt.len) != 0,
          "send_sync must report success as nonzero");
    err[0] = 0;
    CHECK(!pulsar_tp_wait_command_ack(tp, SID, "sync", err, sizeof(err)) &&
          std::strstr(err, "sync failed") != NULL,
          "the leader must read the worker's sync refusal as a failed ack: %s", err);

    CHECK(pulsar_tp_send_eval(tp, SID, 4242, 99) != 0, "send_eval must report success as nonzero");
    err[0] = 0;
    CHECK(!pulsar_tp_wait_command_ack(tp, SID, "eval", err, sizeof(err)) &&
          std::strstr(err, "eval failed") != NULL,
          "the leader must read the worker's eval refusal: %s", err);

    pulsar_tp_batch_item items[2];
    for (int i = 0; i < 2; i++) {
        items[i].session_id = SID;
        items[i].bank = 1;
        items[i].pos = 10 + i;
        items[i].token = 1000 + i;
        items[i].reserved = 0;
    }
    CHECK(pulsar_tp_send_eval_batch(tp, items, 2) != 0, "send_eval_batch must report success as nonzero");
    err[0] = 0;
    CHECK(!pulsar_tp_wait_command_ack(tp, SID, "batch decode", err, sizeof(err)) &&
          std::strstr(err, "batch decode failed") != NULL,
          "the leader must read the worker's batch refusal (the frame now carries its session id): %s", err);

    CHECK(pulsar_tp_send_mixed_batch(tp, items, 2, 3u) != 0, "send_mixed_batch must report success as nonzero");
    err[0] = 0;
    CHECK(!pulsar_tp_wait_command_ack(tp, SID, "mixed batch", err, sizeof(err)) &&
          std::strstr(err, "mixed batch failed") != NULL,
          "the leader must read the worker's mixed refusal: %s", err);

    /* A'. The verdict frames for an unknown session: the worker answers a
     * NEGATIVE status and the verdict collector reads it as a refusal. */
    {
        int status = -99;
        CHECK(pulsar_tp_send_bank_state_restore(tp, SID, 1u) != 0, "send_bank_state_restore must report success");
        err[0] = 0;
        CHECK(!pulsar_tp_wait_command_status(tp, SID, "bank state restore", &status, err, sizeof(err)) &&
              std::strstr(err, "refused") != NULL,
              "an unknown-session restore must come back as a refusal, not a verdict: %s", err);
        const int t3[3] = { 1, 2, 3 };
        CHECK(pulsar_tp_send_bank_fork(tp, 1, SID, 0u, 1u, t3, 3u, 2) != 0, "send_bank_fork must report success");
        err[0] = 0;
        CHECK(!pulsar_tp_wait_command_status(tp, SID, "partial bank fork", &status, err, sizeof(err)) &&
              std::strstr(err, "refused") != NULL,
              "an unknown-session fork must come back as a refusal: %s", err);
        CHECK(pulsar_tp_send_rewrite_from_common(tp, SID, t3, 3u, 1) != 0, "send_rewrite must report success");
        err[0] = 0;
        CHECK(!pulsar_tp_wait_command_status(tp, SID, "rewrite from common", &status, err, sizeof(err)) &&
              std::strstr(err, "refused") != NULL,
              "an unknown-session rewrite must come back as a refusal: %s", err);
        const float lg[4] = { 0.f, 1.f, 2.f, 3.f };
        CHECK(pulsar_tp_send_set_logits(tp, SID, lg, 4u) != 0, "send_set_logits must report success");
        err[0] = 0;
        CHECK(!pulsar_tp_wait_command_status(tp, SID, "set logits", &status, err, sizeof(err)) &&
              std::strstr(err, "refused") != NULL,
              "an unknown-session set-logits must come back as a refusal: %s", err);
        pulsar_tp_spec_command sc;
        std::memset(&sc, 0, sizeof(sc));
        sc.session_id = SID; sc.rng = 42;
        CHECK(pulsar_tp_send_spec(tp, PULSAR_TP_FRAME_SPEC_NEXT_BASE, &sc, NULL, NULL) != 0, "send_spec must report success");
        err[0] = 0;
        CHECK(!pulsar_tp_wait_command_status(tp, SID, "spec_next_base", &status, err, sizeof(err)) &&
              std::strstr(err, "refused") != NULL,
              "an unknown-session spec_next_base must come back as a refusal: %s", err);
        CHECK(pulsar_tp_send_bank_kv(tp, 1, SID, 2u, "kv-none.bin") != 0, "send_bank_kv must report success");
        err[0] = 0;
        CHECK(!pulsar_tp_wait_command_status(tp, SID, "bank kv load", &status, err, sizeof(err)) &&
              std::strstr(err, "refused") != NULL,
              "an unknown-session kv load must come back as a refusal: %s", err);
    }

    /* B. A void frame for an unknown session marks the worker's pair failed
     * silently; the next acked frame must carry that back AT ONCE. */
    CHECK(pulsar_tp_send_rewind(tp, SID, 12) != 0, "send_rewind must report success as nonzero");
    CHECK(pulsar_tp_send_sync(tp, SID, prompt.v, 4) != 0, "post-void sync send failed");
    err[0] = 0;
    CHECK(!pulsar_tp_wait_command_ack(tp, SID, "sync", err, sizeof(err)),
          "the leader must see the worker's refusal after the void divergence");
    CHECK(std::strstr(err, "sync failed") != NULL && std::strstr(err, "did not answer") == NULL,
          "propagation must be a failed ack, not a deadline: %s", err);

    /* C. STOP ends the loop; the worker then stays alive and silent past the
     * control-plane deadline, so this collect must time out rather than hang. */
    CHECK(pulsar_tp_send_stop(tp) != 0, "send_stop must report success as nonzero");
    {
        char terr[256];
        terr[0] = 0;
        CHECK(!pulsar_tp_wait_command_ack(tp, SID, "silent peer", terr, sizeof(terr)) &&
              std::strstr(terr, "did not answer") != NULL,
              "a stopped-but-alive peer must time the collect out, not satisfy it: %s", terr);
    }

    /* D. A leader whose transport is already dead refuses before it sends
     * anything.  Last, because marking the pair failed poisons the transport. */
    pulsar_tp_mark_failed(tp);
    pulsar_engine *e = fabricate_engine(tp);
    pulsar_session *s = fabricate_session(e, SID);
    err[0] = 0;
    const int rc = pulsar_session_sync_mm(s, &prompt, NULL, 0, err, sizeof(err));
    CHECK(rc == 1 && std::strstr(err, "transport failed earlier") != NULL,
          "a leader on a dead transport must refuse before sending, got rc=%d: %s", rc, err);
    std::free(s);
    std::free(e);
    return g_failures;
}

static int run_worker(pulsar_tp *tp) {
    char err[512];
    err[0] = 0;
    pulsar_engine *e = fabricate_engine(tp);

    /* E. A worker rank does not drive sessions: the leader wrappers refuse it
     * by name, before any frame moves. */
    {
        pulsar_session *s = fabricate_session(e, SID);
        int toks[4];
        pulsar_tokens prompt;
        fill_prompt(&prompt, toks, 4);
        const int rc = pulsar_session_sync_mm(s, &prompt, NULL, 0, err, sizeof(err));
        CHECK(rc == 1 && std::strstr(err, "is a worker") != NULL,
              "a worker calling a leader wrapper must be refused by name, got rc=%d: %s", rc, err);
        pulsar_session *created = (pulsar_session *)0x1;
        CHECK(pulsar_session_create(&created, e, 4096) == 1 && created == NULL,
              "a worker creating a session from its own driver must be refused");
        std::free(s);
    }

    /* The real loop, on the fabricated engine.  Every frame names a session it
     * never created, so nothing here reaches a graph. */
    err[0] = 0;
    const int rc = pulsar_tp_worker_run(e, err, sizeof(err));
    CHECK(rc == 0, "the loop must end cleanly on STOP even with the pair marked failed, got rc=%d: %s", rc, err);
    CHECK(pulsar_tp_failed(tp), "the void divergence (rewind for an unknown session) must have marked the pair failed");
    CHECK(e->tp_worker_n == 0 && e->tp_worker_slots == NULL,
          "the registry must be empty and released after the loop");
    std::free(e);

    /* Stay ALIVE and SILENT past the leader's deadline (the target sets
     * PULSAR_TP_TIMEOUT_SEC=1) so its last collect tests a timeout, not a
     * closed channel. */
    std::fflush(stdout);
    std::fflush(stderr);
    sleep(3);
    return g_failures;
}

static int run_mirror_rank(int rank, int n, const int *ports) {
    char err[512];
    char peers[1024] = "";
    for (int k = 0; k < n; k++) {
        char one[128];
        std::snprintf(one, sizeof(one), "%s127.0.0.1:%d", k ? "," : "", ports[k]);
        std::strncat(peers, one, sizeof(peers) - std::strlen(peers) - 1);
    }
    pulsar_tp_options opt;
    std::memset(&opt, 0, sizeof(opt));
    opt.role = PULSAR_TP_ROLE_LEADER;   /* create_mesh orders by `rank`, not role */
    opt.rank = rank;
    opt.n_ranks = n;
    opt.peers = peers;
    opt.port = ports[rank];

    pulsar_tp *tp = NULL;
    pulsar_tp_identity id;
    std::memset(&id, 0, sizeof(id));
    id.gguf_bytes = 123456789ull;
    id.model_id = 1u;
    id.n_layer = MIRROR_N_LAYER;
    id.n_embd = MIRROR_N_EMBD;
    id.n_vocab = 1000u;
    id.quant_bits = 2u;
    if (!pulsar_tp_create_mesh(&tp, &opt, &id, err, sizeof(err))) {
        CHECK(0, "rank %d mesh create failed: %s", rank, err);
        return 1;
    }
    const int rc = rank == 0 ? run_leader(tp) : run_worker(tp);
    pulsar_tp_free(tp);
    return rc != 0;
}

int main(void) {
    const int n = 2;
    int ports[8];
    for (int k = 0; k < n; k++) {
        ports[k] = free_port();
        if (ports[k] <= 0) {
            std::fprintf(stderr, "tp_mirror_test: no free loopback port (rank %d)\n", k);
            return 1;
        }
    }
    pid_t pids[8];
    for (int k = 1; k < n; k++) {
        pid_t pid = fork();
        if (pid < 0) {
            std::fprintf(stderr, "tp_mirror_test: fork failed\n");
            return 1;
        }
        if (pid == 0) {
            const int rc = run_mirror_rank(k, n, ports);
            std::fflush(stdout);
            std::fflush(stderr);
            _exit(rc != 0);
        }
        pids[k] = pid;
    }
    int rc = run_mirror_rank(0, n, ports);
    for (int k = 1; k < n; k++) {
        int wstatus = 0;
        if (waitpid(pids[k], &wstatus, 0) < 0) continue;
        if (!WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0) rc = 1;
    }
    if (rc == 0)
        std::printf("tp_mirror_test: ok (worker loop refusals + acks, void divergence propagation, clean STOP, dead-transport and worker-rank refusals)\n");
    else
        std::printf("tp_mirror_test: FAILED\n");
    return rc;
}

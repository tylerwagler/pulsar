/* 4e ENGINE-MIRROR test: the layer between the transport and the graph.
 *
 * WHY THIS EXISTS
 * ---------------
 * `tp_mesh_test` proves the TRANSPORT (frames, acks, all-reduce) and the
 * single-Spark gates prove the NON-TP path, but neither ever enters the engine
 * wrappers that decide whether a session operation is mirrored at all.  That
 * gap shipped a real bug: all seven transport send/recv calls in the first
 * three mirroring increments read the transport's convention backwards
 * (nonzero is SUCCESS), so the leader refused every successful send and the
 * worker treated a failed ack as delivered.  A two-process engine pair cannot
 * cover this on one box -- earlyoom kills it during model load -- so this test
 * takes the other route.
 *
 * HOW
 * ---
 * A session is FABRICATED: a zeroed pulsar_engine carrying a real mesh
 * transport, and a zeroed pulsar_session whose `tp_session_id` is deliberately
 * NOT the id the leader mirrors under.  Every path asserted here returns
 * BEFORE the wrapper reaches the graph, so no model, no weights and no GPU work
 * are needed -- and the assertions are exactly the plumbing that was inverted:
 *
 *   A. worker, session-id mismatch  -> refuse loudly, and ack the refusal
 *   B. worker, frame-type mismatch  -> refuse loudly, and ack the refusal
 *   C. leader, dead transport       -> refuse before sending anything
 *
 * The leader side of A and B drives the transport directly (pulsar_tp_send_sync
 * / send_eval + wait_command_ack), because the leader's wrapper would call
 * sync() on the fabricated session.  What that still proves is the half that
 * matters: the worker's ack reaches the leader AND the leader reports the
 * failure, which is only true if send reports success as nonzero and the ack
 * carries a nonzero status.
 *
 * The target is run under `timeout`: a missing ack would otherwise HANG the
 * leader in wait_command_ack rather than fail it, and a hang is not a test
 * result. */

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

/** A session the graph never sees.  Every path this test asserts on returns
 * before the wrapper dereferences the graph, so the all-zero graph is not a
 * lie the test tells -- it is the boundary of what is being asserted. */
static pulsar_session *fabricate_session(pulsar_tp *tp, uint64_t mirror_id) {
    pulsar_session *s = (pulsar_session *)std::calloc(1, sizeof(*s));
    pulsar_engine *e = (pulsar_engine *)std::calloc(1, sizeof(*e));
    if (!s || !e) {
        std::fprintf(stderr, "tp_mirror_test: out of memory\n");
        std::exit(1);
    }
    e->tp = tp;
    s->engine = e;
    s->tp_session_id = mirror_id;
    return s;
}

static void free_session(pulsar_session *s) {
    pulsar_engine *e = s->engine;
    std::free(s);
    std::free(e);
}

/** A borrowed token view: the wrapper must not free a caller's buffer. */
static void fill_prompt(pulsar_tokens *p, int *v, int n) {
    std::memset(p, 0, sizeof(*p));
    for (int i = 0; i < n; i++) v[i] = 100 + i;
    p->v = v;
    p->len = n;
    p->cap = n;
}

/* The leader: the id every frame carries, and the id the worker does NOT have. */
#define LEADER_SID 7ull
#define WORKER_SID 8ull

static int run_leader(pulsar_tp *tp) {
    char err[512];
    int toks[16];
    pulsar_tokens prompt;
    fill_prompt(&prompt, toks, 16);

    /* A. The worker holds session 8; every frame here says 7, so the worker
     * must refuse and say so.  send_sync reporting nonzero is itself the
     * convention check: a zero here means the transport refused a good frame. */
    CHECK(pulsar_tp_send_sync(tp, LEADER_SID, prompt.v, (uint32_t)prompt.len) != 0,
          "send_sync must report success as nonzero");
    err[0] = 0;
    CHECK(!pulsar_tp_wait_command_ack(tp, LEADER_SID, "sync", err, sizeof(err)),
          "the leader must see the worker's refusal, not a success");
    CHECK(std::strstr(err, "sync failed") != NULL,
          "the refusal must name the operation the leader was waiting on: %s", err);

    /* B. A frame the worker is not expecting (EVAL where it called sync): the
     * refusal must come from the frame-TYPE check, and travel the same way. */
    CHECK(pulsar_tp_send_eval(tp, LEADER_SID, 4242, 99) != 0,
          "send_eval must report success as nonzero");
    err[0] = 0;
    CHECK(!pulsar_tp_wait_command_ack(tp, LEADER_SID, "eval", err, sizeof(err)),
          "the leader must see the worker's frame-type refusal");
    CHECK(std::strstr(err, "eval failed") != NULL,
          "the refusal must name the operation: %s", err);

    /* C. The batched decode.  The worker is on session 8 again, so the frame's
     * id (7) must stop it before it builds a single row -- and this is the only
     * place the EVAL_BATCH payload's encode/decode is driven by the engine's own
     * sender. */
    {
        pulsar_tp_batch_item items[2];
        for (int i = 0; i < 2; i++) {
            items[i].session_id = LEADER_SID;
            items[i].bank = 1;
            items[i].pos = 10 + i;
            items[i].token = 1000 + i;
            items[i].reserved = 0;
        }
        CHECK(pulsar_tp_send_eval_batch(tp, items, 2) != 0,
              "send_eval_batch must report success as nonzero");
        err[0] = 0;
        CHECK(!pulsar_tp_wait_command_ack(tp, LEADER_SID, "batch decode", err, sizeof(err)),
              "the leader must see the worker's batch refusal");
        CHECK(std::strstr(err, "batch decode failed") != NULL,
              "the batch refusal must name the operation: %s", err);
    }

    /* E. The mixed step, on its own frame type.  Same divergence, but this also
     * pins WHICH frame the wrapper expects: decode_mixed and decode_multiseq
     * take identical rows, so the type is the only thing that says which
     * contract the leader is in. */
    {
        pulsar_tp_batch_item items[2];
        for (int i = 0; i < 2; i++) {
            items[i].session_id = LEADER_SID;
            items[i].bank = 4;
            items[i].pos = 900 + i;
            items[i].token = 80000 + i;
            items[i].reserved = 0;
        }
        CHECK(pulsar_tp_send_mixed_batch(tp, items, 2) != 0,
              "send_mixed_batch must report success as nonzero");
        err[0] = 0;
        CHECK(!pulsar_tp_wait_command_ack(tp, LEADER_SID, "mixed batch", err, sizeof(err)),
              "the leader must see the worker's mixed-batch refusal");
        CHECK(std::strstr(err, "mixed batch failed") != NULL,
              "the mixed refusal must name the operation: %s", err);
    }

    /* F. A leader whose transport is already dead refuses before it sends
     * anything -- no frame, no graph, no half-mirrored operation.  Last,
     * because marking the pair failed poisons the transport for good. */
    pulsar_tp_mark_failed(tp);
    pulsar_session *s = fabricate_session(tp, LEADER_SID);
    err[0] = 0;
    const int rc = pulsar_session_sync_mm(s, &prompt, NULL, 0, err, sizeof(err));
    CHECK(rc == 1, "a leader on a dead transport must refuse, got rc=%d", rc);
    CHECK(std::strstr(err, "transport failed earlier") != NULL,
          "the refusal must name the dead transport: %s", err);
    free_session(s);
    return g_failures;
}

static int run_worker(pulsar_tp *tp) {
    char err[512];
    int toks[16];
    pulsar_tokens prompt;
    fill_prompt(&prompt, toks, 16);

    /* A. Session-id mismatch: the wrapper must refuse and ACK the refusal.  If
     * it returns without acking, the leader's collect above hangs -- which the
     * `timeout` on this target turns into a failure rather than a hang. */
    pulsar_session *s = fabricate_session(tp, WORKER_SID);
    err[0] = 0;
    int rc = pulsar_session_sync_mm(s, &prompt, NULL, 0, err, sizeof(err));
    CHECK(rc == 1, "the worker must refuse a frame for another session, got rc=%d", rc);
    CHECK(std::strstr(err, "diverged") != NULL,
          "the refusal must name the divergence: %s", err);

    /* B. Frame-type mismatch, with the session id now MATCHING: the refusal has
     * to come from the type check, so this covers the branch the id check
     * short-circuits in A. */
    free_session(s);
    s = fabricate_session(tp, LEADER_SID);
    err[0] = 0;
    rc = pulsar_session_sync_mm(s, &prompt, NULL, 0, err, sizeof(err));
    CHECK(rc == 1, "the worker must refuse an unexpected frame type, got rc=%d", rc);
    CHECK(std::strstr(err, "expected a mirrored sync") != NULL,
          "the refusal must name the frame it expected: %s", err);

    /* D. The batched decode, same divergence: the wrapper must refuse on the
     * frame's session id before it builds a row, and ack.  A fresh session on
     * the MISMATCHED id, because the id is what has to stop it -- with a
     * matching id the wrapper would go on to decode into the fabricated
     * session's graph. */
    free_session(s);
    s = fabricate_session(tp, WORKER_SID);
    {
        pulsar_multiseq_req rows[2];
        rows[0].bank = 1; rows[0].pos = 10; rows[0].token = 1000;
        rows[1].bank = 1; rows[1].pos = 11; rows[1].token = 1001;
        float logits[8] = { 0 };
        err[0] = 0;
        rc = pulsar_session_decode_multiseq(s, rows, 2, logits, 8, err, sizeof(err));
        CHECK(rc == 1, "the worker must refuse a batch for another session, got rc=%d", rc);
        CHECK(std::strstr(err, "diverged") != NULL,
              "the batch refusal must name the divergence: %s", err);
    }
    free_session(s);

    /* E. The mixed step: a fresh session on the mismatched id again, so the
     * frame's session id is what stops it. */
    s = fabricate_session(tp, WORKER_SID);
    {
        pulsar_multiseq_req rows[2];
        rows[0].bank = 4; rows[0].pos = 900; rows[0].token = 80000;
        rows[1].bank = 4; rows[1].pos = 901; rows[1].token = 80001;
        float logits[8] = { 0 };
        uint32_t out_rows = 0;
        err[0] = 0;
        rc = pulsar_session_decode_mixed(s, rows, 2, logits, 8, &out_rows,
                                         PULSAR_MSEQ_HEAD_ALL_ROWS, err, sizeof(err));
        CHECK(rc == 1, "the worker must refuse a mixed batch for another session, got rc=%d", rc);
        CHECK(std::strstr(err, "diverged") != NULL,
              "the mixed refusal must name the divergence: %s", err);

        /* F. Speculation FAILS CLOSED on a pair whose rng stream it does not
         * share: the accept walk draws from the caller's rng, so a round may
         * only begin once pulsar_session_spec_next_base has taken the leader's
         * state.  The refusal comes from the round gate before anything reads
         * the graph, which is what makes it assertable on a fabricated
         * session. */
        pulsar_spec_round *round = pulsar_spec_round_new();
        CHECK(round != NULL, "pulsar_spec_round_new failed");
        if (round) {
            err[0] = 0;
            const int brc = pulsar_session_spec_round_begin(s, round, 12345, 8, 8,
                                                            0.0f, 0, 1.0f, 0.0f,
                                                            err, sizeof(err));
            CHECK(brc != 0, "a pair must refuse to begin a spec round, got rc=%d", brc);
            CHECK(std::strstr(err, "has not synchronized its speculation rng") != NULL,
                  "the refusal must say why: %s", err);
            pulsar_spec_round_free(round);
        }
    }
    free_session(s);
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
        std::printf("tp_mirror_test: ok (worker refusals + acks, leader collect, dead-transport refusal)\n");
    else
        std::printf("tp_mirror_test: FAILED\n");
    return rc;
}

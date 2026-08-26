/*
 * Tensor-parallel transport and lockstep protocol (ledger L102, plan 102,
 * docs/tensor-parallel-port.md slice 3).  Faithful C++ lift of upstream
 * antirez/ds4 ds4_tp.c: two ranks run the same logical model, each holding one
 * contiguous half of the routed experts, exchanging f32 partial layer outputs
 * through a registered slab (RDMA SEND/RECV when verbs are available,
 * full-duplex TCP otherwise).
 *
 * Wire format is byte-identical to upstream (magic 0x44533454, protocol
 * version 7, the pulsar_tp_hello_fixed frame and the ds4_tp.h frame-type
 * numbers).  The control socket is a plain blocking TCP stream carrying
 * framed commands; gate traffic rides a dedicated full-duplex TCP socket (or
 * RDMA) so control frames never interleave with gate payloads.
 *
 * Deferred to the engine-wiring slice (see pulsar_tp.h): pulsar_tp_worker_run,
 * the CLI/engine validation helpers, and anything touching the ds4_gpu_*
 * Metal/MTL gate callbacks or the intra-host DS4_MAX_GPUS/NCCL axis.
 */

#include "tp/pulsar_tp.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <dlfcn.h>
#include <new>
#include <pthread.h>

#define PULSAR_TP_BATCH_MAGIC UINT32_C(0x44533442) /* "DS4B" */

/* Default gate timeout is generous: the first gate after a sync waits for the
 * peer's whole (possibly cold page cache) prefill. */
#define PULSAR_TP_DEFAULT_TIMEOUT_SEC 300

typedef struct {
    uint32_t magic;
    uint32_t type;
    uint32_t bytes;
} pulsar_tp_frame_header;

/* TCP gate frames carry a small header so a desynchronized pair fails loudly
 * instead of silently mixing partials. */
typedef struct {
    uint32_t magic;
    uint16_t layer;
    uint16_t gate;
    uint64_t seq;
} pulsar_tp_gate_header;

struct pulsar_tp {
    pulsar_tp_options opt;
    int rank;                   /* 0 leader, 1 worker */
    int control_fd;
    int data_fd;                /* TCP fallback, headers, and verify gates */
    bool rdma_active;
    uint32_t peer_ctx;
    uint32_t n_layer;
    uint32_t n_embd;
    uint64_t vec_bytes;
    uint32_t n_slots;
    /* Decode gate schedule (see pulsar_tp_identity). */
    uint32_t gate_slot_start;
    uint32_t gate_slot_step;
    uint32_t gates_per_token;
    uint8_t *slab;              /* registered slab base VA (caller-owned) */
    uint64_t slab_bytes;
    /* Slab layout, computed once by pulsar_tp_slab_layout_init (slice 1). */
    pulsar_tp_slab layout;
    uint64_t timeout_sec;
    std::atomic<bool> failed{false};
};

/* ------------------------------------------------------------------------
 * Small socket helpers (same conventions as upstream ds4_tp.c).
 * --------------------------------------------------------------------- */

static double tp_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void tp_set_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || !errlen) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static int tp_write_full(int fd, const void *buf, size_t len) {
    const char *p = static_cast<const char *>(buf);
    while (len) {
#ifdef MSG_NOSIGNAL
        ssize_t w = send(fd, p, len, MSG_NOSIGNAL);
#else
        ssize_t w = send(fd, p, len, 0);
#endif
        if (w < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (w == 0) return 0;
        p += w;
        len -= (size_t)w;
    }
    return 1;
}

static int tp_read_full(int fd, void *buf, size_t len) {
    char *p = static_cast<char *>(buf);
    while (len) {
        ssize_t r = read(fd, p, len);
        if (r < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (r == 0) return 0;
        p += r;
        len -= (size_t)r;
    }
    return 1;
}

static void tp_socket_tune(int fd) {
    int one = 1;
#ifdef SO_NOSIGPIPE
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    /* Gate exchanges are latency-critical 16KB messages; large socket
     * buffers only matter for the TCP fallback's pipelining. */
    int sz = 4 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
}

static int tp_listen(const char *host, int port, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    struct addrinfo hints = {}, *res = NULL;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    int rc = getaddrinfo(host && host[0] ? host : NULL, portbuf, &hints, &res);
    if (rc != 0) {
        tp_set_err(err, errlen, "tp listen resolve %s:%d: %s", host, port, gai_strerror(rc));
        return -1;
    }
    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && listen(fd, 2) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) tp_set_err(err, errlen, "tp listen %s:%d: %s", host, port, strerror(errno));
    return fd;
}

static int tp_dial(const char *host, int port, double timeout_sec, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    double deadline = tp_now_sec() + timeout_sec;
    int last_errno = 0;
    uint32_t attempts = 0;
    do {
        struct addrinfo hints = {}, *res = NULL;
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        int gai = getaddrinfo(host, portbuf, &hints, &res);
        if (gai == 0) {
            for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
                int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
                if (fd < 0) continue;
                if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
                    freeaddrinfo(res);
                    return fd;
                }
                last_errno = errno;
                close(fd);
            }
            freeaddrinfo(res);
        }
        /* Retrying is normal while the peer loads its model; still say why
         * every ~10s so a wrong address or a policy block is visible. */
        if (attempts++ % 50 == 0) {
            fprintf(stderr, "pulsar-tp: connecting to %s:%d ... (%s)\n", host, port,
                    gai != 0 ? gai_strerror(gai) :
                    last_errno ? strerror(last_errno) : "no address worked");
        }
        usleep(200 * 1000);
    } while (tp_now_sec() < deadline);
    tp_set_err(err, errlen, "tp connect %s:%d: %s", host, port,
               last_errno ? strerror(last_errno) : "unreachable");
    return -1;
}

static int tp_send_frame(int fd, uint32_t type, const void *payload, uint32_t bytes) {
    pulsar_tp_frame_header h = { PULSAR_TP_MAGIC, type, bytes };
    if (!tp_write_full(fd, &h, sizeof(h))) return 0;
    if (bytes && !tp_write_full(fd, payload, bytes)) return 0;
    return 1;
}

static int tp_read_frame_header(int fd, uint32_t *type, uint32_t *bytes) {
    pulsar_tp_frame_header h;
    if (!tp_read_full(fd, &h, sizeof(h))) return 0;
    if (h.magic != PULSAR_TP_MAGIC) return 0;
    *type = h.type;
    *bytes = h.bytes;
    return 1;
}

/* ------------------------------------------------------------------------
 * Slab layout (slice 1, unchanged) and registration.
 * --------------------------------------------------------------------- */

void pulsar_tp_slab_layout_init(uint32_t n_layer, uint32_t n_embd, pulsar_tp_slab *s) {
    const uint64_t vec = (uint64_t)n_embd * sizeof(float);
    const uint64_t slots = (uint64_t)n_layer * PULSAR_TP_GATES_PER_LAYER;
    s->out_off = 0;
    s->in_off = slots * vec;
    s->in_flags_off = s->in_off + slots * vec;
    s->token_off = s->in_flags_off + slots * 8;
    s->out_flags_off = s->token_off + 16;
    s->gpu_flags_off = s->out_flags_off + slots * 8;
    s->batch_out_off = s->gpu_flags_off + slots * 4;
    s->batch_in_off = s->batch_out_off + (uint64_t)n_layer * PULSAR_TP_BATCH_MAX_ROWS * vec;
    s->slab_bytes = s->batch_in_off + (uint64_t)n_layer * PULSAR_TP_BATCH_MAX_ROWS * vec;
}

uint64_t pulsar_tp_slab_bytes(uint32_t n_layer, uint32_t n_embd) {
    const uint64_t vec = (uint64_t)n_embd * sizeof(float);
    const uint64_t slots = (uint64_t)n_layer * PULSAR_TP_GATES_PER_LAYER;
    return slots * vec * 2 +    /* out + in vectors */
           slots * 8 * 2 +      /* in seq flags + out flag staging */
           16 +                 /* token slot */
           slots * 4 +          /* GPU-written gate-ready flags */
           (uint64_t)n_layer * PULSAR_TP_BATCH_MAX_ROWS * vec * 2; /* batch out+in */
}

static uint64_t tp_slot(uint32_t layer, uint32_t gate) {
    return (uint64_t)layer * PULSAR_TP_GATES_PER_LAYER + gate;
}

uint64_t pulsar_tp_slab_out_offset(const pulsar_tp_slab *s, uint32_t layer, uint32_t gate, uint64_t vec_bytes) {
    return s->out_off + tp_slot(layer, gate) * vec_bytes;
}

uint64_t pulsar_tp_slab_in_offset(const pulsar_tp_slab *s, uint32_t layer, uint32_t gate, uint64_t vec_bytes) {
    return s->in_off + tp_slot(layer, gate) * vec_bytes;
}

uint64_t pulsar_tp_slab_batch_out_offset(const pulsar_tp_slab *s, uint32_t layer, uint64_t vec_bytes) {
    return s->batch_out_off + (uint64_t)layer * PULSAR_TP_BATCH_MAX_ROWS * vec_bytes;
}

uint64_t pulsar_tp_slab_batch_in_offset(const pulsar_tp_slab *s, uint32_t layer, uint64_t vec_bytes) {
    return s->batch_in_off + (uint64_t)layer * PULSAR_TP_BATCH_MAX_ROWS * vec_bytes;
}

uint64_t pulsar_tp_slab_gpu_flags_offset(const pulsar_tp_slab *s) {
    return s->gpu_flags_off;
}

int pulsar_tp_identity_check(const pulsar_tp_identity *mine,
                             const pulsar_tp_identity *theirs,
                             char *err, size_t errlen) {
    if (theirs->gguf_bytes != mine->gguf_bytes ||
        theirs->model_id != mine->model_id ||
        theirs->n_layer != mine->n_layer ||
        theirs->n_embd != mine->n_embd ||
        theirs->n_vocab != mine->n_vocab ||
        theirs->quant_bits != mine->quant_bits ||
        theirs->gate_slot_start != mine->gate_slot_start ||
        theirs->gate_slot_step != mine->gate_slot_step ||
        theirs->gates_per_token != mine->gates_per_token) {
        if (errlen)
            snprintf(err, errlen,
                     "tp: model mismatch (peer gguf=%llu id=%u layers=%u embd=%u "
                     "vocab=%u qbits=%u)",
                     (unsigned long long)theirs->gguf_bytes, theirs->model_id,
                     theirs->n_layer, theirs->n_embd, theirs->n_vocab,
                     theirs->quant_bits);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------------
 * Bring-up.
 * --------------------------------------------------------------------- */

static pulsar_tp *tp_alloc(void) {
    void *raw = malloc(sizeof(pulsar_tp));
    if (!raw) return NULL;
    return new (raw) pulsar_tp{};
}

static void tp_destroy(pulsar_tp *tp) {
    if (!tp) return;
#ifdef PULSAR_TP_HAVE_VERBS
    tp_rdma_close(tp);
#endif
    if (tp->control_fd >= 0) close(tp->control_fd);
    if (tp->data_fd >= 0) close(tp->data_fd);
    tp->~pulsar_tp();
    free(tp);
}

static int tp_hello_exchange(pulsar_tp *tp, const pulsar_tp_identity *id, int rdma_ok,
                             char *err, size_t errlen) {
    pulsar_tp_hello_fixed mine = {
        .magic = PULSAR_TP_MAGIC,
        .version = PULSAR_TP_PROTOCOL_VERSION,
        .role = (uint32_t)tp->opt.role,
        .rdma_ok = (uint32_t)rdma_ok,
        .gguf_bytes = id->gguf_bytes,
        .model_id = id->model_id,
        .n_layer = id->n_layer,
        .n_embd = id->n_embd,
        .n_vocab = id->n_vocab,
        .quant_bits = id->quant_bits,
        .ctx_size = id->ctx_size,
        .gate_slot_start = id->gate_slot_start,
        .gate_slot_step = id->gate_slot_step,
        .gates_per_token = id->gates_per_token,
    };
    pulsar_tp_hello_fixed theirs;
    if (!tp_write_full(tp->control_fd, &mine, sizeof(mine)) ||
        !tp_read_full(tp->control_fd, &theirs, sizeof(theirs))) {
        tp_set_err(err, errlen, "tp hello exchange failed");
        return 0;
    }
    if (theirs.magic != PULSAR_TP_MAGIC) {
        tp_set_err(err, errlen, "tp hello: bad magic (mixed byte order or wrong peer?)");
        return 0;
    }
    if (theirs.version != PULSAR_TP_PROTOCOL_VERSION) {
        tp_set_err(err, errlen, "tp hello: protocol version %u != %u",
                   theirs.version, PULSAR_TP_PROTOCOL_VERSION);
        return 0;
    }
    if (theirs.role == mine.role) {
        tp_set_err(err, errlen, "tp hello: both sides claim role %u", mine.role);
        return 0;
    }
    if (theirs.gguf_bytes != mine.gguf_bytes || theirs.model_id != mine.model_id ||
        theirs.n_layer != mine.n_layer || theirs.n_embd != mine.n_embd ||
        theirs.n_vocab != mine.n_vocab || theirs.quant_bits != mine.quant_bits ||
        theirs.gate_slot_start != mine.gate_slot_start ||
        theirs.gate_slot_step != mine.gate_slot_step ||
        theirs.gates_per_token != mine.gates_per_token) {
        tp_set_err(err, errlen,
                   "tp hello: model mismatch (peer gguf=%llu id=%u layers=%u embd=%u "
                   "vocab=%u qbits=%u)",
                   (unsigned long long)theirs.gguf_bytes, theirs.model_id,
                   theirs.n_layer, theirs.n_embd, theirs.n_vocab, theirs.quant_bits);
        return 0;
    }
    tp->peer_ctx = theirs.ctx_size;
    tp->n_layer = id->n_layer;
    tp->n_embd = id->n_embd;
    tp->vec_bytes = (uint64_t)id->n_embd * sizeof(float);
    tp->n_slots = id->n_layer * PULSAR_TP_GATES_PER_LAYER;
    tp->gate_slot_start = id->gate_slot_start;
    tp->gate_slot_step = id->gate_slot_step;
    tp->gates_per_token = id->gates_per_token;
    pulsar_tp_slab_layout_init(tp->n_layer, tp->n_embd, &tp->layout);
    tp->slab_bytes = tp->layout.slab_bytes;
    /* Transport decision: RDMA only when both sides can.  slice-1 options
     * carry no --transport switch, so this keeps upstream's AUTO default;
     * when verbs are absent (this box) rdma_ok stays 0 and the pair rides
     * full-duplex TCP. */
    tp->rdma_active = rdma_ok && theirs.rdma_ok;
    return 1;
}

int pulsar_tp_create(pulsar_tp **out, const pulsar_tp_options *opt,
                     const pulsar_tp_identity *id, char *err, size_t errlen) {
    *out = NULL;
    pulsar_tp *tp = tp_alloc();
    if (!tp) {
        tp_set_err(err, errlen, "tp: out of memory");
        return 0;
    }
    tp->opt = *opt;
    tp->rank = opt->role == PULSAR_TP_ROLE_LEADER ? 0 : 1;
    tp->control_fd = -1;
    tp->data_fd = -1;
    tp->timeout_sec = PULSAR_TP_DEFAULT_TIMEOUT_SEC;
    const char *tmo = getenv("PULSAR_TP_TIMEOUT_SEC");
    if (tmo) tp->timeout_sec = (uint64_t)atoi(tmo);

    int rdma_ok = 0;
#ifdef PULSAR_TP_HAVE_VERBS
    if ((uint64_t)id->n_embd * sizeof(float) <= 2ull * PULSAR_TP_RDMA_MAX_MSG)
        rdma_ok = tp_rdma_probe(&tp->rdma.api);
#endif

    int listener = -1;
    if (tp->rank == 0) {
        listener = tp_listen(opt->peer, opt->port, err, errlen);
        if (listener < 0) goto fail;
        fprintf(stderr, "pulsar-tp: waiting for worker on %s:%d ...\n",
                opt->peer ? opt->peer : "0.0.0.0", opt->port);
        tp->control_fd = accept(listener, NULL, NULL);
        if (tp->control_fd < 0) {
            tp_set_err(err, errlen, "tp accept: %s", strerror(errno));
            goto fail;
        }
    } else {
        tp->control_fd = tp_dial(opt->peer, opt->port,
                                 (double)tp->timeout_sec, err, errlen);
        if (tp->control_fd < 0) goto fail;
    }
    tp_socket_tune(tp->control_fd);

    if (!tp_hello_exchange(tp, id, rdma_ok, err, errlen)) goto fail;

#ifdef PULSAR_TP_HAVE_VERBS
    if (tp->rdma_active) {
        if (!tp_rdma_open(tp, err, errlen)) goto fail;
    }
#endif
    {
        /* Second socket dedicated to gate traffic so control frames never
         * interleave with gate payloads.  Created under RDMA too for
         * headers, verify-block gates, and transport fallback. */
        if (tp->rank == 0) {
            tp->data_fd = accept(listener, NULL, NULL);
            if (tp->data_fd < 0) {
                tp_set_err(err, errlen, "tp data accept: %s", strerror(errno));
                goto fail;
            }
        } else {
            tp->data_fd = tp_dial(opt->peer, opt->port,
                                  (double)tp->timeout_sec, err, errlen);
            if (tp->data_fd < 0) goto fail;
        }
        tp_socket_tune(tp->data_fd);
    }
    if (listener >= 0) close(listener);
    fprintf(stderr, "pulsar-tp: %s connected, transport=%s\n",
            tp->rank == 0 ? "worker" : "leader",
            tp->rdma_active ? "rdma" : "tcp");
    *out = tp;
    return 1;
fail:
    if (listener >= 0) close(listener);
    tp_destroy(tp);
    return 0;
}

int pulsar_tp_attach_slab(pulsar_tp *tp, void *base, char *err, size_t errlen) {
    tp->slab = static_cast<uint8_t *>(base);
    memset(tp->slab + tp->layout.in_flags_off, 0, (uint64_t)tp->n_slots * 8);
    memset(tp->slab + tp->layout.token_off, 0, 16);
#ifdef PULSAR_TP_HAVE_VERBS
    if (tp->rdma_active) return tp_rdma_register_and_exchange(tp, err, errlen);
#endif
    (void)err; (void)errlen;
    return 1;
}

void pulsar_tp_free(pulsar_tp *tp) {
    tp_destroy(tp);
}

int pulsar_tp_rank(const pulsar_tp *tp) { return tp->rank; }
bool pulsar_tp_is_rdma(const pulsar_tp *tp) { return tp->rdma_active; }
uint32_t pulsar_tp_peer_ctx(const pulsar_tp *tp) { return tp->peer_ctx; }
bool pulsar_tp_failed(const pulsar_tp *tp) {
    return tp && tp->failed.load(std::memory_order_acquire);
}
void pulsar_tp_mark_failed(pulsar_tp *tp) {
    if (tp) tp->failed.store(true, std::memory_order_release);
}

/* ------------------------------------------------------------------------
 * Gate exchange.
 * --------------------------------------------------------------------- */

int pulsar_tp_gate_exchange(pulsar_tp *tp, uint32_t layer, uint32_t gate, uint64_t seq) {
#ifdef PULSAR_TP_HAVE_VERBS
    if (tp->rdma_active) return tp_rdma_gate_exchange(tp, layer, gate, seq);
#endif
    const uint64_t out_off =
        pulsar_tp_slab_out_offset(&tp->layout, layer, gate, tp->vec_bytes);
    const uint64_t in_off =
        pulsar_tp_slab_in_offset(&tp->layout, layer, gate, tp->vec_bytes);
    /* TCP: both sides write their partial then read the peer's.  16KB per
     * direction fits comfortably in the socket buffers, so the symmetric
     * write-then-read cannot deadlock.  Header and payload go out in one
     * writev so NODELAY does not split them into two segments. */
    pulsar_tp_gate_header h = { PULSAR_TP_MAGIC, (uint16_t)layer, (uint16_t)gate, seq };
    struct iovec iov[2] = {
        { &h, sizeof(h) },
        { tp->slab + out_off, tp->vec_bytes },
    };
    size_t want = sizeof(h) + tp->vec_bytes;
    ssize_t w = writev(tp->data_fd, iov, 2);
    if (w < 0 || (size_t)w != want) {
        /* Short writev: finish with the plain path. */
        if (w < 0) return 0;
        size_t done = (size_t)w;
        if (done < sizeof(h)) {
            if (!tp_write_full(tp->data_fd, reinterpret_cast<char *>(&h) + done,
                               sizeof(h) - done)) return 0;
            done = sizeof(h);
        }
        uint64_t payload_done = done - sizeof(h);
        if (!tp_write_full(tp->data_fd,
                           tp->slab + out_off + payload_done,
                           tp->vec_bytes - payload_done))
            return 0;
    }
    pulsar_tp_gate_header ph;
    if (!tp_read_full(tp->data_fd, &ph, sizeof(ph))) return 0;
    if (ph.magic != PULSAR_TP_MAGIC || ph.layer != layer || ph.gate != gate || ph.seq != seq) {
        fprintf(stderr,
                "pulsar-tp: gate desync: got l=%u g=%u seq=%llu, want l=%u g=%u seq=%llu\n",
                ph.layer, ph.gate, (unsigned long long)ph.seq,
                layer, gate, (unsigned long long)seq);
        return 0;
    }
    if (!tp_read_full(tp->data_fd, tp->slab + in_off, tp->vec_bytes))
        return 0;
    return 1;
}

/* Verify-block batch gate: one exchange per layer moving all block rows at
 * once.  The payload lives in the registered slab, so RDMA sends it directly;
 * TCP remains the symmetric write-then-read fallback. */
int pulsar_tp_batch_gate_exchange(pulsar_tp *tp, uint32_t layer, uint32_t rows,
                                  uint64_t seq) {
    if (tp->data_fd < 0 || rows == 0 || rows > PULSAR_TP_BATCH_MAX_ROWS) return 0;
    const uint64_t bytes = (uint64_t)rows * tp->vec_bytes;
    const uint64_t batch_out =
        pulsar_tp_slab_batch_out_offset(&tp->layout, layer, tp->vec_bytes);
    const uint64_t batch_in =
        pulsar_tp_slab_batch_in_offset(&tp->layout, layer, tp->vec_bytes);
    pulsar_tp_gate_header h = { PULSAR_TP_BATCH_MAGIC, (uint16_t)layer,
                                (uint16_t)rows, seq };
#ifdef PULSAR_TP_HAVE_VERBS
    if (tp->rdma_active && tp_rdma_big_gate_capable(tp)) {
        if (!tp_write_full(tp->data_fd, &h, sizeof(h))) return 0;
        pulsar_tp_gate_header ph;
        if (!tp_read_full(tp->data_fd, &ph, sizeof(ph))) return 0;
        if (ph.magic != PULSAR_TP_BATCH_MAGIC || ph.layer != layer ||
            ph.gate != rows || ph.seq != seq) {
            fprintf(stderr,
                    "pulsar-tp: batch gate desync: got l=%u rows=%u seq=%llu, "
                    "want l=%u rows=%u seq=%llu\n",
                    ph.layer, ph.gate, (unsigned long long)ph.seq,
                    layer, rows, (unsigned long long)seq);
            return 0;
        }
        if (!tp_rdma_drain_decode_window(tp)) return 0;
        return tp_rdma_big_gate_exchange(
                tp,
                tp->slab + batch_out,
                tp->slab + batch_in,
                bytes);
    }
#endif
    struct iovec iov[2] = {
        { &h, sizeof(h) },
        { tp->slab + batch_out, bytes },
    };
    size_t want = sizeof(h) + bytes;
    ssize_t w = writev(tp->data_fd, iov, 2);
    if (w < 0) return 0;
    if ((size_t)w != want) {
        size_t done = (size_t)w;
        if (done < sizeof(h)) {
            if (!tp_write_full(tp->data_fd, reinterpret_cast<char *>(&h) + done,
                               sizeof(h) - done))
                return 0;
            done = sizeof(h);
        }
        uint64_t payload_done = done - sizeof(h);
        if (!tp_write_full(tp->data_fd,
                           tp->slab + batch_out + payload_done,
                           bytes - payload_done))
            return 0;
    }
    pulsar_tp_gate_header ph;
    if (!tp_read_full(tp->data_fd, &ph, sizeof(ph))) return 0;
    if (ph.magic != PULSAR_TP_BATCH_MAGIC || ph.layer != layer ||
        ph.gate != rows || ph.seq != seq) {
        fprintf(stderr,
                "pulsar-tp: batch gate desync: got l=%u rows=%u seq=%llu, "
                "want l=%u rows=%u seq=%llu\n",
                ph.layer, ph.gate, (unsigned long long)ph.seq,
                layer, rows, (unsigned long long)seq);
        return 0;
    }
    return tp_read_full(tp->data_fd, tp->slab + batch_in, bytes);
}

/* Prefill batch gate: RDMA uses the pipelined registered-slab path above.
 * The fallback alternates 2MB TCP write/read rounds in the same order, so
 * neither side can fill its send buffer while the peer is also only writing
 * (the 4MB socket buffers absorb one round). */
#define PULSAR_TP_BIG_CHUNK (2ull * 1024ull * 1024ull)

int pulsar_tp_big_gate_exchange(pulsar_tp *tp, uint32_t layer, uint64_t seq,
                                const void *out, void *in, uint64_t bytes) {
    if (tp->data_fd < 0 || !out || !in || bytes == 0) return 0;
    pulsar_tp_gate_header h = { PULSAR_TP_BATCH_MAGIC, (uint16_t)layer, 0xB16u, seq };
    if (!tp_write_full(tp->data_fd, &h, sizeof(h))) return 0;
    pulsar_tp_gate_header ph;
    if (!tp_read_full(tp->data_fd, &ph, sizeof(ph))) return 0;
    if (ph.magic != PULSAR_TP_BATCH_MAGIC || ph.layer != layer ||
        ph.gate != 0xB16u || ph.seq != seq) {
        fprintf(stderr,
                "pulsar-tp: big gate desync: got l=%u tag=%x seq=%llu, want l=%u seq=%llu\n",
                ph.layer, ph.gate, (unsigned long long)ph.seq,
                layer, (unsigned long long)seq);
        return 0;
    }
#ifdef PULSAR_TP_HAVE_VERBS
    if (tp->rdma_active && tp_rdma_big_gate_capable(tp)) {
        if (!tp_rdma_drain_decode_window(tp)) return 0;
        return tp_rdma_big_gate_exchange(tp, out, in, bytes);
    }
#endif
    uint64_t off = 0;
    while (off < bytes) {
        const uint64_t n = bytes - off > PULSAR_TP_BIG_CHUNK ?
                           PULSAR_TP_BIG_CHUNK : bytes - off;
        if (!tp_write_full(tp->data_fd, static_cast<const char *>(out) + off, n)) return 0;
        if (!tp_read_full(tp->data_fd, static_cast<char *>(in) + off, n)) return 0;
        off += n;
    }
    if (getenv("PULSAR_TP_GLM_DEBUG")) {
        const float *o = static_cast<const float *>(out);
        const float *i = static_cast<const float *>(in);
        fprintf(stderr,
                "pulsar-tp: big gate l=%u seq=%llu out[0..3]=%g %g %g %g in[0..3]=%g %g %g %g\n",
                layer, (unsigned long long)seq,
                o[0], o[1], o[2], o[3], i[0], i[1], i[2], i[3]);
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * Lockstep control plane.
 * --------------------------------------------------------------------- */

typedef struct {
    uint64_t session_id;
    uint32_t count;
    uint32_t reserved;
} pulsar_tp_token_command_header;

typedef struct {
    uint64_t session_id;
    int32_t value;
    uint32_t reserved;
} pulsar_tp_value_command;

typedef struct {
    uint64_t session_id;
    uint64_t seq;
    int32_t token;
    uint32_t reserved;
} pulsar_tp_eval_command;

typedef struct {
    uint32_t count;
    uint32_t reserved;
} pulsar_tp_batch_command_header;

typedef struct {
    uint64_t prefill_session_id;
    uint32_t prompt_count;
    uint32_t item_count;
} pulsar_tp_mixed_command_header;

typedef struct {
    uint64_t session_id;
    int32_t status;
    uint32_t reserved;
} pulsar_tp_command_ack;

static int tp_send_token_command(pulsar_tp *tp, uint32_t type,
                                 uint64_t session_id, const int *tokens,
                                 uint32_t count) {
    const uint64_t bytes64 = sizeof(pulsar_tp_token_command_header) +
                             (uint64_t)count * sizeof(int32_t);
    if (!tp || (!tokens && count != 0) || bytes64 > UINT32_MAX) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = static_cast<uint8_t *>(malloc(bytes ? bytes : 1u));
    if (!payload) return 0;
    pulsar_tp_token_command_header h = { session_id, count, 0 };
    memcpy(payload, &h, sizeof(h));
    int32_t *wire_tokens = reinterpret_cast<int32_t *>(payload + sizeof(h));
    for (uint32_t i = 0; i < count; i++) wire_tokens[i] = (int32_t)tokens[i];
    const int ok = tp_send_frame(tp->control_fd, type, payload, bytes);
    free(payload);
    return ok;
}

int pulsar_tp_send_session_create(pulsar_tp *tp, uint64_t session_id, int ctx_size) {
    pulsar_tp_value_command msg = { session_id, (int32_t)ctx_size, 0 };
    return tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_SESSION_CREATE,
                         &msg, sizeof(msg));
}

int pulsar_tp_send_session_destroy(pulsar_tp *tp, uint64_t session_id) {
    return tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_SESSION_DESTROY,
                         &session_id, sizeof(session_id));
}

int pulsar_tp_send_sync(pulsar_tp *tp, uint64_t session_id,
                        const int *tokens, uint32_t n_tokens) {
    return tp_send_token_command(tp, PULSAR_TP_FRAME_SYNC, session_id,
                                 tokens, n_tokens);
}

int pulsar_tp_send_eval(pulsar_tp *tp, uint64_t session_id,
                        uint64_t seq, int token) {
    pulsar_tp_eval_command msg = { session_id, seq, (int32_t)token, 0 };
    return tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_EVAL, &msg, sizeof(msg));
}

int pulsar_tp_send_rewind(pulsar_tp *tp, uint64_t session_id, int pos) {
    pulsar_tp_value_command msg = { session_id, (int32_t)pos, 0 };
    return tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_REWIND,
                         &msg, sizeof(msg));
}

int pulsar_tp_send_invalidate(pulsar_tp *tp, uint64_t session_id) {
    return tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_INVALIDATE,
                         &session_id, sizeof(session_id));
}

int pulsar_tp_send_eval_batch(pulsar_tp *tp, const pulsar_tp_batch_item *items,
                              uint32_t count) {
    const uint64_t bytes64 = sizeof(pulsar_tp_batch_command_header) +
                             (uint64_t)count * sizeof(*items);
    if (!tp || !items || count == 0 || bytes64 > UINT32_MAX) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = static_cast<uint8_t *>(malloc(bytes));
    if (!payload) return 0;
    pulsar_tp_batch_command_header h = { count, 0 };
    memcpy(payload, &h, sizeof(h));
    memcpy(payload + sizeof(h), items, (size_t)count * sizeof(*items));
    const int ok = tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_EVAL_BATCH,
                                 payload, bytes);
    free(payload);
    return ok;
}

int pulsar_tp_send_mixed_batch(pulsar_tp *tp, uint64_t prefill_session_id,
                               const int *prompt, uint32_t prompt_count,
                               const pulsar_tp_batch_item *items,
                               uint32_t count) {
    const uint64_t prompt_bytes = (uint64_t)prompt_count * sizeof(int32_t);
    const uint64_t item_bytes = (uint64_t)count * sizeof(*items);
    const uint64_t bytes64 = sizeof(pulsar_tp_mixed_command_header) +
                             prompt_bytes + item_bytes;
    if (!tp || !prompt || prompt_count == 0 || !items || count == 0 ||
        bytes64 > UINT32_MAX) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = static_cast<uint8_t *>(malloc(bytes));
    if (!payload) return 0;
    pulsar_tp_mixed_command_header h = {
        prefill_session_id, prompt_count, count
    };
    memcpy(payload, &h, sizeof(h));
    int32_t *wire_tokens = reinterpret_cast<int32_t *>(payload + sizeof(h));
    for (uint32_t i = 0; i < prompt_count; i++) {
        wire_tokens[i] = (int32_t)prompt[i];
    }
    memcpy(payload + sizeof(h) + prompt_bytes, items, (size_t)item_bytes);
    const int ok = tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_MIXED_BATCH,
                                 payload, bytes);
    free(payload);
    return ok;
}

int pulsar_tp_send_command_ack(pulsar_tp *tp, uint64_t session_id, int status) {
    pulsar_tp_command_ack ack = { session_id, (int32_t)status, 0 };
    return tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_COMMAND_ACK,
                         &ack, sizeof(ack));
}

int pulsar_tp_wait_command_ack(pulsar_tp *tp, uint64_t session_id,
                               const char *operation, char *err, size_t errlen) {
    uint32_t type = 0, bytes = 0;
    pulsar_tp_command_ack ack;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != PULSAR_TP_FRAME_COMMAND_ACK || bytes != sizeof(ack) ||
        !tp_read_full(tp->control_fd, &ack, sizeof(ack))) {
        pulsar_tp_mark_failed(tp);
        tp_set_err(err, errlen, "tp: worker failed during %s",
                   operation ? operation : "command");
        return 0;
    }
    if (ack.session_id != session_id || ack.status != 0) {
        tp_set_err(err, errlen,
                   "tp: worker %s failed (session %llu, status %d)",
                   operation ? operation : "command",
                   (unsigned long long)ack.session_id, (int)ack.status);
        return 0;
    }
    return 1;
}

int pulsar_tp_send_stop(pulsar_tp *tp) {
    return tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_STOP, NULL, 0);
}

void pulsar_tp_command_free(pulsar_tp_command *command) {
    if (!command) return;
    free(command->tokens);
    free(command->items);
    memset(command, 0, sizeof(*command));
    command->type = PULSAR_TP_FRAME_ERROR;
}

static int tp_command_decode_tokens(pulsar_tp_command *command,
                                    const uint8_t *payload,
                                    uint32_t bytes,
                                    char *err, size_t errlen) {
    if (bytes < sizeof(pulsar_tp_token_command_header)) return 0;
    pulsar_tp_token_command_header h;
    memcpy(&h, payload, sizeof(h));
    const uint64_t want = sizeof(h) + (uint64_t)h.count * sizeof(int32_t);
    if (want != bytes) return 0;
    int *tokens = static_cast<int *>(malloc(h.count ? (size_t)h.count * sizeof(*tokens) : 1u));
    if (!tokens) {
        tp_set_err(err, errlen, "tp: command token allocation failed");
        return -1;
    }
    const int32_t *wire_tokens = reinterpret_cast<const int32_t *>(payload + sizeof(h));
    for (uint32_t i = 0; i < h.count; i++) tokens[i] = wire_tokens[i];
    command->session_id = h.session_id;
    command->tokens = tokens;
    command->n_tokens = h.count;
    return 1;
}

int pulsar_tp_recv_command(pulsar_tp *tp, pulsar_tp_command *command,
                           char *err, size_t errlen) {
    memset(command, 0, sizeof(*command));
    command->type = PULSAR_TP_FRAME_ERROR;
    uint32_t ftype = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &ftype, &bytes)) {
        tp_set_err(err, errlen, "tp: control channel closed");
        return 0;
    }
    uint8_t *payload = NULL;
    if (bytes != 0) {
        payload = static_cast<uint8_t *>(malloc(bytes));
        if (!payload || !tp_read_full(tp->control_fd, payload, bytes)) {
            free(payload);
            tp_set_err(err, errlen, "tp: truncated command frame");
            return 0;
        }
    }
    int ok = 1;
    switch (ftype) {
    case PULSAR_TP_FRAME_SYNC:
    case PULSAR_TP_FRAME_VERIFY:
        ok = tp_command_decode_tokens(command, payload, bytes, err, errlen);
        break;
    case PULSAR_TP_FRAME_SESSION_CREATE:
    case PULSAR_TP_FRAME_REWIND: {
        pulsar_tp_value_command msg;
        if (bytes != sizeof(msg)) { ok = 0; break; }
        memcpy(&msg, payload, sizeof(msg));
        command->session_id = msg.session_id;
        command->value = msg.value;
        break;
    }
    case PULSAR_TP_FRAME_SESSION_DESTROY:
    case PULSAR_TP_FRAME_INVALIDATE:
        if (bytes != sizeof(command->session_id)) { ok = 0; break; }
        memcpy(&command->session_id, payload, sizeof(command->session_id));
        break;
    case PULSAR_TP_FRAME_EVAL: {
        pulsar_tp_eval_command msg;
        if (bytes != sizeof(msg)) { ok = 0; break; }
        memcpy(&msg, payload, sizeof(msg));
        command->session_id = msg.session_id;
        command->seq = msg.seq;
        command->value = msg.token;
        break;
    }
    case PULSAR_TP_FRAME_EVAL_BATCH: {
        pulsar_tp_batch_command_header h;
        if (bytes < sizeof(h)) { ok = 0; break; }
        memcpy(&h, payload, sizeof(h));
        const uint64_t want = sizeof(h) +
                              (uint64_t)h.count * sizeof(pulsar_tp_batch_item);
        if (h.count == 0 || want != bytes) { ok = 0; break; }
        command->items = static_cast<pulsar_tp_batch_item *>(
            malloc((size_t)h.count * sizeof(*command->items)));
        if (!command->items) { ok = -1; break; }
        memcpy(command->items, payload + sizeof(h),
               (size_t)h.count * sizeof(*command->items));
        command->n_items = h.count;
        break;
    }
    case PULSAR_TP_FRAME_MIXED_BATCH: {
        pulsar_tp_mixed_command_header h;
        if (bytes < sizeof(h)) { ok = 0; break; }
        memcpy(&h, payload, sizeof(h));
        const uint64_t token_bytes =
            (uint64_t)h.prompt_count * sizeof(int32_t);
        const uint64_t item_bytes =
            (uint64_t)h.item_count * sizeof(pulsar_tp_batch_item);
        const uint64_t want = sizeof(h) + token_bytes + item_bytes;
        if (h.prompt_count == 0 || h.item_count == 0 || want != bytes) {
            ok = 0;
            break;
        }
        command->tokens = static_cast<int *>(malloc(
            (size_t)h.prompt_count * sizeof(*command->tokens)));
        command->items = static_cast<pulsar_tp_batch_item *>(malloc(
            (size_t)h.item_count * sizeof(*command->items)));
        if (!command->tokens || !command->items) { ok = -1; break; }
        const int32_t *wire_tokens =
            reinterpret_cast<const int32_t *>(payload + sizeof(h));
        for (uint32_t i = 0; i < h.prompt_count; i++) {
            command->tokens[i] = wire_tokens[i];
        }
        memcpy(command->items, payload + sizeof(h) + token_bytes,
               (size_t)item_bytes);
        command->session_id = h.prefill_session_id;
        command->n_tokens = h.prompt_count;
        command->n_items = h.item_count;
        break;
    }
    case PULSAR_TP_FRAME_STOP:
        if (bytes != 0) ok = 0;
        break;
    default:
        ok = 0;
        break;
    }
    free(payload);
    if (ok <= 0) {
        pulsar_tp_command_free(command);
        if (ok == 0) {
            tp_set_err(err, errlen, "tp: invalid command frame type %u (%u bytes)",
                       ftype, bytes);
        } else if (!err || !err[0]) {
            tp_set_err(err, errlen, "tp: command allocation failed");
        }
        return 0;
    }
    command->type = static_cast<pulsar_tp_frame_type>(ftype);
    return 1;
}

int pulsar_tp_send_logits_half(pulsar_tp *tp, const float *half, uint32_t count) {
    return tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_LOGITS,
                         half, count * sizeof(float));
}

int pulsar_tp_recv_logits_half(pulsar_tp *tp, float *half, uint32_t count) {
    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != PULSAR_TP_FRAME_LOGITS || bytes != count * sizeof(float)) {
        fprintf(stderr, "pulsar-tp: bad logits frame (type %u bytes %u)\n", type, bytes);
        return 0;
    }
    return tp_read_full(tp->control_fd, half, bytes);
}

int pulsar_tp_send_verify(pulsar_tp *tp, uint64_t session_id,
                          const int *drafts, uint32_t n) {
    return tp_send_token_command(tp, PULSAR_TP_FRAME_VERIFY, session_id,
                                 drafts, n);
}

int pulsar_tp_send_verify_commit(pulsar_tp *tp, int32_t full_accept, int32_t replay_n) {
    struct { int32_t full; int32_t replay; } msg = { full_accept, replay_n };
    return tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_VERIFY_COMMIT,
                         &msg, sizeof(msg));
}

int pulsar_tp_recv_verify_commit(pulsar_tp *tp, int32_t *full_accept, int32_t *replay_n) {
    uint32_t type = 0, bytes = 0;
    struct { int32_t full; int32_t replay; } msg;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != PULSAR_TP_FRAME_VERIFY_COMMIT || bytes != sizeof(msg) ||
        !tp_read_full(tp->control_fd, &msg, sizeof(msg))) {
        fprintf(stderr, "pulsar-tp: bad verify-commit frame (type %u bytes %u)\n",
                type, bytes);
        return 0;
    }
    *full_accept = msg.full;
    *replay_n = msg.replay;
    return 1;
}

int pulsar_tp_hash_check(pulsar_tp *tp, uint64_t seq, uint64_t hash,
                         char *err, size_t errlen) {
    struct { uint64_t seq; uint64_t hash; } mine = { seq, hash }, theirs;
    if (!tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_HASH, &mine, sizeof(mine))) {
        tp_set_err(err, errlen, "tp: hash send failed");
        return 0;
    }
    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != PULSAR_TP_FRAME_HASH || bytes != sizeof(theirs) ||
        !tp_read_full(tp->control_fd, &theirs, sizeof(theirs))) {
        tp_set_err(err, errlen, "tp: hash recv failed");
        return 0;
    }
    if (theirs.seq != seq || theirs.hash != hash) {
        tp_set_err(err, errlen,
                   "tp: LOCKSTEP DIVERGENCE at seq %llu: local %016llx peer %016llx",
                   (unsigned long long)seq,
                   (unsigned long long)hash, (unsigned long long)theirs.hash);
        return -1;
    }
    return 1;
}

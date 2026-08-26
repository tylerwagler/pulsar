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

/* ------------------------------------------------------------------------
 * RDMA path (verbs over the RDMA NIC).
 *
 * Upstream antirez/ds4 dlopens the verbs library through a function-pointer
 * table so builds and machines without the RDMA stack (or with it disabled)
 * fall back to TCP with no link-time cost.  This port goes one step further
 * and never includes <infiniband/verbs.h>: the few structs whose layout we
 * depend on (SGE / WR / CQ / MR / port) are reproduced below as a
 * self-contained ABI thunk, and EVERY entry point — including
 * ibv_post_send / ibv_post_recv / ibv_poll_cq, header inlines upstream — is
 * resolved with dlsym.  rdma_ok stays 0 when the library is absent OR no
 * HCA shows up in ibv_get_device_list, and the pair then rides full-duplex
 * TCP exactly as upstream.
 *
 * Device selection is explicit: the first device with an ACTIVE port is used
 * unless PULSAR_TP_RDMA_DEV names one (the pair is wired by two QSFP cables;
 * pinning a device per rank keeps a later two-link bench possible — no
 * multi-link merge is built yet).
 *
 * The thunk layouts follow current rdma-core <infiniband/verbs.h>; only the
 * leading ABI-stable fields are load-bearing today (two-sided SEND/RECV, no
 * inline data, no atomic/rdma-write WQEs).  The two-link RoCE bring-up slice
 * must field-validate the thunk against the installed rdma-core.
 * --------------------------------------------------------------------- */

/* Wire-exchanged RDMA identity block (upstream ds4_tp_rdma_info, same
 * layout, part of the version-7 wire contract). */
typedef struct {
    uint64_t slab_base;
    uint32_t rkey;
    uint32_t qpn;
    uint32_t psn;
    uint32_t mtu;
    uint16_t lid;
    uint8_t gid[16];
    uint8_t link_layer;
} pulsar_tp_rdma_info;

/* ---- verbs ABI thunk (see head-of-section note) ---- */
typedef void *tp_ibv_ctx;           /* struct ibv_context * */
typedef void *tp_ibv_pd;            /* struct ibv_pd * */
typedef void *tp_ibv_cq;            /* struct ibv_cq * */
typedef void *tp_ibv_qp;            /* struct ibv_qp * */
typedef void *tp_ibv_mr;            /* struct ibv_mr * */
typedef void *tp_ibv_device;        /* struct ibv_device * */
typedef void *tp_ibv_ah;            /* struct ibv_ah * */
typedef void *tp_ibv_srq;           /* struct ibv_srq * */
typedef void *tp_ibv_comp_channel;  /* struct ibv_comp_channel * */

/* enum ibv_mtu */
enum { TP_IBV_MTU_256 = 1, TP_IBV_MTU_512 = 2, TP_IBV_MTU_1024 = 3,
       TP_IBV_MTU_2048 = 4, TP_IBV_MTU_4096 = 5 };
/* enum ibv_qp_type (only UC is used; upstream validated UC-only QPs) */
enum { TP_IBV_QPT_UC = 3 };
/* enum ibv_qp_state */
enum { TP_IBV_QPS_RESET = 0, TP_IBV_QPS_INIT = 1, TP_IBV_QPS_RTR = 2,
       TP_IBV_QPS_RTS = 3, TP_IBV_QPS_SQD = 4, TP_IBV_QPS_SQE = 5,
       TP_IBV_QPS_ERR = 6 };
/* enum ibv_port_state */
enum { TP_IBV_PORT_NOP = 0, TP_IBV_PORT_DOWN = 1, TP_IBV_PORT_ACTIVE = 2,
       TP_IBV_PORT_INIT = 3, TP_IBV_PORT_ARMED = 4 };
/* enum ibv_wr_opcode (only SEND is used) */
enum { TP_IBV_WR_SEND = 0 };
/* enum ibv_wc_status (only success/error are distinguished) */
enum { TP_IBV_WC_SUCCESS = 0, TP_IBV_WC_GENERAL_ERR = 19 };
/* enum ibv_wc_opcode: recv completions share the bit upstream tests */
enum { TP_IBV_WC_SEND = 0, TP_IBV_WC_RECV = 1 << 7 };
/* enum ibv_access_flags */
enum { TP_IBV_ACCESS_LOCAL_WRITE = 1, TP_IBV_ACCESS_REMOTE_WRITE = 1 << 1,
       TP_IBV_ACCESS_REMOTE_READ = 1 << 2 };
/* enum ibv_send_flags */
enum { TP_IBV_SEND_SIGNALED = 2 };
/* enum ibv_qp_attr_mask */
enum { TP_IBV_QP_ACCESS_FLAGS = 1 << 3, TP_IBV_QP_PKEY_INDEX = 1 << 4,
       TP_IBV_QP_PORT = 1 << 5, TP_IBV_QP_AV = 1 << 7,
       TP_IBV_QP_PATH_MTU = 1 << 8, TP_IBV_QP_RQ_PSN = 1 << 12,
       TP_IBV_QP_SQ_PSN = 1 << 16, TP_IBV_QP_DEST_QPN = 1 << 20,
       TP_IBV_QP_STATE = 1 };

union tp_ibv_gid {
    uint8_t raw[16];
    struct { uint64_t subnet_prefix; uint64_t interface_id; } global;
};

struct tp_ibv_sge { uint64_t addr; uint32_t length; uint32_t lkey; };

struct tp_ibv_send_wr {
    uint64_t wr_id;
    struct tp_ibv_send_wr *next;
    struct tp_ibv_sge *sg_list;
    int num_sge;
    int opcode;      /* enum ibv_wr_opcode */
    int send_flags;  /* enum ibv_send_flags */
    union {
        struct { uint64_t remote_addr; uint32_t rkey; } rdma;
        struct { uint64_t remote_addr; uint64_t compare_add; uint64_t swap;
                 uint32_t rkey; } atomic;
        struct { tp_ibv_ah *ah; uint32_t remote_qpn; uint32_t remote_qkey; } ud;
        struct { uint32_t remote_qpn; uint32_t remote_qkey; uint32_t pkey_index; } xrc;
    } wr;
    union { uint32_t imm_data; uint32_t invalidate_rkey; } ex;
};

struct tp_ibv_recv_wr {
    uint64_t wr_id;
    struct tp_ibv_recv_wr *next;
    struct tp_ibv_sge *sg_list;
    int num_sge;
};

struct tp_ibv_wc {
    uint64_t wr_id;
    int status;      /* enum ibv_wc_status */
    int opcode;      /* enum ibv_wc_opcode */
    uint32_t vendor_err;
    uint32_t byte_len;
    uint32_t imm_data;
    uint32_t qp_num;
    uint32_t src_qp;
    int wc_flags;
    uint16_t pkey_index;
    uint16_t slid;
    uint8_t sl;
    uint8_t dlid_path_bits;
    uint8_t port_num;
    uint8_t pad;
};

struct tp_ibv_qp_cap { uint32_t max_send_wr; uint32_t max_recv_wr;
                       uint32_t max_send_sge; uint32_t max_recv_sge;
                       uint32_t max_inline_data; };

struct tp_ibv_qp_init_attr {
    void *qp_context;
    tp_ibv_cq send_cq;
    tp_ibv_cq recv_cq;
    tp_ibv_srq srq;
    struct tp_ibv_qp_cap cap;
    int qp_type;     /* enum ibv_qp_type */
    int sq_sig_all;
};

struct tp_ibv_global_route {
    union tp_ibv_gid dgid;
    uint32_t flow_label;
    uint8_t sgid_index;
    uint8_t hop_limit;
    uint8_t traffic_class;
    uint8_t pad;
};

struct tp_ibv_ah_attr {
    struct tp_ibv_global_route grh;
    uint16_t dlid;
    uint8_t sl;
    uint8_t src_path_bits;
    uint8_t static_rate;
    uint8_t is_global;
    uint8_t port_num;
    uint8_t pad2;
};

struct tp_ibv_qp_attr {
    int qp_state;       /* enum ibv_qp_state */
    int cur_qp_state;   /* enum ibv_qp_state */
    int path_mtu;       /* enum ibv_mtu */
    int path_mig_state; /* enum ibv_mig_state */
    uint32_t qkey;
    uint32_t rq_psn;
    uint32_t sq_psn;
    uint32_t dest_qp_num;
    uint32_t qp_access_flags;
    struct tp_ibv_qp_cap cap;
    struct tp_ibv_ah_attr ah_attr;
    struct tp_ibv_ah_attr alt_ah_attr;
    uint16_t pkey_index;
    uint16_t alt_pkey_index;
    uint8_t en_sqd_async_notify;
    uint8_t sq_draining;
    uint8_t max_rd_atomic;
    uint8_t max_dest_rd_atomic;
    uint8_t min_rnr_timer;
    uint8_t port_num;
    uint8_t timeout;
    uint8_t retry_cnt;
    uint8_t rnr_retry;
    uint8_t alt_port_num;
    uint8_t alt_timeout;
    uint32_t rate_limit;
};

/* struct ibv_port_attr: stable-prefix fields only; the version-varying tail
 * is absorbed by padding.  Load-bearing offsets — state@0, active_mtu@8,
 * gid_tbl_len@12, lid@34, link_layer@39 — are constant across rdma-core. */
struct tp_ibv_port_attr {
    int state;              /* enum ibv_port_state      @0  */
    int max_mtu;            /* enum ibv_mtu             @4  */
    int active_mtu;         /* enum ibv_mtu             @8  */
    uint32_t gid_tbl_len;   /*                           @12 */
    uint32_t port_cap_flags_stub;
    uint32_t max_msg_sz;
    uint32_t bad_pkey_cntr;
    uint32_t qkey_viol_cntr;
    uint16_t pkey_tbl_len;  /*                           @32 */
    uint16_t lid;           /*                           @34 */
    uint16_t sm_lid;
    uint8_t lmc;
    uint8_t link_layer;     /* enum ibv_link_layer      @39 */
    uint8_t max_vl_num;
    uint8_t sm_sl;
    uint8_t subnet_timeout;
    uint8_t init_type_reply;
    uint8_t active_speed;
    uint8_t phys_state;
    uint8_t pad[32];        /* version-varying tail */
};

/* Object prefixes read back from library-owned memory (only the leading,
 * stable members are touched). */
struct tp_ibv_qp_layout {   /* qp_num lives at the canonical offset */
    tp_ibv_ctx context;
    void *qp_context;
    tp_ibv_pd pd;
    tp_ibv_cq send_cq;
    tp_ibv_cq recv_cq;
    tp_ibv_srq srq;
    uint32_t handle;
    uint32_t qp_num;        /* @52 */
    uint32_t pad[8];
};
struct tp_ibv_mr_layout {
    tp_ibv_ctx context;
    tp_ibv_pd pd;
    void *addr;
    size_t length;
    uint32_t handle;
    uint32_t lkey;          /* @28 */
    uint32_t rkey;          /* @32 */
    uint32_t pad;
};

/* librdma/libibverbs is loaded at runtime so builds and machines without the
 * RDMA stack (or with it disabled) fall back to TCP with no link-time cost.
 * ibv_post_send()/ibv_post_recv()/ibv_poll_cq() are header inlines upstream;
 * here they are resolved with dlsym like every other entry point.  The
 * tp_ibv_* typedefs already denote pointers, so the table uses them bare. */
typedef struct {
    void *handle;
    tp_ibv_device *(*get_device_list)(int *);
    void (*free_device_list)(tp_ibv_device);
    const char *(*get_device_name)(tp_ibv_device);
    tp_ibv_ctx (*open_device)(tp_ibv_device);
    int (*close_device)(tp_ibv_ctx);
    int (*query_port)(tp_ibv_ctx, uint8_t, struct tp_ibv_port_attr *);
    int (*query_gid)(tp_ibv_ctx, uint8_t, int, union tp_ibv_gid *);
    tp_ibv_pd (*alloc_pd)(tp_ibv_ctx);
    int (*dealloc_pd)(tp_ibv_pd);
    tp_ibv_mr (*reg_mr)(tp_ibv_pd, void *, size_t, int);
    int (*dereg_mr)(tp_ibv_mr);
    tp_ibv_cq (*create_cq)(tp_ibv_ctx, int, void *, tp_ibv_comp_channel, int);
    int (*destroy_cq)(tp_ibv_cq);
    tp_ibv_qp (*create_qp)(tp_ibv_pd, struct tp_ibv_qp_init_attr *);
    int (*destroy_qp)(tp_ibv_qp);
    int (*modify_qp)(tp_ibv_qp, struct tp_ibv_qp_attr *, int);
    int (*post_send)(tp_ibv_qp, struct tp_ibv_send_wr *,
                     struct tp_ibv_send_wr **);
    int (*post_recv)(tp_ibv_qp, struct tp_ibv_recv_wr *,
                     struct tp_ibv_recv_wr **);
    int (*poll_cq)(tp_ibv_cq, int, struct tp_ibv_wc *);
} pulsar_tp_verbs_api;

/* Per-connection RDMA state.  Data-plane notes (upstream's validated design):
 * UC is two-sided SEND/RECV only; messages above 16KB are chunked; UC
 * delivery is in-order and the gate sequence is globally deterministic, so
 * the recv completion for gate seq s IS the arrival signal. */
#define PULSAR_TP_RDMA_MAX_MSG 16384
#define PULSAR_TP_RDMA_RECV_WINDOW 16
#define PULSAR_TP_RDMA_BULK_SLOTS 64
#define PULSAR_TP_RDMA_BULK_WR_TAG (UINT64_C(1) << 63)

typedef struct {
    pulsar_tp_verbs_api api;
    tp_ibv_ctx ctx;
    tp_ibv_pd pd;
    tp_ibv_cq cq;
    tp_ibv_qp qp;
    tp_ibv_mr mr;
    struct tp_ibv_port_attr port;
    union tp_ibv_gid gid;
    int gid_index;
    uint32_t max_inline;
    pulsar_tp_rdma_info peer;
    uint32_t send_outstanding;  /* signaled sends not yet reaped */
    uint64_t recv_done;         /* highest gate seq whose recv completed */
    uint64_t last_gate_seq;     /* last real decode receive consumed */
    bool recv_window_active;    /* decode recvs are queued ahead */
    pthread_mutex_t post_lock;
} pulsar_tp_rdma;

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
    pulsar_tp_rdma rdma;    /* RDMA state (loaded lazily at create/attach) */
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
 * RDMA bring-up and data plane.
 * --------------------------------------------------------------------- */

/* UC queue pairs do not report a dead remote reliably.  The control socket
 * does, so sample it while polling an RDMA completion and abort before the
 * GPU command-buffer watchdog fires. */
static int tp_peer_closed(const pulsar_tp *tp) {
    char byte;
    const ssize_t n = recv(tp->control_fd, &byte, 1,
                           MSG_PEEK | MSG_DONTWAIT);
    if (n == 0) return 1;
    if (n > 0) return 0;
    return errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR;
}

static int tp_rdma_load_api(pulsar_tp_verbs_api *api) {
    if (api->handle) return 1;
    /* Linux rdma-core (upstream dlopens librdma.dylib; the two-link RoCE
     * bring-up slice validates against the installed libibverbs). */
    const char *libs[] = { "libibverbs.so.1", "libibverbs.so", NULL };
    void *h = NULL;
    for (int i = 0; libs[i] && !h; i++)
        h = dlopen(libs[i], RTLD_NOW | RTLD_LOCAL);
    if (!h) return 0;
#define TP_SYM(field, name) \
    do { \
        api->field = reinterpret_cast<__typeof__(api->field)>(dlsym(h, name)); \
        if (!api->field) { dlclose(h); return 0; } \
    } while (0)
    TP_SYM(get_device_list, "ibv_get_device_list");
    TP_SYM(free_device_list, "ibv_free_device_list");
    TP_SYM(get_device_name, "ibv_get_device_name");
    TP_SYM(open_device, "ibv_open_device");
    TP_SYM(close_device, "ibv_close_device");
    TP_SYM(query_port, "ibv_query_port");
    TP_SYM(query_gid, "ibv_query_gid");
    TP_SYM(alloc_pd, "ibv_alloc_pd");
    TP_SYM(dealloc_pd, "ibv_dealloc_pd");
    TP_SYM(reg_mr, "ibv_reg_mr");
    TP_SYM(dereg_mr, "ibv_dereg_mr");
    TP_SYM(create_cq, "ibv_create_cq");
    TP_SYM(destroy_cq, "ibv_destroy_cq");
    TP_SYM(create_qp, "ibv_create_qp");
    TP_SYM(destroy_qp, "ibv_destroy_qp");
    TP_SYM(modify_qp, "ibv_modify_qp");
    TP_SYM(post_send, "ibv_post_send");
    TP_SYM(post_recv, "ibv_post_recv");
    TP_SYM(poll_cq, "ibv_poll_cq");
#undef TP_SYM
    api->handle = h;
    return 1;
}

/* Probe only: does this machine expose a usable verbs stack right now?
 * rdma_ok = dlopen succeeded AND at least one HCA is listed.  On a box whose
 * libibverbs is present but has no RDMA device this must return 0 so the
 * pair degrades to full-duplex TCP.  PULSAR_TP_RDMA_DEV (an HCA name from
 * /sys/class/infiniband) pins the device for the two-cable pair. */
static int tp_rdma_probe(pulsar_tp_verbs_api *api) {
    if (!tp_rdma_load_api(api)) return 0;
    const char *want_name = getenv("PULSAR_TP_RDMA_DEV");
    int num = 0;
    tp_ibv_device *devs = api->get_device_list(&num);
    if (!devs) return 0;
    if (want_name) {
        int found = 0;
        for (int i = 0; i < num && !found; i++)
            found = strcmp(want_name, api->get_device_name(devs[i])) == 0;
        if (!found) {
            api->free_device_list(devs);
            return 0;
        }
    }
    api->free_device_list(devs);
    return num > 0;
}

static int tp_rdma_open(pulsar_tp *tp, char *err, size_t errlen) {
    pulsar_tp_rdma *r = &tp->rdma;
    int num = 0;
    tp_ibv_device *devs = r->api.get_device_list(&num);
    if (!devs || num == 0) {
        tp_set_err(err, errlen, "tp rdma: no verbs devices");
        if (devs) r->api.free_device_list(devs);
        return 0;
    }
    /* One verbs device per link.  The pair is wired by two QSFP cables, so
     * PULSAR_TP_RDMA_DEV pins the device per rank (later two-link bench);
     * the default auto-picks the first ACTIVE port. */
    const char *want_name = getenv("PULSAR_TP_RDMA_DEV");
    char states[256] = "";
    int chose;
    for (int i = 0; i < num && !r->ctx; i++) {
        const char *name = r->api.get_device_name(devs[i]);
        if (want_name && strcmp(want_name, name) != 0) continue;
        tp_ibv_ctx ctx = r->api.open_device(devs[i]);
        if (!ctx) continue;
        struct tp_ibv_port_attr pa;
        (void)memset(&pa, 0, sizeof(pa));
        chose = r->api.query_port(ctx, 1, &pa) == 0 &&
                (pa.state == TP_IBV_PORT_ACTIVE || want_name);
        if (chose) {
            r->ctx = ctx;
            r->port = pa;
            fprintf(stderr, "pulsar-tp: rdma device %s (port state %d)\n",
                    name, (int)pa.state);
            break;
        }
        size_t off = strlen(states);
        snprintf(states + off, sizeof(states) - off, "%s%s=%d",
                 off ? ", " : "", name, (int)pa.state);
        r->api.close_device(ctx);
    }
    r->api.free_device_list(devs);
    if (!r->ctx) {
        tp_set_err(err, errlen,
                   "tp rdma: no device with an active port (%s); is the peer "
                   "up and the link enabled on both machines?", states);
        return 0;
    }
    /* The Thunderbolt driver only connects through the IPv4-mapped GID
     * (::ffff:a.b.c.d); Linux RoCE GID selection is a bring-up-slice
     * refinement.  Otherwise scan the port's GIDs for that pattern. */
    r->gid_index = -1;
    for (uint32_t j = 0; j < r->port.gid_tbl_len; j++) {
        union tp_ibv_gid tmp;
        const int i = (int)j;
        if (r->api.query_gid(r->ctx, 1, i, &tmp) != 0) continue;
        uint64_t hi;
        uint16_t mid, v4tag;
        memcpy(&hi, &tmp.raw[0], 8);
        memcpy(&mid, &tmp.raw[8], 2);
        memcpy(&v4tag, &tmp.raw[10], 2);
        if (hi == 0 && mid == 0 && v4tag == 0xffff) {
            r->gid = tmp;
            r->gid_index = i;
            break;
        }
    }
    if (r->gid_index < 0) {
        tp_set_err(err, errlen,
                   "tp rdma: no IPv4-mapped GID on the active port");
        return 0;
    }
    r->pd = r->api.alloc_pd(r->ctx);
    if (!r->pd) {
        tp_set_err(err, errlen, "tp rdma: alloc_pd failed");
        return 0;
    }
    r->cq = r->api.create_cq(r->ctx, 512, NULL, NULL, 0);
    if (!r->cq) {
        tp_set_err(err, errlen, "tp rdma: create_cq failed");
        return 0;
    }
    struct tp_ibv_qp_init_attr qia;
    (void)memset(&qia, 0, sizeof(qia));
    qia.send_cq = r->cq;
    qia.recv_cq = r->cq;
    qia.qp_type = TP_IBV_QPT_UC;
    qia.cap.max_send_wr = 256;
    qia.cap.max_recv_wr = 64;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    qia.cap.max_inline_data = 0;
    r->qp = r->api.create_qp(r->pd, &qia);
    if (!r->qp) {
        tp_set_err(err, errlen, "tp rdma: create_qp(UC): %s", strerror(errno));
        return 0;
    }
    r->max_inline = qia.cap.max_inline_data;
    pthread_mutex_init(&r->post_lock, NULL);
    return 1;
}

/* Slab slot a given gate seq lands in.  DS4 fires every slot in order
 * (identity mapping); a projected schedule from the hello skips dense layers
 * and the ATTN slots. */
static uint32_t tp_gate_slot(const pulsar_tp *tp, uint64_t seq) {
    if (tp->gates_per_token == 0)
        return (uint32_t)((seq - 1) % tp->n_slots);
    return tp->gate_slot_start +
           (uint32_t)((seq - 1) % tp->gates_per_token) * tp->gate_slot_step;
}

static int tp_rdma_post_gate_recv(pulsar_tp *tp, uint64_t seq);

static int tp_rdma_register_and_exchange(pulsar_tp *tp, char *err,
                                         size_t errlen) {
    pulsar_tp_rdma *r = &tp->rdma;
    r->mr = r->api.reg_mr(r->pd, tp->slab, (size_t)tp->slab_bytes,
                          TP_IBV_ACCESS_LOCAL_WRITE | TP_IBV_ACCESS_REMOTE_READ |
                          TP_IBV_ACCESS_REMOTE_WRITE);
    if (!r->mr) {
        tp_set_err(err, errlen, "tp rdma: reg_mr(%llu bytes): %s",
                   (unsigned long long)tp->slab_bytes, strerror(errno));
        return 0;
    }
    struct tp_ibv_mr_layout *mr = reinterpret_cast<struct tp_ibv_mr_layout *>(r->mr);
    struct tp_ibv_qp_layout *qp = reinterpret_cast<struct tp_ibv_qp_layout *>(r->qp);
    pulsar_tp_rdma_info mine;
    (void)memset(&mine, 0, sizeof(mine));
    mine.slab_base = (uint64_t)(uintptr_t)tp->slab;
    mine.rkey = mr->rkey;
    mine.qpn = qp->qp_num;
    mine.psn = (uint32_t)(getpid() ^ (uintptr_t)tp) & 0xffffff;
    mine.mtu = (uint32_t)r->port.active_mtu;
    mine.lid = r->port.lid;
    memcpy(mine.gid, r->gid.raw, 16);
    mine.link_layer = r->port.link_layer;
    if (!tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_RDMA_INFO,
                       &mine, sizeof(mine))) {
        tp_set_err(err, errlen, "tp rdma: info send failed");
        return 0;
    }
    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != PULSAR_TP_FRAME_RDMA_INFO || bytes != sizeof(r->peer) ||
        !tp_read_full(tp->control_fd, &r->peer, sizeof(r->peer))) {
        tp_set_err(err, errlen, "tp rdma: info recv failed");
        return 0;
    }

    /* INIT -> RTR -> RTS with GRH addressing through the exchanged GID. */
    struct tp_ibv_qp_attr a;
    (void)memset(&a, 0, sizeof(a));
    a.qp_state = TP_IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = 1;
    a.qp_access_flags = TP_IBV_ACCESS_LOCAL_WRITE | TP_IBV_ACCESS_REMOTE_READ |
                        TP_IBV_ACCESS_REMOTE_WRITE;
    if (r->api.modify_qp(r->qp, &a,
            TP_IBV_QP_STATE | TP_IBV_QP_PKEY_INDEX | TP_IBV_QP_PORT |
            TP_IBV_QP_ACCESS_FLAGS) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify INIT: %s", strerror(errno));
        return 0;
    }
    (void)memset(&a, 0, sizeof(a));
    a.qp_state = TP_IBV_QPS_RTR;
    a.path_mtu = TP_IBV_MTU_1024;
    a.dest_qp_num = r->peer.qpn;
    a.rq_psn = r->peer.psn;
    a.ah_attr.dlid = (uint16_t)r->peer.lid;
    a.ah_attr.port_num = 1;
    a.ah_attr.is_global = 1;
    memcpy(a.ah_attr.grh.dgid.raw, r->peer.gid, 16);
    a.ah_attr.grh.sgid_index = (uint8_t)r->gid_index;
    a.ah_attr.grh.hop_limit = 1;
    if (r->api.modify_qp(r->qp, &a,
            TP_IBV_QP_STATE | TP_IBV_QP_AV | TP_IBV_QP_PATH_MTU |
            TP_IBV_QP_DEST_QPN | TP_IBV_QP_RQ_PSN) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify RTR: %s", strerror(errno));
        return 0;
    }
    (void)memset(&a, 0, sizeof(a));
    a.qp_state = TP_IBV_QPS_RTS;
    a.sq_psn = mine.psn;
    if (r->api.modify_qp(r->qp, &a, TP_IBV_QP_STATE | TP_IBV_QP_SQ_PSN) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify RTS: %s", strerror(errno));
        return 0;
    }
    if (tp->vec_bytes > 2ull * PULSAR_TP_RDMA_MAX_MSG) {
        tp_set_err(err, errlen,
                   "tp rdma: gate vector %llu bytes exceeds twice the driver's "
                   "%u message limit",
                   (unsigned long long)tp->vec_bytes, PULSAR_TP_RDMA_MAX_MSG);
        return 0;
    }
    if (tp->vec_bytes > PULSAR_TP_RDMA_MAX_MSG)
        fprintf(stderr,
                "pulsar-tp: rdma gate vectors ride as 2 chunked messages "
                "(%llu bytes > %u limit)\n",
                (unsigned long long)tp->vec_bytes, PULSAR_TP_RDMA_MAX_MSG);
    /* Leave the receive queue empty for an initial bulk prefill; the first
     * decode gate arms the normal lookahead window. */
    if (!tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_RDMA_READY, NULL, 0)) {
        tp_set_err(err, errlen, "tp rdma: ready send failed");
        return 0;
    }
    uint32_t rtype = 0, rbytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &rtype, &rbytes) ||
        rtype != PULSAR_TP_FRAME_RDMA_READY || rbytes != 0) {
        tp_set_err(err, errlen, "tp rdma: ready barrier failed");
        return 0;
    }
    return 1;
}

static const char *tp_wc_status_str(int status) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "wc status %d", status);
    return buf;
}

/* Reap completions: send CQEs free send-queue slots, recv CQEs advance the
 * arrival watermark (UC is in-order, so gate seq recv completions arrive
 * monotonically).  Returns 0 on any completion error. */
static int tp_rdma_drain_cq(pulsar_tp *tp) {
    pulsar_tp_rdma *r = &tp->rdma;
    struct tp_ibv_wc wc[16];
    int n = r->api.poll_cq(r->cq, 16, wc);
    if (n < 0) return 0;
    for (int i = 0; i < n; i++) {
        if (wc[i].status != TP_IBV_WC_SUCCESS) {
            fprintf(stderr, "pulsar-tp: rdma completion error: %s (wr_id %llu)\n",
                    tp_wc_status_str(wc[i].status),
                    (unsigned long long)wc[i].wr_id);
            return 0;
        }
        if (wc[i].opcode & TP_IBV_WC_RECV) {
            if (wc[i].wr_id > r->recv_done) r->recv_done = wc[i].wr_id;
        } else if (r->send_outstanding > 0) {
            r->send_outstanding--;
        }
    }
    return 1;
}

/* Arm the receive for gate seq: UC delivery order pairs the peer's seq'th
 * send with our seq'th posted recv, landing it in the in-slot the combine
 * kernel reads. */
static int tp_rdma_post_gate_recv(pulsar_tp *tp, uint64_t seq) {
    pulsar_tp_rdma *r = &tp->rdma;
    const uint32_t slot = tp_gate_slot(tp, seq);
    const uintptr_t base =
        (uintptr_t)(tp->slab + tp->layout.in_off +
                    (uint64_t)slot * tp->vec_bytes);
    struct tp_ibv_mr_layout *mr = reinterpret_cast<struct tp_ibv_mr_layout *>(r->mr);
    /* Vectors above the driver's 16KB message cap ride as two chunks landing
     * contiguously in the slot.  Both sides post/send strictly in seq order,
     * so the k'th send always matches the k'th recv; only the FINAL chunk
     * carries the seq as wr_id, so the arrival watermark advances when the
     * slot is whole. */
    uint64_t off = 0;
    while (off < tp->vec_bytes) {
        const uint64_t len = tp->vec_bytes - off > PULSAR_TP_RDMA_MAX_MSG ?
            PULSAR_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
        const int last = off + len == tp->vec_bytes;
        struct tp_ibv_sge sge;
        struct tp_ibv_recv_wr wr;
        struct tp_ibv_recv_wr *bad = NULL;
        (void)memset(&wr, 0, sizeof(wr));
        sge.addr = base + off;
        sge.length = (uint32_t)len;
        sge.lkey = mr->lkey;
        wr.wr_id = last ? seq : 0;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        if (r->api.post_recv(r->qp, &wr, &bad) != 0) {
            fprintf(stderr, "pulsar-tp: rdma post_recv(seq %llu off %llu): %s\n",
                    (unsigned long long)seq, (unsigned long long)off,
                    strerror(errno));
            return 0;
        }
        off += len;
    }
    return 1;
}

/* One decode gate: ensure the receive window is armed, send our partial,
 * wait for the peer's receive completion, and advance the window. */
static int tp_rdma_gate_exchange(pulsar_tp *tp, uint32_t layer, uint32_t gate,
                                 uint64_t seq) {
    pulsar_tp_rdma *r = &tp->rdma;
    const uint32_t slot = layer * PULSAR_TP_GATES_PER_LAYER + gate;
    if (getenv("PULSAR_TP_GATE_TRACE")) {
        fprintf(stderr, "pulsar-tp: gate trace l=%u g=%u seq=%llu want_slot=%u\n",
                layer, gate, (unsigned long long)seq, tp_gate_slot(tp, seq));
    }
    if (slot != tp_gate_slot(tp, seq)) {
        fprintf(stderr, "pulsar-tp: gate order broke: layer %u gate %u vs seq %llu\n",
                layer, gate, (unsigned long long)seq);
        return 0;
    }
    struct tp_ibv_mr_layout *mr = reinterpret_cast<struct tp_ibv_mr_layout *>(r->mr);
    const uintptr_t send_base =
        (uintptr_t)(tp->slab + tp->layout.out_off +
                    (uint64_t)slot * tp->vec_bytes);
    pthread_mutex_lock(&r->post_lock);
    int ok = 1;
    if (!r->recv_window_active) {
        for (uint64_t s = seq; ok && s < seq + PULSAR_TP_RDMA_RECV_WINDOW; s++)
            ok = tp_rdma_post_gate_recv(tp, s);
        if (ok) r->recv_window_active = true;
    }
    for (uint64_t off = 0; ok && off < tp->vec_bytes; ) {
        const uint64_t len = tp->vec_bytes - off > PULSAR_TP_RDMA_MAX_MSG ?
            PULSAR_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
        struct tp_ibv_sge sge;
        struct tp_ibv_send_wr wr;
        struct tp_ibv_send_wr *bad = NULL;
        (void)memset(&wr, 0, sizeof(wr));
        sge.addr = send_base + off;
        sge.length = (uint32_t)len;
        sge.lkey = mr->lkey;
        wr.wr_id = seq;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = TP_IBV_WR_SEND;
        wr.send_flags = TP_IBV_SEND_SIGNALED;
        ok = r->api.post_send(r->qp, &wr, &bad) == 0;
        if (!ok) {
            fprintf(stderr, "pulsar-tp: rdma post_send: %s\n", strerror(errno));
        } else {
            r->send_outstanding++;
        }
        off += len;
    }

    double deadline = 0.0;
    uint32_t peer_poll = 0;
    while (ok && r->recv_done < seq) {
        ok = tp_rdma_drain_cq(tp);
        if (ok && (peer_poll++ & 0x3fffu) == 0 && tp_peer_closed(tp)) {
            fprintf(stderr, "pulsar-tp: peer disconnected during rdma gate\n");
            ok = 0;
        }
        if (deadline == 0.0) deadline = tp_now_sec() + (double)tp->timeout_sec;
        else if (tp_now_sec() > deadline) {
            fprintf(stderr, "pulsar-tp: timeout waiting gate seq %llu (recv_done %llu)\n",
                    (unsigned long long)seq, (unsigned long long)r->recv_done);
            ok = 0;
        }
    }
    if (ok) ok = tp_rdma_post_gate_recv(tp, seq + PULSAR_TP_RDMA_RECV_WINDOW);
    if (ok) r->last_gate_seq = seq;
    pthread_mutex_unlock(&r->post_lock);
    return ok;
}

static int tp_rdma_big_gate_capable(const pulsar_tp *tp) {
    const uint64_t stage_bytes =
        (uint64_t)PULSAR_TP_RDMA_BULK_SLOTS * PULSAR_TP_RDMA_MAX_MSG;
    const uint64_t batch_region_bytes =
        (uint64_t)tp->n_layer * PULSAR_TP_BATCH_MAX_ROWS * tp->vec_bytes;
    return tp->rdma.qp && tp->rdma.mr && batch_region_bytes >= stage_bytes;
}

/* Decode keeps a lookahead window of receives on the latency QP.  Before a
 * later prompt can reuse that QP for bulk rows, consume those receives with
 * dummy sends on both ranks.  The TCP big-gate header exchange is the barrier
 * that guarantees both sides have reached this transition. */
static int tp_rdma_drain_decode_window(pulsar_tp *tp) {
    pulsar_tp_rdma *r = &tp->rdma;
    if (!r->recv_window_active) return 1;

    const uint32_t chunks_per_gate =
        (uint32_t)((tp->vec_bytes + PULSAR_TP_RDMA_MAX_MSG - 1u) /
                   PULSAR_TP_RDMA_MAX_MSG);
    const uint32_t nwr = PULSAR_TP_RDMA_RECV_WINDOW * chunks_per_gate;
    struct tp_ibv_sge sge[PULSAR_TP_RDMA_RECV_WINDOW * 2u];
    struct tp_ibv_send_wr wr[PULSAR_TP_RDMA_RECV_WINDOW * 2u];
    (void)memset(wr, 0, sizeof(wr));
    uint8_t *scratch = tp->slab + tp->layout.batch_out_off;
    struct tp_ibv_mr_layout *mr = reinterpret_cast<struct tp_ibv_mr_layout *>(r->mr);
    uint32_t wi = 0;
    for (uint32_t gate = 0; gate < PULSAR_TP_RDMA_RECV_WINDOW; gate++) {
        for (uint64_t off = 0; off < tp->vec_bytes; ) {
            const uint64_t len = tp->vec_bytes - off > PULSAR_TP_RDMA_MAX_MSG ?
                PULSAR_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
            sge[wi].addr = (uintptr_t)(scratch + off);
            sge[wi].length = (uint32_t)len;
            sge[wi].lkey = mr->lkey;
            wr[wi].wr_id = PULSAR_TP_RDMA_BULK_WR_TAG | ((uint64_t)wi + 1u);
            wr[wi].sg_list = &sge[wi];
            wr[wi].num_sge = 1;
            wr[wi].opcode = TP_IBV_WR_SEND;
            wr[wi].send_flags = wi + 1u == nwr ? TP_IBV_SEND_SIGNALED : 0;
            if (wi > 0) wr[wi - 1u].next = &wr[wi];
            wi++;
            off += len;
        }
    }

    pthread_mutex_lock(&r->post_lock);
    struct tp_ibv_send_wr *bad = NULL;
    if (r->api.post_send(r->qp, wr, &bad) != 0) {
        fprintf(stderr, "pulsar-tp: rdma receive-window drain post failed: %s\n",
                strerror(errno));
        pthread_mutex_unlock(&r->post_lock);
        return 0;
    }

    uint32_t recv_done = 0;
    int send_done = 0;
    const double deadline = tp_now_sec() + (double)tp->timeout_sec;
    uint32_t peer_poll = 0;
    while (recv_done < nwr || !send_done) {
        struct tp_ibv_wc wc[PULSAR_TP_RDMA_RECV_WINDOW * 2u + 1u];
        int n = r->api.poll_cq(r->cq,
                               (int)(PULSAR_TP_RDMA_RECV_WINDOW * 2u + 1u), wc);
        if (n < 0) {
            pthread_mutex_unlock(&r->post_lock);
            return 0;
        }
        for (int i = 0; i < n; i++) {
            if (wc[i].status != TP_IBV_WC_SUCCESS) {
                fprintf(stderr, "pulsar-tp: rdma receive-window drain: %s\n",
                        tp_wc_status_str(wc[i].status));
                pthread_mutex_unlock(&r->post_lock);
                return 0;
            }
            if (wc[i].opcode & TP_IBV_WC_RECV) {
                recv_done++;
            } else if (wc[i].wr_id & PULSAR_TP_RDMA_BULK_WR_TAG) {
                send_done = 1;
            } else if (r->send_outstanding > 0) {
                r->send_outstanding--;
            }
        }
        if ((peer_poll++ & 0x3fffu) == 0 && tp_peer_closed(tp)) {
            fprintf(stderr,
                    "pulsar-tp: peer disconnected while draining rdma receives\n");
            pthread_mutex_unlock(&r->post_lock);
            return 0;
        }
        if (tp_now_sec() > deadline) {
            fprintf(stderr,
                    "pulsar-tp: timeout draining rdma receive window (%u/%u)\n",
                    recv_done, nwr);
            pthread_mutex_unlock(&r->post_lock);
            return 0;
        }
    }
    r->recv_done = r->last_gate_seq;
    r->recv_window_active = false;
    pthread_mutex_unlock(&r->post_lock);
    return 1;
}

/* Large prefill row swaps share the latency QP.  No future decode receives
 * are queued, so each round can post its 1 MiB receive window before sending
 * the matching 16 KiB messages.  Verify scratch provides already-registered
 * staging memory and is idle during normal prefill. */
static int tp_rdma_big_gate_exchange(pulsar_tp *tp, const void *out, void *in,
                                     uint64_t bytes) {
    pulsar_tp_rdma *r = &tp->rdma;
    if (!tp_rdma_big_gate_capable(tp) || r->recv_window_active) return 0;

    /* Payloads already inside the registered slab (verify batches) can ride
     * directly; ordinary prefill tensors use the idle verify regions as
     * registered staging. */
    const uintptr_t slab_lo = (uintptr_t)tp->slab;
    const uintptr_t slab_hi = slab_lo + tp->slab_bytes;
    const uintptr_t out_lo = (uintptr_t)out;
    const uintptr_t in_lo = (uintptr_t)in;
    const bool direct =
        out_lo >= slab_lo && out_lo <= slab_hi && bytes <= slab_hi - out_lo &&
        in_lo >= slab_lo && in_lo <= slab_hi && bytes <= slab_hi - in_lo;
    uint8_t *stage_send = tp->slab + tp->layout.batch_out_off;
    uint8_t *stage_recv = tp->slab + tp->layout.batch_in_off;
    struct tp_ibv_mr_layout *mr = reinterpret_cast<struct tp_ibv_mr_layout *>(r->mr);
    uint64_t off = 0;
    while (off < bytes) {
        const uint64_t remaining = bytes - off;
        uint32_t chunks = (uint32_t)((remaining + PULSAR_TP_RDMA_MAX_MSG - 1u) /
                                     PULSAR_TP_RDMA_MAX_MSG);
        if (chunks > PULSAR_TP_RDMA_BULK_SLOTS)
            chunks = PULSAR_TP_RDMA_BULK_SLOTS;

        uint32_t lens[PULSAR_TP_RDMA_BULK_SLOTS];
        uint64_t chunk_off[PULSAR_TP_RDMA_BULK_SLOTS];
        uint64_t round_bytes = 0;
        for (uint32_t i = 0; i < chunks; i++) {
            const uint64_t left = remaining - round_bytes;
            lens[i] = (uint32_t)(left > PULSAR_TP_RDMA_MAX_MSG ?
                                 PULSAR_TP_RDMA_MAX_MSG : left);
            chunk_off[i] = direct ? round_bytes :
                (uint64_t)i * PULSAR_TP_RDMA_MAX_MSG;
            if (!direct) {
                memcpy(stage_send + chunk_off[i],
                       static_cast<const uint8_t *>(out) + off + round_bytes,
                       lens[i]);
            }
            round_bytes += lens[i];
        }

        struct tp_ibv_sge recv_sge[PULSAR_TP_RDMA_BULK_SLOTS];
        struct tp_ibv_recv_wr recv_wr[PULSAR_TP_RDMA_BULK_SLOTS];
        (void)memset(recv_wr, 0, sizeof(recv_wr));
        for (uint32_t i = 0; i < chunks; i++) {
            recv_sge[i].addr = direct ? in_lo + off + chunk_off[i] :
                                 (uintptr_t)(stage_recv + chunk_off[i]);
            recv_sge[i].length = lens[i];
            recv_sge[i].lkey = mr->lkey;
            recv_wr[i].wr_id = PULSAR_TP_RDMA_BULK_WR_TAG | ((uint64_t)i + 1u);
            recv_wr[i].sg_list = &recv_sge[i];
            recv_wr[i].num_sge = 1;
            recv_wr[i].next = i + 1u < chunks ? &recv_wr[i + 1u] : NULL;
        }
        struct tp_ibv_recv_wr *bad_recv = NULL;
        if (r->api.post_recv(r->qp, recv_wr, &bad_recv) != 0) {
            fprintf(stderr, "pulsar-tp: bulk rdma post_recv: %s\n",
                    strerror(errno));
            return 0;
        }
        std::atomic_thread_fence(std::memory_order_release);
        struct tp_ibv_sge send_sge[PULSAR_TP_RDMA_BULK_SLOTS];
        struct tp_ibv_send_wr send_wr[PULSAR_TP_RDMA_BULK_SLOTS];
        (void)memset(send_wr, 0, sizeof(send_wr));
        for (uint32_t i = 0; i < chunks; i++) {
            send_sge[i].addr = direct ? out_lo + off + chunk_off[i] :
                                 (uintptr_t)(stage_send + chunk_off[i]);
            send_sge[i].length = lens[i];
            send_sge[i].lkey = mr->lkey;
            send_wr[i].wr_id = PULSAR_TP_RDMA_BULK_WR_TAG | ((uint64_t)i + 1u);
            send_wr[i].sg_list = &send_sge[i];
            send_wr[i].num_sge = 1;
            send_wr[i].opcode = TP_IBV_WR_SEND;
            send_wr[i].send_flags = i + 1u == chunks ? TP_IBV_SEND_SIGNALED : 0;
            send_wr[i].next = i + 1u < chunks ? &send_wr[i + 1u] : NULL;
        }
        struct tp_ibv_send_wr *bad_send = NULL;
        if (r->api.post_send(r->qp, send_wr, &bad_send) != 0) {
            fprintf(stderr, "pulsar-tp: bulk rdma post_send: %s\n",
                    strerror(errno));
            return 0;
        }

        uint32_t recv_done = 0;
        int send_done = 0;
        const double deadline = tp_now_sec() + (double)tp->timeout_sec;
        uint32_t peer_poll = 0;
        while (recv_done < chunks || !send_done) {
            struct tp_ibv_wc wc[PULSAR_TP_RDMA_BULK_SLOTS + 1u];
            int n = r->api.poll_cq(r->cq,
                                   (int)(PULSAR_TP_RDMA_BULK_SLOTS + 1u), wc);
            if (n < 0) return 0;
            for (int i = 0; i < n; i++) {
                if (wc[i].status != TP_IBV_WC_SUCCESS) {
                    fprintf(stderr,
                            "pulsar-tp: bulk rdma completion error: %s\n",
                            tp_wc_status_str(wc[i].status));
                    return 0;
                }
                if ((wc[i].wr_id & PULSAR_TP_RDMA_BULK_WR_TAG) == 0) {
                    /* A final latency-QP send completion can remain queued
                     * when a later prompt starts a bulk gate. */
                    if (wc[i].opcode & TP_IBV_WC_RECV) {
                        if (wc[i].wr_id > r->recv_done)
                            r->recv_done = wc[i].wr_id;
                    } else if (r->send_outstanding > 0) {
                        r->send_outstanding--;
                    }
                    continue;
                }
                if (wc[i].opcode & TP_IBV_WC_RECV) recv_done++;
                else send_done = 1;
            }
            if ((peer_poll++ & 0x3fffu) == 0 && tp_peer_closed(tp)) {
                fprintf(stderr,
                        "pulsar-tp: peer disconnected during bulk rdma gate\n");
                return 0;
            }
            if (tp_now_sec() > deadline) {
                fprintf(stderr,
                        "pulsar-tp: timeout waiting for bulk rdma round "
                        "(%u/%u recvs, send=%d)\n",
                        recv_done, chunks, send_done);
                return 0;
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (!direct) {
            round_bytes = 0;
            for (uint32_t i = 0; i < chunks; i++) {
                memcpy(static_cast<uint8_t *>(in) + off + round_bytes,
                       stage_recv + chunk_off[i], lens[i]);
                round_bytes += lens[i];
            }
        }
        off += round_bytes;
    }
    return 1;
}

static void tp_rdma_close(pulsar_tp *tp) {
    pulsar_tp_rdma *r = &tp->rdma;
    if (r->qp) r->api.destroy_qp(r->qp);
    if (r->mr) r->api.dereg_mr(r->mr);
    if (r->cq) r->api.destroy_cq(r->cq);
    if (r->pd) r->api.dealloc_pd(r->pd);
    if (r->ctx) r->api.close_device(r->ctx);
    r->qp = NULL; r->mr = NULL; r->cq = NULL; r->pd = NULL; r->ctx = NULL;
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
    tp_rdma_close(tp);
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
    if ((uint64_t)id->n_embd * sizeof(float) <= 2ull * PULSAR_TP_RDMA_MAX_MSG)
        rdma_ok = tp_rdma_probe(&tp->rdma.api);

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

    if (tp->rdma_active) {
        if (!tp_rdma_open(tp, err, errlen)) goto fail;
    }
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
    if (tp->rdma_active) return tp_rdma_register_and_exchange(tp, err, errlen);
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
    if (tp->rdma_active) return tp_rdma_gate_exchange(tp, layer, gate, seq);
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
    if (tp->rdma_active && tp_rdma_big_gate_capable(tp)) {
        if (!tp_rdma_drain_decode_window(tp)) return 0;
        return tp_rdma_big_gate_exchange(tp, out, in, bytes);
    }
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

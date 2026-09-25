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
#include <poll.h>
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
#include <dirent.h>
#include <signal.h>
#include <algorithm>
#include <string>
#include <vector>
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
    /* v14: the BULK lane -- this rank's registered bulk buffer and the second
     * (bulk) QP; all zero when this rank attached no bulk buffer. */
    uint64_t bulk_base;
    uint32_t bulk_rkey;
    uint32_t bulk_qpn;
    uint32_t bulk_psn;
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

/* ------------------------------------------------------------------------
 * Struct selection gate: <infiniband/verbs.h> presence at build time.
 *
 * The pair reached transport=rdma but the first gate exchange never delivered.
 * Root cause (confirmed on the box's GID table): the scan picked the FIRST
 * IPv4-mapped GID, which on the pair is the RoCEv1 entry — the RoCEv2 gid
 * carries IDENTICAL bytes at a later index, only distinguishable by gid_type.
 * RC rejected that AV with EINVAL at RTR; UC swallowed it into a dead QP.
 * Secondary finding: the mirrored tp_ibv_port_attr put link_layer at offset
 * 39 where the real struct has it at 46.  Both argue for the same discipline —
 * when libibverbs-dev is installed (dev + both Sparks) use the REAL rdma-core
 * structs for every value handed to the verbs stack, and query the per-entry
 * gid_type via _ibv_query_gid_ex.  The TP_IBV_* names below alias to the real
 * enumerations; the self-contained ABI thunk is the no-header fallback.
 * --------------------------------------------------------------------- */
#if defined(__has_include)
#  if __has_include(<infiniband/verbs.h>)
#    include <infiniband/verbs.h>
#    define PULSAR_TP_VERBS_HDR 1
#  endif
#endif

#ifdef PULSAR_TP_VERBS_HDR
/* rdma-core enumerations (values are ABI-fixed; aliasing also keeps the int ->
 * enum assignments into the real *_attr fields type-correct under C++). */
#define TP_IBV_MTU_256          IBV_MTU_256
#define TP_IBV_MTU_512          IBV_MTU_512
#define TP_IBV_MTU_1024         IBV_MTU_1024
#define TP_IBV_MTU_2048         IBV_MTU_2048
#define TP_IBV_MTU_4096         IBV_MTU_4096
#define TP_IBV_QPT_UC           IBV_QPT_UC
#define TP_IBV_QPT_RC           IBV_QPT_RC
#define TP_IBV_QPS_RESET        IBV_QPS_RESET
#define TP_IBV_QPS_INIT         IBV_QPS_INIT
#define TP_IBV_QPS_RTR          IBV_QPS_RTR
#define TP_IBV_QPS_RTS          IBV_QPS_RTS
#define TP_IBV_QPS_SQD          IBV_QPS_SQD
#define TP_IBV_QPS_SQE          IBV_QPS_SQE
#define TP_IBV_QPS_ERR          IBV_QPS_ERR
#define TP_IBV_PORT_NOP         IBV_PORT_NOP
#define TP_IBV_PORT_DOWN        IBV_PORT_DOWN
#define TP_IBV_PORT_ACTIVE      IBV_PORT_ACTIVE
#define TP_IBV_PORT_INIT        IBV_PORT_INIT
#define TP_IBV_PORT_ARMED       IBV_PORT_ARMED
#define TP_IBV_WR_SEND          IBV_WR_SEND
#define TP_IBV_WR_RDMA_WRITE    IBV_WR_RDMA_WRITE
#define TP_IBV_WR_RDMA_WRITE_WITH_IMM IBV_WR_RDMA_WRITE_WITH_IMM
#define TP_IBV_WC_SUCCESS       IBV_WC_SUCCESS
#define TP_IBV_WC_SEND          IBV_WC_SEND
#define TP_IBV_WC_RECV          IBV_WC_RECV
#define TP_IBV_WC_GENERAL_ERR   IBV_WC_GENERAL_ERR
#define TP_IBV_ACCESS_LOCAL_WRITE   IBV_ACCESS_LOCAL_WRITE
#define TP_IBV_ACCESS_REMOTE_WRITE  IBV_ACCESS_REMOTE_WRITE
#define TP_IBV_ACCESS_REMOTE_READ   IBV_ACCESS_REMOTE_READ
#define TP_IBV_SEND_SIGNALED    IBV_SEND_SIGNALED
#define TP_IBV_QP_ACCESS_FLAGS  IBV_QP_ACCESS_FLAGS
#define TP_IBV_QP_PKEY_INDEX    IBV_QP_PKEY_INDEX
#define TP_IBV_QP_PORT          IBV_QP_PORT
#define TP_IBV_QP_AV            IBV_QP_AV
#define TP_IBV_QP_PATH_MTU      IBV_QP_PATH_MTU
#define TP_IBV_QP_RQ_PSN        IBV_QP_RQ_PSN
#define TP_IBV_QP_SQ_PSN        IBV_QP_SQ_PSN
#define TP_IBV_QP_DEST_QPN      IBV_QP_DEST_QPN
#define TP_IBV_QP_STATE         IBV_QP_STATE
#define TP_IBV_QP_TIMEOUT       IBV_QP_TIMEOUT
#define TP_IBV_QP_RETRY_CNT     IBV_QP_RETRY_CNT
#define TP_IBV_QP_RNR_RETRY     IBV_QP_RNR_RETRY
#define TP_IBV_QP_MAX_QP_RD_ATOMIC   IBV_QP_MAX_QP_RD_ATOMIC
#define TP_IBV_QP_MAX_DEST_RD_ATOMIC IBV_QP_MAX_DEST_RD_ATOMIC
#define TP_IBV_QP_MIN_RNR_TIMER      IBV_QP_MIN_RNR_TIMER
#define TP_IBV_GID_TYPE_IB        IBV_GID_TYPE_IB
#define TP_IBV_GID_TYPE_ROCE_V1   IBV_GID_TYPE_ROCE_V1
#define TP_IBV_GID_TYPE_ROCE_V2   IBV_GID_TYPE_ROCE_V2
/* struct/union aliases to the real layouts handed to the verbs stack. */
#define tp_ibv_gid_entry          ibv_gid_entry
#define tp_ibv_gid              ibv_gid
#define tp_ibv_sge              ibv_sge
#define tp_ibv_send_wr          ibv_send_wr
#define tp_ibv_recv_wr          ibv_recv_wr
#define tp_ibv_wc               ibv_wc
#define tp_ibv_qp_cap           ibv_qp_cap
#define tp_ibv_qp_init_attr     ibv_qp_init_attr
#define tp_ibv_global_route     ibv_global_route
#define tp_ibv_ah_attr          ibv_ah_attr
#define tp_ibv_qp_attr          ibv_qp_attr
#define tp_ibv_port_attr        ibv_port_attr
#else
/* -------------------- no-header fallback ABI thunk -------------------- */
/* enum ibv_mtu */
enum { TP_IBV_MTU_256 = 1, TP_IBV_MTU_512 = 2, TP_IBV_MTU_1024 = 3,
       TP_IBV_MTU_2048 = 4, TP_IBV_MTU_4096 = 5 };
/* enum ibv_qp_type (default UC, upstream-validated; RC is the A/B bring-up
 * knob — PULSAR_TP_RDMA_RC — since RC rejects a wrong RTR where UC only
 * silently dead-connects) */
enum { TP_IBV_QPT_RC = 2, TP_IBV_QPT_UC = 3 };
/* enum ibv_qp_state */
enum { TP_IBV_QPS_RESET = 0, TP_IBV_QPS_INIT = 1, TP_IBV_QPS_RTR = 2,
       TP_IBV_QPS_RTS = 3, TP_IBV_QPS_SQD = 4, TP_IBV_QPS_SQE = 5,
       TP_IBV_QPS_ERR = 6 };
/* enum ibv_port_state */
enum { TP_IBV_PORT_NOP = 0, TP_IBV_PORT_DOWN = 1, TP_IBV_PORT_ACTIVE = 2,
       TP_IBV_PORT_INIT = 3, TP_IBV_PORT_ARMED = 4 };
/* enum ibv_wr_opcode: RDMA_WRITE, RDMA_WRITE_WITH_IMM, SEND -- the order in
 * <infiniband/verbs.h> (SEND is 2; this thunk used to say 0, i.e. RDMA_WRITE) */
enum { TP_IBV_WR_RDMA_WRITE = 0, TP_IBV_WR_RDMA_WRITE_WITH_IMM = 1, TP_IBV_WR_SEND = 2 };
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
       TP_IBV_QP_PATH_MTU = 1 << 8, TP_IBV_QP_TIMEOUT = 1 << 9,
       TP_IBV_QP_RETRY_CNT = 1 << 10, TP_IBV_QP_RNR_RETRY = 1 << 11,
       TP_IBV_QP_RQ_PSN = 1 << 12, TP_IBV_QP_MAX_QP_RD_ATOMIC = 1 << 13,
       TP_IBV_QP_MIN_RNR_TIMER = 1 << 15,
       TP_IBV_QP_SQ_PSN = 1 << 16, TP_IBV_QP_MAX_DEST_RD_ATOMIC = 1 << 17,
       TP_IBV_QP_DEST_QPN = 1 << 20, TP_IBV_QP_STATE = 1 };
/* enum ibv_gid_type */
enum { TP_IBV_GID_TYPE_IB = 0, TP_IBV_GID_TYPE_ROCE_V1 = 1,
       TP_IBV_GID_TYPE_ROCE_V2 = 2 };

union tp_ibv_gid {
    uint8_t raw[16];
    struct { uint64_t subnet_prefix; uint64_t interface_id; } global;
};

/* struct ibv_gid_entry (gid_type at @24): returned by _ibv_query_gid_ex.
 * RoCEv1 and RoCEv2 GIDs carry the SAME 16 bytes — only this per-entry type
 * tells them apart; the pair's table has ::ffff:192.168.9.x as v1 (idx 2)
 * AND v2 (idx 3), and the v1 index is an invalid AV under RoCEv2. */
struct tp_ibv_gid_entry {
    union tp_ibv_gid gid;         /* @0  */
    uint32_t gid_index;           /* @16 */
    uint32_t port_num;            /* @20 */
    uint32_t gid_type;            /* @24 */
    uint32_t ndev_ifindex;        /* @28 */
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
    union { uint32_t imm_data; uint32_t invalidate_rkey; };   /* anonymous, as in <infiniband/verbs.h> */
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

#endif /* PULSAR_TP_VERBS_HDR: real structs above, ABI-thunk fallback */

#ifdef PULSAR_TP_VERBS_HDR
#define TP_QPN(q)   (((struct ibv_qp *)(q))->qp_num)
#define TP_LKEY(m)  (((struct ibv_mr *)(m))->lkey)
#define TP_RKEY(m)  (((struct ibv_mr *)(m))->rkey)
#else
#define TP_QPN(q)   (((struct tp_ibv_qp_layout *)(q))->qp_num)
#define TP_LKEY(m)  (((struct tp_ibv_mr_layout *)(m))->lkey)
#define TP_RKEY(m)  (((struct tp_ibv_mr_layout *)(m))->rkey)
#endif

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
    /* _ibv_query_gid_ex: optional (may be NULL on older rdma-core).  Gives the
     * per-entry gid_type so selection can pick RoCEv2 over a byte-identical
     * RoCEv1 GID. */
    int (*query_gid_ex)(tp_ibv_ctx, uint32_t, uint32_t,
                        struct tp_ibv_gid_entry *, uint32_t, size_t);
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
/* The bulk lane (v14): one RDMA write per piece, the last WITH_IMM. */
#define PULSAR_TP_BULK_PIECE (UINT64_C(4) << 20)
#define PULSAR_TP_BULK_IMM_WINDOW 16u
#define PULSAR_TP_BULK_WR_TAG (UINT64_C(1) << 62)
/* TCP fallback write/read round: small enough that two simultaneous rounds can
 * never fill both send buffers under ANY kernel socket-buffer clamp (the pair
 * hosts clamp SO_SNDBUF to net.core.wmem_max ~212K; a 2 MiB round there meant
 * both sides blocked writing with full recv-queues). */
#define PULSAR_TP_TCP_ROUND (64ull * 1024ull)

/* SHARED per-process RDMA state: one HCA context, one PD, one CQ, and the
 * slab's MR -- registered ONCE against the HCA and used by every peer's QP
 * (design rule 2).  Nothing here is per-connection; the QP, the peer's
 * exchanged registered-slab address, and the completion bookkeeping all live in
 * pulsar_tp_rdma_link below, one per peer. */
typedef struct {
    pulsar_tp_verbs_api api;
    tp_ibv_ctx ctx;
    tp_ibv_pd pd;
    tp_ibv_cq cq;
    tp_ibv_mr mr;
    tp_ibv_cq bulk_cq;   /* the bulk lane's own CQ: its completions never mix with the gates' */
    tp_ibv_mr bulk_mr;   /* the registered bulk buffer (pulsar_tp_set_bulk), or NULL */
    struct tp_ibv_port_attr port;
    union tp_ibv_gid gid;
    int gid_index;
    char dev_name[PULSAR_TP_NODE_STR];   /* the HCA tp_rdma_open chose, for the NODE frame */
} pulsar_tp_rdma;

/* PER-PEER RDMA state: one QP connected to one peer, that peer's exchanged
 * registered-slab address/rkey, and the completion bookkeeping for exchanges
 * with it.  An n-rank group holds n_ranks-1 of these, one per peer; the HCA,
 * PD, CQ and slab MR above are shared by all of them. */
typedef struct {
    tp_ibv_qp qp;
    pulsar_tp_rdma_info peer;
    uint32_t send_outstanding;  /* signaled sends not yet reaped */
    uint64_t recv_done;         /* highest gate seq whose recv completed */
    uint64_t last_gate_seq;     /* last real decode receive consumed */
    bool recv_window_active;    /* decode recvs are queued ahead */
    pthread_mutex_t post_lock;
    /* The bulk lane (v14): a second UC QP whose receive queue holds only the
     * zero-length receives the peer's RDMA_WRITE_WITH_IMM consumes -- so it
     * never mixes with the gate QP's 16 KB receives. */
    tp_ibv_qp bulk_qp;
    uint64_t bulk_imm_count;    /* imm arrivals reaped (the peer's bulk exchanges, in order) */
    uint32_t bulk_last_imm;     /* the exchange id the latest arrival carried */
} pulsar_tp_rdma_link;

/* One peer in the TP mesh.  A full-mesh rank connects to every other rank
 * (n_ranks-1 peers); each peer has its own control + data sockets.  The
 * `control_fd`/`data_fd` on pulsar_tp are the primary-link fields (== peers[0]
 * for a mesh) so the pairwise decode/batch/control-plane paths keep working
 * unchanged for n=2 and are guarded loudly for n>2. */
typedef struct {
    int rank;           /* peer's rank in the group */
    int control_fd;
    int data_fd;
    uint32_t peer_ctx;
    bool connected;
    /* What the peer told us about itself in the NODE frame (protocol v13). */
    pulsar_tp_node node;
    bool node_known;
    /* This peer's own QP and completion state.  The HCA, PD, CQ and the slab MR
     * are shared (pulsar_tp_rdma on the transport). */
    pulsar_tp_rdma_link rdma;
} pulsar_tp_peer;

struct pulsar_tp {
    pulsar_tp_options opt;
    int rank;                   /* 0 leader, 1 worker */
    int n_ranks;                /* ranks in this TP group (2 for the pair) */
    int n_peers;                /* connected peers (n_ranks-1) */
    pulsar_tp_peer *peers;      /* array [n_peers], sorted by peer rank */
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
    /* The cross-rank logits identity tally (leader only; L243). */
    uint64_t identity_frames;
    uint64_t identity_matched;
    /* The DEFERRED digest ack (L241 4g-2, pulsar_tp_defer_command_ack_digest):
     * the leader's own digest of a step whose peer acks it has not read yet.
     * Settled before any other ack is read. */
    bool deferred_armed = false;
    uint64_t deferred_session = 0, deferred_digest = 0;
    char deferred_operation[96] = {0};
    /* The row lane (4g-2).  Engine thread: gate_seq = last message numbered,
     * row_exch = last exchange numbered.  Proxy thread: proxy_* below. */
    uint64_t gate_seq = 0;
    uint64_t row_exch = 0;
    uint64_t last_gate_exch = 0;   /* the latest exchange numbered that uses the GATE QP (not bulk) */
    pthread_t proxy;
    bool proxy_started = false;
    std::atomic<bool> proxy_stop{false};
    std::atomic<bool> proxy_failed{false};
    pulsar_tp_rdma rdma;    /* RDMA state (loaded lazily at create/attach) */
    pulsar_tp_node self;    /* this rank's own NODE record */
    /* The bulk lane (pulsar_tp_set_bulk): caller-owned, host-pinned and
     * GPU-mapped, laid out out | in[0] | in[1], each bulk_cap bytes. */
    uint8_t *bulk = nullptr;
    uint64_t bulk_bytes = 0;
    uint64_t bulk_cap = 0;
    uint64_t bulk_seq = 0;          /* engine thread: bulk exchanges numbered (buffer parity) */
    uint64_t peer_bulk_base = 0;    /* the peer's bulk buffer and rkey (v14 info) */
    uint32_t peer_bulk_rkey = 0;
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
    /* Keepalive: an idle pair waits forever (pulsar_tp_recv_command), so a
     * peer HOST that disappears without closing (power, cable) must still
     * surface -- ~60 s: 30 s idle, then 3 probes 10 s apart.  Probes are
     * answered by the peer's kernel, so an idle-but-alive peer never trips it. */
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
#ifdef TCP_KEEPIDLE
    int idle = 30, intvl = 10, cnt = 3;
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof(cnt));
#endif
    /* Gate exchanges are latency-critical 16KB messages; large socket
     * buffers only matter for the TCP fallback's pipelining.  The kernel
     * clamps these to net.core.wmem_max anyway.  PULSAR_TP_TEST_TINY_BUFFERS
     * hammers the fallback's write/read-round protocol against the smallest
     * practical buffers (test-only; validates the no-symmetric-write rule). */
    int sz = 4 * 1024 * 1024;
    if (getenv("PULSAR_TP_TEST_TINY_BUFFERS")) sz = 32 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
}

static int tp_listen(const char *host, int port, int backlog, char *err, size_t errlen) {
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
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && listen(fd, backlog) == 0) break;
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

/** Block until `fd` is readable or the deadline passes.  Returns 1 readable,
 * 0 on timeout.  The control plane had NO deadline: a peer that stopped talking
 * left the other side blocked in pulsar_tp_recv_command/wait_command_ack
 * forever, which is why every blocking mirror in slice 4e was avoided and why
 * the tests wrap themselves in `timeout`.  A deadline turns that hang into the
 * refusal it should always have been. */
static int tp_wait_readable(int fd, double deadline) {
    for (;;) {
        const double remaining = deadline - tp_now_sec();
        if (remaining <= 0.0) return 0;
        struct pollfd pfd;
        pfd.fd = fd;
        pfd.events = POLLIN;
        pfd.revents = 0;
        int ms = (int)(remaining * 1000.0);
        if (ms < 1) ms = 1;
        const int rc = poll(&pfd, 1, ms);
        if (rc > 0) return 1;
        if (rc == 0) return 0;
        if (errno == EINTR) continue;
        return 0;
    }
}

/** The control-plane deadline for one wait, or 0 (never expires) when the
 * transport was built with no timeout. */
static double tp_control_deadline(const pulsar_tp *tp) {
    return tp->timeout_sec ? tp_now_sec() + (double)tp->timeout_sec : 0.0;
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
    /* ibv_query_gid_ex is a header-inline wrapper over the EXPORTED
     * _ibv_query_gid_ex (same pattern as post/recv/poll).  Optional: the GID
     * scan prefers RoCEv2 per-entry but falls back to the raw-byte scan. */
    api->query_gid_ex = reinterpret_cast<__typeof__(api->query_gid_ex)>(
        dlsym(h, "_ibv_query_gid_ex"));
    TP_SYM(alloc_pd, "ibv_alloc_pd");
    TP_SYM(dealloc_pd, "ibv_dealloc_pd");
    TP_SYM(reg_mr, "ibv_reg_mr");
    TP_SYM(dereg_mr, "ibv_dereg_mr");
    TP_SYM(create_cq, "ibv_create_cq");
    TP_SYM(destroy_cq, "ibv_destroy_cq");
    TP_SYM(create_qp, "ibv_create_qp");
    TP_SYM(destroy_qp, "ibv_destroy_qp");
    TP_SYM(modify_qp, "ibv_modify_qp");
    /* post_send/post_recv/poll_cq are header inlines, NOT exported symbols;
     * resolved from the verbs context ops table in tp_rdma_open. */
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

/* The link a PAIR-path function means.  Every entry that used the transport's
 * single former RDMA context is 2-rank only and refuses n>2 by name, and for
 * n==2 there is exactly one peer -- so they all mean peers[0].  n-general paths
 * take a link per peer instead of calling this. */
static pulsar_tp_rdma_link *tp_pair_link(pulsar_tp *tp) {
    return (tp->n_peers > 0) ? &tp->peers[0].rdma : NULL;
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
    /* Auto-pick deterministically: both ranks sort by name so an unpinned
     * selection at least chooses the same ordinal on both hosts.  True link
     * correctness on the two-cable pair still needs PULSAR_TP_RDMA_DEV to
     * name the same physical cable on BOTH ranks. */
    for (int i = 1; i < num; i++)
        for (int j = i; j > 0; j--)
            if (strcmp(r->api.get_device_name(devs[j - 1]),
                       r->api.get_device_name(devs[j])) <= 0)
                break;
            else {
                tp_ibv_device t = devs[j - 1];
                devs[j - 1] = devs[j];
                devs[j] = t;
            }
    if (num > 1 && !getenv("PULSAR_TP_RDMA_DEV"))
        fprintf(stderr,
                "pulsar-tp: %d verbs devices; auto-picked by name — pin the "
                "SAME cable on both ranks with PULSAR_TP_RDMA_DEV=<hca>\n", num);
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
            snprintf(r->dev_name, sizeof(r->dev_name), "%s", name);
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
    }    /* post_send/post_recv/poll_cq are header inlines, NOT exported:
     * resolve them from the ibv_context ops table.  With the real header in
     * scope, read the typed members directly so the layout can never drift.
     * The no-header fallback hard-codes the rdma-core ordering (poll=12,
     * send=26, recv=27, 1-based): the earlier 11/25/26 hit _compat_* stubs —
     * poll_cq became a no-op and our post_recv actually called post_send,
     * so the receive queue was NEVER armed and the peer's messages were
     * dropped without error (the pair's symmetric silent non-delivery). */
#ifdef PULSAR_TP_VERBS_HDR
    {
        struct ibv_context *c = (struct ibv_context *)r->ctx;
        r->api.poll_cq = (__typeof__(r->api.poll_cq))c->ops.poll_cq;
        r->api.post_send = (__typeof__(r->api.post_send))c->ops.post_send;
        r->api.post_recv = (__typeof__(r->api.post_recv))c->ops.post_recv;
    }
#else
    {
        typedef void *const *tramp;
        tramp ops = (tramp)((uint8_t *)r->ctx + sizeof(void *));
        r->api.poll_cq = (__typeof__(r->api.poll_cq))ops[12];
        r->api.post_send = (__typeof__(r->api.post_send))ops[26];
        r->api.post_recv = (__typeof__(r->api.post_recv))ops[27];
    }
#endif
    if (!r->api.poll_cq || !r->api.post_send || !r->api.post_recv) {
        tp_set_err(err, errlen, "tp rdma: verbs ops table null pointers");
        return 0;
    }

    /* GID selection.  Upstream (two-Mac/TB) wants the IPv4-mapped GID, which
     * is a Thunderbolt-ism.  Linux RoCEv2 commonly exposes the usable GID at
     * a non-zero index, so: honor PULSAR_TP_RDMA_GID_INDEX, else prefer the
     * RoCEv2 IPv4-mapped GID (per gid_type), else a RoCEv2 routable GID, else
     * the first non-link-local (routable) GID on the legacy path. */
    const char *gid_env = getenv("PULSAR_TP_RDMA_GID_INDEX");
    r->gid_index = -1;
    if (gid_env) {
        const int gi = atoi(gid_env);
        if (gi >= 0 && gi < (int)r->port.gid_tbl_len &&
            r->api.query_gid(r->ctx, 1, gi, &r->gid) == 0) {
            r->gid_index = gi;
        } else {
            tp_set_err(err, errlen,
                       "tp rdma: bad PULSAR_TP_RDMA_GID_INDEX %d (table %u)",
                       gi, r->port.gid_tbl_len);
            return 0;
        }
    } else {
        /* Prefer a RoCEv2 GID, keyed by its per-entry gid_type.  RoCEv1 and
         * RoCEv2 GIDs share IDENTICAL 16 bytes on an IP'd port (the pair's
         * ::ffff:192.168.9.x is both v1@index2 and v2@index3; only the type
         * tells them apart).  Under RoCEv2 the v1 index is not a valid AV:
         * RC rejects it at RTR with EINVAL, UC silently dead-connects — the
         * pair's original hang class.  pick1 = first RoCEv2 IPv4-mapped
         * (break immediately); pick2 = first RoCEv2 non-link-local; pick3 =
         * any other usable GID (legacy readers / non-RoCE fabrics). */
        int pick1 = -1, pick2 = -1, pick3 = -1;
        for (int j = 0; j < (int)r->port.gid_tbl_len; j++) {
            union tp_ibv_gid tmp;
            uint32_t gtype = TP_IBV_GID_TYPE_IB;   /* unknown with old reader */
            if (r->api.query_gid_ex) {
                struct tp_ibv_gid_entry e;
                (void)memset(&e, 0, sizeof(e));
                if (r->api.query_gid_ex(r->ctx, 1, (uint32_t)j, &e, 0,
                                        sizeof(e)) == 0) {
                    tmp = e.gid;
                    gtype = e.gid_type;
                } else if (r->api.query_gid(r->ctx, 1, j, &tmp) != 0) {
                    continue;
                }
            } else if (r->api.query_gid(r->ctx, 1, j, &tmp) != 0) {
                continue;
            }
            uint64_t hi;
            uint16_t mid, v4tag, top;
            memcpy(&hi, &tmp.raw[0], 8);
            memcpy(&mid, &tmp.raw[8], 2);
            memcpy(&v4tag, &tmp.raw[10], 2);
            memcpy(&top, &tmp.raw[0], 2);
            const bool ipv4_mapped = (hi == 0 && mid == 0 && v4tag == 0xffff);
            const bool link_local = (top == 0xfe80);
            if (link_local) continue;
            if (gtype == TP_IBV_GID_TYPE_ROCE_V2) {
                if (ipv4_mapped) { pick1 = j; break; }
                if (pick2 < 0) pick2 = j;
            } else if (pick3 < 0) {
                pick3 = j;
            }
        }
        r->gid_index = (pick1 >= 0) ? pick1 : (pick2 >= 0) ? pick2 : pick3;
        if (r->gid_index >= 0)
            r->api.query_gid(r->ctx, 1, r->gid_index, &r->gid);
    }
    if (r->gid_index < 0) {
        tp_set_err(err, errlen,
                   "tp rdma: no usable GID on the active port "
                   "(try PULSAR_TP_RDMA_GID_INDEX)");
        return 0;
    }
    fprintf(stderr,
            "pulsar-tp: gid index %d (%02x:%02x:%02x:%02x:...)\n",
            r->gid_index, r->gid.raw[0], r->gid.raw[1], r->gid.raw[2],
            r->gid.raw[3]);
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
    r->bulk_cq = r->api.create_cq(r->ctx, 256, NULL, NULL, 0);
    if (!r->bulk_cq) {
        tp_set_err(err, errlen, "tp rdma: create_cq (bulk) failed");
        return 0;
    }
    /* The QP itself is per PEER and is created later, once the peers exist --
     * the HCA open runs before they do.  See tp_rdma_qp_create. */
    return 1;
}

/* Bring up ONE peer's QP over the shared HCA/PD/CQ.  Split out of the HCA open
 * because the open runs before peers are known: the pair does this after its
 * single peer exists, and the mesh does it once per peer (n>2).  The slab MR is
 * NOT created here -- it is registered once for the whole HCA. */
static int tp_rdma_qp_create(pulsar_tp *tp, pulsar_tp_rdma_link *l,
                             char *err, size_t errlen) {
    if (!l) {
        tp_set_err(err, errlen, "tp rdma: no peer link for QP bring-up");
        return 0;
    }
    struct tp_ibv_qp_init_attr qia;
    (void)memset(&qia, 0, sizeof(qia));
    qia.send_cq = (decltype(qia.send_cq))tp->rdma.cq;
    qia.recv_cq = (decltype(qia.recv_cq))tp->rdma.cq;
    qia.qp_type = TP_IBV_QPT_UC;
    qia.cap.max_send_wr = 256;
    qia.cap.max_recv_wr = 64;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    qia.cap.max_inline_data = 0;
    l->qp = tp->rdma.api.create_qp(tp->rdma.pd, &qia);
    if (!l->qp) {
        tp_set_err(err, errlen, "tp rdma: create_qp(UC): %s", strerror(errno));
        return 0;
    }
    pthread_mutex_init(&l->post_lock, NULL);
    return 1;
}

/* Slab slot a given gate seq lands in.  DS4 fires every slot in order
 * (identity mapping); a projected schedule from the hello skips dense layers
 * and the ATTN slots. */
void pulsar_tp_identity_init_defaults(pulsar_tp_identity *id,
                                      uint64_t gguf_bytes,
                                      uint32_t model_id,
                                      uint32_t n_layer,
                                      uint32_t n_embd,
                                      uint32_t n_vocab,
                                      uint32_t quant_bits,
                                      uint32_t ctx_size) {
    memset(id, 0, sizeof(*id));
    id->gguf_bytes = gguf_bytes;
    id->model_id = model_id;
    id->n_layer = n_layer;
    id->n_embd = n_embd;
    id->n_vocab = n_vocab;
    id->quant_bits = quant_bits;
    id->ctx_size = ctx_size;
    /* DS: every layer fires ATTN then FFN, so there is no sparse schedule —
     * gates_per_token 0 selects the transport's identity slot mapping. */
    id->gate_slot_start = 0;
    id->gate_slot_step = 0;
    id->gates_per_token = 0;
}

uint32_t pulsar_tp_gate_slot(uint32_t n_slots, uint64_t seq,
                             uint32_t gate_slot_start, uint32_t gate_slot_step,
                             uint32_t gates_per_token) {
    if (gates_per_token == 0)
        return (uint32_t)((seq - 1) % n_slots);
    return gate_slot_start +
           (uint32_t)((seq - 1) % gates_per_token) * gate_slot_step;
}

static uint32_t tp_gate_slot(const pulsar_tp *tp, uint64_t seq) {
    return pulsar_tp_gate_slot(tp->n_slots, seq,
                               tp->gate_slot_start, tp->gate_slot_step,
                               tp->gates_per_token);
}

static int tp_rdma_post_gate_recv(pulsar_tp *tp, uint64_t seq);

/* Register the slab with the HCA -- ONCE, shared by every peer's QP (rule 2).
 * Split out of the per-connection bring-up below because an n-rank mesh runs
 * n_ranks-1 QPs over this one MR: register once, connect many. */
static int tp_rdma_register_slab(pulsar_tp *tp, char *err, size_t errlen) {
    tp->rdma.mr = tp->rdma.api.reg_mr(tp->rdma.pd, tp->slab, (size_t)tp->slab_bytes,
                          TP_IBV_ACCESS_LOCAL_WRITE | TP_IBV_ACCESS_REMOTE_READ |
                          TP_IBV_ACCESS_REMOTE_WRITE);
    if (!tp->rdma.mr) {
        tp_set_err(err, errlen, "tp rdma: reg_mr(%llu bytes): %s",
                   (unsigned long long)tp->slab_bytes, strerror(errno));
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
    return 1;
}

/* INIT -> RTR -> RTS for one UC QP toward `peer` (GRH addressing through its
 * GID, the lower of the two ports' MTUs).  The gate QP and the bulk QP share it. */
static int tp_rdma_qp_connect(pulsar_tp *tp, tp_ibv_qp qp, const pulsar_tp_rdma_info *peer,
                              uint32_t dest_qpn, uint32_t dest_psn, uint32_t my_psn, int rank,
                              char *err, size_t errlen) {
    /* INIT -> RTR -> RTS with GRH addressing through the exchanged GID. */
    struct tp_ibv_qp_attr a;
    (void)memset(&a, 0, sizeof(a));
    a.qp_state = TP_IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = 1;
    a.qp_access_flags = TP_IBV_ACCESS_LOCAL_WRITE | TP_IBV_ACCESS_REMOTE_READ |
                        TP_IBV_ACCESS_REMOTE_WRITE;
    if (tp->rdma.api.modify_qp(qp, &a,
            TP_IBV_QP_STATE | TP_IBV_QP_PKEY_INDEX | TP_IBV_QP_PORT |
            TP_IBV_QP_ACCESS_FLAGS) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify INIT rank %d: %s", rank, strerror(errno));
        return 0;
    }
    (void)memset(&a, 0, sizeof(a));
    a.qp_state = TP_IBV_QPS_RTR;
    /* path_mtu must AGREE between the two QPs.  The pair's second link
     * reports active_mtu 1024 on one rank and 4096 on the other; each rank
     * using its own makes the faster side emit frames the narrower port
     * silently drops (bisected with a probe: >1024 fails with per-port
     * path_mtu, delivers with a uniform 1024 on both).  Use the lower of the
     * two ports' MTUs, on both ranks. */
    a.path_mtu = tp->rdma.port.active_mtu;
    if (peer->mtu != 0 && (int)peer->mtu < (int)a.path_mtu)
        a.path_mtu = (decltype(a.path_mtu))(int)peer->mtu;
    a.dest_qp_num = dest_qpn;
    a.rq_psn = dest_psn;
    a.ah_attr.dlid = (uint16_t)peer->lid;
    a.ah_attr.port_num = 1;
    a.ah_attr.is_global = 1;
    memcpy(a.ah_attr.grh.dgid.raw, peer->gid, 16);
    a.ah_attr.grh.sgid_index = (uint8_t)tp->rdma.gid_index;
    a.ah_attr.grh.hop_limit = 1;
    if (tp->rdma.api.modify_qp(qp, &a,
            TP_IBV_QP_STATE | TP_IBV_QP_AV | TP_IBV_QP_PATH_MTU |
            TP_IBV_QP_DEST_QPN | TP_IBV_QP_RQ_PSN) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify RTR rank %d: %s", rank, strerror(errno));
        return 0;
    }
    (void)memset(&a, 0, sizeof(a));
    a.qp_state = TP_IBV_QPS_RTS;
    a.sq_psn = my_psn;
    if (tp->rdma.api.modify_qp(qp, &a, TP_IBV_QP_STATE | TP_IBV_QP_SQ_PSN) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify RTS rank %d: %s", rank, strerror(errno));
        return 0;
    }
    return 1;
}

/* The bulk lane's zero-length receive: consumed by the peer's final
 * RDMA_WRITE_WITH_IMM of an exchange (the data itself lands by RDMA write). */
static int tp_bulk_post_imm_recv(pulsar_tp *tp, pulsar_tp_rdma_link *r) {
    struct tp_ibv_recv_wr wr;
    struct tp_ibv_recv_wr *bad = NULL;
    (void)memset(&wr, 0, sizeof(wr));
    wr.wr_id = PULSAR_TP_BULK_WR_TAG;
    wr.sg_list = NULL;
    wr.num_sge = 0;
    if (tp->rdma.api.post_recv(r->bulk_qp, &wr, &bad) != 0) {
        fprintf(stderr, "pulsar-tp: bulk imm post_recv: %s\n", strerror(errno));
        return 0;
    }
    return 1;
}

/* Bring up ONE peer's QP: create it over the shared PD/CQ, exchange this rank's
 * registered-slab address and QP identity with THAT peer over its own control
 * socket, then INIT -> RTR -> RTS.  The MR is already registered above, so
 * every peer sees the same slab at the same rkey.  This is the function the
 * pair calls once and the mesh calls n_ranks-1 times. */
static int tp_rdma_link_bringup(pulsar_tp *tp, pulsar_tp_peer *pp,
                                char *err, size_t errlen) {
    pulsar_tp_rdma_link *r = &pp->rdma;
    if (!tp_rdma_qp_create(tp, r, err, errlen)) return 0;
    pulsar_tp_rdma_info mine;
    (void)memset(&mine, 0, sizeof(mine));
    mine.slab_base = (uint64_t)(uintptr_t)tp->slab;
    mine.rkey = TP_RKEY(tp->rdma.mr);
    mine.qpn = TP_QPN(r->qp);
    /* PSND must differ per peer, so mix the peer in. */
    mine.psn = (uint32_t)(getpid() ^ (uintptr_t)tp ^ (uintptr_t)pp) & 0xffffff;
    if (tp->rdma.bulk_mr && tp->n_ranks == 2) {
        struct tp_ibv_qp_init_attr qia;
        (void)memset(&qia, 0, sizeof(qia));
        qia.send_cq = (decltype(qia.send_cq))tp->rdma.bulk_cq;
        qia.recv_cq = (decltype(qia.recv_cq))tp->rdma.bulk_cq;
        qia.qp_type = TP_IBV_QPT_UC;
        qia.cap.max_send_wr = 64;
        qia.cap.max_recv_wr = 2u * PULSAR_TP_BULK_IMM_WINDOW;
        qia.cap.max_send_sge = 1;
        qia.cap.max_recv_sge = 1;
        r->bulk_qp = tp->rdma.api.create_qp(tp->rdma.pd, &qia);
        if (!r->bulk_qp) {
            tp_set_err(err, errlen, "tp rdma: create_qp(UC, bulk): %s", strerror(errno));
            return 0;
        }
        mine.bulk_base = (uint64_t)(uintptr_t)tp->bulk;
        mine.bulk_rkey = TP_RKEY(tp->rdma.bulk_mr);
        mine.bulk_qpn = TP_QPN(r->bulk_qp);
        mine.bulk_psn = (mine.psn ^ 0x5a5a5au) & 0xffffff;
    }
    mine.mtu = (uint32_t)tp->rdma.port.active_mtu;
    mine.lid = tp->rdma.port.lid;
    memcpy(mine.gid, tp->rdma.gid.raw, 16);
    mine.link_layer = tp->rdma.port.link_layer;
    if (!tp_send_frame(pp->control_fd, PULSAR_TP_FRAME_RDMA_INFO,
                       &mine, sizeof(mine))) {
        tp_set_err(err, errlen, "tp rdma: info send to rank %d failed", pp->rank);
        return 0;
    }
    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(pp->control_fd, &type, &bytes) ||
        type != PULSAR_TP_FRAME_RDMA_INFO || bytes != sizeof(r->peer) ||
        !tp_read_full(pp->control_fd, &r->peer, sizeof(r->peer))) {
        tp_set_err(err, errlen, "tp rdma: info recv from rank %d failed", pp->rank);
        return 0;
    }

    if (!tp_rdma_qp_connect(tp, r->qp, &r->peer, r->peer.qpn, r->peer.psn, mine.psn, pp->rank,
                            err, errlen))
        return 0;
    /* The bulk lane: both ranks attached a bulk buffer (the info says so), so
     * connect the second QP and post its imm window BEFORE the ready barrier --
     * a WRITE_WITH_IMM that arrives before the peer posted its receive is
     * dropped by UC. */
    if (r->bulk_qp && r->peer.bulk_qpn != 0) {
        if (!tp_rdma_qp_connect(tp, r->bulk_qp, &r->peer, r->peer.bulk_qpn, r->peer.bulk_psn,
                                mine.bulk_psn, pp->rank, err, errlen))
            return 0;
        tp->peer_bulk_base = r->peer.bulk_base;
        tp->peer_bulk_rkey = r->peer.bulk_rkey;
        for (uint32_t k = 0; k < PULSAR_TP_BULK_IMM_WINDOW; k++)
            if (!tp_bulk_post_imm_recv(tp, r)) {
                tp_set_err(err, errlen, "tp rdma: bulk imm window post to rank %d failed", pp->rank);
                return 0;
            }
    }
    /* Leave the receive queue empty for an initial bulk prefill; the first
     * decode gate arms the normal lookahead window. */
    if (!tp_send_frame(pp->control_fd, PULSAR_TP_FRAME_RDMA_READY, NULL, 0)) {
        tp_set_err(err, errlen, "tp rdma: ready send to rank %d failed", pp->rank);
        return 0;
    }
    uint32_t rtype = 0, rbytes = 0;
    if (!tp_read_frame_header(pp->control_fd, &rtype, &rbytes) ||
        rtype != PULSAR_TP_FRAME_RDMA_READY || rbytes != 0) {
        tp_set_err(err, errlen, "tp rdma: ready barrier with rank %d failed", pp->rank);
        return 0;
    }
    fprintf(stderr, "pulsar-tp: rdma link up to rank %d (remote qpn %u)\n",
            pp->rank, (unsigned)r->peer.qpn);
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
    pulsar_tp_rdma_link *r = tp_pair_link(tp);
    struct tp_ibv_wc wc[16];
    int n = tp->rdma.api.poll_cq(tp->rdma.cq, 16, wc);
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
    pulsar_tp_rdma_link *r = tp_pair_link(tp);
    const uint32_t slot = tp_gate_slot(tp, seq);
    const uintptr_t base =
        (uintptr_t)(tp->slab + tp->layout.in_off +
                    (uint64_t)slot * tp->vec_bytes);
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
        sge.lkey = TP_LKEY(tp->rdma.mr);
        wr.wr_id = last ? seq : 0;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        if (tp->rdma.api.post_recv(r->qp, &wr, &bad) != 0) {
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
    pulsar_tp_rdma_link *r = tp_pair_link(tp);
    const uint32_t slot = layer * PULSAR_TP_GATES_PER_LAYER + gate;
    if (slot != tp_gate_slot(tp, seq)) {
        fprintf(stderr, "pulsar-tp: gate order broke: layer %u gate %u vs seq %llu\n",
                layer, gate, (unsigned long long)seq);
        return 0;
    }
    const uintptr_t send_base =
        (uintptr_t)(tp->slab + tp->layout.out_off +
                    (uint64_t)slot * tp->vec_bytes);
    pthread_mutex_lock(&r->post_lock);
    int ok = 1;
    if (!r->recv_window_active) {
        for (uint64_t s = seq; ok && s < seq + PULSAR_TP_RDMA_RECV_WINDOW; s++)
            ok = tp_rdma_post_gate_recv(tp, s);
        /* A rank that SENDS before the peer's window is armed silently loses
         * its first N messages under UC (no error, no completion), shifting
         * every later pairing by one.  That happens on the pair because the
         * leader runs ahead of the worker while it is still in create/the
         * control handshake.  The control socket is full-duplex, so: after
         * arming, both sides exchange RDMA_GATE_ARMED and only then send. */
        if (ok) ok = tp_send_frame(tp->control_fd,
                                   PULSAR_TP_FRAME_RDMA_GATE_ARMED, NULL, 0);
        if (ok) {
            uint32_t rtype = 0, rbytes = 0;
            ok = tp_read_frame_header(tp->control_fd, &rtype, &rbytes) &&
                 rtype == PULSAR_TP_FRAME_RDMA_GATE_ARMED && rbytes == 0;
        }
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
        sge.lkey = TP_LKEY(tp->rdma.mr);
        wr.wr_id = seq;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = TP_IBV_WR_SEND;
        wr.send_flags = TP_IBV_SEND_SIGNALED;
        ok = tp->rdma.api.post_send(r->qp, &wr, &bad) == 0;
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

/* The row lane's three slab words (layout: pulsar_tp.h). */
static inline volatile uint64_t *tp_row_desc(pulsar_tp *tp) {
    return (volatile uint64_t *)(tp->slab + tp->layout.out_flags_off);
}
static inline std::atomic<uint64_t> *tp_row_done(pulsar_tp *tp) {
    return reinterpret_cast<std::atomic<uint64_t> *>(tp->slab + tp->layout.in_flags_off);
}
static inline volatile uint32_t *tp_row_err(pulsar_tp *tp) {
    return (volatile uint32_t *)(tp->slab + tp->layout.gpu_flags_off);
}

/* ROW-LANE ABORT (L241 4g-2).  A rank that will not run a step its peer is
 * running -- a refused frame, a failed body -- leaves the peer's GPU spinning
 * in a combine for rows that never come, until the transport timeout (900 s:
 * the pair looked hung).  The failing rank sends this message on the DATA
 * socket, idle during decode (the control socket may hold pipelined acks ahead
 * of it); the peer's proxy peeks for it while it waits on an exchange and
 * latches the lane's error word, which every spinning kernel watches -- the
 * stream drains in microseconds and the drain refuses the step
 * (pulsar_gpu_end_commands).  The message is never consumed: the pair is
 * failed from here on.  The magic is distinct from every data-socket header. */
static const char TP_ROW_ABORT_MAGIC[8] = { 'P', 'T', 'P', 'A', 'B', 'O', 'R', 'T' };
typedef struct {
    char magic[8];
    int32_t rank;
    char why[116];
} tp_row_abort_msg;

static void tp_row_err_latch(pulsar_tp *tp, uint32_t code) {
    if (tp->slab && pulsar_tp_row_lane(tp) && *tp_row_err(tp) == 0) *tp_row_err(tp) = code;
}

void pulsar_tp_row_lane_abort(pulsar_tp *tp, const char *why) {
    if (!tp) return;
    static std::atomic<bool> sent{false};
    pulsar_tp_mark_failed(tp);
    tp_row_err_latch(tp, 2u);
    if (sent.exchange(true)) return;
    fprintf(stderr, "pulsar-tp: rank %d aborts the pair's row lane: %s\n", tp->rank,
            why ? why : "(no reason)");
    tp_row_abort_msg m;
    memset(&m, 0, sizeof(m));
    memcpy(m.magic, TP_ROW_ABORT_MAGIC, sizeof(m.magic));
    m.rank = tp->rank;
    snprintf(m.why, sizeof(m.why), "%s", why ? why : "");
    for (int i = 0; i < tp->n_peers; i++)
        if (tp->peers[i].data_fd >= 0) (void)tp_write_full(tp->peers[i].data_fd, &m, sizeof(m));
}

/* 1 = the peer sent the abort (printed once, `why` filled). */
static int tp_peer_aborted(pulsar_tp *tp) {
    tp_row_abort_msg m;
    const ssize_t n = recv(tp->data_fd, &m, sizeof(m), MSG_PEEK | MSG_DONTWAIT);
    if (n < (ssize_t)sizeof(m) || memcmp(m.magic, TP_ROW_ABORT_MAGIC, sizeof(m.magic)) != 0) return 0;
    m.why[sizeof(m.why) - 1] = '\0';
    fprintf(stderr, "pulsar-tp: rank %d aborted the pair's row lane: %s\n", (int)m.rank, m.why);
    return 1;
}

/* Arm the row lane's receive window at message `first`: post WINDOW
 * receives, then the ARMED handshake on the control socket (a rank that sends
 * before the peer's window is armed silently loses messages under UC).
 * Engine thread only. */
static int tp_row_lane_arm(pulsar_tp *tp, uint64_t first) {
    pulsar_tp_rdma_link *r = tp_pair_link(tp);
    pthread_mutex_lock(&r->post_lock);
    int ok = 1;
    for (uint64_t s = first; ok && s < first + PULSAR_TP_RDMA_RECV_WINDOW; s++)
        ok = tp_rdma_post_gate_recv(tp, s);
    if (ok) ok = tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_RDMA_GATE_ARMED, NULL, 0);
    if (ok) {
        uint32_t rtype = 0, rbytes = 0;
        ok = tp_read_frame_header(tp->control_fd, &rtype, &rbytes) &&
             rtype == PULSAR_TP_FRAME_RDMA_GATE_ARMED && rbytes == 0;
    }
    if (ok) r->recv_window_active = true;
    pthread_mutex_unlock(&r->post_lock);
    return ok;
}

/* One exchange, proxy thread.  Message t is sent from out-slot (t-1)%n_slots
 * (the GPU staged it there) and received into in-slot (t-1)%n_slots (posted
 * WINDOW messages earlier).  Completion = the peer's `rows` landed AND our
 * own sends retired, so the GPU may rewrite those out-slots the moment it
 * sees done.  The window is re-posted before done is written, so the proxy is
 * idle -- and the QP quiet -- whenever the engine observes done == enqueued. */
static int tp_row_proxy_exchange(pulsar_tp *tp, uint64_t first, uint32_t rows) {
    pulsar_tp_rdma_link *r = tp_pair_link(tp);
    const uint64_t vb = tp->vec_bytes, last = first + rows - 1u;
    pthread_mutex_lock(&r->post_lock);
    int ok = r->recv_window_active ? 1 : 0;
    if (!ok) fprintf(stderr, "pulsar-tp: row lane proxy: exchange published with no armed window\n");
    for (uint32_t k = 0; ok && k < rows; k++) {
        const uint64_t t = first + k;
        struct tp_ibv_sge sge;
        struct tp_ibv_send_wr wr;
        struct tp_ibv_send_wr *bad = NULL;
        (void)memset(&wr, 0, sizeof(wr));
        sge.addr = (uintptr_t)(tp->slab + tp->layout.out_off + (uint64_t)tp_gate_slot(tp, t) * vb);
        sge.length = (uint32_t)vb;
        sge.lkey = TP_LKEY(tp->rdma.mr);
        wr.wr_id = t;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = TP_IBV_WR_SEND;
        wr.send_flags = TP_IBV_SEND_SIGNALED;
        ok = tp->rdma.api.post_send(r->qp, &wr, &bad) == 0;
        if (!ok) fprintf(stderr, "pulsar-tp: row lane post_send (seq %llu): %s\n",
                         (unsigned long long)t, strerror(errno));
        else r->send_outstanding++;
    }
    const double t_posted = tp_now_sec();
    const double deadline = t_posted + (double)tp->timeout_sec;
    double next_abort_check = t_posted + 1e-3;
    while (ok && (r->recv_done < last || r->send_outstanding > 0)) {
        ok = tp_rdma_drain_cq(tp);
        if (ok && tp_now_sec() > next_abort_check) {
            next_abort_check = tp_now_sec() + 1e-3;
            if (tp_peer_aborted(tp)) {
                tp_row_err_latch(tp, 2u);
                pulsar_tp_mark_failed(tp);
                ok = 0;
            }
        }
        if (ok && tp_now_sec() > deadline) {
            fprintf(stderr, "pulsar-tp: row lane: timeout at seq %llu (recv_done %llu, "
                            "%u sends outstanding)\n", (unsigned long long)last,
                    (unsigned long long)r->recv_done, r->send_outstanding);
            ok = 0;
        }
    }
    for (uint32_t k = 0; ok && k < rows; k++)
        ok = tp_rdma_post_gate_recv(tp, first + k + PULSAR_TP_RDMA_RECV_WINDOW);
    if (ok) r->last_gate_seq = last;
    pthread_mutex_unlock(&r->post_lock);
    return ok;
}

/* One BULK exchange, proxy thread (v14).  This rank's payload sits in its bulk
 * out-region (the GPU staged it); it is RDMA-written straight into the peer's
 * in[buf], PULSAR_TP_BULK_PIECE at a time, the last write carrying the exchange
 * id as immediate data.  Completion = the peer's matching WITH_IMM arrived (its
 * data is in our in[buf]: UC delivers a QP's writes in order, so the imm lands
 * after them) AND our own writes retired (the out-region may be restaged).
 * One zero-length receive is re-posted per exchange, keeping the window full.
 * No handshake: both ranks number bulk exchanges identically, and the window
 * was armed before the bring-up's ready barrier. */
static int tp_bulk_proxy_exchange(pulsar_tp *tp, uint64_t e, uint64_t bytes, uint32_t buf) {
    pulsar_tp_rdma_link *r = tp_pair_link(tp);
    const uint32_t n = (uint32_t)((bytes + PULSAR_TP_BULK_PIECE - 1u) / PULSAR_TP_BULK_PIECE);
    if (!r || !r->bulk_qp || n == 0 || n > 64u) {
        fprintf(stderr, "pulsar-tp: bulk exchange %llu refused (%llu bytes, %u pieces)\n",
                (unsigned long long)e, (unsigned long long)bytes, n);
        return 0;
    }
    const uint64_t remote = tp->peer_bulk_base + (1u + buf) * tp->bulk_cap;
    struct tp_ibv_sge sge[64];
    struct tp_ibv_send_wr wr[64];
    (void)memset(wr, 0, sizeof(wr));
    for (uint32_t i = 0; i < n; i++) {
        const uint64_t off = (uint64_t)i * PULSAR_TP_BULK_PIECE;
        const uint64_t len = bytes - off < PULSAR_TP_BULK_PIECE ? bytes - off : PULSAR_TP_BULK_PIECE;
        const bool last = i + 1u == n;
        sge[i].addr = (uintptr_t)(tp->bulk + off);
        sge[i].length = (uint32_t)len;
        sge[i].lkey = TP_LKEY(tp->rdma.bulk_mr);
        wr[i].wr_id = PULSAR_TP_BULK_WR_TAG | e;
        wr[i].sg_list = &sge[i];
        wr[i].num_sge = 1;
        wr[i].opcode = last ? TP_IBV_WR_RDMA_WRITE_WITH_IMM : TP_IBV_WR_RDMA_WRITE;
        wr[i].send_flags = last ? TP_IBV_SEND_SIGNALED : 0;
        wr[i].wr.rdma.remote_addr = remote + off;
        wr[i].wr.rdma.rkey = tp->peer_bulk_rkey;
        if (last) wr[i].imm_data = (uint32_t)e;
        wr[i].next = last ? NULL : &wr[i + 1u];
    }
    struct tp_ibv_send_wr *bad = NULL;
    if (tp->rdma.api.post_send(r->bulk_qp, wr, &bad) != 0) {
        fprintf(stderr, "pulsar-tp: bulk rdma write post (exchange %llu): %s\n",
                (unsigned long long)e, strerror(errno));
        return 0;
    }
    const double t_posted = tp_now_sec();
    /* The peer's k-th bulk exchange is our k-th imm arrival. */
    const uint64_t want = r->bulk_imm_count + 1u;
    bool sent = false;
    const double deadline = t_posted + (double)tp->timeout_sec;
    double next_abort_check = t_posted + 1e-3;
    while (!sent || r->bulk_imm_count < want) {
        struct tp_ibv_wc wc[8];
        const int got = tp->rdma.api.poll_cq(tp->rdma.bulk_cq, 8, wc);
        if (got < 0) return 0;
        for (int i = 0; i < got; i++) {
            if (wc[i].status != TP_IBV_WC_SUCCESS) {
                fprintf(stderr, "pulsar-tp: bulk rdma completion error: %s (exchange %llu)\n",
                        tp_wc_status_str(wc[i].status), (unsigned long long)e);
                return 0;
            }
            if (wc[i].opcode & TP_IBV_WC_RECV) {
                r->bulk_imm_count++;
                r->bulk_last_imm = wc[i].imm_data;
            } else {
                sent = true;
            }
        }
        const double now = tp_now_sec();
        if (now > next_abort_check) {
            next_abort_check = now + 1e-3;
            if (tp_peer_aborted(tp)) {
                tp_row_err_latch(tp, 2u);
                pulsar_tp_mark_failed(tp);
                return 0;
            }
            if (now > deadline) {
                fprintf(stderr, "pulsar-tp: bulk exchange %llu timed out (imm %llu/%llu, sent %d)\n",
                        (unsigned long long)e, (unsigned long long)r->bulk_imm_count,
                        (unsigned long long)want, (int)sent);
                return 0;
            }
        }
    }
    /* The ranks number exchanges identically: the peer's arrival must carry
     * this exchange's id, or the lanes are out of step. */
    if (r->bulk_last_imm != (uint32_t)e) {
        fprintf(stderr, "pulsar-tp: bulk exchange %llu received the peer's exchange %u -- the ranks' "
                        "exchange order diverged\n", (unsigned long long)e, r->bulk_last_imm);
        return 0;
    }
    if (!tp_bulk_post_imm_recv(tp, r)) return 0;
    return 1;
}

static void *tp_row_proxy_main(void *arg) {
    pulsar_tp *tp = static_cast<pulsar_tp *>(arg);
    volatile uint64_t *desc = tp_row_desc(tp);
    std::atomic<uint64_t> *done = tp_row_done(tp);
    uint64_t last_exch = 0, next_msg = 1;
    double idle_since = tp_now_sec();
    while (!tp->proxy_stop.load(std::memory_order_acquire)) {
        const uint64_t e = __atomic_load_n(&desc[0], __ATOMIC_ACQUIRE);
        if (e == last_exch) {
            /* busy-poll while decode is running; back off once idle */
            if (tp_now_sec() - idle_since > 0.002) usleep(50);
            continue;
        }
        if (desc[2] & PULSAR_TP_DESC_BULK_FLAG) {
            /* A bulk exchange (v14): desc[1] = bytes, desc[2] low bit = the
             * receive buffer.  It takes no gate messages, so next_msg stays. */
            const uint64_t bytes = desc[1];
            const uint32_t buf = (uint32_t)(desc[2] & 1u);
            if (e != last_exch + 1u || bytes == 0 || bytes > tp->bulk_cap) {
                fprintf(stderr, "pulsar-tp: row lane proxy: bulk descriptor out of order or oversized "
                                "(exchange %llu after %llu, %llu bytes, cap %llu) -- lane failed\n",
                        (unsigned long long)e, (unsigned long long)last_exch,
                        (unsigned long long)bytes, (unsigned long long)tp->bulk_cap);
                tp->proxy_failed.store(true, std::memory_order_release);
                break;
            }
            if (!tp_bulk_proxy_exchange(tp, e, bytes, buf)) {
                tp->proxy_failed.store(true, std::memory_order_release);
                break;
            }
            last_exch = e;
            done->store(e, std::memory_order_release);
            idle_since = tp_now_sec();
            continue;
        }
        const uint64_t first = desc[1];
        const uint32_t rows = (uint32_t)desc[2];
        if (e != last_exch + 1u || first != next_msg || rows == 0 || rows > PULSAR_TP_BATCH_MAX_ROWS) {
            fprintf(stderr, "pulsar-tp: row lane proxy: descriptor out of order (exchange %llu after %llu, "
                            "first msg %llu want %llu, %u rows) -- lane failed\n",
                    (unsigned long long)e, (unsigned long long)last_exch,
                    (unsigned long long)first, (unsigned long long)next_msg, rows);
            tp->proxy_failed.store(true, std::memory_order_release);
            break;
        }
        if (!tp_row_proxy_exchange(tp, first, rows)) {
            tp->proxy_failed.store(true, std::memory_order_release);
            break;
        }
        last_exch = e;
        next_msg = first + rows;
        done->store(e, std::memory_order_release);
        idle_since = tp_now_sec();
    }
    return NULL;
}

static int tp_row_proxy_start(pulsar_tp *tp, char *err, size_t errlen) {
    tp_row_desc(tp)[0] = 0; tp_row_desc(tp)[1] = 0; tp_row_desc(tp)[2] = 0;
    tp_row_done(tp)->store(0, std::memory_order_release);
    *tp_row_err(tp) = 0;
    tp->proxy_stop.store(false);
    if (pthread_create(&tp->proxy, NULL, tp_row_proxy_main, tp) != 0) {
        tp_set_err(err, errlen, "tp: row lane proxy thread: %s", strerror(errno));
        return 0;
    }
    tp->proxy_started = true;
    return 1;
}

static void tp_row_proxy_stop(pulsar_tp *tp) {
    if (!tp->proxy_started) return;
    tp->proxy_stop.store(true, std::memory_order_release);
    pthread_join(tp->proxy, NULL);
    tp->proxy_started = false;
}

static int tp_rdma_big_gate_capable(const pulsar_tp *tp, const pulsar_tp_rdma_link *l) {
    const uint64_t stage_bytes =
        (uint64_t)PULSAR_TP_RDMA_BULK_SLOTS * PULSAR_TP_RDMA_MAX_MSG;
    const uint64_t batch_region_bytes =
        (uint64_t)tp->n_layer * PULSAR_TP_BATCH_MAX_ROWS * tp->vec_bytes;
    return l && l->qp && tp->rdma.mr && batch_region_bytes >= stage_bytes;
}

/* Decode keeps a lookahead window of receives on the latency QP.  Before a
 * later prompt can reuse that QP for bulk rows, consume those receives with
 * dummy sends on both ranks.  The TCP big-gate header exchange is the barrier
 * that guarantees both sides have reached this transition. */
static int tp_rdma_drain_decode_window(pulsar_tp *tp) {
    pulsar_tp_rdma_link *r = tp_pair_link(tp);
    if (!r->recv_window_active) return 1;

    const uint32_t chunks_per_gate =
        (uint32_t)((tp->vec_bytes + PULSAR_TP_RDMA_MAX_MSG - 1u) /
                   PULSAR_TP_RDMA_MAX_MSG);
    const uint32_t nwr = PULSAR_TP_RDMA_RECV_WINDOW * chunks_per_gate;
    struct tp_ibv_sge sge[PULSAR_TP_RDMA_RECV_WINDOW * 2u];
    struct tp_ibv_send_wr wr[PULSAR_TP_RDMA_RECV_WINDOW * 2u];
    (void)memset(wr, 0, sizeof(wr));
    uint8_t *scratch = tp->slab + tp->layout.batch_out_off;
    uint32_t wi = 0;
    for (uint32_t gate = 0; gate < PULSAR_TP_RDMA_RECV_WINDOW; gate++) {
        for (uint64_t off = 0; off < tp->vec_bytes; ) {
            const uint64_t len = tp->vec_bytes - off > PULSAR_TP_RDMA_MAX_MSG ?
                PULSAR_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
            sge[wi].addr = (uintptr_t)(scratch + off);
            sge[wi].length = (uint32_t)len;
            sge[wi].lkey = TP_LKEY(tp->rdma.mr);
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
    if (tp->rdma.api.post_send(r->qp, wr, &bad) != 0) {
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
        int n = tp->rdma.api.poll_cq(tp->rdma.cq,
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
static int tp_rdma_big_gate_exchange(pulsar_tp *tp, pulsar_tp_rdma_link *link,
                                     const void *out, void *in, uint64_t bytes) {
    pulsar_tp_rdma_link *r = link;
    if (!tp_rdma_big_gate_capable(tp, r) || r->recv_window_active) return 0;

    /* Payloads already inside the registered slab (the engine's <=
     * PULSAR_TP_BATCH_MAX_ROWS gate rows) ride DIRECT; ordinary prefill tensors
     * use the idle verify regions as registered staging. */
    const uintptr_t out_lo = (uintptr_t)out;
    const uintptr_t in_lo = (uintptr_t)in;
    const bool direct = pulsar_tp_in_slab(tp, out, bytes) &&
                        pulsar_tp_in_slab(tp, in, bytes);
    uint8_t *stage_send = tp->slab + tp->layout.batch_out_off;
    uint8_t *stage_recv = tp->slab + tp->layout.batch_in_off;
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
            recv_sge[i].lkey = TP_LKEY(tp->rdma.mr);
            recv_wr[i].wr_id = PULSAR_TP_RDMA_BULK_WR_TAG | ((uint64_t)i + 1u);
            recv_wr[i].sg_list = &recv_sge[i];
            recv_wr[i].num_sge = 1;
            recv_wr[i].next = i + 1u < chunks ? &recv_wr[i + 1u] : NULL;
        }
        struct tp_ibv_recv_wr *bad_recv = NULL;
        if (tp->rdma.api.post_recv(r->qp, recv_wr, &bad_recv) != 0) {
            fprintf(stderr, "pulsar-tp: bulk rdma post_recv: %s\n",
                    strerror(errno));
            return 0;
        }
        /* Same arming race as the first gate: if our sends leave before the
         * peer has posted this round's recv window, UC drops the tail of the
         * batch and the round wait hangs forever (bisected with a settle: a
         * 2 ms pause before sending makes the whole transport test pass).
         * Deterministic fix: exchange an armed ack over the control socket
         * (quiescent here after the batch header handshake) before sending —
         * both sides have the round's recvs posted once both acks crossed. */
        if (!tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_RDMA_GATE_ARMED,
                           NULL, 0))
            return 0;
        {
            uint32_t rtype = 0, rbytes = 0;
            if (!tp_read_frame_header(tp->control_fd, &rtype, &rbytes) ||
                rtype != PULSAR_TP_FRAME_RDMA_GATE_ARMED || rbytes != 0)
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
            send_sge[i].lkey = TP_LKEY(tp->rdma.mr);
            send_wr[i].wr_id = PULSAR_TP_RDMA_BULK_WR_TAG | ((uint64_t)i + 1u);
            send_wr[i].sg_list = &send_sge[i];
            send_wr[i].num_sge = 1;
            send_wr[i].opcode = TP_IBV_WR_SEND;
            send_wr[i].send_flags = i + 1u == chunks ? TP_IBV_SEND_SIGNALED : 0;
            send_wr[i].next = i + 1u < chunks ? &send_wr[i + 1u] : NULL;
        }
        struct tp_ibv_send_wr *bad_send = NULL;
        if (tp->rdma.api.post_send(r->qp, send_wr, &bad_send) != 0) {
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
            int n = tp->rdma.api.poll_cq(tp->rdma.cq,
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
    /* Safe when bring-up never completed: a failed hello leaves tp->peers
     * unallocated (or n_peers 0), so there is no link and no per-peer QP to
     * destroy -- only the shared HCA objects, each checked individually. */
    pulsar_tp_rdma_link *r = tp_pair_link(tp);
    if (r && r->qp) tp->rdma.api.destroy_qp(r->qp);
    if (r && r->bulk_qp) tp->rdma.api.destroy_qp(r->bulk_qp);
    if (tp->rdma.mr) tp->rdma.api.dereg_mr(tp->rdma.mr);
    if (tp->rdma.bulk_mr) tp->rdma.api.dereg_mr(tp->rdma.bulk_mr);
    if (tp->rdma.cq) tp->rdma.api.destroy_cq(tp->rdma.cq);
    if (tp->rdma.bulk_cq) tp->rdma.api.destroy_cq(tp->rdma.bulk_cq);
    if (tp->rdma.pd) tp->rdma.api.dealloc_pd(tp->rdma.pd);
    if (tp->rdma.ctx) tp->rdma.api.close_device(tp->rdma.ctx);
    if (r) { r->qp = NULL; r->bulk_qp = NULL; }
    tp->rdma.mr = NULL; tp->rdma.cq = NULL; tp->rdma.pd = NULL; tp->rdma.ctx = NULL;
    tp->rdma.bulk_mr = NULL; tp->rdma.bulk_cq = NULL;
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
    tp_row_proxy_stop(tp);   /* before the QP it posts to goes away */
    tp_rdma_close(tp);
    /* The primary link's fds are ALSO peers[0]'s once bring-up completed, so
     * each descriptor is closed once: the primary here, and below every peer
     * descriptor that is not the primary.  On a bring-up that failed before
     * the primary was assigned (control_fd still -1) the peers' own fds are
     * the only handles, and the comparison closes them too. */
    if (tp->control_fd >= 0) close(tp->control_fd);
    if (tp->data_fd >= 0) close(tp->data_fd);
    if (tp->peers) {
        for (int i = 0; i < tp->n_peers; i++) {
            if (tp->peers[i].control_fd >= 0 && tp->peers[i].control_fd != tp->control_fd)
                close(tp->peers[i].control_fd);
            if (tp->peers[i].data_fd >= 0 && tp->peers[i].data_fd != tp->data_fd)
                close(tp->peers[i].data_fd);
        }
        free(tp->peers);
        tp->peers = NULL;
    }
    tp->~pulsar_tp();
    free(tp);
}

/* Return the peer with the given rank, or NULL. */
static pulsar_tp_peer *tp_peer_by_rank(pulsar_tp *tp, int rank) {
    for (int i = 0; i < tp->n_peers; i++)
        if (tp->peers[i].rank == rank) return &tp->peers[i];
    return NULL;
}

static int tp_hello_exchange(pulsar_tp *tp, int fd, int expected_rank,
                             const pulsar_tp_identity *id, int rdma_ok,
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
        .rank = (uint32_t)tp->rank,
        .n_ranks = (uint32_t)tp->n_ranks,
    };
    pulsar_tp_hello_fixed theirs;
    if (!tp_write_full(fd, &mine, sizeof(mine)) ||
        !tp_read_full(fd, &theirs, sizeof(theirs))) {
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
    /* Role-differ applies to the legacy leader/worker pair only; an n-way mesh
     * has symmetric ranks (no leader/worker), so it is keyed by rank. */
    if (!tp->opt.peers && theirs.role == mine.role) {
        tp_set_err(err, errlen, "tp hello: both sides claim role %u", mine.role);
        return 0;
    }
    if (theirs.rank == mine.rank) {
        tp_set_err(err, errlen, "tp hello: peer claims this rank %u", mine.rank);
        return 0;
    }
    if (expected_rank >= 0 && (int)theirs.rank != expected_rank) {
        tp_set_err(err, errlen, "tp hello: peer rank %u != expected %d", theirs.rank,
                   expected_rank);
        return 0;
    }
    if (theirs.n_ranks != mine.n_ranks) {
        tp_set_err(err, errlen, "tp hello: group size mismatch (peer n_ranks=%u != %u)",
                   theirs.n_ranks, mine.n_ranks);
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

/* The NODE frame (protocol v13): who each rank is, for operators.  Sent once per
 * peer on the control socket at the very end of bring-up -- after the hello,
 * after the RDMA device is open -- so the device named is the one actually in
 * use.  Fixed-size strings, NUL-terminated on receipt whatever the peer sent. */
#define PULSAR_TP_NODE_MAGIC UINT32_C(0x4E4F4445)   /* "NODE" */
typedef struct {
    uint32_t magic;
    uint32_t rank;
    char host[PULSAR_TP_NODE_STR];
    char build[PULSAR_TP_NODE_STR];
    char addr[PULSAR_TP_NODE_STR];
    char rdma_device[PULSAR_TP_NODE_STR];
    uint32_t rdma_port;
    uint32_t pad;
} pulsar_tp_node_wire;

static void tp_fill_self(pulsar_tp *tp, const char *addr) {
    pulsar_tp_node *n = &tp->self;
    memset(n, 0, sizeof(*n));
    n->rank = tp->rank;
    if (gethostname(n->host, sizeof(n->host)) != 0) n->host[0] = '\0';
    n->host[sizeof(n->host) - 1] = '\0';
    snprintf(n->build, sizeof(n->build), "%s", tp->opt.build ? tp->opt.build : "");
    snprintf(n->addr, sizeof(n->addr), "%s", addr ? addr : "");
    if (tp->rdma_active && tp->rdma.ctx) {
        snprintf(n->rdma_device, sizeof(n->rdma_device), "%s", tp->rdma.dev_name);
        n->rdma_port = 1;   /* tp_rdma_open always queries and uses port 1 */
    }
}

static void tp_terminate(char *s, size_t n) { s[n - 1] = '\0'; }

static int tp_node_exchange(pulsar_tp *tp, pulsar_tp_peer *pp, char *err, size_t errlen) {
    pulsar_tp_node_wire mine;
    memset(&mine, 0, sizeof(mine));
    mine.magic = PULSAR_TP_NODE_MAGIC;
    mine.rank = (uint32_t)tp->self.rank;
    memcpy(mine.host, tp->self.host, sizeof(mine.host));
    memcpy(mine.build, tp->self.build, sizeof(mine.build));
    memcpy(mine.addr, tp->self.addr, sizeof(mine.addr));
    memcpy(mine.rdma_device, tp->self.rdma_device, sizeof(mine.rdma_device));
    mine.rdma_port = (uint32_t)tp->self.rdma_port;
    pulsar_tp_node_wire theirs;
    if (!tp_write_full(pp->control_fd, &mine, sizeof(mine)) ||
        !tp_read_full(pp->control_fd, &theirs, sizeof(theirs))) {
        tp_set_err(err, errlen, "tp: node exchange with peer %d failed", pp->rank);
        return 0;
    }
    if (theirs.magic != PULSAR_TP_NODE_MAGIC || (int)theirs.rank != pp->rank) {
        tp_set_err(err, errlen, "tp: bad node frame from peer %d (magic %08x rank %u)",
                   pp->rank, theirs.magic, theirs.rank);
        return 0;
    }
    tp_terminate(theirs.host, sizeof(theirs.host));
    tp_terminate(theirs.build, sizeof(theirs.build));
    tp_terminate(theirs.addr, sizeof(theirs.addr));
    tp_terminate(theirs.rdma_device, sizeof(theirs.rdma_device));
    pulsar_tp_node *n = &pp->node;
    n->rank = pp->rank;
    memcpy(n->host, theirs.host, sizeof(n->host));
    memcpy(n->build, theirs.build, sizeof(n->build));
    memcpy(n->addr, theirs.addr, sizeof(n->addr));
    memcpy(n->rdma_device, theirs.rdma_device, sizeof(n->rdma_device));
    n->rdma_port = (int)theirs.rdma_port;
    pp->node_known = true;
    return 1;
}

/* "ip:port" of this end of `fd`, or "" -- the pair path's only record of its
 * own endpoint (the mesh has the authoritative peers list instead). */
static void tp_local_addr(int fd, int port, char *out, size_t outlen) {
    out[0] = '\0';
    struct sockaddr_storage ss;
    socklen_t len = sizeof(ss);
    if (fd < 0 || getsockname(fd, (struct sockaddr *)&ss, &len) != 0) return;
    char ip[INET6_ADDRSTRLEN] = "";
    if (getnameinfo((struct sockaddr *)&ss, len, ip, sizeof(ip), NULL, 0, NI_NUMERICHOST) != 0)
        return;
    if (port > 0) snprintf(out, outlen, "%s:%d", ip, port);
    else snprintf(out, outlen, "%s", ip);
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
    /* Legacy 2-rank (no peers list) always derives rank from role; the explicit
     * rank/n_ranks are honored only in n-way mode, where peers is set. */
    if (opt->peers) {
        tp->rank = opt->rank >= 0 ? opt->rank : 0;
        tp->n_ranks = opt->n_ranks > 1 ? opt->n_ranks : 2;
    } else {
        tp->rank = opt->role == PULSAR_TP_ROLE_LEADER ? 0 : 1;
        tp->n_ranks = 2;
    }
    tp->n_peers = 0;
    tp->peers = NULL;
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
        listener = tp_listen(opt->peer, opt->port, 2, err, errlen);
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

    if (!tp_hello_exchange(tp, tp->control_fd, 1 - tp->rank, id, rdma_ok, err, errlen))
        goto fail;

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
    /* Register the single peer in the mesh array so the n-way primitives (and
     * the n=2 specialization of the all-reduce) can iterate uniformly. */
    tp->n_peers = 1;
    tp->peers = (pulsar_tp_peer *)calloc(1, sizeof(pulsar_tp_peer));
    if (!tp->peers) {
        tp_set_err(err, errlen, "tp: out of memory (peer array)");
        goto fail;
    }
    tp->peers[0].rank = 1 - tp->rank;
    tp->peers[0].control_fd = tp->control_fd;
    tp->peers[0].data_fd = tp->data_fd;
    tp->peers[0].peer_ctx = tp->peer_ctx;
    tp->peers[0].connected = true;
    {
        /* The leader knows its listen port; a dialing worker has none of its own. */
        char addr[PULSAR_TP_NODE_STR];
        tp_local_addr(tp->control_fd, tp->rank == 0 ? opt->port : 0, addr, sizeof(addr));
        tp_fill_self(tp, addr);
    }
    if (!tp_node_exchange(tp, &tp->peers[0], err, errlen)) goto fail;
    fprintf(stderr, "pulsar-tp: rank %d connected (n_ranks=%d, %d peer), transport=%s\n",
            tp->rank, tp->n_ranks, tp->n_peers,
            tp->rdma_active ? "rdma" : "tcp");
    *out = tp;
    return 1;
fail:
    if (listener >= 0) close(listener);
    tp_destroy(tp);
    return 0;
}

/* n-way full-mesh rendezvous.  Each rank R is told its own index, the group
 * size N, its own listen port, and the ordered "host:port" list of every rank.
 * Rank-ordered connect: lower index dials, higher index accepts.  Every rank
 * dials its R lower peers and accepts its N-1-R higher peers; the connect
 * label (pulsar_tp_connect_hdr) lets an acceptor attribute a raw accepted fd
 * to the right peer and socket kind.  Backlog >= N-1 so no dialer is made to
 * wait for the acceptor (the anti-deadlock property of the mesh). */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t rank;   /* dialer's rank */
    uint32_t kind;   /* 0 = control socket, 1 = data socket */
} pulsar_tp_connect_hdr;

typedef struct { char host[256]; int port; } tp_endpoint;

static int tp_parse_endpoints(const char *list, int n, tp_endpoint *ep,
                              char *err, size_t errlen) {
    if (!list || n <= 0) {
        tp_set_err(err, errlen, "tp: empty peers list");
        return 0;
    }
    int count = 1;
    for (const char *s = list; *s; s++) if (*s == ',') count++;
    if (count != n) {
        tp_set_err(err, errlen, "tp: peers list has %d entries, want %d", count, n);
        return 0;
    }
    char buf[4096];
    snprintf(buf, sizeof(buf), "%s", list);
    char *tok = strtok(buf, ",");
    for (int i = 0; tok && i < n; i++, tok = strtok(NULL, ",")) {
        char *colon = strrchr(tok, ':');
        if (!colon) {
            tp_set_err(err, errlen, "tp: peer %d '%s' has no :port", i, tok);
            return 0;
        }
        *colon = '\0';
        snprintf(ep[i].host, sizeof(ep[i].host), "%s", tok);
        ep[i].port = atoi(colon + 1);
    }
    return 1;
}

int pulsar_tp_create_mesh(pulsar_tp **out, const pulsar_tp_options *opt,
                          const pulsar_tp_identity *id, char *err, size_t errlen) {
    const int N = opt->n_ranks, R = opt->rank;
    if (N < 2 || R < 0 || R >= N) {
        tp_set_err(err, errlen, "tp: mesh needs n_ranks>=2 and 0<=rank<n_ranks (rank=%d n=%d)", R, N);
        return 0;
    }
    tp_endpoint *ep = (tp_endpoint *)calloc((size_t)N, sizeof(tp_endpoint));
    if (!ep || !tp_parse_endpoints(opt->peers, N, ep, err, errlen)) {
        free(ep);
        return 0;
    }
    /* ONE authority for this rank's listen port: its own entry in the peers
     * list (that is what every other rank dials).  A separate --tp-port that
     * disagrees is a configuration error, not a second opinion. */
    if (opt->port > 0 && opt->port != ep[R].port) {
        tp_set_err(err, errlen,
                   "tp: --tp-port %d disagrees with this rank's peers entry %s:%d; "
                   "the peers list is the authority",
                   opt->port, ep[R].host, ep[R].port);
        free(ep);
        return 0;
    }
    pulsar_tp *tp = tp_alloc();
    if (!tp) { free(ep); tp_set_err(err, errlen, "tp: out of memory"); return 0; }
    tp->opt = *opt;
    tp->rank = R;
    tp->n_ranks = N;
    tp->n_peers = N - 1;
    tp->peers = (pulsar_tp_peer *)calloc((size_t)(N - 1), sizeof(pulsar_tp_peer));
    tp->control_fd = -1;
    tp->data_fd = -1;
    tp->timeout_sec = PULSAR_TP_DEFAULT_TIMEOUT_SEC;
    const char *tmo = getenv("PULSAR_TP_TIMEOUT_SEC");
    if (tmo) tp->timeout_sec = (uint64_t)atoi(tmo);
    if (!tp->peers) {
        free(ep); tp_destroy(tp);
        tp_set_err(err, errlen, "tp: out of memory (peer array)");
        return 0;
    }
    /* The probe is an INPUT to the transport decision, not the decision:
     * tp_hello_exchange assigns `rdma_active = rdma_ok && theirs.rdma_ok`, so
     * both sides must report.  This call used to pass a literal 0, which made
     * that assignment a constant false however capable the hardware was -- the
     * reason an n-rank mesh never rode RoCE.
     *
     * A probe that says yes and an HCA open that then fails is a REFUSAL, not a
     * fall-back to TCP (rule 1): the hello below advertises the probe result to
     * every peer, so a rank that quietly dropped to TCP after a failed open
     * would still be believed to speak RDMA by the whole group -- the pair path
     * (pulsar_tp_create) refuses the same way. */
    const int mesh_rdma_ok = tp_rdma_probe(&tp->rdma.api) != 0;
    tp->rdma_active = mesh_rdma_ok;
    if (tp->rdma_active && !tp_rdma_open(tp, err, errlen)) {
        free(ep);
        tp_destroy(tp);
        return 0;
    }

    int listener = tp_listen(ep[R].host, ep[R].port, N > 1 ? N : 2, err, errlen);
    if (listener < 0) { free(ep); tp_destroy(tp); return 0; }

    int slot = 0;
    for (int p = 0; p < N; p++) {
        if (p == R) continue;
        pulsar_tp_peer *pp = &tp->peers[slot++];
        pp->rank = p;
        pp->control_fd = -1;
        pp->data_fd = -1;
    }

    /* Control sockets: dial lower peers, accept higher peers. */
    for (int p = 0; p < R; p++) {                       /* R dials lower */
        pulsar_tp_peer *pp = tp_peer_by_rank(tp, p);
        pp->control_fd = tp_dial(ep[p].host, ep[p].port, (double)tp->timeout_sec,
                                 err, errlen);
        if (pp->control_fd < 0) goto fail;
        tp_socket_tune(pp->control_fd);
        pulsar_tp_connect_hdr ch = { PULSAR_TP_MAGIC, PULSAR_TP_PROTOCOL_VERSION,
                                     (uint32_t)R, 0u };
        if (!tp_write_full(pp->control_fd, &ch, sizeof(ch))) {
            tp_set_err(err, errlen, "tp: control label write to peer %d failed", p);
            goto fail;
        }
    }
    for (int a = 0; a < (N - 1 - R); a++) {             /* R accepts higher */
        int fd = accept(listener, NULL, NULL);
        if (fd < 0) { tp_set_err(err, errlen, "tp: control accept: %s", strerror(errno)); goto fail; }
        tp_socket_tune(fd);
        pulsar_tp_connect_hdr ch;
        if (!tp_read_full(fd, &ch, sizeof(ch)) || ch.magic != PULSAR_TP_MAGIC ||
            ch.version != PULSAR_TP_PROTOCOL_VERSION || ch.kind != 0u) {
            close(fd);
            tp_set_err(err, errlen, "tp: bad control connect label from a higher peer");
            goto fail;
        }
        pulsar_tp_peer *pp = tp_peer_by_rank(tp, (int)ch.rank);
        if (!pp || pp->control_fd >= 0) {
            close(fd);
            tp_set_err(err, errlen, "tp: control accept from unexpected rank %u", ch.rank);
            goto fail;
        }
        pp->control_fd = fd;
    }

    /* Hello + identity per peer on its control socket.  The transport decision
     * must be GROUP-consistent: tp_hello_exchange assigns
     * `rdma_active = mesh_rdma_ok && theirs.rdma_ok` for THIS peer, so AND every
     * peer's answer -- a last-write-wins flag would let two ranks disagree about
     * the payload transport and desync on the wire. */
    /* Declared then assigned: initialising here would be crossed by the
     * `goto fail` below, which C++ rejects. */
    int all_rdma;
    all_rdma = mesh_rdma_ok;
    for (int i = 0; i < tp->n_peers; i++) {
        pulsar_tp_peer *pp = &tp->peers[i];
        if (pp->control_fd < 0) {
            tp_set_err(err, errlen, "tp: peer %d control socket not established", pp->rank);
            goto fail;
        }
        if (!tp_hello_exchange(tp, pp->control_fd, pp->rank, id, mesh_rdma_ok, err, errlen))
            goto fail;
        all_rdma = all_rdma && tp->rdma_active;
        pp->peer_ctx = (uint32_t)tp->peer_ctx;
    }
    tp->rdma_active = all_rdma;

    /* Data sockets: dial lower peers, accept higher peers. */
    for (int p = 0; p < R; p++) {
        pulsar_tp_peer *pp = tp_peer_by_rank(tp, p);
        pp->data_fd = tp_dial(ep[p].host, ep[p].port, (double)tp->timeout_sec,
                              err, errlen);
        if (pp->data_fd < 0) goto fail;
        tp_socket_tune(pp->data_fd);
        pulsar_tp_connect_hdr ch = { PULSAR_TP_MAGIC, PULSAR_TP_PROTOCOL_VERSION,
                                     (uint32_t)R, 1u };
        if (!tp_write_full(pp->data_fd, &ch, sizeof(ch))) {
            tp_set_err(err, errlen, "tp: data label write to peer %d failed", p);
            goto fail;
        }
    }
    for (int a = 0; a < (N - 1 - R); a++) {
        int fd = accept(listener, NULL, NULL);
        if (fd < 0) { tp_set_err(err, errlen, "tp: data accept: %s", strerror(errno)); goto fail; }
        tp_socket_tune(fd);
        pulsar_tp_connect_hdr ch;
        if (!tp_read_full(fd, &ch, sizeof(ch)) || ch.magic != PULSAR_TP_MAGIC ||
            ch.version != PULSAR_TP_PROTOCOL_VERSION || ch.kind != 1u) {
            close(fd);
            tp_set_err(err, errlen, "tp: bad data connect label from a higher peer");
            goto fail;
        }
        pulsar_tp_peer *pp = tp_peer_by_rank(tp, (int)ch.rank);
        if (!pp || pp->data_fd >= 0) {
            close(fd);
            tp_set_err(err, errlen, "tp: data accept from unexpected rank %u", ch.rank);
            goto fail;
        }
        pp->data_fd = fd;
    }

    /* Primary link (for the pairwise/guarded paths) = lowest-rank peer. */
    tp->control_fd = tp->peers[0].control_fd;
    tp->data_fd = tp->peers[0].data_fd;
    tp->peer_ctx = tp->peers[0].peer_ctx;

    {
        char addr[PULSAR_TP_NODE_STR];
        /* Bounded: the record is for display, and a 255-byte peers entry must
         * not be able to push the port out of it. */
        snprintf(addr, sizeof(addr), "%.48s:%d", ep[R].host, ep[R].port);
        tp_fill_self(tp, addr);
    }
    for (int i = 0; i < tp->n_peers; i++)
        if (!tp_node_exchange(tp, &tp->peers[i], err, errlen)) goto fail;

    close(listener);
    free(ep);
    fprintf(stderr, "pulsar-tp: rank %d/%d mesh connected (%d peers), transport=%s\n",
            R, N, tp->n_peers, tp->rdma_active ? "rdma" : "tcp");
    *out = tp;
    return 1;
fail:
    free(ep);
    if (listener >= 0) close(listener);
    tp_destroy(tp);
    return 0;
}

int pulsar_tp_attach_slab(pulsar_tp *tp, void *base, char *err, size_t errlen) {
    tp->slab = static_cast<uint8_t *>(base);
    memset(tp->slab + tp->layout.in_flags_off, 0, (uint64_t)tp->n_slots * 8);
    memset(tp->slab + tp->layout.token_off, 0, 16);
    if (tp->rdma_active) {
        /* Register the slab ONCE against the HCA, then bring up one QP per peer
         * -- the pair has one, the mesh n_ranks-1, and every peer sees the same
         * slab address/rkey. */
        if (!tp_rdma_register_slab(tp, err, errlen)) return 0;
        if (tp->bulk) {
            tp->rdma.bulk_mr = tp->rdma.api.reg_mr(tp->rdma.pd, tp->bulk, (size_t)tp->bulk_bytes,
                                                   TP_IBV_ACCESS_LOCAL_WRITE | TP_IBV_ACCESS_REMOTE_WRITE);
            if (!tp->rdma.bulk_mr) {
                tp_set_err(err, errlen, "tp rdma: reg_mr(bulk, %llu bytes): %s",
                           (unsigned long long)tp->bulk_bytes, strerror(errno));
                return 0;
            }
        }
        for (int i = 0; i < tp->n_peers; i++)
            if (!tp_rdma_link_bringup(tp, &tp->peers[i], err, errlen)) return 0;
        if (pulsar_tp_row_lane(tp) && !tp_row_proxy_start(tp, err, errlen)) return 0;
        return 1;
    }
    (void)err; (void)errlen;
    return 1;
}

void pulsar_tp_free(pulsar_tp *tp) {
    tp_destroy(tp);
}

int pulsar_tp_rank(const pulsar_tp *tp) { return tp->rank; }
uint32_t pulsar_tp_n_ranks(const pulsar_tp *tp) { return tp->n_ranks; }

int pulsar_tp_node_info(const pulsar_tp *tp, int rank, pulsar_tp_node *out) {
    if (!tp || !out) return 0;
    if (rank == tp->rank) { *out = tp->self; return 1; }
    for (int i = 0; i < tp->n_peers; i++)
        if (tp->peers[i].rank == rank && tp->peers[i].node_known) {
            *out = tp->peers[i].node;
            return 1;
        }
    return 0;
}

int pulsar_tp_owned_range(int rank, uint32_t n_ranks, uint32_t n_total,
                                 uint32_t *lo, uint32_t *hi) {
    if (!lo || !hi || rank < 0 || (uint32_t)rank >= (n_ranks ? n_ranks : 1u))
        return 0;
    if (n_ranks <= 1 || n_total == 0) { *lo = 0; *hi = n_total; return 1; }
    *lo = (uint32_t)(((uint64_t)rank    * n_total) / n_ranks);
    *hi = (uint32_t)(((uint64_t)(rank + 1) * n_total) / n_ranks);
    return 1;
}

bool pulsar_tp_is_rdma(const pulsar_tp *tp) { return tp->rdma_active; }
uint32_t pulsar_tp_peer_ctx(const pulsar_tp *tp) { return tp->peer_ctx; }
uint32_t pulsar_tp_n_layer(const pulsar_tp *tp) { return tp->n_layer; }
uint64_t pulsar_tp_vec_bytes(const pulsar_tp *tp) { return tp->vec_bytes; }

/* The per-layer batch regions, through the layout arithmetic's ONE authority.
 * A layer >= n_layer would address past the reservation, so it refuses. */
void *pulsar_tp_slab_batch_out(const pulsar_tp *tp, uint32_t layer) {
    if (!tp || !tp->slab || layer >= tp->n_layer) return NULL;
    return tp->slab + pulsar_tp_slab_batch_out_offset(&tp->layout, layer, tp->vec_bytes);
}
void *pulsar_tp_slab_batch_in(const pulsar_tp *tp, uint32_t layer) {
    if (!tp || !tp->slab || layer >= tp->n_layer) return NULL;
    return tp->slab + pulsar_tp_slab_batch_in_offset(&tp->layout, layer, tp->vec_bytes);
}
bool pulsar_tp_row_lane(const pulsar_tp *tp) {
    return tp && tp->n_ranks == 2 && tp->rdma_active && tp->slab &&
           tp->vec_bytes > 0 && tp->vec_bytes <= PULSAR_TP_RDMA_MAX_MSG &&
           PULSAR_TP_RDMA_RECV_WINDOW + PULSAR_TP_BATCH_MAX_ROWS <= tp->n_slots;
}

static int tp_row_lane_arm(pulsar_tp *tp, uint64_t first);

void pulsar_tp_set_bulk(pulsar_tp *tp, void *base, uint64_t bytes) {
    if (!tp) return;
    tp->bulk = static_cast<uint8_t *>(base);
    tp->bulk_bytes = base ? bytes : 0;
    tp->bulk_cap = base ? (bytes / 3u) / tp->vec_bytes * tp->vec_bytes : 0;
}

bool pulsar_tp_bulk_lane(const pulsar_tp *tp) {
    if (!pulsar_tp_row_lane(tp) || !tp->bulk || tp->bulk_cap == 0 || !tp->rdma.bulk_mr ||
        tp->peer_bulk_base == 0 || !tp->proxy_started)
        return false;
    const pulsar_tp_rdma_link *r = tp_pair_link(const_cast<pulsar_tp *>(tp));
    return r && r->bulk_qp;
}

void pulsar_tp_bulk_layout(const pulsar_tp *tp, pulsar_tp_bulk_layout_t *out) {
    memset(out, 0, sizeof(*out));
    if (!tp) return;
    out->cap_bytes = tp->bulk_cap;
    out->out_off = 0;
    out->in_off[0] = tp->bulk_cap;
    out->in_off[1] = 2u * tp->bulk_cap;
}

int pulsar_tp_bulk_begin(pulsar_tp *tp, uint64_t bytes, uint64_t *exch, uint32_t *buf) {
    if (!pulsar_tp_bulk_lane(tp) || bytes == 0 || bytes > tp->bulk_cap || !exch || !buf) {
        fprintf(stderr, "pulsar-tp: bulk lane refused (%llu bytes, cap %llu; pair+rdma+bulk buffer "
                        "required) -- the caller must pick the lane\n",
                (unsigned long long)bytes, (unsigned long long)(tp ? tp->bulk_cap : 0));
        return 0;
    }
    if (tp->proxy_failed.load(std::memory_order_acquire) || *tp_row_err(tp)) {
        fprintf(stderr, "pulsar-tp: bulk lane refused: an earlier exchange failed "
                        "(proxy %s, device spin %s)\n",
                tp->proxy_failed.load() ? "failed" : "ok",
                *tp_row_err(tp) == 2u ? "aborted" : *tp_row_err(tp) ? "timed out" : "ok");
        return 0;
    }
    static int said = 0;
    if (!said) { said = 1; fprintf(stderr, "pulsar-tp: bulk lane armed (async: GPU stage/publish/combine + "
                                   "RDMA writes on a second QP, %llu MiB per exchange)\n",
                                   (unsigned long long)(tp->bulk_cap >> 20)); }
    *exch = ++tp->row_exch;
    *buf = (uint32_t)(tp->bulk_seq++ & 1u);
    return 1;
}

void pulsar_tp_row_lane_layout(const pulsar_tp *tp, pulsar_tp_row_lane_layout_t *out) {
    memset(out, 0, sizeof(*out));
    if (!tp) return;
    out->out_off = tp->layout.out_off;
    out->in_off = tp->layout.in_off;
    out->desc_off = tp->layout.out_flags_off;
    out->done_off = tp->layout.in_flags_off;
    out->err_off = tp->layout.gpu_flags_off;
    out->vec_bytes = tp->vec_bytes;
    out->n_slots = tp->n_slots;
    out->timeout_ns = tp->timeout_sec * 1000000000ull;
}


int pulsar_tp_row_lane_begin(pulsar_tp *tp, uint32_t rows, uint64_t *first_msg, uint64_t *exch) {
    if (!pulsar_tp_row_lane(tp) || !tp->proxy_started || rows == 0 ||
        rows > PULSAR_TP_BATCH_MAX_ROWS || !first_msg || !exch) {
        fprintf(stderr, "pulsar-tp: row lane refused (%u rows; pair+rdma+slab+proxy required, "
                        "rows 1..%u) -- the caller must pick the lane\n", rows, PULSAR_TP_BATCH_MAX_ROWS);
        return 0;
    }
    if (tp->proxy_failed.load(std::memory_order_acquire) || *tp_row_err(tp)) {
        fprintf(stderr, "pulsar-tp: row lane refused: an earlier exchange failed "
                        "(proxy %s, device spin %s)\n",
                tp->proxy_failed.load() ? "failed" : "ok", *tp_row_err(tp) == 2u ? "aborted" : *tp_row_err(tp) ? "timed out" : "ok");
        return 0;
    }
    static int said = 0;
    if (!said) { said = 1; fprintf(stderr, "pulsar-tp: row lane armed (async: GPU stage/publish/combine "
                                   "+ verbs proxy thread; rdma window %u, <= %u rows per exchange)\n",
                                   (unsigned)PULSAR_TP_RDMA_RECV_WINDOW, PULSAR_TP_BATCH_MAX_ROWS); }
    pulsar_tp_rdma_link *r = tp_pair_link(tp);
    if (!r->recv_window_active) {
        /* The window is down before the first row exchange, or after a big
         * gate drained it (transports with no bulk lane, after a stream sync).
         * Arming posts receives on the GATE QP, so only a gate-QP exchange may
         * not be in flight; bulk exchanges (v14) ride their own QP and may
         * still be running -- a prefill's last bulk exchanges usually are when
         * its head's vocab gather arms the lane. */
        if (tp_row_done(tp)->load(std::memory_order_acquire) < tp->last_gate_exch) {
            fprintf(stderr, "pulsar-tp: row lane: re-arm with gate exchange %llu still in flight "
                            "(done %llu) -- refusing\n", (unsigned long long)tp->last_gate_exch,
                    (unsigned long long)tp_row_done(tp)->load());
            return 0;
        }
        if (!tp_row_lane_arm(tp, tp->gate_seq + 1u)) return 0;
    }
    *first_msg = tp->gate_seq + 1u;
    tp->gate_seq += rows;
    *exch = ++tp->row_exch;
    tp->last_gate_exch = *exch;
    return 1;
}

int pulsar_tp_row_lane_check(pulsar_tp *tp) {
    if (!tp || !tp->proxy_started) return 1;
    const uint64_t done = tp_row_done(tp)->load(std::memory_order_acquire);
    const uint32_t err = *tp_row_err(tp);
    if (tp->proxy_failed.load(std::memory_order_acquire) || err || done != tp->row_exch) {
        fprintf(stderr, "pulsar-tp: row lane check FAILED at a drained stream: exchange %llu "
                        "enqueued, %llu completed, proxy %s, device spin %s -- refusing\n",
                (unsigned long long)tp->row_exch, (unsigned long long)done,
                tp->proxy_failed.load() ? "failed" : "ok", err == 2u ? "aborted" : err ? "timed out" : "ok");
        tp->failed.store(true, std::memory_order_release);
        return 0;
    }
    return 1;
}

/* The RDMA big gate's DIRECT rule -- see pulsar_tp.h.  Boundary matches the
 * arithmetic it replaced exactly: a pointer AT slab_hi is "inside" only for a
 * zero-length payload. */
bool pulsar_tp_in_slab(const pulsar_tp *tp, const void *ptr, uint64_t bytes) {
    if (!tp || !tp->slab || !ptr) return false;
    const uintptr_t lo = (uintptr_t)tp->slab;
    const uintptr_t hi = lo + tp->slab_bytes;
    const uintptr_t p = (uintptr_t)ptr;
    return p >= lo && p <= hi && bytes <= (uint64_t)(hi - p);
}
bool pulsar_tp_failed(const pulsar_tp *tp) {
    return tp && tp->failed.load(std::memory_order_acquire);
}
void pulsar_tp_mark_failed(pulsar_tp *tp) {
    if (tp) tp->failed.store(true, std::memory_order_release);
}

/* ------------------------------------------------------------------------
 * Gate exchange.
 * --------------------------------------------------------------------- */

/* One pairwise slab-slot gate over `fd`: send our out-slot, receive the
 * peer's payload into the in-slot.  Both ranks run it symmetrically (a 16 KB
 * write-then-read cannot deadlock), so the n-way loop below calls it once per
 * peer unchanged. */
static int tp_gate_exchange_fd(int fd, uint32_t layer, uint32_t gate, uint64_t seq,
                               uint8_t *slab, uint64_t out_off, uint64_t in_off,
                               uint64_t vec_bytes) {
    /* TCP: both sides write their partial then read the peer's.  16KB per
     * direction fits comfortably in the socket buffers, so the symmetric
     * write-then-read cannot deadlock.  Header and payload go out in one
     * writev so NODELAY does not split them into two segments. */
    pulsar_tp_gate_header h = { PULSAR_TP_MAGIC, (uint16_t)layer, (uint16_t)gate, seq };
    struct iovec iov[2] = {
        { &h, sizeof(h) },
        { slab + out_off, vec_bytes },
    };
    size_t want = sizeof(h) + vec_bytes;
    /* sendmsg(MSG_NOSIGNAL) not writev(): if the peer dies mid-exchange the
     * survivor must get a gate failure (0), not be killed by SIGPIPE --
     * SO_NOSIGPIPE is BSD-only and a no-op on Linux, so an unprotected writev
     * to a closed socket terminates the rank silently (tp_fault_test). */
    struct msghdr mh = {};
    mh.msg_iov = iov;
    mh.msg_iovlen = 2;
#ifdef MSG_NOSIGNAL
    ssize_t w = sendmsg(fd, &mh, MSG_NOSIGNAL);
#else
    ssize_t w = writev(fd, iov, 2);
#endif
    if (w < 0 || (size_t)w != want) {
        /* Short writev: finish with the plain path. */
        if (w < 0) return 0;
        size_t done = (size_t)w;
        if (done < sizeof(h)) {
            if (!tp_write_full(fd, reinterpret_cast<char *>(&h) + done,
                               sizeof(h) - done)) return 0;
            done = sizeof(h);
        }
        uint64_t payload_done = done - sizeof(h);
        if (!tp_write_full(fd,
                           slab + out_off + payload_done,
                           vec_bytes - payload_done))
            return 0;
    }
    pulsar_tp_gate_header ph;
    if (!tp_read_full(fd, &ph, sizeof(ph))) return 0;
    if (ph.magic != PULSAR_TP_MAGIC || ph.layer != layer || ph.gate != gate || ph.seq != seq) {
        fprintf(stderr,
                "pulsar-tp: gate desync: got l=%u g=%u seq=%llu, want l=%u g=%u seq=%llu\n",
                ph.layer, ph.gate, (unsigned long long)ph.seq,
                layer, gate, (unsigned long long)seq);
        return 0;
    }
    if (!tp_read_full(fd, slab + in_off, vec_bytes))
        return 0;
    return 1;
}

int pulsar_tp_gate_exchange(pulsar_tp *tp, uint32_t layer, uint32_t gate, uint64_t seq) {
    const uint64_t out_off =
        pulsar_tp_slab_out_offset(&tp->layout, layer, gate, tp->vec_bytes);
    const uint64_t in_off =
        pulsar_tp_slab_in_offset(&tp->layout, layer, gate, tp->vec_bytes);
    if (tp->n_ranks <= 2) {
        /* The pair is UNCHANGED: RDMA when both can, else the fd swap. */
        if (tp->rdma_active) return tp_rdma_gate_exchange(tp, layer, gate, seq);
        return tp_gate_exchange_fd(tp->data_fd, layer, gate, seq, tp->slab,
                                   out_off, in_off, tp->vec_bytes);
    }
    /* n>2: ONE pairwise exchange per peer, accumulating every peer's partial
     * into the in-slot.  That preserves the PAIR'S CONTRACT -- `out` holds the
     * local partial, `in` holds what the caller adds -- generalised from "the
     * peer's partial" to "the SUM of every peer's": no caller changes shape, and
     * a one-peer loop would be bit-identical to the pair.  TCP here, because the
     * per-peer RDMA gate exchange is not implemented and the engine does not
     * drive this lane yet; announced once so the transport is never a guess. */
    static int said_nway_gate = 0;
    if (!said_nway_gate) {
        said_nway_gate = 1;
        fprintf(stderr, "pulsar-tp: n-way per-layer gate over TCP (n=%d); per-peer "
                        "RDMA gate exchange is not implemented\n", tp->n_ranks);
    }
    const uint64_t nelt = tp->vec_bytes / sizeof(float);
    float *acc = (float *)calloc((size_t)nelt, sizeof(float));
    if (!acc) return 0;
    for (int i = 0; i < tp->n_peers; i++) {
        pulsar_tp_peer *pp = &tp->peers[i];
        if (!pp || pp->data_fd < 0 ||
            !tp_gate_exchange_fd(pp->data_fd, layer, gate, seq, tp->slab,
                                 out_off, in_off, tp->vec_bytes)) {
            free(acc);
            return 0;
        }
        const float *src = (const float *)(tp->slab + in_off);
        for (uint64_t q = 0; q < nelt; q++) acc[q] += src[q];
    }
    memcpy(tp->slab + in_off, acc, tp->vec_bytes);
    free(acc);
    return 1;
}

/* Verify-block batch gate: one exchange per layer moving all block rows at
 * once.  The payload lives in the registered slab, so RDMA sends it directly;
 * TCP remains the symmetric write-then-read fallback. */
/* The batch gate's fd exchange factored out: header (BATCH_MAGIC, rows tag),
 * then alternating write/read rounds.  See the clamp note below. */
static int tp_batch_gate_exchange_fd(int fd, uint32_t layer, uint32_t rows,
                                     uint64_t seq, uint8_t *slab,
                                     uint64_t batch_out, uint64_t batch_in,
                                     uint64_t vec_bytes) {
    const uint64_t bytes = (uint64_t)rows * vec_bytes;
    pulsar_tp_gate_header h = { PULSAR_TP_BATCH_MAGIC, (uint16_t)layer,
                                (uint16_t)rows, seq };
    /* TCP fallback: header first, then alternate small write/read rounds.  A
     * single 2 MiB writev round deadlocked on the pair hosts: the kernel clamps
     * SO_SNDBUF to net.core.wmem_max (~212K there), so both sides filled their
     * send buffers before either side drained (recv-queue ~457K stuck both
     * ways).  Rounds of PULSAR_TP_TCP_ROUND are safe under any sane clamp. */
    if (!tp_write_full(fd, &h, sizeof(h))) return 0;
    pulsar_tp_gate_header ph;
    if (!tp_read_full(fd, &ph, sizeof(ph))) return 0;
    if (ph.magic != PULSAR_TP_BATCH_MAGIC || ph.layer != layer ||
        ph.gate != rows || ph.seq != seq) {
        fprintf(stderr,
                "pulsar-tp: batch gate desync: got l=%u rows=%u seq=%llu, "
                "want l=%u rows=%u seq=%llu\n",
                ph.layer, ph.gate, (unsigned long long)ph.seq,
                layer, rows, (unsigned long long)seq);
        return 0;
    }
    uint64_t off = 0;
    while (off < bytes) {
        const uint64_t n = bytes - off > PULSAR_TP_TCP_ROUND ?
                           PULSAR_TP_TCP_ROUND : bytes - off;
        if (!tp_write_full(fd, slab + batch_out + off, n)) return 0;
        if (!tp_read_full(fd, slab + batch_in + off, n)) return 0;
        off += n;
    }
    return 1;
}

int pulsar_tp_batch_gate_exchange(pulsar_tp *tp, uint32_t layer, uint32_t rows,
                                  uint64_t seq) {
    if (tp->data_fd < 0 || rows == 0 || rows > PULSAR_TP_BATCH_MAX_ROWS) return 0;
    const uint64_t bytes = (uint64_t)rows * tp->vec_bytes;
    const uint64_t batch_out =
        pulsar_tp_slab_batch_out_offset(&tp->layout, layer, tp->vec_bytes);
    const uint64_t batch_in =
        pulsar_tp_slab_batch_in_offset(&tp->layout, layer, tp->vec_bytes);
    if (tp->n_ranks <= 2) {
        /* The pair is UNCHANGED. */
        pulsar_tp_gate_header h = { PULSAR_TP_BATCH_MAGIC, (uint16_t)layer,
                                    (uint16_t)rows, seq };
        if (tp->rdma_active && tp_rdma_big_gate_capable(tp, tp_pair_link(tp))) {
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
            return tp_rdma_big_gate_exchange(tp, tp_pair_link(tp),
                                             tp->slab + batch_out,
                                             tp->slab + batch_in, bytes);
        }
        return tp_batch_gate_exchange_fd(tp->data_fd, layer, rows, seq, tp->slab,
                                         batch_out, batch_in, tp->vec_bytes);
    }
    /* n>2: same contract as the per-layer gate -- every peer's rows accumulate
     * into the IN region while the OUT region keeps this rank's rows.  TCP: the
     * per-peer RDMA batch path is not implemented and this lane is not driven by
     * the engine yet.  Announced once. */
    static int said_nway_batch = 0;
    if (!said_nway_batch) {
        said_nway_batch = 1;
        fprintf(stderr, "pulsar-tp: n-way verify-batch gate over TCP (n=%d); per-peer "
                        "RDMA batch gate is not implemented\n", tp->n_ranks);
    }
    const uint64_t nelt = bytes / sizeof(float);
    float *acc = (float *)calloc((size_t)nelt, sizeof(float));
    if (!acc) return 0;
    for (int i = 0; i < tp->n_peers; i++) {
        pulsar_tp_peer *pp = &tp->peers[i];
        if (!pp || pp->data_fd < 0 ||
            !tp_batch_gate_exchange_fd(pp->data_fd, layer, rows, seq, tp->slab,
                                       batch_out, batch_in, tp->vec_bytes)) {
            free(acc);
            return 0;
        }
        const float *src = (const float *)(tp->slab + batch_in);
        for (uint64_t q = 0; q < nelt; q++) acc[q] += src[q];
    }
    memcpy(tp->slab + batch_in, acc, bytes);
    free(acc);
    return 1;
}

/* Prefill batch gate: RDMA uses the pipelined registered-slab path above.
 * The TCP fallback alternates small write/read rounds so neither side can fill
 * its send buffer while the peer is also only writing, under ANY kernel
 * socket-buffer clamp (the former 2 MiB rounds deadlocked on the pair hosts
 * where net.core.wmem_max ~212K; see pulsar_tp_batch_gate_exchange). */

/* Defined below with the n-way all-reduce, which also loops peers with it. */
static int tp_big_gate_exchange_peer(pulsar_tp *tp, pulsar_tp_peer *pp,
                                     uint32_t layer, uint64_t seq,
                                     const void *out, void *in, uint64_t bytes);

int pulsar_tp_big_gate_exchange(pulsar_tp *tp, uint32_t layer, uint64_t seq,
                                const void *out, void *in, uint64_t bytes) {
    if (!tp || !out || !in || bytes == 0) return 0;
    if (tp->n_ranks > 2) {
        /* n>2, same contract as the slab gates: `out` keeps this rank's partial
         * and `in` ACCUMULATES THE SUM of every peer's, so a caller that adds the
         * in-buffer gets the full sum -- the pair's "in is the peer's partial"
         * read at n ranks.  One per-peer exchange each, over that peer's own RDMA
         * link when the group decided RDMA (tp_big_gate_exchange_peer), else its
         * data socket. */
        const uint64_t nelt = bytes / sizeof(float);
        float *acc = (float *)calloc((size_t)nelt, sizeof(float));
        if (!acc) return 0;
        for (int i = 0; i < tp->n_peers; i++) {
            if (!tp_big_gate_exchange_peer(tp, &tp->peers[i], layer, seq,
                                           out, in, bytes)) {
                free(acc);
                return 0;
            }
            const float *src = (const float *)in;
            for (uint64_t q = 0; q < nelt; q++) acc[q] += src[q];
        }
        memcpy(in, acc, bytes);
        free(acc);
        return 1;
    }
    if (tp->data_fd < 0) return 0;
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
    if (tp->rdma_active && tp_rdma_big_gate_capable(tp, tp_pair_link(tp))) {
        if (!tp_rdma_drain_decode_window(tp)) return 0;
        return tp_rdma_big_gate_exchange(tp, tp_pair_link(tp), out, in, bytes);
    }
    uint64_t off = 0;
    while (off < bytes) {
        const uint64_t n = bytes - off > PULSAR_TP_TCP_ROUND ?
                           PULSAR_TP_TCP_ROUND : bytes - off;
        if (!tp_write_full(tp->data_fd, static_cast<const char *>(out) + off, n)) return 0;
        if (!tp_read_full(tp->data_fd, static_cast<char *>(in) + off, n)) return 0;
        off += n;
    }
    return 1;
}

/* Per-peer TCP big-gate exchange over `fd` (the mesh all-reduce drives this on
 * each peer's data socket, in peer-rank order so both ends of a link meet). */
static int tp_big_gate_exchange_fd(int fd, uint32_t layer,
                                   uint64_t seq, const void *out, void *in,
                                   uint64_t bytes) {
    if (fd < 0 || !out || !in || bytes == 0) return 0;
    pulsar_tp_gate_header h = { PULSAR_TP_BATCH_MAGIC, (uint16_t)layer, 0xB16u, seq };
    if (!tp_write_full(fd, &h, sizeof(h))) return 0;
    pulsar_tp_gate_header ph;
    if (!tp_read_full(fd, &ph, sizeof(ph))) return 0;
    if (ph.magic != PULSAR_TP_BATCH_MAGIC || ph.layer != layer ||
        ph.gate != 0xB16u || ph.seq != seq) {
        fprintf(stderr,
                "pulsar-tp: big gate desync: got l=%u tag=%x seq=%llu, want l=%u seq=%llu\n",
                ph.layer, ph.gate, (unsigned long long)ph.seq,
                layer, (unsigned long long)seq);
        return 0;
    }
    uint64_t off = 0;
    while (off < bytes) {
        const uint64_t n = bytes - off > PULSAR_TP_TCP_ROUND ?
                           PULSAR_TP_TCP_ROUND : bytes - off;
        if (!tp_write_full(fd, static_cast<const char *>(out) + off, n)) return 0;
        if (!tp_read_full(fd, static_cast<char *>(in) + off, n)) return 0;
        off += n;
    }
    return 1;
}

/* All-gather + local sum across the whole mesh.  `out` starts as this rank's
 * owned partial and returns the combined sum of every rank's partial.  n==2 is
 * byte-identical to the old pairwise exchange + add (the engine's tp_prefill
 * big gate).  n>2 accumulates in canonical ascending-rank order so every rank
 * ends up with the bit-identical full sum (float addition is not associative). */
/* n-general per-peer gate payload: header over THIS peer's data socket, then the
 * payload over its OWN QP when the group decided RDMA and this peer's link can
 * carry it.  The pair's pulsar_tp_big_gate_exchange is the same shape on the
 * primary link; this is the entry the mesh's peer loops call. */
static int tp_big_gate_exchange_peer(pulsar_tp *tp, pulsar_tp_peer *pp,
                                     uint32_t layer, uint64_t seq,
                                     const void *out, void *in, uint64_t bytes) {
    if (!pp || pp->data_fd < 0 || !out || !in || bytes == 0) return 0;
    /* One branch for the WHOLE group, so both ends of a link agree: the decision
     * is group-consistent by construction (see create_mesh's all_rdma). */
    if (!(tp->rdma_active && tp_rdma_big_gate_capable(tp, &pp->rdma)))
        return tp_big_gate_exchange_fd(pp->data_fd, layer, seq, out, in, bytes);
    pulsar_tp_gate_header h = { PULSAR_TP_BATCH_MAGIC, (uint16_t)layer, 0xB16u, seq };
    if (!tp_write_full(pp->data_fd, &h, sizeof(h))) return 0;
    pulsar_tp_gate_header ph;
    if (!tp_read_full(pp->data_fd, &ph, sizeof(ph))) return 0;
    if (ph.magic != PULSAR_TP_BATCH_MAGIC || ph.layer != layer ||
        ph.gate != 0xB16u || ph.seq != seq) {
        fprintf(stderr, "pulsar-tp: peer big gate desync with rank %d: got l=%u tag=%x "
                        "seq=%llu, want l=%u seq=%llu\n",
                pp->rank, ph.layer, ph.gate, (unsigned long long)ph.seq,
                layer, (unsigned long long)seq);
        return 0;
    }
    /* A per-peer decode window would need its own drain; the mesh data plane
     * never arms one, and refusing beats silently leaving receives queued. */
    if (pp->rdma.recv_window_active) {
        fprintf(stderr, "pulsar-tp: rank %d has an armed decode window and per-peer "
                        "drain is not implemented -- refusing\n", pp->rank);
        return 0;
    }
    return tp_rdma_big_gate_exchange(tp, &pp->rdma, out, in, bytes);
}

int pulsar_tp_allreduce_sum(pulsar_tp *tp, uint32_t layer, uint64_t seq,
                            void *out, const void *in, uint64_t bytes) {
    if (!tp || !out || !in || bytes == 0) return 0;
    /* Rule 5: announce the lane once per shape (this runs per layer). */
    static int said = 0;
    if (!said) {
        said = 1;
        fprintf(stderr, "pulsar-tp: n-way all-reduce n=%d lane=%s\n",
                tp->n_ranks, tp->rdma_active ? "rdma" : "tcp");
    }
    const uint64_t nelt = bytes / sizeof(float);
    if (tp->n_ranks <= 1) return 1;         /* single rank: own partial is the sum */
    if (tp->n_ranks == 2) {
        /* The RDMA-capable entry, NOT the raw fd.  It takes the verbs path when
         * RDMA is up -- and the DIRECT branch when out/in already lie inside the
         * registered slab, which is exactly what the engine's
         * <=PULSAR_TP_BATCH_MAX_ROWS staging is for -- and otherwise falls back
         * to the same chunked fd exchange.  Calling tp_big_gate_exchange_fd here
         * left every pair gate exchange on TCP even with a live 200G RoCE link,
         * which the plan calls load-bearing. */
        /* `in` is caller SCRATCH here (the peer's payload lands in it), so the
         * const in this function's signature is dropped -- as it was when this
         * leg called tp_big_gate_exchange_fd directly. */
        if (!pulsar_tp_big_gate_exchange(tp, layer, seq, out, (void *)in, bytes)) return 0;
        float *acc = (float *)out;
        const float *peer = (const float *)in;
        for (uint64_t i = 0; i < nelt; i++) acc[i] += peer[i];
        return 1;
    }
    /* n>2: canonical ascending-rank all-gather + sum.  `out` (own partial) is
     * sent unchanged to every peer; acc sums P0+P1+.. in rank order so every
     * rank bits the same result. */
    float *acc = (float *)calloc(bytes ? (size_t)bytes : sizeof(float), 1);
    if (!acc) return 0;
    for (int k = 0; k < tp->n_ranks; k++) {
        if (k == tp->rank) {
            const float *own = (const float *)out;
            for (uint64_t i = 0; i < nelt; i++) acc[i] += own[i];
        } else {
            pulsar_tp_peer *pp = tp_peer_by_rank(tp, k);
            float *scratch = (float *)in;
            if (!pp || !tp_big_gate_exchange_peer(tp, pp, layer, seq, out,
                                                  scratch, bytes)) {
                free(acc);
                return 0;
            }
            for (uint64_t i = 0; i < nelt; i++) acc[i] += scratch[i];
        }
    }
    memcpy(out, acc, bytes);
    free(acc);
    return 1;
}

/* n-way ROW all-gather (slice 4d vocab, 4g attention groups) -- see
 * pulsar_tp.h for the contract.  Concatenation, NOT the sum above: a mistaken
 * allreduce here would multiply every rank's logits by the group size.
 *
 * Loop shape: every rank walks k = 0..n-1 in ascending rank order, so the pair
 * (r,k) reaches its exchange in the same round on both sides and the existing
 * symmetric write-then-read exchange cannot deadlock.  The stride is padded to
 * ceil(n_units/n_ranks) * unit so both directions of every exchange carry the
 * same byte count even when the partition is uneven. */
int pulsar_tp_allgather_rows(pulsar_tp *tp, uint32_t layer, uint64_t seq,
                             float *full_out, const float *own_slice,
                             float *scratch, uint32_t n_rows,
                             uint32_t n_units, uint32_t unit) {
    if (!tp || !full_out || !own_slice || !scratch || n_rows == 0 || n_units == 0 || unit == 0) return 0;
    const uint32_t n_ranks = (uint32_t)(tp->n_ranks > 0 ? tp->n_ranks : 1);
    const uint32_t n_total = n_units * unit;
    const uint32_t stride = ((n_units + n_ranks - 1u) / n_ranks) * unit;   /* padded slice */
    for (uint32_t k = 0; k < n_ranks; k++) {
        uint32_t ulo = 0, uhi = 0;
        if (!pulsar_tp_owned_range((int)k, n_ranks, n_units, &ulo, &uhi)) return 0;
        const uint32_t lo = ulo * unit;
        const uint64_t elems = (uint64_t)(uhi - ulo) * unit;
        const float *src = own_slice;
        if (k != (uint32_t)tp->rank) {
            pulsar_tp_peer *pp = tp_peer_by_rank(tp, (int)k);
            if (!pp || pp->data_fd < 0) {
                fprintf(stderr, "pulsar-tp: row all-gather has no channel to rank %u\n", k);
                return 0;
            }
            /* Same rule as allreduce_sum: the pair rides the RDMA-capable entry
             * (which stages through the registered slab itself, and rides DIRECT
             * when the caller's buffers are already in it); the n>2 mesh keeps
             * the plain fd, per-peer RDMA being pair-gated. */
            const bool xok = tp->n_ranks == 2
                ? pulsar_tp_big_gate_exchange(tp, layer, seq, own_slice, scratch,
                                              (uint64_t)n_rows * stride * sizeof(float)) != 0
                : tp_big_gate_exchange_peer(tp, pp, layer, seq, own_slice, scratch,
                                            (uint64_t)n_rows * stride * sizeof(float)) != 0;
            if (!xok) return 0;
            src = scratch;
        }
        /* Place this rank's range into every row.  The own-rank case copies from
         * own_slice, so an n_ranks==1 group still ends with a complete block. */
        for (uint32_t r = 0; r < n_rows; r++) {
            memcpy(full_out + (uint64_t)r * n_total + lo,
                   src + (uint64_t)r * stride,
                   elems * sizeof(float));
        }
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * Lockstep control plane.
 * --------------------------------------------------------------------- */

typedef struct {
    uint64_t session_id;
    uint32_t count;
    int32_t value;        /* REWRITE_FROM_COMMON: the common-prefix length; 0 elsewhere */
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
    uint32_t head_runs;   /* the mixed step's max_head_runs (the caller's head policy); 0 on EVAL_BATCH */
} pulsar_tp_batch_command_header;

typedef struct {
    uint64_t session_id;
    int32_t status;
    uint32_t flags;      /* PULSAR_TP_ACK_HAS_DIGEST: `digest` is the sender's assembled logits (v11) */
    uint64_t digest;
} pulsar_tp_command_ack;

/* Broadcast ONE frame to every peer's control socket.  The command plane is
 * leader -> workers, so an n-rank group must reach all n_ranks-1 of them; for
 * the pair this is exactly the old single send, because control_fd IS
 * peers[0].  Worker -> leader frames (ack, logits half, verify) do NOT come
 * through here: a worker has one peer, the leader, so its control_fd is already
 * that link. */
static int tp_send_frame_to_peers(pulsar_tp *tp, uint32_t type,
                                  const void *payload, uint32_t bytes) {
    if (!tp || tp->n_peers < 1) return 0;
    for (int i = 0; i < tp->n_peers; i++) {
        if (tp->peers[i].control_fd < 0) return 0;
        if (!tp_send_frame(tp->peers[i].control_fd, type, payload, bytes)) return 0;
    }
    return 1;
}

static int tp_send_token_command(pulsar_tp *tp, uint32_t type,
                                 uint64_t session_id, const int *tokens,
                                 uint32_t count, int32_t value = 0) {
    const uint64_t bytes64 = sizeof(pulsar_tp_token_command_header) +
                             (uint64_t)count * sizeof(int32_t);
    if (!tp || (!tokens && count != 0) || bytes64 > UINT32_MAX) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = static_cast<uint8_t *>(malloc(bytes ? bytes : 1u));
    if (!payload) return 0;
    pulsar_tp_token_command_header h = { session_id, count, value };
    memcpy(payload, &h, sizeof(h));
    int32_t *wire_tokens = reinterpret_cast<int32_t *>(payload + sizeof(h));
    for (uint32_t i = 0; i < count; i++) wire_tokens[i] = (int32_t)tokens[i];
    const int ok = tp_send_frame_to_peers(tp, type, payload, bytes);
    free(payload);
    return ok;
}

int pulsar_tp_send_session_create(pulsar_tp *tp, uint64_t session_id, int ctx_size,
                                  uint32_t n_banks) {
    if (n_banks == 0) {
        fprintf(stderr, "pulsar-tp: session create with a zero bank pool -- refusing to ship it\n");
        return 0;
    }
    pulsar_tp_value_command msg = { session_id, (int32_t)ctx_size, n_banks };
    return tp_send_frame_to_peers(tp, PULSAR_TP_FRAME_SESSION_CREATE,
                         &msg, sizeof(msg));
}

int pulsar_tp_send_session_destroy(pulsar_tp *tp, uint64_t session_id) {
    return tp_send_frame_to_peers(tp, PULSAR_TP_FRAME_SESSION_DESTROY,
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
    return tp_send_frame_to_peers(tp, PULSAR_TP_FRAME_EVAL, &msg, sizeof(msg));
}

int pulsar_tp_send_chunk_verdict(pulsar_tp *tp, uint64_t session_id, int stop) {
    pulsar_tp_value_command msg = { session_id, stop ? 1 : 0, 0 };
    return tp_send_frame_to_peers(tp, PULSAR_TP_FRAME_CHUNK_VERDICT, &msg, sizeof(msg));
}

typedef struct {
    uint64_t session_id;
    int32_t pos;
    uint32_t reserved;
    uint64_t digest;
} pulsar_tp_sync_check_command;

int pulsar_tp_send_sync_check(pulsar_tp *tp, uint64_t session_id, int pos, uint64_t digest) {
    pulsar_tp_sync_check_command msg = { session_id, (int32_t)pos, 0u, digest };
    return tp_send_frame_to_peers(tp, PULSAR_TP_FRAME_SYNC_CHECK, &msg, sizeof(msg));
}

int pulsar_tp_recv_chunk_verdict(pulsar_tp *tp, uint64_t session_id, int *stop,
                                 char *err, size_t errlen) {
    if (!tp || !stop) return 0;
    const double deadline = tp_control_deadline(tp);
    uint32_t type = 0, bytes = 0;
    pulsar_tp_value_command msg;
    if ((deadline > 0.0 && !tp_wait_readable(tp->control_fd, deadline)) ||
        !tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != PULSAR_TP_FRAME_CHUNK_VERDICT || bytes != sizeof(msg) ||
        !tp_read_full(tp->control_fd, &msg, sizeof(msg)) || msg.session_id != session_id) {
        pulsar_tp_mark_failed(tp);
        tp_set_err(err, errlen, "tp: expected the leader's chunk verdict for session %llu inside a "
                                "mirrored prefill (frame type %u) -- the ranks' prefills are out of step",
                   (unsigned long long)session_id, type);
        return 0;
    }
    *stop = msg.value != 0;
    return 1;
}

int pulsar_tp_send_rewind(pulsar_tp *tp, uint64_t session_id, int pos) {
    pulsar_tp_value_command msg = { session_id, (int32_t)pos, 0 };
    return tp_send_frame_to_peers(tp, PULSAR_TP_FRAME_REWIND,
                         &msg, sizeof(msg));
}

int pulsar_tp_send_invalidate(pulsar_tp *tp, uint64_t session_id) {
    return tp_send_frame_to_peers(tp, PULSAR_TP_FRAME_INVALIDATE,
                         &session_id, sizeof(session_id));
}

/** One row-batch frame, two operations.  The frame TYPE is what tells the
 * worker which engine operation the leader is in (decode_multiseq vs
 * decode_mixed); without it a driver that diverged between the two would
 * decode the same rows through a different contract and say nothing. */
static int tp_send_batch(pulsar_tp *tp, uint32_t frame_type,
                         const pulsar_tp_batch_item *items, uint32_t count,
                         uint32_t head_runs) {
    const uint64_t bytes64 = sizeof(pulsar_tp_batch_command_header) +
                             (uint64_t)count * sizeof(*items);
    if (!tp || !items || count == 0 || bytes64 > UINT32_MAX) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = static_cast<uint8_t *>(malloc(bytes));
    if (!payload) return 0;
    pulsar_tp_batch_command_header h = { count, head_runs };
    memcpy(payload, &h, sizeof(h));
    memcpy(payload + sizeof(h), items, (size_t)count * sizeof(*items));
    const int ok = tp_send_frame_to_peers(tp, frame_type, payload, bytes);
    free(payload);
    return ok;
}

int pulsar_tp_send_eval_batch(pulsar_tp *tp, const pulsar_tp_batch_item *items,
                              uint32_t count) {
    return tp_send_batch(tp, PULSAR_TP_FRAME_EVAL_BATCH, items, count, 0u);
}

int pulsar_tp_send_mixed_batch(pulsar_tp *tp,
                               const pulsar_tp_batch_item *items,
                               uint32_t count, uint32_t max_head_runs) {
    return tp_send_batch(tp, PULSAR_TP_FRAME_MIXED_BATCH, items, count, max_head_runs);
}


static int tp_send_bank_value(pulsar_tp *tp, uint32_t type, uint64_t session_id, uint32_t bank) {
    pulsar_tp_value_command msg = { session_id, (int32_t)bank, 0 };
    return tp_send_frame_to_peers(tp, type, &msg, sizeof(msg));
}
int pulsar_tp_send_bank_state_save(pulsar_tp *tp, uint64_t session_id, uint32_t bank) {
    return tp_send_bank_value(tp, PULSAR_TP_FRAME_BANK_STATE_SAVE, session_id, bank);
}
int pulsar_tp_send_bank_state_restore(pulsar_tp *tp, uint64_t session_id, uint32_t bank) {
    return tp_send_bank_value(tp, PULSAR_TP_FRAME_BANK_STATE_RESTORE, session_id, bank);
}
int pulsar_tp_send_bank_repoint(pulsar_tp *tp, uint64_t session_id, uint32_t bank) {
    return tp_send_bank_value(tp, PULSAR_TP_FRAME_BANK_REPOINT, session_id, bank);
}

typedef struct {
    uint64_t session_id;
    int32_t src;
    int32_t dst;
    int32_t n_cached;
    uint32_t count;      /* request tokens following the header */
} pulsar_tp_fork_command_header;

int pulsar_tp_send_bank_fork(pulsar_tp *tp, int partial, uint64_t session_id,
                             uint32_t src, uint32_t dst,
                             const int *tokens, uint32_t n_tokens, int n_cached) {
    const uint64_t bytes64 = sizeof(pulsar_tp_fork_command_header) + (uint64_t)n_tokens * sizeof(int32_t);
    if (!tp || (n_tokens && !tokens) || bytes64 > UINT32_MAX) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = static_cast<uint8_t *>(malloc(bytes));
    if (!payload) return 0;
    pulsar_tp_fork_command_header h = { session_id, (int32_t)src, (int32_t)dst, (int32_t)n_cached, n_tokens };
    memcpy(payload, &h, sizeof(h));
    int32_t *wire = reinterpret_cast<int32_t *>(payload + sizeof(h));
    for (uint32_t i = 0; i < n_tokens; i++) wire[i] = (int32_t)tokens[i];
    const int ok = tp_send_frame_to_peers(tp,
            partial ? PULSAR_TP_FRAME_BANK_FORK_PARTIAL : PULSAR_TP_FRAME_BANK_FORK, payload, bytes);
    free(payload);
    return ok;
}

/* The verdict collector behind pulsar_tp_wait_command_status (want_digest 0)
 * and pulsar_tp_wait_command_status_digest (want_digest 1). */
static int tp_settle_deferred(pulsar_tp *tp, char *err, size_t errlen);

static int tp_collect_status_body(pulsar_tp *tp, uint64_t session_id, const char *operation,
                                  int *status, int want_digest, uint64_t own_digest,
                                  char *err, size_t errlen);

/* Every ack read settles the deferred one first: the peers answer in frame
 * order, so its ack is the next one in the socket.  A failed settle fails
 * this collect too (its own acks are still read, so no frame is left behind),
 * and its message is the one reported. */
static int tp_collect_status(pulsar_tp *tp, uint64_t session_id, const char *operation,
                             int *status, int want_digest, uint64_t own_digest,
                             char *err, size_t errlen) {
    char scratch[256];
    const int settled = tp_settle_deferred(tp, err, errlen);
    const int rc = tp_collect_status_body(tp, session_id, operation, status, want_digest, own_digest,
                                          settled ? err : scratch, settled ? errlen : sizeof(scratch));
    return settled && rc;
}

static int tp_collect_status_body(pulsar_tp *tp, uint64_t session_id, const char *operation,
                                  int *status, int want_digest, uint64_t own_digest,
                                  char *err, size_t errlen) {
    if (!tp || tp->n_peers < 1 || !status) return 0;
    const double deadline = tp_control_deadline(tp);
    int agreed = 0, have = 0, bad = 0;
    for (int i = 0; i < tp->n_peers; i++) {
        const int pfd = tp->peers[i].control_fd;
        uint32_t type = 0, bytes = 0;
        pulsar_tp_command_ack ack;
        if (pfd < 0 || (deadline > 0.0 && !tp_wait_readable(pfd, deadline))) {
            pulsar_tp_mark_failed(tp);
            tp_set_err(err, errlen,
                       "tp: rank %d did not answer %s within %llu s -- the ranks are not in "
                       "lockstep or the peer is wedged",
                       tp->peers[i].rank, operation ? operation : "the command",
                       (unsigned long long)tp->timeout_sec);
            return 0;
        }
        if (!tp_read_frame_header(pfd, &type, &bytes) ||
            type != PULSAR_TP_FRAME_COMMAND_ACK || bytes != sizeof(ack) ||
            !tp_read_full(pfd, &ack, sizeof(ack))) {
            pulsar_tp_mark_failed(tp);
            tp_set_err(err, errlen, "tp: rank %d failed during %s",
                       tp->peers[i].rank, operation ? operation : "command");
            return 0;
        }
        /* Every peer's ack is read even after a refusal or a disagreement, so
         * no ack is left in a socket to shift the next frame. */
        if (bad) continue;
        const int has_digest = (ack.flags & PULSAR_TP_ACK_HAS_DIGEST) != 0;
        if (ack.session_id != session_id || ack.status < 0) {
            bad = 1;
            tp_set_err(err, errlen, "tp: rank %d refused %s (session %llu, status %d)",
                       tp->peers[i].rank, operation ? operation : "command",
                       (unsigned long long)ack.session_id, (int)ack.status);
        } else if (has_digest != (want_digest && ack.status > 0)) {
            /* A digest rides a POSITIVE verdict of a logits-producing verdict
             * operation and nothing else; any other shape means the peer
             * answered a different command than the leader sent. */
            bad = 1;
            pulsar_tp_mark_failed(tp);
            tp_set_err(err, errlen,
                       "tp: rank %d answered %s (status %d) with %s logits digest -- the ranks "
                       "are not running the same operation",
                       tp->peers[i].rank, operation ? operation : "the command", (int)ack.status,
                       has_digest ? "an unexpected" : "no");
        } else if (have && ack.status != agreed) {
            bad = 1;
            tp_set_err(err, errlen, "tp: %s verdict SPLIT: an earlier rank said %d, rank %d says %d",
                       operation ? operation : "command", agreed, tp->peers[i].rank, (int)ack.status);
        } else {
            agreed = ack.status;
            have = 1;
            if (has_digest) {
                tp->identity_frames++;
                if (ack.digest == own_digest) {
                    tp->identity_matched++;
                } else {
                    bad = 1;
                    pulsar_tp_mark_failed(tp);
                    tp_set_err(err, errlen,
                               "tp: rank %d's logits differ from the leader's on %s (digest "
                               "%016llx vs %016llx) -- the ranks did not assemble the same vector; "
                               "refusing, the group is marked failed",
                               tp->peers[i].rank, operation ? operation : "the command",
                               (unsigned long long)ack.digest, (unsigned long long)own_digest);
                }
            }
        }
    }
    if (bad) return 0;
    *status = agreed;
    return 1;
}

int pulsar_tp_wait_command_status(pulsar_tp *tp, uint64_t session_id,
                                  const char *operation, int *status,
                                  char *err, size_t errlen) {
    return tp_collect_status(tp, session_id, operation, status, 0, 0ull, err, errlen);
}

int pulsar_tp_wait_command_status_digest(pulsar_tp *tp, uint64_t session_id,
                                         const char *operation, int *status,
                                         uint64_t own_digest, char *err, size_t errlen) {
    return tp_collect_status(tp, session_id, operation, status, 1, own_digest, err, errlen);
}

int pulsar_tp_send_rewrite_from_common(pulsar_tp *tp, uint64_t session_id,
                                       const int *tokens, uint32_t n_tokens, int common) {
    return tp_send_token_command(tp, PULSAR_TP_FRAME_REWRITE_FROM_COMMON, session_id,
                                 tokens, n_tokens, (int32_t)common);
}
int pulsar_tp_send_note_committed(pulsar_tp *tp, uint64_t session_id,
                                  const int *tokens, uint32_t n_tokens) {
    return tp_send_token_command(tp, PULSAR_TP_FRAME_NOTE_COMMITTED, session_id, tokens, n_tokens);
}

typedef struct {
    uint64_t session_id;
    uint32_t count;
    uint32_t reserved;
} pulsar_tp_logits_command_header;

int pulsar_tp_send_set_logits(pulsar_tp *tp, uint64_t session_id,
                              const float *logits, uint32_t n) {
    const uint64_t bytes64 = sizeof(pulsar_tp_logits_command_header) + (uint64_t)n * sizeof(float);
    if (!tp || !logits || n == 0 || bytes64 > UINT32_MAX) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = static_cast<uint8_t *>(malloc(bytes));
    if (!payload) return 0;
    pulsar_tp_logits_command_header h = { session_id, n, 0 };
    memcpy(payload, &h, sizeof(h));
    memcpy(payload + sizeof(h), logits, (size_t)n * sizeof(float));
    const int ok = tp_send_frame_to_peers(tp, PULSAR_TP_FRAME_SET_LOGITS, payload, bytes);
    free(payload);
    return ok;
}

int pulsar_tp_send_spec(pulsar_tp *tp, uint32_t frame_type, const pulsar_tp_spec_command *cmd,
                        const uint32_t *banks, const uint64_t *rngs) {
    if (!tp || !cmd) return 0;
    const uint32_t n = frame_type == PULSAR_TP_FRAME_SPEC_REDRAFT_BATCH ? cmd->count : 0u;
    if (n && (!banks || !rngs)) return 0;
    const uint64_t bytes64 = sizeof(*cmd) + (uint64_t)n * (sizeof(uint32_t) + sizeof(uint64_t));
    if (bytes64 > UINT32_MAX) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = static_cast<uint8_t *>(malloc(bytes));
    if (!payload) return 0;
    memcpy(payload, cmd, sizeof(*cmd));
    if (n) {
        memcpy(payload + sizeof(*cmd), banks, (size_t)n * sizeof(uint32_t));
        memcpy(payload + sizeof(*cmd) + (size_t)n * sizeof(uint32_t), rngs, (size_t)n * sizeof(uint64_t));
    }
    const int ok = tp_send_frame_to_peers(tp, frame_type, payload, bytes);
    free(payload);
    return ok;
}

int pulsar_tp_send_bank_free_physical(pulsar_tp *tp, uint64_t session_id, uint32_t bank) {
    return tp_send_bank_value(tp, PULSAR_TP_FRAME_BANK_FREE_PHYSICAL, session_id, bank);
}
int pulsar_tp_send_bank_alloc_physical(pulsar_tp *tp, uint64_t session_id, uint32_t bank) {
    return tp_send_bank_value(tp, PULSAR_TP_FRAME_BANK_ALLOC_PHYSICAL, session_id, bank);
}

typedef struct {
    uint64_t session_id;
    int32_t bank;
    uint32_t key_len;    /* bytes of key following, without a terminator */
} pulsar_tp_spill_command_header;

static int tp_send_keyed(pulsar_tp *tp, uint32_t type, uint64_t session_id, int32_t value,
                         const char *key) {
    if (!tp || !key || !key[0]) return 0;
    const size_t kl = strlen(key);
    if (kl > 4096) return 0;
    const uint32_t bytes = (uint32_t)(sizeof(pulsar_tp_spill_command_header) + kl);
    uint8_t *payload = static_cast<uint8_t *>(malloc(bytes));
    if (!payload) return 0;
    pulsar_tp_spill_command_header h = { session_id, value, (uint32_t)kl };
    memcpy(payload, &h, sizeof(h));
    memcpy(payload + sizeof(h), key, kl);
    const int ok = tp_send_frame_to_peers(tp, type, payload, bytes);
    free(payload);
    return ok;
}

int pulsar_tp_send_bank_kv(pulsar_tp *tp, int load, uint64_t session_id, uint32_t bank, const char *key) {
    return tp_send_keyed(tp, load ? PULSAR_TP_FRAME_BANK_KV_LOAD : PULSAR_TP_FRAME_BANK_KV_SAVE,
                         session_id, (int32_t)bank, key);
}

int pulsar_tp_send_kvstore(pulsar_tp *tp, pulsar_tp_frame_type type, uint64_t session_id,
                           const char *key, int32_t value) {
    if (type != PULSAR_TP_FRAME_KVSTORE_SAVE && type != PULSAR_TP_FRAME_KVSTORE_DROP) return 0;
    return tp_send_keyed(tp, (uint32_t)type, session_id, value, key);
}

typedef struct {
    uint64_t session_id;
    int32_t n_tokens;
    uint32_t key_len;    /* bytes of key following, without a terminator */
    uint64_t digest;
} pulsar_tp_kvload_command_header;

typedef struct {
    uint64_t session_id;   /* 0: not a session operation */
    uint32_t n_keys;
    uint32_t reserved;
} pulsar_tp_kvreconcile_command_header;

int pulsar_tp_send_kvstore_reconcile(pulsar_tp *tp, const char *keys, uint32_t n_keys) {
    if (!tp || (n_keys > 0 && !keys) || n_keys > (1u << 20)) return 0;
    const uint64_t bytes64 = sizeof(pulsar_tp_kvreconcile_command_header) + (uint64_t)n_keys * 40u;
    uint8_t *payload = static_cast<uint8_t *>(malloc((size_t)bytes64));
    if (!payload) return 0;
    pulsar_tp_kvreconcile_command_header h = { 0u, n_keys, 0u };
    memcpy(payload, &h, sizeof(h));
    if (n_keys) memcpy(payload + sizeof(h), keys, (size_t)n_keys * 40u);
    const int ok = tp_send_frame_to_peers(tp, PULSAR_TP_FRAME_KVSTORE_RECONCILE, payload, (uint32_t)bytes64);
    free(payload);
    return ok;
}

static bool tp_kv_hex40(const char *p) {
    for (int i = 0; i < 40; i++) {
        const char c = p[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

bool pulsar_tp_kv_blob_name(const char *name, char key[41]) {
    static const char pre[] = "tp-kv-", suf[] = ".payload";
    const size_t lp = sizeof(pre) - 1u, ls = sizeof(suf) - 1u;
    if (!name || strlen(name) != lp + 40u + ls || strncmp(name, pre, lp) != 0 ||
        strcmp(name + lp + 40u, suf) != 0 || !tp_kv_hex40(name + lp)) return false;
    if (key) { memcpy(key, name + lp, 40u); key[40] = '\0'; }
    return true;
}

/* "tp-kv-<key>.payload.tmp.<pid>" whose writer is gone (the worker's own
 * tmp + rename recipe, interrupted). */
static bool tp_kv_abandoned_tmp(const char *name) {
    const char *t = strstr(name, ".payload.tmp.");
    if (!t) return false;
    char base[128];
    const size_t n = (size_t)(t - name) + strlen(".payload");
    if (n >= sizeof(base)) return false;
    memcpy(base, name, n);
    base[n] = '\0';
    if (!pulsar_tp_kv_blob_name(base, NULL)) return false;
    char *end = NULL;
    const long pid = strtol(t + strlen(".payload.tmp."), &end, 10);
    if (!end || *end != '\0' || pid <= 0) return false;
    return kill((pid_t)pid, 0) != 0 && errno == ESRCH;
}

bool pulsar_tp_kv_reconcile_dir(const char *dir, const char *keys, uint32_t n_keys,
                                int *kept, int *removed, char *err, size_t errlen) {
    if (kept) *kept = 0;
    if (removed) *removed = 0;
    DIR *d = dir ? opendir(dir) : NULL;
    if (!d) {
        tp_set_err(err, errlen, "cannot read %s: %s", dir ? dir : "(null)", strerror(errno));
        return false;
    }
    std::vector<std::string> live;
    live.reserve(n_keys);
    for (uint32_t i = 0; i < n_keys; i++) live.emplace_back(keys + (size_t)i * 40u, 40u);
    std::sort(live.begin(), live.end());
    std::vector<std::string> doomed;
    while (struct dirent *de = readdir(d)) {
        char key[41];
        if (pulsar_tp_kv_blob_name(de->d_name, key)) {
            if (std::binary_search(live.begin(), live.end(), std::string(key, 40u))) {
                if (kept) (*kept)++;
            } else {
                doomed.emplace_back(de->d_name);
            }
        } else if (tp_kv_abandoned_tmp(de->d_name)) {
            doomed.emplace_back(de->d_name);
        }
    }
    closedir(d);
    for (const std::string &name : doomed) {
        const std::string path = std::string(dir) + "/" + name;
        if (unlink(path.c_str()) == 0 && removed) (*removed)++;
    }
    return true;
}

int pulsar_tp_send_kvstore_load(pulsar_tp *tp, uint64_t session_id, const char *key,
                                int n_tokens, uint64_t digest) {
    if (!tp || !key || !key[0]) return 0;
    const size_t kl = strlen(key);
    if (kl > 4096) return 0;
    const uint32_t bytes = (uint32_t)(sizeof(pulsar_tp_kvload_command_header) + kl);
    uint8_t *payload = static_cast<uint8_t *>(malloc(bytes));
    if (!payload) return 0;
    pulsar_tp_kvload_command_header h = { session_id, (int32_t)n_tokens, (uint32_t)kl, digest };
    memcpy(payload, &h, sizeof(h));
    memcpy(payload + sizeof(h), key, kl);
    const int ok = tp_send_frame_to_peers(tp, PULSAR_TP_FRAME_KVSTORE_LOAD, payload, bytes);
    free(payload);
    return ok;
}

typedef struct {
    uint64_t session_id;
    uint32_t n_tokens;
    uint32_t n_images;
} pulsar_tp_sync_mm_header;
typedef struct {
    int32_t  start_pos;
    uint32_t len;
} pulsar_tp_image_entry;

int pulsar_tp_send_sync_mm(pulsar_tp *tp, uint64_t session_id, const int *tokens, uint32_t n_tokens,
                           const pulsar_image_ref *images, uint32_t n_images) {
    if (!tp || (n_tokens && !tokens) || n_images == 0 || !images) return 0;
    uint64_t img_bytes = 0;
    for (uint32_t i = 0; i < n_images; i++) {
        if (!images[i].bytes || images[i].len == 0 || images[i].len > UINT32_MAX) return 0;
        img_bytes += images[i].len;
    }
    const uint64_t bytes64 = sizeof(pulsar_tp_sync_mm_header) + (uint64_t)n_tokens * sizeof(int32_t) +
                             (uint64_t)n_images * sizeof(pulsar_tp_image_entry) + img_bytes;
    if (bytes64 > UINT32_MAX) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = static_cast<uint8_t *>(malloc(bytes));
    if (!payload) return 0;
    uint8_t *p = payload;
    pulsar_tp_sync_mm_header h = { session_id, n_tokens, n_images };
    memcpy(p, &h, sizeof(h)); p += sizeof(h);
    for (uint32_t i = 0; i < n_tokens; i++) { const int32_t t = (int32_t)tokens[i]; memcpy(p, &t, sizeof(t)); p += sizeof(t); }
    for (uint32_t i = 0; i < n_images; i++) {
        pulsar_tp_image_entry e = { (int32_t)images[i].start_pos, (uint32_t)images[i].len };
        memcpy(p, &e, sizeof(e)); p += sizeof(e);
    }
    for (uint32_t i = 0; i < n_images; i++) { memcpy(p, images[i].bytes, images[i].len); p += images[i].len; }
    const int ok = tp_send_frame_to_peers(tp, PULSAR_TP_FRAME_SYNC_MM, payload, bytes);
    free(payload);
    return ok;
}

int pulsar_tp_send_command_ack(pulsar_tp *tp, uint64_t session_id, int status) {
    pulsar_tp_command_ack ack = { session_id, (int32_t)status, 0u, 0ull };
    return tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_COMMAND_ACK,
                         &ack, sizeof(ack));
}

int pulsar_tp_send_command_ack_digest(pulsar_tp *tp, uint64_t session_id, int status,
                                      uint64_t digest) {
    pulsar_tp_command_ack ack = { session_id, (int32_t)status, PULSAR_TP_ACK_HAS_DIGEST, digest };
    return tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_COMMAND_ACK,
                         &ack, sizeof(ack));
}

uint64_t pulsar_tp_logits_digest(const float *logits, uint32_t n_rows, uint32_t width) {
    /* Four independent multiply-xor chains over uint64 words, folded at the
     * end; the row count and width seed the chains so a short or long vector
     * cannot collide with a full one by matching bytes.  Word-wise because a
     * byte-wise hash costs a served batch (tens of MB per step) milliseconds;
     * this one runs near memory speed. */
    const uint64_t K = 0x9E3779B97F4A7C15ull;
    uint64_t h0 = K ^ ((uint64_t)n_rows << 32 | width), h1 = h0 * 3, h2 = h0 * 5, h3 = h0 * 7;
    const size_t n_bytes = (size_t)n_rows * width * sizeof(float);
    const unsigned char *b = (const unsigned char *)logits;
    size_t i = 0;
    for (; i + 32 <= n_bytes; i += 32) {
        uint64_t w0, w1, w2, w3;
        memcpy(&w0, b + i, 8); memcpy(&w1, b + i + 8, 8);
        memcpy(&w2, b + i + 16, 8); memcpy(&w3, b + i + 24, 8);
        h0 = (h0 ^ w0) * K; h1 = (h1 ^ w1) * K; h2 = (h2 ^ w2) * K; h3 = (h3 ^ w3) * K;
    }
    for (; i + 8 <= n_bytes; i += 8) {
        uint64_t w; memcpy(&w, b + i, 8);
        h0 = (h0 ^ w) * K;
    }
    if (i < n_bytes) {
        uint64_t w = 0; memcpy(&w, b + i, n_bytes - i);
        h0 = (h0 ^ w) * K;
    }
    uint64_t h = (h0 ^ (h1 >> 29)) * K ^ (h2 * 3 + (h3 >> 17));
    h ^= h >> 32; h *= K; h ^= h >> 29;
    return h;
}

void pulsar_tp_identity_stats(const pulsar_tp *tp, uint64_t *frames, uint64_t *matched) {
    if (frames) *frames = tp ? tp->identity_frames : 0;
    if (matched) *matched = tp ? tp->identity_matched : 0;
}

/* The one collector behind the three ack waits.  `mode`: PLAIN = one ack per
 * peer, a digest-bearing ack is a protocol confusion and refuses; DIGEST =
 * every peer's ack must carry a digest equal to `own_digest`; DRAIN = read the
 * acks in any shape and ignore their verdicts (the caller's own body already
 * failed). */
enum { TP_ACK_PLAIN = 0, TP_ACK_DIGEST = 1, TP_ACK_DRAIN = 2 };

static int tp_collect_acks_body(pulsar_tp *tp, uint64_t session_id, const char *operation,
                                int mode, uint64_t own_digest, char *err, size_t errlen);

static int tp_collect_acks(pulsar_tp *tp, uint64_t session_id, const char *operation,
                           int mode, uint64_t own_digest, char *err, size_t errlen) {
    char scratch[256];
    const int settled = tp_settle_deferred(tp, err, errlen);
    const int rc = tp_collect_acks_body(tp, session_id, operation, mode, own_digest,
                                        settled ? err : scratch, settled ? errlen : sizeof(scratch));
    return settled && rc;
}

static int tp_settle_deferred(pulsar_tp *tp, char *err, size_t errlen) {
    if (!tp || !tp->deferred_armed) return 1;
    tp->deferred_armed = false;
    return tp_collect_acks_body(tp, tp->deferred_session, tp->deferred_operation,
                                TP_ACK_DIGEST, tp->deferred_digest, err, errlen);
}

int pulsar_tp_defer_command_ack_digest(pulsar_tp *tp, uint64_t session_id,
                                       const char *operation, uint64_t own_digest) {
    if (!tp || tp->deferred_armed) {
        fprintf(stderr, "pulsar-tp: a deferred ack is already pending for %s -- settle it before "
                        "deferring another (refusing)\n", tp ? tp->deferred_operation : "?");
        return 0;
    }
    tp->deferred_armed = true;
    tp->deferred_session = session_id;
    tp->deferred_digest = own_digest;
    snprintf(tp->deferred_operation, sizeof(tp->deferred_operation), "%s", operation ? operation : "the step");
    return 1;
}

int pulsar_tp_settle_deferred_ack(pulsar_tp *tp, char *err, size_t errlen) {
    return tp_settle_deferred(tp, err, errlen);
}

static int tp_collect_acks_body(pulsar_tp *tp, uint64_t session_id, const char *operation,
                                int mode, uint64_t own_digest, char *err, size_t errlen) {
    /* One ack per PEER: every worker must have applied the command, and all
     * must succeed.  The first failure names the rank that refused -- but a
     * refusal (an ack that arrived with a nonzero status, a foreign session
     * id, or a digest that does not match) does NOT stop the collect: the
     * remaining peers' acks are still read, because an ack left in a socket
     * would be consumed by the NEXT operation and shift every later frame on
     * that link by one.  Only a dead link (timeout, closed channel, malformed
     * frame) returns at once, since nothing more can be trusted from it. */
    if (!tp || tp->n_peers < 1) return 0;
    const double deadline = tp_control_deadline(tp);
    int refused = 0;
    for (int i = 0; i < tp->n_peers; i++) {
        const int pfd = tp->peers[i].control_fd;
        uint32_t type = 0, bytes = 0;
        pulsar_tp_command_ack ack;
        if (pfd < 0 || (deadline > 0.0 && !tp_wait_readable(pfd, deadline))) {
            pulsar_tp_mark_failed(tp);
            tp_set_err(err, errlen,
                       "tp: rank %d did not answer %s within %llu s -- the ranks are not in "
                       "lockstep or the peer is wedged",
                       tp->peers[i].rank, operation ? operation : "the command",
                       (unsigned long long)tp->timeout_sec);
            return 0;
        }
        if (!tp_read_frame_header(pfd, &type, &bytes) ||
            type != PULSAR_TP_FRAME_COMMAND_ACK || bytes != sizeof(ack) ||
            !tp_read_full(pfd, &ack, sizeof(ack))) {
            pulsar_tp_mark_failed(tp);
            tp_set_err(err, errlen, "tp: rank %d failed during %s",
                       tp->peers[i].rank, operation ? operation : "command");
            return 0;
        }
        if (mode == TP_ACK_DRAIN) continue;
        const int has_digest = (ack.flags & PULSAR_TP_ACK_HAS_DIGEST) != 0;
        if (ack.session_id != session_id || ack.status != 0) {
            if (!refused) {
                refused = 1;
                tp_set_err(err, errlen,
                           "tp: rank %d %s failed (session %llu, status %d)",
                           tp->peers[i].rank, operation ? operation : "command",
                           (unsigned long long)ack.session_id, (int)ack.status);
            }
            continue;
        }
        if (has_digest != (mode == TP_ACK_DIGEST)) {
            /* The peer answered a different kind of command than the leader
             * sent: the two are not running the same operation. */
            pulsar_tp_mark_failed(tp);
            if (!refused) {
                refused = 1;
                tp_set_err(err, errlen,
                           "tp: rank %d answered %s with %s logits digest -- the ranks are not "
                           "running the same operation",
                           tp->peers[i].rank, operation ? operation : "the command",
                           has_digest ? "an unexpected" : "no");
            }
            continue;
        }
        if (mode == TP_ACK_DIGEST) {
            tp->identity_frames++;
            if (ack.digest == own_digest) {
                tp->identity_matched++;
            } else {
                pulsar_tp_mark_failed(tp);
                if (!refused) {
                    refused = 1;
                    tp_set_err(err, errlen,
                               "tp: rank %d's logits differ from the leader's on %s (digest "
                               "%016llx vs %016llx) -- the ranks did not assemble the same vector; "
                               "refusing, the group is marked failed",
                               tp->peers[i].rank, operation ? operation : "the command",
                               (unsigned long long)ack.digest, (unsigned long long)own_digest);
                }
            }
        }
    }
    return refused ? 0 : 1;
}

int pulsar_tp_wait_command_ack(pulsar_tp *tp, uint64_t session_id,
                               const char *operation, char *err, size_t errlen) {
    return tp_collect_acks(tp, session_id, operation, TP_ACK_PLAIN, 0ull, err, errlen);
}

int pulsar_tp_wait_command_ack_digest(pulsar_tp *tp, uint64_t session_id,
                                      const char *operation, uint64_t own_digest,
                                      char *err, size_t errlen) {
    return tp_collect_acks(tp, session_id, operation, TP_ACK_DIGEST, own_digest, err, errlen);
}

void pulsar_tp_own_step_failed(pulsar_tp *tp, const char *operation) {
    /* A peer that failed the step too answers at once; one still running it is
     * spinning on an exchange this rank will never join -- no step stays silent
     * for seconds between exchanges -- so after a short wait the lane is
     * aborted, which frees the peer's GPU and brings its (failed) ack. */
    for (int i = 0; tp && i < tp->n_peers; i++) {
        const int pfd = tp->peers[i].control_fd;
        if (pfd >= 0 && !tp_wait_readable(pfd, tp_now_sec() + 5.0)) {
            char why[160];
            snprintf(why, sizeof(why), "this rank's %s failed and the peer did not answer within 5 s "
                                       "(it is still running the step)", operation ? operation : "step");
            pulsar_tp_row_lane_abort(tp, why);
        }
    }
}

void pulsar_tp_drain_command_acks(pulsar_tp *tp) {
    pulsar_tp_own_step_failed(tp, "step");
    (void)tp_collect_acks(tp, 0ull, "drain", TP_ACK_DRAIN, 0ull, NULL, 0);
}

int pulsar_tp_send_stop(pulsar_tp *tp) {
    /* Broadcast STOP to every peer so an n-way mesh tears down cleanly (this is
     * on the engine's live destroy path, session.cpp). */
    int ok = 1;
    if (tp->n_peers > 0) {
        for (int i = 0; i < tp->n_peers; i++)
            if (tp->peers[i].control_fd >= 0 &&
                !tp_send_frame(tp->peers[i].control_fd, PULSAR_TP_FRAME_STOP, NULL, 0))
                ok = 0;
    } else if (tp->control_fd >= 0) {
        ok = tp_send_frame_to_peers(tp, PULSAR_TP_FRAME_STOP, NULL, 0);
    }
    return ok;
}

void pulsar_tp_command_free(pulsar_tp_command *command) {
    if (!command) return;
    free(command->tokens);
    free(command->items);
    memset(command, 0, sizeof(*command));
    command->type = PULSAR_TP_FRAME_ERROR;
    free(command->logits);
    command->logits = NULL;
    command->n_logits = 0;
    free(command->spec_banks);
    free(command->spec_rngs);
    command->spec_banks = NULL;
    command->spec_rngs = NULL;
    free(command->spill_key);
    command->spill_key = NULL;
    free(command->images);
    free(command->image_bytes);
    command->images = NULL;
    command->image_bytes = NULL;
    command->n_images = 0;
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
    command->value = h.value;
    return 1;
}

int pulsar_tp_recv_command(pulsar_tp *tp, pulsar_tp_command *command,
                           char *err, size_t errlen) {
    memset(command, 0, sizeof(*command));
    command->type = PULSAR_TP_FRAME_ERROR;
    uint32_t ftype = 0, bytes = 0;
    /* NO deadline: between commands a worker waits as long as its leader is
     * idle -- a server with no traffic is not a failure (Tyler: "they should be
     * able to idle forever"; the 300 s deadline killed an idle pair).  A leader
     * that exits or crashes closes the socket, which wakes this wait and fails
     * the header read below; a peer HOST that vanishes is caught by the
     * socket's keepalive (tp_socket_tune).  Mid-operation silence is still
     * bounded where it can occur: the ack collects, the exchanges and the
     * device spins keep the transport timeout. */
    struct pollfd pfd;
    pfd.fd = tp->control_fd;
    pfd.events = POLLIN;
    int prc;
    do {
        pfd.revents = 0;
        prc = poll(&pfd, 1, -1);
    } while (prc < 0 && errno == EINTR);
    if (prc < 0) {
        pulsar_tp_mark_failed(tp);
        tp_set_err(err, errlen, "tp: waiting for the leader's next command: %s", strerror(errno));
        return 0;
    }
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
    case PULSAR_TP_FRAME_REWRITE_FROM_COMMON:
    case PULSAR_TP_FRAME_NOTE_COMMITTED:
        ok = tp_command_decode_tokens(command, payload, bytes, err, errlen);
        break;
    case PULSAR_TP_FRAME_SPEC_NEXT_BASE:
    case PULSAR_TP_FRAME_SPEC_ROUND_BEGIN:
    case PULSAR_TP_FRAME_SPEC_ARM_CAPTURE:
    case PULSAR_TP_FRAME_SPEC_ROUND_END:
    case PULSAR_TP_FRAME_SPEC_ROUND_ABORT:
    case PULSAR_TP_FRAME_SPEC_REDRAFT_BATCH:
    case PULSAR_TP_FRAME_SPEC_REDRAFT_COMMIT:
    case PULSAR_TP_FRAME_GENERATE_SPECULATIVE: {
        if (bytes < sizeof(command->spec)) { ok = 0; break; }
        memcpy(&command->spec, payload, sizeof(command->spec));
        command->session_id = command->spec.session_id;
        command->value = command->spec.bank;
        const uint32_t n = ftype == PULSAR_TP_FRAME_SPEC_REDRAFT_BATCH ? command->spec.count : 0u;
        const uint64_t want = sizeof(command->spec) + (uint64_t)n * (sizeof(uint32_t) + sizeof(uint64_t));
        if (want != bytes) { ok = 0; break; }
        if (n) {
            command->spec_banks = static_cast<uint32_t *>(malloc((size_t)n * sizeof(uint32_t)));
            command->spec_rngs = static_cast<uint64_t *>(malloc((size_t)n * sizeof(uint64_t)));
            if (!command->spec_banks || !command->spec_rngs) { ok = -1; break; }
            memcpy(command->spec_banks, payload + sizeof(command->spec), (size_t)n * sizeof(uint32_t));
            memcpy(command->spec_rngs, payload + sizeof(command->spec) + (size_t)n * sizeof(uint32_t),
                   (size_t)n * sizeof(uint64_t));
        }
        break;
    }
    case PULSAR_TP_FRAME_SET_LOGITS: {
        pulsar_tp_logits_command_header h;
        if (bytes < sizeof(h)) { ok = 0; break; }
        memcpy(&h, payload, sizeof(h));
        const uint64_t want = sizeof(h) + (uint64_t)h.count * sizeof(float);
        if (h.count == 0 || want != bytes) { ok = 0; break; }
        command->logits = static_cast<float *>(malloc((size_t)h.count * sizeof(float)));
        if (!command->logits) { ok = -1; break; }
        memcpy(command->logits, payload + sizeof(h), (size_t)h.count * sizeof(float));
        command->n_logits = h.count;
        command->session_id = h.session_id;
        break;
    }
    case PULSAR_TP_FRAME_BANK_FORK:
    case PULSAR_TP_FRAME_BANK_FORK_PARTIAL: {
        pulsar_tp_fork_command_header h;
        if (bytes < sizeof(h)) { ok = 0; break; }
        memcpy(&h, payload, sizeof(h));
        const uint64_t want = sizeof(h) + (uint64_t)h.count * sizeof(int32_t);
        if (want != bytes) { ok = 0; break; }
        command->session_id = h.session_id;
        command->bank_src = h.src;
        command->bank_dst = h.dst;
        command->n_cached = h.n_cached;
        if (h.count) {
            command->tokens = static_cast<int *>(malloc((size_t)h.count * sizeof(int)));
            if (!command->tokens) { ok = -1; break; }
            const int32_t *wire = reinterpret_cast<const int32_t *>(payload + sizeof(h));
            for (uint32_t i = 0; i < h.count; i++) command->tokens[i] = wire[i];
        }
        command->n_tokens = h.count;
        break;
    }
    case PULSAR_TP_FRAME_SYNC_MM: {
        pulsar_tp_sync_mm_header h;
        if (bytes < sizeof(h)) { ok = 0; break; }
        memcpy(&h, payload, sizeof(h));
        const uint64_t fixed = sizeof(h) + (uint64_t)h.n_tokens * sizeof(int32_t) +
                               (uint64_t)h.n_images * sizeof(pulsar_tp_image_entry);
        if (h.n_images == 0 || fixed > bytes) { ok = 0; break; }
        const uint8_t *p = payload + sizeof(h);
        command->tokens = static_cast<int *>(malloc(h.n_tokens ? (size_t)h.n_tokens * sizeof(int) : 1u));
        command->images = static_cast<pulsar_image_ref *>(malloc((size_t)h.n_images * sizeof(pulsar_image_ref)));
        if (!command->tokens || !command->images) { ok = -1; break; }
        for (uint32_t i = 0; i < h.n_tokens; i++) { int32_t t; memcpy(&t, p, sizeof(t)); p += sizeof(t); command->tokens[i] = t; }
        uint64_t img_total = 0;
        for (uint32_t i = 0; i < h.n_images; i++) {
            pulsar_tp_image_entry e; memcpy(&e, p, sizeof(e)); p += sizeof(e);
            command->images[i].start_pos = e.start_pos;
            command->images[i].len = e.len;
            command->images[i].bytes = NULL;
            img_total += e.len;
        }
        if (fixed + img_total != bytes || img_total == 0) { ok = 0; break; }
        command->image_bytes = static_cast<uint8_t *>(malloc((size_t)img_total));
        if (!command->image_bytes) { ok = -1; break; }
        memcpy(command->image_bytes, p, (size_t)img_total);
        uint64_t off = 0;
        for (uint32_t i = 0; i < h.n_images; i++) { command->images[i].bytes = command->image_bytes + off; off += command->images[i].len; }
        command->session_id = h.session_id;
        command->n_tokens = h.n_tokens;
        command->n_images = h.n_images;
        break;
    }
    case PULSAR_TP_FRAME_SYNC_CHECK: {
        pulsar_tp_sync_check_command msg;
        if (bytes != sizeof(msg)) { ok = 0; break; }
        memcpy(&msg, payload, sizeof(msg));
        command->session_id = msg.session_id;
        command->value = msg.pos;      /* the leader's cached position */
        command->seq = msg.digest;     /* the leader's cached-prefix digest */
        break;
    }
    case PULSAR_TP_FRAME_KVSTORE_RECONCILE: {
        pulsar_tp_kvreconcile_command_header h;
        if (bytes < sizeof(h)) { ok = 0; break; }
        memcpy(&h, payload, sizeof(h));
        if (h.n_keys > (1u << 20) || sizeof(h) + (uint64_t)h.n_keys * 40u != bytes) { ok = 0; break; }
        command->spill_key = static_cast<char *>(malloc((size_t)h.n_keys * 40u + 1u));
        if (!command->spill_key) { ok = -1; break; }
        memcpy(command->spill_key, payload + sizeof(h), (size_t)h.n_keys * 40u);
        command->spill_key[(size_t)h.n_keys * 40u] = '\0';
        command->session_id = h.session_id;
        command->value = (int)h.n_keys;
        break;
    }
    case PULSAR_TP_FRAME_KVSTORE_LOAD: {
        pulsar_tp_kvload_command_header h;
        if (bytes < sizeof(h)) { ok = 0; break; }
        memcpy(&h, payload, sizeof(h));
        if (h.key_len == 0 || h.key_len > 4096 || sizeof(h) + h.key_len != bytes) { ok = 0; break; }
        command->spill_key = static_cast<char *>(malloc((size_t)h.key_len + 1u));
        if (!command->spill_key) { ok = -1; break; }
        memcpy(command->spill_key, payload + sizeof(h), h.key_len);
        command->spill_key[h.key_len] = '\0';
        command->session_id = h.session_id;
        command->value = h.n_tokens;   /* the leader's checkpoint length after its load */
        command->seq = h.digest;       /* ...and its checkpoint digest */
        break;
    }
    case PULSAR_TP_FRAME_BANK_KV_SAVE:
    case PULSAR_TP_FRAME_BANK_KV_LOAD:
    case PULSAR_TP_FRAME_KVSTORE_SAVE:
    case PULSAR_TP_FRAME_KVSTORE_DROP: {
        pulsar_tp_spill_command_header h;
        if (bytes < sizeof(h)) { ok = 0; break; }
        memcpy(&h, payload, sizeof(h));
        if (h.key_len == 0 || h.key_len > 4096 || sizeof(h) + h.key_len != bytes) { ok = 0; break; }
        command->spill_key = static_cast<char *>(malloc((size_t)h.key_len + 1u));
        if (!command->spill_key) { ok = -1; break; }
        memcpy(command->spill_key, payload + sizeof(h), h.key_len);
        command->spill_key[h.key_len] = '\0';
        command->session_id = h.session_id;
        command->value = h.bank;
        break;
    }
    case PULSAR_TP_FRAME_BANK_FREE_PHYSICAL:
    case PULSAR_TP_FRAME_BANK_ALLOC_PHYSICAL:
    case PULSAR_TP_FRAME_BANK_STATE_SAVE:
    case PULSAR_TP_FRAME_BANK_STATE_RESTORE:
    case PULSAR_TP_FRAME_BANK_REPOINT:
    case PULSAR_TP_FRAME_SESSION_CREATE:
    case PULSAR_TP_FRAME_REWIND: {
        pulsar_tp_value_command msg;
        if (bytes != sizeof(msg)) { ok = 0; break; }
        memcpy(&msg, payload, sizeof(msg));
        command->session_id = msg.session_id;
        command->value = msg.value;
        /* SESSION_CREATE's bank-pool size (v12); the other value frames send 0. */
        command->seq = msg.reserved;
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
    case PULSAR_TP_FRAME_EVAL_BATCH:
    case PULSAR_TP_FRAME_MIXED_BATCH: {
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
        /* The frame's session is the rows' session (every row carries the same
         * one), and the mixed step's head policy rides the header.  Before this
         * was set, a CORRECT batched decode would have been refused as a
         * session divergence: the worker compared session_id 0 to its own. */
        command->session_id = command->items[0].session_id;
        command->value = (int)h.head_runs;
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


int pulsar_tp_send_verify(pulsar_tp *tp, uint64_t session_id,
                          const int *drafts, uint32_t n) {
    return tp_send_token_command(tp, PULSAR_TP_FRAME_VERIFY, session_id,
                                 drafts, n);
}

int pulsar_tp_send_verify_commit(pulsar_tp *tp, int32_t full_accept, int32_t replay_n) {
    /* Worker -> leader: a worker has ONE peer (the leader), so its control_fd is
     * that link at any n -- nothing to loop. */
    struct { int32_t full; int32_t replay; } msg = { full_accept, replay_n };
    return tp_send_frame(tp->control_fd, PULSAR_TP_FRAME_VERIFY_COMMIT,
                         &msg, sizeof(msg));
}

int pulsar_tp_recv_verify_commit(pulsar_tp *tp, int32_t *full_accept, int32_t *replay_n) {
    /* One commit per PEER, and every one must AGREE.  Every rank verifies
     * against the same assembled logits (the vocab split all-gathers them), so a
     * split verdict means the ranks disagree about the same computation -- a
     * real bug, and this REFUSES rather than silently taking one worker's word.
     * n=2 is the one-peer case and behaves exactly as before. */
    if (!tp || tp->n_peers < 1 || !full_accept || !replay_n) return 0;
    int32_t first_full = 0, first_replay = 0;
    for (int i = 0; i < tp->n_peers; i++) {
        const int pfd = tp->peers[i].control_fd;
        uint32_t type = 0, bytes = 0;
        struct { int32_t full; int32_t replay; } msg;
        if (pfd < 0 || !tp_read_frame_header(pfd, &type, &bytes) ||
            type != PULSAR_TP_FRAME_VERIFY_COMMIT || bytes != sizeof(msg) ||
            !tp_read_full(pfd, &msg, sizeof(msg))) {
            fprintf(stderr,
                    "pulsar-tp: bad verify-commit frame from rank %d (type %u bytes %u)\n",
                    tp->peers[i].rank, type, bytes);
            return 0;
        }
        if (i == 0) {
            first_full = msg.full;
            first_replay = msg.replay;
        } else if (msg.full != first_full || msg.replay != first_replay) {
            fprintf(stderr,
                    "pulsar-tp: verify-commit SPLIT: rank %d says (%d,%d), rank %d says "
                    "(%d,%d) -- refusing\n",
                    tp->peers[0].rank, first_full, first_replay,
                    tp->peers[i].rank, msg.full, msg.replay);
            return 0;
        }
    }
    *full_accept = first_full;
    *replay_n = first_replay;
    return 1;
}

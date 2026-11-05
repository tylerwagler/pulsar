/** @file
 *  The fixed-algorithm fp16 / bf16 cuBLASLt GEMM: one algorithm per shape and row class, its caches. */
/* The fixed-algorithm fp16 GEMM (L321) -- see pulsar_cuda_lt_fixed.h for the contract. */
#include "pulsar_cuda_lt_fixed.h"

#include "pulsar.h"       /* PULSAR_PREFILL_CHUNK_DEFAULT: the prompt class's canonical row count */
#include "pulsar_gpu.h"   /* PULSAR_GPU_MNEUTRAL_ROWS_MAX: the decode class's */

#include <cstdio>
#include <cuda_runtime.h>
#include <cublasLt.h>

namespace {

/// L336: the committed kernel choices (pulsar_cuda_lt_picks.inc, written by tests/lt_picks).
const pulsar_lt_pick k_picks[] = {
#include "pulsar_cuda_lt_picks.inc"
    {},   /* the end: cc 0 matches no device */
};

/// The module's cuBLASLt handle, created on the first GEMM.
cublasLtHandle_t g_lt = nullptr;
/// The module's device workspace (kWs bytes), allocated on the first GEMM.
void *g_ws = nullptr;
/// Workspace bytes every algorithm is chosen and checked against.
constexpr size_t kWs = PULSAR_LT_FIXED_WORKSPACE;

/* ONE algorithm per (A's layout, row class, K, N), chosen at the class's canonical row count */
/// A cached shape: its matmul descriptor and the one algorithm chosen at the row class's canonical count.
struct lt_shape {
    /// A's layout (pulsar_lt_a)
    int a, cls, K, N;
    /// the element type of A and X (CUDA_R_16F or, L336, CUDA_R_16BF)
    cudaDataType_t el;
    /** @var cls
     *  row class (pulsar_lt_rows) */
    /** @var K
     *  reduction depth */
    /** @var N
     *  full output width (A's columns) */
    /// slot in use
    bool valid;
    /// f32-compute descriptor (TRANSA set for PULSAR_LT_A_NK); owned, destroyed on eviction
    cublasLtMatmulDesc_t op;
    /// the algorithm the heuristic returned at canon_rows(cls) rows, split-K disabled
    cublasLtMatmulAlgo_t algo;
};
/* per (shape, M, columns n, output ld): the call's layouts, checked against the shape's algorithm */
/// A cached call: the matrix layouts for one (shape, M, n, ldd), verified against the shape's algorithm.
struct lt_call {
    /// the owning shape's A layout (pulsar_lt_a)
    int a, cls, K, N, M, n, ldd;
    /// the owning shape's element type
    cudaDataType_t el;
    /** @var cls
     *  the owning shape's row class (pulsar_lt_rows) */
    /** @var K
     *  the owning shape's reduction depth */
    /** @var N
     *  the owning shape's full output width */
    /** @var M
     *  the call's activation rows */
    /** @var n
     *  the call's output columns */
    /** @var ldd
     *  the call's output row pitch, in floats */
    /// slot in use
    bool valid;
    /// A's layout (the element type, n columns of the weight); owned, destroyed on eviction
    cublasLtMatrixLayout_t la, lb, ld;
    /** @var lb
     *  X's layout (the element type, K x M, ld K); owned */
    /** @var ld
     *  D's layout (f32 n x M, ld ldd); owned */
};
/// Shape cache, 32 slots, replaced round-robin (evicting a shape also drops its calls).
lt_shape g_shapes[32];
/// Next g_shapes slot to fill.
int g_shapes_next;
/// Call cache, 128 slots, replaced round-robin.
lt_call g_calls[128];
/// Next g_calls slot to fill.
int g_calls_next;

/// The row class's canonical row count, the one its algorithm is chosen at.
/// @param cls pulsar_lt_rows row class
/// @return PULSAR_PREFILL_CHUNK_DEFAULT for PROMPT, PULSAR_GPU_MNEUTRAL_ROWS_MAX for DECODE
int canon_rows(int cls) { return cls == PULSAR_LT_PROMPT ? (int)PULSAR_PREFILL_CHUNK_DEFAULT : (int)PULSAR_GPU_MNEUTRAL_ROWS_MAX; }

/* A's layout for columns [0, n) of the weight: KN = n x K at ld N; NK = K x n at ld K (the op transposes it) */
/** Creates A's cuBLASLt layout for weight columns [0, n): KN = n x K at ld N; NK = K x n at ld K.
 *  @param la out: the created layout (caller destroys)
 *  @param el the element type
 *  @param a  pulsar_lt_a layout id
 *  @param K  reduction depth
 *  @param N  full output width (the KN row pitch)
 *  @param n  columns this layout covers */
void a_layout(cublasLtMatrixLayout_t *la, cudaDataType_t el, int a, int K, int N, int n) {
    if (a == PULSAR_LT_A_KN) cublasLtMatrixLayoutCreate(la, el, n, K, N);
    else cublasLtMatrixLayoutCreate(la, el, K, n, K);
}

/** Finds or builds the cached shape (descriptor + heuristic algorithm at canon_rows(cls) rows, reduction NONE).
 *  @param el  the element type of A and X
 *  @param a   pulsar_lt_a layout id
 *  @param cls pulsar_lt_rows row class
 *  @param K   reduction depth
 *  @param N   full output width
 *  @param tag caller's name for the refusal message
 *  @return the shape, or NULL (refused on stderr: no algorithm) */
const lt_shape *shape_for(cudaDataType_t el, int a, int cls, int K, int N, const char *tag) {
    for (lt_shape &sh : g_shapes)
        if (sh.valid && sh.el == el && sh.a == a && sh.cls == cls && sh.K == K && sh.N == N) return &sh;
    lt_shape ns = {};
    ns.el = el; ns.a = a; ns.cls = cls; ns.K = K; ns.N = N;
    if (cublasLtMatmulDescCreate(&ns.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return nullptr;
    if (a == PULSAR_LT_A_NK) {
        const cublasOperation_t t = CUBLAS_OP_T;
        cublasLtMatmulDescSetAttribute(ns.op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t));
    }
    const int R = canon_rows(cls);
    cublasLtMatrixLayout_t la, lb, ld;
    a_layout(&la, el, a, K, N, N);
    cublasLtMatrixLayoutCreate(&lb, el, K, R, K);
    cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, N, R, N);
    const char *const eln = el == CUDA_R_16BF ? "bf16" : "fp16";
    const char *const an = a == PULSAR_LT_A_KN ? "KN" : "NK";
    const char *const cn = cls == PULSAR_LT_PROMPT ? "prompt" : "decode";
    /* L336: a recorded pick, when the table has this shape on this device and cuBLASLt -- checked to run at the
     * class's canonical rows like the heuristic's (an entry that does not is a bad table: refused, said) */
    if (const pulsar_lt_pick *p = pulsar_lt_pick_find(el == CUDA_R_16BF ? 1 : 0, a, cls, K, N)) {
        cublasLtMatmulHeuristicResult_t chk;
        const bool ok = pulsar_lt_pick_algo(g_lt, p, &ns.algo) &&
                        cublasLtMatmulAlgoCheck(g_lt, ns.op, la, lb, ld, ld, &ns.algo, &chk) == CUBLAS_STATUS_SUCCESS &&
                        chk.state == CUBLAS_STATUS_SUCCESS && chk.workspaceSize <= kWs;
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld);
        if (!ok) {
            cublasLtMatmulDescDestroy(ns.op);
            fprintf(stderr, "%s: the recorded cuBLASLt pick for %s %s K=%d N=%d (%s rows; algo %d tile %d) does not run "
                            "here -- a bad pulsar_cuda_lt_picks.inc entry; refusing\n", tag, eln, an, K, N, cn,
                    p->algo_id, p->tile);
            return nullptr;
        }
        fprintf(stderr, "pulsar: fixed GEMM %s %s K=%d N=%d (%s rows): the recorded pick, algo %d tile %d stages %d\n",
                eln, an, K, N, cn, p->algo_id, p->tile, p->stages);
    } else {
        cublasLtMatmulPreference_t pf;
        cublasLtMatmulPreferenceCreate(&pf);
        cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWs, sizeof(kWs));
        const uint32_t red = CUBLASLT_REDUCTION_SCHEME_NONE;
        cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &red, sizeof(red));
        cublasLtMatmulHeuristicResult_t hr;
        int got = 0;
        const cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(g_lt, ns.op, la, lb, ld, ld, pf, 1, &hr, &got);
        cublasLtMatmulPreferenceDestroy(pf);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld);
        if (hs != CUBLAS_STATUS_SUCCESS || !got) {
            cublasLtMatmulDescDestroy(ns.op);
            fprintf(stderr, "%s: no fixed-order %s cuBLASLt kernel for K=%d N=%d at %d rows -- refusing\n", tag, eln, K,
                    N, R);
            return nullptr;
        }
        ns.algo = hr.algo;
        pulsar_lt_pick seen{};
        pulsar_lt_pick_from_algo(&ns.algo, &seen);
        fprintf(stderr, "pulsar: fixed GEMM %s %s K=%d N=%d (%s rows): cuBLASLt's first suggestion, algo %d tile %d "
                        "stages %d (no recorded pick)\n", eln, an, K, N, cn, seen.algo_id, seen.tile, seen.stages);
    }
    ns.valid = true;
    lt_shape &slot = g_shapes[g_shapes_next];
    g_shapes_next = (g_shapes_next + 1) & 31;
    if (slot.valid) {
        /* a call built on the evicted shape must not outlive it */
        for (lt_call &c : g_calls)
            if (c.valid && c.el == slot.el && c.a == slot.a && c.cls == slot.cls && c.K == slot.K && c.N == slot.N) {
                cublasLtMatrixLayoutDestroy(c.la);
                cublasLtMatrixLayoutDestroy(c.lb);
                cublasLtMatrixLayoutDestroy(c.ld);
                c.valid = false;
            }
        cublasLtMatmulDescDestroy(slot.op);
    }
    slot = ns;
    return &slot;
}

/** Finds or builds the cached call for the shape at M rows, n columns, output pitch ldd, checking the shape's
 *  algorithm supports it (cublasLtMatmulAlgoCheck, workspace <= kWs).
 *  @param sh  the shape
 *  @param M   activation rows
 *  @param n   output columns
 *  @param ldd output row pitch in floats
 *  @param tag caller's name for the refusal message
 *  @return the call, or NULL (refused on stderr: the algorithm does not support it) */
const lt_call *call_for(const lt_shape &sh, int M, int n, int ldd, const char *tag) {
    for (lt_call &c : g_calls)
        if (c.valid && c.el == sh.el && c.a == sh.a && c.cls == sh.cls && c.K == sh.K && c.N == sh.N && c.M == M &&
            c.n == n && c.ldd == ldd)
            return &c;
    lt_call nc = {};
    nc.el = sh.el; nc.a = sh.a; nc.cls = sh.cls; nc.K = sh.K; nc.N = sh.N; nc.M = M; nc.n = n; nc.ldd = ldd;
    a_layout(&nc.la, sh.el, sh.a, sh.K, sh.N, n);
    cublasLtMatrixLayoutCreate(&nc.lb, sh.el, sh.K, M, sh.K);
    cublasLtMatrixLayoutCreate(&nc.ld, CUDA_R_32F, n, M, ldd);
    cublasLtMatmulHeuristicResult_t chk;
    const cublasStatus_t cs = cublasLtMatmulAlgoCheck(g_lt, sh.op, nc.la, nc.lb, nc.ld, nc.ld, &sh.algo, &chk);
    if (cs != CUBLAS_STATUS_SUCCESS || chk.state != CUBLAS_STATUS_SUCCESS || chk.workspaceSize > kWs) {
        cublasLtMatrixLayoutDestroy(nc.la);
        cublasLtMatrixLayoutDestroy(nc.lb);
        cublasLtMatrixLayoutDestroy(nc.ld);
        fprintf(stderr, "%s: the fixed kernel for K=%d N=%d (chosen at %d rows) does not support %d rows x %d columns "
                        "(ld %d) -- refusing\n", tag, sh.K, sh.N, canon_rows(sh.cls), M, n, ldd);
        return nullptr;
    }
    nc.valid = true;
    lt_call &slot = g_calls[g_calls_next];
    g_calls_next = (g_calls_next + 1) & 127;
    if (slot.valid) {
        cublasLtMatrixLayoutDestroy(slot.la);
        cublasLtMatrixLayoutDestroy(slot.lb);
        cublasLtMatrixLayoutDestroy(slot.ld);
    }
    slot = nc;
    return &slot;
}

/** The fixed GEMM in either element type (pulsar_lt_fixed_gemm / _bf16: the header's contract).
 *  @param el the element type of A and X (2 bytes each)
 *  @return 0, -1 refused, -3 the GEMM failed */
/// @param a_layout_id how A is stored (pulsar_lt_a)
/// @param rows        the row class whose algorithm runs
/// @param a           device weight A
/// @param K           reduction depth
/// @param N           A's full output width
/// @param n_off       first output column of A to compute
/// @param n           output columns
/// @param x           device activations [M][K]
/// @param M           activation rows
/// @param d           device f32 output [M][ldd]
/// @param ldd         d row pitch in floats
/// @param stream      stream to launch on
/// @param tag         caller's name for refusal messages
int fixed_gemm(cudaDataType_t el, pulsar_lt_a a_layout_id, pulsar_lt_rows rows, const void *a, int K, int N, int n_off,
               int n, const void *x, int M, float *d, int ldd, cudaStream_t stream, const char *tag) {
    if (!a || !x || !d || K <= 0 || N <= 0 || n <= 0 || n_off < 0 || n_off + n > N || M <= 0 || ldd < n) {
        fprintf(stderr, "%s: fixed GEMM: bad arguments (K=%d N=%d columns [%d, %d) M=%d ld %d) -- refusing\n", tag, K, N,
                n_off, n_off + n, M, ldd);
        return -1;
    }
    if (!g_lt && cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) {
        g_lt = nullptr;
        fprintf(stderr, "%s: cuBLASLt unavailable -- refusing\n", tag);
        return -1;
    }
    if (!g_ws && cudaMalloc(&g_ws, kWs) != cudaSuccess) {
        g_ws = nullptr;
        fprintf(stderr, "%s: the fixed GEMM's %zu B workspace refused -- refusing\n", tag, kWs);
        return -1;
    }
    const lt_shape *sh = shape_for(el, a_layout_id, rows, K, N, tag);
    const lt_call *c = sh ? call_for(*sh, M, n, ldd, tag) : nullptr;
    if (!c) return -1;
    const uint16_t *ap = static_cast<const uint16_t *>(a);   /* 2-byte elements (fp16 or bf16) */
    const void *a_cols = a_layout_id == PULSAR_LT_A_KN ? (const void *)(ap + n_off) : (const void *)(ap + (size_t)n_off * K);
    const float one = 1.0f, zero = 0.0f;
    /* L338: stream 0 here is the per-thread default stream (the engine builds --default-stream per-thread); cuBLAS,
     * built apart, would read 0 as the LEGACY stream -- outside a captured step's stream (status 14 under capture) */
    const cudaStream_t s = stream ? stream : cudaStreamPerThread;
    const cublasStatus_t st = cublasLtMatmul(g_lt, sh->op, &one, a_cols, c->la, x, c->lb, &zero, d, c->ld, d, c->ld,
                                             &sh->algo, g_ws, kWs, s);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cuBLAS GEMM failed (%d) -- refusing\n", tag, (int)st);
        return -3;
    }
    return 0;
}

}  // namespace

const pulsar_lt_pick *pulsar_lt_pick_find(int el, int a, int cls, int K, int N) {
    static int cc = -1, ver = 0;
    if (cc < 0) {
        int dev = 0, ma = 0, mi = 0;
        cc = cudaGetDevice(&dev) == cudaSuccess &&
                     cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                     cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess
                 ? ma * 10 + mi
                 : 0;
        ver = (int)cublasLtGetVersion();
    }
    for (const pulsar_lt_pick &p : k_picks)
        if (p.cc != 0 && p.cc == cc && p.lt_version == ver && p.el == el && p.a == a && p.cls == cls && p.K == K && p.N == N)
            return &p;
    return nullptr;
}

void pulsar_lt_pick_from_algo(const void *algo, pulsar_lt_pick *out) {
    const cublasLtMatmulAlgo_t *al = static_cast<const cublasLtMatmulAlgo_t *>(algo);
    size_t sz = 0;
    int32_t id = 0;
    uint32_t tile = 0, stages = 0, splitk = 0, red = 0, swz = 0, cust = 0;
    uint16_t inner = 0, cluster = 0;
    cublasLtMatmulAlgoConfigGetAttribute(al, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, &sz);
    cublasLtMatmulAlgoConfigGetAttribute(al, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile, &sz);
    cublasLtMatmulAlgoConfigGetAttribute(al, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof stages, &sz);
    cublasLtMatmulAlgoConfigGetAttribute(al, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof splitk, &sz);
    cublasLtMatmulAlgoConfigGetAttribute(al, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof red, &sz);
    cublasLtMatmulAlgoConfigGetAttribute(al, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof swz, &sz);
    cublasLtMatmulAlgoConfigGetAttribute(al, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &cust, sizeof cust, &sz);
    cublasLtMatmulAlgoConfigGetAttribute(al, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner, sizeof inner, &sz);
    cublasLtMatmulAlgoConfigGetAttribute(al, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster, sizeof cluster, &sz);
    out->algo_id = id;
    out->tile = (int)tile;
    out->stages = (int)stages;
    out->splitk = (int)splitk;
    out->reduction = (int)red;
    out->swizzle = (int)swz;
    out->custom = (int)cust;
    out->inner = (int)inner;
    out->cluster = (int)cluster;
}

bool pulsar_lt_pick_algo(void *handle, const pulsar_lt_pick *p, void *algo) {
    cublasLtMatmulAlgo_t *al = static_cast<cublasLtMatmulAlgo_t *>(algo);
    const cudaDataType_t el = p->el ? CUDA_R_16BF : CUDA_R_16F;
    if (cublasLtMatmulAlgoInit(static_cast<cublasLtHandle_t>(handle), CUBLAS_COMPUTE_32F, CUDA_R_32F, el, el, CUDA_R_32F,
                               CUDA_R_32F, p->algo_id, al) != CUBLAS_STATUS_SUCCESS)
        return false;
    const uint32_t tile = (uint32_t)p->tile, stages = (uint32_t)p->stages, splitk = (uint32_t)p->splitk,
                   red = (uint32_t)p->reduction, swz = (uint32_t)p->swizzle, cust = (uint32_t)p->custom;
    const uint16_t inner = (uint16_t)p->inner, cluster = (uint16_t)p->cluster;
    return cublasLtMatmulAlgoConfigSetAttribute(al, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile) == 0 &&
           cublasLtMatmulAlgoConfigSetAttribute(al, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof stages) == 0 &&
           cublasLtMatmulAlgoConfigSetAttribute(al, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof splitk) == 0 &&
           cublasLtMatmulAlgoConfigSetAttribute(al, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof red) == 0 &&
           cublasLtMatmulAlgoConfigSetAttribute(al, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof swz) == 0 &&
           cublasLtMatmulAlgoConfigSetAttribute(al, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &cust, sizeof cust) == 0 &&
           cublasLtMatmulAlgoConfigSetAttribute(al, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner, sizeof inner) == 0 &&
           cublasLtMatmulAlgoConfigSetAttribute(al, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster, sizeof cluster) == 0;
}

int pulsar_lt_fixed_gemm(pulsar_lt_a a_layout_id, pulsar_lt_rows rows, const void *a, int K, int N, int n_off, int n,
                         const void *x, int M, float *d, int ldd, cudaStream_t stream, const char *tag) {
    return fixed_gemm(CUDA_R_16F, a_layout_id, rows, a, K, N, n_off, n, x, M, d, ldd, stream, tag);
}

int pulsar_lt_fixed_gemm_bf16(pulsar_lt_a a_layout_id, pulsar_lt_rows rows, const void *a, int K, int N, int n_off,
                              int n, const void *x, int M, float *d, int ldd, cudaStream_t stream, const char *tag) {
    return fixed_gemm(CUDA_R_16BF, a_layout_id, rows, a, K, N, n_off, n, x, M, d, ldd, stream, tag);
}

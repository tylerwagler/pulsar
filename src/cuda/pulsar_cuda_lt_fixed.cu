/** @file
 *  The fixed-algorithm fp16 cuBLASLt GEMM: one algorithm per shape and row class, its caches. */
/* The fixed-algorithm fp16 GEMM (L321) -- see pulsar_cuda_lt_fixed.h for the contract. */
#include "pulsar_cuda_lt_fixed.h"

#include "pulsar.h"       /* PULSAR_PREFILL_CHUNK_DEFAULT: the prompt class's canonical row count */
#include "pulsar_gpu.h"   /* PULSAR_GPU_MNEUTRAL_ROWS_MAX: the decode class's */

#include <cstdio>
#include <cublasLt.h>

namespace {

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
    /// A's layout (fp16, n columns of the weight); owned, destroyed on eviction
    cublasLtMatrixLayout_t la, lb, ld;
    /** @var lb
     *  X's layout (fp16 K x M, ld K); owned */
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
 *  @param a  pulsar_lt_a layout id
 *  @param K  reduction depth
 *  @param N  full output width (the KN row pitch)
 *  @param n  columns this layout covers */
void a_layout(cublasLtMatrixLayout_t *la, int a, int K, int N, int n) {
    if (a == PULSAR_LT_A_KN) cublasLtMatrixLayoutCreate(la, CUDA_R_16F, n, K, N);
    else cublasLtMatrixLayoutCreate(la, CUDA_R_16F, K, n, K);
}

/** Finds or builds the cached shape (descriptor + heuristic algorithm at canon_rows(cls) rows, reduction NONE).
 *  @param a   pulsar_lt_a layout id
 *  @param cls pulsar_lt_rows row class
 *  @param K   reduction depth
 *  @param N   full output width
 *  @param tag caller's name for the refusal message
 *  @return the shape, or NULL (refused on stderr: no algorithm) */
const lt_shape *shape_for(int a, int cls, int K, int N, const char *tag) {
    for (lt_shape &sh : g_shapes)
        if (sh.valid && sh.a == a && sh.cls == cls && sh.K == K && sh.N == N) return &sh;
    lt_shape ns = {};
    ns.a = a; ns.cls = cls; ns.K = K; ns.N = N;
    if (cublasLtMatmulDescCreate(&ns.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return nullptr;
    if (a == PULSAR_LT_A_NK) {
        const cublasOperation_t t = CUBLAS_OP_T;
        cublasLtMatmulDescSetAttribute(ns.op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t));
    }
    const int R = canon_rows(cls);
    cublasLtMatrixLayout_t la, lb, ld;
    a_layout(&la, a, K, N, N);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, R, K);
    cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, N, R, N);
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
        fprintf(stderr, "%s: no fixed-order cuBLASLt kernel for K=%d N=%d at %d rows -- refusing\n", tag, K, N, R);
        return nullptr;
    }
    ns.algo = hr.algo;
    ns.valid = true;
    lt_shape &slot = g_shapes[g_shapes_next];
    g_shapes_next = (g_shapes_next + 1) & 31;
    if (slot.valid) {
        /* a call built on the evicted shape must not outlive it */
        for (lt_call &c : g_calls)
            if (c.valid && c.a == slot.a && c.cls == slot.cls && c.K == slot.K && c.N == slot.N) {
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
        if (c.valid && c.a == sh.a && c.cls == sh.cls && c.K == sh.K && c.N == sh.N && c.M == M && c.n == n &&
            c.ldd == ldd)
            return &c;
    lt_call nc = {};
    nc.a = sh.a; nc.cls = sh.cls; nc.K = sh.K; nc.N = sh.N; nc.M = M; nc.n = n; nc.ldd = ldd;
    a_layout(&nc.la, sh.a, sh.K, sh.N, n);
    cublasLtMatrixLayoutCreate(&nc.lb, CUDA_R_16F, sh.K, M, sh.K);
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

}  // namespace

int pulsar_lt_fixed_gemm(pulsar_lt_a a_layout_id, pulsar_lt_rows rows, const void *a, int K, int N, int n_off, int n,
                         const void *x, int M, float *d, int ldd, cudaStream_t stream, const char *tag) {
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
    const lt_shape *sh = shape_for(a_layout_id, rows, K, N, tag);
    const lt_call *c = sh ? call_for(*sh, M, n, ldd, tag) : nullptr;
    if (!c) return -1;
    const uint16_t *ap = static_cast<const uint16_t *>(a);   /* fp16 elements */
    const void *a_cols = a_layout_id == PULSAR_LT_A_KN ? (const void *)(ap + n_off) : (const void *)(ap + (size_t)n_off * K);
    const float one = 1.0f, zero = 0.0f;
    const cublasStatus_t st = cublasLtMatmul(g_lt, sh->op, &one, a_cols, c->la, x, c->lb, &zero, d, c->ld, d, c->ld,
                                             &sh->algo, g_ws, kWs, stream);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cuBLAS GEMM failed (%d) -- refusing\n", tag, (int)st);
        return -3;
    }
    return 0;
}

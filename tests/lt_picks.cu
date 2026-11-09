/** @file
 *  L336: write pulsar_lt_fixed's kernel table -- time cuBLASLt's candidates per shape, keep the row-independent ones. */
/* L336: the tool that writes src/cuda/pulsar_cuda_lt_picks.inc.  For each shape (from the arguments, or every
 * "fixed GEMM ..." line of a server log: --from-log FILE) it asks cuBLASLt for up to 16 candidates at the row class's
 * canonical rows with split-K off (pulsar_lt_fixed's own preference), and keeps a candidate only when
 *   - a row's bytes are the same at the canonical rows, at a middle count and at 1 row (lt_fixed's contract), and
 *   - the algorithm rebuilt from its record (pulsar_lt_pick_algo, what the server runs) writes the same bytes;
 * then times the kept ones (median of 5 rounds of 20) and prints a table row for the fastest when it beats the first
 * suggestion by >= 3 % -- else a comment: the shape keeps the first suggestion and needs no row.  Measure on a quiet
 * device (a Sparky window): the times decide the table, and the table decides the bits.
 *
 *   tests/lt_picks [--from-log FILE] [fp16|bf16:KN|NK:prompt|decode:K:N ...] > rows.txt */
#include "pulsar_cuda_lt_fixed.h"
#include "pulsar.h"
#include "pulsar_gpu.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

/// One shape to measure.
struct shape {
    int el;    ///< 0 fp16, 1 bf16
    int a;     ///< pulsar_lt_a
    int cls;   ///< pulsar_lt_rows
    int K;     ///< reduction depth
    int N;     ///< A's full output width
};

/// cuBLASLt's workspace for every call (lt_fixed's size).
constexpr size_t kWs = PULSAR_LT_FIXED_WORKSPACE;

/// @param s a "fp16:KN:prompt:K:N" spelling
/// @param out the shape
/// @return true when it parsed
bool parse_arg(const char *s, shape *out) {
    char el[8], a[8], cls[8];
    if (sscanf(s, "%7[^:]:%7[^:]:%7[^:]:%d:%d", el, a, cls, &out->K, &out->N) != 5) return false;
    out->el = !strcmp(el, "bf16");
    out->a = !strcmp(a, "NK");
    out->cls = !strcmp(cls, "decode");
    return out->K > 0 && out->N > 0;
}

/// Every shape a server log names (its "fixed GEMM fp16 KN K=.. N=.. (prompt rows)" lines), each once.
/// @param path the log
/// @param out appended to
void parse_log(const char *path, std::vector<shape> *out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "lt_picks: cannot read %s\n", path); exit(2); }
    char line[1024];
    while (fgets(line, sizeof line, f)) {
        const char *p = strstr(line, "fixed GEMM ");
        if (!p) continue;
        char el[8], a[8], cls[8];
        shape s{};
        if (sscanf(p, "fixed GEMM %7s %7s K=%d N=%d (%7s", el, a, &s.K, &s.N, cls) != 5) continue;
        s.el = !strcmp(el, "bf16");
        s.a = !strcmp(a, "NK");
        s.cls = !strcmp(cls, "decode");
        bool seen = false;
        for (const shape &o : *out) seen |= o.el == s.el && o.a == s.a && o.cls == s.cls && o.K == s.K && o.N == s.N;
        if (!seen) out->push_back(s);
    }
    fclose(f);
}

/// The matrix layouts of a call at M rows (lt_fixed's: A per its layout, X K x M, D f32 N x M).
void layouts(const shape &s, int M, cublasLtMatrixLayout_t *la, cublasLtMatrixLayout_t *lb, cublasLtMatrixLayout_t *ld) {
    const cudaDataType_t el = s.el ? CUDA_R_16BF : CUDA_R_16F;
    if (s.a == PULSAR_LT_A_KN) cublasLtMatrixLayoutCreate(la, el, s.N, s.K, s.N);
    else cublasLtMatrixLayoutCreate(la, el, s.K, s.N, s.K);
    cublasLtMatrixLayoutCreate(lb, el, s.K, M, s.K);
    cublasLtMatrixLayoutCreate(ld, CUDA_R_32F, s.N, M, s.N);
}

/// One GEMM; false when the algorithm does not run at this M.
bool gemm(cublasLtHandle_t lt, cublasLtMatmulDesc_t op, const shape &s, const cublasLtMatmulAlgo_t &al, const void *A,
          const void *X, float *D, int M, void *ws) {
    cublasLtMatrixLayout_t la, lb, ld;
    layouts(s, M, &la, &lb, &ld);
    cublasLtMatmulHeuristicResult_t chk;
    bool ok = cublasLtMatmulAlgoCheck(lt, op, la, lb, ld, ld, &al, &chk) == CUBLAS_STATUS_SUCCESS &&
              chk.state == CUBLAS_STATUS_SUCCESS && chk.workspaceSize <= kWs;
    const float one = 1.0f, zero = 0.0f;
    if (ok) ok = cublasLtMatmul(lt, op, &one, A, la, X, lb, &zero, D, ld, D, ld, &al, ws, kWs, 0) == CUBLAS_STATUS_SUCCESS;
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(ld);
    return ok;
}

/// D's rows as the host sees them.
std::vector<float> down(const float *D, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), D, n * 4, cudaMemcpyDeviceToHost);
    return h;
}

}  // namespace

int main(int argc, char **argv) {
    std::vector<shape> shapes;
    for (int i = 1; i < argc; i++) {
        shape s{};
        if (!strcmp(argv[i], "--from-log") && i + 1 < argc) parse_log(argv[++i], &shapes);
        else if (parse_arg(argv[i], &s)) shapes.push_back(s);
        else { fprintf(stderr, "lt_picks: %s is not el:layout:class:K:N\n", argv[i]); return 2; }
    }
    if (shapes.empty()) { fprintf(stderr, "lt_picks: no shapes\n"); return 2; }
    cublasLtHandle_t lt;
    void *ws = nullptr;
    if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS || cudaMalloc(&ws, kWs) != cudaSuccess) return 1;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int cc = prop.major * 10 + prop.minor, ver = (int)cublasLtGetVersion();
    printf("/* measured by tests/lt_picks on %s (sm_%d), cuBLASLt %d */\n", prop.name, cc, ver);
    int rows = 0;
    for (const shape &s : shapes) {
        const int R = s.cls == PULSAR_LT_PROMPT ? (int)PULSAR_PREFILL_CHUNK_DEFAULT : (int)PULSAR_GPU_MNEUTRAL_ROWS_MAX;
        const int Mmid = R > 64 ? 37 : 5;
        const char *name = s.el ? "bf16" : "fp16";
        cublasLtMatmulDesc_t op;
        cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (s.a == PULSAR_LT_A_NK) {
            const cublasOperation_t t = CUBLAS_OP_T;
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t);
        }
        cublasLtMatrixLayout_t la, lb, ld;
        layouts(s, R, &la, &lb, &ld);
        cublasLtMatmulPreference_t pf;
        cublasLtMatmulPreferenceCreate(&pf);
        cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWs, sizeof kWs);
        const uint32_t red = CUBLASLT_REDUCTION_SCHEME_NONE;
        cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &red, sizeof red);
        cublasLtMatmulHeuristicResult_t hr[16];
        int got = 0;
        cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pf, 16, hr, &got);
        cublasLtMatmulPreferenceDestroy(pf);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld);
        /* data: weights ~0.05, activations ~4 -- the magnitudes of a prefill's rows */
        const size_t an = (size_t)s.K * s.N, xn = (size_t)R * s.K;
        std::vector<uint16_t> ha(an), hx(xn);
        srand(1105);
        for (auto &v : ha) {
            const float f = (rand() / (float)RAND_MAX - 0.5f) * 0.05f;
            v = s.el ? __bfloat16_as_ushort(__float2bfloat16(f)) : __half_as_ushort(__float2half(f));
        }
        for (auto &v : hx) {
            const float f = (rand() / (float)RAND_MAX - 0.5f) * 4.0f;
            v = s.el ? __bfloat16_as_ushort(__float2bfloat16(f)) : __half_as_ushort(__float2half(f));
        }
        void *A = nullptr, *X = nullptr;
        float *D = nullptr, *D2 = nullptr;
        cudaMalloc(&A, an * 2);
        cudaMalloc(&X, xn * 2);
        cudaMalloc(&D, (size_t)R * s.N * 4);
        cudaMalloc(&D2, (size_t)R * s.N * 4);
        cudaMemcpy(A, ha.data(), an * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(X, hx.data(), xn * 2, cudaMemcpyHostToDevice);
        const uint16_t *X5 = static_cast<const uint16_t *>(X) + (size_t)5 * s.K;
        double t_first = 0.0, t_best = 0.0;
        int best = -1;
        pulsar_lt_pick best_pick{};
        fprintf(stderr, "== %s %s K=%d N=%d (%s rows, %d): %d candidates\n", name, s.a ? "NK" : "KN", s.K, s.N,
                s.cls ? "decode" : "prompt", R, got);
        for (int i = 0; i < got; i++) {
            pulsar_lt_pick p{cc, ver, s.el, s.a, s.cls, s.K, s.N, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            pulsar_lt_pick_from_algo(&hr[i].algo, &p);
            if (!gemm(lt, op, s, hr[i].algo, A, X, D, R, ws)) continue;
            const std::vector<float> full = down(D, (size_t)R * s.N);
            /* the contract: a row's bytes at a middle count and at 1 row are the canonical call's */
            bool indep = gemm(lt, op, s, hr[i].algo, A, X5, D2, Mmid, ws) &&
                         !memcmp(down(D2, (size_t)Mmid * s.N).data(), &full[(size_t)5 * s.N], (size_t)Mmid * s.N * 4) &&
                         gemm(lt, op, s, hr[i].algo, A, X5, D2, 1, ws) &&
                         !memcmp(down(D2, (size_t)s.N).data(), &full[(size_t)5 * s.N], (size_t)s.N * 4);
            /* what the server runs: the algorithm rebuilt from the record writes the same bytes */
            cublasLtMatmulAlgo_t rebuilt;
            const bool same = pulsar_lt_pick_algo(lt, &p, &rebuilt) && gemm(lt, op, s, rebuilt, A, X, D2, R, ws) &&
                              !memcmp(down(D2, (size_t)R * s.N).data(), full.data(), full.size() * 4);
            std::vector<double> rounds;
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            for (int w = 0; w < 3; w++) gemm(lt, op, s, hr[i].algo, A, X, D, R, ws);
            for (int r = 0; r < 5; r++) {
                cudaEventRecord(e0);
                for (int k = 0; k < 20; k++) gemm(lt, op, s, hr[i].algo, A, X, D, R, ws);
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                rounds.push_back(ms / 20.0);
            }
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            std::sort(rounds.begin(), rounds.end());
            const double t = rounds[2];
            if (i == 0) t_first = t;
            fprintf(stderr, "  #%-2d algo %3d tile %3d stages %3d custom %3d  %.3f ms  rows %s  rebuilt %s\n", i, p.algo_id,
                    p.tile, p.stages, p.custom, t, indep ? "independent" : "DEPEND ON M", same ? "same" : "DIFFERS");
            if (indep && same && (best < 0 || t < t_best)) { best = i; t_best = t; best_pick = p; }
        }
        const double gain = t_first > 0.0 ? 100.0 * (t_first - t_best) / t_first : 0.0;
        if (best > 0 && t_best <= 0.97 * t_first) {
            const pulsar_lt_pick &p = best_pick;
            printf("{%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d},   /* %s %s K=%d N=%d %s: %.3f -> %.3f "
                   "ms (-%.0f %%) */\n", p.cc, p.lt_version, p.el, p.a, p.cls, p.K, p.N, p.algo_id, p.tile, p.stages,
                   p.splitk, p.reduction, p.swizzle, p.custom, p.inner, p.cluster, name, s.a ? "NK" : "KN", s.K, s.N,
                   s.cls ? "decode" : "prompt", t_first, t_best, gain);
            rows++;
        } else {
            printf("/* %s %s K=%d N=%d %s: the first suggestion kept (best of the kept candidates %.1f %% faster) */\n",
                   name, s.a ? "NK" : "KN", s.K, s.N, s.cls ? "decode" : "prompt", best < 0 ? 0.0 : gain);
        }
        cudaFree(A);
        cudaFree(X);
        cudaFree(D);
        cudaFree(D2);
        cublasLtMatmulDescDestroy(op);
    }
    fprintf(stderr, "lt_picks: %zu shapes, %d rows\n", shapes.size(), rows);
    cudaFree(ws);
    cublasLtDestroy(lt);
    return 0;
}

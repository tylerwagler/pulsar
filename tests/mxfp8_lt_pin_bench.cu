/* L382: the MXFP8 prefill arm's kernel choice -- cuBLASLt's per-row-count heuristic against ONE kernel per shape
 * chosen at the canonical row count (the bf16 arm's L183 pin).  Times both on random E4M3 data for each shape and
 * row count, and checks the pinned kernel's rows are the same bytes at every row count (the M-independence the pin
 * buys).  Usage: mxfp8_lt_pin_bench IN:OUT[:f16] ...   (no model needed) */
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define LK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s: status %d\n", #x, (int)s_); exit(1); } } while (0)

static const int CANON = 4096;
static const size_t WS = 32u << 20;
static int rup(int a, int b) { return (a + b - 1) / b * b; }
static size_t sf_bytes(int rows, int k) { return (size_t)rup(rows, 128) * rup(k / 32, 4); }

__global__ void fill_kernel(unsigned char *p, size_t n, unsigned seed, int scale) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        unsigned h = (unsigned)i * 2654435761u ^ seed; h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
        /* scales near 2^0 (E8M0 127 +- 3); data: finite E4M3 (exponent field != 15 with mantissa 7) */
        p[i] = scale ? (unsigned char)(124 + (h % 7)) : (unsigned char)((h & 0x77) | (h & 0x80));
    }
}
static void fill(void *p, size_t n, unsigned seed, int scale) { fill_kernel<<<256, 256>>>((unsigned char *)p, n, seed, scale); CK(cudaGetLastError()); }

struct Ctx { cublasLtHandle_t h; void *ws; };

static cublasLtMatmulDesc_t make_op(void *sa, void *sb) {
    cublasLtMatmulDesc_t op; LK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    cublasLtMatmulMatrixScale_t mo = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mo, sizeof(mo));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mo, sizeof(mo));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb));
    return op;
}

static bool heuristic(Ctx &c, cublasLtMatmulDesc_t op, int in, int out, int rows, int f16, cublasLtMatmulAlgo_t *algo) {
    cublasLtMatrixLayout_t la, lb, ld;
    cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, in, out, in);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, in, rows, in);
    cublasLtMatrixLayoutCreate(&ld, f16 ? CUDA_R_16F : CUDA_R_32F, out, rows, out);
    cublasLtMatmulPreference_t pf; cublasLtMatmulPreferenceCreate(&pf);
    size_t wz = WS; cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wz, sizeof(wz));
    uint32_t red = CUBLASLT_REDUCTION_SCHEME_NONE;
    cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &red, sizeof(red));
    cublasLtMatmulHeuristicResult_t r; int got = 0;
    cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(c.h, op, la, lb, ld, ld, pf, 1, &r, &got);
    cublasLtMatmulPreferenceDestroy(pf); cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(ld);
    if (s != CUBLAS_STATUS_SUCCESS || !got) return false;
    *algo = r.algo; return true;
}

/* time `reps` calls; returns microseconds per call, or -1 when the algo refuses this row count */
static double run(Ctx &c, cublasLtMatmulDesc_t op, const cublasLtMatmulAlgo_t *algo, int in, int out, int rows, int f16,
                  const void *w, const void *x, void *d, int reps) {
    cublasLtMatrixLayout_t la, lb, ld;
    cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, in, out, in);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, in, rows, in);
    cublasLtMatrixLayoutCreate(&ld, f16 ? CUDA_R_16F : CUDA_R_32F, out, rows, out);
    cublasLtMatmulHeuristicResult_t chk;
    double us = -1;
    if (cublasLtMatmulAlgoCheck(c.h, op, la, lb, ld, ld, algo, &chk) == CUBLAS_STATUS_SUCCESS &&
        chk.state == CUBLAS_STATUS_SUCCESS && chk.workspaceSize <= WS) {
        const float al = 1.f, be = 0.f;
        for (int i = 0; i < 3; i++) LK(cublasLtMatmul(c.h, op, &al, w, la, x, lb, &be, d, ld, d, ld, algo, c.ws, WS, 0));
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < reps; i++) LK(cublasLtMatmul(c.h, op, &al, w, la, x, lb, &be, d, ld, d, ld, algo, c.ws, WS, 0));
        cudaEventRecord(b); CK(cudaEventSynchronize(b));
        float ms; cudaEventElapsedTime(&ms, a, b); us = ms * 1000.0 / reps;
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(ld);
    return us;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s IN:OUT[:f16] ...\n", argv[0]); return 2; }
    Ctx c; LK(cublasLtCreate(&c.h)); CK(cudaMalloc(&c.ws, WS));
    const int rows_list[] = {1, 4, 11, 16, 32, 64, 100, 128, 200, 256, 384, 512, 768, 1024, 2048, 4096};
    double tot_h = 0, tot_p = 0, tot_h_small = 0, tot_p_small = 0;
    int refused = 0, mismatched = 0;
    for (int a = 1; a < argc; a++) {
        int in = 0, out = 0; char f[8] = {0};
        if (sscanf(argv[a], "%d:%d:%7s", &in, &out, f) < 2) { fprintf(stderr, "bad shape %s\n", argv[a]); return 2; }
        const int f16 = !strcmp(f, "f16"), osz = f16 ? 2 : 4;
        void *w, *sw, *x, *sx, *d, *dref;
        CK(cudaMalloc(&w, (size_t)in * out)); CK(cudaMalloc(&sw, sf_bytes(out, in)));
        CK(cudaMalloc(&x, (size_t)in * CANON)); CK(cudaMalloc(&sx, sf_bytes(CANON, in)));
        CK(cudaMalloc(&d, (size_t)out * CANON * osz)); CK(cudaMalloc(&dref, (size_t)out * CANON * osz));
        fill(w, (size_t)in * out, 1u + a, 0); fill(sw, sf_bytes(out, in), 7u + a, 1);
        fill(x, (size_t)in * CANON, 3u + a, 0); fill(sx, sf_bytes(CANON, in), 5u + a, 1);
        cublasLtMatmulDesc_t op = make_op(sw, sx);
        cublasLtMatmulAlgo_t pin;
        if (!heuristic(c, op, in, out, CANON, f16, &pin)) { fprintf(stderr, "%d:%d no kernel at %d\n", in, out, CANON); return 1; }
        /* reference rows from the pinned kernel at the canonical count */
        run(c, op, &pin, in, out, CANON, f16, w, x, dref, 1);
        std::vector<unsigned char> ref((size_t)out * CANON * osz), got;
        CK(cudaMemcpy(ref.data(), dref, ref.size(), cudaMemcpyDeviceToHost));
        printf("shape %d->%d %s\n  %6s %10s %10s %7s  %s\n", in, out, f16 ? "f16" : "f32", "rows", "heur us", "pinned us", "ratio", "pinned rows == canonical rows");
        for (int rows : rows_list) {
            /* the first `rows` rows of x (and their 128-row scale tiles, row-tile major) are a rows-row operand */
            cublasLtMatmulAlgo_t h;
            const int reps = rows <= 256 ? 200 : rows <= 1024 ? 50 : 20;
            double th = heuristic(c, op, in, out, rows, f16, &h) ? run(c, op, &h, in, out, rows, f16, w, x, d, reps) : -1;
            double tp = run(c, op, &pin, in, out, rows, f16, w, x, d, reps);
            const char *same = "refused";
            if (tp >= 0) {
                got.resize((size_t)out * rows * osz);
                CK(cudaMemcpy(got.data(), d, got.size(), cudaMemcpyDeviceToHost));
                same = memcmp(got.data(), ref.data(), got.size()) ? "DIFFERENT" : "same";
                if (got.size() && memcmp(got.data(), ref.data(), got.size())) mismatched++;
            } else refused++;
            printf("  %6d %10.1f %10.1f %7.3f  %s\n", rows, th, tp, th > 0 && tp > 0 ? tp / th : 0.0, same);
            if (th > 0 && tp > 0) { tot_h += th; tot_p += tp; if (rows <= 512) { tot_h_small += th; tot_p_small += tp; } }
        }
        cublasLtMatmulDescDestroy(op);
        cudaFree(w); cudaFree(sw); cudaFree(x); cudaFree(sx); cudaFree(d); cudaFree(dref);
    }
    printf("TOTAL heur %.1f us pinned %.1f us ratio %.3f | rows<=512: heur %.1f pinned %.1f ratio %.3f | refused %d, byte mismatches %d\n",
           tot_h, tot_p, tot_p / tot_h, tot_h_small, tot_p_small, tot_p_small / tot_h_small, refused, mismatched);
    return 0;
}

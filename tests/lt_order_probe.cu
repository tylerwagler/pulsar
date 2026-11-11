/** @file
 *  L381: does the fixed cuBLASLt GEMM still follow the accumulation order the in-place prompt kernel implements? */
/* The dense EXL3 prompt arm (src/cuda/mmq/qwen_exl3_dense_prefill.cu) has two kernels with one set of bytes: the
 * fixed-algorithm cuBLASLt GEMM -- pulsar_lt_fixed_gemm(PULSAR_LT_A_KN, PULSAR_LT_PROMPT): fp16 W_q [K][N] times fp16
 * rows [M][K], f32 accumulate and output, ONE algorithm a shape (a recorded pick or the first suggestion at 4096
 * rows, split-K off) -- and, for small chunks, dense_inplace_kernel, which reads the trellis in place and accumulates
 * each output as ONE ascending chain of pulsar_mma_f16_16816 over the 16-wide k-tiles ("k16 xA asc G1 S0" below; the
 * rows on the MMA's m16 side).  The two agree only while the VENDOR kernel follows that order, which is a property of
 * (compute capability, cuBLASLt version, shape).  This tool establishes it, on the device, per shape:
 *
 *   1. THE CONTRACT: the vendor call at 1, 5, 30, 64 and 256 rows writes the bytes its 4096-row call wrote for those
 *      rows (lt_fixed's promise; tests/lt_picks checks it for recorded picks only).
 *   2. THE SHIPPED ORDER: a slow reference kernel computes the same product in the shipped order with the shipped
 *      instruction, compared with the vendor's output BYTE FOR BYTE on
 *        - random operands shaped like the arm's (rows spanning +-2^14 like a prescaled plane, weights like codebook
 *          values), `trials` seeds: orders that differ anywhere differ on most outputs;
 *        - a ONE-HOT scan (below), which separates orders random data could in principle confuse.
 *   3. THE SURVEY, for a shape the shipped order does not reproduce (or every shape, with --survey): the same
 *      comparison for a family of 80 orders -- the instruction (k16 = pulsar_mma_f16_16816: mma.m16n8k16, or below
 *      sm_80 its two-m16n8k8 body; k8 = two chained m16n8k8) x the orientation (xA: the rows are the MMA's m16 side,
 *      the weight its n8 side -- the shipped EXL3 kernels' -- or wA: the weight's outputs on the m16 side) x the k
 *      direction (ascending / descending 16-wide k-tiles from k = 0) x the accumulation: G accumulators taking the
 *      k-tiles round-robin (1, 2, 4; summed in order, or pairwise for 4) and a stage length S (0: every MMA
 *      accumulates into its accumulator, a plain chain; S = 1, 2, 4, 8: S chained MMAs into a fresh fragment, then
 *      one IEEE add into the accumulator) -- and the vendor's raw one-hot signature, so a human can read the
 *      structure when no candidate matches: every weight 2^12; a row holds B = 2^12 at k = p (product 2^24) and
 *      2^-12 at q and q + d (two products of 1).  In f32, 2^24 + 1 rounds back to 2^24, so the row's sum is 2^24 + 2
 *      only if the two small products meet EACH OTHER before either meets B: '2' = they did, '0' = they did not, 'x'
 *      = anything else.  One line per d in {1, 8, 16, 32, 64}, q from p - 96 to p + 95: an ascending chain of
 *      16-wide instructions reads '2' everywhere below p's tile and, above it, '2' exactly where q and q + d share an
 *      instruction; interleaved accumulators and stages show as longer-range '2's.
 *
 * VERDICT per shape: REPRODUCED by the shipped order, or NOT (then the survey's winners, or NONE with the closest).
 * EXIT STATUS: 0 iff the contract holds and EVERY shape is reproduced by the shipped order -- not merely by some
 * order; 3 otherwise; 2 on a setup error.
 *
 *   tests/lt_order_probe [--trials N=3] [--survey] [--gate] [K:N ...]
 *
 * Default shapes: the dense prompt shapes of Qwen3.8-Flash-Next, Nemotron-3.5 Super, Gemma 4 and the Nemotron
 * encoders.  --gate (the battery's `lt-order-gate`, kernels tier): on a device the rule does not give the in-place
 * kernel (ds4_exl3_dense_prompt_arm: not sm_12x) there is nothing to hold -- an N/A line, exit 0.
 *
 * WHEN TO RUN IT.  (1) After any move of the CUDA toolkit, cuBLASLt or the driver on a host that serves the arm: the
 * battery runs it (kernels tier) and exl3-dense-gate section 10 holds the real kernels byte for byte; a serving
 * process is safe without either, because the arm's own check keeps a shape whose bytes differ on the GEMM
 * (docs/QA_BEFORE_RELEASES.md, "After a toolchain move").  (2) To ESTABLISH the order on a new architecture before
 * adding it to ds4_exl3_dense_prompt_arm: run it there without --gate, read the verdicts; if another order wins, the
 * in-place kernel needs that order on that architecture (a kernel change, keyed by the device), never a guess.
 *
 * Compared region: rows [0, 1024) and [3072, 4096) of the 4096-row call (the vendor's first tiles and its last,
 * partial ones) x columns [0, 512) and the last 256 (all of N when N <= 768); exl3-dense-gate compares every output
 * of the real kernels.  Built for the device it runs on (the engine's CUDA_ARCH).  The GPU to itself is not needed
 * (nothing is timed), only ~1.5 GB of free device memory. */
#include "pulsar_cuda_lt_fixed.h"
#include "cuda/mmq/ds4_exl3_dense.cuh"   /* ds4_exl3_dense_prompt_arm: --gate */
#include "cuda/pulsar_cuda_arch.cuh"          /* pulsar_mma_f16_16816: the shipped instruction */
#include "engine/exl3_trellis.h"   /* exl3_f32_to_f16 */
#include "pulsar.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

namespace {

/// One candidate order (the file's header).
struct cand {
    int kw;       ///< 16: pulsar_mma_f16_16816; 8: two chained m16n8k8 (k 0..7 then 8..15 of the tile; reversed when desc)
    int orient;   ///< 0 xA: rows on the m16 side; 1 wA: the weight's outputs on the m16 side
    int desc;     ///< 0 ascending k-tiles, 1 descending
    int G;        ///< accumulators, the k-tiles round-robin (in the order visited)
    int S;        ///< 0 plain chain; else S chained MMAs into a fresh fragment, then one f32 add
    int tree;     ///< G == 4 only: 1 = (a0 + a1) + (a2 + a3), 0 = ((a0 + a1) + a2) + a3
};

/// One m16n8k8: A registers a0 (rows g), a1 (rows g + 8), B register b, of k 0..7 of a half tile.
/// @param d   the thread's four f32 accumulators (in / out)
/// @param a0  A register, rows g
/// @param a1  A register, rows g + 8
/// @param b   B register
__device__ __forceinline__ void mma_k8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(b));
}

/* The reference kernel: one warp per (m16 tile, n8 tile) of the product, every k-tile in the candidate's order.
 * Fragments are built element by element from global memory in the documented m16n8k16 layout (lane l: g = l / 4,
 * c = l % 4; A: rows g, g + 8 at k 2c, 2c + 1 (a[0], a[1]) and 2c + 8, 2c + 9 (a[2], a[3]); B: column g at k 2c,
 * 2c + 1 (b0) and 2c + 8, 2c + 9 (b1); C: rows g (c[0], c[1]) and g + 8 (c[2], c[3]) at columns 2c, 2c + 1). */
/// @param X  fp16 rows [R][K]
/// @param W  fp16 weight [K][N] row-major
/// @param D  out: f32 [R][N]
/// @param R  rows, a multiple of 16
/// @param N  outputs, a multiple of 16
/// @param K  reduction depth, a multiple of 16
/// @param c  the order
__global__ void __launch_bounds__(256)
cand_kernel(const uint16_t *__restrict__ X, const uint16_t *__restrict__ W, float *__restrict__ D, int R, int N, int K,
            const cand c) {
    const int64_t task = (int64_t)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int MI = c.orient ? N : R, NJ = c.orient ? R : N;
    const int n_jt = NJ / 8;
    if (task >= (int64_t)(MI / 16) * n_jt) return;
    const int i0 = (int)(task / n_jt) * 16, j0 = (int)(task % n_jt) * 8;
    const int lane = threadIdx.x & 31, g = lane >> 2, c2 = (lane & 3) * 2;
    auto ms = [&](int i, int k) -> uint32_t { return c.orient ? W[(size_t)k * N + i] : X[(size_t)i * K + k]; };
    auto ns = [&](int k, int j) -> uint32_t { return c.orient ? X[(size_t)j * K + k] : W[(size_t)k * N + j]; };
    float acc[4][4], d[4][4];
    int pos[4];
    for (int s = 0; s < 4; ++s) {
        pos[s] = 0;
        for (int e = 0; e < 4; ++e) acc[s][e] = d[s][e] = 0.0f;
    }
    const int T = K / 16;
    for (int tt = 0; tt < T; ++tt) {
        const int t = c.desc ? T - 1 - tt : tt, s = tt % c.G, k0 = t * 16;
        uint32_t a[4];
        a[0] = ms(i0 + g, k0 + c2) | (ms(i0 + g, k0 + c2 + 1) << 16);
        a[1] = ms(i0 + g + 8, k0 + c2) | (ms(i0 + g + 8, k0 + c2 + 1) << 16);
        a[2] = ms(i0 + g, k0 + c2 + 8) | (ms(i0 + g, k0 + c2 + 9) << 16);
        a[3] = ms(i0 + g + 8, k0 + c2 + 8) | (ms(i0 + g + 8, k0 + c2 + 9) << 16);
        const uint32_t b0 = ns(k0 + c2, j0 + g) | (ns(k0 + c2 + 1, j0 + g) << 16);
        const uint32_t b1 = ns(k0 + c2 + 8, j0 + g) | (ns(k0 + c2 + 9, j0 + g) << 16);
        float f[4];
        for (int e = 0; e < 4; ++e) f[e] = c.S ? (pos[s] ? d[s][e] : 0.0f) : acc[s][e];
        if (c.kw == 16) {
            pulsar_mma_f16_16816(f, a, b0, b1);   /* the shipped kernels' instruction (its sm_75 body below sm_80) */
        } else if (c.desc) {
            mma_k8(f, a[2], a[3], b1);
            mma_k8(f, a[0], a[1], b0);
        } else {
            mma_k8(f, a[0], a[1], b0);
            mma_k8(f, a[2], a[3], b1);
        }
        if (c.S) {
            const bool close = ++pos[s] == c.S || tt + c.G >= T;   /* the stage is full, or this set has no tile left */
            for (int e = 0; e < 4; ++e) {
                if (close) acc[s][e] = acc[s][e] + f[e];
                else d[s][e] = f[e];
            }
            if (close) pos[s] = 0;
        } else {
            for (int e = 0; e < 4; ++e) acc[s][e] = f[e];
        }
    }
    for (int e = 0; e < 4; ++e) {
        float z = acc[0][e];
        if (c.G == 2) z = acc[0][e] + acc[1][e];
        if (c.G == 4) z = c.tree ? (acc[0][e] + acc[1][e]) + (acc[2][e] + acc[3][e])
                                 : ((acc[0][e] + acc[1][e]) + acc[2][e]) + acc[3][e];
        const int i = i0 + g + 8 * (e >> 1), j = j0 + c2 + (e & 1);
        if (c.orient) D[(size_t)j * N + i] = z;
        else D[(size_t)i * N + j] = z;
    }
}

uint64_t g_rng = 0x9E3779B97F4A7C15ull;
uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 16); }
/// Roughly normal in (-1, 1): the mean of four uniforms, centred.
float rnd_bell() {
    float s = 0;
    for (int i = 0; i < 4; ++i) s += (float)(rnd() & 0xffffff) / 16777216.0f;
    return (s - 2.0f) * 0.5f;
}

std::string cand_name(const cand &c) {
    char b[64];
    snprintf(b, sizeof b, "k%d %s %s G%d%s S%d", c.kw, c.orient ? "wA" : "xA", c.desc ? "desc" : "asc", c.G,
             c.G == 4 ? (c.tree ? "t" : "s") : "", c.S);
    return b;
}

constexpr int kCanon = (int)PULSAR_PREFILL_CHUNK_DEFAULT;   ///< the prompt class's canonical rows (4096)
constexpr int kRC = 1024;                                    ///< compared rows at each end of the canonical call
constexpr int kOneHotHalf = 96;                              ///< the one-hot scan's q window: p - 96 .. p + 95

}  // namespace

int main(int argc, char **argv) {
    int trials = 3;
    bool survey_all = false, gate = false;
    std::vector<std::pair<int, int>> shapes;
    for (int i = 1; i < argc; ++i) {
        int K = 0, N = 0;
        if (!strcmp(argv[i], "--trials") && i + 1 < argc) trials = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--survey")) survey_all = true;
        else if (!strcmp(argv[i], "--gate")) gate = true;
        else if (sscanf(argv[i], "%d:%d", &K, &N) == 2 && K >= 256 && K % 128 == 0 && N >= 128 && N % 128 == 0)
            shapes.push_back({K, N});
        else {
            fprintf(stderr, "usage: lt_order_probe [--trials N=3] [--survey] [--gate] [K:N ...]   (K >= 256, K and N "
                            "multiples of 128)\n");
            return 2;
        }
    }
    if (trials < 1) trials = 1;
    if (shapes.empty())
        shapes = {/* Qwen3.8-Flash-Next's dense prompt launches */
                  {2560, 10240}, {2560, 6144}, {2560, 12288}, {2560, 512}, {6144, 2560}, {2560, 640}, {640, 2560},
                  {2560, 2560},
                  /* Nemotron-3.5 Super's */
                  {4096, 18560}, {8192, 4096}, {4096, 4096}, {4096, 256}, {4096, 1024}, {1024, 4096}, {4096, 5376},
                  {5376, 4096},
                  /* Gemma 4's (window and full attention, the dense MLP at its stored width) */
                  {2816, 4096}, {2816, 2048}, {2816, 8192}, {2816, 1024}, {4096, 2816}, {8192, 2816}, {2816, 2176},
                  {2176, 2816},
                  /* the Nemotron encoders' (q and o are 4096 -> 4096 above) */
                  {4096, 14336}, {14336, 4096}};
    /* cands[0] is THE SHIPPED ORDER: dense_inplace_kernel's chain */
    std::vector<cand> cands;
    for (int kw : {16, 8})
        for (int orient : {0, 1})
            for (int desc : {0, 1}) {
                const int gs[10][3] = {{1, 0, 0}, {1, 1, 0}, {1, 2, 0}, {1, 4, 0}, {1, 8, 0},
                                       {2, 0, 0}, {2, 2, 0}, {2, 4, 0}, {4, 0, 0}, {4, 0, 1}};
                for (const auto &q : gs) cands.push_back(cand{kw, orient, desc, q[0], q[1], q[2]});
            }
    const std::string shipped = cand_name(cands[0]);
    int dev = 0, ma = 0, mi = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev));
    CK(cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev));
    char name[256] = "";
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) snprintf(name, sizeof name, "%s", prop.name);
    const size_t ltv = cublasLtGetVersion();
    const bool named = ds4_exl3_dense_prompt_arm(1, ma * 10 + mi) == DS4_EXL3_DENSE_PROMPT_INPLACE;
    printf("lt_order_probe (L381): %s, sm_%d%d, cuBLASLt %zu; the shipped order is \"%s\" (dense_inplace_kernel); the rule "
           "%s this device the in-place kernel; %d random trials + the one-hot scan a shape; the vendor kernel of each "
           "shape is named on stderr (\"pulsar: fixed GEMM ...\")\n", name, ma, mi, ltv, shipped.c_str(),
           named ? "gives" : "does NOT give", trials);
    if (gate && !named) {
        printf("N/A lt-order-gate on sm_%d%d: ds4_exl3_dense_prompt_arm gives no launch the in-place kernel here, so no "
               "prompt byte depends on the vendor's order; run without --gate to establish it for this architecture\n",
               ma, mi);
        return 0;
    }
    if (ma < 8) printf("  NOTE: below sm_80 the k16 candidates are pulsar_mma_f16_16816's two-m16n8k8 body (there is no "
                       "m16n8k16 for fp16)\n");
    const cudaStream_t st = cudaStreamPerThread;
    bool all_ok = true, contract_ok = true;
    int n_shipped = 0;

    for (const auto &shp : shapes) {
        const int K = shp.first, N = shp.second;
        const int NC = N <= 768 ? N : 768;                    /* compared columns: [0, 512) + the last 256 */
        auto col_of = [&](int j) { return N <= 768 ? j : j < 512 ? j : N - 768 + j; };
        const int R2 = 2 * kRC;                               /* compared rows: [0, kRC) + the last kRC */
        uint16_t *dX = nullptr, *dW = nullptr, *dXc = nullptr, *dWc = nullptr;
        float *dD = nullptr, *dD2 = nullptr, *dDc = nullptr;
        CK(cudaMalloc((void **)&dX, (size_t)kCanon * K * 2));
        CK(cudaMalloc((void **)&dW, (size_t)K * N * 2));
        CK(cudaMalloc((void **)&dXc, (size_t)R2 * K * 2));
        CK(cudaMalloc((void **)&dWc, (size_t)K * NC * 2));
        CK(cudaMalloc((void **)&dD, (size_t)kCanon * N * 4));
        CK(cudaMalloc((void **)&dD2, (size_t)256 * N * 4));
        CK(cudaMalloc((void **)&dDc, (size_t)R2 * NC * 4));
        std::vector<uint16_t> hX((size_t)kCanon * K), hW((size_t)K * N), hWc((size_t)K * NC);
        std::vector<float> vfull((size_t)R2 * N), V((size_t)R2 * NC), C((size_t)R2 * NC), d2((size_t)256 * N);
        const int p = (K / 32) * 16 + 5, dq[5] = {1, 8, 16, 32, 64};
        const uint16_t h_big = exl3_f32_to_f16(4096.0f), h_small = exl3_f32_to_f16(1.0f / 4096.0f);
        const uint64_t rng0 = g_rng;                          /* a survey pass re-runs the shape on the same data */
        std::vector<double> eq_sum;
        std::vector<int> exact;
        bool shape_contract = true, bug = false;
        double worst_sanity = 0;
        std::string onehot_sig[5];
        size_t onehot_cols_differ = 0;

        /* pass 0: the shipped order alone; pass 1 (it failed, or --survey): the whole family */
        for (int pass = survey_all ? 1 : 0; pass < 2; ++pass) {
            const size_t n_c = pass ? cands.size() : 1;
            g_rng = rng0;
            eq_sum.assign(cands.size(), 0.0);
            exact.assign(cands.size(), 0);
            for (size_t ci = 0; ci < n_c; ++ci) exact[ci] = 1;
            shape_contract = true;
            worst_sanity = 0;
            onehot_cols_differ = 0;
            for (auto &sg : onehot_sig) sg.clear();
            for (int ds = 0; ds <= trials; ++ds) {
                const bool onehot = ds == trials;
                if (onehot) {
                    std::fill(hX.begin(), hX.end(), (uint16_t)0);
                    std::fill(hW.begin(), hW.end(), h_big);
                    for (int di = 0; di < 5; ++di)
                        for (int qi = 0; qi < 2 * kOneHotHalf; ++qi) {
                            const int q = p - kOneHotHalf + qi, q2 = q + dq[di];
                            uint16_t *row = &hX[(size_t)(di * 2 * kOneHotHalf + qi) * K];
                            row[p] = h_big;
                            if (q != p && q2 != p && q >= 0 && q2 < K) { row[q] = h_small; row[q2] = h_small; }
                        }
                } else {
                    for (auto &v : hX) v = exl3_f32_to_f16(rnd_bell() * 16000.0f);
                    for (auto &v : hW) v = exl3_f32_to_f16(rnd_bell() * 3.0f);
                }
                for (int k = 0; k < K; ++k)
                    for (int j = 0; j < NC; ++j) hWc[(size_t)k * NC + j] = hW[(size_t)k * N + col_of(j)];
                CK(cudaMemcpy(dX, hX.data(), hX.size() * 2, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dW, hW.data(), hW.size() * 2, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dWc, hWc.data(), hWc.size() * 2, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dXc, hX.data(), (size_t)kRC * K * 2, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dXc + (size_t)kRC * K, hX.data() + (size_t)(kCanon - kRC) * K, (size_t)kRC * K * 2,
                              cudaMemcpyHostToDevice));
                /* the vendor at its canonical rows, as the arm calls it */
                CK(cudaMemset(dD, 0xff, (size_t)kCanon * N * 4));
                if (pulsar_lt_fixed_gemm(PULSAR_LT_A_KN, PULSAR_LT_PROMPT, dW, K, N, 0, N, dX, kCanon, dD, N, st, "lt_order_probe"))
                    return 2;
                CK(cudaStreamSynchronize(st));
                CK(cudaMemcpy(vfull.data(), dD, (size_t)kRC * N * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(vfull.data() + (size_t)kRC * N, dD + (size_t)(kCanon - kRC) * N, (size_t)kRC * N * 4,
                              cudaMemcpyDeviceToHost));
                for (int r = 0; r < R2; ++r)
                    for (int j = 0; j < NC; ++j) V[(size_t)r * NC + j] = vfull[(size_t)r * N + col_of(j)];
                /* 1. the contract */
                for (int M : {1, 5, 30, 64, 256}) {
                    CK(cudaMemset(dD2, 0xff, (size_t)M * N * 4));
                    if (pulsar_lt_fixed_gemm(PULSAR_LT_A_KN, PULSAR_LT_PROMPT, dW, K, N, 0, N, dX, M, dD2, N, st, "lt_order_probe"))
                        return 2;
                    CK(cudaStreamSynchronize(st));
                    CK(cudaMemcpy(d2.data(), dD2, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
                    if (memcmp(d2.data(), vfull.data(), (size_t)M * N * 4)) {
                        shape_contract = false;
                        printf("  K=%d N=%d dataset %d: *** the vendor's %d-row call is NOT its 4096-row call's rows ***\n", K,
                               N, ds, M);
                    }
                }
                /* a sanity anchor in double for 8 x 8 outputs: the vendor and every candidate compute THIS product */
                auto h2f = [](uint16_t h) {   /* fp16 -> double, normals and zero (the data holds nothing else) */
                    const int e = (h >> 10) & 31, m = h & 1023;
                    const double v = e ? ldexp(1024.0 + m, e - 25) : ldexp((double)m, -24);
                    return (h & 0x8000) ? -v : v;
                };
                double ref[8][8], refabs[8][8];
                for (int r = 0; r < 8; ++r)
                    for (int j = 0; j < 8; ++j) {
                        double s = 0, sa = 0;
                        for (int k = 0; k < K; ++k) {
                            const double t = h2f(hX[(size_t)r * K + k]) * h2f(hW[(size_t)k * N + col_of(j)]);
                            s += t; sa += fabs(t);
                        }
                        ref[r][j] = s; refabs[r][j] = sa;
                    }
                auto sanity = [&](const std::vector<float> &o) {
                    double w = 0;
                    for (int r = 0; r < 8; ++r)
                        for (int j = 0; j < 8; ++j)
                            w = std::max(w, fabs((double)o[(size_t)r * NC + j] - ref[r][j]) / (refabs[r][j] > 0 ? refabs[r][j] : 1.0));
                    return w;
                };
                const double vs = sanity(V);
                worst_sanity = std::max(worst_sanity, vs);
                if (vs > 1e-4) {
                    bug = true;
                    printf("  K=%d N=%d dataset %d: *** the vendor's output is not the product (%.2e of sum |terms|) -- the "
                           "tool's call or layout is wrong ***\n", K, N, ds, vs);
                }
                if (onehot) {
                    for (int di = 0; di < 5; ++di)
                        for (int qi = 0; qi < 2 * kOneHotHalf; ++qi) {
                            const size_t r = (size_t)(di * 2 * kOneHotHalf + qi);
                            const float v = V[r * NC];
                            onehot_sig[di] += v == 16777218.0f ? '2' : v == 16777216.0f ? '0' : 'x';
                            for (int j = 1; j < NC; ++j) onehot_cols_differ += memcmp(&V[r * NC + j], &V[r * NC], 4) != 0;
                        }
                }
                /* 2. / 3. the orders */
                for (size_t ci = 0; ci < n_c; ++ci) {
                    CK(cudaMemset(dDc, 0xff, (size_t)R2 * NC * 4));
                    const cand &c = cands[ci];
                    const int64_t tasks = (int64_t)((c.orient ? NC : R2) / 16) * ((c.orient ? R2 : NC) / 8);
                    cand_kernel<<<(unsigned)((tasks + 7) / 8), 256, 0, st>>>(dXc, dWc, dDc, R2, NC, K, c);
                    CK(cudaGetLastError());
                    CK(cudaStreamSynchronize(st));
                    CK(cudaMemcpy(C.data(), dDc, C.size() * 4, cudaMemcpyDeviceToHost));
                    size_t same = 0;
                    for (size_t i = 0; i < C.size(); ++i) same += memcmp(&C[i], &V[i], 4) == 0;
                    eq_sum[ci] += (double)same / (double)C.size();
                    if (same != C.size()) exact[ci] = 0;
                    const double cs = sanity(C);
                    if (cs > 1e-4) {
                        exact[ci] = 0;
                        bug = true;
                        printf("  K=%d N=%d dataset %d: *** candidate %s is not the product (%.2e) -- a tool bug ***\n", K, N,
                               ds, cand_name(c).c_str(), cs);
                    }
                }
            }
            if (exact[0] && !survey_all) break;   /* the shipped order holds: no survey */
        }
        /* the shape's report */
        const bool ship_ok = exact[0] != 0;
        n_shipped += ship_ok;
        printf("\nK=%d N=%d: contract (1 / 5 / 30 / 64 / 256 rows == the 4096-row call's) %s; vendor vs double %.1e of sum "
               "|terms|\n", K, N, shape_contract ? "holds" : "*** FAILS ***", worst_sanity);
        if (ship_ok)
            printf("  VERDICT: REPRODUCED byte for byte by the shipped order (%s) on all %d datasets\n", shipped.c_str(),
                   trials + 1);
        else
            printf("  VERDICT: *** NOT REPRODUCED by the shipped order (%s): %.4f %% of the compared outputs equal ***\n",
                   shipped.c_str(), 100.0 * eq_sum[0] / (trials + 1));
        if (!ship_ok || survey_all) {
            printf("  one-hot scan, vendor (p = %d = tile %d + 5; q = p - %d .. p + %d; '2' the two small products met "
                   "first, '0' not, 'x' other)%s:\n", p, p / 16, kOneHotHalf, kOneHotHalf - 1,
                   onehot_cols_differ ? "  *** COLUMNS DIFFER: the result depends on the output's position ***" : "");
            for (int di = 0; di < 5; ++di) printf("    d=%-2d %s\n", dq[di], onehot_sig[di].c_str());
            std::string winners;
            int n_win = 0;
            for (size_t ci = 0; ci < cands.size(); ++ci)
                if (exact[ci]) winners += (n_win++ ? ", " : "") + cand_name(cands[ci]);
            if (n_win) {
                printf("  survey: reproduced by %d of %zu orders: %s\n", n_win, cands.size(), winners.c_str());
            } else {
                std::vector<size_t> ord(cands.size());
                for (size_t i = 0; i < ord.size(); ++i) ord[i] = i;
                std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return eq_sum[a] > eq_sum[b]; });
                printf("  survey: NO order of the family reproduces it; closest:");
                for (int i = 0; i < 4; ++i)
                    printf(" %s (%.2f %% of outputs equal)%s", cand_name(cands[ord[i]]).c_str(),
                           100.0 * eq_sum[ord[i]] / (trials + 1), i < 3 ? "," : "\n");
            }
        }
        if (!ship_ok || bug) all_ok = false;
        if (!shape_contract) contract_ok = false;
        fflush(stdout);
        cudaFree(dX); cudaFree(dW); cudaFree(dXc); cudaFree(dWc); cudaFree(dD); cudaFree(dD2); cudaFree(dDc);
    }
    printf("\nSUMMARY: %zu shapes on sm_%d%d, cuBLASLt %zu: the row-count contract %s; the shipped order (%s) reproduces "
           "%d of them%s\n", shapes.size(), ma, mi, ltv, contract_ok ? "holds on all" : "*** FAILS ***", shipped.c_str(),
           n_shipped,
           all_ok && contract_ok ? ": the in-place prompt kernel's premise holds here"
           : named ? " -- *** THE IN-PLACE PROMPT KERNEL'S PREMISE DOES NOT HOLD HERE: a serving process finds it in the "
                     "arm's check and keeps the GEMM; exl3-dense-gate fails ***"
                   : " -- this architecture stays off ds4_exl3_dense_prompt_arm");
    return all_ok && contract_ok ? 0 : 3;
}

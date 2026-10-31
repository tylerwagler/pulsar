/** @file
 *  Kernel contract gate (L335) -- model-free, links the PRODUCTION objects.
 *
 *  One case per L335 CUDA fix: each pins the contract the fix restored (an exact helper, a refusal, a bound, a
 *  NaN rule), so the fix cannot quietly come undone.  Built with the production NVCCFLAGS (--use_fast_math), which is
 *  the configuration the fixes were made for.
 *
 *    1. mx-shared-exp   pulsar_mx_shared_exp == the exact floor(log2(amax)) - 7 (double, on the host) on every binade
 *                       edge of the normal f32 range; prints how many the old fast-math floorf(log2f()) got wrong
 *                       (the expression swiglu_pack_e4m3_warp_kernel carried until L335).
 *    2. embed-sentinel  pulsar_gpu_embed_tokens_hc_tensor: an id in [0, n_vocab) copies its row into every carrier; an
 *                       image-slot sentinel id (>= n_vocab) writes a ZERO row (it used to read row 0 -- a real token's
 *                       embedding -- as a silent clamp).
 *    3. indexer-topk    pulsar_gpu_indexer_topk_tensor, top-512 at n_comp 1500 / 3000 / 6000 / 8192 (the widths that
 *                       took the shared-memory bitonic kernels when the cub sort's smem was refused): the one cub arm
 *                       against the host order, selection for selection.
 *    4. hc-expand-nhc   pulsar_gpu_hc_expand_split_tensor runs n_hc 4 and refuses 3 and 5 (a runtime-loop arm used to
 *                       take any other count).
 *
 *  Prints one line per case and KERNEL CONTRACT GATE PASS / FAIL. */
#include "pulsar_gpu.h"
#include "cuda/pulsar_cuda_mx.cuh"

#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>

/// Failures so far (every case adds to it; main returns it as the verdict).
static int g_fail = 0;

/// Records one failed check and says which.
/// @param what the case and the check that failed
static void fail(const char *what) {
    fprintf(stderr, "KERNEL CONTRACT FAIL: %s\n", what);
    g_fail++;
}

/// True when a CUDA call returned cudaSuccess; otherwise records the failure by name.
/// @param e    the call's return code
/// @param what the call
/// @return e == cudaSuccess
static bool ok(cudaError_t e, const char *what) {
    if (e == cudaSuccess) return true;
    fprintf(stderr, "KERNEL CONTRACT FAIL: %s: %s\n", what, cudaGetErrorString(e));
    g_fail++;
    return false;
}

/* ---- 1. mx-shared-exp ------------------------------------------------------------------------------------------ */

/// The exponent the MXFP4 mid packer computed before L335 (kept HERE only, as the measured contrast).
/// @param mx the block amax
/// @return floorf(log2f(mx)) - 7, clamped to [-127, 127]; -127 for mx 0
__device__ static int old_fast_math_exp(float mx) {
    int se = -127;
    if (mx > 0.f) se = (int)floorf(log2f(mx)) - 7;
    if (se < -127) se = -127;
    if (se > 127) se = 127;
    return se;
}

/// One thread per value: the production exponent and the old one.
/// @param v    the amax values
/// @param n    how many
/// @param se   out: pulsar_mx_shared_exp(v[i])
/// @param old  out: old_fast_math_exp(v[i])
__global__ static void mx_exp_kernel(const float *v, int n, int *se, int *old) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    se[i] = pulsar_mx_shared_exp(v[i]);
    old[i] = old_fast_math_exp(v[i]);
}

/// The exact shared exponent, in double: floor(log2(a)) - 7 clamped to [-127, 127], -127 for 0.
/// @param a the amax
/// @return the exponent
static int exact_exp(float a) {
    if (!(a > 0.f)) return -127;
    int e = 0;
    frexp((double)a, &e);
    int se = (e - 1) - 7;
    if (se < -127) se = -127;
    if (se > 127) se = 127;
    return se;
}

/// Case 1: every binade edge 2^e (e = -126..127) -- the float just below, the power itself, the float just above.
static void case_mx_shared_exp(void) {
    std::vector<float> v;
    for (int e = -126; e <= 127; e++) {
        const float p = ldexpf(1.0f, e);
        v.push_back(nextafterf(p, 0.0f));
        v.push_back(p);
        v.push_back(nextafterf(p, INFINITY));
    }
    v.push_back(0.0f);
    const int n = (int)v.size();
    float *dv = nullptr;
    int *dse = nullptr, *dold = nullptr;
    if (!ok(cudaMalloc(&dv, n * sizeof(float)), "mx-shared-exp malloc") ||
        !ok(cudaMalloc(&dse, n * sizeof(int)), "mx-shared-exp malloc") ||
        !ok(cudaMalloc(&dold, n * sizeof(int)), "mx-shared-exp malloc")) return;
    std::vector<int> se(n), old(n);
    if (ok(cudaMemcpy(dv, v.data(), n * sizeof(float), cudaMemcpyHostToDevice), "mx-shared-exp upload")) {
        mx_exp_kernel<<<(n + 127) / 128, 128>>>(dv, n, dse, dold);
        if (ok(cudaDeviceSynchronize(), "mx-shared-exp kernel") &&
            ok(cudaMemcpy(se.data(), dse, n * sizeof(int), cudaMemcpyDeviceToHost), "mx-shared-exp download") &&
            ok(cudaMemcpy(old.data(), dold, n * sizeof(int), cudaMemcpyDeviceToHost), "mx-shared-exp download")) {
            int bad = 0, old_bad = 0;
            for (int i = 0; i < n; i++) {
                const int want = exact_exp(v[i]);
                if (se[i] != want) {
                    if (bad < 4) fprintf(stderr, "  mx-shared-exp: amax %a -> %d, exact %d\n", v[i], se[i], want);
                    bad++;
                }
                if (old[i] != want) old_bad++;
            }
            printf("  1 mx-shared-exp: %d binade-edge values, production %d off, old fast-math floorf(log2f) %d off\n",
                   n, bad, old_bad);
            if (bad) fail("mx-shared-exp: pulsar_mx_shared_exp differs from the exact exponent");
        }
    }
    cudaFree(dv);
    cudaFree(dse);
    cudaFree(dold);
}

/* ---- 2. embed-sentinel ----------------------------------------------------------------------------------------- */

/// Case 2: a small synthetic table (a host mapping with no fd: the runtime copies it to the device).
static void case_embed_sentinel(void) {
    const uint32_t n_vocab = 5u, n_embd = 64u, n_hc = 4u;
    static uint16_t table[5 * 64];
    for (uint32_t r = 0; r < n_vocab; r++)
        for (uint32_t d = 0; d < n_embd; d++) table[r * n_embd + d] = (uint16_t)(0x3f80u + r * 64u + d);   /* bf16 */
    const int32_t ids[4] = {3, 0, (int32_t)n_vocab, 4};   /* ids[2]: an image slot's sentinel (vocab + role 0) */
    const uint32_t n_tok = 4u;
    pulsar_gpu_tensor *tok = pulsar_gpu_tensor_alloc(sizeof(ids));
    pulsar_gpu_tensor *out = pulsar_gpu_tensor_alloc((uint64_t)n_tok * n_hc * n_embd * 2u);
    if (!tok || !out || !pulsar_gpu_tensor_write(tok, 0, ids, sizeof(ids))) {
        fail("embed-sentinel: alloc / upload");
    } else if (!pulsar_gpu_embed_tokens_hc_tensor(out, tok, table, sizeof(table), 0, n_vocab, n_tok, n_embd, n_hc) ||
               !pulsar_gpu_synchronize()) {
        fail("embed-sentinel: launch");
    } else {
        std::vector<uint16_t> got((size_t)n_tok * n_hc * n_embd);
        if (!pulsar_gpu_tensor_read(out, 0, got.data(), got.size() * 2u)) {
            fail("embed-sentinel: read back");
        } else {
            int bad = 0;
            for (uint32_t t = 0; t < n_tok; t++)
                for (uint32_t h = 0; h < n_hc; h++)
                    for (uint32_t d = 0; d < n_embd; d++) {
                        const uint16_t want = (uint32_t)ids[t] < n_vocab ? table[(uint32_t)ids[t] * n_embd + d] : 0u;
                        bad += got[((size_t)t * n_hc + h) * n_embd + d] != want;
                    }
            printf("  2 embed-sentinel: %u tokens x %u carriers, %d elements off (the sentinel row must be zero)\n",
                   n_tok, n_hc, bad);
            if (bad) fail("embed-sentinel: a row is not its token's embedding, or the sentinel row is not zero");
        }
    }
    pulsar_gpu_tensor_free(tok);
    pulsar_gpu_tensor_free(out);
}

/* ---- 3. indexer-topk ------------------------------------------------------------------------------------------- */

/// Case 3: top-512 of rows in each old width bucket (1025..2048, ..4096, ..8192) through the one cub arm, against the
/// host order (score descending, index ascending on a tie).
static void case_indexer_topk(void) {
    static const uint32_t widths[] = {1500u, 3000u, 6000u, 8192u};
    const uint32_t top_k = 512u, n_tok = 3u;
    uint64_t s = 0x2545f4914f6cdd1dull;
    for (uint32_t n_comp : widths) {
        std::vector<float> sc((size_t)n_tok * n_comp);
        for (float &v : sc) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            v = (float)((int)((s >> 33) % 2001u) - 1000) * 0.125f;   /* coarse: plenty of exact ties */
        }
        pulsar_gpu_tensor *dsc = pulsar_gpu_tensor_alloc(sc.size() * sizeof(float));
        pulsar_gpu_tensor *dsel = pulsar_gpu_tensor_alloc((uint64_t)n_tok * top_k * sizeof(uint32_t));
        std::vector<uint32_t> got((size_t)n_tok * top_k);
        if (!dsc || !dsel || !pulsar_gpu_tensor_write(dsc, 0, sc.data(), sc.size() * sizeof(float)) ||
            !pulsar_gpu_indexer_topk_tensor(dsel, dsc, n_comp, n_tok, top_k) || !pulsar_gpu_synchronize() ||
            !pulsar_gpu_tensor_read(dsel, 0, got.data(), got.size() * sizeof(uint32_t))) {
            fail("indexer-topk: alloc / launch / read back");
        } else {
            int bad = 0;
            for (uint32_t t = 0; t < n_tok; t++) {
                const float *row = &sc[(size_t)t * n_comp];
                std::vector<uint32_t> ord(n_comp);
                for (uint32_t i = 0; i < n_comp; i++) ord[i] = i;
                std::stable_sort(ord.begin(), ord.end(), [row](uint32_t a, uint32_t b) { return row[a] > row[b]; });
                for (uint32_t i = 0; i < top_k; i++) bad += got[(size_t)t * top_k + i] != ord[i];
            }
            printf("  3 indexer-topk: n_comp %u, %u rows, %d selections off\n", n_comp, n_tok, bad);
            if (bad) fail("indexer-topk: a selection differs from the host order");
        }
        pulsar_gpu_tensor_free(dsc);
        pulsar_gpu_tensor_free(dsel);
    }
}

/* ---- 4. hc-expand-nhc ------------------------------------------------------------------------------------------ */

/// Case 4: hc_expand runs 4 carriers and refuses any other count (3 and 5 here).
static void case_hc_expand_nhc(void) {
    const uint32_t n_embd = 4096u, n_tok = 2u;
    int bad = 0;
    for (uint32_t n_hc : {3u, 4u, 5u}) {
        const uint32_t mix = 2u * n_hc + n_hc * n_hc;
        pulsar_gpu_tensor *out = pulsar_gpu_tensor_alloc_elt((uint64_t)n_tok * n_hc * n_embd, 2u, PULSAR_HC_ELT_FMT);
        pulsar_gpu_tensor *res = pulsar_gpu_tensor_alloc_elt((uint64_t)n_tok * n_hc * n_embd, 2u, PULSAR_HC_ELT_FMT);
        pulsar_gpu_tensor *blk = pulsar_gpu_tensor_alloc((uint64_t)n_tok * n_embd * 4u);
        pulsar_gpu_tensor *split = pulsar_gpu_tensor_alloc((uint64_t)n_tok * mix * 4u);
        if (!out || !res || !blk || !split) {
            fail("hc-expand-nhc: alloc");
        } else {
            const std::vector<uint8_t> z((size_t)n_tok * n_hc * n_embd * 4u, 0u);
            pulsar_gpu_tensor_write(res, 0, z.data(), (uint64_t)n_tok * n_hc * n_embd * 2u);
            pulsar_gpu_tensor_write(blk, 0, z.data(), (uint64_t)n_tok * n_embd * 4u);
            pulsar_gpu_tensor_write(split, 0, z.data(), (uint64_t)n_tok * mix * 4u);
            const bool ran = pulsar_gpu_hc_expand_split_tensor(out, blk, res, split, n_embd, n_hc) != 0 &&
                             pulsar_gpu_synchronize() != 0;
            if (ran != (n_hc == 4u)) {
                fprintf(stderr, "  hc-expand-nhc: n_hc %u %s\n", n_hc, ran ? "ran" : "refused");
                bad++;
            }
        }
        pulsar_gpu_tensor_free(out);
        pulsar_gpu_tensor_free(res);
        pulsar_gpu_tensor_free(blk);
        pulsar_gpu_tensor_free(split);
    }
    printf("  4 hc-expand-nhc: n_hc 4 runs, 3 and 5 refuse: %s\n", bad ? "NO" : "yes");
    if (bad) fail("hc-expand-nhc: a carrier count other than 4 ran, or 4 refused");
}

/// Runs every case on device 0.
/// @return 0 on PASS, 1 on FAIL
int main(void) {
    if (!pulsar_gpu_init()) {
        fprintf(stderr, "KERNEL CONTRACT FAIL: no GPU\n");
        return 1;
    }
    case_mx_shared_exp();
    case_embed_sentinel();
    case_indexer_topk();
    case_hc_expand_nhc();
    puts(g_fail ? "KERNEL CONTRACT GATE FAIL" : "KERNEL CONTRACT GATE PASS");
    return g_fail ? 1 : 0;
}
